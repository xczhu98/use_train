"""主进程管理的 spawn gpuRIR 服务；Dataset worker 仅用带超时的本机 IPC。

采用 JSON + float32 bytes 协议，不反序列化来自 socket 的 pickle。socket 在
权限 0700 的临时目录内，token 仅在进程内传递。队列与 cache 均有上限。
"""
from __future__ import annotations
import atexit
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import secrets
import shutil
import socket
import struct
import tempfile
import threading
import time
import numpy as np

from core.data.simulation.gpurir_backend import validate_rir_plan_budget

_MAX_FRAME = 64 * 1024 * 1024
_DEFAULT_CACHE_BYTES = 32 * 1024 * 1024
_MAX_TRACK_BYTES = 256 * 1024 * 1024
_SERVICE_RESPONSE_RESERVE_SEC = 0.25
_REQUEST_KINDS = frozenset(
    {"full", "full_direct", "moving_full", "moving_full_direct"}
)
_SCENE_REQUEST_KINDS = frozenset({"scene_static", "scene_moving"})
_MOVING_TRACK_KINDS = frozenset(
    {"moving_full", "moving_full_direct", "scene_moving_full", "scene_moving_full_direct"}
)
# 常驻生成 worker 数：每个 worker 只付一次解释器/框架/gpuRIR/CUDA 启动成本。
# 单次生成约 20 ms，2 个 worker 已远高于 dataloader 的请求速率；可用
# USE_TRAIN_GPURIR_WORKERS 覆盖（1.._MAX_GENERATION_WORKERS）。
_DEFAULT_GENERATION_WORKERS = 2
_MAX_GENERATION_WORKERS = 8
_ACCEPT_POLL_SEC = 0.25
_WARMUP_TIMEOUT_SEC = 60.0
_OWNERS = {}
_LOCK = threading.Lock()


def _resolve_cache_bytes():
    """解析在线 RIR cache 的字节预算；零值表示禁用 waveform cache。"""
    raw = os.environ.get('USE_TRAIN_GPURIR_CACHE_BYTES')
    if raw is None or not raw.strip():
        return _DEFAULT_CACHE_BYTES
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError('USE_TRAIN_GPURIR_CACHE_BYTES must be an integer') from exc
    if value < 0:
        raise ValueError('USE_TRAIN_GPURIR_CACHE_BYTES must be non-negative')
    return value


_CACHE_BYTES = _resolve_cache_bytes()


def _warmup_plan():
    """最小合法 plan：只用于让常驻 worker 付掉 import 与 CUDA 初始化成本。"""
    return {
        'room_dim': [3.0, 3.0, 2.5],
        'source_loc': [[1.0, 1.0, 1.2]],
        'mic_loc': [1.5, 1.5, 1.2],
        'num_samples': 128,
        'sample_rate': 8000,
        'rt60': 0.2,
        'diffuse_db': 15.0,
        'sound_speed': 340.0,
        'group_uid': 'gpurir-service-warmup',
    }


def _json_safe(value):
    """将 backend facts 中的 numpy 标量/数组规范化为 canonical JSON 值。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value

class OnlineRIRError(RuntimeError):
    """在线 backend/IPC 失败必须传播，不能静默改成离线或静音。"""


class RIRServiceMetrics:
    """受管理 RIR service 的进程内累计指标；快照通过每次响应返回。"""

    def __init__(self, device=None):
        self.device = device
        self.started_at = time.time()
        self._lock = threading.Lock()
        self.request_count = 0
        self.cache_hit_count = 0
        self.generation_count = 0
        self.generation_latency_ms_total = 0.0
        self.generation_latency_ms_last = 0.0
        self.queue_wait_ms_total = 0.0
        self.queue_wait_ms_last = 0.0
        self.bytes_in_total = 0
        self.bytes_out_total = 0
        self.failure_count = 0
        self.timeout_count = 0
        self.request_kind_counts: dict[str, int] = {}
        self.track_kind_counts: dict[str, int] = {}

    def record(
        self,
        *,
        request_kind: str,
        track_kind: str,
        cache_hit: bool,
        queue_wait_ms: float,
        generation_latency_ms: float,
        bytes_in: int,
        bytes_out: int,
        failed: bool = False,
        timed_out: bool = False,
    ) -> dict:
        with self._lock:
            self.request_count += 1
            self.cache_hit_count += int(bool(cache_hit))
            if not cache_hit and not failed:
                self.generation_count += 1
            self.generation_latency_ms_total += float(generation_latency_ms)
            self.generation_latency_ms_last = float(generation_latency_ms)
            self.queue_wait_ms_total += float(queue_wait_ms)
            self.queue_wait_ms_last = float(queue_wait_ms)
            self.bytes_in_total += max(0, int(bytes_in))
            self.bytes_out_total += max(0, int(bytes_out))
            self.failure_count += int(bool(failed))
            self.timeout_count += int(bool(timed_out))
            self.request_kind_counts[request_kind] = self.request_kind_counts.get(request_kind, 0) + 1
            self.track_kind_counts[track_kind] = self.track_kind_counts.get(track_kind, 0) + 1
            return self._snapshot_unlocked()

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_unlocked()

    def mark_failure(self, *, timed_out: bool = False) -> None:
        with self._lock:
            self.failure_count += 1
            self.timeout_count += int(bool(timed_out))

    def _snapshot_unlocked(self) -> dict:
        """调用方已持有 lock 时构造快照，避免重复获取非重入锁。"""
        count = self.request_count
        return {
            "rir_request_count": count,
            "rir_cache_hit_count": self.cache_hit_count,
            "rir_cache_hit_rate": (self.cache_hit_count / count) if count else 0.0,
            "rir_generation_count": self.generation_count,
            "rir_generation_latency_ms": self.generation_latency_ms_last,
            "rir_generation_latency_ms_total": self.generation_latency_ms_total,
            "rir_queue_wait_ms": self.queue_wait_ms_last,
            "rir_queue_wait_ms_total": self.queue_wait_ms_total,
            "rir_bytes_transferred": self.bytes_in_total + self.bytes_out_total,
            "rir_bytes_in_total": self.bytes_in_total,
            "rir_bytes_out_total": self.bytes_out_total,
            "rir_failure_count": self.failure_count,
            "rir_timeout_count": self.timeout_count,
            "rir_service_gpu_id": self.device,
            "rir_service_pid": os.getpid(),
            "rir_service_started_at": self.started_at,
            "rir_request_kind_counts": dict(self.request_kind_counts),
            "rir_track_kind_counts": dict(self.track_kind_counts),
        }


def _canonical_scene_request(request_kind, scene, *, normalize=True):
    """规范化 scene-level wire request，并拒绝隐式 legacy 字段。

    scene_static/scene_moving 使用 actor identity 作为 source axis 的唯一事实来源；
    server 只把它转换为内部 gpuRIR plan，不允许再次随机 room 或位置。
    """
    if request_kind not in _SCENE_REQUEST_KINDS:
        raise OnlineRIRError(f"unsupported scene request kind: {request_kind!r}")
    if not isinstance(scene, dict):
        raise OnlineRIRError("scene request must be an object")
    if type(normalize) is not bool:
        raise OnlineRIRError("scene normalization flag must be a bool")
    # 冻结 wire 内容；tuple/list 统一，同时拒绝 NaN 和非 JSON 对象。
    scene = json.loads(json.dumps(scene, allow_nan=False))
    required = ("scene_uid", "sample_rate", "num_samples", "room", "array", "actors")
    if any(key not in scene for key in required):
        raise OnlineRIRError("scene request requires scene_uid, sample_rate, num_samples, room, array, actors")
    if not isinstance(scene["scene_uid"], str) or not scene["scene_uid"].strip():
        raise OnlineRIRError("scene_uid must be non-empty")
    if type(scene["sample_rate"]) is not int or scene["sample_rate"] < 1:
        raise OnlineRIRError("scene sample_rate must be a positive integer")
    if type(scene["num_samples"]) is not int or scene["num_samples"] < 1:
        raise OnlineRIRError("scene num_samples must be a positive integer")
    room = scene["room"]
    array = scene["array"]
    actors = scene["actors"]
    if not isinstance(room, dict) or not isinstance(array, dict) or not isinstance(actors, list) or not actors:
        raise OnlineRIRError("scene room, array and actors have invalid types")
    room_dim = np.asarray(room.get("room_dim_m", room.get("room_dim")), dtype=np.float64)
    if room_dim.shape != (3,) or not np.isfinite(room_dim).all() or np.any(room_dim <= 0):
        raise OnlineRIRError("scene room dimensions must be positive finite [3]")
    for key, value in (("rt60", room.get("rt60")), ("sound_speed", room.get("sound_speed", 343.0))):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise OnlineRIRError(f"scene room {key} must be positive and finite")
    rir_num_samples = room.get("rir_num_samples")
    if type(rir_num_samples) is not int or rir_num_samples < 1:
        raise OnlineRIRError("scene room requires positive rir_num_samples")
    diffuse_db = room.get("diffuse_db")
    if not isinstance(diffuse_db, (int, float)) or isinstance(diffuse_db, bool) or not np.isfinite(diffuse_db):
        raise OnlineRIRError("scene room requires finite diffuse_db")
    positions = array.get("positions_m", array.get("mic_loc"))
    mic = np.asarray(positions, dtype=np.float32)
    if mic.ndim != 2 or mic.shape[1:] != (3,) or mic.shape[0] < 1:
        raise OnlineRIRError("scene array positions must have shape [C,3]")
    if not np.isfinite(mic).all() or np.any(mic <= 0) or np.any(mic >= room_dim):
        raise OnlineRIRError("scene microphones must be inside the room in absolute coordinates")
    reference = array.get("reference_channel", 0)
    if type(reference) is not int or not 0 <= reference < len(mic):
        raise OnlineRIRError("scene reference_channel must be in [0,C)")
    clock = None
    if request_kind == "scene_moving":
        from core.data.simulation.trajectory import TrajectoryTimebase
        timebase = scene.get("timebase")
        if not isinstance(timebase, dict):
            raise OnlineRIRError("moving scene requires timebase")
        fields = ("sample_rate", "num_samples", "hop_samples", "frame_count", "start_sample")
        clock = TrajectoryTimebase(**{key: timebase[key] for key in fields if key in timebase})
        if clock.sample_rate != scene["sample_rate"] or clock.num_samples != scene["num_samples"] or clock.start_sample != 0:
            raise OnlineRIRError("moving scene timebase must match waveform sample rate and length")
        if not 2 <= clock.frame_count <= 128:
            raise OnlineRIRError("moving scene requires 2..128 trajectory frames")
    actor_order = []
    source_loc = []
    trajectories = {}
    for actor in actors:
        if not isinstance(actor, dict) or not isinstance(actor.get("actor_uid"), str) or not actor["actor_uid"]:
            raise OnlineRIRError("scene actors require non-empty actor_uid")
        uid = actor["actor_uid"]
        if uid in actor_order:
            raise OnlineRIRError("scene actor_order contains duplicates")
        actor_order.append(uid)
        position = actor.get("position_m")
        trajectory = actor.get("trajectory")
        if request_kind == "scene_static":
            if position is None or trajectory is not None:
                raise OnlineRIRError("static scene actors require position_m and forbid trajectory")
            points = np.asarray(position, dtype=np.float64)
            if points.shape != (3,) or not np.isfinite(points).all() or np.any(points <= 0) or np.any(points >= room_dim):
                raise OnlineRIRError("static actor position must be inside room")
            source_loc.append(position)
        else:
            if trajectory is None:
                raise OnlineRIRError("moving scene actors require trajectory")
            points = np.asarray(trajectory, dtype=np.float64)
            if points.shape != (clock.frame_count, 1, 3) or not np.isfinite(points).all():
                raise OnlineRIRError("actor trajectory must have finite [P,1,3] matching timebase")
            if np.any(points <= 0) or np.any(points >= room_dim):
                raise OnlineRIRError("actor trajectory must stay inside room")
            if position is not None and not np.allclose(position, points[0, 0], rtol=0, atol=1e-6):
                raise OnlineRIRError("actor position_m must match trajectory first frame")
            trajectories[uid] = trajectory
            # canonical plan 必须可直接 JSON 序列化；source_loc 是每个 actor
            # 的首帧绝对坐标，不保留 numpy view。
            source_loc.append(np.asarray(trajectory, dtype=np.float32)[0, 0].tolist())
    plan = {
        "version": 3,
        "backend": "online",
        "group_uid": scene["scene_uid"],
        "scene_uid": scene["scene_uid"],
        "sample_rate": scene["sample_rate"],
        "num_samples": int(rir_num_samples),
        "waveform_num_samples": scene["num_samples"],
        "room_dim": room.get("room_dim_m", room.get("room_dim")),
        "rt60": room.get("rt60"),
        "diffuse_db": float(diffuse_db),
        "sound_speed": room.get("sound_speed", 343.0),
        "mic_loc": mic.tolist(),
        "source_loc": source_loc,
        "source_ids": actor_order,
        "actor_roles": [actor.get("role") for actor in actors],
        "reference_channel": array.get("reference_channel", 0),
        "array": array,
        "room": room,
        "num_channels": int(mic.shape[0]),
    }
    if plan["room_dim"] is None or len(plan["room_dim"]) != 3:
        raise OnlineRIRError("scene room requires room_dim_m with length 3")
    if request_kind == "scene_moving":
        timebase = scene.get("timebase")
        if not isinstance(timebase, dict):
            raise OnlineRIRError("moving scene requires timebase")
        plan["actor_trajectories"] = trajectories
        plan["trajectory_timebase"] = timebase
    # scene.num_samples 是 waveform T；plan.num_samples 是 RIR 长度 L。
    # 两者必须在 wire contract 中同时保留，避免用 T 预算/reshape RIR。
    return {"request_kind": request_kind, "scene": scene, "plan": plan, "normalize": bool(normalize)}


def _validate_scene_track_facts(facts, plan, name, *, moving):
    """严格校验单条 scene track 的 metadata；归一化由 provider 统一负责。"""
    if not isinstance(facts, dict):
        raise OnlineRIRError(f"{name} scene facts must be an object")
    if facts.get("scene_uid") != plan.get("scene_uid"):
        raise OnlineRIRError(f"{name} scene_uid mismatch")
    if facts.get("actor_order") != plan.get("source_ids"):
        raise OnlineRIRError(f"{name} actor_order mismatch")
    if facts.get("track_kind") != name:
        raise OnlineRIRError(f"{name} track_kind mismatch")
    if facts.get("generator_version") is None:
        raise OnlineRIRError(f"{name} generator_version is missing")
    expected = {
        "room_dim": plan.get("room_dim"),
        "mic_loc": plan.get("mic_loc"),
        "source_loc": plan.get("source_loc"),
        "sample_rate": plan.get("sample_rate"),
    }
    for key, value in expected.items():
        actual = facts.get(key)
        if key == "sample_rate":
            if type(actual) is not int or actual != value:
                raise OnlineRIRError(f"{name} {key} mismatch")
        elif not np.array_equal(np.asarray(actual), np.asarray(value)):
            raise OnlineRIRError(f"{name} {key} mismatch")
    if moving:
        if facts.get("actor_trajectories") != plan.get("actor_trajectories"):
            raise OnlineRIRError(f"{name} actor_trajectories mismatch")
        if facts.get("trajectory_timebase") != plan.get("trajectory_timebase"):
            raise OnlineRIRError(f"{name} trajectory_timebase mismatch")
    digest = facts.get("rir_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise OnlineRIRError(f"{name} invalid rir_sha256")

def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('RIR request timed out')
    return remaining


def _send(sock, payload, deadline=None):
    if len(payload) > _MAX_FRAME:
        raise OnlineRIRError('RIR IPC frame exceeds memory budget')
    if deadline is not None:
        sock.settimeout(_remaining(deadline))
    sock.sendall(struct.pack('!Q', len(payload)) + payload)

def _receive(sock, deadline, *, max_bytes=None):

    def read(size):
        chunks = []
        while size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('RIR request timed out')
            sock.settimeout(remaining)
            data = sock.recv(min(size, 65536))
            if not data:
                raise EOFError('RIR service disconnected')
            chunks.append(data)
            size -= len(data)
        return b''.join(chunks)
    size = struct.unpack('!Q', read(8))[0]
    if size > _MAX_FRAME or (max_bytes is not None and size > max_bytes):
        raise OnlineRIRError('RIR IPC frame exceeds memory budget')
    return read(size)

def _request_shape(plan, track_kind):
    """在 CUDA、socket 和缓存访问前固化输出 shape 与完整双轨预算。"""
    if track_kind in {'scene_moving_full', 'scene_moving_full_direct'}:
        trajectories = plan.get('actor_trajectories')
        if not isinstance(trajectories, dict) or not trajectories:
            raise OnlineRIRError('scene moving plan requires actor_trajectories')
        counts = {np.asarray(value).shape[0] for value in trajectories.values()}
        if len(counts) != 1:
            raise OnlineRIRError('scene moving actor trajectories must share frame count')
        frames = next(iter(counts))
        mic = np.asarray(plan['mic_loc'])
        if mic.ndim != 2 or mic.shape[1:] != (3,) or frames < 1:
            raise OnlineRIRError('scene moving plan has invalid trajectory or mic geometry')
        shape = (frames, len(trajectories), len(mic), plan['num_samples'])
        frame_bytes = int(np.prod(shape[1:], dtype=object)) * 4
        track_count = 2 if track_kind == 'scene_moving_full_direct' else 1
        if frame_bytes > _MAX_FRAME or track_count * frames * frame_bytes > _MAX_TRACK_BYTES:
            raise OnlineRIRError('scene moving RIR waveform exceeds memory budget')
        return shape
    if track_kind == 'scene_static_full_direct':
        if 'trajectory' in plan or 'trajectory_timebase' in plan:
            raise OnlineRIRError('static scene track_kind cannot carry trajectory fields')
        mic = np.asarray(plan['mic_loc'])
        if mic.ndim != 2 or mic.shape[1:] != (3,) or mic.shape[0] < 1:
            raise OnlineRIRError('scene static plan has invalid mic geometry')
        validate_rir_plan_budget(plan, max_bytes=_MAX_TRACK_BYTES, track_count=2)
        return (len(plan['source_loc']), len(mic), plan['num_samples'])
    if track_kind not in _REQUEST_KINDS:
        raise OnlineRIRError(f'unsupported RIR track_kind: {track_kind!r}')
    if not isinstance(plan, dict) or not isinstance(plan.get('group_uid'), str) or not plan['group_uid']:
        raise OnlineRIRError('RIR plan requires a non-empty group_uid')
    if track_kind in {'moving_full', 'moving_full_direct'}:
        from core.data.simulation.gpurir_backend import validate_moving_rir_plan
        track_count = 2 if track_kind == 'moving_full_direct' else 1
        trajectory = validate_moving_rir_plan(
            plan, max_bytes=_MAX_TRACK_BYTES, track_count=track_count
        )
        mic = np.asarray(plan['mic_loc'])
        legacy_mono = mic.shape == (3,)
        if not legacy_mono and (mic.ndim != 2 or mic.shape[1:] != (3,) or mic.shape[0] < 1):
            raise OnlineRIRError('moving RIR mic_loc must have shape [C,3] or legacy mono [3]')
        # legacy mono plan 的 bank 布局为 [P,K,L]；阵列 plan 为 [P,K,C,L]。
        shape = (
            (len(trajectory), trajectory.shape[1], plan['num_samples'])
            if legacy_mono else
            (len(trajectory), trajectory.shape[1], len(mic), plan['num_samples'])
        )
        frame_bytes = int(np.prod(shape[1:], dtype=object)) * 4
        if frame_bytes > _MAX_FRAME or track_count * shape[0] * frame_bytes > _MAX_TRACK_BYTES:
            raise OnlineRIRError('moving RIR waveform exceeds memory budget')
        return shape
    if 'trajectory' in plan or 'trajectory_timebase' in plan:
        raise OnlineRIRError('trajectory requires a moving track_kind')
    track_count = 2 if track_kind == 'full_direct' else 1
    validate_rir_plan_budget(
        plan, max_bytes=_MAX_TRACK_BYTES, track_count=track_count
    )
    mic = np.asarray(plan['mic_loc'])
    return ((len(plan['source_loc']), plan['num_samples']) if mic.shape == (3,)
            else (len(plan['source_loc']), len(mic), plan['num_samples']))


def _validate_track_facts(facts, plan, name, *, moving):
    if not isinstance(facts, dict):
        raise OnlineRIRError(f'{name} RIR facts must be an object')
    if facts.get('group_uid') != plan['group_uid']:
        raise OnlineRIRError(f'{name} RIR group identity mismatch')
    if facts.get('track_kind') != name:
        raise OnlineRIRError(f'{name} RIR track_kind mismatch')
    digest = facts.get('rir_sha256')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise OnlineRIRError(f'{name} RIR invalid content hash')
    if moving:
        # JSON 严格比较避免 bool 与 int 在 Python 相等比较中混淆。
        expected = json.dumps(plan['trajectory_timebase'], sort_keys=True, allow_nan=False)
        actual = json.dumps(facts.get('trajectory_timebase'), sort_keys=True, allow_nan=False)
        if actual != expected:
            raise OnlineRIRError(f'{name} RIR trajectory_timebase mismatch')
        trajectory = np.asarray(facts.get('trajectory'), dtype=np.float32)
        if not np.array_equal(trajectory, np.asarray(plan['trajectory'], dtype=np.float32)):
            raise OnlineRIRError(f'{name} RIR trajectory mismatch')


def _prepare_tracks(generated, plan, track_kind, expected, hit):
    """发布前验证整组结果，禁止先发送 full 再发现 direct 无效。"""
    names = ('full',) if track_kind in {'full', 'moving_full', 'scene_moving_full'} else ('full', 'direct')
    if not isinstance(generated, dict) or set(generated) != set(names):
        raise OnlineRIRError('RIR response track set mismatch')
    moving = track_kind in _MOVING_TRACK_KINDS
    result, infos = {}, {}
    for name in names:
        wave, facts = generated[name]
        wave = np.ascontiguousarray(wave, dtype='<f4')
        if wave.shape != expected or not np.isfinite(wave).all():
            raise OnlineRIRError(f'{name} RIR response shape/value mismatch')
        facts = dict(facts)
        if track_kind == 'full':
            if facts.get('group_uid') != plan['group_uid']:
                raise OnlineRIRError('full RIR group identity mismatch')
        elif track_kind in {'scene_moving_full', 'scene_moving_full_direct'}:
            _validate_scene_track_facts(facts, plan, name, moving=True)
        else:
            _validate_track_facts(facts, plan, name, moving=moving)
        # 缓存条目在生成时已完成内容身份校验；命中时直接复用其 digest，避免
        # 对同一 waveform 每请求重算整轨 SHA-256（P3 热路径开销）。digest 缺失
        # 或形态非法时仍回退到实际计算并 fail-closed。
        digest = facts.get('rir_sha256') if hit else None
        if not isinstance(digest, str) or len(digest) != 64:
            digest = hashlib.sha256(memoryview(wave).cast('B')).hexdigest()
            if digest != facts.get('rir_sha256'):
                raise OnlineRIRError(f'{name} RIR content identity mismatch')
        facts['cache_hit'] = hit
        frames = wave if moving else (wave,)
        infos[name] = {
            'shape': list(expected), 'facts': facts,
            'frame_hashes': (
                [hashlib.sha256(memoryview(frame).cast('B')).hexdigest() for frame in frames]
                if moving
                else [digest]
            ),
        }
        result[name] = (wave, facts)
    return result, infos


def _stop_worker(process):
    if process.is_alive():
        process.terminate()
        process.join(0.5)
    if process.is_alive():
        process.kill()
        process.join(0.5)
    if process.is_alive():
        raise OnlineRIRError('gpuRIR backend worker could not be stopped')
    process.close()


def _run_backend(plan, track_kind, timeout, backend_factory, requested_timeout_sec):
    """按 track_kind 调用 backend；不涉及进程与 IPC。"""
    backend = backend_factory(plan) if backend_factory is not None else None
    from core.data.simulation import gpurir_backend as backend_impl
    if track_kind in {'scene_moving_full', 'scene_moving_full_direct'}:
        return backend_impl.generate_moving_scene_rir_tracks(
            plan, backend, timeout=timeout,
            track_kind='full' if track_kind == 'scene_moving_full' else 'full_direct',
            _requested_timeout_sec=requested_timeout_sec,
        )
    if track_kind == 'moving_full':
        wave, facts = backend_impl.generate_moving_rir_group(
            plan,
            backend,
            timeout=timeout,
            track_kind='full',
            _requested_timeout_sec=requested_timeout_sec,
        )
        return {'full': (wave, facts)}
    if track_kind == 'moving_full_direct':
        return backend_impl.generate_moving_rir_tracks(
            plan,
            backend,
            timeout=timeout,
            _requested_timeout_sec=requested_timeout_sec,
        )
    if track_kind == 'full_direct':
        return backend_impl.generate_rir_tracks(
            plan,
            backend,
            timeout=timeout,
            _requested_timeout_sec=requested_timeout_sec,
        )
    return {
        'full': backend_impl.generate_rir_group_budgeted(
            plan,
            backend,
            timeout=timeout,
            track_kind=track_kind,
            _requested_timeout_sec=requested_timeout_sec,
        )
    }


def _generate_once(plan, track_kind, conn, timeout, backend_factory, requested_timeout_sec):
    """执行一次生成并把结果写回常驻 worker 的管道。

    应用层异常只作为 error 响应回传，不关闭管道也不退出进程：worker 的
    CUDA context 与已 import 的模块在下一个请求继续复用。
    """
    try:
        generated = _run_backend(
            plan, track_kind, timeout, backend_factory, requested_timeout_sec
        )
        names = ('full',) if track_kind in {'full', 'moving_full', 'scene_moving_full'} else ('full', 'direct')
        tracks = {
            name: {'facts': _json_safe(facts)}
            for name, (wave, facts) in ((name, generated[name]) for name in names)
        }
        conn.send_bytes(json.dumps({'track_kind': track_kind, 'tracks': tracks}, allow_nan=False).encode())
        for name in names:
            wave = np.ascontiguousarray(generated[name][0], dtype='<f4')
            moving = track_kind in _MOVING_TRACK_KINDS
            for frame in (wave if moving else (wave,)):
                conn.send_bytes(frame.tobytes())
    except BaseException as exc:
        try:
            conn.send_bytes(json.dumps({'error': f'{type(exc).__name__}: {exc}'}).encode())
        except (OSError, BrokenPipeError):
            pass


def _generation_worker_loop(conn, backend_factory=None):
    """常驻生成 worker：import 与 CUDA 初始化只付一次，之后按请求复用。"""
    try:
        while True:
            try:
                payload = conn.recv_bytes()
            except (EOFError, OSError):
                return
            try:
                request = json.loads(payload)
            except ValueError:
                return
            if not isinstance(request, dict):
                return
            if request.get('action') == 'close':
                return
            _generate_once(
                request.get('plan'),
                request.get('track_kind'),
                conn,
                request.get('timeout'),
                backend_factory,
                request.get('requested_timeout_sec'),
            )
    finally:
        try:
            conn.close()
        except OSError:
            pass


class _WorkerUnavailable(RuntimeError):
    """worker 进程或管道不可再用；调用方必须回收该 slot 并重建。

    ``error`` 是向上抛给请求方的异常，保证超时/断连仍是 fail-closed。
    """

    def __init__(self, message, error):
        super().__init__(message)
        self.error = error


def _requires_worker_recycle(error):
    """识别 CUDA/backend 致命错误，避免污染后续请求的 CUDA context。"""
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            'out of memory',
            'cuda error',
            'cuda_error',
            'cuda runtime',
            'illegal memory access',
            'device-side assert',
            'cublas_status',
            'cudnn_status',
        )
    )


@dataclass
class _WorkerSlot:
    """一个常驻生成 worker 及其 duplex 管道。"""

    process: object
    conn: object


def _resolve_generation_workers():
    """解析常驻 worker 数；默认 2，可用 USE_TRAIN_GPURIR_WORKERS 覆盖。"""
    raw = os.environ.get('USE_TRAIN_GPURIR_WORKERS')
    if raw is None or not raw.strip():
        return _DEFAULT_GENERATION_WORKERS
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise OnlineRIRError('USE_TRAIN_GPURIR_WORKERS must be an integer') from exc
    if not 1 <= value <= _MAX_GENERATION_WORKERS:
        raise OnlineRIRError(
            f'USE_TRAIN_GPURIR_WORKERS must be in [1, {_MAX_GENERATION_WORKERS}]'
        )
    return value


class _GenerationPool:
    """常驻 gpuRIR 生成 worker 池。

    每个 slot 是一个已经 import gpuRIR、已经初始化 CUDA 的常驻进程；请求经
    duplex Pipe 派发。

    看门狗语义：超时或管道损坏时回收该 slot 并重建，未完成的请求以
    fail-closed 方式抛错；应用层异常（预算/几何校验）不回收 worker，因为
    管道与 CUDA context 仍然健康。
    """

    def __init__(self, size, backend_factory=None):
        if type(size) is not int or size < 1:
            raise OnlineRIRError('gpuRIR generation pool size must be a positive exact int')
        self._size = size
        self._backend_factory = backend_factory
        self._available = queue.Queue()
        self._state_lock = threading.Lock()
        self._slots = []
        self._closed = False
        for _ in range(size):
            slot = self._spawn()
            self._slots.append(slot)
            self._available.put(slot)

    @property
    def size(self):
        return self._size

    def _spawn(self):
        ctx = mp.get_context('spawn')
        parent, child = ctx.Pipe(duplex=True)
        process = ctx.Process(
            target=_generation_worker_loop, args=(child, self._backend_factory)
        )
        try:
            process.start()
        finally:
            child.close()
        return _WorkerSlot(process, parent)

    def _recycle(self, slot):
        """回收不可用的 slot 并补一个新的；回收失败不得覆盖 in-flight 根因。"""
        try:
            _stop_worker(slot.process)
        except OnlineRIRError:
            logging.getLogger(__name__).exception(
                'failed to stop gpuRIR generation worker; keeping the original failure'
            )
        try:
            slot.conn.close()
        except OSError:
            pass
        try:
            replacement = self._spawn()
        except BaseException:
            # 池容量缩小但服务仍可继续；后续请求会在剩余 worker 上排队。
            logging.getLogger(__name__).exception(
                'failed to respawn gpuRIR generation worker'
            )
            return
        with self._state_lock:
            if self._closed:
                try:
                    _stop_worker(replacement.process)
                except OnlineRIRError:
                    logging.getLogger(__name__).exception(
                        'failed to stop replacement gpuRIR generation worker'
                    )
                try:
                    replacement.conn.close()
                except OSError:
                    pass
                return
            self._slots = [item for item in self._slots if item is not slot]
            self._slots.append(replacement)
            self._available.put(replacement)

    def generate(self, plan, track_kind, expected, timeout, requested_timeout_sec=None):
        """派发一次生成并校验返回的轨道集合与逐帧预算。"""
        deadline = time.monotonic() + float(timeout)
        try:
            slot = self._available.get(timeout=_remaining(deadline))
        except (queue.Empty, TimeoutError) as exc:
            # 回收失败导致池容量缩小时不得无限等待，必须 fail-closed。
            raise OnlineRIRError(
                'no gpuRIR generation worker became available within the deadline'
            ) from exc
        try:
            worker_timeout = _remaining(deadline)
            slot.conn.send_bytes(json.dumps({
                'plan': plan, 'track_kind': track_kind, 'timeout': worker_timeout,
                'requested_timeout_sec': requested_timeout_sec,
            }, allow_nan=False).encode())
        except BaseException as exc:
            self._recycle(slot)
            raise OnlineRIRError(f'gpuRIR generation worker pipe failed: {exc}') from exc
        try:
            generated = self._collect(slot, track_kind, expected, deadline)
        except _WorkerUnavailable as exc:
            self._recycle(slot)
            raise exc.error
        except BaseException:
            # 应用层错误：管道与 worker 仍健康，放回池中继续复用。
            self._available.put(slot)
            raise
        self._available.put(slot)
        return generated

    def _collect(self, slot, track_kind, expected, deadline):
        conn = slot.conn
        try:
            ready = conn.poll(_remaining(deadline))
        except TimeoutError as exc:
            raise _WorkerUnavailable(
                'gpuRIR generation worker exceeded deadline',
                TimeoutError('gpuRIR backend worker exceeded deadline and was terminated'),
            ) from exc
        if not ready:
            raise _WorkerUnavailable(
                'gpuRIR generation worker exceeded deadline',
                TimeoutError('gpuRIR backend worker exceeded deadline and was terminated'),
            )
        try:
            header = json.loads(conn.recv_bytes())
        except (EOFError, OSError, TypeError, ValueError) as exc:
            raise _WorkerUnavailable(
                'gpuRIR generation worker returned an invalid header',
                OnlineRIRError(f'gpuRIR backend worker returned an invalid header: {exc}'),
            ) from exc
        if not isinstance(header, dict):
            raise _WorkerUnavailable(
                'gpuRIR generation worker returned a non-object header',
                OnlineRIRError('gpuRIR backend worker returned a non-object header'),
            )
        if 'error' in header:
            error = OnlineRIRError(str(header['error']))
            if _requires_worker_recycle(error):
                raise _WorkerUnavailable(
                    'gpuRIR worker reported a fatal backend error', error
                )
            raise error
        names = ('full',) if track_kind in {'full', 'moving_full', 'scene_moving_full'} else ('full', 'direct')
        infos = header.get('tracks')
        if header.get('track_kind') != track_kind or not isinstance(infos, dict) or set(infos) != set(names):
            raise _WorkerUnavailable(
                'gpuRIR backend worker returned an invalid track set',
                OnlineRIRError('gpuRIR backend worker returned an invalid track set'),
            )
        moving = track_kind in _MOVING_TRACK_KINDS
        frame_shape = expected[1:] if moving else expected
        frame_count = expected[0] if moving else 1
        frame_bytes = int(np.prod(frame_shape, dtype=object)) * 4
        if frame_bytes > _MAX_FRAME:
            raise MemoryError('RIR worker frame exceeds memory budget')
        generated = {}
        for name in names:
            info = infos[name]
            facts = info.get('facts') if isinstance(info, dict) else None
            if not isinstance(facts, dict):
                raise _WorkerUnavailable(
                    'gpuRIR backend worker returned invalid facts',
                    OnlineRIRError(f'{name} RIR facts must be an object'),
                )
            wave = np.empty(expected, dtype='<f4')
            for index in range(frame_count):
                try:
                    ready = conn.poll(_remaining(deadline))
                except TimeoutError as exc:
                    raise _WorkerUnavailable(
                        'gpuRIR generation worker exceeded deadline',
                        TimeoutError('gpuRIR backend worker exceeded deadline and was terminated'),
                    ) from exc
                payload = conn.recv_bytes() if ready else None
                if payload is None or len(payload) != frame_bytes:
                    raise _WorkerUnavailable(
                        'gpuRIR generation worker byte count mismatch',
                        OnlineRIRError(f'{name} RIR worker byte count mismatch'),
                    )
                frame = np.frombuffer(payload, dtype='<f4').reshape(frame_shape)
                if not np.isfinite(frame).all():
                    raise _WorkerUnavailable(
                        'gpuRIR backend worker returned non-finite data',
                        OnlineRIRError(f'non-finite {name} RIR worker response'),
                    )
                if moving:
                    wave[index] = frame
                else:
                    wave[...] = frame
            generated[name] = (wave, facts)
        try:
            _remaining(deadline)
        except TimeoutError as exc:
            raise _WorkerUnavailable(
                'gpuRIR generation worker exceeded deadline',
                TimeoutError('gpuRIR backend worker exceeded deadline and was terminated'),
            ) from exc
        return generated

    def warmup(self):
        """服务启动时预热每个 worker，避免第一次训练 batch 撞上冷启动。

        实测每个 worker 的首次生成要付约 1-2 s 的 gpuRIR/CUDA 初始化，之后
        降到约 20 ms。这里把该成本提前到启动阶段，预热失败即启动失败。
        """
        for _ in range(self._size):
            plan = _warmup_plan()
            self.generate(plan, 'full', _request_shape(plan, 'full'), _WARMUP_TIMEOUT_SEC)

    def close(self):
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            slots, self._slots = self._slots, []
        for slot in slots:
            try:
                slot.conn.send_bytes(json.dumps({'action': 'close'}).encode())
            except (OSError, BrokenPipeError, EOFError):
                pass
        for slot in slots:
            slot.process.join(1.0)
            if slot.process.is_alive():
                try:
                    _stop_worker(slot.process)
                except OnlineRIRError:
                    logging.getLogger(__name__).exception(
                        'failed to stop gpuRIR generation worker'
                    )
            try:
                slot.conn.close()
            except OSError:
                pass


class _RIRCache:
    """有界 RIR 缓存；并发请求下按条目数与字节数双限。

    ``reserve`` 为同一个 canonical key 提供 single-flight：首个请求负责生成，
    后续请求等待该生成结果进入缓存，避免 DataLoader 并发 miss 重复占用 GPU。
    """

    def __init__(self, max_entries, max_bytes=None):
        self._max_entries = max_entries
        self._max_bytes = _CACHE_BYTES if max_bytes is None else max_bytes
        if type(self._max_bytes) is not int or self._max_bytes < 0:
            raise OnlineRIRError('RIR cache byte budget must be a non-negative exact int')
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._entries = OrderedDict()
        self._inflight = set()
        self._bytes = 0

    @staticmethod
    def _size(entry):
        return sum(wave.nbytes for wave, _ in entry.values())

    def take(self, key):
        """命中即取出；生成失败的请求不得把旧条目留在缓存里。"""
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._bytes -= self._size(entry)
            return entry

    def reserve(self, key, deadline):
        """返回 ``(entry, owner)``；owner 负责生成并调用 ``release``。"""
        with self._condition:
            while True:
                entry = self._entries.pop(key, None)
                if entry is not None:
                    self._bytes -= self._size(entry)
                    return entry, False
                if key not in self._inflight:
                    self._inflight.add(key)
                    return None, True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('timed out waiting for in-flight RIR generation')
                self._condition.wait(remaining)

    def release(self, key):
        """结束 single-flight owner，唤醒等待同一 key 的请求。"""
        with self._condition:
            self._inflight.discard(key)
            self._condition.notify_all()

    def put(self, key, entry):
        total = self._size(entry)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= self._size(previous)
            if total > self._max_bytes:
                return
            self._entries[key] = entry
            self._bytes += total
            while len(self._entries) > self._max_entries or self._bytes > self._max_bytes:
                _, old = self._entries.popitem(last=False)
                self._bytes -= self._size(old)


def _serve_requests(
    listener, token, gpu_rir, cache_size, timeout, backend_factory=None, *,
    metrics=None, device=None, generation_pool=None, handler_threads=None,
):
    """生产 IPC 循环；请求并发派发到常驻 worker 池。

    metrics 中的 bytes 指 JSON 请求 payload 与 waveform payload，不包含 socket
    framing；这样不同客户端的网络 framing 不会影响训练吞吐统计。
    ``generation_pool`` 为空时在本进程内直接调用 ``_run_backend``（测试借此
    注入 fake backend）；有池时按池容量并发处理请求，单个慢请求
    不再阻塞其余 dataloader worker。
    """
    del gpu_rir
    metrics = metrics or RIRServiceMetrics(device)
    cache = _RIRCache(cache_size)
    pool = generation_pool
    stop = threading.Event()

    def handle(conn, accepted_ns):
        with conn:
            service_timeout = max(
                timeout - min(_SERVICE_RESPONSE_RESERVE_SEC, timeout * 0.25),
                timeout * 0.5,
            )
            deadline = time.monotonic() + service_timeout
            request_id = None
            request_kind_label = "unknown"
            track_kind_label = "unknown"
            request_bytes = 0
            queue_wait_ms = 0.0
            generation_latency_ms = 0.0
            cache_hit = False
            response_started = False
            metrics_recorded = False
            generation_owner = False
            try:
                request_payload = _receive(conn, deadline)
                request_bytes = len(request_payload)
                request = json.loads(request_payload)
                if not hmac.compare_digest(request.get('token', ''), token):
                    raise OnlineRIRError('RIR IPC authentication failed')
                if request.get('action') == 'close':
                    stop.set()
                    return
                request_id = request['request_id']
                sent_ns = request.get('client_sent_monotonic_ns')
                if isinstance(sent_ns, int):
                    queue_wait_ms = max(0.0, (accepted_ns - sent_ns) / 1_000_000.0)
                request_kind = request.get('request_kind')
                request_kind_label = request_kind or "legacy"
                if request_kind in _SCENE_REQUEST_KINDS:
                    canonical = _canonical_scene_request(
                        request_kind, request.get('scene'),
                        normalize=request.get('normalize', True),
                    )
                    plan = canonical['plan']
                    track_kind = request.get(
                        'track_kind',
                        'full_direct' if request_kind == 'scene_static' else 'scene_moving_full_direct',
                    )
                    allowed = ({'full', 'full_direct'} if request_kind == 'scene_static'
                               else {'scene_moving_full', 'scene_moving_full_direct'})
                    if track_kind not in allowed:
                        raise OnlineRIRError(
                            f'{request_kind} does not support track_kind={track_kind!r}'
                        )
                else:
                    plan = request['plan']
                    track_kind = request.get('track_kind', 'full')
                track_kind_label = track_kind
                expected = _request_shape(plan, track_kind)
                key = json.dumps(
                    {'request_kind': request_kind, 'track_kind': track_kind, 'plan': plan},
                    sort_keys=True, separators=(',', ':'), allow_nan=False,
                )
                generated, generation_owner = cache.reserve(key, deadline)
                cache_hit = generated is not None
                try:
                    if not cache_hit:
                        generation_started = time.perf_counter()
                        if pool is not None:
                            generated = pool.generate(
                                plan, track_kind, expected, _remaining(deadline),
                                requested_timeout_sec=timeout,
                            )
                        else:
                            generated = _run_backend(
                                plan, track_kind, _remaining(deadline),
                                backend_factory, None,
                            )
                        generation_latency_ms = (time.perf_counter() - generation_started) * 1000.0
                    _remaining(deadline)
                    generated, infos = _prepare_tracks(generated, plan, track_kind, expected, cache_hit)
                    total_bytes = sum(wave.nbytes for wave, _ in generated.values())
                    cache.put(key, generated)
                finally:
                    if generation_owner:
                        cache.release(key)
                request_metrics = {
                    "request_kind": request_kind_label,
                    "track_kind": track_kind,
                    "cache_hit": bool(cache_hit),
                    "queue_wait_ms": float(queue_wait_ms),
                    "generation_latency_ms": float(generation_latency_ms),
                    "bytes_in": int(request_bytes),
                    "bytes_out": int(total_bytes),
                    "gpu_id": device,
                }
                service_metrics = metrics.record(
                    request_kind=request_kind_label, track_kind=track_kind,
                    cache_hit=cache_hit, queue_wait_ms=queue_wait_ms,
                    generation_latency_ms=generation_latency_ms,
                    bytes_in=request_bytes, bytes_out=total_bytes,
                )
                metrics_recorded = True
                for info in infos.values():
                    info["facts"]["request_metrics"] = dict(request_metrics)
                    info["facts"]["service_metrics"] = dict(service_metrics)
                if track_kind == 'full':
                    header = {'request_id': request_id, 'track_kind': track_kind, **infos['full']}
                else:
                    header = {'request_id': request_id, 'track_kind': track_kind, 'tracks': infos}
                header["request_metrics"] = request_metrics
                header["service_metrics"] = service_metrics
                header_payload = json.dumps(header, allow_nan=False).encode()
                if len(header_payload) > _MAX_FRAME:
                    # frame budget 不能因为可选指标挤压 waveform framing；
                    # 指标仍累计在 service 内，并在普通预算响应中发布。
                    for info in infos.values():
                        info["facts"].pop("request_metrics", None)
                        info["facts"].pop("service_metrics", None)
                    header.pop("request_metrics", None)
                    header.pop("service_metrics", None)
                    header_payload = json.dumps(header, allow_nan=False).encode()
                response_started = True
                _send(conn, header_payload, deadline)
                names = ('full',) if track_kind in {'full', 'moving_full', 'scene_moving_full'} else ('full', 'direct')
                for name in names:
                    wave, _ = generated[name]
                    for frame in (wave if track_kind in _MOVING_TRACK_KINDS else (wave,)):
                        _send(conn, frame.tobytes(), deadline)
            except Exception as exc:
                timed_out = isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()
                if metrics_recorded:
                    metrics.mark_failure(timed_out=timed_out)
                else:
                    metrics.record(
                        request_kind=request_kind_label, track_kind=track_kind_label,
                        cache_hit=cache_hit, queue_wait_ms=queue_wait_ms,
                        generation_latency_ms=generation_latency_ms,
                        bytes_in=request_bytes, bytes_out=0,
                        failed=True, timed_out=timed_out,
                    )
                # header 后不得把 JSON 错误当作 waveform 包继续发送。
                if not response_started:
                    try:
                        _send(conn, json.dumps({'request_id': request_id, 'error': f'{type(exc).__name__}: {exc}'}).encode())
                    except (OSError, TimeoutError):
                        pass

    # 有池时按池容量并发：每个 handler 线程阻塞在 pool.generate 上等一个空闲
    # worker，单个慢请求不再阻塞其余 dataloader worker。无池（测试 seam）时
    # 单线程串行，保持注入 fake backend 的既有语义。
    thread_count = max(1, handler_threads or (pool.size if pool is not None else 1))
    connections = queue.Queue(maxsize=thread_count)

    def run_handler():
        while True:
            item = connections.get()
            if item is None:
                return
            handle(*item)

    threads = [
        threading.Thread(target=run_handler, daemon=True, name=f'rir-handler-{index}')
        for index in range(thread_count)
    ]
    for thread in threads:
        thread.start()
    previous_timeout = listener.gettimeout()
    listener.settimeout(_ACCEPT_POLL_SEC)
    try:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            connections.put((conn, time.monotonic_ns()))
    finally:
        listener.settimeout(previous_timeout)
        for _ in threads:
            try:
                connections.put(None, timeout=2.0)
            except queue.Full:
                pass
        for thread in threads:
            thread.join(5.0)


def _server(address, token, device, queue_size, cache_size, timeout, ready, backend_factory=None):
    listener = None
    generation_pool = None
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        if backend_factory is None:
            try:
                import torch
                if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
                    raise RuntimeError('CUDA is unavailable in the gpuRIR service process')
                torch.cuda.current_device()
            except Exception as exc:
                raise OnlineRIRError(f'cannot initialize CUDA for gpuRIR: {exc}') from exc
            try:
                import gpuRIR
            except Exception as exc:
                raise OnlineRIRError(f'cannot import gpuRIR: {exc}') from exc
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(address)
        os.chmod(address, 384)
        listener.listen(queue_size)
        generation_pool = _GenerationPool(_resolve_generation_workers(), backend_factory)
        if backend_factory is None:
            # 真实 gpuRIR 服务在报告 ready 之前完成预热；注入 fake backend 的
            # 测试 seam 不做预热，保持既有语义与耗时。
            generation_pool.warmup()
        ready.send_bytes(b'OK')
        ready.close()
        _serve_requests(
            listener, token, None, cache_size, timeout, backend_factory,
            device=device, generation_pool=generation_pool,
        )
    except BaseException as exc:
        try:
            ready.send_bytes(f'{type(exc).__name__}: {exc}'.encode())
        except (OSError, ValueError):
            pass
    finally:
        if listener is not None:
            listener.close()
        if generation_pool is not None:
            generation_pool.close()
        ready.close()

@dataclass(frozen=True)
class RIRClient:
    address: str
    token: str = field(repr=False)
    timeout: float
    last_metrics: dict | None = field(default=None, init=False, repr=False, compare=False)

    def _capture_metrics(self, header: dict) -> None:
        metrics = header.get("service_metrics")
        if isinstance(metrics, dict):
            object.__setattr__(self, "last_metrics", dict(metrics))

    def metrics_snapshot(self) -> dict:
        """返回最近一次服务响应携带的累计指标快照。"""
        return dict(self.last_metrics or {})

    def generate_scene(self, scene, *, moving=False, normalize=True, track_kind=None):
        """发送 canonical scene-level static/moving 请求。"""
        request_kind = 'scene_moving' if moving else 'scene_static'
        canonical = _canonical_scene_request(request_kind, scene, normalize=normalize)
        request_id = secrets.token_hex(12)
        plan = canonical['plan']
        track_kind = track_kind or ('scene_moving_full_direct' if moving else 'full_direct')
        allowed = ({'scene_moving_full', 'scene_moving_full_direct'} if moving else {'full', 'full_direct'})
        if track_kind not in allowed:
            raise OnlineRIRError(
                f'{request_kind} does not support track_kind={track_kind!r}'
            )
        expected = _request_shape(plan, track_kind)
        frame_shape = expected[1:] if moving else expected
        frame_count = expected[0] if moving else 1
        frame_bytes = int(np.prod(frame_shape, dtype=object)) * 4
        deadline = time.monotonic() + self.timeout
        request = {'token': self.token, 'request_id': request_id,
                   'request_kind': request_kind, 'scene': scene,
                   'track_kind': track_kind,
                   'normalize': bool(normalize),
                   'client_sent_monotonic_ns': time.monotonic_ns()}
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(_remaining(deadline)); sock.connect(self.address)
                _send(sock, json.dumps(request, allow_nan=False).encode(), deadline)
                header = json.loads(_receive(sock, deadline))
                self._capture_metrics(header)
                if header.get('request_id') != request_id:
                    raise OnlineRIRError('RIR scene response identity mismatch')
                if 'error' in header:
                    raise OnlineRIRError(header['error'])
                if header.get('track_kind') != track_kind:
                    raise OnlineRIRError('RIR scene response track_kind mismatch')
                names = ('full',) if track_kind in {'full', 'scene_moving_full'} else ('full', 'direct')
                infos = ({'full': header} if track_kind == 'full' else header.get('tracks'))
                if not isinstance(infos, dict) or set(infos) != set(names):
                    raise OnlineRIRError('RIR scene response track set mismatch')
                result = {}
                for name in names:
                    info = infos[name]
                    if tuple(info.get('shape', ())) != expected:
                        raise OnlineRIRError(f'{name} RIR scene shape mismatch')
                    facts = info.get('facts', {})
                    _validate_scene_track_facts(
                        facts, plan, name, moving=moving,
                    )
                    if isinstance(header.get('service_metrics'), dict):
                        facts = dict(facts)
                        facts['service_metrics'] = header['service_metrics']
                    if isinstance(header.get('request_metrics'), dict):
                        facts = dict(facts)
                        facts['request_metrics'] = header['request_metrics']
                    wave = np.empty(expected, dtype='<f4'); digest = hashlib.sha256()
                    for index in range(frame_count):
                        payload = _receive(sock, deadline, max_bytes=frame_bytes)
                        if len(payload) != frame_bytes:
                            raise OnlineRIRError(f'{name} RIR scene byte count mismatch')
                        frame_hashes = info.get('frame_hashes', [])
                        if frame_hashes and hashlib.sha256(payload).hexdigest() != frame_hashes[index]:
                            raise OnlineRIRError(f'{name} RIR scene frame identity mismatch')
                        frame = np.frombuffer(payload, dtype='<f4').reshape(frame_shape)
                        if moving: wave[index] = frame
                        else: wave[...] = frame
                        digest.update(payload)
                    if digest.hexdigest() != facts.get('rir_sha256'):
                        raise OnlineRIRError(f'{name} RIR scene content identity mismatch')
                    result[name] = (wave, facts)
                return result
        except (OSError, EOFError, TimeoutError, ValueError, KeyError, MemoryError, TypeError) as exc:
            raise OnlineRIRError(f'gpuRIR scene service failed: {exc}') from exc

    def generate(self, plan):
        request_id = secrets.token_hex(12)
        request = {'token': self.token, 'request_id': request_id, 'plan': plan,
                   'client_sent_monotonic_ns': time.monotonic_ns()}
        deadline = time.monotonic() + self.timeout
        try:
            _request_shape(plan, 'full')
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(_remaining(deadline))
                sock.connect(self.address)
                _send(sock, json.dumps(request, allow_nan=False).encode(), deadline)
                header = json.loads(_receive(sock, deadline))
                self._capture_metrics(header)
                if header.get('request_id') != request_id:
                    raise OnlineRIRError('RIR response identity mismatch')
                if 'error' in header:
                    raise OnlineRIRError(header['error'])
                expected = _request_shape(plan, 'full')
                shape = header.get('shape')
                if not isinstance(shape, list) or any(type(value) is not int for value in shape) or tuple(shape) != expected:
                    raise OnlineRIRError(f'RIR response geometry mismatch: got {shape}, expected {expected}')
                payload = _receive(sock, deadline)
                if len(payload) != int(np.prod(expected)) * 4:
                    raise OnlineRIRError('RIR response byte count mismatch')
                wave = np.frombuffer(payload, dtype='<f4').reshape(expected).copy()
                if not np.isfinite(wave).all():
                    raise OnlineRIRError('non-finite RIR response')
                facts = header['facts']
                if isinstance(header.get('service_metrics'), dict):
                    facts = dict(facts)
                    facts['service_metrics'] = header['service_metrics']
                if isinstance(header.get('request_metrics'), dict):
                    facts = dict(facts)
                    facts['request_metrics'] = header['request_metrics']
                if facts.get('group_uid') != plan['group_uid']:
                    raise OnlineRIRError('RIR group identity mismatch')
                if hashlib.sha256(payload).hexdigest() != facts.get('rir_sha256'):
                    raise OnlineRIRError('RIR response content identity mismatch')
                return (wave, facts)
        except (OSError, EOFError, TimeoutError, ValueError, KeyError, MemoryError) as exc:
            raise OnlineRIRError(f'gpuRIR service failed: {exc}') from exc

    def generate_tracks(self, plan, *, track_kind="full_direct"):
        """获取静态或 moving 轨道；moving 每个 KCL 帧独立传输并复核 hash。"""
        if track_kind not in {'full_direct', 'moving_full', 'moving_full_direct'}:
            raise OnlineRIRError(f'unsupported grouped track_kind: {track_kind!r}')
        request_id = secrets.token_hex(12)
        deadline = time.monotonic() + self.timeout
        try:
            # 固化请求，调用方后续修改 dict 不得改变响应校验基准。
            plan = json.loads(json.dumps(plan, allow_nan=False))
            expected = _request_shape(plan, track_kind)
            moving = track_kind in _MOVING_TRACK_KINDS
            frame_shape = expected[1:] if moving else expected
            frame_count = expected[0] if moving else 1
            frame_bytes = int(np.prod(frame_shape, dtype=object)) * 4
            request = {'token': self.token, 'request_id': request_id, 'plan': plan, 'track_kind': track_kind,
                       'client_sent_monotonic_ns': time.monotonic_ns()}
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(_remaining(deadline))
                sock.connect(self.address)
                _send(sock, json.dumps(request, allow_nan=False).encode(), deadline)
                header = json.loads(_receive(sock, deadline))
                self._capture_metrics(header)
                if header.get('request_id') != request_id:
                    raise OnlineRIRError('RIR grouped response identity mismatch')
                if 'error' in header:
                    raise OnlineRIRError(header['error'])
                if header.get('track_kind') != track_kind:
                    raise OnlineRIRError('RIR grouped response track_kind mismatch')
                infos = header.get('tracks')
                names = ('full',) if track_kind == 'moving_full' else ('full', 'direct')
                if not isinstance(infos, dict) or set(infos) != set(names):
                    raise OnlineRIRError('RIR response track set mismatch')
                for name, info in infos.items():
                    shape = info['shape']
                    if not isinstance(shape, list) or any(type(v) is not int for v in shape) or tuple(shape) != expected:
                        raise OnlineRIRError(f'{name} RIR response geometry mismatch')
                    _validate_track_facts(info['facts'], plan, name, moving=moving)
                    if isinstance(header.get('service_metrics'), dict):
                        info['facts']['service_metrics'] = header['service_metrics']
                    if isinstance(header.get('request_metrics'), dict):
                        info['facts']['request_metrics'] = header['request_metrics']
                    hashes = info.get('frame_hashes')
                    if not isinstance(hashes, list) or len(hashes) != frame_count:
                        raise OnlineRIRError(f'{name} RIR frame hash count mismatch')
                tracks = {}
                for name in names:
                    info = infos[name]
                    wave = np.empty(expected, dtype='<f4')
                    digest = hashlib.sha256()
                    for index in range(frame_count):
                        payload = _receive(sock, deadline, max_bytes=frame_bytes)
                        if len(payload) != frame_bytes:
                            raise OnlineRIRError(f'{name} RIR response byte count mismatch')
                        if hashlib.sha256(payload).hexdigest() != info['frame_hashes'][index]:
                            raise OnlineRIRError(f'{name} RIR frame content identity mismatch')
                        frame = np.frombuffer(payload, dtype='<f4').reshape(frame_shape)
                        if not np.isfinite(frame).all():
                            raise OnlineRIRError(f'non-finite {name} RIR response')
                        if moving:
                            wave[index] = frame
                        else:
                            wave[...] = frame
                        digest.update(payload)
                    if digest.hexdigest() != info['facts']['rir_sha256']:
                        raise OnlineRIRError(f'{name} RIR content identity mismatch')
                    tracks[name] = (wave, info['facts'])
                _remaining(deadline)
                return tracks
        except (OSError, EOFError, TimeoutError, ValueError, KeyError, MemoryError, TypeError, AttributeError, ImportError) as exc:
            raise OnlineRIRError(f'gpuRIR grouped service failed: {exc}') from exc

class _Owner:

    def __init__(self, cfg, device):
        self.directory = Path(tempfile.mkdtemp(prefix='use_train_gpurir_'))
        os.chmod(self.directory, 448)
        self.client = RIRClient(str(self.directory / 'service.sock'), secrets.token_hex(32), cfg.timeout)
        ctx = mp.get_context('spawn')
        (parent, child) = ctx.Pipe(duplex=False)
        self.process = ctx.Process(target=_server, args=(self.client.address, self.client.token, device, cfg.queue_size, cfg.cache_size, cfg.timeout, child))
        try:
            self.process.start()
            child.close()
            if not parent.poll(cfg.timeout):
                raise OnlineRIRError('gpuRIR service startup timed out')
            reply = parent.recv_bytes()
            if reply != b'OK':
                raise OnlineRIRError(reply.decode())
        except BaseException:
            self.close()
            raise
        finally:
            parent.close()
            child.close()

    def close(self):
        if self.process.pid is not None:
            if self.process.is_alive():
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                        sock.settimeout(1)
                        sock.connect(self.client.address)
                        _send(sock, json.dumps({'token': self.client.token, 'action': 'close'}).encode())
                except OSError:
                    pass
                self.process.join(2)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(2)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(2)
            self.process.close()
        shutil.rmtree(self.directory, ignore_errors=True)

def acquire_rir_client(cfg):
    if mp.current_process().daemon:
        raise OnlineRIRError('online RIR service must be acquired in the Dataset owner before DataLoader workers start')
    index = cfg.device if cfg.device is not None else int(os.environ.get('LOCAL_RANK', '0'))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        tokens = [t.strip() for t in visible.split(',') if t.strip()]
        if index >= len(tokens):
            raise OnlineRIRError('gpurir_device is outside CUDA_VISIBLE_DEVICES')
        device = tokens[index]
    else:
        device = str(index)
    key = (os.getpid(), device, cfg.timeout, cfg.queue_size, cfg.cache_size)
    with _LOCK:
        if key not in _OWNERS:
            _OWNERS[key] = _Owner(cfg, device)
        return _OWNERS[key].client

def close_online_rir_services():
    with _LOCK:
        owners = [_OWNERS.pop(k) for k in list(_OWNERS) if k[0] == os.getpid()]
    for owner in owners:
        try:
            owner.close()
        except BaseException:
            logging.getLogger(__name__).exception('failed to close owned RIR service')
atexit.register(close_online_rir_services)
