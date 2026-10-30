"""gpuRIR 整组调用；仅由受管理 spawn 服务导入 GPU 扩展。"""
from __future__ import annotations
import hashlib
import time
from collections.abc import Mapping

import numpy as np


GPURIR_GENERATOR_VERSION = "gpuRIR-online-v4"
SUPPORTED_TRACK_KINDS = frozenset({"full", "direct"})
# 单次 gpuRIR 调用的内部 allocation 预算。主导项是 ISM 分组缓冲
# K*C*ism_samples*ceil(prod(nb_img)/512)：小房间 + 高 RT60 会让 nb_img 急剧
# 增长，64 MB 会把配置范围内（RT60 ≤ 1.0、最小房间 3×3×2.5、8 麦）的合法
# plan 判死。256 MB 与 moving 路径的 _MAX_TRACK_BYTES 同口径，相对 80 GB 卡
# 仍是保守值；超出时 generate_rir_group_budgeted 会按 source 自适应分块。
GPURIR_MAX_WAVEFORM_BYTES = 256 * 1024 * 1024


def _plan_shape(plan):
    if not isinstance(plan, Mapping):
        raise TypeError("RIR plan must be a mapping")
    sources = np.asarray(plan.get("source_loc"))
    microphones = np.asarray(plan.get("mic_loc"))
    if sources.ndim != 2 or sources.shape[1:] != (3,) or sources.shape[0] < 1:
        raise ValueError("source_loc must have positive shape [K, 3] before budget validation")
    if microphones.shape == (3,):
        channels = 1
    elif microphones.ndim == 2 and microphones.shape[1:] == (3,):
        channels = microphones.shape[0]
    else:
        raise ValueError("mic_loc must have shape [C, 3] before budget validation")
    if channels < 1:
        raise ValueError("mic_loc must have at least one microphone before budget validation")
    length = plan.get("num_samples")
    if type(length) is not int or length < 1:
        raise ValueError("num_samples must be a positive exact integer before budget validation")
    return int(sources.shape[0]), int(channels), int(length)


def _even_samples(value):
    samples = int(np.ceil(float(value)))
    return samples + samples % 2


def _estimate_backend_bytes(plan, nb_img, tdiff):
    """保守估算 gpuRIR ``simulateRIR`` 的同时存活 host/device 分配。"""
    sources, channels, length = _plan_shape(plan)
    images = tuple(int(value) for value in nb_img)
    if len(images) != 3 or any(value < 1 for value in images):
        raise ValueError("nb_img must contain three positive exact integers")
    if not np.isfinite(tdiff) or tdiff < 0:
        raise ValueError("tdiff must be finite and non-negative")
    count = sources * channels
    image_count = int(np.prod(images, dtype=np.int64))
    total_samples = _even_samples(length)
    ism_samples = min(_even_samples(tdiff * int(plan["sample_rate"])), total_samples)
    initial_reduction = 512
    initial_groups = int(np.ceil(image_count / initial_reduction))
    while count * ism_samples * initial_groups > 1_000_000_000 and initial_groups > 1:
        initial_reduction *= 2
        initial_groups = int(np.ceil(image_count / initial_reduction))
    if count * ism_samples * initial_groups > 1_000_000_000:
        # groups 已缩到 1 仍超阈值：继续减组不会收敛，必须 fail-fast 而不是死循环。
        raise MemoryError(
            "gpuRIR internal allocation estimate cannot be reduced below the group budget: "
            f"K*C*ism={count * ism_samples} samples"
        )
    float_bytes = np.dtype("<f4").itemsize
    allocations = (
        2 * count * image_count
        + 3 * count
        + 3 * count * total_samples
        + count * ism_samples * initial_groups
        + (sources + channels) * 12
    ) * float_bytes
    return int(allocations)


def validate_rir_plan_budget(
    plan,
    *,
    max_bytes: int = GPURIR_MAX_WAVEFORM_BYTES,
    track_count: int = 1,
) -> int:
    """在 backend/IPC 工作前校验所有请求轨道的输出内存上限。"""
    _, _, length = _plan_shape(plan)
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive exact integer")
    if type(track_count) is not int or track_count < 1:
        raise ValueError("track_count must be a positive exact integer")
    sources, channels, _ = _plan_shape(plan)
    requested = sources * channels * length * np.dtype("<f4").itemsize * track_count
    if requested > max_bytes:
        raise MemoryError(
            "RIR waveform exceeds memory budget: "
            f"K*C*L*float32*tracks={requested} bytes > {max_bytes} bytes"
        )
    return requested


def _validate_backend_budget(plan, nb_img, *, tdiff, max_bytes, extra_tracks=()):
    total = _estimate_backend_bytes(plan, nb_img, tdiff)
    for extra_nb_img, extra_tdiff in extra_tracks:
        total += _estimate_backend_bytes(plan, extra_nb_img, extra_tdiff)
    if total > max_bytes:
        raise MemoryError(
            "gpuRIR internal allocation exceeds memory budget: "
            f"estimated={total} bytes > {max_bytes} bytes"
        )
    return total


def _validate_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("gpuRIR generation exceeded timeout deadline")


def _orientation(value, expected, name):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (3,):
        array = np.broadcast_to(array, (expected, 3)).copy()
    if array.shape != (expected, 3) or not np.isfinite(array).all():
        raise ValueError(f'{name} must have shape [{expected}, 3] and be finite')
    return array


def generate_rir_group(
    plan,
    gpu_rir=None,
    *,
    timeout=None,
    track_kind="full",
    _deadline=None,
    _requested_timeout_sec=None,
    _extra_track_specs=(),
):
    """Generate one explicit RIR track from one immutable scene plan.

    ``direct`` uses a direct-path-only image-source request. It never falls back
    to ``full`` or ``early`` when the backend cannot honor the request.
    """
    if track_kind not in SUPPORTED_TRACK_KINDS:
        raise ValueError(f"unsupported gpuRIR track_kind: {track_kind!r}")
    started = time.monotonic()
    requested_timeout_sec = (
        None if _requested_timeout_sec is None else float(_requested_timeout_sec)
    )
    if requested_timeout_sec is not None and requested_timeout_sec <= 0:
        raise ValueError("_requested_timeout_sec must be positive when provided")
    deadline = _deadline if _deadline is not None else (None if timeout is None else started + float(timeout))
    room = np.asarray(plan['room_dim'], dtype=np.float32)
    sources = np.asarray(plan['source_loc'], dtype=np.float32)
    microphones = np.asarray(plan['mic_loc'], dtype=np.float32)
    if room.shape != (3,) or not np.isfinite(room).all() or np.any(room <= 0):
        raise ValueError('room_dim must be finite and have shape [3]')
    if sources.ndim != 2 or sources.shape[1:] != (3,) or not np.isfinite(sources).all():
        raise ValueError('source_loc must be finite and have shape [K, 3]')
    legacy_mono = microphones.shape == (3,)
    if legacy_mono:
        microphones = microphones.reshape(1, 3)
    if microphones.ndim != 2 or microphones.shape[1:] != (3,) or not np.isfinite(microphones).all():
        raise ValueError('mic_loc must be finite and have shape [C, 3]')
    channels = microphones.shape[0]
    length = plan['num_samples']
    sample_rate = plan['sample_rate']
    if type(length) is not int or length < 1 or type(sample_rate) is not int or sample_rate < 1:
        raise ValueError('num_samples and sample_rate must be positive exact integers')
    _validate_deadline(deadline)
    validate_rir_plan_budget(plan)
    if gpu_rir is None:
        try:
            import gpuRIR as gpu_rir
        except Exception as exc:
            raise RuntimeError('gpurir_online requires the optional gpuRIR CUDA extension') from exc
    beta = np.asarray(gpu_rir.beta_SabineEstimation(room, plan['rt60']), dtype=np.float32)
    if beta.ndim != 1 or beta.size != 6 or not np.isfinite(beta).all():
        raise ValueError('gpuRIR returned invalid six-wall beta')
    _validate_deadline(deadline)
    maximum = length / sample_rate
    diffuse = float(gpu_rir.att2t_SabineEstimator(plan['diffuse_db'], plan['rt60']))
    if not np.isfinite(diffuse) or diffuse < 0:
        raise ValueError('gpuRIR returned invalid diffuse transition')
    if diffuse >= maximum:
        diffuse = max(maximum * 0.5, maximum - 1.0 / sample_rate)
    images = gpu_rir.t2n(diffuse, room, c=plan['sound_speed'])
    images = tuple(int(value) for value in images)
    if len(images) != 3 or any(value < 1 for value in images):
        raise ValueError('gpuRIR returned invalid image-source counts')
    _validate_deadline(deadline)
    if track_kind == 'direct':
        direct_images = (1, 1, 1)
        # gpuRIR 的 Tdiff 是 ISM 结束时刻。设为 0 会启用全程漫射尾，
        # 而不是生成直达声；Tdiff=Tmax 才关闭漫射并保留唯一 direct image。
        direct_tdiff = maximum
    else:
        direct_images = images
        direct_tdiff = diffuse
    _validate_backend_budget(
        plan,
        direct_images,
        tdiff=direct_tdiff,
        max_bytes=GPURIR_MAX_WAVEFORM_BYTES,
        extra_tracks=_extra_track_specs,
    )
    wave = np.asarray(gpu_rir.simulateRIR(
        room_sz=room,
        beta=beta,
        pos_src=sources,
        pos_rcv=microphones,
        nb_img=direct_images,
        Tmax=maximum,
        fs=sample_rate,
        Tdiff=direct_tdiff,
        spkr_pattern=plan.get('speaker_pattern', 'omni'),
        mic_pattern=plan.get('microphone_pattern', 'omni'),
        orV_src=_orientation(plan.get('speaker_orientation'), len(sources), 'speaker_orientation'),
        orV_rcv=_orientation(plan.get('microphone_orientation'), channels, 'microphone_orientation'),
        c=plan['sound_speed'],
    ), dtype=np.float32)
    _validate_deadline(deadline)
    expected = (len(sources), channels)
    if wave.ndim != 3 or wave.shape[:2] != expected or wave.shape[-1] < 1 or not np.isfinite(wave).all():
        raise ValueError(f'gpuRIR returned invalid source×microphone group: {wave.shape}, expected {expected}')
    wave = np.pad(wave, ((0, 0), (0, 0), (0, max(0, length - wave.shape[-1]))))[:, :, :length].copy()
    if not np.all(np.max(np.abs(wave), axis=2) > 1e-12):
        raise ValueError('gpuRIR produced a silent source response')
    wave = np.ascontiguousarray(wave, dtype='<f4')
    elapsed_sec = time.monotonic() - started
    provenance = {
        'backend': 'gpuRIR',
        'backend_module': getattr(gpu_rir, '__file__', type(gpu_rir).__name__),
        'requested_shape': [len(sources), channels, length],
        'geometry': 'fixed_microphone_array' if not legacy_mono else 'single_microphone_legacy',
        'array_profile': plan.get('array_profile'),
        'coordinate_system': plan.get('coordinate_system', 'cartesian_m'),
        'reference_channel': plan.get('reference_channel', 0),
        'geometry_provenance': plan.get('geometry_provenance'),
        'validation_recipe': plan.get('validation_recipe'),
        'elapsed_sec': elapsed_sec,
        'timeout_sec': timeout if requested_timeout_sec is None else requested_timeout_sec,
        'requested_timeout_sec': requested_timeout_sec if requested_timeout_sec is not None else timeout,
        'deadline_timeout_sec': timeout,
        'deadline_remaining_sec': (
            None if deadline is None else max(0.0, deadline - time.monotonic())
        ),
    }
    facts = {**plan, 'mic_loc': microphones.tolist() if not legacy_mono else plan['mic_loc'],
             'num_channels': channels, 'beta': beta.tolist(), 'Tdiff': direct_tdiff,
             'nb_img': list(direct_images), 'track_kind': track_kind, 'provenance': provenance,
             'generator_version': GPURIR_GENERATOR_VERSION,
             'rir_sha256': hashlib.sha256(wave.tobytes()).hexdigest(),
             'waveform_determinism': 'backend-dependent; fixed geometry is deterministic'}
    # scene-level client 需要逐轨道核对 identity；legacy plan 不受影响。
    if plan.get('scene_uid') is not None:
        facts.update({
            'scene_uid': plan['scene_uid'],
            'actor_order': list(plan.get('source_ids', ())),
            'room': plan.get('room'),
            'array': plan.get('array'),
        })
    return (wave[:, 0, :] if legacy_mono else wave, facts)


def _merge_source_chunks(parts, plan, start, stop):
    """把 source 分块结果拼回单块，并按拼接后的波形重建内容身份。"""
    arrays = [np.asarray(item[0], dtype='<f4') for item in parts]
    if not arrays:
        raise ValueError('source chunk merge requires at least one part')
    trailing_shape = arrays[0].shape[1:]
    if any(array.shape[1:] != trailing_shape for array in arrays[1:]):
        raise ValueError('source chunk waveform shapes are inconsistent')
    wave = np.empty(
        (sum(array.shape[0] for array in arrays), *trailing_shape), dtype='<f4'
    )
    offset = 0
    for array in arrays:
        stop_offset = offset + array.shape[0]
        wave[offset:stop_offset] = array
        offset = stop_offset
    facts = dict(parts[0][1])
    facts['source_loc'] = [list(position) for position in plan['source_loc'][start:stop]]
    source_ids = plan.get('source_ids')
    if isinstance(source_ids, (list, tuple)):
        chunk_ids = list(source_ids)[start:stop]
        facts['source_ids'] = chunk_ids
        if 'actor_order' in facts:
            facts['actor_order'] = chunk_ids
    provenance = dict(facts.get('provenance') or {})
    provenance['requested_shape'] = list(wave.shape)
    provenance['elapsed_sec'] = sum(
        float((item[1].get('provenance') or {}).get('elapsed_sec') or 0.0) for item in parts
    )
    provenance['source_chunks'] = len(parts)
    facts['provenance'] = provenance
    facts['rir_sha256'] = hashlib.sha256(wave.tobytes()).hexdigest()
    return wave, facts


def _generate_rir_group_chunked(
    plan, gpu_rir, *, start, stop, timeout, track_kind, _deadline,
    _requested_timeout_sec, _extra_track_specs,
):
    """在 source 区间 [start, stop) 上生成；超预算时二分，单 source 仍超则抛错。"""
    chunk_plan = dict(plan)
    chunk_plan['source_loc'] = [
        list(position) for position in plan['source_loc'][start:stop]
    ]
    source_ids = plan.get('source_ids')
    if isinstance(source_ids, (list, tuple)):
        chunk_plan['source_ids'] = list(source_ids)[start:stop]
    try:
        return generate_rir_group(
            chunk_plan, gpu_rir, timeout=timeout, track_kind=track_kind,
            _deadline=_deadline, _requested_timeout_sec=_requested_timeout_sec,
            _extra_track_specs=_extra_track_specs,
        )
    except MemoryError:
        if stop - start <= 1:
            raise
        middle = start + (stop - start) // 2
        parts = [
            _generate_rir_group_chunked(
                plan, gpu_rir, start=begin, stop=end, timeout=timeout,
                track_kind=track_kind, _deadline=_deadline,
                _requested_timeout_sec=_requested_timeout_sec,
                _extra_track_specs=_extra_track_specs,
            )
            for begin, end in ((start, middle), (middle, stop))
        ]
        return _merge_source_chunks(parts, plan, start, stop)


def generate_rir_group_budgeted(
    plan, gpu_rir=None, *, timeout=None, track_kind="full", _deadline=None,
    _requested_timeout_sec=None, _extra_track_specs=(),
):
    """按 source 自适应分块生成整组 RIR。

    未超预算时与 ``generate_rir_group`` 完全同路径；超预算时按 source 二分，
    逐块复用同一 backend 调用，最后沿 source 轴拼回，并重建 ``source_loc`` /
    ``source_ids`` / ``actor_order`` / ``rir_sha256``。单 source 仍超预算时保持
    fail-closed，不静默降级或减少 source。

    分块只在整块原本会 ``MemoryError`` 时触发，因此不会改变任何此前成功 plan
    的输出；同一 plan 在同一预算下的分块方式固定，结果可复现。gpuRIR 本身是
    shape-dependent 的（facts 已声明 ``waveform_determinism: backend-dependent``），
    不同分块深度之间不保证逐 bit 相同，但同一分块下逐 bit 一致。
    """
    sources = plan.get('source_loc')
    if not isinstance(sources, (list, tuple)) or len(sources) < 1:
        return generate_rir_group(
            plan, gpu_rir, timeout=timeout, track_kind=track_kind, _deadline=_deadline,
            _requested_timeout_sec=_requested_timeout_sec,
            _extra_track_specs=_extra_track_specs,
        )
    return _generate_rir_group_chunked(
        plan, gpu_rir, start=0, stop=len(sources), timeout=timeout, track_kind=track_kind,
        _deadline=_deadline, _requested_timeout_sec=_requested_timeout_sec,
        _extra_track_specs=_extra_track_specs,
    )


def generate_rir_tracks(
    plan, gpu_rir=None, *, timeout=None, _requested_timeout_sec=None
):
    """Generate full and exact direct tracks from the same scene plan.

    The backend is called twice with identical room, source, microphone, and
    ordering fields; only the explicit track kind changes.
    """
    validate_rir_plan_budget(plan, track_count=2)
    started = time.monotonic()
    deadline = None if timeout is None else started + float(timeout)
    direct_spec = ((1, 1, 1), plan["num_samples"] / plan["sample_rate"])
    full, full_facts = generate_rir_group_budgeted(
        plan,
        gpu_rir,
        timeout=timeout,
        track_kind="full",
        _deadline=deadline,
        _requested_timeout_sec=_requested_timeout_sec,
        _extra_track_specs=(direct_spec,),
    )
    direct, direct_facts = generate_rir_group_budgeted(
        plan,
        gpu_rir,
        timeout=timeout,
        track_kind="direct",
        _deadline=deadline,
        _requested_timeout_sec=_requested_timeout_sec,
    )
    if full.shape != direct.shape:
        raise ValueError(f"full/direct gpuRIR shape mismatch: {full.shape} vs {direct.shape}")
    shared = ("room_dim", "source_loc", "mic_loc", "num_samples", "sample_rate", "group_uid")
    for key in shared:
        if full_facts.get(key) != direct_facts.get(key):
            raise ValueError(f"full/direct scene plan mismatch for {key}")
    return {"full": (full, full_facts), "direct": (direct, direct_facts)}


def validate_moving_rir_plan(plan, *, max_bytes=256 * 1024 * 1024, track_count=2):
    """校验在线 moving 请求并返回 ``[T,K,3]`` 的轨迹。"""
    from core.data.simulation.trajectory import TrajectoryTimebase

    if type(track_count) is not int or track_count < 1:
        raise ValueError("moving RIR track_count must be a positive exact integer")
    frame_bytes = validate_rir_plan_budget(plan)
    trajectory = np.asarray(plan.get("trajectory"), dtype=np.float32)
    if trajectory.ndim != 3 or trajectory.shape[2] != 3 or trajectory.shape[0] < 1 or trajectory.shape[1] < 1:
        raise ValueError("moving RIR plan trajectory must have shape [T,K,3]")
    if not np.isfinite(trajectory).all():
        raise ValueError("moving RIR trajectory must be finite")
    source_loc = np.asarray(plan.get("source_loc"), dtype=np.float32)
    if source_loc.shape != trajectory.shape[1:] or not np.allclose(source_loc, trajectory[0], rtol=0.0, atol=1e-6):
        raise ValueError("moving RIR plan source_loc must equal trajectory[0]")
    timebase = plan.get("trajectory_timebase")
    if not isinstance(timebase, Mapping):
        raise ValueError("moving RIR plan requires trajectory_timebase metadata")
    fields = ("sample_rate", "num_samples", "hop_samples", "frame_count", "start_sample")
    clock = TrajectoryTimebase(**{key: timebase[key] for key in fields if key in timebase})
    if clock.frame_count != trajectory.shape[0]:
        raise ValueError("trajectory_timebase.frame_count does not match trajectory")
    if clock.sample_rate != plan['sample_rate'] or clock.start_sample != 0:
        raise ValueError("moving trajectory requires matching sample_rate and start_sample=0")
    if clock.frame_count > 128 or track_count * clock.frame_count * frame_bytes > max_bytes:
        raise MemoryError("moving RIR exceeds trajectory frame/byte budget")
    room = np.asarray(plan['room_dim'], dtype=np.float32)
    if room.shape != (3,) or not np.isfinite(room).all() or np.any(trajectory <= 0) or np.any(trajectory >= room):
        raise ValueError("moving trajectory must stay inside the room")
    return trajectory


def _moving_batch_frame_limit(plan, trajectory):
    """计算单次 ``simulateRIR`` 的保守轨迹帧批大小。

    gpuRIR 一次调用可以接收多个声源位置，因此 moving bank 先展平为
    ``[frames * sources, 3]``，调用结束后再恢复形状。批大小先受输出预算
    限制，不能绕过现有 IPC/内存合同；gpuRIR 内部 allocation 仍由
    ``generate_rir_group`` 检查，必要时会自适应减半。
    """
    microphones = np.asarray(plan["mic_loc"])
    channels = 1 if microphones.shape == (3,) else int(microphones.shape[0])
    sources = int(trajectory.shape[1])
    length = int(plan["num_samples"])
    per_frame = max(1, sources * channels * length * np.dtype("<f4").itemsize)
    return max(1, min(int(trajectory.shape[0]), GPURIR_MAX_WAVEFORM_BYTES // per_frame))


def _generate_moving_batch_bank(
    plan, trajectory, gpu_rir, *, timeout=None, track_kind="full", _requested_timeout_sec=None
):
    """使用批量声源位置生成 moving bank。

    本 helper 保持 ``[P,K,C,L]``/``[P,K,L]`` 合同，只把轨迹拆成受内存约束的
    少量批次，从而减少 backend 调用次数。
    """
    trajectory = np.asarray(trajectory, dtype=np.float32)
    frame_limit = _moving_batch_frame_limit(plan, trajectory)
    started = time.monotonic()
    deadline = None if timeout is None else started + float(timeout)
    bank_out = None
    frame_facts = []
    offset = 0
    while offset < trajectory.shape[0]:
        count = min(frame_limit, trajectory.shape[0] - offset)
        while True:
            chunk = trajectory[offset : offset + count]
            chunk_plan = dict(plan)
            chunk_plan["source_loc"] = chunk.reshape(-1, 3).tolist()
            # 展平批次只是 backend 实现细节；保留 moving 专用字段会让单帧
            # plan 在校验边界被误判为包含不匹配的 trajectory。
            chunk_plan.pop("trajectory", None)
            chunk_plan.pop("trajectory_timebase", None)
            chunk_plan["moving"] = True
            try:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                wave, facts = generate_rir_group(
                    chunk_plan,
                    gpu_rir,
                    timeout=remaining,
                    track_kind=track_kind,
                    _requested_timeout_sec=_requested_timeout_sec,
                )
                break
            except MemoryError:
                if count <= 1:
                    raise
                count = max(1, count // 2)
        sources = chunk.shape[0]
        source_count = trajectory.shape[1]
        if wave.ndim == 2:
            bank = np.asarray(wave, dtype=np.float32).reshape(sources, source_count, wave.shape[-1])
        else:
            bank = np.asarray(wave, dtype=np.float32).reshape(
                sources, source_count, wave.shape[1], wave.shape[2]
            )
        if bank_out is None:
            bank_out = np.empty(
                (trajectory.shape[0], *bank.shape[1:]), dtype='<f4'
            )
        elif bank.shape[1:] != bank_out.shape[1:]:
            raise ValueError(
                f'moving RIR chunk shape changed: {bank.shape[1:]} vs {bank_out.shape[1:]}'
            )
        bank_out[offset : offset + sources] = bank
        for local, frame in enumerate(bank):
            frame_info = dict(facts)
            frame_info.update(
                source_loc=trajectory[offset + local].tolist(),
                trajectory_frame_index=int(offset + local),
                requested_shape=list(frame.shape),
                rir_sha256=hashlib.sha256(
                    np.ascontiguousarray(frame, dtype="<f4").tobytes()
                ).hexdigest(),
            )
            frame_facts.append(frame_info)
        del bank, wave
        offset += sources
    if bank_out is None:
        raise ValueError('moving RIR generation returned no trajectory frames')
    base = frame_facts[0]
    for key in ("room_dim", "mic_loc", "num_samples", "sample_rate", "group_uid"):
        if any(facts.get(key) != base.get(key) for facts in frame_facts[1:]):
            raise ValueError(f"moving RIR scene identity changed across trajectory frames: {key}")
    return bank_out, frame_facts


def generate_moving_rir_group(
    plan, gpu_rir=None, *, timeout=None, track_kind="full", _requested_timeout_sec=None
):
    """Generate a frame-indexed static-RIR bank for an explicit trajectory.

    This is the online trajectory seam: each frame uses the same room, array,
    source ordering, sample rate and scene identity, while only ``source_loc``
    changes. The returned bank has shape ``[T,K,C,L]`` (or ``[T,K,L]`` for
    legacy mono plans). It deliberately does not call ``simulateTrajectory``;
    that higher-level waveform renderer remains a separate contract.
    """
    track_count = 1 if track_kind == "full" else 2
    trajectory = validate_moving_rir_plan(plan, track_count=track_count)
    bank, frame_facts = _generate_moving_batch_bank(
        plan,
        trajectory,
        gpu_rir,
        timeout=timeout,
        track_kind=track_kind,
        _requested_timeout_sec=_requested_timeout_sec,
    )
    base = frame_facts[0]
    return bank, {
        **base,
        "moving": True,
        "track_kind": track_kind,
        "trajectory": trajectory.tolist(),
        "trajectory_timebase": dict(plan["trajectory_timebase"]),
        "frame_facts": frame_facts,
        "rir_sha256": hashlib.sha256(bank.astype('<f4', copy=False).tobytes()).hexdigest(),
    }


def generate_moving_rir_tracks(
    plan, gpu_rir=None, *, timeout=None, _requested_timeout_sec=None
):
    """Generate frame-indexed full/direct banks from one moving scene plan."""
    started = time.monotonic()
    full, full_facts = generate_moving_rir_group(
        plan,
        gpu_rir,
        timeout=timeout,
        track_kind="full",
        _requested_timeout_sec=_requested_timeout_sec,
    )
    _validate_deadline(None if timeout is None else started + float(timeout))
    remaining = None if timeout is None else float(timeout) - (time.monotonic() - started)
    direct, direct_facts = generate_moving_rir_group(
        plan,
        gpu_rir,
        timeout=remaining,
        track_kind="direct",
        _requested_timeout_sec=_requested_timeout_sec,
    )
    if full.shape != direct.shape:
        raise ValueError(f"moving full/direct gpuRIR shape mismatch: {full.shape} vs {direct.shape}")
    if full_facts.get("trajectory") != direct_facts.get("trajectory"):
        raise ValueError("moving full/direct trajectory mismatch")
    return {"full": (full, full_facts), "direct": (direct, direct_facts)}


def generate_moving_scene_rir_tracks(
    plan, gpu_rir=None, *, timeout=None, track_kind="full_direct", _requested_timeout_sec=None
):
    """生成多 actor 独立轨迹 bank，布局为 ``[P,A,C,L]``。

    所有 actor 复用同一 room、array、采样率和 timebase；各 actor 的
    trajectory 按 frame-major/source-major 展平后合批进入 gpuRIR，再按
    canonical ``source_ids`` 顺序恢复为 scene bank。
    """
    if track_kind not in {"full", "full_direct"}:
        raise ValueError("moving scene track_kind must be full or full_direct")
    trajectories = plan.get("actor_trajectories")
    source_ids = tuple(plan.get("source_ids", ()))
    if not isinstance(trajectories, dict) or tuple(trajectories) != source_ids:
        raise ValueError("moving scene actor trajectories must match source_ids order")
    actor_tracks = []
    for actor_uid in source_ids:
        actor_trajectory = np.asarray(trajectories[actor_uid], dtype=np.float32)
        if actor_trajectory.ndim != 3 or actor_trajectory.shape[1:] != (1, 3):
            raise ValueError("moving scene actor trajectory must have shape [P,1,3]")
        actor_tracks.append(actor_trajectory)
    if len({track.shape[0] for track in actor_tracks}) != 1:
        raise ValueError("moving scene actor trajectories must share frame count")
    trajectory = np.concatenate(actor_tracks, axis=1)
    banks = {}
    facts = {}
    started = time.monotonic()
    deadline = None if timeout is None else started + float(timeout)
    selected = ("full",) if track_kind == "full" else ("full", "direct")
    for name in selected:
        kind = name
        batch_plan = dict(plan)
        batch_plan["source_loc"] = trajectory[0].tolist()
        batch_plan["source_ids"] = list(source_ids)
        batch_plan["trajectory"] = trajectory.tolist()
        bank, frame_facts = _generate_moving_batch_bank(
            batch_plan,
            trajectory,
            gpu_rir,
            timeout=None if deadline is None else max(0.0, deadline - time.monotonic()),
            track_kind=kind,
            _requested_timeout_sec=_requested_timeout_sec,
        )
        banks[name] = bank
        facts[name] = frame_facts[0]
    full = banks["full"]
    direct = banks.get("direct")
    if direct is not None and full.shape != direct.shape:
        raise ValueError("moving scene full/direct shape mismatch")
    # actor-specific facts 只用于证明单 actor backend 调用；scene-level facts
    # 必须完全来自 canonical plan，不能继承最后一个 actor 的 source_loc。
    common = {
        **facts["full"],
        "source_loc": [list(position) for position in plan["source_loc"]],
        "source_ids": list(source_ids),
        "mic_loc": [list(position) for position in plan["mic_loc"]],
        "sample_rate": plan["sample_rate"],
        "num_samples": plan["num_samples"],
        "scene_uid": plan["scene_uid"],
        "actor_order": list(source_ids),
        "trajectory": None,
        # 保留 canonical JSON 轨迹，避免 float32 roundtrip 改变 identity。
        "actor_trajectories": trajectories,
        "trajectory_timebase": plan["trajectory_timebase"],
        # 每条返回轨道拥有自己的 track_kind；scene moving 只是请求级别概念。
        "track_kind": None,
        "moving": True,
        "room": plan.get("room"),
        "array": plan.get("array"),
    }
    full = full.astype("<f4", copy=False)
    if direct is not None:
        direct = direct.astype("<f4", copy=False)
    full_facts = {**common, "track_kind": "full", "generator_version": GPURIR_GENERATOR_VERSION,
                  "rir_sha256": hashlib.sha256(full.tobytes()).hexdigest()}
    result = {"full": (full, full_facts)}
    if direct is not None:
        direct_facts = {**common, "track_kind": "direct", "generator_version": GPURIR_GENERATOR_VERSION,
                        "rir_sha256": hashlib.sha256(direct.tobytes()).hexdigest()}
        result["direct"] = (direct, direct_facts)
    return result
