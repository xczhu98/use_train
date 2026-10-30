"""
D1: Apollo 超分辨率增强。

使用 Apollo-UNI 模型对低于目标采样率的单事件片段进行带宽扩展/质量增强。
生产默认的 subprocess_daemon 模式通常调用
third_party_scripts/Apollo/inference_apollo.py --daemon，入口可由配置覆盖；dry_run 不调用 Apollo daemon。

模式：
  - dry_run:   复制输入音频作为"增强后"音频（测试用）
  - subprocess_daemon: 每 GPU 一个长驻 daemon 进程，模型只加载一次（推荐生产环境）
  其它旧模式（inline / subprocess）已移除，启动时对非法 mode 直接 ValueError。
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import stat
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from ..core.checkpoint import CheckpointTracker
from ..core.manifest_io import (
    AudioRecord,
    read_manifest_with_progress,
)
from ..core.manifest_per_dataset import PerDatasetManifestWriter, init_checkpointed_output
from ..core.processor_base import Processor
from tools.shared.audio_io import (
    copy_or_convert_audio,
    load_audio_robust,
    make_unique_output_path,
    resample_audio,
    save_audio_atomically,
)
from .daemon_pool import BaseDaemonPool, DaemonPoolExhaustedError
from .loudness_normalizer import (
    LoudnessNormalizerProcessor,
    register_dataset_label_namespace,
)


logger = logging.getLogger(__name__)


_DaemonPoolExhaustedError = DaemonPoolExhaustedError


class _ApolloDaemonRetryableError(RuntimeError):
    """Apollo 请求响应不完整或不可信时触发的整批可重试异常。"""


class _ApolloDaemonPool(BaseDaemonPool):
    """D1 subprocess_daemon 模式下管理多 GPU 长驻 Apollo daemon 进程。

    协议与 third_party_scripts/Apollo/inference_apollo.py --daemon 对应。
    生命周期管理（启动/ping/重启/stop/收发/轮询）由 BaseDaemonPool 提供，
    本类只保留 Apollo 协议差异与 enhance 业务逻辑（含失败重试与熔断）。
    """

    # 连续 pool 级 batch 失败的上限：达到后视为基础设施故障，抛
    # _DaemonPoolExhaustedError 中止 stage，避免每个 batch 都重新拉起
    # 全部 daemon 无限重试
    _MAX_CONSECUTIVE_BATCH_FAILURES = 3

    def __init__(
        self,
        python_path: str,
        daemon_script_path: str,
        config_path: str,
        checkpoint: str,
        batch_size: int,
        target_sr: int,
        chunk_sec: float,
        overlap_sec: float,
        gpu_ids: List[int],
        num_gpus: int,
        timeout: int,
        startup_timeout: int = 600,
        stage_name: str = "D1",
    ):
        super().__init__(
            python_path=python_path,
            daemon_script_path=daemon_script_path,
            checkpoint=checkpoint,
            gpu_ids=gpu_ids,
            num_gpus=num_gpus,
            timeout=timeout,
            startup_timeout=startup_timeout,
            stage_name=stage_name,
        )
        self.config_path = config_path
        self.batch_size = max(1, int(batch_size))
        self.target_sr = target_sr
        self.chunk_sec = chunk_sec
        self.overlap_sec = overlap_sec
        self._output_audio_dir: Optional[str] = None
        # 连续 pool 级 batch 失败计数（成功即清零），用于熔断
        self._consecutive_batch_failures = 0

    def _build_cmd(self, gpu_id: int) -> List[str]:
        return [
            "--daemon",
            "--output_audio_dir", self._output_audio_dir,
            "--batch_size", str(self.batch_size),
            "--config_path", self.config_path,
            "--checkpoint", self.checkpoint,
            "--target_sr", str(self.target_sr),
            "--chunk_sec", str(self.chunk_sec),
            "--overlap_sec", str(self.overlap_sec),
            # CUDA_VISIBLE_DEVICES 已经把该子进程限制到唯一一张卡，
            # daemon 内部看到的设备号恒为 cuda:0，避免 cuda:{gpu_id} 越界。
            "--device", "cuda:0",
        ]

    def _ping_payload(self) -> Dict[str, Any]:
        return {"type": "ping"}

    def _check_pong(self, resp: Dict[str, Any]) -> bool:
        return resp.get("type") == "pong"

    def _stop_payload(self) -> Dict[str, Any]:
        return {"type": "stop"}

    def start(self, log_dir: str, output_audio_dir: str) -> None:
        """按需启动所有 GPU 对应的 daemon（每个 GPU 一个独立子进程）。"""
        # _build_cmd 依赖 _output_audio_dir，必须在 super().start 前赋值
        self._output_audio_dir = output_audio_dir
        super().start(log_dir)

    def enhance(self, items: List[Dict[str, Any]], batch_id: int) -> List[Dict[str, Any]]:
        """通过 round-robin 选出的 daemon 处理一个 batch，返回 results 列表。

        首次失败（含 batch_id 不匹配等异常响应）会把故障 daemon 原位重启
        （或移出轮询），随后换一个 daemon 重试一次，避免一个死 daemon
        拖垮整个 micro-batch。重试仍失败时原样上抛；无可用 daemon 或连续
        pool 级失败次数超限时抛 _DaemonPoolExhaustedError，由上层中止 stage。
        """
        last_err: Optional[Exception] = None
        for _attempt in range(2):
            gpu_idx, proc, lock = self._pick()
            with lock:
                try:
                    self._send(proc, {"type": "enhance", "batch_id": batch_id, "items": items})
                    resp = self._recv(proc)
                    # batch_id 校验必须在 try 内：响应错配同样视为 daemon 故障，
                    # 需要触发重启而不是直接上抛
                    if resp.get("type") != "result" or resp.get("batch_id") != batch_id:
                        raise RuntimeError(
                            f"{self.stage_name} Apollo daemon (gpu={gpu_idx}) 返回异常响应: {resp}"
                        )
                    self._consecutive_batch_failures = 0
                    return resp.get("results", [])
                except Exception as e:
                    # daemon 死亡/失联：原位重启替代进程；重启失败则移出轮询，
                    # 避免后续 batch 继续 round-robin 打到死进程。全部不可用时中止。
                    last_err = e
                    self._restart_daemon(gpu_idx)
                    if not self._daemon_procs:
                        raise _DaemonPoolExhaustedError(
                            f"{self.stage_name} 所有 Apollo daemon 均不可用，中止 stage"
                        ) from e
        # 重试一次仍失败：计入连续失败，超限即熔断，防止所有 daemon 都死掉时
        # 每个 batch 都无限重启重试
        self._consecutive_batch_failures += 1
        if self._consecutive_batch_failures >= self._MAX_CONSECUTIVE_BATCH_FAILURES:
            raise _DaemonPoolExhaustedError(
                f"{self.stage_name} Apollo daemon 连续 "
                f"{self._consecutive_batch_failures} 次 pool 级 batch 失败，中止 stage"
            ) from last_err
        raise last_err

    def enhance_parallel(
        self,
        items: List[Dict[str, Any]],
        log_dir: str,
        output_audio_dir: str,
        batch_index: int,
    ) -> List[Dict[str, Any]]:
        """多 GPU 并行增强：把 items 按 daemon 数切片，线程并发驱动各 daemon。

        单 daemon 或数据量不足一个 batch 时退化为单 daemon 串行。
        返回所有子批 results 的拼接（顺序按切片顺序）。
        """
        self.start(log_dir, output_audio_dir)
        n_daemons = len(self._daemon_procs)
        # 每个 daemon 至少分到一个满 batch 才值得并行
        n_shards = min(n_daemons, max(1, len(items) // max(1, self.batch_size)))
        if n_shards <= 1:
            return self.enhance(items, batch_index)

        shard_size = math.ceil(len(items) / n_shards)
        sub_batches = [
            items[i : i + shard_size]
            for i in range(0, len(items), shard_size)
        ]
        # batch_index*1000+i 区分并发子批，便于 daemon 侧日志与响应核对
        with ThreadPoolExecutor(max_workers=len(sub_batches)) as pool:
            futures = [
                pool.submit(self.enhance, sub, batch_index * 1000 + shard_id)
                for shard_id, sub in enumerate(sub_batches)
            ]
            results: List[Dict[str, Any]] = []
            for fut in futures:
                results.extend(fut.result())
        return results


class ApolloSuperResProcessor(Processor):
    """D1: Apollo 超分增强 Processor。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, stage_name="D1")
        self.mode = config.get("mode", "dry_run")
        # 运行模式收敛：只保留 dry_run / subprocess_daemon，非法值启动时 fail-fast
        if self.mode not in ("dry_run", "subprocess_daemon"):
            raise ValueError(
                f"D1 不支持的 mode: {self.mode!r}"
                f"（仅支持 dry_run / subprocess_daemon）"
            )
        self.config_path = config.get("config_path") or config.get("apollo_config")
        self.checkpoint = config.get("checkpoint") or config.get("apollo_checkpoint")
        self.script_path = config.get("script_path")
        self.python_path = config.get("python_path", "python")
        self.batch_size = config.get("batch_size", 1)
        self.target_sr = config.get("target_sr", 44100)
        # output_sr: D1 最终输出采样率；默认与 target_sr 一致，产生统一采样率的噪声库。
        # 显式设为 null 时，Apollo 处理过的样本用 target_sr，跳过的样本保留原始采样率。
        self.output_sr = config.get("output_sr", self.target_sr)
        self.output_format = config.get("output_format", "flac")
        self.chunk_sec = float(config.get("chunk_sec", 10))
        # overlap_sec: 相邻 chunk 的重叠秒数。
        raw_overlap = config.get("overlap_sec") if "overlap_sec" in config else config.get("overlap", 2)
        self.overlap_sec = float(raw_overlap)
        # 与 _prepare_chunks 同口径的范围校验前移到 init：坏配置（如
        # overlap_sec=0 的 fade 崩溃、负值/超界）必须在启动时 fail-fast，
        # 而不是每条音频处理时才报错、整批标 failed 无限重试
        if self.chunk_sec <= 0:
            raise ValueError(f"D1 chunk_sec 必须大于 0，当前: {self.chunk_sec}")
        if self.overlap_sec < 0 or self.overlap_sec >= self.chunk_sec:
            raise ValueError(
                f"D1 overlap_sec 必须满足 0 <= overlap_sec < chunk_sec，"
                f"当前: {self.overlap_sec} vs chunk_sec={self.chunk_sec}"
            )
        # output_sr 非空时必须为正整数：0/负数会让"低于 target_sr 才超分"的
        # 判定恒真全部跳过 Apollo，浮点/字符串到重采样链才 TypeError 误分类
        if self.output_sr is not None:
            try:
                if int(self.output_sr) <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError(
                    f"D1 output_sr 非法: {self.output_sr!r}，必须为正整数或 null"
                ) from None

        # checkpoint / resume
        self.enable_checkpoint = config.get("enable_checkpoint", True)
        self.checkpoint_path = config.get("checkpoint_path")
        # 批量保存 checkpoint，降低百万级文件下的 I/O 频率；默认 100 条保存一次
        self.checkpoint_save_interval = int(config.get("checkpoint_save_interval", 100))
        # manifest 批量追加大小，与 checkpoint_save_interval 保持一致可减少写放大
        self.manifest_write_batch_size = int(config.get("manifest_write_batch_size", 100))

        # 多卡支持（subprocess_daemon 模式）
        self.num_gpus = int(config.get("num_gpus", 1))
        self.gpu_ids = config.get("gpu_ids", [])

        # D1+D2 合并：在 Apollo 输出后立刻做响度归一化/质检，避免再写一次中间音频。
        post_normalize = config.get("post_normalize")
        self.post_normalize = bool(post_normalize)
        self._normalizer: Optional[LoudnessNormalizerProcessor] = None
        if self.post_normalize:
            norm_cfg = post_normalize if isinstance(post_normalize, dict) else {}
            self._normalizer = LoudnessNormalizerProcessor(norm_cfg)
            # orchestrator 合并 d2_cfg 时会把 num_workers（默认 min(8, cpu_count)，
            # 随机器变化）带进 post_normalize；它对合并归一化（D1 内单条调用）
            # 无意义，从本阶段指纹配置中剥离，与 D2 processor 的口径一致
            if isinstance(self.config.get("post_normalize"), dict):
                self.config = {
                    **self.config,
                    "post_normalize": {
                        k: v for k, v in self.config["post_normalize"].items() if k != "num_workers"
                    },
                }
        # 基础设施失败计数（status=failed 且未写 checkpoint 的记录数），
        # 供 orchestrator 判定本轮是否有可重试失败
        self._retryable_failures = 0

        # subprocess_daemon 长驻模式：daemon 生命周期管理拆到独立 helper，
        # 与 C2 AudioTag 的 _AudioTagDaemonPool 同构；daemon 脚本即
        # inference_apollo.py（内置 --daemon 模式），可用 daemon_script_path 覆盖
        self.daemon_script_path = config.get("daemon_script_path") or self.script_path
        self._daemon_pool: Optional[_ApolloDaemonPool] = None
        self._daemon_output_root: Optional[str] = None
        if self.mode == "subprocess_daemon":
            daemon_timeout = int(config.get("daemon_timeout", 3600))
            # ping 握手（含模型加载）用独立的较短启动超时，不用 enhance 的长超时
            daemon_startup_timeout = int(config.get("daemon_startup_timeout", 600))
            self._daemon_pool = _ApolloDaemonPool(
                python_path=self.python_path,
                daemon_script_path=self.daemon_script_path,
                config_path=self.config_path,
                checkpoint=self.checkpoint,
                batch_size=self.batch_size,
                target_sr=self.target_sr,
                chunk_sec=self.chunk_sec,
                overlap_sec=self.overlap_sec,
                gpu_ids=self.gpu_ids,
                num_gpus=self.num_gpus,
                timeout=daemon_timeout,
                startup_timeout=daemon_startup_timeout,
                stage_name=self.stage_name,
            )

    def has_retryable_failures(self) -> bool:
        """是否有未写 checkpoint、可在下次运行重试的基础设施失败。"""
        return self._retryable_failures > 0

    def _effective_output_sr(self, fallback_sr: Optional[int] = None) -> int:
        """返回最终输出采样率。

        ``output_sr`` 显式配置时优先使用；否则使用 ``fallback_sr``；
        两者都未提供时回退到 ``target_sr``。
        """
        if self.output_sr is not None:
            return self.output_sr
        return fallback_sr if fallback_sr is not None else self.target_sr

    def _normalize_record(self, rec: AudioRecord) -> None:
        """归一化 Apollo 产物；dataset_label 终态也保留 ordinal 预期路径。"""
        if self._normalizer is None:
            return
        audio_path = rec.audio["path"]
        input_fingerprint = self._normalizer._record_input_fingerprint(rec, audio_path)
        try:
            # 用 load_audio_robust 而非裸 sf.read，与独立 D2 的质检口径一致
            # （NaN/Inf、峰值越界、过静音三项硬检查）
            audio, sr = load_audio_robust(audio_path)
            # 读取期间若 Apollo 产物发生替换或修改，放弃本次结果并交给 resume 重试。
            if self._normalizer._input_fingerprint(audio_path) != input_fingerprint:
                raise OSError(f"input changed while loading: {audio_path}")
            audio, out_sr, info = self._normalizer._normalize(audio, sr)
            if "rejected_reason" in info:
                # dataset_label 的 rejected 记录也必须保留按当前 ordinal 计算的
                # 最终路径；实际文件仍不写入，resume 才能严格比较旧路径。
                if self._normalizer.output_layout == "dataset_label":
                    rec.audio["path"] = self._normalizer._output_path_for(
                        out_sr, rec, audio_path
                    )
                # 与独立 D2 同口径记录 Apollo 中间输入路径和指纹；源记录身份
                # 另由 dataset_label provenance 保存，恢复时始终校验原始输入。
                rec.mark_stage(
                    "D2", "rejected",
                    extra={
                        **info,
                        "input_path": audio_path,
                        "input_fingerprint": input_fingerprint,
                    },
                )
                rec.rejected_reason = f"d2_{info['rejected_reason']}"
                # 删除 Apollo 中间产物，避免垃圾文件滞留（独立 D2 模式在写文件前就拒绝）
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
                return
            # 最终路径按 D2 output_layout 计算（与独立 D2 同套规则）；
            # dataset_label 必须通过统一 ordinal 路径函数，其他布局保持旧行为。
            if self._normalizer.output_layout == "dataset_label":
                output_path = self._normalizer._output_path_for(out_sr, rec, audio_path)
                if os.path.abspath(audio_path) != os.path.abspath(output_path):
                    try:
                        os.remove(audio_path)
                    except OSError:
                        pass
            else:
                out_dir = self._normalizer._output_dir_for(out_sr, rec)
                expected_ext = f".{self._normalizer.output_format}"
                if (
                    os.path.dirname(os.path.abspath(audio_path)) == os.path.abspath(out_dir)
                    and audio_path.lower().endswith(expected_ext.lower())
                ):
                    output_path = audio_path
                else:
                    output_path = make_unique_output_path(out_dir, audio_path, ext=expected_ext)
                    try:
                        os.remove(audio_path)
                    except OSError:
                        pass
            save_audio_atomically(audio, out_sr, output_path)
            rec.audio["path"] = output_path
            rec.audio["sr"] = out_sr
            rec.audio["duration"] = info["duration"]
            rec.audio["rms"] = info["normalized_rms"]
            rec.audio["peak"] = info["normalized_peak"]
            # 与独立 D2 同口径记录输入指纹：resume 恢复时校验输入未变才复用，
            # 上游重算（路径不变、内容变化）时自动重处理
            rec.mark_stage(
                "D2", "passed",
                extra={
                    **info,
                    "input_path": audio_path,
                    "input_fingerprint": input_fingerprint,
                },
            )
        except OSError as e:
            # 基础设施失败（磁盘满/挂载掉线等）：与独立 D2 同口径标
            # failed+retryable，并把 D1 一并标 failed（覆盖已打的 passed），
            # 阻止调用方写 checkpoint——否则 resume 会永久恢复 rejected/failed
            logger.warning("D1+D2 合并：归一化 %s 失败（基础设施，可重试）: %s", audio_path, e)
            rec.mark_stage("D2", "failed", extra={"reason": str(e), "retryable": True})
            rec.mark_stage("D1", "failed", extra={"reason": "d2_normalize_infra", "retryable": True})
            self._retryable_failures += 1
        except Exception as e:
            # 内容级失败是终态 rejected：dataset_label 必须落回当前 ordinal 的
            # 预期路径，并清理 Apollo 中间产物；该路径即使不存在也可恢复。
            logger.warning("D1+D2 合并：归一化 %s 失败（内容级，终态）: %s", audio_path, e)
            if self._normalizer.output_layout == "dataset_label":
                rec.audio["path"] = self._normalizer._output_path_for(
                    rec.audio.get("sr") or self._effective_output_sr(rec.audio.get("sr")),
                    rec,
                    audio_path,
                )
            rec.mark_stage(
                "D2", "rejected",
                extra={
                    "reason": str(e),
                    "input_path": audio_path,
                    "input_fingerprint": input_fingerprint,
                },
            )
            rec.rejected_reason = "d2_exception"
            if self._normalizer.output_layout == "dataset_label":
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

    def _run_dry_run(
        self, input_path: str, output_path: str
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """dry_run 模拟 Apollo 输出。

        返回 (写入采样率, 归一化 info)。post_normalize 开启时归一化在这里一次完成，
        info 供调用方直接打 D2 标记，避免 _normalize_record 二次读写；
        时长不合格时不写文件，直接返回带 rejected_reason 的 info。
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 先统一重采样到 Apollo 输出采样率（output_sr 或 target_sr）
        audio, sr = sf.read(input_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        target_sr = self._effective_output_sr(self.target_sr)
        if sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
            sr = target_sr

        if self.post_normalize:
            audio, out_sr, info = self._normalizer._normalize(audio, sr)
            if "rejected_reason" in info:
                return out_sr, info
            save_audio_atomically(audio, out_sr, output_path)
            return out_sr, info
        if not np.isfinite(audio).all():
            # NaN/Inf 源音频不得写盘；抛异常由调用方按单条失败标记 rejected
            raise ValueError(f"音频包含 NaN/Inf: {input_path}")
        # 原子写：崩溃不留下半成品文件，避免 resume 探测到残缺产物
        save_audio_atomically(audio, sr, output_path)
        return sr, None

    def _run_subprocess_daemon(
        self,
        records: List[AudioRecord],
        output_dir: str,
        output_audio_dir: str,
        batch_index: int,
    ) -> Dict[str, str]:
        """通过 daemon 增强一个 micro-batch，并只发布本次请求拥有的产物。

        每个 symlink 输入 stem 同时绑定 request UUID 与 source token；响应必须
        回显尚未 claim 的输入，并返回该 stem 对应的 daemon 输出 basename。
        输出还必须是本次请求新建的、无 symlink regular file。只有全部校验
        通过后才写入按 ``record_id`` 索引的 path_map，所有清理都只针对本次
        请求登记的路径。同一 batch 的重复 source path 直接 fail-closed 拒绝。
        """
        if self._daemon_pool is None:
            raise RuntimeError("D1 subprocess_daemon 模式未初始化 daemon pool")

        final_output_root = os.path.abspath(output_audio_dir)
        output_root = os.path.abspath(self._daemon_output_root or output_audio_dir)
        request_id = uuid.uuid4().hex
        tmp_input_dir = os.path.join(
            output_root, f".d1_daemon_{batch_index:04d}_{request_id}_inputs"
        )
        os.makedirs(tmp_input_dir, mode=0o700, exist_ok=False)

        def _snapshot_files(root: str) -> set[str]:
            paths: set[str] = set()
            for current_root, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = [
                    name for name in dirs
                    if not os.path.islink(os.path.join(current_root, name))
                ]
                paths.update(
                    os.path.abspath(os.path.join(current_root, name)) for name in files
                )
            return paths

        preexisting_output_paths = _snapshot_files(output_root)
        preexisting_final_paths = _snapshot_files(final_output_root)

        def _is_owned_regular_file(
            value: Any, root: str, preexisting_paths: set[str]
        ) -> Optional[str]:
            if not isinstance(value, str) or not value:
                return None
            candidate = os.path.abspath(value)
            try:
                if os.path.commonpath((root, candidate)) != root:
                    return None
            except ValueError:
                return None
            if candidate in preexisting_paths or os.path.realpath(candidate) != candidate:
                return None
            try:
                mode = os.lstat(candidate).st_mode
            except OSError:
                return None
            if not stat.S_ISREG(mode):
                return None
            return candidate

        def _matches_daemon_basename(path: str, input_stem: str) -> bool:
            """核对 Apollo 输出文件名。"""
            output_stem, output_ext = os.path.splitext(os.path.basename(path))
            if output_ext != ".wav":
                return False
            if output_stem in (input_stem, f"{input_stem}_enhanced"):
                return True
            prefix = f"{input_stem}_"
            suffix = output_stem[len(prefix):] if output_stem.startswith(prefix) else ""
            return suffix.isdigit() and int(suffix) > 0 and str(int(suffix)) == suffix

        pending_claims: Dict[str, Tuple[str, str, str]] = {}
        hashed_inputs: List[Dict[str, str]] = []
        owned_intermediates: set[str] = set()
        created_daemon_outputs: Dict[str, Tuple[int, int]] = {}
        created_final_targets: Dict[str, Tuple[int, int]] = {}
        path_map: Dict[str, str] = {}

        def _register_created_daemon_output(path: Any) -> None:
            """登记本 request 命名空间内新建的 daemon regular file。"""
            if not isinstance(path, str) or not path:
                return
            candidate = os.path.abspath(path)
            if not os.path.basename(candidate).startswith(f"{request_id}_"):
                return
            try:
                if os.path.commonpath((output_root, candidate)) != output_root:
                    return
                file_stat = os.lstat(candidate)
            except (OSError, ValueError):
                return
            if (
                candidate in preexisting_output_paths
                or not stat.S_ISREG(file_stat.st_mode)
                or os.path.realpath(candidate) != candidate
            ):
                return
            created_daemon_outputs.setdefault(
                candidate, (file_stat.st_dev, file_stat.st_ino)
            )

        def _register_created_target(path: str) -> None:
            """登记本 request 创建的 regular file 及 inode，供失败回收。"""
            candidate = os.path.abspath(path)
            if candidate in preexisting_final_paths:
                return
            try:
                if os.path.commonpath((final_output_root, candidate)) != final_output_root:
                    return
                file_stat = os.lstat(candidate)
            except (OSError, ValueError):
                return
            if not stat.S_ISREG(file_stat.st_mode) or os.path.realpath(candidate) != candidate:
                return
            created_final_targets.setdefault(
                candidate, (file_stat.st_dev, file_stat.st_ino)
            )

        def _cleanup_created_targets() -> None:
            """仅回收仍保持登记 inode 的本 request 目标。"""
            for path, identity in {
                **created_daemon_outputs,
                **created_final_targets,
            }.items():
                try:
                    file_stat = os.lstat(path)
                    if (
                        stat.S_ISREG(file_stat.st_mode)
                        and (file_stat.st_dev, file_stat.st_ino) == identity
                    ):
                        os.remove(path)
                except OSError as exc:
                    logger.warning("D1 回收本次目标失败 %s: %s", path, exc)

        request_succeeded = False
        try:
            seen_sources: set[str] = set()
            record_ids = [rec.record_id for rec in records]
            if len(set(record_ids)) != len(record_ids):
                raise _ApolloDaemonRetryableError(
                    f"D1 daemon batch 内 record_id 重复: {record_ids}"
                )
            expected_record_ids = set(record_ids)
            for index, rec in enumerate(records):
                src = rec.audio["path"]
                source_identity = os.path.abspath(src)
                if source_identity in seen_sources:
                    raise _ApolloDaemonRetryableError(
                        f"D1 daemon 拒绝 batch 内重复 source path: {src}"
                    )
                seen_sources.add(source_identity)
                source_token = hashlib.sha256(
                    f"{source_identity}\\0{index}".encode("utf-8")
                ).hexdigest()[:16]
                input_stem = f"{request_id}_{source_token}"
                ext = os.path.splitext(src)[1] or ".wav"
                hashed_path = os.path.join(tmp_input_dir, f"{input_stem}{ext}")
                # 目录是本轮新建且不存在残留项；不复用任何旧 symlink。
                os.symlink(source_identity, hashed_path)
                pending_claims[os.path.abspath(hashed_path)] = (
                    rec.record_id, src, input_stem
                )
                hashed_inputs.append({"audio_path": hashed_path})

            results = self._daemon_pool.enhance_parallel(
                hashed_inputs, output_dir, output_root, batch_index
            )
            if not isinstance(results, list):
                raise _ApolloDaemonRetryableError(
                    f"D1 daemon 返回非法 results 类型: {type(results).__name__}"
                )
            for item in results:
                if not isinstance(item, dict):
                    raise _ApolloDaemonRetryableError(
                        f"D1 daemon 返回非法 result item: {item!r}"
                    )
                response_input = item.get("audio_path")
                if not isinstance(response_input, str):
                    raise _ApolloDaemonRetryableError(
                        f"D1 daemon result 缺少合法 audio_path: {item!r}"
                    )
                _register_created_daemon_output(item.get("enhanced_audio_path"))
                claim = pending_claims.pop(response_input, None)
                if claim is None:
                    raise _ApolloDaemonRetryableError(
                        f"D1 daemon 响应输入未登记或重复 claim: {response_input!r}"
                    )
                record_id, src_path, expected_stem = claim
                enhanced_path = _is_owned_regular_file(
                    item.get("enhanced_audio_path"),
                    output_root,
                    preexisting_output_paths,
                )
                if (
                    enhanced_path is None
                    or os.path.dirname(enhanced_path) != output_root
                    or not _matches_daemon_basename(enhanced_path, expected_stem)
                ):
                    raise _ApolloDaemonRetryableError(
                        f"D1 daemon 输出 basename/归属/文件校验失败: "
                        f"{src_path} ({item.get('error')})"
                    )
                if enhanced_path in owned_intermediates:
                    raise _ApolloDaemonRetryableError(
                        f"D1 daemon 输出重复 claim: {enhanced_path}"
                    )
                owned_intermediates.add(enhanced_path)
                target_path = make_unique_output_path(
                    final_output_root, src_path, ext=f".{self.output_format}"
                )
                if os.path.abspath(enhanced_path) == os.path.abspath(target_path):
                    raise _ApolloDaemonRetryableError(
                        f"D1 daemon 输出不得直接复用确定性目标: {enhanced_path}"
                    )
                if os.path.lexists(target_path):
                    raise _ApolloDaemonRetryableError(
                        f"D1 拒绝覆盖既有确定性目标: {target_path}"
                    )
                # 复制失败时登记可能已原子发布的目标；批失败时由 request 级
                # inode 校验统一回收，成功目标则保留。
                try:
                    copy_or_convert_audio(
                        enhanced_path, target_path, target_sr=self.output_sr, max_peak=1.2
                    )
                except Exception as exc:
                    _register_created_target(target_path)
                    raise _ApolloDaemonRetryableError(
                        f"D1 daemon 输出转换失败: {target_path}: {exc}"
                    ) from exc
                target_file = _is_owned_regular_file(
                    target_path, final_output_root, preexisting_final_paths
                )
                if target_file is None:
                    _register_created_target(target_path)
                    raise _ApolloDaemonRetryableError(
                        f"D1 daemon 转换目标不是本阶段 regular file: {target_path}"
                    )
                _register_created_target(target_path)
                path_map[record_id] = target_path
            if pending_claims:
                raise _ApolloDaemonRetryableError(
                    f"D1 daemon partial response，未 claim 输入: {sorted(pending_claims)}"
                )
            if set(path_map) != expected_record_ids:
                raise _ApolloDaemonRetryableError(
                    f"D1 daemon path_map key 不完整: expected={expected_record_ids}, "
                    f"actual={set(path_map)}"
                )
            request_succeeded = True
        finally:
            if not request_succeeded:
                # daemon 可能创建了结果但未在 response 中回显；按 request UUID
                # 扫描并登记其 direct output，仍以 inode/regular-file 校验安全回收。
                for candidate in _snapshot_files(output_root):
                    _register_created_daemon_output(candidate)
                _cleanup_created_targets()
            for intermediate in owned_intermediates:
                try:
                    os.remove(intermediate)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning("D1 删除本次中间产物失败 %s: %s", intermediate, exc)
            # 仅删除当前 request 自己新建的输入树，绝不触碰其他批次路径。
            shutil.rmtree(tmp_input_dir, ignore_errors=True)
        return path_map

    def _validate_model_paths(self) -> None:
        if self.mode == "dry_run":
            return
        missing = []
        if not self.config_path or not os.path.isfile(self.config_path):
            missing.append(f"Apollo config: {self.config_path}")
        if not self.checkpoint or not os.path.isfile(self.checkpoint):
            missing.append(f"Apollo checkpoint: {self.checkpoint}")
        if self.mode == "subprocess_daemon" and (
            not self.daemon_script_path or not os.path.isfile(self.daemon_script_path)
        ):
            missing.append(f"Apollo daemon script: {self.daemon_script_path}")
        if missing:
            raise FileNotFoundError(
                "D1 模型文件缺失，请运行以下命令下载：\n"
                "  python tools/noise_filter_pipeline/scripts/download_hive_models.py\n"
                "缺失文件：\n" + "\n".join(f"  - {m}" for m in missing)
            )

    @staticmethod
    def _is_readable_audio(path: Optional[str]) -> bool:
        """校验音频产物存在且头信息可解析。

        resume 恢复 checkpoint 记录时使用：仅 ``os.path.isfile`` 会把
        崩溃中途写坏的文件当作合法 passed 产物恢复，这里用 ``sf.info``
        做一次轻量可读性探测（只读头部，不加载音频数据）。
        """
        if not path:
            return False
        try:
            file_stat = os.lstat(path)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size <= 0:
                return False
            info = sf.info(path)
        except Exception:
            return False
        return info.frames > 0 and info.samplerate > 0

    def _update_record_after_enhance(self, rec: AudioRecord, new_path: str) -> None:
        """Apollo 增强后更新记录路径、采样率，并可选做 D2 归一化。"""
        if new_path and os.path.isfile(new_path):
            rec.audio["path"] = new_path
            rec.audio["sr"] = self._effective_output_sr(self.target_sr)
            rec.mark_stage("D1", "passed")
            if self.post_normalize:
                self._normalize_record(rec)
        else:
            rec.mark_stage("D1", "rejected", extra={"reason": "apollo_missing_output"})
            rec.rejected_reason = "d1_apollo_missing_output"

    def _skip_apollo(self, rec: AudioRecord, output_audio_dir: str) -> None:
        """采样率已达 target_sr 的样本跳过 Apollo，直接复制并按目标格式/采样率保存。

        默认输出到 output_sr（与 target_sr 一致），产生统一采样率的噪声库；
        当 output_sr 显式设为 null 时保留原始采样率。
        """
        input_path = rec.audio["path"]
        original_sr = rec.audio.get("sr")
        # post_normalize 时复制目标即 D2 最终位置：按 D2 output_layout 布局，
        # 避免先写平铺中间产物再搬运；最终采样率 = D2 output_sr（未配置则
        # 与 copy_or_convert_audio 的目标采样率一致）
        if self.post_normalize and self._normalizer is not None:
            final_sr = self._normalizer.output_sr or self._effective_output_sr(original_sr)
            out_dir = self._normalizer._output_dir_for(final_sr, rec)
            out_ext = self._normalizer.output_format
        else:
            out_dir = output_audio_dir
            out_ext = self.output_format
        if self.post_normalize and self._normalizer is not None:
            output_path = self._normalizer._output_path_for(
                self._normalizer.output_sr or self._effective_output_sr(original_sr),
                rec,
                input_path,
            )
        else:
            output_path = make_unique_output_path(
                out_dir, input_path, ext=f".{out_ext}"
            )
        try:
            target_sr = self._effective_output_sr(original_sr)
            # max_peak=1.2：MP3 解码峰值可达 1.2+，默认值会误杀正常文件
            copy_or_convert_audio(input_path, output_path, target_sr=target_sr, max_peak=1.2)
            rec.audio["path"] = output_path
            rec.audio["sr"] = target_sr
            rec.mark_stage(
                "D1",
                "passed",
                extra={
                    "skipped": True,
                    "reason": "sample_rate_not_below_target",
                    "original_sr": original_sr,
                },
            )
            if self.post_normalize:
                self._normalize_record(rec)
        except OSError as e:
            # 基础设施失败（磁盘满/挂载掉线/IO 等）：标 failed（非终态 rejected），
            # 调用方不写 checkpoint，resume 时自动重试
            logger.warning("D1 跳过 Apollo 复制 %s 失败（基础设施，可重试）: %s", input_path, e)
            rec.mark_stage("D1", "failed", extra={"reason": str(e), "retryable": True})
            self._retryable_failures += 1
        except Exception as e:
            # 内容级失败（源文件损坏/解码异常等）：终态 rejected 写 checkpoint，
            # 与 D2 的异常分类口径一致，避免永久损坏文件每轮中止 pipeline
            logger.warning("D1 跳过 Apollo 复制 %s 失败（内容级，终态）: %s", input_path, e)
            rec.mark_stage("D1", "rejected", extra={"reason": str(e)})
            rec.rejected_reason = "d1_copy_exception"

    def _flush_pending(
        self,
        pending: List[AudioRecord],
        output_dir: str,
        output_audio_dir: str,
        manifest_writer: PerDatasetManifestWriter,
        ckpt: CheckpointTracker,
        batch_index: List[int],
    ) -> None:
        """对 micro-batch 执行 Apollo 增强并批量写出结果。"""
        if not pending:
            return

        if self.mode == "dry_run":
            for rec in pending:
                try:
                    input_path = rec.audio["path"]
                    # post_normalize 时 D2 的 output_format 是最终格式，输出扩展名应与之匹配，
                    # 且按 D2 output_layout 布局直接写最终位置（无平铺中间产物）；
                    # 最终采样率 = D2 output_sr（未配置则 D1 输出采样率）
                    out_ext = self.output_format
                    out_dir = output_audio_dir
                    if self.post_normalize and self._normalizer is not None:
                        out_ext = self._normalizer.output_format
                        final_sr = self._normalizer.output_sr or self._effective_output_sr(self.target_sr)
                        out_dir = self._normalizer._output_dir_for(final_sr, rec)
                    if self.post_normalize and self._normalizer is not None:
                        output_path = self._normalizer._output_path_for(
                            self._normalizer.output_sr or self._effective_output_sr(self.target_sr),
                            rec,
                            input_path,
                        )
                    else:
                        output_path = make_unique_output_path(
                            out_dir, input_path, ext=f".{out_ext}"
                        )
                    out_sr, norm_info = self._run_dry_run(input_path, output_path)
                    rec.mark_stage("D1", "passed", extra={"dry_run": True})
                    if norm_info is not None:
                        # post_normalize 已在 _run_dry_run 内一次完成，直接打 D2 标记
                        norm_extra = {
                            **norm_info,
                            "input_path": input_path,
                            "input_fingerprint": self._normalizer._record_input_fingerprint(
                                rec, input_path
                            ) if self._normalizer is not None else None,
                        }
                        if "rejected_reason" in norm_info:
                            if self._normalizer.output_layout == "dataset_label":
                                rec.audio["path"] = output_path
                            rec.mark_stage("D2", "rejected", extra=norm_extra)
                            rec.rejected_reason = f"d2_{norm_info['rejected_reason']}"
                        else:
                            rec.audio["path"] = output_path
                            rec.audio["sr"] = out_sr
                            rec.audio["duration"] = norm_info["duration"]
                            rec.audio["rms"] = norm_info["normalized_rms"]
                            rec.audio["peak"] = norm_info["normalized_peak"]
                            rec.mark_stage("D2", "passed", extra=norm_extra)
                    else:
                        rec.audio["path"] = output_path
                        rec.audio["sr"] = out_sr
                    manifest_writer.append(rec)
                    ckpt.mark_done(rec.record_id)
                except OSError as exc:
                    # 磁盘满、权限或挂载故障属于基础设施失败：不写 checkpoint，
                    # 由 orchestrator 中止 stage，重跑时自动重试。
                    logger.warning("D1 dry_run 处理 %s 失败（基础设施，可重试）: %s", rec.audio.get("path"), exc)
                    rec.mark_stage(
                        "D1", "failed",
                        extra={"reason": "dry_run_infra", "error": str(exc), "retryable": True},
                    )
                    manifest_writer.append(rec)
                    self._retryable_failures += 1
                except Exception as exc:  # noqa: BLE001
                    # 损坏音频等内容级失败是终态：写 checkpoint，避免每轮重复处理。
                    logger.warning("D1 dry_run 处理 %s 失败（内容级，终态）: %s", rec.audio.get("path"), exc)
                    rec.mark_stage("D1", "rejected", extra={"reason": "dry_run_failed", "error": str(exc)})
                    rec.rejected_reason = "d1_dry_run_failed"
                    manifest_writer.append(rec)
                    ckpt.mark_done(rec.record_id)
        elif self.mode == "subprocess_daemon":
            try:
                path_map = self._run_subprocess_daemon(
                    pending, output_dir, output_audio_dir, batch_index[0]
                )
            except Exception as exc:
                logger.error("D1 Apollo daemon batch %d 失败: %s", batch_index[0], exc)
                for rec in pending:
                    rec.mark_stage(
                        "D1", "failed",
                        extra={"reason": "daemon_failed", "error": str(exc), "retryable": True},
                    )
                    manifest_writer.append(rec)
                    self._retryable_failures += 1
                pending.clear()
                batch_index[0] += 1
                # pool exhausted 仍需中止 stage，但整批失败记录已经落 manifest；
                # 由于没有 mark_done，checkpoint 不会恢复这批记录。
                if isinstance(exc, _DaemonPoolExhaustedError):
                    raise
                return
            for rec in pending:
                new_path = path_map.get(rec.record_id)
                self._update_record_after_enhance(rec, new_path)
                manifest_writer.append(rec)
                # 合并 D2 的基础设施失败会把 D1/D2 标为 failed；这类记录
                # 必须不写 checkpoint，重跑才会重新执行而非恢复失败产物。
                if rec.stages.get("D1", {}).get("status") != "failed":
                    ckpt.mark_done(rec.record_id)
        else:
            # init 已做 mode 白名单校验，此处仅防御性兜底
            raise ValueError(f"D1 不支持的 mode: {self.mode}")

        pending.clear()
        batch_index[0] += 1

    def close(self) -> None:
        """统一资源收口（Processor.close 契约）：回收 daemon 进程池。"""
        if self._daemon_pool is not None:
            self._daemon_pool.stop()
        if self._daemon_output_root is not None:
            shutil.rmtree(self._daemon_output_root, ignore_errors=True)
            self._daemon_output_root = None

    def process(
        self,
        input_manifest: str,
        output_dir: str,
        fingerprint_input: Optional[str] = None,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_manifest = os.path.join(output_dir, "d1_manifest.jsonl")
        output_audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(output_audio_dir, exist_ok=True)
        if self.mode == "subprocess_daemon" and self._daemon_output_root is None:
            self._daemon_output_root = os.path.join(
                output_audio_dir, f".d1_daemon_outputs_{uuid.uuid4().hex}"
            )
            os.makedirs(self._daemon_output_root, mode=0o700, exist_ok=False)
        # D1+D2 合并模式：最终音频根目录即 D1 audio 目录，复用 D2 的
        # output_layout 与 dataset ordinal 稳定数字命名规则。
        if self._normalizer is not None:
            self._normalizer._prepare_output_audio(output_dir)

        manifest_writer, ckpt, prev_records = init_checkpointed_output(
            output_manifest, output_dir, self.config, input_manifest,
            enable_checkpoint=self.enable_checkpoint,
            checkpoint_path=self.checkpoint_path,
            save_interval=self.checkpoint_save_interval,
            manifest_batch_size=self.manifest_write_batch_size,
            stage_label="Stage D1",
            dry_run=self.mode == "dry_run",
            fingerprint_input=fingerprint_input,
        )

        if self.mode != "dry_run":
            self._validate_model_paths()

        total = 0
        pending: List[AudioRecord] = []
        batch_index = [0]
        dataset_ordinals: Dict[str, int] = {}
        dataset_namespaces: Dict[str, str] = {}
        category_namespaces: Dict[str, Dict[str, str]] = {}

        try:
            with manifest_writer:
                for rec in read_manifest_with_progress(input_manifest, self.stage_name):
                    total += 1
                    ordinal: Optional[int] = None
                    if (
                        self.post_normalize
                        and self._normalizer is not None
                        and self._normalizer.output_layout == "dataset_label"
                    ):
                        register_dataset_label_namespace(
                            rec, dataset_namespaces, category_namespaces
                        )
                        dataset_key = rec.dataset_id or "unknown"
                        ordinal = dataset_ordinals.get(dataset_key, 0)
                        dataset_ordinals[dataset_key] = ordinal + 1
                        # 必须在 should_skip/checkpoint 前写入并校验 provenance；
                        # category 变化不影响同一 dataset 的全局输入序号。
                        self._normalizer._set_dataset_label_provenance(rec, ordinal)

                    if self.should_skip(rec):
                        manifest_writer.append(rec)
                        continue
                    if ckpt.is_done(rec.record_id):
                        prev = prev_records.get(rec.record_id)
                        if (
                            prev is not None
                            and self.post_normalize
                            and self._normalizer is not None
                            and self._normalizer.output_layout == "dataset_label"
                        ):
                            # dataset_label 恢复必须显式使用本次输入顺序的 ordinal，
                            # 严格拒绝旧版本、错 ordinal、错 shard_size 或错输出路径。
                            if (
                                ordinal is None
                                or not self._normalizer._can_restore_checkpoint_record(
                                    rec,
                                    prev,
                                    ordinal,
                                )
                            ):
                                prev = None
                        elif prev is not None and self.post_normalize:
                            d2_stage = prev.stages.get("D2", {})
                            if d2_stage.get("status") in ("passed", "rejected"):
                                fp = d2_stage.get("input_fingerprint")
                                src = d2_stage.get("input_path")
                                if (
                                    fp is not None
                                    and src is not None
                                    and os.path.isfile(src)
                                    and fp != self._normalizer._input_fingerprint(src)
                                ):
                                    prev = None
                        # D2 rejected 是内容级终态；其 Apollo 中间产物已被
                        # _normalize_record 按预期删除，产物可读性校验必失败，
                        # 但通过严格 provenance/指纹校验后可直接恢复。
                        if prev is not None and prev.stages.get("D2", {}).get("status") == "rejected":
                            manifest_writer.append(prev)
                            continue
                        # 恢复前校验音频产物仍在盘上且可读；manifest 残留但文件被清理
                        # 或上次崩溃写坏时，回退为重新处理，避免恢复出坏文件的 passed 记录。
                        if prev is not None and self._is_readable_audio(prev.audio.get("path")):
                            manifest_writer.append(prev)
                            continue
                        # 上次输出缺失（如被清理）：回退为重新处理，避免丢失结果。

                    # 采样率已达 target_sr 的音频无需再做频带扩展，直接跳过 Apollo
                    # sr 可能为 None（上游缺元信息），按 0 处理走 Apollo 流程
                    if (rec.audio.get("sr") or 0) >= self.target_sr:
                        self._skip_apollo(rec, output_audio_dir)
                        manifest_writer.append(rec)
                        # 复制/转码失败（status=failed）不写 checkpoint，resume 自动重试
                        if rec.stages.get("D1", {}).get("status") != "failed":
                            ckpt.mark_done(rec.record_id)
                        continue

                    pending.append(rec)
                    effective_gpus = max(1, len(self.gpu_ids) if self.gpu_ids else self.num_gpus)
                    chunk_size = self.batch_size * effective_gpus if effective_gpus > 1 else self.batch_size
                    if len(pending) >= chunk_size:
                        self._flush_pending(
                            pending, output_dir, output_audio_dir, manifest_writer, ckpt, batch_index
                        )

                self._flush_pending(
                    pending, output_dir, output_audio_dir, manifest_writer, ckpt, batch_index
                )
                ckpt.save(force=True)
        finally:
            # 正常结束或异常退出都要回收 daemon，避免残留进程占 GPU
            self.close()

        logger.info("D1 处理完成：%d 条记录，输出 %s", total, output_manifest)
        if self.has_retryable_failures():
            logger.warning(
                "%s 有 %d 条基础设施失败记录（status=failed，未写 checkpoint），重跑可自动重试。",
                self.stage_name, self._retryable_failures,
            )
        return output_dir
