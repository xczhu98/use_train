"""
C2: AudioTag AudioSet 事实提取（可选）。

使用 icefall AudioTag 模型输出固定 top-5 证据（``audioset_top5``，恰 5 项，
prob 降序 + index 升序，index/mid/label 与 canonical label space 一致），
排序在表单层基于完整 527 维独立 sigmoid 分数内部完成；本层不加载或执行
任何准入 policy，也不再校验或持久化 527 分数数组。

由于 icefall 依赖与主环境通常隔离，本 Processor 支持两种模式：
  - dry_run:           测试/回归用，返回确定性非零分数 top-5
  - subprocess_daemon: 长驻 daemon 模式（推荐生产环境），模型只加载一次

注：Stage C 现在只保留 AudioTag 作为 AudioSet 分类步骤，默认位于 C2；
代码中统一使用 self.stage_name，可通过 stage_name 配置兼容旧编号。
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ..core.checkpoint import CheckpointTracker
from ..core.execution_contract import write_stage_c_processor_mode
from ..core.exceptions import InfraAbortError
from ..core.manifest_io import AudioRecord, read_manifest_strict
from ..core.manifest_per_dataset import PerDatasetManifestWriter, init_checkpointed_output
from ..core.processor_base import Processor
from .daemon_pool import BaseDaemonPool
from third_party_scripts.icefall_audiotaq.audiotag_result import (
    AUDIOTAG_TOP_K,
    CLASSIFIER_LABEL_COUNT,
    EVIDENCE_CONTRACT_VERSION,
    SCORE_CONTRACT,
    format_audiotag_results,
    load_label_metadata,
)

logger = logging.getLogger(__name__)

C2_EVIDENCE_CONTRACT_VERSION = EVIDENCE_CONTRACT_VERSION
C2_SCORE_CONTRACT = SCORE_CONTRACT
C2_CLASS_COUNT = CLASSIFIER_LABEL_COUNT
C2_TOP_K = AUDIOTAG_TOP_K
C2_TOP5_KEY = "audioset_top5"
_IDENTITY_SAMPLE_LIMIT = 20
_IDENTITY_TEXT_LIMIT = 160
_ERROR_MESSAGE_LIMIT = 512


def _bounded_error_text(value: Any, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class DaemonInferenceError(RuntimeError):
    """本地 daemon 推理分类错误；禁止降级为普通记录失败。"""

    def __init__(self, error_kind, error_code, message, record_ids=None):
        safe_kind = _bounded_error_text(error_kind, 64)
        safe_code = _bounded_error_text(error_code, 96)
        safe_message = _bounded_error_text(message, _ERROR_MESSAGE_LIMIT)
        super().__init__(f"daemon 推理失败 [{safe_kind}/{safe_code}]: {safe_message}")
        self.error_kind, self.error_code = error_kind, error_code
        if isinstance(record_ids, list):
            self.record_id_count = len(record_ids)
            self.record_ids = [
                _bounded_error_text(value, _IDENTITY_TEXT_LIMIT)
                for value in record_ids[:_IDENTITY_SAMPLE_LIMIT]
            ]
        else:
            self.record_id_count = 0
            self.record_ids = record_ids


def _write_audiotag_input_json(records: List[AudioRecord], input_json: str) -> None:
    """写出 AudioTag daemon 输入 JSON（携带 record_id 供身份守恒校验）。"""
    with open(input_json, "w", encoding="utf-8") as f:
        # 紧凑 JSON：该文件每 batch 写读一次即删除，indent 只增加体积
        json.dump(
            [{"record_id": rec.record_id,
              "audio_path": rec.audio["path"]}
             for rec in records],
            f,
            ensure_ascii=False,
        )


class _AudioTagDaemonPool(BaseDaemonPool):
    """AudioTag subprocess_daemon 模式下管理多 GPU 长驻 daemon 进程。

    生命周期管理（启动/ping/重启/stop/收发/轮询）由 BaseDaemonPool 提供，
    本类只保留 AudioTag 协议差异与 infer 业务逻辑。
    """

    def __init__(
        self,
        python_path: str,
        daemon_script_path: str,
        checkpoint: str,
        label_dict: str,
        sample_rate: int,
        batch_size: int,
        gpu_ids: List[int],
        num_gpus: int,
        timeout: int,
        stage_name: str,
        startup_timeout: int = 600,
        cpu_threads: int = 1,
        interop_threads: int = 1,
        audio_workers: int = 4,
        timing_log_interval: int = 20,
        request_batches: int = 1,
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
        self.label_dict = label_dict
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.cpu_threads = cpu_threads
        self.interop_threads = interop_threads
        self.audio_workers = audio_workers
        self.timing_log_interval = timing_log_interval
        # 每个 daemon 请求包含 batch_size × request_batches 条记录：daemon 内部
        # 按 batch_size 循环多个 sub-batch，批间双缓冲（解码与 GPU 推理重叠）
        # 只有请求内含 ≥2 个 sub-batch 才生效；同时减少 JSON 文件往返次数。
        self.request_batches = request_batches

    @staticmethod
    def _default_script_path(script_path: Optional[str]) -> str:
        if script_path:
            script_dir = os.path.dirname(os.path.abspath(script_path))
        else:
            # 默认与 icefall_audiotaq 模型同目录（use_train 根目录 third_party_scripts/icefall_audiotaq）
            # __file__ = tools/noise_filter_pipeline/processors/audiotag_classifier.py，需 4 层 dirname 到 use_train 根
            use_train_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
            third_party = os.environ.get("USE_TRAIN_THIRD_PARTY_DIR") or os.path.join(
                use_train_root, "third_party_scripts"
            )
            script_dir = os.path.join(third_party, "icefall_audiotaq")
        return os.path.join(script_dir, "audiotag_daemon.py")

    def _build_cmd(self, gpu_id: int) -> List[str]:
        return [
            "--checkpoint", self.checkpoint,
            "--label-dict", self.label_dict,
            "--sample-rate", str(self.sample_rate),
            "--batch-size", str(self.batch_size),
            "--cpu-threads", str(self.cpu_threads),
            "--interop-threads", str(self.interop_threads),
            "--audio-workers", str(self.audio_workers),
            "--timing-log-interval", str(self.timing_log_interval),
            # CUDA_VISIBLE_DEVICES 已经把该子进程限制到唯一一张卡，
            # daemon 内部看到的设备号恒为 cuda:0，避免 cuda:{gpu_id} 越界。
            "--device", "cuda:0",
        ]

    def _ping_payload(self) -> Dict[str, Any]:
        return {"cmd": "ping"}

    def _check_pong(self, resp: Dict[str, Any]) -> bool:
        return resp.get("status") == "ok"

    def _stop_payload(self) -> Dict[str, Any]:
        return {"cmd": "exit"}

    def _validate_required_files(self) -> None:
        if not self.label_dict or not os.path.isfile(self.label_dict):
            raise FileNotFoundError(
                f"{self.stage_name} daemon 模式需要 label_dict: {self.label_dict}"
            )

    def infer(self, pending: List[AudioRecord], output_dir: str, batch_index: int) -> List[Dict[str, Any]]:
        """通过长驻 daemon 处理一个 batch，多 GPU 时按 round-robin 分发。

        首次失败（daemon 死亡/失联/异常响应）会把故障 daemon 原位重启
        （或移出轮询），随后换一个 daemon 重试一次，与 Apollo 侧行为对齐；
        重试仍失败原样上抛。全部 daemon 不可用时抛 InfraAbortError 中止 stage。
        """
        self.start(output_dir)
        expected_ids = [rec.record_id for rec in pending]
        last_err: Optional[Exception] = None
        for _attempt in range(2):
            gpu_idx, proc, lock = self._pick()
            with lock:
                input_json = os.path.join(
                    output_dir, f"{self.stage_name.lower()}_input_gpu{gpu_idx}_{batch_index:04d}.json"
                )
                output_json = os.path.join(
                    output_dir, f"{self.stage_name.lower()}_output_gpu{gpu_idx}_{batch_index:04d}.json"
                )
                _write_audiotag_input_json(pending, input_json)
                try:
                    self._send(proc, {
                        "cmd": "infer",
                        "input_path": input_json,
                        "output_path": output_json,
                        "record_ids": expected_ids,
                    })
                    resp = self._recv(proc)
                except Exception as e:
                    # 仅进程级故障（daemon 失联/超时/协议损坏）才原位重启替代
                    # 进程；重启失败则移出轮询，避免后续 batch 继续 round-robin
                    # 打到死进程。全部不可用时中止。
                    last_err = e
                    self._restart_daemon(gpu_idx, proc)
                    # 清理本次失败尝试的中间 JSON，避免重试/后续 batch 积累垃圾文件
                    for p in (input_json, output_json):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                    if not self._daemon_procs:
                        # 基础设施中止：抛 InfraAbortError，由上层透传以中止
                        # stage，避免被 batch 级 blanket except 吞掉降级为逐批失败
                        raise InfraAbortError(
                            f"{self.stage_name} 所有 AudioTag daemon 均不可用，中止 stage"
                        ) from e
                    continue
                expected_ids = [rec.record_id for rec in pending]
                try:
                    status = resp.get("status") if isinstance(resp, dict) else None
                    if status == "error":
                        kind, code = resp.get("error_kind"), resp.get("error_code")
                        message, ids = resp.get("message"), resp.get("record_ids")
                        if not (kind in ("content", "infra", "internal")
                                and isinstance(code, str) and code.strip()
                                and isinstance(message, str) and message.strip()
                                and ids == expected_ids):
                            kind, code, message, ids = "internal", "invalid_error_response", repr(resp), expected_ids
                        raise DaemonInferenceError(kind, code, message, ids)
                    if status != "ok":
                        raise DaemonInferenceError(
                            "internal", "invalid_status_response", repr(resp), expected_ids
                        )
                    with open(output_json, "r", encoding="utf-8") as f:
                        return json.load(f)
                except DaemonInferenceError:
                    raise
                except Exception as exc:
                    raise DaemonInferenceError(
                        "internal", "invalid_output_json", str(exc), expected_ids
                    ) from exc
                finally:
                    # 所有响应路径均清理本批中间文件。
                    for p in (input_json, output_json):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
        raise DaemonInferenceError(
            "infra", "daemon_transport", str(last_err),
            [rec.record_id for rec in pending],
        ) from last_err

    def infer_parallel(
        self, pending: List[AudioRecord], output_dir: str, batch_index: int,
        infer_shard=None,
    ) -> List[Any]:
        """按固定分片（batch_size × request_batches）并发驱动可用 daemon，按提交顺序返回结果。"""
        self.start(output_dir)
        shard_size = self.batch_size * self.request_batches
        sub_batches = [
            pending[i : i + shard_size]
            for i in range(0, len(pending), shard_size)
        ]
        if infer_shard is None:
            infer_shard = lambda shard, index: self.infer(shard, output_dir, index)
        if len(sub_batches) == 1:
            return infer_shard(sub_batches[0], batch_index * 1000)
        # 每个初始分片使用唯一请求序号，递归拆分由对应线程串行完成。
        with ThreadPoolExecutor(
            max_workers=min(len(self._daemon_procs), len(sub_batches))
        ) as pool:
            futures = [
                pool.submit(infer_shard, sub, batch_index * 1000 + shard_id)
                for shard_id, sub in enumerate(sub_batches)
            ]
            results: List[Any] = []
            for future in futures:
                results.extend(future.result())
        return results


class AudioTagClassifierProcessor(Processor):
    """C2: AudioSet top-5 事实提取 Processor。

    支持两种模式：
      - dry_run:           测试/回归用，返回确定性非零分数 top-5
      - subprocess_daemon: 长驻 daemon 模式，模型只加载一次（推荐生产环境）
    其它旧模式（inline / subprocess）已移除，启动时对非法 mode 直接 ValueError。
    """

    def __init__(self, config: Dict[str, Any]):
        # Stage C 现在只保留 AudioTag 作为分类步骤，默认位于 C2；保留 stage_name 以兼容旧编号
        stage_name = config.get("stage_name", "C2")
        super().__init__(config, stage_name=stage_name)
        self.enable = config.get("enable", True)
        if type(self.enable) is not bool:
            raise ValueError("enable 必须是严格 bool")
        self.mode = config.get("mode", "dry_run")
        # 运行模式收敛：只保留 dry_run / subprocess_daemon，非法值启动时 fail-fast
        if self.mode not in ("dry_run", "subprocess_daemon"):
            raise ValueError(
                f"{self.stage_name} 不支持的 mode: {self.mode!r}"
                f"（仅支持 dry_run / subprocess_daemon）"
            )
        self.checkpoint = config.get("checkpoint") or config.get("audiotag_checkpoint")
        self.label_dict = config.get("label_dict") or config.get("audiotag_label_dict")
        self.sample_rate = config.get("sample_rate", 16000)
        self.batch_size = config.get("batch_size", 16)
        for name, value in (
            ("sample_rate", self.sample_rate),
            ("batch_size", self.batch_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是严格正整数")
        self.script_path = config.get("script_path")
        self.python_path = config.get("python_path", "python")
        self.dry_run_labels = config.get("dry_run_labels", ["Male speech, man speaking"])
        # C2 输出 schema 是代码常量，不提供可配置入口。旧 full527/top10、
        # 可变 top-k 与业务 policy/calibration 键一律 fail-closed，避免旧配置
        # 被静默解释为新契约。
        legacy_contract_keys = {
            "output_label_key",
            "topk_output_key",
            "topk_keep_k",
            "policy_path",
            "calibration_path",
            "calibration_identity",
        }
        configured_legacy_keys = sorted(legacy_contract_keys.intersection(config))
        if configured_legacy_keys:
            raise ValueError(
                "C2 只输出固定 audioset_top5，禁止旧契约配置: "
                + ", ".join(configured_legacy_keys)
            )
        for name, default in (
            ("cpu_threads", 1), ("interop_threads", 1),
            ("audio_workers", 4), ("timing_log_interval", 20),
            ("request_batches", 1),
        ):
            value = config.get(name, default)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是严格正整数")
            setattr(self, name, value)

        # checkpoint / resume
        self.enable_checkpoint = config.get("enable_checkpoint", True)
        if type(self.enable_checkpoint) is not bool:
            raise ValueError("enable_checkpoint 必须是严格 bool")
        self.checkpoint_path = config.get("checkpoint_path")
        # 批量保存 checkpoint / manifest，降低百万级文件下的 I/O 频率
        self.checkpoint_save_interval = config.get("checkpoint_save_interval", 100)
        self.manifest_write_batch_size = config.get("manifest_write_batch_size", 100)

        # 多卡支持（subprocess_daemon 模式）
        self.num_gpus = config.get("num_gpus", 1)
        self.gpu_ids = config.get("gpu_ids", [])
        daemon_timeout = config.get("daemon_timeout", 3600)
        startup_timeout = config.get("startup_timeout", 600)
        for name, value in (
            ("checkpoint_save_interval", self.checkpoint_save_interval),
            ("manifest_write_batch_size", self.manifest_write_batch_size),
            ("num_gpus", self.num_gpus),
            ("daemon_timeout", daemon_timeout),
            ("startup_timeout", startup_timeout),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是严格正整数")
        if not isinstance(self.gpu_ids, list) or any(
            type(gpu_id) is not int or gpu_id < 0 for gpu_id in self.gpu_ids
        ):
            raise ValueError("gpu_ids 必须是非负整数列表")

        # canonical label-space metadata 与文件 identity 按需加载。
        self._label_metadata = None
        self._label_space_sha256: Optional[str] = None
        configured_identity = config.get("artifact_fingerprints")
        self._configured_identity = (
            dict(configured_identity) if isinstance(configured_identity, dict) else None
        )
        if configured_identity is not None and self._configured_identity is None:
            raise ValueError("C2 artifact_fingerprints 必须是 object")

        # subprocess_daemon 长驻模式配置：把 daemon 生命周期管理拆到独立 helper，
        # 减少本类的职责。
        self._daemon_pool: Optional[_AudioTagDaemonPool] = None
        if self.mode == "subprocess_daemon":
            daemon_script_path = config.get("daemon_script_path") or _AudioTagDaemonPool._default_script_path(
                self.script_path
            )
            self.daemon_script_path = daemon_script_path
            self._daemon_pool = _AudioTagDaemonPool(
                python_path=self.python_path,
                daemon_script_path=daemon_script_path,
                checkpoint=self.checkpoint,
                label_dict=self.label_dict,
                sample_rate=self.sample_rate,
                batch_size=self.batch_size,
                cpu_threads=self.cpu_threads,
                interop_threads=self.interop_threads,
                audio_workers=self.audio_workers,
                timing_log_interval=self.timing_log_interval,
                request_batches=self.request_batches,
                gpu_ids=self.gpu_ids,
                num_gpus=self.num_gpus,
                timeout=daemon_timeout,
                startup_timeout=startup_timeout,
                stage_name=self.stage_name,
            )

    def _load_label_metadata(self):
        if self._label_metadata is None:
            if not self.label_dict or not os.path.isfile(self.label_dict):
                raise FileNotFoundError(
                    f"{self.stage_name} 需要 label dict: {self.label_dict}"
                )
            self._label_metadata = load_label_metadata(self.label_dict)
        return self._label_metadata

    def _label_space_identity(self) -> str:
        if self._label_space_sha256 is None:
            self._label_space_sha256 = self._load_label_metadata().label_space_sha256
        return self._label_space_sha256

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _resolve_identity(self) -> Dict[str, Any]:
        """解析并严格绑定 checkpoint、label space 与固定 C2 事实契约。"""
        label_space_sha256 = self._label_space_identity()
        checkpoint_sha256 = (
            self._sha256_file(self.checkpoint)
            if self.checkpoint and os.path.isfile(self.checkpoint)
            else "unset:audiotag_checkpoint_sha256"
        )
        expected = {
            "audiotag_label_dict_sha256": label_space_sha256,
            "audiotag_checkpoint_sha256": checkpoint_sha256,
            "evidence_contract_version": C2_EVIDENCE_CONTRACT_VERSION,
            "score_contract": C2_SCORE_CONTRACT,
            "class_count": C2_CLASS_COUNT,
            "top_k": C2_TOP_K,
        }
        if self._configured_identity is not None:
            if set(self._configured_identity) != set(expected):
                raise ValueError("C2 identity 字段必须精确匹配固定事实契约")
            if self._configured_identity != expected:
                raise ValueError(
                    f"C2 identity 与实际资产/事实契约不一致: "
                    f"{self._configured_identity!r} != {expected!r}"
                )
        self._configured_identity = dict(expected)
        self.config["artifact_fingerprints"] = dict(expected)
        return expected

    def _run_dry_run(self, records: List[AudioRecord]) -> List[Dict[str, Any]]:
        """生成与真实 daemon 相同的 C2 v5 top-5 事实。"""
        labels = self.dry_run_labels or ["unknown"]
        metadata = self._load_label_metadata()
        fallback_labels = (
            "Rain",
            "Music",
            "Wind",
            "Thunder",
            "Water",
            "Background music",
        )
        full_scores = []
        for index, _record in enumerate(records):
            primary_label = labels[index % len(labels)]
            if primary_label not in metadata.label_to_mid:
                raise ValueError(
                    f"dry-run top label {primary_label!r} 不在 527 类 label dict 中"
                )
            ranked_labels = [primary_label]
            ranked_labels.extend(
                label for label in fallback_labels if label != primary_label
            )
            scores = [0.0] * CLASSIFIER_LABEL_COUNT
            for rank, label in enumerate(ranked_labels[:AUDIOTAG_TOP_K]):
                mid = metadata.label_to_mid[label]
                scores[metadata.index_of(mid)] = 1.0 - rank * 0.1
            full_scores.append(scores)
        return format_audiotag_results(
            full_scores,
            metadata,
            record_ids=[record.record_id for record in records],
        )

    def _validate_model_paths(self) -> None:
        missing = []
        if not self.label_dict or not os.path.isfile(self.label_dict):
            missing.append(f"AudioTag label dict: {self.label_dict}")
        if self.mode != "dry_run" and (
            not self.checkpoint or not os.path.isfile(self.checkpoint)
        ):
            missing.append(f"AudioTag checkpoint: {self.checkpoint}")
        if self.mode == "subprocess_daemon" and (
            not self.daemon_script_path or not os.path.isfile(self.daemon_script_path)
        ):
            missing.append(f"AudioTag daemon script: {self.daemon_script_path}")
        if missing:
            raise FileNotFoundError(
                f"{self.stage_name} 模型文件缺失，请运行以下命令下载：\n"
                "  python tools/noise_filter_pipeline/scripts/download_hive_models.py\n"
                "缺失文件：\n" + "\n".join(f"  - {m}" for m in missing)
            )

    def _validate_output_item(
        self, rec: AudioRecord, out_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """严格校验 AudioTag C2 v5 top-5 事实，不修改记录。"""
        if not isinstance(out_item, dict):
            raise ValueError(f"{self.stage_name} 输出项必须是 object")
        expected_keys = {
            "record_id", C2_TOP5_KEY,
            "evidence_contract_version", "score_contract",
            "class_count", "label_space_sha256",
        }
        if set(out_item) != expected_keys:
            raise ValueError(
                f"{self.stage_name} 输出字段必须精确为 {sorted(expected_keys)!r}，"
                f"实际为 {sorted(out_item)!r}"
            )
        output_record_id = out_item.get("record_id")
        if not isinstance(output_record_id, str) or not output_record_id.strip():
            raise ValueError(f"{self.stage_name} 输出缺少非空 record_id")
        if output_record_id != rec.record_id:
            raise ValueError(
                f"{self.stage_name} 输出 record_id {output_record_id!r} "
                f"与当前记录 {rec.record_id!r} 不一致"
            )
        if out_item.get("evidence_contract_version") != C2_EVIDENCE_CONTRACT_VERSION:
            raise ValueError(f"{self.stage_name} evidence_contract_version 非法")
        if out_item.get("score_contract") != C2_SCORE_CONTRACT:
            raise ValueError(f"{self.stage_name} score_contract 非法")
        if type(out_item.get("class_count")) is not int or out_item["class_count"] != C2_CLASS_COUNT:
            raise ValueError(f"{self.stage_name} class_count 必须为 {C2_CLASS_COUNT}")
        label_space_sha256 = out_item.get("label_space_sha256")
        expected_sha256 = self._label_space_identity()
        if label_space_sha256 != expected_sha256:
            raise ValueError(
                f"{self.stage_name} label-space identity 不一致: "
                f"{label_space_sha256!r} != {expected_sha256!r}"
            )

        top5 = out_item.get(C2_TOP5_KEY)
        if not isinstance(top5, list) or len(top5) != C2_TOP_K:
            raise ValueError(
                f"{self.stage_name} {C2_TOP5_KEY} 必须恰好包含 {C2_TOP_K} 项"
            )
        metadata = self._load_label_metadata()
        if metadata.class_count != C2_CLASS_COUNT:
            raise ValueError(
                f"{self.stage_name} canonical label-space 必须为 {C2_CLASS_COUNT} 类"
            )
        normalized_top5 = []
        seen_indices = set()
        for position, item in enumerate(top5):
            if not isinstance(item, dict) or set(item) != {"index", "mid", "label", "prob"}:
                raise ValueError(f"{self.stage_name} top5[{position}] schema 非法")
            item_index = item.get("index")
            if isinstance(item_index, bool) or type(item_index) is not int:
                raise ValueError(f"{self.stage_name} top5[{position}].index 非法")
            if item_index < 0 or item_index >= metadata.class_count:
                raise ValueError(
                    f"{self.stage_name} top5[{position}].index 超出 canonical label space"
                )
            if item_index in seen_indices:
                raise ValueError(f"{self.stage_name} top5 含重复 index {item_index}")
            seen_indices.add(item_index)
            mid = metadata.index_to_mid[item_index]
            label = metadata.label_of(mid)
            if item.get("mid") != mid or item.get("label") != label:
                raise ValueError(
                    f"{self.stage_name} top5[{position}] label-space identity 非法"
                )
            raw_prob = item.get("prob")
            if isinstance(raw_prob, bool) or not isinstance(raw_prob, (int, float)):
                raise ValueError(f"{self.stage_name} top5[{position}].prob 非法")
            prob = float(raw_prob)
            if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
                raise ValueError(
                    f"{self.stage_name} top5[{position}].prob 必须是 [0,1] 内有限值"
                )
            if position > 0:
                prev = normalized_top5[position - 1]
                # 排序契约：prob 降序，并列时 index 升序
                if prob > prev["prob"] or (
                    prob == prev["prob"] and item_index <= prev["index"]
                ):
                    raise ValueError(
                        f"{self.stage_name} top5 排序非法：位置 {position} "
                        f"(index={item_index}, prob={prob}) 违反 prob 降序/index 升序"
                    )
            normalized_top5.append({
                "index": item_index, "mid": mid, "label": label, "prob": prob,
            })

        return {
            "top5": normalized_top5,
            "label_space_sha256": expected_sha256,
        }

    def _apply_validated_output(
        self, rec: AudioRecord, payload: Dict[str, Any]
    ) -> None:
        """应用已经整批纯验证通过的 C2 v5 top-5 事实。"""
        identity = self._configured_identity or self._resolve_identity()
        rec.labels[C2_TOP5_KEY] = payload["top5"]
        rec.mark_stage(
            self.stage_name,
            "passed",
            extra={
                "evidence_contract_version": identity["evidence_contract_version"],
                "score_contract": identity["score_contract"],
                "class_count": identity["class_count"],
                "label_space_sha256": payload["label_space_sha256"],
                "audiotag_checkpoint_sha256": identity[
                    "audiotag_checkpoint_sha256"
                ],
                "top_k": identity["top_k"],
            },
        )

    def _apply_output_to_record(self, rec: AudioRecord, out_item: Dict[str, Any]) -> None:
        """兼容单条调用：先纯验证，再应用。"""
        self._apply_validated_output(rec, self._validate_output_item(rec, out_item))

    def _flush_pending(
        self,
        pending: List[AudioRecord],
        output_dir: str,
        manifest_writer: PerDatasetManifestWriter,
        ckpt: CheckpointTracker,
        batch_index: List[int],
    ) -> None:
        """对 micro-batch 做 AudioTag 推理并批量写出结果。"""
        if not pending:
            return

        if not self.enable:
            for rec in pending:
                rec.provenance = write_stage_c_processor_mode(
                    rec.provenance, "c2", "disabled"
                )
                rec.mark_stage(
                    self.stage_name,
                    "passed",
                    extra={"enabled": False},
                )
                manifest_writer.append(rec)
                ckpt.mark_done(rec.record_id)
            pending.clear()
            return

        self._validate_model_paths()
        expected_ids = [rec.record_id for rec in pending]
        if (any(not isinstance(rid, str) or not rid for rid in expected_ids)
                or len(set(expected_ids)) != len(expected_ids)):
            raise DaemonInferenceError("internal", "invalid_pending_ids",
                                       "pending record_id 缺失或重复", expected_ids)
        def infer_shard(shard: List[AudioRecord], request_index: int) -> List[Any]:
            shard_ids = [rec.record_id for rec in shard]
            try:
                output = (self._daemon_pool.infer(shard, output_dir, request_index)
                          if self.mode == "subprocess_daemon" else self._run_dry_run(shard))
            except DaemonInferenceError as exc:
                if exc.error_kind != "content":
                    raise
                if len(shard) == 1:
                    return [(shard_ids[0], ("rejected", exc))]
                middle = len(shard) // 2
                return (infer_shard(shard[:middle], request_index)
                        + infer_shard(shard[middle:], request_index))
            try:
                output_ids = [item.get("record_id") for item in output]
                if output_ids != shard_ids:
                    raise ValueError(f"输出 ID 不守恒: {output_ids!r} != {shard_ids!r}")
                payloads = [self._validate_output_item(rec, item)
                            for rec, item in zip(shard, output)]
            except Exception as exc:
                raise DaemonInferenceError("internal", "invalid_output_contract",
                                           str(exc), shard_ids) from exc
            return [(rec.record_id, ("passed", payload))
                    for rec, payload in zip(shard, payloads)]
        if self.mode == "subprocess_daemon":
            outcome_items = self._daemon_pool.infer_parallel(
                pending, output_dir, batch_index[0], infer_shard)
        else:
            outcome_items = []
            for start in range(0, len(pending), self.batch_size):
                outcome_items.extend(infer_shard(
                    pending[start:start + self.batch_size], batch_index[0]))
        if [record_id for record_id, _ in outcome_items] != expected_ids:
            raise DaemonInferenceError("internal", "id_conservation_failed",
                                       "输出 ID 顺序/完整性不守恒", expected_ids)
        outcomes = dict(outcome_items)
        rejected_count = sum(value[0] == "rejected" for value in outcomes.values())
        percentage_limit_exceeded = (
            len(pending) >= self.batch_size
            and rejected_count * 100 > len(pending) * 10
        )
        if rejected_count >= 8 or percentage_limit_exceeded:
            # 内容级失败熔断：先落盘本批结果并写 checkpoint，再中止。
            # 否则本批既无 manifest 也无 checkpoint，重跑时同序输入构成同一批、
            # 同一条件再次熔断，stage 只能在同一批反复中止而无法收敛。
            self._apply_batch_outcomes(
                pending, outcomes, manifest_writer, ckpt, batch_index,
                force_checkpoint=True,
            )
            raise DaemonInferenceError("internal", "content_failure_circuit_breaker",
                f"singleton content failures={rejected_count}/{len(pending)}", expected_ids)
        self._apply_batch_outcomes(pending, outcomes, manifest_writer, ckpt, batch_index)

    def _apply_batch_outcomes(
        self,
        pending: List[AudioRecord],
        outcomes: Dict[str, Any],
        manifest_writer: PerDatasetManifestWriter,
        ckpt: CheckpointTracker,
        batch_index: List[int],
        *,
        force_checkpoint: bool = False,
    ) -> None:
        """把本批推理结果写入 manifest 并推进 checkpoint。

        force_checkpoint=True 供熔断中止路径使用：必须以 ``ckpt.save(force=True)``
        让 checkpoint 与 manifest 同时落盘（pre-save callback 先 flush manifest），
        否则中止后重跑会重复消费同一批记录。
        """
        for rec in pending:
            status, payload = outcomes[rec.record_id]
            if status == "passed":
                mode = self.config.get("execution_mode", self.mode)
                rec.provenance = write_stage_c_processor_mode(rec.provenance, "c2", mode)
                self._apply_validated_output(rec, payload)
            else:
                rec.mark_stage(self.stage_name, "rejected", extra={
                    "reason": "singleton_content_error", "error": str(payload),
                })
                rec.rejected_reason = f"{self.stage_name.lower()}_content_error"
            manifest_writer.append(rec)
        ckpt.mark_done_batch([rec.record_id for rec in pending])
        if force_checkpoint:
            ckpt.save(force=True)
        pending.clear()
        batch_index[0] += 1

    def close(self) -> None:
        """统一资源收口（Processor.close 契约）：回收 daemon 进程池。"""
        if self._daemon_pool is not None:
            self._daemon_pool.stop()

    def process(
        self,
        input_manifest: str,
        output_dir: str,
        fingerprint_input: Optional[str] = None,
    ) -> str:
        # C2 在创建输出/checkpoint 前完整执行严格 reader：任一坏行或跨文件
        # record_id 重复都立即失败，禁止已写入部分结果后才发现输入损坏。
        for _ in read_manifest_strict(input_manifest, reject_duplicate_ids=True):
            pass
        if self.enable:
            self._validate_model_paths()
            self._resolve_identity()

        os.makedirs(output_dir, exist_ok=True)
        output_manifest = os.path.join(output_dir, f"{self.stage_name.lower()}_manifest.jsonl")

        manifest_writer, ckpt, prev_records = init_checkpointed_output(
            output_manifest, output_dir, self.config, input_manifest,
            enable_checkpoint=self.enable_checkpoint,
            checkpoint_path=self.checkpoint_path,
            save_interval=self.checkpoint_save_interval,
            manifest_batch_size=self.manifest_write_batch_size,
            stage_label="Stage C2",
            dry_run=self.mode == "dry_run",
            fingerprint_input=fingerprint_input,
        )

        total = 0
        pending: List[AudioRecord] = []
        batch_index = [0]

        try:
            with manifest_writer:
                for rec in read_manifest_strict(
                    input_manifest, reject_duplicate_ids=True
                ):
                    total += 1
                    if self.should_skip(rec):
                        manifest_writer.append(rec)
                        continue
                    if ckpt.is_done(rec.record_id):
                        prev = prev_records.get(rec.record_id)
                        if prev is not None:
                            # 上次输出仍在：恢复已处理记录
                            manifest_writer.append(prev)
                            continue
                        # 上次输出缺失（如被清理）：回退为重新处理，避免丢失结果
                    pending.append(rec)
                    effective_gpus = max(1, len(self.gpu_ids) if self.gpu_ids else self.num_gpus)
                    shard_size = self.batch_size * self.request_batches
                    chunk_size = shard_size * effective_gpus
                    if len(pending) >= chunk_size:
                        self._flush_pending(pending, output_dir, manifest_writer, ckpt, batch_index)

                self._flush_pending(pending, output_dir, manifest_writer, ckpt, batch_index)
                ckpt.save(force=True)
        finally:
            self.close()

        logger.info("%s 处理完成：%d 条记录，输出 %s", self.stage_name, total, output_manifest)
        return output_dir
