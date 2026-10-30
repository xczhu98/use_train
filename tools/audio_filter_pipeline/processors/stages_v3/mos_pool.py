"""Bounded-memory, CUDA-isolated scorer pool for Generation-3 Step3."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..mos import normalize_selectable_mos_models, verify_urgent_pk_trusted_assets
from ..mos_contract import (
    MOS_ASSET_LOAD_BINDING,
    MOS_ASSET_PROVENANCE_FIELDS,
    MOS_MODEL_ORDER,
    MOS_MODEL_REQUEST_OPTION_FIELDS,
    MOS_RUNTIME_OPTION_FIELDS,
    MOS_SCORING_EXECUTION_POLICY,
    MOS_SHIPPED_MODELS,
    MOS_WORKER_ORDINAL_SIDECAR_FIELDS,
    MOS_WORKER_PROTOCOL,
)
from ..optional_runtime import local_model_tree_sha256
from .mos_asset_preflight import VerifiedMosExecutionPlan
from .mos_assets import selected_mos_asset_provenance
from .mos_lpt import MAX_WORKER_INPUT_SUBSHARDS
from .mos_gpu_job import (
    _GpuScoringJobSettings,
    _OwnedGpuScoringJob,
    _HEARTBEAT_PROTOCOL,
    _HEARTBEAT_STDOUT_LIMIT_BYTES,
    _PROGRESS_PREFIX,
    _write_heartbeat_launcher,
)
from .mos_resume import Step3ResumeStore
from .mos_runtime import _freeze_model_options
from .runner import DatasetRunLockCapability
from .wespeaker_runtime import terminate_process_group

logger = logging.getLogger(__name__)

__all__ = ["Step3MosScorerPool", "Step3MosWorkerFleet"]

_PROTOCOL = MOS_WORKER_PROTOCOL
# 保留的兼容默认 tuple，不代表 shipped 生产 route；实际集合由配置策略与 request 决定。
_MODELS = MOS_SHIPPED_MODELS
# binding v3 与 worker protocol v6 是独立的精确契约，均不协商旧 artifact。
ASSET_LOAD_BINDING = MOS_ASSET_LOAD_BINDING
_WORKER = Path(__file__).resolve().parents[1] / "workers" / "step3_mos_worker.py"
_JSONL_MAX_LINE_BYTES = 8 * 1024 * 1024
_TREE_BOUND_OPTION_FIELDS = MappingProxyType({
    model: MOS_RUNTIME_OPTION_FIELDS[model] for model in ("scoreq", "dnsmos")
})
_ORDINAL_SIDECAR_FIELDS = MOS_WORKER_ORDINAL_SIDECAR_FIELDS


from ...utils.strict_jsonl import (
    add_cleanup_note as _add_cleanup_note,
    encode_json_line as _shared_encode_json_line,
    record_cleanup_error as _record_cleanup_error,
)


class _FrozenList(tuple):
    """保留 JSON array 相等语义的不可变序列。"""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        return False

    __hash__ = tuple.__hash__


def _freeze_json(value: Any, label: str) -> Any:
    """严格复制并冻结注入的 JSON provenance。"""
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{label} object keys must be strings")
        return MappingProxyType({
            key: _freeze_json(item, f"{label}.{key}")
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return _FrozenList(
            _freeze_json(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{label} must contain only JSON values")


def _plain_json(value: Any) -> Any:
    """返回与内部冻结快照隔离的 JSON-safe 副本。"""
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _same_json(left: Any, right: Any) -> bool:
    """按 worker 可见的严格 JSON 外形比较两个快照。"""
    return _plain_json(left) == _plain_json(right)


def _tree_sha256(value: Any, label: str) -> str:
    """校验真实 loader 所消费的模型树内容身份。"""
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be 64 lowercase hex characters")
    return value


def _validated_urgent_pk_assets(value: Any, label: str) -> dict[str, str]:
    """校验 URGENT-PK checkpoint、commit 与 pinned package 身份。"""
    expected_fields = MOS_ASSET_PROVENANCE_FIELDS["urgent_pk"]
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected_fields
    ):
        raise ValueError(f"{label} fields are invalid")
    checkpoint_sha256 = _tree_sha256(
        value["checkpoint_sha256"], f"{label} checkpoint_sha256",
    )
    source_package_sha256 = _tree_sha256(
        value["source_package_sha256"], f"{label} source_package_sha256",
    )
    source_commit = value["source_commit"]
    if (
        type(source_commit) is not str
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError(f"{label} source_commit is invalid")
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "source_commit": source_commit,
        "source_package_sha256": source_package_sha256,
    }


def _validate_tree_bound_options(model: str, value: Any, label: str) -> None:
    """按模型 canonical schema 校验 tree-bound worker options。"""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    expected_fields = _TREE_BOUND_OPTION_FIELDS[model]
    if any(type(key) is not str for key in value) or set(value) != expected_fields:
        raise ValueError(
            f"{label} fields must be exactly {sorted(expected_fields)!r}"
        )
    if type(value["model_dir"]) is not str or not value["model_dir"]:
        raise ValueError(f"{label} model_dir must be a non-empty string")
    _tree_sha256(
        value["verified_model_tree_sha256"],
        f"{label} verified_model_tree_sha256",
    )


def _freeze_preverified_provenance(
    value: Mapping[str, Any],
    selected: tuple[str, ...],
    options_by_model: Mapping[str, Mapping[str, Any] | None],
) -> Mapping[str, Any]:
    """校验并锁定与 options 绑定的 selected-only provenance。"""
    if any(type(key) is not str for key in value) or set(value) != {
        "selected_models", "models",
    }:
        raise ValueError("preverified_provenance fields are invalid")
    # provenance 按 registry 的 canonical 顺序投影，不沿用策略执行顺序。
    legacy_selected = tuple(
        model for model in MOS_MODEL_ORDER if model in selected
    )
    provenance_selected = value["selected_models"]
    if (
        not isinstance(provenance_selected, (list, tuple))
        or tuple(provenance_selected) != legacy_selected
    ):
        raise ValueError(
            "preverified_provenance selected_models must match configured models"
        )
    models = value["models"]
    if not isinstance(models, Mapping):
        raise TypeError("preverified_provenance models must be a mapping")
    if any(type(key) is not str for key in models) or tuple(models) != legacy_selected:
        raise ValueError(
            "preverified_provenance models must match configured models in order"
        )

    for model in legacy_selected:
        options = options_by_model[model]
        provenance = models[model]
        if options is None:
            raise ValueError(f"preverified_provenance {model} has no configured options")
        if not isinstance(provenance, Mapping):
            raise TypeError("preverified_provenance model entries must be mappings")
        if (
            any(type(key) is not str for key in provenance)
            or set(provenance) != MOS_ASSET_PROVENANCE_FIELDS[model]
        ):
            raise ValueError(f"preverified_provenance {model} fields are invalid")
        if model in {"scoreq", "dnsmos"}:
            _validate_tree_bound_options(model, options, f"{model}_options")
            model_dir = options["model_dir"]
            if type(provenance["model_dir"]) is not str:
                raise TypeError(f"{model} provenance model_dir must be a string")
            options_tree = _tree_sha256(
                options["verified_model_tree_sha256"],
                f"{model}_options verified_model_tree_sha256",
            )
            provenance_tree = _tree_sha256(
                provenance["verified_model_tree_sha256"],
                f"{model} provenance verified_model_tree_sha256",
            )
            if (
                os.path.realpath(os.path.abspath(model_dir)) != provenance["model_dir"]
                or options_tree != provenance_tree
            ):
                raise ValueError(f"{model} provenance is detached from configured options")
        elif model == "urgent_pk":
            assets = _validated_urgent_pk_assets(
                provenance, "urgent_pk provenance",
            )
            if (
                options.get("expected_sha256") != assets["checkpoint_sha256"]
                or options.get("expected_commit") != assets["source_commit"]
            ):
                raise ValueError("urgent_pk provenance is detached from configured options")
        else:
            model_path = options.get("model_path")
            if type(model_path) is not str or type(provenance["model_path"]) is not str:
                raise TypeError(f"{model} model_path identity must be a string")
            linked = (
                os.path.realpath(os.path.abspath(model_path)) == provenance["model_path"]
                and options.get("model_version") == provenance["model_version"]
                and options.get("model_sha256") == provenance["expected_model_tree_sha256"]
                and options.get("model_sha256") == provenance["verified_model_tree_sha256"]
                and options.get("checkpoint_sha256") == provenance["checkpoint_sha256"]
                and options.get("checkpoint_size_bytes") == provenance["checkpoint_size_bytes"]
            )
            if not linked:
                raise ValueError(f"{model} provenance is detached from configured options")
    return _freeze_json(value, "preverified_provenance")


def _close_process_streams(proc: Any) -> None:
    first_error = None
    for name in ("stdin", "stdout", "stderr"):
        try:
            stream = getattr(proc, name, None)
        except BaseException as exc:
            first_error = _record_cleanup_error(
                first_error, exc, f"process {name} lookup",
            )
            continue
        if stream is None:
            continue
        try:
            stream.close()
        except BaseException as exc:
            first_error = _record_cleanup_error(
                first_error, exc, f"process {name} close",
            )
    if first_error is not None:
        raise first_error


def _encode_json_line(value: Mapping[str, Any], label: str) -> bytes:
    return _shared_encode_json_line(
        value, label, max_line_bytes=_JSONL_MAX_LINE_BYTES,
    )


def _positive_timeout(value: Any, label: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        suffix = " or None" if nullable else ""
        raise ValueError(f"{label} must be a positive finite number{suffix}")
    return float(value)


class Step3MosScorerPool:
    """校验并发布 Step3 GPU 评分配置、provenance 与 runtime attestation。

    每次评分的 weighted LPT、临时文件、worker 调度和结果恢复由
    ``_OwnedGpuScoringJob`` 完整拥有。pool 刻意不提供 CPU 模式；五个 options
    只是可选槽位，实际 worker 模型并集取自配置或 verified plan 的非空项。
    canonical worker request 禁止 fallback，任一所选模型失败均 fail-closed。
    """

    def __init__(
        self,
        gpu_ids: Sequence[int],
        *,
        urgent_options: Mapping[str, Any] | None = None,
        scoreq_options: Mapping[str, Any] | None = None,
        dnsmos_options: Mapping[str, Any] | None = None,
        nisqa_options: Mapping[str, Any] | None = None,
        utmos_options: Mapping[str, Any] | None = None,
        preverified_provenance: Mapping[str, Any] | None = None,
        verified_plan: VerifiedMosExecutionPlan | None = None,
        dataset_id: str | None = None,
        worker_idle_timeout_seconds: float | None = None,
        worker_hard_timeout_seconds: float | None = None,
        worker_stderr_tail_bytes: int = 4_096,
        popen: Callable[..., Any] = subprocess.Popen,
        request_runner: Callable[[int, str, str], None] | None = None,
        request_abort: Callable[[BaseException], None] | None = None,
        join_aborted: Callable[[BaseException], None] | None = None,
    ) -> None:
        if type(dataset_id) is not str or not dataset_id.strip():
            raise ValueError("dataset_id must be a nonblank string")
        if (isinstance(gpu_ids, (str, bytes)) or not gpu_ids
                or any(type(gpu) is not int or gpu < 0 for gpu in gpu_ids)
                or len(set(gpu_ids)) != len(gpu_ids)):
            raise ValueError("gpu_ids must be a nonempty sequence of unique nonnegative integers")
        if urgent_options is not None and not isinstance(urgent_options, Mapping):
            raise TypeError("urgent_options must be a mapping or None")
        if scoreq_options is not None and not isinstance(scoreq_options, Mapping):
            raise TypeError("scoreq_options must be a mapping or None")
        if dnsmos_options is not None and not isinstance(dnsmos_options, Mapping):
            raise TypeError("dnsmos_options must be a mapping or None")
        if nisqa_options is not None and not isinstance(nisqa_options, Mapping):
            raise TypeError("nisqa_options must be a mapping or None")
        if utmos_options is not None and not isinstance(utmos_options, Mapping):
            raise TypeError("utmos_options must be a mapping or None")
        if (
            preverified_provenance is not None
            and not isinstance(preverified_provenance, Mapping)
        ):
            raise TypeError("preverified_provenance must be a mapping or None")
        if verified_plan is not None and type(verified_plan) is not VerifiedMosExecutionPlan:
            raise TypeError("verified_plan must be a VerifiedMosExecutionPlan or None")
        if preverified_provenance is not None and verified_plan is None:
            raise ValueError(
                "preverified_provenance requires its verified_plan capability"
            )
        if worker_idle_timeout_seconds is None:
            worker_idle_timeout_seconds = 900.0
        worker_idle_timeout_seconds = _positive_timeout(
            worker_idle_timeout_seconds, "worker_idle_timeout_seconds",
        )
        worker_hard_timeout_seconds = _positive_timeout(
            worker_hard_timeout_seconds,
            "worker_hard_timeout_seconds",
            nullable=True,
        )
        if type(worker_stderr_tail_bytes) is not int or not 0 < worker_stderr_tail_bytes <= 65_536:
            raise ValueError("worker_stderr_tail_bytes must be an integer in [1, 65536]")
        if not callable(popen):
            raise TypeError("popen must be callable")
        if request_runner is not None and not callable(request_runner):
            raise TypeError("request_runner must be callable or None")
        if request_abort is not None and not callable(request_abort):
            raise TypeError("request_abort must be callable or None")
        if join_aborted is not None and not callable(join_aborted):
            raise TypeError("join_aborted must be callable or None")
        self._gpu_ids = tuple(gpu_ids)
        self._dataset_id = dataset_id

        explicit_options = {
            "scoreq": scoreq_options,
            "urgent_pk": urgent_options,
            "dnsmos": dnsmos_options,
            "nisqa": nisqa_options,
            "utmos": utmos_options,
        }
        if verified_plan is not None:
            # production 只消费一个完整 Verified capability；平铺参数仅作兼容校验。
            verified_options = {
                model: verified_plan.options_by_model.get(model)
                for model in explicit_options
            }
            for model, options in explicit_options.items():
                if options is not None and not _same_json(
                    options, verified_options[model],
                ):
                    raise ValueError(
                        f"{model} options must match verified_plan exactly"
                    )
            if (
                preverified_provenance is not None
                and not _same_json(preverified_provenance, verified_plan.provenance)
            ):
                raise ValueError(
                    "preverified_provenance must match verified_plan exactly"
                )
            explicit_options = verified_options
            preverified_provenance = verified_plan.provenance

        frozen_options = {
            model: (
                _freeze_model_options(
                    options, model, label=f"{model}_options",
                )
                if options is not None else None
            )
            for model, options in explicit_options.items()
        }
        selected = tuple(
            model for model, options in frozen_options.items()
            if options is not None
        )
        self._verified_models = (
            tuple(verified_plan.selected_models)
            if verified_plan is not None else selected
        )
        self._asset_provenance = (
            _freeze_preverified_provenance(
                preverified_provenance, selected, frozen_options,
            )
            if preverified_provenance is not None else None
        )
        self._scoreq_options = frozen_options["scoreq"]
        self._urgent_options = frozen_options["urgent_pk"]
        self._dnsmos_options = frozen_options["dnsmos"]
        self._nisqa_options = frozen_options["nisqa"]
        self._utmos_options = frozen_options["utmos"]
        self._worker_idle_timeout_seconds = worker_idle_timeout_seconds
        self._worker_hard_timeout_seconds = worker_hard_timeout_seconds
        self._worker_stderr_tail_bytes = worker_stderr_tail_bytes
        self._popen = popen
        self._request_runner = request_runner
        self._request_abort = request_abort
        self._join_aborted = join_aborted
        self._progress_callback: Callable[[int], None] | None = None
        self._progress_callback_lock = threading.Lock()
        self._runtime_attestation: Mapping[str, Any] | None = None
        self._runtime_attestation_lock = threading.Lock()
        self._resume_lock = threading.Lock()
        self._resume_session: tuple[Step3ResumeStore, DatasetRunLockCapability] | None = None

    def configure_resume(
        self,
        store: Step3ResumeStore,
        capability: DatasetRunLockCapability,
    ) -> None:
        """为下一次评分绑定 runner 持锁的持久 resume session。"""
        if not isinstance(store, Step3ResumeStore):
            raise TypeError("resume store must be a Step3ResumeStore")
        if not isinstance(capability, DatasetRunLockCapability):
            raise TypeError("resume capability must be a DatasetRunLockCapability")
        capability.validate(store.output_root, store.dataset_id)
        with self._resume_lock:
            if self._resume_session is not None:
                raise RuntimeError("MOS scorer already has an active resume session")
            self._resume_session = (store, capability)

    def clear_resume(self) -> None:
        with self._resume_lock:
            self._resume_session = None

    def set_progress_callback(self, callback: Callable[[int], None] | None) -> None:
        """设置仅用于显示的已完成 MOS 行数回调。"""
        if callback is not None and not callable(callback):
            raise TypeError("progress callback must be callable or None")
        with self._progress_callback_lock:
            self._progress_callback = callback

    def _notify_progress(self, completed_rows: int) -> None:
        with self._progress_callback_lock:
            callback = self._progress_callback
        if callback is None:
            return
        try:
            callback(completed_rows)
        except BaseException:
            # 进度显示不能改变评分结果或触发管线失败。
            with self._progress_callback_lock:
                if self._progress_callback is callback:
                    self._progress_callback = None

    def provenance_payload(self) -> dict[str, Any]:
        """返回可序列化的执行信息与 selected-only 资产 provenance。"""
        options_by_model = {
            model: getattr(self, f"_{field}")
            for model, field in MOS_MODEL_REQUEST_OPTION_FIELDS.items()
        }
        selected = tuple(
            name for name, options in options_by_model.items() if options is not None
        )
        if self._asset_provenance is not None:
            # preflight 已完成资产验证；这里只返回隔离副本，禁止重复扫描。
            assets = _plain_json(self._asset_provenance)
        else:
            assets = selected_mos_asset_provenance(
                selected,
                options_by_model,
                tree_digest=local_model_tree_sha256,
                urgent_verifier=verify_urgent_pk_trusted_assets,
            )
        for model in ("scoreq", "dnsmos"):
            options = options_by_model[model]
            if options is None:
                continue
            expected_tree = _tree_sha256(
                options["verified_model_tree_sha256"],
                f"{model}_options verified_model_tree_sha256",
            )
            actual_tree = _tree_sha256(
                assets["models"][model]["verified_model_tree_sha256"],
                f"{model} provenance verified_model_tree_sha256",
            )
            if actual_tree != expected_tree:
                raise ValueError(f"{model} verified model tree digest mismatch")
        payload = {
            "scorer_execution": "step3_mos_pool.v6",
            "provider_class": "cuda_subprocess",
            "device_class": "gpu",
            "protocol": _PROTOCOL,
            "models": list(selected),
            "model_options": {
                model: _plain_json(options_by_model[model]) for model in selected
            },
            "worker_script": {
                "sha256": hashlib.sha256(_WORKER.read_bytes()).hexdigest(),
            },
            "asset_load_binding": _plain_json(ASSET_LOAD_BINDING),
            "scoring_execution_policy": _plain_json(
                MOS_SCORING_EXECUTION_POLICY,
            ),
            "assets": assets,
            "cpu_fallback": False,
        }
        for model in ("scoreq", "dnsmos"):
            model_asset = assets["models"].get(model)
            payload[f"{model}_model_dir"] = (
                model_asset["model_dir"] if model_asset else None
            )
            payload[f"{model}_model_tree_sha256"] = (
                model_asset["verified_model_tree_sha256"] if model_asset else None
            )
        payload["urgent_pk_assets"] = assets["models"].get("urgent_pk")
        return payload

    def resume_execution_policy(self) -> dict[str, Any]:
        """返回仅影响稳定 LPT 分片的逻辑容量，不绑定具体 GPU 编号。"""
        return {
            "scheduler": "weighted_lpt_ordinals",
            "planning_worker_limit": len(self._gpu_ids),
            "request_max_subshards": MAX_WORKER_INPUT_SUBSHARDS,
            "resume_protocol": "request_parent_verified_seal.v1",
            "scoring_execution_policy": _plain_json(
                MOS_SCORING_EXECUTION_POLICY,
            ),
        }

    def runtime_attestation(self) -> Mapping[str, Any]:
        """返回最近一次成功执行后由 parent 汇总的 provider 证明。"""
        with self._runtime_attestation_lock:
            if self._runtime_attestation is None:
                raise RuntimeError(
                    "MOS pool runtime attestation is unavailable before successful scoring"
                )
            return _plain_json(self._runtime_attestation)

    config_payload = provenance_payload

    def validate_model_tuples(
        self, model_tuples: Sequence[Sequence[str]],
    ) -> None:
        """在 Stage 创建任何 tuple spool 前校验 parent-verified 模型集合。"""
        if isinstance(model_tuples, (str, bytes)) or not isinstance(
            model_tuples, Sequence,
        ):
            raise TypeError("Step3 MOS model tuples must be a sequence")
        for value in model_tuples:
            models = normalize_selectable_mos_models(value)
            if not set(models).issubset(self._verified_models):
                raise ValueError(
                    "requested MOS models must be a subset of parent-verified models"
                )

    def score_task_groups(
        self,
        groups: Sequence[Mapping[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        """运行一次 owned GPU 评分事务并发布其成功 attestation。"""
        options_by_model = {
            model: getattr(self, f"_{field}")
            for model, field in MOS_MODEL_REQUEST_OPTION_FIELDS.items()
        }
        with self._resume_lock:
            resume_session = self._resume_session
        settings = _GpuScoringJobSettings(
            gpu_ids=self._gpu_ids,
            dataset_id=self._dataset_id,
            verified_models=self._verified_models,
            options_by_model=options_by_model,
            asset_provenance=self._asset_provenance,
            worker_idle_timeout_seconds=self._worker_idle_timeout_seconds,
            worker_hard_timeout_seconds=self._worker_hard_timeout_seconds,
            worker_stderr_tail_bytes=self._worker_stderr_tail_bytes,
            popen=self._popen,
            progress_callback=self._notify_progress,
            request_abort=self._request_abort,
            join_aborted=self._join_aborted,
            resume_store=None if resume_session is None else resume_session[0],
            resume_capability=None if resume_session is None else resume_session[1],
        )
        execution_attestation = yield from _OwnedGpuScoringJob(
            settings, self._request_runner,
        ).run(groups)
        if execution_attestation.get("runtime") != {"execution_state": "no_tasks"}:
            frozen = _freeze_json(
                execution_attestation, "MOS pool execution attestation",
            )
            with self._runtime_attestation_lock:
                self._runtime_attestation = frozen
        return _plain_json(execution_attestation)


_FLEET_READY = b"audio_filter_pipeline.step3_mos_worker.fleet.ready.v1\n"
_FLEET_DONE = b"audio_filter_pipeline.step3_mos_worker.fleet.done.v1\n"


class _FleetWorkerSlot:
    """一个 GPU 的惰性常驻 worker；锁保证同一时刻仅处理一个 request。"""

    def __init__(self, fleet: "Step3MosWorkerFleet", gpu_id: int) -> None:
        self._fleet = fleet
        self.gpu_id = gpu_id
        self._lock = threading.RLock()
        self._proc: Any = None
        self._forced_shutdown = False
        self._cancelled = threading.Event()
        self._stderr_tail = bytearray()

    @staticmethod
    def _signal_process_group(proc: Any) -> None:
        pid = getattr(proc, "pid", None)
        if type(pid) is not int or pid <= 0:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def _register_spawned(self, proc: Any) -> bool:
        with self._lock:
            if self._cancelled.is_set():
                cancelled = True
            else:
                self._proc = proc
                self._forced_shutdown = False
                cancelled = False
        if cancelled:
            primary = RuntimeError("MOS fleet worker spawn was cancelled")
            try:
                terminate_process_group(proc, killpg=os.killpg, force=True)
            except BaseException as cleanup_error:
                _add_cleanup_note(primary, "MOS late worker termination", cleanup_error)
            try:
                _close_process_streams(proc)
            except BaseException as cleanup_error:
                _add_cleanup_note(primary, "MOS late worker pipe close", cleanup_error)
            if getattr(proc, "poll", lambda: None)() is None:
                raise primary
            return False
        return True

    def _stderr_text(self) -> str:
        return bytes(self._stderr_tail).decode("utf-8", "replace").strip()

    def _spawn(self, _request_dir: str) -> Any:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        environment["AUDIO_FILTER_STEP3_HEARTBEAT_SECONDS"] = str(
            max(0.01, min(30.0, self._fleet._idle_timeout / 3.0))
        )
        launcher_dir = self._fleet._launcher_directory()
        launcher = os.path.join(launcher_dir, "heartbeat_launcher.py")
        proc = self._fleet._popen(
            [sys.executable, launcher, "--serve"],
            shell=False, env=environment, cwd=launcher_dir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        if not self._register_spawned(proc):
            raise RuntimeError("MOS fleet worker spawn was cancelled")
        self._read_control(_FLEET_READY)
        return proc

    def _read_control(self, expected: bytes) -> None:
        proc = self._proc
        if proc is None:
            raise RuntimeError("MOS fleet worker is unavailable")
        selector = selectors.DefaultSelector()
        pending = bytearray()
        start = time.monotonic()
        idle_deadline = start + self._fleet._idle_timeout
        hard_deadline = (
            None if self._fleet._hard_timeout is None
            else start + self._fleet._hard_timeout
        )
        try:
            for stream, name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
                descriptor = stream.fileno()
                os.set_blocking(descriptor, False)
                selector.register(descriptor, selectors.EVENT_READ, name)
            while True:
                if self._cancelled.is_set():
                    raise RuntimeError("MOS fleet worker request was cancelled")
                now = time.monotonic()
                deadline = idle_deadline if hard_deadline is None else min(
                    idle_deadline, hard_deadline,
                )
                if now >= deadline:
                    kind = "hard" if hard_deadline is not None and now >= hard_deadline else "idle"
                    raise RuntimeError(
                        f"MOS fleet GPU {self.gpu_id} {kind} timeout; "
                        f"stderr_tail={self._stderr_text()!r}"
                    )
                events = selector.select(min(0.1, deadline - now))
                if not events:
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"MOS fleet GPU {self.gpu_id} exited with return code "
                            f"{proc.returncode}; stderr_tail={self._stderr_text()!r}"
                        )
                    continue
                for key, _mask in events:
                    chunk = os.read(key.fd, 64 * 1024)
                    if not chunk:
                        selector.unregister(key.fd)
                        continue
                    if key.data == "stderr":
                        self._stderr_tail.extend(chunk)
                        if len(self._stderr_tail) > self._fleet._stderr_tail_bytes:
                            del self._stderr_tail[:-self._fleet._stderr_tail_bytes]
                        continue
                    pending.extend(chunk)
                    if len(pending) > _HEARTBEAT_STDOUT_LIMIT_BYTES:
                        raise RuntimeError("MOS fleet worker exceeded bounded stdout limit")
                    while b"\n" in pending:
                        line, _, remainder = pending.partition(b"\n")
                        pending = bytearray(remainder)
                        framed = line + b"\n"
                        if framed == _HEARTBEAT_PROTOCOL:
                            idle_deadline = time.monotonic() + self._fleet._idle_timeout
                        elif framed.startswith(_PROGRESS_PREFIX):
                            value = framed[len(_PROGRESS_PREFIX):-1]
                            try:
                                completed_rows = int(value.decode("ascii"))
                            except (UnicodeDecodeError, ValueError) as exc:
                                raise RuntimeError("MOS fleet progress frame is invalid") from exc
                            if completed_rows <= 0:
                                raise RuntimeError("MOS fleet progress frame is invalid")
                            self._fleet._notify_progress(completed_rows)
                        elif framed == expected:
                            return
                        else:
                            raise RuntimeError("MOS fleet worker emitted invalid control output")
        finally:
            selector.close()

    def run_request(
        self, dataset_id: str, request_path: str, cwd: str,
    ) -> None:
        error = None
        with self._lock:
            self._fleet._check_usable()
            if self._cancelled.is_set():
                raise RuntimeError("MOS fleet worker request was cancelled")
            try:
                proc = self._proc or self._spawn(cwd)
                proc.stdin.write(_encode_json_line({
                    "dataset_id": dataset_id, "request_path": request_path,
                }, "MOS fleet request control"))
                proc.stdin.flush()
                self._read_control(_FLEET_DONE)
            except BaseException as exc:
                error = exc
        if error is not None:
            self._fleet.request_abort(error)
            raise error

    def abort(self) -> None:
        self._cancelled.set()
        proc = self._proc
        if proc is not None:
            self._signal_process_group(proc)
            self._forced_shutdown = True

    def shutdown(self) -> None:
        with self._lock:
            proc = self._proc
            if proc is None:
                return
            primary = None
            try:
                if not self._forced_shutdown:
                    proc.stdin.close()
                proc.wait(timeout=self._fleet._shutdown_timeout)
                if not self._forced_shutdown and proc.returncode != 0:
                    raise RuntimeError(
                        f"MOS fleet GPU {self.gpu_id} shutdown failed with return "
                        f"code {proc.returncode}; stderr_tail={self._stderr_text()!r}"
                    )
            except BaseException as exc:
                primary = exc
                try:
                    terminate_process_group(proc, killpg=os.killpg, force=True)
                except BaseException as cleanup_error:
                    _add_cleanup_note(exc, "MOS fleet worker termination", cleanup_error)
                else:
                    self._forced_shutdown = True
            try:
                _close_process_streams(proc)
            except BaseException as cleanup_error:
                if primary is None:
                    primary = cleanup_error
                else:
                    _add_cleanup_note(
                        primary, "MOS fleet worker pipe close", cleanup_error,
                    )
            if primary is not None:
                # 失败时保留 owned process，后续 close 必须能重试同一资源。
                raise primary
            self._proc = None
            self._forced_shutdown = False


class _FleetDatasetView:
    """仅绑定 dataset；真正的 GPU worker 由共享 fleet 惰性提供。"""

    def __init__(self, fleet: "Step3MosWorkerFleet", dataset_id: str) -> None:
        self._fleet = fleet
        self._pool = Step3MosScorerPool(
            fleet._gpu_ids,
            verified_plan=fleet._verified_plan,
            dataset_id=dataset_id,
            worker_idle_timeout_seconds=fleet._idle_timeout,
            worker_hard_timeout_seconds=fleet._hard_timeout,
            worker_stderr_tail_bytes=fleet._stderr_tail_bytes,
            request_runner=lambda gpu_id, request_path, cwd: fleet._run_request(
                dataset_id, gpu_id, request_path, cwd,
            ),
            request_abort=fleet.request_abort,
            join_aborted=fleet.join_aborted,
        )

    def provenance_payload(self) -> dict[str, Any]:
        return self._pool.provenance_payload()

    config_payload = provenance_payload

    def resume_execution_policy(self) -> dict[str, Any]:
        return self._pool.resume_execution_policy()

    def configure_resume(
        self,
        store: Step3ResumeStore,
        capability: DatasetRunLockCapability,
    ) -> None:
        self._pool.configure_resume(store, capability)

    def clear_resume(self) -> None:
        self._pool.clear_resume()

    def set_progress_callback(self, callback: Callable[[int], None] | None) -> None:
        """将 Step3 进度回调转发到共享 fleet。"""
        self._fleet.set_progress_callback(callback)

    def runtime_attestation(self) -> Mapping[str, Any]:
        return self._pool.runtime_attestation()

    def validate_model_tuples(
        self, model_tuples: Sequence[Sequence[str]],
    ) -> None:
        self._fleet._check_usable()
        self._pool.validate_model_tuples(model_tuples)

    def score_task_groups(
        self, groups: Sequence[Mapping[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        self._fleet._check_usable()
        try:
            return (yield from self._pool.score_task_groups(groups))
        except GeneratorExit:
            raise
        except BaseException as exc:
            self._fleet.request_abort(exc)
            # 连锁中止会让其他 slot 的 worker 报 -15/closed 类次生错误，as_completed
            # 的完成顺序竞争可能使次生错误先浮出；这里优先改抛 fleet 记录的首个
            # 根因错误，保留次生错误作为 __cause__ 供排查。
            root = self._fleet.root_cause()
            if root is not None and root is not exc and isinstance(exc, Exception):
                raise root from exc
            raise


class Step3MosWorkerFleet:
    """一次 CLI 生命周期共享的惰性 per-GPU Step3 worker fleet。"""

    def __init__(
        self,
        gpu_ids: Sequence[int],
        *,
        verified_plan: VerifiedMosExecutionPlan,
        worker_idle_timeout_seconds: float = 900.0,
        worker_hard_timeout_seconds: float | None = None,
        worker_stderr_tail_bytes: int = 4_096,
        shutdown_timeout_seconds: float = 5.0,
        popen: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        probe = Step3MosScorerPool(
            gpu_ids, verified_plan=verified_plan, dataset_id="fleet-validation",
            worker_idle_timeout_seconds=worker_idle_timeout_seconds,
            worker_hard_timeout_seconds=worker_hard_timeout_seconds,
            worker_stderr_tail_bytes=worker_stderr_tail_bytes, popen=popen,
        )
        self._gpu_ids = probe._gpu_ids
        self._verified_plan = verified_plan
        self._idle_timeout = probe._worker_idle_timeout_seconds
        self._hard_timeout = probe._worker_hard_timeout_seconds
        self._stderr_tail_bytes = probe._worker_stderr_tail_bytes
        self._shutdown_timeout = _positive_timeout(
            shutdown_timeout_seconds, "shutdown_timeout_seconds",
        )
        self._popen = popen
        self._state_lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._launcher_temp: tempfile.TemporaryDirectory[str] | None = None
        self._slots = {
            gpu_id: _FleetWorkerSlot(self, gpu_id) for gpu_id in self._gpu_ids
        }
        self._broken: BaseException | None = None
        self._closing = False
        self._closed = False
        self._progress_callback: Callable[[int], None] | None = None
        self._progress_callback_lock = threading.Lock()

    def set_progress_callback(self, callback: Callable[[int], None] | None) -> None:
        """设置仅用于显示的已完成 MOS 行数回调。"""
        if callback is not None and not callable(callback):
            raise TypeError("progress callback must be callable or None")
        with self._progress_callback_lock:
            self._progress_callback = callback

    def _notify_progress(self, completed_rows: int) -> None:
        with self._progress_callback_lock:
            callback = self._progress_callback
        if callback is not None:
            try:
                callback(completed_rows)
            except BaseException:
                with self._progress_callback_lock:
                    if self._progress_callback is callback:
                        self._progress_callback = None

    def for_dataset(self, dataset_id: str) -> _FleetDatasetView:
        """返回绑定单个 dataset、共享本 fleet GPU workers 的 scorer。"""
        self._check_usable()
        if type(dataset_id) is not str or not dataset_id.strip():
            raise ValueError("dataset_id must be a nonblank string")
        return _FleetDatasetView(self, dataset_id)

    def _check_usable(self) -> None:
        with self._state_lock:
            if self._broken is not None:
                raise RuntimeError("MOS worker fleet is broken") from self._broken
            if self._closed:
                raise RuntimeError("MOS worker fleet is closed")
            if self._closing:
                raise RuntimeError("MOS worker fleet is closing")

    def _launcher_directory(self) -> str:
        with self._state_lock:
            self._check_usable()
            if self._launcher_temp is None:
                self._launcher_temp = tempfile.TemporaryDirectory(
                    prefix="audio-filter-step3-mos-fleet-",
                )
                os.chmod(self._launcher_temp.name, stat.S_IRWXU)
                _write_heartbeat_launcher(self._launcher_temp.name)
            return self._launcher_temp.name

    def _run_request(
        self, dataset_id: str, gpu_id: int, request_path: str, cwd: str,
    ) -> None:
        self._slots[gpu_id].run_request(dataset_id, request_path, cwd)

    def root_cause(self) -> BaseException | None:
        """返回触发 fleet 中止的首个根因错误；未中止时为 None。"""
        with self._state_lock:
            return self._broken

    def request_abort(self, error: BaseException) -> None:
        """记录首个根因并仅广播非阻塞终止信号。"""
        with self._state_lock:
            if self._broken is not None:
                return
            self._broken = error
        logger.error(
            "MOS worker fleet aborting on root error: %r",
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
        for slot in self._slots.values():
            try:
                slot.abort()
            except BaseException as cleanup_error:
                _add_cleanup_note(error, "MOS fleet abort", cleanup_error)

    _break = request_abort

    def join_aborted(self, primary: BaseException) -> None:
        """在 worker executor 全部收敛后回收 fleet owned 资源。"""
        self._shutdown_slots(primary)

    def _shutdown_slots(self, primary: BaseException | None = None) -> None:
        first_error = None
        for slot in self._slots.values():
            try:
                slot.shutdown()
            except BaseException as exc:
                first_error = _record_cleanup_error(
                    first_error, exc, f"MOS fleet GPU {slot.gpu_id} shutdown",
                )
        with self._state_lock:
            launcher_temp = self._launcher_temp
        if first_error is None and launcher_temp is not None:
            try:
                launcher_temp.cleanup()
            except BaseException as exc:
                first_error = _record_cleanup_error(
                    first_error, exc, "MOS fleet launcher directory cleanup",
                )
            else:
                with self._state_lock:
                    if self._launcher_temp is launcher_temp:
                        self._launcher_temp = None
        if first_error is not None:
            if primary is not None:
                _add_cleanup_note(primary, "MOS fleet shutdown", first_error)
            else:
                raise first_error

    def close(self) -> None:
        """关闭全部资源；失败时保留 owned 状态供后续调用重试。"""
        with self._close_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closing = True
            try:
                self._shutdown_slots()
            except BaseException:
                # 已进入关闭流程后禁止重新调度，但保留可重试的未收敛状态。
                raise
            with self._state_lock:
                self._closed = True
                self._closing = False

    def __enter__(self) -> "Step3MosWorkerFleet":
        self._check_usable()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            _add_cleanup_note(exc, "MOS fleet shutdown", cleanup_error)
        return False
