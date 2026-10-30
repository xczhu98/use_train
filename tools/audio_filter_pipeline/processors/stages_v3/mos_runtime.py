"""Generation-3 Step3 的真实 MOS 运行时适配器。"""

from __future__ import annotations

import contextlib
import heapq
import math
import os
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from ..mos_contract import (
    MOS_ASSET_LOAD_BINDING,
    MOS_ASSET_PROVENANCE_FIELDS,
    MOS_MODEL_ORDER,
    MOS_MODEL_REQUEST_OPTION_FIELDS,
    MOS_PROVIDER_ATTESTATION_FIELDS,
    MOS_RUNTIME_OPTION_FIELDS,
    MOS_SCORING_EXECUTION_POLICY,
    is_mos_device_attestation,
    validate_mos_provider_attestation,
)
from ..mos import (
    bind_dnsmos_options,
    bind_nisqa_options,
    bind_scoreq_options,
    bind_urgent_pk_options,
    bind_utmos_options,
    close_mos_estimators,
    compute_mos_scores,
    compute_mos_scores_batch,
    get_mos_estimator,
    map_mos_raw_result,
    normalize_selectable_mos_models,
    urgent_pk_verified_assets,
    verify_urgent_pk_trusted_assets,
)
from ...utils.strict_jsonl import (
    add_cleanup_note,
    decode_json_line,
    file_signature,
    finalize_streams,
    iter_bounded_jsonl,
    open_regular_stream,
    record_cleanup_error,
)
from .mos_asset_preflight import VerifiedMosExecutionPlan
from .mos_assets import selected_mos_asset_provenance
from .select import TUPLE_SPOOL_MAX_FILE_BYTES
from .source_io import decode_verified_source, open_verified_source_session


_SCORING_EXECUTION_POLICY = MOS_SCORING_EXECUTION_POLICY
_MAX_BATCH_ROWS = _SCORING_EXECUTION_POLICY["batching"]["max_rows"]
_MAX_BATCH_FLOAT32_BYTES = _SCORING_EXECUTION_POLICY["batching"]["max_float32_bytes"]
_MODEL_ORDER = MOS_MODEL_ORDER
_SOURCE_IDENTITY_FIELDS = (
    "input_root", "source_relative_path", "source_size_bytes", "source_sha256",
    "sample_rate", "frames", "channels", "format", "subtype",
)
_MODEL_OPTION_FIELDS = MOS_RUNTIME_OPTION_FIELDS
# 资产 binding、评分执行 policy 与 worker protocol 独立精确绑定，不协商旧 artifact。
ASSET_LOAD_BINDING = MOS_ASSET_LOAD_BINDING
SCORING_EXECUTION_POLICY = MOS_SCORING_EXECUTION_POLICY
_ASSET_PROVENANCE_MODEL_FIELDS = MOS_ASSET_PROVENANCE_FIELDS


def _freeze_json(value: Any, label: str) -> Any:
    """递归复制并冻结严格 JSON 值。"""
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
    if type(value) in {list, tuple}:
        return tuple(
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


def _validate_sha256(value: Any, label: str) -> str:
    """校验 lowercase SHA-256 内容身份。"""
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _freeze_model_options(
    value: Any, model: str, *, label: str,
) -> Mapping[str, Any]:
    """校验单模型 options 的精确字段并保存严格 JSON 快照。"""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    expected_fields = _MODEL_OPTION_FIELDS[model]
    if any(type(key) is not str for key in value) or set(value) != expected_fields:
        raise ValueError(f"{label} fields must be exactly {sorted(expected_fields)!r}")
    if model in {"scoreq", "dnsmos"}:
        _validate_sha256(
            value["verified_model_tree_sha256"],
            f"{label}.verified_model_tree_sha256",
        )
    return _freeze_json(value, label)


# Step3 GPU worker 直接消费的执行工具：以公开别名声明跨模块契约，
# worker 不再 reach-through 本模块的私有符号。
plain_json = _plain_json
freeze_model_options = _freeze_model_options


def _freeze_asset_provenance(
    value: Any, configured_models: tuple[str, ...],
) -> Mapping[str, Any]:
    """校验 preflight 资产身份外形，并保存不可变快照。"""
    if not isinstance(value, Mapping):
        raise TypeError("asset_provenance must be a mapping or None")
    if any(type(key) is not str for key in value) or set(value) != {
        "selected_models", "models",
    }:
        raise ValueError(
            "asset_provenance fields must be exactly ['models', 'selected_models']"
        )
    selected_models = value["selected_models"]
    if type(selected_models) not in {list, tuple}:
        raise TypeError("asset_provenance selected_models must be an array")
    selected = tuple(selected_models)
    if selected != configured_models:
        raise ValueError(
            "asset_provenance selected_models must exactly match configured MOS assets"
        )
    models = value["models"]
    if not isinstance(models, Mapping):
        raise TypeError("asset_provenance models must be a mapping")
    if any(type(key) is not str for key in models) or tuple(models) != selected:
        raise ValueError(
            "asset_provenance models must exactly match selected_models in order"
        )
    for model, provenance in models.items():
        expected_fields = _ASSET_PROVENANCE_MODEL_FIELDS.get(model)
        if expected_fields is None:
            raise ValueError(f"unsupported MOS asset provenance model: {model!r}")
        if not isinstance(provenance, Mapping):
            raise TypeError(f"asset_provenance models.{model} must be a mapping")
        if (
            any(type(key) is not str for key in provenance)
            or set(provenance) != expected_fields
        ):
            raise ValueError(
                f"asset_provenance models.{model} fields must be exactly "
                f"{sorted(expected_fields)!r}"
            )
        if "verified_model_tree_sha256" in provenance:
            _validate_sha256(
                provenance["verified_model_tree_sha256"],
                f"asset_provenance models.{model}.verified_model_tree_sha256",
            )
    return _freeze_json(value, "asset_provenance")


def _options_by_model(options: "MosRuntimeOptions") -> dict[str, Mapping[str, Any] | None]:
    """按 registry 定义的 canonical provenance 顺序投影 runtime options。"""
    return {
        model: getattr(options, field)
        for model, field in MOS_MODEL_REQUEST_OPTION_FIELDS.items()
    }


def _configured_models(
    options_by_model: Mapping[str, Mapping[str, Any] | None],
) -> tuple[str, ...]:
    return tuple(
        model for model in _MODEL_ORDER if options_by_model.get(model) is not None
    )


def _validate_options_provenance_binding(
    options_by_model: Mapping[str, Mapping[str, Any] | None],
    provenance: Mapping[str, Any],
    selected_models: tuple[str, ...],
) -> None:
    """逐模型绑定执行 options 与已验证资产身份。"""
    models = provenance["models"]
    for model in selected_models:
        options = options_by_model.get(model)
        if options is None:
            raise ValueError(f"selected {model} is missing runtime options")
        frozen_options = _freeze_model_options(
            options, model, label=f"runtime options.{model}",
        )
        identity = models[model]
        if model in {"scoreq", "dnsmos"}:
            if os.path.realpath(os.path.abspath(frozen_options["model_dir"])) != identity["model_dir"]:
                raise ValueError(f"{model} model_dir is detached from asset provenance")
            if (
                frozen_options["verified_model_tree_sha256"]
                != identity["verified_model_tree_sha256"]
            ):
                raise ValueError(
                    f"{model} verified model tree digest is detached from asset provenance"
                )
            continue
        if model == "urgent_pk":
            if frozen_options["expected_sha256"] != identity["checkpoint_sha256"]:
                raise ValueError("urgent_pk checkpoint is detached from asset provenance")
            if frozen_options["expected_commit"] != identity["source_commit"]:
                raise ValueError("urgent_pk source is detached from asset provenance")
            _validate_sha256(
                identity["source_package_sha256"],
                "urgent_pk source package is detached from asset provenance",
            )
            continue
        if os.path.realpath(os.path.abspath(frozen_options["model_path"])) != identity["model_path"]:
            raise ValueError(f"{model} model_path is detached from asset provenance")
        bindings = (
            ("model_version", "model_version"),
            ("model_sha256", "expected_model_tree_sha256"),
            ("model_sha256", "verified_model_tree_sha256"),
            ("checkpoint_sha256", "checkpoint_sha256"),
            ("checkpoint_size_bytes", "checkpoint_size_bytes"),
        )
        if any(frozen_options[option_key] != identity[identity_key]
               for option_key, identity_key in bindings):
            raise ValueError(f"{model} options are detached from asset provenance")


def _runtime_options_from_verified(
    verified_options: Mapping[str, Mapping[str, Any]],
    selected_models: tuple[str, ...],
    explicit: "MosRuntimeOptions | None",
) -> "MosRuntimeOptions":
    """从 Verified snapshot 构造 options，并严格交叉绑定显式值。"""
    if tuple(verified_options) != selected_models:
        raise ValueError("verified options must exactly match selected models in order")
    frozen = {
        model: _freeze_model_options(
            verified_options[model], model, label=f"verified options_by_model.{model}",
        )
        for model in selected_models
    }
    if explicit is not None:
        explicit_by_model = _options_by_model(explicit)
        if _configured_models(explicit_by_model) != tuple(
            model for model in _MODEL_ORDER if model in selected_models
        ):
            raise ValueError(
                "explicit MOS runtime options must exactly match verified selected models"
            )
        for model in selected_models:
            explicit_options = explicit_by_model[model]
            assert explicit_options is not None
            if explicit_options != frozen[model]:
                raise ValueError(
                    f"explicit {model} options do not match verified execution plan"
                )
        use_gpu = explicit.use_gpu
        use_cache = explicit.use_cache
    else:
        use_gpu = False
        use_cache = True
    return MosRuntimeOptions(
        use_gpu=use_gpu,
        use_cache=use_cache,
        scoreq_options=frozen.get("scoreq"),
        urgent_options=frozen.get("urgent_pk"),
        dnsmos_options=frozen.get("dnsmos"),
        nisqa_options=frozen.get("nisqa"),
        utmos_options=frozen.get("utmos"),
    )


def _first_channel_owned_float32(audio: Any) -> np.ndarray:
    """选择 Python 索引 0 通道，并返回连续、独占的 float32 mono 数组。"""
    array = np.asarray(audio)
    if array.ndim == 2:
        if array.shape[1] == 0:
            raise ValueError("MOS waveform must contain at least one channel")
        array = array[:, 0]
    elif array.ndim != 1:
        raise ValueError("MOS waveform must be one- or two-dimensional")
    return np.array(array, dtype=np.float32, order="C", copy=True)


@dataclass(frozen=True)
class MosRuntimeOptions:
    """调用现有 MOS 运行时所需的显式、任务级参数。

    五个 ``*_options`` 是 registry 支持模型的可选槽位，不代表固定执行五模型；
    只有配置策略选中且槽位非空的模型才会 preflight、加载和评分。
    """

    use_gpu: bool = False
    use_cache: bool = True
    urgent_options: Mapping[str, Any] | None = None
    scoreq_options: Mapping[str, Any] | None = None
    dnsmos_options: Mapping[str, Any] | None = None
    nisqa_options: Mapping[str, Any] | None = None
    utmos_options: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if type(self.use_gpu) is not bool or type(self.use_cache) is not bool:
            raise ValueError("MOS runtime boolean options must be bool")
        option_models = {
            field: model
            for model, field in MOS_MODEL_REQUEST_OPTION_FIELDS.items()
        }
        for name, model in option_models.items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping or None")
            if value is not None:
                object.__setattr__(
                    self, name,
                    _freeze_model_options(value, model, label=name),
                )


def _validate_source_binding(row: Mapping[str, Any]) -> None:
    """防止 artifact 中的展示路径与经验证 source descriptor 脱钩。"""
    for field in ("input_root", "original_path", "source_relative_path"):
        if type(row.get(field)) is not str or not row[field]:
            raise ValueError(f"Step2 row {field} must be a nonblank string")
    root = os.path.realpath(os.path.abspath(row["input_root"]))
    expected_path = os.path.realpath(os.path.join(root, row["source_relative_path"]))
    actual_path = os.path.realpath(os.path.abspath(row["original_path"]))
    if actual_path != expected_path:
        raise ValueError("Step2 original_path is not bound to input_root/source_relative_path")


def _decode_verified_source_audio(row: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    """安全解码完整 source；调用方统一按 Step2 半开区间裁剪。"""
    _validate_source_binding(row)
    decoded = decode_verified_source(row)
    return decoded.audio, decoded.sample_rate


def _runtime_provider_entry(
    model: str,
    value: Any,
    *,
    requested: str,
    expected_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    """校验 estimator 公共证明的现有 exact wire schema。"""
    if model in {"scoreq", "dnsmos"}:
        validate_mos_provider_attestation(
            model,
            value,
            requested=requested,
            schema_label=f"Step3 {model} runtime attestation schema",
        )
    else:
        expected_fields = MOS_PROVIDER_ATTESTATION_FIELDS[model]
        if (
            type(value) is not dict
            or any(type(key) is not str for key in value)
            or set(value) != expected_fields
        ):
            raise ValueError(f"Step3 {model} runtime attestation schema is invalid")
        device = value["device"]
        parameters = value["parameters"]
        if (
            not is_mos_device_attestation(device, requested=requested)
            or not is_mos_device_attestation(parameters, requested=requested)
        ):
            raise ValueError(f"Step3 {model} device is not {requested}")
        if model in {"nisqa", "utmos"}:
            checkpoint_sha256 = _validate_sha256(
                value["checkpoint_sha256"],
                f"Step3 {model} runtime attestation checkpoint_sha256",
            )
            if checkpoint_sha256 != expected_checkpoint_sha256:
                raise ValueError(
                    f"Step3 {model} checkpoint digest mismatch: "
                    f"expected={expected_checkpoint_sha256}, "
                    f"actual={checkpoint_sha256}"
                )
    return _plain_json(_freeze_json(value, f"Step3 {model} runtime attestation"))


@contextlib.contextmanager
def _close_uncached_provider_estimator(estimator: Any):
    """关闭 provider attestation 临时 estimator，并保留检查主异常。"""
    primary = None
    try:
        yield
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            close_mos_estimators((estimator,))
        except BaseException as cleanup:
            if primary is None:
                raise
            add_note = getattr(primary, "add_note", None)
            note = f"provider estimator cleanup failed: {cleanup}"
            if callable(add_note):
                add_note(note)
            else:
                notes = getattr(primary, "__notes__", None)
                if notes is None:
                    notes = []
                    primary.__notes__ = notes
                notes.append(note)


def _runtime_provider_provenance(
    options: MosRuntimeOptions,
    selected_models: Sequence[str],
) -> dict[str, Any]:
    """加载所选 estimator 并验证真实 provider/device，禁止静默 fallback。

    canonical runtime 对每个所选 estimator 固定传入 ``allow_fallback=False``；
    加载、推理或 provider 不匹配均 fail-closed，不接受 direct estimator 的
    fallback。URGENT-PK verifier 还会把实际 package blob 内容绑定批准 commit，
    真实 loader 仅从其安全扫描后的 owned snapshot 导入。
    """
    selected = normalize_selectable_mos_models(selected_models)
    requested = "cuda" if options.use_gpu else "cpu"
    observed: dict[str, Any] = {}
    options_by_model = _options_by_model(options)
    for model in selected:
        if options_by_model[model] is None:
            raise ValueError(f"Step3 selected {model} is missing runtime options")
    option_bindings = (
        (bind_urgent_pk_options, options.urgent_options),
        (bind_scoreq_options, options.scoreq_options),
        (bind_dnsmos_options, options.dnsmos_options),
        (bind_nisqa_options, options.nisqa_options),
        (bind_utmos_options, options.utmos_options),
    )
    urgent_verified = None
    if "urgent_pk" in selected:
        if options.urgent_options is None:
            raise ValueError("Step3 selected URGENT-PK without configured assets")
        urgent_verified = verify_urgent_pk_trusted_assets(
            options.urgent_options["checkpoint"], options.urgent_options["source_dir"],
            options.urgent_options.get("expected_sha256"),
            options.urgent_options.get("expected_commit"),
        )
    with contextlib.ExitStack() as stack:
        for binder, value in option_bindings:
            stack.enter_context(
                binder(_plain_json(value) if value is not None else None)
            )
        for model in selected:
            estimator = get_mos_estimator(
                model,
                use_gpu=options.use_gpu,
                use_cache=options.use_cache,
                allow_fallback=False,
            )
            if not options.use_cache:
                stack.enter_context(_close_uncached_provider_estimator(estimator))
            model_options = options_by_model[model]
            assert model_options is not None
            observed[model] = _runtime_provider_entry(
                model,
                estimator.runtime_attestation(),
                requested=requested,
                expected_checkpoint_sha256=model_options.get("checkpoint_sha256"),
            )
    result = {
        "requested_device": requested,
        "actual_providers": observed,
        "fallback": False,
        "selected_models": list(selected),
        "asset_load_binding": _plain_json(ASSET_LOAD_BINDING),
        "scoring_execution_policy": _plain_json(SCORING_EXECUTION_POLICY),
    }
    if urgent_verified is not None:
        assets = urgent_pk_verified_assets(
            options.urgent_options.get("checkpoint"),
            options.urgent_options.get("source_dir"),
        )
        if assets != urgent_verified:
            raise ValueError("Step3 URGENT-PK verified asset identity is unavailable")
        result["urgent_pk_assets"] = assets
    return result


class _Step3MosSourceGroup:
    """一个 immutable source 的有界、可复用 verified session。"""

    def __init__(
        self, runtime: "Step3MosRuntime", row: Mapping[str, Any], models: Sequence[str],
    ) -> None:
        self._runtime = runtime
        self._models = tuple(models)
        self._identity = runtime.source_identity(row)
        self._session_context = runtime._open_source_session(row)
        self._session: Any | None = None

    @property
    def identity(self) -> tuple[Any, ...]:
        """返回用于连续 source 分组的不可变身份。"""
        return self._identity

    def __enter__(self) -> "_Step3MosSourceGroup":
        self._session = self._session_context.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        self._session = None
        return self._session_context.__exit__(*args)

    def score_batch(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """解码同源区间，投影到第一通道 mono 后批量映射 MOS 结果。"""
        if (isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence)
                or not rows or len(rows) > _MAX_BATCH_ROWS):
            raise ValueError(f"Step3 MOS batch must contain 1 to {_MAX_BATCH_ROWS} rows")
        if self._session is None:
            raise RuntimeError("Step3 MOS source group must be entered before scoring")
        decoded: list[tuple[np.ndarray, int]] = []
        decoded_bytes = 0
        mapped_results: list[dict[str, Any]] = []
        for row in rows:
            if self._runtime.source_identity(row) != self._identity:
                raise ValueError("Step3 MOS source group rows must share one immutable source identity")
            audio, sample_rate = self._session.decode_interval(
                row["start_sample"], row["end_sample"],
            )
            if (type(audio) is not np.ndarray or audio.ndim not in (1, 2)
                    or not audio.size or type(sample_rate) is not int or sample_rate <= 0):
                raise ValueError("source session returned an invalid interval waveform")
            owned_audio = _first_channel_owned_float32(audio)
            if owned_audio.nbytes > _MAX_BATCH_FLOAT32_BYTES:
                raise ValueError("decoded MOS interval exceeds the float32 batch byte limit")
            if decoded and decoded_bytes + owned_audio.nbytes > _MAX_BATCH_FLOAT32_BYTES:
                mapped_results.extend(self._runtime._score_decoded_batch(decoded, self._models))
                decoded = []
                decoded_bytes = 0
            decoded.append((owned_audio, sample_rate))
            decoded_bytes += owned_audio.nbytes
        mapped_results.extend(self._runtime._score_decoded_batch(decoded, self._models))
        return mapped_results


class _Step3MosWaveformCoordinator:
    """按模型 tuple 聚合跨 source 的受限 mono waveform 队列。"""

    def __init__(self, runtime: "Step3MosRuntime", models: Sequence[str]) -> None:
        self._runtime = runtime
        self._models = tuple(models)
        self._items: list[tuple[Any, tuple[np.ndarray, int]]] = []
        self._waveform_bytes = 0
        self._identity: tuple[Any, ...] | None = None
        self._source_context: Any | None = None
        self._source_session: Any | None = None
        self._closed = False

    @staticmethod
    def _decode_reservation(row: Mapping[str, Any]) -> int:
        samples = row["end_sample"] - row["start_sample"]
        channels = row["channels"]
        if type(samples) is not int or type(channels) is not int or samples <= 0 or channels <= 0:
            raise ValueError("Step3 MOS interval reservation is invalid")
        itemsize = np.dtype(np.float32).itemsize
        reservation = samples * (channels + 1) * itemsize
        if reservation > _MAX_BATCH_FLOAT32_BYTES:
            raise ValueError("decoded MOS interval exceeds the float32 batch byte limit")
        return reservation

    def _close_source(self, *args: Any) -> None:
        context, self._source_context = self._source_context, None
        self._source_session = None
        self._identity = None
        if context is not None:
            context.__exit__(*args)

    def _flush(self) -> list[tuple[Any, dict[str, Any]]]:
        if not self._items:
            return []
        values = [value for value, _audio_data in self._items]
        audio_data = [item for _value, item in self._items]
        scores = self._runtime._score_decoded_batch(audio_data, self._models)
        if len(scores) != len(values):
            raise ValueError("MOS batch result count must exactly match queued waveforms")
        self._items.clear()
        self._waveform_bytes = 0
        return list(zip(values, scores, strict=True))

    def push(self, row: Mapping[str, Any], value: Any) -> list[tuple[Any, dict[str, Any]]]:
        if self._closed:
            raise RuntimeError("Step3 MOS waveform coordinator is closed")
        identity = self._runtime.source_identity(row)
        reservation = self._decode_reservation(row)
        completed: list[tuple[Any, dict[str, Any]]]
        if (len(self._items) == _MAX_BATCH_ROWS
                or self._items and self._waveform_bytes + reservation > _MAX_BATCH_FLOAT32_BYTES):
            completed = self._flush()
        else:
            completed = []
        if identity != self._identity:
            self._close_source(None, None, None)
            context = self._runtime._open_source_session(row)
            self._source_session = context.__enter__()
            self._source_context = context
            self._identity = identity
        assert self._source_session is not None
        audio, sample_rate = self._source_session.decode_interval(
            row["start_sample"], row["end_sample"],
        )
        if (type(audio) is not np.ndarray or audio.ndim not in (1, 2)
                or not audio.size or type(sample_rate) is not int or sample_rate <= 0):
            raise ValueError("source session returned an invalid interval waveform")
        owned_audio = _first_channel_owned_float32(audio)
        if owned_audio.nbytes > _MAX_BATCH_FLOAT32_BYTES:
            raise ValueError("decoded MOS interval exceeds the float32 batch byte limit")
        if self._items and self._waveform_bytes + owned_audio.nbytes > _MAX_BATCH_FLOAT32_BYTES:
            raise ValueError("decoded MOS interval exceeded its pre-decode reservation")
        self._items.append((value, (owned_audio, sample_rate)))
        self._waveform_bytes += owned_audio.nbytes
        del audio
        return completed

    def finish(self) -> list[tuple[Any, dict[str, Any]]]:
        if self._closed:
            raise RuntimeError("Step3 MOS waveform coordinator is closed")
        try:
            return self._flush()
        finally:
            self._closed = True
            self._close_source(*sys.exc_info())

    def abort(self, *args: Any) -> None:
        if self._closed:
            return
        self._closed = True
        self._items.clear()
        self._waveform_bytes = 0
        self._close_source(*args)


class Step3MosRuntime:
    """将真实 MOS 结果转换为 ``Step3MosSelectionStage`` 的严格 scorer 契约。"""

    def __init__(
        self,
        *,
        decoder: Callable[[Mapping[str, Any]], tuple[np.ndarray, int]] = _decode_verified_source_audio,
        compute_scores: Callable[..., Mapping[str, Any]] = compute_mos_scores,
        open_source_session: Callable[[Mapping[str, Any]], Any] = open_verified_source_session,
        compute_scores_batch: Callable[..., Sequence[Mapping[str, Any]]] = compute_mos_scores_batch,
        options: MosRuntimeOptions | None = None,
        verified_plan: VerifiedMosExecutionPlan | None = None,
        preverified_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if not all(callable(value) for value in (
            decoder, compute_scores, open_source_session, compute_scores_batch,
        )):
            raise TypeError("MOS runtime dependencies must be callable")
        if options is not None and not isinstance(options, MosRuntimeOptions):
            raise TypeError("options must be MosRuntimeOptions or None")
        if verified_plan is not None and preverified_provenance is not None:
            raise TypeError(
                "verified_plan and preverified_provenance are mutually exclusive"
            )

        verified_models: tuple[str, ...] | None = None
        if verified_plan is not None:
            if type(verified_plan) is not VerifiedMosExecutionPlan:
                raise TypeError(
                    "verified_plan must be created by verify_mos_assets()"
                )
            snapshot = verified_plan.as_mapping()
            if set(snapshot) != {
                "selected_models", "options_by_model", "provenance",
            } or any(type(key) is not str for key in snapshot):
                raise ValueError("verified plan snapshot fields are invalid")
            selected_value = snapshot["selected_models"]
            if type(selected_value) is not list:
                raise TypeError("verified selected_models must be an array")
            verified_models = normalize_selectable_mos_models(selected_value)
            if verified_models != verified_plan.selected_models:
                raise ValueError("verified selected_models snapshot is inconsistent")
            verified_options = snapshot["options_by_model"]
            if not isinstance(verified_options, Mapping):
                raise TypeError("verified options_by_model must be a mapping")
            runtime_options = _runtime_options_from_verified(
                verified_options, verified_models, options,
            )
            legacy_models = tuple(
                model for model in _MODEL_ORDER if model in verified_models
            )
            asset_provenance = _freeze_asset_provenance(
                snapshot["provenance"], legacy_models,
            )
            _validate_options_provenance_binding(
                _options_by_model(runtime_options), asset_provenance, legacy_models,
            )
        else:
            if options is None:
                raise TypeError("options must be provided without verified_plan")
            runtime_options = options
            options_by_model = _options_by_model(runtime_options)
            configured_models = _configured_models(options_by_model)
            asset_provenance = None
            if preverified_provenance is not None:
                asset_provenance = _freeze_asset_provenance(
                    preverified_provenance, configured_models,
                )
                _validate_options_provenance_binding(
                    options_by_model, asset_provenance, configured_models,
                )
                # 必须重建真实身份后精确匹配，防止伪造 options/provenance 绕过资产验证。
                plain_options = {
                    model: (_plain_json(value) if value is not None else None)
                    for model, value in options_by_model.items()
                }
                rebuilt = _freeze_asset_provenance(
                    selected_mos_asset_provenance(
                        configured_models, plain_options,
                    ),
                    configured_models,
                )
                if rebuilt != asset_provenance:
                    raise ValueError(
                        "preverified_provenance does not match verified runtime assets"
                    )

        self._decoder = decoder
        self._compute_scores = compute_scores
        self._open_source_session = open_source_session
        self._compute_scores_batch = compute_scores_batch
        self._options = runtime_options
        self._verified_models = verified_models
        self._asset_provenance = asset_provenance
        self._provider_provenance: dict[tuple[str, ...], Mapping[str, Any]] = {}
        self._provider_provenance_lock = threading.Lock()

    def _selected_models(self, models: Sequence[str]) -> tuple[str, ...]:
        """规范化评分 tuple，并限制在 Verified capability 内。"""
        selected = normalize_selectable_mos_models(models)
        if self._verified_models is not None and not set(selected).issubset(
            self._verified_models
        ):
            raise ValueError(
                "requested MOS models must be a subset of verified selected models"
            )
        return selected

    def runtime_attestation(
        self, models: Sequence[str],
    ) -> Mapping[str, Any]:
        """导出所选模型的真实 provider 证明，并隔离内部缓存。"""
        selected = self._selected_models(models)
        with self._provider_provenance_lock:
            frozen = self._provider_provenance.get(selected)
            if frozen is None:
                frozen = _freeze_json(
                    _runtime_provider_provenance(self._options, selected),
                    "runtime attestation",
                )
                self._provider_provenance[selected] = frozen
        return _plain_json(frozen)

    def provenance_payload(self) -> Mapping[str, Any]:
        """返回配置所绑定模型的轻量资产身份，不加载 estimator。"""
        if self._asset_provenance is not None:
            assets = _plain_json(self._asset_provenance)
        else:
            frozen_options = _options_by_model(self._options)
            configured = _configured_models(frozen_options)
            options_by_model = {
                model: (_plain_json(value) if value is not None else None)
                for model, value in frozen_options.items()
            }
            assets = selected_mos_asset_provenance(configured, options_by_model)
            frozen_assets = _freeze_asset_provenance(assets, configured)
            _validate_options_provenance_binding(
                frozen_options, frozen_assets, configured,
            )
            assets = _plain_json(frozen_assets)
        options_by_model = _options_by_model(self._options)
        configured = _configured_models(options_by_model)
        return {
            "scorer_execution": "single_process",
            "provider_class": "in_process",
            "device_class": "gpu" if self._options.use_gpu else "cpu",
            "models": list(configured),
            "model_options": {
                model: _plain_json(options_by_model[model]) for model in configured
            },
            "cpu_fallback": False,
            "asset_load_binding": _plain_json(ASSET_LOAD_BINDING),
            "scoring_execution_policy": _plain_json(SCORING_EXECUTION_POLICY),
            "assets": assets,
        }

    @staticmethod
    def _iter_task_group(
        path: str, models: tuple[str, ...], row_count: int,
    ) -> Iterator[dict[str, Any]]:
        stream, initial = open_regular_stream(path, "Step3 tuple task spool")
        signature = file_signature(initial)
        payloads = None
        try:
            payloads = iter(iter_bounded_jsonl(
                stream,
                "Step3 tuple task spool",
                max_line_bytes=8 * 1024 * 1024,
                max_file_bytes=TUPLE_SPOOL_MAX_FILE_BYTES,
            ))
            count = 0
            for payload in payloads:
                task = decode_json_line(
                    payload,
                    "Step3 tuple task row",
                    max_line_bytes=8 * 1024 * 1024,
                )
                if (not isinstance(task, dict)
                        or set(task) != {
                            "ordinal", "row", "models", "scoring_policy_hash",
                        }
                        or task["models"] != list(models)
                        or type(task["ordinal"]) is not int
                        or not isinstance(task["row"], Mapping)):
                    raise ValueError("Step3 tuple task row is invalid")
                count += 1
                if count > row_count:
                    raise ValueError("Step3 tuple task row count exceeds descriptor")
                yield task
            if count != row_count:
                raise ValueError("Step3 tuple task row count differs from descriptor")
            if file_signature(os.fstat(stream.fileno())) != signature:
                raise ValueError("Step3 tuple task spool changed during traversal")
        finally:
            primary = sys.exc_info()[1]
            cleanup_error = None
            close = getattr(payloads, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    cleanup_error = record_cleanup_error(
                        cleanup_error, exc, "Step3 tuple payload iterator",
                    )
            try:
                finalize_streams(
                    (stream,),
                    None if isinstance(primary, GeneratorExit) else primary or cleanup_error,
                    sync=False,
                )
            except BaseException as exc:
                cleanup_error = record_cleanup_error(
                    cleanup_error, exc, "Step3 tuple task spool stream",
                )
            if isinstance(primary, GeneratorExit) and cleanup_error is not None:
                raise cleanup_error
            if primary is not None:
                if cleanup_error is not None:
                    add_cleanup_note(primary, "Step3 tuple task cleanup", cleanup_error)
            elif cleanup_error is not None:
                raise cleanup_error

    def score_task_groups(
        self, groups: Sequence[Mapping[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        """将同一模型 tuple 的连续物理 parts 作为单个有序流评分。"""
        if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
            raise TypeError("Step3 MOS task groups must be a sequence")
        descriptors: list[tuple[tuple[str, ...], list[tuple[str, int]]]] = []
        seen_models: set[tuple[str, ...]] = set()
        seen_paths: set[str] = set()
        for group in groups:
            if not isinstance(group, Mapping) or set(group) != {
                "models", "path", "row_count",
            }:
                raise ValueError("Step3 MOS task group schema is invalid")
            models = self._selected_models(group["models"])
            path, row_count = group["path"], group["row_count"]
            if (
                type(path) is not str
                or not os.path.isabs(path)
                or type(row_count) is not int
                or row_count <= 0
            ):
                raise ValueError("Step3 MOS task group values are invalid")
            canonical_path = os.path.realpath(path)
            if canonical_path in seen_paths:
                raise ValueError("Step3 MOS task group path is duplicate")
            seen_paths.add(canonical_path)
            if descriptors and descriptors[-1][0] == models:
                descriptors[-1][1].append((path, row_count))
            else:
                if models in seen_models:
                    raise ValueError("Step3 MOS model tuple parts must be contiguous")
                seen_models.add(models)
                descriptors.append((models, [(path, row_count)]))

        def iter_chain(
            models: tuple[str, ...], parts: Sequence[tuple[str, int]],
        ) -> Iterator[dict[str, Any]]:
            previous = -1
            for path, row_count in parts:
                iterator = iter(self._iter_task_group(path, models, row_count))
                try:
                    for task in iterator:
                        ordinal = task["ordinal"]
                        if ordinal <= previous:
                            raise ValueError(
                                "Step3 model tuple part ordinals must be strictly increasing"
                            )
                        previous = ordinal
                        yield task
                finally:
                    primary = sys.exc_info()[1]
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        try:
                            close()
                        except BaseException as cleanup_error:
                            if isinstance(primary, GeneratorExit) or primary is None:
                                raise
                            add_cleanup_note(
                                primary, "Step3 tuple part cleanup", cleanup_error,
                            )

        iterators = [
            iter(iter_chain(models, parts)) for models, parts in descriptors
        ]
        pending: list[tuple[int, int, dict[str, Any]]] = []
        try:
            for index, iterator in enumerate(iterators):
                try:
                    task = next(iterator)
                except StopIteration:
                    continue
                heapq.heappush(pending, (task["ordinal"], index, task))

            expected = 0

            def pop_next() -> tuple[int, int, dict[str, Any]]:
                nonlocal expected
                ordinal, index, task = heapq.heappop(pending)
                if ordinal != expected:
                    raise ValueError("Step3 tuple task ordinals are invalid or incomplete")
                expected += 1
                try:
                    next_task = next(iterators[index])
                except StopIteration:
                    pass
                else:
                    heapq.heappush(
                        pending, (next_task["ordinal"], index, next_task),
                    )
                return ordinal, index, task

            coordinator: _Step3MosWaveformCoordinator | None = None
            coordinator_index: int | None = None

            def emit(items: Sequence[tuple[Any, dict[str, Any]]]) -> Iterator[dict[str, Any]]:
                for value, score in items:
                    ordinal, task = value
                    yield {
                        "ordinal": ordinal,
                        "row": task["row"],
                        "models": task["models"],
                        "scores": score,
                    }

            while pending:
                ordinal, index, task = pop_next()
                if coordinator_index != index:
                    if coordinator is not None:
                        yield from emit(coordinator.finish())
                    coordinator = self.open_waveform_coordinator(descriptors[index][0])
                    coordinator_index = index
                assert coordinator is not None
                yield from emit(coordinator.push(task["row"], (ordinal, task)))
            if coordinator is not None:
                yield from emit(coordinator.finish())
                coordinator = None
        finally:
            primary = sys.exc_info()[1]
            cleanup_error = None
            coordinator = locals().get("coordinator")
            if coordinator is not None:
                try:
                    coordinator.abort(*sys.exc_info())
                except BaseException as exc:
                    cleanup_error = record_cleanup_error(
                        cleanup_error, exc, "Step3 MOS waveform coordinator",
                    )
            for iterator in iterators:
                close = getattr(iterator, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as exc:
                        cleanup_error = record_cleanup_error(
                            cleanup_error, exc, "Step3 tuple task iterator",
                        )
            if isinstance(primary, GeneratorExit) and cleanup_error is not None:
                raise cleanup_error
            if primary is not None:
                if cleanup_error is not None:
                    add_cleanup_note(primary, "Step3 tuple iterator cleanup", cleanup_error)
            elif cleanup_error is not None:
                raise cleanup_error

        tuple_attestations = []
        for models, _parts in descriptors:
            if self._compute_scores_batch is compute_mos_scores_batch:
                attestation = self.runtime_attestation(models)
            else:
                attestation = {
                    "selected_models": list(models),
                    "provider_validation": "injected_compute_scores_batch",
                }
            tuple_attestations.append({
                "models": list(models),
                "attestation": attestation,
            })
        return {
            "provider_class": "in_process",
            "device_class": "gpu" if self._options.use_gpu else "cpu",
            "model_tuple_attestations": tuple_attestations,
        }

    def _score_decoded_batch(
        self, decoded: Sequence[tuple[np.ndarray, int]], models: Sequence[str],
    ) -> list[dict[str, Any]]:
        """执行一个已拥有 mono waveform 的有界同 tuple 批次。"""
        raw_results = self._compute_scores_batch(
            decoded,
            mos_types=list(models),
            use_gpu=self._options.use_gpu,
            use_cache=self._options.use_cache,
            batch_size=len(decoded),
            model_specific_sr=True,
            allow_fallback=False,
            urgent_options=(_plain_json(self._options.urgent_options)
                           if self._options.urgent_options is not None else None),
            scoreq_options=(_plain_json(self._options.scoreq_options)
                           if self._options.scoreq_options is not None else None),
            dnsmos_options=(_plain_json(self._options.dnsmos_options)
                           if self._options.dnsmos_options is not None else None),
            nisqa_options=(_plain_json(self._options.nisqa_options)
                           if self._options.nisqa_options is not None else None),
            utmos_options=(_plain_json(self._options.utmos_options)
                          if self._options.utmos_options is not None else None),
        )
        if (isinstance(raw_results, (str, bytes))
                or not isinstance(raw_results, Sequence)
                or len(raw_results) != len(decoded)):
            raise ValueError("MOS batch result count must exactly match decoded intervals")
        return [self._map_result(raw, models) for raw in raw_results]

    def open_waveform_coordinator(
        self, models: Sequence[str],
    ) -> _Step3MosWaveformCoordinator:
        """打开一个仅限单模型 tuple 的跨 source waveform 批协调器。"""
        selected = self._selected_models(models)
        if self._compute_scores_batch is compute_mos_scores_batch:
            self.runtime_attestation(selected)
        return _Step3MosWaveformCoordinator(self, selected)

    def __call__(
        self, row: Mapping[str, Any], models: Sequence[str],
    ) -> Mapping[str, Any]:
        selected = self._selected_models(models)
        if self._compute_scores is compute_mos_scores:
            self.runtime_attestation(selected)
        _validate_source_binding(row)
        audio, sample_rate = self._decoder(row)
        if not isinstance(audio, np.ndarray) or audio.ndim not in (1, 2) or not audio.size:
            raise ValueError("segment decoder must return a nonempty numpy waveform")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("segment decoder must return a positive Python sample rate")
        start = row.get("start_sample")
        end = row.get("end_sample")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= audio.shape[0]:
            raise ValueError("Step2 segment interval is outside the decoded source")
        # Step3 仅分析第一通道；owned copy 阻断 decoder 与 estimator 的内存 alias。
        segment = _first_channel_owned_float32(audio[start:end])
        raw = self._compute_scores(
            audio_data=(segment, sample_rate),
            mos_types=list(selected),
            use_gpu=self._options.use_gpu,
            use_cache=self._options.use_cache,
            model_specific_sr=True,
            allow_fallback=False,
            urgent_options=(_plain_json(self._options.urgent_options)
                            if self._options.urgent_options is not None else None),
            scoreq_options=(_plain_json(self._options.scoreq_options)
                            if self._options.scoreq_options is not None else None),
            dnsmos_options=(_plain_json(self._options.dnsmos_options)
                            if self._options.dnsmos_options is not None else None),
            nisqa_options=(_plain_json(self._options.nisqa_options)
                           if self._options.nisqa_options is not None else None),
            utmos_options=(_plain_json(self._options.utmos_options)
                           if self._options.utmos_options is not None else None),
        )
        return self._map_result(raw, selected)

    @staticmethod
    def _map_result(raw: Any, models: Sequence[str]) -> dict[str, Any]:
        return map_mos_raw_result(raw, models)

    @staticmethod
    def source_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        """验证 Step2 source descriptor 并返回其 immutable identity。"""
        if not isinstance(row, Mapping):
            raise ValueError("Step3 MOS batch row must be a mapping")
        _validate_source_binding(row)
        try:
            identity = tuple(row[field] for field in _SOURCE_IDENTITY_FIELDS)
            start, end = row["start_sample"], row["end_sample"]
        except KeyError as exc:
            raise ValueError(f"Step2 row missing {exc.args[0]}") from exc
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= identity[5]:
            raise ValueError("Step2 segment interval is outside the verified source")
        return identity

    def open_source_group(
        self, row: Mapping[str, Any], models: Sequence[str],
    ) -> _Step3MosSourceGroup:
        """打开一个 source group；其 session 覆盖多个有界同源批次。"""
        selected = self._selected_models(models)
        if self._compute_scores_batch is compute_mos_scores_batch:
            self.runtime_attestation(selected)
        return _Step3MosSourceGroup(self, row, selected)

    def score_batch(
        self, rows: Sequence[Mapping[str, Any]], models: Sequence[str],
    ) -> list[dict[str, Any]]:
        """对一个有界、同 tuple row 批次跨 source 聚合 waveform 后按输入顺序返回。"""
        if (isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence)
                or not rows or len(rows) > _MAX_BATCH_ROWS):
            raise ValueError(f"Step3 MOS batch must contain 1 to {_MAX_BATCH_ROWS} rows")
        coordinator = self.open_waveform_coordinator(models)
        completed: dict[int, dict[str, Any]] = {}
        try:
            for index, row in enumerate(rows):
                for value, score in coordinator.push(row, index):
                    completed[value] = score
            for value, score in coordinator.finish():
                completed[value] = score
        except BaseException:
            coordinator.abort(*sys.exc_info())
            raise
        if set(completed) != set(range(len(rows))):
            raise ValueError("Step3 MOS batch results are incomplete")
        return [completed[index] for index in range(len(rows))]
