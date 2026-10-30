"""SSL 模型、runtime 字段与显式 ``ModelSpec`` 的合同校验。

SSL 同时承载项目原生 IPD 基线和若干官方模型复现路径。它们共享场景元数据，
但特征表示、pair 排布、时间压缩和 DP-IPD 到 DOA 的映射并不相同。本模块把这些
差异冻结在 ``ModelSpec``，避免根据模型名称或 tensor rank 猜测合同。
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

from core.models.io_contract import ModelIOContract


SSL_RUNTIME_INPUTS = (
    "features",
    "target_dpipd",
    "pair_index",
    "pair_mask",
    "geometry",
    "source_distances",
    "input_lengths",
    "output_lengths",
    "frame_origin",
    "time_stride",
    "input_representation",
    "target_representation",
    "coordinate_frame",
)
SSL_RUNTIME_OUTPUTS = ("dpipd", "vad_logits", "doa_logits", "doa_vector", "range")
SSL_MODEL_INPUTS = ("features", "multichannel_stft_real_imag", "geometry")
SSL_LOSS_FIELDS = (
    "target_dpipd",
    "doa",
    "vad",
    "pair_mask",
    "pair_index",
    "geometry",
    "source_distances",
    "input_lengths",
    "output_lengths",
    "frame_origin",
    "time_stride",
)


_FEATURE_CONTRACTS = frozenset({"multichannel_stft_real_imag", "reference_ipd"})
_TARGET_CONTRACTS = frozenset({"dpipd_pair"})
_PAIR_LAYOUTS = frozenset({"reference_to_all", "all_pairs", "pair_batched"})
_GEOMETRY_MODES = frozenset({"fixed_array", "variable_array"})
_FIELD_MODES = frozenset({"far_field", "near_field"})
_REDUCTIONS = frozenset(
    {"model_defined", "average_pool_12", "causal_conv_pool", "mamba_time_pool"}
)
_MAPPINGS = frozenset(
    {"far_field_grid", "near_field_doa_range_grid", "learned_range_head"}
)
_TARGET_LAYOUTS = frozenset({"trailing_ri", "freq_concat_ri"})
_OFFICIAL_MODEL_TYPES = frozenset(
    {"FNSSL", "IPDNetFixed", "IPDNetVariable", "IPDNet2"}
)
_NATIVE_MODEL_TYPES = frozenset({"native_ipd_baseline", "FixedArrayIPDNet"})
_MODEL_SPEC_FIELDS = frozenset(
    {
        "model_type",
        "feature_contract",
        "target_contract",
        "pair_layout",
        "geometry_mode",
        "field_mode",
        "reduction",
        "causal",
        "mapping",
        "target_layout",
    }
)
_SSL_AUXILIARY_OUTPUT_CONTRACTS = {
    "doa_logits": ("logits", ("batch", "frame", "track", "class")),
    "doa_vector": ("label", ("batch", "frame", "track", "angle")),
    "vad_logits": ("logits", ("batch", "frame", "track")),
    "range": ("label", ("batch", "frame", "track")),
}


@dataclass(frozen=True)
class ModelSpec:
    """SSL 模型的显式输入、目标、几何和时间合同。

    ``model_type`` 默认为 ``"unspecified"``，方便在独立的配置解析阶段先验证
    合同，再由 registry 的 ``model.type`` 注入具体名称。官方模型必须使用
    ``multichannel_stft_real_imag``；``reference_ipd`` 只允许项目原生 baseline。
    """

    feature_contract: str
    target_contract: str
    pair_layout: str
    geometry_mode: str
    field_mode: str
    reduction: str
    causal: bool
    mapping: str
    target_layout: str
    model_type: str = "unspecified"

    def __post_init__(self) -> None:
        values = {
            "feature_contract": (_FEATURE_CONTRACTS, self.feature_contract),
            "target_contract": (_TARGET_CONTRACTS, self.target_contract),
            "pair_layout": (_PAIR_LAYOUTS, self.pair_layout),
            "geometry_mode": (_GEOMETRY_MODES, self.geometry_mode),
            "field_mode": (_FIELD_MODES, self.field_mode),
            "reduction": (_REDUCTIONS, self.reduction),
            "mapping": (_MAPPINGS, self.mapping),
            "target_layout": (_TARGET_LAYOUTS, self.target_layout),
        }
        for name, (allowed, value) in values.items():
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"SSL ModelSpec.{name} must be one of {sorted(allowed)!r}")
        if type(self.causal) is not bool:
            raise TypeError("SSL ModelSpec.causal must be a bool")
        if not isinstance(self.model_type, str) or not self.model_type.strip():
            raise ValueError("SSL ModelSpec.model_type must be a non-empty string")
        if self.field_mode == "far_field" and self.mapping != "far_field_grid":
            raise ValueError("far-field SSL ModelSpec requires mapping='far_field_grid'")
        if self.field_mode == "near_field" and self.mapping == "far_field_grid":
            raise ValueError("near-field SSL ModelSpec requires a near-field mapping")
        if self.model_type in _OFFICIAL_MODEL_TYPES and self.feature_contract != "multichannel_stft_real_imag":
            raise ValueError(
                f"official SSL model {self.model_type!r} must use feature_contract='multichannel_stft_real_imag'"
            )
        if self.model_type in _NATIVE_MODEL_TYPES and self.feature_contract != "reference_ipd":
            raise ValueError(
                "native_ipd_baseline must use feature_contract='reference_ipd'; "
                "official models require multichannel_stft_real_imag"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, model_type: str | None = None) -> "ModelSpec":
        """从顶层 ``model_spec`` mapping 构造并严格拒绝未知键。"""
        if not isinstance(value, Mapping):
            raise TypeError("model_spec must be a mapping")
        unknown = set(value) - _MODEL_SPEC_FIELDS
        if unknown:
            raise ValueError(f"unknown model_spec keys: {sorted(unknown)}")
        data = dict(value)
        declared_type = data.get("model_type")
        if model_type is not None:
            if declared_type is not None and declared_type != model_type:
                raise ValueError(
                    "model_spec.model_type conflicts with model.type: "
                    f"{declared_type!r} != {model_type!r}"
                )
            data["model_type"] = model_type
        data.setdefault("model_type", "unspecified")
        missing = set(_MODEL_SPEC_FIELDS - set(data))
        missing.discard("model_type")
        if missing:
            raise ValueError(f"model_spec is missing required keys: {sorted(missing)}")
        return cls(**{key: data[key] for key in _MODEL_SPEC_FIELDS})

    def as_dict(self) -> dict[str, Any]:
        """返回可写入 resolved config/provenance 的普通 mapping。"""
        return asdict(self)


def parse_model_spec(value: ModelSpec | Mapping[str, Any], *, model_type: str | None = None) -> ModelSpec:
    """解析 ``ModelSpec`` 或配置 mapping。"""
    if isinstance(value, ModelSpec):
        if model_type is not None and value.model_type not in {"unspecified", model_type}:
            raise ValueError(
                "ModelSpec.model_type conflicts with model.type: "
                f"{value.model_type!r} != {model_type!r}"
            )
        if value.model_type == "unspecified" and model_type is not None:
            return ModelSpec(**{**value.as_dict(), "model_type": model_type})
        return value
    return ModelSpec.from_mapping(value, model_type=model_type)


def validate_model_spec(
    value: ModelSpec | Mapping[str, Any], *, model_type: str | None = None
) -> ModelSpec:
    """验证并返回 immutable ``ModelSpec``。"""
    return parse_model_spec(value, model_type=model_type)


# 允许 entry/runtime 使用更短的名称；解析逻辑仍只有一份。
resolve_model_spec = parse_model_spec


def _unwrap_model(model: Any) -> Any:
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "_orig_mod"):
            current = current._orig_mod
            continue
        break
    return current


def resolve_ssl_model_spec(model: Any, spec: ModelSpec | Mapping[str, Any] | None = None) -> ModelSpec:
    """解析模型类声明的 ``MODEL_SPEC``/``model_spec``，并保持 wrapper 透明。

    当调用方同时提供配置侧 ``model_spec`` 时，必须与模型类声明逐字段一致。
    否则配置可以把一个 proxy 的模型包装成另一个官方模型，直到第一次 forward
    才暴露输入、pair 或时间压缩不一致。
    """
    raw_model = _unwrap_model(model)
    declared = getattr(raw_model, "MODEL_SPEC", None)
    if declared is None:
        declared = getattr(raw_model, "model_spec", None)
    if declared is None and spec is None:
        raise ValueError(
            f"SSL model {type(raw_model).__name__} must declare MODEL_SPEC/model_spec"
        )
    model_type = getattr(raw_model, "MODEL_TYPE", None)
    if model_type is None and type(raw_model).__name__ in _OFFICIAL_MODEL_TYPES | _NATIVE_MODEL_TYPES:
        model_type = type(raw_model).__name__
    declared_spec = (
        None if declared is None else parse_model_spec(declared, model_type=model_type)
    )
    configured_spec = (
        None if spec is None else parse_model_spec(spec, model_type=model_type)
    )
    if declared_spec is not None and configured_spec is not None:
        if declared_spec.as_dict() != configured_spec.as_dict():
            raise ValueError(
                "configured SSL model_spec does not match the model's declared MODEL_SPEC: "
                f"declared={declared_spec.as_dict()!r}, configured={configured_spec.as_dict()!r}"
            )
    resolved = configured_spec or declared_spec
    if resolved is None:
        raise ValueError(
            f"SSL model {type(raw_model).__name__} must declare MODEL_SPEC/model_spec"
        )
    return resolved


def _validate_forward_signature(model: Any, input_names: tuple[str, ...]) -> None:
    signature = inspect.signature(model.forward)
    parameters = signature.parameters
    for name in input_names:
        parameter = parameters.get(name)
        if parameter is None:
            raise ValueError(
                f"SSL model forward must explicitly declare named input {name!r}; "
                "a **kwargs catch-all does not satisfy the model I/O contract"
            )
        if parameter is not None and parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            raise ValueError(f"SSL model forward input {name!r} must be named")


def _validate_model_contract(contract: ModelIOContract, spec: ModelSpec) -> None:
    if spec.target_layout != "trailing_ri":
        raise ValueError(
            "SSL public model I/O must use target_layout='trailing_ri'; "
            "freq_concat_ri is an internal implementation layout"
        )
    if contract.tasks != frozenset({"ssl"}):
        raise ValueError("SSL model contract must support only ssl")
    input_fields = {field.name: field for field in contract.inputs}
    if spec.feature_contract == "reference_ipd":
        field = input_fields.get("features")
        if field is None or field.domain != "embedding" or "feature" not in field.axes:
            raise ValueError("reference_ipd SSL models must declare embedding input 'features'")
    else:
        field = input_fields.get("multichannel_stft_real_imag")
        if field is None or field.domain != "spectrum" or "microphone" not in field.axes:
            raise ValueError(
                "multichannel_stft_real_imag SSL models must declare spectrum input "
                "'multichannel_stft_real_imag' with a microphone axis"
            )
    output_fields = {field.name: field for field in contract.outputs}
    legacy_outputs = {"prediction", "ipd_prediction"}.intersection(output_fields)
    if legacy_outputs:
        raise ValueError(
            "SSL model output fields are retired: "
            f"{sorted(legacy_outputs)!r}; declare canonical 'dpipd'"
        )
    dpipd = output_fields.get("dpipd")
    if dpipd is None or dpipd.domain != "spectrum":
        raise ValueError("SSL model must declare a spectrum output named 'dpipd'")
    if "frequency" not in dpipd.axes or "track" not in dpipd.axes or "complex" not in dpipd.axes:
        raise ValueError("SSL DP-IPD output must declare frequency, track and complex axes")
    unknown_outputs = set(output_fields) - {"dpipd", *_SSL_AUXILIARY_OUTPUT_CONTRACTS}
    if unknown_outputs:
        raise ValueError(
            "SSL model declares unsupported output fields: "
            f"{sorted(unknown_outputs)!r}"
        )
    for name, (domain, axes) in _SSL_AUXILIARY_OUTPUT_CONTRACTS.items():
        field = output_fields.get(name)
        if field is None:
            continue
        if field.domain != domain or field.axes != axes:
            raise ValueError(
                f"SSL model output {name!r} must declare domain={domain!r} "
                f"and axes={axes!r}"
            )


def validate_ssl_model_output(
    model: Any,
    output: Any,
    *,
    expected_shape,
    expected_device: torch.device | str | None = None,
) -> dict[str, Any]:
    """校验 forward 的实际字段和 shape 与声明的 SSL 输出合同一致。"""
    raw_model = _unwrap_model(model)
    contract = getattr(raw_model, "IO_CONTRACT", None)
    if not isinstance(contract, ModelIOContract):
        raise TypeError("SSL model must declare ModelIOContract before output validation")
    if isinstance(output, Mapping):
        fields = dict(output)
    else:
        fields = {"dpipd": output}
    legacy = {"prediction", "ipd_prediction"}.intersection(fields)
    if legacy:
        raise ValueError(
            "SSL model output rejects legacy prediction fields; use canonical 'dpipd'"
        )
    declared = {field.name: field for field in contract.outputs}
    unknown = set(fields) - set(declared)
    if unknown:
        raise ValueError(
            "SSL model returned undeclared output fields: "
            f"{sorted(unknown)!r}"
        )
    missing = {name for name, field in declared.items() if field.required and name not in fields}
    if missing:
        raise ValueError(f"SSL model omitted required output fields: {sorted(missing)!r}")
    if "dpipd" not in fields:
        raise ValueError("SSL model output must provide canonical 'dpipd'")
    expected_shape = tuple(expected_shape)
    if len(expected_shape) != 6 or expected_shape[-1] != 2:
        raise ValueError("expected SSL DP-IPD shape must be [B,T,F,K,P,2]")
    for name, value in fields.items():
        if not torch.is_tensor(value):
            raise TypeError(f"SSL model output {name!r} must be a floating-point tensor")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"SSL model output {name!r} must be finite floating-point")
    dpipd_device = fields["dpipd"].device
    for name, value in fields.items():
        if value.device != dpipd_device:
            raise ValueError(
                f"SSL model output {name!r} must share device {dpipd_device} with 'dpipd'"
            )
    if expected_device is not None:
        expected_device = torch.device(expected_device)
        if dpipd_device != expected_device:
            raise ValueError(
                "SSL dpipd output must be on the current feature device "
                f"{expected_device}, got {dpipd_device}"
            )
    if tuple(fields["dpipd"].shape) != expected_shape:
        raise ValueError(
            "SSL dpipd output shape must match canonical target shape "
            f"{expected_shape}, got {tuple(fields['dpipd'].shape)}"
        )
    batch, frames, _, tracks, _, _ = expected_shape
    expected_aux_shapes = {
        "doa_vector": (batch, frames, tracks, 2),
        "vad_logits": (batch, frames, tracks),
        "range": (batch, frames, tracks),
    }
    for name, shape in expected_aux_shapes.items():
        if name in fields and tuple(fields[name].shape) != shape:
            raise ValueError(f"SSL {name} output must have shape {shape}")
    if "doa_logits" in fields:
        logits = fields["doa_logits"]
        if logits.ndim != 4 or tuple(logits.shape[:3]) != (batch, frames, tracks):
            raise ValueError("SSL doa_logits output must have shape [B,T,K,Q]")
        if logits.shape[-1] < 2:
            raise ValueError("SSL doa_logits must contain at least two classes")
    return fields


def validate_ssl_loss_output_contract(
    model: Any,
    loss_config: Mapping[str, Any],
    *,
    target_representation: str,
) -> None:
    """拒绝当前模型合同没有声明的、已启用的 SSL auxiliary head。"""
    if not isinstance(loss_config, Mapping):
        raise TypeError("SSL loss_config must be a mapping")
    loss_type = loss_config.get("type")
    if loss_type not in {"SSLDOACompositeLoss", "SSLDPIPDPITLoss"}:
        return
    raw_args = loss_config.get("args")
    args = {} if raw_args is None else raw_args
    if not isinstance(args, Mapping):
        raise TypeError(f"{loss_type} args must be a mapping")
    raw_model = _unwrap_model(model)
    contract = getattr(raw_model, "IO_CONTRACT", None)
    if not isinstance(contract, ModelIOContract):
        raise TypeError("SSL model must declare ModelIOContract before loss validation")
    outputs = {field.name for field in contract.outputs}
    required_outputs = (
        {"doa_loss_weight": "doa_vector"}
        if loss_type == "SSLDPIPDPITLoss"
        else {
            "doa_classification_weight": "doa_logits",
            "doa_regression_weight": "doa_vector",
            "vad_weight": "vad_logits",
            "range_weight": "range",
        }
    )
    for weight_name, output_name in required_outputs.items():
        weight = args.get(weight_name, 0.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError(f"SSL {weight_name} must be numeric")
        if not torch.isfinite(torch.tensor(float(weight))).item() or float(weight) < 0.0:
            raise ValueError(f"SSL {weight_name} must be finite and non-negative")
        if float(weight) > 0.0 and output_name not in outputs:
            raise ValueError(
                f"SSL loss enables {weight_name}, but model {type(raw_model).__name__} "
                f"does not declare output head {output_name!r}"
            )
    if (
        loss_type == "SSLDOACompositeLoss"
        and float(args.get("range_weight", 0.0)) > 0.0
        and target_representation != "dpipd_near_field"
    ):
        raise ValueError("SSL range_weight requires target_representation='dpipd_near_field'")


def validate_ssl_model_io(model, model_spec: ModelSpec | Mapping[str, Any] | None = None) -> ModelIOContract:
    """验证 SSL ``ModelIOContract``、``ModelSpec`` 与 ``forward`` 参数的一致性。"""
    raw_model = _unwrap_model(model)
    contract = getattr(raw_model, "IO_CONTRACT", None)
    if not isinstance(contract, ModelIOContract):
        raise TypeError("SSL model must declare ModelIOContract")
    spec = resolve_ssl_model_spec(raw_model, model_spec)
    _validate_model_contract(contract, spec)
    expected_input = (
        ("features",)
        if spec.feature_contract == "reference_ipd"
        else ("multichannel_stft_real_imag",)
    )
    _validate_forward_signature(raw_model, expected_input)
    return contract


__all__ = [
    "ModelSpec",
    "SSL_LOSS_FIELDS",
    "SSL_MODEL_INPUTS",
    "SSL_RUNTIME_INPUTS",
    "SSL_RUNTIME_OUTPUTS",
    "parse_model_spec",
    "resolve_model_spec",
    "resolve_ssl_model_spec",
    "validate_model_spec",
    "validate_ssl_model_io",
    "validate_ssl_model_output",
    "validate_ssl_loss_output_contract",
]
