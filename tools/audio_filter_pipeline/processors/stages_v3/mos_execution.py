"""Generation-3 Step3 MOS 的纯执行规划。

本模块只把已解析配置投影为不可变计划；不会检查资产是否存在、读取或哈希
文件、导入或加载模型，也不会启动 worker。资产验证和执行仍属于既有 runtime
边界。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..mos_contract import (
    MOS_ASSET_PATH_FIELDS,
    MOS_ASSET_SETTING_FIELDS,
    MOS_MODEL_ORDER,
)


__all__ = ["MosAssetSpec", "MosExecutionPlan"]

_SUPPORTED_MODELS = MOS_MODEL_ORDER
_MODEL_SCHEMAS = MappingProxyType({
    model: (MOS_ASSET_PATH_FIELDS[model], MOS_ASSET_SETTING_FIELDS[model])
    for model in MOS_MODEL_ORDER
})
_DECISION_ONLY_FIELDS = frozenset({
    "threshold", "thresholds", "mos_threshold", "mos_thresholds", "speaker",
    "speaker_consistency", "quorum", "min_mos_models", "fusion", "mos_fusion",
    "early_exit",
})
_CPU_TOPOLOGY_FIELDS = frozenset({"kind", "device"})
_GPU_TOPOLOGY_FIELDS = frozenset({
    "kind", "device", "gpu_ids", "worker_idle_timeout_seconds",
    "worker_hard_timeout_seconds", "worker_stderr_tail_bytes", "cpu_fallback",
})


def _freeze_json(value: Any, label: str) -> Any:
    """校验并递归冻结严格 JSON 值；对象 key 不做字符串化。"""
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


def _plain(value: Any) -> Any:
    """返回 JSON-safe 副本，避免暴露计划内部的只读容器。"""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _exact_fields(value: Mapping[Any, Any], expected: frozenset[str], label: str) -> None:
    if any(type(key) is not str for key in value):
        raise TypeError(f"{label} field names must be strings")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields must be exactly {sorted(expected)!r}; "
            f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}"
        )


def _required_string(value: Any, label: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} must not be blank")


def _validate_urgent_pk_score_range(value: Any, label: str) -> None:
    """按 runtime 归一化契约校验 URGENT-PK 分数范围，不执行归一化。"""
    if type(value) is str:
        if value == "mos_1_to_5":
            return
        raise ValueError(f"{label} has an unsupported string value")
    if value is None or type(value) is bool:
        raise ValueError(f"{label} must define a score range")
    if isinstance(value, Mapping):
        _exact_fields(value, frozenset({"min", "max"}), label)
        endpoints = (value["min"], value["max"])
    elif type(value) in {list, tuple} and len(value) == 2:
        endpoints = (value[0], value[1])
    else:
        raise ValueError(
            f"{label} must be 'mos_1_to_5', a two-item range, or an exact "
            "{'min', 'max'} mapping"
        )
    if any(endpoint is None or type(endpoint) is bool for endpoint in endpoints):
        raise ValueError(f"{label} endpoints must be numbers")
    try:
        minimum, maximum = (float(endpoint) for endpoint in endpoints)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} endpoints must be numbers") from exc
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
        raise ValueError(f"{label} must be finite and strictly increasing")


def _public_value(config: Any, name: str, default: Any = None) -> Any:
    """只读取指定模型或 topology 的公开配置属性。"""
    return getattr(config, name, default)


@dataclass(frozen=True)
class MosAssetSpec:
    """一个 selected MOS model 的严格、不可变配置级资产声明。"""

    model: str
    paths: Mapping[str, str]
    settings: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if type(self.model) is not str or self.model not in _SUPPORTED_MODELS:
            raise ValueError(f"unsupported MOS model: {self.model!r}")
        if not isinstance(self.paths, Mapping) or not isinstance(self.settings, Mapping):
            raise TypeError("MOS asset paths and settings must be mappings")
        path_fields, setting_fields = _MODEL_SCHEMAS[self.model]
        _exact_fields(self.paths, path_fields, f"{self.model}.paths")
        _exact_fields(self.settings, setting_fields, f"{self.model}.settings")
        decision_fields = _all_keys(self.settings) & _DECISION_ONLY_FIELDS
        if decision_fields:
            raise ValueError(
                f"{self.model}.settings contains decision-only fields: "
                f"{sorted(decision_fields)!r}"
            )
        for name, value in self.paths.items():
            _required_string(value, f"{self.model}.paths.{name}")
        for name, value in self.settings.items():
            label = f"{self.model}.settings.{name}"
            if name == "score_range":
                _validate_urgent_pk_score_range(value, label)
            else:
                _required_string(value, label)
        object.__setattr__(self, "paths", _freeze_json(self.paths, f"{self.model}.paths"))
        object.__setattr__(
            self, "settings", _freeze_json(self.settings, f"{self.model}.settings"),
        )

    def as_mapping(self) -> Mapping[str, Any]:
        """返回严格 JSON-safe 的配置级资产副本。"""
        return {
            "model": self.model,
            "paths": _plain(self.paths),
            "settings": _plain(self.settings),
        }


def _asset_spec(config: Any, model: str) -> MosAssetSpec:
    if model == "scoreq":
        return MosAssetSpec(
            model, {"model_dir": _public_value(config, "scoreq_model_dir")},
        )
    if model == "dnsmos":
        return MosAssetSpec(
            model, {"model_dir": _public_value(config, "dnsmos_model_dir")},
        )
    if model == "urgent_pk":
        return MosAssetSpec(
            model,
            {
                "checkpoint": _public_value(config, "urgent_pk_checkpoint"),
                "source_dir": _public_value(config, "urgent_pk_source_dir"),
            },
            {
                "encoder": _public_value(config, "urgent_pk_encoder"),
                "backbone": _public_value(config, "urgent_pk_backbone"),
                "score_range": _public_value(config, "urgent_pk_score_range"),
                "expected_sha256": _public_value(config, "urgent_pk_expected_sha256"),
                "expected_commit": _public_value(config, "urgent_pk_expected_commit"),
            },
        )
    return MosAssetSpec(
        model,
        {"model_path": _public_value(config, f"{model}_model_path")},
        {
            "model_version": _public_value(config, f"{model}_model_version"),
            "model_sha256": _public_value(config, f"{model}_model_sha256"),
        },
    )


def _selected_models(policy_source: Any) -> tuple[str, ...]:
    selector = getattr(policy_source, "selected_mos_models", None)
    if not callable(selector):
        raise TypeError("policy source must expose public selected_mos_models()")
    raw = selector()
    if type(raw) is not tuple:
        raise TypeError("selected_mos_models() must return a tuple")
    if (
        not raw
        or any(type(model) is not str or model not in _SUPPORTED_MODELS for model in raw)
        or len(set(raw)) != len(raw)
    ):
        raise ValueError("selected_mos_models() returned an invalid model sequence")
    return raw


def _validated_topology(topology: Mapping[Any, Any]) -> Mapping[str, Any]:
    if any(type(key) is not str for key in topology):
        raise TypeError("topology field names must be strings")
    kind = topology.get("kind")
    if kind == "single_process":
        _exact_fields(topology, _CPU_TOPOLOGY_FIELDS, "CPU topology")
        if topology["device"] != "cpu":
            raise ValueError("CPU topology device must be 'cpu'")
        return _freeze_json(dict(topology), "topology")
    if kind != "gpu_process_pool":
        raise ValueError("topology kind is unsupported")
    _exact_fields(topology, _GPU_TOPOLOGY_FIELDS, "GPU topology")
    if topology["device"] != "cuda" or topology["cpu_fallback"] is not False:
        raise ValueError("GPU topology requires cuda with cpu_fallback disabled")
    gpu_ids = topology["gpu_ids"]
    if (
        type(gpu_ids) is not list
        or not gpu_ids
        or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError("GPU topology gpu_ids must be unique nonnegative integers")
    idle_timeout = topology["worker_idle_timeout_seconds"]
    if (
        type(idle_timeout) not in {int, float}
        or not math.isfinite(float(idle_timeout))
        or float(idle_timeout) <= 0.0
    ):
        raise ValueError(
            "GPU topology worker_idle_timeout_seconds must be positive and finite"
        )
    hard_timeout = topology["worker_hard_timeout_seconds"]
    if hard_timeout is not None and (
        type(hard_timeout) not in {int, float}
        or not math.isfinite(float(hard_timeout))
        or float(hard_timeout) <= 0.0
    ):
        raise ValueError(
            "GPU topology worker_hard_timeout_seconds must be positive and finite or None"
        )
    stderr_bytes = topology["worker_stderr_tail_bytes"]
    if type(stderr_bytes) is not int or not 0 < stderr_bytes <= 65_536:
        raise ValueError("GPU topology worker_stderr_tail_bytes must be in [1, 65536]")
    canonical = dict(topology)
    canonical["gpu_ids"] = list(gpu_ids)
    canonical["worker_idle_timeout_seconds"] = float(idle_timeout)
    canonical["worker_hard_timeout_seconds"] = (
        None if hard_timeout is None else float(hard_timeout)
    )
    return _freeze_json(canonical, "topology")


@dataclass(frozen=True)
class MosExecutionPlan:
    """selected-only 资产声明与执行 topology 的严格不可变计划。"""

    selected_models: tuple[str, ...]
    assets: tuple[MosAssetSpec, ...]
    topology: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.selected_models) is not tuple:
            raise TypeError("selected_models must be a tuple")
        selected = self.selected_models
        if (
            not selected
            or any(type(model) is not str or model not in _SUPPORTED_MODELS for model in selected)
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("selected_models must be a unique supported model tuple")
        if type(self.assets) is not tuple:
            raise TypeError("assets must be a tuple")
        if any(type(asset) is not MosAssetSpec for asset in self.assets):
            raise TypeError("assets must contain only MosAssetSpec values")
        if tuple(asset.model for asset in self.assets) != selected:
            raise ValueError("assets must exactly match selected_models in order")
        if not isinstance(self.topology, Mapping):
            raise TypeError("topology must be a mapping")
        object.__setattr__(self, "topology", _validated_topology(self.topology))

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        use_gpu: bool = False,
        policy_book: Any | None = None,
    ) -> "MosExecutionPlan":
        """从公开配置与策略来源构建 selected-only 计划，不执行外部操作。"""
        if type(use_gpu) is not bool:
            raise TypeError("use_gpu must be bool")
        selected = _selected_models(policy_book if policy_book is not None else config)
        assets = tuple(_asset_spec(config, model) for model in selected)
        if not use_gpu:
            topology = {"kind": "single_process", "device": "cpu"}
        else:
            configured_gpu_ids = _public_value(config, "step3_gpu_ids", ())
            if type(configured_gpu_ids) not in {list, tuple}:
                raise TypeError("step3_gpu_ids must be a list or tuple")
            topology = {
                "kind": "gpu_process_pool",
                "device": "cuda",
                "gpu_ids": list(configured_gpu_ids),
                "worker_idle_timeout_seconds": _public_value(
                    config, "step3_worker_idle_timeout_seconds", 900.0,
                ),
                "worker_hard_timeout_seconds": _public_value(
                    config, "step3_worker_hard_timeout_seconds",
                ),
                "worker_stderr_tail_bytes": _public_value(
                    config, "step3_worker_stderr_tail_bytes", 4_096,
                ),
                "cpu_fallback": False,
            }
        return cls(selected_models=selected, assets=assets, topology=topology)


def _all_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(key)
            keys.update(_all_keys(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            keys.update(_all_keys(nested))
    return keys
