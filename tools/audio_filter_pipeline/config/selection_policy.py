"""Generation-3 selection policy 的纯解析、路由与哈希实现。

本模块只处理内存中的配置值，不访问 filesystem；``PipelineConfig`` 负责把
数据集 registry 投影为 ``SelectionRoute``，并保留原有 public facade。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import posixpath
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple


QUALITY_PROFILES = ("quality_balanced", "quality_high", "quality_premium")
_SUPPORTED_MOS_FUSION = "min"


def freeze_policy_value(value: Any) -> Any:
    """递归冻结 resolved policy，阻断嵌套可变值绕过只读边界。"""
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): freeze_policy_value(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(freeze_policy_value(item) for item in value)
    return value


def freeze_snapshot_value(value: Any) -> Any:
    """递归冻结 canonical policy snapshot。"""
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): freeze_snapshot_value(item) for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(freeze_snapshot_value(item) for item in value)
    return value


def canonical_policy_value(value: Any) -> Any:
    """把嵌套策略值转为确定性 JSON 值，禁止依赖对象 identity/hash。"""
    if isinstance(value, Mapping):
        return {
            str(key): canonical_policy_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_policy_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        "selection policy value is not canonically serializable: "
        f"{type(value).__name__}"
    )


def canonical_source_route_path(value: Any, label: str) -> str:
    """规范化 exact-source route，拒绝绝对路径、回退路径与平台歧义。"""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{label} must be a non-empty string without surrounding whitespace"
        )
    if "\\" in value or "\x00" in value or posixpath.isabs(value):
        raise ValueError(f"{label} must be a safe POSIX relative path")
    if ".." in value.split("/"):
        raise ValueError(f"{label} must not contain '..' path segments")
    canonical = posixpath.normpath(value)
    if canonical in {"", "."} or canonical.startswith("../"):
        raise ValueError(f"{label} must identify a file below the dataset root")
    return canonical


def validate_canonical_policy_json(value: Any, path: str) -> None:
    """拒绝会被字符串化碰撞或无法稳定 JSON 编码的共享 policy 值。"""
    if isinstance(value, Mapping):
        if any(type(key) is not str or not key.strip() for key in value):
            raise ValueError(f"{path} object keys must be non-empty strings")
        for key, nested in value.items():
            validate_canonical_policy_json(nested, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            validate_canonical_policy_json(nested, f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} numbers must be finite")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return
    raise ValueError(
        f"{path} must contain only canonical JSON values, got "
        f"{type(value).__name__}"
    )


def canonical_quality_profiles(
    profiles: Any,
    *,
    get_mos_spec: Callable[[str], Any],
) -> Dict[str, Any]:
    """规范化固定 profile slug，并保留原始 threshold 数值类型。"""
    if not isinstance(profiles, Mapping):
        raise ValueError("quality_profiles must be a mapping")
    for name in profiles:
        if type(name) is not str or not name.strip():
            raise ValueError("quality_profiles names must be non-empty strings")
    unknown = set(profiles) - set(QUALITY_PROFILES)
    if unknown:
        raise ValueError(
            f"quality_profiles has unknown profile names: {sorted(unknown)!r}"
        )
    missing = set(QUALITY_PROFILES) - set(profiles)
    if missing:
        raise ValueError(
            f"quality_profiles must define all profiles, missing: {sorted(missing)!r}"
        )
    canonical = {}
    for profile in QUALITY_PROFILES:
        body = profiles[profile]
        if not isinstance(body, Mapping):
            raise ValueError(f"quality_profiles.{profile} must be a mapping")
        for field in body:
            if type(field) is not str or not field.strip():
                raise ValueError(
                    f"quality_profiles.{profile} field names must be non-empty strings"
                )
        extra = set(body) - {"thresholds"}
        if extra:
            raise ValueError(
                f"quality_profiles.{profile} allows only thresholds, "
                f"got {sorted(extra)!r}"
            )
        thresholds = body.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise ValueError(f"quality_profiles.{profile}.thresholds must be a mapping")
        canonical_thresholds = {}
        normalized_names = set()
        for model, value in thresholds.items():
            if type(model) is not str or not model.strip():
                raise ValueError(
                    f"quality_profiles.{profile}.thresholds model names must be "
                    "nonblank strings"
                )
            normalized = model.strip().lower()
            if normalized in normalized_names:
                raise ValueError(
                    f"quality_profiles.{profile}.thresholds has duplicate model "
                    f"{normalized!r}"
                )
            try:
                spec = get_mos_spec(normalized)
            except ValueError as exc:
                raise ValueError(
                    f"quality_profiles.{profile}.thresholds has unknown model "
                    f"{normalized!r}"
                ) from exc
            if not spec.selectable:
                raise ValueError(
                    "quality_profiles."
                    f"{profile}.thresholds model is not selectable: {normalized}"
                )
            dimensions = tuple(spec.validate_threshold or ())
            if dimensions:
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"quality_profiles.{profile}.thresholds.{normalized} "
                        "must be a mapping"
                    )
                unknown_dimensions = set(value) - set(dimensions)
                if unknown_dimensions:
                    raise ValueError(
                        f"quality_profiles.{profile}.thresholds.{normalized} has "
                        f"unknown dimensions: {sorted(unknown_dimensions)!r}"
                    )
                missing_dimensions = set(dimensions) - set(value)
                if missing_dimensions:
                    raise ValueError(
                        f"quality_profiles.{profile}.thresholds.{normalized} missing "
                        f"dimensions: {sorted(missing_dimensions)!r}"
                    )
            validate_canonical_policy_json(
                value, f"quality_profiles.{profile}.thresholds.{normalized}"
            )
            canonical_thresholds[normalized] = canonical_policy_value(value)
            normalized_names.add(normalized)
        if not canonical_thresholds:
            raise ValueError(f"quality_profiles.{profile}.thresholds must be nonempty")
        canonical[profile] = {"thresholds": canonical_thresholds}
    return canonical


def normalize_language(language: Optional[str]) -> str:
    """稳定规范化语言标签。"""
    value = str(language or "").strip().lower().replace("_", "-")
    return value.split("-", 1)[0] if value else ""


@dataclass(frozen=True)
class SelectionRoute:
    """selection policy 解析所需的数据集路由身份。"""

    dataset_id: str
    language: str
    category: Optional[str]
    quality_profile: Optional[str]
    quality_profile_explicit: bool = True


@dataclass(frozen=True)
class ResolvedSelectionPolicy:
    """完成所有策略层合并后的不可变 selection policy。"""

    policy_id: str
    category: Optional[str]
    quality_profile: Optional[str]
    mos_models: Tuple[str, ...]
    thresholds: Mapping[str, Any]
    min_mos_models: int
    mos_fusion: str
    early_exit: Any
    policy_hash: str


@dataclass(frozen=True)
class MosScoringProjection:
    """只绑定 Step3 exact route 与模型顺序的 threshold-free 投影。"""

    dataset_id: str
    language: str
    category: Optional[str]
    quality_profile: Optional[str]
    route_identity: Optional[Mapping[str, str]]
    mos_models: Tuple[str, ...]
    scoring_policy_hash: str

    def as_mapping(self) -> Mapping[str, Any]:
        """按既有 facade 契约返回只读外层 mapping。"""
        route_identity = None
        if self.route_identity is not None:
            route_identity = dict(self.route_identity)
        return MappingProxyType({
            "dataset_id": self.dataset_id,
            "language": self.language,
            "category": self.category,
            "quality_profile": self.quality_profile,
            "route_identity": route_identity,
            "mos_models": list(self.mos_models),
            "scoring_policy_hash": self.scoring_policy_hash,
        })


class SelectionPolicyBook:
    """Canonical selection policy snapshot 及其纯路由解析器。"""

    _ALLOWED_LAYERS = (
        "global", "categories", "languages", "datasets", "quality_profiles",
        "sources",
    )

    def __init__(
        self,
        raw_policies: Any,
        *,
        routes: Mapping[str, SelectionRoute],
        serialized_datasets: Mapping[str, Any],
        default_policy: Mapping[str, Any],
        get_mos_spec: Callable[[str], Any],
    ) -> None:
        self._routes = MappingProxyType(dict(routes))
        self._serialized_datasets = freeze_snapshot_value(
            canonical_policy_value(serialized_datasets)
        )
        self._get_mos_spec = get_mos_spec
        self._snapshot = self._canonicalize(raw_policies, default_policy)
        dependency_state = {
            "datasets": self._serialized_datasets,
            "selection_policies": self._snapshot,
        }
        dependency_payload = json.dumps(
            canonical_policy_value(dependency_state),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._namespace = hashlib.sha256(
            dependency_payload.encode("utf-8")
        ).hexdigest()
        self._resolve_cache: Dict[Tuple[Any, ...], ResolvedSelectionPolicy] = {}

    @property
    def snapshot(self) -> Mapping[str, Any]:
        """返回构造时固定的 canonical snapshot。"""
        return self._snapshot

    def serialized(self) -> Dict[str, Any]:
        """返回 JSON-safe canonical snapshot 副本。"""
        return canonical_policy_value(self._snapshot)

    def validate_structure(self) -> None:
        """确认 snapshot 仍属于本实例；构造阶段已完成完整结构校验。"""
        if not isinstance(self._snapshot, Mapping):
            raise ValueError("selection_policies must be a mapping")

    def validate_semantics(self) -> None:
        """校验全部生产可达 baseline 与 exact-source resolved policy 语义。"""
        def configured_policies() -> Tuple[Tuple[str, Mapping[str, Any]], ...]:
            configured = [("selection_policies.global", self._snapshot["global"])]
            for layer in ("categories", "languages", "datasets"):
                for target, policy in self._snapshot.get(layer, {}).items():
                    configured.append((f"selection_policies.{layer}.{target}", policy))
            for dataset_id, routes in self._snapshot.get("sources", {}).items():
                for source_path, policy in routes.items():
                    configured.append((
                        f"selection_policies.sources.{dataset_id}.{source_path}",
                        policy,
                    ))
            return tuple(configured)

        def validate_configured_controls() -> None:
            for where, configured in configured_policies():
                fusion = configured.get("mos_fusion")
                if fusion is not None and fusion != _SUPPORTED_MOS_FUSION:
                    raise ValueError(
                        f"{where}.mos_fusion must be 'min'; generation 3 uses "
                        "per-model thresholds with K/N decisions"
                    )
                early_exit = configured.get("early_exit")
                if (
                    isinstance(early_exit, Mapping)
                    and early_exit.get("enabled") is True
                ):
                    raise ValueError(
                        f"{where}.early_exit.enabled=true is not implemented for "
                        "generation 3"
                    )

        def validate_policy(policy: ResolvedSelectionPolicy, where: str) -> None:
            if not policy.mos_models:
                raise ValueError("resolved selection policy mos_models must be non-empty")
            if policy.min_mos_models > len(policy.mos_models):
                raise ValueError("min_mos_models must be <= number of MOS models")
            if policy.mos_fusion != _SUPPORTED_MOS_FUSION:
                raise ValueError(
                    "resolved selection policy mos_fusion must be 'min'; "
                    "generation 3 uses per-model thresholds with K/N decisions"
                )

            specs = {}
            for model in policy.mos_models:
                try:
                    spec = self._get_mos_spec(model)
                except ValueError as exc:
                    raise ValueError(f"unknown MOS type: {model}") from exc
                if not spec.selectable:
                    raise ValueError(f"MOS type is not selectable: {model}")
                specs[model] = spec

            for model, value in policy.thresholds.items():
                try:
                    spec = specs.get(model) or self._get_mos_spec(model)
                except ValueError as exc:
                    raise ValueError(f"unknown MOS type: {model}") from exc
                threshold_where = f"{where}.thresholds[{model!r}]"
                if spec.validate_threshold:
                    # 维度型模型（如 dnsmos/urgent_pk）要求含全部维度的 mapping，
                    # 与运行期 mos.validate_mos_threshold 的合同一致。
                    if not isinstance(value, Mapping):
                        raise ValueError(
                            f"{threshold_where} 阈值必须使用含全部维度的 mapping，"
                            f"合法维度: {'/'.join(spec.validate_threshold)}"
                        )
                    if not value:
                        raise ValueError(
                            f"{threshold_where} dict 不能为空（至少给出一个维度阈值）"
                        )
                    for dimension, dimension_value in value.items():
                        if dimension not in spec.validate_threshold:
                            raise ValueError(
                                f"{threshold_where} 未知维度键 {dimension!r}，合法维度: "
                                f"{'/'.join(spec.validate_threshold)}"
                            )
                        if type(dimension_value) not in (int, float) or not (
                            1.0 <= float(dimension_value) <= 5.0
                        ):
                            raise ValueError(
                                f"{threshold_where} 维度 {dimension} 阈值必须在 "
                                f"[1.0, 5.0]，got {dimension_value!r}"
                            )
                elif type(value) not in (int, float) or not (
                    1.0 <= float(value) <= 5.0
                ):
                    raise ValueError(
                        f"{threshold_where} 阈值必须在范围 [1.0, 5.0]，got {value!r}"
                    )

            early_exit = policy.early_exit
            if early_exit is None:
                return
            if not isinstance(early_exit, Mapping):
                raise ValueError(f"{where}.early_exit must be None or a mapping")
            unknown_early_fields = set(early_exit) - {"enabled", "threshold", "order"}
            if unknown_early_fields:
                raise ValueError(
                    f"unknown early_exit fields in {where}: "
                    f"{sorted(unknown_early_fields)!r}"
                )
            enabled = early_exit.get("enabled", False)
            if type(enabled) is not bool:
                raise ValueError(f"{where}.early_exit.enabled must be bool")
            threshold = early_exit.get("threshold")
            if threshold is not None and not (
                type(threshold) in (int, float)
                and 1.0 <= float(threshold) <= 5.0
            ) and not (
                type(threshold) is str and threshold.strip().lower() == "dnsmos"
            ):
                raise ValueError(
                    f"{where}.early_exit.threshold must be a number in [1.0, 5.0], "
                    "'dnsmos', or None"
                )
            order = early_exit.get("order")
            if order is not None:
                if not isinstance(order, (list, tuple)) or any(
                    type(model) is not str or not model.strip() for model in order
                ):
                    raise ValueError(
                        f"{where}.early_exit.order must be a list of model names"
                    )
                normalized_order = tuple(model.strip().lower() for model in order)
                if len(set(normalized_order)) != len(normalized_order):
                    raise ValueError(f"{where}.early_exit.order has duplicate models")
                unknown_order_models = set(normalized_order) - set(policy.mos_models)
                if unknown_order_models:
                    raise ValueError(
                        f"{where}.early_exit.order contains unknown policy models: "
                        f"{sorted(unknown_order_models)!r}"
                    )
            if enabled:
                raise ValueError(
                    f"{where}.early_exit.enabled=true is not implemented for "
                    "generation 3"
                )

        if not self._routes:
            global_policy = self._snapshot["global"]
            validate_policy(
                ResolvedSelectionPolicy(
                    "global",
                    category=None,
                    quality_profile=None,
                    mos_models=tuple(global_policy["mos_models"]),
                    thresholds=global_policy["thresholds"],
                    min_mos_models=global_policy["min_mos_models"],
                    mos_fusion=global_policy["mos_fusion"],
                    early_exit=global_policy["early_exit"],
                    policy_hash="",
                ),
                "selection_policies.global",
            )
            validate_configured_controls()
            return

        source_routes = self._snapshot.get("sources", {})
        for dataset_id, route in self._routes.items():
            validate_policy(
                self.resolve(dataset_id, route.language),
                f"resolved selection policy {dataset_id!r}/{route.language!r}",
            )
            for source_path in source_routes.get(dataset_id, {}):
                validate_policy(
                    self.resolve(
                        dataset_id,
                        route.language,
                        source_relative_path=source_path,
                    ),
                    "resolved source selection policy "
                    f"{dataset_id!r}/{route.language!r}/{source_path!r}",
                )
        validate_configured_controls()

    def selected_mos_models(self) -> Tuple[str, ...]:
        """返回生产可达 baseline 与 exact-source policy 的稳定模型并集。"""
        selected: Dict[str, None] = {}
        source_routes = self._snapshot.get("sources", {})
        for dataset_id, route in self._routes.items():
            for model in self.resolve(dataset_id, route.language).mos_models:
                selected.setdefault(model, None)
            for source_path in source_routes.get(dataset_id, {}):
                for model in self.resolve(
                    dataset_id,
                    route.language,
                    source_relative_path=source_path,
                ).mos_models:
                    selected.setdefault(model, None)
        return tuple(selected)

    def _canonicalize(
        self, raw_policies: Any, default_policy: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if raw_policies is not None and not isinstance(raw_policies, Mapping):
            raise ValueError("selection_policies must be a mapping")
        policies = copy.deepcopy(raw_policies or {})
        for layer in policies:
            if type(layer) is not str or not layer.strip():
                raise ValueError(
                    "selection policy layer names must be non-empty strings"
                )
        unknown_layers = set(policies) - set(self._ALLOWED_LAYERS)
        if unknown_layers:
            raise ValueError(
                f"unknown selection policy key: {sorted(unknown_layers)!r}"
            )
        if any(layer != "global" for layer in policies) and not self._routes:
            raise ValueError(
                "datasets must be configured for non-global selection_policies"
            )
        if "quality_profiles" not in policies and self._routes:
            raise ValueError(
                "datasets require selection_policies.quality_profiles defining "
                "all quality profiles"
            )

        known_categories = {
            route.category
            for route in self._routes.values()
            if route.category is not None
        }

        def canonical_policy(policy: Any, path: str) -> Dict[str, Any]:
            if not isinstance(policy, Mapping):
                raise ValueError(f"selection policy {path} must be a mapping")
            allowed_fields = {
                "models", "mos_models", "thresholds", "min_mos_models",
                "mos_fusion", "early_exit",
            }
            for field in policy:
                if type(field) is not str or not field.strip():
                    raise ValueError(
                        f"selection policy {path} field names must be non-empty strings"
                    )
            extra = set(policy) - allowed_fields
            if extra:
                raise ValueError(f"unknown policy key in {path}: {sorted(extra)!r}")
            if "models" in policy and "mos_models" in policy:
                raise ValueError(
                    f"selection policy {path} cannot define both models and mos_models"
                )

            canonical = {}
            models = policy.get("mos_models", policy.get("models"))
            if models is not None:
                if not isinstance(models, (list, tuple)):
                    raise ValueError(
                        f"selection policy {path} mos_models must be a list or tuple"
                    )
                normalized_models = []
                seen_models = set()
                for model in models:
                    if type(model) is not str or not model.strip():
                        raise ValueError(
                            f"selection policy {path} mos_models entries must be "
                            "non-empty strings"
                        )
                    normalized = model.strip().lower()
                    if normalized in seen_models:
                        raise ValueError(
                            f"selection policy {path} has duplicate model after "
                            f"normalization: {normalized!r}"
                        )
                    seen_models.add(normalized)
                    normalized_models.append(normalized)
                canonical["mos_models"] = normalized_models

            if "thresholds" in policy:
                thresholds = policy["thresholds"]
                if not isinstance(thresholds, Mapping):
                    raise ValueError(
                        f"selection policy {path} thresholds must be a mapping"
                    )
                normalized_thresholds = {}
                for model, value in thresholds.items():
                    if type(model) is not str or not model.strip():
                        raise ValueError(
                            f"selection policy {path} threshold model names must be "
                            "non-empty strings"
                        )
                    normalized = model.strip().lower()
                    if normalized in normalized_thresholds:
                        raise ValueError(
                            f"selection policy {path} has duplicate threshold model "
                            f"after normalization: {normalized!r}"
                        )
                    validate_canonical_policy_json(
                        value, f"selection policy {path}.thresholds.{normalized}"
                    )
                    normalized_thresholds[normalized] = canonical_policy_value(value)
                canonical["thresholds"] = {
                    model: normalized_thresholds[model]
                    for model in sorted(normalized_thresholds)
                }

            if "min_mos_models" in policy:
                minimum = policy["min_mos_models"]
                if type(minimum) is not int or minimum < 1:
                    raise ValueError(
                        f"selection policy {path} min_mos_models must be an integer >= 1"
                    )
                canonical["min_mos_models"] = minimum

            if "mos_fusion" in policy:
                fusion = policy["mos_fusion"]
                if type(fusion) is not str or not fusion.strip():
                    raise ValueError(
                        f"selection policy {path} mos_fusion must be a non-empty string"
                    )
                canonical["mos_fusion"] = fusion.strip().lower()

            if "early_exit" in policy:
                early_exit = policy["early_exit"]
                validate_canonical_policy_json(
                    early_exit, f"selection policy {path}.early_exit"
                )
                canonical["early_exit"] = canonical_policy_value(early_exit)
            return canonical

        canonical_global = canonical_policy_value(default_policy)
        canonical_global["mos_fusion"] = _SUPPORTED_MOS_FUSION
        if "global" in policies:
            canonical_global.update(canonical_policy(policies["global"], "global"))
        canonical_layers = {"global": canonical_global}
        if "quality_profiles" in policies:
            canonical_layers["quality_profiles"] = canonical_quality_profiles(
                policies["quality_profiles"],
                get_mos_spec=self._get_mos_spec,
            )
            for route in self._routes.values():
                if route.quality_profile not in canonical_layers["quality_profiles"]:
                    raise ValueError(
                        f"dataset {route.dataset_id!r} declares unknown "
                        f"quality_profile {route.quality_profile!r}"
                    )
        if "sources" in policies:
            source_datasets = policies["sources"]
            if not isinstance(source_datasets, Mapping):
                raise ValueError("selection policy layer 'sources' must be a mapping")
            unknown = set(source_datasets) - set(self._routes)
            if unknown:
                raise ValueError(
                    f"unknown dataset in source selection policies: {sorted(unknown)!r}"
                )
            canonical_sources = {}
            for dataset_id in sorted(source_datasets):
                routes = source_datasets[dataset_id]
                if not isinstance(routes, Mapping):
                    raise ValueError(
                        f"selection policy sources.{dataset_id} must be a mapping"
                    )
                canonical_routes = {}
                for raw_path, policy in routes.items():
                    canonical_path = canonical_source_route_path(
                        raw_path, f"selection policy sources.{dataset_id} route"
                    )
                    if canonical_path in canonical_routes:
                        raise ValueError(
                            "duplicate source route after canonicalization: "
                            f"{dataset_id!r}/{canonical_path!r}"
                        )
                    canonical_routes[canonical_path] = canonical_policy(
                        policy, f"sources.{dataset_id}.{raw_path}"
                    )
                canonical_sources[dataset_id] = {
                    path: canonical_routes[path] for path in sorted(canonical_routes)
                }
            canonical_layers["sources"] = canonical_sources
        for layer in self._ALLOWED_LAYERS:
            if layer in ("global", "quality_profiles", "sources") or layer not in policies:
                continue
            entries = policies[layer]
            if not isinstance(entries, Mapping):
                raise ValueError(f"selection policy layer {layer!r} must be a mapping")
            for target in entries:
                if type(target) is not str or not target.strip():
                    raise ValueError(
                        f"selection policy {layer} keys must be non-empty strings"
                    )
            if layer == "datasets":
                unknown = set(entries) - set(self._routes)
                if unknown:
                    raise ValueError(
                        f"unknown dataset in selection policies: {sorted(unknown)!r}"
                    )
            if layer == "categories":
                unknown = set(entries) - known_categories
                if unknown:
                    raise ValueError(
                        f"unknown category in selection policies: {sorted(unknown)!r}"
                    )
            if layer == "languages":
                normalized_entries = {}
                for target, policy in entries.items():
                    canonical_target = normalize_language(target)
                    if not canonical_target:
                        raise ValueError(
                            "language policy target must normalize to a non-empty string"
                        )
                    if canonical_target in normalized_entries:
                        raise ValueError(
                            "duplicate language policy target after normalization: "
                            f"{canonical_target!r}"
                        )
                    normalized_entries[canonical_target] = canonical_policy(
                        policy, f"{layer}.{target}"
                    )
                canonical_layers[layer] = {
                    target: normalized_entries[target]
                    for target in sorted(normalized_entries)
                }
            else:
                canonical_layers[layer] = {
                    target: canonical_policy(entries[target], f"{layer}.{target}")
                    for target in sorted(entries)
                }
        return freeze_snapshot_value(canonical_layers)

    @staticmethod
    def _merge_policy(
        base: Dict[str, Any], override: Mapping[str, Any],
    ) -> Dict[str, Any]:
        result = dict(base)
        for key, value in override.items():
            if key == "models" or key == "mos_models":
                result["mos_models"] = list(value)
            elif key == "thresholds":
                thresholds = {
                    str(model).strip().lower(): threshold
                    for model, threshold in (result.get("thresholds", {}) or {}).items()
                }
                for model, threshold in value.items():
                    normalized_model = str(model).strip().lower()
                    if (
                        isinstance(threshold, Mapping)
                        and isinstance(thresholds.get(normalized_model), Mapping)
                    ):
                        merged = dict(thresholds[normalized_model])
                        merged.update(threshold)
                        thresholds[normalized_model] = merged
                    else:
                        thresholds[normalized_model] = threshold
                result["thresholds"] = thresholds
            else:
                result[key] = value
        return result

    def resolve(
        self,
        dataset_id: str,
        language: Optional[str] = None,
        *,
        source_relative_path: Optional[str] = None,
    ) -> ResolvedSelectionPolicy:
        """按 global→category→profile→language→dataset→source 顺序解析。"""
        if language is not None:
            if type(language) is not str or not language.strip():
                raise ValueError("language must be a non-empty string")
            normalized_requested_language = normalize_language(language)
            if not normalized_requested_language:
                raise ValueError("language must normalize to a non-empty string")
        else:
            normalized_requested_language = None
        canonical_source_path = None
        if source_relative_path is not None:
            canonical_source_path = canonical_source_route_path(
                source_relative_path, "source_relative_path"
            )
        cache_key = (
            self._namespace,
            dataset_id,
            normalized_requested_language,
            canonical_source_path,
        )
        cached = self._resolve_cache.get(cache_key)
        if cached is not None:
            return cached
        route = self._routes.get(dataset_id)
        if route is None:
            raise ValueError(f"unknown dataset: {dataset_id}")
        resolved_language = normalize_language(language or route.language)
        policies = self._snapshot
        result = self._merge_policy({}, policies["global"])
        if route.category:
            result = self._merge_policy(
                result,
                policies.get("categories", {}).get(route.category, {}),
            )
        if route.quality_profile_explicit and route.quality_profile is not None:
            result = self._merge_policy(
                result,
                policies.get("quality_profiles", {}).get(route.quality_profile, {}),
            )
        result = self._merge_policy(
            result, policies.get("languages", {}).get(resolved_language, {})
        )
        result = self._merge_policy(
            result, policies.get("datasets", {}).get(dataset_id, {})
        )
        dataset_source_routes = policies.get("sources", {}).get(dataset_id, {})
        source_route_matched = canonical_source_path in dataset_source_routes
        route_identity = None
        if source_route_matched:
            result = self._merge_policy(
                result, dataset_source_routes[canonical_source_path]
            )
            route_identity = {
                "dataset_id": dataset_id,
                "source_relative_path": canonical_source_path,
            }
        models = tuple(
            str(model).strip().lower() for model in result["mos_models"]
        )
        thresholds = {
            str(key).strip().lower(): value
            for key, value in result.get("thresholds", {}).items()
        }
        if route.quality_profile_explicit:
            missing = [model for model in models if model not in thresholds]
            if missing:
                raise ValueError(
                    f"resolved selection policy for dataset {dataset_id!r} "
                    f"model(s) {missing} are missing final threshold(s)"
                )
        effective_policy = {
            "mos_models": models,
            "thresholds": thresholds,
            "min_mos_models": int(result["min_mos_models"]),
            "early_exit": result.get("early_exit"),
        }
        canonical = canonical_policy_value({
            "dataset_id": dataset_id,
            "language": resolved_language,
            "category": route.category,
            "quality_profile": route.quality_profile,
            "route_identity": route_identity,
            "effective_policy": effective_policy,
        })
        payload = json.dumps(
            canonical,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        policy_id = f"{dataset_id}:{resolved_language}"
        if route_identity is not None:
            policy_id = f"{policy_id}:{canonical_source_path}"
        policy = ResolvedSelectionPolicy(
            policy_id,
            category=route.category,
            quality_profile=route.quality_profile,
            mos_models=models,
            thresholds=freeze_policy_value(thresholds),
            min_mos_models=int(result["min_mos_models"]),
            mos_fusion=str(result["mos_fusion"]),
            early_exit=freeze_policy_value(result.get("early_exit")),
            policy_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )
        self._resolve_cache[cache_key] = policy
        return policy

    def _dataset_route_paths(self, dataset_id: str) -> Tuple[Optional[str], ...]:
        """返回一个数据集实际可达的 baseline 与 exact-source 策略路由。"""
        route = self._routes.get(dataset_id)
        if route is None:
            raise ValueError(f"unknown dataset: {dataset_id}")
        source_paths = tuple(sorted(
            self._snapshot.get("sources", {}).get(dataset_id, {})
        ))
        return (None, *source_paths)

    def dataset_scoring_binding(self, dataset_id: str) -> Mapping[str, Any]:
        """返回仅绑定目标数据集 Step3 评分语义的 canonical payload。"""
        route = self._routes.get(dataset_id)
        if route is None:
            raise ValueError(f"unknown dataset: {dataset_id}")
        projections = []
        for source_path in self._dataset_route_paths(dataset_id):
            projection = self.scoring_projection(
                dataset_id,
                route.language,
                source_relative_path=source_path,
            )
            projections.append(canonical_policy_value(projection.as_mapping()))
        return freeze_snapshot_value({
            "schema": "audio_filter_pipeline.dataset_scoring_binding.v1",
            "dataset_id": dataset_id,
            "projections": projections,
        })

    def dataset_decision_binding(self, dataset_id: str) -> Mapping[str, Any]:
        """返回仅绑定目标数据集 Step4/6 决策语义的 canonical payload。"""
        route = self._routes.get(dataset_id)
        if route is None:
            raise ValueError(f"unknown dataset: {dataset_id}")
        policies = []
        for source_path in self._dataset_route_paths(dataset_id):
            policy = self.resolve(
                dataset_id,
                route.language,
                source_relative_path=source_path,
            )
            policies.append({
                "source_relative_path": source_path,
                "policy_id": policy.policy_id,
                "category": policy.category,
                "quality_profile": policy.quality_profile,
                "mos_models": list(policy.mos_models),
                "thresholds": canonical_policy_value(policy.thresholds),
                "min_mos_models": policy.min_mos_models,
                "mos_fusion": policy.mos_fusion,
                "early_exit": canonical_policy_value(policy.early_exit),
                "policy_hash": policy.policy_hash,
            })
        return freeze_snapshot_value({
            "schema": "audio_filter_pipeline.dataset_decision_binding.v1",
            "dataset_id": dataset_id,
            "policies": policies,
        })

    def scoring_projection(
        self,
        dataset_id: str,
        language: Optional[str] = None,
        *,
        source_relative_path: Optional[str] = None,
    ) -> MosScoringProjection:
        """解析 threshold-free Step3 scoring projection。"""
        canonical_source_path = None
        if source_relative_path is not None:
            canonical_source_path = canonical_source_route_path(
                source_relative_path, "source_relative_path"
            )
        policy = self.resolve(
            dataset_id,
            language,
            source_relative_path=canonical_source_path,
        )
        route = self._routes.get(dataset_id)
        if route is None:
            raise ValueError(f"unknown dataset: {dataset_id}")
        route_identity = None
        if canonical_source_path in self._snapshot.get("sources", {}).get(
            dataset_id, {}
        ):
            route_identity = {
                "dataset_id": dataset_id,
                "source_relative_path": canonical_source_path,
            }
        projection = {
            "dataset_id": dataset_id,
            "language": normalize_language(language or route.language),
            "category": policy.category,
            "quality_profile": policy.quality_profile,
            "route_identity": route_identity,
            "mos_models": list(policy.mos_models),
        }
        payload = json.dumps(
            projection,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return MosScoringProjection(
            dataset_id=dataset_id,
            language=projection["language"],
            category=policy.category,
            quality_profile=policy.quality_profile,
            route_identity=(
                None if route_identity is None else MappingProxyType(route_identity)
            ),
            mos_models=policy.mos_models,
            scoring_policy_hash=hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest(),
        )
