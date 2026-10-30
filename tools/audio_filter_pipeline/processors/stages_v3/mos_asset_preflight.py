"""Generation-3 Step3 MOS 执行计划的本地资产 preflight。"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from types import MappingProxyType
from typing import Any

from ..mos_contract import (
    MOS_ASSET_PROVENANCE_FIELDS,
    MOS_MODEL_ORDER,
    MOS_RUNTIME_OPTION_FIELDS,
)
from .mos_execution import MosExecutionPlan


__all__ = ["VerifiedMosExecutionPlan", "verify_mos_assets"]

# 变量名保留兼容，但其值定义的是 registry 的 canonical provenance 顺序，
# 与配置中的策略执行顺序解耦，不表示当前仍存在 legacy 执行路径。
_LEGACY_MODEL_ORDER = MOS_MODEL_ORDER
_SUPPORTED_MODELS = frozenset(MOS_MODEL_ORDER)
_MODEL_OPTION_FIELDS = MOS_RUNTIME_OPTION_FIELDS
_MODEL_PROVENANCE_FIELDS = MOS_ASSET_PROVENANCE_FIELDS
_CONSTRUCTION_TOKEN = object()


class _LocalFilesystem:
    """默认使用真实本地文件系统的最小路径 seam。"""

    @staticmethod
    def canonical_path(path: str) -> str:
        return os.path.realpath(os.path.abspath(path))

    @staticmethod
    def is_directory(path: str) -> bool:
        return os.path.isdir(path)

    @staticmethod
    def is_file(path: str) -> bool:
        return os.path.isfile(path)


_LOCAL_FILESYSTEM = _LocalFilesystem()


class _FrozenList(tuple):
    """保留 JSON array 相等语义的不可变序列。"""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        return False

    __hash__ = tuple.__hash__


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
        return _FrozenList(
            _freeze_json(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{label} must contain only JSON values")


def _plain(value: Any) -> Any:
    """返回与内部只读容器完全隔离的 JSON-safe 副本。"""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _plain(item) for key, item in value.items()}


def _required_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if not value.strip() or "\0" in value:
        raise ValueError(f"{label} must be a nonblank path or identifier")
    return value


def _canonical_path(filesystem: Any, path: str, label: str) -> str:
    _required_string(path, label)
    canonical = filesystem.canonical_path(path)
    if type(canonical) is not str:
        raise TypeError(f"{label} canonical path must be a string")
    if (
        not canonical
        or "\0" in canonical
        or not os.path.isabs(canonical)
        or canonical != os.path.realpath(os.path.abspath(canonical))
    ):
        raise ValueError(f"{label} canonical path must be absolute and normalized")
    return canonical


def _path_predicate(filesystem: Any, name: str, path: str, label: str) -> bool:
    result = getattr(filesystem, name)(path)
    if type(result) is not bool:
        raise TypeError(f"{label} {name}() must return bool")
    return result


def _sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 hex digest")
    return value


def _git_commit(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 40-character git commit")
    return value


def _resolve_tree_digest(dependency: Any) -> Any:
    if dependency is None:
        from ..optional_runtime import local_model_tree_sha256

        dependency = local_model_tree_sha256
    if not callable(dependency):
        raise TypeError("tree_digest must be callable for selected tree assets")
    return dependency


def _resolve_checkpoint_snapshot(dependency: Any) -> Any:
    if dependency is None:
        from ..mos import snapshot_local_checkpoint_identity

        dependency = snapshot_local_checkpoint_identity
    if not callable(dependency):
        raise TypeError("checkpoint_snapshot must be callable for selected checkpoints")
    return dependency


def _resolve_urgent_verifier(dependency: Any) -> Any:
    if dependency is None:
        from ..mos import verify_urgent_pk_trusted_assets

        dependency = verify_urgent_pk_trusted_assets
    if not callable(dependency):
        raise TypeError("urgent_verifier must be callable when urgent_pk is selected")
    return dependency


def _tree_identity(tree_digest: Any, path: str, label: str) -> str:
    return _sha256(tree_digest(path), label)


def _checkpoint_identity(checkpoint_snapshot: Any, path: str, model: str) -> dict[str, Any]:
    identity = checkpoint_snapshot(path)
    if not isinstance(identity, Mapping):
        raise TypeError(f"Step3 {model} checkpoint identity must be a mapping")
    expected_fields = {"checkpoint_sha256", "checkpoint_size_bytes"}
    if set(identity) != expected_fields or any(type(key) is not str for key in identity):
        raise ValueError(
            f"Step3 {model} checkpoint identity fields must be exactly "
            f"{sorted(expected_fields)!r}"
        )
    digest = _sha256(
        identity["checkpoint_sha256"],
        f"Step3 {model} checkpoint_sha256",
    )
    size = identity["checkpoint_size_bytes"]
    if type(size) is not int or size < 0:
        raise ValueError(
            f"Step3 {model} checkpoint_size_bytes must be a nonnegative integer"
        )
    return {"checkpoint_sha256": digest, "checkpoint_size_bytes": size}


def _urgent_identity(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("Step3 urgent_pk verifier provenance must be a mapping")
    expected_fields = {
        "checkpoint_sha256", "source_commit", "source_package_sha256",
    }
    if set(value) != expected_fields or any(type(key) is not str for key in value):
        raise ValueError(
            "Step3 urgent_pk verifier provenance fields must be exactly "
            f"{sorted(expected_fields)!r}"
        )
    return {
        "checkpoint_sha256": _sha256(
            value["checkpoint_sha256"], "Step3 urgent_pk checkpoint_sha256",
        ),
        "source_commit": _git_commit(
            value["source_commit"], "Step3 urgent_pk source_commit",
        ),
        "source_package_sha256": _sha256(
            value["source_package_sha256"],
            "Step3 urgent_pk source_package_sha256",
        ),
    }


def _exact_mapping_fields(value: Any, expected: frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(type(key) is not str for key in value) or set(value) != expected:
        raise ValueError(f"{label} fields must be exactly {sorted(expected)!r}")


def _validate_verified_model(
    model: str,
    options: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    _exact_mapping_fields(options, _MODEL_OPTION_FIELDS[model], f"{model} options")
    _exact_mapping_fields(
        provenance, _MODEL_PROVENANCE_FIELDS[model], f"{model} provenance",
    )
    if model in {"scoreq", "dnsmos"}:
        _required_string(options["model_dir"], f"{model} options model_dir")
        _required_string(provenance["model_dir"], f"{model} provenance model_dir")
        if options["model_dir"] != provenance["model_dir"]:
            raise ValueError(f"{model} model_dir identity mismatch")
        options_tree = _sha256(
            options["verified_model_tree_sha256"],
            f"{model} options verified_model_tree_sha256",
        )
        provenance_tree = _sha256(
            provenance["verified_model_tree_sha256"],
            f"{model} provenance verified_model_tree_sha256",
        )
        if options_tree != provenance_tree:
            raise ValueError(f"{model} verified tree identity mismatch")
        return
    if model == "urgent_pk":
        for name in ("checkpoint", "source_dir", "encoder", "backbone"):
            _required_string(options[name], f"urgent_pk options {name}")
        expected_sha256 = _sha256(
            options["expected_sha256"], "urgent_pk expected_sha256",
        )
        expected_commit = _git_commit(
            options["expected_commit"], "urgent_pk expected_commit",
        )
        actual = _urgent_identity(provenance)
        if actual["checkpoint_sha256"] != expected_sha256:
            raise ValueError("urgent_pk checkpoint identity does not match expected_sha256")
        if actual["source_commit"] != expected_commit:
            raise ValueError("urgent_pk source identity does not match expected_commit")
        _sha256(
            actual["source_package_sha256"],
            "urgent_pk verified source package identity",
        )
        return

    for name in ("model_path", "model_version"):
        _required_string(options[name], f"{model} options {name}")
        _required_string(provenance[name], f"{model} provenance {name}")
    if options["model_path"] != provenance["model_path"]:
        raise ValueError(f"{model} model_path identity mismatch")
    expected_tree = _sha256(options["model_sha256"], f"{model} model_sha256")
    checkpoint_sha256 = _sha256(
        options["checkpoint_sha256"], f"{model} checkpoint_sha256",
    )
    checkpoint_size = options["checkpoint_size_bytes"]
    if type(checkpoint_size) is not int or checkpoint_size < 0:
        raise ValueError(f"{model} checkpoint_size_bytes must be a nonnegative integer")
    if provenance["model_version"] != options["model_version"]:
        raise ValueError(f"{model} model_version identity mismatch")
    if provenance["expected_model_tree_sha256"] != expected_tree:
        raise ValueError(f"{model} expected tree identity mismatch")
    if provenance["verified_model_tree_sha256"] != expected_tree:
        raise ValueError(f"{model} verified tree identity mismatch")
    if provenance["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(f"{model} checkpoint digest identity mismatch")
    if provenance["checkpoint_size_bytes"] != checkpoint_size:
        raise ValueError(f"{model} checkpoint size identity mismatch")


@dataclass(frozen=True)
class VerifiedMosExecutionPlan:
    """仅由本模块 preflight 工厂构造的不可变验证结果。"""

    selected_models: tuple[str, ...]
    options_by_model: Mapping[str, Mapping[str, Any]]
    provenance: Mapping[str, Any]
    _construction_token: InitVar[object]

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("VerifiedMosExecutionPlan must be created by verify_mos_assets()")
        selected = self.selected_models
        if type(selected) is not tuple:
            raise TypeError("selected_models must be a tuple")
        if (
            not selected
            or any(type(model) is not str or model not in _SUPPORTED_MODELS for model in selected)
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("selected_models must be nonempty, unique, and supported")
        if not isinstance(self.options_by_model, Mapping):
            raise TypeError("options_by_model must be a mapping")
        if tuple(self.options_by_model) != selected:
            raise ValueError("options_by_model must match selected_models in policy order")
        _exact_mapping_fields(
            self.provenance,
            frozenset({"selected_models", "models"}),
            "provenance",
        )
        # 内部兼容别名仍指向 registry 的 canonical provenance 顺序；
        # execution 的 policy 顺序与 provenance 的 registry 顺序相互独立。
        provenance_order = tuple(model for model in _LEGACY_MODEL_ORDER if model in selected)
        provenance_selected = self.provenance["selected_models"]
        if type(provenance_selected) is not list or tuple(provenance_selected) != provenance_order:
            raise ValueError("provenance selected_models must use canonical registry order")
        models = self.provenance["models"]
        if not isinstance(models, Mapping) or tuple(models) != provenance_order:
            raise ValueError("provenance models must use canonical registry order")
        for model in selected:
            _validate_verified_model(
                model, self.options_by_model[model], models[model],
            )
        object.__setattr__(
            self,
            "options_by_model",
            _freeze_json(self.options_by_model, "options_by_model"),
        )
        object.__setattr__(
            self,
            "provenance",
            _freeze_json(self.provenance, "provenance"),
        )

    def as_mapping(self) -> dict[str, Any]:
        """返回可安全修改且使用 canonical provenance 外形的严格 JSON 副本。"""
        return {
            "selected_models": list(self.selected_models),
            "options_by_model": _plain(self.options_by_model),
            "provenance": _plain(self.provenance),
        }


def _verified_plan(
    selected_models: tuple[str, ...],
    options_by_model: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> VerifiedMosExecutionPlan:
    """私有工厂是唯一合法 construction token 持有者。"""
    return VerifiedMosExecutionPlan(
        selected_models=selected_models,
        options_by_model=options_by_model,
        provenance=provenance,
        _construction_token=_CONSTRUCTION_TOKEN,
    )


def verify_mos_assets(
    plan: MosExecutionPlan,
    *,
    filesystem: Any = _LOCAL_FILESYSTEM,
    tree_digest: Any = None,
    checkpoint_snapshot: Any = None,
    urgent_verifier: Any = None,
) -> VerifiedMosExecutionPlan:
    """验证 selected-only 本地资产，并生成 canonical runtime 消费的身份。

    canonical preflight 对所选资产的路径、摘要与绑定失败一律抛出，不提供
    fallback；未选模型既不验证，也不进入 provenance。
    """
    if type(plan) is not MosExecutionPlan:
        raise TypeError("plan must be a MosExecutionPlan")
    filesystem_methods = (
        getattr(filesystem, "canonical_path", None),
        getattr(filesystem, "is_directory", None),
        getattr(filesystem, "is_file", None),
    )
    if not all(callable(method) for method in filesystem_methods):
        raise TypeError("filesystem must provide canonical_path/is_directory/is_file")

    options_by_model: dict[str, dict[str, Any]] = {}
    provenance_by_model: dict[str, dict[str, Any]] = {}

    # 严格按 policy 顺序验证，失败时不构造或暴露部分 Verified 结果；
    # canonical preflight 不提供 fallback，也不触碰未选模型。
    for asset in plan.assets:
        model = asset.model
        paths = _plain_mapping(asset.paths)
        settings = _plain_mapping(asset.settings)

        if model in {"scoreq", "dnsmos"}:
            original_path = paths["model_dir"]
            canonical = _canonical_path(
                filesystem, original_path, f"Step3 {model} model_dir",
            )
            if not _path_predicate(
                filesystem, "is_directory", canonical, f"Step3 {model} model_dir",
            ):
                raise ValueError(
                    f"Step3 {model} model_dir must be an existing directory"
                )
            actual_tree = _tree_identity(
                _resolve_tree_digest(tree_digest), canonical,
                f"Step3 {model} verified_model_tree_sha256",
            )
            options_by_model[model] = {
                "model_dir": canonical,
                "verified_model_tree_sha256": actual_tree,
            }
            provenance_by_model[model] = {
                "model_dir": canonical,
                "verified_model_tree_sha256": actual_tree,
            }
            continue

        if model == "urgent_pk":
            options = {**paths, **settings}
            checkpoint = _canonical_path(
                filesystem, options["checkpoint"], "Step3 urgent_pk checkpoint",
            )
            source_dir = _canonical_path(
                filesystem, options["source_dir"], "Step3 urgent_pk source_dir",
            )
            if not _path_predicate(
                filesystem, "is_file", checkpoint, "Step3 urgent_pk checkpoint",
            ):
                raise ValueError("Step3 urgent_pk checkpoint must be an existing file")
            if not _path_predicate(
                filesystem, "is_directory", source_dir, "Step3 urgent_pk source_dir",
            ):
                raise ValueError("Step3 urgent_pk source_dir must be an existing directory")
            expected_sha256 = _sha256(
                options["expected_sha256"], "Step3 urgent_pk expected_sha256",
            )
            expected_commit = _git_commit(
                options["expected_commit"], "Step3 urgent_pk expected_commit",
            )
            options["checkpoint"] = checkpoint
            options["source_dir"] = source_dir
            # verifier 同时绑定 checkpoint、批准 commit 与实际 urgentpk package 的
            # Git blob 内容；真实 loader 仍从其 owned snapshot 导入。
            verified = _resolve_urgent_verifier(urgent_verifier)(
                checkpoint,
                source_dir,
                options["expected_sha256"],
                options["expected_commit"],
            )
            identity = _urgent_identity(verified)
            if identity["checkpoint_sha256"] != expected_sha256:
                raise ValueError("Step3 urgent_pk checkpoint digest mismatch")
            if identity["source_commit"] != expected_commit:
                raise ValueError("Step3 urgent_pk source commit mismatch")
            options_by_model[model] = options
            provenance_by_model[model] = identity
            continue

        original_path = paths["model_path"]
        canonical = _canonical_path(
            filesystem, original_path, f"Step3 {model} model_path",
        )
        if not _path_predicate(
            filesystem, "is_file", canonical, f"Step3 {model} model_path",
        ):
            raise ValueError(f"Step3 {model} model_path must be an existing file")
        expected_tree = _sha256(
            settings["model_sha256"], f"Step3 {model} expected_model_tree_sha256",
        )
        actual_tree = _tree_identity(
            _resolve_tree_digest(tree_digest),
            os.path.dirname(canonical),
            f"Step3 {model} verified_model_tree_sha256",
        )
        if expected_tree != actual_tree:
            raise ValueError(
                f"Step3 {model} model tree digest mismatch: "
                f"expected={expected_tree}, actual={actual_tree}"
            )
        checkpoint_identity = _checkpoint_identity(
            _resolve_checkpoint_snapshot(checkpoint_snapshot), canonical, model,
        )
        options_by_model[model] = {
            "model_path": canonical,
            "model_version": settings["model_version"],
            "model_sha256": expected_tree,
            **checkpoint_identity,
        }
        provenance_by_model[model] = {
            "model_path": canonical,
            "model_version": settings["model_version"],
            "expected_model_tree_sha256": expected_tree,
            "verified_model_tree_sha256": actual_tree,
            **checkpoint_identity,
        }

    # 这是 canonical provenance projection；局部变量名仅为兼容保留。
    legacy_selected = tuple(
        model for model in _LEGACY_MODEL_ORDER if model in plan.selected_models
    )
    provenance = {
        "selected_models": list(legacy_selected),
        "models": {
            model: provenance_by_model[model]
            for model in legacy_selected
        },
    }
    return _verified_plan(plan.selected_models, options_by_model, provenance)
