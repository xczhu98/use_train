"""SE/AEC checkpoint family 的严格加载工具。

该模块只处理本仓库训练产出的单模型与 GAN generator checkpoint。TSE 及第三方
官方 checkpoint 由各自 loader 维护兼容规则，禁止在此自动猜测格式。
"""

from __future__ import annotations

import copy
from enum import Enum
import hashlib
import logging
import os
import stat
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping

import torch

from core.utils import safe_load as safe_load_module

logger = logging.getLogger(__name__)


class CheckpointFamily(str, Enum):
    """受支持的本仓库 SE/AEC checkpoint family。"""

    SINGLE_MODEL = "single_model"
    GAN_GENERATOR = "gan_generator"


_SINGLE_MODEL_KEYS = ("model", "model_state_dict", "state_dict")


def _is_extra_state_key(key: str) -> bool:
    return key == "_extra_state" or key.endswith("._extra_state")


def _is_raw_state_dict(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and any(isinstance(item, torch.Tensor) for item in value.values())
        and all(
            isinstance(item, torch.Tensor) or _is_extra_state_key(key)
            for key, item in value.items()
        )
    )


def _require_state_dict(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(f"{context} 必须是非空 state_dict mapping")
    invalid_keys = [key for key in value if not isinstance(key, str)]
    non_tensors = [
        key
        for key, item in value.items()
        if isinstance(key, str)
        and not isinstance(item, torch.Tensor)
        and not _is_extra_state_key(key)
    ]
    if invalid_keys or non_tensors:
        raise RuntimeError(
            f"{context} 不是合法 state_dict: "
            f"invalid_keys={invalid_keys[:5]}, non_tensor_keys={non_tensors[:5]}"
        )
    return dict(value)


def extract_state_dict(
    checkpoint: Any,
    family: CheckpointFamily | str,
) -> dict[str, Any]:
    """按显式 family 提取 state_dict，不跨 family 猜测。"""
    family = CheckpointFamily(family)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("checkpoint 顶层必须是 mapping")

    if family is CheckpointFamily.GAN_GENERATOR:
        if "model_generator" not in checkpoint:
            raise RuntimeError("gan_generator checkpoint 缺少 'model_generator'")
        state = checkpoint["model_generator"]
        context = "checkpoint['model_generator']"
    else:
        present = [key for key in _SINGLE_MODEL_KEYS if key in checkpoint]
        if len(present) > 1:
            raise RuntimeError(
                f"single_model checkpoint 包含多个候选 state 键: {present}"
            )
        if present:
            key = present[0]
            state = checkpoint[key]
            context = f"checkpoint[{key!r}]"
        elif _is_raw_state_dict(checkpoint):
            state = checkpoint
            context = "raw checkpoint"
        else:
            raise RuntimeError(
                "single_model checkpoint 必须包含且仅包含一个权重键 "
                f"{list(_SINGLE_MODEL_KEYS)}，或本身为裸 tensor state_dict"
            )

    return _require_state_dict(state, context=context)


def strip_ddp_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """逐键移除 DDP ``module.`` 前缀，并拒绝归一化后的键冲突。"""
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = key[len("module.") :] if key.startswith("module.") else key
        if not normalized_key:
            raise RuntimeError("DDP state_dict 包含空参数名")
        if normalized_key in normalized:
            raise RuntimeError(
                f"移除 DDP prefix 后参数键冲突: {normalized_key!r}"
            )
        normalized[normalized_key] = value
    return normalized


def _state_alias_groups(
    model: torch.nn.Module,
    model_state: Mapping[str, Any],
) -> list[tuple[torch.Tensor, list[str], str]]:
    """按对象 identity 收集 parameter 与 persistent-buffer 的 state alias。"""
    groups: dict[int, tuple[torch.Tensor, list[str], str]] = {}
    for kind, named_values in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("persistent buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, value in named_values:
            if kind == "persistent buffer" and name not in model_state:
                continue
            identity = id(value)
            if identity not in groups:
                groups[identity] = (value, [], kind)
            groups[identity][1].append(name)
    return list(groups.values())


def _tensor_values_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """逐元素精确比较 tensor，要求 dtype/shape 一致并将同位 NaN 视为相同。"""
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    try:
        if left.device != right.device:
            right = right.to(device=left.device)
        if torch.equal(left, right):
            return True
        if not (left.is_floating_point() or left.is_complex()):
            return False
        return bool(
            torch.equal(torch.isnan(left), torch.isnan(right))
            and torch.equal(torch.nan_to_num(left), torch.nan_to_num(right))
        )
    except (RuntimeError, TypeError):
        return False


def _state_values_equal(left: Any, right: Any) -> bool:
    """精确比较 state_dict value，包括可嵌套的 extra state。"""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and _tensor_values_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_state_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_state_values_equal(a, b) for a, b in zip(left, right))
        )
    try:
        return bool(left == right)
    except (RuntimeError, TypeError, ValueError):
        return False


def _clone_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    """冻结逐 key 期望值，避免 checkpoint view 在提交期间同步变化。"""
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in state.items()
    }


def _validate_committed_state(
    model: torch.nn.Module,
    expected_state: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """提交成功前逐 key 复核完整 state，捕获 shared-storage/view 覆写。"""
    actual_state = model.state_dict()
    missing_keys = sorted(set(expected_state) - set(actual_state))
    unexpected_keys = sorted(set(actual_state) - set(expected_state))
    mismatched_keys = [
        key
        for key in expected_state.keys() & actual_state.keys()
        if not _state_values_equal(actual_state[key], expected_state[key])
    ]
    if missing_keys or unexpected_keys or mismatched_keys:
        raise RuntimeError(
            f"{context} 提交后 state_dict 与最终 prepared state 不一致: "
            f"missing_keys={missing_keys[:5]}, unexpected_keys={unexpected_keys[:5]}, "
            f"mismatched_keys={mismatched_keys[:5]}"
        )


def _validate_alias_values(
    state: Mapping[str, Any],
    alias_groups: list[tuple[torch.Tensor, list[str], str]],
    *,
    context: str,
) -> None:
    """拒绝会依赖 state_dict 遍历顺序的 tied state 冲突。"""
    for _value, names, kind in alias_groups:
        present = [name for name in names if name in state]
        if len(present) < 2:
            continue
        reference = state[present[0]]
        conflicts = [
            name
            for name in present[1:]
            if not isinstance(reference, torch.Tensor)
            or not isinstance(state[name], torch.Tensor)
            or not _tensor_values_equal(reference, state[name])
        ]
        if conflicts:
            raise RuntimeError(
                f"{context} 同一 {kind} 的 alias 值不一致: "
                f"aliases={present[:8]}, conflicts={conflicts[:5]}"
            )


def _capture_model_snapshot(model: torch.nn.Module) -> dict[str, Any]:
    """捕获 checkpoint 提交可恢复的有限模型内状态。"""
    modules = list(model.modules())
    registries = [
        (
            module,
            dict(module._parameters),
            dict(module._buffers),
            dict(module._modules),
            set(module._non_persistent_buffers_set),
        )
        for module in modules
    ]

    tensors: dict[int, tuple[torch.Tensor, torch.Tensor, bool | None]] = {}
    for _module, parameters, buffers, _children, non_persistent in registries:
        for value in parameters.values():
            if value is not None and id(value) not in tensors:
                tensors[id(value)] = (
                    value,
                    value.detach().clone(),
                    bool(value.requires_grad),
                )
        for name, value in buffers.items():
            if (
                value is not None
                and name not in non_persistent
                and id(value) not in tensors
            ):
                tensors[id(value)] = (
                    value,
                    value.detach().clone(),
                    bool(value.requires_grad),
                )

    extra_states: list[tuple[torch.nn.Module, Any]] = []
    for module in modules:
        if (
            getattr(module.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
            is torch.nn.Module.get_extra_state
            or getattr(module.__class__, "set_extra_state", torch.nn.Module.set_extra_state)
            is torch.nn.Module.set_extra_state
        ):
            continue
        try:
            extra_states.append((module, copy.deepcopy(module.get_extra_state())))
        except Exception:
            # 无法复制的自定义 extra state 不在有限回滚范围内。
            continue
    return {
        "registries": registries,
        "tensors": list(tensors.values()),
        "extra_states": extra_states,
    }


def _restore_model_snapshot(snapshot: Mapping[str, Any]) -> None:
    """尽最大努力恢复有限快照；所有恢复尝试完成后再报告失败。"""
    errors: list[str] = []

    def restore_registries_and_tensors() -> None:
        for module, parameters, buffers, children, non_persistent in snapshot["registries"]:
            try:
                module._parameters.clear()
                module._parameters.update(parameters)
                module._buffers.clear()
                module._buffers.update(buffers)
                module._modules.clear()
                module._modules.update(children)
                module._non_persistent_buffers_set.clear()
                module._non_persistent_buffers_set.update(non_persistent)
            except Exception as exc:
                errors.append(f"registry {type(module).__name__}: {exc}")
        with torch.no_grad():
            for value, saved_value, requires_grad in snapshot["tensors"]:
                try:
                    if (
                        value.shape == saved_value.shape
                        and value.dtype == saved_value.dtype
                        and value.device == saved_value.device
                    ):
                        value.copy_(saved_value)
                    else:
                        value.data = saved_value.detach().clone()
                    if requires_grad is not None:
                        value.requires_grad_(requires_grad)
                except Exception as exc:
                    errors.append(f"tensor {type(value).__name__}: {exc}")

    restore_registries_and_tensors()
    for module, extra_state in snapshot["extra_states"]:
        try:
            module.set_extra_state(copy.deepcopy(extra_state))
        except Exception as exc:
            errors.append(f"extra state {type(module).__name__}: {exc}")
    restore_registries_and_tensors()
    if errors:
        raise RuntimeError("; ".join(errors[:8]))


def _commit_model_transaction(
    model: torch.nn.Module,
    operation: Callable[[], None],
    expected_state: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """执行并复核有限模型内事务，不承诺回滚自定义 hook 的外部副作用。"""
    snapshot = _capture_model_snapshot(model)
    frozen_expected_state = _clone_state_dict(expected_state)
    try:
        operation()
        _validate_committed_state(model, frozen_expected_state, context=context)
    except BaseException as commit_exc:
        try:
            _restore_model_snapshot(snapshot)
        except BaseException as rollback_exc:
            raise RuntimeError(
                f"{context} 提交失败，且有限模型状态回滚失败: {rollback_exc}"
            ) from commit_exc
        raise


def _prepare_state_dict_checked(
    model: torch.nn.Module,
    state_dict: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """纯校验并规范化 state_dict，不写入模型。"""
    state = strip_ddp_prefix(
        _require_state_dict(state_dict, context=f"{context} state_dict")
    )
    model_state = model.state_dict()
    model_keys = set(model_state)
    state_keys = set(state)
    parameter_keys = {name for name, _ in model.named_parameters()}
    matched_parameters = parameter_keys.intersection(state_keys)
    if not matched_parameters:
        raise RuntimeError(f"{context}: zero model parameters matched")

    missing_keys = sorted(model_keys - state_keys)
    unexpected_keys = sorted(state_keys - model_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"{context} 与模型 state_dict 不匹配: "
            f"missing_keys={len(missing_keys)}, "
            f"unexpected_keys={len(unexpected_keys)} "
            f"(missing 示例: {missing_keys[:5]}, "
            f"unexpected 示例: {unexpected_keys[:5]})"
        )

    shape_mismatches = [
        f"{key}: checkpoint={tuple(value.shape)}, model={tuple(model_state[key].shape)}"
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
        and isinstance(model_state[key], torch.Tensor)
        and value.shape != model_state[key].shape
    ]
    if shape_mismatches:
        raise RuntimeError(
            f"{context} tensor shape 不匹配: {shape_mismatches[:5]}"
        )

    non_tensor_mismatches = [
        key
        for key, value in state.items()
        if not isinstance(value, torch.Tensor)
        and not _is_extra_state_key(key)
    ]
    if non_tensor_mismatches:
        raise RuntimeError(
            f"{context} 包含非预期非 tensor state: {non_tensor_mismatches[:5]}"
        )
    _validate_alias_values(
        state,
        _state_alias_groups(model, model_state),
        context=context,
    )
    return state


def load_state_dict_checked(
    model: torch.nn.Module,
    state_dict: Mapping[str, Any],
    *,
    context: str = "checkpoint",
) -> None:
    """完整校验并事务加载；仅保证有限模型内状态可回滚。

    回滚覆盖 parameter、persistent buffer 的对象/alias/值/``requires_grad``，以及
    可 ``deepcopy`` 的 extra state；不承诺撤销自定义 load hook 的外部副作用。
    """
    state = _prepare_state_dict_checked(
        model,
        state_dict,
        context=context,
    )
    _commit_model_transaction(
        model,
        lambda: model.load_state_dict(state, strict=True),
        state,
        context=context,
    )


def _validate_ema_shadow_params(
    shadow_params: Any,
    parameters: list[torch.nn.Parameter],
) -> None:
    """EMA 复制前严格校验参数数量、Tensor 类型、dtype 与 shape。"""
    if not isinstance(shadow_params, list):
        raise RuntimeError("EMA shadow_params 必须是 list")
    if len(shadow_params) != len(parameters):
        raise RuntimeError(
            "EMA shadow_params 数量与模型参数不一致: "
            f"ema={len(shadow_params)}, model={len(parameters)}"
        )
    for index, (shadow, parameter) in enumerate(zip(shadow_params, parameters)):
        if not isinstance(shadow, torch.Tensor):
            raise RuntimeError(
                f"EMA shadow_params[{index}] 必须是 Tensor，实际为 {type(shadow).__name__}"
            )
        if shadow.dtype != parameter.dtype:
            raise RuntimeError(
                f"EMA shadow_params[{index}] dtype 与模型参数不一致: "
                f"ema={shadow.dtype}, model={parameter.dtype}"
            )
        if shadow.shape != parameter.shape:
            raise RuntimeError(
                f"EMA shadow_params[{index}] shape 与模型参数不一致: "
                f"ema={tuple(shadow.shape)}, model={tuple(parameter.shape)}"
            )


def _prepare_ema_if_present(
    checkpoint: Mapping[str, Any],
    model: torch.nn.Module,
) -> Any | None:
    """解析并完整校验 EMA 状态，不写入模型参数。"""
    if "ema" not in checkpoint:
        return None
    try:
        from torch_ema import ExponentialMovingAverage
    except ImportError as exc:
        raise RuntimeError("checkpoint 含 EMA，但当前环境缺少 torch_ema") from exc

    parameters = list(model.parameters())
    ema_state = checkpoint["ema"]
    if not isinstance(ema_state, Mapping):
        raise RuntimeError("checkpoint EMA 状态必须是 mapping")
    ema = ExponentialMovingAverage(parameters, decay=0.999)
    try:
        ema.load_state_dict(ema_state)
        _validate_ema_shadow_params(ema.shadow_params, parameters)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("checkpoint EMA 状态无法应用到当前模型") from exc
    return ema


def _commit_ema(ema: Any, model: torch.nn.Module) -> None:
    """提交已预校验的 EMA 参数。"""
    try:
        ema.copy_to(list(model.parameters()))
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("checkpoint EMA 状态无法应用到当前模型") from exc


def _state_with_ema_parameters(
    model: torch.nn.Module,
    state: Mapping[str, Any],
    shadow_params: Any,
) -> dict[str, Any]:
    """将 EMA 参数展开到完整 state_dict 的所有 identity alias。"""
    parameters = list(model.parameters())
    prepared = dict(state)
    shadow_by_identity = {
        id(parameter): shadow
        for parameter, shadow in zip(parameters, shadow_params)
    }
    for parameter, aliases, kind in _state_alias_groups(model, model.state_dict()):
        if kind != "parameter":
            continue
        shadow = shadow_by_identity[id(parameter)]
        for name in aliases:
            prepared[name] = shadow
    return prepared


def apply_ema_if_present(checkpoint: Mapping[str, Any], model: torch.nn.Module) -> bool:
    """事务应用 ``torch_ema``；有限回滚边界与 ``load_state_dict_checked`` 相同。"""
    ema = _prepare_ema_if_present(checkpoint, model)
    if ema is None:
        return False
    expected_state = _state_with_ema_parameters(
        model,
        model.state_dict(),
        ema.shadow_params,
    )
    _commit_model_transaction(
        model,
        lambda: _commit_ema(ema, model),
        expected_state,
        context="checkpoint EMA",
    )
    return True


def apply_checkpoint(
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    *,
    family: CheckpointFamily | str,
    context: str = "checkpoint",
) -> torch.nn.Module:
    """预校验并事务提交 base/EMA，禁止 alias 冲突与部分模型写入。

    提交异常时恢复 parameter、persistent buffer 的对象/alias/值/``requires_grad``
    及可 ``deepcopy`` 的 extra state；自定义加载代码造成的外部副作用不在保证内。
    """
    state = _prepare_state_dict_checked(
        model,
        extract_state_dict(checkpoint, family),
        context=context,
    )
    ema = _prepare_ema_if_present(checkpoint, model)
    if ema is not None:
        state = _state_with_ema_parameters(model, state, ema.shadow_params)
    _commit_model_transaction(
        model,
        lambda: model.load_state_dict(state, strict=True),
        state,
        context=context,
    )
    return model


def _checkpoint_identity(file_stat: os.stat_result) -> tuple[int, int, int, int]:
    """返回 FD 与路径复核共用的文件身份字段。"""
    return (
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
    )


def _open_checkpoint_snapshot(
    path: Path,
) -> tuple[Path, BinaryIO, dict[str, Any], tuple[int, int, int, int]]:
    """用单一不可跟随链接的 FD 固化 checkpoint 内容身份。"""
    expanded = path.expanduser()
    try:
        if expanded.is_symlink():
            raise RuntimeError(f"checkpoint 不允许符号链接: {expanded}")
        resolved = expanded.resolve(strict=True)
        descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise RuntimeError(f"无法安全打开 checkpoint: {expanded}") from exc
    stream = os.fdopen(descriptor, "rb", closefd=True)
    try:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"checkpoint 必须是普通文件: {resolved}")
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        fingerprint = {
            "path": str(resolved),
            "size": int(opened.st_size),
            "sha256": digest.hexdigest(),
        }
        stream.seek(0)
        return resolved, stream, fingerprint, _checkpoint_identity(opened)
    except BaseException:
        stream.close()
        raise


def _verify_checkpoint_path_identity(
    path: Path,
    stream: BinaryIO,
    opened_identity: tuple[int, int, int, int],
    *,
    stage: str,
) -> None:
    """加载后复核 FD 与当前路径仍指向打开时的同一文件。"""
    try:
        fd_identity = _checkpoint_identity(os.fstat(stream.fileno()))
        path_identity = _checkpoint_identity(path.stat())
    except OSError as exc:
        raise RuntimeError(f"checkpoint {stage}无法复核文件身份: {path}") from exc
    if fd_identity != opened_identity or path_identity != opened_identity:
        raise RuntimeError(f"checkpoint {stage}内容发生变化或文件身份被替换: {path}")


def _verify_checkpoint_content_identity(
    stream: BinaryIO,
    fingerprint: Mapping[str, Any],
    *,
    stage: str,
) -> None:
    """在同一 FD 上重算内容指纹，防止元数据可伪造的原地篡改。"""
    try:
        stream.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        current = {
            "path": fingerprint["path"],
            "size": int(os.fstat(stream.fileno()).st_size),
            "sha256": digest.hexdigest(),
        }
        stream.seek(0)
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"checkpoint {stage}无法复核内容身份") from exc
    if current != dict(fingerprint):
        raise RuntimeError(f"checkpoint {stage}内容发生变化")


def load_checkpoint_payload(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    purpose: str = "SE/AEC model checkpoint",
    expected_fingerprint: Mapping[str, Any] | None = None,
    checkpoint_loader: Any | None = None,
) -> tuple[Path, Mapping[str, Any]]:
    """从单一 FD 读取 checkpoint payload，供需先解析内嵌配置的调用方使用。"""
    resolved, stream, actual_fingerprint, opened_identity = _open_checkpoint_snapshot(
        Path(checkpoint_path)
    )
    try:
        if expected_fingerprint is not None and actual_fingerprint != dict(expected_fingerprint):
            raise RuntimeError(f"checkpoint 加载前内容发生变化: {resolved}")
        loader = checkpoint_loader or safe_load_module.safe_torch_load
        checkpoint = loader(stream, map_location=map_location, purpose=purpose)
        _verify_checkpoint_content_identity(stream, actual_fingerprint, stage="加载期间")
        _verify_checkpoint_path_identity(
            resolved, stream, opened_identity, stage="加载期间"
        )
    finally:
        stream.close()
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"checkpoint 顶层必须是 mapping: {resolved}")
    return resolved, checkpoint


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    family: CheckpointFamily | str,
    map_location: str | torch.device = "cpu",
    purpose: str = "SE/AEC model checkpoint",
    expected_fingerprint: Mapping[str, Any] | None = None,
    checkpoint_loader: Any | None = None,
) -> tuple[torch.nn.Module, Mapping[str, Any]]:
    """从单一 FD 安全读取并应用 checkpoint，拒绝加载窗口内的文件替换。"""
    resolved, checkpoint = load_checkpoint_payload(
        checkpoint_path,
        map_location=map_location,
        purpose=purpose,
        expected_fingerprint=expected_fingerprint,
        checkpoint_loader=checkpoint_loader,
    )
    apply_checkpoint(
        model,
        checkpoint,
        family=family,
        context=f"checkpoint {resolved}",
    )
    return model, checkpoint
