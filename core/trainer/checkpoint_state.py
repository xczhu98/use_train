"""Trainer checkpoint 的纯 schema、identity 与 model-state 校验。"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


RESOLVED_CONFIG_PROVENANCE_VERSION = 1
RESOLVED_CONFIG_IDENTITY_ALGORITHM = "sha256-json-v1"


def resolved_config_identity(config: Mapping[str, Any]) -> str:
    """对完整 resolved config 生成确定性 SHA-256 identity。"""
    if not isinstance(config, Mapping):
        raise TypeError("resolved config must be a mapping")
    try:
        canonical = json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("resolved config contains non-canonical values") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_resolved_config_source(source: object) -> str:
    """限制 provenance source 为 checkpoint 同目录的 resolved-config sidecar。"""
    if not isinstance(source, str) or not source:
        raise RuntimeError("resolved config provenance source must be a non-empty string")
    path = Path(source)
    if (
        path.is_absolute()
        or path.name != source
        or source in {".", ".."}
        or not source.startswith("resolved_config")
        or path.suffix not in {".yaml", ".yml", ".json"}
    ):
        raise RuntimeError(
            "resolved config provenance source must name a resolved_config sidecar "
            "in the checkpoint directory"
        )
    return source


def build_resolved_config_provenance(
    config: Mapping[str, Any],
    source: str,
) -> dict[str, object]:
    """构造 checkpoint 与同目录 resolved-config sidecar 的版本化绑定。"""
    return {
        "version": RESOLVED_CONFIG_PROVENANCE_VERSION,
        "algorithm": RESOLVED_CONFIG_IDENTITY_ALGORITHM,
        "sha256": resolved_config_identity(config),
        "source": _validate_resolved_config_source(source),
    }


def parse_resolved_config_provenance(value: object) -> dict[str, object]:
    """严格解析 checkpoint resolved-config provenance metadata。"""
    fields = {"version", "algorithm", "sha256", "source"}
    if not isinstance(value, Mapping) or set(value) != fields:
        got = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise RuntimeError(
            "resolved config provenance fields mismatch: "
            f"expected={sorted(fields)}, got={got}"
        )
    if (
        type(value["version"]) is not int
        or value["version"] != RESOLVED_CONFIG_PROVENANCE_VERSION
    ):
        raise RuntimeError(
            "unsupported resolved config provenance version: "
            f"{value['version']!r}"
        )
    if value["algorithm"] != RESOLVED_CONFIG_IDENTITY_ALGORITHM:
        raise RuntimeError(
            "unsupported resolved config provenance algorithm: "
            f"{value['algorithm']!r}"
        )
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise RuntimeError("resolved config provenance SHA-256 is invalid")
    return {
        "version": RESOLVED_CONFIG_PROVENANCE_VERSION,
        "algorithm": RESOLVED_CONFIG_IDENTITY_ALGORITHM,
        "sha256": digest,
        "source": _validate_resolved_config_source(value["source"]),
    }


def checkpoint_content_identity(path: Path) -> tuple[int, str]:
    """计算稳定的内容 identity，并拒绝读取期间发生变化的文件。"""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(source.fileno())
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or size != after.st_size
    ):
        raise RuntimeError("checkpoint changed while computing content identity")
    return size, digest.hexdigest()


def _require_nonnegative_exact_int(checkpoint: Mapping[str, Any], key: str) -> int:
    """读取非负精确整数，拒绝 bool、浮点数和数字字符串。"""
    value = checkpoint[key]
    if type(value) is not int:
        raise TypeError(
            f"Resume progress field '{key}' must be an exact int, got "
            f"{type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"Resume progress field '{key}' must be nonnegative")
    return value


def validate_resume_progress_schema(
    checkpoint: Mapping[str, Any],
    *,
    accumulation_steps: int,
) -> None:
    """校验进度 schema；未对齐的已提交尾窗留待 loader 边界确认。"""
    for key in ("epoch", "n_steps"):
        if key not in checkpoint:
            raise RuntimeError(f"Resume: checkpoint is missing required state '{key}'")
        _require_nonnegative_exact_int(checkpoint, key)

    if "epoch_incomplete" in checkpoint and type(checkpoint["epoch_incomplete"]) is not bool:
        raise TypeError(
            "Resume progress field 'epoch_incomplete' must be an exact bool, got "
            f"{type(checkpoint['epoch_incomplete']).__name__}"
        )
    for key in ("batch_cursor", "batch"):
        if key in checkpoint:
            _require_nonnegative_exact_int(checkpoint, key)

    if checkpoint.get("epoch_incomplete") is True and not (
        "batch_cursor" in checkpoint or "batch" in checkpoint
    ):
        raise RuntimeError(
            "Resume: incomplete checkpoint must include a progress field "
            "('batch_cursor' or legacy 'batch')"
        )

    accumulation = max(1, int(accumulation_steps))
    legacy_cursor = (
        ((checkpoint["batch"] + 1) // accumulation) * accumulation
        if "batch" in checkpoint
        else None
    )
    if "batch_cursor" in checkpoint:
        cursor = checkpoint["batch_cursor"]
        completed_tail = (
            checkpoint.get("epoch_incomplete") is True
            and "batch" not in checkpoint
        )
        if cursor % accumulation != 0 and not completed_tail:
            raise ValueError(
                f"Resume batch_cursor={cursor} is not aligned to "
                f"accumulation_steps={accumulation}"
            )
        if legacy_cursor is not None and cursor != legacy_cursor:
            raise ValueError(
                "Resume progress fields are inconsistent: "
                f"batch_cursor={cursor}, legacy batch={checkpoint['batch']} "
                f"resolves to {legacy_cursor}"
            )


def validate_resume_progress(
    checkpoint: Mapping[str, Any],
    *,
    accumulation_steps: int,
    train_loader_len: int | None = None,
) -> None:
    """最终恢复门禁；非对齐 cursor 必须证明是当前 loader 的已提交尾窗。"""
    validate_resume_progress_schema(checkpoint, accumulation_steps=accumulation_steps)
    accumulation = max(1, int(accumulation_steps))
    cursor = resume_batch_cursor(checkpoint, accumulation)
    completed_tail = (
        checkpoint.get("epoch_incomplete") is True
        and train_loader_len is not None
        and cursor == train_loader_len
        and "batch" not in checkpoint
    )
    if cursor % accumulation != 0 and not completed_tail:
        raise ValueError(
            f"Resume batch_cursor={cursor} is not aligned to "
            f"accumulation_steps={accumulation}"
        )
    if train_loader_len is not None:
        if type(train_loader_len) is not int or train_loader_len < 0:
            raise ValueError("train_loader_len must be a nonnegative exact int")
        if "batch" in checkpoint and checkpoint["batch"] >= train_loader_len:
            raise ValueError(
                f"Resume legacy batch {checkpoint['batch']} is outside train loader "
                f"length {train_loader_len}"
            )
        if cursor > train_loader_len:
            raise ValueError(
                f"Resume batch cursor {cursor} exceeds train loader length "
                f"{train_loader_len}"
            )


def resume_batch_cursor(checkpoint: Mapping[str, Any], accumulation_steps: int) -> int:
    """按当前兼容合同解析 middle-checkpoint 的已完成 batch cursor。"""
    if "batch_cursor" in checkpoint:
        return checkpoint["batch_cursor"]
    if "batch" in checkpoint:
        accumulation = max(1, int(accumulation_steps))
        return ((checkpoint["batch"] + 1) // accumulation) * accumulation
    return 0


def resume_start_epoch(checkpoint: Mapping[str, Any], accumulation_steps: int) -> int:
    """按 checkpoint schema 推导恢复起始 epoch。

    新 checkpoint 用 ``epoch_incomplete=True`` 明确表示未完成 epoch（包括
    ``batch_cursor=0``）；遗留 middle checkpoint 可由 ``batch`` 字段识别，即使
    首个 accumulation 尚未完成、解析出的 cursor 仍为 0，也恢复同一 epoch。
    仅含遗留 ``batch_cursor=0`` 的完整 checkpoint 保持进入下一 epoch 的兼容语义。
    """
    epoch = int(checkpoint["epoch"])
    if checkpoint.get("epoch_incomplete") is True or "batch" in checkpoint:
        return epoch
    cursor = resume_batch_cursor(checkpoint, accumulation_steps)
    return epoch if cursor > 0 else epoch + 1


def select_pretrained_model_state(
    checkpoint: Mapping[str, Any],
    *,
    model_key: str | None,
    multi_model: bool,
) -> Any:
    """保持 legacy key 优先级，选择 pretrained model state。"""
    if model_key and multi_model:
        selected = checkpoint.get(f"model_{model_key}")
        if selected is not None:
            return selected
    return (
        checkpoint.get("model_state_dict")
        or checkpoint.get("model")
        or checkpoint.get("state_dict")
        or checkpoint
    )


def validate_model_state(
    state_dict: Any,
    model_state: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """归一化完整 legacy ``module.`` 前缀并严格校验 model state。"""
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"{context}: expected a state_dict mapping, got "
            f"{type(state_dict).__name__}"
        )
    keys = list(state_dict.keys())
    if not keys or not all(isinstance(key, str) for key in keys):
        raise RuntimeError(
            f"{context}: state_dict is empty or contains non-string keys"
        )
    has_module = [key.startswith("module.") for key in keys]
    if any(has_module) and not all(has_module):
        raise RuntimeError(
            f"{context}: mixed 'module.' and unprefixed keys are not supported"
        )
    normalized = (
        {key[len("module.") :]: value for key, value in state_dict.items()}
        if all(has_module)
        else state_dict
    )

    model_keys = set(model_state)
    state_keys = set(normalized)
    if not model_keys.intersection(state_keys):
        raise RuntimeError(f"{context}: zero model parameters matched")

    missing_keys = sorted(model_keys - state_keys)
    unexpected_keys = sorted(state_keys - model_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"{context}: incomplete model state; missing={missing_keys[:5]}, "
            f"unexpected={unexpected_keys[:5]}"
        )

    shape_errors: list[str] = []
    non_tensor_keys: list[str] = []
    for key in sorted(model_keys):
        value = normalized[key]
        expected = model_state[key]
        if key.endswith("_extra_state"):
            continue
        if not isinstance(value, torch.Tensor):
            non_tensor_keys.append(key)
        elif not isinstance(expected, torch.Tensor):
            non_tensor_keys.append(key)
        elif tuple(value.shape) != tuple(expected.shape):
            shape_errors.append(
                f"{key}: expected {tuple(expected.shape)}, "
                f"got {tuple(value.shape)}"
            )
    if non_tensor_keys:
        raise RuntimeError(
            f"{context}: non-tensor model state values: {non_tensor_keys[:5]}"
        )
    if shape_errors:
        raise RuntimeError(f"{context}: shape mismatch: {shape_errors[:5]}")
    return normalized


def model_state_key_digest(state_dict: Mapping[str, Any]) -> str:
    """为跨 rank 结果比对生成与 tensor 内容无关的稳定 key digest。"""
    return hashlib.sha256("\0".join(sorted(state_dict)).encode()).hexdigest()
