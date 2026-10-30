#!/usr/bin/env python3
"""将可信训练 checkpoint 提取为可安全 weights-only 加载的 model-only checkpoint。"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.utils.inference_artifacts import fingerprint_file
from core.utils.safe_load import safe_torch_load
from evaluate.checkpoint_loading import (
    CheckpointFamily,
    extract_state_dict,
    strip_ddp_prefix,
)

MODEL_ONLY_SCHEMA = "use_train.model_only.v1"
DEFAULT_MODEL_ONLY_NAME = "best_model.model_only.pt"
_TRUSTED_EXTRACTION_PURPOSE = (
    "extract model-only weights from trusted self-generated training checkpoint"
)
_PRIMITIVE_TYPES = (type(None), bool, int, float, str, bytes)


def is_model_only_checkpoint_path(path: str | Path) -> bool:
    """返回路径是否为 canonical model-only checkpoint 文件名。"""
    return Path(path).name.endswith(".model_only.pt")


def default_model_only_path(source: str | Path) -> Path:
    """返回训练 checkpoint 同目录下的稳定、来源专属 model-only 路径。"""
    source_path = Path(source).expanduser().resolve()
    if source_path.name == "best_model.tar":
        name = DEFAULT_MODEL_ONLY_NAME
    elif source_path.name == "best_model":
        # 无扩展名来源会占用历史保留名，重复后缀使其落在禁止作为来源的命名空间。
        name = f"{DEFAULT_MODEL_ONLY_NAME}.model_only.pt"
    else:
        name = f"{source_path.name}.model_only.pt"
    return source_path.with_name(name)


def checkpoint_family_from_config(config: Mapping[str, Any]) -> CheckpointFamily:
    """按 inference.py / onnx/export.py 语义从配置推导 checkpoint family。"""
    model = config.get("model", {})
    if isinstance(model, Mapping) and "generator" in model:
        return CheckpointFamily.GAN_GENERATOR
    return CheckpointFamily.SINGLE_MODEL


def _safe_data_to_cpu(
    value: Any, *, context: str, allow_tuple: bool = True
) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, _PRIMITIVE_TYPES):
        return value
    if isinstance(value, list):
        return [
            _safe_data_to_cpu(
                item,
                context=f"{context}[{index}]",
                allow_tuple=allow_tuple,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        if not allow_tuple:
            raise RuntimeError(f"{context} EMA 仅允许 primitive/Tensor/list/dict")
        return tuple(
            _safe_data_to_cpu(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        normalized: dict[Any, Any] = {}
        for key, item in value.items():
            if not isinstance(key, _PRIMITIVE_TYPES):
                raise RuntimeError(
                    f"{context} 包含不安全 mapping key: {type(key).__name__}"
                )
            normalized[key] = _safe_data_to_cpu(
                item,
                context=f"{context}[{key!r}]",
                allow_tuple=allow_tuple,
            )
        return normalized
    raise RuntimeError(f"{context} 包含不安全值: {type(value).__name__}")


def _canonical_state(checkpoint: Mapping[str, Any], family: CheckpointFamily) -> dict:
    state = strip_ddp_prefix(extract_state_dict(checkpoint, family))
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            normalized[key] = value.detach().cpu()
        elif key == "_extra_state" or key.endswith("._extra_state"):
            normalized[key] = _safe_data_to_cpu(
                value, context=f"state_dict[{key!r}]"
            )
        else:
            raise RuntimeError(f"state_dict[{key!r}] 必须是 Tensor")
    return normalized


def _canonical_ema(checkpoint: Mapping[str, Any]) -> dict | None:
    if "ema" not in checkpoint:
        return None
    ema = checkpoint["ema"]
    if not isinstance(ema, Mapping):
        raise RuntimeError("checkpoint EMA 状态必须是 mapping")
    shadow_params = ema.get("shadow_params")
    if not isinstance(shadow_params, list):
        raise RuntimeError("EMA shadow_params 必须是 list")
    if not all(isinstance(item, torch.Tensor) for item in shadow_params):
        raise RuntimeError("EMA shadow_params 的每个元素必须是 Tensor")
    normalized = _safe_data_to_cpu(ema, context="ema", allow_tuple=False)
    normalized["shadow_params"] = [item.detach().cpu() for item in shadow_params]
    return normalized


def _build_model_only_payload(
    checkpoint: Mapping[str, Any],
    *,
    family: CheckpointFamily,
    source_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    model_key = (
        "model_generator"
        if family is CheckpointFamily.GAN_GENERATOR
        else "model"
    )
    payload: dict[str, Any] = {
        "schema": MODEL_ONLY_SCHEMA,
        "family": family.value,
        "source_fingerprint": dict(source_fingerprint),
        model_key: _canonical_state(checkpoint, family),
    }
    ema = _canonical_ema(checkpoint)
    if ema is not None:
        payload["ema"] = ema
    return payload


def validate_model_only_checkpoint(
    checkpoint_or_path: Mapping[str, Any] | str | Path,
    *,
    source_path: str | Path | None = None,
    family: CheckpointFamily | str | None = None,
) -> dict[str, Any]:
    """安全加载并严格验证 canonical model-only checkpoint。"""
    if isinstance(checkpoint_or_path, Mapping):
        checkpoint = checkpoint_or_path
    else:
        checkpoint_path = Path(checkpoint_or_path).expanduser()
        if checkpoint_path.is_symlink():
            raise RuntimeError("model-only checkpoint 不允许符号链接")
        checkpoint = safe_torch_load(
            checkpoint_path,
            map_location="cpu",
            purpose="validate model-only checkpoint",
        )
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("model-only checkpoint 顶层必须是 mapping")
    if checkpoint.get("schema") != MODEL_ONLY_SCHEMA:
        raise RuntimeError("model-only checkpoint schema 不匹配")
    try:
        actual_family = CheckpointFamily(checkpoint.get("family"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("model-only checkpoint family 非法") from exc
    if family is not None and actual_family is not CheckpointFamily(family):
        raise RuntimeError("model-only checkpoint family 与预期不一致")

    model_key = (
        "model_generator"
        if actual_family is CheckpointFamily.GAN_GENERATOR
        else "model"
    )
    allowed_keys = {"schema", "family", "source_fingerprint", model_key, "ema"}
    unexpected = sorted(set(checkpoint) - allowed_keys)
    if unexpected:
        raise RuntimeError(f"model-only checkpoint 包含非预期字段: {unexpected}")
    raw_state = checkpoint.get(model_key)
    if isinstance(raw_state, Mapping) and any(
        isinstance(key, str) and key.startswith("module.") for key in raw_state
    ):
        raise RuntimeError(
            "model-only checkpoint state_dict 不是 canonical DDP-free 格式"
        )
    _canonical_state(checkpoint, actual_family)
    _canonical_ema(checkpoint)

    source_fingerprint = checkpoint.get("source_fingerprint")
    if not isinstance(source_fingerprint, Mapping):
        raise RuntimeError("model-only checkpoint 缺少 source_fingerprint")
    required_fingerprint_keys = {"path", "size", "sha256"}
    if set(source_fingerprint) != required_fingerprint_keys:
        raise RuntimeError("model-only checkpoint source_fingerprint 结构非法")
    fingerprint_path = source_fingerprint.get("path")
    fingerprint_size = source_fingerprint.get("size")
    fingerprint_sha256 = source_fingerprint.get("sha256")
    if (
        not isinstance(fingerprint_path, str)
        or not fingerprint_path
        or isinstance(fingerprint_size, bool)
        or not isinstance(fingerprint_size, int)
        or fingerprint_size < 0
        or not isinstance(fingerprint_sha256, str)
        or len(fingerprint_sha256) != 64
        or any(char not in "0123456789abcdef" for char in fingerprint_sha256)
    ):
        raise RuntimeError("model-only checkpoint source_fingerprint 类型非法")
    if source_path is not None:
        actual_source = fingerprint_file(source_path)
        if dict(source_fingerprint) != actual_source:
            raise RuntimeError("model-only checkpoint 来源 fingerprint 已失配")
    return dict(checkpoint)


def _load_training_checkpoint(
    source: Path, *, trusted_training_checkpoint: bool
) -> Mapping[str, Any]:
    try:
        checkpoint = safe_torch_load(
            source,
            map_location="cpu",
            purpose="extract model-only weights",
        )
    except Exception:
        if not trusted_training_checkpoint:
            raise
        checkpoint = safe_torch_load(
            source,
            map_location="cpu",
            allow_unsafe=True,
            purpose=_TRUSTED_EXTRACTION_PURPOSE,
        )
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("训练 checkpoint 顶层必须是 mapping")
    return checkpoint


def _staged_save_and_verify(
    payload: Mapping[str, Any], output: Path, *, source: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        validate_model_only_checkpoint(temporary, source_path=source)
        os.replace(temporary, output)
        directory_fd = os.open(
            output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize_model_only_checkpoint(
    source: str | Path,
    *,
    family: CheckpointFamily | str,
    output: str | Path | None = None,
    trusted_training_checkpoint: bool = False,
    force: bool = False,
) -> Path:
    """提取或复用 model-only checkpoint，并返回规范绝对路径。"""
    configured_source = Path(source).expanduser()
    if configured_source.is_symlink():
        raise ValueError("训练 checkpoint 不允许符号链接")
    if is_model_only_checkpoint_path(configured_source):
        raise ValueError("model-only checkpoint 不能作为训练来源")
    source_path = configured_source.resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f"训练 checkpoint 不是普通文件: {source_path}")
    checkpoint_family = CheckpointFamily(family)
    automatic_output = output is None
    if automatic_output:
        output_path = default_model_only_path(source_path)
    else:
        configured_output = Path(output).expanduser()
        if configured_output.is_symlink():
            raise ValueError("model-only 输出不允许符号链接")
        output_path = configured_output.resolve()
    if output_path == source_path:
        raise ValueError("model-only 输出不能覆盖来源训练 checkpoint")
    if output_path.is_symlink():
        raise ValueError("model-only 输出不允许符号链接")

    source_fingerprint = fingerprint_file(source_path)
    if output_path.exists():
        if automatic_output:
            try:
                existing = validate_model_only_checkpoint(output_path)
            except (OSError, RuntimeError, TypeError, ValueError):
                pass
            else:
                existing_source_path = existing["source_fingerprint"]["path"]
                if existing_source_path != str(source_path):
                    raise FileExistsError(
                        "默认 model-only 输出已绑定其他来源训练 checkpoint: "
                        f"{output_path}"
                    )
                try:
                    validate_model_only_checkpoint(
                        output_path,
                        source_path=source_path,
                        family=checkpoint_family,
                    )
                    return output_path
                except (OSError, RuntimeError, TypeError, ValueError):
                    pass
        elif not force:
            raise FileExistsError(f"拒绝覆盖已存在输出: {output_path}")

    checkpoint = _load_training_checkpoint(
        source_path,
        trusted_training_checkpoint=trusted_training_checkpoint,
    )
    if fingerprint_file(source_path) != source_fingerprint:
        raise RuntimeError("训练 checkpoint 在提取期间发生变化")
    payload = _build_model_only_payload(
        checkpoint,
        family=checkpoint_family,
        source_fingerprint=source_fingerprint,
    )
    _staged_save_and_verify(payload, output_path, source=source_path)
    validate_model_only_checkpoint(
        output_path,
        source_path=source_path,
        family=checkpoint_family,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从训练 checkpoint 安全提取 model-only checkpoint"
    )
    parser.add_argument("source", help="来源训练 checkpoint")
    parser.add_argument(
        "--family",
        required=True,
        choices=[family.value for family in CheckpointFamily],
        help="checkpoint family（禁止自动猜测）",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "输出路径；默认同目录：best_model.tar 使用 "
            f"{DEFAULT_MODEL_ONLY_NAME}，其它来源使用 <source-name>.model_only.pt；"
            "无扩展名 best_model 为避免历史名冲突使用双 .model_only.pt 后缀"
        ),
    )
    parser.add_argument(
        "--trusted-training-checkpoint",
        action="store_true",
        help="显式信任自生成训练 checkpoint；仅安全加载失败时允许 unsafe fallback",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖用户显式指定的已有输出；不会覆盖来源 checkpoint",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = materialize_model_only_checkpoint(
        args.source,
        family=args.family,
        output=args.output,
        trusted_training_checkpoint=args.trusted_training_checkpoint,
        force=args.force,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
