#!/usr/bin/env python3
"""将 MambAttention/RWSA 官方 generator checkpoint 转为本仓 GAN 格式。"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.utils.safe_load import safe_torch_load
class CheckpointConversionError(RuntimeError):
    """官方 checkpoint 不满足严格转换契约。"""

def _convert_state(checkpoint: object, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping) or "generator" not in checkpoint:
        raise CheckpointConversionError("official checkpoint must contain 'generator'")
    source = checkpoint["generator"]
    if not isinstance(source, Mapping) or not source:
        raise CheckpointConversionError("checkpoint['generator'] must be non-empty")

    converted: dict[str, torch.Tensor] = {}
    for raw_key, tensor in source.items():
        if not isinstance(raw_key, str) or not isinstance(tensor, torch.Tensor):
            raise CheckpointConversionError("official generator must be tensor-only")
        key = f"model.{raw_key.removeprefix('module.')}"
        if key in converted:
            raise CheckpointConversionError(f"normalized key collision: {key!r}")
        converted[key] = tensor

    expected = model.state_dict()
    matched = set(converted) & set(expected)
    if not matched:
        raise CheckpointConversionError("0 matching tensors after adding 'model.' prefix")
    missing = sorted(set(expected) - set(converted))
    unexpected = sorted(set(converted) - set(expected))
    mismatched = sorted(
        key for key in matched
        if converted[key].shape != expected[key].shape
        or converted[key].dtype != expected[key].dtype
    )
    if missing or unexpected or mismatched:
        raise CheckpointConversionError(
            "official generator does not strictly match selected model; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_or_dtype_mismatches={mismatched[:5]}"
        )
    return {key: converted[key].detach().cpu().clone().contiguous() for key in expected}

def convert_checkpoint(
    source_path: str | Path, output_path: str | Path, *, model: torch.nn.Module
) -> dict[str, int]:
    """安全加载、严格校验并输出 ``model_generator`` checkpoint。"""
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise CheckpointConversionError(f"refusing to overwrite output: {output}")
    checkpoint = safe_torch_load(
        source_path, map_location="cpu",
        purpose="convert official MambAttention/RWSA generator checkpoint",
    )
    converted = _convert_state(checkpoint, model)
    model.load_state_dict(converted, strict=True)
    descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save({"model_generator": converted}, handle)
        reloaded = safe_torch_load(output, map_location="cpu")
        if not isinstance(reloaded, Mapping) or set(reloaded) != {"model_generator"}:
            raise CheckpointConversionError("converted checkpoint schema mismatch")
        model.load_state_dict(reloaded["model_generator"], strict=True)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return dict(
        tensor_count=len(converted), matched_tensor_count=len(converted),
        parameter_count=sum(value.numel() for value in converted.values()),
    )

def _build_model(name: str) -> torch.nn.Module:
    if name == "mambattention":
        from tasks.se.models.gan.generators.mambattention import MambAttentionGenerator
        return MambAttentionGenerator()
    from tasks.se.models.gan.generators.rwsa_mamba_unet import RWSAMambaUNetGenerator
    hidden, blocks = {
        "rwsa-xs": (16, 2), "rwsa-s": (16, 4), "rwsa-m": (24, 4)
    }[name]
    return RWSAMambaUNetGenerator(hid_feature=hidden, num_tfmamba=blocks)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="官方 .pth checkpoint")
    parser.add_argument("output", help="输出的本仓 GAN generator checkpoint")
    parser.add_argument(
        "--model", required=True,
        choices=("mambattention", "rwsa-xs", "rwsa-s", "rwsa-m"),
    )
    args = parser.parse_args()
    stats = convert_checkpoint(args.input, args.output, model=_build_model(args.model))
    for key, value in stats.items():
        print(f"{key}={value}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
