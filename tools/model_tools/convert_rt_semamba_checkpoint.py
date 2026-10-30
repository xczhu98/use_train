#!/usr/bin/env python3
"""严格验证并转换官方 RT-SEMamba KD1/KD8 checkpoint。

输入只通过 ``safe_torch_load(weights_only=True)`` 读取；输出是本仓
``RT_SEMamba.network`` 的纯 state_dict，不携带 optimizer 或任意 pickle 对象。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.utils.safe_load import safe_torch_load


_VARIANT_DEPTH = {"KD1": 1, "KD8": 8}


def _extract_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    state = checkpoint.get("generator", checkpoint.get("model", checkpoint))
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint contains no non-empty generator/model state_dict")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise TypeError("state_dict must map string keys to tensors")
    prefixes = ("module.network.", "network.", "module.")
    keys = list(state)
    for prefix in prefixes:
        if keys and all(key.startswith(prefix) for key in keys):
            return {key[len(prefix) :]: value for key, value in state.items()}
    return dict(state)


def _validate_state(
    source: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor]
) -> None:
    missing = sorted(set(expected) - set(source))
    unexpected = sorted(set(source) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & set(source)
        if expected[key].shape != source[key].shape
        or expected[key].dtype != source[key].dtype
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "checkpoint is incompatible with RT_SEMamba: "
            f"missing={missing[:12]}, unexpected={unexpected[:12]}, "
            f"shape_or_dtype_mismatch={mismatched[:12]}"
        )


def convert_checkpoint(
    input_path: Path,
    variant: str,
    output_path: Path | None,
) -> dict[str, torch.Tensor]:
    """安全加载、严格校验，并按需保存 model-only state_dict。"""
    checkpoint = safe_torch_load(
        input_path,
        map_location="cpu",
        purpose="convert official RT-SEMamba model-only checkpoint",
    )
    source = _extract_state(checkpoint)
    try:
        from tasks.se.models.discriminative.rt_semamba import RT_SEMamba
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "RT-SEMamba conversion requires optional mamba_ssm/causal_conv1d backend"
        ) from exc

    model = RT_SEMamba(
        num_tfmamba=_VARIANT_DEPTH[variant],
        n_fft=400,
        hop_length=100,
        win_length=400,
        window="hann",
        center=False,
    )
    expected = model.network.state_dict()
    _validate_state(source, expected)
    model.network.load_state_dict(source, strict=True)
    converted = {key: value.detach().cpu() for key, value in model.network.state_dict().items()}
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(converted, output_path)
    return converted


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="官方或本仓 checkpoint")
    parser.add_argument("--variant", choices=sorted(_VARIANT_DEPTH), required=True)
    parser.add_argument("--output", type=Path, help="输出 model-only state_dict 路径")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="仅严格验证，不写输出；此时不得同时指定 --output",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.verify_only and args.output is not None:
        raise SystemExit("--verify-only cannot be combined with --output")
    if not args.verify_only and args.output is None:
        raise SystemExit("--output is required unless --verify-only is set")
    state = convert_checkpoint(
        args.input,
        args.variant,
        None if args.verify_only else args.output,
    )
    action = "verified" if args.verify_only else f"converted -> {args.output}"
    print(f"RT-SEMamba {args.variant}: {action}; tensors={len(state)}")


if __name__ == "__main__":
    main()
