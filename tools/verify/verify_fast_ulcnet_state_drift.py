#!/usr/bin/env python3
"""用 synthetic 输入分离验证 Fast-ULCNet 整段前向与 temporal hidden drift。

默认参数仅用于 CPU smoke。长序列应优先增大 ``--temporal-frames``；整模型整段
forward 的内存随帧数增长，超过确认门限时需显式允许。本工具不评估语音质量，也不构成论文复现。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tasks.se.models.discriminative.fast_ulcnet import FastULCNet  # noqa: E402


_WHOLE_SEGMENT_CONFIRM_THRESHOLD = 1000


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float | bool]:
    detached = tensor.detach().float()
    return {
        "finite": bool(torch.isfinite(detached).all()),
        "rms": float(detached.square().mean().sqrt()),
        "peak_abs": float(detached.abs().max()),
    }


def verify_whole_segment_forward(
    model: FastULCNet,
    *,
    frames: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, object]:
    """对完整 spectrum-in/out 网络执行一次整段 synthetic forward。"""
    spectrum = torch.randn(
        1,
        257,
        frames,
        2,
        generator=generator,
        device=device,
    )
    with torch.inference_mode():
        enhanced = model(spectrum)
    return {
        "kind": "whole_segment_forward",
        "frames": frames,
        "input_shape": list(spectrum.shape),
        "output_shape": list(enhanced.shape),
        "shape_preserved": enhanced.shape == spectrum.shape,
        "output": _tensor_stats(enhanced),
    }


def verify_temporal_hidden_drift(
    model: FastULCNet,
    *,
    frames: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, object]:
    """直接观测官方 temporal sub-band Comfi-FastGRNN 的 hidden 轨迹。"""
    temporal_input = torch.randn(
        1,
        frames,
        model.sub_band_rnn1.input_size,
        generator=generator,
        device=device,
    )
    with torch.inference_mode():
        hidden_sequence, final_hidden = model.sub_band_rnn1(temporal_input)
    step_delta = hidden_sequence[:, 1:] - hidden_sequence[:, :-1]
    first = hidden_sequence[:, 0]
    last = hidden_sequence[:, -1]
    return {
        "kind": "temporal_hidden_drift",
        "frames": frames,
        "hidden_shape": list(hidden_sequence.shape),
        "trajectory": _tensor_stats(hidden_sequence),
        "first_hidden": _tensor_stats(first),
        "last_hidden": _tensor_stats(last),
        "final_state": _tensor_stats(final_hidden),
        "first_to_last_rms": float((last - first).float().square().mean().sqrt()),
        "mean_step_delta_rms": (
            float(step_delta.float().square().mean().sqrt()) if frames > 1 else 0.0
        ),
    }


def _validate_frame_counts(
    *,
    whole_segment_frames: int,
    temporal_frames: int,
    skip_whole_segment: bool,
    allow_large_whole_segment: bool,
) -> None:
    if whole_segment_frames <= 0 or temporal_frames <= 0:
        raise ValueError("frame counts must be positive")
    if (
        not skip_whole_segment
        and whole_segment_frames > _WHOLE_SEGMENT_CONFIRM_THRESHOLD
        and not allow_large_whole_segment
    ):
        raise ValueError(
            f"--whole-segment-frames exceeds the confirmation threshold "
            f"({_WHOLE_SEGMENT_CONFIRM_THRESHOLD}); full-model memory grows with frames. "
            "Use --skip-whole-segment for long temporal drift checks, or explicitly "
            "pass --allow-large-whole-segment after sizing device memory."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic Fast-ULCNet smoke：分别报告整模型整段 forward 与 temporal "
            "Comfi hidden drift；不读取外部数据，不评估语音质量。"
        )
    )
    parser.add_argument(
        "--whole-segment-frames",
        type=int,
        default=4,
        help=(
            "整模型 synthetic forward 帧数（默认 4，超过 "
            f"{_WHOLE_SEGMENT_CONFIRM_THRESHOLD} 需显式确认）"
        ),
    )
    parser.add_argument(
        "--skip-whole-segment",
        action="store_true",
        help="跳过高内存整模型 forward；长序列 hidden drift 检查推荐使用",
    )
    parser.add_argument(
        "--allow-large-whole-segment",
        action="store_true",
        help="显式允许整模型超过确认门限；可能耗尽 CPU/GPU 内存",
    )
    parser.add_argument(
        "--temporal-frames",
        type=int,
        default=128,
        help="temporal Comfi-FastGRNN hidden 轨迹帧数（长序列检查不走整模型）",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _validate_frame_counts(
        whole_segment_frames=args.whole_segment_frames,
        temporal_frames=args.temporal_frames,
        skip_whole_segment=args.skip_whole_segment,
        allow_large_whole_segment=args.allow_large_whole_segment,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    model = FastULCNet().to(device).eval()
    whole = None
    if not args.skip_whole_segment:
        whole = verify_whole_segment_forward(
            model,
            frames=args.whole_segment_frames,
            generator=generator,
            device=device,
        )
    temporal = verify_temporal_hidden_drift(
        model,
        frames=args.temporal_frames,
        generator=generator,
        device=device,
    )
    report = {
        "mode": "synthetic_smoke",
        "device": str(device),
        "seed": args.seed,
        "whole_segment_forward": whole,
        "temporal_hidden_drift": temporal,
        "limitations": (
            "synthetic finite/state statistics only; not speech-quality evaluation, "
            "checkpoint validation, or paper reproduction"
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    whole_passed = whole is None or bool(
        whole["shape_preserved"] and whole["output"]["finite"]
    )
    passed = bool(
        whole_passed
        and temporal["trajectory"]["finite"]
        and temporal["final_state"]["finite"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
