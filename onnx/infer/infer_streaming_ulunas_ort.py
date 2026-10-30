#!/usr/bin/env python3
"""UL-UNAS 官方 streaming ONNX Runtime 逐帧推理。

固定部署前端为 16 kHz、512 点 Hann STFT、256 点 hop；模型图只处理
``mix [B,257,1,2]`` 和官方三个 flat cache。当前脚本沿用 ``center=True`` 的
离线 STFT 语义，因此网络零 look-ahead 不等于端到端零未来样本；带 checkpoint
时额外运行 PT streaming 对照，但 ORT 输出仍以 ONNX metadata 初始化 cache。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore[import]
import torch


def _load_wav(path: Path, sample_rate: int) -> torch.Tensor:
    data, source_rate = sf.read(str(path), always_2d=True, dtype="float32")
    wav = torch.from_numpy(data.T.copy())
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        raise ValueError(f"Input sample rate must be {sample_rate} Hz, got {source_rate} Hz")
    return wav


def _preflight_onnx_session(session):
    """在读取音频前严格校验 ONNX runtime metadata 与 I/O ABI。"""
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    return StreamingORTFrameRunner(session, "ulunas", batch_size=1)


def _require_stft_parameters_match(stft, **used: int) -> None:
    """artifact 是 STFT 真源：已声明字段必须与本次推理使用的参数一致。"""
    if stft.window != "hann":
        raise ValueError(
            f"Script hardcodes window='hann' but ONNX metadata declares {stft.window!r}"
        )
    for name, value in used.items():
        declared = getattr(stft, name)
        if declared is not None and int(value) != int(declared):
            raise ValueError(
                f"{name}={value!r} does not match ONNX metadata {declared!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="UL-UNAS streaming ONNX Runtime inference")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional strict PT checkpoint for comparison")
    parser.add_argument("--frames", type=int, default=0, help="Process only the first N frames (0 = all)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from onnx.utils import create_ort_session

    sample_rate, n_fft, hop, win = 16000, 512, 256, 512
    try:
        session = create_ort_session(args.onnx)
        runner = _preflight_onnx_session(session)
    except Exception as exc:
        parser.error(f"Failed to validate ONNX runtime metadata: {exc}")

    # 固定前端必须与 artifact STFT metadata 对齐，避免错误参数拖到首帧 shape 校验。
    try:
        _require_stft_parameters_match(
            runner.spec.stft,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            sample_rate=sample_rate,
        )
    except ValueError as exc:
        parser.error(str(exc))

    wav = _load_wav(args.input, sample_rate)
    _, n_samples = wav.shape
    window = torch.hann_window(win)
    spec = torch.view_as_real(torch.stft(
        torch.nn.functional.pad(wav, (0, hop)), n_fft=n_fft, hop_length=hop,
        win_length=win, window=window, return_complex=True, center=True,
    ))
    frame_count = spec.shape[2] if args.frames <= 0 else min(args.frames, spec.shape[2])
    spec = spec[:, :, :frame_count, :]

    pt_frames: list[torch.Tensor] = []
    if args.checkpoint:
        from onnx.models.streaming_ulunas import StreamingULUNAS
        from onnx.verify.verify_common import load_checkpoint_strict
        from tasks.se.models.discriminative.ulunas import ULUNAS

        base = ULUNAS(n_fft=n_fft).eval()
        load_checkpoint_strict(base, args.checkpoint)
        stream = StreamingULUNAS(base).eval()
        cache_pt = stream.init_cache(1, "cpu", spec.dtype)
        with torch.inference_mode():
            for index in range(frame_count):
                enhanced, cache_pt = stream(spec[:, :, index : index + 1, :], cache_pt)
                pt_frames.append(enhanced)

    result = runner.run(
        ((spec[:, :, index : index + 1, :],) for index in range(frame_count))
    )
    frame_times = list(result.frame_times)
    enhanced_spec = result.to_torch()
    enhanced_wav = torch.istft(
        torch.view_as_complex(enhanced_spec), n_fft=n_fft, hop_length=hop,
        win_length=win, window=window, center=True, length=n_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), enhanced_wav.squeeze(0).numpy().astype(np.float32), sample_rate)

    audio_seconds = n_samples / sample_rate
    total = sum(frame_times)
    print(f"Saved: {args.output}")
    print(f"Frames: {frame_count}, ORT RTF: {total / audio_seconds:.4f}, mean frame: {np.mean(frame_times) * 1000:.3f} ms")
    if pt_frames:
        pt_spec = torch.cat(pt_frames, dim=2)
        print(f"PT vs ORT MSE: {torch.mean((pt_spec - enhanced_spec) ** 2).item():.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
