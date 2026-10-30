#!/usr/bin/env python3
"""StreamingGTCRN frame-by-frame inference (ONNX Runtime).

Runs ``gtcrn_stream.onnx`` (StreamingGTCRN, n_fft=512/hop=256) on a single
waveform: 16k resample -> STFT -> per-frame ONNX step with cache update ->
iSTFT overlap-add -> 16k enhanced output.

Usage:
    python onnx/infer/infer_streaming_gtcrn_ort.py \
        --onnx onnx/onnx_model/gtcrn_stream.onnx \
        --input exp/eval/onnx_badcase/in/badcase-meeting-women.wav \
        --output exp/eval/onnx_badcase/gtcrn/badcase-meeting-women.wav \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.stft import StftIstftModule

TARGET_SR = 16000
STFT_CFG = dict(n_fft=512, hop_length=256, win_length=512, window="hann")


def _preflight_onnx_session(session):
    """在读取音频前严格校验 ONNX runtime metadata 与 I/O ABI。"""
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    return StreamingORTFrameRunner(session, "gtcrn", batch_size=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="StreamingGTCRN ONNX frame-by-frame inference")
    ap.add_argument("--onnx", type=Path, required=True, help="GTCRN streaming ONNX path")
    ap.add_argument("--input", type=Path, required=True, help="input wav")
    ap.add_argument("--output", type=Path, required=True, help="output wav path")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    if not args.onnx.exists():
        ap.error(f"ONNX not found: {args.onnx}")
    if not args.input.exists():
        ap.error(f"Input not found: {args.input}")

    from onnx.utils import create_ort_session

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.type == "cuda"
        else ["CPUExecutionProvider"]
    )
    try:
        sess = create_ort_session(args.onnx, providers=providers)
        runner = _preflight_onnx_session(sess)
    except Exception as exc:
        ap.error(f"Failed to validate ONNX runtime metadata: {exc}")
    print(f"[Load ONNX] {args.onnx}")
    print(f"[Load ONNX] inputs={[info.name for info in sess.get_inputs()]}")

    wav, sr = torchaudio.load(str(args.input))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    wav = wav.to(device)

    stft = StftIstftModule(**STFT_CFG).to(device)

    with torch.no_grad():
        spec = stft(wav)  # (1, F=257, T, 2)
    spec = spec.cpu()
    B, F, T, _ = spec.shape
    print(f"[Infer] feat {TARGET_SR}Hz, frames={T}, device={device}")

    if B != runner.batch_size:
        raise RuntimeError(
            f"Preflight batch size {runner.batch_size} does not match audio batch {B}"
        )
    result = runner.run(
        ((spec[:, :, t : t + 1, :].numpy(),) for t in range(T))
    )
    enh_spec = result.to_torch().to(device)  # (1, F, T, 2)
    with torch.no_grad():
        y16 = stft.inverse(enh_spec, wav.shape[-1])
    y16 = y16.cpu().clamp(min=-1.0, max=1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), y16, TARGET_SR)
    print(f"[Done] -> {args.output} ({TARGET_SR}Hz)")


if __name__ == "__main__":
    main()
