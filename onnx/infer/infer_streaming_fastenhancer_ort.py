#!/usr/bin/env python3
"""流式 FastEnhancer 推理脚本：PyTorch / ONNX Runtime 单帧 STFT 逐帧处理。

用法示例::

    # ONNX Runtime 纯推理（无需 checkpoint，要求 ONNX 含 fastenhancer_cache_cfg metadata）
    python onnx/infer/infer_streaming_fastenhancer_ort.py \
        --onnx onnx/onnx_model/fastenhancer_b_stream.onnx \
        --input onnx/test_audio/noisy/voicebank_noisy.wav \
        --output enhanced.wav

    # ONNX Runtime + PT 对照（需 checkpoint）
    python onnx/infer/infer_streaming_fastenhancer_ort.py \
        --checkpoint ... --onnx onnx/onnx_model/fastenhancer_b_stream.onnx \
        --input noisy.wav --output enhanced.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore[import]
import torch
import torchaudio


def _load_wav(path: Path, target_sr: int, device: torch.device) -> torch.Tensor:
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = torch.from_numpy(data.T.copy())
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.to(device)


def _preflight_onnx_session(session):
    """在读取音频前严格校验 ONNX runtime metadata 与 I/O ABI。"""
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    return StreamingORTFrameRunner(session, "fastenhancer", batch_size=1)


def _reconstruct_waveform(
    spectra: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    center: bool,
    length: int,
) -> torch.Tensor:
    """使用 artifact 绑定的 STFT 参数合成 waveform。"""
    return torch.istft(
        torch.view_as_complex(spectra),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=length,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Streaming FastEnhancer inference (PT / ORT)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="PyTorch checkpoint (.pth / .tar); if provided, run PT streaming inference")
    ap.add_argument("--onnx", type=Path, required=True, help="ONNX model path")
    ap.add_argument("--input", type=Path, required=True, help="Input wav")
    ap.add_argument("--output", type=Path, required=True, help="Output wav")
    ap.add_argument("--variant", type=str, default="B", choices=["T", "B", "S", "M", "L"])
    ap.add_argument("--n-fft", type=int, default=None, help="只能重复 artifact metadata 中的 n_fft")
    ap.add_argument("--hop-size", type=int, default=None, help="只能重复 artifact metadata 中的 hop_length")
    ap.add_argument("--win-size", type=int, default=None, help="只能重复 artifact metadata 中的 win_length")
    ap.add_argument("--sample-rate", type=int, default=None, help="只能重复 artifact metadata 中的 sample_rate")
    ap.add_argument("--device", type=str, default="cpu", help="PyTorch device")
    ap.add_argument("--ort-cuda", action="store_true", help="ORT CUDAExecutionProvider")
    ap.add_argument("--progress-every", type=int, default=500, help="Print progress every N frames")
    ns = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from onnx.utils import create_ort_session

    prov = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ns.ort_cuda else ["CPUExecutionProvider"]
    )
    try:
        sess = create_ort_session(ns.onnx.resolve(), providers=prov)
        runner = _preflight_onnx_session(sess)
    except Exception as exc:
        ap.error(f"Failed to validate ONNX runtime metadata: {exc}")

    stft = runner.spec.stft
    required = ("n_fft", "hop_length", "win_length", "sample_rate", "center")
    missing = tuple(name for name in required if getattr(stft, name) is None)
    if missing or stft.window != "hann":
        ap.error(f"FastEnhancer artifact STFT metadata is incomplete or unsupported: {missing}")
    overrides = {
        "n_fft": ns.n_fft,
        "hop_length": ns.hop_size,
        "win_length": ns.win_size,
        "sample_rate": ns.sample_rate,
    }
    for name, requested in overrides.items():
        if requested is not None and requested != getattr(stft, name):
            ap.error(
                f"--{name.replace('_', '-')}={requested!r} conflicts with "
                f"ONNX metadata {getattr(stft, name)!r}"
            )

    dev = torch.device(ns.device)
    sr = int(stft.sample_rate)
    n_fft = int(stft.n_fft)
    hop = int(stft.hop_length)
    win = int(stft.win_length)
    center = bool(stft.center)

    # Load audio
    wav = _load_wav(ns.input, sr, dev)
    B, N = wav.shape
    audio_sec = N / sr

    # 图外 STFT 参数由经过 preflight 的 artifact metadata 唯一确定。
    wav_padded = torch.nn.functional.pad(wav, (0, hop))
    window = torch.hann_window(win, device=dev)
    spec = torch.view_as_real(torch.stft(
        wav_padded, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=center,
    ))  # (B, F, T, 2)
    _, F_bins, T, _ = spec.shape

    # ------------------------------------------------------------------
    # PyTorch streaming inference (only when checkpoint is provided)
    # ------------------------------------------------------------------
    if ns.checkpoint:
        from tasks.se.models.discriminative.fastenhancer import FastEnhancer
        from onnx.models.streaming_fastenhancer import StreamingFastEnhancer
        from onnx.verify.verify_common import load_checkpoint_strict

        base_model = FastEnhancer(variant=ns.variant, n_fft=n_fft).eval().to(dev)
        load_checkpoint_strict(base_model, ns.checkpoint)

        stream = StreamingFastEnhancer(
            base_model, hop_length=hop, win_length=win, sample_rate=sr,
            center=center,
        ).eval().to(dev)
        cache = stream.init_cache(B, dev, spec.dtype)
        outs: list[torch.Tensor] = []
        pt_times: list[float] = []

        t0 = time.perf_counter()
        with torch.inference_mode():
            for t in range(T):
                if ns.progress_every and t > 0 and t % ns.progress_every == 0:
                    print(f"  PT frame {t}/{T}", flush=True)
                sp = spec[:, :, t:t + 1, :]
                tic = time.perf_counter()
                enh, cache = stream(sp, cache)
                pt_times.append(time.perf_counter() - tic)
                outs.append(enh)
        pt_total = time.perf_counter() - t0

        enh_spec = torch.cat(outs, dim=2)
        enh_wav = torch.istft(
            torch.view_as_complex(enh_spec),
            n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, center=center, length=N,
        )

        out_dir = ns.output.parent
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(ns.output),
            enh_wav.squeeze(0).cpu().numpy().astype(np.float32),
            sr,
            subtype="PCM_16",
        )

        pt_rtf = pt_total / audio_sec if audio_sec > 0 else float("nan")
        pt_measure_times = pt_times[1:] or pt_times  # 单帧时保留唯一计时样本
        pt_mean_ms = np.mean(pt_measure_times) * 1000 if pt_measure_times else float("nan")
        pt_max_ms = np.max(pt_measure_times) * 1000 if pt_measure_times else float("nan")
        print(f"Saved: {ns.output}")
        print(f"  Audio duration: {audio_sec:.2f}s  STFT frames: {T}")
        print(f"  PT total time:  {pt_total:.3f}s  RTF: {pt_rtf:.4f}")
        print(f"  PT per-frame:   mean={pt_mean_ms:.2f}ms  max={pt_max_ms:.2f}ms")

    # ------------------------------------------------------------------
    # ONNX Runtime inference
    # ------------------------------------------------------------------
    if B != runner.batch_size:
        raise RuntimeError(
            f"Preflight batch size {runner.batch_size} does not match audio batch {B}"
        )

    ort_frames: list[np.ndarray] = []
    ort_times: list[float] = []
    t0 = time.perf_counter()
    for t in range(T):
        if ns.progress_every and t > 0 and t % ns.progress_every == 0:
            print(f"  ORT frame {t}/{T}", flush=True)
        frame, elapsed = runner.run_frame(spec[:, :, t:t + 1, :])
        ort_frames.append(frame)
        ort_times.append(elapsed)
    ort_total = time.perf_counter() - t0
    ort_spec = torch.from_numpy(np.concatenate(ort_frames, axis=2)).to(dev)
    ort_wav = _reconstruct_waveform(
        ort_spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=center,
        length=N,
    )

    if ns.checkpoint:
        ort_out_path = ns.output.parent / (ns.output.stem + "_ort.wav")
    else:
        ort_out_path = ns.output
        out_dir = ns.output.parent
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(ort_out_path),
        ort_wav.squeeze(0).cpu().numpy().astype(np.float32),
        sr,
        subtype="PCM_16",
    )

    ort_rtf = ort_total / audio_sec if audio_sec > 0 else float("nan")
    ort_measure_times = ort_times[1:] or ort_times  # 单帧时保留唯一计时样本
    ort_mean_ms = np.mean(ort_measure_times) * 1000 if ort_measure_times else float("nan")
    ort_max_ms = np.max(ort_measure_times) * 1000 if ort_measure_times else float("nan")
    print(f"Saved ORT: {ort_out_path}")
    print(f"  Audio duration: {audio_sec:.2f}s  STFT frames: {T}")
    print(f"  ORT total time: {ort_total:.3f}s  RTF: {ort_rtf:.4f}")
    print(f"  ORT per-frame:  mean={ort_mean_ms:.2f}ms  max={ort_max_ms:.2f}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
