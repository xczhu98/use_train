#!/usr/bin/env python3
"""流式 Ultra-DPC-AEC 推理脚本：PyTorch / ONNX Runtime 单帧 STFT 逐帧处理。

用法示例::

    # ONNX Runtime 纯推理（无需 checkpoint，要求 ONNX 含 ultra_dpc_cache_cfg metadata）
    python onnx/infer/infer_streaming_ultra_dpc_ort.py \
        --onnx runs/onnx/ultra_dpc_test.onnx \
        --mic mic.wav --ref ref.wav --err err.wav \
        --output enhanced.wav

    # ONNX Runtime + PT 对照（需 checkpoint）
    python onnx/infer/infer_streaming_ultra_dpc_ort.py \
        --checkpoint path/to/checkpoint.tar \
        --onnx runs/onnx/ultra_dpc_test.onnx \
        --mic mic.wav --ref ref.wav --err err.wav \
        --output enhanced.wav
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

    return StreamingORTFrameRunner(session, "ultra_dpc", batch_size=1)


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
    ap = argparse.ArgumentParser(description="Streaming Ultra-DPC-AEC inference (PT / ORT)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="PyTorch checkpoint (.pth / .tar)")
    ap.add_argument("--onnx", type=Path, required=True, help="ONNX model path")
    ap.add_argument("--mic", type=Path, required=True, help="Mic wav")
    ap.add_argument("--ref", type=Path, required=True, help="Reference wav")
    ap.add_argument("--err", type=Path, required=True, help="Error signal wav (from linear AEC)")
    ap.add_argument("--output", type=Path, required=True, help="Output wav")
    ap.add_argument("--n-fft", type=int, default=320)
    ap.add_argument("--hop-size", type=int, default=160)
    ap.add_argument("--win-size", type=int, default=320)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--ort-cuda", action="store_true")
    ap.add_argument("--progress-every", type=int, default=500)
    ns = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from onnx.utils import create_ort_session

    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ns.ort_cuda else ["CPUExecutionProvider"]
    try:
        sess = create_ort_session(ns.onnx.resolve(), providers=prov)
        runner = _preflight_onnx_session(sess)
    except Exception as exc:
        ap.error(f"Failed to validate ONNX runtime metadata: {exc}")

    # 读取音频前先与 artifact STFT metadata 对齐，避免错误参数拖到首帧 shape 校验。
    try:
        _require_stft_parameters_match(
            runner.spec.stft,
            n_fft=ns.n_fft,
            hop_length=ns.hop_size,
            win_length=ns.win_size,
            sample_rate=ns.sample_rate,
        )
    except ValueError as exc:
        ap.error(str(exc))

    dev = torch.device(ns.device)
    sr = ns.sample_rate
    n_fft = ns.n_fft
    hop = ns.hop_size
    win = ns.win_size

    mic_wav = _load_wav(ns.mic, sr, dev)
    ref_wav = _load_wav(ns.ref, sr, dev)
    err_wav = _load_wav(ns.err, sr, dev)
    B, N = mic_wav.shape
    audio_sec = N / sr

    window = torch.hann_window(win, device=dev)
    mic_spec = torch.view_as_real(torch.stft(
        mic_wav, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=True,
    ))
    ref_spec = torch.view_as_real(torch.stft(
        ref_wav, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=True,
    ))
    err_spec = torch.view_as_real(torch.stft(
        err_wav, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=True,
    ))
    _, F_bins, T, _ = mic_spec.shape

    # ------------------------------------------------------------------
    # PyTorch streaming inference
    # ------------------------------------------------------------------
    if ns.checkpoint:
        from tasks.aec.models.discriminative.ultra_dpc_aec import UltraDPCAEC
        from onnx.models.streaming_ultra_dpc import StreamingUltraDPCAEC
        from onnx.verify.verify_common import load_checkpoint_strict

        base_model = UltraDPCAEC(n_fft=n_fft, hop_length=hop, embed_dim=48, n_dp_blocks=4,
                                 encoder_layers=2, decoder_layers=2, gru_dim=32,
                                 freq_ratio=1, time_skip=1).eval().to(dev)
        load_checkpoint_strict(base_model, ns.checkpoint)

        stream = StreamingUltraDPCAEC(base_model).eval().to(dev)
        cache = stream.init_cache(B, dev, mic_spec.dtype)
        outs: list[torch.Tensor] = []
        pt_times: list[float] = []

        t0 = time.perf_counter()
        with torch.inference_mode():
            for t in range(T):
                if ns.progress_every and t > 0 and t % ns.progress_every == 0:
                    print(f"  PT frame {t}/{T}", flush=True)
                tic = time.perf_counter()
                enh, cache = stream(
                    mic_spec[:, :, t:t + 1, :],
                    ref_spec[:, :, t:t + 1, :],
                    err_spec[:, :, t:t + 1, :],
                    cache,
                )
                pt_times.append(time.perf_counter() - tic)
                outs.append(enh)
        pt_total = time.perf_counter() - t0

        enh_spec = torch.cat(outs, dim=2)
        enh_wav = torch.istft(
            torch.view_as_complex(enh_spec),
            n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, center=True, length=N,
        )

        out_dir = ns.output.parent
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(ns.output), enh_wav.squeeze(0).cpu().numpy().astype(np.float32), sr, subtype="PCM_16")

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
        frame, elapsed = runner.run_frame(
            mic_spec[:, :, t:t + 1, :],
            ref_spec[:, :, t:t + 1, :],
            err_spec[:, :, t:t + 1, :],
        )
        ort_frames.append(frame)
        ort_times.append(elapsed)
    ort_total = time.perf_counter() - t0
    ort_spec = torch.from_numpy(np.concatenate(ort_frames, axis=2)).to(dev)
    ort_wav = torch.istft(
        torch.view_as_complex(ort_spec),
        n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, center=True, length=N,
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
