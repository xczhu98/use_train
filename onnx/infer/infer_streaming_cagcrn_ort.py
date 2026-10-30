#!/usr/bin/env python3
"""流式 CAGCRN 推理脚本：PyTorch / ONNX Runtime 单帧 STFT 逐帧处理。

用法示例::

    # PyTorch 推理
    python onnx/infer/infer_streaming_cagcrn_ort.py \
        --checkpoint exp/aec/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1/checkpoints/.../best_model.tar \
        --mic data/test/mic.wav --ref data/test/far.wav \
        --output enhanced_pt.wav

    # ONNX Runtime 纯推理（无需 checkpoint，要求完整版本化 runtime metadata）
    python onnx/infer/infer_streaming_cagcrn_ort.py \
        --onnx cagcrn_stream.onnx \
        --mic data/test/mic.wav --ref data/test/far.wav \
        --output enhanced.wav

    # ONNX Runtime + PT 对照（需 checkpoint）
    python onnx/infer/infer_streaming_cagcrn_ort.py \
        --checkpoint ... --onnx cagcrn_stream.onnx \
        --mic ... --ref ... --output enhanced.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
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
    """严格校验完整 cache、STFT、I/O 与版本化 delay metadata。"""
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    runner = StreamingORTFrameRunner(session, "cagcrn", batch_size=1)
    if runner.spec.cagcrn is None:
        raise RuntimeError("CAGCRN runtime metadata was not parsed")
    return runner, runner.spec.cagcrn


def _resolve_stft_parameters(artifact, overrides: dict[str, object]):
    """以 artifact 为真源；显式 CLI 参数只能重复相同值。"""
    from onnx.runtime_spec import STFTRuntimeSpec

    names = ("n_fft", "hop_length", "win_length", "sample_rate", "window", "center")
    values = artifact.to_metadata()
    missing = tuple(name for name in names if values[name] is None)
    if missing:
        raise ValueError(f"CAGCRN artifact STFT metadata is incomplete: {missing}")
    for name, value in overrides.items():
        if value is not None and value != values[name]:
            raise ValueError(
                f"CLI {name}={value!r} does not match ONNX metadata {values[name]!r}"
            )
    return STFTRuntimeSpec(**values)


def _hann_window(window: str, win_length: int, device: torch.device) -> torch.Tensor:
    if window != "hann":
        raise ValueError(f"CAGCRN supports only window='hann', got {window!r}")
    return torch.hann_window(win_length, device=device)


def main() -> int:
    ap = argparse.ArgumentParser(description="Streaming CAGCRN inference (PT / ORT)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="PyTorch checkpoint (.tar / .pth); if provided, run PT streaming inference")
    ap.add_argument("--onnx", type=Path, default=None, help="ONNX model path; if provided, run ORT streaming inference")
    ap.add_argument("--mic", type=Path, required=True, help="Mic input wav")
    ap.add_argument("--ref", type=Path, required=True, help="Far-end reference wav")
    ap.add_argument("--output", type=Path, required=True, help="Output wav")
    ap.add_argument("--n-fft", type=int, default=None, help="必须与 ONNX STFT metadata 一致")
    ap.add_argument("--hop-length", type=int, default=None, help="必须与 ONNX STFT metadata 一致")
    ap.add_argument("--win-length", type=int, default=None, help="必须与 ONNX STFT metadata 一致")
    ap.add_argument("--sample-rate", type=int, default=None, help="必须与 ONNX STFT metadata 一致")
    ap.add_argument("--window", type=str, default=None, help="必须与 ONNX STFT metadata 一致")
    ap.add_argument("--device", type=str, default="cpu", help="PyTorch device")
    ap.add_argument("--ort-cuda", action="store_true", help="ORT CUDAExecutionProvider")
    center_group = ap.add_mutually_exclusive_group()
    center_group.add_argument("--center", dest="center", action="store_true", help="要求 STFT center=True")
    center_group.add_argument("--no-center", dest="center", action="store_false", help="要求 STFT center=False")
    ap.set_defaults(center=None)
    ap.add_argument("--progress-every", type=int, default=500, help="Print progress every N frames")
    ns = ap.parse_args()

    if not ns.checkpoint and not ns.onnx:
        ap.error("Must provide at least one of --checkpoint or --onnx")

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from onnx.utils import create_ort_session

    ort_providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ns.ort_cuda else ["CPUExecutionProvider"]
    )
    runner = None
    cagcrn_runtime = None
    if ns.onnx:
        try:
            session = create_ort_session(ns.onnx.resolve(), providers=ort_providers)
            runner, cagcrn_runtime = _preflight_onnx_session(session)
        except Exception as exc:
            ap.error(f"Failed to validate ONNX runtime metadata: {exc}")

    overrides = {
        "n_fft": ns.n_fft,
        "hop_length": ns.hop_length,
        "win_length": ns.win_length,
        "sample_rate": ns.sample_rate,
        "window": ns.window,
        "center": ns.center,
    }
    if runner is not None:
        try:
            stft_spec = _resolve_stft_parameters(runner.spec.stft, overrides)
        except ValueError as exc:
            ap.error(str(exc))
    else:
        from onnx.runtime_spec import STFTRuntimeSpec

        n_fft_default = ns.n_fft or 512
        stft_spec = STFTRuntimeSpec(
            n_fft=n_fft_default,
            hop_length=ns.hop_length or 256,
            win_length=ns.win_length or n_fft_default,
            sample_rate=ns.sample_rate or 16000,
            window=ns.window or "hann",
            center=False if ns.center is None else ns.center,
        )

    dev = torch.device(ns.device)
    sr = int(stft_spec.sample_rate)
    n_fft = int(stft_spec.n_fft)
    hop = int(stft_spec.hop_length)
    win_length = int(stft_spec.win_length)
    center = bool(stft_spec.center)

    stream = None
    if ns.checkpoint:
        from onnx.models.streaming_cagcrn import StreamingCAGCRN
        from tasks.aec.models.discriminative.cagcrn import CAGCRN
        from onnx.verify.verify_common import load_checkpoint_strict

        base_model = CAGCRN(
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_length,
            window=stft_spec.window,
            center=center,
            sample_rate=sr,
        ).eval().to(dev)
        load_checkpoint_strict(base_model, ns.checkpoint)
        stream = StreamingCAGCRN(base_model).eval().to(dev)
        if cagcrn_runtime is not None and (
            stream.encoder.stream_align.d != cagcrn_runtime.delay_d
            or stream.tfag.algorithmic_delay_frames
            != cagcrn_runtime.algorithmic_delay_frames
        ):
            raise RuntimeError("Checkpoint wrapper delay contract does not match ONNX metadata")

    # Load audio
    mic_wav = _load_wav(ns.mic, sr, dev)
    ref_wav = _load_wav(ns.ref, sr, dev)
    if mic_wav.shape != ref_wav.shape:
        min_len = min(mic_wav.shape[-1], ref_wav.shape[-1])
        mic_wav = mic_wav[:, :min_len]
        ref_wav = ref_wav[:, :min_len]
    B, N = mic_wav.shape
    audio_sec = N / sr

    # STFT
    window = _hann_window(stft_spec.window, win_length, dev)
    mic_spec = torch.stft(
        mic_wav, n_fft=n_fft, hop_length=hop, win_length=win_length,
        window=window, return_complex=True, center=center,
    )  # (B, F, T)
    mic_spec = torch.view_as_real(mic_spec)  # (B, F, T, 2)
    ref_spec = torch.stft(
        ref_wav, n_fft=n_fft, hop_length=hop, win_length=win_length,
        window=window, return_complex=True, center=center,
    )
    ref_spec = torch.view_as_real(ref_spec)  # (B, F, T, 2)
    _, F_bins, T, _ = mic_spec.shape

    # ------------------------------------------------------------------
    # PyTorch streaming inference (only when checkpoint is provided)
    # ------------------------------------------------------------------
    if ns.checkpoint:
        cache = stream.init_cache(B, str(dev), mic_spec.dtype)
        outs: list[torch.Tensor] = []
        pt_times: list[float] = []

        t0 = time.perf_counter()
        with torch.inference_mode():
            for t in range(T):
                if ns.progress_every and t > 0 and t % ns.progress_every == 0:
                    print(f"  PT frame {t}/{T}", flush=True)
                sp = mic_spec[:, :, t:t+1, :]
                fp = ref_spec[:, :, t:t+1, :]
                tic = time.perf_counter()
                enh, cache = stream(sp, fp, cache)
                pt_times.append(time.perf_counter() - tic)
                outs.append(enh)
        pt_total = time.perf_counter() - t0

        enh_spec = torch.cat(outs, dim=2)
        enh_wav = torch.istft(
            torch.view_as_complex(enh_spec),
            n_fft=n_fft, hop_length=hop, win_length=win_length,
            window=window, center=center, length=N,
        )

        # Save
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
    # ONNX Runtime inference (optional)
    # ------------------------------------------------------------------
    if ns.onnx is not None and ns.onnx.exists():
        if runner is None:
            raise RuntimeError("ONNX session did not complete ABI preflight")
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
            )
            ort_frames.append(frame)
            ort_times.append(elapsed)
        ort_total = time.perf_counter() - t0
        ort_spec = torch.from_numpy(np.concatenate(ort_frames, axis=2)).to(dev)
        ort_wav = torch.istft(
            torch.view_as_complex(ort_spec),
            n_fft=n_fft, hop_length=hop, win_length=win_length,
            window=window, center=center, length=N,
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
        print(f"  ORT total time: {ort_total:.3f}s  RTF: {ort_rtf:.4f}")
        print(f"  ORT per-frame:  mean={ort_mean_ms:.2f}ms  max={ort_max_ms:.2f}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
