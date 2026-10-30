#!/usr/bin/env python3
"""ZipEnhancer chunk-based inference (PyTorch + ONNX).

Supports both the official PyTorch checkpoint and the ONNX export from ModelScope.
Processing is done in overlapping STFT chunks to bound memory usage on long audio.

Usage (PyTorch):
    python onnx/infer/infer_chunk_zipenhancer.py \
        --checkpoint onnx/pytorch_model/zipenhancer_pytorch.bin \
        --input_dir onnx/test_audio/noisy \
        --output_dir onnx/test_audio/zipenhancer_pt \
        --device cpu

Usage (ONNX):
    python onnx/infer/infer_chunk_zipenhancer.py \
        --onnx onnx/onnx_model/zipenhancer.onnx \
        --input_dir onnx/test_audio/noisy \
        --output_dir onnx/test_audio/zipenhancer_ort \
        --device cpu

The ONNX model exported by the official checkpoint accepts magnitude / phase
(because the STFT is baked into the original pipeline).  This script handles
both interfaces transparently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

# Allow running from project root
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tasks.se.models.discriminative.zipenhancer import ZipEnhancer
from core.stft import build_stft_istft
from core.utils.safe_load import safe_torch_load
from onnx.utils import create_ort_session

STFT_CFG = {
    "n_fft": 400,
    "hop_length": 100,
    "win_length": 400,
    "window": "hann",
    "center": True,
}

TARGET_SR = 16000
_AUDIO_EXTS = {".wav", ".flac", ".mp3"}

# ZipEnhancer uses power-law compression; the ONNX export from ModelScope
# assumes the caller feeds raw magnitude (un-compressed) and phase.
# The model internally does compression / decompression.
COMPRESS_FACTOR = 0.3


def load_pt_model(checkpoint_path: str, device: str) -> ZipEnhancer:
    """Load ZipEnhancer from official PyTorch checkpoint."""
    ckpt = safe_torch_load(
        checkpoint_path,
        map_location=device,
        allow_unsafe=True,
        purpose="trusted official ZipEnhancer checkpoint compatibility",
    )
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("generator"), dict):
        raise RuntimeError("ZipEnhancer checkpoint must contain a generator state dict")
    sd = ckpt["generator"]

    # Official checkpoint uses TSConformer; our impl uses ts_conformer
    renamed = {
        k.replace("TSConformer", "ts_conformer"): v
        for k, v in sd.items()
    }

    model = ZipEnhancer(
        dense_channel=64,
        num_tsconformers=4,
        downsampling_factor=(1, 2, 2, 1),
        f_downsampling_factor=(1, 2, 2, 1),
        encoder_dim=(64, 64, 64, 64),
        num_encoder_layers=(1, 1, 1, 1),
        encoder_unmasked_dim=(64, 64, 64, 64),
        num_heads=4,
        query_head_dim=12,
        pos_head_dim=4,
        value_head_dim=8,
        feedforward_dim=(256, 256, 256, 256),
        cnn_module_kernel=15,
        pos_dim=24,
        compress_factor=COMPRESS_FACTOR,
        causal=False,
    )
    missing, unexpected = model.load_state_dict(renamed, strict=True)
    if missing:
        raise RuntimeError(f"Missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")

    model.to(device).eval()
    return model


def _sine_window(size: int, device: torch.device | str) -> torch.Tensor:
    """Sine window for overlap-add (perfect reconstruction with 50 % overlap)."""
    return torch.sin(
        torch.linspace(0, torch.pi, size, device=device)
    ).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, size, 1)


def _hann_window(size: int, device: torch.device | str) -> torch.Tensor:
    """Hann window."""
    return torch.hann_window(size, device=device).view(1, 1, size, 1)


def enhance_pt_chunked(
    model: ZipEnhancer,
    stft,
    wav: torch.Tensor,
    chunk_frames: int,
    hop_frames: int,
) -> torch.Tensor:
    """Chunked enhancement with overlap-add (PyTorch model).

    Args:
        model: ZipEnhancer instance.
        stft: STFT/iSTFT module.
        wav: (1, N) waveform.
        chunk_frames: Number of STFT frames per chunk.
        hop_frames: Hop between consecutive chunks.

    Returns:
        enhanced_wav: (1, N) waveform.
    """
    spec = stft(wav)  # (1, F, T, 2)
    _, F, T, _ = spec.shape
    dev = spec.device

    if T <= chunk_frames:
        with torch.no_grad():
            enh_spec = model(spec)
        return stft.inverse(enh_spec, wav.shape[-1])

    win = _sine_window(chunk_frames, dev)  # (1, 1, chunk, 1)
    out_acc = torch.zeros_like(spec)
    weight_acc = torch.zeros(1, 1, T, 1, device=dev)

    last_end = 0
    for start in range(0, T - chunk_frames + 1, hop_frames):
        end = start + chunk_frames
        chunk = spec[:, :, start:end, :]
        chunk_win = chunk * win
        with torch.no_grad():
            enh_chunk = model(chunk_win)
        out_acc[:, :, start:end, :] += enh_chunk * win
        weight_acc[:, :, start:end, :] += win
        last_end = end

    # Handle tail (last incomplete chunk)
    if last_end < T:
        start = T - chunk_frames
        end = T
        chunk = spec[:, :, start:end, :]
        chunk_win = chunk * win
        with torch.no_grad():
            enh_chunk = model(chunk_win)
        out_acc[:, :, start:end, :] += enh_chunk * win
        weight_acc[:, :, start:end, :] += win

    out_acc = out_acc / (weight_acc + 1e-8)
    return stft.inverse(out_acc, wav.shape[-1])


def enhance_onnx_chunked(
    sess,
    stft,
    wav: torch.Tensor,
    chunk_frames: int,
    hop_frames: int,
) -> torch.Tensor:
    """Chunked enhancement with overlap-add (ONNX Runtime).

    The ONNX model takes magnitude / phase (B, F, T) and returns
    enhanced magnitude / phase of the same shape.
    """
    spec = stft(wav)  # (1, F, T, 2)
    _, F, T, _ = spec.shape
    dev = spec.device

    real = spec[..., 0]
    imag = spec[..., 1]
    mag = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-9)
    pha = torch.atan2(imag, real + 1e-5)

    if T <= chunk_frames:
        mag_np = mag.cpu().numpy()
        pha_np = pha.cpu().numpy()
        amp_g, pha_g = sess.run(None, {"noisy_mag": mag_np, "noisy_pha": pha_np})
        enh_mag = torch.from_numpy(amp_g).to(dev)
        enh_pha = torch.from_numpy(pha_g).to(dev)
        enh_spec = torch.stack([
            enh_mag * torch.cos(enh_pha),
            enh_mag * torch.sin(enh_pha),
        ], dim=-1)
        return stft.inverse(enh_spec, wav.shape[-1])

    # Overlap-add in magnitude / phase domain
    win = _sine_window(chunk_frames, dev).squeeze(-1)  # (1, 1, chunk)
    mag_acc = torch.zeros_like(mag)
    pha_sin_acc = torch.zeros_like(mag)
    pha_cos_acc = torch.zeros_like(mag)
    weight_acc = torch.zeros(1, 1, T, device=dev)

    last_end = 0
    for start in range(0, T - chunk_frames + 1, hop_frames):
        end = start + chunk_frames
        mag_chunk = mag[:, :, start:end]
        pha_chunk = pha[:, :, start:end]

        mag_np = mag_chunk.cpu().numpy()
        pha_np = pha_chunk.cpu().numpy()
        amp_g, pha_g = sess.run(None, {"noisy_mag": mag_np, "noisy_pha": pha_np})

        enh_mag = torch.from_numpy(amp_g).to(dev)
        enh_pha = torch.from_numpy(pha_g).to(dev)

        w = win.squeeze(0)  # (1, chunk)
        mag_acc[:, :, start:end] += enh_mag * w
        pha_sin_acc[:, :, start:end] += torch.sin(enh_pha) * w
        pha_cos_acc[:, :, start:end] += torch.cos(enh_pha) * w
        weight_acc[:, :, start:end] += w
        last_end = end

    # Tail
    if last_end < T:
        start = T - chunk_frames
        end = T
        mag_chunk = mag[:, :, start:end]
        pha_chunk = pha[:, :, start:end]

        mag_np = mag_chunk.cpu().numpy()
        pha_np = pha_chunk.cpu().numpy()
        amp_g, pha_g = sess.run(None, {"noisy_mag": mag_np, "noisy_pha": pha_np})

        enh_mag = torch.from_numpy(amp_g).to(dev)
        enh_pha = torch.from_numpy(pha_g).to(dev)

        w = win.squeeze(0)
        mag_acc[:, :, start:end] += enh_mag * w
        pha_sin_acc[:, :, start:end] += torch.sin(enh_pha) * w
        pha_cos_acc[:, :, start:end] += torch.cos(enh_pha) * w
        weight_acc[:, :, start:end] += w

    mag_acc = mag_acc / (weight_acc + 1e-8)
    pha_acc = torch.atan2(pha_sin_acc / (weight_acc + 1e-8),
                          pha_cos_acc / (weight_acc + 1e-8))

    enh_spec = torch.stack([
        mag_acc * torch.cos(pha_acc),
        mag_acc * torch.sin(pha_acc),
    ], dim=-1)
    return stft.inverse(enh_spec, wav.shape[-1])


def process_file(
    pt_model,
    ort_sess,
    stft,
    wav_path: Path,
    out_path: Path,
    chunk_frames: int,
    hop_frames: int,
    device: str,
):
    """Enhance a single file."""
    wav, sr = torchaudio.load(str(wav_path))
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.to(device)

    if pt_model is not None:
        enh = enhance_pt_chunked(pt_model, stft, wav, chunk_frames, hop_frames)
    else:
        enh = enhance_onnx_chunked(ort_sess, stft, wav, chunk_frames, hop_frames)

    enh = enh.cpu().clamp(min=-1.0, max=1.0)
    torchaudio.save(str(out_path), enh, TARGET_SR)
    print(f"  Saved: {out_path.name}")


def main():
    ap = argparse.ArgumentParser(description="ZipEnhancer chunk-based inference (PT / ONNX)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="PyTorch checkpoint (.bin / .pth)")
    ap.add_argument("--onnx", type=Path, default=None,
                    help="ONNX model path")
    ap.add_argument("--input_dir", type=Path, required=True,
                    help="Directory with noisy audio files")
    ap.add_argument("--output_dir", type=Path, required=True,
                    help="Directory to save enhanced audio")
    ap.add_argument("--chunk-frames", type=int, default=400,
                    help="STFT frames per chunk (default: 400 ≈ 4 s)")
    ap.add_argument("--hop-frames", type=int, default=200,
                    help="Hop between chunks (default: 200 ≈ 2 s, 50 %% overlap)")
    ap.add_argument("--device", type=str, default="cpu",
                    help="torch device (cpu or cuda:0)")
    args = ap.parse_args()

    if args.checkpoint is None and args.onnx is None:
        ap.error("At least one of --checkpoint or --onnx must be provided")
    if args.checkpoint is not None and not args.checkpoint.exists():
        ap.error(f"Checkpoint not found: {args.checkpoint}")
    if args.onnx is not None and not args.onnx.exists():
        ap.error(f"ONNX model not found: {args.onnx}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pt_model = None
    ort_sess = None

    if args.checkpoint is not None:
        print(f"[Load PT] {args.checkpoint}")
        pt_model = load_pt_model(str(args.checkpoint), args.device)
        params = sum(p.numel() for p in pt_model.parameters())
        print(f"[Load PT] Params: {params:,} ({params / 1e6:.2f}M)")

    if args.onnx is not None:
        print(f"[Load ONNX] {args.onnx}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if args.device.startswith("cuda") else ["CPUExecutionProvider"]
        ort_sess = create_ort_session(args.onnx, providers=providers)
        in_names = [info.name for info in ort_sess.get_inputs()]
        out_names = [info.name for info in ort_sess.get_outputs()]
        print(f"[Load ONNX] Inputs: {in_names}, Outputs: {out_names}")

    stft = build_stft_istft(STFT_CFG).to(args.device)

    files = sorted([f for f in args.input_dir.iterdir()
                    if f.suffix.lower() in _AUDIO_EXTS])
    if not files:
        print(f"[Warn] No audio files found in {args.input_dir}")
        return

    mode_str = "PT" if pt_model else "ONNX"
    print(f"[Infer] {mode_str} | {len(files)} file(s) | "
          f"chunk={args.chunk_frames} hop={args.hop_frames}")

    for wav_path in files:
        out_path = args.output_dir / f"{wav_path.stem}_enhanced{wav_path.suffix}"
        print(f"  Processing: {wav_path.name}")
        process_file(
            pt_model, ort_sess, stft,
            wav_path, out_path,
            args.chunk_frames, args.hop_frames, args.device,
        )

    print("[Done]")


if __name__ == "__main__":
    main()
