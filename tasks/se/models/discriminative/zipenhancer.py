"""ZipEnhancer for speech enhancement.

Training-only interface: spectrum-in / spectrum-out. STFT/iSTFT is done by the
caller (trainer / inference pipeline).

Architecture (see Fig. 1 in paper):
  1. Input: noisy_spec (B, F, T, 2) -> mag / phase decomposition
  2. Power-law compression on magnitude (c=0.3)
  3. DenseEncoder: 2-D conv + dilated dense block -> (B, C, T, F//2)
  4. Zipformer2DualPathEncoder: dual-path Zipformer blocks with T/F down-up
     sampling -> (B, C, T, F//2)
  5. Parallel decoders:
     - MappingDecoder -> compressed clean magnitude
     - PhaseDecoder   -> clean wrapped phase
  6. Decompress magnitude and assemble complex spectrum -> (B, F, T, 2)

论文与实现参考:
  [1] H. Wang & B. Tian, "ZipEnhancer: Dual-Path Down-Up Sampling-based
      Zipformer for Monaural Speech Enhancement", arXiv:2501.05183, 2025.
  [2] Z. Yao et al., "Zipformer: A faster and better encoder for automatic
      speech recognition", ICLR 2024.
  [3] Y.-X. Lu et al., "MP-SENet: A Speech Enhancement Model with Parallel
      Denoising of Magnitude and Phase Spectra", Interspeech 2023.
  非官方参考代码: https://github.com/boreas-l/zipEnhancer

Run directly to test params and compute:
  python tasks/se/models/zipenhancer.py [--sr 16000] [--dur 1.0]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tasks.se.models.base_model import BaseSEModel
from tasks.se.models.modules.zipenhancer_impl import (
    DenseEncoder,
    MappingDecoder,
    PhaseDecoder,
    Zipformer2DualPathEncoder,
)
from tasks.se.models.modules.zipenhancer_impl.scaling import ScheduledFloat
from core.stft import _get_window
from core.utils.registry import MODEL_REGISTRY


def _validate_framework_stft_kwargs(
    *,
    hop_length: int | None,
    win_length: int | None,
    window: str | None,
    center: bool | None,
) -> None:
    """校验 ``build_model`` 注入的框架 STFT 上下文（本模型不做 STFT，不消费这些键）。"""
    for key, value in (("hop_length", hop_length), ("win_length", win_length)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(
                f"ZipEnhancer model.args.{key} must be an exact positive int, got {value!r}"
            )
    if center is not None and type(center) is not bool:
        raise ValueError(f"ZipEnhancer model.args.center must be a bool, got {center!r}")
    if window is not None:
        # 复用 core.stft 的窗名白名单，拼写错误在模型构造期即报错。
        _get_window(
            str(window).lower(),
            int(win_length or hop_length or 512),
            torch.device("cpu"),
        )


@MODEL_REGISTRY.register("ZipEnhancer")
class ZipEnhancer(BaseSEModel):
    """ZipEnhancer: spectrum-in / spectrum-out speech enhancement model.

    Args:
        dense_channel: Encoder / decoder channel width. Default 64 (ZipEnhancer-S).
        num_tsconformers: Number of dual-path Zipformer blocks. Default 4.
        downsampling_factor: Tuple of T-downsample ratios per block. Default (1, 2, 2, 1).
        f_downsampling_factor: Tuple of F-downsample ratios per block. Default (1, 2, 2, 1).
        encoder_dim: Tuple of embedding dims per block. Default (64, 64, 64, 64).
        num_encoder_layers: Tuple of layer counts per block. Default (1, 1, 1, 1).
        num_heads: Number of attention heads. Default 4.
        query_head_dim: Query dimension per head. Default 12.
        pos_head_dim: Positional encoding projection dim per head. Default 4.
        value_head_dim: Value dimension per head. Default 8.
        feedforward_dim: Tuple of FFN hidden dims per block. Default (256, 256, 256, 256).
        cnn_module_kernel: Depthwise conv kernel size. Default 15.
        pos_dim: Relative positional encoding dimension. Default 24.
        compress_factor: Magnitude power-law compression factor. Default 0.3.
        causal: If True, use causal depthwise conv (for future streaming variant).
        n_fft: FFT size; determines the accepted frequency bins (F = n_fft // 2 + 1).
        hop_length: Framework STFT context injected by ``build_model``; validated, not consumed here.
        win_length: Framework STFT context injected by ``build_model``; validated, not consumed here.
        window: Framework STFT context injected by ``build_model``; validated, not consumed here.
        center: Framework STFT context injected by ``build_model``; validated, not consumed here.
    """

    def __init__(
        self,
        dense_channel: int = 64,
        num_tsconformers: int = 4,
        downsampling_factor: tuple = (1, 2, 2, 1),
        f_downsampling_factor: tuple = (1, 2, 2, 1),
        encoder_dim: tuple = (64, 64, 64, 64),
        num_encoder_layers: tuple = (1, 1, 1, 1),
        encoder_unmasked_dim: tuple = (64, 64, 64, 64),
        num_heads: int = 4,
        query_head_dim: int = 12,
        pos_head_dim: int = 4,
        value_head_dim: int = 8,
        feedforward_dim: tuple = (256, 256, 256, 256),
        cnn_module_kernel: int = 15,
        pos_dim: int = 24,
        compress_factor: float = 0.3,
        causal: bool = False,
        n_fft: int = 512,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str | None = None,
        center: bool | None = None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(f"ZipEnhancer got unexpected keyword arguments: {sorted(kwargs)}")
        if n_fft <= 0:
            raise ValueError(f"ZipEnhancer n_fft must be positive, got {n_fft}")
        # 框架 STFT 上下文由 build_model 注入：本模型不做 STFT，故只校验取值不消费，
        # 避免 YAML 里的 window/hop_length 拼写错误被静默忽略。
        _validate_framework_stft_kwargs(
            hop_length=hop_length, win_length=win_length, window=window, center=center
        )
        self.n_fft = n_fft
        self.compress_factor = compress_factor
        self.causal = causal

        # ---- Encoder: maps [mag^c, pha] -> hidden features -----------------
        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2)

        # ---- Dual-Path Zipformer Encoder ------------------------------------
        former_conf = {
            "num_encoder_layers": num_encoder_layers,
            "downsampling_factor": downsampling_factor,
            "f_downsampling_factor": f_downsampling_factor,
            "encoder_dim": encoder_dim,
            "encoder_unmasked_dim": encoder_unmasked_dim,
            "num_heads": num_heads,
            "query_head_dim": query_head_dim,
            "pos_head_dim": pos_head_dim,
            "value_head_dim": value_head_dim,
            "feedforward_dim": feedforward_dim,
            "cnn_module_kernel": cnn_module_kernel,
            "pos_dim": pos_dim,
            "causal": causal,
            "warmup_batches": 4000.0,
        }
        self.ts_conformer = Zipformer2DualPathEncoder(
            output_downsampling_factor=1,
            dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
            **former_conf,
        )

        # ---- Parallel Decoders ----------------------------------------------
        self.mask_decoder = MappingDecoder(dense_channel, out_channel=1)
        self.phase_decoder = PhaseDecoder(dense_channel, out_channel=1)

    def forward(
        self, noisy_spec: torch.Tensor, lengths: torch.Tensor = None, fs: int = None
    ) -> torch.Tensor:
        """Spectrum-in / spectrum-out.

        Args:
            noisy_spec: (B, F, T, 2) real-valued complex spectrum.
            lengths: optional (B,) valid lengths (unused, kept for API).
            fs: optional sampling rate (unused, kept for API).

        Returns:
            enhanced_spec: (B, F, T, 2).
        """
        if noisy_spec.dim() != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(
                "ZipEnhancer accepts only spectrum input (B, F, T, 2). "
                "Do STFT in the caller (trainer/inference) and pass the result here."
            )
        if noisy_spec.shape[1] != self.n_fft // 2 + 1:
            raise ValueError(f"Expected F={self.n_fft // 2 + 1}, got F={noisy_spec.shape[1]}")
        if not noisy_spec.is_floating_point():
            raise TypeError("ZipEnhancer noisy_spec must use a floating-point dtype")

        # ---- Decompose into magnitude and phase -----------------------------
        real = noisy_spec[..., 0]
        imag = noisy_spec[..., 1]
        mag = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-9)
        pha = torch.atan2(imag, real + 1e-5)

        # ---- Power-law compression ------------------------------------------
        mag_c = torch.pow(mag, self.compress_factor)

        # ---- Stack as 2-channel image [mag^c, pha] --------------------------
        # Input shape: (B, F, T) -> (B, 2, T, F)
        mag_c = mag_c.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        pha = pha.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        x = torch.cat([mag_c, pha], dim=1)  # (B, 2, T, F)

        # ---- Encoder --------------------------------------------------------
        x = self.dense_encoder(x)  # (B, C, T, F//2)

        # ---- Dual-Path Zipformer --------------------------------------------
        x = self.ts_conformer(x)  # (B, C, T, F//2)

        # ---- Parallel decoders ----------------------------------------------
        pred_mag = self.mask_decoder(x)   # (B, 1, T, F')  F' may be F or F-1
        pred_pha = self.phase_decoder(x)  # (B, 1, T, F')

        # The encoder's stride-2 conv paired with sub-pixel upsampling is not
        # perfectly invertible for all F: when F is even the decoder outputs
        # F-1 because the final Conv2d(…, (1,2)) shaves one bin. Crop or pad to
        # match the original frequency count.
        target_f = mag_c.shape[-1]
        if pred_mag.shape[-1] != target_f:
            if pred_mag.shape[-1] > target_f:
                pred_mag = pred_mag[..., :target_f]
                pred_pha = pred_pha[..., :target_f]
            else:
                pad = target_f - pred_mag.shape[-1]
                pred_mag = torch.nn.functional.pad(pred_mag, (0, pad))
                pred_pha = torch.nn.functional.pad(pred_pha, (0, pad))

        # ---- Extract and decompress magnitude -------------------------------
        # pred_mag: (B, 1, T, F) -> (B, F, T)
        pred_mag = pred_mag[:, 0, :, :].permute(0, 2, 1).contiguous()
        pred_mag = torch.pow(pred_mag.clamp(min=0.0), 1.0 / self.compress_factor)

        # ---- Extract phase --------------------------------------------------
        # pred_pha: (B, 1, T, F) -> (B, F, T)
        pred_pha = pred_pha[:, 0, :, :].permute(0, 2, 1).contiguous()

        # ---- Reassemble complex spectrum ------------------------------------
        enhanced_real = pred_mag * torch.cos(pred_pha)
        enhanced_imag = pred_mag * torch.sin(pred_pha)
        enhanced_spec = torch.stack([enhanced_real, enhanced_imag], dim=-1)

        return enhanced_spec


# ---------------------------------------------------------------------------
# Standalone profile script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core.utils.model_profile import profile_model, make_profile_parser
    from core.stft import build_stft_istft
    import argparse

    args = argparse.ArgumentParser(
        parents=[make_profile_parser(default_sr=16000)], add_help=False
    ).parse_args()

    SAMPLE_RATE = args.sr
    STFT_CFG = {
        "n_fft": 400,
        "hop_length": 100,
        "win_length": 400,
        "window": "hann",
    }
    MODEL_ARGS = {
        "dense_channel": 64,
        "num_tsconformers": 4,
        "downsampling_factor": (1, 2, 2, 1),
        "f_downsampling_factor": (1, 2, 2, 1),
        "encoder_dim": (64, 64, 64, 64),
        "num_encoder_layers": (1, 1, 1, 1),
        "encoder_unmasked_dim": (64, 64, 64, 64),
        "num_heads": 4,
        "query_head_dim": 12,
        "pos_head_dim": 4,
        "value_head_dim": 8,
        "feedforward_dim": (256, 256, 256, 256),
        "cnn_module_kernel": 15,
        "pos_dim": 24,
        "compress_factor": 0.3,
        "causal": False,
        "n_fft": STFT_CFG["n_fft"],
        "hop_length": STFT_CFG["hop_length"],
        "win_length": STFT_CFG["win_length"],
    }

    model = ZipEnhancer(**MODEL_ARGS)
    stft_module = build_stft_istft(STFT_CFG)

    class _Wrapper(nn.Module):
        def __init__(self, m, stft_m):
            super().__init__()
            self._m, self._stft = m, stft_m

        def forward(self, wav, **kw):
            spec = self._stft(wav)
            out_spec = self._m(spec, **kw)
            return self._stft.inverse(out_spec, wav.shape[-1])

    wrapped = _Wrapper(model, stft_module)
    print("\n[Config] STFT:", STFT_CFG, "| sample_rate:", SAMPLE_RATE, "| model args:", MODEL_ARGS)
    profile_model(
        wrapped,
        sample_rate=SAMPLE_RATE,
        duration=args.dur,
        batch_size=args.batch,
        device=args.device,
        repeats=args.repeats,
        label="ZipEnhancer",
    )
