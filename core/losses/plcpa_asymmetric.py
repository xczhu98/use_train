"""PLCPA-ASYM loss: power-law compressed phase-aware loss with asymmetric term.

Reference implementation:
  - TeaPoly/PLCPA-ASYM-Loss: https://github.com/TeaPoly/PLCPA-ASYM-Loss
  - Paper: arXiv:2211.03036, "PLCPA-ASYM loss ... for voice conversion in the
    presence of background sound".

Original PLCPA formulation (Eskimez et al., 2021):
    L_a(t,f) = ||clean|^p - |enh|^p|^2
    L_p(t,f) = ||clean|^p e^{jφ(clean)} - |enh|^p e^{jφ(enh)}|^2
    L_PLCPA  = mean( α * L_a + (1-α) * L_p )

Asymmetric over-suppression term (TEA-PSE / arXiv:2211.03036):
    delta    = |clean|^p - |enh|^p
    L_asym   = mean( ReLU(delta)^2 )

Final loss:
    L_PLCPA_ASYM = scale_mag * L_a + scale_phase * L_p + scale_asym * L_asym

Note: the reference code sums (not averages) L_asym, which makes scale_asym
sensitive to signal length. This implementation averages all terms for a
length-consistent scale_asym.

修复验证 (2026-07-15): 此前复数 F.mse_loss(phase 项)与 complex mask clamp
导致本 loss 前向必崩, 从未跑通。修复后 phase 项取复数差模平方、mask 保持
实数 dtype, lengths 给定时三项均按有效帧 masked mean; 有/无 lengths 前向
均通过 smoke 测试(tests/test_loss_audit_fixes.py), 输出有限。center=True 时还会
复用共享的 length-aware centered-STFT 输入准备和有效帧 mask，使短输入可稳定
前反向，并保持 trim、轻量右 padding 与完整 batch padding 的 loss/梯度等价。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.losses.utils import (
    compute_spec_frame_lengths,
    create_frame_mask,
    prepare_centered_stft_inputs,
)
from core.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("PLCPAAsymmetricLoss")
class PLCPAAsymmetricLoss(nn.Module):
    """PLCPA magnitude + phase-aware loss with an asymmetric over-suppression term.

    Args:
        n_fft, hop_length, win_length, center: STFT parameters.
        compress_factor: power-law compression exponent p (default 0.3).
        scale_mag: weight for the compressed magnitude MSE term.
        scale_phase: weight for the compressed phase-aware MSE term.
        scale_asym: weight for the asymmetric over-suppression term.
        window: "hann" or "hamming". The reference implementation uses hamming.
        eps: numerical stability epsilon.
    """

    def __init__(
        self,
        n_fft: int = 320,
        hop_length: int = 160,
        win_length: int = 320,
        compress_factor: float = 0.3,
        scale_mag: float = 1.0,
        scale_phase: float = 1.0,
        scale_asym: float = 1.0,
        window: str = "hann",
        eps: float = 1e-8,
        center: bool = False,
    ):
        super().__init__()
        for name, value in (
            ("n_fft", n_fft),
            ("hop_length", hop_length),
            ("win_length", win_length),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact positive integer, got {value!r}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if win_length > n_fft:
            raise ValueError("win_length must satisfy win_length <= n_fft")
        if type(center) is not bool:
            raise TypeError(f"center must be bool, got {center!r}")
        for name, value, lower, inclusive in (
            ("compress_factor", compress_factor, 0.0, False),
            ("eps", eps, 0.0, False),
            ("scale_mag", scale_mag, 0.0, True),
            ("scale_phase", scale_phase, 0.0, True),
            ("scale_asym", scale_asym, 0.0, True),
        ):
            if type(value) not in (int, float):
                raise TypeError(f"{name} must be a finite number, got {value!r}")
            if not math.isfinite(value) or (value < lower if inclusive else value <= lower):
                relation = ">=" if inclusive else ">"
                raise ValueError(f"{name} must be finite and {relation} {lower}, got {value!r}")
        if compress_factor > 1.0:
            raise ValueError(
                f"compress_factor must be finite and in (0, 1], got {compress_factor!r}"
            )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.p = float(compress_factor)
        self.scale_mag = float(scale_mag)
        self.scale_phase = float(scale_phase)
        self.scale_asym = float(scale_asym)
        self.eps = float(eps)
        self.center = center

        wname = (window or "hann").lower()
        if wname == "hann":
            wbuf = torch.hann_window(min(win_length, n_fft))
        elif wname == "hamming":
            wbuf = torch.hamming_window(min(win_length, n_fft))
        else:
            raise ValueError(f"PLCPAAsymmetricLoss window '{window}' not supported; use hann or hamming")
        self.register_buffer("_window", wbuf, persistent=False)

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            x.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window,
            center=self.center,
            return_complex=True,
            onesided=True,
        )

    def _mask_frames(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Zero out STFT frames beyond the valid sample length. Returns (x, mask)."""
        if lengths is None:
            return x, None
        n_frames = x.shape[-1]
        t = torch.arange(n_frames, device=lengths.device, dtype=lengths.dtype)
        frame_len = compute_spec_frame_lengths(
            lengths.long(),
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=self.center,
        ).to(device=x.device)
        # mask 全程保持实数 dtype: x 为复谱时复数×实数广播合法。
        mask = (t[None, :] < frame_len[:, None]).to(
            device=x.device, dtype=torch.float32
        )
        # Expand mask to (B, 1, T) for broadcasting over frequency
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(1)
        return x * mask, mask

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        enhanced = enhanced.float()
        clean = clean.float()
        if enhanced.shape != clean.shape or enhanced.ndim != 2:
            raise ValueError(
                "enhanced and clean must be same-shape [B, T] tensors, got "
                f"{tuple(enhanced.shape)} and {tuple(clean.shape)}"
            )
        if not self.center and enhanced.shape[1] < self.n_fft:
            raise ValueError(
                "PLCPAAsymmetricLoss(center=False) requires waveform length "
                f"to be >= n_fft={self.n_fft}"
            )
        if lengths is not None:
            if lengths.shape != (enhanced.shape[0],):
                raise ValueError(
                    f"lengths must be [B]={enhanced.shape[0]}, got {tuple(lengths.shape)}"
                )
            lengths = lengths.to(device=enhanced.device, dtype=torch.long)
            if bool((lengths < 0).any()) or bool((lengths > enhanced.shape[1]).any()):
                raise ValueError("lengths must satisfy 0 <= lengths <= waveform length")
            # 先遮蔽时域 padding，保证后续 STFT 从不读取无效 tail。
            time = torch.arange(enhanced.shape[1], device=enhanced.device)
            sample_mask = (time.unsqueeze(0) < lengths.unsqueeze(1)).to(enhanced.dtype)
            enhanced = enhanced * sample_mask
            clean = clean * sample_mask
            if not self.center and bool((lengths < self.n_fft).any()):
                raise ValueError(
                    "PLCPAAsymmetricLoss(center=False) requires every valid length "
                    f"to be >= n_fft={self.n_fft}"
                )

        if self.center:
            (enhanced, clean), lengths, _ = prepare_centered_stft_inputs(
                enhanced,
                clean,
                lengths=lengths,
                n_fft=self.n_fft,
            )

        enh_spec = self._stft(enhanced)
        cln_spec = self._stft(clean)

        if self.center:
            mask = create_frame_mask(
                lengths,
                enh_spec.shape[-1],
                self.n_fft,
                self.win_length,
                self.hop_length,
                enh_spec.device,
                center=True,
            ).unsqueeze(1).to(enh_spec.real.dtype)
            enh_spec = enh_spec * mask
            cln_spec = cln_spec * mask
        else:
            enh_spec, mask = self._mask_frames(enh_spec, lengths)
            cln_spec, _ = self._mask_frames(cln_spec, lengths)

        enh_mag = enh_spec.abs().clamp(min=self.eps)
        cln_mag = cln_spec.abs().clamp(min=self.eps)

        # Power-law compressed magnitudes
        enh_mag_c = enh_mag ** self.p
        cln_mag_c = cln_mag ** self.p

        # Phase-aware compressed spectra: |·|^p e^{jφ}
        enh_c = enh_spec / (enh_mag ** (1.0 - self.p))
        cln_c = cln_spec / (cln_mag ** (1.0 - self.p))

        # Magnitude term: ||clean|^p - |enh|^p|^2
        # Phase-aware term: ||clean|^p e^{jφ(clean)} - |enh|^p e^{jφ(enh)}|^2
        # 复数张量 F.mse_loss 未实现(调用必崩), phase 项取复数差的模平方
        mag_diff = (cln_mag_c - enh_mag_c).square()
        phase_diff = (cln_c - enh_c).abs().square()
        if mask is not None:
            # 有效帧 masked mean, 分母含 F 维广播因子
            denom = (mask.sum() * mag_diff.shape[1]).clamp(min=1.0)
            mag_loss = (mag_diff * mask).sum() / denom
            phase_loss = (phase_diff * mask).sum() / denom
        else:
            mag_loss = F.mse_loss(cln_mag_c, enh_mag_c)
            phase_loss = phase_diff.mean()

        loss = self.scale_mag * mag_loss + self.scale_phase * phase_loss

        if self.scale_asym > 0.0:
            # Asymmetric over-suppression term:
            # penalize only when |clean|^p > |enh|^p (i.e. enhanced is too quiet)
            asym_sq = F.relu(cln_mag_c - enh_mag_c) ** 2
            if mask is not None:
                asym_loss = (asym_sq * mask).sum() / denom
            else:
                asym_loss = asym_sq.mean()
            loss = loss + self.scale_asym * asym_loss

        return loss
