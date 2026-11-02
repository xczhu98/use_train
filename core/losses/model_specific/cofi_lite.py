"""CoFi-Lite 官方 HybridLoss（core/losses 注册实现）。

上游 ``loss_factory.py::HybridLoss`` 的逐项移植：

    L = lamda_ri * (L_real + L_imag) + lamda_mag * L_mag - SI-SNR

其中压缩复谱损失使用 ``c = compress_factor``、``eps = 1e-12``，SI-SNR 使用
上游的 ``-2*log10(||projection|| / ||residual||)`` 形式。上游训练片段是固定
长度，本实现额外支持仓库 collate 的可变长度：谱项按有效 STFT 帧掩码归约，
SI-SNR 项把 padding 区置零，固定长度输入与上游公式逐项一致。

YAML 示例见 ``configs/se/dns_challenge3/cofi_lite_causal_offline.yaml``。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from core.losses.utils import create_frame_mask, prepare_stft_waveform_at_lengths
from core.utils.registry import LOSS_REGISTRY


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_positive_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an exact positive int, got {value!r}")
    if value <= 0:
        raise ValueError(f"{field} must be positive, got {value!r}")
    return value


def _validate_finite_real(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def _validate_pair(
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    lengths: torch.Tensor | None,
) -> None:
    for name, value in (("enhanced", enhanced), ("clean", clean)):
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2:
            raise ValueError(f"{name} must be [B, T], got {tuple(value.shape)}")
        if value.shape[0] < 1 or value.shape[1] < 1:
            raise ValueError(
                f"{name} must have non-empty B and T dimensions, got {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if enhanced.shape != clean.shape:
        raise ValueError(
            f"enhanced/clean shape mismatch: {tuple(enhanced.shape)} vs "
            f"{tuple(clean.shape)}"
        )
    if enhanced.device != clean.device:
        raise ValueError(
            f"enhanced/clean device mismatch: {enhanced.device} vs {clean.device}"
        )
    if enhanced.dtype != clean.dtype:
        raise ValueError(
            f"enhanced/clean dtype mismatch: {enhanced.dtype} vs {clean.dtype}"
        )
    if lengths is None:
        return
    if not torch.is_tensor(lengths):
        raise TypeError("lengths must be a torch.Tensor")
    if lengths.ndim != 1 or lengths.shape[0] != enhanced.shape[0]:
        raise ValueError(
            f"lengths must be [B] with B={enhanced.shape[0]}, got "
            f"{tuple(lengths.shape)}"
        )
    if lengths.device != enhanced.device:
        raise ValueError(
            f"lengths device {lengths.device} does not match {enhanced.device}"
        )
    if lengths.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"lengths must be integer dtype, got {lengths.dtype}")
    if bool(((lengths < 1) | (lengths > enhanced.shape[1])).any().item()):
        raise ValueError(
            f"lengths must be in [1, {enhanced.shape[1]}], got {lengths.tolist()}"
        )


@LOSS_REGISTRY.register("CoFiLiteHybridLoss")
class CoFiLiteHybridLoss(nn.Module):
    """CoFi-Lite 官方 HybridLoss：压缩 RI + 压缩 magnitude + 负 SI-SNR。

    Args:
        n_fft / hop_length / win_length: 与上游 loss 一致的 512/256/512。
        window: 上游使用普通 Hann 窗（不是模型分析/合成的 sqrt-Hann）。
        center: 上游 ``torch.stft`` 默认 center=True。
        compress_factor: 幅度压缩指数 c，上游 0.3。
        eps: 上游 1e-12。
        lamda_ri / lamda_mag: 上游 30 / 70。
    """

    # 上游 loss 固定使用普通 Hann / center=True 的 512/256/512 几何；CoFi-Lite 是
    # waveform-native 模型，顶层 stft 不属于它的前端契约，因此不继承框架 STFT 默认值。
    USE_FRAMEWORK_STFT_DEFAULTS = False

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        window: str = "hann",
        center: bool = True,
        compress_factor: float = 0.3,
        eps: float = 1e-12,
        lamda_ri: float = 30.0,
        lamda_mag: float = 70.0,
    ):
        super().__init__()
        n_fft = _validate_positive_int(n_fft, field="n_fft")
        hop_length = _validate_positive_int(hop_length, field="hop_length")
        win_length = _validate_positive_int(win_length, field="win_length")
        if win_length > n_fft:
            raise ValueError("win_length must not exceed n_fft")
        if hop_length > win_length:
            raise ValueError("hop_length must not exceed win_length")
        if not isinstance(window, str) or window.lower() != "hann":
            raise ValueError(
                f"CoFiLiteHybridLoss fixes the upstream Hann loss window, got {window!r}"
            )
        if center is not True:
            raise ValueError(
                "CoFiLiteHybridLoss requires center=True to match upstream torch.stft"
            )
        compress_factor = _validate_finite_real(
            compress_factor, field="compress_factor"
        )
        if not 0.0 < compress_factor <= 1.0:
            raise ValueError("compress_factor must be in (0, 1]")
        eps = _validate_finite_real(eps, field="eps")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        lamda_ri = _validate_finite_real(lamda_ri, field="lamda_ri")
        lamda_mag = _validate_finite_real(lamda_mag, field="lamda_mag")
        if lamda_ri < 0.0:
            raise ValueError("lamda_ri must be non-negative")
        if lamda_mag < 0.0:
            raise ValueError("lamda_mag must be non-negative")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window_name = "hann"
        self.center = True
        self.c = compress_factor
        self.eps = eps
        self.lamda_ri = lamda_ri
        self.lamda_mag = lamda_mag
        self.register_buffer(
            "_window", torch.hann_window(win_length), persistent=False
        )

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window.to(
                device=waveform.device, dtype=waveform.dtype
            ),
            center=self.center,
            normalized=False,
            onesided=True,
            return_complex=True,
        )

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_pair(enhanced, clean, lengths)
        enhanced = enhanced.float()
        clean = clean.float()
        if lengths is not None:
            lengths = lengths.to(device=enhanced.device, dtype=torch.long)

        if lengths is None:
            enhanced_input, clean_input = enhanced, clean
            spectral_lengths = None
        else:
            enhanced_input, spectral_lengths = prepare_stft_waveform_at_lengths(
                enhanced, lengths, n_fft=self.n_fft, center=self.center
            )
            clean_input, _ = prepare_stft_waveform_at_lengths(
                clean, lengths, n_fft=self.n_fft, center=self.center
            )

        pred_stft = self._stft(enhanced_input)
        true_stft = self._stft(clean_input)

        # 与上游 loss_factory.py 完全一致的幅度/压缩顺序。
        pred_mag = torch.abs(pred_stft).clamp(min=self.eps)
        true_mag = torch.abs(true_stft).clamp(min=self.eps)
        pred_stft_c = pred_stft / pred_mag ** (1 - self.c)
        true_stft_c = true_stft / true_mag ** (1 - self.c)

        if spectral_lengths is None:
            real_loss = torch.mean((pred_stft_c.real - true_stft_c.real) ** 2)
            imag_loss = torch.mean((pred_stft_c.imag - true_stft_c.imag) ** 2)
            mag_loss = torch.mean((pred_mag ** self.c - true_mag ** self.c) ** 2)
        else:
            frame_mask = create_frame_mask(
                spectral_lengths,
                n_frames=pred_stft_c.shape[-1],
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                device=pred_stft_c.device,
                center=self.center,
            ).to(dtype=pred_stft_c.real.dtype).unsqueeze(1)
            denom = (frame_mask.sum() * pred_stft_c.shape[1]).clamp(min=1.0)
            real_loss = (
                ((pred_stft_c.real - true_stft_c.real) ** 2) * frame_mask
            ).sum() / denom
            imag_loss = (
                ((pred_stft_c.imag - true_stft_c.imag) ** 2) * frame_mask
            ).sum() / denom
            mag_loss = (
                ((pred_mag ** self.c - true_mag ** self.c) ** 2) * frame_mask
            ).sum() / denom

        # 上游 SI-SNR：不做零均值，保留 projection/residual 的原始范数比。
        if lengths is None:
            pred_sisnr = enhanced
            true_sisnr = clean
        else:
            time_mask = (
                torch.arange(enhanced.shape[1], device=enhanced.device)[None, :]
                < lengths[:, None]
            ).to(dtype=enhanced.dtype)
            pred_sisnr = enhanced * time_mask
            true_sisnr = clean * time_mask
        y_norm = (
            torch.sum(true_sisnr * pred_sisnr, dim=-1, keepdim=True)
            * true_sisnr
            / (torch.sum(true_sisnr.square(), dim=-1, keepdim=True) + 1e-8)
        )
        sisnr = -2 * torch.log10(
            torch.norm(y_norm, dim=-1, keepdim=True)
            / torch.norm(pred_sisnr - y_norm, dim=-1, keepdim=True).clamp(min=self.eps)
            + self.eps
        ).mean()

        return (
            self.lamda_ri * (real_loss + imag_loss)
            + self.lamda_mag * mag_loss
            + sisnr
        )
