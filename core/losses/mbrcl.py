"""论文 MBRCL：非学习式多频带相对对比语音增强损失。"""

from __future__ import annotations

import math
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.losses.utils import create_frame_mask, reflect_extend_at_lengths
from core.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("MultiBandRelativeContrastiveLoss")
class MultiBandRelativeContrastiveLoss(nn.Module):
    """Multi-Band Relative Contrastive Loss（MBRCL）。

    该类只返回论文的频带相对对比项与包络项；SI-SNR、MR-STFT 等基础项
    应由 ``CompositeLoss`` 在配置中显式组合。
    """

    REQUIRES_NOISY_REFERENCE = True
    SUPPORTS_SS_UNDESIRED_ENERGY = False

    REQUIRES_SAMPLE_RATE = True
    REQUIRES_MATCHING_STFT_CONTEXT = True

    def __init__(
        self,
        sample_rate: int | None = None,
        n_fft: int | None = None,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str | None = None,
        center: bool | None = None,
        num_bands: int = 16,
        margin: float = 0.06,
        eps: float = 1e-8,
        band_weight: float = 0.5,
        envelope_weight: float = 0.2,
    ) -> None:
        super().__init__()
        missing = [
            name
            for name, value in (("n_fft", n_fft), ("hop_length", hop_length))
            if value is None
        ]
        if missing:
            raise ValueError(
                "MultiBandRelativeContrastiveLoss requires analysis STFT geometry "
                "n_fft/hop_length; "
                f"missing={missing}. Construct it through build_loss with the model "
                "top-level stft config, or pass both fields explicitly."
            )
        if win_length is None:
            win_length = n_fft
        if window is None:
            window = "hann"
        if center is None:
            center = True
        for name, value in (
            ("n_fft", n_fft),
            ("hop_length", hop_length),
            ("win_length", win_length),
            ("num_bands", num_bands),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if sample_rate is not None:
            self._validate_sample_rate(sample_rate)
        assert n_fft is not None
        assert hop_length is not None
        if win_length > n_fft:
            raise ValueError(
                f"win_length must be <= n_fft, got {win_length} > {n_fft}"
            )
        if not isinstance(center, bool):
            raise TypeError(f"center must be bool, got {type(center).__name__}")
        window_name = window.lower() if isinstance(window, str) else ""
        if window_name not in {"hann", "hamming"}:
            raise ValueError(
                f"window must be 'hann' or 'hamming', got {window!r}"
            )
        for name, value in (
            ("margin", margin),
            ("eps", eps),
            ("band_weight", band_weight),
            ("envelope_weight", envelope_weight),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if margin < 0:
            raise ValueError(f"margin must be >= 0, got {margin}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if band_weight < 0 or envelope_weight < 0:
            raise ValueError("band_weight and envelope_weight must be >= 0")
        if band_weight == 0 and envelope_weight == 0:
            raise ValueError("band_weight and envelope_weight cannot both be zero")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window_name = window_name
        self.center = center
        self.num_bands = num_bands
        self.margin = float(margin)
        self.eps = float(eps)
        self.band_weight = float(band_weight)
        self.envelope_weight = float(envelope_weight)

        window_tensor = (
            torch.hamming_window(win_length)
            if window_name == "hamming"
            else torch.hann_window(win_length)
        )
        self.register_buffer("_window", window_tensor, persistent=False)
        initial_band_matrix = (
            self._build_band_matrix(sample_rate)
            if sample_rate is not None
            else torch.empty(0, dtype=torch.float32)
        )
        self.register_buffer("_band_matrix", initial_band_matrix, persistent=False)
        # Runtime SR 条目保持 canonical CPU FP32；按需在 einsum 前搬到计算设备。
        self._runtime_band_matrices: dict[int, torch.Tensor] = {}
        self._runtime_band_matrices_lock = threading.Lock()

    def __getstate__(self) -> dict:
        """序列化模块持久状态，不携带可重建 cache 与不可 pickle 的 lock。"""
        state = super().__getstate__()
        state.pop("_runtime_band_matrices", None)
        state.pop("_runtime_band_matrices_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self._runtime_band_matrices = {}
        self._runtime_band_matrices_lock = threading.Lock()

    def _is_valid_runtime_band_matrix(self, matrix: object) -> bool:
        return bool(
            torch.is_tensor(matrix)
            and matrix.shape == (self.num_bands, self.n_fft // 2 + 1)
            and matrix.device.type == "cpu"
            and matrix.dtype == torch.float32
        )

    def _ensure_runtime_cache_state(self) -> None:
        lock = getattr(self, "_runtime_band_matrices_lock", None)
        if not (
            callable(getattr(lock, "acquire", None))
            and callable(getattr(lock, "release", None))
        ):
            lock = threading.Lock()
            self._runtime_band_matrices_lock = lock
        with lock:
            if not isinstance(getattr(self, "_runtime_band_matrices", None), dict):
                self._runtime_band_matrices = {}

    @staticmethod
    def _validate_sample_rate(sample_rate: int) -> int:
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            raise ValueError(
                f"sample_rate must be a positive integer, got {sample_rate!r}"
            )
        return sample_rate

    def _build_band_matrix(self, sample_rate: int) -> torch.Tensor:
        sample_rate = self._validate_sample_rate(sample_rate)
        fft_freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / sample_rate)
        max_bin_hz = float(fft_freqs[-1].item())
        nyquist_hz = sample_rate / 2.0
        mel_max = 2595.0 * math.log10(1.0 + nyquist_hz / 700.0)
        mel_edges = torch.linspace(0.0, mel_max, self.num_bands + 1)
        freq_mels = 2595.0 * torch.log10(1.0 + fft_freqs / 700.0)
        band_index = torch.bucketize(freq_mels, mel_edges[1:-1], right=True)
        band_counts = torch.bincount(band_index, minlength=self.num_bands)
        if bool((band_counts == 0).any().item()):
            empty = (band_counts == 0).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(
                "num_bands creates empty Mel hard bands; "
                f"empty bands={empty}, sample_rate={sample_rate}, n_fft={self.n_fft}, "
                f"rfft_bins={fft_freqs.numel()}, max_bin_hz={max_bin_hz:.6g}, "
                f"num_bands={self.num_bands}"
            )
        return F.one_hot(band_index, num_classes=self.num_bands).T.float()

    def _band_matrix_for_sample_rate(self, sample_rate: int) -> torch.Tensor:
        sample_rate = self._validate_sample_rate(sample_rate)
        if sample_rate == self.sample_rate:
            return self._band_matrix
        self._ensure_runtime_cache_state()
        matrix = self._runtime_band_matrices.get(sample_rate)
        if self._is_valid_runtime_band_matrix(matrix):
            return matrix
        with self._runtime_band_matrices_lock:
            matrix = self._runtime_band_matrices.get(sample_rate)
            if not self._is_valid_runtime_band_matrix(matrix):
                matrix = self._build_band_matrix(sample_rate).to(
                    device="cpu", dtype=torch.float32
                )
                self._runtime_band_matrices[sample_rate] = matrix
        return matrix

    def _loss_from_band_energies(
        self,
        enhanced_energy: torch.Tensor,
        clean_energy: torch.Tensor,
        noisy_energy: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """按论文公式从 ``[B, bands, frames]`` 频带能量计算逐样本损失。"""
        if not (
            enhanced_energy.shape == clean_energy.shape == noisy_energy.shape
            and enhanced_energy.ndim == 3
            and enhanced_energy.shape[1] == self.num_bands
        ):
            raise ValueError(
                "band energies must share shape [B, num_bands, frames], got "
                f"{tuple(enhanced_energy.shape)}, {tuple(clean_energy.shape)}, "
                f"{tuple(noisy_energy.shape)}"
            )
        bsz, _, n_frames = enhanced_energy.shape
        if frame_mask is None:
            frame_mask = torch.ones(
                bsz, n_frames, device=enhanced_energy.device, dtype=torch.bool
            )
        elif (
            frame_mask.shape != (bsz, n_frames)
            or frame_mask.device != enhanced_energy.device
            or frame_mask.dtype != torch.bool
        ):
            raise ValueError(
                "frame_mask must be bool [B, frames] on the energy device, got "
                f"shape={tuple(frame_mask.shape)}, device={frame_mask.device}, "
                f"dtype={frame_mask.dtype}"
            )

        mask = frame_mask[:, None, :].to(enhanced_energy.dtype)
        valid_frames = mask.sum(dim=-1).clamp(min=1.0)
        log_clean = torch.log(clean_energy + self.eps)
        d_enh = (
            torch.abs(torch.log(enhanced_energy + self.eps) - log_clean) * mask
        ).sum(dim=-1) / valid_frames
        d_noisy = (
            torch.abs(torch.log(noisy_energy + self.eps) - log_clean) * mask
        ).sum(dim=-1) / valid_frames
        band_loss = F.softplus(d_enh - d_noisy + self.margin).sum(dim=1)

        if n_frames < 2:
            envelope_loss = enhanced_energy.sum(dim=(1, 2)) * 0.0
        else:
            enhanced_env = torch.sqrt(enhanced_energy + self.eps)
            clean_env = torch.sqrt(clean_energy + self.eps)
            pair_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
            pair_mask_f = pair_mask[:, None, :].to(enhanced_energy.dtype)
            pair_count = pair_mask_f.sum(dim=-1)
            envelope_sum = (
                torch.abs(
                    torch.diff(enhanced_env, dim=-1)
                    - torch.diff(clean_env, dim=-1)
                )
                * pair_mask_f
            ).sum(dim=(1, 2))
            envelope_loss = torch.where(
                pair_count[:, 0] > 0,
                envelope_sum / pair_count[:, 0].clamp(min=1.0),
                enhanced_energy.sum(dim=(1, 2)) * 0.0,
            )

        return self.band_weight * band_loss + self.envelope_weight * envelope_loss

    @staticmethod
    def _validate_waveform(name: str, value: torch.Tensor, reference: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.shape != reference.shape:
            raise ValueError(
                f"{name} must match enhanced shape {tuple(reference.shape)}, "
                f"got {tuple(value.shape)}"
            )
        if value.device != reference.device:
            raise ValueError(
                f"{name} must be on device {reference.device}, got {value.device}"
            )
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {value.dtype}")

    @torch.amp.autocast("cuda", enabled=False)
    @torch.amp.autocast("cpu", enabled=False)
    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        noisy: torch.Tensor | None = None,
        sample_rate: int | None = None,
        is_pure_noise: torch.Tensor | None = None,
        is_pure_echo: torch.Tensor | None = None,
        is_pure_echo_noise: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # pure metadata 仅用于 trainer/Composite 兼容；MBRCL 对全 batch 生效，
        # 包括 clean=0 的 target-absent 场景。
        del is_pure_noise, is_pure_echo, is_pure_echo_noise, kwargs
        effective_sample_rate = self.sample_rate if sample_rate is None else sample_rate
        if effective_sample_rate is None:
            raise ValueError(
                "MultiBandRelativeContrastiveLoss requires sample_rate from either "
                "construction context or the forward call"
            )
        effective_sample_rate = self._validate_sample_rate(effective_sample_rate)
        if not torch.is_tensor(enhanced) or enhanced.ndim != 2:
            shape = tuple(enhanced.shape) if torch.is_tensor(enhanced) else None
            raise ValueError(f"enhanced must be [B, T], got {shape}")
        if not enhanced.is_floating_point():
            raise TypeError(f"enhanced must be floating point, got {enhanced.dtype}")
        self._validate_waveform("clean", clean, enhanced)
        if noisy is None:
            raise ValueError("MultiBandRelativeContrastiveLoss requires noisy reference")
        self._validate_waveform("noisy", noisy, enhanced)

        bsz, n_samples = enhanced.shape
        if lengths is None:
            lengths = torch.full(
                (bsz,), n_samples, device=enhanced.device, dtype=torch.long
            )
        else:
            if not torch.is_tensor(lengths):
                raise TypeError("lengths must be a torch.Tensor")
            if lengths.dtype not in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }:
                raise TypeError(f"lengths must use integer dtype, got {lengths.dtype}")
            if lengths.shape != (bsz,):
                raise ValueError(f"lengths must have shape ({bsz},), got {tuple(lengths.shape)}")
            lengths = lengths.to(device=enhanced.device, dtype=torch.long)
            if bool(((lengths < 1) | (lengths > n_samples)).any().item()):
                raise ValueError(f"lengths must be in [1, {n_samples}]")

        signals = torch.cat((enhanced, clean, noisy), dim=0).float()
        if not self.center and (
            n_samples < self.n_fft
            or (lengths is not None and bool((lengths < self.n_fft).any().item()))
        ):
            raise ValueError(
                "center=False requires every waveform length to be at least n_fft="
                f"{self.n_fft}"
            )
        if lengths is not None:
            sample_mask = (
                torch.arange(n_samples, device=signals.device)[None, :]
                < lengths[:, None]
            ).to(signals.dtype)
            signals = signals * sample_mask.repeat(3, 1)
        if self.center:
            pad = self.n_fft // 2
            if lengths is not None:
                # 固定追加完整右侧 reflect 余量，避免读取 GPU lengths 触发同步；
                # 同时修复总长落在 length 与 length + pad 之间时的边界漂移。
                signals = F.pad(signals, (0, pad))
                signals = reflect_extend_at_lengths(
                    signals,
                    lengths.repeat(3),
                    pad,
                )
            elif signals.shape[-1] <= pad:
                signals = F.pad(signals, (0, pad + 1 - signals.shape[-1]))

        spectra = torch.stft(
            signals,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window.to(device=signals.device, dtype=torch.float32),
            center=self.center,
            return_complex=True,
            onesided=True,
        )
        power = spectra.abs().square()
        band_matrix = self._band_matrix_for_sample_rate(effective_sample_rate).to(
            device=power.device, dtype=power.dtype
        )
        band_energy = torch.einsum("bf,nft->nbt", band_matrix, power)
        enhanced_energy, clean_energy, noisy_energy = band_energy.chunk(3, dim=0)

        if lengths is not None:
            frame_mask = create_frame_mask(
                lengths,
                enhanced_energy.shape[-1],
                self.n_fft,
                self.win_length,
                self.hop_length,
                enhanced_energy.device,
                center=self.center,
            )

        per_sample = self._loss_from_band_energies(
            enhanced_energy,
            clean_energy,
            noisy_energy,
            frame_mask,
        )
        return per_sample.mean()
