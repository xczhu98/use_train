"""Frequency-Weighted Neg-SISDR Loss.

Adapted from: Monir et al., "Frequency-Weighted Training Losses for
Phoneme-Level DNN-based Speech Enhancement", MMSP 2025, arXiv:2506.18714.

Core idea: compute SI-SDR in the frequency domain instead of time domain,
with per-band weights that emphasize low-SNR regions where speech energy
is still present (typically fricatives / plosives that are critical for
phoneme intelligibility).

Weights are derived from the target speech and noise spectra, NOT from
learnable parameters.  The default ``log_snr_speech_norm`` strategy combines
inverse SNR with speech magnitude:

    w_f,t = softplus(-log10(SNR_f)) × |S_f,t|^γ

where SNR_f is the per-band time-averaged SNR, and γ=0.2 compresses the
dynamic range of the speech magnitude weighting.  softplus 保证权重严格为正:
低 SNR 频带 ≈ -log10(SNR_f)(大权重), 高 SNR 频带 → 0+(小权重)。论文官方实现
直接用 -log10(SNR_f)(可为负), 依赖加权和负负相消, 混合符号频带会产生 NaN;
本仓此前用逐项 clamp 防 NaN 又导致正 SNR 输入零梯度(JAE-H01), 故改为 softplus。

.. note::

   Like SI-SDR, this loss produces **zero gradient** when ``clean == 0``
   (pure-noise / pure-echo branches).  When used inside ``CompositeLoss``,
   the class declares ``REQUIRES_SPEECH_REFERENCE = True`` so pure-branch
   samples are automatically skipped.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.losses.stft_cache import cached_hann_window
from core.losses.utils import reflect_extend_at_lengths, compute_spec_frame_lengths
from core.utils.registry import LOSS_REGISTRY

# ---------------------------------------------------------------------------
# Critical-band / Mel-band parameter tables (from the paper / asteroid impl)
# ---------------------------------------------------------------------------

# 18 critical bands (centre freq, bandwidth in Hz) for sr=16000
_CRITICAL_18_FC = torch.tensor([
    160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250,
    1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
], dtype=torch.float64)
_CRITICAL_18_BW = torch.tensor([
    37.05, 46.31, 57.90, 72.95, 92.62, 115.78, 145.88, 185.25, 231.56,
    289.46, 370.50, 463.12, 578.91, 729.43, 926.26, 1157.82, 1458.85, 1852.51,
], dtype=torch.float64)

# Perceptual weights for 18 critical bands
_PERCEPTUAL_18 = torch.tensor([
    0.0083, 0.0095, 0.0150, 0.0289, 0.0440, 0.0578, 0.0653, 0.0711, 0.0818,
    0.0844, 0.0882, 0.0898, 0.0868, 0.0844, 0.0771, 0.0527, 0.0364, 0.0185,
], dtype=torch.float64)


# ---------------------------------------------------------------------------
# 滤波器组 / 变换模块缓存（热路径：避免每次 forward 重建，键空间有界）
# ---------------------------------------------------------------------------

_MAX_FB_CACHE = 16
_crit_filter_cache: OrderedDict[tuple[int, int, str], torch.Tensor] = OrderedDict()
_mel_scale_cache: OrderedDict[tuple[int, int, int, str], object] = OrderedDict()
# 模块级 LRU 可能被多线程（验证/DataLoader worker）同时命中，读写必须持锁。
_cache_lock = threading.Lock()


def _cached_crit_filter(
    sr: int,
    n_fft: int,
    device: torch.device,
    cent_freq: torch.Tensor,
    bandwidth: torch.Tensor,
) -> torch.Tensor:
    """按 (sr, n_fft, device) 缓存 18 临界带高斯滤波器组（float64）。

    cent_freq / bandwidth 固定为模块级常量表，不参与 key。
    """
    key = (int(sr), int(n_fft), str(device))
    with _cache_lock:
        hit = _crit_filter_cache.get(key)
        if hit is not None:
            _crit_filter_cache.move_to_end(key)
            return hit

    max_freq = sr / 2
    num_crit = len(cent_freq)
    n_fft_by_2 = n_fft // 2
    bw_min = bandwidth[0]

    min_factor = torch.exp(torch.tensor(-30.0 / (2.0 * 2.303), dtype=torch.float64, device=device))
    crit_filter = torch.zeros((num_crit, n_fft_by_2), dtype=torch.float64, device=device)
    j = torch.arange(0, n_fft_by_2, dtype=torch.float64, device=device)

    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * n_fft_by_2
        bw = (bandwidth[i] / max_freq) * n_fft_by_2
        norm_factor = torch.log(bw_min) - torch.log(bandwidth[i])
        crit_filter[i, :] = torch.exp(-11.0 * ((j - torch.floor(f0)) / bw) ** 2 + norm_factor)
        crit_filter[i, :] *= (crit_filter[i, :] > min_factor)

    with _cache_lock:
        _crit_filter_cache[key] = crit_filter
        while len(_crit_filter_cache) > _MAX_FB_CACHE:
            _crit_filter_cache.popitem(last=False)
    return crit_filter


def _cached_mel_scale(sr: int, n_fft: int, n_mels: int, device: torch.device):
    """按 (sr, n_fft, n_mels, device) 缓存 torchaudio MelScale（内部 fb 只建一次）。"""
    from torchaudio.transforms import MelScale

    key = (int(sr), int(n_fft), int(n_mels), str(device))
    with _cache_lock:
        hit = _mel_scale_cache.get(key)
        if hit is not None:
            _mel_scale_cache.move_to_end(key)
            return hit
    mel = MelScale(
        n_mels=n_mels, sample_rate=sr, n_stft=n_fft // 2 + 1,
        f_min=0.0, f_max=sr / 2, norm=None, mel_scale="htk",
    ).to(device)
    with _cache_lock:
        _mel_scale_cache[key] = mel
        while len(_mel_scale_cache) > _MAX_FB_CACHE:
            _mel_scale_cache.popitem(last=False)
    return mel


def _signal_to_critic_band_spec(
    x: torch.Tensor,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    cent_freq: torch.Tensor,
    bandwidth: torch.Tensor,
) -> torch.Tensor:
    """Convert waveform [B, T] → critical-band spectrogram [B, n_bands, T_frames].

    Each band is a Gaussian-shaped filter :: exp(-11 * ((f-fc)/bw)² + norm),
    with ``norm = log(bw_min) - log(bw_i)`` so the narrowest band has unit
    gain and wider bands are attenuated.
    """
    device = x.device
    crit_filter = _cached_crit_filter(sr, n_fft, device, cent_freq, bandwidth)
    window = cached_hann_window(win_length, device, torch.float32)

    # 批量化 STFT：与原逐样本循环相同的 padding 行为（整批同长，pad 量一致）
    if x.shape[-1] < win_length:
        x = F.pad(x, (0, win_length - x.shape[-1]))
    stft = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, return_complex=True,
    ).abs()  # [B, n_fft//2+1, T_frames]
    stft = stft[:, :-1, :]  # drop Nyquist bin → [B, n_fft_by_2, T_frames]
    return torch.matmul(crit_filter, stft.to(torch.float64)).to(torch.float32)


def _signal_to_mel_spec(
    x: torch.Tensor,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> torch.Tensor:
    """Convert waveform [B, T] → mel spectrogram [B, n_mels, T_frames]."""
    window = cached_hann_window(n_fft, x.device, torch.float32)
    stft = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, center=True, pad_mode="reflect", return_complex=True,
    )  # [B, F, T]
    mag = torch.abs(stft)

    mel = _cached_mel_scale(sr, n_fft, n_mels, x.device)
    # MelScale expects (..., n_stft, time); mag is [B, F, T] → permute
    return mel(mag).to(torch.float32)  # [B, n_mels, T]


def _compute_fw_sdr(
    est_speech: torch.Tensor,
    target_speech: torch.Tensor,
    target_noise: torch.Tensor,
    *,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_bins: int,
    scale: str,
    weights_type: str,
    gamma: float,
    eps: float,
    zero_mean: bool = True,
    lengths: torch.Tensor | None = None,
    sdr_min_db: float = -10.0,
    sdr_max_db: float = 35.0,
) -> torch.Tensor:
    """Core FW-SDR computation, adapted from the paper's ``NegSISDRLoss.forward``.

    Args:
        est_speech:   [B, T]  enhanced/estimated speech
        target_speech:[B, T]  clean reference
        target_noise: [B, T]  undesired reference paired with target_speech,
                              typically degraded_input - target_speech
        lengths:      [B]     有效样本数; 提供时零均值/频带功率/加权聚合只在有效区
                              计算(JAE-H02 padding 不变), None 时保持原行为。
                              注意: 仅在 zero_mean=True 时生效(``len_active =
                              lengths is not None and zero_mean``); zero_mean=False
                              时 lengths 不生效——时域 mask、reflect 延展、帧 mask
                              均跳过, padding 会泄漏进结果。
        sdr_min_db:   JAE-H04 切线拐点(dB)。加权 SDR 低于该值的样本走切线延伸
                      (C¹ 连续、梯度有界), 高于则走对数域(与官方一致)。
        sdr_max_db:   上界硬截断(dB), easy sample 饱和; 与官方 35 一致。
    Returns:
        Scalar loss = -mean(FW-SDR).
    """
    bsz = est_speech.shape[0]
    len_active = lengths is not None and zero_mean

    if zero_mean:
        if lengths is not None:
            # JAE-H02: 零均值只在有效样点上计算, 减均值后乘 mask;
            # padding 置零后投影/能量求和自然只在有效区。
            t = torch.arange(est_speech.shape[-1], device=est_speech.device)
            t_mask = (t.unsqueeze(0) < lengths.unsqueeze(1)).to(est_speech.dtype)
            n_valid = lengths.to(est_speech.dtype).unsqueeze(-1).clamp(min=1.0)
            target_speech = (target_speech - (target_speech * t_mask).sum(dim=-1, keepdim=True) / n_valid) * t_mask
            est_speech = (est_speech - (est_speech * t_mask).sum(dim=-1, keepdim=True) / n_valid) * t_mask
            target_noise = (target_noise - (target_noise * t_mask).sum(dim=-1, keepdim=True) / n_valid) * t_mask
        else:
            target_speech = target_speech - target_speech.mean(dim=-1, keepdim=True)
            est_speech = est_speech - est_speech.mean(dim=-1, keepdim=True)
            target_noise = target_noise - target_noise.mean(dim=-1, keepdim=True)

    # 1.  projection / distortion in the time domain
    _dot_s = torch.sum(est_speech * target_speech, dim=-1, keepdim=True)
    _eng_s = torch.sum(target_speech ** 2, dim=-1, keepdim=True) + eps
    proj = (_dot_s / _eng_s) * target_speech
    e_dist = est_speech - proj

    if len_active:
        # JAE-H02: 频域变换前在有效边界补 reflect 内容, 使 STFT 边界帧与 trim 后
        # 计算逐点一致(mel/critical 路径均 center=True、win_length=n_fft)。共享 helper
        # 只会写入已有 storage, 因此必须先预留完整右余量；否则轻量 padding(< pad)
        # 会截断反射尾，critical 的极短输入还会因后续补零到 win_length 放大差异。
        pad = n_fft // 2
        proj = reflect_extend_at_lengths(F.pad(proj, (0, pad)), lengths, pad)
        e_dist = reflect_extend_at_lengths(F.pad(e_dist, (0, pad)), lengths, pad)
        target_speech = reflect_extend_at_lengths(
            F.pad(target_speech, (0, pad)), lengths, pad
        )
        target_noise = reflect_extend_at_lengths(
            F.pad(target_noise, (0, pad)), lengths, pad
        )

    # 2.  spectral transform
    if scale == "critical":
        cent_freq = _CRITICAL_18_FC
        bandwidth = _CRITICAL_18_BW
        proj_spec = _signal_to_critic_band_spec(
            proj, sr, n_fft, hop_length, n_fft, cent_freq, bandwidth,
        )
        e_dist_spec = _signal_to_critic_band_spec(
            e_dist, sr, n_fft, hop_length, n_fft, cent_freq, bandwidth,
        )
    elif scale == "mel":
        proj_spec = _signal_to_mel_spec(proj, sr, n_fft, hop_length, n_bins)
        e_dist_spec = _signal_to_mel_spec(e_dist, sr, n_fft, hop_length, n_bins)
    else:
        raise ValueError(f"FW-SDR: unsupported scale '{scale}'; use 'mel' or 'critical'.")

    if len_active:
        # JAE-H02: 有效帧 mask, padding 帧不参与频带功率与加权聚合。
        # mel/critical STFT 均 center=True、win_length=n_fft, 帧数公式一致。
        frame_len = compute_spec_frame_lengths(
            lengths, n_fft=n_fft, win_length=n_fft, hop_length=hop_length, center=True
        ).to(device=proj_spec.device)
        ft = torch.arange(proj_spec.shape[-1], device=proj_spec.device)
        frame_mask = (ft.unsqueeze(0) < frame_len.unsqueeze(1).clamp(min=1)).to(proj_spec.dtype).unsqueeze(1)

    # 3.  frequency weights
    if weights_type == "perceptual":
        weights = _PERCEPTUAL_18.to(proj_spec.device)
        weights = weights.view(1, len(weights), 1).repeat(bsz, 1, proj_spec.shape[2])

    elif weights_type == "speech_norm":
        if scale == "critical":
            tgt_speech_spec = _signal_to_critic_band_spec(
                target_speech, sr, n_fft, hop_length, n_fft, cent_freq, bandwidth,
            )
        else:
            tgt_speech_spec = _signal_to_mel_spec(target_speech, sr, n_fft, hop_length, n_bins)
        weights = torch.abs(tgt_speech_spec) ** gamma

    else:
        # data-driven weights: need noise spectrum + speech spectrum
        if scale == "critical":
            tgt_speech_spec = _signal_to_critic_band_spec(
                target_speech, sr, n_fft, hop_length, n_fft, cent_freq, bandwidth,
            )
            tgt_noise_spec = _signal_to_critic_band_spec(
                target_noise, sr, n_fft, hop_length, n_fft, cent_freq, bandwidth,
            )
        else:
            tgt_speech_spec = _signal_to_mel_spec(target_speech, sr, n_fft, hop_length, n_bins)
            tgt_noise_spec = _signal_to_mel_spec(target_noise, sr, n_fft, hop_length, n_bins)

        if len_active:
            # JAE-H02: 频带功率的帧均值只在有效帧计算, padding 帧不占分母。
            f_denom = frame_mask.sum(dim=-1).clamp(min=1.0)
            power_speech = ((torch.abs(tgt_speech_spec) ** 2) * frame_mask).sum(dim=-1) / f_denom
            power_noise = ((torch.abs(tgt_noise_spec) ** 2) * frame_mask).sum(dim=-1) / f_denom
        else:
            power_speech = torch.mean(torch.abs(tgt_speech_spec) ** 2, dim=-1)  # [B, n_bands]
            power_noise = torch.mean(torch.abs(tgt_noise_spec) ** 2, dim=-1)
        snr = power_speech / (power_noise + eps)

        if weights_type == "log_snr_speech_norm":
            log_snr = torch.log10(snr + eps).unsqueeze(-1).expand_as(tgt_speech_spec)
            # softplus(-log_snr) 严格为正, 保留"低 SNR 频带高权重"语义:
            # SNR<<1 时 ≈ -log_snr(大), SNR>>1 时 → 0+(小但非零)。
            # 官方实现直接用 -log_snr(可为负), 依赖加权和负负相消, 混合符号频带
            # 会产生 NaN; 本仓此前的逐项 clamp 修复又导致正 SNR 输入零梯度(JAE-H01)。
            weights = F.softplus(-log_snr) * (torch.abs(tgt_speech_spec) ** gamma)

        elif weights_type == "softmax_inverse_snr_speech_norm":
            inv_snr_soft = torch.softmax(1.0 / (snr + eps), dim=1).unsqueeze(-1).expand_as(tgt_speech_spec)
            weights = inv_snr_soft * (torch.abs(tgt_speech_spec) ** gamma)

        elif weights_type == "inverse_spectral_diff":
            spectral_diff = torch.abs(torch.abs(tgt_speech_spec) - torch.abs(tgt_noise_spec))
            weights = 1.0 / (spectral_diff + eps)

        else:
            raise ValueError(f"FW-SDR: unsupported weights_type '{weights_type}'.")

    weights = weights.to(proj_spec.device, dtype=proj_spec.dtype)
    if len_active:
        # JAE-H02: padding 帧权重置零, 加权聚合与 trim 后计算逐点一致。
        weights = weights * frame_mask.to(weights.dtype)

    # 4.  weighted SDR — 聚合方式与官方 sdr_xp.py 一致(不逐项 clamp)。
    # 所有 weights_type 权重均非负 → weighted_proj/dist >= 0, sdr 不会为负,
    # 不需要防 NaN 的逐项 clamp(该 clamp 会杀死正 SNR 输入的梯度, 见 JAE-H01)。
    weighted_proj = torch.sum(weights * torch.abs(proj_spec) ** 2, dim=(-1, -2))
    weighted_dist = torch.sum(weights * torch.abs(e_dist_spec) ** 2, dim=(-1, -2))
    ratio = weighted_proj / (weighted_dist + eps)
    sdr = 10.0 * torch.log10(ratio + eps)
    # JAE-H04: 官方 sdr_xp.py 硬 clamp(min=-10) 区间外梯度为 0, sdr < 拐点的
    # hard sample 完全拿不到训练信号; 直接去掉下界则 dB 空间梯度 ∝ 1/(ratio+eps),
    # 灾难样本(ratio→0)梯度爆炸。采用拐点(sdr_min_db)处的切线延伸(Huber 对偶
    # 结构: 工作区内对数 = 恒定相对力, 尾部线性 = 恒定绝对力): 斜率由 C¹ 连续性
    # 唯一确定(无调参), 区间外梯度恒定有界, loss 亦有界(最坏 ratio=0 →
    # sdr_min_db - 10/(ln10)·... 即 sdr_min_db - slope·knee_ratio)。
    # 上界 sdr_max_db 硬截断: easy sample 饱和是良性设计。拐点默认 -10 与官方
    # 一致; 若任务数据包含更低 SDR 的困难样本, 配置侧可将其调低至 -20。
    knee_ratio = 10.0 ** (sdr_min_db / 10.0)
    slope = 10.0 / (math.log(10.0) * knee_ratio)
    tangent = sdr_min_db + slope * (ratio - knee_ratio)
    sdr = torch.where(ratio >= knee_ratio, sdr, tangent)
    sdr = torch.clamp(sdr, max=sdr_max_db)
    return -sdr.mean()


@LOSS_REGISTRY.register("FWNegSISDRLoss")
class FWNegSISDRLoss(nn.Module):
    """Frequency-Weighted Negative SI-SDR Loss.

    Like standard SI-SDR, but computes the SDR in the frequency domain
    (mel or critical-band) after applying per-band perceptual / data-driven
    weights. The weights emphasise low-SNR frequency bands where target speech
    energy is still present — precisely the bands that carry phoneme
    information and are most susceptible to being dropped by enhancement
    models.

    Usage inside CompositeLoss (recommended)::

        loss:
          type: CompositeLoss
          components:
            - type: GTCRNCompressedSpectralLoss
              weight: 1.0
              args:
                compress_factor: 0.3
                lamda_ri: 30.0
                lamda_mag: 70.0
            - type: GTCRNLogRatioLoss
              weight: 1.0
            - type: FWNegSISDRLoss
              weight: 0.5
              args:
                scale: mel
                n_bins: 18
                weights_type: log_snr_speech_norm
                n_fft: 320
                hop_length: 160
                # JAE-H04: 可选, 默认 -10/35 与官方一致; 对极低 SDR 数据
                # 可按任务配置将 sdr_min_db 调低至 -20
                sdr_min_db: -20.0
                sdr_max_db: 35.0

    ``sdr_min_db`` 是切线拐点(低于它的样本走 C¹ 切线延伸, 梯度有界非零),
    ``sdr_max_db`` 是上界硬截断(easy sample 饱和)。

    **Important**: requires ``undesired_ref`` as a Tensor with the same shape
    and device as ``clean`` and a floating-point dtype. It is the real
    undesired reference paired with the target, typically
    ``degraded_input - target``; the calling adapter must provide it. Invalid
    or missing references fail closed before any silent-reference early return;
    the loss never estimates or falls back to an output-derived reference.

    When ``clean ≈ 0`` (pure-noise / pure-echo / pure-noise-echo branches),
    returns a graph-connected zero loss (gradient is undefined: proj ≡ 0 makes
    the SDR mathematically constant w.r.t. the estimate).  ``CompositeLoss``
    automatically skips these samples via the ``REQUIRES_SPEECH_REFERENCE``
    capability declared by this class.
    """

    # 显式 capability：子类化后仍保留 pure metadata 门控，不依赖类名 fallback
    REQUIRES_SPEECH_REFERENCE = True

    def __init__(
        self,
        n_fft: int = 320,
        hop_length: int = 160,
        sr: int = 16000,
        n_bins: int = 18,
        scale: str = "mel",
        weights_type: str = "log_snr_speech_norm",
        gamma: float = 0.2,
        eps: float = 1e-8,
        sdr_min_db: float = -10.0,
        sdr_max_db: float = 35.0,
    ):
        super().__init__()
        if scale not in ("mel", "critical"):
            raise ValueError(f"FW-SDR scale must be 'mel' or 'critical', got {scale}")
        if n_bins not in (18,):
            raise ValueError(f"FW-SDR n_bins must be 18, got {n_bins}")
        if weights_type not in (
            "log_snr_speech_norm", "softmax_inverse_snr_speech_norm",
            "speech_norm", "perceptual", "inverse_spectral_diff",
        ):
            raise ValueError(f"FW-SDR unsupported weights_type: {weights_type}")
        if not sdr_min_db < sdr_max_db:
            raise ValueError(
                f"FW-SDR sdr_min_db 必须 < sdr_max_db, got {sdr_min_db} >= {sdr_max_db}"
            )
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.sr = int(sr)
        self.n_bins = int(n_bins)
        self.scale = scale
        self.weights_type = weights_type
        self.gamma = float(gamma)
        self.eps = float(eps)
        # JAE-H04: 切线拐点/上界, 语义见 _compute_fw_sdr step 4 注释
        self.sdr_min_db = float(sdr_min_db)
        self.sdr_max_db = float(sdr_max_db)

    REQUIRES_UNDESIRED_REFERENCE = True

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        undesired_ref: torch.Tensor | None = None,
        **__,
    ) -> torch.Tensor:
        """Compute frequency-weighted negative SI-SDR.

        Args:
            enhanced:       [B, T]  enhanced/estimated target speech
            clean:          [B, T]  clean/target speech reference
            lengths:        [B]     有效样本数; 提供时零均值/频带功率/加权聚合
                                    均只在有效区计算, 保证对 batch 零填充不变
                                    (JAE-H02); None 时保持原行为。
            undesired_ref:  [B, T]  real undesired reference paired with
                                    ``clean``, typically ``degraded_input -
                                    clean``. The calling adapter must provide
                                    it as a shape/device-matched floating
                                    Tensor; missing or invalid references
                                    raise before the silent early return. No
                                    output-derived estimate or fallback is used.

        Returns:
            Scalar loss.
        """
        if not isinstance(undesired_ref, torch.Tensor):
            raise TypeError(
                "FWNegSISDRLoss requires undesired_ref to be a Tensor, "
                f"got {type(undesired_ref).__name__}"
            )
        if undesired_ref.shape != clean.shape:
            raise ValueError(
                "FWNegSISDRLoss requires undesired_ref.shape to match clean.shape, "
                f"got {tuple(undesired_ref.shape)} vs {tuple(clean.shape)}"
            )
        if undesired_ref.device != clean.device:
            raise ValueError(
                "FWNegSISDRLoss requires undesired_ref.device to match clean.device, "
                f"got {undesired_ref.device} vs {clean.device}"
            )
        if not undesired_ref.is_floating_point():
            raise TypeError(
                "FWNegSISDRLoss requires undesired_ref to have a floating-point dtype, "
                f"got {undesired_ref.dtype}"
            )

        # Mask out samples where clean is near-zero (no useful training signal).
        # lengths 给定时只统计有效样点，避免 padding 抬高能量并误放行静音样本。
        if lengths is not None:
            t = torch.arange(clean.shape[-1], device=clean.device)
            valid_mask = (t.unsqueeze(0) < lengths.unsqueeze(1)).to(clean.dtype)
            valid_count = lengths.to(clean.dtype).clamp(min=1.0)
            clean_rms = ((clean ** 2) * valid_mask).sum(dim=-1) / valid_count
        else:
            clean_rms = (clean ** 2).mean(dim=-1)
        valid = clean_rms > self.eps
        if not valid.any():
            # 全静音：返回图保留零（经 enhanced 连图），保证 DDP hook 与
            # 优化器正常；与 time_domain.SISNRLoss 的 (0.0*snr).sum() 同范式。
            return (enhanced * 0.0).sum()

        enhanced_v = enhanced[valid]
        clean_v = clean[valid]
        noise_v = undesired_ref[valid]
        lengths_v = lengths[valid] if lengths is not None else None

        return _compute_fw_sdr(
            enhanced_v, clean_v, noise_v,
            sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_bins=self.n_bins, scale=self.scale,
            weights_type=self.weights_type, gamma=self.gamma, eps=self.eps,
            lengths=lengths_v,
            sdr_min_db=self.sdr_min_db, sdr_max_db=self.sdr_max_db,
        )
