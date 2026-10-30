"""DeepFilterNet / DPDFNet paper-specific losses and adapters.

Registered:
  DeepFilterNetMultiResSpecLoss — multi-resolution magnitude spec loss
  DPDFNetOverAttenuationLoss      — anti-over-attenuation penalty
  DeepFilterNetLSNRLoss           — local SNR prediction MSE
  DfAlphaLoss                     — alpha control loss
  DeepFilterNetFullLossWithLSNR   — full loss combining enhancement + LSNR (+ alpha)
  DeepFilterNetLSNRComponent      — CompositeLoss adapter for LSNR
  DfAlphaLossComponent            — CompositeLoss adapter for alpha

See docs/guides/loss/Loss_Configuration_Guide.md for combination recipes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.losses.utils import create_frame_mask, prepare_centered_stft_inputs
from core.utils.registry import LOSS_REGISTRY


# =============================================================================
# DeepFilterNet Multi-Resolution Spec Loss
# =============================================================================
@LOSS_REGISTRY.register("DeepFilterNetMultiResSpecLoss")
class DeepFilterNetMultiResSpecLoss(nn.Module):
    """DeepFilterNet / DeepFilterNet2 / DeepFilterNet3 使用的多分辨率幅度谱损失。

    对多组 FFT 分辨率分别计算增强与干净幅度谱的 L1 距离并取平均。低 NFFT 侧重时域，
    高 NFFT 侧重频域。论文/实现中常用 NFFT 480, 960, 1920（48kHz），可针对 16kHz 等调整。

    可选 max_freq_hz：仅在该频率以下计算 loss（如 5000，对应论文中约 5kHz 以下）。
    """

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (480, 960, 1920),
        hop_factor: float = 0.5,
        sample_rate: int = 48000,
        max_freq_hz: float | None = 5000.0,
        reduction: str = "mean",
    ):
        super().__init__()
        try:
            self.fft_sizes = tuple(fft_sizes)
        except TypeError as exc:
            raise TypeError("fft_sizes must be an iterable of positive integers") from exc
        if not self.fft_sizes:
            raise ValueError("fft_sizes must not be empty")
        for index, size in enumerate(self.fft_sizes):
            if type(size) is not int or size <= 0:
                raise ValueError(
                    f"fft_sizes[{index}] must be an exact positive integer, got {size!r}"
                )
        if type(hop_factor) not in (int, float) or not math.isfinite(hop_factor) or hop_factor <= 0:
            raise ValueError(f"hop_factor must be a finite positive number, got {hop_factor!r}")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError(f"sample_rate must be an exact positive integer, got {sample_rate!r}")
        if max_freq_hz is not None and (
            type(max_freq_hz) not in (int, float)
            or not math.isfinite(max_freq_hz)
            or max_freq_hz <= 0
        ):
            raise ValueError(
                f"max_freq_hz must be None or a finite positive number, got {max_freq_hz!r}"
            )
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        self.hop_factor = float(hop_factor)
        self.sample_rate = sample_rate
        self.max_freq_hz = None if max_freq_hz is None else float(max_freq_hz)
        self.reduction = reduction

        self._win_lengths = {}
        for n in self.fft_sizes:
            wlen = min(n, max(1, n // 2))
            self._win_lengths[n] = wlen
            self.register_buffer(f"_win_{n}", torch.hann_window(wlen), persistent=False)

    def _stft_mag(
        self,
        x: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: torch.Tensor,
        center: bool = True,
    ) -> torch.Tensor:
        spec = torch.stft(
            x.float(),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            return_complex=True,
            onesided=True,
        )
        return spec.abs().clamp(min=1e-8)

    def _one_resolution(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        n_fft: int,
        lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        hop_length = max(1, int(n_fft * self.hop_factor))
        win = getattr(self, f"_win_{n_fft}")
        win_length = self._win_lengths.get(n_fft, win.shape[0])
        has_lengths = lengths is not None
        (enhanced, clean), lengths, _ = prepare_centered_stft_inputs(
            enhanced, clean, lengths=lengths, n_fft=n_fft
        )

        enh_mag = self._stft_mag(enhanced, n_fft, hop_length, win_length, win)
        cln_mag = self._stft_mag(clean, n_fft, hop_length, win_length, win)

        if self.max_freq_hz is not None and self.max_freq_hz < self.sample_rate / 2:
            # one-sided bin k 的频率为 k * sample_rate / n_fft；截断上界
            # 包含不超过 max_freq_hz 的最后一个 bin，故切片 stop 需 +1。
            max_bin = int(math.floor(self.max_freq_hz * n_fft / self.sample_rate))
            max_stop = min(max_bin + 1, enh_mag.shape[1])
            enh_mag = enh_mag[:, :max_stop]
            cln_mag = cln_mag[:, :max_stop]

        diff = torch.abs(enh_mag - cln_mag)
        frame_mask = create_frame_mask(
            lengths,
            diff.shape[-1],
            n_fft,
            win_length,
            hop_length,
            diff.device,
        ).unsqueeze(1).to(diff.dtype)
        n_valid = (frame_mask.sum() * diff.shape[1]).clamp(min=1.0)
        masked_sum = (diff * frame_mask).sum()
        if has_lengths or self.reduction == "mean":
            return masked_sum / n_valid
        return masked_sum

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
        total = sum(
            self._one_resolution(enhanced, clean, n, lengths)
            for n in self.fft_sizes
        )
        return total / len(self.fft_sizes)


# =============================================================================
# DPDFNet Over-Attenuation Loss
# =============================================================================
@LOSS_REGISTRY.register("DeepFilterNetOverAttenuationLoss")
class DeepFilterNetOverAttenuationLoss(nn.Module):
    """DPDFNet 论文中的抗过衰减损失：惩罚增强幅度远小于干净幅度（过度抑制）。

    在 STFT 幅度域计算：对每帧每频点，若 |enhanced| < |clean| 则惩罚
    (|clean| - |enhanced|) / (|clean| + eps)，使模型避免把语音压得过低。

    可与 DeepFilterNetMultiResSpecLoss 通过 CompositeLoss 组合：
    L = L_multi_res_spec + lambda * L_over_attenuation
    """

    def __init__(
        self,
        n_fft: int = 960,
        hop_length: int | None = None,
        win_length: int | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 2
        self.win_length = min(win_length or self.n_fft, self.n_fft)
        self.eps = eps
        self.register_buffer(
            "_window",
            torch.hann_window(self.win_length),
            persistent=False,
        )

    def _stft_mag(
        self,
        x: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: torch.Tensor,
        center: bool = True,
    ) -> torch.Tensor:
        spec = torch.stft(
            x.float(),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            return_complex=True,
            onesided=True,
        )
        return spec.abs().clamp(min=1e-8)

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
        (enhanced, clean), lengths, _ = prepare_centered_stft_inputs(
            enhanced, clean, lengths=lengths, n_fft=self.n_fft
        )

        enh_mag = self._stft_mag(
            enhanced,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self._window,
        )
        cln_mag = self._stft_mag(
            clean,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self._window,
        )

        magnitude_delta = cln_mag - enh_mag
        under = torch.where(
            magnitude_delta > 0.0,
            magnitude_delta,
            magnitude_delta * 0.0,
        )
        per_bin = under / (cln_mag + self.eps)
        frame_mask = create_frame_mask(
            lengths,
            per_bin.shape[-1],
            self.n_fft,
            self.win_length,
            self.hop_length,
            per_bin.device,
        ).unsqueeze(1).to(per_bin.dtype)
        n_valid = (frame_mask.sum() * per_bin.shape[1]).clamp(min=1.0)
        return (per_bin * frame_mask).sum() / n_valid


# =============================================================================
# LSNR / Alpha / Full / Component adapters
# =============================================================================
def compute_frame_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    frame_length: int,
    hop_length: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算帧级 SNR（ground truth）。

    参考 DeepFilterNet 官方实现的 LocalSnrTarget。
    使用滑窗计算局部 SNR，与论文中 20ms 窗口对应。

    Args:
        clean: (B, T) 干净语音
        noise: (B, T) 噪声信号
        frame_length: 帧长（采样点数），建议 960 对应 20ms @ 48kHz
        hop_length: 帧移
        eps: 数值稳定项

    Returns:
        (B, n_frames) 帧级 SNR，单位 dB
    """
    B, T = clean.shape

    # Unfold: (B, T) -> (B, n_frames, frame_length)
    clean_frames = clean.unfold(1, frame_length, hop_length)  # (B, n_frames, frame_length)
    noise_frames = noise.unfold(1, frame_length, hop_length)

    # 计算每帧能量
    clean_power = (clean_frames ** 2).mean(dim=-1).clamp(min=eps)  # (B, n_frames)
    noise_power = (noise_frames ** 2).mean(dim=-1).clamp(min=eps)

    # SNR (dB)
    snr_db = 10 * torch.log10(clean_power / noise_power)

    return snr_db


@LOSS_REGISTRY.register("DeepFilterNetLSNRLoss")
class DeepFilterNetLSNRLoss(nn.Module):
    """DeepFilterNet Local SNR 辅助损失（MSE）。

    计算模型预测的 lsnr 与 ground truth SNR 之间的 MSE。
    需要数据集提供 clean 和 noise 信号。

    参考：
      - DeepFilterNet 官方 loss.py: LocalSnrLoss (factor=0.0005)
      - 论文: 20ms 窗口计算 LSNR

    Args:
        frame_length: 计算 SNR 的帧长（采样点）
                      建议：960 @ 48kHz (20ms) 或 320 @ 16kHz (20ms)
        hop_length: 帧移（采样点）
        weight: loss 权重，官方默认 0.0005（非常小的辅助任务）
    """

    def __init__(
        self,
        frame_length: int = 960,    # 20ms @ 48kHz (官方论文窗口)
        hop_length: int = 480,      # 10ms @ 48kHz
        weight: float = 0.0005,     # 官方默认值（从 0.01 修正）
    ):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.weight = weight

    def forward(
        self,
        predicted_lsnr: torch.Tensor,
        clean: torch.Tensor,
        noise: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            predicted_lsnr: (B, T_frames) 模型预测的 lsnr（已压缩到 1 维，需 squeeze）
            clean: (B, T) 干净语音
            noise: (B, T) 噪声信号
            lengths: (B,) 可选，有效样本数

        Returns:
            loss: 标量 MSE loss
        """
        # 如果 predicted_lsnr 是 (B, T, 1)，squeeze 到 (B, T)
        if predicted_lsnr.ndim == 3:
            predicted_lsnr = predicted_lsnr.squeeze(-1)

        if clean.shape[1] < self.frame_length:
            return (predicted_lsnr * 0.0).sum()

        # 计算 ground truth SNR
        gt_snr = compute_frame_snr(
            clean, noise,
            self.frame_length,
            self.hop_length
        )

        # 对齐帧数（模型输出可能略有不同）
        min_frames = min(predicted_lsnr.shape[1], gt_snr.shape[1])
        if min_frames == 0:
            return (predicted_lsnr * 0.0).sum()
        predicted_lsnr = predicted_lsnr[:, :min_frames]
        gt_snr = gt_snr[:, :min_frames]

        if lengths is not None:
            fl, hl = self.frame_length, self.hop_length
            if lengths.shape[0] != predicted_lsnr.shape[0]:
                raise ValueError(
                    f"lengths batch {lengths.shape[0]} != predicted_lsnr batch {predicted_lsnr.shape[0]}"
                )
            # 每 utterance 有效帧数（与 unfold 一致），忽略 padding 上无意义帧
            n_valid = (lengths - fl) // hl + 1
            if (n_valid <= 0).any():
                import warnings
                warnings.warn(
                    f"DeepFilterNetLSNRLoss: {(n_valid<=0).sum()}/{n_valid.numel()} samples "
                    f"shorter than frame_length={fl}. Check data pipeline.",
                    RuntimeWarning, stacklevel=2,
                )
            n_valid = n_valid.clamp(min=0)
            n_valid = torch.minimum(
                n_valid,
                torch.full_like(n_valid, min_frames),
            )
            b = predicted_lsnr.shape[0]
            idx = torch.arange(min_frames, device=predicted_lsnr.device).unsqueeze(0).expand(b, -1)
            mask = (idx < n_valid.unsqueeze(1)).to(predicted_lsnr.dtype)
            msum = mask.sum()
            if msum <= 0:
                return (predicted_lsnr * 0.0).sum()
            loss = ((predicted_lsnr - gt_snr) ** 2 * mask).sum() / msum
        else:
            loss = nn.functional.mse_loss(predicted_lsnr, gt_snr)

        return self.weight * loss


@LOSS_REGISTRY.register("DeepFilterNetAlphaLoss")
class DeepFilterNetAlphaLoss(nn.Module):
    """DeepFilterNet Alpha 控制损失。

    控制 Deep Filtering (DF) 的应用时机：
      - 低 SNR（噪声主导）：惩罚 α，强制 DF 关闭（避免增强噪声周期性）
      - 高 SNR（语音主导）：惩罚 (1-α)，鼓励 DF 开启（增强语音周期性）

    参考：
      - DeepFilterNet 论文公式 (7): L_α
      - 官方 loss.py: DeepFilterNetAlphaLoss (factor=0.05)

    Args:
        factor: loss 权重，官方默认 0.05
        lsnr_thresh: SNR 阈值（dB），低于此值开始惩罚 DF 使用
        lsnr_min: 最小 SNR（dB），低于此值完全禁止 DF
    """

    def __init__(
        self,
        factor: float = 0.05,       # 官方默认
        lsnr_thresh: float = -7.5,  # 官方默认
        lsnr_min: float = -10.0,    # 官方默认
        frame_length: int = 320,    # lengths 有效帧掩码用帧长（16kHz 20ms）
        hop_length: int = 160,      # lengths 有效帧掩码用帧移
    ):
        super().__init__()
        self.factor = factor
        self.lsnr_thresh = lsnr_thresh
        self.lsnr_min = lsnr_min
        self.frame_length = frame_length
        self.hop_length = hop_length

    def forward(
        self,
        pred_alpha: torch.Tensor,
        target_lsnr: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_alpha: (B, T, 1) 或 (B, T) 预测的 alpha 权重（0~1）
            target_lsnr: (B, T) ground truth 局部 SNR（dB）
            lengths: (B,) 可选，有效样本数；提供时按有效帧构造逐帧掩码
                     做 masked mean（与 DeepFilterNetLSNRLoss 同模式）

        Returns:
            loss: 标量 alpha loss
        """
        # 确保维度匹配
        if pred_alpha.ndim == 3:
            pred_alpha = pred_alpha.squeeze(-1)  # (B, T, 1) -> (B, T)

        # 帧数对齐：模型输出与 GT SNR 帧网格可能差 1-2 帧（如 center=True
        # 的 padding 帧），统一截断到 min_frames，避免广播错位崩溃。
        min_frames = min(pred_alpha.shape[-1], target_lsnr.shape[-1])
        if min_frames == 0:
            # 无有效帧：返回图保留零（经 pred_alpha 连图），保证
            # DDP hook 与优化器正常。
            return (pred_alpha * 0.0).sum()
        pred_alpha = pred_alpha[:, :min_frames]
        target_lsnr = target_lsnr[:, :min_frames]

        # Loss 1: 低 SNR 场景，惩罚 DF 使用（α 应接近 0）
        # lsnr < -10dB 时权重为 1，lsnr > -7.5dB 时权重为 0
        w_off = self._lsnr_mapping(target_lsnr, self.lsnr_thresh, self.lsnr_min)
        l_off = (pred_alpha * w_off).square()

        # Loss 2: 高 SNR 场景，鼓励 DF 使用（α 应接近 1）
        # lsnr > 0dB 时权重为 1，lsnr < -5dB 时权重为 0
        w_on = self._lsnr_mapping(target_lsnr, self.lsnr_thresh + 2.5, 0.0)
        l_on = 0.1 * ((1 - pred_alpha) * w_on).abs()

        if lengths is not None:
            # 有效帧掩码（照 DeepFilterNetLSNRLoss 356-384 行模式）：
            # 每样本有效帧数 = (lengths - fl) // hl + 1，上限 min_frames。
            if lengths.shape[0] != pred_alpha.shape[0]:
                raise ValueError(
                    f"lengths batch {lengths.shape[0]} != pred_alpha batch {pred_alpha.shape[0]}"
                )
            lengths = lengths.to(device=pred_alpha.device)
            n_valid = (lengths - self.frame_length) // self.hop_length + 1
            n_valid = torch.minimum(
                n_valid.clamp(min=0),
                torch.full_like(n_valid, min_frames),
            )
            b = pred_alpha.shape[0]
            idx = torch.arange(min_frames, device=pred_alpha.device).unsqueeze(0).expand(b, -1)
            mask = (idx < n_valid.unsqueeze(1)).to(pred_alpha.dtype)
            msum = mask.sum()
            if msum <= 0:
                # 全无有效帧：返回图保留零
                return (pred_alpha * 0.0).sum()
            loss = ((l_off + l_on) * mask).sum() / msum
        else:
            loss = (l_off + l_on).mean()

        return loss * self.factor

    def _lsnr_mapping(
        self,
        lsnr: torch.Tensor,
        lsnr_thresh: float,
        lsnr_min: float,
    ) -> torch.Tensor:
        """将 LSNR 映射到 [0, 1] 权重。

        映射规则：
          - lsnr <= lsnr_min:    weight = 1
          - lsnr >= lsnr_thresh: weight = 0
          - 中间线性过渡
        """
        # 线性映射: s = a * lsnr + b
        # 使得 lsnr_min -> 1, lsnr_thresh -> 0
        a = 1 / (lsnr_thresh - lsnr_min)
        b = -a * lsnr_min
        return 1 - torch.clamp(a * lsnr + b, 0.0, 1.0)


@LOSS_REGISTRY.register("DeepFilterNetFullLossWithLSNR")
class DeepFilterNetFullLossWithLSNR(nn.Module):
    """DeepFilterNet 完整 Loss：增强 loss + lsnr loss + alpha loss（可选）。

    适用于 CompositeLoss 的替代方案，直接组合主 loss 和辅助 loss。
    需要模型支持 return_lsnr=True 和 return_alpha=True（如果使用 alpha loss）。

    Args:
        enhancement_loss_type: 主增强 loss 类型（如 "SISNRLoss"）
        enhancement_loss_args: 主 loss 参数；显式采样率参数优先于上下文注入
        stft_cfg: 透传给内部 build_loss 的 STFT 配置
        lsnr_weight: lsnr MSE loss 权重（官方默认 0.0005）
        lsnr_frame_length: lsnr 计算窗口（采样点），建议 960 @ 48kHz (20ms)
        lsnr_hop_length: lsnr 帧移
        use_alpha_loss: 是否使用 alpha 控制 loss（需要模型输出 alpha）
        alpha_weight: alpha loss 权重（官方默认 0.05）
        sample_rate: 构造期采样率上下文；未提供时保持增强 loss 原有默认值
    """

    def __init__(
        self,
        enhancement_loss_type: str = "SISNRLoss",
        enhancement_loss_args: dict | None = None,
        stft_cfg: dict | None = None,
        lsnr_weight: float = 0.0005,        # 修正：官方默认值
        lsnr_frame_length: int = 960,       # 20ms @ 48kHz
        lsnr_hop_length: int = 480,         # 10ms @ 48kHz
        use_alpha_loss: bool = False,       # 是否启用 alpha loss
        alpha_weight: float = 0.05,         # 官方默认值
        sample_rate: int | None = None,
    ):
        super().__init__()
        from core.losses.build import build_loss

        # 构建主 loss；build_loss 负责将采样率上下文注入显式支持的参数，且保留显式 args。
        enh_cfg = {"type": enhancement_loss_type}
        if enhancement_loss_args:
            enh_cfg["args"] = enhancement_loss_args
        self.enhancement_loss = build_loss(
            enh_cfg,
            stft_cfg=stft_cfg,
            sample_rate=sample_rate,
        )

        # LSNR loss
        self.lsnr_loss = DeepFilterNetLSNRLoss(
            frame_length=lsnr_frame_length,
            hop_length=lsnr_hop_length,
            weight=lsnr_weight,
        )

        # Alpha loss（可选）
        self.use_alpha_loss = use_alpha_loss
        if use_alpha_loss:
            self.alpha_loss = DeepFilterNetAlphaLoss(
                factor=alpha_weight,
                frame_length=lsnr_frame_length,
                hop_length=lsnr_hop_length,
            )

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        predicted_lsnr: torch.Tensor | None = None,
        predicted_alpha: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            enhanced: (B, T) 增强波形
            clean: (B, T) 干净语音
            lengths: (B,) 可选
            predicted_lsnr: (B, T_frames) 可选，模型预测的 lsnr
            predicted_alpha: (B, T_frames) 可选，模型预测的 alpha
            noise: (B, T) 可选；未传时若传入 noisy 则 noise = noisy - clean（与 DeepFilterNet-main 一致）

        Returns:
            total_loss: 组合 loss
        """
        if noise is None and kwargs.get("noisy") is not None:
            noise = kwargs["noisy"] - clean
        # 主 loss
        total_loss = self.enhancement_loss(enhanced, clean, lengths)

        # LSNR loss（如果提供）
        if predicted_lsnr is not None and noise is not None:
            lsnr_loss = self.lsnr_loss(predicted_lsnr, clean, noise, lengths)
            total_loss = total_loss + lsnr_loss

            # Alpha loss（如果启用且提供）
            if self.use_alpha_loss and predicted_alpha is not None:
                if clean.shape[1] < self.lsnr_loss.frame_length:
                    # 帧长守卫（对齐 DeepFilterNetLSNRLoss 339-340 行）：
                    # 短于帧长无有效帧，跳过 alpha 项并注入图保留零，
                    # 保持 alpha 头梯度路径完整（DDP 不误判 unused 参数）。
                    total_loss = total_loss + (predicted_alpha * 0.0).sum()
                else:
                    # 计算 GT SNR 用于 alpha loss
                    gt_snr = compute_frame_snr(
                        clean, noise,
                        self.lsnr_loss.frame_length,
                        self.lsnr_loss.hop_length
                    )
                    alpha_loss = self.alpha_loss(predicted_alpha, gt_snr, lengths)
                    total_loss = total_loss + alpha_loss

        return total_loss


@LOSS_REGISTRY.register("DeepFilterNetLSNRComponent")
class DeepFilterNetLSNRComponent(nn.Module):
    """LSNR 的 CompositeLoss 用适配器，与 DeepFilterNet-main 一致：noise 在 loss 内用 noisy - clean 得到。

    接口：predicted_lsnr 必选；noise 可选。当 noise 未提供时，若提供 noisy 则 noise = noisy - clean（严格复现官方）。
    在 CompositeLoss 中与 DeepFilterNetMultiResSpecLoss、DeepFilterNetOverAttenuationLoss 一起使用。
    """

    def __init__(
        self,
        frame_length: int = 320,
        hop_length: int = 160,
    ):
        super().__init__()
        self._lsnr = DeepFilterNetLSNRLoss(
            frame_length=frame_length,
            hop_length=hop_length,
            weight=1.0,
        )

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        predicted_lsnr: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if predicted_lsnr is None:
            raise ValueError(
                "DeepFilterNetLSNRComponent requires predicted_lsnr when enabled"
            )
        # 与 DeepFilterNet-main 一致：noise = noisy - clean（未传入 noise 时用 noisy 计算）
        if noise is None and kwargs.get("noisy") is not None:
            noise = kwargs["noisy"] - clean
        if noise is None:
            raise ValueError(
                "DeepFilterNetLSNRComponent requires noise or noisy when enabled"
            )
        return self._lsnr(predicted_lsnr, clean, noise, lengths)


@LOSS_REGISTRY.register("DfAlphaLossComponent")
class DfAlphaLossComponent(nn.Module):
    """Alpha 的 CompositeLoss 用适配器。noise 未传时可用 noisy（noise = noisy - clean）与 DeepFilterNet-main 一致。"""

    def __init__(
        self,
        frame_length: int = 320,
        hop_length: int = 160,
        factor: float = 0.05,
        lsnr_thresh: float = -7.5,
        lsnr_min: float = -10.0,
    ):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self._alpha_loss = DeepFilterNetAlphaLoss(
            factor=factor,
            lsnr_thresh=lsnr_thresh,
            lsnr_min=lsnr_min,
            frame_length=frame_length,
            hop_length=hop_length,
        )

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        predicted_alpha: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if predicted_alpha is None:
            raise ValueError(
                "DfAlphaLossComponent requires predicted_alpha when enabled"
            )
        if noise is None and kwargs.get("noisy") is not None:
            noise = kwargs["noisy"] - clean
        if noise is None:
            raise ValueError(
                "DfAlphaLossComponent requires noise or noisy when enabled"
            )
        if clean.shape[1] < self.frame_length:
            # 帧长守卫（对齐 DeepFilterNetLSNRLoss 339-340 行）：短于帧长
            # 无有效帧，返回图保留零（经 predicted_alpha 连图），不抛异常。
            return (predicted_alpha * 0.0).sum()
        gt_snr = compute_frame_snr(clean, noise, self.frame_length, self.hop_length)
        return self._alpha_loss(predicted_alpha, gt_snr, lengths)
