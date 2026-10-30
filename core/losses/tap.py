"""TAPLoss: Temporal Acoustic Parameter Loss for speech enhancement.

Reference implementation:
  - Yunyang Zeng et al., "TAPLoss: A Temporal Acoustic Parameter Loss for
    Speech Enhancement", arXiv:2302.08088.
  - Official code: https://github.com/YunyangZeng/TAPLoss

项目接口遵循仓库 loss 约定::

    loss(enhanced, clean, lengths=None) -> scalar

其中 ``enhanced`` 在前、``clean`` 在后；官方 ``AcousticLoss`` 的
``(clean_waveform, enhan_waveform, mode)`` 参数顺序不能直接套用到本项目。
TAP estimator 的参数被冻结，但前向图仍保留到 enhanced waveform，使该 loss
可以作为增强模型的可微辅助目标。官方实现的 `frame_energy_weighted_*` 模式
使用增强语音 STFT 的逐帧能量作为权重。``mode`` 仅作为兼容参数接受；
``mode='eval'`` 不会自动关闭 autograd，调用方若需要无梯度评估必须显式使用
``torch.no_grad()``。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from core.utils.registry import LOSS_REGISTRY
from core.utils.safe_load import safe_torch_load


_TAP_LOSS_TYPES = frozenset(
    {"l2", "l1", "frame_energy_weighted_l2", "frame_energy_weighted_l1"}
)


class _TAPAcousticEstimator(nn.Module):
    """官方 TAP estimator 架构：514 维复谱输入，输出 25 个声学参数。"""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(514, 256, 3, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 25)
        self.act = nn.ReLU()

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(spectrogram)
        hidden = self.linear1(hidden)
        hidden = self.act(hidden)
        return self.linear2(hidden)


@LOSS_REGISTRY.register("TAPLoss")
class TAPLoss(nn.Module):
    """Temporal Acoustic Parameter Loss。

    项目统一 loss 契约是 ``forward(enhanced, clean, lengths=None)``，因此参数
    顺序与官方 ``AcousticLoss(clean_waveform, enhan_waveform, mode)`` 不同；
    ``AcousticLoss`` 在本项目仅是 registry/导入别名，不保留官方参数顺序。
    ``mode`` 是保留给旧调用方的兼容参数，只接受 ``"train"``/``"eval"``；它
    不改变计算图，尤其 ``mode="eval"`` 仍允许 enhanced 的梯度传播。需要无梯度
    评估时，调用方必须显式使用 ``torch.no_grad()``。

    Args:
        loss_type: ``l2``、``l1``、``frame_energy_weighted_l2`` 或
            ``frame_energy_weighted_l1``，四种官方 loss mode 均已实现。
        acoustic_model_path: TAP estimator checkpoint。checkpoint 可以是
            ``{"model_state_dict": state_dict}`` 或直接的 state dict；使用项目
            的 ``safe_torch_load`` 安全加载并严格校验参数，加载后的 estimator
            参数冻结。
        device: estimator 与矩形窗 buffer 的初始设备，默认 CPU；训练框架可随后
            通过 ``.to(device)`` 将整个 loss 移到目标设备。
        sample_rate: 只能为官方训练采样率 16000 Hz。
        n_fft: 只能为官方固定值 512。
        hop_length: 只能为官方固定值 160。
        win_length: 只能为官方固定值 512；``None`` 等价于 512。

    TAP estimator 的时频几何固定为 16 kHz、``512/160/512``，不会继承模型顶层
    STFT。STFT 使用显式 ones 矩形窗和 ``center=True``，实虚部按每个频点交替的
    ``[real_0, imag_0, real_1, imag_1, ...]`` 布局送入 estimator。官方帧能量为
    ``2 / n_fft * sum(one_sided_power)``，包括 DC/Nyquist 在内均乘 2，保持官方
    数值口径而非严格 one-sided Parseval 端点修正。

    全部 STFT、差值、权重和归约在 FP32 计算。``lengths`` 给定时，每条样本先按
    有效 waveform 独立计算 STFT/estimator，避免 padding 污染双向 LSTM；最终
    reduction 是全 batch 所有有效帧与 25 个参数的加权 mean，而非先逐样本 mean
    再平均，padding 区不参与 loss 或梯度。
    """

    # builder 读取这些契约，拒绝不匹配的运行采样率并保持 estimator 固定几何。
    REQUIRED_SAMPLE_RATE = 16000
    USE_FRAMEWORK_STFT_DEFAULTS = False
    # clean-speech estimator 不能用于 undesired/noise residual 的 SS 分支。
    SUPPORTS_SS_UNDESIRED_ENERGY = False

    def __init__(
        self,
        loss_type: str = "l1",
        acoustic_model_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int | None = None,
    ) -> None:
        super().__init__()
        if loss_type not in _TAP_LOSS_TYPES:
            raise ValueError(
                f"Invalid TAPLoss loss_type {loss_type!r}; "
                f"choose one of {sorted(_TAP_LOSS_TYPES)}"
            )
        if acoustic_model_path is None:
            raise ValueError("TAPLoss requires acoustic_model_path")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate != 16000:
            raise ValueError(
                f"TAPLoss requires sample_rate=16000, got {sample_rate!r}"
            )
        if n_fft != 512:
            raise ValueError("TAPLoss estimator requires n_fft=512 (input size 514)")
        if hop_length != 160:
            raise ValueError(f"TAPLoss requires hop_length=160, got {hop_length!r}")
        if win_length is None:
            win_length = n_fft
        if win_length != 512:
            raise ValueError("TAPLoss requires win_length == n_fft == 512")

        self.loss_type = loss_type
        self.sample_rate = sample_rate
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.acoustic_model_path = str(acoustic_model_path)
        self.register_buffer(
            "rectangular_window",
            torch.ones(self.win_length, device=device, dtype=torch.float32),
            persistent=False,
        )
        self.estimate_acoustics = _TAPAcousticEstimator()

        checkpoint = safe_torch_load(
            acoustic_model_path,
            map_location="cpu",
        )
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                "TAPLoss checkpoint must be a mapping containing model_state_dict"
            )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, Mapping):
            raise TypeError("TAPLoss checkpoint model_state_dict must be a mapping")
        # strict=True 防止静默使用部分随机初始化的 estimator。
        self.estimate_acoustics.load_state_dict(state_dict, strict=True)
        self.estimate_acoustics.to(device)
        for parameter in self.estimate_acoustics.parameters():
            parameter.requires_grad_(False)
        # estimator 没有 dropout/BN，但固定 eval 可避免外部 loss.train() 改变行为。
        self.estimate_acoustics.eval()

    def train(self, mode: bool = True) -> "TAPLoss":
        """切换 loss 状态，同时保持冻结的 TAP estimator 为 eval 模式。"""
        super().train(mode)
        self.estimate_acoustics.eval()
        return self

    def _stft_features(
        self, waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算官方交替实虚布局与 ``2/n_fft`` one-sided 帧能量。"""
        if waveform.shape[-1] <= self.n_fft // 2:
            raise ValueError(
                "TAPLoss requires each valid waveform length > n_fft // 2 "
                f"({self.n_fft // 2}) for center=True reflect padding"
            )
        spec = torch.stft(
            waveform.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.rectangular_window.float(),
            center=True,
            return_complex=True,
        )
        # (B, F, T) -> (B, T, 2F)，每个频点的 real/imag 相邻，保持官方布局。
        spec_real = spec.real
        spec_imag = spec.imag
        features = torch.view_as_real(spec).permute(0, 2, 1, 3).reshape(
            spec.shape[0], spec.shape[-1], 2 * (self.n_fft // 2 + 1)
        )
        # 官方实现对 one-sided 所有频点统一乘 2，包括 DC 与 Nyquist。
        energy = (spec_real.square() + spec_imag.square()).sum(dim=1) * (2.0 / self.n_fft)
        return features.float(), energy.float()

    def _forward_unpadded(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enhanced_spec, enhanced_energy = self._stft_features(enhanced)
        clean_spec, _ = self._stft_features(clean)
        # TAP estimator 与后续距离始终在 FP32；仅冻结参数，不切断 enhanced 梯度。
        with torch.autocast(device_type=enhanced_spec.device.type, enabled=False):
            enhanced_acoustics = self.estimate_acoustics(enhanced_spec.float())
            clean_acoustics = self.estimate_acoustics(clean_spec.float())
        return (
            enhanced_acoustics.float(),
            clean_acoustics.float(),
            enhanced_energy.float(),
        )

    @staticmethod
    def _validate_lengths(
        lengths: torch.Tensor, batch_size: int, total_samples: int
    ) -> None:
        if lengths.dim() != 1 or lengths.shape[0] != batch_size:
            raise ValueError(
                f"lengths must be [B], got {tuple(lengths.shape)} for B={batch_size}"
            )
        if lengths.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(f"lengths must have integer dtype, got {lengths.dtype}")
        if not bool((lengths >= 1).all()) or not bool((lengths <= total_samples).all()):
            raise ValueError(
                f"lengths values must be in [1, {total_samples}], got {lengths.tolist()}"
            )

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        mode: str = "train",
    ) -> torch.Tensor:
        """比较 ``enhanced`` 与 ``clean`` 的逐帧 25 维 TAP 参数。

        ``mode`` 仅校验兼容调用方传入的训练/评估标记，不会隐式调用
        ``torch.no_grad()``；``mode="eval"`` 仍保留到 ``enhanced`` 的梯度。
        """
        if enhanced.dim() != 2 or clean.dim() != 2:
            raise ValueError(
                f"enhanced and clean must be [B, T], got "
                f"{tuple(enhanced.shape)} and {tuple(clean.shape)}"
            )
        if enhanced.shape != clean.shape:
            raise ValueError(
                f"enhanced/clean shape mismatch: {tuple(enhanced.shape)} vs "
                f"{tuple(clean.shape)}"
            )
        if mode not in {"train", "eval"}:
            raise ValueError("mode must be either 'train' or 'eval'")
        if lengths is not None:
            self._validate_lengths(lengths, enhanced.shape[0], enhanced.shape[1])

        # 无 lengths 时保留官方整批均值口径；有 lengths 时逐样本 trim，避免双向
        # LSTM 看到 batch padding 后改变有效帧的 backward hidden state。
        frame_mask = None
        if lengths is None:
            enhanced_acoustics, clean_acoustics, enhanced_energy = self._forward_unpadded(
                enhanced, clean
            )
        else:
            enhanced_parts = []
            clean_parts = []
            energy_parts = []
            for index, length in enumerate(lengths.tolist()):
                current = self._forward_unpadded(
                    enhanced[index : index + 1, :length],
                    clean[index : index + 1, :length],
                )
                enhanced_parts.append(current[0].squeeze(0))
                clean_parts.append(current[1].squeeze(0))
                energy_parts.append(current[2].squeeze(0))
            enhanced_acoustics = pad_sequence(enhanced_parts, batch_first=True)
            clean_acoustics = pad_sequence(clean_parts, batch_first=True)
            enhanced_energy = pad_sequence(energy_parts, batch_first=True)
            frame_lengths = torch.tensor(
                [part.shape[0] for part in enhanced_parts],
                device=enhanced.device,
                dtype=torch.long,
            )
            frame_mask = (
                torch.arange(enhanced_acoustics.shape[1], device=enhanced.device)[None, :]
                < frame_lengths[:, None]
            ).unsqueeze(-1)

        difference = enhanced_acoustics.float() - clean_acoustics.float()
        enhanced_energy = enhanced_energy.float()
        if frame_mask is None:
            if self.loss_type == "l2":
                return F.mse_loss(difference, torch.zeros_like(difference))
            if self.loss_type == "l1":
                return F.l1_loss(difference, torch.zeros_like(difference))
            if self.loss_type == "frame_energy_weighted_l2":
                weights = torch.sigmoid(enhanced_energy).sqrt().unsqueeze(-1)
                return (weights * difference).square().mean()
            weights = torch.sigmoid(enhanced_energy).unsqueeze(-1)
            return (weights * difference.abs()).mean()

        # 全 batch 有效帧/参数统一归约；帧多的样本按有效帧数自然占更大权重。
        mask = frame_mask.to(dtype=torch.float32)
        denom = mask.sum() * difference.shape[-1]
        if self.loss_type == "l2":
            return (difference.square() * mask).sum() / denom.clamp(min=1.0)
        if self.loss_type == "l1":
            return (difference.abs() * mask).sum() / denom.clamp(min=1.0)
        if self.loss_type == "frame_energy_weighted_l2":
            weights = torch.sigmoid(enhanced_energy).sqrt().unsqueeze(-1)
            return ((weights * difference).square() * mask).sum() / denom.clamp(min=1.0)
        weights = torch.sigmoid(enhanced_energy).unsqueeze(-1)
        return ((weights * difference.abs()) * mask).sum() / denom.clamp(min=1.0)


# 官方仓库类名为 AcousticLoss；注册别名方便迁移官方配置，同时实现保持唯一。
AcousticLoss = TAPLoss
LOSS_REGISTRY.register("AcousticLoss")(TAPLoss)

__all__ = ["TAPLoss", "AcousticLoss"]
