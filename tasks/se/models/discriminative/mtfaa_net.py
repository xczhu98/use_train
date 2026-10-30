"""MTFAA-Net 的框架频谱适配器。

参考实现是非官方 PyTorch 实现，其 README 明确声明：
"Implementation details may not be consistent with the paper"。本模块不声称与论文
实现逐项一致，也不继承论文或参考仓库报告的指标。来源与许可证见
``tasks/se/models/modules/mtfaa_impl/PROVENANCE``。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from tasks.aec.models.modules.mtfaa_mask import MTFAAMaskHead
from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.base_model import BaseSEModel
from tasks.se.models.modules.mtfaa_impl import MTFAABackbone


@MODEL_REGISTRY.register("MTFAANet")
class MTFAANet(BaseSEModel):
    """MTFAA-Net 风格单通道语音增强模型。

    输入和输出均为 ``(B, F, T, 2)`` 复频谱。模型内部只执行 Phase Encoder、
    线性 filter bank、multi-scale TFCM/axial attention 与共享 MTFAA mask head；
    STFT/iSTFT 始终由顶层 ``stft`` 块管理。

    注意：参考仓库是非官方实现，且明确声明实现细节可能与论文不一致。本适配器
    不承诺论文参数量、复杂度或增强指标。

    Args:
        n_fft: 顶层 STFT 的 FFT 长度。
        hop_length: 顶层 STFT 的帧移，仅记录接口契约。
        win_length: 顶层 STFT 的窗长，必须不大于 ``n_fft``。
        window: 顶层 STFT 的窗类型。
        center: 顶层 STFT 是否居中补齐。
        sample_rate: 模型预期采样率；传入 ``fs`` 时必须一致。
        phase_channels: Phase Encoder 输出复通道数，当前固定为 4。
        channels: 三层 encoder 的通道数。
        filter_bands: 线性 filter bank 的频带数。
        bottleneck_layers: bottleneck stage 数量。
        tfcm_layers: 每个 TFCM 的膨胀卷积层数。
        causal: 是否使用因果时间卷积与因果时间注意力。
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        window: str = "hann",
        center: bool = True,
        sample_rate: int = 16000,
        phase_channels: int = 4,
        channels: Sequence[int] = (48, 96, 192),
        filter_bands: int = 64,
        bottleneck_layers: int = 2,
        tfcm_layers: int = 6,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if n_fft <= 0 or hop_length <= 0 or win_length <= 0:
            raise ValueError("n_fft, hop_length and win_length must be positive")
        if win_length > n_fft:
            raise ValueError("win_length must not exceed n_fft")
        if not isinstance(window, str) or not window:
            raise ValueError("window must be a non-empty string")
        if not isinstance(center, bool):
            raise TypeError("center must be a bool")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.sample_rate = sample_rate
        self.causal = causal
        self.backbone = MTFAABackbone(
            n_fft // 2 + 1,
            phase_channels=phase_channels,
            channels=channels,
            filter_bands=filter_bands,
            bottleneck_layers=bottleneck_layers,
            tfcm_layers=tfcm_layers,
            causal=causal,
        )
        self.mask_head = MTFAAMaskHead(phase_channels)

    def forward(
        self,
        noisy_spec: Tensor,
        lengths: Tensor | None = None,
        *,
        fs: int | None = None,
    ) -> Tensor:
        """增强框架 STFT 产生的单通道复频谱。"""
        del lengths
        expected_frequencies = self.n_fft // 2 + 1
        if noisy_spec.ndim != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(
                "MTFAANet expects noisy_spec with shape (B, F, T, 2)"
            )
        if noisy_spec.shape[1] != expected_frequencies:
            raise ValueError(
                f"MTFAANet expects {expected_frequencies} frequency bins for "
                f"n_fft={self.n_fft}, got {noisy_spec.shape[1]}"
            )
        if fs is not None and fs != self.sample_rate:
            raise ValueError(
                f"MTFAANet was configured for {self.sample_rate} Hz, got fs={fs}"
            )
        if noisy_spec.shape[0] == 0 or noisy_spec.shape[2] == 0:
            raise ValueError("MTFAANet requires non-empty batch and time dimensions")
        if not torch.is_floating_point(noisy_spec):
            raise TypeError("noisy_spec must be a floating-point tensor")

        spectrum_2ch = noisy_spec.permute(0, 3, 1, 2).contiguous()
        decoder_features = self.backbone(spectrum_2ch)
        return self.mask_head(decoder_features, noisy_spec)
