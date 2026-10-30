"""原始 DCCRN 的框架 STFT 适配：复谱输入、复谱输出。

模型依据 Hu et al., *DCCRN: Deep Complex Convolution Recurrent Network for
Phase-Aware Speech Enhancement*（Interspeech 2020）及官方实现重写。其
encoder–complex LSTM–decoder 属于经典 complex-domain phase-aware SE 基线；任务描述中的
“Luo & Yu 风格”仅表示这类复数域标准基线的谱系，不改变论文实际作者归属。

本适配删除上游内部 ConvSTFT/ConviSTFT，只接受并返回框架统一的
``(B,F,T,2)`` 频谱。它是 DCCRN，不是 ``clearervoice.py`` 中的 FRCRN；两者网络、
checkpoint 和注册名均不兼容，不应混用。固定上游版本与许可证见
``tasks.se.models.modules.dccrn_impl.PROVENANCE``。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.base_model import BaseSEModel
from tasks.se.models.modules.dccrn_impl import DCCRNCore


@MODEL_REGISTRY.register("DCCRN")
class DCCRN(BaseSEModel):
    """DCCRN complex-domain SE 基线，仅处理框架 ``(B,F,T,2)`` 复谱。

    论文出处为 Hu et al.（Interspeech 2020）的原始 DCCRN。模型保留复卷积、复 LSTM、
    complex skip connection 与 E/C/R 掩码；STFT/iSTFT 完全由顶层 ``stft`` 块管理。
    本类与 ClearerVoice FRCRN 不共享结构或 checkpoint。

    Args:
        n_fft: 框架 STFT FFT 长度；频率轴必须满足六层二倍降采样几何。
        hop_length: 框架 STFT 帧移，仅记录并校验，不在模型内部使用。
        win_length: 框架 STFT 窗长，仅记录并校验。
        window: 框架 STFT 窗类型，仅记录并校验。
        center: 框架 STFT 中心填充开关，仅记录并校验。
        rnn_layers: recurrent bottleneck 层数。
        rnn_units: 复 LSTM 的实部与虚部合计 hidden size。
        masking_mode: ``E``（幅相）、``C``（复乘）或 ``R``（分量缩放）。
        use_clstm: 是否使用上游 DCCRN-CL 复 LSTM；否则使用实 LSTM。
        use_cbn: 是否使用复 BatchNorm；默认沿用官方示例的普通 BatchNorm。
        kernel_size: encoder/decoder 频率轴卷积核大小。
        kernel_num: 六层 encoder 的实部与虚部合计通道数。
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 100,
        win_length: int = 400,
        window: str = "hann",
        center: bool = True,
        rnn_layers: int = 2,
        rnn_units: int = 256,
        masking_mode: str = "E",
        use_clstm: bool = True,
        use_cbn: bool = False,
        kernel_size: int = 5,
        kernel_num: Sequence[int] = (16, 32, 64, 128, 256, 256),
    ) -> None:
        super().__init__()
        if n_fft <= 0 or n_fft % 2:
            raise ValueError("n_fft must be a positive even integer")
        if hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if win_length <= 0 or win_length > n_fft:
            raise ValueError("win_length must satisfy 0 < win_length <= n_fft")
        if hop_length > win_length:
            raise ValueError("hop_length must not exceed win_length")
        if not isinstance(window, str) or not window:
            raise ValueError("window must be a non-empty string")
        if not isinstance(center, bool):
            raise TypeError("center must be a bool")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.window = window
        self.center = center
        self.core = DCCRNCore(
            n_fft=n_fft,
            rnn_layers=rnn_layers,
            rnn_units=rnn_units,
            masking_mode=masking_mode,
            use_clstm=use_clstm,
            use_cbn=use_cbn,
            kernel_size=kernel_size,
            kernel_num=kernel_num,
        )

    def forward(
        self,
        noisy_spec: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        fs: int | None = None,
    ) -> torch.Tensor:
        """增强 ``(B,F,T,2)`` 频谱并返回相同布局和形状。"""
        if noisy_spec.ndim != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(
                "DCCRN expects spectrum input shaped (B,F,T,2); configure a top-level stft block"
            )
        expected_frequencies = self.n_fft // 2 + 1
        if noisy_spec.shape[1] != expected_frequencies:
            raise ValueError(
                f"DCCRN expects F={expected_frequencies} from n_fft={self.n_fft}, "
                f"got F={noisy_spec.shape[1]}"
            )
        if noisy_spec.shape[0] == 0 or noisy_spec.shape[2] == 0:
            raise ValueError("DCCRN requires non-empty batch and time dimensions")
        return self.core(noisy_spec)
