"""TF-SkiMNet：inplace 时频建模与跨 block skipping memory。

论文：Zixuan Li et al., "TF-SkiMNet: Speech Enhancement Based on Inplace
Modeling and Skipping Memory in Time-Frequency Domain", Interspeech 2025。

本实现是 ``BaseSEModel`` 标准谱进谱出适配：输入和输出均为 ``(B,F,T,2)``，
STFT/iSTFT 由框架顶层 ``stft`` 块统一管理。频率 BiLSTM 只在单个时间帧内沿
频率轴双向扫描；所有时间方向算子均为单向 LSTM 或 causal Conv1d。当前提供
离线整段 causal forward，不提供部署级逐帧 cache API。固定上游来源、无许可证
边界及“不声称论文数值复现”声明见 ``tf_skimnet_impl/PROVENANCE``。
"""

from __future__ import annotations

import torch

from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.amp_utils import fp32_forward
from tasks.se.models.base_model import BaseSEModel
from tasks.se.models.modules.tf_skimnet_impl import TFSkiMBackbone


def _validate_odd_positive(value: int, *, name: str) -> None:
    if type(value) is not int or value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value!r}")


@MODEL_REGISTRY.register("TFSkiMNet")
class TFSkiMNet(BaseSEModel):
    """TF-SkiMNet 单通道复谱映射模型。

    Args:
        n_fft/hop_length/win_length/window/center/normalized: 框架注入的 STFT
            契约；模型只记录和校验，不在内部执行变换。
        num_blocks: TF-SkiM block 数，论文/官方默认 6。官方 TF 版本逐帧提取
            full-band state，没有暴露原始时域 SkiM 的 chunk size/segment length；
            本参数控制的是 block 循环与相邻 block 间的 state transition 次数。
        hidden_size: 隐藏通道数，论文/官方默认 32。
        encoder_kernel_size: encoder/decoder 的 causal Conv1d kernel。
        local_frequency_kernel: Local-F 邻频卷积 kernel。
        local_temporal_kernel: Local-T causal Conv1d kernel。
        local_temporal_groups: Local-T grouped convolution 的 groups。
        use_hidden_state_memory/use_cell_state_memory: 是否用单向 Global-T LSTM
            处理对应状态；关闭时仍传播未经时间 LSTM 处理的 skipping state。
        use_global_temporal: 是否把当前 block 的 hidden/cell 传给下一 block。
        use_local_temporal: 是否启用 Local-T。
    """

    def __init__(
        self,
        n_fft: int = 320,
        hop_length: int = 160,
        win_length: int = 320,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        num_blocks: int = 6,
        hidden_size: int = 32,
        encoder_kernel_size: int = 5,
        local_frequency_kernel: int = 5,
        local_temporal_kernel: int = 5,
        local_temporal_groups: int = 2,
        use_hidden_state_memory: bool = True,
        use_cell_state_memory: bool = True,
        use_global_temporal: bool = True,
        use_local_temporal: bool = True,
    ) -> None:
        super().__init__()
        for name, value in {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "num_blocks": num_blocks,
            "hidden_size": hidden_size,
            "local_temporal_groups": local_temporal_groups,
        }.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not hop_length <= win_length <= n_fft:
            raise ValueError("STFT geometry must satisfy hop_length <= win_length <= n_fft")
        if type(window) is not str or not window:
            raise ValueError("window must be a non-empty string")
        if type(center) is not bool or type(normalized) is not bool:
            raise TypeError("center and normalized must be bool values")
        _validate_odd_positive(encoder_kernel_size, name="encoder_kernel_size")
        _validate_odd_positive(local_frequency_kernel, name="local_frequency_kernel")
        _validate_odd_positive(local_temporal_kernel, name="local_temporal_kernel")
        if hidden_size % local_temporal_groups != 0:
            raise ValueError("hidden_size must be divisible by local_temporal_groups")
        for name, value in {
            "use_hidden_state_memory": use_hidden_state_memory,
            "use_cell_state_memory": use_cell_state_memory,
            "use_global_temporal": use_global_temporal,
            "use_local_temporal": use_local_temporal,
        }.items():
            if type(value) is not bool:
                raise TypeError(f"{name} must be a bool")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window.lower()
        self.center = center
        self.normalized = normalized
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.backbone = TFSkiMBackbone(
            num_blocks=num_blocks,
            input_channels=2,
            output_channels=2,
            hidden_size=hidden_size,
            encoder_kernel_size=encoder_kernel_size,
            local_frequency_kernel=local_frequency_kernel,
            local_temporal_kernel=local_temporal_kernel,
            local_temporal_groups=local_temporal_groups,
            use_hidden_state_memory=use_hidden_state_memory,
            use_cell_state_memory=use_cell_state_memory,
            use_global_temporal=use_global_temporal,
            use_local_temporal=use_local_temporal,
        )

    @fp32_forward
    def forward(
        self,
        noisy_spec: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        fs: int | None = None,
    ) -> torch.Tensor:
        """增强 ``(B,F,T,2)`` 复谱并保持原布局。"""
        if noisy_spec.ndim != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(
                "TFSkiMNet expects spectrum input (B, F, T, 2), got "
                f"{tuple(noisy_spec.shape)}"
            )
        expected_frequency = self.n_fft // 2 + 1
        if noisy_spec.shape[1] != expected_frequency:
            raise ValueError(
                f"TFSkiMNet expects {expected_frequency} frequency bins for "
                f"n_fft={self.n_fft}, got {noisy_spec.shape[1]}"
            )
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != noisy_spec.shape[0]:
                raise ValueError("lengths must have shape (B,)")
        if fs is not None and fs != 16000:
            raise ValueError(f"TFSkiMNet paper configuration requires fs=16000, got {fs}")
        return self.backbone(noisy_spec.float())
