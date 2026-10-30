"""SpatialNet / OnlineSpatialNet 的 SE 任务模型（多通道谱进、单通道参考谱出）。

实现细节在 ``tasks.se.models.modules.online_spatialnet_impl``：

  - ``backbone.OnlineSpatialBackbone``：causal encoder + 交错 cross-band/narrow-band
    layers + decoder，整段与逐帧两条前向共享参数与 cache ABI；
  - ``backbone.resolve_attention``：``mhsa(scope)`` / ``ret(factor,share_qk)`` /
    ``mamba(d_state,d_conv)`` 三个窄带分支的配置解析；
  - ``retention`` / ``mamba``：Retention 与 selective SSM 的窄带实现。

本文件只负责 SE 任务的 IO 合同、几何参数（``n_channel`` / ``ref_channel`` / ``n_fft``）
与输入边界校验。论文 small 配置（``dim_hidden=96``、``dim_ffn=192``、``num_heads=4``、
``num_layers=8``）对应已发表的 1.2 M 参数预算；构造默认值仍是更宽的 ``dim_hidden=192``，
由配置显式给出 small 参数。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import Tensor

from core.models.io_contract import ModelIOContract, TensorField
from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.base_model import BaseSEModel
from tasks.se.models.modules.online_spatialnet_impl.backbone import OnlineSpatialBackbone


@MODEL_REGISTRY.register("OnlineSpatialNet_SE")
class OnlineSpatialNetSE(BaseSEModel):
    """Causal OnlineSpatialNet with tensor-only streaming caches."""

    supports_multichannel_input = True
    streaming_spatial_abi_version = 2
    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"se"}),
        inputs=(TensorField("noisy_spec", "spectrum", ("batch", "microphone", "frequency", "frame", "complex")),),
        outputs=(TensorField("enhanced_spec", "spectrum", ("batch", "frequency", "frame", "complex")),),
        forward_kind="estimate",
        frontend_owner="runtime",
    )

    def __init__(
        self,
        n_fft: int = 512,
        n_channel: int = 6,
        ref_channel: int = 0,
        num_layers: int = 8,
        dim_hidden: int = 192,
        dim_ffn: int = 384,
        dim_squeeze: int = 8,
        num_heads: int = 2,
        attention_scope: int | None = None,
        attention: str | None = None,
        decay: Sequence[int] | int | bool | None = None,
        rope: bool = False,
        encoder_kernel_size: int = 5,
        kernel_size: Sequence[int] = (5, 3),
        conv_groups: Sequence[int] = (8, 8),
        norms: Sequence[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
        dropout: Sequence[float] = (0, 0, 0),
        full_share: int = 0,
        gradient_checkpointing: bool = False,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str | None = None,
        center: bool | None = None,
        target_fs: int | None = None,
    ) -> None:
        super().__init__()
        if type(n_fft) is not int or n_fft < 2:
            raise ValueError("n_fft must be an integer >= 2")
        if type(n_channel) is not int or n_channel < 1:
            raise ValueError("n_channel must be a positive integer")
        if type(ref_channel) is not int or not 0 <= ref_channel < n_channel:
            raise ValueError("ref_channel must index an available microphone")
        self.n_fft = n_fft
        self.n_channel = n_channel
        self.ref_channel = ref_channel
        self.num_freqs = n_fft // 2 + 1
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window or "hann"
        self.center = center
        self.target_fs = target_fs
        self.backbone = OnlineSpatialBackbone(
            n_channel=n_channel,
            num_freqs=self.num_freqs,
            output_dim=2,
            num_layers=num_layers,
            dim_hidden=dim_hidden,
            dim_ffn=dim_ffn,
            dim_squeeze=dim_squeeze,
            num_heads=num_heads,
            attention_spec=attention,
            attention_scope=attention_scope,
            decay=decay,
            rope=rope,
            encoder_kernel_size=encoder_kernel_size,
            kernel_size=kernel_size,
            conv_groups=conv_groups,
            norms=norms,
            dropout=dropout,
            full_share=full_share,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.CACHE_KEYS = self.backbone.CACHE_KEYS
        self.attention_branch = self.backbone.attention_branch
        self.attention_spec = self.backbone.attention_spec

    def init_cache(self, batch_size: int, device, dtype):
        return self.backbone.init_cache(batch_size, device, dtype)

    def forward_streaming(self, mixture_spec: Tensor, cache: Mapping[str, Tensor]):
        if mixture_spec.ndim != 5:
            raise ValueError("OnlineSpatialNetSE expects [B,C,F,T,2] spectrum input")
        return self.backbone.forward_streaming(mixture_spec, cache)

    def forward(self, noisy_spec: Tensor, lengths=None, *, fs: int | None = None) -> Tensor:
        del lengths
        if fs is not None and self.target_fs is not None and fs != self.target_fs:
            raise ValueError(f"OnlineSpatialNetSE requires fs={self.target_fs}, got {fs}")
        if noisy_spec.ndim == 4:
            if self.n_channel != 1:
                raise ValueError("OnlineSpatialNetSE expects [B,C,F,T,2] for n_channel > 1")
            noisy_spec = noisy_spec.unsqueeze(1)
        if noisy_spec.ndim != 5:
            raise ValueError("OnlineSpatialNetSE expects spectrum shape [B,C,F,T,2]")
        return self.backbone.forward_sequence(noisy_spec)
