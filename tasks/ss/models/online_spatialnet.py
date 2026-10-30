"""OnlineSpatialNet 的多源分离版本：多通道 mixture 谱进、K 路源谱出。

与离线 ``tasks/ss/models/spatialnet.py`` 使用同一套上游 NBSS SpatialNet 组网，
但把窄带时序替换为 causal 分支（``mhsa(scope)`` / ``ret(factor,share_qk)`` /
``mamba(d_state,d_conv)``，见 ``tasks.se.models.modules.online_spatialnet_impl``），
因此训练与流式推理共享同一套权重和 cache ABI。

构造默认值与离线 SS ``SpatialNet`` 一致（``dim_hidden=96``、``dim_ffn=192``、
``num_heads=4``、``num_layers=8``），即论文 SpatialNet-small 口径。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import Tensor, nn

from core.models.io_contract import ModelIOContract, TensorField
from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.modules.online_spatialnet_impl.backbone import OnlineSpatialBackbone


@MODEL_REGISTRY.register("OnlineSpatialNet")
class OnlineSpatialNet(nn.Module):
    """Causal OnlineSpatialNet separator: ``[B,C,F,T,2]`` mixture to ``[B,K,F,T,2]`` sources."""

    input_domain = "spectrum"
    output_domain = "spectrum"
    supports_multichannel_input = True
    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"ss"}),
        inputs=(TensorField("mixture_spectra", "spectrum", ("batch", "microphone", "frequency", "frame", "complex")),),
        outputs=(TensorField("est_sources_spectra", "spectrum", ("batch", "source", "frequency", "frame", "complex"), multiple=True),),
        forward_kind="estimate",
        frontend_owner="runtime",
    )

    def __init__(
        self,
        n_channel: int = 6,
        num_sources: int = 2,
        ref_channel: int = 0,
        dim_hidden: int = 96,
        dim_ffn: int = 192,
        dim_squeeze: int = 8,
        num_layers: int = 8,
        num_heads: int = 4,
        num_freqs: int = 129,
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
        n_fft: int | None = None,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str | None = None,
        center: bool | None = None,
    ) -> None:
        super().__init__()
        if type(n_channel) is not int or n_channel < 1:
            raise ValueError("n_channel must be a positive int")
        if type(num_sources) is not int or num_sources < 1:
            raise ValueError("num_sources must be a positive int")
        if type(ref_channel) is not int or not 0 <= ref_channel < n_channel:
            raise ValueError("ref_channel must index an available microphone")
        if type(num_freqs) is not int or num_freqs < 1:
            raise ValueError("num_freqs must be a positive int")
        if n_fft is not None and n_fft // 2 + 1 != num_freqs:
            raise ValueError("OnlineSpatialNet num_freqs must match stft.n_fft")
        if hop_length is not None and hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if win_length is not None and win_length <= 0:
            raise ValueError("win_length must be positive")
        if window is not None and window.lower() not in {"hann", "hamming"}:
            raise ValueError("OnlineSpatialNet supports hann or hamming windows")
        if center is not None and type(center) is not bool:
            raise TypeError("center must be bool")
        self.n_channel = n_channel
        self.num_sources = num_sources
        self.ref_channel = ref_channel
        self.num_freqs = num_freqs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window or "hann"
        self.center = center
        self.backbone = OnlineSpatialBackbone(
            n_channel=n_channel,
            num_freqs=num_freqs,
            output_dim=num_sources * 2,
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

    def _validate_input(self, mixture_spectra: Tensor) -> Tensor:
        if mixture_spectra.ndim == 4:
            if self.n_channel != 1:
                raise ValueError("OnlineSpatialNet expects [B,C,F,T,2] for n_channel > 1")
            mixture_spectra = mixture_spectra.unsqueeze(1)
        if mixture_spectra.ndim != 5 or mixture_spectra.shape[1] != self.n_channel or mixture_spectra.shape[-1] != 2:
            raise ValueError("OnlineSpatialNet expects mixture_spectra with shape [B, C, F, T, 2]")
        if mixture_spectra.shape[2] != self.num_freqs:
            raise ValueError(
                f"OnlineSpatialNet expected {self.num_freqs} frequency bins, got {mixture_spectra.shape[2]}"
            )
        return mixture_spectra

    def _sources(self, spectrum: Tensor) -> dict[str, Tensor]:
        batch, freqs, frames, _ = spectrum.shape
        sources = spectrum.reshape(batch, freqs, frames, self.num_sources, 2)
        return {"est_sources_spectra": sources.permute(0, 3, 1, 2, 4)}

    def init_cache(self, batch_size: int, device, dtype):
        return self.backbone.init_cache(batch_size, device, dtype)

    def forward_streaming(self, mixture_spectra: Tensor, cache: Mapping[str, Tensor]):
        mixture_spectra = self._validate_input(mixture_spectra)
        output, next_cache = self.backbone.forward_streaming(mixture_spectra, cache)
        return self._sources(output), next_cache

    def forward(self, mixture_spectra: Tensor, lengths=None, *, fs=None):
        del lengths, fs
        mixture_spectra = self._validate_input(mixture_spectra)
        return self._sources(self.backbone.forward_sequence(mixture_spectra))
