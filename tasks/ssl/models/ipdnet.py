"""FN-SSL FixedArray IPDnet 的最小、离线、纯 PyTorch 实现。

输入布局明确为 ``(B, C, F, T)`` 的 complex STFT，或 ``(B, C, F, T, 2)`` 的实虚布局。
feature contract 输出 ``(B, T, F, 2 * (C - 1))``，以 microphone 0 为 reference，
每个通道依次为 cos(IPD), sin(IPD)。模型输出 ``(B, T_out, F, tracks, 2)``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from core.models.io_contract import ModelIOContract, TensorField
from core.utils.registry import MODEL_REGISTRY
from tasks.ssl.model_io import ModelSpec


@dataclass(frozen=True)
class IPDFeatureContract:
    """固定阵列 IPD 特征的布局与 reference-channel 约束。"""

    reference_channel: int = 0

    def validate(self, spectra: Tensor) -> None:
        if spectra.ndim not in (4, 5):
            raise ValueError("spectra must have shape (B,C,F,T) or (B,C,F,T,2)")
        if spectra.shape[1] < 2:
            raise ValueError("IPD features require at least two microphone channels")
        if not 0 <= self.reference_channel < spectra.shape[1]:
            raise ValueError("reference_channel is outside the microphone dimension")

    def extract(self, spectra: Tensor) -> Tensor:
        self.validate(spectra)
        if spectra.ndim == 5:
            if spectra.shape[-1] != 2:
                raise ValueError("last dimension of real/imag spectra must be 2")
            spectra = torch.view_as_complex(spectra.contiguous())
        if not spectra.is_complex():
            raise TypeError("spectra must be complex or have a final real/imag dimension")
        ref = spectra[:, self.reference_channel : self.reference_channel + 1]
        others = torch.cat([spectra[:, : self.reference_channel], spectra[:, self.reference_channel + 1 :]], dim=1)
        phase = others * ref.conj()
        phase = phase / phase.abs().clamp_min(torch.finfo(phase.real.dtype).eps)
        features = torch.stack((phase.real, phase.imag), dim=-1)
        return features.permute(0, 3, 2, 1, 4).reshape(
            features.shape[0], features.shape[3], features.shape[2], -1
        ).contiguous()


class DPIPD(nn.Module):
    """生成固定阵列的 direct-path IPD 模板，供空间谱投影使用。

    ``forward`` 返回 ``(B, T, E, A)``，其中 E/A 是 elevation/azimuth 候选轴。

    注意：候选轴的 elevation 是物理球坐标极角（从 +Z 轴起算，0..π，
    见 ``_make_template`` 的 direction 构造），与
    ``tasks/ssl/data/projection.py::_doa`` 产出的标签口径一致；它不是
    TSE spatial cue 使用的水平面 elevation（见
    ``tasks/tse/data/spatial_cues.py::azimuth_elevation_from_positions``），
    两者相差 ``π/2 - 极角``，不得直接互换。
    """

    def __init__(self, mic_location: Tensor, doa_candidates: tuple[int, int], n_fft: int = 512,
                 sample_rate: int = 16000, speed: float = 343.0, reference_channel: int = 0):
        super().__init__()
        locations = torch.as_tensor(mic_location, dtype=torch.float32)
        if locations.ndim != 2 or locations.shape[1] != 3 or locations.shape[0] < 2:
            raise ValueError("mic_location must have shape (C, 3), with C >= 2")
        if len(doa_candidates) != 2 or min(doa_candidates) < 1:
            raise ValueError("doa_candidates must be (num_elevation, num_azimuth)")
        if not 0 <= reference_channel < locations.shape[0]:
            raise ValueError("reference_channel is outside mic_location")
        self.n_fft, self.sample_rate, self.speed = n_fft, sample_rate, speed
        self.reference_channel = reference_channel
        self.register_buffer("mic_location", locations)
        self.register_buffer("frequencies", torch.linspace(0, sample_rate / 2, n_fft // 2 + 1))
        self.register_buffer("template", self._make_template(locations, doa_candidates))

    def _make_template(self, locations: Tensor, candidates: tuple[int, int]) -> Tensor:
        ne, na = candidates
        elevation = torch.linspace(0.0, math.pi, ne, device=locations.device)
        azimuth = torch.linspace(-math.pi, math.pi, na, device=locations.device)
        direction = torch.stack((torch.sin(elevation)[:, None] * torch.cos(azimuth)[None],
                                 torch.sin(elevation)[:, None] * torch.sin(azimuth)[None],
                                 torch.cos(elevation)[:, None].expand(ne, na)), dim=-1)
        others = torch.cat((locations[: self.reference_channel], locations[self.reference_channel + 1 :]))
        delay = torch.einsum("eak,ck->eac", direction, locations[self.reference_channel] - others) / self.speed
        phase = -2 * math.pi * delay[..., None, :] * self.frequencies[None, None, :, None]
        unit = torch.stack((phase.cos(), phase.sin()), dim=-1)
        return unit.permute(0, 1, 2, 3, 4).contiguous()

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[-1] != 2 * (self.mic_location.shape[0] - 1):
            raise ValueError("features must have shape (B,T,F,2*(C-1))")
        b, t, f, _ = features.shape
        if f != self.frequencies.numel():
            raise ValueError(f"feature frequency bins {f} != configured {self.frequencies.numel()}")
        feat = features.view(b, t, f, -1, 2)
        template = self.template.to(features)
        return torch.einsum("btfcp,eafcp->btea", feat, template) / feat.shape[3]


@MODEL_REGISTRY.register("FixedArrayIPDNet")
class FixedArrayIPDNet(nn.Module):
    """项目原生 IPD baseline。

    该模型只消费已经提取的 ``reference_ipd``，不能用作官方 FN-SSL
    复现。输出保留显式 pair 轴，便于和 canonical DP-IPD target 对齐。
    """

    MODEL_TYPE = "FixedArrayIPDNet"
    MODEL_SPEC = ModelSpec(
        model_type=MODEL_TYPE,
        feature_contract="reference_ipd",
        target_contract="dpipd_pair",
        pair_layout="reference_to_all",
        geometry_mode="fixed_array",
        field_mode="far_field",
        reduction="model_defined",
        causal=False,
        mapping="far_field_grid",
        target_layout="trailing_ri",
    )

    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"ssl"}),
        inputs=(TensorField("features", "embedding", ("batch", "frame", "frequency", "feature")),),
        outputs=(TensorField("dpipd", "spectrum", ("batch", "frame", "frequency", "track", "pair", "complex"), multiple=True),),
        forward_kind="estimate",
        frontend_owner="model",
    )

    def __init__(self, input_size: int = 2, hidden_size: int = 64, max_track: int = 2,
                 n_fft: int = 512, downsample: int = 12, hop_length: int | None = None,
                 win_length: int | None = None, window: str | None = None,
                 center: bool | None = None, **kwargs):
        super().__init__()
        if kwargs:
            raise TypeError(f"unexpected model arguments: {sorted(kwargs)}")
        for name, value, minimum in (
            ("input_size", input_size, 1),
            ("hidden_size", hidden_size, 2),
            ("max_track", max_track, 1),
            ("downsample", downsample, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an exact integer >= {minimum}")
        if type(n_fft) is not int or n_fft < 2:
            raise ValueError("n_fft must be an exact integer >= 2")
        if hop_length is not None and (type(hop_length) is not int or hop_length < 1):
            raise ValueError("hop_length must be a positive exact integer")
        if win_length is not None and (type(win_length) is not int or not 1 <= win_length <= n_fft):
            raise ValueError("win_length must satisfy 1 <= win_length <= n_fft")
        if window is not None and window not in {"hann", "hamming"}:
            raise ValueError("window must be hann or hamming")
        if center is not None and type(center) is not bool:
            raise TypeError("center must be a bool")
        self.input_size, self.downsample, self.max_track = input_size, downsample, max_track
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length
        self.window, self.center = window, center
        self.temporal = nn.GRU(2 * input_size, hidden_size, batch_first=True, bidirectional=True)
        self.head = nn.Conv1d(2 * hidden_size, 2 * max_track * input_size, kernel_size=1)

    def forward(self, features: Tensor, *, offline_inference: bool = False) -> Tensor:
        if features.ndim != 4 or features.shape[-1] != 2 * self.input_size:
            raise ValueError("features must have shape (B,T,F,2*(C-1))")
        b, t, f, _ = features.shape
        if f != self.n_fft // 2 + 1:
            raise ValueError(f"feature frequency bins {f} != configured n_fft bins {self.n_fft // 2 + 1}")
        x = features.permute(0, 2, 1, 3).reshape(b * f, t, -1)
        x, _ = self.temporal(x)
        x = self.head(x.transpose(1, 2)).transpose(1, 2)
        x = x.view(b, f, t, self.max_track, self.input_size, 2).permute(0, 2, 1, 3, 4, 5)
        if offline_inference and t >= self.downsample:
            usable = t - t % self.downsample
            x = x[:, :usable].view(b, usable // self.downsample, self.downsample, f, self.max_track, 2).mean(2)
        return x


class _PairwiseIPDModel(nn.Module):
    """官方模型接入阶段使用的多通道 STFT 代理骨干。

    这里保留一个独立的 proxy 实现用于合同和 CPU smoke；其名称不能被当作
    官方 LSTM/FN block/oSpatialNet 复现结论。多通道 STFT 的 microphone 轴在
    输入张量中显式存在（或以 ``2C`` 实虚维度扁平表示），内部只做相位特征
    准备，输出始终保留 ``[K,P,2]``。
    """

    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"ssl"}),
        inputs=(TensorField(
            "multichannel_stft_real_imag", "spectrum",
            ("batch", "frame", "frequency", "microphone", "complex"),
        ),),
        outputs=(TensorField(
            "dpipd", "spectrum",
            ("batch", "frame", "frequency", "track", "pair", "complex"),
            multiple=True,
        ),),
        forward_kind="estimate",
        frontend_owner="model",
    )

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        max_track: int = 2,
        n_fft: int = 512,
        stride: int = 1,
        causal: bool = False,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str | None = None,
        center: bool | None = None,
        exclude_dc: bool = False,
        normalization: str = "none",
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(f"unexpected model arguments: {sorted(kwargs)}")
        for name, value, minimum in (("input_size", input_size, 1), ("hidden_size", hidden_size, 2),
                                     ("max_track", max_track, 1), ("n_fft", n_fft, 2), ("stride", stride, 1)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an exact integer >= {minimum}")
        if type(causal) is not bool:
            raise TypeError("causal must be a bool")
        if hop_length is not None and (type(hop_length) is not int or hop_length < 1):
            raise ValueError("hop_length must be a positive exact integer")
        if win_length is not None and (type(win_length) is not int or not 1 <= win_length <= n_fft):
            raise ValueError("win_length must satisfy 1 <= win_length <= n_fft")
        if window is not None and window not in {"hann", "hamming"}:
            raise ValueError("window must be hann or hamming")
        if center is not None and type(center) is not bool:
            raise TypeError("center must be a bool")
        if type(exclude_dc) is not bool:
            raise TypeError("exclude_dc must be a bool")
        if normalization not in {"none", "forgetting_laplace"}:
            raise ValueError("normalization must be none or forgetting_laplace")
        self.input_size, self.max_track, self.n_fft = input_size, max_track, n_fft
        self.stride, self.causal = stride, causal
        self.hop_length, self.win_length = hop_length, win_length
        self.window, self.center = window, center
        self.exclude_dc, self.normalization = exclude_dc, normalization
        self.num_freqs = n_fft // 2 + 1 - int(exclude_dc)
        directions = 1 if causal else 2
        self.temporal = nn.GRU(2 * input_size, hidden_size, batch_first=True, bidirectional=not causal)
        self.head = nn.Linear(directions * hidden_size, 2 * max_track * input_size)

    def _features(self, value: Tensor) -> Tensor:
        if value.ndim == 5:
            if value.shape[-1] != 2:
                raise ValueError("multichannel_stft_real_imag last dimension must be 2")
            # 接受 batch-first ``[B,T,F,C,2]``，以及历史 STFT ``[B,C,F,T,2]``。
            if value.shape[1] == self.input_size + 1 and value.shape[3] != self.input_size + 1:
                value = value.permute(0, 3, 2, 1, 4).contiguous()
            elif value.shape[3] == self.input_size + 1:
                value = value.permute(0, 3, 2, 1, 4).contiguous()
            elif value.shape[3] != self.input_size + 1:
                raise ValueError("multichannel STFT must contain C=input_size+1 microphones")
        elif value.ndim == 4:
            if value.shape[-1] != 2 * (self.input_size + 1):
                raise ValueError("multichannel STFT features must have last dimension 2*C")
            value = value.view(value.shape[0], value.shape[1], value.shape[2], self.input_size + 1, 2)
        else:
            raise ValueError("multichannel STFT input must have shape [B,T,F,2C] or [B,T,F,C,2]")
        if value.shape[2] != self.n_fft // 2 + 1 and value.shape[2] != self.num_freqs:
            raise ValueError(
                f"multichannel STFT frequency bins {value.shape[2]} != configured {self.n_fft // 2 + 1}"
            )
        if self.exclude_dc:
            if value.shape[2] == self.n_fft // 2 + 1:
                value = value[:, :, 1:]
        ref = value[..., :1, :]
        others = value[..., 1:, :]
        ref_complex = torch.view_as_complex(ref.contiguous())
        others_complex = torch.view_as_complex(others.contiguous())
        phase = others_complex * ref_complex.conj()
        phase = phase / phase.abs().clamp_min(torch.finfo(phase.real.dtype).eps)
        return torch.stack((phase.real, phase.imag), dim=-1).reshape(
            value.shape[0], value.shape[1], value.shape[2], -1
        )

    def forward(self, multichannel_stft_real_imag: Tensor | None = None, *, features: Tensor | None = None) -> Tensor:
        if multichannel_stft_real_imag is None:
            if features is None:
                raise TypeError("missing required input: multichannel_stft_real_imag")
            multichannel_stft_real_imag = features
        elif features is not None:
            raise ValueError("provide only multichannel_stft_real_imag or features")
        value = self._features(multichannel_stft_real_imag)
        b, t, f, _ = value.shape
        x = value.permute(0, 2, 1, 3).reshape(b * f, t, -1)
        x, _ = self.temporal(x)
        if self.stride > 1:
            x = x[:, :: self.stride]
        t_out = x.shape[1]
        x = self.head(x).view(b, f, t_out, self.max_track, self.input_size, 2)
        return x.permute(0, 2, 1, 3, 4, 5).contiguous()


class _FNBlock(nn.Module):
    """FN-SSL 的 full-band/narrow-band 融合块。

    该实现保留上游 ``FNblock`` 的两个时频扫描方向、skip connection 与
    双向/单向 narrow-band LSTM 语义。输入已经是单个 microphone pair 的
    ``[N,T,F,D]`` 特征；pair 维度由 ``FNSSL`` 在块外显式合并。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        dropout: float,
        causal: bool,
        is_first: bool,
    ) -> None:
        super().__init__()
        if hidden_size % 2:
            raise ValueError("FN hidden_size must be even")
        if type(is_first) is not bool or type(causal) is not bool:
            raise TypeError("FN is_first and causal must be bool")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("FN dropout must be in [0,1)")
        self.is_first = is_first
        self.causal = causal
        self.full_hidden_size = hidden_size // 2
        self.narrow_hidden_size = hidden_size if causal else hidden_size // 2
        self.dropout_full = nn.Dropout(float(dropout))
        self.dropout_narrow = nn.Dropout(float(dropout))
        self.full_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.full_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        narrow_input = 2 * self.full_hidden_size + (input_size if is_first else 0)
        self.narrow_lstm = nn.LSTM(
            input_size=narrow_input,
            hidden_size=self.narrow_hidden_size,
            batch_first=True,
            bidirectional=not causal,
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        full_skip: torch.Tensor | None = None,
        narrow_skip: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if value.ndim != 4:
            raise ValueError("FN block input must have shape [N,T,F,D]")
        batch, frames, frequencies, _ = value.shape
        original_narrow = value.permute(0, 2, 1, 3).reshape(
            batch * frequencies, frames, -1
        )
        full = value.reshape(batch * frames, frequencies, -1)
        if not self.is_first:
            if full_skip is None or narrow_skip is None:
                raise ValueError("non-first FN block requires both skip tensors")
            if full.shape != full_skip.shape:
                raise ValueError("FN full-band skip shape does not match block input")
            full = full + full_skip
        full, _ = self.full_lstm(full)
        full_skip = full
        full = self.dropout_full(full)
        narrow = full.reshape(batch, frames, frequencies, -1).permute(0, 2, 1, 3)
        narrow = narrow.reshape(batch * frequencies, frames, -1)
        if self.is_first:
            narrow = torch.cat((narrow, original_narrow), dim=-1)
        else:
            if narrow_skip is None or narrow.shape != narrow_skip.shape:
                raise ValueError("FN narrow-band skip shape does not match block input")
            narrow = narrow + narrow_skip
        narrow, _ = self.narrow_lstm(narrow)
        narrow_skip = narrow
        narrow = self.dropout_narrow(narrow)
        output = narrow.reshape(batch, frequencies, frames, -1).permute(0, 2, 1, 3)
        return output, full_skip, narrow_skip


class _OfficialFNSSL(nn.Module):
    """FN-SSL full/narrow LSTM backbone with an explicit pair axis.

    The upstream implementation receives pair-batched multichannel STFT real/imag
    features.  The framework keeps the microphone axis in the canonical input
    contract, so this adapter creates the same pair batches internally and restores
    ``[B,T,F,K,P,2]`` at the output boundary.
    """

    MODEL_TYPE = "FNSSL"
    MODEL_SPEC = ModelSpec(
        model_type=MODEL_TYPE,
        feature_contract="multichannel_stft_real_imag",
        target_contract="dpipd_pair",
        pair_layout="reference_to_all",
        geometry_mode="fixed_array",
        field_mode="far_field",
        reduction="average_pool_12",
        causal=False,
        mapping="far_field_grid",
        target_layout="trailing_ri",
    )
    IO_CONTRACT = _PairwiseIPDModel.IO_CONTRACT

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 256,
        max_track: int = 1,
        n_fft: int = 512,
        stride: int = 12,
        causal: bool = False,
        dropout: float = 0.2,
        reference_channel: int = 0,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str | None = None,
        center: bool | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise TypeError(f"unexpected model arguments: {sorted(kwargs)}")
        for name, value, minimum in (
            ("input_size", input_size, 1),
            ("hidden_size", hidden_size, 2),
            ("max_track", max_track, 1),
            ("n_fft", n_fft, 2),
            ("stride", stride, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an exact integer >= {minimum}")
        if hidden_size % 2:
            raise ValueError("hidden_size must be even for FN full/narrow fusion")
        if stride != 12:
            raise ValueError("FNSSL official average-pool time stride must be 12")
        if type(causal) is not bool or causal:
            raise ValueError("FNSSL official backbone is non-causal; set causal=false")
        if type(reference_channel) is not int or reference_channel < 0:
            raise ValueError("reference_channel must be a non-negative integer")
        if hop_length is not None and (type(hop_length) is not int or hop_length < 1):
            raise ValueError("hop_length must be a positive exact integer")
        if win_length is not None and (type(win_length) is not int or not 1 <= win_length <= n_fft):
            raise ValueError("win_length must satisfy 1 <= win_length <= n_fft")
        if window is not None and window not in {"hann", "hamming"}:
            raise ValueError("window must be hann or hamming")
        if center is not None and type(center) is not bool:
            raise TypeError("center must be a bool")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_track = max_track
        self.n_fft = n_fft
        self.stride = stride
        self.reference_channel = reference_channel
        self.hop_length, self.win_length = hop_length, win_length
        self.window, self.center = window, center
        self.block_1 = _FNBlock(
            4, hidden_size, dropout=dropout, causal=False, is_first=True
        )
        self.block_2 = _FNBlock(
            hidden_size, hidden_size, dropout=dropout, causal=False, is_first=False
        )
        self.block_3 = _FNBlock(
            hidden_size, hidden_size, dropout=dropout, causal=False, is_first=False
        )
        self.pooling = nn.AvgPool1d(kernel_size=stride, stride=stride)
        self.head = nn.Linear(hidden_size, 2 * max_track)
        self.activation = nn.Tanh()

    def _canonical_stft(self, value: Tensor) -> Tensor:
        if value.ndim == 4:
            if value.shape[-1] != 2 * (self.input_size + 1):
                raise ValueError("multichannel STFT features must have last dimension 2*C")
            value = value.reshape(
                value.shape[0], value.shape[1], value.shape[2], self.input_size + 1, 2
            )
        elif value.ndim == 5:
            if value.shape[-1] != 2:
                raise ValueError("multichannel STFT last dimension must be 2")
            if value.shape[1] == self.input_size + 1 and value.shape[3] != self.input_size + 1:
                value = value.permute(0, 3, 2, 1, 4).contiguous()
            elif value.shape[3] != self.input_size + 1:
                raise ValueError("multichannel STFT must contain C=input_size+1 microphones")
        else:
            raise ValueError("multichannel STFT input must have shape [B,T,F,2C] or [B,T,F,C,2]")
        if value.shape[3] != self.input_size + 1:
            raise ValueError("multichannel STFT microphone count must equal input_size+1")
        if value.shape[2] != self.n_fft // 2 + 1:
            raise ValueError(
                f"multichannel STFT frequency bins {value.shape[2]} != configured {self.n_fft // 2 + 1}"
            )
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("multichannel STFT real/imag input must be finite floating point")
        return value

    def _pair_batch(self, value: Tensor) -> Tensor:
        value = self._canonical_stft(value)
        channels = value.shape[3]
        if not 0 <= self.reference_channel < channels:
            raise ValueError("reference_channel is outside multichannel STFT")
        others = [index for index in range(channels) if index != self.reference_channel]
        pairs = []
        reference = value[..., self.reference_channel, :]
        for index in others:
            other = value[..., index, :]
            # Upstream FN-SSL concatenates real channels, then imag channels.
            pairs.append(torch.cat((reference[..., :1], other[..., :1], reference[..., 1:], other[..., 1:]), dim=-1))
        return torch.stack(pairs, dim=1).flatten(0, 1)

    def forward(self, multichannel_stft_real_imag: Tensor | None = None, *, features: Tensor | None = None) -> Tensor:
        if multichannel_stft_real_imag is None:
            if features is None:
                raise TypeError("missing required input: multichannel_stft_real_imag")
            multichannel_stft_real_imag = features
        elif features is not None:
            raise ValueError("provide only multichannel_stft_real_imag or features")
        pair_batch = self._pair_batch(multichannel_stft_real_imag)
        batch = multichannel_stft_real_imag.shape[0]
        pairs = self.input_size
        x, full_skip, narrow_skip = self.block_1(pair_batch)
        x, full_skip, narrow_skip = self.block_2(x, full_skip=full_skip, narrow_skip=narrow_skip)
        x, _, _ = self.block_3(x, full_skip=full_skip, narrow_skip=narrow_skip)
        pooled = self.pooling(x.permute(0, 2, 3, 1).reshape(-1, self.hidden_size, x.shape[1]))
        pooled = pooled.transpose(1, 2)
        pooled = self.activation(self.head(pooled))
        output_frames = pooled.shape[1]
        output = pooled.reshape(batch, pairs, x.shape[2], output_frames, self.max_track, 2)
        return output.permute(0, 3, 2, 4, 1, 5).contiguous()


@MODEL_REGISTRY.register("FNSSL")
class FNSSL(_OfficialFNSSL):
    """FN-SSL full/narrow LSTM implementation with canonical multi-channel input."""
