# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
"""WeSep spatial frontend 的 canonical TSE runtime-spectrum 适配。

来源：WeSep commit ``dc11cb18`` 的
``wesep/modules/spatial/spatial_frontend.py``。本前端只消费外部 STFT，不创建
window、不执行 STFT/iSTFT，也不从 track 点数推断时间位置。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import torch
import torch.nn as nn

from tasks.tse.data.spatial_cues import align_spatial_cue_to_frames
from tasks.tse.models.modules.spatial.doa_features import (
    CyclicDOAFeature,
    InitStateDOAFeature,
)
from tasks.tse.models.modules.spatial.fixed_features import (
    CDFSpatialFeature,
    DeltaSTFTFeature,
    IPDFeature,
    SDFSpatialFeature,
    as_complex_spectrum,
)


_FEATURE_ORDER = (
    "ipd",
    "cdf",
    "sdf",
    "delta_stft",
    "cyc_doaemb",
    "initstate_emb",
)
_FIXED_FEATURES = frozenset({"ipd", "cdf", "sdf", "delta_stft"})


InitialState = dict[str, tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class SpatialFrontendOutput:
    """空间前端输出，不负责与原始 RI 谱或 separator latent 融合。

    ``spectral`` 为 ``[B,S,F,T]``；``cyclic_embedding`` 为可选
    ``[B,1,D,T]``；``initial_states`` 每个 block 含四个 ``[B,2D]``
    tensor，其中 ``D=mix_dim``。缺方向行的方向依赖输出均为零。
    """

    spectral: torch.Tensor
    spectral_by_name: Mapping[str, torch.Tensor]
    cyclic_embedding: torch.Tensor | None
    cyclic_fusion: str | None
    initial_states: tuple[InitialState, ...] | None
    aligned_direction_rad: torch.Tensor
    present: torch.Tensor


class SpatialFrontend(nn.Module):
    """构造 IPD/CDF/SDF/Delta、cyclic embedding 与 BSRNN 初始状态。

    Args:
        representation: canonical ``static`` 或 ``track``。
        features: 来自 ``CueSpec.options["features"]`` 的 feature map。
        pairs: 有序 pair ``(first, second)``，符号不做排序或归一化。
        mic_positions_m: CDF/SDF 所需的同一坐标系米制坐标 ``[C,3]``。
        speed_of_sound_mps: CDF/SDF 必须显式传入的场景实际声速。
    """

    def __init__(
        self,
        *,
        representation: Literal["static", "track"],
        features: Mapping[str, Mapping[str, object]],
        pairs: Sequence[Sequence[int]],
        num_mics: int,
        sample_rate: int,
        n_fft: int,
        mic_positions_m: Sequence[Sequence[Real]] | None = None,
        speed_of_sound_mps: Real | None = None,
        mix_dim: int = 128,
        num_layers: int = 1,
        track_alignment: Literal["renderer_segment"] = "renderer_segment",
    ):
        super().__init__()
        if representation not in {"static", "track"}:
            raise ValueError("representation must be 'static' or 'track'")
        if track_alignment != "renderer_segment":
            raise ValueError(
                "only track_alignment='renderer_segment' is supported; "
                "point-count resizing is forbidden"
            )
        if not isinstance(features, Mapping) or not features:
            raise ValueError("features must be a non-empty mapping")
        if any(not isinstance(name, str) for name in features):
            raise TypeError("feature names must be strings")
        unknown = sorted(set(features) - set(_FEATURE_ORDER))
        if unknown:
            raise ValueError(f"spatial frontend contains unknown features: {unknown}")
        for name, config in features.items():
            if not isinstance(config, Mapping):
                raise TypeError(f"features.{name} must be a mapping")
            if name in _FIXED_FEATURES and config:
                raise ValueError(f"features.{name} does not accept options: {sorted(config)}")
        for name, value in (
            ("num_mics", num_mics),
            ("sample_rate", sample_rate),
            ("n_fft", n_fft),
            ("mix_dim", mix_dim),
            ("num_layers", num_layers),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact int")
        if num_mics < 2:
            raise ValueError("spatial frontend requires num_mics >= 2")

        self.representation = representation
        self.track_alignment = track_alignment
        self.num_mics = num_mics
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.mix_dim = mix_dim
        self.num_layers = num_layers
        self.feature_names = tuple(name for name in _FEATURE_ORDER if name in features)
        self.spectral_feature_names = tuple(
            name for name in self.feature_names if name in _FIXED_FEATURES
        )

        if isinstance(pairs, (str, bytes)):
            raise TypeError("pairs must be a sequence of microphone index pairs")
        resolved_pairs: list[tuple[int, int]] = []
        for pair_index, pair in enumerate(pairs):
            values = tuple(pair)
            if len(values) != 2 or any(type(index) is not int for index in values):
                raise ValueError(
                    f"pairs[{pair_index}] must contain two exact integer indices"
                )
            first, second = values
            if first == second or not 0 <= first < num_mics or not 0 <= second < num_mics:
                raise ValueError(f"invalid microphone pair {values} for C={num_mics}")
            if values in resolved_pairs:
                raise ValueError(f"duplicate microphone pair {values}")
            resolved_pairs.append(values)
        pairs = tuple(resolved_pairs)
        pair_count = len(pairs)
        if pair_count < 1:
            raise ValueError("pairs must contain at least one microphone pair")

        directional_args = None
        if {"cdf", "sdf"}.intersection(features):
            if mic_positions_m is None:
                raise ValueError("CDF/SDF require mic_positions_m in metres")
            if speed_of_sound_mps is None:
                raise ValueError("CDF/SDF require the actual speed_of_sound_mps")
            directional_args = {
                "pairs": pairs,
                "mic_positions_m": mic_positions_m,
                "sample_rate": sample_rate,
                "n_fft": n_fft,
                "speed_of_sound_mps": speed_of_sound_mps,
            }

        self.pairs = pairs
        self.spectral_channels = 0
        if "ipd" in features:
            self.ipd = IPDFeature(pairs, num_mics)
            self.spectral_channels += pair_count
        if "cdf" in features:
            assert directional_args is not None
            self.cdf = CDFSpatialFeature(**directional_args)
            self.spectral_channels += pair_count
        if "sdf" in features:
            assert directional_args is not None
            self.sdf = SDFSpatialFeature(**directional_args)
            self.spectral_channels += pair_count
        if "delta_stft" in features:
            self.delta_stft = DeltaSTFTFeature(pairs, num_mics)
            self.spectral_channels += pair_count * 2
        if "cyc_doaemb" in features:
            self.cyc_doaemb = CyclicDOAFeature(
                features["cyc_doaemb"], mix_dim=mix_dim
            )
        if "initstate_emb" in features:
            self.initstate_emb = InitStateDOAFeature(
                features["initstate_emb"], mix_dim=mix_dim, num_layers=num_layers
            )

    def compute(
        self,
        mix_spectrum: torch.Tensor,
        *,
        spatial_cue: torch.Tensor | None,
        present: torch.Tensor,
        frame_sample_indices: torch.Tensor | None = None,
        sample_anchors: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> SpatialFrontendOutput:
        """计算空间特征。

        ``mix_spectrum`` 接受 complex ``[B,C,F,T]`` 或 RI
        ``[B,C,F,T,2]``。track 必须提供相对当前 mixture 起点的
        ``frame_sample_indices[T]``、``sample_anchors[B,P]`` 与
        ``lengths[B]``；static 不接受 anchor/length。
        """
        spectrum = as_complex_spectrum(mix_spectrum)
        batch, channels, frequency_bins, frames = spectrum.shape
        if channels != self.num_mics:
            raise ValueError(f"mixture has C={channels}, expected {self.num_mics}")
        if not isinstance(present, torch.Tensor) or tuple(present.shape) != (batch,):
            raise ValueError(f"present must have shape [{batch}]")
        if present.dtype != torch.bool or present.device != spectrum.device:
            raise TypeError("present must be torch.bool on the spectrum device")
        if frame_sample_indices is None:
            if self.representation == "track":
                raise ValueError("track representation requires explicit frame_sample_indices")
            frame_sample_indices = torch.arange(frames, device=spectrum.device)
        elif frame_sample_indices.device != spectrum.device:
            raise ValueError("frame_sample_indices must be on the spectrum device")
        if frame_sample_indices.numel() != frames:
            raise ValueError(
                f"frame_sample_indices has T={frame_sample_indices.numel()}, expected {frames}"
            )
        if spatial_cue is not None and spatial_cue.device != spectrum.device:
            raise ValueError("spatial_cue must be on the spectrum device")

        direction = align_spatial_cue_to_frames(
            spatial_cue,
            representation=self.representation,
            present=present,
            frame_sample_indices=frame_sample_indices,
            sample_anchors=sample_anchors,
            lengths=lengths,
            track_alignment=self.track_alignment,
        ).to(dtype=spectrum.real.dtype)

        by_name: dict[str, torch.Tensor] = {}
        if hasattr(self, "ipd"):
            by_name["ipd"] = self.ipd.compute(spectrum)
        if hasattr(self, "cdf"):
            by_name["cdf"] = self.cdf.compute(spectrum, direction, present)
        if hasattr(self, "sdf"):
            by_name["sdf"] = self.sdf.compute(spectrum, direction, present)
        if hasattr(self, "delta_stft"):
            by_name["delta_stft"] = self.delta_stft.compute(spectrum)
        spectral = (
            torch.cat([by_name[name] for name in self.spectral_feature_names], dim=1)
            if self.spectral_feature_names
            else spectrum.real.new_zeros((batch, 0, frequency_bins, frames))
        )
        if spectral.shape[1] != self.spectral_channels:
            raise RuntimeError("spatial spectral channel accounting mismatch")

        cyclic = None
        cyclic_fusion = None
        if hasattr(self, "cyc_doaemb"):
            cyclic = self.cyc_doaemb.compute(direction, present)
            cyclic_fusion = self.cyc_doaemb.fusion
        initial_states = None
        if hasattr(self, "initstate_emb"):
            initial_states = self.initstate_emb.compute(direction, present)
        return SpatialFrontendOutput(
            spectral=spectral,
            spectral_by_name=by_name,
            cyclic_embedding=cyclic,
            cyclic_fusion=cyclic_fusion,
            initial_states=initial_states,
            aligned_direction_rad=direction,
            present=present,
        )

    def forward(
        self,
        mix_spectrum: torch.Tensor,
        *,
        spatial_cue: torch.Tensor | None,
        present: torch.Tensor,
        frame_sample_indices: torch.Tensor | None = None,
        sample_anchors: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> SpatialFrontendOutput:
        return self.compute(
            mix_spectrum,
            spatial_cue=spatial_cue,
            present=present,
            frame_sample_indices=frame_sample_indices,
            sample_anchors=sample_anchors,
            lengths=lengths,
        )


__all__ = ["InitialState", "SpatialFrontend", "SpatialFrontendOutput"]
