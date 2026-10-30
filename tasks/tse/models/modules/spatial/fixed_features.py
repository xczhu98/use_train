# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
"""WeSep 固定空间特征的 runtime-spectrum 移植。

来源：WeSep commit ``dc11cb18`` 的
``wesep/modules/spatial/fixed_features.py``。本模块不持有 STFT；pair
``(first, second)`` 始终表示 ``phase(first) - phase(second)``，对应基线
``position[first] - position[second]``。
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Real

import torch
import torch.nn as nn


def as_complex_spectrum(value: torch.Tensor) -> torch.Tensor:
    """接受 complex ``[B,C,F,T]`` 或框架 RI ``[B,C,F,T,2]``。"""
    if not isinstance(value, torch.Tensor):
        raise TypeError("mixture spectrum must be a tensor")
    if value.is_complex():
        if value.ndim != 4:
            raise ValueError("complex mixture spectrum must have shape [B,C,F,T]")
        result = value
    else:
        if value.ndim != 5 or value.shape[-1] != 2 or not value.dtype.is_floating_point:
            raise ValueError("RI mixture spectrum must have shape [B,C,F,T,2]")
        result = torch.complex(value[..., 0], value[..., 1])
    if min(result.shape) < 1:
        raise ValueError("mixture spectrum axes must be non-empty")
    if not bool(torch.isfinite(result.real).all().item()) or not bool(
        torch.isfinite(result.imag).all().item()
    ):
        raise ValueError("mixture spectrum must contain only finite values")
    return result


def _resolve_pairs(
    pairs: Sequence[Sequence[int]], num_mics: int
) -> tuple[tuple[int, int], ...]:
    if isinstance(pairs, (str, bytes)):
        raise TypeError("pairs must be a sequence of microphone index pairs")
    resolved: list[tuple[int, int]] = []
    for pair_index, pair in enumerate(pairs):
        values = tuple(pair)
        if len(values) != 2 or any(type(index) is not int for index in values):
            raise ValueError(f"pairs[{pair_index}] must contain two exact integer indices")
        first, second = values
        if first == second or not 0 <= first < num_mics or not 0 <= second < num_mics:
            raise ValueError(f"invalid microphone pair {values} for C={num_mics}")
        if values in resolved:
            raise ValueError(f"duplicate microphone pair {values}")
        resolved.append(values)
    if not resolved:
        raise ValueError("at least one microphone pair is required")
    return tuple(resolved)


def _positive_real(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


class SpatialFeature(nn.Module):
    """多通道谱与 microphone pair 的公共校验。"""

    def __init__(self, pairs: Sequence[Sequence[int]], num_mics: int):
        super().__init__()
        if type(num_mics) is not int or num_mics < 2:
            raise ValueError("spatial pair features require num_mics >= 2")
        resolved = _resolve_pairs(pairs, num_mics)
        self.num_mics = num_mics
        self.register_buffer("pairs", torch.tensor(resolved, dtype=torch.long))

    def _spectrum(self, mix_spectrum: torch.Tensor) -> torch.Tensor:
        spectrum = as_complex_spectrum(mix_spectrum)
        if spectrum.shape[1] != self.num_mics:
            raise ValueError(
                f"mixture spectrum has C={spectrum.shape[1]}, expected {self.num_mics}"
            )
        return spectrum

    def _ipd_from_complex(self, spectrum: torch.Tensor) -> torch.Tensor:
        first = spectrum[:, self.pairs[:, 0]]
        second = spectrum[:, self.pairs[:, 1]]
        phase = torch.angle(first) - torch.angle(second)
        return torch.remainder(phase + math.pi, 2 * math.pi) - math.pi


class DirectionalPhaseFeature(SpatialFeature):
    """使用米制阵列与实际声速计算目标 pair phase。"""

    def __init__(
        self,
        *,
        pairs: Sequence[Sequence[int]],
        mic_positions_m: Sequence[Sequence[Real]],
        sample_rate: int,
        n_fft: int,
        speed_of_sound_mps: Real,
    ):
        positions = torch.as_tensor(mic_positions_m, dtype=torch.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
            raise ValueError("mic_positions_m must have shape [C,3] in metres")
        if not bool(torch.isfinite(positions).all().item()):
            raise ValueError("mic_positions_m must contain finite metre coordinates")
        super().__init__(pairs, int(positions.shape[0]))
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive exact int")
        if type(n_fft) is not int or n_fft <= 0:
            raise ValueError("n_fft must be a positive exact int")
        speed = _positive_real(speed_of_sound_mps, "speed_of_sound_mps")
        baselines = positions[self.pairs[:, 0]] - positions[self.pairs[:, 1]]
        frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate, dtype=torch.float64)
        self.register_buffer("baselines_m", baselines.to(torch.float32))
        self.register_buffer(
            "angular_frequencies", (2.0 * math.pi * frequencies).to(torch.float32)
        )
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.speed_of_sound_mps = speed

    def _target_phase(self, direction_rad: torch.Tensor, frequency_bins: int) -> torch.Tensor:
        if (
            not isinstance(direction_rad, torch.Tensor)
            or direction_rad.ndim != 3
            or direction_rad.shape[1] != 2
        ):
            raise ValueError("aligned direction must have shape [B,2,T]")
        if not direction_rad.dtype.is_floating_point:
            raise TypeError("aligned direction must use a floating-point dtype")
        if frequency_bins != self.angular_frequencies.numel():
            raise ValueError(
                f"spectrum has F={frequency_bins}, expected n_fft//2+1="
                f"{self.angular_frequencies.numel()}"
            )
        azimuth = direction_rad[:, 0]
        elevation = direction_rad[:, 1]
        # 方向角 → 单位向量：azimuth=atan2(y, x)、elevation 从水平面起算，
        # 与 tasks/tse/data/spatial_cues.py::azimuth_elevation_from_positions 同源。
        unit_direction = torch.stack(
            [
                torch.cos(elevation) * torch.cos(azimuth),
                torch.cos(elevation) * torch.sin(azimuth),
                torch.sin(elevation),
            ],
            dim=-1,
        )
        baselines = self.baselines_m.to(dtype=direction_rad.dtype)
        delays_m = torch.einsum("btc,pc->bpt", unit_direction, baselines)
        angular = self.angular_frequencies.to(dtype=direction_rad.dtype)
        return angular.view(1, 1, -1, 1) * delays_m.unsqueeze(2) / self.speed_of_sound_mps

    @staticmethod
    def _present(value: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        if value is None:
            return torch.ones(batch, dtype=torch.bool, device=device)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch,):
            raise ValueError(f"present must have shape [{batch}]")
        if value.dtype != torch.bool or value.device != device:
            raise TypeError("present must be torch.bool on the spectrum device")
        return value


class IPDFeature(SpatialFeature):
    """观测通道间相位差，输出 ``[B,P,F,T]``。"""

    def compute(self, mix_spectrum: torch.Tensor) -> torch.Tensor:
        return self._ipd_from_complex(self._spectrum(mix_spectrum))


class CDFSpatialFeature(DirectionalPhaseFeature):
    """观测 IPD 与目标相位差的一致性余弦，输出 ``[B,P,F,T]``。"""

    def compute(
        self,
        mix_spectrum: torch.Tensor,
        direction_rad: torch.Tensor | None,
        present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        spectrum = self._spectrum(mix_spectrum)
        mask = self._present(present, spectrum.shape[0], spectrum.device)
        output = spectrum.real.new_zeros(
            spectrum.shape[0], self.pairs.shape[0], spectrum.shape[2], spectrum.shape[3]
        )
        if not bool(mask.any().item()):
            return output
        if direction_rad is None:
            raise ValueError("CDF requires aligned target directions for present rows")
        if direction_rad.device != spectrum.device or direction_rad.shape[0] != spectrum.shape[0]:
            raise ValueError("direction and spectrum must share batch axis and device")
        ipd = self._ipd_from_complex(spectrum[mask])
        target = self._target_phase(direction_rad[mask], spectrum.shape[2])
        output[mask] = torch.cos(ipd - target)
        return output


class SDFSpatialFeature(DirectionalPhaseFeature):
    """观测 IPD 与目标相位差的正弦残差，输出 ``[B,P,F,T]``。"""

    def compute(
        self,
        mix_spectrum: torch.Tensor,
        direction_rad: torch.Tensor | None,
        present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        spectrum = self._spectrum(mix_spectrum)
        mask = self._present(present, spectrum.shape[0], spectrum.device)
        output = spectrum.real.new_zeros(
            spectrum.shape[0], self.pairs.shape[0], spectrum.shape[2], spectrum.shape[3]
        )
        if not bool(mask.any().item()):
            return output
        if direction_rad is None:
            raise ValueError("SDF requires aligned target directions for present rows")
        if direction_rad.device != spectrum.device or direction_rad.shape[0] != spectrum.shape[0]:
            raise ValueError("direction and spectrum must share batch axis and device")
        ipd = self._ipd_from_complex(spectrum[mask])
        target = self._target_phase(direction_rad[mask], spectrum.shape[2])
        output[mask] = torch.sin(ipd - target)
        return output


class DeltaSTFTFeature(SpatialFeature):
    """``X[first]-X[second]`` 的 real/imag，输出 ``[B,2P,F,T]``。"""

    def compute(self, mix_spectrum: torch.Tensor) -> torch.Tensor:
        spectrum = self._spectrum(mix_spectrum)
        delta = spectrum[:, self.pairs[:, 0]] - spectrum[:, self.pairs[:, 1]]
        return torch.cat([delta.real, delta.imag], dim=1)


__all__ = [
    "CDFSpatialFeature",
    "DeltaSTFTFeature",
    "DirectionalPhaseFeature",
    "IPDFeature",
    "SDFSpatialFeature",
    "SpatialFeature",
    "as_complex_spectrum",
]
