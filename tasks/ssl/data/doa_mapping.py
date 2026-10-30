"""把 DP-IPD 复数输出映射成 DOA 或 DOA×range。

映射层消费模型已经产生的 ``[B,T,F,K,P,2]`` DP-IPD，不重新执行 STFT、PIT
或声学仿真。候选模板只由 geometry、pair_index、采样率、声速和显式候选网格
生成。默认角度口径与 ``projection._doa`` 一致：``(polar, azimuth)``，其中
``polar`` 从 ``+Z`` 轴起算；需要水平面 elevation 时必须显式设置
``elevation_convention="horizontal"`` 和对应的 ``angle_order``。

远场模板使用
``tau=(mic_n-mic_m) dot direction / c``，复数模板为
``exp(phase_sign * j * 2*pi*f*tau)``。``phase_sign`` 默认为 ``+1``，与
``projection._direct_ipd_target`` 的 ``others * conj(ref)`` 相位约定绑定
（pair (ref, other) 的 target 相位 = +2πf(mic_other−mic_ref)·u/c）；若数据
投影采用相反的 reference-IPD 符号，必须在配置中显式改为 ``-1``，不能在映射层
静默翻转。该默认值由 tests/test_ssl_phase_sign_roundtrip.py 的解析双麦克风
round-trip 校准。

近场候选 range 以 canonical reference-to-all pair 的共同 reference 麦克风为
原点，``tau=(distance_m-distance_n)/c``；mapper 不消费 target-side 的
``source_distances``。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch
from torch import Tensor


_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


def _grid_tensor(value: Sequence[float] | Tensor, *, name: str) -> Tensor:
    """解析一维候选网格并拒绝隐式标量/空网格。"""
    result = torch.as_tensor(value, dtype=torch.float64)
    if result.ndim != 1 or result.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional grid")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result.contiguous()


def _check_grid_range(grid: Tensor, *, name: str, lower: float, upper: float) -> None:
    if bool((grid < lower).any().item()) or bool((grid > upper).any().item()):
        raise ValueError(f"{name} must be within [{lower}, {upper}] radians")


@dataclass(frozen=True)
class DOAMappingConfig:
    """DP-IPD→DOA 映射的显式配置。

    ``elevation_convention="polar"`` 与 SSL projection 当前的 canonical ``doa``
    标签一致；``horizontal`` 使用从水平面起算的 elevation。候选 grid 的顺序
    会保留在返回的 ``grid_shape`` 中，避免调用方按 tensor rank 猜测语义。
    """

    n_fft: int = 512
    sample_rate: int = 16000
    azimuth_grid: Sequence[float] | Tensor = (0.0,)
    elevation_grid: Sequence[float] | Tensor = (math.pi / 2.0,)
    range_grid: Sequence[float] | Tensor | None = None
    field_mode: str = "far_field"
    sound_speed: float = 343.0
    exclude_dc: bool = False
    # 与 projection 的 others*conj(ref) 相位约定绑定的模板符号，默认 +1。
    phase_sign: float = 1.0
    elevation_convention: str = "polar"
    angle_order: str = "elevation_azimuth"
    coordinate_frame: str = "room_xyz_m"
    vad_threshold: float = 0.5

    def __post_init__(self) -> None:
        if type(self.n_fft) is not int or self.n_fft < 2:
            raise ValueError("DOA mapping n_fft must be an exact integer >= 2")
        if type(self.sample_rate) is not int or self.sample_rate < 1000:
            raise ValueError("DOA mapping sample_rate must be an exact integer >= 1000")
        if self.field_mode not in {"far_field", "near_field"}:
            raise ValueError("DOA mapping field_mode must be far_field or near_field")
        if (
            isinstance(self.sound_speed, bool)
            or not isinstance(self.sound_speed, (int, float))
            or not math.isfinite(float(self.sound_speed))
            or float(self.sound_speed) <= 0.0
        ):
            raise ValueError("DOA mapping sound_speed must be positive and finite")
        if type(self.exclude_dc) is not bool:
            raise TypeError("DOA mapping exclude_dc must be a bool")
        if (
            isinstance(self.phase_sign, bool)
            or not isinstance(self.phase_sign, (int, float))
            or not math.isfinite(float(self.phase_sign))
            or float(self.phase_sign) not in {-1.0, 1.0}
        ):
            raise ValueError("DOA mapping phase_sign must be -1 or +1")
        if self.elevation_convention not in {"polar", "horizontal"}:
            raise ValueError("elevation_convention must be polar or horizontal")
        if self.angle_order not in {"elevation_azimuth", "azimuth_elevation"}:
            raise ValueError("angle_order must be elevation_azimuth or azimuth_elevation")
        if not isinstance(self.coordinate_frame, str) or not self.coordinate_frame.strip():
            raise ValueError("DOA mapping coordinate_frame must be a non-empty string")
        if (
            isinstance(self.vad_threshold, bool)
            or not isinstance(self.vad_threshold, (int, float))
            or not math.isfinite(float(self.vad_threshold))
            or not 0.0 <= float(self.vad_threshold) <= 1.0
        ):
            raise ValueError("DOA mapping vad_threshold must be in [0,1]")
        azimuth = _grid_tensor(self.azimuth_grid, name="azimuth_grid")
        elevation = _grid_tensor(self.elevation_grid, name="elevation_grid")
        _check_grid_range(azimuth, name="azimuth_grid", lower=-math.pi, upper=math.pi)
        if self.elevation_convention == "polar":
            _check_grid_range(elevation, name="elevation_grid", lower=0.0, upper=math.pi)
        else:
            _check_grid_range(elevation, name="elevation_grid", lower=-math.pi / 2.0, upper=math.pi / 2.0)
        if self.field_mode == "near_field":
            if self.range_grid is None:
                raise ValueError("near-field DOA mapping requires a positive range_grid")
            ranges = _grid_tensor(self.range_grid, name="range_grid")
            if bool((ranges <= 0.0).any().item()):
                raise ValueError("range_grid must contain strictly positive distances")
        elif self.range_grid is not None:
            raise ValueError("far-field DOA mapping must not provide range_grid")

    @property
    def frequencies(self) -> Tensor:
        """返回与 canonical one-sided STFT 对齐的频率轴。"""
        frequencies = torch.arange(self.n_fft // 2 + 1, dtype=torch.float64)
        frequencies = frequencies * (float(self.sample_rate) / float(self.n_fft))
        return frequencies[1:] if self.exclude_dc else frequencies


@dataclass(frozen=True)
class DOAMappingResult:
    """映射结果及其可审计的候选分数。"""

    doa: Tensor
    scores: Tensor
    valid_mask: Tensor
    selected_indices: Tensor
    candidate_angles: Tensor
    candidate_directions: Tensor
    grid_shape: tuple[int, ...]
    coordinate_frame: str
    candidate_ranges: Tensor | None = None
    candidate_positions: Tensor | None = None
    selected_ranges: Tensor | None = None
    selected_positions: Tensor | None = None

    @property
    def scores_grid(self) -> Tensor:
        """把最后的候选轴恢复成 ``[E,A]`` 或 ``[R,E,A]``。"""
        return self.scores.reshape(*self.scores.shape[:-1], *self.grid_shape)

    @property
    def selected_directions(self) -> Tensor:
        """返回选中候选的三维单位方向，形状为 ``[B,T,K,3]``。"""
        directions = self.candidate_directions[self.selected_indices]
        return torch.where(
            self.valid_mask.unsqueeze(-1), directions, torch.zeros_like(directions)
        )


def _candidate_grid(config: DOAMappingConfig) -> tuple[Tensor, Tensor, Tensor, tuple[int, ...]]:
    azimuth = _grid_tensor(config.azimuth_grid, name="azimuth_grid")
    elevation = _grid_tensor(config.elevation_grid, name="elevation_grid")
    elevation_mesh, azimuth_mesh = torch.meshgrid(elevation, azimuth, indexing="ij")
    if config.elevation_convention == "polar":
        direction = torch.stack(
            (
                elevation_mesh.sin() * azimuth_mesh.cos(),
                elevation_mesh.sin() * azimuth_mesh.sin(),
                elevation_mesh.cos(),
            ),
            dim=-1,
        )
    else:
        direction = torch.stack(
            (
                elevation_mesh.cos() * azimuth_mesh.cos(),
                elevation_mesh.cos() * azimuth_mesh.sin(),
                elevation_mesh.sin(),
            ),
            dim=-1,
        )
    base_angles = (
        torch.stack((elevation_mesh, azimuth_mesh), dim=-1)
        if config.angle_order == "elevation_azimuth"
        else torch.stack((azimuth_mesh, elevation_mesh), dim=-1)
    )
    base_directions = direction.reshape(-1, 3).to(torch.float32)
    base_angles = base_angles.reshape(-1, 2).to(torch.float32)
    if config.field_mode == "far_field":
        return base_angles, base_directions, torch.empty(0, dtype=torch.float32), (elevation.numel(), azimuth.numel())
    ranges = _grid_tensor(config.range_grid, name="range_grid")
    count = base_angles.shape[0]
    return (
        base_angles.repeat(ranges.numel(), 1),
        base_directions.repeat(ranges.numel(), 1),
        ranges.to(torch.float32).repeat_interleave(count),
        (ranges.numel(), elevation.numel(), azimuth.numel()),
    )


def _expand_geometry(geometry: Tensor, *, batch: int) -> Tensor:
    if geometry.ndim == 2:
        geometry = geometry.unsqueeze(0)
    if geometry.ndim != 3 or geometry.shape[-1] != 3:
        raise ValueError("DOA mapping geometry must have shape [B,C,3] or [C,3]")
    if geometry.shape[0] not in {1, batch}:
        raise ValueError("DOA mapping geometry batch dimension must be 1 or match prediction B")
    geometry = geometry.expand(batch, -1, -1).to(dtype=torch.float32)
    if geometry.shape[1] < 2 or not torch.isfinite(geometry).all():
        raise ValueError("DOA mapping geometry must contain at least two finite microphones")
    return geometry


def _expand_pairs(pair_index: Tensor, *, batch: int, pairs: int, channels: int) -> Tensor:
    if pair_index.ndim == 2:
        pair_index = pair_index.unsqueeze(0)
    if pair_index.ndim != 3 or pair_index.shape[-1] != 2:
        raise ValueError("DOA mapping pair_index must have shape [B,P,2] or [P,2]")
    if pair_index.shape[0] not in {1, batch}:
        raise ValueError("DOA mapping pair_index batch dimension must be 1 or match prediction B")
    if pair_index.shape[1] != pairs or pair_index.dtype not in _INTEGER_DTYPES:
        raise ValueError("DOA mapping pair_index must match prediction P and use integer dtype")
    pair_index = pair_index.expand(batch, -1, -1).to(dtype=torch.long)
    if bool((pair_index < 0).any().item()) or bool((pair_index >= channels).any().item()):
        raise ValueError("DOA mapping pair_index contains an out-of-range microphone")
    if bool((pair_index[..., 0] == pair_index[..., 1]).any().item()):
        raise ValueError("DOA mapping pair_index cannot contain self pairs")
    for batch_index in range(batch):
        if torch.unique(pair_index[batch_index], dim=0).shape[0] != pairs:
            raise ValueError("DOA mapping pair_index must contain unique microphone pairs")
    return pair_index


def _validate_pair_geometry(geometry: Tensor, pair_index: Tensor) -> None:
    """拒绝 pair 引用的重合麦克风，避免零基线模板退化。"""
    batch, pairs, _ = pair_index.shape
    mic_m = geometry.gather(1, pair_index[..., 0].unsqueeze(-1).expand(batch, pairs, 3))
    mic_n = geometry.gather(1, pair_index[..., 1].unsqueeze(-1).expand(batch, pairs, 3))
    baseline = torch.linalg.vector_norm(mic_n - mic_m, dim=-1)
    if not torch.isfinite(baseline).all() or bool((baseline == 0.0).any().item()):
        raise ValueError("DOA mapping pair geometry must have non-zero microphone baselines")


def _near_field_origin(geometry: Tensor, pair_index: Tensor) -> Tensor:
    """返回 canonical reference-to-all pair 共享的 reference microphone。"""
    reference_index = pair_index[..., 0]
    if not torch.equal(reference_index, reference_index[:, :1].expand_as(reference_index)):
        raise ValueError(
            "near-field DOA mapping requires reference_to_all pairs with one common reference microphone"
        )
    batch = geometry.shape[0]
    return geometry.gather(
        1, reference_index[:, :1].unsqueeze(-1).expand(batch, 1, geometry.shape[-1])
    ).squeeze(1)


def _expand_pair_mask(pair_mask: Tensor | None, *, batch: int, pairs: int, device: torch.device) -> Tensor:
    if pair_mask is None:
        result = torch.ones((batch, pairs), dtype=torch.bool, device=device)
    else:
        pair_mask = torch.as_tensor(pair_mask, device=device)
        if pair_mask.ndim == 1:
            pair_mask = pair_mask.unsqueeze(0)
        if pair_mask.ndim != 2 or pair_mask.shape[1] != pairs or pair_mask.shape[0] not in {1, batch}:
            raise ValueError("DOA mapping pair_mask must have shape [B,P] or [P]")
        if pair_mask.dtype is not torch.bool:
            raise TypeError("DOA mapping pair_mask must use bool dtype")
        result = pair_mask.expand(batch, -1).to(device=device)
    if not result.any(dim=1).all():
        raise ValueError("DOA mapping pair_mask must keep at least one pair per item")
    return result


def _length_mask(lengths: Tensor | Sequence[int] | None, *, batch: int, frames: int, device: torch.device) -> Tensor:
    if lengths is None:
        values = torch.full((batch,), frames, dtype=torch.int64, device=device)
    else:
        values = torch.as_tensor(lengths, device=device)
        if values.ndim != 1 or values.shape[0] != batch or values.dtype not in _INTEGER_DTYPES:
            raise ValueError("DOA mapping output_lengths must have integer shape [B]")
        values = values.to(torch.int64)
        if bool((values < 1).any().item()) or bool((values > frames).any().item()):
            raise ValueError("DOA mapping output_lengths must be within [1,T]")
    frame_ids = torch.arange(frames, device=device, dtype=torch.int64).view(1, frames)
    return frame_ids < values.view(batch, 1)


def _feature_weight(
    value: Tensor | None,
    *,
    name: str,
    batch: int,
    frames: int,
    frequencies: int,
    tracks: int,
    device: torch.device,
    maximum: float | None = None,
) -> Tensor:
    """把频率可靠性/coherence 统一成 ``[B,T,F,K,1]``。"""
    if value is None:
        return torch.ones((batch, frames, frequencies, tracks, 1), dtype=torch.float32, device=device)
    value = torch.as_tensor(value, device=device, dtype=torch.float32)
    shape = tuple(value.shape)
    if shape == (frequencies,):
        weight = value.reshape(1, 1, frequencies, 1, 1)
    elif shape == (batch, frequencies):
        weight = value.reshape(batch, 1, frequencies, 1, 1)
    elif shape == (batch, frames, frequencies):
        weight = value.reshape(batch, frames, frequencies, 1, 1)
    elif shape == (batch, frames, tracks, frequencies):
        weight = value.permute(0, 1, 3, 2).unsqueeze(-1)
    else:
        raise ValueError(
            f"DOA mapping {name} must have shape [F], [B,F], [B,T,F] or [B,T,K,F]"
        )
    if not torch.isfinite(weight).all() or bool((weight < 0).any().item()):
        raise ValueError(f"DOA mapping {name} must be finite and non-negative")
    if maximum is not None and bool((weight > maximum).any().item()):
        raise ValueError(f"DOA mapping {name} must be <= {maximum}")
    return weight.expand(batch, frames, frequencies, tracks, 1)


def _validate_optional_timebase(
    frame_origin: Tensor | None,
    *,
    batch: int,
    device: torch.device,
    time_stride: int,
) -> None:
    if frame_origin is not None:
        frame_origin = torch.as_tensor(frame_origin, device=device)
        if frame_origin.ndim != 1 or frame_origin.shape[0] != batch or frame_origin.dtype not in _INTEGER_DTYPES:
            raise ValueError("DOA mapping frame_origin must have integer shape [B]")
    if type(time_stride) is not int or time_stride < 1:
        raise ValueError("DOA mapping time_stride must be an exact positive integer")


class DPIPDToDOAMapper:
    """使用显式候选模板完成 far-field 或 near-field DP-IPD 映射。"""

    def __init__(self, config: DOAMappingConfig | None = None, **config_kwargs: Any):
        if config is not None and config_kwargs:
            raise ValueError("provide either DOAMappingConfig or mapping keyword arguments")
        self.config = config or DOAMappingConfig(**config_kwargs)

    def _templates(
        self,
        geometry: Tensor,
        pair_index: Tensor,
        candidate_directions: Tensor,
        candidate_ranges: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        batch, _, _ = geometry.shape
        pairs = pair_index.shape[1]
        mic_m = geometry.gather(1, pair_index[..., 0].unsqueeze(-1).expand(batch, pairs, 3))
        mic_n = geometry.gather(1, pair_index[..., 1].unsqueeze(-1).expand(batch, pairs, 3))
        if self.config.field_mode == "far_field":
            delays = torch.einsum(
                "bpk,qk->bpq", mic_n - mic_m, candidate_directions.to(geometry)
            ) / float(self.config.sound_speed)
            positions = None
        else:
            reference_origin = _near_field_origin(geometry, pair_index)
            positions = reference_origin[:, None, :] + (
                candidate_ranges.to(geometry).view(1, -1, 1)
                * candidate_directions.to(geometry).view(1, -1, 3)
            )
            distances = torch.linalg.vector_norm(
                positions[:, :, None, :] - geometry[:, None, :, :], dim=-1
            )
            distance_m = distances.gather(2, pair_index[..., 0].unsqueeze(1).expand(batch, distances.shape[1], pairs))
            distance_n = distances.gather(2, pair_index[..., 1].unsqueeze(1).expand(batch, distances.shape[1], pairs))
            # projection 的 target 是 X_other * conj(X_ref)。在 STFT 的
            # exp(-jωt) 约定下，其近场相位对应 -(distance_n-distance_m)。
            delays = (distance_m - distance_n).permute(0, 2, 1) / float(self.config.sound_speed)
        frequencies = self.config.frequencies.to(device=geometry.device, dtype=geometry.dtype)
        phase = float(self.config.phase_sign) * 2.0 * math.pi * frequencies[None, :, None, None] * delays[:, None, :, :]
        template = torch.polar(torch.ones_like(phase), phase)
        return template, positions

    def map(
        self,
        prediction: Tensor,
        *,
        geometry: Tensor,
        pair_index: Tensor,
        pair_mask: Tensor | None = None,
        vad: Tensor | None = None,
        output_lengths: Tensor | Sequence[int] | None = None,
        frequency_weights: Tensor | None = None,
        coherence: Tensor | None = None,
        pair_weights: Tensor | None = None,
        coordinate_frame: str | None = None,
        frame_origin: Tensor | None = None,
        time_stride: int = 1,
    ) -> DOAMappingResult:
        """映射预测并返回候选分数、选中方向和有效帧 mask。

        ``source_distances`` 没有出现在接口中：近场映射只能从显式 range grid
        估计距离，避免 ground-truth distance 泄漏到推理观测。
        """
        if coordinate_frame is None:
            coordinate_frame = self.config.coordinate_frame
        if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
            raise ValueError("DOA mapping coordinate_frame must be a non-empty string")
        if coordinate_frame != self.config.coordinate_frame:
            raise ValueError(
                "DOA mapping coordinate_frame does not match the configured geometry frame"
            )
        if (
            not isinstance(prediction, Tensor)
            or prediction.ndim != 6
            or prediction.shape[-1] != 2
        ):
            raise ValueError("DP-IPD prediction must have shape [B,T,F,K,P,2]")
        if any(int(size) < 1 for size in prediction.shape[:5]):
            raise ValueError("DP-IPD prediction must have positive B,T,F,K,P dimensions")
        if not torch.isfinite(prediction).all():
            raise ValueError("DP-IPD prediction must be finite")
        batch, frames, freq_count, tracks, pairs, _ = prediction.shape
        if tracks < 1 or pairs < 1:
            raise ValueError("DP-IPD prediction must contain at least one track and pair")
        if freq_count != self.config.frequencies.numel():
            raise ValueError(
                f"DP-IPD frequency bins {freq_count} != configured {self.config.frequencies.numel()}"
            )
        device = prediction.device
        geometry = _expand_geometry(torch.as_tensor(geometry, device=device), batch=batch)
        pair_index = _expand_pairs(
            torch.as_tensor(pair_index, device=device),
            batch=batch,
            pairs=pairs,
            channels=geometry.shape[1],
        )
        _validate_pair_geometry(geometry, pair_index)
        pair_mask = _expand_pair_mask(pair_mask, batch=batch, pairs=pairs, device=device)
        _validate_optional_timebase(frame_origin, batch=batch, device=device, time_stride=time_stride)
        length_mask = _length_mask(output_lengths, batch=batch, frames=frames, device=device)
        if vad is None:
            vad_value = torch.ones((batch, frames, tracks), dtype=torch.float32, device=device)
        else:
            vad_value = torch.as_tensor(vad, device=device, dtype=torch.float32)
            if vad_value.shape != (batch, frames, tracks):
                raise ValueError("DOA mapping vad must have shape [B,T,K]")
            if not torch.isfinite(vad_value).all() or bool(((vad_value < 0.0) | (vad_value > 1.0)).any().item()):
                raise ValueError("DOA mapping vad must be finite and in [0,1]")
        pair_weight = pair_mask.to(torch.float32)
        if pair_weights is not None:
            pair_weights_value = torch.as_tensor(pair_weights, device=device, dtype=torch.float32)
            if pair_weights_value.ndim == 1:
                pair_weights_value = pair_weights_value.unsqueeze(0)
            if pair_weights_value.shape not in {(1, pairs), (batch, pairs)}:
                raise ValueError("DOA mapping pair_weights must have shape [B,P] or [P]")
            if not torch.isfinite(pair_weights_value).all() or bool((pair_weights_value < 0).any().item()):
                raise ValueError("DOA mapping pair_weights must be finite and non-negative")
            pair_weight = pair_weight * pair_weights_value.expand(batch, pairs)
            if not pair_weight.any(dim=1).all():
                raise ValueError("DOA mapping pair_weights must keep at least one positive pair per item")
        feature_weight = _feature_weight(
            frequency_weights,
            name="frequency_weights",
            batch=batch,
            frames=frames,
            frequencies=freq_count,
            tracks=tracks,
            device=device,
        )
        feature_weight = feature_weight * _feature_weight(
            coherence,
            name="coherence",
            batch=batch,
            frames=frames,
            frequencies=freq_count,
            tracks=tracks,
            device=device,
            maximum=1.0,
        )
        candidate_angles, candidate_directions, candidate_ranges, grid_shape = _candidate_grid(self.config)
        candidate_angles = candidate_angles.to(device=device)
        candidate_directions = candidate_directions.to(device=device)
        candidate_ranges = candidate_ranges.to(device=device)
        template, candidate_positions = self._templates(
            geometry, pair_index, candidate_directions, candidate_ranges
        )
        value = prediction.to(dtype=torch.float32).contiguous()
        complex_prediction = torch.view_as_complex(value)
        magnitude = complex_prediction.abs()
        eps = torch.finfo(value.dtype).eps
        nonzero_prediction = magnitude > eps
        unit_prediction = torch.where(
            nonzero_prediction,
            complex_prediction / magnitude.clamp_min(eps),
            torch.zeros_like(complex_prediction),
        )
        correlation = (
            unit_prediction.conj().unsqueeze(-1) * template[:, None, :, None, :, :]
        ).real
        weights = (
            feature_weight
            * pair_weight[:, None, None, None, :]
            * vad_value[:, :, None, :, None]
            * length_mask[:, :, None, None, None].to(torch.float32)
            * nonzero_prediction.to(torch.float32)
        )
        numerator = (correlation * weights.unsqueeze(-1)).sum(dim=(2, 4))
        denominator = weights.sum(dim=(2, 4))
        active_observation = (vad_value > 0.0) & length_mask[:, :, None]
        if bool((active_observation & (denominator <= 0.0)).any().item()):
            raise ValueError(
                "DOA mapping has no positive prediction/feature weight on an active frame"
            )
        valid_vad = vad_value >= float(self.config.vad_threshold)
        valid = valid_vad & length_mask[:, :, None] & (denominator > 0.0)
        scores = torch.where(
            denominator.unsqueeze(-1) > 0.0,
            numerator / denominator.clamp_min(eps).unsqueeze(-1),
            torch.zeros_like(numerator),
        )
        selected_indices = scores.argmax(dim=-1)
        selected_angles = candidate_angles[selected_indices]
        selected_angles = torch.where(valid.unsqueeze(-1), selected_angles, torch.zeros_like(selected_angles))
        selected_ranges = None
        selected_positions = None
        if self.config.field_mode == "near_field":
            selected_ranges = candidate_ranges[selected_indices]
            selected_ranges = torch.where(valid, selected_ranges, torch.zeros_like(selected_ranges))
            assert candidate_positions is not None
            selected_positions = candidate_positions.gather(
                1, selected_indices.reshape(batch, -1).unsqueeze(-1).expand(-1, -1, 3)
            ).reshape(batch, frames, tracks, 3)
            selected_positions = torch.where(
                valid.unsqueeze(-1), selected_positions, torch.zeros_like(selected_positions)
            )
            candidate_positions = candidate_positions.to(torch.float32)
        return DOAMappingResult(
            doa=selected_angles,
            scores=scores,
            valid_mask=valid,
            selected_indices=selected_indices,
            candidate_angles=candidate_angles,
            candidate_directions=candidate_directions,
            grid_shape=grid_shape,
            coordinate_frame=coordinate_frame,
            candidate_ranges=None if self.config.field_mode == "far_field" else candidate_ranges,
            candidate_positions=candidate_positions,
            selected_ranges=selected_ranges,
            selected_positions=selected_positions,
        )

    __call__ = map


def map_dpipd_to_doa(
    prediction: Tensor,
    *,
    geometry: Tensor,
    pair_index: Tensor,
    config: DOAMappingConfig | None = None,
    **kwargs: Any,
) -> DOAMappingResult:
    """函数式 DP-IPD→DOA 入口；``kwargs`` 仅用于 mapper 配置。"""
    mapper_kwargs = {
        key: kwargs.pop(key)
        for key in tuple(kwargs)
        if key
        in {
            "n_fft",
            "sample_rate",
            "azimuth_grid",
            "elevation_grid",
            "range_grid",
            "field_mode",
            "sound_speed",
            "exclude_dc",
            "phase_sign",
            "elevation_convention",
            "angle_order",
            "coordinate_frame",
            "vad_threshold",
        }
    }
    if config is not None and mapper_kwargs:
        raise ValueError("provide either config or mapper configuration keywords")
    mapper = DPIPDToDOAMapper(config, **mapper_kwargs)
    return mapper.map(prediction, geometry=geometry, pair_index=pair_index, **kwargs)


__all__ = [
    "DOAMappingConfig",
    "DOAMappingResult",
    "DPIPDToDOAMapper",
    "map_dpipd_to_doa",
]
