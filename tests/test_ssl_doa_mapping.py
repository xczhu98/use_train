"""DP-IPD 到 DOA/距离映射的解析合同测试。"""

from __future__ import annotations

import math

import pytest
import torch

from tasks.ssl.data.doa_mapping import (
    DOAMappingConfig,
    DPIPDToDOAMapper,
)


pytestmark = pytest.mark.unit


def _direction(*, polar: float, azimuth: float) -> torch.Tensor:
    return torch.tensor(
        [
            math.sin(polar) * math.cos(azimuth),
            math.sin(polar) * math.sin(azimuth),
            math.cos(polar),
        ],
        dtype=torch.float32,
    )


def _far_prediction(
    config: DOAMappingConfig,
    geometry: torch.Tensor,
    pair_index: torch.Tensor,
    *,
    polar: float,
    azimuth: float,
    frames: int = 2,
    tracks: int = 1,
    amplitude: torch.Tensor | None = None,
) -> torch.Tensor:
    direction = _direction(polar=polar, azimuth=azimuth)
    frequencies = config.frequencies.to(torch.float32)
    values = []
    for first, second in pair_index.tolist():
        delay = torch.dot(geometry[second] - geometry[first], direction) / config.sound_speed
        phase = config.phase_sign * 2.0 * math.pi * frequencies * delay
        values.append(torch.polar(torch.ones_like(phase), phase))
    complex_value = torch.stack(values, dim=-1)
    if amplitude is not None:
        complex_value = complex_value * amplitude.view(-1, 1)
    return torch.view_as_real(complex_value).view(
        1, 1, frequencies.numel(), 1, pair_index.shape[0], 2
    ).expand(1, frames, -1, tracks, -1, -1).contiguous()


def _near_prediction(
    config: DOAMappingConfig,
    geometry: torch.Tensor,
    pair_index: torch.Tensor,
    *,
    polar: float,
    azimuth: float,
    source_range: float,
    frames: int = 2,
) -> torch.Tensor:
    direction = _direction(polar=polar, azimuth=azimuth)
    source = geometry[pair_index[0, 0]] + source_range * direction
    distances = torch.linalg.vector_norm(source[None, :] - geometry, dim=-1)
    frequencies = config.frequencies.to(torch.float32)
    values = []
    for first, second in pair_index.tolist():
        # projection 的 target 是 X_other * conj(X_ref)，对应负的传播距离差。
        delay = (distances[first] - distances[second]) / config.sound_speed
        phase = config.phase_sign * 2.0 * math.pi * frequencies * delay
        values.append(torch.polar(torch.ones_like(phase), phase))
    complex_value = torch.stack(values, dim=-1)
    return torch.view_as_real(complex_value).view(
        1, 1, frequencies.numel(), 1, pair_index.shape[0], 2
    ).expand(1, frames, -1, -1, -1, -1).contiguous()


def test_far_field_mapping_recovers_direction_after_unit_circle_normalization():
    geometry = torch.tensor(
        [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.015, 0.07, 0.0]],
        dtype=torch.float32,
    )
    pair_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64)
    config = DOAMappingConfig(
        n_fft=32,
        sample_rate=16000,
        azimuth_grid=(-math.pi / 2.0, 0.0, math.pi / 2.0),
        elevation_grid=(math.pi / 2.0,),
        phase_sign=-1.0,
    )
    prediction = _far_prediction(
        config,
        geometry,
        pair_index,
        polar=math.pi / 2.0,
        azimuth=math.pi / 2.0,
        amplitude=torch.linspace(0.1, 2.0, config.frequencies.numel()),
    )
    result = DPIPDToDOAMapper(config)(
        prediction,
        geometry=geometry,
        pair_index=pair_index,
        pair_mask=torch.ones(2, dtype=torch.bool),
        vad=torch.ones(1, 2, 1),
    )
    assert result.grid_shape == (1, 3)
    assert result.selected_indices.tolist() == [[[2], [2]]]
    torch.testing.assert_close(
        result.doa[..., 1], torch.full((1, 2, 1), math.pi / 2.0), atol=1e-5, rtol=0.0
    )
    assert result.valid_mask.all()
    assert torch.isfinite(result.scores).all()


def test_mapping_supports_per_batch_geometry_and_pair_permutation():
    """每个 batch 的 geometry/pair_index 都必须参与各自模板生成。"""
    config = DOAMappingConfig(
        n_fft=32,
        sample_rate=16000,
        azimuth_grid=(-math.pi / 2.0, -math.pi / 4.0, 0.0, math.pi / 4.0, math.pi / 2.0),
        elevation_grid=(math.pi / 2.0,),
    )
    geometry = torch.stack(
        (
            torch.tensor(
                [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.015, 0.07, 0.0]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.10, -0.20, 0.30], [0.18, -0.20, 0.30], [0.115, -0.13, 0.30]],
                dtype=torch.float32,
            ),
        )
    )
    pair_index = torch.stack(
        (
            torch.tensor([[0, 1], [0, 2]], dtype=torch.int64),
            torch.tensor([[0, 2], [0, 1]], dtype=torch.int64),
        )
    )
    prediction = torch.cat(
        (
            _far_prediction(
                config,
                geometry[0],
                pair_index[0],
                polar=math.pi / 2.0,
                azimuth=math.pi / 4.0,
            ),
            _far_prediction(
                config,
                geometry[1],
                pair_index[1],
                polar=math.pi / 2.0,
                azimuth=-math.pi / 4.0,
            ),
        ),
        dim=0,
    )
    result = DPIPDToDOAMapper(config)(
        prediction,
        geometry=geometry,
        pair_index=pair_index,
        pair_mask=torch.ones(2, 2, dtype=torch.bool),
        vad=torch.ones(2, 2, 1),
    )
    assert result.valid_mask.all()
    expected = torch.tensor(
        [math.pi / 4.0, -math.pi / 4.0], dtype=torch.float32
    ).view(2, 1, 1).expand(2, 2, 1)
    torch.testing.assert_close(result.doa[..., 1], expected, atol=1e-5, rtol=0.0)


def test_mapping_horizontal_elevation_and_angle_order_are_explicit():
    """horizontal elevation 必须按显式 angle_order 返回，而非沿用 polar 标签。"""
    geometry = torch.tensor(
        [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.015, 0.07, 0.0]],
        dtype=torch.float32,
    )
    pair_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64)
    azimuth = math.pi / 4.0
    horizontal_elevation = math.pi / 6.0
    config = DOAMappingConfig(
        n_fft=32,
        sample_rate=16000,
        azimuth_grid=(0.0, math.pi / 4.0, math.pi / 2.0),
        elevation_grid=(0.0, math.pi / 6.0, math.pi / 3.0),
        elevation_convention="horizontal",
        angle_order="azimuth_elevation",
    )
    prediction = _far_prediction(
        config,
        geometry,
        pair_index,
        polar=math.pi / 2.0 - horizontal_elevation,
        azimuth=azimuth,
    )
    result = DPIPDToDOAMapper(config)(
        prediction,
        geometry=geometry,
        pair_index=pair_index,
        vad=torch.ones(1, 2, 1),
    )
    assert result.valid_mask.all()
    expected = torch.tensor([azimuth, horizontal_elevation], dtype=torch.float32)
    torch.testing.assert_close(
        result.doa,
        expected.view(1, 1, 1, 2).expand_as(result.doa),
        atol=1e-5,
        rtol=0.0,
    )


def test_mapping_handles_azimuth_wrap_boundary_with_unit_direction():
    """azimuth 的 -π/π 等价边界必须保持有限且方向一致。"""
    geometry = torch.tensor(
        [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.015, 0.07, 0.0]],
        dtype=torch.float32,
    )
    pair_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64)
    config = DOAMappingConfig(
        n_fft=32,
        sample_rate=16000,
        azimuth_grid=(-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi),
        elevation_grid=(math.pi / 2.0,),
    )
    prediction = _far_prediction(
        config,
        geometry,
        pair_index,
        polar=math.pi / 2.0,
        azimuth=math.pi,
    )
    result = DPIPDToDOAMapper(config)(
        prediction,
        geometry=geometry,
        pair_index=pair_index,
        vad=torch.ones(1, 2, 1),
    )
    assert result.valid_mask.all()
    assert torch.isfinite(result.doa).all()
    torch.testing.assert_close(
        result.selected_directions,
        torch.tensor([-1.0, 0.0, 0.0]).view(1, 1, 1, 3).expand(1, 2, 1, 3),
        atol=1e-5,
        rtol=0.0,
    )


def test_pair_mask_frequency_weight_vad_and_lengths_gate_mapping():
    geometry = torch.tensor(
        [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.015, 0.07, 0.0]],
        dtype=torch.float32,
    )
    pair_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64)
    config = DOAMappingConfig(
        n_fft=16,
        sample_rate=16000,
        azimuth_grid=(-math.pi / 2.0, 0.0, math.pi / 2.0),
        elevation_grid=(math.pi / 2.0,),
    )
    good = _far_prediction(
        config, geometry, pair_index[:1], polar=math.pi / 2.0, azimuth=0.0, frames=2
    )
    wrong = _far_prediction(
        config, geometry, pair_index[1:], polar=math.pi / 2.0, azimuth=math.pi / 2.0, frames=2
    )
    prediction = torch.cat((good, wrong), dim=4)
    result = DPIPDToDOAMapper(config)(
        prediction,
        geometry=geometry,
        pair_index=pair_index,
        pair_mask=torch.tensor([True, False]),
        vad=torch.tensor([[[1.0], [0.0]]]),
        output_lengths=torch.tensor([1], dtype=torch.int64),
        frequency_weights=torch.ones(config.frequencies.numel()),
    )
    assert result.selected_indices[0, 0, 0].item() == 1
    assert result.valid_mask.tolist() == [[[True], [False]]]
    torch.testing.assert_close(result.doa[0, 1, 0], torch.zeros(2))


def test_near_field_mapping_estimates_range_without_ground_truth_distance_input():
    geometry = torch.tensor(
        [[-0.04, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.075, 0.0]],
        dtype=torch.float32,
    )
    pair_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64)
    config = DOAMappingConfig(
        n_fft=64,
        sample_rate=16000,
        azimuth_grid=(0.0, math.pi / 4.0, math.pi / 2.0),
        elevation_grid=(math.pi / 2.0,),
        range_grid=(0.5, 1.0, 1.5),
        field_mode="near_field",
    )
    prediction = _near_prediction(
        config,
        geometry,
        pair_index,
        polar=math.pi / 2.0,
        azimuth=math.pi / 4.0,
        source_range=1.0,
    )
    result = DPIPDToDOAMapper(config)(
        prediction,
        geometry=geometry,
        pair_index=pair_index,
        vad=torch.ones(1, 2, 1),
    )
    assert result.grid_shape == (3, 1, 3)
    torch.testing.assert_close(result.selected_ranges, torch.full((1, 2, 1), 1.0))
    torch.testing.assert_close(
        result.doa[..., 1], torch.full((1, 2, 1), math.pi / 4.0), atol=1e-5, rtol=0.0
    )
    assert result.selected_positions.shape == (1, 2, 1, 3)
    with pytest.raises(TypeError, match="source_distances"):
        DPIPDToDOAMapper(config).map(
            prediction,
            geometry=geometry,
            pair_index=pair_index,
            source_distances=torch.ones(1, 2, 1),
        )


def test_near_field_mapping_requires_one_reference_for_range_origin():
    """near-field range 必须以 canonical reference-to-all pair 作为原点。"""
    geometry = torch.tensor(
        [[-0.04, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.075, 0.0]],
        dtype=torch.float32,
    )
    pair_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    config = DOAMappingConfig(
        n_fft=32,
        sample_rate=16000,
        azimuth_grid=(0.0, math.pi / 4.0),
        elevation_grid=(math.pi / 2.0,),
        range_grid=(0.5, 1.0),
        field_mode="near_field",
    )
    prediction = _near_prediction(
        config,
        geometry,
        pair_index,
        polar=math.pi / 2.0,
        azimuth=math.pi / 4.0,
        source_range=1.0,
    )
    with pytest.raises(ValueError, match="one common reference"):
        DPIPDToDOAMapper(config)(prediction, geometry=geometry, pair_index=pair_index)


def test_mapping_rejects_invalid_pairs_masks_and_near_field_configuration():
    geometry = torch.tensor([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]])
    pair_index = torch.tensor([[0, 1]], dtype=torch.int64)
    config = DOAMappingConfig(n_fft=8, sample_rate=8000)
    prediction = _far_prediction(
        config, geometry, pair_index, polar=math.pi / 2.0, azimuth=0.0
    )
    mapper = DPIPDToDOAMapper(config)
    with pytest.raises(ValueError, match="pair_mask"):
        mapper(prediction, geometry=geometry, pair_index=pair_index, pair_mask=torch.tensor([False]))
    with pytest.raises(ValueError, match="out-of-range"):
        mapper(prediction, geometry=geometry, pair_index=torch.tensor([[0, 2]]))
    with pytest.raises(ValueError, match="range_grid"):
        DOAMappingConfig(n_fft=8, sample_rate=8000, field_mode="near_field")


def test_mapping_rejects_duplicate_pairs_zero_baselines_empty_axes_and_zero_weights():
    geometry = torch.tensor(
        [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.015, 0.07, 0.0]],
        dtype=torch.float32,
    )
    pair_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64)
    config = DOAMappingConfig(
        n_fft=16,
        sample_rate=16000,
        azimuth_grid=(0.0, math.pi / 2.0),
        elevation_grid=(math.pi / 2.0,),
    )
    prediction = _far_prediction(
        config, geometry, pair_index, polar=math.pi / 2.0, azimuth=0.0
    )
    mapper = DPIPDToDOAMapper(config)
    with pytest.raises(ValueError, match="unique microphone pairs"):
        mapper(
            prediction,
            geometry=geometry,
            pair_index=torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="non-zero microphone baselines"):
        mapper(
            prediction[:, :, :, :, :1],
            geometry=torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
            ),
            pair_index=torch.tensor([[0, 1]], dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="positive B,T,F,K,P"):
        mapper(prediction[:, :0], geometry=geometry, pair_index=pair_index)
    with pytest.raises(ValueError, match="no positive prediction/feature weight"):
        mapper(
            prediction,
            geometry=geometry,
            pair_index=pair_index,
            frequency_weights=torch.zeros(config.frequencies.numel()),
        )
    with pytest.raises(ValueError, match="no positive prediction/feature weight"):
        mapper(
            torch.zeros_like(prediction),
            geometry=geometry,
            pair_index=pair_index,
        )
