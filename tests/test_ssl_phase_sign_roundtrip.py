"""projection→mapping 相位符号 round-trip 强制测试。

与 tests/test_ssl_doa_mapping.py 的区别：该文件用 mapping 自身公式生成预测，
只能验证自洽；本文件用 ``project_ssl_scene`` 的 direct-path 谱产出 target，
把 target 当作“模型预测”送入 ``DPIPDToDOAMapper``，验证 projection 的
``others * conj(ref)`` 相位约定与 mapping 模板 ``phase_sign`` 的绑定。
方位取半平面内非对称的 az=60°：符号错配时峰值必然落在镜像 az=120°，
断言必然失败。发现镜像时只允许修改配置层 ``phase_sign``，不得改
projection/mapping 内部公式。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tasks.ssl.data.doa_mapping import DOAMappingConfig, DPIPDToDOAMapper
from tasks.ssl.data.projection import SSLProjectionConfig, SSLSceneInput, project_ssl_scene


pytestmark = pytest.mark.unit

_SAMPLE_RATE = 16000
_N_FFT = 256
_HOP = 128
# 整数样本时延：tau = 2 个样本 = d*cos(60°)/c，d = 2*c/(sr*cos60°) = 0.08575 m。
_DELAY_SAMPLES = 2
_MIC_SPACING_M = 2.0 * 343.0 / (_SAMPLE_RATE * 0.5)
_NEAR_SOUND_SPEED = 343.0  # 与 SimulationConfigs 的 canonical RIR 默认一致。
_AZIMUTH = math.pi / 3.0  # 60°，半平面 [0, π] 内非对称方位。
_MIRROR_AZIMUTH = math.pi - _AZIMUTH  # 符号错配时的镜像：120°。
_SOURCE_POSITION = (1.0, math.sqrt(3.0), 0.0)  # r=2 m，az=60°，polar=90°。
_GRID_STEP = math.pi / 180.0


def _analytic_scene_item() -> dict:
    """构造解析双麦克风 scene 并走 canonical ``project_ssl_scene`` 投影。

    源方向 u=(cos60°, sin60°, 0)，mic1 更靠近源，信号相对 mic0 提前
    ``_DELAY_SAMPLES`` 个样本；target 相位 = arg(X_other·conj(X_ref))
    = +2πf(mic_other-mic_ref)·u/c。
    """
    length = 4096
    generator = np.random.default_rng(20260930)
    signal = generator.standard_normal(length + _DELAY_SAMPLES).astype(np.float32)
    mic0 = signal[:length]
    mic1 = signal[_DELAY_SAMPLES : _DELAY_SAMPLES + length]
    direct = np.stack((mic0, mic1), axis=0)[None, :, :]  # [K=1, C=2, T]
    scene = SSLSceneInput(
        observation=direct[0].copy(),
        direct_tracks=direct,
        mic_positions=np.array(
            [[0.0, 0.0, 0.0], [_MIC_SPACING_M, 0.0, 0.0]], dtype=np.float32
        ),
        source_positions=np.array([_SOURCE_POSITION], dtype=np.float32),
        sample_rate=_SAMPLE_RATE,
        provenance={"generator_version": "phase-sign-roundtrip-fixture-v1"},
    )
    return project_ssl_scene(
        scene,
        SSLProjectionConfig(
            n_fft=_N_FFT,
            hop_length=_HOP,
            win_length=_N_FFT,
            window="hamming",
            center=False,
        ),
    )


def _analytic_three_mic_scene_item(*, exclude_dc: bool = False) -> dict:
    """构造三麦解析 scene，保留 projection 的 canonical pair 顺序。

    三个麦克风沿 x 轴按同一间距排列；对 az=60°，相邻麦克风的
    ``_DELAY_SAMPLES`` 个采样时延可精确实现，因而两个 reference-to-all
    pair 同时来自同一个已知方向。该 fixture 专门用于验证 pair 轴置换和
    ``exclude_dc`` 的真实 projection→mapping round-trip。
    """
    length = 4096
    generator = np.random.default_rng(20261001)
    signal = generator.standard_normal(length + 2 * _DELAY_SAMPLES).astype(np.float32)
    direct = np.stack(
        (
            signal[:length],
            signal[_DELAY_SAMPLES : _DELAY_SAMPLES + length],
            signal[2 * _DELAY_SAMPLES : 2 * _DELAY_SAMPLES + length],
        ),
        axis=0,
    )[None, :, :]
    scene = SSLSceneInput(
        observation=direct[0].copy(),
        direct_tracks=direct,
        mic_positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [_MIC_SPACING_M, 0.0, 0.0],
                [2.0 * _MIC_SPACING_M, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        source_positions=np.array([_SOURCE_POSITION], dtype=np.float32),
        sample_rate=_SAMPLE_RATE,
        provenance={"generator_version": "phase-sign-pair-fixture-v1"},
    )
    return project_ssl_scene(
        scene,
        SSLProjectionConfig(
            n_fft=_N_FFT,
            hop_length=_HOP,
            win_length=_N_FFT,
            window="hamming",
            center=False,
            exclude_dc=exclude_dc,
        ),
    )


def _analytic_near_field_nonzero_reference_scene_item() -> dict:
    """构造 ref=1 的近场 scene，直接走 projection 的 physical pair target。

    ref 麦克风周围放置两个非共线的 other 麦克风；两个传播距离差都取
    2 个采样时延，源位于 ref 麦克风正前方 1 m。非共线 pair 使 range
    候选可辨识，同时可解析验证 reference-channel 原点。
    """
    length = 4096
    generator = np.random.default_rng(20261001)
    signal = generator.standard_normal(length + _DELAY_SAMPLES).astype(np.float32)
    spacing = _NEAR_SOUND_SPEED * _DELAY_SAMPLES / _SAMPLE_RATE
    lateral = math.sqrt((1.0 + spacing) ** 2 - 1.0)
    direct = np.stack(
        (
            signal[:length],
            signal[_DELAY_SAMPLES : _DELAY_SAMPLES + length],
            signal[:length],
        ),
        axis=0,
    )[None, :, :]
    source = (1.0, 0.0, 0.0)
    scene = SSLSceneInput(
        observation=direct[0].copy(),
        direct_tracks=direct,
        mic_positions=np.array(
            [[-spacing, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, lateral, 0.0]],
            dtype=np.float32,
        ),
        source_positions=np.array([source], dtype=np.float32),
        sample_rate=_SAMPLE_RATE,
        provenance={"generator_version": "near-field-reference-origin-fixture-v1"},
    )
    return project_ssl_scene(
        scene,
        SSLProjectionConfig(
            n_fft=_N_FFT,
            hop_length=_HOP,
            win_length=_N_FFT,
            window="hamming",
            center=False,
            reference_channel=1,
            target_representation="dpipd_near_field",
            sound_speed=_NEAR_SOUND_SPEED,
        ),
    )


def _map_target(item: dict, config: DOAMappingConfig):
    mapper = DPIPDToDOAMapper(config)
    return mapper(
        item["target_dpipd"].unsqueeze(0),
        geometry=item["geometry"],
        pair_index=item["pair_index"],
        pair_mask=item["pair_mask"],
        vad=item["vad"].unsqueeze(0),
        output_lengths=torch.tensor([item["output_lengths"]], dtype=torch.int64),
    )


def _grid_config(**overrides) -> DOAMappingConfig:
    kwargs = {
        "n_fft": _N_FFT,
        "sample_rate": _SAMPLE_RATE,
        # 官方 search_space=[0,π] 的半平面方位网格，1° 分辨率，含 60°/120°。
        "azimuth_grid": tuple(float(v) for v in np.linspace(0.0, math.pi, 181)),
        "elevation_grid": (math.pi / 2.0,),
    }
    kwargs.update(overrides)
    return DOAMappingConfig(**kwargs)


def test_projection_target_roundtrip_recovers_analytic_doa_with_default_config():
    """默认配置必须恢复解析 DOA；失败即默认 phase_sign 与 projection 镜像。"""
    item = _analytic_scene_item()
    assert abs(item["doa"][0, 0, 1].item() - _AZIMUTH) < 1e-5
    result = _map_target(item, _grid_config())
    assert result.valid_mask.all()
    torch.testing.assert_close(
        result.doa[..., 1],
        torch.full_like(result.doa[..., 1], _AZIMUTH),
        atol=_GRID_STEP / 2.0,
        rtol=0.0,
    )


def test_mirror_phase_sign_selects_mirror_azimuth_not_analytic_doa():
    """显式 phase_sign=-1 必须落到镜像 120°，证明 round-trip 对符号敏感。"""
    item = _analytic_scene_item()
    result = _map_target(item, _grid_config(phase_sign=-1.0))
    assert result.valid_mask.all()
    estimated = result.doa[..., 1]
    torch.testing.assert_close(
        estimated,
        torch.full_like(estimated, _MIRROR_AZIMUTH),
        atol=_GRID_STEP / 2.0,
        rtol=0.0,
    )
    assert (estimated - _AZIMUTH).abs().max().item() > _GRID_STEP


def test_projection_target_roundtrip_preserves_pair_axis_order():
    """置换 target 与 pair_index 的同一 pair 轴必须保持 DOA 不变。"""
    item = _analytic_three_mic_scene_item()
    assert item["pair_index"].tolist() == [[0, 1], [0, 2]]
    result = _map_target(item, _grid_config())
    assert result.valid_mask.all()
    torch.testing.assert_close(
        result.doa[..., 1],
        torch.full_like(result.doa[..., 1], _AZIMUTH),
        atol=_GRID_STEP / 2.0,
        rtol=0.0,
    )

    pair_permutation = torch.tensor([1, 0], dtype=torch.int64)
    permuted = dict(item)
    permuted["target_dpipd"] = item["target_dpipd"][:, :, :, pair_permutation, :]
    permuted["pair_index"] = item["pair_index"][pair_permutation]
    permuted["pair_mask"] = item["pair_mask"][pair_permutation]
    permuted_result = _map_target(permuted, _grid_config())
    torch.testing.assert_close(permuted_result.doa, result.doa, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(permuted_result.scores, result.scores, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("exclude_dc", [False, True])
def test_projection_target_roundtrip_tracks_exclude_dc_frequency_axis(exclude_dc: bool):
    """DC 删除必须同时改变投影频率轴和 mapper 模板，并保留 60°。"""
    item = _analytic_three_mic_scene_item(exclude_dc=exclude_dc)
    expected_freqs = _N_FFT // 2 + 1 - int(exclude_dc)
    assert item["features"].shape[1] == expected_freqs
    assert item["target_dpipd"].shape[1] == expected_freqs
    result = _map_target(item, _grid_config(exclude_dc=exclude_dc))
    assert result.valid_mask.all()
    torch.testing.assert_close(
        result.doa[..., 1],
        torch.full_like(result.doa[..., 1], _AZIMUTH),
        atol=_GRID_STEP / 2.0,
        rtol=0.0,
    )


def test_projection_near_field_roundtrip_uses_nonzero_reference_origin_and_range():
    """近场 projection target 必须恢复 ref=1 原点的 DOA×range。"""
    item = _analytic_near_field_nonzero_reference_scene_item()
    assert item["pair_index"].tolist() == [[1, 0], [1, 2]]
    assert item["provenance"]["sound_speed_mps"] == _NEAR_SOUND_SPEED
    torch.testing.assert_close(
        item["source_distances"],
        torch.ones_like(item["source_distances"]),
        atol=1e-6,
        rtol=0.0,
    )
    config = DOAMappingConfig(
        n_fft=_N_FFT,
        sample_rate=_SAMPLE_RATE,
        azimuth_grid=(0.0, math.pi),
        elevation_grid=(math.pi / 2.0,),
        range_grid=(0.5, 1.0, 1.5),
        field_mode="near_field",
        sound_speed=_NEAR_SOUND_SPEED,
    )
    result = _map_target(item, config)
    assert result.valid_mask.all()
    torch.testing.assert_close(
        result.doa[..., 1],
        torch.zeros_like(result.doa[..., 1]),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.selected_ranges,
        torch.ones_like(result.selected_ranges),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.selected_positions[..., 0],
        torch.full_like(result.selected_positions[..., 0], 1.0),
        atol=1e-5,
        rtol=0.0,
    )


def test_projection_provenance_carries_configured_phase_sign():
    """projection 输出必须把 mapping 相位符号写入 provenance，避免配置脱链。"""
    # 信号长度必须不小于 n_fft，否则 torch.stft 无法建窗。
    length = 4 * _N_FFT
    waveform = np.sin(2.0 * np.pi * np.arange(length, dtype=np.float32) / length)
    scene = SSLSceneInput(
        observation=np.stack((waveform, np.roll(waveform, 1)), axis=0),
        direct_tracks=np.stack(
            (np.stack((waveform, np.roll(waveform, 1)), axis=0),), axis=0
        ),
        mic_positions=np.array(
            [[0.0, 0.0, 0.0], [_MIC_SPACING_M, 0.0, 0.0]], dtype=np.float32
        ),
        source_positions=np.array([_SOURCE_POSITION], dtype=np.float32),
        sample_rate=_SAMPLE_RATE,
        provenance={"generator_version": "phase-sign-provenance-fixture-v1"},
    )
    item = project_ssl_scene(
        scene,
        SSLProjectionConfig(
            n_fft=_N_FFT,
            hop_length=_HOP,
            win_length=_N_FFT,
            window="hamming",
            center=False,
            phase_sign=-1.0,
        ),
    )
    assert item["provenance"]["phase_sign"] == -1.0
