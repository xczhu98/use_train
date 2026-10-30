"""wind_noise sidechain waveform 数值回归测试。"""

from __future__ import annotations

import numpy as np
import pytest

from core.data.simulation.urgent_impl import wind_noise


pytestmark = [pytest.mark.unit]

_SPEECH = np.array(
    [[0.60, -0.40, 0.20, -0.10, 0.50, -0.30, 0.15, -0.45]],
    dtype=np.float32,
)
_WIND = np.array(
    [[0.20, -0.10, 0.40, -0.30, 0.15, -0.35, 0.25, -0.05]],
    dtype=np.float32,
)
_CONFIG = {
    "fs": 1000,
    "threshold": 0.16,
    "ratio": 4.0,
    "attack": 4.0,
    "release": 8.0,
    "sc_gain": 1.5,
    "clipping": False,
    "clipping_threshold": 0.55,
    "snr": 3.0,
}


def _manual_urgent_wind_oracle(
    speech: np.ndarray,
    wind: np.ndarray,
    *,
    fs: int,
    threshold: float,
    ratio: float,
    attack: float,
    release: float,
    sc_gain: float,
    snr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按 URGENT sidechaincompress + amix 公式独立逐样本复算。"""
    speech_active = speech[:, np.sqrt(np.mean(speech**2, axis=0)) > 1e-4]
    wind_active = wind[:, np.sqrt(np.mean(wind**2, axis=0)) > 1e-4]
    speech_power = np.mean(speech_active**2)
    wind_power = np.mean(wind_active**2)
    wind_scale = 10 ** (-snr / 20.0) * np.sqrt(speech_power) / np.sqrt(
        max(wind_power, 1e-10)
    )
    scaled_wind = wind_scale * wind

    normalization = 0.9 / max(
        np.max(np.abs(speech)), np.max(np.abs(scaled_wind)), 1e-10
    )
    speech_norm = speech.astype(np.float64) * normalization
    wind_norm = scaled_wind.astype(np.float64) * normalization

    # ffmpeg sidechaincompress 默认 knee=2.82843、RMS detection、average link。
    knee = 2.82843
    log_threshold = np.log(threshold)
    knee_start = np.log(threshold / np.sqrt(knee))
    knee_stop = np.log(threshold * np.sqrt(knee))
    compressed_knee_stop = (knee_stop - log_threshold) / ratio + log_threshold
    active_level_threshold = (threshold / np.sqrt(knee)) ** 2
    attack_coeff = min(1.0, 1.0 / (attack * fs / 4000.0))
    release_coeff = min(1.0, 1.0 / (release * fs / 4000.0))

    sidechain_level = np.mean(np.abs(wind_norm * sc_gain), axis=0) ** 2
    envelope = 0.0
    gains = np.ones(sidechain_level.shape, dtype=np.float64)
    for index, level in enumerate(sidechain_level):
        coeff = attack_coeff if level > envelope else release_coeff
        envelope += (level - envelope) * coeff
        if envelope <= active_level_threshold:
            continue

        slope = 0.5 * np.log(envelope)
        compressed_slope = (slope - log_threshold) / ratio + log_threshold
        if slope < knee_stop:
            width = knee_stop - knee_start
            t = (slope - knee_start) / width
            tangent_start = width
            tangent_stop = width / ratio
            compressed_slope = (
                (2 * t**3 - 3 * t**2 + 1) * knee_start
                + (t**3 - 2 * t**2 + t) * tangent_start
                + (-2 * t**3 + 3 * t**2) * compressed_knee_stop
                + (t**3 - t**2) * tangent_stop
            )
        gains[index] = np.exp(compressed_slope - slope)

    compressed_speech = speech_norm * gains
    # amix normalize=1：两个输入各乘 1/2；随后撤销防削顶归一化。
    mixture = (compressed_speech + wind_norm) / 2.0 / normalization
    wind_stem = wind_norm / 2.0 / normalization
    return mixture, wind_stem


def test_wind_noise_waveform_matches_manual_urgent_oracle():
    expected_mix, expected_stem = _manual_urgent_wind_oracle(
        _SPEECH,
        _WIND,
        **{
            key: _CONFIG[key]
            for key in ("fs", "threshold", "ratio", "attack", "release", "sc_gain", "snr")
        },
    )

    mixture, wind_stem = wind_noise(
        _SPEECH.copy(),
        _WIND.copy(),
        rng=np.random.default_rng(2026),
        **_CONFIG,
    )

    assert mixture.shape == wind_stem.shape == _SPEECH.shape
    assert mixture.dtype == wind_stem.dtype == np.dtype(np.float64)
    assert np.all(np.isfinite(mixture))
    assert np.all(np.isfinite(wind_stem))
    np.testing.assert_allclose(mixture, expected_mix, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(wind_stem, expected_stem, rtol=1e-12, atol=1e-12)

    # 返回 stem 必须是实际进入 amix 的风噪，即 SNR 缩放后的 wind / 2。
    speech_power = np.mean(_SPEECH**2)
    wind_power = np.mean(_WIND**2)
    wind_scale = 10 ** (-_CONFIG["snr"] / 20.0) * np.sqrt(
        speech_power / wind_power
    )
    np.testing.assert_allclose(
        wind_stem,
        wind_scale * _WIND / 2.0,
        rtol=1e-12,
        atol=1e-12,
    )


def test_wind_noise_audit_is_json_safe_and_waveform_neutral():
    import json

    rng_plain = np.random.default_rng(2026)
    rng_audit = np.random.default_rng(2026)
    audit = {}
    plain = wind_noise(_SPEECH.copy(), _WIND.copy(), rng=rng_plain, **_CONFIG)
    audited = wind_noise(
        _SPEECH.copy(), _WIND.copy(), rng=rng_audit, audit_out=audit, **_CONFIG
    )

    for expected, actual in zip(plain, audited):
        np.testing.assert_array_equal(actual, expected)
    assert rng_audit.bit_generator.state == rng_plain.bit_generator.state
    assert audit["name"] == "mixture.wind_snr"
    assert audit["policy"] == "active_mean_power_legacy_dtype"
    assert audit["calibration_measure_point"] == "pre_sidechain_pre_amix"
    assert audit["final_realized_db"] is None
    assert audit["applicable"] is False
    assert audit["reason"] == "final_not_measured"
    json.dumps(audit, allow_nan=False)


def test_wind_noise_clips_only_mixture_with_urgent_bounds():
    unclipped_mix, unclipped_stem = wind_noise(
        _SPEECH.copy(),
        _WIND.copy(),
        rng=np.random.default_rng(17),
        **_CONFIG,
    )
    clipped_mix, clipped_stem = wind_noise(
        _SPEECH.copy(),
        _WIND.copy(),
        rng=np.random.default_rng(17),
        **{**_CONFIG, "clipping": True},
    )

    clipping_threshold = _CONFIG["clipping_threshold"]
    expected_clipped = np.clip(
        unclipped_mix,
        clipping_threshold * np.min(unclipped_mix),
        clipping_threshold * np.max(unclipped_mix),
    )
    assert np.any(clipped_mix != unclipped_mix)
    np.testing.assert_allclose(clipped_mix, expected_clipped, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(clipped_stem, unclipped_stem)


def test_wind_noise_same_seed_is_deterministic_and_equal_reference_is_neutral():
    first = wind_noise(
        _SPEECH.copy(),
        _WIND.copy(),
        rng=np.random.default_rng(99),
        **_CONFIG,
    )
    second = wind_noise(
        _SPEECH.copy(),
        _WIND.copy(),
        rng=np.random.default_rng(99),
        **_CONFIG,
    )
    equal_reference = wind_noise(
        _SPEECH.copy(),
        _WIND.copy(),
        rng=np.random.default_rng(99),
        snr_reference=_SPEECH.copy(),
        **_CONFIG,
    )

    for expected, actual in zip(first, second):
        np.testing.assert_array_equal(actual, expected)
    for expected, actual in zip(first, equal_reference):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("sample_name", ["speech_sample", "noise_sample", "snr_reference"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_wind_noise_rejects_nonfinite_audio_inputs(sample_name, nonfinite):
    kwargs = {**_CONFIG, "rng": np.random.default_rng(123)}
    speech = _SPEECH.copy()
    noise = _WIND.copy()
    if sample_name == "speech_sample":
        speech[0, 0] = nonfinite
    elif sample_name == "noise_sample":
        noise[0, 0] = nonfinite
    else:
        reference = _SPEECH.copy()
        reference[0, 0] = nonfinite
        kwargs["snr_reference"] = reference

    with pytest.raises(ValueError, match=rf"wind_noise {sample_name}.*finite"):
        wind_noise(speech, noise, **kwargs)


@pytest.mark.parametrize(
    "parameter",
    ["threshold", "ratio", "attack", "release", "sc_gain", "clipping_threshold", "snr"],
)
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_wind_noise_rejects_nonfinite_numeric_parameters(parameter, nonfinite):
    with pytest.raises(ValueError, match=rf"wind_noise {parameter}.*finite"):
        wind_noise(
            _SPEECH.copy(),
            _WIND.copy(),
            rng=np.random.default_rng(321),
            **{**_CONFIG, parameter: nonfinite},
        )


def test_mix_background_wind_branch_returns_contract_dtype(monkeypatch):
    """wind_noise 数学内部按 float64 计算；Adapter 边界必须回到 track 合同的 float32。

    回归场景：TSE dynamic_tse 严格模式下 float64 observation 会在
    WaveformBundle 构造期 fail（track 'mixture' must have dtype float32）。
    """
    from core.data.dynamic_dataset_base import BaseDynamicMixingDataset
    from core.data.simulation.background import mix_background
    from core.data.simulation_configs import SimulationConfigs
    from core.data.simulation.role_policy import BackgroundPolicy, BackgroundSpec

    speech = np.linspace(-0.5, 0.5, 4000, dtype=np.float32).reshape(1, -1)
    wind = np.linspace(-0.3, 0.3, 4000, dtype=np.float32).reshape(1, -1)

    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimulationConfigs.from_dict(
        {
            "prob_wind_noise_sidechain": 1.0,
            "noise_mix_prob": 0.0,
            "prob_colored_noise": 0.0,
            "snr_low": 0.0,
            "snr_high": 10.0,
            "snr_reference": "all_speech",
            "reuse_noise": False,
            "wind_noise_config": {
                "threshold": [0.1, 0.3],
                "ratio": [1, 20],
                "attack": [5, 100],
                "release": [5, 100],
                "sc_gain": [0.8, 1.2],
                "clipping_threshold": [0.85, 1.0],
                "clipping_chance": 0.75,
                "wind_noise_snr_range": [-10.0, 15.0],
            },
        },
        strict=True,
    )
    dataset.export_simulation_stems = False
    monkeypatch.setattr(
        dataset, "_mix_acoustic_get_wind_sample", lambda sr, max_dur, rng, reuse: wind
    )

    result = mix_background(
        dataset,
        speech,
        speech,
        16000,
        speech.shape[1],
        speech.shape[1],
        np.random.default_rng(0),
        True,
        5.0,
        background_spec=BackgroundSpec(BackgroundPolicy.ADDITIVE_MONO),
    )

    assert result.used_sidechain_wind is True
    assert result.observation.dtype == np.float32
    assert result.additive.dtype == np.float32


def test_spatial_wind_sidechain_projects_diffuse_control_without_mono_broadcast(monkeypatch):
    from core.data.dynamic_dataset_base import BaseDynamicMixingDataset
    from core.data.simulation.background import mix_background
    from core.data.simulation.role_policy import build_background_spec
    from core.data.simulation_configs import SimulationConfigs

    sample_length = 4000
    speech = np.vstack(
        [
            np.linspace(-0.5, 0.5, sample_length),
            np.linspace(0.25, -0.25, sample_length),
        ]
    ).astype(np.float32)
    wind = np.linspace(-0.3, 0.3, sample_length, dtype=np.float32).reshape(1, -1)
    geometry = np.asarray([[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]], dtype=np.float32)
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "prob_wind_noise_sidechain": 1.0,
            "noise_mix_prob": 0.0,
            "prob_colored_noise": 0.0,
            "snr_low": 0.0,
            "snr_high": 10.0,
            "snr_reference": "all_speech",
            "reuse_noise": False,
            "wind_noise_config": {
                "threshold": [0.2, 0.2],
                "ratio": [4.0, 4.0],
                "attack": [4.0, 4.0],
                "release": [8.0, 8.0],
                "sc_gain": [1.5, 1.5],
                "clipping_threshold": [0.9, 0.9],
                "clipping_chance": 0.0,
                "wind_noise_snr_range": [3.0, 3.0],
            },
        },
        strict=True,
    )
    dataset.export_simulation_stems = False
    monkeypatch.setattr(
        dataset, "_mix_acoustic_get_wind_sample", lambda sr, max_dur, rng, reuse: wind
    )
    monkeypatch.setattr(
        dataset,
        "_gen_wind_noise_params",
        lambda rng: {
            "threshold": 0.2,
            "ratio": 4.0,
            "attack": 4.0,
            "release": 8.0,
            "sc_gain": 1.5,
            "clipping": False,
            "clipping_threshold": 0.9,
        },
    )

    result = mix_background(
        dataset,
        speech.copy(),
        speech.copy(),
        16000,
        sample_length,
        sample_length,
        np.random.default_rng(17),
        True,
        5.0,
        image_geometry=geometry,
        background_spec=build_background_spec(image_geometry=geometry, shared_scene=True),
    )

    assert result.used_sidechain_wind is True
    assert result.observation.shape == (2, sample_length)
    assert result.additive.shape == (2, sample_length)
    assert not np.array_equal(result.observation[0], result.observation[1])
    facts = result.noise_image_provenance["wind_sidechain"]
    assert facts["shape"] == [2, sample_length]
    assert facts["projection"] == "diffuse_coherence"
    assert facts["coherence_policy"] == dataset.sim_cfg.spatial_diffuse_coherence_model
    assert result.noise_image_provenance["wind_sidechain_applied"] is True


def test_spatial_wind_sidechain_rejects_mono_direct_multichannel_control(monkeypatch):
    from core.data.dynamic_dataset_base import BaseDynamicMixingDataset
    from core.data.simulation.background import mix_background
    from core.data.simulation.role_policy import BackgroundPolicy, BackgroundSpec
    from core.data.simulation_configs import SimulationConfigs

    sample_length = 256
    speech = np.ones((2, sample_length), dtype=np.float32)
    wind = np.ones((1, sample_length), dtype=np.float32)
    geometry = np.asarray([[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]], dtype=np.float32)
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "prob_wind_noise_sidechain": 1.0,
            "noise_mix_prob": 0.0,
            "prob_colored_noise": 0.0,
            "wind_noise_config": {
                "threshold": [0.2, 0.2], "ratio": [4.0, 4.0],
                "attack": [4.0, 4.0], "release": [8.0, 8.0],
                "sc_gain": [1.5, 1.5], "clipping_threshold": [0.9, 0.9],
                "clipping_chance": 0.0, "wind_noise_snr_range": [3.0, 3.0],
            },
        },
        strict=True,
    )
    dataset.export_simulation_stems = False
    monkeypatch.setattr(
        dataset, "_mix_acoustic_get_wind_sample", lambda sr, max_dur, rng, reuse: wind
    )
    monkeypatch.setattr(
        dataset,
        "_gen_wind_noise_params",
        lambda rng: {
            "threshold": 0.2, "ratio": 4.0, "attack": 4.0, "release": 8.0,
            "sc_gain": 1.5, "clipping": False, "clipping_threshold": 0.9,
        },
    )

    with pytest.raises(ValueError, match=r"explicit \[C,T\] wind asset"):
        mix_background(
            dataset,
            speech,
            speech,
            16000,
            sample_length,
            sample_length,
            np.random.default_rng(17),
            True,
            5.0,
            image_geometry=geometry,
            background_spec=BackgroundSpec(
                BackgroundPolicy.DIRECT_MULTICHANNEL,
                geometry_bound=True,
            ),
        )


def test_spatial_wind_sidechain_reuses_point_source_background_spec():
    from core.data.dynamic_dataset_base import BaseDynamicMixingDataset
    from core.data.simulation.noise_image import NoiseImage
    from core.data.simulation.role_policy import BackgroundPolicy, BackgroundSpec
    from core.data.simulation_configs import SimulationConfigs

    class PointProvider:
        def __init__(self, geometry):
            self.geometry = geometry

        def validate_geometry(self, value):
            np.testing.assert_array_equal(value, self.geometry)

        def render(self, waveform, *, sample_rate):
            value = np.asarray(waveform, dtype=np.float32)
            return NoiseImage(
                np.vstack([value, -value]).astype(np.float32),
                {
                    "source_identity": "wind_actor/point-1",
                    "sample_rate": sample_rate,
                },
            )

    geometry = np.asarray([[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]], dtype=np.float32)
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
        },
        strict=True,
    )
    facts = {}
    image = dataset._render_wind_sidechain_image(
        np.ones((1, 32), dtype=np.float32),
        image_geometry=geometry,
        background_spec=BackgroundSpec(
            BackgroundPolicy.POINT_SOURCE_RIR,
            geometry_bound=True,
            point_source_provider=PointProvider(geometry),
        ),
        sr=16000,
        rng=np.random.default_rng(0),
        source_identity="wind_scp:wind-1",
        facts_out=facts,
    )

    assert image.shape == (2, 32)
    assert not np.array_equal(image[0], image[1])
    assert facts["projection"] == "point_source_rir"
    assert facts["source_identity"] == "wind_scp:wind-1"
    assert facts["projection_provenance"]["source_identity"] == "wind_actor/point-1"
