"""noise_scp 波形到 observation-side noise image 的内部合同。"""

import numpy as np
import pytest

from core.data.simulation.noise_image import NoiseImage, adapt_noise_waveform
from core.data.simulation.role_policy import BackgroundPolicy, BackgroundSpec

pytestmark = pytest.mark.unit


def _geometry(channels: int = 2) -> np.ndarray:
    return np.asarray([[index * 0.04, 0.0, 0.0] for index in range(channels)], dtype=np.float64)


def test_multichannel_noise_is_used_directly_and_records_provenance():
    noise = np.arange(12, dtype=np.float32).reshape(2, 6)
    result = adapt_noise_waveform(
        noise, geometry=_geometry(), sample_rate=16000,
        source_identity="noise_scp:fixture-1", provenance={"split": "train"},
    )
    np.testing.assert_array_equal(result.waveform, noise)
    assert result.provenance["projection"] == "direct_multichannel"
    assert result.provenance["input_shape"] == [2, 6]
    assert result.provenance["output_shape"] == [2, 6]
    assert result.provenance["rir_applied"] is False
    assert result.provenance["split"] == "train"


def test_multichannel_noise_rejects_geometry_channel_mismatch():
    with pytest.raises(ValueError, match=r"C=2.*geometry C=3"):
        adapt_noise_waveform(np.zeros((2, 8), np.float32), geometry=_geometry(3), sample_rate=16000)


def test_mono_noise_uses_geometry_aware_renderer_and_records_policy():
    noise = np.random.default_rng(4).normal(size=128).astype(np.float32)
    result = adapt_noise_waveform(
        noise, geometry=_geometry(), sample_rate=16000,
        rng=np.random.default_rng(5), source_identity="noise_scp:mono",
    )
    assert result.waveform.shape == (2, 128)
    assert result.provenance["projection"] == "geometry_aware"
    assert result.provenance["source_identity"] == "noise_scp:mono"
    assert result.provenance["rir_applied"] is False
    assert not np.array_equal(result.waveform[0], result.waveform[1])


def test_mono_noise_broadcast_is_explicit_and_does_not_need_rng():
    noise = np.arange(5, dtype=np.float32)
    result = adapt_noise_waveform(noise, geometry=_geometry(3), sample_rate=16000, mono_policy="broadcast")
    expected = np.broadcast_to(noise[None, :], (3, 5))
    np.testing.assert_array_equal(result.waveform, expected)
    assert result.provenance["projection"] == "broadcast"


@pytest.mark.parametrize("noise", [np.zeros((2, 3, 4), np.float32), np.zeros((0,), np.float32)])
def test_noise_input_shape_and_empty_input_are_rejected(noise):
    with pytest.raises((ValueError, TypeError)):
        adapt_noise_waveform(noise, geometry=_geometry(), sample_rate=16000, mono_policy="broadcast")


def test_noise_image_validates_output_contract():
    with pytest.raises(TypeError, match="float32"):
        NoiseImage(np.zeros((2, 4), dtype=np.float64), {})


def test_single_channel_image_renders_for_array_geometry():
    noise = np.arange(6, dtype=np.float32)[None, :]
    result = adapt_noise_waveform(noise, geometry=_geometry(2), sample_rate=16000, rng=np.random.default_rng(9))
    assert not np.array_equal(result.waveform[0], noise[0])
    assert result.waveform.shape == (2, 6)
    assert result.provenance["projection"] == "geometry_aware"


def test_full_image_snr_uses_one_gain_for_asymmetric_channels():
    from core.data.simulation.noise_image import scale_noise_image
    speech = np.asarray([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
    noise = NoiseImage(np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32), {})
    result = scale_noise_image(speech, noise, snr_db=0.0)
    expected_gain = np.sqrt(np.sum(speech.astype(np.float64) ** 2) / 4.0)
    np.testing.assert_allclose(result.waveform, noise.waveform * expected_gain, rtol=1e-6)
    assert result.provenance["snr_policy"] == "full_image_energy"
    assert result.provenance["snr_nominal_db"] == 0.0
    np.testing.assert_allclose(result.provenance["snr_realized_db"], 0.0, atol=1e-6)


@pytest.mark.parametrize("shape", [(17,), (1, 17)])
def test_c1_is_exact_identity_without_rng(shape):
    noise = np.linspace(-.7, .3, 17, dtype=np.float32).reshape(shape)
    class NoRandom:
        def __getattr__(self, name):
            raise AssertionError(f"C=1 must not draw {name}")
    image = adapt_noise_waveform(noise, geometry=_geometry(1), sample_rate=8000, rng=NoRandom())
    np.testing.assert_array_equal(image.waveform, noise.reshape(1, -1))
    assert image.provenance["projection"] == "mono_passthrough"


def test_spatial_background_ordinary_path_scales_only_after_projection(monkeypatch):
    from types import SimpleNamespace
    from core.data.simulation import background
    from core.data.simulation_configs import SimulationConfigs
    cfg = SimulationConfigs.from_dict({})
    cfg.noise_mix_prob = 0.0
    cfg.prob_wind_noise_sidechain = 0.0
    noise = np.asarray([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    speech = np.asarray([[1., 1., 1.], [3., 3., 3.]], dtype=np.float32)
    dataset = SimpleNamespace(
        sim_cfg=cfg, export_simulation_stems=False,
        _acoustic_plan_record=lambda: None,
        _read_one_additive_noise_scp_only_with_identity=lambda *_: ("noise_scp:measured", noise.copy()),
        _mix_acoustic_colored_noise=lambda observed, reference, additive, *_: (observed, additive),
    )
    def forbidden_mono_mix(*args, **kwargs):
        raise AssertionError("spatial ordinary noise must not use mono mix_noise")
    monkeypatch.setattr(background, "mix_noise", forbidden_mono_mix)
    result = background.mix_background(dataset, speech.copy(), speech, 8000, 3, 3,
                                       np.random.default_rng(1), True, 6.,
                                       image_geometry=_geometry(2),
                                       background_spec=BackgroundSpec(
                                           BackgroundPolicy.DIFFUSE_COHERENCE,
                                           geometry_bound=True,
                                           shared_scene=True,
                                       ))
    gain = np.sqrt(np.sum(speech.astype(np.float64)**2) / np.sum(noise.astype(np.float64)**2) / 10**.6)
    np.testing.assert_allclose(result.additive, noise * gain, rtol=1e-6)
    np.testing.assert_allclose(result.observation, speech + noise * gain, rtol=1e-6)
    assert result.noise_image_provenance["source_identity"] == "noise_scp:measured"
    assert len(result.noise_image_provenance["ordinary_components"]) == 1


def test_spatial_noise_disk_channels_are_not_lost_in_mono_cache(tmp_path):
    import soundfile as sf
    from core.data.simulation_audio_utils import _load_audio_cached, _read_spatial_noise_2d
    path = tmp_path / "stereo.wav"
    wave = np.stack((np.arange(40), -2 * np.arange(40)), axis=1).astype(np.float32) / 100
    sf.write(path, wave, 8000, subtype="FLOAT")
    mono = _load_audio_cached(str(path), 8000, True)
    np.testing.assert_array_equal(mono, wave[:, 0])
    crop = {}
    actual = _read_spatial_noise_2d(str(path), 8000, 17, strict=True,
                                  rng=np.random.default_rng(3), crop_record=crop)
    start = crop["crop"].start_sample
    np.testing.assert_array_equal(actual, wave[start:start+17].T)
    actual[:] = 0
    again = _read_spatial_noise_2d(
        str(path), 8000, strict=True, rng=np.random.default_rng(0)
    )
    np.testing.assert_array_equal(again, wave.T)


def test_spatial_background_accepts_adapter_without_plan_record():
    from types import SimpleNamespace
    from core.data.simulation import background
    from core.data.simulation_configs import SimulationConfigs
    cfg = SimulationConfigs.from_dict({})
    cfg.noise_mix_prob = 0.0
    cfg.prob_wind_noise_sidechain = 0.0
    noise = np.asarray([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    speech = np.asarray([[1., 1., 1.], [3., 3., 3.]], dtype=np.float32)
    # 缺 _acoustic_plan_record 的最小 Adapter 必须走 seam 而不是 AttributeError。
    dataset = SimpleNamespace(
        sim_cfg=cfg, export_simulation_stems=False,
        _read_one_additive_noise_scp_only_with_identity=lambda *_: ("noise_scp:measured", noise.copy()),
        _mix_acoustic_colored_noise=lambda observed, reference, additive, *_: (observed, additive),
    )
    result = background.mix_background(dataset, speech.copy(), speech, 8000, 3, 3,
                                       np.random.default_rng(2), True, 6.,
                                       image_geometry=_geometry(2),
                                       background_spec=BackgroundSpec(
                                           BackgroundPolicy.DIFFUSE_COHERENCE,
                                           geometry_bound=True,
                                           shared_scene=True,
                                       ))
    assert result.observation.shape == speech.shape
    assert result.noise_image_provenance["source_identity"] == "noise_scp:measured"


def test_real_se_noise_scp_keeps_recorded_microphone_channels(tmp_path):
    import soundfile as sf
    from test_se_spatial_target_only import _build_target_only_dataset
    dataset = _build_target_only_dataset(tmp_path, 2)
    t = np.linspace(0, 5, 128, dtype=np.float32)
    sf.write(tmp_path / "noise.wav", np.stack((np.sin(t), np.cos(t)), axis=1) * .02,
             8000, subtype="FLOAT")
    result = dataset.get_sample_with_trace(0, force_branch="prob_target_noise")
    facts = result.trace.metadata_dict()["sources"]["scene"]["scene"]["noise_image_provenance"]
    assert facts["projection"] == "direct_multichannel"
    assert facts["input_shape"] == [2, 128]
    assert facts["source_identity"] == "noise_scp:noise"
    assert result.sample[1].shape == (2, 128)
