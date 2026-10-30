"""空间背景与多通道 observation 退化合同。"""

import numpy as np
import pytest

from core.data.simulation.spatial_background import (
    SpatialBackground,
    parse_spatial_background_manifest_record,
    render_diffuse_background,
    validate_spatial_background,
)

pytestmark = pytest.mark.unit


def test_diffuse_background_is_explicit_multichannel_and_finite():
    source = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    positions = np.asarray([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]], dtype=np.float64)
    result = render_diffuse_background(source, positions, 16000, np.random.default_rng(3))
    assert result.waveform.shape == (2, 128)
    assert result.waveform.dtype == np.float32
    assert np.isfinite(result.waveform).all()
    assert not np.array_equal(result.waveform[0], result.waveform[1])
    assert result.source_identity.startswith("inline:")
    assert result.coherence_policy == "habets_spherical"
    assert result.sample_rate == 16000
    assert result.geometry["microphone_positions_m"] == positions.tolist()
    assert result.physics_status == "approximate"
    assert result.asset_kind == "generated"
    assert result.provenance["spatial_model"] == "habets_spherical_sinc"
    assert len(result.content_hash) == 64
    assert result.to_metadata()["content_hash"] == result.content_hash


def test_diffuse_background_matches_spherical_sinc_coherence():
    from scipy.signal import stft

    sample_rate = 8000
    spacing_m = 0.08
    source = np.random.default_rng(19).normal(size=131072).astype(np.float32)
    positions = np.asarray([[0.0, 0.0, 0.0], [spacing_m, 0.0, 0.0]])
    rendered = render_diffuse_background(
        source, positions, sample_rate, np.random.default_rng(20)
    )
    frequencies, _, spectra = stft(
        rendered.waveform,
        fs=sample_rate,
        window="hann",
        nperseg=512,
        noverlap=384,
        nfft=512,
        axis=-1,
    )
    cross = np.mean(spectra[0] * np.conj(spectra[1]), axis=-1)
    power_0 = np.mean(np.square(np.abs(spectra[0])), axis=-1)
    power_1 = np.mean(np.square(np.abs(spectra[1])), axis=-1)
    coherence = cross / np.sqrt(power_0 * power_1)
    index = int(np.argmin(np.abs(frequencies - 1000.0)))
    expected = np.sinc(2.0 * frequencies[index] * spacing_m / 343.0)
    assert coherence[index].real == pytest.approx(expected, abs=0.12)
    assert abs(coherence[index].imag) < 0.12


def test_diffuse_background_calibrates_the_declared_reference_channel():
    source = np.zeros(64, dtype=np.float32)
    source[4] = 1.0
    positions = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
    result = render_diffuse_background(
        source,
        positions,
        16000,
        np.random.default_rng(4),
        reference_channel=1,
    )
    source_rms = np.sqrt(np.mean(source.astype(np.float64) ** 2))
    reference_rms = np.sqrt(np.mean(result.waveform[1].astype(np.float64) ** 2))
    np.testing.assert_allclose(reference_rms, source_rms, rtol=1e-6, atol=1e-8)
    assert not np.array_equal(result.waveform[0], result.waveform[1])


def test_spatial_background_rejects_mono_broadcast_and_wrong_shape():
    with pytest.raises(TypeError, match=r"explicit float32 \[C,T\]"):
        validate_spatial_background(np.zeros(8, dtype=np.float32), num_channels=2, length=8)
    with pytest.raises(ValueError, match=r"must be \(2, 8\)"):
        validate_spatial_background(np.zeros((1, 8), dtype=np.float32), num_channels=2, length=8)


def test_diffuse_background_requires_geometry_and_generator():
    with pytest.raises(ValueError, match="microphone_positions"):
        render_diffuse_background(
            np.ones(8, np.float32), np.zeros((2, 2)), 16000, np.random.default_rng(1)
        )


def test_spatial_background_rejects_hash_mismatch_and_noise92_without_exact_metadata():
    waveform = np.zeros((2, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="content_hash"):
        SpatialBackground(
            waveform,
            source_identity="noise92:fixture",
            sample_rate=16000,
            geometry={"array_profile": "linear_2ch_80mm"},
            coherence_policy="noise92_measured_coherence",
            generator_version="noise92-fixture-v1",
            provenance={"dataset": "Noise92"},
            content_hash="0" * 64,
            asset_kind="noise92",
            physics_status="exact_asset",
        )
    with pytest.raises(ValueError, match="physics_status"):
        SpatialBackground(
            waveform,
            source_identity="noise92:fixture",
            sample_rate=16000,
            geometry={"array_profile": "linear_2ch_80mm"},
            coherence_policy="noise92_measured_coherence",
            generator_version="noise92-fixture-v1",
            provenance={"dataset": "Noise92"},
            asset_kind="noise92",
        )


def test_spatial_background_manifest_record_requires_identity_contract():
    record = {
        "path": "noise92/scene-1.npy",
        "source_identity": "noise92:scene-1",
        "sample_rate": 16000,
        "geometry": {"array_profile": "linear_2ch_80mm"},
        "coherence_policy": "noise92_measured_coherence",
        "generator_version": "noise92-export-v1",
        "provenance": {"dataset": "Noise92", "release": "fixture"},
        "content_hash": "a" * 64,
        "asset_kind": "noise92",
        "physics_status": "exact_asset",
    }
    spec = parse_spatial_background_manifest_record(record, location="fixture:1")
    assert spec.source_identity == "noise92:scene-1"
    assert spec.sample_rate == 16000
    assert spec.asset_kind == "noise92"
    with pytest.raises(ValueError, match="exact_asset"):
        parse_spatial_background_manifest_record(
            {**record, "physics_status": "approximate"}, location="fixture:2"
        )
    with pytest.raises(ValueError, match="missing"):
        parse_spatial_background_manifest_record(
            {key: value for key, value in record.items() if key != "source_identity"},
            location="fixture:3",
        )
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        render_diffuse_background(
            np.ones(8, np.float32), np.zeros((2, 3)), 16000, object()
        )
