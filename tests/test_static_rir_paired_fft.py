"""静态 mono RIR paired FFT 回归测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.signal

import core.data.dynamic_dataset_base as dynamic_base
import core.data.simulation.urgent_impl as urgent
from core.data.dynamic_dataset_base import BaseDynamicMixingDataset


pytestmark = [pytest.mark.unit]


def _normalize_with_peak(
    rir: np.ndarray, enabled: bool, peak: float
) -> np.ndarray:
    if not enabled or peak <= 1e-8:
        return rir
    return rir / peak


@pytest.mark.parametrize("normalize_rir", [True, False])
@pytest.mark.parametrize("speech_dtype,rir_dtype", [(np.float32, np.float32), (np.float64, np.float32)])
def test_paired_fft_matches_independent_scipy_oracles(
    speech_dtype, rir_dtype, normalize_rir
):
    rng = np.random.default_rng(17)
    speech = rng.standard_normal((1, 37)).astype(speech_dtype)
    rir = np.zeros((1, 13), dtype=rir_dtype)
    rir[0, 0] = 0.2
    rir[0, 1] = 0.1
    rir[0, 2] = 0.05
    rir[0, 7] = 1.0
    early_rir = urgent.estimate_early_rir(rir, early_rir_sec=0.00025, fs=16000)

    full, early = urgent.add_paired_reverberation(
        speech,
        rir,
        early_rir_sec=0.00025,
        fs=16000,
        normalize_rir=normalize_rir,
    )
    full_peak = float(np.max(np.abs(rir)))
    full_oracle = scipy.signal.convolve(
        speech, _normalize_with_peak(rir, normalize_rir, full_peak), mode="full"
    )[:, : speech.shape[1]]
    early_oracle = scipy.signal.convolve(
        speech, _normalize_with_peak(early_rir, normalize_rir, full_peak), mode="full"
    )[:, : speech.shape[1]]

    assert full.shape == speech.shape
    assert early.shape == speech.shape
    assert full.dtype == np.result_type(speech.dtype, rir.dtype, np.float32)
    assert early.dtype == full.dtype
    np.testing.assert_allclose(full, full_oracle, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(early, early_oracle, rtol=2e-6, atol=2e-6)
    # full/early 共用 full peak，因此 direct-path 相对增益不随 normalize 开关改变。
    assert np.isclose(early[0, 0] / full[0, 0], 1.0, rtol=2e-5)


@pytest.mark.parametrize("speech_len,rir_len", [(16, 8), (256, 128), (4096, 1024)])
def test_paired_fft_representative_lengths_match_direct_convolution(
    speech_len, rir_len
):
    rng = np.random.default_rng(speech_len + rir_len)
    speech = rng.standard_normal((1, speech_len)).astype(np.float32)
    rir = rng.standard_normal((1, rir_len)).astype(np.float32)
    early_rir = urgent.estimate_early_rir(rir, early_rir_sec=0.01, fs=16000)

    full, early = urgent.add_paired_reverberation(
        speech, rir, early_rir_sec=0.01, fs=16000, normalize_rir=False
    )
    full_oracle = scipy.signal.convolve(speech, rir, mode="full")[:, :speech_len]
    early_oracle = scipy.signal.convolve(speech, early_rir, mode="full")[:, :speech_len]

    np.testing.assert_allclose(full, full_oracle, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(early, early_oracle, rtol=1e-4, atol=1e-4)


def test_static_raw_mode_preserves_gain_across_rirs():
    speech = np.zeros((1, 4), dtype=np.float32)
    speech[0, 0] = 1.0
    quiet = np.array([[0.25, 0.0]], dtype=np.float32)
    loud = np.array([[2.0, 0.0]], dtype=np.float32)

    quiet_full, _ = urgent.add_paired_reverberation(
        speech, quiet, early_rir_sec=0.0, fs=16000, normalize_rir=False
    )
    loud_full, _ = urgent.add_paired_reverberation(
        speech, loud, early_rir_sec=0.0, fs=16000, normalize_rir=False
    )

    assert loud_full[0, 0] / quiet_full[0, 0] == pytest.approx(8.0)


def test_paired_fft_zero_near_zero_and_empty_inputs_preserve_contract():
    speech = np.arange(5, dtype=np.float32).reshape(1, -1)
    for rir in (
        np.zeros((1, 3), dtype=np.float32),
        np.full((1, 3), 1e-9, dtype=np.float32),
    ):
        full, early = urgent.add_paired_reverberation(
            speech, rir, early_rir_sec=0.05, fs=16000
        )
        expected = scipy.signal.convolve(speech, rir, mode="full")[:, :5]
        np.testing.assert_allclose(full, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(early, expected, rtol=2e-6, atol=2e-6)
        assert full.dtype == speech.dtype
        assert early.dtype == speech.dtype

    empty = np.empty((1, 0), dtype=np.float32)
    full, early = urgent.add_paired_reverberation(
        empty, np.ones((1, 3), dtype=np.float32), early_rir_sec=0.05, fs=16000
    )
    assert full.shape == empty.shape and early.shape == empty.shape
    assert full.dtype == empty.dtype and early.dtype == empty.dtype


def test_paired_fft_empty_outputs_follow_normal_result_type():
    speech = np.empty((1, 0), dtype=np.float32)
    rir = np.ones((1, 3), dtype=np.float64)
    full, early = urgent.add_paired_reverberation(
        speech, rir, early_rir_sec=0.05, fs=16000
    )
    expected_dtype = np.result_type(speech.dtype, rir.dtype)
    assert full.dtype == expected_dtype
    assert early.dtype == expected_dtype

    speech = np.ones((1, 4), dtype=np.float32)
    rir = np.empty((1, 0), dtype=np.float64)
    full, early = urgent.add_paired_reverberation(
        speech, rir, early_rir_sec=0.05, fs=16000
    )
    assert full.dtype == expected_dtype
    assert early.dtype == expected_dtype


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("bad_input", ["speech", "rir"])
def test_full_static_and_moving_rir_reject_nonfinite_samples(bad_value, bad_input):
    speech = np.ones((1, 8), dtype=np.float32)
    static_rir = np.ones((1, 3), dtype=np.float32)
    moving_rir = np.ones((2, 3), dtype=np.float32)

    if bad_input == "speech":
        speech[0, 0] = bad_value
    else:
        static_rir[0, 0] = bad_value
        moving_rir[0, 0] = bad_value

    with pytest.raises(ValueError, match="samples must be finite"):
        urgent.add_paired_reverberation(
            speech, static_rir, early_rir_sec=0.05, fs=16000
        )
    with pytest.raises(ValueError, match="samples must be finite"):
        urgent.add_reverberation(speech, static_rir)
    with pytest.raises(ValueError, match="samples must be finite"):
        urgent.add_moving_reverberation(
            speech, moving_rir, segment_sec=0.25, fs=16
        )


def test_moving_rir_rejects_nonfinite_explicit_normalization_peak():
    with pytest.raises(ValueError, match="normalization_peak must be finite"):
        urgent.add_moving_reverberation(
            np.ones((1, 8), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            segment_sec=0.25,
            fs=16,
            normalization_peak=np.nan,
        )


@pytest.mark.parametrize("rir_level", [0.0, 1e-9])
def test_static_and_moving_rir_keep_zero_and_near_zero_contract(rir_level):
    speech = np.zeros((1, 8), dtype=np.float32)
    speech[0, [0, 4]] = 1.0
    static_rir = np.array([[rir_level, 0.0]], dtype=np.float32)
    moving_rir = np.repeat(static_rir, 2, axis=0)

    static = urgent.add_reverberation(speech, static_rir)
    moving = urgent.add_moving_reverberation(
        speech,
        moving_rir,
        segment_sec=1.0,
        fs=4,
        normalize_rir=True,
    )

    expected = speech * rir_level
    np.testing.assert_allclose(static, expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(moving, expected, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize(
    "early_rir_sec,fs,error_type",
    [
        (-0.01, 16000, ValueError),
        (np.nan, 16000, ValueError),
        (np.inf, 16000, ValueError),
        (0.05, 0, ValueError),
        (0.05, -16000, ValueError),
        (0.05, np.inf, ValueError),
        (1e300, 1.0, ValueError),
        (1 + 0j, 16000, TypeError),
        (np.complex64(0.05 + 1j), 16000, TypeError),
        (0.05, 16000 + 0j, TypeError),
        (True, 16000, TypeError),
        (0.05, False, TypeError),
    ],
)
def test_early_rir_parameters_reject_invalid_values_before_empty_return(
    early_rir_sec, fs, error_type
):
    rir = np.ones((1, 3), dtype=np.float32)
    with pytest.raises(error_type):
        urgent.estimate_early_rir(rir, early_rir_sec=early_rir_sec, fs=fs)
    with pytest.raises(error_type):
        urgent.add_paired_reverberation(
            np.empty((1, 0), dtype=np.float32),
            rir,
            early_rir_sec=early_rir_sec,
            fs=fs,
        )


@pytest.mark.parametrize(
    ("normalize_rir", "expected_first", "expected_second"),
    [(True, 1.0, 0.25), (False, 2.0, 0.5)],
)
def test_moving_rir_uses_one_full_sequence_peak_and_preserves_raw_segment_gain(
    normalize_rir, expected_first, expected_second
):
    speech = np.zeros((1, 8), dtype=np.float32)
    speech[0, [0, 4]] = 1.0
    rir_seq = np.array([[2.0, 0.0], [0.5, 0.0]], dtype=np.float32)

    output = urgent.add_moving_reverberation(
        speech,
        rir_seq,
        segment_sec=1.0,
        fs=4,
        normalize_rir=normalize_rir,
    )

    assert output[0, 0] == pytest.approx(expected_first)
    assert output[0, 4] == pytest.approx(expected_second)
    assert output[0, 4] / output[0, 0] == pytest.approx(0.25)


def test_moving_early_sequence_reuses_full_sequence_peak():
    speech = np.zeros((1, 8), dtype=np.float32)
    speech[0, [0, 4]] = 1.0
    full_rir = np.array([[2.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    early_rir = np.array([[0.25, 0.0], [0.125, 0.0]], dtype=np.float32)
    full_peak = float(np.max(np.abs(full_rir)))

    early = urgent.add_moving_reverberation(
        speech,
        early_rir,
        segment_sec=1.0,
        fs=4,
        normalize_rir=True,
        normalization_peak=full_peak,
    )

    assert early[0, 0] == pytest.approx(0.125)
    assert early[0, 4] == pytest.approx(0.0625)


def _source_volume_mix_cfg(
    *,
    prob_reverberation: float = 1.0,
    target_type: str = "early",
    use_rms: bool = True,
):
    return SimpleNamespace(
        prob_reverberation=prob_reverberation,
        snr_low=0.0,
        snr_high=0.0,
        reuse_rir=False,
        rir_normalize=True,
        early_rir_sec=0.05,
        target_type=target_type,
        segment_sec=0.5,
        source_volume_normalization_enabled=True,
        source_volume_db_range=[-20.0, -20.0],
        source_volume_use_rms=use_rms,
        source_volume_distribution="uniform",
        source_volume_beta_a=2.0,
        source_volume_beta_b=2.0,
    )


def test_forced_source_volume_helper_does_not_consume_rng():
    signal = np.array([[1.0, -0.5, 0.25, 0.0]], dtype=np.float32)
    rng = np.random.default_rng(20260826)
    before = rng.bit_generator.state.copy()
    calibration = {}

    scaled, _ = dynamic_base.source_volume_normalize(
        signal,
        -30.0,
        -10.0,
        rng=rng,
        forced_gain=2.5,
        calibration_out=calibration,
    )

    assert rng.bit_generator.state == before
    np.testing.assert_array_equal(scaled, signal * 2.5)
    assert calibration["forced_gain"] is True
    assert calibration["gain"] == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("rir_mode", "prob_reverberation", "expected_full_to_early_ratio"),
    [("static", 1.0, 2.0), ("moving", 1.0, 2.0), ("none", 0.0, 1.0)],
)
def test_post_rir_source_volume_hits_fixed_early_target_for_all_rir_modes(
    monkeypatch, rir_mode, prob_reverberation, expected_full_to_early_ratio
):
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = _source_volume_mix_cfg(prob_reverberation=prob_reverberation)
    dataset._rir_reuse_cache = None
    dataset.export_simulation_stems = False
    dataset.strict_load = True
    dataset.rirs = {16000: ["rir-uid"]}
    dataset.rirs_flt = {"rir-uid": "rir-path"}
    dataset._select_sample = lambda sr, source, rng: "rir-uid"

    if rir_mode == "static":
        monkeypatch.setattr(
            dynamic_base,
            "_read_cached_2d",
            lambda *args, **kwargs: np.ones((1, 3), dtype=np.float32),
        )
        monkeypatch.setattr(
            dynamic_base,
            "add_paired_reverberation",
            lambda speech, *args, **kwargs: (speech * 4.0, speech * 2.0),
        )
    elif rir_mode == "moving":
        monkeypatch.setattr(
            dynamic_base,
            "_read_cached_2d",
            lambda *args, **kwargs: np.ones((2, 3), dtype=np.float32),
        )
        monkeypatch.setattr(dynamic_base, "_read_rir_meta_segment_sec", lambda path: None)
        moving_calls = []

        def moving_reverb(speech, *args, **kwargs):
            moving_calls.append(kwargs)
            return speech * (4.0 if len(moving_calls) == 1 else 2.0)

        monkeypatch.setattr(dynamic_base, "add_moving_reverberation", moving_reverb)

    speech = np.sin(2.0 * np.pi * 200.0 * np.arange(1600) / 16000.0).astype(
        np.float32
    )[None, :]
    output = dataset._mix_acoustic(
        speech,
        16000,
        speech.shape[1],
        speech.shape[1],
        np.random.default_rng(7),
        False,
    )

    active_rms = float(np.sqrt(np.mean(output.clean_target.astype(np.float64) ** 2)))
    assert active_rms == pytest.approx(10.0 ** (-20.0 / 20.0), abs=2e-6)
    assert np.linalg.norm(output.noisy_after_rir) / np.linalg.norm(
        output.clean_target
    ) == pytest.approx(expected_full_to_early_ratio)
    calibration = output.source_volume_calibration
    assert calibration["reference"] == "early_rir"
    assert calibration["stage"] == "post_rir_pre_noise_echo"
    assert calibration["target_db_requested"] == pytest.approx(-20.0)
    assert calibration["reference_db_after"] == pytest.approx(-20.0, abs=2e-4)
    assert calibration["gain_clipped"] is False
    assert calibration["status"] == "applied"
    assert calibration["reason"] is None
    assert len(output) == 12
    json.dumps(calibration, allow_nan=False)


@pytest.mark.parametrize("target_type", ["early", "dry"])
def test_full_early_and_dry_stems_share_source_gain_and_preserve_ratios(target_type):
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = _source_volume_mix_cfg(target_type=target_type, use_rms=False)
    dataset.export_simulation_stems = False
    observed = {}
    speech = np.array([[1.0, -0.5, 0.25, 0.0]], dtype=np.float32)
    full = speech * 4.0
    early = speech * 2.0
    dry = speech.copy()
    dataset._mix_acoustic_rir = lambda *args: (
        full.copy(),
        early.copy(),
        (dry if target_type == "dry" else early).copy(),
    )
    dataset._record_post_rir_calibration = lambda full_rir, early_ref, gain: observed.update(
        full=full_rir.copy(), early=early_ref.copy(), gain=gain
    )

    output = dataset._mix_acoustic(
        speech,
        16000,
        speech.shape[1],
        speech.shape[1],
        np.random.default_rng(0),
        False,
        forced_source_volume_gain=3.0,
    )

    np.testing.assert_array_equal(observed["full"], full * 3.0)
    np.testing.assert_array_equal(observed["early"], early * 3.0)
    np.testing.assert_array_equal(output.noisy_after_rir, full * 3.0)
    expected_target = dry if target_type == "dry" else early
    np.testing.assert_array_equal(output.clean_target, expected_target * 3.0)
    assert np.linalg.norm(observed["full"]) / np.linalg.norm(observed["early"]) == pytest.approx(2.0)
    assert np.linalg.norm(observed["early"]) / np.linalg.norm(output.clean_target) == pytest.approx(
        2.0 if target_type == "dry" else 1.0
    )
    calibration = output.source_volume_calibration
    assert observed["gain"] == pytest.approx(3.0)
    assert calibration["gain"] == pytest.approx(3.0)
    assert calibration["gain_clipped"] is False
    assert calibration["reason"] is None
    assert calibration["reference_db_after"] == pytest.approx(
        20.0 * np.log10(float(np.max(np.abs(early * 3.0))) + 1e-8)
    )


def test_dynamic_static_path_keeps_rng_and_uses_paired_helper(monkeypatch):
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimpleNamespace(
        reuse_rir=False,
        rir_normalize=True,
        early_rir_sec=0.05,
        target_type="early",
    )
    dataset._rir_reuse_cache = None
    dataset._active_acoustic_plan_record = {"noise_sources": [], "noise_assets": []}
    dataset.strict_load = True
    dataset.rirs = {16000: ["rir-uid"]}
    dataset.rirs_flt = {"rir-uid": "rir-path"}
    dataset._select_sample = lambda sr, source, rng: "rir-uid"
    rir = np.array([[1.0, 0.0, 0.25]], dtype=np.float32)
    monkeypatch.setattr(dynamic_base, "_read_cached_2d", lambda *args, **kwargs: rir.copy())
    calls = []

    def paired(*args, **kwargs):
        calls.append(kwargs)
        return np.full((1, 4), 2.0, dtype=np.float32), np.full((1, 4), 3.0, dtype=np.float32)

    monkeypatch.setattr(dynamic_base, "add_paired_reverberation", paired)
    rng = np.random.default_rng(123)
    before = rng.bit_generator.state
    noisy, reference, target = dataset._mix_acoustic_rir(
        np.ones((1, 4), dtype=np.float32), 16000, rng, True
    )
    after = rng.bit_generator.state
    assert calls == [{
        "early_rir_sec": 0.05,
        "fs": 16000,
        "normalize_rir": True,
    }]
    np.testing.assert_array_equal(noisy, 2.0)
    np.testing.assert_array_equal(reference, 3.0)
    np.testing.assert_array_equal(target, 3.0)
    plan = dataset._active_acoustic_plan_record["rir_plan"]
    assert plan.to_dict() == {
        "enabled": True,
        "uid": "rir-uid",
        "path": "rir-path",
        "moving": False,
        "reuse_enabled": False,
        "cache_hit": False,
        "segment_sec": None,
        "normalize": True,
        "early_rir_sec": 0.05,
        "target_type": "early",
        "crop": None,
    }
    np.testing.assert_array_equal(
        dataset._active_acoustic_plan_record["rir_asset"], rir
    )
    assert before == after


def test_dynamic_moving_path_does_not_use_paired_helper(monkeypatch):
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimpleNamespace(
        reuse_rir=False,
        rir_normalize=True,
        early_rir_sec=0.05,
        target_type="dry",
        segment_sec=0.5,
    )
    dataset._rir_reuse_cache = None
    dataset.strict_load = True
    dataset.rirs = {16000: ["rir-uid"]}
    dataset.rirs_flt = {"rir-uid": "rir-path"}
    dataset._select_sample = lambda sr, source, rng: "rir-uid"
    moving = np.array([[2.0, 0.0, 0.25], [0.5, 0.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(dynamic_base, "_read_cached_2d", lambda *args, **kwargs: moving.copy())
    monkeypatch.setattr(dynamic_base, "add_paired_reverberation", lambda *a, **k: (_ for _ in ()).throw(AssertionError("paired path used")))
    calls = []

    def moving_reverb(speech, rir, *args, **kwargs):
        calls.append(kwargs)
        return speech.copy()

    monkeypatch.setattr(dynamic_base, "add_moving_reverberation", moving_reverb)
    noisy, reference, target = dataset._mix_acoustic_rir(
        np.ones((1, 4), dtype=np.float32), 16000, np.random.default_rng(0), True
    )
    assert calls == [
        {"normalize_rir": True, "normalization_peak": 2.0},
        {"normalize_rir": True, "normalization_peak": 2.0},
    ]
    np.testing.assert_array_equal(noisy, 1.0)
    np.testing.assert_array_equal(reference, 1.0)
    np.testing.assert_array_equal(target, 1.0)


def test_dynamic_moving_reuse_preserves_resolved_metadata_segment_sec(monkeypatch):
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimpleNamespace(
        reuse_rir=True,
        rir_normalize=True,
        early_rir_sec=0.05,
        target_type="early",
        segment_sec=0.5,
    )
    dataset._rir_reuse_cache = None
    dataset.strict_load = True
    dataset.rirs = {16000: ["moving-rir"]}
    dataset.rirs_flt = {"moving-rir": "moving-rir.pkl"}
    moving = np.array([[2.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    select_calls = []
    read_calls = []
    metadata_calls = []
    reverb_calls = []

    def select_rir(sr, source, rng):
        select_calls.append((sr, source))
        rng.integers(0, 1000)
        return "moving-rir"

    def read_rir(path, sr, **kwargs):
        read_calls.append((path, sr))
        return moving.copy()

    def read_metadata(path):
        metadata_calls.append(path)
        return 0.25

    def moving_reverb(speech, rir, segment_sec, fs, **kwargs):
        reverb_calls.append((rir.copy(), segment_sec, fs))
        return speech.copy()

    dataset._select_sample = select_rir
    monkeypatch.setattr(dynamic_base, "_read_cached_2d", read_rir)
    monkeypatch.setattr(dynamic_base, "_read_rir_meta_segment_sec", read_metadata)
    monkeypatch.setattr(dynamic_base, "add_moving_reverberation", moving_reverb)

    rng = np.random.default_rng(123)
    expected_rng = np.random.default_rng(123)
    expected_rng.integers(0, 1000)
    speech = np.ones((1, 4), dtype=np.float32)
    dataset._mix_acoustic_rir(speech, 16000, rng, True)
    dataset._mix_acoustic_rir(speech, 16000, rng, True)

    assert select_calls == [(16000, dataset.rirs)]
    assert read_calls == [("moving-rir.pkl", 16000)]
    assert metadata_calls == ["moving-rir.pkl"]
    assert dataset._rir_reuse_cache[2:] == (0.25, "moving-rir")
    assert rng.bit_generator.state == expected_rng.bit_generator.state
    assert [segment_sec for _, segment_sec, _ in reverb_calls] == [0.25] * 4
    assert [fs for _, _, fs in reverb_calls] == [16000] * 4
    np.testing.assert_array_equal(reverb_calls[0][0], reverb_calls[2][0])
    np.testing.assert_array_equal(reverb_calls[1][0], reverb_calls[3][0])


def test_dynamic_moving_reuse_accepts_legacy_two_item_cache(monkeypatch):
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimpleNamespace(
        reuse_rir=True,
        rir_normalize=False,
        early_rir_sec=0.05,
        target_type="early",
        segment_sec=0.5,
    )
    moving = np.ones((2, 3), dtype=np.float32)
    dataset._rir_reuse_cache = (16000, moving.copy())
    dataset._select_sample = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("legacy cache should be reused")
    )
    monkeypatch.setattr(
        dynamic_base,
        "_read_rir_meta_segment_sec",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy cache has no RIR identity")
        ),
    )
    segment_secs = []

    def moving_reverb(speech, rir, segment_sec, fs, **kwargs):
        segment_secs.append(segment_sec)
        return speech.copy()

    monkeypatch.setattr(dynamic_base, "add_moving_reverberation", moving_reverb)
    dataset._mix_acoustic_rir(
        np.ones((1, 4), dtype=np.float32),
        16000,
        np.random.default_rng(0),
        True,
    )

    assert segment_secs == [0.5, 0.5]
