"""Phase 3B shared dynamic facade 的分段合同测试。"""

from __future__ import annotations

from copy import deepcopy
from types import MethodType

import numpy as np
import pytest

from core.data.dynamic_dataset_base import (
    AcousticBundleResult,
    BaseDynamicMixingDataset,
    MixAcousticOutput,
    PreparedSpeech,
)
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation.speech_preparation import SpeechPreparation
from core.data.simulation_trace import SourceSimulationTrace
from core.data.simulation_utils import align_to_max_duration
from core.data.time_transform import POLICY_FINAL_CENTER_CROP, TileOp
from core.data.waveform_bundle import WaveformBundle


pytestmark = pytest.mark.unit


def _layer_a_dataset(speech: np.ndarray) -> BaseDynamicMixingDataset:
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.speech_preparation = SpeechPreparation()
    dataset.sim_cfg = SimulationConfigs.from_dict(
        {
            "silence_pad_prob": 1.0,
            "silence_pad_position": "front",
            "use_high_pass": False,
        }
    )
    dataset.speech_flt = {"speech": "speech.wav"}
    dataset._read_one_speech = lambda *_args, **_kwargs: speech.copy()
    dataset.repeat_to_max_duration = True
    dataset.repeat_tile_silence_gap = False
    dataset.repeat_tile_silence_gap_range = (0.02, 0.2)
    dataset.export_simulation_stems = True
    dataset._last_simulation_stems = None
    return dataset


def test_prepared_speech_result_is_stable_and_replays_rng_state():
    speech = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result_dataset = _layer_a_dataset(speech)
    legacy_dataset = _layer_a_dataset(speech)
    result_rng = np.random.default_rng(20260826)
    legacy_rng = np.random.default_rng(20260826)

    result = result_dataset._prepare_clean_speech_result(
        "speech", 10, 20, result_rng
    )
    replay = legacy_dataset._prepare_clean_speech_result(
        "speech", 10, 20, legacy_rng
    )

    assert isinstance(result, PreparedSpeech)
    assert result.actual_len == replay.actual_len == 20
    np.testing.assert_array_equal(result.speech_1d, replay.speech_1d)
    np.testing.assert_array_equal(result.speech_2d, replay.speech_2d)
    assert result.fixed_silence_canvas_active is True
    assert result_dataset._fixed_silence_canvas_active is True
    assert legacy_dataset._fixed_silence_canvas_active is True
    assert result.repeat_export["align_mode"] == "silence_fixed_canvas"
    assert result_rng.bit_generator.state == legacy_rng.bit_generator.state


def test_prepared_speech_bundle_is_one_explicit_readonly_stage_snapshot():
    dataset = _layer_a_dataset(np.array([0.25, -0.5], dtype=np.float32))
    result = dataset._prepare_clean_speech_result(
        "speech", 10, 8, np.random.default_rng(7)
    )

    bundle = result.to_bundle()

    assert bundle.num_samples == result.actual_len
    np.testing.assert_array_equal(bundle.require("speech"), result.speech_2d)
    assert bundle.require("speech").dtype == np.float32
    assert not bundle.require("speech").flags.writeable
    assert bundle.metadata["fixed_silence_canvas_active"] is True
    assert result.speech_2d.flags.writeable


def _install_fake_acoustic(dataset: BaseDynamicMixingDataset) -> None:
    def fake_mix(
        _self,
        speech_2d,
        _sr,
        actual_len,
        _max_dur_sr,
        rng,
        _need_noise,
        *,
        forced_source_volume_gain=None,
    ):
        del forced_source_volume_gain
        offset = np.float32(rng.uniform(0.0, 1.0))
        speech = np.asarray(speech_2d, dtype=np.float32)
        assert actual_len == speech.shape[-1]
        return MixAcousticOutput(
            noisy_speech=speech + offset,
            clean_target=speech + np.float32(2.0),
            noisy_after_rir=speech + np.float32(3.0),
            additive_noise_2d=np.full_like(speech, offset),
            additive_snr_db_list=[float(offset)],
            used_sidechain_wind=False,
            mix_two=False,
            want_sidechain=False,
            wn_snr=0.0,
            snr_primary_drawn_db=float(offset),
            wind_noise_snr_candidate_db=None,
            use_reverb=False,
        )

    dataset._mix_acoustic = MethodType(fake_mix, dataset)


def test_acoustic_bundle_facade_preserves_output_and_rng_cursor():
    speech = np.arange(5, dtype=np.float32).reshape(1, -1)
    legacy_dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    facade_dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    _install_fake_acoustic(legacy_dataset)
    _install_fake_acoustic(facade_dataset)
    legacy_rng = np.random.default_rng(11)
    facade_rng = np.random.default_rng(11)

    legacy = legacy_dataset._mix_acoustic(
        speech, 16_000, speech.shape[-1], 16_000, legacy_rng, True
    )
    result = facade_dataset._mix_acoustic_bundle(
        WaveformBundle({"speech": speech}),
        16_000,
        16_000,
        facade_rng,
        True,
    )

    assert isinstance(result, AcousticBundleResult)
    assert isinstance(result.acoustic_output, MixAcousticOutput)
    assert len(result.acoustic_output) == len(legacy) == 12
    assert len(tuple(result.acoustic_output)) == 12
    for role, expected in {
        "noisy_speech": legacy.noisy_speech,
        "clean_target": legacy.clean_target,
        "noisy_after_rir": legacy.noisy_after_rir,
        "additive_noise": legacy.additive_noise_2d,
    }.items():
        np.testing.assert_array_equal(result.bundle.require(role), expected)
    assert facade_rng.bit_generator.state == legacy_rng.bit_generator.state


def test_acoustic_bundle_trace_adapter_never_changes_sample_interface():
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    _install_fake_acoustic(dataset)
    sample = (np.ones(3, dtype=np.float32), 16_000, 3)
    enabled_trace = SourceSimulationTrace("scene", capture_stems=True)
    disabled_trace = SourceSimulationTrace("disabled", capture_stems=False)
    speech = WaveformBundle({"speech": np.ones((1, 3), dtype=np.float32)})

    result = dataset._mix_acoustic_bundle(
        speech,
        16_000,
        16_000,
        np.random.default_rng(2),
        False,
        source_trace=enabled_trace,
    )
    result.record_stems(disabled_trace)

    assert len(sample) == 3
    assert set(enabled_trace.stems) == {
        "noisy_speech",
        "clean_target",
        "noisy_after_rir",
        "additive_noise",
    }
    assert not disabled_trace.stems


def _final_dataset(*, max_duration: int = 8) -> BaseDynamicMixingDataset:
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.repeat_to_max_duration = True
    dataset.repeat_tile_silence_gap = True
    dataset.repeat_tile_silence_gap_range = (0.2, 0.2)
    dataset._fixed_silence_canvas_active = False
    dataset._max_duration_for_sr = lambda _sr: max_duration
    dataset.export_simulation_stems = False
    dataset._last_simulation_stems = None
    return dataset


def test_final_tile_planner_owns_draws_and_apply_is_rng_free():
    dataset = _final_dataset()
    speech = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    noisy = speech + 10.0
    rng = np.random.default_rng(99)

    plan = dataset._build_final_time_plan(speech, noisy, 10, rng)
    state_after_plan = deepcopy(rng.bit_generator.state)
    clean_out, noisy_out, length, _ = dataset._apply_final_time_plan(
        plan, speech, noisy
    )

    assert rng.bit_generator.state == state_after_plan
    assert isinstance(plan.transform.operations[0], TileOp)
    assert plan.transform.policy == "repeat_to_max_duration"
    assert length == len(clean_out) == len(noisy_out) == 8
    np.testing.assert_array_equal(noisy_out[clean_out != 0], clean_out[clean_out != 0] + 10.0)
    np.testing.assert_array_equal(noisy_out[clean_out == 0], 0.0)


def test_final_center_crop_is_distinct_policy_and_consumes_zero_draws():
    dataset = _final_dataset(max_duration=4)
    speech = np.arange(8, dtype=np.float32)
    rng = np.random.default_rng(123)
    before = deepcopy(rng.bit_generator.state)

    plan = dataset._build_final_time_plan(speech, speech.copy(), 16_000, rng)
    clean, noisy, length, _ = dataset._apply_final_time_plan(
        plan, speech, speech.copy()
    )

    assert rng.bit_generator.state == before
    assert plan.transform.policy == POLICY_FINAL_CENTER_CROP
    np.testing.assert_array_equal(clean, speech[2:6])
    np.testing.assert_array_equal(noisy, clean)
    assert length == 4


def test_fixed_canvas_explicit_plan_value_preserves_fail_fast_contract():
    dataset = _final_dataset(max_duration=8)
    dataset._fixed_silence_canvas_active = False

    with pytest.raises(RuntimeError, match="refusing to tile"):
        dataset._build_final_time_plan(
            np.ones(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            10,
            np.random.default_rng(1),
            fixed_silence_canvas_active=True,
        )


def test_align_empty_extras_share_zero_canvas_and_legacy_arity():
    empty = np.array([], dtype=np.float32)
    extra = np.array([], dtype=np.float64)

    speech, noisy, extras = align_to_max_duration(
        empty, empty.copy(), 6, extra_arrays=[extra]
    )
    base = align_to_max_duration(empty, empty.copy(), 6)

    assert len(base) == 2
    assert len((speech, noisy, extras)) == 3
    for waveform in (speech, noisy, extras[0]):
        assert waveform.shape == (6,)
        assert waveform.dtype == np.float32
        np.testing.assert_array_equal(waveform, 0.0)
