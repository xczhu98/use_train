"""AEC Layer A 固定 silence canvas 专项回归。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

from tasks.aec.data.collate import aec_collate_fn
from tasks.aec.data.dynamic_dataset import DynamicMixingDataset
import core.data.dynamic_dataset_base as dynamic_dataset_base
from core.data.post_mix_pipeline import POST_MIX_STAGE_ORDER
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation.speech_preparation import SpeechPreparation
import core.data.simulation.speech_preparation as speech_preparation
from core.data.simulation_trace import SimulationTrace
import core.data.simulation_utils as simulation_utils
from core.data.simulation_utils import (
    align_speech_with_silence_pad,
    apply_silence_position,
)
from core.data.temporal_config import CanvasSpec

pytestmark = [pytest.mark.unit]


def _align(speech, *, seed=1, fixed=True, prob=1.0, max_len=20):
    return align_speech_with_silence_pad(
        np.asarray(speech, dtype=np.float32),
        sr=10,
        max_dur_sr=max_len,
        silence_pad_prob=prob,
        silence_pad_total_range=[0.2, 0.2],
        silence_pad_position="scene_aware",
        repeat_to_max_duration=True,
        rng=np.random.default_rng(seed),
        fixed_silence_canvas=fixed,
        return_silence_pad_applied=True,
    )


def test_fixed_canvas_short_speech_occurs_once_and_is_deterministic():
    speech = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
    first, applied = _align(speech, seed=7)
    second, _ = _align(speech, seed=7)

    assert applied is True
    assert len(first) == 20
    np.testing.assert_array_equal(first, second)
    starts = np.flatnonzero(first == speech[0])
    assert starts.size == 1
    np.testing.assert_array_equal(first[starts[0] : starts[0] + len(speech)], speech)
    assert np.count_nonzero(first) == len(speech)


@pytest.mark.parametrize("crop_start", [0, 22])
def test_fixed_canvas_long_speech_is_one_contiguous_crop(monkeypatch, crop_start):
    speech = np.arange(1, 31, dtype=np.float32)
    crop_bounds = []

    def select_crop_boundary(_rng, lo, hi):
        crop_bounds.append((lo, hi))
        return crop_start

    monkeypatch.setattr(simulation_utils, "rng_randint", select_crop_boundary)
    out = apply_silence_position(
        speech, 10, 8, 1.0, [0.2, 0.2], "scene_aware",
        rng=np.random.default_rng(9), fixed_canvas=True,
    )

    assert crop_bounds == [(0, 22)]
    np.testing.assert_array_equal(out, speech[crop_start : crop_start + 8])


def test_default_adaptive_padding_and_mode_a_tile_remain_compatible():
    speech = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    adaptive, applied = _align(speech, fixed=False, max_len=20)
    tiled, mode_b = _align(speech, prob=0.0, max_len=20)

    assert applied is True and len(adaptive) == 5
    assert mode_b is False and len(tiled) == 20
    assert np.count_nonzero(tiled) == 20


def test_preselected_mode_reuses_decision_without_second_bernoulli():
    speech = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out, applied = align_speech_with_silence_pad(
        speech, 10, 20, 0.0, [0.2, 0.2], "front",
        rng=np.random.default_rng(2), fixed_silence_canvas=True,
        return_silence_pad_applied=True, preselected_silence_pad=True,
    )
    assert applied is True
    assert len(out) == 20
    assert np.count_nonzero(out) == 3


def _bare_aec_dataset(speech: np.ndarray) -> DynamicMixingDataset:
    ds = DynamicMixingDataset.__new__(DynamicMixingDataset)
    ds.speech_preparation = SpeechPreparation()
    ds.sim_cfg = SimulationConfigs.from_dict({
        "silence_pad_prob": 1.0,
        "silence_pad_position": "scene_aware",
        "use_high_pass": False,
    })
    ds.speech_flt = {"u": "main"}
    ds._read_one_speech = lambda *_args, **_kwargs: speech.copy()
    ds.repeat_to_max_duration = True
    ds.repeat_tile_silence_gap = True
    ds.repeat_tile_silence_gap_range = (0.02, 0.2)
    ds.export_simulation_stems = True
    ds._last_simulation_stems = None
    ds.max_duration_seconds = 2.0
    ds.max_duration = 20
    ds.canvas_spec = CanvasSpec(
        max_duration_seconds=2.0,
        max_duration_samples=20,
        repeat_to_max_duration=True,
        repeat_tile_silence_gap=True,
        repeat_tile_silence_gap_range=(0.02, 0.2),
    )
    return ds


def test_aec_opt_in_audit_tracks_fixed_canvas_and_all_context_is_full_length():
    ds = _bare_aec_dataset(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    prepared = ds._prepare_clean_speech_result("u", 10, 20, np.random.default_rng(3))
    nearend = prepared.speech_1d
    length = prepared.actual_len
    farend = np.full(20, 0.25, dtype=np.float32)
    ctx = {"prob_target_echo_disk": {
        "farend": farend, "echo": farend.copy(),
    }}
    mic = nearend + farend
    plan = ds._build_final_time_plan(nearend, mic, 10, np.random.default_rng(4))
    clean_out, mic_out, out_len, ctx_out = ds._apply_final_time_plan(
        plan, nearend, mic, ctx=ctx
    )
    ds._enrich_repeat_export_final(ds._last_simulation_stems["repeat_export"], 10, length, out_len)

    assert length == out_len == 20
    assert np.count_nonzero(clean_out) == 4
    assert np.any(mic_out[clean_out == 0] != 0)
    assert all(len(ctx_out["prob_target_echo_disk"][key]) == 20 for key in ("farend", "echo"))
    audit = ds._last_simulation_stems["repeat_export"]
    assert audit["align_mode"] == "silence_fixed_canvas"
    assert audit["final_align_mode"] == "already_fixed_canvas"


def test_fixed_canvas_final_short_output_fails_instead_of_tiling():
    ds = _bare_aec_dataset(np.ones(4, dtype=np.float32))
    ds._fixed_silence_canvas_active = True
    with pytest.raises(RuntimeError, match="refusing to tile"):
        ds._build_final_time_plan(
            np.ones(5, dtype=np.float32), np.ones(5, dtype=np.float32),
            10, np.random.default_rng(5),
        )


def test_mode_a_exact_max_audit_does_not_report_crop():
    ds = _bare_aec_dataset(np.arange(1, 21, dtype=np.float32))
    ds.sim_cfg = SimulationConfigs.from_dict({
        "silence_pad_prob": 0.0,
        "use_high_pass": False,
    })

    length = ds._prepare_clean_speech_result("u", 10, 20, np.random.default_rng(10)).actual_len

    assert length == 20
    assert ds._last_simulation_stems["repeat_export"]["align_mode"] == "off"


def test_fixed_canvas_final_overlong_output_reports_crop():
    ds = _bare_aec_dataset(np.ones(4, dtype=np.float32))
    ds._fixed_silence_canvas_active = True
    speech = np.arange(24, dtype=np.float32)
    audit = {}

    noisy_in = speech.copy()
    plan = ds._build_final_time_plan(speech, noisy_in, 10, np.random.default_rng(11))
    clean, noisy, length, _ = ds._apply_final_time_plan(plan, speech, noisy_in)
    ds._enrich_repeat_export_final(audit, 10, len(speech), length)

    assert length == 20
    np.testing.assert_array_equal(clean, speech[2:22])
    np.testing.assert_array_equal(noisy, clean)
    assert audit["final_align_mode"] == "crop_to_max"


def test_mode_a_still_tiles_aec_context_with_shared_schedule():
    ds = _bare_aec_dataset(np.ones(4, dtype=np.float32))
    ds._fixed_silence_canvas_active = False
    ds.repeat_tile_silence_gap = False
    speech = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    farend = speech * 10.0
    ctx = {"prob_target_echo_disk": {"farend": farend}}

    noisy_in = speech.copy()
    plan = ds._build_final_time_plan(speech, noisy_in, 10, np.random.default_rng(6))
    clean, noisy, length, out_ctx = ds._apply_final_time_plan(
        plan, speech, noisy_in, ctx=ctx
    )

    assert length == 20
    np.testing.assert_array_equal(clean, noisy)
    np.testing.assert_array_equal(out_ctx["prob_target_echo_disk"]["farend"], clean * 10.0)


@pytest.mark.parametrize(
    ("silence_pad_prob", "expected_reads", "expects_extra"),
    [(1.0, ["main"], False), (0.0, ["main", "extra"], True)],
)
def test_aec_splice_uses_single_source_only_for_mode_b(
    silence_pad_prob, expected_reads, expects_extra,
):
    ds = _bare_aec_dataset(np.ones(1, dtype=np.float32))
    ds.sim_cfg = SimulationConfigs.from_dict({
        "speech_splice": True,
        "speech_splice_seconds": 1.0,
        "speech_splice_gap_range": [0.1, 0.1],
        "speech_splice_equal_volume": False,
        "speech_splice_max_repeat_times": 1,
        "silence_pad_prob": silence_pad_prob,
        "silence_pad_position": "scene_aware",
        "use_high_pass": False,
    })
    ds.repeat_tile_silence_gap = False
    ds.speech_flt = {"u": "main"}
    ds._all_speech_paths = ["extra"]
    reads = []
    waveforms = {
        "main": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "extra": np.arange(11.0, 17.0, dtype=np.float32),
    }

    def read_one(path, _sr):
        reads.append(path)
        return waveforms[path].copy()

    ds._read_one_speech = read_one
    prepared = ds._prepare_clean_speech_result("u", 10, 20, np.random.default_rng(8))
    clean = prepared.speech_1d
    length = prepared.actual_len

    assert reads == expected_reads
    assert length == 20
    assert bool(np.any(clean == 11.0)) is expects_extra
    if silence_pad_prob == 1.0:
        assert np.count_nonzero(clean) == 3


@pytest.mark.parametrize(
    ("mode_draw", "expected_reads", "expected_fixed_canvas"),
    [
        (0.25, ["main"], True),
        (0.75, ["main", "extra"], False),
    ],
)
def test_production_mode_bernoulli_is_drawn_once_and_reused(
    monkeypatch, mode_draw, expected_reads, expected_fixed_canvas,
):
    ds = _bare_aec_dataset(np.ones(1, dtype=np.float32))
    ds.sim_cfg = SimulationConfigs.from_dict({
        "speech_splice": True,
        "speech_splice_seconds": 1.0,
        "speech_splice_gap_range": [0.1, 0.1],
        "speech_splice_equal_volume": False,
        "speech_splice_max_repeat_times": 1,
        "silence_pad_prob": 0.5,
        "silence_pad_position": "front",
        "use_high_pass": False,
    })
    ds.repeat_tile_silence_gap = False
    ds.speech_flt = {"u": "main"}
    ds._all_speech_paths = ["extra"]
    reads = []
    waveforms = {
        "main": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "extra": np.arange(11.0, 17.0, dtype=np.float32),
    }
    base_draws = []
    helper_draws = []
    original_helper_random = simulation_utils.rng_random

    def controlled_mode_draw(_rng):
        base_draws.append(mode_draw)
        return mode_draw

    def record_helper_draw(rng):
        helper_draws.append(None)
        return original_helper_random(rng)

    def read_one(path, _sr):
        reads.append(path)
        return waveforms[path].copy()

    monkeypatch.setattr(speech_preparation, "rng_random", controlled_mode_draw)
    monkeypatch.setattr(simulation_utils, "rng_random", record_helper_draw)
    ds._read_one_speech = read_one

    prepared = ds._prepare_clean_speech_result(
        "u", 10, 20, np.random.default_rng(12),
    )
    clean = prepared.speech_1d
    length = prepared.actual_len

    assert base_draws == [mode_draw]
    assert helper_draws == []
    assert reads == expected_reads
    assert ds._fixed_silence_canvas_active is expected_fixed_canvas
    assert length == 20
    assert bool(np.any(clean == 11.0)) is (not expected_fixed_canvas)


def test_single_utterance_uses_one_contiguous_source_without_extra_reads():
    ds = DynamicMixingDataset.__new__(DynamicMixingDataset)
    ds.speech_preparation = SpeechPreparation()
    ds.sim_cfg = SimulationConfigs.from_dict({
        "speech_splice": True,
        "speech_splice_seconds": 0.4,
        "speech_random_start": False,
    })
    ds.speech_flt = {"u": "main"}
    ds._all_speech_paths = ["extra"]
    reads = []

    def read_one(path, _sr):
        reads.append(path)
        return np.arange(1.0, 9.0, dtype=np.float32)

    ds._read_one_speech = read_one
    clean = ds._get_clean_speech(
        "u", 10, rng=np.random.default_rng(9), single_utterance=True,
    )

    assert reads == ["main"]
    assert len(clean) in (4, 8)
    np.testing.assert_array_equal(
        clean,
        np.arange(1.0, len(clean) + 1.0, dtype=np.float32),
    )


def _write_source_scp(path: Path, entries: dict[str, Path]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for uid, audio_path in entries.items():
            info = soundfile.info(str(audio_path))
            handle.write(f"{uid} {info.samplerate} {info.frames} {audio_path}\n")


@pytest.fixture
def make_e2e_aec_dataset(tmp_path: Path):
    """构造可走完整 AEC 生产流水线的长短语音与独立背景 stems。"""
    sr = 8000
    canvas_samples = sr
    speech_samples = sr // 4
    long_speech_samples = canvas_samples + canvas_samples // 2
    t_speech = np.arange(speech_samples, dtype=np.float32) / sr
    t_canvas = np.arange(canvas_samples, dtype=np.float32) / sr

    main_wav = tmp_path / "main.wav"
    long_main_wav = tmp_path / "long_main.wav"
    extra_wav = tmp_path / "extra.wav"
    noise_wav = tmp_path / "noise.wav"
    rir_wav = tmp_path / "rir.wav"
    farend_wav = tmp_path / "farend.wav"
    echo_wav = tmp_path / "echo.wav"
    soundfile.write(
        main_wav,
        0.25 * np.sin(2.0 * np.pi * 310.0 * t_speech),
        sr,
        subtype="FLOAT",
    )
    soundfile.write(
        long_main_wav,
        np.arange(1, long_speech_samples + 1, dtype=np.float32) / 32768.0,
        sr,
        subtype="FLOAT",
    )
    soundfile.write(
        extra_wav,
        0.2 * np.sin(2.0 * np.pi * 530.0 * t_speech),
        sr,
        subtype="FLOAT",
    )
    soundfile.write(
        noise_wav,
        0.08 * np.sin(2.0 * np.pi * 877.0 * t_canvas + 0.3),
        sr,
        subtype="FLOAT",
    )
    rir = np.zeros(sr // 20, dtype=np.float32)
    rir[0] = 1.0
    soundfile.write(rir_wav, rir, sr, subtype="FLOAT")
    farend = 0.18 * np.sin(2.0 * np.pi * 421.0 * t_canvas + 0.2)
    echo = 0.12 * np.sin(2.0 * np.pi * 421.0 * t_canvas + 0.7)
    soundfile.write(farend_wav, farend, sr, subtype="FLOAT")
    soundfile.write(echo_wav, echo, sr, subtype="FLOAT")
    stereo_wav = tmp_path / "stereo.wav"
    soundfile.write(stereo_wav, np.stack([echo, farend], axis=1), sr, subtype="FLOAT")

    speech_scp = tmp_path / "speech.scp"
    long_speech_scp = tmp_path / "long_speech.scp"
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    pair_json = tmp_path / "farend_echo.json"
    _write_source_scp(speech_scp, {"main": main_wav, "extra": extra_wav})
    _write_source_scp(
        long_speech_scp,
        {"main": long_main_wav, "extra": extra_wav},
    )
    _write_source_scp(noise_scp, {"noise": noise_wav})
    _write_source_scp(rir_scp, {"rir": rir_wav})
    pair_json.write_text(
        json.dumps({"pairs": [str(stereo_wav)]}),
        encoding="utf-8",
    )

    def _build(
        *,
        silence_pad_prob: float,
        state_labels: bool = False,
        speech_splice_seconds: float = 0.75,
        long_main: bool = False,
    ):
        simulation_cfg = {
            "speech_splice": True,
            "speech_splice_seconds": speech_splice_seconds,
            "speech_splice_gap_range": [0.01, 0.01],
            "speech_splice_equal_volume": False,
            "speech_splice_max_repeat_times": 3,
            "silence_pad_prob": silence_pad_prob,
            "silence_pad_position": "center",
            "prob_target_only": 0.0,
            "prob_noise_only": 0.0,
            "prob_echo_only": 0.0,
            "prob_noise_echo_only": 0.0,
            "prob_target_noise": 0.0,
            "prob_target_echo": 1.0,
            "prob_target_noise_echo": 0.0,
            "snr_low": 6.0,
            "snr_high": 6.0,
            "ser_range": [6.0, 6.0],
            # Layer-A 职责不约束 ERL；覆盖 physical range 以保持旧抽样。
            "erl_range": [-60.0, 60.0],
            "prob_channel_enabled": False,
            "prob_reverberation": 0.0,
            "prob_wind_noise_sidechain": 0.0,
            "prob_colored_noise": 0.0,
            "noisy_post_mix_degradations_enabled": False,
            "use_high_pass": False,
        }
        state_kwargs = (
            {
                "state_detector_hop_length": 80,
                "state_detector_win_length": 160,
                "state_detector_hangover_frames": 0,
            }
            if state_labels
            else {}
        )
        dataset = DynamicMixingDataset(
            speech_scp=str(long_speech_scp if long_main else speech_scp),
            noise_scp=str(noise_scp),
            rir_scp=str(rir_scp),
            simulation_cfg=simulation_cfg,
            real_farend_echo_pair_json=str(pair_json),
            aec_ref_echo_fine_gcc_align=False,
            aec_ref_echo_random_delay_prob=0.0,
            max_duration_seconds=1.0,
            repeat_to_max_duration=True,
            filter_by_duration=False,
            target_sr=sr,
            task="joint",
            strict_load=True,
            export_simulation_stems=True,
            **state_kwargs,
        )
        dataset.set_epoch(0, base_seed=37)
        return dataset

    return _build


def test_mode_b_constructor_allows_splice_target_over_max_duration(
    make_e2e_aec_dataset,
):
    dataset = make_e2e_aec_dataset(
        silence_pad_prob=1.0,
        speech_splice_seconds=2.0,
    )

    sample = dataset._getitem_simulation_verify(
        0, force_branch="prob_target_noise_echo",
    )

    assert int(sample[4]) == 8000
    assert dataset._last_simulation_stems["simulation_by_layer"]["A"][
        "a3_alignment_mode"
    ] == "silence_fixed_canvas"
    with pytest.raises(ValueError, match="speech_splice"):
        make_e2e_aec_dataset(
            silence_pad_prob=0.5,
            speech_splice_seconds=2.0,
        )


def test_aec_dataset_e2e_long_mode_b_is_one_crop_without_final_tile(
    make_e2e_aec_dataset,
):
    dataset = make_e2e_aec_dataset(
        silence_pad_prob=1.0,
        long_main=True,
    )
    speech_reads = []
    layer_a_outputs = []
    original_read = dataset._read_one_speech
    original_prepare = dataset.speech_preparation.prepare_result

    def _record_speech_read(path, sr):
        speech = original_read(path, sr)
        speech_reads.append((Path(path).name, speech.copy()))
        return speech

    def _record_layer_a_output(*args, **kwargs):
        output = original_prepare(*args, **kwargs)
        layer_a_outputs.append(output.speech_1d.copy())
        return output

    dataset._read_one_speech = _record_speech_read
    dataset.speech_preparation.prepare_result = _record_layer_a_output
    sample = dataset._getitem_simulation_verify(
        0, force_branch="prob_target_noise_echo",
    )

    assert len(speech_reads) == 1
    assert speech_reads[0][0] == "long_main.wav"
    assert len(layer_a_outputs) == 1
    source = speech_reads[0][1]
    layer_a_nearend = layer_a_outputs[0]
    assert len(source) == 12000
    assert len(layer_a_nearend) == 8000

    crop_starts = np.flatnonzero(source == layer_a_nearend[0])
    assert crop_starts.size == 1
    crop_start = int(crop_starts[0])
    np.testing.assert_array_equal(
        layer_a_nearend,
        source[crop_start : crop_start + len(layer_a_nearend)],
    )

    assert int(sample[4]) == 8000
    assert all(len(sample[index]) == 8000 for index in (0, 1, 2, 5))
    stems = dataset._last_simulation_stems
    repeat_audit = stems["repeat_export"]
    layer_a = stems["simulation_by_layer"]["A"]
    assert repeat_audit["samples_before_repeat_step"] == 12000
    assert repeat_audit["samples_after_repeat_step"] == 8000
    assert repeat_audit["align_mode"] == "silence_fixed_canvas"
    assert repeat_audit["samples_before_final_align"] == 8000
    assert repeat_audit["samples_after_final_align"] == 8000
    assert repeat_audit["final_align_mode"] == "already_fixed_canvas"
    assert layer_a["a3_mode"] == "fixed_silence_canvas"
    assert layer_a["a3_alignment_mode"] == "silence_fixed_canvas"
    assert layer_a["a3_fixed_canvas_applied"] is True
    assert layer_a["a3_tile_applied"] is False
    assert layer_a["repeat_to_max_duration_applied"] is False
    assert layer_a["samples_before_repeat"] == 12000
    assert layer_a["samples_after_repeat_before_hp"] == 8000
    assert layer_a["samples_after_A"] == 8000


@pytest.mark.parametrize(("state_labels", "tuple_length"), [(False, 11), (True, 12)])
def test_aec_dataset_e2e_fixed_canvas_preserves_independent_stems_and_tuple_contract(
    make_e2e_aec_dataset, state_labels, tuple_length,
):
    dataset = make_e2e_aec_dataset(
        silence_pad_prob=1.0,
        state_labels=state_labels,
    )
    speech_reads = []
    original_read = dataset._read_one_speech

    def _record_speech_read(path, sr):
        speech_reads.append(Path(path).name)
        return original_read(path, sr)

    dataset._read_one_speech = _record_speech_read
    sample = dataset._getitem_simulation_verify(0, force_branch="prob_target_noise_echo")

    assert len(sample) == tuple_length
    assert speech_reads == ["main.wav"]
    assert sample[3] == 8000
    assert int(sample[4]) == 8000
    assert all(len(sample[index]) == 8000 for index in (0, 1, 2, 5))

    batch = aec_collate_fn([sample])
    batch.validate()
    assert (batch.state is not None) is state_labels
    waves = (batch.nearend, batch.mic, batch.farend, batch.error)
    flags = (batch.error_mask, batch.is_pure_noise, batch.is_pure_echo, batch.is_pure_echo_noise, batch.has_echo)
    assert all(wav.shape == (1, 8000) for wav in waves)
    assert all(wav.dtype == torch.float32 for wav in waves)
    assert batch.fs.shape == torch.Size([])
    assert batch.fs.dtype == torch.int32
    assert batch.lengths.shape == (1,)
    assert batch.lengths.dtype == torch.int32
    assert all(flag.shape == (1,) for flag in flags)
    assert all(flag.dtype == torch.bool for flag in flags)
    if state_labels:
        state = sample[11]
        assert state.shape == (8000 // 80 + 1,)
        assert state.dtype == torch.int64
        assert batch.state.shape == (1, 8000 // 80 + 1)
        assert batch.state.dtype == torch.int64
        torch.testing.assert_close(batch.state[0], state)
        assert torch.unique(state).numel() >= 2
        assert torch.any((state & 4) != 0)
        assert torch.any(((state & 4) == 0) & ((state & 3) == 3))

    nearend = sample[0].numpy()
    mic = sample[1].numpy()
    farend = sample[2].numpy()
    stems = dataset._last_simulation_stems
    scene_metadata = dataset._last_simulation_trace.sources["scene"].metadata
    acoustic_plan = scene_metadata["acoustic_scene_plan"]
    assert acoustic_plan["schema_version"] == 1
    assert acoustic_plan["need_noise"] is True
    assert "aec_context" not in acoustic_plan
    assert not ({"ser", "erl", "delay", "clipping"} & set(acoustic_plan))
    assert scene_metadata["post_mix_stage_order"] == [
        "pre-finite",
        "D-normalize",
        "D-guard",
        "E",
        "F",
        "task-post-EF",
        "final-guard",
        "post-finite",
    ]
    assert scene_metadata["post_mix_stage_order"] == list(POST_MIX_STAGE_ORDER)
    assert scene_metadata["post_mix_stage_applied"]["D-normalize"] is False
    assert "D-guard" in scene_metadata["post_mix_stage_applied"]
    assert "aec_input_frontend" not in scene_metadata
    assert scene_metadata["final_time_transform"]["output_length"] == 8000
    echo = stems["final_echo"]
    noise = stems["final_noise"]
    nearend_silence = np.abs(nearend) < 1e-8
    assert nearend_silence.sum() > len(nearend) // 2
    for independent_signal in (farend, echo, noise, mic):
        assert np.any(np.abs(independent_signal[nearend_silence]) > 1e-6)

    repeat_audit = stems["repeat_export"]
    layer_a = stems["simulation_by_layer"]["A"]
    assert repeat_audit["align_mode"] == "silence_fixed_canvas"
    assert repeat_audit["final_align_mode"] == "already_fixed_canvas"
    assert layer_a["a3_mode"] == "fixed_silence_canvas"
    assert layer_a["a3_alignment_mode"] == "silence_fixed_canvas"
    assert layer_a["a3_fixed_canvas_applied"] is True
    assert layer_a["a3_tile_applied"] is False
    assert layer_a["repeat_to_max_duration_applied"] is False
    assert layer_a["samples_before_repeat"] == 2000
    assert layer_a["samples_after_A"] == 8000


def test_aec_dataset_e2e_mode_a_build_sample_keeps_tile_audit(
    make_e2e_aec_dataset,
):
    dataset = make_e2e_aec_dataset(silence_pad_prob=0.0)
    trace = SimulationTrace(
        index=0, uid="main", task="aec", capture_stems=True
    )
    completed = dataset._build_sample(
        0,
        "main",
        dataset.source_length["main"],
        8000,
        force_branch="prob_target_noise_echo",
        trace=trace,
    )

    assert completed.trace is trace
    assert not trace.sealed
    assert len(completed.sample) == 11
    assert int(completed.sample[4]) == 8000
    layer_a = trace.legacy_stems("scene")["simulation_by_layer"]["A"]
    assert layer_a["a3_mode"] == "standard"
    assert layer_a["a3_alignment_mode"] == "tile_to_max"
    assert layer_a["a3_fixed_canvas_applied"] is False
    assert layer_a["a3_tile_applied"] is True
    assert layer_a["repeat_to_max_duration_applied"] is True
    assert layer_a["samples_before_repeat"] == 6000
    assert layer_a["samples_after_A"] == 8000
