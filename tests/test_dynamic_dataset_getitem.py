"""End-to-end regression tests for SE/AEC DynamicMixingDataset.__getitem__.

These tests create minimal on-disk SCP / JSON manifests and synthetic WAV files
in a temporary directory, then exercise the full simulation pipeline. They guard
against regressions when refactoring the shared base class.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

from tasks.aec.data.dynamic_dataset import DynamicMixingDataset as AECDynamicMixingDataset
from core.data.errors import RecoverableResourceError
from tasks.aec.data.dynamic_dataset import validate_aec_dataset_config
from core.data.collate import collate_fn
from core.data.post_mix_pipeline import POST_MIX_STAGE_ORDER
from core.data.simulation_trace import SimulationTrace
from tasks.se.data.dynamic_dataset import DynamicMixingDataset as SEDynamicMixingDataset


def _write_scp(path: Path, entries: dict[str, Path]) -> None:
    """Write a source SCP file with ``uid fs num_samples path`` columns."""
    with path.open("w", encoding="utf-8") as f:
        for uid, audio_path in entries.items():
            info = soundfile.info(str(audio_path))
            f.write(f"{uid} {info.samplerate} {info.frames} {audio_path}\n")


def _make_wav(path: Path, duration: float, sr: int, *, seed: int = 0) -> None:
    """Write a synthetic mono WAV file (sine + small noise)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    signal = np.sin(2.0 * np.pi * 300.0 * t).astype(np.float32)
    signal += 0.01 * rng.standard_normal(signal.shape).astype(np.float32)
    signal = np.clip(signal, -0.99, 0.99)
    soundfile.write(path, signal, sr)


def _make_impulse_rir(path: Path, sr: int) -> None:
    """Write a trivial impulse RIR so reverb is identity-like."""
    rir = np.zeros(sr, dtype=np.float32)
    rir[0] = 1.0
    soundfile.write(path, rir, sr)


@pytest.fixture
def se_dataset(tmp_path: Path):
    """Build a small SE DynamicMixingDataset backed by synthetic files."""
    sr = 16000
    duration = 2.0

    speech_dir = tmp_path / "speech"
    noise_dir = tmp_path / "noise"
    rir_dir = tmp_path / "rir"
    wind_dir = tmp_path / "wind"
    for d in (speech_dir, noise_dir, rir_dir, wind_dir):
        d.mkdir()

    speech_wav = speech_dir / "spk1_001.wav"
    noise_wav = noise_dir / "noise_001.wav"
    rir_wav = rir_dir / "rir_001.wav"
    wind_wav = wind_dir / "wind_001.wav"

    _make_wav(speech_wav, duration, sr, seed=1)
    _make_wav(noise_wav, duration, sr, seed=2)
    _make_impulse_rir(rir_wav, sr)
    _make_wav(wind_wav, duration, sr, seed=3)

    speech_scp = tmp_path / "speech.scp"
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    wind_scp = tmp_path / "wind.scp"
    _write_scp(speech_scp, {"spk1_001": speech_wav})
    _write_scp(noise_scp, {"noise_001": noise_wav})
    _write_scp(rir_scp, {"rir_001": rir_wav})
    _write_scp(wind_scp, {"wind_001": wind_wav})

    simulation_cfg = {
        "prob_target_only": 0.2,
        "prob_noise_only": 0.2,
        "snr_low": 5.0,
        "snr_high": 20.0,
        "prob_channel_enabled": False,
        "noisy_post_mix_degradations_enabled": False,
        "prob_reverberation": 0.5,
    }

    ds = SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=str(wind_scp),
        simulation_cfg=simulation_cfg,
        max_duration_seconds=1.0,
        target_sr=sr,
        filter_by_duration=True,
        strict_load=True,
    )
    ds.set_epoch(0, base_seed=123)
    return ds


@pytest.fixture
def se_fixed_canvas_dataset(tmp_path: Path):
    """Build a short-speech SE dataset that must use the mode-B fixed canvas."""
    sr = 16000
    speech_dir = tmp_path / "speech"
    noise_dir = tmp_path / "noise"
    rir_dir = tmp_path / "rir"
    wind_dir = tmp_path / "wind"
    for directory in (speech_dir, noise_dir, rir_dir, wind_dir):
        directory.mkdir()

    main_wav = speech_dir / "main.wav"
    extra_wav = speech_dir / "extra.wav"
    noise_wav = noise_dir / "noise.wav"
    rir_wav = rir_dir / "rir.wav"
    wind_wav = wind_dir / "wind.wav"
    _make_wav(main_wav, 0.25, sr, seed=101)
    _make_wav(extra_wav, 0.25, sr, seed=102)
    _make_wav(noise_wav, 1.0, sr, seed=103)
    _make_impulse_rir(rir_wav, sr)
    _make_wav(wind_wav, 1.0, sr, seed=104)

    speech_scp = tmp_path / "speech.scp"
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    wind_scp = tmp_path / "wind.scp"
    _write_scp(speech_scp, {"main": main_wav, "extra": extra_wav})
    _write_scp(noise_scp, {"noise": noise_wav})
    _write_scp(rir_scp, {"rir": rir_wav})
    _write_scp(wind_scp, {"wind": wind_wav})

    ds = SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=str(wind_scp),
        simulation_cfg={
            "prob_target_only": 0.0,
            "prob_noise_only": 0.0,
            "prob_reverberation": 0.0,
            "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": False,
            "speech_splice": True,
            "speech_splice_seconds": 0.75,
            "speech_splice_max_repeat_times": 4,
            "silence_pad_prob": 1.0,
            "silence_pad_position": "center",
            "use_high_pass": False,
        },
        max_duration_seconds=1.0,
        repeat_to_max_duration=True,
        target_sr=sr,
        filter_by_duration=False,
        strict_load=True,
    )
    ds.set_epoch(0, base_seed=789)
    return ds, main_wav, extra_wav


@pytest.fixture
def se_long_fixed_canvas_dataset(tmp_path: Path):
    """构造长语音 SE 数据集，要求模式 B 只裁剪一个源窗口。"""
    sr = 16000
    speech_dir = tmp_path / "speech"
    noise_dir = tmp_path / "noise"
    rir_dir = tmp_path / "rir"
    wind_dir = tmp_path / "wind"
    for directory in (speech_dir, noise_dir, rir_dir, wind_dir):
        directory.mkdir()

    speech_wav = speech_dir / "long.wav"
    noise_wav = noise_dir / "noise.wav"
    rir_wav = rir_dir / "rir.wav"
    wind_wav = wind_dir / "wind.wav"
    _make_wav(speech_wav, 2.25, sr, seed=201)
    _make_wav(noise_wav, 1.0, sr, seed=202)
    _make_impulse_rir(rir_wav, sr)
    _make_wav(wind_wav, 1.0, sr, seed=203)

    speech_scp = tmp_path / "speech.scp"
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    wind_scp = tmp_path / "wind.scp"
    _write_scp(speech_scp, {"long": speech_wav})
    _write_scp(noise_scp, {"noise": noise_wav})
    _write_scp(rir_scp, {"rir": rir_wav})
    _write_scp(wind_scp, {"wind": wind_wav})

    ds = SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=str(wind_scp),
        simulation_cfg={
            "prob_target_only": 0.0,
            "prob_noise_only": 0.0,
            "prob_reverberation": 0.0,
            "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": False,
            "speech_splice": True,
            "speech_splice_seconds": 0.75,
            "speech_splice_max_repeat_times": 1,
            "silence_pad_prob": 1.0,
            "silence_pad_position": "center",
            "use_high_pass": False,
            "prob_speed_perturb": 0.0,
            "prob_pitch_shift": 0.0,
            "prob_eq_perturb": 0.0,
            "prob_add_extra_space": 0.0,
            "source_volume_normalization_enabled": False,
        },
        max_duration_seconds=1.0,
        repeat_to_max_duration=True,
        target_sr=sr,
        filter_by_duration=False,
        strict_load=True,
    )
    ds.set_epoch(0, base_seed=987)
    return ds, speech_wav


def test_se_fixed_canvas_prob_target_only_e2e(
    se_fixed_canvas_dataset, monkeypatch,
):
    """Mode B keeps one utterance on a fixed canvas and preserves collate flags."""
    ds, main_wav, extra_wav = se_fixed_canvas_dataset
    source, _ = soundfile.read(main_wav, dtype="float32")
    original_read = ds._read_one_speech
    original_get_clean = ds.speech_preparation.get_clean_speech
    reads: list[str] = []
    single_utterance_calls: list[bool] = []

    def tracked_read(path: str, sr: int) -> np.ndarray:
        reads.append(path)
        assert Path(path) != extra_wav, "mode B must not read a splice extra"
        return original_read(path, sr)

    def tracked_get_clean(
        dataset,
        speech_uid: str,
        sr: int,
        rng: np.random.Generator | None = None,
        single_utterance: bool = False,
        accounting: dict | None = None,
    ) -> np.ndarray:
        clean_stage = original_get_clean(
            dataset,
            speech_uid,
            sr,
            rng=rng,
            single_utterance=single_utterance,
            accounting=accounting,
        )
        single_utterance_calls.append(single_utterance)
        assert single_utterance is True
        np.testing.assert_array_equal(clean_stage, source)
        return clean_stage

    monkeypatch.setattr(ds, "_read_one_speech", tracked_read)
    monkeypatch.setattr(ds.speech_preparation, "get_clean_speech", tracked_get_clean)

    first = ds._getitem_simulation_verify(0, force_branch="prob_target_only")
    first_reads = reads.copy()
    first_single_utterance_calls = single_utterance_calls.copy()
    reads.clear()
    single_utterance_calls.clear()
    second = ds._getitem_simulation_verify(0, force_branch="prob_target_only")

    assert len(first) == 7
    clean, noisy, sr_out, length, is_pure_noise, is_pure_echo, is_pure_echo_noise = first
    assert first_reads == [str(main_wav)]
    assert reads == [str(main_wav)]
    assert first_single_utterance_calls == [True]
    assert single_utterance_calls == [True]
    assert sr_out == 16000
    assert length == 16000
    assert clean.shape == noisy.shape == (length,)
    np.testing.assert_array_equal(clean, noisy)
    assert (is_pure_noise, is_pure_echo, is_pure_echo_noise) == (False, False, False)

    support = np.flatnonzero(clean)
    assert support.size > 0
    start = int(support[0])
    stop = int(support[-1]) + 1
    assert start > 0
    assert stop < length
    assert stop - start == len(source)
    np.testing.assert_array_equal(clean[:start], 0.0)
    np.testing.assert_array_equal(clean[stop:], 0.0)
    np.testing.assert_allclose(clean[start:stop], source, rtol=3e-5, atol=3e-5)

    for actual, expected in zip(second, first):
        if isinstance(actual, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected

    batch = collate_fn([first, second])
    batch_clean, batch_noisy, batch_sr = batch.clean, batch.noisy, batch.sr
    lengths = batch.lengths
    pure_noise, pure_echo, pure_echo_noise = batch.is_pure_noise, batch.is_pure_echo, batch.is_pure_echo_noise
    assert batch_clean.shape == batch_noisy.shape == (2, 16000)
    assert batch_sr.item() == 16000
    torch.testing.assert_close(lengths, torch.tensor([16000, 16000], dtype=torch.int32))
    torch.testing.assert_close(pure_noise, torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(pure_echo, torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(pure_echo_noise, torch.zeros(2, dtype=torch.bool))


def test_se_long_fixed_canvas_is_one_contiguous_source_window(
    se_long_fixed_canvas_dataset, monkeypatch,
):
    """长模式 B 语音只裁一次，禁止预裁后补零或重复铺满。"""
    ds, speech_wav = se_long_fixed_canvas_dataset
    source, _ = soundfile.read(speech_wav, dtype="float32")
    canvas = 16000
    assert len(source) > canvas

    original_get_clean = ds.speech_preparation.get_clean_speech
    single_utterance_calls: list[bool] = []

    def tracked_get_clean(
        dataset,
        speech_uid: str,
        sr: int,
        rng: np.random.Generator | None = None,
        single_utterance: bool = False,
        accounting: dict | None = None,
    ) -> np.ndarray:
        clean_stage = original_get_clean(
            dataset,
            speech_uid,
            sr,
            rng=rng,
            single_utterance=single_utterance,
            accounting=accounting,
        )
        single_utterance_calls.append(single_utterance)
        assert single_utterance is True
        np.testing.assert_array_equal(clean_stage, source)
        return clean_stage

    monkeypatch.setattr(ds.speech_preparation, "get_clean_speech", tracked_get_clean)
    sample = ds._getitem_simulation_verify(0, force_branch="prob_target_only")

    assert len(sample) == 7
    clean, noisy, sr_out, length, is_pure_noise, is_pure_echo, is_pure_echo_noise = sample
    assert single_utterance_calls == [True]
    assert sr_out == 16000
    assert length == canvas
    assert clean.shape == noisy.shape == (canvas,)
    np.testing.assert_array_equal(clean, noisy)
    assert (is_pure_noise, is_pure_echo, is_pure_echo_noise) == (False, False, False)

    # 不依赖随机 crop 起点：用输出首段在源 waveform 的全部连续窗口中反查位置。
    anchor_len = 64
    source_anchors = np.lib.stride_tricks.sliding_window_view(source, anchor_len)
    anchor = clean[:anchor_len]
    denominators = np.sum(source_anchors * source_anchors, axis=1)
    gains = (source_anchors @ anchor) / np.maximum(denominators, 1e-12)
    residuals = np.mean(
        (source_anchors * gains[:, np.newaxis] - anchor) ** 2,
        axis=1,
    )
    crop_start = int(np.argmin(residuals))
    expected_window = source[crop_start : crop_start + canvas]
    assert len(expected_window) == canvas

    gain = float(np.dot(expected_window, clean) / np.dot(expected_window, expected_window))
    assert np.isfinite(gain) and gain > 0.0
    np.testing.assert_allclose(
        clean,
        expected_window * gain,
        rtol=3e-5,
        atol=3e-5,
    )

    batch = collate_fn([sample, sample])
    batch_clean, batch_noisy, batch_sr = batch.clean, batch.noisy, batch.sr
    lengths = batch.lengths
    pure_noise, pure_echo, pure_echo_noise = batch.is_pure_noise, batch.is_pure_echo, batch.is_pure_echo_noise
    assert batch_clean.shape == batch_noisy.shape == (2, canvas)
    assert batch_sr.item() == 16000
    torch.testing.assert_close(lengths, torch.tensor([canvas, canvas], dtype=torch.int32))
    torch.testing.assert_close(pure_noise, torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(pure_echo, torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(pure_echo_noise, torch.zeros(2, dtype=torch.bool))


def test_se_getitem_returns_expected_tuple(se_dataset: SEDynamicMixingDataset):
    sample = se_dataset[0]
    assert len(sample) == 7
    clean, noisy, sr_out, length, is_pure_noise, is_pure_echo, is_pure_echo_noise = sample
    assert isinstance(clean, np.ndarray)
    assert isinstance(noisy, np.ndarray)
    assert sr_out == 16000
    assert length > 0
    assert len(clean) == length
    assert len(noisy) == length
    assert np.isfinite(clean).all()
    assert np.isfinite(noisy).all()


def test_se_getitem_simulation_verify_force_branch(se_dataset: SEDynamicMixingDataset):
    sample = se_dataset._getitem_simulation_verify(0, force_branch="prob_target_only")
    clean, noisy, sr_out, length, *_ = sample
    assert sr_out == 16000
    assert length > 0
    np.testing.assert_allclose(clean, noisy, atol=1e-6)


def test_se_getitem_export_stems(se_dataset: SEDynamicMixingDataset, tmp_path: Path):
    se_dataset.export_simulation_stems = True
    _ = se_dataset[0]
    stems = se_dataset._last_simulation_stems
    assert stems is not None
    assert "final_clean" in stems
    assert "final_noisy" in stems
    assert np.isfinite(stems["final_clean"]).all()
    assert np.isfinite(stems["final_noisy"]).all()


def test_se_get_sample_with_trace_is_sample_local_and_preserves_legacy_interface(
    se_dataset: SEDynamicMixingDataset,
):
    first = se_dataset.get_sample_with_trace(
        0, force_branch="prob_target_only", capture_stems=True
    )
    second = se_dataset.get_sample_with_trace(
        0, force_branch="prob_noise_only", capture_stems=True
    )
    legacy_stems = se_dataset._last_simulation_stems

    assert len(first.sample) == len(second.sample) == 7
    assert first.trace is not second.trace
    assert first.trace.sealed and second.trace.sealed
    assert first.trace.metadata["branch"] == "prob_target_only"
    assert second.trace.metadata["branch"] == "prob_noise_only"
    assert first.trace.sources["scene"].metadata["branch"] == "prob_target_only"
    assert second.trace.sources["scene"].metadata["branch"] == "prob_noise_only"
    assert tuple(first.trace.sources["scene"].metadata["post_mix_stage_order"]) == (
        POST_MIX_STAGE_ORDER
    )
    assert tuple(second.trace.sources["scene"].metadata["post_mix_stage_order"]) == (
        POST_MIX_STAGE_ORDER
    )
    scene = first.trace.sources["scene"].metadata
    guard_keys = {"roles", "threshold", "peak", "scale", "applied"}
    assert set(scene["layer_d_facts"]) == {"stage_order", "normalize", "guard"}
    assert scene["layer_d_facts"]["stage_order"] == ["D-normalize", "D-guard"]
    assert legacy_stems["layer_d_facts"]["stage_order"] == (
        "D-normalize",
        "D-guard",
    )
    assert legacy_stems["simulation_by_layer"]["D"]["stage_order"] == (
        "D-normalize",
        "D-guard",
    )
    assert set(scene["layer_d_facts"]["guard"]) == guard_keys
    assert scene["simulation_by_layer"]["D"] == scene["layer_d_facts"]
    assert "D-normalize" not in scene["simulation_by_layer"]
    assert "layer_d_normalize" not in scene
    assert "layer_d" not in scene
    assert "aec_input_frontend" not in scene
    assert scene["final_time_transform"]["output_length"] == first.sample[3]
    required_tracks = scene["final_time_transform"]["required_tracks"]
    assert required_tracks[:2] == ["observation", "target"]
    assert required_tracks[2:] == [
        "export/layer_d_noisy",
        "export/layer_d_clean",
        "export/layer_d_noise_additive",
        "export/layer_d_noise_residual_mixed",
    ]
    assert first.trace.sources["scene"].stems["final_clean"] is not second.trace.sources["scene"].stems["final_clean"]


def test_se_get_sample_with_trace_capture_false_keeps_no_waveforms(
    se_dataset: SEDynamicMixingDataset,
):
    se_dataset._last_debug = {"stale": True}
    captured = se_dataset.get_sample_with_trace(
        0, force_branch="prob_noise_only", capture_stems=True
    )
    captured_legacy_stems = se_dataset._last_simulation_stems
    traced = se_dataset.get_sample_with_trace(
        0, force_branch="prob_target_only", capture_stems=False
    )

    assert traced.trace.capture_stems is False
    assert all(not source.stems for source in traced.trace.sources.values())
    assert se_dataset._last_simulation_stems is None
    assert traced.trace.metadata["branch"] == "prob_target_only"
    assert traced.trace.sources["scene"].metadata["branch"] == "prob_target_only"
    assert captured_legacy_stems is not None
    assert "_last_debug" not in captured_legacy_stems
    assert "simulation_by_layer" in captured.trace.sources["scene"].metadata
    assert "simulation_by_layer" not in traced.trace.sources["scene"].metadata
    assert "sim_meta" not in traced.trace.sources["scene"].metadata
    for result in (captured, traced):
        assert "_last_debug" not in result.trace.metadata
        assert "_last_debug" not in result.trace.sources["scene"].metadata
        assert "_last_debug" not in result.trace.legacy_snapshot()


def test_se_get_sample_with_trace_lenient_placeholder_discards_partial_trace(
    se_dataset: SEDynamicMixingDataset, monkeypatch,
):
    completed = se_dataset.get_sample_with_trace(
        0, force_branch="prob_target_only", capture_stems=True
    )
    assert se_dataset._last_simulation_stems is not None
    assert se_dataset._last_run_branch == "prob_target_only"
    se_dataset.strict_load = False

    def fail_after_partial(*args, trace, **kwargs):
        se_dataset._last_debug = {"partial": True}
        trace.source("scene").record_metadata("partial", True)
        trace.source("scene").record_stem("partial", np.ones(3, dtype=np.float32))
        raise RecoverableResourceError("injected failure")

    monkeypatch.setattr(se_dataset, "_build_sample", fail_after_partial)
    traced = se_dataset.get_sample_with_trace(0, capture_stems=True)

    assert traced.trace is not completed.trace
    assert traced.trace.placeholder is True
    assert "injected failure" in traced.trace.metadata["placeholder"]["reason"]
    assert "partial" not in traced.trace.sources["scene"].metadata
    assert "partial" not in traced.trace.sources["scene"].stems
    assert "_last_debug" not in traced.trace.metadata
    assert "_last_debug" not in traced.trace.legacy_snapshot()
    assert se_dataset._last_simulation_stems is None
    assert se_dataset._last_run_branch is None
    assert len(traced.sample) == 7


def test_se_get_sample_with_trace_rejects_raw_hook_result_in_lenient_mode(
    se_dataset: SEDynamicMixingDataset, monkeypatch,
):
    completed = se_dataset.get_sample_with_trace(0, force_branch="prob_target_only")
    se_dataset.strict_load = False

    def return_raw_sample(*args, **kwargs):
        return completed.sample

    monkeypatch.setattr(se_dataset, "_build_sample", return_raw_sample)
    with pytest.raises(RuntimeError, match=r"_build_sample.*SampleBuildResult"):
        se_dataset.get_sample_with_trace(0)

    assert se_dataset._last_simulation_trace is completed.trace
    assert se_dataset._last_run_branch == "prob_target_only"


def test_se_forced_branch_failure_never_returns_lenient_placeholder(
    se_dataset: SEDynamicMixingDataset, monkeypatch,
):
    completed = se_dataset.get_sample_with_trace(0, force_branch="prob_target_only")
    se_dataset.strict_load = False

    def fail_forced_sample(*args, **kwargs):
        raise ValueError("injected forced failure")

    monkeypatch.setattr(se_dataset, "_build_sample", fail_forced_sample)
    with pytest.raises(ValueError, match="injected forced failure"):
        se_dataset.get_sample_with_trace(0, force_branch="prob_noise_only")

    assert se_dataset._last_simulation_trace is completed.trace
    assert se_dataset._last_run_branch == "prob_target_only"


def test_se_get_sample_with_trace_strict_failure_keeps_last_completed(
    se_dataset: SEDynamicMixingDataset, monkeypatch,
):
    completed = se_dataset.get_sample_with_trace(
        0, force_branch="prob_target_only", capture_stems=True
    )
    previous_stems = se_dataset._last_simulation_stems
    se_dataset._last_debug = {"completed": True, "gains": [1.0]}
    previous_debug = se_dataset._last_debug

    def fail_after_partial(*args, trace, **kwargs):
        se_dataset._last_run_branch = "partial"
        se_dataset._last_simulation_stems = {"partial": np.ones(2, dtype=np.float32)}
        previous_debug["completed"] = False
        previous_debug["gains"][0] = 9.0
        se_dataset._last_debug = {"partial": True}
        trace.source("scene").record_metadata("partial", True)
        raise ValueError("injected strict failure")

    monkeypatch.setattr(se_dataset, "_build_sample", fail_after_partial)
    with pytest.raises(RuntimeError, match="injected strict failure"):
        se_dataset.get_sample_with_trace(0, capture_stems=True)

    assert se_dataset._last_simulation_trace is completed.trace
    assert se_dataset._last_simulation_stems is previous_stems
    assert se_dataset._last_run_branch == "prob_target_only"
    assert se_dataset._last_debug is previous_debug
    assert previous_debug == {"completed": True, "gains": [1.0]}


def test_build_result_rejects_any_source_sealed_by_task_hook(
    se_dataset: SEDynamicMixingDataset, monkeypatch: pytest.MonkeyPatch,
):
    completed = se_dataset.get_sample_with_trace(0, force_branch="prob_target_only")

    from core.data.dynamic_dataset_base import SampleBuildResult

    def sealed_result(*args, trace, **kwargs):
        trace.source("scene").seal()
        return SampleBuildResult(sample=completed.sample, trace=trace)

    monkeypatch.setattr(se_dataset, "_build_sample", sealed_result)
    with pytest.raises(RuntimeError, match="source traces before owner commit"):
        se_dataset.get_sample_with_trace(0)

    assert se_dataset._last_simulation_trace is completed.trace
    assert se_dataset._last_run_branch == "prob_target_only"


def test_seal_failure_restores_complete_last_completed_snapshot(
    se_dataset: SEDynamicMixingDataset, monkeypatch: pytest.MonkeyPatch,
):
    completed = se_dataset.get_sample_with_trace(
        0, force_branch="prob_target_only", capture_stems=True
    )
    previous_stems = se_dataset._last_simulation_stems
    original_seal = SimulationTrace.seal

    def fail_after_seal(trace):
        original_seal(trace)
        raise ValueError("injected seal failure")

    monkeypatch.setattr(SimulationTrace, "seal", fail_after_seal)
    with pytest.raises(ValueError, match="injected seal failure"):
        se_dataset.get_sample_with_trace(0, force_branch="prob_noise_only")

    assert se_dataset._last_simulation_trace is completed.trace
    assert se_dataset._last_simulation_stems is previous_stems
    assert se_dataset._last_run_branch == "prob_target_only"


def test_publish_failure_restores_complete_last_completed_snapshot(
    se_dataset: SEDynamicMixingDataset, monkeypatch: pytest.MonkeyPatch,
):
    completed = se_dataset.get_sample_with_trace(
        0, force_branch="prob_target_only", capture_stems=True
    )
    previous_stems = se_dataset._last_simulation_stems

    def fail_publish(trace):
        se_dataset._last_simulation_trace = trace
        se_dataset._last_simulation_stems = {"partial": True}
        se_dataset._last_run_branch = "partial"
        raise ValueError("injected publish failure")

    monkeypatch.setattr(se_dataset, "_publish_completed_trace", fail_publish)
    with pytest.raises(ValueError, match="injected publish failure"):
        se_dataset.get_sample_with_trace(0, force_branch="prob_noise_only")

    assert se_dataset._last_simulation_trace is completed.trace
    assert se_dataset._last_simulation_stems is previous_stems
    assert se_dataset._last_run_branch == "prob_target_only"


@pytest.fixture
def aec_dataset(tmp_path: Path):
    """Build a small AEC DynamicMixingDataset backed by synthetic files."""
    sr = 16000
    duration = 2.0

    speech_dir = tmp_path / "speech"
    noise_dir = tmp_path / "noise"
    rir_dir = tmp_path / "rir"
    wind_dir = tmp_path / "wind"
    aec_dir = tmp_path / "aec"
    for d in (speech_dir, noise_dir, rir_dir, wind_dir, aec_dir):
        d.mkdir()

    speech_wav = speech_dir / "spk1_001.wav"
    noise_wav = noise_dir / "noise_001.wav"
    rir_wav = rir_dir / "rir_001.wav"
    wind_wav = wind_dir / "wind_001.wav"
    farend_wav = aec_dir / "farend_001.wav"
    echo_wav = aec_dir / "echo_001.wav"

    _make_wav(speech_wav, duration, sr, seed=10)
    _make_wav(noise_wav, duration, sr, seed=20)
    _make_impulse_rir(rir_wav, sr)
    _make_wav(wind_wav, duration, sr, seed=30)
    _make_wav(farend_wav, duration, sr, seed=40)
    # Echo is a delayed/attenuated copy of farend.
    farend, _ = soundfile.read(farend_wav, dtype="float32")
    echo = np.zeros_like(farend)
    delay = int(0.05 * sr)
    echo[delay:] = farend[:-delay] * 0.3
    soundfile.write(echo_wav, echo, sr)
    stereo_wav = aec_dir / "stereo_001.wav"
    soundfile.write(stereo_wav, np.stack([echo, farend], axis=1), sr)

    speech_scp = tmp_path / "speech.scp"
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    wind_scp = tmp_path / "wind.scp"
    pair_json = tmp_path / "farend_echo.json"
    _write_scp(speech_scp, {"spk1_001": speech_wav})
    _write_scp(noise_scp, {"noise_001": noise_wav})
    _write_scp(rir_scp, {"rir_001": rir_wav})
    _write_scp(wind_scp, {"wind_001": wind_wav})
    with pair_json.open("w", encoding="utf-8") as f:
        json.dump({"pairs": [str(stereo_wav)]}, f)

    simulation_cfg = {
        "prob_target_only": 0.15,
        "prob_noise_only": 0.15,
        "prob_echo_only": 0.15,
        "prob_noise_echo_only": 0.15,
        "prob_target_noise": 0.15,
        "prob_target_echo": 0.15,
        "prob_target_noise_echo": 0.10,
        "snr_low": 5.0,
        "snr_high": 20.0,
        "prob_channel_enabled": False,
        "noisy_post_mix_degradations_enabled": False,
        "prob_reverberation": 0.5,
    }

    ds = AECDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=str(wind_scp),
        simulation_cfg=simulation_cfg,
        real_farend_echo_pair_json=str(pair_json),
        aec_ref_echo_random_delay_prob=0.0,
        max_duration_seconds=1.0,
        target_sr=sr,
        task="joint",
        strict_load=True,
    )
    ds.set_epoch(0, base_seed=456)
    return ds


@pytest.mark.parametrize("key", ["aec_stream_align", "linear_aec"])
def test_aec_dataset_validator_rejects_retired_frontend_keys(key):
    with pytest.raises(ValueError, match=rf"{key}.*aec_frontend"):
        validate_aec_dataset_config(
            simulation_cfg={"prob_target_only": 1.0},
            **{key: {"enabled": True}},
        )


@pytest.mark.parametrize("key", ["aec_stream_align", "linear_aec"])
def test_aec_dataset_validator_rejects_retired_frontend_simulation_keys(key):
    with pytest.raises(ValueError, match=rf"simulation_cfg\.{key}.*aec_frontend"):
        validate_aec_dataset_config(
            simulation_cfg={"prob_target_only": 1.0, key: {"enabled": True}},
        )


def test_aec_dataset_validator_rejects_probability_residual():
    """七个分支的总和不能留下会被末分支吸收的隐含概率。"""
    simulation_cfg = {
        "prob_target_only": 0.05,
        "prob_noise_only": 0.05,
        "prob_echo_only": 0.05,
        "prob_noise_echo_only": 0.05,
        "prob_target_noise": 0.199995,
        "prob_target_echo": 0.20,
        "prob_target_noise_echo": 0.40,
    }
    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_aec_dataset_config(simulation_cfg=simulation_cfg)


def test_aec_getitem_returns_expected_tuple(aec_dataset: AECDynamicMixingDataset):
    sample = aec_dataset[0]
    assert len(sample) == 11
    nearend, mic, farend, sr_out, length, err, err_mask, *flags = sample
    assert isinstance(nearend, torch.Tensor)
    assert isinstance(mic, torch.Tensor)
    assert isinstance(farend, torch.Tensor)
    assert sr_out == 16000
    assert int(length) > 0
    assert nearend.numel() == int(length)
    assert mic.numel() == int(length)
    assert farend.numel() == int(length)
    assert torch.isfinite(nearend).all()
    assert torch.isfinite(mic).all()
    assert torch.isfinite(farend).all()
    assert torch.count_nonzero(err) == 0
    assert err_mask is False
    assert not hasattr(aec_dataset, "linear_aec")
    assert not hasattr(aec_dataset, "_gpu_batch_pfdkf")
    assert not hasattr(aec_dataset, "_aec_stream_align_cfg")


def test_aec_getitem_simulation_verify_force_branch(aec_dataset: AECDynamicMixingDataset):
    traced = aec_dataset.get_sample_with_trace(0, force_branch="prob_target_only")
    nearend, mic, farend, sr_out, length, *_ = traced.sample
    assert sr_out == 16000
    assert int(length) > 0
    assert traced.trace.metadata["branch"] == "prob_target_only"
    assert traced.trace.sources["scene"].metadata["branch"] == "prob_target_only"
    # Clean-only branch: mic equals nearend, farend is zero placeholder.
    np.testing.assert_allclose(nearend.numpy(), mic.numpy(), atol=1e-6)
    np.testing.assert_allclose(farend.numpy(), 0.0, atol=1e-6)


def test_aec_branch_selection_consumes_prob_target_noise_echo_probability(
    aec_dataset: AECDynamicMixingDataset, monkeypatch: pytest.MonkeyPatch
):
    """容差内的概率和也必须保留显式 prob_target_noise_echo 配置值。"""
    # 构造一个能通过初始化校验的边界配置：前六项之和与显式第七项
    # 相差 5e-6。旧实现用前六项补集，会把第七项改成 0.100005。
    aec_dataset.sim_cfg.prob_target_noise = 0.149995
    captured: dict[str, float] = {}
    original_select_branch = aec_dataset._select_branch

    def capture_select_branch(
        branches: list[tuple[str, float]],
        rng: np.random.Generator,
        force_branch: str | None = None,
        normalize_force=None,
    ):
        captured.update(dict(branches))
        return original_select_branch(
            branches,
            rng,
            force_branch=force_branch,
            normalize_force=normalize_force,
        )

    monkeypatch.setattr(aec_dataset, "_select_branch", capture_select_branch)
    aec_dataset._run(
        "spk1_001",
        32000,
        16000,
        index=0,
        force_branch="prob_target_only",
        rng=np.random.default_rng(0),
    )

    assert captured["prob_target_noise_echo"] == pytest.approx(0.10, abs=1e-12)


def test_aec_dataset_keeps_c_aec_gcc_in_layer_c(
    aec_dataset: AECDynamicMixingDataset,
):
    traced = aec_dataset.get_sample_with_trace(
        0, force_branch="prob_target_echo", capture_stems=True
    )

    error, err_mask = traced.sample[5:7]
    scene = traced.trace.sources["scene"]
    assert torch.count_nonzero(error) == 0
    assert err_mask is False
    assert "aec_input_frontend" not in scene.metadata
    c3_load = scene.metadata["simulation_by_layer"]["C"]["c3_disk_pair_load"]
    assert c3_load["gcc_mode"] != "disabled"
    assert c3_load["gcc_tau_ms"] is not None


def test_aec_getitem_export_stems(aec_dataset: AECDynamicMixingDataset, tmp_path: Path):
    aec_dataset.export_simulation_stems = True
    _ = aec_dataset[0]
    stems = aec_dataset._last_simulation_stems
    assert stems is not None
    assert "final_clean" in stems
    assert "final_noisy" in stems
    assert np.isfinite(stems["final_clean"]).all()
    assert np.isfinite(stems["final_noisy"]).all()


def test_aec_lenient_placeholder_neutralizes_all_managed_legacy_state(
    aec_dataset: AECDynamicMixingDataset, monkeypatch: pytest.MonkeyPatch,
):
    aec_dataset.get_sample_with_trace(
        0, force_branch="prob_target_echo", capture_stems=True
    )
    aec_dataset.strict_load = False

    def fail_sample(*args, **kwargs):
        raise RecoverableResourceError("injected AEC failure")

    monkeypatch.setattr(aec_dataset, "_build_sample", fail_sample)
    traced = aec_dataset.get_sample_with_trace(0, capture_stems=True)

    assert traced.trace.placeholder is True
    assert traced.trace.metadata["branch"] is None
    assert traced.trace.sources["scene"].metadata["branch"] is None
    assert aec_dataset._last_simulation_stems is None
    assert aec_dataset._last_run_branch is None
    assert aec_dataset._last_noise_additive_1d is None
    assert aec_dataset._last_layer_e_result is None
    assert aec_dataset._last_layer_f_applied_ops == []
    assert aec_dataset._last_layer_f_clipping_facts is None
    assert aec_dataset._last_layer_ef_decomposable is None
    assert aec_dataset._last_decomposable_reason_codes == []
    assert aec_dataset._last_c0_audit_stems is None
    assert aec_dataset._last_mic_extra_noise_policy is None


def test_aec_echo_clipping_audit_fields_present_when_disabled(aec_dataset: AECDynamicMixingDataset):
    aec_dataset.export_simulation_stems = True
    _ = aec_dataset[0]
    layer_c = aec_dataset._last_simulation_stems["simulation_by_layer"]["C"]
    assert "c3_echo_clipping" in layer_c
    assert layer_c["c3_echo_clipping"]["applied"] is False
    assert layer_c["c3_echo_clipping"]["path"] is None


def test_se_getitem_deterministic_within_epoch(se_dataset: SEDynamicMixingDataset):
    """Fixed epoch seed → two __getitem__ calls on the same index match exactly."""
    first = se_dataset[0]
    second = se_dataset[0]
    for a, b in zip(first, second):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b


def test_se_getitem_deterministic_across_cache_clear(se_dataset: SEDynamicMixingDataset):
    """Audio cache state (byte-LRU / uncached speech) must never change outputs."""
    from core.data.simulation_audio_utils import _load_audio_cached_impl

    first = se_dataset[0]
    _load_audio_cached_impl.cache_clear()
    se_dataset.set_epoch(0, base_seed=123)
    second = se_dataset[0]
    for a, b in zip(first, second):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b


def test_aec_getitem_deterministic_across_cache_clear(aec_dataset: AECDynamicMixingDataset):
    """Mono + stereo audio caches must never change AEC __getitem__ outputs."""
    from core.data.simulation_audio_utils import _load_audio_cached_impl
    from tasks.aec.data.dynamic_dataset import _load_stereo_farend_echo_cached_impl

    first = aec_dataset[0]
    _load_audio_cached_impl.cache_clear()
    _load_stereo_farend_echo_cached_impl.cache_clear()
    aec_dataset.set_epoch(0, base_seed=456)
    second = aec_dataset[0]
    for a, b in zip(first, second):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b


def test_se_splice_extra_load_failure_is_non_fatal(tmp_path: Path, monkeypatch):
    """A corrupted splice extra must not crash training when strict_load=True."""
    sr = 16000
    duration = 0.5

    speech_dir = tmp_path / "speech"
    noise_dir = tmp_path / "noise"
    rir_dir = tmp_path / "rir"
    wind_dir = tmp_path / "wind"
    for d in (speech_dir, noise_dir, rir_dir, wind_dir):
        d.mkdir()

    main_wav = speech_dir / "main.wav"
    good_wav = speech_dir / "good.wav"
    bad_wav = speech_dir / "bad.wav"
    _make_wav(main_wav, duration, sr, seed=1)
    _make_wav(good_wav, duration, sr, seed=2)
    # bad.wav 占位，读取时由 monkeypatch 模拟失败
    _make_wav(bad_wav, duration, sr, seed=3)

    noise_wav = noise_dir / "noise_001.wav"
    rir_wav = rir_dir / "rir_001.wav"
    wind_wav = wind_dir / "wind_001.wav"
    _make_wav(noise_wav, duration, sr, seed=4)
    _make_impulse_rir(rir_wav, sr)
    _make_wav(wind_wav, duration, sr, seed=5)

    speech_scp = tmp_path / "speech.scp"
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    wind_scp = tmp_path / "wind.scp"
    _write_scp(speech_scp, {"main": main_wav, "good": good_wav, "bad": bad_wav})
    _write_scp(noise_scp, {"noise_001": noise_wav})
    _write_scp(rir_scp, {"rir_001": rir_wav})
    _write_scp(wind_scp, {"wind_001": wind_wav})

    simulation_cfg = {
        "prob_target_only": 0.0,
        "prob_noise_only": 0.0,
        "snr_low": 5.0,
        "snr_high": 20.0,
        "prob_channel_enabled": False,
        "noisy_post_mix_degradations_enabled": False,
        "prob_reverberation": 0.0,
        "speech_splice": True,
        "speech_splice_seconds": 2.0,
        "speech_splice_max_repeat_times": 5,
    }

    ds = SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=str(wind_scp),
        simulation_cfg=simulation_cfg,
        max_duration_seconds=2.0,
        target_sr=sr,
        filter_by_duration=True,
        strict_load=True,
    )
    ds.set_epoch(0, base_seed=42)

    original_read = ds._read_one_speech
    call_count = {"n": 0}

    def flaky_read(path: str, sr_arg: int) -> np.ndarray:
        call_count["n"] += 1
        if "bad" in path and call_count["n"] <= 2:
            raise RuntimeError("simulated corrupted splice extra")
        return original_read(path, sr_arg)

    monkeypatch.setattr(ds, "_read_one_speech", flaky_read)

    sample = ds[0]
    clean, noisy, sr_out, length, *_ = sample
    assert sr_out == sr
    assert length > 0
    assert len(clean) == length
    assert np.isfinite(clean).all()

pytestmark = [pytest.mark.unit]
