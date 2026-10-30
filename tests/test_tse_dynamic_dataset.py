"""Tests for tasks/tse/data/dynamic_dataset.py (TSEDynamicMixingDataset).

使用 tmp_path 下的最小合成数据（wav/strict JSONL/nested enrollment）验证
``_build_sample`` 产出形状、enroll 定长与 RMS 归一、以及概率全 0 时的确定性行为。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile
import torch

from core.data.dynamic_dataset_base import LayerEOperationResult, LayerEResult
from core.data.sampling import SamplingConfig
from core.models.build import build_model
from core.stft import StftIstftModule as STFT
from tasks.tse.data.collate import tse_collate_fn
from tasks.tse.data.dynamic_dataset import (
    EnrollmentAugmentationProviderError,
    TSEDynamicMixingDataset,
)
from tasks.tse.data.enrollment_augment import EnrollmentAugmentor
from tasks.tse.model_io import build_tse_model_inputs, decode_tse_model_output
from tasks.tse.models import register_tse_models

SR = 16000


def _write_scp(path: Path, entries: dict[str, Path]) -> None:
    """Write a source SCP file with ``uid fs num_samples path`` columns."""
    with path.open("w", encoding="utf-8") as f:
        for uid, audio_path in entries.items():
            info = soundfile.info(str(audio_path))
            f.write(f"{uid} {info.samplerate} {info.frames} {audio_path}\n")


def _make_wav(path: Path, duration: float, sr: int, *, seed: int = 0, freq: float = 300.0) -> None:
    """Write a synthetic mono WAV file (sine + small noise)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    signal = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    signal += 0.01 * rng.standard_normal(signal.shape).astype(np.float32)
    signal = (signal * 0.5).astype(np.float32)
    soundfile.write(path, signal, sr)


def _assert_single_scaled_utterance_on_canvas(
    target: np.ndarray, source: np.ndarray, source_samples: int
) -> None:
    """验证画布只含一次完整源波形，允许混音链施加统一增益。"""
    target_active = np.flatnonzero(target != 0.0)
    source_active = np.flatnonzero(source != 0.0)
    assert 0 < target_active.size <= source_samples
    assert source_active.size > 0

    start = int(target_active[0] - source_active[0])
    assert 0 <= start <= len(target) - len(source)
    np.testing.assert_array_equal(target_active, start + source_active)

    segment = target[start : start + len(source)]
    scale = float(np.dot(segment.astype(np.float64), source.astype(np.float64)))
    scale /= float(np.dot(source.astype(np.float64), source.astype(np.float64)))
    assert scale > 0.0
    np.testing.assert_allclose(segment, source * scale, rtol=1e-5, atol=1e-6)
    assert np.count_nonzero(target[:start]) == 0
    assert np.count_nonzero(target[start + len(source) :]) == 0


def _assert_single_scaled_contiguous_crop(target: np.ndarray, source: np.ndarray) -> None:
    """验证输出是长源波形的某个连续 crop，允许混音链施加统一增益。"""
    assert 0 < len(target) < len(source)
    probe_samples = min(128, len(target))
    probe = target[:probe_samples].astype(np.float64)
    max_start = len(source) - len(target)
    source_windows = np.lib.stride_tricks.sliding_window_view(
        source.astype(np.float64), probe_samples
    )[: max_start + 1]
    denominators = np.einsum("ij,ij->i", source_windows, source_windows)
    scales = np.einsum("ij,j->i", source_windows, probe) / denominators
    residuals = np.sum(
        np.square(source_windows * scales[:, None] - probe[None, :]), axis=1
    )
    start = int(np.argmin(residuals))
    scale = float(scales[start])

    assert scale > 0.0
    np.testing.assert_allclose(
        target,
        source[start : start + len(target)] * scale,
        rtol=1e-5,
        atol=1e-6,
    )


def _make_impulse_rir(path: Path, sr: int) -> None:
    rir = np.zeros(sr, dtype=np.float32)
    rir[0] = 1.0
    soundfile.write(path, rir, sr)


@pytest.fixture
def tse_data(tmp_path: Path) -> dict:
    """最小 TSE 数据集资源：2 个说话人（spk1/spk2）+ noise/rir/wind/music。"""
    speech_dir = tmp_path / "speech"
    noise_dir = tmp_path / "noise"
    rir_dir = tmp_path / "rir"
    wind_dir = tmp_path / "wind"
    music_dir = tmp_path / "music"
    enroll_dir = tmp_path / "enroll"
    for d in (speech_dir, noise_dir, rir_dir, wind_dir, music_dir, enroll_dir):
        d.mkdir()

    spk1_wav = speech_dir / "spk1_001.wav"
    spk2_wav = speech_dir / "spk2_001.wav"
    noise_wav = noise_dir / "noise_001.wav"
    rir_wav = rir_dir / "rir_001.wav"
    wind_wav = wind_dir / "wind_001.wav"
    music_wav = music_dir / "music_001.wav"
    enroll1_wav = enroll_dir / "spk1_enroll_a.wav"
    enroll2_wav = enroll_dir / "spk1_enroll_b.wav"
    enroll3_wav = enroll_dir / "spk2_enroll.wav"

    _make_wav(spk1_wav, 2.0, SR, seed=1, freq=300.0)
    _make_wav(spk2_wav, 2.0, SR, seed=2, freq=500.0)
    _make_wav(noise_wav, 2.0, SR, seed=3)
    _make_impulse_rir(rir_wav, SR)
    _make_wav(wind_wav, 2.0, SR, seed=4)
    _make_wav(music_wav, 2.0, SR, seed=5, freq=700.0)
    _make_wav(enroll1_wav, 1.0, SR, seed=6, freq=320.0)
    _make_wav(enroll2_wav, 1.0, SR, seed=7, freq=330.0)
    _make_wav(enroll3_wav, 1.0, SR, seed=8, freq=520.0)

    speech_jsonl = tmp_path / "speech.jsonl"
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    wind_scp = tmp_path / "wind.scp"
    music_scp = tmp_path / "music.scp"
    speech_jsonl.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "uid": "spk1_001", "fs": SR, "length": 2 * SR,
                    "path": str(spk1_wav), "language": "zh",
                    "dataset_id": "fixture", "speaker_id": "spk1",
                },
                {
                    "uid": "spk2_001", "fs": SR, "length": 2 * SR,
                    "path": str(spk2_wav), "language": "zh",
                    "dataset_id": "fixture", "speaker_id": "spk2",
                },
            )
        ) + "\n",
        encoding="utf-8",
    )
    _write_scp(noise_scp, {"noise_001": noise_wav})
    _write_scp(rir_scp, {"rir_001": rir_wav})
    _write_scp(wind_scp, {"wind_001": wind_wav})
    _write_scp(music_scp, {"music_001": music_wav})

    spk2enroll_json = tmp_path / "spk2enroll.json"
    with spk2enroll_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "fixture": {
                    "spk1": [str(enroll1_wav), str(enroll2_wav)],
                    "spk2": [str(enroll3_wav)],
                }
            },
            f,
        )

    return {
        "speech_scp": str(speech_jsonl),
        "noise_scp": str(noise_scp),
        "rir_scp": str(rir_scp),
        "windnoise_scp": str(wind_scp),
        "music_scp": str(music_scp),
        "spk2enroll_json": str(spk2enroll_json),
        "noise_wav": noise_wav,
        "music_wav": music_wav,
        "enroll1_wav": enroll1_wav,
    }


def _base_sim_cfg() -> dict:
    """全部 TSE/退化概率置 0 的确定性 baseline 仿真配置。"""
    return {'snr_low': 5.0, 'snr_high': 20.0, 'prob_reverberation': 0.0, 'prob_channel_enabled': False, 'noisy_post_mix_degradations_enabled': False, 'music_ratio': 0.0, 'prob_target_only': 0.0, 'prob_interference_only': 0.0, 'prob_noise_only': 0.0, 'prob_interference_noise_only': 0.0, 'prob_target_noise': 1.0, 'prob_target_interference': 0.0, 'prob_target_interference_noise': 0.0}


def _build_dataset(tse_data: dict, sim_overrides: dict | None = None, **kwargs):
    sim_cfg = _base_sim_cfg()
    if sim_overrides:
        from core.data.simulation.tse_scenarios import TSE_SCENARIO_KEYS
        if any(k in TSE_SCENARIO_KEYS for k in sim_overrides):
            sim_cfg.update({k: 0.0 for k in TSE_SCENARIO_KEYS})
        sim_cfg.update(sim_overrides)
    params = dict(
        speech_scp=tse_data["speech_scp"],
        noise_scp=tse_data["noise_scp"],
        rir_scp=tse_data["rir_scp"],
        windnoise_scp=tse_data["windnoise_scp"],
        spk2enroll_json=tse_data["spk2enroll_json"],
        simulation_cfg=sim_cfg,
        max_duration_seconds=1.0,
        target_sr=SR,
        filter_by_duration=True,
        strict_load=True,
    )
    params.update(kwargs)
    ds = TSEDynamicMixingDataset(**params)
    ds.set_epoch(0, base_seed=123)
    return ds


def _target_row(sample: dict, index: int = 0) -> dict:
    return sample["targets"][index]


def _enrollment(sample: dict, index: int = 0) -> np.ndarray:
    cue = _target_row(sample, index)["cues"].speaker_waveform
    assert cue is not None and cue.present and cue.value is not None
    return np.asarray(cue.value)


def _assert_same_sample(first: dict, second: dict) -> None:
    np.testing.assert_array_equal(first["mixture"], second["mixture"])
    assert first["sr"] == second["sr"]
    assert first["length"] == second["length"]
    assert len(first["targets"]) == len(second["targets"])
    for first_target, second_target in zip(first["targets"], second["targets"]):
        np.testing.assert_array_equal(first_target["target"], second_target["target"])
        np.testing.assert_array_equal(
            _enrollment({"targets": [first_target]}),
            _enrollment({"targets": [second_target]}),
        )
        assert first_target["spk_label"] == second_target["spk_label"]
        assert first_target["target_present"] is second_target["target_present"]


def test_build_sample_shapes(tse_data):
    """canonical item 保留定长 enrollment、目标身份与稳定 speaker label。"""
    ds = _build_dataset(tse_data, enrollment_duration=0.5)
    sample = ds[0]
    row = _target_row(sample)
    enroll = _enrollment(sample)
    cue = row["cues"].speaker_waveform
    assert cue is not None
    assert isinstance(sample["mixture"], np.ndarray)
    assert isinstance(row["target"], np.ndarray)
    assert sample["sr"] == SR
    assert sample["length"] > 0
    assert sample["mixture"].shape == (1, sample["length"])
    assert len(row["target"]) == sample["length"]
    assert cue.length == int(round(0.5 * SR))
    assert len(enroll) == cue.length
    assert np.isfinite(sample["mixture"]).all()
    assert np.isfinite(row["target"]).all()
    assert np.isfinite(enroll).all()
    fs, i = ds._get_from_index(0)
    uid = ds.speech_uids[fs][i]
    assert row["speaker_id"] == uid
    assert row["spk_label"] == ds.speaker_identity_to_idx[
        ds.uid_to_speaker_identity[uid]
    ]


def test_single_target_fixed_silence_canvas_and_collate_end_to_end(
    tse_data, monkeypatch
):
    """模式 B 只读并放置一次主目标；collate 产出 validated TSEBatch。"""
    canvas_samples = 3 * SR
    source_samples = 2 * SR
    enroll_samples = int(round(0.5 * SR))
    ds = _build_dataset(
        tse_data,
        {'speech_splice': True, 'speech_splice_seconds': 3.0, 'silence_pad_prob': 1.0, 'silence_pad_position': 'front', 'use_high_pass': False, 'prob_target_only': 1.0},
        max_duration_seconds=3.0,
        repeat_to_max_duration=True,
        enrollment_duration=0.5,
        enroll_policy="fixed",
    )
    fs, i = ds._get_from_index(0)
    uid = ds.speech_uids[fs][i]
    expected_speech_path = ds.speech_flt[uid]
    speech_paths = set(ds.speech_flt.values())
    speech_reads: list[tuple[str, np.ndarray]] = []
    original_read = ds._read_one_speech

    def _record_speech_read(path, sr):
        wav = original_read(path, sr)
        if path in speech_paths:
            speech_reads.append((path, wav.copy()))
        return wav

    monkeypatch.setattr(ds, "_read_one_speech", _record_speech_read)
    sample = ds[0]
    assert [path for path, _ in speech_reads] == [expected_speech_path]

    target = _target_row(sample)["target"]
    enroll = _enrollment(sample)
    cue = _target_row(sample)["cues"].speaker_waveform
    assert cue is not None
    assert sample["sr"] == SR
    assert sample["length"] == sample["mixture"].shape[-1] == len(target) == canvas_samples
    np.testing.assert_allclose(sample["mixture"][0], target, atol=1e-6)
    _assert_single_scaled_utterance_on_canvas(
        target, speech_reads[0][1], source_samples
    )
    assert cue.length == len(enroll) == enroll_samples
    expected_label = ds.speaker_identity_to_idx[ds.uid_to_speaker_identity[uid]]
    assert _target_row(sample)["spk_label"] == expected_label

    out = tse_collate_fn([sample], cue_specs=ds.cue_specs)
    speaker_waveform = out.cues.speaker_waveform
    assert speaker_waveform is not None and speaker_waveform.lengths is not None
    assert out.mixture.shape == (1, 1, canvas_samples)
    assert out.target.shape == (1, canvas_samples)
    assert speaker_waveform.value.shape == (1, enroll_samples)
    assert int(out.sr) == SR
    assert out.lengths.tolist() == [canvas_samples]
    assert speaker_waveform.lengths.tolist() == [enroll_samples]
    assert out.spk_labels.tolist() == [expected_label]


def test_dynamic_dataset_collate_drives_modular_speaker_frontend_backward(tse_data):
    """验证我方 dynamic 数据主干可直接驱动 Modular speaker frontend。"""
    ds = _build_dataset(
        tse_data,
        {"prob_target_only": 1.0, "use_high_pass": False},
        max_duration_seconds=0.25,
        enrollment_duration=0.125,
        enroll_policy="fixed",
        filter_min_duration=0.0,
    )
    batch = tse_collate_fn([ds[0]], cue_specs=ds.cue_specs)

    model_args = {
        "backbone": "bsrnn",
        "separator": {
            "feature_dim": 8,
            "num_repeat": 2,
            "band_width": [16, 17],
            "causal": False,
        },
        "cues": {
            "speaker": {
                "representation": "waveform",
                "features": {"tfmap": {"type": "spec"}},
            }
        },
        "composition": {},
        "sample_rate": SR,
        "n_channel": 1,
        "ref_channel": 0,
    }
    stft_config = {"n_fft": 64, "hop_length": 16, "win_length": 64}
    stft = STFT(**stft_config)
    register_tse_models(only="TSE_Modular")
    model = build_model(
        {"type": "TSE_Modular", "args": model_args}, stft_cfg=stft_config
    ).train()
    model_inputs = build_tse_model_inputs(
        model=model,
        stft=stft,
        mixture=batch.mixture,
        sample_rate=SR,
        lengths=batch.lengths,
        num_targets=batch.num_targets,
        cues=batch.cues,
    )
    output = decode_tse_model_output(
        model(model_inputs), stft=stft, reference_length=batch.mixture.shape[-1]
    )["target_estimate"]
    output.square().mean().backward()
    assert output.shape == batch.target.shape
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_single_target_fixed_silence_canvas_long_utterance_end_to_end(
    tse_data, tmp_path, monkeypatch
):
    """模式 B 长主目标只连续裁剪一次；注册音独立且批合同保持 canonical。"""
    canvas_samples = SR
    enroll_samples = int(round(0.5 * SR))
    long_speech = tmp_path / "spk1_long.wav"
    _make_wav(long_speech, 3.0, SR, seed=19, freq=410.0)
    long_speech_jsonl = tmp_path / "long_speech.jsonl"
    long_speech_jsonl.write_text(
        json.dumps(
            {
                "uid": "spk1_long", "fs": SR, "length": 3 * SR,
                "path": str(long_speech), "language": "zh",
                "dataset_id": "fixture", "speaker_id": "spk1",
            }
        ) + "\n",
        encoding="utf-8",
    )
    long_enrollment = tmp_path / "long_spk2enroll.json"
    long_enrollment.write_text(
        json.dumps({"fixture": {"spk1": [str(tse_data["enroll1_wav"])]}}),
        encoding="utf-8",
    )
    long_data = {
        **tse_data,
        "speech_scp": str(long_speech_jsonl),
        "spk2enroll_json": str(long_enrollment),
    }
    ds = _build_dataset(
        long_data,
        {'speech_splice': True, 'speech_splice_seconds': 1.0, 'silence_pad_prob': 1.0, 'silence_pad_position': 'front', 'prob_speed_perturb': 0.0, 'prob_pitch_shift': 0.0, 'prob_eq_perturb': 0.0, 'prob_add_extra_space': 0.0, 'use_high_pass': False, 'source_volume_normalization_enabled': False, 'peak_normalization_mode': 'off', 'prob_target_only': 1.0},
        max_duration_seconds=1.0,
        repeat_to_max_duration=True,
        enrollment_duration=0.5,
        enroll_policy="fixed",
    )
    reads: list[tuple[str, np.ndarray]] = []
    original_read = ds._read_one_speech

    def _record_read(path, sr):
        wav = original_read(path, sr)
        reads.append((path, wav.copy()))
        return wav

    monkeypatch.setattr(ds, "_read_one_speech", _record_read)
    sample = ds[0]

    assert [path for path, _ in reads] == [
        str(long_speech),
        str(tse_data["enroll1_wav"]),
    ]
    target = _target_row(sample)["target"]
    enroll = _enrollment(sample)
    cue = _target_row(sample)["cues"].speaker_waveform
    assert cue is not None
    assert sample["sr"] == SR
    assert sample["length"] == sample["mixture"].shape[-1] == len(target) == canvas_samples
    np.testing.assert_allclose(sample["mixture"][0], target, atol=1e-6)
    _assert_single_scaled_contiguous_crop(target, reads[0][1])
    assert cue.length == len(enroll) == enroll_samples
    assert not np.shares_memory(target, enroll)
    expected_label = ds.speaker_identity_to_idx[ds.uid_to_speaker_identity["spk1_long"]]
    assert _target_row(sample)["spk_label"] == expected_label

    out = tse_collate_fn([sample], cue_specs=ds.cue_specs)
    speaker_waveform = out.cues.speaker_waveform
    assert speaker_waveform is not None and speaker_waveform.lengths is not None
    assert out.mixture.shape == (1, 1, canvas_samples)
    assert out.target.shape == (1, canvas_samples)
    assert speaker_waveform.value.shape == (1, enroll_samples)
    assert int(out.sr) == SR
    assert out.lengths.tolist() == [canvas_samples]
    assert speaker_waveform.lengths.tolist() == [enroll_samples]
    assert out.spk_labels.tolist() == [expected_label]


def test_speaker_identity_to_idx_stable_sorted_mapping(tse_data):
    """speaker_identity_to_idx 按 strict canonical identity 稳定排序。"""
    ds1 = _build_dataset(tse_data)
    ds2 = _build_dataset(tse_data)
    assert ds1.speaker_identity_to_idx == ds2.speaker_identity_to_idx
    assert ds1.speaker_identity_to_idx == {
        ("fixture", "spk1"): 0,
        ("fixture", "spk2"): 1,
    }
    labels = {
        _target_row(ds1[idx])["spk_label"] for idx in range(len(ds1))
    }
    assert labels == {0, 1}


def test_enroll_fixed_length_and_rms(tse_data):
    """enroll 输出恒为 enrollment_duration 定长，且 RMS 归一到 enrollment_db。"""
    ds = _build_dataset(tse_data, enrollment_duration=0.25, enrollment_db=-20.0)
    for idx in range(len(ds)):
        sample = ds[idx]
        enroll = _enrollment(sample)
        cue = _target_row(sample)["cues"].speaker_waveform
        assert cue is not None and cue.length == int(round(0.25 * SR))
        assert len(enroll) == cue.length
        rms = float(np.sqrt(np.mean(np.square(enroll, dtype=np.float64))))
        rms_db = 20.0 * np.log10(max(rms, 1e-10))
        assert rms_db == pytest.approx(-20.0, abs=1.0)


def test_all_probs_zero_deterministic(tse_data):
    """概率全 0 时同一 epoch 种子下两次取样完全一致（确定性行为）。"""
    ds = _build_dataset(tse_data, enrollment_duration=0.5)
    _assert_same_sample(ds[0], ds[0])




def test_single_target_trace_capture_is_complete_and_capture_off_is_identical(tse_data):
    ds = _build_dataset(tse_data)
    plain = ds.get_sample_with_trace(0, capture_stems=False)
    traced = ds.get_sample_with_trace(0, capture_stems=True)
    _assert_same_sample(plain.sample, traced.sample)
    assert all(not src.stems for src in plain.trace.sources.values())
    np.testing.assert_array_equal(
        traced.trace.sources['scene'].stems['mixture'], traced.sample["mixture"][0]
    )
    np.testing.assert_array_equal(
        traced.trace.sources['target'].stems['final_target'],
        _target_row(traced.sample)["target"],
    )
    assert traced.trace.metadata_dict()['scene']['branch'] == 'prob_target_noise'
    json.dumps(traced.trace.metadata_dict(), allow_nan=False)



def test_trace_publishes_structured_layer_d_and_final_guard_facts(tse_data):
    ds = _build_dataset(tse_data, {'peak_normalization_mode':'joint'})
    scene = ds.get_sample_with_trace(0).trace.metadata_dict()['scene']
    assert scene['layer_d_facts']['stage_order'] == ['D-normalize', 'D-guard']
    assert scene['layer_d_facts']['normalize']['preserves_decomposition'] is True
    assert scene['final_peak_guard']['roles'] == ['mixture']



def test_layer_f_trace_reports_selector_miss(tse_data):
    """Layer F 已启用但 clipping selector 未命中时发布空 realized facts。"""
    ds = _build_dataset(
        tse_data,
        {'noisy_post_mix_degradations_enabled': True, 'prob_clip': 0.0, 'prob_target_only': 1.0},
    )

    traced = ds.get_sample_with_trace(0, capture_stems=False)

    assert len(traced.sample["targets"]) == 1
    assert traced.trace.metadata_dict()["scene"]["layer_f"] == {
        "noisy_post_mix_degradations_enabled": True,
        "applied_ops": [],
        "observation_changed": False,
        "clipping_skipped_due_to_layer_e": False,
        "clipping_facts": {},
    }


def test_layer_f_trace_publishes_actual_clipping_facts(tse_data):
    """fixed-rail 实际改变 observation 时把 realized clipping facts 发布到 scene。"""
    ds = _build_dataset(
        tse_data,
        {'noisy_post_mix_degradations_enabled': True, 'prob_clip': 1.0, 'clip_variants': ['fixed_rail'], 'clip_variant_probs': {'fixed_rail': 1.0}, 'clip_pre_drive_db_range': [12.0, 12.0], 'prob_target_only': 1.0},
    )

    traced = ds.get_sample_with_trace(0, capture_stems=False)

    assert len(traced.sample["targets"]) == 1
    layer_f = traced.trace.metadata_dict()["scene"]["layer_f"]
    assert layer_f["applied_ops"] == ["clip"]
    assert layer_f["observation_changed"] is True
    assert layer_f["clipping_skipped_due_to_layer_e"] is False
    facts = layer_f["clipping_facts"]
    assert facts["variant"] == "fixed_rail"
    assert facts["pre_drive_db"] == pytest.approx(12.0)
    assert facts["applied"] is True
    assert facts["changed_sample_count"] > 0
    assert facts["changed_sample_ratio"] > 0.0
    json.dumps(layer_f, allow_nan=False)


def test_layer_f_trace_reports_layer_e_clipping_skip(tse_data, monkeypatch):
    """Layer E clipping 实际生效时只跳过 Layer F clipping，并在 scene 明确发布。"""
    ds = _build_dataset(
        tse_data,
        {'noisy_post_mix_degradations_enabled': True, 'prob_clip': 1.0, 'prob_target_only': 1.0},
    )

    def _layer_e(waveform, _sr, _actual_len, _rng, _used_sidechain_wind):
        changed = (np.asarray(waveform) * np.float32(0.5)).astype(np.float32)
        return LayerEResult(
            waveform=changed,
            enabled=True,
            num_augmentations_drawn=1,
            selected=("clipping",),
            operations=(
                LayerEOperationResult(
                    name="clipping",
                    applied=True,
                    observation_changed=True,
                ),
            ),
            observation_changed=True,
        )

    monkeypatch.setattr(ds, "_apply_channel_degradation_result", _layer_e)
    traced = ds.get_sample_with_trace(0, capture_stems=False)

    assert len(traced.sample["targets"]) == 1
    layer_f = traced.trace.metadata_dict()["scene"]["layer_f"]
    assert layer_f == {
        "noisy_post_mix_degradations_enabled": True,
        "applied_ops": [],
        "observation_changed": False,
        "clipping_skipped_due_to_layer_e": True,
        "clipping_facts": {},
    }
    assert traced.trace.metadata_dict()["scene"]["layer_e"]["applied"] == [
        "clipping"
    ]


def test_target_only_without_reverb_equals_target(tse_data):
    """prob_target_only 分支（force_branch）：mixture == target。"""
    ds = _build_dataset(tse_data, enrollment_duration=0.5)
    traced = ds.get_sample_with_trace(0, force_branch="prob_target_only")
    assert traced.sample["sr"] == SR
    np.testing.assert_allclose(
        traced.sample["mixture"][0],
        _target_row(traced.sample)["target"],
        atol=1e-6,
    )

    branch_selection = traced.trace.metadata["branch_selection"]
    assert branch_selection == {
        "selected": "prob_target_only",
        "force_branch": "prob_target_only",
        "uniform_u": None,
    }
    assert traced.trace.metadata_dict()["scene"]["branch"] == "prob_target_only"



def test_source_volume_scene_gain_does_not_scale_enrollment(tse_data):
    """scene source gain 只作用于 mixture/target，enrollment 保持独立仿真电平。"""
    common = {'source_volume_normalization_enabled': True, 'source_volume_use_rms': True, 'use_high_pass': False, 'prob_target_only': 1.0}
    quiet = _build_dataset(
        tse_data,
        {**common, "source_volume_db_range": [-40.0, -40.0]},
        enrollment_duration=0.5,
        enroll_policy="fixed",
    )[0]
    loud = _build_dataset(
        tse_data,
        {**common, "source_volume_db_range": [-15.0, -15.0]},
        enrollment_duration=0.5,
        enroll_policy="fixed",
    )[0]

    assert not np.allclose(
        _target_row(quiet)["target"],
        _target_row(loud)["target"],
        rtol=1e-4,
        atol=1e-6,
    )
    quiet_enroll = _enrollment(quiet)
    loud_enroll = _enrollment(loud)
    np.testing.assert_array_equal(quiet_enroll, loud_enroll)
    quiet_enroll_rms = float(np.sqrt(np.mean(np.square(quiet_enroll, dtype=np.float64))))
    loud_enroll_rms = float(np.sqrt(np.mean(np.square(loud_enroll, dtype=np.float64))))
    assert 20.0 * np.log10(quiet_enroll_rms) == pytest.approx(-30.0, abs=1.0)
    assert 20.0 * np.log10(loud_enroll_rms) == pytest.approx(-30.0, abs=1.0)


def test_enroll_policy_fixed_picks_first_candidate(tse_data):
    """enroll_policy=fixed 时注册音固定取 spk 列表首条（概率全 0 → 可精确比对）。"""
    ds = _build_dataset(tse_data, enrollment_duration=0.5, enroll_policy="fixed")
    # 找到目标 uid 为 spk1_001 的样本（index 0 对应哪个 uid 由 dataset 内部决定，
    # 直接对两个 index 验证：enroll 必须是各自 spk 首条注册音经定长+RMS 归一的产物）
    import soundfile as sf

    expected_by_spk = {}
    for spk, paths in ds.spk2enroll.items():
        wav, _ = sf.read(paths[0], dtype="float32")
        n_target = int(round(0.5 * SR))
        fitted = EnrollmentAugmentor._fit_length(wav, n_target, ds._get_rng(0), "fixed")
        expected_by_spk[spk] = EnrollmentAugmentor._rms_normalize(fitted, ds.enrollment_db)

    checked = 0
    for idx in range(len(ds)):
        fs, i = ds._get_from_index(idx)
        uid = ds.speech_uids[fs][i]
        spk = ds.uid_to_speaker_identity[uid]
        np.testing.assert_allclose(
            _enrollment(ds[idx]), expected_by_spk[spk], atol=1e-5
        )
        checked += 1
    assert checked == len(ds)


def test_missing_enroll_candidates_raises(tse_data):
    """目标 spk 无注册音候选时报错（strict_load=True 下包装为 RuntimeError）。"""
    import json as _json

    ds = _build_dataset(tse_data, enrollment_duration=0.5)
    ds.spk2enroll = {}  # 清空候选
    with pytest.raises(RuntimeError, match="no enrollment candidates"):
        ds[0]


def test_sim_cfg_validation_music_ratio_requires_music_scp(tse_data):
    """music_ratio > 0 但 music 池为空（无 music_scp 且 noise 非分类 JSONL）时 fail fast。"""
    with pytest.raises(ValueError, match="music_ratio > 0 but no music entries"):
        _build_dataset(tse_data, sim_overrides={"music_ratio": 0.5})


def test_target_interference_requires_sir_range(tse_data):
    """prob_interference > 0 但缺 sir_range 时构造期 fail fast。"""
    with pytest.raises(ValueError, match="require.*sir_range"):
        _build_dataset(tse_data, sim_overrides={'prob_target_only': 0.0, 'prob_interference_only': 0.0, 'prob_noise_only': 0.0, 'prob_interference_noise_only': 0.0, 'prob_target_noise': 0.5, 'prob_target_interference': 0.0, 'prob_target_interference_noise': 0.5})


def test_export_simulation_stems_uses_base_capture_switch(tse_data):
    ds = _build_dataset(tse_data, export_simulation_stems=True)
    traced = ds.get_sample_with_trace(0)
    assert traced.trace.capture_stems
    np.testing.assert_array_equal(
        traced.trace.sources['scene'].stems['mixture'], traced.sample["mixture"][0]
    )



def test_music_scp_empty_pool_raises(tse_data, tmp_path):
    """music_scp 提供但解析为 0 条且 music_ratio > 0 时构造期 fail fast。"""
    empty_music = tmp_path / "music_empty.scp"
    empty_music.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="music_ratio > 0 but no music entries"):
        _build_dataset(
            tse_data,
            sim_overrides={"music_ratio": 0.5},
            music_scp=str(empty_music),
        )



def test_spk2enroll_zero_coverage_raises(tse_data, tmp_path):
    """nested enrollment 对 strict speech identity 覆盖为 0 时 fail fast。"""
    bad_json = tmp_path / "spk2enroll_bad.json"
    with bad_json.open("w", encoding="utf-8") as f:
        json.dump({"other": {"speaker": [tse_data["enroll1_wav"].as_posix()]}}, f)
    with pytest.raises(ValueError, match="coverage"):
        _build_dataset(tse_data, spk2enroll_json=str(bad_json))


def test_enroll_reverb_empty_rir_pool_fails_fast(tse_data, tmp_path):
    path=tmp_path/'rir_empty.scp';path.write_text('')
    with pytest.raises(ValueError,match='non-empty RIR provider'):
        _build_dataset(tse_data,enrollment_augment={'enabled':True,'prob_reverb':.5,'rir_scp':str(path)})





def test_tse_level_audits_separate_mixture_sir_and_enrollment_snr(tse_data):
    ds = _build_dataset(tse_data, {'prob_target_interference_noise':1.,'sir_range':[3.,3.]},
        enrollment_augment={'enabled':True,'prob_eq':0.,'prob_noise':1.,'snr_range':[12.,12.],'noise_scp':tse_data['noise_scp']})
    trace=ds.get_sample_with_trace(0).trace.metadata_dict()
    assert trace['scene']['sir'][0]['realized_db']==pytest.approx(3.,abs=1e-4)
    audit=trace['sources']['enrollment/0']['level_audit']
    assert audit['calibration_realized_db']==pytest.approx(12.,abs=1e-4)
    assert audit['final_realized_db']==pytest.approx(12.,abs=1e-4)
    json.dumps(trace,allow_nan=False)



def test_tse_level_audits_fail_closed_after_enrollment_eq(tse_data):
    ds=_build_dataset(tse_data,enrollment_augment={'enabled':True,'prob_eq':0.,'prob_eq_post':1.,'prob_noise':1.,'snr_range':[12.,12.],'noise_scp':tse_data['noise_scp']})
    audit=ds.get_sample_with_trace(0).trace.metadata_dict()['sources']['enrollment/0']['level_audit']
    assert audit['calibration_realized_db']==pytest.approx(12.,abs=1e-4)
    assert audit['final_realized_db'] is None and not audit['applicable']





def test_fit_enroll_length_short_tile_is_rolled(tse_data):
    """短注册音 tile 拼接后做循环移位：输出是硬拼接结果的某个循环移位。"""
    rng = np.random.default_rng(0)
    wav = np.sin(np.linspace(0, 50, 1000)).astype(np.float32)
    n_target = 2500
    out = EnrollmentAugmentor._fit_length(wav, n_target, rng, "random")
    assert len(out) == n_target
    tiled = np.tile(wav, (n_target + len(wav) - 1) // len(wav))[:n_target]
    matched = [s for s in range(n_target) if np.array_equal(out, np.roll(tiled, s))]
    assert len(matched) == 1


def test_fixed_short_enroll_keeps_tile_position_across_epochs(tse_data):
    """fixed 同时固定候选、裁剪及短语音接缝位置；独立增强仍由配置控制。"""
    ds = _build_dataset(tse_data, enrollment_duration=2.5, enroll_policy="fixed")
    ds.set_epoch(0, base_seed=123)
    e0 = _enrollment(ds[0])
    ds.set_epoch(1, base_seed=123)
    e1 = _enrollment(ds[0])
    assert len(e0) == len(e1) == int(round(2.5 * SR))
    np.testing.assert_array_equal(e0, e1)


def test_splice_sanity_check_uses_seconds_range_upper_bound(tse_data):
    """设 speech_splice_seconds_range 且不设固定值时，sanity check 按 range 上限校验。

    max_duration_seconds=1.0 小于固定值默认 20.0s：旧实现拿默认固定值对比必 raise；
    修复后按 range 上限（0.8s < 1.0s cap）应正常构建。不设 range 时固定默认 20s
    超过 1.0s cap，仍应 raise（行为不变）。
    """
    ds = _build_dataset(
        tse_data,
        sim_overrides={
            "speech_splice": True,
            "speech_splice_seconds_range": [0.5, 0.8],
        },
    )
    assert len(ds) > 0

    with pytest.raises(ValueError, match="speech_splice"):
        _build_dataset(
            tse_data,
            sim_overrides={"speech_splice": True},  # 无 range：走固定值分支
        )


def _write_classified_noise_jsonl(path: Path, records: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return str(path)


def _noise_record(uid: str, wav: Path, category_id: int, category: str, role: str) -> dict:
    return {
        "uid": uid,
        "fs": SR,
        "length": soundfile.info(str(wav)).frames,
        "path": str(wav),
        "category_id": category_id,
        "balance_category": category,
        "roles": [role],
    }


def test_pure_music_pool_allows_empty_ordinary_only_at_ratio_one(tse_data, tmp_path):
    """Music 分池后 ordinary 可空，但仅限每次读取必选 Music 的 ratio=1。"""
    noise_scp = _write_classified_noise_jsonl(
        tmp_path / "pure_music.jsonl",
        [_noise_record("embedded-music", tse_data["music_wav"], 10, "Music", "music")],
    )
    dataset = _build_dataset(
        tse_data,
        {"music_ratio": 1.0},
        noise_scp=noise_scp,
        music_scp=None,
    )
    assert not dataset.noise_flt
    assert set(dataset.music_flt) == {"jsonl_music_embedded-music"}
    assert dataset._select_tse_additive_source(SR, np.random.default_rng(7))[0] == "music"

    with pytest.raises(ValueError, match="ordinary noise pool is empty"):
        _build_dataset(
            tse_data,
            {"music_ratio": 0.999},
            noise_scp=noise_scp,
            music_scp=None,
        )


def test_enrollment_noise_pool_is_independent_of_mixture_music(tse_data, tmp_path):
    noise_scp = _write_classified_noise_jsonl(tmp_path/'pure_music.jsonl',
        [_noise_record('embedded-music',tse_data['music_wav'],10,'Music','music')])
    ds=_build_dataset(tse_data,{'music_ratio':1.},noise_scp=noise_scp,music_scp=None,
        enrollment_augment={'enabled':True,'prob_eq':0.,'prob_noise':1.,'noise_scp':tse_data['noise_scp']})
    trace=ds.get_sample_with_trace(0).trace.metadata_dict()
    assert 'Music' not in trace['sources']['enrollment/0']['enrollment_augment']['noise']['uid']
    assert trace['scene']['noise_sources'][0]['family']=='music'



@pytest.mark.parametrize("balance_mode", ["category", "temperature"])
def test_balanced_dual_music_override_and_explicit_pool_expansion(
    tse_data, tmp_path, balance_mode
):
    """balanced dual 每次 read 都应用 Music override，内嵌与显式池合并。"""
    noise_scp = _write_classified_noise_jsonl(
        tmp_path / f"classified_{balance_mode}.jsonl",
        [
            _noise_record("car", tse_data["noise_wav"], 1, "Car", "noise"),
            _noise_record("rain", tse_data["noise_wav"], 2, "Rain", "noise"),
            _noise_record("embedded-music", tse_data["music_wav"], 3, "Music", "music"),
        ],
    )
    sampling_cfg = SamplingConfig.from_dict(
        {
            "noise": {
                "balance_categories": True,
                "balance_mode": balance_mode,
                "alpha": 0.5,
            }
        }
    )
    dataset = _build_dataset(
        tse_data,
        {"music_ratio": 0.5},
        noise_scp=noise_scp,
        music_scp=tse_data["music_scp"],
        sampling_cfg=sampling_cfg,
    )
    assert set(dataset.music_flt) == {"music_001", "jsonl_music_embedded-music"}

    rng = np.random.default_rng(20260822)
    pairs = [dataset._select_tse_dual_sources(SR, rng) for _ in range(128)]
    assert any(len(pair) == 2 and pair[0][0] == "music" for pair in pairs)
    assert any(
        len(pair) == 2 and pair[0][0] == "ordinary" and pair[1][0] == "music"
        for pair in pairs
    )
    assert all(
        not (len(pair) == 2 and pair[0][0] == pair[1][0] == "music")
        for pair in pairs
    )
    for pair in pairs:
        ordinary = [source for source in pair if source[0] == "ordinary"]
        if len(ordinary) == 2:
            categories = [dataset._noise_category_id_by_uid[source[1]] for source in ordinary]
            assert categories[0] != categories[1]


@pytest.mark.parametrize("sampling_cfg", [None, "category", "temperature"])
def test_dual_ordinary_sources_are_different_categories(tse_data, tmp_path, sampling_cfg):
    """Natural/category/temperature 的 classified ordinary 双源均须异类。"""
    noise_scp = _write_classified_noise_jsonl(
        tmp_path / f"ordinary_{sampling_cfg}.jsonl",
        [
            _noise_record("car", tse_data["noise_wav"], 1, "Car", "noise"),
            _noise_record("rain", tse_data["noise_wav"], 2, "Rain", "noise"),
        ],
    )
    parsed_sampling = (
        None
        if sampling_cfg is None
        else SamplingConfig.from_dict(
            {
                "noise": {
                    "balance_categories": True,
                    "balance_mode": sampling_cfg,
                    "alpha": 0.5,
                }
            }
        )
    )
    dataset = _build_dataset(
        tse_data,
        {"music_ratio": 0.0},
        noise_scp=noise_scp,
        music_scp=None,
        sampling_cfg=parsed_sampling,
    )
    rng = np.random.default_rng(314159)
    for _ in range(32):
        pair = dataset._select_tse_dual_sources(SR, rng)
        assert len(pair) == 2
        categories = [dataset._noise_category_id_by_uid[source[1]] for source in pair]
        assert categories[0] != categories[1]






def test_music_selector_100k_oversized_pool_reuses_public_selector_cache(monkeypatch):
    """Music 超预算池复用公共 selector，连续抽样只物化一次 UID tuple。"""
    import core.data.dynamic_dataset_base as ddb

    monkeypatch.setattr(ddb, "_SAMPLE_SELECTOR_CACHE_MAX_UID_REFS", 3)

    class CountingKeysDict(dict):
        def __init__(self, items):
            super().__init__(items)
            self.keys_calls = 0

        def keys(self):
            self.keys_calls += 1
            return super().keys()

    music = CountingKeysDict((f"music-{index}", None) for index in range(100_001))
    dataset = TSEDynamicMixingDataset.__new__(TSEDynamicMixingDataset)
    dataset.sim_cfg = SimpleNamespace(music_ratio=1.0)
    dataset.music_source = {SR: music}
    dataset.music_flt = {uid: f"/{uid}.wav" for uid in music}
    dataset.balanced_noise_sampling_enabled = False

    cold_rng = np.random.default_rng(20260822)
    cold = [dataset._select_tse_additive_source(SR, cold_rng) for _ in range(64)]
    warm_rng = np.random.default_rng(20260822)
    warm = [dataset._select_tse_additive_source(SR, warm_rng) for _ in range(64)]

    assert warm == cold
    assert music.keys_calls == 1
    assert dataset._sample_selector_cache_uid_refs == len(music) > 3
    assert len(dataset._sample_selector_source_cache) == 1
    entry = next(iter(dataset._sample_selector_source_cache.values()))
    assert entry.source_by_fs is dataset.music_source
    assert len(entry.uids_by_fs[SR]) == len(music)


def test_default_music_policy_is_ordinary_and_preserves_selector_rng(tse_data):
    """music_ratio=0 保留原 family draw/UID draw 次序，并记录 Layer-C policy。"""
    dataset = _build_dataset(tse_data)
    rng = np.random.default_rng(20260827)
    expected_rng = np.random.default_rng(20260827)

    source = dataset._select_tse_additive_source(SR, rng)
    # 原路径的 ratio=0 仍先消费一次 Bernoulli，再经普通 selector 选择 UID。
    expected_rng.random()
    expected_uid = dataset._select_sample(SR, dataset.noise_source, rng=expected_rng)
    assert source == ("ordinary", expected_uid, dataset.noise_flt[expected_uid])
    assert rng.bit_generator.state == expected_rng.bit_generator.state

    record = {"noise_sources": [], "noise_source_policies": [], "noise_assets": []}
    dataset._active_acoustic_plan_record = record
    from core.data.background_sources import read_source

    read_source(dataset, source, SR, SR, np.random.default_rng(17), spatial=False)
    policy = record["noise_source_policies"][0]
    assert policy.family.value == "ordinary"
    assert policy.application_role.value == "layer_c_additive"
    assert policy.cache_provenance.value == "shared_audio_cache_unknown"


def test_enrollment_augment_disabled_is_bitwise_compatible(tse_data):
    """缺省和显式 disabled 不新增 RNG draw，canonical item 逐位一致。"""
    baseline = _build_dataset(tse_data, enrollment_duration=0.5)[0]
    disabled = _build_dataset(
        tse_data,
        enrollment_duration=0.5,
        enrollment_augment={"enabled": False},
    )[0]
    _assert_same_sample(baseline, disabled)


def test_enrollment_augment_only_changes_enrollment_and_trace_role(tse_data):
    """启用后只改变 enrollment，并在其 role-local trace 写 JSON-safe facts。"""
    baseline = _build_dataset(tse_data, enrollment_duration=0.5, enroll_policy="fixed")
    augmented = _build_dataset(
        tse_data,
        enrollment_duration=0.5,
        enroll_policy="fixed",
        enrollment_augment={"enabled": True, "prob_eq": 1.0, "eq_db_range": [-9, -9]},
    )
    expected = baseline[0]
    traced = augmented.get_sample_with_trace(0, capture_stems=False)
    actual = traced.sample

    np.testing.assert_array_equal(actual["mixture"], expected["mixture"])
    np.testing.assert_array_equal(
        _target_row(actual)["target"], _target_row(expected)["target"]
    )
    assert actual["sr"] == expected["sr"]
    assert actual["length"] == expected["length"]
    actual_enroll = _enrollment(actual)
    expected_enroll = _enrollment(expected)
    assert actual_enroll.shape == expected_enroll.shape
    assert not np.array_equal(actual_enroll, expected_enroll)
    trace = traced.trace.metadata_dict()
    assert "enrollment_augment" not in trace["sources"]["scene"]
    facts = trace["sources"]["enrollment/0"]["enrollment_augment"]
    assert facts["operations"] == ["eq_pre", "rms"] and facts["applied"] is True
    assert not traced.trace.get_source("enrollment/0").stems
    json.dumps(trace, allow_nan=False)


def test_enrollment_augment_rir_noise_require_configured_resources(tse_data):
    """local RIR/noise gate 在构造期要求非空资源 provider。"""
    common = dict(enrollment_duration=0.5)
    with pytest.raises(ValueError, match="prob_reverb > 0 requires non-empty rir_scp"):
        _build_dataset(
            tse_data,
            **common,
            enrollment_augment={"enabled": True, "prob_reverb": 1.0},
        )
    with pytest.raises(ValueError, match="prob_noise > 0 requires non-empty noise_scp"):
        _build_dataset(
            tse_data,
            **common,
            enrollment_augment={"enabled": True, "prob_noise": 1.0},
        )


def test_enrollment_augment_provider_errors_do_not_return_placeholder(tse_data, tmp_path):
    """即使 Dataset lenient，local provider 损坏也必须 fail-closed。"""
    missing_rir_scp = tmp_path / "missing_rir.scp"
    missing_rir_scp.write_text(
        f"missing 16000 16000 {tmp_path / 'does-not-exist.wav'}\n", encoding="utf-8"
    )
    dataset = _build_dataset(
        tse_data,
        strict_load=False,
        enrollment_augment={
            "enabled": True,
            "prob_reverb": 1.0,
            "rir_scp": str(missing_rir_scp),
        },
    )

    with pytest.raises(RuntimeError, match="failed to read enrollment RIR provider"):
        dataset[0]






def test_local_enrollment_moving_rir_is_rejected_at_construction(tse_data, tmp_path):
    """local enrollment augmentor 当前仅支持静态卷积，构造期拒绝 moving RIR。"""
    import pickle

    moving_rir = tmp_path / "moving_local_rir.pkl"
    with moving_rir.open("wb") as handle:
        pickle.dump(
            {"moving": True, "rir": np.ones((2, 8), dtype=np.float32)}, handle
        )
    moving_scp = tmp_path / "moving_local_rir.scp"
    moving_scp.write_text(
        f"moving-local {SR} 8 {moving_rir}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="static RIR only.*moving RIR"):
        _build_dataset(
            tse_data,
            enrollment_augment={
                "enabled": True,
                "prob_reverb": 1.0,
                "rir_scp": str(moving_scp),
            },
        )




def test_enrollment_augment_rir_noise_only_changes_enrollment(tse_data):
    """local RIR/noise provider 可读取资源，且不改变 mixture/target。"""
    baseline = _build_dataset(tse_data, enrollment_duration=0.5, enroll_policy="fixed")
    augmented = _build_dataset(
        tse_data,
        enrollment_duration=0.5,
        enroll_policy="fixed",
        enrollment_augment={
            "enabled": True,
            "prob_eq": 0.0,
            "prob_reverb": 1.0,
            "rir_scp": tse_data["rir_scp"],
            "prob_noise": 1.0,
            "noise_scp": tse_data["noise_scp"],
            "snr_range": [10.0, 10.0],
        },
    )

    expected = baseline[0]
    traced = augmented.get_sample_with_trace(0, capture_stems=False)
    actual = traced.sample
    np.testing.assert_array_equal(actual["mixture"], expected["mixture"])
    np.testing.assert_array_equal(
        _target_row(actual)["target"], _target_row(expected)["target"]
    )
    assert actual["sr"] == expected["sr"]
    assert actual["length"] == expected["length"]
    actual_enroll = _enrollment(actual)
    expected_enroll = _enrollment(expected)
    assert actual_enroll.shape == expected_enroll.shape
    assert np.isfinite(actual_enroll).all()
    assert not np.array_equal(actual_enroll, expected_enroll)


def test_enrollment_augment_data_relative_provider_paths(tse_data, tmp_path, monkeypatch):
    """嵌套 provider SCP 的 ``data/...`` 路径锚定仓库根。"""
    import core.data.resource_paths as resource_paths

    repo_root = tmp_path / "repo"
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True)
    rir_scp = data_dir / "local_rir.scp"
    noise_scp = data_dir / "local_noise.scp"
    rir_scp.write_text(Path(tse_data["rir_scp"]).read_text(encoding="utf-8"), encoding="utf-8")
    noise_scp.write_text(
        Path(tse_data["noise_scp"]).read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setattr(resource_paths, "repository_root", lambda: repo_root)

    dataset = _build_dataset(
        tse_data,
        enrollment_augment={
            "enabled": True,
            "prob_reverb": 1.0,
            "rir_scp": "data/local_rir.scp",
            "prob_noise": 1.0,
            "noise_scp": "data/local_noise.scp",
        },
    )

    expected_rir_path = Path(tse_data["rir_scp"]).read_text(encoding="utf-8").split()[3]
    expected_noise_path = Path(tse_data["noise_scp"]).read_text(encoding="utf-8").split()[3]
    assert dataset._enrollment_augment_rirs == (("rir_001", expected_rir_path),)
    assert dataset._enrollment_augment_noises == (("noise_001", expected_noise_path),)


pytestmark = [pytest.mark.unit]
