"""Tests for speaker-axis 多目标（targets_per_mixture > 1，省内存版）。

覆盖：
  - canonical dataset dict 与 per-target typed cue，主目标恒为 targets[0]；
  - 备选目标 = 干扰说话人的独立 clean stem（与 mixture 残差一致）；
  - 与默认路径（targets_per_mixture=1）的 mixture/target 一致性；
  - Layer D normalize stage 的 joint 模式对所有 alternate targets 施加共同 follower scale；
  - collate 端到端打包 validated TSEBatch 与 num_targets；
  - repeat_to_max_duration 对所有 scene targets 重放同一 time plan；
  - 构造期 fail fast（非法 targets_per_mixture 及 separate/single D-normalize）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

import core.data.acoustic_scene_replay as acoustic_scene_replay
import core.data.dynamic_dataset_base as dynamic_dataset_base
from core.data.acoustic_scene_plan import C0AssetReplayError, ResolvedAssets
from core.data.post_mix_pipeline import POST_MIX_STAGE_ORDER
from tasks.tse.data.collate import tse_collate_fn
from tasks.tse.data.batch import TSEBatch
from tasks.tse.data.dynamic_dataset import (
    EnrollmentAugmentationProviderError,
    TSEDynamicMixingDataset,
)

SR = 16000


def _write_scp(path: Path, entries: dict[str, Path]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for uid, audio_path in entries.items():
            info = soundfile.info(str(audio_path))
            f.write(f"{uid} {info.samplerate} {info.frames} {audio_path}\n")


def _make_wav(path: Path, duration: float, sr: int, *, seed: int = 0, freq: float = 300.0) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    signal = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    signal += 0.01 * rng.standard_normal(signal.shape).astype(np.float32)
    soundfile.write(path, (signal * 0.5).astype(np.float32), sr)


def _assert_single_scaled_utterance_on_canvas(
    target: np.ndarray, source: np.ndarray, source_samples: int
) -> None:
    """验证画布只含一次完整源波形，允许 SIR/peak guard 统一缩放。"""
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


def _make_impulse_rir(path: Path, sr: int) -> None:
    rir = np.zeros(sr, dtype=np.float32)
    rir[0] = 1.0
    soundfile.write(path, rir, sr)


@pytest.fixture
def tse_data(tmp_path: Path) -> dict:
    """最小 TSE 资源：2 说话人 + impulse RIR（mixture - 主目标 ≈ 备选目标）。"""
    speech_dir = tmp_path / "speech"
    noise_dir = tmp_path / "noise"
    rir_dir = tmp_path / "rir"
    wind_dir = tmp_path / "wind"
    enroll_dir = tmp_path / "enroll"
    for d in (speech_dir, noise_dir, rir_dir, wind_dir, enroll_dir):
        d.mkdir()

    spk1_wav = speech_dir / "spk1_001.wav"
    spk2_wav = speech_dir / "spk2_001.wav"
    noise_wav = noise_dir / "noise_001.wav"
    rir_wav = rir_dir / "rir_001.wav"
    wind_wav = wind_dir / "wind_001.wav"
    enroll1_wav = enroll_dir / "spk1_enroll.wav"
    enroll2_wav = enroll_dir / "spk2_enroll.wav"

    _make_wav(spk1_wav, 2.0, SR, seed=1, freq=300.0)
    _make_wav(spk2_wav, 2.0, SR, seed=2, freq=500.0)
    _make_wav(noise_wav, 2.0, SR, seed=3)
    _make_impulse_rir(rir_wav, SR)
    _make_wav(wind_wav, 2.0, SR, seed=4)
    _make_wav(enroll1_wav, 1.0, SR, seed=6, freq=320.0)
    _make_wav(enroll2_wav, 1.0, SR, seed=8, freq=520.0)

    speech_jsonl = tmp_path / "speech.jsonl"
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    wind_scp = tmp_path / "wind.scp"
    speech_jsonl.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"uid": "spk1_001", "fs": SR, "length": 2 * SR, "path": str(spk1_wav), "language": "zh", "dataset_id": "fixture", "speaker_id": "spk1"},
                {"uid": "spk2_001", "fs": SR, "length": 2 * SR, "path": str(spk2_wav), "language": "zh", "dataset_id": "fixture", "speaker_id": "spk2"},
            )
        ) + "\n",
        encoding="utf-8",
    )
    _write_scp(noise_scp, {"noise_001": noise_wav})
    _write_scp(rir_scp, {"rir_001": rir_wav})
    _write_scp(wind_scp, {"wind_001": wind_wav})
    spk2enroll_json = tmp_path / "spk2enroll.json"
    with spk2enroll_json.open("w", encoding="utf-8") as f:
        json.dump({"fixture": {"spk1": [str(enroll1_wav)], "spk2": [str(enroll2_wav)]}}, f)

    return {
        "speech_scp": str(speech_jsonl),
        "noise_scp": str(noise_scp),
        "rir_scp": str(rir_scp),
        "windnoise_scp": str(wind_scp),
        "spk2enroll_json": str(spk2enroll_json),
    }


def _base_sim_cfg() -> dict:
    return {'snr_low': 100.0, 'snr_high': 100.0, 'prob_reverberation': 0.0, 'prob_channel_enabled': False, 'noisy_post_mix_degradations_enabled': False, 'sir_range': [0.0, 0.0], 'music_ratio': 0.0, 'prob_target_only': 0.0, 'prob_interference_only': 0.0, 'prob_noise_only': 0.0, 'prob_interference_noise_only': 0.0, 'prob_target_noise': 0.0, 'prob_target_interference': 0.0, 'prob_target_interference_noise': 1.0, 'num_interferers': {1: 1.0}}


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
        enrollment_duration=0.5,
    )
    params.update(kwargs)
    ds = TSEDynamicMixingDataset(**params)
    ds.set_epoch(0, base_seed=123)
    return ds


def _spk1_index(ds) -> int:
    """找到目标 uid 为 spk1_001 的样本下标（spk2 可作干扰）。"""
    for idx in range(len(ds)):
        fs, i = ds._get_from_index(idx)
        if ds.speech_uids[fs][i] == "spk1_001":
            return idx
    raise AssertionError("no spk1_001 sample found")


def _target(sample: dict, index: int = 0) -> dict:
    return sample["targets"][index]


def _speaker_cue(sample: dict, index: int = 0):
    cue = _target(sample, index)["cues"].speaker_waveform
    assert cue is not None
    return cue


def _enrollment(sample: dict, index: int = 0) -> np.ndarray:
    cue = _speaker_cue(sample, index)
    assert cue.present and cue.value is not None
    return np.asarray(cue.value)


def test_multi_target_sample_structure(tse_data):
    """canonical dict 保持主目标优先和 sample-major 备选目标顺序。"""
    ds = _build_dataset(tse_data, targets_per_mixture=2)
    sample = ds[_spk1_index(ds)]
    assert isinstance(sample, dict)
    assert set(sample) == {
        "key",
        "mixture",
        "sr",
        "length",
        "targets",
        "geometry",
        "timebase",
        "provenance",
    }
    assert sample["mixture"].shape == (1, sample["length"])
    assert sample["sr"] == SR
    assert len(sample["targets"]) == 2
    main = _target(sample, 0)
    alternate = _target(sample, 1)
    main_enroll = _enrollment(sample, 0)
    alternate_enroll = _enrollment(sample, 1)
    main_cue = _speaker_cue(sample, 0)
    alternate_cue = _speaker_cue(sample, 1)
    assert len(main["target"]) == sample["length"] == len(alternate["target"])
    assert main_cue.length == len(main_enroll) == int(round(0.5 * SR))
    assert alternate_cue.length == len(alternate_enroll)
    assert (main["spk_label"], alternate["spk_label"]) == (0, 1)
    assert (main["speaker_id"], alternate["speaker_id"]) == (
        "spk1_001",
        "spk2_001",
    )
    assert main["target_present"] is True
    assert alternate["target_present"] is True
    assert np.isfinite(alternate["target"]).all()
    assert np.isfinite(alternate_enroll).all()


def test_multi_target_trace_uses_role_local_namespaces(tse_data):
    ds=_build_dataset(tse_data,targets_per_mixture=2)
    traced=ds.get_sample_with_trace(_spk1_index(ds),capture_stems=False)
    assert {'source/0','source/1','scene','enrollment/0','enrollment/1'} <= set(traced.trace.sources)
    assert len(traced.trace.metadata_dict()['scene']['source_plans'])==2
    assert all(not source.stems for source in traced.trace.sources.values())



def test_multi_target_trace_capture_records_only_final_published_stems(tse_data):
    ds=_build_dataset(tse_data,targets_per_mixture=2)
    traced=ds.get_sample_with_trace(_spk1_index(ds),capture_stems=True)
    stems=traced.trace.sources['scene'].stems
    assert {'source_full_rir/0','source_full_rir/1','target/0','target/1','background','mixture'} <= set(stems)
    for i, target in enumerate(traced.sample['targets']):
        np.testing.assert_array_equal(stems[f'target/{i}'], target["target"])
        cue = target["cues"].speaker_waveform
        assert cue is not None and cue.value is not None
        np.testing.assert_array_equal(
            traced.trace.sources[f'enrollment/{i}'].stems['final_enrollment'],
            cue.value,
        )



def test_multi_target_alt_matches_residual(tse_data):
    """impulse RIR + 高 SNR 下 mixture - 主目标 ≈ 备选目标（干扰 clean stem）。"""
    ds = _build_dataset(tse_data, targets_per_mixture=2)
    sample = ds[_spk1_index(ds)]
    residual = sample["mixture"][0] - _target(sample, 0)["target"]
    alternate = _target(sample, 1)["target"]
    denom = float(np.linalg.norm(residual) * np.linalg.norm(alternate) + 1e-10)
    corr = float(np.dot(residual, alternate) / denom)
    assert corr > 0.99


def test_multi_target_joint_d_normalize_scales_all_targets(tse_data):
    off=_build_dataset(tse_data,{'peak_normalization_mode':'off'},targets_per_mixture=2)
    joint=_build_dataset(tse_data,{'peak_normalization_mode':'joint','peak_normalization_scale_range':[.25,.25]},targets_per_mixture=2)
    a=off.get_sample_with_trace(_spk1_index(off));b=joint.get_sample_with_trace(_spk1_index(joint))
    af=a.trace.metadata_dict()['scene'];bf=b.trace.metadata_dict()['scene']
    gain=(bf['layer_d_facts']['normalize']['scale']*bf['layer_d_facts']['guard']['scale'])/(af['layer_d_facts']['normalize']['scale']*af['layer_d_facts']['guard']['scale'])
    for index, (t0, t1) in enumerate(zip(a.sample['targets'], b.sample['targets'])):
        np.testing.assert_allclose(t1["target"], t0["target"] * gain, atol=1e-6)
        np.testing.assert_array_equal(
            _enrollment(b.sample, index), _enrollment(a.sample, index)
        )
    mix_gain=gain*bf['final_peak_guard']['scale']/af['final_peak_guard']['scale']
    np.testing.assert_allclose(b.sample['mixture'],a.sample['mixture']*mix_gain,atol=1e-6)



def test_multi_target_mixture_matches_default_path(tse_data):
    """多目标模式与默认路径（targets_per_mixture=1）的 mixture/主目标完全一致。

    备选目标通过重放 rng 状态计算，不污染主路径抽样序列。
    """
    ds_def = _build_dataset(tse_data)
    ds_mt = _build_dataset(tse_data, targets_per_mixture=2)
    idx = _spk1_index(ds_def)
    sample_default = ds_def[idx]
    sample_mt = ds_mt[idx]
    np.testing.assert_allclose(
        sample_mt["mixture"], sample_default["mixture"], atol=1e-6
    )
    np.testing.assert_allclose(
        _target(sample_mt)["target"],
        _target(sample_default)["target"],
        atol=1e-6,
    )








def test_multi_target_no_interference_single_target(tse_data):
    """无干扰的样本（prob_interference=0）dict 中只有 1 个目标。"""
    ds = _build_dataset(tse_data, {'prob_target_only': 0.0, 'prob_interference_only': 0.0, 'prob_noise_only': 0.0, 'prob_interference_noise_only': 0.0, 'prob_target_noise': 1.0, 'prob_target_interference': 0.0, 'prob_target_interference_noise': 0.0}, targets_per_mixture=2)
    sample = ds[0]
    assert isinstance(sample, dict)
    assert len(sample["targets"]) == 1


def test_multi_target_deterministic(tse_data):
    """多目标模式同 (epoch, base_seed) 两次取样一致。"""
    ds = _build_dataset(tse_data, targets_per_mixture=2)
    idx = _spk1_index(ds)
    a = ds[idx]
    b = ds[idx]
    np.testing.assert_array_equal(a["mixture"], b["mixture"])
    assert len(a["targets"]) == len(b["targets"])
    for index, (ta, tb) in enumerate(zip(a["targets"], b["targets"])):
        np.testing.assert_array_equal(ta["target"], tb["target"])
        np.testing.assert_array_equal(_enrollment(a, index), _enrollment(b, index))
        assert ta["speaker_id"] == tb["speaker_id"]
        assert ta["spk_label"] == tb["spk_label"]
        assert ta["target_present"] is tb["target_present"]
        assert _speaker_cue(a, index).length == _speaker_cue(b, index).length


def test_multi_target_collate_end_to_end(tse_data):
    """dataset → collate：mixture 按 B 保留，target/cue 按 sample-major R 展开。"""
    ds = _build_dataset(tse_data, targets_per_mixture=2)
    idx = _spk1_index(ds)
    items = [ds[idx], ds[idx]]
    batch = tse_collate_fn(items, cue_specs=ds.cue_specs)
    speaker_waveform = batch.cues.speaker_waveform
    assert speaker_waveform is not None and speaker_waveform.lengths is not None
    assert batch.mixture.shape == (2, 1, SR)
    assert batch.target.shape[0] == int(batch.num_targets.sum()) == 4
    assert speaker_waveform.value.shape[0] == 4
    assert batch.num_targets.tolist() == [2, 2]
    assert batch.spk_labels.tolist() == [0, 1, 0, 1]
    assert batch.target_present.tolist() == [True, True, True, True]
    expanded = torch_repeat_interleave_check(batch)
    assert expanded == 4


def test_multi_target_fixed_silence_canvas_repeat_false_end_to_end(
    tse_data, monkeypatch
):
    """模式 B：主目标与干扰各只读、放置一次，并按目标行展开 TSEBatch。"""
    canvas_samples = 3 * SR
    source_samples = 2 * SR
    enroll_samples = int(round(0.5 * SR))
    ds = _build_dataset(
        tse_data,
        {
            "speech_splice": True,
            "speech_splice_seconds": 3.0,
            "silence_pad_prob": 1.0,
            "silence_pad_position": "front",
            "use_high_pass": False,
        },
        targets_per_mixture=2,
        max_duration_seconds=3.0,
        repeat_to_max_duration=False,
    )
    idx = _spk1_index(ds)
    expected_uids = ["spk1_001", "spk2_001"]
    expected_paths = [ds.speech_flt[uid] for uid in expected_uids]
    speech_paths = set(ds.speech_flt.values())
    speech_reads: list[tuple[str, np.ndarray]] = []
    original_read = ds._read_one_speech

    def _record_speech_read(path, sr):
        wav = original_read(path, sr)
        if path in speech_paths:
            speech_reads.append((path, wav.copy()))
        return wav

    monkeypatch.setattr(ds, "_read_one_speech", _record_speech_read)
    sample = ds[idx]
    assert [path for path, _ in speech_reads] == expected_paths

    assert sample["length"] == sample["mixture"].shape[-1] == canvas_samples
    assert sample["mixture"].shape == (1, canvas_samples)
    assert len(sample["targets"]) == 2
    main_target = _target(sample, 0)["target"]
    alternate_target = _target(sample, 1)["target"]
    main_enroll = _enrollment(sample, 0)
    alternate_enroll = _enrollment(sample, 1)
    assert len(main_target) == len(alternate_target) == sample["length"]
    _assert_single_scaled_utterance_on_canvas(
        main_target, speech_reads[0][1], source_samples
    )
    _assert_single_scaled_utterance_on_canvas(
        alternate_target, speech_reads[1][1], source_samples
    )
    assert _speaker_cue(sample, 0).length == len(main_enroll) == enroll_samples
    assert _speaker_cue(sample, 1).length == len(alternate_enroll) == enroll_samples
    assert tuple(row["spk_label"] for row in sample["targets"]) == (0, 1)

    batch = tse_collate_fn([sample], cue_specs=ds.cue_specs)
    speaker_waveform = batch.cues.speaker_waveform
    assert speaker_waveform is not None and speaker_waveform.lengths is not None
    assert batch.mixture.shape == (1, 1, canvas_samples)
    assert batch.target.shape == (2, canvas_samples)
    assert speaker_waveform.value.shape == (2, enroll_samples)
    assert int(batch.sr) == SR
    assert batch.lengths.tolist() == [canvas_samples]
    assert speaker_waveform.lengths.tolist() == [enroll_samples, enroll_samples]
    assert batch.spk_labels.tolist() == [0, 1]
    assert batch.num_targets.tolist() == [2]




def torch_repeat_interleave_check(tb: TSEBatch) -> int:
    """模拟 trainer 侧展开：stft(mixture) 一次后按 num_targets 展开的行数。"""
    import torch

    mix_spec = torch.randn(tb.mixture.shape[0], 10, 10)  # 假设的谱
    expanded = torch.repeat_interleave(mix_spec, tb.num_targets.long(), dim=0)
    return expanded.shape[0]


# ----------------------------------------------------------------------
# 构造期 fail fast
# ----------------------------------------------------------------------
def test_targets_per_mixture_invalid(tse_data):
    with pytest.raises(ValueError, match="targets_per_mixture must be >= 1"):
        _build_dataset(tse_data, targets_per_mixture=0)


def test_multi_target_repeat_to_max_duration_keeps_all_targets_aligned(tse_data):
    ds = _build_dataset(
        tse_data,
        targets_per_mixture=2,
        repeat_to_max_duration=True,
        repeat_tile_silence_gap=True,
        repeat_tile_silence_gap_range=(0.01, 0.01),
    )

    sample = ds[_spk1_index(ds)]

    assert sample["length"] == sample["mixture"].shape[-1]
    assert len(sample["targets"]) == 2
    assert all(
        len(target["target"]) == sample["length"]
        for target in sample["targets"]
    )


@pytest.mark.parametrize("mode", ["separate", "single"])
def test_multi_target_non_joint_d_normalize_unsupported(tse_data, mode):
    with pytest.raises(ValueError, match="off.*joint"):
        _build_dataset(
            tse_data,
            {"peak_normalization_mode": mode},
            targets_per_mixture=2,
        )


def test_placeholder_matches_multi_target_structure(tse_data):
    """失败占位保留目标 cardinality，并显式标记 speaker cue 缺失。"""
    ds = _build_dataset(tse_data, targets_per_mixture=2, strict_load=False)
    placeholder = ds._placeholder_sample(SR, 0, "spk1_001")
    assert isinstance(placeholder, dict)
    assert set(placeholder) == {
        "key",
        "mixture",
        "sr",
        "length",
        "targets",
        "geometry",
        "timebase",
        "provenance",
    }
    assert placeholder["mixture"].shape == (1, placeholder["length"])
    assert len(placeholder["targets"]) == 1
    row = _target(placeholder)
    cue = _speaker_cue(placeholder)
    assert row["spk_label"] == -1
    assert row["speaker_id"] == "spk1_001"
    assert row["target_present"] is True
    assert np.count_nonzero(row["target"]) == 0
    assert cue.present is False and cue.value is None and cue.length == 0
    batch = tse_collate_fn([placeholder], cue_specs=ds.cue_specs)
    speaker_waveform = batch.cues.speaker_waveform
    assert speaker_waveform is not None and speaker_waveform.lengths is not None
    assert batch.num_targets.tolist() == [1]
    assert batch.target_present.tolist() == [True]
    assert speaker_waveform.present.tolist() == [False]
    assert speaker_waveform.lengths.tolist() == [0]


# ----------------------------------------------------------------------
# 多目标模式的干扰说话人 enrollment 覆盖
# ----------------------------------------------------------------------
def _drop_spk2_enroll(tse_data: dict) -> None:
    """把 spk2 从 spk2enroll_json 中删掉（模拟干扰池覆盖缺口）。"""
    p = Path(tse_data["spk2enroll_json"])
    with p.open(encoding="utf-8") as f:
        m = json.load(f)
    m["fixture"].pop("spk2", None)
    with p.open("w", encoding="utf-8") as f:
        json.dump(m, f)


def test_multi_target_missing_interf_enroll_fail_fast(tse_data):
    """targets_per_mixture > 1：干扰池 spk 在 spk2enroll 无候选 → 构造期 raise。"""
    _drop_spk2_enroll(tse_data)
    with pytest.raises(ValueError, match="enrollment candidates"):
        _build_dataset(tse_data, targets_per_mixture=2)


def test_single_target_missing_interf_enroll_warn_only(tse_data):
    """targets_per_mixture = 1：同样的覆盖缺口只 warning 不 raise（现状不变）。"""
    _drop_spk2_enroll(tse_data)
    ds = _build_dataset(tse_data)  # 不抛错即通过
    assert ds.targets_per_mixture == 1


@pytest.mark.parametrize("strict_load", [True, False])
def test_multi_target_alt_enroll_missing_fails_closed(tse_data, strict_load):
    """运行期 enrollment 缺失必须抛错，不能发布稀疏 alternate/enrollment role。"""
    ds = _build_dataset(
        tse_data, targets_per_mixture=2, strict_load=strict_load
    )
    # 绕过构造期 fail fast，模拟运行期覆盖缺口（防御性兜底路径）。
    ds.spk2enroll[("fixture", "spk2")] = []
    idx = _spk1_index(ds)
    with pytest.raises(
        EnrollmentAugmentationProviderError,
        match="no enrollment",
    ):
        ds[idx]


def test_multi_target_duplicate_interferer_is_rejected(tse_data, monkeypatch):
    from core.data.dynamic_dataset_base import SampleBuildContractError
    ds=_build_dataset(tse_data,targets_per_mixture=3)
    monkeypatch.setattr(ds,"_sample_interference_uids",lambda *args:["spk2_001","spk2_001"])
    with pytest.raises(SampleBuildContractError,match="distinct non-target"):
        ds[_spk1_index(ds)]


def test_multi_target_alt_enroll_construction_failure_is_nonrecoverable(
    tse_data, monkeypatch
):
    """备选 enrollment 构造失败时，lenient 模式也不得回退为 placeholder。"""
    ds = _build_dataset(tse_data, targets_per_mixture=2, strict_load=False)
    original_build = ds._build_enrollment

    def _fail_alternate(uid, sr, rng, **kwargs):
        if kwargs.get("role") == "enrollment/1":
            raise ValueError("mutated enrollment provider")
        return original_build(uid, sr, rng, **kwargs)

    monkeypatch.setattr(ds, "_build_enrollment", _fail_alternate)
    with pytest.raises(RuntimeError, match="failed to construct enrollment for alternate"):
        ds[_spk1_index(ds)]
    assert ds._last_simulation_trace is None






pytestmark = [pytest.mark.unit]
