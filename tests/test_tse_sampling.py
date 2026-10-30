"""Tests for TSE strict JSONL canonical speaker identity.

动态 TSE 只接受 strict JSONL speech manifest：``speaker_id`` 是 canonical
identity（``(dataset_id, speaker_id)``）的主事实来源，enrollment 必须采用 nested
``dataset_id -> speaker_id -> [paths]`` 结构。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

from core.data.sampling import SamplingConfig
from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset

SR = 16000


def _make_wav(path: Path, duration: float, sr: int, *, seed: int = 0, freq: float = 300.0) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    signal = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    signal += 0.01 * rng.standard_normal(signal.shape).astype(np.float32)
    soundfile.write(path, (signal * 0.5).astype(np.float32), sr)


def _make_impulse_rir(path: Path, sr: int) -> None:
    rir = np.zeros(sr, dtype=np.float32)
    rir[0] = 1.0
    soundfile.write(path, rir, sr)


def _write_scp(path: Path, entries: dict[str, Path]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for uid, audio_path in entries.items():
            info = soundfile.info(str(audio_path))
            f.write(f"{uid} {info.samplerate} {info.frames} {audio_path}\n")


def _write_strict_speech_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _base_sim_cfg() -> dict:
    return {'snr_low': 5.0, 'snr_high': 20.0, 'prob_reverberation': 0.0, 'prob_channel_enabled': False, 'noisy_post_mix_degradations_enabled': False, 'music_ratio': 0.0, 'prob_target_only': 0.0, 'prob_interference_only': 0.0, 'prob_noise_only': 0.0, 'prob_interference_noise_only': 0.0, 'prob_target_noise': 1.0, 'prob_target_interference': 0.0, 'prob_target_interference_noise': 0.0}


@pytest.fixture
def strict_tse_data(tmp_path: Path) -> dict:
    """strict JSONL fixture：两条相同裸 speaker（"001"）、不同 dataset 的记录。"""
    speech_dir = tmp_path / "speech"
    enroll_dir = tmp_path / "enroll"
    for d in (speech_dir, enroll_dir, tmp_path / "noise", tmp_path / "rir", tmp_path / "wind"):
        d.mkdir(exist_ok=True)

    wav_a = speech_dir / "a.wav"
    wav_b = speech_dir / "b.wav"
    enroll_a = enroll_dir / "enroll_a.wav"
    enroll_b = enroll_dir / "enroll_b.wav"
    noise_wav = tmp_path / "noise" / "n.wav"
    rir_wav = tmp_path / "rir" / "r.wav"
    wind_wav = tmp_path / "wind" / "w.wav"
    _make_wav(wav_a, 2.0, SR, seed=1, freq=300.0)
    _make_wav(wav_b, 2.0, SR, seed=2, freq=500.0)
    _make_wav(enroll_a, 1.0, SR, seed=3, freq=320.0)
    _make_wav(enroll_b, 1.0, SR, seed=4, freq=520.0)
    _make_wav(noise_wav, 2.0, SR, seed=5)
    _make_impulse_rir(rir_wav, SR)
    _make_wav(wind_wav, 2.0, SR, seed=6)

    speech_jsonl = tmp_path / "speech.jsonl"
    _write_strict_speech_jsonl(
        speech_jsonl,
        [
            {"uid": "a", "fs": SR, "length": 2 * SR, "path": str(wav_a),
             "language": "zh", "dataset_id": "set-a", "speaker_id": "001"},
            {"uid": "b", "fs": SR, "length": 2 * SR, "path": str(wav_b),
             "language": "zh", "dataset_id": "set-b", "speaker_id": "001"},
        ],
    )
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    wind_scp = tmp_path / "wind.scp"
    _write_scp(noise_scp, {"n": noise_wav})
    _write_scp(rir_scp, {"r": rir_wav})
    _write_scp(wind_scp, {"w": wind_wav})

    nested_enroll = tmp_path / "spk2enroll.json"
    nested_enroll.write_text(
        json.dumps(
            {
                "set-a": {"001": [str(enroll_a)]},
                "set-b": {"001": [str(enroll_b)]},
            }
        ),
        encoding="utf-8",
    )
    return {
        "speech_scp": str(speech_jsonl),
        "noise_scp": str(noise_scp),
        "rir_scp": str(rir_scp),
        "windnoise_scp": str(wind_scp),
        "spk2enroll_json": str(nested_enroll),
    }


def _build_strict_dataset(strict_tse_data: dict, sim_overrides: dict | None = None, **kwargs) -> TSEDynamicMixingDataset:
    sim_cfg = _base_sim_cfg()
    if sim_overrides:
        from core.data.simulation.tse_scenarios import TSE_SCENARIO_KEYS
        if any(k in TSE_SCENARIO_KEYS for k in sim_overrides):
            sim_cfg.update({k: 0.0 for k in TSE_SCENARIO_KEYS})
        sim_cfg.update(sim_overrides)
    params = dict(
        noise_scp=strict_tse_data["noise_scp"],
        rir_scp=strict_tse_data["rir_scp"],
        windnoise_scp=strict_tse_data["windnoise_scp"],
        spk2enroll_json=strict_tse_data["spk2enroll_json"],
        simulation_cfg=sim_cfg,
        max_duration_seconds=1.0,
        target_sr=SR,
        filter_by_duration=True,
        strict_load=True,
    )
    params.update(kwargs)
    return TSEDynamicMixingDataset(speech_scp=strict_tse_data["speech_scp"], **params)


# ---------------------------------------------------------------------------
# Step 1: strict JSONL 主事实
# ---------------------------------------------------------------------------


def test_strict_jsonl_identity_is_canonical_per_dataset(strict_tse_data):
    """同一裸 speaker 跨 dataset 是独立 canonical identity。"""
    dataset = _build_strict_dataset(strict_tse_data)

    assert dataset.uid_to_speaker_identity == {
        "a": ("set-a", "001"),
        "b": ("set-b", "001"),
    }
    assert len(dataset.speaker_identity_to_idx) == 2
    assert set(dataset.spk2enroll) == {("set-a", "001"), ("set-b", "001")}


def test_strict_identity_validates_entire_pool_even_duration_ineligible(
    strict_tse_data, tmp_path
):
    """duration-ineligible 记录的非法 identity 也必须构造期 fail-fast。"""
    speech_jsonl = tmp_path / "speech_bad.jsonl"
    _write_strict_speech_jsonl(
        speech_jsonl,
        [
            {"uid": "good", "fs": SR, "length": 2 * SR, "path": "x.wav",
             "language": "zh", "dataset_id": "set-a", "speaker_id": "001"},
            # 0.1s < filter_min_duration(0.5s 默认在 _base_sim_cfg 之外由 filter 参数给)
            {"uid": "bad", "fs": SR, "length": SR // 10, "path": "y.wav",
             "language": "zh", "dataset_id": "set-a", "speaker_id": "unknown"},
        ],
    )
    strict_tse_data["speech_scp"] = str(speech_jsonl)
    with pytest.raises(ValueError, match="speaker_id must not be unknown"):
        _build_strict_dataset(strict_tse_data, filter_min_duration=0.5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("speaker_id", "unknown"),
        ("speaker_id", "Unknown"),
        ("speaker_id", "UNKNOWN"),
        ("speaker_id", " 001"),
        ("speaker_id", ""),
        ("dataset_id", ""),
        ("dataset_id", " set-a"),
        ("dataset_id", None),
    ],
)
def test_strict_identity_rejects_invalid_identity_fields(
    strict_tse_data, tmp_path, field, value
):
    record = {
        "uid": "a", "fs": SR, "length": 2 * SR, "path": "x.wav",
        "language": "zh", "dataset_id": "set-a", "speaker_id": "001",
    }
    record[field] = value
    speech_jsonl = tmp_path / "speech_bad.jsonl"
    _write_strict_speech_jsonl(speech_jsonl, [record])
    strict_tse_data["speech_scp"] = str(speech_jsonl)
    with pytest.raises(ValueError, match=field):
        _build_strict_dataset(strict_tse_data)


def test_strict_manifest_suffix_check_is_case_insensitive():
    from core.data.simulation_utils import is_strict_jsonl_manifest

    assert is_strict_jsonl_manifest("a/b.JSONL")
    assert is_strict_jsonl_manifest("a/b.jsonl")
    assert not is_strict_jsonl_manifest("a/b.json")
    assert not is_strict_jsonl_manifest("a/b.scp")


# ---------------------------------------------------------------------------
# Step 3: strict nested enrollment
# ---------------------------------------------------------------------------


def test_strict_nested_enrollment_keys_are_tuple_identities(strict_tse_data):
    dataset = _build_strict_dataset(strict_tse_data)
    assert all(isinstance(key, tuple) for key in dataset.spk2enroll)
    assert dataset.spk2enroll[("set-a", "001")] == tuple(
        dataset.spk2enroll[("set-a", "001")]
    )


def test_strict_flat_enrollment_map_is_rejected(strict_tse_data, tmp_path):
    flat = tmp_path / "flat_enroll.json"
    flat.write_text(json.dumps({"001": ["x.wav"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="flat speaker enrollment maps are not supported"):
        _build_strict_dataset(strict_tse_data, spk2enroll_json=str(flat))


def test_strict_enrollment_unknown_identity_is_coverage_error(strict_tse_data, tmp_path):
    bad = tmp_path / "bad_enroll.json"
    bad.write_text(
        json.dumps(
            {
                "set-a": {"001": ["x.wav"]},
                "set-b": {"001": ["y.wav"]},
                "set-c": {"009": ["z.wav"]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="coverage"):
        _build_strict_dataset(strict_tse_data, spk2enroll_json=str(bad))


def test_strict_enrollment_rejects_unknown_empty_dataset(
    strict_tse_data, tmp_path
):
    """未知 dataset 即使没有 speaker key 也不能绕过 coverage 校验。"""
    bad = tmp_path / "unknown_empty_dataset.json"
    bad.write_text(
        json.dumps(
            {
                "set-a": {"001": ["x.wav"]},
                "set-b": {"001": ["y.wav"]},
                "set-c": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="set-c.*coverage|coverage.*set-c"):
        _build_strict_dataset(strict_tse_data, spk2enroll_json=str(bad))


def test_strict_enrollment_rejects_empty_known_dataset(
    strict_tse_data, tmp_path
):
    """已知 dataset 的 speaker map 必须非空。"""
    bad = tmp_path / "empty_known_dataset.json"
    bad.write_text(
        json.dumps({"set-a": {}, "set-b": {"001": ["y.wav"]}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="set-a.*non-empty|non-empty.*set-a"):
        _build_strict_dataset(strict_tse_data, spk2enroll_json=str(bad))


@pytest.mark.parametrize(
    "paths",
    [[], [""], [" "], [" x.wav"], ["x.wav "], [123], "not-a-list"],
)
def test_strict_enrollment_rejects_invalid_path_lists(strict_tse_data, tmp_path, paths):
    bad = tmp_path / "bad_enroll.json"
    bad.write_text(
        json.dumps({"set-a": {"001": paths}, "set-b": {"001": ["y.wav"]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty.*trimmed strings|nested mapping"):
        _build_strict_dataset(strict_tse_data, spk2enroll_json=str(bad))


# ---------------------------------------------------------------------------
# Step 4: 非 strict speech manifest 必须 fail-closed
# ---------------------------------------------------------------------------


def test_scp_speech_manifest_is_rejected(strict_tse_data, tmp_path):
    speech_scp = tmp_path / "speech.scp"
    speech_scp.write_text("a 16000 32000 a.wav\n", encoding="utf-8")
    strict_tse_data["speech_scp"] = str(speech_scp)

    with pytest.raises(ValueError, match="strict JSONL"):
        _build_strict_dataset(strict_tse_data)


# ---------------------------------------------------------------------------
# Task 8: runtime canonical identity 与 primary-only 边界
# ---------------------------------------------------------------------------


def _multi_identity_strict_data(tmp_path: Path) -> dict:
    """5 条记录：("set-a","001")×2（a/a2）、("set-b","001")×2（b/c）、("set-a","002")×1。"""
    base = strict_tse_data.__wrapped__(tmp_path)
    speech_dir = tmp_path / "speech"
    extra = [
        ("a2", "set-a", "001", 310.0),
        ("c", "set-b", "001", 340.0),
        ("d", "set-a", "002", 540.0),
    ]
    records = []
    for uid, dataset_id, speaker_id, freq in extra:
        wav = speech_dir / f"{uid}.wav"
        _make_wav(wav, 2.0, SR, seed=len(uid) + int(freq), freq=freq)
        records.append(
            {"uid": uid, "fs": SR, "length": 2 * SR, "path": str(wav),
             "language": "zh", "dataset_id": dataset_id, "speaker_id": speaker_id}
        )
    enroll_dir = tmp_path / "enroll"
    enroll_c = enroll_dir / "enroll_c.wav"
    enroll_d = enroll_dir / "enroll_d.wav"
    _make_wav(enroll_c, 1.0, SR, seed=21, freq=340.0)
    _make_wav(enroll_d, 1.0, SR, seed=22, freq=540.0)
    with open(base["speech_scp"], "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    existing = json.loads(Path(base["spk2enroll_json"]).read_text())
    nested = {
        "set-a": {"001": existing["set-a"]["001"], "002": [str(enroll_d)]},
        "set-b": {"001": [str(enroll_c)]},
    }
    Path(base["spk2enroll_json"]).write_text(json.dumps(nested), encoding="utf-8")
    return base


def test_interference_excludes_canonical_identity_not_raw_label(tmp_path):
    """Step 1：interference 排除同 tuple identity；跨 dataset 同裸 speaker 可作候选。"""
    data = _multi_identity_strict_data(tmp_path)
    dataset = _build_strict_dataset(data)
    # target "a" 的 identity 是 ("set-a", "001")：同 identity 的 "a2" 必须被排除；
    # "b"/"c" 是跨 dataset 同裸 label 的不同 identity（允许作候选）
    target_identity = dataset.uid_to_speaker_identity["a"]
    assert target_identity == ("set-a", "001")
    assert dataset.uid_to_speaker_identity["a2"] == target_identity
    assert dataset.uid_to_speaker_identity["c"] == ("set-b", "001")

    rng = np.random.default_rng(0)
    # b/c 同属 set-b/001；只能采到两个不同身份，reduce 策略如实降低人数。
    with pytest.raises(RuntimeError, match="distinct speakers"):
        dataset._sample_interference_uids("a", SR, 3, rng)
    chosen = dataset._sample_interference_uids("a", SR, 2, rng)
    assert len(chosen) == 2 and "d" in chosen
    assert len(set(chosen) & {"b", "c"}) == 1
    chosen_identities = {dataset.uid_to_speaker_identity[c] for c in chosen}
    assert target_identity not in chosen_identities
    assert ("set-b", "001") in chosen_identities  # 跨 dataset 同裸 label 允许


def test_enrollment_and_spk_label_use_canonical_identity(strict_tse_data):
    """Step 2：跨 dataset 同裸 speaker 得到不同 label 与不同 enrollment pool。"""
    dataset = _build_strict_dataset(strict_tse_data)
    idx_a = dataset.speaker_identity_to_idx[("set-a", "001")]
    idx_b = dataset.speaker_identity_to_idx[("set-b", "001")]
    assert idx_a != idx_b
    assert dataset.spk2enroll[("set-a", "001")] != dataset.spk2enroll[("set-b", "001")]
    assert dataset._spk_label("a") == idx_a
    assert dataset._spk_label("b") == idx_b

    # 真实样本路径：spk_label 与 primary UID 的 identity 一致
    dataset.set_epoch(0, base_seed=7)
    sample = dataset[0]
    fs, i = dataset._get_from_index(0)
    uid = dataset.speech_uids[fs][i]
    assert sample["targets"][0]["speaker_id"] == uid
    assert sample["targets"][0]["spk_label"] == dataset.speaker_identity_to_idx[
        dataset.uid_to_speaker_identity[uid]
    ]


def test_speaker_controller_leaves_carry_canonical_identity(strict_tse_data):
    """Step 7：启用 speaker controller 时 leaf key 含 canonical identity。"""
    sampling_cfg = SamplingConfig.from_dict(
        {"speech": {"speaker": {"max_samples_per_epoch": 10}}}
    )
    dataset = _build_strict_dataset(strict_tse_data, sampling_cfg=sampling_cfg)
    assert set(dataset.speech_sampling_leaves) == {
        ("zh", "set-a", ("set-a", "001"), SR),
        ("zh", "set-b", ("set-b", "001"), SR),
    }


def test_planner_receives_only_primary_anchor_leaves(strict_tse_data, monkeypatch):
    """Step 3：spy 捕获 planner 输入——只有 primary external indices；
    speaker_quotas 总和 == actual（cap 只计 primary anchor）。"""
    from core.data import dynamic_dataset_base

    captured = {}
    real_make_plan = dynamic_dataset_base.make_speech_plan

    def spy(leaves, **kwargs):
        captured["uid_indices"] = sorted(
            index for leaf in leaves for index in leaf.uid_indices
        )
        return real_make_plan(leaves, **kwargs)

    monkeypatch.setattr(dynamic_dataset_base, "make_speech_plan", spy)
    sampling_cfg = SamplingConfig.from_dict(
        {"speech": {"speaker": {"max_samples_per_epoch": 10}}}
    )
    dataset = _build_strict_dataset(strict_tse_data, sampling_cfg=sampling_cfg)
    plan = dataset.get_speech_epoch_plan(
        requested=2, world_size=1, batch_size=1, drop_last=False, seed=1, epoch=0
    )
    assert captured["uid_indices"] == list(range(len(dataset._filtered_indices)))
    assert sum(plan.speech_plan.speaker_quotas.values()) == plan.speech_plan.actual
    # Step 4：summary 的 primary samples 只统计 selected primary anchors
    speaker_summary = plan.speech_plan.summary["joint_solver"]["speaker"]
    assert speaker_summary["primary_samples"] == plan.speech_plan.actual


def test_multi_target_expansion_does_not_change_primary_plan(strict_tse_data):
    """Step 3：targets_per_mixture=2 不改变 primary plan（quota 与选择逐位一致）。"""
    sampling_cfg = SamplingConfig.from_dict(
        {"speech": {"speaker": {"max_samples_per_epoch": 10}}}
    )
    ds1 = _build_strict_dataset(strict_tse_data, sampling_cfg=sampling_cfg)
    ds2 = _build_strict_dataset(
        strict_tse_data, sampling_cfg=sampling_cfg, targets_per_mixture=2
    )
    kwargs = dict(requested=2, world_size=1, batch_size=1, drop_last=False, seed=1, epoch=0)
    p1 = ds1.get_speech_epoch_plan(**kwargs)
    p2 = ds2.get_speech_epoch_plan(**kwargs)
    assert p1.speech_plan.leaf_quotas == p2.speech_plan.leaf_quotas
    assert p1.speech_plan.actual == p2.speech_plan.actual
    assert p1.selected_indices == p2.selected_indices


def test_runtime_secondary_choices_independent_of_sampling_toggle(strict_tse_data):
    """Step 4：同 seed/epoch 与同 primary external index 下，sampling 开关不改变
    runtime 的 interference / enrollment 选择（planner 不参与 secondary 决策）。"""
    overrides = {'sir_range': [-5.0, 5.0], 'prob_target_only': 0.0, 'prob_interference_only': 0.0, 'prob_noise_only': 0.0, 'prob_interference_noise_only': 0.0, 'prob_target_noise': 0.0, 'prob_target_interference': 0.0, 'prob_target_interference_noise': 1.0}
    sampling_cfg = SamplingConfig.from_dict(
        {"speech": {"speaker": {"max_samples_per_epoch": 10}}}
    )
    ds_off = _build_strict_dataset(strict_tse_data, sim_overrides=overrides)
    ds_on = _build_strict_dataset(
        strict_tse_data, sim_overrides=overrides, sampling_cfg=sampling_cfg
    )
    for ds in (ds_off, ds_on):
        ds.set_epoch(0, base_seed=42)
    sample_off = ds_off[0]
    sample_on = ds_on[0]
    np.testing.assert_array_equal(sample_off["mixture"], sample_on["mixture"])
    assert sample_off["sr"] == sample_on["sr"]
    assert sample_off["length"] == sample_on["length"]
    assert len(sample_off["targets"]) == len(sample_on["targets"])
    for off, on in zip(sample_off["targets"], sample_on["targets"]):
        np.testing.assert_array_equal(off["target"], on["target"])
        off_cue = off["cues"].speaker_waveform
        on_cue = on["cues"].speaker_waveform
        assert off_cue is not None and on_cue is not None
        np.testing.assert_array_equal(off_cue.value, on_cue.value)
        assert off["speaker_id"] == on["speaker_id"]
        assert off["spk_label"] == on["spk_label"]
        assert off["target_present"] is on["target_present"]


def test_tse_splice_candidates_use_canonical_identity_manifest_order(tmp_path):
    """同裸 speaker 跨 dataset 隔离，identity 内 UID 保持 manifest 顺序。"""
    dataset = _build_strict_dataset(_multi_identity_strict_data(tmp_path))

    assert dataset.speaker_identity_to_uids == {
        ("set-a", "001"): ("a", "a2"),
        ("set-b", "001"): ("b", "c"),
        ("set-a", "002"): ("d",),
    }
    assert dataset._speech_splice_candidate_uids("a") == ("a", "a2")
    assert dataset._speech_splice_candidate_uids("b") == ("b", "c")
    assert dataset._speech_splice_candidate_uids("d") == ("d",)




def test_tse_missing_identity_and_enrollment_do_not_escape_splice_boundary(
    strict_tse_data, monkeypatch
):
    dataset = _build_strict_dataset(
        strict_tse_data,
        sim_overrides={
            "speech_splice": True,
            "speech_splice_seconds": 3.0,
            "speech_splice_gap_range": [0.01, 0.01],
            "speech_splice_equal_volume": False,
            "speech_splice_max_repeat_times": 1,
            "speech_random_start": False,
        },
        max_duration_seconds=3.0,
    )
    dataset.uid_to_speaker_identity.pop("a")
    assert dataset._speech_splice_candidate_uids("a") == ("a",)
    anchor_path = dataset.speech_flt["a"]
    original_read = dataset._read_one_speech
    reads = []

    def _record_read(path, sr):
        if path == anchor_path:
            reads.append(path)
        return original_read(path, sr)

    monkeypatch.setattr(dataset, "_read_one_speech", _record_read)
    dataset._get_clean_speech("a", SR, rng=np.random.default_rng(3))
    assert reads == [anchor_path, anchor_path]

    monkeypatch.setattr(
        dataset,
        "_speech_splice_candidate_uids",
        lambda _uid: pytest.fail("enrollment 不应进入 speech splice"),
    )
    enrollment = dataset._build_enrollment("b", SR, np.random.default_rng(3))
    assert enrollment.shape == (int(dataset.enrollment_duration * SR),)


def test_se_aec_default_splice_pool_and_rng_order_are_unchanged():
    """默认 seam 仍按 manifest-order 全池抽样，路径选择与后续 RNG 状态不变。"""
    from types import SimpleNamespace

    from tasks.aec.data.dynamic_dataset import DynamicMixingDataset as AECDataset
    from tasks.se.data.dynamic_dataset import DynamicMixingDataset as SEDataset
    from core.data.simulation.speech_preparation import SpeechPreparation

    paths = {"u0": "p0", "u1": "p1", "u2": "p2"}
    waves = {
        "p0": np.full(5, 1.0, dtype=np.float32),
        "p1": np.full(5, 2.0, dtype=np.float32),
        "p2": np.full(5, 3.0, dtype=np.float32),
    }
    cfg = SimpleNamespace(
        speech_splice=True,
        speech_splice_seconds=2.0,
        speech_splice_seconds_range=None,
        speech_splice_equal_volume=False,
        speech_splice_gap_range=[0.1, 0.1],
        speech_splice_max_repeat_times=2,
        speech_random_start=False,
    )

    for dataset_cls in (SEDataset, AECDataset):
        dataset = object.__new__(dataset_cls)
        dataset.speech_preparation = SpeechPreparation()
        dataset.sim_cfg = cfg
        dataset.speech_flt = paths
        dataset._all_speech_paths = list(paths.values())
        dataset._read_one_speech = lambda path, _sr: waves[path].copy()
        assert dataset._speech_splice_candidate_uids("u0") is None

        actual_rng = np.random.default_rng(17)
        actual = dataset._get_clean_speech("u0", 10, rng=actual_rng)

        expected_rng = np.random.default_rng(17)
        expected = waves["p0"].copy()
        log_lo = np.log(0.1)
        log_hi = np.log(0.100001)
        for _ in range(2):
            gap_len = int(np.exp(expected_rng.uniform(log_lo, log_hi)) * 10)
            path = list(paths.values())[
                int(expected_rng.integers(0, len(paths)))
            ]
            expected = np.concatenate(
                (expected, np.zeros(gap_len, dtype=np.float32), waves[path])
            )
        expected = np.concatenate(
            (expected, np.zeros(20 - len(expected), dtype=np.float32))
        )

        np.testing.assert_array_equal(actual, expected)
        assert actual_rng.random() == expected_rng.random()


# ---------------------------------------------------------------------------
# Task 9: validation label 空间复用
# ---------------------------------------------------------------------------


def test_set_speaker_label_mapping_reuses_dense_train_space(strict_tse_data):
    """validation 复用连续 train label 空间；未知 identity 回退 -1。"""
    dataset = _build_strict_dataset(strict_tse_data)
    train_map = {("set-a", "001"): 0, ("set-b", "001"): 1}
    dataset.set_speaker_label_mapping(train_map)
    assert dataset._spk_label("a") == 0
    assert dataset._spk_label("b") == 1

    # 映射中不存在的 identity → -1（不动态扩展 train map）
    dataset.set_speaker_label_mapping({("set-a", "001"): 0})
    assert dataset._spk_label("a") == 0
    assert dataset._spk_label("b") == -1


def test_set_speaker_label_mapping_rejects_sparse_label_space(strict_tse_data):
    dataset = _build_strict_dataset(strict_tse_data)

    with pytest.raises(ValueError, match="contiguous.*0|0.*contiguous"):
        dataset.set_speaker_label_mapping(
            {("set-a", "001"): 5, ("set-b", "001"): 7}
        )


@pytest.mark.parametrize(
    "mapping",
    [
        {"s1": 0},                          # 非 tuple key
        {("set-a",): 0},                    # 长度错误的 tuple
        {("set-a", "s1"): -1},              # 负值
        {("set-a", "s1"): 1.5},             # 非 int
        {("set-a", "s1"): 0, ("set-b", "s2"): 0},  # 重复 value
    ],
)
def test_set_speaker_label_mapping_validates_shape(strict_tse_data, mapping):
    dataset = _build_strict_dataset(strict_tse_data)
    with pytest.raises(ValueError, match="speaker label mapping"):
        dataset.set_speaker_label_mapping(mapping)


pytestmark = [pytest.mark.unit]
