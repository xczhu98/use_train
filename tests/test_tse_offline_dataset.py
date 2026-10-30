"""Tests for tasks.tse.data.offline_dataset (offline_tse mode, 官方 wesep 格式)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from core.data.offline_multitrack import frame_window_between_domains
from tasks.tse.data import build_dataset
from tasks.tse.data.batch import TSEBatch
from tasks.tse.data.collate import tse_collate_fn
from tasks.tse.data.dynamic_dataset import EnrollmentAugmentationProviderError
from tasks.tse.data.offline_dataset import (
    TSEOfflineDataset,
    build_offline_tse_dataset,
)
from core.trainer.entry import detect_task


def _write_wav(path: Path, n: int, fs: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, (np.random.randn(n) * 0.1).astype(np.float32), fs)


def _make_wesep_manifest(root: Path):
    """构造官方 wesep schema 数据：samples.jsonl + cues.yaml + cues/audio.json。

    3 个 sample（a/b/c），2 个说话人（spk0 出现在 a/b，spk1 在 c）；
    目标说话人 = samples.spk[0]；target 故意比 mixture 短（验证取公共长度）：
      - a: mixture 16000, target 15900
      - b: mixture 24000, target 24000
      - c: mixture 16000, target 16000
    enrollment 资源：spk0 两条、spk1 一条。
    """
    mix_n = {"a": 16000, "b": 24000, "c": 16000}
    tgt_n = {"a": 15900, "b": 24000, "c": 16000}
    spk = {"a": "spk0", "b": "spk0", "c": "spk1"}
    mix_p, tgt_p = {}, {}
    for u in spk:
        mix_p[u] = str(root / f"{u}_mix.wav")
        tgt_p[u] = str(root / f"{u}_tgt.wav")
        _write_wav(Path(mix_p[u]), mix_n[u])
        _write_wav(Path(tgt_p[u]), tgt_n[u])
    enroll = {}
    for s in ("spk0", "spk1"):
        n = 2 if s == "spk0" else 1
        ps = [root / f"enr_{s}_{i}.wav" for i in range(n)]
        for p in ps:
            _write_wav(p, 8000)
        enroll[s] = [{"utt_id": f"{s}_{i}", "path": str(p)} for i, p in enumerate(ps)]
    cues_dir = root / "cues"
    cues_dir.mkdir(parents=True, exist_ok=True)
    (cues_dir / "audio.json").write_text(json.dumps(enroll))
    cues_yaml = root / "cues.yaml"
    cues_yaml.write_text(
        "cues:\n"
        "  audio:\n"
        "    type: raw\n"
        "    guaranteed: true\n"
        "    scope: speaker\n"
        "    policy:\n"
        "      type: random\n"
        "      key: spk_id\n"
        f"      resource: {(cues_dir / 'audio.json').resolve()}\n"
    )
    samples = root / "samples.jsonl"
    with open(samples, "w") as f:
        for u in spk:
            f.write(json.dumps({
                "key": u,
                "spk": [spk[u]],
                "mix": {"default": [mix_p[u]]},
                "src": {spk[u]: [tgt_p[u]]},
            }) + "\n")
    raw_list = root / "raw.list"
    raw_list.write_text(str(samples.resolve()) + "\n")
    return str(raw_list), str(cues_yaml), str(samples), spk


def _dataset(**kwargs) -> TSEOfflineDataset:
    kwargs.setdefault(
        "manifest_format", "path_list" if kwargs.get("samples_list") is not None else "jsonl"
    )
    kwargs.setdefault("path_base", "manifest_relative")
    kwargs.setdefault("target_selection", "all")
    kwargs.setdefault("mix_variant", "default")
    kwargs.setdefault("cue_schema", "realtse_raw")
    kwargs.setdefault("enabled_cues", ("audio",))
    return TSEOfflineDataset(**kwargs)


def test_wesep_contract_and_collate(tmp_path):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    ds = _dataset(samples_list=raw_list, cues_yaml=cues_yaml, target_sr=16000)
    assert len(ds) == 3
    assert ds.speaker_identity_to_idx == {"spk0": 0, "spk1": 1}
    item = ds[0]
    assert set(item) == {
        "key", "mixture", "sr", "length", "targets", "geometry", "timebase", "provenance"
    }
    assert item["targets"][0]["speaker_id"] == "spk0"
    assert ds[2]["targets"][0]["spk_label"] == 1
    assert item["mixture"].shape == (1, 15900)
    assert item["targets"][0]["target"].shape == (15900,)
    cue = item["targets"][0]["cues"].speaker_waveform
    assert cue is not None and cue.present and cue.length == 8000
    assert item["sr"] == 16000 and item["length"] == 15900
    assert ds.get_srs() == [16000, 16000, 16000]

    batch = tse_collate_fn([ds[0], ds[1]], cue_specs=ds.cue_specs)
    assert isinstance(batch, TSEBatch)
    assert batch.mixture.shape[0] == 2
    assert batch.spk_labels.tolist() == [0, 0]
    assert batch.num_targets.tolist() == [1, 1]


def test_enroll_policy_from_cues_and_fixed(tmp_path):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    ds = _dataset(samples_list=raw_list, cues_yaml=cues_yaml, target_sr=16000)
    assert ds.enroll_policy == "random"
    ds_fixed = _dataset(
        samples_list=raw_list,
        cues_yaml=cues_yaml,
        target_sr=16000,
        enroll_policy="fixed",
    )
    first = ds_fixed[0]["targets"][0]["cues"].speaker_waveform
    replay = ds_fixed[0]["targets"][0]["cues"].speaker_waveform
    assert first is not None and replay is not None
    assert torch.equal(torch.as_tensor(first.value), torch.as_tensor(replay.value))
    assert first.length == 8000

    with pytest.raises(ValueError, match="enroll_policy"):
        _dataset(
            samples_list=raw_list,
            cues_yaml=cues_yaml,
            target_sr=16000,
            enroll_policy="bogus",
        )


def test_missing_cues_or_samples_raises(tmp_path):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    with pytest.raises(ValueError, match="requires cues_yaml"):
        _dataset(samples_list=raw_list, target_sr=16000)
    with pytest.raises(ValueError, match="exactly one"):
        _dataset(cues_yaml=cues_yaml, target_sr=16000)
    bad = tmp_path / "cues" / "audio.json"
    resource = json.loads(bad.read_text())
    del resource["spk1"]
    bad.write_text(json.dumps(resource))
    with pytest.raises(KeyError, match="guaranteed legacy audio cue is missing"):
        _dataset(samples_list=raw_list, cues_yaml=cues_yaml, target_sr=16000)


def test_mix_missing_default_refuses_non_dict_src(tmp_path):
    root = tmp_path
    _write_wav(root / "m.wav", 8000)
    _write_wav(root / "t.wav", 8000)
    cues_yaml = _make_wesep_manifest(tmp_path / "cuesbase")[1]
    bad_json = root / "bad.jsonl"
    # src 缺失
    bad_json.write_text(json.dumps({"key": "k", "spk": ["s"],
                                    "mix": {"default": [str(root / "m.wav")]}}) + "\n")
    with pytest.raises(ValueError, match="src must be a mapping"):
        _dataset(samples_jsonl=bad_json, cues_yaml=cues_yaml, target_sr=16000)
    bad_json.write_text(json.dumps({"key": "k", "spk": ["s"], "src": {"s": "t"},
                                    "mix": {"default": "x"}}) + "\n")
    with pytest.raises(ValueError, match="non-empty path list"):
        _dataset(samples_jsonl=bad_json, cues_yaml=cues_yaml, target_sr=16000)


def test_duration_filter(tmp_path):
    root = tmp_path / "dur"
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(root)
    # 追加一个短样本 short（mix 0.25s < 1s 过滤）
    _write_wav(root / "short_mix.wav", 4000)
    _write_wav(root / "short_tgt.wav", 4000)
    with open(root / "samples.jsonl", "a") as f:
        f.write(json.dumps({"key": "d", "spk": ["spk1"],
                            "mix": {"default": [str(root / "short_mix.wav")]},
                            "src": {"spk1": [str(root / "short_tgt.wav")]}}) + "\n")
    ds = _dataset(samples_list=raw_list, cues_yaml=cues_yaml,
                           filter_by_duration=True,
                           filter_min_duration=1.0, filter_max_duration=5.0,
                           target_sr=16000)
    assert len(ds) == 3  # 4 个 sample 中 "d"(0.25s) 被过滤
    kept = [ds.uids[i] for i in ds._filtered_indices]
    assert "d" not in kept


def test_max_duration_seconds_crops_aligned(tmp_path):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    ds = _dataset(
        samples_list=raw_list,
        cues_yaml=cues_yaml,
        max_duration_seconds=1.0,
        target_sr=16000,
    )
    item = ds[1]
    assert item["mixture"].shape == (1, 16000)
    assert item["targets"][0]["target"].shape == (16000,)
    assert item["length"] == 16000
    assert ds.get_source_length()[1] == 16000


def test_fast_path_window_equiv_full_for_uniform_sr(tmp_path):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    ds = _dataset(
        samples_list=raw_list,
        cues_yaml=cues_yaml,
        max_duration_seconds=1.0,
        target_sr=None,
    )
    record = ds.records[1]
    win = ds._window_plan(record, ds._local_rng(1))
    fast = ds._getitem_fast(1, win, fs=ds.utt2fs[record.key])
    slow = ds._getitem_impl_full(1, win)
    assert torch.equal(fast["mixture"], slow["mixture"])
    assert torch.equal(
        fast["targets"][0]["target"], slow["targets"][0]["target"]
    )
    fast_cue = fast["targets"][0]["cues"].speaker_waveform
    slow_cue = slow["targets"][0]["cues"].speaker_waveform
    assert fast_cue is not None and slow_cue is not None
    assert torch.equal(torch.as_tensor(fast_cue.value), torch.as_tensor(slow_cue.value))
    assert fast["length"] == slow["length"] == 16000


def test_local_rng_deterministic_and_epoch(tmp_path):
    """样本内 RNG 与 (seed, epoch, index) 绑定：同项可复现，换 epoch 变化。"""
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    ds = _dataset(samples_list=raw_list, cues_yaml=cues_yaml, seed=3)
    r0 = ds._local_rng(0)
    r0b = ds._local_rng(0)          # 同 (seed, epoch, index) → 同随机序列
    assert r0.randint(0, 10**9) == r0b.randint(0, 10**9)
    ds.set_epoch(1)
    r1 = ds._local_rng(0)
    # 同 index、不同 epoch → 序列不同
    assert r0.randint(0, 10**9) != r1.randint(0, 10**9)
    # seed 不同 → 序列不同
    ds2 = _dataset(samples_list=raw_list, cues_yaml=cues_yaml, seed=9)
    r2 = ds2._local_rng(0)
    ds.set_epoch(0)
    assert ds._local_rng(0).randint(0, 10**9) != r2.randint(0, 10**9)


def test_set_epoch_base_seed_is_idempotent(tmp_path):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    ds = _dataset(samples_list=raw_list, cues_yaml=cues_yaml, seed=3)

    ds.set_epoch(4, base_seed=11)
    first = ds._local_rng(1).randint(0, 10**9)
    ds.set_epoch(4, base_seed=11)
    second = ds._local_rng(1).randint(0, 10**9)

    assert first == second
    assert ds._rng_seed == 3


def test_fast_path_applies_finite_rules_to_canonical_item(tmp_path, monkeypatch):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    ds = _dataset(samples_list=raw_list, cues_yaml=cues_yaml, strict_load=False)
    record = ds.records[0]
    win = ds._window_plan(record, ds._local_rng(0))

    def mono_nonfinite(_path, _start, frames):
        waveform = torch.ones(1, frames)
        waveform[0, 0] = float("nan")
        return waveform, ds.utt2fs[record.key]

    monkeypatch.setattr(
        "tasks.tse.data.offline_dataset.load_audio_window", mono_nonfinite
    )
    item = ds._getitem_fast(0, win, fs=ds.utt2fs[record.key])
    cue = item["targets"][0]["cues"].speaker_waveform
    assert cue is not None
    assert item["mixture"].isfinite().all()
    assert item["targets"][0]["target"].isfinite().all()
    assert torch.as_tensor(cue.value).isfinite().all()


def test_slow_path_converts_source_window_to_output_domain(tmp_path):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    for name in ("b_mix.wav", "b_tgt.wav", "enr_spk0_0.wav", "enr_spk0_1.wav"):
        _write_wav(tmp_path / name, 72000 if name.startswith("b_") else 24000, fs=48000)
    ds = _dataset(
        samples_list=raw_list,
        cues_yaml=cues_yaml,
        max_duration_seconds=0.5,
        max_enroll_duration_seconds=0.25,
        target_sr=16000,
    )
    record = ds.records[1]
    window = ds._window_plan(record, ds._local_rng(1))
    output_start, _ = frame_window_between_domains(
        start=window["mix_start"],
        length=window["mix_len"],
        source_sr=48000,
        output_sr=16000,
    )

    item = ds[1]
    cue = item["targets"][0]["cues"].speaker_waveform
    assert cue is not None
    assert item["sr"] == 16000
    assert item["mixture"].shape == (1, 8000)
    assert item["targets"][0]["target"].shape == (8000,)
    assert item["length"] == 8000
    assert torch.as_tensor(cue.value).shape == (4000,)
    assert cue.length == 4000
    assert item["provenance"]["source_start_sample"] == window["mix_start"]
    assert item["provenance"]["source_sample_rate"] == 48000
    assert item["provenance"]["output_start_sample"] == output_start
    assert ds.get_source_length()[1] == 8000


def test_tse_mixed_sr_contract_is_strict_or_resamples_to_mixture(tmp_path):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    _write_wav(tmp_path / "a_tgt.wav", 7950, fs=8000)

    strict = _dataset(samples_list=raw_list, cues_yaml=cues_yaml, strict_load=True)
    with pytest.raises(RuntimeError, match="sample rate mismatch"):
        strict[0]

    lenient = _dataset(samples_list=raw_list, cues_yaml=cues_yaml, strict_load=False)
    item = lenient[0]
    assert item["sr"] == 16000
    assert item["mixture"].shape == (1, 15900)
    assert item["targets"][0]["target"].shape == (15900,)
    assert item["length"] == 15900


def test_source_length_is_cached(tmp_path, monkeypatch):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    ds = _dataset(samples_list=raw_list, cues_yaml=cues_yaml)
    expected = ds.get_source_length()
    monkeypatch.setattr(
        ds,
        "_source_length",
        lambda _index: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    assert ds.get_source_length() == expected


def test_build_offline_tse_dataset_passes_seed(tmp_path):
    """build_offline_tse_dataset 从 meta.seed 透传 seed。"""
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    cfg = {
        "data": {
            "mode": "offline_tse",
            "sample_rate": 16000,
            "train": {
                "samples_list": raw_list,
                "cues_yaml": cues_yaml,
                "manifest_format": "path_list",
                "path_base": "manifest_relative",
                "target_selection": "all",
                "mix_variant": "default",
                "cue_schema": "realtse_raw",
                "enabled_cues": ["audio"],
            },
        },
        "meta": {"seed": 42},
    }
    ds = build_offline_tse_dataset(cfg, "train")
    assert ds._rng_seed == 42
    assert ds._epoch == 0
    # 缺省 meta.seed → 0
    ds0 = build_offline_tse_dataset({"data": cfg["data"]}, "train")
    assert ds0._rng_seed == 0


def test_enrollment_augment_opt_in_preserves_window_plan_and_dict_contract(tmp_path):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    common = dict(samples_list=raw_list, cues_yaml=cues_yaml, seed=17)
    baseline = _dataset(**common)
    disabled = _dataset(**common, enrollment_augment={"enabled": False})
    augmented = _dataset(
        **common,
        enrollment_augment={"enabled": True, "prob_eq": 1.0, "eq_db_range": [-9, -9]},
    )

    baseline_plan = baseline._window_plan(baseline.records[1], baseline._local_rng(1))
    augmented_plan = augmented._window_plan(augmented.records[1], augmented._local_rng(1))
    assert augmented_plan == baseline_plan
    expected = baseline[1]
    unchanged = disabled[1]
    actual = augmented[1]
    assert torch.equal(actual["mixture"], expected["mixture"])
    assert torch.equal(actual["targets"][0]["target"], expected["targets"][0]["target"])
    assert torch.equal(unchanged["mixture"], expected["mixture"])
    expected_cue = expected["targets"][0]["cues"].speaker_waveform
    unchanged_cue = unchanged["targets"][0]["cues"].speaker_waveform
    actual_cue = actual["targets"][0]["cues"].speaker_waveform
    assert expected_cue is not None and unchanged_cue is not None and actual_cue is not None
    assert torch.equal(torch.as_tensor(unchanged_cue.value), torch.as_tensor(expected_cue.value))
    assert torch.as_tensor(actual_cue.value).shape == torch.as_tensor(expected_cue.value).shape
    assert not torch.equal(torch.as_tensor(actual_cue.value), torch.as_tensor(expected_cue.value))


def test_enrollment_augment_rng_replays_and_changes_across_epochs(tmp_path):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    ds = _dataset(
        samples_list=raw_list,
        cues_yaml=cues_yaml,
        seed=17,
        enrollment_augment={"enabled": True, "prob_eq": 1.0, "eq_db_range": [-9, -9]},
    )

    def cue_value():
        cue = ds[0]["targets"][0]["cues"].speaker_waveform
        assert cue is not None
        return torch.as_tensor(cue.value)

    ds.set_epoch(0, base_seed=3)
    first = cue_value()
    ds.set_epoch(1, base_seed=3)
    changed = cue_value()
    ds.set_epoch(0, base_seed=3)
    replay = cue_value()
    assert torch.equal(first, replay)
    assert not torch.equal(first, changed)


def test_enrollment_augment_rir_noise_require_configured_resources(tmp_path):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    common = dict(samples_list=raw_list, cues_yaml=cues_yaml)
    with pytest.raises(ValueError, match="prob_reverb > 0 requires non-empty rir_scp"):
        _dataset(**common, enrollment_augment={"enabled": True, "prob_reverb": 1.0})
    with pytest.raises(ValueError, match="prob_noise > 0 requires non-empty noise_scp"):
        _dataset(**common, enrollment_augment={"enabled": True, "prob_noise": 1.0})


def test_enrollment_augment_rir_noise_only_changes_enrollment(tmp_path):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    rir = tmp_path / "rir.wav"
    noise = tmp_path / "noise.wav"
    _write_wav(rir, 256)
    _write_wav(noise, 16000)
    rir_scp = tmp_path / "rir.scp"
    noise_scp = tmp_path / "noise.scp"
    rir_scp.write_text(f"rir 16000 256 {rir}\n", encoding="utf-8")
    noise_scp.write_text(f"noise 16000 16000 {noise}\n", encoding="utf-8")
    common = dict(samples_list=raw_list, cues_yaml=cues_yaml, seed=9)
    baseline = _dataset(**common)
    augmented = _dataset(
        **common,
        enrollment_augment={
            "enabled": True,
            "prob_eq": 0.0,
            "prob_reverb": 1.0,
            "rir_scp": str(rir_scp),
            "prob_noise": 1.0,
            "noise_scp": str(noise_scp),
            "snr_range": [10.0, 10.0],
            "prob_eq_post": 1.0,
            "eq_db_range": [-6.0, -6.0],
        },
    )
    expected, actual = baseline[0], augmented[0]
    assert torch.equal(actual["mixture"], expected["mixture"])
    assert torch.equal(actual["targets"][0]["target"], expected["targets"][0]["target"])
    expected_cue = expected["targets"][0]["cues"].speaker_waveform
    actual_cue = actual["targets"][0]["cues"].speaker_waveform
    assert expected_cue is not None and actual_cue is not None
    expected_value = torch.as_tensor(expected_cue.value)
    actual_value = torch.as_tensor(actual_cue.value)
    assert actual_value.shape == expected_value.shape
    assert not torch.equal(actual_value, expected_value)
    assert torch.isfinite(actual_value).all()


def test_enrollment_augment_provider_paths_resolve_data_root(tmp_path, monkeypatch):
    """嵌套 provider SCP 的 ``data/...`` 路径锚定仓库根。"""
    import core.data.resource_paths as resource_paths

    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path / "manifest")
    repo_root = tmp_path / "repo"
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True)
    rir = data_dir / "rir.wav"
    noise = data_dir / "noise.wav"
    _write_wav(rir, 256)
    _write_wav(noise, 16000)
    rir_scp = data_dir / "rir.scp"
    noise_scp = data_dir / "noise.scp"
    rir_scp.write_text(f"rir 16000 256 {rir}\n", encoding="utf-8")
    noise_scp.write_text(f"noise 16000 16000 {noise}\n", encoding="utf-8")
    monkeypatch.setattr(resource_paths, "repository_root", lambda: repo_root)

    dataset = _dataset(
        samples_list=raw_list,
        cues_yaml=cues_yaml,
        enrollment_augment={
            "enabled": True,
            "prob_reverb": 1.0,
            "rir_scp": "data/rir.scp",
            "prob_noise": 1.0,
            "noise_scp": "data/noise.scp",
            "snr_range": [10.0, 10.0],
        },
    )

    assert dataset._enrollment_augment_rirs == (str(rir),)
    assert dataset._enrollment_augment_noises == (str(noise),)


def test_enrollment_augment_rir_provider_error_is_not_placeholder_when_lenient(
    tmp_path,
):
    """离线 provider 读取失败时，即使 lenient 也必须 fail-closed。"""
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path / "manifest")
    missing_rir = tmp_path / "missing_rir.wav"
    rir_scp = tmp_path / "rir.scp"
    rir_scp.write_text(
        f"rir 16000 256 {missing_rir}\n", encoding="utf-8"
    )
    dataset = _dataset(
        samples_list=raw_list,
        cues_yaml=cues_yaml,
        strict_load=False,
        enrollment_augment={
            "enabled": True,
            "prob_reverb": 1.0,
            "rir_scp": str(rir_scp),
        },
    )

    with pytest.raises(
        EnrollmentAugmentationProviderError,
        match="failed to read enrollment RIR provider",
    ):
        dataset[0]


def test_enrollment_augment_noise_provider_error_is_not_placeholder_when_lenient(
    tmp_path,
):
    """离线 noise provider 读取失败时，即使 lenient 也必须 fail-closed。"""
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path / "manifest")
    missing_noise = tmp_path / "missing_noise.wav"
    noise_scp = tmp_path / "noise.scp"
    noise_scp.write_text(
        f"noise 16000 16000 {missing_noise}\n", encoding="utf-8"
    )
    dataset = _dataset(
        samples_list=raw_list,
        cues_yaml=cues_yaml,
        strict_load=False,
        enrollment_augment={
            "enabled": True,
            "prob_noise": 1.0,
            "noise_scp": str(noise_scp),
        },
    )

    with pytest.raises(
        EnrollmentAugmentationProviderError,
        match="failed to read enrollment noise provider",
    ):
        dataset[0]


def test_enrollment_augment_moving_rir_is_rejected_at_construction(tmp_path):
    """离线 enrollment augmentor 当前仅支持静态 RIR，构造期拒绝 moving RIR。"""
    import pickle

    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path / "manifest")
    moving_rir = tmp_path / "moving_rir.pkl"
    with moving_rir.open("wb") as handle:
        pickle.dump(
            {"moving": True, "rir": np.ones((2, 8), dtype=np.float32)}, handle
        )
    rir_scp = tmp_path / "moving_rir.scp"
    rir_scp.write_text(
        f"moving 16000 8 {moving_rir}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="static RIR only.*moving RIR"):
        _dataset(
            samples_list=raw_list,
            cues_yaml=cues_yaml,
            enrollment_augment={
                "enabled": True,
                "prob_reverb": 1.0,
                "rir_scp": str(rir_scp),
            },
        )


def test_metadata_failure_is_not_papered_over_by_fake_lengths(tmp_path):
    """target 元数据不可读时必须 fail-fast，不得伪造长度继续读取窗口。"""
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path / "manifest")
    (tmp_path / "manifest" / "a_tgt.wav").write_bytes(b"not-audio")
    with pytest.raises(RuntimeError, match="cannot read audio metadata"):
        _dataset(samples_list=raw_list, cues_yaml=cues_yaml)


def test_set_speaker_label_mapping_enforces_dense_string_space(tmp_path):
    """offline label 映射与 dynamic 同样要求 dense label 空间（key 为 speaker id 字符串）。"""
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path / "manifest")
    ds = _dataset(samples_list=raw_list, cues_yaml=cues_yaml)
    with pytest.raises(ValueError, match="contiguous"):
        ds.set_speaker_label_mapping({"spk0": 0, "spk1": 3})
    with pytest.raises(ValueError, match="unique"):
        ds.set_speaker_label_mapping({"spk0": 0, "spk1": 0})
    with pytest.raises(ValueError, match="non-empty speaker id"):
        ds.set_speaker_label_mapping({("spk0", "x"): 0})
    with pytest.raises(ValueError, match="non-negative ints"):
        ds.set_speaker_label_mapping({"spk0": -1})
    ds.set_speaker_label_mapping({"spk0": 0, "spk1": 1})
    assert ds.speaker_identity_to_idx == {"spk0": 0, "spk1": 1}


def test_unexpected_kwargs_rejected(tmp_path):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path)
    with pytest.raises(TypeError, match="unexpected kwargs"):
        _dataset(samples_list=raw_list, cues_yaml=cues_yaml,
                          target_sr=16000, typo_key=1)


def test_build_dataset_offline_tse_branch(tmp_path):
    raw_list, cues_yaml, _, _ = _make_wesep_manifest(tmp_path / "d1")
    raw_list2, cues_yaml2, _, _ = _make_wesep_manifest(tmp_path / "d2")
    common = {
        "manifest_format": "path_list",
        "path_base": "manifest_relative",
        "target_selection": "all",
        "mix_variant": "default",
        "cue_schema": "realtse_raw",
        "enabled_cues": ["audio"],
    }
    cfg = {
        "mode": "offline_tse",
        "sample_rate": 16000,
        "train": {**common, "samples_list": raw_list, "cues_yaml": cues_yaml},
        "validation": {
            **common,
            "samples_list": raw_list2,
            "cues_yaml": cues_yaml2,
        },
    }
    train_ds = build_dataset(cfg, mode="train")
    val_ds = build_offline_tse_dataset({"data": cfg}, "val")
    assert isinstance(train_ds, TSEOfflineDataset)
    assert len(train_ds) == 3 and len(val_ds) == 3
    assert train_ds.speaker_identity_to_idx == {"spk0": 0, "spk1": 1}
    # val 复用 train 映射（这里 spk 集合相同，标签一致）
    val_ds.set_speaker_label_mapping(train_ds.speaker_identity_to_idx)


def test_detect_task_routes_offline_tse():
    cfg = {
        "data": {"mode": "offline_tse"},
        "model": {"type": "TSE_BSRNN_SPK"},
        "trainer": {"type": "TSETrainer"},
    }
    info = detect_task(cfg)
    assert info.task == "tse"


pytestmark = [pytest.mark.unit]
