"""TSE 离线真实 worker 与输出样点域长度回归。"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader, get_worker_info

from tasks.tse.data.batch import TSEBatch
from tasks.tse.data.collate import tse_collate_fn
from tasks.tse.data.offline_dataset import TSEOfflineDataset

pytestmark = pytest.mark.unit


@pytest.fixture
def tmp_path():
    with TemporaryDirectory(prefix="tse-offline-regression-") as directory:
        yield Path(directory)


def _write_fixture(root: Path, *, sample_rate=16000, frames=16000, targets=None):
    targets = targets or [("speaker", sample_rate, frames)]
    for index, (name, sr, count) in enumerate(
        [("mix", sample_rate, frames), ("enroll", sample_rate, sample_rate // 4)]
        + targets
    ):
        signal = (0.1 * np.sin(np.arange(count) * (index + 1) * 0.01)).astype(np.float32)
        sf.write(root / f"{name}.wav", signal, sr, subtype="FLOAT")
    manifest = root / "samples.jsonl"
    manifest.write_text(
        json.dumps({
            "key": "sample", "spk": [name for name, _, _ in targets],
            "mix": {"default": ["mix.wav"]},
            "src": {name: [f"{name}.wav"] for name, _, _ in targets},
        }) + "\n", encoding="utf-8",
    )
    (root / "audio.json").write_text(
        json.dumps({name: [{"path": "enroll.wav", "nested": {"labels": ["one", "two"]}}]
                    for name, _, _ in targets}), encoding="utf-8",
    )
    cues = root / "cues.yaml"
    cues.write_text(
        "cues:\n  audio:\n    type: wav\n    format: waveform\n"
        "    guaranteed: true\n    collate: {nested: {labels: [one, two]}}\n"
        "    policy: {type: fixed, key: spk_id, resource: audio.json}\n",
        encoding="utf-8",
    )
    return manifest, cues


def _dataset(root: Path, **kwargs):
    return TSEOfflineDataset(
        samples_jsonl=root / "samples.jsonl", cues_yaml=root / "cues.yaml",
        manifest_format="jsonl", path_base="manifest_relative", mix_variant="default",
        target_selection="all", cue_schema="wesep_v0_1", enabled_cues=("audio",),
        **kwargs,
    )


def _assert_frozen_records(dataset):
    for target in dataset.records[0].targets:
        cue = target.cue("audio")
        for mapping in (cue.items[0], cue.spec.format_metadata["collate"]):
            assert mapping["nested"] == {"labels": ("one", "two")}
            with pytest.raises(TypeError):
                mapping["other"] = 1
            with pytest.raises(TypeError):
                mapping["nested"]["labels"] = ()
            with pytest.raises(TypeError):
                mapping["nested"]["labels"][0] = "changed"
            with pytest.raises(AttributeError):
                mapping.new_attribute = {}


def _check_worker_records(_worker_id):
    torch.set_num_threads(1)
    _assert_frozen_records(get_worker_info().dataset)


@pytest.mark.integration
@pytest.mark.subprocess
def test_real_spawn_matches_zero_workers_and_preserves_frozen_cues(tmp_path):
    _write_fixture(tmp_path, targets=[("speaker", 16000, 16000), ("second", 16000, 12000)])
    dataset = _dataset(tmp_path, max_duration=8000, seed=7)
    _assert_frozen_records(dataset)
    collate = partial(tse_collate_fn, cue_specs=dataset.cue_specs)
    expected = list(DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate))
    actual = list(DataLoader(
        dataset, batch_size=1, num_workers=1, multiprocessing_context="spawn",
        collate_fn=collate, worker_init_fn=_check_worker_records, timeout=30,
    ))
    assert len(actual) == len(expected) == 1
    for left, right in zip(expected, actual):
        assert isinstance(left, TSEBatch) and isinstance(right, TSEBatch)
        left.validate()
        right.validate()
        for field in ("mixture", "target", "sr", "lengths", "num_targets", "spk_labels", "target_present"):
            assert torch.equal(getattr(left, field), getattr(right, field))
        assert left.speaker_ids == right.speaker_ids == ("speaker", "second")
        assert left.keys == right.keys == ("sample",)
        assert left.provenance == right.provenance
        for field in ("value", "lengths", "present"):
            assert torch.equal(
                getattr(left.cues.speaker_waveform, field),
                getattr(right.cues.speaker_waveform, field),
            )


def _assert_output_length(dataset, expected):
    item = dataset[0]
    assert dataset.get_source_length() == [expected]
    assert item["length"] == expected
    assert item["mixture"].shape == (1, expected)
    assert all(target["target"].shape == (expected,) for target in item["targets"])
    batch = tse_collate_fn([item], cue_specs=dataset.cue_specs)
    batch.validate()
    assert batch.lengths.tolist() == [expected]
    assert batch.target.shape == (len(item["targets"]), expected)
    return item


def test_48k_one_second_with_16k_output_cap_is_exactly_16000(tmp_path):
    _write_fixture(tmp_path, sample_rate=48000, frames=48000)
    dataset = _dataset(tmp_path, target_sr=16000, max_duration=16000)
    _assert_output_length(dataset, 16000)


class _LastWindow:
    def randint(self, lower, upper):
        assert 0 <= lower <= upper
        return upper


@pytest.mark.parametrize("source_sr,output_sr,frames,full_length", [
    (16000, 16000, 16001, 16001),
    (48000, 16000, 48001, 16001),
    (16000, 48000, 16001, 48003),
    (44100, 16000, 44101, 16001),
    (16000, 44100, 16001, 44103),
])
@pytest.mark.parametrize("cap_kind", ["short", "equal", "long", "unlimited"])
def test_output_cap_and_last_window_match_full_resampled_audio(
    tmp_path, monkeypatch, source_sr, output_sr, frames, full_length, cap_kind,
):
    from core.data.offline_multitrack import load_audio, resample

    cap = {"short": 1001, "equal": full_length, "long": full_length + 100,
           "unlimited": -1}[cap_kind]
    expected = 1001 if cap_kind == "short" else full_length
    _write_fixture(tmp_path, sample_rate=source_sr, frames=frames)
    dataset = _dataset(tmp_path, target_sr=output_sr, max_duration=cap)
    monkeypatch.setattr(dataset, "_local_rng", lambda _: _LastWindow())
    item = _assert_output_length(dataset, expected)
    assert item["sr"] == output_sr
    start = item["provenance"]["output_start_sample"]
    if cap_kind == "short":
        assert start > 0
    for name, actual in (("mix", item["mixture"]),
                         ("speaker", item["targets"][0]["target"].unsqueeze(0))):
        waveform, sr = load_audio(str(tmp_path / f"{name}.wav"))
        full = resample(waveform, sr, output_sr)
        torch.testing.assert_close(actual, full[:, start:start + expected], rtol=0, atol=0)


@pytest.mark.parametrize("target_sr,target_frames,cap,expected", [
    (48000, 36001, -1, 12001),
    (48000, 36001, 10001, 10001),
    (8000, 6001, -1, 12002),
    (8000, 6001, 11000, 11000),
])
def test_shorter_targets_align_and_mixed_sr_remains_strict(
    tmp_path, monkeypatch, target_sr, target_frames, cap, expected,
):
    from core.data.offline_multitrack import load_audio, resample

    _write_fixture(tmp_path, sample_rate=48000, frames=48000,
                   targets=[("speaker", 48000, 48000), ("short", target_sr, target_frames)])
    if target_sr != 48000:
        strict = _dataset(tmp_path, target_sr=16000, max_duration=cap)
        with pytest.raises(RuntimeError, match="mixture/target sample rate mismatch"):
            strict[0]
    dataset = _dataset(tmp_path, target_sr=16000, max_duration=cap, strict_load=False)
    monkeypatch.setattr(dataset, "_local_rng", lambda _: _LastWindow())
    item = _assert_output_length(dataset, expected)
    assert not item["provenance"]["placeholder"]
    assert [target["speaker_id"] for target in item["targets"]] == ["speaker", "short"]
    start = item["provenance"]["output_start_sample"]
    for target in item["targets"]:
        waveform, sr = load_audio(str(tmp_path / f'{target["speaker_id"]}.wav'))
        full = resample(resample(waveform, sr, 48000), 48000, 16000)
        torch.testing.assert_close(target["target"], full[0, start:start + expected], rtol=0, atol=0)


def test_integer_cap_fast_and_full_paths_are_identical(tmp_path):
    _write_fixture(tmp_path, frames=16001)
    dataset = _dataset(tmp_path, max_duration=1001)
    window = dataset._window_plan(dataset.records[0], _LastWindow())
    fast = dataset._getitem_fast(0, window, 16000)
    full = dataset._getitem_impl_full(0, window)
    assert fast["length"] == full["length"] == 1001
    assert fast["provenance"] == full["provenance"]
    assert torch.equal(fast["mixture"], full["mixture"])
    assert torch.equal(fast["targets"][0]["target"], full["targets"][0]["target"])
    assert torch.equal(fast["targets"][0]["cues"].speaker_waveform.value,
                       full["targets"][0]["cues"].speaker_waveform.value)


@pytest.mark.parametrize("options,expected", [
    ({"max_duration": 16000}, 16000),
    ({"max_duration": 20000}, 20000),
    ({"max_duration": -1}, 321),
    ({"max_duration": 123, "max_duration_seconds": 0.125}, 2000),
])
def test_real_read_failure_placeholder_uses_output_cap(tmp_path, options, expected):
    _write_fixture(tmp_path, sample_rate=48000, frames=48000,
                   targets=[("speaker", 48000, 48000), ("second", 48000, 40000)])
    dataset = _dataset(tmp_path, target_sr=16000, strict_load=False,
                       placeholder_num_samples=321, **options)
    (tmp_path / "mix.wav").write_bytes(b"broken audio")
    item = dataset[0]
    assert item["provenance"]["placeholder"]
    assert item["sr"] == 16000
    assert item["length"] == expected
    assert item["mixture"].shape == (1, expected)
    assert not item["mixture"].any()
    assert len(item["targets"]) == 2
    for target in item["targets"]:
        assert target["target"].shape == (expected,)
        assert not target["target"].any()
        cue = target["cues"].speaker_waveform
        assert not cue.present and cue.value is None and cue.length == 0
    batch = tse_collate_fn([item], cue_specs=dataset.cue_specs)
    batch.validate()
    assert batch.lengths.tolist() == [expected]
    assert not batch.cues.speaker_waveform.present.any()


def test_seconds_cap_overrides_sample_cap_in_output_domain(tmp_path):
    _write_fixture(tmp_path, sample_rate=44100, frames=44101)
    dataset = _dataset(tmp_path, target_sr=16000, max_duration=123, max_duration_seconds=0.125)
    _assert_output_length(dataset, 2000)
