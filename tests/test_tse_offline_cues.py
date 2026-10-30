"""Offline TSE canonical dict、multi-target 与 typed cue 接线测试。"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from core.models.io_contract import ModelIOContract, TensorField
from tasks.tse.data.collate import tse_collate_fn
from tasks.tse.data.offline_dataset import TSEOfflineDataset
from tasks.tse.model_io import build_tse_model_inputs

pytestmark = pytest.mark.unit


def _write_wav(path: Path, value: float, frames: int, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.full(frames, value, dtype=np.float32), sample_rate)


def _dataset(manifest: Path, **kwargs) -> TSEOfflineDataset:
    return TSEOfflineDataset(
        samples_jsonl=manifest,
        manifest_format="jsonl",
        path_base="manifest_relative",
        target_selection="all",
        mix_variant="default",
        **kwargs,
    )


def _write_waveform_fixture(root: Path) -> tuple[Path, Path]:
    audio = root / "audio"
    for name, value in (
        ("mix-left.wav", 0.1),
        ("mix-right.wav", 0.2),
        ("s1-left.wav", 0.3),
        ("s1-right.wav", 0.4),
        ("s2-left.wav", 0.5),
        ("s2-right.wav", 0.6),
    ):
        _write_wav(audio / name, value, 12000)
    _write_wav(audio / "s1-enroll.wav", 0.7, 5000)
    _write_wav(audio / "s2-enroll.wav", 0.8, 7000)
    manifest = root / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "key": "mix",
                "spk": ["s1", "s2"],
                "mix": {"default": ["audio/mix-left.wav", "audio/mix-right.wav"]},
                "src": {
                    "s1": ["audio/s1-left.wav", "audio/s1-right.wav"],
                    "s2": ["audio/s2-left.wav", "audio/s2-right.wav"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cue_dir = root / "cues"
    cue_dir.mkdir()
    (cue_dir / "audio.json").write_text(
        json.dumps(
            {
                "s1": [{"path": "../audio/s1-enroll.wav"}],
                "s2": [{"path": "../audio/s2-enroll.wav"}],
            }
        ),
        encoding="utf-8",
    )
    cues_yaml = root / "cues.yaml"
    cues_yaml.write_text(
        """cues:
  audio:
    type: wav
    format: waveform
    scope: speaker
    guaranteed: true
    policy:
      type: fixed
      key: spk_id
      resource: cues/audio.json
""",
        encoding="utf-8",
    )
    return manifest, cues_yaml


def test_multitarget_multifile_channels_preserve_order_and_reference_target(tmp_path: Path):
    manifest, cues_yaml = _write_waveform_fixture(tmp_path)
    dataset = _dataset(
        manifest,
        cues_yaml=cues_yaml,
        cue_schema="wesep_v0_1",
        enabled_cues=("audio",),
        enroll_policy="fixed",
        placeholder_num_samples=100,
        spatial_enabled=True,
        custom_geometry=((0.0, 0.0, 0.0), (0.05, 0.0, 0.0)),
        reference_channel=1,
        channel_order=("left", "right"),
    )

    item = dataset[0]
    assert item["mixture"].shape == (2, 12000)
    assert [target["speaker_id"] for target in item["targets"]] == ["s1", "s2"]
    assert len(item["targets"]) == 2
    assert torch.allclose(
        item["targets"][0]["target"].mean(), torch.tensor(0.4), atol=2e-4
    )
    assert torch.allclose(
        item["targets"][1]["target"].mean(), torch.tensor(0.6), atol=2e-4
    )
    spec = dataset.cue_specs.speaker_waveform
    assert spec is not None and spec.value_shape == (100,)

    batch = tse_collate_fn([item], cue_specs=dataset.cue_specs)
    assert batch.mixture.shape == (1, 2, 12000)
    assert batch.num_targets.tolist() == [2]
    assert batch.speaker_ids == ("s1", "s2")
    assert batch.cues.speaker_waveform is not None
    assert batch.cues.speaker_waveform.value.shape == (2, 7000)
    assert batch.cues.speaker_waveform.lengths.tolist() == [5000, 7000]


def test_all_waveform_and_embedding_resource_paths_validate_before_sampling(tmp_path: Path):
    manifest = tmp_path / "samples.jsonl"
    _write_wav(tmp_path / "mix.wav", 0.1, 4000)
    _write_wav(tmp_path / "target.wav", 0.2, 4000)
    _write_wav(tmp_path / "enroll.wav", 0.3, 2000)
    manifest.write_text(
        json.dumps(
            {
                "key": "x",
                "spk": ["s"],
                "mix": {"default": ["mix.wav"]},
                "src": {"s": ["target.wav"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "audio.json").write_text(
        json.dumps({"s": [{"path": "enroll.wav"}, {"path": "missing.wav"}]}),
        encoding="utf-8",
    )
    waveform_yaml = tmp_path / "waveform.yaml"
    waveform_yaml.write_text(
        """cues:
  audio:
    type: wav
    format: waveform
    guaranteed: true
    scope: speaker
    policy: {type: fixed, key: spk_id, resource: audio.json}
""",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="cannot read audio metadata.*missing.wav"):
        _dataset(
            manifest,
            cues_yaml=waveform_yaml,
            cue_schema="wesep_v0_1",
            enabled_cues=("audio",),
        )

    np.save(tmp_path / "good.npy", np.ones(4, dtype=np.float32))
    np.save(tmp_path / "bad.npy", np.ones(5, dtype=np.float32))
    (tmp_path / "embedding.json").write_text(
        json.dumps({"s": [{"path": "good.npy"}, {"path": "bad.npy"}]}),
        encoding="utf-8",
    )
    embedding_yaml = tmp_path / "embedding.yaml"
    embedding_yaml.write_text(
        """cues:
  audio:
    type: npy
    format: spk_embedding
    guaranteed: true
    scope: speaker
    policy: {type: fixed, key: spk_id, resource: embedding.json}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="embedding dimensions must be fixed"):
        _dataset(
            manifest,
            cues_yaml=embedding_yaml,
            cue_schema="wesep_v0_1",
            enabled_cues=("audio",),
        )


def test_embedding_and_no_cue_datasets_publish_actual_cue_specs(tmp_path: Path):
    _write_wav(tmp_path / "mix.wav", 0.1, 4000)
    _write_wav(tmp_path / "target.wav", 0.2, 4000)
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "key": "x",
                "spk": ["s"],
                "mix": {"default": ["mix.wav"]},
                "src": {"s": ["target.wav"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    np.save(tmp_path / "embedding.npy", np.arange(4, dtype=np.float32))
    (tmp_path / "embedding.json").write_text(
        json.dumps({"s": [{"path": "embedding.npy"}]}), encoding="utf-8"
    )
    cues_yaml = tmp_path / "cues.yaml"
    cues_yaml.write_text(
        """cues:
  audio:
    type: npy
    format: spk_embedding
    guaranteed: true
    scope: speaker
    policy: {type: fixed, key: spk_id, resource: embedding.json}
""",
        encoding="utf-8",
    )
    embedding_dataset = _dataset(
        manifest,
        cues_yaml=cues_yaml,
        cue_schema="wesep_v0_1",
        enabled_cues=("audio",),
    )
    embedding_item = embedding_dataset[0]
    embedding = embedding_item["targets"][0]["cues"].speaker_embedding
    assert embedding is not None and embedding.present
    assert embedding_dataset.cue_specs.speaker_embedding is not None
    assert embedding_dataset.cue_specs.speaker_waveform is None
    batch = tse_collate_fn([embedding_item], cue_specs=embedding_dataset.cue_specs)
    assert batch.cues.speaker_embedding is not None
    assert batch.cues.speaker_embedding.value.shape == (1, 4)

    no_cue_dataset = _dataset(
        manifest,
        cues_yaml=None,
        cue_schema="none",
        enabled_cues=(),
    )
    no_cue_item = no_cue_dataset[0]
    assert no_cue_item["targets"][0]["cues"].speaker_waveform is None
    assert no_cue_dataset.cue_specs.speaker_waveform is None
    no_cue_batch = tse_collate_fn([no_cue_item], cue_specs=no_cue_dataset.cue_specs)
    assert no_cue_batch.cues.speaker_waveform is None


@pytest.mark.parametrize(
    ("cue_name", "cue_body", "message"),
    [
        (
            "visual",
            """  visual:
    type: mp4
    format: raw_video
    guaranteed: true
    scope: speaker
    policy: {type: fixed, key: mix_spk_id, resource: cue.json}
""",
            "raw video visual cue requires an optional video decoder",
        ),
        (
            "visual",
            """  visual:
    type: npy
    format: raw_video
    guaranteed: true
    scope: speaker
    policy: {type: fixed, key: mix_spk_id, resource: cue.json}
""",
            "unsupported visual cue type/format",
        ),
        (
            "textual",
            """  textual:
    type: json
    format: keyword_phoneme
    guaranteed: true
    scope: speaker
    selection: {type: fixed}
    policy: {type: fixed, key: mix_spk_id, resource: cue.json}
""",
            "unsupported textual cue type/format",
        ),
    ],
)
def test_unsupported_cue_representation_is_rejected_after_strict_index_import(
    tmp_path: Path, cue_name: str, cue_body: str, message: str
):
    _write_wav(tmp_path / "mix.wav", 0.1, 4000)
    _write_wav(tmp_path / "target.wav", 0.2, 4000)
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "key": "x",
                "spk": ["s"],
                "mix": {"default": ["mix.wav"]},
                "src": {"s": ["target.wav"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if cue_name == "visual":
        (tmp_path / "cue.json").write_text(
            json.dumps({"x::s": [{"path": "visual.npy"}]}), encoding="utf-8"
        )
    else:
        (tmp_path / "cue.json").write_text(
            json.dumps({"x::s": {"phn_label": [[1]], "kw_candidate": [0]}}),
            encoding="utf-8",
        )
    cues_yaml = tmp_path / "cues.yaml"
    cues_yaml.write_text("cues:\n" + cue_body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _dataset(
            manifest,
            cues_yaml=cues_yaml,
            cue_schema="wesep_v0_1",
            enabled_cues=(cue_name,),
        )


def test_cue_schema_selection_is_explicit_and_no_fallback_occurs(tmp_path: Path):
    manifest, v01_cues = _write_waveform_fixture(tmp_path)
    with pytest.raises(ValueError, match="realtse_raw.*type='raw'"):
        _dataset(
            manifest,
            cues_yaml=v01_cues,
            cue_schema="realtse_raw",
            enabled_cues=("audio",),
        )

    with pytest.raises(ValueError, match="requires explicit parameters"):
        TSEOfflineDataset(samples_jsonl=manifest)


def test_lenient_single_sample_failure_returns_canonical_placeholder(tmp_path: Path):
    _write_wav(tmp_path / "mix.wav", 0.1, 4000)
    _write_wav(tmp_path / "target.wav", 0.2, 4000)
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "key": "x",
                "spk": ["s"],
                "mix": {"default": ["mix.wav"]},
                "src": {"s": ["target.wav"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = _dataset(
        manifest,
        cues_yaml=None,
        cue_schema="none",
        enabled_cues=(),
        strict_load=False,
        placeholder_num_samples=321,
    )
    (tmp_path / "mix.wav").write_bytes(b"broken after validation")

    item = dataset[0]
    assert item["provenance"] == {"placeholder": True}
    assert item["mixture"].shape == (1, 321)
    assert torch.count_nonzero(item["mixture"]) == 0
    assert item["length"] == 321
    assert len(item["targets"]) == 1
    assert item["targets"][0]["spk_label"] == -1
    assert torch.count_nonzero(item["targets"][0]["target"]) == 0
    batch = tse_collate_fn([item], cue_specs=dataset.cue_specs)
    assert batch.num_targets.tolist() == [1]


def test_all_missing_enabled_waveform_uses_fallback_capacity_without_fake_cue(tmp_path: Path):
    _write_wav(tmp_path / "mix.wav", 0.1, 4000)
    _write_wav(tmp_path / "target.wav", 0.2, 4000)
    _write_wav(tmp_path / "unused.wav", 0.3, 1000)
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "key": "x",
                "spk": ["s"],
                "mix": {"default": ["mix.wav"]},
                "src": {"s": ["target.wav"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "audio.json").write_text(
        json.dumps({"other": [{"path": "unused.wav"}]}), encoding="utf-8"
    )
    cues_yaml = tmp_path / "cues.yaml"
    cues_yaml.write_text(
        """cues:
  audio:
    type: wav
    format: waveform
    guaranteed: false
    scope: speaker
    policy: {type: fixed, key: spk_id, resource: audio.json}
""",
        encoding="utf-8",
    )
    dataset = _dataset(
        manifest,
        cues_yaml=cues_yaml,
        cue_schema="wesep_v0_1",
        enabled_cues=("audio",),
        placeholder_num_samples=123,
    )

    spec = dataset.cue_specs.speaker_waveform
    cue = dataset[0]["targets"][0]["cues"].speaker_waveform
    assert spec is not None and spec.value_shape == (123,)
    assert cue is not None and cue.present is False and cue.length == 0
    batch = tse_collate_fn([dataset[0]], cue_specs=dataset.cue_specs)
    assert batch.cues.speaker_waveform is not None
    assert batch.cues.speaker_waveform.value.shape == (1, 123)
    assert batch.cues.speaker_waveform.present.tolist() == [False]


def test_unselected_long_waveform_does_not_inflate_short_batch(tmp_path: Path):
    _write_wav(tmp_path / "mix.wav", 0.1, 1000)
    _write_wav(tmp_path / "target.wav", 0.2, 1000)
    _write_wav(tmp_path / "short.wav", 0.3, 80)
    _write_wav(tmp_path / "long.wav", 0.4, 200000)
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "key": "x",
                "spk": ["s"],
                "mix": {"default": ["mix.wav"]},
                "src": {"s": ["target.wav"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "audio.json").write_text(
        json.dumps({"s": [{"path": "short.wav"}, {"path": "long.wav"}]}),
        encoding="utf-8",
    )
    cues_yaml = tmp_path / "cues.yaml"
    cues_yaml.write_text(
        """cues:
  audio:
    type: wav
    format: waveform
    guaranteed: true
    scope: speaker
    policy: {type: fixed, key: spk_id, resource: audio.json}
""",
        encoding="utf-8",
    )
    dataset = _dataset(
        manifest,
        cues_yaml=cues_yaml,
        cue_schema="wesep_v0_1",
        enabled_cues=("audio",),
        placeholder_num_samples=64,
    )

    spec = dataset.cue_specs.speaker_waveform
    assert spec is not None and spec.value_shape == (64,)
    assert str(tmp_path / "long.wav") in dataset._waveform_cue_info
    item = dataset[0]
    cue = item["targets"][0]["cues"].speaker_waveform
    assert cue is not None and cue.length == 80
    batch = tse_collate_fn([item], cue_specs=dataset.cue_specs)
    assert batch.cues.speaker_waveform is not None
    assert batch.cues.speaker_waveform.value.shape == (1, 80)


def test_cue_resampling_keeps_main_tracks_on_window_path(tmp_path: Path, monkeypatch):
    _write_wav(tmp_path / "mix.wav", 0.1, 3200, sample_rate=16000)
    _write_wav(tmp_path / "target.wav", 0.2, 3200, sample_rate=16000)
    _write_wav(tmp_path / "cue.wav", 0.3, 1600, sample_rate=8000)
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "key": "x",
                "spk": ["s"],
                "mix": {"default": ["mix.wav"]},
                "src": {"s": ["target.wav"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "audio.json").write_text(
        json.dumps({"s": [{"path": "cue.wav"}]}), encoding="utf-8"
    )
    cues_yaml = tmp_path / "cues.yaml"
    cues_yaml.write_text(
        """cues:
  audio:
    type: wav
    format: waveform
    guaranteed: true
    scope: speaker
    policy: {type: random, key: spk_id, resource: audio.json}
""",
        encoding="utf-8",
    )
    dataset = _dataset(
        manifest,
        cues_yaml=cues_yaml,
        cue_schema="wesep_v0_1",
        enabled_cues=("audio",),
        strict_load=False,
        seed=11,
        max_duration=1000,
        max_enroll_duration_seconds=0.05,
        placeholder_num_samples=64,
    )
    record = dataset.records[0]
    window = dataset._window_plan(record, dataset._local_rng(0))
    calls = []
    load_window = dataset._load_audio_group_window

    def tracked_window(paths, start, length, *, source):
        calls.append((tuple(paths), start, length))
        return load_window(paths, start, length, source=source)

    monkeypatch.setattr(dataset, "_load_audio_group_window", tracked_window)
    monkeypatch.setattr(
        dataset,
        "_getitem_impl_full",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cue-only resampling must not full-load main tracks")
        ),
    )

    item = dataset[0]
    cue = item["targets"][0]["cues"].speaker_waveform
    assert item["mixture"].shape == (1, 1000)
    assert item["targets"][0]["target"].shape == (1000,)
    assert cue is not None and cue.length == 800
    assert (
        (str(tmp_path / "cue.wav"),),
        window["cue_plans"][0]["start"],
        window["cue_plans"][0]["length"],
    ) in calls


def test_multitarget_placeholder_is_zero_and_drops_real_cues(tmp_path: Path):
    manifest, cues_yaml = _write_waveform_fixture(tmp_path)
    dataset = _dataset(
        manifest,
        cues_yaml=cues_yaml,
        cue_schema="wesep_v0_1",
        enabled_cues=("audio",),
        enroll_policy="fixed",
        strict_load=False,
        placeholder_num_samples=32,
        spatial_enabled=True,
        custom_geometry=((0.0, 0.0, 0.0), (0.05, 0.0, 0.0)),
        reference_channel=1,
        channel_order=("left", "right"),
    )
    (tmp_path / "audio" / "mix-left.wav").write_bytes(b"broken after validation")

    item = dataset[0]
    assert item["provenance"] == {"placeholder": True}
    assert item["mixture"].shape == (2, 32)
    assert torch.count_nonzero(item["mixture"]) == 0
    assert len(item["targets"]) == 2
    for target in item["targets"]:
        assert target["spk_label"] == -1
        assert torch.count_nonzero(target["target"]) == 0
        cue = target["cues"].speaker_waveform
        assert cue is not None and cue.present is False and cue.length == 0
    batch = tse_collate_fn([item], cue_specs=dataset.cue_specs)
    assert batch.spk_labels.tolist() == [-1, -1]
    assert torch.count_nonzero(batch.mixture) == 0
    assert torch.count_nonzero(batch.target) == 0


def _write_fallback_manifest(root: Path, specs) -> tuple[Path, dict[str, Path]]:
    records = []
    mixture_paths = {}
    for key, sample_rate, target_count, channels in specs:
        mixture = root / f"{key}-mix.wav"
        waveform = np.zeros((800, channels), dtype=np.float32)
        sf.write(mixture, waveform, sample_rate)
        mixture_paths[key] = mixture
        speakers = [f"{key}-speaker-{index}" for index in range(target_count)]
        sources = {}
        for speaker in speakers:
            target = root / f"{speaker}-target.wav"
            _write_wav(target, 0.1, 800, sample_rate=sample_rate)
            sources[speaker] = [target.name]
        records.append(
            {
                "key": key,
                "spk": speakers,
                "mix": {"default": [mixture.name]},
                "src": sources,
            }
        )
    manifest = root / "fallback-samples.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest, mixture_paths


def test_fallback_skips_different_native_sr_when_target_sr_is_none(tmp_path: Path):
    manifest, mixtures = _write_fallback_manifest(
        tmp_path,
        (("bad-8k", 8000, 1, 1), ("wrong-16k", 16000, 1, 1), ("same-8k", 8000, 1, 1)),
    )
    dataset = _dataset(
        manifest,
        cues_yaml=None,
        cue_schema="none",
        enabled_cues=(),
        strict_load=False,
        target_sr=None,
    )
    mixtures["bad-8k"].write_bytes(b"broken after validation")

    item = dataset[0]
    assert item["key"] == "same-8k"
    assert item["sr"] == 8000
    assert item["provenance"]["placeholder"] is False


def test_fallback_uses_same_output_sr_candidate(tmp_path: Path):
    manifest, mixtures = _write_fallback_manifest(
        tmp_path,
        (("bad-16k", 16000, 1, 1), ("same-16k", 16000, 1, 1)),
    )
    dataset = _dataset(
        manifest,
        cues_yaml=None,
        cue_schema="none",
        enabled_cues=(),
        strict_load=False,
        target_sr=None,
    )
    mixtures["bad-16k"].write_bytes(b"broken after validation")

    item = dataset[0]
    assert item["key"] == "same-16k"
    assert item["sr"] == 16000
    assert item["provenance"]["placeholder"] is False


def test_fallback_without_same_sr_preserves_failed_placeholder_contract(tmp_path: Path):
    manifest, mixtures = _write_fallback_manifest(
        tmp_path,
        (("bad-8k", 8000, 2, 2), ("wrong-16k", 16000, 1, 2)),
    )
    dataset = _dataset(
        manifest,
        cues_yaml=None,
        cue_schema="none",
        enabled_cues=(),
        strict_load=False,
        target_sr=None,
        placeholder_num_samples=64,
        spatial_enabled=True,
        custom_geometry=((0.0, 0.0, 0.0), (0.05, 0.0, 0.0)),
        reference_channel=1,
        channel_order=("left", "right"),
    )
    mixtures["bad-8k"].write_bytes(b"broken after validation")

    item = dataset[0]
    assert item["key"] == "bad-8k"
    assert item["sr"] == 8000
    assert item["provenance"] == {"placeholder": True}
    assert item["geometry"] is dataset.resolved_microphone_array
    assert item["mixture"].shape == (2, 64)
    assert len(item["targets"]) == 2
    assert all(target["spk_label"] == -1 for target in item["targets"])
    assert torch.count_nonzero(item["mixture"]) == 0
    assert all(torch.count_nonzero(target["target"]) == 0 for target in item["targets"])


def test_fixed_target_sr_allows_fallback_across_native_sr(tmp_path: Path):
    manifest, mixtures = _write_fallback_manifest(
        tmp_path,
        (("bad-native-8k", 8000, 1, 1), ("good-native-16k", 16000, 1, 1)),
    )
    dataset = _dataset(
        manifest,
        cues_yaml=None,
        cue_schema="none",
        enabled_cues=(),
        strict_load=False,
        target_sr=16000,
    )
    mixtures["bad-native-8k"].write_bytes(b"broken after validation")

    item = dataset[0]
    assert item["key"] == "good-native-16k"
    assert item["sr"] == 16000
    assert item["provenance"]["placeholder"] is False


_SPATIAL_SPLIT = {
    "spatial_enabled": True,
    "custom_geometry": ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0)),
    "reference_channel": 0,
    "channel_order": ("left", "right"),
    "spatial_angle_unit": "radian",
    "spatial_coordinate_frame": "array_local_azimuth_elevation_from_xy",
    "spatial_speed_of_sound_mps": 343.0,
}


class _OfflineSpatialModel:
    sample_rate = 16000
    supports_multichannel_input = True
    reference_channel = 0

    def __init__(self, inputs):
        self.IO_CONTRACT = ModelIOContract(
            tasks=frozenset({"tse"}),
            inputs=inputs,
            outputs=(
                TensorField("target_estimate", "waveform", ("batch", "time")),
            ),
            forward_kind="estimate",
            frontend_owner="runtime",
        )


def _write_spatial_fixture(
    root: Path,
    *,
    spatial_values: dict[str, np.ndarray | None],
    fields: tuple[str, ...],
    defaults: dict[str, float] | None = None,
    include_audio: bool = False,
    track_clock: bool = False,
) -> tuple[Path, Path]:
    frames = 1600
    mixture = root / "mix.wav"
    mixture.parent.mkdir(parents=True, exist_ok=True)
    sf.write(mixture, np.zeros((frames, 2), dtype=np.float32), 16000)
    speakers = list(spatial_values)
    sources = {}
    spatial_resource = {}
    audio_resource = {}
    for index, speaker in enumerate(speakers):
        target = root / f"{speaker}-target.wav"
        _write_wav(target, 0.1 + index * 0.1, frames)
        sources[speaker] = [target.name]
        spatial_value = spatial_values[speaker]
        if spatial_value is not None:
            spatial_path = root / f"{speaker}-spatial.npy"
            np.save(spatial_path, spatial_value, allow_pickle=False)
            item = {"path": spatial_path.name}
            if track_clock:
                item.update(
                    {
                        "sample_rate": 16000,
                        "num_samples": frames,
                        "duration_sec": frames / 16000,
                    }
                )
            spatial_resource[f"mix::{speaker}"] = item
        if include_audio:
            enrollment = root / f"{speaker}-enroll.wav"
            _write_wav(enrollment, 0.3 + index * 0.1, 320)
            audio_resource[speaker] = [{"path": enrollment.name}]
    manifest = root / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "key": "mix",
                "spk": speakers,
                "mix": {"default": [mixture.name]},
                "src": sources,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "spatial.json").write_text(
        json.dumps(spatial_resource), encoding="utf-8"
    )
    sections = [
        "  spatial:\n",
        "    type: npy\n",
        "    format: fields\n",
        f"    guaranteed: {str(all(value is not None for value in spatial_values.values())).lower()}\n",
        "    scope: speaker\n",
        f"    fields: {json.dumps(list(fields))}\n",
        f"    field_defaults: {json.dumps(defaults or {})}\n",
        "    policy: {type: fixed, key: mix_spk_id, resource: spatial.json}\n",
    ]
    if include_audio:
        (root / "audio.json").write_text(
            json.dumps(audio_resource), encoding="utf-8"
        )
        sections.extend(
            [
                "  audio:\n",
                "    type: wav\n",
                "    format: waveform\n",
                "    guaranteed: true\n",
                "    scope: speaker\n",
                "    policy: {type: fixed, key: spk_id, resource: audio.json}\n",
            ]
        )
    cues_yaml = root / "cues.yaml"
    cues_yaml.write_text("cues:\n" + "".join(sections), encoding="utf-8")
    return manifest, cues_yaml


def _spatial_dataset(manifest: Path, cues_yaml: Path, **kwargs):
    options = {**_SPATIAL_SPLIT, **kwargs}
    return _dataset(
        manifest,
        cues_yaml=cues_yaml,
        cue_schema="wesep_v0_1",
        enabled_cues=options.pop("enabled_cues", ("spatial",)),
        **options,
    )


def test_offline_static_spatial_reorders_defaults_and_reaches_model_io(tmp_path: Path):
    structured = np.zeros(1, dtype=[("azimuth", "f4")])
    structured["azimuth"] = -0.4
    manifest, cues_yaml = _write_spatial_fixture(
        tmp_path,
        spatial_values={
            "s1": np.array([[0.25, 1.0]], dtype=np.float32),
            "s2": structured,
        },
        fields=("elevation", "azimuth"),
        defaults={"elevation": 0.15},
    )
    dataset = _spatial_dataset(
        manifest,
        cues_yaml,
        spatial_representation="static",
    )
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=1,
                collate_fn=partial(tse_collate_fn, cue_specs=dataset.cue_specs),
            )
        )
    )

    assert dataset.cue_specs.spatial is not None
    assert dataset.cue_specs.spatial.representation == "static"
    assert batch.num_targets.tolist() == [2]
    assert batch.cues.spatial is not None
    torch.testing.assert_close(
        batch.cues.spatial.value,
        torch.tensor([[1.0, 0.25], [-0.4, 0.15]]),
    )
    assert batch.cues.spatial.lengths is None
    assert batch.cues.spatial.timebase is None
    spatial_provenance = batch.provenance[0]["spatial_cue"]
    assert spatial_provenance["speed_of_sound_mps"] == 343.0
    assert spatial_provenance["field_order"] == ("elevation", "azimuth")

    model = _OfflineSpatialModel(
        (
            TensorField(
                "mix_wav", "waveform", ("batch", "microphone", "time")
            ),
            TensorField("spatial", "embedding", ("batch", "coordinate")),
            TensorField("spatial_present", "mask", ("batch",)),
        )
    )
    inputs = build_tse_model_inputs(
        model=model,
        stft=None,
        mixture=batch.mixture,
        cues=batch.cues,
        target=batch.target,
        sample_rate=16000,
        lengths=batch.lengths,
        num_targets=batch.num_targets,
        reference_channel=batch.geometry[0].reference_channel,
    )
    assert inputs["mix_wav"].shape == (2, 2, 1600)
    torch.testing.assert_close(inputs["spatial"], batch.cues.spatial.value)
    assert inputs["spatial_present"].tolist() == [True, True]


def test_offline_speaker_and_spatial_cues_cover_all_targets(tmp_path: Path):
    manifest, cues_yaml = _write_spatial_fixture(
        tmp_path,
        spatial_values={
            "s1": np.array([[0.1, 0.2]], dtype=np.float32),
            "s2": np.array([[-0.3, 0.4]], dtype=np.float32),
        },
        fields=("azimuth", "elevation"),
        include_audio=True,
    )
    dataset = _spatial_dataset(
        manifest,
        cues_yaml,
        enabled_cues=("audio", "spatial"),
        spatial_representation="static",
        enroll_policy="fixed",
    )
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=1,
                collate_fn=partial(tse_collate_fn, cue_specs=dataset.cue_specs),
            )
        )
    )

    assert batch.num_targets.tolist() == [2]
    assert batch.speaker_ids == ("s1", "s2")
    assert batch.cues.speaker_waveform is not None
    assert batch.cues.speaker_waveform.present.tolist() == [True, True]
    assert batch.cues.spatial is not None
    assert batch.cues.spatial.present.tolist() == [True, True]
    model = _OfflineSpatialModel(
        (
            TensorField(
                "mix_wav", "waveform", ("batch", "microphone", "time")
            ),
            TensorField("speaker_waveform", "waveform", ("batch", "time")),
            TensorField("speaker_waveform_lengths", "label", ("batch",)),
            TensorField("speaker_waveform_present", "mask", ("batch",)),
            TensorField("spatial", "embedding", ("batch", "coordinate")),
            TensorField("spatial_present", "mask", ("batch",)),
        )
    )
    inputs = build_tse_model_inputs(
        model=model,
        stft=None,
        mixture=batch.mixture,
        cues=batch.cues,
        sample_rate=16000,
        lengths=batch.lengths,
        num_targets=batch.num_targets,
        reference_channel=0,
    )
    assert inputs["speaker_waveform"].shape[0] == 2
    assert inputs["speaker_waveform"].shape[1] >= 320
    assert inputs["speaker_waveform_lengths"].tolist() == [320, 320]
    assert inputs["spatial"].shape == (2, 2)


def test_offline_embedding_and_spatial_cues_coexist(tmp_path: Path):
    manifest, cues_yaml = _write_spatial_fixture(
        tmp_path,
        spatial_values={"s1": np.array([[0.2, -0.1]], dtype=np.float32)},
        fields=("azimuth", "elevation"),
    )
    np.save(tmp_path / "embedding.npy", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    (tmp_path / "audio.json").write_text(
        json.dumps({"s1": [{"path": "embedding.npy"}]}), encoding="utf-8"
    )
    with cues_yaml.open("a", encoding="utf-8") as stream:
        stream.write(
            "  audio:\n"
            "    type: npy\n"
            "    format: spk_embedding\n"
            "    guaranteed: true\n"
            "    scope: speaker\n"
            "    policy: {type: fixed, key: spk_id, resource: audio.json}\n"
        )
    dataset = _spatial_dataset(
        manifest,
        cues_yaml,
        enabled_cues=("audio", "spatial"),
        spatial_representation="static",
    )
    batch = tse_collate_fn([dataset[0]], cue_specs=dataset.cue_specs)

    assert batch.cues.speaker_embedding is not None
    assert batch.cues.speaker_embedding.value.shape == (1, 3)
    assert batch.cues.spatial is not None
    model = _OfflineSpatialModel(
        (
            TensorField(
                "mix_wav", "waveform", ("batch", "microphone", "time")
            ),
            TensorField("speaker_embedding", "embedding", ("batch", "embedding")),
            TensorField("speaker_embedding_present", "mask", ("batch",)),
            TensorField("spatial", "embedding", ("batch", "coordinate")),
            TensorField("spatial_present", "mask", ("batch",)),
        )
    )
    inputs = build_tse_model_inputs(
        model=model,
        stft=None,
        mixture=batch.mixture,
        cues=batch.cues,
        sample_rate=16000,
        num_targets=batch.num_targets,
        reference_channel=0,
    )
    assert inputs["speaker_embedding"].shape == (1, 3)
    assert inputs["spatial"].shape == (1, 2)


def test_offline_optional_missing_spatial_is_explicit(tmp_path: Path):
    manifest, cues_yaml = _write_spatial_fixture(
        tmp_path,
        spatial_values={
            "s1": np.array([[0.1, -0.2]], dtype=np.float32),
            "s2": None,
        },
        fields=("azimuth", "elevation"),
    )
    dataset = _spatial_dataset(
        manifest,
        cues_yaml,
        spatial_representation="static",
    )
    batch = tse_collate_fn([dataset[0]], cue_specs=dataset.cue_specs)

    assert batch.cues.spatial is not None
    assert batch.cues.spatial.present.tolist() == [True, False]
    torch.testing.assert_close(batch.cues.spatial.value[1], torch.zeros(2))


def test_offline_track_spatial_uses_source_clock_and_crop_relative_anchors(tmp_path: Path):
    payload = np.array(
        [
            [0, 0.0, 0.1],
            [400, 0.2, 0.3],
            [900, 0.4, 0.5],
            [1400, 0.6, 0.7],
        ],
        dtype=np.float64,
    )
    manifest, cues_yaml = _write_spatial_fixture(
        tmp_path,
        spatial_values={"s1": payload},
        fields=("sample_anchor", "elevation", "azimuth"),
        track_clock=True,
    )
    dataset = _spatial_dataset(
        manifest,
        cues_yaml,
        spatial_representation="track",
        spatial_track_timebase="source_samples",
        max_duration=400,
        target_sr=8000,
        seed=5,
    )
    record = dataset.records[0]
    window = dataset._window_plan(record, dataset._local_rng(0))
    item = dataset[0]
    cue = item["targets"][0]["cues"].spatial

    assert cue is not None and cue.present and cue.representation == "track"
    assert window["mix_start"] == 552
    anchors = torch.as_tensor(cue.timebase)
    assert anchors.tolist() == [0, 174]
    torch.testing.assert_close(
        torch.as_tensor(cue.value),
        torch.tensor([[0.3, 0.5], [0.2, 0.4]]),
    )
    assert int(anchors[-1].item()) < item["length"]
    assert cue.length == cue.value.shape[1] == anchors.numel()
    assert cue.value.shape[1] < item["length"]
    assert item["provenance"]["source_start_sample"] == window["mix_start"]
    batch = tse_collate_fn([item], cue_specs=dataset.cue_specs)
    assert batch.cues.spatial is not None
    assert batch.cues.spatial.timebase is not None
    assert batch.cues.spatial.timebase[0, 0].item() == 0


def test_offline_spatial_split_and_payload_validation_fail_fast(tmp_path: Path):
    manifest, cues_yaml = _write_spatial_fixture(
        tmp_path / "static",
        spatial_values={"s1": np.array([[0.1, 0.2]], dtype=np.float32)},
        fields=("azimuth", "elevation"),
    )
    with pytest.raises(ValueError, match="spatial_enabled=True with geometry"):
        _dataset(
            manifest,
            cues_yaml=cues_yaml,
            cue_schema="wesep_v0_1",
            enabled_cues=("spatial",),
            spatial_representation="static",
            spatial_angle_unit="radian",
            spatial_coordinate_frame="array_local_azimuth_elevation_from_xy",
            spatial_speed_of_sound_mps=343.0,
        )
    with pytest.raises(ValueError, match="spatial_angle_unit.*radian"):
        _spatial_dataset(
            manifest,
            cues_yaml,
            spatial_representation="static",
            spatial_angle_unit="degree",
        )

    bad_manifest, bad_cues = _write_spatial_fixture(
        tmp_path / "bad-shape",
        spatial_values={"s1": np.array([[0.1, 0.2, 0.3]], dtype=np.float32)},
        fields=("azimuth", "elevation"),
    )
    with pytest.raises(ValueError, match="declared field order"):
        _spatial_dataset(
            bad_manifest,
            bad_cues,
            spatial_representation="static",
        )

    track_manifest, track_cues = _write_spatial_fixture(
        tmp_path / "bad-track",
        spatial_values={
            "s1": np.array([[1, 0.1, 0.2], [200, 0.2, 0.3]], dtype=np.float64)
        },
        fields=("sample_anchor", "azimuth", "elevation"),
        track_clock=True,
    )
    with pytest.raises(ValueError, match="start at source sample 0"):
        _spatial_dataset(
            track_manifest,
            track_cues,
            spatial_representation="track",
            spatial_track_timebase="source_samples",
        )

    clock_root = tmp_path / "bad-clock"
    clock_manifest, clock_cues = _write_spatial_fixture(
        clock_root,
        spatial_values={
            "s1": np.array([[0, 0.1, 0.2], [200, 0.2, 0.3]], dtype=np.float64)
        },
        fields=("sample_anchor", "azimuth", "elevation"),
        track_clock=True,
    )
    resource = json.loads((clock_root / "spatial.json").read_text())
    resource["mix::s1"]["sample_rate"] = 8000
    (clock_root / "spatial.json").write_text(json.dumps(resource))
    with pytest.raises(ValueError, match="sample clock.*does not match"):
        _spatial_dataset(
            clock_manifest,
            clock_cues,
            spatial_representation="track",
            spatial_track_timebase="source_samples",
        )
