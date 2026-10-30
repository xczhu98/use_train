"""Offline TSE visual/textual cue 解码：canonical typed cue → collate → TSECueBatch。

fixture 全部是本机生成的小 npy/json：visual 走预提取 ``npy/muse_frontend``，
textual 走 ``json/dae_keyword_phoneme``；raw mp4 与缺失资源必须显式 fail-fast。
"""

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

FRAMES = 16000
SAMPLE_RATE = 16000
VISUAL_DIM = 5
VISUAL_FRAMES = 50
VISUAL_DURATION = 1.0
VISUAL_FPS = VISUAL_FRAMES / VISUAL_DURATION

_S1_WORDS = [[7, 45], [22, 5], [36, 32], [56, 39]]
_S2_WORDS = [[11, 12], [13, 14], [15, 16]]


def _write_wav(path: Path, value: float, frames: int, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.full(frames, value, dtype=np.float32), sample_rate)


def _write_manifest(root: Path, speakers: tuple[str, ...], *, frames: int = FRAMES) -> Path:
    mixture = root / "mix.wav"
    sf.write(mixture, np.zeros(frames, dtype=np.float32), SAMPLE_RATE)
    sources = {}
    for index, speaker in enumerate(speakers):
        target = root / f"{speaker}-target.wav"
        _write_wav(target, 0.1 + 0.1 * index, frames)
        sources[speaker] = [target.name]
    manifest = root / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "key": "mix",
                "spk": list(speakers),
                "mix": {"default": [mixture.name]},
                "src": sources,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _visual_payload(dim: int = VISUAL_DIM, frames: int = VISUAL_FRAMES, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((dim, frames)).astype(np.float32)


def _write_visual_resource(
    root: Path,
    items: dict[str, dict],
    *,
    guaranteed: bool,
) -> str:
    resource = {}
    for speaker, item in items.items():
        payload = item.get("payload")
        name = item.get("name", f"{speaker}-visual.npy")
        if payload is not None:
            np.save(root / name, payload, allow_pickle=False)
        entry = {"path": name}
        for key in ("duration_sec", "fps", "utt_id"):
            if item.get(key) is not None:
                entry[key] = item[key]
        resource[f"mix::{speaker}"] = [entry]
    (root / "visual.json").write_text(json.dumps(resource), encoding="utf-8")
    return (
        "  visual:\n"
        "    type: npy\n"
        "    format: muse_frontend\n"
        f"    guaranteed: {str(guaranteed).lower()}\n"
        "    scope: speaker\n"
        "    policy: {type: fixed, key: mix_spk_id, resource: visual.json}\n"
    )


def _write_textual_resource(
    root: Path,
    values: dict[str, dict],
    *,
    guaranteed: bool,
    selection: dict,
    collate: dict | None = None,
) -> str:
    resource = {f"mix::{speaker}": value for speaker, value in values.items()}
    (root / "textual.json").write_text(json.dumps(resource), encoding="utf-8")
    section = (
        "  textual:\n"
        "    type: json\n"
        "    format: dae_keyword_phoneme\n"
        f"    guaranteed: {str(guaranteed).lower()}\n"
        "    scope: speaker\n"
        f"    selection: {json.dumps(selection)}\n"
    )
    if collate is not None:
        section += f"    collate: {json.dumps(collate)}\n"
    section += "    policy: {type: fixed, key: mix_spk_id, resource: textual.json}\n"
    return section


def _write_audio_resource(root: Path, speakers: tuple[str, ...]) -> str:
    resource = {}
    for index, speaker in enumerate(speakers):
        _write_wav(root / f"{speaker}-enroll.wav", 0.5 + 0.1 * index, 3200)
        resource[speaker] = [{"path": f"{speaker}-enroll.wav"}]
    (root / "audio.json").write_text(json.dumps(resource), encoding="utf-8")
    return (
        "  audio:\n"
        "    type: wav\n"
        "    format: waveform\n"
        "    guaranteed: true\n"
        "    scope: speaker\n"
        "    policy: {type: fixed, key: spk_id, resource: audio.json}\n"
    )


def _build(
    root: Path,
    *,
    speakers: tuple[str, ...] = ("s1", "s2"),
    visual_items: dict[str, dict] | None = None,
    visual_guaranteed: bool = True,
    textual_values: dict[str, dict] | None = None,
    textual_guaranteed: bool = True,
    textual_selection: dict | None = None,
    textual_collate: dict | None = None,
    include_audio: bool = False,
    frames: int = FRAMES,
    **kwargs,
) -> TSEOfflineDataset:
    root.mkdir(parents=True, exist_ok=True)
    manifest = _write_manifest(root, speakers, frames=frames)
    sections = []
    if include_audio:
        sections.append(_write_audio_resource(root, speakers))
    if visual_items is not None:
        sections.append(
            _write_visual_resource(root, visual_items, guaranteed=visual_guaranteed)
        )
    if textual_values is not None:
        sections.append(
            _write_textual_resource(
                root,
                textual_values,
                guaranteed=textual_guaranteed,
                selection=textual_selection or {"type": "fixed"},
                collate=textual_collate,
            )
        )
    cues_yaml = root / "cues.yaml"
    cues_yaml.write_text("cues:\n" + "".join(sections), encoding="utf-8")
    enabled = []
    if include_audio:
        enabled.append("audio")
    if visual_items is not None:
        enabled.append("visual")
    if textual_values is not None:
        enabled.append("textual")
    return TSEOfflineDataset(
        samples_jsonl=manifest,
        cues_yaml=cues_yaml,
        manifest_format="jsonl",
        path_base="manifest_relative",
        target_selection="all",
        mix_variant="default",
        cue_schema="wesep_v0_1",
        enabled_cues=tuple(enabled),
        **kwargs,
    )


def _full_visual_items() -> dict[str, dict]:
    return {
        "s1": {"payload": _visual_payload(seed=1), "duration_sec": VISUAL_DURATION},
        "s2": {"payload": _visual_payload(seed=2), "duration_sec": VISUAL_DURATION},
    }


def _fixed_textual_values() -> dict[str, dict]:
    return {
        "s1": {"phn_label": _S1_WORDS, "kw_candidate": [0, 2], "words": ["a", "b", "c", "d"]},
        "s2": {"phn_label": _S2_WORDS, "kw_candidate": [1], "words": ["x", "y", "z"]},
    }


def _collate(dataset: TSEOfflineDataset, batch_size: int = 1):
    return next(
        iter(
            DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=partial(tse_collate_fn, cue_specs=dataset.cue_specs),
            )
        )
    )


def test_visual_and_textual_cues_publish_typed_specs_and_batch(tmp_path: Path):
    dataset = _build(
        tmp_path,
        visual_items=_full_visual_items(),
        textual_values=_fixed_textual_values(),
        textual_selection={"type": "fixed"},
    )

    assert dataset.cue_specs.visual is not None
    assert dataset.cue_specs.visual.representation == "features"
    assert dataset.cue_specs.visual.value_shape == (VISUAL_DIM, 1)
    assert dataset.cue_specs.textual is not None
    assert dataset.cue_specs.textual.representation == "phoneme_ids"
    assert dataset.cue_specs.textual.value_shape == (1,)
    assert dataset.cue_specs.speaker_waveform is None

    item = dataset[0]
    visual = item["targets"][0]["cues"].visual
    assert visual is not None and visual.present and visual.representation == "features"
    assert visual.value.shape == (VISUAL_DIM, VISUAL_FRAMES)
    assert visual.length == VISUAL_FRAMES
    torch.testing.assert_close(
        torch.as_tensor(visual.timebase),
        torch.arange(VISUAL_FRAMES, dtype=torch.float64) / VISUAL_FPS,
    )
    assert float(torch.as_tensor(visual.timebase)[-1]) < item["length"] / item["sr"]

    textual = item["targets"][0]["cues"].textual
    assert textual is not None and textual.present and textual.timebase is None
    assert textual.value.tolist() == [7, 45, 36, 32]
    assert textual.length == 4
    assert item["targets"][1]["cues"].textual.value.tolist() == [13, 14]
    provenance = item["provenance"]
    assert provenance["visual_cue"]["feature_dim"] == VISUAL_DIM
    assert provenance["textual_cue"]["reserved_ids"] == {"padding": 0, "bos": 71, "eos": 72}

    batch = _collate(dataset)
    assert batch.cues.visual is not None
    assert batch.cues.visual.value.shape == (2, VISUAL_DIM, VISUAL_FRAMES)
    assert batch.cues.visual.present.tolist() == [True, True]
    assert batch.cues.visual.lengths.tolist() == [VISUAL_FRAMES, VISUAL_FRAMES]
    assert batch.cues.visual.timebase is not None
    assert batch.cues.visual.timebase.shape == (2, VISUAL_FRAMES)
    assert batch.cues.textual is not None
    assert batch.cues.textual.value.shape == (2, 4)
    assert batch.cues.textual.value.dtype == torch.int64
    assert batch.cues.textual.lengths.tolist() == [4, 2]
    assert batch.cues.textual.timebase is None
    assert batch.speaker_ids == ("s1", "s2")


def test_speaker_visual_textual_combination_keeps_all_slots(tmp_path: Path):
    dataset = _build(
        tmp_path,
        visual_items=_full_visual_items(),
        textual_values=_fixed_textual_values(),
        include_audio=True,
    )
    batch = _collate(dataset)

    assert batch.cues.speaker_waveform is not None
    assert batch.cues.speaker_waveform.present.tolist() == [True, True]
    assert batch.cues.visual is not None and batch.cues.visual.present.tolist() == [True, True]
    assert batch.cues.textual is not None and batch.cues.textual.present.tolist() == [True, True]
    assert batch.cues.spatial is None


class _VisualTextualModel:
    sample_rate = SAMPLE_RATE
    supports_multichannel_input = False
    reference_channel = 0

    def __init__(self):
        self.IO_CONTRACT = ModelIOContract(
            tasks=frozenset({"tse"}),
            inputs=(
                TensorField("mix_wav", "waveform", ("batch", "time")),
                TensorField("visual", "embedding", ("batch", "feature", "frame")),
                TensorField("visual_present", "mask", ("batch",)),
                TensorField("visual_lengths", "label", ("batch",)),
                TensorField("visual_timebase", "label", ("batch", "frame")),
                TensorField("textual", "label", ("batch", "token")),
                TensorField("textual_present", "mask", ("batch",)),
                TensorField("textual_lengths", "label", ("batch",)),
            ),
            outputs=(TensorField("target_estimate", "waveform", ("batch", "time")),),
            forward_kind="estimate",
            frontend_owner="runtime",
        )


def test_visual_textual_cues_reach_model_io_boundary(tmp_path: Path):
    dataset = _build(
        tmp_path,
        visual_items=_full_visual_items(),
        textual_values=_fixed_textual_values(),
    )
    batch = _collate(dataset)
    inputs = build_tse_model_inputs(
        model=_VisualTextualModel(),
        stft=None,
        mixture=batch.mixture,
        cues=batch.cues,
        sample_rate=SAMPLE_RATE,
        lengths=batch.lengths,
        num_targets=batch.num_targets,
        reference_channel=0,
    )

    assert inputs["visual"].shape == (2, VISUAL_DIM, VISUAL_FRAMES)
    assert inputs["visual_lengths"].tolist() == [VISUAL_FRAMES, VISUAL_FRAMES]
    assert inputs["visual_present"].tolist() == [True, True]
    torch.testing.assert_close(
        inputs["visual_timebase"][0].to(torch.float64),
        torch.arange(VISUAL_FRAMES, dtype=torch.float64) / VISUAL_FPS,
    )
    assert inputs["textual"].shape == (2, 4)
    assert inputs["textual_lengths"].tolist() == [4, 2]


def test_optional_missing_visual_and_textual_keep_presence_false(tmp_path: Path):
    dataset = _build(
        tmp_path,
        visual_items={"s1": {"payload": _visual_payload(seed=3), "duration_sec": VISUAL_DURATION}},
        visual_guaranteed=False,
        textual_values={"s1": {"phn_label": _S1_WORDS, "kw_candidate": [1]}},
        textual_guaranteed=False,
    )
    item = dataset[0]
    visual = item["targets"][1]["cues"].visual
    assert visual is not None and visual.present is False and visual.length == 0
    assert visual.value is None and visual.timebase is None
    textual = item["targets"][1]["cues"].textual
    assert textual is not None and textual.present is False and textual.length == 0
    assert textual.value is None

    batch = _collate(dataset)
    assert batch.cues.visual is not None
    assert batch.cues.visual.present.tolist() == [True, False]
    assert batch.cues.visual.lengths.tolist() == [VISUAL_FRAMES, 0]
    assert torch.count_nonzero(batch.cues.visual.value[1]) == 0
    assert batch.cues.textual is not None
    assert batch.cues.textual.present.tolist() == [True, False]
    assert batch.cues.textual.lengths.tolist() == [2, 0]
    assert torch.count_nonzero(batch.cues.textual.value[1]) == 0


def test_visual_crop_uses_source_seconds_and_output_domain(tmp_path: Path):
    dataset = _build(
        tmp_path,
        visual_items=_full_visual_items(),
        max_duration=FRAMES // 2,
        seed=5,
    )
    record = dataset.records[0]
    window = dataset._window_plan(record, dataset._local_rng(0))
    item = dataset[0]
    cue = item["targets"][0]["cues"].visual

    assert cue is not None and cue.present
    offsets = torch.as_tensor(cue.timebase)
    assert cue.length == cue.value.shape[1] == offsets.numel()
    assert float(offsets[0]) == 0.0
    assert bool((offsets[1:] > offsets[:-1]).all().item())
    assert float(offsets[-1]) < item["length"] / item["sr"]
    assert 1 < cue.length < VISUAL_FRAMES
    assert cue.value.shape[1] < item["length"]
    assert item["provenance"]["source_start_sample"] == window["mix_start"]
    batch = _collate(dataset)
    assert batch.cues.visual is not None
    assert batch.cues.visual.value.shape[2] == cue.length
    assert batch.cues.visual.timebase is not None
    assert float(batch.cues.visual.timebase[0, 0]) == 0.0
    assert torch.count_nonzero(batch.cues.visual.value[0, :, cue.length :]) == 0


def test_visual_cue_timebase_stays_in_output_domain_after_resample(tmp_path: Path):
    dataset = _build(
        tmp_path,
        visual_items=_full_visual_items(),
        target_sr=8000,
        seed=3,
    )
    item = dataset[0]
    assert item["sr"] == 8000
    cue = item["targets"][0]["cues"].visual

    assert cue is not None and cue.present
    offsets = torch.as_tensor(cue.timebase)
    assert cue.length == VISUAL_FRAMES
    assert float(offsets[0]) == 0.0
    assert float(offsets[-1]) < item["length"] / item["sr"]
    batch = _collate(dataset)
    assert batch.cues.visual is not None
    assert batch.cues.visual.value.shape == (2, VISUAL_DIM, VISUAL_FRAMES)
    assert batch.cues.visual.timebase is not None
    assert float(batch.cues.visual.timebase[0, -1]) < item["length"] / item["sr"]


def test_visual_cue_is_not_resized_to_audio_sample_count(tmp_path: Path):
    frames = 20
    dataset = _build(
        tmp_path,
        speakers=("s1",),
        visual_items={
            "s1": {
                "payload": _visual_payload(frames=frames, seed=9),
                "duration_sec": VISUAL_DURATION,
            }
        },
    )
    item = dataset[0]
    cue = item["targets"][0]["cues"].visual

    assert cue is not None and cue.present
    assert cue.length == frames
    assert cue.value.shape == (VISUAL_DIM, frames)
    torch.testing.assert_close(
        torch.as_tensor(cue.timebase),
        torch.arange(frames, dtype=torch.float64) / (frames / VISUAL_DURATION),
    )
    assert cue.length != item["length"]


def test_textual_random_selection_is_seeded_and_reproducible(tmp_path: Path):
    dataset = _build(
        tmp_path,
        textual_values={
            "s1": {"phn_label": _S1_WORDS},
            "s2": {"phn_label": _S2_WORDS},
        },
        textual_selection={"type": "random", "word_range": [2, 3]},
        seed=7,
    )
    first = dataset[0]["targets"][0]["cues"].textual
    repeat = _build(
        tmp_path / "repeat",
        speakers=("s1",),
        textual_values={"s1": {"phn_label": _S1_WORDS}},
        textual_selection={"type": "random", "word_range": [2, 3]},
        seed=7,
    )[0]["targets"][0]["cues"].textual

    allowed = {
        tuple(token for word in _S1_WORDS[start : start + count] for token in word)
        for count in (2, 3)
        for start in range(0, len(_S1_WORDS) - count + 1)
    }
    assert first is not None and first.present
    assert repeat is not None and first.value.tolist() == repeat.value.tolist()
    assert tuple(first.value.tolist()) in allowed
    assert first.length == first.value.numel()
    dataset.set_epoch(1)
    assert tuple(dataset[0]["targets"][0]["cues"].textual.value.tolist()) in allowed


def test_visual_and_textual_fail_fast_on_missing_or_invalid_resources(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="cannot read visual cue npy"):
        _build(
            tmp_path / "absent",
            speakers=("s1",),
            visual_items={"s1": {"payload": None, "name": "missing.npy"}},
        )

    with pytest.raises(ValueError, match="must declare duration_sec or fps"):
        _build(
            tmp_path / "no-clock",
            speakers=("s1",),
            visual_items={"s1": {"payload": _visual_payload(seed=4)}},
        )

    with pytest.raises(ValueError, match="disagree"):
        _build(
            tmp_path / "bad-clock",
            speakers=("s1",),
            visual_items={
                "s1": {
                    "payload": _visual_payload(seed=4),
                    "duration_sec": VISUAL_DURATION,
                    "fps": 25.0,
                }
            },
        )

    with pytest.raises(ValueError, match="edge padding is not applied"):
        _build(
            tmp_path / "short-video",
            speakers=("s1",),
            visual_items={"s1": {"payload": _visual_payload(frames=10, seed=4), "duration_sec": 0.1}},
        )

    with pytest.raises(ValueError, match="visual feature dimensions must be fixed"):
        _build(
            tmp_path / "bad-dim",
            visual_items={
                "s1": {"payload": _visual_payload(dim=VISUAL_DIM, seed=4), "duration_sec": VISUAL_DURATION},
                "s2": {"payload": _visual_payload(dim=VISUAL_DIM + 1, seed=5), "duration_sec": VISUAL_DURATION},
            },
        )

    with pytest.raises(ValueError, match="reserved token"):
        _build(
            tmp_path / "reserved-token",
            speakers=("s1",),
            textual_values={"s1": {"phn_label": [[7, 71]], "kw_candidate": [0]}},
        )

    with pytest.raises(ValueError, match="zero-fills padding"):
        _build(
            tmp_path / "bad-fill",
            speakers=("s1",),
            textual_values={"s1": {"phn_label": _S1_WORDS, "kw_candidate": [0]}},
            textual_collate={"fill_value": -1},
        )


def test_raw_video_visual_cue_is_rejected_after_strict_index_import(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = _write_manifest(tmp_path, ("s1",))
    (tmp_path / "visual.json").write_text(
        json.dumps({"mix::s1": [{"path": "s1.mp4"}]}), encoding="utf-8"
    )
    cues_yaml = tmp_path / "cues.yaml"
    cues_yaml.write_text(
        "cues:\n"
        "  visual:\n"
        "    type: mp4\n"
        "    format: raw_video\n"
        "    guaranteed: true\n"
        "    scope: speaker\n"
        "    policy: {type: fixed, key: mix_spk_id, resource: visual.json}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="raw video visual cue requires an optional video decoder"):
        TSEOfflineDataset(
            samples_jsonl=manifest,
            cues_yaml=cues_yaml,
            manifest_format="jsonl",
            path_base="manifest_relative",
            target_selection="all",
            mix_variant="default",
            cue_schema="wesep_v0_1",
            enabled_cues=("visual",),
        )


def test_legacy_realtse_raw_behaviour_is_unchanged(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = _write_manifest(tmp_path, ("s1",))
    _write_wav(tmp_path / "s1-enroll.wav", 0.7, 4000)
    (tmp_path / "audio.json").write_text(
        json.dumps({"s1": [{"path": "s1-enroll.wav"}]}), encoding="utf-8"
    )
    cues_yaml = tmp_path / "cues.yaml"
    cues_yaml.write_text(
        "cues:\n"
        "  audio:\n"
        "    type: raw\n"
        "    scope: speaker\n"
        "    guaranteed: true\n"
        "    policy: {type: fixed, key: spk_id, resource: audio.json}\n",
        encoding="utf-8",
    )
    dataset = TSEOfflineDataset(
        samples_jsonl=manifest,
        cues_yaml=cues_yaml,
        manifest_format="jsonl",
        path_base="manifest_relative",
        target_selection="all",
        mix_variant="default",
        cue_schema="realtse_raw",
        enabled_cues=("audio",),
    )
    cue = dataset[0]["targets"][0]["cues"].speaker_waveform
    assert cue is not None and cue.present and cue.length == 4000
    assert dataset.cue_specs.speaker_waveform is not None
    assert dataset.cue_specs.visual is None
    assert dataset.cue_specs.textual is None
