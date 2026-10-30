"""WeSep 资源索引只读导入适配器测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tasks.tse.data.wesep_adapter import (
    MANIFEST_FORMAT_JSONL,
    MANIFEST_FORMAT_PATH_LIST,
    PATH_BASE_MANIFEST_RELATIVE,
    PATH_BASE_RECIPE_ROOT,
    load_wesep_records,
)

pytestmark = pytest.mark.unit


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _sample(key: str = "mix", speakers: tuple[str, ...] = ("s1", "s2")) -> dict:
    return {
        "key": key,
        "spk": list(speakers),
        "mix": {"default": ["audio/mix-ch0.wav", "audio/mix-ch1.wav"]},
        "src": {
            speaker: [
                f"audio/{speaker}-target0.wav",
                f"audio/{speaker}-target1.wav",
            ]
            for speaker in speakers
        },
    }


def _load(
    manifest: Path,
    *,
    manifest_format: str = MANIFEST_FORMAT_JSONL,
    path_base: str = PATH_BASE_MANIFEST_RELATIVE,
    target_selection="all",
    recipe_root: Path | None = None,
    cues_yaml: Path | None = None,
    enabled_cues=(),
):
    return load_wesep_records(
        manifest,
        manifest_format=manifest_format,
        path_base=path_base,
        mix_variant="default",
        target_selection=target_selection,
        recipe_root=recipe_root,
        cues_yaml=cues_yaml,
        enabled_cues=enabled_cues,
    )


def test_jsonl_raw_list_preserves_all_targets_and_ordered_paths(tmp_path: Path):
    raw_list = tmp_path / "raw.list"
    _write_jsonl(raw_list, [_sample()])

    records = _load(raw_list)

    assert len(records) == 1
    record = records[0]
    assert record.key == "mix"
    assert record.manifest_path == str(raw_list)
    assert record.mixture_paths == (
        str(tmp_path / "audio/mix-ch0.wav"),
        str(tmp_path / "audio/mix-ch1.wav"),
    )
    assert [target.speaker_id for target in record.targets] == ["s1", "s2"]
    assert record.targets[0].source_paths == (
        str(tmp_path / "audio/s1-target0.wav"),
        str(tmp_path / "audio/s1-target1.wav"),
    )
    assert record.targets[1].source_paths == (
        str(tmp_path / "audio/s2-target0.wav"),
        str(tmp_path / "audio/s2-target1.wav"),
    )

    first = _load(raw_list, target_selection="first")[0]
    assert [target.speaker_id for target in first.targets] == ["s1"]
    selected = _load(raw_list, target_selection={"mix": ["s2"]})[0]
    assert [target.speaker_id for target in selected.targets] == ["s2"]


def test_path_list_and_path_base_are_explicit_without_exists_fallback(tmp_path: Path):
    recipe_root = tmp_path / "recipe"
    samples = recipe_root / "manifests/samples.jsonl"
    _write_jsonl(samples, [_sample(speakers=("s1",))])
    path_list = tmp_path / "paths.list"
    path_list.write_text("recipe/manifests/samples.jsonl\n", encoding="utf-8")

    manifest_relative = _load(
        path_list,
        manifest_format=MANIFEST_FORMAT_PATH_LIST,
    )[0]
    assert manifest_relative.mixture_paths[0] == str(
        recipe_root / "manifests/audio/mix-ch0.wav"
    )
    assert not Path(manifest_relative.mixture_paths[0]).exists()

    path_list.write_text("manifests/samples.jsonl\n", encoding="utf-8")
    recipe_relative = _load(
        path_list,
        manifest_format=MANIFEST_FORMAT_PATH_LIST,
        path_base=PATH_BASE_RECIPE_ROOT,
        recipe_root=recipe_root,
    )[0]
    assert recipe_relative.mixture_paths[0] == str(
        recipe_root / "audio/mix-ch0.wav"
    )
    assert not Path(recipe_relative.mixture_paths[0]).exists()


def test_all_cue_indexes_preserve_representation_metadata(tmp_path: Path):
    manifest = tmp_path / "samples.jsonl"
    _write_jsonl(manifest, [_sample()])
    cue_dir = tmp_path / "cues"
    _write_json(
        cue_dir / "audio.json",
        {
            "s1": [
                {"utt_id": "a", "path": "payload/a.wav"},
                {"utt_id": "b", "path": "payload/b.wav"},
            ],
            "s2": [{"utt_id": "c", "path": "payload/c.wav"}],
        },
    )
    _write_json(
        cue_dir / "spatial.json",
        {
            "mix::s1": {"path": "payload/s1.npy", "duration_sec": 2.5},
            "mix::s2": {"path": "payload/s2.npy"},
        },
    )
    _write_json(
        cue_dir / "visual.json",
        {
            "mix::s1": [{"path": "payload/s1.npy", "duration_sec": 2.5}],
            "mix::s2": [{"path": "payload/s2.npy"}],
        },
    )
    _write_json(
        cue_dir / "textual.json",
        {
            "mix::s1": {"phn_label": [[1, 2], [3]], "kw_candidate": [0]},
            "mix::s2": {"phn_label": [[4], [5, 6]], "kw_candidate": [1]},
        },
    )
    cues_yaml = tmp_path / "cues.yaml"
    cues_yaml.write_text(
        """cues:
  audio:
    type: wav
    format: waveform
    scope: speaker
    guaranteed: true
    policy:
      type: random
      key: spk_id
      resource: cues/audio.json
  spatial:
    type: npy
    format: fields
    scope: speaker
    guaranteed: true
    fields: [azimuth, elevation]
    field_defaults: {elevation: 0.0}
    policy:
      type: fixed
      key: mix_spk_id
      resource: cues/spatial.json
  visual:
    type: npy
    format: muse_frontend
    scope: speaker
    guaranteed: true
    policy:
      type: fixed
      key: mix_spk_id
      resource: cues/visual.json
  textual:
    type: json
    format: dae_keyword_phoneme
    scope: speaker
    guaranteed: true
    selection:
      type: fixed
    collate:
      fill_value: 0
      emit_present: false
    policy:
      type: fixed
      key: mix_spk_id
      resource: cues/textual.json
""",
        encoding="utf-8",
    )

    record = _load(
        manifest,
        cues_yaml=cues_yaml,
        enabled_cues={
            "audio": True,
            "spatial": True,
            "visual": True,
            "textual": True,
        },
    )[0]
    target = record.targets[0]

    audio = target.cue("audio")
    assert audio.present is True
    assert audio.spec.policy_type == "random"
    assert audio.spec.cue_type == "wav"
    assert audio.spec.cue_format == "waveform"
    assert [item["utt_id"] for item in audio.items] == ["a", "b"]
    assert audio.items[0]["path"] == str(cue_dir / "payload/a.wav")

    spatial = target.cue("spatial")
    assert spatial.spec.format_metadata["fields"] == ("azimuth", "elevation")
    assert spatial.spec.format_metadata["field_defaults"] == {"elevation": 0.0}
    assert spatial.items[0]["duration_sec"] == 2.5

    visual = target.cue("visual")
    assert visual.spec.cue_format == "muse_frontend"
    assert visual.items[0]["path"] == str(cue_dir / "payload/s1.npy")

    textual = target.cue("textual")
    assert textual.spec.format_metadata["selection"] == {"type": "fixed"}
    assert textual.spec.format_metadata["collate"] == {
        "fill_value": 0,
        "emit_present": False,
    }
    assert textual.items[0]["phn_label"] == ((1, 2), (3,))


def test_audio_embedding_format_is_indexed_without_loading_payload(tmp_path: Path):
    manifest = tmp_path / "samples.jsonl"
    _write_jsonl(manifest, [_sample(speakers=("s1",))])
    _write_json(
        tmp_path / "embedding.json",
        {"s1": [{"path": "embeddings/s1.npy", "encoder": "ecapa"}]},
    )
    cues_yaml = tmp_path / "cues.yaml"
    cues_yaml.write_text(
        """cues:
  audio:
    type: npy
    format: spk_embedding
    guaranteed: true
    scope: speaker
    policy:
      type: fixed
      key: spk_id
      resource: embedding.json
""",
        encoding="utf-8",
    )

    cue = _load(
        manifest,
        cues_yaml=cues_yaml,
        enabled_cues=("audio",),
    )[0].targets[0].cue("audio")

    assert cue.spec.cue_format == "spk_embedding"
    assert cue.items[0]["path"] == str(tmp_path / "embeddings/s1.npy")
    assert not Path(cue.items[0]["path"]).exists()


def test_optional_missing_cue_is_distinct_from_guaranteed_missing(tmp_path: Path):
    manifest = tmp_path / "samples.jsonl"
    _write_jsonl(manifest, [_sample()])
    _write_json(
        tmp_path / "audio.json",
        {"s1": [{"path": "s1.wav"}]},
    )
    cues_yaml = tmp_path / "cues.yaml"
    template = """cues:
  audio:
    type: wav
    format: waveform
    guaranteed: {guaranteed}
    scope: speaker
    policy:
      type: fixed
      key: spk_id
      resource: audio.json
"""
    cues_yaml.write_text(template.format(guaranteed="false"), encoding="utf-8")

    records = _load(
        manifest,
        cues_yaml=cues_yaml,
        enabled_cues=("audio",),
    )
    missing = records[0].targets[1].cue("audio")
    assert missing.present is False
    assert missing.items == ()
    assert missing.spec.guaranteed is False

    cues_yaml.write_text(template.format(guaranteed="true"), encoding="utf-8")
    with pytest.raises(KeyError, match="guaranteed audio cue is missing"):
        _load(
            manifest,
            cues_yaml=cues_yaml,
            enabled_cues=("audio",),
        )


@pytest.mark.parametrize(
    ("cue_body", "message"),
    [
        (
            "type: raw\n    format: waveform\n    scope: speaker",
            "unsupported audio cue type/format",
        ),
        (
            "type: wav\n    format: waveform\n    scope: mixture",
            "unsupported audio cue scope",
        ),
    ],
)
def test_unknown_cue_format_or_scope_fails_fast(
    tmp_path: Path, cue_body: str, message: str
):
    manifest = tmp_path / "samples.jsonl"
    _write_jsonl(manifest, [_sample(speakers=("s1",))])
    _write_json(tmp_path / "audio.json", {"s1": [{"path": "s1.wav"}]})
    cues_yaml = tmp_path / "cues.yaml"
    cues_yaml.write_text(
        "cues:\n"
        "  audio:\n"
        f"    {cue_body}\n"
        "    guaranteed: true\n"
        "    policy:\n"
        "      type: fixed\n"
        "      key: spk_id\n"
        "      resource: audio.json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        _load(
            manifest,
            cues_yaml=cues_yaml,
            enabled_cues=("audio",),
        )


def test_unknown_enabled_cue_and_mismatched_manifest_format_fail_fast(
    tmp_path: Path,
):
    raw_list = tmp_path / "raw.list"
    _write_jsonl(raw_list, [_sample(speakers=("s1",))])

    with pytest.raises(ValueError, match="unknown enabled cues"):
        _load(raw_list, enabled_cues=("oracle",))
    with pytest.raises(ValueError, match="contains JSON content"):
        _load(raw_list, manifest_format=MANIFEST_FORMAT_PATH_LIST)
    with pytest.raises(ValueError, match="recipe_root is required"):
        _load(raw_list, path_base=PATH_BASE_RECIPE_ROOT)


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ({"path": "payload/s1.npy", "fps": "fast"}, "fps must be numeric"),
        ({"path": "payload/s1.npy", "fps": 0}, "fps must be positive"),
        ({"path": "payload/s1.npy", "duration_sec": -1.0}, "duration_sec must be positive"),
    ],
)
def test_visual_item_clock_metadata_is_validated_at_index_boundary(
    tmp_path: Path, item: dict, message: str
):
    manifest = tmp_path / "samples.jsonl"
    _write_jsonl(manifest, [_sample(speakers=("s1",))])
    _write_json(tmp_path / "visual.json", {"mix::s1": [item]})
    cues_yaml = tmp_path / "cues.yaml"
    cues_yaml.write_text(
        """cues:
  visual:
    type: npy
    format: muse_frontend
    guaranteed: true
    scope: speaker
    policy: {type: fixed, key: mix_spk_id, resource: visual.json}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        _load(manifest, cues_yaml=cues_yaml, enabled_cues=("visual",))


def test_visual_item_clock_metadata_is_preserved(tmp_path: Path):
    manifest = tmp_path / "samples.jsonl"
    _write_jsonl(manifest, [_sample(speakers=("s1",))])
    _write_json(
        tmp_path / "visual.json",
        {"mix::s1": [{"path": "payload/s1.npy", "duration_sec": 2.0, "fps": 25.0}]},
    )
    cues_yaml = tmp_path / "cues.yaml"
    cues_yaml.write_text(
        """cues:
  visual:
    type: npy
    format: muse_frontend
    guaranteed: true
    scope: speaker
    policy: {type: fixed, key: mix_spk_id, resource: visual.json}
""",
        encoding="utf-8",
    )

    item = _load(
        manifest, cues_yaml=cues_yaml, enabled_cues=("visual",)
    )[0].targets[0].cue("visual").items[0]

    assert item["path"] == str(tmp_path / "payload/s1.npy")
    assert item["duration_sec"] == 2.0
    assert item["fps"] == 25.0
