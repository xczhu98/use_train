"""独立 speaker metadata resolver 的行为与安全契约测试。"""

import dataclasses
import hashlib
import json
import multiprocessing
import os
import threading
from pathlib import Path

import pytest

from audio_filter_pipeline.config import SpeakerMetadataSpec
from audio_filter_pipeline.utils.speaker_metadata import SpeakerMetadataResolver


pytestmark = pytest.mark.unit


def _resolver(tmp_path, spec, *, mapping_base_dir=None):
    input_root = tmp_path / "input"
    input_root.mkdir(exist_ok=True)
    return SpeakerMetadataResolver(
        spec,
        input_root=input_root,
        mapping_base_dir=mapping_base_dir,
    )


def _write_mapping(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _construct_mapping_resolver_in_child(root, mapping_base_dir, result_queue):
    spec = SpeakerMetadataSpec(
        source="mapping_file", mapping_path="speakers.jsonl", required=True,
    )
    try:
        SpeakerMetadataResolver(
            spec,
            input_root=os.path.join(root, "input"),
            mapping_base_dir=mapping_base_dir,
        )
    except Exception as exc:
        result_queue.put((type(exc).__name__, str(exc)))
    else:
        result_queue.put((None, None))


def test_none_required_fails_closed_at_resolver_construction(tmp_path):
    spec = SpeakerMetadataSpec(source="none", required=True)

    with pytest.raises(ValueError, match="source 'none'.*required=False"):
        _resolver(tmp_path, spec)


def test_none_never_infers_speaker_and_exports_only_effective_fields(tmp_path):
    resolver = _resolver(
        tmp_path,
        SpeakerMetadataSpec(source="none", required=False),
    )

    assert resolver.resolve("speaker/session/utt.wav") is None
    assert resolver.config_payload() == {"required": False, "source": "none"}


def test_path_component_counts_parent_components_from_left(tmp_path):
    resolver = _resolver(
        tmp_path,
        SpeakerMetadataSpec(
            source="path_component", path_component=1, required=True,
        ),
    )

    assert resolver.resolve("corpus/speaker/session/utt.wav") == "speaker"


@pytest.mark.parametrize("source_path", ["utt.wav", "speaker/utt.wav"])
def test_path_component_missing_is_optional_or_required(tmp_path, source_path):
    optional = _resolver(
        tmp_path,
        SpeakerMetadataSpec(
            source="path_component", path_component=1, required=False,
        ),
    )
    required = _resolver(
        tmp_path,
        SpeakerMetadataSpec(
            source="path_component", path_component=1, required=True,
        ),
    )

    assert optional.resolve(source_path) is None
    with pytest.raises(ValueError, match="required speaker label.*missing"):
        required.resolve(source_path)


def test_regex_searches_complete_path_and_uses_configured_named_group(tmp_path):
    resolver = _resolver(
        tmp_path,
        SpeakerMetadataSpec(
            source="regex",
            regex=r"session-(?P<voice>spk-[^/]+)/",
            regex_group="voice",
            required=True,
        ),
    )

    assert resolver.resolve("prefix/session-spk-07/utt.wav") == "spk-07"


@pytest.mark.parametrize(
    "pattern,path",
    [
        (r"speaker-(?P<speaker_id>[^/]+)/", "other/utt.wav"),
        (r"(?P<speaker_id>)/utt[.]wav$", "folder/utt.wav"),
    ],
)
def test_regex_unmatched_or_empty_group_obeys_required_flag(tmp_path, pattern, path):
    optional = _resolver(
        tmp_path,
        SpeakerMetadataSpec(
            source="regex", regex=pattern, required=False,
        ),
    )
    required = _resolver(
        tmp_path,
        SpeakerMetadataSpec(
            source="regex", regex=pattern, required=True,
        ),
    )

    assert optional.resolve(path) is None
    with pytest.raises(ValueError, match="required speaker label.*missing"):
        required.resolve(path)


def test_relative_mapping_is_anchored_to_explicit_base_and_cwd_independent(
    tmp_path, monkeypatch,
):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    mapping = config_dir / "speakers.jsonl"
    _write_mapping(
        mapping,
        [{"source_relative_path": "audio/utt.wav", "speaker_id": "spk-1"}],
    )
    resolver = _resolver(
        tmp_path,
        SpeakerMetadataSpec(
            source="mapping_file", mapping_path="speakers.jsonl", required=True,
        ),
        mapping_base_dir=config_dir,
    )
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    monkeypatch.chdir(other_dir)

    assert resolver.resolve("audio/utt.wav") == "spk-1"
    assert resolver.config_payload() == {
        "mapping_path": str(mapping.resolve()),
        "mapping_sha256": hashlib.sha256(mapping.read_bytes()).hexdigest(),
        "required": True,
        "source": "mapping_file",
    }


def test_mapping_snapshot_and_digest_change_only_for_new_resolver(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    mapping = config_dir / "speakers.jsonl"
    first_bytes = b'{"source_relative_path":"utt.wav","speaker_id":"first"}\n'
    second_bytes = b'{"source_relative_path":"utt.wav","speaker_id":"second"}\n'
    mapping.write_bytes(first_bytes)
    spec = SpeakerMetadataSpec(
        source="mapping_file", mapping_path="speakers.jsonl", required=True,
    )
    first = _resolver(tmp_path, spec, mapping_base_dir=config_dir)

    mapping.write_bytes(second_bytes)
    second = _resolver(tmp_path, spec, mapping_base_dir=config_dir)

    assert first.resolve("utt.wav") == "first"
    assert second.resolve("utt.wav") == "second"
    assert first.config_payload()["mapping_sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert second.config_payload()["mapping_sha256"] == hashlib.sha256(second_bytes).hexdigest()


def test_mapping_rejects_same_inode_write_during_snapshot_read(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.speaker_metadata as speaker_metadata

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    mapping = config_dir / "speakers.jsonl"
    initial = b'{"source_relative_path":"utt.wav","speaker_id":"first"}\n'
    changed = b'{"source_relative_path":"utt.wav","speaker_id":"rewritten"}\n'
    mapping.write_bytes(initial)
    original_identity = (mapping.stat().st_dev, mapping.stat().st_ino)
    read_complete = threading.Event()
    write_complete = threading.Event()
    original_read = speaker_metadata._read_bounded_snapshot

    def rewrite_mapping():
        assert read_complete.wait(timeout=1.0)
        with mapping.open("r+b") as stream:
            stream.write(changed)
            stream.flush()
            os.fsync(stream.fileno())
        changed_stat = mapping.stat()
        os.utime(
            mapping,
            ns=(changed_stat.st_atime_ns, changed_stat.st_mtime_ns + 1_000_000_000),
        )
        write_complete.set()

    writer = threading.Thread(target=rewrite_mapping)
    writer.start()

    def read_then_allow_rewrite(fd):
        snapshot = original_read(fd)
        read_complete.set()
        assert write_complete.wait(timeout=1.0)
        return snapshot

    monkeypatch.setattr(
        speaker_metadata, "_read_bounded_snapshot", read_then_allow_rewrite,
    )
    spec = SpeakerMetadataSpec(
        source="mapping_file", mapping_path="speakers.jsonl", required=True,
    )

    with pytest.raises(ValueError, match="changed while reading"):
        _resolver(tmp_path, spec, mapping_base_dir=config_dir)

    writer.join(timeout=1.0)
    assert not writer.is_alive()
    assert (mapping.stat().st_dev, mapping.stat().st_ino) == original_identity
    assert mapping.read_bytes() == changed


@pytest.mark.parametrize(
    "raw,match",
    [
        (b"\n", "blank"),
        (b"not-json\n", "JSON"),
        (b"[]\n", "object"),
        (
            b'{"source_relative_path":"utt.wav","speaker_id":"spk","extra":1}\n',
            "fields",
        ),
        (b'{"source_relative_path":"utt.wav"}\n', "fields"),
        (
            b'{"source_relative_path":"utt.wav","speaker_id":"a"}\n'
            b'{"source_relative_path":"utt.wav","speaker_id":"b"}\n',
            "duplicate source_relative_path",
        ),
        (
            b'{"source_relative_path":"utt.wav","speaker_id":"a",'
            b'"speaker_id":"b"}\n',
            "duplicate JSON field",
        ),
        (b'\xff\n', "UTF-8"),
    ],
)
def test_mapping_rejects_malformed_or_non_strict_jsonl(tmp_path, raw, match):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "speakers.jsonl").write_bytes(raw)
    spec = SpeakerMetadataSpec(
        source="mapping_file", mapping_path="speakers.jsonl", required=True,
    )

    with pytest.raises(ValueError, match=match):
        _resolver(tmp_path, spec, mapping_base_dir=config_dir)


@pytest.mark.parametrize("base", [None, "", "   "])
def test_mapping_requires_nonempty_explicit_base_directory(tmp_path, base):
    spec = SpeakerMetadataSpec(
        source="mapping_file", mapping_path="speakers.jsonl", required=True,
    )

    with pytest.raises(ValueError, match="mapping_base_dir"):
        _resolver(tmp_path, spec, mapping_base_dir=base)


def test_mapping_fifo_is_rejected_without_blocking(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    os.mkfifo(config_dir / "speakers.jsonl")
    (tmp_path / "input").mkdir()
    context = multiprocessing.get_context("fork")
    result_queue = context.Queue()
    process = context.Process(
        target=_construct_mapping_resolver_in_child,
        args=(str(tmp_path), str(config_dir), result_queue),
    )
    process.start()
    process.join(timeout=1.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        pytest.fail("mapping FIFO blocked resolver construction")

    assert process.exitcode == 0
    error_type, message = result_queue.get(timeout=0.5)
    assert error_type == "ValueError"
    assert "regular file" in message


def test_mapping_missing_file_fails_closed(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    spec = SpeakerMetadataSpec(
        source="mapping_file", mapping_path="speakers.jsonl", required=True,
    )

    with pytest.raises(ValueError, match="cannot open speaker mapping file"):
        _resolver(tmp_path, spec, mapping_base_dir=config_dir)


def test_mapping_directory_fails_closed(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "speakers.jsonl").mkdir()
    spec = SpeakerMetadataSpec(
        source="mapping_file", mapping_path="speakers.jsonl", required=True,
    )

    with pytest.raises(ValueError, match="regular file"):
        _resolver(tmp_path, spec, mapping_base_dir=config_dir)


@pytest.mark.parametrize("absolute", [False, True])
def test_mapping_path_cannot_escape_canonical_base(tmp_path, absolute):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("", encoding="utf-8")
    mapping_path = str(outside) if absolute else "../outside.jsonl"
    spec = SpeakerMetadataSpec(
        source="mapping_file", mapping_path=mapping_path, required=True,
    )

    with pytest.raises(ValueError, match="escape.*mapping_base_dir"):
        _resolver(tmp_path, spec, mapping_base_dir=config_dir)


def test_mapping_rejects_final_symlink(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    target = config_dir / "target.jsonl"
    target.write_text("", encoding="utf-8")
    link = config_dir / "speakers.jsonl"
    link.symlink_to(target.name)
    spec = SpeakerMetadataSpec(
        source="mapping_file", mapping_path=link.name, required=True,
    )

    with pytest.raises(ValueError, match="symlink"):
        _resolver(tmp_path, spec, mapping_base_dir=config_dir)


def test_mapping_rejects_parent_symlink_swap_before_open(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.speaker_metadata as speaker_metadata

    config_dir = tmp_path / "config"
    mapping_dir = config_dir / "mapping"
    mapping_dir.mkdir(parents=True)
    mapping = mapping_dir / "speakers.jsonl"
    mapping.write_text("", encoding="utf-8")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / mapping.name).write_text(
        '{"source_relative_path":"utt.wav","speaker_id":"outside"}\n',
        encoding="utf-8",
    )
    original_open = speaker_metadata.os.open
    swapped = False

    def swap_parent_then_open(path, flags):
        nonlocal swapped
        if not swapped and path == str(mapping):
            swapped = True
            mapping_dir.rename(config_dir / "original")
            mapping_dir.symlink_to(outside_dir, target_is_directory=True)
        return original_open(path, flags)

    monkeypatch.setattr(speaker_metadata.os, "open", swap_parent_then_open)
    spec = SpeakerMetadataSpec(
        source="mapping_file", mapping_path="mapping/speakers.jsonl", required=True,
    )

    with pytest.raises(ValueError, match="escape.*mapping_base_dir"):
        _resolver(tmp_path, spec, mapping_base_dir=config_dir)


def test_mapping_missing_key_obeys_required_flag(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "speakers.jsonl").write_text("", encoding="utf-8")
    optional = _resolver(
        tmp_path,
        SpeakerMetadataSpec(
            source="mapping_file", mapping_path="speakers.jsonl", required=False,
        ),
        mapping_base_dir=config_dir,
    )
    required = _resolver(
        tmp_path,
        SpeakerMetadataSpec(
            source="mapping_file", mapping_path="speakers.jsonl", required=True,
        ),
        mapping_base_dir=config_dir,
    )

    assert optional.resolve("missing.wav") is None
    with pytest.raises(ValueError, match="required speaker label.*missing"):
        required.resolve("missing.wav")


@pytest.mark.parametrize(
    "source_path",
    [
        "",
        "/absolute.wav",
        r"speaker\utt.wav",
        "./utt.wav",
        "speaker/../utt.wav",
        "speaker//utt.wav",
        "speaker/./utt.wav",
        "speaker/",
    ],
)
def test_resolve_rejects_non_normalized_source_relative_paths(tmp_path, source_path):
    resolver = _resolver(
        tmp_path,
        SpeakerMetadataSpec(source="none", required=False),
    )

    with pytest.raises(ValueError, match="source_relative_path"):
        resolver.resolve(source_path)


def test_resolve_rejects_nul_in_source_relative_path(tmp_path):
    resolver = _resolver(
        tmp_path,
        SpeakerMetadataSpec(source="none", required=False),
    )

    with pytest.raises(ValueError, match="source_relative_path"):
        resolver.resolve("speaker\x00/utt.wav")


def test_mapping_rejects_nul_in_source_relative_path_key(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_mapping(
        config_dir / "speakers.jsonl",
        [{"source_relative_path": "speaker\x00/utt.wav", "speaker_id": "spk"}],
    )
    spec = SpeakerMetadataSpec(
        source="mapping_file", mapping_path="speakers.jsonl", required=True,
    )

    with pytest.raises(ValueError, match="source_relative_path"):
        _resolver(tmp_path, spec, mapping_base_dir=config_dir)


def test_public_source_relative_path_validator_is_reusable():
    from audio_filter_pipeline.utils.speaker_metadata import (
        validate_source_relative_path,
    )

    assert validate_source_relative_path("speaker/utt.wav") == "speaker/utt.wav"
    with pytest.raises(ValueError, match="source_relative_path"):
        validate_source_relative_path("speaker\x00/utt.wav")


@pytest.mark.parametrize(
    "row,match",
    [
        ({"source_relative_path": "../utt.wav", "speaker_id": "spk"}, "source_relative_path"),
        ({"source_relative_path": "utt.wav", "speaker_id": ""}, "speaker_id"),
        ({"source_relative_path": "utt.wav", "speaker_id": " spk"}, "speaker_id"),
        ({"source_relative_path": "utt.wav", "speaker_id": "spk "}, "speaker_id"),
        ({"source_relative_path": "utt.wav", "speaker_id": 7}, "speaker_id"),
    ],
)
def test_mapping_rejects_invalid_paths_and_speaker_ids(tmp_path, row, match):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_mapping(config_dir / "speakers.jsonl", [row])
    spec = SpeakerMetadataSpec(
        source="mapping_file", mapping_path="speakers.jsonl", required=True,
    )

    with pytest.raises(ValueError, match=match):
        _resolver(tmp_path, spec, mapping_base_dir=config_dir)


def test_regex_rejects_whitespace_padded_resolved_speaker(tmp_path):
    resolver = _resolver(
        tmp_path,
        SpeakerMetadataSpec(
            source="regex",
            regex=r"(?P<speaker_id> [^/]+)/",
            required=True,
        ),
    )

    with pytest.raises(ValueError, match="speaker_id"):
        resolver.resolve(" spk/utt.wav")


def test_resolver_is_immutable_and_payload_is_canonical_json_safe(tmp_path):
    resolver = _resolver(
        tmp_path,
        SpeakerMetadataSpec(
            source="regex",
            regex=r"(?P<voice>[^/]+)/",
            regex_group="voice",
            required=False,
        ),
    )
    expected = {
        "regex": r"(?P<voice>[^/]+)/",
        "regex_group": "voice",
        "required": False,
        "source": "regex",
    }

    with pytest.raises(dataclasses.FrozenInstanceError):
        resolver.input_root = "/changed"
    first = resolver.config_payload()
    first["source"] = "none"

    assert resolver.config_payload() == expected
    assert json.loads(json.dumps(resolver.config_payload(), sort_keys=True)) == expected
