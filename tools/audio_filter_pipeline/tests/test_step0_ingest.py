"""Artifact-backed Step0 ingest 的行为与安全契约测试。"""

import errno
import gc
import hashlib
import json
import logging
import math
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from audio_filter_pipeline.config import (
    DatasetSpec,
    PipelineConfig,
    SourceSelectorSpec,
    SpeakerMetadataSpec,
)
from audio_filter_pipeline.processors.stages_v3 import ingest as ingest_module
from audio_filter_pipeline.processors.stages_v3.base import (
    EMPTY_LINEAGE_DIGEST,
    StageRunContext,
)
from audio_filter_pipeline.processors.stages_v3.ingest import Step0IngestStage
from audio_filter_pipeline.utils.artifacts import ArtifactWriter, PipelineStep, iter_artifact_rows
from audio_filter_pipeline.utils.path_utils import AUDIO_EXTENSIONS, v3_artifact_dir


pytestmark = pytest.mark.unit
EXPECTED_ROW_FIELDS = (
    "source_key", "dataset_id", "language", "category", "quality_profile",
    "input_root", "original_path", "original_relative_path",
    "source_relative_path", "source_size_bytes", "source_sha256",
    "sample_rate", "frames", "channels", "duration_sec", "format", "subtype",
    "speaker_id", "speaker_scope",
)
ROW_FIELDS = set(EXPECTED_ROW_FIELDS)


def _write_wav(path, *, samplerate=16000, frames=1600, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((frames, channels), dtype=np.float32)
    if channels == 1:
        data = data[:, 0]
    sf.write(str(path), data, samplerate, subtype="PCM_16")


def _write_flac(path, audio, *, samplerate=48000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), samplerate, format="FLAC", subtype="PCM_16")


def _truncate_flac_after_metadata(path):
    payload = path.read_bytes()
    assert payload[:4] == b"fLaC"
    offset = 4
    while True:
        assert offset + 4 <= len(payload)
        block_header = payload[offset]
        block_size = int.from_bytes(payload[offset + 1:offset + 4], "big")
        offset += 4 + block_size
        assert offset <= len(payload)
        if block_header & 0x80:
            break
    path.write_bytes(payload[:offset])


def _dataset(root, *, dataset_id="alpha", language="en", category="read_speech",
             quality_profile="quality_high", speaker=None, source_selector=None,
             source_layout=None):
    values = {
        "dataset_id": dataset_id,
        "input_root": str(root),
        "language": language,
        "category": category,
        "quality_profile": quality_profile,
        "speaker": speaker or SpeakerMetadataSpec(source="none", required=False),
        "source_selector": source_selector or SourceSelectorSpec(),
    }
    if source_layout is not None:
        values["source_layout"] = source_layout
    return DatasetSpec(**values)


def _raw_dataset(spec):
    speaker = {"source": spec.speaker.source, "required": spec.speaker.required}
    if spec.speaker.source == "path_component":
        speaker["path_component"] = spec.speaker.path_component
    elif spec.speaker.source == "regex":
        speaker.update(regex=spec.speaker.regex, regex_group=spec.speaker.regex_group)
    elif spec.speaker.source == "mapping_file":
        speaker["mapping_path"] = spec.speaker.mapping_path
    raw = {
        "input_root": spec.input_root,
        "language": spec.language,
        "category": spec.category,
        "quality_profile": spec.quality_profile,
        "speaker": speaker,
    }
    if spec.source_selector != SourceSelectorSpec():
        selector = {}
        if spec.source_selector.sample_rates_hz:
            selector["sample_rates_hz"] = list(spec.source_selector.sample_rates_hz)
        if spec.source_selector.include_speaker_ids:
            selector["include_speaker_ids"] = list(
                spec.source_selector.include_speaker_ids
            )
        if spec.source_selector.exclude_relative_paths:
            selector["exclude_relative_paths"] = list(
                spec.source_selector.exclude_relative_paths
            )
        raw["source_selector"] = selector
    source_layout = getattr(spec, "source_layout", "regular_file")
    if source_layout != "regular_file":
        raw["source_layout"] = source_layout
    return raw


def _quality_profiles():
    """generation-3 封闭 quality_profiles registry（Task 1 精确结构）。"""
    return {
        "quality_balanced": {"thresholds": {
            "dnsmos": {"ovrl": 3.2, "sig": 3.0, "bak": 3.8, "p808": 3.6},
            "scoreq": 3.6,
            "urgent_pk": 3.5,
        }},
        "quality_high": {"thresholds": {
            "dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8},
            "scoreq": 4.2,
            "urgent_pk": 3.5,
        }},
        "quality_premium": {"thresholds": {
            "dnsmos": {"ovrl": 3.4, "sig": 3.6, "bak": 4.2, "p808": 4.0},
            "scoreq": 4.5,
            "urgent_pk": 3.5,
        }},
    }


def _config(
    output_root,
    spec,
    *,
    threshold=3.0,
    extra_specs=(),
    min_source_sample_rate_hz=None,
):
    datasets = {spec.dataset_id: _raw_dataset(spec)}
    datasets.update({item.dataset_id: _raw_dataset(item) for item in extra_specs})
    config = PipelineConfig(
        pipeline_generation=3,
        step="step0",
        output_root=str(output_root),
        datasets=datasets,
        selection_policies={
            "global": {
                "mos_models": ["dnsmos"],
                "thresholds": {"dnsmos": threshold},
                "min_mos_models": 1,
            },
            "quality_profiles": _quality_profiles(),
        },
        min_source_sample_rate_hz=min_source_sample_rate_hz,
    )
    return config


def _context(config, dataset_id="alpha"):
    return StageRunContext(config.output_root, dataset_id, config)


def _source_key(dataset_id, input_root, relative_path):
    material = f"{dataset_id}\0{os.path.abspath(input_root)}\0{relative_path}".encode()
    return hashlib.sha256(material).hexdigest()[:16]


def _payload_digest(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_path(output_root, dataset_id="alpha"):
    return Path(v3_artifact_dir(str(output_root), "step0", dataset_id))


def _assert_no_artifact_or_staging(output_root, dataset_id="alpha"):
    target = _artifact_path(output_root, dataset_id)
    assert not target.exists()
    if target.parent.exists():
        assert not list(target.parent.glob(f".{dataset_id}.stage-*"))


def _fake_scanned_source(path):
    return ingest_module._ScannedSource(path=path, snapshot=(0,) * 7)


def test_run_publishes_exact_sorted_rows_and_resume_does_not_rescan(tmp_path, monkeypatch):
    root = tmp_path / "input"
    first = root / "a" / "first.wav"
    second = root / "z.wav"
    _write_wav(second, samplerate=8000, frames=800, channels=2)
    _write_wav(first, samplerate=16000, frames=1600)
    first_bytes = first.read_bytes()
    second_bytes = second.read_bytes()
    spec = _dataset(root)
    config = _config(tmp_path / "out", spec)
    stage = Step0IngestStage(spec)

    index = stage.run(_context(config))
    rows = list(iter_artifact_rows(index))

    canonical_root = os.path.abspath(root)
    assert index.step == PipelineStep.STEP0.value
    assert index.lineage_digest == EMPTY_LINEAGE_DIGEST
    assert index.row_count == 2
    assert [row["source_relative_path"] for row in rows] == ["a/first.wav", "z.wav"]
    assert rows == [
        {
            "source_key": _source_key("alpha", root, "a/first.wav"),
            "dataset_id": "alpha", "language": "en", "category": "read_speech",
            "quality_profile": "quality_high",
            "input_root": canonical_root, "original_path": str(first.resolve()),
            "original_relative_path": "a", "source_relative_path": "a/first.wav",
            "source_size_bytes": len(first_bytes),
            "source_sha256": hashlib.sha256(first_bytes).hexdigest(),
            "sample_rate": 16000, "frames": 1600, "channels": 1,
            "duration_sec": 0.1, "format": "WAV", "subtype": "PCM_16",
            "speaker_id": None, "speaker_scope": None,
        },
        {
            "source_key": _source_key("alpha", root, "z.wav"),
            "dataset_id": "alpha", "language": "en", "category": "read_speech",
            "quality_profile": "quality_high",
            "input_root": canonical_root, "original_path": str(second.resolve()),
            "original_relative_path": "", "source_relative_path": "z.wav",
            "source_size_bytes": len(second_bytes),
            "source_sha256": hashlib.sha256(second_bytes).hexdigest(),
            "sample_rate": 8000, "frames": 800, "channels": 2,
            "duration_sec": 0.1, "format": "WAV", "subtype": "PCM_16",
            "speaker_id": None, "speaker_scope": None,
        },
    ]
    assert all(set(row) == ROW_FIELDS for row in rows)

    def forbidden_scan(*args, **kwargs):
        raise AssertionError("resume must not rescan")

    monkeypatch.setattr(Step0IngestStage, "_scan_sources", forbidden_scan)
    resumed = stage.run(_context(config))
    assert resumed.artifact_digest == index.artifact_digest
    assert resumed.config_digest == index.config_digest
    assert list(iter_artifact_rows(resumed)) == rows


def test_same_header_audio_with_different_payload_has_different_source_hash(tmp_path):
    root = tmp_path / "input"
    first = root / "first.wav"
    second = root / "second.wav"
    _write_wav(first)
    _write_wav(second)
    payload = bytearray(second.read_bytes())
    payload[-1] ^= 1
    second.write_bytes(payload)
    spec = _dataset(root)

    rows = list(iter_artifact_rows(
        Step0IngestStage(spec).run(_context(_config(tmp_path / "out", spec)))
    ))

    assert rows[0]["sample_rate"] == rows[1]["sample_rate"] == 16000
    assert rows[0]["frames"] == rows[1]["frames"] == 1600
    assert rows[0]["source_size_bytes"] == rows[1]["source_size_bytes"]
    assert rows[0]["source_sha256"] != rows[1]["source_sha256"]


def test_resume_after_live_source_mutation_returns_historical_snapshot_without_scan(
    tmp_path, monkeypatch,
):
    root = tmp_path / "input"
    source = root / "utt.wav"
    _write_wav(source)
    original_bytes = source.read_bytes()
    spec = _dataset(root)
    stage = Step0IngestStage(spec)
    config = _config(tmp_path / "out", spec)
    first = stage.run(_context(config))
    original_row = next(iter_artifact_rows(first))

    source.write_bytes(original_bytes + b"later mutation")

    def forbidden_scan(*args, **kwargs):
        raise AssertionError("historical resume must not scan live input")

    monkeypatch.setattr(Step0IngestStage, "_scan_sources", forbidden_scan)
    resumed = stage.run(_context(config))
    resumed_row = next(iter_artifact_rows(resumed))

    assert resumed.artifact_digest == first.artifact_digest
    assert resumed_row == original_row
    assert resumed_row["source_size_bytes"] == len(original_bytes)
    assert resumed_row["source_sha256"] == hashlib.sha256(original_bytes).hexdigest()


def test_source_mutation_during_hash_fails_closed_and_cleans_staging(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.ingest as ingest

    root = tmp_path / "input"
    source = root / "utt.wav"
    _write_wav(source)
    spec = _dataset(root)
    output_root = tmp_path / "out"
    real_pread = ingest.os.pread
    mutated = False

    def mutate_during_hash(descriptor, size, offset):
        nonlocal mutated
        chunk = real_pread(descriptor, size, offset)
        if chunk and not mutated:
            with source.open("ab") as stream:
                stream.write(b"mutation")
            mutated = True
        return chunk

    monkeypatch.setattr(ingest.os, "pread", mutate_during_hash)

    with pytest.raises(
        ValueError, match="source.*changed|pathname binding|extra data",
    ):
        Step0IngestStage(spec).run(_context(_config(output_root, spec)))

    assert mutated is True
    _assert_no_artifact_or_staging(output_root)


def test_stable_source_hardlink_is_allowed(tmp_path):
    root = tmp_path / "input"
    source = root / "utt.wav"
    external_link = tmp_path / "external.wav"
    _write_wav(source)
    os.link(source, external_link)
    spec = _dataset(root)

    row = next(iter_artifact_rows(
        Step0IngestStage(spec).run(_context(_config(tmp_path / "out", spec)))
    ))

    assert row["source_relative_path"] == "utt.wav"
    assert source.stat().st_nlink == 2


@pytest.mark.parametrize("phase", ["header", "hash"])
def test_source_link_count_change_during_snapshot_fails_closed(
    tmp_path, monkeypatch, phase,
):
    root = tmp_path / "input"
    source = root / "utt.wav"
    added_link = tmp_path / "added.wav"
    _write_wav(source)
    output_root = tmp_path / "out"
    spec = _dataset(root)
    mutated = False

    if phase == "header":
        real_info = ingest_module.soundfile.info

        def mutate_during_header(stream):
            nonlocal mutated
            metadata = real_info(stream)
            os.link(source, added_link)
            mutated = True
            return metadata

        monkeypatch.setattr(ingest_module.soundfile, "info", mutate_during_header)
    else:
        real_pread = ingest_module.os.pread

        def mutate_during_hash(descriptor, size, offset):
            nonlocal mutated
            chunk = real_pread(descriptor, size, offset)
            if chunk and not mutated:
                os.link(source, added_link)
                mutated = True
            return chunk

        monkeypatch.setattr(ingest_module.os, "pread", mutate_during_hash)

    with pytest.raises(ValueError, match="source.*changed|pathname binding"):
        Step0IngestStage(spec).run(_context(_config(output_root, spec)))

    assert mutated is True
    _assert_no_artifact_or_staging(output_root)


def test_continuously_growing_source_hash_is_bounded_and_cleans_resources(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.ingest as ingest

    root = tmp_path / "input"
    source = root / "utt.wav"
    _write_wav(source)
    expected_size = source.stat().st_size
    output_root = tmp_path / "out"
    requests = []
    source_descriptors = []

    def continuously_growing_pread(descriptor, size, offset):
        requests.append((size, offset))
        source_descriptors.append(descriptor)
        if sum(request_size for request_size, _ in requests) > expected_size + 1:
            raise AssertionError("source hash exceeded its initial snapshot bound")
        return b"x" * size

    monkeypatch.setattr(ingest.os, "pread", continuously_growing_pread)

    with pytest.raises(ValueError, match="extra data"):
        Step0IngestStage(_dataset(root)).run(
            _context(_config(output_root, _dataset(root)))
        )

    _assert_no_artifact_or_staging(output_root)
    assert requests == [(expected_size, 0), (1, expected_size)]
    assert sum(size for size, _ in requests) == expected_size + 1
    assert source_descriptors
    with pytest.raises(OSError) as closed:
        os.fstat(source_descriptors[0])
    assert closed.value.errno == errno.EBADF


def test_source_hash_early_eof_is_bounded_and_cleans_resources(tmp_path, monkeypatch):
    import audio_filter_pipeline.processors.stages_v3.ingest as ingest

    root = tmp_path / "input"
    source = root / "utt.wav"
    _write_wav(source)
    expected_size = source.stat().st_size
    output_root = tmp_path / "out"
    requests = []
    source_descriptors = []

    def early_eof_pread(descriptor, size, offset):
        requests.append((size, offset))
        source_descriptors.append(descriptor)
        if len(requests) == 1:
            return b"x" * (expected_size - 1)
        return b""

    monkeypatch.setattr(ingest.os, "pread", early_eof_pread)

    with pytest.raises(ValueError, match="EOF before expected size"):
        Step0IngestStage(_dataset(root)).run(
            _context(_config(output_root, _dataset(root)))
        )

    _assert_no_artifact_or_staging(output_root)
    assert requests == [(expected_size, 0), (1, expected_size - 1)]
    assert sum(size for size, _ in requests) == expected_size + 1
    assert source_descriptors
    with pytest.raises(OSError) as closed:
        os.fstat(source_descriptors[0])
    assert closed.value.errno == errno.EBADF


def test_step0_source_path_components_use_public_validator(tmp_path, monkeypatch):
    root = tmp_path / "input"
    source = root / "utt.wav"
    _write_wav(source)
    spec = _dataset(root)
    stage = Step0IngestStage(spec)
    called = []

    def recording_validator(value):
        called.append(value)
        return value

    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.ingest.validate_source_relative_path",
        recording_validator,
    )

    assert stage._source_path_components(str(source), "utt.wav") == ("utt.wav",)
    assert called == ["utt.wav"]


def test_path_component_handles_root_and_nested_paths_and_duplicate_basenames(tmp_path):
    root = tmp_path / "input"
    root_wav = root / "root.wav"
    left = root / "speaker-a" / "same.wav"
    right = root / "speaker-b" / "same.wav"
    for path in (root_wav, left, right):
        _write_wav(path)
    spec = _dataset(
        root,
        speaker=SpeakerMetadataSpec(
            source="path_component", path_component=0, required=False,
        ),
    )
    index = Step0IngestStage(spec).run(_context(_config(tmp_path / "out", spec)))
    rows = {row["source_relative_path"]: row for row in iter_artifact_rows(index)}

    assert rows["root.wav"]["speaker_id"] is None
    assert rows["root.wav"]["speaker_scope"] is None
    assert rows["speaker-a/same.wav"]["speaker_id"] == "speaker-a"
    assert rows["speaker-a/same.wav"]["speaker_scope"] == "alpha"
    assert rows["speaker-b/same.wav"]["speaker_id"] == "speaker-b"
    assert rows["speaker-b/same.wav"]["speaker_scope"] == "alpha"
    assert rows["speaker-a/same.wav"]["source_key"] != rows["speaker-b/same.wav"]["source_key"]


@pytest.mark.parametrize(
    "speaker,relative_path,expected",
    [
        (SpeakerMetadataSpec(source="none", required=False), "named/spk.wav", None),
        (SpeakerMetadataSpec(source="path_component", path_component=0, required=True), "pc/spk.wav", "pc"),
        (SpeakerMetadataSpec(source="regex", regex=r"voice-(?P<speaker_id>[^/]+)/", required=True), "voice-rx/spk.wav", "rx"),
    ],
    ids=["none", "path-component", "regex"],
)
def test_integrates_non_mapping_speaker_resolvers(tmp_path, speaker, relative_path, expected):
    root = tmp_path / "input"
    _write_wav(root / relative_path)
    spec = _dataset(root, speaker=speaker)
    row = next(iter_artifact_rows(
        Step0IngestStage(spec).run(_context(_config(tmp_path / "out", spec)))
    ))
    assert row["speaker_id"] == expected
    assert row["speaker_scope"] == ("alpha" if expected is not None else None)


def test_integrates_mapping_resolver_and_exact_bytes_change_digest(tmp_path):
    root = tmp_path / "input"
    _write_wav(root / "mapped.wav")
    mapping_dir = tmp_path / "config"
    mapping_dir.mkdir()
    mapping = mapping_dir / "speakers.jsonl"
    mapping.write_bytes(b'{"source_relative_path":"mapped.wav","speaker_id":"one"}\n')
    spec = _dataset(
        root,
        speaker=SpeakerMetadataSpec(
            source="mapping_file", mapping_path="speakers.jsonl", required=True,
        ),
    )
    first_config = _config(tmp_path / "out-first", spec)
    first = Step0IngestStage(spec, mapping_base_dir=mapping_dir)
    first_payload = first.config_payload(first_config)
    first_digest = first.config_digest(first_config)

    mapping.write_bytes(b'{"source_relative_path":"mapped.wav","speaker_id":"two"}\n')

    first_row = next(iter_artifact_rows(first.run(_context(first_config))))
    assert first_payload["speaker"]["mapping_sha256"] == hashlib.sha256(
        b'{"source_relative_path":"mapped.wav","speaker_id":"one"}\n'
    ).hexdigest()
    assert first_row["speaker_id"] == "one"
    assert first_row["speaker_scope"] == "alpha"

    second_config = _config(tmp_path / "out-second", spec)
    second = Step0IngestStage(spec, mapping_base_dir=mapping_dir)
    second_digest = second.config_digest(second_config)
    second_row = next(iter_artifact_rows(second.run(_context(second_config))))
    assert second_digest != first_digest
    assert second_row["speaker_id"] == "two"
    assert second_row["speaker_scope"] == "alpha"


def test_mapping_never_uses_process_cwd_as_implicit_base(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "speakers.jsonl").write_text("", encoding="utf-8")
    monkeypatch.chdir(cwd)
    spec = _dataset(
        root,
        speaker=SpeakerMetadataSpec(
            source="mapping_file", mapping_path="speakers.jsonl", required=False,
        ),
    )
    with pytest.raises(ValueError, match="mapping_base_dir"):
        Step0IngestStage(spec)


@pytest.mark.parametrize("required", [False, True], ids=["optional", "required"])
def test_missing_resolved_label_obeys_required_flag(tmp_path, required):
    root = tmp_path / "input"
    _write_wav(root / "root.wav")
    spec = _dataset(
        root,
        speaker=SpeakerMetadataSpec(
            source="path_component", path_component=0, required=required,
        ),
    )
    stage = Step0IngestStage(spec)
    if required:
        with pytest.raises(ValueError, match="required speaker label"):
            stage.run(_context(_config(tmp_path / "out", spec)))
        assert not _artifact_path(tmp_path / "out").exists()
    else:
        row = next(iter_artifact_rows(
            stage.run(_context(_config(tmp_path / "out", spec)))
        ))
        assert row["speaker_id"] is None
        assert row["speaker_scope"] is None


@pytest.mark.parametrize("workers", [1, 2], ids=["serial", "parallel"])
def test_sample_rate_floor_filters_before_required_speaker_resolution(tmp_path, workers):
    root = tmp_path / "input"
    _write_wav(root / "missing-speaker.wav", samplerate=8000)
    spec = _dataset(
        root,
        speaker=SpeakerMetadataSpec(
            source="path_component", path_component=0, required=True,
        ),
    )

    index = Step0IngestStage(spec, workers=workers).run(_context(_config(
        tmp_path / "out",
        spec,
        min_source_sample_rate_hz=16000,
    )))

    assert index.row_count == 0
    assert list(iter_artifact_rows(index)) == []


@pytest.mark.parametrize("workers", [1, 2], ids=["serial", "parallel"])
def test_min_source_sample_rate_filters_before_sha_and_revalidates_descriptor(
    tmp_path, monkeypatch, workers,
):
    root = tmp_path / "input"
    source_8k = root / "source-8k.wav"
    source_16k = root / "source-16k.wav"
    _write_wav(source_8k, samplerate=8000)
    _write_wav(source_16k, samplerate=16000)
    spec = _dataset(root)
    real_hash = ingest_module._hash_source_descriptor
    real_verify = Step0IngestStage._verify_source_descriptor
    hashed_paths = []
    verified_paths = []

    def recording_hash(descriptor, expected_size):
        hashed_paths.append(ingest_module._descriptor_path(descriptor, "source"))
        return real_hash(descriptor, expected_size)

    def recording_verify(self, descriptor, source_path, *args, **kwargs):
        verified_paths.append(source_path)
        return real_verify(self, descriptor, source_path, *args, **kwargs)

    monkeypatch.setattr(ingest_module, "_hash_source_descriptor", recording_hash)
    monkeypatch.setattr(
        Step0IngestStage, "_verify_source_descriptor", recording_verify,
    )

    rows = list(iter_artifact_rows(
        Step0IngestStage(spec, workers=workers).run(_context(_config(
            tmp_path / "out",
            spec,
            min_source_sample_rate_hz=16000,
        )))
    ))

    assert [(row["source_relative_path"], row["sample_rate"]) for row in rows] == [
        ("source-16k.wav", 16000),
    ]
    assert hashed_paths == [str(source_16k.resolve())]
    assert set(verified_paths) == {str(source_8k), str(source_16k)}


@pytest.mark.parametrize("workers", [1, 2], ids=["serial", "parallel"])
def test_min_source_sample_rate_can_filter_every_source_without_selector_failure(
    tmp_path, workers,
):
    root = tmp_path / "input"
    _write_wav(root / "source-8k.wav", samplerate=8000)
    spec = _dataset(root)

    index = Step0IngestStage(spec, workers=workers).run(_context(_config(
        tmp_path / "out",
        spec,
        min_source_sample_rate_hz=16000,
    )))

    assert index.row_count == 0
    assert list(iter_artifact_rows(index)) == []


@pytest.mark.parametrize("workers", [1, 2], ids=["serial", "parallel"])
def test_source_selector_sample_rate_filters_nonmatches_without_error(tmp_path, workers):
    root = tmp_path / "input"
    _write_wav(root / "selected-16k.wav", samplerate=16000)
    _write_wav(root / "ignored-32k.wav", samplerate=32000)
    spec = _dataset(
        root,
        source_selector=SourceSelectorSpec(sample_rates_hz=(16000,)),
    )

    rows = list(iter_artifact_rows(
        Step0IngestStage(spec, workers=workers).run(
            _context(_config(tmp_path / "out", spec))
        )
    ))

    assert [(row["source_relative_path"], row["sample_rate"]) for row in rows] == [
        ("selected-16k.wav", 16000),
    ]


@pytest.mark.parametrize("workers", [1, 2], ids=["serial", "parallel"])
def test_source_selector_excludes_exact_relative_paths_without_touching_other_sources(
    tmp_path, workers,
):
    root = tmp_path / "input"
    _write_wav(root / "noise" / "040" / "noise_DX01C01.wav")
    _write_wav(root / "noise" / "040" / "noise_DX02C01.wav")
    _write_wav(root / "noise" / "040" / "keep.wav")
    spec = _dataset(
        root,
        source_selector=SourceSelectorSpec(
            exclude_relative_paths=(
                "noise/040/noise_DX01C01.wav",
                "noise/040/noise_DX02C01.wav",
            ),
        ),
    )

    rows = list(iter_artifact_rows(
        Step0IngestStage(spec, workers=workers).run(
            _context(_config(tmp_path / "out", spec))
        )
    ))

    assert [row["source_relative_path"] for row in rows] == [
        "noise/040/keep.wav",
    ]


@pytest.mark.parametrize("workers", [1, 2], ids=["serial", "parallel"])
def test_min_source_sample_rate_combines_with_source_selector(tmp_path, workers):
    root = tmp_path / "input"
    _write_wav(root / "source-8k.wav", samplerate=8000)
    _write_wav(root / "source-16k.wav", samplerate=16000)
    spec = _dataset(
        root,
        source_selector=SourceSelectorSpec(sample_rates_hz=(16000,)),
    )

    rows = list(iter_artifact_rows(
        Step0IngestStage(spec, workers=workers).run(_context(_config(
            tmp_path / "out",
            spec,
            min_source_sample_rate_hz=16000,
        )))
    ))

    assert [(row["source_relative_path"], row["sample_rate"]) for row in rows] == [
        ("source-16k.wav", 16000),
    ]


def test_source_selector_filters_speaker_ids(tmp_path):
    root = tmp_path / "input"
    _write_wav(root / "speaker-a" / "selected.wav")
    _write_wav(root / "speaker-b" / "ignored.wav")
    spec = _dataset(
        root,
        speaker=SpeakerMetadataSpec(
            source="path_component", path_component=0, required=True,
        ),
        source_selector=SourceSelectorSpec(include_speaker_ids=("speaker-a",)),
    )

    rows = list(iter_artifact_rows(
        Step0IngestStage(spec).run(_context(_config(tmp_path / "out", spec)))
    ))

    assert [(row["source_relative_path"], row["speaker_id"]) for row in rows] == [
        ("speaker-a/selected.wav", "speaker-a"),
    ]


def test_source_selector_combines_sample_rate_and_speaker_filters(tmp_path):
    root = tmp_path / "input"
    _write_wav(root / "speaker-a" / "selected-16k.wav", samplerate=16000)
    _write_wav(root / "speaker-a" / "ignored-32k.wav", samplerate=32000)
    _write_wav(root / "speaker-b" / "ignored-16k.wav", samplerate=16000)
    spec = _dataset(
        root,
        speaker=SpeakerMetadataSpec(
            source="path_component", path_component=0, required=True,
        ),
        source_selector=SourceSelectorSpec(
            sample_rates_hz=(16000,), include_speaker_ids=("speaker-a",),
        ),
    )

    rows = list(iter_artifact_rows(
        Step0IngestStage(spec).run(_context(_config(tmp_path / "out", spec)))
    ))

    assert [row["source_relative_path"] for row in rows] == [
        "speaker-a/selected-16k.wav",
    ]
    assert rows[0]["sample_rate"] == 16000
    assert rows[0]["speaker_id"] == "speaker-a"


@pytest.mark.parametrize("workers", [1, 2], ids=["serial", "parallel"])
def test_combined_selector_requires_each_speaker_to_match_final_rows(tmp_path, workers):
    root = tmp_path / "input"
    _write_wav(root / "speaker-a" / "selected-16k.wav", samplerate=16000)
    _write_wav(root / "speaker-b" / "filtered-32k.wav", samplerate=32000)
    spec = _dataset(
        root,
        speaker=SpeakerMetadataSpec(
            source="path_component", path_component=0, required=True,
        ),
        source_selector=SourceSelectorSpec(
            sample_rates_hz=(16000,),
            include_speaker_ids=("speaker-a", "speaker-b"),
        ),
    )
    output_root = tmp_path / "out"

    with pytest.raises(ValueError, match="include_speaker_ids.*speaker-b"):
        Step0IngestStage(spec, workers=workers).run(
            _context(_config(output_root, spec))
        )

    _assert_no_artifact_or_staging(output_root)


@pytest.mark.parametrize("workers", [1, 2], ids=["serial", "parallel"])
def test_combined_selector_accepts_speakers_with_final_matching_rows(tmp_path, workers):
    root = tmp_path / "input"
    _write_wav(root / "speaker-a" / "selected-16k.wav", samplerate=16000)
    _write_wav(root / "speaker-b" / "selected-16k.wav", samplerate=16000)
    spec = _dataset(
        root,
        speaker=SpeakerMetadataSpec(
            source="path_component", path_component=0, required=True,
        ),
        source_selector=SourceSelectorSpec(
            sample_rates_hz=(16000,),
            include_speaker_ids=("speaker-a", "speaker-b"),
        ),
    )

    rows = list(iter_artifact_rows(
        Step0IngestStage(spec, workers=workers).run(
            _context(_config(tmp_path / "out", spec))
        )
    ))

    assert [(row["speaker_id"], row["sample_rate"]) for row in rows] == [
        ("speaker-a", 16000),
        ("speaker-b", 16000),
    ]


def test_source_selector_declared_sample_rate_without_matches_fails_closed(tmp_path):
    root = tmp_path / "input"
    _write_wav(root / "source-16k.wav", samplerate=16000)
    _write_wav(root / "source-32k.wav", samplerate=32000)
    spec = _dataset(
        root,
        source_selector=SourceSelectorSpec(sample_rates_hz=(48000,)),
    )
    output_root = tmp_path / "out"

    with pytest.raises(ValueError) as failure:
        Step0IngestStage(spec).run(_context(_config(output_root, spec)))

    message = str(failure.value)
    assert "source_selector" in message
    assert "sample_rates_hz" in message
    assert "matched no" in message
    _assert_no_artifact_or_staging(output_root)


def test_source_selector_declared_speaker_without_sources_fails_closed(tmp_path):
    root = tmp_path / "input"
    _write_wav(root / "speaker-a" / "source.wav")
    spec = _dataset(
        root,
        speaker=SpeakerMetadataSpec(
            source="path_component", path_component=0, required=True,
        ),
        source_selector=SourceSelectorSpec(include_speaker_ids=("missing",)),
    )
    output_root = tmp_path / "out"

    with pytest.raises(ValueError) as failure:
        Step0IngestStage(spec).run(_context(_config(output_root, spec)))

    message = str(failure.value)
    assert "source_selector" in message
    assert "include_speaker_ids" in message
    assert "matched no" in message
    _assert_no_artifact_or_staging(output_root)


def test_config_digest_owns_only_step0_inputs(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    spec = _dataset(root)
    stage = Step0IngestStage(spec, recursive=False)
    low = _config(tmp_path / "out", spec, threshold=2.0)
    high = _config(tmp_path / "out", spec, threshold=4.5)
    assert stage.config_digest(low) == stage.config_digest(high)
    assert stage.config_payload(low) == {
        "dataset": {
            "dataset_id": "alpha", "input_root": os.path.abspath(root),
            "language": "en", "category": "read_speech",
            "quality_profile": "quality_high",
        },
        "speaker": {"source": "none", "required": False},
        "scanner": {
            "version": 3,
            "recursive": False,
            "audio_extensions": sorted(AUDIO_EXTENSIONS),
        },
        "row_contract": {
            "version": 5,
            "fields": list(EXPECTED_ROW_FIELDS),
        },
    }
    gated = _config(
        tmp_path / "gated",
        spec,
        min_source_sample_rate_hz=16000,
    )
    assert stage.config_payload(gated)["scanner"] == {
        "version": 3,
        "recursive": False,
        "audio_extensions": sorted(AUDIO_EXTENSIONS),
        "min_source_sample_rate_hz": 16000,
    }
    assert stage.config_digest(gated) != stage.config_digest(low)

    changed = _dataset(root, language="fr")
    changed_config = _config(tmp_path / "other", changed)
    assert Step0IngestStage(changed).config_digest(changed_config) != stage.config_digest(low)

    # category 与 quality_profile 必须独立影响 Step0 config digest（不能互相推断）
    changed_category = _dataset(root, category="expressive")
    category_config = _config(tmp_path / "category", changed_category)
    assert Step0IngestStage(changed_category).config_digest(category_config) != stage.config_digest(low)
    changed_profile = _dataset(root, quality_profile="quality_premium")
    profile_config = _config(tmp_path / "profile", changed_profile)
    assert Step0IngestStage(changed_profile).config_digest(profile_config) != stage.config_digest(low)
    assert Step0IngestStage(changed_profile).config_digest(profile_config) != \
        Step0IngestStage(changed_category).config_digest(category_config)


def test_source_selector_changes_config_payload_and_digest(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    speaker = SpeakerMetadataSpec(
        source="path_component", path_component=0, required=True,
    )
    base_spec = _dataset(root, speaker=speaker)
    selected_spec = _dataset(
        root,
        speaker=speaker,
        source_selector=SourceSelectorSpec(
            sample_rates_hz=(16000, 32000),
            include_speaker_ids=("speaker-a",),
        ),
    )
    base_stage = Step0IngestStage(base_spec)
    selected_stage = Step0IngestStage(selected_spec)
    base_config = _config(tmp_path / "base", base_spec)
    selected_config = _config(tmp_path / "selected", selected_spec)

    base_payload = base_stage.config_payload(base_config)
    selected_payload = selected_stage.config_payload(selected_config)

    assert "source_selector" not in base_payload["dataset"]
    assert selected_payload["dataset"]["source_selector"] == {
        "sample_rates_hz": [16000, 32000],
        "include_speaker_ids": ["speaker-a"],
    }
    assert selected_payload != base_payload
    assert selected_stage.config_digest(selected_config) != base_stage.config_digest(
        base_config
    )


def test_row_contract_marker_matches_module_constant_and_produced_row_order(tmp_path):
    import audio_filter_pipeline.processors.stages_v3.ingest as ingest

    root = tmp_path / "input"
    _write_wav(root / "utt.wav")
    spec = _dataset(root)
    config = _config(tmp_path / "out", spec)
    stage = Step0IngestStage(spec)

    marker = stage.config_payload(config)["row_contract"]
    row = next(iter(stage.produce_rows(_context(config), None, ())))

    assert marker == {"version": 5, "fields": list(EXPECTED_ROW_FIELDS)}
    assert ingest.STEP0_ROW_FIELDS == EXPECTED_ROW_FIELDS
    assert tuple(row) == EXPECTED_ROW_FIELDS
    assert set(row) == set(marker["fields"])


def test_legacy_17_field_artifact_digest_mismatch_fails_without_scan(
    tmp_path, monkeypatch,
):
    root = tmp_path / "input"
    root.mkdir()
    spec = _dataset(root)
    config = _config(tmp_path / "out", spec)
    stage = Step0IngestStage(spec)
    legacy_payload = dict(stage.config_payload(config))
    legacy_payload.pop("row_contract", None)
    legacy_digest = _payload_digest(legacy_payload)
    legacy_fields = tuple(
        field for field in EXPECTED_ROW_FIELDS
        if field not in {"source_size_bytes", "source_sha256"}
    )
    writer = ArtifactWriter(
        config.output_root,
        PipelineStep.STEP0,
        "alpha",
        legacy_digest,
        EMPTY_LINEAGE_DIGEST,
    )
    writer.append({field: None for field in legacy_fields})
    legacy_index = writer.finalize()
    assert legacy_index.row_count == 1
    assert legacy_digest != stage.config_digest(config)

    def forbidden_scan(*args, **kwargs):
        raise AssertionError("legacy artifact rejection must not scan live input")

    monkeypatch.setattr(Step0IngestStage, "_scan_sources", forbidden_scan)

    with pytest.raises(ValueError, match="expected config digest mismatch"):
        stage.run(_context(config))


def test_previous_row_contract_version_resume_fails_before_scan(
    tmp_path, monkeypatch,
):
    """旧 row_contract（version 2）payload 的 artifact：resume 必须在扫描前
    fail closed，且已有 artifact 目录不被覆盖或改写。"""
    root = tmp_path / "input"
    root.mkdir()
    spec = _dataset(root)
    config = _config(tmp_path / "out", spec)
    stage = Step0IngestStage(spec)
    previous_payload = dict(stage.config_payload(config))
    previous_payload["row_contract"] = {
        "version": 2, "fields": list(EXPECTED_ROW_FIELDS),
    }
    previous_digest = _payload_digest(previous_payload)
    assert previous_digest != stage.config_digest(config)
    writer = ArtifactWriter(
        config.output_root,
        PipelineStep.STEP0,
        "alpha",
        previous_digest,
        EMPTY_LINEAGE_DIGEST,
    )
    writer.append({"stale": True})
    writer.finalize()
    artifact_dir = _artifact_path(config.output_root)
    rows_before = (artifact_dir / "rows.jsonl").read_bytes()

    def forbidden_scan(*args, **kwargs):
        raise AssertionError("previous row-contract resume must not scan live input")

    monkeypatch.setattr(Step0IngestStage, "_scan_sources", forbidden_scan)

    with pytest.raises(ValueError, match="expected config digest mismatch"):
        stage.run(_context(config))

    assert (artifact_dir / "rows.jsonl").read_bytes() == rows_before


def test_symlink_escape_rejects_entire_stage_and_cleans_output(tmp_path):
    root = tmp_path / "input"
    _write_wav(root / "safe.wav")
    outside = tmp_path / "outside.wav"
    _write_wav(outside)
    (root / "escape.wav").symlink_to(outside)
    spec = _dataset(root)

    with pytest.raises(ValueError, match="symlink|input_root"):
        Step0IngestStage(spec).run(_context(_config(tmp_path / "out", spec)))
    assert not _artifact_path(tmp_path / "out").exists()


def test_source_retarget_after_identity_check_fails_closed_and_cleans_staging(
    tmp_path, monkeypatch,
):
    root = tmp_path / "input"
    safe = root / "safe.wav"
    outside = tmp_path / "outside.wav"
    link = root / "race.wav"
    _write_wav(safe, samplerate=8000)
    _write_wav(outside, samplerate=48000)
    link.symlink_to(safe.name)
    spec = _dataset(root)
    output_root = tmp_path / "out"
    real_identity = __import__(
        "audio_filter_pipeline.utils.path_utils", fromlist=["make_source_identity"]
    ).make_source_identity
    raced = False

    def retarget_after_identity(input_root, source_path, dataset_id, language):
        nonlocal raced
        identity = real_identity(input_root, source_path, dataset_id, language)
        if source_path == str(link) and not raced:
            raced = True
            link.unlink()
            link.symlink_to(outside)
        return identity

    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.ingest.make_source_identity",
        retarget_after_identity,
    )

    with pytest.raises((RuntimeError, ValueError), match="changed|symlink|source"):
        Step0IngestStage(spec).run(_context(_config(output_root, spec)))

    # descriptor walker 会在 identity 构造前拒绝 symlink；若更晚才遇到也必须拒绝。
    assert raced is False
    _assert_no_artifact_or_staging(output_root)


def test_scan_is_descriptor_bound_and_never_calls_live_path_expander(
    tmp_path, monkeypatch,
):
    root = tmp_path / "input"
    _write_wav(root / "nested" / "utt.wav")
    spec = _dataset(root)
    monkeypatch.setattr(
        "audio_filter_pipeline.utils.path_utils.expand_audio_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("live pathname scanner must not run")
        ),
    )

    rows = list(iter_artifact_rows(
        Step0IngestStage(spec).run(_context(_config(tmp_path / "out", spec)))
    ))

    assert [row["source_relative_path"] for row in rows] == ["nested/utt.wav"]


def test_wide_directory_scan_holds_child_descriptors_only_to_recursion_depth(
    tmp_path, monkeypatch,
):
    root = tmp_path / "input"
    fan_out = 40
    for index in range(fan_out):
        _write_wav(root / f"child-{index:02d}" / "nested" / "utt.wav")
    spec = _dataset(root)
    stage = Step0IngestStage(spec)
    real_open = ingest_module.os.open
    real_close = ingest_module.os.close
    held_directory_descriptors = set()
    peak_held = 0

    def tracking_open(path, flags, *args, **kwargs):
        nonlocal peak_held
        descriptor = real_open(path, flags, *args, **kwargs)
        if kwargs.get("dir_fd") is not None and flags & getattr(os, "O_DIRECTORY", 0):
            held_directory_descriptors.add(descriptor)
            peak_held = max(peak_held, len(held_directory_descriptors))
        return descriptor

    def tracking_close(descriptor):
        held_directory_descriptors.discard(descriptor)
        return real_close(descriptor)

    monkeypatch.setattr(ingest_module.os, "open", tracking_open)
    monkeypatch.setattr(ingest_module.os, "close", tracking_close)

    root_descriptor, single_file = stage._open_root_descriptor()
    try:
        sources = list(stage._scan_sources(root_descriptor, single_file))
    finally:
        os.close(root_descriptor)

    assert len(sources) == fan_out
    assert peak_held <= 2
    assert held_directory_descriptors == set()
    assert [source.path for source in sources] == sorted(source.path for source in sources)


def test_source_scan_spool_is_removed_when_iterator_closes_early(tmp_path, monkeypatch):
    root = tmp_path / "input"
    _write_wav(root / "z.wav")
    _write_wav(root / "a.wav")
    stage = Step0IngestStage(_dataset(root))
    temporary_paths = []
    real_temporary_directory = ingest_module.tempfile.TemporaryDirectory

    class TrackingTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            self._temporary = real_temporary_directory(*args, **kwargs)
            self.name = self._temporary.name
            temporary_paths.append(Path(self.name))

        def cleanup(self):
            return self._temporary.cleanup()

    monkeypatch.setattr(ingest_module.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        ingest_module.tempfile, "TemporaryDirectory", TrackingTemporaryDirectory,
    )
    root_descriptor, single_file = stage._open_root_descriptor()
    try:
        sources = stage._scan_sources(root_descriptor, single_file)
        assert next(sources).path.endswith("a.wav")
        sources.close()
    finally:
        os.close(root_descriptor)

    assert len(temporary_paths) == 1
    assert temporary_paths[0].parent == tmp_path
    assert not temporary_paths[0].exists()


def test_source_swap_after_descriptor_scan_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "input"
    source = root / "utt.wav"
    replacement = tmp_path / "replacement.wav"
    held = tmp_path / "held.wav"
    _write_wav(source, samplerate=8000)
    _write_wav(replacement, samplerate=48000)
    spec = _dataset(root)
    output_root = tmp_path / "out"
    real_identity = ingest_module.make_source_identity
    swapped = False

    def swap_before_open(*args, **kwargs):
        nonlocal swapped
        identity = real_identity(*args, **kwargs)
        if not swapped:
            source.rename(held)
            replacement.rename(source)
            swapped = True
        return identity

    monkeypatch.setattr(ingest_module, "make_source_identity", swap_before_open)

    with pytest.raises(ValueError, match="changed after descriptor scan"):
        Step0IngestStage(spec).run(_context(_config(output_root, spec)))

    assert swapped is True
    _assert_no_artifact_or_staging(output_root)


@pytest.mark.parametrize("component", ["root", "intermediate"])
def test_directory_swap_and_restore_during_descriptor_scan_fails_closed(
    tmp_path, monkeypatch, component,
):
    root = tmp_path / "input"
    nested = root / "nested"
    replacement = tmp_path / "replacement"
    held = tmp_path / "held"
    parked = tmp_path / "parked"
    _write_wav(nested / "original.wav")
    _write_wav(replacement / "replacement.wav")
    spec = _dataset(root)
    output_root = tmp_path / "out"
    real_scandir = os.scandir
    target = root if component == "root" else nested
    target_identity = (target.stat().st_dev, target.stat().st_ino)
    swapped = False

    def swap_and_restore(descriptor):
        nonlocal swapped
        if isinstance(descriptor, int):
            opened = os.fstat(descriptor)
            if not swapped and (opened.st_dev, opened.st_ino) == target_identity:
                target.rename(held)
                replacement.rename(target)
                target.rename(parked)
                held.rename(target)
                swapped = True
        return real_scandir(descriptor)

    monkeypatch.setattr(ingest_module.os, "scandir", swap_and_restore)

    with pytest.raises(ValueError, match="scan|directory|input_root|changed"):
        Step0IngestStage(spec).run(_context(_config(output_root, spec)))

    assert swapped is True
    _assert_no_artifact_or_staging(output_root)


def test_directory_component_symlink_is_rejected(tmp_path):
    root = tmp_path / "input"
    real_directory = root / "real"
    linked_directory = root / "linked"
    _write_wav(real_directory / "utt.wav")
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    spec = _dataset(root)
    output_root = tmp_path / "out"

    with pytest.raises(ValueError, match="directory component|symlink|source"):
        Step0IngestStage(spec).run(_context(_config(output_root, spec)))

    _assert_no_artifact_or_staging(output_root)


@pytest.mark.parametrize(
    "source_relative_path",
    ["", "/utt.wav", ".", "..", "nested//utt.wav", "nested/../utt.wav"],
    ids=["empty", "absolute", "dot", "dot-dot", "empty-component", "parent-component"],
)
def test_invalid_source_relative_path_components_fail_closed(
    tmp_path, monkeypatch, source_relative_path,
):
    root = tmp_path / "input"
    source = root / "nested" / "utt.wav"
    _write_wav(source)
    spec = _dataset(root)
    output_root = tmp_path / "out"
    real_identity = __import__(
        "audio_filter_pipeline.utils.path_utils", fromlist=["make_source_identity"]
    ).make_source_identity

    def invalid_identity(input_root, source_path, dataset_id, language):
        identity = real_identity(input_root, source_path, dataset_id, language)
        return type(identity)(
            identity.dataset_id,
            identity.language,
            identity.original_relative_path,
            source_relative_path,
            identity.source_key,
            identity.input_root,
        )

    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.ingest.make_source_identity",
        invalid_identity,
    )

    with pytest.raises(ValueError, match="source_relative_path|path component"):
        Step0IngestStage(spec).run(_context(_config(output_root, spec)))

    _assert_no_artifact_or_staging(output_root)


def test_unreadable_audio_fails_whole_stage_and_cleans_output(tmp_path):
    root = tmp_path / "input"
    _write_wav(root / "good.wav")
    (root / "bad.wav").write_bytes(b"not audio")
    spec = _dataset(root)

    with pytest.raises((RuntimeError, ValueError)):
        Step0IngestStage(spec).run(_context(_config(tmp_path / "out", spec)))
    assert not _artifact_path(tmp_path / "out").exists()


def test_duplicate_stream_open_failure_closes_all_descriptors_and_staging(
    tmp_path, monkeypatch,
):
    root = tmp_path / "input"
    _write_wav(root / "utt.wav")
    spec = _dataset(root)
    output_root = tmp_path / "out"
    real_fdopen = os.fdopen

    def fail_read_stream(descriptor, mode="r", *args, **kwargs):
        if mode == "rb":
            raise OSError("injected duplicate stream open failure")
        return real_fdopen(descriptor, mode, *args, **kwargs)

    baseline = len(os.listdir("/proc/self/fd"))
    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.ingest.os.fdopen",
        fail_read_stream,
    )

    with pytest.raises(OSError, match="duplicate stream open failure"):
        Step0IngestStage(spec).run(_context(_config(output_root, spec)))

    assert len(os.listdir("/proc/self/fd")) == baseline
    _assert_no_artifact_or_staging(output_root)


def test_header_projection_preserves_held_source_descriptor_offset(
    tmp_path, monkeypatch,
):
    root = tmp_path / "input"
    _write_wav(root / "utt.wav")
    spec = _dataset(root)
    real_verify = Step0IngestStage._verify_source_descriptor
    observed_offsets = []

    def consuming_info(stream):
        assert stream.read(1)
        return SimpleNamespace(
            samplerate=16000,
            frames=1600,
            channels=1,
            format="WAV",
            subtype="PCM_16",
        )

    def verify_offset(self, descriptor, *args, **kwargs):
        observed_offsets.append(os.lseek(descriptor, 0, os.SEEK_CUR))
        return real_verify(self, descriptor, *args, **kwargs)

    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.ingest.soundfile.info",
        consuming_info,
    )
    monkeypatch.setattr(
        Step0IngestStage, "_verify_source_descriptor", verify_offset,
    )

    Step0IngestStage(spec).run(_context(_config(tmp_path / "out", spec)))

    assert observed_offsets == [0, 0]


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("samplerate", True, "sample rate"), ("samplerate", 0, "sample rate"),
        ("frames", True, "frames"), ("frames", 0, "frames"),
        ("channels", True, "channels"), ("channels", 0, "channels"),
        ("format", "", "format"), ("format", None, "format"),
        ("subtype", "", "subtype"), ("subtype", None, "subtype"),
    ],
)
def test_invalid_header_values_fail_closed(tmp_path, monkeypatch, field, value, error):
    root = tmp_path / "input"
    wav = root / "bad.wav"
    _write_wav(wav)
    values = dict(samplerate=16000, frames=1600, channels=1, format="WAV", subtype="PCM_16")
    values[field] = value
    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.ingest.soundfile.info",
        lambda _path: SimpleNamespace(**values),
    )
    monkeypatch.setattr(
        ingest_module,
        "_hash_source_descriptor",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("invalid audio header must fail before SHA-256")
        ),
    )
    spec = _dataset(root)
    with pytest.raises(ValueError, match=error):
        Step0IngestStage(spec).run(_context(_config(tmp_path / "out", spec)))
    assert not _artifact_path(tmp_path / "out").exists()


@pytest.mark.parametrize("duration", [math.inf, math.nan, 0.0, -1.0, True])
def test_nonfinite_or_nonpositive_duration_fails_closed(tmp_path, monkeypatch, duration):
    root = tmp_path / "input"
    _write_wav(root / "bad.wav")
    spec = _dataset(root)
    config = _config(tmp_path / "out", spec)
    stage = Step0IngestStage(spec)

    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.ingest._duration_sec",
        lambda _frames, _samplerate: duration,
    )
    monkeypatch.setattr(
        ingest_module,
        "_hash_source_descriptor",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("invalid audio duration must fail before SHA-256")
        ),
    )
    with pytest.raises(ValueError, match="duration"):
        stage.run(_context(config))
    assert not _artifact_path(tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["missing", "unsupported-file", "fifo"])
def test_invalid_input_root_types_fail_closed(tmp_path, kind):
    root = tmp_path / "input"
    if kind == "unsupported-file":
        root.write_text("not audio", encoding="utf-8")
    elif kind == "fifo":
        os.mkfifo(root)
    spec = _dataset(root)
    with pytest.raises(ValueError, match="input_root|audio extension|regular|directory"):
        Step0IngestStage(spec).run(_context(_config(tmp_path / "out", spec)))
    assert not _artifact_path(tmp_path / "out").exists()


def test_supported_single_file_root_is_ingested(tmp_path):
    wav = tmp_path / "single.wav"
    _write_wav(wav)
    spec = _dataset(wav)
    row = next(iter_artifact_rows(
        Step0IngestStage(spec).run(_context(_config(tmp_path / "out", spec)))
    ))
    assert row["original_relative_path"] == ""
    assert row["source_relative_path"] == str(wav.resolve()).lstrip("/")
    assert row["original_path"] == str(wav.resolve())


def test_nonrecursive_scan_ignores_nested_audio(tmp_path):
    root = tmp_path / "input"
    _write_wav(root / "root.wav")
    _write_wav(root / "nested" / "hidden.wav")
    spec = _dataset(root)
    rows = list(iter_artifact_rows(
        Step0IngestStage(spec, recursive=False).run(
            _context(_config(tmp_path / "out", spec))
        )
    ))
    assert [row["source_relative_path"] for row in rows] == ["root.wav"]


def test_duplicate_source_key_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "input"
    _write_wav(root / "a.wav")
    _write_wav(root / "b.wav")
    spec = _dataset(root)
    real_identity = __import__(
        "audio_filter_pipeline.utils.path_utils", fromlist=["make_source_identity"]
    ).make_source_identity

    def colliding_identity(input_root, source_path, dataset_id, language):
        identity = real_identity(input_root, source_path, dataset_id, language)
        return type(identity)(
            identity.dataset_id, identity.language, identity.original_relative_path,
            identity.source_relative_path, "same-key", identity.input_root,
        )

    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.ingest.make_source_identity",
        colliding_identity,
    )
    with pytest.raises(ValueError, match="duplicate source_key"):
        Step0IngestStage(spec).run(_context(_config(tmp_path / "out", spec)))
    assert not _artifact_path(tmp_path / "out").exists()


@pytest.mark.parametrize("mismatch", ["context-id", "config-root", "config-semantics"])
def test_context_config_and_bound_dataset_mismatch_fails_closed(tmp_path, mismatch):
    root = tmp_path / "input"
    root.mkdir()
    bound = _dataset(root)
    other = _dataset(tmp_path / "other", dataset_id="beta")
    stage = Step0IngestStage(bound)
    if mismatch == "context-id":
        config = _config(tmp_path / "out", bound, extra_specs=(other,))
        context = _context(config, "beta")
    elif mismatch == "config-root":
        configured = _dataset(tmp_path / "different")
        config = _config(tmp_path / "out", configured)
        context = _context(config)
    else:
        configured = _dataset(root, language="fr")
        config = _config(tmp_path / "out", configured)
        context = _context(config)

    with pytest.raises(ValueError, match="dataset.*mismatch|bound dataset"):
        stage.run(context)
    assert not _artifact_path(tmp_path / "out", context.dataset_id).exists()


def test_context_dataset_mismatch_fails_before_resume_fast_path(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    bound = _dataset(root)
    other = _dataset(tmp_path / "other", dataset_id="beta")
    config = _config(tmp_path / "out", bound, extra_specs=(other,))
    stage = Step0IngestStage(bound)
    writer = ArtifactWriter(
        config.output_root,
        PipelineStep.STEP0,
        "beta",
        stage.config_digest(config),
        EMPTY_LINEAGE_DIGEST,
    )
    writer.append({"forged": "resume"})
    writer.finalize()

    with pytest.raises(ValueError, match="context dataset mismatch"):
        stage.run(_context(config, "beta"))


@pytest.mark.parametrize("mismatch", ["config-root", "config-semantics"])
def test_config_mismatch_fails_before_resume_reopen_or_rescan(
    tmp_path, monkeypatch, mismatch,
):
    root = tmp_path / "input"
    root.mkdir()
    bound = _dataset(root)
    stage = Step0IngestStage(bound)
    valid = _config(tmp_path / "out", bound)
    writer = ArtifactWriter(
        valid.output_root,
        PipelineStep.STEP0,
        "alpha",
        stage.config_digest(valid),
        EMPTY_LINEAGE_DIGEST,
    )
    writer.append({"existing": True})
    writer.finalize()
    configured = (
        _dataset(tmp_path / "different")
        if mismatch == "config-root"
        else _dataset(root, language="fr")
    )
    mismatched = _config(tmp_path / "out", configured)

    def forbidden(*args, **kwargs):
        raise AssertionError("dataset binding must fail before reopen or rescan")

    monkeypatch.setattr(Step0IngestStage, "_open_output", forbidden)
    monkeypatch.setattr(Step0IngestStage, "_scan_sources", forbidden)

    with pytest.raises(ValueError, match="config dataset mismatch"):
        stage.run(_context(mismatched))


def test_relative_config_created_after_cwd_change_fails_closed(tmp_path, monkeypatch):
    construction = tmp_path / "construction"
    runtime = tmp_path / "runtime"
    construction.mkdir()
    runtime.mkdir()
    monkeypatch.chdir(construction)
    _write_wav(construction / "relative" / "utt.wav")
    stage = Step0IngestStage(_dataset("relative"))

    monkeypatch.chdir(runtime)
    config = _config(tmp_path / "out", _dataset(construction / "relative"))

    index = stage.run(_context(config))

    assert next(iter_artifact_rows(index))["input_root"] == str(
        (construction / "relative").resolve()
    )


def test_absolute_config_is_canonical_equivalent_to_relative_stage_spec(
    tmp_path, monkeypatch,
):
    construction = tmp_path / "construction"
    construction.mkdir()
    monkeypatch.chdir(construction)
    root = construction / "relative"
    _write_wav(root / "utt.wav")
    stage = Step0IngestStage(_dataset("relative"))
    config = _config(tmp_path / "out", _dataset(root))

    row = next(iter_artifact_rows(stage.run(_context(config))))

    assert row["input_root"] == str(root.resolve())


def test_constructor_freezes_absolute_input_identity_and_execution_state(tmp_path, monkeypatch):
    construction = tmp_path / "construction"
    runtime = tmp_path / "runtime"
    construction.mkdir()
    runtime.mkdir()
    monkeypatch.chdir(construction)
    _write_wav(construction / "relative" / "utt.wav")
    relative_spec = _dataset("relative")
    stage = Step0IngestStage(relative_spec)
    frozen_state = dict(vars(stage))
    config = _config(tmp_path / "out", _dataset(construction / "relative"))

    monkeypatch.chdir(runtime)
    index = stage.run(_context(config))

    assert list(iter_artifact_rows(index))[0]["input_root"] == str((construction / "relative").resolve())
    assert vars(stage) == frozen_state


def test_pooled_rows_failure_cancels_pending_and_preserves_primary_error(
    tmp_path, monkeypatch,
):
    root = tmp_path / "input"
    root.mkdir()
    stage = Step0IngestStage(_dataset(root), workers=2)
    cancelled = []
    shutdowns = []

    class Future:
        def __init__(self, source):
            self.source_path = source.path

        def result(self):
            if self.source_path == "first.wav":
                raise RuntimeError("step0 worker boom")
            return {"source_key": self.source_path}

        def cancel(self):
            cancelled.append(self.source_path)
            return True

    class Executor:
        def __init__(self, *, max_workers):
            assert max_workers == 2

        def submit(
            self,
            _function,
            source_path,
            _root_descriptor,
            _single_file,
            _min_source_sample_rate_hz,
        ):
            return Future(source_path)

        def shutdown(self, **kwargs):
            shutdowns.append(kwargs)

    monkeypatch.setattr(ingest_module, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(
        Step0IngestStage,
        "_scan_sources",
        lambda _self, _root_descriptor, _single_file: [
            _fake_scanned_source("first.wav"),
            _fake_scanned_source("second.wav"),
        ],
    )

    with pytest.raises(RuntimeError, match="step0 worker boom") as failure:
        list(stage._pooled_rows(0, False, set()))

    assert cancelled == ["second.wav"]
    assert shutdowns == [{"wait": False, "cancel_futures": True}]
    assert any(
        "bounded executor shutdown: stage=step0-ingest:alpha" in note
        for note in getattr(failure.value, "__notes__", ())
    )


def test_pooled_rows_releases_completed_ownerships_with_worker_bound(
    tmp_path, monkeypatch,
):
    root = tmp_path / "input"
    root.mkdir()
    stage = Step0IngestStage(_dataset(root), workers=2)
    references = []
    real_ownership = ingest_module._SubmittedRootDescriptor

    def tracking_ownership(descriptor):
        ownership = real_ownership(descriptor)
        references.append(weakref.ref(ownership))
        return ownership

    class ImmediateFuture:
        def __init__(self, source):
            self.source_path = source.path

        def result(self):
            return {"source_key": self.source_path}

        def cancel(self):
            return False

        def add_done_callback(self, _callback):
            return None

    class ImmediateExecutor:
        def __init__(self, *, max_workers):
            assert max_workers == 2

        def submit(self, _function, source_path, *_args):
            return ImmediateFuture(source_path)

        def shutdown(self, **_kwargs):
            return None

    monkeypatch.setattr(ingest_module, "_SubmittedRootDescriptor", tracking_ownership)
    monkeypatch.setattr(ingest_module, "ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(
        Step0IngestStage,
        "_scan_sources",
        lambda *_args: [
            _fake_scanned_source(f"source-{index}.wav") for index in range(50)
        ],
    )

    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        rows = stage._pooled_rows(descriptor, False, set())
        for _row in rows:
            gc.collect()
            assert sum(reference() is not None for reference in references) <= 5
    finally:
        os.close(descriptor)


def test_pooled_rows_ownership_constructor_failure_closes_duplicate(
    tmp_path, monkeypatch,
):
    root = tmp_path / "input"
    root.mkdir()
    stage = Step0IngestStage(_dataset(root), workers=2)

    class Executor:
        def __init__(self, *, max_workers):
            assert max_workers == 2

        def shutdown(self, **_kwargs):
            return None

    monkeypatch.setattr(ingest_module, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(
        Step0IngestStage,
        "_scan_sources",
        lambda *_args: [_fake_scanned_source("one.wav")],
    )
    monkeypatch.setattr(
        ingest_module,
        "_SubmittedRootDescriptor",
        lambda _descriptor: (_ for _ in ()).throw(RuntimeError("ownership boom")),
    )
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    baseline = len(os.listdir("/proc/self/fd"))
    try:
        with pytest.raises(RuntimeError, match="ownership boom"):
            list(stage._pooled_rows(descriptor, False, set()))
        assert len(os.listdir("/proc/self/fd")) == baseline
    finally:
        os.close(descriptor)


def test_pooled_rows_submit_failure_closes_pending_ownership(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    stage = Step0IngestStage(_dataset(root), workers=2)

    class Executor:
        def __init__(self, *, max_workers):
            assert max_workers == 2

        def submit(self, *_args):
            raise RuntimeError("submit boom")

        def shutdown(self, **_kwargs):
            return None

    monkeypatch.setattr(ingest_module, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(
        Step0IngestStage,
        "_scan_sources",
        lambda *_args: [_fake_scanned_source("one.wav")],
    )
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    baseline = len(os.listdir("/proc/self/fd"))
    try:
        with pytest.raises(RuntimeError, match="submit boom"):
            list(stage._pooled_rows(descriptor, False, set()))
        assert len(os.listdir("/proc/self/fd")) == baseline
    finally:
        os.close(descriptor)


def test_pooled_rows_callback_registration_failure_closes_pending_ownership(
    tmp_path, monkeypatch,
):
    root = tmp_path / "input"
    root.mkdir()
    stage = Step0IngestStage(_dataset(root), workers=2)
    cancelled = []

    class Future:
        def cancel(self):
            cancelled.append(True)
            return True

        def add_done_callback(self, _callback):
            raise RuntimeError("callback boom")

    class Executor:
        def __init__(self, *, max_workers):
            assert max_workers == 2

        def submit(self, *_args):
            return Future()

        def shutdown(self, **_kwargs):
            return None

    monkeypatch.setattr(ingest_module, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(
        Step0IngestStage,
        "_scan_sources",
        lambda *_args: [_fake_scanned_source("one.wav")],
    )
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    baseline = len(os.listdir("/proc/self/fd"))
    try:
        with pytest.raises(RuntimeError, match="callback boom"):
            list(stage._pooled_rows(descriptor, False, set()))
        assert cancelled == [True]
        assert len(os.listdir("/proc/self/fd")) == baseline
    finally:
        os.close(descriptor)


def test_pooled_rows_generator_close_cancels_pending_and_closes_ownerships(
    tmp_path, monkeypatch,
):
    root = tmp_path / "input"
    root.mkdir()
    stage = Step0IngestStage(_dataset(root), workers=2)
    cancelled = []
    shutdowns = []

    class Future:
        def __init__(self, source):
            self.source_path = source.path

        def result(self):
            return {"source_key": self.source_path}

        def cancel(self):
            cancelled.append(self.source_path)
            return True

        def add_done_callback(self, _callback):
            return None

    class Executor:
        def __init__(self, *, max_workers):
            assert max_workers == 2

        def submit(self, _function, source_path, *_args):
            return Future(source_path)

        def shutdown(self, **kwargs):
            shutdowns.append(kwargs)

    monkeypatch.setattr(ingest_module, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(
        Step0IngestStage,
        "_scan_sources",
        lambda *_args: [
            _fake_scanned_source(path)
            for path in ("one.wav", "two.wav", "three.wav", "four.wav")
        ],
    )
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    baseline = len(os.listdir("/proc/self/fd"))
    try:
        rows = stage._pooled_rows(descriptor, False, set())
        assert next(rows)["source_key"] == "one.wav"
        rows.close()
        assert set(cancelled) == {"two.wav", "three.wav", "four.wav"}
        assert shutdowns == [{"wait": False, "cancel_futures": True}]
        assert len(os.listdir("/proc/self/fd")) == baseline
    finally:
        os.close(descriptor)


def test_produce_rows_root_close_does_not_replace_primary_error(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    spec = _dataset(root)
    stage = Step0IngestStage(spec)
    config = _config(tmp_path / "out", spec)
    root_descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    real_close = ingest_module.os.close

    monkeypatch.setattr(
        Step0IngestStage,
        "_open_root_descriptor",
        lambda _self: (root_descriptor, False),
    )
    monkeypatch.setattr(
        Step0IngestStage,
        "_scan_sources",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("scan boom")),
    )

    def fail_root_close(descriptor):
        if descriptor == root_descriptor:
            real_close(descriptor)
            raise OSError("root close boom")
        return real_close(descriptor)

    monkeypatch.setattr(ingest_module.os, "close", fail_root_close)

    with pytest.raises(RuntimeError, match="scan boom") as failure:
        list(stage.produce_rows(_context(config), None, ()))

    assert any(
        "root descriptor close failed" in note
        for note in getattr(failure.value, "__notes__", ())
    )


def test_streaming_flac_aggregate_is_one_exact_step0_row(tmp_path):
    root = tmp_path / "input"
    aggregate = root / "utterance_active"
    first = np.array([0.10, 0.20, 0.30], dtype=np.float32)
    second = np.array([-0.40, -0.50, -0.60, -0.70, -0.80], dtype=np.float32)
    _write_flac(aggregate / "utterance_active_0000.flac", first)
    _write_flac(aggregate / "utterance_active_0001.flac", second)
    spec = _dataset(root, source_layout="streaming_flac_aggregate")
    config = _config(tmp_path / "out", spec)

    rows = list(iter_artifact_rows(Step0IngestStage(spec).run(_context(config))))

    assert len(rows) == 1
    row = rows[0]
    assert set(row) == ROW_FIELDS
    assert len(row) == len(EXPECTED_ROW_FIELDS) == 19
    assert row["source_key"] == _source_key("alpha", root, "utterance_active")
    assert row["input_root"] == str(root.resolve())
    assert row["original_path"] == str(aggregate.resolve())
    assert row["original_relative_path"] == ""
    assert row["source_relative_path"] == "utterance_active"
    assert row["source_size_bytes"] == (
        (aggregate / "utterance_active_0000.flac").stat().st_size
        + (aggregate / "utterance_active_0001.flac").stat().st_size
    )
    assert row["sample_rate"] == 48_000
    assert row["frames"] == len(first) + len(second)
    assert row["channels"] == 1
    assert row["duration_sec"] == row["frames"] / row["sample_rate"]
    assert row["format"] == "FLAC"
    assert row["subtype"] == "PCM_16"
    assert len(row["source_sha256"]) == 64
    raw_concat_hash = hashlib.sha256(
        (aggregate / "utterance_active_0000.flac").read_bytes()
        + (aggregate / "utterance_active_0001.flac").read_bytes()
    ).hexdigest()
    assert row["source_sha256"] != raw_concat_hash


def test_streaming_flac_aggregate_excludes_metadata_only_member_and_versions_policy(
    tmp_path, monkeypatch, caplog,
):
    root = tmp_path / "input"
    broken = root / "broken_active"
    healthy = root / "healthy_active"
    broken_member = broken / "broken_active_0000.flac"
    _write_flac(broken_member, np.linspace(-0.5, 0.5, 8, dtype=np.float32))
    _truncate_flac_after_metadata(broken_member)
    _write_flac(
        healthy / "healthy_active_0000.flac",
        np.linspace(-0.25, 0.25, 6, dtype=np.float32),
    )
    spec = _dataset(root, source_layout="streaming_flac_aggregate")
    config = _config(tmp_path / "out", spec)
    stage = Step0IngestStage(spec)

    payload = stage.config_payload(config)
    assert payload["dataset"]["source_layout"] == "streaming_flac_aggregate"
    assert payload["scanner"]["source_layout"] == "streaming_flac_aggregate"
    assert payload["scanner"]["manifest"] == {
        "domain": ingest_module.aggregate_source.AGGREGATE_MANIFEST_DOMAIN,
        "version": ingest_module.aggregate_source.AGGREGATE_MANIFEST_VERSION,
    }
    assert payload["scanner"]["policy"] == {
        "scanner_version": ingest_module.aggregate_source.AGGREGATE_SCANNER_VERSION,
        "policy_version": ingest_module.aggregate_source.AGGREGATE_POLICY_VERSION,
        "read_chunk_frames": ingest_module.aggregate_source.AGGREGATE_READ_CHUNK_FRAMES,
        "max_member_frames": ingest_module.aggregate_source.AGGREGATE_MAX_MEMBER_FRAMES,
        "max_members": ingest_module.aggregate_source.AGGREGATE_MAX_MEMBERS,
    }
    original_digest = stage.config_digest(config)
    monkeypatch.setattr(
        ingest_module.aggregate_source,
        "AGGREGATE_POLICY_VERSION",
        ingest_module.aggregate_source.AGGREGATE_POLICY_VERSION + 1,
    )
    assert stage.config_digest(config) != original_digest

    with caplog.at_level(logging.WARNING, logger=ingest_module.logger.name):
        rows = list(iter_artifact_rows(stage.run(_context(config))))

    assert [row["source_relative_path"] for row in rows] == ["healthy_active"]
    warning = next(
        record for record in caplog.records
        if "excluding empty streaming FLAC aggregate" in record.message
    )
    assert "dataset_id=alpha" in warning.message
    assert "source_relative_path=broken_active" in warning.message
    assert "member=broken_active_0000.flac" in warning.message
    assert "reason=streaming FLAC aggregate member is empty" in warning.message
