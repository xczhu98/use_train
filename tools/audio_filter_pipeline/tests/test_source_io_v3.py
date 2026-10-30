"""Generation 3 source descriptor 解码边界测试。"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import audio_filter_pipeline.processors.stages_v3.aggregate_source as aggregate_source
from audio_filter_pipeline.config import DatasetSpec, PipelineConfig, SpeakerMetadataSpec
from audio_filter_pipeline.processors.stages_v3.base import StageRunContext
from audio_filter_pipeline.processors.stages_v3.ingest import Step0IngestStage
from audio_filter_pipeline.processors.stages_v3.source_io import (
    DecodedSource,
    decode_verified_source,
    open_verified_source_session,
)
from audio_filter_pipeline.utils.artifacts import iter_artifact_rows


pytestmark = pytest.mark.unit


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def _write_flac(path: Path, audio: np.ndarray, sample_rate: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, format="FLAC", subtype="PCM_16")


def _set_flac_total_samples(path: Path, total_samples: int) -> None:
    payload = bytearray(path.read_bytes())
    assert payload[:4] == b"fLaC"
    assert payload[4] & 0x7F == 0
    assert int.from_bytes(payload[5:8], "big") == 34
    total_samples_offset = 8 + 10
    packed = int.from_bytes(
        payload[total_samples_offset:total_samples_offset + 8], "big",
    )
    payload[total_samples_offset:total_samples_offset + 8] = (
        (packed & ~((1 << 36) - 1)) | total_samples
    ).to_bytes(8, "big")
    path.write_bytes(payload)


def _truncate_flac_after_metadata(path: Path) -> None:
    payload = path.read_bytes()
    assert payload[:4] == b"fLaC"
    offset = 4
    while True:
        assert offset + 4 <= len(payload)
        header = payload[offset]
        block_size = int.from_bytes(payload[offset + 1:offset + 4], "big")
        offset += 4 + block_size
        assert offset <= len(payload)
        if header & 0x80:
            break
    path.write_bytes(payload[:offset])


def _row(root: Path, relative_path: str) -> dict:
    source = root if root.is_file() else root / relative_path
    payload = source.read_bytes()
    info = sf.info(str(source))
    return {
        "input_root": os.path.realpath(os.path.abspath(root)),
        "source_relative_path": relative_path,
        "source_size_bytes": len(payload),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "sample_rate": info.samplerate,
        "frames": info.frames,
        "channels": info.channels,
        "duration_sec": info.frames / info.samplerate,
        "format": info.format,
        "subtype": info.subtype,
        "unrelated_step0_field": {"must": "remain unchanged"},
    }


def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def _aggregate_row(tmp_path: Path) -> tuple[dict, np.ndarray, aggregate_source.AggregateDescriptor]:
    root = tmp_path / "input"
    aggregate = root / "utterance_active"
    first = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32)
    second = np.array([-0.40, -0.50, -0.60, -0.70, -0.80], dtype=np.float32)
    _write_flac(aggregate / "utterance_active_0000.flac", first)
    _write_flac(aggregate / "utterance_active_0001.flac", second)
    directory_descriptor = os.open(aggregate, os.O_RDONLY | os.O_DIRECTORY)
    try:
        descriptor = aggregate_source.scan_aggregate_directory(
            directory_descriptor,
            str(root.resolve()),
            "utterance_active",
        )
    finally:
        os.close(directory_descriptor)
    return {
        "input_root": str(root.resolve()),
        "source_relative_path": "utterance_active",
        "source_size_bytes": descriptor.source_size_bytes,
        "source_sha256": descriptor.manifest_sha256,
        "sample_rate": descriptor.sample_rate,
        "frames": descriptor.frames,
        "channels": descriptor.channels,
        "duration_sec": descriptor.frames / descriptor.sample_rate,
        "format": descriptor.format,
        "subtype": descriptor.subtype,
    }, np.concatenate((first, second)), descriptor


def _resource_parameters(
    *,
    max_decoded_bytes=512 * 1024 * 1024,
    max_raw_source_bytes=512 * 1024 * 1024,
    max_vad_analysis_bytes=256 * 1024 * 1024,
    min_source_sample_rate=8000,
    max_source_sample_rate=192000,
):
    from audio_filter_pipeline.processors.stages_v3.source_io import (
        SourceDecodeParameters,
    )

    return SourceDecodeParameters(
        max_decoded_bytes=max_decoded_bytes,
        max_raw_source_bytes=max_raw_source_bytes,
        max_vad_analysis_bytes=max_vad_analysis_bytes,
        min_source_sample_rate=min_source_sample_rate,
        max_source_sample_rate=max_source_sample_rate,
    )


def test_decodes_verified_mono_wav_without_mutating_row(tmp_path):
    root = tmp_path / "input"
    source = root / "speaker" / "mono.wav"
    expected = np.linspace(-0.8, 0.8, 321, dtype=np.float32)
    _write_wav(source, expected)
    row = _row(root, "speaker/mono.wav")
    original = dict(row)
    original["unrelated_step0_field"] = dict(row["unrelated_step0_field"])

    decoded = decode_verified_source(row)

    assert isinstance(decoded, DecodedSource)
    assert decoded.sample_rate == 16000
    assert decoded.frames == 321
    assert decoded.channels == 1
    assert decoded.format == "WAV"
    assert decoded.subtype == "PCM_16"
    assert decoded.source_size_bytes == source.stat().st_size
    assert decoded.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert decoded.audio.shape == (321,)
    assert decoded.audio.dtype == np.float32
    assert decoded.audio.flags.c_contiguous
    assert decoded.audio.flags.owndata
    np.testing.assert_allclose(decoded.audio, expected, atol=1 / 32768)
    assert row == original


def test_decodes_verified_stereo_wav_retaining_only_first_channel(tmp_path):
    root = tmp_path / "input"
    source = root / "stereo.wav"
    left = np.linspace(-0.75, 0.25, 257, dtype=np.float32)
    right = np.linspace(0.6, -0.4, 257, dtype=np.float32)
    expected = np.column_stack((left, right))
    _write_wav(source, expected, sample_rate=8000)

    decoded = decode_verified_source(_row(root, "stereo.wav"))

    assert decoded.audio.shape == (257,)
    assert decoded.channels == 2
    np.testing.assert_allclose(decoded.audio, expected[:, 0], atol=1 / 32768)


def test_accepts_non_bool_numpy_real_duration(tmp_path):
    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(8000, dtype=np.float32))
    row = _row(root, "source.wav")
    row["duration_sec"] = np.float32(0.5)

    decoded = decode_verified_source(row)

    assert decoded.frames == 8000


def test_accepts_canonical_regular_file_root_with_exact_posix_relative_name(tmp_path):
    source = tmp_path / "single.wav"
    _write_wav(source, np.zeros(80, dtype=np.float32), sample_rate=8000)
    canonical = os.path.realpath(os.path.abspath(source))

    decoded = decode_verified_source(_row(source, canonical.lstrip("/")))

    assert decoded.frames == 80
    assert decoded.audio.shape == (80,)


def test_accepts_filesystem_root_as_canonical_directory_root(tmp_path):
    source = tmp_path / "rooted.wav"
    _write_wav(source, np.zeros(80, dtype=np.float32), sample_rate=8000)
    relative_path = os.path.abspath(source).lstrip("/")
    row = _row(Path("/"), relative_path)

    decoded = decode_verified_source(row)

    assert decoded.frames == 80
    assert decoded.audio.shape == (80,)


def test_rejects_same_size_byte_mutation_by_sha256_before_decode(tmp_path, monkeypatch):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "mutated.wav"
    _write_wav(source, np.zeros(100, dtype=np.float32))
    row = _row(root, "mutated.wav")
    payload = bytearray(source.read_bytes())
    payload[-1] ^= 1
    source.write_bytes(payload)

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("decoder must not run before SHA-256 verification")

    monkeypatch.setattr(source_io, "_decode_source_descriptor", forbidden_decode)

    with pytest.raises(ValueError, match="sha256"):
        decode_verified_source(row)


def test_rejects_size_mismatch_before_decoder_entry(tmp_path, monkeypatch):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "size.wav"
    _write_wav(source, np.zeros(100, dtype=np.float32))
    row = _row(root, "size.wav")
    row["source_size_bytes"] += 1

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("decoder must not run before size verification")

    monkeypatch.setattr(source_io, "_decode_source_descriptor", forbidden_decode)

    with pytest.raises(ValueError, match="size"):
        decode_verified_source(row)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("sample_rate", 8000),
        ("frames", 99),
        ("channels", 2),
        ("format", "FLAC"),
        ("subtype", "FLOAT"),
        ("duration_sec", 0.123456789),
    ],
)
def test_rejects_row_header_mismatch(tmp_path, field, replacement):
    root = tmp_path / "input"
    source = root / "header.wav"
    _write_wav(source, np.zeros(100, dtype=np.float32))
    row = _row(root, "header.wav")
    row[field] = replacement

    with pytest.raises(ValueError, match="header"):
        decode_verified_source(row)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda source: source.unlink(),
        lambda source: (source.unlink(), source.symlink_to("target.wav")),
    ],
    ids=["deleted", "source-symlink"],
)
def test_rejects_deleted_or_symlink_source(tmp_path, mutator):
    root = tmp_path / "input"
    source = root / "source.wav"
    target = root / "target.wav"
    _write_wav(source, np.zeros(40, dtype=np.float32))
    _write_wav(target, np.zeros(40, dtype=np.float32))
    row = _row(root, "source.wav")
    mutator(source)

    with pytest.raises((ValueError, OSError), match="binding|source"):
        decode_verified_source(row)


def test_rejects_intermediate_directory_symlink(tmp_path):
    root = tmp_path / "input"
    real_parent = root / "real"
    source = real_parent / "source.wav"
    _write_wav(source, np.zeros(40, dtype=np.float32))
    row = _row(root, "real/source.wav")
    alias = root / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    row["source_relative_path"] = "alias/source.wav"

    with pytest.raises(ValueError, match="binding|directory|source"):
        decode_verified_source(row)


def test_rejects_root_symlink(tmp_path):
    real_root = tmp_path / "real"
    source = real_root / "source.wav"
    _write_wav(source, np.zeros(40, dtype=np.float32))
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)
    row = _row(real_root, "source.wav")
    row["input_root"] = os.path.abspath(alias)

    with pytest.raises(ValueError, match="input_root|binding"):
        decode_verified_source(row)


def test_rejects_fifo_source_without_blocking(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    fifo = root / "source.wav"
    os.mkfifo(fifo)
    row = {
        "input_root": os.path.realpath(os.path.abspath(root)),
        "source_relative_path": "source.wav",
        "source_size_bytes": 0,
        "source_sha256": hashlib.sha256(b"").hexdigest(),
        "sample_rate": 16000,
        "frames": 1,
        "channels": 1,
        "duration_sec": 1 / 16000,
        "format": "WAV",
        "subtype": "PCM_16",
    }

    with pytest.raises(ValueError, match="source|regular"):
        decode_verified_source(row)


@pytest.mark.parametrize("phase", ["before", "post"])
def test_source_swap_matrix_decodes_held_fd_then_rejects_binding(
    tmp_path, monkeypatch, phase,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    replacement = root / "replacement.wav"
    _write_wav(source, np.full(64, 0.2, dtype=np.float32))
    _write_wav(replacement, np.full(64, -0.6, dtype=np.float32))
    row = _row(root, "source.wav")
    real_decode = source_io._decode_source_descriptor
    decoded_means = []

    def replace_named_source():
        source.rename(root / "held-original.wav")
        replacement.rename(source)

    def record_held_decode(descriptor, **kwargs):
        audio, metadata = real_decode(descriptor, **kwargs)
        decoded_means.append(float(audio.mean()))
        return audio, metadata

    seam = "_before_decode_test_hook" if phase == "before" else "_post_decode_test_hook"
    monkeypatch.setattr(source_io, seam, replace_named_source)
    monkeypatch.setattr(source_io, "_decode_source_descriptor", record_held_decode)

    with pytest.raises(ValueError, match="binding"):
        decode_verified_source(row)

    assert decoded_means == pytest.approx([0.2], abs=1 / 32768)


def test_post_decode_source_rewalk_starts_from_same_held_root_fd(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "a" / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    row = _row(root, "a/source.wav")
    real_open_source = source_io._open_directory_source
    root_anchors = []

    def recording_open_source(root_fd, relative_path):
        root_anchors.append(root_fd)
        return real_open_source(root_fd, relative_path)

    monkeypatch.setattr(source_io, "_open_directory_source", recording_open_source)

    decode_verified_source(row)

    assert len(root_anchors) == 2
    assert root_anchors[1] == root_anchors[0]


@pytest.mark.parametrize("phase", ["before", "post"])
def test_directory_root_swap_matrix_rejects_binding(tmp_path, monkeypatch, phase):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    row = _row(root, "source.wav")

    def replace_named_root():
        root.rename(tmp_path / "held-root")
        root.mkdir()

    seam = "_before_decode_test_hook" if phase == "before" else "_post_decode_test_hook"
    monkeypatch.setattr(source_io, seam, replace_named_root)

    with pytest.raises(ValueError, match="binding"):
        decode_verified_source(row)


def test_directory_root_swap_between_type_probe_and_held_open_rejects_binding(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    original = tmp_path / "original-root"
    real_open_absolute = source_io._open_absolute_binding
    calls = 0

    def swap_after_probe(*args, **kwargs):
        nonlocal calls
        descriptor = real_open_absolute(*args, **kwargs)
        calls += 1
        if calls == 1:
            root.rename(original)
            root.mkdir()
        return descriptor

    monkeypatch.setattr(source_io, "_open_absolute_binding", swap_after_probe)

    with pytest.raises(ValueError, match="binding"):
        source_io._open_root(os.path.realpath(os.path.abspath(root)))


@pytest.mark.parametrize("phase", ["before", "post"])
def test_regular_file_root_swap_matrix_rejects_binding(tmp_path, monkeypatch, phase):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    source = tmp_path / "single.wav"
    replacement = tmp_path / "replacement.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    _write_wav(replacement, np.ones(64, dtype=np.float32))
    canonical = os.path.realpath(os.path.abspath(source))
    row = _row(source, canonical.lstrip("/"))

    def replace_regular_file_root():
        source.rename(tmp_path / "held-single.wav")
        replacement.rename(source)

    seam = "_before_decode_test_hook" if phase == "before" else "_post_decode_test_hook"
    monkeypatch.setattr(source_io, seam, replace_regular_file_root)

    with pytest.raises(ValueError, match="binding"):
        decode_verified_source(row)


@pytest.mark.parametrize("phase", ["before", "post"])
def test_deep_parent_symlink_to_moved_original_rejects_binding(
    tmp_path, monkeypatch, phase,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    parent = root / "a" / "b"
    source = parent / "source.wav"
    moved = root / "moved-b"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    row = _row(root, "a/b/source.wav")

    def replace_parent_with_symlink_to_original():
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)

    seam = "_before_decode_test_hook" if phase == "before" else "_post_decode_test_hook"
    monkeypatch.setattr(source_io, seam, replace_parent_with_symlink_to_original)

    with pytest.raises(ValueError, match="binding"):
        decode_verified_source(row)


@pytest.mark.parametrize("phase", ["before", "post"])
def test_absolute_root_ancestor_symlink_to_moved_original_rejects_binding(
    tmp_path, monkeypatch, phase,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    ancestor = tmp_path / "ancestor"
    root = ancestor / "nested" / "input"
    source = root / "source.wav"
    moved = tmp_path / "moved-ancestor"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    row = _row(root, "source.wav")

    def replace_ancestor_with_symlink_to_original():
        ancestor.rename(moved)
        ancestor.symlink_to(moved, target_is_directory=True)

    seam = "_before_decode_test_hook" if phase == "before" else "_post_decode_test_hook"
    monkeypatch.setattr(source_io, seam, replace_ancestor_with_symlink_to_original)

    with pytest.raises(ValueError, match="binding"):
        decode_verified_source(row)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("input_root", "relative/root", "input_root"),
        ("input_root", 123, "input_root"),
        ("source_relative_path", "/absolute.wav", "source_relative_path"),
        ("source_relative_path", "../escape.wav", "source_relative_path"),
        ("source_relative_path", "dot/./source.wav", "source_relative_path"),
        ("source_relative_path", "empty//source.wav", "source_relative_path"),
        ("source_relative_path", "back\\slash.wav", "source_relative_path"),
        ("source_relative_path", "nul\x00source.wav", "source_relative_path"),
        ("source_size_bytes", True, "size"),
        ("source_size_bytes", -1, "size"),
        ("source_sha256", "A" * 64, "sha256"),
        ("source_sha256", "0" * 63, "sha256"),
        ("sample_rate", True, "header"),
        ("sample_rate", 0, "header"),
        ("frames", 0, "header"),
        ("channels", 0, "header"),
        ("duration_sec", True, "header"),
        ("duration_sec", float("inf"), "header"),
        ("duration_sec", 0.0, "header"),
        ("format", "", "header"),
        ("subtype", 3, "header"),
    ],
)
def test_rejects_malformed_row_fields(tmp_path, field, value, match):
    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    row = _row(root, "source.wav")
    row[field] = value

    with pytest.raises((ValueError, OSError), match=match):
        decode_verified_source(row)


def test_rejects_missing_required_row_field(tmp_path):
    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    row = _row(root, "source.wav")
    del row["source_sha256"]

    with pytest.raises(ValueError, match="sha256"):
        decode_verified_source(row)


def test_repeated_success_and_failure_do_not_leak_file_descriptors(tmp_path):
    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    valid = _row(root, "source.wav")
    invalid = dict(valid, source_sha256="0" * 64)
    baseline = _fd_count()

    for _ in range(30):
        decode_verified_source(valid)
        with pytest.raises(ValueError, match="sha256"):
            decode_verified_source(invalid)

    assert _fd_count() <= baseline + 1


def test_read_and_soundfile_close_double_failure_preserves_decode_error(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    row = _row(root, "source.wav")
    real_soundfile = source_io.sf.SoundFile
    close_attempts = []

    class ReadAndCloseFail:
        def __init__(self, *args, **kwargs):
            self._stream = real_soundfile(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def read(self, *args, **kwargs):
            raise OSError("injected read failure")

        def close(self):
            close_attempts.append("soundfile")
            self._stream.close()
            raise OSError("injected SoundFile.close failure")

    monkeypatch.setattr(source_io.sf, "SoundFile", ReadAndCloseFail)
    baseline = _fd_count()

    with pytest.raises(ValueError, match="decode") as caught:
        decode_verified_source(row)

    assert close_attempts == ["soundfile"]
    assert any("cleanup" in note for note in getattr(caught.value, "__notes__", ()))
    assert _fd_count() <= baseline


def test_binding_and_fd_close_double_failure_preserves_binding_and_closes_all(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    replacement = root / "replacement.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    _write_wav(replacement, np.ones(64, dtype=np.float32))
    row = _row(root, "source.wav")
    real_close = source_io.os.close
    real_verify_snapshot = source_io._verify_snapshot
    close_attempts = []
    binding_detected = False
    failed_once = False

    def replace_source():
        source.rename(root / "held-source.wav")
        replacement.rename(source)

    def record_binding_detection(*args, **kwargs):
        nonlocal binding_detected
        try:
            return real_verify_snapshot(*args, **kwargs)
        except ValueError:
            binding_detected = True
            raise

    def close_then_fail_once(descriptor):
        nonlocal failed_once
        close_attempts.append((descriptor, binding_detected))
        real_close(descriptor)
        if binding_detected and not failed_once:
            failed_once = True
            raise OSError("injected fd cleanup failure")

    monkeypatch.setattr(source_io, "_post_decode_test_hook", replace_source)
    monkeypatch.setattr(source_io, "_verify_snapshot", record_binding_detection)
    monkeypatch.setattr(source_io.os, "close", close_then_fail_once)
    baseline = _fd_count()

    with pytest.raises(ValueError, match="binding") as caught:
        decode_verified_source(row)

    assert failed_once is True
    assert sum(after_binding for _, after_binding in close_attempts) >= 3
    assert any("cleanup" in note for note in getattr(caught.value, "__notes__", ()))
    assert _fd_count() <= baseline


def test_terminal_parent_close_after_release_never_recloses_reused_sentinel(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "a" / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    real_open_component = source_io._open_component
    real_open = source_io.os.open
    real_close = source_io.os.close
    terminal_parent = None
    terminal_fd = None
    sentinel_fd = None
    injected = False

    def record_terminal(component, flags, parent_fd, label):
        nonlocal terminal_parent, terminal_fd
        descriptor = real_open_component(component, flags, parent_fd, label)
        if label.endswith("terminal"):
            terminal_parent = parent_fd
            terminal_fd = descriptor
        return descriptor

    def close_release_reuse_then_fail(descriptor):
        nonlocal sentinel_fd, injected
        real_close(descriptor)
        if descriptor == terminal_parent and not injected:
            injected = True
            sentinel_fd = real_open("/dev/null", os.O_RDONLY)
            assert sentinel_fd == descriptor
            raise OSError("injected terminal-parent close-after-release failure")

    monkeypatch.setattr(source_io, "_open_component", record_terminal)
    monkeypatch.setattr(source_io.os, "close", close_release_reuse_then_fail)
    try:
        with pytest.raises(ValueError, match="binding.*cleanup"):
            source_io._open_directory_source(root_fd, "a/source.wav")

        assert injected is True
        os.fstat(sentinel_fd)
        with pytest.raises(OSError):
            os.fstat(terminal_fd)
    finally:
        monkeypatch.setattr(source_io.os, "close", real_close)
        for descriptor in (sentinel_fd, terminal_fd, root_fd):
            if descriptor is not None:
                try:
                    real_close(descriptor)
                except OSError:
                    pass


def test_directory_probe_close_after_release_never_recloses_reused_sentinel(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    real_open_absolute = source_io._open_absolute_binding
    real_open = source_io.os.open
    real_close = source_io.os.close
    opened_roots = []
    slash_fd = None
    sentinel_fd = None
    filler_fds = []
    injected = False

    def record_open(path, flags, *args, **kwargs):
        nonlocal slash_fd
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == "/" and kwargs.get("dir_fd") is None and slash_fd is None:
            slash_fd = descriptor
        return descriptor

    def record_open_absolute(*args, **kwargs):
        descriptor = real_open_absolute(*args, **kwargs)
        opened_roots.append(descriptor)
        return descriptor

    def close_release_reuse_then_fail(descriptor):
        nonlocal sentinel_fd, injected
        if opened_roots and descriptor == opened_roots[0] and not injected:
            injected = True
            while True:
                filler_fd = real_open("/dev/null", os.O_RDONLY)
                filler_fds.append(filler_fd)
                if filler_fd > descriptor:
                    break
            real_close(descriptor)
            sentinel_fd = real_open("/dev/null", os.O_RDONLY)
            assert sentinel_fd == descriptor
            raise OSError("injected directory-probe close-after-release failure")
        real_close(descriptor)

    monkeypatch.setattr(source_io.os, "open", record_open)
    monkeypatch.setattr(source_io, "_open_absolute_binding", record_open_absolute)
    monkeypatch.setattr(source_io.os, "close", close_release_reuse_then_fail)
    try:
        with pytest.raises(ValueError, match="binding.*cleanup"):
            source_io._open_root(os.path.realpath(os.path.abspath(root)))

        assert injected is True
        os.fstat(sentinel_fd)
        with pytest.raises(OSError):
            os.fstat(opened_roots[1])
        with pytest.raises(OSError):
            os.fstat(slash_fd)
    finally:
        monkeypatch.setattr(source_io.os, "close", real_close)
        for descriptor in (sentinel_fd, *filler_fds, *opened_roots, slash_fd):
            if descriptor is not None:
                try:
                    real_close(descriptor)
                except OSError:
                    pass


def test_directory_walk_transfers_child_ownership_before_old_parent_cleanup(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "a" / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    real_open = source_io.os.open
    real_close = source_io.os.close
    opened_children = []
    close_attempts = []
    failed_once = False

    def recording_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if kwargs.get("dir_fd") is not None:
            opened_children.append(descriptor)
        return descriptor

    def fail_first_cleanup(descriptor):
        nonlocal failed_once
        close_attempts.append(descriptor)
        try:
            real_close(descriptor)
        except OSError:
            pass
        if not failed_once:
            failed_once = True
            raise OSError("injected old parent cleanup failure")

    monkeypatch.setattr(source_io.os, "open", recording_open)
    monkeypatch.setattr(source_io.os, "close", fail_first_cleanup)
    try:
        with pytest.raises(ValueError, match="binding|cleanup"):
            source_io._open_directory_source(root_fd, "a/source.wav")
    finally:
        monkeypatch.setattr(source_io.os, "close", real_close)
        for descriptor in opened_children:
            try:
                real_close(descriptor)
            except OSError:
                pass
        real_close(root_fd)

    assert opened_children[0] in close_attempts


def test_directory_root_transfers_owned_fd_before_probe_cleanup(tmp_path, monkeypatch):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    real_open_absolute = source_io._open_absolute_binding
    real_close = source_io.os.close
    opened_roots = []
    close_attempts = []
    failed_once = False

    def recording_open_absolute(*args, **kwargs):
        descriptor = real_open_absolute(*args, **kwargs)
        opened_roots.append(descriptor)
        return descriptor

    def fail_probe_cleanup(descriptor):
        nonlocal failed_once
        close_attempts.append(descriptor)
        try:
            real_close(descriptor)
        except OSError:
            pass
        if len(opened_roots) >= 2 and descriptor == opened_roots[0] and not failed_once:
            failed_once = True
            raise OSError("injected root probe cleanup failure")

    monkeypatch.setattr(source_io, "_open_absolute_binding", recording_open_absolute)
    monkeypatch.setattr(source_io.os, "close", fail_probe_cleanup)
    leaked_directory_fd = False
    try:
        with pytest.raises(ValueError, match="binding.*cleanup"):
            source_io._open_root(os.path.realpath(os.path.abspath(root)))
        try:
            os.fstat(opened_roots[1])
        except OSError:
            pass
        else:
            leaked_directory_fd = True
    finally:
        monkeypatch.setattr(source_io.os, "close", real_close)
        for descriptor in opened_roots:
            try:
                real_close(descriptor)
            except OSError:
                pass

    assert failed_once is True
    assert leaked_directory_fd is False


def test_directory_root_second_fstat_failure_closes_probe_and_directory_fds(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    real_open_absolute = source_io._open_absolute_binding
    real_fstat = source_io._fstat
    opened_roots = []
    fstat_calls = 0

    def recording_open_absolute(*args, **kwargs):
        descriptor = real_open_absolute(*args, **kwargs)
        opened_roots.append(descriptor)
        return descriptor

    def fail_second_fstat(*args, **kwargs):
        nonlocal fstat_calls
        fstat_calls += 1
        if fstat_calls == 2:
            raise source_io.SourceVerificationError("binding injected fstat failure")
        return real_fstat(*args, **kwargs)

    monkeypatch.setattr(source_io, "_open_absolute_binding", recording_open_absolute)
    monkeypatch.setattr(source_io, "_fstat", fail_second_fstat)

    with pytest.raises(ValueError, match="binding"):
        source_io._open_root(os.path.realpath(os.path.abspath(root)))

    assert len(opened_roots) == 2
    for descriptor in opened_roots:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize(
    "fault,category",
    [
        ("soundfile-constructor", "decode"),
        ("header", "header"),
        ("read", "decode"),
        ("short-read", "decode"),
        ("hook", "binding"),
    ],
)
def test_representative_owned_resource_failures_do_not_leak_fds(
    tmp_path, monkeypatch, fault, category,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    row = _row(root, "source.wav")
    real_soundfile = source_io.sf.SoundFile

    class FaultySoundFile:
        def __init__(self, *args, **kwargs):
            if fault == "soundfile-constructor":
                raise OSError("injected constructor failure")
            self._stream = real_soundfile(*args, **kwargs)

        def __getattr__(self, name):
            if fault == "header" and name == "samplerate":
                raise OSError("injected header failure")
            return getattr(self._stream, name)

        def read(self, *args, **kwargs):
            if fault == "read":
                raise OSError("injected read failure")
            audio = self._stream.read(*args, **kwargs)
            return audio[:-1] if fault == "short-read" else audio

        def close(self):
            self._stream.close()

    if fault == "hook":
        def fail_hook():
            raise OSError("injected hook failure")

        monkeypatch.setattr(source_io, "_before_decode_test_hook", fail_hook)
    else:
        monkeypatch.setattr(source_io.sf, "SoundFile", FaultySoundFile)

    baseline = _fd_count()
    for _ in range(10):
        with pytest.raises(ValueError, match=category):
            decode_verified_source(row)
    assert _fd_count() <= baseline + 1


def test_intermediate_open_failure_repeatedly_closes_parent_descriptors(tmp_path):
    root = tmp_path / "input"
    parent = root / "a"
    source = parent / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    row = _row(root, "a/source.wav")
    parent.rename(root / "moved-a")
    baseline = _fd_count()

    for _ in range(20):
        with pytest.raises(ValueError, match="binding"):
            decode_verified_source(row)

    assert _fd_count() <= baseline + 1


def test_pread_failure_has_sha256_diagnostic(tmp_path, monkeypatch):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))

    def fail_pread(*args, **kwargs):
        raise OSError("injected pread failure")

    monkeypatch.setattr(source_io.os, "pread", fail_pread)

    with pytest.raises(ValueError, match="sha256"):
        decode_verified_source(_row(root, "source.wav"))


def test_decode_dup_failure_has_decode_diagnostic(tmp_path, monkeypatch):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    source = tmp_path / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    descriptor = os.open(source, os.O_RDONLY)

    def fail_dup(*args, **kwargs):
        raise OSError("injected dup failure")

    monkeypatch.setattr(source_io.os, "dup", fail_dup)
    try:
        with pytest.raises(ValueError, match="decode"):
            source_io._decode_source_descriptor(descriptor)
    finally:
        os.close(descriptor)


class _BlockReadTestDouble:
    """按块语义返回预置全通道波形的 SoundFile test double。

    第 1 次 read 返回全部剩余帧（模拟单块覆盖小文件），之后返回空块；
    块数组不授予解码结果任何所有权，用于证明 Step1 只驻留 channel 0 的自有
    mono 数组。"""

    samplerate = 16000
    format = "WAV"
    subtype = "FLOAT"

    def __init__(self, payload, *args, **kwargs):
        self._payload = payload
        self.frames = int(payload.shape[0])
        self.channels = int(payload.shape[1])
        self._position = 0

    def read(self, frames=-1, *, dtype, always_2d):
        assert dtype == "float32"
        assert always_2d is True
        remaining = self.frames - self._position
        rows = remaining if frames is None or frames < 0 else min(int(frames), remaining)
        block = self._payload[self._position : self._position + rows]
        self._position += rows
        return block

    def close(self):
        pass


def test_decode_retains_only_first_channel_with_owned_mono_contract(monkeypatch):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    provided = np.arange(12, dtype=np.float32).reshape(6, 2).copy()

    monkeypatch.setattr(
        source_io.sf,
        "SoundFile",
        lambda *args, **kwargs: _BlockReadTestDouble(provided, *args, **kwargs),
    )
    descriptor = os.open("/dev/null", os.O_RDONLY)
    try:
        decoded, _metadata = source_io._decode_source_descriptor(descriptor)
    finally:
        os.close(descriptor)

    assert type(decoded) is np.ndarray
    assert decoded.dtype == np.float32
    assert decoded.shape == (6,)
    assert decoded.flags.owndata
    assert decoded.flags.c_contiguous
    assert decoded.flags.writeable
    assert decoded.base is None
    np.testing.assert_array_equal(decoded, provided[:, 0])


@pytest.mark.parametrize(
    "case",
    [
        "non-owned-c",
        "owned-f",
        "wrong-dtype",
        "readonly",
        "subclass",
        "close-readonly",
    ],
)
def test_decode_extracts_first_channel_from_each_unsafe_array(case, monkeypatch):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    expected = np.arange(12, dtype=np.float32).reshape(6, 2)
    if case == "non-owned-c":
        provided = np.empty((7, 2), dtype=np.float32)[:6]
        provided[:] = expected
    elif case == "owned-f":
        provided = np.asfortranarray(expected)
    elif case == "wrong-dtype":
        provided = expected.astype(np.float64)
    elif case == "readonly":
        provided = expected.copy()
        provided.flags.writeable = False
    elif case == "subclass":
        class ArraySubclass(np.ndarray):
            pass

        provided = ArraySubclass((6, 2), dtype=np.float32)
        provided[:] = expected
    else:
        provided = expected.copy()

    class UnsafeArraySoundFile:
        samplerate = 16000
        frames = 6
        channels = 2
        format = "WAV"
        subtype = "FLOAT"

        def __init__(self, *_args, **_kwargs):
            pass

        _first = True

        def read(self, frames=-1, *, dtype, always_2d):
            block = provided if type(self)._first else np.empty((0, 2), dtype=np.float32)
            type(self)._first = False
            return block

        def close(self):
            if case == "close-readonly":
                provided.flags.writeable = False

    monkeypatch.setattr(source_io.sf, "SoundFile", UnsafeArraySoundFile)
    descriptor = os.open("/dev/null", os.O_RDONLY)
    try:
        decoded, _metadata = source_io._decode_source_descriptor(descriptor)
    finally:
        os.close(descriptor)

    assert type(decoded) is np.ndarray
    assert decoded.dtype == np.float32
    assert decoded.flags.owndata
    assert decoded.flags.c_contiguous
    assert decoded.flags.writeable
    assert decoded.base is None
    np.testing.assert_array_equal(decoded, expected[:, 0])


def test_real_soundfile_decode_returns_owned_first_channel_mono_usable_after_close(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    source = tmp_path / "real.wav"
    expected = np.linspace(-0.75, 0.75, 64, dtype=np.float32)[:, None]
    _write_wav(source, expected[:, 0])
    real_read = source_io.sf.SoundFile.read
    real_close = source_io.sf.SoundFile.close
    captured = {}

    def recording_read(stream, *args, **kwargs):
        audio = real_read(stream, *args, **kwargs)
        captured["audio"] = audio
        return audio

    def recording_close(stream):
        real_close(stream)
        captured["closed"] = True

    monkeypatch.setattr(source_io.sf.SoundFile, "read", recording_read)
    monkeypatch.setattr(source_io.sf.SoundFile, "close", recording_close)
    descriptor = os.open(source, os.O_RDONLY)
    try:
        decoded, _metadata = source_io._decode_source_descriptor(descriptor)
    finally:
        os.close(descriptor)

    assert captured["closed"] is True
    assert type(decoded) is np.ndarray
    assert decoded.dtype == np.float32
    assert decoded.shape == (64,)
    assert decoded.flags.c_contiguous
    assert decoded.flags.owndata
    assert decoded.flags.writeable
    assert decoded.base is None
    np.testing.assert_allclose(decoded, captured["audio"][:, 0], atol=1 / 32768)
    decoded[0] = 0.125
    assert decoded[0] == pytest.approx(0.125)


def test_missing_o_nonblock_capability_fails_closed_with_binding_diagnostic(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(64, dtype=np.float32))
    row = _row(root, "source.wav")
    monkeypatch.delattr(source_io.os, "O_NONBLOCK")

    with pytest.raises(ValueError, match="binding.*O_NONBLOCK"):
        decode_verified_source(row)


def test_decode_resource_parameters_are_strict_positive_python_ints_and_frozen():
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    values = {
        "max_decoded_bytes": 4,
        "max_raw_source_bytes": 6,
        "max_vad_analysis_bytes": 8,
        "min_source_sample_rate": 8000,
        "max_source_sample_rate": 192000,
    }
    parameters = source_io.SourceDecodeParameters(**values)

    assert parameters.max_decoded_bytes == 4
    assert parameters.max_raw_source_bytes == 6
    assert parameters.max_vad_analysis_bytes == 8
    assert parameters.min_source_sample_rate == 8000
    assert parameters.max_source_sample_rate == 192000
    with pytest.raises(AttributeError):
        parameters.max_decoded_bytes = 8
    for field in values:
        for invalid in (True, 0, -1, 1.5, np.int64(1)):
            invalid_values = {**values, field: invalid}
            with pytest.raises(ValueError, match=f"{field}.*positive Python integer"):
                source_io.SourceDecodeParameters(**invalid_values)
    with pytest.raises(ValueError, match="sample rate range"):
        source_io.SourceDecodeParameters(
            **{
                **values,
                "min_source_sample_rate": 48000,
                "max_source_sample_rate": 16000,
            }
        )


def test_raw_source_byte_gate_allows_exact_boundary_and_rejects_before_read(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(16, dtype=np.float32))
    row = _row(root, "source.wav")
    source_size = row["source_size_bytes"]
    read_sizes = []
    real_spool = source_io._spool_verified_source

    def recording_spool(descriptor, expected_size, *, expected_sha256):
        read_sizes.append(expected_size)
        spool = real_spool(
            descriptor,
            expected_size,
            expected_sha256=expected_sha256,
        )
        assert spool._rolled is True
        return spool

    monkeypatch.setattr(source_io, "_spool_verified_source", recording_spool)
    decoded = decode_verified_source(
        row,
        parameters=_resource_parameters(max_raw_source_bytes=source_size),
    )
    assert decoded.frames == 16
    assert read_sizes == [source_size]

    read_sizes.clear()
    monkeypatch.setattr(
        source_io,
        "_decode_source_descriptor",
        lambda *_args, **_kwargs: pytest.fail("raw byte gate must precede decoder"),
    )
    with pytest.raises(ValueError, match="raw source bytes"):
        decode_verified_source(
            row,
            parameters=_resource_parameters(max_raw_source_bytes=source_size - 1),
        )
    assert read_sizes == []


def test_decode_budget_allows_exact_boundary_and_rejects_cap_plus_four_before_read(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "stereo.wav"
    _write_wav(source, np.zeros((6, 2), dtype=np.float32))
    row = _row(root, "stereo.wav")
    real_soundfile = source_io.sf.SoundFile
    read_calls = []

    class BudgetProbeSoundFile:
        def __init__(self, *args, **kwargs):
            self._stream = real_soundfile(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def read(self, *args, **kwargs):
            read_calls.append(True)
            return self._stream.read(*args, **kwargs)

        def close(self):
            self._stream.close()

    monkeypatch.setattr(source_io.sf, "SoundFile", BudgetProbeSoundFile)
    # 护栏口径为驻留 channel 0 mono（frames × float32），与源通道数无关。
    decoded_bytes = 6 * np.dtype(np.float32).itemsize
    baseline = _fd_count()

    decoded = decode_verified_source(
        row,
        parameters=_resource_parameters(max_decoded_bytes=decoded_bytes),
        vad_target_sample_rate=16000,
    )
    assert decoded.audio.shape == (6,)
    assert decoded.vad_capacity_samples == 6
    assert read_calls == [True]

    read_calls.clear()
    with pytest.raises(ValueError, match="source resource limit exceeded"):
        decode_verified_source(
            row,
            parameters=_resource_parameters(max_decoded_bytes=decoded_bytes - 4),
            vad_target_sample_rate=16000,
        )
    assert read_calls == []
    assert _fd_count() <= baseline + 1


def test_vad_capacity_preflight_identity_has_no_plus_one_before_read(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(6, dtype=np.float32), sample_rate=16000)
    row = _row(root, "source.wav")
    real_soundfile = source_io.sf.SoundFile
    read_calls = []

    class ReadProbeSoundFile:
        def __init__(self, *args, **kwargs):
            self._stream = real_soundfile(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def read(self, *args, **kwargs):
            read_calls.append(True)
            return self._stream.read(*args, **kwargs)

        def close(self):
            self._stream.close()

    monkeypatch.setattr(source_io.sf, "SoundFile", ReadProbeSoundFile)
    decoded = decode_verified_source(
        row,
        parameters=_resource_parameters(max_vad_analysis_bytes=6 * 4),
        vad_target_sample_rate=16000,
    )

    assert decoded.vad_capacity_samples == 6
    assert read_calls == [True]


def test_vad_capacity_preflight_nonidentity_uses_exact_ceil_ratio_plus_one(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(3, dtype=np.float32), sample_rate=8000)
    row = _row(root, "source.wav")
    real_soundfile = source_io.sf.SoundFile
    read_calls = []

    class ReadProbeSoundFile:
        def __init__(self, *args, **kwargs):
            self._stream = real_soundfile(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def read(self, *args, **kwargs):
            read_calls.append(True)
            return self._stream.read(*args, **kwargs)

        def close(self):
            self._stream.close()

    monkeypatch.setattr(source_io.sf, "SoundFile", ReadProbeSoundFile)
    decoded = decode_verified_source(
        row,
        parameters=_resource_parameters(max_vad_analysis_bytes=7 * 4),
        vad_target_sample_rate=16000,
    )
    assert decoded.vad_capacity_samples == 7
    assert read_calls == [True]

    read_calls.clear()
    baseline = _fd_count()
    with pytest.raises(ValueError, match="VAD.*resource limit exceeded"):
        decode_verified_source(
            row,
            parameters=_resource_parameters(max_vad_analysis_bytes=6 * 4),
            vad_target_sample_rate=16000,
        )
    assert read_calls == []
    assert _fd_count() <= baseline + 1


@pytest.mark.parametrize("sample_rate", [7999, 192001])
def test_source_sample_rate_outside_envelope_rejects_before_read(
    tmp_path, monkeypatch, sample_rate,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(8, dtype=np.float32), sample_rate=16000)
    row = _row(root, "source.wav")
    row["sample_rate"] = sample_rate
    row["duration_sec"] = row["frames"] / sample_rate
    real_soundfile = source_io.sf.SoundFile
    read_calls = []

    class HeaderProbeSoundFile:
        def __init__(self, *args, **kwargs):
            self._stream = real_soundfile(*args, **kwargs)
            self.samplerate = sample_rate

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def read(self, *args, **kwargs):
            read_calls.append(True)
            raise AssertionError("out-of-range sample rate must not read waveform")

        def close(self):
            self._stream.close()

    monkeypatch.setattr(source_io.sf, "SoundFile", HeaderProbeSoundFile)
    with pytest.raises(ValueError, match="source sample rate.*range"):
        decode_verified_source(row, vad_target_sample_rate=16000)
    assert read_calls == []


def test_header_mismatch_precedes_resource_preflight(tmp_path, monkeypatch):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(6, dtype=np.float32), sample_rate=16000)
    row = _row(root, "source.wav")
    real_soundfile = source_io.sf.SoundFile
    read_calls = []

    class MismatchedHugeHeaderSoundFile:
        def __init__(self, *args, **kwargs):
            self._stream = real_soundfile(*args, **kwargs)
            self.frames = self._stream.frames + 1
            self.samplerate = 1

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def read(self, *args, **kwargs):
            read_calls.append(True)
            raise AssertionError("mismatched header must not read waveform")

        def close(self):
            self._stream.close()

    monkeypatch.setattr(source_io.sf, "SoundFile", MismatchedHugeHeaderSoundFile)
    with pytest.raises(ValueError, match="header sample_rate mismatch"):
        decode_verified_source(
            row,
            parameters=_resource_parameters(max_decoded_bytes=1),
            vad_target_sample_rate=16000,
        )
    assert read_calls == []


def test_vad_target_sample_rate_requires_strict_positive_python_int(tmp_path):
    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(6, dtype=np.float32), sample_rate=16000)
    row = _row(root, "source.wav")

    for invalid in (True, 0, -1, 1.5, np.int64(16000)):
        with pytest.raises(ValueError, match="vad_target_sample_rate.*positive Python integer"):
            decode_verified_source(row, vad_target_sample_rate=invalid)


def test_verified_header_mismatch_is_rejected_before_soundfile_read(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    _write_wav(source, np.zeros(6, dtype=np.float32))
    row = _row(root, "source.wav")
    real_soundfile = source_io.sf.SoundFile
    read_calls = []

    class MismatchedHeaderSoundFile:
        def __init__(self, *args, **kwargs):
            self._stream = real_soundfile(*args, **kwargs)
            self.frames = self._stream.frames + 1

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def read(self, *args, **kwargs):
            read_calls.append(True)
            return self._stream.read(*args, **kwargs)

        def close(self):
            self._stream.close()

    monkeypatch.setattr(source_io.sf, "SoundFile", MismatchedHeaderSoundFile)

    with pytest.raises(ValueError, match="header frames mismatch"):
        decode_verified_source(row)

    assert read_calls == []


def test_verified_source_session_decodes_exact_half_open_stereo_interval(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.source_io import (
        open_verified_source_session,
    )

    root = tmp_path / "input"
    expected = np.arange(48, dtype=np.float32).reshape(24, 2) / 64
    source = root / "source.wav"
    _write_wav(source, expected)

    with open_verified_source_session(_row(root, "source.wav")) as session:
        assert session.sample_rate == 16_000
        assert session.frames == 24
        assert session.channels == 2
        decoded, sample_rate = session.decode_interval(5, 17)

    assert sample_rate == 16_000
    assert decoded.shape == (12, 2)
    assert decoded.dtype == np.float32
    assert decoded.flags.c_contiguous
    assert decoded.flags.owndata
    assert decoded.base is None
    np.testing.assert_allclose(decoded, expected[5:17], atol=1 / 32768)


@pytest.mark.parametrize(
    "start,end",
    [(True, 2), (0, True), (-1, 2), (2, 2), (3, 2), (0, 25)],
)
def test_verified_source_session_rejects_invalid_half_open_interval(tmp_path, start, end):
    from audio_filter_pipeline.processors.stages_v3.source_io import (
        open_verified_source_session,
    )

    root = tmp_path / "input"
    _write_wav(root / "source.wav", np.zeros(24, dtype=np.float32))

    with open_verified_source_session(_row(root, "source.wav")) as session:
        with pytest.raises(ValueError, match="interval.*half-open"):
            session.decode_interval(start, end)


def test_verified_source_session_hashes_and_validates_header_only_at_open(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    _write_wav(root / "source.wav", np.zeros(24, dtype=np.float32))
    row = _row(root, "source.wav")
    real_hash = source_io._hash_source_descriptor
    real_header = source_io._read_verified_header
    calls = []

    def recording_hash(*args, **kwargs):
        calls.append("hash")
        return real_hash(*args, **kwargs)

    def recording_header(*args, **kwargs):
        calls.append("header")
        return real_header(*args, **kwargs)

    monkeypatch.setattr(source_io, "_hash_source_descriptor", recording_hash)
    monkeypatch.setattr(source_io, "_read_verified_header", recording_header)

    with source_io.open_verified_source_session(row) as session:
        session.decode_interval(0, 4)
        session.decode_interval(4, 8)

    assert calls == ["hash", "header"]


def test_verified_source_session_decodes_held_source_then_rejects_named_swap(
    tmp_path,
):
    from audio_filter_pipeline.processors.stages_v3.source_io import (
        open_verified_source_session,
    )

    root = tmp_path / "input"
    source = root / "source.wav"
    replacement = root / "replacement.wav"
    original = np.full(24, 0.25, dtype=np.float32)
    _write_wav(source, original)
    _write_wav(replacement, np.full(24, -0.75, dtype=np.float32))
    row = _row(root, "source.wav")

    with pytest.raises(ValueError, match="binding"):
        with open_verified_source_session(row) as session:
            source.rename(root / "held-original.wav")
            replacement.rename(source)
            decoded, _sample_rate = session.decode_interval(3, 9)
            np.testing.assert_allclose(decoded[:, 0], original[3:9], atol=1 / 32768)


def test_verified_source_session_short_interval_read_closes_duplicate_fd(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    _write_wav(root / "source.wav", np.zeros(24, dtype=np.float32))
    row = _row(root, "source.wav")
    real_soundfile = source_io.sf.SoundFile
    baseline = _fd_count()

    class ShortReadSoundFile:
        def __init__(self, *args, **kwargs):
            self._stream = real_soundfile(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def read(self, *args, **kwargs):
            return self._stream.read(*args, **kwargs)[:-1]

        def close(self):
            self._stream.close()

    monkeypatch.setattr(source_io.sf, "SoundFile", ShortReadSoundFile)
    with source_io.open_verified_source_session(row) as session:
        with pytest.raises(ValueError, match="interval decode was short"):
            session.decode_interval(2, 10)
    assert _fd_count() <= baseline + 1


def test_step6_interval_decode_uses_verified_row_api(tmp_path):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    waveform = np.arange(24, dtype=np.float32) / 24.0
    _write_wav(root / "source.wav", waveform)
    row = _row(root, "source.wav")
    row.update({"start_sample": 2, "end_sample": 10})

    audio, sample_rate = source_io.read_verified_source_interval(row)

    assert sample_rate == 16_000
    assert audio.shape == (8, 1)


def test_verified_source_session_rejects_interval_over_byte_limit_before_read(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    _write_wav(root / "source.wav", np.zeros(24, dtype=np.float32))
    row = _row(root, "source.wav")
    read_calls = []
    over_limit_frames = source_io.MAX_STEP3_INTERVAL_DECODED_BYTES // 4 + 1

    with source_io.open_verified_source_session(row) as session:
        assert session._metadata is not None
        session._metadata["frames"] = over_limit_frames
        session._metadata["channels"] = 1
        monkeypatch.setattr(
            source_io.sf,
            "SoundFile",
            lambda *_args, **_kwargs: pytest.fail("interval decoder must not open"),
        )
        with pytest.raises(ValueError, match="Step3 interval resource limit exceeded") as caught:
            session.decode_interval(0, over_limit_frames)

    assert str(source_io.MAX_STEP3_INTERVAL_DECODED_BYTES) in str(caught.value)
    assert str(over_limit_frames * 4) in str(caught.value)
    assert read_calls == []


def test_verified_source_session_allows_exact_interval_byte_limit(tmp_path, monkeypatch):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    _write_wav(root / "source.wav", np.zeros(24, dtype=np.float32))
    row = _row(root, "source.wav")
    frame_count = source_io.MAX_STEP3_INTERVAL_DECODED_BYTES // 4
    read_calls = []

    class ExactLimitSoundFile:
        def __init__(self, *_args, **_kwargs):
            pass

        def seek(self, offset):
            return offset

        def read(self, frames, *, dtype, always_2d):
            read_calls.append((frames, dtype, always_2d))
            return np.zeros((frames, 1), dtype=np.float32)

        def close(self):
            pass

    with source_io.open_verified_source_session(row) as session:
        assert session._metadata is not None
        session._metadata["frames"] = frame_count
        session._metadata["channels"] = 1
        monkeypatch.setattr(source_io.sf, "SoundFile", ExactLimitSoundFile)
        decoded, sample_rate = session.decode_interval(0, frame_count)

    assert sample_rate == 16_000
    assert decoded.shape == (frame_count, 1)
    assert read_calls == [(frame_count, "float32", True)]


def test_session_exit_preserves_body_error_and_resets_state_after_binding_failure(
    tmp_path,
):
    import audio_filter_pipeline.processors.stages_v3.source_io as source_io

    root = tmp_path / "input"
    source = root / "source.wav"
    replacement = root / "replacement.wav"
    _write_wav(source, np.zeros(24, dtype=np.float32))
    _write_wav(replacement, np.ones(24, dtype=np.float32))
    session = source_io.open_verified_source_session(_row(root, "source.wav"))

    with pytest.raises(RuntimeError, match="body failure") as caught:
        with session:
            source.rename(root / "held-source.wav")
            replacement.rename(source)
            raise RuntimeError("body failure")

    assert any("session exit binding verification" in note for note in caught.value.__notes__)
    with pytest.raises(ValueError, match="session is not open"):
        session.decode_interval(0, 1)


def test_streaming_flac_aggregate_source_io_decodes_full_and_cross_member_interval(tmp_path):
    root = tmp_path / "input"
    aggregate = root / "utterance_active"
    first = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32)
    second = np.array([-0.40, -0.50, -0.60, -0.70, -0.80], dtype=np.float32)
    _write_flac(aggregate / "utterance_active_0000.flac", first)
    _write_flac(aggregate / "utterance_active_0001.flac", second)
    spec = DatasetSpec(
        dataset_id="alpha",
        input_root=str(root),
        language="en",
        category="read_speech",
        quality_profile="quality_high",
        speaker=SpeakerMetadataSpec(source="none", required=False),
        source_layout="streaming_flac_aggregate",
    )
    config = PipelineConfig(
        pipeline_generation=3,
        step="step0",
        output_root=str(tmp_path / "out"),
        datasets={
            "alpha": {
                "input_root": str(root),
                "language": "en",
                "category": "read_speech",
                "quality_profile": "quality_high",
                "speaker": {"source": "none", "required": False},
                "source_layout": "streaming_flac_aggregate",
            },
        },
        selection_policies={
            "global": {
                "mos_models": ["dnsmos"],
                "thresholds": {"dnsmos": 3.0},
                "min_mos_models": 1,
            },
            "quality_profiles": {
                "quality_balanced": {"thresholds": {"dnsmos": {"ovrl": 3.2, "sig": 3.0, "bak": 3.8, "p808": 3.6}, "scoreq": 3.6, "urgent_pk": 3.5}},
                "quality_high": {"thresholds": {"dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8}, "scoreq": 4.2, "urgent_pk": 3.5}},
                "quality_premium": {"thresholds": {"dnsmos": {"ovrl": 3.4, "sig": 3.6, "bak": 4.2, "p808": 4.0}, "scoreq": 4.5, "urgent_pk": 3.5}},
            },
        },
    )
    step0_index = Step0IngestStage(spec).run(
        StageRunContext(str(config.output_root), "alpha", config)
    )
    row = next(iter_artifact_rows(step0_index))
    expected = np.concatenate((first, second))

    decoded = decode_verified_source(row)
    assert decoded.frames == len(expected)
    assert decoded.sample_rate == 48_000
    assert decoded.channels == 1
    assert decoded.audio.shape == (len(expected),)
    assert decoded.audio.dtype == np.float32
    assert decoded.audio.flags.c_contiguous
    assert decoded.audio.flags.owndata
    assert decoded.audio.flags.writeable
    np.testing.assert_allclose(decoded.audio, expected, atol=1 / 32768)

    with open_verified_source_session(row) as session:
        interval, sample_rate = session.decode_interval(2, 7)

    assert sample_rate == 48_000
    assert interval.shape == (5, 1)
    assert interval.dtype == np.float32
    assert interval.flags.c_contiguous
    assert interval.flags.owndata
    assert interval.flags.writeable
    np.testing.assert_allclose(interval[:, 0], expected[2:7], atol=1 / 32768)


def test_aggregate_full_decode_reuses_initial_manifest_and_returns_owned_array(
    tmp_path, monkeypatch,
):
    row, expected, _aggregate = _aggregate_row(tmp_path)
    real_scan = aggregate_source._scan_aggregate_directory
    scan_calls = []

    def count_scan(*args, **kwargs):
        scan_calls.append(True)
        return real_scan(*args, **kwargs)

    monkeypatch.setattr(aggregate_source, "_scan_aggregate_directory", count_scan)
    decoded = decode_verified_source(row)

    assert len(scan_calls) == 1
    assert decoded.audio.shape == (len(expected),)
    assert decoded.audio.dtype == np.float32
    assert decoded.audio.flags.c_contiguous
    assert decoded.audio.flags.owndata
    assert decoded.audio.flags.writeable
    assert decoded.audio.base is None
    np.testing.assert_allclose(decoded.audio, expected, atol=1 / 32768)


def test_aggregate_session_reuses_manifest_and_rechecks_only_snapshots(
    tmp_path, monkeypatch,
):
    row, expected, _aggregate = _aggregate_row(tmp_path)
    real_scan = aggregate_source._scan_aggregate_directory
    scan_calls = []

    def count_scan(*args, **kwargs):
        scan_calls.append(True)
        return real_scan(*args, **kwargs)

    monkeypatch.setattr(aggregate_source, "_scan_aggregate_directory", count_scan)
    with open_verified_source_session(row) as session:
        first, _ = session.decode_interval(0, 3)
        second, _ = session.decode_interval(3, len(expected))

    assert len(scan_calls) == 1
    np.testing.assert_allclose(first[:, 0], expected[:3], atol=1 / 32768)
    np.testing.assert_allclose(second[:, 0], expected[3:], atol=1 / 32768)
    for decoded in (first, second):
        assert decoded.shape[1] == 1
        assert decoded.dtype == np.float32
        assert decoded.flags.c_contiguous
        assert decoded.flags.owndata
        assert decoded.flags.writeable
        assert decoded.base is None


def test_aggregate_session_rejects_member_snapshot_change_without_rescan(tmp_path):
    row, _expected, _aggregate = _aggregate_row(tmp_path)
    member_path = Path(row["input_root"]) / "utterance_active" / (
        "utterance_active_0000.flac"
    )
    original = member_path.stat()

    with pytest.raises(ValueError, match="binding"):
        with open_verified_source_session(row) as session:
            os.utime(
                member_path,
                ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000),
            )
            session.decode_interval(0, 2)


def test_aggregate_member_frame_limit_allows_exact_eof_boundary(tmp_path, monkeypatch):
    root = tmp_path / "input"
    aggregate = root / "utterance_active"
    samples = np.arange(7, dtype=np.float32) / 10
    member_path = aggregate / "utterance_active_0000.flac"
    _write_flac(member_path, samples)
    directory_descriptor = os.open(aggregate, os.O_RDONLY | os.O_DIRECTORY)
    monkeypatch.setattr(
        aggregate_source, "AGGREGATE_MAX_MEMBER_FRAMES", len(samples),
    )
    try:
        descriptor = aggregate_source.scan_aggregate_directory(
            directory_descriptor,
            str(root.resolve()),
            "utterance_active",
        )
    finally:
        os.close(directory_descriptor)

    assert descriptor.frames == len(samples)


def test_streaming_flac_unknown_frame_count_scans_and_decodes_exact_eof(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(aggregate_source, "AGGREGATE_READ_CHUNK_FRAMES", 257)
    root = tmp_path / "input"
    aggregate = root / "utterance_active"
    samples = np.linspace(-0.8, 0.8, 4_097, dtype=np.float32)
    member_path = aggregate / "utterance_active_0000.flac"
    _write_flac(member_path, samples)
    payload = bytearray(member_path.read_bytes())
    assert payload[:4] == b"fLaC"
    assert payload[4] & 0x7F == 0
    assert int.from_bytes(payload[5:8], "big") == 34
    total_samples_offset = 8 + 10
    packed = int.from_bytes(
        payload[total_samples_offset:total_samples_offset + 8], "big",
    )
    payload[total_samples_offset:total_samples_offset + 8] = (
        packed & ~((1 << 36) - 1)
    ).to_bytes(8, "big")
    member_path.write_bytes(payload)

    directory_descriptor = os.open(aggregate, os.O_RDONLY | os.O_DIRECTORY)
    try:
        descriptor = aggregate_source.scan_aggregate_directory(
            directory_descriptor,
            str(root.resolve()),
            "utterance_active",
        )
        assert descriptor.frames == len(samples)
        decoded = aggregate_source.decode_aggregate_full(
            directory_descriptor,
            descriptor,
            full_scan=False,
            verify_bindings=True,
        )
        interval = aggregate_source.decode_aggregate_interval(
            directory_descriptor,
            descriptor,
            1_234,
            3_456,
            full_scan=False,
            verify_bindings=True,
        )
    finally:
        os.close(directory_descriptor)

    np.testing.assert_allclose(decoded[:, 0], samples, atol=1 / 32768)
    np.testing.assert_allclose(interval[:, 0], samples[1_234:3_456], atol=1 / 32768)


def test_streaming_flac_aggregate_rejects_metadata_only_member(tmp_path):
    root = tmp_path / "input"
    aggregate = root / "utterance_active"
    member = aggregate / "utterance_active_0000.flac"
    _write_flac(member, np.linspace(-0.5, 0.5, 8, dtype=np.float32))
    _truncate_flac_after_metadata(member)

    directory_descriptor = os.open(aggregate, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(aggregate_source.AggregateMemberEmptyError) as caught:
            aggregate_source.scan_aggregate_directory(
                directory_descriptor,
                str(root.resolve()),
                "utterance_active",
            )
    finally:
        os.close(directory_descriptor)

    assert caught.value.member_name == "utterance_active_0000.flac"
    assert caught.value.aggregate_relative_path == "utterance_active"


def test_streaming_flac_scan_uses_actual_frames_not_final_child_total_samples(tmp_path):
    root = tmp_path / "input"
    aggregate = root / "utterance_active"
    first = np.linspace(-0.5, 0.5, 4, dtype=np.float32)
    second = np.linspace(0.5, 0.9, 5, dtype=np.float32)
    first_path = aggregate / "utterance_active_0000.flac"
    second_path = aggregate / "utterance_active_0001.flac"
    _write_flac(first_path, first)
    _write_flac(second_path, second)
    _set_flac_total_samples(second_path, len(first) + len(second))

    directory_descriptor = os.open(aggregate, os.O_RDONLY | os.O_DIRECTORY)
    try:
        descriptor = aggregate_source.scan_aggregate_directory(
            directory_descriptor,
            str(root.resolve()),
            "utterance_active",
        )
    finally:
        os.close(directory_descriptor)

    assert [member.frames for member in descriptor.members] == [len(first), len(second)]
    assert descriptor.frames == len(first) + len(second)


def test_aggregate_session_reuses_sequential_member_stream(tmp_path, monkeypatch):
    row, expected, _aggregate = _aggregate_row(tmp_path)
    real_soundfile = aggregate_source.sf.SoundFile
    opened = []

    def counting_soundfile(*args, **kwargs):
        opened.append(True)
        return real_soundfile(*args, **kwargs)

    with open_verified_source_session(row) as session:
        monkeypatch.setattr(aggregate_source.sf, "SoundFile", counting_soundfile)
        first, _ = session.decode_interval(0, 2)
        second, _ = session.decode_interval(2, 4)

    assert len(opened) == 1
    np.testing.assert_allclose(first[:, 0], expected[:2], atol=1 / 32768)
    np.testing.assert_allclose(second[:, 0], expected[2:4], atol=1 / 32768)


def test_aggregate_member_binding_error_is_not_allowed_to_replace_read_error(
    tmp_path, monkeypatch,
):
    row, _expected, aggregate = _aggregate_row(tmp_path)
    directory = Path(row["input_root"]) / "utterance_active"
    directory_descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    real_soundfile = aggregate_source.sf.SoundFile

    class ReadFailureSoundFile:
        def __init__(self, *args, **kwargs):
            self._stream = real_soundfile(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def read(self, *args, **kwargs):
            raise ValueError("primary interval read failure")

        def close(self):
            self._stream.close()

    monkeypatch.setattr(aggregate_source.sf, "SoundFile", ReadFailureSoundFile)
    monkeypatch.setattr(
        aggregate_source,
        "_verify_member_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("binding cleanup failure")
        ),
    )
    try:
        with pytest.raises(ValueError, match="primary interval read failure") as caught:
            aggregate_source._read_member_interval(
                directory_descriptor, aggregate.members[0], 0, 2,
            )
    finally:
        os.close(directory_descriptor)

    assert any(
        "binding cleanup failure" in note
        for note in getattr(caught.value, "__notes__", ())
    )
