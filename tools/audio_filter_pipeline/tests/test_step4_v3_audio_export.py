"""Generation-3 Step4 MOS keep 段的 SE 干净数据集导出测试。"""

from __future__ import annotations

import errno
import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter
from audio_filter_pipeline.processors.stages_v3.mos_filter import STEP4_ROW_FIELDS
from audio_filter_pipeline.tests.audio_export_test_helpers import (
    install_export_rows_stub,
)


pytestmark = pytest.mark.unit
_MOS_SELECTION_FIELDS = (
    "decision", "passed_models", "required_models", "policy_id", "policy_hash",
)


@pytest.fixture(autouse=True)
def _export_rows_stub(monkeypatch):
    install_export_rows_stub(monkeypatch, Step4AudioExporter)


def _mos_selection(**overrides: object) -> dict[str, object]:
    selection: dict[str, object] = {
        "decision": "keep",
        "passed_models": 3,
        "required_models": 3,
        "policy_id": "alpha:en",
        "policy_hash": "h" * 64,
    }
    assert tuple(selection) == _MOS_SELECTION_FIELDS
    selection.update(overrides)
    return selection


def _step4_row(source: Path, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_key": "source-1",
        "dataset_id": "alpha",
        "language": "en",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "input_root": str(source.parent.resolve()),
        "original_path": str(source),
        "original_relative_path": source.name,
        "source_relative_path": source.name,
        "source_size_bytes": source.stat().st_size,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sample_rate": 16_000,
        "frames": 16_000,
        "channels": 1,
        "duration_sec": 1.0,
        "format": "WAV",
        "subtype": "FLOAT",
        "speaker_id": None,
        "speaker_scope": None,
        "source_start_sample": 0,
        "source_end_sample": 16_000,
        "signal_finite": True,
        "signal_zero_ratio": 0.0,
        "signal_peak": 0.5,
        "signal_clipping_ratio": 0.0,
        "signal_dc_offset": 0.0,
        "signal_rms_dbfs": -12.0,
        "signal_num_samples": 16_000,
        "bandwidth_hz": 3_000.0,
        "estimated_sr": 16_000,
        "vad_speech_ratio": 1.0,
        "vad_timeline_sample_rate": 16_000,
        "vad_timeline_num_samples": 16_000,
        "segment_id": "segment-1",
        "start_sample": 4_000,
        "end_sample": 12_000,
        "boundary_reason": "silence",
        "mos_models": ["scoreq", "urgent_pk", "dnsmos"],
        "mos_scoring_policy_hash": "g" * 64,
        "mos_scores": {
            "scoreq": 4.5,
            "urgent_pk": 4.0,
            "dnsmos": {"ovrl": 4.0, "sig": 4.0, "bak": 4.0, "p808": 4.0},
        },
        "mos_selection": _mos_selection(),
    }
    assert tuple(row) == STEP4_ROW_FIELDS
    row.update(overrides)
    return row


def _write_source(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)
    sf.write(path, samples, 16_000, subtype="FLOAT")


def test_step4_exporter_writes_keep_segment_at_routed_rate_under_v3_layout(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)

    result = Step4AudioExporter(str(tmp_path / "out"))._export_test_rows([
        _step4_row(source),
    ])

    assert len(result) == 1
    exported = result[0]
    destination = Path(exported.output_path)
    assert destination.is_file()
    relative_parts = destination.relative_to(tmp_path / "out").parts
    assert relative_parts[:3] == ("audio_exports", "step4", "alpha")
    assert len(relative_parts[3]) == 64
    assert relative_parts[4:] == ("16000hz", "source--segment-1.flac")
    assert exported.segment_id == "segment-1"
    assert exported.target_sample_rate == 16_000
    audio, sample_rate = sf.read(destination, dtype="float32", always_2d=True)
    assert sample_rate == 16_000
    assert audio.shape == (8_000, 1)
    info = sf.info(destination)
    assert info.format == "FLAC"
    assert info.subtype == "PCM_24"


@pytest.mark.parametrize(
    ("source_sample_rate", "bandwidth_hz"), [(8_000, 3_900.0), (11_025, 5_000.0)],
)
def test_step4_exporter_keeps_sub_16k_sources_at_native_rate(
        tmp_path, source_sample_rate, bandwidth_hz):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "source.wav"
    _write_source(source)
    row = _step4_row(
        source, sample_rate=source_sample_rate,
        bandwidth_hz=bandwidth_hz, estimated_sr=source_sample_rate,
    )

    assert Step4AudioExporter(str(tmp_path / "out"))._routed_sample_rate(row) == (
        source_sample_rate
    )


def test_step4_exporter_skips_rejected_rows(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)

    result = Step4AudioExporter(str(tmp_path / "out"))._export_test_rows([
        _step4_row(source, mos_selection=_mos_selection(
            decision="reject", passed_models=2,
        )),
    ])

    assert result == ()


def test_step4_exporter_rejects_legacy_rejected_artifact_before_selection(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    current = _step4_row(
        source, mos_selection=_mos_selection(decision="reject", passed_models=2),
    )
    legacy_fields = (
        "dataset_id", "source_key", "segment_id", "input_root", "original_path",
        "source_relative_path", "source_size_bytes", "source_sha256", "sample_rate",
        "frames", "channels", "duration_sec", "format", "subtype", "start_sample",
        "end_sample", "bandwidth_hz", "estimated_sr", "mos_selection",
    )
    legacy_row = {field: current[field] for field in legacy_fields}

    with pytest.raises(ValueError, match="step4 row fields.*exact contract"):
        Step4AudioExporter(str(tmp_path / "out"))._export_test_rows([legacy_row])


def test_step4_exporter_rejects_duplicate_but_preserves_legacy_layout(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    exporter = Step4AudioExporter(str(tmp_path / "out"))
    row = _step4_row(source)

    with pytest.raises(ValueError, match="duplicate destination"):
        exporter._export_test_rows([row, row])

    destination = (
        tmp_path / "out" / "audio_exports" / "step4" / "alpha" / "16000hz"
        / "source--segment-1.flac"
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"must remain untouched")
    exported = exporter._export_test_rows([row])
    assert Path(exported[0].output_path).is_file()
    assert Path(exported[0].output_path) != destination
    assert destination.read_bytes() == b"must remain untouched"


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        pytest.param("missing", r"missing=\['bandwidth_hz'\], extra=\[\]", id="missing"),
        pytest.param(
            "extra", r"missing=\[\], extra=\['legacy_passthrough'\]", id="extra",
        ),
    ],
)
def test_step4_exporter_fails_closed_on_nonexact_top_level_fields(
    tmp_path, mutation, error,
):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    row = _step4_row(source)
    if mutation == "missing":
        row.pop("bandwidth_hz")
    else:
        row["legacy_passthrough"] = {"opaque": True}

    with pytest.raises(ValueError, match=error):
        Step4AudioExporter(str(tmp_path / "out"))._export_test_rows([row])


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_step4_exporter_fails_closed_on_nonexact_mos_selection_fields(
    tmp_path, mutation,
):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    selection = _mos_selection()
    if mutation == "missing":
        selection.pop("policy_hash")
    else:
        selection["legacy"] = True

    with pytest.raises(ValueError, match="mos_selection fields.*exact contract"):
        Step4AudioExporter(str(tmp_path / "out"))._export_test_rows([
            _step4_row(source, mos_selection=selection),
        ])


def test_step4_exporter_fails_closed_on_invalid_decision_or_bandwidth(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    exporter = Step4AudioExporter(str(tmp_path / "out"))

    with pytest.raises(ValueError, match="mos_selection decision"):
        exporter._export_test_rows([
            _step4_row(source, mos_selection=_mos_selection(decision="maybe")),
        ])
    with pytest.raises(ValueError, match="bandwidth_hz"):
        exporter._export_test_rows([_step4_row(source, bandwidth_hz=9_000.0)])
    with pytest.raises(ValueError, match="estimated_sr"):
        exporter._export_test_rows([_step4_row(source, estimated_sr=20_000)])


@pytest.mark.parametrize(
    "bandwidth_hz",
    [True, float("nan"), float("inf"), -float("inf"), -0.1, 8_000.1],
)
def test_step4_exporter_rejects_invalid_real_bandwidth_values(
    tmp_path, bandwidth_hz
):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    with pytest.raises(ValueError, match="bandwidth_hz"):
        Step4AudioExporter(str(tmp_path / "out"))._validate_row(
            _step4_row(source, bandwidth_hz=bandwidth_hz)
        )


@pytest.mark.parametrize("bandwidth_hz", [3_000, Fraction(6_000, 2), np.float32(3_000)])
def test_step4_exporter_accepts_finite_non_bool_real_bandwidth(
    tmp_path, bandwidth_hz
):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    exporter = Step4AudioExporter(str(tmp_path / "out"))
    exporter._validate_row(_step4_row(source, bandwidth_hz=bandwidth_hz))

    # 同一数值的 bool 伪装必须被拒绝，证明上面的通过不是校验被跳过。
    with pytest.raises(ValueError, match="bandwidth_hz"):
        exporter._validate_row(
            _step4_row(source, bandwidth_hz=bool(bandwidth_hz))
        )


def _step4_destination(root: Path) -> Path:
    matches = list((root / "audio_exports").glob(
        "step4/alpha/*/16000hz/source--segment-1.flac"
    ))
    assert len(matches) == 1
    return matches[0]


def test_step4_exporter_resume_skips_decode_and_write_after_manifest_publication(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "input" / "source.wav"
    _write_source(source)
    output_root = tmp_path / "out"
    row = _step4_row(source)
    first = Step4AudioExporter(str(output_root), workers=1)._export_test_rows([row])

    def should_not_decode(_row):
        raise AssertionError("resume must not decode")

    def should_not_write(_path, _audio, _sample_rate):
        raise AssertionError("resume must not write")

    resumed = Step4AudioExporter(
        str(output_root), decoder=should_not_decode, writer=should_not_write, workers=1,
    )._export_test_rows([row])

    assert resumed == first
    assert resumed[0].output_path == str(_step4_destination(output_root))
    assert len(list((output_root / "audio_exports").glob(
        "step4/alpha/*/_export_manifest.v2.jsonl"
    ))) == 1


def test_staging_owner_reclaims_orphan_owner_temp_before_creation(tmp_path):
    exporter = Step4AudioExporter(str(tmp_path / "out"))
    destination = tmp_path / "out" / "audio_exports" / "step4" / "alpha" / "digest" / "16000hz" / "segment.flac"
    staging_dir = exporter._staging_dir(destination)
    staging_dir.mkdir(parents=True)
    orphan = staging_dir / (".owner-" + "a" * 32 + ".tmp")
    orphan.write_bytes(b"orphan")

    token = exporter._create_staging_owner(destination)

    assert len(token) == 64
    assert (staging_dir / f".owner-{token}.json").is_file()
    assert not orphan.exists()


def test_step4_exporter_explicit_root_must_not_overlap_input_root(tmp_path):
    from audio_filter_pipeline.processors.stages_v3.audio_export import Step4AudioExporter

    source = tmp_path / "input" / "source.wav"
    source.parent.mkdir()
    _write_source(source)

    with pytest.raises(ValueError, match="input_root"):
        Step4AudioExporter(
            str(tmp_path / "out"), export_root=str(source.parent / "exports"), workers=1,
        )._export_test_rows([_step4_row(source)])


class _FakeResource:
    """可编程的 resource 替身：记录 setrlimit 调用，不对当前进程生效。

    常量名与真实 ``resource`` 模块一致：限制名用 RLIMIT_* 前缀，
    无穷大是 RLIM_INFINITY。
    """

    RLIMIT_NOFILE = 7
    RLIM_INFINITY = 1 << 63

    def __init__(self, soft: int, hard: int) -> None:
        self._soft = soft
        self._hard = hard
        self.raised: list[tuple[int, int]] = []

    def getrlimit(self, which: int) -> tuple[int, int]:
        assert which == self.RLIMIT_NOFILE
        return (self._soft, self._hard)

    def setrlimit(self, which: int, value: tuple[int, int]) -> None:
        assert which == self.RLIMIT_NOFILE
        self.raised.append(value)


def _patch_resource(monkeypatch, soft: int, hard: int) -> _FakeResource:
    from audio_filter_pipeline.processors.stages_v3 import audio_export

    fake = _FakeResource(soft, hard)
    monkeypatch.setattr(audio_export, "resource", fake)
    monkeypatch.setattr(audio_export, "_count_open_fds", lambda: 0)
    return fake


def test_export_nofile_preflight_passes_when_soft_sufficient(monkeypatch):
    from audio_filter_pipeline.processors.stages_v3.audio_export import (
        preflight_export_nofile,
    )

    fake = _patch_resource(monkeypatch, soft=1048576, hard=1048576)
    preflight_export_nofile(32, max_pending=64)
    assert fake.raised == []


def test_export_nofile_preflight_raises_soft_limit_within_hard(monkeypatch):
    from audio_filter_pipeline.processors.stages_v3.audio_export import (
        preflight_export_nofile,
    )

    fake = _patch_resource(monkeypatch, soft=1024, hard=1048576)
    preflight_export_nofile(32, max_pending=64)
    assert fake.raised == [(32 * 256 + 64 + 1024, 1048576)]




def test_export_nofile_preflight_includes_existing_process_fds(monkeypatch):
    from audio_filter_pipeline.processors.stages_v3 import audio_export
    from audio_filter_pipeline.processors.stages_v3.audio_export import (
        preflight_export_nofile,
    )

    fake = _patch_resource(monkeypatch, soft=1024, hard=1048576)
    monkeypatch.setattr(audio_export, "_count_open_fds", lambda: 900)
    preflight_export_nofile(32, max_pending=64)
    assert fake.raised == [(900 + 32 * 256 + 64 + 1024, 1048576)]


def test_export_nofile_preflight_accepts_infinite_hard(monkeypatch):
    from audio_filter_pipeline.processors.stages_v3.audio_export import (
        preflight_export_nofile,
    )

    fake = _patch_resource(monkeypatch, soft=1024, hard=_FakeResource.RLIM_INFINITY)
    preflight_export_nofile(32, max_pending=64)
    assert fake.raised == [(32 * 256 + 64 + 1024, _FakeResource.RLIM_INFINITY)]


@pytest.mark.parametrize("error_number", [errno.EMFILE, errno.ENFILE, errno.EBADF, errno.ENOMEM, errno.EIO])
def test_directory_component_resource_errors_are_not_reported_as_non_directories(error_number):
    from audio_filter_pipeline.utils.path_utils import _directory_component_error

    error = OSError(error_number, "resource exhausted")
    message = str(_directory_component_error(-1, "filter", error))

    assert "safe directory component open failed" in message
    assert "not a real directory" not in message
    assert errno.errorcode[error_number] in message


def test_export_nofile_preflight_fails_closed_when_hard_insufficient(monkeypatch):
    from audio_filter_pipeline.processors.stages_v3.audio_export import (
        preflight_export_nofile,
    )

    _patch_resource(monkeypatch, soft=1024, hard=1024)
    with pytest.raises(ValueError, match="hard limit 1024 is below"):
        preflight_export_nofile(32, max_pending=64)


def test_export_nofile_preflight_reports_raise_failure(monkeypatch):
    from audio_filter_pipeline.processors.stages_v3.audio_export import (
        preflight_export_nofile,
    )

    fake = _patch_resource(monkeypatch, soft=1024, hard=1048576)

    def refusing_setrlimit(_which, _value):
        raise OSError("Operation not permitted")

    fake.setrlimit = refusing_setrlimit
    with pytest.raises(ValueError, match="could not be raised automatically"):
        preflight_export_nofile(32, max_pending=64)


def test_export_nofile_preflight_validates_arguments(monkeypatch):
    from audio_filter_pipeline.processors.stages_v3.audio_export import (
        preflight_export_nofile,
    )

    _patch_resource(monkeypatch, soft=1024, hard=1024)
    with pytest.raises(ValueError, match="workers"):
        preflight_export_nofile(0, max_pending=64)
    with pytest.raises(ValueError, match="max_pending"):
        preflight_export_nofile(32, max_pending=0)


def test_export_nofile_preflight_skipped_without_resource(monkeypatch):
    from audio_filter_pipeline.processors.stages_v3.audio_export import (
        preflight_export_nofile,
    )

    fake = _patch_resource(monkeypatch, soft=1024, hard=1024)
    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.audio_export.resource",
        None,
    )
    # 无 RLIMIT 语义的平台直接跳过，不做任何检查或提升。
    preflight_export_nofile(32, max_pending=64)

    assert fake.raised == []
    # 跳过的是 RLIMIT 语义，不是入参校验。
    with pytest.raises(ValueError, match="workers"):
        preflight_export_nofile(0, max_pending=64)
