"""Dynamic speech length 完整性门禁的 focused unit tests。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.data.dynamic_dataset_base as dynamic_dataset_base
import core.data.simulation.speech_preparation as speech_preparation
from core.data.dynamic_dataset_base import _read_speech_length_file
from core.data.simulation.speech_preparation import SpeechPreparation
from tasks.se.data.dynamic_dataset import DynamicMixingDataset

pytestmark = pytest.mark.unit


def _write_source_scp(
    path: Path,
    rows: list[tuple[str, int, int | None, str]],
) -> None:
    lines = []
    for uid, sample_rate, length, audio_path in rows:
        if length is None:
            raise ValueError("canonical SCP requires length")
        lines.append(f"{uid} {sample_rate} {length} {audio_path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_source_jsonl(
    path: Path,
    rows: list[tuple[str, int, int | None, str]],
) -> None:
    records = []
    for uid, sample_rate, length, audio_path in rows:
        record = {"uid": uid, "fs": sample_rate, "path": audio_path}
        if length is not None:
            record["length"] = length
        records.append(json.dumps(record))
    path.write_text("\n".join(records) + "\n", encoding="utf-8")


def _dataset_kwargs(tmp_path: Path, speech_rows, *, speech_format="scp", **overrides):
    speech_scp = tmp_path / f"speech.{speech_format}"
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    if speech_format == "scp":
        _write_source_scp(speech_scp, speech_rows)
    else:
        _write_source_jsonl(speech_scp, speech_rows)
    _write_source_scp(
        noise_scp,
        [("noise", 16000, 16000, str(tmp_path / "noise.wav"))],
    )
    _write_source_scp(
        rir_scp,
        [("rir", 16000, 16000, str(tmp_path / "rir.wav"))],
    )
    kwargs = {
        "speech_scp": str(speech_scp),
        "noise_scp": str(noise_scp),
        "rir_scp": str(rir_scp),
        "simulation_cfg": {"prob_wind_noise_sidechain": 0.0},
        "filter_by_duration": False,
        "strict_load": True,
    }
    kwargs.update(overrides)
    return kwargs


def test_read_speech_length_file_accepts_shlex_two_columns(tmp_path: Path):
    length_file = tmp_path / "speech.lengths"
    length_file.write_text(
        '# 注释行\n"utt one" 16000\nutt-two 8000 # 行尾注释\n',
        encoding="utf-8",
    )

    assert _read_speech_length_file(
        str(length_file),
        {"utt one", "utt-two"},
    ) == {"utt one": 16000, "utt-two": 8000}


@pytest.mark.parametrize(
    ("content", "expected_uids", "message"),
    [
        ("utt 10 extra\n", {"utt"}, "exactly two columns"),
        ("utt nope\n", {"utt"}, "integer > 0"),
        ("utt 0\n", {"utt"}, "integer > 0"),
        ("utt 10\nutt 20\n", {"utt"}, "duplicate speech uid"),
        ("other 10\n", {"utt"}, "unexpected speech uid"),
        ("utt 10\n", {"utt", "missing"}, "missing 1 speech UID"),
    ],
)
def test_read_speech_length_file_rejects_invalid_input(
    tmp_path: Path,
    content: str,
    expected_uids: set[str],
    message: str,
):
    length_file = tmp_path / "speech.lengths"
    length_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _read_speech_length_file(str(length_file), expected_uids)


@pytest.mark.parametrize(
    "speech_rows",
    [
        [("speech", 16000, None, "/missing/speech.wav")],
        [
            ("speech-a", 16000, 16000, "/missing/a.wav"),
            ("speech-b", 16000, None, "/missing/b.wav"),
        ],
    ],
)
def test_source_manifest_rejects_missing_length_before_dataset_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    speech_rows,
):
    """canonical JSONL/SCP 均要求 length，strict_load 不能绕过 schema 门禁。"""
    def forbidden_info(_path):
        raise AssertionError("manifest schema 门禁不应调用 soundfile.info")

    monkeypatch.setattr(dynamic_dataset_base.soundfile, "info", forbidden_info)

    with pytest.raises(ValueError, match="length must be an exact integer"):
        DynamicMixingDataset(
            **_dataset_kwargs(tmp_path, speech_rows, speech_format="jsonl")
        )


def test_strict_load_uses_complete_sidecar_without_probing_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    length_file = tmp_path / "speech.lengths"
    length_file.write_text("speech-a 12000\nspeech-b 8000\n", encoding="utf-8")

    def forbidden_info(_path):
        raise AssertionError("完整 sidecar 不应调用 soundfile.info")

    monkeypatch.setattr(dynamic_dataset_base.soundfile, "info", forbidden_info)
    rows = [
        ("speech-a", 16000, 16000, "/missing/a.wav"),
        ("speech-b", 16000, 16000, "/missing/b.wav"),
    ]
    dataset = DynamicMixingDataset(
        **_dataset_kwargs(
            tmp_path,
            rows,
            speech_length_file=str(length_file),
        )
    )

    assert dataset.source_length == {"speech-a": 12000, "speech-b": 8000}
    assert dataset.get_source_length() == [12000, 8000]


def test_non_strict_load_cannot_bypass_canonical_length_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """non-strict 仅影响读取失败处理，不允许回退到无 length manifest。"""
    def forbidden_info(_path):
        raise AssertionError("manifest schema 门禁不应调用 soundfile.info")

    monkeypatch.setattr(dynamic_dataset_base.soundfile, "info", forbidden_info)
    rows = [("speech", 16000, None, "/missing/speech.wav")]
    with pytest.raises(ValueError, match="length must be an exact integer"):
        DynamicMixingDataset(
            **_dataset_kwargs(
                tmp_path, rows, speech_format="jsonl", strict_load=False
            )
        )


def test_unlimited_speech_splice_caps_consecutive_load_failures(monkeypatch):
    dataset = object.__new__(DynamicMixingDataset)
    dataset.speech_preparation = SpeechPreparation()
    dataset.sim_cfg = SimpleNamespace(
        speech_splice=True,
        speech_splice_seconds=1.0,
        speech_splice_seconds_range=None,
        speech_splice_equal_volume=False,
        speech_splice_gap_range=[0.02, 0.02],
        speech_splice_max_repeat_times=0,
        speech_random_start=False,
    )
    dataset.speech_flt = {"speech": "/initial.wav"}
    dataset._all_speech_paths = ["/broken.wav"]
    calls = 0

    def read_speech(path, sr):
        nonlocal calls
        calls += 1
        if path == "/initial.wav":
            return dynamic_dataset_base.np.ones(10, dtype=dynamic_dataset_base.np.float32)
        raise RuntimeError("broken")

    dataset._read_one_speech = read_speech
    monkeypatch.setattr(speech_preparation, "SPEECH_SPLICE_MAX_CONSECUTIVE_FAILURES", 3)
    monkeypatch.setattr(speech_preparation, "SPEECH_SPLICE_UNLIMITED_MAX_ATTEMPTS", 100)

    output = dataset._get_clean_speech("speech", 100, rng=dynamic_dataset_base.np.random.default_rng(0))

    assert output.shape == (100,)
    assert calls == 4
    assert dynamic_dataset_base.np.count_nonzero(output) == 10


def test_unlimited_speech_splice_caps_total_attempts(monkeypatch):
    dataset = object.__new__(DynamicMixingDataset)
    dataset.speech_preparation = SpeechPreparation()
    dataset.sim_cfg = SimpleNamespace(
        speech_splice=True,
        speech_splice_seconds=1.0,
        speech_splice_seconds_range=None,
        speech_splice_equal_volume=False,
        speech_splice_gap_range=[0.02, 0.02],
        speech_splice_max_repeat_times=0,
        speech_random_start=False,
    )
    dataset.speech_flt = {"speech": "/initial.wav"}
    dataset._all_speech_paths = ["/short.wav"]
    calls = 0

    def read_speech(path, sr):
        nonlocal calls
        calls += 1
        length = 10 if path == "/initial.wav" else 1
        return dynamic_dataset_base.np.ones(length, dtype=dynamic_dataset_base.np.float32)

    dataset._read_one_speech = read_speech
    monkeypatch.setattr(speech_preparation, "SPEECH_SPLICE_MAX_CONSECUTIVE_FAILURES", 100)
    monkeypatch.setattr(speech_preparation, "SPEECH_SPLICE_UNLIMITED_MAX_ATTEMPTS", 3)

    output = dataset._get_clean_speech("speech", 100, rng=dynamic_dataset_base.np.random.default_rng(0))

    assert output.shape == (100,)
    assert calls == 4
    assert dynamic_dataset_base.np.count_nonzero(output) == 13
