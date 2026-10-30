"""Libri2Mix dynamic_tse source producer 的契约测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.prepare_data_tools.generators.tse import prepare_librimix_dynamic_source as producer


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _write_mix(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(160, dtype=np.float32), 16000)


def test_build_source_records_uses_split_local_same_speaker_enrollment(tmp_path: Path) -> None:
    root = tmp_path / "wav16k" / "min"
    split = root / "train-100"
    ids = (
        "1-10-0001_2-20-0001",
        "1-10-0002_2-20-0002",
    )
    for mixture_id in ids:
        for source in ("s1", "s2"):
            _write_mix(split / source / f"{mixture_id}.wav")

    records = producer.build_source_records(root, "train-100")
    assert len(records) == 4
    first = next(item for item in records if item["uid"] == f"s1:{ids[0]}")
    assert first["speaker_id"] == "1"
    assert first["enrollments"] == [str((split / "s1" / f"{ids[1]}.wav").resolve())]
    assert all(item["path"] not in item["enrollments"] for item in records)

    output = tmp_path / "source_records.jsonl"
    assert producer.write_source_jsonl(records, output) == 4
    assert json.loads(output.read_text(encoding="utf-8").splitlines()[0]) == records[0]


def test_build_source_records_rejects_missing_same_speaker_enrollment(tmp_path: Path) -> None:
    root = tmp_path / "wav16k" / "min" / "dev"
    mixture_id = "1-10-0001_2-20-0001"
    for source in ("s1", "s2"):
        _write_mix(root / source / f"{mixture_id}.wav")

    with pytest.raises(ValueError, match="no enrollment candidate"):
        producer.build_source_records(root.parent, "dev")
