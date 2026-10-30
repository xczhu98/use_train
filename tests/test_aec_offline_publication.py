"""AEC offline GenerationManifest publication 下线合同。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.prepare_data_tools.scp import prepare_aec_offline as generator

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.full(64, 0.1, dtype=np.float32), 16000, subtype="FLOAT")


def test_scan_wavs_rejects_unreadable_and_duplicate_uids(tmp_path: Path) -> None:
    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    (unreadable / "bad.wav").write_text("not wav", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        generator.scan_wavs(str(unreadable))

    duplicate = tmp_path / "duplicate"
    _wav(duplicate / "a" / "same.wav")
    _wav(duplicate / "b" / "same.wav")
    with pytest.raises(ValueError, match="duplicate uid"):
        generator.scan_wavs(str(duplicate))


def test_aec_offline_publication_is_retired(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "published"
    argv = [
        "--mic_dir", str(tmp_path / "mic"),
        "--farend_dir", str(tmp_path / "farend"),
        "--nearend_dir", str(tmp_path / "nearend"),
        "--output-dir", str(output),
    ]

    assert generator.main(argv) == 1
    assert "GenerationManifest v1 publication 已下线" in capsys.readouterr().err
    assert not output.exists()
