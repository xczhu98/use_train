"""AEC Challenge GenerationManifest publication 下线合同。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.prepare_data_tools.scp import prepare_aec_challenge_offline as producer

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_aec_challenge_publication_is_retired(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "published"
    argv = [
        "--synthetic-dir", str(tmp_path / "synthetic"),
        "--synthetic-fullband-dir", str(tmp_path / "synthetic-fullband"),
        "--real-fullband-dir", str(tmp_path / "real-fullband"),
        "--real-mobile-dir", str(tmp_path / "real-mobile"),
        "--output-dir", str(output),
    ]

    assert producer.main(argv) == 1
    assert "GenerationManifest v1 publication 已下线" in capsys.readouterr().err
    assert not output.exists()
