"""Smoke-test that onnx/export.py argument parsing works when torch is available."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in this environment")
def test_onnx_export_help() -> None:
    """``python onnx/export.py --help`` should exit 0."""
    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "onnx" / "export.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "ONNX export" in result.stdout
    # argparse 会按终端宽度折行，去掉空白后再断言模型名。
    assert "streaming_deepvqe" in "".join(result.stdout.split())

pytestmark = [pytest.mark.unit]
