"""环境能力门禁合同。"""

from __future__ import annotations

import pytest

from tools.environment.environment_capabilities import capability_report, require_capabilities

pytestmark = pytest.mark.unit


def test_capability_report_is_structured_and_nonfatal():
    report = capability_report()
    assert {"socket", "numba", "librosa", "torchaudio", "onnxruntime"} <= set(report)
    assert all(isinstance(item.available, bool) and item.detail for item in report.values())


def test_require_capabilities_reports_missing_capability(monkeypatch):
    monkeypatch.setattr(
        "tools.environment.environment_capabilities.capability_report",
        lambda: {"socket": type("C", (), {"name": "socket", "available": False, "detail": "blocked"})()},
    )
    with pytest.raises(RuntimeError, match="socket: blocked"):
        require_capabilities(("socket",))
