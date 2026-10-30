"""DNSMOS 空输入回归测试（无需 ONNX 权重）。"""

import math

import numpy as np
import pytest

from core.metrics.mos_models import _DNSMOSScorer
from evaluate.eval_pipeline_tse import _OfficialDNSMOSScorer


def _assert_all_nan(result: dict[str, float], expected_keys: set[str]) -> None:
    assert set(result) == expected_keys
    assert all(math.isnan(value) for value in result.values())


def test_core_dnsmos_empty_input_returns_canonical_nan_keys(monkeypatch):
    scorer = _DNSMOSScorer.__new__(_DNSMOSScorer)

    def fail_resample(*args, **kwargs):
        raise AssertionError("空输入不应进入重采样")

    monkeypatch.setattr("core.metrics.mos_models._resample_np", fail_resample)
    result = scorer(np.array([], dtype=np.float32), 8000)

    _assert_all_nan(result, {"OVRL", "SIG", "BAK"})


def test_tse_official_dnsmos_empty_input_does_not_repeat_or_resample(monkeypatch):
    scorer = _OfficialDNSMOSScorer.__new__(_OfficialDNSMOSScorer)

    def fail_append(*args, **kwargs):
        raise AssertionError("空输入不应进入 repeat loop")

    monkeypatch.setattr(np, "append", fail_append)
    result = scorer(np.array([], dtype=np.float32), 8000)

    _assert_all_nan(result, {"SIG", "BAK", "OVRL", "P808"})


pytestmark = [pytest.mark.unit]
