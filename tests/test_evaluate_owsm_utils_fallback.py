"""evaluate 的 OWSM 工具模块 fallback 合同测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import evaluate.evaluate_urgent as evaluate_urgent

pytestmark = [pytest.mark.unit]

_VENDORED_OWSM_UTILS = Path(evaluate_urgent.__file__).with_name("owsm_utils.py")


def test_require_owsm_utils_returns_existing_file() -> None:
    """WER 入口拿到的路径必须真实存在，不能只在报错时提示。"""
    path = evaluate_urgent._require_owsm_utils()

    assert isinstance(path, Path)
    assert path.is_file()


def test_require_owsm_utils_falls_back_to_vendored_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """官方 checkout 缺失时回退到仓库内 vendored 副本。"""
    monkeypatch.setattr(
        evaluate_urgent, "_URGENT_METRICS_DIR", tmp_path / "evaluation_metrics"
    )

    assert evaluate_urgent._require_owsm_utils() == _VENDORED_OWSM_UTILS


def test_vendored_owsm_utils_exposes_predict_api() -> None:
    """vendored 模块可独立 import，并暴露 owsm_predict 与固定常量。"""
    spec = importlib.util.spec_from_file_location(
        "_vendored_owsm_utils_under_test", _VENDORED_OWSM_UTILS
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.owsm_predict)
    assert module.TARGET_FS == 16000
    assert module.CHUNK_SIZE == 30


def test_require_owsm_utils_fail_fast_when_both_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """官方路径和 vendored 副本都缺失时仍在加载阶段报错。"""
    monkeypatch.setattr(
        evaluate_urgent, "_URGENT_METRICS_DIR", tmp_path / "evaluation_metrics"
    )
    monkeypatch.setattr(
        evaluate_urgent, "_VENDORED_OWSM_UTILS", tmp_path / "owsm_utils.py"
    )

    with pytest.raises(ImportError, match="vendored"):
        evaluate_urgent._require_owsm_utils()
