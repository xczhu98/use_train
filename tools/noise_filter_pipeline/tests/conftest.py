"""noise_filter_pipeline 测试目录级设施。"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_lock_registry(tmp_path, monkeypatch):
    """把锁 registry 固定到用例私有目录。

    默认 registry（~/.cache/use_train/pipeline-locks）按设计持久保留，
    测试若不隔离就会在真实家目录累积锁文件。
    """
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(tmp_path / "lock-registry"))
