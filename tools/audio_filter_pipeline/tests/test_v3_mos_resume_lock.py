"""Step3 resume 的显式 dataset 锁 capability 测试。"""

from __future__ import annotations

import threading

import pytest

from audio_filter_pipeline.processors.stages_v3.runner import (
    DatasetRunContext,
    DatasetRunLockCapability,
)


pytestmark = pytest.mark.unit


def test_persistent_capability_requires_active_context_and_expires(tmp_path):
    context = DatasetRunContext(str(tmp_path), "alpha")
    with pytest.raises(RuntimeError, match="active"):
        context.persistent_write_capability()

    with context:
        capability = context.persistent_write_capability()
        assert isinstance(capability, DatasetRunLockCapability)
        capability.validate(str(tmp_path), "alpha")

    with pytest.raises(RuntimeError, match="no longer active"):
        capability.validate(str(tmp_path), "alpha")


def test_capability_is_explicitly_transferable_to_worker_without_contextvar(tmp_path):
    context = DatasetRunContext(str(tmp_path), "alpha")
    result: list[BaseException | str] = []

    with context:
        capability = context.persistent_write_capability()

        def worker() -> None:
            try:
                capability.validate(str(tmp_path), "alpha")
            except BaseException as exc:
                result.append(exc)
            else:
                result.append("validated")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert result == ["validated"]


def test_capability_cannot_be_constructed_or_rebound(tmp_path):
    with DatasetRunContext(str(tmp_path), "alpha") as context:
        capability = context.persistent_write_capability()
        with pytest.raises(TypeError, match="cannot be constructed"):
            DatasetRunLockCapability(context, "token", object())
        with pytest.raises(ValueError, match="dataset_id"):
            capability.validate(str(tmp_path), "beta")
        with pytest.raises(ValueError, match="output_root"):
            capability.validate(str(tmp_path / "other"), "alpha")
