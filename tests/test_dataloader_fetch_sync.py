"""DataLoader iterator 异常同步边界回归测试。"""

from __future__ import annotations

import pytest

from core.trainer.metrics_trainer import MetricsCapableTrainer

pytestmark = pytest.mark.unit


class _FailingLoader:
    def __iter__(self):
        yield "first"
        raise RuntimeError("worker/collate failure")


class _OneItemLoader:
    def __iter__(self):
        yield "only"


class _ProbeTrainer:
    _iter_loader_with_synced_fetch = MetricsCapableTrainer._iter_loader_with_synced_fetch
    world_size = 2

    def __init__(self):
        self.phases = []

    def _sync_train_batch_exception(self, error, *, phase):
        self.phases.append((error, phase))
        return None

    def _allreduce_bool_ands(self, values, *, phase):
        del phase
        if tuple(values) == (True, False):
            return False, False
        return tuple(values)


def test_loader_fetch_exception_enters_ddp_sync_boundary():
    trainer = _ProbeTrainer()

    with pytest.raises(RuntimeError, match="worker/collate failure"):
        list(trainer._iter_loader_with_synced_fetch(_FailingLoader(), phase="train_loader"))

    assert [phase for _, phase in trainer.phases] == [
        "train_loader_iterator_init_error",
        "train_loader_fetch_error",
        "train_loader_fetch_error",
    ]
    assert trainer.phases[1][0] is None
    assert isinstance(trainer.phases[2][0], RuntimeError)


def test_loader_exhaustion_mismatch_fails_before_collective_deadlock():
    trainer = _ProbeTrainer()

    with pytest.raises(RuntimeError, match="exhausted on only a subset"):
        list(trainer._iter_loader_with_synced_fetch(_OneItemLoader(), phase="validation_loader"))
