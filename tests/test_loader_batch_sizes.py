"""共享 loader 的 validation 独立 batch size 解析与装配。"""

import logging
from types import SimpleNamespace

import pytest
from torch.utils.data import default_collate

import core.data.sampler as sampler_module
from core.trainer.entry import UnifiedTrainingEntry
from core.trainer.entry_components import (
    build_loaders_common,
    resolve_loader_batch_sizes,
)


pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize(
    "dl_cfg,expected",
    [
        ({"batch_size": 12}, (12, 12)),
        ({"batch_size": 20, "val_batch_size": 1}, (20, 1)),
    ],
)
def test_resolve_loader_batch_sizes_returns_train_and_val(dl_cfg, expected):
    assert resolve_loader_batch_sizes(dl_cfg) == expected


def test_resolve_loader_batch_sizes_requires_train_batch_size():
    with pytest.raises(ValueError, match="data.dataloader.batch_size"):
        resolve_loader_batch_sizes({})


@pytest.mark.parametrize("key", ["batch_size", "val_batch_size"])
@pytest.mark.parametrize("value", [True, False, 2.0, 0, -1, "8"])
def test_resolve_loader_batch_sizes_rejects_non_positive_int(key, value):
    dl_cfg = {"batch_size": 8, "val_batch_size": 2}
    dl_cfg[key] = value
    with pytest.raises(ValueError) as exc_info:
        resolve_loader_batch_sizes(dl_cfg)
    assert f"data.dataloader.{key}" in str(exc_info.value)


class _Dataset:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return float(index)


class _RecordingBatchSampler:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.batch_size = kwargs["batch_size"]
        self.kwargs = kwargs
        self.epoch = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return 1

    def __iter__(self):
        yield list(range(min(self.batch_size, len(self.dataset))))


def _entry():
    return SimpleNamespace(
        logger=logging.getLogger(__name__),
        _validate_num_per_epoch=UnifiedTrainingEntry._validate_num_per_epoch,
    )


def test_build_loaders_common_uses_val_batch_size(monkeypatch, caplog):
    calls = []

    class _FakeSampler(_RecordingBatchSampler):
        def __init__(self, dataset, **kwargs):
            super().__init__(dataset, **kwargs)
            calls.append(self)

    monkeypatch.setattr(sampler_module, "GroupedBatchSampler", _FakeSampler)
    cfg = {
        "meta": {"seed": 17},
        "data": {
            "dataloader": {
                "batch_size": 20,
                "val_batch_size": 1,
                "num_workers": 0,
                "pin_memory": False,
            }
        },
    }
    with caplog.at_level(logging.WARNING):
        train_loader, val_loader, train_sampler, _ = build_loaders_common(
            _entry(),
            cfg,
            rank=0,
            local_rank=0,
            world_size=2,
            train_ds=_Dataset(8),
            val_ds=_Dataset(1),
            train_collate_fn=default_collate,
        )
    assert [call.batch_size for call in calls] == [20, 1]
    assert [call.kwargs["drop_last"] for call in calls] == [True, False]
    assert train_loader.batch_sampler is train_sampler is calls[0]
    assert val_loader.batch_sampler is calls[1]
    assert "world_size * batch_size (2)" in caplog.text
