"""严格 validation 在 collective 前拒绝全局空分片。"""

from datetime import timedelta
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import default_collate

from core.data.sampler import GroupedBatchSampler
from core.trainer.entry import UnifiedTrainingEntry
from core.trainer.entry_components import build_loaders_common


pytestmark = [pytest.mark.unit]


class _Dataset:
    def __init__(self, srs):
        self.srs = list(srs)

    def __len__(self):
        return len(self.srs)

    def __getitem__(self, index):
        return torch.tensor([float(index)])

    def get_srs(self):
        return self.srs

    def get_source_length(self):
        return list(range(len(self)))


@pytest.mark.parametrize("direct_iteration", [False, True])
@pytest.mark.parametrize("srs,counts", [([8000, 16000], [2, 0]), ([], [0, 0])])
def test_validation_full_epoch_rejects_empty_rank_before_collective(
    monkeypatch, direct_iteration, srs, counts
):
    collective_calls = []
    monkeypatch.setattr(
        dist, "all_reduce", lambda *args, **kwargs: collective_calls.append(1)
    )
    errors = []
    for rank in range(2):
        sampler = GroupedBatchSampler(
            _Dataset(srs), batch_size=1, rank=rank, world_size=2,
            require_non_empty_all_ranks=True,
        )
        state_before = vars(sampler).copy()
        with pytest.raises(ValueError, match="0 real batch") as exc_info:
            if direct_iteration:
                list(sampler)
            else:
                sampler.set_epoch(0)
        errors.append(str(exc_info.value))
        assert vars(sampler) == state_before

    assert errors[0] == errors[1]
    assert f"rank_sample_counts={counts}" in errors[0]
    assert f"rank_batch_counts={counts}" in errors[0]
    assert "epoch=0" in errors[0]
    assert "num_per_epoch=None" in errors[0]
    assert collective_calls == []


class _BalancedDataset(_Dataset):
    balanced_speech_sampling_enabled = True

    def get_speech_epoch_plan(self, **request):
        indices = tuple(range(len(self)))
        return {
            "selected_indices": indices,
            "selected_uids": tuple(str(index) for index in indices),
            "speech_plan": {
                "actual": len(indices),
                "sr_quotas": {16000: len(indices)} if indices else {},
                "cell_quotas": {("zh", "set", 16000): len(indices)},
            },
        }


def test_validation_balanced_empty_plan_rejected_before_collective(monkeypatch):
    collective_calls = []
    monkeypatch.setattr(
        dist, "all_reduce", lambda *args, **kwargs: collective_calls.append(1)
    )
    errors = []
    for rank in range(2):
        with pytest.raises(ValueError, match="0 real batch") as exc_info:
            GroupedBatchSampler(
                _BalancedDataset([]), batch_size=1, rank=rank, world_size=2,
                require_non_empty_all_ranks=True,
            )
        errors.append(str(exc_info.value))
    assert errors[0] == errors[1]
    assert "rank_batch_counts=[0, 0]" in errors[0]
    assert collective_calls == []


@pytest.mark.parametrize("num_per_epoch", [None, 4])
def test_validation_direct_iteration_matches_prepared_epoch(num_per_epoch):
    kwargs = dict(
        batch_size=1, seed=17, num_per_epoch=num_per_epoch,
        require_non_empty_all_ranks=True,
    )
    direct = GroupedBatchSampler(_Dataset([16000] * 8), **kwargs)
    prepared = GroupedBatchSampler(_Dataset([16000] * 8), **kwargs)
    prepared.set_epoch(0)
    expected = list(prepared)
    assert list(direct) == expected
    assert list(direct) == expected
    assert sum(map(len, expected)) == (8 if num_per_epoch is None else 4)
    assert len(direct) == len(prepared)


def _build_loaders(rank, validation_srs):
    entry = SimpleNamespace(
        logger=logging.getLogger(__name__),
        _validate_num_per_epoch=UnifiedTrainingEntry._validate_num_per_epoch,
    )
    return build_loaders_common(
        entry,
        {"meta": {"seed": 17}, "data": {"dataloader": {
            "batch_size": 1, "num_workers": 0, "pin_memory": False,
        }}},
        rank=rank, local_rank=rank, world_size=2,
        train_ds=_Dataset([16000] * 4), val_ds=_Dataset(validation_srs),
        train_collate_fn=default_collate,
    )


class _BufferedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.register_buffer("offset", torch.tensor(3.0))

    def forward(self, values):
        return self.weight * values + self.offset


def _gloo_validation_worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/store", rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        collective_calls = 0
        original_all_reduce = dist.all_reduce

        def track_all_reduce(*args, **kwargs):
            nonlocal collective_calls
            collective_calls += 1
            return original_all_reduce(*args, **kwargs)

        dist.all_reduce = track_all_reduce
        try:
            _build_loaders(rank, [8000, 16000])
        except ValueError as exc:
            error = str(exc)
        else:
            raise AssertionError("unsafe validation loader was returned")
        finally:
            dist.all_reduce = original_all_reduce
        errors = [None, None]
        dist.all_gather_object(errors, (error, collective_calls))
        assert errors[0] == errors[1]
        assert "rank_batch_counts=[2, 0]" in error
        assert collective_calls == 0

        _, loader, _, _ = _build_loaders(rank, [16000] * 3)
        model = DistributedDataParallel(_BufferedModel(), broadcast_buffers=True)
        model.eval()
        forwarded = []
        padding = []
        with torch.no_grad():
            for step, batch in enumerate(loader):
                torch.testing.assert_close(model(batch), batch * 2 + 3)
                forwarded.extend(batch[:, 0].tolist())
                padding.append(step >= loader.batch_sampler._last_iter_real_len)
        results = [None, None]
        dist.all_gather_object(results, (forwarded, padding))
        assert sorted(results[0][0]) == [0.0, 2.0]
        assert results[1][0] == [1.0, 1.0]
        assert results[0][1] == [False, False]
        assert results[1][1] == [False, True]
        dist.barrier()
        Path(root, f"rank{rank}.json").write_text(json.dumps({
            "preflight_collectives": collective_calls,
            "forward_counts": [len(item[0]) for item in results],
            "padding": [item[1] for item in results],
        }), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.ddp
@pytest.mark.subprocess
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_gloo_shared_validation_rejection_then_uneven_ddp_forward(tmp_path):
    with TemporaryDirectory(dir=tmp_path) as root:
        context = mp.spawn(_gloo_validation_worker, args=(root,), nprocs=2, join=False)
        deadline = time.monotonic() + 60
        try:
            while not context.join(timeout=1):
                if time.monotonic() >= deadline:
                    raise TimeoutError("validation Gloo workers did not finish")
            reports = [
                json.loads(Path(root, f"rank{rank}.json").read_text(encoding="utf-8"))
                for rank in range(2)
            ]
            assert reports[0] == reports[1] == {
                "preflight_collectives": 0,
                "forward_counts": [2, 2],
                "padding": [[False, False], [False, True]],
            }
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
