"""GroupedBatchSampler resume determinism 回归测试（审计 A15）。

覆盖：
(a) 同 seed+epoch：fresh 构造的 sampler 与已迭代过的 sampler 产出完全相同的
    batch 顺序，且迭代后 self.buckets 保持构造时初态；
(b) 连续两次 set_epoch(e) + 迭代：同 epoch 顺序一致、跨 epoch 顺序不同；
(c) DDP ws=2 两个 rank（打桩 dist.all_reduce 模拟 MAX 语义）：两 rank batch
    数一致，且每个 batch 内样本同 SR（collate 约束）；
(d) DDP 尾部补齐时 _last_iter_real_len 记录补齐前真实批数（val 据此剔除复制批
    贡献）；单卡时 _last_iter_real_len 等于迭代长度。
"""
from datetime import timedelta
import os
from pathlib import Path
import time

import pytest

import torch.distributed as dist
import torch.multiprocessing as mp

from core.data.sampler import GroupedBatchSampler


class _FakeDataset:
    """最小 dataset 替身：只提供 GroupedBatchSampler 依赖的两个接口。"""

    def __init__(self, srs, lengths):
        assert len(srs) == len(lengths)
        self._srs = list(srs)
        self._lengths = list(lengths)

    def get_srs(self):
        return list(self._srs)

    def get_source_length(self):
        return list(self._lengths)


def _make_dataset():
    # 两个 SR、长度各异，确保分桶与桶内排序非平凡
    srs = [16000] * 30 + [8000] * 26
    lengths = [(i * 37) % 500 + 50 for i in range(len(srs))]
    return _FakeDataset(srs, lengths)


def _make_sampler(seed=7):
    s = GroupedBatchSampler(
        _make_dataset(), batch_size=4, seed=seed, bucket_size_mult=3
    )
    s.set_epoch(3)
    return s


def test_fresh_and_reiterated_sampler_same_order():
    s1 = _make_sampler()
    buckets_before = [list(b) for b in s1.buckets]

    first = list(iter(s1))
    second = list(iter(s1))  # 同一 sampler 再迭代一次（模拟 resume 后继续跑）

    s2 = _make_sampler()
    fresh = list(iter(s2))

    assert first == second == fresh
    # 迭代不得修改构造时的桶初态（含桶间顺序与桶内顺序）
    assert s1.buckets == buckets_before
    assert s2.buckets == buckets_before


def test_set_epoch_same_epoch_deterministic_cross_epoch_differs():
    s = _make_sampler()
    s.set_epoch(5)
    e5a = list(iter(s))
    s.set_epoch(5)
    e5b = list(iter(s))
    s.set_epoch(6)
    e6 = list(iter(s))

    assert e5a == e5b
    assert e5a != e6
    # 跨 epoch 只是顺序不同，样本集合不变
    assert sorted(b for batch in e5a for b in batch) == sorted(
        b for batch in e6 for b in batch
    )


def test_ddp_two_ranks_same_batch_count_and_same_sr(monkeypatch):
    # 样本数对 ws=2 不均分，stride 后两 rank 本地 batch 数不同，
    # 必须靠 all_reduce(MAX) + 桶内补齐对齐
    ds = _FakeDataset([16000] * 25 + [8000] * 23, list(range(48)))
    kwargs = dict(batch_size=4, world_size=2, seed=11, bucket_size_mult=3)
    s0 = GroupedBatchSampler(ds, rank=0, **kwargs)
    s1 = GroupedBatchSampler(ds, rank=1, **kwargs)
    s0.set_epoch(2)
    s1.set_epoch(2)

    # 第一遍：all_reduce 打桩为 no-op，各 rank 保持本地 batch 数，量出 target
    monkeypatch.setattr(dist, "all_reduce", lambda t, op=None: None)
    n0 = len(list(iter(s0)))
    n1 = len(list(iter(s1)))
    assert n0 != n1  # 确认测试场景确实触发补齐路径
    target = max(n0, n1)

    # 第二遍：all_reduce 打桩为 MAX 语义（两 rank 统一为 target）
    monkeypatch.setattr(
        dist, "all_reduce", lambda t, op=None: t.fill_(target)
    )
    b0 = list(iter(s0))
    b1 = list(iter(s1))
    assert len(b0) == len(b1) == target

    # 补齐只允许复制同 SR 桶内 batch：每个 batch 内样本必须同 SR
    srs = ds.get_srs()
    for batch in b0 + b1:
        assert len({srs[i] for i in batch}) == 1


@pytest.mark.parametrize(
    ("backend", "expected_device"),
    [("gloo", "cpu"), ("nccl", "cuda:3")],
)
def test_ddp_cardinality_collective_device_follows_backend(
    monkeypatch, backend, expected_device
):
    import torch

    requested_devices = []
    real_tensor = torch.tensor

    def capture_tensor(*args, **kwargs):
        requested_devices.append(str(kwargs.pop("device")))
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: backend)
    monkeypatch.setattr(torch, "tensor", capture_tensor)
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None: None)
    sampler = GroupedBatchSampler(
        _FakeDataset([16000] * 4, list(range(4))),
        batch_size=2,
        rank=0,
        local_rank=3,
        world_size=2,
    )

    list(sampler)

    assert requested_devices == [expected_device]


def test_ddp_real_len_marks_padding(monkeypatch):
    # 与 test_ddp_two_ranks_same_batch_count_and_same_sr 相同的不均分场景
    ds = _FakeDataset([16000] * 25 + [8000] * 23, list(range(48)))
    kwargs = dict(batch_size=4, world_size=2, seed=11, bucket_size_mult=3)
    s0 = GroupedBatchSampler(ds, rank=0, **kwargs)
    s1 = GroupedBatchSampler(ds, rank=1, **kwargs)
    s0.set_epoch(2)
    s1.set_epoch(2)

    # 先量出各 rank 补齐前的真实批数
    monkeypatch.setattr(dist, "all_reduce", lambda t, op=None: None)
    n0 = len(list(iter(s0)))
    n1 = len(list(iter(s1)))
    assert n0 != n1
    target = max(n0, n1)

    monkeypatch.setattr(dist, "all_reduce", lambda t, op=None: t.fill_(target))
    b0 = list(iter(s0))
    b1 = list(iter(s1))
    assert len(b0) == len(b1) == target
    # real_len 等于补齐前的真实批数；尾部 target - real_len 个为复制批
    assert s0._last_iter_real_len == n0
    assert s1._last_iter_real_len == n1
    for b, real in ((b0, n0), (b1, n1)):
        for pad in b[real:]:
            assert pad == b[real - 1]


def test_real_len_single_rank_equals_iter_len():
    s = _make_sampler()
    n = len(list(iter(s)))
    assert s._last_iter_real_len == n


@pytest.mark.parametrize(
    ("drop_last", "constructor_len", "epoch_one_len"),
    [(False, 3, 4), (True, 0, 1)],
)
def test_virtual_epoch_constructor_keeps_legacy_estimate_until_set_epoch(
    drop_last, constructor_len, epoch_one_len
):
    ds = _FakeDataset(
        [16000] * 5 + [8000] * 5 + [48000] * 5,
        list(range(15)),
    )
    sampler = GroupedBatchSampler(
        ds,
        batch_size=3,
        seed=1,
        num_per_epoch=7,
        drop_last=drop_last,
    )

    assert len(sampler) == constructor_len

    sampler.set_epoch(1)
    assert len(sampler) == epoch_one_len
    assert len(list(sampler)) == epoch_one_len


@pytest.mark.parametrize("drop_last", [False, True])
def test_validation_virtual_epoch_multi_sr_len_matches_first_iteration(drop_last):
    ds = _FakeDataset(
        [16000] * 3 + [8000] * 3 + [48000] * 3,
        list(range(9)),
    )
    sampler = GroupedBatchSampler(
        ds,
        batch_size=2,
        seed=17,
        num_per_epoch=2,
        drop_last=drop_last,
        require_non_empty_all_ranks=True,
    )
    sampler.set_epoch(0)

    before = len(sampler)
    actual = len(list(sampler))

    assert before == actual
    assert actual == (0 if drop_last else 2)


def test_train_preflight_rejects_zero_sr_shard_symmetrically_before_collective(
    monkeypatch,
):
    ds = _FakeDataset([8000, 16000], [0, 1])
    collective_calls = []
    monkeypatch.setattr(
        dist,
        "all_reduce",
        lambda *args, **kwargs: collective_calls.append((args, kwargs)),
    )
    errors = []

    for rank in range(2):
        sampler = GroupedBatchSampler(
            ds,
            batch_size=1,
            rank=rank,
            world_size=2,
            drop_last=True,
            require_non_empty_train_shards=True,
        )
        with pytest.raises(ValueError) as exc_info:
            sampler.preflight_train_shards(epoch=0)
        errors.append(str(exc_info.value))

    assert errors[0] == errors[1]
    assert "0 local samples" in errors[0]
    assert "rank_sample_counts=[2, 0]" in errors[0]
    assert "rank_batch_counts=[2, 0]" in errors[0]
    assert collective_calls == []


def test_train_preflight_rejects_zero_real_batch_even_with_local_samples(monkeypatch):
    ds = _FakeDataset([8000, 8000, 16000, 16000], [0, 1, 2, 3])
    collective_calls = []
    monkeypatch.setattr(
        dist,
        "all_reduce",
        lambda *args, **kwargs: collective_calls.append((args, kwargs)),
    )
    errors = []

    for rank in range(2):
        sampler = GroupedBatchSampler(
            ds,
            batch_size=2,
            rank=rank,
            world_size=2,
            drop_last=True,
            require_non_empty_train_shards=True,
        )
        with pytest.raises(ValueError, match="0 real batches") as exc_info:
            sampler.preflight_train_shards(epoch=0)
        errors.append(str(exc_info.value))

    assert errors[0] == errors[1]
    assert "rank_sample_counts=[2, 2]" in errors[0]
    assert "rank_batch_counts=[0, 0]" in errors[0]
    assert collective_calls == []


@pytest.mark.parametrize("num_per_epoch", [None, 4, 99])
@pytest.mark.parametrize("require_train_shards", [False, True])
def test_train_full_path_keeps_dummy_padding_when_rank_has_no_real_batch(
    monkeypatch, num_per_epoch, require_train_shards
):
    ds = _FakeDataset([8000] * 3 + [16000], list(range(4)))
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None: tensor.fill_(1))

    samplers = [
        GroupedBatchSampler(
            ds,
            batch_size=2,
            rank=rank,
            world_size=2,
            num_per_epoch=num_per_epoch,
            drop_last=True,
            require_non_empty_train_shards=require_train_shards,
        )
        for rank in range(2)
    ]
    for sampler in samplers:
        sampler.set_epoch(0)
    batches = [list(sampler) for sampler in samplers]

    assert [len(rank_batches) for rank_batches in batches] == [1, 1]
    assert [sampler._last_iter_real_len for sampler in samplers] == [1, 0]
    assert len(batches[1][0]) == 2
    assert len({ds.get_srs()[index] for index in batches[1][0]}) == 1


def test_validation_virtual_epoch_ddp_multi_sr_empty_rank_fails_before_collective(
    monkeypatch,
):
    ds = _FakeDataset(
        [16000] * 3 + [8000] * 3 + [48000] * 3,
        list(range(9)),
    )
    collective_calls = []
    monkeypatch.setattr(
        dist,
        "all_reduce",
        lambda *args, **kwargs: collective_calls.append((args, kwargs)),
    )
    errors = []

    for rank in range(2):
        sampler = GroupedBatchSampler(
            ds,
            batch_size=2,
            rank=rank,
            world_size=2,
            seed=17,
            num_per_epoch=2,
            drop_last=False,
            require_non_empty_all_ranks=True,
        )
        with pytest.raises(ValueError) as exc_info:
            sampler.set_epoch(0)
        errors.append(str(exc_info.value))

    assert errors[0] == errors[1]
    assert "0 real batch" in errors[0]
    assert collective_calls == []


def test_validation_failed_set_epoch_preserves_previous_state(monkeypatch):
    ds = _FakeDataset(
        [16000] * 4 + [8000] * 4 + [48000] * 4,
        list(range(12)),
    )
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None: None)
    sampler = GroupedBatchSampler(
        ds,
        batch_size=2,
        rank=0,
        world_size=2,
        seed=0,
        num_per_epoch=2,
        drop_last=False,
        require_non_empty_all_ranks=True,
    )
    sampler.set_epoch(2)
    successful_batches = list(sampler)
    state_before = (
        sampler.epoch,
        sampler._shuffle_salt,
        [list(bucket) for bucket in sampler.buckets],
        list(sampler._sr_group_sizes),
        sampler._local_samples,
        sampler._len_estimate,
        sampler._last_iter_len,
        sampler._last_iter_real_len,
    )

    with pytest.raises(ValueError, match="0 real batch"):
        sampler.set_epoch(0)

    state_after = (
        sampler.epoch,
        sampler._shuffle_salt,
        [list(bucket) for bucket in sampler.buckets],
        list(sampler._sr_group_sizes),
        sampler._local_samples,
        sampler._len_estimate,
        sampler._last_iter_len,
        sampler._last_iter_real_len,
    )
    assert state_after == state_before
    assert len(sampler) == len(successful_batches)
    assert list(sampler) == successful_batches


@pytest.mark.parametrize(
    ("drop_last", "seed", "num_per_epoch"),
    [(False, 4, 2), (True, 93, 4)],
)
def test_validation_virtual_epoch_ddp_len_uses_exact_safe_sr_distribution(
    monkeypatch, drop_last, seed, num_per_epoch
):
    ds = _FakeDataset(
        [16000] * 4 + [8000] * 4 + [48000] * 4,
        list(range(12)),
    )
    collective_calls = []
    monkeypatch.setattr(
        dist,
        "all_reduce",
        lambda *args, **kwargs: collective_calls.append((args, kwargs)),
    )

    for rank in range(2):
        sampler = GroupedBatchSampler(
            ds,
            batch_size=2,
            rank=rank,
            world_size=2,
            seed=seed,
            num_per_epoch=num_per_epoch,
            drop_last=drop_last,
            require_non_empty_all_ranks=True,
        )
        sampler.set_epoch(0)
        calls_before_len = len(collective_calls)
        before = len(sampler)
        assert len(collective_calls) == calls_before_len
        actual = len(list(sampler))
        assert before == actual == 1


_GLOO_WORLD_SIZE = 2
_GLOO_TIMEOUT_SEC = 45


def _gloo_sampler_contract_worker(rank, world_size, init_file, result_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_GLOO_TIMEOUT_SEC - 10),
    )
    try:
        reviewer_ds = _FakeDataset([8000, 16000], [0, 1])
        reviewer_sampler = GroupedBatchSampler(
            reviewer_ds,
            batch_size=1,
            rank=rank,
            world_size=world_size,
            drop_last=True,
            require_non_empty_train_shards=True,
        )
        original_all_reduce = dist.all_reduce
        reviewer_collective_calls = 0

        def track_reviewer_all_reduce(*args, **kwargs):
            nonlocal reviewer_collective_calls
            reviewer_collective_calls += 1
            return original_all_reduce(*args, **kwargs)

        dist.all_reduce = track_reviewer_all_reduce
        try:
            reviewer_sampler.preflight_train_shards(epoch=0)
        except ValueError as exc:
            reviewer_error = str(exc)
        else:
            raise AssertionError("train preflight did not reject a zero SR shard")
        finally:
            dist.all_reduce = original_all_reduce

        gathered_reviewer = [None] * world_size
        dist.all_gather_object(
            gathered_reviewer, (reviewer_error, reviewer_collective_calls)
        )
        assert gathered_reviewer[0] == gathered_reviewer[1]
        assert "0 local samples" in reviewer_error
        assert reviewer_collective_calls == 0

        validation_ds = _FakeDataset(
            [16000] * 3 + [8000] * 3 + [48000] * 3,
            list(range(9)),
        )
        validation_sampler = GroupedBatchSampler(
            validation_ds,
            batch_size=2,
            rank=rank,
            world_size=world_size,
            seed=17,
            num_per_epoch=2,
            drop_last=False,
            require_non_empty_all_ranks=True,
        )
        original_all_reduce = dist.all_reduce
        collective_calls = 0

        def track_all_reduce(*args, **kwargs):
            nonlocal collective_calls
            collective_calls += 1
            return original_all_reduce(*args, **kwargs)

        dist.all_reduce = track_all_reduce
        try:
            validation_sampler.set_epoch(0)
        except ValueError as exc:
            error = str(exc)
        else:
            raise AssertionError("validation gate did not reject unsafe virtual epoch")
        finally:
            dist.all_reduce = original_all_reduce

        gathered_errors = [None] * world_size
        dist.all_gather_object(gathered_errors, (error, collective_calls))
        assert gathered_errors[0] == gathered_errors[1]
        assert "0 real batch" in error
        assert collective_calls == 0

        train_ds = _FakeDataset([8000] * 3 + [16000], list(range(4)))
        for num_per_epoch in (None, 4, 99):
            train_sampler = GroupedBatchSampler(
                train_ds,
                batch_size=2,
                rank=rank,
                world_size=world_size,
                num_per_epoch=num_per_epoch,
                drop_last=True,
            )
            train_sampler.set_epoch(0)
            batches = list(train_sampler)
            local_result = (len(batches), train_sampler._last_iter_real_len)
            gathered_results = [None] * world_size
            dist.all_gather_object(gathered_results, local_result)
            assert gathered_results == [(1, 1), (1, 0)]

        dist.barrier()
        Path(result_dir, f"rank{rank}.ok").write_text("ok", encoding="utf-8")
    finally:
        dist.destroy_process_group()


def _spawn_gloo_sampler_contract(worker, tmp_path):
    ctx = mp.spawn(
        worker,
        args=(
            _GLOO_WORLD_SIZE,
            str(tmp_path / "init_store"),
            str(tmp_path),
        ),
        nprocs=_GLOO_WORLD_SIZE,
        join=False,
    )
    deadline = time.monotonic() + _GLOO_TIMEOUT_SEC
    while not ctx.join(max(0.0, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            for process in ctx.processes:
                if process.is_alive():
                    process.terminate()
            raise TimeoutError("Gloo sampler contract test timed out")


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_gloo_validation_gate_precedes_collective_and_train_full_paths_succeed(tmp_path):
    _spawn_gloo_sampler_contract(_gloo_sampler_contract_worker, tmp_path)
    assert (tmp_path / "rank0.ok").exists()
    assert (tmp_path / "rank1.ok").exists()


pytestmark = [pytest.mark.unit]
