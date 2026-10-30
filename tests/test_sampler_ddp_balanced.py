"""DDP-safe consumption tests for Dataset-owned balanced speech plans."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

import pytest

from core.data.dynamic_dataset_base import (
    BaseDynamicMixingDataset,
    SpeechSamplingLeaf,
    _speech_cells_from_leaves,
)
from core.data.sampler import GroupedBatchSampler
from core.data.sampling import SamplingConfig


class _BalancedDataset:
    def __init__(self, plan, *, enabled=True, srs=None):
        self.balanced_speech_sampling_enabled = enabled
        self._plan = plan
        self.plan_requests = []
        self._srs = [16000] * 8 if srs is None else list(srs)
        self._lengths = list(range(len(self._srs)))

    def get_srs(self):
        return list(self._srs)

    def get_source_length(self):
        return list(self._lengths)

    def get_speech_epoch_plan(self, *, requested, world_size, batch_size, drop_last, seed, epoch):
        self.plan_requests.append((requested, world_size, batch_size, drop_last, seed, epoch))
        return self._plan


class _LegacyDataset:
    def __init__(self):
        self.balanced_speech_sampling_enabled = False
        self._srs = [16000] * 8
        self._lengths = list(range(8))

    def get_srs(self):
        return list(self._srs)

    def get_source_length(self):
        return list(self._lengths)


def _plan(indices=(0, 1, 2, 3), *, sr_quotas=None, cell_quotas=None, cell_keys=None, actual=None):
    actual = len(indices) if actual is None else actual
    cell_keys = cell_keys or tuple(("zh", "set", 16000) for _ in indices)
    return {
        "selected_indices": tuple(indices),
        "selected_uids": tuple(f"uid-{index}" for index in indices),
        "selected_cell_keys": tuple(cell_keys),
        "speech_plan": {
            "actual": actual,
            "sr_quotas": sr_quotas or {16000: len(indices)},
            "cell_quotas": cell_quotas or {("zh", "set", 16000): len(indices)},
        },
    }


def _dataset_plan_fixture():
    dataset = object.__new__(BaseDynamicMixingDataset)
    dataset._total = 8
    dataset.speech_sampling_leaves = {
        ("zh", "set-a", None, 16000): SpeechSamplingLeaf(
            "zh", "set-a", None, 16000, (0, 1, 2, 3), ("a0", "a1", "a2", "a3"), Fraction(4)
        ),
        ("zh", "set-b", None, 16000): SpeechSamplingLeaf(
            "zh", "set-b", None, 16000, (4, 5, 6, 7), ("b0", "b1", "b2", "b3"), Fraction(4)
        ),
    }
    dataset.speech_sampling_cells = _speech_cells_from_leaves(
        dataset.speech_sampling_leaves
    )
    dataset._speech_sampling_config = SamplingConfig.from_dict(
        {"speech": {"language_probs": {"zh": 1}, "max_dataset_ratio": 1}}
    ).speech
    dataset.balanced_speech_sampling_enabled = True
    return dataset


def test_dataset_epoch_plan_is_deterministic_unique_and_matches_quotas():
    dataset = _dataset_plan_fixture()

    first = dataset.get_speech_epoch_plan(
        requested=4, world_size=2, batch_size=1, drop_last=False, seed=13, epoch=2
    )
    repeated = dataset.get_speech_epoch_plan(
        requested=4, world_size=2, batch_size=1, drop_last=False, seed=13, epoch=2
    )
    changed_epoch = dataset.get_speech_epoch_plan(
        requested=4, world_size=2, batch_size=1, drop_last=False, seed=13, epoch=3
    )

    assert first == repeated
    assert first.selected_indices != changed_epoch.selected_indices
    assert len(set(first.selected_indices)) == first.speech_plan.actual == 4
    assert sum(first.speech_plan.cell_quotas.values()) == len(first.selected_indices)
    assert Counter(first.sample_rates) == first.speech_plan.sr_quotas


def test_dataset_reuses_fixed_quota_plan_across_epochs(monkeypatch):
    from core.data import dynamic_dataset_base

    dataset = _dataset_plan_fixture()
    real_make_speech_plan = dynamic_dataset_base.make_speech_plan
    planner_calls = 0

    def count_planner_calls(*args, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        return real_make_speech_plan(*args, **kwargs)

    monkeypatch.setattr(dynamic_dataset_base, "make_speech_plan", count_planner_calls)
    first = dataset.get_speech_epoch_plan(
        requested=4, world_size=2, batch_size=1, drop_last=False, seed=13, epoch=2
    )
    changed_epoch = dataset.get_speech_epoch_plan(
        requested=4, world_size=2, batch_size=1, drop_last=False, seed=13, epoch=3
    )

    assert planner_calls == 1
    assert first.speech_plan == changed_epoch.speech_plan
    assert first.selected_indices != changed_epoch.selected_indices


def test_dataset_retries_fixed_quota_plan_after_initial_failure(monkeypatch):
    from core.data import dynamic_dataset_base

    dataset = _dataset_plan_fixture()
    real_make_speech_plan = dynamic_dataset_base.make_speech_plan
    planner_calls = 0

    def fail_once(*args, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls == 1:
            raise ValueError("simulated first planning failure")
        return real_make_speech_plan(*args, **kwargs)

    monkeypatch.setattr(dynamic_dataset_base, "make_speech_plan", fail_once)
    with pytest.raises(ValueError, match="simulated first planning failure"):
        dataset.get_speech_epoch_plan(
            requested=4, world_size=2, batch_size=1, drop_last=False, seed=13, epoch=2
        )

    plan = dataset.get_speech_epoch_plan(
        requested=4, world_size=2, batch_size=1, drop_last=False, seed=13, epoch=2
    )
    assert planner_calls == 2
    assert plan.speech_plan.actual == 4


def test_balanced_ddp_shards_dataset_plan_without_collectives_or_padding(monkeypatch):
    plan = _plan()
    dataset = _BalancedDataset(plan)
    calls = []
    monkeypatch.setattr("torch.distributed.all_reduce", lambda *args, **kwargs: calls.append(args))

    samplers = [
        GroupedBatchSampler(dataset, batch_size=1, rank=rank, world_size=2, seed=7, num_per_epoch=4)
        for rank in range(2)
    ]
    for sampler in samplers:
        sampler.set_epoch(5)
    shards = [list(sampler) for sampler in samplers]

    assert dataset.plan_requests == [
        (4, 2, 1, False, 7, 0), (4, 2, 1, False, 7, 0),
        (4, 2, 1, False, 7, 5), (4, 2, 1, False, 7, 5),
    ]
    assert calls == []
    assert len(shards[0]) == len(shards[1]) == 2
    assert set(index for batch in shards[0] for index in batch).isdisjoint(
        index for batch in shards[1] for index in batch
    )
    assert set(index for batches in shards for batch in batches for index in batch) == set(plan["selected_indices"])


def test_balanced_same_epoch_set_is_idempotent_and_does_not_replan():
    dataset = _BalancedDataset(_plan())
    sampler = GroupedBatchSampler(
        dataset,
        batch_size=1,
        world_size=2,
        seed=7,
        num_per_epoch=4,
    )

    sampler.set_epoch(1)
    first_batches = [list(batch) for batch in sampler._balanced_batches]
    sampler.set_epoch(1)

    assert dataset.plan_requests == [
        (4, 2, 1, False, 7, 0),
        (4, 2, 1, False, 7, 1),
    ]
    assert sampler._balanced_batches == first_batches


def test_balanced_validation_epoch_zero_plan_is_consumed_once_across_ddp_ranks(
    monkeypatch,
):
    plan = _plan()
    datasets = [_BalancedDataset(plan) for _ in range(2)]
    collective_calls = []
    monkeypatch.setattr(
        "torch.distributed.all_reduce",
        lambda *args, **kwargs: collective_calls.append((args, kwargs)),
    )

    samplers = [
        GroupedBatchSampler(
            dataset,
            batch_size=1,
            rank=rank,
            world_size=2,
            seed=7,
            num_per_epoch=4,
        )
        for rank, dataset in enumerate(datasets)
    ]
    shards = [list(sampler) for sampler in samplers]
    rank_indices = [
        [index for batch in shard for index in batch] for shard in shards
    ]
    rank_uids = [
        [f"uid-{index}" for index in indices] for indices in rank_indices
    ]

    assert [dataset.plan_requests for dataset in datasets] == [
        [(4, 2, 1, False, 7, 0)],
        [(4, 2, 1, False, 7, 0)],
    ]
    assert collective_calls == []
    assert [sampler._last_iter_real_len for sampler in samplers] == [2, 2]
    assert all(len(indices) == len(set(indices)) for indices in rank_indices)
    assert all(len(uids) == len(set(uids)) for uids in rank_uids)
    assert set(rank_indices[0]).isdisjoint(rank_indices[1])
    assert set(rank_uids[0]).isdisjoint(rank_uids[1])
    assert set().union(*(set(indices) for indices in rank_indices)) == set(
        plan["selected_indices"]
    )
    assert set().union(*(set(uids) for uids in rank_uids)) == set(
        plan["selected_uids"]
    )


def test_balanced_sampler_rejects_invalid_dataset_plans():
    cases = [
        (_plan((0, 8, 2, 3)), "range"),
        (_plan((0, 0, 2, 3)), "unique"),
        (_plan(sr_quotas={8000: 4}), "sample-rate"),
        (_plan(cell_quotas={("zh", "set", 16000): 3}), "cell quota"),
        (_plan(cell_quotas={("zh", "set-a", 16000): 2, ("zh", "set-b", 16000): 2}), "cell quota"),
    ]
    for plan, match in cases:
        with pytest.raises(ValueError, match=match):
            GroupedBatchSampler(_BalancedDataset(plan), batch_size=1, num_per_epoch=4)


def test_balanced_drop_last_false_keeps_aligned_partial_batches_without_collectives(monkeypatch):
    plan = _plan((0, 1, 2, 3, 4, 5))
    dataset = _BalancedDataset(plan)
    calls = []
    monkeypatch.setattr("torch.distributed.all_reduce", lambda *args, **kwargs: calls.append(args))

    samplers = [
        GroupedBatchSampler(dataset, batch_size=2, rank=rank, world_size=2, drop_last=False)
        for rank in range(2)
    ]
    shards = [list(sampler) for sampler in samplers]

    assert calls == []
    assert shards == [[[0, 2], [4]], [[1, 3], [5]]]
    assert len(shards[0]) == len(shards[1]) == 2
    assert set(index for batches in shards for batch in batches for index in batch) == set(plan["selected_indices"])


def test_balanced_multi_sr_sharding_never_mixes_sample_rates(monkeypatch):
    plan = {
        "selected_indices": tuple(range(8)),
        "selected_uids": tuple(f"uid-{index}" for index in range(8)),
        "sample_rates": (16000, 16000, 16000, 16000, 8000, 8000, 8000, 8000),
        "speech_plan": {
            "actual": 8,
            "sr_quotas": {16000: 4, 8000: 4},
            "cell_quotas": {("zh", "set", 16000): 8},
        },
    }
    dataset = _BalancedDataset(plan, srs=plan["sample_rates"])
    calls = []
    monkeypatch.setattr("torch.distributed.all_reduce", lambda *args, **kwargs: calls.append(args))
    samplers = [
        GroupedBatchSampler(dataset, batch_size=2, rank=rank, world_size=2, drop_last=True)
        for rank in range(2)
    ]
    shards = [list(sampler) for sampler in samplers]

    assert calls == []
    assert [len(shard) for shard in shards] == [2, 2]
    assert all(len({dataset.get_srs()[index] for index in batch}) == 1 for shard in shards for batch in shard)
    assert set(index for shard in shards for batch in shard for index in batch) == set(range(8))


def test_balanced_multi_sr_partial_batches_never_mix_sample_rates():
    plan = {
        "selected_indices": tuple(range(12)),
        "selected_uids": tuple(f"uid-{index}" for index in range(12)),
        "sample_rates": (16000,) * 6 + (8000,) * 6,
        "speech_plan": {
            "actual": 12,
            "sr_quotas": {16000: 6, 8000: 6},
            "cell_quotas": {("zh", "set", 16000): 12},
        },
    }
    dataset = _BalancedDataset(plan, srs=plan["sample_rates"])
    samplers = [
        GroupedBatchSampler(dataset, batch_size=2, rank=rank, world_size=2, drop_last=False)
        for rank in range(2)
    ]
    shards = [list(sampler) for sampler in samplers]

    assert [len(shard) for shard in shards] == [4, 4]
    assert all(len({dataset.get_srs()[index] for index in batch}) == 1 for shard in shards for batch in shard)
    assert sorted(len(batch) for batch in shards[0]) == [1, 1, 2, 2]
    assert sorted(len(batch) for batch in shards[1]) == [1, 1, 2, 2]


def test_balanced_len_is_initialized_from_epoch_zero_plan_and_cannot_drift():
    plan = _plan((0, 1, 2, 3))
    datasets = [_BalancedDataset(plan) for _ in range(2)]
    samplers = [
        GroupedBatchSampler(dataset, batch_size=2, rank=rank, world_size=2, num_per_epoch=4)
        for rank, dataset in enumerate(datasets)
    ]

    assert [len(sampler) for sampler in samplers] == [1, 1]
    for sampler in samplers:
        sampler.set_epoch(3)
    assert [len(sampler) for sampler in samplers] == [1, 1]
    assert [len(list(sampler)) for sampler in samplers] == [1, 1]

    class _DriftingDataset(_BalancedDataset):
        def get_speech_epoch_plan(self, **kwargs):
            return _plan((0, 1, 2, 3)) if kwargs["epoch"] == 0 else _plan((0, 1))

    drifting = GroupedBatchSampler(_DriftingDataset(_plan()), batch_size=1, num_per_epoch=4)
    state_before = (
        drifting.epoch,
        drifting._balanced_plan,
        drifting._balanced_batches,
        drifting._last_iter_len,
        drifting._last_iter_real_len,
    )
    with pytest.raises(ValueError, match="batch count changed"):
        drifting.set_epoch(1)
    assert (
        drifting.epoch,
        drifting._balanced_plan,
        drifting._balanced_batches,
        drifting._last_iter_len,
        drifting._last_iter_real_len,
    ) == state_before


def test_balanced_snapshot_reads_duck_plan_once_and_rejects_authority_mismatches():
    class _ChangingPlan:
        def __init__(self):
            self.reads = 0

        @property
        def selected_indices(self):
            self.reads += 1
            return (0, 1, 2, 3) if self.reads == 1 else (4, 5, 6, 7)

        selected_uids = ("uid-0", "uid-1", "uid-2", "uid-3")
        selected_cell_keys = (("zh", "set", 16000),) * 4
        speech_plan = {"actual": 4, "sr_quotas": {16000: 4}, "cell_quotas": {("zh", "set", 16000): 4}}

    plan = _ChangingPlan()
    sampler = GroupedBatchSampler(_BalancedDataset(plan), batch_size=1, num_per_epoch=4)
    assert plan.reads == 1
    assert list(sampler) == [[0], [1], [2], [3]]

    cells = _dataset_plan_fixture().speech_sampling_cells
    authoritative_plan = _plan(
        (0, 1, 2, 3),
        cell_quotas={("zh", "set-a", 16000): 4},
        cell_keys=(("zh", "set-a", 16000),) * 4,
    )
    dataset = _BalancedDataset(authoritative_plan, enabled=False)
    dataset.speech_sampling_cells = cells
    bad_uid = _plan((0, 1, 2, 3), cell_quotas={("zh", "set-a", 16000): 4})
    bad_uid["selected_uids"] = ("wrong", "a1", "a2", "a3")
    bad_uid["selected_cell_keys"] = (("zh", "set-a", 16000),) * 4
    with pytest.raises(ValueError, match="UID"):
        GroupedBatchSampler(dataset, batch_size=1, num_per_epoch=4)._validate_balanced_plan(bad_uid)

    bad_cell = _plan((0, 1, 2, 3), cell_quotas={("zh", "set-b", 16000): 4})
    bad_cell["selected_uids"] = ("b0", "b1", "b2", "b3")
    bad_cell["selected_cell_keys"] = (("zh", "set-b", 16000),) * 4
    with pytest.raises(ValueError, match="cell key"):
        GroupedBatchSampler(dataset, batch_size=1, num_per_epoch=4)._validate_balanced_plan(bad_cell)


@pytest.mark.parametrize("size", [5_000, 20_000])
def test_balanced_ownership_validation_is_linear_without_per_selection_index(size):
    class IndexSpy(tuple):
        index_calls = 0
        iterated_items = 0

        def index(self, *args, **kwargs):
            type(self).index_calls += 1
            return super().index(*args, **kwargs)

        def __iter__(self):
            for value in super().__iter__():
                type(self).iterated_items += 1
                yield value

    indices = IndexSpy(range(size))
    IndexSpy.index_calls = 0
    IndexSpy.iterated_items = 0
    cell_key = ("zh", "large", 16000)
    uids = tuple(f"uid-{index}" for index in range(size))
    plan = {
        "selected_indices": tuple(range(size)),
        "selected_uids": uids,
        "selected_cell_keys": (cell_key,) * size,
        "speech_plan": {
            "actual": size,
            "sr_quotas": {16000: size},
            "cell_quotas": {cell_key: size},
        },
    }
    dataset = _BalancedDataset(plan, srs=[16000] * size)
    dataset.speech_sampling_cells = {
        cell_key: type("Cell", (), {"external_indices": indices, "uids": uids})()
    }

    GroupedBatchSampler(dataset, batch_size=1, num_per_epoch=size)

    assert IndexSpy.index_calls == 0
    assert IndexSpy.iterated_items <= size


def test_balanced_plan_accepts_explicit_none_cell_keys():
    plan = _plan()
    plan["selected_cell_keys"] = None

    sampler = GroupedBatchSampler(_BalancedDataset(plan), batch_size=1, num_per_epoch=4)

    assert list(sampler) == [[0], [1], [2], [3]]


def test_balanced_resume_reproduces_dataset_plan_shard():
    plan = _plan((4, 5, 6, 7))
    first = GroupedBatchSampler(_BalancedDataset(plan), batch_size=1, rank=1, world_size=2, seed=17)
    resumed = GroupedBatchSampler(_BalancedDataset(plan), batch_size=1, rank=1, world_size=2, seed=17)

    first.set_epoch(9)
    resumed.set_epoch(9)

    assert list(first) == list(resumed) == [[5], [7]]


def test_disabled_dataset_keeps_legacy_sampler_behavior():
    legacy = _LegacyDataset()
    sampler = GroupedBatchSampler(legacy, batch_size=2, seed=3, num_per_epoch=4)
    sampler.set_epoch(1)

    assert len(list(sampler)) == 2
    assert not hasattr(legacy, "plan_requests")


# ---------------------------------------------------------------------------
# Task 5: Dataset leaf 构建与全局 epoch plan 物化
# ---------------------------------------------------------------------------


def _leaf_builder_fixture(metadata_by_uid, cached_srs, config):
    """直接驱动 ``_build_speech_sampling_leaves`` 的最小 dataset fixture。"""
    dataset = object.__new__(BaseDynamicMixingDataset)
    uids = list(metadata_by_uid)
    dataset._filtered_indices = [(16000, index) for index in range(len(uids))]
    dataset.speech_uids = {16000: uids}
    dataset._speech_manifest_metadata = metadata_by_uid
    dataset._cached_srs = list(cached_srs)
    dataset._speech_sampling_config = config
    return dataset


def test_leaf_batching_sr_uses_get_srs_not_manifest_fs():
    """Step 1：leaf key 的 batching_sr 以 get_srs() 为准，不用 manifest fs。"""
    metadata = {
        "u0": {"language": "zh", "dataset_id": "set-a", "fs": 48000, "length": 48000},
        "u1": {"language": "zh", "dataset_id": "set-a", "fs": 48000, "length": 48000},
    }
    config = SamplingConfig.from_dict({"speech": {"language_probs": {"zh": 1}}}).speech

    dataset = _leaf_builder_fixture(metadata, [16000, 16000], config)
    assert set(dataset._build_speech_sampling_leaves()) == {("zh", "set-a", None, 16000)}

    # 同一 manifest fs、不同 get_srs() 的 external indices 分到不同 leaf
    split = _leaf_builder_fixture(metadata, [16000, 8000], config)
    assert set(split._build_speech_sampling_leaves()) == {
        ("zh", "set-a", None, 16000),
        ("zh", "set-a", None, 8000),
    }


def test_speaker_leaves_merge_to_none_when_controller_disabled():
    """Step 2：未启用 speaker 时同三元多 speaker 聚合为一个 None leaf。"""
    metadata = {
        "u0": {"language": "zh", "dataset_id": "set-a", "fs": 16000, "length": 16000, "speaker_id": "s1"},
        "u1": {"language": "zh", "dataset_id": "set-a", "fs": 16000, "length": 16000, "speaker_id": "s2"},
    }
    config = SamplingConfig.from_dict({"speech": {"language_probs": {"zh": 1}}}).speech
    dataset = _leaf_builder_fixture(metadata, [16000, 16000], config)

    leaves = dataset._build_speech_sampling_leaves()
    assert set(leaves) == {("zh", "set-a", None, 16000)}
    assert leaves[("zh", "set-a", None, 16000)].uids == ("u0", "u1")


def test_speaker_leaves_carry_canonical_identity_when_enabled():
    """Step 2：启用 speaker 时 leaf key 含 canonical (dataset_id, speaker_id)。"""
    metadata = {
        "u0": {"language": "zh", "dataset_id": "set-a", "fs": 16000, "length": 16000, "speaker_id": "s1"},
        "u1": {"language": "zh", "dataset_id": "set-a", "fs": 16000, "length": 16000, "speaker_id": "s2"},
    }
    config = SamplingConfig.from_dict(
        {"speech": {"speaker": {"max_samples_per_epoch": 10}}}
    ).speech
    dataset = _leaf_builder_fixture(metadata, [16000, 16000], config)

    leaves = dataset._build_speech_sampling_leaves()
    assert set(leaves) == {
        ("zh", "set-a", ("set-a", "s1"), 16000),
        ("zh", "set-a", ("set-a", "s2"), 16000),
    }


def test_speaker_leaves_reject_missing_identity_when_enabled():
    """Step 2：启用 speaker 但记录缺 speaker_id → 构造期 fail-fast。"""
    metadata = {
        "u0": {"language": "zh", "dataset_id": "set-a", "fs": 16000, "length": 16000},
    }
    config = SamplingConfig.from_dict(
        {"speech": {"speaker": {"max_samples_per_epoch": 10}}}
    ).speech
    dataset = _leaf_builder_fixture(metadata, [16000], config)

    with pytest.raises(ValueError, match="metadata invalid"):
        dataset._build_speech_sampling_leaves()


def test_dataset_epoch_plan_leaf_quotas_match_selected_leaf_counts():
    """Step 3：leaf-local 稳定无放回物化；quota 不变量跨 epoch 保持。"""
    dataset = _dataset_plan_fixture()
    plan_a = dataset.get_speech_epoch_plan(
        requested=4, world_size=2, batch_size=2, drop_last=True, seed=17, epoch=3
    )
    plan_b = dataset.get_speech_epoch_plan(
        requested=4, world_size=2, batch_size=2, drop_last=True, seed=17, epoch=3
    )
    assert plan_a.selected_indices == plan_b.selected_indices
    assert len(set(plan_a.selected_indices)) == len(plan_a.selected_indices)

    leaf_of = {
        index: key
        for key, leaf in dataset.speech_sampling_leaves.items()
        for index in leaf.external_indices
    }
    assert Counter(leaf_of[index] for index in plan_a.selected_indices) == Counter(
        plan_a.speech_plan.leaf_quotas
    )

    changed = dataset.get_speech_epoch_plan(
        requested=4, world_size=2, batch_size=2, drop_last=True, seed=17, epoch=4
    )
    # 跨 epoch 允许选择变化：按 leaf 比较选中集合（与全局顺序无关，降低对
    # 具体哈希值的耦合）；quota 不变量必须保持
    selected_sets_a = {
        key: {
            index for index in plan_a.selected_indices if leaf_of[index] == key
        }
        for key in dataset.speech_sampling_leaves
    }
    selected_sets_b = {
        key: {
            index for index in changed.selected_indices if leaf_of[index] == key
        }
        for key in dataset.speech_sampling_leaves
    }
    assert any(
        selected_sets_a[key] != selected_sets_b[key] for key in selected_sets_a
    )
    assert Counter(leaf_of[index] for index in changed.selected_indices) == Counter(
        plan_a.speech_plan.leaf_quotas
    )


def test_dataset_plan_cache_rejects_changed_ddp_parameters():
    """Step 4：同 key 复用 quota plan；DDP 参数变化必须 fail-fast。"""
    dataset = _dataset_plan_fixture()
    dataset.get_speech_epoch_plan(
        requested=4, world_size=2, batch_size=1, drop_last=False, seed=13, epoch=2
    )
    with pytest.raises(RuntimeError, match="parameters changed"):
        dataset.get_speech_epoch_plan(
            requested=8, world_size=2, batch_size=1, drop_last=False, seed=13, epoch=2
        )


def test_balanced_world_size_4_shards_are_disjoint_and_complete():
    """Step 5：world_size=4 下 shard 两两无交集、并集等于全局 plan、batch 数一致。"""
    plan = _plan(tuple(range(8)))
    dataset = _BalancedDataset(plan)
    samplers = [
        GroupedBatchSampler(dataset, batch_size=1, rank=rank, world_size=4, seed=7, num_per_epoch=8)
        for rank in range(4)
    ]
    shards = [list(sampler) for sampler in samplers]

    assert [len(shard) for shard in shards] == [2, 2, 2, 2]
    per_rank = [
        set(index for batch in shard for index in batch) for shard in shards
    ]
    for left in range(4):
        for right in range(left + 1, 4):
            assert per_rank[left].isdisjoint(per_rank[right])
    assert set().union(*per_rank) == set(plan["selected_indices"])


pytestmark = [pytest.mark.unit]
