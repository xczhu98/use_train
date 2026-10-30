"""Grouped batch sampler — group by sample rate, same SR within each batch.

Source: urgent2026 baseline_code/dataset.py, adapted for DDP multi-GPU training.

How it works:
  1. Group all samples by sample rate
  2. Sort by audio length within each group, split into buckets
  3. Each epoch: ``set_epoch(epoch)`` 用 ``seed`` 与 ``epoch`` 派生的确定性 PRNG 刷新
     ``_shuffle_salt``，再 shuffle buckets / 桶内顺序 / batch 顺序（同 seed+epoch 可复现）
  4. DDP: each rank takes its subset (stride=world_size)
  5. DDP 下通过 dist.all_reduce(MAX) 使各 rank 的 batch 数一致；补齐时只复制
     同一 SR 桶内的 batch / 样本，绝不跨桶拼 batch（collate 要求同 batch 同 SR）。
  6. ``__len__`` 构造期使用虚拟 SR 比例估算；显式 ``set_epoch()`` 后按
     该 epoch 的具体 buckets 更新。``batch_count_for_epoch()`` 可无副作用精确查询
     完整 epoch 计划，供 auto per-step scheduler 求和且不触发 collective。

DDP train 保留既有 MAX 对齐与同 SR dummy padding；启用
``require_non_empty_train_shards`` 时由全局 SR sizes 预检每个 rank 是否拥有同 SR
本地样本，且全局至少有一个真实 batch；有本地样本的零 batch rank 保留 dummy padding。
显式启用 ``require_non_empty_all_ranks`` 的普通、虚拟和 balanced validation 均要求每个
rank 在 padding 前至少有一个真实 batch；全局预检在状态提交和任何 collective 前对称失败。
"""

import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch.utils.data import BatchSampler


@dataclass(frozen=True)
class _BalancedPlanSnapshot:
    selected_indices: tuple[int, ...]
    selected_uids: tuple[str, ...]
    sample_rates: tuple[int, ...]
    selected_cell_keys: tuple[tuple[str, str, int], ...] | None
    actual: int
    sr_quotas: Mapping
    cell_quotas: Mapping


class GroupedBatchSampler(BatchSampler):

    def __init__(
        self,
        dataset,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        drop_last: bool = False,
        bucket_size_mult: int = 100,
        sampler=None,
        local_rank: int | None = None,
        num_per_epoch: int | None = None,
        require_non_empty_all_ranks: bool = False,
        require_non_empty_train_shards: bool = False,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.bucket_size = batch_size * bucket_size_mult
        self.epoch = 0
        self.rank = rank
        self.local_rank = rank if local_rank is None else int(local_rank)
        self.world_size = world_size
        self.seed = seed
        self.num_per_epoch = num_per_epoch
        self.require_non_empty_all_ranks = require_non_empty_all_ranks
        self.require_non_empty_train_shards = require_non_empty_train_shards
        self.dataset = dataset
        self._balanced = getattr(dataset, "balanced_speech_sampling_enabled", False) is True
        self._balanced_plan = None
        self._balanced_batches: list[list[int]] | None = None

        # C-01: 保存全量索引信息用于虚拟 epoch 重采样（num_per_epoch 模式）
        self._idx_sr = list(dataset.get_srs())
        self._full_source_length = dataset.get_source_length()
        self._all_indices = list(range(len(self._idx_sr)))

        self._build_buckets(self._all_indices)
        if num_per_epoch is not None:
            self._sr_group_sizes = self._virtual_sr_sizes()
        self._len_estimate = self._compute_target_batches()
        self._last_iter_len: int | None = None
        self._last_iter_real_len: int | None = None
        self._prepared_epoch: int | None = None
        # 每 epoch 在 set_epoch 中由 (seed, epoch) 确定性刷新（resume / 复现实验对齐）
        self._shuffle_salt: int = 0
        if self._balanced:
            self._set_balanced_epoch_plan()

    def _bucket_state_for_indices(self, indices):
        """纯函数构造候选 buckets、全局 SR sizes 与本 rank 样本数。"""
        sr_groups = defaultdict(list)
        for idx in indices:
            sr_groups[self._idx_sr[idx]].append(idx)
        buckets = []
        sr_sizes = []
        for _sr, idxs in sr_groups.items():
            si = sorted(idxs, key=lambda x: self._full_source_length[x])
            # DDP: each rank takes its shard (different data per rank)
            si = si[self.rank::self.world_size]
            for i in range(0, len(si), self.bucket_size):
                buckets.append(si[i:i + self.bucket_size])
            sr_sizes.append(len(idxs))
        return buckets, sr_sizes, sum(len(bucket) for bucket in buckets)

    def _build_buckets(self, indices):
        """从索引集（全量或虚拟 epoch 子集）提交 DDP-strided buckets。"""
        self.buckets, self._sr_group_sizes, self._local_samples = (
            self._bucket_state_for_indices(indices)
        )

    def _virtual_indices_for_epoch(self, epoch: int) -> list[int]:
        """返回虚拟 epoch 的确定性全局索引。"""
        if self.num_per_epoch is None or self.num_per_epoch >= len(self._all_indices):
            return self._all_indices
        sample_rng = random.Random((self.seed * 1_000_007) ^ epoch)
        return sample_rng.sample(self._all_indices, self.num_per_epoch)

    def _virtual_sr_sizes(self) -> list[int]:
        """按全局 SR 比例 floor 估算 num_per_epoch 的各 SR sizes。"""
        total = len(self._all_indices)
        if total == 0:
            return [0]
        effective = min(self.num_per_epoch, total)
        sr_counts = Counter(self._idx_sr)
        return [int(effective * count / total) for count in sr_counts.values()]

    def _compute_rank_sample_counts(self, sr_group_sizes=None) -> list[int]:
        """按给定全局 SR sizes 纯函数计算每个 rank 的同 SR shard 样本数。"""
        sizes = self._sr_group_sizes if sr_group_sizes is None else sr_group_sizes
        ws = max(1, self.world_size)
        return [
            sum(max(0, (group_size - rank + ws - 1) // ws) for group_size in sizes)
            for rank in range(ws)
        ]

    def _compute_rank_batch_counts(self, sr_group_sizes=None) -> list[int]:
        """按给定全局 SR sizes 纯函数计算每个 rank 的真实 batch 数。"""

        def _local_count(shard_sizes: list[int]) -> int:
            n = 0
            for s in shard_sizes:
                for i in range(0, s, self.bucket_size):
                    m = min(self.bucket_size, s - i)
                    if self.drop_last:
                        n += m // self.batch_size
                    else:
                        n += (m + self.batch_size - 1) // self.batch_size
            return n

        sizes = self._sr_group_sizes if sr_group_sizes is None else sr_group_sizes
        ws = max(1, self.world_size)
        return [
            _local_count(
                [max(0, (group_size - rank + ws - 1) // ws) for group_size in sizes]
            )
            for rank in range(ws)
        ]

    def _compute_target_batches(self, sr_group_sizes=None) -> int:
        """由给定各 SR 桶样本数确定性计算 DDP 对齐后的 batch 数。"""
        return max(self._compute_rank_batch_counts(sr_group_sizes), default=0)

    def batch_count_for_epoch(self, epoch: int) -> int:
        """不修改 sampler 状态地返回指定 epoch 的 DDP 对齐 batch 数。"""
        if self._balanced or self.num_per_epoch is None:
            return self._len_estimate
        indices = self._virtual_indices_for_epoch(epoch)
        sr_group_sizes = list(
            Counter(self._idx_sr[index] for index in indices).values()
        )
        return self._compute_target_batches(sr_group_sizes)

    def _validate_train_shards(self, sr_group_sizes, *, epoch: int) -> None:
        """在 collective 前由全局 SR sizes 对所有 rank 做确定性 train 门禁。"""
        if not self.require_non_empty_train_shards or self.world_size <= 1:
            return
        rank_sample_counts = self._compute_rank_sample_counts(sr_group_sizes)
        rank_batch_counts = self._compute_rank_batch_counts(sr_group_sizes)
        zero_samples = any(count == 0 for count in rank_sample_counts)
        zero_epoch = all(count == 0 for count in rank_batch_counts)
        if not zero_samples and not zero_epoch:
            return
        detail = "0 local samples" if zero_samples else "0 real batches on every rank"
        raise ValueError(
            "GroupedBatchSampler train preflight would give at least one DDP rank "
            f"{detail}: epoch={epoch}, world_size={self.world_size}, "
            f"batch_size={self.batch_size}, drop_last={self.drop_last}, "
            f"num_per_epoch={self.num_per_epoch}, sr_group_sizes={list(sr_group_sizes)}, "
            f"rank_sample_counts={rank_sample_counts}, "
            f"rank_batch_counts={rank_batch_counts}"
        )

    def preflight_train_shards(self, epoch: int = 0) -> None:
        """校验指定 train epoch 是否每个 rank 都有同 SR dummy 的本地来源。"""
        if self._balanced:
            sr_group_sizes = list(Counter(self._balanced_plan.sample_rates).values())
        else:
            indices = self._virtual_indices_for_epoch(epoch)
            sr_group_sizes = list(Counter(self._idx_sr[index] for index in indices).values())
        self._validate_train_shards(sr_group_sizes, epoch=epoch)

    def _validate_validation_shards(self, sr_group_sizes, *, epoch: int) -> None:
        """在 collective 前由全局 SR sizes 对所有 rank 做严格 validation 门禁。"""
        if not self.require_non_empty_all_ranks or self.world_size <= 1:
            return
        rank_batch_counts = self._compute_rank_batch_counts(sr_group_sizes)
        if any(count == 0 for count in rank_batch_counts):
            raise ValueError(
                "GroupedBatchSampler validation preflight would give at least one "
                f"DDP rank 0 real batch before padding: epoch={epoch}, "
                f"world_size={self.world_size}, batch_size={self.batch_size}, "
                f"drop_last={self.drop_last}, num_per_epoch={self.num_per_epoch}, "
                f"sr_group_sizes={list(sr_group_sizes)}, "
                f"rank_sample_counts={self._compute_rank_sample_counts(sr_group_sizes)}, "
                f"rank_batch_counts={rank_batch_counts}"
            )

    def set_epoch(self, epoch: int) -> None:
        if self._prepared_epoch == epoch:
            return
        if self._balanced:
            self._set_balanced_epoch_plan(epoch)
            return

        # 所有候选状态先在 locals 中生成并校验，失败时 sampler 保持原状态。
        rng = random.Random((self.seed * 1_000_003) ^ epoch)
        candidate_salt = rng.getrandbits(31)
        candidate_buckets = self.buckets
        candidate_sr_sizes = self._sr_group_sizes
        candidate_local_samples = self._local_samples
        candidate_len = self._len_estimate

        # C-01: 虚拟 epoch — 每 epoch 从全量索引 random.sample(num_per_epoch) 重采样。
        # 各 rank 用相同 (seed,epoch) 派生 PRNG；num_per_epoch >= 全量时退化为全量。
        if self.num_per_epoch is not None:
            candidate_indices = self._virtual_indices_for_epoch(epoch)
            candidate_buckets, candidate_sr_sizes, candidate_local_samples = (
                self._bucket_state_for_indices(candidate_indices)
            )
            candidate_len = self._compute_target_batches(candidate_sr_sizes)

        self._validate_validation_shards(candidate_sr_sizes, epoch=epoch)
        self._validate_train_shards(candidate_sr_sizes, epoch=epoch)
        self.epoch = epoch
        self._shuffle_salt = candidate_salt
        self.buckets = candidate_buckets
        self._sr_group_sizes = candidate_sr_sizes
        self._local_samples = candidate_local_samples
        self._len_estimate = candidate_len
        self._prepared_epoch = epoch
        self._last_iter_len = None
        self._last_iter_real_len = None

    def _set_balanced_epoch_plan(self, epoch: int | None = None) -> None:
        candidate_epoch = self.epoch if epoch is None else epoch
        request_size = self.num_per_epoch if self.num_per_epoch is not None else len(set(self._all_indices))
        getter = getattr(self.dataset, "get_speech_epoch_plan", None)
        if not callable(getter):
            raise RuntimeError("balanced speech sampling requires dataset.get_speech_epoch_plan()")
        plan = getter(
            requested=request_size,
            world_size=self.world_size,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            seed=self.seed,
            epoch=candidate_epoch,
        )
        snapshot = self._canonicalize_balanced_plan(plan)
        self._validate_balanced_plan(snapshot)
        sr_groups: dict[int, list[int]] = {}
        for index, sample_rate in zip(snapshot.selected_indices, snapshot.sample_rates):
            sr_groups.setdefault(sample_rate, []).append(index)
        batches: list[list[int]] = []
        for indices in sr_groups.values():
            local = indices[self.rank::self.world_size]
            if self.drop_last and len(local) % self.batch_size:
                raise ValueError("balanced plan is not DDP batch aligned")
            batches.extend(
                local[offset : offset + self.batch_size]
                for offset in range(0, len(local), self.batch_size)
            )
        candidate_len = len(batches)
        if self._balanced_plan is not None and candidate_len != self._len_estimate:
            raise ValueError("balanced plan batch count changed across epochs")
        candidate_sr_sizes = list(Counter(snapshot.sample_rates).values())
        self._validate_validation_shards(candidate_sr_sizes, epoch=candidate_epoch)
        self._validate_train_shards(candidate_sr_sizes, epoch=candidate_epoch)
        self.epoch = candidate_epoch
        self._len_estimate = candidate_len
        self._balanced_plan = snapshot
        self._balanced_batches = batches
        self._prepared_epoch = candidate_epoch
        self._last_iter_len = None
        self._last_iter_real_len = None

    @staticmethod
    def _plan_value(plan, name):
        if isinstance(plan, Mapping):
            return plan[name]
        return getattr(plan, name)

    def _canonicalize_balanced_plan(self, plan) -> _BalancedPlanSnapshot:
        """Read a duck plan exactly once into an immutable sampler-owned snapshot."""
        indices = tuple(self._plan_value(plan, "selected_indices"))
        uids = tuple(self._plan_value(plan, "selected_uids"))
        has_sample_rates = isinstance(plan, Mapping) and "sample_rates" in plan or hasattr(plan, "sample_rates")
        sample_rates = tuple(self._plan_value(plan, "sample_rates")) if has_sample_rates else tuple(
            self._idx_sr[index] for index in indices if 0 <= index < len(self._idx_sr)
        )
        has_cell_keys = isinstance(plan, Mapping) and "selected_cell_keys" in plan or hasattr(plan, "selected_cell_keys")
        raw_cell_keys = self._plan_value(plan, "selected_cell_keys") if has_cell_keys else None
        cell_keys = tuple(raw_cell_keys) if raw_cell_keys is not None else None
        summary = self._plan_value(plan, "speech_plan")
        actual = self._plan_value(summary, "actual")
        sr_quotas = MappingProxyType(dict(self._plan_value(summary, "sr_quotas")))
        cell_quotas = MappingProxyType(dict(self._plan_value(summary, "cell_quotas")))
        return _BalancedPlanSnapshot(indices, uids, sample_rates, cell_keys, actual, sr_quotas, cell_quotas)

    def _validate_balanced_plan(self, plan) -> None:
        snapshot = plan if isinstance(plan, _BalancedPlanSnapshot) else self._canonicalize_balanced_plan(plan)
        indices = snapshot.selected_indices
        uids = snapshot.selected_uids
        sample_rates = snapshot.sample_rates
        cell_keys = snapshot.selected_cell_keys
        if any(not 0 <= index < len(self._idx_sr) for index in indices):
            raise ValueError("balanced plan index range is invalid")
        if len(indices) != snapshot.actual or len(indices) != len(uids) or len(indices) != len(sample_rates):
            raise ValueError("balanced plan actual does not match selections")
        if len(set(indices)) != len(indices) or len(set(uids)) != len(uids):
            raise ValueError("balanced plan selections must be unique")
        if tuple(self._idx_sr[index] for index in indices) != sample_rates:
            raise ValueError("balanced plan sample-rate mapping is invalid")
        if dict(Counter(sample_rates)) != dict(snapshot.sr_quotas):
            raise ValueError("balanced plan sample-rate quotas do not match selections")
        if sum(snapshot.cell_quotas.values()) != len(indices):
            raise ValueError("balanced plan cell quota total does not match selections")
        if cell_keys is not None and Counter(cell_keys) != Counter(snapshot.cell_quotas):
            raise ValueError("balanced plan cell quota does not match selections")
        for sample_rate, quota in snapshot.sr_quotas.items():
            divisor = self.world_size * self.batch_size if self.drop_last else self.world_size
            if quota % divisor:
                raise ValueError("balanced plan sample-rate quota is not DDP batch aligned")
        cells = getattr(self.dataset, "speech_sampling_cells", None)
        if cells is not None and cell_keys is not None:
            ownership = {}
            for cell_key in set(cell_keys):
                cell = cells.get(cell_key)
                if cell is None:
                    raise ValueError("balanced plan cell key is not owned by dataset")
                ownership[cell_key] = dict(zip(cell.external_indices, cell.uids))
            for index, uid, cell_key in zip(indices, uids, cell_keys):
                owned_uid = ownership[cell_key].get(index)
                if owned_uid is None:
                    raise ValueError("balanced plan cell key does not own selected index")
                if owned_uid != uid:
                    raise ValueError("balanced plan UID does not match dataset index")

    def __iter__(self):
        if self.require_non_empty_all_ranks and self._prepared_epoch is None:
            self.set_epoch(self.epoch)
        if self._balanced:
            if self._balanced_batches is None:
                self._set_balanced_epoch_plan()
            self._last_iter_real_len = len(self._balanced_batches)
            self._last_iter_len = len(self._balanced_batches)
            return iter(self._balanced_batches)
        rng = random.Random(self.seed + self.epoch * 1000003 + self._shuffle_salt)
        # 在副本上 shuffle：self.buckets 必须保持构造时的不可变初态，
        # 否则连续迭代与 fresh resume 的桶初态不同，破坏同 seed+epoch 的
        # resume determinism（审计 A15）
        buckets = [list(b) for b in self.buckets]
        rng.shuffle(buckets)
        batches = []
        for bi, bucket in enumerate(buckets):
            rng_b = random.Random(self.seed + self.epoch + bi + self._shuffle_salt)
            rng_b.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                b = bucket[i:i + self.batch_size]
                if len(b) < self.batch_size and self.drop_last:
                    continue
                batches.append(b)
        rng.shuffle(batches)

        # DDP safety: use all_reduce(MAX) so every rank agrees on exactly the
        # same target batch count.  Per-rank sample counts can differ by up to
        # (world_size - 1) after strided sharding, producing different local
        # batch counts and causing DDP deadlock（broadcast_buffers=True 时每次
        # forward 都有 buffer 广播 collective，各 rank 迭代数必须一致）。
        # train 侧复制末尾 batch 无害（少量重复梯度）；val 侧复制批的 forward
        # 必须照常跑（维持同步），但 loss/指标贡献不得计入 —— trainer 依据
        # _last_iter_real_len 识别并剔除（见 core MetricsCapableTrainer
        # ._val_iter_is_ddp_padding）。
        if self.world_size > 1:
            import torch.distributed as dist
            # collective tensor 必须匹配 process-group backend，而不是 CUDA
            # 可见性：Gloo 即使能看到 GPU 也使用 CPU；NCCL 才使用 local CUDA。
            backend = str(dist.get_backend()).lower() if dist.is_initialized() else ""
            _device = (
                torch.device(f"cuda:{self.local_rank}")
                if backend == "nccl"
                else torch.device("cpu")
            )
            n = torch.tensor(len(batches), dtype=torch.int64, device=_device)
            dist.all_reduce(n, op=dist.ReduceOp.MAX)
            target_n = int(n.item())
        else:
            target_n = len(batches)

        # 本 rank 的真实批数（补齐/截断前），供 val 排除 DDP 对齐复制批的贡献。
        self._last_iter_real_len = min(len(batches), target_n)

        if len(batches) > target_n:
            batches = batches[:target_n]
        elif len(batches) < target_n:
            if batches:
                # batches[-1] 来自同一个 SR 桶，复制它不破坏同 batch 同 SR 约束
                while len(batches) < target_n:
                    batches.append(batches[-1])
            else:
                # 本 rank 有样本但没有任何合法 batch（如 drop_last 丢弃了所有不足
                # batch_size 的尾批）。只在单个 SR 桶内取样本凑 dummy batch
                # （不足 batch_size 时复制桶内最后一个样本），绝不跨桶拼接。
                src_bucket = max(self.buckets, key=len, default=None)
                if src_bucket:
                    if len(src_bucket) >= self.batch_size:
                        dummy = list(src_bucket[-self.batch_size:])
                    else:
                        dummy = list(src_bucket) + [src_bucket[-1]] * (self.batch_size - len(src_bucket))
                    batches = [list(dummy) for _ in range(target_n)]
                else:
                    # No indices at all on this rank (empty after stride).
                    if self.world_size > 1 and target_n > 0:
                        raise RuntimeError(
                            f"[GroupedBatchSampler] rank {self.rank} has 0 local samples but "
                            f"other ranks have {target_n} batch(es). This would cause a DDP "
                            f"deadlock on the next collective. "
                            f"Fix: ensure total samples >= world_size * batch_size, "
                            f"or reduce world_size."
                        )

        self._last_iter_len = len(batches)
        return iter(batches)

    def __len__(self) -> int:
        """返回构造期估算或最近一次成功 epoch/迭代的 batch 数。

        显式 ``set_epoch()`` 后，虚拟 epoch 长度按该 epoch 的具体 SR 分布更新；
        构造期估算仅用于 sampler 独立构造场景，trainer scheduler 会先初始化
        实际 ``start_epoch`` 再读取长度。本方法自身不调用 collective。
        """
        if self._last_iter_len is not None:
            return self._last_iter_len
        return self._len_estimate
