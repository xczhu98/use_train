"""AbsentTargetLoss 的真实跨 rank 有效样本归约与重复迭代回归。"""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

pytestmark = [pytest.mark.unit, pytest.mark.ddp]


def _absent_worker(rank, rendezvous):
    from core.losses.build import build_loss

    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        linear = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            linear.weight.fill_(2.)
        model = DistributedDataParallel(linear)
        loss_fn = build_loss({'type': 'CompositeLoss', 'components': [
            {'type': 'AbsentTargetLoss', 'weight': 1.},
        ]})
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            prediction = model(torch.ones(1, 2, 1)).squeeze(-1)
            target = torch.zeros_like(prediction) if rank == 0 else torch.ones_like(prediction)
            loss = loss_fn(prediction, target, torch.tensor([2]))
            loss.backward()
            # 全局只有 rank0 一个 absent 样本：d(2**2)/d(weight)=4。
            torch.testing.assert_close(linear.weight.grad, torch.full_like(linear.weight, 4.))
            reported = loss.detach().clone()
            dist.all_reduce(reported)
            torch.testing.assert_close(reported / 2, torch.tensor(4.))
    finally:
        dist.destroy_process_group()


def test_absent_target_global_mean_with_empty_rank_twice(tmp_path):
    rendezvous = (tmp_path / 'gloo-rendezvous').resolve().as_uri()
    mp.spawn(_absent_worker, args=(rendezvous,), nprocs=2, join=True)
