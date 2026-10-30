"""OnlineSpatialNet 逐层 gradient checkpointing 在 DDP 下的回归测试。"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from conftest import spawn_ddp_workers

pytestmark = [pytest.mark.unit, pytest.mark.ddp, pytest.mark.subprocess]

_WORLD_SIZE = 2
_TIMEOUT_SECONDS = 60


def _build_model() -> torch.nn.Module:
    from tasks.se.models.discriminative.online_spatialnet import OnlineSpatialNetSE

    return OnlineSpatialNetSE(
        n_fft=64,
        n_channel=2,
        ref_channel=0,
        num_layers=3,
        dim_hidden=16,
        dim_ffn=32,
        dim_squeeze=4,
        num_heads=2,
        attention="mhsa(8)",
        kernel_size=[5, 3],
        conv_groups=[4, 4],
        full_share=0,
        gradient_checkpointing=True,
    )


def _ddp_checkpoint_worker(rank: int, init_file: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=_TIMEOUT_SECONDS),
    )
    try:
        generator = torch.Generator().manual_seed(11)
        spectrum = torch.randn(1, 2, 33, 16, 2, generator=generator)
        target = torch.randn(1, 33, 16, 2, generator=generator)

        torch.manual_seed(7)
        model = DDP(_build_model(), find_unused_parameters=False)
        model.train()
        loss = F.l1_loss(model(spectrum), target)
        loss.backward()
        grads = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
        assert torch.isfinite(grads).all()

        # 与同种子构造、关闭 checkpoint 的同构 DDP 模型对拍梯度
        torch.manual_seed(7)
        reference = DDP(_build_model(), find_unused_parameters=False)
        reference.module.backbone.gradient_checkpointing = False
        reference.train()
        ref_loss = F.l1_loss(reference(spectrum), target)
        ref_loss.backward()
        ref_grads = torch.cat(
            [p.grad.reshape(-1) for p in reference.parameters() if p.grad is not None]
        )
        assert torch.isfinite(ref_grads).all()
        torch.testing.assert_close(grads, ref_grads, rtol=1e-4, atol=1e-5)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo backend unavailable",
)
def test_gradient_checkpointing_trains_under_ddp_with_shared_full_module(tmp_path):
    """full_share 共享 full 模块时，逐层 checkpoint 在 DDP 下必须完成 backward 且梯度不变。

    回归依据：use_reentrant=True 时共享参数的 DDP ready hook 在每个 checkpoint 段
    各触发一次，第一个 step 即抛 "Expected to mark a variable ready only once"。
    """
    init_file = str(Path(tmp_path) / "online_spatialnet_ckpt_init_store")
    spawn_ddp_workers(_ddp_checkpoint_worker, args=(init_file,), nprocs=_WORLD_SIZE)
