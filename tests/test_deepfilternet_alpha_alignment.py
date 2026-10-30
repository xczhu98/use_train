"""WS1 审计修复：DeepFilterNet alpha 损失帧对齐 / 帧长守卫回归测试。

覆盖:
  a) DeepFilterNetAlphaLoss 取 min_frames 对齐 pred/target 帧数：
     center=True 使预测帧数比 unfold 网格多 2 帧时不再广播错位崩溃，
     loss 有限且可反传（截断区梯度为 0）。
  b) DfAlphaLossComponent / DeepFilterNetFullLossWithLSNR 对短于
     frame_length 的输入早退返回图保留零，不抛异常。
  c) 传 lengths 时按有效帧构造 mask 做 masked mean（与
     DeepFilterNetLSNRLoss 同模式），全无效返回图保留零。
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from conftest import spawn_ddp_workers
from core.losses.model_specific.deepfilternet import (
    DfAlphaLossComponent,
    DeepFilterNetAlphaLoss,
    DeepFilterNetFullLossWithLSNR,
    DeepFilterNetLSNRLoss,
    compute_frame_snr,
)


def _alpha_inputs(b: int = 2, t: int = 640, fl: int = 320, hl: int = 160,
                  extra_frames: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 (pred_alpha, target_lsnr)。

    target 帧数由 unfold 网格 (t-fl)//hl+1 决定；pred_alpha 额外多
    extra_frames 帧，模拟 center=True STFT 的 padding 帧数差。
    """
    g = torch.Generator().manual_seed(7)
    clean = torch.randn(b, t, generator=g)
    noise = 0.3 * torch.randn(b, t, generator=g)
    target = compute_frame_snr(clean, noise, fl, hl)  # (b, (t-fl)//hl+1)
    n_pred = target.shape[-1] + extra_frames
    pred = torch.rand(b, n_pred, generator=g, requires_grad=True)
    return pred, target


def test_alpha_loss_frame_mismatch_truncates():
    """pred/target 帧数差 2（center=True）：min_frames 截断生效，不崩、可反传。"""
    pred, target = _alpha_inputs(extra_frames=2)
    assert target.shape[-1] == 3 and pred.shape[-1] == 5

    loss_fn = DeepFilterNetAlphaLoss(frame_length=320, hop_length=160)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)
    assert loss.requires_grad

    loss.backward()
    assert pred.grad is not None
    # 截断后仅前 3 帧参与，后 2 帧梯度为 0
    assert torch.all(pred.grad[:, 3:] == 0.0)


def test_alpha_component_short_input_graph_preserving_zero():
    """clean 短于 frame_length：DfAlphaLossComponent 早退返回图保留零不抛异常。"""
    loss_fn = DfAlphaLossComponent(frame_length=320, hop_length=160)
    enhanced = torch.randn(1, 128)
    clean = torch.randn(1, 128)
    noise = torch.randn(1, 128)
    predicted_alpha = torch.rand(1, 4, requires_grad=True)

    out = loss_fn(
        enhanced, clean, predicted_alpha=predicted_alpha, noise=noise
    )
    assert torch.isfinite(out)
    assert out.item() == 0.0
    assert out.requires_grad

    out.backward()
    assert predicted_alpha.grad is not None
    assert torch.all(predicted_alpha.grad == 0.0)


def test_full_loss_short_input_alpha_branch_guard():
    """FullLoss 的 clean 短于 lsnr_frame_length：alpha 分支跳过，
    总 loss 只含增强 loss（图保留零注入，不抛异常）。"""
    g = torch.Generator().manual_seed(3)
    enhanced = torch.randn(1, 128, generator=g)
    clean = torch.randn(1, 128, generator=g)
    noise = torch.randn(1, 128, generator=g)

    loss_fn = DeepFilterNetFullLossWithLSNR(
        enhancement_loss_type="L1WavLoss",
        lsnr_frame_length=320,
        lsnr_hop_length=160,
        use_alpha_loss=True,
    )
    total = loss_fn(
        enhanced,
        clean,
        predicted_lsnr=torch.randn(1, 1),
        predicted_alpha=torch.rand(1, 4, requires_grad=True),
        noise=noise,
    )
    assert torch.isfinite(total)
    assert total.requires_grad
    expected = torch.mean(torch.abs(enhanced - clean))
    assert torch.allclose(total, expected, atol=1e-6)
    total.backward()


def test_alpha_loss_lengths_masked_mean():
    """带 lengths：padding 帧被有效帧掩码排除，masked mean 与手算一致。"""
    b, t, fl, hl = 2, 640, 320, 160
    g = torch.Generator().manual_seed(11)
    clean = torch.randn(b, t, generator=g)
    noise = 0.3 * torch.randn(b, t, generator=g)
    target = compute_frame_snr(clean, noise, fl, hl)  # (2, 3)
    pred = torch.rand(b, target.shape[-1], generator=g, requires_grad=True)
    lengths = torch.tensor([640, 320])  # 样本 1 只保留第一帧有效

    loss_fn = DeepFilterNetAlphaLoss(frame_length=fl, hop_length=hl)
    loss = loss_fn(pred, target, lengths)

    min_frames = target.shape[-1]
    pred_t = pred[:, :min_frames]
    tgt_t = target[:, :min_frames]
    w_off = loss_fn._lsnr_mapping(tgt_t, loss_fn.lsnr_thresh, loss_fn.lsnr_min)
    l_off = (pred_t * w_off).square()
    w_on = loss_fn._lsnr_mapping(tgt_t, loss_fn.lsnr_thresh + 2.5, 0.0)
    l_on = 0.1 * ((1 - pred_t) * w_on).abs()
    n_valid = (lengths - fl) // hl + 1
    n_valid = torch.minimum(n_valid.clamp(min=0), torch.full_like(n_valid, min_frames))
    idx = torch.arange(min_frames).unsqueeze(0).expand(b, -1)
    mask = (idx < n_valid.unsqueeze(1)).to(pred.dtype)
    expected = ((l_off + l_on) * mask).sum() / mask.sum() * loss_fn.factor

    assert loss.requires_grad
    assert torch.allclose(loss, expected, atol=1e-7)


def test_alpha_loss_all_invalid_lengths_graph_preserving_zero():
    """lengths 全部不足以构成任何帧（<=0 有效帧）：返回图保留零可反传。"""
    pred, target = _alpha_inputs(extra_frames=0)
    lengths = torch.tensor([100, 50])
    loss_fn = DeepFilterNetAlphaLoss(frame_length=320, hop_length=160)

    loss = loss_fn(pred, target, lengths)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert loss.requires_grad

    loss.backward()
    assert pred.grad is not None
    assert torch.all(pred.grad == 0.0)


def test_alpha_loss_empty_frames_graph_preserving_zero():
    """target 帧数为 0（min_frames==0）：返回图保留零不抛异常。"""
    pred = torch.rand(2, 4, requires_grad=True)
    target_lsnr = torch.empty(2, 0)
    loss_fn = DeepFilterNetAlphaLoss(frame_length=320, hop_length=160)

    loss = loss_fn(pred, target_lsnr)
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert pred.grad is not None
    assert torch.all(pred.grad == 0.0)


def _assert_lsnr_zero_backpropagates(
    predicted_lsnr: torch.Tensor,
    clean: torch.Tensor,
    noise: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> None:
    loss_fn = DeepFilterNetLSNRLoss(frame_length=320, hop_length=160)
    loss = loss_fn(predicted_lsnr, clean, noise, lengths)

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert predicted_lsnr.grad is not None
    assert torch.all(predicted_lsnr.grad == 0.0)


def test_lsnr_loss_short_input_graph_preserving_zero():
    """波形短于 frame_length 时，零损失仍连接 LSNR 预测图。"""
    predicted_lsnr = torch.randn(2, 3, requires_grad=True)
    clean = torch.randn(2, 128)
    noise = torch.randn(2, 128)

    _assert_lsnr_zero_backpropagates(predicted_lsnr, clean, noise)


def test_lsnr_loss_empty_prediction_graph_preserving_zero():
    """min_frames=0 时，空预测张量仍可安全反传到上游图。"""
    predicted_lsnr = torch.empty(2, 0, requires_grad=True)
    clean = torch.randn(2, 640)
    noise = torch.randn(2, 640)

    _assert_lsnr_zero_backpropagates(predicted_lsnr, clean, noise)


def test_lsnr_loss_all_invalid_lengths_graph_preserving_zero():
    """所有样本均无有效帧时，零损失仍连接 LSNR 预测图。"""
    predicted_lsnr = torch.randn(2, 3, requires_grad=True)
    clean = torch.randn(2, 640)
    noise = torch.randn(2, 640)
    lengths = torch.tensor([100, 319])

    with pytest.warns(RuntimeWarning, match="shorter than frame_length"):
        _assert_lsnr_zero_backpropagates(
            predicted_lsnr, clean, noise, lengths
        )


class _TinyLSNRHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


def _gloo_lsnr_zero_worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        model = nn.parallel.DistributedDataParallel(_TinyLSNRHead())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = DeepFilterNetLSNRLoss(frame_length=320, hop_length=160)

        for scenario in ("short", "empty", "all_invalid"):
            optimizer.zero_grad(set_to_none=True)
            predicted_lsnr = model(torch.ones(2, 4))
            clean = torch.ones(2, 640)
            noise = torch.ones(2, 640)
            lengths = None

            if rank == 0 and scenario == "short":
                clean = clean[:, :128]
                noise = noise[:, :128]
            elif rank == 0 and scenario == "empty":
                predicted_lsnr = predicted_lsnr[:, :0]
            elif rank == 0 and scenario == "all_invalid":
                lengths = torch.tensor([100, 319])

            loss = loss_fn(predicted_lsnr, clean, noise, lengths)
            loss.backward()
            assert model.module.projection.weight.grad is not None
            assert torch.isfinite(model.module.projection.weight.grad).all()
            optimizer.step()

        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.ddp
@pytest.mark.subprocess
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo unavailable",
)
def test_lsnr_graph_preserving_zero_handles_rank_asymmetric_ddp(tmp_path):
    """一侧 rank 命中三种零损失分支时，DDP 反传序列仍保持一致。"""
    spawn_ddp_workers(
        _gloo_lsnr_zero_worker,
        args=(str(tmp_path / "lsnr_zero_init"),),
        nprocs=2,
    )


pytestmark = [pytest.mark.unit]
