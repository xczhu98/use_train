"""WS1 审计修复：共享 loss 图保留零 / CompositeLoss 权重校验回归测试。

覆盖:
  a) FWNegSISDRLoss 全静音 clean 分支返回图保留零 (enhanced*0.0).sum():
     backward 成功且 enhanced.grad 为 0 张量（非 None）。
  b) CompositeLoss 整批 pure（全部跳过依赖语音参考的分量）时 total 不连
     模型输出，forward 末尾注入 enhanced.sum()*0.0 兜底：返回可求导、
     backward 成功且 enhanced.grad 为 0。
  c) CompositeLoss 全零权重 / NaN / Inf 权重构造抛 ValueError。
"""

from __future__ import annotations

import pytest
import torch

from core.losses.composite import CompositeLoss
from core.losses.fw_sdr import FWNegSISDRLoss
from core.losses.time_domain import L1WavLoss  # noqa: F401 — 触发 L1WavLoss 注册
from core.losses.time_domain import SISNRLoss  # noqa: F401 — 触发 SISNRLoss 注册


def test_fw_sdr_all_silent_backward_zero_grad():
    """全静音 batch：返回图保留零，backward 成功且 grad 为 0 非 None。"""
    loss_fn = FWNegSISDRLoss()
    enhanced = torch.randn(2, 1600, requires_grad=True)
    clean = torch.zeros(2, 1600)

    loss = loss_fn(enhanced, clean, undesired_ref=torch.zeros_like(clean))
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert loss.requires_grad

    loss.backward()
    assert enhanced.grad is not None
    assert torch.all(enhanced.grad == 0.0)


def test_fw_sdr_all_silent_with_lengths_backward_ok():
    """全静音 + lengths 传入时同样走图保留零分支，backward 不崩。"""
    loss_fn = FWNegSISDRLoss()
    enhanced = torch.randn(2, 1600, requires_grad=True)
    clean = torch.zeros(2, 1600)
    lengths = torch.full((2,), 800, dtype=torch.long)

    loss = loss_fn(
        enhanced,
        clean,
        lengths,
        undesired_ref=torch.zeros_like(clean),
    )
    assert loss.item() == 0.0
    loss.backward()
    assert enhanced.grad is not None
    assert torch.all(enhanced.grad == 0.0)


def test_composite_all_pure_batch_backward_ok():
    """整批 pure：全部依赖语音参考的分量被跳过，图保留兜底生效、可求导。"""
    enhanced = torch.randn(2, 1600, requires_grad=True)
    clean = torch.zeros(2, 1600)
    lengths = torch.full((2,), 1600, dtype=torch.long)
    is_pure_noise = torch.ones(2, dtype=torch.bool)

    loss_fn = CompositeLoss(components=[{"type": "SISNRLoss", "weight": 1.0}])
    total = loss_fn(enhanced, clean, lengths, is_pure_noise=is_pure_noise)

    assert torch.isfinite(total)
    assert total.item() == 0.0
    assert total.requires_grad

    total.backward()
    assert enhanced.grad is not None
    assert torch.all(enhanced.grad == 0.0)


def test_composite_all_zero_weights_raises():
    """全零权重配置无监督信号，构造必须抛 ValueError。"""
    with pytest.raises(ValueError, match="非零权重"):
        CompositeLoss(
            components=[
                {"type": "L1WavLoss", "weight": 0.0},
                {"type": "L1WavLoss", "weight": 0.0},
            ]
        )


def test_composite_nan_weight_raises():
    """NaN 权重会静默污染总 loss，构造必须抛 ValueError。"""
    with pytest.raises(ValueError, match="finite"):
        CompositeLoss(
            components=[
                {"type": "L1WavLoss", "weight": 1.0},
                {"type": "L1WavLoss", "weight": float("nan")},
            ]
        )


def test_composite_inf_weight_raises():
    """Inf 权重同样非有限值，构造必须抛 ValueError。"""
    with pytest.raises(ValueError, match="finite"):
        CompositeLoss(
            components=[{"type": "L1WavLoss", "weight": float("inf")}]
        )


pytestmark = [pytest.mark.unit]