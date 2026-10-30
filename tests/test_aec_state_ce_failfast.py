"""AEC StateCrossEntropyLoss 与 trainer fail-fast 守卫测试。

回归目标：``StateCrossEntropyLoss`` 自身要求 ``state_probs`` 与 ``state`` 均存在，
并严格校验其 batch/time 维一致，避免辅助监督因缺模型 state head、缺 Batch 的
``state`` label 或错位帧标签而静默失效。本测试同时验证
``tasks/aec/trainer/aec_trainer.py`` 的守卫：
- ``_has_active_state_ce`` 只对非零权重 StateCE 分量返回 True；
- train/val loss 路径在非零 StateCE 且模型输出缺 ``state_probs`` 或数据缺
  ``state`` 时 raise ValueError（fail-fast）；
- StateCE weight=0 时不触发，训练正常返回有限 loss；
- ``_composite_ss_undesired_kwargs`` 不再为 weight=0 的 StateCE 注入
  ``state``/``state_probs``。
"""

from __future__ import annotations

import pytest

import torch
import torch.nn as nn

from tasks.aec.data.batch import AECBatch
from tasks.aec.trainer.aec_trainer import (
    AECTrainer,
    _composite_ss_undesired_kwargs,
    _has_active_state_ce,
)
from core.losses.build import _StrictLossWrapper
from core.losses.composite import CompositeLoss
from core.losses.state_ce import StateCrossEntropyLoss


def _state_ce(weight: float) -> CompositeLoss:
    # WS1 权重校验：CompositeLoss 整体必须至少一个非零权重。
    # weight=0 时补一个非零伴生分量，保持"StateCE 分量零权重"的被测语义
    # （_has_active_state_ce / _composite_ss_undesired_kwargs 均按
    # StateCE 分量自身权重过滤，不受伴生分量影响）。
    components = [{"type": "StateCrossEntropyLoss", "weight": weight}]
    if weight == 0.0:
        components.append({"type": "L1WavLoss", "weight": 1.0})
    return CompositeLoss(components=components)


class _BatchStub(AECBatch):
    """``_forward_and_compute_loss`` 需要的最小结构化 AEC batch。"""

    def __init__(self, b: int = 2, t: int = 1600) -> None:
        super().__init__(
            nearend=torch.zeros(b, t),
            mic=torch.zeros(b, t),
            farend=torch.zeros(b, t),
            fs=torch.tensor(16000),
            lengths=torch.full((b,), t, dtype=torch.long),
        )


def _make_stub_trainer(
    loss_fn: nn.Module, forward_extra: dict | None = None
) -> AECTrainer:
    """跳过 ``__init__`` 构造最小 AECTrainer。

    ``forward_extra`` 中的键值追加到桩前向返回的 data（如 ``state_probs`` /
    ``state``），用于模拟模型带 state head 的输出；不传时前向只返回
    ``nearend_estimate``（缺 ``state_probs``）。
    """
    trainer = object.__new__(AECTrainer)
    trainer.loss_fn = loss_fn
    trainer.rank = 0
    # validation 路径的异常同步 wrapper 需要 world_size；单 rank 桩固定为 1。
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer.use_amp = False
    trainer.amp_dtype = torch.float32
    trainer._strict_checks = False
    _data = {"nearend_estimate": torch.zeros(2, 1600)}
    if forward_extra:
        _data.update(forward_extra)
    trainer._forward_aec_batch = lambda _batch: _data
    trainer._accumulate_output_sanitize = lambda *_args, **_kwargs: None
    return trainer


# ---------------------------------------------------------------------------
# StateCrossEntropyLoss 时间维合同
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["state_probs", "state"])
def test_state_ce_rejects_missing_optional_input(missing):
    kwargs = {
        "state_probs": torch.randn(1, 2, 3),
        "state": torch.zeros(1, 2, dtype=torch.long),
    }
    kwargs[missing] = None

    with pytest.raises(TypeError, match=rf"{missing} must be present and be a Tensor"):
        StateCrossEntropyLoss()(
            torch.zeros(1, 8),
            torch.zeros(1, 8),
            **kwargs,
        )


def test_state_ce_accepts_equal_time_dimensions():
    logits = torch.tensor([[[2.0, -1.0], [-0.5, 1.5]]], requires_grad=True)
    state = torch.tensor([[0, 1]])

    loss = StateCrossEntropyLoss()(
        torch.zeros(1, 8),
        torch.zeros(1, 8),
        state_probs=logits,
        state=state,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


@pytest.mark.parametrize("t_probs,t_state", [(3, 4), (4, 3)])
def test_state_ce_rejects_off_by_one_time_dimensions(t_probs, t_state):
    with pytest.raises(ValueError, match="same batch/time dimensions"):
        StateCrossEntropyLoss()(
            torch.zeros(1, 8),
            torch.zeros(1, 8),
            state_probs=torch.randn(1, t_probs, 3),
            state=torch.zeros(1, t_state, dtype=torch.long),
        )


def test_state_ce_rejects_clearly_misaligned_time_dimensions():
    with pytest.raises(
        ValueError,
        match=r"got \(1, 2\) and \(1, 11\)",
    ):
        StateCrossEntropyLoss()(
            torch.zeros(1, 8),
            torch.zeros(1, 8),
            state_probs=torch.randn(1, 2, 3),
            state=torch.zeros(1, 11, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# _has_active_state_ce 纯函数
# ---------------------------------------------------------------------------


def test_has_active_state_ce_true_for_nonzero_state_ce():
    assert _has_active_state_ce(_state_ce(weight=0.001)) is True


def test_has_active_state_ce_true_through_strict_wrapper():
    loss_fn = _StrictLossWrapper(_state_ce(weight=0.001))
    assert _has_active_state_ce(loss_fn) is True


def test_has_active_state_ce_false_for_zero_weight_state_ce():
    assert _has_active_state_ce(_state_ce(weight=0.0)) is False


def test_has_active_state_ce_false_without_state_ce():
    loss_fn = CompositeLoss(components=[{"type": "L1WavLoss", "weight": 1.0}])
    assert _has_active_state_ce(loss_fn) is False


def test_has_active_state_ce_false_for_non_composite_loss():
    assert _has_active_state_ce(nn.MSELoss()) is False
    assert _has_active_state_ce(_StrictLossWrapper(nn.MSELoss())) is False


# ---------------------------------------------------------------------------
# trainer 守卫：非零 StateCE 缺 state_probs → fail-fast
# ---------------------------------------------------------------------------


def test_train_forward_raises_when_state_ce_missing_state_probs():
    loss_fn = _StrictLossWrapper(_state_ce(weight=0.001))
    trainer = _make_stub_trainer(loss_fn)
    with pytest.raises(ValueError, match="state_probs"):
        trainer._forward_and_compute_loss(_BatchStub(), epoch=0, batch_idx=0, fs_val=16000)


def test_val_forward_raises_when_state_ce_missing_state_probs():
    loss_fn = _StrictLossWrapper(_state_ce(weight=0.001))
    trainer = _make_stub_trainer(loss_fn)
    raw = AECBatch(
        nearend=torch.zeros(2, 1600),
        mic=torch.zeros(2, 1600),
        farend=torch.zeros(2, 1600),
        fs=torch.tensor(16000, dtype=torch.long),
        lengths=torch.full((2,), 1600, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="state_probs"):
        trainer._val_forward_loss(raw, epoch=0, n_loops=0)


def test_train_forward_ok_when_state_ce_weight_zero():
    # weight=0 的 StateCE 不触发守卫，训练正常返回有限 loss。
    loss_fn = _StrictLossWrapper(
        CompositeLoss(
            components=[
                {"type": "StateCrossEntropyLoss", "weight": 0.0},
                {"type": "L1WavLoss", "weight": 1.0},
            ]
        )
    )
    trainer = _make_stub_trainer(loss_fn)
    loss, loss_was_finite, info = trainer._forward_and_compute_loss(
        _BatchStub(), epoch=0, batch_idx=0, fs_val=16000
    )
    assert bool(torch.isfinite(loss).item())
    assert loss_was_finite is True
    assert info["loss_was_finite"] is True


def test_train_forward_raises_when_state_ce_missing_state():
    # 模型输出含 state_probs（模型带 state head）但 batch.state 为 None（等价旧
    # tuple 无 state label）：双查守卫必须兜住"数据缺 state"，否则
    # StateCrossEntropyLoss 静默返回零，state 监督失效。
    loss_fn = _StrictLossWrapper(_state_ce(weight=0.001))
    trainer = _make_stub_trainer(loss_fn, {"state_probs": torch.randn(2, 10, 8)})
    with pytest.raises(ValueError, match="数据缺 state"):
        trainer._forward_and_compute_loss(_BatchStub(), epoch=0, batch_idx=0, fs_val=16000)


def test_val_forward_raises_when_state_ce_missing_state_in_batch():
    # Batch 没有 state label：即便模型输出含 state_probs，验证路径
    # 的双查守卫同样 fail-fast。
    loss_fn = _StrictLossWrapper(_state_ce(weight=0.001))
    trainer = _make_stub_trainer(loss_fn, {"state_probs": torch.randn(2, 10, 8)})
    raw = AECBatch(
        torch.zeros(2, 1600),  # nearend
        torch.zeros(2, 1600),  # mic
        torch.zeros(2, 1600),  # farend
        torch.tensor(16000, dtype=torch.long),  # fs
        torch.full((2,), 1600, dtype=torch.long),  # lengths
        None,  # error
        None,  # error_mask
        torch.zeros(2, dtype=torch.bool),  # is_pure_noise
        torch.zeros(2, dtype=torch.bool),  # is_pure_echo
        torch.zeros(2, dtype=torch.bool),  # is_pure_echo_noise
        torch.zeros(2, dtype=torch.bool),  # has_echo
    )
    with pytest.raises(ValueError, match="数据缺 state"):
        trainer._val_forward_loss(raw, epoch=0, n_loops=0)


def test_train_forward_ok_when_state_probs_and_state_present():
    # 正常路径：state_probs 与 state 齐全，守卫不触发，训练返回有限 loss。
    loss_fn = _StrictLossWrapper(_state_ce(weight=0.001))
    batch = _BatchStub()
    batch.state = torch.zeros(2, 10, dtype=torch.long)
    trainer = _make_stub_trainer(
        loss_fn,
        {"state_probs": torch.randn(2, 10, 3), "state": batch.state},
    )
    loss, loss_was_finite, info = trainer._forward_and_compute_loss(
        batch, epoch=0, batch_idx=0, fs_val=16000
    )
    assert bool(torch.isfinite(loss).item())
    assert loss_was_finite is True
    assert info["loss_was_finite"] is True


# ---------------------------------------------------------------------------
# _composite_ss_undesired_kwargs：weight=0 的 StateCE 不再注入 state kwargs
# ---------------------------------------------------------------------------


def test_state_kwargs_suppressed_when_state_ce_weight_zero():
    noisy = torch.randn(2, 16000)
    clean = torch.randn(2, 16000)
    enhanced = torch.randn(2, 16000)
    data = {
        "state": torch.zeros(2, 10, dtype=torch.long),
        "state_probs": torch.randn(2, 10, 8),
    }
    loss_fn = _state_ce(weight=0.0)
    out = _composite_ss_undesired_kwargs(loss_fn, noisy, clean, enhanced, data)
    assert "state" not in out
    assert "state_probs" not in out


pytestmark = [pytest.mark.unit]
