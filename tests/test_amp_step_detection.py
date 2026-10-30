"""A20 回归测试：AMP FP16 真实 step 判定使用 GradScaler 公开 API。

原实现依赖私有属性 ``optimizer._step_count`` 判断 GradScaler 是否真的执行了
optimizer.step()；torch 版本不保证该属性存在，误判会让 scheduler / EMA /
全局 step 计数漂移。修复后改为 HuggingFace Trainer 同款判定：
``scaler.update()`` 后 ``get_scale()`` 下降 ⇔ 因 inf/NaN 梯度跳过了 step。

本测试在 CPU 上用真实的 ``torch.amp.GradScaler("cpu")`` 注入 inf 梯度，
断言跳过时 n_steps / scheduler / EMA 均不推进，正常 step 时全部推进；
同时断言非 AMP（FP32）路径行为不变（sanitize 后恒为真实 step），以及
没有 backward 梯度时不执行空 optimizer step。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
import torch.nn as nn


from core.trainer.base_trainer import BaseTrainer
from core.trainer.optimizer_step import PreparedOptimizerStep, amp_input_is_finite


def _close_log_handlers(exp_dir: Path) -> None:
    """关闭指向 exp_dir 下 train.log 的 FileHandler，避免临时目录清理失败。"""
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                if str(exp_dir.resolve()) in str(Path(h.baseFilename).resolve()):
                    h.close()
                    root.removeHandler(h)
            except Exception:
                pass


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _StubEMA:
    """记录 update 调用次数的 EMA 替身（不依赖 torch_ema）。"""

    def __init__(self) -> None:
        self.n_updates = 0

    def update(self) -> None:
        self.n_updates += 1


def _build_trainer(tmp_path: Path, use_amp: bool) -> BaseTrainer:
    config = {
        "model": {"type": "TinyModel", "args": {}},
        "trainer": {
            "epochs": 1,
            "exp_path": str(tmp_path),
            "log_interval": 1,
            "lr_schedule_per_step": True,
        },
        "meta": {"seed": 42, "use_amp": use_amp, "amp_dtype": "float16"},
        "data": {},
    }
    torch.manual_seed(42)
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = BaseTrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=nn.MSELoss(),
        config=config,
    )
    trainer.ema = _StubEMA()
    return trainer


def test_grad_scaler_uses_trainer_device_type(tmp_path: Path, monkeypatch) -> None:
    import core.trainer.base_trainer as base_trainer_module

    devices = []
    real_grad_scaler = torch.amp.GradScaler

    def capture_grad_scaler(device, *, enabled):
        devices.append(device)
        return real_grad_scaler(device, enabled=enabled)

    monkeypatch.setattr(base_trainer_module, "GradScaler", capture_grad_scaler)
    trainer = _build_trainer(tmp_path, use_amp=True)
    try:
        assert devices == [trainer.device.type]
        assert trainer.scaler.is_enabled()
    finally:
        _close_log_handlers(tmp_path)


def _backward_once(trainer: BaseTrainer, inf_grad: bool = False) -> None:
    """模拟训练循环的 backward（先 scale 再 backward，与生产路径一致）。"""
    x = torch.randn(4, 8, device=trainer.device)
    loss = trainer.model(x).pow(2).sum()
    trainer.scaler.scale(loss).backward()
    if inf_grad:
        for p in trainer.model.parameters():
            p.grad = torch.full_like(p, float("inf"))


def test_amp_skipped_step_does_not_advance_state(tmp_path: Path) -> None:
    """inf 梯度 → scaler 跳过 step：n_steps / scheduler / EMA 均不推进。"""
    trainer = _build_trainer(tmp_path, use_amp=True)
    try:
        # 正常 step：全部推进
        _backward_once(trainer)
        scale_before = trainer.scaler.get_scale()
        trainer._optimizer_step()
        assert trainer.n_steps == 1
        assert trainer.scheduler.last_epoch == 1
        assert trainer.ema.n_updates == 1
        assert trainer.scaler.get_scale() == scale_before  # 无 backoff

        # inf 梯度 step：scaler 跳过，三者均不动，scale 下降
        _backward_once(trainer, inf_grad=True)
        trainer._optimizer_step()
        assert trainer.scaler.get_scale() < scale_before
        assert trainer.n_steps == 1, "skipped step must not advance n_steps"
        assert trainer.scheduler.last_epoch == 1, "skipped step must not advance scheduler"
        assert trainer.ema.n_updates == 1, "skipped step must not advance EMA"

        # 再次正常 step：继续推进（计数与真实 step 对齐）
        _backward_once(trainer)
        trainer._optimizer_step()
        assert trainer.n_steps == 2
        assert trainer.scheduler.last_epoch == 2
        assert trainer.ema.n_updates == 2
    finally:
        _close_log_handlers(tmp_path)


def test_amp_overflow_discard_resets_growth_tracker(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, use_amp=True)
    try:
        scaler_state = trainer.scaler.state_dict()
        scaler_state["_growth_tracker"] = 17
        trainer.scaler.load_state_dict(scaler_state)
        _backward_once(trainer, inf_grad=True)

        trainer._optimizer_step()

        assert trainer.scaler.get_scale() < 65536.0
        assert trainer.scaler.state_dict()["_growth_tracker"] == 0
    finally:
        _close_log_handlers(tmp_path)


def test_non_amp_path_always_steps(tmp_path: Path) -> None:
    """非 AMP（FP32）路径行为不变：sanitize 后恒为真实 step，全部推进。"""
    trainer = _build_trainer(tmp_path, use_amp=False)
    try:
        _backward_once(trainer)
        trainer._optimizer_step()
        assert trainer.n_steps == 1
        assert trainer.scheduler.last_epoch == 1
        assert trainer.ema.n_updates == 1

        # inf 梯度在非 AMP 路径会被 _sanitize_gradients 清零，step 照常执行
        _backward_once(trainer, inf_grad=True)
        trainer._optimizer_step()
        assert trainer.n_steps == 2
        assert trainer.scheduler.last_epoch == 2
        assert trainer.ema.n_updates == 2
    finally:
        _close_log_handlers(tmp_path)


@pytest.mark.parametrize("use_amp", [False, True])
def test_empty_gradient_window_skips_all_optimizer_side_effects(
    tmp_path: Path, use_amp: bool
) -> None:
    """没有 backward 梯度时不得执行空 optimizer step 或推进共享状态。"""
    trainer = _build_trainer(tmp_path / ("amp" if use_amp else "fp32"), use_amp=use_amp)
    try:
        before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
        assert trainer._optimizer_step() is False
        assert trainer.n_steps == 0
        assert trainer.scheduler.last_epoch == 0
        assert trainer.ema.n_updates == 0
        for parameter, expected in zip(trainer.model.parameters(), before):
            torch.testing.assert_close(parameter, expected)
    finally:
        _close_log_handlers(tmp_path)


def test_post_clip_nonfinite_gradients_skip_before_optimizer_step(
    tmp_path: Path, monkeypatch
) -> None:
    trainer = _build_trainer(tmp_path, use_amp=False)
    try:
        _backward_once(trainer)
        before = [parameter.detach().clone() for parameter in trainer.model.parameters()]

        def poison_gradients(parameters, _max_norm):
            parameters = list(parameters)
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.fill_(float("inf"))
            return torch.tensor(1.0, device=trainer.device)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", poison_gradients)
        assert trainer._optimizer_step() is False
        assert trainer.n_steps == 0
        assert trainer.scheduler.last_epoch == 0
        assert trainer.ema.n_updates == 0
        for parameter, expected in zip(trainer.model.parameters(), before):
            torch.testing.assert_close(parameter, expected)
    finally:
        _close_log_handlers(tmp_path)


def test_amp_gate_uses_explicit_flag_not_reason_text() -> None:
    """AMP 溢出判定绑定显式标志：改写 reason 文案不得改变 GradScaler backoff 结果。"""
    flagged = PreparedOptimizerStep(
        key="default",
        optimizer=None,
        amp_fp16=True,
        eligible=False,
        reason="overflow",
        amp_nonfinite=True,
    )
    assert amp_input_is_finite(flagged) is False

    reworded = PreparedOptimizerStep(
        key="default",
        optimizer=None,
        amp_fp16=True,
        eligible=False,
        reason="unscaled gradients are non-finite",
    )
    assert reworded.amp_nonfinite is False
    assert amp_input_is_finite(reworded) is True


pytestmark = [
    pytest.mark.unit,
    pytest.mark.usefixtures("preserve_tensorboard_module_state"),
]
