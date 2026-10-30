"""B5 回归测试：resolve_scheduler_auto_args 的 per_step 单位语义。

LinearWarmupCosineAnnealing 的 warmup_steps / decay_until_step 在
lr_schedule_per_step=true 时以 optimizer step 为单位，false 时以 epoch 为单位
（scheduler 每个 epoch 只 step 一次）。修复前 auto 分支无条件按 per-step 语义
计算 total = epochs * steps_per_epoch：per_step=false 的配置下 auto warmup
变成 5% × 总 step 数，换算成 epoch 后覆盖整个训练，LR 峰值仅 ~8%，静默欠训练。
"""

from __future__ import annotations

import logging

import pytest

from core.optim.scheduler import resolve_scheduler_auto_args


def _sch_cfg(warmup_ratio: float = 0.05) -> dict:
    return {
        "type": "LinearWarmupCosineAnnealing",
        "args": {
            "warmup_steps": "auto",
            "decay_until_step": "auto",
            "max_lr": 1e-3,
            "min_lr": 1e-5,
        },
        "auto": {"warmup_ratio": warmup_ratio},
    }


def test_auto_args_per_step_true_uses_step_units() -> None:
    """per_step=True：total = epochs * steps_per_epoch，warmup = ratio × total。"""
    args = resolve_scheduler_auto_args(
        _sch_cfg(), steps_per_epoch=500, epochs=100, per_step=True,
    )
    assert args["decay_until_step"] == 100 * 500
    assert args["warmup_steps"] == int(0.05 * 100 * 500)
    # 其它超参原样保留
    assert args["max_lr"] == 1e-3
    assert args["min_lr"] == 1e-5


def test_auto_args_per_step_false_uses_epoch_units() -> None:
    """per_step=False：total = epochs，warmup = ratio × epochs（修复的核心）。"""
    args = resolve_scheduler_auto_args(
        _sch_cfg(), steps_per_epoch=500, epochs=100, per_step=False,
    )
    assert args["decay_until_step"] == 100
    assert args["warmup_steps"] == 5


def test_auto_args_per_step_false_warmup_at_least_one() -> None:
    """ratio × epochs 向下取整为 0 时至少 warmup 1 个 epoch。"""
    args = resolve_scheduler_auto_args(
        _sch_cfg(warmup_ratio=0.01), steps_per_epoch=500, epochs=50, per_step=False,
    )
    assert args["warmup_steps"] == 1


def test_auto_args_explicit_values_pass_through() -> None:
    """显式给的 warmup/decay 值原样保留（warning 分支不修改值）。"""
    cfg = _sch_cfg()
    cfg["args"]["warmup_steps"] = 7
    cfg["args"]["decay_until_step"] = 100
    args = resolve_scheduler_auto_args(
        cfg, steps_per_epoch=500, epochs=100, per_step=False,
    )
    assert args["warmup_steps"] == 7
    assert args["decay_until_step"] == 100


def test_auto_args_explicit_decay_warns_in_epoch_units(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """per_step=False 时显式 decay 的 warning 按 epoch 总数判断（不拿 step 总数比）。"""
    cfg = _sch_cfg()
    cfg["args"]["decay_until_step"] = 80  # epoch 单位下小于 total=100 → 应告警
    with caplog.at_level(logging.WARNING, logger="scheduler"):
        resolve_scheduler_auto_args(
            cfg, steps_per_epoch=500, epochs=100, per_step=False,
        )
    assert any("decay_until_step=80" in r.message for r in caplog.records)


def test_auto_args_explicit_decay_matching_epochs_no_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """per_step=False 时显式 decay == epochs 不应误报（修复前会拿 50000 比而误报）。"""
    cfg = _sch_cfg()
    cfg["args"]["decay_until_step"] = 100
    with caplog.at_level(logging.WARNING, logger="scheduler"):
        resolve_scheduler_auto_args(
            cfg, steps_per_epoch=500, epochs=100, per_step=False,
        )
    assert not any("decay_until_step" in r.message for r in caplog.records)


def test_auto_args_other_scheduler_type_unchanged() -> None:
    """非 LinearWarmupCosineAnnealing 的 args 原样返回，per_step 无影响。"""
    cfg = {"type": "StepLR", "args": {"step_size": 10, "gamma": 0.5}}
    for per_step in (True, False):
        args = resolve_scheduler_auto_args(
            cfg, steps_per_epoch=500, epochs=100, per_step=per_step,
        )
        assert args == {"step_size": 10, "gamma": 0.5}

pytestmark = [pytest.mark.unit]
