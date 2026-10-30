"""WeSep ExponentialDecrease 调度器合同测试。"""

from __future__ import annotations

import math

import pytest
import torch

from core.optim.scheduler import ExponentialDecrease, resolve_scheduler_auto_args


pytestmark = pytest.mark.unit


def _optimizer(lr: float = 1e-3) -> torch.optim.Optimizer:
    return torch.optim.Adam([torch.nn.Parameter(torch.ones(()))], lr=lr)


def test_auto_horizon_uses_full_optimizer_step_plan():
    args = resolve_scheduler_auto_args(
        {
            "type": "ExponentialDecrease",
            "args": {
                "initial_lr": 1e-3,
                "final_lr": 2.5e-5,
                "total_steps": "auto",
            },
        },
        steps_per_epoch=463,
        epochs=150,
        per_step=True,
    )

    assert args["total_steps"] == 150 * 463


def test_schedule_matches_wesep_formula_and_endpoint():
    optimizer = _optimizer()
    scheduler = ExponentialDecrease(
        optimizer,
        initial_lr=1e-3,
        final_lr=2.5e-5,
        total_steps=4,
    )
    expected = [1e-3 * math.exp((i / 4) * math.log(2.5e-5 / 1e-3)) for i in range(5)]

    actual = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        actual.append(optimizer.param_groups[0]["lr"])

    assert actual == pytest.approx(expected, rel=1e-14, abs=0.0)
    assert actual[-1] == pytest.approx(2.5e-5, rel=1e-14)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"initial_lr": 0.0, "final_lr": 1e-5, "total_steps": 4}, "initial_lr"),
        ({"initial_lr": 1e-4, "final_lr": 1e-3, "total_steps": 4}, "final_lr"),
        ({"initial_lr": 1e-3, "final_lr": 1e-5, "total_steps": True}, "total_steps"),
    ],
)
def test_invalid_domains_fail_fast(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ExponentialDecrease(_optimizer(), **kwargs)


def test_optimizer_lr_must_equal_initial_lr():
    with pytest.raises(ValueError, match="optimizer lr"):
        ExponentialDecrease(
            _optimizer(2e-3),
            initial_lr=1e-3,
            final_lr=2.5e-5,
            total_steps=10,
        )


def test_rebase_updates_curve_parameters_without_resetting_progress():
    optimizer = _optimizer()
    scheduler = ExponentialDecrease(
        optimizer,
        initial_lr=1e-3,
        final_lr=1e-5,
        total_steps=4,
    )
    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == 1

    optimizer.param_groups[0]["lr"] = 2e-3
    scheduler.rebase()

    assert scheduler.last_epoch == 1
    assert scheduler.initial_lr == pytest.approx(2e-3)
    assert scheduler.final_lr == pytest.approx(2e-5)
    assert scheduler._log_ratio == pytest.approx(math.log(1e-2))
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-3 * 1e-2 ** 0.5)
