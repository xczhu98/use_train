"""真实 step-best 续训及 optimizer 状态 schema 审计。"""

from __future__ import annotations

import copy
import shutil

import pytest
import torch

from core.trainer.base_trainer import BaseTrainer
from test_middle_checkpoint_resume import _build_trainer, _close_log_handlers


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(
            actual, expected, rtol=1e-6, atol=1e-8, check_dtype=False
        )
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_state_equal(a, e)
    else:
        assert actual == expected


@pytest.mark.integration
@pytest.mark.parametrize(
    ("accum", "saved_step", "cursor"),
    [(1, 1, 1), (2, 1, 2), (1, 5, 5), (2, 3, 5)],
    ids=["single-mid", "accum-mid", "last-full-window", "committed-tail"],
)
def test_step_best_resume_matches_uninterrupted_epochs(tmp_path, accum, saved_step, cursor):
    trainers = []

    def build(path, *, resume_from=None):
        trainer = _build_trainer(
            path, n_train=10, accum=accum,
            resume=resume_from is not None, resume_from=resume_from,
        )
        trainers.append(trainer)
        trainer.epochs = 2
        trainer.validation_steps_interval = 1
        trainer.validation_epoch_interval = 0
        trainer.save_top_k = 100
        trainer.save_curve_interval = 0
        trainer._curve_x_is_step = True
        return trainer

    try:
        control = build(tmp_path / "control")
        control.train()
        candidates = list(control.ckpt_dir.glob(f"best_step={saved_step:06d}-*.tar"))
        assert len(candidates) == 1
        checkpoint_state = BaseTrainer._safe_load(candidates[0])
        assert checkpoint_state["epoch"] == 1
        assert checkpoint_state["epoch_incomplete"] is True
        assert checkpoint_state["batch_cursor"] == cursor
        assert checkpoint_state["scheduler"]["last_epoch"] == 0
        assert checkpoint_state["n_steps"] == saved_step
        assert checkpoint_state["val_losses"]

        checkpoint = tmp_path / "resume" / "checkpoints" / "run" / candidates[0].name
        checkpoint.parent.mkdir(parents=True)
        shutil.copy2(candidates[0], checkpoint)
        resumed = build(tmp_path / "resume", resume_from=checkpoint)
        assert resumed.start_epoch == 1
        assert resumed._resume_skip_batches == cursor
        assert resumed.n_steps == saved_step
        resumed.train()

        assert resumed.processed == (
            [(1, batch) for batch in range(cursor, 5)]
            + [(2, batch) for batch in range(5)]
        )
        _assert_state_equal(resumed.model.state_dict(), control.model.state_dict())
        _assert_state_equal(resumed.optimizer.state_dict(), control.optimizer.state_dict())
        _assert_state_equal(resumed.scheduler.state_dict(), control.scheduler.state_dict())
        assert resumed.n_steps == control.n_steps == 2 * ((5 + accum - 1) // accum)
        assert resumed.scheduler.last_epoch == control.scheduler.last_epoch == 2
        for trainer in (control, resumed):
            final = BaseTrainer._safe_load(trainer.ckpt_dir / "latest_model.tar")
            assert final["epoch"] == 2
            assert "epoch_incomplete" not in final
            assert "batch_cursor" not in final
            assert final["scheduler"]["last_epoch"] == 2
    finally:
        for trainer in trainers:
            if trainer.writer is not None:
                trainer.writer.close()
        _close_log_handlers(tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "name", ["momentum_buffer", "exp_avg", "exp_avg_sq", "max_exp_avg_sq",
             "exp_inf", "sum", "grad_avg", "square_avg", "acc_delta", "ax",
             "prev", "step_size"],
)
@pytest.mark.parametrize("malformed", ["non_tensor", "shape", "dtype"])
def test_known_parameter_state_rejects_malformed_values(name, malformed):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    value = {
        "non_tensor": 1.0,
        "shape": torch.ones(3),
        "dtype": torch.ones(2, 3, dtype=torch.float64),
    }[malformed]
    state = optimizer.state_dict()
    state["state"][0] = {name: value}
    with pytest.raises((TypeError, ValueError), match=name):
        BaseTrainer._validate_optimizer_state_dict(state, optimizer, context="audit")


@pytest.mark.unit
@pytest.mark.parametrize(
    "step", [True, 1j, -1, float("nan"), float("inf"), "1", torch.tensor(True),
             torch.tensor(1j), torch.tensor(-1.0), torch.tensor(float("nan")),
             torch.tensor(float("inf")), torch.ones(2)],
)
def test_step_requires_finite_nonnegative_real_scalar(step):
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(2))])
    state = optimizer.state_dict()
    state["state"][0] = {"step": step}
    with pytest.raises(ValueError, match="finite nonnegative real scalar"):
        BaseTrainer._validate_optimizer_state_dict(state, optimizer, context="audit")


@pytest.mark.unit
@pytest.mark.parametrize("step", [0, 1, 1.5, torch.tensor(2), torch.tensor(3.0)])
def test_step_accepts_real_scalar_formats(step):
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(2))])
    state = optimizer.state_dict()
    state["state"][0] = {"step": step}
    BaseTrainer._validate_optimizer_state_dict(state, optimizer, context="audit")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "kwargs"),
    [("SGD", {"momentum": 0.9}), ("Adam", {"amsgrad": True}),
     ("AdamW", {}), ("Adamax", {}), ("Adagrad", {}), ("Adadelta", {}),
     ("ASGD", {}), ("RMSprop", {"centered": True, "momentum": 0.9}),
     ("Rprop", {}), ("NAdam", {}), ("RAdam", {})],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_torch_optimizer_state_restores_and_continues(name, kwargs, dtype):
    parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype))
    optimizer = getattr(torch.optim, name)([parameter], lr=0.01, **kwargs)
    parameter.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = copy.deepcopy(optimizer.state_dict())
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = getattr(torch.optim, name)([restored_parameter], lr=0.01, **kwargs)
    BaseTrainer._validate_optimizer_state_dict(state, restored, context=name)
    restored.load_state_dict(state)
    for p, opt in ((parameter, optimizer), (restored_parameter, restored)):
        p.square().sum().backward()
        opt.step()
    _assert_state_equal(restored_parameter, parameter)
    _assert_state_equal(restored.state_dict(), optimizer.state_dict())


@pytest.mark.unit
def test_lbfgs_global_history_restores_and_continues():
    parameters = [torch.nn.Parameter(torch.tensor([3.0, -2.0])),
                  torch.nn.Parameter(torch.tensor([5.0]))]

    def step(opt, params):
        def closure():
            opt.zero_grad()
            loss = params[0].pow(4).sum() + params[1].square().sum()
            loss.backward()
            return loss
        opt.step(closure)

    optimizer = torch.optim.LBFGS(parameters, lr=0.1, max_iter=3)
    step(optimizer, parameters)
    state = copy.deepcopy(optimizer.state_dict())
    assert state["state"][0]["d"].shape == (3,)
    assert state["state"][0]["old_dirs"]
    restored_parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    restored = torch.optim.LBFGS(restored_parameters, lr=0.1, max_iter=3)
    BaseTrainer._validate_optimizer_state_dict(state, restored, context="LBFGS")
    restored.load_state_dict(state)
    step(optimizer, parameters)
    step(restored, restored_parameters)
    _assert_state_equal(restored_parameters, parameters)
    _assert_state_equal(restored.state_dict(), optimizer.state_dict())


@pytest.mark.unit
def test_unknown_global_and_nested_state_is_not_assumed_parameter_shaped():
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    state = optimizer.state_dict()
    state["state"][0] = {"global_stat": torch.ones(9, dtype=torch.float64),
                         "history": [torch.ones(8), {"count": torch.tensor(2)}]}
    BaseTrainer._validate_optimizer_state_dict(state, optimizer, context="custom")
    optimizer.load_state_dict(state)
    assert optimizer.state[parameter]["global_stat"].shape == (9,)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["AdamP", "SGDP"])
def test_adamp_optimizer_state_schema(name):
    adamp = pytest.importorskip("adamp")
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = getattr(adamp, name)([parameter], lr=0.01)
    parameter.square().sum().backward()
    optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = getattr(adamp, name)([restored_parameter], lr=0.01)
    BaseTrainer._validate_optimizer_state_dict(state, restored, context=name)
    restored.load_state_dict(state)
    _assert_state_equal(restored.state_dict(), optimizer.state_dict())
