"""core utils 与 scheduler 残留审计修复的定向回归测试。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from core.optim.scheduler import (
    CosineAnnealingWarmupRestarts,
    LinearWarmupCosineAnnealing,
    resolve_scheduler_auto_args,
)
from core.utils import inference_artifacts
from core.utils.model_profile import profile_model

pytestmark = [pytest.mark.unit]


def _optimizer(lr: float = 1e-3) -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=lr)


def test_atomic_write_audio_fsyncs_staging_before_publish(monkeypatch, tmp_path) -> None:
    target = tmp_path / "output.wav"
    fsynced_staging = False
    real_fsync = inference_artifacts.os.fsync
    real_publish = inference_artifacts._publish_staged_files

    def record_fsync(descriptor: int) -> None:
        nonlocal fsynced_staging
        if inference_artifacts.os.fstat(descriptor).st_nlink == 1:
            fsynced_staging = True
        real_fsync(descriptor)

    def verify_publish(staged, targets, **kwargs) -> None:
        assert fsynced_staging
        real_publish(staged, targets, **kwargs)

    monkeypatch.setattr(inference_artifacts.os, "fsync", record_fsync)
    monkeypatch.setattr(inference_artifacts, "_publish_staged_files", verify_publish)

    inference_artifacts.atomic_write_audio(
        target, lambda staging: staging.write_bytes(b"complete audio")
    )

    assert target.read_bytes() == b"complete audio"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_profile_model_restores_each_tensor_original_device() -> None:
    class MixedDeviceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(8, 8).to("cuda:0")
            self.second = nn.Linear(8, 8).to("cuda:1")
            self.register_buffer("second_buffer", torch.ones(1, device="cuda:1"))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.second(self.first(value)) + self.second_buffer

    model = MixedDeviceModel()
    profile_model(
        model,
        sample_rate=8,
        duration=1.0,
        device="cuda:0",
        warmup=0,
        repeats=1,
        quiet=True,
    )

    assert model.first.weight.device == torch.device("cuda:0")
    assert model.second.weight.device == torch.device("cuda:1")
    assert model.second_buffer.device == torch.device("cuda:1")


@pytest.mark.parametrize(
    ("scheduler", "kwargs"),
    [
        (LinearWarmupCosineAnnealing, {}),
        (CosineAnnealingWarmupRestarts, {}),
    ],
)
def test_schedulers_reject_zero_optimizer_base_lr(scheduler, kwargs) -> None:
    with pytest.raises(ValueError, match="base_lr"):
        scheduler(_optimizer(0.0), **kwargs)


@pytest.mark.parametrize(
    ("scheduler", "kwargs"),
    [
        (LinearWarmupCosineAnnealing, {"warmup_steps": True}),
        (
            CosineAnnealingWarmupRestarts,
            {"first_cycle_steps": 4, "warmup_steps": True},
        ),
    ],
)
def test_schedulers_reject_bool_warmup_steps(scheduler, kwargs) -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        scheduler(_optimizer(), **kwargs)


@pytest.mark.parametrize("argument", ["warmup_steps", "decay_until_step"])
def test_scheduler_auto_args_reject_bool_explicit_steps(argument: str) -> None:
    args = {argument: True}
    with pytest.raises(ValueError, match=argument):
        resolve_scheduler_auto_args(
            {"type": "LinearWarmupCosineAnnealing", "args": args},
            steps_per_epoch=1,
            epochs=1,
        )
