"""STFT 非对齐与模型 profile 状态恢复的审计回归测试。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from core.stft import StftIstftModule
from core.utils.model_profile import profile_model

pytestmark = [pytest.mark.unit]


def test_noncentered_stft_right_padding_restores_non_hop_aligned_waveform() -> None:
    """center=False 输入补齐完整帧后，iSTFT 必须能裁回原始非对齐长度。"""
    module = StftIstftModule(
        n_fft=64,
        hop_length=32,
        win_length=64,
        window="hamming",
        center=False,
    )
    waveform = torch.randn(2, 181, generator=torch.Generator().manual_seed(31))

    spectrum = module(waveform)
    restored = module.inverse(spectrum, waveform.shape[-1])

    assert spectrum.shape[-2] == 5
    torch.testing.assert_close(restored, waveform, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (torch.ones(8), r"shape \(B, T\)"),
        (torch.ones(1, 8, dtype=torch.int64), "floating-point dtype"),
    ],
)
def test_stft_forward_rejects_invalid_waveform_schema(
    value: torch.Tensor, match: str
) -> None:
    module = StftIstftModule(64, 32, 64)

    with pytest.raises((TypeError, ValueError), match=match):
        module(value)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (torch.ones(1, 32, 3, 2), "F=33"),
        (torch.ones(1, 33, 3), "real/imag"),
        (torch.ones(1, 33, 3, 2, dtype=torch.int64), "floating-point or complex"),
    ],
)
def test_stft_inverse_rejects_invalid_spectrum_schema(
    value: torch.Tensor, match: str
) -> None:
    module = StftIstftModule(64, 32, 64)

    with pytest.raises((TypeError, ValueError), match=match):
        module.inverse(value, 64)


def test_stft_finite_validation_rejects_nan_waveform_when_enabled(monkeypatch) -> None:
    """全量有限性检查默认关闭（避免每步同步），显式开启后仍必须 fail-fast。"""
    from core import stft as stft_module

    monkeypatch.setattr(stft_module, "VALIDATE_FINITE_INPUTS", True)
    module = StftIstftModule(64, 32, 64)

    with pytest.raises(ValueError, match="finite values"):
        module(torch.tensor([[float("nan")]]))


def test_stft_finite_validation_rejects_inf_spectrum_when_enabled(monkeypatch) -> None:
    from core import stft as stft_module

    monkeypatch.setattr(stft_module, "VALIDATE_FINITE_INPUTS", True)
    module = StftIstftModule(64, 32, 64)

    with pytest.raises(ValueError, match="finite values"):
        module.inverse(torch.full((1, 33, 3, 2), float("inf")), 64)


def test_profile_model_restores_nested_training_states() -> None:
    model = nn.Sequential(nn.Linear(8, 4), nn.Dropout())
    model.train()
    model[0].eval()
    original_states = [module.training for module in model.modules()]

    profile_model(
        model,
        sample_rate=8,
        duration=1.0,
        warmup=0,
        repeats=1,
        quiet=True,
    )

    assert [module.training for module in model.modules()] == original_states
    assert next(model.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device migration")
def test_profile_model_restores_original_device_after_cuda_profile() -> None:
    """profile 临时迁移到 CUDA 后必须将调用方模型迁回原 CPU 设备。"""
    model = nn.Linear(8, 4).train()

    profile_model(
        model,
        sample_rate=8,
        duration=1.0,
        device="cuda",
        warmup=0,
        repeats=1,
        quiet=True,
    )

    assert model.training is True
    assert next(model.parameters()).device.type == "cpu"


def test_profile_model_restores_state_after_profile_forward_failure() -> None:
    class RaisingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("profile probe failure")

    model = RaisingModel().train()

    result = profile_model(
        model,
        sample_rate=8,
        duration=1.0,
        warmup=0,
        repeats=1,
        quiet=True,
    )

    assert model.training is True
    assert next(model.parameters()).device.type == "cpu"
    assert result["rtf_error"] == "RuntimeError: profile probe failure"
