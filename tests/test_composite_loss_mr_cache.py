"""CompositeLoss 的 MR-STFT 精确 schedule 共享缓存测试。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from core.losses.composite import CompositeLoss, _mr_cache_signature
from core.utils.registry import LOSS_REGISTRY


_SCALE_FACTORS = [0.125, 0.25, 0.5, 1, 2, 4]
_DBGNET_FRAME_SCHEDULES = [
    pytest.param(
        {"stft_scale_factors": _SCALE_FACTORS, "win_length": 320, "hop_length": 160},
        ((40, 80, 160, 320, 640, 1280), (20, 40, 80, 160, 320, 640)),
        id="dbgnet-16k",
    ),
    pytest.param(
        {"stft_scale_factors": _SCALE_FACTORS, "win_length": 640, "hop_length": 320},
        ((80, 160, 320, 640, 1280, 2560), (40, 80, 160, 320, 640, 1280)),
        id="dbgnet-32k",
    ),
]


def _components(
    si_args: dict,
    suppress_args: dict | None = None,
) -> list[dict]:
    return [
        {
            "type": "MRSIRelCspecL2Loss",
            "weight": 1.0,
            "args": {**si_args, "use_frequency_weight": False},
        },
        {
            "type": "MRSpecSuppressPenaltyLoss",
            "weight": 0.5,
            "args": suppress_args if suppress_args is not None else si_args,
        },
    ]


def _value_and_grads(
    loss_fn: nn.Module,
    enhanced: torch.Tensor,
    clean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    enhanced_leaf = enhanced.detach().clone().requires_grad_(True)
    clean_leaf = clean.detach().clone().requires_grad_(True)
    value = loss_fn(enhanced_leaf, clean_leaf)
    enhanced_grad, clean_grad = torch.autograd.grad(
        value, (enhanced_leaf, clean_leaf)
    )
    return value.detach(), enhanced_grad.detach(), clean_grad.detach()


def _assert_value_and_grads_close(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-5)


class _ExactScheduleConsumer(nn.Module):
    def __init__(self, fft_sizes: tuple[int, ...], hop_sizes: tuple[int, ...]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes

    def forward(self, enhanced, clean, lengths=None, stft_cache=None):
        return torch.mean((enhanced - clean) ** 2)


def test_same_exact_non_power_of_two_schedule_is_paired():
    first = _ExactScheduleConsumer((75, 150, 300), (30, 60, 120))
    second = _ExactScheduleConsumer((75, 150, 300), (30, 60, 120))

    assert CompositeLoss._find_mr_cache_pair(nn.ModuleList([first, second])) == (0, 1)
    assert _mr_cache_signature(first) == ((75, 150, 300), (30, 60, 120))


@pytest.mark.parametrize(
    "second_sizes,second_hops",
    [
        ((76, 150, 300), (30, 60, 120)),
        ((75, 150, 300), (31, 60, 120)),
        ((300, 150, 75), (120, 60, 30)),
    ],
    ids=["different-size", "different-hop", "different-order"],
)
def test_different_exact_schedule_is_not_paired(second_sizes, second_hops):
    first = _ExactScheduleConsumer((75, 150, 300), (30, 60, 120))
    second = _ExactScheduleConsumer(second_sizes, second_hops)

    assert CompositeLoss._find_mr_cache_pair(nn.ModuleList([first, second])) == (-1, -1)


@pytest.mark.parametrize("mr_args, expected", _DBGNET_FRAME_SCHEDULES)
def test_real_frame_relative_losses_share_exact_schedule_cache(mr_args, expected):
    loss_fn = CompositeLoss(components=_components(mr_args))

    assert (loss_fn._mr_cache_si, loss_fn._mr_cache_sj) == (0, 1)
    assert _mr_cache_signature(loss_fn.losses[0]) == expected
    assert _mr_cache_signature(loss_fn.losses[1]) == expected


def test_legacy_and_same_geometry_canonical_schedule_match_value_and_gradients():
    torch.manual_seed(0)
    enhanced = torch.randn(1, 4096)
    clean = torch.randn(1, 4096)
    legacy_args = {"fft_power_min": 6, "fft_power_max": 11}
    canonical_args = {
        "stft_scale_factors": _SCALE_FACTORS,
        "win_length": 512,
        "hop_length": 256,
    }

    legacy = CompositeLoss(components=_components(legacy_args))
    canonical = CompositeLoss(components=_components(canonical_args))

    assert _mr_cache_signature(legacy.losses[0]) == _mr_cache_signature(
        canonical.losses[0]
    )
    _assert_value_and_grads_close(
        _value_and_grads(canonical, enhanced, clean),
        _value_and_grads(legacy, enhanced, clean),
    )


@pytest.mark.parametrize("mr_args, _", _DBGNET_FRAME_SCHEDULES)
def test_shared_exact_schedule_cache_matches_uncached_value_and_gradients(mr_args, _):
    torch.manual_seed(1)
    enhanced = torch.randn(1, 2048)
    clean = torch.randn(1, 2048)
    cached = CompositeLoss(components=_components(mr_args))
    uncached = CompositeLoss(components=_components(mr_args))
    uncached._mr_cache_si, uncached._mr_cache_sj = -1, -1

    assert (cached._mr_cache_si, cached._mr_cache_sj) == (0, 1)
    _assert_value_and_grads_close(
        _value_and_grads(cached, enhanced, clean),
        _value_and_grads(uncached, enhanced, clean),
    )


def test_legacy_duck_typed_power_consumers_still_share_cache():
    """旧消费方只暴露 fft_power 范围时仍按等价几何参与共享。"""

    @LOSS_REGISTRY.register()
    class _DummyMRLossA(nn.Module):
        def __init__(self, fft_power_min: int = 6, fft_power_max: int = 7):
            super().__init__()
            self.fft_power_min = fft_power_min
            self.fft_power_max = fft_power_max

        def forward(self, enhanced, clean, lengths=None, stft_cache=None):
            # 收到缓存时，每个 legacy power 必须恰好对应一个分辨率。
            if stft_cache is not None:
                assert len(stft_cache) == self.fft_power_max - self.fft_power_min + 1
            return torch.mean((enhanced - clean) ** 2)

    @LOSS_REGISTRY.register()
    class _DummyMRLossB(nn.Module):
        def __init__(self, fft_power_min: int = 6, fft_power_max: int = 7):
            super().__init__()
            self.fft_power_min = fft_power_min
            self.fft_power_max = fft_power_max

        def forward(self, enhanced, clean, lengths=None, stft_cache=None):
            if stft_cache is not None:
                assert len(stft_cache) == self.fft_power_max - self.fft_power_min + 1
            return torch.mean(torch.abs(enhanced - clean))

    loss_fn = CompositeLoss(
        components=[
            {"type": "_DummyMRLossA", "weight": 1.0},
            {"type": "_DummyMRLossB", "weight": 1.0},
        ]
    )
    assert {loss_fn._mr_cache_si, loss_fn._mr_cache_sj} == {0, 1}

    assert _mr_cache_signature(loss_fn.losses[0]) == (
        (64, 128),
        (32, 64),
    )
    enhanced = torch.randn(2, 1024)
    clean = torch.randn(2, 1024)
    total = loss_fn(enhanced, clean)
    assert torch.isfinite(total)


def test_signature_none_when_no_stft_cache_param():
    class _PlainLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.fft_power_min = 6
            self.fft_power_max = 7

        def forward(self, enhanced, clean, lengths=None):
            return torch.mean((enhanced - clean) ** 2)

    assert _mr_cache_signature(_PlainLoss()) is None


@pytest.mark.parametrize("mr_args, expected", _DBGNET_FRAME_SCHEDULES)
def test_frame_relative_shared_cache_skipped_when_lengths_given(monkeypatch, mr_args, expected):
    """lengths 非空时 frame-relative 消费方必须各自做 trim 一致 STFT。"""
    import core.losses.composite as composite_mod

    calls = []
    original = composite_mod.compute_mr_stft_shared_cache

    def spy(enhanced, clean, fft_sizes, hop_sizes):
        calls.append((tuple(fft_sizes), tuple(hop_sizes)))
        return original(enhanced, clean, fft_sizes, hop_sizes)

    monkeypatch.setattr(composite_mod, "compute_mr_stft_shared_cache", spy)
    torch.manual_seed(2)
    loss_fn = CompositeLoss(components=_components(mr_args))
    enhanced = torch.randn(2, 2048)
    clean = torch.randn(2, 2048)

    loss_fn(enhanced, clean, lengths=torch.tensor([1024, 2048]))
    assert calls == []

    loss_fn(enhanced, clean)
    assert calls == [expected]

pytestmark = [pytest.mark.unit]
