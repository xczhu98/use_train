"""Unit tests for core/losses/composite.py.

Covers the most common CompositeLoss behaviours that are easy to get wrong when
kwargs are sprayed from the trainer: weighted summation, pure-branch gating of
speech-reference-dependent components, and SS (undesired-energy) mixing.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from core.losses.build import build_loss
from core.losses.composite import StrictFiniteLossError
from core.losses.hybrid import GTCRNCompressedSpectralLoss, GTCRNLogRatioLoss
from core.utils.registry import LOSS_REGISTRY
from core.losses.composite import CompositeLoss
from core.losses.se_train_mr import MRSIRelCspecL2Loss
from core.losses.time_domain import L1WavLoss, SISNRLoss

_TEST_LOSS_NAMES = (
    "_FractionalMassSpeechLoss",
    "_ZeroMassSpeechLoss",
    "_ZeroWeightSpeechLoss",
    "_DirectSpectrumSpeechLoss",
)


@pytest.fixture(autouse=True, scope="module")
def _unregister_test_losses():
    """模块结束后注销本文件注册的临时 loss，避免污染全局 LOSS_REGISTRY。

    ``Registry`` 目前没有公开的注销 API，测试直接操作内部映射。
    """
    yield
    for name in _TEST_LOSS_NAMES:
        LOSS_REGISTRY._registry.pop(name, None)


@LOSS_REGISTRY.register("_FractionalMassSpeechLoss")
class _FractionalMassSpeechLoss(nn.Module):
    """使用小于 1 的合法 reduction mass，回归分母不得 clamp。"""

    REQUIRES_SPEECH_REFERENCE = True

    def speech_reference_reduction_mass(self, clean, lengths):
        return torch.full(
            (clean.shape[0],),
            0.25,
            device=clean.device,
            dtype=torch.float32,
        )

    def forward(self, enhanced, clean, lengths=None):
        return (enhanced - clean).abs().mean()


@LOSS_REGISTRY.register("_ZeroMassSpeechLoss")
class _ZeroMassSpeechLoss(nn.Module):
    """零质量组件允许查询质量，但不得执行 forward。"""

    REQUIRES_SPEECH_REFERENCE = True

    def speech_reference_reduction_mass(self, clean, lengths):
        return torch.zeros(clean.shape[0], device=clean.device)

    def forward(self, enhanced, clean, lengths=None):
        raise AssertionError("zero-mass forward must not run")


@LOSS_REGISTRY.register("_ZeroWeightSpeechLoss")
class _ZeroWeightSpeechLoss(nn.Module):
    """零权重组件的 callback/forward 均不得被调用。"""

    REQUIRES_SPEECH_REFERENCE = True

    def __init__(self):
        super().__init__()
        self.valid_calls = 0
        self.mass_calls = 0
        self.forward_calls = 0

    def speech_reference_valid_mask(self, clean, lengths):
        self.valid_calls += 1
        raise AssertionError("zero-weight valid callback must not run")

    def speech_reference_reduction_mass(self, clean, lengths):
        self.mass_calls += 1
        raise AssertionError("zero-weight mass callback must not run")

    def forward(self, enhanced, clean, lengths=None):
        self.forward_calls += 1
        raise AssertionError("zero-weight forward must not run")


@LOSS_REGISTRY.register("_DirectSpectrumSpeechLoss")
class _DirectSpectrumSpeechLoss(nn.Module):
    """显式消费 direct spectra 的 speech-reference 路由测试 loss。"""

    REQUIRES_SPEECH_REFERENCE = True
    SUPPORTS_DIRECT_SPECTRA = True

    def speech_reference_valid_mask(self, clean, lengths):
        return clean.square().sum(dim=-1) > 0

    def forward(
        self,
        enhanced,
        clean,
        lengths=None,
        *,
        enhanced_spectra,
        target_spectra,
    ):
        assert enhanced_spectra.shape[0] == enhanced.shape[0]
        assert target_spectra.shape[0] == clean.shape[0]
        return enhanced_spectra.float().mean() + target_spectra.float().mean()


def _deterministic_mrsi_inputs():
    t = torch.linspace(-1.0, 1.0, 64)
    phase = torch.arange(64, dtype=torch.float32)
    clean = torch.stack(
        [t, torch.sin(phase * 0.2) + 0.2, torch.cos(phase * 0.13) - 0.1]
    )
    undesired_ref = torch.stack(
        [0.3 * torch.cos(phase * 0.17), 0.6 * t.flip(0), 0.4 * torch.sin(phase * 0.11)]
    )
    enhanced = 0.75 * clean + 0.08 * torch.cos(phase * 0.07).unsqueeze(0)
    undesired_enhanced = 0.35 * undesired_ref + 0.05 * torch.sin(
        phase * 0.19
    ).unsqueeze(0)
    return enhanced, clean, undesired_enhanced, undesired_ref


def _pure_metadata_variants():
    false = torch.zeros(3, dtype=torch.bool)
    return (
        {},
        {
            "is_pure_noise": false,
            "is_pure_echo": false,
            "is_pure_echo_noise": false,
        },
        {
            "is_pure_noise": torch.tensor([True, False, False]),
            "is_pure_echo": torch.tensor([False, True, False]),
            "is_pure_echo_noise": false,
        },
        {
            "is_pure_noise": torch.tensor([True, False, False]),
            "is_pure_echo": torch.tensor([False, True, False]),
            "is_pure_echo_noise": torch.tensor([False, False, True]),
        },
    )


def test_weighted_sum_of_two_components():
    enhanced = torch.randn(2, 16000)
    clean = torch.randn(2, 16000)

    loss_fn = CompositeLoss(
        components=[
            {"type": "L1WavLoss", "weight": 1.0},
            {"type": "L1WavLoss", "weight": 0.5},
        ]
    )
    total = loss_fn(enhanced, clean)

    l1 = torch.mean(torch.abs(enhanced - clean))
    expected = l1 + 0.5 * l1
    assert torch.allclose(total, expected, atol=1e-5)

    bd = loss_fn._last_breakdown
    assert len(bd) == 2
    assert bd[0][2] == pytest.approx(l1.item(), abs=1e-5)
    assert bd[1][2] == pytest.approx(l1.item(), abs=1e-5)


def test_fractional_global_mass_is_not_clamped_to_one(monkeypatch):
    enhanced = torch.tensor([[2.0, -1.0]], requires_grad=True)
    clean = torch.zeros_like(enhanced)
    loss_fn = CompositeLoss(
        components=[{"type": "_FractionalMassSpeechLoss", "weight": 1.0}]
    )

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(value, op=None):
        value.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    actual = loss_fn(enhanced, clean)
    expected = (enhanced - clean).abs().mean()
    torch.testing.assert_close(actual, expected)


def test_global_zero_mass_returns_graph_connected_zero_without_forward():
    enhanced = torch.randn(2, 16, requires_grad=True)
    clean = torch.randn_like(enhanced)
    loss_fn = CompositeLoss(
        components=[{"type": "_ZeroMassSpeechLoss", "weight": 1.0}]
    )

    actual = loss_fn(enhanced, clean)

    assert actual.item() == 0.0
    actual.backward()
    assert enhanced.grad is not None
    assert torch.count_nonzero(enhanced.grad).item() == 0


def test_zero_weight_speech_component_only_runs_fixed_zero_mass_collective(monkeypatch):
    enhanced = torch.randn(2, 16, requires_grad=True)
    clean = torch.randn_like(enhanced)
    loss_fn = CompositeLoss(
        components=[
            {"type": "_ZeroWeightSpeechLoss", "weight": 0.0},
            {"type": "L1WavLoss", "weight": 1.0},
        ]
    )
    reduced = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(value, op=None):
        reduced.append(value.detach().clone())

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    actual = loss_fn(enhanced, clean)

    component = loss_fn.losses[0]
    assert component.valid_calls == 0
    assert component.mass_calls == 0
    assert component.forward_calls == 0
    assert len(reduced) == 2
    assert reduced[0].item() == 0.0
    torch.testing.assert_close(actual, (enhanced - clean).abs().mean())


@pytest.mark.parametrize("name", ["clean", "enhanced", "noisy"])
def test_strict_nonfinite_input_cannot_be_hidden_by_speech_valid_mask(name):
    clean = torch.ones(2, 16)
    enhanced = torch.zeros_like(clean)
    noisy = torch.zeros_like(clean)
    values = {"clean": clean, "enhanced": enhanced, "noisy": noisy}
    values[name][0, 0] = float("nan")
    loss_fn = CompositeLoss(
        components=[{"type": "SISNRLoss", "weight": 1.0}],
        strict=True,
    )

    with pytest.raises(
        StrictFiniteLossError,
        match=rf"'{name}'.*NaN/Inf",
    ):
        loss_fn(
            values["enhanced"],
            values["clean"],
            noisy=values["noisy"],
            is_pure_noise=torch.tensor([True, False]),
        )


def test_speech_reference_direct_spectra_survive_capability_aware_subsetting():
    enhanced = torch.randn(3, 8)
    clean = torch.stack((torch.ones(8), torch.zeros(8), torch.full((8,), 2.0)))
    enhanced_spectra = torch.stack(
        (torch.full((2, 3, 2), 10.0), torch.full((2, 3, 2), 20.0), torch.full((2, 3, 2), 30.0))
    )
    target_spectra = torch.stack(
        (torch.full((2, 3, 2), 1.0), torch.full((2, 3, 2), 2.0), torch.full((2, 3, 2), 3.0))
    )
    loss_fn = CompositeLoss(
        components=[{"type": "_DirectSpectrumSpeechLoss", "weight": 1.0}]
    )

    actual = loss_fn(
        enhanced,
        clean,
        enhanced_spectra=enhanced_spectra,
        target_spectra=target_spectra,
        is_pure_noise=torch.tensor([True, False, False]),
    )

    # 第 0 条由 pure metadata 排除，第 1 条由 clean-energy guard 排除。
    expected = enhanced_spectra[2:3].mean() + target_spectra[2:3].mean()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("metadata", "speech_indices"),
    [
        pytest.param(
            {"is_pure_noise": torch.tensor([False, True, False, False])},
            [0, 2, 3],
            id="pure-noise",
        ),
        pytest.param(
            {"is_pure_echo": torch.tensor([False, False, True, False])},
            [0, 1, 3],
            id="pure-echo",
        ),
        pytest.param(
            {"is_pure_echo_noise": torch.tensor([True, False, False, False])},
            [1, 2, 3],
            id="pure-echo-noise",
        ),
        pytest.param(
            {
                "is_pure_noise": torch.tensor([True, False, False, False]),
                "is_pure_echo": torch.tensor([False, True, False, False]),
                "is_pure_echo_noise": torch.tensor([False, False, True, False]),
            },
            [3],
            id="bitwise-or-all-three",
        ),
    ],
)
def test_gtcrn_speech_reference_routing_ors_pure_flags(metadata, speech_indices):
    """三种 pure flag 按位 OR，仅裁 speech-reference ratio 组件。"""
    g = torch.Generator().manual_seed(7)
    clean = torch.randn(4, 256, generator=g)
    enhanced = clean + 0.1 * torch.randn(4, 256, generator=g)
    lengths = torch.tensor([256, 224, 192, 160])
    stft_args = {"n_fft": 64, "hop_length": 16, "win_length": 64}
    loss_fn = build_loss(
        {
            "type": "CompositeLoss",
            "components": [
                {"type": "GTCRNCompressedSpectralLoss", "weight": 1.0},
                {"type": "GTCRNLogRatioLoss", "weight": 1.0},
            ],
        },
        stft_cfg=stft_args,
    )

    actual = loss_fn(enhanced, clean, lengths, **metadata)
    speech_idx = torch.tensor(speech_indices, dtype=torch.long)
    expected = GTCRNCompressedSpectralLoss(**stft_args)(enhanced, clean, lengths)
    expected = expected + GTCRNLogRatioLoss()(
        enhanced.index_select(0, speech_idx),
        clean.index_select(0, speech_idx),
        lengths.index_select(0, speech_idx),
    )

    assert GTCRNLogRatioLoss.REQUIRES_SPEECH_REFERENCE is True
    assert not hasattr(GTCRNCompressedSpectralLoss, "REQUIRES_SPEECH_REFERENCE")
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "flag_name",
    ["is_pure_noise", "is_pure_echo", "is_pure_echo_noise"],
)
def test_pure_metadata_rejects_non_tensor_on_cpu(flag_name):
    loss_fn = CompositeLoss(components=[{"type": "SISNRLoss", "weight": 1.0}])
    enhanced = torch.randn(2, 64)
    clean = torch.randn(2, 64)

    with pytest.raises(
        TypeError,
        match=rf"{flag_name} must be a \[B\] Tensor, got list",
    ):
        loss_fn(enhanced, clean, **{flag_name: [False, True]})


@pytest.mark.parametrize(
    "bad_flag,bad_shape",
    [
        (torch.tensor(False), r"\(\)"),
        (torch.tensor([[False], [True]]), r"\(2, 1\)"),
        (torch.tensor([False]), r"\(1,\)"),
    ],
)
def test_pure_metadata_rejects_wrong_shape_on_cpu(bad_flag, bad_shape):
    loss_fn = CompositeLoss(components=[{"type": "SISNRLoss", "weight": 1.0}])
    enhanced = torch.randn(2, 64)
    clean = torch.randn(2, 64)

    with pytest.raises(
        ValueError,
        match=rf"is_pure_noise must be a \[B\] Tensor, got shape {bad_shape} for B=2",
    ):
        loss_fn(enhanced, clean, is_pure_noise=bad_flag)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_pure_metadata_matches_cuda_result_and_gradient():
    torch.manual_seed(1234)
    device = torch.device("cuda")
    clean = torch.randn(3, 256, device=device)
    base_enhanced = clean + 0.15 * torch.randn_like(clean)
    lengths = torch.tensor([256, 224, 192], device=device, dtype=torch.long)
    cpu_metadata = {
        "is_pure_noise": torch.tensor([False, True, False]),
        "is_pure_echo": torch.tensor([False, False, True]),
        "is_pure_echo_noise": torch.zeros(3, dtype=torch.bool),
    }

    def _loss_and_grad(metadata):
        enhanced = base_enhanced.detach().clone().requires_grad_()
        loss_fn = CompositeLoss(
            components=[
                {"type": "L1WavLoss", "weight": 0.25},
                {"type": "SISNRLoss", "weight": 1.0},
            ]
        ).to(device)
        loss = loss_fn(enhanced, clean, lengths, **metadata)
        grad = torch.autograd.grad(loss, enhanced)[0]
        return loss.detach(), grad

    cpu_flag_loss, cpu_flag_grad = _loss_and_grad(cpu_metadata)
    cuda_metadata = {name: flag.to(device) for name, flag in cpu_metadata.items()}
    cuda_flag_loss, cuda_flag_grad = _loss_and_grad(cuda_metadata)

    assert torch.allclose(cpu_flag_loss, cuda_flag_loss, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_flag_grad, cuda_flag_grad, atol=1e-6, rtol=1e-6)


def test_ss_undesired_energy_mixing():
    """enable_ss_undesired_energy produces per-sample energy-weighted mixing."""
    enhanced = torch.randn(2, 16000)
    clean = torch.randn(2, 16000)
    noisy = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 16000], dtype=torch.long)

    loss_fn = CompositeLoss(
        components=[{"type": "L1WavLoss", "weight": 1.0, "enable_ss_undesired_energy": True}]
    )
    total = loss_fn(
        enhanced,
        clean,
        lengths,
        undesired_enhanced=noisy - enhanced,
        undesired_ref=noisy - clean,
    )

    # Manual per-sample mixing.
    l1 = torch.nn.L1Loss(reduction="none")
    lt = l1(enhanced, clean).mean(dim=-1)
    lu = l1(noisy - enhanced, noisy - clean).mean(dim=-1)
    e_clean = (clean ** 2).mean(dim=-1)
    e_undes = ((noisy - clean) ** 2).mean(dim=-1)
    a = e_clean / (e_clean + e_undes + 1e-8)
    expected = (a * lt + (1 - a) * lu).mean()

    assert torch.allclose(total, expected, atol=1e-5)


def test_mrsi_ss_is_invariant_to_pure_metadata():
    """MRSI 是否执行 SS 仅由顶层或 component 开关决定。"""
    enhanced, clean, undesired_enhanced, undesired_ref = _deterministic_mrsi_inputs()
    component_args = {
        "fft_power_min": 6,
        "fft_power_max": 6,
        "use_frequency_weight": False,
    }
    mrsi = MRSIRelCspecL2Loss(**component_args)

    target_losses = torch.stack(
        [mrsi(enhanced[b : b + 1], clean[b : b + 1]) for b in range(3)]
    )
    removed_losses = torch.stack(
        [
            mrsi(undesired_enhanced[b : b + 1], undesired_ref[b : b + 1])
            for b in range(3)
        ]
    )
    target_energy = clean.square().sum(dim=1)
    removed_energy = undesired_ref.square().sum(dim=1)
    target_weight = target_energy / (target_energy + removed_energy + 1e-8)
    expected_ss = (
        target_weight * target_losses + (1.0 - target_weight) * removed_losses
    ).mean()
    prob_target_only = mrsi(enhanced, clean)
    assert not torch.allclose(expected_ss, prob_target_only, atol=1e-5, rtol=1e-5)

    global_ss = CompositeLoss(
        components=[
            {"type": "MRSIRelCspecL2Loss", "weight": 1.0, "args": component_args}
        ],
        enable_ss_undesired_energy=True,
    )
    global_results = [
        global_ss(
            enhanced,
            clean,
            undesired_enhanced=undesired_enhanced,
            undesired_ref=undesired_ref,
            **metadata,
        )
        for metadata in _pure_metadata_variants()
    ]
    for result in global_results:
        assert torch.allclose(result, expected_ss, atol=1e-5, rtol=1e-5)

    prob_target_only_override = CompositeLoss(
        components=[
            {
                "type": "MRSIRelCspecL2Loss",
                "weight": 1.0,
                "args": component_args,
                "enable_ss_undesired_energy": False,
            }
        ],
        enable_ss_undesired_energy=True,
    )
    override_results = [
        prob_target_only_override(enhanced, clean, **metadata)
        for metadata in _pure_metadata_variants()
    ]
    for result in override_results:
        assert torch.allclose(result, prob_target_only, atol=1e-5, rtol=1e-5)


_FRAME_RELATIVE_FACTORS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]


def _frame_relative_mrsi_composite(
    win_length: int,
    hop_length: int,
    *,
    component_ss: bool,
) -> CompositeLoss:
    # 通过 builder 注入顶层 STFT，覆盖 frame-relative schedule 的真实入口。
    return build_loss(
        {
            "type": "CompositeLoss",
            "components": [
                {
                    "type": "MRSIRelCspecL2Loss",
                    "weight": 1.0,
                    "args": {
                        "stft_scale_factors": _FRAME_RELATIVE_FACTORS,
                        "use_frequency_weight": False,
                    },
                    "enable_ss_undesired_energy": component_ss,
                }
            ],
            # 同时覆盖 component 对顶层开关的双向覆盖语义。
            "enable_ss_undesired_energy": not component_ss,
        },
        stft_cfg={"win_length": win_length, "hop_length": hop_length},
    )


@pytest.mark.unit
@pytest.mark.parametrize("sample_rate,win_length,hop_length", [(16000, 320, 160), (32000, 640, 320)])
@pytest.mark.parametrize("component_ss", [False, True])
def test_frame_relative_mrsi_component_forward_backward_is_finite(
    sample_rate,
    win_length,
    hop_length,
    component_ss,
):
    torch.manual_seed(sample_rate + int(component_ss))
    clean = torch.randn(1, 4096)
    undesired_ref = 0.25 * torch.randn_like(clean)
    enhanced = (0.8 * clean + 0.1 * torch.randn_like(clean)).requires_grad_()
    loss_fn = _frame_relative_mrsi_composite(
        win_length,
        hop_length,
        component_ss=component_ss,
    )

    kwargs = {}
    if component_ss:
        kwargs = {
            "undesired_enhanced": 0.7 * undesired_ref + 0.1 * enhanced,
            "undesired_ref": undesired_ref,
        }
    loss = loss_fn(enhanced, clean, **kwargs)

    assert torch.isfinite(loss)
    loss.backward()
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()


@pytest.mark.unit
def test_frame_relative_mrsi_b_schedule_preserves_physical_timescale():
    mrsi_16k = _frame_relative_mrsi_composite(
        320,
        160,
        component_ss=False,
    ).losses[0]
    mrsi_32k = _frame_relative_mrsi_composite(
        640,
        320,
        component_ss=False,
    ).losses[0]

    assert mrsi_16k.win_lengths == (40, 80, 160, 320, 640, 1280)
    assert mrsi_16k.hop_lengths == (20, 40, 80, 160, 320, 640)
    assert mrsi_32k.win_lengths == (80, 160, 320, 640, 1280, 2560)
    assert mrsi_32k.hop_lengths == (40, 80, 160, 320, 640, 1280)
    assert [value / 16000 for value in mrsi_16k.win_lengths] == pytest.approx(
        [value / 32000 for value in mrsi_32k.win_lengths]
    )
    assert [value / 16000 for value in mrsi_16k.hop_lengths] == pytest.approx(
        [value / 32000 for value in mrsi_32k.hop_lengths]
    )


pytestmark = [pytest.mark.unit]
