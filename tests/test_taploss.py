"""TAPLoss 的 registry、公式模式和长度边界测试。"""

import inspect

import pytest
import torch

from core.losses import AcousticLoss, CompositeLoss, TAPLoss
from core.losses.build import build_loss
from core.losses.tap import _TAPAcousticEstimator
from core.utils.registry import LOSS_REGISTRY

pytestmark = pytest.mark.unit


def _checkpoint(path):
    torch.save({"model_state_dict": _TAPAcousticEstimator().state_dict()}, path)


def test_taploss_is_registered_under_canonical_and_official_names(tmp_path):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)

    loss = TAPLoss(acoustic_model_path=checkpoint)

    assert LOSS_REGISTRY.get("TAPLoss") is TAPLoss
    assert LOSS_REGISTRY.get("AcousticLoss") is TAPLoss
    assert isinstance(loss, AcousticLoss)
    assert all(not parameter.requires_grad for parameter in loss.estimate_acoustics.parameters())
    assert not loss.estimate_acoustics.training


def test_taploss_uses_project_argument_order_and_eval_keeps_autograd(tmp_path):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)
    loss = TAPLoss(acoustic_model_path=checkpoint)

    parameters = list(inspect.signature(TAPLoss.forward).parameters.values())
    assert [parameter.name for parameter in parameters[:4]] == [
        "self",
        "enhanced",
        "clean",
        "lengths",
    ]
    assert parameters[4].name == "mode"
    assert parameters[4].kind is inspect.Parameter.KEYWORD_ONLY

    enhanced = torch.randn(1, 768, requires_grad=True)
    clean = torch.randn(1, 768)
    value = loss(enhanced, clean, mode="eval")

    assert value.ndim == 0
    assert value.requires_grad
    value.backward()
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()
    assert all(parameter.grad is None for parameter in loss.estimate_acoustics.parameters())


def test_taploss_supports_all_official_loss_types_and_backpropagates(tmp_path):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)
    enhanced = torch.randn(2, 1024, requires_grad=True)
    clean = torch.randn(2, 1024)

    for loss_type in (
        "l2",
        "l1",
        "frame_energy_weighted_l2",
        "frame_energy_weighted_l1",
    ):
        enhanced.grad = None
        loss = TAPLoss(loss_type=loss_type, acoustic_model_path=checkpoint)
        value = loss(enhanced, clean)
        assert value.ndim == 0
        assert torch.isfinite(value)
        value.backward()
        assert enhanced.grad is not None
        assert torch.isfinite(enhanced.grad).all()
        assert all(parameter.grad is None for parameter in loss.estimate_acoustics.parameters())


def test_taploss_lengths_ignore_waveform_padding(tmp_path):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)
    generator = torch.Generator().manual_seed(4)
    clean = torch.randn(2, 768, generator=generator)
    enhanced = (clean + 0.05 * torch.randn(2, 768, generator=generator)).requires_grad_()

    loss = TAPLoss(acoustic_model_path=checkpoint)
    value = loss(enhanced, clean, lengths=torch.tensor([512, 768]))
    value.backward()

    assert torch.isfinite(value)
    assert torch.all(enhanced.grad[0, 512:] == 0)


def test_taploss_rejects_invalid_type_and_short_waveform(tmp_path):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)

    with pytest.raises(ValueError, match="Invalid TAPLoss loss_type"):
        TAPLoss(loss_type="smooth_l1", acoustic_model_path=checkpoint)

    loss = TAPLoss(acoustic_model_path=checkpoint)
    with pytest.raises(ValueError, match="length > n_fft"):
        loss(torch.randn(1, 256), torch.randn(1, 256))

    with pytest.raises(ValueError, match="lengths values"):
        loss(
            torch.randn(1, 512),
            torch.randn(1, 512),
            lengths=torch.tensor([513]),
        )

    with pytest.raises(ValueError, match="mode must be"):
        loss(torch.randn(1, 512), torch.randn(1, 512), mode="validation")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_rate": 8000}, "sample_rate=16000"),
        ({"sample_rate": 32000}, "sample_rate=16000"),
        ({"n_fft": 1024}, "n_fft=512"),
        ({"hop_length": 80}, "hop_length=160"),
        ({"hop_length": 320}, "hop_length=160"),
        ({"win_length": 400}, "win_length.*512"),
    ],
)
def test_taploss_rejects_non_official_sample_rate_and_hop(tmp_path, kwargs, message):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)

    with pytest.raises(ValueError, match=message):
        TAPLoss(acoustic_model_path=checkpoint, **kwargs)


def test_taploss_stft_matches_official_interleaved_layout_and_energy(tmp_path):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)
    loss = TAPLoss(acoustic_model_path=checkpoint)
    waveform = torch.arange(512, dtype=torch.float32).unsqueeze(0) / 512.0

    features, energy = loss._stft_features(waveform)
    spec = torch.stft(
        waveform,
        n_fft=512,
        hop_length=160,
        win_length=512,
        window=torch.ones(512),
        center=True,
        return_complex=True,
    )
    expected_features = torch.view_as_real(spec).permute(0, 2, 1, 3).reshape(1, -1, 514)
    expected_energy = (2.0 / 512.0) * spec.abs().square().sum(dim=1)

    assert "rectangular_window" in dict(loss.named_buffers())
    assert torch.equal(loss.rectangular_window, torch.ones(512))
    assert torch.allclose(features, expected_features)
    assert torch.allclose(energy, expected_energy)
    assert torch.equal(features[..., 0::2], spec.real.transpose(1, 2))
    assert torch.equal(features[..., 1::2], spec.imag.transpose(1, 2))


@pytest.mark.parametrize(
    ("loss_type", "expected"),
    [
        ("l1", 2.0),
        ("l2", 5.0),
        (
            "frame_energy_weighted_l1",
            (torch.sigmoid(torch.tensor(0.0)) + 3.0 * torch.sigmoid(torch.tensor(2.0))) / 2.0,
        ),
        (
            "frame_energy_weighted_l2",
            (torch.sigmoid(torch.tensor(0.0)) + 9.0 * torch.sigmoid(torch.tensor(2.0))) / 2.0,
        ),
    ],
)
def test_taploss_l1_l2_and_energy_weighted_numerical_oracle(
    tmp_path, monkeypatch, loss_type, expected
):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)
    loss = TAPLoss(loss_type=loss_type, acoustic_model_path=checkpoint)
    enhanced_acoustics = torch.tensor([[[1.0] * 25, [3.0] * 25]])
    clean_acoustics = torch.zeros_like(enhanced_acoustics)
    energy = torch.tensor([[0.0, 2.0]])
    monkeypatch.setattr(
        loss,
        "_forward_unpadded",
        lambda _enhanced, _clean: (enhanced_acoustics, clean_acoustics, energy),
    )

    actual = loss(torch.zeros(1, 512), torch.zeros(1, 512))

    assert actual.dtype == torch.float32
    assert actual == pytest.approx(float(expected), rel=1e-6)


def test_taploss_variable_lengths_use_global_valid_frame_weighted_mean_fp32(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)
    loss = TAPLoss(loss_type="l1", acoustic_model_path=checkpoint)

    def fake_forward(enhanced, _clean):
        if enhanced.shape[-1] == 300:
            acoustics = torch.tensor([[[1.0] * 25, [1.0] * 25]], dtype=torch.float16)
        else:
            acoustics = torch.tensor([[[4.0] * 25]], dtype=torch.float16)
        return acoustics, torch.zeros_like(acoustics), torch.zeros(
            1, acoustics.shape[1], dtype=torch.float16
        )

    monkeypatch.setattr(loss, "_forward_unpadded", fake_forward)
    actual = loss(
        torch.zeros(2, 400),
        torch.zeros(2, 400),
        lengths=torch.tensor([300, 400]),
    )

    assert actual.dtype == torch.float32
    assert actual == pytest.approx(2.0)


@pytest.mark.parametrize("loss_type", ["TAPLoss", "AcousticLoss"])
def test_builder_does_not_inject_top_level_stft_into_tap(tmp_path, loss_type):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)
    cfg = {"type": loss_type, "args": {"acoustic_model_path": checkpoint}}

    single = build_loss(
        cfg,
        stft_cfg={"n_fft": 1024, "hop_length": 256, "win_length": 1024},
        sample_rate=16000,
    )
    composite = build_loss(
        {"type": "CompositeLoss", "components": [cfg]},
        stft_cfg={"n_fft": 1024, "hop_length": 256, "win_length": 1024},
        sample_rate=16000,
    )

    assert (single.n_fft, single.hop_length, single.win_length) == (512, 160, 512)
    assert (
        composite.losses[0].n_fft,
        composite.losses[0].hop_length,
        composite.losses[0].win_length,
    ) == (512, 160, 512)


@pytest.mark.parametrize("explicit_sample_rate", [None, 16000])
def test_builder_sample_rate_context_fails_fast_for_tap(
    tmp_path, explicit_sample_rate
):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)
    args = {"acoustic_model_path": checkpoint}
    if explicit_sample_rate is not None:
        args["sample_rate"] = explicit_sample_rate

    with pytest.raises(ValueError, match="sample_rate=16000"):
        build_loss(
            {"type": "TAPLoss", "args": args},
            sample_rate=32000,
        )


@pytest.mark.parametrize("loss_type", ["TAPLoss", "AcousticLoss"])
@pytest.mark.parametrize(
    ("global_ss", "component_ss"),
    [(True, None), (False, True), (True, True)],
)
def test_composite_rejects_effective_ss_for_tap(
    tmp_path, loss_type, global_ss, component_ss
):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)
    component = {
        "type": loss_type,
        "args": {"acoustic_model_path": checkpoint},
    }
    if component_ss is not None:
        component["enable_ss_undesired_energy"] = component_ss

    with pytest.raises(ValueError, match=r"TAPLoss|AcousticLoss"):
        CompositeLoss(
            components=[component],
            enable_ss_undesired_energy=global_ss,
        )


def test_composite_allows_tap_component_to_disable_global_ss(tmp_path):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)

    loss = CompositeLoss(
        components=[
            {
                "type": "TAPLoss",
                "args": {"acoustic_model_path": checkpoint},
                "enable_ss_undesired_energy": False,
            }
        ],
        enable_ss_undesired_energy=True,
    )

    assert loss.any_ss_enabled is False


@pytest.mark.parametrize("pure_flag", ["is_pure_noise", "is_pure_echo", "is_pure_echo_noise"])
def test_composite_gates_tap_on_samples_without_speech_reference(tmp_path, pure_flag):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)
    loss = CompositeLoss(
        components=[
            {
                "type": "TAPLoss",
                "args": {"acoustic_model_path": checkpoint},
            }
        ]
    )
    enhanced = torch.randn(2, 512, requires_grad=True)
    clean = torch.randn(2, 512)
    pure = torch.tensor([False, True])

    actual = loss(enhanced, clean, **{pure_flag: pure})
    expected = loss.losses[0](enhanced[:1], clean[:1])

    assert torch.allclose(actual, expected)


def test_composite_all_pure_tap_batch_returns_graph_preserving_zero(tmp_path):
    checkpoint = tmp_path / "tap.pt"
    _checkpoint(checkpoint)
    loss = CompositeLoss(
        components=[
            {
                "type": "TAPLoss",
                "args": {"acoustic_model_path": checkpoint},
            }
        ]
    )
    enhanced = torch.randn(2, 512, requires_grad=True)

    actual = loss(
        enhanced,
        torch.zeros_like(enhanced),
        is_pure_noise=torch.ones(2, dtype=torch.bool),
    )
    actual.backward()

    assert actual.item() == 0.0
    assert enhanced.grad is not None
    assert torch.equal(enhanced.grad, torch.zeros_like(enhanced))
