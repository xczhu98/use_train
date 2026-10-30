"""MeanFlowSE complex-STFT 集成测试。"""

from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from core.models.build import build_model
from core.utils.config import load_merged_config
from tasks.se.models.generative.meanflowse.model import (
    MeanFlowSE,
    inverse_spectral_transform,
    spectral_transform,
)
from tasks.se.trainer.entry import SETrainingEntry
from tasks.se.trainer.meanflowse_trainer import MeanFlowSETrainer


pytestmark = pytest.mark.unit


def _tiny_model(**kwargs) -> MeanFlowSE:
    defaults = {
        "n_fft": 14,
        "hop_length": 4,
        "win_length": 14,
        "nf": 8,
        "ch_mult": (1, 2),
        "num_res_blocks": 1,
        "attention_heads": 2,
        "mf_jvp_chunk": 1,
    }
    defaults.update(kwargs)
    return MeanFlowSE(**defaults)


def _spec(batch: int = 2, bins: int = 8, frames: int = 9) -> torch.Tensor:
    return torch.randn(batch, bins, frames, 2)


def test_official_spectral_transform_is_invertible():
    spec = _spec()
    spec[0, 0, 0] = torch.tensor([1.0e-20, 0.0])
    transformed = spectral_transform(spec, factor=0.15, exponent=0.5)
    restored = inverse_spectral_transform(transformed, factor=0.15, exponent=0.5)

    assert transformed.is_complex()
    assert transformed[0, 0, 0].real.item() == pytest.approx(1.5e-11)
    torch.testing.assert_close(
        torch.view_as_real(restored), spec, rtol=2.0e-5, atol=2.0e-5
    )


def test_flow_path_matches_official_mean_and_velocity_math():
    model = _tiny_model(sigma_min=0.1, sigma_max=0.5)
    clean = torch.complex(torch.ones(2, 8, 3), torch.zeros(2, 8, 3))
    noisy = 3.0 * clean
    noise = 2.0 * clean
    t = torch.tensor([0.25, 0.75])

    x_t, velocity = model.flow_path(clean, noisy, t, noise)
    view_t = t[:, None, None]
    sigma = ((1.0 - t) * 0.1 + t * 0.5)[:, None, None]
    expected_x = (1.0 - view_t) * clean + view_t * noisy + sigma * noise
    expected_velocity = noisy - clean + 0.4 * noise

    torch.testing.assert_close(x_t, expected_x)
    torch.testing.assert_close(velocity, expected_velocity)


def test_fd_jvp_matches_linear_vector_field():
    model = _tiny_model(mf_jvp_eps=1.0e-3, mf_jvp_clip=1.0e6)

    def linear_velocity(self, x, t, condition, r=None):
        return (
            2.0 * x
            + 3.0 * t[:, None, None]
            + 5.0 * condition
            + 7.0 * r[:, None, None]
        )

    model.velocity = MethodType(linear_velocity, model)
    x_t = torch.randn(2, 8, 3, dtype=torch.complex64)
    direction = torch.randn_like(x_t)
    t = torch.tensor([0.4, 0.8])
    jvp = model.finite_difference_jvp(
        x_t,
        t,
        torch.zeros_like(x_t),
        t - 0.1,
        direction,
        torch.ones_like(t),
    )

    torch.testing.assert_close(jvp, 2.0 * direction + 3.0, rtol=2.0e-3, atol=2.0e-3)
    assert not jvp.requires_grad


def test_cfm_meanflow_objective_weighting_and_backward():
    model = _tiny_model(
        mf_weight_final=0.25,
        mf_warmup_frac=0.5,
        mf_skip_weight_thresh=0.0,
        mf_r_equals_t_prob=1.0,
    )
    noisy = _spec()
    clean = _spec()
    t = torch.full((2,), 0.6)
    noise = torch.zeros(2, 8, 9, dtype=torch.complex64)

    loss, diagnostics = model.compute_objective(
        noisy,
        clean,
        progress=1.0,
        t=t,
        noise=noise,
        r=t,
    )
    expected = 0.75 * diagnostics["loss_cfm"] + 0.25 * diagnostics["loss_mf"]

    torch.testing.assert_close(loss.detach(), expected)
    torch.testing.assert_close(diagnostics["loss_mf"], diagnostics["loss_cfm"])
    assert diagnostics["mf_weight"].item() == pytest.approx(0.25)
    assert model._mf_weight(0.25) == pytest.approx(0.125)
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_one_step_euler_mf_standard_inference_shape_and_finite():
    model = _tiny_model().eval()
    noisy = _spec(frames=11)

    enhanced = model(noisy)

    assert enhanced.shape == noisy.shape
    assert not enhanced.is_complex()
    assert torch.isfinite(enhanced).all()


def test_inference_is_independent_of_call_history():
    model = _tiny_model(inference_seed=1234).eval()
    noisy = _spec(frames=11)

    first = model(noisy)
    torch.randn(97)
    second = model(noisy)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_different_inference_seeds_change_output():
    first_model = _tiny_model(inference_seed=1234).eval()
    second_model = _tiny_model(inference_seed=5678).eval()
    second_model.load_state_dict(first_model.state_dict())
    noisy = _spec(frames=11)

    first = first_model(noisy)
    second = second_model(noisy)

    assert not torch.equal(first, second)


def test_training_objective_keeps_global_rng_and_backward_semantics():
    model = _tiny_model(mf_weight_final=0.0)
    noisy = _spec()
    clean = _spec()
    rng_before = torch.random.get_rng_state()

    first_loss, _ = model.compute_objective(noisy, clean)
    rng_after_first = torch.random.get_rng_state()
    second_loss, _ = model.compute_objective(noisy, clean)

    assert not torch.equal(rng_before, rng_after_first)
    assert not torch.equal(first_loss.detach(), second_loss.detach())
    second_loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_one_step_uses_official_backward_displacement_sign():
    model = _tiny_model(sigma_min=0.0, sigma_max=0.0).eval()
    transformed = torch.full((1, 8, 5), 2.0 + 0.0j)
    model._transform = MethodType(lambda self, spec: transformed, model)
    model._inverse_transform = MethodType(lambda self, spec: spec, model)
    model.velocity = MethodType(
        lambda self, x, t, condition, r=None: torch.ones_like(x), model
    )

    enhanced = torch.view_as_complex(model.euler_mf(_spec(1, 8, 5)).contiguous())

    torch.testing.assert_close(enhanced, transformed - (model.t_max - model.t_eps))


def test_length_mask_removes_padded_frames_from_cfm_objective():
    model = _tiny_model(mf_weight_final=0.0)
    model.velocity = MethodType(
        lambda self, x, t, condition, r=None: torch.zeros_like(x), model
    )
    clean = torch.zeros(1, 8, 5, 2)
    noisy = torch.zeros_like(clean)
    noisy[..., 3:, :] = 100.0
    noise = torch.zeros(1, 8, 5, dtype=torch.complex64)

    masked, _ = model.compute_objective(
        noisy, clean, lengths=torch.tensor([8]), t=torch.tensor([0.5]), noise=noise
    )
    unmasked, _ = model.compute_objective(
        noisy, clean, t=torch.tensor([0.5]), noise=noise
    )

    assert masked.item() == pytest.approx(0.0)
    assert unmasked.item() > 0.0


def test_constructor_and_frequency_contracts_fail_fast():
    assert _tiny_model().inference_peak_normalization is True
    assert _tiny_model(inference_peak_normalization=False).inference_peak_normalization is False
    with pytest.raises(TypeError, match="unexpected"):
        _tiny_model(unknown_option=True)
    with pytest.raises(TypeError, match="inference_peak_normalization must be bool"):
        _tiny_model(inference_peak_normalization="true")
    with pytest.raises(ValueError, match="strict MeanFlowSE preset"):
        _tiny_model(strict_preset=True)
    with pytest.raises(ValueError, match="frequency bins"):
        _tiny_model()(_spec(bins=7))


def test_registry_build_uses_injected_stft_and_tiny_backbone():
    model = build_model(
        {
            "type": "MeanFlowSE",
            "args": {
                "nf": 8,
                "ch_mult": [1, 2],
                "num_res_blocks": 1,
                "attention_heads": 2,
            },
        },
        stft_cfg={
            "n_fft": 14,
            "hop_length": 4,
            "win_length": 14,
            "window": "hann",
            "center": True,
        },
    )

    assert isinstance(model, MeanFlowSE)
    assert model.expected_freq_bins == 8


def test_task_local_trainer_resolver_finds_meanflowse_short_name():
    trainer_cls = SETrainingEntry("test")._import_trainer_class("MeanFlowSETrainer")

    assert trainer_cls is MeanFlowSETrainer


class _RecordingWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.called = False
        self.lengths = None

    def forward(self, noisy_spec, lengths, *, fs, clean_spec, progress):
        self.called = True
        self.lengths = lengths.clone()
        assert noisy_spec.shape == clean_spec.shape
        torch.testing.assert_close(
            noisy_spec.abs().flatten(1).amax(dim=1), torch.ones(2)
        )
        assert lengths.shape == (2,)
        assert fs == 16000
        assert progress == pytest.approx(0.4)
        loss = self.weight * (noisy_spec - clean_spec).square().mean()
        return loss, {"loss_cfm": loss.detach()}


class _IdentitySTFT:
    n_fft = 2

    def to(self, device):
        del device
        return self

    def __call__(self, wav):
        return wav[:, None, :, None].repeat(1, 2, 1, 2)

    def inverse(self, spec, length):
        return spec[:, 0, :length, 0]


class _PeakRecordingModel(nn.Module):
    inference_peak_normalization = True

    def __init__(self):
        super().__init__()
        self.received_peak = None

    def forward(self, noisy_spec, lengths):
        del lengths
        self.received_peak = noisy_spec.abs().amax().item()
        return noisy_spec


class _PlainRecordingModel(_PeakRecordingModel):
    inference_peak_normalization = False


class _ValidationWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, noisy_spec, lengths, *, fs, clean_spec=None, progress=1.0):
        self.calls.append(clean_spec is not None)
        if clean_spec is None:
            return noisy_spec
        return noisy_spec.new_tensor(3.0), {"loss_cfm": noisy_spec.new_tensor(3.0)}


@pytest.mark.parametrize(
    ("model_type", "wav", "expected_model_peak"),
    [
        (_PeakRecordingModel, torch.tensor([[0.5, -2.0, 1.0]]), 1.0),
        (_PeakRecordingModel, torch.zeros(1, 3), 0.0),
        (_PlainRecordingModel, torch.tensor([[0.5, -2.0, 1.0]]), 2.0),
    ],
)
def test_public_inference_peak_normalization_round_trip_is_model_gated(
    monkeypatch, tmp_path, model_type, wav, expected_model_peak
):
    import torchaudio

    from core.inference import FileInferenceItem
    from tasks.se.inference.adapter import SEInferenceAdapter

    captured = {}
    model = model_type()
    monkeypatch.setattr(torchaudio, "load", lambda _path: (wav.clone(), 16000))
    monkeypatch.setattr(
        torchaudio,
        "save",
        lambda path, data, rate: captured.update(path=path, data=data, rate=rate),
    )

    adapter = SEInferenceAdapter(
        model,
        stft_module=_IdentitySTFT(),
        sample_rate=16000,
    )
    output = adapter.infer(
        FileInferenceItem("u1", {"noisy": "fake.wav"}, tmp_path / "u1.wav")
    )
    output.writer(tmp_path / "u1.wav")

    assert model.received_peak == pytest.approx(expected_model_peak)
    torch.testing.assert_close(captured["data"], wav)
    assert torch.isfinite(captured["data"]).all()


def test_trainer_recipe_padding_is_symmetric_zero_not_repeat():
    trainer = object.__new__(MeanFlowSETrainer)
    trainer.recipe_num_samples = 6
    noisy = torch.tensor([[1.0, -2.0, 99.0, 99.0]])
    clean = torch.tensor([[0.5, -1.0, 88.0, 88.0]])

    padded_noisy, padded_clean, lengths = trainer._pad_training_pair(
        noisy, clean, torch.tensor([2])
    )

    torch.testing.assert_close(
        padded_noisy, torch.tensor([[0.0, 0.0, 1.0, -2.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(
        padded_clean, torch.tensor([[0.0, 0.0, 0.5, -1.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(lengths, torch.tensor([6]))


def test_meanflowse_recipe_config_and_provenance_contract():
    root = Path(__file__).resolve().parents[1]
    config = load_merged_config(
        root / "configs/se/voicebank_demand/meanflowse_ncsnpp_offline.yaml"
    )
    train_config = config["data"]["train"]
    assert train_config["max_duration"] == 32640
    assert train_config["repeat_to_max_duration"] is False
    assert config["model"]["args"]["inference_seed"] == 42
    assert config["model"]["args"]["inference_peak_normalization"] is True

    source_dir = root / "tasks/se/models/generative/meanflowse"
    provenance = (source_dir / "PROVENANCE").read_text(encoding="utf-8")
    assert "确定性适配" in provenance
    assert "task-local 项目适配" in provenance
    assert "LICENSE-MIT-MeanFlowSE" in provenance
    assert "LICENSE-Apache-2.0-score-sde" in provenance
    assert (source_dir / "LICENSE-MIT-MeanFlowSE").is_file()
    assert (source_dir / "LICENSE-Apache-2.0-score-sde").is_file()
    assert "``PROVENANCE``" in (source_dir / "model.py").read_text(encoding="utf-8")


def test_trainer_training_forward_uses_model_wrapper_and_returns_diagnostics():
    trainer = object.__new__(MeanFlowSETrainer)
    wrapper = _RecordingWrapper()
    trainer.model = wrapper
    trainer.stft_module = _IdentitySTFT()
    trainer.current_epoch = 5
    trainer.epochs = 10
    trainer.recipe_num_samples = 8
    batch = SimpleNamespace(
        noisy=torch.randn(2, 6),
        clean=torch.randn(2, 6),
        lengths=torch.tensor([6, 5]),
    )

    loss, finite, info = trainer._forward_and_compute_loss(batch, 5, 0, 16000)

    assert wrapper.called
    torch.testing.assert_close(wrapper.lengths, torch.tensor([8, 8]))
    assert loss.ndim == 0
    assert finite
    assert "diagnostics" in info


def test_trainer_validation_uses_objective_and_restores_noisy_peak_scale():
    from core.data.speech_enhancement_batch import SEBatch

    trainer = object.__new__(MeanFlowSETrainer)
    wrapper = _ValidationWrapper()
    trainer.model = wrapper
    trainer.stft_module = _IdentitySTFT()
    trainer.device = torch.device("cpu")
    trainer.epochs = 10
    trainer.current_epoch = 5
    trainer.use_amp = False
    trainer.amp_dtype = torch.float16
    trainer._strict_checks = True
    trainer.sample_rate = 16000
    trainer._sanitize_model_output = MethodType(lambda self, wav: wav, trainer)
    batch = SEBatch(
        torch.tensor([[0.5, -0.5, 0.25, 0.0]]),
        torch.tensor([[2.0, -2.0, 1.0, 0.0]]),
        torch.tensor(16000),
        torch.tensor([4]),
    )

    loss, enhanced, clean, lengths, fs, _ = trainer._val_forward_loss(batch, 5, 1)

    assert loss.item() == pytest.approx(3.0)
    assert wrapper.calls == [True, False]
    torch.testing.assert_close(enhanced, batch.noisy)
    torch.testing.assert_close(clean, batch.clean)
    torch.testing.assert_close(lengths, batch.lengths)
    assert fs == 16000
