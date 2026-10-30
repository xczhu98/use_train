"""MeanSE complex-STFT 集成测试。"""

from __future__ import annotations

from pathlib import Path
from types import MethodType

import pytest
import torch

from core.models.build import build_model
from core.utils.config import load_merged_config
from tasks.se.models.generative.meanse.model import MeanSE
from tasks.se.trainer.entry import SETrainingEntry
from tasks.se.trainer.meanflowse_trainer import MeanFlowSETrainer


pytestmark = pytest.mark.unit


def _tiny_model(**kwargs) -> MeanSE:
    defaults = {
        "n_fft": 14,
        "hop_length": 4,
        "win_length": 14,
        "nf": 8,
        "ch_mult": (1, 2),
        "num_res_blocks": 1,
        "attention_heads": 2,
    }
    defaults.update(kwargs)
    return MeanSE(**defaults)


def _spec(batch: int = 2, bins: int = 8, frames: int = 9) -> torch.Tensor:
    return torch.randn(batch, bins, frames, 2)


def test_registry_build_injects_stft_and_forward_is_finite():
    model = build_model(
        {
            "type": "MeanSE",
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
    ).eval()

    output = model(_spec(frames=11))

    assert isinstance(model, MeanSE)
    assert output.shape == (2, 8, 11, 2)
    assert torch.isfinite(output).all()


def test_constructor_and_frequency_contracts_fail_fast():
    with pytest.raises(TypeError, match="unexpected"):
        _tiny_model(unknown_option=True)
    with pytest.raises(ValueError, match="strict MeanSE preset"):
        _tiny_model(strict_preset=True)
    with pytest.raises(ValueError, match="sampling_steps"):
        _tiny_model(sampling_steps=0)
    with pytest.raises(ValueError, match="frequency bins"):
        _tiny_model()(_spec(bins=7))


def test_flow_path_matches_observation_centered_gaussian_math():
    model = _tiny_model(sigma=0.5)
    clean = torch.complex(torch.ones(2, 8, 3), torch.zeros(2, 8, 3))
    noisy = 3.0 * clean
    noise = 2.0 * clean
    t = torch.tensor([0.25, 0.75])

    state, velocity, prior = model.flow_path(clean, noisy, t, noise)
    expected_prior = noisy + 0.5 * noise
    expected_state = (1.0 - t[:, None, None]) * clean + t[:, None, None] * expected_prior

    torch.testing.assert_close(prior, expected_prior)
    torch.testing.assert_close(state, expected_state)
    torch.testing.assert_close(velocity, expected_prior - clean)


def test_flow_mixup_and_curriculum_interval_contract():
    model = _tiny_model(flow_ratio=0.75, max_interval=0.2)

    r_flow, t_flow, train_flow = model.sample_r_t(128, torch.device("cpu"), force_flow=True)
    r_mean, t_mean, train_mean = model.sample_r_t(1024, torch.device("cpu"), force_flow=False)

    torch.testing.assert_close(r_flow, t_flow)
    assert train_flow
    assert not train_mean
    assert torch.all(r_mean <= t_mean)
    assert torch.all(t_mean - r_mean <= 0.2 + 1.0e-7)


def test_meanflow_objective_has_detached_target_and_finite_backward():
    model = _tiny_model(loss_type="mse", flow_ratio=0.0)
    noisy = _spec()
    clean = _spec()
    t = torch.tensor([0.6, 0.8])
    r = torch.tensor([0.2, 0.3])
    noise = torch.zeros(2, 8, 9, dtype=torch.complex64)

    loss, diagnostics = model.compute_objective(
        noisy, clean, r=r, t=t, noise=noise
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert diagnostics["train_flow"].item() == pytest.approx(0.0)
    assert diagnostics["mean_interval"].item() == pytest.approx(0.45)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_r_equals_t_reduces_target_to_instantaneous_velocity():
    model = _tiny_model(loss_type="mse")
    noisy = _spec()
    clean = _spec()
    t = torch.tensor([0.4, 0.7])
    noise = torch.zeros(2, 8, 9, dtype=torch.complex64)

    loss, diagnostics = model.compute_objective(
        noisy, clean, r=t, t=t, noise=noise
    )

    assert torch.isfinite(loss)
    assert diagnostics["train_flow"].item() == pytest.approx(1.0)
    assert diagnostics["mean_interval"].item() == pytest.approx(0.0)


@pytest.mark.parametrize("steps", (1, 2, 5))
def test_sampling_steps_control_nfe_and_uniform_intervals(steps: int):
    model = _tiny_model(sigma=0.0, sampling_steps=steps).eval()
    transformed = torch.full((1, 8, 5), 2.0 + 0.0j)
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    model._transform = MethodType(lambda self, spec: transformed, model)
    model._inverse_transform = MethodType(lambda self, spec: spec, model)

    def constant_velocity(self, state, r, t, condition):
        calls.append((r.clone(), t.clone()))
        return torch.ones_like(state)

    model.velocity = MethodType(constant_velocity, model)

    enhanced = torch.view_as_complex(model.sample(_spec(1, 8, 5)).contiguous())

    assert len(calls) == steps
    torch.testing.assert_close(enhanced, transformed - 1.0)
    for index, (r, t) in enumerate(calls):
        assert t.item() == pytest.approx(1.0 - index / steps)
        assert r.item() == pytest.approx(1.0 - (index + 1) / steps)


def test_inference_is_independent_of_call_history_and_seed_sensitive():
    first = _tiny_model(inference_seed=1234).eval()
    second = _tiny_model(inference_seed=5678).eval()
    second.load_state_dict(first.state_dict())
    noisy = _spec(frames=11)

    first_output = first(noisy)
    torch.randn(97)
    repeated_output = first(noisy)
    second_output = second(noisy)

    torch.testing.assert_close(first_output, repeated_output, rtol=0.0, atol=0.0)
    assert not torch.equal(first_output, second_output)


def test_time_fusion_is_initialized_as_original_t_embedding():
    model = _tiny_model()
    t = torch.tensor([0.3, 0.8])
    r = torch.tensor([0.1, 0.2])

    original_t = model.backbone.time_mlp(model.backbone.time_embed(t))
    fused = model.backbone.fused_time_embedding(t, r)

    torch.testing.assert_close(fused, original_t)


def test_config_trainer_and_provenance_contract():
    root = Path(__file__).resolve().parents[1]
    config = load_merged_config(
        root / "configs/se/voicebank_demand/meanse_ncsnpp_offline.yaml"
    )

    assert config["model"]["type"] == "MeanSE"
    assert config["model"]["args"]["sampling_steps"] == 1
    assert config["model"]["args"]["flow_ratio"] == pytest.approx(0.75)
    assert config["model"]["args"]["max_interval"] == pytest.approx(1.0)
    assert config["data"]["train"]["max_duration"] == 40000
    assert config["trainer"]["type"] == "MeanFlowSETrainer"
    assert SETrainingEntry("test")._import_trainer_class("MeanFlowSETrainer") is MeanFlowSETrainer

    source_dir = root / "tasks/se/models/generative/meanse"
    provenance = (source_dir / "PROVENANCE").read_text(encoding="utf-8")
    model_source = (source_dir / "model.py").read_text(encoding="utf-8")
    assert "a4ced60cf3aa365c481319632e8ea762d178922b" in provenance
    assert "许可证：未确认" in provenance
    assert "不是同一篇论文" in provenance
    assert "不是同一篇论文" in model_source
    assert "``PROVENANCE``" in model_source
