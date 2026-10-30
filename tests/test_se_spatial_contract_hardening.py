"""SE spatial hardening contracts for runtime, trainer, and inference boundaries."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from core.data.dynamic_dataset_base import BaseDynamicMixingDataset
from core.trainer.entry import UnifiedTrainingEntry
from tasks.se.inference.adapter import SEInferenceAdapter
from tasks.se.trainer.entry import SETrainingEntry
from tasks.se.trainer.gan_trainer import GANTrainer


pytestmark = pytest.mark.unit


def test_spatial_placeholder_preserves_microphone_axis():
    dataset = object.__new__(BaseDynamicMixingDataset)
    dataset.max_duration = -1
    dataset.max_duration_seconds = None
    dataset.placeholder_num_samples = 16
    dataset.sim_cfg = SimpleNamespace(spatial_enabled=True)
    dataset._simulation_view = SimpleNamespace(
        shared=SimpleNamespace(
            spatial=SimpleNamespace(
                microphone_positions=((-0.04, 0.0, 0.0), (0.04, 0.0, 0.0))
            )
        )
    )
    observation, length = dataset._make_observation_placeholder(8000)
    assert observation.shape == (2, 16)
    assert length == 16


def test_default_gan_forward_rejects_spatial_noisy_input():
    trainer = object.__new__(GANTrainer)
    trainer.stft_module = object()
    with pytest.raises(ValueError, match=r"default GANTrainer.*\[B,C,T\]"):
        trainer._forward_generator(
            {
                "noisy": torch.zeros(1, 2, 16),
                "lengths": torch.tensor([16]),
            }
        )


def test_se_entry_rejects_spatial_gan_before_ddp_setup(monkeypatch):
    monkeypatch.setattr(UnifiedTrainingEntry, "_validate_config", lambda *_args: None)
    entry = object.__new__(SETrainingEntry)
    config = {
        "model": {"generator": {"type": "GTCRN", "args": {}}},
        "data": {
            "train": {
                "simulation_cfg": {
                    "spatial_enabled": True,
                }
            }
        },
    }
    with pytest.raises(ValueError, match="SE GAN trainers do not support spatial_enabled"):
        entry._validate_config(config)


def test_se_entry_keeps_mono_gan_ddp_derivation(monkeypatch):
    monkeypatch.setattr(UnifiedTrainingEntry, "_validate_config", lambda *_args: None)
    entry = object.__new__(SETrainingEntry)
    config = {
        "model": {"generator": {"type": "GTCRN", "args": {}}},
        "data": {"train": {"simulation_cfg": {"spatial_enabled": False}}},
    }
    entry._validate_config(config)
    assert config["distributed"]["find_unused_parameters"] is True


class _ConflictingRateModel(nn.Module):
    sample_rate = 16000
    target_fs = 8000

    def forward(self, *_args, **_kwargs):
        raise AssertionError("forward should not be called")


def test_se_inference_rejects_conflicting_model_sample_rate_declarations():
    with pytest.raises(ValueError, match="conflicting model sample-rate declarations"):
        SEInferenceAdapter(
            _ConflictingRateModel(),
            sample_rate=16000,
            supports_multichannel_input=False,
        )
