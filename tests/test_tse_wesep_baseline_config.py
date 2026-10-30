"""严格 WeSep baseline reproduction 配置合同测试。"""

from __future__ import annotations

import pytest
import torch

from core.optim.scheduler import resolve_scheduler_auto_args
from core.utils.config import load_merged_config
from core.utils.registry import SCHEDULER_REGISTRY


pytestmark = pytest.mark.unit


def _config() -> dict:
    return load_merged_config(
        "configs/tse/offline/tse_bsrnn_spk_wesep_baseline.yaml"
    )


def test_declared_wesep_baseline_hyperparameters_are_frozen():
    cfg = _config()
    train = cfg["data"]["train"]
    validation = cfg["data"]["validation"]
    encoder = cfg["model"]["args"]["speaker_model"]

    assert cfg["data"]["mode"] == "offline_tse"
    assert train["max_duration_seconds"] == 3.0
    assert validation["max_duration_seconds"] == 3.0
    assert encoder["fbank"]["dither"] == 1.0
    assert cfg["model"]["args"]["features"]["spkemb"] == {
        "enabled": True,
        "fusion": "multiply",
    }
    assert cfg["data"]["dataloader"]["batch_size"] == 15
    assert cfg["trainer"]["epochs"] == 150
    assert cfg["optimizer"] == {
        "type": "Adam",
        "args": {"lr": 1e-3, "weight_decay": 1e-4},
    }
    assert cfg["scheduler"]["args"]["initial_lr"] == 1e-3
    assert cfg["scheduler"]["args"]["final_lr"] == 2.5e-5
    speaker = encoder["speaker_encoder"]
    assert speaker["pretrained"] == "tasks/tse/pretrained_models/spk_emb_100/avg_model.pt"
    assert speaker["pretrained_prefix"] == "spk_ft.encoder.spk_model."


def test_baseline_scheduler_resolves_and_builds_from_registry():
    cfg = _config()
    args = resolve_scheduler_auto_args(
        cfg["scheduler"],
        steps_per_epoch=463,
        epochs=cfg["trainer"]["epochs"],
        per_step=cfg["trainer"]["lr_schedule_per_step"],
    )
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
    import core.optim  # noqa: F401

    scheduler = SCHEDULER_REGISTRY.build(
        cfg["scheduler"]["type"], optimizer=optimizer, **args
    )

    assert scheduler.total_steps == 150 * 463
    assert scheduler.initial_lr == 1e-3
    assert scheduler.final_lr == 2.5e-5


@pytest.mark.parametrize(
    ("name", "model_type"),
    [
        ("tse_bsrnn_spk_dual_self_estimated_wesep.yaml", "TSE_BSRNN_SPK"),
        (
            "tse_modular_speaker_tfmap_emb_wesep.yaml",
            "TSE_Modular",
        ),
        (
            "tse_modular_speaker_tfmap_emb_multi_fuse_wesep.yaml",
            "TSE_Modular",
        ),
    ],
)
def test_advanced_wesep_configs_inherit_strict_baseline(name, model_type):
    cfg = load_merged_config(
        f"configs/tse/offline/{name}"
    )

    assert cfg["model"]["type"] == model_type
    assert cfg["data"]["mode"] == "offline_tse"
    assert cfg["data"]["train"]["max_duration_seconds"] == 3.0
    assert cfg["trainer"]["epochs"] == 150
    assert cfg["scheduler"]["type"] == "ExponentialDecrease"
    if model_type == "TSE_Modular":
        assert cfg["model"]["args"]["separator"]["rnn_norm_eps"] == pytest.approx(1e-5)
        assert cfg["model"]["args"]["cues"]["speaker"]["encoder"]["pretrained_prefix"] == (
            "spk_ft.encoder.spk_model."
        )
        expected_injection = (
            "block"
            if name.endswith("multi_fuse_wesep.yaml")
            else "latent"
        )
        assert (
            cfg["model"]["args"]["cues"]["speaker"]["features"]["context"]["injection"]
            == expected_injection
        )
    if name == "tse_bsrnn_spk_dual_self_estimated_wesep.yaml":
        assert cfg["trainer"]["objectives"]["dual_self_estimated"] == {
            "enabled": True,
            "original_loss_weight": 0.4,
            "self_loss_weight": 0.6,
        }
