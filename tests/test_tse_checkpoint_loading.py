"""TSEExtractor 官方 wrapper 与 fail-closed checkpoint 回归测试。"""

from __future__ import annotations

import json

import pytest
import torch

from tasks.tse.inference.extractor import TSEExtractor, _load_model_state_fail_closed
from tasks.tse.models.speaker.encoder import SpeakerEncoder


pytestmark = pytest.mark.unit


def _small_config() -> dict:
    return {
        "meta": {"inference_only": True},
        "stft": {
            "n_fft": 512,
            "hop_length": 128,
            "win_length": 512,
            "window": "hann",
        },
        "model": {
            "type": "TSE_BSRNN_SPK",
            "args": {
                "sr": 16000,
                "feature_dim": 16,
                "num_repeat": 1,
                "causal": False,
                "joint_training": True,
                "features": {"spkemb": {"enabled": True}},
                "speaker_model": {
                    "speaker_encoder": {"pretrained": None},
                },
            },
        },
        "data": {"mode": "dynamic_tse", "sample_rate": 16000},
    }


def test_joint_official_wrapper_skips_independent_speaker_preload(
    tmp_path, monkeypatch
):
    from core.models.build import build_model
    from tasks.tse.models import register_tse_models

    cfg = _small_config()
    register_tse_models(only="TSE_BSRNN_SPK")
    model = build_model(cfg["model"], stft_cfg=cfg["stft"])
    checkpoint = tmp_path / "avg_model.pt"
    torch.save({"models": [model.state_dict()]}, checkpoint)

    external_speaker = tmp_path / "speaker.pt"
    external_speaker.write_bytes(b"must-not-be-loaded")
    cfg["model"]["args"]["speaker_model"]["speaker_encoder"]["pretrained"] = str(
        external_speaker
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    def unexpected_preload(*_args, **_kwargs):
        raise AssertionError("joint checkpoint construction must not preload speaker weights")

    monkeypatch.setattr(
        "tasks.tse.models.speaker.encoder.SpeakerEncoder._load_wespeaker_ckpt",
        unexpected_preload,
    )
    extractor = TSEExtractor(
        str(checkpoint), config_path=str(config_path), device="cpu"
    )

    assert extractor.speaker_encoder_pretrained == str(external_speaker)
    assert extractor.external_speaker_encoder_pretrained is None


def test_strict_state_validation_does_not_copy_before_shape_gate():
    model = torch.nn.Linear(3, 2)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    bad = dict(before)
    bad["weight"] = torch.zeros(3, 3)

    with pytest.raises(RuntimeError, match="tensor mismatches"):
        _load_model_state_fail_closed(
            model,
            bad,
            checkpoint_label="synthetic official checkpoint",
        )

    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value)


def test_partial_checkpoint_loading_flag_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="partial checkpoint loading"):
        TSEExtractor(str(tmp_path / "unused.pt"), allow_partial_load=True)


def test_speaker_encoder_can_strictly_select_explicit_full_model_prefix(tmp_path):
    reference = SpeakerEncoder(pretrained=None)
    prefix = "spk_ft.encoder.spk_model."
    wrapped = {
        f"{prefix}{key}": value.detach().clone()
        for key, value in reference.spk_model.state_dict().items()
    }
    checkpoint = tmp_path / "full-tse.pt"
    torch.save({"models": [wrapped]}, checkpoint)

    loaded = SpeakerEncoder(
        pretrained=str(checkpoint),
        pretrained_prefix=prefix,
        freeze=True,
    )

    for key, value in reference.spk_model.state_dict().items():
        assert torch.equal(loaded.spk_model.state_dict()[key], value)
