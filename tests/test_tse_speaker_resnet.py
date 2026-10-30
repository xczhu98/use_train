"""TSE ResNet34 speaker encoder 插件测试。"""

from __future__ import annotations

import pytest
import torch

from tasks.tse.models.speaker.encoder import SpeakerEncoder


pytestmark = pytest.mark.unit


def test_resnet34_sentence_and_frame_outputs_are_deterministic():
    encoder = SpeakerEncoder(
        model="ResNet34",
        freeze=False,
        fbank_args={"dither": 1.0},
    ).eval()
    wav = torch.linspace(-0.1, 0.1, 16000).unsqueeze(0)

    with torch.no_grad():
        first = encoder(wav)
        second = encoder(wav)
        frame = encoder.forward_frame_level(wav)

    assert encoder.embed_dim == 256
    assert encoder.frame_dim == 2560
    assert first.shape == (1, 256)
    assert frame.shape == (1, 2560, 13)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert torch.isfinite(frame).all()


def test_resnet34_wespeaker_state_keys_load_strictly(tmp_path):
    reference = SpeakerEncoder(model="ResNet34", freeze=False)
    state = {
        f"module.{key}": value.detach().clone()
        for key, value in reference.spk_model.state_dict().items()
    }
    state["projection.weight"] = torch.ones(1)
    checkpoint = tmp_path / "resnet34.pt"
    torch.save({"state_dict": state}, checkpoint)

    loaded = SpeakerEncoder(
        model="ResNet34",
        pretrained=str(checkpoint),
        freeze=True,
    )

    assert len(loaded.spk_model.state_dict()) == 218
    for key, value in reference.spk_model.state_dict().items():
        assert torch.equal(loaded.spk_model.state_dict()[key], value)


def test_resnet34_incomplete_checkpoint_fails_before_partial_model_is_returned(
    tmp_path,
):
    reference = SpeakerEncoder(model="ResNet34", freeze=False)
    state = dict(reference.spk_model.state_dict())
    state.pop(next(iter(state)))
    checkpoint = tmp_path / "incomplete.pt"
    torch.save(state, checkpoint)

    with pytest.raises(ValueError, match="缺失键"):
        SpeakerEncoder(model="ResNet34", pretrained=str(checkpoint))


def test_resnet34_builds_inside_tse_bsrnn_spk_contract():
    from tasks.tse.models.discriminative.tse_bsrnn_spk import TSE_BSRNN_SPK

    model = TSE_BSRNN_SPK(
        sr=16000,
        feature_dim=16,
        num_repeat=1,
        n_fft=512,
        hop_length=128,
        win_length=512,
        window="hann",
        joint_training=True,
        features={"spkemb": {"enabled": True}},
        speaker_model={
            "speaker_encoder": {
                "model": "ResNet34",
                "pretrained": None,
                "freeze": False,
                "spk_args": {
                    "feat_dim": 80,
                    "embed_dim": 256,
                    "pooling_func": "TSTP",
                },
            }
        },
    ).eval()
    generator = torch.Generator().manual_seed(34)
    batch = {
        "mix_spectra": torch.randn(1, 257, 8, 2, generator=generator),
        "enroll_wav": torch.randn(1, 16000, generator=generator) * 0.01,
        "enroll_lengths": torch.tensor([16000]),
    }

    with torch.no_grad():
        output = model(batch)["target_estimate_spectra"]

    assert output.shape == (1, 257, 8, 2)
    assert torch.isfinite(output).all()
