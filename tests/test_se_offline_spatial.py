"""SE offline spatial waveform, collate, model, and cIRM loss integration."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from core.data.collate import collate_fn
from core.losses.build import build_loss
from core.stft import build_stft_istft
from tasks.se.data import build_dataset
from tasks.se.data.batch import SEBatch
from tasks.se.models.discriminative.spatialnet import SpatialNetSE
from tasks.se.trainer.metrics_trainer import _direct_spectral_loss_kwargs, _se_loss_kwargs

pytestmark = pytest.mark.unit


def _write_pair(root: Path, *, noisy_channels: int = 2):
    sample_rate = 16000
    samples = np.arange(512, dtype=np.float32) / sample_rate
    clean = (0.2 * np.sin(2 * np.pi * 300 * samples)).astype(np.float32)
    interference = (0.1 * np.sin(2 * np.pi * 800 * samples)).astype(np.float32)
    noisy = np.stack((clean + interference, clean - interference), axis=1)
    if noisy_channels != 2:
        noisy = noisy[:, :noisy_channels]
    clean_path = root / "clean.wav"
    noisy_path = root / "noisy.wav"
    sf.write(clean_path, clean, sample_rate, subtype="FLOAT")
    sf.write(noisy_path, noisy, sample_rate, subtype="FLOAT")
    clean_scp = root / "clean.scp"
    noisy_scp = root / "noisy.scp"
    clean_scp.write_text(f"utt 16000 512 {clean_path}\n", encoding="utf-8")
    noisy_scp.write_text(f"utt 16000 512 {noisy_path}\n", encoding="utf-8")
    return clean_scp, noisy_scp


def _offline_config(clean_scp, noisy_scp, **split_overrides):
    train = {
        "clean_scp": str(clean_scp),
        "noisy_scp": str(noisy_scp),
        "spatial_enabled": True,
        "array_profile": "linear_2ch_80mm",
        "reference_channel": 1,
        "channel_order": ["mic_left", "mic_right"],
    }
    train.update(split_overrides)
    return {
        "data": {
            "mode": "offline",
            "sample_rate": 16000,
            "train": train,
        }
    }


def test_offline_spatial_scp_preserves_microphones_through_collate_and_cirm_backward(tmp_path):
    clean_scp, noisy_scp = _write_pair(tmp_path)
    dataset = build_dataset(_offline_config(clean_scp, noisy_scp), mode="train")
    item = dataset[0]
    assert item[0].shape == (1, 512)
    assert item[1].shape == (2, 512)
    assert dataset.channel_order == ("mic_left", "mic_right")
    assert dataset.resolved_microphone_array.reference_channel == 1

    batch = collate_fn([item], allow_spatial_noisy=True)
    assert isinstance(batch, SEBatch)
    assert batch.clean.shape == (1, 512)
    assert batch.noisy.shape == (1, 2, 512)

    stft_cfg = {
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "window": "hann",
        "center": True,
    }
    stft = build_stft_istft(stft_cfg)
    model = SpatialNetSE(
        n_fft=64,
        n_channel=2,
        ref_channel=1,
        num_layers=1,
        dim_hidden=16,
        dim_ffn=32,
        dim_squeeze=4,
        num_heads=4,
        kernel_size=(3, 3),
        conv_groups=(1, 1),
    )
    noisy_spectra = stft(batch.noisy.reshape(2, 512)).reshape(1, 2, 33, -1, 2)
    enhanced_spectra = model(noisy_spectra)
    enhanced = stft.inverse(enhanced_spectra, 512)
    loss = build_loss(
        {
            "type": "CompositeLoss",
            "strict": True,
            "components": [
                {
                    "type": "ComplexIdealRatioMaskLoss",
                    "weight": 1.0,
                    "args": {"reference_channel": 1},
                }
            ],
        },
        stft_cfg=stft_cfg,
    )
    loss_kwargs = _se_loss_kwargs(
        loss,
        batch,
        enhanced,
        16000,
        noisy=batch.noisy,
        clean=batch.clean,
        reference_channel=model.ref_channel,
    )
    loss_kwargs.update(
        _direct_spectral_loss_kwargs(
            loss,
            stft,
            batch.clean,
            enhanced_spectra,
            lengths=batch.lengths,
        )
    )
    value = loss(enhanced, batch.clean, batch.lengths, **loss_kwargs)
    value.backward()

    assert torch.isfinite(value)
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert loss_kwargs["noisy"].shape == batch.clean.shape
    torch.testing.assert_close(loss_kwargs["noisy"], batch.noisy[:, 1])


def test_offline_spatial_requires_channel_order_geometry_and_matching_wav_channels(tmp_path):
    clean_scp, noisy_scp = _write_pair(tmp_path, noisy_channels=1)
    with pytest.raises(ValueError, match="requires channel_order"):
        build_dataset(
            _offline_config(clean_scp, noisy_scp, channel_order=None), mode="train"
        )

    with pytest.raises(ValueError, match="expected geometry C=2"):
        build_dataset(_offline_config(clean_scp, noisy_scp), mode="train")


def test_offline_mono_path_keeps_canonical_mono_batch(tmp_path):
    clean_scp, noisy_scp = _write_pair(tmp_path, noisy_channels=1)
    dataset = build_dataset(
        {
            "data": {
                "mode": "offline",
                "sample_rate": 16000,
                "train": {"clean_scp": str(clean_scp), "noisy_scp": str(noisy_scp)},
            }
        },
        mode="train",
    )
    batch = collate_fn([dataset[0]])
    assert batch.clean.shape == batch.noisy.shape == (1, 512)
