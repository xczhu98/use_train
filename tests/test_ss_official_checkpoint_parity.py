"""External SS checkpoint parity gates.

Run explicitly after preparing public checkpoints:

    USE_TRAIN_TDANET_CHECKPOINT=/path/tdanet.bin \
    USE_TRAIN_TIGER_CHECKPOINT=/path/tiger.safetensors \
    python -m pytest -m external tests/test_ss_official_checkpoint_parity.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import soundfile as sf
import torch

from core.stft import build_stft_istft
from core.utils.safe_load import safe_torch_load
from tasks.ss.models._third_party.tdanet import TDANet as TDANetReference
from tasks.ss.models._third_party.tiger import TIGER as TIGERReference
from tasks.ss.models.tdanet import TDANet
from tasks.ss.models.tiger import TIGER


pytestmark = pytest.mark.external


def _real_voicebank_mixture() -> torch.Tensor:
    paths = sorted(Path("data/se/voicebank_demand/clean_trainset_wav").glob("*.wav"))[:2]
    if len(paths) < 2:
        pytest.skip("local VoiceBank clean wavs are unavailable")
    waveforms = []
    for path in paths:
        waveform, sample_rate = sf.read(path, dtype="float32")
        if waveform.ndim != 1 or sample_rate != 16000:
            pytest.skip("local VoiceBank parity fixture is not mono 16 kHz")
        waveforms.append(torch.from_numpy(waveform))
    length = min(16000, *(int(waveform.numel()) for waveform in waveforms))
    return torch.stack([waveform[:length] for waveform in waveforms]).sum(dim=0).unsqueeze(0)


@pytest.mark.unit
def test_official_tdanet_checkpoint_matches_reference_on_real_audio():
    checkpoint_path = os.environ.get("USE_TRAIN_TDANET_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("set USE_TRAIN_TDANET_CHECKPOINT to run official parity")
    checkpoint = safe_torch_load(
        checkpoint_path,
        purpose="run public TDANet checkpoint parity",
    )
    args = checkpoint["model_args"]
    reference = TDANetReference(**args).eval()
    adapter = TDANet(**args).eval()
    reference.load_state_dict(checkpoint["state_dict"], strict=True)
    adapter.load_state_dict(checkpoint["state_dict"], strict=True)
    mixture = _real_voicebank_mixture()
    with torch.no_grad():
        expected = reference(mixture)
        actual = adapter(mixture)["est_sources"]
    assert torch.allclose(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.unit
def test_official_tiger_checkpoint_matches_external_stft_on_real_audio():
    checkpoint_path = os.environ.get("USE_TRAIN_TIGER_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("set USE_TRAIN_TIGER_CHECKPOINT to run official parity")
    safetensors = pytest.importorskip("safetensors.torch")
    state = safetensors.load_file(checkpoint_path, device="cpu")
    args = dict(
        out_channels=128,
        in_channels=256,
        num_blocks=8,
        upsampling_depth=5,
        win=640,
        stride=160,
        num_sources=2,
        sample_rate=16000,
    )
    reference = TIGERReference(**args).eval()
    adapter = TIGER(
        **args,
        n_fft=640,
        hop_length=160,
        win_length=640,
        window="hann",
        center=True,
    ).eval()
    reference.load_state_dict(state, strict=True)
    adapter.load_state_dict(state, strict=True)
    mixture = _real_voicebank_mixture()
    stft = build_stft_istft(
        {
            "n_fft": 640,
            "hop_length": 160,
            "win_length": 640,
            "window": "hann",
            "center": True,
        }
    )
    with torch.no_grad():
        expected = reference(mixture.unsqueeze(1))[:, 0]
        spectra = adapter(stft(mixture), torch.tensor([mixture.shape[-1]]))["est_sources_spectra"]
        actual = stft.inverse(
            spectra.reshape(2, *spectra.shape[2:]), mixture.shape[-1]
        ).reshape(1, 2, -1)
    assert torch.allclose(actual, expected, atol=2e-4, rtol=2e-4)
