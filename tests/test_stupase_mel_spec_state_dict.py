"""StuPASE MelSpec state_dict 合同与官方 CFM artifact 加载回归。

官方 CFM.pt 的 ``model`` state_dict 不含 mel 前端 buffer，而 ``CFM.from_pretrained``
走 fail-closed 严格匹配；``MelSpec`` 把 ``spectrogram.window`` 与 ``mel_scale.fb``
注册为 non-persistent buffer 后，二者不进入 state_dict，但仍在 forward 中生效。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchaudio

from tasks.se.models.pase_family.stupase.flow.cfm import CFM
from tasks.se.models.pase_family.stupase.flow.modules import MelSpec

_CFM_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "tasks/se/models/pase_family/stupase/open_model_checkpoint/CFM.pt"
)

_MEL_SPEC_KWARGS = dict(
    n_fft=64,
    hop_length=16,
    win_length=64,
    n_mel_channels=8,
    target_sample_rate=16000,
)
_REFERENCE_MEL_KWARGS = dict(
    sample_rate=16000,
    n_fft=64,
    win_length=64,
    hop_length=16,
    n_mels=8,
    power=1,
    center=True,
    normalized=False,
    norm=None,
)


@pytest.mark.unit
def test_mel_spec_state_dict_excludes_window_and_filterbank():
    """window/filterbank 不进入 state_dict，转发行为与原生 torchaudio 前端一致。"""
    mel_spec = MelSpec(**_MEL_SPEC_KWARGS)

    state = mel_spec.state_dict()
    assert "mel_stft.spectrogram.window" not in state
    assert "mel_stft.mel_scale.fb" not in state

    # buffer 仍实际存在：window 长度与 filterbank 形状未被移除
    assert mel_spec.mel_stft.spectrogram.window.numel() == _MEL_SPEC_KWARGS["n_fft"]
    assert mel_spec.mel_stft.mel_scale.fb.shape == (
        _MEL_SPEC_KWARGS["n_fft"] // 2 + 1,
        _MEL_SPEC_KWARGS["n_mel_channels"],
    )

    wav = torch.randn(2, 1600) * 0.1
    out = mel_spec(wav)
    n_frames = 1 + wav.shape[-1] // _MEL_SPEC_KWARGS["hop_length"]
    assert out.shape == (2, _MEL_SPEC_KWARGS["n_mel_channels"], n_frames)
    assert torch.isfinite(out).all()

    reference = torchaudio.transforms.MelSpectrogram(**_REFERENCE_MEL_KWARGS)(wav)
    reference = reference.clamp(min=1e-5).log()
    assert torch.allclose(out, reference, rtol=0.0, atol=1e-6)


@pytest.mark.external
@pytest.mark.skipif(
    not _CFM_ARTIFACT.is_file(),
    reason=f"official CFM artifact missing: {_CFM_ARTIFACT}",
)
def test_official_cfm_artifact_loads_strictly():
    """官方 CFM.pt 可被 fail-closed 严格加载，且加载后 mel 前端仍可用。"""
    model = CFM.from_pretrained(str(_CFM_ARTIFACT))

    state = model.mel_spec.state_dict()
    assert "mel_stft.spectrogram.window" not in state
    assert "mel_stft.mel_scale.fb" not in state
    assert not model.training

    with torch.no_grad():
        mel = model.mel_spec(torch.randn(1, 16000) * 0.1)
    assert mel.shape[0] == 1
    assert mel.shape[1] == model.num_channels
    assert torch.isfinite(mel).all()
