"""StuPASE CFM 采样状态与 mel 前端契约（审计 P2-3 / P2-4 / P2-5 回归）。"""

from __future__ import annotations

import pytest
import torch

from tasks.se.models.pase_family.stupase.flow import cfm as cfm_module
from tasks.se.models.pase_family.stupase.flow.cfm import CFM
from tasks.se.models.pase_family.stupase.flow.modules import MelSpec, get_vocos_mel_spectrogram

pytestmark = [pytest.mark.unit]

_DIT_KWARGS = dict(
    dim=16,
    depth=1,
    heads=2,
    dim_head=8,
    ff_mult=2,
    dropout=0.0,
    mel_dim=8,
    text_in_dim=4,
    text_dim=4,
)
_MEL_SPEC_KWARGS = dict(
    n_fft=32, hop_length=8, win_length=32, n_mel_channels=8, target_sample_rate=16000
)


def _tiny_cfm() -> CFM:
    return CFM(dit_kwargs=dict(_DIT_KWARGS), mel_spec_kwargs=dict(_MEL_SPEC_KWARGS))


def _sample_inputs(batch: int = 2, length: int = 8, channels: int = 8, text_channels: int = 4) -> dict:
    return dict(
        cond=torch.randn(batch, length, channels),
        cond_noisy=torch.randn(batch, length, channels),
        text=torch.randn(batch, length, text_channels),
    )


def test_cfm_sample_seeded_noise_differs_per_batch_item():
    """固定 seed 时同批各 utterance 的初始噪声必须不同（不再逐项重播同一 seed）。"""
    cfm = _tiny_cfm()
    _, trajectory = cfm.sample(**_sample_inputs(), duration=8, steps=1, seed=1234)

    y0 = trajectory[0]
    assert y0.shape == (2, 8, cfm.num_channels)
    assert not torch.allclose(y0[0], y0[1])
    assert torch.count_nonzero(y0[0]) > 0


def test_cfm_sample_seeded_noise_is_reproducible():
    """同一 seed 的两次采样给出相同初始噪声（可复现性不回退）。"""
    first = _tiny_cfm().sample(**_sample_inputs(), duration=8, steps=1, seed=7)[1][0]
    second = _tiny_cfm().sample(**_sample_inputs(), duration=8, steps=1, seed=7)[1][0]
    torch.testing.assert_close(first, second)


def test_cfm_sample_seed_does_not_pollute_global_rng():
    """带 seed 采样不得改写全局 RNG 状态（原实现在循环内调用 manual_seed）。"""
    cfm = _tiny_cfm()
    inputs = _sample_inputs()
    torch.manual_seed(0)
    state_before = torch.random.get_rng_state()

    cfm.sample(**inputs, duration=8, steps=1, seed=1234)

    assert torch.equal(torch.random.get_rng_state(), state_before)


def test_cfm_sample_clears_dit_text_cache_when_odeint_raises(monkeypatch):
    """采样中途异常（OOM/中断）也必须清空 DiT text 缓存，避免下一次复用陈旧嵌入。"""
    cfm = _tiny_cfm()

    def failing_odeint(fn, y0, t, **_kwargs):
        fn(t[0], y0)  # 先填充缓存，再模拟采样中途失败
        raise RuntimeError("odeint aborted")

    monkeypatch.setattr(cfm_module, "odeint", failing_odeint)
    with pytest.raises(RuntimeError, match="odeint aborted"):
        cfm.sample(**_sample_inputs(), duration=8, steps=1, seed=0)

    assert cfm.transformer.text_cond is None
    assert cfm.transformer.text_uncond is None


def test_melspec_reuses_one_filterbank_across_forwards():
    """filterbank 只构造一次（submodule buffer），不再每次 forward 重建并改模块设备。"""
    mel_spec = MelSpec(**_MEL_SPEC_KWARGS)
    transform = mel_spec.mel_stft
    wav = torch.randn(2, 128)

    first = mel_spec(wav)
    second = mel_spec(wav)

    assert mel_spec.mel_stft is transform
    assert mel_spec.mel_stft.mel_scale.fb.shape == (17, 8)
    torch.testing.assert_close(first, second)


def test_melspec_matches_standalone_log_mel_extractor():
    """重构后 MelSpec 与模块级 log-mel 辅助函数数值一致。"""
    mel_spec = MelSpec(**_MEL_SPEC_KWARGS)
    wav = torch.randn(2, 128)

    expected = get_vocos_mel_spectrogram(
        waveform=wav,
        n_fft=_MEL_SPEC_KWARGS["n_fft"],
        n_mel_channels=_MEL_SPEC_KWARGS["n_mel_channels"],
        target_sample_rate=_MEL_SPEC_KWARGS["target_sample_rate"],
        hop_length=_MEL_SPEC_KWARGS["hop_length"],
        win_length=_MEL_SPEC_KWARGS["win_length"],
    )

    torch.testing.assert_close(mel_spec(wav), expected)
