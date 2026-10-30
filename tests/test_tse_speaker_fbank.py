"""Speaker frontend 的 dither 与单帧方差稳定性测试。"""

from __future__ import annotations

import pytest
import torch

from tasks.tse.models.speaker.ecapa import ASTP
from tasks.tse.models.speaker.encoder import SpeakerEncoder
from tasks.tse.models.speaker.fbank import FbankKaldi


def _astp_reference_with_unbiased_context(pool: ASTP, x: torch.Tensor) -> torch.Tensor:
    """复刻多帧 ASTP 的原始无偏 context 方差路径。"""
    context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
    context_std = torch.sqrt(
        torch.var(x, dim=-1, keepdim=True) + 1e-7
    ).expand_as(x)
    x_in = torch.cat((x, context_mean, context_std), dim=1)
    alpha = torch.tanh(pool.linear1(x_in))
    alpha = torch.softmax(pool.linear2(alpha), dim=2)
    mean = torch.sum(alpha * x, dim=2)
    var = torch.sum(alpha * (x**2), dim=2) - mean**2
    std = torch.sqrt(var.clamp(min=1e-7))
    return torch.cat([mean, std], dim=1)


@pytest.mark.unit
def test_dither_only_applies_during_training():
    """评估态应确定性，训练态仍保留 Kaldi dither 随机增强。"""
    fbank = FbankKaldi(dither=0.5)
    wav = torch.linspace(-0.2, 0.2, 1600).unsqueeze(0)

    fbank.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        eval_first = fbank(wav)
    torch.manual_seed(1)
    with torch.no_grad():
        eval_second = fbank(wav)

    assert torch.equal(eval_first, eval_second)

    fbank.train()
    torch.manual_seed(0)
    with torch.no_grad():
        train_first = fbank(wav)
    torch.manual_seed(1)
    with torch.no_grad():
        train_second = fbank(wav)

    assert not torch.equal(train_first, train_second)


@pytest.mark.unit
@pytest.mark.parametrize("num_samples", [400, 559])
def test_single_frame_speaker_encoder_is_finite_and_eval_deterministic(num_samples):
    """真实 SpeakerEncoder 对单帧输入区间两端均产生稳定有限 embedding。"""
    encoder = SpeakerEncoder(
        freeze=False,
        spk_args={"feat_dim": 8, "embed_dim": 4},
        fbank_args={"num_mel_bins": 8},
    ).eval()
    wav = torch.linspace(-0.2, 0.2, num_samples).unsqueeze(0)

    with torch.no_grad():
        first = encoder(wav)
        second = encoder(wav)

    assert first.shape == (1, 4)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


@pytest.mark.unit
def test_astp_single_frame_context_variance_is_finite_and_deterministic():
    """直接覆盖 global-context ASTP 的单帧总体方差分支。"""
    pool = ASTP(in_dim=4, bottleneck_dim=3, global_context_att=True).eval()
    x = torch.tensor([[[1.0], [-2.0], [3.0], [0.5]]])

    with torch.no_grad():
        first = pool(x)
        second = pool(x)

    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


@pytest.mark.unit
def test_astp_multiframe_keeps_unbiased_context_variance_semantics():
    """多帧 ASTP 必须保持官方 checkpoint 所依赖的原始无偏方差数值。"""
    torch.manual_seed(0)
    pool = ASTP(in_dim=4, bottleneck_dim=3, global_context_att=True).eval()
    x = torch.randn(2, 4, 5)

    with torch.no_grad():
        actual = pool(x)
        expected = _astp_reference_with_unbiased_context(pool, x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.unit
def test_fbank_norm_var_single_frame_is_finite():
    """单帧 CMVN 的总体方差应为有限零值，而不是 NaN。"""
    fbank = FbankKaldi(num_mel_bins=8, norm_var=True).eval()
    wav = torch.linspace(-0.2, 0.2, 400).unsqueeze(0)

    with torch.no_grad():
        features = fbank(wav)

    assert features.shape == (1, 1, 8)
    assert torch.isfinite(features).all()
    assert torch.equal(features, torch.zeros_like(features))


@pytest.mark.unit
def test_fbank_norm_var_multiframe_keeps_unbiased_variance_semantics():
    """多帧 CMVN 仍须逐值匹配原有 torch.var 默认无偏语义。"""
    torch.manual_seed(0)
    wav = torch.randn(2, 720) * 0.05
    fbank = FbankKaldi(num_mel_bins=8, norm_mean=False, norm_var=False).eval()

    with torch.no_grad():
        raw = fbank(wav)
        expected = raw / torch.sqrt(raw.var(dim=1, keepdim=True) + 1e-8)
        fbank.norm_var = True
        actual = fbank(wav)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
