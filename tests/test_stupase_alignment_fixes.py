"""StuPASE 与官方实现的两处对齐回归：Mel 前端 global max、输出峰值归一化。"""

import inspect

import pytest
import torch
from torchaudio.transforms import MelSpectrogram

from tasks.se.models.pase_family.common.utils import scale_waveform_to_reference_peak
from tasks.se.models.pase_family.stupase.mel_feature_extractor import Mel_feat
from tasks.se.models.pase_family.stupase.stupase_model import StuPASE

pytestmark = [pytest.mark.unit]

_MEL_KWARGS = dict(
    sample_rate=16000,
    n_fft=64,
    win_length=64,
    hop_length=16,
    n_mels=8,
    f_min=0.0,
    f_max=None,
    power=1.0,
)


@pytest.mark.unit
def test_mel_feat_peak_normalization_uses_batch_global_max():
    """Mel_feat 必须按整个 batch 的 global max 归一，对齐官方 feature_extractor.py:35。"""
    extractor = Mel_feat(**_MEL_KWARGS)

    # 两个样本峰值分别为 0.5 与 10.0；逐样本归一化会让两者的归一化尺度不同
    first = torch.linspace(-0.5, 0.5, 128)
    second = torch.linspace(-10.0, 10.0, 128)
    batch = torch.stack([first, second])

    out = extractor(batch)

    global_peak = batch.abs().max()
    normalized = batch / (global_peak + 1e-8) * 0.9
    reference = MelSpectrogram(**_MEL_KWARGS)(normalized)
    reference = torch.log(torch.clamp(reference, min=1e-7))

    torch.testing.assert_close(out, reference, rtol=0.0, atol=1e-6)

    # 同一 batch 内的样本共用 global max：与单独送入首个样本（逐样本峰值 0.5）不同
    single = extractor(first.unsqueeze(0))
    assert not torch.allclose(out[0], single[0], atol=1e-3)


@pytest.mark.unit
def test_scale_waveform_to_reference_peak_is_per_sample_and_finite_for_zero_peak():
    """峰值缩放按样本独立对齐 reference 峰值，零峰值输入不产生 NaN。"""
    waveform = torch.stack(
        [torch.linspace(-0.5, 0.5, 64), torch.linspace(-2.0, 2.0, 64)]
    )
    reference = torch.stack(
        [torch.linspace(-0.25, 0.25, 64), torch.linspace(-3.0, 3.0, 64)]
    )

    out = scale_waveform_to_reference_peak(waveform, reference)

    expected = (
        waveform
        / (waveform.abs().amax(dim=-1, keepdim=True) + 1e-8)
        * reference.abs().amax(dim=-1, keepdim=True)
    )
    torch.testing.assert_close(out, expected, rtol=0.0, atol=1e-7)

    # 逐样本：各自峰值被缩放到对应 reference 峰值（互不串用）
    torch.testing.assert_close(
        out.abs().amax(dim=-1),
        reference.abs().amax(dim=-1),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(out[1], waveform[1] * (3.0 / 2.0), rtol=0.0, atol=1e-6)

    # 输入峰值为 0：输出全 0 且有限，不产生 NaN
    zeros = scale_waveform_to_reference_peak(torch.zeros(2, 64), reference)
    assert torch.count_nonzero(zeros) == 0
    assert torch.isfinite(zeros).all()

    # reference 峰值为 0 时同样保持有限
    silent_ref = scale_waveform_to_reference_peak(waveform, torch.zeros(2, 64))
    assert torch.count_nonzero(silent_ref) == 0
    assert torch.isfinite(silent_ref).all()


@pytest.mark.unit
def test_stupase_forward_exposes_normalize_output_keyword():
    """StuPASE.forward 暴露 keyword-only normalize_output，其余采样参数默认值不变。"""
    signature = inspect.signature(StuPASE.forward)
    parameter = signature.parameters["normalize_output"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is True
    assert signature.parameters["steps"].default == 8
    assert signature.parameters["cfg_strength"].default == 0.5
    assert signature.parameters["sway_sampling_coef"].default == -1.0
