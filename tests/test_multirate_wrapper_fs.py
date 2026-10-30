"""多采样率训练：PASE-family 固定输出率 wrapper 的 fs 接入回归。

覆盖三部分：
- ``resample_waveform_and_lengths`` 的重采样 shape 与 lengths 时间基换算；
- ``PASEGANTrainer._forward_generator`` 只在 generator 声明 ``fs`` 时传入
  batch 实际采样率，旧 generator 保持原调用；
- 四个固定输出率 wrapper 的 forward 暴露 keyword-only ``fs``（不构造大模型）。
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from tasks.se.models.pase_family.common.utils import resample_waveform_and_lengths
from tasks.se.models.pase_family.gap_urgenet.gan_wrappers import GAPURGENetPostNetGenerator
from tasks.se.models.pase_family.stupase.mel_gan_wrapper import StuPASEMelVocoderGenerator
from tasks.se.models.pase_family.unipase.gan_wrappers import (
    UniPASEPostNetGenerator,
    UniPASEVocoderGenerator,
)
from tasks.se.trainer.pase_gan_trainer import PASEGANTrainer, _forward_accepts_fs

pytestmark = [pytest.mark.unit]


# ----------------------------------------------------------------------
# resample_waveform_and_lengths
# ----------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    "orig_sr, target_sr, in_samples, out_samples",
    [(16000, 48000, 1600, 4800), (48000, 16000, 4800, 1600)],
)
def test_resample_waveform_and_lengths_scales_shape_and_lengths(
    orig_sr, target_sr, in_samples, out_samples
):
    waveform = torch.randn(2, in_samples)
    lengths = torch.tensor([in_samples, in_samples // 2])

    out, out_lengths = resample_waveform_and_lengths(waveform, lengths, orig_sr, target_sr)

    assert out.shape == (2, out_samples)
    assert out_lengths.dtype == lengths.dtype
    # lengths 按采样率比例换算到新时间基
    expected = (lengths.to(torch.float64) * (target_sr / orig_sr)).round().long()
    torch.testing.assert_close(out_lengths, expected, rtol=0, atol=0)


@pytest.mark.unit
def test_resample_waveform_and_lengths_is_noop_for_equal_rates_and_none_lengths():
    waveform = torch.randn(2, 640)
    lengths = torch.tensor([640, 512])

    out, out_lengths = resample_waveform_and_lengths(waveform, lengths, 16000, 16000)

    assert out is waveform
    assert out_lengths is lengths
    _, none_lengths = resample_waveform_and_lengths(waveform, None, 16000, 48000)
    assert none_lengths is None


@pytest.mark.unit
@pytest.mark.parametrize("orig_sr, target_sr", [(0, 16000), (16000, -1), (16000.0, 48000)])
def test_resample_waveform_and_lengths_rejects_non_positive_integer_rates(orig_sr, target_sr):
    with pytest.raises(ValueError):
        resample_waveform_and_lengths(torch.zeros(1, 16), None, orig_sr, target_sr)


# ----------------------------------------------------------------------
# PASEGANTrainer._forward_generator 的 fs 传递
# ----------------------------------------------------------------------
class _RecordingGenerator(nn.Module):
    """记录 forward 收到的 fs，用于断言 trainer 的传参行为。"""

    def __init__(self) -> None:
        super().__init__()
        self.seen_fs: list[int | None] = []

    def forward(self, wav: torch.Tensor, lengths: torch.Tensor = None, *, fs=None):
        self.seen_fs.append(fs)
        return wav


class _LegacyGenerator(nn.Module):
    """未声明 fs 的旧 generator：trainer 必须保持原调用。"""

    def forward(self, wav: torch.Tensor, lengths: torch.Tensor = None):
        return wav


def _make_forward_trainer(generator: nn.Module) -> PASEGANTrainer:
    """绕过重量级 __init__，只装配 _forward_generator 依赖的属性。"""
    trainer = object.__new__(PASEGANTrainer)
    trainer.generator = generator
    trainer._generator_accepts_fs = _forward_accepts_fs(generator)
    trainer._generator_output_sample_rate = getattr(
        generator, "output_sample_rate", None
    )
    trainer.sample_rate = 16000
    return trainer


class _FixedRateGenerator(nn.Module):
    """模拟固定 16k 输出率 wrapper：输入按 fs 重采样，输出固定 16k。"""

    output_sample_rate = 16000

    def forward(self, wav, lengths=None, *, fs=None):
        wav, _ = resample_waveform_and_lengths(
            wav, None, 16000 if fs is None else fs, 16000
        )
        return wav


def _batch_dict(fs: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "noisy": torch.randn(2, 320),
        "clean": torch.randn(2, 320),
        "lengths": torch.tensor([320, 256]),
        "fs": fs,
    }


def test_forward_generator_passes_batch_fs_when_generator_accepts_it():
    generator = _RecordingGenerator()
    trainer = _make_forward_trainer(generator)

    out = trainer._forward_generator(_batch_dict(torch.tensor([48000, 48000])))

    assert generator.seen_fs == [48000]
    assert "enhanced" in out


def test_forward_generator_keeps_legacy_call_when_generator_lacks_fs():
    trainer = _make_forward_trainer(_LegacyGenerator())

    out = trainer._forward_generator(_batch_dict(torch.tensor([48000, 48000])))

    assert "enhanced" in out


def test_forward_generator_resamples_clean_to_fixed_output_rate():
    """固定输出率 generator 的 clean/lengths/fs 必须对齐到输出率。"""
    trainer = _make_forward_trainer(_FixedRateGenerator())
    batch = _batch_dict(torch.tensor([48000, 48000]))
    batch["noisy"] = torch.randn(2, 4800)
    batch["clean"] = torch.randn(2, 4800)
    batch["lengths"] = torch.tensor([4800, 2400])

    out = trainer._forward_generator(batch)

    assert out["enhanced"].shape[-1] == 1600
    assert out["clean"].shape[-1] == 1600
    assert out["fs"] == 16000
    torch.testing.assert_close(
        out["lengths"], torch.tensor([1600, 800]), rtol=0, atol=0
    )


def test_forward_accepts_fs_unwraps_compile_wrapper():
    """torch.compile 的 forward 是 (*args, **kwargs)，必须解包 _orig_mod 后探测。"""
    inner = _RecordingGenerator()
    wrapper = nn.Module()
    wrapper._orig_mod = inner

    assert _forward_accepts_fs(inner) is True
    assert _forward_accepts_fs(wrapper) is True
    assert _forward_accepts_fs(_LegacyGenerator()) is False


# ----------------------------------------------------------------------
# 四个 wrapper 的 forward 签名
# ----------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    "wrapper_cls",
    [
        StuPASEMelVocoderGenerator,
        UniPASEVocoderGenerator,
        UniPASEPostNetGenerator,
        GAPURGENetPostNetGenerator,
    ],
)
def test_fixed_rate_wrapper_forward_exposes_keyword_only_fs(wrapper_cls):
    parameter = inspect.signature(wrapper_cls.forward).parameters.get("fs")

    assert parameter is not None, f"{wrapper_cls.__name__}.forward 缺少 fs 参数"
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
