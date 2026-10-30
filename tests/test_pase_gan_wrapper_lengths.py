"""PASE GAN wrapper 必须将 batch 有效长度传给 WavLM 前端。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tasks.se.models.pase_family.gap_urgenet.gan_wrappers import GAPURGENetPostNetGenerator
from tasks.se.models.pase_family.unipase.gan_wrappers import (
    UniPASEAdapterGenerator,
    UniPASEPostNetGenerator,
    UniPASEVocoderGenerator,
)


class _RecordingEncoder(nn.Module):
    """记录 wrapper 传入的 lengths，并提供最小可执行特征。"""

    def __init__(self, return_pair: bool) -> None:
        super().__init__()
        self.return_pair = return_pair
        self.lengths_calls = []

    def forward(self, wav: torch.Tensor, **kwargs):
        self.lengths_calls.append(kwargs.get("lengths"))
        features = wav.new_zeros(wav.shape[0], 4, 2)
        return (features, features) if self.return_pair else features


class _IdentityAdapter(nn.Module):
    def forward(self, acoustic: torch.Tensor, phonetic: torch.Tensor) -> torch.Tensor:
        return acoustic


class _WaveDecoder(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, 0]


class _RecordingPostNet(nn.Module):
    """记录 wrapper 传入的有效长度，并原样返回波形。"""

    def __init__(self) -> None:
        super().__init__()
        self.lengths_calls = []

    def forward(self, wav: torch.Tensor, sr_in: int, lengths=None) -> torch.Tensor:
        self.lengths_calls.append(lengths)
        return wav if wav.ndim == 2 else wav.mean(dim=1)


class _RecordingPredictor(nn.Module):
    """记录 wrapper 传入的有效长度，并原样返回波形。"""

    def __init__(self) -> None:
        super().__init__()
        self.lengths_calls = []

    def forward(self, wav: torch.Tensor, sr: int, lengths=None) -> torch.Tensor:
        self.lengths_calls.append(lengths)
        return wav


def _new_wrapper(wrapper_type: type[nn.Module]) -> nn.Module:
    """不加载 checkpoint，构造仅验证参数透传的 wrapper。"""
    wrapper = wrapper_type.__new__(wrapper_type)
    nn.Module.__init__(wrapper)
    return wrapper


@pytest.mark.unit
def test_unipase_vocoder_wrapper_forwards_lengths_to_encoder():
    generator = _new_wrapper(UniPASEVocoderGenerator)
    generator.sr_in = 16000
    generator.encoder = _RecordingEncoder(return_pair=False)
    generator.vocoder = _WaveDecoder()
    lengths = torch.tensor([9, 16])

    generator(torch.randn(2, 16), lengths=lengths)

    assert len(generator.encoder.lengths_calls) == 1
    assert generator.encoder.lengths_calls[0] is lengths


@pytest.mark.unit
def test_unipase_adapter_wrapper_forwards_lengths_to_both_encoder_paths():
    generator = _new_wrapper(UniPASEAdapterGenerator)
    generator.enable_plc = True
    generator.encoder = _RecordingEncoder(return_pair=True)
    generator.adapter = _IdentityAdapter()
    lengths = torch.tensor([9, 16])

    generator(torch.randn(2, 16), clean_wav=torch.randn(2, 16), lengths=lengths)

    assert len(generator.encoder.lengths_calls) == 2
    assert all(call is lengths for call in generator.encoder.lengths_calls)


@pytest.mark.unit
def test_unipase_postnet_wrapper_forwards_lengths_to_encoder():
    generator = _new_wrapper(UniPASEPostNetGenerator)
    generator.sr_in = 48000
    generator.enable_plc = True
    generator.encoder = _RecordingEncoder(return_pair=True)
    generator.adapter = _IdentityAdapter()
    generator.decoder = _WaveDecoder()
    generator.postnet = _RecordingPostNet()
    lengths = torch.tensor([27, 48])

    generator(torch.randn(2, 48), lengths=lengths)

    assert len(generator.encoder.lengths_calls) == 1
    assert generator.encoder.lengths_calls[0] is lengths
    # 有效长度还必须透传到 PostNet，避免 padding 参与归一化统计
    assert len(generator.postnet.lengths_calls) == 1
    assert generator.postnet.lengths_calls[0] is not None


@pytest.mark.unit
def test_gap_postnet_wrapper_forwards_lengths_to_encoder():
    generator = _new_wrapper(GAPURGENetPostNetGenerator)
    generator.sr_in = 48000
    generator.enable_plc = True
    generator.encoder = _RecordingEncoder(return_pair=True)
    generator.adapter = _IdentityAdapter()
    generator.decoder = _WaveDecoder()
    generator.predictor = _RecordingPredictor()
    generator.postnet = _RecordingPostNet()
    lengths = torch.tensor([27, 48])

    generator(torch.randn(2, 48), lengths=lengths)

    assert len(generator.encoder.lengths_calls) == 1
    assert generator.encoder.lengths_calls[0] is lengths
    # Predictor 与 PostNet 都必须收到换算后的有效长度
    assert len(generator.predictor.lengths_calls) == 1
    assert generator.predictor.lengths_calls[0] is not None
    assert len(generator.postnet.lengths_calls) == 1
    assert generator.postnet.lengths_calls[0] is not None
