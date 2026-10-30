"""ClearerVoice MossFormer2 的 Kaldi dither 训练/评估模式测试。"""

from __future__ import annotations

import pytest
import torch


class _MaskNet(torch.nn.Module):
    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.feat_dim = feat_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.new_ones(features.shape[0], features.shape[-1], self.feat_dim)


def test_mossformer2_kaldi_dither_only_during_training(monkeypatch):
    import tasks.se.models.discriminative.clearervoice as clearervoice

    model = clearervoice.MossFormer2_SE_48K.__new__(clearervoice.MossFormer2_SE_48K)
    torch.nn.Module.__init__(model)
    model.sampling_rate = 48000
    model._n_fft = 4
    model._hop_length = 2
    model._win_length = 4
    model._win_type_kaldi = "hamming"
    model._kaldi_dither = 0.75
    model._feat_dim = 3
    model._num_mels = 2
    model.masknet = _MaskNet(model._feat_dim)
    model.register_buffer("_win_buf", torch.ones(model._win_length), persistent=False)

    dithers = []

    def fake_fbank(wav, **kwargs):
        dithers.append(kwargs["dither"])
        return wav.new_zeros(2, model._num_mels * 3)

    monkeypatch.setattr(clearervoice, "kaldi_fbank_plus_deltas", fake_fbank)
    monkeypatch.setattr(
        clearervoice.torch,
        "istft",
        lambda spec, **kwargs: spec.real.new_zeros(kwargs["length"]),
    )

    noisy_spec = torch.zeros(1, model._feat_dim, 3, 2)
    lengths = torch.tensor([8])

    model.train()
    model(noisy_spec, lengths)
    model.eval()
    model(noisy_spec, lengths)

    assert dithers == [pytest.approx(0.75), pytest.approx(0.0)]


pytestmark = [pytest.mark.unit]
