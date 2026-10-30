"""WavLM 前端右侧补齐及 PLC 路径回归测试。"""

from __future__ import annotations

import pytest
import torch

from tasks.se.models.pase_family.common.wavlm.WavLM import ConvFeatureExtractionModel
from tasks.se.models.pase_family.common.wavlm_frontend import WavLM_feat
from tasks.se.models.pase_family.unipase.wavlm_frontend_plc import WavLM_feat_PLC


pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("length", [0, 1, 79, 80, 81, 319, 320, 321, 400, 401, 640])
def test_pad_to_multiple_only_pads_right_and_is_idempotent(length: int):
    source = torch.arange(length, dtype=torch.float32)

    padded = WavLM_feat.pad_to_multiple(source, multiple=320, offset=80)
    if length < 400:
        expected_length = 400
    else:
        expected_length = length // 320 * 320 + 80

    assert padded.shape == (1, expected_length)
    assert expected_length >= 400
    assert expected_length % 320 == 80
    kept = min(length, expected_length)
    torch.testing.assert_close(padded[0, :kept], source[:kept])
    if expected_length > length:
        assert torch.count_nonzero(padded[0, length:]) == 0
    assert WavLM_feat.pad_to_multiple(padded).data_ptr() == padded.data_ptr()


@pytest.mark.parametrize("length", [0, 1, 80, 399, 400])
def test_short_padding_runs_default_wavlm_conv_geometry(length: int):
    conv_layers = [(2, 10, 5)] + [(2, 3, 2)] * 4 + [(2, 2, 2)] * 2
    frontend = ConvFeatureExtractionModel(conv_layers=conv_layers)
    padded = WavLM_feat.pad_to_multiple(torch.ones(length))

    features = frontend(padded)

    assert features.shape == (1, 2, 1)


def test_plc_detects_packets_after_shared_padding(monkeypatch):
    calls = []

    def fake_forward(self, wav, sr=None, mask=False, mask_indices=None, lengths=None):
        calls.append((wav, sr, mask, mask_indices, lengths))
        return wav

    monkeypatch.setattr(WavLM_feat, "forward", fake_forward)
    frontend = WavLM_feat_PLC.__new__(WavLM_feat_PLC)
    torch.nn.Module.__init__(frontend)
    frontend.enable_plc = True

    result = frontend(torch.zeros(1, 81), sr=16000, mask=True)

    wav, sr, mask, mask_indices, lengths = calls.pop()
    assert result.shape == (1, 400)
    assert wav.shape == (1, 400)
    assert sr == 16000
    assert mask is True
    assert mask_indices.shape == (1, 1)
    assert mask_indices.dtype == torch.bool
    assert mask_indices.item() is True
    assert lengths is None


def test_plc_disabled_preserves_wavlm_mask_request_and_lengths(monkeypatch):
    calls = []

    def fake_forward(self, wav, sr=None, mask=False, mask_indices=None, lengths=None):
        calls.append((mask, mask_indices, lengths))
        return wav

    monkeypatch.setattr(WavLM_feat, "forward", fake_forward)
    frontend = WavLM_feat_PLC.__new__(WavLM_feat_PLC)
    torch.nn.Module.__init__(frontend)
    frontend.enable_plc = False
    lengths = torch.tensor([64])

    frontend(torch.ones(1, 81), sr=16000, mask=True, lengths=lengths)

    assert calls == [(True, None, lengths)]


def test_plc_scales_int32_48k_lengths_without_overflow(monkeypatch):
    """48 kHz/4 s 的 int32 lengths 在换算到 16k 时不得溢出。"""
    calls = []

    def fake_forward(self, wav, sr=None, mask=False, mask_indices=None, lengths=None):
        calls.append((sr, lengths))
        return wav

    monkeypatch.setattr(WavLM_feat, "forward", fake_forward)
    frontend = WavLM_feat_PLC.__new__(WavLM_feat_PLC)
    torch.nn.Module.__init__(frontend)
    frontend.enable_plc = True

    frontend(
        torch.zeros(1, 192000),
        sr=48000,
        mask=True,
        lengths=torch.tensor([192000], dtype=torch.int32),
    )

    sr, scaled = calls.pop()
    assert sr == 16000
    assert scaled.dtype == torch.int64
    assert scaled.tolist() == [64000]
