"""PASE 前端长度、归一化与样本隔离回归测试。"""

import pytest
import torch
import torchaudio
from torch import nn

from tasks.se.models.pase_family.common.wavlm_frontend import WavLM_feat
from tasks.se.models.pase_family.stupase.flow_train_wrapper import StuPASEFlowTrainModel
from tasks.se.models.pase_family.stupase.mel_feature_extractor import Mel_feat

pytestmark = [pytest.mark.unit]


def _fake_layer_reps(wav: torch.Tensor) -> torch.Tensor:
    """每个样本 4 帧、每帧 ``[wav.mean(), frame_index]``（真实 WavLM 的 (B, T, D) 形状）。"""
    return torch.stack(
        [
            torch.stack(
                [
                    torch.stack([wav[index].mean(), torch.tensor(float(frame), device=wav.device)])
                    for frame in range(4)
                ]
            )
            for index in range(wav.shape[0])
        ]
    )


class _FakeWavLM(nn.Module):
    def extract_features(self, wav, padding_mask=None, **_kwargs):
        features = _fake_layer_reps(wav)
        valid = torch.tensor([4 if not padded.any() else 2 for padded in padding_mask], device=wav.device)
        feature_padding = torch.arange(4, device=wav.device).unsqueeze(0) >= valid.unsqueeze(1)
        return ({"layer_reps": [features, features], "padding_mask": feature_padding}, None)


def _frontend() -> WavLM_feat:
    frontend = WavLM_feat.__new__(WavLM_feat)
    nn.Module.__init__(frontend)
    frontend.output_layer = 1
    frontend.wavlm = _FakeWavLM()
    return frontend


def test_wavlm_frontend_padding_and_permutation_do_not_change_valid_features():
    frontend = _frontend()
    first = torch.linspace(-1.0, 1.0, 400)
    second = torch.linspace(1.0, -1.0, 200)
    batch = torch.stack([first, torch.nn.functional.pad(second, (0, 200))])

    batched = frontend(batch, lengths=torch.tensor([400, 200]))
    reversed_batch = frontend(batch.flip(0), lengths=torch.tensor([200, 400])).flip(0)
    single = frontend(first.unsqueeze(0), lengths=torch.tensor([400]))

    torch.testing.assert_close(batched[0], single[0])
    torch.testing.assert_close(batched[0], reversed_batch[0])
    assert torch.count_nonzero(batched[1, 2:]) == 0


def test_wavlm_frontend_layer_norm_covers_valid_frames_only():
    """向量化归一化必须等价于「逐样本有效帧整体 LayerNorm + padding 帧置零」。"""
    frontend = _frontend()
    first = torch.linspace(-1.0, 1.0, 400)
    second = torch.linspace(1.0, -1.0, 200)
    batch = torch.stack([first, torch.nn.functional.pad(second, (0, 200))])

    batched = frontend(batch, lengths=torch.tensor([400, 200]))

    feats = _fake_layer_reps(batch)
    torch.testing.assert_close(
        batched[0], torch.nn.functional.layer_norm(feats[0, :4], (4, 2), eps=1e-6)
    )
    torch.testing.assert_close(
        batched[1, :2], torch.nn.functional.layer_norm(feats[1, :2], (2, 2), eps=1e-6)
    )
    assert torch.count_nonzero(batched[1, 2:]) == 0


def test_stupase_mel_peak_normalization_uses_batch_global_max():
    """官方 Mel_feat 按整个 batch 的 global max 归一化，不是逐样本。"""
    extractor = Mel_feat(n_fft=64, hop_length=16, win_length=64, n_mels=8)
    batch = torch.stack(
        [torch.linspace(-0.5, 0.5, 128), torch.linspace(-10.0, 10.0, 128)]
    )
    normalized = batch / (batch.abs().max() + 1e-8) * 0.9
    reference = torch.log(
        torch.clamp(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=64,
                win_length=64,
                hop_length=16,
                n_mels=8,
                power=1.0,
            )(normalized),
            min=1e-7,
        )
    )

    torch.testing.assert_close(extractor(batch), reference, rtol=0.0, atol=1e-6)


class _CaptureEncoder(nn.Module):
    def forward(self, wav, **kwargs):
        self.kwargs = kwargs
        return torch.zeros(wav.shape[0], 2, 4)


class _CaptureCFM(nn.Module):
    class _MelSpec:
        target_sample_rate = 16000
        hop_length = 64

    mel_spec = _MelSpec()

    def forward(self, *_args, lens=None):
        self.lens = lens
        return torch.tensor(0.0), None, None


def test_stupase_flow_maps_waveform_lengths_to_mel_lengths():
    model = StuPASEFlowTrainModel.__new__(StuPASEFlowTrainModel)
    nn.Module.__init__(model)
    model.encoder = _CaptureEncoder()
    model.cfm = _CaptureCFM()
    model._sample_rate = 16000

    model(torch.zeros(2, 256), torch.zeros(2, 256), lengths=torch.tensor([128, 256]))

    assert torch.equal(model.cfm.lens, torch.tensor([3, 5]))
    assert torch.equal(model.encoder.kwargs["lengths"], torch.tensor([128, 256]))


def test_wavlm_frontend_batch_norm_matches_official_layer_norm():
    """PASE/StuPASE 口径：归一化统计量包含 batch 维。"""
    frontend = WavLM_feat.__new__(WavLM_feat)
    nn.Module.__init__(frontend)
    frontend.feature_norm = "batch"
    feat = torch.randn(2, 5, 4)
    out = frontend.normalize_features(feat, None)
    expected = torch.nn.functional.layer_norm(feat, feat.shape, eps=1e-6)
    torch.testing.assert_close(out, expected)
