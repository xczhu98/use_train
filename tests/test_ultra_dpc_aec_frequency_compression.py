import math

import pytest
import torch

from tasks.aec.models.discriminative.cagcrn import CATA
from tasks.aec.models.discriminative.ultra_dpc_aec import (
    FreqCompression,
    FreqDecompression,
    PostProcessingNet,
    TimeCompression,
    TimeDecompression,
    UltraDPCAEC,
    _mel_band_boundaries,
)
from tasks.aec.models import register_aec_models
from core.models.build import build_model


@pytest.mark.unit
@pytest.mark.parametrize("n_fft,n_bands", [(16, 4), (320, 80), (512, 128)])
def test_mel_bands_are_nonempty_and_official_overlap(n_fft, n_bands):
    low, high = _mel_band_boundaries(n_fft, n_bands)

    assert low[0] == 0
    assert high[-1] == n_fft // 2 + 1
    assert all(lo < hi for lo, hi in zip(low, high))
    assert any(left > right for left, right in zip(high[:-1], low[1:]))
    assert low[1] < high[-2]


@pytest.mark.unit
@pytest.mark.parametrize(
    "n_fft,n_bands,sr",
    [(1, 1, 16000), (16, 0, 16000), (16, 10, 16000), (16, 5, 0)],
)
def test_mel_bands_reject_invalid_partition_parameters(n_fft, n_bands, sr):
    with pytest.raises(ValueError):
        _mel_band_boundaries(n_fft, n_bands, sr)


@pytest.mark.unit
def test_freq_compression_and_decompression_layouts():
    n_fft, n_bands, embed_dim = 16, 4, 4
    x = torch.randn(2, 2, 3, n_fft // 2 + 1)
    compressed = FreqCompression(n_fft, n_bands, in_channels=2, embed_dim=embed_dim)(x)
    restored = FreqDecompression(n_fft, n_bands, out_channels=3, embed_dim=embed_dim)(compressed)

    assert compressed.shape == (2, embed_dim, 3, n_bands)
    assert restored.shape == (2, 3, 3, n_fft // 2 + 1)


@pytest.mark.unit
def test_ultra_dpc_frequency_compression_uses_dict_spectrum_interface():
    model = UltraDPCAEC(
        n_fft=16,
        hop_length=8,
        embed_dim=4,
        n_dp_blocks=1,
        n_heads=2,
        encoder_layers=1,
        decoder_layers=1,
        freq_ratio=2,
    ).eval()
    spectra = torch.randn(1, 9, 3, 2)

    with torch.inference_mode():
        output = model(
            {
                "mic_spectra": spectra,
                "farend_spectra": spectra,
                "error_spectra": spectra,
            }
        )["nearend_estimate_spectra"]

    assert output.shape == spectra.shape
    assert torch.isfinite(output).all()


@pytest.mark.unit
def test_cata_scales_mic_attention_by_query_channel_dimension(monkeypatch):
    cata = CATA(in_channels=4, hidden_channels=4).eval()
    x_mic = torch.randn(1, 4, 2, 5)
    x_ref = torch.randn_like(x_mic)
    captured = []
    original_softmax = torch.nn.functional.softmax

    def capture_softmax(input, *args, **kwargs):
        captured.append(input)
        return original_softmax(input, *args, **kwargs)

    monkeypatch.setattr("tasks.aec.models.discriminative.cagcrn.F.softmax", capture_softmax)
    with torch.inference_mode():
        cata(x_mic, x_ref)

    q_mic = cata.pconv_mic(x_mic)
    k_mic = cata.pconv_mic2(q_mic)
    raw_scores = torch.matmul(q_mic.permute(0, 2, 3, 1), k_mic.permute(0, 2, 1, 3))
    torch.testing.assert_close(captured[0], raw_scores / math.sqrt(q_mic.shape[1]))


@pytest.mark.unit
def test_freq_decompression_overlap_add_and_tanh():
    module = FreqDecompression(16, 4, out_channels=1, embed_dim=1)
    for layer in module.proj:
        layer.weight.data.zero_()
        layer.bias.data.fill_(2.0)
    x = torch.ones(1, 1, 2, 4)
    y = module(x)
    low, high = _mel_band_boundaries(16, 4)
    assert torch.all(y[..., low[1]:high[-2]] < 1.0)
    assert torch.all(y[..., low[1]:high[-2]] > 0.0)
    assert torch.all(y.abs() <= 1.0)


@pytest.mark.unit
def test_time_compression_left_padding_and_official_decompression_shape():
    comp = TimeCompression(channels=2, skip=3)
    decomp = TimeDecompression(channels=2, skip=3)
    x = torch.arange(2 * 2 * 4 * 1, dtype=torch.float32).reshape(2, 2, 4, 1)
    compressed, pad = comp(x)
    restored = decomp(compressed, pad)
    assert pad == 2
    assert compressed.shape == (2, 2, 2, 1)
    assert restored.shape == x.shape

    # forward 无状态：其它形状的调用不得影响下一次的 pad 与结果。
    comp(torch.zeros(1, 2, 7, 1))
    compressed_again, pad_again = comp(x)
    assert pad_again == 2
    assert torch.equal(compressed_again, compressed)


@pytest.mark.unit
@pytest.mark.parametrize("freq_ratio,time_skip", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_ultra_dpc_forward_shapes_for_all_compression_modes(freq_ratio, time_skip):
    model = UltraDPCAEC(
        n_fft=16, hop_length=8, embed_dim=4, n_dp_blocks=1, n_heads=2,
        encoder_layers=1, decoder_layers=1, freq_ratio=freq_ratio,
        time_skip=time_skip,
    ).eval()
    spectra = torch.randn(1, 9, 5, 2)
    with torch.inference_mode():
        output = model.forward_spectra(spectra, spectra, spectra)
    assert output.shape == spectra.shape
    assert torch.isfinite(output).all()


@pytest.mark.unit
def test_postnet_is_active_in_eval_and_returns_real_residual_mask():
    postnet = PostProcessingNet(n_fft=16, freq_ratio=2)
    postnet.eval()
    x = torch.randn(1, 1, 5, 9)
    with torch.inference_mode():
        mask = postnet(x, x)
    assert mask.shape == (1, 1, 5, 9)
    assert torch.all((mask >= 0) & (mask <= 1))


@pytest.mark.unit
def test_postnet_stateful_frames_match_full_sequence():
    torch.manual_seed(12)
    postnet = PostProcessingNet(n_fft=16, freq_ratio=2).eval()
    x = torch.randn(1, 1, 5, 9)
    hidden = None
    frames = []
    with torch.inference_mode():
        full = postnet(x, x)
        for frame in range(x.shape[2]):
            mask, hidden = postnet.forward_with_state(
                x[:, :, frame : frame + 1],
                x[:, :, frame : frame + 1],
                hidden,
            )
            frames.append(mask)
    torch.testing.assert_close(torch.cat(frames, dim=2), full)


@pytest.mark.unit
def test_ultra_dpc_registry_build_consumes_stft_context_and_rejects_unknown_args():
    register_aec_models(only="UltraDPCAEC")
    model = build_model(
        {
            "type": "UltraDPCAEC",
            "args": {
                "embed_dim": 4,
                "n_dp_blocks": 1,
                "n_heads": 2,
                "encoder_layers": 1,
                "decoder_layers": 1,
            },
        },
        stft_cfg={
            "n_fft": 16,
            "hop_length": 8,
            "win_length": 16,
            "window": "hann",
            "center": True,
        },
    )
    assert (model.n_fft, model.hop_length, model.win_length) == (16, 8, 16)
    assert model.window == "hann" and model.center is True
    with pytest.raises(TypeError, match="unexpected keyword arguments.*typo"):
        UltraDPCAEC(typo=1)
