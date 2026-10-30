import pytest
import torch

from tasks.aec.models.discriminative.ultra_dpc_aec import UltraDPCAEC
from onnx.models.streaming_ultra_dpc import StreamingUltraDPCAEC, StreamingUltraDPCAECFlat


def _small_model(**kwargs):
    use_postnet = kwargs.pop("use_postnet", False)
    return UltraDPCAEC(
        n_fft=16,
        hop_length=8,
        embed_dim=4,
        n_dp_blocks=1,
        n_heads=2,
        encoder_layers=1,
        decoder_layers=1,
        use_postnet=use_postnet,
        **kwargs,
    ).eval()


def _run_stream(stream, spectra):
    cache = stream.init_cache(spectra.shape[0], spectra.device, spectra.dtype)
    frames = []
    for t in range(spectra.shape[2]):
        output, cache = stream(
            spectra[:, :, t : t + 1],
            spectra[:, :, t : t + 1],
            spectra[:, :, t : t + 1],
            cache,
        )
        frames.append(output)
    return torch.cat(frames, dim=2), cache


@pytest.mark.unit
def test_uncompressed_streaming_matches_offline_causal_path():
    torch.manual_seed(10)
    base = _small_model(freq_ratio=1, time_skip=1)
    stream = StreamingUltraDPCAEC(base).eval()
    spectra = torch.randn(1, 9, 5, 2)
    cache = stream.init_cache(1, "cpu", spectra.dtype)
    frames = []
    with torch.inference_mode():
        offline = base.forward_spectra(spectra, spectra, spectra)
        for t in range(spectra.shape[2]):
            output, cache = stream(
                spectra[:, :, t : t + 1],
                spectra[:, :, t : t + 1],
                spectra[:, :, t : t + 1],
                cache,
            )
            frames.append(output)
    torch.testing.assert_close(torch.cat(frames, dim=2), offline, rtol=2e-5, atol=2e-6)


@pytest.mark.unit
@pytest.mark.parametrize("freq_ratio,time_skip", [(2, 1), (1, 2), (2, 2)])
def test_compressed_streaming_has_explicit_frame_output_shape(freq_ratio, time_skip):
    base = _small_model(freq_ratio=freq_ratio, time_skip=time_skip)
    stream = StreamingUltraDPCAEC(base).eval()
    spectra = torch.randn(1, 9, 1, 2)
    cache = stream.init_cache(1, "cpu", spectra.dtype)
    with torch.inference_mode():
        output, next_cache = stream(spectra, spectra, spectra, cache)
    assert output.shape == spectra.shape
    assert set(next_cache) == set(cache)
    assert torch.isfinite(output).all()


@pytest.mark.unit
@pytest.mark.parametrize("freq_ratio", [1, 2])
def test_time_compressed_streaming_matches_offline_causal_path(freq_ratio):
    torch.manual_seed(11)
    base = _small_model(
        freq_ratio=freq_ratio,
        time_skip=2,
        use_postnet=True,
    )
    stream = StreamingUltraDPCAEC(base).eval()
    spectra = torch.randn(1, 9, 7, 2)
    with torch.inference_mode():
        offline = base.forward_spectra(spectra, spectra, spectra)
        streamed, cache = _run_stream(stream, spectra)
    torch.testing.assert_close(streamed, offline, rtol=2e-5, atol=2e-6)
    assert {"time_history", "time_feature", "time_phase", "postnet_h"} <= set(cache)
    assert cache["time_phase"].item() == spectra.shape[2] % base.time_skip


@pytest.mark.unit
def test_streaming_flat_cache_abi_is_derived_from_init_cache():
    stream = StreamingUltraDPCAEC(_small_model(freq_ratio=1, time_skip=1))
    flat = StreamingUltraDPCAECFlat(stream)
    caches = flat.init_cache_flat(2, "cpu", torch.float32)
    assert len(caches) == len(flat.get_input_names()) - 3
    nested_cache = stream.init_cache(2, "cpu", torch.float32)
    expected_shapes = [
        tuple(state.shape)
        for block_cache in nested_cache.values()
        for key in ("kv_state", "k_state", "gru_h")
        if (state := block_cache.get(key)) is not None
    ]
    assert [tuple(value.shape) for value in caches] == expected_shapes
    spec = torch.randn(2, 9, 1, 2)
    with torch.inference_mode():
        outputs = flat(spec, spec, spec, *caches)
    assert outputs[0].shape == spec.shape
    assert len(outputs) == len(caches) + 1
