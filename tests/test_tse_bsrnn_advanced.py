"""WeSep BSRNN advanced TSE variants and training objective tests."""

from __future__ import annotations

import pytest
import torch

from tasks.tse.models.discriminative.tse_bsrnn_spk import TSE_BSRNN_SPK


_STFT = {
    "n_fft": 512,
    "hop_length": 128,
    "win_length": 512,
    "window": "hann",
    "center": True,
}

@pytest.mark.unit
def test_multi_fuse_runs_distinct_fusion_before_every_separator_block():
    model = TSE_BSRNN_SPK(
        feature_dim=16,
        num_repeat=2,
        causal=False,
        joint_training=False,
        features={"spkemb": {"enabled": True, "fusion": "multiply"}},
        multi_fuse=True,
        **_STFT,
    )
    calls = [0, 0]
    handles = []
    for index, layer in enumerate(model.multi_fuse_layers):
        handles.append(
            layer.register_forward_hook(
                lambda _module, _inputs, _output, i=index: calls.__setitem__(
                    i, calls[i] + 1
                )
            )
        )
    generator = torch.Generator().manual_seed(7)
    batch = {
        "mix_spectra": torch.randn(2, 257, 5, 2, generator=generator),
        "spk_embedding": torch.randn(2, 192, generator=generator),
    }

    output = model(batch)["target_estimate_spectra"]
    output.square().mean().backward()
    for handle in handles:
        handle.remove()

    assert calls == [1, 1]
    assert model.multi_fuse_layers[0] is not model.multi_fuse_layers[1]
    for layer in model.multi_fuse_layers:
        grads = [parameter.grad for parameter in layer.parameters()]
        assert grads and all(grad is not None for grad in grads)

@pytest.mark.unit
def test_causal_multi_fuse_streaming_matches_offline():
    model = TSE_BSRNN_SPK(
        feature_dim=16,
        num_repeat=2,
        causal=True,
        joint_training=False,
        features={"spkemb": {"enabled": True, "fusion": "multiply"}},
        multi_fuse=True,
        **_STFT,
    ).eval()
    generator = torch.Generator().manual_seed(8)
    spectra = torch.randn(1, 257, 7, 2, generator=generator)
    embedding = torch.randn(1, 192, generator=generator)

    with torch.no_grad():
        offline = model(
            {"mix_spectra": spectra, "spk_embedding": embedding}
        )["target_estimate_spectra"]
        state = model.init_streaming_state({"spk_embedding": embedding})
        first, state = model.forward_stream_chunk(
            {"mix_spectra": spectra[:, :, :3]}, state
        )
        second, _ = model.forward_stream_chunk(
            {"mix_spectra": spectra[:, :, 3:]}, state
        )

    streamed = torch.cat(
        (first["target_estimate_spectra"], second["target_estimate_spectra"]),
        dim=2,
    )
    torch.testing.assert_close(streamed, offline, rtol=1e-5, atol=2e-6)


@pytest.mark.unit
@pytest.mark.parametrize("fusion", ["concat", "additive", "multiply", "FiLM"])
def test_multi_fuse_supports_every_registered_sentence_fusion(fusion):
    model = TSE_BSRNN_SPK(
        feature_dim=16,
        num_repeat=1,
        joint_training=False,
        features={"spkemb": {"enabled": True, "fusion": fusion}},
        multi_fuse=True,
        **_STFT,
    )
    generator = torch.Generator().manual_seed(9)
    output = model(
        {
            "mix_spectra": torch.randn(1, 257, 3, 2, generator=generator),
            "spk_embedding": torch.randn(1, 192, generator=generator),
        }
    )["target_estimate_spectra"]

    assert output.shape == (1, 257, 3, 2)
