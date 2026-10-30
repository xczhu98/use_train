"""TSE waveform streaming、flat ABI、TorchScript 与 timeline 回归测试。"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

from onnx.models.streaming_tse import StreamingTSEFlat
from onnx.runtime_spec import runtime_spec_from_wrapper
from tools.verify import plot_tse_timing_timeline
from tasks.tse.inference.streaming import CenteredStreamingSTFTOLA
from tasks.tse.inference.torchscript import export_streaming_tse_torchscript
from tasks.tse.models.discriminative.tse_bsrnn_spk import TSE_BSRNN_SPK


pytestmark = pytest.mark.unit

_STFT = {
    "n_fft": 512,
    "hop_length": 128,
    "win_length": 512,
    "window": "hann",
}


def _flat_model() -> StreamingTSEFlat:
    model = TSE_BSRNN_SPK(
        sr=16000,
        feature_dim=16,
        num_repeat=1,
        causal=True,
        joint_training=False,
        features={"spkemb": {"enabled": True, "fusion": "multiply"}},
        **_STFT,
    ).eval()
    return StreamingTSEFlat(model, enrollment_samples=1024).eval()


def test_centered_streaming_stft_ola_identity_matches_waveform():
    frontend = CenteredStreamingSTFTOLA(
        lambda frame: frame,
        n_fft=512,
        hop_length=128,
        win_length=512,
    )
    rng = np.random.default_rng(7)
    waveform = rng.normal(0.0, 0.03, 4097).astype(np.float32)
    outputs = []
    for start in range(0, waveform.size, 333):
        value = frontend.push(waveform[start : start + 333])
        if value.size:
            outputs.append(value)
    outputs.append(frontend.flush())
    actual = np.concatenate(outputs)

    np.testing.assert_allclose(actual, waveform, rtol=0.0, atol=3e-7)
    assert frontend.emitted_samples == waveform.size


def test_streaming_tse_flat_runtime_spec_binds_cache_order_and_delay():
    flat = _flat_model()
    spec = runtime_spec_from_wrapper(flat, "tse")

    assert spec.tse is not None
    assert spec.input_names == tuple(flat.get_input_names())
    assert spec.output_names == tuple(flat.get_output_names())
    assert spec.tse.condition_names == ("spkemb",)
    assert spec.tse.cache_names == flat.CACHE_KEYS
    assert spec.tse.waveform_output_delay_samples == flat.waveform_output_delay_samples


def test_torchscript_flat_runtime_roundtrip_and_sidecar(tmp_path):
    flat = _flat_model()
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "streaming_tse.pt"
    config.write_text("model: synthetic\n", encoding="utf-8")
    checkpoint.write_bytes(b"synthetic-checkpoint-identity")

    metadata = export_streaming_tse_torchscript(
        flat,
        output,
        config_path=config,
        checkpoint_path=checkpoint,
    )

    sidecar = output.with_name(f"{output.name}.meta.json")
    assert output.is_file() and sidecar.is_file()
    assert json.loads(sidecar.read_text(encoding="utf-8")) == metadata
    loaded = torch.jit.load(str(output), map_location="cpu").eval()
    frame = torch.randn(1, 257, 1, 2)
    cache = flat.init_cache_flat(1, "cpu", torch.float32)
    with torch.inference_mode():
        expected = flat(frame, *cache)
        actual = loaded(frame, *cache)
    for reference, candidate in zip(expected, actual):
        torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=2e-6)
    assert metadata["schema"] == "tse-streaming-torchscript-v1"
    assert metadata["runtime_kind"] == "positional-flat-cache"
    assert metadata["trace_batch_size"] == 1
    assert metadata["tse"]["cache_semantics"] == "fixed-enrollment-loopback-v1"


def test_timeline_cli_reaches_input_validation_after_backend_imports(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_tse_timing_timeline.py",
            "--eval-output-dir",
            str(tmp_path / "missing-eval"),
            "--timing-gt-json-base-dir",
            str(tmp_path / "missing-gt"),
            "--output-dir",
            str(tmp_path / "plots"),
        ],
    )
    with pytest.raises(FileNotFoundError):
        plot_tse_timing_timeline.main()


def test_embedding_tfmap_causal_is_rejected_at_base_model_boundary():
    with pytest.raises(ValueError, match="mixture-waveform chunk contract"):
        TSE_BSRNN_SPK(
            feature_dim=16,
            num_repeat=1,
            causal=True,
            joint_training=True,
            features={
                "tfmap": {"enabled": True, "type": "embedding"},
                "context": {"enabled": True},
            },
            **_STFT,
        )
