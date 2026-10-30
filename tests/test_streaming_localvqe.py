"""StreamingLocalVQE 的逐帧数值与 ONNX ABI 合同。"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tasks.aec.models.discriminative.localvqe import LocalVQE
from onnx.models.streaming_localvqe import StreamingLocalVQE, StreamingLocalVQEFlat

pytestmark = pytest.mark.unit


def _small_model(*, arch_version: int = 3, align_delay: int = 3) -> LocalVQE:
    """保持 F=257，仅缩小通道与 S4D state 以控制单测开销。"""
    torch.manual_seed(7)
    return LocalVQE(
        n_fft=512,
        align_delay=align_delay,
        mic_channels=[2, 2, 2, 2, 2, 2],
        far_channels=[2, 2, 2],
        align_hidden=2,
        bottleneck_hidden=3,
        arch_version=arch_version,
    ).eval()


@pytest.mark.parametrize("arch_version", [2, 3])
def test_streaming_matches_offline_multiframe(arch_version):
    base = _small_model(arch_version=arch_version)
    streaming = StreamingLocalVQE(base).eval()
    mic = torch.randn(1, 257, 5, 2)
    far = torch.randn_like(mic)

    with torch.inference_mode():
        offline = base.forward_spectra(mic, far)
        cache = streaming.init_cache(1, torch.device("cpu"), torch.float32)
        frames = []
        for frame in range(mic.shape[2]):
            enhanced, cache = streaming(
                mic[:, :, frame : frame + 1],
                far[:, :, frame : frame + 1],
                cache,
            )
            frames.append(enhanced)
    actual = torch.cat(frames, dim=2)
    torch.testing.assert_close(actual, offline, rtol=2e-5, atol=5e-6)


def test_cache_reset_and_update():
    streaming = StreamingLocalVQE(_small_model()).eval()
    initial = streaming.init_cache(2, torch.device("cpu"), torch.float32)
    assert all(torch.count_nonzero(value) == 0 for value in initial.values())

    mic = torch.randn(2, 257, 1, 2)
    far = torch.randn_like(mic)
    with torch.inference_mode():
        first, updated = streaming(mic, far, initial)
        reset = streaming.init_cache(2, torch.device("cpu"), torch.float32)
        repeated, repeated_cache = streaming(mic, far, reset)

    torch.testing.assert_close(first, repeated)
    assert any(torch.count_nonzero(value) > 0 for value in updated.values())
    for name in streaming.CACHE_KEYS:
        torch.testing.assert_close(updated[name], repeated_cache[name])


def test_default_cache_abi_has_30_structure_derived_entries():
    streaming = StreamingLocalVQE(_small_model(align_delay=4))
    cache = streaming.init_cache(3, torch.device("cpu"), torch.float32)

    expected_keys = (
        "mic_enc1_conv", "mic_enc1_res", "mic_enc2_conv", "mic_enc2_res",
        "mic_enc3_conv", "mic_enc3_res", "mic_enc4_conv", "mic_enc4_res",
        "mic_enc5_conv", "mic_enc5_res", "far_enc1_conv", "far_enc1_res",
        "far_enc2_conv", "far_enc2_res", "align_far", "align_v", "align_k",
        "s4d_real", "s4d_imag", "dec5_res", "dec5_deconv", "dec4_res",
        "dec4_deconv", "dec3_res", "dec3_deconv", "dec2_res", "dec2_deconv",
        "dec1_res", "dec1_deconv", "ccm_mic",
    )
    assert streaming.CACHE_KEYS == expected_keys
    assert tuple(cache) == expected_keys
    assert all(value.shape[0] == 3 for value in cache.values())
    assert cache["mic_enc1_conv"].shape == (3, 2, 3, 257)
    assert cache["mic_enc1_res"].shape == (3, 2, 3, 129)
    assert cache["align_far"].shape == (3, 2, 3, 65)
    assert cache["align_v"].shape == (3, 2, 4, 4)
    assert cache["align_k"].shape == (3, 2, 3, 65)
    assert cache["s4d_real"].shape == (3, 3)
    assert cache["s4d_imag"].dtype == torch.float32
    assert cache["dec1_deconv"].shape == (3, 2, 3, 129)
    assert cache["ccm_mic"].shape == (3, 2, 2, 257)


def test_flat_names_axes_and_metadata_shape_source():
    flat = StreamingLocalVQEFlat(StreamingLocalVQE(_small_model()))
    inputs = flat.get_input_names()
    outputs = flat.get_output_names()
    axes = flat.get_dynamic_axes()
    config = flat.get_cache_config()

    assert inputs[:2] == ["mic_spec", "far_spec"]
    assert outputs[0] == "enh_spec"
    assert outputs[1:] == [f"out_{name}" for name in inputs[2:]]
    assert len(inputs) == 32
    assert len(outputs) == 31
    assert set(axes) == set(inputs + outputs)
    assert all(value == {0: "batch_size"} for value in axes.values())
    assert list(config) == inputs[2:]
    assert config["s4d_real"] == [1, 3]
    assert flat.center is True
    assert flat.target_fs == 16000


def test_offline_forward_validates_far_before_feature_extraction():
    base = _small_model()
    valid = torch.zeros(1, 257, 1, 2)

    with pytest.raises(ValueError, match="far_spec"):
        base.forward_spectra(valid, torch.zeros(1, 257, 1))
    with pytest.raises(ValueError, match="exactly match"):
        base.forward_spectra(valid.expand(2, -1, -1, -1), valid)


def test_streaming_rejects_invalid_inputs_and_dtype():
    streaming = StreamingLocalVQE(_small_model())
    cache = streaming.init_cache(1, torch.device("cpu"), torch.float32)
    valid = torch.zeros(1, 257, 1, 2)

    with pytest.raises(ValueError, match="mic_spec"):
        streaming(valid[:, :, 0], valid, cache)
    with pytest.raises(ValueError, match="far_spec"):
        streaming(valid, torch.zeros(1, 257, 2, 2), cache)
    with pytest.raises(ValueError, match="exactly match"):
        streaming(valid.expand(2, -1, -1, -1), valid, cache)
    with pytest.raises(TypeError, match="FP32"):
        streaming(valid.double(), valid.double(), cache)
    with pytest.raises(ValueError, match="torch.float32"):
        streaming.init_cache(1, torch.device("cpu"), torch.float16)

    with pytest.raises(ValueError, match="F=257"):
        StreamingLocalVQE(LocalVQE(n_fft=320, arch_version=3))


@pytest.mark.subprocess
def test_direct_export_requires_config():
    result = subprocess.run(
        [sys.executable, "onnx/export.py", "--model", "streaming_localvqe"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "streaming_localvqe requires --config" in result.stderr


_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None


@pytest.mark.ort
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_ORT, reason="onnxruntime is required for ONNX/ORT smoke")
def test_onnx_ort_multiframe_cache_parity(tmp_path):
    from onnx.export import export_model
    from onnx.runtime_spec import runtime_spec_from_wrapper
    from onnx.utils import create_ort_session, write_onnx_metadata

    flat = StreamingLocalVQEFlat(StreamingLocalVQE(_small_model())).eval()
    spec = runtime_spec_from_wrapper(flat, "localvqe")
    output = tmp_path / "streaming_localvqe.onnx"

    try:
        exported_wrapper, _ = export_model(flat, str(output), opset=17)
        assert exported_wrapper is flat
        write_onnx_metadata(output, dict(spec.metadata))
        session = create_ort_session(output)
    except RuntimeError as exc:
        pytest.skip(f"external onnx/ORT export dependencies unavailable: {exc}")

    for batch_size in (1, 2):
        torch_cache = flat.init_cache_flat(batch_size, torch.device("cpu"), torch.float32)
        ort_cache = [value.numpy() for value in torch_cache]
        for _ in range(3):
            mic = torch.randn(batch_size, 257, 1, 2)
            far = torch.randn_like(mic)
            with torch.inference_mode():
                expected = flat(mic, far, *torch_cache)
            feeds = dict(zip(spec.input_names, (mic.numpy(), far.numpy(), *ort_cache)))
            actual = session.run(list(spec.output_names), feeds)
            for actual_value, expected_value in zip(actual, expected):
                torch.testing.assert_close(
                    torch.from_numpy(actual_value), expected_value, rtol=1e-4, atol=1e-5
                )
            torch_cache = list(expected[1:])
            ort_cache = actual[1:]
