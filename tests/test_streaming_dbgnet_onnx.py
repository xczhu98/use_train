"""DBGNet state-off compact 流式 ONNX/ORT 端到端回归。"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from tasks.aec.models.discriminative.dbgnet import DBGNet
from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner
from onnx.models.streaming_dbgnet import StreamingDBGNet, StreamingDBGNetFlat
from onnx.runtime_spec import (
    dbgnet_model_fingerprint,
    runtime_spec_from_session,
    runtime_spec_from_wrapper,
)
from onnx.utils import export_onnx, pypi_onnx_scope


pytestmark = [pytest.mark.integration, pytest.mark.ort]

_N_FFT = 128
_HOP_LENGTH = 64
_N_FRAMES = 4
_MAX_ABS_ERROR = 5e-4


def _require_onnx_dependencies():
    try:
        with pypi_onnx_scope():
            pass
    except ImportError as exc:
        pytest.skip(f"PyPI onnx is unavailable: {exc}")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        pytest.skip(f"onnxruntime is unavailable: {exc}")
    return ort


def _compact_state_off_config() -> dict:
    return {
        "model": {
            "type": "DBGNet",
            "args": {
                "n_fft": _N_FFT,
                "hop_length": _HOP_LENGTH,
                "win_length": _N_FFT,
                "sample_rate": 16000,
                "erb_low_bins": 17,
                "erb_bands": 16,
                "mic_channels": [4, 8, 12, 16],
                "far_channels": [2, 4, 6, 8],
                "use_state_detector": False,
            },
        },
        "stft": {
            "n_fft": _N_FFT,
            "hop_length": _HOP_LENGTH,
            "win_length": _N_FFT,
            "window": "hann",
            "center": False,
        },
        "data": {"sample_rate": 16000},
        "aec_frontend": {"input_routing": "mic_farend"},
    }


def _assert_real_mask_contract(enhanced: np.ndarray, microphone: np.ndarray) -> None:
    mask_from_real = enhanced[..., 0] / microphone[..., 0]
    mask_from_imag = enhanced[..., 1] / microphone[..., 1]
    assert np.all(np.isfinite(mask_from_real))
    assert np.all(np.isfinite(mask_from_imag))
    np.testing.assert_allclose(mask_from_real, mask_from_imag, rtol=2e-5, atol=2e-6)
    assert np.all(mask_from_real >= 0.0)
    assert np.all(mask_from_real <= 1.0)


def _assert_pt_ort_tensor_close(
    *,
    label: str,
    frame: int,
    pt_value: torch.Tensor,
    ort_value: np.ndarray,
) -> None:
    pt_array = pt_value.detach().cpu().numpy()
    assert pt_array.shape == ort_value.shape, (
        f"frame={frame} {label} shape mismatch: PT={pt_array.shape}, ORT={ort_value.shape}"
    )
    assert pt_array.dtype == np.float32
    assert ort_value.dtype == np.float32
    assert np.all(np.isfinite(pt_array))
    assert np.all(np.isfinite(ort_value))
    max_abs = float(np.max(np.abs(pt_array - ort_value)))
    assert np.isfinite(max_abs)
    assert max_abs <= _MAX_ABS_ERROR, (
        f"frame={frame} {label} max_abs={max_abs:.6e} exceeds {_MAX_ABS_ERROR:.6e}"
    )


def test_state_off_compact_dbgnet_onnx_streaming_cache_and_reset(tmp_path):
    ort = _require_onnx_dependencies()
    from onnx.export import write_streaming_runtime_metadata

    torch.manual_seed(20260831)
    config = _compact_state_off_config()
    model = DBGNet(**config["model"]["args"]).eval()
    stream = StreamingDBGNet(
        model,
        window=config["stft"]["window"],
        center=config["stft"]["center"],
        deployment_config=config,
    ).eval()
    flat = StreamingDBGNetFlat(stream).eval()

    n_freqs = _N_FFT // 2 + 1
    mic = torch.rand(1, n_freqs, _N_FRAMES, 2, dtype=torch.float32) + 0.25
    mic[..., 1].neg_()
    far = torch.randn(1, n_freqs, _N_FRAMES, 2, dtype=torch.float32)
    initial_cache = flat.init_cache(1, "cpu", torch.float32)
    first_mic = mic[:, :, :1, :]
    first_far = far[:, :, :1, :]

    onnx_path = tmp_path / "streaming_dbgnet_state_off_compact.onnx"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_onnx(
            flat,
            (first_mic, first_far, *initial_cache),
            str(onnx_path),
            input_names=flat.get_input_names(),
            output_names=flat.get_output_names(),
            dynamic_axes=flat.get_dynamic_axes(),
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
            dynamo=False,
        )
    eager_validation_warnings = [
        item for item in caught
        if issubclass(item.category, torch.jit.TracerWarning)
        and (
            item.filename.endswith("tasks/aec/models/discriminative/dbgnet.py")
            or item.filename.endswith("onnx/models/streaming_dbgnet.py")
        )
    ]
    assert eager_validation_warnings == []
    expected_spec = runtime_spec_from_wrapper(flat, "dbgnet")
    written_metadata = write_streaming_runtime_metadata(str(onnx_path), flat)
    assert written_metadata == dict(expected_spec.metadata)

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    runtime_spec = runtime_spec_from_session(session, "dbgnet")
    assert runtime_spec.input_names == expected_spec.input_names
    assert runtime_spec.output_names == expected_spec.output_names
    assert dict(runtime_spec.cache_shapes) == dict(expected_spec.cache_shapes)
    assert {
        name: dict(axes) for name, axes in runtime_spec.dynamic_axes.items()
    } == {
        name: dict(axes) for name, axes in expected_spec.dynamic_axes.items()
    }
    assert runtime_spec.stft == expected_spec.stft
    assert runtime_spec.dbgnet == expected_spec.dbgnet
    assert dict(runtime_spec.deployment_config or {}) == dict(
        expected_spec.deployment_config or {}
    )
    assert runtime_spec.dbgnet is not None
    assert runtime_spec.dbgnet.use_state_detector is False
    assert dict(runtime_spec.dbgnet.model_fingerprint) == dbgnet_model_fingerprint(model)
    assert "state_probs" not in runtime_spec.output_names
    assert tuple(runtime_spec.cache_input_names) == tuple(flat.CACHE_KEYS)

    runner = StreamingORTFrameRunner(session, spec=runtime_spec)
    pt_cache = initial_cache
    first_pt_outputs = None
    first_ort_enhancement = None
    first_ort_cache = None

    for frame in range(_N_FRAMES):
        mic_frame = mic[:, :, frame : frame + 1, :]
        far_frame = far[:, :, frame : frame + 1, :]
        with torch.inference_mode():
            pt_outputs = flat(mic_frame, far_frame, *pt_cache)
        pt_enhancement = pt_outputs[0]
        pt_cache = pt_outputs[1:]
        ort_enhancement, _ = runner.run_frame(mic_frame, far_frame)
        ort_cache = runner.state

        _assert_pt_ort_tensor_close(
            label="enhancement",
            frame=frame,
            pt_value=pt_enhancement,
            ort_value=ort_enhancement,
        )
        _assert_real_mask_contract(
            pt_enhancement.detach().cpu().numpy(), mic_frame.numpy()
        )
        _assert_real_mask_contract(ort_enhancement, mic_frame.numpy())

        assert len(pt_cache) == len(flat.CACHE_KEYS)
        assert tuple(ort_cache) == tuple(flat.CACHE_KEYS)
        for cache_name, pt_value in zip(flat.CACHE_KEYS, pt_cache):
            assert tuple(pt_value.shape) == runtime_spec.cache_shapes[cache_name]
            assert ort_cache[cache_name].shape == runtime_spec.cache_shapes[cache_name]
            _assert_pt_ort_tensor_close(
                label=f"cache {cache_name}",
                frame=frame,
                pt_value=pt_value,
                ort_value=ort_cache[cache_name],
            )

        if frame == 0:
            first_pt_outputs = tuple(value.detach().clone() for value in pt_outputs)
            first_ort_enhancement = ort_enhancement.copy()
            first_ort_cache = {
                name: ort_cache[name].copy() for name in flat.CACHE_KEYS
            }

    assert first_pt_outputs is not None
    assert first_ort_enhancement is not None
    assert first_ort_cache is not None

    with torch.inference_mode():
        reset_pt_outputs = flat(
            first_mic,
            first_far,
            *flat.init_cache(1, "cpu", torch.float32),
        )
    runner.reset()
    reset_ort_enhancement, _ = runner.run_frame(first_mic, first_far)
    reset_ort_cache = runner.state

    for original, reset in zip(first_pt_outputs, reset_pt_outputs):
        torch.testing.assert_close(reset, original, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(reset_ort_enhancement, first_ort_enhancement)
    for cache_name in flat.CACHE_KEYS:
        np.testing.assert_array_equal(
            reset_ort_cache[cache_name], first_ort_cache[cache_name]
        )

    source_snapshot = runner.state
    same_artifact_runner = StreamingORTFrameRunner(session, spec=runtime_spec)
    with pytest.raises(ValueError, match="runner identity mismatch"):
        same_artifact_runner.reset(source_snapshot)

    torch.manual_seed(20260901)
    other_model = DBGNet(**config["model"]["args"]).eval()
    other_flat = StreamingDBGNetFlat(StreamingDBGNet(
        other_model,
        window=config["stft"]["window"],
        center=config["stft"]["center"],
        deployment_config=config,
    )).eval()
    other_cache = other_flat.init_cache(1, "cpu", torch.float32)
    other_path = tmp_path / "streaming_dbgnet_same_config_other_weights.onnx"
    export_onnx(
        other_flat,
        (first_mic, first_far, *other_cache),
        str(other_path),
        input_names=other_flat.get_input_names(),
        output_names=other_flat.get_output_names(),
        dynamic_axes=other_flat.get_dynamic_axes(),
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
        dynamo=False,
    )
    write_streaming_runtime_metadata(str(other_path), other_flat)
    other_session = ort.InferenceSession(
        str(other_path), providers=["CPUExecutionProvider"]
    )
    other_runtime_spec = runtime_spec_from_session(other_session, "dbgnet")
    assert (
        other_runtime_spec.dbgnet.model_fingerprint["sha256"]
        != runtime_spec.dbgnet.model_fingerprint["sha256"]
    )
    other_runner = StreamingORTFrameRunner(other_session, spec=other_runtime_spec)
    with pytest.raises(ValueError, match="model identity mismatch"):
        other_runner.reset(source_snapshot)
