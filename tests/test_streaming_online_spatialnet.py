"""OnlineSpatialNet streaming cache 与多通道 observation 合同。"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from onnx.models.streaming_online_spatialnet import (
    OnlineSpatialNetSession,
    StreamingOnlineSpatialNetFlat,
)
from onnx.runtime_spec import runtime_spec_from_wrapper


pytestmark = pytest.mark.unit


class _CausalBackend(nn.Module):
    streaming_spatial_abi_version = 2
    n_channel = 2
    n_fft = 8
    hop_length = 4
    win_length = 8
    target_fs = 16000
    window = "hann"
    center = False
    CACHE_KEYS = ("history", "flattened")

    def init_cache(self, batch_size, device, dtype):
        return {
            "history": torch.zeros(batch_size, 1, device=device, dtype=dtype),
            "flattened": torch.zeros(1, batch_size * 5, 2, device=device, dtype=dtype),
        }

    def forward_streaming(self, mixture_spec, cache):
        history = cache["history"]
        frames = []
        for index in range(mixture_spec.shape[3]):
            frame = mixture_spec[:, :, :, index : index + 1, :].mean(dim=1)
            frames.append(frame + history[:, :, None, None])
            history = history + 1
        output = torch.cat(frames, dim=2)
        return output, {
            "history": history,
            "flattened": cache["flattened"] + 1,
        }


def test_online_spatialnet_session_matches_full_and_reset():
    wrapper = StreamingOnlineSpatialNetFlat(_CausalBackend()).eval()
    full_input = torch.randn(1, 2, 5, 3, 2)
    full = OnlineSpatialNetSession(wrapper)
    full_output = full.process([full_input])

    chunked = OnlineSpatialNetSession(wrapper)
    chunked_output = chunked.process([full_input[:, :, :, :2], full_input[:, :, :, 2:]])
    torch.testing.assert_close(chunked_output, full_output)
    assert chunked.state["history"].item() == 3
    assert chunked.state["flattened"].shape == (1, 5, 2)

    chunked.reset()
    assert torch.count_nonzero(chunked.state["history"]) == 0
    assert torch.count_nonzero(chunked.state["flattened"]) == 0


def test_online_spatialnet_wrapper_exposes_derived_abi_and_metadata():
    wrapper = StreamingOnlineSpatialNetFlat(_CausalBackend()).eval()
    spec = runtime_spec_from_wrapper(wrapper, "online_spatialnet")

    assert spec.input_names == ("mixture_spec", "history", "flattened")
    assert spec.output_names == (
        "enhanced_spec", "history_out", "flattened_out",
    )
    assert dict(spec.cache_shapes) == {
        "history": (1, 1), "flattened": (1, 5, 2),
    }
    assert spec.dynamic_axes["mixture_spec"] == {0: "batch"}
    assert spec.dynamic_axes["history"] == {0: "batch"}
    assert spec.dynamic_axes["flattened"] == {1: "batch_flattened"}
    assert spec.metadata["online_spatialnet_cache_cfg"] == {
        "history": [1, 1], "flattened": [1, 5, 2],
    }
    assert spec.online_spatialnet.cache_names == ("history", "flattened")
    assert spec.online_spatialnet.cache_abi_version == 2


def test_online_spatialnet_rejects_offline_backend_and_non_mono_output():
    from tasks.ss.models.spatialnet import SpatialNet

    with pytest.raises(ValueError, match="offline SpatialNet"):
        StreamingOnlineSpatialNetFlat(SpatialNet(n_channel=2, num_sources=1))

    class BadBackend(_CausalBackend):
        def forward_streaming(self, mixture_spec, cache):
            return mixture_spec, cache

    wrapper = StreamingOnlineSpatialNetFlat(BadBackend())
    with pytest.raises(ValueError, match="mono spectrum"):
        wrapper(
            torch.randn(1, 2, 5, 1, 2),
            *wrapper.init_cache_flat(1, "cpu", torch.float32),
        )


def test_online_spatialnet_requires_exact_cache_mapping_order():
    class BadCacheBackend(_CausalBackend):
        def init_cache(self, batch_size, device, dtype):
            values = super().init_cache(batch_size, device, dtype)
            return {"flattened": values["flattened"], "history": values["history"]}

    with pytest.raises(ValueError, match="names/order mismatch"):
        StreamingOnlineSpatialNetFlat(BadCacheBackend()).init_cache_flat()


@pytest.mark.integration
@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("onnx") is None
    or __import__("importlib.util").util.find_spec("onnxruntime") is None,
    reason="PyPI onnx and onnxruntime are required for ONNX ABI integration",
)
def test_online_spatialnet_onnx_roundtrip_uses_same_cache_abi(tmp_path):
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner
    from onnx.runtime_spec import runtime_spec_from_session
    from onnx.utils import create_ort_session, export_onnx, write_onnx_metadata

    wrapper = StreamingOnlineSpatialNetFlat(_CausalBackend()).eval()
    runtime = runtime_spec_from_wrapper(wrapper, "online_spatialnet")
    cache = wrapper.init_cache_flat()
    path = tmp_path / "online_spatialnet.onnx"
    export_onnx(
        wrapper,
        (torch.randn(1, 2, 5, 1, 2), *cache),
        str(path),
        input_names=list(runtime.input_names),
        output_names=list(runtime.output_names),
        dynamic_axes={name: dict(axes) for name, axes in runtime.dynamic_axes.items()},
        opset_version=17,
        dynamo=False,
    )
    write_onnx_metadata(path, dict(runtime.metadata))
    session = create_ort_session(path)
    loaded = runtime_spec_from_session(session, "online_spatialnet")
    runner = StreamingORTFrameRunner(session, spec=loaded)
    output, _ = runner.run_frame(torch.randn(1, 2, 5, 1, 2).numpy())
    assert output.shape == (1, 5, 1, 2)


@pytest.mark.integration
@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("onnx") is None
    or __import__("importlib.util").util.find_spec("onnxruntime") is None,
    reason="PyPI onnx and onnxruntime are required for ONNX ABI integration",
)
def test_online_spatialnet_history_zero_onnx_keeps_empty_cache_inputs(tmp_path):
    from onnx.export import export_model, write_streaming_runtime_metadata
    from onnx.runtime_spec import runtime_spec_from_session
    from onnx.utils import create_ort_session
    from tasks.se.models import build_model

    model = build_model(
        {
            "type": "OnlineSpatialNet_SE",
            "args": {
                "n_fft": 8,
                "n_channel": 2,
                "num_layers": 1,
                "dim_hidden": 8,
                "dim_ffn": 8,
                "dim_squeeze": 2,
                "num_heads": 2,
                "attention_scope": 1,
                "encoder_kernel_size": 1,
                "kernel_size": (1, 1),
                "conv_groups": (1, 1),
            },
        }
    ).eval()
    wrapper = StreamingOnlineSpatialNetFlat(model)
    path = tmp_path / "history_zero.onnx"
    export_model(wrapper, str(path))
    write_streaming_runtime_metadata(path, wrapper)
    session = create_ort_session(path)
    loaded = runtime_spec_from_session(session, "online_spatialnet")
    assert tuple(value.name for value in session.get_inputs()) == loaded.input_names
    assert loaded.cache_shapes["layer0_key"] == (1, 5, 2, 1, 4)
    assert loaded.cache_shapes["layer0_value"] == (1, 5, 2, 1, 4)
    assert loaded.cache_shapes["frame_index"] == (1, 1)
    assert loaded.cache_shapes["layer0_temporal0"] == (1, 5, 8, 0)
