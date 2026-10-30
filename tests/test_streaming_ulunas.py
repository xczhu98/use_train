"""UL-UNAS 官方 streaming ONNX ABI 与逐帧等价性测试。"""

from __future__ import annotations

import pytest
import torch

from onnx.models.streaming_ulunas import StreamingULUNAS, StreamingULUNASFlat
from tasks.se.models.discriminative.ulunas import ULUNAS


@pytest.fixture(scope="module")
def models() -> tuple[ULUNAS, StreamingULUNAS, StreamingULUNASFlat]:
    torch.manual_seed(7)
    base = ULUNAS(n_fft=512).eval()
    stream = StreamingULUNAS(base).eval()
    return base, stream, StreamingULUNASFlat(stream).eval()


@pytest.mark.unit
@pytest.mark.parametrize("batch_size", [1, 2])
def test_official_cache_shapes(models, batch_size: int) -> None:
    _, stream, flat = models
    cache = stream.init_cache(batch_size, "cpu", torch.float32)
    assert {key: tuple(value.shape) for key, value in cache.items()} == {
        "conv_cache": (batch_size, 5358),
        "tfa_cache": (batch_size, 402),
        "inter_cache": (batch_size, 1056),
    }
    assert [tuple(value.shape) for value in flat.init_cache_flat(batch_size, "cpu", torch.float32)] == [
        (batch_size, 5358),
        (batch_size, 402),
        (batch_size, 1056),
    ]


@pytest.mark.unit
def test_single_frame_batch_output_and_official_flat_abi(models) -> None:
    _, stream, flat = models
    batch_size = 2
    spec = torch.randn(batch_size, 257, 1, 2)
    cache = stream.init_cache(batch_size, "cpu", spec.dtype)
    with torch.inference_mode():
        enhanced, next_cache = stream(spec, cache)
        flat_outputs = flat(spec, *flat.init_cache_flat(batch_size, "cpu", spec.dtype))

    assert tuple(enhanced.shape) == (batch_size, 257, 1, 2)
    assert {key: tuple(value.shape) for key, value in next_cache.items()} == {
        "conv_cache": (batch_size, 5358),
        "tfa_cache": (batch_size, 402),
        "inter_cache": (batch_size, 1056),
    }
    assert [tuple(value.shape) for value in flat_outputs] == [
        (batch_size, 257, 1, 2),
        (batch_size, 5358),
        (batch_size, 402),
        (batch_size, 1056),
    ]
    assert flat.get_input_names() == ["mix", "conv_cache", "tfa_cache", "inter_cache"]
    assert flat.get_output_names() == ["enh", "conv_cache_out", "tfa_cache_out", "inter_cache_out"]
    assert flat.get_dynamic_axes() == {
        "mix": {0: "batch"},
        "conv_cache": {0: "batch"},
        "tfa_cache": {0: "batch"},
        "inter_cache": {0: "batch"},
        "enh": {0: "batch"},
        "conv_cache_out": {0: "batch"},
        "tfa_cache_out": {0: "batch"},
        "inter_cache_out": {0: "batch"},
    }


@pytest.mark.unit
def test_random_default_offline_matches_frame_by_frame_streaming(models) -> None:
    base, stream, _ = models
    torch.manual_seed(11)
    spec = torch.randn(1, 257, 6, 2)
    cache = stream.init_cache(1, "cpu", spec.dtype)
    frames: list[torch.Tensor] = []
    with torch.inference_mode():
        offline = base(spec)
        for frame_index in range(spec.shape[2]):
            enhanced, cache = stream(spec[:, :, frame_index : frame_index + 1, :], cache)
            frames.append(enhanced)
    streaming = torch.cat(frames, dim=2)

    torch.testing.assert_close(streaming, offline, rtol=2e-5, atol=2e-6)
