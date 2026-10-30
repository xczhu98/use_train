"""真实 OnlineSpatialNet backend 的构造、三分支 cache 与 chunk parity。"""

import pytest
import torch

from onnx.models.streaming_online_spatialnet import (
    OnlineSpatialNetSession,
    StreamingOnlineSpatialNetFlat,
)
from tasks.se.models import build_model


pytestmark = pytest.mark.unit


def _build(n_channel=2, *, attention_scope=3, kernel_size=(5, 3), target_fs=None):
    args = {
        "n_fft": 8,
        "n_channel": n_channel,
        "num_layers": 1,
        "dim_hidden": 8,
        "dim_ffn": 8,
        "dim_squeeze": 2,
        "num_heads": 2,
        "attention_scope": attention_scope,
        "kernel_size": kernel_size,
        "conv_groups": (1, 1),
    }
    if target_fs is not None:
        args["target_fs"] = target_fs
    return build_model(
        {"type": "OnlineSpatialNet_SE", "args": args}
    ).eval()


def test_online_spatialnet_real_backend_builds_and_chunk_parity_holds():
    model = _build()
    signal = torch.randn(1, 2, 5, 4, 2)
    full = model(signal)
    wrapper = StreamingOnlineSpatialNetFlat(model)
    assert wrapper.CACHE_KEYS
    session = OnlineSpatialNetSession(wrapper)
    chunked = session.process((signal[:, :, :, :2], signal[:, :, :, 2:]))
    torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-6)
    assert chunked.shape == (1, 5, 4, 2)


def test_online_spatialnet_real_backend_keeps_c1_legacy_input():
    model = _build(n_channel=1)
    output = model(torch.randn(1, 5, 3, 2))
    assert output.shape == (1, 5, 3, 2)


def test_online_spatialnet_rejects_runtime_sample_rate_mismatch():
    model = _build(target_fs=16000)
    signal = torch.randn(1, 2, 5, 3, 2)
    with pytest.raises(ValueError, match="requires fs=16000, got 8000"):
        model(signal, fs=8000)
    assert model(signal, fs=16000).shape == (1, 5, 3, 2)


def test_online_spatialnet_history_zero_keeps_exact_empty_cache_shapes():
    model = _build(attention_scope=1, kernel_size=(1, 1))
    wrapper = StreamingOnlineSpatialNetFlat(model)
    cache = wrapper.init_cache()
    # mhsa 分支：增量 K/V 形状 [B, F, heads, scope, head_dim]；scope=1 时保留 1 帧窗口。
    assert cache["layer0_key"].shape == (1, 5, 2, 1, 4)
    assert cache["layer0_value"].shape == (1, 5, 2, 1, 4)
    assert cache["frame_index"].shape == (1, 1)
    assert cache["layer0_temporal0"].shape == (1, 5, 8, 0)
    output, updated = model.forward_streaming(
        torch.randn(1, 2, 5, 1, 2), cache
    )
    assert output.shape == (1, 5, 1, 2)
    assert updated["layer0_key"].shape == cache["layer0_key"].shape
    assert updated["layer0_value"].shape == cache["layer0_value"].shape
    assert updated["frame_index"].item() == 1.0
    assert updated["layer0_temporal0"].shape == cache["layer0_temporal0"].shape


def test_online_spatialnet_real_backend_rejects_mamba_without_capability(monkeypatch):
    import tasks.se.models.modules.online_spatialnet_impl.backbone as backbone_module

    monkeypatch.setattr(backbone_module, "mamba_available", lambda: False)
    with pytest.raises(RuntimeError, match="mamba_ssm"):
        build_model(
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
                    "attention": "mamba(16,4)",
                    "conv_groups": (1, 1),
                },
            }
        )


def test_online_spatialnet_real_backend_rejects_cache_order():
    model = _build()
    wrapper = StreamingOnlineSpatialNetFlat(model)
    cache = wrapper.init_cache_flat()
    with pytest.raises(ValueError, match="cache names/order mismatch"):
        model.forward_streaming(
            torch.randn(1, 2, 5, 1, 2),
            {name: value for name, value in zip(reversed(wrapper.CACHE_KEYS), reversed(cache))},
        )
