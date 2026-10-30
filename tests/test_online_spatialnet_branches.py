"""OnlineSpatialNet 三分支（mhsa/ret/mamba）构造、cache 与等价性合同。"""

from types import SimpleNamespace

import pytest
import torch

from core.data.simulation_configs import SimulationConfigs
from core.models.io_contract import validate_spatial_model_geometry
from tasks.se.models import build_model

pytestmark = pytest.mark.unit

SE_BASE = dict(
    n_fft=8,
    n_channel=2,
    ref_channel=0,
    num_layers=1,
    dim_hidden=8,
    dim_ffn=8,
    dim_squeeze=2,
    num_heads=2,
    conv_groups=(1, 1),
)

BRANCHES = (
    ("mhsa(4)", {}),
    ("ret(2,not_share_qk)", {"decay": [4, 5]}),
    ("ret(2,share_qk)", {"decay": [4, 5]}),
    ("mamba(4,4)", {}),
)


def _build(attention, extra=None, **overrides):
    args = {**SE_BASE, "attention": attention, **(extra or {}), **overrides}
    return build_model({"type": "OnlineSpatialNet_SE", "args": args}).eval()


def _dataset(reference_channel=0):
    return SimpleNamespace(
        sim_cfg=SimulationConfigs.from_dict(
            {
                "spatial_enabled": True,
                "array_profile": "linear_2ch_80mm",
                "reference_channel": reference_channel,
            },
            strict=True,
        )
    )


@pytest.mark.parametrize("attention,extra", BRANCHES)
def test_branch_sequence_matches_frame_streaming(attention, extra):
    torch.manual_seed(3)
    model = _build(attention, extra)
    spectrum = torch.randn(1, 2, 5, 6, 2)
    with torch.no_grad():
        full = model(spectrum)
        cache = model.init_cache(1, torch.device("cpu"), torch.float32)
        frames = []
        for index in range(6):
            frame, cache = model.forward_streaming(spectrum[:, :, :, index : index + 1], cache)
            frames.append(frame)
    torch.testing.assert_close(torch.cat(frames, dim=2), full, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("attention,extra", BRANCHES)
def test_branch_cache_is_resettable_without_history(attention, extra):
    torch.manual_seed(5)
    model = _build(attention, extra)
    spectrum = torch.randn(1, 2, 5, 3, 2)
    with torch.no_grad():
        first_cache = model.init_cache(1, torch.device("cpu"), torch.float32)
        first = []
        for index in range(3):
            frame, first_cache = model.forward_streaming(spectrum[:, :, :, index : index + 1], first_cache)
            first.append(frame)
        fresh_cache = model.init_cache(1, torch.device("cpu"), torch.float32)
        second = []
        for index in range(3):
            frame, fresh_cache = model.forward_streaming(spectrum[:, :, :, index : index + 1], fresh_cache)
            second.append(frame)
    torch.testing.assert_close(torch.cat(first, dim=2), torch.cat(second, dim=2))


@pytest.mark.parametrize("attention,extra", BRANCHES)
def test_branch_cache_abi_is_derived_and_consistent(attention, extra):
    model = _build(attention, extra)
    assert model.streaming_spatial_abi_version == 2
    cache = model.init_cache(2, torch.device("cpu"), torch.float32)
    assert tuple(cache) == model.CACHE_KEYS
    expected = model.backbone.expected_cache_shapes(2)
    for name in model.CACHE_KEYS:
        assert tuple(cache[name].shape) == expected[name]
    model.backbone.validate_cache(cache, 2, torch.device("cpu"), torch.float32)


def test_mhsa_branch_uses_incremental_kv_cache():
    model = _build("mhsa(4)")
    assert model.CACHE_KEYS == ("encoder", "frame_index", "layer0_key", "layer0_value",
                                "layer0_temporal0", "layer0_temporal1", "layer0_temporal2")
    cache = model.init_cache(1, torch.device("cpu"), torch.float32)
    assert cache["layer0_key"].shape == (1, 5, 2, 4, 4)


def test_ret_and_mamba_branches_do_not_keep_mhsa_windows():
    ret = _build("ret(2,not_share_qk)", {"decay": [4, 5]})
    assert "layer0_key" not in ret.CACHE_KEYS
    assert {"layer0_ret_kv", "layer0_ret_scale"} <= set(ret.CACHE_KEYS)
    mamba = _build("mamba(4,4)")
    assert {"layer0_attn_conv", "layer0_attn_ssm", "layer0_ffn_conv", "layer0_ffn_ssm"} <= set(mamba.CACHE_KEYS)
    assert "layer0_temporal0" not in mamba.CACHE_KEYS


@pytest.mark.parametrize("attention,extra", BRANCHES)
def test_branch_training_step_produces_finite_gradients(attention, extra):
    torch.manual_seed(17)
    model = _build(attention, extra)
    spectrum = torch.randn(1, 2, 5, 6, 2)
    target = torch.randn(1, 5, 6, 2)
    loss = torch.nn.functional.l1_loss(model(spectrum), target)
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(torch.count_nonzero(grad) > 0 for grad in grads)


def test_long_sequence_attention_matches_streaming_and_stays_finite():
    """T > scope 时整段窗口 mask 必须有限且与逐帧流式一致（长序列 LF 训练的前提）。"""
    torch.manual_seed(23)
    model = _build("mhsa(16)")
    spectrum = torch.randn(1, 2, 5, 64, 2)
    with torch.no_grad():
        full = model(spectrum)
        cache = model.init_cache(1, torch.device("cpu"), torch.float32)
        frames = []
        for index in range(64):
            frame, cache = model.forward_streaming(spectrum[:, :, :, index : index + 1], cache)
            frames.append(frame)
    assert torch.isfinite(full).all()
    torch.testing.assert_close(torch.cat(frames, dim=2), full, rtol=1e-5, atol=1e-6)


def test_gradient_checkpointing_matches_plain_training_step():
    """逐层 activation recomputation 不改变前向输出与梯度。"""
    torch.manual_seed(29)
    plain = _build("mhsa(16)")
    checkpointed = _build("mhsa(16)", {"gradient_checkpointing": True})
    checkpointed.load_state_dict(plain.state_dict())
    spectrum = torch.randn(1, 2, 5, 64, 2)

    def step(model):
        model.train()
        model.zero_grad(set_to_none=True)
        output = model(spectrum)
        output.abs().mean().backward()
        grads = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
        return output.detach(), grads

    out_plain, grad_plain = step(plain)
    out_ckpt, grad_ckpt = step(checkpointed)
    torch.testing.assert_close(out_ckpt, out_plain, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad_ckpt, grad_plain, rtol=1e-4, atol=1e-5)


def test_ret_rope_branch_keeps_frame_index_counter():
    torch.manual_seed(13)
    model = _build("ret(2,not_share_qk)", {"decay": [4, 5], "rope": True})
    assert "frame_index" in model.CACHE_KEYS
    spectrum = torch.randn(1, 2, 5, 5, 2)
    with torch.no_grad():
        full = model(spectrum)
        cache = model.init_cache(1, torch.device("cpu"), torch.float32)
        frames = []
        for index in range(5):
            frame, cache = model.forward_streaming(spectrum[:, :, :, index : index + 1], cache)
            frames.append(frame)
    torch.testing.assert_close(torch.cat(frames, dim=2), full, rtol=1e-5, atol=1e-6)
    assert cache["frame_index"].item() == 5.0


def test_online_spatialnet_geometry_contract_matches_dataset():
    model = _build("mhsa(4)")
    validate_spatial_model_geometry(model, _dataset(reference_channel=0))
    with pytest.raises(ValueError, match="reference_channel"):
        validate_spatial_model_geometry(model, _dataset(reference_channel=1))


@pytest.mark.parametrize(
    "attention,extra,message",
    (
        ("mhsa(inf)", {}, "finite"),
        ("mhsa(0)", {}, "positive integer"),
        ("mhsa(4)", {"rope": True}, "rope"),
        ("mhsa(4)", {"decay": [4, 5]}, "decay"),
        ("ret(2,bogus)", {}, r"ret\(factor"),
        ("ret(2,share_qk)", {"decay": [4, 5, 6]}, "decay length"),
        ("mamba(16)", {}, r"mamba\(d_state,d_conv\)"),
        ("mamba(16,4)", {"rope": True}, "rope"),
        ("bogus", {}, "mhsa\\(scope\\)"),
    ),
)
def test_invalid_attention_specs_fail_fast(attention, extra, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        _build(attention, extra)
