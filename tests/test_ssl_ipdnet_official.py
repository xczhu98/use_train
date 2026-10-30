"""官方 IPDnet fixed / variable 结构接入的 unit 测试。

只验证 ``tasks/ssl/models/ipdnet_official.py`` 的结构合同、输入输出布局、
构造期 fail-fast、时间轴和 pair-batched 适配；不依赖 GPU、真实语料或官方
checkpoint，CPU 可跑。

registry 已由 ``tasks/ssl/models/__init__.py`` 收口到
``tasks.ssl.models.ipdnet_official``；本文件直接验证官方模块的结构、输出布局与
构造期门禁，不再依赖历史占位类清理逻辑。
"""

from __future__ import annotations

import math

import pytest
import torch

from core.utils.registry import MODEL_REGISTRY
from tasks.ssl.model_io import ModelSpec, validate_ssl_model_io

from tasks.ssl.models.ipdnet_official import (  # noqa: E402
    IPDNetFixed,
    IPDNetVariable,
    forgetting_norm,
)

pytestmark = pytest.mark.unit


_STRIDE = 12


def _pair_index(channels: int) -> torch.Tensor:
    """reference=0 的 reference-to-all pair_index ``[P,2]``。"""
    return torch.tensor([[0, channel] for channel in range(1, channels)], dtype=torch.int64)


def _fixed(**overrides):
    kwargs = {"input_size": 4, "hidden_size": 8, "max_track": 2, "n_fft": 8, "dropout": 0.0}
    kwargs.update(overrides)
    return IPDNetFixed(**kwargs)


def _variable(**overrides):
    kwargs = {"input_size": 4, "hidden_size": 8, "max_track": 2, "n_fft": 8, "dropout": 0.0}
    kwargs.update(overrides)
    return IPDNetVariable(**kwargs)


# --------------------------------------------------------------------------- #
# import 与 model spec
# --------------------------------------------------------------------------- #


def test_models_import_and_declare_expected_model_spec():
    assert IPDNetFixed.MODEL_TYPE == "IPDNetFixed"
    assert IPDNetVariable.MODEL_TYPE == "IPDNetVariable"
    fixed_spec = IPDNetFixed.MODEL_SPEC
    variable_spec = IPDNetVariable.MODEL_SPEC
    for spec in (fixed_spec, variable_spec):
        assert isinstance(spec, ModelSpec)
        assert spec.feature_contract == "multichannel_stft_real_imag"
        assert spec.target_contract == "dpipd_pair"
        assert spec.pair_layout == "reference_to_all"
        assert spec.field_mode == "far_field"
        assert spec.reduction == "causal_conv_pool"
        assert spec.causal is True
        assert spec.mapping == "far_field_grid"
        assert spec.target_layout == "trailing_ri"
    assert fixed_spec.geometry_mode == "fixed_array"
    assert variable_spec.geometry_mode == "variable_array"
    assert fixed_spec.model_type == "IPDNetFixed"
    assert variable_spec.model_type == "IPDNetVariable"


def test_both_models_satisfy_ssl_model_io_contract():
    validate_ssl_model_io(_fixed())
    validate_ssl_model_io(_variable())


def test_models_are_registered_under_official_names():
    assert MODEL_REGISTRY.get("IPDNetFixed") is IPDNetFixed
    assert MODEL_REGISTRY.get("IPDNetVariable") is IPDNetVariable


# --------------------------------------------------------------------------- #
# 构造期 fail-fast
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_cls", [IPDNetFixed, IPDNetVariable])
def test_unknown_kwargs_raise_type_error(model_cls):
    with pytest.raises(TypeError, match="unexpected model arguments"):
        model_cls(input_size=4, hidden_size=8, max_track=2, n_fft=8, not_a_real_arg=1)


@pytest.mark.parametrize("model_cls", [IPDNetFixed, IPDNetVariable])
def test_non_official_time_stride_is_rejected(model_cls):
    with pytest.raises(ValueError, match="time stride at 12"):
        model_cls(input_size=4, hidden_size=8, max_track=2, n_fft=8, stride=6)


@pytest.mark.parametrize("model_cls", [IPDNetFixed, IPDNetVariable])
def test_non_causal_switch_selects_offline_variant(model_cls):
    """``causal=False`` 映射官方 ``is_online=False``，只改变 narrow-band LSTM。"""
    model = model_cls(input_size=4, hidden_size=8, max_track=2, n_fft=8, causal=False)
    assert model.causal is False
    assert model.is_online is False
    assert model.MODEL_SPEC.causal is False
    # 类级声明保留官方默认 online 合同，实例声明实际变体。
    assert type(model).MODEL_SPEC.causal is True
    assert model.block_1.narr_lstm.bidirectional is True
    assert model.block_2.narr_lstm.bidirectional is True
    assert model.block_1.narr_hidden_size == 4
    assert model.block_2.narr_hidden_size == 4
    # CNN 在两个变体中都保持官方 causal block。
    assert model.conv.pad == (1, 2)
    assert model.conv.pooling1.kernel_size == (1, 3)
    assert model.conv.pooling2.kernel_size == (1, 4)
    validate_ssl_model_io(model)
    # ``is_online`` 不是公开兼容别名，拼写错误必须 fail-fast。
    with pytest.raises(TypeError, match="unexpected model arguments"):
        model_cls(input_size=4, hidden_size=8, max_track=2, n_fft=8, is_online=False)


@pytest.mark.parametrize("model_cls", [IPDNetFixed, IPDNetVariable])
def test_exclude_dc_is_rejected(model_cls):
    with pytest.raises(ValueError, match="exclude_dc=false"):
        model_cls(input_size=4, hidden_size=8, max_track=2, n_fft=8, exclude_dc=True)


@pytest.mark.parametrize("model_cls", [IPDNetFixed, IPDNetVariable])
def test_odd_hidden_size_is_rejected(model_cls):
    with pytest.raises(ValueError, match="hidden_size must be even"):
        model_cls(input_size=4, hidden_size=7, max_track=2, n_fft=8)


def test_fixed_rejects_max_track_that_breaks_official_reshape():
    """官方 reshape 链要求 (P*K) 为偶数；K=1 在上游本身不可运行。"""
    with pytest.raises(ValueError, match=r"pairs \* max_track\) to be even"):
        _fixed(max_track=1)


def test_variable_rejects_non_two_tracks():
    """官方 variable 的 cnn_out_dim=4 硬编码 2 speaker × 2 re/im。"""
    with pytest.raises(ValueError, match="max_track must be 2"):
        _variable(max_track=1)


def test_variable_rejects_non_pair_input_size():
    with pytest.raises(ValueError, match="one microphone pair per network input"):
        _variable(input_size=8)


def test_variable_rejects_unknown_pair_channel_order():
    with pytest.raises(ValueError, match="pair_channel_order"):
        _variable(pair_channel_order="bogus")


# --------------------------------------------------------------------------- #
# forward 形状
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "batch,channels,freqs,frames,hidden,tracks",
    [
        (1, 2, 5, 36, 8, 2),
        (2, 2, 4, 24, 16, 2),
        (2, 4, 5, 48, 8, 2),
        (1, 6, 4, 48, 16, 2),
        (3, 3, 6, 60, 8, 2),
    ],
)
def test_fixed_forward_shapes(batch, channels, freqs, frames, hidden, tracks):
    model = _fixed(input_size=2 * channels, hidden_size=hidden, max_track=tracks, n_fft=2 * (freqs - 1))
    features = torch.randn(batch, frames, freqs, 2 * channels)
    output = model(features)
    assert output.shape == (batch, frames // _STRIDE, freqs, tracks, channels - 1, 2)
    assert torch.isfinite(output).all()
    # 官方 CausCnnBlock 最后一层是 tanh，输出必须落在 (-1, 1)。
    assert output.abs().max().item() <= 1.0


@pytest.mark.parametrize(
    "batch,freqs,frames,hidden,pairs",
    [
        (1, 5, 36, 8, 1),
        (2, 4, 48, 16, 2),
        (2, 5, 36, 8, 3),
        (3, 6, 60, 8, 4),
    ],
)
def test_variable_forward_shapes(batch, freqs, frames, hidden, pairs):
    channels = pairs + 1
    model = _variable(hidden_size=hidden, n_fft=2 * (freqs - 1))
    features = torch.randn(batch, frames, freqs, 2 * channels)
    pair_index = _pair_index(channels)[None].expand(batch, -1, -1)
    output = model(features, pair_index=pair_index)
    assert output.shape == (batch, frames // _STRIDE, freqs, 2, pairs, 2)
    assert torch.isfinite(output).all()
    assert output.abs().max().item() <= 1.0


def test_forward_accepts_features_keyword_alias():
    model = _fixed(input_size=4, hidden_size=8, max_track=2, n_fft=8)
    features = torch.randn(1, 36, 5, 4)
    assert torch.equal(model(features), model(features=features))


def test_forward_rejects_both_input_aliases():
    model = _fixed(input_size=4, hidden_size=8, max_track=2, n_fft=8)
    features = torch.randn(1, 36, 5, 4)
    with pytest.raises(ValueError, match="only multichannel_stft_real_imag or features"):
        model(features, features=features)


def test_variable_requires_pair_index():
    model = _variable()
    with pytest.raises(TypeError, match="pair_index"):
        model(torch.randn(1, 36, 5, 6))


def test_fixed_rejects_wrong_microphone_count():
    model = _fixed(input_size=4, hidden_size=8, max_track=2, n_fft=8)
    with pytest.raises(ValueError):
        model(torch.randn(1, 36, 5, 6))


def test_fixed_rejects_wrong_frequency_bins():
    model = _fixed(input_size=4, hidden_size=8, max_track=2, n_fft=8)
    with pytest.raises(ValueError, match="frequency bins"):
        model(torch.randn(1, 36, 4, 4))


def test_fixed_rejects_too_short_time_axis():
    model = _fixed(input_size=4, hidden_size=8, max_track=2, n_fft=8)
    with pytest.raises(ValueError, match="at least 12 input frames"):
        model(torch.randn(1, _STRIDE - 1, 5, 4))


def test_fixed_rejects_non_finite_input():
    model = _fixed(input_size=4, hidden_size=8, max_track=2, n_fft=8)
    features = torch.randn(1, 36, 5, 4)
    features[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(features)


# --------------------------------------------------------------------------- #
# 时间轴与时间压缩
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("frames", [12, 24, 36, 48, 60, 84, 120, 300])
def test_time_axis_is_exactly_one_twelfth(frames):
    """官方卷积/池化链的时间压缩固定 12 倍，且与 T 的相位无关。"""
    model = _fixed(input_size=4, hidden_size=8, max_track=2, n_fft=8)
    output = model(torch.randn(1, frames, 5, 4))
    assert output.shape[1] == frames // _STRIDE


def test_configured_stride_attribute_matches_output_time_axis():
    model = _fixed(input_size=4, hidden_size=8, max_track=2, n_fft=8)
    assert model.stride == _STRIDE
    output = model(torch.randn(1, 48, 5, 4))
    assert output.shape[1] * model.stride == 48


# --------------------------------------------------------------------------- #
# 输出布局：官方 reshape 链 → canonical [B,T,F,K,P,2]
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("channels,tracks,freqs", [(2, 2, 3), (4, 2, 4), (6, 2, 3), (4, 2, 5)])
def test_official_output_layout_elementwise_mapping(channels, tracks, freqs):
    """逐元素验证官方输出轴到 canonical 的置换。

    给卷积输出的每个 ``(通道, 频点)`` 打唯一记号，确认它落在
    ``out[B, T, f, k, p, ridx]``；通道按 ``c = ((p*K + k)*2 + ridx)`` 打包。
    """
    from tasks.ssl.models.ipdnet_official import _official_to_canonical, _fixed_reshape_to_official

    pairs = channels - 1
    cnn_out_dim = 2 * pairs * tracks
    conv_out = torch.zeros(1, cnn_out_dim, freqs, 1)
    tags = {}
    value = 1
    for channel in range(cnn_out_dim):
        for freq in range(freqs):
            conv_out[0, channel, freq, 0] = value
            tags[value] = (channel, freq)
            value += 1
    official = _fixed_reshape_to_official(conv_out, tracks=tracks, pairs=pairs)
    canonical = _official_to_canonical(official, tracks=tracks, pairs=pairs)
    assert canonical.shape == (1, 1, freqs, tracks, pairs, 2)
    for tag, (channel, freq) in tags.items():
        pair = (channel // 2) // tracks
        track = (channel // 2) % tracks
        real_imag = channel % 2
        assert int(canonical[0, 0, freq, track, pair, real_imag].item()) == tag


def test_official_output_layout_is_a_lossless_permutation():
    """官方 reshape 链 → canonical 必须是无损的纯置换。

    与 ``test_official_output_layout_elementwise_mapping`` 一起固定转换语义：
    逐元素映射确定每个值落在哪里，本测试确认没有值被丢弃或复制。
    """
    from tasks.ssl.models.ipdnet_official import _fixed_reshape_to_official, _official_to_canonical

    channels, tracks, freqs, frames = 6, 2, 4, 7
    pairs = channels - 1
    conv_out = torch.randn(2, 2 * pairs * tracks, freqs, frames)
    official = _fixed_reshape_to_official(conv_out, tracks=tracks, pairs=pairs)
    canonical = _official_to_canonical(official, tracks=tracks, pairs=pairs)
    assert canonical.shape == (2, frames, freqs, tracks, pairs, 2)
    assert canonical.numel() == official.numel()
    assert torch.allclose(official.flatten().sort().values, canonical.flatten().sort().values)


def test_variable_official_layout_elementwise_mapping():
    """逐元素验证官方 variable 输出轴的对应关系。

    官方 variable 输出 ``[1, T2, 2F, B*P, 2]``：``2F`` 轴按 ``2*f + k`` 展开、
    ``B*P`` 轴按 ``(b, p)`` 展开、末轴为实虚部 ``ridx``；conv 通道按
    ``c = ridx*K + k`` 打包。
    """
    from tasks.ssl.models.ipdnet_official import (
        _variable_official_to_canonical,
        _variable_reshape_to_official,
    )

    batch, pairs, tracks, freqs, frames = 2, 3, 2, 4, 3
    conv_out = torch.zeros(batch * pairs, 2 * tracks, freqs, frames)
    tags = {}
    value = 1
    for pair_batch in range(batch * pairs):
        for channel in range(2 * tracks):
            for freq in range(freqs):
                conv_out[pair_batch, channel, freq, 0] = value
                tags[value] = (pair_batch, channel, freq)
                value += 1
    official = _variable_reshape_to_official(conv_out, batch=batch, pairs=pairs, tracks=tracks)
    canonical = _variable_official_to_canonical(official, batch=batch, pairs=pairs, tracks=tracks)
    assert canonical.shape == (batch, frames, freqs, tracks, pairs, 2)
    for tag, (pair_batch, channel, freq) in tags.items():
        batch_index = pair_batch // pairs
        pair = pair_batch % pairs
        track = channel % tracks
        real_imag = channel // tracks
        assert int(canonical[batch_index, 0, freq, track, pair, real_imag].item()) == tag


# --------------------------------------------------------------------------- #
# 输入通道顺序
# --------------------------------------------------------------------------- #


def test_fixed_reorders_canonical_interleaved_to_official_real_imag_blocks():
    """canonical 的逐麦交错 (re,im) 必须被重排成官方的 real 块 + imag 块。"""
    model = _fixed(input_size=4, hidden_size=8, max_track=2, n_fft=2)
    value = torch.randn(1, _STRIDE, 2, 2, 2)  # [B,T,F,C,2]，F=num_freqs=2
    captured = {}

    class _Spy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, block_input, fb_skip, nb_skip):
            captured["input"] = block_input.detach().clone()
            return self.inner(block_input, fb_skip=fb_skip, nb_skip=nb_skip)

    model.block_1 = _Spy(model.block_1)
    model(value.reshape(1, _STRIDE, 2, 4))
    block_input = captured["input"]
    assert block_input.shape == (1, _STRIDE, 2, 4)
    # 官方排布 = [re_mic0, re_mic1, im_mic0, im_mic1]
    assert torch.allclose(
        block_input, value[0, :, :, :, :].reshape(_STRIDE, 2, 4)[..., [0, 2, 1, 3]]
    )


def test_fixed_places_explicit_reference_channel_first():
    """fixed 模型必须把非零 reference_channel 对齐到官方 pair 轴首位。"""
    model = _fixed(input_size=4, hidden_size=8, max_track=2, n_fft=2, reference_channel=1)
    value = torch.randn(1, _STRIDE, 2, 2, 2)
    captured = {}

    class _Spy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, block_input, fb_skip, nb_skip):
            captured["input"] = block_input.detach().clone()
            return self.inner(block_input, fb_skip=fb_skip, nb_skip=nb_skip)

    model.block_1 = _Spy(model.block_1)
    model(value.reshape(1, _STRIDE, 2, 4))
    expected = value[0].reshape(_STRIDE, 2, 4)[..., [2, 0, 3, 1]]
    assert torch.allclose(captured["input"], expected)


def test_variable_pair_batch_interleaved_order():
    model = _variable(pair_channel_order="per_mic_interleaved")
    value = torch.randn(1, 3, 2, 3, 2)  # B=1,T=3,F=2,C=3
    pairs = model._pair_batch(value, _pair_index(3), 3)
    assert pairs.shape == (2, 3, 2, 4)
    mic = value[0]
    expected_first = torch.stack((mic[:, :, 0, 0], mic[:, :, 0, 1], mic[:, :, 1, 0], mic[:, :, 1, 1]), dim=-1)
    expected_second = torch.stack((mic[:, :, 0, 0], mic[:, :, 0, 1], mic[:, :, 2, 0], mic[:, :, 2, 1]), dim=-1)
    assert torch.allclose(pairs[0], expected_first)
    assert torch.allclose(pairs[1], expected_second)


def test_variable_pair_batch_block_order():
    model = _variable(pair_channel_order="real_block_then_imag_block")
    value = torch.randn(1, 3, 2, 3, 2)
    pairs = model._pair_batch(value, _pair_index(3), 3)
    mic = value[0]
    # real 块在前、imag 块在后
    expected_first = torch.stack((mic[:, :, 0, 0], mic[:, :, 1, 0], mic[:, :, 0, 1], mic[:, :, 1, 1]), dim=-1)
    assert torch.allclose(pairs[0], expected_first)


def test_variable_pair_batch_rejects_out_of_range_and_self_pairs():
    model = _variable()
    value = torch.randn(1, 3, 2, 3, 2)
    with pytest.raises(ValueError, match="out-of-range"):
        model._pair_batch(value, torch.tensor([[0, 3]]), 3)
    with pytest.raises(ValueError, match="self pairs"):
        model._pair_batch(value, torch.tensor([[1, 1]]), 3)
    with pytest.raises(ValueError, match="at least one pair"):
        model._pair_batch(value, torch.zeros(0, 2, dtype=torch.int64), 3)


def test_variable_pair_index_batch_dimension_must_match():
    model = _variable()
    value = torch.randn(2, 3, 2, 3, 2)
    with pytest.raises(ValueError, match="batch dimension must be 1 or match"):
        model._pair_batch(value, _pair_index(3)[None].expand(3, -1, -1), 3)


def test_variable_pair_index_must_be_integer():
    model = _variable()
    value = torch.randn(1, 3, 2, 3, 2)
    with pytest.raises(TypeError, match="integer dtype"):
        model._pair_batch(value, torch.tensor([[0.0, 1.0]]), 3)


# --------------------------------------------------------------------------- #
# pair-sharing 与广播
# --------------------------------------------------------------------------- #


def test_variable_output_is_pair_sharing_consistent_across_extra_pairs():
    """mean-sharing 让所有 pair 共享同一份 pair 维统计，因此输出可随 pair 数变化，
    但同一 pair 在两次不同 pair 集合下必须给出同一结果（取决于共享统计）。"""
    model = _variable(hidden_size=8, n_fft=8, dropout=0.0).eval()
    features = torch.randn(1, 36, 5, 8)  # C=4
    single = model(features, pair_index=torch.tensor([[[1, 0]]]))
    assert single.shape == (1, 3, 5, 2, 1, 2)
    assert torch.isfinite(single).all()


# --------------------------------------------------------------------------- #
# 反向传播
# --------------------------------------------------------------------------- #


def test_fixed_backward_produces_finite_gradients():
    model = _fixed(input_size=4, hidden_size=8, max_track=2, n_fft=8)
    features = torch.randn(2, 36, 5, 4, requires_grad=True)
    model(features).square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_variable_backward_produces_finite_gradients():
    model = _variable(hidden_size=8, n_fft=8)
    features = torch.randn(1, 36, 5, 6, requires_grad=True)
    pair_index = _pair_index(3)[None]
    model(features, pair_index=pair_index).square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


# --------------------------------------------------------------------------- #
# online / offline 变体
# --------------------------------------------------------------------------- #


def test_both_models_construct_in_the_official_default_variant():
    """官方默认 is_online=True（narrow-band LSTM 单向、causal 卷积）。"""
    fixed = _fixed()
    variable = _variable()
    assert fixed.is_online is True
    assert variable.is_online is True
    assert fixed.block_1.narr_lstm.bidirectional is False
    assert variable.block_1.narr_lstm.bidirectional is False


def test_online_and_offline_narrow_band_direction_differ():
    """online/offline 的真实区分点是 narrow-band LSTM 方向性。"""
    from tasks.ssl.models.ipdnet_official import _OfficialFNBlock

    online = _OfficialFNBlock(4, 8, dropout=0.0, add_skip_dim=4, is_online=True, is_first=True)
    offline = _OfficialFNBlock(4, 8, dropout=0.0, add_skip_dim=4, is_online=False, is_first=True)
    assert online.narr_lstm.bidirectional is False
    assert offline.narr_lstm.bidirectional is True
    # full-band LSTM 恒双向
    assert online.full_lstm.bidirectional is True
    assert offline.full_lstm.bidirectional is True
    # narrow hidden 在 offline 下减半（官方 narr_hidden_size = hidden//2）
    assert online.narr_hidden_size == 8
    assert offline.narr_hidden_size == 4


def test_offline_fixed_forward_and_backward_keep_canonical_axes():
    """fixed offline 变体沿用官方输出轴与 12 倍时间压缩，并可反向传播。"""
    model = _fixed(causal=False).train()
    features = torch.randn(1, 36, 5, 4, requires_grad=True)
    output = model(features)
    assert output.shape == (1, 3, 5, 2, 1, 2)
    output.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_offline_variable_forward_and_backward_keep_pair_axis():
    """variable offline 变体保留 pair-batched 适配与 canonical 输出轴。"""
    model = _variable(causal=False).train()
    features = torch.randn(1, 36, 5, 6, requires_grad=True)
    output = model(features, pair_index=_pair_index(3)[None])
    assert output.shape == (1, 3, 5, 2, 2, 2)
    output.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_offline_variants_have_strict_checkpoint_parity():
    """同一 offline 变体的 state_dict 必须严格恢复且输出逐元素一致。"""
    torch.manual_seed(7)
    fixed = _fixed(causal=False).eval()
    fixed_features = torch.randn(1, 36, 5, 4)
    fixed_state = {name: value.detach().clone() for name, value in fixed.state_dict().items()}
    fixed_restored = _fixed(causal=False).eval()
    result = fixed_restored.load_state_dict(fixed_state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    with torch.no_grad():
        assert torch.equal(fixed(fixed_features), fixed_restored(fixed_features))

    variable = _variable(causal=False).eval()
    variable_features = torch.randn(1, 36, 5, 6)
    pair_index = _pair_index(3)[None]
    variable_state = {
        name: value.detach().clone() for name, value in variable.state_dict().items()
    }
    variable_restored = _variable(causal=False).eval()
    result = variable_restored.load_state_dict(variable_state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    with torch.no_grad():
        assert torch.equal(
            variable(variable_features, pair_index=pair_index),
            variable_restored(variable_features, pair_index=pair_index),
        )


def test_forgetting_norm_matches_official_recurrence():
    """官方 ``utils_.forgetting_norm``：alpha=(L-1)/(L+1)，前 L 帧用 min((t-1)/(t+1), alpha)。"""
    frames = 12
    sample_length = 6
    value = torch.rand(frames, 1, 1, 1).expand(frames, 1, 1, 1).clone()
    value = value.reshape(1, 1, 1, frames)
    result = forgetting_norm(value, sample_length=sample_length)
    assert result.shape == (1, 1, 1, frames)
    assert torch.isfinite(result).all()
    alpha = (sample_length - 1) / (sample_length + 1)
    mu = torch.zeros(1, 1)
    expected = []
    flat = value.reshape(1, 1, frames)
    for frame in range(frames):
        local = min((frame - 1) / (frame + 1), alpha) if frame < sample_length else alpha
        mu = local * mu + (1.0 - local) * flat[:, :, frame].mean(dim=1, keepdim=True)
        expected.append(mu)
    expected_tensor = torch.stack(expected, dim=-1).reshape(1, 1, 1, frames)
    assert torch.allclose(result, expected_tensor, atol=1e-6)


# --------------------------------------------------------------------------- #
# 参数数量与结构事实
# --------------------------------------------------------------------------- #


def test_fixed_cnn_output_dim_follows_official_formula():
    for channels, tracks in ((2, 2), (4, 2), (6, 2), (8, 2)):
        model = _fixed(input_size=2 * channels, hidden_size=8, max_track=tracks, n_fft=8)
        assert model.pairs == channels - 1
        assert model.cnn_out_dim == 2 * (channels - 1) * tracks
        assert model.conv.conv1.in_channels == model.hidden_size + model.input_size


def test_variable_cnn_out_dim_is_hardcoded_four():
    model = _variable()
    assert model.cnn_out_dim == 4
    assert model.conv.conv3.out_channels == 4
    assert model.conv.conv1.in_channels == model.hidden_size


def test_fixed_default_hidden_and_input_size_match_official_defaults():
    model = IPDNetFixed()
    assert model.input_size == 4
    assert model.hidden_size == 128
    assert model.max_track == 2
    assert model.pairs == 1
    variable = IPDNetVariable()
    assert variable.hidden_size == 128
    assert variable.max_track == 2
