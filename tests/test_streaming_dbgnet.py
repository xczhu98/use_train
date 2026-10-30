"""Unit tests for the ONNX streaming DBGNet wrapper.

These tests verify that ``StreamingDBGNet`` / ``StreamingDBGNetFlat`` produce
frame-by-frame outputs consistent with the full ``DBGNet`` forward pass.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tasks.aec.models.discriminative.dbgnet import DBGNet, _DecoderBlock, _ERB
from onnx.models.streaming_dbgnet import (
    StreamingDBGNet,
    StreamingDBGNetFlat,
    StreamingDBGNetFrontendAdapter,
)


N_FFT = 512
HOP = 256
N_FREQS = N_FFT // 2 + 1


def _random_spec(batch: int = 1, n_frames: int = 20) -> torch.Tensor:
    return torch.randn(batch, N_FREQS, n_frames, 2)


def _small_dbgnet(**kwargs) -> DBGNet:
    args = {
        "n_fft": N_FFT,
        "hop_length": HOP,
        "win_length": N_FFT,
        "sample_rate": 16000,
        "mic_channels": [4, 8, 12, 16],
        "far_channels": [2, 4, 6, 8],
        "use_state_detector": False,
    }
    args.update(kwargs)
    return DBGNet(**args)


@pytest.mark.parametrize(
    ("eps", "error_type"),
    [
        (True, TypeError),
        ("1e-12", TypeError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_dbgnet_rejects_invalid_eps(eps, error_type):
    """eps 必须是有限正实数，且 bool 不按整数兼容。"""
    with pytest.raises(error_type, match="eps must be a finite positive number"):
        _small_dbgnet(eps=eps)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"low_bins": 0, "erb_bands": 64}, "low_bins must be in"),
        ({"low_bins": N_FREQS, "erb_bands": 64}, "low_bins must be in"),
        ({"low_bins": 65, "erb_bands": 1}, "erb_bands must be at least 2"),
        ({"low_bins": 65, "erb_bands": 64, "high_freq": 0}, "high_freq must be finite"),
        ({"low_bins": 65, "erb_bands": 64, "high_freq": 8001}, "high_freq must be finite"),
        ({"low_bins": 65, "erb_bands": 64, "high_freq": 2000}, "must exceed the first ERB"),
        ({"low_bins": 65, "erb_bands": 193}, "strictly increasing"),
    ],
)
def test_erb_rejects_invalid_boundaries_and_repeated_centers(kwargs, match):
    """ERB 边界及取整后重复中心必须在构造期明确失败。"""
    with pytest.raises(ValueError, match=match):
        _ERB(n_fft=N_FFT, sample_rate=16000, **kwargs)


def test_dbgnet_rejects_non_mirrored_decoder_during_construction(monkeypatch):
    """合法 ERB 下，decoder 几何不镜像时不得延迟到 forward 才失败。"""
    original_init = _DecoderBlock.__init__
    construction_count = 0

    def build_non_mirrored_first_decoder(self, *args, **kwargs):
        nonlocal construction_count
        original_init(self, *args, **kwargs)
        construction_count += 1
        if construction_count == 1:
            self.deconv[0].output_padding = (1, 0)

    monkeypatch.setattr(_DecoderBlock, "__init__", build_non_mirrored_first_decoder)
    with pytest.raises(ValueError, match="deconv4 frequency geometry is not the encoder mirror"):
        _small_dbgnet()


def test_offline_spectra_inputs_fail_fast_on_contract_violations():
    """离线谱在进入网络前校验 Tensor、维度、shape、dtype 与 device。"""
    model = _small_dbgnet().eval()
    valid = _random_spec(n_frames=2)

    with pytest.raises(TypeError, match="mic_spec must be a torch.Tensor"):
        model.forward_spectra(np.zeros(valid.shape, dtype=np.float32), valid)
    with pytest.raises(ValueError, match="mic_spec must have rank 4"):
        model.forward_spectra(valid.squeeze(0), valid)
    with pytest.raises(ValueError, match="mic_spec RI dimension must be 2"):
        model.forward_spectra(valid[..., :1], valid)
    with pytest.raises(ValueError, match="mic_spec frequency dimension must be"):
        model.forward_spectra(valid[:, :-1], valid[:, :-1])
    with pytest.raises(ValueError, match="batch and time dimensions must be positive"):
        model.forward_spectra(valid[:0], valid[:0])
    with pytest.raises(ValueError, match="batch and time dimensions must be positive"):
        model.forward_spectra(valid[:, :, :0], valid[:, :, :0])
    with pytest.raises(ValueError, match="must have identical .* shapes"):
        model.forward_spectra(valid, valid.expand(2, -1, -1, -1))
    with pytest.raises(TypeError, match="must use a floating-point RI dtype"):
        integer_spec = valid.to(torch.int32)
        model.forward_spectra(integer_spec, integer_spec)
    with pytest.raises(TypeError, match="must have the same dtype"):
        model.forward_spectra(valid, valid.double())
    with pytest.raises(TypeError, match="spectra dtype .* must match model dtype"):
        model.forward_spectra(valid.double(), valid.double())

    meta_spec = torch.empty(valid.shape, device="meta")
    with pytest.raises(ValueError, match="must be on the same device"):
        model.forward_spectra(valid, meta_spec)
    with pytest.raises(ValueError, match="spectra device .* must match model device"):
        model.forward_spectra(meta_spec, meta_spec)


def test_legacy_onnx_export_skips_only_python_shape_guards(monkeypatch):
    """Legacy export 依赖 static flat ABI/runtime preflight，不固化 Python shape bool。"""
    model = _small_dbgnet().eval()
    stream = StreamingDBGNet(model).eval()
    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)

    rank_three = _random_spec(n_frames=1).squeeze(0)
    model._validate_spectra_inputs(rank_three, rank_three)
    two_frames = _random_spec(n_frames=2)
    stream._validate_streaming_inputs(two_frames, two_frames)
    with pytest.raises(TypeError, match="floating-point RI dtype"):
        model._validate_spectra_inputs(rank_three.int(), rank_three.int())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_magnitude_clamps_to_dtype_finite_max(dtype):
    """FP16/BF16 幅度在转回目标 dtype 前截到其最大有限值。"""
    model = _small_dbgnet().to(dtype=dtype).eval()
    dtype_max = torch.finfo(dtype).max
    spec = torch.full((1, 1, 1, 2), dtype_max, dtype=dtype)
    if dtype == torch.bfloat16:
        assert torch.isinf(spec.float().square()).all()

    magnitude = model._magnitude(spec)

    assert magnitude.dtype == dtype
    assert torch.isfinite(magnitude).all()
    torch.testing.assert_close(
        magnitude, torch.full_like(magnitude, dtype_max), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_full_low_precision_network_is_finite_and_streaming_consistent(dtype):
    """固定 RNG 的常规动态范围低精度网络须有限，单帧流式须匹配离线。"""
    torch.manual_seed(20260831)
    model = _small_dbgnet().to(dtype=dtype).eval()
    stream = StreamingDBGNet(model).eval()
    mic = (torch.randn(1, N_FREQS, 1, 2) * 2.0).to(dtype=dtype)
    far = (torch.randn_like(mic) * 2.0).to(dtype=dtype)

    with torch.inference_mode():
        offline = model.forward_spectra(mic, far)
        streaming, state_logits, _ = stream.forward_streaming(
            mic, far, stream.init_cache(1, "cpu", dtype)
        )

    assert state_logits is None
    assert torch.isfinite(offline).all()
    assert torch.isfinite(streaming).all()
    torch.testing.assert_close(streaming, offline, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [(torch.float16, 1000.0), (torch.bfloat16, 1e15)],
)
def test_extreme_low_precision_output_fails_closed_offline_and_streaming(dtype, value):
    """有限但极端的低精度输入不得从离线或 Python 流式接口静默返回 NaN。"""
    torch.manual_seed(0)
    model = _small_dbgnet().to(dtype=dtype).eval()
    stream = StreamingDBGNet(model).eval()
    flat = StreamingDBGNetFlat(stream).eval()
    mic = torch.full((1, N_FREQS, 1, 2), value, dtype=dtype)
    far = torch.full_like(mic, value)
    assert torch.isfinite(mic).all()
    assert torch.isfinite(far).all()

    with torch.inference_mode():
        with pytest.raises(FloatingPointError, match="non-finite enhanced spectrogram"):
            model.forward_spectra(mic, far)
        with pytest.raises(FloatingPointError, match="non-finite enhanced spectrogram"):
            stream.forward_streaming(
                mic, far, stream.init_cache(1, "cpu", dtype)
            )
        with pytest.raises(FloatingPointError, match="non-finite enhanced spectrogram"):
            flat(mic, far, *flat.init_cache(1, "cpu", dtype))


def test_dbg_mag_output_is_bounded_real_gain_with_microphone_phase():
    """真实 DBG-Mag 前向仅施加 [0,1] 实值 gain，并保持麦克风相位。"""
    model = _small_dbgnet().eval()
    mic = _random_spec(n_frames=3)
    far = _random_spec(n_frames=3)
    mic[..., 0].add_(0.25)
    mic[..., 1].add_(0.5)

    with torch.inference_mode():
        enhanced = model.forward_spectra(mic, far)

    denominator = mic[..., 0].square() + mic[..., 1].square()
    gain_real = (
        enhanced[..., 0] * mic[..., 0] + enhanced[..., 1] * mic[..., 1]
    ) / denominator
    gain_imag = (
        enhanced[..., 1] * mic[..., 0] - enhanced[..., 0] * mic[..., 1]
    ) / denominator
    torch.testing.assert_close(gain_imag, torch.zeros_like(gain_imag), rtol=0, atol=1e-6)
    assert torch.all(gain_real >= 0)
    assert torch.all(gain_real <= 1)


def test_offline_encoders_consume_causal_padded_recursive_guidance():
    """第 2～4 层双分支 encoder 必须消费前一级 SLM guidance。"""
    model = DBGNet(
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        sample_rate=16000,
        mic_channels=[4, 8, 12, 16],
        far_channels=[2, 4, 6, 8],
        use_state_detector=True,
        state_nclass=3,
    ).eval()
    observed = {}

    def capture_slm(level):
        def hook(_module, _inputs, output):
            observed[f"gy{level}"], observed[f"gx{level}"], _ = output

        return hook

    def capture_conv_input(name):
        def hook(_module, inputs):
            observed[name] = inputs[0]

        return hook

    handles = []
    for level in range(1, 4):
        handles.extend(
            [
                getattr(model, f"slm{level}").register_forward_hook(capture_slm(level)),
                getattr(model, f"conv_mic{level + 1}").register_forward_pre_hook(
                    capture_conv_input(f"conv_mic{level + 1}")
                ),
                getattr(model, f"conv_far{level + 1}").register_forward_pre_hook(
                    capture_conv_input(f"conv_far{level + 1}")
                ),
            ]
        )
    try:
        with torch.inference_mode():
            model.forward_spectra(_random_spec(n_frames=5), _random_spec(n_frames=5))
    finally:
        for handle in handles:
            handle.remove()

    for level in range(1, 4):
        for branch, guidance in (("mic", "gy"), ("far", "gx")):
            guided = observed[f"{guidance}{level}"]
            expected = torch.cat([torch.zeros_like(guided[..., :1]), guided], dim=-1)
            torch.testing.assert_close(
                observed[f"conv_{branch}{level + 1}"], expected, rtol=0, atol=0
            )


def test_all_enhancement_parameters_receive_finite_gradients_without_state_detector():
    """关闭 state detector 后，全部增强参数均在有效计算图中。"""
    model = _small_dbgnet().eval()
    guiding_modules = [getattr(model, f"slm{level}").guiding_conv_far for level in range(1, 5)]

    enhanced = model.forward_spectra(_random_spec(n_frames=5), _random_spec(n_frames=5))
    enhanced.square().mean().backward()

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing = [name for name, parameter in trainable.items() if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in trainable.items()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert not missing, f"enhancement parameters without gradients: {missing}"
    assert not nonfinite, f"enhancement parameters with non-finite gradients: {nonfinite}"

    key_groups = {
        "far_frontend": ("fpb.", "conv_far", "ftg"),
        "mic_encoder": ("conv_mic",),
        "slm": ("slm",),
        "bottleneck": ("fgru.", "tgru."),
        "decoder": ("deconv", "out_conv."),
    }
    for group, prefixes in key_groups.items():
        gradients = [
            parameter.grad
            for name, parameter in trainable.items()
            if name.startswith(prefixes)
        ]
        assert gradients, f"no parameters found for enhancement group {group}"
        assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients), (
            f"enhancement group {group} has only zero gradients"
        )

    for level, module in enumerate(guiding_modules, start=1):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0, (
            f"slm{level}.guiding_conv_far has only zero gradients"
        )


@pytest.mark.parametrize("use_state_detector", [False, True])
def test_recursive_topology_key_gates_strict_checkpoint_loading(use_state_detector):
    """新 checkpoint 可严格加载，旧拓扑因缺少版本化 key 而失败。"""
    model = DBGNet(use_state_detector=use_state_detector)
    state = model.state_dict()
    DBGNet(use_state_detector=use_state_detector).load_state_dict(state, strict=True)

    legacy_state = dict(state)
    legacy_state.pop("_recursive_topology_v1")
    with pytest.raises(RuntimeError, match="_recursive_topology_v1"):
        model.load_state_dict(legacy_state, strict=True)


def test_streaming_matches_full_model():
    """逐帧流式输出必须与离线整段输出达到 fp32 舍入级一致。"""
    model = DBGNet(
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        sample_rate=16000,
        use_state_detector=True,
        state_nclass=8,
    ).eval()
    stream = StreamingDBGNet(model).eval()

    n_frames = 20
    mic = _random_spec(n_frames=n_frames)
    far = _random_spec(n_frames=n_frames)

    with torch.inference_mode():
        full_enh, full_state = model.forward_spectra(mic, far)

    cache = stream.init_cache(1, "cpu", torch.float32)
    enh_frames = []
    state_frames = []
    with torch.inference_mode():
        for t in range(n_frames):
            enh_t, state_t, cache = stream.forward_streaming(
                mic[:, :, t : t + 1, :], far[:, :, t : t + 1, :], cache
            )
            enh_frames.append(enh_t)
            state_frames.append(state_t)

    stream_enh = torch.cat(enh_frames, dim=2)
    stream_state = torch.cat(state_frames, dim=1)

    # 流式逐帧 GRU 与离线整段计算只允许 fp32 舍入级差异。
    torch.testing.assert_close(stream_enh, full_enh, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(stream_state, full_state, rtol=1e-5, atol=1e-6)


def test_streaming_32k_small_matches_full_model_and_derives_erb_cache():
    """32 kHz Small 的 321-bin 输入和 81+80 ERB cache 必须保持流式一致。"""
    n_fft = 640
    model = DBGNet(
        n_fft=n_fft,
        hop_length=320,
        win_length=n_fft,
        sample_rate=32000,
        erb_low_bins=81,
        erb_bands=80,
        mic_channels=[16, 32, 48, 64],
        far_channels=[8, 16, 24, 32],
        use_state_detector=True,
        state_nclass=3,
    ).eval()
    stream = StreamingDBGNet(model).eval()

    n_frames = 6
    mic = torch.randn(1, n_fft // 2 + 1, n_frames, 2)
    far = torch.randn_like(mic)
    with torch.inference_mode():
        full_enh, full_state = model.forward_spectra(mic, far)

    cache = stream.init_cache(1, "cpu", torch.float32)
    assert stream.target_fs == 32000
    assert stream.nfreqs == 321
    assert cache["mic_in"].shape == (1, 1, 161, 1)
    assert cache["far_in"].shape == (1, 1, 161, 1)

    enh_frames = []
    state_frames = []
    with torch.inference_mode():
        for frame in range(n_frames):
            enh_t, state_t, cache = stream.forward_streaming(
                mic[:, :, frame : frame + 1, :],
                far[:, :, frame : frame + 1, :],
                cache,
            )
            enh_frames.append(enh_t)
            state_frames.append(state_t)

    stream_enh = torch.cat(enh_frames, dim=2)
    stream_state = torch.cat(state_frames, dim=1)
    assert stream_enh.shape == (1, 321, n_frames, 2)
    assert stream_state.shape == (1, n_frames, 3)
    torch.testing.assert_close(stream_enh, full_enh, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(stream_state, full_state, rtol=1e-5, atol=1e-6)


def test_streaming_without_state_detector():
    """Streaming wrapper works when the auxiliary state head is disabled."""
    model = DBGNet(
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        sample_rate=16000,
        use_state_detector=False,
    ).eval()
    stream = StreamingDBGNet(model).eval()
    flat = StreamingDBGNetFlat(stream).eval()

    mic = _random_spec(n_frames=1)
    far = _random_spec(n_frames=1)
    cache = flat.init_cache(1, "cpu", torch.float32)

    with torch.inference_mode():
        out = flat(mic, far, *cache)

    assert out[0].shape == mic.shape
    # No state_probs output when state detector is disabled.
    assert len(out) == len(cache) + 1


def test_flat_cache_io_consistency():
    """The flat wrapper's input/output cache keys should be aligned."""
    model = DBGNet(
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        sample_rate=16000,
        use_state_detector=True,
        state_nclass=8,
    ).eval()
    flat = StreamingDBGNetFlat(StreamingDBGNet(model))

    input_names = flat.get_input_names()
    output_names = flat.get_output_names()

    assert input_names[:2] == ["mic_spec", "far_spec"]
    assert output_names[0] == "enh"
    assert output_names[1] == "state_probs"

    # Each cache input has a corresponding _out output.
    cache_keys = flat._cache_keys
    assert tuple(cache_keys) == flat.CACHE_KEYS
    assert tuple(cache_keys) == tuple(flat.inner.init_cache(1, "cpu", torch.float32))
    assert input_names[2:] == cache_keys
    assert cache_keys[2:8] == [
        "gy1_in", "gy2_in", "gy3_in", "gx1_in", "gx2_in", "gx3_in",
    ]
    assert output_names[2:] == [f"{k}_out" for k in cache_keys]


def test_init_cache_shapes():
    """Cache shapes must match the expected encoder/GRU dimensions."""
    model = DBGNet(
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        sample_rate=16000,
        use_state_detector=True,
        state_nclass=8,
    ).eval()
    stream = StreamingDBGNet(model)

    batch_sizes = [1, 2]
    for b in batch_sizes:
        cache = stream.init_cache(b, "cpu", torch.float32)
        assert cache["mic_in"].shape == (b, 1, 129, 1)
        assert cache["far_in"].shape == (b, 1, 129, 1)
        assert cache["gy3_in"].shape == (b, 96, 31, 1)
        assert cache["gx3_in"].shape == (b, 48, 31, 1)
        assert cache["ftg_h4"].shape == (1, b * 64, 15)
        assert cache["tgru_h"].shape == (1, b * 15, 128)
        assert cache["state_h"].shape == (1, b, 128)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_streaming_rejects_non_positive_batch_size(batch_size):
    """流式 cache 仅接受正 batch。"""
    stream = StreamingDBGNet(DBGNet(use_state_detector=False).eval())
    with pytest.raises(ValueError, match="positive integer"):
        stream.init_cache(batch_size, "cpu", torch.float32)


@pytest.mark.parametrize("n_frames", [0, 2])
def test_streaming_rejects_non_single_frame_input(n_frames):
    """流式 forward 对空帧和多帧输入均立即拒绝。"""
    stream = StreamingDBGNet(DBGNet(use_state_detector=False).eval())
    cache = stream.init_cache(1, "cpu", torch.float32)
    with pytest.raises(ValueError, match="positive|exactly one frame"):
        stream.forward_streaming(
            _random_spec(n_frames=n_frames), _random_spec(n_frames=n_frames), cache
        )


def test_flat_deployment_contract_is_static_batch_one():
    """Python cache math may batch, but formal flat ONNX deployment may not."""
    model = DBGNet(
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        sample_rate=16000,
        use_state_detector=True,
        state_nclass=8,
    ).eval()
    stream = StreamingDBGNet(model)
    flat = StreamingDBGNetFlat(stream)

    assert stream.init_cache(2, "cpu", torch.float32)["mic_in"].shape[0] == 2
    assert flat.get_dynamic_axes() == {}
    with pytest.raises(ValueError, match="only supports batch_size=1"):
        flat.init_cache(2, "cpu", torch.float32)
    mic = _random_spec(batch=2, n_frames=1)
    far = _random_spec(batch=2, n_frames=1)
    cache = stream.init_cache(2, "cpu", torch.float32)
    with pytest.raises(ValueError, match="only supports batch_size=1"):
        flat(mic, far, *cache.values())


def test_frontend_adapter_reuses_three_return_streaming_api():
    """通用 frontend adapter 只丢弃 state logits，不改变增强与 cache。"""
    model = DBGNet(
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        sample_rate=16000,
        use_state_detector=True,
        state_nclass=8,
    ).eval()
    stream = StreamingDBGNet(model).eval()
    adapter = StreamingDBGNetFrontendAdapter(stream).eval()
    mic = _random_spec(n_frames=1)
    far = _random_spec(n_frames=1)

    direct_cache = stream.init_cache(1, "cpu", torch.float32)
    adapter_cache = adapter.init_cache(1, "cpu", torch.float32)
    with torch.inference_mode():
        direct_enh, state_logits, direct_updated = stream.forward_streaming(
            mic, far, direct_cache
        )
        adapter_enh, adapter_updated = adapter(mic, far, adapter_cache)

    assert state_logits is not None
    assert tuple(adapter.CACHE_KEYS) == tuple(adapter_updated)
    assert tuple(adapter.CACHE_KEYS) == tuple(direct_updated)
    torch.testing.assert_close(adapter_enh, direct_enh)
    for key in adapter.CACHE_KEYS:
        torch.testing.assert_close(adapter_updated[key], direct_updated[key])


pytestmark = [pytest.mark.unit]
