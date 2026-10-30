"""CAGCRN 离线/流式结构与三路验证 gate 的单元测试。"""

from __future__ import annotations

import pytest
import torch

import tasks.aec.models.discriminative.cagcrn as cagcrn_module
from tasks.aec.models.discriminative.cagcrn import CATA, CAGCRN, TFAG
from onnx.models.streaming_cagcrn import (
    StreamCATA,
    StreamingCAGCRN,
    StreamingCAGCRNFlat,
)
from onnx.verify.verify_3way_cagcrn import (
    aligned_comparison_slices,
    comparison_status,
)


pytestmark = pytest.mark.unit


def test_cata_reference_attention_normalizes_delay_axis(monkeypatch):
    """CATA reference score 必须沿 d=11 的 sliding window 轴归一化。"""
    cata = CATA(in_channels=12, hidden_channels=12).eval()
    seen = []
    original_softmax = cagcrn_module.F.softmax

    def capture_softmax(input_tensor, dim=None, _stacklevel=3, dtype=None):
        output = original_softmax(input_tensor, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
        if input_tensor.ndim == 5:
            seen.append((dim, output.detach()))
        return output

    monkeypatch.setattr(cagcrn_module.F, "softmax", capture_softmax)
    with torch.inference_mode():
        cata(torch.randn(2, 12, 1, 65), torch.randn(2, 12, 1, 65))

    assert seen
    assert all(dim == -2 for dim, _ in seen)
    assert torch.allclose(seen[0][1].sum(dim=-2), torch.ones_like(seen[0][1][..., 0, :]), atol=1e-6)


def test_offline_cata_matches_stream_cata_for_single_frame():
    """单帧、eval 模式下，流式 zero cache 应与离线 CATA 的左 padding 一致。"""
    torch.manual_seed(0)
    cata = CATA(in_channels=12, hidden_channels=12, delay_frames=6).eval()
    stream_cata = StreamCATA(cata).eval()
    mic = torch.randn(2, 12, 1, 65)
    ref = torch.randn(2, 12, 1, 65)
    k_cache, v_cache = stream_cata.init_cache(2, "cpu", mic.dtype)

    with torch.inference_mode():
        offline_out, offline_mic = cata(mic, ref)
        stream_out, stream_mic, _, _ = stream_cata(mic, ref, k_cache, v_cache)

    assert stream_cata.d == 6
    assert k_cache.shape[2] == v_cache.shape[2] == 5
    assert torch.allclose(offline_out, stream_out, atol=1e-6, rtol=1e-5)
    assert torch.allclose(offline_mic, stream_mic, atol=1e-6, rtol=1e-5)


def test_tfag_does_not_consume_future_frames():
    """论文声明的 causal TFAG 不得让未来帧改变当前及历史输出。"""
    torch.manual_seed(1)
    tfag = TFAG(24).eval()
    x1 = torch.randn(1, 24, 12, 33)
    x2 = torch.randn(1, 24, 12, 33)
    changed = x1.clone()
    changed[:, :, 8:] += 100.0

    with torch.inference_mode():
        baseline = tfag(x1, x2)
        perturbed = tfag(changed, x2)

    assert torch.allclose(baseline[:, :, :8], perturbed[:, :, :8], atol=1e-6, rtol=1e-5)


def test_streaming_cagcrn_preserves_stft_delay_and_batch_cache_contract():
    """STFT、两类 delay 与 flat cache batch 语义必须来自 wrapper 权威属性。"""
    base = CAGCRN(
        n_fft=320,
        hop_length=160,
        win_length=320,
        window="hann",
        center=False,
        sample_rate=16000,
        align_delay=6,
    ).eval()
    stream = StreamingCAGCRN(base).eval()
    flat = StreamingCAGCRNFlat(stream).eval()

    assert (
        stream.n_fft,
        stream.hop_length,
        stream.win_length,
        stream.window,
        stream.center,
        stream.target_fs,
    ) == (320, 160, 320, "hann", False, 16000)
    assert flat.delay_d == 6
    assert flat.algorithmic_delay_frames == 0

    cache_one = flat.init_cache_flat(batch_size=1, device="cpu", dtype=torch.float32)
    cache_two = flat.init_cache_flat(batch_size=2, device="cpu", dtype=torch.float32)
    assert len(cache_one) == len(cache_two) == 40
    assert flat.get_input_names()[2:] == [f"cache_{index}" for index in range(40)]
    assert flat.get_output_names()[1:] == [f"out_cache_{index}" for index in range(40)]
    axes = flat.get_dynamic_axes()
    for index, (batch_one, batch_two) in enumerate(zip(cache_one, cache_two)):
        name = f"cache_{index}"
        batch_axis = next(
            axis for axis, label in axes[name].items()
            if label in {"batch_size", "batch_flattened"}
        )
        assert batch_two.shape[batch_axis] == 2 * batch_one.shape[batch_axis]

    assert cache_two[-1].shape == (2, 161, 9, 2)


def test_streaming_cagcrn_matches_offline_without_tail_flush():
    """因果 TFAG 应逐帧对齐离线结果，最后一帧无需额外零帧 flush。"""
    torch.manual_seed(2)
    base = CAGCRN(center=False, align_delay=6).eval()
    stream = StreamingCAGCRN(base).eval()
    mic = torch.randn(1, 257, 12, 2)
    ref = torch.randn(1, 257, 12, 2)
    cache = stream.init_cache(batch_size=1, device="cpu", dtype=mic.dtype)

    with torch.inference_mode():
        offline = base._net(mic, ref)
        frames = []
        for frame_index in range(mic.shape[2]):
            frame, cache = stream(
                mic[:, :, frame_index : frame_index + 1],
                ref[:, :, frame_index : frame_index + 1],
                cache,
            )
            frames.append(frame)
        streamed = torch.cat(frames, dim=2)

    assert streamed.shape == offline.shape
    assert torch.allclose(streamed, offline, atol=1e-5, rtol=1e-4)
    assert torch.allclose(streamed[:, :, -1:], offline[:, :, -1:], atol=1e-5, rtol=1e-4)


def test_cagcrn_rejects_unknown_model_kwargs():
    """未知模型参数必须 fail-fast，不能吞掉 YAML 拼写错误。"""
    import pytest

    with pytest.raises(TypeError, match="unexpected keyword"):
        CAGCRN(typo_stft_field=123)


def test_short_verification_sequence_cannot_pass_empty_comparison():
    """T<=warmup 时 A/C 必须 FAIL，不能把空比较的零 MSE 当作 PASS。"""
    offline_slice, stream_slice, valid_frames = aligned_comparison_slices(20, 20, 4)

    assert offline_slice.start == offline_slice.stop
    assert stream_slice.start == stream_slice.stop
    assert valid_frames == 0
    assert comparison_status(0.0, 1e-5, valid_frames) == "FAIL"


def test_verification_slices_keep_algorithmic_delay_independent_from_warmup():
    """有效比较区间应分别应用 warmup 和 algorithmic delay。"""
    offline_slice, stream_slice, valid_frames = aligned_comparison_slices(30, 20, 4)

    assert (offline_slice.start, offline_slice.stop) == (16, 26)
    assert (stream_slice.start, stream_slice.stop) == (20, 30)
    assert valid_frames == 10
