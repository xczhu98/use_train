"""TSE 模型残留审计问题的回归测试。"""

from __future__ import annotations

import pytest
import torch

from tasks.tse.model_io import TSEStreamingState
from tasks.tse.models.discriminative import tse_bsrnn_spk
from tasks.tse.models.discriminative.tse_bsrnn_spk import TSE_BSRNN_SPK
from tasks.tse.models.modules.fusion.speech import CrossFuse
from tasks.tse.models.modules.separator.bsrnn import BSRNN

pytestmark = pytest.mark.unit


_N_FFT = 512
_FREQ = _N_FFT // 2 + 1


def _causal_spkemb_model() -> TSE_BSRNN_SPK:
    return TSE_BSRNN_SPK(
        sr=16000,
        feature_dim=16,
        num_repeat=1,
        causal=True,
        joint_training=False,
        features={"spkemb": {"enabled": True}},
        n_fft=_N_FFT,
        hop_length=128,
        win_length=_N_FFT,
        window="hann",
        center=True,
    ).eval()


def _loader_kwargs() -> dict:
    return {
        "sr": 16000,
        "feature_dim": 16,
        "num_repeat": 1,
        "n_fft": _N_FFT,
        "hop_length": 128,
        "win_length": _N_FFT,
        "window": "hann",
        "center": True,
    }


def test_strict_pretrained_loader_rejects_official_training_head(monkeypatch) -> None:
    """strict=True 不得忽略官方训练头的额外键。"""
    kwargs = _loader_kwargs()
    source = TSE_BSRNN_SPK(**kwargs)
    state = dict(source.state_dict())
    state["loss.weight"] = torch.ones(1)
    monkeypatch.setattr(tse_bsrnn_spk, "safe_torch_load", lambda *args, **kw: state)

    with pytest.raises(ValueError, match="未知键"):
        TSE_BSRNN_SPK.load_pretrained("unused.pt", strict=True, **kwargs)

    loaded = TSE_BSRNN_SPK.load_pretrained("unused.pt", strict=False, **kwargs)
    for key, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_bsrnn_rejects_any_non_positive_band_width() -> None:
    """高采样率下首批零宽子带也必须在构造期拒绝。"""
    with pytest.raises(ValueError, match="非正宽度.*\(0, 0\)"):
        BSRNN(sr=192000, win=_N_FFT, stride=128, feature_dim=16, num_repeat=1)


def test_crossfuse_guard_counts_attention_heads() -> None:
    """多头注意力矩阵超过阈值时，不能遗漏 head 维度。"""
    layer = CrossFuse(
        embed_dim=8,
        atten_dim=8,
        mix_dim=8,
        num_heads=8,
        batch_first=True,
    )
    with pytest.raises(ValueError, match="16,000,000"):
        layer(torch.zeros(1, 8, 2000), torch.zeros(1, 8, 1001))


def test_streaming_enrollment_tensors_are_isolated_from_callers() -> None:
    """初始化输入与公开 enrollment 读取结果都不得污染私有缓存。"""
    source = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    state = TSEStreamingState(
        batch_size=1,
        enrollment={"spkemb": source},
        separator_states=(),
    )
    source.add_(10)
    exposed = state.enrollment["spkemb"]
    exposed.add_(20)

    assert torch.equal(state.enrollment["spkemb"], torch.arange(4).reshape(1, 4))


def test_streaming_sample_rate_requires_integer_scalar() -> None:
    """流式路径应与整句路径同样拒绝浮点 sample_rate。"""
    model = _causal_spkemb_model()
    state = model.init_streaming_state({"spk_embedding": torch.zeros(1, 192)})
    batch = {
        "mix_spectra": torch.zeros(1, _FREQ, 1, 2),
        "sample_rate": 16000.0,
    }
    with pytest.raises(TypeError, match="integer scalar"):
        model.forward_stream_chunk(batch, state)
