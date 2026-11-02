"""CoFi-Lite 模型合同与上游结构对齐测试。"""

from __future__ import annotations

import pytest
import torch

from core.losses.build import build_loss
from core.models.build import build_model
from core.models.io_contract import ModelIOContract
from tasks.se.models import register_se_models
from tasks.se.models.discriminative.cofi_lite import CoFiLite

pytestmark = [pytest.mark.unit]

EXPECTED_TOTAL_PARAMS = 108_214
EXPECTED_TRAINABLE_PARAMS = 83_124


def _build_model() -> CoFiLite:
    register_se_models(only="CoFiLite")
    return CoFiLite()


def test_cofi_lite_is_registered_with_waveform_owned_frontend():
    register_se_models(only="CoFiLite")
    model = CoFiLite()
    assert isinstance(model.IO_CONTRACT, ModelIOContract)
    assert model.IO_CONTRACT.frontend_owner == "model"
    assert [field.name for field in model.IO_CONTRACT.inputs] == ["noisy"]
    assert [field.name for field in model.IO_CONTRACT.outputs] == ["enhanced"]
    assert model.IO_CONTRACT.inputs[0].domain == "waveform"
    assert model.IO_CONTRACT.outputs[0].domain == "waveform"
    assert model.target_fs == 16000


def test_cofi_lite_matches_released_parameter_budget():
    model = _build_model()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert total == EXPECTED_TOTAL_PARAMS
    assert trainable == EXPECTED_TRAINABLE_PARAMS
    # 两个固定 ERB 滤波器组是 total 与 trainable 的差值来源。
    assert total - trainable == 25_090
    assert model.ERB.erb_fc.weight.requires_grad is False
    assert model.ERB.ierb_fc.weight.requires_grad is False


def test_cofi_lite_keeps_upstream_state_dict_keys():
    state = _build_model().state_dict()
    for key in (
        "stft_encoder.window",
        "ERB.erb_fc.weight",
        "ERB.ierb_fc.weight",
        "en_erb1.pwconv1.1.weight",
        "spgrnn.rnn.rnn1.weight_ih_l0",
        "inter_rnn1.inter_ln.weight",
        "de_lfb1.tra.att_fc.weight",
        "stft_decoder.window",
    ):
        assert key in state
    assert state["ERB.erb_fc.weight"].shape == (65, 193)
    assert state["ERB.ierb_fc.weight"].shape == (193, 65)


def test_build_model_accepts_matching_stft_context():
    register_se_models(only="CoFiLite")
    model = build_model(
        {"type": "CoFiLite", "args": {}},
        stft_cfg={
            "n_fft": 512,
            "hop_length": 256,
            "win_length": 512,
            "window": "hann",
            "center": True,
        },
    )
    assert isinstance(model, CoFiLite)


def test_build_model_rejects_conflicting_stft_context():
    register_se_models(only="CoFiLite")
    with pytest.raises(ValueError, match="fixes n_fft=512"):
        build_model(
            {"type": "CoFiLite", "args": {}},
            stft_cfg={"n_fft": 256, "hop_length": 128},
        )


@pytest.mark.parametrize("length", [1600, 12345])
def test_cofi_lite_forward_preserves_waveform_length(length):
    model = _build_model().eval()
    noisy = torch.randn(2, length)
    with torch.no_grad():
        enhanced = model(noisy)
    assert enhanced.shape == noisy.shape
    assert enhanced.dtype == torch.float32
    assert torch.isfinite(enhanced).all()


def test_cofi_lite_forward_is_causal_beyond_center_lookahead():
    model = _build_model().eval()
    torch.manual_seed(0)
    prefix_seconds = 0.5
    prefix = torch.randn(1, int(16000 * prefix_seconds))
    first_suffix = torch.randn(1, int(16000 * prefix_seconds))
    second_suffix = torch.randn(1, int(16000 * prefix_seconds))
    first = torch.cat([prefix, first_suffix], dim=1)
    second = torch.cat([prefix, second_suffix], dim=1)

    with torch.no_grad():
        first_out = model(first)
        second_out = model(second)

    lookahead = model.n_fft // 2 + model.hop_length
    boundary = prefix.shape[1] - lookahead
    assert boundary > 0
    torch.testing.assert_close(
        first_out[:, :boundary], second_out[:, :boundary], rtol=0, atol=0
    )


def test_cofi_lite_forward_keeps_fp32_under_cpu_autocast():
    model = _build_model().eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        enhanced = model(torch.zeros(1, 1600))
    assert enhanced.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16])
def test_cofi_lite_normalizes_input_dtype_to_fp32(dtype):
    model = _build_model().eval()
    with torch.no_grad():
        enhanced = model(torch.randn(1, 1000, dtype=dtype))
    assert enhanced.dtype == torch.float32
    assert torch.isfinite(enhanced).all()


def test_cofi_lite_rejects_non_16k_sample_rate():
    model = _build_model().eval()
    with pytest.raises(ValueError, match="requires fs=16000"):
        model(torch.zeros(1, 1600), fs=8000)


def test_cofi_lite_rejects_multichannel_input():
    model = _build_model().eval()
    with pytest.raises(ValueError, match=r"mono waveform \[B, T\]"):
        model(torch.zeros(1, 2, 1600))


def test_cofi_lite_rejects_empty_batch_and_time_dimensions():
    model = _build_model().eval()
    with pytest.raises(ValueError, match="non-empty batch and time dimensions"):
        model(torch.zeros(0, 1600))
    with pytest.raises(ValueError, match="non-empty batch and time dimensions"):
        model(torch.zeros(1, 0))


def test_cofi_lite_rejects_non_floating_input():
    model = _build_model().eval()
    with pytest.raises(TypeError, match="floating point"):
        model(torch.zeros(1, 1600, dtype=torch.int16))


def test_cofi_lite_validates_lengths_contract():
    model = _build_model().eval()
    noisy = torch.zeros(2, 1600)
    with pytest.raises(ValueError, match="lengths must have shape"):
        model(noisy, torch.tensor([[1600], [1600]]))
    with pytest.raises(TypeError, match="integer dtype"):
        model(noisy, torch.tensor([1600.0, 1600.0]))


def test_cofi_lite_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        CoFiLite(typo=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_fft": 256},
        {"hop_length": 128},
        {"win_length": 256},
        {"window": "hamming"},
        {"center": False},
    ],
)
def test_cofi_lite_rejects_non_upstream_stft_geometry(kwargs):
    with pytest.raises(ValueError):
        CoFiLite(**kwargs)


def _upstream_hybrid_loss_reference(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    n_fft: int = 512,
    hop_len: int = 256,
    win_len: int = 512,
    compress_factor: float = 0.3,
    eps: float = 1e-12,
    lamda_ri: float = 30.0,
    lamda_mag: float = 70.0,
) -> torch.Tensor:
    """上游 loss_factory.py::HybridLoss 的逐行参考实现（无 lengths）。"""
    window = torch.hann_window(win_len)
    pred_stft = torch.stft(
        y_pred, n_fft, hop_len, win_len, window, return_complex=True
    )
    true_stft = torch.stft(
        y_true, n_fft, hop_len, win_len, window, return_complex=True
    )
    pred_mag = torch.abs(pred_stft).clamp(eps)
    true_mag = torch.abs(true_stft).clamp(eps)
    pred_stft_c = pred_stft / pred_mag ** (1 - compress_factor)
    true_stft_c = true_stft / true_mag ** (1 - compress_factor)
    real_loss = torch.mean((pred_stft_c.real - true_stft_c.real) ** 2)
    imag_loss = torch.mean((pred_stft_c.imag - true_stft_c.imag) ** 2)
    mag_loss = torch.mean((pred_mag**compress_factor - true_mag**compress_factor) ** 2)
    y_norm = (
        torch.sum(y_true * y_pred, dim=-1, keepdim=True)
        * y_true
        / (torch.sum(y_true.square(), dim=-1, keepdim=True) + 1e-8)
    )
    sisnr = -2 * torch.log10(
        torch.norm(y_norm, dim=-1, keepdim=True)
        / torch.norm(y_pred - y_norm, dim=-1, keepdim=True).clamp(eps)
        + eps
    ).mean()
    return lamda_ri * (real_loss + imag_loss) + lamda_mag * mag_loss + sisnr


def test_cofi_lite_hybrid_loss_matches_upstream_formula():
    torch.manual_seed(0)
    y_pred = torch.randn(2, 4000) * 0.1
    y_true = torch.randn(2, 4000) * 0.1
    loss = build_loss(
        {
            "type": "CoFiLiteHybridLoss",
            "args": {
                "n_fft": 512,
                "hop_length": 256,
                "win_length": 512,
                "compress_factor": 0.3,
                "eps": 1e-12,
                "lamda_ri": 30.0,
                "lamda_mag": 70.0,
            },
        }
    )
    expected = _upstream_hybrid_loss_reference(y_pred, y_true)
    actual = loss(y_pred, y_true)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_cofi_lite_hybrid_loss_accepts_padded_lengths():
    torch.manual_seed(0)
    y_pred = torch.randn(2, 4000) * 0.1
    y_true = torch.randn(2, 4000) * 0.1
    loss = build_loss({"type": "CoFiLiteHybridLoss", "args": {}})
    value = loss(y_pred, y_true, torch.tensor([4000, 3000]))
    assert value.ndim == 0
    assert torch.isfinite(value)
    full_length = loss(y_pred, y_true, torch.tensor([4000, 4000]))
    torch.testing.assert_close(full_length, loss(y_pred, y_true), rtol=0, atol=0)


def test_cofi_lite_hybrid_loss_rejects_wrong_window():
    with pytest.raises(ValueError, match="Hann loss window"):
        build_loss(
            {"type": "CoFiLiteHybridLoss", "args": {"window": "hamming"}}
        )


def test_cofi_lite_hybrid_loss_owns_its_stft_geometry():
    """waveform-native 模型的 loss 不继承顶层 stft 的几何。"""
    loss = build_loss(
        {"type": "CoFiLiteHybridLoss", "args": {}},
        stft_cfg={
            "n_fft": 256,
            "hop_length": 128,
            "win_length": 256,
            "window": "hann",
            "center": True,
        },
    )
    assert (loss.n_fft, loss.hop_length, loss.win_length) == (512, 256, 512)
    assert loss.center is True
