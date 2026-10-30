"""Tests for DeepVQE strict kwargs validation (拼写错误不得静默吞掉）。"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import pytest
import torch

from tasks.aec.models.discriminative.deepvqe import DeepVQE


def test_unknown_kwarg_raises():
    """compresson 拼写错误必须显式 TypeError，不得以默认 compression=0.3 悄悄构造。"""
    with pytest.raises(TypeError, match="compresson"):
        DeepVQE(variant="small", compresson=0.9)


def test_multiple_unknown_kwargs_listed():
    with pytest.raises(TypeError) as exc_info:
        DeepVQE(variant="small", compresson=0.9, windo="hann")
    msg = str(exc_info.value)
    assert "compresson" in msg and "windo" in msg


def test_build_model_injected_stft_keys_accepted():
    """build_model 注入的 STFT 键不报错（state detector 字段不再被容忍）。"""
    m = DeepVQE(
        variant="small",
        n_fft=320,
        hop_length=160,
        win_length=320,
        window="hann",
        center=True,
    )
    assert m.n_fft == 320
    assert m.hop_length == 160


@pytest.mark.parametrize("state_nclass", [None, 0, 3, 8])
def test_any_explicit_state_nclass_is_rejected(state_nclass):
    """state_nclass 是 DBGNet 专属字段；任何显式取值都必须报错，不再容忍 0/None。"""
    with pytest.raises(ValueError, match="does not support state_nclass"):
        DeepVQE(variant="small", state_nclass=state_nclass)


@pytest.mark.parametrize("use_state_detector", [False, True])
def test_any_explicit_use_state_detector_is_rejected(use_state_detector):
    """use_state_detector 显式传入即报错，不再静默容忍 False。"""
    with pytest.raises(ValueError, match="does not support use_state_detector"):
        DeepVQE(variant="small", use_state_detector=use_state_detector)


def test_explicit_known_args_still_work():
    m = DeepVQE(variant="small", align_delay=50, compression=0.5)
    assert m.fe.c == 0.5
    assert m.align.unfold[1].kernel_size[0] == 50


def test_small_forward_uses_declared_two_spectra_contract():
    m = DeepVQE(variant="small", n_fft=32, hop_length=16, align_delay=3).eval()
    mic = torch.randn(1, 17, 8, 2)
    far = torch.randn_like(mic)
    with torch.inference_mode():
        output = m({"mic_spectra": mic, "farend_spectra": far})
    assert output["nearend_estimate_spectra"].shape == mic.shape


def test_build_model_injects_center_from_stft():
    """build_model 从 stft 块注入 center（显式 center 时模型跟随，不报错）。"""
    from core.models.build import build_model

    m = build_model(
        {"path": "tasks.aec.models.discriminative.deepvqe.DeepVQE", "args": {"variant": "small"}},
        stft_cfg={"n_fft": 320, "hop_length": 160, "win_length": 320, "window": "hann", "center": True},
    )
    assert m.n_fft == 320
    assert m.hop_length == 160


def test_build_model_omits_center_when_stft_not_declared():
    """stft 块未显式声明 center 时不注入 center，模型构建不受影响。"""
    from core.models.build import build_model

    m = build_model(
        {"path": "tasks.aec.models.discriminative.deepvqe.DeepVQE", "args": {"variant": "small"}},
        stft_cfg={"n_fft": 320, "hop_length": 160, "win_length": 320, "window": "hann"},
    )
    assert m.n_fft == 320
    assert m.hop_length == 160


def test_build_model_omits_none_stft_keys():
    """stft 块缺 n_fft/hop 等键时 build_model 不注入 None（与 build_loss 对齐）。"""
    from core.models.build import build_model

    m = build_model(
        {"path": "tasks.aec.models.discriminative.deepvqe.DeepVQE", "args": {"variant": "small"}},
        stft_cfg={"window": "hann", "center": True},
    )
    # stft_cfg 未提供 n_fft/hop_length → 不注入 None，模型使用默认 320/160
    assert m.n_fft == 320
    assert m.hop_length == 160


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_fft": 0}, "n_fft"),
        ({"hop_length": 0}, "hop_length"),
        ({"align_delay": 0}, "align_delay"),
        ({"compression": 0.0}, "compression"),
        ({"compression": 1.1}, "compression"),
    ],
)
def test_invalid_constructor_parameters_fail_fast(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DeepVQE(variant="small", **kwargs)


def test_forward_requires_dict_and_required_keys():
    model = DeepVQE(variant="small", n_fft=32, align_delay=2)
    with pytest.raises(TypeError, match="dict batch"):
        model(torch.zeros(1, 17, 2, 2))
    with pytest.raises(KeyError, match="farend_spectra"):
        model({"mic_spectra": torch.zeros(1, 17, 2, 2)})


def test_forward_rejects_shape_and_frequency_mismatch():
    model = DeepVQE(variant="small", n_fft=32, align_delay=2)
    mic = torch.zeros(1, 17, 3, 2)
    with pytest.raises(ValueError, match="shapes must match"):
        model.forward_spectra(mic, torch.zeros(1, 17, 2, 2))
    with pytest.raises(ValueError, match=r"n_fft//2\+1=17"):
        model.forward_spectra(
            torch.zeros(1, 16, 3, 2), torch.zeros(1, 16, 3, 2)
        )
    with pytest.raises(ValueError, match=r"\[B, F, T, 2\]"):
        model.forward_spectra(torch.zeros(1, 17, 3), torch.zeros(1, 17, 3))


def test_forward_rejects_dtype_mismatch():
    model = DeepVQE(variant="small", n_fft=32, align_delay=2)
    with pytest.raises(TypeError, match="dtypes must match"):
        model.forward_spectra(
            torch.zeros(1, 17, 3, 2, dtype=torch.float32),
            torch.zeros(1, 17, 3, 2, dtype=torch.float64),
        )


pytestmark = [pytest.mark.unit]
