"""框架 STFT/iSTFT 参数与重建覆盖契约测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.stft import StftIstftModule, build_stft_istft
from core.utils.config import load_merged_config

pytestmark = [pytest.mark.unit]

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_fft": True, "hop_length": 16, "win_length": 64},
        {"n_fft": 64, "hop_length": 65, "win_length": 64},
        {"n_fft": 64, "hop_length": 16, "win_length": 65},
        {"n_fft": 64, "hop_length": 16, "win_length": 64, "window": "blackman"},
        {"n_fft": 64, "hop_length": 16, "win_length": 64, "center": 1},
        {"n_fft": 64, "hop_length": 16, "win_length": 64, "normalized": 0},
        {
            "n_fft": 64,
            "hop_length": 16,
            "win_length": 64,
            "window": "hann",
            "center": False,
        },
        {
            "n_fft": 64,
            "hop_length": 16,
            "win_length": 48,
            "window": "hamming",
            "center": False,
        },
        {
            "n_fft": 64,
            "hop_length": 64,
            "win_length": 64,
            "window": "hann",
            "center": True,
        },
    ],
)
def test_direct_constructor_reuses_strict_stft_validation(kwargs):
    """直接构造不能绕过 build/resolve 使用的参数与 NOLA 校验。"""
    with pytest.raises(ValueError):
        StftIstftModule(**kwargs)
    with pytest.raises(ValueError):
        build_stft_istft(kwargs)


@pytest.mark.parametrize("normalized", [False, True])
def test_non_hop_aligned_nonzero_round_trip_preserves_tail(normalized):
    """两种 normalization 模式都应完整重建非 hop 对齐的非零波形。"""
    module = StftIstftModule(
        n_fft=64,
        hop_length=16,
        win_length=64,
        window="hann",
        center=True,
        normalized=normalized,
    )
    waveform = torch.randn(2, 181, generator=torch.Generator().manual_seed(17))

    restored = module.inverse(module(waveform), waveform.shape[-1])

    assert module.normalized is normalized
    assert torch.count_nonzero(waveform[:, -5:]) > 0
    assert torch.count_nonzero(restored[:, -5:]) > 0
    torch.testing.assert_close(restored, waveform, atol=1e-5, rtol=1e-5)


def test_gtcrn_direct_ratio_build_inherits_normalized_framework_stft():
    """direct-ratio 继承顶层 STFT 完整上下文，normalized=True 合法且落盘。"""
    from core.losses.build import build_loss

    loss = build_loss(
        {"type": "GTCRNLogRatioLoss"},
        stft_cfg={
            "n_fft": 64,
            "hop_length": 16,
            "win_length": 64,
            "window": "hann",
            "center": True,
            "normalized": True,
        },
    )

    assert loss.normalized is True
    assert (loss.n_fft, loss.hop_length, loss.win_length) == (64, 16, 64)
    assert loss.window_name == "hann"
    assert loss.center is True


def test_noncentered_forward_pads_and_inverse_crops_to_original_length():
    """center=False 的非对齐输入补齐完整 frame 后必须可裁回原始长度。"""
    module = StftIstftModule(
        n_fft=64,
        hop_length=32,
        win_length=64,
        window="hamming",
        center=False,
    )
    waveform = torch.randn(1, 181, generator=torch.Generator().manual_seed(23))
    spectrum = module(waveform)
    maximum_length = module._maximum_inverse_length(spectrum.shape[-2])

    assert spectrum.shape[-2] == 5
    assert maximum_length >= waveform.shape[-1]
    restored = module.inverse(spectrum, waveform.shape[-1])
    torch.testing.assert_close(restored, waveform, atol=1e-5, rtol=1e-5)

    with pytest.raises(
        ValueError,
        match=rf"requested length={maximum_length + 1}.*maximum covered length={maximum_length}",
    ):
        module.inverse(spectrum, maximum_length + 1)


def test_all_repository_top_level_stft_configs_pass_strict_constructor():
    """所有现有顶层 stft 配置都必须通过与直接构造一致的严格校验。"""
    config_paths = sorted((*CONFIG_ROOT.rglob("*.yaml"), *CONFIG_ROOT.rglob("*.yml")))
    checked = 0
    for config_path in config_paths:
        config = load_merged_config(config_path)
        stft_config = config.get("stft")
        if stft_config is None:
            continue
        checked += 1
        module = build_stft_istft(stft_config)
        direct = StftIstftModule(**stft_config)
        assert (
            direct.n_fft,
            direct.hop_length,
            direct.win_length,
            direct.window_name,
            direct.center,
        ) == (
            module.n_fft,
            module.hop_length,
            module.win_length,
            module.window_name,
            module.center,
        ), config_path.relative_to(CONFIG_ROOT)

        # 非 hop 对齐、非零输入同时覆盖配置可构造性与真实 inverse 数值路径。
        waveform_length = 2 * module.n_fft + module.hop_length + 1
        waveform = torch.randn(
            1,
            waveform_length,
            generator=torch.Generator().manual_seed(checked),
        )
        spectrum = module(waveform)
        # center=false 配置（如 rt_semamba 系列的 hamming 400/100）帧从 0 起步、
        # 无预填充，窗末端平方和恰好越过 NOLA 阈值，2*n_fft+hop+1 会超出实际
        # 可重建右边界并触发 stft.inverse 的 fail-closed 拒绝（与
        # test_inverse_rejects_requested_tail_outside_spectral_coverage 互补）。
        # round-trip 取实际可重建前缀，既保留非 hop 对齐与非零尾部覆盖，
        # 也不放松 core.stft 的 fail-closed 校验。
        inverse_length = min(
            waveform_length, module._maximum_inverse_length(spectrum.shape[-1])
        )
        restored = module.inverse(spectrum, inverse_length)
        torch.testing.assert_close(
            restored,
            waveform[:, :inverse_length],
            atol=1e-5,
            rtol=1e-5,
            msg=lambda message: (
                f"{config_path.relative_to(CONFIG_ROOT)} round-trip failed: {message}"
            ),
        )

    assert checked > 0


def test_stft_finite_validation_is_opt_in(monkeypatch):
    """有限性全量检查默认关闭（避免每步 GPU→CPU 同步），显式开启后 fail-fast。"""
    from core import stft as stft_module

    module = StftIstftModule(n_fft=64, hop_length=16, win_length=64)
    waveform = torch.zeros(1, 65)
    waveform[0, 7] = float("nan")
    nan_spectrum = torch.full((1, 33, 2, 2), float("nan"))

    assert torch.isnan(module(waveform)).any()
    assert torch.isnan(module.inverse(nan_spectrum, 32)).any()

    monkeypatch.setattr(stft_module, "VALIDATE_FINITE_INPUTS", True)
    with pytest.raises(ValueError, match="stft.forward waveform must contain only finite"):
        module(waveform)
    with pytest.raises(ValueError, match="stft.inverse spectrum must contain only finite"):
        module.inverse(nan_spectrum, 32)
