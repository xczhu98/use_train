"""审计修复回归测试。

  A1  metricgan._masked_mse 分母补广播因子: lengths 全满时 masked 路径
      必须与 lengths=None(F.mse_loss) 结果一致。
  B4  PLCPAAsymmetricLoss 复数崩溃修复: 有/无 lengths 前向均不崩、输出有限。
  JAE-H04  FW-SDR 末端硬 clamp(min=-10) 改 -10 dB 处切线延伸下界:
      SDR < -10 dB 的 hard sample 梯度恢复非零且有界(C¹ 连续, 无调参),
      上界 35 dB 硬截断保持官方行为。
"""
import pytest

import torch

from core.losses.build import build_loss
from core.losses.fw_sdr import FWNegSISDRLoss
from core.losses.hybrid import GTCRNCompressedSpectralLoss, GTCRNLogRatioLoss
from core.losses.model_specific.metricgan import MPSEnetCompositeLoss
from core.losses.plcpa_asymmetric import PLCPAAsymmetricLoss


def _metricgan_forward_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    """构造自洽的 forward_dict: 谱帧数与波形 STFT 帧数一致。

    stft_n_fft=64, hop=16, win=64, center=True -> 帧数 = 1 + 1600 // 16 = 101,
    F = 64 // 2 + 1 = 33。
    """
    g = torch.Generator().manual_seed(seed)
    b, f, t, wav = 2, 33, 101, 1600
    return {
        "clean_mag": torch.rand(b, f, t, generator=g) + 0.1,
        "clean_pha": torch.rand(b, f, t, generator=g) * 6.28,
        "clean_spec": torch.randn(b, f, t, 2, generator=g),
        "enhanced_mag": torch.rand(b, f, t, generator=g) + 0.1,
        "enhanced_pha": torch.rand(b, f, t, generator=g) * 6.28,
        "enhanced_spec": torch.randn(b, f, t, 2, generator=g),
        "enhanced": torch.randn(b, wav, generator=g),
        "clean": torch.randn(b, wav, generator=g),
    }


def test_metricgan_masked_mse_full_lengths_parity():
    """A1: lengths 全满(==T)时 masked 路径与 lengths=None 结果一致。

    修复前全 1 mask 下 masked/unmasked = F(复谱 2F), 修复后逐点相等。
    """
    loss_fn = MPSEnetCompositeLoss(stft_n_fft=64, stft_hop_length=16, stft_win_length=64)
    fd = _metricgan_forward_dict()
    out_none = loss_fn(fd)
    out_full = loss_fn({**fd, "lengths": torch.tensor([1600, 1600])})
    assert torch.allclose(out_full, out_none, rtol=1e-6, atol=1e-8), (
        f"lengths 全满与 None 不一致: {out_full.item():.8e} vs {out_none.item():.8e}"
    )


def _build_gtcrn_composite():
    """按生产配置构造显式 GTCRN 两组件损失。"""
    return build_loss(
        {
            "type": "CompositeLoss",
            "components": [
                {
                    "type": "GTCRNCompressedSpectralLoss",
                    "weight": 1.0,
                    "args": {
                        "compress_factor": 0.3,
                        "lamda_ri": 30.0,
                        "lamda_mag": 70.0,
                    },
                },
                {"type": "GTCRNLogRatioLoss", "weight": 1.0},
            ],
        },
        stft_cfg={"n_fft": 64, "hop_length": 16, "win_length": 64},
    )


def _official_gtcrn_log_ratio_oracle(
    enhanced: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """复现 gtcrn-main/loss.py 的投影与 -log10 ratio 公式。"""
    projection = (
        torch.sum(target * enhanced, dim=-1, keepdim=True)
        / (torch.sum(target.square(), dim=-1, keepdim=True) + eps)
    ) * target
    ratio = torch.sum(projection.square(), dim=-1) / (
        torch.sum((enhanced - projection).square(), dim=-1) + eps
    )
    return -torch.log10(ratio + eps).mean()


def test_gtcrn_composite_matches_component_formula_and_gradient():
    """生产 Composite 路径必须等于 compressed spectral 与官方 ratio 之和。"""
    g = torch.Generator().manual_seed(123)
    clean = torch.randn(2, 256, generator=g)
    base_enhanced = clean + 0.1 * torch.randn(2, 256, generator=g)

    enhanced = base_enhanced.clone().requires_grad_(True)
    total = _build_gtcrn_composite()(enhanced, clean)
    total_grad = torch.autograd.grad(total, enhanced)[0]

    oracle_enhanced = base_enhanced.clone().requires_grad_(True)
    expected = GTCRNCompressedSpectralLoss(
        n_fft=64,
        hop_length=16,
        win_length=64,
        compress_factor=0.3,
        lamda_ri=30.0,
        lamda_mag=70.0,
    )(oracle_enhanced, clean) + _official_gtcrn_log_ratio_oracle(
        oracle_enhanced, clean
    )
    expected_grad = torch.autograd.grad(expected, oracle_enhanced)[0]

    assert torch.allclose(total, expected, rtol=1e-6, atol=1e-7)
    assert torch.allclose(total_grad, expected_grad, rtol=1e-5, atol=1e-7)
    assert torch.isfinite(total_grad).all()
    assert torch.count_nonzero(total_grad) > 0


@pytest.mark.parametrize("mode", ["enhanced-zero", "enhanced-tiny", "perfect"])
def test_gtcrn_log_ratio_matches_official_formula_oracle(mode):
    """零输出、极小输出和完美输出均保持官方非 dB log-ratio 公式。"""
    target = torch.tensor([[0.4, -0.7, 1.2, -0.2]], dtype=torch.float32)
    if mode == "enhanced-zero":
        base_enhanced = torch.zeros_like(target)
    elif mode == "enhanced-tiny":
        base_enhanced = 1e-6 * torch.tensor([[0.3, -0.1, 0.8, 0.5]])
    else:
        base_enhanced = target.clone()

    enhanced = base_enhanced.clone().requires_grad_(True)
    actual = GTCRNLogRatioLoss()(enhanced, target)
    actual_grad = torch.autograd.grad(actual, enhanced)[0]

    oracle_enhanced = base_enhanced.clone().requires_grad_(True)
    expected = _official_gtcrn_log_ratio_oracle(oracle_enhanced, target)
    expected_grad = torch.autograd.grad(expected, oracle_enhanced)[0]

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert torch.allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-7)
    assert torch.isfinite(actual_grad).all()


def test_gtcrn_log_ratio_zero_target_returns_graph_zero():
    """target=0 时必须返回可 backward 的图零且不产生伪梯度。"""
    enhanced = torch.randn(2, 128).requires_grad_(True)
    loss = GTCRNLogRatioLoss()(enhanced, torch.zeros_like(enhanced))

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert enhanced.grad is not None
    assert torch.count_nonzero(enhanced.grad) == 0


@pytest.mark.parametrize(
    "args",
    [
        {"n_fft": 64, "hop_length": 16, "win_length": 64},
        {
            "n_fft": 80,
            "hop_length": 20,
            "win_length": 64,
            "window": "hamming",
            "center": False,
            "normalized": True,
        },
    ],
)
def test_gtcrn_compressed_spectral_padding_parity(args):
    """不同 STFT geometry 下 compressed spectral 忽略 batch padding。"""
    from core.losses.hybrid import GTCRNCompressedSpectralLoss

    g = torch.Generator().manual_seed(19)
    clean_valid = torch.randn(192, generator=g)
    enhanced_valid = clean_valid + 0.1 * torch.randn(192, generator=g)
    loss_fn = GTCRNCompressedSpectralLoss(**args)
    results = []
    for total in (192, 256):
        clean = torch.cat([clean_valid, torch.zeros(total - 192)]).unsqueeze(0)
        enhanced = torch.cat([enhanced_valid, torch.zeros(total - 192)]).unsqueeze(0)
        enhanced.requires_grad_(True)
        loss = loss_fn(enhanced, clean, torch.tensor([192]))
        loss.backward()
        results.append((loss.detach(), enhanced.grad))

    assert torch.allclose(results[1][0], results[0][0], rtol=1e-6, atol=1e-7)
    assert torch.allclose(results[1][1][:, :192], results[0][1], rtol=1e-5, atol=1e-7)
    assert torch.count_nonzero(results[1][1][:, 192:]) == 0


def test_plcpa_asymmetric_smoke():
    """B4: 有/无 lengths 前向均不崩、输出有限、梯度有限。

    修复前复数 F.mse_loss(phase 项)与 complex mask clamp 使前向必崩。
    """
    g = torch.Generator().manual_seed(0)
    clean = torch.randn(2, 4096, generator=g)
    enhanced = clean + 0.05 * torch.randn(2, 4096, generator=g)
    loss_fn = PLCPAAsymmetricLoss()

    out_none = loss_fn(enhanced, clean)
    assert torch.isfinite(out_none), f"lengths=None 输出非有限: {out_none}"

    enh_g = enhanced.clone().requires_grad_(True)
    out_len = loss_fn(enh_g, clean, torch.tensor([4096, 2048]))
    assert torch.isfinite(out_len), f"lengths 给定输出非有限: {out_len}"
    out_len.backward()
    assert torch.isfinite(enh_g.grad).all(), "lengths 给定梯度非有限"


def test_fw_neg_sisdr_tangent_floor_keeps_gradient():
    """JAE-H04: 加权 SDR < -10 dB 的 hard sample 梯度不再被硬 clamp 截断。

    enhanced 与 clean 独立(投影 ≈ 0)时加权 SDR 远低于 -10 dB。修复前
    clamp(min=-10) 使 enhanced 梯度恒 0; 修复后 -10 dB 处切线延伸
    (斜率 10/(ln10·0.1), C¹ 连续)保持非零且有界梯度, loss 亦有界
    (最坏 ratio=0 → 10 + 4.343 ≈ 14.34)。
    """
    g = torch.Generator().manual_seed(0)
    clean = torch.randn(1, 4096, generator=g)
    noise = 0.05 * torch.randn(1, 4096, generator=g)
    enhanced = torch.randn(1, 4096, generator=g).requires_grad_(True)
    loss_fn = FWNegSISDRLoss(n_fft=320, hop_length=160)

    loss = loss_fn(enhanced, clean, None, undesired_ref=noise)
    loss.backward()
    assert torch.isfinite(loss), f"hard sample 输出非有限: {loss}"
    assert torch.isfinite(enhanced.grad).all(), "hard sample 梯度非有限"
    assert enhanced.grad.abs().max() > 0, "hard sample 梯度被截断为 0"
    assert loss.item() > 10.0, (
        f"测试样本未落入 < -10 dB 尾部区(loss={loss.item():.4f} <= 10), "
        "梯度截断断言失去意义"
    )
    assert loss.item() < 14.35, (
        f"切线延伸下界后 loss 应 < 14.35(最坏 ratio=0 → 14.34), got {loss.item():.4f}"
    )


def test_fw_neg_sisdr_customizable_bounds():
    """JAE-H04: sdr_min_db / sdr_max_db 构造参数生效。

    深尾样本(enhanced = clean + 30×独立噪声, ratio ≈ 1e-3)在默认拐点 -10
    与拐点 -20 下都走切线, 但切线斜率 10× 关系(43.4 vs 434.3), 梯度应呈
    同比例放大; 上界 40 时 perfect reconstruction 截断为 loss=-40。
    """
    g = torch.Generator().manual_seed(0)
    clean = torch.randn(1, 4096, generator=g)
    noise = 0.05 * torch.randn(1, 4096, generator=g)
    enhanced = clean + 30.0 * torch.randn(1, 4096, generator=g)

    grads = {}
    losses = {}
    for kw, tag in (({}, "default"), ({"sdr_min_db": -20.0, "sdr_max_db": 40.0}, "custom")):
        enh = enhanced.clone().requires_grad_(True)
        loss_fn = FWNegSISDRLoss(n_fft=320, hop_length=160, **kw)
        loss = loss_fn(enh, clean, None, undesired_ref=noise)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(enh.grad).all()
        assert enh.grad.abs().max() > 0, f"{tag}: 深尾样本梯度被截断"
        losses[tag] = loss.item()
        grads[tag] = enh.grad.abs().max().item()

    # 拐点 -20 的切线斜率是 -10 的 10 倍 → 深尾样本梯度同比放大
    ratio = grads["custom"] / grads["default"]
    assert 5.0 < ratio < 20.0, f"拐点参数化未生效: 梯度比 {ratio:.2f} 应 ≈ 10"
    # 拐点 -20 切线尾界 |−20 − 434.3×0.01| = 24.34; 实测样本 loss 应在其内侧
    assert 20.0 < losses["custom"] < 24.4, (
        f"knee=-20 深尾样本 loss 应落在 (20, 24.4), got {losses['custom']:.4f}"
    )

    # 上界参数化: perfect reconstruction → loss = -sdr_max_db
    loss_hi = FWNegSISDRLoss(n_fft=320, hop_length=160, sdr_max_db=40.0)
    lp = loss_hi(clean.clone(), clean, None, undesired_ref=noise)
    assert torch.allclose(lp, torch.tensor(-40.0), atol=1e-4), (
        f"sdr_max_db=40 时 perfect reconstruction 应截断为 -40, got {lp.item():.6f}"
    )


def test_fw_neg_sisdr_upper_clamp_preserved():
    """JAE-H04: 上界 35 dB 硬截断保持官方行为(perfect reconstruction → loss=-35)。"""
    g = torch.Generator().manual_seed(0)
    clean = torch.randn(1, 4096, generator=g)
    noise = 0.05 * torch.randn(1, 4096, generator=g)
    loss_fn = FWNegSISDRLoss(n_fft=320, hop_length=160)

    loss = loss_fn(clean.clone(), clean, None, undesired_ref=noise)
    assert torch.allclose(loss, torch.tensor(-35.0), atol=1e-4), (
        f"perfect reconstruction 应被上界截断为 loss=-35, got {loss.item():.6f}"
    )


@pytest.mark.parametrize(
    ("loss_cls", "kwargs", "field"),
    [
        (GTCRNCompressedSpectralLoss, {"n_fft": True}, "n_fft"),
        (GTCRNCompressedSpectralLoss, {"hop_length": 16.0}, "hop_length"),
        (GTCRNCompressedSpectralLoss, {"win_length": "64"}, "win_length"),
        (GTCRNCompressedSpectralLoss, {"compress_factor": True}, "compress_factor"),
        (GTCRNCompressedSpectralLoss, {"lamda_ri": "30"}, "lamda_ri"),
        (GTCRNCompressedSpectralLoss, {"lamda_mag": None}, "lamda_mag"),
        (GTCRNCompressedSpectralLoss, {"center": 1}, "center"),
        (GTCRNCompressedSpectralLoss, {"normalized": 0}, "normalized"),
        (GTCRNCompressedSpectralLoss, {"window": 1}, "window"),
        (GTCRNLogRatioLoss, {"eps": "1e-8"}, "eps"),
    ],
)
def test_gtcrn_constructor_rejects_wrong_parameter_types(loss_cls, kwargs, field):
    """直接构造 GTCRN loss 时不得接受 bool coercion 或隐式数值转换。"""
    with pytest.raises(TypeError, match=field):
        loss_cls(**kwargs)


@pytest.mark.parametrize(
    ("loss_cls", "kwargs", "field"),
    [
        (GTCRNCompressedSpectralLoss, {"compress_factor": float("nan")}, "compress_factor"),
        (GTCRNCompressedSpectralLoss, {"lamda_ri": float("inf")}, "lamda_ri"),
        (GTCRNCompressedSpectralLoss, {"lamda_mag": -float("inf")}, "lamda_mag"),
        (GTCRNLogRatioLoss, {"eps": float("nan")}, "eps"),
    ],
)
def test_gtcrn_constructor_rejects_non_finite_parameters(loss_cls, kwargs, field):
    """所有浮点超参数必须在构造期拒绝 NaN/Inf。"""
    with pytest.raises(ValueError, match=field):
        loss_cls(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fft_sizes": (), "hop_sizes": (), "win_lengths": ()},
        {"fft_sizes": (64, 128), "hop_sizes": (16,), "win_lengths": (64, 128)},
    ],
)
def test_multi_res_stft_rejects_empty_or_mismatched_schedules(kwargs):
    """三条 resolution schedule 必须非空且长度完全一致，禁止 zip 静默截断。"""
    from core.losses.freq_domain import MultiResSTFTLoss

    with pytest.raises(ValueError, match="schedule"):
        MultiResSTFTLoss(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fft_sizes": (True,), "hop_sizes": (16,), "win_lengths": (64,)},
        {"fft_sizes": (64,), "hop_sizes": (16.0,), "win_lengths": (64,)},
        {"fft_sizes": (64,), "hop_sizes": (16,), "win_lengths": (0,)},
    ],
)
def test_multi_res_stft_rejects_non_positive_exact_integer_schedule(kwargs):
    """schedule 样本数仅接受精确正整数。"""
    from core.losses.freq_domain import MultiResSTFTLoss

    with pytest.raises((TypeError, ValueError), match="positive integer"):
        MultiResSTFTLoss(**kwargs)


def test_multi_res_stft_rejects_win_length_greater_than_n_fft():
    """每个 resolution 都必须满足 win_length <= n_fft。"""
    from core.losses.freq_domain import MultiResSTFTLoss

    with pytest.raises(ValueError, match="win_length <= n_fft"):
        MultiResSTFTLoss(
            fft_sizes=(64,), hop_sizes=(16,), win_lengths=(80,)
        )


def test_multi_res_stft_runs_with_hop_length_greater_than_win_length():
    """forward-only STFT 支持 hop > win，零 target 仅使用一次 log-mag。"""
    from core.losses.freq_domain import MultiResSTFTLoss

    generator = torch.Generator().manual_seed(29)
    enhanced = torch.randn(2, 256, generator=generator)
    clean = torch.zeros_like(enhanced)
    loss_fn = MultiResSTFTLoss(
        fft_sizes=(64,), hop_sizes=(48,), win_lengths=(32,)
    )

    actual = loss_fn(enhanced, clean)
    enhanced_mag = torch.stft(
        enhanced,
        n_fft=64,
        hop_length=48,
        win_length=32,
        window=torch.hann_window(32),
        return_complex=True,
        center=True,
    ).abs().clamp(min=1e-8)
    expected_mag = (
        torch.log(enhanced_mag) - torch.log(torch.full_like(enhanced_mag, 1e-8))
    ).abs().mean()

    assert torch.isfinite(actual)
    assert torch.allclose(actual, expected_mag, rtol=1e-6, atol=1e-7)


pytestmark = [pytest.mark.unit]
