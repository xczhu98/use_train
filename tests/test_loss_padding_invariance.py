"""JAE-H02: AEC 训练 loss 对 batch 零填充(padding)不变性测试。

各 loss 在 lengths=[VALID_LEN] 下, 同一有效信号(4096 样点)补零到
4096/8192/16384 时必须满足:
  1. loss 相对误差 < 1e-6;
  2. enhanced 梯度的 padding 区严格为 0;
  3. enhanced 梯度的有效区与未补零(T=VALID_LEN)情形相对误差 < 1e-5。
另检查 lengths 全有效(==T)时与 lengths=None 结果一致(既有行为不变)。

GTCRN Composite / fw 用 n_fft=320、hop=160, batch=1。
"""
import pytest

import torch

from core.losses.build import build_loss
from core.losses.freq_domain import MultiResSTFTLoss
from core.losses.fw_sdr import FWNegSISDRLoss
from core.losses.se_train_mr import (
    MRSIRelCspecL2Loss,
    MRSIRelCspecL2SuppressPenaltyLoss,
    MRSpecSuppressPenaltyLoss,
)
from core.losses.time_domain import SIL1WavSeTrainLoss, SISNRLoss

VALID_LEN = 4096
PAD_TOTALS = (8192, 16384)
LOSS_RTOL = 1e-6
GRAD_RTOL = 1e-5


def _base_signals():
    """固定种子生成有效信号: enhanced = clean + 弱噪声。"""
    g = torch.Generator().manual_seed(0)
    clean = torch.randn(VALID_LEN, generator=g)
    noise = 0.05 * torch.randn(VALID_LEN, generator=g)
    return clean + noise, clean, noise


def _pad(x: torch.Tensor, total: int) -> torch.Tensor:
    """右补零到 total 并加 batch 维 -> [1, total]。"""
    return torch.cat([x, x.new_zeros(total - VALID_LEN)]).unsqueeze(0)


def _loss_and_grad(loss_fn, total: int, *, fw: bool):
    enhanced, clean, noise = _base_signals()
    enh = _pad(enhanced, total).requires_grad_(True)
    cln = _pad(clean, total)
    lengths = torch.tensor([VALID_LEN])
    if fw:
        out = loss_fn(enh, cln, lengths, undesired_ref=_pad(noise, total))
    else:
        out = loss_fn(enh, cln, lengths)
    out.backward()
    return out.detach(), enh.grad


def _assert_padding_invariant(loss_fn, *, fw: bool = False):
    name = type(loss_fn).__name__
    ref_loss, ref_grad = _loss_and_grad(loss_fn, VALID_LEN, fw=fw)
    for total in PAD_TOTALS:
        loss, grad = _loss_and_grad(loss_fn, total, fw=fw)
        rel = (loss - ref_loss).abs() / ref_loss.abs().clamp(min=1e-12)
        assert rel < LOSS_RTOL, (
            f"{name}: 补零到 {total} 时 loss 相对误差 {rel.item():.3e} "
            f"(ref={ref_loss.item():.8e}, got={loss.item():.8e})"
        )
        assert torch.all(grad[:, VALID_LEN:] == 0.0), (
            f"{name}: 补零到 {total} 时 enhanced 梯度 padding 区非零: "
            f"max={grad[:, VALID_LEN:].abs().max().item():.3e}"
        )
        # 相对误差以有效区梯度最大幅值为基准, 避免近零元素放大 float32 求和噪声
        grel = (grad[:, :VALID_LEN] - ref_grad).abs() / ref_grad.abs().max().clamp(min=1e-12)
        assert grel.max() < GRAD_RTOL, (
            f"{name}: 补零到 {total} 时有效区梯度相对误差 {grel.max().item():.3e}"
        )


def _assert_full_valid_parity(loss_fn, *, fw: bool = False):
    """lengths 全有效(==T)时必须与 lengths=None 结果一致(既有行为不变)。"""
    name = type(loss_fn).__name__
    enhanced, clean, noise = _base_signals()
    enh, cln = _pad(enhanced, VALID_LEN), _pad(clean, VALID_LEN)
    lengths = torch.tensor([VALID_LEN])
    if fw:
        with_len = loss_fn(enh, cln, lengths, undesired_ref=_pad(noise, VALID_LEN))
        no_len = loss_fn(enh, cln, None, undesired_ref=_pad(noise, VALID_LEN))
    else:
        with_len = loss_fn(enh, cln, lengths)
        no_len = loss_fn(enh, cln, None)
    assert torch.allclose(with_len, no_len, rtol=1e-6, atol=1e-8), (
        f"{name}: lengths 全有效与 lengths=None 不一致: "
        f"{with_len.item():.8e} vs {no_len.item():.8e}"
    )


def test_gtcrn_composite_loss_padding_invariance():
    """生产配置的 GTCRN 两组件组合必须整体忽略 batch padding。"""
    loss_fn = build_loss(
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
        stft_cfg={"n_fft": 320, "hop_length": 160, "win_length": 320},
    )
    _assert_padding_invariant(loss_fn)
    _assert_full_valid_parity(loss_fn)


def test_sil1_wav_se_train_padding_invariance():
    # 对照组: 该 loss 原本已按 lengths 做 masked mean, 必须保持不变
    loss_fn = SIL1WavSeTrainLoss()
    _assert_padding_invariant(loss_fn)
    _assert_full_valid_parity(loss_fn)


def test_multi_res_stft_padding_invariance():
    # 审计 A3: log-mag 项改有效帧 masked mean, SC 项改有效帧 masked
    # Frobenius 范数, 波形经边界 reflect 延展后与 trim 一致
    loss_fn = MultiResSTFTLoss()
    _assert_padding_invariant(loss_fn)
    _assert_full_valid_parity(loss_fn)


def test_sisnr_padding_invariance():
    # 审计 A4: 零均值只在有效样点上计算, 减均值后再乘 mask
    loss_fn = SISNRLoss()
    _assert_padding_invariant(loss_fn)
    _assert_full_valid_parity(loss_fn)


def test_mr_si_rel_cspec_l2_padding_invariance():
    loss_fn = MRSIRelCspecL2Loss()
    _assert_padding_invariant(loss_fn)
    _assert_full_valid_parity(loss_fn)


@pytest.mark.parametrize(
    ("win_length", "hop_length", "expected_win_lengths", "expected_hop_lengths"),
    [
        pytest.param(
            320,
            160,
            (40, 80, 160, 320, 640, 1280),
            (20, 40, 80, 160, 320, 640),
            id="dbgnet-16k",
        ),
        pytest.param(
            640,
            320,
            (80, 160, 320, 640, 1280, 2560),
            (40, 80, 160, 320, 640, 1280),
            id="dbgnet-32k",
        ),
    ],
)
def test_frame_relative_mr_si_rel_padding_invariance(
    win_length,
    hop_length,
    expected_win_lengths,
    expected_hop_lengths,
):
    """DBGNet A/B 的非二次幂六档 schedule 必须忽略 batch padding。"""
    loss_fn = MRSIRelCspecL2Loss(
        win_length=win_length,
        hop_length=hop_length,
        stft_scale_factors=[0.125, 0.25, 0.5, 1, 2, 4],
        use_frequency_weight=False,
    )

    assert loss_fn.win_lengths == expected_win_lengths
    assert loss_fn.hop_lengths == expected_hop_lengths
    _assert_padding_invariant(loss_fn)


def test_mr_spec_suppress_penalty_padding_invariance():
    loss_fn = MRSpecSuppressPenaltyLoss()
    _assert_padding_invariant(loss_fn)
    _assert_full_valid_parity(loss_fn)


def test_mr_si_rel_cspec_l2_suppress_penalty_padding_invariance():
    # 组合类: penalty 路径与独立类 MRSpecSuppressPenaltyLoss 同口径
    # (trim 一致波形 STFT + masked mean), 补零后 loss/梯度均不变
    loss_fn = MRSIRelCspecL2SuppressPenaltyLoss()
    _assert_padding_invariant(loss_fn)
    _assert_full_valid_parity(loss_fn)


@pytest.mark.parametrize(
    ("loss_cls", "loss_kwargs"),
    [
        pytest.param(
            MRSIRelCspecL2Loss,
            {"use_frequency_weight": False},
            id="mr-si-rel-cspec-l2",
        ),
        pytest.param(MRSpecSuppressPenaltyLoss, {}, id="mr-spec-suppress-penalty"),
        pytest.param(
            MRSIRelCspecL2SuppressPenaltyLoss,
            {"use_frequency_weight": False},
            id="combined",
        ),
    ],
)
@pytest.mark.parametrize("valid_len", [1, 64], ids=["extremely-short", "regular"])
@pytest.mark.parametrize("extra", [1, 31], ids=["extra-1", "extra-pad-minus-1"])
def test_centered_mr_losses_keep_loss_and_valid_grad_with_short_right_padding(
    loss_cls,
    loss_kwargs,
    valid_len,
    extra,
):
    """centered STFT 的有效结果不能依赖输入是否已预留完整右侧 pad。"""
    n_fft = 64
    pad = n_fft // 2
    clean_valid = torch.linspace(0.2, 1.0, valid_len)
    phase = torch.linspace(0.0, torch.pi, valid_len)
    enhanced_valid = 1.2 * clean_valid + 0.05 * torch.sin(phase)
    lengths = torch.tensor([valid_len])
    loss_fn = loss_cls(fft_power_min=6, fft_power_max=6, **loss_kwargs)

    def loss_and_grad(right_padding):
        enhanced = torch.cat([enhanced_valid, torch.zeros(right_padding)]).unsqueeze(0)
        enhanced.requires_grad_(True)
        clean = torch.cat([clean_valid, torch.zeros(right_padding)]).unsqueeze(0)
        loss = loss_fn(enhanced, clean, lengths)
        loss.backward()
        return loss.detach(), enhanced.grad.detach()

    expected_loss, expected_grad = loss_and_grad(pad)
    actual_loss, actual_grad = loss_and_grad(extra)

    assert torch.isfinite(actual_loss)
    assert torch.isfinite(actual_grad).all()
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        actual_grad[:, :valid_len],
        expected_grad[:, :valid_len],
        rtol=1e-5,
        atol=1e-7,
    )
    assert torch.all(actual_grad[:, valid_len:] == 0.0)


@pytest.mark.parametrize("scale", ["mel", "critical"])
def test_fw_neg_sisdr_padding_invariance(scale):
    """正常长度下两条频带公式均保持 padding 不变与 full-valid 兼容。"""
    loss_fn = FWNegSISDRLoss(n_fft=320, hop_length=160, scale=scale)
    _assert_padding_invariant(loss_fn, fw=True)
    _assert_full_valid_parity(loss_fn, fw=True)


def test_fw_neg_sisdr_low_energy_gate_ignores_padding():
    """低能量语音门控只能查看 lengths 内的有效样点。"""
    valid_len = 512
    total = 1024
    g = torch.Generator().manual_seed(1)
    clean_valid = 1e-5 * torch.randn(valid_len, generator=g)
    enhanced_valid = torch.randn(valid_len, generator=g)
    noise_valid = torch.randn(valid_len, generator=g)

    clean = torch.cat([clean_valid, torch.ones(total - valid_len)]).unsqueeze(0)
    enhanced = torch.cat([
        enhanced_valid,
        torch.randn(total - valid_len, generator=g),
    ]).unsqueeze(0).requires_grad_(True)
    noise = torch.cat([
        noise_valid,
        torch.randn(total - valid_len, generator=g),
    ]).unsqueeze(0)

    loss_fn = FWNegSISDRLoss(n_fft=320, hop_length=160)
    loss = loss_fn(
        enhanced,
        clean,
        torch.tensor([valid_len]),
        undesired_ref=noise,
    )

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert enhanced.grad is not None
    assert torch.all(enhanced.grad == 0.0)


def _fw_light_padding_loss_and_grad(
    loss_fn: FWNegSISDRLoss,
    enhanced_valid: torch.Tensor,
    clean_valid: torch.Tensor,
    noise_valid: torch.Tensor,
    total: int,
):
    """计算一条 trim/轻量右补零样本的 FW-SDR loss 与 enhanced 梯度。"""
    valid_len = clean_valid.numel()

    def _right_pad(x):
        return torch.cat([x, x.new_zeros(total - valid_len)]).unsqueeze(0)

    enhanced = _right_pad(enhanced_valid).requires_grad_(True)
    loss = loss_fn(
        enhanced,
        _right_pad(clean_valid),
        torch.tensor([valid_len]),
        undesired_ref=_right_pad(noise_valid),
    )
    loss.backward()
    return loss.detach(), enhanced.grad


@pytest.mark.parametrize("scale", ["mel", "critical"])
@pytest.mark.parametrize("extra", [1, 63], ids=["extra-1", "pad-minus-1"])
def test_fw_neg_sisdr_light_padding_matches_trim(scale, extra):
    """不足完整 reflect 右余量的轻量 padding 也必须与单条 trim 一致。"""
    valid_len = 192
    generator = torch.Generator().manual_seed(37)
    clean = torch.randn(valid_len, generator=generator)
    noise = 0.1 * torch.randn(valid_len, generator=generator)
    enhanced = clean + 0.2 * torch.randn(valid_len, generator=generator)
    loss_fn = FWNegSISDRLoss(
        n_fft=128,
        hop_length=32,
        scale=scale,
        weights_type="log_snr_speech_norm",
    )

    trim_loss, trim_grad = _fw_light_padding_loss_and_grad(
        loss_fn, enhanced, clean, noise, valid_len
    )
    padded_loss, padded_grad = _fw_light_padding_loss_and_grad(
        loss_fn, enhanced, clean, noise, valid_len + extra
    )

    assert torch.allclose(padded_loss, trim_loss, rtol=1e-6, atol=1e-7)
    assert torch.allclose(
        padded_grad[:, :valid_len], trim_grad, rtol=1e-5, atol=1e-7
    )
    assert torch.count_nonzero(padded_grad[:, valid_len:]) == 0


@pytest.mark.parametrize("valid_len", [1, 3], ids=["length-1", "length-3"])
@pytest.mark.parametrize("extra", [1, 31], ids=["extra-1", "pad-minus-1"])
def test_fw_neg_sisdr_critical_short_input_padding_matches_trim(valid_len, extra):
    """critical 极短输入的预补零路径不得因 batch 轻量 padding 改变结果。"""
    clean = torch.tensor([0.8, -0.3, 0.2])[:valid_len]
    noise = torch.tensor([0.2, 0.4, -0.2])[:valid_len]
    enhanced = torch.tensor([0.6, -0.1, 0.5])[:valid_len]
    loss_fn = FWNegSISDRLoss(
        n_fft=64,
        hop_length=16,
        scale="critical",
        weights_type="log_snr_speech_norm",
    )

    trim_loss, trim_grad = _fw_light_padding_loss_and_grad(
        loss_fn, enhanced, clean, noise, valid_len
    )
    padded_loss, padded_grad = _fw_light_padding_loss_and_grad(
        loss_fn, enhanced, clean, noise, valid_len + extra
    )

    assert torch.isfinite(trim_loss) and torch.isfinite(padded_loss)
    assert torch.allclose(padded_loss, trim_loss, rtol=1e-6, atol=1e-7)
    assert torch.allclose(
        padded_grad[:, :valid_len], trim_grad, rtol=1e-5, atol=1e-7
    )
    assert torch.count_nonzero(padded_grad[:, valid_len:]) == 0


pytestmark = [pytest.mark.unit]
