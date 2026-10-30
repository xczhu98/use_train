"""MetricGAN composite loss 的 lengths、短输入与生产配置回归测试。"""

from pathlib import Path

import pytest
import torch
import yaml

from core.losses.model_specific.metricgan import MPSEnetCompositeLoss, phase_losses
from core.losses.utils import compute_spec_frame_lengths, prepare_centered_stft_inputs


_REPO_ROOT = Path(__file__).parents[1]
_PRODUCTION_CONFIGS = (
    "mp_senet_metricgan_dynamic.yaml",
    "mambattention_metricgan_dynamic.yaml",
    "mossformer_gan_metricgan_dynamic.yaml",
    "rwsa_mamba_unet_metricgan_dynamic.yaml",
)


def _loss_fn(**weights):
    return MPSEnetCompositeLoss(
        stft_n_fft=64,
        stft_hop_length=16,
        stft_win_length=64,
        compress_factor=0.3,
        **weights,
    )


def _spectral_fields(clean: torch.Tensor, enhanced: torch.Tensor):
    (clean_stft, enhanced_stft), lengths, _ = prepare_centered_stft_inputs(
        clean.unsqueeze(0), enhanced.unsqueeze(0), lengths=None, n_fft=64
    )
    kwargs = {
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "window": torch.hann_window(64),
        "center": True,
        "return_complex": True,
    }
    clean_spec = torch.stft(clean_stft, **kwargs)
    enhanced_spec = torch.stft(enhanced_stft, **kwargs)
    n_frames = int(compute_spec_frame_lengths(lengths, 64, 64, 16).item())
    clean_spec = clean_spec[..., :n_frames]
    enhanced_spec = enhanced_spec[..., :n_frames]
    clean_mag = clean_spec.abs().pow(0.3)
    enhanced_mag = enhanced_spec.abs().pow(0.3) * 1.05
    return {
        "clean_mag": clean_mag,
        "clean_pha": torch.angle(clean_spec),
        "clean_spec": torch.view_as_real(clean_spec),
        "enhanced_mag": enhanced_mag,
        "enhanced_pha": torch.angle(enhanced_spec),
        "enhanced_spec": torch.view_as_real(enhanced_spec),
    }


def _loss_and_waveform_grad(
    loss_fn: MPSEnetCompositeLoss,
    clean_valid: torch.Tensor,
    enhanced_valid: torch.Tensor,
    extra: int,
):
    fields = _spectral_fields(clean_valid, enhanced_valid)
    clean = torch.cat((clean_valid, torch.zeros(extra))).unsqueeze(0)
    enhanced = torch.cat((enhanced_valid, torch.zeros(extra))).unsqueeze(0)
    enhanced.requires_grad_(True)
    forward_dict = {**fields, "clean": clean, "enhanced": enhanced}
    if extra:
        forward_dict["lengths"] = torch.tensor([clean_valid.numel()])
    loss = loss_fn(forward_dict)
    loss.backward()
    return loss.detach(), enhanced.grad.detach()


@pytest.mark.parametrize("valid_len", [1, 7, 96], ids=["one-sample", "short", "regular"])
@pytest.mark.parametrize("extra", [1, 31], ids=["extra-1", "pad-minus-1"])
def test_metricgan_short_centered_stft_padding_matches_trim(valid_len, extra):
    """短输入可反向，extra=1/pad-1 与 trim 的 loss/有效梯度等价。"""
    clean = torch.linspace(0.2, 0.8, valid_len)
    enhanced = 1.08 * clean + torch.linspace(0.02, -0.01, valid_len)
    loss_fn = _loss_fn()

    trim_loss, trim_grad = _loss_and_waveform_grad(loss_fn, clean, enhanced, 0)
    padded_loss, padded_grad = _loss_and_waveform_grad(
        loss_fn, clean, enhanced, extra
    )

    assert torch.isfinite(trim_loss) and torch.isfinite(padded_loss)
    assert torch.isfinite(trim_grad).all() and torch.isfinite(padded_grad).all()
    torch.testing.assert_close(padded_loss, trim_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        padded_grad[:, :valid_len], trim_grad, rtol=1e-5, atol=1e-7
    )
    assert torch.count_nonzero(padded_grad[:, valid_len:]) == 0


def test_metricgan_phase_uses_valid_spec_mask_and_zeroes_padding_gradient():
    """IP/GD/IAF 仅消费有效谱帧，padding phase 不贡献 loss 或梯度。"""
    b, f, frames, wav_len = 1, 33, 7, 96
    clean_phase = torch.zeros(b, f, frames)
    enhanced_phase = torch.full((b, f, frames), 0.8)
    enhanced_phase[..., :3] = 0.2
    enhanced_phase.requires_grad_(True)
    mag = torch.ones(b, f, frames)
    waveform = torch.zeros(b, wav_len)
    forward_dict = {
        "clean_mag": mag,
        "clean_pha": clean_phase,
        "clean_spec": torch.zeros(b, f, frames, 2),
        "enhanced_mag": mag,
        "enhanced_pha": enhanced_phase,
        "enhanced_spec": torch.zeros(b, f, frames, 2),
        "enhanced": waveform,
        "clean": waveform,
        "lengths": torch.tensor([32]),
    }
    loss = _loss_fn(
        weight_mag=0.0,
        weight_pha=1.0,
        weight_com=0.0,
        weight_time=0.0,
        weight_stft=0.0,
    )(forward_dict)
    loss.backward()

    expected = sum(phase_losses(clean_phase[..., :3], enhanced_phase.detach()[..., :3]))
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(enhanced_phase.grad[..., :3]) > 0
    assert torch.count_nonzero(enhanced_phase.grad[..., 3:]) == 0


def test_metricgan_time_mean_counts_only_valid_samples():
    """time L1 按全 batch 有效 samples 归约，padding 梯度严格为零。"""
    enhanced = torch.tensor(
        [[1.0, 2.0, 3.0, 40.0], [4.0, 50.0, 60.0, 70.0]],
        requires_grad=True,
    )
    clean = torch.zeros_like(enhanced)
    b, f, frames = 2, 33, 1
    mag = torch.ones(b, f, frames)
    phase = torch.zeros(b, f, frames)
    forward_dict = {
        "clean_mag": mag,
        "clean_pha": phase,
        "clean_spec": torch.zeros(b, f, frames, 2),
        "enhanced_mag": mag,
        "enhanced_pha": phase,
        "enhanced_spec": torch.zeros(b, f, frames, 2),
        "enhanced": enhanced,
        "clean": clean,
        "lengths": torch.tensor([3, 1]),
    }
    loss = _loss_fn(
        weight_mag=0.0,
        weight_pha=0.0,
        weight_com=0.0,
        weight_time=1.0,
        weight_stft=0.0,
    )(forward_dict)
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor((1.0 + 2.0 + 3.0 + 4.0) / 4.0))
    torch.testing.assert_close(
        enhanced.grad,
        torch.tensor([[0.25, 0.25, 0.25, 0.0], [0.25, 0.0, 0.0, 0.0]]),
    )


def test_metricgan_phase_preserves_full_valid_formula():
    """常规完整输入保持原 IP + GD + IAF anti-wrapping 公式。"""
    generator = torch.Generator().manual_seed(71)
    clean_phase = torch.rand(2, 5, 7, generator=generator) * 6.28
    enhanced_phase = torch.rand(2, 5, 7, generator=generator) * 6.28
    actual = phase_losses(clean_phase, enhanced_phase)

    def anti_wrap(value):
        return torch.abs(value - torch.round(value / (2.0 * torch.pi)) * 2.0 * torch.pi)

    expected = (
        anti_wrap(clean_phase - enhanced_phase).mean(),
        anti_wrap(
            torch.diff(clean_phase, dim=1) - torch.diff(enhanced_phase, dim=1)
        ).mean(),
        anti_wrap(
            torch.diff(clean_phase, dim=2) - torch.diff(enhanced_phase, dim=2)
        ).mean(),
    )
    for actual_part, expected_part in zip(actual, expected):
        torch.testing.assert_close(actual_part, expected_part, rtol=1e-6, atol=1e-7)


def test_metricgan_loss_prefers_raw_generator_spectrum():
    """sanitize 后的判别器谱不得覆盖 generator 原始 loss 谱。"""
    phase = torch.zeros(1, 33, 2)
    waveform = torch.zeros(1, 16)
    fields = {
        "clean_mag": torch.ones_like(phase),
        "clean_pha": phase,
        "enhanced_mag": torch.ones_like(phase),
        "enhanced_pha": phase,
        "loss_enhanced_spectra": torch.stack(
            [torch.full_like(phase, 2.0 ** (1.0 / 0.3)), phase], dim=-1
        ),
        "enhanced": waveform,
        "clean": waveform,
        "lengths": torch.tensor([16]),
    }
    loss = _loss_fn(
        weight_mag=0.0, weight_pha=0.0, weight_com=1.0,
        weight_time=0.0, weight_stft=0.0,
    )(fields)
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_mambattention_rwsa_synthetic_checkpoint_roundtrip(tmp_path):
    from core.utils.safe_load import safe_torch_load
    from tools.model_tools.convert_mambattention_rwsa_checkpoint import convert_checkpoint

    model = torch.nn.Sequential()
    model.add_module("model", torch.nn.Linear(2, 1))
    official = {
        key.removeprefix("model."): value.detach().clone()
        for key, value in model.state_dict().items()
    }
    source, output = tmp_path / "official.pth", tmp_path / "converted.pt"
    torch.save({"generator": official}, source)

    stats = convert_checkpoint(source, output, model=model)
    converted = safe_torch_load(output, map_location="cpu")

    assert stats["matched_tensor_count"] == len(official)
    assert set(converted) == {"model_generator"}
    for key, expected in model.state_dict().items():
        torch.testing.assert_close(converted["model_generator"][key], expected)


@pytest.mark.parametrize("config_name", _PRODUCTION_CONFIGS)
def test_metricgan_production_configs_use_length_aware_composite_loss(config_name):
    """四份生产 MetricGAN 配置均走本 composite loss，且保持既有权重/几何。"""
    config_path = _REPO_ROOT / "configs/se/urgent2026/gan" / config_name
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loss = config["loss"]
    trainer = config["trainer"]
    stft = config["stft"]

    assert loss["type"] == "MPSEnetCompositeLoss"
    assert loss["args"]["stft_n_fft"] == stft["n_fft"]
    assert loss["args"]["stft_hop_length"] == stft["hop_length"]
    assert loss["args"]["stft_win_length"] == stft["win_length"]
    assert loss["args"]["compress_factor"] == 0.3
    if config_name in {
        "mambattention_metricgan_dynamic.yaml",
        "rwsa_mamba_unet_metricgan_dynamic.yaml",
    }:
        assert trainer["lr_schedule_per_step"] is False
    assert (
        trainer["loss_mag_weight"],
        trainer["loss_pha_weight"],
        trainer["loss_com_weight"],
        trainer["loss_time_weight"],
        trainer["loss_stft_weight"],
    ) == (0.9, 0.3, 0.1, 0.2, 0.1)


pytestmark = [pytest.mark.unit]
