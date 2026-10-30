"""DBGNet mask domain（DBG-Mag / DBG-Com）与论文几何单元测试。

覆盖：
- 默认 ``mask_domain="mag"`` 与历史实现逐键一致（通道维度不变）；
- ``com`` 的 2 通道 RI 输入、tanh 复数掩码、复数乘法与反向梯度；
- streaming 与 offline 在两种掩码域下逐帧一致，cache 通道数随掩码域变化；
- 论文 Table I 几何（ERB 129、频域尺寸、F-GRU 192、T-GRU 128）与参数量。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.models.build import build_model
from core.utils.config import load_merged_config
from tasks.aec.models.discriminative.dbgnet import DBGNet
from onnx.models.streaming_dbgnet import StreamingDBGNet

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_CONFIG = (
    REPO_ROOT
    / "configs/aec/microsoft_challenge/dbgnet/"
    "dbgnet_paper_16k_16ms_delay1000ms_loss_recon.yaml"
)

# 论文 Table I：mic/far 通道、ERB 65+64=129、频域尺寸 129→129→64→31→15。
# 不写 fgru_hidden：F-GRU 每方向宽度由论文规则 mic[-1]+far[-1] = 192 推导。
PAPER_MODEL_ARGS = {
    "n_fft": 512,
    "hop_length": 256,
    "win_length": 512,
    "sample_rate": 16000,
    "erb_low_bins": 65,
    "erb_bands": 64,
    "mic_channels": [32, 64, 96, 128],
    "far_channels": [16, 32, 48, 64],
    "use_state_detector": False,
}
PAPER_PARAMS = 1_500_000

# 论文 Table I 的逐层输出形状。取 B=2、T=63、F=257，使 batch 轴也可判定：
# F-GRU 行在 Table I 记作 (B×T, 15, 384) → 2×63=126；T-GRU 行记作 (B×15, T, 128) → 2×15=30。
TABLE_I_LAYER_SHAPES = (
    ("band_split_merge", (2, 129, 63)),
    ("conv_block_1_mic", (2, 32, 129, 63)),
    ("conv_block_1_far", (2, 16, 129, 63)),
    ("conv_block_2_mic", (2, 64, 64, 63)),
    ("conv_block_2_far", (2, 32, 64, 63)),
    ("conv_block_3_mic", (2, 96, 31, 63)),
    ("conv_block_3_far", (2, 48, 31, 63)),
    ("conv_block_4_mic", (2, 128, 15, 63)),
    ("conv_block_4_far", (2, 64, 15, 63)),
    ("concatenation", (2, 192, 15, 63)),
    ("fgru_input", (126, 15, 192)),
    ("fgru", (126, 15, 384)),
    ("tgru_input", (30, 63, 384)),
    ("tgru", (30, 63, 128)),
    ("decoder_input", (2, 128, 15, 63)),
    ("deconv_block_1", (2, 96, 31, 63)),
    ("deconv_block_2", (2, 64, 64, 63)),
    ("deconv_block_3", (2, 32, 129, 63)),
    ("deconv_block_4", (2, 1, 129, 63)),
    ("pointwise_conv", (2, 1, 129, 63)),
    ("band_split_merge_out", (2, 257, 63)),
    ("enhanced_spec", (2, 257, 63, 2)),
)


def _table_i_layer_shapes(model: DBGNet, mic, far):
    """按 forward_spectra 的执行顺序重放子模块，返回逐层输出形状。

    每一步都记录完整 4 轴（B/C/F/T），F-GRU/T-GRU 行额外体现 Table I 的 batch 折叠约定。
    """
    def pad_t(x):
        return torch.nn.functional.pad(x, (1, 0, 0, 0))

    shapes = []
    with torch.no_grad():
        x_mic = model.erb.bm(model._magnitude(mic))
        x_far = model.erb.bm(model._magnitude(far))
        shapes.append(("band_split_merge", tuple(x_mic.shape)))
        x_mic = x_mic.unsqueeze(1)
        x_far = model.fpb(x_far.unsqueeze(1))
        dy1 = model.conv_mic1(pad_t(x_mic))
        dx1 = model.ftg1(model.conv_far1(pad_t(x_far)))
        gy1, gx1, _ = model.slm1(dy1, dx1)
        shapes += [("conv_block_1_mic", tuple(dy1.shape)), ("conv_block_1_far", tuple(dx1.shape))]
        dy2 = model.conv_mic2(pad_t(gy1))
        dx2 = model.ftg2(model.conv_far2(pad_t(gx1)))
        gy2, gx2, _ = model.slm2(dy2, dx2)
        shapes += [("conv_block_2_mic", tuple(dy2.shape)), ("conv_block_2_far", tuple(dx2.shape))]
        dy3 = model.conv_mic3(pad_t(gy2))
        dx3 = model.ftg3(model.conv_far3(pad_t(gx2)))
        gy3, gx3, _ = model.slm3(dy3, dx3)
        shapes += [("conv_block_3_mic", tuple(dy3.shape)), ("conv_block_3_far", tuple(dx3.shape))]
        dy4 = model.conv_mic4(pad_t(gy3))
        dx4 = model.ftg4(model.conv_far4(pad_t(gx3)))
        gy4, gx4, _ = model.slm4(dy4, dx4)
        shapes += [("conv_block_4_mic", tuple(dy4.shape)), ("conv_block_4_far", tuple(dx4.shape))]
        y = torch.cat([gy4, gx4], dim=1)
        shapes.append(("concatenation", tuple(y.shape)))
        batch = y.shape[0]
        y = y.permute(0, 3, 2, 1).flatten(0, 1)
        shapes.append(("fgru_input", tuple(y.shape)))
        y, _ = model.fgru(y)
        shapes.append(("fgru", tuple(y.shape)))
        y = y.view(batch, -1, *y.shape[1:]).permute(0, 2, 1, 3).flatten(0, 1)
        shapes.append(("tgru_input", tuple(y.shape)))
        y, _ = model.tgru(y)
        shapes.append(("tgru", tuple(y.shape)))
        y = y.view(batch, -1, *y.shape[1:]).permute(0, 3, 1, 2)
        shapes.append(("decoder_input", tuple(y.shape)))
        y = model.deconv4(y, gy4)[..., :-1]
        shapes.append(("deconv_block_1", tuple(y.shape)))
        y = model.deconv3(y, gy3)[..., :-1]
        shapes.append(("deconv_block_2", tuple(y.shape)))
        y = model.deconv2(y, gy2)[..., :-1]
        shapes.append(("deconv_block_3", tuple(y.shape)))
        y = model.deconv1(y, gy1)[..., :-1]
        shapes.append(("deconv_block_4", tuple(y.shape)))
        y = model.out_conv(y)
        shapes.append(("pointwise_conv", tuple(y.shape)))
        mask = torch.sigmoid(model.erb.bs(y.squeeze(1)))
        shapes.append(("band_split_merge_out", tuple(mask.shape)))
        shapes.append(("enhanced_spec", tuple((mic * mask[..., None]).shape)))
    return tuple(shapes)


def _small_dbgnet(**kwargs) -> DBGNet:
    args = {
        "n_fft": 320,
        "hop_length": 160,
        "win_length": 320,
        "sample_rate": 16000,
        "erb_low_bins": 41,
        "erb_bands": 40,
        "mic_channels": [4, 8, 12, 16],
        "far_channels": [2, 4, 6, 8],
        "use_state_detector": False,
    }
    args.update(kwargs)
    return DBGNet(**args)


def _specs(batch: int = 1, n_freqs: int = 161, n_frames: int = 12):
    mic = torch.randn(batch, n_freqs, n_frames, 2)
    far = torch.randn(batch, n_freqs, n_frames, 2)
    return mic, far


def test_default_mask_domain_is_mag_and_keeps_legacy_module_shapes():
    """默认 mag：掩码域敏感模块保持 1 通道，state_dict 键集合与 com 一致。"""
    mag = _small_dbgnet()
    com = _small_dbgnet(mask_domain="com")

    assert mag.mask_domain == "mag"
    assert com.mask_domain == "com"
    assert tuple(mag.fpb[0].weight.shape) == (1, 1, 1, 1)
    assert tuple(mag.conv_mic1[0].weight.shape[1:2]) == (1,)
    assert tuple(mag.conv_far1[0].weight.shape[1:2]) == (1,)
    assert tuple(mag.out_conv.weight.shape) == (1, 1, 1, 1)

    assert tuple(com.fpb[0].weight.shape) == (2, 2, 1, 1)
    assert tuple(com.conv_mic1[0].weight.shape[1:2]) == (2,)
    assert tuple(com.conv_far1[0].weight.shape[1:2]) == (2,)
    assert tuple(com.out_conv.weight.shape) == (2, 1, 1, 1)

    # 两种掩码域的 state_dict 键集合相同，只有掩码域敏感模块的形状不同。
    assert set(mag.state_dict()) == set(com.state_dict())
    different = {
        key
        for key, value in mag.state_dict().items()
        if value.shape != com.state_dict()[key].shape
    }
    assert different == {
        "fpb.0.weight",
        "fpb.0.bias",
        "fpb.1.weight",
        "fpb.1.bias",
        "fpb.1.running_mean",
        "fpb.1.running_var",
        "conv_mic1.0.weight",
        "conv_far1.0.weight",
        "out_conv.weight",
        "out_conv.bias",
    }


@pytest.mark.parametrize("value", ["complex", "", 1, None])
def test_invalid_mask_domain_fails_fast(value):
    """mask_domain 只接受 'mag' / 'com'；拼写错误必须显式报错。"""
    with pytest.raises((TypeError, ValueError)):
        _small_dbgnet(mask_domain=value)


def test_mask_domain_is_normalized_case_insensitively():
    """大小写与空白归一化后仍是合法掩码域。"""
    assert _small_dbgnet(mask_domain=" MAG ").mask_domain == "mag"
    assert _small_dbgnet(mask_domain="Com").mask_domain == "com"


def test_mag_preserves_mic_phase_and_com_changes_it():
    """mag 保留麦克风相位；com 输出相位被复数掩码改写。"""
    torch.manual_seed(0)
    mag = _small_dbgnet().eval()
    com = _small_dbgnet(mask_domain="com").eval()
    mic, far = _specs()

    with torch.no_grad():
        out_mag = mag({"mic_spectra": mic, "farend_spectra": far})[
            "nearend_estimate_spectra"
        ]
        out_com = com({"mic_spectra": mic, "farend_spectra": far})[
            "nearend_estimate_spectra"
        ]

    assert out_mag.shape == mic.shape
    assert out_com.shape == mic.shape
    phase_mic = torch.atan2(mic[..., 1], mic[..., 0])
    phase_mag = torch.atan2(out_mag[..., 1], out_mag[..., 0])
    phase_com = torch.atan2(out_com[..., 1], out_com[..., 0])
    assert float((phase_mag - phase_mic).abs().max()) < 1e-4
    assert float((phase_com - phase_mic).abs().max()) > 1e-2
    assert not torch.allclose(out_mag, out_com)


def test_com_forward_backward_and_tanh_mask_bound():
    """com 可反向传播，且输出幅度不超过 tanh 掩码的 |m| <= sqrt(2) 上界。"""
    torch.manual_seed(0)
    com = _small_dbgnet(mask_domain="com")
    mic, far = _specs()

    out = com({"mic_spectra": mic, "farend_spectra": far})[
        "nearend_estimate_spectra"
    ]
    assert torch.isfinite(out).all()
    mic_mag = torch.sqrt(mic[..., 0].square() + mic[..., 1].square())
    out_mag = torch.sqrt(out[..., 0].square() + out[..., 1].square())
    assert bool((out_mag <= (2.0**0.5) * mic_mag + 1e-6).all())

    out.abs().mean().backward()
    grads = [
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in com.parameters()
        if p.requires_grad
    ]
    assert all(grads)


def test_com_with_state_detector_returns_state_logits():
    """com 与辅助 state detector 组合时返回增强谱与帧级 logits。"""
    model = _small_dbgnet(mask_domain="com", use_state_detector=True).eval()
    mic, far = _specs()
    with torch.no_grad():
        out = model({"mic_spectra": mic, "farend_spectra": far})
    assert out["nearend_estimate_spectra"].shape == mic.shape
    assert out["state_probs"].shape == (mic.shape[0], mic.shape[2], 3)


@pytest.mark.parametrize("mask_domain", ["mag", "com"])
def test_streaming_matches_offline_for_both_mask_domains(mask_domain):
    """两种掩码域的逐帧 streaming 与 offline 前向一致。"""
    torch.manual_seed(0)
    model = _small_dbgnet(mask_domain=mask_domain).eval()
    stream = StreamingDBGNet(model).eval()
    mic, far = _specs(n_frames=16)

    with torch.no_grad():
        offline = model({"mic_spectra": mic, "farend_spectra": far})[
            "nearend_estimate_spectra"
        ]
        cache = stream.init_cache(mic.shape[0], torch.device("cpu"), torch.float32)
        frames = []
        for index in range(mic.shape[2]):
            enhanced, _, cache = stream.forward_streaming(
                mic[:, :, index : index + 1],
                far[:, :, index : index + 1],
                cache,
            )
            frames.append(enhanced)
        streamed = torch.cat(frames, dim=2)

    assert streamed.shape == offline.shape
    assert float((offline - streamed).abs().max()) < 1e-5


@pytest.mark.parametrize(("mask_domain", "expected_channels"), [("mag", 1), ("com", 2)])
def test_streaming_cache_input_channels_follow_mask_domain(mask_domain, expected_channels):
    """cache 名字/顺序不变，只有首层输入 cache 的通道数随掩码域变化。"""
    model = _small_dbgnet(mask_domain=mask_domain).eval()
    stream = StreamingDBGNet(model).eval()
    cache = stream.init_cache(1, torch.device("cpu"), torch.float32)

    assert tuple(cache) == stream.CACHE_KEYS
    assert tuple(cache["mic_in"].shape[1:2]) == (expected_channels,)
    assert tuple(cache["far_in"].shape[1:2]) == (expected_channels,)
    assert cache["mic_in"].shape[2] == 81  # 41 low bins + 40 ERB bands
    for key in stream.CACHE_KEYS:
        if key not in ("mic_in", "far_in"):
            assert tuple(cache[key].shape) == tuple(
                stream.init_cache(1, torch.device("cpu"), torch.float32)[key].shape
            )


def test_paper_geometry_and_param_count():
    """论文 Table I 几何由默认规则给出：ERB 129、F-GRU 192、T-GRU 128、参数量 ~1.5M。"""
    assert "fgru_hidden" not in PAPER_MODEL_ARGS  # 必须走默认推导，而不是写死数值
    model = DBGNet(**PAPER_MODEL_ARGS)
    assert model.mask_domain == "mag"
    assert tuple(model._freq_sizes) == (129, 129, 64, 31, 15)
    # 论文 Table I：Concatenation 192 → F-GRU 每方向 192（双向输出 384）→ T-GRU 128。
    assert model.fgru.hidden_size == 128 + 64
    assert model.fgru.input_size == 192
    assert model.tgru.input_size == 384
    assert model.tgru.hidden_size == 128

    erb_in = torch.randn(1, 257, 3)
    assert tuple(model.erb.bm(erb_in).shape) == (1, 129, 3)
    assert tuple(model.erb.bs(model.erb.bm(erb_in)).shape) == (1, 257, 3)

    params = sum(p.numel() for p in model.parameters())
    assert abs(params - PAPER_PARAMS) <= 0.05 * PAPER_PARAMS


def test_table_i_layer_shapes_match_forward_spectra():
    """论文 Table I 的逐层形状必须与 forward_spectra 的实际张量逐轴一致。

    B=2 使 batch 轴可判定；F-GRU/T-GRU 两行同时校验 Table I 的 (B×T)/(B×F) 折叠约定。
    """
    from tasks.aec.models import register_aec_models

    cfg = load_merged_config(PAPER_CONFIG)
    register_aec_models(only="DBGNet")
    model = build_model(cfg["model"], cfg.get("stft")).eval()
    torch.manual_seed(0)
    mic = torch.randn(2, 257, 63, 2)
    far = torch.randn(2, 257, 63, 2)

    assert _table_i_layer_shapes(model, mic, far) == TABLE_I_LAYER_SHAPES


@pytest.mark.parametrize(
    "config_name",
    [
        "dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml",
        "dbgnet_small_32k_10ms_delay80ms_loss_hybrid.yaml",
        "dbgnet_small_32k_10ms_delay80ms_loss_mbrcl.yaml",
    ],
)
def test_production_dbgnet_configs_follow_paper_fgru_rule(config_name):
    """三份生产配置不写 fgru_hidden，F-GRU 宽度按论文规则 mic[-1]+far[-1] 推导。"""
    from tasks.aec.models import register_aec_models

    cfg = load_merged_config(
        REPO_ROOT / "configs/aec/microsoft_challenge/dbgnet" / config_name
    )
    register_aec_models(only="DBGNet")
    assert "fgru_hidden" not in cfg["model"]["args"]
    args = cfg["model"]["args"]
    assert args["mic_channels"][-1] + args["far_channels"][-1] == 96
    model = build_model(cfg["model"], cfg.get("stft"))
    assert model.fgru.hidden_size == 96
    assert model.fgru.input_size == 96
    assert model.tgru.input_size == 192
    assert model.tgru.hidden_size == args["mic_channels"][-1]


def test_paper_config_builds_paper_geometry():
    """论文对齐配置构建出的模型与论文几何一致（F-GRU 192 由默认规则推导）。"""
    from tasks.aec.models import register_aec_models

    cfg = load_merged_config(PAPER_CONFIG)
    register_aec_models(only="DBGNet")
    model = build_model(cfg["model"], cfg.get("stft"))

    assert cfg["model"]["args"]["mask_domain"] == "mag"
    assert "fgru_hidden" not in cfg["model"]["args"]
    assert model.fgru.hidden_size == 192
    assert tuple(model._freq_sizes) == (129, 129, 64, 31, 15)
    assert (model.n_fft, model.hop_length, model.win_length) == (512, 256, 512)
    assert model.sample_rate == 16000
    assert model.erb.low_bins == 65
    assert model.erb.erb_bands == 64
