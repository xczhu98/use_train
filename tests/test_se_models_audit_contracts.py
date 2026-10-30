"""SE 模型审计修复的公开边界回归测试。"""

import importlib
import inspect
from pathlib import Path

import pytest
import torch

from tasks.se.models.discriminative.conformer import Conformer, RelPositionalEncoding
from tasks.se.models.discriminative.gtcrn import GTCRN
from tasks.se.models.discriminative.ulunas import ULUNAS
from tasks.se.models.pase_family.pase.model import PASE
from tasks.se.models.pase_family.stupase.stupase_model import StuPASE

pytestmark = [pytest.mark.unit]

_SE_SPECTRAL_MODEL_MODULES = [
    "tasks.se.models.discriminative.bs_roformer",
    "tasks.se.models.discriminative.bsrnn",
    "tasks.se.models.discriminative.clearervoice",
    "tasks.se.models.discriminative.conformer",
    "tasks.se.models.discriminative.dccrn",
    "tasks.se.models.discriminative.gtcrn",
    "tasks.se.models.discriminative.mossformer2",
    "tasks.se.models.discriminative.mtfaa_net",
    "tasks.se.models.discriminative.tf_gridnet",
    "tasks.se.models.discriminative.tf_skimnet",
    "tasks.se.models.discriminative.ulunas",
    "tasks.se.models.generative.meanflowse.model",
    "tasks.se.models.generative.meanse.model",
]


@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (GTCRN, {}),
        (ULUNAS, {}),
        (Conformer, {"num_blocks": 1, "d_model": 8, "n_heads": 2, "d_ff": 16}),
    ],
)
def test_registered_spectral_models_reject_unknown_kwargs(model_cls, kwargs):
    with pytest.raises(TypeError, match="unexpected keyword arguments.*typo"):
        model_cls(**kwargs, typo=True)


@pytest.mark.parametrize("model_cls", [GTCRN, ULUNAS])
def test_fixed_erb_models_fail_closed_for_sample_rate_and_freq_bins(model_cls):
    model = model_cls()
    valid = torch.zeros(1, 257, 2, 2)
    with pytest.raises(ValueError, match="requires fs=16000"):
        model(valid, fs=8000)
    with pytest.raises(ValueError, match="Expected F=257"):
        model(torch.zeros(1, 129, 2, 2), fs=16000)


def test_conformer_position_cache_is_a_nonpersistent_buffer():
    encoding = RelPositionalEncoding(d_model=8, dropout=0.0)
    assert "pe" in dict(encoding.named_buffers())
    _, position = encoding(torch.zeros(1, 3, 8))
    assert position.shape == (1, 3, 8)
    assert "pe" not in encoding.state_dict()


def test_conformer_rejects_mismatched_frequency_bins():
    model = Conformer(num_blocks=1, d_model=8, n_heads=2, d_ff=16)
    with pytest.raises(ValueError, match="Expected F=257"):
        model(torch.zeros(1, 129, 2, 2))


def test_pase_requires_artifacts_unless_explicit_training_initialization():
    with pytest.raises(ValueError, match="requires wavlm_ckpt_path and vocoder_ckpt_path"):
        PASE()


def test_stupase_requires_all_inference_artifacts():
    with pytest.raises(ValueError, match="requires dewavlm_ckpt_path"):
        StuPASE()


def test_pase_rejects_single_stream_decoder_with_multi_layer_output():
    """反向方向也必须 fail closed：单流 decoder + 多层 output_layer 会静默丢声学支路。"""
    with pytest.raises(ValueError, match="use_dual_stream=False"):
        PASE(output_layer=[1, 24], use_dual_stream=False, allow_random_init=True)


def test_pase_random_init_guard_allows_eval_forward_when_training_declared():
    """allow_random_init=True 是显式“从头训练”声明：训练期 validation 会以 eval 模式
    forward（GAN val loss/指标依赖此路径），守卫不能再要求 self.training=True。"""
    model = PASE.__new__(PASE)
    torch.nn.Module.__init__(model)
    model._artifacts_ready = False

    model._allow_random_init = False
    with pytest.raises(RuntimeError, match="randomly initialized"):
        model(torch.zeros(1, 1600))

    model._allow_random_init = True
    model.eval()
    # 守卫应放行进入后续 forward；缺 encoder 报 AttributeError 而非 RuntimeError
    with pytest.raises(AttributeError):
        model(torch.zeros(1, 1600))


def test_wavlmdec_rejects_acoustic_features_in_single_stream_mode():
    from tasks.se.models.pase_family.common.vocoder_backend import WavLMDec

    decoder = WavLMDec(
        input_channels=32,
        dim=32,
        intermediate_dim=64,
        num_layers=1,
        n_fft=16,
        hop_length=4,
        use_dual_stream=False,
    )
    with pytest.raises(ValueError, match="use_dual_stream=False"):
        decoder(torch.zeros(1, 4, 32), torch.zeros(1, 4, 32))


@pytest.mark.parametrize("max_forward_seconds", [0, -1.0, float("inf")])
def test_gap_urgenet_rejects_invalid_max_forward_seconds(max_forward_seconds):
    from tasks.se.models.pase_family.gap_urgenet.gap_urgenet_model import GAP_URGENet

    with pytest.raises(ValueError, match="max_forward_seconds"):
        GAP_URGENet(max_forward_seconds=max_forward_seconds)


def test_gap_urgenet_rejects_oversized_single_shot_input():
    """超长单条输入必须显式报错（T×T 自注意力会 OOM 掉共享推理进程），而不是只告警。"""
    from tasks.se.models.pase_family.gap_urgenet.gap_urgenet_model import GAP_URGENet

    model = GAP_URGENet.__new__(GAP_URGENet)
    torch.nn.Module.__init__(model)
    model.sr_out = None
    model.enable_plc = True
    model.max_forward_seconds = 60.0
    model._encoder_pretrained = True
    model.adapter = model.decoder = model.predictor = model.postnet = torch.nn.Identity()

    with pytest.raises(RuntimeError, match="max_forward_seconds"):
        model(torch.zeros(1, 61 * 16000), fs=16000)


def test_tfgridnet_rejects_invalid_structure_with_value_error():
    """GridNet/BandSplit 的结构校验必须是 ValueError（assert 在 python -O 下会被剥离）。"""
    from tasks.se.models.pase_family.unipase.postnet.tfgrid_cws_res import (
        AllHeadPReLULayerNormalization4DC,
        BandSplit,
        GridNetV3Block,
        TFGridNet,
    )

    with pytest.raises(ValueError, match="769"):
        BandSplit(n_bands=3)(torch.zeros(1, 2, 3, 257))
    with pytest.raises(ValueError, match="6 通道"):
        BandSplit(n_bands=3).inverse(torch.zeros(1, 4, 3, 257))
    with pytest.raises(ValueError, match="input_dimension"):
        AllHeadPReLULayerNormalization4DC((2, 2, 3))
    with pytest.raises(ValueError, match="4D"):
        AllHeadPReLULayerNormalization4DC((2, 2))(torch.zeros(2, 4, 3))
    with pytest.raises(ValueError, match="emb_dim"):
        GridNetV3Block(6, 4, 1, 8, n_head=4, qk_output_channel=2)
    with pytest.raises(ValueError, match="activation"):
        GridNetV3Block(8, 4, 1, 8, n_head=2, qk_output_channel=2, activation="relu")

    postnet = TFGridNet(
        n_imics=2, n_layers=1, lstm_hidden_units=8, attn_n_head=2, attn_qk_output_channel=2, emb_dim=8
    )
    with pytest.raises(ValueError, match="n_imics"):
        postnet(torch.zeros(1, 2, 160))


def test_tfgridnet_registers_nonpersistent_hann_window_buffer():
    """STFT/iSTFT 窗注册为 non-persistent buffer，避免每次前向重建。"""
    from tasks.se.models.pase_family.gap_urgenet.predictor import TFGridNet

    model = TFGridNet(
        n_layers=1, lstm_hidden_units=8, attn_n_head=2, attn_qk_output_channel=2, emb_dim=8
    )

    out = model(torch.randn(1, 512), sr=16000)

    buffers = dict(model.named_buffers())
    window = buffers["_hann_window_512"]
    torch.testing.assert_close(window, torch.hann_window(512))
    assert out.shape == (1, 512)
    assert "_hann_window_512" not in model.state_dict()


def test_spectral_model_forward_keeps_fp32_under_cpu_autocast():
    model = GTCRN()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        enhanced = model(torch.zeros(1, 257, 2, 2))
    assert enhanced.dtype == torch.float32


def test_spectral_models_reject_empty_batch_and_time_dimensions():
    model = GTCRN()
    with pytest.raises(ValueError, match="non-empty batch and time dimensions"):
        model(torch.zeros(0, 257, 2, 2))
    with pytest.raises(ValueError, match="non-empty batch and time dimensions"):
        model(torch.zeros(1, 257, 0, 2))


@pytest.mark.parametrize("module_name", _SE_SPECTRAL_MODEL_MODULES)
def test_spectral_model_modules_do_not_hardcode_cuda_autocast(module_name):
    source = Path(inspect.getfile(importlib.import_module(module_name))).read_text(encoding="utf-8")
    assert 'autocast("cuda"' not in source
