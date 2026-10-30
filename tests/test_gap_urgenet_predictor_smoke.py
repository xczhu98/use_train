"""GAP-URGENet Phase 1 (Predictor, 16kHz) smoke-test + 级联模型形状测试。

用 tiny TFGridNet（1 层 / 8 维）驱动真实的 ``GAPURGENetPredictor``，经
``SETrainer`` + ``GAPHybridLoss`` 在 16k 数据上跑 1 个 train epoch + 1 个
val epoch；另验证：
  - 官方发布配置（cfg_train_predictor.yaml:1-15）参数量恰为 22.09M；
  - wrapper 为 TFGridNet 直接子类，state_dict key 与官方格式逐字一致；
  - 随机 tiny 五模块链路下 GAP_URGENet 级联的 resample/stack/对齐/峰值归一化。

无需官方仓库 / 真实 GAP-URGENet 权重，CI 可直接运行。
"""

from __future__ import annotations
import pytest

import shutil
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from core.data.collate import collate_fn as _collate

from core.losses.pase_family import GAPHybridLoss
from tasks.se.models.pase_family.gap_urgenet.gan_wrappers import GAPURGENetPredictor
from tasks.se.models.pase_family.gap_urgenet.gap_urgenet_model import GAP_URGENet
from tasks.se.models.pase_family.gap_urgenet.postnet import TFGridNet as PostNet
from tasks.se.models.pase_family.gap_urgenet.predictor import TFGridNet as Predictor
from tasks.se.models.pase_family.unipase.adapter import VocosAdapter
from tasks.se.models.pase_family.unipase.vocoder import VocosVocoder
from tasks.se.models.pase_family.unipase.wavlm_frontend_plc import WavLM_feat_PLC
from tasks.se.trainer.se_trainer import SETrainer

# 官方发布配置（gap-urgenet/configs/cfg_train_predictor.yaml:1-15）
_OFFICIAL_PREDICTOR_CFG = dict(
    n_layers=10,
    lstm_hidden_units=200,
    attn_n_head=8,
    attn_qk_output_channel=4,
    emb_dim=96,
    emb_ks=4,
    emb_hs=1,
)

_TINY_PREDICTOR_KW = dict(
    n_layers=1,
    lstm_hidden_units=8,
    attn_n_head=2,
    attn_qk_output_channel=2,
    emb_dim=8,
)

_TINY_POSTNET_KW = dict(
    n_imics=2,
    n_layers=1,
    lstm_hidden_units=8,
    attn_n_head=2,
    attn_qk_output_channel=2,
    emb_dim=8,
)

_TINY_WAVLM_CFG = dict(
    encoder_layers=2,
    encoder_embed_dim=32,
    encoder_ffn_embed_dim=64,
    encoder_attention_heads=2,
    layer_norm_first=True,
    gru_rel_pos=True,
    relative_position_embedding=True,
)

_TINY_ADAPTER_KW = dict(
    input_channels=32,
    dim=32,
    intermediate_dim=64,
    num_layers=2,
    num_res=2,
    num_attn=1,
    output_channels=32,
)

_TINY_VOCODER_KW = dict(
    input_channels=32,
    dim=32,
    intermediate_dim=64,
    num_layers=2,
    num_res=2,
    num_attn=1,
    n_fft=1280,
    hop_length=320,
)

_SR = 16000
_N_SAMPLES = 3200  # 0.2s @16k


class _TinyNoisyCleanDataset16k(Dataset):
    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        torch.manual_seed(300 + idx)
        clean = torch.randn(_N_SAMPLES) * 0.3
        noisy = clean + torch.randn(_N_SAMPLES) * 0.05
        return (clean, noisy, _SR, torch.tensor(_N_SAMPLES))


def test_official_config_param_count():
    """官方发布配置实例化参数量恰为 22.09M（官方 README 标注）。"""
    model = GAPURGENetPredictor(**_OFFICIAL_PREDICTOR_CFG)
    n = sum(p.numel() for p in model.parameters())
    assert abs(n / 1e6 - 22.09) < 0.01, f"params={n/1e6:.3f}M != 22.09M"


def test_wrapper_state_dict_key_compat():
    """wrapper 为 TFGridNet 直接子类：state_dict key 与官方格式逐字一致，
    训练产物可直接经 predictor.TFGridNet 语义导出为官方 Predictor.pt。"""
    wrapper = GAPURGENetPredictor(**_TINY_PREDICTOR_KW)
    plain = Predictor(**_TINY_PREDICTOR_KW)
    assert set(wrapper.state_dict().keys()) == set(plain.state_dict().keys())


def test_gap_urgenet_predictor_train_epoch():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        model = GAPURGENetPredictor(**_TINY_PREDICTOR_KW)
        ds = _TinyNoisyCleanDataset16k()
        dl = DataLoader(ds, batch_size=2, collate_fn=_collate)
        loss_fn = GAPHybridLoss(sample_rate=_SR)
        trainer = SETrainer(
            rank=0,
            world_size=1,
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            scheduler=None,
            loss_fn=loss_fn,
            train_loader=dl,
            val_loader=dl,
            config={
                "trainer": {
                    "exp_path": str(tmpdir / "exp"),
                    "epochs": 1,
                    "val_metrics": {"enabled": False},
                },
                "meta": {"allow_no_stft": True},
            },
            sample_rate=_SR,
            stft_module=None,
        )

        p = next(trainer.model.parameters())
        before = p.detach().clone()
        loss = trainer._train_epoch(1)
        assert loss is not None and torch.isfinite(torch.tensor(loss)).item()
        assert not torch.allclose(before, p.detach()), "predictor did not update"

        val_loss = trainer._validation_epoch(1)
        assert val_loss is not None and torch.isfinite(torch.tensor(val_loss)).item()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _make_tiny_wavlm_ckpt(path: Path) -> None:
    from tasks.se.models.pase_family.common.wavlm.WavLM import WavLM, WavLMConfig

    torch.manual_seed(0)
    model = WavLM(WavLMConfig(dict(_TINY_WAVLM_CFG)))
    torch.save({"cfg": dict(_TINY_WAVLM_CFG), "model": model.state_dict()}, path)


def _build_random_cascade(tmpdir: Path, sr_out=None) -> GAP_URGENet:
    """随机 tiny 五模块链路（绕过 __init__ 的大 WavLM 构造与 ckpt fail-closed，
    仅验证数据流形状，不验证数值）。"""
    wavlm_ckpt = tmpdir / "tiny_wavlm.pt"
    _make_tiny_wavlm_ckpt(wavlm_ckpt)

    model = GAP_URGENet.__new__(GAP_URGENet)
    torch.nn.Module.__init__(model)
    model.sr_out = sr_out
    model.enable_plc = True
    model.max_forward_seconds = 60.0
    model._encoder_pretrained = True
    model.encoder = WavLM_feat_PLC(
        wavlm_ckpt_path=str(wavlm_ckpt),
        output_layer=[1, 2],  # tiny 模型只有 2 层 transformer：官方 [1,24] -> [1,2]
        load_pretrained=True,
        frozen=True,
        enable_plc=True,
    )
    model.adapter = VocosAdapter(**_TINY_ADAPTER_KW)
    model.decoder = VocosVocoder(**_TINY_VOCODER_KW)
    model.predictor = Predictor(**_TINY_PREDICTOR_KW)
    model.postnet = PostNet(**_TINY_POSTNET_KW)
    for m in (model.adapter, model.decoder, model.predictor, model.postnet):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
    return model


def test_gap_urgenet_cascade_shape_and_peak_norm():
    """(1, 16000) 输入过随机 tiny 级联：验证 resample/stack/对齐/峰值归一化。"""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        torch.manual_seed(0)
        x = torch.randn(1, 16000) * 0.3
        x[:, 2 * 320 : 3 * 320] = 0.0  # 20ms 丢包 @16k，触发 PLC 路径

        # sr_out=None -> 与输入同采样率同长度
        model = _build_random_cascade(tmpdir, sr_out=None)
        y = model(x, fs=16000)
        assert y.shape == (1, 16000), y.shape
        assert torch.isfinite(y).all()
        # normalize_output=True（默认）：输出峰值归一化到输入峰值
        # （官方 inference/inference.py:84-85 行为）
        assert torch.isclose(y.abs().max(), x.abs().max(), rtol=1e-4), (
            f"peak mismatch: out={y.abs().max().item():.6f} in={x.abs().max().item():.6f}"
        )

        # sr_out=48000 -> 重采样输出 3 倍长度；不归一化时仅形状约束
        model48 = _build_random_cascade(tmpdir, sr_out=48000)
        y48 = model48(x, fs=16000, normalize_output=False)
        assert y48.shape == (1, 48000), y48.shape
        assert torch.isfinite(y48).all()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_official_config_param_count()
    test_wrapper_state_dict_key_compat()
    test_gap_urgenet_predictor_train_epoch()
    test_gap_urgenet_cascade_shape_and_peak_norm()
    print("GAP-URGENet predictor smoke tests passed.")

pytestmark = [pytest.mark.external]
