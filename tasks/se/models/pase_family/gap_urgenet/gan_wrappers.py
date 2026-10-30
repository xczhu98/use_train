"""GAP-URGENet Predictor / PostNet 训练包装器。

``GAPURGENetPredictor`` 将 Predictor 适配为 SETrainer 接口。
``GAPURGENetPostNetGenerator`` 冻结 encoder、adapter、vocoder、predictor
四条链路，将 48 kHz 双通道输入送入可训练 PostNet；生成支路在堆叠前必须与
预测支路对齐长度。PostNet 支持 ``legacy_centered`` 和 ``paper_variance``
normalization 模式。
"""

import logging
from collections.abc import Mapping

import torch
import torch.nn as nn
import torchaudio

from core.utils.registry import MODEL_REGISTRY

from ..unipase.adapter import VocosAdapter
from ..unipase.vocoder import VocosVocoder
from ..unipase.wavlm_frontend_plc import WavLM_feat_PLC
from ..common.utils import (
    align_waveform_length,
    resample_waveform_and_lengths,
    scale_lengths_to,
)
from .postnet import TFGridNet as PostNet
from .predictor import TFGridNet as Predictor

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register("GAPURGENetPredictor")
class GAPURGENetPredictor(Predictor):
    """Phase 1 Predictor 训练封装：TFGridNet 子类，仅适配 SETrainer forward 接口。

    结构参数与 ``predictor.tfgridnet.TFGridNet`` 完全一致（官方发布配置见
    gap-urgenet/configs/cfg_train_predictor.yaml:1-15：n_layers=10,
    lstm_hidden_units=200, attn_n_head=8, attn_qk_output_channel=4,
    emb_dim=96, emb_ks=4, emb_hs=1，参数量 22.09M）。

    因子类不改任何模块，state_dict key 与官方 ``models/predictor/tfgridnet.py``
    逐字一致：训练 checkpoint 可直接经
    ``predictor.tfgridnet.TFGridNet.from_pretrained`` 语义导出为 Predictor.pt。
    """

    input_domain = "waveform"
    output_domain = "waveform"
    forward_kind = "estimate"
    frontend_owner = "model"

    # validation 验证音频：官方 gap-urgenet/train/train_predictor.py:250-263。
    val_audio_stage = "predictor"
    val_audio_vocoder_roles = ()

    def forward(
        self,
        noisy_wav: torch.Tensor,
        lengths: torch.Tensor = None,
        *,
        fs=None,
    ) -> torch.Tensor:
        """SETrainer 接口。Args: noisy_wav (B, L)，采样率 fs（None 视为 16000）。
        Returns: (B, L) 同采样率增强波形。"""
        sr = int(fs) if fs is not None else 16000
        return super().forward(noisy_wav, sr=sr)

    def val_audio_outputs(
        self,
        noisy_wav: torch.Tensor,
        clean_wav: torch.Tensor,
        lengths: torch.Tensor = None,
        *,
        fs=None,
        vocoders=None,
        sample_settings=None,
        forward_dict=None,
        raw_enhanced: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """官方 GAP Phase 1 验证音频：三路共用 ``0.9 / max(|noisy|,|clean|,|enh|)``。"""
        sr = int(fs) if fs is not None else 16000
        enhanced = self(noisy_wav, lengths, fs=sr)
        peak = torch.stack(
            [
                noisy_wav.abs().amax(),
                clean_wav.abs().amax(),
                enhanced.abs().amax(),
            ]
        ).amax()
        scale = 0.9 / (peak + 1e-12)
        return {
            "noisy": noisy_wav * scale,
            "clean": clean_wav * scale,
            "enh_epoch{epoch:03d}": enhanced * scale,
        }


@MODEL_REGISTRY.register("GAPURGENetPostNetGenerator")
class GAPURGENetPostNetGenerator(nn.Module):
    """Phase 2 PostNet GAN generator：冻结生成+预测双支路 + 可训练 PostNet 融合.

    ``output_sample_rate = 48000``：多采样率训练时 PASEGANTrainer 据此把
    clean/lengths 对齐到 48k，再计算重建/判别损失。

    Args:
        encoder_ckpt_path: DeWavLM-Omni checkpoint。
        adapter_ckpt_path: VocosAdapter checkpoint。
        vocoder_ckpt_path: VocosVocoder checkpoint。
        predictor_ckpt_path: Predictor checkpoint（Phase 1 产物；默认可先用官方
            发布权重）。
        sr_in: 输入/输出波形采样率（官方 cfg_train_postnet.yaml 固定 48000）。
        enable_plc: noisy 特征是否启用 PLC 掩码（对齐官方
            ``models/gap_urgenet.py:52`` 的 ``encoder(x, sr=sr_in, mask=enable_plc)``）。
        postnet_kwargs: PostNet ``TFGridNet`` 结构参数（官方
            cfg_train_postnet.yaml:17-23：n_imics=2, n_layers=5,
            lstm_hidden_units=100, attn_n_head=4, attn_qk_output_channel=2,
            emb_dim=96，参数量 4.78M）。``normalization_mode=legacy_centered``
            保持官方发布 checkpoint 数值；新训练使用 ``paper_variance``，按
            TF-GridNet 论文只除 mixture std、不减均值。
    """

    output_sample_rate = 48000
    val_audio_sample_rate = 48000

    # validation 验证音频：官方 gap-urgenet/train/train_postnet.py:400-415。
    val_audio_stage = "gap_postnet"
    val_audio_vocoder_roles = ()

    def __init__(
        self,
        encoder_ckpt_path: str = "",
        adapter_ckpt_path: str = "",
        vocoder_ckpt_path: str = "",
        predictor_ckpt_path: str = "",
        sr_in: int = 48000,
        enable_plc: bool = True,
        postnet_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__()
        for key in ("n_fft", "hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"GAPURGENetPostNetGenerator got unexpected keyword arguments: {sorted(kwargs)}")
        # 内部按 48 kHz 处理且不重采样回原率，其他采样率会静默产生错误输出，必须 fail-closed。
        if sr_in != 48000:
            raise ValueError(
                f"GAPURGENetPostNetGenerator 仅支持 sr_in=48000（官方 Phase-2 配置），得到 {sr_in}"
            )
        self.sr_in = sr_in
        self.enable_plc = enable_plc

        self.encoder = WavLM_feat_PLC(
            wavlm_ckpt_path=encoder_ckpt_path,
            output_layer=[1, 24],
            load_pretrained=bool(encoder_ckpt_path),
            frozen=True,
            enable_plc=enable_plc,
        )
        # 缺少冻结链路权重时必须 fail-closed，避免随机输入训练 PostNet。
        if not adapter_ckpt_path or not vocoder_ckpt_path or not predictor_ckpt_path:
            raise ValueError(
                "GAPURGENetPostNetGenerator 需要 adapter_ckpt_path、vocoder_ckpt_path 与 "
                "predictor_ckpt_path（UniPASE Stage 2/3 与 GAP Phase 1 的产物）；"
                "仅 smoke test 可改用 _init_random_chain_for_test()。"
            )
        self.adapter = VocosAdapter.from_pretrained(adapter_ckpt_path, frozen=True)
        self.decoder = VocosVocoder.from_pretrained(vocoder_ckpt_path, frozen=True)
        self.predictor = Predictor.from_pretrained(predictor_ckpt_path, frozen=True)
        self.postnet = PostNet(**(postnet_kwargs or {}))

    def train(self, mode: bool = True):
        """只让可训练 PostNet 进入 train；冻结的生成/预测支路保持 eval。

        官方 Phase-2 只对 PostNet/D 调 ``.train()``，冻结链路始终 eval
        （``train_postnet.py:85-89,193-199``）。
        """
        super().train(mode)
        self.adapter.eval()
        self.decoder.eval()
        self.predictor.eval()
        return self

    def _init_random_chain_for_test(
        self,
        adapter_kwargs: dict = None,
        vocoder_kwargs: dict = None,
        predictor_kwargs: dict = None,
    ) -> None:
        """仅供测试：用随机初始化的 adapter/vocoder/predictor 替代缺失 checkpoint。

        训练配置不得使用本方法（__init__ 已对空路径 fail closed）。
        """
        self.adapter = VocosAdapter(**(adapter_kwargs or {}))
        self.decoder = VocosVocoder(**(vocoder_kwargs or {}))
        self.predictor = Predictor(**(predictor_kwargs or {}))
        for m in (self.adapter, self.decoder, self.predictor):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def _frozen_prefix(self, noisy_wav: torch.Tensor, lengths: torch.Tensor, fs):
        """冻结生成/预测双支路：返回 (48k 输入, lengths, gen_16k, gen_48k, pred_48k, 双通道输入)。"""
        sr = self.sr_in if fs is None else int(fs)
        noisy_wav, lengths = resample_waveform_and_lengths(
            noisy_wav, lengths, sr, self.sr_in
        )
        # 四条冻结链路必须在 torch.no_grad() 内。
        with torch.no_grad():
            # 生成支路（官方 :52-55）
            feat_a_noisy, feat_p_enh = self.encoder(
                noisy_wav, sr=self.sr_in, mask=self.enable_plc, lengths=lengths
            )
            feat_a_enh = self.adapter(feat_a_noisy, feat_p_enh)
            gen_16k = self.decoder(feat_a_enh.transpose(1, 2))  # (B, L16)
            gen_48k = torchaudio.functional.resample(gen_16k, orig_freq=16000, new_freq=48000)
            # 预测支路（官方 :60-63）
            x_16k = torchaudio.functional.resample(noisy_wav, orig_freq=self.sr_in, new_freq=16000)
            lengths_16k = scale_lengths_to(lengths, noisy_wav.shape[-1], x_16k.shape[-1])
            pred_16k = self.predictor(x_16k, sr=16000, lengths=lengths_16k)
            pred_48k = torchaudio.functional.resample(pred_16k, orig_freq=16000, new_freq=48000)
            # 长度对齐后 stack 双通道（官方 :68-73）
            gen_48k = align_waveform_length(gen_48k, pred_48k.shape[-1])
            enh_2ch = torch.stack([pred_48k, gen_48k], dim=1)  # (B, 2, L)
        return noisy_wav, lengths, gen_16k, gen_48k, pred_48k, enh_2ch

    def forward(
        self,
        noisy_wav: torch.Tensor,
        lengths: torch.Tensor = None,
        *,
        fs=None,
    ) -> torch.Tensor:
        """官方 ``models/gap_urgenet.py:49-74`` 的单步逻辑（训练模式）。

        Args: noisy_wav (B, L)，采样率 ``fs``（None 视为 ``sr_in``）。
            ``fs`` 为 batch 实际采样率；多采样率训练时先重采样到 48k 再走
            冻结链路，``lengths`` 同步换算，``fs == sr_in`` 时行为不变。
        Returns: (B, L') 48k 增强波形（PostNet 训练模式 forward 输出），
            L' 与重采样后的输入等长。
        """
        noisy_wav, lengths, _gen_16k, _gen_48k, _pred_48k, enh_2ch = self._frozen_prefix(
            noisy_wav, lengths, fs
        )
        n_samples = noisy_wav.shape[-1]
        # 训练模式 forward（n_imics=2 融合，无残差无软门控）
        lengths_48k = scale_lengths_to(lengths, noisy_wav.shape[-1], enh_2ch.shape[-1])
        esti_wav = self.postnet(enh_2ch, sr_in=48000, lengths=lengths_48k)
        return align_waveform_length(esti_wav, n_samples)

    def val_audio_outputs(
        self,
        noisy_wav: torch.Tensor,
        clean_wav: torch.Tensor,
        lengths: torch.Tensor = None,
        *,
        fs=None,
        vocoders: Mapping[str, torch.nn.Module] = None,
        sample_settings=None,
        forward_dict: Mapping[str, torch.Tensor] = None,
        raw_enhanced: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """官方 ``gap-urgenet/train/train_postnet.py:400-415``。

        ``_enh_pred_16k`` / ``_enh_gen_16k`` 是 16k 两支路重采样到 48k 的中间
        波形（官方同名文件实际为 48k），``_enh_epoch`` 为完整链 48k 输出。
        """
        noisy_48k, _lengths, gen_16k, gen_48k, pred_48k, _enh_2ch = self._frozen_prefix(
            noisy_wav, lengths, fs
        )
        enhanced = raw_enhanced
        if enhanced is None and forward_dict is not None:
            enhanced = forward_dict["enhanced"]
        if enhanced is None:
            raise ValueError(
                "GAPURGENetPostNetGenerator val audio requires the validation "
                "forward output (raw_enhanced/forward_dict)"
            )
        return {
            "noisy": noisy_48k,
            "clean": clean_wav,
            "enh_pred_16k": pred_48k,
            "enh_gen_16k": gen_48k,
            "enh_epoch{epoch:03d}": enhanced,
        }
