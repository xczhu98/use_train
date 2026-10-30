"""UniPASE 4 阶段训练流水线中 Stage 2/3/4 的 GAN generator 包装器。

上游来源：<https://github.com/Xiaobin-Rong/UniPASE>；对齐其中 ``train/`` 下各训练脚本的
generator 结构。本仓可运行 recipe 分别为
``configs/se/urgent2026/pase/unipase/unipase_phase2_vocoder_gan.yaml``、
``configs/se/urgent2026/pase/unipase/unipase_phase3_adapter_gan.yaml`` 与
``configs/se/urgent2026/pase/unipase/unipase_phase4_postnet_gan.yaml``：

- ``UniPASEVocoderGenerator``（Stage 2）：官方 ``train/train_vocoder.py:268-269``——
  冻结 DeWavLM-Omni encoder（output_layer=1，mask=False）+ 可训练
  ``VocosVocoder``，clean-only 自编码训练（配合 ``prob_target_only: 1.0``，
  ``PASEGANTrainer._forward_generator`` 传入的 noisy 即 clean）。
- ``UniPASEAdapterGenerator``（Stage 3）：官方 ``train/train_adapter.py:240-243``——
  冻结 DeWavLM-Omni encoder（output_layer=[1,24]）+ 可训练 ``VocosAdapter``，
  输出增强 L1 特征；配合 ``UniPASEAdapterGANTrainer``（特征域 MSE + LSGAN）。
- ``UniPASEPostNetGenerator``（Stage 4）：官方 ``train/train_postnet.py:268-275``——
  冻结 encoder+adapter+vocoder 16k 链路（no_grad），重采样到 48k 后送入可训练
  ``TFGridNet.forward``（训练模式全时域残差，非 infer()），输出 48k 波形。

三个包装器都只暴露训练所需的模块；encoder/adapter/decoder 的冻结语义与官方一致
（官方脚本中它们从未被加入 optimizer，也未调用 .train()）。三者都实现
``val_audio_outputs()``，按官方各阶段 ``_validation_epoch`` 的中间波形与文件命名
返回验证音频；该路径只在 validation、rank 0 与 ``trainer.val_audio`` 启用时执行。
"""

import logging
from collections.abc import Mapping

import torch
import torch.nn as nn
import torchaudio

from core.utils.registry import MODEL_REGISTRY

from ..common.utils import (
    align_waveform_length,
    peak_normalize_to,
    resample_waveform_and_lengths,
    scale_lengths_to,
)
from .adapter import VocosAdapter
from .postnet import TFGridNet
from .vocoder import VocosVocoder
from .wavlm_frontend_plc import WavLM_feat_PLC

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register("UniPASEVocoderGenerator")
class UniPASEVocoderGenerator(nn.Module):
    """Stage 2 vocoder GAN generator：冻结 DeWavLM-Omni L1 encoder + 可训练 VocosVocoder。

    ``output_sample_rate = 16000``：多采样率训练时 PASEGANTrainer 据此把
    clean/lengths 对齐到 16k，再计算重建/判别损失。

    Args:
        encoder_ckpt_path: DeWavLM-Omni checkpoint 路径（Stage 1 产物；空则随机初始化
            的 WavLM-Large 结构，仅用于 smoke test）。
        sr_in: 输入波形采样率（官方 ``train_vocoder.py`` 固定 16k，对应 YAML
            ``data.sample_rate``）。
        其余参数为 ``VocosVocoder`` 结构参数，默认值对齐官方
        ``configs/cfg_train_vocoder.yaml:5-11``。
    """

    output_sample_rate = 16000
    val_audio_sample_rate = 16000

    # validation 验证音频：官方 train_vocoder.py:365-378（UniPASE Stage 2）。
    val_audio_stage = "vocoder"
    val_audio_vocoder_roles = ()

    def __init__(
        self,
        encoder_ckpt_path: str = "",
        sr_in: int = 16000,
        input_channels: int = 1024,
        dim: int = 1024,
        intermediate_dim: int = 3072,
        num_layers: int = 12,
        num_res: int = 4,
        num_attn: int = 1,
        n_fft: int = 1280,
        hop_length: int = 320,
        generator_ckpt_path: str = "",
        **kwargs,
    ):
        super().__init__()
        for key in ("win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"UniPASEVocoderGenerator got unexpected keyword arguments: {sorted(kwargs)}")
        self.sr_in = sr_in
        # 官方 encoder 用 feature_extractor_plc.WavLM_feat 但 mask=False
        # （train_vocoder.py:268），PLC 路径不触发，enable_plc=False 行为等价
        self.encoder = WavLM_feat_PLC(
            wavlm_ckpt_path=encoder_ckpt_path,
            output_layer=1,
            load_pretrained=bool(encoder_ckpt_path),
            frozen=True,
            enable_plc=False,
        )
        self.vocoder = (
            VocosVocoder.from_pretrained(generator_ckpt_path, frozen=False)
            if generator_ckpt_path
            else VocosVocoder(
                input_channels=input_channels,
                dim=dim,
                intermediate_dim=intermediate_dim,
                num_layers=num_layers,
                num_res=num_res,
                num_attn=num_attn,
                n_fft=n_fft,
                hop_length=hop_length,
            )
        )

    def forward(
        self,
        wav: torch.Tensor,
        lengths: torch.Tensor = None,
        *,
        fs=None,
    ) -> torch.Tensor:
        """Args: wav (B, L)（clean-only 训练下 noisy == clean），采样率 ``fs``。

        ``fs`` 为 batch 实际采样率（None 视为 ``sr_in``）。多采样率训练时先
        重采样到 ``sr_in`` 再送冻结 encoder；``fs == sr_in`` 时行为不变。
        Returns: (B, L') 16k 波形，L' 与重采样后的输入等长。
        """
        sr = self.sr_in if fs is None else int(fs)
        wav, lengths = resample_waveform_and_lengths(wav, lengths, sr, self.sr_in)
        n_samples = wav.shape[-1]
        # 官方 train_vocoder.py:268-269：feat = encoder(true_wav, mask=False).transpose(1,2)
        with torch.no_grad():
            feat = self.encoder(wav, sr=self.sr_in, mask=False, lengths=lengths)  # (B, T, D)
        esti = self.vocoder(feat.transpose(1, 2))  # (B, L')
        # 官方 align_length(esti_wav, true_wav)（train_vocoder.py:271）
        return align_waveform_length(esti, n_samples)

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
        """官方 ``train_vocoder.py:365-378``：``{uid}_true.wav`` + 当前 esti 波形。

        clean-only 训练下 noisy == clean；esti 取 validation 前向的原始输出
        （``raw_enhanced``），不重复前向也不使用 clamp 后的 sanitize 结果。
        """
        esti = raw_enhanced
        if esti is None and forward_dict is not None:
            esti = forward_dict["enhanced"]
        if esti is None:
            raise ValueError(
                "UniPASEVocoderGenerator val audio requires the validation "
                "forward output (raw_enhanced/forward_dict)"
            )
        return {
            "true": clean_wav,
            "esti_epoch{epoch:03d}": esti,
        }


@MODEL_REGISTRY.register("UniPASEAdapterGenerator")
class UniPASEAdapterGenerator(nn.Module):
    """Stage 3 adapter GAN generator：冻结 DeWavLM-Omni [L1,L24] encoder + 可训练 VocosAdapter。

    输出 (B, T, 1024) 增强 L1 特征而非波形，配合
    ``tasks.se.trainer.unipase_adapter_gan_trainer.UniPASEAdapterGANTrainer``（该 trainer
    在 ``_forward_generator`` 中把 noisy 和 clean 一起传入本 forward）。

    Args:
        encoder_ckpt_path: DeWavLM-Omni checkpoint 路径（Stage 1 产物）。
        enable_plc: 学生侧 noisy 特征是否启用 PLC 掩码（官方 train_adapter.py:241
            对 noisy 用 mask=True）。
        decoder_ckpt_path / wavlm_decoder_ckpt_path: 官方验证音频使用的冻结
            vocoder（``Vocoder_DWO-L1`` / ``Vocoder_WavLM-L24``）。仅在
            ``val_audio_outputs()`` 首次调用时懒加载，不注册为子模块，
            因此不改变训练 forward 与 state_dict/checkpoint 内容。
        其余参数为 ``VocosAdapter`` 结构参数，默认值对齐官方
        ``configs/cfg_train_adapter.yaml:11-16``。
    """

    # validation 验证音频：官方 train_adapter.py:340-380（UniPASE Stage 3）。
    val_audio_stage = "adapter"
    val_audio_vocoder_roles = ("vocoder", "wavlm_vocoder")
    val_audio_sample_rate = 16000

    def __init__(
        self,
        encoder_ckpt_path: str = "",
        enable_plc: bool = True,
        input_channels: int = 1024,
        dim: int = 1024,
        intermediate_dim: int = 3072,
        num_layers: int = 12,
        num_res: int = 4,
        num_attn: int = 1,
        output_channels: int = 1024,
        decoder_ckpt_path: str = "",
        wavlm_decoder_ckpt_path: str = "",
        **kwargs,
    ):
        super().__init__()
        for key in ("n_fft", "hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"UniPASEAdapterGenerator got unexpected keyword arguments: {sorted(kwargs)}")
        self.enable_plc = enable_plc
        self.decoder_ckpt_path = str(decoder_ckpt_path or "")
        self.wavlm_decoder_ckpt_path = str(wavlm_decoder_ckpt_path or "")
        # plain dict：冻结 vocoder 不得进入 state_dict / optimizer。
        self._val_vocoders: dict[str, nn.Module] = {}
        self.encoder = WavLM_feat_PLC(
            wavlm_ckpt_path=encoder_ckpt_path,
            output_layer=[1, 24],
            load_pretrained=bool(encoder_ckpt_path),
            frozen=True,
            enable_plc=enable_plc,
        )
        self.adapter = VocosAdapter(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            num_res=num_res,
            num_attn=num_attn,
            output_channels=output_channels,
        )

    def forward(
        self,
        noisy_wav: torch.Tensor,
        clean_wav: torch.Tensor = None,
        lengths: torch.Tensor = None,
        *,
        fs=None,
    ):
        """官方 train_adapter.py:240-243 的单步逻辑。

        Args:
            noisy_wav: (B, L) 带噪波形（16k）。
            clean_wav: (B, L) 干净波形，用于计算 MSE 目标 clean L1 特征；为 None
                时只返回增强特征（推理/特征前向）。
        Returns:
            clean_wav 提供时：``(enh_feat, target_feat)``，均为 (B, T, D)；
            否则仅 ``enh_feat``。
        """
        # 官方：target, _ = encoder(clean, mask=False)；input_a, input_p = encoder(noisy, mask=True)
        input_a, input_p = self.encoder(
            noisy_wav, sr=fs, mask=self.enable_plc, lengths=lengths
        )
        enh_feat = self.adapter(input_a, input_p)  # (B, T, D) 增强 L1
        if clean_wav is None:
            return enh_feat
        with torch.no_grad():
            target_feat, _ = self.encoder(clean_wav, sr=fs, mask=False, lengths=lengths)
        return enh_feat, target_feat

    def _load_val_vocoder(self, role: str, ckpt_path: str) -> nn.Module:
        """懒加载官方验证 vocoder（UniPASE ``VocosVocoder``，输入 ``(B, D, T)``）。

        独立调用 ``val_audio_outputs``（未注入 ``vocoders``）时的回退路径；
        trainer 驱动场景由 ``ValAudioWriter.load_vocoders`` 统一加载并注入，
        本路径不会命中，两套缓存不会同时占用显存。
        """
        module = self._val_vocoders.get(role)
        if module is None:
            module = VocosVocoder.from_pretrained(ckpt_path, frozen=True)
            module.eval()
            module = module.to(next(self.parameters()).device)
            self._val_vocoders[role] = module
        return module

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
        """官方 ``train_adapter.py:340-380`` 的验证音频中间量。

        ``recon = decoder(target_feat)``、``enh0 = wavlm_decoder(input_p)``、
        ``enhanced = decoder(enh_feat)``；recon/enhanced 对齐 clean 长度，三者
        各自峰值归一到 0.9。``target_feat``/``enh_feat`` 直接复用 validation
        前向结果，仅重算一次冻结 encoder 取 ``input_p``。
        """
        if forward_dict is None:
            raise ValueError(
                "UniPASEAdapterGenerator val audio requires the validation forward_dict"
            )
        decoder = None
        wavlm_decoder = None
        if vocoders:
            decoder = vocoders.get("vocoder")
            wavlm_decoder = vocoders.get("wavlm_vocoder")
        if decoder is None:
            if not self.decoder_ckpt_path:
                raise ValueError(
                    "UniPASEAdapterGenerator val audio requires "
                    "trainer.val_audio.vocoder_ckpt_path"
                )
            decoder = self._load_val_vocoder("vocoder", self.decoder_ckpt_path)
        if wavlm_decoder is None:
            if not self.wavlm_decoder_ckpt_path:
                raise ValueError(
                    "UniPASEAdapterGenerator val audio requires "
                    "trainer.val_audio.wavlm_vocoder_ckpt_path"
                )
            wavlm_decoder = self._load_val_vocoder(
                "wavlm_vocoder", self.wavlm_decoder_ckpt_path
            )
        target_feat = forward_dict["target_feat"]
        enh_feat = (
            raw_enhanced if raw_enhanced is not None else forward_dict["enhanced"]
        )
        with torch.no_grad():
            _input_a, input_p = self.encoder(
                noisy_wav, sr=fs, mask=self.enable_plc, lengths=lengths
            )
            recon = decoder(target_feat.transpose(1, 2))
            enh0 = wavlm_decoder(input_p.transpose(1, 2))
            enhanced = decoder(enh_feat.transpose(1, 2))
        sr = int(fs) if fs is not None else self.val_audio_sample_rate
        noisy_out = noisy_wav
        clean_out = clean_wav
        if sr != self.val_audio_sample_rate:
            noisy_out = torchaudio.functional.resample(
                noisy_wav, orig_freq=sr, new_freq=self.val_audio_sample_rate
            )
            clean_out = torchaudio.functional.resample(
                clean_wav, orig_freq=sr, new_freq=self.val_audio_sample_rate
            )
        n_samples = clean_out.shape[-1]
        recon = align_waveform_length(recon, n_samples)
        enhanced = align_waveform_length(enhanced, n_samples)
        return {
            "noisy": noisy_out,
            "clean": clean_out,
            "recon": peak_normalize_to(recon),
            "enh0": peak_normalize_to(enh0),
            "enh_epoch{epoch:03d}": peak_normalize_to(enhanced),
        }


@MODEL_REGISTRY.register("UniPASEPostNetGenerator")
class UniPASEPostNetGenerator(nn.Module):
    """Stage 4 postnet GAN generator：冻结 16k 增强链路 + 可训练 TFGridNet PostNet.

    ``output_sample_rate = 48000``：多采样率训练时 PASEGANTrainer 据此把
    clean/lengths 对齐到 48k，再计算重建/判别损失。

    Args:
        encoder_ckpt_path: DeWavLM-Omni checkpoint（Stage 1 产物）。
        adapter_ckpt_path: VocosAdapter checkpoint（Stage 3 产物）。
        vocoder_ckpt_path: Vocoder_DWO-L1 checkpoint（Stage 2 产物）。
        sr_in: 输入/输出波形采样率（官方 cfg_train_postnet.yaml 固定 48000）。
        enable_plc: noisy 特征是否启用 PLC 掩码（官方 train_postnet.py:269 调用
            ``encoder(noisy, sr=fs)``，官方 PLC 版 forward 默认 mask=True）。
        postnet_kwargs: ``TFGridNet`` 结构参数（官方 cfg_train_postnet.yaml:11-17:
            n_layers=5, lstm_hidden_units=100, attn_n_head=4,
            attn_qk_output_channel=2, emb_dim=48）。
    """

    output_sample_rate = 48000
    val_audio_sample_rate = 48000

    # validation 验证音频：官方 train_postnet.py:375-400（UniPASE Stage 4）。
    val_audio_stage = "postnet"
    val_audio_vocoder_roles = ()

    def __init__(
        self,
        encoder_ckpt_path: str = "",
        adapter_ckpt_path: str = "",
        vocoder_ckpt_path: str = "",
        sr_in: int = 48000,
        enable_plc: bool = True,
        postnet_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__()
        for key in ("n_fft", "hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"UniPASEPostNetGenerator got unexpected keyword arguments: {sorted(kwargs)}")
        # review P2-3：内部 16k→48k 重采样与 postnet(sr_in=48000) 均按官方
        # 48k 训练配置硬编码，且不重采样回原率；sr_in != 48000 时输出长度/采样率
        # 会静默错误，因此 fail closed（官方仅支持 48k 训练）
        if sr_in != 48000:
            raise ValueError(
                f"UniPASEPostNetGenerator 仅支持 sr_in=48000（官方 Stage-4 配置），得到 {sr_in}"
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
        # Stage 4 需要已训练好的 adapter/vocoder 权重，缺失时 fail closed
        # （随机初始化的冻结链路会产生无意义输入，静默训练出错误 PostNet）
        if not adapter_ckpt_path or not vocoder_ckpt_path:
            raise ValueError(
                "UniPASEPostNetGenerator 需要 adapter_ckpt_path 与 vocoder_ckpt_path "
                "（Stage 3 / Stage 2 的产物）；仅 smoke test 可改用 _init_random_chain_for_test()。"
            )
        self.adapter = VocosAdapter.from_pretrained(adapter_ckpt_path, frozen=True)
        self.decoder = VocosVocoder.from_pretrained(vocoder_ckpt_path, frozen=True)
        self.postnet = TFGridNet(**(postnet_kwargs or {}))

    def train(self, mode: bool = True):
        """只让可训练 PostNet 进入 train；冻结的 adapter/vocoder 保持 eval。

        官方 Stage-4 只对 PostNet 调 ``.train()``，冻结链路始终 eval
        （``train_postnet.py:193-199``）。``BaseTrainer`` 会递归调用
        ``generator.train()``，若不在 wrapper 边界拦住，Vocos 的 dropout 会在
        训练时生效。
        """
        super().train(mode)
        self.adapter.eval()
        self.decoder.eval()
        return self

    def _init_random_chain_for_test(self, adapter_kwargs: dict = None, vocoder_kwargs: dict = None) -> None:
        """仅供测试：用随机初始化的 adapter/vocoder 替代缺失 checkpoint。

        训练配置不得使用本方法（__init__ 已对空路径 fail closed）。
        """
        self.adapter = VocosAdapter(**(adapter_kwargs or {}))
        self.decoder = VocosVocoder(**(vocoder_kwargs or {}))
        for m in (self.adapter, self.decoder):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def _frozen_prefix(self, noisy_wav: torch.Tensor, lengths: torch.Tensor, fs):
        """冻结 16k 增强链路：返回 (48k 输入, lengths, esti_16k, esti_48k)。"""
        sr = self.sr_in if fs is None else int(fs)
        noisy_wav, lengths = resample_waveform_and_lengths(
            noisy_wav, lengths, sr, self.sr_in
        )
        # 官方：冻结链路整体在 torch.no_grad() 内（train_postnet.py:268）
        with torch.no_grad():
            feat_a_noisy, feat_p_enh = self.encoder(
                noisy_wav, sr=self.sr_in, mask=self.enable_plc, lengths=lengths
            )
            feat_a_enh = self.adapter(feat_a_noisy, feat_p_enh)
            esti_wav_16k = self.decoder(feat_a_enh.transpose(1, 2))  # (B, L16)
            esti_wav_48k = torchaudio.functional.resample(
                esti_wav_16k, orig_freq=16000, new_freq=48000
            )
        return noisy_wav, lengths, esti_wav_16k, esti_wav_48k

    def forward(
        self,
        noisy_wav: torch.Tensor,
        lengths: torch.Tensor = None,
        *,
        fs=None,
    ) -> torch.Tensor:
        """官方 train_postnet.py:268-277 的单步逻辑。

        Args: noisy_wav (B, L)，采样率 ``fs``（None 视为 ``sr_in``）。
            ``fs`` 为 batch 实际采样率；多采样率训练时先重采样到 48k 再走
            冻结链路，``lengths`` 同步换算，``fs == sr_in`` 时行为不变。
        Returns: (B, L') 48k 增强波形（TFGridNet 训练模式全时域残差输出），
            L' 与重采样后的输入等长。
        """
        noisy_wav, lengths, _esti_16k, esti_wav_48k = self._frozen_prefix(
            noisy_wav, lengths, fs
        )
        n_samples = noisy_wav.shape[-1]
        # 训练模式 forward（全时域残差），官方 train_postnet.py:275
        lengths_48k = scale_lengths_to(
            lengths, noisy_wav.shape[-1], esti_wav_48k.shape[-1]
        )
        esti_wav = self.postnet(esti_wav_48k, sr_in=48000, lengths=lengths_48k)
        # 官方 align_length(esti_wav, clean)；48k 下 noisy 与 clean 等长
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
        """官方 ``train_postnet.py:375-400``（UniPASE Stage 4）。

        ``_enh_16k`` 是 16k 中间波形重采样到 48k 后按 48k 保存（官方同名文件
        实际内容为 48k）；``_enh_epoch`` 为完整链 48k 输出。
        """
        noisy_48k, _lengths, esti_16k, _esti_48k = self._frozen_prefix(
            noisy_wav, lengths, fs
        )
        enhanced = raw_enhanced
        if enhanced is None and forward_dict is not None:
            enhanced = forward_dict["enhanced"]
        if enhanced is None:
            raise ValueError(
                "UniPASEPostNetGenerator val audio requires the validation "
                "forward output (raw_enhanced/forward_dict)"
            )
        enh_16k_48sr = torchaudio.functional.resample(
            esti_16k, orig_freq=16000, new_freq=48000
        )
        return {
            "noisy": noisy_48k,
            "clean": clean_wav,
            "enh_16k": enh_16k_48sr,
            "enh_epoch{epoch:03d}": enhanced,
        }
