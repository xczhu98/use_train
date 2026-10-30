"""SE model registration and construction.

SE model registry overview（非完整清单；YAML 通过 ``model.type`` 选择，权威全集见
下方 ``_MODEL_REGISTRY_MAP``）：
  - GTCRN:           Lightweight real-time model, 23.67K trainable (~48K total 含 ERB), 16kHz
  - BSRNN:           Band-Split RNN（含 causal/noncausal、USR 多采样率）；usr_bsrnn_* / bsrnn_* 均用此 type；流式/ONNX 用 causal BSRNN + ``onnx/StreamingBSRNNWrapper``
  - DeepFilterNet:   DeepFilterNet2/3 or DPDFNet2/4/8 via YAML params (enc_concat, enc_lin_groups, etc.), 16kHz
  - FastEnhancer:    RNNFormer SE model, 5 variants (T/B/S/M/L), 16kHz
  - FastULCNet:      Spectrum-in/out（与 GTCRN/DeepFilterNet 一致）；需 stft 块，本地 YAML 选择 512/128/512
  - DCCRN:           原始 DCCRN complex-domain SE 基线；框架 STFT 谱进/谱出，不与 ClearerVoice FRCRN 混用
  - MTFAANet:         MTFAA-Net 风格单通道 SE（Phase Encoder + 线性 filter bank + TFCM/axial attention + 共享 MTFAA mask head）
  - ULUNAS:          Ultra-lightweight U-Net (NAS), spectrum-in/out, ERB + XConv/XDWS/XMB + DPGRNN
  - FRCRN_SE_16K:    ClearerVoice FRCRN，**框架 STFT** 谱 ``(B,F,T,2)`` 进/出（无内部 ConvSTFT 前向）；构造参数与顶层 ``stft`` 对齐（``n_fft`` / ``hop_length`` / ``win_length`` / ``window``）；见 ``clearervoice.py``
  - MossFormer2_SE_48K:   MossFormer2 风格 PSM；**框架 STFT** 谱进/出；Mel 由 STFT 功率谱 + MelScale（48 kHz 典型 1920/384）；同上通用 STFT 参数名
  - TFGridNet:        TF-GridNet (ICASSP 2023), 频域 BLSTM + 时域 BLSTM + Self-Attention Grid 结构
  - TFSkiMNet:        TF-SkiMNet (Interspeech 2025), inplace 时频建模 + 跨 block skipping memory
  - BSRoFormer:       Band-Split RoFormer (arXiv 2309.02612), Band-Split + RoPE Transformer 时频交替
  - Conformer:      Conformer (Interspeech 2020), Macaron FFN + Rel-MHSA + Conv Module SE 适配
  - MossFormer2:      MossFormer2 (ICASSP 2024), FLASH + Gated FSMN 纯频谱进出版本
  - SpatialNet_SE / OnlineSpatialNet_SE: 多通道 observation、单通道谱输出；OnlineSpatialNet 的窄带分支可配置
    ``attention='mhsa(scope)' | 'ret(factor,share_qk)' | 'mamba(d_state,d_conv)'``，使用 tensor-only cache
    （ABI v2：增量 K/V、recurrent 状态或 conv/ssm 状态）
  - MeanSE:           MeanSE 生成式 complex-STFT SE（Wang et al., arXiv:2509.21214；Mean Flow 采样，
    训练由 MeanFlowSETrainer 驱动 loss-returning）
  - PASE / StuPASE:   WavLM 基生成式增强（pase_family 包，waveform-native）；
    训练 wrapper：DeWavLMDistillModel / StuPASEFlowTrainModel / StuPASEMelVocoderGenerator
  - UniPASE:          DeWavLM-Omni(PLC) → VocosAdapter → VocosVocoder → TFGridNet PostNet
    四模块级联（pase_family/unipase，纯推理）；训练 wrapper：UniPASEVocoderGenerator /
    UniPASEAdapterGenerator / UniPASEPostNetGenerator；判别器 UniPASEMultiScaleDiscriminator
  - GAP_URGENet:      生成支路 DeWavLM-Omni(PLC) → VocosAdapter → VocosVocoder 与预测支路
    TFGridNetV3 Predictor 双路上采样 48k 后由双通道 TFGridNetV3 PostNet 融合
    （pase_family/gap_urgenet，五模块级联，纯推理）；训练 wrapper：
    GAPURGENetPredictor / GAPURGENetPostNetGenerator（判别器复用 PASEDiscriminator）

For AEC models (DeepVQE, CAGCRN), see ``tasks.aec.models``.

Note: Models use lazy-load registration (import module on first build_model call),
      avoiding duplicate registration conflicts when running model files directly.

论文与原作者（细节见各子模块 docstring；本包负责注册与 build_model 封装）:
  - GTCRN: Xiaobin Rong et al., "GTCRN: A Speech Enhancement Model Requiring Ultralow
    Computational Resources", IEEE ICASSP 2024. https://github.com/Xiaobin-Rong/gtcrn
  - ULUNAS: Xiaobin Rong et al., UL-UNAS, arXiv:2503.00340. https://github.com/Xiaobin-Rong/ul-unas
  - BSRNN: J. Yu et al., Interspeech 2023; J. Yu & Y. Luo, ICASSP 2023 (USR). 见 ``bsrnn_impl``.
  - DeepFilterNet 基线: H. Schröter et al., DeepFilterNet; DPDFNet 见 ``deepfilternet_impl``.
  - FastEnhancer: S. Ahn et al., arXiv:2509.21867. https://github.com/aask1357/fastenhancer
  - FastULCNet: Fast-ULCNet / Comfi-FastGRNN 结构与官方实现对齐；见 ``fast_ulcnet.py`` 与 ``fast_ulcnet_impl``.
  - DCCRN: Y. Hu et al., "DCCRN", Interspeech 2020；见 ``dccrn.py`` 与 ``dccrn_impl``。
  - ClearerVoice: FRCRN / MossFormerGAN / MossFormer2 见 ``clearervoice.py``（上游 ModelScope ClearerVoice-Studio）。
  - TFGridNet: Z.-Q. Wang et al., "TF-GridNet", ICASSP/TASLP 2023. 见 ``tf_gridnet.py``.
  - BS-RoFormer: J.-C. Wang et al., arXiv:2309.02612. 见 ``bs_roformer.py``.
  - Conformer: A. Gulati et al., "Conformer", Interspeech 2020. 见 ``conformer.py``.
  - MossFormer2: S. Zhao et al., "MossFormer2", ICASSP 2024. 见 ``mossformer2.py``（纯频谱版）.
  - ZipEnhancer: H. Wang & B. Tian, "ZipEnhancer", arXiv:2501.05183. 见 ``zipenhancer.py``.
  - RT_SEMamba: R. Chao et al., "RT-SEMamba", Interspeech 2026；见 ``rt_semamba.py`` 与 ``rt_semamba_impl``。
  - MPSEnetGenerator:  MP-SENet (Interspeech 2023), DenseEncoder + TSTransformer + Mask/Phase Decoder
  - MambAttentionGenerator:  MambAttention (IEEE/ACM TASLP 2026), DenseEncoder + MambAttentionBlock + Mask/Phase Decoder
  - RWSAMambaUNetGenerator:  RWSAMamba-UNet (ICASSP 2026), 3-level U-Net + RWSA + DeformConv
"""

import logging

from core.models import register_lazy_models
from core.utils.registry import MODEL_REGISTRY  # noqa: F401

logger = logging.getLogger(__name__)

# Model name -> module path mapping (lazy load, import only on first build)
_MODEL_REGISTRY_MAP: dict[str, str] = {
    # Discriminative models (single-model supervised)
    "GTCRN":          "tasks.se.models.discriminative.gtcrn",
    "BSRNN":          "tasks.se.models.discriminative.bsrnn",
    "DeepFilterNet": "tasks.se.models.discriminative.deepfilternet",
    "FastEnhancer":   "tasks.se.models.discriminative.fastenhancer",
    "FastEnhancerB":  "tasks.se.models.discriminative.fastenhancer",
    "FastULCNet":     "tasks.se.models.discriminative.fast_ulcnet",
    "DCCRN":          "tasks.se.models.discriminative.dccrn",
    "MTFAANet":       "tasks.se.models.discriminative.mtfaa_net",
    "ULUNAS":         "tasks.se.models.discriminative.ulunas",
    "FRCRN_SE_16K":   "tasks.se.models.discriminative.clearervoice",
    "MossFormer2_SE_48K":   "tasks.se.models.discriminative.clearervoice",
    "TFGridNet":      "tasks.se.models.discriminative.tf_gridnet",
    "TFSkiMNet":      "tasks.se.models.discriminative.tf_skimnet",
    "BSRoFormer":     "tasks.se.models.discriminative.bs_roformer",
    "Conformer":     "tasks.se.models.discriminative.conformer",
    "MossFormer2":    "tasks.se.models.discriminative.mossformer2",
    "ZipEnhancer":    "tasks.se.models.discriminative.zipenhancer",
    "RT_SEMamba":     "tasks.se.models.discriminative.rt_semamba",
    "RT_SEMambaDistill": "tasks.se.models.discriminative.rt_semamba",
    "SpatialNet_SE": "tasks.se.models.discriminative.spatialnet",
    "OnlineSpatialNet_SE": "tasks.se.models.discriminative.online_spatialnet",
    # Generative complex-STFT models
    "MeanFlowSE": "tasks.se.models.generative.meanflowse.model",
    "MeanSE": "tasks.se.models.generative.meanse.model",
    # GAN discriminators
    "MetricDiscriminator":   "tasks.se.models.gan.discriminators.discriminator",
    "SpectralDiscriminator": "tasks.se.models.gan.discriminators.discriminator",
    "MetricMagDiscriminator": "tasks.se.models.gan.discriminators.metric_mag_discriminator",
    # GAN generators (MetricGAN-style)
    "MPSEnetGenerator":      "tasks.se.models.gan.generators.mp_senet",
    "MambAttentionGenerator": "tasks.se.models.gan.generators.mambattention",
    "RWSAMambaUNetGenerator": "tasks.se.models.gan.generators.rwsa_mamba_unet",
    "MossFormerGANGenerator": "tasks.se.models.gan.generators.mossformer_gan",
    # Pretrained generative models (PASE)
    "PASE": "tasks.se.models.pase_family.pase",
    "PASEDiscriminator": "tasks.se.models.pase_family.pase",
    "DeWavLMDistillModel": "tasks.se.models.pase_family.common.dewavlm_distill",
    # Pretrained generative models (StuPASE)
    "StuPASE": "tasks.se.models.pase_family.stupase.stupase_model",
    "StuPASEFlowTrainModel": "tasks.se.models.pase_family.stupase.flow_train_wrapper",
    "StuPASEMelVocoderGenerator": "tasks.se.models.pase_family.stupase.mel_gan_wrapper",
    # Pretrained generative models (UniPASE)
    "UniPASE": "tasks.se.models.pase_family.unipase.unipase_model",
    "UniPASEMultiScaleDiscriminator": "tasks.se.models.pase_family.unipase.adapter_discriminators",
    "UniPASEVocoderGenerator": "tasks.se.models.pase_family.unipase.gan_wrappers",
    "UniPASEAdapterGenerator": "tasks.se.models.pase_family.unipase.gan_wrappers",
    "UniPASEPostNetGenerator": "tasks.se.models.pase_family.unipase.gan_wrappers",
    # Pretrained generative models (GAP-URGENet)
    "GAP_URGENet": "tasks.se.models.pase_family.gap_urgenet.gap_urgenet_model",
    "GAPURGENetPredictor": "tasks.se.models.pase_family.gap_urgenet.gan_wrappers",
    "GAPURGENetPostNetGenerator": "tasks.se.models.pase_family.gap_urgenet.gan_wrappers",
}


MODEL_REGISTRY.set_lazy_map(_MODEL_REGISTRY_MAP)

_OPTIONAL_MODEL_DEPENDENCIES: dict[str, frozenset[str]] = {
    "MambAttentionGenerator": frozenset({"mamba_ssm", "causal_conv1d"}),
    # RWSA-MambaUNet 的模型实现直接依赖 torchvision.ops.deform_conv。
    "RWSAMambaUNetGenerator": frozenset({"mamba_ssm", "causal_conv1d", "torchvision"}),
    "RT_SEMamba": frozenset({"mamba_ssm"}),
    "RT_SEMambaDistill": frozenset({"mamba_ssm"}),
}


def register_se_models(only: str | None = None) -> None:
    """Register SE model class(es) into the shared ``MODEL_REGISTRY``."""
    register_lazy_models(
        _MODEL_REGISTRY_MAP,
        task="se",
        only=only,
        logger=logger,
        optional_dependencies=_OPTIONAL_MODEL_DEPENDENCIES,
    )


from core.models.build import build_model  # noqa: F401,E402
