"""pase_family 共享底座：被 pase / stupase / unipase / gap_urgenet 复用的公共组件。

清单：
  - ``wavlm/``：WavLM 结构（WavLM.py + modules.py）；
  - ``vocos/``：Vocos backbone / head / conv；
  - ``wavlm_frontend.py``：``WavLM_feat``（重采样/pad/逐层提取/checkpoint 加载）；
  - ``vocoder_backend.py``：``WavLMDec``（双/单流 Vocos 声码器）；
  - ``discriminator.py``：``CombinedDiscriminator``（MPD+MBD）；
  - ``dewavlm_distill.py``：``DeWavLMDistillModel``（DeWavLM 蒸馏训练 wrapper）；
  - ``utils.py``：``align_waveform_length`` / ``load_artifact_module`` 等共享辅助函数。

本 __init__ 不做隐式 import（避免子包加载时的循环依赖与无谓开销），
使用方按需 ``from tasks.se.models.pase_family.common.<module> import ...``。
"""
