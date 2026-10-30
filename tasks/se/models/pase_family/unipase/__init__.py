"""UniPASE (Cisco, 2025) generative speech enhancement package.

Ported from https://github.com/Xiaobin-Rong/UniPASE（官方仓库）。结构：
DeWavLM-Omni (WavLM + PLC) -> VocosAdapter (L24+L1 -> enh L1)
-> VocosVocoder (16kHz 波形) -> TFGridNet PostNet (8k->24kHz 宽带扩展)。

复用本框架共享组件：common/wavlm_frontend.WavLM_feat、common/vocos/backbone.py 的
VocosBackbone、common/vocos/head.py 的 ISTFTHead；判别器直接复用
pase_family/common/discriminator.py（官方 vocoder/discriminators.py 与
pase 版本逐字节一致）。
"""
