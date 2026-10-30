"""PASE 模型家族（pase_family）包。

子包：
  - ``pase/``：PASE（WavLM 基生成式增强）与 PASEDiscriminator；
  - ``stupase/``：StuPASE（学生版，flow + mel vocoder）；
  - ``unipase/``：UniPASE（DeWavLM-Omni + VocosAdapter/Vocoder + PostNet）；
  - ``gap_urgenet/``：GAP-URGENet（生成-预测双支路 + PostNet 融合）；
  - ``common/``：四者共享的底座（WavLM、Vocos、前端、判别器、工具函数）。
"""
