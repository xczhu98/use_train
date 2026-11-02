"""Standard base class for single-input SE enhancement models.

标准判别式路径使用顶层 YAML ``stft``：trainer 完成 STFT/iSTFT，模型实现
``forward(noisy_spec, lengths=None, *, fs=None) -> enhanced_spec``，不得在模型内重复变换。

没有顶层 ``stft`` 的 waveform-native 路径仅适用于明确的端到端/生成式模型（例如
PASE 家族）和登记为 waveform-native 的判别式模型（``CoFiLite``）；它们实现
``forward(noisy_wav, lengths=None, *, fs=None) -> enhanced_wav``，
并由对应模型自行管理其特征前端。

Multi-SR models (e.g. BSRNN, DeepFilterNet) may accept ``fs`` in forward;
concrete subclasses may omit support when the sample rate is not needed.

Not every registered SE model inherits this class. Explicit exceptions include:
  - dual-input or loss-returning training models such as
    ``DeWavLMDistillModel`` and ``StuPASEFlowTrainModel``，它们接受
    ``(noisy, clean, lengths, *, fs)`` 并由 ``DualInputLossTrainer`` 驱动；
  - GAN generators/discriminators and stage-specific wrappers，按各自 trainer
    contract 接受双输入、返回 feature/tuple，或只在对抗训练中消费；
  - ``MossFormer2_SE_48K``（``tasks/se/models/discriminative/clearervoice.py``）
    在模型内部对 ``noisy_spec`` 做 iSTFT 以提取官方 Kaldi fbank 前端特征，
    属显式登记的前端例外；其对外 I/O 仍是频谱进/频谱出。
这些例外不得由标准 ``SETrainer`` 的单输入增强路径隐式驱动。
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseSEModel(ABC, nn.Module):
    """Base class for standard single-input SE enhancement models.

    Registered dual-input, loss-returning and GAN-stage models are explicit
    exceptions and may inherit ``nn.Module`` directly. Subclasses here implement either:
      forward(noisy_spec, lengths=None, *, fs=None) -> enhanced_spec   (when stft in config)
      forward(noisy_wav, lengths=None, *, fs=None) -> enhanced_wav
    """

    @abstractmethod
    def forward(self, noisy_wav: torch.Tensor,
                lengths: torch.Tensor = None, *, fs: int = None) -> torch.Tensor:
        """
        Args:
            noisy_wav: (B, T) waveform or (B, F, T, 2) spectrum depending on pipeline.
            lengths:   (B,)  valid lengths (optional).
            fs:        (optional) input sample rate.

        Returns:
            enhanced_wav or enhanced_spec, matching input type.
        """
        ...
