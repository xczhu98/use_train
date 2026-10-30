"""SE 单样本辅助类型与共享 SEBatch 的任务入口。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from core.data.speech_enhancement_batch import SEBatch


@dataclass
class SEBatchItem:
    """SE 单样本的可选结构化表示，不是默认 Dataset/collate 返回合同。

    默认数据路径仍返回位置 tuple，并保持 4/7 元组兼容；本类型仅供显式
    结构化调用方使用。
    """

    clean: torch.Tensor
    noisy: torch.Tensor
    sr: int
    length: int
    is_pure_noise: bool = False
    is_pure_echo: bool = False
    is_pure_echo_noise: bool = False


__all__ = ["SEBatch", "SEBatchItem"]
