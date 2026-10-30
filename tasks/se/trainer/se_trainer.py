"""Speech enhancement trainer SETrainer.

Implements SE (Speech Enhancement) training loop on top of SETrainerBase:
  - AMP autocast (FP16/BF16) + gradient scaling
  - non_blocking GPU transfer (async with pin_memory)
  - Gradient accumulation (accumulation_steps for larger effective batch)
    Uses model.no_sync() during intermediate accumulation steps to suppress
    redundant DDP AllReduce — only the final step triggers synchronization.
  - NaN safety (merged isfinite checks — 1 GPU sync per check, not 2)
  - Auto-detect if model needs fs parameter
  - Validation objective metrics: SI-SDR / PESQ / STOI / ESTOI

Validation metrics config (trainer.val_metrics in YAML):
  val_metrics:
    enabled: true
    metrics: [SI-SDR, PESQ, STOI]   # intrusive: SI-SDR PESQ STOI ESTOI
    interval: 5                      # compute every N validations
    max_utterances: 200              # 每 epoch 全局总数；各 rank 自动取 max_utterances // world_size
    max_utterances_metrics: null     # 用于指标计算的全局总数（null = 用 max_utterances）；同上按 rank 均分
    metrics_chunk_size: 512         # chunk size for incremental metric computation; reduces CPU memory
    strict: false                   # true: missing objective result raises error
    metrics_workers: 0               # ProcessPool workers for intrusive (0 = auto, cap by cpu_count)

DNSMOS/NISQA/UTMOS/SCOREQ 等 MOS 通过 ``evaluate/`` 离线计算。
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import math
import time
from tqdm import tqdm
from collections import deque
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from tasks.se.trainer.metrics_trainer import SETrainerBase
from tasks.se.trainer.val_audio import ValAudioWriter

logger = logging.getLogger(__name__)


class SETrainer(SETrainerBase):
    """Speech enhancement trainer.

    Args:
        train_loader:  Training DataLoader.
        val_loader:   Validation DataLoader.
        sample_rate:  Default sample rate.
        stft_module:  Optional framework STFT/iSTFT module (spectrum-in/spectrum-out models).
    """


    _val_audio = None

    def __init__(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        sample_rate: int = 16000,
        stft_module: Any = None,
        **kw: Any,
    ):
        super().__init__(
            train_loader=train_loader,
            val_loader=val_loader,
            sample_rate=sample_rate,
            stft_module=stft_module,
            **kw,
        )
        self._val_audio_ctx = None
        # 只有标准 SETrainer 自己消费 val_audio；DualInputLossTrainer 等子类有
        # 各自的验证循环与 writer，不能在父类构造期按 predictor 口径校验。
        self._val_audio = None
        if type(self) is SETrainer:
            self._val_audio = ValAudioWriter.from_config(
                self.config,
                rank=self.rank,
                device=self.device,
                sample_rate=sample_rate,
            )
            if self._val_audio is not None:
                stage = getattr(self._raw_model(), "val_audio_stage", None)
                if stage != "predictor":
                    raise ValueError(
                        f"{type(self._raw_model()).__name__} does not support "
                        f"trainer.val_audio (stage={stage!r}); SETrainer val_audio "
                        "只覆盖 GAP Phase 1 predictor"
                    )
                self._val_audio.validate_roles(
                    getattr(self._raw_model(), "val_audio_vocoder_roles", ())
                )

    def _val_prepare_batch(
        self, raw_batch: Any, epoch: int, n_loops: int
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """标准 SE 验证：保存音频所需 batch 事实留给 ``_val_accumulate_extra``。"""
        prepared = super()._val_prepare_batch(raw_batch, epoch, n_loops)
        if self._val_audio is not None and self.rank == 0:
            _se_batch, clean, noisy, lengths, fs_val = prepared
            self._val_audio_ctx = (noisy, clean, lengths, fs_val, epoch, n_loops)
        return prepared

    def _val_accumulate_extra(self, extra: Any, ctx: Any, is_ddp_pad: bool) -> None:
        """在 loss 累计后、collective 之前写 GAP Phase 1 验证音频。"""
        super()._val_accumulate_extra(extra, ctx, is_ddp_pad)
        if self._val_audio is None:
            return
        cached = self._val_audio_ctx
        self._val_audio_ctx = None
        if is_ddp_pad or cached is None:
            return
        noisy, clean, lengths, fs_val, epoch, step = cached
        self._val_audio.capture(
            self._raw_model(),
            noisy,
            clean,
            lengths,
            fs=fs_val,
            epoch=epoch,
            step=step,
            start_epoch=self.start_epoch,
        )
