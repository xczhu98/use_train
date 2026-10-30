"""Base trainer BaseTrainer — DDP / AMP / EMA / checkpoint infrastructure.

Core features:
  - DDP distributed training: auto-wrap DistributedDataParallel
  - AMP mixed precision: GradScaler + autocast, FP16 / BF16
  - EMA exponential moving average: torch_ema (optional, smooth model weights);
    DDP 下跨 rank 同步 EMA shadow 的周期由 ``meta.ema_sync_every_steps`` 配置（默认 1000，0 表示不同步）
  - Gradient accumulation: configurable accumulation_steps, simulate larger batch
  - CUDA acceleration: cudnn.benchmark + TF32 (Ampere+ GPU)
  - torch.compile: optional model compilation for faster forward/backward
  - Checkpoint management: periodic save / best save / resume / mid-batch save
  - Logging: TensorBoard + optional WandB
  - LR scheduling: per-step or per-epoch

Subclasses must implement: _train_epoch(epoch), _validation_epoch(epoch)

Design note (unified training-loop template):
  SE / AEC / TSE trainers inherit from ``core.trainer.metrics_trainer.MetricsCapableTrainer``
  (SE 侧为 ``SETrainerBase``)，它持有 train/validation epoch 骨架（gradient
  accumulation / DDP no_sync / AMP / NaN 安全 / epoch-tail 残差 / val metrics），
  各任务通过 protected hooks（_unpack_batch、_forward_and_compute_loss、
  _val_forward_loss、_val_accumulate_extra 等）注入差异；DataLoader 装配由
  ``UnifiedTrainingEntry._build_loaders_common`` 统一收敛。BaseTrainer 只保留
  DDP/AMP/EMA/checkpoint 等任务无关基础设施。
"""

from __future__ import annotations

import contextlib
import copy
import gc
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from core.trainer.optimizer_step import PreparedOptimizerStep

logger = logging.getLogger(__name__)


def _normalize_max_steps(value: object) -> int | None:
    """严格解析 optimizer step 上限，避免隐式截断配置错误。"""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            "max_steps must be None, an integer, an integer-valued float, or an integer string"
        )
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                "max_steps must be None, an integer, an integer-valued float, or an integer string"
            )
        normalized = int(value)
    elif isinstance(value, str):
        try:
            normalized = int(value)
        except ValueError as exc:
            raise ValueError(
                "max_steps must be None, an integer, an integer-valued float, or an integer string"
            ) from exc
    else:
        raise ValueError(
            "max_steps must be None, an integer, an integer-valued float, or an integer string"
        )
    if normalized < 0:
        raise ValueError("max_steps must be non-negative")
    return normalized


class BaseTrainer:
    """Base class for all trainers.

    Args:
        rank: Current process rank in DDP (0 for single GPU).
        world_size: Total DDP processes (1 for single GPU).
        model: nn.Module model.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        loss_fn: Loss function (nn.Module or callable with same signature as forward(enhanced, clean, lengths)).
        config: Full config dict.
        resume: Whether to restore the full training state; when resume_from is not
            specified, automatically discovers a checkpoint.
        resume_from: 完整状态恢复的 checkpoint 路径视图；默认也作为持久
            source。入口冻结 checkpoint 时，该值持续指向 immutable snapshot。
        resume_load_from: 仅本次完整状态恢复读取的 checkpoint 路径；提供时与
            resume_from 一样持续指向 immutable snapshot。
        resume_source_from: 持久 checkpoint source 路径，仅用于确定 checkpoint/log
            运行目录，不暴露给 resume hook 作为加载路径。
        use_wandb: Whether to enable WandB logging.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | object,
        loss_fn: torch.nn.Module | Callable[..., torch.Tensor],
        config: dict,
        *,
        local_rank: int | None = None,
        train_sampler: torch.utils.data.Sampler | None = None,
        train_dataset: torch.utils.data.Dataset | None = None,
        resume: bool = False,
        resume_from: str | None = None,
        resume_load_from: str | None = None,
        resume_source_from: str | None = None,
        use_wandb: bool = False,
    ):
        self.rank = rank
        self.local_rank = rank if local_rank is None else int(local_rank)
        if local_rank is None and rank > 0:
            logger.warning(
                "local_rank not provided (falling back to global rank=%d). "
                "Multi-node DDP training may select the wrong CUDA device. "
                "Pass --local_rank explicitly in your launch script.",
                rank,
            )
        self.world_size = world_size
        self.config = config
        self._resolved_config_provenance: dict = {}
        self.train_sampler = train_sampler
        self.train_dataset = train_dataset
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        # 信号处理器只允许置 flag；checkpoint/destroy 等副作用统一在训练安全边界执行。
        self._shutdown_requested = False
        self._shutdown_signal: int | None = None

        # ── CUDA performance settings ──
        meta = config.get("meta", {})
        if torch.cuda.is_available():
            # When deterministic=True, cudnn.benchmark must be False (they conflict).
            deterministic = meta.get("deterministic", False)
            torch.backends.cudnn.benchmark = False if deterministic else meta.get("cudnn_benchmark", True)
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = meta.get("allow_tf32", True)
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = meta.get("allow_tf32", True)

        # ── Model ──
        # Support dict of models for multi-model training (e.g. GAN: generator + discriminator)
        self._is_multi_model = isinstance(model, dict)
        if self._is_multi_model:
            self.models = {k: v.to(self.device) for k, v in model.items()}
            # Backward compat: default to generator or first model
            # 显式判 None：不依赖 nn.Module 的真值语义
            _gen = self.models.get("generator")
            self.model = _gen if _gen is not None else next(iter(self.models.values()))
        else:
            self.model = model.to(self.device)
            self.models = {"default": self.model}

        # ── Gradient checkpointing (activation recomputation) ──
        # Trade ~33% extra compute for ~40-60% VRAM reduction on activations.
        # Especially valuable for BSRNN (12 LSTM passes) and long sequences.
        self._gradient_checkpointing = meta.get("gradient_checkpointing", False)
        if self._gradient_checkpointing:
            for m in self.models.values():
                self._apply_gradient_checkpointing(m)
            if rank == 0:
                logger.info("Gradient checkpointing enabled (activation recomputation)")

        if meta.get("compile_model", False) and hasattr(torch, "compile"):
            for name, m in list(self.models.items()):
                self.models[name] = torch.compile(m, dynamic=True)
            # 显式判 None：不依赖 nn.Module 的真值语义
            self.model = next(
                (m for m in (self.models.get("generator"), self.models.get("default")) if m is not None),
                next(iter(self.models.values())),
            )
            if rank == 0:
                logger.info("torch.compile(model, dynamic=True) enabled")

        if world_size > 1:
            for name, m in list(self.models.items()):
                ddp_kwargs = {
                    "find_unused_parameters": config.get("distributed", {}).get(
                        "find_unused_parameters", False
                    ),
                }
                # CPU/Gloo DDP 不接受 CUDA 专用的 device_ids/output_device。
                if self.device.type == "cuda":
                    ddp_kwargs.update(
                        device_ids=[self.local_rank], output_device=self.local_rank
                    )
                self.models[name] = DDP(m, **ddp_kwargs)
            # 显式判 None：不依赖 nn.Module 的真值语义
            self.model = next(
                (m for m in (self.models.get("generator"), self.models.get("default")) if m is not None),
                next(iter(self.models.values())),
            )

        # ── Optimizer ──
        # Support dict of optimizers for multi-optimizer training (e.g. GAN)
        self._is_multi_optimizer = isinstance(optimizer, dict)
        if self._is_multi_optimizer:
            self.optimizers = optimizer
            self.optimizer = self.optimizers.get("generator") or next(iter(self.optimizers.values()))
        else:
            self.optimizer = optimizer
            self.optimizers = {"default": self.optimizer}

        # ── Scheduler ──
        self._is_multi_scheduler = isinstance(scheduler, dict)
        if self._is_multi_scheduler:
            self.schedulers = scheduler
            self.scheduler = self.schedulers.get("generator") or next(iter(self.schedulers.values()))
        else:
            self.scheduler = scheduler
            self.schedulers = {"default": self.scheduler}

        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, torch.nn.Module) else loss_fn

        # ── AMP mixed precision ──
        self.use_amp = meta.get("use_amp", False)
        amp_dtype = meta.get("amp_dtype", "float16")
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        self.scaler = GradScaler(
            self.device.type,
            enabled=(self.use_amp and self.amp_dtype == torch.float16),
        )

        # ── EMA ──
        self.use_ema = meta.get("use_ema", False)
        self.ema = None
        if self.use_ema:
            try:
                from torch_ema import ExponentialMovingAverage
                raw = self._raw_model()
                self.ema = ExponentialMovingAverage(
                    raw.parameters(), decay=meta.get("ema_decay", 0.999),
                )
            except ImportError:
                logger.warning("torch_ema not installed, EMA disabled")
                self.use_ema = False
        # DDP：跨 rank 广播 EMA shadow 的周期（步）；默认 1000。0=从不跨 rank 同步（仅本地 update，易漂移）
        _ema_sync = meta.get("ema_sync_every_steps", 1000)
        self.ema_sync_every_steps = (
            max(0, int(_ema_sync)) if _ema_sync is not None else 1000
        )

        # ── Training config ──
        tc = config.get("trainer", {})
        self.epochs = tc.get("epochs", 200)
        # None（含省略）表示无限制；非负值在 optimizer-step 边界检查，0 表示训练前停止。
        self.max_steps = _normalize_max_steps(tc.get("max_steps"))
        self.clip_grad = tc.get("gradient_clip_value", 5.0)
        self.save_interval = tc.get("save_checkpoint_interval", 5)
        # validation_epoch_interval: 每 N 个 epoch 做一次验证（0=关闭 epoch 末验证）
        # validation_steps_interval: 每 N 个 optimizer step 做一次验证（0=关闭步级验证）；与 save_top_k 配合 step-best
        self.validation_epoch_interval = int(tc.get("validation_epoch_interval", 1) or 0)
        if self.validation_epoch_interval < 0:
            self.validation_epoch_interval = 0
        if self.validation_epoch_interval == 0 and self.rank == 0:
            logger.info(
                "trainer.validation_epoch_interval=0: skipping epoch-end validation; "
                "set validation_steps_interval>0 for step-based validation."
            )
        _vsi = tc.get("validation_steps_interval", 0)
        self.validation_steps_interval = max(0, int(_vsi) if _vsi is not None else 0)
        self.save_top_k = max(0, int(tc.get("save_top_k", 3)))  # keep top-k step-best checkpoints
        # Step validation (validation_steps_interval + save_top_k) vs epoch-end (validation_epoch_interval): mutually exclusive
        if (
            self.validation_steps_interval > 0
            and self.save_top_k > 0
            and self.validation_epoch_interval > 0
        ):
            if self.rank == 0:
                logger.warning(
                    "Both validation_epoch_interval>0 and step validation (validation_steps_interval>0, save_top_k>0) are set; "
                    "disabling epoch-end validation. Set validation_epoch_interval=0 for step-only, or validation_steps_interval=0 "
                    "for epoch-only validation."
                )
            self.validation_epoch_interval = 0
        # 步级验证由 save_top_k>0 激活；以下组合实际没有任何验证运行
        # （step/epoch 两侧都关闭），plateau scheduler 永不 step，提前告警。
        if (
            self.validation_steps_interval > 0
            and self.save_top_k == 0
            and self.validation_epoch_interval == 0
        ):
            if self.rank == 0:
                logger.warning(
                    "validation_steps_interval>0 requires save_top_k>0 to run step validation, "
                    "and validation_epoch_interval=0 disables epoch-end validation — no validation "
                    "will run and ReduceLROnPlateau will never be stepped. "
                    "Set save_top_k>0 or validation_epoch_interval>0."
                )
        # True when only step-based validation runs: loss/metrics curves use optimizer step as x-axis
        self._curve_x_is_step = (
            self.validation_epoch_interval == 0 and self.validation_steps_interval > 0
        )
        _mvb = tc.get("max_val_batches", 0)
        self.max_val_batches = max(0, int(_mvb) if _mvb is not None else 0)  # 0 = no limit; cap validation length
        # 官方 PASE 家族在训练前跑一次 _validation_epoch(start_epoch-1)，用于
        # epoch-0 val loss 与验证音频（epoch000）。默认只在启用 trainer.val_audio
        # 且 fresh start 时开启；显式 initial_validation 可覆盖。
        raw_val_audio = tc.get("val_audio") or {}
        self.initial_validation = bool(
            tc.get(
                "initial_validation",
                raw_val_audio.get("enabled", False)
                if isinstance(raw_val_audio, dict)
                else False,
            )
        )
        self.log_interval = tc.get("log_interval", 100)
        self.log_step_sub_losses = tc.get("log_step_sub_losses", 0)
        self.save_batch_interval = tc.get("save_checkpoint_batch_interval", 0)
        self.save_curve_interval = max(0, int(tc.get("save_curve_interval", 1)))  # 0=never, 1=every validation (default)
        _figsize = tc.get("curve_figsize", [8, 4.5])
        self.curve_figsize = (_figsize[0], _figsize[1]) if len(_figsize) >= 2 else (8, 4.5)
        self.curve_dpi = max(72, int(tc.get("curve_dpi", 150)))
        self.lr_per_step = tc.get("lr_schedule_per_step", False)
        self.accumulation_steps = tc.get("accumulation_steps", 1)
        self.patience = tc.get("early_stop_patience", 0)
        self._patience_counter = 0
        # 每 epoch / 验证前是否主动清 CUDA cache（默认 False）。
        # 审计 A17：无条件 synchronize+empty_cache+gc 会强制 host 等在途 kernel，形成 GPU 空泡；
        # 仅在 OOM 恢复或本开关显式开启（确有显存碎片问题的环境）时清理。
        self.clear_cache_each_epoch = bool(tc.get("clear_cache_each_epoch", False))
        self._placeholder_num_samples = config.get("data", {}).get("placeholder_num_samples", 1024)

        self.n_steps = 0
        # B7：update_scaler=False 的 _optimizer_step 调用（如 GAN 先 D 后 G 共用 scaler）
        # 延迟 step 的 per-step scheduler key 列表；由 update_scaler=True 的调用在判定
        # step_skipped 后统一补 step 或丢弃，保证同一边界内 D/G scheduler 对齐。
        self._pending_sched_step_keys: list[str] = []
        self.start_epoch = 1
        self.current_epoch = 1  # updated each epoch; safe to read from signal handlers
        # Sliding window of last log_interval steps for step-level loss averaging
        _li = max(1, int(self.log_interval))
        self._train_log_loss_win: deque[float] = deque(maxlen=_li)
        # best_score = best validation loss seen so far (lower is better; first val uses inf sentinel)
        self.best_score = np.inf
        self.best_epoch: int = -1  # epoch at which best_model was saved (for curve marker)
        self.best_at_step: int | None = None  # step-only val: x for best marker; epoch-mode uses None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.curve_epochs: list[int] = []  # epoch index for each curve point
        # Step-based best: list of (step, val_loss, filename) sorted by val_loss ascending (best first)
        self._step_best_list: list[tuple[int, float, str]] = []

        # ── Checkpoint & log dirs ──
        exp_path = Path(tc.get("exp_path", "exp/default"))
        resume_source = str(resume_source_from or resume_from or "") or None
        resume_load = resume_load_from or resume_from
        self._resume_from = str(resume_load) if resume_load else None
        self._resume_load_from = str(resume_load_from) if resume_load_from else None
        # resume 到 middle checkpoint 时，本 epoch 内需跳过的已完成 batch 数
        # （由 _resume_checkpoint 从 checkpoint 的 batch cursor 恢复；训练循环消费）
        self._resume_skip_batches = 0
        checkpoint_dir_claimed = False
        ckpt_base = exp_path / "checkpoints"
        if resume:
            if resume_source:
                # 持久 source 只在构造期决定运行目录，不保存到 Trainer 实例。
                self.ckpt_dir = Path(resume_source).resolve().parent
            else:
                # 仅 rank 0 选择最近的可恢复目录；所有 rank 无条件执行一次相同广播，
                # 避免按本地文件系统状态决定 collective 序列。
                existing_run_dir = None
                if rank == 0 and ckpt_base.exists():
                    candidates = []
                    for d in sorted(ckpt_base.iterdir()):
                        if not d.is_dir():
                            continue
                        if (d / "latest_model.tar").exists() or self._find_latest_middle_checkpoint(d) is not None:
                            candidates.append(d)
                    if candidates:
                        existing_run_dir = candidates[-1]
                self.ckpt_dir, checkpoint_dir_claimed = (
                    self._select_or_allocate_checkpoint_run_dir(
                        ckpt_base,
                        rank=rank,
                        world_size=self.world_size,
                        existing_run_dir=existing_run_dir,
                    )
                )
                if checkpoint_dir_claimed and rank == 0:
                    logger.warning("Resume requested but no checkpoint found, starting fresh")
        else:
            # rank 0 依次用时间戳、PID、递增后缀原子抢占；所有候选目录均
            # exist_ok=False，DDP 再广播唯一目录名，禁止复用同秒残留目录。
            self.ckpt_dir, checkpoint_dir_claimed = (
                self._select_or_allocate_checkpoint_run_dir(
                    ckpt_base,
                    rank=rank,
                    world_size=self.world_size,
                )
            )

        self.log_dir = exp_path / "logs" / self.ckpt_dir.name
        if rank == 0:
            if not checkpoint_dir_claimed:
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Auto-add file handler so all console logs are persisted under log_dir/train.log
        if rank == 0:
            log_file = self.log_dir / "train.log"
            # Avoid duplicate handlers on resume
            root = logging.getLogger()
            _log_file_abs = str(log_file.resolve())
            for h in root.handlers:
                if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == _log_file_abs:
                    break
            else:
                fh = logging.FileHandler(log_file, mode="a")
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
                root.addHandler(fh)

        self.writer = None
        self.wandb_run = None
        if rank == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                logger.warning("TensorBoard unavailable, scalar logging disabled")
            else:
                self.writer = SummaryWriter(str(self.log_dir))
            # Save experiment context for reproducibility
            self._save_experiment_context()
            if use_wandb:
                try:
                    import wandb
                    import hashlib
                    if resume:
                        wandb_id = hashlib.md5(str(exp_path).encode()).hexdigest()[:12]
                    else:
                        wandb_id = hashlib.md5(
                            f"{exp_path}:{self.ckpt_dir.name}".encode()
                        ).hexdigest()[:12]
                    self.wandb_run = wandb.init(
                        project=config.get("wandb_project", "SE_Train"),
                        name=exp_path.name,
                        id=wandb_id,
                        config=config, resume="allow",
                    )
                except ImportError:
                    pass

        if resume:
            self._resume_checkpoint()

    @staticmethod
    def _claim_checkpoint_run_dir(checkpoint_base: Path) -> Path:
        """原子抢占有界唯一的 checkpoint run 目录。"""
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        pid = os.getpid()
        max_candidates = 1000
        for index in range(max_candidates):
            if index == 0:
                name = timestamp
            elif index == 1:
                name = f"{timestamp}_p{pid}"
            else:
                name = f"{timestamp}_p{pid}_{index - 1}"
            candidate = checkpoint_base / name
            try:
                candidate.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return candidate
        raise RuntimeError(
            f"Unable to allocate checkpoint run directory after {max_candidates} "
            f"attempts under {checkpoint_base}"
        )

    @classmethod
    def _select_or_allocate_checkpoint_run_dir(
        cls,
        checkpoint_base: Path,
        *,
        rank: int,
        world_size: int,
        existing_run_dir: Path | None = None,
    ) -> tuple[Path, bool]:
        """由 rank 0 选择或抢占 run 目录，并向所有 DDP rank 广播结果。"""
        distributed = world_size > 1
        if distributed and not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "DDP checkpoint run directory allocation requires an initialized "
                "process group"
            )

        outcome: list[tuple[str, str, bool] | None] = [None]
        if rank == 0:
            try:
                claimed = existing_run_dir is None
                selected = (
                    cls._claim_checkpoint_run_dir(checkpoint_base)
                    if claimed
                    else existing_run_dir
                )
                outcome[0] = ("ok", str(selected), claimed)
            except Exception as exc:
                outcome[0] = ("error", f"{type(exc).__name__}: {exc}", False)

        if distributed:
            dist.broadcast_object_list(outcome, src=0)

        if outcome[0] is None:
            raise RuntimeError("Checkpoint run directory allocation returned no result")
        status, payload, claimed = outcome[0]
        if status != "ok":
            raise RuntimeError(f"Checkpoint run directory allocation failed: {payload}")
        return Path(payload), claimed

    # ── Utility methods ──

    @staticmethod
    def _apply_gradient_checkpointing(model: torch.nn.Module) -> None:
        """Enable gradient checkpointing on eligible submodules.

        Wraps GRU and LSTM layers so that intermediate activations are
        recomputed during backward instead of being stored.  Reduces peak
        VRAM by ~40-60% for BSRNN / FastEnhancerB at the cost of ~33% extra
        forward compute.
        """
        from torch.utils.checkpoint import checkpoint

        _ckpt_types = (torch.nn.GRU, torch.nn.LSTM)

        def _wrap_forward(module: torch.nn.Module):
            orig_fwd = module.forward

            def _checkpointed_forward(*args, **kwargs):
                return checkpoint(orig_fwd, *args, use_reentrant=False, **kwargs)

            module.forward = _checkpointed_forward  # type: ignore[assignment]

        count = 0
        for _name, child in model.named_modules():
            if isinstance(child, _ckpt_types):
                _wrap_forward(child)
                count += 1

        if count > 0:
            logger.info(
                f"  Gradient checkpointing applied to {count} GRU/LSTM modules"
            )
        else:
            logger.warning(
                "  Gradient checkpointing: no GRU/LSTM modules found. "
                "Model may not benefit from this option."
            )

    def request_shutdown(self, signum: int) -> None:
        """记录信号停机请求；可被 signal handler 安全调用，只写本地 flag。"""
        self._shutdown_signal = int(signum)
        self._shutdown_requested = True

    def shutdown_requested(self) -> bool:
        """训练主流程在安全边界读取该 flag，决定是否进入优雅退出。"""
        return bool(getattr(self, "_shutdown_requested", False))

    def _sync_shutdown_signum(self) -> int:
        """固定序列 collective：同步各 rank 的信号停机请求（MAX over signum）。

        所有 rank 必须在相同边界调用本方法：训练侧仅在 optimizer/epoch-tail
        boundary，验证侧在每次 forward 前与 terminal boundary，metrics 侧在固定
        chunk boundary，另含 epoch 结算边界。这样既避免每个 microbatch 阻塞
        all-reduce，也保证只有一个 rank 收到信号时其余 rank 一致退出。返回 0
        表示无停机请求，否则为
        信号编号（多 rank 不同信号时取最大，仅为确定性）。
        """
        requested = bool(getattr(self, "_shutdown_requested", False))
        local = int(getattr(self, "_shutdown_signal", None) or 0) if requested else 0
        if self.world_size <= 1 or not (
            dist.is_available() and dist.is_initialized()
        ):
            return local
        t = torch.tensor([local], dtype=torch.int64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        synced = int(t.item())
        if synced:
            # 让本 rank 后续本地查询（shutdown_requested）与同步结果一致。
            self._shutdown_signal = synced
            self._shutdown_requested = True
        return synced

    def _raw_model(self, model_key: str | None = None) -> torch.nn.Module:
        """Get underlying model (return .module when DDP-wrapped, else direct).

        Args:
            model_key: For multi-model training, which model to unwrap.
                      None uses the default model (generator if available).
        """
        if model_key and self._is_multi_model:
            m = self.models.get(model_key)
            if m is None:
                m = self.model
        else:
            m = self.model
        while True:
            if isinstance(m, DDP):
                m = m.module
            elif hasattr(m, "_orig_mod"):
                m = m._orig_mod
            else:
                break
        return m

    def _no_sync_all(self):
        """Context manager that applies no_sync to all DDP-wrapped models."""
        from contextlib import ExitStack

        stack = ExitStack()
        for m in self.models.values():
            if hasattr(m, "no_sync"):
                stack.enter_context(m.no_sync())
        return stack

    def _ema_context(self):
        """Return EMA context manager (switch to EMA weights when available, else no-op)."""
        if self.ema:
            return self.ema.average_parameters()
        return contextlib.nullcontext()

    def _nccl_flow_log(self, phase: str) -> None:
        """When USE_TRAIN_NCCL_DEBUG=1, log before each collective to pinpoint timeout flow."""
        if os.environ.get("USE_TRAIN_NCCL_DEBUG"):
            logger.info("[NCCL_FLOW] rank %s -> %s", self.rank, phase)

    def _allreduce_sum(self, value: float, phase: str = "allreduce_sum") -> float:
        """All-reduce SUM across ranks; single-GPU returns value unchanged.

        For global means (e.g. val metrics), allreduce sum and count separately:
        ``mean = _allreduce_sum(sum_local) / max(_allreduce_sum(count_local), 1)``.
        Do not divide this return value by world_size — that pattern is easy to misuse.
        """
        if self.world_size <= 1:
            return float(value)
        self._nccl_flow_log(phase)
        t = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())

    def _allreduce_bool_ands(
        self,
        values: tuple[bool, ...],
        phase: str = "allreduce_bool_ands",
    ) -> tuple[bool, ...]:
        """一次 collective 对多个 bool 分别取跨 rank MIN。"""
        if (
            self.world_size <= 1
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            return tuple(bool(value) for value in values)
        self._nccl_flow_log(phase)
        flags = torch.tensor(
            [1 if value else 0 for value in values],
            device=self.device,
            dtype=torch.int32,
        )
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        return tuple(bool(value) for value in flags.tolist())

    def _allreduce_bool_and(self, value: bool, phase: str = "allreduce_bool_and") -> bool:
        """跨 rank 取 bool MIN；未初始化分布式时保留本地值。"""
        return self._allreduce_bool_ands((value,), phase=phase)[0]

    def _sync_max_steps_reached(self) -> bool:
        """在固定训练边界同步 max_steps 停止判定。"""
        limit = getattr(self, "max_steps", None)
        if limit is None:
            return False
        reached = self.n_steps >= limit
        all_reached, all_not_reached = self._allreduce_bool_ands(
            (reached, not reached), phase="train_max_steps_boundary"
        )
        if not all_reached and not all_not_reached:
            raise RuntimeError(
                "max_steps boundary differs across ranks; n_steps is not synchronized"
            )
        return all_reached

    def _has_gradients(self, model_key: str | None = None) -> bool:
        """检查目标模型是否至少有一个可训练参数梯度。"""
        if model_key and self._is_multi_model:
            target_model = self.models.get(model_key, self.model)
        else:
            target_model = self.model
        return any(
            p.requires_grad and p.grad is not None for p in target_model.parameters()
        )

    def _gradients_are_finite(self, model_key: str | None = None) -> bool:
        """用一个聚合标量检查目标模型梯度，最多触发一次 host sync。"""
        if model_key and self._is_multi_model:
            target_model = self.models.get(model_key, self.model)
        else:
            target_model = self.model
        gradients = [
            parameter.grad
            for parameter in target_model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            return False
        finite = torch.ones((), dtype=torch.bool, device=gradients[0].device)
        for gradient in gradients:
            finite.logical_and_(torch.isfinite(gradient).all())
        return bool(finite.item())

    def _sanitize_gradients(self, model_key: str | None = None) -> None:
        """原地清零 NaN/Inf 梯度，不分配参数大小的替换 tensor。"""
        if model_key and self._is_multi_model:
            target_model = self.models.get(model_key, self.model)
        else:
            target_model = self.model
        for parameter in target_model.parameters():
            if parameter.grad is not None:
                torch.nan_to_num_(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Checkpoint management ──
    #
    # 保存路径风险简述（为何 checkpoint 保存可能导致 exitcode -6 / 挂起）：
    # 1) 调用链：epoch 结束 / step-best(validation_steps_interval) / 中间 checkpoint → _save_checkpoint 或
    #    _save_step_best / _save_middle_checkpoint → _build_state → _atomic_save。

    # 2) _build_state 中 raw.state_dict()、optimizer.state_dict()、_get_rng_state() 等若在 CUDA 异常
    #    状态下执行（如之前有 OOM/异步错误），.cpu() 或序列化时的 CUDA 同步可能触发 abort。
    # 3) torch.save(state, tmp) 时若 state 含 CUDA 张量，会在序列化过程中拷贝到 CPU，易在坏状态下崩。
    # 4) 所有 rank 必须在 _save_* 后的 barrier 会合；若 rank 0 在保存过程中崩溃，其他 rank 会一直等
    #    barrier，launcher 随后发 SIGTERM → 表现为“先 exitcode -6，再收到 SIGTERM”。
    # 对策：保存前将 model/optimizer/scheduler/scaler/ema 等 state 统一转到 CPU；_get_rng_state 与
    # _atomic_save 用 try/except 包裹，避免未捕获异常导致进程 abort。
    # -------------------------------------------------------------------------

    @staticmethod
    def _tensors_to_cpu(obj):
        """Recursively move all tensors in state dicts to CPU for safe torch.save (avoids CUDA sync during pickle)."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        if isinstance(obj, dict):
            return {k: BaseTrainer._tensors_to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(BaseTrainer._tensors_to_cpu(x) for x in obj)
        return obj

    @staticmethod
    def _fsync_checkpoint_path(path: Path) -> None:
        """fsync 单个文件，确保 replace 发布前内容已落到稳定存储。"""
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _fsync_checkpoint_dir(directory: Path) -> None:
        """fsync 目录项，确保 replace 后的目录状态在掉电后可恢复。"""
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _stage_checkpoint(state: dict, path: Path, generation: str) -> Path:
        """把 checkpoint 序列化到目标目录内的唯一临时文件，并 fsync 文件内容。"""
        import tempfile

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.checkpoint-{generation}-",
            suffix=".tmp",
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            torch.save(state, tmp)
            BaseTrainer._fsync_checkpoint_path(tmp)
            return tmp
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    @staticmethod
    def _link_checkpoint_stage(master: Path, path: Path, generation: str) -> Path:
        """为已序列化的 checkpoint 建同目录硬链接，避免重复序列化整份状态。"""
        import shutil
        import uuid

        link = path.parent / (
            f".{path.name}.checkpoint-{generation}-{uuid.uuid4().hex}.tmp"
        )
        try:
            os.link(str(master), str(link))
        except OSError:
            # 文件系统不支持硬链接时退回复制，保持单次序列化语义。
            link.unlink(missing_ok=True)
            shutil.copy2(master, link)
        return link

    @staticmethod
    def _replace_checkpoint_stage(tmp: Path, path: Path, generation: str) -> None:
        """发布邻接临时文件；EXDEV 时仍经目标目录第二临时文件 replace。"""
        import shutil
        import tempfile

        try:
            os.replace(str(tmp), str(path))
        except OSError as exc:
            if exc.errno not in (17, 18):  # EEXIST / EXDEV
                raise

            fd, fallback_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.checkpoint-{generation}-fallback-",
                suffix=".tmp",
            )
            os.close(fd)
            fallback = Path(fallback_name)
            try:
                shutil.copy2(tmp, fallback)
                BaseTrainer._fsync_checkpoint_path(fallback)
                os.replace(str(fallback), str(path))
            finally:
                fallback.unlink(missing_ok=True)
                tmp.unlink(missing_ok=True)
        BaseTrainer._fsync_checkpoint_dir(path.parent)

    @staticmethod
    def _atomic_save(state: dict, path: Path):
        """单文件原子保存；跨设备回退也不直接覆盖最终路径。"""
        import uuid

        generation = uuid.uuid4().hex
        tmp = BaseTrainer._stage_checkpoint(state, path, generation)
        try:
            BaseTrainer._replace_checkpoint_stage(tmp, path, generation)
        except Exception as exc:
            logger.error("checkpoint publish failed %s -> %s: %s", tmp, path, exc)
            tmp.unlink(missing_ok=True)
            raise

    @staticmethod
    def _backup_checkpoint(path: Path, backup: Path) -> None:
        """以硬链接复用旧 alias 数据块；文件系统不支持硬链接时退回复制。"""
        import shutil

        backup.unlink(missing_ok=True)
        try:
            os.link(str(path), str(backup))
        except OSError:
            backup.unlink(missing_ok=True)
            shutil.copy2(path, backup)

    @staticmethod
    def _atomic_save_aliases(
        state: dict,
        paths: list[Path],
        *,
        generation: str | None = None,
    ) -> None:
        """以同一 generation 事务式发布一组 checkpoint aliases。

        所有 alias 先完成序列化；发布任一路径失败时恢复全部旧 alias，避免
        ``latest``/epoch/``best`` 指向不同训练状态。文件系统不提供多文件原子
        rename，因此以邻接备份实现失败回滚；``generation`` 只用于命名临时与
        备份文件（发布失败会整体回滚，alias 集合无需读者侧核对字段）。
        """
        import uuid

        unique_paths = list(dict.fromkeys(Path(path) for path in paths))
        if not unique_paths:
            return
        generation = generation or uuid.uuid4().hex
        staged: dict[Path, Path] = {}
        backups: dict[Path, Path] = {}
        existed: dict[Path, bool] = {}
        retain_recovery_backups = False
        try:
            # 整组只序列化一次，其余 alias 用硬链接复用同一份数据，避免 N 倍写盘。
            master = BaseTrainer._stage_checkpoint(state, unique_paths[0], generation)
            staged[unique_paths[0]] = master
            for path in unique_paths[1:]:
                staged[path] = BaseTrainer._link_checkpoint_stage(
                    master, path, generation
                )
            for path in unique_paths:
                existed[path] = path.exists()
                if existed[path]:
                    backup = path.parent / (
                        f".{path.name}.checkpoint-{generation}-backup.tmp"
                    )
                    try:
                        BaseTrainer._backup_checkpoint(path, backup)
                    except Exception:
                        backup.unlink(missing_ok=True)
                        raise
                    backups[path] = backup
            for path in unique_paths:
                BaseTrainer._replace_checkpoint_stage(
                    staged[path], path, generation
                )
        except Exception as publish_error:
            rollback_errors: list[tuple[Path, Exception]] = []
            for path in reversed(unique_paths):
                backup = backups.get(path)
                try:
                    if backup is not None and backup.exists():
                        os.replace(str(backup), str(path))
                    elif path in existed and not existed[path]:
                        path.unlink(missing_ok=True)
                except Exception as rollback_error:
                    rollback_errors.append((path, rollback_error))
                    logger.exception("checkpoint alias rollback failed for %s", path)
            if rollback_errors:
                # 回滚失败时 backup 是旧 checkpoint 的唯一可靠副本，必须保留给人工恢复。
                retain_recovery_backups = True
                details = "; ".join(
                    f"{path}: {type(error).__name__}: {error}"
                    for path, error in rollback_errors
                )
                raise RuntimeError(
                    "checkpoint alias publish failed and rollback incomplete; "
                    f"recovery backups retained ({details})"
                ) from publish_error
            raise
        finally:
            for tmp in staged.values():
                tmp.unlink(missing_ok=True)
            if not retain_recovery_backups:
                for backup in backups.values():
                    backup.unlink(missing_ok=True)

    def _build_state(
        self,
        epoch: int,
        *,
        rng_states_by_rank: list[dict] | None = None,
        **extra,
    ) -> dict:
        """Build checkpoint state dict with full reproducibility info.

        Always save **raw** training weights under ``model`` so optimizer/scheduler resume stays
        consistent. EMA shadow weights are stored separately under ``ema``; inference loads
        raw + ``ema`` and applies ``copy_to`` (see ``inference._load_model``).

        The full resolved training config is saved under both ``config`` (legacy key) and
        ``resolved_config`` (explicit alias, same object) so every checkpoint is
        self-contained and traceable.

        For multi-model training, saves ``model_{name}``, ``optimizer_{name}``, ``scheduler_{name}``.
        Normal distributed saves provide ``rng_states_by_rank`` so every rank resumes its own
        RNG stream; ``rng_state`` remains as a rank-0-compatible legacy alias.
        """
        legacy_rng_state = (
            rng_states_by_rank[0]
            if rng_states_by_rank is not None
            else self._get_rng_state()
        )
        state = {
            "epoch": epoch,
            "n_steps": self.n_steps,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_at_step": getattr(self, "best_at_step", None),
            "patience_counter": self._patience_counter,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "curve_epochs": self.curve_epochs,
            # Reproducibility info (may skip rng/git on failure to avoid abort during save)
            "rng_state": legacy_rng_state,
            "config": self.config,  # Full resolved config
            # resolved_config：与 "config" 同一对象（pickle 共享引用，不占额外空间），
            # 显式命名以便下游工具稳定地取到训练入口解析后的完整配置。
            "resolved_config": self.config,
            # 每个 checkpoint 精确绑定不可变 sidecar；resume 配置漂移不再依赖文件名猜测。
            "resolved_config_provenance": dict(self._resolved_config_provenance),
            "git_info": self._get_git_info_safe(),
        }

        # Model state(s)
        if self._is_multi_model:
            for name, m in self.models.items():
                raw = self._raw_model(name)
                state[f"model_{name}"] = raw.state_dict()
        else:
            raw = self._raw_model()
            state["model"] = raw.state_dict()

        # Optimizer state(s)
        if self._is_multi_optimizer:
            for name, opt in self.optimizers.items():
                state[f"optimizer_{name}"] = opt.state_dict()
        else:
            state["optimizer"] = self.optimizer.state_dict()

        # Scheduler state(s)
        if self._is_multi_scheduler:
            for name, sch in self.schedulers.items():
                state[f"scheduler_{name}"] = sch.state_dict()
        else:
            state["scheduler"] = self.scheduler.state_dict()

        # Sampler epoch (for deterministic resumption)
        if self.train_sampler is not None and hasattr(self.train_sampler, "state_dict"):
            state["sampler_state"] = self.train_sampler.state_dict()
        if self.use_amp and self.amp_dtype == torch.float16:
            state["scaler"] = self.scaler.state_dict()
        if self.ema:
            state["ema"] = self.ema.state_dict()
        if hasattr(self, "val_metrics_epochs"):
            state["val_metrics_epochs"] = getattr(self, "val_metrics_epochs", [])
        if hasattr(self, "val_metrics_history"):
            state["val_metrics_history"] = getattr(self, "val_metrics_history", {})
        if rng_states_by_rank is not None:
            state["rng_states_by_rank"] = rng_states_by_rank
            state["rng_world_size"] = len(rng_states_by_rank)
        state.update(extra)
        # Move all tensor-containing state to CPU so torch.save does not trigger CUDA sync during pickle
        for key in list(state.keys()):
            if state[key] is not None and (
                key in ("model", "optimizer", "scheduler", "scaler", "ema", "sampler_state")
                or key.startswith(("model_", "optimizer_", "scheduler_"))
            ):
                state[key] = self._tensors_to_cpu(state[key])
        return state

    def _sync_checkpoint_save_result(
        self,
        save_error: str | None,
        *,
        skip_barrier: bool = False,
        barrier_phase: str,
    ) -> None:
        """同步 checkpoint 保存结果，失败时所有 rank 在 barrier 前退出。"""
        dist_ready = (
            self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
            and not skip_barrier
        )
        if not dist_ready:
            if save_error is not None:
                raise RuntimeError(save_error)
            return

        self._nccl_flow_log("checkpoint_save_status")
        status = torch.tensor(
            [1 if save_error is None else 0],
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        if status.item() == 0:
            errors: list[str | None] = [None] * self.world_size
            dist.all_gather_object(errors, save_error)
            details = "; ".join(
                f"rank {rank}: {error}"
                for rank, error in enumerate(errors)
                if error
            ) or "an unspecified rank reported failure"
            raise RuntimeError(f"Checkpoint save failed: {details}")

        self._nccl_flow_log(barrier_phase)
        dist.barrier()

    def _save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        skip_barrier: bool = False,
        *,
        batch_cursor: int | None = None,
    ) -> None:
        """Save checkpoint (rank 0 writes, all ranks sync after success).

        Normal DDP saves first synchronize the rank-0 save result.  A failure is
        propagated to every rank before the post-save barrier, preventing peers
        from waiting forever when rank 0 cannot build or write the checkpoint.
        ``batch_cursor`` 用于 max_steps 或信号停机截断的未完成 epoch 精确续跑。
        ``skip_barrier=True`` keeps the emergency-save path free of collectives.
        """
        rng_states_by_rank = (
            None if skip_barrier else self._gather_rng_states_for_checkpoint()
        )
        save_error: str | None = None
        if self.rank == 0:
            try:
                extra = (
                    {
                        "batch_cursor": int(batch_cursor),
                        "epoch_incomplete": True,
                    }
                    if batch_cursor is not None
                    else {}
                )
                state = self._build_state(
                    epoch,
                    rng_states_by_rank=rng_states_by_rank,
                    **extra,
                )
                aliases = [
                    self.ckpt_dir / "latest_model.tar",
                    self.ckpt_dir / f"model_{epoch:04d}.pth",
                ]
                if is_best:
                    aliases.append(self.ckpt_dir / "best_model.tar")
                self._atomic_save_aliases(state, aliases)
                logger.info(f"Epoch checkpoint saved: model_{epoch:04d}.pth")
                if is_best:
                    logger.info(f"Best model saved at epoch {epoch}")
            except Exception as exc:
                save_error = f"rank 0 checkpoint save failed: {type(exc).__name__}: {exc}"
                logger.error(save_error)

        # 所有 rank 都执行相同的结果同步；失败时不再进入 barrier。
        self._sync_checkpoint_save_result(
            save_error,
            skip_barrier=skip_barrier,
            barrier_phase="barrier_after_checkpoint_save",
        )

    def _maybe_validate_and_save_step_best(
        self, epoch: int, step: int, batch_cursor: int | None = None
    ) -> None:
        """Validate and save step-best with the completed epoch batch cursor."""
        if self.validation_steps_interval <= 0 or self.save_top_k <= 0 or step <= 0:
            return
        if step % self.validation_steps_interval != 0:
            return
        # Cross-rank sync: skip if any rank has no validation batches
        if self.world_size > 1:
            can_validate = torch.tensor(
                [1 if len(self.val_loader) > 0 else 0],
                dtype=torch.int32, device=self.device,
            )
            dist.all_reduce(can_validate, op=dist.ReduceOp.MIN)
            if can_validate.item() == 0:
                return
        for m in self.models.values():
            m.eval()
        try:
            val_loss = self._validation_epoch(epoch)
        finally:
            for m in self.models.values():
                m.train()
        if not self._is_valid_validation_result(val_loss):
            if self.rank == 0:
                logger.warning(
                    "Skipping step validation result at step %s because it is not a "
                    "finite real scalar: %r",
                    step,
                    val_loss,
                )
            return
        val_loss = float(val_loss)
        # 发布资格使用未舍入的分数，不依赖候选文件是否仍在目录中。
        is_new_best = val_loss < self.best_score
        if is_new_best:
            self.best_score = val_loss
            self.best_epoch = epoch
            self.best_at_step = step
        # B6 修复：步级验证模式（validation_epoch_interval=0 + validation_steps_interval>0）
        # 下 epoch 末 did_validate 恒 False，ReduceLROnPlateau 永不 step；在此补 step(val_loss)。
        # epoch 验证与步级验证互斥（见 __init__ 中 validation_epoch_interval 清零逻辑），
        # 因此不会与 epoch 末的 plateau step 重复。
        if not self.lr_per_step:
            if self._is_multi_scheduler:
                for sch in self.schedulers.values():
                    if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        sch.step(val_loss)
            elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
        if self.rank == 0:
            self._log_scalar("val/step_loss", val_loss, step)
            logger.info(f"  [Val @ step {step}] val_loss {val_loss:.4f}")
        # Step-only mode: val_loss vs optimizer step; train column is NaN until epoch ends
        if getattr(self, "_curve_x_is_step", False):
            self.train_losses.append(float("nan"))
            self.val_losses.append(val_loss)
            self.curve_epochs.append(step)
            self._on_loss_curve_point_added(step, float("nan"), val_loss)
            if self.rank == 0 and self._should_save_curve(len(self.curve_epochs)):
                self._save_loss_curve()
        rng_states_by_rank = self._gather_rng_states_for_checkpoint()
        save_error: str | None = None
        if self.rank == 0:
            try:
                self._save_step_best(
                    epoch,
                    step,
                    val_loss,
                    is_new_best=is_new_best,
                    batch_cursor=batch_cursor,
                    rng_states_by_rank=rng_states_by_rank,
                )
            except Exception as exc:
                save_error = (
                    f"rank 0 step-best checkpoint save failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.error(save_error)
        self._sync_checkpoint_save_result(
            save_error,
            barrier_phase="barrier_after_step_best_save",
        )

    def _save_step_best(
        self,
        epoch: int,
        step: int,
        val_loss: float,
        *,
        is_new_best: bool = False,
        batch_cursor: int | None = None,
        rng_states_by_rank: list[dict] | None = None,
    ) -> None:
        """保存候选 checkpoint；仅显式严格改进时原子发布 best alias。"""
        if self.rank != 0 or self.save_top_k <= 0:
            return
        import re
        # Populate _step_best_list from existing best_step=*.tar on first use (resume)
        if not self._step_best_list and self.ckpt_dir.exists():
            pattern = re.compile(
                r"^best_step=(\d+)-val_loss=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\.tar$"
            )
            for p in self.ckpt_dir.glob("best_step=*.tar"):
                m = pattern.fullmatch(p.name)
                if m:
                    s, v = int(m.group(1)), float(m.group(2))
                    if math.isfinite(v):
                        self._step_best_list.append((s, v, p.name))
            self._step_best_list.sort(key=lambda x: x[1])  # lowest loss first
            self._step_best_list = self._step_best_list[: self.save_top_k]

        fname = f"best_step={step:06d}-val_loss={val_loss:.3f}.tar"
        candidate_list = self._step_best_list + [(step, val_loss, fname)]
        candidate_list.sort(key=lambda x: x[1])
        is_top_one = is_new_best
        extra = (
            {"batch_cursor": int(batch_cursor), "epoch_incomplete": True}
            if batch_cursor is not None
            else {}
        )
        state = self._build_state(
            epoch,
            rng_states_by_rank=rng_states_by_rank,
            **extra,
        )
        aliases = [self.ckpt_dir / fname]
        if is_top_one:
            aliases.append(self.ckpt_dir / "best_model.tar")
        self._atomic_save_aliases(state, aliases)
        self._step_best_list = candidate_list
        # Keep top save_top_k; remove files that dropped out
        for t in self._step_best_list[self.save_top_k :]:
            _, _, fn = t
            path = self.ckpt_dir / fn
            if path.exists():
                try:
                    path.unlink()
                    logger.info(f"Removed old step-best checkpoint (out of top_k): {fn}")
                except OSError as e:
                    logger.warning("Could not remove %s: %s", path, e)
        self._step_best_list = self._step_best_list[: self.save_top_k]

        if is_top_one:
            logger.info(f"Best model (step) saved at step {step} (val_loss={val_loss:.3f})")

    @staticmethod
    def _load_resolved_config_sidecar(path: Path) -> dict:
        """读取 trainer 自己生成的 resolved-config sidecar。"""
        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as source:
                config = json.load(source)
        else:
            import yaml

            with path.open("r", encoding="utf-8") as source:
                config = yaml.safe_load(source)
        if not isinstance(config, dict):
            raise RuntimeError(f"resolved config sidecar must contain a mapping: {path}")
        return config

    def _select_resolved_config_sidecar(self) -> Path:
        """为当前配置选择不可变 sidecar；配置漂移时使用内容寻址文件名。"""
        from core.trainer.checkpoint_state import resolved_config_identity

        identity = resolved_config_identity(self.config)
        candidates = sorted(
            {
                *self.ckpt_dir.glob("resolved_config*.yaml"),
                *self.ckpt_dir.glob("resolved_config*.yml"),
                *self.ckpt_dir.glob("resolved_config*.json"),
            },
            key=lambda path: (
                path.name not in {"resolved_config.yaml", "resolved_config.json"},
                path.name,
            ),
        )
        for candidate in candidates:
            existing = self._load_resolved_config_sidecar(candidate)
            if resolved_config_identity(existing) == identity:
                return candidate

        try:
            import yaml  # noqa: F401
        except ImportError:
            suffix = ".json"
        else:
            suffix = ".yaml"
        original = self.ckpt_dir / f"resolved_config{suffix}"
        if not candidates and not original.exists():
            return original
        return self.ckpt_dir / f"resolved_config_resumed_{identity}{suffix}"

    def _write_resolved_config_sidecar(self, path: Path) -> None:
        """仅创建或复核 immutable sidecar，禁止覆盖既有配置来源。"""
        from core.trainer.checkpoint_state import resolved_config_identity

        expected_identity = resolved_config_identity(self.config)
        try:
            with path.open("x", encoding="utf-8") as destination:
                if path.suffix == ".json":
                    json.dump(self.config, destination, indent=2)
                else:
                    import yaml

                    yaml.safe_dump(
                        self.config,
                        destination,
                        default_flow_style=False,
                        sort_keys=False,
                    )
        except FileExistsError:
            existing = self._load_resolved_config_sidecar(path)
            if resolved_config_identity(existing) != expected_identity:
                raise RuntimeError(
                    f"resolved config sidecar identity collision: {path}"
                )

    @staticmethod
    def _write_provenance_json(path: Path, payload: dict) -> Path:
        """以 ``open("x")`` 独占写入 provenance；同名已存在时追加序号，绝不覆盖既有文件。"""
        index = 0
        while True:
            candidate = (
                path
                if index == 0
                else path.with_name(f"{path.stem}_{index}{path.suffix}")
            )
            try:
                with candidate.open("x", encoding="utf-8") as destination:
                    json.dump(payload, destination, indent=2)
                return candidate
            except FileExistsError:
                index += 1

    def _save_experiment_context(self):
        """保存实验上下文，并固化当前 checkpoint 配置来源。"""
        import platform
        from core.trainer.checkpoint_state import build_resolved_config_provenance

        config_path = self._select_resolved_config_sidecar()
        self._write_resolved_config_sidecar(config_path)
        self._resolved_config_provenance = build_resolved_config_provenance(
            self.config,
            config_path.name,
        )

        # Git info
        git_info = self._get_git_info()
        git_base = "git_info_resumed" if (self.ckpt_dir / "git_info.json").exists() else "git_info"
        git_path = self.ckpt_dir / f"{git_base}.json"
        self._write_provenance_json(git_path, git_info)

        # Environment info
        env_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "command_line": " ".join(sys.argv),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        env_base = "environment_resumed" if (self.ckpt_dir / "environment.json").exists() else "environment"
        env_path = self.ckpt_dir / f"{env_base}.json"
        self._write_provenance_json(env_path, env_info)

        logger.info(f"Experiment context saved to {self.ckpt_dir}")

    @staticmethod
    def _get_git_info() -> dict:
        """Get git commit hash and branch name (if in git repo)."""
        info = {"commit": None, "branch": None, "is_dirty": False, "remote_url": None}
        try:
            # Check if in git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return info

            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["commit"] = result.stdout.strip()

            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["branch"] = result.stdout.strip()

            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["is_dirty"] = len(result.stdout.strip()) > 0

            # Get remote URL
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["remote_url"] = result.stdout.strip()

        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass  # Git not available or not in repo
        return info

    def _get_git_info_safe(self) -> dict:
        """Wrapper that never raises during checkpoint build (avoids abort if git hangs or fails)."""
        try:
            return self._get_git_info()
        except Exception as e:
            logger.warning("Could not get git info for checkpoint: %s", e)
            return {"commit": None, "branch": None, "is_dirty": False, "remote_url": None}

    def _get_rng_state(self) -> dict:
        """Get RNG states for reproducibility. On CUDA/git failure returns partial state to avoid abort during save.

        说明：动态混合数据集（``DynamicMixingDataset`` / AEC 同类）的样本顺序与增强主要由各 rank 的
        ``SeedSequence``（及 ``meta.seed``）驱动；checkpoint 内的 RNG 状态主要用于优化器/丢弃与设备上
        随机算子的一致性。**不建议**依赖「仅恢复 RNG」来复现整条动态数据流水顺序；若需可复现性，应固定
        ``meta.seed``、DataLoader worker/deterministic 设置，并查阅各 Dataset 文档。
        """
        rng_state = {"torch": None, "torch_cuda": None, "numpy": None, "python": None}
        try:
            rng_state["torch"] = torch.get_rng_state().cpu().numpy().tolist()
        except Exception as e:
            logger.warning("Could not get torch RNG state for checkpoint: %s", e)
        try:
            if torch.cuda.is_available():
                rng_state["torch_cuda"] = [s.cpu().numpy().tolist() for s in torch.cuda.get_rng_state_all()]
        except Exception as e:
            logger.warning("Could not get CUDA RNG state for checkpoint: %s", e)
        try:
            rng_state["numpy"] = np.random.get_state()
        except Exception as e:
            logger.warning("Could not get numpy RNG state for checkpoint: %s", e)
        try:
            rng_state["python"] = random.getstate() if hasattr(random, "getstate") else None
        except Exception as e:
            logger.warning("Could not get python RNG state for checkpoint: %s", e)
        return rng_state

    def _gather_rng_states_for_checkpoint(self) -> list[dict]:
        """在正常 checkpoint 保存前按 rank 顺序收集 RNG 状态。"""
        local_state = self._get_rng_state()
        if self.world_size <= 1:
            return [local_state]
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "Distributed checkpoint RNG gathering requires an initialized process group"
            )
        gathered: list[dict | None] = [None] * self.world_size
        dist.all_gather_object(gathered, local_state)
        if any(not isinstance(state, dict) for state in gathered):
            raise RuntimeError("Distributed checkpoint RNG gathering returned invalid state")
        return [state for state in gathered if state is not None]

    def _save_middle_checkpoint(self, epoch: int, batch: int):
        """Save a middle checkpoint and synchronize only after a successful save.

        Rank 0's save result is synchronized before the barrier.  If building or
        writing the checkpoint fails, every initialized DDP rank raises instead
        of entering a barrier that rank 0 can no longer reach.

        保存完整训练状态（与 latest_model.tar 同格式，含 model/optimizer/scheduler/
        EMA/scaler/epoch/n_steps），并额外记录 batch cursor：本 epoch 已完整完成
        （含 optimizer step 边界）的 batch 数。传入的 batch 是刚处理完的 0-based
        下标，已完成数为 batch+1；若 crash 发生在 gradient accumulation 中途，
        未 step 的累积梯度无法恢复，故 cursor 向下对齐到上一个 accumulation 边界，
        resume 时重做该窗口内的 batch（不重复/遗漏 optimizer step）。
        """
        rng_states_by_rank = self._gather_rng_states_for_checkpoint()
        save_error: str | None = None
        if self.rank == 0:
            try:
                accum = max(1, int(self.accumulation_steps))
                batch_cursor = ((batch + 1) // accum) * accum
                state = self._build_state(
                    epoch,
                    rng_states_by_rank=rng_states_by_rank,
                    batch=batch,
                    batch_cursor=batch_cursor,
                    epoch_incomplete=True,
                )
                path = self.ckpt_dir / f"model_E{epoch:04d}_B{batch:06d}.pth"
                self._atomic_save(state, path)
                logger.info(
                    f"Middle checkpoint saved: {path.name} (batch cursor={batch_cursor})"
                )
            except Exception as exc:
                save_error = (
                    f"rank 0 middle checkpoint save failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.error(save_error)

        self._sync_checkpoint_save_result(
            save_error,
            barrier_phase="barrier_after_mid_epoch_save",
        )

    @staticmethod
    def _middle_checkpoint_key(path: Path) -> tuple[int, int] | None:
        """解析 middle checkpoint 文件名中的 `(epoch, batch)`。"""
        import re

        match = re.match(r"^model_E(\d+)_B(\d+)\.pth$", path.name)
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    @classmethod
    def _find_latest_middle_checkpoint(cls, directory: Path) -> Path | None:
        """在 directory 中找 `(epoch, batch)` 最大的 middle checkpoint。"""
        candidates = (
            (key, path)
            for path in directory.glob("model_E*_B*.pth")
            if (key := cls._middle_checkpoint_key(path)) is not None
        )
        return max(candidates, key=lambda item: (item[0], str(item[1])), default=(None, None))[1]

    @classmethod
    def _find_latest_middle_checkpoint_across_runs(
        cls, checkpoint_root: Path
    ) -> Path | None:
        """跨运行目录按 `(epoch, batch)` 选择最新 middle checkpoint。"""
        candidates = (
            (key, path)
            for path in checkpoint_root.glob("*/model_E*_B*.pth")
            if (key := cls._middle_checkpoint_key(path)) is not None
        )
        return max(candidates, key=lambda item: (item[0], str(item[1])), default=(None, None))[1]

    @staticmethod
    def _safe_load(path, **kwargs) -> dict:
        """显式以 unsafe 模式加载可信的完整 resume checkpoint。

        optimizer/scheduler state 可能依赖 pickle；调用方只能用于恢复完整训练状态，
        不得用于仅加载模型权重的路径。
        """
        from core.utils.safe_load import safe_torch_load
        return safe_torch_load(path, allow_unsafe=True, purpose="resume (optimizer state)")

    @staticmethod
    def _safe_load_model_only(path) -> dict:
        """安全加载 torch 或 safetensors model-only checkpoint。"""
        from core.utils.safe_load import safe_model_weights_load

        return safe_model_weights_load(
            path,
            purpose="initialize trainer model weights from pretrained checkpoint",
        )

    def _load_model_state_checked(
        self,
        state_dict: dict,
        model_key: str | None = None,
        *,
        context: str = "checkpoint",
    ) -> None:
        """严格加载模型权重，只迁移完整的 legacy ``module.`` 前缀。"""
        from core.trainer.checkpoint_state import validate_model_state

        model = self._raw_model(model_key)
        normalized = validate_model_state(
            state_dict,
            model.state_dict(),
            context=context,
        )
        try:
            model.load_state_dict(normalized, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"{context}: failed to load model state: {exc}") from exc

    @staticmethod
    def _checkpoint_content_identity(path: Path) -> tuple[int, str]:
        """计算稳定的 checkpoint 内容身份，拒绝读取期间发生变化的文件。"""
        from core.trainer.checkpoint_state import checkpoint_content_identity

        return checkpoint_content_identity(path)

    def _run_distributed_checkpoint_load(
        self,
        context: str,
        path: Path | None,
        loader: Callable[[Path], tuple],
        *,
        allow_missing: bool,
    ) -> tuple | None:
        """各 rank 完成本地 load 后汇聚 outcome，并一致地成功或抛错。"""
        local_result: tuple | None = None
        local_error: BaseException | None = None
        identity: tuple[int, str] | None = None
        try:
            if path is None or not path.exists():
                if not allow_missing:
                    raise FileNotFoundError(f"checkpoint not found: {path}")
            else:
                identity = self._checkpoint_content_identity(path)
                local_result = loader(path)
                if self._checkpoint_content_identity(path) != identity:
                    raise RuntimeError("checkpoint changed while loading local state")
        except Exception as exc:
            local_error = exc

        distributed = (
            self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )
        if not distributed:
            if local_error is not None:
                raise local_error
            return local_result

        local_outcome = (
            "error",
            type(local_error).__name__,
            str(local_error),
        ) if local_error is not None else ("ok", identity, local_result)
        outcomes: list[tuple | None] = [None] * self.world_size
        self._nccl_flow_log(f"{context}_load_outcome")
        dist.all_gather_object(outcomes, local_outcome)
        failures = [
            f"rank {rank}: {outcome[1]}: {outcome[2]}"
            for rank, outcome in enumerate(outcomes)
            if outcome is not None and outcome[0] == "error"
        ]
        if failures:
            raise RuntimeError(f"{context} failed across ranks: " + "; ".join(failures))
        if any(outcome != outcomes[0] for outcome in outcomes[1:]):
            details = "; ".join(
                f"rank {rank}: {outcome!r}"
                for rank, outcome in enumerate(outcomes)
            )
            raise RuntimeError(f"{context} results differ across ranks: {details}")
        return local_result

    def _resume_checkpoint(self):
        """Resume training state from checkpoint (model/optimizer/scheduler/AMP/EMA/RNG/batch cursor).

        ``resume_load_from`` 提供时严格读取该路径；否则搜索顺序为：
        --resume-from 显式路径 → latest_model.tar（本目录，再跨运行目录）→ 最新 middle
        checkpoint（model_E*_B*.pth，本目录取 E/B 最大者，再跨目录取最新）。middle checkpoint
        含 batch cursor 时，resume 到 (epoch E, batch B)：epoch E 内跳过前 B 个 batch
        继续训练，而不是从 epoch E+1 开始。

        探测语义分工：本方法内的「显式路径不存在则 warn + 回退自动发现」只服务直接构造
        BaseTrainer 的调用方；训练入口（``UnifiedTrainingEntry``）在 preview 阶段自行解析
        checkpoint 并始终传入严格 ``resume_load_from``，显式路径缺失时由 entry 侧 fail-fast
        （见 ``core/trainer/entry.py::_resolve_resume_source_context``），不会走到本回退分支。
        """
        resume_load_from = getattr(self, "_resume_load_from", None)
        ckpt_path = Path(resume_load_from) if resume_load_from else None
        if ckpt_path is None and self._resume_from:
            p = Path(self._resume_from)
            if p.exists():
                ckpt_path = p
            else:
                logger.warning(f"--resume-from checkpoint not found: {p}; falling back to auto discovery")
        if ckpt_path is None:
            ckpt_path = self.ckpt_dir / "latest_model.tar"
            if not ckpt_path.exists():
                candidates = sorted(self.ckpt_dir.parent.glob("*/latest_model.tar"))
                ckpt_path = candidates[-1] if candidates else None
            if ckpt_path is None or not ckpt_path.exists():
                # 回退到 middle checkpoint：先本目录取 E/B 最大者，再跨运行目录取最新
                ckpt_path = self._find_latest_middle_checkpoint(self.ckpt_dir)
                if ckpt_path is None:
                    ckpt_path = self._find_latest_middle_checkpoint_across_runs(
                        self.ckpt_dir.parent
                    )
        self._resume_transaction_snapshots = []
        try:
            result = self._run_distributed_checkpoint_load(
                "resume checkpoint",
                ckpt_path,
                self._load_resume_checkpoint_local,
                allow_missing=not bool(resume_load_from),
            )
            if result is None:
                logger.warning("No checkpoint found for resume")
        except BaseException:
            for snapshot in reversed(self._resume_transaction_snapshots):
                self._restore_resume_snapshot(snapshot)
            raise
        finally:
            del self._resume_transaction_snapshots

    @staticmethod
    def _validate_optimizer_state_dict(
        state: object,
        optimizer: torch.optim.Optimizer,
        *,
        context: str,
    ) -> None:
        """校验分组、已知逐参数累积量与有限非负实数 step。

        torch.optim / AdamP / SGDP 的已知动量与统计量必须是同 shape/dtype
        Tensor；其他字段留给 optimizer 自身恢复，包括 LBFGS 的全局扁平向量、
        history 列表和 ASGD 的标量 eta/mu，不假设它们具有单参数的形状。
        """
        if not isinstance(state, dict):
            raise TypeError(f"{context}: optimizer state must be a mapping")
        saved_groups = state.get("param_groups")
        saved_states = state.get("state")
        if not isinstance(saved_groups, list) or not isinstance(saved_states, dict):
            raise TypeError(f"{context}: optimizer state has invalid param_groups/state")
        current_groups = optimizer.param_groups
        if len(saved_groups) != len(current_groups):
            raise ValueError(
                f"{context}: loaded state dict has a different number of parameter groups"
            )
        saved_to_parameter: dict[object, torch.nn.Parameter] = {}
        for group_index, (saved_group, current_group) in enumerate(
            zip(saved_groups, current_groups)
        ):
            if not isinstance(saved_group, dict) or not isinstance(
                saved_group.get("params"), list
            ):
                raise TypeError(f"{context}: parameter group {group_index} is malformed")
            saved_params = saved_group["params"]
            current_params = current_group.get("params", [])
            if len(saved_params) != len(current_params):
                raise ValueError(
                    f"{context}: parameter group {group_index} has {len(saved_params)} "
                    f"parameters, expected {len(current_params)}"
                )
            for saved_id, parameter in zip(saved_params, current_params):
                if isinstance(saved_id, bool) or not isinstance(saved_id, int):
                    raise TypeError(f"{context}: parameter id {saved_id!r} is invalid")
                if saved_id in saved_to_parameter:
                    raise ValueError(f"{context}: duplicate parameter id {saved_id}")
                saved_to_parameter[saved_id] = parameter

        for saved_id, parameter_state in saved_states.items():
            if saved_id not in saved_to_parameter:
                raise ValueError(
                    f"{context}: state contains unknown parameter id {saved_id!r}"
                )
            if not isinstance(parameter_state, dict):
                raise TypeError(
                    f"{context}: state for parameter id {saved_id} must be a mapping"
                )
            parameter = saved_to_parameter[saved_id]
            for state_name, value in parameter_state.items():
                if state_name == "step":
                    scalar = value
                    if isinstance(value, torch.Tensor):
                        if value.ndim != 0 or value.is_complex() or value.dtype == torch.bool:
                            raise ValueError(
                                f"{context}: state 'step' must be a finite nonnegative real scalar"
                            )
                        scalar = value.item()
                    if (
                        isinstance(scalar, (bool, np.bool_))
                        or not isinstance(scalar, (int, float, np.integer, np.floating))
                        or not math.isfinite(scalar)
                        or scalar < 0
                    ):
                        raise ValueError(
                            f"{context}: state 'step' must be a finite nonnegative real scalar"
                        )
                    continue
                if state_name not in {
                    "momentum_buffer", "exp_avg", "exp_avg_sq", "max_exp_avg_sq",
                    "exp_inf", "sum", "grad_avg", "square_avg", "acc_delta",
                    "ax", "prev", "step_size",
                }:
                    continue
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"{context}: state '{state_name}' for parameter id {saved_id} "
                        "must be a tensor"
                    )
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(
                        f"{context}: state '{state_name}' for parameter id {saved_id} "
                        f"shape mismatch: expected {tuple(parameter.shape)}, "
                        f"got {tuple(value.shape)}"
                    )
                if value.dtype != parameter.dtype:
                    raise TypeError(
                        f"{context}: state '{state_name}' for parameter id {saved_id} "
                        f"dtype mismatch: expected {parameter.dtype}, got {value.dtype}"
                    )

    def _validate_ema_state(self, state: object, *, context: str = "Resume EMA") -> None:
        """严格校验 EMA shadow 参数与 raw model parameters 的数量和类型。"""
        if not isinstance(state, dict):
            raise TypeError(f"{context}: EMA state must be a mapping")
        shadows = state.get("shadow_params")
        if not isinstance(shadows, (list, tuple)):
            raise TypeError(f"{context}: shadow_params must be a list or tuple")
        parameters = list(self._raw_model().parameters())
        if len(shadows) != len(parameters):
            raise ValueError(
                f"{context}: shadow_params count mismatch: expected {len(parameters)}, "
                f"got {len(shadows)}"
            )
        for index, (shadow, parameter) in enumerate(zip(shadows, parameters)):
            if not isinstance(shadow, torch.Tensor):
                raise TypeError(
                    f"{context}: shadow_params[{index}] must be a tensor, "
                    f"got {type(shadow).__name__}"
                )
            if tuple(shadow.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"{context}: shadow_params[{index}] shape mismatch: "
                    f"expected {tuple(parameter.shape)}, got {tuple(shadow.shape)}"
                )
            if shadow.dtype != parameter.dtype:
                raise TypeError(
                    f"{context}: shadow_params[{index}] dtype mismatch: "
                    f"expected {parameter.dtype}, got {shadow.dtype}"
                )

    def _validate_resume_checkpoint(self, ckpt: dict) -> None:
        """在修改 live trainer 前校验进度、模型、optimizer 与 EMA 合同。"""
        from core.trainer.checkpoint_state import (
            validate_model_state,
            validate_resume_progress,
        )

        train_loader_len = None
        train_loader = getattr(self, "train_loader", None)
        if train_loader is not None:
            try:
                train_loader_len = len(train_loader)
            except TypeError:
                # 直接 BaseTrainer 与 iterable-only loader 没有可用边界；仍执行
                # checkpoint schema、模型及 optimizer/scheduler 的完整合同校验。
                pass
        validate_resume_progress(
            ckpt,
            accumulation_steps=self.accumulation_steps,
            train_loader_len=train_loader_len,
        )
        model_items = (
            [(name, f"model_{name}") for name in self.models]
            if self._is_multi_model
            else [(None, "model")]
        )
        for name, key in model_items:
            if key not in ckpt:
                raise RuntimeError(
                    f"Resume: checkpoint is missing required model state '{key}'"
                )
            validate_model_state(
                ckpt[key],
                self._raw_model(name).state_dict(),
                context=f"Resume [{name}]" if name is not None else "Resume",
            )
        optimizer_items = (
            [(name, opt, f"optimizer_{name}") for name, opt in self.optimizers.items()]
            if self._is_multi_optimizer
            else [(None, self.optimizer, "optimizer")]
        )
        for _name, optimizer, key in optimizer_items:
            if key not in ckpt:
                raise RuntimeError(
                    f"Resume: checkpoint is missing required optimizer state '{key}'"
                )
            self._validate_optimizer_state_dict(
                ckpt[key],
                optimizer,
                context=f"Resume optimizer [{_name or 'default'}]",
            )
        scheduler_keys = (
            [f"scheduler_{name}" for name in self.schedulers]
            if self._is_multi_scheduler
            else ["scheduler"]
        )
        for key in scheduler_keys:
            if key not in ckpt:
                raise RuntimeError(
                    f"Resume: checkpoint is missing required scheduler state '{key}'"
                )
        if self.use_amp and self.amp_dtype == torch.float16 and "scaler" not in ckpt:
            raise RuntimeError(
                "Resume: checkpoint is missing required AMP scaler state 'scaler'"
            )
        if self.ema and "ema" in ckpt:
            self._validate_ema_state(ckpt["ema"])
        self._rng_state_for_current_rank(ckpt)

    def _rng_state_for_current_rank(self, ckpt: dict) -> dict | None:
        """选择当前 rank 的 RNG 状态；旧 checkpoint 回退到 legacy 键。"""
        if "rng_states_by_rank" not in ckpt:
            return ckpt.get("rng_state")

        saved_world_size = ckpt.get("rng_world_size")
        if (
            isinstance(saved_world_size, bool)
            or not isinstance(saved_world_size, int)
            or saved_world_size <= 0
        ):
            raise ValueError(
                "Resume: checkpoint 'rng_world_size' must be a positive integer"
            )
        states = ckpt["rng_states_by_rank"]
        if not isinstance(states, list) or len(states) != saved_world_size:
            raise ValueError(
                "Resume: checkpoint 'rng_states_by_rank' must be a list whose "
                "length matches 'rng_world_size'"
            )
        if any(not isinstance(state, dict) for state in states):
            raise TypeError(
                "Resume: every 'rng_states_by_rank' entry must be a mapping"
            )
        if saved_world_size != self.world_size:
            raise ValueError(
                "Resume: per-rank RNG world size mismatch: "
                f"checkpoint={saved_world_size}, current={self.world_size}"
            )
        if self.rank < 0 or self.rank >= saved_world_size:
            raise ValueError(
                f"Resume: rank {self.rank} is outside saved RNG world size "
                f"{saved_world_size}"
            )
        return states[self.rank]

    def _snapshot_resume_state(self) -> dict:
        """捕获 resume 可能修改的 live state，用于异常回滚。"""
        tracked_attrs = (
            "_resume_skip_batches", "start_epoch", "current_epoch", "n_steps",
            "best_score", "best_epoch", "best_at_step", "_patience_counter",
            "train_losses", "val_losses", "curve_epochs", "val_metrics_epochs",
            "val_metrics_history", "val_metrics_coverage_history", "_vm_epoch_count",
            "_train_loss_comp_hist",
            "_val_loss_comp_hist", "_curve_comp_names",
            "val_metrics_loss_comp_hist",
        )
        return {
            "models": {
                name: copy.deepcopy(
                    self._tensors_to_cpu(self._raw_model(name).state_dict())
                )
                for name in self.models
            },
            "optimizers": {
                name: copy.deepcopy(self._tensors_to_cpu(opt.state_dict()))
                for name, opt in self.optimizers.items()
            },
            "schedulers": {
                name: copy.deepcopy(self._tensors_to_cpu(sch.state_dict()))
                for name, sch in self.schedulers.items()
            },
            "scaler": copy.deepcopy(self._tensors_to_cpu(self.scaler.state_dict()))
            if self.use_amp and self.amp_dtype == torch.float16
            else None,
            "ema": copy.deepcopy(self._tensors_to_cpu(self.ema.state_dict()))
            if self.ema
            else None,
            "sampler": copy.deepcopy(
                self._tensors_to_cpu(self.train_sampler.state_dict())
            )
            if self.train_sampler is not None
            and hasattr(self.train_sampler, "state_dict")
            else None,
            "rng": self._get_rng_state(),
            "attrs": {
                key: (
                    key in self.__dict__,
                    copy.deepcopy(self._tensors_to_cpu(self.__dict__.get(key))),
                )
                for key in tracked_attrs
            },
        }

    def _restore_resume_snapshot(self, snapshot: dict) -> None:
        """恢复 `_snapshot_resume_state` 捕获的全部状态。"""
        for name, state in snapshot["models"].items():
            self._raw_model(name).load_state_dict(state, strict=True)
        for name, state in snapshot["optimizers"].items():
            self.optimizers[name].load_state_dict(state)
        for name, state in snapshot["schedulers"].items():
            self.schedulers[name].load_state_dict(state)
        if snapshot["scaler"] is not None:
            self.scaler.load_state_dict(snapshot["scaler"])
        if self.ema and snapshot["ema"] is not None:
            self.ema.load_state_dict(snapshot["ema"])
        if snapshot["sampler"] is not None:
            self.train_sampler.load_state_dict(snapshot["sampler"])
        self._restore_rng_state(snapshot["rng"], strict=True)
        for key, (existed, value) in snapshot["attrs"].items():
            if existed:
                setattr(self, key, value)
            else:
                self.__dict__.pop(key, None)

    def _load_resume_checkpoint_local(self, ckpt_path: Path) -> tuple:
        """恢复本 rank；快照保留至外层 identity/outcome gate 成功后释放。"""
        ckpt = self._safe_load(ckpt_path)
        self._validate_resume_checkpoint(ckpt)
        snapshot = self._snapshot_resume_state()
        transaction = getattr(self, "_resume_transaction_snapshots", None)
        if transaction is not None:
            transaction.append(snapshot)
            return self._apply_resume_checkpoint_local(ckpt_path, ckpt)
        try:
            return self._apply_resume_checkpoint_local(ckpt_path, ckpt)
        except BaseException:
            self._restore_resume_snapshot(snapshot)
            raise

    def _apply_resume_checkpoint_local(self, ckpt_path: Path, ckpt: dict) -> tuple:
        """在预校验与快照保护下应用已反序列化的 resume 状态。"""
        # Config drift detection: warn if checkpoint config differs from current
        ckpt_cfg = ckpt.get("config")
        if ckpt_cfg is not None and self.rank == 0:
            import yaml
            try:
                ckpt_yaml = yaml.safe_dump(dict(ckpt_cfg), default_flow_style=False, sort_keys=False)
                cur_yaml = yaml.safe_dump(dict(self.config), default_flow_style=False, sort_keys=False)
                if ckpt_yaml != cur_yaml:
                    logger.warning(
                        "Resume: checkpoint config differs from current config. "
                        "Critical hyperparameters (lr, batch_size, model arch, loss weights) "
                        "may have changed. Verify before continuing training."
                    )
            except Exception as exc:
                # yaml 缺失或配置不可序列化时必须留痕，否则漂移告警被静默丢弃。
                logger.warning(
                    "Resume: could not compare checkpoint config with current "
                    "config (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
        # batch cursor：middle checkpoint 记录 epoch 内已完整完成（含 optimizer step
        # 边界）的 batch 数。恢复时在 epoch 内跳过前 cursor 个 batch（sampler 顺序由
        # seed+epoch 确定性派生，跳过即复现 crash 位置；DDP 各 rank 读同一份
        # checkpoint，cursor 天然一致），而不是从 epoch+1 开始。
        from core.trainer.checkpoint_state import (
            resume_batch_cursor,
            resume_start_epoch,
        )

        batch_cursor = resume_batch_cursor(ckpt, self.accumulation_steps)
        self._resume_skip_batches = batch_cursor
        self.start_epoch = resume_start_epoch(ckpt, self.accumulation_steps)
        self.n_steps = ckpt["n_steps"]
        _bs = ckpt.get("best_score")
        self.best_score = np.inf if _bs is None else float(_bs)
        self.best_epoch = ckpt.get("best_epoch", -1)
        self.best_at_step = ckpt.get("best_at_step", None)
        self._patience_counter = ckpt.get("patience_counter", 0)
        self.train_losses = ckpt.get("train_losses", [])
        self.val_losses = ckpt.get("val_losses", [])
        self.curve_epochs = ckpt.get("curve_epochs", [])
        if hasattr(self, "val_metrics_epochs"):
            self.val_metrics_epochs = ckpt.get("val_metrics_epochs", [])
        if hasattr(self, "val_metrics_history"):
            self.val_metrics_history = ckpt.get("val_metrics_history", {})
        # Load model state(s) fail-closed: a partial/randomly initialized model
        # must never continue training after resume.
        if self._is_multi_model:
            for name in self.models:
                sd_key = f"model_{name}"
                if sd_key not in ckpt:
                    raise RuntimeError(
                        f"Resume: checkpoint is missing required model state '{sd_key}'"
                    )
                self._load_model_state_checked(
                    ckpt[sd_key], name, context=f"Resume [{name}]"
                )
        else:
            if "model" not in ckpt:
                raise RuntimeError("Resume: checkpoint is missing required model state 'model'")
            self._load_model_state_checked(ckpt["model"], context="Resume")

        # Load optimizer state(s) fail-closed: resuming without optimizer state
        # would silently reset momentum/step counters and change training semantics.
        lr_overrides: dict[str, bool] = {}
        configured_weight_decays = {
            name: [group.get("weight_decay", 0.0) for group in opt.param_groups]
            for name, opt in self.optimizers.items()
        }
        if self._is_multi_optimizer:
            for name, opt in self.optimizers.items():
                sd_key = f"optimizer_{name}"
                if sd_key not in ckpt:
                    raise RuntimeError(
                        f"Resume: checkpoint is missing required optimizer state '{sd_key}'"
                    )
                opt.load_state_dict(ckpt[sd_key])
                lr_overrides[name] = self._override_optimizer_hp_from_config(
                    opt, name, ckpt_cfg,
                    configured_weight_decays=configured_weight_decays[name],
                )
        else:
            if "optimizer" not in ckpt:
                raise RuntimeError(
                    "Resume: checkpoint is missing required optimizer state 'optimizer'"
                )
            self.optimizer.load_state_dict(ckpt["optimizer"])
            lr_overrides["default"] = self._override_optimizer_hp_from_config(
                self.optimizer,
                None,
                ckpt_cfg,
                configured_weight_decays=configured_weight_decays["default"],
            )

        # Load scheduler progress first; an explicit current-config LR override then
        # rebases scheduler LR fields without resetting last_epoch or other progress.
        if self._is_multi_scheduler:
            for name, sch in self.schedulers.items():
                sd_key = f"scheduler_{name}"
                if sd_key not in ckpt:
                    raise RuntimeError(
                        f"Resume: checkpoint is missing required scheduler state '{sd_key}'"
                    )
                sch.load_state_dict(ckpt[sd_key])
                if lr_overrides.get(name, False):
                    self._rebase_scheduler_lrs(sch, self.optimizers[name])
        else:
            if "scheduler" not in ckpt:
                raise RuntimeError(
                    "Resume: checkpoint is missing required scheduler state 'scheduler'"
                )
            self.scheduler.load_state_dict(ckpt["scheduler"])
            if lr_overrides.get("default", False):
                self._rebase_scheduler_lrs(self.scheduler, self.optimizer)

        if self.use_amp and self.amp_dtype == torch.float16:
            if "scaler" not in ckpt:
                raise RuntimeError(
                    "Resume: checkpoint is missing required AMP scaler state 'scaler'"
                )
            self.scaler.load_state_dict(ckpt["scaler"])
        # 缺少 EMA 状态时，从当前模型权重重新同步 EMA shadow。
        if self.ema:
            if "ema" in ckpt:
                self.ema.load_state_dict(ckpt["ema"])
            else:
                if self.rank == 0:
                    logger.warning(
                        "Resume: checkpoint has no 'ema' key; EMA shadow re-synced "
                        "from current model weights (was random-initialized)."
                    )
                raw = self._raw_model()
                self.ema.shadow_params = [p.clone().detach() for p in raw.parameters()]
                self.ema.num_updates = 0

        self._restore_rng_state(self._rng_state_for_current_rank(ckpt), strict=True)

        # Restore sampler state if available
        if self.train_sampler is not None and "sampler_state" in ckpt:
            if hasattr(self.train_sampler, "load_state_dict"):
                self.train_sampler.load_state_dict(ckpt["sampler_state"])

        self.current_epoch = self.start_epoch  # sync current_epoch with resumed state
        _lr = self.optimizer.param_groups[0]["lr"]
        _last = getattr(self.scheduler, "last_epoch", None)
        _cursor_msg = (
            f", batch cursor={batch_cursor} (skip first {batch_cursor} batches of epoch {self.start_epoch})"
            if batch_cursor > 0 else ""
        )
        logger.info(
            f"Resumed from {ckpt_path.name}: epoch {ckpt['epoch']} (n_steps={self.n_steps})"
            f"{_cursor_msg}, current lr={_lr:.2e}, scheduler.last_epoch={_last}"
        )

        self._after_resume_checkpoint(ckpt)
        return (
            int(ckpt["epoch"]),
            int(self.n_steps),
            int(batch_cursor),
            int(self.start_epoch),
            getattr(self.scheduler, "last_epoch", None),
        )

    def _restore_rng_state(
        self,
        rng_state: dict | None,
        *,
        strict: bool = False,
    ) -> None:
        """Restore RNG states; resume 用 strict 模式汇聚本地恢复失败。"""
        if rng_state is None:
            logger.warning("No RNG state in checkpoint, using fresh random state")
            return

        try:
            # Restore torch CPU RNG
            if "torch" in rng_state and rng_state["torch"] is not None:
                torch.set_rng_state(torch.tensor(rng_state["torch"], dtype=torch.uint8))

            # Restore torch CUDA RNG
            if "torch_cuda" in rng_state and rng_state["torch_cuda"] is not None and torch.cuda.is_available():
                cuda_states = [torch.tensor(s, dtype=torch.uint8) for s in rng_state["torch_cuda"]]
                torch.cuda.set_rng_state_all(cuda_states)

            # Restore numpy RNG
            if "numpy" in rng_state and rng_state["numpy"] is not None:
                np.random.set_state(rng_state["numpy"])

            # Restore Python random
            if "python" in rng_state and rng_state["python"] is not None:
                random.setstate(tuple(rng_state["python"]))

            logger.info("RNG state restored from checkpoint")
        except Exception as e:
            if strict:
                raise
            logger.warning(f"Failed to restore RNG state: {e}")

    def load_pretrained(self, path: str, model_key: str | None = None):
        """Load pretrained weights (model params only, no optimizer/scheduler).

        Compatible with multiple checkpoint formats: model / model_state_dict / state_dict.
        For multi-model training, use ``model_key`` to specify which model to load.
        Checkpoint deserialization uses ``weights_only=True`` and fails closed when
        unsupported pickle globals are required.
        """
        path_obj = Path(path)
        self._run_distributed_checkpoint_load(
            "pretrained checkpoint",
            path_obj,
            lambda local_path: self._load_pretrained_local(local_path, model_key),
            allow_missing=False,
        )

    def _load_pretrained_local(
        self,
        path: Path,
        model_key: str | None,
    ) -> tuple:
        """在当前 rank 安全加载 pretrained 权重并返回确定性摘要。"""
        ckpt = self._safe_load_model_only(path)

        from core.trainer.checkpoint_state import (
            model_state_key_digest,
            select_pretrained_model_state,
        )

        sd = select_pretrained_model_state(
            ckpt,
            model_key=model_key,
            multi_model=self._is_multi_model,
        )
        if not isinstance(sd, dict):
            raise TypeError(
                f"Checkpoint does not contain a state_dict (got {type(sd).__name__}). "
                "Expected keys: 'model', 'model_state_dict', or 'state_dict', or a raw state_dict."
            )
        self._load_model_state_checked(
            sd,
            model_key,
            context=f"Pretrained [{model_key}]" if model_key else "Pretrained",
        )
        if self.ema is not None:
            raw = self._raw_model()
            self.ema.shadow_params = [p.clone().detach() for p in raw.parameters()]
            self.ema.num_updates = 0
            if self.rank == 0:
                logger.debug("EMA shadow synced from loaded pretrained weights")
        key_digest = model_state_key_digest(sd)
        return (model_key, len(sd), key_digest)

    # ── Shared training step helpers ──

    def _sync_ema_across_ranks(self) -> None:
        """Broadcast EMA shadow parameters from rank 0 to all ranks."""
        if self.ema is None or self.world_size <= 1:
            return
        for p in self.ema.shadow_params:
            dist.broadcast(p.data, src=0)

    def _prepare_optimizer_step(
        self,
        optimizer_key: str | None = None,
        *,
        skip_step: bool = False,
    ) -> PreparedOptimizerStep:
        """在任何参数更新前完成 unscale、sanitize、clip 与有限性检查。"""
        key = optimizer_key or "default"
        optimizer = self.optimizers.get(key, self.optimizer)
        target_model = (
            self.models.get(key)
            if key != "default" and self._is_multi_model
            else self.model
        )
        if target_model is None:
            target_model = self.model
        amp_fp16 = self.use_amp and self.amp_dtype == torch.float16
        prepared = PreparedOptimizerStep(
            key=key,
            optimizer=optimizer,
            amp_fp16=amp_fp16,
            eligible=False,
        )
        if skip_step:
            prepared.reason = "optimizer step was explicitly disabled"
            return prepared

        model_key = key if self._is_multi_model else None
        if not self._has_gradients(model_key=model_key):
            prepared.reason = "no gradients are present"
            return prepared

        if amp_fp16:
            self.scaler.unscale_(optimizer)
        gradients_finite = self._gradients_are_finite(model_key=model_key)
        if not gradients_finite:
            self._sanitize_gradients(model_key=model_key)
            if amp_fp16:
                prepared.reason = "unscaled gradients are non-finite"
                prepared.amp_nonfinite = True

        if self.clip_grad > 0:
            norm = torch.nn.utils.clip_grad_norm_(
                target_model.parameters(), self.clip_grad
            )
            if (
                self.rank == 0
                and (self.n_steps + 1) % self.log_interval == 0
            ):
                prepared.grad_norm = float(norm.item())

        post_clip_finite = self._gradients_are_finite(model_key=model_key)
        if not post_clip_finite:
            prepared.reason = "gradient clipping produced non-finite gradients"
            if amp_fp16:
                # 与原字符串判定保持一致：clip 后仍 non-finite 同样触发 AMP backoff。
                prepared.amp_nonfinite = True
        prepared.eligible = bool(
            post_clip_finite and (gradients_finite or not amp_fp16)
        )
        return prepared

    def _discard_prepared_optimizer_step(
        self,
        prepared: PreparedOptimizerStep,
        *,
        update_scaler: bool,
        amp_overflow: bool = False,
    ) -> None:
        """丢弃已 prepare 的梯度，并在边界末统一清理 GradScaler 状态。"""
        prepared.optimizer.zero_grad(set_to_none=True)
        self._pending_sched_step_keys.clear()
        if not prepared.amp_fp16 or not update_scaler:
            return
        current_scale = self.scaler.get_scale()
        new_scale = (
            current_scale * self.scaler.get_backoff_factor()
            if amp_overflow
            else current_scale
        )
        try:
            self.scaler.update(new_scale=new_scale)
            if amp_overflow:
                scaler_state = self.scaler.state_dict()
                if "_growth_tracker" in scaler_state:
                    scaler_state["_growth_tracker"] = 0
                    self.scaler.load_state_dict(scaler_state)
        except AssertionError:
            pass

    def _commit_prepared_optimizer_step(
        self,
        prepared: PreparedOptimizerStep,
        *,
        update_scaler: bool,
        increment_step: bool,
    ) -> bool:
        """提交已通过全局 gate 的 optimizer step 及其边界级副作用。"""
        if not prepared.eligible:
            raise RuntimeError("cannot commit an ineligible optimizer step")
        key = prepared.key
        optimizer = prepared.optimizer
        self.scaler.step(optimizer)
        if prepared.amp_fp16 and update_scaler:
            self.scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if increment_step:
            self.n_steps += 1

        if self.ema and update_scaler:
            ema_targets = getattr(self, "_ema_targets", None)
            if ema_targets is None or key in ema_targets:
                self.ema.update()
                if (
                    self.world_size > 1
                    and self.ema_sync_every_steps > 0
                    and self.n_steps % self.ema_sync_every_steps == 0
                ):
                    self._sync_ema_across_ranks()
        if (
            self.rank == 0
            and self.n_steps % self.log_interval == 0
            and self.n_steps > 0
        ):
            self._log_scalar("train/grad_norm", prepared.grad_norm, self.n_steps)

        if self.lr_per_step:
            scheduler = (
                self.schedulers.get(key, self.scheduler)
                if self._is_multi_scheduler
                else self.scheduler
            )
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.n_steps == 1:
                    logger.warning(
                        "lr_schedule_per_step=True is incompatible with "
                        "ReduceLROnPlateau — it will never be stepped.  "
                        "Set lr_schedule_per_step: false in your config."
                    )
            else:
                self._last_step_lr = (
                    scheduler.get_last_lr()[0]
                    if hasattr(scheduler, "get_last_lr")
                    else optimizer.param_groups[0]["lr"]
                )
                if not update_scaler:
                    self._pending_sched_step_keys.clear()
                    self._pending_sched_step_keys.append(key)
                else:
                    for pending_key in self._pending_sched_step_keys:
                        if pending_key == key:
                            continue
                        pending_scheduler = (
                            self.schedulers.get(pending_key, self.scheduler)
                            if self._is_multi_scheduler
                            else self.scheduler
                        )
                        if not isinstance(
                            pending_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                        ):
                            pending_scheduler.step()
                    self._pending_sched_step_keys.clear()
                    scheduler.step()
        return True

    def _optimizer_step(
        self,
        optimizer_key: str | None = None,
        *,
        update_scaler: bool = True,
        increment_step: bool = True,
        skip_step: bool = False,
    ) -> bool:
        """Prepare, globally gate, then commit one optimizer update."""
        prepared = self._prepare_optimizer_step(
            optimizer_key,
            skip_step=skip_step,
        )
        from core.trainer.optimizer_step import optimizer_gate_values

        eligible, no_amp_overflow = self._allreduce_bool_ands(
            optimizer_gate_values(prepared),
            phase=f"optimizer_step_{prepared.key}_prepare",
        )
        if not eligible:
            self._discard_prepared_optimizer_step(
                prepared,
                update_scaler=update_scaler,
                amp_overflow=not no_amp_overflow,
            )
            if self.rank == 0 and prepared.reason:
                logger.warning("Skipping optimizer step because %s.", prepared.reason)
            return False
        return self._commit_prepared_optimizer_step(
            prepared,
            update_scaler=update_scaler,
            increment_step=increment_step,
        )

    # ── Logging ──

    def _log_train_loss_after_step(self) -> None:
        """Log every log_interval steps: average total loss, optional sub-losses, and lr."""
        if self.rank != 0:
            return
        if self.n_steps % self.log_interval != 0 or self.n_steps <= 0:
            return
        win = self._train_log_loss_win
        if not win:
            return
        avg = sum(win) / len(win)
        lr = getattr(self, "_last_step_lr", self.optimizer.param_groups[0]["lr"])
        self._log_scalar("train/loss", avg, self.n_steps)
        self._log_scalar("train/lr", lr, self.n_steps)
        sub_parts = ""
        if self.log_step_sub_losses > 0:
            bd = getattr(self.loss_fn, "_last_breakdown", None)
            if bd:
                # bd: list of (name, weight, raw, weighted)
                top = sorted(bd, key=lambda x: x[3], reverse=True)[: self.log_step_sub_losses]
                sub_parts = " | " + " | ".join(f"{n}={w:.4f}" for n, _, _, w in top)
                for n, _, _, w in top:
                    self._log_scalar(f"train/step_loss_comp/{n}", w, self.n_steps)
        logger.info(
            f"  step {self.n_steps} | loss {avg:.4f} | lr {lr:.2e}{sub_parts}"
        )

    def _log_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        if self.wandb_run:
            import wandb
            wandb.log({tag: value}, step=step)

    def _apply_curve_xticks(self, ax, x: list) -> None:
        """Set x-axis ticks: every point if n<=30, else step to ~15 ticks."""
        if len(x) <= 30:
            ax.set_xticks(x)
        else:
            step = max(1, (max(x) - min(x)) // 15)
            ax.set_xticks(range(min(x), max(x) + 1, step))

    def _save_curve_fig(self, fig, path: Path) -> bool:
        """Save figure to path; return True on success. Never raises. Always closes fig."""
        try:
            fig.savefig(path, dpi=self.curve_dpi, bbox_inches="tight")
            return True
        except OSError as e:
            logger.warning("Could not save curve image %s: %s", path, e)
            return False
        finally:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(fig)
            except Exception:
                pass

    def _should_save_curve(self, curve_length: int) -> bool:
        """True if we should save curve this time (save_curve_interval: 1=every time, N=every N validations)."""
        if self.save_curve_interval <= 0:
            return False
        if self.save_curve_interval == 1:
            return True
        return curve_length % self.save_curve_interval == 0

    def _save_loss_curve(self) -> None:
        """Plot train/val convergence curve and save to log_dir (rank 0 only).

        Saves PNG with title and annotation: model type, loss type from config.
        Optionally marks best epoch; exports CSV for external plotting.
        Reference: tse_pse wesep/bin/train_se.py (matplotlib PNG in exp_dir).
        """
        if self.rank != 0 or not self.train_losses or not self.val_losses:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping loss curve save")
            return
        n = min(len(self.train_losses), len(self.val_losses), len(self.curve_epochs))
        if n == 0:
            return
        x = self.curve_epochs[:n]
        x_is_step = getattr(self, "_curve_x_is_step", False)
        x_name = "Step" if x_is_step else "Epoch"
        model_type = self.config.get("model", {}).get("type", "model")
        loss_type = self.config.get("loss", {}).get("type", "loss")
        title = f"Train & Validation Loss — {model_type} | {loss_type}"

        fig, ax = plt.subplots(figsize=self.curve_figsize)
        tl = np.asarray(self.train_losses[:n], dtype=np.float64)
        train_mask = np.isfinite(tl)
        if train_mask.any():
            ax.plot(
                np.asarray(x)[train_mask],
                tl[train_mask],
                "b-",
                label="Train Loss",
                linewidth=0.8,
            )
        elif x_is_step:
            ax.plot([], [], "b-", label="Train Loss (N/A until epoch end in step-val mode)", linewidth=0.8)
        # val_losses: mean val loss (lower is better)
        ax.plot(x, self.val_losses[:n], "c-", label="Val loss", linewidth=0.8)
        best_x = None
        best_lbl = None
        if x_is_step and getattr(self, "best_at_step", None) is not None:
            bx = self.best_at_step
            if bx is not None and bx in x:
                best_x, best_lbl = bx, f"Best (step {bx})"
        elif not x_is_step and self.best_epoch >= 0 and self.best_epoch in x:
            best_x, best_lbl = self.best_epoch, f"Best (E{self.best_epoch})"
        if best_x is not None:
            ax.axvline(
                x=best_x,
                color="green",
                linestyle="--",
                alpha=0.7,
                linewidth=0.8,
                label=best_lbl,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(x_name)
        ax.set_ylabel("Loss")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        self._apply_curve_xticks(ax, x)
        exp_path = self.log_dir.parent.parent
        fig.text(0.5, 0.01, f"{exp_path}  |  Val = mean val loss (lower is better)", ha="center", fontsize=7, color="gray")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = self.log_dir / "train_val_loss.png"
        if self._save_curve_fig(fig, path):
            logger.info(f"Loss curve saved: {path}")

        self._write_train_val_curve_csv(n, x, x_is_step)

    def _write_train_val_curve_csv(self, n: int, x: list, x_is_step: bool) -> None:
        """Write train_val_curve.csv (epoch/step, train_loss, val_loss). Subclasses may extend."""
        csv_path = self.log_dir / "train_val_curve.csv"
        try:
            xcol = "step" if x_is_step else "epoch"
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(f"{xcol},train_loss,val_loss\n")
                for i in range(n):
                    tr = self.train_losses[i]
                    if isinstance(tr, (int, float)) and tr == tr:
                        tr_s = f"{float(tr):.6f}"
                    else:
                        tr_s = "nan"
                    f.write(f"{x[i]},{tr_s},{self.val_losses[i]:.6f}\n")
        except OSError as e:
            logger.warning("Could not write loss curve CSV: %s", e)

    def _on_loss_curve_point_added(
        self, x_val: int, train_loss: Optional[float], val_loss: float
    ) -> None:
        """Hook when train/val loss curve gains a point (after append). Override in subclass."""
        return

    def _override_optimizer_hp_from_config(
        self,
        optimizer: torch.optim.Optimizer,
        name: str | None,
        ckpt_cfg: dict | None = None,
        *,
        configured_weight_decays: list[float] | None = None,
    ) -> bool:
        """恢复后应用当前 optimizer 配置，并返回是否显式覆盖了 LR。

        当 checkpoint 与当前配置中的 LR 相同，不覆盖 checkpoint 内已经衰减的
        optimizer LR；当当前配置明确改变 LR 时，调用方必须同步重设 scheduler
        的 LR 基准，但保留其 ``last_epoch`` 等进度状态。
        """
        opt_cfg = self.config.get("optimizer", {})
        if name is not None:
            opt_cfg = opt_cfg.get(name, opt_cfg)
        new_args = opt_cfg.get("args", {})
        if not new_args:
            return False
        hp_keys = {"lr", "betas", "eps", "weight_decay", "momentum", "dampening"}
        saved_lr = None
        saved_weight_decay = None
        if ckpt_cfg is not None:
            saved_opt_cfg = ckpt_cfg.get("optimizer", {})
            if name is not None:
                saved_opt_cfg = saved_opt_cfg.get(name, saved_opt_cfg)
            saved_args = saved_opt_cfg.get("args", {}) or {}
            saved_lr = saved_args.get("lr")
            saved_weight_decay = saved_args.get("weight_decay")
        if saved_lr is not None and saved_lr == new_args.get("lr"):
            hp_keys.discard("lr")
        weight_decay_override = "weight_decay" in new_args and (
            saved_weight_decay is None
            or not (
                isinstance(saved_weight_decay, (int, float))
                and isinstance(new_args["weight_decay"], (int, float))
                and math.isclose(
                    float(saved_weight_decay),
                    float(new_args["weight_decay"]),
                    rel_tol=1e-9,
                )
            )
        )
        changed = []
        lr_changed = False
        for group_index, group in enumerate(optimizer.param_groups):
            for key in (hp_keys - {"weight_decay"}) & new_args.keys():
                old = group.get(key)
                new = new_args[key]
                if not (
                    isinstance(old, float)
                    and isinstance(new, float)
                    and math.isclose(old, new, rel_tol=1e-9)
                ) and old != new:
                    group[key] = new
                    lr_changed = lr_changed or key == "lr"
                    changed.append(f"{key}: {old} -> {new}")
            if not weight_decay_override:
                continue
            old = group.get("weight_decay", 0.0)
            new = new_args["weight_decay"]
            if new == 0.0:
                target = new
            elif configured_weight_decays is not None and group_index < len(
                configured_weight_decays
            ):
                target = new if configured_weight_decays[group_index] != 0.0 else 0.0
            elif old != 0.0:
                target = new
            else:
                target = old
            if not (
                isinstance(old, float)
                and isinstance(target, float)
                and math.isclose(old, target, rel_tol=1e-9)
            ) and old != target:
                group["weight_decay"] = target
                changed.append(
                    f"weight_decay[group={group_index}]: {old} -> {target}"
                )
        if changed and self.rank == 0:
            logger.info(f"Resume: overridden optimizer hyperparameters: {', '.join(changed)}")
        return lr_changed

    @staticmethod
    def _rebase_scheduler_lrs(scheduler: Any, optimizer: torch.optim.Optimizer) -> None:
        """让已恢复 scheduler 使用当前显式 LR，同时保留调度进度。

        PyTorch 的乘法调度器使用 ``base_lrs`` 或当前 optimizer LR；项目内的
        warmup/cosine 调度器则由 ``max_lr``/``min_lr`` 生成绝对 LR。必须同步
        缩放其绝对 LR 边界，避免下一次 ``step()`` 使用错误的 LR 量级。
        """
        current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        rebase = getattr(scheduler, "rebase", None)
        if callable(rebase):
            rebase(optimizer=optimizer)
            return
        previous_base_lrs = list(getattr(scheduler, "base_lrs", ()))
        if previous_base_lrs and float(previous_base_lrs[0]) != 0.0:
            scale = current_lrs[0] / float(previous_base_lrs[0])
            for attr in ("max_lr", "min_lr"):
                value = getattr(scheduler, attr, None)
                if isinstance(value, (int, float)):
                    setattr(scheduler, attr, float(value) * scale)
        for group, lr in zip(optimizer.param_groups, current_lrs):
            group["initial_lr"] = lr
        if hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = list(current_lrs)
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = list(current_lrs)

    def _after_resume_checkpoint(self, ckpt: dict) -> None:
        """Hook after checkpoint state restored. Override in subclass for extra state."""
        return

    def _clear_cuda_cache(self, reason: str) -> None:
        """同步 CUDA 流并释放缓存显存（审计 A17：仅 OOM 恢复或显式开启时调用）。

        无条件的每 epoch ``synchronize + empty_cache + gc.collect`` 会强制 host
        等待所有在途 kernel（NCCL all-reduce、DataLoader prefetch），形成 GPU
        空泡；默认不清理，需要时由 ``trainer.clear_cache_each_epoch`` 开启，
        或在捕获 CUDA OOM 时由训练循环调用。
        """
        if not torch.cuda.is_available():
            return
        # 先 synchronize 再 empty_cache：否则在途 kernel 仍引用的 block 无法回收
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        if self.rank == 0:
            _alloc = torch.cuda.memory_allocated() / 1024**3
            _resv = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"  [CUDA cache cleared: {reason}] "
                f"allocated={_alloc:.2f}GB, reserved={_resv:.2f}GB"
            )

    @staticmethod
    def _is_valid_validation_result(value: object) -> bool:
        """仅有限实数可提交到 scheduler、曲线和 early-stop 状态。"""
        if value is None or isinstance(value, (bool, np.bool_)):
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    # ── Main training loop ──

    def train(self) -> None:
        """Main training loop: epoch → train → validate → checkpoint."""
        val_loss = None
        # 记录最近一次已保存（周期保存或 best 保存）的 epoch，供循环结束后的
        # 兜底保存判重，避免同一 epoch 重复写盘
        last_saved_epoch = self.start_epoch - 1
        epochs_ran = 0
        try:
            if (
                # 保留 getattr 缺省：测试替身（BaseTrainer 子类但不跑完整
                # __init__）会直接进入 train()，直接属性访问会 AttributeError。
                getattr(self, "initial_validation", False)
                and self.start_epoch == 1
                and not self._sync_max_steps_reached()
            ):
                if self.rank == 0:
                    logger.info(
                        "Initial validation before epoch 1 (official PASE-family "
                        "epoch-0 validation)."
                    )
                self.current_epoch = 0
                for m in self.models.values():
                    m.eval()
                try:
                    self._validation_epoch(0)
                finally:
                    for m in self.models.values():
                        m.train()
            for epoch in range(self.start_epoch, self.epochs + 1):
                # 已恢复到上限（含 max_steps=0）时零 batch、零 step、零额外保存。
                if self._sync_max_steps_reached():
                    if self.rank == 0:
                        logger.info(
                            f"max_steps={self.max_steps} already reached; no training run."
                        )
                    break
                epochs_ran += 1
                self.current_epoch = epoch  # expose to signal handlers
                if self.world_size > 1:
                    self._nccl_flow_log("barrier_epoch_start")
                    dist.barrier()
                if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
                    self.train_sampler.set_epoch(epoch)
                # Set epoch for dataset to enable deterministic augmentation
                if self.train_dataset is not None and hasattr(self.train_dataset, "set_epoch"):
                    base_seed = self.config.get("meta", {}).get("seed", 42)
                    self.train_dataset.set_epoch(epoch, base_seed=base_seed)

                logger.info(f"===== Epoch {epoch}/{self.epochs} =====")
                _epoch_t0 = time.monotonic()
                for m in self.models.values():
                    m.train()
                step_callback = None
                if self.validation_steps_interval > 0 and self.save_top_k > 0:
                    step_callback = self._maybe_validate_and_save_step_best
                try:
                    train_loss = self._train_epoch(epoch, step_callback=step_callback)
                except torch.cuda.OutOfMemoryError:
                    # OOM 恢复路径：清 cache 后向上抛（重试同一 batch 会再次 OOM，无意义）
                    self._clear_cuda_cache(f"OOM recovery (train epoch {epoch})")
                    raise
                if self._sync_max_steps_reached() and not getattr(
                    self, "_train_epoch_completed", True
                ):
                    batch_cursor = int(
                        getattr(self, "_last_completed_batch_cursor", 0)
                    )
                    if self.rank == 0:
                        logger.info(
                            f"Reached max_steps={self.max_steps} at epoch {epoch} "
                            f"batch cursor {batch_cursor}; exiting training."
                        )
                    # 未完成 epoch 不推进 epoch scheduler/validation；checkpoint 携带
                    # accumulation 边界 cursor，恢复后继续同一 epoch。
                    self._save_checkpoint(epoch, batch_cursor=batch_cursor)
                    last_saved_epoch = epoch
                    break
                # 信号停机：未完成训练 epoch 时按 cursor 保存并退出；若最后一个
                # train batch 已完整提交，则继续完成本 epoch 的 validation、scheduler
                # 与 checkpoint finalization，避免恢复时跳过尚未执行的结算阶段。
                shutdown_after_train = self._sync_shutdown_signum()
                if shutdown_after_train and not getattr(
                    self, "_train_epoch_completed", True
                ):
                    batch_cursor = int(
                        getattr(self, "_last_completed_batch_cursor", 0)
                    )
                    if self.rank == 0:
                        logger.info(
                            f"Shutdown requested at epoch {epoch} batch cursor "
                            f"{batch_cursor}; saving checkpoint and exiting."
                        )
                    self._save_checkpoint(epoch, batch_cursor=batch_cursor)
                    last_saved_epoch = epoch
                    break
                if shutdown_after_train and self.rank == 0:
                    logger.info(
                        "Shutdown requested after the final train batch; "
                        "finishing epoch validation/scheduler before exit."
                    )
                _train_time = time.monotonic() - _epoch_t0
                if self.rank == 0:
                    self._log_scalar("system/train_epoch_time_sec", _train_time, epoch)
                    logger.info(f"  Train: {_train_time:.1f}s")
                if torch.cuda.is_available():
                    _alloc = torch.cuda.memory_allocated() / 1024**3
                    _resv = torch.cuda.memory_reserved() / 1024**3
                    _peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
                    _peak_resv = torch.cuda.max_memory_reserved() / 1024**3
                    if self.rank == 0:
                        self._log_scalar("system/gpu_mem_allocated_gb", _alloc, epoch)
                        self._log_scalar("system/gpu_mem_reserved_gb", _resv, epoch)
                        self._log_scalar("system/gpu_mem_peak_allocated_gb", _peak_alloc, epoch)
                        self._log_scalar("system/gpu_mem_peak_reserved_gb", _peak_resv, epoch)
                        logger.info(
                            f"  GPU memory: allocated={_alloc:.2f}GB, reserved={_resv:.2f}GB "
                            f"(peak: allocated={_peak_alloc:.2f}GB, reserved={_peak_resv:.2f}GB)"
                        )
                    torch.cuda.reset_peak_memory_stats()
                    if self.clear_cache_each_epoch:
                        self._clear_cuda_cache(f"epoch {epoch} end")

                did_validate = False
                validation_valid = False
                is_best = False
                should_stop = False
                _val_time = 0.0
                val_loss: Optional[float] = None
                if self.validation_epoch_interval > 0 and epoch % self.validation_epoch_interval == 0:
                    if self.clear_cache_each_epoch:
                        self._clear_cuda_cache(f"pre-validation epoch {epoch}")
                    did_validate = True
                    for m in self.models.values():
                        m.eval()
                    _val_t0 = time.monotonic()
                    try:
                        val_loss = self._validation_epoch(epoch)
                    except torch.cuda.OutOfMemoryError:
                        # OOM 恢复路径：清 cache 后向上抛
                        self._clear_cuda_cache(f"OOM recovery (validation epoch {epoch})")
                        raise
                    _val_time = time.monotonic() - _val_t0
                    if self.rank == 0:
                        self._log_scalar("system/val_epoch_time_sec", _val_time, epoch)
                        logger.info(f"  Val: {_val_time:.1f}s")
                    validation_valid = self._is_valid_validation_result(val_loss)
                    if not validation_valid:
                        if self.rank == 0:
                            logger.warning(
                                "Skipping epoch %s validation result because it is not a "
                                "finite real scalar: %r",
                                epoch,
                                val_loss,
                            )
                    else:
                        val_loss = float(val_loss)
                        if train_loss is not None:
                            self.train_losses.append(train_loss)
                            self.val_losses.append(val_loss)
                            self.curve_epochs.append(epoch)
                            self._on_loss_curve_point_added(epoch, train_loss, val_loss)
                        if self.rank == 0 and self._should_save_curve(len(self.curve_epochs)):
                            self._save_loss_curve()
                        # Epoch-level best: always persist best_model.tar when this validation improves,
                        # so downstream inference/export docs remain valid regardless of step-best (save_top_k).
                        if val_loss < self.best_score:
                            self.best_score = val_loss
                            self.best_epoch = epoch
                            self.best_at_step = None  # epoch-level best marker on loss curve
                            self._patience_counter = 0
                            is_best = True
                        elif self.patience > 0:
                            self._patience_counter += 1
                            if self.rank == 0:
                                logger.info(
                                    f"  No improvement ({self._patience_counter}/{self.patience})"
                                )
                            if self._patience_counter >= self.patience:
                                if self.rank == 0:
                                    logger.info(
                                        f"Early stopping: no improvement for "
                                        f"{self.patience} validations"
                                    )
                                should_stop = True

                if not self.lr_per_step:
                    if self._is_multi_scheduler:
                        for name, sch in self.schedulers.items():
                            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                if did_validate and validation_valid:
                                    sch.step(val_loss)
                            else:
                                sch.step()
                    elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if did_validate and validation_valid:
                            self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                if (
                    is_best
                    or (
                        self.save_interval > 0
                        and epoch % self.save_interval == 0
                        and epoch != last_saved_epoch
                    )
                ):
                    self._save_checkpoint(epoch, is_best=is_best)
                    last_saved_epoch = epoch

                # 信号可能在 validation/metrics 期间到达；scheduler 与 epoch
                # checkpoint 已提交后，再用固定边界 collective 传播并统一退出。
                if self._sync_shutdown_signum():
                    if self.rank == 0:
                        logger.info(
                            f"Shutdown requested after epoch {epoch} validation; exiting."
                        )
                    break
                if should_stop:
                    break

            # 兜底保存最终状态：覆盖 epochs % save_interval != 0 的正常结束、
            # early-stop 与 max_steps 退出路径；若该 epoch 已在循环内保存过则跳过，
            # 避免重复写盘。epochs_ran == 0（start_epoch > epochs 的重复启动）不保存。
            if epochs_ran > 0 and self.current_epoch > last_saved_epoch:
                self._save_checkpoint(self.current_epoch)
        finally:
            # ── Explicitly shut down DataLoader workers before Python exit ──
            # PyTorch _MultiProcessingDataLoaderIter._shutdown_workers() is a no-op
            # when python_exit_status is True/None (i.e. during interpreter shutdown),
            # so persistent_workers=True processes survive and the main process hangs
            # waiting for them.  We must shut them down while Python is still alive.
            # finally 保证 _train_epoch 抛异常（OOM 重抛、strict_checks、数据错误）
            # 时 persistent workers 也会被回收。
            try:
                for _loader_attr in ('train_loader', 'val_loader'):
                    _dl = getattr(self, _loader_attr, None)
                    if _dl is not None and _dl._iterator is not None:
                        _dl._iterator._shutdown_workers()
                        _dl._iterator = None
                # Detach attributes so GC can reclaim DataLoader + worker pool cleanly
                self.train_loader = None
                self.val_loader = None
            except Exception:
                # 清理异常不得掩盖训练主异常（review P3-8）
                logger.exception("DataLoader worker shutdown failed during train() cleanup")
        if self.rank == 0:
            logger.info("Training complete.")

    def _train_epoch(self, epoch: int, step_callback: Optional[Callable[[int, int, int], None]] = None) -> Optional[float]:
        """Run one training epoch. Returns average train loss or None.

        step_callback(epoch, step, batch_cursor) runs after each optimizer step;
        batch_cursor counts all completed batches, including a committed tail.
        """
        raise NotImplementedError

    def _validation_epoch(self, epoch: int) -> Optional[float]:
        raise NotImplementedError
