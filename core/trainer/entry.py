#!/usr/bin/env python3
"""Unified training entry point base class, shared by SE, AEC, TSE and SS tasks.

Task-specific concrete entries (``tasks.se.trainer.entry.SETrainingEntry``,
``tasks.aec.trainer.entry.AECTrainingEntry``, ``tasks.tse.trainer.entry.TSETrainingEntry``,
``tasks.ss.trainer.entry.SSTrainingEntry``)
subclass ``UnifiedTrainingEntry`` here and implement
``build_loaders``/``build_model``/``get_trainer_cls`` (and,
for GAN-capable tasks, ``_default_gan_trainer_cls``).
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.utils.python_version import check_python_version
from core.data.constants import normalize_data_mode
from tasks.registry import get_task_for_data_mode

from core.trainer.validation_metrics_config import (
    validate_no_trainer_mos_config,
    validate_trainer_config,
)

import functools
import logging
import os
import random
import signal
import sys
import warnings
from pathlib import Path

# Suppress torchaudio C++ "vorbis/opus -strict experimental" warnings
# before torch/torchaudio load. Set TORCH_CPP_LOG_LEVEL yourself to override.
if "TORCH_CPP_LOG_LEVEL" not in os.environ:
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# Suppress common nuisance warnings from third-party libraries
warnings.filterwarnings("ignore", message=".*Empty filters detected in mel frequency basis.*")
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Specified provider.*is not in available provider.*")
# PyTorch 2.4 internal gradient-checkpointing deprecation (fixed in 2.5+)
warnings.filterwarnings("ignore", message=r".*torch\.cpu\.amp\.autocast.*is deprecated.*", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def set_spawn_start_method() -> None:
    """Set multiprocessing start method to 'spawn' for DataLoader compatibility.

    Safe to call inside ``if __name__ == '__main__':`` guards in training entry
    scripts. Ignores RuntimeError if the method has already been set.
    """
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


_FILE_DESCRIPTOR_LIMIT_LOGGED = False


def _ensure_file_descriptor_limit(min_fd: int = 4096, rank: int | None = None) -> None:
    """Bump ``ulimit -n`` if too low; warn if hard limit prevents safe value.

    The limit itself is per-process, so it is raised on every rank.  The
    informational log is emitted once per process from rank 0 to avoid the
    noisy flood of identical lines in DDP spawn launches.
    """
    global _FILE_DESCRIPTOR_LIMIT_LOGGED
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < min_fd:
            target = min(min_fd, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            if rank is None or rank == 0:
                logger = logging.getLogger(__name__)
                if not _FILE_DESCRIPTOR_LIMIT_LOGGED:
                    logger.info(
                        f"RLIMIT_NOFILE raised: {soft} → {target} "
                        f"(hard limit: {hard}).  Was too low for DataLoader workers."
                    )
                    _FILE_DESCRIPTOR_LIMIT_LOGGED = True
                else:
                    logger.debug(
                        "RLIMIT_NOFILE already adjusted in this process; current soft=%s",
                        target,
                    )
    except Exception as exc:
        # 提升失败必须留痕：DataLoader worker 的 "Too many open files" 依赖此线索定位。
        if rank is None or rank == 0:
            logging.getLogger(__name__).warning(
                "RLIMIT_NOFILE could not be raised (%s: %s); DataLoader workers "
                "may hit 'Too many open files'",
                type(exc).__name__,
                exc,
            )


def optimizer_steps_per_epoch(
    n_train_batches: int, accumulation_steps: int, *, world_size: int
) -> int:
    """按 trainer 的单卡 tail-step / DDP tail-drop 合同估算每 epoch step 数。"""
    batches = max(0, int(n_train_batches))
    accumulation = max(1, int(accumulation_steps))
    if world_size > 1:
        return batches // accumulation
    return (batches + accumulation - 1) // accumulation


# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskInfo:
    """Detected task metadata for logging and routing."""

    task: str           # "se" | "aec" | "tse"
    subtask: str        # "dynamic" | "offline" | "joint" | "aec_only" | "dynamic_tse"
    paradigm: str       # "discriminative" | "gan" | "diffusion"
    has_laec: bool      # True if linear_aec frontend is configured
    model_type: str
    logger_name: str
    channels: str = "mono"   # "mono" | "spatial (C=N, <profile>)"，启动报告展示


_RESUME_SOURCE_CONTEXT_UNSET = object()


@dataclass(frozen=True)
class ResumePreparation:
    """scheduler 与 trainer 构造之间显式传递的 resume 双路径结果。"""

    start_epoch: int
    source_path: Path | None
    snapshot_path: Path | None

    @classmethod
    def normalize(
        cls,
        value,
        *,
        source_path=_RESUME_SOURCE_CONTEXT_UNSET,
    ) -> "ResumePreparation":
        """校验并规范化 scheduler 返回的 ``ResumePreparation``。"""
        if not isinstance(value, cls):
            raise TypeError("resume preparation must be a ResumePreparation")
        context_provided = source_path is not _RESUME_SOURCE_CONTEXT_UNSET
        context_source = (
            cls._normalize_path(source_path, "source_path")
            if context_provided
            else None
        )
        start_epoch = value.start_epoch
        result_source = cls._normalize_path(value.source_path, "source_path")
        snapshot_path = cls._normalize_path(value.snapshot_path, "snapshot_path")

        if type(start_epoch) is not int:
            raise TypeError("ResumePreparation start_epoch must be an exact int")
        if start_epoch <= 0:
            raise ValueError("ResumePreparation start_epoch must be positive")
        if context_provided and result_source != context_source:
            raise ValueError(
                "ResumePreparation source_path differs from frozen source context"
            )
        if (result_source is None) != (snapshot_path is None):
            raise ValueError(
                "ResumePreparation source_path and snapshot_path must both be set or both be None"
            )
        normalized = cls(start_epoch, result_source, snapshot_path)
        return value if value == normalized else normalized

    @staticmethod
    def _normalize_path(value, field: str) -> Path | None:
        if value is None:
            return None
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError(f"ResumePreparation {field} must be path-like or None")
        return Path(value)


def _dig_mapping(mapping, *path):
    """安全取嵌套 dict 值；任一层缺失或非 dict 返回 None。"""
    node = mapping
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _detect_channels(data_cfg: dict, model_cfg: dict) -> str:
    """从 train split 仿真配置（分层/平面）与模型构造参数推断通道模式。

    仅用于启动报告展示；取值口径与 dataset 构造一致——spatial 判定以
    ``simulation_cfg`` 的 spatial 开关为准，C 与阵列 profile 仅作展示。
    """
    train_cfg = data_cfg.get("train") or {}
    sim_cfg = train_cfg.get("simulation_cfg")
    spatial_enabled = train_cfg.get("spatial_enabled") is True
    profile = train_cfg.get("array_profile")
    if isinstance(sim_cfg, dict):
        nested = sim_cfg.get("spatial")
        if isinstance(nested, dict):
            spatial_enabled = spatial_enabled or nested.get("enabled") is True
            geometry = nested.get("geometry") or {}
            profile = profile or geometry.get("profile")
        spatial_enabled = spatial_enabled or sim_cfg.get("spatial_enabled") is True
        profile = profile or sim_cfg.get("array_profile")
    if not spatial_enabled:
        return "mono"
    model_args = model_cfg.get("args") or {}
    n_channel = model_args.get("n_channel") if isinstance(model_args, dict) else None
    details = []
    if isinstance(n_channel, int) and n_channel > 0:
        details.append(f"C={n_channel}")
    if isinstance(profile, str) and profile:
        details.append(profile)
    return "spatial (" + ", ".join(details) + ")" if details else "spatial"


def _spatial_background_gate_nominals(data_cfg: dict) -> list[str]:
    """spatial 背景三门的 nominal 配置值（point_source / wind_background / colored）。

    返回可读的 ``name=percent`` 列表；非 spatial 配置返回空列表。
    """
    train_cfg = data_cfg.get("train") or {}
    sim_cfg = train_cfg.get("simulation_cfg")
    spatial_enabled = train_cfg.get("spatial_enabled") is True
    point_source = wind_background = colored = None
    if isinstance(sim_cfg, dict):
        nested_spatial = sim_cfg.get("spatial")
        if isinstance(nested_spatial, dict):
            spatial_enabled = spatial_enabled or nested_spatial.get("enabled") is True
            background = nested_spatial.get("background") or {}
            point_source = background.get("point_source_prob")
            wind_background = background.get("wind_background_prob")
        spatial_enabled = spatial_enabled or sim_cfg.get("spatial_enabled") is True
        point_source = (
            point_source
            if point_source is not None
            else sim_cfg.get("prob_point_source_background")
        )
        wind_background = (
            wind_background
            if wind_background is not None
            else sim_cfg.get("prob_wind_noise_background")
        )
        colored = _dig_mapping(sim_cfg, "acoustic", "background", "colored_noise_prob")
        if colored is None:
            colored = _dig_mapping(sim_cfg, "background", "colored_noise_prob")
        if colored is None:
            colored = sim_cfg.get("prob_colored_noise")
    if not spatial_enabled:
        return []
    gates = []
    for name, value in (
        ("point_source", point_source),
        ("wind_background", wind_background),
        ("colored", colored),
    ):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            gates.append(f"{name}={float(value) * 100.0:.1f}%")
    return gates


def _current_dist_rank() -> int:
    """返回当前 distributed rank；未初始化或非分布式时返回 0。"""
    try:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def detect_task(cfg: dict) -> TaskInfo:
    """Auto-detect task type, paradigm, and subtask from config dict."""
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        raise TypeError("Config 'data' section must be a mapping")
    data_mode = normalize_data_mode(data_cfg.get("mode")).task_mode
    model_cfg = cfg.get("model", {})
    trainer_cfg = cfg.get("trainer", {})

    # Detect paradigm
    paradigm = "discriminative"
    if "generator" in model_cfg or "discriminator" in model_cfg:
        paradigm = "gan"
    elif trainer_cfg.get("type") == "GANTrainer":
        paradigm = "gan"

    # 任务包由 registry 解析；AEC 前端保持任务专属语义。
    descriptor = get_task_for_data_mode(data_mode)
    task = descriptor.name
    subtask = data_cfg.get("task", "joint") if task == "aec" else data_mode
    has_laec = False
    if task == "aec":
        from tasks.aec.inference.aec_frontend import resolve_aec_frontend_config

        has_laec = resolve_aec_frontend_config(cfg).linear_aec is not None
    selected_model = model_cfg.get("generator", model_cfg) if paradigm == "gan" else model_cfg
    model_type = selected_model.get("type") or selected_model.get("path", "unknown")
    channels = _detect_channels(data_cfg, model_cfg)
    return TaskInfo(task, subtask, paradigm, has_laec, model_type, f"{task.upper()}_Train", channels)


class UnifiedTrainingEntry(ABC):
    """Base class for SE/AEC unified training entry."""

    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
        self._shutdown_requested = False

    # ------------------------------------------------------------------
    # CLI / bootstrap
    # ------------------------------------------------------------------
    def parse_args(self) -> argparse.Namespace:
        p = argparse.ArgumentParser()
        p.add_argument("-C", "--config", required=True)
        p.add_argument("--resume", action="store_true")
        p.add_argument(
            "--resume-from",
            dest="resume_from",
            default=None,
            metavar="PATH",
            help="Resume from an explicit checkpoint file (full training state: "
                 "latest_model.tar or middle checkpoint model_E*_B*.pth). Implies --resume.",
        )
        p.add_argument(
            "--checkpoint",
            default=None,
            help="Pretrained checkpoint path (model only, no optimizer/scheduler)",
        )
        p.add_argument("--wandb", action="store_true")
        p.add_argument(
            "--strict",
            action="store_true",
            help="Enable strict checks for trainer/loss/val_metrics (simulation_cfg unknown keys always fail fast).",
        )
        p.add_argument(
            "--max_steps",
            type=int,
            default=None,
            help="If set, stop after this many optimizer steps (for quick verification).",
        )
        p.add_argument(
            "--exp_path",
            type=str,
            default=None,
            help="Override trainer.exp_path (e.g. /tmp/verify to avoid saving outputs).",
        )
        p.add_argument(
            "--log_level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Console logging level (default: INFO).",
        )
        p.add_argument(
            "--log_file",
            type=str,
            default=None,
            help="Optional log file path. If set, logs are also written to this file.",
        )
        return p.parse_args()

    def _log_task_info(
        self, task_info: TaskInfo, cfg: dict | None = None, config_path=None
    ) -> None:
        """Print explicit task detection report at training startup."""
        self.logger.info("=" * 60)
        self.logger.info("Task Detection Report")
        self.logger.info("=" * 60)
        self.logger.info("  Task Type      : %s", task_info.task.upper())
        self.logger.info("  Subtask        : %s", task_info.subtask)
        self.logger.info("  Paradigm       : %s", task_info.paradigm)
        self.logger.info("  Model          : %s", task_info.model_type)
        self.logger.info("  Channels       : %s", task_info.channels)
        if task_info.task == "aec":
            self.logger.info(
                "  LAEC Frontend  : %s",
                "enabled" if task_info.has_laec else "disabled",
            )
        if cfg is not None:
            digest = hashlib.sha256(
                json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:12]
            meta_cfg = cfg.get("meta") or {}
            run_bits = [f"config_digest={digest}"]
            if config_path:
                run_bits.append(f"config={Path(str(config_path)).name}")
            exp_path = (cfg.get("trainer") or {}).get("exp_path")
            if exp_path:
                run_bits.append(f"exp_path={exp_path}")
            run_bits.append(f"seed={meta_cfg.get('seed', 42)}")
            run_bits.append(f"amp={'on' if meta_cfg.get('use_amp') else 'off'}")
            run_bits.append(
                f"deterministic={meta_cfg.get('deterministic', False)}"
            )
            self.logger.info("  Run Context    : %s", " | ".join(run_bits))
        self.logger.info("=" * 60)

    def _log_exception_rank_aware(self, exc: BaseException, context: str) -> None:
        """DDP 下只由 rank 0 打完整 traceback，其余 rank 打一行摘要。

        8 rank 同时失败时避免日志被 8 份相同堆叠淹没；排错以 rank 0 为准。
        调用方负责在异常对象上标记 ``_entry_traceback_logged`` 以防外层重复打。
        """
        rank = _current_dist_rank()
        if rank == 0:
            self.logger.exception("%s: %s", context, exc)
        else:
            self.logger.error(
                "rank %d: %s: %s: %s", rank, context, type(exc).__name__, exc
            )

    def _log_data_statistics(
        self,
        task_info: TaskInfo,
        train_ds,
        val_ds,
        n_train_batches: int,
        n_val_batches: int,
        cfg: dict,
    ) -> None:
        """Print training/validation data distribution before the first epoch."""
        if not callable(getattr(train_ds, "get_data_summary", None)):
            return

        dl_cfg = cfg.get("data", {}).get("dataloader", {})
        splits = [("train", train_ds, n_train_batches)]
        if val_ds is not None and callable(getattr(val_ds, "get_data_summary", None)):
            splits.append(("val", val_ds, n_val_batches))

        self.logger.info("=" * 70)
        self.logger.info("Data Distribution Summary")
        self.logger.info("=" * 70)
        self.logger.info(
            "Task=%s | Subtask=%s | Paradigm=%s | Model=%s",
            task_info.task.upper(),
            task_info.subtask,
            task_info.paradigm,
            task_info.model_type,
        )

        for split, ds, n_batches in splits:
            summary = ds.get_data_summary(split, n_batches, dl_cfg)
            self.logger.info(
                "[%s] samples=%d, batches=%d",
                split.upper(),
                summary.n_samples,
                summary.n_batches,
            )
            self.logger.info(
                "  %-10s %8s %10s %8s %8s %8s %s",
                "Source", "Clips", "Hours", "Min(h)", "Max(h)", "Mean(h)", "SR counts",
            )
            for name, src in summary.sources.items():
                hours = src.total_hours
                hours_str = f"{hours:.2f}" if hours is not None else "n/a"
                min_h = src.min_hours
                max_h = src.max_hours
                mean_h = src.mean_hours
                min_str = f"{min_h:.4f}" if min_h is not None else "n/a"
                max_str = f"{max_h:.4f}" if max_h is not None else "n/a"
                mean_str = f"{mean_h:.4f}" if mean_h is not None else "n/a"
                sr_str = ", ".join(
                    f"{sr}:{cnt}" for sr, cnt in sorted(src.sr_counts.items())
                )
                self.logger.info(
                    "  %-10s %8d %10s %8s %8s %8s %s",
                    name, src.clips, hours_str, min_str, max_str, mean_str, sr_str,
                )

            if summary.noise_categories:
                self.logger.info("  Top noise categories:")
                for name, count, ratio in summary.noise_categories:
                    self.logger.info(
                        "    %-30s %6d (%5.2f%%)", name, count, ratio * 100.0
                    )

            if summary.scenario_probs:
                self.logger.info("  Scenario probabilities:")
                for name, prob in summary.scenario_probs.items():
                    self.logger.info("    %-30s %5.2f%%", name, prob * 100.0)

            if summary.extra:
                self.logger.info("  Extra: %s", summary.extra)

        gate_nominals = _spatial_background_gate_nominals(cfg.get("data") or {})
        if gate_nominals:
            self.logger.info(
                "Spatial background gates (nominal config values): %s",
                " | ".join(gate_nominals),
            )

        self.logger.info(
            "DataLoader config: %s", ", ".join(f"{k}={v}" for k, v in dl_cfg.items())
        )
        self.logger.info("=" * 70)

    def _ensure_free_master_port(self) -> None:
        """If MASTER_PORT is unset, find and set a free port."""
        import socket

        requested = os.environ.get("MASTER_PORT")
        if requested:
            return
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            free_port = s.getsockname()[1]
        os.environ["MASTER_PORT"] = str(free_port)

    def seed_everything(self, seed: int, rank: int = 0, deterministic: bool = False) -> None:
        """Set global random seed.

        Model weights are initialized with the same seed on all DDP ranks
        for consistency. Data shuffling uses a per-rank offset for diversity.
        """
        # Same seed for model init across all ranks
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Per-rank offset for data augmentation / shuffling
        s = seed + rank
        random.seed(s)
        np.random.seed(s)
        torch.backends.cudnn.deterministic = deterministic

    def _validate_config(self, cfg: dict) -> None:
        """Fail-fast validation of required config keys."""
        required_top = ["data", "model", "optimizer", "scheduler", "trainer"]
        for k in required_top:
            if k not in cfg:
                raise KeyError(f"Config missing required top-level key: '{k}'")

        validate_no_trainer_mos_config(cfg)
        validate_trainer_config(cfg)

        data_cfg = cfg["data"]
        if not isinstance(data_cfg, dict):
            raise TypeError("Config 'data' section must be a mapping")
        # 训练 batch 合同目前没有逐样本有效性 mask。strict_load=False 会把
        # 可恢复的读盘/仿真错误转换为静音 placeholder，若继续进入 Trainer，
        # placeholder 会参与 loss 与样本计数。宽松加载仍可供诊断/推理 Dataset
        # 使用，但统一训练入口必须在配置边界 fail-fast。
        from core.data.config_flags import require_strict_training_load

        require_strict_training_load(cfg)
        # 方案 B 的分层 simulation_cfg 在公共入口归一化；任务 entry 后续的
        # capability 检查只能看到同一份 canonical flat mapping。
        from core.data.simulation_schema import normalize_config_simulation_sections

        normalize_config_simulation_sections(cfg)
        normalize_data_mode(data_cfg.get("mode"))
        if "dataloader" not in data_cfg:
            raise KeyError("Config 'data' section must contain 'dataloader'")
        dl_cfg = data_cfg["dataloader"]
        if "batch_size" not in dl_cfg:
            raise KeyError("Config 'data.dataloader' section must contain 'batch_size'")
        if "num_workers" not in dl_cfg:
            raise KeyError("Config 'data.dataloader' section must contain 'num_workers'")

        model_cfg = cfg["model"]
        # GAN configs use dict form with 'generator' / 'discriminator' keys
        is_gan_model = isinstance(model_cfg, dict) and "generator" in model_cfg
        if not is_gan_model and "type" not in model_cfg and "path" not in model_cfg:
            raise KeyError("Config 'model' section must contain 'type' or 'path'")

        # Optimizer may be dict (GAN) or single config
        opt_cfg = cfg["optimizer"]
        _is_multi_opt = isinstance(opt_cfg, dict) and any(k in opt_cfg for k in ("generator", "discriminator"))
        if _is_multi_opt:
            for name, sub in opt_cfg.items():
                if isinstance(sub, dict) and "type" not in sub:
                    raise KeyError(f"Config 'optimizer.{name}' section must contain 'type'")
        elif "type" not in opt_cfg:
            raise KeyError("Config 'optimizer' section must contain 'type'")

        # Scheduler may be dict (GAN) or single config
        sch_cfg = cfg["scheduler"]
        _is_multi_sch = isinstance(sch_cfg, dict) and any(k in sch_cfg for k in ("generator", "discriminator"))
        if _is_multi_sch:
            for name, sub in sch_cfg.items():
                if isinstance(sub, dict) and "type" not in sub:
                    raise KeyError(f"Config 'scheduler.{name}' section must contain 'type'")
        elif "type" not in sch_cfg:
            raise KeyError("Config 'scheduler' section must contain 'type'")

        if "type" not in (cfg.get("loss") or {}):
            raise KeyError("Config 'loss' section must contain 'type'")
        if not cfg.get("meta", {}).get("allow_no_stft", False) and not cfg.get("stft"):
            raise KeyError(
                "Config must contain top-level 'stft' (n_fft, hop_length, win_length) "
                "so training uses framework STFT → model → iSTFT. "
                "Set meta.allow_no_stft: true for waveform-native models that do not "
                "consume the framework STFT."
            )

    def _apply_strict_mode(self, cfg: dict, enabled: bool, rank: int) -> None:
        """Inject strict runtime flags into config dict."""
        if not enabled:
            return
        trainer = cfg.setdefault("trainer", {})
        trainer["strict_checks"] = True
        val_metrics = trainer.setdefault("val_metrics", {})
        val_metrics["strict"] = True
        loss_cfg = cfg.setdefault("loss", {})
        loss_cfg["strict"] = True
        if rank == 0:
            self.logger.warning(
                "Strict mode enabled: fail-fast checks for data/loss/metrics are "
                "active (simulation_cfg unknown keys always fail fast)."
            )

    # ------------------------------------------------------------------
    # Load & setup (ordering must be preserved exactly)
    # ------------------------------------------------------------------
    def load_and_setup(
        self,
        *,
        config: dict | None = None,
        config_path: str | Path | None = None,
    ):
        """执行 bootstrap，并可消费路由阶段已加载的同一配置对象。

        未传入 ``config`` / ``config_path`` 时保持独立 CLI 启动行为。传入时两者
        必须同时提供，且 ``config_path`` 必须与 CLI ``--config`` 指向同一文件。

        Returns (args, cfg, rank, local_rank, world_size).

        Ordering: parse_args → load config → validate → DDP init → logging →
        strict mode → max_steps/exp_path override → seed.
        """
        from core.trainer.entry_components import bootstrap_entry

        return bootstrap_entry(self, config=config, config_path=config_path)

    # ------------------------------------------------------------------
    # Builder helpers (shared)
    # ------------------------------------------------------------------
    def _build_single_optimizer(self, opt_cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Build a single optimizer for a single model."""
        from core.trainer.entry_components import build_single_optimizer

        return build_single_optimizer(opt_cfg, model)

    def build_optimizer(self, cfg: dict, model) -> torch.optim.Optimizer | dict[str, torch.optim.Optimizer]:
        """Build optimizer(s). Supports single model or dict of models (e.g. GAN)."""
        opt_cfg = cfg.get("optimizer", {})
        if isinstance(model, dict):
            # Multi-optimizer for multi-model training
            result = {}
            for name, m in model.items():
                sub_cfg = opt_cfg.get(name, opt_cfg)
                result[name] = self._build_single_optimizer(sub_cfg, m)
            return result
        return self._build_single_optimizer(opt_cfg, model)

    def _build_single_scheduler(
        self, sch_cfg: dict, n_train_batches: int, optimizer: torch.optim.Optimizer,
        cfg: dict, *, world_size: int, scheduler_name: str = "default",
    ):
        """Build a single scheduler for a single optimizer."""
        from core.trainer.entry_components import build_single_scheduler

        return build_single_scheduler(
            self,
            sch_cfg,
            n_train_batches,
            optimizer,
            cfg,
            world_size=world_size,
            scheduler_name=scheduler_name,
        )

    def build_scheduler(
        self, cfg: dict, n_train_batches: int, optimizer, *, world_size: int
    ):
        """Build scheduler(s). Supports single optimizer or dict of optimizers (e.g. GAN).

        ``world_size`` 是必填 keyword 参数，由 ``run_training`` 传入；override 也
        必须接受它，否则调用点会直接报错。
        """
        sch_cfg = cfg.get("scheduler", {})
        if isinstance(optimizer, dict):
            # Multi-scheduler for multi-optimizer training
            result = {}
            for name, opt in optimizer.items():
                sub_cfg = sch_cfg.get(name, sch_cfg) if isinstance(sch_cfg, dict) and name in sch_cfg else sch_cfg
                result[name] = self._build_single_scheduler(
                    sub_cfg,
                    n_train_batches,
                    opt,
                    cfg,
                    world_size=world_size,
                    scheduler_name=name,
                )
            return result
        return self._build_single_scheduler(
            sch_cfg,
            n_train_batches,
            optimizer,
            cfg,
            world_size=world_size,
            scheduler_name="default",
        )

    def build_stft_module(self, cfg: dict, rank: int):
        from core.trainer.entry_components import build_stft_module

        return build_stft_module(self, cfg, rank)

    @staticmethod
    def _validate_num_per_epoch(value, field: str) -> None:
        """校验 shared sampler 的可选虚拟 epoch 样本数。"""
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(
                f"{field} must be None or an exact positive int, got {value!r}"
            )

    def _build_loaders_common(
        self,
        cfg: dict,
        rank: int,
        local_rank: int,
        world_size: int,
        *,
        train_ds,
        val_ds,
        train_collate_fn,
        val_collate_fn=None,
        train_num_per_epoch=None,
        val_num_per_epoch=None,
    ):
        """SE/AEC/TSE 三个 entry 共用的 DataLoader 装配（收敛三处重复逻辑）。

        任务差异（dataset 构建、collate 绑定、前置 cfg 解析、val_ds.set_epoch 等）
        由子类在 ``build_loaders`` 中完成，本方法统一处理：
        GroupedBatchSampler（含 DDP 0 样本 / val 过小检查）、DataLoader 参数与
        persistent workers 策略、train/val 固定 generator（跨 run 种子一致）、
        batch 数日志。

        Args:
            train_collate_fn / val_collate_fn: 各任务 collate（AEC/TSE 通常绑定
                placeholder 参数的 partial；val 缺省复用 train 的 collate）。

        Returns:
            (train_loader, val_loader, train_sampler, train_ds)，与各 entry
            build_loaders 的返回契约一致。
        """
        from core.trainer.entry_components import build_loaders_common
        from core.models.io_contract import validate_split_microphone_contract

        validate_split_microphone_contract(train_ds, val_ds)
        return build_loaders_common(
            self,
            cfg,
            rank,
            local_rank,
            world_size,
            train_ds=train_ds,
            val_ds=val_ds,
            train_collate_fn=train_collate_fn,
            val_collate_fn=val_collate_fn,
            train_num_per_epoch=train_num_per_epoch,
            val_num_per_epoch=val_num_per_epoch,
        )

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------
    def _graceful_shutdown(self, signum, frame, trainer, rank, world_size):
        if self._shutdown_requested:
            # 第二次信号：主流程尚未消费停机请求，立即强制退出，避免被
            # 卡死的主流程无限拖延。
            os._exit(128 + signum)
        # 第一次信号严格 flag-only：checkpoint、日志、process group 销毁和
        # sys.exit 都由训练主流程在安全边界执行。异步信号上下文调用
        # CUDA/NCCL/文件 IO 或 logger 都可能在持锁状态下死锁。
        self._shutdown_requested = True
        request = getattr(trainer, "request_shutdown", None)
        if callable(request):
            request(signum)

    def setup_signal_handlers(self, trainer, rank, world_size):
        self._shutdown_requested = False
        signal.signal(
            signal.SIGTERM,
            lambda s, f: self._graceful_shutdown(s, f, trainer, rank, world_size),
        )
        if hasattr(signal, "SIGINT"):
            signal.signal(
                signal.SIGINT,
                lambda s, f: self._graceful_shutdown(s, f, trainer, rank, world_size),
            )

    # ------------------------------------------------------------------
    # Core training flow
    # ------------------------------------------------------------------
    @staticmethod
    def _resume_checkpoint_path(cfg: dict, args: argparse.Namespace) -> Path | None:
        """按 BaseTrainer 的优先级解析本次 resume 实际使用的 checkpoint。

        与 ``BaseTrainer._resume_checkpoint`` 的差异是刻意的：显式 ``--resume-from``
        指向不存在的文件时这里直接返回 ``None``，由 ``_resolve_resume_source_context``
        fail-fast（各 rank 一致报错），而 BaseTrainer 只在直接构造时才会 warn 后回退
        自动发现；训练入口始终向 trainer 传严格 ``resume_load_from``。
        """
        from core.trainer.base_trainer import BaseTrainer

        explicit = getattr(args, "resume_from", None)
        if explicit:
            path = Path(explicit)
            return path if path.exists() else None

        ckpt_base = (
            Path(cfg.get("trainer", {}).get("exp_path", "exp/default"))
            / "checkpoints"
        )
        candidates = []
        if ckpt_base.exists():
            for directory in sorted(ckpt_base.iterdir()):
                if not directory.is_dir():
                    continue
                if (
                    (directory / "latest_model.tar").exists()
                    or BaseTrainer._find_latest_middle_checkpoint(directory)
                    is not None
                ):
                    candidates.append(directory)
        if not candidates:
            return None
        checkpoint_dir = candidates[-1]

        latest = checkpoint_dir / "latest_model.tar"
        if latest.exists():
            return latest
        latest_candidates = sorted(checkpoint_dir.parent.glob("*/latest_model.tar"))
        if latest_candidates:
            return latest_candidates[-1]
        middle = BaseTrainer._find_latest_middle_checkpoint(checkpoint_dir)
        if middle is not None:
            return middle
        middle_candidates = sorted(checkpoint_dir.parent.glob("*/model_E*_B*.pth"))
        return middle_candidates[-1] if middle_candidates else None

    def _resolve_resume_source_context(
        self,
        cfg: dict,
        args: argparse.Namespace,
    ) -> Path | None:
        """在调用可覆写 preview hook 前冻结持久 checkpoint source。"""
        resume_requested = bool(
            getattr(args, "resume", False) or getattr(args, "resume_from", None)
        )
        if not resume_requested:
            return None
        source_path = self._resume_checkpoint_path(cfg, args)
        if source_path is None:
            raise FileNotFoundError(
                "resume requested but local rank found no checkpoint"
            )
        try:
            return source_path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "resume requested but local rank found no checkpoint"
            ) from exc

    @staticmethod
    def _checkpoint_start_epoch(cfg: dict, checkpoint_path: Path) -> int:
        """只读 checkpoint 元数据，复用 BaseTrainer 的 epoch/cursor 恢复合同。"""
        from core.trainer.checkpoint_state import (
            resume_start_epoch,
            validate_resume_progress_schema,
        )
        from core.utils.safe_load import safe_torch_load

        checkpoint = safe_torch_load(
            checkpoint_path,
            allow_unsafe=True,
            purpose="resume epoch metadata",
        )
        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"resume checkpoint must contain a mapping: {checkpoint_path}"
            )
        accumulation = max(
            1, int(cfg.get("trainer", {}).get("accumulation_steps", 1))
        )
        # 复用 trainer 侧同一份 schema 校验：畸形 checkpoint 必须报契约错误，
        # 而不是在 resume_start_epoch 中抛 KeyError 或静默截断非整数 epoch。
        validate_resume_progress_schema(checkpoint, accumulation_steps=accumulation)
        return resume_start_epoch(checkpoint, accumulation)

    @staticmethod
    def _resume_snapshot_lock_path(snapshot: Path) -> Path:
        """返回 rank-local snapshot 的独立 consumer lease 文件。"""
        from core.trainer.resume_transaction import snapshot_lock_path

        return snapshot_lock_path(snapshot)

    @staticmethod
    def _resume_snapshot_guard_path(snapshot: Path) -> Path:
        """从唯一 snapshot 名恢复同 checkpoint basename 的转换 guard。"""
        from core.trainer.resume_transaction import snapshot_guard_path

        return snapshot_guard_path(snapshot)

    @staticmethod
    def _checkpoint_content_identity(checkpoint_path: Path) -> tuple[int, str]:
        """返回与路径、device 和 inode 无关的 checkpoint 内容身份。"""
        from core.trainer.checkpoint_state import checkpoint_content_identity

        return checkpoint_content_identity(checkpoint_path)

    def _freeze_resume_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        staging_dir: Path | None = None,
    ) -> tuple[Path, tuple[int, str]]:
        """复制 checkpoint 到本地可写 staging，并持有 lease 直至 load 完成。"""
        from core.trainer.resume_transaction import ResumeSnapshotTransaction

        return ResumeSnapshotTransaction.freeze(
            self,
            checkpoint_path,
            staging_dir=staging_dir,
        )

    @staticmethod
    def _best_effort_release_resume_snapshot(
        snapshot: Path,
        lock_path: Path,
        lease_fd: int,
        snapshot_identity: tuple[int, str],
        lock_identity: tuple[int, int],
    ) -> None:
        """释放本地 lease；仅最后一个 consumer 校验 identity 后删除目录项。"""
        from core.trainer.resume_transaction import ResumeSnapshotTransaction

        ResumeSnapshotTransaction.best_effort_release(
            snapshot,
            lock_path,
            lease_fd,
            snapshot_identity,
            lock_identity,
        )

    def _atexit_release_resume_checkpoint_owner(
        self,
        checkpoint_path: Path,
    ) -> None:
        """进程退出时关闭 owner 转换窗口并执行 identity-aware 清理。"""
        self._complete_resume_checkpoint_consumer_acquisition(
            checkpoint_path,
            suppress_errors=True,
        )
        self._release_resume_checkpoint_owner(checkpoint_path)

    def _release_resume_checkpoint_owner(
        self,
        checkpoint_path: Path | None,
    ) -> None:
        """仅创建方可释放 acquisition 失败后保留的 owner lease。"""
        if checkpoint_path is None:
            return
        leases = getattr(self, "_resume_snapshot_leases", {})
        owner = leases.pop(checkpoint_path, None)
        if owner is None:
            return
        lease_fd, lock_path, callback, identity, lock_identity = owner
        atexit.unregister(callback)
        self._best_effort_release_resume_snapshot(
            checkpoint_path,
            lock_path,
            lease_fd,
            identity,
            lock_identity,
        )

    def _acquire_resume_checkpoint_consumer(
        self,
        checkpoint_path: Path | None,
    ) -> None:
        """将 owner lease 转为本地共享锁，或为当前 rank 获取本地共享锁。"""
        if checkpoint_path is None:
            return
        identity = getattr(self, "_resolved_resume_identity", None)
        expected_lock_identity = getattr(
            self, "_resolved_resume_lock_identity", None
        )
        if identity is None:
            raise RuntimeError("resume snapshot identity is unavailable")

        owner_leases = getattr(self, "_resume_snapshot_leases", {})
        owner = owner_leases.get(checkpoint_path)
        guard_fd = None
        lease_fd = None
        lock_identity = (
            tuple(expected_lock_identity)
            if expected_lock_identity is not None
            else None
        )
        old_callback = None
        callback = None
        try:
            if owner is not None:
                (
                    lease_fd,
                    lock_path,
                    old_callback,
                    owner_identity,
                    lock_identity,
                ) = owner
                if owner_identity != identity:
                    raise RuntimeError("resume snapshot owner identity changed")
                guard_path = self._resume_snapshot_guard_path(checkpoint_path)
                guard_fd = os.open(guard_path, os.O_RDWR | os.O_CREAT, 0o600)
                fcntl.flock(guard_fd, fcntl.LOCK_EX)
                fcntl.flock(lease_fd, fcntl.LOCK_SH)
            else:
                lock_path = self._resume_snapshot_lock_path(checkpoint_path)
                lease_fd = os.open(lock_path, os.O_RDWR)
                fcntl.flock(lease_fd, fcntl.LOCK_SH)
                lock_stat = os.fstat(lease_fd)
                lock_identity = (lock_stat.st_dev, lock_stat.st_ino)

            if (
                expected_lock_identity is not None
                and lock_identity != tuple(expected_lock_identity)
            ):
                raise RuntimeError(
                    "resume snapshot sidecar identity validation failed: "
                    f"expected={tuple(expected_lock_identity)}, actual={lock_identity}"
                )
            local_error = self._checkpoint_identity_error(checkpoint_path, identity)
            if local_error is not None:
                raise RuntimeError(
                    f"resume checkpoint identity validation failed: {local_error}"
                )

            callback = functools.partial(
                self._atexit_release_resume_checkpoint_consumer,
                checkpoint_path,
            )
            atexit.register(callback)
            consumers = getattr(self, "_resume_snapshot_consumers", None)
            if consumers is None:
                consumers = {}
                self._resume_snapshot_consumers = consumers
            consumers[checkpoint_path] = (
                lease_fd,
                lock_path,
                callback,
                identity,
                lock_identity,
            )
            if owner is not None:
                owner_leases.pop(checkpoint_path, None)
                atexit.unregister(old_callback)
            lease_fd = None
            callback = None
            if guard_fd is not None:
                guards = getattr(self, "_resume_snapshot_transition_guards", None)
                if guards is None:
                    guards = {}
                    self._resume_snapshot_transition_guards = guards
                guards[checkpoint_path] = guard_fd
                guard_fd = None
        except BaseException:
            if callback is not None:
                atexit.unregister(callback)
            if owner is not None:
                # consumer 注册失败前不转移 owner 所有权；保留已降级的 lease，
                # 并持有转换 guard 到 all-rank acquisition 汇聚完成。
                lease_fd = None
                if guard_fd is not None:
                    guards = getattr(
                        self, "_resume_snapshot_transition_guards", None
                    )
                    if guards is None:
                        guards = {}
                        self._resume_snapshot_transition_guards = guards
                    guards[checkpoint_path] = guard_fd
                    guard_fd = None
            elif lease_fd is not None:
                try:
                    fcntl.flock(lease_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lease_fd)
                lease_fd = None
            raise
        finally:
            if guard_fd is not None:
                try:
                    fcntl.flock(guard_fd, fcntl.LOCK_UN)
                finally:
                    os.close(guard_fd)

    def _complete_resume_checkpoint_consumer_acquisition(
        self,
        checkpoint_path: Path | None,
        *,
        suppress_errors: bool = False,
    ) -> None:
        """all-rank local lease 获取完成后释放当前 rank 的转换 guard。"""
        if checkpoint_path is None:
            return
        guards = getattr(self, "_resume_snapshot_transition_guards", {})
        guard_fd = guards.get(checkpoint_path)
        if guard_fd is None:
            return
        if suppress_errors:
            guards.pop(checkpoint_path, None)
            try:
                fcntl.flock(guard_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(guard_fd)
            except OSError:
                pass
            return
        fcntl.flock(guard_fd, fcntl.LOCK_UN)
        # close() 报错时 POSIX fd 状态可能已不确定；unlock 已成功后先移除
        # ownership，失败 cleanup 不得对同一 fd 再次 close。
        guards.pop(checkpoint_path, None)
        os.close(guard_fd)

    def _atexit_release_resume_checkpoint_consumer(
        self,
        checkpoint_path: Path,
    ) -> None:
        """进程正常退出时先关闭 owner 转换窗口，再释放本地 consumer。"""
        self._complete_resume_checkpoint_consumer_acquisition(
            checkpoint_path,
            suppress_errors=True,
        )
        self._release_resume_checkpoint_consumer(checkpoint_path)

    def _release_resume_checkpoint_consumer(
        self,
        checkpoint_path: Path | None,
    ) -> None:
        """释放当前 rank 的 local lease；最后一个本地 consumer 安全删除。"""
        if checkpoint_path is None:
            return
        consumers = getattr(self, "_resume_snapshot_consumers", {})
        consumer = consumers.pop(checkpoint_path, None)
        if consumer is None:
            return
        lease_fd, lock_path, callback, identity, lock_identity = consumer
        atexit.unregister(callback)
        self._best_effort_release_resume_snapshot(
            checkpoint_path,
            lock_path,
            lease_fd,
            identity,
            lock_identity,
        )

    def _prepare_resume_checkpoint_consumers(
        self,
        checkpoint_path: Path | None,
    ) -> None:
        """在 trainer 前为所有 rank 获取各自 local lease，并汇聚转换结果。"""
        if checkpoint_path is None:
            return
        def cleanup_failed_transition() -> None:
            # 失败发生在 trainer 前：每 rank 释放本地转换 guard、consumer 与 owner。
            # 每项独立 best-effort，可容忍部分 unlock/close，且不覆盖原错误。
            cleanup_operations = (
                lambda: self._complete_resume_checkpoint_consumer_acquisition(
                    checkpoint_path,
                    suppress_errors=True,
                ),
                lambda: self._release_resume_checkpoint_consumer(checkpoint_path),
                lambda: self._release_resume_checkpoint_owner(checkpoint_path),
            )
            for cleanup in cleanup_operations:
                try:
                    cleanup()
                except Exception:
                    pass

        local_exception = None
        try:
            self._acquire_resume_checkpoint_consumer(checkpoint_path)
        except Exception as exc:
            local_exception = exc

        distributed = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        if distributed:
            local_error = (
                None
                if local_exception is None
                else f"{type(local_exception).__name__}: {local_exception}"
            )
            errors = [None] * dist.get_world_size()
            try:
                dist.all_gather_object(errors, local_error)
            except BaseException:
                cleanup_failed_transition()
                raise
            failures = [
                f"rank {rank}: {error}"
                for rank, error in enumerate(errors)
                if error is not None
            ]
        else:
            failures = (
                []
                if local_exception is None
                else [
                    f"rank 0: {type(local_exception).__name__}: {local_exception}"
                ]
            )

        if failures:
            cleanup_failed_transition()
            raise RuntimeError(
                "resume checkpoint consumer acquisition failed: "
                + "; ".join(failures)
            )

        completion_exception = None
        try:
            self._complete_resume_checkpoint_consumer_acquisition(checkpoint_path)
        except Exception as exc:
            completion_exception = exc

        if distributed:
            completion_error = (
                None
                if completion_exception is None
                else f"{type(completion_exception).__name__}: {completion_exception}"
            )
            completion_errors = [None] * dist.get_world_size()
            try:
                dist.all_gather_object(completion_errors, completion_error)
            except BaseException:
                cleanup_failed_transition()
                raise
            completion_failures = [
                f"rank {rank}: {error}"
                for rank, error in enumerate(completion_errors)
                if error is not None
            ]
        else:
            completion_failures = (
                []
                if completion_exception is None
                else [
                    "rank 0: "
                    f"{type(completion_exception).__name__}: {completion_exception}"
                ]
            )

        if completion_failures:
            cleanup_failed_transition()
            raise RuntimeError(
                "resume checkpoint consumer completion failed: "
                + "; ".join(completion_failures)
            )

    @staticmethod
    def _checkpoint_identity_error(
        checkpoint_path: Path,
        expected: tuple[int, str],
    ) -> str | None:
        """校验 snapshot 内容 digest/size，避免路径与 inode 假设。"""
        try:
            actual = UnifiedTrainingEntry._checkpoint_content_identity(
                checkpoint_path
            )
        except (OSError, RuntimeError) as exc:
            return f"{type(exc).__name__}: {exc}"
        if actual != expected:
            return f"expected={expected}, actual={actual}"
        return None

    def _validate_resume_checkpoint_before_trainer(
        self,
        checkpoint_path: Path | None,
    ) -> None:
        """在 trainer 构造前汇聚各 rank 的 snapshot identity 校验结果。"""
        identity = getattr(self, "_resolved_resume_identity", None)
        if checkpoint_path is None or identity is None:
            return
        local_error = self._checkpoint_identity_error(checkpoint_path, identity)
        distributed = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        if not distributed:
            if local_error is None:
                return
            self._cleanup_resume_checkpoint_snapshot(checkpoint_path)
            raise RuntimeError(
                f"resume checkpoint identity validation failed: {local_error}"
            )

        errors = [None] * dist.get_world_size()
        try:
            dist.all_gather_object(errors, local_error)
        except BaseException:
            try:
                self._cleanup_resume_checkpoint_snapshot(checkpoint_path)
            except BaseException:
                pass
            raise
        failures = [
            f"rank {rank}: {error}"
            for rank, error in enumerate(errors)
            if error is not None
        ]
        if not failures:
            return

        cleanup_error = None
        try:
            self._cleanup_resume_checkpoint_snapshot(
                checkpoint_path,
                synchronize=True,
            )
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
        message = "resume checkpoint identity validation failed: " + "; ".join(
            failures
        )
        if cleanup_error is not None:
            message += f"; snapshot cleanup failed: {cleanup_error}"
        raise RuntimeError(message)

    def _release_resume_checkpoint_snapshot(
        self, checkpoint_path: Path | None
    ) -> None:
        """trainer 前删除本地 snapshot，并释放创建方仍持有的 owner lease。"""
        if checkpoint_path is None:
            return
        leases = getattr(self, "_resume_snapshot_leases", {})
        lease = leases.pop(checkpoint_path, None)
        cleanup_error = None
        try:
            checkpoint_path.unlink(missing_ok=True)
            self._resume_snapshot_lock_path(checkpoint_path).unlink(missing_ok=True)
        except OSError as exc:
            cleanup_error = exc
        finally:
            if lease is not None:
                lease_fd, _lock_path, exit_callback, _identity, _lock_identity = lease
                atexit.unregister(exit_callback)
                try:
                    fcntl.flock(lease_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lease_fd)
        if cleanup_error is not None:
            raise cleanup_error

    def _cleanup_resume_checkpoint_snapshot(
        self,
        checkpoint_path: Path | None,
        *,
        synchronize: bool = False,
    ) -> None:
        """清理当前 rank 的本地 snapshot；同步模式汇聚各 rank 错误。"""
        if checkpoint_path is None:
            return
        distributed = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        if not distributed or not synchronize:
            self._release_resume_checkpoint_snapshot(checkpoint_path)
            return
        local_error = None
        try:
            self._release_resume_checkpoint_snapshot(checkpoint_path)
        except OSError as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, local_error)
        failures = [
            f"rank {rank}: {error}"
            for rank, error in enumerate(errors)
            if error is not None
        ]
        if failures:
            raise RuntimeError(
                "resume checkpoint snapshot cleanup failed: " + "; ".join(failures)
            )

    def _run_distributed_stage(
        self,
        stage: str,
        operation,
        *,
        checkpoint_path: Path | None = None,
    ):
        """先汇聚 rank-local 结果，再由所有 rank 对称清理并抛出同一错误。"""
        result = None
        local_exception = None
        try:
            result = operation()
        except Exception as exc:
            local_exception = exc

        distributed = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        if not distributed:
            if local_exception is not None:
                self._cleanup_resume_checkpoint_snapshot(checkpoint_path)
                raise local_exception
            return result

        local_error = (
            None
            if local_exception is None
            else f"{type(local_exception).__name__}: {local_exception}"
        )
        errors = [None] * dist.get_world_size()
        try:
            dist.all_gather_object(errors, local_error)
        except BaseException:
            try:
                self._cleanup_resume_checkpoint_snapshot(checkpoint_path)
            except BaseException:
                pass
            raise
        failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
        if not failures:
            return result

        cleanup_error = None
        try:
            self._cleanup_resume_checkpoint_snapshot(
                checkpoint_path,
                synchronize=True,
            )
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
        message = f"{stage} failed: " + "; ".join(failures)
        if cleanup_error is not None:
            message += f"; snapshot cleanup failed: {cleanup_error}"
        raise RuntimeError(message)

    def _prepare_train_sampler_for_scheduler(
        self,
        cfg: dict,
        args: argparse.Namespace,
        train_sampler,
    ) -> ResumePreparation:
        """解析首个训练 epoch，并显式返回持久 source 与本地 snapshot。"""
        resume_requested = bool(
            getattr(args, "resume", False) or getattr(args, "resume_from", None)
        )
        distributed = (
            resume_requested
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        self._resolved_resume_identity = None
        self._resolved_resume_lock_identity = None

        def freeze_local_checkpoint():
            source_path = getattr(self, "_resume_source_context", None)
            if source_path is None:
                source_path = self._resolve_resume_source_context(cfg, args)
            if source_path is None:
                raise FileNotFoundError("resume requested but local rank found no checkpoint")
            exp_path = cfg.get("trainer", {}).get("exp_path", "exp/default")
            staging_dir = Path(exp_path).resolve() / "resume-staging"
            if distributed:
                staging_dir = staging_dir / f"rank-{dist.get_rank()}"
            snapshot_path, local_identity = self._freeze_resume_checkpoint(
                source_path, staging_dir=staging_dir
            )
            try:
                local_start_epoch = self._checkpoint_start_epoch(cfg, snapshot_path)
            except Exception:
                self._release_resume_checkpoint_snapshot(snapshot_path)
                raise
            local_lease = self._resume_snapshot_leases[snapshot_path]
            return (
                source_path,
                snapshot_path,
                local_start_epoch,
                local_identity,
                local_lease[4],
            )

        if distributed:
            local_payload = None
            try:
                (
                    source_path,
                    checkpoint_path,
                    start_epoch,
                    identity,
                    lock_identity,
                ) = freeze_local_checkpoint()
                local_payload = (
                    str(checkpoint_path),
                    start_epoch,
                    identity,
                    lock_identity,
                    None,
                )
            except Exception as exc:
                local_payload = (
                    None,
                    None,
                    None,
                    None,
                    f"{type(exc).__name__}: {exc}",
                )
            payloads = [None] * dist.get_world_size()
            try:
                dist.all_gather_object(payloads, local_payload)
            except BaseException:
                self._release_resume_checkpoint_snapshot(
                    Path(local_payload[0]) if local_payload[0] is not None else None
                )
                raise
            failures = [
                f"rank {rank}: {payload[4]}"
                for rank, payload in enumerate(payloads)
                if payload[4] is not None
            ]
            if not failures:
                identities = {tuple(payload[2]) for payload in payloads}
                start_epochs = {int(payload[1]) for payload in payloads}
                if len(identities) != 1:
                    failures.append("checkpoint content digest/size differs across ranks")
                if len(start_epochs) != 1:
                    failures.append("checkpoint start epoch differs across ranks")
            if failures:
                self._release_resume_checkpoint_snapshot(
                    Path(local_payload[0]) if local_payload[0] is not None else None
                )
                raise RuntimeError("resume preview failed: " + "; ".join(failures))
            self._resolved_resume_identity = tuple(identity)
            self._resolved_resume_lock_identity = tuple(lock_identity)
        elif resume_requested:
            try:
                (
                    source_path,
                    checkpoint_path,
                    start_epoch,
                    identity,
                    lock_identity,
                ) = freeze_local_checkpoint()
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "resume requested but local rank found no checkpoint"
                ) from exc
            self._resolved_resume_identity = identity
            self._resolved_resume_lock_identity = lock_identity
        else:
            source_path = None
            checkpoint_path = None
            start_epoch = 1

        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            self._run_distributed_stage(
                "sampler preparation",
                lambda: train_sampler.set_epoch(start_epoch),
                checkpoint_path=checkpoint_path,
            )
        return ResumePreparation(start_epoch, source_path, checkpoint_path)

    @staticmethod
    def _scheduler_config_uses_auto_horizon(scheduler_cfg: dict) -> bool:
        """返回 LinearWarmupCosineAnnealing 是否含实际 auto step 参数。"""
        if scheduler_cfg.get("type") != "LinearWarmupCosineAnnealing":
            return False
        args = scheduler_cfg.get("args", {})
        return any(args.get(name) in (None, "auto") for name in (
            "warmup_steps",
            "decay_until_step",
        ))

    def _base_scheduler_needs_planned_steps(self, cfg: dict, optimizer) -> bool:
        """仅基类 auto per-step scheduler 需要完整 optimizer-step 计划。"""
        if not bool(cfg.get("trainer", {}).get("lr_schedule_per_step", False)):
            return False
        bound_builder = self.build_scheduler
        if getattr(bound_builder, "__func__", None) is not UnifiedTrainingEntry.build_scheduler:
            return False
        scheduler_cfg = cfg.get("scheduler", {})
        if isinstance(optimizer, dict):
            return any(
                self._scheduler_config_uses_auto_horizon(
                    scheduler_cfg.get(name, scheduler_cfg)
                    if isinstance(scheduler_cfg, dict) and name in scheduler_cfg
                    else scheduler_cfg
                )
                for name in optimizer
            )
        return self._scheduler_config_uses_auto_horizon(scheduler_cfg)

    @staticmethod
    def _planned_optimizer_steps(
        cfg: dict,
        train_sampler,
        n_train_batches: int,
        *,
        world_size: int,
    ) -> int:
        """返回 auto per-step scheduler 的完整训练 optimizer-step 计划。"""
        trainer_cfg = cfg.get("trainer", {})
        epochs = max(0, int(trainer_cfg.get("epochs", 200)))
        accumulation = int(trainer_cfg.get("accumulation_steps", 1))
        batch_count = getattr(train_sampler, "batch_count_for_epoch", None)
        if callable(batch_count):
            return sum(
                optimizer_steps_per_epoch(
                    batch_count(epoch), accumulation, world_size=world_size
                )
                for epoch in range(1, epochs + 1)
            )
        return epochs * optimizer_steps_per_epoch(
            n_train_batches, accumulation, world_size=world_size
        )

    def run_training(
        self,
        cfg: dict,
        args: argparse.Namespace,
        rank: int,
        local_rank: int,
        world_size: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_sampler,
        train_ds,
        model: torch.nn.Module | dict[str, torch.nn.Module],
        trainer_cls,
    ) -> None:
        """Build loss/optimizer/scheduler/trainer, load checkpoint, setup signals, run train()."""
        # Loss
        loss_cfg = cfg.setdefault("loss", {})
        if cfg.get("trainer", {}).get("strict_checks", True) and "strict" not in loss_cfg:
            loss_cfg["strict"] = True
        from core.losses.build import build_loss

        loss_fn = build_loss(
            cfg["loss"],
            stft_cfg=cfg.get("stft"),
            sample_rate=cfg["data"].get("sample_rate"),
        )

        # Optimizer：构造与完整参数覆盖检查都使用既有 distributed stage 汇聚，
        # 禁止单 rank 启动失败后其余 rank 继续进入 scheduler/DDP 构造。
        def build_and_validate_optimizer():
            from core.trainer.entry_components import (
                validate_optimizer_parameter_coverage,
            )

            built = self.build_optimizer(cfg, model)
            validate_optimizer_parameter_coverage(model, built)
            return built

        optimizer = self._run_distributed_stage(
            "optimizer startup preflight", build_and_validate_optimizer
        )

        # Scheduler：先将 persistent source 固化到非 virtual 入口，再调用可覆写的
        # sampler preview seam；seam 必须返回 ResumePreparation dataclass。
        resume_source = self._run_distributed_stage(
            "resume source preparation",
            lambda: self._resolve_resume_source_context(cfg, args),
        )
        _missing_resume_context = object()
        _previous_resume_context = getattr(
            self, "_resume_source_context", _missing_resume_context
        )
        self._resume_source_context = resume_source
        try:
            resume_preparation = ResumePreparation.normalize(
                self._prepare_train_sampler_for_scheduler(cfg, args, train_sampler),
                source_path=resume_source,
            )
        finally:
            if _previous_resume_context is _missing_resume_context:
                self.__dict__.pop("_resume_source_context", None)
            else:
                self._resume_source_context = _previous_resume_context

        def prepare_trainer_kwargs():
            n_train_batches = len(train_loader)
            # scheduler 构建期间把 planned steps / resolved args 经 entry context
            # 传给 build_single_scheduler。保存旧值可支持同线程重入，finally 保证
            # 异常时不泄漏到后续 run。
            _missing = object()
            _previous_planned_steps = getattr(
                self, "_scheduler_planned_optimizer_steps", _missing
            )
            _previous_resolved_args = getattr(
                self, "_resolved_scheduler_args", _missing
            )
            try:
                _needs_planned_steps = self._base_scheduler_needs_planned_steps(
                    cfg, optimizer
                )
                _planned_steps_for_log = (
                    self._planned_optimizer_steps(
                        cfg,
                        train_sampler,
                        n_train_batches,
                        world_size=world_size,
                    )
                    if _needs_planned_steps
                    else int(cfg.get("trainer", {}).get("epochs", 200))
                    * optimizer_steps_per_epoch(
                        n_train_batches,
                        int(cfg.get("trainer", {}).get("accumulation_steps", 1)),
                        world_size=world_size,
                    )
                )
                self._scheduler_planned_optimizer_steps = (
                    _planned_steps_for_log if _needs_planned_steps else None
                )
                self._resolved_scheduler_args = {}
                scheduler = self.build_scheduler(
                    cfg, n_train_batches, optimizer, world_size=world_size
                )
                if rank == 0:
                    _trainer_cfg = cfg.get("trainer", {})
                    _accum_steps = max(1, int(_trainer_cfg.get("accumulation_steps", 1)))
                    _per_rank_batch_size = int(
                        cfg.get("data", {}).get("dataloader", {}).get(
                            "batch_size", getattr(train_loader, "batch_size", 1) or 1
                        )
                    )
                    _effective_batch_size = (
                        _per_rank_batch_size * world_size * _accum_steps
                    )
                    _steps_per_epoch = optimizer_steps_per_epoch(
                        n_train_batches,
                        _accum_steps,
                        world_size=world_size,
                    )
                    _resolved_args = getattr(self, "_resolved_scheduler_args", {})
                    _scheduler_horizon = " | ".join(
                        f"{name}(warmup={args.get('warmup_steps')},decay={args.get('decay_until_step')})"
                        for name, args in _resolved_args.items()
                    ) or "unknown"
                    self.logger.info(
                        "Training scale: world_size=%d | per_rank_train_batches=%d | "
                        "per_rank_batch_size=%d | global_effective_batch_size=%d "
                        "(accumulation_steps=%d) | optimizer_steps_per_epoch=%d | "
                        "planned_total_optimizer_steps=%d | scheduler_horizon=%s",
                        world_size,
                        n_train_batches,
                        _per_rank_batch_size,
                        _effective_batch_size,
                        _accum_steps,
                        _steps_per_epoch,
                        _planned_steps_for_log,
                        _scheduler_horizon,
                    )
                    _opt_cfg = cfg.get("optimizer") or {}
                    _sch_cfg = cfg.get("scheduler") or {}
                    self.logger.info(
                        "Training plan: epochs=%s | optimizer=%s | lr=%s | "
                        "scheduler=%s | clip=%s | save_top_k=%s | "
                        "val_interval=%s epochs | accumulation_steps=%s",
                        _trainer_cfg.get("epochs"),
                        _opt_cfg.get("type"),
                        (_opt_cfg.get("args") or {}).get("lr"),
                        _sch_cfg.get("type"),
                        _trainer_cfg.get("gradient_clip_value"),
                        _trainer_cfg.get("save_top_k"),
                        _trainer_cfg.get("validation_epoch_interval"),
                        _accum_steps,
                    )
            finally:
                if _previous_planned_steps is _missing:
                    self.__dict__.pop("_scheduler_planned_optimizer_steps", None)
                else:
                    self._scheduler_planned_optimizer_steps = _previous_planned_steps
                if _previous_resolved_args is _missing:
                    self.__dict__.pop("_resolved_scheduler_args", None)
                else:
                    self._resolved_scheduler_args = _previous_resolved_args
            stft_module = self.build_stft_module(cfg, rank)
            return {
                "train_loader": train_loader,
                "val_loader": val_loader,
                "sample_rate": cfg["data"].get("sample_rate", 16000),
                "stft_module": stft_module,
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "loss_fn": loss_fn,
                "config": cfg,
                "train_sampler": train_sampler,
                "train_dataset": train_ds,
                "resume": resume_preparation.snapshot_path is not None,
                "resume_from": (
                    str(resume_preparation.snapshot_path)
                    if resume_preparation.snapshot_path is not None
                    else None
                ),
                "use_wandb": args.wandb,
            }

        trainer_kw = self._run_distributed_stage(
            "trainer preparation",
            prepare_trainer_kwargs,
            checkpoint_path=resume_preparation.snapshot_path,
        )
        self._validate_resume_checkpoint_before_trainer(
            resume_preparation.snapshot_path
        )
        self._prepare_resume_checkpoint_consumers(resume_preparation.snapshot_path)
        try:
            if resume_preparation.snapshot_path is not None:
                from core.trainer.base_trainer import BaseTrainer

                if issubclass(trainer_cls, BaseTrainer):
                    trainer_kw["resume_load_from"] = str(
                        resume_preparation.snapshot_path
                    )
                    trainer_kw["resume_source_from"] = str(
                        resume_preparation.source_path
                    )
            trainer = trainer_cls(**trainer_kw)
        finally:
            # 构造完成或失败都只释放本 rank 的 local lease；最后一个本地 consumer
            # 按 digest/size 校验后清理，不执行任何构造后 distributed collective。
            self._release_resume_checkpoint_consumer(resume_preparation.snapshot_path)
        if args.checkpoint and not args.resume:
            ckpt_path = str(Path(args.checkpoint).resolve())
            trainer.load_pretrained(ckpt_path)
            if rank == 0:
                self.logger.info("Loaded pretrained weights from %s", ckpt_path)

        # Signal handlers
        self.setup_signal_handlers(trainer, rank, world_size)

        self._run_trainer_main_loop(trainer, world_size)

    def _destroy_process_group_noexcept(self) -> None:
        """Best-effort 销毁 process group，且绝不覆盖主流程异常或退出码。"""
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except BaseException as cleanup_error:
            try:
                self.logger.exception(
                    "Failed to destroy distributed process group during cleanup: %s",
                    cleanup_error,
                )
            except BaseException:
                pass

    def _run_trainer_main_loop(self, trainer, world_size: int) -> None:
        """运行训练并统一处理 distributed cleanup 与信号退出码。"""
        try:
            trainer.train()
        except Exception as error:
            try:
                self._log_exception_rank_aware(error, "Training failed")
                # 标记已记录，外层 run() 的 Fatal handler 不再重复打 traceback。
                setattr(error, "_entry_traceback_logged", True)
            except BaseException:
                pass
            raise
        finally:
            if world_size > 1:
                self._destroy_process_group_noexcept()

        # 训练主流程已在安全边界保存 checkpoint 并退出循环；此处补回与原
        # 信号 handler 一致的退出码语义（128+signum），供调度系统识别。
        shutdown_query = getattr(trainer, "shutdown_requested", None)
        if callable(shutdown_query) and shutdown_query():
            signum = getattr(trainer, "_shutdown_signal", None) or signal.SIGTERM
            if _current_dist_rank() == 0:
                self.logger.info(
                    "Graceful shutdown complete (signal %s); exiting.", signum
                )
            sys.exit(128 + int(signum))

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        config: dict | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """启动训练；可接收顶层路由已加载的配置与其来源路径。"""
        check_python_version()
        self._ensure_free_master_port()
        try:
            if config is None and config_path is None:
                args, cfg, rank, local_rank, world_size = self.load_and_setup()
            else:
                args, cfg, rank, local_rank, world_size = self.load_and_setup(
                    config=config,
                    config_path=config_path,
                )

            # Task detection and explicit logging
            task_info = detect_task(cfg)
            if rank == 0:
                self._log_task_info(task_info, cfg, config_path)

            _ensure_file_descriptor_limit(rank=rank)

            def build_and_preflight_loaders():
                result = self.build_loaders(cfg, rank, local_rank, world_size)
                if not isinstance(result, tuple) or len(result) != 4:
                    raise TypeError(
                        "build_loaders must return "
                        "(train_loader, val_loader, train_sampler, train_ds)"
                    )
                train_loader, val_loader, train_sampler, train_ds = result
                len(train_loader)
                len(val_loader)
                strict_load = getattr(train_ds, "strict_load", None)
                if type(strict_load) is not bool or strict_load is not True:
                    raise ValueError(
                        "training dataset must expose strict_load=True before the "
                        "first batch; lenient datasets may emit placeholders that "
                        "would enter loss without a validity mask"
                    )
                return train_loader, val_loader, train_sampler, train_ds

            train_loader, val_loader, train_sampler, train_ds = (
                self._run_distributed_stage(
                    "loader startup preflight", build_and_preflight_loaders
                )
            )

            def log_data_statistics():
                if rank != 0:
                    return None
                val_ds = getattr(val_loader, "dataset", None)
                self._log_data_statistics(
                    task_info,
                    train_ds,
                    val_ds,
                    len(train_loader),
                    len(val_loader),
                    cfg,
                )

            self._run_distributed_stage(
                "data statistics startup preflight", log_data_statistics
            )
            model = self._run_distributed_stage(
                "model startup preflight",
                lambda: self.build_model_for_training(
                    cfg,
                    rank,
                    train_dataset=train_ds,
                ),
            )

            # Auto-select trainer based on paradigm and explicit trainer.type
            trainer_type = cfg.get("trainer", {}).get("type", "").strip()
            if task_info.paradigm == "gan":
                # Allow explicit subclass (e.g. MetricGANCompositeTrainer) via trainer.type
                trainer_cls = self._import_trainer_class(trainer_type or "GANTrainer")
            else:
                trainer_cls = self._import_trainer_class(trainer_type or "default")

            self.run_training(
                cfg, args, rank, local_rank, world_size,
                train_loader, val_loader, train_sampler, train_ds,
                model, trainer_cls,
            )
        except KeyboardInterrupt:
            if _current_dist_rank() == 0:
                self.logger.info("Training interrupted by user (KeyboardInterrupt).")
            self._destroy_process_group_noexcept()
            sys.exit(130)
        except Exception as e:
            if not getattr(e, "_entry_traceback_logged", False):
                self._log_exception_rank_aware(e, "Fatal error in training")
            sys.exit(1)
        finally:
            # 仅释放本进程拥有的在线声学服务，不引入 distributed collective。
            from core.data.simulation.rir_service import close_online_rir_services

            close_online_rir_services()

    # ------------------------------------------------------------------
    # Trainer class resolution
    # ------------------------------------------------------------------
    def _trainer_resolver(self):
        """返回绑定当前任务 entry 的 Trainer 注册 facade。"""
        from core.trainer.trainer_resolver import TaskTrainerResolver

        resolver = getattr(self, "_task_trainer_resolver", None)
        if resolver is None:
            package_name = type(self).__module__.rsplit(".", 1)[0]
            resolver = TaskTrainerResolver(package_name)
            resolver.register("default", self.get_trainer_cls)
            resolver.register("GANTrainer", self._default_gan_trainer_cls)
            self._task_trainer_resolver = resolver
        return resolver

    def _import_trainer_class(self, class_name: str):
        """按 fully-qualified path 或当前任务短名解析 Trainer class。"""
        return self._trainer_resolver().resolve(class_name)

    # ------------------------------------------------------------------
    # Abstract methods (subclasses must implement)
    # ------------------------------------------------------------------
    @abstractmethod
    def build_loaders(self, cfg: dict, rank: int, local_rank: int, world_size: int):
        """Return (train_loader, val_loader, train_sampler, train_ds)."""
        raise NotImplementedError

    def build_model_for_training(
        self,
        cfg: dict,
        rank: int,
        *,
        train_dataset,
    ) -> torch.nn.Module | dict[str, torch.nn.Module]:
        """由训练启动路径将 train dataset 事实交给任务专属模型适配层。"""
        return self.build_model(cfg, rank)

    @abstractmethod
    def build_model(self, cfg: dict, rank: int) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_trainer_cls(self):
        raise NotImplementedError

    def _default_gan_trainer_cls(self):
        """Task-specific default GAN trainer class, used when paradigm=='gan' and
        trainer.type is unset (or == 'GANTrainer') in the YAML. Subclasses that
        support GAN training must override; the base raises so an unsupported
        task fails loudly instead of silently picking up another task's trainer.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no default GAN trainer. "
            f"Set trainer.type explicitly in YAML (e.g. a fully-qualified "
            f"'module.ClassName') if you intend to run GAN-paradigm training."
        )
