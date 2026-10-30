"""UnifiedTrainingEntry 的 bootstrap、loader 与通用组件 builder。"""

from __future__ import annotations

import datetime
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from core.utils.validation import require_positive_int


def bootstrap_entry(
    entry: Any,
    *,
    config: dict | None = None,
    config_path: str | Path | None = None,
):
    """执行 entry bootstrap，并保持原初始化顺序与返回 tuple。"""
    if (config is None) != (config_path is None):
        raise ValueError("config and config_path must be provided together")

    args = entry.parse_args()
    if not Path(args.config).exists():
        entry.logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    if args.log_file and rank == 0:
        handler = logging.FileHandler(args.log_file, mode="a")
        handler.setLevel(log_level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        )
        logging.getLogger().addHandler(handler)

    if config is not None:
        if Path(args.config).resolve() != Path(config_path).resolve():
            raise ValueError(
                "provided config_path must resolve to the CLI --config path: "
                f"cli={Path(args.config).resolve()}, provided={Path(config_path).resolve()}"
            )
        cfg = config
    else:
        from core.utils.config import load_merged_config

        cfg = load_merged_config(args.config)
    entry._validate_config(cfg)
    if rank == 0:
        entry.logger.info(
            f"Config: {Path(args.config).resolve()} | "
            f"seed={cfg.get('meta', {}).get('seed', 42)}"
        )

    if world_size > 1:
        backend = cfg.get("distributed", {}).get("backend", "nccl")
        timeout_min = int(
            cfg.get("distributed", {}).get("nccl_timeout_minutes", 30)
        )
        cuda_available = torch.cuda.is_available()
        if backend == "nccl" and not cuda_available:
            raise RuntimeError(
                "distributed.backend=nccl requires CUDA; use backend=gloo for CPU DDP"
            )
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(minutes=timeout_min),
        )
        if cuda_available:
            torch.cuda.set_device(local_rank)
        if rank == 0:
            device_kind = "cuda" if cuda_available else "cpu"
            entry.logger.info(
                f"DDP backend={backend} | device={device_kind} | "
                f"timeout={timeout_min} min"
            )

    entry._apply_strict_mode(cfg, args.strict, rank)
    if args.max_steps is not None:
        cfg.setdefault("trainer", {})["max_steps"] = int(args.max_steps)
        if rank == 0:
            entry.logger.info(f"Verify mode: max_steps={args.max_steps}")
    if args.exp_path is not None:
        cfg.setdefault("trainer", {})["exp_path"] = str(args.exp_path)
        if rank == 0:
            entry.logger.info(f"Override exp_path: {args.exp_path}")

    entry.seed_everything(
        cfg.get("meta", {}).get("seed", 42),
        rank,
        deterministic=cfg.get("meta", {}).get("deterministic", False),
    )
    return args, cfg, rank, local_rank, world_size


def build_single_optimizer(
    opt_cfg: dict,
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    """按原参数分组合同构造单 optimizer。"""
    opt_type = opt_cfg["type"]
    opt_args = opt_cfg.get("args", {})
    weight_decay = opt_args.get("weight_decay", 0.0)
    no_decay_names: set[str] = set()
    normalization_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.GroupNorm,
    )
    for module_name, module in model.named_modules():
        if isinstance(module, normalization_types):
            for parameter_name, _ in module.named_parameters():
                no_decay_names.add(
                    f"{module_name}.{parameter_name}"
                    if module_name
                    else parameter_name
                )
        else:
            for parameter_name, _ in module.named_parameters():
                if parameter_name.endswith("bias"):
                    no_decay_names.add(
                        f"{module_name}.{parameter_name}"
                        if module_name
                        else parameter_name
                    )

    parameters = dict(model.named_parameters())
    groups = [
        {
            "params": [
                parameter
                for name, parameter in parameters.items()
                if name not in no_decay_names
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in parameters.items()
                if name in no_decay_names
            ],
            "weight_decay": 0.0,
        },
    ]
    if hasattr(torch.optim, opt_type):
        return getattr(torch.optim, opt_type)(groups, **opt_args)
    try:
        import adamp

        return getattr(adamp, opt_type)(groups, **opt_args)
    except (ImportError, AttributeError):
        raise ValueError(
            f"Unknown optimizer '{opt_type}'. Not in torch.optim or adamp."
        )


def validate_optimizer_parameter_coverage(
    model: torch.nn.Module | Mapping[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer | Mapping[str, torch.optim.Optimizer],
) -> None:
    """拒绝 optimizer 遗漏、重复或引用模型外参数。"""
    models = dict(model) if isinstance(model, Mapping) else {"default": model}
    optimizers = (
        dict(optimizer) if isinstance(optimizer, Mapping) else {"default": optimizer}
    )
    if set(models) != set(optimizers):
        raise ValueError(
            "Model/optimizer keys must match exactly: "
            f"models={sorted(models)}, optimizers={sorted(optimizers)}"
        )

    for name, current_model in models.items():
        named_parameters = dict(current_model.named_parameters())
        parameter_names = {id(parameter): key for key, parameter in named_parameters.items()}
        trainable_ids = {
            id(parameter)
            for parameter in named_parameters.values()
            if parameter.requires_grad
        }
        counts: dict[int, int] = {}
        foreign: list[str] = []
        for group_index, group in enumerate(optimizers[name].param_groups):
            for parameter_index, parameter in enumerate(group.get("params", ())):
                parameter_id = id(parameter)
                counts[parameter_id] = counts.get(parameter_id, 0) + 1
                if parameter_id not in parameter_names:
                    foreign.append(f"group {group_index} parameter {parameter_index}")
        missing = sorted(
            parameter_names[parameter_id]
            for parameter_id in trainable_ids
            if counts.get(parameter_id, 0) == 0
        )
        duplicate = sorted(
            parameter_names.get(parameter_id, f"<foreign:{parameter_id}>")
            for parameter_id, count in counts.items()
            if count != 1
        )
        if missing or duplicate or foreign:
            raise ValueError(
                f"Optimizer parameter coverage failed for model '{name}': "
                f"missing_trainable={missing[:10]}, duplicate={duplicate[:10]}, "
                f"foreign={foreign[:10]}"
            )


def build_single_scheduler(
    entry: Any,
    sch_cfg: dict,
    n_train_batches: int,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    *,
    world_size: int,
    scheduler_name: str,
):
    """构造单 scheduler，并保留 auto horizon 与日志参数状态。"""
    from core.optim.scheduler import resolve_scheduler_auto_args
    from core.utils.registry import SCHEDULER_REGISTRY
    import core.optim  # noqa: F401

    accumulation = max(
        1, int(cfg.get("trainer", {}).get("accumulation_steps", 1))
    )
    if world_size > 1:
        steps_per_epoch = max(0, int(n_train_batches)) // accumulation
    else:
        steps_per_epoch = (
            max(0, int(n_train_batches)) + accumulation - 1
        ) // accumulation
    per_step = bool(cfg.get("trainer", {}).get("lr_schedule_per_step", False))
    epochs = cfg.get("trainer", {}).get("epochs", 200)
    planned_steps = getattr(entry, "_scheduler_planned_optimizer_steps", None)
    if per_step and planned_steps is not None:
        steps_per_epoch = int(planned_steps)
        epochs = 1
        if sch_cfg.get("type") == "LinearWarmupCosineAnnealing" and steps_per_epoch == 0:
            raise ValueError(
                "auto per-step scheduler horizon is zero; no optimizer step is "
                "planned for the configured epochs, accumulation, and world size"
            )
    scheduler_args = resolve_scheduler_auto_args(
        sch_cfg,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        rank=int(os.environ.get("RANK", 0)),
        per_step=per_step,
    )
    if not isinstance(getattr(entry, "_resolved_scheduler_args", None), dict):
        entry._resolved_scheduler_args = {}
    entry._resolved_scheduler_args[scheduler_name] = dict(scheduler_args)
    return SCHEDULER_REGISTRY.build(
        sch_cfg["type"], optimizer=optimizer, **scheduler_args
    )


def build_stft_module(entry: Any, cfg: dict, rank: int):
    """构造 framework STFT module。"""
    stft_cfg = cfg.get("stft")
    if not stft_cfg:
        return None
    from core.stft import build_stft_istft

    stft_module = build_stft_istft(stft_cfg)
    if rank == 0:
        entry.logger.info(
            "Framework STFT pipeline: wav → STFT → model → iSTFT → wav "
            "(all models when stft is set)."
        )
    return stft_module


def _require_positive_int(dl_cfg: Mapping, key: str) -> int:
    """读取精确正整数配置项，报错信息携带完整键名。"""
    full_key = f"data.dataloader.{key}"
    if key not in dl_cfg:
        raise ValueError(f"{full_key} is required")
    return require_positive_int(dl_cfg[key], full_key)


def resolve_loader_batch_sizes(dl_cfg: Mapping) -> tuple[int, int]:
    """返回 (train_batch_size, val_batch_size)；val_batch_size 缺省等于 train。"""
    train_batch_size = _require_positive_int(dl_cfg, "batch_size")
    if "val_batch_size" in dl_cfg:
        val_batch_size = _require_positive_int(dl_cfg, "val_batch_size")
    else:
        val_batch_size = train_batch_size
    return train_batch_size, val_batch_size


def build_loaders_common(
    entry: Any,
    cfg: dict,
    rank: int,
    local_rank: int,
    world_size: int,
    *,
    train_ds: Any,
    val_ds: Any,
    train_collate_fn: Any,
    val_collate_fn: Any = None,
    train_num_per_epoch: int | None = None,
    val_num_per_epoch: int | None = None,
):
    """装配共享 loader；validation 在构造期按 epoch 0 严格预检全部 DDP 分片。"""
    from core.data.sampler import GroupedBatchSampler
    from core.utils.dataloader_worker import dataloader_worker_init_fn

    val_collate_fn = val_collate_fn or train_collate_fn
    dl_cfg = cfg["data"]["dataloader"]
    experiment_seed = cfg.get("meta", {}).get("seed", 42)
    train_batch_size, val_batch_size = resolve_loader_batch_sizes(dl_cfg)
    entry._validate_num_per_epoch(
        train_num_per_epoch, "data.train.num_per_epoch"
    )
    entry._validate_num_per_epoch(
        val_num_per_epoch, "data.validation.num_per_epoch"
    )
    if (
        val_num_per_epoch is not None
        and world_size > 1
        and val_num_per_epoch < world_size
    ):
        raise ValueError(
            "data.validation.num_per_epoch is a global validation cardinality "
            f"and must be >= world_size={world_size} under DDP, got "
            f"{val_num_per_epoch}"
        )

    train_sampler = GroupedBatchSampler(
        train_ds,
        batch_size=train_batch_size,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        seed=experiment_seed,
        drop_last=True,
        num_per_epoch=train_num_per_epoch,
        require_non_empty_train_shards=True,
    )
    val_sampler = GroupedBatchSampler(
        val_ds,
        batch_size=val_batch_size,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        seed=experiment_seed,
        drop_last=False,
        num_per_epoch=val_num_per_epoch,
        require_non_empty_all_ranks=True,
    )
    val_sampler.set_epoch(0)
    if (
        world_size > 1
        and len(val_ds) > 0
        and len(val_ds) < world_size * val_batch_size
        and rank == 0
    ):
        entry.logger.warning(
            "Validation set size (%s) < world_size * batch_size (%s). "
            "DDP padding may add duplicate forward passes on some ranks; "
            "padding batches are excluded from validation loss and metrics.",
            len(val_ds),
            world_size * val_batch_size,
        )

    train_num_workers = int(dl_cfg.get("num_workers", 4))
    val_num_workers = int(dl_cfg.get("val_num_workers", train_num_workers))
    train_persistent_workers = train_num_workers > 0 and bool(
        dl_cfg.get("persistent_workers", True)
    )
    val_persistent_workers = val_num_workers > 0 and bool(
        dl_cfg.get(
            "val_persistent_workers",
            dl_cfg.get("persistent_workers", True),
        )
    )
    common_kwargs = {
        "pin_memory": bool(dl_cfg.get("pin_memory", True)),
        "worker_init_fn": dataloader_worker_init_fn,
    }
    timeout = int(dl_cfg.get("timeout", 0) or 0)
    prefetch_factor = dl_cfg.get("prefetch_factor", 4)
    train_kwargs = {
        **common_kwargs,
        "num_workers": train_num_workers,
        "persistent_workers": train_persistent_workers,
    }
    val_kwargs = {
        **common_kwargs,
        "num_workers": val_num_workers,
        "persistent_workers": val_persistent_workers,
    }
    if train_num_workers > 0:
        train_kwargs["prefetch_factor"] = prefetch_factor
        train_kwargs["timeout"] = timeout
    if val_num_workers > 0:
        val_kwargs["prefetch_factor"] = prefetch_factor
        val_kwargs["timeout"] = timeout
    train_generator = torch.Generator().manual_seed(experiment_seed)
    val_generator = torch.Generator().manual_seed(experiment_seed + 1)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=train_collate_fn,
        generator=train_generator,
        **train_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=val_collate_fn,
        generator=val_generator,
        **val_kwargs,
    )
    if rank == 0:
        entry.logger.info(
            f"Train: {len(train_ds)} samples, {len(train_loader)} batches | "
            f"Val: {len(val_ds)} samples, {len(val_loader)} batches"
        )
    return train_loader, val_loader, train_sampler, train_ds
