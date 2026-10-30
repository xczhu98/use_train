"""Learning rate schedulers (LR Schedulers).

Available schedulers (via YAML scheduler.type):
  - LinearWarmupCosineAnnealing:       Linear warmup + cosine annealing (from SEtrain-plus, recommended)
  - CosineAnnealingWarmupRestarts:     Cosine annealing + linear warmup + cycle restarts
  - ExponentialDecrease:               Exponential interpolation from initial_lr to final_lr
  - ExponentialLR:                     Exponential decay (USR-BSRNN paper: gamma=0.99 per epoch)
  - StepLR:                            Fixed step decay
  - CosineAnnealingLR:                 Pure cosine annealing
  - ReduceLROnPlateau:                 Reduce LR based on validation metric
"""

import logging
import math
from typing import Any

from torch.optim.lr_scheduler import _LRScheduler
from core.utils.registry import SCHEDULER_REGISTRY

logger = logging.getLogger("scheduler")


def _validate_optimizer_base_lrs(optimizer) -> None:
    """在 scheduler 初始化前拒绝会破坏比例缩放的 optimizer 基准学习率。"""
    for index, group in enumerate(optimizer.param_groups):
        for key in ("lr", "initial_lr"):
            if key not in group:
                continue
            base_lr = group[key]
            if (
                isinstance(base_lr, bool)
                or not isinstance(base_lr, (int, float))
                or not math.isfinite(float(base_lr))
                or float(base_lr) <= 0.0
            ):
                raise ValueError(
                    "optimizer param group base_lr must be finite and positive, got "
                    f"{base_lr!r} for {key} at index {index}"
                )


def resolve_scheduler_auto_args(
    sch_cfg: dict[str, Any],
    steps_per_epoch: int,
    epochs: int,
    rank: int = 0,
    per_step: bool = True,
) -> dict[str, Any]:
    """Auto-compute warmup_steps / decay_until_step for step-based schedulers.

    Called after the DataLoader is built so ``steps_per_epoch`` (``len(train_loader)``)
    and ``epochs`` (from ``trainer.epochs``) are known at runtime.

    Behaviour:

    - ``ExponentialDecrease``
        * ``total_steps`` missing / ``None`` / ``"auto"`` → full optimizer-step
          horizon when ``per_step=True``, otherwise total epochs
        * explicit positive integer → used as-is

    - ``decay_until_step``
        * missing / ``None`` / ``"auto"`` → ``epochs * steps_per_epoch``
          (``per_step=True``) or ``epochs`` (``per_step=False``)
        * explicit integer → used as-is (warns if it differs from the total)
    - ``warmup_steps``
        * missing / ``None`` / ``"auto"`` → ``max(1, int(warmup_ratio * total))``
        * explicit integer → used as-is

    The ``warmup_ratio`` is read from ``sch_cfg["auto"]["warmup_ratio"]``
    (default 0.05 = 5 % of the total).

    ``per_step`` must mirror ``trainer.lr_schedule_per_step``: when False the
    scheduler is stepped once per epoch, so ``warmup_steps`` / ``decay_until_step``
    are in **epoch** units (see ``LinearWarmupCosineAnnealing`` docstring) and the
    total is ``epochs``, not ``epochs * steps_per_epoch``.

    Args:
        sch_cfg: The raw ``cfg["scheduler"]`` dict loaded from YAML.
        steps_per_epoch: Number of optimizer steps per epoch
            (``len(train_loader) // accumulation_steps``).
        epochs: Total training epochs (from ``trainer.epochs``).
        rank: Current process rank; logs are emitted only on rank 0.
        per_step: True when the scheduler steps every optimizer step
            (``trainer.lr_schedule_per_step: true``); False means epoch units.

    Returns:
        A **new** ``args`` dict that should be passed to the scheduler constructor.
        The original ``sch_cfg`` is not modified.
    """
    sch_type = sch_cfg.get("type", "")
    if sch_type not in {"LinearWarmupCosineAnnealing", "ExponentialDecrease"}:
        # Other schedulers don't use these keys; return args unchanged.
        return dict(sch_cfg.get("args", {}))

    args = dict(sch_cfg.get("args", {}))
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    if per_step and (not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0):
        raise ValueError(
            f"steps_per_epoch must be a positive integer for per-step scheduling, got {steps_per_epoch!r}"
        )

    # per_step=False 时 scheduler 每 epoch 只 step 一次，warmup/decay 以 epoch 为单位
    if per_step:
        total_steps = epochs * steps_per_epoch
    else:
        total_steps = epochs
    _unit = "step" if per_step else "epoch"

    if sch_type == "ExponentialDecrease":
        raw_total = args.get("total_steps")
        if raw_total is None or raw_total == "auto":
            args["total_steps"] = total_steps
            if rank == 0:
                logger.info(
                    "[Scheduler] auto total_steps = %d %ss for ExponentialDecrease",
                    total_steps,
                    _unit,
                )
        elif (
            not isinstance(raw_total, int)
            or isinstance(raw_total, bool)
            or raw_total <= 0
        ):
            raise ValueError(
                "total_steps must be a positive integer or 'auto', "
                f"got {raw_total!r}"
            )
        elif raw_total != total_steps and rank == 0:
            logger.warning(
                "[Scheduler] total_steps=%d does not match planned total_%ss=%d",
                raw_total,
                _unit,
                total_steps,
            )
        return args

    auto = sch_cfg.get("auto", {})
    warmup_ratio = float(auto.get("warmup_ratio", 0.05))
    if not math.isfinite(warmup_ratio) or not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"warmup_ratio must be finite and in [0, 1], got {warmup_ratio}")

    # ---------- decay_until_step ----------
    raw_decay = args.get("decay_until_step")
    if raw_decay is None or raw_decay == "auto":
        args["decay_until_step"] = total_steps
        if rank == 0:
            if per_step:
                logger.info(
                    "[Scheduler] auto decay_until_step = %d (epochs=%d * steps_per_epoch=%d)",
                    total_steps, epochs, steps_per_epoch,
                )
            else:
                logger.info(
                    "[Scheduler] auto decay_until_step = %d epochs (lr_schedule_per_step=false, epochs=%d)",
                    total_steps, epochs,
                )
    else:
        if not isinstance(raw_decay, int) or isinstance(raw_decay, bool):
            raise ValueError(
                f"decay_until_step must be a positive integer or 'auto', got {raw_decay!r}"
            )
        decay_val = raw_decay
        if decay_val != total_steps and rank == 0:
            logger.warning(
                "[Scheduler] decay_until_step=%d does not match total_%ss=%d "
                "(epochs=%d, steps_per_epoch=%d, lr_schedule_per_step=%s). Cosine will %s "
                "before training ends. "
                "Set decay_until_step to null or 'auto' to auto-compute.",
                decay_val,
                _unit,
                total_steps,
                epochs,
                steps_per_epoch,
                per_step,
                "plateau at min_lr" if decay_val < total_steps else "still be decaying",
            )

    # ---------- warmup_steps ----------
    raw_warmup = args.get("warmup_steps")
    if raw_warmup is None or raw_warmup == "auto":
        computed_warmup = max(1, int(warmup_ratio * total_steps))
        args["warmup_steps"] = computed_warmup
        if rank == 0:
            logger.info(
                "[Scheduler] auto warmup_steps = %d (ratio=%.2f%% of total_%ss=%d)",
                computed_warmup, warmup_ratio * 100, _unit, total_steps,
            )
    else:
        if not isinstance(raw_warmup, int) or isinstance(raw_warmup, bool):
            raise ValueError(
                f"warmup_steps must be a non-negative integer or 'auto', got {raw_warmup!r}"
            )
        warmup_val = raw_warmup
        if warmup_val >= total_steps and rank == 0:
            logger.warning(
                "[Scheduler] warmup_steps=%d >= total_%ss=%d. "
                "LR remains in linear warmup through training, then reaches min_lr on the next step. "
                "Set warmup_steps to null or 'auto' to auto-compute.",
                warmup_val, _unit, total_steps,
            )

    return args


@SCHEDULER_REGISTRY.register("LinearWarmupCosineAnnealing")
class LinearWarmupCosineAnnealing(_LRScheduler):
    """Linear warmup + cosine annealing scheduler.

    LR schedule: 0 -> max_lr (warmup) -> min_lr (cosine decay) -> min_lr (hold).
    Note: warmup_steps and decay_until_step use steps when lr_schedule_per_step=true,
          epochs when lr_schedule_per_step=false.
    """
    def __init__(self, optimizer, warmup_steps=5000, decay_until_step=200000,
                 max_lr=1e-3, min_lr=1e-5, last_epoch=-1,
                 zero_based_warmup: bool = False):
        _validate_optimizer_base_lrs(optimizer)
        if type(zero_based_warmup) is not bool:
            raise ValueError(
                f"zero_based_warmup must be bool, got {zero_based_warmup!r}"
            )
        self.zero_based_warmup = zero_based_warmup
        if (
            not isinstance(warmup_steps, int)
            or isinstance(warmup_steps, bool)
            or warmup_steps < 0
        ):
            raise ValueError(f"warmup_steps must be a non-negative integer, got {warmup_steps!r}")
        if (
            not isinstance(decay_until_step, int)
            or isinstance(decay_until_step, bool)
            or decay_until_step <= 0
        ):
            raise ValueError(
                f"decay_until_step must be a positive integer, got {decay_until_step!r}"
            )
        if warmup_steps > decay_until_step:
            raise ValueError(
                "warmup_steps must not exceed decay_until_step, got "
                f"{warmup_steps} > {decay_until_step}"
            )
        if not all(math.isfinite(float(lr)) and float(lr) >= 0.0 for lr in (max_lr, min_lr)):
            raise ValueError("max_lr and min_lr must be finite and non-negative")
        if float(min_lr) > float(max_lr):
            raise ValueError(f"min_lr must not exceed max_lr, got {min_lr} > {max_lr}")
        self.warmup_steps = warmup_steps; self.decay_until_step = decay_until_step
        self.max_lr = max_lr; self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        s = max(self.last_epoch, 0)
        if s < self.warmup_steps:
            # 保留 getattr 缺省：与 _LRScheduler 反序列化/测试替身路径兼容。
            if getattr(self, "zero_based_warmup", False):
                # 官方 LinearWarmupCosineAnnealingLR：step 0 的 lr=0。
                lr = self.max_lr * s / max(self.warmup_steps, 1)
            else:
                lr = self.max_lr * (s + 1) / max(self.warmup_steps, 1)
        elif s < self.decay_until_step:
            p = (s - self.warmup_steps) / max(self.decay_until_step - self.warmup_steps, 1)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * p))
        else:
            lr = self.min_lr
        # Preserve multi-param-group ratios: scale by each group's base_lr
        base0 = self.base_lrs[0] if self.base_lrs else 1.0
        return [lr * (base_lr / base0) for base_lr in self.base_lrs]


@SCHEDULER_REGISTRY.register("ExponentialDecrease")
class ExponentialDecrease(_LRScheduler):
    """按 WeSep 语义在固定 horizon 内从 ``initial_lr`` 指数衰减到 ``final_lr``。

    ``trainer.lr_schedule_per_step=true`` 时，``total_steps`` 由配置解析层按完整
    optimizer-step 数自动注入。初始 optimizer step 使用 ``initial_lr``；完成
    ``total_steps`` 次 scheduler step 后精确到达 ``final_lr``。
    """

    def __init__(
        self,
        optimizer,
        initial_lr: float,
        final_lr: float,
        total_steps: int,
        last_epoch: int = -1,
    ):
        _validate_optimizer_base_lrs(optimizer)
        for name, value in (("initial_lr", initial_lr), ("final_lr", final_lr)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if float(final_lr) > float(initial_lr):
            raise ValueError(
                "final_lr must not exceed initial_lr for ExponentialDecrease, "
                f"got {final_lr} > {initial_lr}"
            )
        if (
            not isinstance(total_steps, int)
            or isinstance(total_steps, bool)
            or total_steps <= 0
        ):
            raise ValueError(
                f"total_steps must be a positive integer, got {total_steps!r}"
            )
        for index, group in enumerate(optimizer.param_groups):
            if not math.isclose(
                float(group["lr"]), float(initial_lr), rel_tol=0.0, abs_tol=0.0
            ):
                raise ValueError(
                    "optimizer lr must equal initial_lr for exact WeSep reproduction, "
                    f"group {index}: {group['lr']} != {initial_lr}"
                )
        self.initial_lr = float(initial_lr)
        self.final_lr = float(final_lr)
        self.total_steps = total_steps
        self._log_ratio = math.log(self.final_lr / self.initial_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        position = min(max(self.last_epoch, 0), self.total_steps)
        lr = self.initial_lr * math.exp(
            (position / self.total_steps) * self._log_ratio
        )
        return [lr for _ in self.optimizer.param_groups]

    def rebase(self, optimizer=None) -> None:
        """以当前 optimizer LR 重建指数曲线，同时保留 scheduler progress。"""
        optimizer = self.optimizer if optimizer is None else optimizer
        current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if not current_lrs or any(
            not math.isfinite(lr) or lr <= 0.0 for lr in current_lrs
        ):
            raise ValueError("ExponentialDecrease rebase requires finite positive optimizer LR")
        scale = current_lrs[0] / self.initial_lr
        self.initial_lr = current_lrs[0]
        self.final_lr *= scale
        if self.final_lr > self.initial_lr:
            raise ValueError(
                "ExponentialDecrease rebase would make final_lr exceed initial_lr"
            )
        self._log_ratio = math.log(self.final_lr / self.initial_lr)
        self.base_lrs = list(current_lrs)
        self._last_lr = list(current_lrs)
        for group, lr in zip(optimizer.param_groups, current_lrs):
            group["initial_lr"] = lr


@SCHEDULER_REGISTRY.register("CosineAnnealingWarmupRestarts")
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """Cosine annealing + linear warmup + cycle restarts.

    Per cycle: warmup -> cosine decay -> restart.
    Cycle length can multiply (cycle_mult > 1).

    Args:
        first_cycle_steps: Total steps in first cycle (including warmup).
        cycle_mult:        Cycle length multiplier per restart (default 1.0).
        warmup_steps:      Warmup steps per cycle.
        max_lr:            Peak LR (warmup target).
        min_lr:            Valley LR (cosine bottom).
        gamma:             max_lr decay factor per restart (default 1.0, no decay).
    """

    def __init__(self, optimizer, first_cycle_steps: int = 50000,
                 cycle_mult: float = 1.0, warmup_steps: int = 500,
                 max_lr: float = 2e-3, min_lr: float = 1e-6,
                 gamma: float = 1.0, last_epoch: int = -1):
        _validate_optimizer_base_lrs(optimizer)
        if (
            not isinstance(first_cycle_steps, int)
            or isinstance(first_cycle_steps, bool)
            or first_cycle_steps <= 0
        ):
            raise ValueError(
                f"first_cycle_steps must be a positive integer, got {first_cycle_steps!r}"
            )
        if (
            not isinstance(warmup_steps, int)
            or isinstance(warmup_steps, bool)
            or not 0 <= warmup_steps < first_cycle_steps
        ):
            raise ValueError(
                "warmup_steps must be an integer in [0, first_cycle_steps), got "
                f"{warmup_steps!r} for first_cycle_steps={first_cycle_steps}"
            )
        if not math.isfinite(float(cycle_mult)) or float(cycle_mult) < 1.0:
            raise ValueError(f"cycle_mult must be finite and >= 1.0, got {cycle_mult}")
        if not math.isfinite(float(gamma)) or not 0.0 < float(gamma) <= 1.0:
            raise ValueError(f"gamma must be finite and in (0, 1], got {gamma}")
        if not all(math.isfinite(float(lr)) and float(lr) >= 0.0 for lr in (max_lr, min_lr)):
            raise ValueError("max_lr and min_lr must be finite and non-negative")
        if float(min_lr) > float(max_lr):
            raise ValueError(f"min_lr must not exceed max_lr, got {min_lr} > {max_lr}")
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        super().__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        # Preserve optimizer's original base_lrs instead of overwriting with max_lr
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

    def get_lr(self):
        if self.step_in_cycle < 0:
            return [self.min_lr] * len(self.base_lrs)

        if self.step_in_cycle < self.warmup_steps:
            # +1 so step 0 starts above min_lr instead of min_lr
            ratio = (self.step_in_cycle + 1) / max(self.warmup_steps, 1)
            cur_max = self.max_lr * (self.gamma ** self.cycle)
            lr = self.min_lr + (cur_max - self.min_lr) * ratio
        else:
            progress = (self.step_in_cycle - self.warmup_steps) / max(
                self.cur_cycle_steps - self.warmup_steps, 1
            )
            cur_max = self.max_lr * (self.gamma ** self.cycle)
            lr = self.min_lr + 0.5 * (cur_max - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        # Preserve multi-param-group ratios: scale by each group's base_lr
        base0 = self.base_lrs[0] if self.base_lrs else 1.0
        return [lr * (base_lr / base0) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                self.cur_cycle_steps = int(
                    self.cur_cycle_steps * self.cycle_mult
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.cycle = int(epoch / self.first_cycle_steps)
                    self.step_in_cycle = epoch % self.first_cycle_steps
                else:
                    n = int(math.log(
                        (epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                        self.cycle_mult,
                    ))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = int(
                        self.first_cycle_steps * self.cycle_mult ** n
                    )
            else:
                self.cycle = 0
                self.step_in_cycle = epoch

        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


@SCHEDULER_REGISTRY.register("ExponentialLR")
class _Exp:
    def __new__(cls, optimizer, **kw):
        from torch.optim.lr_scheduler import ExponentialLR
        return ExponentialLR(optimizer, **kw)


@SCHEDULER_REGISTRY.register("StepLR")
class _StepLR:
    def __new__(cls, optimizer, **kw):
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, **kw)


@SCHEDULER_REGISTRY.register("CosineAnnealingLR")
class _Cos:
    def __new__(cls, optimizer, **kw):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kw)


@SCHEDULER_REGISTRY.register("ReduceLROnPlateau")
class _Plateau:
    def __new__(cls, optimizer, **kw):
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer, **kw)
