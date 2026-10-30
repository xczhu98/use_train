"""Validation objective metrics 配置的共享 fail-fast 契约。"""

from __future__ import annotations

import math
from collections.abc import Mapping

_REMOVED_TRAINER_MOS_KEYS = (
    "mos_models",
    "mos_interval",
    "mos_batch_size",
)
_REMOVED_TRAINER_MOS_METRIC_PREFIXES = (
    "DNSMOS",
    "NISQA",
    "UTMOS",
    "SCOREQ",
    "AECMOS",
)
_OFFLINE_MOS_GUIDANCE = (
    "Trainer no longer computes MOS; use evaluate/eval_pipeline_*.py "
    "--audio_quality or evaluate/eval_single_mos.py."
)

# 共享 Trainer 与当前 SE/AEC 专用 Trainer 实际消费的配置键；新增参数必须先由
# 对应 Trainer 显式消费，再同步更新此集合，禁止 YAML 拼写错误静默落空。
_TRAINER_KEYS = frozenset(
    {
        "accumulation_steps",
        "adversarial_weight",
        "clear_cache_each_epoch",
        "curve_dpi",
        "curve_figsize",
        "discriminator_steps",
        "early_stop_patience",
        "epochs",
        "exp_path",
        "generator_steps",
        "gradient_clip_value",
        "initial_validation",
        "lamda_adv",
        "lamda_feat",
        "lamda_recons",
        "log_interval",
        "log_interval_discriminator",
        "log_step_sub_losses",
        "loss_com_weight",
        "loss_mag_weight",
        "loss_pha_weight",
        "loss_stft_weight",
        "loss_time_weight",
        "lr_schedule_per_step",
        "max_steps",
        "max_val_batches",
        "pesq_sr",
        "save_checkpoint_batch_interval",
        "save_checkpoint_interval",
        "save_curve_interval",
        "save_top_k",
        "ssa_prob",
        "strict_checks",
        "type",
        "objectives",
        "official_warm_start_ckpt",
        "output_clamp",
        "use_lsnr_loss",
        "use_pesq_injection",
        "val_audio",
        "val_metrics",
        "validation_epoch_interval",
        "validation_steps_interval",
    }
)
_VAL_METRICS_KEYS = frozenset(
    {
        "enabled",
        "interval",
        "max_utterances",
        "max_utterances_erle",
        "max_utterances_metrics",
        "metrics",
        "metrics_chunk_size",
        "min_coverage",
        "metrics_workers",
        "parallel_intrusive",
        "strict",
    }
)
_TRAINER_BOOL_KEYS = frozenset(
    {
        "clear_cache_each_epoch",
        "lr_schedule_per_step",
        "strict_checks",
        "use_lsnr_loss",
        "use_pesq_injection",
    }
)
_TRAINER_POSITIVE_INT_KEYS = frozenset(
    {
        "accumulation_steps",
        "curve_dpi",
        "discriminator_steps",
        "epochs",
        "generator_steps",
        "log_interval",
        "pesq_sr",
    }
)
_TRAINER_NONNEGATIVE_INT_KEYS = frozenset(
    {
        "early_stop_patience",
        "log_interval_discriminator",
        "log_step_sub_losses",
        "max_steps",
        "max_val_batches",
        "save_checkpoint_batch_interval",
        "save_checkpoint_interval",
        "save_curve_interval",
        "save_top_k",
        "validation_epoch_interval",
        "validation_steps_interval",
    }
)
_TRAINER_NONNEGATIVE_NUMBER_KEYS = frozenset(
    {
        "adversarial_weight",
        "gradient_clip_value",
        "lamda_adv",
        "lamda_feat",
        "lamda_recons",
        "loss_com_weight",
        "loss_mag_weight",
        "loss_pha_weight",
        "loss_stft_weight",
        "loss_time_weight",
    }
)
_VAL_METRICS_BOOL_KEYS = frozenset({"enabled", "parallel_intrusive", "strict"})
_VAL_METRICS_POSITIVE_INT_KEYS = frozenset(
    {"interval", "max_utterances", "max_utterances_erle", "metrics_chunk_size"}
)
_VAL_METRICS_NONNEGATIVE_INT_KEYS = frozenset({"metrics_workers"})
_DEFAULT_TRAINER_METRICS = ("SI-SDR", "PESQ", "ESTOI")
_SUPPORTED_TRAINER_METRICS = frozenset(
    {*_DEFAULT_TRAINER_METRICS, "STOI", "ERLE"}
)

def _validate_bool(mapping: Mapping[str, object], key: str, path: str) -> None:
    value = mapping.get(key)
    if key in mapping and not isinstance(value, bool):
        raise TypeError(f"{path}.{key} must be a boolean")


def _validate_task_objectives(config: Mapping[str, object]) -> None:
    """通过静态任务描述路由目标配置，未登记的任务拒绝该字段。"""
    trainer = config["trainer"]
    if "objectives" not in trainer:
        return
    if not isinstance(trainer["objectives"], Mapping):
        raise TypeError("trainer.objectives must be a mapping")
    from importlib import import_module
    from tasks.registry import get_task_for_data_mode

    data = config.get("data")
    if not isinstance(data, Mapping) or "mode" not in data:
        raise ValueError("trainer.objectives requires explicit data.mode")
    descriptor = get_task_for_data_mode(data["mode"])
    validator_path = descriptor.trainer_objective_validator
    if validator_path is None:
        raise ValueError(f"task {descriptor.name!r} does not support trainer.objectives")
    module, symbol = validator_path.split(":", 1)
    getattr(import_module(module), symbol)(config)


def _validate_int(
    mapping: Mapping[str, object],
    key: str,
    path: str,
    *,
    minimum: int,
    allow_none: bool = False,
) -> None:
    if key not in mapping:
        return
    value = mapping[key]
    if allow_none and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}.{key} must be an integer")
    if value < minimum:
        raise ValueError(f"{path}.{key} must be >= {minimum}")


def validate_trainer_config(config: Mapping[str, object]) -> None:
    """严格校验 Trainer 配置结构与未知参数。"""
    trainer = config.get("trainer")
    if not isinstance(trainer, Mapping):
        raise TypeError("Config 'trainer' section must be a mapping")
    unknown = sorted(set(trainer) - _TRAINER_KEYS)
    if unknown:
        raise KeyError(f"Unknown trainer config keys: {unknown}")
    for key in _TRAINER_BOOL_KEYS:
        _validate_bool(trainer, key, "trainer")
    for key in _TRAINER_POSITIVE_INT_KEYS:
        _validate_int(trainer, key, "trainer", minimum=1)
    for key in _TRAINER_NONNEGATIVE_INT_KEYS:
        _validate_int(
            trainer,
            key,
            "trainer",
            minimum=0,
            allow_none=key == "max_steps",
        )
    for key in _TRAINER_NONNEGATIVE_NUMBER_KEYS:
        if key not in trainer:
            continue
        value = trainer[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"trainer.{key} must be a number")
        if not math.isfinite(float(value)):
            raise ValueError(f"trainer.{key} must be finite")
        if value < 0:
            raise ValueError(f"trainer.{key} must be >= 0")
    _validate_task_objectives(config)
    for key in ("type", "exp_path"):
        value = trainer.get(key)
        if key in trainer and (
            not isinstance(value, str) or not value.strip()
        ):
            raise TypeError(f"trainer.{key} must be a non-empty string")
    ssa_prob = trainer.get("ssa_prob")
    if "ssa_prob" in trainer and (
        isinstance(ssa_prob, bool)
        or not isinstance(ssa_prob, (int, float))
        or not 0 <= ssa_prob <= 1
    ):
        raise ValueError("trainer.ssa_prob must be a number in [0, 1]")
    curve_figsize = trainer.get("curve_figsize")
    if "curve_figsize" in trainer and (
        not isinstance(curve_figsize, (list, tuple))
        or len(curve_figsize) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in curve_figsize
        )
    ):
        raise ValueError("trainer.curve_figsize must contain two positive numbers")

    if "val_metrics" not in trainer:
        return
    val_metrics = trainer["val_metrics"]
    if not isinstance(val_metrics, Mapping):
        raise TypeError("Config 'trainer.val_metrics' section must be a mapping")
    unknown_metrics = sorted(set(val_metrics) - _VAL_METRICS_KEYS)
    if unknown_metrics:
        raise KeyError(
            f"Unknown trainer.val_metrics config keys: {unknown_metrics}"
        )
    for key in _VAL_METRICS_BOOL_KEYS:
        _validate_bool(val_metrics, key, "trainer.val_metrics")
    for key in _VAL_METRICS_POSITIVE_INT_KEYS:
        _validate_int(val_metrics, key, "trainer.val_metrics", minimum=1)
    _validate_int(
        val_metrics,
        "max_utterances_metrics",
        "trainer.val_metrics",
        minimum=1,
        allow_none=True,
    )
    for key in _VAL_METRICS_NONNEGATIVE_INT_KEYS:
        _validate_int(
            val_metrics,
            key,
            "trainer.val_metrics",
            minimum=0,
            allow_none=key == "metrics_workers",
        )
    metrics = val_metrics.get("metrics", _DEFAULT_TRAINER_METRICS)
    if (
        not isinstance(metrics, (list, tuple))
        or not all(isinstance(metric, str) and metric.strip() for metric in metrics)
    ):
        raise TypeError(
            "trainer.val_metrics.metrics must be a list/tuple of non-empty strings"
        )
    if len(metrics) != len(set(metrics)):
        raise ValueError("trainer.val_metrics.metrics must not contain duplicates")
    unknown_metric_names = sorted(set(metrics) - _SUPPORTED_TRAINER_METRICS)
    if unknown_metric_names:
        raise ValueError(
            "Unknown trainer.val_metrics.metrics values: "
            f"{unknown_metric_names}; supported: {sorted(_SUPPORTED_TRAINER_METRICS)}"
        )

    min_coverage = val_metrics.get("min_coverage", {})
    if not isinstance(min_coverage, Mapping):
        raise TypeError("trainer.val_metrics.min_coverage must be a mapping")
    if not all(
        isinstance(metric, str) and metric.strip() for metric in min_coverage
    ):
        raise TypeError(
            "trainer.val_metrics.min_coverage keys must be non-empty strings"
        )
    unknown_coverage_metrics = sorted(set(min_coverage) - set(metrics))
    if unknown_coverage_metrics:
        raise ValueError(
            "trainer.val_metrics.min_coverage contains metrics not enabled in "
            f"trainer.val_metrics.metrics: {unknown_coverage_metrics}"
        )
    for metric, threshold in min_coverage.items():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError(
                f"trainer.val_metrics.min_coverage.{metric} must be a number"
            )
        if not math.isfinite(float(threshold)):
            raise ValueError(
                f"trainer.val_metrics.min_coverage.{metric} must be finite"
            )
        if not 0 <= float(threshold) <= 1:
            raise ValueError(
                f"trainer.val_metrics.min_coverage.{metric} must be in [0, 1]"
            )


def validate_validation_metric_runtime_result(
    metric: str,
    *,
    count: float,
    eligible: float,
    strict: bool,
    min_coverage: float | None = None,
    error: BaseException | str | None = None,
) -> bool:
    """执行单项 validation metric 的统一 runtime 结果契约。

    worker/runtime 异常和非法统计值始终 fail-closed，不受 ``strict`` 影响；合法的
    ``count == 0`` 仅在 ``strict=true`` 时失败，否则返回 ``False`` 供调用方跳过。
    未配置 ``min_coverage`` 时维持历史兼容语义：任意正 count 即可；显式阈值未满足
    则作为配置的 coverage gate 失败，与 ``strict`` 无关。
    """
    if error is not None:
        if isinstance(error, BaseException):
            raise RuntimeError(f"{metric} validation failed: {error}") from error
        raise RuntimeError(f"{metric} validation failed: {error}")
    if not math.isfinite(float(count)) or count < 0:
        raise RuntimeError(f"{metric} validation returned invalid count: {count!r}")
    needs_coverage = min_coverage is not None or count > 0
    if needs_coverage and (
        not math.isfinite(float(eligible)) or eligible <= 0 or count > eligible
    ):
        raise RuntimeError(
            f"{metric} validation returned invalid eligible/count: "
            f"eligible={eligible!r}, count={count!r}"
        )
    if min_coverage is not None:
        coverage = count / eligible
        if coverage < min_coverage:
            raise RuntimeError(
                f"{metric} validation coverage {coverage:.6f} is below configured "
                f"minimum {min_coverage:.6f} ({count:g}/{eligible:g})"
            )
    if count == 0:
        if strict:
            raise RuntimeError(
                f"val_metrics.strict=True but {metric} has no valid global result"
            )
        return False
    return True


def _is_removed_trainer_mos_metric(name: object) -> bool:
    normalized = str(name).strip().upper()
    return any(
        normalized == prefix or normalized.startswith(f"{prefix}_")
        for prefix in _REMOVED_TRAINER_MOS_METRIC_PREFIXES
    )


def validate_no_trainer_mos_config(config: Mapping[str, object]) -> None:
    """拒绝已移除的 Trainer MOS 配置，并指向离线评估入口。"""
    trainer = config.get("trainer") or {}
    if not isinstance(trainer, Mapping):
        return
    val_metrics = trainer.get("val_metrics") or {}
    if not isinstance(val_metrics, Mapping):
        return
    for key in _REMOVED_TRAINER_MOS_KEYS:
        if key in val_metrics:
            raise ValueError(
                f"trainer.val_metrics.{key} is no longer supported: "
                f"{_OFFLINE_MOS_GUIDANCE}"
            )

    metrics = val_metrics.get("metrics") or ()
    if not isinstance(metrics, (list, tuple)):
        return
    for metric in metrics:
        if _is_removed_trainer_mos_metric(metric):
            raise ValueError(
                "trainer.val_metrics.metrics contains removed MOS metric "
                f"{metric!r}: {_OFFLINE_MOS_GUIDANCE}"
            )
