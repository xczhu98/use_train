"""TSE 训练目标的纯配置合同；entry 与 Trainer 使用同一解析器。"""

from __future__ import annotations

import math
from collections.abc import Mapping


def resolve_dual_self_estimated_objective(trainer: Mapping) -> tuple[float, float] | None:
    """只有 canonical Trainer 配置能启用双路监督，缺省为关闭。"""
    if not isinstance(trainer, Mapping):
        raise TypeError("trainer must be a mapping")
    if "objectives" not in trainer:
        return None
    objectives = trainer["objectives"]
    if not isinstance(objectives, Mapping):
        raise TypeError("trainer.objectives must be a mapping")
    unknown = set(objectives) - {"dual_self_estimated"}
    if unknown:
        raise KeyError(f"Unknown trainer.objectives keys: {sorted(map(str, unknown))}")
    if "dual_self_estimated" not in objectives:
        return None
    raw = objectives["dual_self_estimated"]
    path = "trainer.objectives.dual_self_estimated"
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path} must be a mapping")
    unknown = set(raw) - {"enabled", "original_loss_weight", "self_loss_weight"}
    if unknown:
        raise KeyError(f"Unknown {path} keys: {sorted(map(str, unknown))}")
    enabled = raw.get("enabled", True)
    if type(enabled) is not bool:
        raise TypeError(f"{path}.enabled must be a bool")
    weights = []
    for name, default in (("original_loss_weight", 0.4), ("self_loss_weight", 0.6)):
        value = raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{path}.{name} must be a number")
        try:
            weight = float(value)
        except OverflowError as exc:
            raise ValueError(f"{path}.{name} must be finite and positive") from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"{path}.{name} must be finite and positive")
        weights.append(weight)
    if not enabled:
        if {"original_loss_weight", "self_loss_weight"} & set(raw):
            raise ValueError(f"{path}: disabled objective must omit unused weights")
        return None
    if trainer.get("ssa_prob", 0.0) != 0.0:
        raise ValueError("dual self-estimated objective cannot be combined with trainer.ssa_prob")
    return weights[0], weights[1]


def validate_tse_objective_config(config: Mapping) -> None:
    """共享入口经 TaskDescriptor 调用，不在 core 中维护 TSE 字段。"""
    if not isinstance(config, Mapping):
        raise TypeError("TSE objective config must be a mapping")
    if "model" in config and not isinstance(config["model"], Mapping):
        raise TypeError("model must be a mapping")
    model = config.get("model", {})
    if "args" in model and not isinstance(model["args"], Mapping):
        raise TypeError("model.args must be a mapping")
    args = model.get("args", {})
    if model.get("type") == "TSE_BSRNN_Multi":
        raise ValueError("TSE_BSRNN_Multi is removed; use TSE_BSRNN_SPK with trainer.objectives.dual_self_estimated")
    removed = {"original_loss_weight", "self_loss_weight", "ssa_prob"} & set(args)
    if removed:
        raise ValueError(f"model.args keys are removed: {sorted(removed)}; configure the training objective under trainer")
    weights = resolve_dual_self_estimated_objective(config.get("trainer", {}))
    if weights is not None:
        if args.get("joint_training", True) is not True:
            raise ValueError("dual self-estimated objective requires joint_training=true")
        if model.get("type") == "TSE_Modular":
            raise ValueError("TSE_Modular does not support dual self-estimated enrollment")


def validate_dual_objective_model(model, weights: tuple[float, float] | None) -> None:
    """运行时要求显式能力声明，已移除的模型属性始终报错。"""
    if hasattr(model, "dual_self_estimated_loss_weights"):
        raise ValueError("model.dual_self_estimated_loss_weights is removed; use trainer.objectives.dual_self_estimated")
    if weights is not None and getattr(model, "supports_dual_self_estimated", False) is not True:
        raise ValueError("dual self-estimated objective requires model.supports_dual_self_estimated=True")
