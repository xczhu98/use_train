"""Shared model construction from config dict.

Used by both SE and AEC training entries. Task-specific model maps are
registered via ``MODEL_REGISTRY.set_lazy_map()`` in each task's
``models/__init__.py``.
"""

from __future__ import annotations

import importlib

from core.stft import (
    STFT_CONFIG_KEYS,
    inject_stft_defaults,
    resolve_stft_config,
)
from core.utils.registry import MODEL_REGISTRY


def _inject_effective_stft_context(
    args: dict, stft_cfg: dict | None
) -> dict:
    """向模型参数注入框架实际使用的 STFT 上下文，并拒绝显式冲突。"""
    if not stft_cfg or "n_fft" not in stft_cfg or "hop_length" not in stft_cfg:
        return inject_stft_defaults(args, stft_cfg)

    effective_stft = resolve_stft_config(stft_cfg)
    for key in STFT_CONFIG_KEYS:
        if key not in args:
            continue
        explicit_value = args[key]
        context_value = effective_stft[key]
        if key == "window" and isinstance(explicit_value, str):
            explicit_value = explicit_value.lower()
        if explicit_value != context_value:
            raise ValueError(
                f"{key} context conflict: model args.{key}={args[key]!r}, "
                f"effective stft.{key}={context_value!r}"
            )
    return inject_stft_defaults(args, effective_stft)


def build_model(cfg: dict, stft_cfg: dict | None = None) -> "torch.nn.Module":
    """Instantiate a model from config dict.

    Supports two modes:
    - type + args: cfg["type"] uses MODEL_REGISTRY (lazy-load registered).
    - path + args: cfg["path"] dynamically imports the class.

    完整 ``stft_cfg``（至少声明 ``n_fft`` / ``hop_length``）先解析为框架实际
    使用的五项有效参数，再注入 ``model.args``；模型显式 STFT 参数仅在与该上下文
    一致时允许。partial ``stft_cfg`` 保持历史的逐项默认注入行为。
    """
    args = _inject_effective_stft_context(dict(cfg.get("args", {})), stft_cfg)
    return resolve_model_class(cfg)(**args)


def resolve_model_class(cfg: dict):
    """解析实际模型类，供构造前的 geometry 参数校验和正式 build 共用。"""
    if "path" in cfg:
        path = cfg["path"]
        if not path or "." not in path:
            raise KeyError("[model] config 'path' must be a dotted class path (e.g. 'tasks.se.models.discriminative.gtcrn.GTCRN')")
        module_path, _, class_name = path.rpartition(".")
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls
    if "type" not in cfg:
        raise KeyError("[model] config dict must contain 'type' or 'path'")
    model_type = cfg["type"]
    MODEL_REGISTRY.ensure_registered(model_type)
    return MODEL_REGISTRY.get(model_type)
