"""Core model construction and task registration facade."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping

from core.utils.registry import MODEL_REGISTRY

from tasks.registry import get_task, iter_tasks


def _missing_optional_dependency(
    error: ImportError, declared: frozenset[str]
) -> str | None:
    """从显式异常链提取声明过的缺失 optional dependency 顶层包名。"""
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, ModuleNotFoundError) and current.name:
            missing_root = current.name.split(".", 1)[0]
            return missing_root if missing_root in declared else None
        current = current.__cause__
    return None


def register_lazy_models(
    mapping: Mapping[str, str],
    *,
    task: str,
    only: str | None = None,
    logger: logging.Logger | None = None,
    optional_dependencies: Mapping[str, frozenset[str]] | None = None,
) -> None:
    """按任务注册 lazy model；批量模式只隔离显式声明的可选依赖缺失。"""
    optional_dependencies = optional_dependencies or {}
    unknown_optional_models = set(optional_dependencies) - set(mapping)
    if unknown_optional_models:
        raise ValueError(
            "optional dependency declarations reference unknown model types: "
            f"{sorted(unknown_optional_models)}"
        )
    if only is not None:
        if only not in mapping:
            raise KeyError(
                f"Unknown {task.upper()} model type {only!r}. "
                f"Known keys: {list(mapping)}"
            )
        MODEL_REGISTRY.ensure_registered(only)
        return

    for model_type in mapping:
        try:
            MODEL_REGISTRY.ensure_registered(model_type)
        except ImportError as exc:
            declared = optional_dependencies.get(model_type, frozenset())
            missing_root = _missing_optional_dependency(exc, declared)
            if missing_root is None:
                raise
            if logger is not None:
                logger.warning(
                    "Skipping %s model registration for %r "
                    "(optional dependency %r is missing)",
                    task.upper(),
                    model_type,
                    missing_root,
                )


def register_task_models(task: str, only: str | None = None) -> None:
    """通过统一 facade 注册指定任务的模型，且仅按需导入任务模型包。"""
    try:
        descriptor = get_task(task)
    except KeyError:
        raise KeyError(
            f"Unknown model task {task!r}. Known tasks: {[item.name for item in iter_tasks()]}"
        ) from None

    module = importlib.import_module(descriptor.model_registry_module)
    register = getattr(module, descriptor.model_registrar)
    register(only=only)


__all__ = ["register_lazy_models", "register_task_models"]
