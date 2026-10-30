"""任务插件注册与 data.mode 路由。"""

from __future__ import annotations

import importlib

from core.data.constants import normalize_data_mode
from tasks.descriptors import TASK_DESCRIPTORS, TaskDescriptor

_BY_NAME = {descriptor.name: descriptor for descriptor in TASK_DESCRIPTORS}
_BY_MODE = {
    mode: descriptor
    for descriptor in TASK_DESCRIPTORS
    for mode in descriptor.data_modes
}

if len(_BY_NAME) != len(TASK_DESCRIPTORS):
    raise RuntimeError("duplicate task descriptor name")
if sum(len(descriptor.data_modes) for descriptor in TASK_DESCRIPTORS) != len(_BY_MODE):
    raise RuntimeError("duplicate task data.mode")


def get_task(name: str) -> TaskDescriptor:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"Unknown task {name!r}; known tasks: {sorted(_BY_NAME)}") from None


def get_task_for_data_mode(mode: str) -> TaskDescriptor:
    mode = normalize_data_mode(mode).task_mode
    try:
        return _BY_MODE[mode]
    except KeyError:
        raise ValueError(
            f"Unknown data.mode {mode!r}; known modes: {sorted(_BY_MODE)}"
        ) from None


def load_training_entry(task: str):
    descriptor = get_task(task)
    module_path, symbol = descriptor.training_entrypoint.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), symbol)


def load_data_module(task: str):
    """按任务 descriptor 加载统一 data module。"""

    descriptor = get_task(task)
    if descriptor.data_module is None:
        raise RuntimeError(f"task {task!r} has no data module registration")
    return importlib.import_module(descriptor.data_module)


def build_inference_adapter(task: str, *args, **kwargs):
    """通过任务 descriptor 构造文件推理 adapter。

    根入口只负责生命周期和输入 inventory；波形、STFT 与任务输入投影由任务
    adapter 自己拥有。没有公共根入口的任务在这里统一给出边界错误。
    """
    descriptor = get_task(task)
    factory_path = descriptor.file_inference_adapter_factory
    if factory_path is None:
        raise ValueError(descriptor.root_file_inference_error)
    module_path, symbol = factory_path.split(":", 1)
    factory = getattr(importlib.import_module(module_path), symbol)
    return factory(*args, **kwargs)


def iter_tasks() -> tuple[TaskDescriptor, ...]:
    return TASK_DESCRIPTORS
