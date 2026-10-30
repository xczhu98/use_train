"""任务插件命名空间。"""

from tasks.descriptors import TASK_DESCRIPTORS, TaskDescriptor
from tasks.registry import (
    build_inference_adapter,
    get_task,
    get_task_for_data_mode,
    iter_tasks,
    load_data_module,
    load_training_entry,
)
from tasks.task_api import TaskDataAPI, load_task_data_api

__all__ = [
    "TASK_DESCRIPTORS",
    "TaskDescriptor",
    "build_inference_adapter",
    "get_task",
    "get_task_for_data_mode",
    "iter_tasks",
    "load_data_module",
    "load_training_entry",
    "TaskDataAPI",
    "load_task_data_api",
]
