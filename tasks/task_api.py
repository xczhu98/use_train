"""跨任务公共 data API。

任务包继续拥有各自的数据实现，但通过 descriptor 暴露统一的动态/离线构造与
collate 查找入口。该层只负责路由和合同检查，不复制任何任务数据逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class TaskDataAPI:
    """一个任务的数据构造和批处理公共入口。"""

    task: str
    module: ModuleType
    dataset_builder: str | None
    collate_function: str | None
    dataset_validator: str | None

    def __reduce__(self):
        # train.py 强制 spawn start method 后，DataLoader worker 需要 pickle
        # entry 传入的 ``partial(api.collate, ...)``；module 对象不可 pickle，
        # 因此按任务名重建（load_task_data_api(task) 是纯函数，语义不变）。
        return (load_task_data_api, (self.task,))

    def resolve(self, reference: str):
        """解析任务 data module 内或其子模块中的显式 callable 路径。"""

        import importlib

        if ":" in reference:
            module_path, symbol = reference.split(":", 1)
            return getattr(importlib.import_module(module_path), symbol)
        return getattr(self.module, reference)

    def build_dataset(self, *args: Any, **kwargs: Any):
        if self.dataset_builder is None:
            raise NotImplementedError(
                f"task {self.task!r} does not expose a generic dataset builder"
            )
        builder = self.resolve(self.dataset_builder) if self.dataset_builder else None
        if not callable(builder):
            raise RuntimeError(
                f"task {self.task!r} descriptor points to missing dataset builder "
                f"{self.dataset_builder!r} in {self.module.__name__}"
            )
        dataset = builder(*args, **kwargs)
        if self.dataset_validator is not None:
            import importlib

            module_path, symbol = self.dataset_validator.split(":", 1)
            validator = getattr(importlib.import_module(module_path), symbol)
            return validator(dataset)
        return dataset

    def collate(self, *args: Any, **kwargs: Any):
        if self.collate_function is None:
            raise NotImplementedError(
                f"task {self.task!r} does not expose a generic collate function"
            )
        if ":" in self.collate_function:
            import importlib

            module_path, symbol = self.collate_function.split(":", 1)
            collate = getattr(importlib.import_module(module_path), symbol, None)
        else:
            collate = getattr(self.module, self.collate_function, None)
        if not callable(collate):
            raise RuntimeError(
                f"task {self.task!r} descriptor points to missing collate function "
                f"{self.collate_function!r}"
            )
        return collate(*args, **kwargs)


def load_task_data_api(task: str) -> TaskDataAPI:
    """按任务 descriptor 惰性加载 data module。"""

    from tasks.registry import get_task

    descriptor = get_task(task)
    if descriptor.data_module is None:
        raise RuntimeError(f"task {task!r} has no data module registration")
    import importlib

    module = importlib.import_module(descriptor.data_module)
    return TaskDataAPI(
        task=descriptor.name,
        module=module,
        dataset_builder=descriptor.data_builder,
        collate_function=descriptor.collate_function,
        dataset_validator=descriptor.dataset_validator,
    )


__all__ = ["TaskDataAPI", "load_task_data_api"]
