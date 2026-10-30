"""Task-scoped Trainer class 注册与解析 facade。"""

from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


TrainerFactory = Callable[[], type]


@dataclass
class TaskTrainerResolver:
    """只在当前任务 trainer package 内解析短类名。

    默认 Trainer 由任务 entry 的现有 hook 注册；额外短名通过静态 AST 索引定位
    单个模块后惰性导入，避免 core 维护跨任务候选模块列表。
    """

    package_name: str
    _factories: dict[str, TrainerFactory] = field(default_factory=dict)

    def register(self, name: str, factory: TrainerFactory) -> None:
        """注册 task entry 已知的默认或特殊 Trainer factory。"""
        self._factories[name] = factory

    def resolve(self, class_name: str) -> type:
        """解析 fully-qualified path、已注册名称或 task-local 短名。"""
        factory = self._factories.get(class_name)
        if factory is not None:
            return self._validate_trainer_class(factory(), class_name)
        if "." in class_name:
            module_path, resolved_name = class_name.rsplit(".", 1)
            module = importlib.import_module(module_path)
            trainer_cls = getattr(module, resolved_name, None)
            if trainer_cls is None:
                raise ValueError(
                    f"Trainer class '{resolved_name}' is not available in "
                    f"'{module_path}'."
                )
            return self._validate_trainer_class(trainer_cls, class_name)

        module_names = self._modules_declaring(class_name)
        if not module_names:
            raise ValueError(
                f"Unknown trainer class '{class_name}' for task package "
                f"'{self.package_name}'. Use a fully-qualified 'module.ClassName' "
                "for custom trainers."
            )
        if len(module_names) > 1:
            raise ValueError(
                f"Ambiguous trainer class '{class_name}' in task package "
                f"'{self.package_name}': {module_names}. Use a fully-qualified path."
            )
        module = importlib.import_module(module_names[0])
        trainer_cls = getattr(module, class_name, None)
        if trainer_cls is None:
            raise ValueError(
                f"Trainer class '{class_name}' was declared in '{module_names[0]}' "
                "but is not available after import."
            )
        return self._validate_trainer_class(trainer_cls, class_name)

    @staticmethod
    def _validate_trainer_class(trainer_cls: object, requested_name: str) -> type:
        """拒绝函数、普通对象及未继承共享 Trainer 基类的 class。"""
        from core.trainer.base_trainer import BaseTrainer

        if not inspect.isclass(trainer_cls) or not issubclass(trainer_cls, BaseTrainer):
            raise ValueError(
                f"Trainer '{requested_name}' must resolve to a BaseTrainer subclass, "
                f"got {trainer_cls!r}."
            )
        return trainer_cls

    def _modules_declaring(self, class_name: str) -> list[str]:
        """静态索引当前 task package，避免为查找短名 import 所有 trainer。"""
        package = importlib.import_module(self.package_name)
        roots = [Path(path) for path in getattr(package, "__path__", ())]
        matches: list[str] = []
        for root in roots:
            for path in sorted(root.rglob("*.py")):
                if path.name == "__init__.py":
                    continue
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                except (OSError, SyntaxError, UnicodeError):
                    continue
                if not any(
                    isinstance(node, ast.ClassDef) and node.name == class_name
                    for node in tree.body
                ):
                    continue
                relative = path.relative_to(root).with_suffix("")
                matches.append(
                    ".".join((self.package_name, *relative.parts))
                )
        return matches
