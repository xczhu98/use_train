"""Universal Registry — name->class mapping for models, losses, schedulers.

Usage:
    @MODEL_REGISTRY.register("GTCRN")     # with name
    class GTCRN: ...

    @MODEL_REGISTRY.register()            # auto-use class name
    class BSRNN: ...

    model = MODEL_REGISTRY.build("GTCRN", n_fft=512)   # instantiate
    cls   = MODEL_REGISTRY.get("GTCRN")                 # get class
"""

from typing import Dict, Optional, Type, Any


class Registry:
    """Name -> class mapping registry, supports build (instantiate) and get (retrieve class)."""

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type] = {}
        self._lazy_map: dict[str, str] = {}

    def register(self, name_or_cls=None):
        """Decorator supporting @register("name") and @register().

        Duplicate registration of same class (e.g. second import from running model file)
        is silently ignored. Different class with same key raises ValueError.
        """
        if isinstance(name_or_cls, type):
            key = name_or_cls.__name__
            if key in self._registry:
                if self._registry[key] is not name_or_cls:
                    raise ValueError(f"[{self._name}] '{key}' already registered by a different class")
                return name_or_cls  # same class duplicate, idempotent ignore
            self._registry[key] = name_or_cls
            return name_or_cls

        name: Optional[str] = name_or_cls

        def decorator(cls):
            key = name or cls.__name__
            if key in self._registry:
                if self._registry[key] is not cls:
                    raise ValueError(f"[{self._name}] '{key}' already registered by a different class")
                return cls  # same class duplicate, idempotent ignore
            self._registry[key] = cls
            return cls
        return decorator

    def build(self, name: str, **kwargs) -> Any:
        if name not in self._registry:
            raise KeyError(
                f"[{self._name}] '{name}' not found. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name](**kwargs)

    def get(self, name: str) -> Type:
        if name not in self._registry:
            raise KeyError(
                f"[{self._name}] '{name}' not found. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def keys(self):
        return self._registry.keys()

    def __contains__(self, name: str):
        return name in self._registry

    def set_lazy_map(self, mapping: "dict[str, str]") -> None:
        """Register a lazy-load mapping: name -> module path.

        同名同路径重复注册幂等；同名但指向不同模块路径视为冲突，直接报错。
        """
        for name, path in mapping.items():
            existing = self._lazy_map.get(name)
            if existing is not None and existing != path:
                raise ValueError(
                    f"[{self._name}] lazy name '{name}' already mapped to "
                    f"'{existing}', conflicting re-registration to '{path}'"
                )
        self._lazy_map.update(mapping)

    def ensure_registered(self, name: str) -> None:
        """If *name* is not yet registered, import its module from the lazy map."""
        if name in self._registry:
            return
        module_path = self._lazy_map.get(name)
        if module_path is None:
            return
        import importlib
        importlib.import_module(module_path)

    def __repr__(self):
        return f"Registry(name={self._name}, items={list(self._registry.keys())})"


MODEL_REGISTRY = Registry("model")
LOSS_REGISTRY = Registry("loss")
SCHEDULER_REGISTRY = Registry("scheduler")
