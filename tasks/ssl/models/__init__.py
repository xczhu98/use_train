"""FN-SSL 模型的正式共享 registry 入口。"""

from __future__ import annotations

import importlib
import logging

from core.models import register_lazy_models
from core.models.build import build_model as _build_model
from core.utils.registry import MODEL_REGISTRY
from tasks.ssl.model_io import validate_ssl_model_io

logger = logging.getLogger(__name__)
_MODEL_REGISTRY_MAP = {
    "FixedArrayIPDNet": "tasks.ssl.models.ipdnet",
    "FNSSL": "tasks.ssl.models.ipdnet",
    "IPDNetFixed": "tasks.ssl.models.ipdnet_official",
    "IPDNetVariable": "tasks.ssl.models.ipdnet_official",
    "IPDNet2": "tasks.ssl.models.ipdnet2",
}
SSL_MODEL_TYPES = frozenset(_MODEL_REGISTRY_MAP)
MODEL_REGISTRY.set_lazy_map(_MODEL_REGISTRY_MAP)

# 具体模型只在属性访问时导入，保证包导入不触发 @MODEL_REGISTRY.register 副作用。
_EXPORT_MODULES = {
    "DPIPD": "tasks.ssl.models.ipdnet",
    "FixedArrayIPDNet": "tasks.ssl.models.ipdnet",
    "IPDFeatureContract": "tasks.ssl.models.ipdnet",
    "FNSSL": "tasks.ssl.models.ipdnet",
    "IPDNetFixed": "tasks.ssl.models.ipdnet_official",
    "IPDNetVariable": "tasks.ssl.models.ipdnet_official",
    "IPDNet2": "tasks.ssl.models.ipdnet2",
}


def __getattr__(name: str):
    module_path = _EXPORT_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def register_ssl_models(only: str | None = None) -> None:
    register_lazy_models(_MODEL_REGISTRY_MAP, task="ssl", only=only, logger=logger)


def build_model(cfg: dict, stft_cfg: dict | None = None):
    if not isinstance(cfg, dict) or "type" not in cfg:
        raise KeyError("SSL model config must contain 'type'")
    model = _build_model(cfg, stft_cfg=stft_cfg)
    validate_ssl_model_io(model)
    return model


__all__ = [
    "DPIPD",
    "FixedArrayIPDNet",
    "FNSSL",
    "IPDNetFixed",
    "IPDNetVariable",
    "IPDNet2",
    "IPDFeatureContract",
    "MODEL_REGISTRY",
    "SSL_MODEL_TYPES",
    "build_model",
    "register_ssl_models",
]
