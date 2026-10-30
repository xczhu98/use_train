"""受限 pickle 反序列化（RIR 等 .pkl 数据文件的统一安全入口）与统一保存。

默认仅允许 numpy 与基础 builtins/collections 类型，禁止任意代码对象；
仅可信数据可在环境中设置 ``PREPARE_SCP_ALLOW_UNSAFE_PICKLE=1`` 并由调用方
提供非空 ``purpose``，才能回退完整 ``pickle.load``（环境变量名保持与
``prepare_scp.py`` 既有约定一致）。保存侧统一走 :func:`save_pkl_file`，
默认协议与裸 ``pickle.dump`` 一致，便于未来统一升级协议。
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 环境变量名沿用 prepare_scp.py 的既有约定，勿改名（文档已引用）
ALLOW_UNSAFE_PICKLE_ENV = "PREPARE_SCP_ALLOW_UNSAFE_PICKLE"

_SAFE_BUILTINS = frozenset({
    "dict", "list", "tuple", "set", "frozenset", "bytes", "str", "int", "float",
    "bool", "complex", "bytearray", "slice", "range", "type", "NoneType", "object",
    "enumerate", "map", "zip", "property",
})

_SAFE_COLLECTIONS = frozenset({"OrderedDict", "deque", "defaultdict"})

# 仅允许 ndarray、numpy scalar 与 dtype 的 pickle 重建入口；同时兼容
# NumPy 1.x 的 numpy.core.* 和 NumPy 2.x 的 numpy._core.* 模块路径。
_SAFE_NUMPY_GLOBALS = frozenset({
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"),
})


def _pickle_global_allowed(module: str, name: str) -> bool:
    if module == "builtins" and name in _SAFE_BUILTINS:
        return True
    if module == "collections" and name in _SAFE_COLLECTIONS:
        return True
    if module in ("copyreg", "_compat_pickle"):
        return name in ("_reconstructor",)
    if module == "_codecs" and name == "encode":
        return True
    return (module, name) in _SAFE_NUMPY_GLOBALS


class RestrictedUnpickler(pickle.Unpickler):
    """Only numpy + basic builtin/collections types (no arbitrary code objects)."""

    def find_class(self, module: str, name: str):
        if not _pickle_global_allowed(module, name):
            raise pickle.UnpicklingError(f"forbidden global {module}.{name}")
        return super().find_class(module, name)


def load_pkl_file(path: str, *, purpose: str = "") -> object:
    """加载 .pkl 文件；unsafe 环境开关启用时必须提供审计用途。"""
    with open(path, "rb") as f:
        if os.environ.get(ALLOW_UNSAFE_PICKLE_ENV, "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            normalized_purpose = purpose.strip()
            if not normalized_purpose:
                raise ValueError(
                    f"{ALLOW_UNSAFE_PICKLE_ENV} requires a non-empty purpose"
                )
            logger.warning(
                "[SECURITY] %s is set: using full pickle.load() on %s. Purpose: %s",
                ALLOW_UNSAFE_PICKLE_ENV,
                path,
                normalized_purpose,
            )
            return pickle.load(f)
        return RestrictedUnpickler(f).load()


def save_pkl_file(path: str | Path, obj: Any) -> None:
    """统一写入 .pkl 文件；协议保持与裸 ``pickle.dump`` 默认一致。"""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)
