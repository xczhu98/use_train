"""Safe torch.load wrapper with explicit opt-in for pickle."""
from contextlib import contextmanager
from pathlib import Path
import importlib
import logging
import os
import re
import threading

import torch
import torch.serialization as _ts

logger = logging.getLogger(__name__)

# Blocked-global name as reported by weights_only errors, in either the
# inspection form ("GLOBAL numpy.dtype ...") or the runtime form
# ("... but got <class 'numpy.dtypes.Float32DType'>").
_BLOCKED_GLOBAL_RE = re.compile(r"GLOBAL\s+([\w.]+)|got <class '([\w.]+)'>")
_SAFE_GLOBALS_LOCK = threading.RLock()


def safe_torch_load(
    path,
    *,
    map_location: str = "cpu",
    allow_unsafe: bool = False,
    purpose: str = "",
) -> dict:
    """Load a torch checkpoint path or seekable binary stream with safety defaults.

    Args:
        path: checkpoint path or seekable binary file-like object
        map_location: torch map_location
        allow_unsafe: if True, use weights_only=False (allows pickle RCE)
        purpose: human-readable reason for opening unsafe mode (logged)

    Safety contract:
        With ``allow_unsafe=False`` (default) the checkpoint is ALWAYS loaded in
        ``weights_only=True`` mode.  Training checkpoints commonly pickle numpy
        scalars/arrays, which ``weights_only`` rejects by default; those pure-data
        numpy globals are enumerated (without executing them) and allowlisted, then
        loading is retried IN SAFE MODE.  We never silently fall back to unsafe
        pickle loading: callers must set ``allow_unsafe=True`` and provide a
        non-empty ``purpose``.
    """
    is_path = isinstance(path, (str, os.PathLike))
    source = Path(path) if is_path else path
    label = str(source) if is_path else getattr(source, "name", "<checkpoint stream>")
    if is_path:
        if not source.exists():
            raise FileNotFoundError(f"checkpoint not found: {source}")
    elif not callable(getattr(source, "read", None)) or not callable(
        getattr(source, "seek", None)
    ):
        raise TypeError("checkpoint source must be a path or seekable binary stream")

    if not allow_unsafe:
        try:
            return _torch_load_rewound(
                source, map_location=map_location, weights_only=True
            )
        except TypeError as e:
            if _weights_only_kw_unsupported(e):
                raise RuntimeError(
                    f"Cannot safely load {label}: this PyTorch version does not support "
                    "weights_only. Upgrade PyTorch, or for a TRUSTED source pass "
                    "allow_unsafe=True with a non-empty purpose."
                ) from e
            raise RuntimeError(
                f"Cannot safely load {label} with weights_only=True ({e})."
            ) from e
        except Exception as e:
            try:
                return _load_safe_allowlisting_static_globals(
                    source, label, map_location
                )
            except _UnsafeGlobalRequired as ue:
                raise RuntimeError(
                    f"Cannot safely load {label} with weights_only=True: it requires "
                    f"non-allowlisted pickled global(s) {ue.names}. If the file is from a "
                    f"TRUSTED source (e.g. resuming optimizer/scheduler state), pass "
                    f"allow_unsafe=True explicitly with a non-empty purpose."
                ) from e
            except Exception:
                raise RuntimeError(
                    f"Cannot safely load {label} with weights_only=True ({e}). "
                    f"If it is from a TRUSTED source, pass allow_unsafe=True explicitly "
                    f"with a non-empty purpose."
                ) from e

    normalized_purpose = purpose.strip()
    if not normalized_purpose:
        raise ValueError("allow_unsafe=True requires a non-empty purpose")
    logger.warning(
        "[SECURITY] Loading %s with weights_only=False (pickle allowed). "
        "Only use for checkpoints from trusted sources. Purpose: %s",
        label, normalized_purpose,
    )
    return _torch_load_rewound(
        source, map_location=map_location, weights_only=False
    )


def safe_model_weights_load(
    path,
    *,
    map_location: str = "cpu",
    purpose: str = "load model-only weights",
) -> dict:
    """安全加载 model-only 权重，并按文件格式显式分派。

    ``safetensors`` 不使用 pickle；其余格式继续通过 ``safe_torch_load`` 的
    ``weights_only=True`` 路径。调用方仍需对返回的 state dict 执行完整结构、
    key、dtype 与 shape 校验。
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("model weights source must be a filesystem path")
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    if source.suffix.lower() != ".safetensors":
        return safe_torch_load(
            source,
            map_location=map_location,
            purpose=purpose,
        )
    if map_location != "cpu":
        raise ValueError("safetensors model weights must be loaded on CPU first")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "loading a .safetensors checkpoint requires the optional safetensors package"
        ) from exc
    state = load_file(str(source), device="cpu")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"safetensors checkpoint contains no tensors: {source}")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise TypeError("safetensors checkpoint must contain a string-to-tensor state dict")
    return state


def _torch_load_rewound(source, **kwargs):
    """每次加载前回卷 stream，保证安全 allowlist 重试读取相同内容。"""
    if not isinstance(source, Path):
        source.seek(0)
    return torch.load(source, **kwargs)


def _weights_only_kw_unsupported(error: TypeError) -> bool:
    """返回错误是否仅表示当前 PyTorch 不支持 weights_only 参数。"""
    message = str(error).lower()
    return "weights_only" in message and "unexpected keyword" in message


class _UnsafeGlobalRequired(Exception):
    """Raised internally when a checkpoint needs non-numpy pickled global(s)."""

    def __init__(self, names):
        super().__init__(", ".join(names))
        self.names = list(names)


# 仅允许 numpy 序列化 ndarray/scalar 所需的纯数据重建对象，以及 PyTorch
# weights_only 反序列化标准 bytes 所需的 ``_codecs.encode``。这里刻意不接受任意
# ``numpy.*``：numpy.ctypeslib.load_library 等可调用对象不能进入 allowlist。NumPy
# 1.x/2.x 同时保留 core 与 _core 路径以兼容已发布 checkpoint。
_CODEC_SAFE_GLOBAL_NAMES = frozenset({
    "_codecs.encode",
    # 官方模型权重常把 PyTorch 版本信息作为纯元数据序列化；该类型不执行模型代码。
    "torch.torch_version.TorchVersion",
})
_NUMPY_SAFE_GLOBAL_NAMES = frozenset(
    {
        "numpy._core.multiarray._reconstruct",
        "numpy.core.multiarray._reconstruct",
        "numpy._core.multiarray.scalar",
        "numpy.core.multiarray.scalar",
        "numpy.ndarray",
        "numpy.dtype",
        "numpy.dtypes.BoolDType",
        "numpy.dtypes.Int8DType",
        "numpy.dtypes.Int16DType",
        "numpy.dtypes.Int32DType",
        "numpy.dtypes.Int64DType",
        "numpy.dtypes.UInt8DType",
        "numpy.dtypes.UInt16DType",
        "numpy.dtypes.UInt32DType",
        "numpy.dtypes.UInt64DType",
        "numpy.dtypes.Float16DType",
        "numpy.dtypes.Float32DType",
        "numpy.dtypes.Float64DType",
        "numpy.dtypes.Complex64DType",
        "numpy.dtypes.Complex128DType",
    }
)
_NUMPY_SAFE_DTYPE_NAMES = (
    "bool_",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)
_BLOCKED_NUMPY_NAMESPACES = (
    "numpy.ctypeslib.",
    "numpy.f2py.",
    "numpy.lib.npyio.",
)


def _is_safe_global(name: str) -> bool:
    """返回名称是否属于经审查的静态安全 allowlist。"""
    if name.startswith(_BLOCKED_NUMPY_NAMESPACES):
        return False
    return name in _CODEC_SAFE_GLOBAL_NAMES or name in _NUMPY_SAFE_GLOBAL_NAMES


def _load_safe_allowlisting_static_globals(source, label, map_location):
    """在局部 ``safe_globals`` 上下文中用固定的纯数据 allowlist 单次加载。

    所有来自 checkpoint inspection 或运行时错误的名称必须属于静态审查名单；
    未列名的 numpy 子模块与其他全局均 fail-closed，且不会污染进程级
    ``torch.serialization`` allowlist。allowlist 在加载前一次算定且加载期间
    不变，失败直接抛原异常。
    """
    # inspection API 仅用于提早拒绝非白名单全局；实际对象始终来自静态列表。
    getter = getattr(_ts, "get_unsafe_globals_in_checkpoint", None)
    if getter is not None:
        try:
            if not isinstance(source, Path):
                source.seek(0)
            needed = list(getter(source))
        except Exception:
            needed = []
        disallowed = [name for name in needed if not _is_safe_global(name)]
        if disallowed:
            raise _UnsafeGlobalRequired(disallowed)

    seed = _static_safe_globals()
    with _scoped_safe_globals(seed):
        try:
            result = _torch_load_rewound(
                source, map_location=map_location, weights_only=True
            )
        except Exception as e:
            name = _blocked_global_name(str(e))
            if name is None:
                raise
            if not _is_safe_global(name):
                raise _UnsafeGlobalRequired([name])
            raise
        if seed:
            logger.warning(
                "[safe_load] %s: used scoped static globals; loaded in SAFE mode.",
                label,
            )
        return result


@contextmanager
def _scoped_safe_globals(globals_to_add):
    """在局部上下文中注册 safe globals，并严格恢复调用方状态。"""
    safe_globals = getattr(_ts, "safe_globals", None)
    if safe_globals is not None:
        with safe_globals(globals_to_add):
            yield
        return

    get_safe_globals = getattr(_ts, "get_safe_globals", None)
    add_safe_globals = getattr(_ts, "add_safe_globals", None)
    clear_safe_globals = getattr(_ts, "clear_safe_globals", None)
    if None in (get_safe_globals, add_safe_globals, clear_safe_globals):
        raise RuntimeError("PyTorch does not provide reversible safe_globals support")
    with _SAFE_GLOBALS_LOCK:
        previous = list(get_safe_globals())
        add_safe_globals(globals_to_add)
        body_raised = False
        try:
            yield
        except BaseException:
            body_raised = True
            raise
        finally:
            try:
                clear_safe_globals()
                add_safe_globals(previous)
            except BaseException:
                if body_raised:
                    logger.exception(
                        "[safe_load] 恢复 safe_globals 失败，保留原加载异常"
                    )
                else:
                    raise


def _blocked_global_name(msg: str) -> str | None:
    """Extract a blocked global's dotted name from a weights_only error message."""
    m = _BLOCKED_GLOBAL_RE.search(msg)
    if not m:
        return None
    return m.group(1) or m.group(2)


def _resolve_dotted(dotted: str):
    """Resolve a dotted name like ``numpy._core.multiarray._reconstruct`` to the
    object, importing submodules as needed. Returns None on failure."""
    parts = dotted.split(".")
    module = None
    idx = len(parts)
    while idx > 0:
        try:
            module = importlib.import_module(".".join(parts[:idx]))
            break
        except Exception:
            idx -= 1
    if module is None:
        return None
    obj = module
    for part in parts[idx:]:
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
    return obj


def _static_safe_globals() -> list:
    """返回固定审查名单对应的 bytes 与 numpy 纯数据重建对象。"""
    candidates = [
        obj
        for name in _CODEC_SAFE_GLOBAL_NAMES
        if (obj := _resolve_dotted(name)) is not None
    ]
    try:
        import numpy as np
    except ImportError:
        return candidates
    candidates.extend(
        obj
        for name in _NUMPY_SAFE_GLOBAL_NAMES
        if (obj := _resolve_dotted(name)) is not None
    )
    candidates.extend(
        obj for name in _NUMPY_SAFE_DTYPE_NAMES if (obj := getattr(np, name, None)) is not None
    )
    return candidates
