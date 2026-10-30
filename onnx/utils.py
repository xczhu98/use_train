"""Common utilities for ONNX export, inference, and verification.

Handles PyPI ``onnx`` package isolation (project root has an ``onnx/`` sub-package
that shadows the PyPI package) and shared helpers for precision checks.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# ``sys.path`` / ``sys.modules`` 是进程全局状态，exporter 边界须用可重入锁串行化；
# 纯文件 metadata 修改在隔离子进程中执行。
_PYPI_ONNX_SCOPE_LOCK = threading.RLock()
_PYPI_ONNX_SCOPE_STATE = threading.local()
_PYPI_ONNX_MODULE_CACHE: dict[str, object] = {}


def _is_local_onnx_module(module: object, root: Path = _PROJECT_ROOT) -> bool:
    """Return whether a loaded module belongs to this repository's ``onnx/`` package."""
    locations = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        locations.append(module_file)
    locations.extend(getattr(module, "__path__", ()) or ())
    for location in locations:
        try:
            Path(location).resolve().relative_to(root / "onnx")
        except (OSError, ValueError, TypeError):
            continue
        return True
    return False


def _is_project_path(entry: str, root: Path = _PROJECT_ROOT) -> bool:
    """Match path entries that could resolve this repository's local package."""
    try:
        if not entry:
            return Path.cwd().resolve() == root.resolve()
        return Path(entry).resolve() == root.resolve()
    except (OSError, RuntimeError):
        return False


@contextmanager
def pypi_onnx_scope() -> Iterator[object]:
    """Temporarily import PyPI ``onnx`` and fully restore local import state.

    This in-process fallback is required only for operations such as
    ``torch.onnx.export`` that hold a live PyTorch model.  A process-wide ``RLock``
    prevents concurrent mutation of ``sys.path``/``sys.modules``; same-thread nested
    scopes reuse the already-imported external package.  File-only ONNX operations
    should prefer an isolated subprocess instead.

    约束（进程级副作用，无法靠锁消除）：scope 生效期间 ``sys.path``/``sys.modules``
    被整体改写，任何**其它线程**在同一窗口内 ``import onnx`` 都会绑定到错误的实现。
    ``RLock`` 只能串行化本 helper 的协作调用者，因此并发导出必须在调用方侧串行化，
    或改用独立子进程；scope 退出后会复核原 import 状态是否被并发修改，漂移即 fail-fast。
    """
    with _PYPI_ONNX_SCOPE_LOCK:
        depth = getattr(_PYPI_ONNX_SCOPE_STATE, "depth", 0)
        if depth:
            external = sys.modules.get("onnx")
            if external is None or _is_local_onnx_module(external):
                raise RuntimeError("nested PyPI onnx scope lost its external module")
            _PYPI_ONNX_SCOPE_STATE.depth = depth + 1
            try:
                yield external
            finally:
                _PYPI_ONNX_SCOPE_STATE.depth = depth
            return

        original_path = list(sys.path)
        original_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "onnx" or name.startswith("onnx.")
        }
        external = None
        _PYPI_ONNX_SCOPE_STATE.depth = 1
        try:
            for name in list(original_modules):
                sys.modules.pop(name, None)
            sys.path[:] = [entry for entry in sys.path if not _is_project_path(entry)]
            if _PYPI_ONNX_MODULE_CACHE:
                sys.modules.update(_PYPI_ONNX_MODULE_CACHE)
            external = importlib.import_module("onnx")
            if _is_local_onnx_module(external):
                raise ImportError(
                    "PyPI onnx could not be imported because the repository package still shadows it"
                )
            yield external
        finally:
            # 恢复期间若仍有异常在传播，必须保留原始异常，不做额外断言。
            pending_exception = sys.exc_info()[0] is not None
            # scope 结束时 onnx 绑定若已不是本次注入的实现，说明窗口内有并发或
            # 代码改写了进程级 import 状态，隔离本身已不可信。
            isolation_broken = sys.modules.get("onnx") is not external
            if external is not None and not _is_local_onnx_module(external):
                _PYPI_ONNX_MODULE_CACHE.clear()
                _PYPI_ONNX_MODULE_CACHE.update(
                    {
                        name: module
                        for name, module in sys.modules.items()
                        if name == "onnx" or name.startswith("onnx.")
                    }
                )
            for name in list(sys.modules):
                if name == "onnx" or name.startswith("onnx."):
                    sys.modules.pop(name, None)
            sys.modules.update(original_modules)
            sys.path[:] = original_path
            _PYPI_ONNX_SCOPE_STATE.depth = 0
            if not pending_exception and isolation_broken:
                raise RuntimeError(
                    "PyPI onnx scope isolation was broken: sys.modules['onnx'] was "
                    "replaced during the scope; do not import onnx from other threads "
                    "while an export is running"
                )


def export_onnx(*args, **kwargs):
    """Run ``torch.onnx.export`` under the serialized in-process fallback."""
    with pypi_onnx_scope():
        return torch.onnx.export(*args, **kwargs)


def write_onnx_metadata(onnx_path, metadata: dict[str, object]) -> None:
    """稳定地替换多个 ONNX metadata entry，操作在隔离子进程完成。"""
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("metadata must be a non-empty dict")
    if any(not isinstance(key, str) or not key for key in metadata):
        raise ValueError("metadata keys must be non-empty strings")
    # 禁止 sort_keys：cache_cfg 等 value 的 JSON 对象键顺序即 ABI 顺序，
    # 排序会在写入前就被破坏；entry 写入顺序由子进程内 sorted(updates) 保证。
    payload = json.dumps(metadata, separators=(",", ":"))
    injector = (
        "import json, sys, onnx\n"
        "path = sys.argv[1]\n"
        "updates = json.load(sys.stdin)\n"
        "model = onnx.load(path)\n"
        "kept = [(p.key, p.value) for p in model.metadata_props if p.key not in updates]\n"
        "del model.metadata_props[:]\n"
        "for old_key, old_value in kept:\n"
        "    prop = model.metadata_props.add(); prop.key = old_key; prop.value = old_value\n"
        "for key in sorted(updates):\n"
        "    prop = model.metadata_props.add(); prop.key = key\n"
        # value 内部禁止 sort_keys：cache_cfg 等 metadata 的 JSON 对象键顺序即
        # ABI 顺序（如 GTCRN conv/tra/inter），排序会破坏 runtime spec 校验。
        "    prop.value = json.dumps(updates[key], separators=(',', ':'))\n"
        "onnx.save(model, path)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", injector, str(onnx_path)],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"Failed to write ONNX metadata {sorted(metadata)!r} into {onnx_path}: {detail}"
        )


def require_onnxruntime():
    """Import ONNX Runtime with an actionable missing-dependency error."""
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is required for ONNX verification/inference; "
            "install the optional 'onnxruntime' dependency in the active environment."
        ) from exc


def create_ort_session(onnx_path, *, providers=None):
    """Create an ORT session through the shared optional-dependency boundary."""
    ort = require_onnxruntime()
    selected = providers or ["CPUExecutionProvider"]
    try:
        return ort.InferenceSession(str(onnx_path), providers=selected)
    except Exception as exc:
        raise RuntimeError(f"Failed to load ONNX Runtime model {onnx_path}: {exc}") from exc


def read_onnx_metadata(onnx_path, *, providers=None) -> dict[str, str]:
    """Read custom metadata through ORT, avoiding PyPI/local ``onnx`` shadowing."""
    session = create_ort_session(onnx_path, providers=providers)
    metadata = session.get_modelmeta().custom_metadata_map
    return dict(metadata or {})


def parse_onnx_cache_config(metadata: dict[str, str], metadata_key: str) -> dict[str, list[int]]:
    """Decode a non-empty batch-one cache template from ONNX metadata."""
    raw = metadata.get(metadata_key)
    if raw is None:
        raise RuntimeError(f"ONNX metadata is missing required {metadata_key!r}")
    try:
        config = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"ONNX metadata {metadata_key!r} is not valid JSON") from exc
    if not isinstance(config, dict) or not config:
        raise RuntimeError(f"ONNX metadata {metadata_key!r} must be a non-empty object")

    normalized: dict[str, list[int]] = {}
    for name, shape in config.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"ONNX metadata {metadata_key!r} contains an invalid cache name")
        if not isinstance(shape, list) or not shape or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
        ):
            raise RuntimeError(
                f"ONNX metadata {metadata_key!r} contains invalid shape for {name!r}: {shape!r}"
            )
        normalized[name] = list(shape)
    return normalized


def materialize_cache_shapes(
    cache_config: dict[str, list[int]],
    input_infos,
    batch_size: int,
    *,
    allowed_dynamic_axes: frozenset[str] = frozenset(),
) -> dict[str, list[int]]:
    """校验 graph/cache ABI 并实例化 batch-one metadata template。

    非 batch 动态轴默认拒绝；仅模型 spec 明确声明的轴名可使用 metadata 中的
    冷启动长度（例如 CAGCRN ``time`` cache 从 0 增长到上限）。
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    infos = {info.name: info for info in input_infos}
    materialized: dict[str, list[int]] = {}
    for name, template in cache_config.items():
        info = infos.get(name)
        if info is None:
            raise RuntimeError(f"ONNX cache metadata names unknown input {name!r}")
        graph_shape = getattr(info, "shape", None)
        if graph_shape is None or len(graph_shape) != len(template):
            raise RuntimeError(
                f"ONNX input {name!r} shape {graph_shape!r} does not match metadata {template!r}"
            )

        batch_axes = []
        permitted_dynamic_axes = []
        unknown_axes = []
        for axis, dimension in enumerate(graph_shape):
            if isinstance(dimension, str):
                normalized = dimension.lower()
                if normalized in {"batch", "batch_size", "batch_flattened"}:
                    batch_axes.append(axis)
                elif normalized in allowed_dynamic_axes:
                    permitted_dynamic_axes.append(axis)
                else:
                    unknown_axes.append(axis)
            elif dimension is None:
                batch_axes.append(axis)
            elif isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
                unknown_axes.append(axis)
        if unknown_axes:
            raise RuntimeError(
                f"ONNX input {name!r} has unknown dynamic axes {unknown_axes}; refusing to guess cache shape"
            )
        if len(batch_axes) > 1:
            raise RuntimeError(f"ONNX input {name!r} exposes multiple dynamic batch axes")

        shape = list(template)
        if batch_axes:
            batch_axis = batch_axes[0]
            graph_dimension = graph_shape[batch_axis]
            is_flattened = (
                isinstance(graph_dimension, str)
                and graph_dimension.lower() == "batch_flattened"
            )
            batch_multiplier = template[batch_axis] if is_flattened else 1
            if is_flattened and batch_multiplier < 1:
                raise RuntimeError(
                    f"ONNX cache metadata {name!r} batch multiplier must be positive, "
                    f"got {batch_multiplier}"
                )
            if not is_flattened and template[batch_axis] != 1:
                raise RuntimeError(
                    f"ONNX cache metadata {name!r} batch template must be 1, "
                    f"got {template[batch_axis]}"
                )
            shape[batch_axis] = int(batch_size) * batch_multiplier
        elif int(batch_size) != 1:
            raise RuntimeError(
                f"Static ONNX cache input {name!r} only supports batch_size=1"
            )
        for axis, dimension in enumerate(graph_shape):
            if axis in batch_axes or axis in permitted_dynamic_axes:
                continue
            if int(dimension) != template[axis]:
                raise RuntimeError(
                    f"ONNX input {name!r} static axis {axis}={dimension!r} "
                    f"does not match metadata {template[axis]!r}"
                )
        materialized[name] = shape
    return materialized
