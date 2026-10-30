"""tools/prepare_data_tools/ 内部脚本共享的轻量辅助函数。"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


def _git_commit() -> str:
    """返回当前 git HEAD 的 commit hash；非 git 仓库或执行失败时返回 "unknown"。

    供数据生成脚本写入 manifest 做谱系记录。
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def ensure_project_root() -> None:
    """Insert project root into ``sys.path`` if not already present."""
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def parse_float_pair(s: str) -> tuple[float, float]:
    """解析 ``'lo,hi'`` 形式的浮点区间（CLI 参数专用）。"""
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected 'lo,hi', got {s!r}")
    return (parts[0], parts[1])


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure a logger with a consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    return logging.getLogger(name)


def json_safe(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable native Python types.

    Handles ``numpy.ndarray``, ``numpy.bool_``, ``numpy.integer``,
    ``numpy.floating``, ``torch.Tensor``, and ``pathlib.Path``.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    # torch is an optional dependency here
    if hasattr(obj, "detach"):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)
