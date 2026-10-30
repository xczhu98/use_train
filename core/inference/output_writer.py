"""共享推理输出发布接口。"""

from __future__ import annotations

from pathlib import Path

from core.utils.inference_artifacts import (
    atomic_write_audio,
    atomic_write_json,
    atomic_write_text,
    inference_output_lock,
    publish_inference_metadata,
)

__all__ = [
    "atomic_write_audio",
    "atomic_write_json",
    "atomic_write_text",
    "inference_output_lock",
    "publish_inference_metadata",
    "output_path",
]


def output_path(output_dir: str | Path, uid: str, *, suffix: str = ".wav") -> Path:
    """生成安全的平铺输出路径；UID 不得穿越目录。"""
    if not isinstance(uid, str) or not uid or uid in {".", ".."} or Path(uid).name != uid:
        raise ValueError(f"非法 output uid: {uid!r}")
    return Path(output_dir).expanduser() / f"{uid}{suffix}"
