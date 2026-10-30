"""Shared file scanning utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


AUDIO_EXTENSIONS: set[str] = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".sph",
    ".wv",
    ".aif",
    ".aiff",
    ".aifc",
}


def scan_audio_files(
    root: str | Path,
    exts: Iterable[str] | None = None,
    recursive: bool = True,
) -> list[Path]:
    """Recursively (or flat) scan a directory for audio files.

    Args:
        root: Directory to scan.
        exts: Suffixes to include (default: common audio extensions).
            带不带前导点均可（``"wav"`` 与 ``".wav"`` 等价），内部会统一补 ``.``。
        recursive: If False, only scan immediate children.

    Returns:
        Sorted list of absolute Path objects.

    Note:
        递归扫描基于 ``os.walk``（``followlinks=False``），
        symlink 指向的目录会被跳过。
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory or does not exist: {root}")

    # 归一化扩展名：不带前导点（如 "wav"）时自动补 "."，避免静默匹配 0 个文件
    allowed = (
        {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
        if exts
        else AUDIO_EXTENSIONS
    )

    files: list[Path] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if Path(fn).suffix.lower() in allowed:
                    files.append(Path(dirpath) / fn)
    else:
        for fp in root.iterdir():
            if fp.is_file() and fp.suffix.lower() in allowed:
                files.append(fp)

    files.sort(key=lambda p: str(p).lower())
    return [p.resolve() for p in files]
