"""Local path checks for AEC manifests (JSON ref/echo pair lists, etc.)."""

from __future__ import annotations

import os
from pathlib import Path


def usable_audio_path(p: str | Path) -> bool:
    """True if *p* expands to an existing regular file."""
    if p is None:
        return False
    s = str(p).strip()
    if not s:
        return False
    try:
        # os.path.isfile is measurably faster than Path.is_file() for large
        # manifests because it avoids parsing the path into Path segments.
        return os.path.isfile(os.path.expanduser(s))
    except (TypeError, ValueError, OSError):
        return False
