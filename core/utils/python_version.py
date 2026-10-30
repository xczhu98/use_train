"""Python version check for use_train.

Code uses Python 3.10+ features (e.g. dict | None, type union syntax).
Call check_python_version() at entry points to fail fast with a clear message.
"""
from __future__ import annotations

import sys


def check_python_version(min_version: tuple[int, int] = (3, 10)) -> None:
    """Require at least the given Python version; exit with message if not met."""
    if sys.version_info < min_version:
        need = ".".join(map(str, min_version))
        cur = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        sys.exit(
            f"SE_Train requires Python {need} or later (current: {cur}).\n"
            f"  Example: conda create -n use_train python=3.10 && conda activate use_train"
        )
