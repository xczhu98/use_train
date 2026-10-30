"""evaluate package — shared metric utilities re-exported for sub-scripts.

evaluate_urgent.py imports metric helpers from this package via:
    from evaluate import compute_pesq, load_dnsmos, ...

All metric implementations live in core.metrics; audio I/O helpers live in
evaluate.eval_pipeline_common. This __init__ re-exports everything so that sibling
scripts can use a simple `from evaluate import ...` import style without
needing to know where each function is defined.

WARNING — HuggingFace evaluate conflict
---------------------------------------
HuggingFace's ``evaluate`` library uses the same top-level package name.
If both are installed, whichever appears first on sys.path wins.

If you see ``ImportError: cannot import name 'compute_pesq' from 'evaluate'``:
  Option A: ``pip uninstall evaluate``  (remove HuggingFace package)
  Option B: Always run scripts from the repository root so the local
            ``evaluate/`` directory precedes installed packages in sys.path.
  Option C: Import directly: ``from core.metrics import compute_pesq``
"""
from __future__ import annotations

import sys as _sys, os as _os
_project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from core.metrics import (
    _resample_np,
    compute_sisdr,
    compute_pesq,
    compute_stoi,
    compute_estoi,
    load_dnsmos,
    compute_dnsmos,
    load_utmos,
    compute_utmos,
    load_nisqa,
    compute_nisqa,
    load_scoreq,
    compute_scoreq,
)

from evaluate.eval_pipeline_common import load_audio

__all__ = [
    "_resample_np",
    "compute_sisdr",
    "compute_pesq",
    "compute_stoi",
    "compute_estoi",
    "load_dnsmos",
    "compute_dnsmos",
    "load_utmos",
    "compute_utmos",
    "load_nisqa",
    "compute_nisqa",
    "load_scoreq",
    "compute_scoreq",
    "load_audio",
]
