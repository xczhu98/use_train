"""Core objective metrics computation module.

Intrusive metrics (require clean reference):
    SI-SDR, PESQ, STOI, ESTOI

AEC-specific intrusive metric:
    ERLE

Non-intrusive MOS metrics (enhanced audio only):
    DNSMOS, NISQA, UTMOS, SCOREQ

Canonical MOS model loaders live in ``core.metrics.mos_models``. These APIs
serve offline evaluation and data filtering, not Trainer validation.
"""

from __future__ import annotations

from core.metrics.aec import compute_erle
from core.metrics.batch import compute_metrics_batch
from core.metrics.intrusive import (
    compute_estoi,
    compute_pesq,
    compute_sisdr,
    compute_sisdr_np,
    compute_stoi,
)
from core.metrics.mos import (
    compute_dnsmos,
    compute_mos_batch,
    compute_nisqa,
    compute_scoreq,
    compute_utmos,
)
from core.metrics.utils import _resample_np

__all__ = [
    "_resample_np",
    "compute_sisdr",
    "compute_sisdr_np",
    "compute_pesq",
    "compute_stoi",
    "compute_estoi",
    "compute_erle",
    "compute_dnsmos",
    "compute_nisqa",
    "compute_utmos",
    "compute_scoreq",
    "compute_mos_batch",
    "compute_metrics_batch",
    "load_dnsmos",
    "load_nisqa",
    "load_scoreq",
    "load_utmos",
    "load_aecmos",
    "compute_aecmos",
]

# MOS model loaders
from core.metrics.mos_models import (  # noqa: F401
    load_dnsmos,
    load_nisqa,
    load_scoreq,
    load_utmos,
)

# AECMOS
from core.metrics.aecmos import load_aecmos, compute_aecmos  # noqa: F401
