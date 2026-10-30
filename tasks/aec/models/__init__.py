"""AEC model registration and construction.

Available AEC models (specify via ``model.type`` in YAML):

  - **DeepVQE**: End-to-end AEC, dual-branch encoder + alignment (~8M base / ~0.6M small)
  - **CAGCRN**: Cross-Attention Gated CRN (AECNS-CAGCRN-main)
  - **DBGNet**: Dual-Branch Guidance Encoder with optional frame-level state detector
  - **UltraDPCAEC**: Ultra Dual-Path Compression AEC (~110K params, AEC+NS joint)
  - **LocalVQE**: Streaming-friendly AEC with S4D bottleneck, CausalGroupNorm (~0.78M params)

Uses shared ``MODEL_REGISTRY`` from ``core.utils.registry``.
"""

import logging

from core.models import register_lazy_models
from core.utils.registry import MODEL_REGISTRY  # noqa: F401

logger = logging.getLogger(__name__)

AEC_MODEL_TYPES: set[str] = {
    "DeepVQE",
    "CAGCRN",
    "UltraDPCAEC",
    "DBGNet",
    "LocalVQE",
}

_AEC_MODEL_REGISTRY_MAP: dict[str, str] = {
    "DeepVQE": "tasks.aec.models.discriminative.deepvqe",
    "CAGCRN": "tasks.aec.models.discriminative.cagcrn",
    "UltraDPCAEC": "tasks.aec.models.discriminative.ultra_dpc_aec",
    "DBGNet": "tasks.aec.models.discriminative.dbgnet",
    "LocalVQE": "tasks.aec.models.discriminative.localvqe",
}


MODEL_REGISTRY.set_lazy_map(_AEC_MODEL_REGISTRY_MAP)


def register_aec_models(only: str | None = None) -> None:
    """Register AEC model class(es) into the shared ``MODEL_REGISTRY``.

    Args:
        only: If set (e.g. from ``cfg["model"]["type"]``), import **only** that model.
            If ``None``, register every entry in ``_AEC_MODEL_REGISTRY_MAP`` (for scripts that
            may build more than one type).
    """
    register_lazy_models(
        _AEC_MODEL_REGISTRY_MAP,
        task="aec",
        only=only,
        logger=logger,
    )
