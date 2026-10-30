"""Speech Separation model registry and construction."""

from __future__ import annotations

import logging

from core.models import register_lazy_models
from core.models.build import build_model as _build_model
from core.utils.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

_MODEL_REGISTRY_MAP = {
    "TDANet": "tasks.ss.models.tdanet",
    "TIGER": "tasks.ss.models.tiger",
    "SpatialNet": "tasks.ss.models.spatialnet",
    "OnlineSpatialNet": "tasks.ss.models.online_spatialnet",
}
SS_MODEL_TYPES = frozenset(_MODEL_REGISTRY_MAP)
MODEL_REGISTRY.set_lazy_map(_MODEL_REGISTRY_MAP)


def register_ss_models(only: str | None = None) -> None:
    """Register SS model class(es) through the shared task facade."""
    register_lazy_models(
        _MODEL_REGISTRY_MAP,
        task="ss",
        only=only,
        logger=logger,
    )


def build_model(
    cfg: dict,
    stft_cfg: dict | None = None,
    *,
    num_sources: int | None = None,
):
    model_cfg = dict(cfg)
    args = dict(model_cfg.get("args", {}))
    if num_sources is not None:
        if type(num_sources) is not int or num_sources < 1:
            raise ValueError("num_sources must be an exact positive int")
        explicit = args.get("num_sources")
        if explicit is not None and explicit != num_sources:
            raise ValueError(
                "model.args.num_sources conflicts with canonical ss.num_sources: "
                f"{explicit} != {num_sources}"
            )
        args["num_sources"] = num_sources
    model_cfg["args"] = args
    return _build_model(model_cfg, stft_cfg=stft_cfg)


__all__ = ["build_model", "register_ss_models", "SS_MODEL_TYPES"]
