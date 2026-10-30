"""TSE model registration and construction.

Available TSE models (specify via ``model.type`` in YAML):

  - **TSE_BSRNN_SPK**: BandSplit-RNN 目标说话人提取（dict 谱进谱出），
    5 种可插拔 speaker cue（listen/usef/tfmap/context/spkemb），
    speaker encoder 为 vendored ECAPA-TDNN（去 wespeaker 依赖）。
  - **TSE_Modular**: runtime STFT + BSRNN，speaker/spatial/visual/textual cue；
    speaker USEF append、visual 帧级条件与 textual 句级 KCE 条件为本地接线，
    visual raw video 与 listen/其他 backbone 尚未实现。

Uses shared ``MODEL_REGISTRY`` from ``core.utils.registry``.
"""

import logging

from core.models import register_lazy_models
from core.utils.registry import MODEL_REGISTRY  # noqa: F401

logger = logging.getLogger(__name__)

TSE_MODEL_TYPES: set[str] = {
    "TSE_BSRNN_SPK",
    "TSE_Modular",
}

_TSE_MODEL_REGISTRY_MAP: dict[str, str] = {
    "TSE_BSRNN_SPK": "tasks.tse.models.discriminative.tse_bsrnn_spk",
    "TSE_Modular": "tasks.tse.models.discriminative.tse_modular",
}


MODEL_REGISTRY.set_lazy_map(_TSE_MODEL_REGISTRY_MAP)


def register_tse_models(only: str | None = None) -> None:
    """Register TSE model class(es) into the shared ``MODEL_REGISTRY``.

    Args:
        only: If set (e.g. from ``cfg["model"]["type"]``), import **only** that model.
            If ``None``, register every entry in ``_TSE_MODEL_REGISTRY_MAP``.
    """
    register_lazy_models(
        _TSE_MODEL_REGISTRY_MAP,
        task="tse",
        only=only,
        logger=logger,
    )
