"""TSE dynamic/offline dataset 的统一运行时接口。"""

from __future__ import annotations

from typing import Any

from tasks.tse.cue_contract import TSECueSpecs


def validate_tse_dataset_interface(dataset: Any):
    """校验 dynamic/offline Dataset 都提供 canonical TSE 数据面。"""

    cue_specs = getattr(dataset, "cue_specs", None)
    if not isinstance(cue_specs, TSECueSpecs):
        raise TypeError("TSE dataset must expose cue_specs as TSECueSpecs")
    contract = getattr(dataset, "contract", None)
    if contract is not None and getattr(contract, "task", None) != "tse":
        raise TypeError("TSE dataset contract.task must be 'tse'")
    return dataset


__all__ = ["validate_tse_dataset_interface"]
