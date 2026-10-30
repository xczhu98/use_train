"""任务无关的电平审计 Module。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class LevelAudit:
    """一个命名比例从候选抽样到最终实测的 JSON-safe 审计。"""

    name: str
    candidate_drawn_db: float | None
    selected_nominal_db: float | None
    calibration_realized_db: float | None
    final_realized_db: float | None
    numerator: str
    denominator: str
    policy: str
    calibration_measure_point: str
    final_measure_point: str
    applicable: bool
    reason: str | None = None
    observations: Mapping[str, float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.applicable and self.reason is not None:
            raise ValueError("applicable LevelAudit cannot carry a reason")
        if not self.applicable and not self.reason:
            raise ValueError("non-applicable LevelAudit requires a reason")
        for name in (
            "candidate_drawn_db",
            "selected_nominal_db",
            "calibration_realized_db",
            "final_realized_db",
        ):
            value = getattr(self, name)
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or None")

    def with_final(
        self,
        value: float | None,
        *,
        measure_point: str,
        applicable: bool,
        reason: str | None = None,
    ) -> "LevelAudit":
        """返回补齐 final stage 的新审计，不修改原对象。"""
        return replace(
            self,
            final_realized_db=value,
            final_measure_point=str(measure_point),
            applicable=bool(applicable),
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        """递归转换为严格 JSON-safe Python 值。"""
        return json_safe(asdict(self))


def json_safe(value: Any) -> Any:
    """递归转换 numpy 类型，并把非有限浮点降级为 ``None``。"""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        items = value.tolist() if isinstance(value, np.ndarray) else value
        return [json_safe(item) for item in items]
    return None
