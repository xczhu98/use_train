"""AEC ERL configured/physical range coupling contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from core.data.errors import SimulationContractError


ERL_RANGE_EMPTY_INTERSECTION = "erl_configured_physical_range_empty"


class ERLCouplingContractError(SimulationContractError):
    """ERL 配置与当前物理场景不兼容，调用方不得降级为占位样本。"""

    def __init__(self, reason: str, metadata: dict[str, object]) -> None:
        self.reason = str(reason)
        self.metadata = metadata
        super().__init__(f"AEC ERL coupling contract failed: {self.reason}; metadata={metadata}")


@dataclass(frozen=True)
class ERLCouplingRange:
    """ERL configured、physical 与 effective 三层闭区间。"""

    configured: tuple[float, float]
    physical: tuple[float, float]
    effective: tuple[float, float]
    far_db_range: tuple[float, float]
    echo_db: float

    def to_audit(self, selected_erl_db: float) -> dict[str, object]:
        """返回稳定、JSON-safe 的抽样审计字段。"""
        selected = _finite_float(selected_erl_db, "selected_erl_db")
        return {
            "configured_range_db": list(self.configured),
            "physical_range_db": list(self.physical),
            "effective_range_db": list(self.effective),
            "selected_erl_db": selected,
        }


def resolve_erl_coupling_range(
    configured_range: Sequence[float],
    far_db_range: Sequence[float],
    echo_db: float,
) -> ERLCouplingRange:
    """求 configured ERL 与当前 far/echo 物理可行域的闭区间交集。

    ``lo == hi`` 是合法钉死区间；空交集抛出带稳定 reason 和 JSON-safe
    metadata 的 :class:`ERLCouplingContractError`。
    """
    configured = _closed_range(configured_range, "configured_range")
    far_range = _closed_range(far_db_range, "far_db_range")
    echo = _finite_float(echo_db, "echo_db")
    physical = (far_range[0] - echo, far_range[1] - echo)
    effective = (max(configured[0], physical[0]), min(configured[1], physical[1]))
    if effective[0] > effective[1]:
        raise ERLCouplingContractError(
            ERL_RANGE_EMPTY_INTERSECTION,
            {
                "configured_range_db": list(configured),
                "physical_range_db": list(physical),
                "effective_range_db": None,
                "far_db_range_db": list(far_range),
                "echo_db": echo,
            },
        )
    return ERLCouplingRange(
        configured=configured,
        physical=physical,
        effective=effective,
        far_db_range=far_range,
        echo_db=echo,
    )


def _closed_range(values: Sequence[float], name: str) -> tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ERLCouplingContractError(
            f"erl_{name}_invalid",
            {"field": name, "value": _json_safe_sequence(values)},
        )
    lo = _finite_float(values[0], f"{name}[0]")
    hi = _finite_float(values[1], f"{name}[1]")
    if lo > hi:
        raise ERLCouplingContractError(
            f"erl_{name}_invalid",
            {"field": name, "range_db": [lo, hi]},
        )
    return lo, hi


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        result = math.nan
    else:
        try:
            result = float(value)
        except (TypeError, ValueError):
            result = math.nan
    if not math.isfinite(result):
        raise ERLCouplingContractError(
            f"erl_{name.split('[', 1)[0]}_invalid",
            {"field": name, "value": None},
        )
    return result


def _json_safe_sequence(value: object) -> list[object] | None:
    if not isinstance(value, (list, tuple)):
        return None
    safe: list[object] = []
    for item in value:
        if isinstance(item, bool):
            safe.append(item)
        elif isinstance(item, (str, int)) or item is None:
            safe.append(item)
        elif isinstance(item, float) and math.isfinite(item):
            safe.append(item)
        else:
            safe.append(None)
    return safe
