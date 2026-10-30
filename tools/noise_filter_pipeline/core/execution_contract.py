"""Stage C 执行模式与 strict manifest 发布资格契约。"""
from __future__ import annotations

from typing import Any, Mapping

C1_EXECUTION_MODES = frozenset({"production", "dry_run"})
C2_EXECUTION_MODES = frozenset({"subprocess_daemon", "dry_run", "disabled"})


def build_stage_c_execution_contract(
    c1_config: Mapping[str, Any],
    c2_config: Mapping[str, Any],
) -> dict[str, Any]:
    """从现有 C1/C2 配置派生唯一的发布资格，不引入新配置入口。"""
    c1_mode = "dry_run" if bool(c1_config.get("dry_run", False)) else "production"
    c2_mode = (
        "disabled"
        if not bool(c2_config.get("enable", True))
        else c2_config.get("mode", "dry_run")
    )
    execution_mode = {"c1": c1_mode, "c2": c2_mode}
    publishable = c1_mode == "production" and c2_mode == "subprocess_daemon"
    return {"execution_mode": execution_mode, "publishable": publishable}


def validate_stage_c_execution_contract(
    execution_mode: Any,
    publishable: Any,
    *,
    location: str,
) -> tuple[dict[str, str], bool]:
    """严格校验执行模式及派生发布资格；缺失、未知或矛盾时拒绝。"""
    if not isinstance(execution_mode, Mapping):
        raise ValueError(f"{location}: Stage C execution_mode must be an object")
    expected_fields = {"c1", "c2"}
    missing = expected_fields - set(execution_mode)
    unknown = set(execution_mode) - expected_fields
    if missing or unknown:
        raise ValueError(
            f"{location}: invalid Stage C execution_mode fields: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    c1_mode = execution_mode.get("c1")
    c2_mode = execution_mode.get("c2")
    if c1_mode not in C1_EXECUTION_MODES:
        raise ValueError(f"{location}: invalid C1 execution mode: {c1_mode!r}")
    if c2_mode not in C2_EXECUTION_MODES:
        raise ValueError(f"{location}: invalid C2 execution mode: {c2_mode!r}")
    expected_publishable = c1_mode == "production" and c2_mode == "subprocess_daemon"
    if type(publishable) is not bool or publishable is not expected_publishable:
        raise ValueError(
            f"{location}: Stage C publishable must be {expected_publishable} "
            "for the declared execution_mode"
        )
    return {"c1": c1_mode, "c2": c2_mode}, publishable


def write_stage_c_processor_mode(provenance: Any, stage: str, mode: str) -> dict[str, Any]:
    """把单个 processor 的执行模式合并到记录 provenance。"""
    normalized = dict(provenance) if isinstance(provenance, Mapping) else {}
    current = normalized.get("stage_c_execution")
    execution = dict(current) if isinstance(current, Mapping) else {}
    execution[stage] = mode
    normalized["stage_c_execution"] = execution
    return normalized
