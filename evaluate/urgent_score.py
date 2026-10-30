"""URGENT normalized score 的唯一结构化合同。"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

SCORE_FILENAME = "urgent_score.json"
SCORE_SCHEMA = "urgent-normalized-score-v2"
PROJECTION_SCHEMA = "urgent-normalized-score-projection-v2"
RESULTS_FILENAME = "RESULTS.txt"
NORMALIZED_METRICS = (
    "DNSMOS",
    "NISQA",
    "UTMOS",
    "SCOREQ",
    "PESQ",
    "ESTOI",
)
NORMALIZATION_RANGES = {
    "DNSMOS": (1.0, 5.0),
    "NISQA": (1.0, 5.0),
    "UTMOS": (1.0, 5.0),
    "SCOREQ": (1.0, 5.0),
    "PESQ": (-0.5, 4.5),
    "ESTOI": (0.0, 1.0),
}


class ResultsBindingMismatch(ValueError):
    """sidecar 与 sibling ``RESULTS.txt`` 的内容身份不一致（旧 runner 可忽略）。"""


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize_metric(value: float, metric: str) -> float:
    """按 URGENT 固定区间裁剪并归一化单项指标。"""
    if metric not in NORMALIZATION_RANGES:
        raise ValueError(f"unsupported normalized metric: {metric}")
    if not _finite_number(value):
        raise ValueError(f"metric value must be a finite number: {value!r}")
    lower, upper = NORMALIZATION_RANGES[metric]
    clipped = max(lower, min(upper, float(value)))
    return (clipped - lower) / (upper - lower)


def _coverage_entry(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {
            "eligible": None,
            "computed": None,
            "coverage": None,
            "status": "unknown",
        }
    eligible = value.get("eligible")
    computed = value.get("computed")
    if (
        not isinstance(eligible, int)
        or isinstance(eligible, bool)
        or eligible < 0
        or not isinstance(computed, int)
        or isinstance(computed, bool)
        or computed < 0
        or computed > eligible
    ):
        return {
            "eligible": None,
            "computed": None,
            "coverage": None,
            "status": "unknown",
        }
    complete = eligible > 0 and computed == eligible
    return {
        "eligible": eligible,
        "computed": computed,
        "coverage": computed / eligible if eligible else None,
        "status": "complete" if complete else "partial",
    }


def _dimensions(values: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    raw_values = {
        metric: float(values[metric])
        for metric in NORMALIZED_METRICS
        if _finite_number(values.get(metric))
    }
    missing = [metric for metric in NORMALIZED_METRICS if metric not in raw_values]
    return (
        {
            "valid": len(raw_values),
            "required": len(NORMALIZED_METRICS),
            "missing": missing,
        },
        raw_values,
    )


def _status(dimensions: Mapping[str, Any], coverage: Mapping[str, Mapping[str, Any]]) -> str:
    if dimensions["valid"] == 0:
        return "unavailable"
    if any(item["status"] == "unknown" for item in coverage.values()):
        return "legacy-unknown"
    if dimensions["valid"] == dimensions["required"] and all(
        item["status"] == "complete" for item in coverage.values()
    ):
        return "official"
    return "partial"


def build_expected_metric_coverage(
    expected_population: int, metric_counts: Mapping[str, Any]
) -> dict[str, dict[str, int]]:
    """以本轮输入 UID population 为共同分母，失败样本只减少 computed。"""
    if (
        not isinstance(expected_population, int)
        or isinstance(expected_population, bool)
        or expected_population < 0
    ):
        raise ValueError("expected population must be a non-negative integer")
    coverage: dict[str, dict[str, int]] = {}
    for metric in NORMALIZED_METRICS:
        computed = metric_counts.get(metric, 0)
        if (
            not isinstance(computed, int)
            or isinstance(computed, bool)
            or computed < 0
            or computed > expected_population
        ):
            raise ValueError(f"invalid computed count for {metric}: {computed!r}")
        coverage[metric] = {
            "eligible": expected_population,
            "computed": computed,
        }
    return coverage


def _results_binding(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("invalid RESULTS binding")
    filename = value.get("filename")
    size = value.get("size")
    sha256 = value.get("sha256")
    if (
        filename != RESULTS_FILENAME
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or any(char not in "0123456789abcdef" for char in sha256)
    ):
        raise ValueError("invalid RESULTS binding")
    return {"filename": filename, "size": size, "sha256": sha256}


def build_normalized_score(
    values: Mapping[str, Any],
    metric_coverage: Optional[Mapping[str, Mapping[str, Any]]],
    *,
    results_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """从指标值、真实覆盖率与可选 RESULTS 身份构造 normalized score。"""
    dimensions, raw_values = _dimensions(values)
    normalized_values = {
        metric: normalize_metric(value, metric) for metric, value in raw_values.items()
    }
    coverage = {
        metric: _coverage_entry(
            metric_coverage.get(metric) if isinstance(metric_coverage, Mapping) else None
        )
        for metric in NORMALIZED_METRICS
    }
    value = (
        sum(normalized_values.values()) / len(normalized_values)
        if normalized_values
        else None
    )
    score = {
        "schema": SCORE_SCHEMA,
        "status": _status(dimensions, coverage),
        "value": value,
        "dimensions": dimensions,
        "sample_coverage": coverage,
        "raw_values": raw_values,
        "normalized_values": normalized_values,
    }
    binding = _results_binding(results_binding)
    if binding is not None:
        score["results"] = binding
    return score


def bind_normalized_score(
    score: Mapping[str, Any], results_fingerprint: Mapping[str, Any]
) -> dict[str, Any]:
    """将 score DTO 精确绑定到同目录 RESULTS.txt 的内容身份。"""
    binding = {
        "filename": RESULTS_FILENAME,
        "size": results_fingerprint.get("size"),
        "sha256": results_fingerprint.get("sha256"),
    }
    return build_normalized_score(
        score.get("raw_values", {}),
        score.get("sample_coverage"),
        results_binding=binding,
    )


def _projection_coverage_entry(
    metric: str,
    raw_values: Mapping[str, float],
    metric_coverage: Optional[Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    if metric not in raw_values:
        return {
            "eligible": None,
            "computed": None,
            "coverage": None,
            "status": "missing",
        }
    source = metric_coverage.get(metric) if isinstance(metric_coverage, Mapping) else None
    computed = source.get("computed") if isinstance(source, Mapping) else None
    if (
        isinstance(computed, int)
        and not isinstance(computed, bool)
        and computed >= 0
    ):
        return {
            "eligible": None,
            "computed": computed,
            "coverage": None,
            "status": "observed",
        }
    return {
        "eligible": None,
        "computed": None,
        "coverage": None,
        "status": "unknown",
    }


def project_existing_score(
    values: Mapping[str, Any],
    metric_coverage: Optional[Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """保留历史舍入值，只投影实际观测 count 与缺失维度。"""
    dimensions, raw_values = _dimensions(values)
    coverage = {
        metric: _projection_coverage_entry(metric, raw_values, metric_coverage)
        for metric in NORMALIZED_METRICS
    }
    existing = values.get("Normalized_Average")
    value = float(existing) if _finite_number(existing) else None
    observed = any(item["status"] == "observed" for item in coverage.values())
    status = "unavailable" if value is None else (
        "partial" if observed else "legacy-unknown"
    )
    return {
        "schema": PROJECTION_SCHEMA,
        "source": "historical-results-rounded",
        "status": status,
        "value": value,
        "dimensions": dimensions,
        "sample_coverage": coverage,
    }


def validate_normalized_score(
    payload: Any,
    *,
    expected_results: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """重建 score，并在读取 sidecar 时精确复核 sibling RESULTS 身份。"""
    if not isinstance(payload, Mapping) or payload.get("schema") != SCORE_SCHEMA:
        raise ValueError("unsupported URGENT normalized score schema")
    binding = _results_binding(payload.get("results"))
    if expected_results is not None:
        expected_binding = _results_binding(expected_results)
        if binding is None or binding != expected_binding:
            raise ResultsBindingMismatch("URGENT normalized score RESULTS binding mismatch")
    expected = build_normalized_score(
        payload.get("raw_values", {}),
        payload.get("sample_coverage"),
        results_binding=binding,
    )
    if dict(payload) != expected:
        raise ValueError("URGENT normalized score payload is inconsistent")
    return expected
