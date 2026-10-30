"""
Stage 统计报告工具。

每个 stage orchestrator 在处理完成后可调用 `write_stage_stats`，
输出结构化的 JSON 统计文件，便于审计与后续分析。
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .manifest_io import AudioRecord, read_manifest, read_manifest_with_progress

logger = logging.getLogger(__name__)


def _safe_label(record: AudioRecord) -> Optional[str]:
    """从 record 中提取一个代表性标签。"""
    for key in ("leaf", "audioset", "original"):
        label = record.labels.get(key)
        if label is None:
            continue
        if isinstance(label, str):
            return label
        if isinstance(label, list) and label:
            return label[0]
    return None


# 大阶段的子阶段键（如 B1..B4）按 "{stage}{数字}" 模式从记录上实际出现的
# 标记动态推导，不再硬编码子阶段列表：子阶段增删时统计口径自动跟随，
# 避免 stats 与各 orchestrator 的子阶段定义两处维护导致静默出错
_SUBSTAGE_PATTERN = re.compile(r"^([A-Z]+)(\d+)$")


def _substages_of(rec: AudioRecord, stage_name: str) -> List[str]:
    """返回记录上实际标记过的、属于 stage_name 的子阶段键（按数字序）。"""
    keys = []
    for key in rec.stages:
        m = _SUBSTAGE_PATTERN.match(key)
        if m and m.group(1) == stage_name:
            keys.append((int(m.group(2)), key))
    return [key for _, key in sorted(keys)]


def _record_stage_status(rec: AudioRecord, stage_name: str) -> str:
    """取记录在大阶段的状态：优先精确匹配，缺失时按子阶段聚合。

    聚合规则：任一子阶段 rejected 即 rejected；否则取最后一个出现的子阶段
    状态（passed / skipped）；无标记为 missing。

    注意：stage B/C/D 的尾部日志直接复用本聚合结果统计"末子阶段 passed"
    条数，依赖的不变量是"记录链路的最后一个子阶段 passed ⇒ 聚合状态为
    passed"（skipped/rejected 透传记录的末子阶段不可能是 passed）。
    """
    exact = rec.stages.get(stage_name)
    if isinstance(exact, dict) and exact.get("status"):
        return exact["status"]
    present = [
        rec.stages[s].get("status")
        for s in _substages_of(rec, stage_name)
        if isinstance(rec.stages.get(s), dict) and rec.stages[s].get("status")
    ]
    if not present:
        return "missing"
    if "rejected" in present:
        return "rejected"
    if present[-1] in ("passed", "skipped"):
        return present[-1]
    return "other"


def _duration_bucket(duration: float) -> str:
    if duration <= 0:
        return "unknown"
    if duration < 2:
        return "<2s"
    if duration < 5:
        return "2-5s"
    if duration < 10:
        return "5-10s"
    if duration < 20:
        return "10-20s"
    return ">=20s"


def compute_stage_stats(
    manifest_path: str,
    stage_name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """计算某个 stage 输出 manifest 的统计信息。"""
    total = 0
    passed = 0
    rejected = 0
    skipped = 0
    total_duration = 0.0
    # 只保留 running min/max，不把百万级 duration 全量物化为 list
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    label_counter: Counter = Counter()
    dataset_counter: Counter = Counter()
    duration_buckets: Counter = Counter()
    substage_counters: Dict[str, Counter] = defaultdict(Counter)

    # 大 manifest 的统计是小时级单遍扫描，用带 30s 进度打印的读取器，
    # 避免长时间无输出被误判为卡死
    for rec in read_manifest_with_progress(manifest_path, f"{stage_name} 统计"):
        total += 1
        status = _record_stage_status(rec, stage_name)
        if status == "passed":
            passed += 1
        elif status == "rejected":
            rejected += 1
        elif status == "skipped":
            skipped += 1
        for s in _substages_of(rec, stage_name):
            sub_status = rec.stages.get(s, {}).get("status")
            if sub_status:
                substage_counters[s][sub_status] += 1

        duration = rec.audio.get("duration", 0.0) or 0.0
        total_duration += duration
        min_duration = duration if min_duration is None else min(min_duration, duration)
        max_duration = duration if max_duration is None else max(max_duration, duration)
        duration_buckets[_duration_bucket(duration)] += 1

        dataset_counter[rec.dataset_id] += 1
        label = _safe_label(rec)
        if label:
            label_counter[label] += 1

    stats = {
        "stage": stage_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": total,
        "status": {
            "passed": passed,
            "rejected": rejected,
            "skipped": skipped,
            "other": total - passed - rejected - skipped,
        },
        # 筛选率：passed/total 与 rejected/total（四舍五入 4 位小数）。
        # 注意分母是 manifest 全部记录数，包含 skipped（透传）与 other
        # （未标记/异常状态）记录，因此 pass_rate + reject_rate 一般 ≤ 1，
        # skipped 占比高时两者会显著小于实际筛选比例
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "reject_rate": round(rejected / total, 4) if total else 0.0,
        "duration": {
            "total_seconds": round(total_duration, 2),
            "mean_seconds": round(total_duration / max(1, total), 2),
            "min_seconds": round(min_duration, 2) if min_duration is not None else 0.0,
            "max_seconds": round(max_duration, 2) if max_duration is not None else 0.0,
            "buckets": dict(duration_buckets),
        },
        "top_labels": dict(label_counter.most_common(20)),
        "datasets": dict(dataset_counter),
    }
    substage_stats = {s: dict(c) for s, c in substage_counters.items() if c}
    if substage_stats:
        stats["substages"] = substage_stats
    if extra:
        stats["extra"] = extra
    return stats


def write_stage_stats(
    manifest_path: str,
    stage_name: str,
    output_path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """写入 stage 统计 JSON 文件，返回 stats 字典（供调用方复用计数，
    避免为日志再全量解析一遍 manifest）。"""
    stats = compute_stage_stats(manifest_path, stage_name, extra)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("Stage %s 统计报告: %s", stage_name, output_path)
    return stats
