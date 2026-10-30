"""噪声库统计聚合器（promote 等发布流程共用）。

当前 ``dataset_label`` 布局由 promote 直接重算统计，无需 rebuild。旧 ``sr_*``
布局只属于兼容迁移背景；``LibraryStats`` 独立保留在 core 供发布流程使用。
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class LibraryStats:
    """全库统计聚合器（与运行报告第三/六章口径一致）。

    by_label 以**标签目录名**（sanitize 后的 key）聚合：sanitize 非单射
    （"A, B" 与 "A B" 同为 ``A_B``），直接按原始标签字符串聚合会虚增类别数；
    目录名才是库内真实的类别边界。display 字段保留一个代表性原始标签名。
    """

    def __init__(self) -> None:
        self.total_chunks = 0
        self.total_duration_s = 0.0
        self.by_dataset: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"chunks": 0, "duration_s": 0.0, "sample_rates": Counter()}
        )
        self.by_sample_rate: Counter = Counter()
        self.by_label: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"chunks": 0, "duration_s": 0.0, "display": ""}
        )

    def add(
        self,
        dataset_id: str,
        label_key: str,
        label_display: str,
        sr: Optional[int],
        duration: Optional[float],
    ) -> None:
        for field_name, value in (
            ("dataset_id", dataset_id),
            ("label_key", label_key),
            ("label_display", label_display),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} 必须是非空字符串")
        if duration is None:
            dur = 0.0
        elif isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError("duration 必须是非负有限数值或 None")
        else:
            dur = float(duration)
            if not math.isfinite(dur) or dur < 0:
                raise ValueError("duration 必须是非负有限数值或 None")
        if sr is not None and (type(sr) is not int or sr <= 0):
            raise ValueError("sr 必须是正整数或 None")

        self.total_chunks += 1
        self.total_duration_s += dur
        ds = self.by_dataset[dataset_id]
        ds["chunks"] += 1
        ds["duration_s"] += dur
        if sr is not None:
            ds["sample_rates"][sr] += 1
            self.by_sample_rate[sr] += 1
        lab = self.by_label[label_key]
        lab["chunks"] += 1
        lab["duration_s"] += dur
        if not lab["display"]:
            lab["display"] = label_display

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_chunks": self.total_chunks,
            "total_duration_hours": round(self.total_duration_s / 3600.0, 2),
            "dataset_count": len(self.by_dataset),
            "label_count": len(self.by_label),
            "by_dataset": {
                ds: {
                    "chunks": v["chunks"],
                    "duration_hours": round(v["duration_s"] / 3600.0, 2),
                    "sample_rates": {str(k): c for k, c in sorted(v["sample_rates"].items())},
                }
                for ds, v in sorted(self.by_dataset.items())
            },
            "by_sample_rate": {str(k): c for k, c in sorted(self.by_sample_rate.items())},
            "by_label": {
                lab: {
                    "chunks": v["chunks"],
                    "duration_hours": round(v["duration_s"] / 3600.0, 2),
                    "display": v["display"],
                }
                for lab, v in sorted(self.by_label.items(), key=lambda kv: -kv[1]["chunks"])
            },
        }
