"""AudioTag C2 事实格式化：固定 top-5 证据（排序自完整 527 内部排名）。

C2 只输出 ``audioset_top5``（恰好 5 项，prob 降序、index 升序），不再持久化
完整 527 分数数组；排序仍在内部基于完整 527 sigmoid 分数完成。
"""
from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


CLASSIFIER_LABEL_COUNT = 527
AUDIOTAG_TOP_K = 5
EVIDENCE_CONTRACT_VERSION = 5
SCORE_CONTRACT = "audioset_527_independent_sigmoid_v1"


@dataclass(frozen=True)
class LabelMetadata:
    """AudioSet canonical CSV 的严格索引与身份信息。"""

    index_to_mid: Mapping[int, str] = field(default_factory=dict)
    mid_to_index: Mapping[str, int] = field(default_factory=dict)
    mid_to_label: Mapping[str, str] = field(default_factory=dict)
    label_to_mid: Mapping[str, str] = field(default_factory=dict)
    label_space_sha256: str = ""

    @property
    def class_count(self) -> int:
        return len(self.index_to_mid)

    def label_of(self, mid: str) -> str:
        try:
            return self.mid_to_label[mid]
        except KeyError:
            raise ValueError(f"unknown classifier MID {mid!r}") from None

    def index_of(self, mid: str) -> int:
        try:
            return self.mid_to_index[mid]
        except KeyError:
            raise ValueError(f"unknown classifier MID {mid!r}") from None


def _default_label_dict_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "class_labels_indices.csv"


def load_label_metadata(label_dict_path: Optional[str] = None) -> LabelMetadata:
    """严格读取并绑定完整 527 类 canonical label CSV。"""
    path = Path(label_dict_path) if label_dict_path else _default_label_dict_path()
    raw_bytes = path.read_bytes()
    rows: list[tuple[int, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = [str(column).strip() for column in (reader.fieldnames or [])]
        if header != ["index", "mid", "display_name"]:
            raise ValueError(
                f"label dict header must be index,mid,display_name, got {header}"
            )
        for line_no, row in enumerate(reader, start=2):
            try:
                index = int(str(row["index"]).strip())
                mid = str(row["mid"]).strip()
                label = str(row["display_name"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"label dict invalid row {line_no}: {row!r}") from exc
            if not mid or not label:
                raise ValueError(f"label dict invalid row {line_no}: empty MID/label")
            rows.append((index, mid, label))

    if len(rows) != CLASSIFIER_LABEL_COUNT:
        raise ValueError(
            f"label dict must contain exactly {CLASSIFIER_LABEL_COUNT} rows, got {len(rows)}"
        )
    indices = [index for index, _, _ in rows]
    mids = [mid for _, mid, _ in rows]
    labels = [label for _, _, label in rows]
    if indices != list(range(CLASSIFIER_LABEL_COUNT)):
        raise ValueError("label dict indices must be ordered and contiguous 0..526")
    if len(set(mids)) != CLASSIFIER_LABEL_COUNT:
        raise ValueError("label dict contains duplicate mids")
    if len(set(labels)) != CLASSIFIER_LABEL_COUNT:
        raise ValueError("label dict contains duplicate display names")
    return LabelMetadata(
        index_to_mid={index: mid for index, mid, _ in rows},
        mid_to_index={mid: index for index, mid, _ in rows},
        mid_to_label={mid: label for _, mid, label in rows},
        label_to_mid={label: mid for _, mid, label in rows},
        label_space_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _normalize_score(value: Any, *, row_index: int, class_index: int) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"score[{row_index}][{class_index}] must be a non-bool number"
        )
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"score[{row_index}][{class_index}] must be numeric"
        ) from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(
            f"score[{row_index}][{class_index}] must be finite and within [0,1]"
        )
    return score


def format_audiotag_results(
    full_scores: Sequence[Sequence[Any]],
    label_metadata: LabelMetadata,
    record_ids: Optional[Sequence[str]] = None,
) -> list[Dict[str, Any]]:
    """格式化固定、稳定排序的 top-5（排序基于完整 527 分数内部计算）。"""
    if label_metadata.class_count != CLASSIFIER_LABEL_COUNT:
        raise ValueError(
            f"label metadata must contain {CLASSIFIER_LABEL_COUNT} classes, "
            f"got {label_metadata.class_count}"
        )
    batch_size = len(full_scores)
    if record_ids is not None and len(record_ids) != batch_size:
        raise ValueError("record_ids length must equal batch size")

    results: list[Dict[str, Any]] = []
    for row_index, row in enumerate(full_scores):
        if len(row) != CLASSIFIER_LABEL_COUNT:
            raise ValueError(
                f"score row {row_index} must contain exactly "
                f"{CLASSIFIER_LABEL_COUNT} values, got {len(row)}"
            )
        scores = [
            _normalize_score(value, row_index=row_index, class_index=class_index)
            for class_index, value in enumerate(row)
        ]
        ranked_indices = sorted(
            range(CLASSIFIER_LABEL_COUNT), key=lambda index: (-scores[index], index)
        )[:AUDIOTAG_TOP_K]
        top5 = []
        for index in ranked_indices:
            mid = label_metadata.index_to_mid[index]
            top5.append(
                {
                    "index": index,
                    "mid": mid,
                    "label": label_metadata.label_of(mid),
                    "prob": scores[index],
                }
            )
        result: Dict[str, Any] = {
            "audioset_top5": top5,
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "score_contract": SCORE_CONTRACT,
            "class_count": CLASSIFIER_LABEL_COUNT,
            "label_space_sha256": label_metadata.label_space_sha256,
        }
        if record_ids is not None:
            record_id = record_ids[row_index]
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"record_ids[{row_index}] must be a non-empty string")
            result["record_id"] = record_id
        results.append(result)
    return results


def build_dry_run_result(
    top_label: str,
    label_metadata: LabelMetadata,
    *,
    record_id: Optional[str] = None,
) -> Dict[str, Any]:
    """构造完整 527 维 one-hot dry-run 事实，不引入业务判断。"""
    if top_label not in label_metadata.label_to_mid:
        raise ValueError(
            f"dry-run top label {top_label!r} 不在 527 类 label dict 中"
        )
    scores = [0.0] * CLASSIFIER_LABEL_COUNT
    scores[label_metadata.index_of(label_metadata.label_to_mid[top_label])] = 1.0
    record_ids = [record_id] if record_id is not None else None
    return format_audiotag_results([scores], label_metadata, record_ids=record_ids)[0]
