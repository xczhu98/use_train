#!/usr/bin/env python3
"""REAL-TSE 评测结果分组汇总（官方 aggregate_eval_summary.py 移植版）。

移植自官方 ``REAL-TSE-Challenge/utils/aggregate_eval_summary.py``：按 dataset /
language / overall 重新计算分组均值，生成两级表头的 ascii 汇总报表
``<output_name>_summary.txt``。可识别的 current ``tse_eval_summary.json`` 是本轮
事实源，只读取其中 active 且 ``computed > 0`` 的指标 CSV；summary 未提供显式
``summary.metrics`` 时，可由 ``computed > 0`` 推断。仅非空 ``run_id`` 触发同一
run 的 ``result_manifest.json`` 绑定。summary 缺失或损坏时 payload 为 ``None``，
回退 legacy 模式并固定读取官方约定的五个 CSV；summary 存在但没有
``summary.coverage`` 时 fail-closed，不读取任何指标 CSV。分组、列名映射、语言
归一化与报表格式逐行对齐官方实现；数值一律从 CSV 重算，不复用已有 summary TXT。

官方 CSV 约定（``eval_pipeline_tse.py`` 评测后自动导出到
``<output_dir>/eval_metrics/``，亦兼容官方工具链的扁平路径）：

- ``<name>_TER.csv``：``mixture_utterance,source,language,wer_or_cer``
- ``<name>_spk_similarity.csv``：``utterance,dataset,language,speaker_cosine_similarity,status``
  （enroll↔tse，仅 ``status=ok`` 行参与）
- ``<name>_spk_similarity_mixture_enrol.csv``：同上（enroll↔mixture baseline）
- ``<name>_dnsmos.csv``：``utterance,dataset,SIG,BAK,OVRL,P808,status``
- ``<name>_TSE_TIMING.csv``：``utterance,dataset,language,precision,recall,f1``

⚠️ 注意：官方开源工具链不含 leaderboard 的 dense ranking 排名（排名在官方
评测服务器侧计算），本脚本只做分组均值汇总。

Usage::

    python -m evaluate.aggregate_eval_summary --output_dir exp/eval/tse/tse_bsrnn_spk
    # 多目录：
    python -m evaluate.aggregate_eval_summary --output_dir exp/eval/tse/m1 --output_dir exp/eval/tse/m2
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluate.common import validate_result_manifest
from tasks.tse.inference import artifacts as tse_inference_artifacts

# 官方默认排序：dataset 按 REAL-T 五语料顺序，language 按 en/chs
DEFAULT_DATASET_ORDER = ["AISHELL-4", "AMI", "AliMeeting", "CHiME6", "DipCo"]
DEFAULT_LANG_ORDER = ["en", "chs"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate REAL-T evaluation CSVs into one summary report."
    )
    parser.add_argument(
        "--output_dir",
        action="append",
        required=True,
        help="Output result directory. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--output_txt_name",
        default=None,
        help="Output TXT filename under each output_dir. Default: <output_name>_summary.txt.",
    )
    parser.add_argument(
        "--metrics_subdir",
        default="eval_metrics",
        help=(
            "Subdirectory under each output_dir that stores metric CSVs. "
            "Default: eval_metrics. Falls back to legacy flat paths when files are missing."
        ),
    )
    return parser.parse_args()


def format_float(value: object) -> str:
    if value is None:
        return "nan"
    try:
        if pd.isna(value):
            return "nan"
    except TypeError:
        pass
    return f"{float(value):.6f}"


def build_ascii_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    lines = [sep, header_line, sep]
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |")
    lines.append(sep)
    return lines


def build_two_level_table(
    index_name: str,
    row_names: Sequence[str],
    grouped_columns: Sequence[tuple[str, str, str]],
    values: Dict[str, Dict[str, float]],
    samples: Dict[str, int] | None = None,
) -> List[str]:
    top_headers = [index_name]
    sub_headers = [""]
    if samples is not None:
        top_headers.append("samples")
        sub_headers.append("")
    top_headers += [group for group, _, _ in grouped_columns]
    sub_headers += [sub for _, sub, _ in grouped_columns]

    rows: List[List[str]] = []
    for row_name in row_names:
        row = [row_name]
        if samples is not None:
            row.append(str(samples.get(row_name, 0)))
        row_values = values.get(row_name, {})
        for _, _, metric_key in grouped_columns:
            row.append(format_float(row_values.get(metric_key)))
        rows.append(row)

    widths = [max(len(top_headers[i]), len(sub_headers[i])) for i in range(len(top_headers))]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines = [sep]
    lines.append("| " + " | ".join(top_headers[i].ljust(widths[i]) for i in range(len(top_headers))) + " |")
    lines.append("| " + " | ".join(sub_headers[i].ljust(widths[i]) for i in range(len(sub_headers))) + " |")
    lines.append(sep)
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |")
    lines.append(sep)
    return lines


def normalize_language(value: object) -> str:
    """语种归一化（官方口径：zh/zho/chs/cn/chinese/mandarin→chs，en/eng/english→en）。"""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in {"zh", "zho", "chs", "cn", "chinese", "mandarin"}:
        return "chs"
    if text in {"en", "eng", "english"}:
        return "en"
    return text


def ordered_names(names: Iterable[str], preferred: Sequence[str]) -> List[str]:
    clean = [str(name) for name in names if str(name)]
    seen = set()
    ordered = []
    for name in preferred:
        if name in clean and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in sorted(clean):
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def resolve_metric_csv(
    output_dir: Path, metrics_subdir: str, filename: str, *, required: bool = True
) -> Path | None:
    """解析指标 CSV；新版 coverage 产物可将未请求指标视为缺项。"""
    candidates: List[Path] = []
    subdir = metrics_subdir.strip().strip("/")
    if subdir and subdir != ".":
        candidates.append(output_dir / subdir / filename)
    candidates.append(output_dir / filename)

    for path in candidates:
        if path.is_file():
            return path

    if not required:
        return None
    raise FileNotFoundError(
        "Required CSV not found. Tried: {}".format(
            ", ".join(str(path) for path in candidates)
        )
    )


def mean_by_group(
    df: pd.DataFrame,
    group_col: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[group_col, *metrics])
    return (
        df.groupby(group_col, dropna=False)[list(metrics)]
        .mean(numeric_only=True)
        .reset_index()
    )


def mean_overall(
    df: pd.DataFrame,
    metrics: Sequence[str],
    row_name: str = "overall",
    key_col: str = "group",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[key_col, *metrics])

    overall = df[list(metrics)].mean(numeric_only=True).to_dict()
    overall[key_col] = row_name
    return pd.DataFrame([overall], columns=[key_col, *metrics])


def to_metric_map(
    df: pd.DataFrame,
    key_col: str,
    rename_map: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    if df.empty:
        return result

    for row in df.to_dict(orient="records"):
        key = str(row[key_col])
        result[key] = {}
        for src_col, dst_col in rename_map.items():
            result[key][dst_col] = row.get(src_col)
    return result


def merge_metric_maps(*metric_maps: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    merged: Dict[str, Dict[str, float]] = {}
    for metric_map in metric_maps:
        for name, metrics in metric_map.items():
            merged.setdefault(name, {}).update(metrics)
    return merged


def extract_language_maps(
    timing_df: pd.DataFrame,
    ter_df: pd.DataFrame,
    *utterance_dfs: pd.DataFrame,
) -> tuple[Dict[str, str], Dict[str, str]]:
    """从本轮含 language 的 CSV 补齐 DNSMOS 所需语种映射。"""
    utterance_to_lang: Dict[str, str] = {}
    dataset_to_lang: Dict[str, str] = {}

    for data in (timing_df, *utterance_dfs):
        if data.empty:
            continue
        for row in data[["utterance", "dataset", "language"]].dropna(
            subset=["utterance"]
        ).itertuples(index=False):
            lang = normalize_language(row.language)
            if lang:
                utterance_to_lang.setdefault(str(row.utterance), lang)
                dataset_to_lang.setdefault(str(row.dataset), lang)

    if not ter_df.empty:
        ter_lang_df = ter_df.copy()
        ter_lang_df["language"] = ter_lang_df["language"].map(normalize_language)
        for row in ter_lang_df[["mixture_utterance", "source", "language"]].dropna(
            subset=["mixture_utterance"]
        ).itertuples(index=False):
            lang = normalize_language(row.language)
            if lang:
                utterance_to_lang.setdefault(str(row.mixture_utterance), lang)
                dataset_to_lang.setdefault(str(row.source), lang)

    return utterance_to_lang, dataset_to_lang


def attach_language(
    df: pd.DataFrame,
    utterance_col: str,
    dataset_col: str,
    utterance_to_lang: Dict[str, str],
    dataset_to_lang: Dict[str, str],
) -> pd.DataFrame:
    data = df.copy()
    data["language"] = data[utterance_col].map(utterance_to_lang)
    fallback = data[dataset_col].map(dataset_to_lang)
    data["language"] = data["language"].fillna(fallback)
    data["language"] = data["language"].map(normalize_language)
    return data


def _load_summary_payload(output_dir: Path) -> dict[str, Any] | None:
    """读取可识别的本轮 summary；缺失、损坏或不识别时返回 ``None``。"""
    path = output_dir / "tse_eval_summary.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("summary"), dict):
        return None
    return payload


def _summary_metric_entries(payload: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    """读取 ``summary.coverage`` 的逐指标覆盖条目；无法识别时返回 ``None``。"""
    coverage = payload["summary"].get("coverage")
    if not isinstance(coverage, dict):
        return None
    return {
        str(metric): value
        for metric, value in coverage.items()
        if isinstance(value, dict)
    }


def load_coverage(output_dir: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    """读取 TSE summary coverage；summary 缺失或无 coverage 时显式返回 unknown。"""
    payload = _load_summary_payload(output_dir)
    if payload is None:
        return "unknown", {}
    coverage = _summary_metric_entries(payload)
    if coverage is None:
        return "unknown", {}
    status = payload["summary"].get("status")
    if status not in {"done", "partial", "error"}:
        status = "unknown"
    return status, coverage


def _metric_computed(entry: dict[str, Any]) -> int | None:
    """读取本轮该指标条目里的 ``computed`` 计数。"""
    value = entry.get("computed")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _metric_csv_policy(
    payload: dict[str, Any] | None,
) -> dict[str, bool] | None:
    """返回本轮允许读取 CSV 的 metric；``None`` 表示无 summary、回退官方五 CSV。"""
    if payload is None:
        return None
    entries = _summary_metric_entries(payload)
    if entries is None:
        # summary schema 不可识别时 fail-closed，避免误读 CSV。
        return {metric: False for metric in ("ter", "spksim", "dnsmos", "timing")}

    summary_metrics = payload["summary"].get("metrics")
    current_active = None
    current_skipped: set[str] = set()
    if isinstance(summary_metrics, dict):
        if isinstance(summary_metrics.get("active"), list):
            current_active = {str(metric) for metric in summary_metrics["active"]}
        skipped = summary_metrics.get("skipped")
        if isinstance(skipped, list):
            current_skipped = {
                str(item.get("metric") if isinstance(item, dict) else item)
                for item in skipped
            }

    policy: dict[str, bool] = {}
    for metric in ("ter", "spksim", "dnsmos", "timing"):
        entry = entries.get(metric, {})
        computed = _metric_computed(entry)
        explicit_active = entry.get("active")
        state = str(entry.get("status", "")).strip().lower()
        if metric in current_skipped:
            active = False
        elif current_active is not None:
            active = metric in current_active
        elif isinstance(explicit_active, bool):
            active = explicit_active
        elif state in {"skipped", "inactive", "disabled"}:
            active = False
        else:
            # coverage 缺少 active/metrics 时，以实际产出计数判定。
            active = computed is not None and computed > 0
        policy[metric] = active and computed is not None and computed > 0
    return policy


def format_coverage(value: Any) -> str:
    if not isinstance(value, dict):
        return "unknown"
    keys = ("eligible", "attempted", "computed", "failed")
    if any(isinstance(value.get(key), bool) or not isinstance(value.get(key), int)
           for key in keys):
        return "unknown"
    ratio = value.get("coverage")
    ratio_text = "n/a" if ratio is None else f"{float(ratio):.2%}"
    return (
        f"eligible={value['eligible']} attempted={value['attempted']} "
        f"computed={value['computed']} failed={value['failed']} coverage={ratio_text}"
    )


_RESULT_METRIC_KEYS = {
    "ter": ("ter",),
    "spksim": ("spksim_tse_enroll", "spksim_mix_enroll"),
    "dnsmos": ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"),
    "timing": ("timing_precision", "timing_recall", "timing_f1"),
}
_SAMPLE_ID_COLUMNS = (
    "dataset",
    "mixture_utterance",
    "enrolment_speakers_utterance",
    "language",
)


def _complete_metric_mask(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """按 ``run_evaluation._computed`` 的全字段有限值合同筛选结果行。"""
    if any(column not in df.columns for column in columns):
        return pd.Series(False, index=df.index, dtype=bool)
    numeric = df[list(columns)].apply(pd.to_numeric, errors="coerce")
    return numeric.apply(
        lambda row: all(math.isfinite(value) for value in row), axis=1
    )


def _sample_rows_from_results(
    results_df: pd.DataFrame,
    csv_policy: dict[str, bool],
) -> pd.DataFrame:
    """从内部逐行结果构造绑定 dataset+mixture+enrollment 的 metric UID union。"""
    if results_df.empty and not len(results_df.columns):
        return pd.DataFrame(columns=_SAMPLE_ID_COLUMNS)
    missing = [column for column in _SAMPLE_ID_COLUMNS if column not in results_df]
    if missing:
        raise ValueError(
            "tse_eval_results.csv missing sample identity columns: "
            + ", ".join(missing)
        )

    metric_rows = []
    for metric, result_keys in _RESULT_METRIC_KEYS.items():
        if not csv_policy.get(metric, False):
            continue
        mask = _complete_metric_mask(results_df, result_keys)
        if mask.any():
            metric_rows.append(results_df.loc[mask, list(_SAMPLE_ID_COLUMNS)])
    if not metric_rows:
        return pd.DataFrame(columns=_SAMPLE_ID_COLUMNS)
    return pd.concat(metric_rows, ignore_index=True)


def _sample_rows_from_official_csvs(
    csv_policy: dict[str, bool] | None,
    ter_work: pd.DataFrame,
    sim_work: pd.DataFrame,
    sim_baseline_work: pd.DataFrame,
    dnsmos_work: pd.DataFrame,
    timing_work: pd.DataFrame,
) -> pd.DataFrame:
    """legacy/缺内部 results 时按官方 utterance 合同兼容构造 UID union。"""
    enabled = lambda metric: csv_policy is None or csv_policy.get(metric, False)
    metric_rows: list[pd.DataFrame] = []
    if enabled("ter"):
        mask = _complete_metric_mask(ter_work, ("wer_or_cer",))
        metric_rows.append(ter_work.loc[mask, ["utterance", "dataset", "language"]])
    if enabled("spksim"):
        sim_mask = _complete_metric_mask(sim_work, ("speaker_cosine_similarity",))
        baseline_mask = _complete_metric_mask(
            sim_baseline_work, ("speaker_cosine_similarity",)
        )
        sim_keys = ["utterance", "dataset", "language"]
        complete_sim = sim_work.loc[sim_mask, sim_keys].merge(
            sim_baseline_work.loc[baseline_mask, sim_keys].drop_duplicates(),
            on=sim_keys,
            how="inner",
        )
        metric_rows.append(complete_sim)
    if enabled("dnsmos"):
        mask = _complete_metric_mask(dnsmos_work, ("SIG", "BAK", "OVRL", "P808"))
        metric_rows.append(
            dnsmos_work.loc[mask, ["utterance", "dataset", "language"]]
        )
    if enabled("timing"):
        mask = _complete_metric_mask(timing_work, ("precision", "recall", "f1"))
        metric_rows.append(
            timing_work.loc[mask, ["utterance", "dataset", "language"]]
        )
    nonempty = [frame for frame in metric_rows if not frame.empty]
    if not nonempty:
        return pd.DataFrame(columns=["utterance", "dataset", "language"])
    return pd.concat(nonempty, ignore_index=True)


def _summarize_one_output_dir_locked(
    output_dir: Path,
    output_txt_name: str | None,
    metrics_subdir: str,
) -> Path:
    output_name = output_dir.name
    output_txt = output_dir / (output_txt_name or f"{output_name}_summary.txt")
    summary_payload = _load_summary_payload(output_dir)
    coverage_status, coverage = load_coverage(output_dir)
    csv_policy = _metric_csv_policy(summary_payload)
    expected_run_id = None
    allowed_artifacts: set[str] | None = None
    if summary_payload is not None:
        candidate_run_id = summary_payload.get("run_id") or summary_payload["summary"].get(
            "run_id"
        )
        if candidate_run_id is not None:
            if not isinstance(candidate_run_id, str) or not candidate_run_id:
                raise ValueError("TSE summary run_id is invalid")
            expected_run_id = candidate_run_id
            manifest = validate_result_manifest(
                output_dir, expected_run_id=expected_run_id
            )
            allowed_artifacts = set(manifest["files"])

    def _resolve(metric: str, filename: str) -> Path | None:
        if csv_policy is not None and not csv_policy[metric]:
            return None
        path = resolve_metric_csv(output_dir, metrics_subdir, filename, required=True)
        if allowed_artifacts is not None:
            relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
            if relative not in allowed_artifacts:
                raise ValueError(f"metric CSV is not bound by result manifest: {relative}")
        return path

    ter_path = _resolve("ter", f"{output_name}_TER.csv")
    sim_path = _resolve("spksim", f"{output_name}_spk_similarity.csv")
    sim_baseline_path = _resolve(
        "spksim", f"{output_name}_spk_similarity_mixture_enrol.csv"
    )
    dnsmos_path = _resolve("dnsmos", f"{output_name}_dnsmos.csv")
    timing_path = _resolve("timing", f"{output_name}_TSE_TIMING.csv")

    def _read_or_empty(path: Path | None, columns: Sequence[str]) -> pd.DataFrame:
        return pd.read_csv(path) if path is not None else pd.DataFrame(columns=columns)

    ter_df = _read_or_empty(
        ter_path, ["mixture_utterance", "source", "language", "wer_or_cer"])
    sim_columns = [
        "utterance", "dataset", "language", "speaker_cosine_similarity", "status"
    ]
    sim_df = _read_or_empty(sim_path, sim_columns)
    sim_baseline_df = _read_or_empty(sim_baseline_path, sim_columns)
    dnsmos_df = _read_or_empty(
        dnsmos_path, ["utterance", "dataset", "SIG", "BAK", "OVRL", "P808", "status"])
    timing_df = _read_or_empty(
        timing_path, ["utterance", "dataset", "language", "precision", "recall", "f1"])

    utterance_to_lang, dataset_to_lang = extract_language_maps(
        timing_df, ter_df, sim_df, sim_baseline_df
    )

    ter_work = ter_df.copy()
    ter_work["language"] = ter_work["language"].map(normalize_language)
    ter_work = ter_work.rename(
        columns={"mixture_utterance": "utterance", "source": "dataset"}
    )

    sim_work = sim_df[sim_df["status"] == "ok"].copy()
    sim_work["language"] = sim_work["language"].map(normalize_language)

    sim_baseline_work = sim_baseline_df[sim_baseline_df["status"] == "ok"].copy()
    sim_baseline_work["language"] = sim_baseline_work["language"].map(normalize_language)

    timing_work = timing_df.copy()
    timing_work["language"] = timing_work["language"].map(normalize_language)

    dnsmos_work = dnsmos_df[dnsmos_df["status"] == "ok"].copy()
    dnsmos_work = attach_language(
        dnsmos_work,
        utterance_col="utterance",
        dataset_col="dataset",
        utterance_to_lang=utterance_to_lang,
        dataset_to_lang=dataset_to_lang,
    )

    ter_dataset = mean_by_group(ter_work, "dataset", ["wer_or_cer"]).rename(
        columns={"wer_or_cer": "ter_zipformer"}
    )
    ter_lang = mean_by_group(ter_work, "language", ["wer_or_cer"]).rename(
        columns={"language": "lang", "wer_or_cer": "ter_zipformer"}
    )

    sim_dataset = mean_by_group(sim_work, "dataset", ["speaker_cosine_similarity"]).rename(
        columns={"speaker_cosine_similarity": "sim_enrol_tse"}
    )
    sim_lang = mean_by_group(sim_work, "language", ["speaker_cosine_similarity"]).rename(
        columns={"language": "lang", "speaker_cosine_similarity": "sim_enrol_tse"}
    )

    sim_baseline_dataset = mean_by_group(
        sim_baseline_work, "dataset", ["speaker_cosine_similarity"]
    ).rename(columns={"speaker_cosine_similarity": "sim_enrol_mixture"})
    sim_baseline_lang = mean_by_group(
        sim_baseline_work, "language", ["speaker_cosine_similarity"]
    ).rename(columns={"language": "lang", "speaker_cosine_similarity": "sim_enrol_mixture"})

    dnsmos_dataset = mean_by_group(dnsmos_work, "dataset", ["SIG", "BAK", "OVRL", "P808"]).rename(
        columns={
            "SIG": "dnsmos_sig",
            "BAK": "dnsmos_bak",
            "OVRL": "dnsmos_ovrl",
            "P808": "dnsmos_p808",
        }
    )
    dnsmos_lang = mean_by_group(dnsmos_work, "language", ["SIG", "BAK", "OVRL", "P808"]).rename(
        columns={
            "language": "lang",
            "SIG": "dnsmos_sig",
            "BAK": "dnsmos_bak",
            "OVRL": "dnsmos_ovrl",
            "P808": "dnsmos_p808",
        }
    )

    timing_dataset = mean_by_group(timing_work, "dataset", ["precision", "recall", "f1"]).rename(
        columns={"precision": "ratio_precision", "recall": "ratio_recall", "f1": "ratio_f1"}
    )
    timing_lang = mean_by_group(timing_work, "language", ["precision", "recall", "f1"]).rename(
        columns={"language": "lang", "precision": "ratio_precision", "recall": "ratio_recall", "f1": "ratio_f1"}
    )

    ter_overall = mean_overall(ter_work, ["wer_or_cer"], key_col="scope").rename(
        columns={"wer_or_cer": "ter_zipformer"}
    )
    sim_overall = mean_overall(sim_work, ["speaker_cosine_similarity"], key_col="scope").rename(
        columns={"speaker_cosine_similarity": "sim_enrol_tse"}
    )
    sim_baseline_overall = mean_overall(
        sim_baseline_work, ["speaker_cosine_similarity"], key_col="scope"
    ).rename(columns={"speaker_cosine_similarity": "sim_enrol_mixture"})
    dnsmos_overall = mean_overall(dnsmos_work, ["SIG", "BAK", "OVRL", "P808"], key_col="scope").rename(
        columns={
            "SIG": "dnsmos_sig",
            "BAK": "dnsmos_bak",
            "OVRL": "dnsmos_ovrl",
            "P808": "dnsmos_p808",
        }
    )
    timing_overall = mean_overall(timing_work, ["precision", "recall", "f1"], key_col="scope").rename(
        columns={"precision": "ratio_precision", "recall": "ratio_recall", "f1": "ratio_f1"}
    )

    dataset_values = merge_metric_maps(
        to_metric_map(ter_dataset, "dataset", {"ter_zipformer": "ter_zipformer"}),
        to_metric_map(sim_baseline_dataset, "dataset", {"sim_enrol_mixture": "sim_enrol_mixture"}),
        to_metric_map(sim_dataset, "dataset", {"sim_enrol_tse": "sim_enrol_tse"}),
        to_metric_map(
            dnsmos_dataset,
            "dataset",
            {
                "dnsmos_sig": "dnsmos_sig",
                "dnsmos_bak": "dnsmos_bak",
                "dnsmos_ovrl": "dnsmos_ovrl",
                "dnsmos_p808": "dnsmos_p808",
            },
        ),
        to_metric_map(
            timing_dataset,
            "dataset",
            {
                "ratio_precision": "ratio_precision",
                "ratio_recall": "ratio_recall",
                "ratio_f1": "ratio_f1",
            },
        ),
    )

    lang_values = merge_metric_maps(
        to_metric_map(ter_lang, "lang", {"ter_zipformer": "ter_zipformer"}),
        to_metric_map(sim_baseline_lang, "lang", {"sim_enrol_mixture": "sim_enrol_mixture"}),
        to_metric_map(sim_lang, "lang", {"sim_enrol_tse": "sim_enrol_tse"}),
        to_metric_map(
            dnsmos_lang,
            "lang",
            {
                "dnsmos_sig": "dnsmos_sig",
                "dnsmos_bak": "dnsmos_bak",
                "dnsmos_ovrl": "dnsmos_ovrl",
                "dnsmos_p808": "dnsmos_p808",
            },
        ),
        to_metric_map(
            timing_lang,
            "lang",
            {
                "ratio_precision": "ratio_precision",
                "ratio_recall": "ratio_recall",
                "ratio_f1": "ratio_f1",
            },
        ),
    )

    overall_values = merge_metric_maps(
        to_metric_map(ter_overall, "scope", {"ter_zipformer": "ter_zipformer"}),
        to_metric_map(sim_baseline_overall, "scope", {"sim_enrol_mixture": "sim_enrol_mixture"}),
        to_metric_map(sim_overall, "scope", {"sim_enrol_tse": "sim_enrol_tse"}),
        to_metric_map(
            dnsmos_overall,
            "scope",
            {
                "dnsmos_sig": "dnsmos_sig",
                "dnsmos_bak": "dnsmos_bak",
                "dnsmos_ovrl": "dnsmos_ovrl",
                "dnsmos_p808": "dnsmos_p808",
            },
        ),
        to_metric_map(
            timing_overall,
            "scope",
            {
                "ratio_precision": "ratio_precision",
                "ratio_recall": "ratio_recall",
                "ratio_f1": "ratio_f1",
            },
        ),
    )

    results_path = output_dir / "tse_eval_results.csv"
    results_relative = results_path.resolve().relative_to(output_dir.resolve()).as_posix()
    internal_results_allowed = (
        summary_payload is not None
        and allowed_artifacts is not None
        and results_relative in allowed_artifacts
        and results_path.is_file()
    )
    if internal_results_allowed:
        # validate_result_manifest 已在上方复核 manifest 内所有文件的内容哈希。
        results_df = pd.read_csv(results_path)
        sample_rows = _sample_rows_from_results(results_df, csv_policy or {})
        sample_uid_columns = [
            "dataset", "mixture_utterance", "enrolment_speakers_utterance"
        ]
        samples_basis = (
            "internal results composite UID union: unique "
            "(dataset, mixture_utterance, enrolment_speakers_utterance) across "
            "active metrics' complete computed rows"
        )
    else:
        sample_rows = _sample_rows_from_official_csvs(
            csv_policy,
            ter_work,
            sim_work,
            sim_baseline_work,
            dnsmos_work,
            timing_work,
        )
        sample_uid_columns = ["dataset", "utterance"]
        samples_basis = (
            "official CSV fallback UID union: unique (dataset, utterance); "
            "enrollment identity is unavailable"
        )

    sample_rows = sample_rows.dropna(subset=sample_uid_columns).copy()
    for column in sample_uid_columns:
        sample_rows[column] = sample_rows[column].astype(str).str.strip()
    sample_rows["dataset"] = sample_rows["dataset"].astype(str).str.strip()
    sample_rows["language"] = sample_rows["language"].map(normalize_language)
    sample_rows = sample_rows[
        sample_rows[sample_uid_columns].ne("").all(axis=1)
    ]

    dataset_uid_rows = sample_rows.drop_duplicates(sample_uid_columns)
    language_uid_rows = sample_rows[sample_rows["language"] != ""].drop_duplicates(
        ["language", *sample_uid_columns]
    )
    overall_uid_rows = sample_rows.drop_duplicates(sample_uid_columns)
    dataset_samples = dataset_uid_rows.groupby("dataset").size().to_dict()
    lang_samples = language_uid_rows.groupby("language").size().to_dict()
    overall_samples = {"overall": len(overall_uid_rows)}

    dataset_names = ordered_names(
        set(dataset_values) | set(dataset_samples), DEFAULT_DATASET_ORDER
    )
    lang_names = ordered_names(set(lang_values) | set(lang_samples), DEFAULT_LANG_ORDER)

    grouped_columns = [
        ("TER", "zipformer-zh/en", "ter_zipformer"),
        ("SIM", "enrol-mixture", "sim_enrol_mixture"),
        ("SIM", "enrol-tse", "sim_enrol_tse"),
        ("DNSMOS", "SIG", "dnsmos_sig"),
        ("DNSMOS", "BAK", "dnsmos_bak"),
        ("DNSMOS", "OVRL", "dnsmos_ovrl"),
        ("DNSMOS", "P808", "dnsmos_p808"),
        ("RATIO", "precision", "ratio_precision"),
        ("RATIO", "recall", "ratio_recall"),
        ("RATIO", "f1", "ratio_f1"),
    ]

    lines: List[str] = []
    lines.append("REAL-T Aggregated Evaluation Summary")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Output dir: {output_dir}")
    lines.append("")
    lines.append("Mean by dataset")
    lines.extend(build_two_level_table("dataset", dataset_names, grouped_columns, dataset_values, samples=dataset_samples))
    lines.append("")
    lines.append("Mean by language")
    lines.extend(build_two_level_table("lang", lang_names, grouped_columns, lang_values, samples=lang_samples))
    lines.append("")
    lines.append("Mean overall")
    lines.extend(build_two_level_table("scope", ["overall"], grouped_columns, overall_values, samples=overall_samples))
    lines.append("")
    lines.append("Coverage")
    lines.append(f"  status: {coverage_status}")
    if coverage:
        for metric in sorted(coverage):
            lines.append(f"  {metric}: {format_coverage(coverage[metric])}")
    else:
        lines.append("  metrics: unknown (summary missing or without coverage)")
    lines.append("")
    lines.append("Input CSVs")
    lines.extend(
        build_ascii_table(
            ["metric", "file"],
            [
                ["TER", str(ter_path)],
                ["SIM enrol-mixture", str(sim_baseline_path)],
                ["SIM enrol-tse", str(sim_path)],
                ["DNSMOS", str(dnsmos_path)],
                ["RATIO precision/recall/f1", str(timing_path)],
            ],
        )
    )
    lines.append("")
    lines.append("Notes")
    lines.append("  RATIO columns are the mean of precision / recall / f1 from TSE_TIMING CSV.")
    lines.append(f"  samples = {samples_basis}.")
    lines.append(
        "  language samples use the same UID union grouped by normalized language."
    )
    lines.append("  All values are recomputed from CSV files instead of reusing existing summary TXT files.")

    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_txt


def summarize_one_output_dir(
    output_dir: Path,
    output_txt_name: str | None,
    metrics_subdir: str,
) -> Path:
    """公开 writer：持有 TSE output-dir 锁后生成汇总文件。"""
    with tse_inference_artifacts.exclusive_output_run(output_dir) as locked_output:
        return _summarize_one_output_dir_locked(
            locked_output, output_txt_name, metrics_subdir
        )


def main() -> None:
    args = parse_args()
    for output_dir_raw in args.output_dir:
        output_dir = Path(output_dir_raw).resolve()
        if not output_dir.is_dir():
            raise SystemExit(f"Output directory not found: {output_dir}")
        output_txt = summarize_one_output_dir(
            output_dir=output_dir,
            output_txt_name=args.output_txt_name,
            metrics_subdir=args.metrics_subdir,
        )
        print(f"[Saved] {output_txt}")


if __name__ == "__main__":
    main()
