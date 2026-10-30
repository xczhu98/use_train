#!/usr/bin/env python3
"""Compare PASE/StuPASE reproduction metrics against RESULTS.txt files.

Does NOT compute metrics itself -- run ``evaluate/eval_pipeline_se.py``
(or ``evaluate/evaluate_urgent.py``) first to produce ``RESULTS.txt`` files,
then pass their paths here to build a side-by-side comparison table. This
tool never fabricates numbers: any metric missing from an input file is
rendered as "-" rather than guessed.

Usage:
    python -m tools.ops.evaluation.compare_pase_metrics \
        --label PASE_v1 exp/pase_phase3_vocoder_dual_gan/eval/urgent2026/RESULTS.txt \
        --label StuPASE_v1 exp/stupase_phase2_flow/eval/urgent2026/RESULTS.txt \
        --output docs/reports/experiments/pase_compare.md
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_results_txt(path: Path) -> dict[str, float]:
    """Parse a RESULTS.txt file (``MetricName: value`` per line) into a dict."""
    metrics: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, _, value = line.partition(":")
        name = name.strip()
        value = value.strip()
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics


def build_table(labeled_results: list[tuple[str, dict[str, float]]]) -> str:
    """Render a Markdown comparison table. Missing metrics render as '-'."""
    all_metrics: list[str] = []
    seen = set()
    for _, metrics in labeled_results:
        for name in metrics:
            if name not in seen:
                seen.add(name)
                all_metrics.append(name)

    header = "| 指标 | " + " | ".join(label for label, _ in labeled_results) + " |"
    sep = "|---" * (len(labeled_results) + 1) + "|"
    rows = [header, sep]
    for name in all_metrics:
        cells = []
        for _, metrics in labeled_results:
            v = metrics.get(name)
            cells.append(f"{v:.4f}" if v is not None else "-")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PASE/StuPASE RESULTS.txt files")
    parser.add_argument(
        "--label", nargs=2, action="append", metavar=("NAME", "RESULTS_TXT_PATH"),
        required=True,
        help="Repeatable: a (label, RESULTS.txt path) pair, e.g. --label PASE_v1 exp/.../RESULTS.txt",
    )
    parser.add_argument("--output", default=None, help="Write the Markdown table to this file (also prints to stdout)")
    args = parser.parse_args()

    labeled_results = []
    for label, path_str in args.label:
        path = Path(path_str)
        if not path.is_file():
            print(f"[WARN] {label}: file not found: {path}, skipping")
            continue
        labeled_results.append((label, parse_results_txt(path)))

    if not labeled_results:
        print("[ERROR] No valid RESULTS.txt files provided.")
        return

    table = build_table(labeled_results)
    print(table)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table + "\n", encoding="utf-8")
        print(f"\n[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
