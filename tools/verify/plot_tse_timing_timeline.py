#!/usr/bin/env python3
"""基于 use_train 官方 timing 语义绘制 TSE GT/pred timeline。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _segments(value: object, label: str) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{label}[{index}] must be [start, end]")
        start, end = item
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(end) <= float(start)
        ):
            raise ValueError(f"{label}[{index}] has invalid bounds")
        result.append((float(start), float(end)))
    return result


def load_prediction_jsonl(path: Path) -> dict[str, list[tuple[float, float]]]:
    predictions = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != {"uid", "pred_segments"}:
                raise ValueError(
                    f"prediction line {line_number} must contain uid/pred_segments"
                )
            uid = value["uid"]
            if not isinstance(uid, str) or not uid or uid in predictions:
                raise ValueError(f"prediction line {line_number} has invalid/duplicate uid")
            predictions[uid] = _segments(
                value["pred_segments"], f"prediction line {line_number}.pred_segments"
            )
    return predictions


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [dict(row) for row in reader]
    required = {
        "dataset", "mixture_utterance", "enrolment_speakers_utterance", "speaker"
    }
    if not rows:
        raise ValueError(f"evaluation result CSV is empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"evaluation result CSV is missing columns: {sorted(missing)}")
    return rows


def plot_timeline(
    *,
    uid: str,
    row: dict[str, str],
    duration: float,
    gt_segments: list[tuple[float, float]],
    pred_segments: list[tuple[float, float]],
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch, Rectangle
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for timing timeline plots") from exc

    figure, axis = plt.subplots(figsize=(12, 3.4))
    for y, segments, color in (
        (1.0, gt_segments, "#2f855a"),
        (0.0, pred_segments, "#2b6cb0"),
    ):
        for start, end in segments:
            axis.add_patch(
                Rectangle(
                    (start, y - 0.28),
                    end - start,
                    0.56,
                    facecolor=color,
                    edgecolor="#1a202c",
                    linewidth=0.8,
                )
            )
    axis.set_xlim(0.0, duration)
    axis.set_ylim(-0.65, 1.65)
    axis.set_yticks([0.0, 1.0])
    axis.set_yticklabels(["TSE prediction", f"GT target ({row['speaker']})"])
    axis.set_xlabel("Time (s)")
    axis.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.45)
    axis.legend(
        handles=[
            Patch(facecolor="#2f855a", label="GT target + collar"),
            Patch(facecolor="#2b6cb0", label="VAD prediction"),
        ],
        loc="upper right",
    )
    axis.set_title(
        f"{row['dataset']} | {row['mixture_utterance']} | "
        f"enroll={row['enrolment_speakers_utterance']}"
    )
    figure.text(
        0.5,
        0.02,
        "UID={}  precision={:.4f}  recall={:.4f}  F1={:.4f}".format(
            uid,
            metrics["timing_precision"],
            metrics["timing_recall"],
            metrics["timing_f1"],
        ),
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(bottom=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot REAL-TSE timing GT/pred timelines")
    parser.add_argument("--eval-output-dir", type=Path, required=True)
    parser.add_argument("--timing-gt-json-base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, default=None)
    parser.add_argument("--timing-vad-backend", choices=("firered", "silero", "energy"), default="firered")
    parser.add_argument("--fireredasr2s-root", type=Path, default=None)
    parser.add_argument(
        "--allow-vad-fallback",
        action="store_true",
        help="FireRed VAD 不可用时降级到 silero/energy；每次降级都会在 stderr 打出 [WARN]",
    )
    parser.add_argument("--timing-frame-shift", type=float, default=0.01)
    parser.add_argument("--timing-collar", type=float, default=0.05)
    parser.add_argument("--timing-match-tolerance", type=float, default=0.02)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--uid", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    from evaluate.eval_pipeline_tse import (
        _EnergyVADSegments,
        _FireRedVAD,
        _SileroVADSegments,
        _TimingGTLabels,
        _clip_intervals,
        _make_uid,
        _timing_frame_metrics,
    )
    from evaluate.eval_io import load_audio

    result_csv = args.eval_output_dir / "tse_eval_results.csv"
    rows = _load_rows(result_csv)
    predictions = (
        load_prediction_jsonl(args.predictions_jsonl)
        if args.predictions_jsonl is not None
        else None
    )
    timing_gt = _TimingGTLabels(
        str(args.timing_gt_json_base_dir),
        collar=args.timing_collar,
        match_tolerance=args.timing_match_tolerance,
    )
    vad = None
    if predictions is None:
        if args.timing_vad_backend == "firered":
            root = args.fireredasr2s_root or (_ROOT / "third_party_scripts" / "FireRedASR2S")
            try:
                vad = _FireRedVAD(root, use_gpu=False)
            except Exception as exc:
                if not args.allow_vad_fallback:
                    raise
                # 降级会改变 timing 指标的语义，必须留下显式提示而非静默切换
                print(
                    f"[WARN] FireRed VAD 不可用（{type(exc).__name__}: {exc}），"
                    "按 --allow-vad-fallback 降级到 silero",
                    file=sys.stderr,
                )
                try:
                    vad = _SileroVADSegments()
                except Exception as exc2:
                    print(
                        f"[WARN] Silero VAD 不可用（{type(exc2).__name__}: {exc2}），"
                        "降级到 energy VAD；结果与请求的后端语义不一致",
                        file=sys.stderr,
                    )
                    vad = _EnergyVADSegments()
        elif args.timing_vad_backend == "silero":
            vad = _SileroVADSegments()
        else:
            vad = _EnergyVADSegments()

    requested_uids = set(args.uid)
    generated = 0
    for row in rows:
        uid = _make_uid(row)
        if args.dataset is not None and row["dataset"] != args.dataset:
            continue
        if requested_uids and uid not in requested_uids:
            continue
        label = timing_gt.label_for_row(row)
        if label is None:
            continue
        duration, gt_segments = label
        if predictions is not None:
            if uid not in predictions:
                continue
            pred_segments = predictions[uid]
        else:
            wav_path = args.eval_output_dir / "wav" / f"{uid}.wav"
            waveform, sample_rate = load_audio(str(wav_path))
            pred_segments = vad.segments(waveform, sample_rate)
        pred_segments = _clip_intervals(pred_segments, 0.0, duration)
        metrics = _timing_frame_metrics(
            pred_segments, gt_segments, duration, args.timing_frame_shift
        )
        plot_timeline(
            uid=uid,
            row=row,
            duration=duration,
            gt_segments=gt_segments,
            pred_segments=pred_segments,
            metrics=metrics,
            output_path=args.output_dir / f"{uid}.png",
        )
        generated += 1
        if args.limit is not None and generated >= args.limit:
            break
    print(f"Generated {generated} timing timeline plot(s) in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
