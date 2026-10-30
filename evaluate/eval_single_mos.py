#!/usr/bin/env python3
"""单音频文件非侵入式 MOS 评估（支持长音频自动分块）。

直接对单个音频文件评估 DNSMOS / NISQA / UTMOS / SCOREQ，无需准备 SCP。
长音频会自动切分为等长片段分别推理，最后对片段分数取平均。

Usage:
    # 评估单个文件（默认全部非侵入指标，自动分块 10s）
    python evaluate/eval_single_mos.py --audio /path/to/audio.wav

    # 只跑 DNSMOS + UTMOS，分块长度 20s
    python evaluate/eval_single_mos.py --audio /path/to/audio.wav \
        --metrics dnsmos utmos --chunk_sec 20

    # 重叠分块（50% overlap），对时变质量更敏感
    python evaluate/eval_single_mos.py --audio /path/to/audio.wav \
        --chunk_sec 10 --overlap_sec 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("eval_single_mos")

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.metrics import (
    _resample_np,
    load_dnsmos,
    compute_dnsmos,
    load_utmos,
    compute_utmos,
    load_nisqa,
    compute_nisqa,
    load_scoreq,
    compute_scoreq,
)

# ── 常量 ──
TARGET_SR = 16000  # DNSMOS / UTMOS / SCOREQ 统一用 16kHz
MIN_CHUNK_RATIO = 0.5  # 最后一段若短于 chunk_sec 的 50% 则丢弃


def parse_args():
    p = argparse.ArgumentParser(
        description="Single-file non-intrusive MOS evaluation with chunking support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--audio", required=True, help="Path to input audio file")
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["dnsmos", "nisqa", "utmos", "scoreq"],
        choices=["dnsmos", "nisqa", "utmos", "scoreq"],
        help="Which MOS metrics to compute (default: all)",
    )
    p.add_argument(
        "--chunk_sec",
        type=float,
        default=10.0,
        help="Chunk duration in seconds (default: 10). "
             "If audio <= this length, no chunking is performed.",
    )
    p.add_argument(
        "--overlap_sec",
        type=float,
        default=0.0,
        help="Overlap between consecutive chunks in seconds (default: 0, no overlap). "
             "Example: --chunk_sec 10 --overlap_sec 5 gives 50%% overlap.",
    )
    p.add_argument(
        "--output_dir",
        default="results/single_mos",
        help="Directory to write result JSON",
    )
    p.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for UTMOS (default: auto)",
    )
    args = p.parse_args()
    if not math.isfinite(args.chunk_sec) or args.chunk_sec <= 0:
        p.error("--chunk_sec must be a positive finite number")
    if not math.isfinite(args.overlap_sec) or args.overlap_sec < 0:
        p.error("--overlap_sec must be a non-negative finite number")
    # overlap >= chunk 时 hop_samples <= 0，分块循环不会前进。
    if args.overlap_sec >= args.chunk_sec:
        p.error(
            f"--overlap_sec ({args.overlap_sec}) must be smaller than "
            f"--chunk_sec ({args.chunk_sec}); otherwise hop_samples <= 0 "
            f"and chunking would loop forever."
        )
    return args


def load_audio_mono(path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load audio and resample to target_sr, mono float32."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = _resample_np(data, sr, target_sr)
        sr = target_sr
    return data, sr


def make_chunks(
    audio: np.ndarray, sr: int, chunk_sec: float, overlap_sec: float = 0.0
) -> list[np.ndarray]:
    """Split audio into chunks. Last chunk dropped if shorter than MIN_CHUNK_RATIO."""
    if not isinstance(sr, int) or isinstance(sr, bool) or sr <= 0:
        raise ValueError(f"sr must be a positive integer, got {sr!r}")
    if not math.isfinite(chunk_sec) or chunk_sec <= 0:
        raise ValueError("chunk_sec must be a positive finite number")
    if not math.isfinite(overlap_sec) or overlap_sec < 0 or overlap_sec >= chunk_sec:
        raise ValueError("overlap_sec must be finite, non-negative, and smaller than chunk_sec")
    total_len = len(audio)
    chunk_samples = int(chunk_sec * sr)
    hop_samples = int((chunk_sec - overlap_sec) * sr)
    if chunk_samples <= 0 or hop_samples <= 0:
        raise ValueError("chunk duration and hop must contain at least one sample")

    if total_len <= chunk_samples:
        return [audio]

    chunks = []
    start = 0
    while start < total_len:
        end = start + chunk_samples
        segment = audio[start:end]
        if len(segment) < chunk_samples * MIN_CHUNK_RATIO:
            # 最后一段太短，丢弃
            break
        if len(segment) < chunk_samples:
            # 用末尾反射填充，避免引入静音
            segment = np.pad(
                segment,
                (0, chunk_samples - len(segment)),
                mode="reflect",
            )
        chunks.append(segment)
        start += hop_samples

    if not chunks:
        # 极端情况：音频长度刚好卡在边界，至少保留一个 chunk
        chunks = [audio[:chunk_samples]]

    return chunks


def _score_dnsmos(model, audio: np.ndarray, sr: int) -> Optional[dict]:
    try:
        r = compute_dnsmos(model, audio, sr)
        return {"OVRL": r.get("OVRL"), "SIG": r.get("SIG"), "BAK": r.get("BAK")} if r else None
    except Exception as e:
        logger.warning(f"DNSMOS chunk failed: {e}")
        return None


def _score_utmos(model, audio: np.ndarray, sr: int) -> Optional[float]:
    try:
        r = compute_utmos(model, audio, sr)
        return r.get("MOS") if r else None
    except Exception as e:
        logger.warning(f"UTMOS chunk failed: {e}")
        return None


def _score_nisqa(model, audio: np.ndarray, sr: int) -> Optional[dict]:
    try:
        r = compute_nisqa(model, audio, sr)
        return {
            "MOS": r.get("MOS"),
            "NOI": r.get("NOI"),
            "COL": r.get("COL"),
            "DIS": r.get("DIS"),
            "LOUD": r.get("LOUD"),
        } if r else None
    except Exception as e:
        logger.warning(f"NISQA chunk failed: {e}")
        return None


def _score_scoreq(model, audio: np.ndarray, sr: int) -> Optional[float]:
    try:
        r = compute_scoreq(model, audio, sr)
        return r.get("MOS") if r else None
    except Exception as e:
        logger.warning(f"SCOREQ chunk failed: {e}")
        return None


def main():
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load audio ──
    logger.info(f"Loading: {audio_path}")
    audio, sr = load_audio_mono(str(audio_path), TARGET_SR)
    duration = len(audio) / sr
    logger.info(f"Duration: {duration:.2f}s @ {sr}Hz")

    # ── Chunking ──
    chunks = make_chunks(audio, sr, args.chunk_sec, args.overlap_sec)
    if len(chunks) > 1:
        logger.info(
            f"Chunking: {len(chunks)} chunks (chunk={args.chunk_sec}s, "
            f"overlap={args.overlap_sec}s)"
        )
    else:
        logger.info("No chunking needed (audio shorter than chunk_sec)")

    # ── Load models ──
    metrics = [m.lower() for m in args.metrics]
    models = {}
    if "dnsmos" in metrics:
        logger.info("Loading DNSMOS...")
        models["dnsmos"] = load_dnsmos()
    if "utmos" in metrics:
        logger.info(f"Loading UTMOS (device={args.device})...")
        models["utmos"] = load_utmos(device=args.device)
    if "nisqa" in metrics:
        logger.info("Loading NISQA...")
        models["nisqa"] = load_nisqa()
    if "scoreq" in metrics:
        logger.info("Loading SCOREQ...")
        models["scoreq"] = load_scoreq()

    # ── Evaluate per chunk ──
    records = []
    for idx, chunk in enumerate(chunks, start=1):
        rec: dict[str, float | dict | None] = {"chunk_id": idx, "len_samples": len(chunk)}
        if "dnsmos" in models:
            rec["DNSMOS"] = _score_dnsmos(models["dnsmos"], chunk, sr)
        if "utmos" in models:
            rec["UTMOS"] = _score_utmos(models["utmos"], chunk, sr)
        if "nisqa" in models:
            rec["NISQA"] = _score_nisqa(models["nisqa"], chunk, sr)
        if "scoreq" in models:
            rec["SCOREQ"] = _score_scoreq(models["scoreq"], chunk, sr)
        records.append(rec)

    # ── Aggregate ──
    def _collect(key: str, sub_key: Optional[str] = None):
        vals = []
        for r in records:
            v = r.get(key)
            if v is None:
                continue
            if sub_key is not None:
                v = v.get(sub_key)
            if v is not None and np.isfinite(v):
                vals.append(v)
        return vals

    summary = {}
    for m in ["UTMOS", "SCOREQ"]:
        if m.lower() in metrics:
            vals = _collect(m)
            if vals:
                summary[m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "n_chunks": len(vals),
                }

    if "dnsmos" in metrics:
        for sub in ["OVRL", "SIG", "BAK"]:
            vals = _collect("DNSMOS", sub)
            if vals:
                summary[f"DNSMOS_{sub}"] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "n_chunks": len(vals),
                }

    if "nisqa" in metrics:
        for sub in ["MOS", "NOI", "COL", "DIS", "LOUD"]:
            vals = _collect("NISQA", sub)
            if vals:
                summary[f"NISQA_{sub}"] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "n_chunks": len(vals),
                }

    # ── Print ──
    print("\n" + "=" * 60)
    print(f"  Non-intrusive MOS Results")
    print(f"  File: {audio_path.name}")
    print(f"  Duration: {duration:.2f}s | Chunks: {len(chunks)}")
    print("=" * 60)
    print(f"  {'Metric':20s} | {'Mean':>8s} | {'Std':>8s} | {'N':>4s}")
    print("-" * 60)
    for k, v in summary.items():
        print(f"  {k:20s} | {v['mean']:8.4f} | {v['std']:8.4f} | {v['n_chunks']:4d}")
    print("=" * 60)

    # ── Save JSON ──
    output = {
        "file": str(audio_path),
        "duration_sec": round(duration, 3),
        "sample_rate": sr,
        "chunk_sec": args.chunk_sec,
        "overlap_sec": args.overlap_sec,
        "n_chunks": len(chunks),
        "summary": summary,
        "per_chunk": records,
    }
    out_json = outdir / f"{audio_path.stem}_mos.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Results saved: {out_json}")


if __name__ == "__main__":
    main()
