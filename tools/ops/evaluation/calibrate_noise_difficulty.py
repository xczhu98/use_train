# -*- coding: utf-8 -*-
"""用参考模型实测各类噪声的抑制难度，校准 ``DEFAULT_DIFFICULTY``。

原理
----
难度 = 模型对该类噪声的抑制能力。用 best DeepVQE（或任一 AEC/SE 模型）在
固定 SNR 的加性噪声场景（mic = clean + noise，farend=0）测 **SI-SDR 改善
ΔSI-SDR = SI-SDR(clean, enh) - SI-SDR(clean, mic)**：

- 抑制增益大 → 模型处理得好 → 易；
- 抑制增益小 → 模型处理不好 → 难（difficulty ∝ 1/Δ）。

最终把桶级 mean Δ 线性映射回 [1,5] 作为"建议难度"，与先验
``core/data/sampling.py::DEFAULT_DIFFICULTY`` 对照输出 JSON；**不自动覆盖**
常量，由人审校后决定是否更新。

用法::

    python -m tools.ops.evaluation.calibrate_noise_difficulty \\
        --config     exp/aec/.../resolved_config.yaml \\
        --checkpoint exp/aec/.../best_model.tar \\
        --mode representative   # 每 (统计×时间) 桶取 top-UID 代表标签，快速出趋势
        --mode all              # 全部标签（较久，建议后台）
        --out exp/eval/noise_difficulty_calib.json
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# 支持直接运行本模块时导入仓库内的 core/tasks
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

logger = logging.getLogger(__name__)

CLEAN_SCP = "data/shared/clean/filter/val_16k.scp"
NOISE_JSONL = "data/shared/noise/filter/train_noise_sources.jsonl"
SR = 16000
CHUNK_SEC = 4.0
TOP_LABELS_PER_BUCKET = 2


def _read(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    return sf.read(path, dtype="float32")


def _resample(x: np.ndarray, fs_in: int, fs_out: int = SR) -> np.ndarray:
    if fs_in == fs_out:
        return x
    n = int(round(len(x) * fs_out / fs_in))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def _tile(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.tile(x, (n // len(x)) + 1)[:n].astype(np.float32)


def _mix(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    speech_rms = float(np.sqrt(np.mean(clean ** 2)) + 1e-12)
    noise_rms = float(np.sqrt(np.mean(noise ** 2)) + 1e-12)
    scale = (speech_rms / noise_rms) * 10.0 ** (-snr_db / 20.0)
    return (clean + scale * noise).astype(np.float32)


def _load_clean(n: int) -> list[np.ndarray]:
    paths = []
    with open(CLEAN_SCP) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 4:
                paths.append(parts[3])
            if len(paths) >= n:
                break
    if len(paths) < 3:
        raise SystemExit(f"clean pool 不足: {CLEAN_SCP}")
    clips = []
    for p in paths:
        wav, fs = _read(p)
        clips.append(_tile(_resample(wav, fs), int(CHUNK_SEC * SR)))
    return clips


def _load_noise_by_label() -> dict[str, list[str]]:
    by_label: dict[str, list[str]] = collections.defaultdict(list)
    with open(NOISE_JSONL) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            by_label[m["category_name"]].append(m["path"])
    return {k: sorted(v) for k, v in by_label.items()}


def _selection(by_label, difficulty_table, mode: str) -> list[str]:
    if mode == "all":
        return sorted(by_label)
    from core.data.noise_attributes import NOISE_ATTRIBUTES
    label_uid = {label: len(paths) for label, paths in by_label.items()}
    bucket_labels: dict[str, list[str]] = collections.defaultdict(list)
    for label in by_label:
        s, t = NOISE_ATTRIBUTES[label][1], NOISE_ATTRIBUTES[label][2]
        key = f"{s}:{t}"
        if key in difficulty_table:
            bucket_labels[key].append(label)
    picked = []
    for key in sorted(bucket_labels):
        picked.extend(
            sorted(bucket_labels[key], key=lambda x: -label_uid[x])[:TOP_LABELS_PER_BUCKET]
        )
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["representative", "all"], default="representative")
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--n-speech", type=int, default=3)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="exp/eval/noise_difficulty_calib.json")
    args = parser.parse_args()

    from core.data.noise_attributes import NOISE_ATTRIBUTES
    from core.data.sampling import DEFAULT_DIFFICULTY
    import soundfile as sf

    by_label = _load_noise_by_label()
    selection = _selection(by_label, DEFAULT_DIFFICULTY, args.mode)
    clean_clips = _load_clean(args.n_speech)
    logger.info("校准标签 %d 个（mode=%s）", len(selection), args.mode)

    # plan: seq -> (label, clean_idx, noise_path)
    plan: list[tuple[int, str, int, str]] = []
    with tempfile.TemporaryDirectory(prefix="diffcal_mix_") as td_in, tempfile.TemporaryDirectory(
        prefix="diffcal_out_"
    ) as td_out:
        mix_dir = Path(td_in)
        out_dir = Path(td_out)
        seq = 0
        for label in selection:
            paths = by_label.get(label, [])
            if not paths:
                continue
            for ci in range(args.per_class):
                noise_path = paths[ci % len(paths)]
                uid = f"n{seq:05d}"
                wav, fs = _read(noise_path)
                noise16 = _tile(_resample(wav, fs), int(CHUNK_SEC * SR))
                clean = clean_clips[seq % len(clean_clips)]
                sf.write(str(mix_dir / f"{uid}.wav"), _mix(clean, noise16, args.snr), SR)
                plan.append((seq, label, seq % len(clean_clips), noise_path))
                seq += 1

        # purpose: noise difficulty calibration —— checkpoint 来自本仓库已
        # 训练实验（trusted），allow_unsafe 仅用于读取模型权重反序列化。
        from evaluate.eval_pipeline_common import run_aec_inference_zero_farend
        run_aec_inference_zero_farend(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            input_dir=str(mix_dir),
            output_dir=str(out_dir),
            device=args.device,
            allow_partial=True,
        )

        per_label: dict[str, list[float]] = collections.defaultdict(list)
        per_pair_delta: dict[tuple[str, int], float] = {}
        for seq_id, label, clean_idx, _noise_path in plan:
            uid = f"n{seq_id:05d}"
            enh_path = out_dir / f"{uid}.wav"
            mix_path = mix_dir / f"{uid}.wav"
            if not enh_path.exists() or not mix_path.exists():
                logger.warning("缺失 %s，跳过", uid)
                continue
            enh, _ = _read(str(enh_path))
            mix, _ = _read(str(mix_path))
            clean = clean_clips[clean_idx]
            n = min(len(clean), len(mix), len(enh))
            baseline = _sisdr(clean[:n], mix[:n])
            improved = _sisdr(clean[:n], enh[:n])
            per_label[label].append(improved - baseline)
            per_pair_delta[(label, seq_id)] = improved - baseline

    # 桶级聚合
    bucket_delta: dict[str, list[float]] = collections.defaultdict(list)
    for label, deltas in per_label.items():
        s, t = NOISE_ATTRIBUTES[label][1], NOISE_ATTRIBUTES[label][2]
        bucket_delta[f"{s}:{t}"].extend(deltas)
    bucket_mean = {k: float(np.mean(v)) for k, v in bucket_delta.items()}
    if bucket_mean:
        lo, hi = min(bucket_mean.values()), max(bucket_mean.values())
        rng = (hi - lo) or 1.0
        suggested = {k: 1.0 + 4.0 * (hi - v) / rng for k, v in bucket_mean.items()}
    else:
        suggested = {}

    result = {
        "mode": args.mode,
        "snr_db": args.snr,
        "checkpoint": args.checkpoint,
        "n_labels": len(selection),
        "per_bucket_delta": {k: round(v, 3) for k, v in bucket_mean.items()},
        "suggested_difficulty": {k: round(v, 2) for k, v in suggested.items()},
        "prior_difficulty": {k: DEFAULT_DIFFICULTY[k] for k in bucket_mean if k in DEFAULT_DIFFICULTY},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("结果已写 %s", args.out)

    print("\n=== 桶级 ΔSI-SDR 与建议难度（对照先验） ===")
    print(f"{'桶':34s}{'ΔSDR':>7s}{'n':>4s}{'建议':>6s}{'先验':>6s}")
    for k in sorted(bucket_mean, key=lambda x: -bucket_mean[x]):
        print(
            f"{k:34s}{bucket_mean[k]:7.2f}{len(bucket_delta[k]):4d}"
            f"{suggested[k]:6.1f}{DEFAULT_DIFFICULTY.get(k, float('nan')):6.1f}"
        )
    print("\n=== 每标签均值 ΔSI-SDR（最易 8 / 最难 8） ===")
    label_mean = {k: float(np.mean(v)) for k, v in per_label.items()}
    ordered = sorted(label_mean.items(), key=lambda x: -x[1])
    for k, v in ordered[:8]:
        print(f"  易  {k:42s} {v:7.2f}")
    for k, v in ordered[-8:]:
        print(f"  难  {k:42s} {v:7.2f}")
    return 0


def _sisdr(ref: np.ndarray, est: np.ndarray) -> float:
    from core.metrics.intrusive import compute_sisdr_np
    return compute_sisdr_np(ref, est)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
