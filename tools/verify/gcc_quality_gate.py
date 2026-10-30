#!/usr/bin/env python3
"""GCC 延迟估计质量门工具（JAE-M09）。

审计发现 ``tasks/aec/inference/gcc_phat_core.py::estimate_gcc_tau_windowed`` 在低置信时
tau=0 保留样本，但全仓库缺少 delay MAE / false-lock / low-confidence 比例的可审计
统计。本工具只做"质量门报告"，不改变任何训练数据语义。

两种模式：

- ``--synthetic``（默认，零数据依赖）：确定性合成语音源（宽带调制思路参考
  ``tests/test_offline_alignment.py`` 的 ``_speech_like``，本工具另加多频正弦叠加、
  零填充线性延迟与 10x 网格分数延迟），对每 (source, tau, degrade) 组合跑估计器，
  统计仅高置信样本的 delay 误差
  MAE/P50/P95（ms）、false-lock 率（高置信但 |误差| > 5ms）、low_confidence 率。
- ``--pairs-json <path>``：扫真实 far/echo manifest（schema 与训练侧
  ``real_farend_echo_pair_json`` 完全一致，复用
  ``tasks.aec.data.dynamic_dataset.parse_real_farend_echo_pair_file`` 解析：
  ``[far, echo]`` / 单路径 stereo 行 / dict；stereo 行 echo=ch0、farend=ch1，
  与 ``DynamicMixingDataset`` 通道约定一致），聚合置信分布并把低置信 pair
  按 mode 分桶写入 ``<out>/low_confidence_pairs.txt``（供后续降权/重采样决策，
  本任务不做重采样）。manifest 相对路径按 manifest parent 解析；不可读 pair
  跳过并计数（``skipped_unreadable``，清单写
  ``<out>/skipped_unreadable.txt``），默认 ``--max-skipped-unreadable 0``，因此任何
  跳过都会使 gate 失败；只有显式提高上限才接受已审计的跳过。大 manifest 可用
  ``--max-pairs N --sample-seed S`` 确定性均匀
  抽样（无放回、按索引排序，同参重跑逐位一致），aggregate 的
  ``manifest_pair_count`` 记录全量对数以区分抽样与全量。

oracle 口径：合成模式以施加的真值延迟为 oracle；真实对无 oracle，只报置信分布。

用户决策（2026-07-15）：真实 pairs 确定性抽样（5000/88962，sample_seed=0）
实测 low_confidence 占比 3.8%、time_varying 0.08%、skipped 0，推算全量约
3400 对。判定为小占比，**处置决策=不动**——不降权、不重采样、不剔除，
训练数据语义维持现状；本工具仅作可审计报告与后续复测能力保留。

Usage::

    cd use_train
    python tools/verify/gcc_quality_gate.py --synthetic \
        --taus-ms 0,50,120,200,299 --degrades clean,reverb,noise,clip \
        --out gcc_quality_gate_report/

    python tools/verify/gcc_quality_gate.py \
        --pairs-json path/to/pairs.json --out gcc_quality_gate_report/

    # 任一阈值超标、空集、非有限指标或默认不可读跳过均 exit 1
    python tools/verify/gcc_quality_gate.py \
        --max-mae-ms 5 --max-false-lock-rate 0.01 --max-low-conf-rate 0.3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tasks.aec.data.dynamic_dataset import (  # noqa: E402
    parse_real_farend_echo_pair_file,
)
from tasks.aec.inference.gcc_phat_core import estimate_gcc_tau_windowed  # noqa: E402

# 高置信但 |误差| 超过该值（ms）记为 false-lock
FALSE_LOCK_TOLERANCE_MS = 5.0

# oracle 口径说明，写入 report.json 的 note 字段
ORACLE_NOTE = (
    "oracle 口径：synthetic 模式以施加的真值延迟为 oracle，误差统计（MAE/P50/P95、"
    "false-lock）仅覆盖高置信样本；pairs-json 真实对无 oracle，只报置信分布"
    "（low_confidence/time_varying 比例与 peak/prominence/tau_ms 分位数），不做误差判定。"
)

_DEFAULT_TAUS_MS = "0,50,120,200,299"
_DEFAULT_DEGRADES = "clean,reverb,noise,clip"


def _git_commit() -> str:
    """记录谱系用；失败时返回 unknown（与 canonical AEC testset 生成器同口径）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_ROOT,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------
# 合成信号（宽带调制思路参考 tests/test_offline_alignment.py::_speech_like）
# ---------------------------------------------------------------------------
def _speech_like(n: int, sr: int, seed: int = 0) -> np.ndarray:
    """类语音信号：多频正弦叠加 × 慢变包络（宽带调制，适合 GCC-PHAT）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / float(sr)
    # 频点取互素整数（Hz），避免公共周期落入 GCC 搜索窗（≤200ms）造成虚假锁定：
    # 若各分量在 tau 处都完成整数个周期，自相关会在该 tau 处重现峰值，lag 0 被 argmax 抢先命中
    freqs = np.array([113.0, 217.0, 349.0, 533.0, 797.0, 1217.0, 1811.0, 2609.0])
    phases = rng.uniform(0.0, 2.0 * np.pi, freqs.size)
    x = np.zeros(n, np.float64)
    for f, p in zip(freqs, phases):
        x += np.sin(2.0 * np.pi * f * t + p)
    x /= freqs.size
    # 掺入低通噪声分量（8 点滑动平均，同 test_aec_fine_delay_align.py 的 _speech_like）：
    # 纯正弦叠加的自相关是余弦和、衰减极慢，超出搜索窗的 tau 会在窗内局部峰虚假锁定；
    # 宽带噪声分量让自相关快速衰减，出窗 tau 才会正确落入 low_confidence
    x = 0.7 * x + 0.3 * np.convolve(rng.standard_normal(n), np.ones(8) / 8.0, mode="same")
    # 慢变包络：低通噪声幅值，模拟语音的起伏调制
    env = 0.3 + 0.7 * np.abs(
        np.convolve(rng.standard_normal(n), np.ones(400) / 400.0, mode="same")
    )
    x = x * env
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.9).astype(np.float32)


def _fractional_delay(x: np.ndarray, delay_ms: float, sr: int, grid: int = 10) -> np.ndarray:
    """10x 网格分数延迟：升采样 → 整数样点零填充移位 → 降采样（非环绕移位）。"""
    fs_hi = int(sr) * grid
    d_hi = int(round(delay_ms / 1000.0 * fs_hi))
    x_hi = resample_poly(x, grid, 1)
    y_hi = np.concatenate([np.zeros(d_hi, np.float32), x_hi])
    return resample_poly(y_hi, 1, grid)[: x.size].astype(np.float32)


def _apply_degrade(x: np.ndarray, degrade: str, sr: int, seed: int) -> np.ndarray:
    """对 echo 施加退化：clean / reverb / noise(SNR=20dB) / clip(峰值 0.7 硬截幅)。"""
    if degrade == "clean":
        return x
    if degrade == "reverb":
        # 简单混响核：直达 1.0 + 200ms 指数衰减噪声尾（T60≈200ms）
        rng = np.random.default_rng(seed)
        n_ir = int(0.2 * sr)
        ir = rng.standard_normal(n_ir) * np.exp(-6.91 * np.arange(n_ir) / n_ir) * 0.1
        ir[: int(0.005 * sr)] = 0.0  # 前 5ms 只有直达声
        ir[0] = 1.0
        y = np.convolve(x, ir.astype(np.float32))[: x.size]
        return (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
    if degrade == "noise":
        rng = np.random.default_rng(seed)
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        noise_rms = rms * 10.0 ** (-20.0 / 20.0)  # SNR=20dB 加性白噪声
        return (x + noise_rms * rng.standard_normal(x.size)).astype(np.float32)
    if degrade == "clip":
        return np.clip(x, -0.7, 0.7).astype(np.float32)
    raise ValueError(f"未知 degrade 类型: {degrade!r}（可选 clean/reverb/noise/clip）")


# ---------------------------------------------------------------------------
# 估计与聚合
# ---------------------------------------------------------------------------
def run_estimate(far: np.ndarray, echo: np.ndarray, sr: int) -> dict[str, Any]:
    """对单对信号跑估计器；输入或结果非有限时 fail-closed。"""
    far_arr = np.ascontiguousarray(far, dtype=np.float32)
    echo_arr = np.ascontiguousarray(echo, dtype=np.float32)
    if int(sr) <= 0:
        raise ValueError(f"sample rate must be positive, got {sr!r}")
    if not np.isfinite(far_arr).all() or not np.isfinite(echo_arr).all():
        raise ValueError("GCC input contains NaN or Inf")
    est = estimate_gcc_tau_windowed(far_arr, echo_arr, int(sr))
    numeric = np.asarray(
        [float(est.tau_sec), float(est.peak), float(est.prominence)],
        dtype=np.float64,
    )
    if not np.isfinite(numeric).all():
        raise RuntimeError("GCC estimate contains NaN or Inf")
    if numeric[1] < 0.0 or numeric[2] < 0.0:
        raise RuntimeError("GCC quality values must be non-negative")
    return {
        "tau_ms": float(numeric[0]) * 1000.0,
        "low_confidence": bool(est.low_confidence),
        "time_varying": bool(est.time_varying),
        "mode": str(est.mode),
        "peak": float(numeric[1]),
        "prominence": float(numeric[2]),
    }


def _percentiles(values: list[float]) -> dict[str, Optional[float]]:
    """P5/P50/P95 分位数；空列表返回 None，非有限值立即失败。"""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"p5": None, "p50": None, "p95": None}
    if not np.isfinite(arr).all():
        raise ValueError("percentile input contains NaN or Inf")
    return {
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合逐条估计记录。

    误差统计（mae/p50/p95、false-lock）仅覆盖高置信且带 oracle 误差
    （``error_ms`` 非 None）的记录；无 oracle 的记录只计入置信比例。
    ``false_lock_rate`` 分母为高置信样本数（len(high)），与 MAE/P50/P95
    同口径，避免低置信占比高时稀释 false-lock 指标。
    """
    total = len(records)
    for index, record in enumerate(records):
        for key in ("tau_ms", "peak", "prominence"):
            if key in record and not np.isfinite(float(record[key])):
                raise ValueError(f"record {index} has non-finite {key}")
        if record.get("error_ms") is not None and not np.isfinite(
            float(record["error_ms"])
        ):
            raise ValueError(f"record {index} has non-finite error_ms")
    low = [r for r in records if r["low_confidence"]]
    high = [r for r in records if not r["low_confidence"]]
    errs = [abs(float(r["error_ms"])) for r in high if r.get("error_ms") is not None]
    false_locks = [e for e in errs if e > FALSE_LOCK_TOLERANCE_MS]
    tv = [r for r in records if r["time_varying"]]
    return {
        "total": total,
        "high_confidence_count": len(high),
        "low_confidence_count": len(low),
        "low_confidence_rate": (len(low) / total) if total else 0.0,
        "time_varying_count": len(tv),
        "time_varying_rate": (len(tv) / total) if total else 0.0,
        "false_lock_count": len(false_locks),
        # 分母取高置信样本数：与"高置信但 |误差|>5ms 的比例"定义一致，
        # 且与同行打印的 MAE/P50/P95（仅高置信）口径统一；无高置信样本时记 0.0
        # （此时若设了 false-lock 阈值，由 _gate_violations 报"无高置信样本"违规兜底）
        "false_lock_rate": (len(false_locks) / len(high)) if high else 0.0,
        "mae_ms": float(np.mean(errs)) if errs else None,
        "p50_ms": float(np.percentile(errs, 50)) if errs else None,
        "p95_ms": float(np.percentile(errs, 95)) if errs else None,
    }


# ---------------------------------------------------------------------------
# 模式一：synthetic（确定性合成，带 oracle）
# ---------------------------------------------------------------------------
def run_synthetic(
    taus_ms: list[float],
    degrades: list[str],
    num_sources: int,
    seed: int,
    sr: int,
    duration_sec: float,
) -> dict[str, Any]:
    """对每 (source, tau, degrade) 组合跑估计器，返回逐组合明细与聚合。"""
    n = int(round(duration_sec * sr))
    records: list[dict[str, Any]] = []
    for si in range(num_sources):
        far = _speech_like(n, sr, seed=seed * 1000 + si)
        for tau in taus_ms:
            echo_delayed = _fractional_delay(far, float(tau), sr)
            for dg in degrades:
                echo = _apply_degrade(echo_delayed, dg, sr, seed=seed * 1000 + 100 + si)
                rec: dict[str, Any] = {
                    "source_idx": si,
                    "tau_ms_true": float(tau),
                    "degrade": dg,
                }
                rec.update(run_estimate(far, echo, sr))
                # 低置信时估计器 tau=0（zero_pad 语义），不计入误差统计
                rec["error_ms"] = (
                    None if rec["low_confidence"] else rec["tau_ms"] - float(tau)
                )
                records.append(rec)
    return {"per_combination": records, "aggregate": aggregate_records(records)}


# ---------------------------------------------------------------------------
# 模式二：pairs-json（真实对，无 oracle）
# ---------------------------------------------------------------------------
def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    """读 wav 为 float32 单声道，返回 (信号, 采样率)。"""
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def _read_pair_signals(far_p: Path, echo_p: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """读一对 far/echo 信号。

    ``far_p == echo_p`` 为 stereo 单文件行（训练 manifest 的单路径 schema）：
    通道约定与 ``DynamicMixingDataset`` 一致——echo=ch0、farend=ch1；
    文件实为单声道时退化为两路同信号（tau=0 高置信，属预期）。
    """
    if far_p == echo_p:
        x, sr = sf.read(str(far_p), dtype="float32")
        if x.ndim > 1 and x.shape[1] >= 2:
            echo = np.ascontiguousarray(x[:, 0], dtype=np.float32)
            far = np.ascontiguousarray(x[:, 1], dtype=np.float32)
        else:
            mono = x if x.ndim == 1 else x.mean(axis=1)
            far = echo = np.ascontiguousarray(mono, dtype=np.float32)
        return far, echo, int(sr)
    wf, sr_f = _read_mono(far_p)
    we, sr_e = _read_mono(echo_p)
    if sr_f != sr_e:
        raise ValueError(f"采样率不一致（{sr_f} vs {sr_e}）: {far_p} / {echo_p}")
    return wf, we, sr_f


def run_pairs_json(
    manifest_path: Path,
    out_dir: Path,
    max_pairs: Optional[int] = None,
    sample_seed: int = 0,
) -> dict[str, Any]:
    """逐对跑估计器，聚合置信分布，并把低置信 pair 按 mode 分桶写清单。

    ``max_pairs`` 给定且 manifest 超过该数量时，按 ``sample_seed`` 确定性
    均匀抽样（无放回、按索引排序），同参重跑逐位一致；aggregate 中
    ``manifest_pair_count`` 记录 manifest 全量对数，便于区分抽样与全量。
    """
    # 复用训练同款 canonical stereo 单路径解析。
    manifest_path = manifest_path.expanduser().resolve(strict=False)
    raw_pairs = parse_real_farend_echo_pair_file(manifest_path)
    base = manifest_path.parent
    manifest_pair_count = len(raw_pairs)
    if max_pairs is not None and len(raw_pairs) > max_pairs:
        idx = np.random.default_rng(sample_seed).choice(
            len(raw_pairs), size=int(max_pairs), replace=False
        )
        raw_pairs = [raw_pairs[int(i)] for i in sorted(int(i) for i in idx)]

    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for far_s, echo_s in raw_pairs:
        far_p, echo_p = Path(far_s).expanduser(), Path(echo_s).expanduser()
        if not far_p.is_absolute():
            far_p = base / far_p
        if not echo_p.is_absolute():
            echo_p = base / echo_p
        far_p = far_p.resolve(strict=False)
        echo_p = echo_p.resolve(strict=False)
        try:
            wf, we, sr = _read_pair_signals(far_p, echo_p)
            rec = run_estimate(wf, we, sr)
        except Exception as exc:
            # 跳过必须计数并由默认 gate 判失败；清单保留具体异常类型和消息。
            skipped.append(
                f"{far_p}\t{echo_p}\t{type(exc).__name__}: {exc}"
            )
            continue
        rec["far"] = str(far_p)
        rec["echo"] = str(echo_p)
        records.append(rec)

    aggregate = aggregate_records(records)
    # 真实对无 oracle，只补置信分布分位数
    aggregate["peak"] = _percentiles([r["peak"] for r in records])
    aggregate["prominence"] = _percentiles([r["prominence"] for r in records])
    aggregate["tau_ms"] = _percentiles([r["tau_ms"] for r in records])
    aggregate["manifest_pair_count"] = manifest_pair_count

    # 低置信 pair 按 mode 分桶写清单（供后续降权/重采样决策，本任务不做重采样）
    low = [r for r in records if r["low_confidence"]]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in low:
        buckets.setdefault(r["mode"], []).append(r)
    lines = ["# GCC 低置信 pair 清单（按估计器 mode 分桶）"]
    for mode in sorted(buckets):
        lines.append(f"[mode={mode}] count={len(buckets[mode])}")
        lines.append("far\techo\tpeak\tprominence\ttau_ms")
        for r in buckets[mode]:
            lines.append(
                f"{r['far']}\t{r['echo']}\t{r['peak']:.6f}\t{r['prominence']:.3f}\t{r['tau_ms']:.3f}"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "low_confidence_pairs.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    aggregate["skipped_unreadable"] = len(skipped)
    if skipped:
        (out_dir / "skipped_unreadable.txt").write_text(
            "# 不可读/解析失败被跳过的 pair（far\techo\terror）\n"
            + "\n".join(skipped)
            + "\n",
            encoding="utf-8",
        )
    return {"pairs_detail": records, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# 阈值门与汇总输出
# ---------------------------------------------------------------------------
def _gate_violations(
    aggregate: dict[str, Any],
    *,
    max_mae_ms: Optional[float],
    max_false_lock_rate: Optional[float],
    max_low_conf_rate: Optional[float],
    has_oracle: bool,
    max_skipped_unreadable: Optional[int] = 0,
) -> list[str]:
    """逐阈值检查；空结果、非有限指标和超量跳过均 fail-closed。"""
    v: list[str] = []
    total_value = aggregate.get("total", 0)
    try:
        total = float(total_value)
    except (TypeError, ValueError, OverflowError):
        total = float("nan")
    if not np.isfinite(total) or total <= 0.0:
        v.append("total: 没有可评估样本或值非有限")
    for key in ("low_confidence_rate", "time_varying_rate", "false_lock_rate"):
        value = aggregate.get(key)
        try:
            finite = value is not None and np.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            v.append(f"{key}: 指标缺失或非有限")
    for key in ("mae_ms", "p50_ms", "p95_ms"):
        value = aggregate.get(key)
        try:
            finite = value is None or np.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            v.append(f"{key}: 指标非有限")
    skipped_value = aggregate.get("skipped_unreadable", 0)
    try:
        skipped = int(skipped_value)
    except (TypeError, ValueError, OverflowError):
        skipped = 0
        v.append("skipped_unreadable: 指标非有限")
    if max_skipped_unreadable is not None and skipped > max_skipped_unreadable:
        v.append(f"skipped_unreadable {skipped} > {max_skipped_unreadable}")
    if has_oracle and max_mae_ms is not None:
        if aggregate["mae_ms"] is None:
            v.append("mae_ms: 无高置信样本，无法计算 MAE")
        elif aggregate["mae_ms"] > max_mae_ms:
            v.append(f"mae_ms {aggregate['mae_ms']:.3f} > {max_mae_ms}")
    if has_oracle and max_false_lock_rate is not None:
        # 与 mae 的"无高置信样本"分支对齐：errs 为空（mae_ms is None）时
        # false_lock_rate 恒为 0.0，若直接比较会 vacuous pass，须同样报违规
        if aggregate["mae_ms"] is None:
            v.append("false_lock_rate: 无高置信样本，无法计算 false-lock 率")
        elif aggregate["false_lock_rate"] > max_false_lock_rate:
            v.append(
                f"false_lock_rate {aggregate['false_lock_rate']:.4f} > {max_false_lock_rate}"
            )
    if max_low_conf_rate is not None:
        if aggregate["low_confidence_rate"] > max_low_conf_rate:
            v.append(
                f"low_confidence_rate {aggregate['low_confidence_rate']:.4f} > {max_low_conf_rate}"
            )
    return v


def _print_summary(report: dict[str, Any]) -> None:
    """stdout 汇总表。"""
    agg = report["aggregate"]
    print(f"== GCC 质量门汇总（{report['mode']}）==")
    print(
        f"total={agg['total']}  high_conf={agg['high_confidence_count']}  "
        f"low_conf={agg['low_confidence_count']} ({agg['low_confidence_rate']:.1%})  "
        f"time_varying={agg['time_varying_count']} ({agg['time_varying_rate']:.1%})"
    )
    if report["mode"] == "synthetic":
        mae = "n/a" if agg["mae_ms"] is None else f"{agg['mae_ms']:.3f}"
        p50 = "n/a" if agg["p50_ms"] is None else f"{agg['p50_ms']:.3f}"
        p95 = "n/a" if agg["p95_ms"] is None else f"{agg['p95_ms']:.3f}"
        print(
            f"delay 误差（仅高置信, ms）: MAE={mae}  P50={p50}  P95={p95}  "
            f"false_lock={agg['false_lock_count']} ({agg['false_lock_rate']:.1%})"
        )
    else:
        mpc = agg.get("manifest_pair_count")
        if mpc is not None and mpc != agg["total"]:
            print(f"（确定性抽样 {agg['total']} / manifest 共 {mpc} 对，sample_seed 见 report params）")
        if agg.get("skipped_unreadable"):
            print(f"skipped_unreadable={agg['skipped_unreadable']}（清单见 skipped_unreadable.txt）")
        for key in ("peak", "prominence", "tau_ms"):
            q = agg[key]
            vals = "  ".join(
                "n/a" if q[k] is None else f"{k}={q[k]:.4g}" for k in ("p5", "p50", "p95")
            )
            print(f"{key}: {vals}")
    gate = report["gate"]
    if gate["violations"]:
        print("gate: FAIL")
        for msg in gate["violations"]:
            print(f"  - {msg}")
    else:
        print("gate: PASS")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--synthetic", action="store_true", help="合成模式（默认；显式写出仅为可读性）")
    p.add_argument(
        "--pairs-json",
        default=None,
        help="真实 far/echo manifest（{'pairs': [[far, echo], ...]}）；给定后覆盖合成模式",
    )
    p.add_argument("--taus-ms", default=_DEFAULT_TAUS_MS, help="逗号分隔的真值延迟（ms）")
    p.add_argument("--degrades", default=_DEFAULT_DEGRADES, help="逗号分隔的退化类型")
    p.add_argument("--num-sources", type=int, default=4, help="合成源个数")
    p.add_argument("--seed", type=int, default=0, help="合成随机种子")
    p.add_argument("--sr", type=int, default=16000, help="合成采样率")
    p.add_argument("--duration-sec", type=float, default=3.0, help="单源时长（秒）")
    p.add_argument("--out", default="gcc_quality_gate_report/", help="报告输出目录")
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="pairs-json 模式下确定性抽样上限（无放回、按索引排序；默认全量）",
    )
    p.add_argument("--sample-seed", type=int, default=0, help="pairs-json 抽样种子")
    p.add_argument("--max-mae-ms", type=float, default=None, help="MAE 阈值 ms（需 oracle，仅合成模式）")
    p.add_argument(
        "--max-false-lock-rate", type=float, default=None, help="false-lock 率阈值（需 oracle，仅合成模式）"
    )
    p.add_argument("--max-low-conf-rate", type=float, default=None, help="low_confidence 率阈值")
    p.add_argument(
        "--max-skipped-unreadable",
        type=int,
        default=0,
        help="允许跳过的不可读 pair 数；默认 0，任何跳过均失败",
    )
    args = p.parse_args(argv)
    finite_args = {
        "--duration-sec": args.duration_sec,
        "--max-mae-ms": args.max_mae_ms,
        "--max-false-lock-rate": args.max_false_lock_rate,
        "--max-low-conf-rate": args.max_low_conf_rate,
    }
    invalid = [
        name for name, value in finite_args.items()
        if value is not None and not np.isfinite(float(value))
    ]
    if invalid:
        p.error("参数必须为有限数值: " + ", ".join(invalid))
    if args.sr <= 0 or args.duration_sec <= 0.0 or args.num_sources < 0:
        p.error("--sr/--duration-sec 必须为正，--num-sources 必须 >= 0")
    if args.max_pairs is not None and args.max_pairs < 0:
        p.error("--max-pairs must be >= 0")
    if args.max_skipped_unreadable < 0:
        p.error("--max-skipped-unreadable must be >= 0")
    if not args.pairs_json:
        try:
            parsed_taus_ms = [
                float(value) for value in str(args.taus_ms).split(",") if value.strip()
            ]
        except ValueError as error:
            p.error(f"--taus-ms 必须为逗号分隔数值: {error}")
        if not np.isfinite(np.asarray(parsed_taus_ms, dtype=np.float64)).all():
            p.error("--taus-ms 必须全部为有限数值")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_json:
        mode = "pairs_json"
        # 无 oracle 模式下这两个阈值会被 _gate_violations 的 has_oracle=False
        # 静默忽略；用户显式传入时打印 WARNING 说明不生效
        ignored = [
            name
            for name, val in (
                ("--max-mae-ms", args.max_mae_ms),
                ("--max-false-lock-rate", args.max_false_lock_rate),
            )
            if val is not None
        ]
        if ignored:
            print(
                f"WARNING: pairs-json 模式无 oracle，{'/'.join(ignored)} "
                f"阈值不生效（已忽略，仅 --max-low-conf-rate 参与 gate）"
            )
        params: dict[str, Any] = {
            "pairs_json": str(args.pairs_json),
            "max_pairs": args.max_pairs,
            "sample_seed": args.sample_seed,
        }
        result = run_pairs_json(
            Path(args.pairs_json), out_dir,
            max_pairs=args.max_pairs, sample_seed=args.sample_seed,
        )
    else:
        mode = "synthetic"
        taus_ms = parsed_taus_ms
        degrades = [d.strip() for d in str(args.degrades).split(",") if d.strip()]
        params = {
            "taus_ms": taus_ms,
            "degrades": degrades,
            "num_sources": args.num_sources,
            "seed": args.seed,
            "sr": args.sr,
            "duration_sec": args.duration_sec,
        }
        result = run_synthetic(
            taus_ms, degrades, args.num_sources, args.seed, args.sr, args.duration_sec
        )

    thresholds = {
        "max_mae_ms": args.max_mae_ms,
        "max_false_lock_rate": args.max_false_lock_rate,
        "max_low_conf_rate": args.max_low_conf_rate,
        "max_skipped_unreadable": args.max_skipped_unreadable,
    }
    violations = _gate_violations(
        result["aggregate"],
        max_mae_ms=args.max_mae_ms,
        max_false_lock_rate=args.max_false_lock_rate,
        max_low_conf_rate=args.max_low_conf_rate,
        has_oracle=(mode == "synthetic"),
        max_skipped_unreadable=args.max_skipped_unreadable,
    )
    report: dict[str, Any] = {
        "git_commit": _git_commit(),
        "tool": "tools/verify/gcc_quality_gate.py",
        "mode": mode,
        "note": ORACLE_NOTE,
        "params": params,
        **result,
        "gate": {"thresholds": thresholds, "violations": violations, "passed": not violations},
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _print_summary(report)
    print(f"report: {out_dir / 'report.json'}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
