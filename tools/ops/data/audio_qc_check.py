#!/usr/bin/env python
"""音频数据批量质检与处置脚本（截幅 / 电平 / 动态范围 / 静音 / 脏数据）。

两层设计：
  1. 检查（check）：计算指标并打 flag
  2. 处置（action）：按固定决策表把 flag 归类为「可修复」或「不可修复」，
     可修复的执行修复并复检，不可修复的隔离/记录

判定与处置决策表（阈值均可通过命令行调整）：
  | flag                | 判定依据（默认）                  | 处置                          |
  |---------------------|-----------------------------------|-------------------------------|
  | clipping_run        | 连续 >=3 个样本触顶（真削波）     | discard —— 不可逆失真，剔除   |
  | clipping_ratio      | 触顶样本占比 > --clip-ratio-max   | discard —— 剔除               |
  | peak_too_high       | peak > -0.1 dBFS 但未触顶         | fix —— 增益降到 --target-peak |
  | peak_too_low        | peak < -20 dBFS 但 rms 高于硬下限 | fix —— peak 归一化            |
  | rms_too_low         | rms < -50 dBFS                    | 尝试 trim 后复检，仍低则剔除  |
  | rms_hard_floor      | rms < -60 dBFS（几乎无声）        | discard —— 剔除               |
  | too_much_silence    | 静音帧占比 > 0.7                  | fix —— 裁首尾静音后复检       |
  | dc_offset           | |mean| > 0.01                     | fix —— 去直流                 |
  | flat_dynamic_range  | 帧能量 p95-p10 < 6 dB             | discard —— 疑似纯噪/常数信号  |
  | too_short           | 时长 < 0.5s（trim 后同样判定）    | discard —— 剔除               |
  | empty_file          | 0 采样点                          | discard —— 剔除               |
  | read_error          | 解码失败                          | discard —— 剔除               |
  | sr_mismatch         | 采样率与 --sr 不一致              | discard —— 剔除（不自动重采样）|
  | multichannel        | 声道数 > 1（--flag-multichannel） | 仅标记，默认不处置            |

action 模式（--action，逗号组合）：
  report     只检查出报告（默认，不改动任何文件）
  fix        对可修复文件执行修复，结果镜像写入 --fix-dir（不覆盖原文件）
  quarantine 对不可修复文件移动到 --quarantine-dir（镜像目录结构）

--fix-dir / --quarantine-dir 均不得等于或位于 root 之内（就地写出会覆盖源文件）；
quarantine 目标路径已存在时报错退出，不静默覆盖。

修复流水线顺序：去直流 -> 裁首尾静音 -> 电平归一化 -> 全项复检。
复检仍不通过的文件按不可修复处理（quarantine 开启时一并隔离）。

用法示例：
  # 1. 小样本试跑，只看报告
  python -m tools.ops.data.audio_qc_check /path/to/clean --limit 1000

  # 2. 全量只读报告
  python -m tools.ops.data.audio_qc_check /path/to/clean \
      --workers 64 -o qc_report.jsonl --summary qc_summary.json

  # 3. 修复 + 隔离（先小样本验证！）
  python -m tools.ops.data.audio_qc_check /path/to/clean --limit 5000 \
      --action fix,quarantine --fix-dir /path/to/clean_fixed \
      --quarantine-dir /path/to/clean_quarantine -o qc_report.jsonl
"""

import argparse
import functools
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np
import soundfile as sf

AUDIO_EXTS = ("flac", "wav", "mp3", "ogg", "opus", "m4a")

ALL_CHECKS = ("clipping", "level", "dynamic_range", "silence", "dc", "duration", "format")

# flag 前缀 -> 处置类别：fatal = 不可修复剔除，fixable = 可尝试修复，info = 仅标记
FATAL_FLAGS = {
    "clipping_run", "clipping_ratio", "rms_hard_floor", "flat_dynamic_range",
    "too_short", "empty_file", "read_error", "sr_mismatch",
}
FIXABLE_FLAGS = {"peak_too_high", "peak_too_low", "rms_too_low", "too_much_silence", "dc_offset"}
INFO_FLAGS = {"multichannel"}


def parse_args():
    p = argparse.ArgumentParser(
        description="批量音频质检与处置：截幅/削波剔除，电平/静音/直流可修复，详见文件头决策表",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("root", help="待检查的音频根目录（递归扫描）")
    p.add_argument("--ext", default="flac,wav", help="检查的扩展名，逗号分隔")
    p.add_argument("--checks", default=",".join(ALL_CHECKS),
                   help=f"启用哪些检查，逗号分隔，可选：{','.join(ALL_CHECKS)}")
    p.add_argument("--action", default="report",
                   help="处置动作，逗号组合：report（只出报告）/ fix（修复到 --fix-dir）/ "
                        "quarantine（不可修复文件移入 --quarantine-dir）")
    p.add_argument("--fix-dir", default="", help="修复后文件镜像输出目录（action 含 fix 时必填）")
    p.add_argument("--quarantine-dir", default="",
                   help="不可修复文件镜像隔离目录（action 含 quarantine 时必填）")
    p.add_argument("--target-peak-db", type=float, default=-3.0,
                   help="电平修复的目标 peak dBFS")
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8),
                   help="并行进程数；1 表示单进程")
    p.add_argument("--limit", type=int, default=0, help="只检查前 N 个文件（0 = 全部）")
    p.add_argument("--frac", type=float, default=1.0,
                   help="随机抽样比例 (0,1]，与 --limit 取较小集合")
    p.add_argument("--seed", type=int, default=0, help="抽样随机种子")
    p.add_argument("-o", "--output", default="", help="问题文件 JSONL 报告输出路径")
    p.add_argument("--dump-all", action="store_true",
                   help="报告里写入全部文件指标（不只是问题文件）")
    p.add_argument("--summary", default="", help="汇总统计 JSON 输出路径")

    # ---- 截幅/削波 ----
    g = p.add_argument_group("clipping 截幅/削波")
    g.add_argument("--clip-threshold", type=float, default=0.9999,
                   help="视为触顶的 |x| 阈值（满幅=1.0）")
    g.add_argument("--clip-min-run", type=int, default=3,
                   help="连续 >=N 个触顶样本即判为削波")
    g.add_argument("--clip-ratio-max", type=float, default=0.0,
                   help="触顶样本占比超过该值即标记（0 = 任何触顶都标记）")

    # ---- 电平 ----
    g = p.add_argument_group("level 电平")
    g.add_argument("--peak-max-db", type=float, default=-0.1,
                   help="peak 高于该 dBFS 视为接近满幅（削波风险，可增益修复）")
    g.add_argument("--peak-min-db", type=float, default=-20.0,
                   help="peak 低于该 dBFS 视为电平过低（可增益修复）")
    g.add_argument("--rms-min-db", type=float, default=-50.0,
                   help="整体 RMS 低于该 dBFS 视为过安静")
    g.add_argument("--rms-hard-floor-db", type=float, default=-60.0,
                   help="整体 RMS 低于该 dBFS 视为几乎无声，直接剔除")

    # ---- 动态范围 ----
    g = p.add_argument_group("dynamic_range 动态范围")
    g.add_argument("--dr-min-db", type=float, default=6.0,
                   help="帧能量 p95-p10 低于该 dB 视为动态范围过平，剔除")
    g.add_argument("--frame-ms", type=float, default=25.0, help="帧长 ms（静音/动态范围共用）")

    # ---- 静音 ----
    g = p.add_argument_group("silence 静音")
    g.add_argument("--silence-db", type=float, default=-50.0,
                   help="帧 RMS 低于该 dBFS 视为静音帧")
    g.add_argument("--silence-ratio-max", type=float, default=0.7,
                   help="静音帧占比超过该值即标记并尝试裁剪")
    g.add_argument("--trim-pad-ms", type=float, default=50.0,
                   help="裁剪首尾静音时两侧保留的 padding ms")

    # ---- 直流偏移 ----
    g = p.add_argument_group("dc 直流偏移")
    g.add_argument("--dc-offset-max", type=float, default=0.01,
                   help="|mean| 超过该值（线性幅度）即标记并修复")

    # ---- 时长 ----
    g = p.add_argument_group("duration 时长")
    g.add_argument("--min-duration", type=float, default=0.5,
                   help="时长低于该秒数即标记剔除（trim 后同样判定）")

    # ---- 格式 ----
    g = p.add_argument_group("format 格式")
    g.add_argument("--sr", type=int, default=0,
                   help="期望采样率，>0 时不一致即标记剔除（0 = 不检查）")
    g.add_argument("--flag-multichannel", action="store_true",
                   help="多声道文件也标记出来（仅标记，不处置）")
    return p.parse_args()


def _db(x, floor=1e-12):
    return 20.0 * np.log10(max(float(x), floor))


def _frame_rms_db(x, frame_len):
    """按帧计算 RMS dB，用于静音占比与动态范围。"""
    n_frames = len(x) // frame_len
    if n_frames < 2:
        return None
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-12))


def _max_run(mask):
    """布尔数组中最长连续 True 长度。"""
    if not mask.any():
        return 0
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return int((ends - starts).max()) if len(starts) else 0


def check_array(x, sr, args, path_metrics=None):
    """对波形 (n_samples, n_channels) float32 做全项检查，返回 (flags, metrics)。"""
    flags = []
    metrics = dict(path_metrics or {})
    metrics["sr"] = sr
    metrics["channels"] = x.shape[1]
    metrics["duration"] = round(len(x) / float(sr), 3)

    if x.size == 0:
        return ["empty_file"], metrics

    if "duration" in args.checks and metrics["duration"] < args.min_duration:
        flags.append(f"too_short:{metrics['duration']}s")

    abs_max = np.abs(x).max(axis=0)
    peak = float(abs_max.max())
    rms_db = _db(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    metrics["peak_db"] = round(_db(peak), 2)
    metrics["rms_db"] = round(rms_db, 2)

    if "clipping" in args.checks:
        clipped = np.abs(x) >= args.clip_threshold
        clip_ratio = float(clipped.mean())
        max_run = max((_max_run(clipped[:, ch]) for ch in range(x.shape[1])), default=0)
        metrics["clip_ratio"] = round(clip_ratio, 6)
        metrics["clip_max_run"] = max_run
        if max_run >= args.clip_min_run:
            flags.append(f"clipping_run:{max_run}")
        elif clip_ratio > args.clip_ratio_max and clip_ratio > 0:
            flags.append(f"clipping_ratio:{clip_ratio:.4f}")

    if "level" in args.checks:
        if metrics["peak_db"] > args.peak_max_db and "clipping_run" not in " ".join(flags):
            flags.append(f"peak_too_high:{metrics['peak_db']}dBFS")
        if metrics["peak_db"] < args.peak_min_db:
            flags.append(f"peak_too_low:{metrics['peak_db']}dBFS")
        if rms_db < args.rms_hard_floor_db:
            flags.append(f"rms_hard_floor:{metrics['rms_db']}dBFS")
        elif rms_db < args.rms_min_db:
            flags.append(f"rms_too_low:{metrics['rms_db']}dBFS")

    if "dc" in args.checks:
        dc = float(np.abs(x.mean(axis=0)).max())
        metrics["dc_offset"] = round(dc, 6)
        if dc > args.dc_offset_max:
            flags.append(f"dc_offset:{dc:.4f}")

    if "silence" in args.checks or "dynamic_range" in args.checks:
        frame_len = max(1, int(sr * args.frame_ms / 1000.0))
        mono = np.abs(x).max(axis=1) if x.shape[1] > 1 else x[:, 0]
        fr = _frame_rms_db(mono, frame_len)
        if fr is not None:
            metrics["frame_rms_p95_db"] = round(float(np.percentile(fr, 95)), 2)
            metrics["frame_rms_p10_db"] = round(float(np.percentile(fr, 10)), 2)
            if "silence" in args.checks:
                sil_ratio = float((fr < args.silence_db).mean())
                metrics["silence_ratio"] = round(sil_ratio, 4)
                if sil_ratio > args.silence_ratio_max:
                    flags.append(f"too_much_silence:{sil_ratio:.2f}")
            if "dynamic_range" in args.checks:
                dr = metrics["frame_rms_p95_db"] - metrics["frame_rms_p10_db"]
                metrics["dynamic_range_db"] = round(dr, 2)
                # 只在非全静音文件上判动态范围
                if metrics.get("silence_ratio", 0.0) <= args.silence_ratio_max \
                        and dr < args.dr_min_db:
                    flags.append(f"flat_dynamic_range:{dr:.1f}dB")
    return flags, metrics


def fix_array(x, sr, args):
    """执行可修复操作：去直流 -> 裁首尾静音 -> 电平归一化。返回 (x_fixed, fix_ops)。"""
    ops = []
    # 1. 去直流
    dc = x.mean(axis=0)
    if np.abs(dc).max() > args.dc_offset_max:
        x = x - dc
        ops.append("remove_dc")
    # 2. 裁首尾静音
    frame_len = max(1, int(sr * args.frame_ms / 1000.0))
    mono = np.abs(x).max(axis=1) if x.shape[1] > 1 else x[:, 0]
    fr = _frame_rms_db(mono, frame_len)
    if fr is not None:
        non_sil = np.where(fr >= args.silence_db)[0]
        if len(non_sil):
            pad = int(sr * args.trim_pad_ms / 1000.0)
            start = max(0, non_sil[0] * frame_len - pad)
            end = min(len(x), (non_sil[-1] + 1) * frame_len + pad)
            if start > 0 or end < len(x):
                x = x[start:end]
                ops.append(f"trim_silence:{start}->{end}")
    # 3. 电平归一化：peak 超出 [peak_min, peak_max] 时缩放到目标 peak
    if len(x):
        peak = float(np.abs(x).max())
        peak_db = _db(peak)
        if peak > 0 and (peak_db > args.peak_max_db or peak_db < args.peak_min_db):
            gain = 10.0 ** ((args.target_peak_db - peak_db) / 20.0)
            # 归一化后不允许引入削波
            if peak * gain < 1.0:
                x = (x * gain).astype(np.float32)
                ops.append(f"normalize:{peak_db:.1f}->{args.target_peak_db}dBFS")
    return x.astype(np.float32), ops


def classify(flags):
    """flag 列表 -> (fatal 列表, fixable 列表, info 列表)。"""
    fatal, fixable, info = [], [], []
    for f in flags:
        key = f.split(":", 1)[0]
        if key in FATAL_FLAGS:
            fatal.append(f)
        elif key in FIXABLE_FLAGS:
            fixable.append(f)
        else:
            info.append(f)
    return fatal, fixable, info


def process_file(path, args):
    """单文件检查 + 处置，返回结果 dict。action=report 时不写任何文件。"""
    rec = {"path": path, "flags": [], "metrics": {}, "action": "ok", "fix_ops": []}
    try:
        info = sf.info(path)
        rec["metrics"] = {"sr": info.samplerate, "channels": info.channels,
                          "duration": round(info.frames / float(info.samplerate or 1), 3)}
        if "format" in args.checks:
            if info.frames == 0:
                rec["flags"] = ["empty_file"]
                return _finish(rec, args)
            if args.sr > 0 and info.samplerate != args.sr:
                rec["flags"].append(f"sr_mismatch:{info.samplerate}")
            if args.flag_multichannel and info.channels > 1:
                rec["flags"].append(f"multichannel:{info.channels}")
        if not any(c in args.checks for c in
                   ("clipping", "level", "dynamic_range", "silence", "dc", "duration")):
            return _finish(rec, args)

        x, sr = sf.read(path, dtype="float32", always_2d=True)
        flags, metrics = check_array(x, sr, args, path_metrics=rec["metrics"])
        rec["flags"].extend(flags)
        rec["metrics"] = metrics
    except Exception as e:  # noqa: BLE001 —— 单文件失败不应中断整批
        rec["flags"] = [f"read_error:{type(e).__name__}"]
        return _finish(rec, args)

    fatal, fixable, _ = classify(rec["flags"])
    # 无可修复项，或 action 不含 fix：直接按判定结果结束
    if fatal or not fixable or "fix" not in args.actions:
        return _finish(rec, args, extra_fatal=fatal)

    # 尝试修复并全项复检
    try:
        x_fix, ops = fix_array(x, sr, args)
        rec["fix_ops"] = ops
        flags2, metrics2 = check_array(x_fix, sr, args, path_metrics=rec["metrics"])
        fatal2, fixable2, _ = classify(flags2)
        if fatal2 or fixable2:
            # 修不好：并入原 flags，按不可修复处理
            rec["flags"] = sorted(set(rec["flags"]) | set(flags2))
            rec["metrics"].update({f"postfix_{k}": v for k, v in metrics2.items()})
            return _finish(rec, args, extra_fatal=fatal2 or ["fix_failed"])
        if args.fix_dir:
            rel = os.path.relpath(path, args.root)
            out_path = os.path.join(args.fix_dir, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            subtype = "PCM_16" if out_path.lower().endswith(".wav") else None
            sf.write(out_path, x_fix, sr, subtype=subtype)
            rec["fixed_path"] = out_path
        rec["metrics"].update({f"postfix_{k}": v for k, v in metrics2.items()})
        rec["action"] = "fixed"
        return rec
    except Exception as e:  # noqa: BLE001
        rec["flags"].append(f"fix_error:{type(e).__name__}")
        return _finish(rec, args, extra_fatal=["fix_error"])


def _finish(rec, args, extra_fatal=None):
    """按最终判定执行 quarantine 并填 action 字段。extra_fatal 为复检/修复失败追加的致命原因。"""
    fatal, fixable, info = classify(rec["flags"])
    for f in (extra_fatal or []):
        if f not in fatal:
            fatal.append(f)
    if fatal:
        rec["action"] = "discard"
        if "quarantine" in args.actions and args.quarantine_dir:
            rel = os.path.relpath(rec["path"], args.root)
            dst = os.path.join(args.quarantine_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # 目标已存在（含悬空 symlink）时拒绝继续：shutil.move 会覆盖或移入，
            # 二者都不符合「隔离」语义，必须报错而不是静默改动已有文件
            if os.path.lexists(dst):
                raise RuntimeError(
                    f"隔离目标已存在，拒绝覆盖: {dst}（源文件: {rec['path']}）"
                )
            shutil.move(rec["path"], dst)
            rec["quarantine_path"] = dst
    elif fixable:
        rec["action"] = "fixable"  # 可修复但未开启 fix，仅在报告中标出
    elif info:
        rec["action"] = "info"
    else:
        rec["action"] = "ok"
    return rec


def collect_files(root, exts):
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.rsplit(".", 1)[-1].lower() in exts:
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def main():
    args = parse_args()
    args.checks = {c.strip() for c in args.checks.split(",") if c.strip()}
    bad = args.checks - set(ALL_CHECKS)
    if bad:
        sys.exit(f"未知检查项: {bad}，可选: {ALL_CHECKS}")
    args.actions = {a.strip() for a in args.action.split(",") if a.strip()}
    bad = args.actions - {"report", "fix", "quarantine"}
    if bad:
        sys.exit(f"未知 action: {bad}，可选: report/fix/quarantine")
    if "fix" in args.actions and not args.fix_dir:
        sys.exit("action 含 fix 时必须指定 --fix-dir")
    if "quarantine" in args.actions and not args.quarantine_dir:
        sys.exit("action 含 quarantine 时必须指定 --quarantine-dir")
    # 输出目录不得等于或位于 root 之内：镜像写出按 root 相对路径落盘，一旦
    # 与源目录重叠就会覆盖原文件（fix 为不可逆即时重写，quarantine 为移动）
    root_abs = os.path.realpath(args.root)
    for flag, out_dir in (("--fix-dir", args.fix_dir), ("--quarantine-dir", args.quarantine_dir)):
        if not out_dir:
            continue
        out_abs = os.path.realpath(out_dir)
        if out_abs == root_abs or out_abs.startswith(root_abs + os.sep):
            sys.exit(
                f"{flag} 不能等于或位于 root 之内（root={root_abs}，{flag}={out_abs}）："
                "就地写出会覆盖源文件"
            )
    exts = {e.strip().lower() for e in args.ext.split(",") if e.strip()}

    print(f"[scan] 扫描 {args.root} ...", flush=True)
    t0 = time.time()
    files = collect_files(args.root, exts)
    print(f"[scan] 共 {len(files)} 个音频文件，耗时 {time.time() - t0:.1f}s", flush=True)
    if not files:
        return

    if args.frac < 1.0:
        rng = np.random.default_rng(args.seed)
        files = rng.choice(files, size=max(1, int(len(files) * args.frac)), replace=False).tolist()
        files.sort()
    if args.limit > 0:
        files = files[: args.limit]
    print(f"[run] 本次检查 {len(files)} 个文件，workers={args.workers}，"
          f"checks={sorted(args.checks)}，action={sorted(args.actions)}", flush=True)

    flag_counts, action_counts = {}, {}
    n_done = 0
    t0 = time.time()
    fout = open(args.output, "w", encoding="utf-8") if args.output else None

    def handle(rec):
        nonlocal n_done
        n_done += 1
        action_counts[rec["action"]] = action_counts.get(rec["action"], 0) + 1
        for f in rec["flags"]:
            key = f.split(":", 1)[0]
            flag_counts[key] = flag_counts.get(key, 0) + 1
        if fout and (rec["flags"] or rec["action"] != "ok" or args.dump_all):
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if n_done % 2000 == 0 or n_done == len(files):
            dt = time.time() - t0
            print(f"[progress] {n_done}/{len(files)}  actions={action_counts}  "
                  f"speed={n_done / max(dt, 1e-6):.0f}/s", flush=True)

    worker_fn = functools.partial(process_file, args=args)
    if args.workers <= 1:
        for f in files:
            handle(worker_fn(f))
    else:
        with mp.Pool(args.workers) as pool:
            for rec in pool.imap_unordered(worker_fn, files, chunksize=16):
                handle(rec)

    if fout:
        fout.close()

    dt = time.time() - t0
    summary = {
        "root": os.path.abspath(args.root),
        "checked": n_done,
        "action_counts": dict(sorted(action_counts.items(), key=lambda kv: -kv[1])),
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda kv: -kv[1])),
        "checks": sorted(args.checks),
        "actions": sorted(args.actions),
        "elapsed_s": round(dt, 1),
    }
    print("\n===== 质检汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        print(f"明细报告: {args.output}")
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"汇总已写入: {args.summary}")


if __name__ == "__main__":
    main()
