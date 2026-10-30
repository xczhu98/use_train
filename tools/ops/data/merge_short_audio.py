#!/usr/bin/env python
"""把过短的音频碎片按顺序拼接成满足最小时长的新文件。

典型用法：配合 audio_qc_check.py 的 report，把「仅 too_short」的干净碎片
（非 spk 相关数据集）拼接成 >= --min-duration 的新文件：

  python -m tools.ops.data.merge_short_audio \
      --from-report data/shared/clean/filter/qc_fix_report.jsonl \
      --out-dir data/shared/clean/filter/audio_filter_en_16k/urgent26_16k_merged \
      --min-duration 2.0

行为说明：
  - 只选择 flags 恰好为 {too_short}（可含 info 类标记）的记录；
    带削波/电平等其他问题的碎片不参与拼接。
  - 文件被 quarantine 移动过时，自动改用报告里的 quarantine_path 读取。
  - 按路径排序顺序拼接，段间插入 --gap-ms 静音避免硬切换爆音；
    累积时长 >= --min-duration 即切出一个新文件（单段超长不截断）。
  - 采样率不一致的碎片分组各自拼接，互不混用。
  - 输出 manifest JSONL 记录每个新文件的来源列表，便于追溯与后续删除确认。
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

INFO_PREFIXES = ("multichannel",)


def parse_args():
    p = argparse.ArgumentParser(
        description="把仅 too_short 的音频碎片拼接成满足最小时长的新文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--from-report", required=True,
                   help="audio_qc_check.py 生成的 JSONL 报告（含 quarantine_path 时优先用）")
    p.add_argument("--out-dir", required=True, help="拼接结果输出目录")
    p.add_argument("--min-duration", type=float, default=2.0, help="拼接目标最小时长 s")
    p.add_argument("--gap-ms", type=float, default=30.0, help="段间静音间隔 ms")
    p.add_argument("--exclude", default="_spk",
                   help="路径含该子串的碎片不参与拼接（spk 数据集不可跨说话人拼接）；"
                        "传空字符串关闭")
    p.add_argument("--prefix", default="merged", help="输出文件名前缀")
    p.add_argument("--manifest", default="",
                   help="来源清单 JSONL 输出路径（默认 <out-dir>/merge_manifest.jsonl）")
    return p.parse_args()


def load_candidates(args):
    """从报告筛出「仅 too_short」且未被 exclude 命中的可读文件路径。"""
    cands = []
    skipped = {"other_flags": 0, "excluded": 0, "missing": 0}
    with open(args.from_report, encoding="utf-8") as fin:
        # 报告可能达数万行：逐行流式消费，不整份读入内存
        for raw_line in fin:
            if not raw_line.strip():
                continue
            rec = json.loads(raw_line)
            keys = {f.split(":", 1)[0] for f in rec.get("flags", [])}
            keys -= {k for k in keys if k.startswith(INFO_PREFIXES)}
            if keys != {"too_short"}:
                if "too_short" in keys:
                    skipped["other_flags"] += 1
                continue
            path = rec.get("quarantine_path") or rec["path"]
            if args.exclude and args.exclude in path:
                skipped["excluded"] += 1
                continue
            if not os.path.isfile(path):
                skipped["missing"] += 1
                continue
            cands.append(path)
    cands.sort()
    return cands, skipped


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = args.manifest or os.path.join(args.out_dir, "merge_manifest.jsonl")

    cands, skipped = load_candidates(args)
    print(f"[load] 候选碎片 {len(cands)} 个；跳过: {skipped}", flush=True)
    if not cands:
        return

    # 按采样率分组拼接，互不混用
    groups = {}
    for p in cands:
        sr = sf.info(p).samplerate
        groups.setdefault(sr, []).append(p)

    n_written = 0
    with open(manifest_path, "w", encoding="utf-8") as fout:
        for sr, paths in sorted(groups.items()):
            gap = np.zeros(int(sr * args.gap_ms / 1000.0), dtype=np.float32)
            min_len = int(sr * args.min_duration)
            buf, srcs = [], []
            buf_len = 0
            idx = 0

            def flush():
                nonlocal n_written, buf, srcs, buf_len, idx
                if not buf:
                    return
                x = np.concatenate(buf)
                name = f"{args.prefix}_{sr}hz_{idx:05d}.flac"
                out_path = os.path.join(args.out_dir, name)
                sf.write(out_path, x, sr)
                fout.write(json.dumps({
                    "path": out_path, "sr": sr,
                    "duration": round(len(x) / sr, 3),
                    "num_sources": len(srcs), "sources": list(srcs),
                }, ensure_ascii=False) + "\n")
                n_written += 1
                idx += 1
                buf, srcs, buf_len = [], [], 0

            for p in paths:
                x, _ = sf.read(p, dtype="float32", always_2d=True)
                if x.shape[1] > 1:  # 多声道混成单声道再拼
                    x = x.mean(axis=1, keepdims=True)
                else:
                    x = x[:, :1]
                if buf:
                    buf.append(gap)
                    buf_len += len(gap)
                buf.append(x[:, 0])
                buf_len += len(x)
                srcs.append(p)
                if buf_len >= min_len:
                    flush()
            flush()  # 尾部不足 min-duration 的也写出，由后续 QC 判定
            print(f"[merge] sr={sr}: {len(paths)} 个碎片 -> {idx} 个文件", flush=True)

    print(f"[done] 共写出 {n_written} 个拼接文件到 {args.out_dir}")
    print(f"[done] 来源清单: {manifest_path}")


if __name__ == "__main__":
    main()
