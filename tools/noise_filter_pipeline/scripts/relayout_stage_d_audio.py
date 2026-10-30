"""
把已完成的 Stage D 输出音频从 sr_dataset 布局迁移为 sr_dataset_label 布局。

做法（对硬链接/manifest 原子性安全）：
  1. 流式读取 D2/d2_merged.jsonl，逐条计算新路径
     {audio_root}/sr_<sr>/<dataset>/<label>/<basename>（与原代码同套清洗规则）；
  2. os.rename 移动音频（同文件系统纯元数据操作，不重写数据）；
  3. 行内字节级替换旧路径为新路径，tmp+replace 原子更新 merged manifest；
  4. 重新把 stage_d/d_manifest.jsonl 硬链接到新 merged，并用同一套路径替换
     重写 stage 根目录的 per-dataset d_manifest_<dataset>.jsonl。

可中断重跑：已在新位置的文件跳过（幂等）。
用法：
  python tools/noise_filter_pipeline/scripts/relayout_stage_d_audio.py \
      --root /data_nvme3n1/noise_filter/stage_d [--label-key audioset] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from noise_filter_pipeline.core.run_lock import stage_d_run_lock  # noqa: E402
from noise_filter_pipeline.utils.sanitize import sanitize_basename_for_rename  # noqa: E402

PATH_RE = re.compile(r'"path":\s*"([^"]+)"')
SR_RE = re.compile(r'"sr":\s*(\d+)')
DS_RE = re.compile(r'"dataset_id":\s*"([^"]+)"')
RID_RE = re.compile(r'"record_id":\s*"([^"]+)"')
D2_PASSED = '"D2": {"status": "passed"'


def _sr_dir_name(sr: int) -> str:
    sr = int(sr)
    if sr >= 1000 and sr % 1000 == 0:
        return f"sr_{sr // 1000}k"
    return f"sr_{sr}"


def _resolve_relayout(line: str, audio_root: str, label_re: re.Pattern) -> tuple:
    """解析一行 D2 passed 记录的目标布局信息。

    返回 (old_path, status, new_path, label)：
    - status "skip"：已在新布局（relpath 为 4 段）或无 path 字段，无需迁移；
    - status "missing_sr"：缺 sr 字段，无法确定采样率目录（由调用方报错）；
    - status "ok"：new_path 为计算出的目标布局路径（可能与 old_path 相同）。

    主循环与 per-dataset manifest 重写共用同一套解析，避免两处逻辑漂移
    （漂移会让中断后重跑无法自愈 per-dataset 的旧路径残留）。
    """
    pm = PATH_RE.search(line)
    if not pm:
        return None, "skip", None, None
    old_path = pm.group(1)
    # 已在三级布局下的路径跳过（幂等）
    rel = os.path.relpath(old_path, audio_root)
    if len(rel.split(os.sep)) == 4:
        return old_path, "skip", None, None
    sm = SR_RE.search(line)
    if not sm:
        return old_path, "missing_sr", None, None
    ds = DS_RE.search(line)
    lm = label_re.search(line)
    dataset = sanitize_basename_for_rename(ds.group(1) if ds else "unknown", is_file=False)
    sr = int(sm.group(1))
    label = sanitize_basename_for_rename(lm.group(1) if lm else "unknown", is_file=False)
    new_path = os.path.join(
        audio_root, _sr_dir_name(sr), dataset, label, os.path.basename(old_path)
    )
    return old_path, "ok", new_path, label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="stage_d 目录（含 D2/ 与 d_manifest.jsonl）")
    parser.add_argument("--label-key", default="audioset", help="标签字段名（默认 audioset）")
    parser.add_argument("--dry-run", action="store_true", help="只统计不移动")
    args = parser.parse_args()

    # 本脚本直接 os.rename 音频树、改写 d2_merged.jsonl 与 d_manifest_*.jsonl，
    # 与 Stage D writer、promote、cleanup 共用同一 run 锁；不持锁改写会让
    # 并发运行的写者读到半迁移状态（对照 cleanup/promote 的同类锁用法）
    with stage_d_run_lock(args.root):
        return _relayout_stage_d(args)


def _relayout_stage_d(args: argparse.Namespace) -> int:
    merged_path = os.path.join(args.root, "D2", "d2_merged.jsonl")
    final_path = os.path.join(args.root, "d_manifest.jsonl")
    audio_root = os.path.join(args.root, "D2", "audio")
    if not os.path.isfile(merged_path):
        print(f"merged manifest 不存在: {merged_path}")
        return 1

    label_re = re.compile(r'"%s":\s*"([^"]+)"' % re.escape(args.label_key))

    moved = already = missing = 0
    label_counter: Counter = Counter()
    mkdir_cache = set()
    tmp_path = merged_path + ".relayout_tmp"
    # sr 字段缺失/为 null 的记录：不得静默归入 sr_0/，收集后统一报错
    bad_sr_records: list = []

    # 第一遍：预扫 merged，先收集缺 sr 的记录——必须在任何 os.rename 之前
    # 中止，否则正常记录的文件已被移走、manifest 却保持旧路径（悬空）。
    # 预扫成本为一次只读遍历，换取"中止即全量无副作用"
    with open(merged_path, "rb") as src:
        for raw in src:
            if D2_PASSED.encode() not in raw:
                continue
            line = raw.decode("utf-8")
            old_path, status, _, _ = _resolve_relayout(line, audio_root, label_re)
            if status == "missing_sr":
                rid = RID_RE.search(line)
                bad_sr_records.append(rid.group(1) if rid else old_path)
    if bad_sr_records:
        print(f"错误：{len(bad_sr_records)} 条 D2 passed 记录缺少 sr 字段，无法确定采样率目录：")
        for rid in bad_sr_records[:20]:
            print(f"  - {rid}")
        if len(bad_sr_records) > 20:
            print(f"  ... 及其余 {len(bad_sr_records) - 20} 条")
        print("请先修复 manifest 中的 sr 字段再重跑。")
        return 1

    with open(merged_path, "rb") as src, open(tmp_path, "wb") as out:
        for raw in src:
            if D2_PASSED.encode() not in raw:
                out.write(raw)
                continue
            line = raw.decode("utf-8")
            old_path, status, new_path, label = _resolve_relayout(line, audio_root, label_re)
            if status in ("skip", "missing_sr"):
                # 预扫已保证无 missing_sr 到达此处（防御性保留分支）；
                # 无 path 字段的记录不参与计数（与旧行为一致）
                if old_path is not None:
                    already += 1
                out.write(raw)
                continue
            label_counter[label] += 1

            if old_path != new_path:
                if os.path.isfile(old_path):
                    if not args.dry_run:
                        d = os.path.dirname(new_path)
                        if d not in mkdir_cache:
                            os.makedirs(d, exist_ok=True)
                            mkdir_cache.add(d)
                        os.rename(old_path, new_path)
                    moved += 1
                elif os.path.isfile(new_path):
                    already += 1
                else:
                    missing += 1
                    out.write(raw)
                    continue
                # 字节级替换路径（路径在记录中唯一出现）
                line = line.replace(old_path, new_path, 1)
            out.write(line.encode("utf-8"))

    if args.dry_run:
        os.unlink(tmp_path)
    else:
        # 原子替换 merged（新 inode），再重建 d_manifest 硬链接
        os.replace(tmp_path, merged_path)
        if os.path.exists(final_path):
            os.remove(final_path)
        os.link(merged_path, final_path)
        # per-dataset d_manifest_<dataset>.jsonl 也持有旧路径：用同一套
        # 字节级替换重写（tmp+replace 原子更新），避免根目录 manifest 与
        # merged 路径不一致导致下游读到已移走的旧位置
        for fname in os.listdir(args.root):
            if not (fname.startswith("d_manifest_") and fname.endswith(".jsonl")):
                continue
            ds_path = os.path.join(args.root, fname)
            ds_tmp = ds_path + ".relayout_tmp"
            with open(ds_path, "rb") as src, open(ds_tmp, "wb") as out:
                for raw in src:
                    if D2_PASSED.encode() not in raw:
                        out.write(raw)
                        continue
                    line = raw.decode("utf-8")
                    # 独立计算目标布局路径，不依赖本次运行的 path_map：
                    # 中断后重跑时 merged 已全在新布局（跳过分支），path_map
                    # 为空，per-dataset 里残留的旧路径只有重新解析才能修复
                    old_path, status, new_path, _ = _resolve_relayout(line, audio_root, label_re)
                    if (
                        status == "ok"
                        and new_path != old_path
                        and os.path.isfile(new_path)
                    ):
                        # 目标布局文件已存在（主循环已迁移或中断前已完成移动）
                        line = line.replace(old_path, new_path, 1)
                    out.write(line.encode("utf-8"))
            os.replace(ds_tmp, ds_path)

    print(f"moved={moved} already={already} missing={missing}")
    print("Top 标签目录:")
    for lab, c in label_counter.most_common(20):
        print(f"  {c:8d}  {lab}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
