#!/usr/bin/env python3
"""Voicebank-Demand dataset preparation script.

Features:
  1. Resample non-16kHz audio to 16kHz (paper requirement)
  2. Generate 4-column SCP format (uid fs length path) readable by use_train

Usage:
    python tools/prepare_data_tools/scp/prepare_voicebank.py \
        --data_root /path/to/voicebank_demand \
        --output-dir data/se/voicebank_demand \
        --target_sr 16000

    # 缺少官方 testset 时，从 trainset 划出 2 个说话人作为 validation：
    python tools/prepare_data_tools/scp/prepare_voicebank.py \
        --data_root data/se/voicebank_demand \
        --output-dir data/se/voicebank_demand \
        --val-speakers 2
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tools.shared.file_scan import scan_audio_files  # noqa: E402
from tools.shared.scp_io import (  # noqa: E402
    serialize_manifest_relative_path,
    serialize_scp_entries,
    write_scp_entries,
)


def _collect_all_wavs(wav_dir: str) -> list[str]:
    """Recursively collect absolute paths of all wav files in directory."""
    files = scan_audio_files(wav_dir, exts=(".wav",), recursive=True)
    return sorted([str(p) for p in files])


def resample_if_needed(wav_dir: str, target_sr: int) -> str:
    """If any audio in directory is not at target sample rate, resample those files to {wav_dir}_16k/ and return new path.

    Sample rate is probed per file; files already at target_sr are copied
    as-is, so a directory with mixed sample rates is converted correctly.
    """
    all_originals = _collect_all_wavs(wav_dir)
    if not all_originals:
        return wav_dir

    # 逐文件探测采样率：与目标不一致的文件才需要重采样
    resample_map: dict[str, int] = {}
    for src in all_originals:
        try:
            sr = sf.info(src).samplerate
        except Exception as e:
            print(f"  Warning: skipping unreadable file {src}: {e}")
            continue
        if sr != target_sr:
            resample_map[src] = sr
    if not resample_map:
        return wav_dir

    out_dir = wav_dir.rstrip("/\\") + f"_{target_sr // 1000}k"
    if os.path.exists(out_dir):
        existing = _collect_all_wavs(out_dir)
        if len(existing) == len(all_originals):
            print(f"  Resampled dir already exists with {len(existing)} files, skipping: {out_dir}")
            return out_dir

    os.makedirs(out_dir, exist_ok=True)
    src_srs = sorted({sr for sr in resample_map.values()})
    print(f"  Resampling {len(resample_map)}/{len(all_originals)} files (sr {src_srs}) -> {target_sr}: "
          f"{wav_dir} -> {out_dir}")

    try:
        import torchaudio
        from torchaudio.transforms import Resample
        resamplers = {sr: Resample(sr, target_sr) for sr in src_srs}

        count = 0
        for src in all_originals:
            rel = os.path.relpath(src, wav_dir)
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if src in resample_map:
                wav, sr = torchaudio.load(src, backend="soundfile")
                wav_rs = resamplers[sr](wav)
                torchaudio.save(dst, wav_rs, target_sr, backend="soundfile")
            else:
                shutil.copy2(src, dst)
            count += 1
            if count % 100 == 0:
                print(f"    {count} files resampled/copied...")
        print(f"  Done: {count} files resampled/copied")
    except Exception:
        import librosa
        count = 0
        for src in all_originals:
            rel = os.path.relpath(src, wav_dir)
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if src in resample_map:
                wav, _sr = librosa.load(src, sr=target_sr, mono=True)
                sf.write(dst, wav, target_sr)
            else:
                shutil.copy2(src, dst)
            count += 1
            if count % 100 == 0:
                print(f"    {count} files resampled/copied...")
        print(f"  Done: {count} files resampled/copied (librosa)")

    return out_dir


def scan_wavs(wav_dir: str) -> list[tuple[str, int, int, str]]:
    """递归扫描 wav，返回 ``(uid, samplerate, frames, absolute_path)``。"""
    # uid 取自文件名 stem；递归目录下同名文件会让 SCP 出现重复行（下游按 uid 建 dict
    # 会静默丢数据），与 prepare_scp._check_uid_collisions 一致直接 fail-fast。
    by_uid: dict[str, tuple[str, int, int, str]] = {}
    for fpath in scan_audio_files(wav_dir, exts=(".wav",), recursive=True):
        try:
            info = sf.info(fpath)
        except Exception as e:
            print(f"  Warning: skipping unreadable file {fpath}: {e}")
            continue
        uid = os.path.splitext(fpath.name)[0]
        entry = (uid, info.samplerate, info.frames, str(fpath))
        previous = by_uid.get(uid)
        if previous is not None:
            raise ValueError(
                f"duplicate uid {uid!r} under {wav_dir}: {previous[3]} and {entry[3]}; "
                "请先重命名冲突文件"
            )
        by_uid[uid] = entry
    return sorted(by_uid.values(), key=lambda e: e[0])


def _write_relative_scp(
    entries: list[tuple[str, int, int, str]], output_path: Path
) -> None:
    """按目标 SCP 所在目录写出可迁移的四列清单。"""
    portable_entries = [
        (uid, fs, frames, serialize_manifest_relative_path(path, output_path))
        for uid, fs, frames, path in entries
    ]
    write_scp_entries(portable_entries, output_path, fmt="scp", four_col=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prepare Voicebank-Demand for use_train")
    p.add_argument("--data_root", required=True,
                    help="Root directory of DS_10283_1942")
    p.add_argument("--output-dir", default="data/se/voicebank_demand",
                    help="Output directory for SCP files")
    p.add_argument("--target_sr", type=int, default=16000,
                    help="Target sample rate (paper: 16000)")
    p.add_argument("--val-speakers", type=int, default=0,
                    help="从 trainset 划出 N 个说话人（按 id 字典序取最后 N 个）作为 "
                         "validation，输出 val/{clean,noisy}.scp 并从 train scp 剔除；"
                         "0 = 不划分（默认，行为不变）")
    args = p.parse_args(argv)

    root = Path(args.data_root)
    out = Path(args.output_dir)

    splits = {
        "train": {
            "clean_dir": "clean_trainset_wav",
            "noisy_dir": "noisy_trainset_wav",
        },
        "test": {
            "clean_dir": "clean_testset_wav",
            "noisy_dir": "noisy_testset_wav",
        },
    }

    for split_name, dirs in splits.items():
        clean_dir = str(root / dirs["clean_dir"])
        noisy_dir = str(root / dirs["noisy_dir"])

        if not os.path.exists(clean_dir):
            print(f"WARNING: {clean_dir} does not exist, skipping {split_name}")
            continue
        if not os.path.exists(noisy_dir):
            print(f"WARNING: {noisy_dir} does not exist, skipping {split_name}")
            continue

        print(f"\n[{split_name}]")

        clean_dir = resample_if_needed(clean_dir, args.target_sr)
        noisy_dir = resample_if_needed(noisy_dir, args.target_sr)

        clean_entries = scan_wavs(clean_dir)
        noisy_entries = scan_wavs(noisy_dir)

        clean_uids = {e[0] for e in clean_entries}
        noisy_uids = {e[0] for e in noisy_entries}
        common = clean_uids & noisy_uids
        if len(common) != len(clean_uids):
            print(f"  WARNING: {len(clean_uids) - len(common)} clean files have no noisy match")
        if len(common) != len(noisy_uids):
            print(f"  WARNING: {len(noisy_uids) - len(common)} noisy files have no clean match")

        clean_entries = [e for e in clean_entries if e[0] in common]
        noisy_entries = [e for e in noisy_entries if e[0] in common]

        # 可选：从 trainset 划出 val_speakers 个说话人作为 validation（防止 train/val 说话人泄漏）
        if split_name == "train" and args.val_speakers > 0:
            speakers = sorted({uid.split("_")[0] for uid, *_ in clean_entries})
            if args.val_speakers >= len(speakers):
                raise ValueError(
                    f"--val-speakers={args.val_speakers} >= trainset 说话人数 {len(speakers)}，无法划分"
                )
            val_spk = set(speakers[-args.val_speakers:])
            val_clean = [e for e in clean_entries if e[0].split("_")[0] in val_spk]
            val_noisy = [e for e in noisy_entries if e[0].split("_")[0] in val_spk]
            clean_entries = [e for e in clean_entries if e[0].split("_")[0] not in val_spk]
            noisy_entries = [e for e in noisy_entries if e[0].split("_")[0] not in val_spk]
            val_clean_path = out / "val" / "clean.scp"
            _write_relative_scp(val_clean, val_clean_path)
            print(f"  Written {len(val_clean)} entries -> {val_clean_path}")
            val_noisy_path = out / "val" / "noisy.scp"
            _write_relative_scp(val_noisy, val_noisy_path)
            print(f"  Written {len(val_noisy)} entries -> {val_noisy_path}")
            print(f"  Held-out val speakers: {sorted(val_spk)} "
                  f"({len(val_clean)} utterances)")

        clean_path = out / split_name / "clean.scp"
        _write_relative_scp(clean_entries, clean_path)
        print(f"  Written {len(clean_entries)} entries -> {clean_path}")
        noisy_path = out / split_name / "noisy.scp"
        _write_relative_scp(noisy_entries, noisy_path)
        print(f"  Written {len(noisy_entries)} entries -> {noisy_path}")

        speakers = sorted({uid.split("_")[0] for uid, *_ in clean_entries})
        srs = {sr for _, sr, _, _ in clean_entries}
        durations = [frames / sr for _, sr, frames, _ in clean_entries]
        print(f"  Speakers: {speakers}")
        print(f"  Sample rates: {srs}")
        print(f"  Files: {len(clean_entries)}")
        if durations:
            print(f"  Duration: {sum(durations)/3600:.2f}h "
                  f"(avg {sum(durations)/len(durations):.2f}s)")
        else:
            print(f"  Duration: 0.00h (no files found)")

    print("\nDone. SCP files are ready for training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())