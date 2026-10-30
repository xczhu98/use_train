#!/usr/bin/env python3
"""AEC 离线三元组扫描工具；GenerationManifest publication 已下线。

``scan_wavs`` 仍提供严格 WAV 扫描；原有 AEC offline publication CLI 在
schema v2 reload 支持完成前显式 fail-closed，不会写入输出目录。

迁移计划：AEC schema v2 reload 就绪后，把 ``main`` 的发布改走
``tools.prepare_data_tools.offline_publication.publish_owned_offline_dataset_roots``；
迁移完成前 ``main`` 以外的骨架保留仅供内部测试，不得作为生产发布路径。

Usage:
    python tools/prepare_data_tools/scp/prepare_aec_offline.py \
        --mic_dir /path/to/mic --farend_dir /path/to/farend \
        --nearend_dir /path/to/nearend [--error_dir /path/to/error] \
        --output-dir data/aec_offline [--val_ratio 0.05]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tools.shared.file_scan import scan_audio_files  # noqa: E402


def scan_wavs(wav_dir: str) -> list[tuple[str, int, int, str]]:
    """严格扫描 wav；unreadable、空音频或重复 uid 均 fail-closed。"""
    by_uid: dict[str, tuple[str, int, int, str]] = {}
    for fpath in scan_audio_files(wav_dir, exts=(".wav",), recursive=True):
        try:
            info = sf.info(fpath)
        except Exception as exc:
            raise ValueError(f"unreadable wav {fpath}: {exc}") from exc
        uid = fpath.stem
        entry = (uid, int(info.samplerate), int(info.frames), str(fpath))
        if entry[1] <= 0 or entry[2] <= 0:
            raise ValueError(f"unreadable wav metadata {fpath}: fs={entry[1]} frames={entry[2]}")
        if uid in by_uid:
            raise ValueError(f"duplicate uid {uid!r}: {by_uid[uid][3]} and {entry[3]}")
        by_uid[uid] = entry
    return sorted(by_uid.values(), key=lambda item: item[0])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prepare and publish offline AEC triplet SCPs")
    p.add_argument("--mic_dir", required=True, help="麦克风信号 wav 目录")
    p.add_argument("--farend_dir", required=True, help="远端参考 wav 目录")
    p.add_argument("--nearend_dir", required=True, help="近端干净语音（监督目标）wav 目录")
    p.add_argument("--error_dir", default=None, help="可选 LAEC 时域残差 wav 目录")
    p.add_argument("--output-dir", required=True, help="正式发布目录")
    p.add_argument("--val_ratio", type=float, default=0.0, help="验证集比例；0 表示仅 train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-unmatched", action="store_true", help="显式丢弃未配对 uid 并记录原因")
    p.parse_args(argv)
    print(
        "ERROR: GenerationManifest v1 publication 已下线；schema v2 reload 当前仅支持 SE；"
        "AEC offline publication 需要迁移后重新生成。",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
