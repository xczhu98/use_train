#!/usr/bin/env python3
"""Generate all indexes required for offline training mode.

Input: clean and noisy audio directories (filenames must match 1:1)
Output: strict four-column ``uid fs length path`` SCPs (train + validation)

Can be used directly in configs/*_offline.yaml for data.train / data.validation.

Usage:
    # Case 1: Separate validation directories exist
    python tools/prepare_data_tools/scp/prepare_offline.py \\
        --clean_dir     data/train/clean \\
        --noisy_dir     data/train/noisy \\
        --val_clean_dir data/val/clean \\
        --val_noisy_dir data/val/noisy \\
        --output-dir    data/offline

    # Case 2: Single dataset, auto-split by ratio (default 95:5)
    python tools/prepare_data_tools/scp/prepare_offline.py \\
        --clean_dir data/clean \\
        --noisy_dir data/noisy \\
        --output-dir data/offline \\
        --val_ratio 0.05

    # Optional: validate expected sample rate
    python tools/prepare_data_tools/scp/prepare_offline.py \\
        --clean_dir data/clean \\
        --noisy_dir data/noisy \\
        --output-dir data/offline \\
        --sample_rate 16000

Output directory structure:
    data/offline/
    ├── train/
    │   ├── spk1.scp  # clean: uid fs length path
    │   └── wav.scp   # noisy: uid fs length path
    └── validation/
        ├── spk1.scp
        └── wav.scp

    The script only publishes this canonical SCP layout. It never deletes retired
    artifacts; if they are present, publication fails before writing and asks the
    operator to move or delete them manually.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import uuid
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core.utils.file_transaction import write_text_files_transactionally  # noqa: E402
from tools.prepare_data_tools._common import setup_logging  # noqa: E402
from tools.shared.file_scan import scan_audio_files  # noqa: E402
from tools.prepare_data_tools.offline_publication import (  # noqa: E402
    ExistingOwnedRoot,
    MultiRootPublicationTarget,
    OwnedPublicationPaths,
    publish_owned_offline_dataset_roots,
    write_ownership_marker,
)
from tools.prepare_data_tools.se_index_publication import (  # noqa: E402
    alias_relative_waveform_path,
    build_se_index_manifest,
    read_existing_generation_identity,
    write_manifest_and_root_markers,
)
from tools.shared.scp_io import (  # noqa: E402
    serialize_manifest_relative_path,
    serialize_scp_entries,
)

logger = setup_logging("prepare_offline")

GENERATOR_ID = "scp/prepare_offline.py"
GENERATOR_VERSION_HISTORY = ("1.0", "2.0", "2.1")
GENERATOR_VERSION = GENERATOR_VERSION_HISTORY[-1]


def _parse_val_ratio(s: str) -> float:
    """解析 --val_ratio，必须在 (0, 1) 区间内，0/负数/>=1 直接报错。"""
    val = float(s)
    if not 0 < val < 1:
        raise argparse.ArgumentTypeError(f"--val_ratio must be in (0, 1), got {val}")
    return val


def scan_dir(d: Path) -> dict[str, Path]:
    """扫描目录返回 {stem: path}；重复 stem 会让 SCP 出现重复 uid，直接 fail-fast。"""
    files = scan_audio_files(d, recursive=True)
    result: dict[str, Path] = {}
    for p in files:
        previous = result.get(p.stem)
        if previous is not None:
            raise ValueError(
                f"duplicate uid {p.stem!r} under {d}: {previous} and {p}; "
                "请先重命名冲突文件"
            )
        result[p.stem] = p
    return result


def match_pairs(
    clean_files: dict, noisy_files: dict, *, allow_unmatched: bool = False
) -> list[str]:
    """按文件名配对；默认对任意 unmatched 输入 fail-closed。"""
    common = sorted(set(clean_files) & set(noisy_files))
    clean_only = sorted(set(clean_files) - set(noisy_files))
    noisy_only = sorted(set(noisy_files) - set(clean_files))
    if (clean_only or noisy_only) and not allow_unmatched:
        raise ValueError(
            "clean/noisy files must match 1:1: "
            f"clean_only={clean_only[:5]}, noisy_only={noisy_only[:5]}"
        )
    if clean_only:
        logger.warning("  %d clean files have no noisy match, skipped", len(clean_only))
    if noisy_only:
        logger.warning("  %d noisy files have no clean match, skipped", len(noisy_only))
    return common


def _four_col_entries(
    uids: list[str],
    files: dict[str, Path],
    expected_sample_rate: int | None,
    *,
    waveform_root: Path | None = None,
) -> list[tuple[str, int, int, str]]:
    """探测每条轨 metadata；verified 模式编码 alias-relative waveform 路径。"""
    entries = []
    for uid in uids:
        path = files[uid]
        info = sf.info(path)
        sr, frames = int(info.samplerate), int(info.frames)
        if sr <= 0 or frames <= 0:
            raise ValueError(f"invalid audio metadata for {path}: fs={sr}, length={frames}")
        if expected_sample_rate is not None and sr != expected_sample_rate:
            raise ValueError(
                f"sample rate mismatch for {path}: expected={expected_sample_rate}, actual={sr}"
            )
        serialized_path = (
            alias_relative_waveform_path(waveform_root, path)
            if waveform_root is not None
            else str(path)
        )
        entries.append((uid, sr, frames, serialized_path))
    return entries


def plan_split_outputs(
    uids: list[str],
    clean_files: dict,
    noisy_files: dict,
    out_dir: Path,
    sample_rate: int | None = None,
    *,
    waveform_root: Path | None = None,
) -> list[tuple[Path, str]]:
    """构造一个 split 的全部文本；普通模式使用 manifest-relative 路径。"""
    clean_path = out_dir / "spk1.scp"
    noisy_path = out_dir / "wav.scp"
    clean_four_col = _four_col_entries(
        uids, clean_files, sample_rate, waveform_root=waveform_root
    )
    noisy_four_col = _four_col_entries(
        uids, noisy_files, sample_rate, waveform_root=waveform_root
    )
    if waveform_root is None:
        clean_four_col = [
            (uid, sr, frames, serialize_manifest_relative_path(path, clean_path))
            for uid, sr, frames, path in clean_four_col
        ]
        noisy_four_col = [
            (uid, sr, frames, serialize_manifest_relative_path(path, noisy_path))
            for uid, sr, frames, path in noisy_four_col
        ]

    return [
        (clean_path, serialize_scp_entries(clean_four_col, four_col=True)),
        (noisy_path, serialize_scp_entries(noisy_four_col, four_col=True)),
    ]


def _check_publish_conflicts(out_dir: Path) -> None:
    """发布前拒绝退役 offline schema artifact；不自动删除任何既有文件。"""
    retired = [
        split / name
        for split in (out_dir / "train", out_dir / "validation")
        for name in ("utt2fs", "speech_length.scp", "clean.list", "noisy.list")
    ]
    retired.extend((out_dir / "clean.scp", out_dir / "noisy.scp"))
    conflicts = [path for path in retired if path.exists()]
    if conflicts:
        formatted = ", ".join(str(path) for path in conflicts)
        raise ValueError(
            "检测到退役 offline artifact："
            f"{formatted}；脚本不会自动删除，请手工移走或删除后重试"
        )


def publish_outputs(outputs: list[tuple[Path, str]]) -> None:
    """一次发布完整 offline file-set；失败时恢复所有旧目标。"""
    write_text_files_transactionally(
        [
            (path, lambda handle, payload=payload: handle.write(payload))
            for path, payload in outputs
        ],
        overwrite=True,
    )


def _publish_verified_indexes(
    *,
    out_dir: Path,
    waveform_root: Path,
    train_uids: list[str],
    train_clean: dict[str, Path],
    train_noisy: dict[str, Path],
    val_uids: list[str],
    val_clean: dict[str, Path],
    val_noisy: dict[str, Path],
    sample_rate: int | None,
    seed: int,
) -> None:
    """索引已有 verified waveform root，并只联合发布新的 index root。"""
    waveform_root = waveform_root.resolve(strict=True)
    owner_id, generation_id = read_existing_generation_identity(waveform_root)
    out_dir = Path(os.path.abspath(os.fspath(out_dir)))
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=out_dir.parent, prefix=f".{out_dir.name}.owned-staging-")
    )
    write_ownership_marker(staging, owner_id=owner_id)

    outputs = plan_split_outputs(
        train_uids,
        train_clean,
        train_noisy,
        staging / "train",
        sample_rate,
        waveform_root=waveform_root,
    )
    outputs.extend(
        plan_split_outputs(
            val_uids,
            val_clean,
            val_noisy,
            staging / "validation",
            sample_rate,
            waveform_root=waveform_root,
        )
    )
    publish_outputs(outputs)
    waveform_paths = [
        files[uid]
        for uids, files in (
            (train_uids, train_clean),
            (train_uids, train_noisy),
            (val_uids, val_clean),
            (val_uids, val_noisy),
        )
        for uid in uids
    ]
    manifest = build_se_index_manifest(
        index_root=staging,
        waveform_root=waveform_root,
        generation_id=generation_id,
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        split_scps={
            "train": ("train/spk1.scp", "train/wav.scp"),
            "validation": (
                "validation/spk1.scp",
                "validation/wav.scp",
            ),
        },
        waveform_paths=waveform_paths,
        parameters={"seed": seed, "sample_rate": sample_rate},
    )
    write_manifest_and_root_markers(
        index_root=staging,
        waveform_root=waveform_root,
        manifest=manifest,
        owner_id=owner_id,
        mark_waveform_root=False,
    )
    paths = OwnedPublicationPaths(
        allowed_parent=out_dir.parent,
        staging=staging,
        target=out_dir,
        backup=out_dir.parent / f".{out_dir.name}.backup-{uuid.uuid4().hex}",
        owner_id=owner_id,
    )
    publish_owned_offline_dataset_roots(
        (MultiRootPublicationTarget(alias="indexes", paths=paths),),
        manifest_root_alias="indexes",
        existing_roots=(
            ExistingOwnedRoot(
                alias="waveforms",
                allowed_parent=waveform_root.parent,
                root=waveform_root,
                owner_id=owner_id,
            ),
        ),
    )


def print_yaml_snippet(out_dir: Path, sample_rate: int | None):
    """Print the canonical SCP config snippet."""
    train_dir = out_dir / "train"
    val_dir = out_dir / "validation"
    sample_rate_line = f"  sample_rate: {sample_rate}\n" if sample_rate else ""

    print("\n" + "=" * 60)
    print("Paste the following into configs/*_offline.yaml data section:")
    print("=" * 60)
    print(f"""
data:
  mode: offline
{sample_rate_line}  train:
    clean_scp: {train_dir / 'spk1.scp'}
    noisy_scp: {train_dir / 'wav.scp'}
    max_duration: 160000
  validation:
    clean_scp: {val_dir / 'spk1.scp'}
    noisy_scp: {val_dir / 'wav.scp'}""")
    print("=" * 60 + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate strict four-column SCP files for offline training mode",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--clean_dir", required=True, help="Clean audio directory (train)")
    p.add_argument("--noisy_dir", required=True, help="Noisy audio directory (train)")
    p.add_argument("--val_clean_dir", default=None,
                   help="Validation clean dir; if not set, split from train by --val_ratio")
    p.add_argument("--val_noisy_dir", default=None,
                   help="Validation noisy dir; if not set, split from train by --val_ratio")
    p.add_argument("--output-dir", required=True,
                   help="Output root (creates train/ and validation/ subdirs)")
    p.add_argument("--val_ratio", type=_parse_val_ratio, default=0.05,
                   help="Validation ratio in (0, 1), default 0.05 (only when --val_clean_dir not set)")
    p.add_argument("--sample_rate", type=int, default=None,
                   help="Expected sample rate; all clean/noisy files are still probed")
    p.add_argument("--seed", type=int, default=42, help="Random seed (for splitting)")
    p.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Legacy opt-in: skip clean/noisy filenames without a peer",
    )
    p.add_argument(
        "--verified-publication",
        action="store_true",
        help="Publish schema v2 indexes only after full OfflineDataset reload",
    )
    p.add_argument(
        "--waveform-root",
        type=Path,
        help="Existing owned waveform generation root for --verified-publication",
    )
    args = p.parse_args(argv)

    if bool(args.val_clean_dir) != bool(args.val_noisy_dir):
        p.error("--val_clean_dir and --val_noisy_dir must be both set or both unset")
    if args.sample_rate is not None and args.sample_rate <= 0:
        p.error("--sample_rate must be positive")
    if args.verified_publication != (args.waveform_root is not None):
        p.error("--verified-publication and --waveform-root must be used together")

    out_dir = Path(args.output_dir)

    # ---- Scan train set ----
    logger.info(f"Scanning clean dir: {args.clean_dir}")
    clean_files = scan_dir(Path(args.clean_dir))
    logger.info(f"  Found {len(clean_files)} audio files")

    logger.info(f"Scanning noisy dir: {args.noisy_dir}")
    noisy_files = scan_dir(Path(args.noisy_dir))
    logger.info(f"  Found {len(noisy_files)} audio files")

    try:
        all_uids = match_pairs(
            clean_files, noisy_files, allow_unmatched=args.allow_unmatched
        )
    except ValueError as exc:
        logger.error("Cannot pair train files: %s", exc)
        return 1
    if not all_uids:
        logger.error("No matching clean/noisy file pairs! Check filenames match.")
        return 1
    logger.info(f"Matched: {len(all_uids)} pairs")

    train_out = out_dir / "train"
    val_out = out_dir / "validation"

    # ---- Split or read validation ----
    if args.val_clean_dir and args.val_noisy_dir:
        logger.info(f"Using separate validation directories")
        val_clean = scan_dir(Path(args.val_clean_dir))
        val_noisy = scan_dir(Path(args.val_noisy_dir))
        try:
            val_uids = match_pairs(
                val_clean, val_noisy, allow_unmatched=args.allow_unmatched
            )
        except ValueError as exc:
            logger.error("Cannot pair validation files: %s", exc)
            return 1
        if not val_uids:
            logger.error("No matching validation clean/noisy file pairs!")
            return 1
        train_uids = all_uids
        train_clean, train_noisy = clean_files, noisy_files

        logger.info(f"Train: {len(train_uids)} entries, Val: {len(val_uids)} entries")
    else:
        rng = random.Random(args.seed)
        shuffled = list(all_uids)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * args.val_ratio))
        if n_val >= len(shuffled):
            logger.error(
                f"val_ratio={args.val_ratio} leaves no samples for training "
                f"(total={len(shuffled)}).  Reduce --val_ratio."
            )
            return 1
        val_uids = sorted(shuffled[:n_val])
        train_uids = sorted(shuffled[n_val:])
        train_clean, train_noisy = clean_files, noisy_files
        val_clean, val_noisy = clean_files, noisy_files

        logger.info(f"Random split (seed={args.seed}, val_ratio={args.val_ratio}): "
                    f"Train {len(train_uids)} entries, Val {len(val_uids)} entries")

    # ---- Write files ----
    # 分离目录可能有重名 uid，因此 train/validation 分别使用各自字典；但文件集合
    # 必须全部序列化成功后一次事务发布，避免新旧 split 清单混合。
    logger.info("Planning train and validation file-set...")
    try:
        if args.verified_publication:
            _publish_verified_indexes(
                out_dir=out_dir,
                waveform_root=args.waveform_root,
                train_uids=train_uids,
                train_clean=train_clean,
                train_noisy=train_noisy,
                val_uids=val_uids,
                val_clean=val_clean,
                val_noisy=val_noisy,
                sample_rate=args.sample_rate,
                seed=args.seed,
            )
        else:
            _check_publish_conflicts(out_dir)
            outputs = plan_split_outputs(
                train_uids,
                train_clean,
                train_noisy,
                train_out,
                args.sample_rate,
            )
            outputs.extend(
                plan_split_outputs(
                    val_uids,
                    val_clean,
                    val_noisy,
                    val_out,
                    args.sample_rate,
                )
            )
            publish_outputs(outputs)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Cannot publish offline indexes transactionally: %s", exc)
        return 1
    logger.info("  -> %s (%d entries)", train_out, len(train_uids))
    logger.info("  -> %s (%d entries)", val_out, len(val_uids))

    # ---- Print YAML config snippet ----
    print_yaml_snippet(out_dir, args.sample_rate)

    logger.info("All done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
