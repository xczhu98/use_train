#!/usr/bin/env python3
"""DNS Challenge 3 offline dataset 生成器的 canonical SE 入口。

本模块独占 kebab-case CLI、参数校验、锁、owned staging 与整集合事务发布。
``--scan-and-build-csv`` 只更新原始数据目录中的 source lists，不进入输出集合发布。

用法::

    python -m tools.prepare_data_tools.generators.se.generate_dns3_offline_dataset \
        --data-root /data/dns_challenge3 \
        --output-dir /data/dns_challenge3_generated \
        --scp-dir data/se/dns_challenge3 \
        --num-workers 8
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import tempfile
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

from tools.prepare_data_tools.offline_publication import (
    ExistingOwnedRoot,
    MultiRootPublicationTarget,
    OwnedPublicationPaths,
    publish_owned_offline_dataset_roots,
    write_ownership_marker,
)
from tools.prepare_data_tools.se_index_publication import (
    WAVEFORM_ROOT_ALIAS,
    build_se_index_manifest,
    read_existing_generation_identity,
    write_manifest_and_root_markers,
)

from . import _dns3_dataset_generation as generation
from ._dns3_dataset_generation import (
    DNS3GenerationConfig,
    FS,
    NUM_TEST,
    NUM_TRAIN,
    NUM_VAL,
    SEED,
    WAV_LEN_SEC,
)

GENERATOR_ID = "tasks/se/generate_dns3_offline_dataset.py"
# 版本历史：
# - 2.0：SCP 使用严格四列 metadata。
# - 2.1：waveform/SCP 共享 schema v2 generation identity 与联合发布门禁。
GENERATOR_VERSION_HISTORY = ("2.0", "2.1")
GENERATOR_VERSION = GENERATOR_VERSION_HISTORY[-1]
PUBLICATION_OWNER_ID = "tasks/se/generate_dns3_offline_dataset"


def _absolute_path(value: Path) -> Path:
    return Path(os.path.abspath(os.fspath(value.expanduser())))


@contextmanager
def _output_lock(target: Path) -> Iterator[None]:
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / f".{target.name}.generate_dns3_offline_dataset.lock"
    descriptor = os.open(
        lock, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _validate_generated_splits(root: Path, config: DNS3GenerationConfig) -> None:
    for split, count in (
        ("train", config.num_train),
        ("val", config.num_val),
        ("test", config.num_test),
    ):
        if count <= 0:
            continue
        clean = {path.stem for path in (root / f"{split}_clean").glob("*.wav")}
        noisy = {path.stem for path in (root / f"{split}_noisy").glob("*.wav")}
        if len(clean) != count or clean != noisy:
            raise RuntimeError(
                f"{split} split incomplete: expected={count}, clean={len(clean)}, "
                f"noisy={len(noisy)}, paired={len(clean & noisy)}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DNS Challenge 3 offline dataset generation")
    parser.add_argument("--data-root", type=Path, required=True, help="DNS3 raw data root directory")
    parser.add_argument("--output-dir", type=Path, help="Generated waveform output directory")
    parser.add_argument("--scp-dir", type=Path, help="SCP output directory for use_train")
    parser.add_argument("--num-train", type=int, default=NUM_TRAIN)
    parser.add_argument("--num-val", type=int, default=NUM_VAL)
    parser.add_argument("--num-test", type=int, default=NUM_TEST)
    parser.add_argument("--fs", type=int, default=FS, help="Sample rate")
    parser.add_argument("--wav-len", type=int, default=WAV_LEN_SEC, help="Audio duration in seconds")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--scan-and-build-csv",
        action="store_true",
        help="Only scan source data and write clean/noise/RIR CSV lists",
    )
    parser.add_argument(
        "--scp-only",
        action="store_true",
        help="Only create SCP files from an existing generated waveform directory",
    )
    parser.add_argument("--write-info-csv", action="store_true")
    args = parser.parse_args(argv)
    if args.scan_and_build_csv and args.scp_only:
        parser.error("--scan-and-build-csv and --scp-only are mutually exclusive")
    if not args.scan_and_build_csv and args.output_dir is None:
        parser.error("--output-dir is required unless --scan-and-build-csv is used")
    if args.fs <= 0 or args.wav_len <= 0 or args.num_workers <= 0:
        parser.error("--fs, --wav-len and --num-workers must be positive")
    return args


def _generation_config(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    scp_dir: Path | None,
) -> DNS3GenerationConfig:
    return DNS3GenerationConfig(
        data_root=_absolute_path(args.data_root),
        output_dir=output_dir,
        scp_dir=scp_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        fs=args.fs,
        wav_len=args.wav_len,
        num_workers=args.num_workers,
        seed=args.seed,
        write_info_csv=args.write_info_csv,
    )


def _publication_layout(
    waveform_root: Path, index_root: Path
) -> tuple[dict[str, tuple[str, str]], list[Path]]:
    split_scps: dict[str, tuple[str, str]] = {}
    waveforms: list[Path] = []
    for split in ("train", "val", "test"):
        clean_dir = waveform_root / f"{split}_clean"
        noisy_dir = waveform_root / f"{split}_noisy"
        if not clean_dir.exists() and not noisy_dir.exists():
            continue
        clean_scp = index_root / split / "clean.scp"
        noisy_scp = index_root / split / "noisy.scp"
        if not clean_scp.is_file() or not noisy_scp.is_file():
            raise RuntimeError(f"{split} SCP output is incomplete")
        split_scps[split] = (f"{split}/clean.scp", f"{split}/noisy.scp")
        waveforms.extend(sorted(clean_dir.glob("*.wav")))
        waveforms.extend(sorted(noisy_dir.glob("*.wav")))
    if not split_scps:
        raise RuntimeError("DNS3 publication contains no generated split")
    return split_scps, waveforms


def _publication_paths(target: Path, staging: Path, owner_id: str) -> OwnedPublicationPaths:
    return OwnedPublicationPaths(
        allowed_parent=target.parent,
        staging=staging,
        target=target,
        backup=target.parent / f".{target.name}.backup-{uuid.uuid4().hex}",
        owner_id=owner_id,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.scan_and_build_csv:
        generation.build_csv_lists(str(_absolute_path(args.data_root)))
        return 0

    output = _absolute_path(args.output_dir)
    scp_target = args.scp_dir
    if scp_target is None:
        if args.scp_only:
            scp_target = Path("data/se/dns_challenge3")
        else:
            print(
                "[ERROR] --scp-dir is required for verified DNS3 generation",
                file=sys.stderr,
            )
            return 1
    scp_target = _absolute_path(scp_target)
    if scp_target == output:
        raise ValueError("--output-dir and --scp-dir must be different")
    if output in scp_target.parents or scp_target in output.parents:
        raise ValueError("--output-dir and --scp-dir must not be nested")

    output.parent.mkdir(parents=True, exist_ok=True)
    scp_target.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        for target in sorted({output, scp_target}, key=lambda path: str(path)):
            locks.enter_context(_output_lock(target))
        staged_scp: Path | None = None
        staged_output: Path | None = None
        try:
            if args.scp_only:
                owner_id, generation_id = read_existing_generation_identity(output)
                staged_output = output
            else:
                owner_id = PUBLICATION_OWNER_ID
                generation_id = f"dns3-{uuid.uuid4().hex}"
                staged_output = Path(
                    tempfile.mkdtemp(
                        dir=output.parent, prefix=f".{output.name}.owned-staging-"
                    )
                )
                write_ownership_marker(staged_output, owner_id=owner_id)

            staged_scp = Path(
                tempfile.mkdtemp(
                    dir=scp_target.parent, prefix=f".{scp_target.name}.owned-staging-"
                )
            )
            write_ownership_marker(staged_scp, owner_id=owner_id)
            config = _generation_config(
                args,
                output_dir=staged_output,
                scp_dir=staged_scp if args.scp_only else None,
            )
            if args.scp_only:
                generation.generate_scp(
                    str(staged_output),
                    str(staged_scp),
                    config.fs,
                    waveform_root_alias=WAVEFORM_ROOT_ALIAS,
                )
            else:
                generation.run_generation(config)
                _validate_generated_splits(staged_output, config)
                generation.generate_scp(
                    str(staged_output),
                    str(staged_scp),
                    config.fs,
                    waveform_root_alias=WAVEFORM_ROOT_ALIAS,
                )

            split_scps, waveform_paths = _publication_layout(
                staged_output, staged_scp
            )
            manifest = build_se_index_manifest(
                index_root=staged_scp,
                waveform_root=staged_output,
                generation_id=generation_id,
                generator_id=GENERATOR_ID,
                generator_version=GENERATOR_VERSION,
                split_scps=split_scps,
                waveform_paths=waveform_paths,
                parameters={"fs": config.fs},
            )
            write_manifest_and_root_markers(
                index_root=staged_scp,
                waveform_root=staged_output,
                manifest=manifest,
                owner_id=owner_id,
                mark_waveform_root=not args.scp_only,
            )

            index_target = MultiRootPublicationTarget(
                alias="indexes",
                paths=_publication_paths(scp_target, staged_scp, owner_id),
            )
            if args.scp_only:
                publish_owned_offline_dataset_roots(
                    (index_target,),
                    manifest_root_alias="indexes",
                    existing_roots=(
                        ExistingOwnedRoot(
                            alias="waveforms",
                            allowed_parent=output.parent,
                            root=output,
                            owner_id=owner_id,
                        ),
                    ),
                )
            else:
                waveform_target = MultiRootPublicationTarget(
                    alias="waveforms",
                    paths=_publication_paths(output, staged_output, owner_id),
                )
                publish_owned_offline_dataset_roots(
                    (index_target, waveform_target),
                    manifest_root_alias="indexes",
                )
            return 0
        except BaseException as exc:
            print(
                f"[ERROR] DNS3 generation aborted without publication: {exc}",
                file=sys.stderr,
            )
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
