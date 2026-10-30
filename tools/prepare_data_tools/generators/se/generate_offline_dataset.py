#!/usr/bin/env python3
"""将 SE dynamic simulation 配置物化为离线 clean/noisy 数据集。

使用与训练时相同的仿真管线（Layer A/B/C、SNR、混响等），预先生成若干条样本并写入磁盘，
后续可用 offline 模式训练，避免动态混合的 CPU 开销。

Usage:
    # 从 data.train 配置生成 10000 条
    python -m tools.prepare_data_tools.generators.se.generate_offline_dataset \
        -C path/to/se_dynamic.yaml \
        --output-dir data/offline_train --num-samples 10000 --num-workers 8

    # 生成并保存 10 条立体声 (clean|noisy) 便于试听
    python -m tools.prepare_data_tools.generators.se.generate_offline_dataset \
        -C path/to/se_dynamic.yaml \
        --output-dir data/offline_train --num-samples 100 --save-debug 10

要求:
    - 传入的 YAML 必须 data.mode: dynamic，且 data.train 段（含 SCP 与 simulation_cfg）已配置。
    - 仿真逻辑与动态训练完全一致（同一 build_dataset），仅将采样结果写为 wav + SCP。

Output:
    output_dir/
        clean/              <uid>.wav
        noisy/              <uid>.wav
        clean.scp           uid fs length path（供 offline 配置中 clean_scp 使用）
        noisy.scp           uid fs length path（供 offline 配置中 noisy_scp 使用）
        reload_*.scp        generation-root 相对路径（仅供正式 reload gate）
        generation_manifest.json     原 SE 业务生成统计合同
        generation_publication_manifest.json  统一 GenerationManifest v1 envelope
        generation_gate_receipt.json final-path full reload 发布回执

发布语义:
    - 非 dry-run 在 output_dir 同级 owned staging 中生成完整集合，并用同级锁串行发布。
    - staging 先做统一 manifest static gate，atomic replace 后在最终路径以生产
      ``core.data.offline_dataset.OfflineDataset(strict_load=True)`` full reload；成功才写 receipt。
    - 任一样本或 gate 失败都不发布；已有成功 output_dir 原样恢复，不留下部分 target。
    - 成功重跑整目录替换，因此旧 WAV/debug 文件不会残留。
    - UID 按成功样本从 utt_00000000 连续编号。
    - canonical 参数使用 kebab-case，同时兼容旧 underscore aliases。

在 offline 配置中使用:
    data:
      mode: offline
      train:
        clean_scp: data/offline_train/clean.scp
        noisy_scp: data/offline_train/noisy.scp
        ...
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import stat
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

import fcntl
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from tools.prepare_data_tools._common import (  # noqa: E402
    ensure_project_root,
    setup_logging,
)
from tools.shared.scp_io import write_scp_entries  # noqa: E402
from tools.prepare_data_tools.generation_manifest import (  # noqa: E402
    ADAPTER_SE,
    GenerationManifest,
    sha256_file,
    write_generation_root_marker,
)
from tools.prepare_data_tools.offline_publication import (  # noqa: E402
    OWNER_FILE as PUBLICATION_OWNER_FILE,
    RECEIPT_FILE,
    MultiRootPublicationTarget,
    OwnedPublicationPaths,
    PublicationCommittedError,
    publish_owned_offline_dataset_roots,
    write_ownership_marker,
)

ensure_project_root()
logger = setup_logging("generate_offline")


GENERATOR_ID = "tasks/se/generate_offline_dataset.py"
# 2.12: 动态数据集离线 RNG 改为 seed/split/epoch/index/stream 派生，不再依赖
# DataLoader worker 生命周期，逐样本增强序列因此改变。
GENERATOR_VERSION_HISTORY = ("2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "2.10", "2.11", "2.12")
GENERATOR_VERSION = GENERATOR_VERSION_HISTORY[-1]
MANIFEST_NAME = "generation_manifest.json"
ENVELOPE_MANIFEST_NAME = "generation_publication_manifest.json"
_ROOT_ALIAS = "dataset"
RELOAD_CLEAN_SCP = "reload_clean.scp"
RELOAD_NOISY_SCP = "reload_noisy.scp"
OWNER_FILE = ".generate_offline_dataset.owner.json"
OWNER_SCHEMA = 1
PUBLICATION_OWNER_ID = GENERATOR_ID
_PUBLICATION_CONTROL_FILES = frozenset(
    {ENVELOPE_MANIFEST_NAME, PUBLICATION_OWNER_FILE, RECEIPT_FILE}
)


def _unpack_se_sample(sample: tuple | list) -> tuple:
    """只接受 canonical 4/7 元组并消费前四项；已移除的历史 5 元组格式显式拒绝。"""
    if not isinstance(sample, (tuple, list)) or len(sample) not in {4, 7}:
        size = len(sample) if isinstance(sample, (tuple, list)) else "non-tuple"
        raise ValueError(f"SE dataset sample must contain 4 or 7 items, got {size}")
    return tuple(sample[:4])


def _validate_publication_sample(
    sample: tuple | list,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """规范化 canonical SE sample，并保留其采样率与有效长度 metadata。"""
    clean_wav, noisy_wav, sample_rate, length = _unpack_se_sample(sample)
    clean_wav = np.asarray(
        clean_wav.detach().cpu().numpy() if hasattr(clean_wav, "detach") else clean_wav
    )
    noisy_wav = np.asarray(
        noisy_wav.detach().cpu().numpy() if hasattr(noisy_wav, "detach") else noisy_wav
    )
    if type(length) not in {int, np.int32, np.int64} or int(length) <= 0:
        raise ValueError(f"SE sample length must be a positive integer, got {length!r}")
    length = int(length)
    if clean_wav.ndim != 1 or noisy_wav.ndim != 1:
        raise ValueError("SE publication waveforms must be one-dimensional")
    if len(clean_wav) < length or len(noisy_wav) < length:
        raise ValueError(
            "SE sample length exceeds waveform payload: "
            f"length={length}, clean={len(clean_wav)}, noisy={len(noisy_wav)}"
        )
    clean_wav = clean_wav[:length]
    noisy_wav = noisy_wav[:length]
    if not np.isfinite(clean_wav).all() or not np.isfinite(noisy_wav).all():
        raise ValueError("SE publication waveforms must contain only finite values")
    if not np.any(clean_wav != 0.0) and not np.any(noisy_wav != 0.0):
        raise ValueError("SE sample is a silence placeholder")
    if type(sample_rate) not in {int, np.int32, np.int64} or int(sample_rate) <= 0:
        raise ValueError(f"SE sample rate must be a positive integer, got {sample_rate!r}")
    return clean_wav, noisy_wav, int(sample_rate), length


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="Materialize an SE dynamic simulation as an offline dataset"
    )
    p.add_argument("-C", "--config", required=True, help="YAML config (dynamic mode)")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--num-samples", dest="num_samples", type=int, default=-1,
                   help="Number of samples to generate (-1 = all)")
    p.add_argument("--num-workers", dest="num_workers", type=int, default=0,
                   help="Reserved; generation currently runs serially")
    p.add_argument("--save-debug", dest="save_debug", type=int, default=0,
                   help="Also save N stereo (clean|noisy) WAV files for listening")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="只打印计划，不写任何文件")
    args = p.parse_args(argv)
    # 0/-2 等取值曾静默退化为“全量生成”，与帮助文本声明的 -1 语义不符，显式拒绝。
    if args.num_samples == 0 or args.num_samples < -1:
        p.error(f"--num-samples 只接受 -1（全量）或正整数, got {args.num_samples}")
    return args


def _owner_payload() -> dict[str, object]:
    return {"generator": GENERATOR_ID, "schema_version": OWNER_SCHEMA}


def _write_owner(root: Path) -> None:
    (root / OWNER_FILE).write_text(
        json.dumps(_owner_payload(), ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _is_owned(root: Path) -> bool:
    try:
        if root.is_symlink() or not root.is_dir():
            return False
        owner_path = root / OWNER_FILE
        if owner_path.is_file() and not owner_path.is_symlink():
            return json.loads(owner_path.read_text(encoding="utf-8")) == _owner_payload()
        manifest_path = root / MANIFEST_NAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("generator_id") == GENERATOR_ID
    except (OSError, json.JSONDecodeError):
        return False


def _remove_owned_tree(path: Path) -> None:
    """逐项清理本生成器 owned tree，拒绝 symlink/hardlink/special node。"""
    if not _is_owned(path):
        raise RuntimeError(f"refusing to remove unowned directory: {path}")
    root_stat = path.lstat()
    root_identity = (root_stat.st_dev, root_stat.st_ino)
    entries: list[tuple[Path, os.stat_result]] = []
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        child_stat = child.lstat()
        if stat.S_ISLNK(child_stat.st_mode):
            raise RuntimeError(f"owned cleanup refuses symlink: {child}")
        if not stat.S_ISDIR(child_stat.st_mode) and not (
            stat.S_ISREG(child_stat.st_mode) and child_stat.st_nlink == 1
        ):
            raise RuntimeError(f"owned cleanup refuses hardlink/special node: {child}")
        entries.append((child, child_stat))
    for child, child_stat in entries:
        child.rmdir() if stat.S_ISDIR(child_stat.st_mode) else child.unlink()
    current = path.lstat()
    if (current.st_dev, current.st_ino) != root_identity:
        raise RuntimeError(f"owned cleanup root changed during removal: {path}")
    path.rmdir()


@contextmanager
def _output_lock(output_dir: Path):
    """用输出目录同级锁串行化 writer，锁文件不参与目录交换。"""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.generate.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _publish_staged_directory(staged: Path, target: Path) -> None:
    """经共享 static/full-reload/receipt 事务发布完整 owned 集合。"""
    if not _is_owned(staged):
        raise RuntimeError(f"staging directory is not owned: {staged}")
    if target.exists() or target.is_symlink():
        if not _is_owned(target):
            raise RuntimeError(f"refusing to replace unowned output directory: {target}")
    try:
        publish_owned_offline_dataset_roots(
            (
                MultiRootPublicationTarget(
                    alias=_ROOT_ALIAS,
                    paths=_publication_paths(staged, target),
                ),
            ),
            manifest_root_alias=_ROOT_ALIAS,
            manifest_path=ENVELOPE_MANIFEST_NAME,
        )
    except PublicationCommittedError as exc:
        # receipt 已提交；只遗留共享模块明确验证过的 owned backup。
        logger.warning(str(exc))


def _materialize_dataset(
    ds,
    *,
    staged: Path,
    published: Path,
    indices: list[int],
    sample_rate: int | None,
    save_debug: int,
    seed: int,
) -> dict[str, object]:
    staged.mkdir(parents=True, exist_ok=True)
    _write_owner(staged)
    clean_dir = staged / "clean"
    noisy_dir = staged / "noisy"
    clean_dir.mkdir()
    noisy_dir.mkdir(parents=True)
    debug_dir = staged / "debug_stereo"
    debug_slots = set(random.Random(seed).sample(
        range(len(indices)), min(save_debug, len(indices))
    )) if save_debug > 0 else set()
    if debug_slots:
        debug_dir.mkdir()

    clean_entries: list[tuple[str, int, int, str]] = []
    noisy_entries: list[tuple[str, int, int, str]] = []
    reload_clean_entries: list[tuple[str, int, int, str]] = []
    reload_noisy_entries: list[tuple[str, int, int, str]] = []
    failures: list[dict[str, object]] = []
    debug_count = 0
    for attempt, idx in enumerate(tqdm(indices, desc="Generating")):
        try:
            clean_wav, noisy_wav, sr, length = _validate_publication_sample(ds[idx])
            if sample_rate is not None and sr != int(sample_rate):
                raise ValueError(
                    "SE sample rate differs from configured sample rate: "
                    f"sample={sr}, configured={sample_rate}"
                )
            # UID 按成功样本连续编号；失败 attempt 不制造编号空洞。
            uid = f"utt_{len(clean_entries):08d}"
            staged_clean = clean_dir / f"{uid}.wav"
            staged_noisy = noisy_dir / f"{uid}.wav"
            sf.write(str(staged_clean), clean_wav, sr)
            sf.write(str(staged_noisy), noisy_wav, sr)
            final_clean = published / "clean" / f"{uid}.wav"
            final_noisy = published / "noisy" / f"{uid}.wav"
            clean_entries.append((uid, sr, length, str(final_clean)))
            noisy_entries.append((uid, sr, length, str(final_noisy)))
            reload_clean_entries.append((uid, sr, length, f"{_ROOT_ALIAS}:clean/{uid}.wav"))
            reload_noisy_entries.append((uid, sr, length, f"{_ROOT_ALIAS}:noisy/{uid}.wav"))
            if attempt in debug_slots:
                stereo = np.stack([clean_wav, noisy_wav], axis=-1)
                sf.write(str(debug_dir / f"{uid}_stereo.wav"), stereo, sr)
                debug_count += 1
        except Exception as exc:
            failures.append({"dataset_index": idx, "error": str(exc)})
            logger.warning(f"Failed sample {idx}: {exc}")

    counts = {
        "requested": len(indices),
        "success": len(clean_entries),
        "failed": len(failures),
    }
    if failures:
        raise RuntimeError(
            "offline generation failed; refusing to publish incomplete collection: "
            f"requested={counts['requested']}, success={counts['success']}, "
            f"failed={counts['failed']}"
        )

    # 公开 SCP 保持最终绝对路径合同；reload SCP 仅供 generation-root 内正式门禁。
    # 两者都直接复用生成时 metadata，禁止为 SCP 重新打开音频 payload。
    write_scp_entries(clean_entries, staged / "clean.scp", four_col=True)
    write_scp_entries(noisy_entries, staged / "noisy.scp", four_col=True)
    write_scp_entries(
        reload_clean_entries, staged / RELOAD_CLEAN_SCP, four_col=True
    )
    write_scp_entries(
        reload_noisy_entries, staged / RELOAD_NOISY_SCP, four_col=True
    )
    metadata = {
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "counts": counts,
        "uid_semantics": "successful_samples_contiguous_zero_based",
        "seed": seed,
        "debug_files": debug_count,
    }
    (staged / MANIFEST_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    envelope = _write_generation_manifest(staged, metadata)
    write_generation_root_marker(
        staged,
        manifest=envelope,
        root_alias=_ROOT_ALIAS,
        owner_id=PUBLICATION_OWNER_ID,
    )
    return metadata


def _generation_inventory(root: Path) -> list[dict[str, object]]:
    """绑定 staging 中全部 WAV、SCP 与业务 metadata。"""
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"GenerationManifest inventory refuses symlink: {path}")
        if not path.is_file() or path.name in _PUBLICATION_CONTROL_FILES:
            continue
        inventory.append(
            {
                "root_alias": _ROOT_ALIAS,
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "kind": "waveform" if path.suffix.lower() == ".wav" else "metadata",
            }
        )
    return inventory


def _write_generation_manifest(
    root: Path, metadata: dict[str, object]
) -> GenerationManifest:
    """写统一 SE offline envelope，并声明生产 OfflineDataset reload gate。"""
    counts = metadata["counts"]
    published = int(counts["success"])
    manifest = GenerationManifest.from_dict(
        {
            "schema_version": 2,
            "generation_id": f"se-offline-{uuid.uuid4().hex}",
            "status": "complete",
            "artifact_kind": "se-offline-dataset",
            "task": "se",
            "generator": {"id": GENERATOR_ID, "version": GENERATOR_VERSION},
            "root_bindings": [{"alias": _ROOT_ALIAS, "role": "manifest"}],
            "splits": ["train"],
            "parameters": {
                "seed": metadata["seed"],
                "debug_files": metadata["debug_files"],
                "uid_semantics": metadata["uid_semantics"],
                "public_clean_scp": "clean.scp",
                "public_noisy_scp": "noisy.scp",
            },
            "source_identities": [],
            "inventory": _generation_inventory(root),
            "counts": {
                "requested": int(counts["requested"]),
                "published": published,
                "failed": int(counts["failed"]),
                "dropped": 0,
            },
            "required_gates": [
                {
                    "name": "train-reload",
                    "adapter": ADAPTER_SE,
                    "split": "train",
                    "inputs": {
                        "clean_scp": {
                            "root_alias": _ROOT_ALIAS,
                            "path": RELOAD_CLEAN_SCP,
                        },
                        "noisy_scp": {
                            "root_alias": _ROOT_ALIAS,
                            "path": RELOAD_NOISY_SCP,
                        },
                    },
                    "options": {"strict_load": True},
                }
            ],
        }
    )
    (root / ENVELOPE_MANIFEST_NAME).write_text(
        manifest.to_json(), encoding="utf-8"
    )
    return manifest


def _publication_paths(staged: Path, target: Path) -> OwnedPublicationPaths:
    return OwnedPublicationPaths(
        allowed_parent=target.parent,
        staging=staged,
        target=target,
        backup=target.parent / f".{target.name}.backup-{uuid.uuid4().hex}",
        owner_id=PUBLICATION_OWNER_ID,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_workers > 0:
        logger.warning("--num-workers is reserved; generation currently runs serially")
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = copy.deepcopy(
        OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    )
    if cfg["data"].get("mode") != "dynamic":
        logger.error("Config must use mode: dynamic")
        return 1
    if "train" not in cfg["data"]:
        logger.error("Config data.train section is missing (SCP + simulation_cfg)")
        return 1

    # 发布专用配置副本强制拒绝运行时静音占位。
    cfg["data"]["strict_load"] = True

    from tasks.se.data import build_dataset
    ds = build_dataset(cfg["data"], mode="train")
    ds.set_epoch(0, base_seed=args.seed)
    total = min(len(ds), args.num_samples) if args.num_samples > 0 else len(ds)
    indices = list(range(total))
    random.shuffle(indices)
    output = Path(os.path.abspath(os.fspath(Path(args.output_dir).expanduser())))
    logger.info(f"Dataset size: {len(ds)}, generating {total} samples")

    if args.dry_run:
        logger.info(f"[dry-run] Would generate {total} samples into {output}")
        for count, idx in enumerate(indices):
            logger.info(f"[dry-run]  {count:04d}: ds[{idx}] -> {output / 'clean'}/utt_{count:08d}.wav")
        return 0

    with _output_lock(output):
        staged = Path(tempfile.mkdtemp(
            dir=output.parent,
            prefix=f".{output.name}.owned-staging-",
        ))
        write_ownership_marker(staged, owner_id=PUBLICATION_OWNER_ID)
        _write_owner(staged)
        try:
            manifest = _materialize_dataset(
                ds,
                staged=staged,
                published=output,
                indices=indices,
                sample_rate=cfg["data"].get("sample_rate"),
                save_debug=args.save_debug,
                seed=args.seed,
            )
            _publish_staged_directory(staged, output)
        except BaseException as exc:
            if staged.exists() or staged.is_symlink():
                try:
                    _remove_owned_tree(staged)
                except (OSError, RuntimeError) as cleanup_exc:
                    logger.error(f"Owned staging cleanup failed: {cleanup_exc}")
            logger.error(f"Generation aborted without publication: {exc}")
            return 1

    counts = manifest["counts"]
    logger.info(
        f"Done. Output: {output}; requested={counts['requested']}, "
        f"success={counts['success']}, failed={counts['failed']}"
    )
    logger.info(f"  four-column clean.scp / noisy.scp / {MANIFEST_NAME} created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
