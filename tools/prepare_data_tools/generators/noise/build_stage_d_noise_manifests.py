#!/usr/bin/env python3
"""从 Stage D per-dataset manifest 生成 strict noise JSONL（schema v6）。

输入为 ``stage_d/d_manifest_<dataset_id>.jsonl``，只导出 D2 passed 记录。
每条候选均委托 ``core.data.noise_admission``，使用固定 ``audioset_top5`` 与
canonical policy 重算 C3；生成器不信任已写 C3 结论，也不复制决策矩阵。

发布链 fail-closed：C1/C2/C3/D2 必须全部 passed，C3 只允许
``accept_noise`` 或 ``accept_babble``，strict schema/policy/evidence 版本必须分别为
6/4/5。任一候选非法会使整次生成失败，不会静默丢弃后发布部分清单。旧
schema、完整 527 持久化 evidence、C4/C/D stage 证据均不转换、不双写。

类别词表、``category_id``、roles 和发布摘要只基于 C3 evaluator 选出的
``balance_category``。类别身份同时由 ``balance_category_mid`` 与 canonical label
绑定；``balance_category_source`` 说明其来自 top-5 中最高排名的 noise MID，或
回退到通用 ``/m/096m7z`` Noise。babble MID 不参与决策或类别选择。词表仍在
生成过程中计算，``category_id``、``category_name`` 与 ``roles`` 直接内嵌每条 JSONL。

``--output-dir`` 模式只发布 train/val JSONL 与
``noise_publication_metadata.json`` 三件套。metadata 顶层 ``manifest_summary`` 记录
发布规模、role 与 train/val 划分，并与生成器版本、参数及两个 manifest 摘要共同
纳入集合 fingerprint；单文件模式使用 ``<output>.metadata.json``，摘要不含 split。

单文件模式的 JSONL + vocab + metadata 与 ``--output-dir`` 的三个文件均按发布
集合提交：全部 staging 文件成功 ``flush``/``fsync`` 后才替换目标；任一替换或
目录 ``fsync`` 失败会恢复完整旧集合，并发 writer 由共享事务锁串行化。未指定
``--overwrite`` 时，仅当 metadata、生成器版本、参数和全部内容 bytes 一致才复用。
使用 ``--overwrite`` 迁移旧发布时，三个普通遗留 sidecar 仅在新版发布成功后删除。

用法::

    # 生成单个全量 JSONL
    python -m tools.prepare_data_tools.generators.noise.build_stage_d_noise_manifests \
        --stage-d-root /data/noise_filter/stage_d \
        --output /data/noise_filter/stage_d/noise_sources.jsonl \
        --category-vocab /data/noise_filter/stage_d/noise_categories.json

    # 推荐调用：生成训练/验证 JSONL
    python -m tools.prepare_data_tools.generators.noise.build_stage_d_noise_manifests \
        --stage-d-root /data/noise_filter/stage_d \
        --output-dir /data/shared/noise/filter \
        --val-ratio 0.05 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import stat
import sys
from pathlib import Path
from typing import Iterator

# 脚本可直接运行，需先把项目根目录加入 sys.path 再导入 core。
_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.data.noise_admission import (
    NOISE_MANIFEST_SCHEMA_VERSION,
    NoiseAdmissionContext,
    validate_noise_manifest_records,
)
from core.data.noise_manifest_adapter import canonical_json_bytes
from core.data.noise_roles import compute_category_roles
from core.utils import file_transaction
from tools.noise_filter_pipeline.core.run_lock import stage_d_run_lock
from tools.noise_filter_pipeline.core.stage_d_noise_manifest_contract import (
    GENERATOR_VERSION,
    build_validated_noise_record,
    validate_exact_stage_d_audio_record,
)


_AUDIO_STATUS = "passed"
_PUBLICATION_METADATA_SCHEMA_VERSION = 3
_SPLIT_METADATA_FILENAME = "noise_publication_metadata.json"
_LEGACY_SIDECAR_FILENAMES = (
    "noise_pool_stats.json",
    "noise_categories.json",
    "noise_category_roles.json",
)


def _manifest_dataset_id(path: Path) -> str:
    prefix = "d_manifest_"
    suffix = ".jsonl"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError(f"invalid Stage D manifest name: {path}")
    dataset_id = path.name[len(prefix) : -len(suffix)]
    if not dataset_id:
        raise ValueError(f"empty dataset_id in manifest name: {path}")
    return dataset_id


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_file_snapshot(
    path: Path,
    *,
    include_bytes: bool,
    root: Path | None = None,
) -> tuple[bytes | None, tuple]:
    """在稳定 FD 链上读取 regular file；可选 root 模式拒绝任意祖先 symlink。"""
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    opened: list[tuple[int, int | None, str | None, tuple]] = []
    descriptor = -1
    try:
        if root is None:
            descriptor = os.open(path, file_flags)
        else:
            root = root.resolve(strict=True)
            relative = path.relative_to(root)
            directory_flags = file_flags | getattr(os, "O_DIRECTORY", 0)
            parent_fd = os.open(root, directory_flags)
            root_identity = _stat_identity(os.fstat(parent_fd))
            opened.append((parent_fd, None, None, root_identity))
            for component in relative.parts[:-1]:
                child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
                child_identity = _stat_identity(os.fstat(child_fd))
                opened.append((child_fd, parent_fd, component, child_identity))
                parent_fd = child_fd
            descriptor = os.open(relative.name, file_flags, dir_fd=parent_fd)
            opened.append(
                (descriptor, parent_fd, relative.name, _stat_identity(os.fstat(descriptor)))
            )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{path}: snapshot source must be a regular file")
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if include_bytes else None
        byte_count = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
        if root is None:
            path_after = os.stat(path, follow_symlinks=False)
            chain_identity: tuple = ()
        else:
            for opened_fd, parent_fd, name, expected in opened:
                if _stat_identity(os.fstat(opened_fd)) != expected:
                    raise ValueError(f"{path}: ancestor identity changed while reading snapshot")
                if parent_fd is not None:
                    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                    if _stat_identity(current) != expected:
                        raise ValueError(f"{path}: ancestor path changed while reading snapshot")
            path_after = os.fstat(descriptor)
            chain_identity = tuple(item[3] for item in opened[:-1])
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"{path}: file changed while reading stable snapshot: {exc}") from exc
    finally:
        if opened:
            for opened_fd, *_ in reversed(opened):
                os.close(opened_fd)
        elif descriptor >= 0:
            os.close(descriptor)
    identity = _stat_identity(before)
    if (
        identity != _stat_identity(after)
        or identity != _stat_identity(path_after)
        or byte_count != before.st_size
    ):
        raise ValueError(f"{path}: file changed while reading stable snapshot")
    return (
        b"".join(chunks) if chunks is not None else None,
        (digest.hexdigest(), *identity, chain_identity),
    )


def _stable_audio_identity(path: Path, *, root: Path) -> tuple:
    """仅用 stat 绑定音频 leaf 身份，并拒绝越界或 symlink 祖先。"""
    root_abs = Path(os.path.abspath(str(root)))
    path_abs = Path(os.path.abspath(str(path)))
    try:
        relative = path_abs.relative_to(root_abs)
    except ValueError as exc:
        raise ValueError(f"{path}: audio path escapes stage-d-root ({root_abs})") from exc
    if not relative.parts:
        raise ValueError(f"{path}: audio path must name a file under stage-d-root")

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    opened: list[tuple[int, int | None, str | None, tuple]] = []
    try:
        parent_fd = os.open(root_abs, directory_flags)
        root_identity = _stat_identity(os.fstat(parent_fd))
        if not stat.S_ISDIR(root_identity[2]):
            raise ValueError(f"{root_abs}: stage-d-root must be a directory")
        opened.append((parent_fd, None, None, root_identity))
        for component in relative.parts[:-1]:
            child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            child_identity = _stat_identity(os.fstat(child_fd))
            opened.append((child_fd, parent_fd, component, child_identity))
            parent_fd = child_fd

        leaf_name = relative.name
        before = os.stat(leaf_name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            raise ValueError(f"{path}: audio path must not be a symbolic link")
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{path}: audio path must be a regular file")
        leaf_identity = _stat_identity(before)

        for opened_fd, ancestor_fd, name, expected in opened:
            if _stat_identity(os.fstat(opened_fd)) != expected:
                raise ValueError(f"{path}: audio ancestor identity changed during stat")
            current = (
                os.stat(root_abs, follow_symlinks=False)
                if ancestor_fd is None
                else os.stat(name, dir_fd=ancestor_fd, follow_symlinks=False)
            )
            if _stat_identity(current) != expected:
                raise ValueError(f"{path}: audio ancestor path changed during stat")
        after = os.stat(leaf_name, dir_fd=parent_fd, follow_symlinks=False)
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(
            f"{path}: audio path and ancestors must be stable non-symlink entries: {exc}"
        ) from exc
    finally:
        for opened_fd, *_ in reversed(opened):
            os.close(opened_fd)

    if _stat_identity(after) != leaf_identity:
        raise ValueError(f"{path}: audio identity changed during stat")
    return (relative.as_posix(), leaf_identity, tuple(item[3] for item in opened))


def _iter_stable_manifest_lines(path: Path) -> Iterator[tuple[int, str]]:
    """在单一 FD 上逐行读取 UTF-8 JSONL，并增量绑定内容摘要与身份。"""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{path}: cannot open stable manifest: {exc}") from exc
    digest = hashlib.sha256()
    byte_count = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{path}: manifest must be a regular file")
        with os.fdopen(descriptor, "rb") as manifest_file:
            descriptor = -1
            for line_number, raw_line in enumerate(manifest_file, start=1):
                digest.update(raw_line)
                byte_count += len(raw_line)
                try:
                    yield line_number, raw_line.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"{path}: manifest is not valid UTF-8") from exc
            after = os.fstat(manifest_file.fileno())
        path_after = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{path}: manifest changed while streaming: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = _stat_identity(before)
    if (
        identity != _stat_identity(after)
        or identity != _stat_identity(path_after)
        or byte_count != before.st_size
    ):
        raise ValueError(f"{path}: manifest changed while streaming")
    return (digest.hexdigest(), *identity, ())


def _iter_records(
    stage_d_root: Path,
    manifest_snapshots: dict[Path, tuple],
) -> Iterator[tuple[Path, int, dict]]:
    manifests = sorted(stage_d_root.glob("d_manifest_*.jsonl"))
    if not manifests:
        raise ValueError(f"no d_manifest_*.jsonl files under {stage_d_root}")
    for manifest_path in manifests:
        expected_dataset_id = _manifest_dataset_id(manifest_path)
        lines = _iter_stable_manifest_lines(manifest_path)
        while True:
            try:
                line_number, raw_line = next(lines)
            except StopIteration as completed:
                manifest_snapshots[manifest_path] = completed.value
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{manifest_path}: line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            record = validate_exact_stage_d_audio_record(
                record, location=f"{manifest_path}: line {line_number}"
            )
            if record.get("stages", {}).get("D2", {}).get("status") != _AUDIO_STATUS:
                continue
            if record.get("dataset_id") != expected_dataset_id:
                raise ValueError(
                    f"{manifest_path}: line {line_number}: dataset_id does not match filename"
                )
            yield manifest_path, line_number, record


def _validated_output_record(
    manifest_path: Path,
    line_number: int,
    record: dict,
    stage_d_root: Path,
    context: NoiseAdmissionContext,
    audio_snapshots: dict[Path, tuple],
) -> dict:
    """校验单条 Stage D 记录并绑定其音频 stat-only 身份。"""
    location = f"{manifest_path}: line {line_number}"
    output_record = build_validated_noise_record(
        record,
        location=location,
        context=context,
    )
    audio_path = Path(output_record["path"])
    root_resolved = stage_d_root.resolve(strict=True)
    path_abs = Path(os.path.abspath(str(audio_path)))
    try:
        identity = _stable_audio_identity(path_abs, root=root_resolved)
    except ValueError as exc:
        raise ValueError(f"{location}: {exc}") from exc
    existing = audio_snapshots.setdefault(path_abs, identity)
    if existing != identity:
        raise ValueError(f"{location}: audio identity changed between records")
    return output_record


def _assert_input_snapshots(
    stage_d_root: Path,
    manifest_snapshots: dict[Path, tuple],
    audio_snapshots: dict[Path, tuple],
) -> None:
    """发布前复验 manifest 内容哈希与全部音频 stat-only 身份。"""
    current_manifests = sorted(stage_d_root.glob("d_manifest_*.jsonl"))
    if current_manifests != sorted(manifest_snapshots):
        raise ValueError(f"{stage_d_root}: manifest file set changed before publication")
    for path, expected in manifest_snapshots.items():
        _, actual = _stable_file_snapshot(path, include_bytes=False)
        if actual != expected:
            raise ValueError(f"{path}: manifest identity changed before publication")
    root_resolved = stage_d_root.resolve(strict=True)
    for path, expected in audio_snapshots.items():
        actual = _stable_audio_identity(path, root=root_resolved)
        if actual != expected:
            raise ValueError(f"{path}: audio identity changed before publication")


def _collect_records(
    stage_d_root: Path,
    *,
    context: NoiseAdmissionContext,
) -> tuple[list[dict], dict[str, int], dict[Path, tuple], dict[Path, tuple]]:
    """预检 UID、strict schema，并绑定 manifest/audio 稳定快照。"""
    records: list[dict] = []
    seen_ids: set[str] = set()
    categories: set[str] = set()
    manifest_snapshots: dict[Path, tuple] = {}
    audio_snapshots: dict[Path, tuple] = {}
    for manifest_path, line_number, record in _iter_records(
        stage_d_root, manifest_snapshots
    ):
        output_record = _validated_output_record(
            manifest_path,
            line_number,
            record,
            stage_d_root,
            context,
            audio_snapshots,
        )
        record_id = output_record["uid"]
        balance_category = output_record["balance_category"]
        if record_id in seen_ids:
            raise ValueError(
                f"{manifest_path}: line {line_number}: duplicate record_id: {record_id!r}"
            )
        seen_ids.add(record_id)
        categories.add(balance_category)
        output_record["roles"] = compute_category_roles(balance_category)
        records.append(output_record)
    if not records:
        raise ValueError(f"no D2 passed records under {stage_d_root}")
    category_vocab = {name: index for index, name in enumerate(sorted(categories))}
    for record in records:
        record["category_id"] = category_vocab[record["balance_category"]]
    validate_noise_manifest_records(
        {record["uid"]: record for record in records},
        source_path=str(stage_d_root / "noise_sources.jsonl"),
        context=context,
    )
    return records, category_vocab, manifest_snapshots, audio_snapshots


def _jsonl_bytes(records: list[dict]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _json_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _bytes_writer(payload: bytes):
    def writer(output_file) -> None:
        output_file.write(payload.decode("utf-8"))

    return writer


def _jsonl_writer(records: list[dict]):
    return _bytes_writer(_jsonl_bytes(records))


def _json_writer(value: object):
    return _bytes_writer(_json_bytes(value))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _publication_metadata(
    *,
    mode: str,
    parameters: dict[str, object],
    artifacts: list[tuple[str, Path, bytes]],
    manifest_summary: dict[str, object],
) -> dict[str, object]:
    """构建绑定生成器版本、参数、发布摘要及文件 bytes 的集合指纹。"""
    fingerprint_payload: dict[str, object] = {
        "metadata_schema_version": _PUBLICATION_METADATA_SCHEMA_VERSION,
        "generator": Path(__file__).name,
        "generator_version": GENERATOR_VERSION,
        "manifest_schema_version": NOISE_MANIFEST_SCHEMA_VERSION,
        "mode": mode,
        "parameters": parameters,
        "manifest_summary": manifest_summary,
        "artifacts": {
            role: {
                "filename": path.name,
                "sha256": _sha256(payload),
                "size": len(payload),
            }
            for role, path, payload in artifacts
        },
    }
    return {
        **fingerprint_payload,
        "publication_fingerprint": _sha256(canonical_json_bytes(fingerprint_payload)),
    }


def _publication_is_reusable(
    outputs: list[tuple[Path, bytes]],
    metadata_path: Path,
    expected_metadata: dict[str, object],
) -> bool:
    """仅当 sidecar 与全部发布文件 bytes 完整匹配时允许复用。"""
    paths = [path for path, _ in outputs] + [metadata_path]
    if not all(path.is_file() and not path.is_symlink() for path in paths):
        return False
    try:
        actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if actual_metadata != expected_metadata:
        return False
    return all(
        path.read_bytes() == expected_payload
        for path, expected_payload in outputs
    )


def _reject_incomplete_publication(paths: list[Path], *, overwrite: bool) -> None:
    """在昂贵校验前拒绝明显不完整的旧发布集合。"""
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing and len(existing) != len(paths):
        raise FileExistsError(
            "output publication is incomplete; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )


def _legacy_sidecar_identities(
    paths: list[Path], *, overwrite: bool
) -> dict[Path, tuple]:
    """发布前确认所有遗留 sidecar leaf 安全，并记录稳定身份。"""
    identities: dict[Path, tuple] = {}
    for path in paths:
        try:
            value = path.lstat()
        except FileNotFoundError:
            continue
        if not overwrite:
            raise FileExistsError(
                f"legacy publication exists; pass --overwrite to migrate: {path}"
            )
        if not stat.S_ISREG(value.st_mode):
            raise RuntimeError(f"legacy publication leaf must be a regular file: {path}")
        identities[path] = _stat_identity(value)
    return identities


def _remove_legacy_sidecars(identities: dict[Path, tuple]) -> None:
    """三件套提交后删除身份未变化的遗留 sidecar，并只固化目录一次。"""
    if not identities:
        return
    for path, expected_identity in identities.items():
        try:
            current = path.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"legacy publication leaf changed before cleanup: {path}") from exc
        if not stat.S_ISREG(current.st_mode) or _stat_identity(current) != expected_identity:
            raise RuntimeError(f"legacy publication leaf changed before cleanup: {path}")
    for path, expected_identity in identities.items():
        current = path.lstat()
        if not stat.S_ISREG(current.st_mode) or _stat_identity(current) != expected_identity:
            raise RuntimeError(f"legacy publication leaf changed before cleanup: {path}")
        path.unlink()
    if os.name != "nt":
        parent = next(iter(identities)).parent
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _publish_or_reuse(
    outputs: list[tuple[Path, bytes]],
    metadata_path: Path,
    metadata: dict[str, object],
    *,
    overwrite: bool,
) -> None:
    publication_paths = [path for path, _ in outputs] + [metadata_path]
    _check_output_paths(publication_paths, overwrite=True)
    if not overwrite:
        if _publication_is_reusable(outputs, metadata_path, metadata):
            return
        existing = [str(path) for path in publication_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "output publication is stale or incomplete; pass --overwrite to replace: "
                + ", ".join(existing)
            )
    file_transaction.write_text_files_transactionally(
        [
            *((path, _bytes_writer(payload)) for path, payload in outputs),
            (metadata_path, _json_writer(metadata)),
        ],
        overwrite=overwrite,
    )


def _compute_stats(
    records: list[dict], vocabulary: dict[str, int]
) -> dict[str, object]:
    """统计 manifest 发布规模与按 SE/AEC/TSE role 划分的类别数。"""
    total_records = len(records)
    total_categories = len(vocabulary)
    wind_categories = set()
    music_categories = set()
    for name in vocabulary:
        roles = compute_category_roles(name)
        if "wind" in roles:
            wind_categories.add(name)
        if "music" in roles:
            music_categories.add(name)
    wind = len(wind_categories)
    music = len(music_categories)
    return {
        "total_categories": total_categories,
        "total_records": total_records,
        "ordinary_noise_se_aec": total_categories - wind,
        "ordinary_noise_tse": total_categories - wind - music,
        "wind": wind,
        "music": music,
    }


def _format_stats_table(stats: dict[str, object]) -> str:
    return (
        "| 任务 | 普通噪声 | 风噪 | 音乐 | 合计 |\n"
        "|------|----------|------|------|------|\n"
        f"| SE / AEC | {stats['ordinary_noise_se_aec']} | {stats['wind']} | — | {stats['total_categories']} |\n"
        f"| TSE | {stats['ordinary_noise_tse']} | {stats['wind']} | {stats['music']} | {stats['total_categories']} |"
    )


def _check_output_paths(paths: list[Path], *, overwrite: bool) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("output files must be different")
    if not overwrite:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                "output already exists; pass --overwrite to replace: " + ", ".join(existing)
            )


def _split_records(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must satisfy 0 <= val_ratio < 1, got {val_ratio}")
    if not records:
        return [], []
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if val_ratio == 0:
        n_val = 0
    else:
        n_val = max(1, int(len(shuffled) * val_ratio))
    if n_val >= len(shuffled):
        raise ValueError(
            f"val_ratio={val_ratio} leaves no samples for training (total={len(shuffled)})"
        )
    val_records = sorted(shuffled[:n_val], key=lambda record: record["uid"])
    train_records = sorted(shuffled[n_val:], key=lambda record: record["uid"])
    return train_records, val_records


def generate(
    stage_d_root: Path,
    output: Path,
    category_vocab: Path,
    *,
    overwrite: bool = False,
    print_stats: bool = False,
) -> tuple[int, int, dict[str, object]]:
    """生成单个严格 JSONL、全局类别词表及带发布摘要的 metadata。

    返回 ``(record_count, category_count, summary)``，其中 ``summary`` 描述发布规模
    与 role 构成，不代表训练 runtime 的采样分布。
    """
    stage_d_root = stage_d_root.resolve()
    if not stage_d_root.is_dir():
        raise ValueError(f"Stage D root is not a directory: {stage_d_root}")
    metadata_path = output.with_name(f"{output.name}.metadata.json")
    with stage_d_run_lock(str(stage_d_root)):
        _reject_incomplete_publication(
            [output, category_vocab, metadata_path],
            overwrite=overwrite,
        )
        context = NoiseAdmissionContext.load()
        records, vocabulary, manifests, audios = _collect_records(
            stage_d_root, context=context
        )
        stats = _compute_stats(records, vocabulary)
        artifacts = [
            ("manifest", output, _jsonl_bytes(records)),
            ("category_vocab", category_vocab, _json_bytes(vocabulary)),
        ]
        metadata = _publication_metadata(
            mode="single",
            parameters={},
            artifacts=artifacts,
            manifest_summary=stats,
        )
        _assert_input_snapshots(stage_d_root, manifests, audios)
        _publish_or_reuse(
            [(path, payload) for _, path, payload in artifacts],
            metadata_path,
            metadata,
            overwrite=overwrite,
        )
    if print_stats:
        print(_format_stats_table(stats))
    return len(records), len(vocabulary), stats


def generate_split(
    stage_d_root: Path,
    output_dir: Path,
    *,
    val_ratio: float = 0.05,
    seed: int = 42,
    overwrite: bool = False,
    print_stats: bool = False,
) -> tuple[int, int, int, dict[str, object]]:
    """生成内嵌类别信息的训练/验证严格 JSONL 与发布 metadata。

    返回 ``(train_count, val_count, category_count, summary)``；metadata 中的
    ``manifest_summary`` 描述发布规模、role 与 split，不代表训练 runtime 采样分布。
    """
    output_dir = output_dir.resolve()
    train_output = output_dir / "train_noise_sources.jsonl"
    val_output = output_dir / "val_noise_sources.jsonl"
    metadata_path = output_dir / _SPLIT_METADATA_FILENAME
    legacy_sidecar_paths = [
        output_dir / filename for filename in _LEGACY_SIDECAR_FILENAMES
    ]
    stage_d_root = stage_d_root.resolve()
    if not stage_d_root.is_dir():
        raise ValueError(f"Stage D root is not a directory: {stage_d_root}")
    with stage_d_run_lock(str(stage_d_root)):
        publication_paths = [train_output, val_output, metadata_path]
        _reject_incomplete_publication(publication_paths, overwrite=overwrite)
        legacy_sidecar_identities = _legacy_sidecar_identities(
            legacy_sidecar_paths,
            overwrite=overwrite,
        )
        context = NoiseAdmissionContext.load()
        records, vocabulary, manifests, audios = _collect_records(
            stage_d_root, context=context
        )
        stats = _compute_stats(records, vocabulary)
        train_records, val_records = _split_records(records, val_ratio, seed)
        manifest_summary = {
            **stats,
            "split": {"train": len(train_records), "val": len(val_records)},
        }
        artifacts = [
            ("train_manifest", train_output, _jsonl_bytes(train_records)),
            ("val_manifest", val_output, _jsonl_bytes(val_records)),
        ]
        metadata = _publication_metadata(
            mode="split",
            parameters={"seed": seed, "val_ratio": val_ratio},
            artifacts=artifacts,
            manifest_summary=manifest_summary,
        )
        _assert_input_snapshots(stage_d_root, manifests, audios)
        _publish_or_reuse(
            [(path, payload) for _, path, payload in artifacts],
            metadata_path,
            metadata,
            overwrite=overwrite,
        )
        _remove_legacy_sidecars(legacy_sidecar_identities)
    if print_stats:
        print(_format_stats_table(stats))
    return len(train_records), len(val_records), len(vocabulary), stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-d-root", type=Path, required=True, help="包含 d_manifest_*.jsonl 的 Stage D 根目录")
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", type=Path, help="单个严格 JSONL 输出路径（兼容旧调用）")
    output_group.add_argument("--output-dir", type=Path, help="训练/验证 JSONL 与发布 metadata 三件套输出目录")
    parser.add_argument("--category-vocab", type=Path, help="单文件模式的 category_name -> category_id 输出路径")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="验证集比例（默认 0.05）")
    parser.add_argument("--seed", type=int, default=42, help="随机划分种子（默认 42）")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有发布集合，并安全清理普通遗留 sidecar")
    parser.add_argument("--print-stats", action="store_true", help="输出当前 manifest 发布摘要 markdown 表格")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output is not None:
        if args.category_vocab is None:
            raise SystemExit("--output requires --category-vocab")
        count, category_count, stats = generate(
            args.stage_d_root,
            args.output,
            args.category_vocab,
            overwrite=args.overwrite,
            print_stats=args.print_stats,
        )
        print(f"wrote {count} records, {category_count} categories")
        print(f"jsonl: {args.output}")
        print(f"vocab: {args.category_vocab}")
        if not args.print_stats:
            print(_format_stats_table(stats))
    else:
        if args.category_vocab is not None:
            raise SystemExit("--category-vocab is only valid with --output")
        train_count, val_count, category_count, stats = generate_split(
            args.stage_d_root,
            args.output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            overwrite=args.overwrite,
            print_stats=args.print_stats,
        )
        print(f"wrote train={train_count}, val={val_count}, total={train_count + val_count} records")
        print(f"categories: {category_count}")
        print(f"output_dir: {args.output_dir}")
        if not args.print_stats:
            print(_format_stats_table(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
