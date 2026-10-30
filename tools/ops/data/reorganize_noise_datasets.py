#!/usr/bin/env python3
"""按数据集名称重排并复制噪声数据目录。

源结构：source/<采样率>/<噪声类型>/<数据集名称>/<哈希分片>/<文件>
目标结构：output/<数据集名称>/<噪声类型>/<文件>
默认源目录和目标目录使用仓库相对路径，也可分别通过
`NOISE_REORGANIZE_SOURCE` 和 `NOISE_REORGANIZE_OUTPUT` 覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import stat
import sys
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.utils.file_transaction import publish_staged_files, unlink_file_if_exists  # noqa: E402

DEFAULT_SOURCE = Path(
    os.environ.get(
        "NOISE_REORGANIZE_SOURCE",
        "runs/noise_filter_pipeline/stage_d/D2/audio",
    )
)
DEFAULT_OUTPUT = Path(
    os.environ.get("NOISE_REORGANIZE_OUTPUT", "data/noise_v1")
)
DEFAULT_WORKERS = min(32, (os.cpu_count() or 1) + 4)
_COPY_CHUNK_SIZE = 1024 * 1024


class ReorganizeError(RuntimeError):
    """输入布局、目标冲突或复制过程错误。"""


@dataclass(frozen=True)
class CopyTask:
    """单个源文件及其重排后的目标文件。"""

    source: Path
    target: Path


@dataclass(frozen=True)
class SourceSnapshot:
    """预检时绑定到源文件描述符的身份、元数据与内容摘要。"""

    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


@dataclass(frozen=True)
class CopyResult:
    """单个文件的处理结果。"""

    status: str
    bytes_copied: int = 0
    source_snapshot: SourceSnapshot | None = None


def _is_relative_to(path: Path, parent: Path) -> bool:
    """兼容性封装：判断 path 是否位于 parent 内。"""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _absolute_path(path: Path) -> Path:
    """只做词法绝对化，避免目标路径校验前跟随符号链接。"""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _open_directory(path: Path, *, create_missing: bool) -> int:
    """逐级拒绝符号链接，并返回稳定的目录文件描述符。"""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(os.sep, flags)
    try:
        for component in path.parts[1:]:
            while True:
                try:
                    component_stat = os.lstat(component, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create_missing:
                        raise
                    try:
                        os.mkdir(component, 0o755, dir_fd=descriptor)
                    except FileExistsError:
                        continue
                    component_stat = os.lstat(component, dir_fd=descriptor)
                if stat.S_ISLNK(component_stat.st_mode):
                    raise ReorganizeError(f"路径中不允许符号链接: {path}")
                if not stat.S_ISDIR(component_stat.st_mode):
                    raise ReorganizeError(f"路径祖先不是目录: {path}")
                next_descriptor = os.open(
                    component,
                    flags,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor
                break
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_output_root(output: Path) -> None:
    """拒绝目标根及其所有已存在祖先中的符号链接。"""
    try:
        descriptor = _open_directory(output, create_missing=False)
    except FileNotFoundError:
        return
    os.close(descriptor)


def validate_roots(source: Path, output: Path) -> tuple[Path, Path]:
    """校验输入输出根目录存在性、相异性和非嵌套关系。"""
    source = source.expanduser().resolve()
    output = _absolute_path(output)

    if not source.is_dir():
        raise ReorganizeError(f"源目录不存在或不是目录: {source}")
    _validate_output_root(output)
    if source == output:
        raise ReorganizeError("源目录与目标目录不能相同")
    if _is_relative_to(output, source) or _is_relative_to(source, output):
        raise ReorganizeError("源目录与目标目录不能互相嵌套")
    return source, output


def iter_copy_tasks(source: Path, output: Path) -> Iterator[CopyTask]:
    """按数据集和噪声类型分组，校验同名冲突后流式生成任务。"""
    roots_by_target: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for sample_rate_entry in os.scandir(source):
        if sample_rate_entry.is_symlink():
            raise ReorganizeError(f"不支持符号链接: {sample_rate_entry.path}")
        if not sample_rate_entry.is_dir(follow_symlinks=False):
            raise ReorganizeError(f"源根目录下应仅包含采样率目录: {sample_rate_entry.path}")

        for noise_type_entry in os.scandir(sample_rate_entry.path):
            if noise_type_entry.is_symlink():
                raise ReorganizeError(f"不支持符号链接: {noise_type_entry.path}")
            if not noise_type_entry.is_dir(follow_symlinks=False):
                raise ReorganizeError(
                    f"采样率目录下应仅包含噪声类型目录: {noise_type_entry.path}"
                )

            noise_type = noise_type_entry.name
            for dataset_entry in os.scandir(noise_type_entry.path):
                if dataset_entry.is_symlink():
                    raise ReorganizeError(f"不支持符号链接: {dataset_entry.path}")
                if not dataset_entry.is_dir(follow_symlinks=False):
                    raise ReorganizeError(
                        f"噪声类型目录下应仅包含数据集目录: {dataset_entry.path}"
                    )
                roots_by_target[(dataset_entry.name, noise_type)].append(
                    Path(dataset_entry.path)
                )

    for (dataset, noise_type), dataset_roots in roots_by_target.items():
        sources_by_name: dict[str, Path] = {}
        for dataset_root in dataset_roots:
            for dirpath, dirnames, filenames in os.walk(dataset_root):
                current = Path(dirpath)
                for dirname in dirnames:
                    child = current / dirname
                    if child.is_symlink():
                        raise ReorganizeError(f"不支持符号链接: {child}")
                for filename in filenames:
                    src = current / filename
                    if src.is_symlink():
                        raise ReorganizeError(f"不支持符号链接: {src}")
                    previous = sources_by_name.setdefault(filename, src)
                    if previous != src:
                        raise ReorganizeError(
                            "移除采样率和哈希分片后出现同名文件: "
                            f"{previous} 与 {src}"
                        )
        for filename, src in sources_by_name.items():
            dst = output / dataset / noise_type / filename
            yield CopyTask(source=src, target=dst)


def _sha256_handle(handle) -> str:
    """从当前位置流式读取文件描述符并计算内容摘要。"""
    digest = hashlib.sha256()
    while chunk := handle.read(_COPY_CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def _open_regular_file(path: Path) -> int:
    """通过稳定父目录打开普通文件，拒绝路径祖先和叶节点符号链接。"""
    directory_descriptor = _open_directory(path.parent, create_missing=False)
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
    except BaseException:
        os.close(directory_descriptor)
        raise
    os.close(directory_descriptor)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ReorganizeError(f"文件不是普通文件: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _source_snapshot(path: Path) -> SourceSnapshot:
    """在单个安全打开的文件描述符上绑定源身份、元数据和摘要。"""
    try:
        descriptor = _open_regular_file(path)
    except OSError as exc:
        raise ReorganizeError(f"无法安全打开源文件: {path}: {exc}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        source_hash = _sha256_handle(handle)
        after = os.fstat(handle.fileno())
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise ReorganizeError(f"源文件在预检读取期间发生变化: {path}")
    return SourceSnapshot(
        device=after.st_dev,
        inode=after.st_ino,
        mode=after.st_mode,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
        sha256=source_hash,
    )


def _target_snapshot(target: Path) -> tuple[os.stat_result, str] | None:
    """安全打开目标普通文件并返回状态和内容摘要。"""
    try:
        descriptor = _open_regular_file(target)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ReorganizeError(f"无法安全打开目标文件: {target}: {exc}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        target_stat = os.fstat(handle.fileno())
        target_hash = _sha256_handle(handle)
    return target_stat, target_hash


def _preflight_task(task: CopyTask, *, dry_run: bool, overwrite: bool) -> CopyResult:
    """只读校验单任务；任何文件落盘前必须完成全树所有任务的预检。"""
    snapshot = _source_snapshot(task.source)
    target_snapshot = _target_snapshot(task.target)
    if target_snapshot is None:
        status = "planned" if dry_run else "copied"
    else:
        target_stat, target_hash = target_snapshot
        if snapshot.size == target_stat.st_size and snapshot.sha256 == target_hash:
            return CopyResult("skipped")
        if not overwrite:
            raise ReorganizeError(f"目标文件冲突: {task.source} -> {task.target}")
        status = "overwritten"
    return CopyResult(
        status,
        bytes_copied=snapshot.size,
        source_snapshot=snapshot,
    )


def _create_staging_file(target: Path) -> tuple[int, Path]:
    """在拒绝 symlink 的稳定目标父目录内创建独占 staging 文件。"""
    directory_descriptor = _open_directory(target.parent, create_missing=True)
    try:
        for _ in range(100):
            # 不用 tempfile 私有 API（_get_candidate_names 无稳定性承诺）；
            # token 只要求同目录内唯一，O_EXCL 已保证不覆盖
            candidate = secrets.token_hex(8)
            name = f".{target.name}.{candidate}.tmp"
            try:
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory_descriptor,
                )
            except FileExistsError:
                continue
            return descriptor, target.parent / name
    finally:
        os.close(directory_descriptor)
    raise FileExistsError(f"无法在目标目录创建 staging 文件: {target.parent}")


def _stage_task(task: CopyTask, snapshot: SourceSnapshot) -> Path:
    """按预检快照安全复制到 staging 文件，并在返回前 fsync。"""
    descriptor, temporary_path = _create_staging_file(task.target)
    try:
        try:
            source_descriptor = _open_regular_file(task.source)
        except OSError as exc:
            raise ReorganizeError(
                f"无法安全重开源文件: {task.source}: {exc}"
            ) from exc
        with os.fdopen(source_descriptor, "rb") as source, os.fdopen(
            descriptor, "wb"
        ) as target:
            source_stat = os.fstat(source.fileno())
            source_identity = (
                source_stat.st_dev,
                source_stat.st_ino,
                source_stat.st_mode,
                source_stat.st_size,
                source_stat.st_mtime_ns,
                source_stat.st_ctime_ns,
            )
            expected_identity = (
                snapshot.device,
                snapshot.inode,
                snapshot.mode,
                snapshot.size,
                snapshot.mtime_ns,
                snapshot.ctime_ns,
            )
            if source_identity != expected_identity:
                raise ReorganizeError(f"源文件在预检后发生变化: {task.source}")

            digest = hashlib.sha256()
            while chunk := source.read(_COPY_CHUNK_SIZE):
                digest.update(chunk)
                target.write(chunk)
            source_after = os.fstat(source.fileno())
            source_after_identity = (
                source_after.st_dev,
                source_after.st_ino,
                source_after.st_mode,
                source_after.st_size,
                source_after.st_mtime_ns,
                source_after.st_ctime_ns,
            )
            if (
                source_after_identity != expected_identity
                or digest.hexdigest() != snapshot.sha256
            ):
                raise ReorganizeError(f"源文件内容与预检结果不一致: {task.source}")
            os.fchmod(target.fileno(), stat.S_IMODE(snapshot.mode))
            target.flush()
            os.fsync(target.fileno())
            os.utime(
                target.fileno(),
                ns=(source_stat.st_atime_ns, snapshot.mtime_ns),
            )
            os.fsync(target.fileno())
        return temporary_path
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        unlink_file_if_exists(temporary_path)
        raise


def _drain_completed(
    pending: dict[Future[Path], int],
    staged_by_index: dict[int, Path],
) -> dict[Future[Path], int]:
    """等待至少一个 staging 任务完成并按任务序号记录结果。"""
    done, _ = wait(pending, return_when=FIRST_COMPLETED)
    for future in done:
        index = pending.pop(future)
        staged_by_index[index] = future.result()
    return pending


def run_reorganization(
    source: Path,
    output: Path,
    *,
    workers: int,
    dry_run: bool,
    overwrite: bool,
    tasks: Iterable[CopyTask] | None = None,
) -> Counter[str]:
    """先完成全树预检，再 staging 并以集合事务发布所有新增/覆盖文件。"""
    if workers < 1:
        raise ReorganizeError("--workers 必须大于等于 1")

    source, output = validate_roots(source, output)
    task_list = list(tasks) if tasks is not None else list(iter_copy_tasks(source, output))
    if not task_list:
        raise ReorganizeError("源目录中没有可复制的文件")

    summary: Counter[str] = Counter()
    publish_entries: list[tuple[CopyTask, SourceSnapshot]] = []
    for task in task_list:
        result = _preflight_task(task, dry_run=dry_run, overwrite=overwrite)
        summary[result.status] += 1
        summary["bytes"] += result.bytes_copied
        if result.status != "skipped":
            if result.source_snapshot is None:
                raise ReorganizeError(f"源文件预检未生成安全快照: {task.source}")
            publish_entries.append((task, result.source_snapshot))

    if dry_run or not publish_entries:
        return summary

    pending: dict[Future[Path], int] = {}
    staged_by_index: dict[int, Path] = {}
    max_pending = workers * 2
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, (task, snapshot) in enumerate(publish_entries):
                future = executor.submit(_stage_task, task, snapshot)
                pending[future] = index
                if len(pending) >= max_pending:
                    pending = _drain_completed(pending, staged_by_index)
            while pending:
                pending = _drain_completed(pending, staged_by_index)
        staged = [staged_by_index[index] for index in range(len(publish_entries))]
        publish_staged_files(
            staged,
            [task.target for task, _snapshot in publish_entries],
        )
    except BaseException:
        for future in pending:
            future.cancel()
        for future in pending:
            if future.cancelled() or not future.done():
                continue
            try:
                temporary_path = future.result()
            except BaseException:
                continue
            unlink_file_if_exists(temporary_path)
        for temporary_path in staged_by_index.values():
            unlink_file_if_exists(temporary_path)
        raise

    return summary


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="重排并复制噪声数据集目录，合并采样率并移除哈希分片目录。"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"源目录（默认: {DEFAULT_SOURCE}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"目标目录（默认: {DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并行复制线程数（默认: {DEFAULT_WORKERS}）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="完整扫描并校验，只汇总计划而不创建目录或复制文件",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖内容不一致的已有目标文件",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    try:
        summary = run_reorganization(
            args.source,
            args.output,
            workers=args.workers,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except (OSError, ReorganizeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    processed = sum(
        summary[key] for key in ("copied", "planned", "overwritten", "skipped")
    )
    mode = "dry-run" if args.dry_run else "copy"
    print(
        f"summary: mode={mode}, files={processed}, "
        f"copied={summary['copied']}, planned={summary['planned']}, "
        f"overwritten={summary['overwritten']}, skipped={summary['skipped']}, "
        f"bytes={summary['bytes']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
