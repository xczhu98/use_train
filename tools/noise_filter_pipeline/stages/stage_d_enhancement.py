"""
Stage D: Quality Enhancement。

串联两个子处理器：
  D1 Apollo Super-Resolution
  D2 Loudness Normalization & Quality Check

配置 ``stage_d.skip_d1: true`` 时可跳过 D1，直接用 D2 处理 Stage C 输出。
整个 run 持有 stage_d 粒度排它锁，避免 promote/cleanup 观察或删除半成品。

Stage D 对 C3 passed 记录执行无阈值复验（共享 evaluator），不再加载或
消费 admission calibration；approved 门禁已删除，未批准不再阻断发布。

启用 D1 且使用生产默认的 subprocess_daemon 模式时，通常调用
third_party_scripts/Apollo/inference_apollo.py --daemon，入口可由配置覆盖；dry_run 或
skip_d1 不调用 Apollo daemon。D2 负责后处理质检。
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import os
import stat
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm import tqdm

from ..core.execution_contract import validate_stage_c_execution_contract
from ..core.manifest_io import (
    AudioRecord,
    config_hash,
    parse_audio_record_exact,
    read_manifest_strict,
    write_json_file_transactionally,
)
from ..core.manifest_per_dataset import (
    link_stage_output,
    link_per_dataset_manifests,
    split_manifest_by_dataset,
    validate_dataset_id,
)
from ..core.pipeline_config import (
    UnifiedPipelineConfig,
    resolve_stage_d_config,
    resolve_stage_d_final_stage,
)
from ..core.run_lock import stage_d_run_lock, substage_writer_lock
from ..core.stage_orchestrator import StageOrchestratorBase
from ..core.stats import write_stage_stats
from ..processors.apollo_superres import ApolloSuperResProcessor
from tools.shared.audio_io import make_unique_output_path
from ..processors.loudness_normalizer import LoudnessNormalizerProcessor

logger = logging.getLogger(__name__)


_SNAPSHOT_SOURCE_PROVENANCE_KEY = "_stage_d_snapshot_source"


def _source_fingerprint(identity: tuple) -> dict[str, int]:
    """把稳定快照身份转换为 D2 既有的 stat 指纹结构。"""
    return {
        "dev": identity[1],
        "ino": identity[2],
        "mode": identity[3],
        "size": identity[4],
        "mtime_ns": identity[5],
        "ctime_ns": identity[6],
    }


def _snapshot_source(rec: AudioRecord) -> Optional[dict[str, Any]]:
    """读取当前记录的 Stage D 私有快照映射。"""
    provenance = rec.provenance
    value = (
        provenance.get(_SNAPSHOT_SOURCE_PROVENANCE_KEY)
        if isinstance(provenance, dict)
        else None
    )
    if value is None:
        return None
    if not (
        isinstance(value, dict)
        and isinstance(value.get("path"), str)
        and isinstance(value.get("snapshot_path"), str)
        and isinstance(value.get("fingerprint"), dict)
        and isinstance(value.get("sha256"), str)
    ):
        raise ValueError("Stage D snapshot source provenance 非法")
    return value


class _SnapshotLoudnessNormalizerProcessor(LoudnessNormalizerProcessor):
    """读取私有快照，同时按原始源路径派生输出与 resume 身份。"""

    def __init__(
        self,
        config: dict[str, Any],
        snapshot_sources: dict[str, tuple[str, tuple]],
        record_sources: dict[str, str],
    ):
        self._snapshot_sources = snapshot_sources
        self._record_sources = record_sources
        super().__init__(config)

    def _logical_source_path(
        self, rec: AudioRecord, input_path: Optional[str]
    ) -> Optional[str]:
        if isinstance(input_path, str) and input_path in self._snapshot_sources:
            return self._record_sources.get(
                rec.record_id, self._snapshot_sources[input_path][0]
            )
        source = _snapshot_source(rec)
        if source is not None and input_path == source["snapshot_path"]:
            return source["path"]
        return input_path

    def _dataset_label_source_identity(self, rec: AudioRecord) -> dict[str, Any]:
        identity = super()._dataset_label_source_identity(rec)
        input_path = rec.audio.get("path")
        source = (
            self._snapshot_sources.get(input_path)
            if isinstance(input_path, str)
            else None
        )
        if source is not None:
            identity["audio"]["path"] = self._logical_source_path(rec, input_path)
            identity["audio"]["fingerprint"] = _source_fingerprint(source[1])
        else:
            provenance_source = _snapshot_source(rec)
            if provenance_source is not None:
                identity["audio"]["path"] = provenance_source["path"]
                identity["audio"]["fingerprint"] = provenance_source["fingerprint"]
        return identity

    def _output_path_for(
        self,
        out_sr: int,
        rec: AudioRecord,
        input_path: str,
        ordinal: Optional[int] = None,
    ) -> str:
        logical_path = self._logical_source_path(rec, input_path)
        return super()._output_path_for(out_sr, rec, logical_path, ordinal)


class _TransformingManifestWriter:
    """在 Apollo 记录落盘前收口其确定性输出路径。"""

    def __init__(self, writer: Any, transform: Callable[[AudioRecord], None]):
        self._writer = writer
        self._transform = transform

    def append(self, rec: AudioRecord) -> None:
        self._transform(rec)
        self._writer.append(rec)


class _SnapshotApolloSuperResProcessor(ApolloSuperResProcessor):
    """Apollo 读取私有快照，但保持源路径派生的既有输出命名。"""

    def __init__(
        self,
        config: dict[str, Any],
        snapshot_sources: dict[str, tuple[str, tuple]],
        record_sources: dict[str, str],
    ):
        self._snapshot_sources = snapshot_sources
        self._record_sources = record_sources
        super().__init__(config)
        if self.post_normalize:
            normalizer_config = config.get("post_normalize")
            self._normalizer = _SnapshotLoudnessNormalizerProcessor(
                normalizer_config if isinstance(normalizer_config, dict) else {},
                snapshot_sources,
                record_sources,
            )

    @staticmethod
    def _is_within(path: str, root: str) -> bool:
        try:
            absolute_root = os.path.abspath(root)
            return os.path.commonpath((os.path.abspath(path), absolute_root)) == absolute_root
        except ValueError:
            return False

    def _normalize_record_output_path(self, rec: AudioRecord, output_audio_dir: str) -> None:
        source = _snapshot_source(rec)
        current_path = rec.audio.get("path")
        if source is None or not isinstance(current_path, str):
            return

        if self.post_normalize:
            if self._normalizer is None or not rec.is_stage_passed("D2"):
                return
            out_sr = rec.audio.get("sr")
            if isinstance(out_sr, bool) or not isinstance(out_sr, int):
                raise RuntimeError(f"D2 passed 记录 {rec.record_id!r} 缺少有效采样率")
            expected_path = self._normalizer._output_path_for(
                out_sr, rec, source["path"]
            )
        else:
            if not rec.is_stage_passed("D1"):
                return
            expected_path = make_unique_output_path(
                output_audio_dir, source["path"], ext=f".{self.output_format}"
            )

        if os.path.abspath(current_path) == os.path.abspath(expected_path):
            return
        if not self._is_within(current_path, output_audio_dir):
            raise RuntimeError(f"Stage D 拒绝移动非 D1 owned 输出: {current_path}")
        if not self._is_within(expected_path, output_audio_dir):
            raise RuntimeError(f"Stage D 拒绝发布到 D1 输出目录外: {expected_path}")
        current_stat = os.lstat(current_path)
        if not stat.S_ISREG(current_stat.st_mode):
            raise RuntimeError(f"Stage D D1 输出不是 regular file: {current_path}")
        os.makedirs(os.path.dirname(expected_path), exist_ok=True)
        os.replace(current_path, expected_path)
        rec.audio["path"] = expected_path

    def _skip_apollo(self, rec: AudioRecord, output_audio_dir: str) -> None:
        super()._skip_apollo(rec, output_audio_dir)
        self._normalize_record_output_path(rec, output_audio_dir)

    def _flush_pending(
        self,
        pending: list[AudioRecord],
        output_dir: str,
        output_audio_dir: str,
        manifest_writer: Any,
        ckpt: Any,
        batch_index: list[int],
    ) -> None:
        writer = _TransformingManifestWriter(
            manifest_writer,
            lambda rec: self._normalize_record_output_path(rec, output_audio_dir),
        )
        super()._flush_pending(
            pending,
            output_dir,
            output_audio_dir,
            writer,
            ckpt,
            batch_index,
        )


class StageDOrchestrator(StageOrchestratorBase):
    """Stage D orchestrator。"""

    COMPLETION_FACT_NAME = "d_completion.json"
    COMPLETION_FACT_SCHEMA = "noise_filter_stage_d_completion"
    COMPLETION_FACT_VERSION = 1
    _SNAPSHOT_COPY_CHUNK_SIZE = 1024 * 1024
    _FICLONE = 0x40049409
    _REFLINK_UNSUPPORTED_ERRNOS = frozenset(
        {errno.EOPNOTSUPP, errno.ENOTTY, errno.EXDEV, errno.EINVAL, errno.ENOSYS}
    )
    _SNAPSHOT_DIR_NAME = ".stage_d_validated_snapshot"
    _SNAPSHOT_SENTINEL_NAME = ".stage_d_owned"
    _SNAPSHOT_SENTINEL = b"stage_d_validated_snapshot_v1\n"

    def __init__(self, config: UnifiedPipelineConfig):
        self.config = config
        self.stage_cfg = resolve_stage_d_config(config.get_stage_config("D"))

    @staticmethod
    def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            stat.S_IFMT(value.st_mode),
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    @classmethod
    def _stable_file_snapshot(
        cls,
        path: Path,
        *,
        include_bytes: bool,
        line_callback: Optional[Callable[[bytes, int], None]] = None,
        copy_path: Path | None = None,
        copy_result: Optional[Callable[[str, int], None]] = None,
    ) -> tuple[bytes | None, tuple]:
        """从稳定源 FD 完整哈希，并可创建经过完整内容校验的私有副本。"""
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ValueError(f"{path}: cannot open stable file snapshot: {exc}") from exc
        copy_descriptor: int | None = None
        copy_method: str | None = None
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{path}: snapshot source must be a regular file")

            if copy_path is not None:
                copy_descriptor = os.open(
                    copy_path,
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                try:
                    fcntl.ioctl(copy_descriptor, cls._FICLONE, descriptor)
                    copy_method = "reflink"
                except OSError as exc:
                    if exc.errno not in cls._REFLINK_UNSUPPORTED_ERRNOS:
                        raise
                    os.ftruncate(copy_descriptor, 0)
                    os.lseek(copy_descriptor, 0, os.SEEK_SET)
                    copy_method = "fallback"

            source_digest = hashlib.sha256()
            chunks: list[bytes] | None = [] if include_bytes else None
            byte_count = 0
            with os.fdopen(descriptor, "rb", closefd=False) as source_stream:
                if line_callback is not None:
                    iterator = enumerate(source_stream, start=1)
                    for line_number, chunk in iterator:
                        source_digest.update(chunk)
                        byte_count += len(chunk)
                        if chunks is not None:
                            chunks.append(chunk)
                        line_callback(
                            chunk[:-1] if chunk.endswith(b"\n") else chunk,
                            line_number,
                        )
                else:
                    while True:
                        chunk = source_stream.read(cls._SNAPSHOT_COPY_CHUNK_SIZE)
                        if not chunk:
                            break
                        source_digest.update(chunk)
                        byte_count += len(chunk)
                        if chunks is not None:
                            chunks.append(chunk)
                        if copy_descriptor is not None and copy_method == "fallback":
                            view = memoryview(chunk)
                            while view:
                                written = os.write(copy_descriptor, view)
                                if written <= 0:
                                    raise OSError("snapshot copy write made no progress")
                                view = view[written:]

            after = os.fstat(descriptor)
            path_after = os.stat(path, follow_symlinks=False)
            identity = cls._stat_identity(before)
            if (
                identity != cls._stat_identity(after)
                or identity != cls._stat_identity(path_after)
                or byte_count != before.st_size
            ):
                raise ValueError(f"{path}: file changed while reading stable snapshot")

            if copy_descriptor is not None and copy_path is not None:
                copy_after = os.fstat(copy_descriptor)
                copy_path_after = os.stat(copy_path, follow_symlinks=False)
                if (
                    not stat.S_ISREG(copy_after.st_mode)
                    or copy_after.st_size != byte_count
                    or copy_after.st_nlink != 1
                    or cls._stat_identity(copy_after) != cls._stat_identity(copy_path_after)
                    or (copy_after.st_dev, copy_after.st_ino)
                    == (before.st_dev, before.st_ino)
                ):
                    raise ValueError(f"{copy_path}: invalid snapshot copy identity")
                os.lseek(copy_descriptor, 0, os.SEEK_SET)
                target_digest = hashlib.sha256()
                target_byte_count = 0
                while True:
                    chunk = os.read(copy_descriptor, cls._SNAPSHOT_COPY_CHUNK_SIZE)
                    if not chunk:
                        break
                    target_digest.update(chunk)
                    target_byte_count += len(chunk)
                copy_final = os.fstat(copy_descriptor)
                copy_path_final = os.stat(copy_path, follow_symlinks=False)
                source_final = os.fstat(descriptor)
                source_path_final = os.stat(path, follow_symlinks=False)
                if (
                    target_byte_count != byte_count
                    or cls._stat_identity(copy_final) != cls._stat_identity(copy_after)
                    or cls._stat_identity(copy_path_final) != cls._stat_identity(copy_after)
                    or copy_final.st_nlink != 1
                    or target_digest.digest() != source_digest.digest()
                    or cls._stat_identity(source_final) != identity
                    or cls._stat_identity(source_path_final) != identity
                ):
                    raise ValueError(f"{copy_path}: snapshot copy content mismatch")
                if copy_result is not None:
                    copy_result(copy_method or "fallback", byte_count)
        except OSError as exc:
            raise ValueError(f"{path}: file changed while reading stable snapshot: {exc}") from exc
        finally:
            if copy_descriptor is not None:
                os.close(copy_descriptor)
            os.close(descriptor)
        return b"".join(chunks) if chunks is not None else None, (
            source_digest.hexdigest(),
            *identity,
        )

    @staticmethod
    def _manifest_paths(input_manifest: str) -> list[Path]:
        candidate = Path(input_manifest)
        paths = sorted(candidate.glob("*_manifest_*.jsonl")) if candidate.is_dir() else [candidate]
        if not paths:
            raise ValueError(f"{input_manifest}: no manifest files found")
        return paths

    @staticmethod
    def _bounded_completion_results(executor, function, items, window):
        """有界提交并按完成顺序产出；异常时取消排队任务并等待已启动任务。"""
        iterator = iter(items)
        pending: set[Future] = set()

        def submit_one() -> bool:
            try:
                item = next(iterator)
            except StopIteration:
                return False
            pending.add(executor.submit(function, item))
            return True

        for _ in range(window):
            if not submit_one():
                break
        try:
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.remove(future)
                    result = future.result()
                    submit_one()
                    yield result
        except BaseException:
            for future in pending:
                future.cancel()
            wait(pending)
            raise

    def _snapshot_num_workers(self) -> int:
        """仅解析 snapshot runtime 并发；配置缺省时不固化机器相关默认值。"""
        return int(
            self.stage_cfg.get("snapshot_num_workers", min(4, os.cpu_count() or 1))
        )

    def _assert_source_snapshots(
        self,
        input_manifest: str,
        manifest_snapshots: dict[Path, tuple],
        audio_snapshots: dict[Path, tuple],
    ) -> None:
        """最终发布前复验 manifest 集合及 manifest/audio 内容身份。"""
        if self._manifest_paths(input_manifest) != sorted(manifest_snapshots):
            raise ValueError(f"{input_manifest}: manifest file set changed before publication")
        for path, expected in manifest_snapshots.items():
            _, actual = self._stable_file_snapshot(path, include_bytes=False)
            if actual != expected:
                raise ValueError(f"{path}: manifest identity changed before publication")

        workers = self._snapshot_num_workers()

        def verify(item):
            path, expected = item
            _, actual = self._stable_file_snapshot(path, include_bytes=False)
            if actual != expected:
                raise ValueError(f"{path}: audio identity changed before publication")
            return path

        with ThreadPoolExecutor(max_workers=workers) as executor, tqdm(
            total=len(audio_snapshots),
            desc="Stage D source revalidation",
            unit="unique-file",
        ) as progress:
            for _ in self._bounded_completion_results(
                executor, verify, audio_snapshots.items(), max(workers * 2, 1)
            ):
                progress.update()
        logger.info(
            "Stage D source revalidation 完成：%d 个 unique source",
            len(audio_snapshots),
        )

    @staticmethod
    def _restore_snapshot_record(
        data: dict[str, Any],
        snapshot_sources: dict[str, tuple[str, tuple]],
        record_sources: dict[str, str],
    ) -> bool:
        """移除单条记录中的私有路径，并恢复对应源 stat 身份。"""
        provenance = data.get("provenance")
        source = (
            provenance.pop(_SNAPSHOT_SOURCE_PROVENANCE_KEY, None)
            if isinstance(provenance, dict)
            else None
        )
        if source is None:
            return False
        if not (
            isinstance(source, dict)
            and isinstance(source.get("path"), str)
            and isinstance(source.get("snapshot_path"), str)
            and isinstance(source.get("fingerprint"), dict)
            and isinstance(source.get("sha256"), str)
        ):
            raise ValueError("Stage D snapshot source provenance 非法")

        snapshot_path = source["snapshot_path"]
        source_path = source["path"]
        mapped = snapshot_sources.get(snapshot_path)
        record_id = data.get("record_id")
        if (
            mapped is None
            or mapped[0] != source_path
            or source["fingerprint"] != _source_fingerprint(mapped[1])
            or source["sha256"] != mapped[1][0]
            or record_sources.get(record_id) != source_path
        ):
            raise ValueError(
                f"Stage D 记录 {record_id!r} 的 snapshot source 映射不一致"
            )

        def restore(value: Any) -> Any:
            if isinstance(value, dict):
                if value.get("path") == snapshot_path:
                    value["path"] = source_path
                    if "fingerprint" in value:
                        value["fingerprint"] = source["fingerprint"]
                if value.get("input_path") == snapshot_path:
                    value["input_path"] = source_path
                    if "input_fingerprint" in value:
                        value["input_fingerprint"] = source["fingerprint"]
                for key, item in list(value.items()):
                    value[key] = restore(item)
                return value
            if isinstance(value, list):
                return [restore(item) for item in value]
            if isinstance(value, str):
                return value.replace(snapshot_path, source_path)
            return value

        restore(data)
        return True

    @classmethod
    def _restore_snapshot_manifests(
        cls,
        manifest_dir: str,
        stage_name: str,
        snapshot_sources: dict[str, tuple[str, tuple]],
        record_sources: dict[str, str],
        *,
        require_outputs: bool = True,
    ) -> tuple[bool, int]:
        """逐文件逐行原子清理子阶段终态及保留过程文件中的私有引用。"""
        manifest_root = Path(manifest_dir)
        stage_prefix = stage_name.lower()
        output_paths = sorted(
            manifest_root.glob(f"{stage_prefix}_manifest_*.jsonl")
        )
        if not output_paths and require_outputs:
            raise RuntimeError(f"{stage_name} 没有可清理的 per-dataset manifest")
        intermediate_paths = [
            path
            for path in (
                manifest_root / f"{stage_prefix}_input_eligible.jsonl",
                manifest_root / f"{stage_prefix}_input_ineligible.jsonl",
            )
            if path.is_file()
        ]

        changed_any = False
        line_count = 0
        for path in [*output_paths, *intermediate_paths]:
            count_for_marker = path in output_paths
            source_stat = os.stat(path, follow_symlinks=False)
            if not stat.S_ISREG(source_stat.st_mode):
                raise ValueError(f"{path}: snapshot restore source must be a regular file")
            descriptor, tmp_path = tempfile.mkstemp(
                prefix=f".{path.name}.snapshot_restore.",
                suffix=".tmp",
                dir=path.parent,
            )
            changed = False
            try:
                os.fchmod(descriptor, stat.S_IMODE(source_stat.st_mode))
                with open(path, "r", encoding="utf-8") as source, os.fdopen(
                    descriptor, "w", encoding="utf-8"
                ) as destination:
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            raise ValueError(f"{path}:{line_number}: blank JSONL line")
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"{path}:{line_number}: invalid JSON: {exc.msg}"
                            ) from exc
                        if not isinstance(data, dict):
                            raise ValueError(
                                f"{path}:{line_number}: JSON value must be an object"
                            )
                        changed = (
                            cls._restore_snapshot_record(
                                data, snapshot_sources, record_sources
                            )
                            or changed
                        )
                        destination.write(
                            json.dumps(
                                data,
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        if count_for_marker:
                            line_count += 1
                    destination.flush()
                    os.fsync(destination.fileno())
                if changed:
                    os.replace(tmp_path, path)
                    changed_any = True
                else:
                    os.unlink(tmp_path)
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        return changed_any, line_count

    def _run_snapshot_substage(
        self,
        *,
        processor: Any,
        input_manifest: str,
        output_dir: str,
        snapshot_sources: dict[str, tuple[str, tuple]],
        record_sources: dict[str, str],
        source_manifest: str,
        **kwargs: Any,
    ) -> str:
        """在同一 writer 锁内运行子阶段、去引用并重绑 marker。"""
        writer_lock_timeout = kwargs.pop("writer_lock_timeout", 30.0)
        marker_path = os.path.join(
            output_dir, f"{processor.stage_name.lower()}_output.complete"
        )
        with substage_writer_lock(output_dir, timeout=writer_lock_timeout):
            self._preflight_processor(processor)
            try:
                result = self._run_substage_locked(
                    processor,
                    input_manifest,
                    output_dir,
                    **kwargs,
                )
            except BaseException as processing_error:
                try:
                    self._restore_snapshot_manifests(
                        output_dir,
                        processor.stage_name,
                        snapshot_sources,
                        record_sources,
                        require_outputs=False,
                    )
                except BaseException as restore_error:
                    note = (
                        "Stage D 子阶段失败现场无法清理私有快照引用: "
                        f"{restore_error}"
                    )
                    add_note = getattr(processing_error, "add_note", None)
                    if callable(add_note):
                        add_note(note)
                    else:
                        processing_error.__notes__ = [
                            *getattr(processing_error, "__notes__", ()),
                            note,
                        ]
                try:
                    os.remove(marker_path)
                except FileNotFoundError:
                    pass
                raise
            try:
                changed, line_count = self._restore_snapshot_manifests(
                    result,
                    processor.stage_name,
                    snapshot_sources,
                    record_sources,
                )
                if changed:
                    self._write_dir_marker(
                        marker_path,
                        input_manifest,
                        processor,
                        line_count,
                        output_dir,
                    )

                snapshot_manifest = os.path.join(
                    os.path.dirname(output_dir),
                    self._SNAPSHOT_DIR_NAME,
                    "stage_c_snapshot.jsonl",
                )
                if os.path.abspath(input_manifest) == os.path.abspath(
                    snapshot_manifest
                ):
                    with open(marker_path, "r", encoding="utf-8") as marker_file:
                        marker = json.load(marker_file)
                    normalized_marker = dict(marker)
                    normalized_marker.pop("input_identities", None)
                    normalized_marker["input_manifest"] = os.path.abspath(
                        source_manifest
                    )
                    if normalized_marker != marker:
                        write_json_file_transactionally(
                            normalized_marker, marker_path
                        )
                return result
            except BaseException:
                try:
                    os.remove(marker_path)
                except FileNotFoundError:
                    pass
                raise

    @classmethod
    def _write_snapshot_sentinel(cls, snapshot_root: Path) -> None:
        sentinel = snapshot_root / cls._SNAPSHOT_SENTINEL_NAME
        descriptor = os.open(
            sentinel,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(cls._SNAPSHOT_SENTINEL)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        os.chmod(snapshot_root, 0o700)

    @classmethod
    def _owned_snapshot_identity(cls, snapshot_root: Path) -> tuple[int, int]:
        """只认可本模块 sentinel 标识的非 symlink 普通目录。"""
        try:
            root_stat = os.lstat(snapshot_root)
        except OSError as exc:
            raise RuntimeError(f"无法检查 Stage D 快照目录 {snapshot_root}: {exc}") from exc
        if not stat.S_ISDIR(root_stat.st_mode):
            raise RuntimeError(f"Stage D 快照路径不是 owned 普通目录: {snapshot_root}")

        sentinel = snapshot_root / cls._SNAPSHOT_SENTINEL_NAME
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(sentinel, flags)
        except OSError as exc:
            raise RuntimeError(f"Stage D 快照目录缺少有效 owned sentinel: {snapshot_root}") from exc
        try:
            sentinel_stat = os.fstat(descriptor)
            payload = os.read(descriptor, len(cls._SNAPSHOT_SENTINEL) + 1)
        finally:
            os.close(descriptor)
        if not stat.S_ISREG(sentinel_stat.st_mode) or payload != cls._SNAPSHOT_SENTINEL:
            raise RuntimeError(f"Stage D 快照目录 owned sentinel 非法: {snapshot_root}")
        return root_stat.st_dev, root_stat.st_ino

    @classmethod
    def _promote_snapshot(
        cls,
        staging_root: Path,
        snapshot_root: Path,
        cleanup_root: Path,
    ) -> tuple[int, int]:
        """原子发布新快照；仅隔离可证明 owned 的崩溃遗留。"""
        quarantine = cleanup_root / "previous_snapshot"
        had_previous = os.path.lexists(snapshot_root)
        if had_previous:
            cls._owned_snapshot_identity(snapshot_root)
            os.replace(snapshot_root, quarantine)
        try:
            os.replace(staging_root, snapshot_root)
        except BaseException:
            if had_previous:
                os.replace(quarantine, snapshot_root)
            raise
        return cls._owned_snapshot_identity(snapshot_root)

    @classmethod
    def _retire_snapshot(
        cls,
        snapshot_root: Path,
        expected_identity: tuple[int, int],
        cleanup_root: Path,
    ) -> None:
        """把当前 run 的 owned 快照移回临时树，由 TemporaryDirectory 回收。"""
        actual_identity = cls._owned_snapshot_identity(snapshot_root)
        if actual_identity != expected_identity:
            raise RuntimeError(
                f"Stage D 快照目录在运行期间被替换: {snapshot_root}"
            )
        os.replace(snapshot_root, cleanup_root / "current_snapshot")

    def run(
        self,
        input_manifest: str,
        output_dir: str,
        output_manifest: str,
        global_pbar: Optional[tqdm] = None,
    ) -> str:
        """持有 run 锁，以预检快照作为 D1/D2 的唯一输入。"""
        with stage_d_run_lock(output_dir):
            stats_path = os.path.join(output_dir, "d_stats.json")
            completion_path = os.path.join(output_dir, self.COMPLETION_FACT_NAME)
            for stale_fact in (stats_path, completion_path):
                try:
                    os.remove(stale_fact)
                except FileNotFoundError:
                    pass

            output_root = Path(os.path.abspath(output_dir))
            output_root.parent.mkdir(parents=True, exist_ok=True)
            snapshot_root = output_root / self._SNAPSHOT_DIR_NAME
            with tempfile.TemporaryDirectory(
                prefix=".stage_d_validated_staging_",
                dir=output_root.parent,
            ) as cleanup_dir:
                cleanup_root = Path(cleanup_dir)
                staging_root = cleanup_root / "payload"
                staging_root.mkdir(mode=0o700)
                staging_snapshot = staging_root / "stage_c_snapshot.jsonl"
                published_snapshot = snapshot_root / staging_snapshot.name
                manifests, audios, snapshot_sources, record_sources = (
                    self._build_validated_snapshot(
                        input_manifest,
                        str(staging_snapshot),
                        snapshot_root,
                    )
                )
                self._write_snapshot_sentinel(staging_root)

                output_root.mkdir(parents=True, exist_ok=True)
                snapshot_identity = self._promote_snapshot(
                    staging_root, snapshot_root, cleanup_root
                )
                try:
                    result, completion = self._run_locked(
                        str(published_snapshot),
                        output_dir,
                        output_manifest,
                        global_pbar,
                        input_manifest,
                        manifests,
                        audios,
                        snapshot_sources,
                        record_sources,
                    )
                except BaseException:
                    try:
                        self._retire_snapshot(
                            snapshot_root, snapshot_identity, cleanup_root
                        )
                    except Exception:
                        logger.exception(
                            "Stage D 异常路径无法回收 owned validated snapshot %s",
                            snapshot_root,
                        )
                    raise
                self._retire_snapshot(snapshot_root, snapshot_identity, cleanup_root)

            try:
                write_json_file_transactionally(completion, completion_path)
            except BaseException:
                try:
                    os.remove(completion_path)
                except FileNotFoundError:
                    pass
                raise
            logger.info(
                "Stage D 完成：%d 条通过 D2，输出 %s",
                completion["d2_passed"],
                output_dir,
            )
            return result

    def _build_validated_snapshot(
        self,
        input_manifest: str,
        snapshot_path: str,
        snapshot_root: Path,
    ) -> tuple[
        dict[Path, tuple],
        dict[Path, tuple],
        dict[str, tuple[str, tuple]],
        dict[str, str],
    ]:
        """严格预检 Stage C，并有界并行构建 unique source 私有副本。"""
        from core.data.noise_admission_policy import (
            C2_LABEL_FIELDS,
            C2_STAGE_FIELDS,
            C3_STAGE_FIELDS,
            LEGACY_C2_LABEL_FIELDS,
            LEGACY_C3_FIELDS,
            load_admission_policy,
            validate_admission_contract,
        )

        policy = load_admission_policy()
        manifest_paths = self._manifest_paths(input_manifest)
        manifest_snapshots: dict[Path, tuple] = {}
        audio_snapshots: dict[Path, tuple] = {}
        snapshot_sources: dict[str, tuple[str, tuple]] = {}
        record_sources: dict[str, str] = {}
        private_audio_paths: dict[Path, Path] = {}
        future_paths: dict[Future, tuple[Path, Path]] = {}
        pending_records: dict[int, tuple[dict[str, Any], str, Path | None]] = {}
        input_record_count = 0
        eligible_count = 0
        next_write_ordinal = 0
        seen_ids: set[str] = set()
        copy_counts = {"reflink": 0, "fallback": 0}
        logical_bytes = 0
        workers = self._snapshot_num_workers()
        window = max(workers * 2, 1)
        tmp_path = snapshot_path + ".tmp"
        private_audio_root = Path(snapshot_path).parent / "audio"
        private_audio_root.mkdir(mode=0o700)

        def copy_source(item: tuple[Path, Path]):
            source_path, private_path = item
            result: list[tuple[str, int]] = []
            _, identity = self._stable_file_snapshot(
                source_path,
                include_bytes=False,
                copy_path=private_path,
                copy_result=lambda method, size: result.append((method, size)),
            )
            if len(result) != 1:
                raise RuntimeError(f"{source_path}: snapshot copy result is missing")
            method, size = result[0]
            return source_path, private_path, identity, method, size

        with open(tmp_path, "w", encoding="utf-8") as snapshot, ThreadPoolExecutor(
            max_workers=workers
        ) as executor, tqdm(
            total=0,
            desc="Stage D snapshot build",
            unit="unique-file",
        ) as progress:
            def flush_ready() -> None:
                nonlocal next_write_ordinal
                while next_write_ordinal in pending_records:
                    data, record_id, audio_path = pending_records[next_write_ordinal]
                    if audio_path is not None and audio_path not in audio_snapshots:
                        break
                    pending_records.pop(next_write_ordinal)
                    if audio_path is not None:
                        identity = audio_snapshots[audio_path]
                        private_audio_path = private_audio_paths[audio_path]
                        snapshot_audio_path = str(
                            (snapshot_root / "audio" / private_audio_path.name).resolve()
                        )
                        snapshot_sources.setdefault(
                            snapshot_audio_path, (str(audio_path), identity)
                        )
                        record_sources[record_id] = str(audio_path)
                        provenance = data.setdefault("provenance", {})
                        provenance[_SNAPSHOT_SOURCE_PROVENANCE_KEY] = {
                            "path": str(audio_path),
                            "snapshot_path": snapshot_audio_path,
                            "fingerprint": _source_fingerprint(identity),
                            "sha256": identity[0],
                        }
                        data["audio"]["path"] = snapshot_audio_path
                    snapshot.write(
                        json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    next_write_ordinal += 1

            def reap_completed(*, block: bool) -> None:
                nonlocal logical_bytes
                if not future_paths:
                    return
                completed, _ = wait(
                    future_paths,
                    timeout=None if block else 0,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    audio_path, private_audio_path = future_paths.pop(future)
                    (
                        result_path,
                        result_private_path,
                        identity,
                        method,
                        size,
                    ) = future.result()
                    if result_path != audio_path or result_private_path != private_audio_path:
                        raise RuntimeError("Stage D snapshot worker returned mismatched paths")
                    audio_snapshots[audio_path] = identity
                    copy_counts[method] += 1
                    logical_bytes += size
                    progress.update()
                flush_ready()

            def cancel_pending() -> None:
                for future in future_paths:
                    future.cancel()
                wait(future_paths)

            try:
                for manifest_path in manifest_paths:
                    def consume_line(raw_line: bytes, line_number: int) -> None:
                        nonlocal input_record_count, eligible_count
                        location = f"{manifest_path}:{line_number}"
                        if not raw_line.strip():
                            raise ValueError(f"{location}: blank JSONL line")
                        try:
                            data = json.loads(raw_line.decode("utf-8"))
                        except UnicodeDecodeError as exc:
                            raise ValueError(
                                f"{manifest_path}: manifest is not valid UTF-8"
                            ) from exc
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"{location}: invalid JSON: {exc.msg}"
                            ) from exc
                        rec = parse_audio_record_exact(
                            data, location=location, require_audio_path=False
                        )
                        validate_dataset_id(rec.dataset_id)
                        if _SNAPSHOT_SOURCE_PROVENANCE_KEY in rec.provenance:
                            raise ValueError(
                                f"{location}: reserved Stage D snapshot provenance key"
                            )
                        if rec.record_id in seen_ids:
                            raise ValueError(
                                f"{location}: duplicate record_id: {rec.record_id!r}"
                            )
                        seen_ids.add(rec.record_id)
                        ordinal = input_record_count
                        input_record_count += 1
                        stages = rec.stages if isinstance(rec.stages, dict) else {}
                        legacy_stages = sorted({"C4", "C", "D"} & set(stages))
                        if legacy_stages:
                            raise ValueError(
                                f"record {rec.record_id!r}: legacy stages are not supported: "
                                f"{legacy_stages!r}"
                            )
                        c3_result = stages.get("C3")
                        if not isinstance(c3_result, dict):
                            raise ValueError(
                                f"record {rec.record_id!r} C3: stage is missing"
                            )
                        legacy_c3 = LEGACY_C3_FIELDS & set(c3_result)
                        if legacy_c3:
                            raise ValueError(
                                f"record {rec.record_id!r} C3: legacy admission fields are "
                                f"forbidden: {sorted(legacy_c3)}"
                            )
                        unknown_c3 = set(c3_result) - C3_STAGE_FIELDS
                        if unknown_c3:
                            raise ValueError(
                                f"record {rec.record_id!r} C3: unknown fields are forbidden: "
                                f"{sorted(unknown_c3)}"
                            )
                        c3_status = c3_result.get("status")
                        if c3_status not in {"passed", "rejected", "skipped"}:
                            raise ValueError(
                                f"record {rec.record_id!r} C3: unsupported status "
                                f"{c3_status!r}"
                            )
                        labels = rec.labels if isinstance(rec.labels, dict) else {}
                        legacy_labels = LEGACY_C2_LABEL_FIELDS & set(labels)
                        if legacy_labels:
                            raise ValueError(
                                f"record {rec.record_id!r} C3: legacy admission fields are "
                                f"forbidden: {sorted(legacy_labels)}"
                            )
                        unknown_labels = set(labels) - C2_LABEL_FIELDS
                        if unknown_labels:
                            raise ValueError(
                                f"record {rec.record_id!r} C3: unknown labels fields are "
                                f"forbidden: {sorted(unknown_labels)}"
                            )
                        c2_result = stages.get("C2")
                        if isinstance(c2_result, dict) and set(c2_result) - C2_STAGE_FIELDS:
                            raise ValueError(
                                f"record {rec.record_id!r} C2: unknown fields are forbidden"
                            )

                        audio_path: Path | None = None
                        if c3_status == "passed":
                            audio_path_value = rec.audio.get("path")
                            if (
                                not isinstance(audio_path_value, str)
                                or not audio_path_value.strip()
                            ):
                                raise ValueError(
                                    f"{location}: invalid AudioRecord fields: "
                                    "audio.path must be a non-empty string"
                                )
                            eligible_count += 1
                            c1_result = stages.get("C1")
                            if (
                                not isinstance(c1_result, dict)
                                or c1_result.get("status") != "passed"
                            ):
                                raise ValueError(
                                    f"record {rec.record_id!r} C1: stage must exist "
                                    "with status='passed'"
                                )
                            if (
                                not isinstance(c2_result, dict)
                                or c2_result.get("status") != "passed"
                            ):
                                raise ValueError(
                                    f"record {rec.record_id!r} C2: stage must exist "
                                    "with status='passed'"
                                )
                            _, publishable = validate_stage_c_execution_contract(
                                c3_result.get("execution_mode"),
                                c3_result.get("publishable"),
                                location=f"record {rec.record_id!r} C3",
                            )
                            if not publishable:
                                raise ValueError(
                                    f"record {rec.record_id!r} C3: Stage C execution "
                                    "is not publishable"
                                )
                            validate_admission_contract(
                                c1_intelligibility=c1_result.get("intelligibility"),
                                c2_labels=labels,
                                admission_stage=c3_result,
                                policy=policy,
                                require_accepted=True,
                                location=f"record {rec.record_id!r} C3",
                                c2_stage=c2_result,
                            )
                            audio_path = Path(os.path.abspath(audio_path_value))
                            if (
                                audio_path not in private_audio_paths
                                and audio_path not in audio_snapshots
                            ):
                                while len(future_paths) >= window:
                                    reap_completed(block=True)
                                private_audio_path = private_audio_root / (
                                    f"{len(private_audio_paths):08d}{audio_path.suffix}"
                                )
                                private_audio_paths[audio_path] = private_audio_path
                                future = executor.submit(
                                    copy_source, (audio_path, private_audio_path)
                                )
                                future_paths[future] = (
                                    audio_path, private_audio_path
                                )
                                progress.total = (progress.total or 0) + 1
                                progress.refresh()

                        pending_records[ordinal] = (data, rec.record_id, audio_path)
                        reap_completed(block=False)
                        flush_ready()
                        while len(pending_records) >= window and future_paths:
                            reap_completed(block=True)

                    _, identity = self._stable_file_snapshot(
                        manifest_path,
                        include_bytes=False,
                        line_callback=consume_line,
                    )
                    manifest_snapshots[manifest_path] = identity

                while future_paths:
                    reap_completed(block=True)
                flush_ready()
                if pending_records:
                    raise RuntimeError("Stage D snapshot records could not be committed")
                snapshot.flush()
                os.fsync(snapshot.fileno())
            except BaseException:
                cancel_pending()
                raise

        if input_record_count == 0:
            raise ValueError("Stage D 输入 manifest 必须包含至少一条输入记录")
        if eligible_count == 0:
            raise ValueError("Stage D 输入没有 C3 passed eligible 记录")
        os.replace(tmp_path, snapshot_path)
        logger.info(
            "Stage D snapshot build 完成：%d 个 unique source，reflink=%d，"
            "fallback=%d，logical bytes=%d",
            len(audio_snapshots),
            copy_counts["reflink"],
            copy_counts["fallback"],
            logical_bytes,
        )
        return manifest_snapshots, audio_snapshots, snapshot_sources, record_sources

    def _run_locked(
        self,
        input_manifest: str,
        output_dir: str,
        output_manifest: str,  # 保留以对齐 StageRunner 统一接口；per-dataset 架构下
                               # 本阶段输出 = output_dir 下 d_manifest_<dataset_id>.jsonl
        global_pbar: Optional[tqdm] = None,
        source_manifest: str | None = None,
        manifest_snapshots: dict[Path, tuple] | None = None,
        audio_snapshots: dict[Path, tuple] | None = None,
        snapshot_sources: dict[str, tuple[str, tuple]] | None = None,
        record_sources: dict[str, str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        os.makedirs(output_dir, exist_ok=True)
        progress_callback = global_pbar.update if global_pbar is not None else None
        # D1/D2 每个子阶段按 eligible 数动态扩充全局 total，避免重复计数超 100%
        def total_callback(n: int) -> None:
            if global_pbar is not None:
                global_pbar.total = (global_pbar.total or 0) + n
                global_pbar.refresh()

        skip_d1 = self.stage_cfg["skip_d1"]
        writer_lock_timeout = self.stage_cfg["writer_lock_timeout"]
        final_stage, manifest_prefix = resolve_stage_d_final_stage(self.stage_cfg)
        d1_cfg = dict(self.stage_cfg.get("d1", {}))
        d2_cfg = dict(self.stage_cfg.get("d2", {}))
        # D2 响度归一化是 I/O + DSP 密集型，默认用多线程（上限 8）而非单线程
        d2_cfg.setdefault("num_workers", int(self.stage_cfg.get("num_workers", min(8, os.cpu_count() or 1))))
        if snapshot_sources is None or record_sources is None:
            raise RuntimeError("Stage D snapshot source mappings are missing")

        if skip_d1:
            # 跳过 D1 Apollo，直接用 D2 处理 Stage C 输出
            logger.info("Stage D 启用 skip_d1，跳过 D1 Apollo，直接执行 D2")
            if d1_cfg.get("post_normalize"):
                logger.info("skip_d1 为 true 时忽略 d1.post_normalize，由 D2 独立完成归一化与质检")
            final_manifest = self._run_snapshot_substage(
                processor=_SnapshotLoudnessNormalizerProcessor(
                    d2_cfg, snapshot_sources, record_sources
                ),
                input_manifest=input_manifest,
                output_dir=os.path.join(output_dir, "D2"),
                snapshot_sources=snapshot_sources,
                record_sources=record_sources,
                source_manifest=source_manifest or input_manifest,
                prev_stage="C3",
                progress_callback=progress_callback,
                total_callback=total_callback,
                writer_lock_timeout=writer_lock_timeout,
            )
        else:
            merge_d2 = bool(d1_cfg.get("post_normalize"))
            if merge_d2:
                # D1+D2 合并时，D2 的全部配置都是归一化器基线；D1 的
                # post_normalize 仅覆盖显式指定字段，避免合并路径丢失布局、
                # checkpoint、时长或峰值等 D2 行为配置。
                post_norm = dict(d1_cfg["post_normalize"]) if isinstance(d1_cfg["post_normalize"], dict) else {}
                d1_cfg["post_normalize"] = {**d2_cfg, **post_norm}

            # D1 只处理 C3 通过的样本，确保 C3 rejected 的人声/污染样本不会进入增强阶段
            d1_manifest = self._run_snapshot_substage(
                processor=_SnapshotApolloSuperResProcessor(
                    d1_cfg, snapshot_sources, record_sources
                ),
                input_manifest=input_manifest,
                output_dir=os.path.join(output_dir, "D1"),
                snapshot_sources=snapshot_sources,
                record_sources=record_sources,
                source_manifest=source_manifest or input_manifest,
                prev_stage="C3",
                progress_callback=progress_callback,
                total_callback=total_callback,
                writer_lock_timeout=writer_lock_timeout,
            )

            if merge_d2:
                # D1 已内置归一化与质检，直接作为 Stage D 最终结果
                final_manifest = d1_manifest
                logger.info("Stage D 启用 D1+D2 合并，跳过独立 D2 阶段")
            else:
                final_manifest = self._run_snapshot_substage(
                    processor=_SnapshotLoudnessNormalizerProcessor(
                        d2_cfg, snapshot_sources, record_sources
                    ),
                    input_manifest=d1_manifest,
                    output_dir=os.path.join(output_dir, "D2"),
                    snapshot_sources=snapshot_sources,
                    record_sources=record_sources,
                    source_manifest=source_manifest or input_manifest,
                    prev_stage="D1",
                    progress_callback=progress_callback,
                    total_callback=total_callback,
                    writer_lock_timeout=writer_lock_timeout,
                )

        # 最终子阶段必须至少产生一条 D2 passed；否则不得发布 stage stats，且
        # 失效子阶段完成 marker，避免零产出被下游当作成功 run 复用。
        d2_passed = 0
        for rec in read_manifest_strict(
            final_manifest,
            reject_duplicate_ids=True,
            require_audio_path=False,
        ):
            if not rec.is_stage_passed("D2"):
                continue
            audio_path = rec.audio.get("path")
            if not isinstance(audio_path, str) or not audio_path.strip():
                raise ValueError(
                    f"record {rec.record_id!r} D2 passed: "
                    "audio.path must be a non-empty string"
                )
            d2_passed += 1
        if d2_passed == 0:
            marker_path = os.path.join(
                output_dir, final_stage, f"{manifest_prefix}_output.complete"
            )
            try:
                os.remove(marker_path)
            except FileNotFoundError:
                pass
            raise RuntimeError("Stage D 最终子阶段没有 D2 passed 记录，拒绝发布")

        if source_manifest is not None:
            if manifest_snapshots is None or audio_snapshots is None:
                raise RuntimeError("Stage D source snapshot identities are missing")
            self._assert_source_snapshots(
                source_manifest, manifest_snapshots, audio_snapshots
            )

        # 汇总只消费 exact final-stage 前缀，禁止目录中其他 stage 的残留
        # manifest 被自动推断并混入 Stage D 根输出。
        if os.path.isdir(final_manifest):
            link_per_dataset_manifests(
                final_manifest, manifest_prefix, output_dir, "d"
            )
        else:
            split_manifest_by_dataset(
                src_path=final_manifest,
                dst_base=os.path.join(output_dir, "d_manifest.jsonl"),
            )

        stats_path = os.path.join(output_dir, "d_stats.json")
        stats = write_stage_stats(
            manifest_path=output_dir,
            stage_name="D",
            output_path=stats_path,
        )
        final_stage_dir = Path(output_dir, final_stage)
        manifest_files = []
        for path in sorted(final_stage_dir.glob(f"{manifest_prefix}_manifest_*.jsonl")):
            dataset_id = path.name[
                len(f"{manifest_prefix}_manifest_") : -len(".jsonl")
            ]
            validate_dataset_id(dataset_id)
            _, identity = self._stable_file_snapshot(path, include_bytes=False)
            manifest_files.append(
                {"name": path.name, "size": identity[4], "sha256": identity[0]}
            )
        if not manifest_files:
            raise RuntimeError("Stage D 最终子阶段没有 exact-prefix manifest，拒绝发布")
        _, stats_identity = self._stable_file_snapshot(
            Path(stats_path), include_bytes=False
        )
        completion = {
            "schema": self.COMPLETION_FACT_SCHEMA,
            "version": self.COMPLETION_FACT_VERSION,
            "final_stage": final_stage,
            "manifest_prefix": manifest_prefix,
            "stage_d_config_hash": config_hash(self.stage_cfg),
            "d2_passed": d2_passed,
            "manifest_files": manifest_files,
            "d_stats": {"size": stats_identity[4], "sha256": stats_identity[0]},
        }
        return output_dir, completion

