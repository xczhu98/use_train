"""
统一 manifest 读写与 schema 校验。

每个 stage 输入/输出均为 JSON Lines，单条记录称为 AudioRecord。
字段命名与结构在 README/docs 中同步说明。
"""
from __future__ import annotations

import fnmatch
import json
import hashlib
import logging
import math
import os
import re
import shutil
import stat
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any, Callable, Iterable, Set

from core.utils.file_transaction import write_text_files_transactionally

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema 定义
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = ("record_id", "dataset_id", "source_id", "audio")
AUDIO_REQUIRED_FIELDS = ("path", "sr", "duration")


def _default_stages() -> Dict[str, Dict[str, Any]]:
    return {}


@dataclass
class AudioRecord:
    """统一音频记录。

    字段说明：
      - record_id: 全局唯一记录 ID，建议格式 {dataset_id}__{source_id}__{chunk_id}
      - dataset_id: 数据集注册表中的 id
      - source_id: 原始音频文件标识
      - chunk_id: 分片标识；未分片时可与 source_id 相同或为空字符串
      - split: train/val/test
      - audio: 当前阶段音频信息（path 指向当前 stage 产物）
      - labels: 各级标签（original / audioset / leaf）
      - stages: 各阶段处理状态与中间结果
      - rejected_reason: 被拒绝原因；passed 时为 None
      - provenance: 版本、配置 hash 等可追溯信息
    """
    record_id: str
    dataset_id: str
    source_id: str
    chunk_id: str = ""
    split: str = "train"
    audio: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, Any] = field(default_factory=dict)
    stages: Dict[str, Dict[str, Any]] = field(default_factory=_default_stages)
    rejected_reason: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioRecord":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def mark_stage(
        self,
        stage_name: str,
        status: str = "passed",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """标记某阶段状态。"""
        entry = {"status": status, "timestamp": datetime.now(timezone.utc).isoformat()}
        if extra:
            entry.update(extra)
        self.stages[stage_name] = entry

    def is_stage_passed(self, stage_name: str) -> bool:
        return self.stages.get(stage_name, {}).get("status") == "passed"

    @staticmethod
    def make_record_id(dataset_id: str, source_id: str, chunk_id: str = "") -> str:
        """生成规范 record_id。"""
        cid = chunk_id or "source"
        return f"{dataset_id}__{source_id}__{cid}"


_STAGE_C_DERIVED_LABEL_KEYS = {
    "audioset",
    "topk",
    "audioset_mid",
    "leaf",
    "audioset_scores",
    "audioset_top10",
    "audioset_top5",
    "admission_decision",
    "admission_reason_codes",
    "admission_allowed",
    "admission_policy_version",
    "admission_policy_sha256",
    "admission_calibration_version",
    "admission_calibration_sha256",
    "top5_hits",
    "artifact_fingerprints",
    "label_space_sha256",
    "reason_codes",
    "review_required",
    "threshold_hits",
    "balance_category_mid",
    "balance_category",
    "balance_category_source",
    "content_kind",
    "quarantine",
    "risk",
    "max_risk",
    "margin",
    "sustain",
    "voice_max_label",
    "voice_max_prob",
    "babble_max_label",
    "babble_max_prob",
    "always_reject_max_label",
    "always_reject_max_prob",
    "background_speech_max_label",
    "background_speech_max_prob",
    "vocal_max_mid",
    "vocal_max_label",
    "vocal_max_prob",
    "foreground_speech_max_mid",
    "foreground_speech_max_label",
    "foreground_speech_max_prob",
    "background_speech_max_mid",
    "background_speech_max_label",
    "background_speech_max_prob",
    "auxiliary_reject_max_mid",
    "auxiliary_reject_max_label",
    "auxiliary_reject_max_prob",
    "background_speech_sustain_ratio",
    "background_speech_longest_run",
    "window_count",
}
_STAGE_C_DERIVED_PROVENANCE_KEYS = {
    "stage_c_execution",
    "stage_c_input_sanitization",
    "stage_c2_input_sanitization",
}


def _reject_stage_c_evidence(
    data: Dict[str, Any],
    *,
    stage_names: tuple[str, ...],
    label_keys: Set[str],
    rejected_prefixes: tuple[str, ...],
    provenance_keys: Set[str],
    seam_name: str,
) -> Dict[str, Any]:
    """拒绝 seam 前已存在的 Stage C 状态或派生准入证据。"""
    found: List[str] = []
    stages = data.get("stages")
    if isinstance(stages, dict):
        found.extend(
            f"stages.{stage_name}" for stage_name in stage_names if stage_name in stages
        )
    labels = data.get("labels")
    if isinstance(labels, dict):
        found.extend(f"labels.{key}" for key in sorted(label_keys.intersection(labels)))
    rejected_reason = data.get("rejected_reason")
    if isinstance(rejected_reason, str) and rejected_reason.startswith(rejected_prefixes):
        found.append("rejected_reason")
    provenance = data.get("provenance")
    if isinstance(provenance, dict):
        found.extend(
            f"provenance.{key}"
            for key in sorted(provenance_keys.intersection(provenance))
        )
    if found:
        raise ValueError(
            f"{seam_name} 输入记录 {data.get('record_id')!r} 含预置 Stage C 状态或派生证据: "
            + ", ".join(found)
        )
    return data


def sanitize_stage_c_input_record(data: Dict[str, Any]) -> Dict[str, Any]:
    """C1 seam 默认 fail-closed，拒绝任何预置 C1-C4 或派生准入证据。"""
    return _reject_stage_c_evidence(
        data,
        stage_names=("C1", "C2", "C3", "C4"),
        label_keys=_STAGE_C_DERIVED_LABEL_KEYS,
        rejected_prefixes=("c1_", "c2_", "c3_", "c4_"),
        provenance_keys=_STAGE_C_DERIVED_PROVENANCE_KEYS,
        seam_name="Stage C",
    )


def sanitize_stage_c2_input_record(data: Dict[str, Any]) -> Dict[str, Any]:
    """C2 seam 保留本轮可信 C1，只拒绝预置 C2-C4 与派生证据。"""
    return _reject_stage_c_evidence(
        data,
        stage_names=("C2", "C3", "C4"),
        label_keys=_STAGE_C_DERIVED_LABEL_KEYS,
        rejected_prefixes=("c2_", "c3_", "c4_"),
        provenance_keys={"stage_c_input_sanitization", "stage_c2_input_sanitization"},
        seam_name="C2",
    )


# ---------------------------------------------------------------------------
# JSONL 读写
# ---------------------------------------------------------------------------
def link_or_copy_file(src: str, dst: str) -> None:
    """以稳定源文件描述符安全发布硬链接或副本。

    临时文件始终在目标目录内创建并由本调用独占，完成后通过 ``os.replace``
    原子发布；目标旧文件不会预先删除。源文件和目标路径的所有组件都拒绝
    符号链接，复制回退从已打开的源 FD 读取，避免源路径在处理期间被替换。
    """
    source = validate_input_path(src)
    target = _absolute_path(dst)
    missing_parents: List[Path] = []
    parent = target.parent
    while not parent.exists():
        missing_parents.append(parent)
        parent = parent.parent
    _validate_existing_path_components(parent, allow_missing_leaf=False)
    for missing_parent in reversed(missing_parents):
        missing_parent.mkdir()
    _validate_existing_path_components(target, allow_missing_leaf=True)

    source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    source_flags |= getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, source_flags)
    temp_path: Optional[str] = None
    try:
        source_before = os.fstat(source_fd)
        if not stat.S_ISREG(source_before.st_mode):
            raise ValueError(f"输入路径必须是regular file: {source}")
        # 建立/删除临时硬链接会合法改变 inode ctime，因此内容身份不能纳入
        # ctime；dev/ino 绑定路径对象，size/mtime 绑定本次读取的内容版本。
        fd_stat_identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
        )
        if fd_stat_identity(source_before) != fd_stat_identity(os.stat(source, follow_symlinks=False)):
            raise ValueError(f"源文件在打开前后身份不一致: {source}")

        temp_fd, temp_path = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        os.close(temp_fd)
        os.unlink(temp_path)

        published = False
        try:
            # 优先建立临时硬链接；通过 temp FD 校验它确实指向最初打开的源 inode。
            os.link(source, temp_path)
            link_fd = os.open(temp_path, source_flags)
            try:
                if fd_stat_identity(os.fstat(link_fd)) != fd_stat_identity(source_before):
                    raise OSError("源文件在建立硬链接期间被替换")
                os.fsync(link_fd)
            finally:
                os.close(link_fd)
            published = True
        except OSError:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            temp_fd = os.open(
                temp_path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                with os.fdopen(os.dup(source_fd), "rb", closefd=True) as source_file, os.fdopen(
                    temp_fd, "wb", closefd=True
                ) as temp_file:
                    shutil.copyfileobj(source_file, temp_file)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
            except Exception:
                try:
                    os.close(temp_fd)
                except OSError:
                    pass
                raise
            published = True

        source_after = os.fstat(source_fd)
        path_after = os.stat(source, follow_symlinks=False)
        if (
            fd_stat_identity(source_after) != fd_stat_identity(source_before)
            or fd_stat_identity(path_after) != fd_stat_identity(source_before)
        ):
            raise ValueError(f"源文件在发布期间发生变化: {source}")
        if not published:
            raise OSError("未生成发布临时文件")

        # 再次检查目标叶子与祖先，确保静态符号链接不会被替换覆盖。
        _validate_existing_path_components(target, allow_missing_leaf=True)
        if os.path.lexists(target) and stat.S_ISLNK(os.lstat(target).st_mode):
            raise ValueError(f"发布目标不得是符号链接: {target}")
        os.replace(temp_path, target)
        temp_path = None
        dir_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        os.close(source_fd)
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _iter_manifest_files(path: str) -> List[Path]:
    """解析 manifest 输入为文件列表。

    支持两种输入：
      - 单文件：直接返回该文件；
      - 目录（per-dataset 模式）：返回目录下所有 ``*_manifest_*.jsonl``
        （如 ``b_manifest_<dataset_id>.jsonl``），按文件名排序保证确定性。
    目录模式下遗留单文件（``b_manifest.jsonl``，无 dataset 后缀）不会被
    返回——StageRunner 检测到该文件会显式拒绝启动（见
    ``manifest_per_dataset.reject_legacy_single_manifest``），不再自动拆分。
    """
    p = Path(path)
    if p.is_dir():
        return sorted(p.glob("*_manifest_*.jsonl"))
    return [p] if p.is_file() else []


def read_manifest(path: str) -> Iterator[AudioRecord]:
    """流式读取 manifest JSONL（支持单文件或 per-dataset 目录）。"""
    for fp in _iter_manifest_files(path):
        yield from _read_manifest_file(str(fp))


def parse_audio_record_exact(
    data: object,
    *,
    location: str,
    require_audio_path: bool = True,
) -> AudioRecord:
    """按当前 ``AudioRecord`` exact schema 解析发布边界记录。

    普通 ``from_dict`` 继续承担历史 manifest 兼容；Stage D 与 promote 等发布
    边界必须调用本函数，禁止未知字段被兼容 parser 静默截断。仅保留未落盘
    rejected/skipped 审计记录的边界可显式关闭 ``require_audio_path``；此时只允许
    空字符串作为无当前产物的哨兵值，非字符串和纯空白路径仍会被拒绝。
    """
    if not isinstance(data, dict):
        raise ValueError(f"{location}: manifest record must be a JSON object")
    expected = set(AudioRecord.__dataclass_fields__)
    missing = expected - set(data)
    unknown = set(data) - expected
    if missing or unknown:
        raise ValueError(
            f"{location}: invalid AudioRecord fields: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    try:
        for field_name in ("record_id", "dataset_id", "source_id"):
            value = data[field_name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in ("chunk_id", "split"):
            if not isinstance(data[field_name], str):
                raise ValueError(f"{field_name} must be a string")
        for field_name in ("audio", "labels", "stages", "provenance"):
            if not isinstance(data[field_name], dict):
                raise ValueError(f"{field_name} must be an object")
        rejected_reason = data["rejected_reason"]
        if rejected_reason is not None and not isinstance(rejected_reason, str):
            raise ValueError("rejected_reason must be a string or null")
        audio = data["audio"]
        missing_audio = [
            field_name for field_name in AUDIO_REQUIRED_FIELDS if field_name not in audio
        ]
        if missing_audio:
            raise ValueError(f"missing audio fields: {missing_audio}")
        audio_path = audio["path"]
        if not isinstance(audio_path, str) or (
            not audio_path.strip() and (require_audio_path or audio_path != "")
        ):
            raise ValueError("audio.path must be a non-empty string")
        sample_rate = audio["sr"]
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("audio.sr must be a positive integer")
        duration = audio["duration"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration <= 0
        ):
            raise ValueError("audio.duration must be a positive finite number")
        return AudioRecord(**data)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise ValueError(f"{location}: invalid AudioRecord fields: {exc}") from exc


def _strict_record_from_line(
    line: str,
    path: str,
    line_number: int,
    *,
    require_audio_path: bool,
) -> AudioRecord:
    """严格解析单行 manifest，并把位置附加到所有错误。"""
    location = f"{path}:{line_number}"
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location}: invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{location}: manifest record must be a JSON object")
    try:
        missing = [field_name for field_name in REQUIRED_FIELDS if field_name not in data]
        if missing:
            raise ValueError(f"missing required fields: {missing}")
        for field_name in ("record_id", "dataset_id", "source_id"):
            value = data[field_name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        audio = data["audio"]
        if not isinstance(audio, dict):
            raise ValueError("audio must be an object")
        missing_audio = [field_name for field_name in AUDIO_REQUIRED_FIELDS if field_name not in audio]
        if missing_audio:
            raise ValueError(f"missing audio fields: {missing_audio}")
        audio_path = audio["path"]
        if not isinstance(audio_path, str) or (
            not audio_path.strip() and (require_audio_path or audio_path != "")
        ):
            raise ValueError("audio.path must be a non-empty string")
        sample_rate = audio["sr"]
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("audio.sr must be a positive integer")
        duration = audio["duration"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration <= 0
        ):
            raise ValueError("audio.duration must be a positive finite number")
        return AudioRecord.from_dict(data)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise ValueError(f"{location}: invalid AudioRecord fields: {exc}") from exc


def read_manifest_strict(
    path: str,
    *,
    reject_duplicate_ids: bool = False,
    require_audio_path: bool = True,
) -> Iterator[AudioRecord]:
    """严格流式读取 manifest，并验证整个读取区间使用一致文件快照。

    与兼容性 reader 不同，本函数不静默跳过损坏输入。目录模式下可选地对
    所有 per-dataset 文件执行全局 ``record_id`` 唯一性检查。只有保留未落盘
    rejected/skipped 审计记录的边界可关闭 ``require_audio_path``。任一文件在
    读取期间被原地修改、替换、删除，或目录文件集合变化时，迭代结束前 fail closed。
    """
    files = _iter_manifest_files(path)
    if not files:
        raise ValueError(f"{path}: no manifest files found")

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    initial_names = [str(fp) for fp in files]
    stable_identities = []
    seen_ids: Set[str] = set()
    for fp in files:
        try:
            path_before = fp.stat()
            with fp.open("r", encoding="utf-8") as manifest_file:
                opened_before = os.fstat(manifest_file.fileno())
                if identity(path_before) != identity(opened_before):
                    raise OSError("file identity changed before read")
                for line_number, raw_line in enumerate(manifest_file, start=1):
                    line = raw_line.strip()
                    if not line:
                        raise ValueError(f"{fp}:{line_number}: blank JSONL line")
                    record = _strict_record_from_line(
                        line,
                        str(fp),
                        line_number,
                        require_audio_path=require_audio_path,
                    )
                    if reject_duplicate_ids:
                        if record.record_id in seen_ids:
                            raise ValueError(
                                f"{fp}:{line_number}: duplicate record_id: {record.record_id!r}"
                            )
                        seen_ids.add(record.record_id)
                    yield record
                opened_after = os.fstat(manifest_file.fileno())
            path_after = fp.stat()
        except OSError as exc:
            raise ValueError(f"{fp}: manifest snapshot changed during read: {exc}") from exc
        expected = identity(opened_before)
        if expected != identity(opened_after) or expected != identity(path_after):
            raise ValueError(f"{fp}: manifest snapshot changed during read")
        stable_identities.append(expected)

    final_files = _iter_manifest_files(path)
    if [str(fp) for fp in final_files] != initial_names:
        raise ValueError(f"{path}: manifest file set changed during read")
    for fp, expected in zip(final_files, stable_identities):
        try:
            current = identity(fp.stat())
        except OSError as exc:
            raise ValueError(f"{fp}: manifest snapshot changed after read: {exc}") from exc
        if current != expected:
            raise ValueError(f"{fp}: manifest snapshot changed during read")


def _read_manifest_file(path: str) -> Iterator[AudioRecord]:
    """流式读取单个 manifest JSONL 文件。"""
    p = Path(path)
    if not p.exists():
        return
    # errors="replace"：崩溃截断的多字节字符（中文标签 ensure_ascii=False）
    # 不得中断读取——replace 成 U+FFFD 后由 json.loads 失败路径跳过
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    # 合法 JSON 但非 object（如纯数字/数组），from_dict 会 AttributeError
                    logger.warning("manifest 行不是 JSON object，已跳过: %.100s", line)
                    continue
                yield AudioRecord.from_dict(data)
            except json.JSONDecodeError as e:
                logger.warning("manifest JSONL 解析失败: %s", e)
            except TypeError as e:
                logger.warning("manifest 字段不匹配: %s", e)


def read_manifest_with_progress(
    path: str,
    stage_name: str,
    log_interval_s: float = 30.0,
) -> Iterator[AudioRecord]:
    """流式读取 manifest JSONL，并定期打印字节级进度。

    处理百万级记录 / 数十 GB manifest 时，processor 主循环可能长时间无输出，
    容易被误判为卡死；本包装器行为与 ``read_manifest`` 一致，额外每
    ``log_interval_s`` 秒打印一次行数 / 字节百分比 / 速率 / ETA
    （格式与 stage_orchestrator 拆分阶段的进度日志对齐）。
    用 print 而非 logger：本管线未配置 logging，INFO 级日志不可见。
    """
    files = _iter_manifest_files(path)
    if not files:
        return
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    read_bytes = 0
    lines_read = 0
    t0 = time.monotonic()
    last_log = t0
    # 二进制模式读取以精确统计字节数（text 模式 tell() 是不透明 cookie）
    for p in files:
        if not p.is_file():
            continue
        with p.open("rb") as f:
            for raw in f:
                read_bytes += len(raw)
                # errors="replace"：截断的多字节字符（进程被杀后的末行）replace
                # 成 U+FFFD，交给 json.loads 失败路径跳过，而非 UnicodeDecodeError
                # 中断整个 stage
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                lines_read += 1
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        # 合法 JSON 但非 object（如纯数字/数组），from_dict 会 AttributeError
                        logger.warning("manifest 行不是 JSON object，已跳过: %.100s", line)
                        continue
                    rec = AudioRecord.from_dict(data)
                except json.JSONDecodeError as e:
                    logger.warning("manifest JSONL 解析失败: %s", e)
                    continue
                except TypeError as e:
                    logger.warning("manifest 字段不匹配: %s", e)
                    continue
                yield rec
                now = time.monotonic()
                if now - last_log >= log_interval_s:
                    last_log = now
                    elapsed = now - t0
                    rate = lines_read / max(elapsed, 1e-6)
                    pct = read_bytes / max(total_bytes, 1) * 100
                    eta_s = (total_bytes - read_bytes) / max(read_bytes / max(elapsed, 1e-6), 1e-6)
                    print(
                        f"  [{stage_name} 处理中] {lines_read} 行 ({pct:.0f}%), "
                        f"{rate:.0f} 行/s, 预计剩余 {eta_s:.0f}s",
                        flush=True,
                    )


def _absolute_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _validate_existing_path_components(path: Path, *, allow_missing_leaf: bool) -> None:
    """校验路径组件不含符号链接，避免发布目标被重定向。"""
    components = path.parts
    current = Path(components[0])
    for index, component in enumerate(components[1:], start=1):
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if allow_missing_leaf and index == len(components) - 1:
                return
            raise ValueError(f"路径组件不存在: {current}") from None
        except OSError as exc:
            raise ValueError(f"无法检查路径组件: {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"路径组件不得是符号链接: {current}")
        if index < len(components) - 1 and not stat.S_ISDIR(mode):
            raise ValueError(f"路径组件必须是目录: {current}")


def validate_input_path(path: str | Path, *, directory: bool = False) -> Path:
    """严格校验 review/manifest 输入为非符号链接的 regular file/目录。"""
    resolved = _absolute_path(path)
    _validate_existing_path_components(resolved, allow_missing_leaf=False)
    try:
        mode = os.lstat(resolved).st_mode
    except FileNotFoundError:
        raise ValueError(f"输入路径不存在: {resolved}") from None
    expected = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    if not expected:
        kind = "目录" if directory else "regular file"
        raise ValueError(f"输入路径必须是{kind}: {resolved}")
    if stat.S_ISLNK(mode):
        raise ValueError(f"输入路径不得是符号链接: {resolved}")
    return resolved


def validate_publish_target(
    path: str | Path,
    *,
    protected_paths: Iterable[str | Path] = (),
    protected_roots: Iterable[str | Path] = (),
) -> Path:
    """校验安全发布目标及其与输入/受保护路径的物理身份。"""
    target = _absolute_path(path)
    _validate_existing_path_components(target, allow_missing_leaf=True)
    if os.path.lexists(target):
        mode = os.lstat(target).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"发布目标不得是符号链接: {target}")
        if not stat.S_ISREG(mode):
            raise ValueError(f"发布目标必须是 regular file: {target}")
        if os.stat(target, follow_symlinks=False).st_nlink > 1:
            raise ValueError(f"发布目标不得是 hardlink: {target}")

    target_real = os.path.realpath(target)
    for protected_path in protected_paths:
        protected = validate_input_path(protected_path)
        if target == protected or target_real == os.path.realpath(protected):
            raise ValueError(
                f"发布目标不得覆盖受保护路径: {target}（{protected}）"
            )
        try:
            if os.path.exists(target) and os.path.exists(protected):
                if os.path.samefile(target, protected):
                    raise ValueError(
                        f"发布目标与受保护输入为同一文件: {target}（{protected}）"
                    )
        except OSError as exc:
            raise ValueError(
                f"无法确认发布目标与受保护输入的身份: {target}（{protected}）: {exc}"
            ) from exc

    for protected_root in protected_roots:
        root = validate_input_path(protected_root, directory=True)
        try:
            if os.path.commonpath((str(target), str(root))) == str(root):
                raise ValueError(
                    f"发布目标不得位于受保护输入目录内: {target}（{root}）"
                )
        except ValueError as exc:
            if "发布目标" in str(exc):
                raise
            raise ValueError(
                f"无法比较发布目标与受保护输入目录: {target}（{root}）"
            ) from exc
    return target


def write_json_file_transactionally(
    payload: Dict[str, Any],
    path: str | Path,
    *,
    protected_paths: Iterable[str | Path] = (),
    protected_roots: Iterable[str | Path] = (),
) -> Path:
    """以异常安全事务写入 JSON，并拒绝输入/输出身份冲突。"""
    target = validate_publish_target(
        path,
        protected_paths=protected_paths,
        protected_roots=protected_roots,
    )

    def _writer(output: Any) -> None:
        json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")

    write_text_files_transactionally([(target, _writer)], overwrite=True)
    return target


def write_jsonl_transactionally(
    records: Iterable[Dict[str, Any]],
    path: str | Path,
    *,
    protected_paths: Iterable[str | Path] = (),
    protected_roots: Iterable[str | Path] = (),
) -> Path:
    """以异常安全事务写入 JSONL，并拒绝输入/输出身份冲突。"""
    target = validate_publish_target(
        path,
        protected_paths=protected_paths,
        protected_roots=protected_roots,
    )

    def _writer(output: Any) -> None:
        for record in records:
            output.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )

    write_text_files_transactionally([(target, _writer)], overwrite=True)
    return target


def atomic_replace_durable(tmp_path: str, path: str) -> None:
    """``os.replace`` 后 fsync 目标目录，保证新目录项在掉电后仍然可见。

    只做 ``os.replace`` 时替换结果只存在于页缓存中，掉电后可能回退到旧
    文件或丢失 marker/断点（完成凭据丢失 = 整段重算）；与
    ``promote_run_to_final._atomic_write_json_durable`` 同一持久化口径。
    调用方负责在删除 tmp 前先对 tmp 文件本身 fsync。
    """
    os.replace(tmp_path, path)
    dir_path = os.path.dirname(os.path.abspath(path))
    dir_fd = os.open(dir_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def write_manifest(records: Iterator[AudioRecord], path: str) -> int:
    """流式写入 manifest JSONL，返回写入条数。

    tmp + 原子替换：目标可能被硬链接（stage 最终输出），原地 "w" 截断会
    同步摧毁硬链接副本；tmp+replace 产生新 inode，调用方中途失败时旧文件
    完好。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_name(p.name + ".tmp")
    count = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for rec in records:
            # allow_nan=False：NaN/Inf 是非法 JSON，读侧 parse_audio_record_exact
            # 要求 duration finite，写出非标准 JSON 只会让该行在下游整行被拒
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    os.replace(tmp_path, p)
    return count


def append_manifest(records: Iterator[AudioRecord], path: str) -> int:
    """追加写入 manifest JSONL，返回写入条数。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("a", encoding="utf-8") as f:
        for rec in records:
            # 与 write_manifest 同口径：拒绝写出非标准 JSON（NaN/Inf）
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    return count


class BatchedManifestWriter:
    """缓冲式 manifest 写入器，按 batch 刷盘，降低高频追加的小文件 I/O。

    用法：
        with BatchedManifestWriter(path, batch_size=100) as writer:
            writer.append(record)
            writer.extend(records)
        # __exit__ 会自动刷出尾部缓冲

    注意：
        - 不保证进程崩溃时缓冲区内记录落盘；但 manifest 仅作为最终产物，
          断点续跑应以 ``CheckpointTracker`` 为准。
        - 本类线程安全，可在多线程 producer 中使用。
    """

    def __init__(self, path: str, batch_size: int = 100):
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        self.path = path
        self.batch_size = int(batch_size)
        self._buffer: List[AudioRecord] = []
        self._lock = threading.Lock()
        self._closed = False

    def append(self, rec: AudioRecord) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("BatchedManifestWriter 已关闭")
            self._buffer.append(rec)
            should_flush = len(self._buffer) >= self.batch_size
        if should_flush:
            self.flush()

    def extend(self, records: Iterable[AudioRecord]) -> None:
        for rec in records:
            self.append(rec)

    def flush(self) -> None:
        # 持锁完成写入，且 close 只有在尾部缓冲落盘后才转换为 closed：
        # 这样 append 不会在 close 的 flush 期间溜入下一批而永久遗失。
        with self._lock:
            if not self._buffer:
                return
            batch = self._buffer
            self._buffer = []
            append_manifest(iter(batch), self.path)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._buffer:
                batch = self._buffer
                self._buffer = []
                append_manifest(iter(batch), self.path)
            self._closed = True

    def __enter__(self) -> "BatchedManifestWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()




# ---------------------------------------------------------------------------
# Schema 校验
# ---------------------------------------------------------------------------
def validate_record(record: AudioRecord, raise_on_error: bool = False) -> List[str]:
    """校验记录字段，返回错误列表。"""
    errors: List[str] = []
    data = record.to_dict()

    for field_name in REQUIRED_FIELDS:
        if not data.get(field_name):
            errors.append(f"缺少必填字段: {field_name}")

    audio = data.get("audio")
    if not isinstance(audio, dict):
        # audio 字段为非 dict（如 list/str）时按缺失处理，交由下方必填校验报错
        audio = {}
    for field_name in AUDIO_REQUIRED_FIELDS:
        if field_name not in audio or audio[field_name] is None:
            errors.append(f"audio 缺少必填字段: {field_name}")

    if audio.get("duration", 0) <= 0:
        errors.append("audio.duration 必须大于 0")

    if raise_on_error and errors:
        raise ValueError("; ".join(errors))
    return errors


# ---------------------------------------------------------------------------
# 增量处理辅助
# ---------------------------------------------------------------------------
def filter_by_stage(
    path: str,
    stage_name: str,
    status: str = "passed",
) -> Iterator[AudioRecord]:
    """只返回指定阶段状态符合要求的记录。"""
    for rec in read_manifest(path):
        if rec.stages.get(stage_name, {}).get("status") == status:
            yield rec


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def config_hash(config: Dict[str, Any]) -> str:
    """对配置做稳定 hash，用于 provenance。"""
    text = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


# manifest 行内上游配置指纹抽取（build_checkpoint_fingerprint 用）：
# 记录行中 "config_hash" 仅出现在 provenance 内（Stage A 写入并沿各 stage 继承）
_UPSTREAM_CONFIG_HASH_RE = re.compile(rb'"config_hash"\s*:\s*"([^"]+)"')


_ManifestIdentity = tuple[int, int, int, int, int]
_ManifestHashCache = Dict[_ManifestIdentity, tuple[str, Set[str]]]


def _manifest_content_fingerprints(
    path: str,
    content_cache: Optional[_ManifestHashCache] = None,
) -> tuple[List[Dict[str, str]], Set[str], List[str], tuple[_ManifestIdentity, ...]]:
    """经稳定 parent dirfd 计算内容摘要，并复验文件集合与物理身份。"""
    cache = content_cache if content_cache is not None else {}
    absolute = os.path.abspath(path)
    _validate_existing_path_components(Path(absolute), allow_missing_leaf=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    dir_flags = flags | getattr(os, "O_DIRECTORY", 0)

    def identity(value: os.stat_result) -> _ManifestIdentity:
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    path_info = os.lstat(absolute)
    is_directory = stat.S_ISDIR(path_info.st_mode)
    if not is_directory and not stat.S_ISREG(path_info.st_mode):
        return [], set(), [], ()
    parent_path = absolute if is_directory else os.path.dirname(absolute) or os.curdir
    parent_fd = os.open(parent_path, dir_flags)
    try:
        parent_before = os.fstat(parent_fd)
        visible_parent = os.stat(parent_path, follow_symlinks=False)
        if (parent_before.st_dev, parent_before.st_ino) != (
            visible_parent.st_dev, visible_parent.st_ino
        ):
            raise OSError(f"扫描前 manifest parent 身份变化: {parent_path}")
        if is_directory:
            names = sorted(
                name for name in os.listdir(parent_fd)
                if fnmatch.fnmatch(name, "*_manifest_*.jsonl")
            )
        else:
            names = [os.path.basename(absolute)]
        if len(names) != len(set(names)):
            raise ValueError(f"manifest 文件集合不稳定: {path}")

        fingerprints: List[Dict[str, str]] = []
        identities: List[_ManifestIdentity] = []
        upstream_hashes: Set[str] = set()
        for name in names:
            fd = os.open(name, flags, dir_fd=parent_fd)
            try:
                fd_before = os.fstat(fd)
                path_before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                current_identity = identity(fd_before)
                if not stat.S_ISREG(fd_before.st_mode) or identity(path_before) != current_identity:
                    raise OSError(f"扫描前文件身份变化或类型异常: {name}")
                cached = cache.get(current_identity)
                if cached is None:
                    digest = hashlib.sha256()
                    file_hashes: Set[str] = set()
                    with os.fdopen(os.dup(fd), "rb", closefd=True) as source:
                        for raw in source:
                            digest.update(raw)
                            match = _UPSTREAM_CONFIG_HASH_RE.search(raw)
                            file_hashes.add(
                                match.group(1).decode("utf-8") if match else ""
                            )
                    cached = digest.hexdigest(), file_hashes
                    cache[current_identity] = cached
                fd_after = os.fstat(fd)
                path_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            finally:
                os.close(fd)
            if identity(fd_after) != current_identity or identity(path_after) != current_identity:
                raise OSError(f"扫描期间文件发生变化: {name}")
            digest_hex, file_hashes = cached
            identities.append(current_identity)
            upstream_hashes.update(file_hashes)
            fingerprints.append({"name": name, "sha256": digest_hex})

        if is_directory:
            final_names = sorted(
                name for name in os.listdir(parent_fd)
                if fnmatch.fnmatch(name, "*_manifest_*.jsonl")
            )
            if final_names != names:
                raise OSError(f"扫描期间 manifest 文件集合发生变化: {path}")
        for name, expected in zip(names, identities):
            if identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != expected:
                raise OSError(f"扫描后文件身份变化: {name}")
        parent_after = os.fstat(parent_fd)
        visible_parent_after = os.stat(parent_path, follow_symlinks=False)
        if (
            (parent_after.st_dev, parent_after.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
            or (visible_parent_after.st_dev, visible_parent_after.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
        ):
            raise OSError(f"扫描期间 manifest parent 身份变化: {parent_path}")
    finally:
        os.close(parent_fd)

    identity_labels = [
        f"{name}:{','.join(str(item) for item in file_identity)}"
        for name, file_identity in zip(names, identities)
    ]
    return fingerprints, upstream_hashes, identity_labels, tuple(sorted(identities))


def build_checkpoint_fingerprint(
    local_config: Dict[str, Any],
    input_manifest: str,
    stat_input: Optional[str] = None,
) -> str:
    """构建 checkpoint 配置指纹：本阶段配置 + 上游配置指纹。

    单用 ``config_hash(local_config)`` 无法感知上游阶段配置变更：改 B2 静音
    参数等会改变本阶段输入内容（如 chunk 波形），但本阶段 config_hash 不变，
    checkpoint 会恢复基于旧输入的结果，造成静默错误。

    上游指纹取自输入 manifest 记录的 ``provenance.config_hash``（Stage A 写
    入的全量 pipeline 配置 hash，并沿 B/C/D 继承）。上游任一配置变更都会
    改变该值，使下游 checkpoint 失效、全量重算——宁可重算也不静默用旧结果。

    仅靠 provenance.config_hash 无法感知"上游重算但配置未变"：该值只在
    Stage A 写入、B/C/D 只继承不刷新，上游 stage 重跑（如 D1 参数变更后
    产物路径不变、内容变化）后下游指纹不变，checkpoint 会静默恢复旧结果。
    因此同步计入 manifest 文件集合的内容 SHA-256：上游任何 stage
    重算导致内容变化时，下游指纹随之失效，即使 size/mtime 被伪造也不能复用。

    ``stat_input``：文件统计的 stat 目标，默认与 ``input_manifest`` 相同。
    子阶段（C1/C2/C3/D1 走 orchestrator 拆分）的 input_manifest 是每次运行
    重新生成的 ``{stage}_input_eligible.jsonl``——崩溃后同内容重跑其 mtime
    必变，若直接对其 stat 会让断点每次必然失效全量重算；此时 stat_input
    应传稳定的上游产物路径（orchestrator 的原始 input_manifest），其内容
    决定 eligible 内容，mtime 只在真正重算上游时才变化。

    输入为空/缺 provenance 时仍计入空 provenance 集合与内容摘要，保持可区分性。

    性能：逐文件二进制读取 + 行级正则抽取，免整行 JSON 解析与 AudioRecord
    构造（百万级记录 × 大 JSON 行时全量构造是分钟级开销）。行内无匹配
    （无 provenance / config_hash 缺失或为 null）与旧实现一样计入空串，
    指纹语义与 AudioRecord 构造版本完全等价。
    """
    content_cache: _ManifestHashCache = {}
    (
        input_fingerprints,
        upstream_hashes,
        input_identities,
        input_collection_identity,
    ) = _manifest_content_fingerprints(input_manifest, content_cache)
    stat_path = stat_input if stat_input else input_manifest
    if os.path.abspath(stat_path) != os.path.abspath(input_manifest):
        (
            stat_fingerprints,
            stat_hashes,
            stat_identities,
            stat_collection_identity,
        ) = _manifest_content_fingerprints(stat_path, content_cache)
        if stat_collection_identity != input_collection_identity:
            input_fingerprints.extend(
                {"name": f"stat:{item['name']}", "sha256": item["sha256"]}
                for item in stat_fingerprints
            )
            upstream_hashes.update(stat_hashes)
            input_identities.extend(f"stat:{item}" for item in stat_identities)
    upstream_hashes.add(
        "content:" + hashlib.sha256(
            json.dumps(input_fingerprints, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    # identity 仅用于扫描期间 fail-closed；稳定 checkpoint 指纹绑定内容摘要，
    # 不因同内容原子重写或 mtime 变化而失效。
    upstream_collection = config_hash({"upstream_hashes": sorted(upstream_hashes)})
    return f"{config_hash(local_config)}::{upstream_collection}"


def count_manifest(path: str) -> int:
    """统计 manifest 行数。"""
    n = 0
    for _ in read_manifest(path):
        n += 1
    return n

def read_records_by_id(
    path: str,
    predicate: Optional[Callable[[AudioRecord], bool]] = None,
) -> Dict[str, AudioRecord]:
    """读取 manifest 并按 record_id 建索引。

    用于 checkpoint resume：processor 重跑时可从上次输出中恢复带阶段标记
    与更新后路径的记录，避免已处理记录退回“未处理”状态。

    predicate: 可选过滤函数，返回 False 的记录不进入索引。resume 场景建议
    传 `lambda rec: ckpt.is_done(rec.record_id)`，只物化 checkpoint 确认过
    的记录，避免百万级 manifest 全量驻留内存。
    """
    records: Dict[str, AudioRecord] = {}
    for rec in read_manifest(path):
        if predicate is not None and not predicate(rec):
            continue
        records[rec.record_id] = rec
    return records
