"""Generation 3 安全 source descriptor 解码边界。"""

from __future__ import annotations

import hashlib
import math
import numbers
import os
import re
import stat
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import soundfile as sf

from ...utils.speaker_metadata import validate_source_relative_path
from .aggregate_source import (
    AggregateDescriptor,
    AggregateStreamReader,
    decode_aggregate_full,
    decode_aggregate_interval,
    scan_aggregate_directory,
    verify_aggregate_identity,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SNAPSHOT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
    "st_nlink",
)
_FLOAT32_ITEMSIZE = np.dtype(np.float32).itemsize
# Step1 驻留解码护栏：只统计常驻的第 1 通道（索引 0）mono waveform
# （frames × float32）。FLAC 帧内通道交织，解码器每块仍 transiently 解出
# 全部通道，但除 channel 0 外立即释放，因此 8 通道 16 kHz 约 40 分钟的
# 会议源（驻留 ~145 MiB）与 48 kHz 双通道约 85 分钟源都在护栏内。
MAX_STEP1_DECODED_BYTES = 512 * 1024 * 1024
MAX_STEP1_RAW_SOURCE_BYTES = 512 * 1024 * 1024
MAX_STEP1_VAD_ANALYSIS_BYTES = 256 * 1024 * 1024
# Step1 分块解码块长（帧数）：8 通道 float32 下每块 transient 约 8 MiB。
_STEP1_DECODE_BLOCK_FRAMES = 262_144
# 单区间解码护栏：64 MiB 与 Step2 的 60s 段长上限配套（≤60s 段在
# 48 kHz 2/3/4 通道下全通道解码最多 ~46 MiB）；区间解码按导出/打分需要
# 解出全部通道，通道 0 在解码后提取。超限 fail-closed。
MAX_STEP3_INTERVAL_DECODED_BYTES = 64 * 1024 * 1024
MIN_STEP1_SOURCE_SAMPLE_RATE = 8_000
MAX_STEP1_SOURCE_SAMPLE_RATE = 192_000


@dataclass(frozen=True)
class SourceDecodeParameters:
    """Step1 完整 waveform 解码与 VAD 输入资源边界。

    max_decoded_bytes 绑定 Step1 常驻的 channel 0 mono float32 waveform
    （frames × 4 字节），不统计 transient 的全通道解码块。
    """

    max_decoded_bytes: int
    max_vad_analysis_bytes: int
    min_source_sample_rate: int
    max_source_sample_rate: int
    max_raw_source_bytes: int = MAX_STEP1_RAW_SOURCE_BYTES

    def __post_init__(self) -> None:
        for field in (
            "max_decoded_bytes",
            "max_raw_source_bytes",
            "max_vad_analysis_bytes",
            "min_source_sample_rate",
            "max_source_sample_rate",
        ):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive Python integer")
        if self.min_source_sample_rate > self.max_source_sample_rate:
            raise ValueError("source sample rate range must be ordered")


STEP1_SOURCE_DECODE_PARAMETERS = SourceDecodeParameters(
    max_decoded_bytes=MAX_STEP1_DECODED_BYTES,
    max_raw_source_bytes=MAX_STEP1_RAW_SOURCE_BYTES,
    max_vad_analysis_bytes=MAX_STEP1_VAD_ANALYSIS_BYTES,
    min_source_sample_rate=MIN_STEP1_SOURCE_SAMPLE_RATE,
    max_source_sample_rate=MAX_STEP1_SOURCE_SAMPLE_RATE,
)


class SourceVerificationError(ValueError):
    """源 descriptor、pathname binding 或已持久化元数据验证失败。"""


def _before_decode_test_hook() -> None:
    """确定性 binding 故障注入 seam；生产默认 no-op。"""


def _post_decode_test_hook() -> None:
    """确定性 binding 故障注入 seam；生产默认 no-op。"""


def _add_cleanup_note(primary: BaseException, label: str, error: BaseException) -> None:
    note = f"cleanup failed for {label}: {error}"
    add_note = getattr(primary, "add_note", None)
    if add_note is not None:
        add_note(note)
    else:
        notes = list(getattr(primary, "__notes__", ()))
        notes.append(note)
        primary.__notes__ = notes


def _close_fd(
    descriptor: int,
    label: str,
    primary: BaseException | None = None,
    *,
    category: str = "binding",
) -> None:
    try:
        os.close(descriptor)
    except BaseException as exc:
        if primary is not None:
            _add_cleanup_note(primary, label, exc)
            return
        raise SourceVerificationError(
            f"{category} cleanup failed for {label}: {exc}"
        ) from exc


def _close_fds(
    resources: list[tuple[int, str, str]],
    primary: BaseException | None = None,
) -> None:
    cleanup_primary = primary
    first_cleanup_error = None
    for descriptor, label, category in resources:
        try:
            _close_fd(descriptor, label, cleanup_primary, category=category)
        except BaseException as exc:
            # 只保留首个清理错误：后续错误挂到它上面，避免中间错误被静默覆盖。
            if first_cleanup_error is None:
                first_cleanup_error = exc
                cleanup_primary = exc
            else:
                _add_cleanup_note(first_cleanup_error, label, exc)
    if primary is None and first_cleanup_error is not None:
        raise first_cleanup_error


def _close_soundfile(stream: Any, primary: BaseException | None = None) -> None:
    try:
        stream.close()
    except BaseException as exc:
        if primary is not None:
            _add_cleanup_note(primary, "SoundFile", exc)
            return
        raise SourceVerificationError(f"decode cleanup failed for SoundFile: {exc}") from exc


@dataclass(frozen=True)
class DecodedSource:
    """Step1 已验证解码结果；audio 为第 1 通道（Python 索引 0）的 mono
    float32 连续一维数组（frames,），完整通道数仅作为 header 元数据保留。"""

    audio: np.ndarray
    sample_rate: int
    frames: int
    channels: int
    format: str
    subtype: str
    source_size_bytes: int
    source_sha256: str
    vad_capacity_samples: int | None = None


def _require_posix_descriptor_support() -> None:
    required_flags = ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")
    missing = [name for name in required_flags if not hasattr(os, name)]
    if os.name != "posix" or missing:
        detail = ", ".join(missing) if missing else "POSIX"
        raise SourceVerificationError(f"binding verification requires {detail}")
    if not hasattr(os, "pread") or os.open not in os.supports_dir_fd:
        raise SourceVerificationError("binding verification requires pread and open(dir_fd=...)")


def _row_value(row: Mapping[str, Any], field: str) -> Any:
    try:
        return row[field]
    except (KeyError, TypeError) as exc:
        raise SourceVerificationError(f"missing {field} for source verification") from exc


def _validate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise SourceVerificationError("source row must be a mapping")

    input_root = _row_value(row, "input_root")
    if type(input_root) is not str or not input_root:
        raise SourceVerificationError("input_root must be a non-empty canonical string")
    canonical_root = os.path.realpath(os.path.abspath(input_root))
    if input_root != canonical_root:
        raise SourceVerificationError("input_root must equal its canonical realpath")

    relative_path = _row_value(row, "source_relative_path")
    try:
        relative_path = validate_source_relative_path(relative_path)
    except ValueError as exc:
        raise SourceVerificationError(f"source_relative_path validation failed: {exc}") from exc

    source_size = _row_value(row, "source_size_bytes")
    if type(source_size) is not int or source_size < 0:
        raise SourceVerificationError("source size must be a non-negative integer")

    source_sha256 = _row_value(row, "source_sha256")
    if type(source_sha256) is not str or _SHA256_RE.fullmatch(source_sha256) is None:
        raise SourceVerificationError("source sha256 must be 64 lowercase hexadecimal characters")

    header: dict[str, Any] = {}
    for field in ("sample_rate", "frames", "channels"):
        value = _row_value(row, field)
        if type(value) is not int or value <= 0:
            raise SourceVerificationError(f"header {field} must be a positive integer")
        header[field] = value

    duration = _row_value(row, "duration_sec")
    if isinstance(duration, bool) or not isinstance(duration, numbers.Real):
        raise SourceVerificationError("header duration_sec must be a finite positive real")
    duration = float(duration)
    expected_duration = header["frames"] / header["sample_rate"]
    if not math.isfinite(duration) or duration <= 0 or duration != expected_duration:
        raise SourceVerificationError("header duration_sec does not match frames / sample_rate")
    header["duration_sec"] = duration

    for field in ("format", "subtype"):
        value = _row_value(row, field)
        if type(value) is not str or not value.strip():
            raise SourceVerificationError(f"header {field} must be a non-empty string")
        header[field] = value

    return {
        "input_root": input_root,
        "source_relative_path": relative_path,
        "source_size_bytes": source_size,
        "source_sha256": source_sha256,
        **header,
    }


def _snapshot(snapshot: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return tuple(getattr(snapshot, field) for field in _SNAPSHOT_FIELDS)


def _dup_fd(descriptor: int, category: str, label: str) -> int:
    try:
        return os.dup(descriptor)
    except OSError as exc:
        raise SourceVerificationError(f"{category} cannot duplicate {label}: {exc}") from exc


def _fstat(descriptor: int, category: str, label: str) -> os.stat_result:
    try:
        return os.fstat(descriptor)
    except OSError as exc:
        raise SourceVerificationError(f"{category} cannot inspect {label}: {exc}") from exc


def _open_component(component: str, flags: int, parent_fd: int, label: str) -> int:
    try:
        return os.open(component, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise SourceVerificationError(f"binding cannot safely open {label}: {exc}") from exc


def _walk_from_descriptor(
    anchor_fd: int,
    components: list[str],
    terminal_flags: int,
    label: str,
) -> int:
    parent_fd = _dup_fd(anchor_fd, "binding", f"{label} anchor")
    terminal_fd = None
    try:
        directory_flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_DIRECTORY
        for index, component in enumerate(components[:-1]):
            child_fd = _open_component(
                component,
                directory_flags,
                parent_fd,
                f"{label} component {index} {component!r}",
            )
            old_parent = parent_fd
            parent_fd = child_fd
            _close_fd(old_parent, f"{label} parent component")
        terminal_fd = _open_component(
            components[-1], terminal_flags, parent_fd, f"{label} terminal"
        )
        terminal_parent_fd = parent_fd
        parent_fd = None
        try:
            _close_fd(terminal_parent_fd, f"{label} terminal parent")
        except BaseException as primary:
            owned_terminal_fd = terminal_fd
            terminal_fd = None
            _close_fd(owned_terminal_fd, f"{label} terminal", primary)
            raise
        result_fd = terminal_fd
        terminal_fd = None
        return result_fd
    except BaseException as primary:
        resources = []
        if terminal_fd is not None:
            resources.append((terminal_fd, f"{label} terminal", "binding"))
        if parent_fd is not None:
            resources.append((parent_fd, f"{label} active parent", "binding"))
        _close_fds(resources, primary)
        raise


def _open_directory_source(
    root_fd: int,
    relative_path: str,
    *,
    aggregate: bool = False,
) -> int:
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    if aggregate:
        flags |= os.O_DIRECTORY
    return _walk_from_descriptor(
        root_fd,
        relative_path.split("/"),
        flags,
        "source binding",
    )


def _absolute_components(path: str) -> list[str]:
    components = path.split("/")[1:]
    if not components or any(component in ("", ".", "..") for component in components):
        raise SourceVerificationError("binding absolute path has invalid components")
    return components


def _open_absolute_binding(slash_fd: int, path: str, terminal_flags: int) -> int:
    if path == "/":
        return _dup_fd(slash_fd, "binding", "filesystem root")
    return _walk_from_descriptor(
        slash_fd, _absolute_components(path), terminal_flags, "input_root binding"
    )


def _open_root(input_root: str) -> tuple[int, int, os.stat_result, bool]:
    try:
        slash_fd = os.open(
            "/", os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_DIRECTORY
        )
    except OSError as exc:
        raise SourceVerificationError(f"binding cannot open absolute root anchor: {exc}") from exc
    descriptor = None
    probe_fd = None
    try:
        descriptor = _open_absolute_binding(
            slash_fd, input_root, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
        )
        held = _fstat(descriptor, "binding", "input_root descriptor")
        single_file = stat.S_ISREG(held.st_mode)
        if not single_file and not stat.S_ISDIR(held.st_mode):
            raise SourceVerificationError(
                "input_root binding must name a regular file or directory"
            )
        if not single_file:
            directory_fd = _open_absolute_binding(
                slash_fd,
                input_root,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_DIRECTORY,
            )
            probe_snapshot = held
            probe_fd = descriptor
            descriptor = directory_fd
            held = _fstat(descriptor, "binding", "input_root directory")
            if _snapshot(held) != _snapshot(probe_snapshot):
                raise SourceVerificationError(
                    "input_root binding changed between probe and held open"
                )
            owned_probe_fd = probe_fd
            probe_fd = None
            _close_fd(owned_probe_fd, "input_root non-directory probe")
        return slash_fd, descriptor, held, single_file
    except BaseException as primary:
        resources = []
        if probe_fd is not None:
            resources.append((probe_fd, "input_root non-directory probe", "binding"))
        if descriptor is not None:
            resources.append((descriptor, "input_root descriptor", "binding"))
        resources.append((slash_fd, "absolute root anchor", "binding"))
        _close_fds(resources, primary)
        raise


def _open_source(
    root_fd: int,
    root_snapshot: os.stat_result,
    single_file: bool,
    input_root: str,
    relative_path: str,
) -> tuple[int, os.stat_result]:
    if single_file:
        if relative_path != input_root.lstrip("/"):
            raise SourceVerificationError(
                "source binding does not match regular-file input_root"
            )
        descriptor = _dup_fd(root_fd, "binding", "regular-file source")
    else:
        descriptor = _open_directory_source(root_fd, relative_path)
    try:
        opened = _fstat(descriptor, "binding", "source descriptor")
        if single_file and not stat.S_ISREG(opened.st_mode):
            raise SourceVerificationError("source binding must resolve to a regular file")
        if single_file and _snapshot(opened) != _snapshot(root_snapshot):
            raise SourceVerificationError(
                "source binding differs from regular-file input_root"
            )
        if (
            not single_file
            and not stat.S_ISDIR(opened.st_mode)
            and not stat.S_ISREG(opened.st_mode)
        ):
            raise SourceVerificationError(
                "source binding must resolve to a regular file or aggregate directory"
            )
        return descriptor, opened
    except BaseException as primary:
        _close_fd(descriptor, "source descriptor", primary)
        raise


def _scan_open_aggregate(
    descriptor: int,
    input_root: str,
    relative_path: str,
    expected: Mapping[str, Any],
) -> AggregateDescriptor | None:
    """从已安全打开的 terminal descriptor 识别并绑定 aggregate manifest。"""
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        return None
    try:
        aggregate = scan_aggregate_directory(descriptor, input_root, relative_path)
    except ValueError as exc:
        raise SourceVerificationError(
            f"aggregate manifest scan failed: {exc}"
        ) from exc
    expected_fields = {
        "source_size_bytes": aggregate.source_size_bytes,
        "source_sha256": aggregate.manifest_sha256,
        "sample_rate": aggregate.sample_rate,
        "frames": aggregate.frames,
        "channels": aggregate.channels,
        "format": aggregate.format,
        "subtype": aggregate.subtype,
    }
    for field, actual in expected_fields.items():
        if expected[field] != actual:
            raise SourceVerificationError(
                f"aggregate {field} mismatch: expected {expected[field]!r}, got {actual!r}"
            )
    return aggregate


def _hash_source_descriptor(descriptor: int, expected_size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < expected_size:
        remaining = expected_size - offset
        try:
            chunk = os.pread(descriptor, min(1024 * 1024, remaining), offset)
        except OSError as exc:
            raise SourceVerificationError(
                f"source sha256 read failed at offset {offset}: {exc}"
            ) from exc
        if not chunk:
            raise SourceVerificationError(
                "source sha256 reached EOF before expected size"
            )
        digest.update(chunk)
        offset += len(chunk)

    try:
        extra = os.pread(descriptor, 1, expected_size)
    except OSError as exc:
        raise SourceVerificationError(
            f"source sha256 size probe failed at offset {expected_size}: {exc}"
        ) from exc
    if extra:
        raise SourceVerificationError(
            "source sha256 has extra data beyond expected size"
        )
    return digest.hexdigest()


def _spool_verified_source(
    descriptor: int,
    expected_size: int,
    *,
    expected_sha256: str,
) -> tempfile.SpooledTemporaryFile[bytes]:
    """冻结 held source 到受限磁盘 spool，供哈希和解码复用同一内容。"""

    spool = tempfile.SpooledTemporaryFile(max_size=1, mode="w+b")
    spool.rollover()
    digest = hashlib.sha256()
    offset = 0
    try:
        while offset < expected_size:
            remaining = expected_size - offset
            try:
                chunk = os.pread(descriptor, min(1024 * 1024, remaining), offset)
            except OSError as exc:
                raise SourceVerificationError(
                    f"source sha256 read failed at offset {offset}: {exc}"
                ) from exc
            if not chunk:
                raise SourceVerificationError(
                    "source sha256 reached EOF before expected size"
                )
            digest.update(chunk)
            spool.write(chunk)
            offset += len(chunk)
        try:
            extra = os.pread(descriptor, 1, expected_size)
        except OSError as exc:
            raise SourceVerificationError(
                f"source sha256 size probe failed at offset {expected_size}: {exc}"
            ) from exc
        if extra:
            raise SourceVerificationError(
                "source sha256 has extra data beyond expected size"
            )
        if digest.hexdigest() != expected_sha256:
            raise SourceVerificationError("source sha256 mismatch before decode")
        spool.seek(0)
        return spool
    except BaseException as primary:
        try:
            spool.close()
        except BaseException as cleanup_error:
            _add_cleanup_note(primary, "raw source spool", cleanup_error)
        raise


def _validate_raw_source_bytes(
    source_size_bytes: int,
    *,
    parameters: SourceDecodeParameters,
) -> None:
    if source_size_bytes > parameters.max_raw_source_bytes:
        raise SourceVerificationError(
            "source resource limit exceeded: raw source bytes require "
            f"{source_size_bytes} bytes, limit is "
            f"{parameters.max_raw_source_bytes} bytes"
        )


def _validate_decode_resources(
    metadata: dict[str, Any],
    *,
    parameters: SourceDecodeParameters,
    vad_target_sample_rate: int,
) -> None:
    source_sample_rate = metadata["sample_rate"]
    if not (
        parameters.min_source_sample_rate
        <= source_sample_rate
        <= parameters.max_source_sample_rate
    ):
        raise SourceVerificationError(
            "source sample rate is outside the configured range: "
            f"{source_sample_rate} not in "
            f"[{parameters.min_source_sample_rate}, "
            f"{parameters.max_source_sample_rate}]"
        )

    decoded_bytes = metadata["frames"] * _FLOAT32_ITEMSIZE
    if decoded_bytes > parameters.max_decoded_bytes:
        raise SourceVerificationError(
            "source resource limit exceeded: decoded channel-0 mono float32 "
            f"waveform requires {decoded_bytes} bytes, limit is "
            f"{parameters.max_decoded_bytes} bytes"
        )

    if source_sample_rate == vad_target_sample_rate:
        vad_capacity_samples = metadata["frames"]
    else:
        scaled_frames = metadata["frames"] * vad_target_sample_rate
        vad_capacity_samples = (
            scaled_frames + source_sample_rate - 1
        ) // source_sample_rate + 1
    vad_capacity_bytes = vad_capacity_samples * _FLOAT32_ITEMSIZE
    if vad_capacity_bytes > parameters.max_vad_analysis_bytes:
        raise SourceVerificationError(
            "VAD analysis resource limit exceeded: float32 waveform capacity "
            f"requires {vad_capacity_bytes} bytes, limit is "
            f"{parameters.max_vad_analysis_bytes} bytes"
        )
    metadata["vad_capacity_samples"] = vad_capacity_samples


def _aggregate_metadata(
    aggregate: AggregateDescriptor,
    expected: Mapping[str, Any],
    *,
    parameters: SourceDecodeParameters,
    vad_target_sample_rate: int,
) -> dict[str, Any]:
    metadata = {
        "sample_rate": aggregate.sample_rate,
        "frames": aggregate.frames,
        "channels": aggregate.channels,
        "format": aggregate.format,
        "subtype": aggregate.subtype,
    }
    _verify_header(metadata, expected)
    _validate_decode_resources(
        metadata,
        parameters=parameters,
        vad_target_sample_rate=vad_target_sample_rate,
    )
    return metadata


def _decode_source_descriptor(
    descriptor: int,
    *,
    expected: Mapping[str, Any] | None = None,
    parameters: SourceDecodeParameters = STEP1_SOURCE_DECODE_PARAMETERS,
    vad_target_sample_rate: int = 16_000,
    raw_source: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not isinstance(parameters, SourceDecodeParameters):
        raise SourceVerificationError(
            "source decode parameters must be SourceDecodeParameters"
        )
    if type(vad_target_sample_rate) is not int or vad_target_sample_rate <= 0:
        raise SourceVerificationError(
            "vad_target_sample_rate must be a positive Python integer"
        )
    if raw_source is not None and not callable(getattr(raw_source, "read", None)):
        raise SourceVerificationError("raw_source must be a readable file object or None")
    # raw_source 使哈希与解码共用同一份冻结的内容，消除两次读之间的 TOCTOU 窗口。
    # 未提供 raw_source 时使用 descriptor 流式读取。
    stream = None
    duplicate = None
    try:
        if raw_source is not None:
            try:
                raw_source.seek(0)
                stream = sf.SoundFile(raw_source, mode="r")
            except Exception as exc:
                raise SourceVerificationError(
                    f"decode failed while opening source header: {exc}"
                ) from exc
        else:
            duplicate = _dup_fd(descriptor, "decode", "source descriptor")
            try:
                stream = sf.SoundFile(duplicate, mode="r", closefd=False)
            except Exception as exc:
                raise SourceVerificationError(
                    f"decode failed while opening source header: {exc}"
                ) from exc
        try:
            metadata = {
                "sample_rate": stream.samplerate,
                "frames": stream.frames,
                "channels": stream.channels,
                "format": stream.format,
                "subtype": stream.subtype,
            }
        except Exception as exc:
            raise SourceVerificationError(f"header read failed: {exc}") from exc
        for field in ("sample_rate", "frames", "channels"):
            if type(metadata[field]) is not int or metadata[field] <= 0:
                raise SourceVerificationError(
                    f"header {field} must be a positive Python integer"
                )
        if any(
            type(metadata[field]) is not str or not metadata[field].strip()
            for field in ("format", "subtype")
        ):
            raise SourceVerificationError("header contains invalid format or subtype")
        if expected is not None:
            _verify_header(metadata, expected)

        source_sample_rate = metadata["sample_rate"]
        if not (
            parameters.min_source_sample_rate
            <= source_sample_rate
            <= parameters.max_source_sample_rate
        ):
            raise SourceVerificationError(
                "source sample rate is outside the configured range: "
                f"{source_sample_rate} not in "
                f"[{parameters.min_source_sample_rate}, "
                f"{parameters.max_source_sample_rate}]"
            )

        decoded_bytes = metadata["frames"] * _FLOAT32_ITEMSIZE
        if decoded_bytes > parameters.max_decoded_bytes:
            raise SourceVerificationError(
                "source resource limit exceeded: decoded channel-0 mono float32 "
                f"waveform requires {decoded_bytes} bytes, limit is "
                f"{parameters.max_decoded_bytes} bytes"
            )

        if source_sample_rate == vad_target_sample_rate:
            vad_capacity_samples = metadata["frames"]
        else:
            scaled_frames = metadata["frames"] * vad_target_sample_rate
            vad_capacity_samples = (
                scaled_frames + source_sample_rate - 1
            ) // source_sample_rate + 1
        vad_capacity_bytes = vad_capacity_samples * _FLOAT32_ITEMSIZE
        if vad_capacity_bytes > parameters.max_vad_analysis_bytes:
            raise SourceVerificationError(
                "VAD analysis resource limit exceeded: float32 waveform capacity "
                f"requires {vad_capacity_bytes} bytes, limit is "
                f"{parameters.max_vad_analysis_bytes} bytes"
            )
        metadata["vad_capacity_samples"] = vad_capacity_samples

        # FLAC 帧内通道交织，libsndfile 按帧解出全部通道；Step1 与下游
        # 只消费第 1 通道（索引 0），因此分块读取并只把 channel 0 写入
        # 预分配 mono 数组，其余通道随块释放。单块 transient 内存由
        # _STEP1_DECODE_BLOCK_FRAMES 与 header 通道数约束。
        expected_shape = (metadata["frames"],)
        expected_channels = metadata["channels"]
        mono = np.empty(expected_shape, dtype=np.float32)
        offset = 0
        while offset < expected_shape[0]:
            try:
                block = stream.read(
                    _STEP1_DECODE_BLOCK_FRAMES,
                    dtype="float32",
                    always_2d=True,
                )
            except Exception as exc:
                raise SourceVerificationError(
                    f"decode failed while reading waveform: {exc}"
                ) from exc
            if not isinstance(block, np.ndarray) or block.ndim != 2:
                raise SourceVerificationError("decode returned a malformed block")
            rows = int(block.shape[0])
            if rows == 0:
                break
            if int(block.shape[1]) != expected_channels:
                raise SourceVerificationError(
                    "decode block channel count mismatch: expected "
                    f"{expected_channels}, got {int(block.shape[1])}"
                )
            if rows > expected_shape[0] - offset:
                raise SourceVerificationError("decode read exceeded header frames")
            mono[offset : offset + rows] = block[:, 0]
            offset += rows
        if offset != expected_shape[0]:
            raise SourceVerificationError(
                f"decode short read: expected {expected_shape[0]} frames, "
                f"got {offset}"
            )
        audio = mono
    except BaseException as primary:
        if stream is not None:
            _close_soundfile(stream, primary)
        if duplicate is not None:
            _close_fd(duplicate, "decode duplicate", primary, category="decode")
        raise
    else:
        try:
            _close_soundfile(stream)
        except BaseException as primary:
            if duplicate is not None:
                _close_fd(duplicate, "decode duplicate", primary, category="decode")
            raise
        try:
            # mono 始终是本函数预分配的自有数组，天然满足 owned/writable
            # 契约，不再依赖 SoundFile.read 返回数组的 OWNDATA 快路径。
            if (
                type(audio) is not np.ndarray
                or audio.shape != expected_shape
                or audio.dtype != np.dtype(np.float32)
                or not audio.flags.c_contiguous
                or not audio.flags.owndata
                or not audio.flags.writeable
                or audio.base is not None
            ):
                raise SourceVerificationError(
                    "decode result does not satisfy mutable owned array contract"
                )
        except BaseException as primary:
            if duplicate is not None:
                _close_fd(duplicate, "decode duplicate", primary, category="decode")
            raise
        if duplicate is not None:
            _close_fd(duplicate, "decode duplicate", category="decode")
        return audio, metadata


def _verify_header(metadata: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for field in ("sample_rate", "frames", "channels", "format", "subtype"):
        if metadata[field] != expected[field]:
            raise SourceVerificationError(
                f"header {field} mismatch: expected {expected[field]!r}, got {metadata[field]!r}"
            )


def _verify_snapshot(
    descriptor: int,
    expected: os.stat_result,
    label: str,
    *,
    regular_file: bool = False,
) -> None:
    current = _fstat(descriptor, "binding", label)
    if regular_file and not stat.S_ISREG(current.st_mode):
        raise SourceVerificationError(f"{label} binding is not a regular file")
    if _snapshot(current) != _snapshot(expected):
        raise SourceVerificationError(f"{label} binding changed")


def _verify_post_decode_bindings(
    source_fd: int,
    source_snapshot: os.stat_result,
    root_fd: int,
    root_snapshot: os.stat_result,
    slash_fd: int,
    input_root: str,
    relative_path: str,
    single_file: bool,
    aggregate: AggregateDescriptor | None = None,
    *,
    aggregate_full_scan: bool = True,
) -> None:
    _verify_snapshot(
        source_fd,
        source_snapshot,
        "source held descriptor",
        regular_file=aggregate is None,
    )
    if aggregate is not None:
        if not stat.S_ISDIR(source_snapshot.st_mode):
            raise SourceVerificationError("aggregate source held descriptor is not a directory")
        try:
            verify_aggregate_identity(
                source_fd,
                aggregate,
                full_scan=aggregate_full_scan,
            )
        except ValueError as exc:
            raise SourceVerificationError(
                f"aggregate source binding changed: {exc}"
            ) from exc
    _verify_snapshot(root_fd, root_snapshot, "input_root held descriptor")

    root_flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    if not single_file:
        root_flags |= os.O_DIRECTORY
    reopened_root = _open_absolute_binding(slash_fd, input_root, root_flags)
    try:
        _verify_snapshot(reopened_root, root_snapshot, "input_root named")
    except BaseException as primary:
        _close_fd(reopened_root, "input_root revalidation", primary)
        raise
    else:
        _close_fd(reopened_root, "input_root revalidation")

    if single_file:
        reopened_source = _open_absolute_binding(slash_fd, input_root, root_flags)
    elif aggregate is None:
        reopened_source = _open_directory_source(root_fd, relative_path)
    else:
        reopened_source = _open_directory_source(
            root_fd,
            relative_path,
            aggregate=True,
        )
    try:
        _verify_snapshot(
            reopened_source,
            source_snapshot,
            "source named",
            regular_file=aggregate is None,
        )
        if aggregate is not None and not stat.S_ISDIR(os.fstat(reopened_source).st_mode):
            raise SourceVerificationError("aggregate source named binding is not a directory")
        if aggregate is not None:
            try:
                verify_aggregate_identity(
                    reopened_source,
                    aggregate,
                    full_scan=aggregate_full_scan,
                )
            except ValueError as exc:
                raise SourceVerificationError(
                    f"aggregate source named binding changed: {exc}"
                ) from exc
    except BaseException as primary:
        _close_fd(reopened_source, "source revalidation", primary)
        raise
    else:
        _close_fd(reopened_source, "source revalidation")


def decode_verified_source(
    row: Mapping[str, Any],
    *,
    parameters: SourceDecodeParameters = STEP1_SOURCE_DECODE_PARAMETERS,
    vad_target_sample_rate: int = 16_000,
) -> DecodedSource:
    """从 Step0 row 安全打开、资源预检并解码同一个 held source descriptor。"""
    if not isinstance(parameters, SourceDecodeParameters):
        raise SourceVerificationError(
            "source decode parameters must be SourceDecodeParameters"
        )
    if type(vad_target_sample_rate) is not int or vad_target_sample_rate <= 0:
        raise SourceVerificationError(
            "vad_target_sample_rate must be a positive Python integer"
        )
    _require_posix_descriptor_support()
    expected = _validate_row(row)
    slash_fd, root_fd, root_snapshot, single_file = _open_root(
        expected["input_root"]
    )
    source_fd = None
    raw_spool = None
    try:
        source_fd, source_snapshot = _open_source(
            root_fd,
            root_snapshot,
            single_file,
            expected["input_root"],
            expected["source_relative_path"],
        )
        aggregate = _scan_open_aggregate(
            source_fd,
            expected["input_root"],
            expected["source_relative_path"],
            expected,
        )
        if aggregate is None:
            if source_snapshot.st_size != expected["source_size_bytes"]:
                raise SourceVerificationError("source size mismatch before decode")
            _validate_raw_source_bytes(
                expected["source_size_bytes"],
                parameters=parameters,
            )
            raw_spool = _spool_verified_source(
                source_fd,
                expected["source_size_bytes"],
                expected_sha256=expected["source_sha256"],
            )

            try:
                _before_decode_test_hook()
            except Exception as exc:
                raise SourceVerificationError(
                    f"binding before-decode hook failed: {exc}"
                ) from exc
            audio, metadata = _decode_source_descriptor(
                source_fd,
                raw_source=raw_spool,
                expected=expected,
                parameters=parameters,
                vad_target_sample_rate=vad_target_sample_rate,
            )
        else:
            metadata = _aggregate_metadata(
                aggregate,
                expected,
                parameters=parameters,
                vad_target_sample_rate=vad_target_sample_rate,
            )
            try:
                verify_aggregate_identity(source_fd, aggregate, full_scan=False)
            except ValueError as exc:
                raise SourceVerificationError(
                    f"aggregate source binding changed before decode: {exc}"
                ) from exc
            try:
                _before_decode_test_hook()
            except Exception as exc:
                raise SourceVerificationError(
                    f"binding before-decode hook failed: {exc}"
                ) from exc
            try:
                full_audio = decode_aggregate_full(
                    source_fd,
                    aggregate,
                    full_scan=False,
                    verify_bindings=False,
                )
            except ValueError as exc:
                raise SourceVerificationError(
                    f"aggregate full decode failed: {exc}"
                ) from exc
            expected_shape = (metadata["frames"], metadata["channels"])
            if (
                type(full_audio) is not np.ndarray
                or full_audio.shape != expected_shape
                or full_audio.dtype != np.dtype(np.float32)
                or not full_audio.flags.c_contiguous
                or not full_audio.flags.owndata
                or not full_audio.flags.writeable
                or full_audio.base is not None
            ):
                raise SourceVerificationError(
                    "aggregate full decode result does not satisfy mutable owned array contract"
                )
            # aggregate 整源解码 transiently 物化全部成员通道；Step1 只消费
            # channel 0，提取为自有 mono 数组后释放全通道数组。
            audio = np.array(full_audio[:, 0], dtype=np.float32, order="C", copy=True)
            if (
                audio.shape != (metadata["frames"],)
                or not audio.flags.owndata
                or audio.base is not None
            ):
                raise SourceVerificationError(
                    "aggregate full decode result does not satisfy mutable owned array contract"
                )
        try:
            _post_decode_test_hook()
        except Exception as exc:
            raise SourceVerificationError(
                f"binding post-decode hook failed: {exc}"
            ) from exc
        _verify_post_decode_bindings(
            source_fd,
            source_snapshot,
            root_fd,
            root_snapshot,
            slash_fd,
            expected["input_root"],
            expected["source_relative_path"],
            single_file,
            aggregate=aggregate,
            aggregate_full_scan=False,
        )
        result = DecodedSource(
            audio=audio,
            sample_rate=int(metadata["sample_rate"]),
            frames=int(metadata["frames"]),
            channels=int(metadata["channels"]),
            format=str(metadata["format"]),
            subtype=str(metadata["subtype"]),
            source_size_bytes=int(expected["source_size_bytes"]),
            source_sha256=str(expected["source_sha256"]),
            vad_capacity_samples=int(metadata["vad_capacity_samples"]),
        )
    except BaseException as primary:
        if raw_spool is not None:
            try:
                raw_spool.close()
            except BaseException as cleanup_error:
                _add_cleanup_note(primary, "raw source spool", cleanup_error)
        resources = []
        if source_fd is not None:
            resources.append((source_fd, "source descriptor", "binding"))
        resources.extend([
            (root_fd, "input_root descriptor", "binding"),
            (slash_fd, "absolute root anchor", "binding"),
        ])
        _close_fds(resources, primary)
        raise
    else:
        resources = [
            (source_fd, "source descriptor", "binding"),
            (root_fd, "input_root descriptor", "binding"),
            (slash_fd, "absolute root anchor", "binding"),
        ]
        try:
            if raw_spool is not None:
                raw_spool.close()
        except BaseException as primary:
            _close_fds(resources, primary)
            raise
        _close_fds(resources)
        return result


def _read_verified_header(descriptor: int, expected: Mapping[str, Any]) -> dict[str, Any]:
    """用 held FD 的独立副本验证容器头，绝不按 pathname 重开。"""
    duplicate = _dup_fd(descriptor, "decode", "source header descriptor")
    stream = None
    try:
        try:
            stream = sf.SoundFile(duplicate, mode="r", closefd=False)
        except Exception as exc:
            raise SourceVerificationError(
                f"decode failed while opening source header: {exc}"
            ) from exc
        try:
            metadata = {
                "sample_rate": stream.samplerate,
                "frames": stream.frames,
                "channels": stream.channels,
                "format": stream.format,
                "subtype": stream.subtype,
            }
        except Exception as exc:
            raise SourceVerificationError(f"header read failed: {exc}") from exc
        for field in ("sample_rate", "frames", "channels"):
            if type(metadata[field]) is not int or metadata[field] <= 0:
                raise SourceVerificationError(
                    f"header {field} must be a positive Python integer"
                )
        if any(
            type(metadata[field]) is not str or not metadata[field].strip()
            for field in ("format", "subtype")
        ):
            raise SourceVerificationError("header contains invalid format or subtype")
        _verify_header(metadata, expected)
    except BaseException as primary:
        if stream is not None:
            _close_soundfile(stream, primary)
        _close_fd(duplicate, "source header duplicate", primary, category="decode")
        raise
    else:
        try:
            _close_soundfile(stream)
        except BaseException as primary:
            _close_fd(duplicate, "source header duplicate", primary, category="decode")
            raise
        _close_fd(duplicate, "source header duplicate", category="decode")
        return metadata


class VerifiedSourceSession:
    """Step3 单源安全会话：持有验证后的 descriptor 并按半开区间解码。"""

    def __init__(self, row: Mapping[str, Any]) -> None:
        self._row = row
        self._expected: dict[str, Any] | None = None
        self._slash_fd: int | None = None
        self._root_fd: int | None = None
        self._source_fd: int | None = None
        self._root_snapshot: os.stat_result | None = None
        self._source_snapshot: os.stat_result | None = None
        self._aggregate: AggregateDescriptor | None = None
        self._aggregate_reader: AggregateStreamReader | None = None
        self._single_file = False
        self._metadata: dict[str, Any] | None = None
        # os.dup 共享 open-file description；同一 session 的 header/seek/read/exit
        # 必须串行，跨 source session 仍可并行。
        self._io_lock = threading.RLock()
        self._interval_decode_calls = 0
        self._interval_decode_frames = 0

    @property
    def io_counts(self) -> Mapping[str, int]:
        with self._io_lock:
            expected_size = 0 if self._expected is None else self._expected["source_size_bytes"]
            return {
                "source_full_sha_reads": 1 if self._expected is not None else 0,
                "source_full_sha_bytes": int(expected_size),
                "interval_decode_calls": self._interval_decode_calls,
                "interval_decode_frames": self._interval_decode_frames,
                "interval_decode_bytes": (
                    self._interval_decode_frames * self.channels * _FLOAT32_ITEMSIZE
                    if self._metadata is not None else 0
                ),
            }

    @property
    def sample_rate(self) -> int:
        self._require_open()
        assert self._metadata is not None
        return int(self._metadata["sample_rate"])

    @property
    def frames(self) -> int:
        self._require_open()
        assert self._metadata is not None
        return int(self._metadata["frames"])

    @property
    def channels(self) -> int:
        self._require_open()
        assert self._metadata is not None
        return int(self._metadata["channels"])

    def _require_open(self) -> None:
        if self._source_fd is None or self._metadata is None:
            raise SourceVerificationError("verified source session is not open")

    def __enter__(self) -> "VerifiedSourceSession":
        if self._source_fd is not None:
            raise SourceVerificationError("verified source session is already open")
        _require_posix_descriptor_support()
        expected = _validate_row(self._row)
        slash_fd, root_fd, root_snapshot, single_file = _open_root(expected["input_root"])
        source_fd = None
        aggregate = None
        aggregate_reader = None
        try:
            source_fd, source_snapshot = _open_source(
                root_fd,
                root_snapshot,
                single_file,
                expected["input_root"],
                expected["source_relative_path"],
            )
            aggregate = _scan_open_aggregate(
                source_fd,
                expected["input_root"],
                expected["source_relative_path"],
                expected,
            )
            if aggregate is None:
                if source_snapshot.st_size != expected["source_size_bytes"]:
                    raise SourceVerificationError("source size mismatch before session open")
                # 与 Step1 解码路径保持一致的原始字节资源护栏。
                _validate_raw_source_bytes(
                    expected["source_size_bytes"],
                    parameters=STEP1_SOURCE_DECODE_PARAMETERS,
                )
                source_sha256 = _hash_source_descriptor(
                    source_fd, expected["source_size_bytes"]
                )
                if source_sha256 != expected["source_sha256"]:
                    raise SourceVerificationError("source sha256 mismatch before session open")
                metadata = _read_verified_header(source_fd, expected)
            else:
                metadata = _aggregate_metadata(
                    aggregate,
                    expected,
                    parameters=STEP1_SOURCE_DECODE_PARAMETERS,
                    vad_target_sample_rate=16_000,
                )
                aggregate_reader = AggregateStreamReader(source_fd, aggregate)
        except BaseException as primary:
            if aggregate_reader is not None:
                try:
                    aggregate_reader.close()
                except BaseException as cleanup_error:
                    _add_cleanup_note(primary, "aggregate stream reader", cleanup_error)
            resources = []
            if source_fd is not None:
                resources.append((source_fd, "source descriptor", "binding"))
            resources.extend([
                (root_fd, "input_root descriptor", "binding"),
                (slash_fd, "absolute root anchor", "binding"),
            ])
            _close_fds(resources, primary)
            raise
        self._expected = expected
        self._slash_fd = slash_fd
        self._root_fd = root_fd
        self._source_fd = source_fd
        self._root_snapshot = root_snapshot
        self._source_snapshot = source_snapshot
        self._aggregate = aggregate
        self._aggregate_reader = aggregate_reader
        self._single_file = single_file
        self._metadata = metadata
        return self

    def decode_interval(self, start_sample: int, end_sample: int) -> tuple[np.ndarray, int]:
        """线程安全地从 held source FD 精确读取半开区间。"""
        with self._io_lock:
            result = self._decode_interval_unlocked(start_sample, end_sample)
            self._interval_decode_calls += 1
            self._interval_decode_frames += end_sample - start_sample
            return result

    def _decode_interval_unlocked(
        self, start_sample: int, end_sample: int,
    ) -> tuple[np.ndarray, int]:
        self._require_open()
        if (
            type(start_sample) is not int
            or type(end_sample) is not int
            or not 0 <= start_sample < end_sample <= self.frames
        ):
            raise SourceVerificationError(
                "source interval must be nonempty, half-open, and within source frames"
            )
        interval_frames = end_sample - start_sample
        decoded_bytes = interval_frames * self.channels * _FLOAT32_ITEMSIZE
        if decoded_bytes > MAX_STEP3_INTERVAL_DECODED_BYTES:
            raise SourceVerificationError(
                "Step3 interval resource limit exceeded: decoded float32 waveform "
                f"requires {decoded_bytes} bytes, limit is "
                f"{MAX_STEP3_INTERVAL_DECODED_BYTES} bytes"
            )
        assert self._source_fd is not None
        if self._aggregate is not None:
            assert self._aggregate_reader is not None
            try:
                audio = decode_aggregate_interval(
                    self._source_fd,
                    self._aggregate,
                    start_sample,
                    end_sample,
                    full_scan=False,
                    verify_bindings=True,
                    reader=self._aggregate_reader,
                )
            except Exception as exc:
                raise SourceVerificationError(
                    f"aggregate source interval decode failed: {exc}"
                ) from exc
            expected_shape = (end_sample - start_sample, self.channels)
            if (
                type(audio) is not np.ndarray
                or audio.shape != expected_shape
                or audio.dtype != np.dtype(np.float32)
                or not audio.flags.c_contiguous
                or not audio.flags.owndata
                or not audio.flags.writeable
                or audio.base is not None
            ):
                raise SourceVerificationError(
                    "aggregate source interval result does not satisfy mutable owned array contract"
                )
            return audio, self.sample_rate
        duplicate = _dup_fd(self._source_fd, "decode", "source interval descriptor")
        stream = None
        try:
            try:
                # os.dup() 共享 open-file description 的当前 offset；强制回到
                # 文件起点，避免入口 header 验证后的偏移污染区间解码。
                os.lseek(duplicate, 0, os.SEEK_SET)
                stream = sf.SoundFile(duplicate, mode="r", closefd=False)
                position = stream.seek(start_sample)
            except Exception as exc:
                raise SourceVerificationError(f"source interval seek failed: {exc}") from exc
            if position != start_sample:
                raise SourceVerificationError("source interval seek did not reach requested start")
            try:
                audio = stream.read(
                    end_sample - start_sample, dtype="float32", always_2d=True
                )
            except Exception as exc:
                raise SourceVerificationError(f"source interval decode failed: {exc}") from exc
            expected_shape = (end_sample - start_sample, self.channels)
            if not isinstance(audio, np.ndarray) or audio.shape != expected_shape:
                raise SourceVerificationError(
                    "source interval decode was short or malformed"
                )
        except BaseException as primary:
            if stream is not None:
                _close_soundfile(stream, primary)
            _close_fd(duplicate, "source interval duplicate", primary, category="decode")
            raise
        else:
            try:
                _close_soundfile(stream)
            except BaseException as primary:
                _close_fd(duplicate, "source interval duplicate", primary, category="decode")
                raise
            try:
                result = np.array(audio, dtype=np.float32, order="C", copy=True)
                if (
                    type(result) is not np.ndarray
                    or result.shape != expected_shape
                    or result.dtype != np.dtype(np.float32)
                    or not result.flags.c_contiguous
                    or not result.flags.owndata
                    or not result.flags.writeable
                    or result.base is not None
                ):
                    raise SourceVerificationError(
                        "source interval result does not satisfy mutable owned array contract"
                    )
            except BaseException as primary:
                _close_fd(duplicate, "source interval duplicate", primary, category="decode")
                raise
            _close_fd(duplicate, "source interval duplicate", category="decode")
            return result, self.sample_rate

    def verify_bindings(self) -> None:
        """在当前 session 仍持有源 descriptor 时复验 source binding。"""
        with self._io_lock:
            self._require_open()
            assert self._source_fd is not None
            assert self._root_fd is not None
            assert self._slash_fd is not None
            assert self._source_snapshot is not None
            assert self._root_snapshot is not None
            assert self._expected is not None
            _verify_post_decode_bindings(
                self._source_fd,
                self._source_snapshot,
                self._root_fd,
                self._root_snapshot,
                self._slash_fd,
                self._expected["input_root"],
                self._expected["source_relative_path"],
                self._single_file,
                aggregate=self._aggregate,
                aggregate_full_scan=False,
            )

    def _close_held_resources(self, primary: BaseException | None = None) -> None:
        cleanup_primary = primary
        aggregate_reader = self._aggregate_reader
        self._aggregate_reader = None
        if aggregate_reader is not None:
            try:
                aggregate_reader.close()
            except BaseException as cleanup_error:
                if cleanup_primary is None:
                    cleanup_primary = cleanup_error
                else:
                    _add_cleanup_note(
                        cleanup_primary,
                        "aggregate stream reader",
                        cleanup_error,
                    )
        resources = [
            (self._source_fd, "source descriptor", "binding"),
            (self._root_fd, "input_root descriptor", "binding"),
            (self._slash_fd, "absolute root anchor", "binding"),
        ]
        self._source_fd = None
        self._root_fd = None
        self._slash_fd = None
        self._aggregate = None
        self._metadata = None
        _close_fds(
            [(fd, label, category) for fd, label, category in resources if fd is not None],
            cleanup_primary,
        )
        if primary is None and cleanup_primary is not None:
            raise cleanup_primary

    def _exit_unlocked(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        self._require_open()
        assert self._source_fd is not None
        assert self._root_fd is not None
        assert self._slash_fd is not None
        assert self._source_snapshot is not None
        assert self._root_snapshot is not None
        assert self._expected is not None
        primary = exc
        try:
            _verify_post_decode_bindings(
                self._source_fd,
                self._source_snapshot,
                self._root_fd,
                self._root_snapshot,
                self._slash_fd,
                self._expected["input_root"],
                self._expected["source_relative_path"],
                self._single_file,
                aggregate=self._aggregate,
                aggregate_full_scan=False,
            )
        except BaseException as verification_error:
            if primary is not None:
                _add_cleanup_note(primary, "session exit binding verification", verification_error)
            else:
                primary = verification_error
        try:
            self._close_held_resources(primary)
        except BaseException as cleanup_error:
            if primary is not None:
                _add_cleanup_note(primary, "session held descriptors", cleanup_error)
            else:
                primary = cleanup_error
        if exc is None and primary is not None:
            raise primary
        return False

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        with self._io_lock:
            return self._exit_unlocked(exc_type, exc, traceback)


def open_verified_source_session(row: Mapping[str, Any]) -> VerifiedSourceSession:
    """创建用于 Step3 精确片段解码的 verified source context manager。"""
    return VerifiedSourceSession(row)


def read_verified_source_interval(row: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    """以 Step0 descriptor binding 安全复验后读取 Step6 精确区间。"""
    if not isinstance(row, Mapping):
        raise SourceVerificationError("source interval row must be a mapping")
    start_sample, end_sample = row.get("start_sample"), row.get("end_sample")
    if type(start_sample) is not int or type(end_sample) is not int:
        raise SourceVerificationError("source interval bounds must be Python integers")
    with open_verified_source_session(row) as session:
        return session.decode_interval(start_sample, end_sample)
