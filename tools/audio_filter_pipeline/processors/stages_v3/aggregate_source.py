"""streaming FLAC aggregate source 的严格扫描、身份摘要与有限解码。"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from ...utils.speaker_metadata import validate_source_relative_path


AGGREGATE_SOURCE_LAYOUT = "streaming_flac_aggregate"
AGGREGATE_MANIFEST_DOMAIN = "audio_filter_pipeline.streaming_flac_aggregate"
AGGREGATE_MANIFEST_VERSION = 2
AGGREGATE_SCANNER_VERSION = 1
AGGREGATE_POLICY_VERSION = 1
AGGREGATE_READ_CHUNK_FRAMES = 65_536
AGGREGATE_MAX_MEMBER_FRAMES = 1_000_000_000
AGGREGATE_MAX_MEMBERS = 10_000


class AggregateMemberEmptyError(ValueError):
    def __init__(
        self,
        message: str = "streaming FLAC aggregate member is empty",
        *,
        member_name: str | None = None,
        aggregate_relative_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.member_name = member_name
        self.aggregate_relative_path = aggregate_relative_path

_SNAPSHOT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
    "st_nlink",
)


@dataclass(frozen=True)
class AggregateMember:
    index: int
    name: str
    size_bytes: int
    raw_sha256: str
    sample_rate: int
    frames: int
    channels: int
    format: str
    subtype: str
    snapshot: tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True)
class AggregateDescriptor:
    input_root: str
    source_relative_path: str
    directory_snapshot: tuple[int, int, int, int, int, int, int]
    members: tuple[AggregateMember, ...]
    source_size_bytes: int
    frames: int
    sample_rate: int
    channels: int
    format: str
    subtype: str
    manifest_sha256: str


def file_snapshot(snapshot: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return tuple(int(getattr(snapshot, field)) for field in _SNAPSHOT_FIELDS)


def _snapshot_payload(snapshot: tuple[int, int, int, int, int, int, int]) -> dict[str, int]:
    return dict(zip(_SNAPSHOT_FIELDS, snapshot))


def _canonical_root(input_root: str) -> str:
    if type(input_root) is not str or not input_root.strip():
        raise ValueError("aggregate input_root must be a non-empty string")
    canonical = os.path.realpath(os.path.abspath(input_root))
    if input_root != canonical:
        raise ValueError("aggregate input_root must be canonical")
    return canonical


def _validate_relative_path(source_relative_path: str) -> str:
    try:
        return validate_source_relative_path(source_relative_path)
    except ValueError as exc:
        raise ValueError(f"aggregate source_relative_path is invalid: {exc}") from exc


def _hash_descriptor(descriptor: int, expected_size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < expected_size:
        chunk = os.pread(descriptor, min(1024 * 1024, expected_size - offset), offset)
        if not chunk:
            raise ValueError("aggregate member ended before its held size")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, expected_size):
        raise ValueError("aggregate member has bytes beyond its held size")
    return digest.hexdigest()


def _header_from_stream(stream: Any) -> dict[str, Any]:
    header = {
        "sample_rate": stream.samplerate,
        "channels": stream.channels,
        "format": stream.format,
        "subtype": stream.subtype,
    }
    if (
        type(header["sample_rate"]) is not int
        or header["sample_rate"] <= 0
        or type(header["channels"]) is not int
        or header["channels"] <= 0
        or type(header["format"]) is not str
        or not header["format"].strip()
        or type(header["subtype"]) is not str
        or not header["subtype"].strip()
    ):
        raise ValueError("aggregate member header is invalid")
    if header["format"] != "FLAC":
        raise ValueError("streaming FLAC aggregate member must use FLAC format")
    return header


def _disable_soundfile_post_read_seek(stream: Any) -> None:
    """绕过未知总帧数 streaming FLAC 的 soundfile seek bookkeeping。"""
    try:
        info = stream._info
        info.seekable = sf._snd.SF_FALSE
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "streaming FLAC aggregate requires soundfile seekability control"
        ) from exc


def _scan_member_frames(
    descriptor: int,
    header: dict[str, Any],
    *,
    member_name: str | None = None,
) -> int:
    duplicate: int | None = None
    stream = None
    total = 0
    try:
        duplicate = os.dup(descriptor)
        os.lseek(duplicate, 0, os.SEEK_SET)
        stream = sf.SoundFile(duplicate, mode="r", closefd=False)
        actual = _header_from_stream(stream)
        if actual != header:
            raise ValueError("aggregate member header changed while scanning")
        _disable_soundfile_post_read_seek(stream)
        while True:
            remaining = AGGREGATE_MAX_MEMBER_FRAMES - total
            request = min(AGGREGATE_READ_CHUNK_FRAMES, remaining) if remaining > 0 else 1
            audio = stream.read(frames=request, dtype="float32", always_2d=True)
            if not isinstance(audio, np.ndarray) or audio.ndim != 2:
                raise ValueError("aggregate member returned malformed waveform chunk")
            if audio.shape[1] != header["channels"] or audio.shape[0] > request:
                raise ValueError("aggregate member returned an invalid waveform chunk")
            if audio.dtype != np.dtype(np.float32):
                raise ValueError("aggregate member returned a non-float32 waveform chunk")
            rows = int(audio.shape[0])
            if rows == 0:
                break
            if remaining <= 0 or rows > remaining:
                raise ValueError("aggregate member exceeds bounded frame limit")
            total += rows
        if total <= 0:
            raise AggregateMemberEmptyError(
                "streaming FLAC aggregate member is empty",
                member_name=member_name,
            )
        return total
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"aggregate member frame scan failed: {exc}") from exc
    finally:
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        if duplicate is not None:
            try:
                os.close(duplicate)
            except OSError:
                pass


def _member_pattern(directory_name: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(directory_name)}_(\d{{4}})\.flac$")


def _enumerate_member_entries(
    directory_descriptor: int,
    directory_name: str,
) -> tuple[tuple[int, int, int, int, int, int, int], dict[str, os.stat_result]]:
    before = os.fstat(directory_descriptor)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("aggregate source descriptor must be a directory")
    before_snapshot = file_snapshot(before)
    pattern = _member_pattern(directory_name)
    entries: dict[str, os.stat_result] = {}
    try:
        with os.scandir(directory_descriptor) as iterator:
            for entry in iterator:
                try:
                    snapshot = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise ValueError(
                        f"cannot inspect aggregate member {entry.name!r}"
                    ) from exc
                if entry.name in entries:
                    raise ValueError(f"duplicate aggregate member name: {entry.name!r}")
                entries[entry.name] = snapshot
    except OSError as exc:
        raise ValueError("cannot enumerate streaming FLAC aggregate directory") from exc
    after = os.fstat(directory_descriptor)
    if file_snapshot(after) != file_snapshot(before):
        raise ValueError("aggregate directory changed while enumerating members")
    if not entries:
        raise ValueError("streaming FLAC aggregate directory is empty")
    if len(entries) > AGGREGATE_MAX_MEMBERS:
        raise ValueError("streaming FLAC aggregate has too many members")
    for name, snapshot in entries.items():
        match = pattern.fullmatch(name)
        if match is None:
            raise ValueError(f"invalid streaming FLAC aggregate member name: {name!r}")
        if stat.S_ISLNK(snapshot.st_mode) or not stat.S_ISREG(snapshot.st_mode):
            raise ValueError(f"aggregate member must be a regular file: {name!r}")
    indices = sorted(int(pattern.fullmatch(name).group(1)) for name in entries)
    if indices != list(range(len(indices))):
        raise ValueError("streaming FLAC aggregate member indices must be contiguous from 0000")
    return before_snapshot, entries


def _open_member(
    directory_descriptor: int,
    name: str,
    expected_snapshot: tuple[int, int, int, int, int, int, int],
) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise ValueError(f"cannot safely open aggregate member {name!r}") from exc
    try:
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or file_snapshot(opened) != expected_snapshot
            or file_snapshot(named) != expected_snapshot
        ):
            raise ValueError(f"aggregate member pathname binding changed: {name!r}")
        return descriptor
    except BaseException as exc:
        try:
            os.close(descriptor)
        except OSError:
            pass
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"cannot verify aggregate member {name!r}") from exc


def _verify_member_snapshot(
    directory_descriptor: int,
    member_descriptor: int,
    member: AggregateMember,
) -> None:
    current = os.fstat(member_descriptor)
    named = os.stat(member.name, dir_fd=directory_descriptor, follow_symlinks=False)
    if (
        file_snapshot(current) != member.snapshot
        or file_snapshot(named) != member.snapshot
        or not stat.S_ISREG(current.st_mode)
    ):
        raise ValueError(f"aggregate member binding changed: {member.name!r}")


def _manifest_payload(
    input_root: str,
    source_relative_path: str,
    directory_snapshot: tuple[int, int, int, int, int, int, int],
    members: Iterable[AggregateMember],
) -> dict[str, Any]:
    return {
        "domain": AGGREGATE_MANIFEST_DOMAIN,
        "version": AGGREGATE_MANIFEST_VERSION,
        "input_root": input_root,
        "source_relative_path": source_relative_path,
        "directory_snapshot": _snapshot_payload(directory_snapshot),
        "members": [
            {
                "index": member.index,
                "name": member.name,
                "size_bytes": member.size_bytes,
                "raw_sha256": member.raw_sha256,
                "sample_rate": member.sample_rate,
                "frames": member.frames,
                "channels": member.channels,
                "format": member.format,
                "subtype": member.subtype,
                "snapshot": _snapshot_payload(member.snapshot),
            }
            for member in members
        ],
    }


def _manifest_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        AGGREGATE_MANIFEST_DOMAIN.encode("ascii") + b"\0" + encoded
    ).hexdigest()


def _descriptor_from_parts(
    input_root: str,
    source_relative_path: str,
    directory_snapshot: tuple[int, int, int, int, int, int, int],
    members: tuple[AggregateMember, ...],
) -> AggregateDescriptor:
    first = members[0]
    if any(
        (
            member.sample_rate,
            member.channels,
            member.format,
            member.subtype,
        )
        != (
            first.sample_rate,
            first.channels,
            first.format,
            first.subtype,
        )
        for member in members[1:]
    ):
        raise ValueError("streaming FLAC aggregate members have inconsistent headers")
    payload = _manifest_payload(
        input_root,
        source_relative_path,
        directory_snapshot,
        members,
    )
    return AggregateDescriptor(
        input_root=input_root,
        source_relative_path=source_relative_path,
        directory_snapshot=directory_snapshot,
        members=members,
        source_size_bytes=sum(member.size_bytes for member in members),
        frames=sum(member.frames for member in members),
        sample_rate=first.sample_rate,
        channels=first.channels,
        format=first.format,
        subtype=first.subtype,
        manifest_sha256=_manifest_sha256(payload),
    )


def _scan_aggregate_directory(
    directory_descriptor: int,
    input_root: str,
    source_relative_path: str,
) -> AggregateDescriptor:
    canonical_root = _canonical_root(input_root)
    relative = _validate_relative_path(source_relative_path)
    directory_name = relative.rsplit("/", 1)[-1]
    if not directory_name.endswith("_active"):
        raise ValueError(
            "streaming FLAC aggregate directory name must end with '_active'"
        )
    directory_snapshot, entries = _enumerate_member_entries(
        directory_descriptor,
        directory_name,
    )
    members: list[AggregateMember] = []
    for name in sorted(entries):
        entry_snapshot = file_snapshot(entries[name])
        match = _member_pattern(directory_name).fullmatch(name)
        assert match is not None
        index = int(match.group(1))
        descriptor = _open_member(directory_descriptor, name, entry_snapshot)
        try:
            source_size = int(entry_snapshot[3])
            raw_sha256 = _hash_descriptor(descriptor, source_size)
            duplicate: int | None = None
            stream = None
            try:
                duplicate = os.dup(descriptor)
                os.lseek(duplicate, 0, os.SEEK_SET)
                stream = sf.SoundFile(duplicate, mode="r", closefd=False)
                header = _header_from_stream(stream)
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"aggregate member header read failed: {name!r}") from exc
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                if duplicate is not None:
                    try:
                        os.close(duplicate)
                    except OSError:
                        pass
            try:
                frames = _scan_member_frames(
                    descriptor,
                    header,
                    member_name=name,
                )
            except AggregateMemberEmptyError as exc:
                if exc.aggregate_relative_path is None:
                    exc.aggregate_relative_path = relative
                raise
            member = AggregateMember(
                index=index,
                name=name,
                size_bytes=source_size,
                raw_sha256=raw_sha256,
                sample_rate=header["sample_rate"],
                frames=frames,
                channels=header["channels"],
                format=header["format"],
                subtype=header["subtype"],
                snapshot=entry_snapshot,
            )
            _verify_member_snapshot(directory_descriptor, descriptor, member)
            members.append(member)
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if file_snapshot(os.fstat(directory_descriptor)) != directory_snapshot:
        raise ValueError("aggregate directory changed while scanning")
    return _descriptor_from_parts(
        canonical_root,
        relative,
        directory_snapshot,
        tuple(sorted(members, key=lambda member: member.index)),
    )


def scan_aggregate_directory(
    directory_descriptor: int,
    input_root: str,
    source_relative_path: str,
) -> AggregateDescriptor:
    """从 held aggregate directory descriptor 构造完整 manifest。"""
    return _scan_aggregate_directory(
        directory_descriptor,
        input_root,
        source_relative_path,
    )


def verify_aggregate_bindings(
    directory_descriptor: int,
    aggregate: AggregateDescriptor,
) -> None:
    """只复验 aggregate 目录、成员 pathname 与 held snapshot 集合。"""
    relative = _validate_relative_path(aggregate.source_relative_path)
    expected_path = os.path.join(aggregate.input_root, *relative.split("/"))
    proc_path = f"/proc/self/fd/{directory_descriptor}"
    if not os.path.exists(proc_path):
        raise ValueError("cannot verify aggregate directory descriptor pathname")
    actual_path = os.path.realpath(proc_path)
    if actual_path.endswith(" (deleted)") or actual_path != expected_path:
        raise ValueError("aggregate directory pathname binding changed")
    current_directory = os.fstat(directory_descriptor)
    if (
        not stat.S_ISDIR(current_directory.st_mode)
        or file_snapshot(current_directory) != aggregate.directory_snapshot
    ):
        raise ValueError("aggregate directory binding changed")

    directory_name = relative.rsplit("/", 1)[-1]
    pattern = _member_pattern(directory_name)
    expected_members = {
        member.name: member.snapshot for member in aggregate.members
    }
    try:
        with os.scandir(directory_descriptor) as iterator:
            entries = tuple(iterator)
    except OSError as exc:
        raise ValueError("cannot enumerate aggregate directory bindings") from exc
    if len(entries) != len(expected_members):
        raise ValueError("aggregate member pathname set changed")
    seen: set[str] = set()
    for entry in entries:
        name = entry.name
        expected_snapshot = expected_members.get(name)
        if expected_snapshot is None or name in seen:
            raise ValueError(f"aggregate member pathname changed: {name!r}")
        seen.add(name)
        if pattern.fullmatch(name) is None:
            raise ValueError(f"aggregate member pathname changed: {name!r}")
        try:
            named = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"cannot inspect aggregate member {name!r}") from exc
        if (
            stat.S_ISLNK(named.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or file_snapshot(named) != expected_snapshot
        ):
            raise ValueError(f"aggregate member binding changed: {name!r}")
    if seen != set(expected_members):
        raise ValueError("aggregate member pathname set changed")
    if file_snapshot(os.fstat(directory_descriptor)) != aggregate.directory_snapshot:
        raise ValueError("aggregate directory binding changed")


def verify_aggregate_identity(
    directory_descriptor: int,
    aggregate: AggregateDescriptor,
    *,
    full_scan: bool = True,
) -> None:
    """验证 aggregate manifest；默认完整重扫，session 可选择廉价 binding 复验。"""
    if not isinstance(full_scan, bool):
        raise ValueError("full_scan must be bool")
    if not full_scan:
        verify_aggregate_bindings(directory_descriptor, aggregate)
        return
    current = _scan_aggregate_directory(
        directory_descriptor,
        aggregate.input_root,
        aggregate.source_relative_path,
    )
    if current != aggregate:
        raise ValueError("aggregate manifest or binding changed")


def _add_exception_note(primary: BaseException, label: str, error: BaseException) -> None:
    note = f"{label}: {error}"
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    notes = list(getattr(primary, "__notes__", ()))
    notes.append(note)
    primary.__notes__ = notes


@dataclass
class _AggregateMemberStream:
    member: AggregateMember
    descriptor: int
    duplicate: int
    stream: Any
    position: int = 0


class AggregateStreamReader:
    """在一个 verified session 内复用 aggregate member 的顺序解码游标。"""

    def __init__(
        self,
        directory_descriptor: int,
        aggregate: AggregateDescriptor,
    ) -> None:
        if type(directory_descriptor) is not int or directory_descriptor < 0:
            raise ValueError("aggregate directory descriptor must be a non-negative integer")
        if not isinstance(aggregate, AggregateDescriptor) or not aggregate.members:
            raise ValueError("aggregate reader requires a non-empty descriptor")
        offsets = []
        offset = 0
        for member in aggregate.members:
            if type(member.frames) is not int or member.frames <= 0:
                raise ValueError("aggregate member frame count must be positive")
            offsets.append(offset)
            offset += member.frames
        if offset != aggregate.frames:
            raise ValueError("aggregate reader frame offsets do not match descriptor")
        self._directory_descriptor = directory_descriptor
        self._aggregate = aggregate
        self._offsets = tuple(offsets)
        self._active: _AggregateMemberStream | None = None
        self._closed = False

    def _close_active(self) -> None:
        active = self._active
        self._active = None
        if active is None:
            return
        cleanup_errors: list[BaseException] = []
        try:
            active.stream.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            os.close(active.duplicate)
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            _verify_member_snapshot(
                self._directory_descriptor,
                active.descriptor,
                active.member,
            )
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            os.close(active.descriptor)
        except BaseException as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            primary = cleanup_errors[0]
            for error in cleanup_errors[1:]:
                _add_exception_note(primary, "aggregate member stream cleanup", error)
            raise primary

    def _open_member(self, index: int) -> None:
        member = self._aggregate.members[index]
        descriptor = None
        duplicate = None
        stream = None
        primary: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        try:
            descriptor = _open_member(
                self._directory_descriptor,
                member.name,
                member.snapshot,
            )
            duplicate = os.dup(descriptor)
            os.lseek(duplicate, 0, os.SEEK_SET)
            stream = sf.SoundFile(duplicate, mode="r", closefd=False)
            header = _header_from_stream(stream)
            expected_header = {
                "sample_rate": member.sample_rate,
                "channels": member.channels,
                "format": member.format,
                "subtype": member.subtype,
            }
            if header != expected_header:
                raise ValueError(f"aggregate member header changed: {member.name!r}")
            _disable_soundfile_post_read_seek(stream)
        except ValueError as exc:
            primary = exc
        except Exception as exc:
            primary = ValueError(
                f"aggregate member stream open failed: {member.name!r}"
            )
            primary.__cause__ = exc
        except BaseException as exc:
            primary = exc
        if primary is not None:
            if stream is not None:
                try:
                    stream.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if duplicate is not None:
                try:
                    os.close(duplicate)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            for error in cleanup_errors:
                _add_exception_note(primary, "aggregate member stream open cleanup", error)
            raise primary
        assert descriptor is not None
        assert duplicate is not None
        assert stream is not None
        self._active = _AggregateMemberStream(
            member=member,
            descriptor=descriptor,
            duplicate=duplicate,
            stream=stream,
        )

    def _ensure_position(self, index: int, local_start: int) -> None:
        active = self._active
        if (
            active is None
            or active.member.index != index
            or local_start < active.position
        ):
            if active is not None:
                self._close_active()
            self._open_member(index)
            active = self._active
        assert active is not None
        to_skip = local_start - active.position
        while to_skip > 0:
            request = min(AGGREGATE_READ_CHUNK_FRAMES, to_skip)
            audio = active.stream.read(
                frames=request,
                dtype="float32",
                always_2d=True,
            )
            if (
                not isinstance(audio, np.ndarray)
                or audio.ndim != 2
                or audio.shape[1] != active.member.channels
                or audio.shape[0] > request
                or audio.dtype != np.dtype(np.float32)
            ):
                raise ValueError(
                    f"aggregate member returned malformed skip chunk: {active.member.name!r}"
                )
            rows = int(audio.shape[0])
            if rows != request:
                raise ValueError(
                    f"aggregate member seekless skip was short: {active.member.name!r}"
                )
            active.position += rows
            to_skip -= rows

    def _read_into(
        self,
        output: np.ndarray,
        start_sample: int,
        end_sample: int,
    ) -> None:
        current = start_sample
        output_offset = 0
        while current < end_sample:
            index = bisect.bisect_right(self._offsets, current) - 1
            member = self._aggregate.members[index]
            member_start = self._offsets[index]
            member_end = member_start + member.frames
            overlap_end = min(end_sample, member_end)
            local_start = current - member_start
            self._ensure_position(index, local_start)
            active = self._active
            assert active is not None
            written = 0
            member_frames = overlap_end - current
            while written < member_frames:
                request = min(AGGREGATE_READ_CHUNK_FRAMES, member_frames - written)
                target = output[output_offset + written:output_offset + written + request]
                try:
                    audio = active.stream.read(
                        frames=request,
                        dtype="float32",
                        always_2d=True,
                        out=target,
                    )
                except TypeError as out_error:
                    try:
                        audio = active.stream.read(
                            frames=request,
                            dtype="float32",
                            always_2d=True,
                        )
                    except TypeError:
                        raise out_error
                if (
                    not isinstance(audio, np.ndarray)
                    or audio.ndim != 2
                    or audio.shape[1] != active.member.channels
                    or audio.shape[0] > request
                    or audio.dtype != np.dtype(np.float32)
                ):
                    raise ValueError(
                        f"aggregate member returned malformed interval: {active.member.name!r}"
                    )
                rows = int(audio.shape[0])
                if rows == 0:
                    raise ValueError(
                        f"aggregate member interval was short: {active.member.name!r}"
                    )
                if audio is not target:
                    target[:rows] = audio
                active.position += rows
                written += rows
            if written != member_frames:
                raise ValueError(
                    f"aggregate member interval was short: {member.name!r}"
                )
            current = overlap_end
            output_offset += member_frames

    def read_interval_into(
        self,
        output: np.ndarray,
        start_sample: int,
        end_sample: int,
    ) -> None:
        if self._closed:
            raise ValueError("aggregate stream reader is closed")
        if (
            type(start_sample) is not int
            or type(end_sample) is not int
            or not 0 <= start_sample < end_sample <= self._aggregate.frames
        ):
            raise ValueError("aggregate interval must be nonempty and within source frames")
        expected_shape = (
            end_sample - start_sample,
            self._aggregate.channels,
        )
        if (
            type(output) is not np.ndarray
            or output.shape != expected_shape
            or output.dtype != np.dtype(np.float32)
            or not output.flags.c_contiguous
            or not output.flags.writeable
        ):
            raise ValueError("aggregate reader output does not satisfy mutable array contract")
        try:
            self._read_into(output, start_sample, end_sample)
        except BaseException as caught:
            primary = caught
            if isinstance(caught, Exception) and not isinstance(caught, ValueError):
                primary = ValueError("aggregate interval decode failed")
                primary.__cause__ = caught
            try:
                self._close_active()
            except BaseException as cleanup_error:
                _add_exception_note(
                    primary,
                    "aggregate member stream failure cleanup",
                    cleanup_error,
                )
            raise primary

    def read_interval(self, start_sample: int, end_sample: int) -> np.ndarray:
        result = np.empty(
            (end_sample - start_sample, self._aggregate.channels),
            dtype=np.float32,
            order="C",
        )
        self.read_interval_into(result, start_sample, end_sample)
        return result

    def matches(
        self,
        directory_descriptor: int,
        aggregate: AggregateDescriptor,
    ) -> bool:
        return (
            not self._closed
            and directory_descriptor == self._directory_descriptor
            and aggregate == self._aggregate
        )

    def verify_bindings(self) -> None:
        if self._closed:
            raise ValueError("aggregate stream reader is closed")
        verify_aggregate_bindings(self._directory_descriptor, self._aggregate)
        active = self._active
        if active is not None:
            _verify_member_snapshot(
                self._directory_descriptor,
                active.descriptor,
                active.member,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_active()


def _read_member_interval(
    directory_descriptor: int,
    member: AggregateMember,
    start: int,
    frames: int,
    *,
    output: np.ndarray | None = None,
    output_offset: int = 0,
) -> np.ndarray:
    """顺序解码 member 到 start 后读取有限的半开区间。"""
    if type(start) is not int or type(frames) is not int or start < 0 or frames <= 0:
        raise ValueError("aggregate member interval must be positive integers")
    if start + frames > member.frames:
        raise ValueError("aggregate member interval exceeds member frame count")
    if type(output_offset) is not int or output_offset < 0:
        raise ValueError("aggregate member output offset must be a non-negative integer")
    owns_result = output is None
    if output is None:
        if output_offset:
            raise ValueError("aggregate member output offset requires an output array")
        result = np.empty((frames, member.channels), dtype=np.float32, order="C")
        output_offset = 0
    else:
        if (
            type(output) is not np.ndarray
            or output.ndim != 2
            or output.dtype != np.dtype(np.float32)
            or not output.flags.c_contiguous
            or not output.flags.writeable
            or output_offset + frames > output.shape[0]
            or output.shape[1] != member.channels
        ):
            raise ValueError("aggregate member output does not satisfy mutable array contract")
        result = output
    descriptor = _open_member(directory_descriptor, member.name, member.snapshot)
    duplicate = None
    stream = None
    primary: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
        duplicate = os.dup(descriptor)
        os.lseek(duplicate, 0, os.SEEK_SET)
        stream = sf.SoundFile(duplicate, mode="r", closefd=False)
        header = _header_from_stream(stream)
        if header != {
            "sample_rate": member.sample_rate,
            "channels": member.channels,
            "format": member.format,
            "subtype": member.subtype,
        }:
            raise ValueError(f"aggregate member header changed: {member.name!r}")
        _disable_soundfile_post_read_seek(stream)
        skipped = 0
        while skipped < start:
            request = min(AGGREGATE_READ_CHUNK_FRAMES, start - skipped)
            discarded = stream.read(
                frames=request,
                dtype="float32",
                always_2d=True,
            )
            if (
                not isinstance(discarded, np.ndarray)
                or discarded.ndim != 2
                or discarded.shape[1] != member.channels
                or discarded.shape[0] > request
                or discarded.dtype != np.dtype(np.float32)
            ):
                raise ValueError(f"aggregate member returned malformed skip chunk: {member.name!r}")
            rows = int(discarded.shape[0])
            if rows != request:
                raise ValueError(f"aggregate member seekless skip was short: {member.name!r}")
            skipped += rows
        written = 0
        while written < frames:
            request = min(AGGREGATE_READ_CHUNK_FRAMES, frames - written)
            target = result[output_offset + written:output_offset + written + request]
            try:
                audio = stream.read(
                    frames=request,
                    dtype="float32",
                    always_2d=True,
                    out=target,
                )
            except TypeError as out_error:
                try:
                    audio = stream.read(
                        frames=request,
                        dtype="float32",
                        always_2d=True,
                    )
                except TypeError:
                    raise out_error
            if (
                not isinstance(audio, np.ndarray)
                or audio.ndim != 2
                or audio.shape[1] != member.channels
                or audio.shape[0] > request
                or audio.dtype != np.dtype(np.float32)
            ):
                raise ValueError(f"aggregate member returned malformed interval: {member.name!r}")
            rows = int(audio.shape[0])
            if rows == 0:
                raise ValueError(f"aggregate member interval was short: {member.name!r}")
            if audio is not target:
                target[:rows] = audio
            written += rows
        if written != frames:
            raise ValueError(f"aggregate member interval was short: {member.name!r}")
    except ValueError as exc:
        primary = exc
    except Exception as exc:
        primary = ValueError(
            f"aggregate member interval decode failed: {member.name!r}"
        )
        primary.__cause__ = exc
    except BaseException as exc:
        primary = exc
    if stream is not None:
        try:
            stream.close()
        except BaseException as exc:
            if primary is not None:
                _add_exception_note(primary, "aggregate member SoundFile cleanup failed", exc)
            else:
                cleanup_errors.append(exc)
    if duplicate is not None:
        try:
            os.close(duplicate)
        except BaseException as exc:
            if primary is not None:
                _add_exception_note(primary, "aggregate member duplicate cleanup failed", exc)
            else:
                cleanup_errors.append(exc)
    try:
        _verify_member_snapshot(directory_descriptor, descriptor, member)
    except BaseException as exc:
        if primary is not None:
            _add_exception_note(primary, "aggregate member binding verification failed", exc)
        else:
            cleanup_errors.append(exc)
    try:
        os.close(descriptor)
    except BaseException as exc:
        if primary is not None:
            _add_exception_note(primary, "aggregate member descriptor cleanup failed", exc)
        else:
            cleanup_errors.append(exc)
    if primary is not None:
        raise primary
    if cleanup_errors:
        raise cleanup_errors[0]
    if owns_result:
        return result
    return result[output_offset:output_offset + frames]


def decode_aggregate_interval(
    directory_descriptor: int,
    aggregate: AggregateDescriptor,
    start_sample: int,
    end_sample: int,
    *,
    full_scan: bool = True,
    verify_bindings: bool = True,
    reader: AggregateStreamReader | None = None,
) -> np.ndarray:
    """按全局半开帧区间跨 member 有界读取 aggregate。"""
    if (
        type(start_sample) is not int
        or type(end_sample) is not int
        or not 0 <= start_sample < end_sample <= aggregate.frames
    ):
        raise ValueError("aggregate interval must be nonempty and within source frames")
    if not isinstance(full_scan, bool):
        raise ValueError("full_scan must be bool")
    if not isinstance(verify_bindings, bool):
        raise ValueError("verify_bindings must be bool")
    if reader is not None:
        if not isinstance(reader, AggregateStreamReader):
            raise ValueError("aggregate reader must be an AggregateStreamReader")
        if not reader.matches(directory_descriptor, aggregate):
            raise ValueError("aggregate reader does not match the requested source")
    if verify_bindings:
        if reader is None:
            verify_aggregate_identity(directory_descriptor, aggregate, full_scan=full_scan)
        else:
            reader.verify_bindings()
    expected_frames = end_sample - start_sample
    result = np.empty((expected_frames, aggregate.channels), dtype=np.float32, order="C")
    if reader is not None:
        reader.read_interval_into(result, start_sample, end_sample)
        if verify_bindings:
            reader.verify_bindings()
    else:
        offset = 0
        wrote_frames = 0
        for member in aggregate.members:
            member_end = offset + member.frames
            overlap_start = max(start_sample, offset)
            overlap_end = min(end_sample, member_end)
            if overlap_start < overlap_end:
                local_start = overlap_start - offset
                member_frames = overlap_end - overlap_start
                _read_member_interval(
                    directory_descriptor,
                    member,
                    local_start,
                    member_frames,
                    output=result,
                    output_offset=overlap_start - start_sample,
                )
                wrote_frames += member_frames
            offset = member_end
            if offset >= end_sample:
                break
        if wrote_frames != expected_frames:
            raise ValueError(
                f"aggregate interval short read: expected {expected_frames} frames, "
                f"got {wrote_frames}"
            )
    if verify_bindings and reader is None:
        verify_aggregate_identity(directory_descriptor, aggregate, full_scan=full_scan)
    return result


def decode_aggregate_full(
    directory_descriptor: int,
    aggregate: AggregateDescriptor,
    *,
    full_scan: bool = True,
    verify_bindings: bool = True,
) -> np.ndarray:
    """完整读取 aggregate，所有 SoundFile.read 均带有限 frames 参数。"""
    return decode_aggregate_interval(
        directory_descriptor,
        aggregate,
        0,
        aggregate.frames,
        full_scan=full_scan,
        verify_bindings=verify_bindings,
    )
