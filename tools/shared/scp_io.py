"""Shared SCP / list / JSON index I/O utilities."""
from __future__ import annotations

import json
import logging
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf

logger = logging.getLogger(__name__)

_RECORD_CONTROL_NAMES = {"\r": "CR", "\n": "LF", "\0": "NUL"}


def _validate_record_field(value: str | Path, field: str) -> str:
    """拒绝可把一条索引记录拆成多行或截断的控制字符。"""
    text = str(value)
    found = [name for char, name in _RECORD_CONTROL_NAMES.items() if char in text]
    if found:
        raise ValueError(f"SCP {field} contains forbidden {'/'.join(found)} character(s): {text!r}")
    return text


def _atomic_write_text(content: str, out_path: Path) -> None:
    """以原子方式写入文本文件，避免写入过程中被其他进程读取到不完整内容。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 必须在 os.replace 前读取目标权限；替换后旧 inode 的 mode 不可再获取。
    try:
        old_mode = out_path.stat().st_mode & 0o777
    except OSError:
        old_mode = 0o644
    fd, tmp = tempfile.mkstemp(dir=out_path.parent, suffix=out_path.suffix + ".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, out_path)
        try:
            os.chmod(out_path, old_mode)
        except OSError:
            pass
    except Exception:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        raise


def probe_audio_meta(path: str | Path, fallback_sr: int = 16000) -> dict[str, Any]:
    """Probe audio file and return ``{sr, frames, duration, channels}``.

    Returns:
        Dict with samplerate (int), frames (int), duration (float), channels (int).
        On failure returns fallback_sr and zeros.
    """
    try:
        info = sf.info(str(path))
        return {
            "sr": int(info.samplerate),
            "frames": int(info.frames),
            "duration": float(info.duration),
            "channels": int(info.channels),
        }
    except Exception as exc:
        logger.warning("无法探测音频 %s: %s", path, exc)
        return {"sr": fallback_sr, "frames": 0, "duration": 0.0, "channels": 1}


def parse_scp_fields(
    line: str,
    scp_path: str | Path,
    lineno: int,
    *,
    allowed_columns: tuple[int, ...] = (1, 2, 4),
) -> tuple[str, ...]:
    """按 shell 引号解析 SCP 字段，并严格校验允许的列数。"""
    _validate_record_field(line, "record")
    try:
        cols = tuple(shlex.split(line))
    except ValueError as exc:
        raise ValueError(
            f"{scp_path}:{lineno}: SCP 行解析失败（引号不匹配）: {line!r}"
        ) from exc
    if not cols:
        raise ValueError(f"{scp_path}:{lineno}: 空 SCP 行")
    if cols[-1] == "|":
        # kaldi pipe 形式（行尾 ``|``）暂不支持，明确报错而非把 "|" 当路径
        raise ValueError(
            f"{scp_path}:{lineno}: 不支持 kaldi pipe 形式（行尾 '|'）: {line!r}"
        )
    if len(cols) not in allowed_columns:
        supported = "/".join(str(count) for count in allowed_columns)
        raise ValueError(
            f"{scp_path}:{lineno}: 非法列数 {len(cols)}，仅支持 {supported} 列: {line!r}"
        )
    return cols


def _parse_scp_entry(line: str, scp_path: str | Path, lineno: int) -> tuple[str, str]:
    """解析一行 SCP，返回 ``(uid, audio_path)``；单列行 uid 为 ``""``。

    解析失败（引号不匹配、列数非法、kaldi pipe 形式）一律 raise，
    不再静默取最后一个 token。
    """
    cols = parse_scp_fields(line, scp_path, lineno)
    if len(cols) == 1:
        return "", cols[0]
    return cols[0], cols[-1]


def read_scp(path: str | Path) -> list[str]:
    """Read an SCP file and return the audio-path column (last column).

    Handles standard 4-column ``uid sr samples path`` SCPs, 2-column
    ``uid path`` SCPs, and single-column path lists. 使用 ``shlex.split``
    解析 shell 引号，允许 uid / path 中含空格。
    列数异常、kaldi pipe 形式（行尾 ``|``）或引号不匹配时 raise ValueError。
    """
    lines: list[str] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            _, audio_path = _parse_scp_entry(line, path, lineno)
            lines.append(audio_path)
    return lines


def read_scp_dict(path: str | Path) -> dict[str, str]:
    """Read an SCP file and return a ``uid -> audio_path`` dict.

    与 :func:`read_scp` 使用相同的解析规则；检测到重复 uid 时 raise。
    单列（纯路径列表）SCP 没有 uid，不适用本函数，会 raise ValueError。
    """
    result: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            uid, audio_path = _parse_scp_entry(line, path, lineno)
            if not uid:
                raise ValueError(
                    f"{path}:{lineno}: 单列 SCP 无 uid，请改用 read_scp"
                )
            if uid in result:
                raise ValueError(f"{path}:{lineno}: 重复 uid: {uid!r}")
            result[uid] = audio_path
    return result


def serialize_manifest_relative_path(
    audio_path: str | Path,
    manifest_path: str | Path,
) -> str:
    """将音频路径写成相对于 manifest parent 的路径。

    输入可以是完整绝对路径，也可以是当前进程 cwd 下的相对路径；绝对路径
    只用于计算相对表示，不会限制读取端继续兼容历史绝对路径清单。
    """
    raw = _validate_record_field(audio_path, "path")
    if raw.startswith(("waveforms:", "dataset:")):
        return raw
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path(os.path.abspath(os.fspath(candidate)))
    manifest_parent = Path(os.path.abspath(os.fspath(manifest_path))).parent
    return os.path.relpath(candidate, manifest_parent)


def _normalize_scp_suffix(out_path: Path) -> Path:
    """SCP 写出统一以 ``.scp`` 结尾；后缀不符时替换并告警（隐式改名易踩坑）。"""
    if out_path.suffix == ".scp":
        return out_path
    scp_path = out_path.with_suffix(".scp")
    logger.warning("SCP 输出路径后缀不是 .scp，已隐式改写: %s -> %s", out_path, scp_path)
    return scp_path


_PUBLIC_FORMATS = {"scp", "list", "json"}


def serialize_scp_entries(
    entries: list[tuple[str, str]] | list[tuple[str, int, int, str]],
    *,
    fmt: str = "scp",
    four_col: bool = False,
) -> str:
    """先校验整组 uid/path，再序列化为 SCP/list/JSON 文本。"""
    if fmt not in _PUBLIC_FORMATS:
        raise ValueError(
            f"Unsupported fmt={fmt!r}; use {', '.join(sorted(_PUBLIC_FORMATS))}"
        )

    if fmt == "list":
        paths = []
        for entry in entries:
            if isinstance(entry, tuple):
                _validate_record_field(entry[0], "uid")
                path = entry[-1]
            else:
                path = entry
            paths.append(_validate_record_field(path, "path"))
        return "\n".join(paths) + ("\n" if paths else "")

    if four_col or (fmt == "json" and entries and len(entries[0]) == 4):
        normalized = [
            (
                _validate_record_field(uid, "uid"),
                sr,
                frames,
                _validate_record_field(path, "path"),
            )
            for uid, sr, frames, path in entries
        ]
        if fmt == "json":
            data = [
                {"uid": uid, "fs": sr, "length": frames, "path": path}
                for uid, sr, frames, path in normalized
            ]
            return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
        lines = [
            f"{shlex.quote(uid)} {sr} {frames} {shlex.quote(path)}"
            for uid, sr, frames, path in normalized
        ]
        return "\n".join(lines) + ("\n" if lines else "")

    normalized_two_col = [
        (
            _validate_record_field(uid, "uid"),
            _validate_record_field(path, "path"),
        )
        for uid, path in entries
    ]
    if fmt == "json":
        data = [{"uid": uid, "path": path} for uid, path in normalized_two_col]
        return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    lines = [
        f"{shlex.quote(uid)} {shlex.quote(path)}"
        for uid, path in normalized_two_col
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def write_scp_entries(
    entries: list[tuple[str, str]] | list[tuple[str, int, int, str]],
    out_path: str | Path,
    *,
    fmt: str = "scp",
    four_col: bool = False,
) -> None:
    """Write SCP / list / JSON entries to disk.

    Args:
        entries: For 2-column SCP / list: list of ``(uid, path)`` tuples.
            For 4-column SCP / JSON: list of ``(uid, sr, frames, path)`` tuples.
        out_path: Output file path.
        fmt: ``"scp"`` writes ``uid path`` (2-column) or 4-column;
            ``"list"`` writes one path per line;
            ``"json"`` writes a JSON array of ``{uid, fs, length, path}``.
        four_col: If True, write 4-column SCP ``uid sr frames path``.
    """
    out_path = Path(out_path)
    content = serialize_scp_entries(entries, fmt=fmt, four_col=four_col)
    if fmt == "scp":
        out_path = _normalize_scp_suffix(out_path)
    _atomic_write_text(content, out_path)
