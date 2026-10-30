"""SCP provenance sidecar：schema 构造、原子写、新鲜度与完整性校验的唯一实现。

sidecar 文件命名为 ``<scp>.provenance.json``，把一份 SCP 与其来源绑定：
``entries``（严格解析口径的条目数）与 ``scp_sha256``（内容 bytes 哈希）保证
sidecar 与 SCP 内容一致；``kind``/``source`` 及调用方附加键（如
``source_snapshot``/``source_fingerprint``/``pattern``）记录来源语义。

- ``kind="generated"``：从目录或条目清单生成，schema version 1
- ``kind="copied"``：从既有 SCP 复制（路径重写），schema version 2

``version`` 编号由本模块私有管理；新鲜度比较（:func:`scp_provenance_fresh`）
为子集语义，调用方不应在 ``expected`` 中包含 ``version``。
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from tools.shared.scp_io import parse_scp_fields

SCP_SIDECAR_SUFFIX = ".provenance.json"

_KIND_VERSIONS = {"generated": 1, "copied": 2}

# 由本模块统一填充的保留键，禁止调用方通过 extra 覆盖
_RESERVED_KEYS = frozenset(
    {"version", "kind", "source", "created_at", "entries", "scp_sha256"}
)


def scp_sidecar_path(scp_path: str | Path) -> Path:
    return Path(f"{scp_path}{SCP_SIDECAR_SUFFIX}")


def parse_scp_snapshot(source: Path, content: bytes) -> dict[str, str]:
    """仅解析给定 bytes snapshot（严格 2/4 列），不在解析过程中重新读盘。"""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: SCP 不是有效 UTF-8") from exc
    parsed: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = parse_scp_fields(line, source, lineno, allowed_columns=(2, 4))
        uid, audio_path = cols[0], cols[-1]
        if uid in parsed:
            raise ValueError(f"{source}:{lineno}: 重复 uid: {uid!r}")
        parsed[uid] = audio_path
    return parsed


def build_scp_provenance(
    scp_content: bytes | str,
    *,
    kind: str,
    source: str,
    scp_path: str | Path | None = None,
    **extra: Any,
) -> dict:
    """从 SCP 内容构造 sidecar payload；内容非法（解析失败/重复 uid）时 fail-closed。

    ``scp_path`` 仅用于解析报错定位，不要求文件已存在。
    """
    if kind not in _KIND_VERSIONS:
        raise ValueError(f"未知 SCP provenance kind: {kind!r}")
    reserved = _RESERVED_KEYS & set(extra)
    if reserved:
        raise TypeError(f"provenance 保留键不允许通过 extra 传入: {sorted(reserved)}")
    data = scp_content.encode("utf-8") if isinstance(scp_content, str) else scp_content
    parsed = parse_scp_snapshot(
        Path(scp_path) if scp_path is not None else Path("<scp-content>"), data
    )
    return {
        "version": _KIND_VERSIONS[kind],
        "kind": kind,
        "source": source,
        **extra,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "entries": len(parsed),
        "scp_sha256": hashlib.sha256(data).hexdigest(),
    }


def write_scp_provenance(scp_path: str | Path, payload: Mapping[str, Any]) -> Path:
    """以 tmp + os.replace 原子写 sidecar，返回 sidecar 路径。"""
    sidecar = scp_sidecar_path(scp_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    tmp = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, sidecar)
    return sidecar


def scp_provenance_fresh(scp_path: str | Path, expected: Mapping[str, Any]) -> bool:
    """sidecar 与当前 SCP bytes 一致且逐键匹配 ``expected`` 子集时返回 True。"""
    scp = Path(scp_path)
    sidecar = scp_sidecar_path(scp)
    if not scp.exists() or not sidecar.exists():
        return False
    try:
        actual = json.loads(sidecar.read_text(encoding="utf-8"))
        content = scp.read_bytes()
        parse_scp_snapshot(scp, content)
        digest = hashlib.sha256(content).hexdigest()
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(actual, dict):
        return False
    return actual.get("scp_sha256") == digest and all(
        actual.get(key) == value for key, value in expected.items()
    )


def validate_scp_provenance(scp_path: str | Path) -> dict:
    """校验已有 SCP sidecar 确实绑定当前 SCP bytes，fail-closed。"""
    scp = Path(scp_path).expanduser().resolve()
    sidecar = scp_sidecar_path(scp)
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        content = scp.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"SCP provenance sidecar missing: {sidecar}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"SCP provenance sidecar is invalid: {sidecar}") from exc
    # 解析失败的 ValueError 直接外抛，保留具体行号与原因
    parsed = parse_scp_snapshot(scp, content)
    if not isinstance(payload, dict) or payload.get("kind") not in _KIND_VERSIONS:
        raise ValueError(f"SCP provenance sidecar schema is invalid: {sidecar}")
    if payload.get("entries") != len(parsed):
        raise ValueError(f"SCP provenance entry count mismatch: {scp}")
    if payload.get("scp_sha256") != hashlib.sha256(content).hexdigest():
        raise ValueError(f"SCP provenance fingerprint mismatch: {scp}")
    return payload
