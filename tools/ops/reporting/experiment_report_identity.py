"""实验报告 checkpoint identity 的纯逻辑合同与迁移分类。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

V3_SCHEMA = "explicit-checkpoint-identity-v3"
_SAFE_IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def canonical_json(payload: Mapping[str, Any]) -> str:
    """返回 identity digest 使用的 exact compact sorted UTF-8 JSON 文本。"""
    return json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")
    return value


def _label(value: str, fallback: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return label[:48] or fallback


def build_payload(
    *,
    schema: str,
    experiment: str,
    source_path: str,
    source_sha256: str,
    config_sha256: str,
) -> dict[str, str]:
    """构造 v3 精确 payload。"""
    if schema != V3_SCHEMA:
        raise ValueError(f"unsupported checkpoint identity schema: {schema}")
    if not isinstance(experiment, str) or not experiment:
        raise ValueError("invalid experiment name")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("invalid source checkpoint path")
    source = Path(source_path)
    return {
        "schema": schema,
        "experiment": experiment,
        "source_run": source.parent.name,
        "source_path": source_path,
        "source_name": source.name,
        "source_sha256": _sha256(source_sha256, "source_sha256"),
        "effective_config_sha256": _sha256(config_sha256, "config_sha256"),
    }


def identity_from_payload(payload: Mapping[str, Any]) -> str:
    """校验 canonical payload 并由其重算 identity。

    identity 是 ``<实验>@<checkpoint 文件名>``：只承载人可读的定位信息，
    内容绑定由 payload 内的 source/effective config sha256 承担；运行前后都以
    payload 复核来源文件与 effective config，漂移即 fail-closed。
    """
    schema = payload.get("schema")
    if schema != V3_SCHEMA:
        raise ValueError(f"unsupported checkpoint identity schema: {schema}")
    expected = build_payload(
        schema=schema,
        experiment=payload.get("experiment"),
        source_path=payload.get("source_path"),
        source_sha256=payload.get("source_sha256"),
        config_sha256=payload.get("effective_config_sha256"),
    )
    if dict(payload) != expected:
        raise ValueError("checkpoint identity payload is not canonical")
    identity = (
        f"{_label(expected['experiment'], 'experiment')}@"
        f"{_label(expected['source_name'], 'checkpoint')}"
    )
    if _SAFE_IDENTITY_RE.fullmatch(identity) is None:
        raise ValueError(f"unsafe checkpoint identity: {identity!r}")
    return identity


def build_identity_record(
    *,
    schema: str,
    experiment: str,
    source_path: str,
    source_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    payload = build_payload(
        schema=schema,
        experiment=experiment,
        source_path=source_path,
        source_sha256=source_sha256,
        config_sha256=config_sha256,
    )
    return {
        "checkpoint_identity_schema": schema,
        "checkpoint_identity_payload": payload,
        "checkpoint_identity": identity_from_payload(payload),
    }
