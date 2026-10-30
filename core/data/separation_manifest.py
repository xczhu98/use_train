"""Strict JSONL manifest parsing for Speech Separation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    path: Path
    length: int


@dataclass(frozen=True)
class SeparationRecord:
    uid: str
    sample_rate: int
    length: int
    mixture: Path
    sources: tuple[SourceRecord, ...]


def _positive_int(value: object, *, field: str, location: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{location}.{field} must be an exact positive int")
    return value


def _relative_resource(value: object, *, root: Path, field: str, location: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{field} must be a non-empty path string")
    path = Path(value.strip())
    return path if path.is_absolute() else (root / path)


def parse_separation_manifest(path: str | Path) -> tuple[SeparationRecord, ...]:
    """Read strict one-object-per-line mixture/source JSONL."""
    manifest = Path(path).expanduser().resolve()
    records: list[SeparationRecord] = []
    seen: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            location = f"{manifest}:{line_number}"
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location} is not valid JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{location} must be a JSON object")
            required = {"uid", "fs", "length", "mixture", "sources"}
            unknown = set(value) - required
            missing = required - set(value)
            if missing or unknown:
                raise ValueError(
                    f"{location} fields mismatch; missing={sorted(missing)}, "
                    f"unknown={sorted(unknown)}"
                )
            uid = value["uid"]
            if not isinstance(uid, str) or not uid.strip() or uid in seen:
                raise ValueError(f"{location}.uid must be unique and non-empty")
            seen.add(uid)
            sample_rate = _positive_int(value["fs"], field="fs", location=location)
            length = _positive_int(value["length"], field="length", location=location)
            mixture = _relative_resource(
                value["mixture"], root=manifest.parent, field="mixture", location=location
            )
            raw_sources = value["sources"]
            if not isinstance(raw_sources, list) or not raw_sources:
                raise ValueError(f"{location}.sources must be a non-empty list")
            sources: list[SourceRecord] = []
            source_ids: set[str] = set()
            for index, raw_source in enumerate(raw_sources):
                source_location = f"{location}.sources[{index}]"
                if not isinstance(raw_source, dict):
                    raise ValueError(f"{source_location} must be an object")
                if set(raw_source) != {"source_id", "path", "length"}:
                    raise ValueError(f"{source_location} fields must be source_id/path/length")
                source_id = raw_source["source_id"]
                if not isinstance(source_id, str) or not source_id.strip() or source_id in source_ids:
                    raise ValueError(f"{source_location}.source_id must be unique and non-empty")
                source_ids.add(source_id)
                sources.append(
                    SourceRecord(
                        source_id=source_id,
                        path=_relative_resource(
                            raw_source["path"],
                            root=manifest.parent,
                            field="path",
                            location=source_location,
                        ),
                        length=_positive_int(
                            raw_source["length"], field="length", location=source_location
                        ),
                    )
                )
            records.append(
                SeparationRecord(
                    uid=uid,
                    sample_rate=sample_rate,
                    length=length,
                    mixture=mixture,
                    sources=tuple(sources),
                )
            )
    if not records:
        raise ValueError(f"{manifest} contains no separation records")
    return tuple(records)
