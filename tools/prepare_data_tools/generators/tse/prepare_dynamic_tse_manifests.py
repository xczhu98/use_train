#!/usr/bin/env python3
"""从显式单说话人目录清单生成 dynamic_tse 严格资源。

输入 JSONL 每行必须包含：

``{"uid", "path", "language", "dataset_id", "speaker_id", "enrollments"}``

仅额外接受 ``recording_id`` 与 ``generator_version``。启用录音身份的 dataset
必须为全部 speech/enrollment 路径声明身份；版本字段保留上游生成器的内容版本。

``enrollments`` 是非空路径数组。音频路径相对输入 JSONL 所在目录解析，输出为
绝对路径；采样率和帧数从音频实际读取。两个输出以同一文件事务发布：

- strict speech JSONL：``uid/fs/length/path/language/dataset_id/speaker_id``；
- nested enrollment JSON：``dataset_id -> speaker_id -> [paths]``。

REAL-T 是 evaluation benchmark，任何位于 ``data/tse/REAL-TSE`` / ``REAL-T-*``
目录下的输入或音频都永久拒绝进入本训练 producer。
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.data.simulation_utils import read_noise_source_manifest  # noqa: E402
from core.utils.file_transaction import write_text_files_transactionally  # noqa: E402
from tools.prepare_data_tools.manifest_contracts import (  # noqa: E402
    PROFILE_DYNAMIC_SPEECH,
    inspect_manifest,
)
from tasks.tse.manifest_contract import (  # noqa: E402
    build_recording_ids,
    build_speaker_identities,
    parse_enrollment_manifest,
)

_FIELDS = {"uid", "path", "language", "dataset_id", "speaker_id", "enrollments"}
_OPTIONAL_FIELDS = {"recording_id", "generator_version"}


def _trimmed(value: object, *, field: str, location: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{location}: {field} must be a non-empty trimmed string")
    return value


def _looks_like_realt(path: Path) -> bool:
    parts = {part.casefold() for part in path.resolve(strict=False).parts}
    return "real-tse" in parts or any(part.startswith("real-t-") for part in parts)


def _regular_audio(
    path_value: object,
    *,
    anchor: Path,
    location: str,
    cache: dict[str, tuple[Path, int, int]] | None = None,
) -> tuple[Path, int, int]:
    raw = _trimmed(path_value, field="path", location=location)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = anchor / candidate
    absolute = Path(os.path.abspath(os.fspath(candidate)))
    if _looks_like_realt(absolute):
        raise PermissionError(
            f"{location}: REAL-T evaluation assets cannot enter dynamic_tse training: {absolute}"
        )
    cache_key = str(absolute)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    file_stat = absolute.lstat()
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"{location}: audio must be a regular non-symlink file: {absolute}")
    info = sf.info(absolute)
    if info.samplerate <= 0 or info.frames <= 0:
        raise ValueError(f"{location}: audio is empty or has invalid sample rate: {absolute}")
    result = absolute.resolve(strict=True), int(info.samplerate), int(info.frames)
    if cache is not None:
        cache[cache_key] = result
    return result


def _load_source_records(source_jsonl: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[str]]]]:
    if _looks_like_realt(source_jsonl):
        raise PermissionError("REAL-T evaluation manifests cannot enter dynamic_tse training")
    records: list[dict[str, Any]] = []
    enrollments: dict[str, dict[str, list[str]]] = {}
    seen_uids: set[str] = set()
    audio_cache: dict[str, tuple[Path, int, int]] = {}
    with source_jsonl.open(encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            location = f"{source_jsonl}:{lineno}"
            payload = json.loads(raw_line)
            if (not isinstance(payload, dict) or not _FIELDS <= set(payload)
                    or set(payload) - _FIELDS - _OPTIONAL_FIELDS):
                actual = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
                raise ValueError(
                    f"{location}: expected exact fields {sorted(_FIELDS)} with optional "
                    f"{sorted(_OPTIONAL_FIELDS)}, got {actual}"
                )
            uid = _trimmed(payload["uid"], field="uid", location=location)
            language = _trimmed(payload["language"], field="language", location=location)
            dataset_id = _trimmed(payload["dataset_id"], field="dataset_id", location=location)
            speaker_id = _trimmed(payload["speaker_id"], field="speaker_id", location=location)
            if speaker_id.casefold() == "unknown":
                raise ValueError(f"{location}: speaker_id must not be unknown")
            if uid in seen_uids:
                raise ValueError(f"{location}: duplicate uid {uid!r}")
            seen_uids.add(uid)
            speech_path, sample_rate, length = _regular_audio(
                payload["path"], anchor=source_jsonl.parent, location=location,
                cache=audio_cache,
            )
            raw_enrollments = payload["enrollments"]
            if not isinstance(raw_enrollments, list) or not raw_enrollments:
                raise ValueError(f"{location}: enrollments must be a non-empty array")
            resolved_enrollments = []
            for index, value in enumerate(raw_enrollments):
                path, _, _ = _regular_audio(
                    value,
                    anchor=source_jsonl.parent,
                    location=f"{location}.enrollments[{index}]",
                    cache=audio_cache,
                )
                resolved_enrollments.append(str(path))
            bucket = enrollments.setdefault(dataset_id, {}).setdefault(speaker_id, [])
            bucket.extend(path for path in resolved_enrollments if path not in bucket)
            records.append({
                "uid": uid,
                "fs": sample_rate,
                "length": length,
                "path": str(speech_path),
                "language": language,
                "dataset_id": dataset_id,
                "speaker_id": speaker_id,
            })
            if "recording_id" in payload:
                records[-1]["recording_id"] = _trimmed(
                    payload["recording_id"], field="recording_id", location=location,
                )
            if "generator_version" in payload:
                version = payload["generator_version"]
                if type(version) is not int or version <= 0:
                    raise ValueError(f"{location}: generator_version must be a positive integer")
                records[-1]["generator_version"] = version
    if not records:
        raise ValueError(f"{source_jsonl}: no records")
    return records, enrollments


def _validate_outputs(speech_jsonl: Path, spk2enroll_json: Path) -> None:
    inspect_manifest(speech_jsonl, profile=PROFILE_DYNAMIC_SPEECH)
    parsed = read_noise_source_manifest(speech_jsonl)
    identities = build_speaker_identities(parsed.metadata, str(speech_jsonl))
    enrollments = parse_enrollment_manifest(spk2enroll_json, set(identities.values()))
    build_recording_ids(parsed.metadata, enrollments, str(speech_jsonl))


def _write_text(handle, content: str) -> None:
    handle.write(content)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare strict speech JSONL and nested enrollment JSON for dynamic_tse"
    )
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--speech-jsonl", required=True)
    parser.add_argument("--spk2enroll-json", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    source = Path(args.source_jsonl).expanduser().resolve(strict=True)
    speech_output = Path(args.speech_jsonl).expanduser()
    enrollment_output = Path(args.spk2enroll_json).expanduser()
    if speech_output.resolve(strict=False) == enrollment_output.resolve(strict=False):
        parser.error("--speech-jsonl and --spk2enroll-json must be different files")
    records, enrollments = _load_source_records(source)
    speech_content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    enrollment_content = json.dumps(
        enrollments, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"

    with tempfile.TemporaryDirectory(prefix="dynamic-tse-manifest-check-") as temp_dir:
        temp_root = Path(temp_dir)
        temp_speech = temp_root / "speech.jsonl"
        temp_enrollment = temp_root / "spk2enroll.json"
        temp_speech.write_text(speech_content, encoding="utf-8")
        temp_enrollment.write_text(enrollment_content, encoding="utf-8")
        _validate_outputs(temp_speech, temp_enrollment)

    write_text_files_transactionally(
        (
            (speech_output, lambda handle: _write_text(handle, speech_content)),
            (enrollment_output, lambda handle: _write_text(handle, enrollment_content)),
        ),
        overwrite=args.overwrite,
    )
    _validate_outputs(speech_output, enrollment_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
