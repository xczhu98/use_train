#!/usr/bin/env python3
"""把官方 WeSep speaker cue 资源转换为 dynamic_tse nested enrollment JSON。

官方 ``cues.yaml`` 的 ``spk_id`` 资源可以直接被 ``TSEOfflineDataset`` 消费；
dynamic_tse 需要额外的 ``dataset_id`` 层。本工具只做这一层无损转换，不改变
音频内容，也不把 ``mix_spk_id`` 的逐 mixture 固定资源误当成全局 speaker 池。
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import yaml


def _trimmed(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _resolve_file(value: object, *, anchor: Path, name: str) -> str:
    raw = _trimmed(value, name=name)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = anchor / candidate
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    if not candidate.is_file():
        raise ValueError(f"{name} must point to an existing file: {candidate}")
    return str(candidate.resolve())


def _load_wesep_audio_resource(cues_yaml: Path) -> dict[str, list[str]]:
    """读取并严格解析官方 speaker-level audio resource。"""
    with cues_yaml.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    audio = (config.get("cues") or {}).get("audio")
    if not isinstance(audio, dict):
        raise ValueError(f"{cues_yaml}: missing cues.audio mapping")
    policy = audio.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(f"{cues_yaml}: missing cues.audio.policy mapping")
    key_field = _trimmed(policy.get("key"), name="cues.audio.policy.key")
    if key_field != "spk_id":
        raise ValueError(
            "cues.audio.policy.key must be 'spk_id' for dynamic_tse conversion; "
            "'mix_spk_id' is an offline fixed-trial resource"
        )
    resource_path = _resolve_file(
        _trimmed(policy.get("resource"), name="cues.audio.policy.resource"),
        anchor=cues_yaml.parent,
        name="cues.audio.policy.resource",
    )
    with Path(resource_path).open(encoding="utf-8") as handle:
        resource = json.load(handle)
    if not isinstance(resource, dict) or not resource:
        raise ValueError(f"{resource_path}: resource must be a non-empty JSON object")

    output: dict[str, list[str]] = {}
    resource_anchor = Path(resource_path).parent
    for speaker_id, items in resource.items():
        speaker = _trimmed(speaker_id, name="speaker resource key")
        if not isinstance(items, list) or not items:
            raise ValueError(f"{resource_path}: {speaker!r} must have a non-empty list")
        paths: list[str] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                raw_path = item
            elif isinstance(item, dict):
                raw_path = item.get("path")
            else:
                raise TypeError(
                    f"{resource_path}: {speaker!r}[{index}] must be a path or mapping"
                )
            path = _resolve_file(
                _trimmed(raw_path, name=f"{speaker!r}[{index}].path"),
                anchor=resource_anchor,
                name=f"{speaker!r}[{index}].path",
            )
            if path not in paths:
                paths.append(path)
        output[speaker] = paths
    return output


def _validate_speech_coverage(
    speech_jsonl: Path, *, dataset_id: str, enrollment: dict[str, list[str]]
) -> None:
    """可选校验 strict speech manifest 的 speaker 覆盖度。"""
    identities: set[str] = set()
    with speech_jsonl.open(encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            location = f"{speech_jsonl}:{lineno}"
            payload: Any = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(f"{location}: expected a JSON object")
            record_dataset = _trimmed(
                payload.get("dataset_id"), name=f"{location}.dataset_id"
            )
            speaker = _trimmed(
                payload.get("speaker_id"), name=f"{location}.speaker_id"
            )
            if record_dataset != dataset_id:
                raise ValueError(
                    f"{location}: dataset_id={record_dataset!r} does not match "
                    f"--dataset-id={dataset_id!r}"
                )
            identities.add(speaker)
    missing = sorted(identity for identity in identities if not enrollment.get(identity))
    if missing:
        raise ValueError(
            f"{speech_jsonl}: missing enrollment candidates for {len(missing)} speakers: "
            f"{missing[:5]}"
        )


def _atomic_write_json(
    path: Path,
    payload: dict[str, dict[str, list[str]]],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {path}")
    if path.is_symlink():
        raise ValueError(f"output must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert official WeSep speaker cues to dynamic_tse nested enrollment JSON"
    )
    parser.add_argument("--cues-yaml", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--speech-jsonl",
        help="optional strict dynamic speech manifest used to verify speaker coverage",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cues_yaml = Path(args.cues_yaml).expanduser().resolve(strict=True)
    dataset_id = _trimmed(args.dataset_id, name="--dataset-id")
    output = Path(args.output_json).expanduser()
    enrollment = _load_wesep_audio_resource(cues_yaml)
    if args.speech_jsonl:
        speech_jsonl = Path(args.speech_jsonl).expanduser().resolve(strict=True)
        _validate_speech_coverage(
            speech_jsonl, dataset_id=dataset_id, enrollment=enrollment
        )
    _atomic_write_json(
        output,
        {dataset_id: enrollment},
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
