#!/usr/bin/env python3
"""AEC Challenge 扫描逻辑；GenerationManifest publication 已下线。

迁移计划：AEC schema v2 reload 就绪后，把 ``main`` 的发布改走
``tools.prepare_data_tools.offline_publication.publish_owned_offline_dataset_roots``；
迁移完成前 ``main`` 以外的骨架保留仅供内部测试，不得作为生产发布路径。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tools.shared.scp_io import write_scp_entries  # noqa: E402
from tools.prepare_data_tools.generation_manifest import sha256_file  # noqa: E402
from tools.prepare_data_tools.offline_publication import OWNER_FILE  # noqa: E402

GENERATOR_VERSION = "1.1"
GENERATOR_ID = "tools/prepare_data_tools/scp/prepare_aec_challenge_offline.py"
OWNER_ID = "prepare-aec-challenge-offline/v1"


@dataclass(frozen=True)
class AudioMeta:
    samplerate: int
    frames: int
    channels: int


@dataclass(frozen=True)
class Candidate:
    uid: str
    source: str
    mic: Path
    farend: Path
    nearend: Path | None
    meta: AudioMeta
    target_kind: str
    nearend_scale: float = 1.0


def _wav_index(root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    if not root.is_dir():
        return result
    for path in sorted(root.rglob("*.wav")):
        if path.is_file():
            result.setdefault(path.stem, []).append(path)
    return result


def _probe(path: Path) -> tuple[AudioMeta | None, str | None]:
    try:
        info = sf.info(path)
    except Exception:
        return None, "unreadable_wav"
    meta = AudioMeta(int(info.samplerate), int(info.frames), int(info.channels))
    if meta.samplerate <= 0 or meta.frames <= 0 or meta.channels <= 0:
        return None, "invalid_audio_metadata"
    return meta, None


def _contract(paths: list[Path]) -> tuple[AudioMeta | None, str | None]:
    metas = []
    for path in paths:
        meta, reason = _probe(path)
        if reason:
            return None, reason
        metas.append(meta)
    if len(set(metas)) != 1:
        return None, "audio_contract_mismatch"
    return metas[0], None


def _select(index: dict[str, list[Path]], stem: str) -> tuple[Path | None, str | None]:
    paths = index.get(stem, [])
    if not paths:
        return None, "missing_track"
    if len(paths) != 1:
        return None, "duplicate_track"
    return paths[0], None


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing metadata CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path}: missing CSV columns {missing}")
        return list(reader)


def _record_rejection(stats: dict[str, object], reason: str) -> None:
    stats["rejected"] += 1
    stats["reasons"][reason] += 1


def _new_stats() -> dict[str, object]:
    return {"accepted": 0, "rejected": 0, "reasons": Counter()}


def _scan_fullband(root: Path) -> tuple[list[Candidate], dict[str, object]]:
    stats = _new_stats()
    rows = _read_csv(root / "meta.csv", {"file_id"})
    index = _wav_index(root)
    seen: set[str] = set()
    accepted: list[Candidate] = []
    for row in rows:
        fileid = (row.get("file_id") or "").strip()
        if not fileid or any(char in fileid for char in "\r\n\0"):
            _record_rejection(stats, "invalid_file_id")
            continue
        if fileid in seen:
            _record_rejection(stats, "duplicate_meta_id")
            continue
        seen.add(fileid)
        selected = []
        reason = None
        for suffix in ("_mic", "_farend", "_target"):
            path, reason = _select(index, fileid + suffix)
            if reason:
                break
            selected.append(path)
        if reason:
            _record_rejection(stats, reason)
            continue
        meta, reason = _contract(selected)
        if reason:
            _record_rejection(stats, reason)
            continue
        accepted.append(Candidate(
            uid=f"fullband/{fileid}", source="fullband",
            mic=selected[0], farend=selected[1], nearend=selected[2],
            meta=meta, target_kind="copy",
        ))
        stats["accepted"] += 1
    return accepted, stats


_DT_RE = re.compile(r"(?:^|[^a-z0-9])(?:dt|double[_ -]?talk)(?:[^a-z0-9]|$)")
_FEST_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:fest|far[_ -]?end[_ -]?(?:single[_ -]?talk|singletalk))(?:[^a-z0-9]|$)"
)


def _real_scenario(key: str) -> str:
    normalized = key.lower()
    if _DT_RE.search(normalized):
        return "double_talk"
    if _FEST_RE.search(normalized):
        return "farend_single_talk"
    return "unknown"


def _scan_real(root: Path, source: str) -> tuple[list[Candidate], dict[str, object]]:
    stats = _new_stats()
    roles: dict[str, dict[str, list[Path]]] = {"mic": {}, "lpb": {}}
    if not root.is_dir():
        raise ValueError(f"input root is not a directory: {root}")
    for path in sorted(root.rglob("*.wav")):
        relative = path.relative_to(root).as_posix()
        for role in roles:
            suffix = f"_{role}.wav"
            if relative.lower().endswith(suffix):
                key = relative[:-len(suffix)]
                roles[role].setdefault(key, []).append(path)
                break
    accepted: list[Candidate] = []
    for key in sorted(set(roles["mic"]) | set(roles["lpb"])):
        mic_paths = roles["mic"].get(key, [])
        lpb_paths = roles["lpb"].get(key, [])
        if not mic_paths or not lpb_paths:
            _record_rejection(stats, "missing_track")
            continue
        if len(mic_paths) != 1 or len(lpb_paths) != 1:
            _record_rejection(stats, "duplicate_track")
            continue
        scenario = _real_scenario(key)
        if scenario == "double_talk":
            _record_rejection(stats, "double_talk_no_aligned_clean_target")
            continue
        if scenario != "farend_single_talk":
            _record_rejection(stats, "unsupported_real_scenario")
            continue
        meta, reason = _contract([mic_paths[0], lpb_paths[0]])
        if reason:
            _record_rejection(stats, reason)
            continue
        accepted.append(Candidate(
            uid=f"{source}/{key}", source=source,
            mic=mic_paths[0], farend=lpb_paths[0], nearend=None,
            meta=meta, target_kind="zero",
        ))
        stats["accepted"] += 1
    return accepted, stats


def _inventory(root: Path) -> list[dict[str, object]]:
    items = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == OWNER_FILE:
            continue
        items.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "kind": "waveform" if path.suffix.lower() == ".wav" else "metadata",
        })
    return items


def _materialize(staging: Path, candidates: list[Candidate]) -> None:
    entries = {name: [] for name in ("mic", "farend", "nearend")}
    generated_index = 0
    for candidate in sorted(candidates, key=lambda item: item.uid):
        paths = {
            "mic": str(candidate.mic),
            "farend": str(candidate.farend),
            "nearend": str(candidate.nearend),
        }
        if candidate.target_kind != "copy":
            relative = Path("generated_targets") / f"{generated_index:08d}.wav"
            generated_index += 1
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if candidate.target_kind == "scaled":
                waveform, samplerate = sf.read(
                    candidate.nearend, dtype="float32", always_2d=True
                )
                waveform = waveform * np.float32(candidate.nearend_scale)
                if not np.isfinite(waveform).all():
                    raise ValueError(f"non-finite scaled nearend: {candidate.uid}")
                sf.write(destination, waveform, samplerate, subtype="FLOAT")
            else:
                zeros = np.zeros(
                    (candidate.meta.frames, candidate.meta.channels), dtype=np.float32
                )
                sf.write(
                    destination, zeros, candidate.meta.samplerate, subtype="FLOAT"
                )
            actual, reason = _probe(destination)
            if reason or actual != candidate.meta:
                raise ValueError(
                    f"materialized metadata mismatch: uid={candidate.uid} "
                    f"expected={candidate.meta} actual={actual} reason={reason}"
                )
            paths["nearend"] = relative.as_posix()
        for name in entries:
            entries[name].append((
                candidate.uid,
                candidate.meta.samplerate,
                candidate.meta.frames,
                paths[name],
            ))
    for name, track_entries in entries.items():
        write_scp_entries(track_entries, staging / f"{name}.scp", four_col=True)


def _validate_staged_dataset(staging: Path, expected_size: int) -> None:
    from tasks.aec.data.offline_dataset import AECOfflineDataset

    dataset = AECOfflineDataset(
        mic_scp=staging / "mic.scp",
        farend_scp=staging / "farend.scp",
        nearend_scp=staging / "nearend.scp",
        strict_load=True,
    )
    if len(dataset) != expected_size:
        raise ValueError(
            f"staged dataset size mismatch: expected={expected_size} actual={len(dataset)}"
        )
    for index in range(len(dataset)):
        dataset[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a full AEC Challenge multi-source offline SCP dataset"
    )
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--synthetic-fullband-dir", required=True)
    parser.add_argument("--real-fullband-dir", required=True)
    parser.add_argument("--real-mobile-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print(
        "ERROR: GenerationManifest v1 publication 已下线；schema v2 reload 当前仅支持 SE；"
        "AEC offline publication 需要迁移后重新生成。",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
