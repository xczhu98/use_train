"""共享 source identity 索引；身份用于采样约束，不等同于分类标签。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.data.noise_manifest_adapter import read_noise_source_manifest


@dataclass(frozen=True)
class SourcePoolRecord:
    uid: str
    speaker_id: str | None
    sample_rate: int
    length: int
    path: Path
    dataset_id: str = ""

    @property
    def identity(self) -> tuple[str, str]:
        # 没有身份时只保证不同 source；保留独立 namespace，避免与真实 speaker 相撞。
        if self.speaker_id is None:
            return (f"source:{self.dataset_id}", self.uid)
        return (self.dataset_id, self.speaker_id)


def parse_source_pool(path: str | Path, *, require_speaker: bool = False) -> tuple[SourcePoolRecord, ...]:
    """接受共享四列 SCP/JSONL；SS 可无身份，TSE 可要求完整 canonical identity。"""
    parsed = read_noise_source_manifest(path)
    records = []
    for uid, audio_path in parsed.flat.items():
        metadata = parsed.metadata.get(uid, {})
        speaker = metadata.get("speaker_id")
        dataset = metadata.get("dataset_id", str(Path(path).resolve()))
        if speaker is not None and (
            not isinstance(speaker, str) or not speaker or speaker != speaker.strip()
            or speaker.casefold() == "unknown"
        ):
            raise ValueError(f"{uid}: speaker_id must be a non-empty trimmed known identity")
        if not isinstance(dataset, str) or not dataset or dataset != dataset.strip():
            raise ValueError(f"{uid}: dataset_id must be a non-empty trimmed string")
        if require_speaker and (speaker is None or "dataset_id" not in metadata):
            raise ValueError(f"{uid}: canonical dataset_id/speaker_id required")
        sr = next(sr for sr, pool in parsed.by_fs.items() if uid in pool)
        records.append(SourcePoolRecord(uid, speaker, sr, parsed.lengths[uid], Path(audio_path), dataset))
    if not records:
        raise ValueError(f"{path}: no source-pool records")
    return tuple(records)


def sample_distinct_identities(
    groups: Mapping[tuple[str, str], tuple[str, ...]],
    count: int,
    rng: np.random.Generator,
    *,
    exclude: tuple[tuple[str, str], ...] = (),
    shortage: str = "error",
) -> list[str]:
    """先按不同身份等概率抽样，再在身份内抽 utterance。"""
    if type(count) is not int or count < 0 or shortage not in {"error", "reduce"}:
        raise ValueError("invalid source count or shortage policy")
    identities = [identity for identity, uids in groups.items() if uids and identity not in exclude]
    if count > len(identities) and shortage == "error":
        raise ValueError(f"requested {count} distinct speakers, available {len(identities)}")
    count = min(count, len(identities))
    selected = rng.choice(len(identities), size=count, replace=False)
    return [str(rng.choice(groups[identities[int(index)]])) for index in selected]
