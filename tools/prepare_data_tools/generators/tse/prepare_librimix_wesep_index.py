#!/usr/bin/env python3
"""从 Libri2Mix ``wav16k/min`` 音频生成 WeSep v0.1 speaker-cue 清单。

对每个 split 在 ``<librimix_root>/<split>/`` 下发布四个文件：
- ``samples.jsonl``：``key`` / ``spk`` / ``mix.default`` / ``src``（绝对路径），
  三路音频（mix_clean、s1、s2）逐条校验同名、采样率、声道数与帧数一致；
- ``cues.yaml``：audio cue（``wav`` / ``waveform``，``scope: speaker``，
  ``policy.key: spk_id``）；``train-*`` split 写 ``policy.type: random``，其余写
  ``fixed``，仅作兜底，``enroll_policy`` 由配置层（train=random / validation=fixed）
  显式覆盖；
- ``enroll.json``：``spk_id -> [{"path", "duration_sec"}]``，由
  ``<enroll_root>/<split>/spk2enroll.json``（dynamic_tse nested JSON）转换，
  逐条校验音频存在、时长为正，且清单中出现的 speaker 必须全部被覆盖；
- ``raw.list``：WeSep 约定的 ``samples.jsonl`` 索引。

已存在的产物文件内容一致时跳过，不一致时报错并要求显式 ``--overwrite``。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from tools.prepare_data_tools.generators.tse.mc_librimix.scan_librimix import speaker_ids
else:
    from .mc_librimix.scan_librimix import speaker_ids

_SPLITS = ("train-100", "dev", "test")
_PRODUCTS = ("samples.jsonl", "cues.yaml", "enroll.json", "raw.list")


def _sf_info(path: Path):
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("扫描 Libri2Mix 音频需要 soundfile") from exc
    return sf.info(str(path))


def _scan_split(split_root: Path, mix_dir_name: str) -> list[dict]:
    """扫描一个 split 的三路音频，返回 WeSep samples 记录。"""
    mix_dir = split_root / mix_dir_name
    s1_dir = split_root / "s1"
    s2_dir = split_root / "s2"
    for directory in (mix_dir, s1_dir, s2_dir):
        if not directory.is_dir():
            raise NotADirectoryError(f"split 目录缺失: {directory}")
    inventories = []
    for directory in (mix_dir, s1_dir, s2_dir):
        inventory = {p.stem: p for p in directory.glob("*.wav")}
        if not inventory:
            raise ValueError(f"split 目录为空: {directory}")
        inventories.append(inventory)
    keys = set(inventories[0])
    for inventory in inventories[1:]:
        if set(inventory) != keys:
            raise ValueError(f"mix/s1/s2 的 WAV ID 不完全一致: {split_root}")

    records = []
    for key in sorted(keys):
        spk1, spk2 = speaker_ids(key)
        paths = [inventory[key].resolve() for inventory in inventories]
        infos = [_sf_info(path) for path in paths]
        shapes = {(i.frames, i.samplerate, i.channels) for i in infos}
        if len(shapes) != 1:
            raise ValueError(f"三路音频 frames/samplerate/channels 不一致: {key}")
        frames, samplerate, channels = shapes.pop()
        if frames <= 0 or samplerate <= 0 or channels <= 0:
            raise ValueError(f"音频元数据非正: {key}")
        records.append(
            {
                "key": key,
                "spk": [spk1, spk2],
                "mix": {"default": [str(paths[0])]},
                "src": {spk1: [str(paths[1])], spk2: [str(paths[2])]},
            }
        )
    return records


def _load_enrollments(enroll_root: Path, split: str, speakers: set[str]) -> dict[str, list[dict]]:
    """把 dynamic_tse nested spk2enroll JSON 转换为 offline audio cue 资源。"""
    path = enroll_root / split / "spk2enroll.json"
    if not path.is_file():
        raise FileNotFoundError(f"enrollment 清单缺失: {path}")
    with path.open(encoding="utf-8") as stream:
        nested = json.load(stream)
    if not isinstance(nested, dict) or len(nested) != 1:
        raise ValueError(f"{path} 必须是单 dataset_id 的 nested JSON")
    dataset_id, by_speaker = next(iter(nested.items()))
    if not isinstance(by_speaker, dict) or not by_speaker:
        raise ValueError(f"{path}[{dataset_id!r}] 必须是非空 speaker 映射")

    missing = sorted(speakers - set(by_speaker))
    if missing:
        raise ValueError(f"{path} 缺少 split 内 speaker 的 enrollment: {missing[:5]}")

    resource: dict[str, list[dict]] = {}
    for spk in sorted(speakers):
        entries = by_speaker[spk]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{path}[{dataset_id!r}][{spk!r}] 必须是非空路径数组")
        items = []
        for raw in entries:
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"{path}[{dataset_id!r}][{spk!r}] 含非字符串路径")
            wav = Path(raw).resolve()
            if not wav.is_file():
                raise FileNotFoundError(f"enrollment 音频不存在: {wav}")
            info = _sf_info(wav)
            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError(f"enrollment 音频元数据非正: {wav}")
            items.append(
                {
                    "path": str(wav),
                    "duration_sec": round(info.frames / info.samplerate, 6),
                }
            )
        resource[spk] = items
    return resource


def _publish(path: Path, content: str, overwrite: bool) -> str:
    """内容一致则跳过；不一致需显式 --overwrite；原子落盘。"""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return "reused"
        if not overwrite:
            raise FileExistsError(f"产物已存在且内容不同（用 --overwrite 覆盖）: {path}")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
    return "written"


def prepare_split(
    librimix_root: Path,
    enroll_root: Path,
    split: str,
    *,
    mix_dir_name: str,
    overwrite: bool,
) -> dict:
    split_root = librimix_root / split
    records = _scan_split(split_root, mix_dir_name)
    speakers = {spk for record in records for spk in record["spk"]}
    enroll = _load_enrollments(enroll_root, split, speakers)

    samples_text = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    policy_type = "random" if split.startswith("train") else "fixed"
    cues_text = yaml.safe_dump(
        {
            "cues": {
                "audio": {
                    "type": "wav",
                    "format": "waveform",
                    "guaranteed": True,
                    "scope": "speaker",
                    "policy": {
                        "type": policy_type,
                        "key": "spk_id",
                        "resource": "enroll.json",
                    },
                }
            }
        },
        sort_keys=False,
    )
    enroll_text = json.dumps(enroll, ensure_ascii=False, indent=1) + "\n"

    actions = {}
    for name, content in (
        ("samples.jsonl", samples_text),
        ("cues.yaml", cues_text),
        ("enroll.json", enroll_text),
        ("raw.list", "samples.jsonl\n"),
    ):
        actions[name] = _publish(split_root / name, content, overwrite)
    return {
        "split": split,
        "samples": len(records),
        "speakers": len(speakers),
        "enroll_items": sum(len(v) for v in enroll.values()),
        "actions": actions,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="从 Libri2Mix wav16k/min 音频生成 WeSep v0.1 speaker-cue 清单。",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--librimix-root",
        required=True,
        help="Libri2Mix wav16k/min 根目录（含 <split>/{mix_clean,s1,s2}/）",
    )
    parser.add_argument(
        "--enroll-root",
        required=True,
        help="dynamic_tse 清单根目录（含 <split>/spk2enroll.json）",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=_SPLITS,
        default=["train-100", "dev"],
        help="要生成的 split（默认 train-100 dev）",
    )
    parser.add_argument(
        "--mix-dir",
        default="mix_clean",
        help="mixture 音频目录名（默认 mix_clean）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在且内容不同的产物",
    )
    args = parser.parse_args(argv)

    librimix_root = Path(args.librimix_root).expanduser().resolve()
    enroll_root = Path(args.enroll_root).expanduser().resolve()
    if not librimix_root.is_dir():
        raise NotADirectoryError(f"librimix 根目录不存在: {librimix_root}")

    stats = [
        prepare_split(
            librimix_root,
            enroll_root,
            split,
            mix_dir_name=args.mix_dir,
            overwrite=args.overwrite,
        )
        for split in args.splits
    ]
    print(json.dumps({"splits": stats}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
