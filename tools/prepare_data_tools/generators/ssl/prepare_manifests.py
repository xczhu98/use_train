#!/usr/bin/env python3
"""整理本地 LibriSpeech 与噪声 SCP 为 dynamic_ssl 四列清单。

音频文件保留在原位置；本工具只生成 SCP 和 provenance JSON，不复制或重采样音频。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.utils.file_transaction import write_text_files_transactionally  # noqa: E402
from tools.shared.scp_io import parse_scp_fields, serialize_scp_entries  # noqa: E402
from tools.shared.scp_io import serialize_manifest_relative_path  # noqa: E402


GENERATOR_ID = "tools/prepare_data_tools/generators/ssl/prepare_manifests.py"
GENERATOR_VERSION = "1.0"
DEFAULT_LIBRISPEECH_ROOT = Path(
    "data/shared/LibriSpeech"
)
DEFAULT_NOISE_ROOT = Path("data/shared/noise/filter/v2")
DEFAULT_OUTPUT_DIR = Path("data/ssl")


@dataclass(frozen=True)
class AudioRecord:
    """已检查的音频清单行。"""

    uid: str
    sample_rate: int
    frames: int
    path: Path


def _positive_int(value: Any, *, field: str, location: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{location}: {field} 必须是精确正整数")
    return value


def _strict_json_object(text: str, *, location: str) -> dict[str, Any]:
    """解析 JSON object，并拒绝重复 key。"""

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{location}: JSON 含重复字段 {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicate_pairs)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location}: JSON 解析失败: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{location}: JSONL 每行必须是 object")
    return value


def _resolve_audio_path(raw_path: str, *, base_dir: Path, location: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path or raw_path != raw_path.strip():
        raise ValueError(f"{location}: path 必须是非空且无首尾空格的字符串")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"{location}: 音频路径不存在: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{location}: 音频路径不是普通文件: {resolved}")
    return resolved


def _probe_audio(
    path: Path,
    *,
    expected_sr: int | None,
    expected_frames: int | None,
    location: str,
) -> tuple[int, int]:
    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise ValueError(f"{location}: 无法读取音频头 {path}: {exc}") from exc
    sample_rate = _positive_int(int(info.samplerate), field="sample_rate", location=location)
    frames = _positive_int(int(info.frames), field="frames", location=location)
    if expected_sr is not None and sample_rate != expected_sr:
        raise ValueError(
            f"{location}: 采样率不匹配: metadata={expected_sr}, audio={sample_rate}, path={path}"
        )
    if expected_frames is not None and frames != expected_frames:
        raise ValueError(
            f"{location}: 帧数不匹配: metadata={expected_frames}, audio={frames}, path={path}"
        )
    return sample_rate, frames


def scan_librispeech_split(
    librispeech_root: str | Path,
    split: str,
    *,
    expected_sample_rate: int = 16000,
    limit: int | None = None,
) -> list[AudioRecord]:
    """递归扫描 LibriSpeech split，按真实音频头生成稳定记录。"""
    if not isinstance(split, str) or not split or split != split.strip():
        raise ValueError("LibriSpeech split 必须是非空字符串")
    _positive_int(expected_sample_rate, field="expected_sample_rate", location="参数")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("speech limit 必须是正整数或 None")

    root = Path(librispeech_root).expanduser().resolve(strict=True)
    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"LibriSpeech split 目录不存在: {split_root}")

    candidates = sorted(split_root.rglob("*.flac"))
    if not candidates:
        raise FileNotFoundError(f"LibriSpeech split 中没有 FLAC 文件: {split_root}")
    candidate_records: list[tuple[str, Path, str]] = []
    seen_uids: set[str] = set()
    seen_paths: set[Path] = set()
    for path in candidates:
        location = f"{split_root}:{path.relative_to(split_root)}"
        if path.is_symlink():
            raise ValueError(f"{location}: 不接受符号链接音频")
        resolved = path.resolve(strict=True)
        if resolved in seen_paths:
            raise ValueError(f"{location}: 重复音频路径: {resolved}")
        seen_paths.add(resolved)
        relative_stem = path.relative_to(split_root).with_suffix("").as_posix()
        uid = f"librispeech:{split}:{relative_stem}"
        if uid in seen_uids:
            raise ValueError(f"{location}: 重复 uid: {uid!r}")
        seen_uids.add(uid)
        candidate_records.append((uid, resolved, location))
    if limit is not None:
        candidate_records = candidate_records[:limit]
    records: list[AudioRecord] = []
    for uid, path, location in candidate_records:
        sample_rate, frames = _probe_audio(
            path,
            expected_sr=expected_sample_rate,
            expected_frames=None,
            location=location,
        )
        records.append(AudioRecord(uid, sample_rate, frames, path))
    return records


def read_speech_jsonl(
    manifest_path: str | Path,
    *,
    expected_sample_rate: int,
    limit: int | None = None,
) -> list[AudioRecord]:
    """从 Libri2Mix canonical speech JSONL 生成验证过的语音记录。"""
    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("speech limit 必须是正整数或 None")
    candidate_records: list[tuple[str, int, int, Path, str]] = []
    seen_uids: set[str] = set()
    seen_paths: set[Path] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{manifest}: 行 {line_number}: 空 JSONL 行")
            location = f"{manifest}: 行 {line_number}"
            item = _strict_json_object(raw, location=location)
            uid = item.get("uid")
            if not isinstance(uid, str) or not uid or uid != uid.strip():
                raise ValueError(f"{location}: uid 必须是非空字符串")
            if uid in seen_uids:
                raise ValueError(f"{location}: 重复 uid: {uid!r}")
            seen_uids.add(uid)
            sample_rate = _positive_int(item.get("fs"), field="fs", location=location)
            frames = _positive_int(item.get("length"), field="length", location=location)
            if sample_rate != expected_sample_rate:
                raise ValueError(
                    f"{location}: 采样率应为 {expected_sample_rate}，实际 metadata={sample_rate}"
                )
            path = _resolve_audio_path(
                item.get("path"), base_dir=manifest.parent, location=location
            )
            if path in seen_paths:
                raise ValueError(f"{location}: 重复音频路径: {path}")
            seen_paths.add(path)
            candidate_records.append((uid, sample_rate, frames, path, location))
    if not candidate_records:
        raise ValueError(f"{manifest}: 未找到可用语音记录")
    if limit is not None:
        candidate_records = candidate_records[:limit]
    records: list[AudioRecord] = []
    for uid, sample_rate, frames, path, location in candidate_records:
        actual_sr, actual_frames = _probe_audio(
            path,
            expected_sr=sample_rate,
            expected_frames=frames,
            location=location,
        )
        records.append(AudioRecord(uid, actual_sr, actual_frames, path))
    return records


def _parse_noise_record(line: str, *, manifest: Path, line_number: int) -> AudioRecord:
    location = f"{manifest}: 行 {line_number}"
    fields = parse_scp_fields(line, manifest, line_number, allowed_columns=(4,))
    uid, sr_text, frames_text, raw_path = fields
    if not uid or uid != uid.strip():
        raise ValueError(f"{location}: uid 必须是非空且无首尾空格的字符串")
    try:
        sample_rate = int(sr_text)
        frames = int(frames_text)
    except ValueError as exc:
        raise ValueError(f"{location}: fs 和 length 必须是整数") from exc
    _positive_int(sample_rate, field="fs", location=location)
    _positive_int(frames, field="length", location=location)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    # 先词法规范化以便全量去重；路径存在性也在全量审计阶段检查。
    normalized = Path(os.path.abspath(os.path.normpath(path)))
    if not normalized.is_file():
        raise FileNotFoundError(f"{location}: 音频路径不存在或不是普通文件: {normalized}")
    return AudioRecord(uid, sample_rate, frames, normalized)


def read_noise_scp(
    manifest_path: str | Path,
    *,
    limit: int = 50000,
    seed: int = 42,
) -> tuple[list[AudioRecord], int]:
    """审计完整四列噪声 SCP，并对发布子集做确定性 reservoir 抽样。

    全量扫描会检查 SCP schema、UID 和规范化路径唯一性；limit 选中的音频再逐条
    检查文件存在、采样率和帧数。limit=0 表示使用全量。
    """
    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    if type(limit) is not int or limit < 0:
        raise ValueError("noise limit 必须是非负整数；0 表示全量")
    if type(seed) is not int:
        raise ValueError("seed 必须是精确整数")
    rng = random.Random(seed)
    reservoir: list[AudioRecord] = []
    seen_uids: set[str] = set()
    seen_paths: set[Path] = set()
    total = 0
    with manifest.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            record = _parse_noise_record(
                stripped, manifest=manifest, line_number=line_number
            )
            if record.uid in seen_uids:
                raise ValueError(f"{manifest}: 行 {line_number}: 重复 uid: {record.uid!r}")
            if record.path in seen_paths:
                raise ValueError(
                    f"{manifest}: 行 {line_number}: 重复音频路径: {record.path}"
                )
            seen_uids.add(record.uid)
            seen_paths.add(record.path)
            total += 1
            if limit == 0 or len(reservoir) < limit:
                reservoir.append(record)
            else:
                slot = rng.randrange(total)
                if slot < limit:
                    reservoir[slot] = record
    if not reservoir:
        raise ValueError(f"{manifest}: 噪声 SCP 为空")

    validated: list[AudioRecord] = []
    for record in sorted(reservoir, key=lambda item: item.uid):
        path = _resolve_audio_path(
            str(record.path), base_dir=manifest.parent, location=f"{manifest}:{record.uid}"
        )
        sample_rate, frames = _probe_audio(
            path,
            expected_sr=record.sample_rate,
            expected_frames=record.frames,
            location=f"{manifest}:{record.uid}",
        )
        validated.append(AudioRecord(record.uid, sample_rate, frames, path))
    return validated, total


def _manifest_text(
    records: Iterable[AudioRecord],
    *,
    output_path: Path,
    absolute_paths: bool,
) -> str:
    entries = []
    for record in records:
        path = str(record.path) if absolute_paths else serialize_manifest_relative_path(
            record.path, output_path
        )
        entries.append((record.uid, record.sample_rate, record.frames, path))
    return serialize_scp_entries(entries, fmt="scp", four_col=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_manifests(
    *,
    output_dir: str | Path,
    speech_train: Sequence[AudioRecord],
    speech_validation: Sequence[AudioRecord],
    noise_train: Sequence[AudioRecord],
    noise_validation: Sequence[AudioRecord],
    speech_test: Sequence[AudioRecord] | None = None,
    source_info: dict[str, Any],
    absolute_paths: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    """事务发布 train/validation SCP 与来源 provenance。"""
    out = Path(output_dir).expanduser().resolve()
    target_map = {
        "speech_train": out / "speech" / "train.scp",
        "speech_validation": out / "speech" / "validation.scp",
        "noise_train": out / "noise" / "train.scp",
        "noise_validation": out / "noise" / "validation.scp",
    }
    text_map = {
        "speech_train": _manifest_text(
            speech_train, output_path=target_map["speech_train"], absolute_paths=absolute_paths
        ),
        "speech_validation": _manifest_text(
            speech_validation,
            output_path=target_map["speech_validation"],
            absolute_paths=absolute_paths,
        ),
        "noise_train": _manifest_text(
            noise_train, output_path=target_map["noise_train"], absolute_paths=absolute_paths
        ),
        "noise_validation": _manifest_text(
            noise_validation,
            output_path=target_map["noise_validation"],
            absolute_paths=absolute_paths,
        ),
    }
    if speech_test is not None:
        target_map["speech_test"] = out / "speech" / "test.scp"
        text_map["speech_test"] = _manifest_text(
            speech_test,
            output_path=target_map["speech_test"],
            absolute_paths=absolute_paths,
        )
    provenance = {
        "schema_version": 1,
        "generator": {"id": GENERATOR_ID, "version": GENERATOR_VERSION},
        "source_info": source_info,
        "path_mode": "absolute" if absolute_paths else "relative_to_manifest",
        "manifests": {
            key: {
                "path": str(target_map[key].relative_to(out)),
                "records": len(text_map[key].splitlines()),
                "sha256": _sha256_text(text_map[key]),
            }
            for key in sorted(text_map)
        },
        "audio_copied": False,
    }
    provenance_text = json.dumps(
        provenance, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    outputs = [
        (target_map[key], lambda stream, value=text_map[key]: stream.write(value))
        for key in sorted(text_map)
    ]
    outputs.append(
        (
            out / "provenance.json",
            lambda stream, value=provenance_text: stream.write(value),
        )
    )
    write_text_files_transactionally(outputs, overwrite=overwrite)
    counts = {
        "speech_train": len(speech_train),
        "speech_validation": len(speech_validation),
        "noise_train": len(noise_train),
        "noise_validation": len(noise_validation),
    }
    if speech_test is not None:
        counts["speech_test"] = len(speech_test)
    return counts


def _repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _classify_noise_manifest(path: Path) -> tuple[str, str]:
    """根据显式清单位置记录噪声来源，避免把 Noise92 标成 HQ noise。"""

    normalized_parts = {part.lower() for part in path.resolve().parts}
    if "noise92" in normalized_parts or path.name.lower() == "noise92.scp":
        return "noise92_scp", "Noise92 local subset; publisher hash/license recorded separately"
    return "local_hq_noise_scp", "HQ noise source pool; not Noise92"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将本机 LibriSpeech 与 HQ noise SCP 整理为 dynamic_ssl 资源清单",
        allow_abbrev=False,
    )
    parser.add_argument("--librispeech-root", type=Path, default=DEFAULT_LIBRISPEECH_ROOT)
    parser.add_argument("--speech-source", choices=("librispeech", "libri2mix-jsonl"), default="librispeech")
    parser.add_argument("--speech-train-jsonl", type=Path)
    parser.add_argument("--speech-validation-jsonl", type=Path)
    parser.add_argument("--speech-test-jsonl", type=Path)
    parser.add_argument("--train-split", default="train-clean-100")
    parser.add_argument("--validation-split", default="dev-clean")
    parser.add_argument("--test-split", help="可选 LibriSpeech test split；传入后额外输出 speech/test.scp")
    parser.add_argument("--speech-limit", type=int, default=0, help="0=全量；正数=按排序后的前 N 条整理")
    parser.add_argument("--noise-train-scp", type=Path, default=DEFAULT_NOISE_ROOT / "train_noise_sources.scp")
    parser.add_argument("--noise-validation-scp", type=Path, default=DEFAULT_NOISE_ROOT / "val_noise_sources.scp")
    parser.add_argument("--noise-train-limit", type=int, default=50000, help="固定种子 reservoir 子集；0=全量")
    parser.add_argument("--noise-validation-limit", type=int, default=20000, help="固定种子 reservoir 子集；0=全量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--absolute-paths", action="store_true", help="清单写入绝对路径，默认使用相对路径")
    parser.add_argument("--overwrite", action="store_true", help="原子替换已存在的全部清单")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.speech_limit < 0:
        raise SystemExit("--speech-limit 必须为非负整数；0 表示全量")
    if args.speech_source == "librispeech":
        if args.speech_train_jsonl or args.speech_validation_jsonl or args.speech_test_jsonl:
            raise SystemExit("--speech-source=librispeech 时不能传 Libri2Mix JSONL")
        speech_train = scan_librispeech_split(
            _repo_path(args.librispeech_root),
            args.train_split,
            expected_sample_rate=args.sample_rate,
            limit=args.speech_limit or None,
        )
        speech_validation = scan_librispeech_split(
            _repo_path(args.librispeech_root),
            args.validation_split,
            expected_sample_rate=args.sample_rate,
            limit=args.speech_limit or None,
        )
        speech_test = (
            scan_librispeech_split(
                _repo_path(args.librispeech_root),
                args.test_split,
                expected_sample_rate=args.sample_rate,
                limit=args.speech_limit or None,
            )
            if args.test_split
            else None
        )
        speech_source_info = {
            "type": "LibriSpeech_directory_scan",
            "root": str(_repo_path(args.librispeech_root).resolve()),
            "train_split": args.train_split,
            "validation_split": args.validation_split,
            "test_split": args.test_split,
            "scan_limit_per_split": args.speech_limit or None,
        }
    else:
        if not args.speech_train_jsonl or not args.speech_validation_jsonl:
            raise SystemExit(
                "--speech-source=libri2mix-jsonl 需要同时提供 --speech-train-jsonl 和 --speech-validation-jsonl"
            )
        train_jsonl = _repo_path(args.speech_train_jsonl).resolve(strict=True)
        validation_jsonl = _repo_path(args.speech_validation_jsonl).resolve(strict=True)
        speech_train = read_speech_jsonl(
            train_jsonl, expected_sample_rate=args.sample_rate, limit=args.speech_limit or None
        )
        speech_validation = read_speech_jsonl(
            validation_jsonl,
            expected_sample_rate=args.sample_rate,
            limit=args.speech_limit or None,
        )
        if bool(args.test_split) != bool(args.speech_test_jsonl):
            raise SystemExit(
                "Libri2Mix 可选 test 输出需要同时提供 --test-split（名称）和 --speech-test-jsonl"
            )
        speech_test = (
            read_speech_jsonl(
                _repo_path(args.speech_test_jsonl),
                expected_sample_rate=args.sample_rate,
                limit=args.speech_limit or None,
            )
            if args.speech_test_jsonl
            else None
        )
        speech_source_info = {
            "type": "Libri2Mix_speech_jsonl",
            "train_manifest": str(train_jsonl),
            "train_sha256": _file_sha256(train_jsonl),
            "validation_manifest": str(validation_jsonl),
            "validation_sha256": _file_sha256(validation_jsonl),
            "test_manifest": (
                str(_repo_path(args.speech_test_jsonl).resolve())
                if args.speech_test_jsonl
                else None
            ),
            "test_sha256": (
                _file_sha256(_repo_path(args.speech_test_jsonl).resolve())
                if args.speech_test_jsonl
                else None
            ),
            "scan_limit_per_split": args.speech_limit or None,
        }

    noise_train_path = _repo_path(args.noise_train_scp).resolve(strict=True)
    noise_validation_path = _repo_path(args.noise_validation_scp).resolve(strict=True)
    noise_train, noise_train_total = read_noise_scp(
        noise_train_path, limit=args.noise_train_limit, seed=args.seed
    )
    noise_validation, noise_validation_total = read_noise_scp(
        noise_validation_path,
        limit=args.noise_validation_limit,
        seed=args.seed + 1,
    )
    train_noise_type, train_noise_name = _classify_noise_manifest(noise_train_path)
    validation_noise_type, validation_noise_name = _classify_noise_manifest(noise_validation_path)
    noise_type = (
        train_noise_type
        if train_noise_type == validation_noise_type
        else "mixed_noise_scp"
    )
    noise_name = (
        train_noise_name
        if train_noise_name == validation_noise_name
        else f"train={train_noise_name}; validation={validation_noise_name}"
    )
    source_info = {
        "speech": speech_source_info,
        "noise": {
            "type": noise_type,
            "train_type": train_noise_type,
            "validation_type": validation_noise_type,
            "train_manifest": str(noise_train_path),
            "train_manifest_sha256": _file_sha256(noise_train_path),
            "train_source_records": noise_train_total,
            "train_selected_records": len(noise_train),
            "validation_manifest": str(noise_validation_path),
            "validation_manifest_sha256": _file_sha256(noise_validation_path),
            "validation_source_records": noise_validation_total,
            "validation_selected_records": len(noise_validation),
            "selection": "deterministic_reservoir",
            "seed": args.seed,
            "dataset_name": noise_name,
        },
    }
    counts = prepare_manifests(
        output_dir=_repo_path(args.output_dir),
        speech_train=speech_train,
        speech_validation=speech_validation,
        noise_train=noise_train,
        noise_validation=noise_validation,
        speech_test=speech_test,
        source_info=source_info,
        absolute_paths=args.absolute_paths,
        overwrite=args.overwrite,
    )
    print(json.dumps({"output_dir": str(_repo_path(args.output_dir)), "records": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
