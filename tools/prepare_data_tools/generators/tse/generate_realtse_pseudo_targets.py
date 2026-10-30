#!/usr/bin/env python3
"""REAL-T-dev teacher 伪标签生成器（offline_tse 数据准备）。

REAL-T-dev（data/tse/REAL-TSE/REAL-T-dev）只有真实 mixture + enrollment，
没有 clean target（REAL-T 论文明确真实混合无参考）。本脚本用冻结的
teacher 模型（默认官方 baseline spk_emb_100）逐 trial 推理生成伪标签
target，再产出 **官方 REAL-TSE 挑战训练格式**（wesep-real-tse schema）清单：

  <manifest-dir>/{train,val}/raw.list + samples.jsonl + cues.yaml + cues/audio.json

正式产物经 GenerationManifest v2、生产 ``TSEOfflineDataset`` full reload gate、
owned directory transaction 与 publication receipt 发布。teacher 输出目录仅作为
可续跑 cache；可训练数据以 ``--manifest-dir`` 指向的完整 publication root 为准。

- ``samples.jsonl`` 每行：``{"key", "spk", "mix": {"default": [mixture]},
  "src": {speaker: [pseudo_target]}}``；所有路径均相对 generation root；
- ``cues.yaml`` + ``cues/audio.json``：speaker → enrollment 候选列表
  （同 split 内该说话人的全部 enrolment 音频），train 用 policy=random，
  val 用 policy=fixed（取列表首条可复现）。

trial 来源：``DEV/*_meta.csv``（5 个源数据集，mixture_utterance /
enrolment_speakers_utterance / speaker / source 列）。uid 取
``{mixture_utterance}--{enrolment_speakers_utterance}``（同一 mixture 配多条
enrollment 的 multi-enrollment trial 各占一行）。

train/val 划分按 session（mixture 命名 ``{session}_mixture_{start}_{end}``），
避免同 session 泄漏：每个源数据集内按 session 字典序每 ``--val-session-mod``
个取 1 个作 val（session 数 < 5 的源不参与划分，全部进 train）。

幂等：已存在的 target wav 跳过推理，可中断续跑；``--no-teacher`` 可从
已生成的 wav 只重建清单。

⚠️ 合规提示：REAL-T v1.0.0 是 evaluation benchmark。DEV 禁止训练、微调和
数据增强，EVAL1/EVAL2 仅用于最终报告。本 CLI 默认拒绝对官方 REAL-T 发布
pseudo-target 数据；只有显式传入文档给出的精确 acknowledgement 才允许生成
内部合规审计所需的受限 publication，且该 acknowledgement 不改变数据许可。

Usage:
    python -m tools.prepare_data_tools.generators.tse.generate_realtse_pseudo_targets \
        --realt_root data/tse/REAL-TSE/REAL-T-dev \
        --teacher_ckpt data/tse/REAL-TSE/baseline/spk_emb_100/avg_model.pt \
        --teacher_config configs/tse/inference/tse_bsrnn_spk_realt_baseline_infer.yaml \
        --device cuda:0 [--max-trials 20]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import tempfile
import uuid
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import soundfile as sf

try:
    import fcntl
except ImportError:  # pragma: no cover - 项目部署目标为 Linux。
    fcntl = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

_log = logging.getLogger("generate_realtse_pseudo_targets")

# 2.0：恢复 schema v2 owned publication，并绑定 REAL-T 合规 acknowledgement。
GENERATOR_VERSION_HISTORY = ("1.0", "1.1", "2.0")
GENERATOR_VERSION = GENERATOR_VERSION_HISTORY[-1]
GENERATOR_ID = "tasks/tse/generate_realtse_pseudo_targets.py"
PUBLICATION_OWNER_ID = GENERATOR_ID
PUBLICATION_ROOT_ALIAS = "dataset"
REAL_T_ACKNOWLEDGEMENT = (
    "I_ACKNOWLEDGE_REAL_T_IS_EVALUATION_ONLY_AND_WILL_NOT_TRAIN_ON_THIS_RELEASE"
)

_META_GLOB = "*_meta.csv"
_PUBLICATION_CONTROL_FILES = frozenset(
    {
        ".offline-publication-owner.json",
        ".generation-root.json",
        "generation_manifest.json",
        "generation_gate_receipt.json",
    }
)


def _session_of(mixture_utterance: str) -> str:
    """``{session}_mixture_{start}_{end}`` → session。"""
    return mixture_utterance.split("_mixture_")[0]


def _read_trials(realt_root: Path) -> list[dict]:
    """读 DEV/*_meta.csv，返回 trial 列表（含 source/session/speaker/路径）。"""
    meta_dir = realt_root / "DEV"
    trials = []
    for csv_path in sorted(meta_dir.glob(_META_GLOB)):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mix = row["mixture_utterance"].strip()
                enr = row["enrolment_speakers_utterance"].strip()
                trials.append({
                    "uid": f"{mix}--{enr}",
                    "mixture": mix,
                    "enrollment": enr,
                    "speaker": row["speaker"].strip(),
                    "source": row["source"].strip(),
                    "session": _session_of(mix),
                    "mix_path": realt_root / "mixtures" / f"{mix}.wav",
                    "enr_path": realt_root / "enrolment_speakers" / f"{enr}.wav",
                })
    if not trials:
        raise FileNotFoundError(f"no trials found under {meta_dir}/{_META_GLOB}")
    uids = [trial["uid"] for trial in trials]
    if len(set(uids)) != len(uids):
        raise ValueError("REAL-T meta contains duplicate mix-enrollment trial identities")
    for uid in uids:
        if uid in {"", ".", ".."} or any(char in uid for char in ("/", "\\", "\x00")):
            raise ValueError(f"unsafe REAL-T trial uid: {uid!r}")
    return trials


def _official_realt_split(realt_root: Path) -> str | None:
    """识别官方 REAL-T split；EVAL split 永不允许进入 producer。"""
    for split in ("DEV", "EVAL1", "EVAL2"):
        if (realt_root / split).is_dir():
            return split
    return None


def _require_realt_acknowledgement(realt_root: Path, acknowledgement: str | None) -> None:
    split = _official_realt_split(realt_root)
    if split in {"EVAL1", "EVAL2"}:
        raise PermissionError(
            f"REAL-T {split} is final-report-only and cannot enter pseudo-target publication"
        )
    if split == "DEV" and acknowledgement != REAL_T_ACKNOWLEDGEMENT:
        raise PermissionError(
            "REAL-T DEV is evaluation-only; generation is disabled by default. "
            "For an explicitly reviewed internal audit, pass "
            f"--acknowledge-real-t-evaluation-only {REAL_T_ACKNOWLEDGEMENT!r}. "
            "This does not grant permission to train, fine-tune, or augment REAL-T."
        )


def _split_train_val(trials: list[dict], val_session_mod: int) -> tuple[list, list]:
    """按 session 划分 train/val：每源每 val_session_mod 个 session 取 1 个作 val。"""
    by_source: dict[str, set[str]] = {}
    for t in trials:
        by_source.setdefault(t["source"], set()).add(t["session"])
    val_sessions: set[tuple[str, str]] = set()
    for source, sessions in sorted(by_source.items()):
        ordered = sorted(sessions)
        if len(ordered) < 5:
            _log.warning(
                "source=%s 仅 %d 个 session，全部划入 train（不参与 val 划分）",
                source, len(ordered),
            )
            continue
        val_sessions |= {
            (source, session)
            for index, session in enumerate(ordered)
            if index % val_session_mod == val_session_mod - 1
        }
    train = [t for t in trials if (t["source"], t["session"]) not in val_sessions]
    val = [t for t in trials if (t["source"], t["session"]) in val_sessions]
    return train, val


def _write_wesep_manifest(trials: list[dict], target_dir: Path, out_dir: Path,
                          policy: str) -> None:
    """wesep 模式（官方 REAL-TSE 挑战格式）：raw.list + samples.jsonl + cues.yaml。

    对齐 wesep-real-tse（scan_librimix.py / prepare_data.sh schema）：

      - samples.jsonl 每行：``{"key": uid, "spk": [speaker],
        "mix": {"default": [mixture]}, "src": {speaker: [pseudo_target]}}``
      - raw.list：指向本目录 samples.jsonl
      - cues.yaml + cues/audio.json：speaker → enrollment 候选列表
        （收集该 split 内所有该说话人的 enrolment 音频），
        policy random（train）/ fixed（val，取列表首条可复现）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    spk2utt: dict[str, dict[str, str]] = defaultdict(dict)  # spk -> path -> utt_id
    n_ok = 0
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for t in sorted(trials, key=lambda x: x["uid"]):
            target_path = target_dir / f"{t['uid']}.wav"
            if not target_path.exists():
                _log.warning("target 缺失，跳过 uid=%s（先完整跑生成）", t["uid"])
                continue
            spk = t["speaker"]
            sample = {
                "key": t["uid"],
                "spk": [spk],
                "mix": {"default": [str(t["mix_path"].resolve())]},
                "src": {spk: [os.path.relpath(target_path, out_dir)]},
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            enr_path = str(t["enr_path"].resolve())
            if enr_path not in spk2utt[spk]:
                spk2utt[spk][enr_path] = enr_path.rsplit("/", 1)[-1]
            n_ok += 1
    (out_dir / "raw.list").write_text(
        "samples.jsonl\n", encoding="utf-8"
    )
    # cues：spk -> [{utt_id, path}]
    audio = {
        spk: [{"utt_id": uid, "path": p} for p, uid in sorted(u2i.items())]
        for spk, u2i in sorted(spk2utt.items())
    }
    cues_dir = out_dir / "cues"
    cues_dir.mkdir(parents=True, exist_ok=True)
    (cues_dir / "audio.json").write_text(
        json.dumps(audio, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (out_dir / "cues.yaml").write_text(
        "cues:\n"
        "  audio:\n"
        "    type: raw\n"
        "    guaranteed: true\n"
        "    scope: speaker\n"
        "    policy:\n"
        f"      type: {policy}\n"
        "      key: spk_id\n"
        "      resource: cues/audio.json\n",
        encoding="utf-8",
    )
    _log.info("wesep manifest (raw.list/samples.jsonl/cues.yaml) -> %s (%d trials)",
              out_dir, n_ok)


def _copy_verified(source: Path, destination: Path) -> None:
    """复制普通单链接文件，并复核源和目标内容身份。"""
    from tools.prepare_data_tools.generation_manifest import sha256_file

    before = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    after = sha256_file(source)
    copied = sha256_file(destination)
    if before != after or before != copied:
        raise RuntimeError(f"file changed while staging publication: {source}")


def _source_snapshot_paths(realt_root: Path) -> list[tuple[str, Path]]:
    sources = [
        (f"meta:{path.name}", path)
        for path in sorted((realt_root / "DEV").glob(_META_GLOB))
    ]
    release_readme = realt_root.parent / "README.md"
    if release_readme.is_file():
        sources.append(("release:README.md", release_readme))
    return sources


def _stage_source_snapshots(
    realt_root: Path, staged: Path
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """把生成依据固化到 publication root，并保留原始路径映射。"""
    from tools.prepare_data_tools.generation_manifest import sha256_file

    identities: list[dict[str, object]] = []
    origins: list[dict[str, str]] = []
    for index, (name, source) in enumerate(_source_snapshot_paths(realt_root)):
        relative = f"source_metadata/{index:02d}-{source.name}"
        destination = staged / relative
        _copy_verified(source, destination)
        identities.append({
            "name": name,
            "root_alias": PUBLICATION_ROOT_ALIAS,
            "path": relative,
            "size": destination.stat().st_size,
            "sha256": sha256_file(destination),
        })
        origins.append({"name": name, "path": str(source.resolve())})
    if not identities:
        raise RuntimeError("publication requires at least one source metadata identity")
    return identities, origins


def _write_source_audio_identity(trials: list[dict], staged: Path) -> Path:
    """绑定 publication 外部 mixture/enrollment 的实际内容身份。"""
    from tools.prepare_data_tools.generation_manifest import sha256_file

    roles_by_path: dict[Path, set[str]] = defaultdict(set)
    for trial in trials:
        roles_by_path[trial["mix_path"].resolve(strict=True)].add("mixture")
        roles_by_path[trial["enr_path"].resolve(strict=True)].add("enrollment")
    records = []
    for path in sorted(roles_by_path, key=str):
        records.append({
            "path": str(path),
            "roles": sorted(roles_by_path[path]),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    output = staged / "source_audio_identity.json"
    output.write_text(
        json.dumps(
            {"schema": "tse-source-audio-identity-v1", "files": records},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return output


def _publication_inventory(staged: Path) -> list[dict[str, object]]:
    """枚举 publication 内容；source snapshots 与控制文件单独声明。"""
    from tools.prepare_data_tools.generation_manifest import sha256_file

    inventory: list[dict[str, object]] = []
    for path in sorted(staged.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"publication inventory refuses symlink: {path}")
        if not path.is_file() or path.name in _PUBLICATION_CONTROL_FILES:
            continue
        relative = path.relative_to(staged).as_posix()
        if relative.startswith("source_metadata/"):
            continue
        inventory.append({
            "root_alias": PUBLICATION_ROOT_ALIAS,
            "path": relative,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "kind": "waveform" if path.suffix.lower() == ".wav" else "metadata",
        })
    return inventory


def _optional_file_identity(path: str | Path) -> dict[str, object] | None:
    from tools.prepare_data_tools.generation_manifest import sha256_file

    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.is_symlink():
        return None
    resolved = candidate.resolve(strict=True)
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _build_generation_manifest(
    staged: Path,
    *,
    requested: int,
    published: int,
    failed: int,
    dropped: int,
    source_identities: list[dict[str, object]],
    source_origins: list[dict[str, str]],
    args: argparse.Namespace,
):
    from tools.prepare_data_tools.generation_manifest import (
        ADAPTER_TSE,
        GenerationManifest,
    )

    manifest = GenerationManifest.from_dict({
        "schema_version": 2,
        "generation_id": f"tse-realt-pseudo-{uuid.uuid4().hex}",
        "status": "complete",
        "artifact_kind": "tse-wesep-offline-pseudo-targets",
        "task": "tse",
        "generator": {"id": GENERATOR_ID, "version": GENERATOR_VERSION},
        "root_bindings": [
            {"alias": PUBLICATION_ROOT_ALIAS, "role": "manifest"},
        ],
        "splits": ["train", "val"],
        "parameters": {
            "source_split": "DEV",
            "source_policy": "evaluation-only; explicit acknowledgement required",
            "acknowledgement": args.acknowledge_real_t_evaluation_only,
            "val_session_mod": args.val_session_mod,
            "allow_partial": bool(args.allow_partial),
            "teacher_mode": "cache-only" if args.no_teacher else "inference",
            "teacher_checkpoint": _optional_file_identity(args.teacher_ckpt),
            "teacher_config": _optional_file_identity(args.teacher_config),
            "source_origins": source_origins,
            "public_train_samples": "train/raw.list",
            "public_train_cues": "train/cues.yaml",
            "public_val_samples": "val/raw.list",
            "public_val_cues": "val/cues.yaml",
        },
        "source_identities": source_identities,
        "inventory": _publication_inventory(staged),
        "counts": {
            "requested": requested,
            "published": published,
            "failed": failed,
            "dropped": dropped,
        },
        "required_gates": [
            {
                "name": f"{split}-reload",
                "adapter": ADAPTER_TSE,
                "split": split,
                "inputs": {
                    "samples_list": {
                        "root_alias": PUBLICATION_ROOT_ALIAS,
                        "path": f"{split}/raw.list",
                    },
                    "cues_yaml": {
                        "root_alias": PUBLICATION_ROOT_ALIAS,
                        "path": f"{split}/cues.yaml",
                    },
                    "samples_jsonl": {
                        "root_alias": PUBLICATION_ROOT_ALIAS,
                        "path": f"{split}/samples.jsonl",
                    },
                    "cues_resource": {
                        "root_alias": PUBLICATION_ROOT_ALIAS,
                        "path": f"{split}/cues/audio.json",
                    },
                    "source_audio_identity": {
                        "root_alias": PUBLICATION_ROOT_ALIAS,
                        "path": "source_audio_identity.json",
                    },
                },
                "options": {
                    "strict_load": True,
                    "target_sr": 16000,
                    "enroll_policy": "fixed",
                    "seed": 0,
                },
            }
            for split in ("train", "val")
        ],
    })
    (staged / "generation_manifest.json").write_text(
        manifest.to_json(), encoding="utf-8"
    )
    return manifest


def _publication_paths(staged: Path, target: Path):
    from tools.prepare_data_tools.offline_publication import OwnedPublicationPaths

    return OwnedPublicationPaths(
        allowed_parent=target.parent,
        staging=staged,
        target=target,
        backup=target.parent / f".{target.name}.backup-{uuid.uuid4().hex}",
        owner_id=PUBLICATION_OWNER_ID,
    )


@contextmanager
def _publication_lock(target: Path):
    if fcntl is None:
        raise RuntimeError("REAL-T publication locking requires fcntl")
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.generate.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _publish_offline_dataset(
    train_trials: list[dict],
    val_trials: list[dict],
    *,
    cache_dir: Path,
    manifest_dir: Path,
    realt_root: Path,
    dispositions: list[dict[str, str]],
    requested: int,
    args: argparse.Namespace,
) -> None:
    """在 owned staging 中组装完整数据集，经 full reload 后原子发布。"""
    from tools.prepare_data_tools.generation_manifest import write_generation_root_marker
    from tools.prepare_data_tools.offline_publication import (
        MultiRootPublicationTarget,
        PublicationCommittedError,
        publish_owned_offline_dataset_roots,
        write_ownership_marker,
    )

    if not train_trials or not val_trials:
        raise RuntimeError(
            f"publication requires non-empty train/val splits: "
            f"train={len(train_trials)}, val={len(val_trials)}"
        )
    with _publication_lock(manifest_dir):
        with tempfile.TemporaryDirectory(
            dir=manifest_dir.parent,
            prefix=f".{manifest_dir.name}.owned-staging-",
        ) as staging_name:
            staged = Path(staging_name)
            write_ownership_marker(staged, owner_id=PUBLICATION_OWNER_ID)
            staged_targets = staged / "pseudo_targets"
            for trial in sorted([*train_trials, *val_trials], key=lambda item: item["uid"]):
                _copy_verified(
                    cache_dir / f"{trial['uid']}.wav",
                    staged_targets / f"{trial['uid']}.wav",
                )
            _write_wesep_manifest(train_trials, staged_targets, staged / "train", "random")
            _write_wesep_manifest(val_trials, staged_targets, staged / "val", "fixed")
            _write_source_audio_identity([*train_trials, *val_trials], staged)
            if dispositions:
                (staged / "excluded_trials.json").write_text(
                    json.dumps(dispositions, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            source_identities, source_origins = _stage_source_snapshots(
                realt_root, staged
            )
            failed = sum(item["disposition"] == "failed" for item in dispositions)
            dropped = sum(item["disposition"] == "dropped" for item in dispositions)
            manifest = _build_generation_manifest(
                staged,
                requested=requested,
                published=len(train_trials) + len(val_trials),
                failed=failed,
                dropped=dropped,
                source_identities=source_identities,
                source_origins=source_origins,
                args=args,
            )
            write_generation_root_marker(
                staged,
                manifest=manifest,
                root_alias=PUBLICATION_ROOT_ALIAS,
                owner_id=PUBLICATION_OWNER_ID,
            )
            try:
                publish_owned_offline_dataset_roots(
                    (
                        MultiRootPublicationTarget(
                            alias=PUBLICATION_ROOT_ALIAS,
                            paths=_publication_paths(staged, manifest_dir),
                        ),
                    ),
                    manifest_root_alias=PUBLICATION_ROOT_ALIAS,
                )
            except PublicationCommittedError as exc:
                _log.warning("publication committed with owned backup retained: %s", exc)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate teacher pseudo-targets for REAL-T-dev (offline_tse)"
    )
    p.add_argument("--realt_root", default="data/tse/REAL-TSE/REAL-T-dev")
    p.add_argument("--teacher_ckpt",
                   default="data/tse/REAL-TSE/baseline/spk_emb_100/avg_model.pt")
    p.add_argument("--teacher_config",
                   default="configs/tse/inference/tse_bsrnn_spk_realt_baseline_infer.yaml")
    p.add_argument("--output-dir", default=None,
                   help="teacher 伪标签续跑 cache（默认 <realt_root>/pseudo_targets）")
    p.add_argument("--manifest-dir", default=None,
                   help="正式 offline_tse publication root（默认 <realt_root>/offline_manifest）")
    p.add_argument("--no-teacher", action="store_true",
                   help="跳过 teacher 推理，只从已存在的 pseudo_targets 重建 manifest"
                        "（配合幂等续跑使用）")
    p.add_argument("--allow-partial", action="store_true",
                   help="显式允许丢弃 teacher 失败或 target 缺失 trial，并记录 reason")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-trials", type=int, default=0,
                   help="probe：只处理前 N 条 trial；不会发布 manifest")
    p.add_argument("--val-session-mod", type=int, default=10,
                   help="每源每 N 个 session 取 1 个作 val（默认 10 ≈ 10%%）")
    p.add_argument(
        "--acknowledge-real-t-evaluation-only",
        default=None,
        metavar="EXACT_TEXT",
        help=(
            "官方 REAL-T DEV 的高摩擦合规确认；必须精确等于 "
            f"{REAL_T_ACKNOWLEDGEMENT!r}。EVAL1/EVAL2 始终拒绝"
        ),
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.val_session_mod <= 0 or args.max_trials < 0:
        p.error("--val-session-mod 必须 > 0，--max-trials 必须 >= 0")
    realt_root = Path(args.realt_root)
    _require_realt_acknowledgement(
        realt_root, args.acknowledge_real_t_evaluation_only
    )
    target_dir = Path(args.output_dir) if args.output_dir else realt_root / "pseudo_targets"
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else realt_root / "offline_manifest"
    target_dir.mkdir(parents=True, exist_ok=True)

    trials = _read_trials(realt_root)
    _log.info("共 %d trials（GENERATOR_VERSION=%s）", len(trials), GENERATOR_VERSION)
    if args.max_trials > 0:
        trials = trials[: args.max_trials]
        _log.info("--max-trials=%d probe：只处理前 %d 条且不发布", args.max_trials, len(trials))

    missing = [t["uid"] for t in trials
               if not t["mix_path"].is_file() or not t["enr_path"].is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} trials 的 mixture/enrollment wav 缺失，示例: {missing[:5]}"
        )
    split_trials = _split_train_val(trials, args.val_session_mod)
    split_by_uid = {
        trial["uid"]: split
        for split, items in zip(("train", "val"), split_trials)
        for trial in items
    }
    pending = [t for t in trials if not (target_dir / f"{t['uid']}.wav").is_file()]
    dispositions: list[dict[str, str]] = []
    if pending and args.no_teacher and not args.allow_partial:
        raise FileNotFoundError(
            f"{len(pending)} trials 的 target 缺失；默认拒绝部分发布，示例: "
            f"{[t['uid'] for t in pending[:5]]}"
        )
    if pending and not args.no_teacher:
        from tasks.tse.inference import TSEExtractor
        from core.utils.inference_artifacts import atomic_write_audio

        extractor = TSEExtractor(
            args.teacher_ckpt,
            config_path=args.teacher_config,
            device=args.device,
            apply_vad=False,
            output_peak=0.9,
        )
        for trial in pending:
            try:
                estimate = extractor.extract_speech(
                    str(trial["mix_path"]), str(trial["enr_path"])
                )
                target_path = target_dir / f"{trial['uid']}.wav"
                atomic_write_audio(
                    target_path,
                    lambda staged_path, waveform=estimate: sf.write(
                        str(staged_path), waveform, extractor.sample_rate
                    ),
                )
            except Exception:
                if not args.allow_partial:
                    raise
                _log.error("teacher 推理失败，显式丢弃 uid=%s", trial["uid"], exc_info=True)
                dispositions.append({
                    "uid": trial["uid"],
                    "split": split_by_uid[trial["uid"]],
                    "disposition": "failed",
                    "reason": "teacher_inference_failed",
                })
    elif pending:
        dispositions.extend({
            "uid": trial["uid"],
            "split": split_by_uid[trial["uid"]],
            "disposition": "dropped",
            "reason": "missing_target:no_teacher",
        } for trial in pending)
    excluded_uids = {item["uid"] for item in dispositions}
    for trial in trials:
        target_path = target_dir / f"{trial['uid']}.wav"
        if trial["uid"] not in excluded_uids and not target_path.is_file():
            if not args.allow_partial:
                raise FileNotFoundError(f"teacher target missing after generation: {target_path}")
            dispositions.append({
                "uid": trial["uid"],
                "split": split_by_uid[trial["uid"]],
                "disposition": "failed",
                "reason": "target_missing_after_generation",
            })
            excluded_uids.add(trial["uid"])
    train_trials, val_trials = (
        [trial for trial in items if trial["uid"] not in excluded_uids]
        for items in split_trials
    )
    if args.max_trials > 0:
        return 0

    _publish_offline_dataset(
        train_trials,
        val_trials,
        cache_dir=target_dir,
        manifest_dir=manifest_dir,
        realt_root=realt_root,
        dispositions=dispositions,
        requested=len(trials),
        args=args,
    )
    _log.info(
        "published offline_tse dataset: %s (train=%d, val=%d, excluded=%d)",
        manifest_dir,
        len(train_trials),
        len(val_trials),
        len(dispositions),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
