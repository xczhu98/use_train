"""
promote_run_to_final 与 cleanup --purge-intermediate 回归测试。

覆盖（设计文档第 4 步 promote 语义）：
  - 音频按相对路径移入 final/audio，audio.path 改写，rejected 记录不进 final
  - 按 record_id 合并已有 final manifest（新记录覆盖同 id 旧记录）
  - 幂等重跑（dst 已存在 src 缺失视为已移动）
  - passed 记录音频缺失 / 路径越界 → 报错且该数据集 manifest 不落盘
  - d_stats.json 完成门槛与 --force
  - dry-run 不落任何变更
  - --purge-intermediate 的年龄保护
"""
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import pytest

from core.data.noise_admission import ADMISSION_POLICY_VERSION, load_canonical_voice_policy
from core.data.noise_admission_policy import (
    decision_to_admission_extra,
    evaluate_noise_admission,
    load_admission_policy,
    parse_c2_evidence,
)
from noise_filter_pipeline.core.manifest_io import (
    AudioRecord,
    config_hash,
    read_manifest,
    write_manifest,
)
from noise_filter_pipeline.core.pipeline_config import resolve_stage_d_config
from noise_filter_pipeline.scripts.promote_run_to_final import (
    _prune_empty_dirs,
    merge_dataset_manifest,
    promote_run,
)

from .module_patch_test_helpers import patched_os

pytestmark = pytest.mark.unit

_CANONICAL_POLICY_SHA = load_canonical_voice_policy()[1]


_POLICY = load_admission_policy()
_FAKE_ARTIFACTS = {
    "voice_policy_sha256": _CANONICAL_POLICY_SHA,
    "audiotag_label_dict_sha256": _POLICY.label_space_sha256,
    "audiotag_checkpoint_sha256": "2" * 64,
    "qwen_model_sha256": "4" * 64,
    "c1_prompt_sha256": "5" * 64,
}


def _write_audio(path: str, content: bytes = b"fake-flac") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    return path


def _top5_evidence(policy, hit_mid):
    from core.data.noise_admission_policy import (
        CLASSIFIER_LABEL_COUNT,
        EVIDENCE_CONTRACT_VERSION,
        SCORE_CONTRACT,
    )

    filler_mids = [
        mid
        for mid in policy.index_to_mid.values()
        if mid != hit_mid and policy.class_of(mid) == "noise"
    ][:4]
    mids = [hit_mid, *filler_mids]
    probabilities = [1.0, 0.8, 0.6, 0.4, 0.2]
    labels = {
        "audioset_top5": [
            {
                "index": policy.mid_to_index[mid],
                "mid": mid,
                "label": policy.label_of(mid),
                "prob": probability,
            }
            for mid, probability in zip(mids, probabilities)
        ],
    }
    c2_extra = {
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "score_contract": SCORE_CONTRACT,
        "class_count": CLASSIFIER_LABEL_COUNT,
        "label_space_sha256": policy.label_space_sha256,
    }
    return labels, c2_extra


def _passed_record(rec_id: str, dataset_id: str, audio_path: str) -> AudioRecord:
    """构造携带固定 top5 evidence、C3 决策和正式 D2 命名 provenance 的记录。"""
    policy = _POLICY
    rain_mid = next(mid for mid, label in policy.mid_to_label.items() if label == "Rain")
    labels, c2_extra = _top5_evidence(policy, rain_mid)
    rec = AudioRecord(
        record_id=rec_id,
        dataset_id=dataset_id,
        source_id="src",
        chunk_id="c0",
        audio={"path": audio_path, "sr": 16000, "duration": 1.0},
        labels=labels,
    )
    rec.mark_stage("C1", "passed", extra={"intelligibility": "noise"})
    rec.mark_stage("C2", "passed", extra=c2_extra)
    evidence = parse_c2_evidence(labels, policy, c2_extra)
    decision = evaluate_noise_admission(
        c1_intelligibility="noise",
        evidence=evidence,
        policy=policy,
    )
    extra = decision_to_admission_extra(
        decision,
        execution_mode={"c1": "production", "c2": "subprocess_daemon"},
        publishable=True,
        artifact_fingerprints=dict(_FAKE_ARTIFACTS),
    )
    rec.mark_stage("C3", "passed", extra=extra)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    ordinal = int(stem) if stem.isdigit() else 0
    source_path = f"/input/{rec.record_id}.wav"
    source_fingerprint = {
        "dev": 1,
        "ino": ordinal + 1,
        "mode": 33188,
        "size": 1,
        "mtime_ns": 1,
        "ctime_ns": 1,
    }
    rec.provenance.update(
        {
            "dataset_label_naming_version": 1,
            "dataset_label_ordinal": ordinal,
            "dataset_label_shard_size": 10000,
            "dataset_label_source_identity": {
                "version": 1,
                "source_id": rec.source_id,
                "chunk_id": rec.chunk_id,
                "split": rec.split,
                "audio": {
                    "path": source_path,
                    "sr": 16000,
                    "duration": 1.0,
                    "fingerprint": source_fingerprint,
                },
            },
        }
    )
    rec.mark_stage(
        "D2",
        "passed",
        extra={"input_path": source_path, "input_fingerprint": source_fingerprint},
    )
    return rec


def _write_stage_d_completion(
    stage_d: str,
    *,
    final_stage: str = "D2",
    stage_cfg: dict | None = None,
) -> None:
    """按当前测试产物重建 Stage D completion fact。"""
    manifest_prefix = final_stage.lower()
    manifest_files = []
    for path in sorted(
        (os.path.join(stage_d, final_stage, name) for name in os.listdir(os.path.join(stage_d, final_stage))),
        key=os.path.basename,
    ):
        name = os.path.basename(path)
        if not (name.startswith(f"{manifest_prefix}_manifest_") and name.endswith(".jsonl")):
            continue
        payload = open(path, "rb").read()
        manifest_files.append(
            {"name": name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        )
    stats_path = os.path.join(stage_d, "d_stats.json")
    stats_payload = open(stats_path, "rb").read()
    completion = {
        "schema": "noise_filter_stage_d_completion",
        "version": 1,
        "final_stage": final_stage,
        "manifest_prefix": manifest_prefix,
        "stage_d_config_hash": config_hash(
            resolve_stage_d_config(
                stage_cfg
                if stage_cfg is not None
                else {
                    "skip_d1": True,
                    "d1": {},
                    "d2": {"dataset_label_shard_size": 10000},
                }
            )
        ),
        "d2_passed": max(1, sum(
            record.is_stage_passed("D2")
            for item in manifest_files
            for record in read_manifest(os.path.join(stage_d, final_stage, item["name"]))
        )),
        "manifest_files": manifest_files,
        "d_stats": {
            "size": len(stats_payload),
            "sha256": hashlib.sha256(stats_payload).hexdigest(),
        },
    }
    with open(os.path.join(stage_d, "d_completion.json"), "w", encoding="utf-8") as output:
        json.dump(completion, output)


def _make_run(tmpdir: str, dataset_id: str = "ds1", n: int = 3) -> str:
    """构造一次运行的 stage_d 目录（正式 dataset_label 布局），返回 stage_d 路径。"""
    stage_d = os.path.join(tmpdir, "run", "stage_d")
    audio_root = os.path.join(stage_d, "D2", "audio")
    recs = []
    for i in range(n):
        rel = os.path.join(
            dataset_id, "Rain", f"shard_{i // 10000:05d}", f"{i % 10000:04d}.flac"
        )
        path = _write_audio(os.path.join(audio_root, rel))
        recs.append(_passed_record(f"{dataset_id}__src__{i}", dataset_id, path))
    # 一条 rejected 审计记录（未落盘，无当前路径）不得进 final manifest
    rejected = AudioRecord(
        record_id=f"{dataset_id}__src__bad",
        dataset_id=dataset_id,
        source_id="src",
        chunk_id="bad",
        audio={"path": "", "sr": 16000, "duration": 0.1},
        labels={"audioset": "Rain"},
    )
    rejected.mark_stage("D2", "rejected")
    recs.append(rejected)
    os.makedirs(os.path.join(stage_d, "D2"), exist_ok=True)
    write_manifest(iter(recs), os.path.join(stage_d, "D2", f"d2_manifest_{dataset_id}.jsonl"))
    # stage 正常收尾的证据
    with open(os.path.join(stage_d, "d_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"stage": "D"}, f)
    _write_stage_d_completion(stage_d)
    return stage_d


def _move_run_output_to_merged_d1(stage_d: str) -> str:
    """把测试 run 改造成 D1 内合并 D2 的真实产物布局。"""
    d2_dir = os.path.join(stage_d, "D2")
    d1_dir = os.path.join(stage_d, "D1")
    manifest_path = os.path.join(d2_dir, "d2_manifest_ds1.jsonl")
    records = list(read_manifest(manifest_path))
    old_audio_root = os.path.join(d2_dir, "audio")
    new_audio_root = os.path.join(d1_dir, "audio")
    for record in records:
        audio_path = record.audio.get("path")
        if isinstance(audio_path, str) and audio_path:
            record.audio["path"] = os.path.join(
                new_audio_root,
                os.path.relpath(audio_path, old_audio_root),
            )
    os.rename(d2_dir, d1_dir)
    os.remove(os.path.join(d1_dir, "d2_manifest_ds1.jsonl"))
    write_manifest(iter(records), os.path.join(d1_dir, "d1_manifest_ds1.jsonl"))
    _write_stage_d_completion(
        stage_d,
        final_stage="D1",
        stage_cfg={
            "skip_d1": False,
            "d1": {"post_normalize": True},
            "d2": {"dataset_label_shard_size": 10000, "output_format": "flac"},
        },
    )
    return d1_dir


def _write_stage_d_config(
    path, *, skip_d1: bool, post_normalize: bool, output_format: str = "flac"
) -> str:
    path.write_text(
        "stage_d:\n"
        f"  skip_d1: {str(skip_d1).lower()}\n"
        "  d1:\n"
        f"    post_normalize: {str(post_normalize).lower()}\n"
        "  d2:\n"
        "    dataset_label_shard_size: 10000\n"
        f"    output_format: {output_format}\n",
        encoding="utf-8",
    )
    return str(path)


@pytest.mark.subprocess
def test_promote_cli_help_bootstraps_repo_imports_without_pythonpath(tmp_path):
    script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "promote_run_to_final.py")
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--stage-d-dir" in result.stdout


def test_promote_merged_d1_output_from_pipeline_config(tmp_path):
    stage_d = _make_run(str(tmp_path), n=1)
    d1_dir = _move_run_output_to_merged_d1(stage_d)
    config_path = _write_stage_d_config(
        tmp_path / "pipeline.yaml", skip_d1=False, post_normalize=True
    )
    final_root = tmp_path / "final"

    assert promote_run(stage_d, str(final_root), config_path=config_path) == 0

    assert not any((tmp_path / "run" / "stage_d" / "D1" / "audio").rglob("*.flac"))
    assert (final_root / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac").is_file()
    recs = list(read_manifest(str(final_root / "manifests" / "ds1.jsonl")))
    assert len(recs) == 1
    assert recs[0].is_stage_passed("D2")
    assert os.path.isdir(d1_dir)


def test_promote_merged_d1_still_rejects_legacy_c4(tmp_path):
    stage_d = _make_run(str(tmp_path), n=1)
    d1_dir = _move_run_output_to_merged_d1(stage_d)
    manifest_path = os.path.join(d1_dir, "d1_manifest_ds1.jsonl")
    records = list(read_manifest(manifest_path))
    records[0].stages["C4"] = records[0].stages.pop("C3")
    write_manifest(iter(records), manifest_path)
    config_path = _write_stage_d_config(
        tmp_path / "pipeline.yaml", skip_d1=False, post_normalize=True
    )

    assert promote_run(stage_d, str(tmp_path / "final"), config_path=config_path) == 1
    assert records[0].audio["path"] and os.path.isfile(records[0].audio["path"])
    assert not (tmp_path / "final").exists()


def test_promote_moves_audio_and_rewrites_paths(tmp_path):
    stage_d = _make_run(str(tmp_path))
    final_root = str(tmp_path / "final")

    assert promote_run(stage_d, final_root) == 0

    final_audio = tmp_path / "final" / "audio"
    # 3 个音频按相对路径移入 final/audio
    moved = sorted(p for p in final_audio.rglob("*.flac"))
    assert len(moved) == 3
    assert (final_audio / "ds1" / "Rain" / "shard_00000" / "0000.flac").is_file()
    # run 侧音频已不存在，空目录被清理
    assert not any((tmp_path / "run" / "stage_d" / "D2" / "audio").rglob("*.flac"))

    # final manifest：3 条 passed，路径已改写；rejected 不在其中
    recs = list(read_manifest(str(tmp_path / "final" / "manifests" / "ds1.jsonl")))
    assert len(recs) == 3
    assert all(r.is_stage_passed("D2") for r in recs)
    for r in recs:
        assert r.audio["path"].startswith(str(final_audio))
        assert os.path.isfile(r.audio["path"])

    # library_stats 已聚合
    stats = json.loads((tmp_path / "final" / "library_stats.json").read_text())
    assert stats["total_chunks"] == 3
    assert stats["dataset_count"] == 1
    assert stats["by_label"]["Rain"]["chunks"] == 3


def test_promote_rejects_speech_candidate_before_move_or_final_write(tmp_path):
    """D2 passed 不能绕过 C1 speech 拒绝；失败时源与已有 final 必须保持不变。"""
    stage_d = _make_run(str(tmp_path), n=1)
    manifest_path = os.path.join(stage_d, "D2", "d2_manifest_ds1.jsonl")
    records = list(read_manifest(manifest_path))
    records[0].stages["C1"]["intelligibility"] = "speech"
    write_manifest(iter(records), manifest_path)

    final_root = tmp_path / "final"
    final_manifest = final_root / "manifests" / "ds1.jsonl"
    existing_path = final_root / "audio" / "ds1" / "Rain" / "shard_00000" / "0003.flac"
    _write_audio(str(existing_path))
    existing = _passed_record("ds1__existing__0", "ds1", str(existing_path))
    write_manifest(iter([existing]), str(final_manifest))
    original_final = final_manifest.read_text(encoding="utf-8")

    assert promote_run(stage_d, str(final_root)) == 1

    src = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    assert src.is_file()
    assert final_manifest.read_text(encoding="utf-8") == original_final
    assert existing_path.is_file()
    assert not (final_root / "library_stats.json").exists()


def test_promote_dry_run_rejects_non_publishable_stage_c_before_move(tmp_path):
    """promote --dry-run 也必须拒绝 Stage C dry-run 产物，而非只检查路径。"""
    stage_d = _make_run(str(tmp_path), n=1)
    manifest_path = os.path.join(stage_d, "D2", "d2_manifest_ds1.jsonl")
    records = list(read_manifest(manifest_path))
    records[0].stages["C3"]["execution_mode"] = {
        "c1": "dry_run",
        "c2": "subprocess_daemon",
    }
    records[0].stages["C3"]["publishable"] = False
    write_manifest(iter(records), manifest_path)

    final_root = tmp_path / "final"
    assert promote_run(stage_d, str(final_root), dry_run=True) == 1

    src = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    assert src.is_file()
    assert not final_root.exists()


def test_promote_rejects_cross_record_artifact_mismatch_before_move(tmp_path):
    """同一 manifest 的 C3 artifact 指纹不一致时不得移动音频。"""
    stage_d = _make_run(str(tmp_path), n=2)
    manifest_path = os.path.join(stage_d, "D2", "d2_manifest_ds1.jsonl")
    records = list(read_manifest(manifest_path))
    records[0].stages["C3"]["artifact_fingerprints"]["qwen_model_sha256"] = "f" * 64
    write_manifest(iter(records), manifest_path)

    final_root = tmp_path / "final"
    assert promote_run(stage_d, str(final_root)) == 1

    src = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    assert src.is_file()
    assert not final_root.exists()


@pytest.mark.parametrize("legacy_c4", [False, True], ids=["no-c3", "c4-only"])
def test_promote_rejects_missing_new_c3_before_move(tmp_path, legacy_c4):
    stage_d = _make_run(str(tmp_path), n=1)
    manifest_path = os.path.join(stage_d, "D2", "d2_manifest_ds1.jsonl")
    records = list(read_manifest(manifest_path))
    c3 = records[0].stages.pop("C3")
    if legacy_c4:
        records[0].stages["C4"] = c3
    write_manifest(iter(records), manifest_path)

    final_root = tmp_path / "final"
    assert promote_run(stage_d, str(final_root)) == 1

    src = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    assert src.is_file()
    assert not final_root.exists()


def test_promote_rejects_missing_execution_contract_before_move(tmp_path):
    """缺少 execution contract 的陈旧 D2 产物不得进入共享 final。"""
    stage_d = _make_run(str(tmp_path), n=1)
    manifest_path = os.path.join(stage_d, "D2", "d2_manifest_ds1.jsonl")
    records = list(read_manifest(manifest_path))
    records[0].stages["C3"].pop("execution_mode")
    records[0].stages["C3"].pop("publishable")
    write_manifest(iter(records), manifest_path)

    final_root = tmp_path / "final"
    assert promote_run(stage_d, str(final_root)) == 1

    src = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    assert src.is_file()
    assert not final_root.exists()


def test_promote_merges_existing_final_manifest(tmp_path):
    final_root = tmp_path / "final"
    final_root.mkdir()
    # 已有 final manifest：同 id 旧记录 + 一条其他记录
    old_path = _write_audio(
        str(final_root / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac")
    )
    keep_path = _write_audio(
        str(final_root / "audio" / "ds1" / "Rain" / "shard_00000" / "0003.flac")
    )
    old_same_id = _passed_record("ds1__src__0", "ds1", old_path)
    old_same_id.audio["duration"] = 9.9  # 应被新记录覆盖
    old_other = _passed_record("ds1__src__keep", "ds1", keep_path)
    write_manifest(
        iter([old_same_id, old_other]), str(final_root / "manifests" / "ds1.jsonl")
    )

    stage_d = _make_run(str(tmp_path))
    assert promote_run(stage_d, str(final_root)) == 0

    recs = {r.record_id: r for r in read_manifest(str(final_root / "manifests" / "ds1.jsonl"))}
    # 3 条新记录 + 1 条保留旧记录；同 id 旧记录被覆盖（duration 更新）
    assert len(recs) == 4
    assert recs["ds1__src__keep"].audio["path"] == keep_path
    assert recs["ds1__src__0"].audio["duration"] == 1.0


def test_promote_rejects_record_id_collision_across_run_and_final_datasets(tmp_path):
    stage_d = _make_run(str(tmp_path), dataset_id="ds1", n=1)
    final_root = tmp_path / "final"
    existing_audio = _write_audio(str(final_root / "audio" / "existing.flac"))
    conflicting = _passed_record("ds1__src__0", "ds2", existing_audio)
    write_manifest(iter([conflicting]), str(final_root / "manifests" / "ds2.jsonl"))

    assert promote_run(stage_d, str(final_root)) == 1
    source = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    assert source.is_file()
    assert not (final_root / "library_stats.json").exists()


def test_promote_rejects_different_record_id_for_existing_final_audio_target(tmp_path):
    stage_d = _make_run(str(tmp_path), dataset_id="ds1", n=1)
    final_root = tmp_path / "final"
    source = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    target = final_root / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    _write_audio(str(target), b"existing-final-audio")
    existing = _passed_record("ds1__old__0", "ds1", str(target))
    final_manifest = final_root / "manifests" / "ds1.jsonl"
    write_manifest(iter([existing]), str(final_manifest))
    original_manifest = final_manifest.read_bytes()
    original_audio = target.read_bytes()

    records = list(read_manifest(str(tmp_path / "run" / "stage_d" / "D2" / "d2_manifest_ds1.jsonl")))
    records[0].record_id = "ds1__new__0"
    write_manifest(iter(records), str(tmp_path / "run" / "stage_d" / "D2" / "d2_manifest_ds1.jsonl"))
    _write_stage_d_completion(stage_d)

    assert promote_run(stage_d, str(final_root)) == 1
    assert target.read_bytes() == original_audio
    assert final_manifest.read_bytes() == original_manifest
    assert source.is_file()
    assert not (tmp_path / "run" / "stage_d" / ".promote_transaction.json").exists()


@pytest.mark.parametrize("target_kind", ["file", "symlink"], ids=["file", "symlink"])
@pytest.mark.parametrize("existing_manifest", [False, True], ids=["no-manifest", "manifest"])
def test_promote_rejects_unowned_existing_final_target(
    tmp_path, target_kind, existing_manifest
):
    """final/audio 中未被任何 manifest 声明的目标必须 fail-closed。"""
    stage_d = _make_run(str(tmp_path), n=1)
    final_root = tmp_path / "final"
    target = final_root / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    orphan_content = b"unowned-final-audio"

    if target_kind == "file":
        _write_audio(str(target), orphan_content)
    else:
        external = tmp_path / "external-orphan.flac"
        _write_audio(str(external), orphan_content)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(external)
        except (NotImplementedError, OSError) as exc:
            pytest.skip(f"symlink unsupported: {exc}")

    final_manifest = final_root / "manifests" / "ds1.jsonl"
    if existing_manifest:
        declared_audio = _write_audio(
            str(final_root / "audio" / "declared" / "keep.flac"), b"declared-final-audio"
        )
        write_manifest(
            iter([_passed_record("ds1__existing__0", "ds1", declared_audio)]),
            str(final_manifest),
        )
        original_manifest = final_manifest.read_bytes()
    else:
        original_manifest = None

    source = (
        tmp_path
        / "run"
        / "stage_d"
        / "D2"
        / "audio"
        / "ds1"
        / "Rain"
        / "shard_00000"
        / "0000.flac"
    )
    original_source = source.read_bytes()

    assert promote_run(stage_d, str(final_root)) == 1
    assert target.read_bytes() == orphan_content
    assert source.is_file()
    assert source.read_bytes() == original_source
    if existing_manifest:
        assert final_manifest.read_bytes() == original_manifest
    else:
        assert not final_manifest.exists()
    assert not (tmp_path / "run" / "stage_d" / ".promote_transaction.json").exists()


def test_promote_idempotent_rerun(tmp_path):
    stage_d = _make_run(str(tmp_path))
    final_root = str(tmp_path / "final")
    assert promote_run(stage_d, final_root) == 0
    # 重跑：dst 已存在、src 缺失视为已移动，不报错
    assert promote_run(stage_d, final_root) == 0
    recs = list(read_manifest(str(tmp_path / "final" / "manifests" / "ds1.jsonl")))
    assert len(recs) == 3


def test_promote_missing_audio_fails_without_writing_manifest(tmp_path):
    stage_d = _make_run(str(tmp_path))
    # 删掉一个 passed 记录引用的音频
    victim = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0001.flac"
    victim.unlink()
    final_root = tmp_path / "final"

    assert promote_run(stage_d, str(final_root)) == 1
    # 该数据集 manifest 不落盘，stats 不更新（重跑可自愈）
    assert not (final_root / "manifests" / "ds1.jsonl").exists()
    assert not (final_root / "library_stats.json").exists()


def test_promote_rejects_audio_path_outside_run_root(tmp_path):
    stage_d = _make_run(str(tmp_path), n=1)
    # 篡改 manifest：audio.path 指向 run 音频根目录之外
    manifest_path = os.path.join(stage_d, "D2", "d2_manifest_ds1.jsonl")
    recs = list(read_manifest(manifest_path))
    recs[0].audio["path"] = str(tmp_path / "elsewhere" / "x.flac")
    write_manifest(iter(recs), manifest_path)

    assert promote_run(stage_d, str(tmp_path / "final")) == 1
    assert not (tmp_path / "final" / "manifests" / "ds1.jsonl").exists()


@pytest.mark.parametrize("final_stage", ["D2", "D1"], ids=["independent-d2", "merged-d1"])
@pytest.mark.parametrize(
    "broken_contract",
    ["category", "shard", "slot", "provenance", "legacy-sr-path"],
)
def test_promote_rejects_nonformal_dataset_label_contract(
    tmp_path, final_stage, broken_contract
):
    """独立 D2 与 D1 合并产物均须满足正式 dataset_label 命名门禁。"""
    stage_d = _make_run(str(tmp_path), n=1)
    if final_stage == "D1":
        _move_run_output_to_merged_d1(stage_d)
    manifest_path = os.path.join(
        stage_d, final_stage, f"{final_stage.lower()}_manifest_ds1.jsonl"
    )
    records = list(read_manifest(manifest_path))
    record = records[0]
    if broken_contract == "category":
        record.stages["C3"]["balance_category"] = "Other"
    elif broken_contract == "shard":
        record.audio["path"] = record.audio["path"].replace(
            "shard_00000", "shard_00001"
        )
    elif broken_contract == "slot":
        record.audio["path"] = record.audio["path"].replace("0000.flac", "0001.flac")
    elif broken_contract == "provenance":
        record.provenance["dataset_label_ordinal"] = 1
    else:
        record.audio["path"] = record.audio["path"].replace(
            os.path.join("ds1", "Rain"), os.path.join("sr_16k", "Rain", "ds1")
        )
    write_manifest(iter(records), manifest_path)
    _write_stage_d_completion(stage_d, final_stage=final_stage)

    assert promote_run(stage_d, str(tmp_path / "final")) == 1
    assert not (tmp_path / "final").exists()


def test_promote_rejects_output_format_path_mismatch_before_any_move(tmp_path):
    stage_d = _make_run(str(tmp_path), n=1)
    _write_stage_d_completion(
        stage_d,
        stage_cfg={
            "skip_d1": True,
            "d1": {},
            "d2": {"dataset_label_shard_size": 10000, "output_format": "ogg"},
        },
    )
    config_path = _write_stage_d_config(
        tmp_path / "pipeline.yaml",
        skip_d1=True,
        post_normalize=False,
        output_format="ogg",
    )

    assert promote_run(stage_d, str(tmp_path / "final"), config_path=config_path) == 1
    assert (tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac").is_file()
    assert not (tmp_path / "final").exists()


@pytest.mark.parametrize("output_format", ["fl_ac", "fl-ac"])
def test_promote_rejects_non_alphanumeric_output_format_before_any_move(
    tmp_path, output_format
):
    stage_d = _make_run(str(tmp_path), n=1)
    stage_cfg = {
        "skip_d1": True,
        "d1": {},
        "d2": {"dataset_label_shard_size": 10000, "output_format": output_format},
    }
    _write_stage_d_completion(stage_d, stage_cfg=stage_cfg)
    config_path = _write_stage_d_config(
        tmp_path / "pipeline.yaml",
        skip_d1=True,
        post_normalize=False,
        output_format=output_format,
    )

    assert promote_run(stage_d, str(tmp_path / "final"), config_path=config_path) == 1
    assert (tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac").is_file()
    assert not (tmp_path / "final").exists()


def test_promote_rejects_existing_final_legacy_sr_path_before_move(tmp_path):
    stage_d = _make_run(str(tmp_path), n=1)
    final_root = tmp_path / "final"
    legacy_path = final_root / "audio" / "sr_16k" / "Rain" / "ds1" / "00" / "legacy.flac"
    _write_audio(str(legacy_path))
    write_manifest(
        iter([_passed_record("ds1__legacy__0", "ds1", str(legacy_path))]),
        str(final_root / "manifests" / "ds1.jsonl"),
    )

    assert promote_run(stage_d, str(final_root)) == 1
    assert (tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac").is_file()
    assert not (final_root / "library_stats.json").exists()


def test_promote_rejects_existing_final_sanitize_collision_before_move(tmp_path):
    stage_d = _make_run(str(tmp_path), n=1)
    final_root = tmp_path / "final"
    first_path = final_root / "audio" / "ds_1" / "Rain" / "shard_00000" / "0000.flac"
    second_path = final_root / "audio" / "ds_1" / "Rain" / "shard_00000" / "0001.flac"
    _write_audio(str(first_path))
    _write_audio(str(second_path))
    first = _passed_record("ds.1__old__0", "ds.1", str(first_path))
    second = _passed_record("ds_1__old__1", "ds_1", str(second_path))
    write_manifest(iter([first]), str(final_root / "manifests" / "ds.1.jsonl"))
    write_manifest(iter([second]), str(final_root / "manifests" / "ds_1.jsonl"))

    assert promote_run(stage_d, str(final_root)) == 1
    assert (tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac").is_file()
    assert not (final_root / "library_stats.json").exists()


def test_promote_rejects_source_symlink_ancestor_before_any_move(tmp_path):
    stage_d = _make_run(str(tmp_path), n=1)
    audio_root = tmp_path / "run" / "stage_d" / "D2" / "audio"
    source_parent = audio_root / "ds1"
    actual_parent = audio_root / "ds1_real"
    source_parent.rename(actual_parent)
    source_parent.symlink_to(actual_parent, target_is_directory=True)
    source = source_parent / "Rain" / "shard_00000" / "0000.flac"

    final_root = tmp_path / "final"
    assert promote_run(stage_d, str(final_root)) == 1
    assert source.is_file()
    assert not (final_root / "manifests").exists()
    assert not (tmp_path / "run" / "stage_d" / ".promote_complete.json").exists()


def test_promote_rejects_destination_symlink_ancestor_before_any_move(tmp_path):
    stage_d = _make_run(str(tmp_path), n=1)
    source = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    final_root = tmp_path / "final"
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    final_root.mkdir()
    (final_root / "audio").symlink_to(escaped, target_is_directory=True)

    assert promote_run(stage_d, str(final_root)) == 1
    assert source.is_file()
    assert not list(escaped.rglob("*.flac"))
    assert not (final_root / "manifests").exists()


def test_promote_rejects_zero_candidates_without_publication(tmp_path):
    stage_d = _make_run(str(tmp_path), n=0)
    os.makedirs(os.path.join(stage_d, "D2", "audio"), exist_ok=True)
    final_root = tmp_path / "final"

    assert promote_run(stage_d, str(final_root)) == 1
    assert not (final_root / "library_stats.json").exists()
    assert not (tmp_path / "run" / "stage_d" / ".promote_complete.json").exists()


def test_promote_rejects_audio_list_before_any_move(tmp_path):
    """audio 非 dict 必须在移动前全量预检失败，而不是后续 `.get` 崩溃。"""
    stage_d = _make_run(str(tmp_path), n=1)
    manifest_path = os.path.join(stage_d, "D2", "d2_manifest_ds1.jsonl")
    raw = json.loads(open(manifest_path, encoding="utf-8").readline())
    raw["audio"] = []
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(raw) + "\n")

    final_root = tmp_path / "final"
    assert promote_run(stage_d, str(final_root)) == 1

    src = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    assert src.is_file()
    assert not final_root.exists()


def test_promote_corrupt_d2_manifest_fails_before_any_move(tmp_path):
    """任一 D2 manifest 损坏时必须全量预检失败，其他数据集也不得先移动。"""
    stage_d = _make_run(str(tmp_path), dataset_id="ds1", n=1)
    d2_dir = os.path.join(stage_d, "D2")
    ds2_audio = _write_audio(
        os.path.join(d2_dir, "audio", "ds2", "Rain", "shard_00000", "0000.flac")
    )
    ds2_manifest = os.path.join(d2_dir, "d2_manifest_ds2.jsonl")
    with open(ds2_manifest, "w", encoding="utf-8") as f:
        f.write(json.dumps(_passed_record("ds2__src__0", "ds2", ds2_audio).to_dict()) + "\n")
        f.write("{broken json}\n")

    final_root = tmp_path / "final"
    assert promote_run(stage_d, str(final_root)) == 1

    ds1_src = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    assert ds1_src.is_file()
    assert os.path.isfile(ds2_audio)
    assert not final_root.exists()
    assert not (tmp_path / "run" / "stage_d" / ".promote_complete.json").exists()


def test_promote_corrupt_existing_final_fails_without_move_or_overwrite(tmp_path):
    """已有 final manifest 损坏时不得移动源、覆盖 manifest 或写 marker。"""
    stage_d = _make_run(str(tmp_path), n=1)
    final_root = tmp_path / "final"
    final_manifest = final_root / "manifests" / "ds1.jsonl"
    final_manifest.parent.mkdir(parents=True)
    original = "{broken existing final}\n"
    final_manifest.write_text(original, encoding="utf-8")

    assert promote_run(stage_d, str(final_root)) == 1

    src = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    assert src.is_file()
    assert final_manifest.read_text(encoding="utf-8") == original
    assert not (final_root / "library_stats.json").exists()
    assert not (tmp_path / "run" / "stage_d" / ".promote_complete.json").exists()


def test_promote_invalid_existing_final_contract_fails_without_move_or_overwrite(tmp_path):
    """参与合并的已有 final 也必须通过完整准入契约。"""
    stage_d = _make_run(str(tmp_path), n=1)
    final_root = tmp_path / "final"
    final_manifest = final_root / "manifests" / "ds1.jsonl"
    stale = _passed_record("ds1__old__0", "ds1", "/old/stale.flac")
    stale.stages["C3"].pop("artifact_fingerprints")
    write_manifest(iter([stale]), str(final_manifest))
    original = final_manifest.read_text(encoding="utf-8")

    assert promote_run(stage_d, str(final_root)) == 1

    src = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    assert src.is_file()
    assert final_manifest.read_text(encoding="utf-8") == original
    assert not (final_root / "audio").exists()
    assert not (final_root / "library_stats.json").exists()


def test_promote_requires_completion_bound_d_stats(tmp_path):
    stage_d = _make_run(str(tmp_path))
    os.remove(os.path.join(stage_d, "d_stats.json"))
    final_root = str(tmp_path / "final")

    # d_stats 是 completion fact 的绑定对象；缺失时 --force 也不得绕过。
    assert promote_run(stage_d, final_root) == 1
    assert promote_run(stage_d, final_root, force=True) == 1
    assert not (tmp_path / "final").exists()


def test_promote_dry_run_changes_nothing(tmp_path):
    stage_d = _make_run(str(tmp_path))
    final_root = tmp_path / "final"
    src_before = sorted(
        str(p) for p in (tmp_path / "run" / "stage_d" / "D2" / "audio").rglob("*.flac")
    )

    assert promote_run(stage_d, str(final_root), dry_run=True) == 0
    # 音频未移动、final 未创建
    assert sorted(
        str(p) for p in (tmp_path / "run" / "stage_d" / "D2" / "audio").rglob("*.flac")
    ) == src_before
    assert not final_root.exists()


def test_merge_dataset_manifest_appends_new_ids(tmp_path):
    path = str(tmp_path / "manifests" / "ds.jsonl")
    audio_root = tmp_path / "audio"
    first_path = _write_audio(str(audio_root / "a.flac"))
    replacement_path = _write_audio(str(audio_root / "a2.flac"))
    second_path = _write_audio(str(audio_root / "b.flac"))
    n = merge_dataset_manifest(
        path, [_passed_record("ds__s__a", "ds", first_path)], dry_run=False
    )
    assert n == 1
    n = merge_dataset_manifest(
        path,
        [
            _passed_record("ds__s__a", "ds", replacement_path),
            _passed_record("ds__s__b", "ds", second_path),
        ],
        dry_run=False,
    )
    assert n == 2
    recs = {r.record_id: r for r in read_manifest(path)}
    assert recs["ds__s__a"].audio["path"] == replacement_path  # 同 id 被覆盖


def test_prune_empty_dirs_keeps_non_empty(tmp_path):
    root = tmp_path / "audio"
    (root / "a" / "b").mkdir(parents=True)
    (root / "c").mkdir()
    (root / "c" / "keep.txt").write_text("x")
    assert _prune_empty_dirs(str(root)) == 2  # b 与 a
    assert (root / "c" / "keep.txt").is_file()
    assert root.is_dir()  # root 自身不删


# ---------------------------------------------------------------------------
# cleanup --purge-intermediate
# ---------------------------------------------------------------------------
def _make_old_tree(root, fname="x.flac", age_hours=72):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, fname)
    with open(path, "wb") as f:
        f.write(b"fake")
    old = time.time() - age_hours * 3600
    os.utime(path, (old, old))
    os.utime(root, (old, old))
    return root


def _stub_valid_cleanup_marker(
    monkeypatch, module, stage_d, final_root, *, final_stage="D2"
):
    """隔离 cleanup 测试：模拟 promoter 已完整验证并返回 marker v4。"""
    final_root = os.path.realpath(final_root)
    os.makedirs(final_root, exist_ok=True)
    with open(os.path.join(final_root, "library_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"total_chunks": 1}, f)
    marker = {
        "stage_d_dir": os.path.realpath(stage_d),
        "final_root": final_root,
        "final_stage": final_stage,
    }
    monkeypatch.setattr(
        module,
        "read_validated_promote_marker",
        lambda requested: (
            (True, "ok", dict(marker))
            if os.path.realpath(requested) == marker["stage_d_dir"]
            else (False, "wrong run", None)
        ),
    )
    return marker


def test_purge_intermediate_requires_current_run_promote_marker(tmp_path):
    """历史 final stats 不能授权删除当前 run 的任一音频树。"""
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    output_root = tmp_path / "run"
    stage_b_audio = output_root / "stage_b" / "audio"
    d2_audio = output_root / "stage_d" / "D2" / "audio"
    _make_old_tree(str(stage_b_audio))
    _make_old_tree(str(d2_audio))
    # 历史/全局统计存在，但当前 run 没有 promote marker。
    (output_root / "final").mkdir()
    (output_root / "final" / "library_stats.json").write_text("{}")

    assert module._find_intermediate_audio(str(output_root), min_age_hours=24) == []
    assert module._purge_intermediate_audio(
        str(output_root / "stage_d"), min_age_hours=24, dry_run=False
    ) == 0
    assert stage_b_audio.is_dir()
    assert d2_audio.is_dir()


def test_purge_intermediate_accepts_validated_marker_v4(tmp_path, monkeypatch):
    """完整 marker v4 在双锁内授权 Stage B/Stage D 两树一起清理。"""
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    stage_d = _make_run(str(tmp_path))
    output_root = tmp_path / "run"
    final_root = tmp_path / "external_final"
    d2_audio = output_root / "stage_d" / "D2" / "audio"
    stage_b_audio = output_root / "stage_b" / "audio"
    _make_old_tree(str(d2_audio), fname="leftover.flac")
    _make_old_tree(str(stage_b_audio))
    _stub_valid_cleanup_marker(monkeypatch, module, stage_d, final_root)

    assert module._purge_intermediate_audio(
        stage_d, min_age_hours=0, dry_run=False
    ) == 2
    assert not stage_b_audio.exists()
    assert not d2_audio.exists()
    assert not list((output_root / "stage_b").glob("*.quarantine"))
    assert not list((output_root / "stage_d" / "D2").glob("*.quarantine"))


def test_purge_intermediate_uses_merged_d1_marker(tmp_path, monkeypatch):
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    stage_d = _make_run(str(tmp_path), n=1)
    _move_run_output_to_merged_d1(stage_d)
    output_root = tmp_path / "run"
    d1_audio = output_root / "stage_d" / "D1" / "audio"
    stage_b_audio = output_root / "stage_b" / "audio"
    _make_old_tree(str(d1_audio), fname="leftover.flac")
    _make_old_tree(str(stage_b_audio))
    _stub_valid_cleanup_marker(
        monkeypatch, module, stage_d, tmp_path / "external_final", final_stage="D1"
    )

    found = module._find_intermediate_audio(str(output_root), min_age_hours=0)
    assert {str(path) for path in found} == {str(stage_b_audio), str(d1_audio)}


def test_purge_intermediate_dry_run_validates_but_never_renames(tmp_path, monkeypatch):
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    stage_d = _make_run(str(tmp_path), n=1)
    output_root = tmp_path / "run"
    stage_b_audio = output_root / "stage_b" / "audio"
    d2_audio = output_root / "stage_d" / "D2" / "audio"
    _make_old_tree(str(stage_b_audio))
    _stub_valid_cleanup_marker(monkeypatch, module, stage_d, tmp_path / "final")
    monkeypatch.setattr(
        module.os,
        "rename",
        lambda *_args: pytest.fail("dry-run 不得调用 rename"),
    )

    assert module._purge_intermediate_audio(
        stage_d, min_age_hours=0, dry_run=True
    ) == 2
    assert stage_b_audio.is_dir()
    assert d2_audio.is_dir()


def test_purge_intermediate_rejects_if_either_tree_is_too_young(tmp_path, monkeypatch):
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    stage_d = _make_run(str(tmp_path), n=1)
    output_root = tmp_path / "run"
    stage_b_audio = output_root / "stage_b" / "audio"
    d2_audio = output_root / "stage_d" / "D2" / "audio"
    _make_old_tree(str(stage_b_audio))
    (d2_audio / "active.flac").write_bytes(b"active")
    _stub_valid_cleanup_marker(monkeypatch, module, stage_d, tmp_path / "final")

    assert module._purge_intermediate_audio(
        stage_d, min_age_hours=24, dry_run=False
    ) == 0
    assert stage_b_audio.is_dir()
    assert d2_audio.is_dir()


def test_purge_intermediate_requires_both_audio_trees(tmp_path, monkeypatch):
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    stage_d = _make_run(str(tmp_path), n=1)
    d2_audio = tmp_path / "run" / "stage_d" / "D2" / "audio"
    _stub_valid_cleanup_marker(monkeypatch, module, stage_d, tmp_path / "final")

    assert module._purge_intermediate_audio(
        stage_d, min_age_hours=0, dry_run=False
    ) == 0
    assert d2_audio.is_dir()


def test_purge_intermediate_rejects_unfinished_promote_journal(tmp_path, monkeypatch):
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    stage_d = _make_run(str(tmp_path), n=1)
    output_root = tmp_path / "run"
    stage_b_audio = output_root / "stage_b" / "audio"
    d2_audio = output_root / "stage_d" / "D2" / "audio"
    _make_old_tree(str(stage_b_audio))
    _stub_valid_cleanup_marker(monkeypatch, module, stage_d, tmp_path / "final")
    (output_root / "stage_d" / module.PROMOTE_JOURNAL_NAME).write_text(
        json.dumps({"phase": "moving"}), encoding="utf-8"
    )

    assert module._purge_intermediate_audio(
        stage_d, min_age_hours=0, dry_run=False
    ) == 0
    assert stage_b_audio.is_dir()
    assert d2_audio.is_dir()


def test_purge_intermediate_rejects_symlink_inside_either_tree(tmp_path, monkeypatch):
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    stage_d = _make_run(str(tmp_path), n=1)
    output_root = tmp_path / "run"
    stage_b_audio = output_root / "stage_b" / "audio"
    d2_audio = output_root / "stage_d" / "D2" / "audio"
    _make_old_tree(str(stage_b_audio))
    external = tmp_path / "external.flac"
    external.write_bytes(b"keep")
    (stage_b_audio / "escape.flac").symlink_to(external)
    _stub_valid_cleanup_marker(monkeypatch, module, stage_d, tmp_path / "final")

    assert module._purge_intermediate_audio(
        stage_d, min_age_hours=0, dry_run=False
    ) == 0
    assert stage_b_audio.is_dir()
    assert d2_audio.is_dir()
    assert external.read_bytes() == b"keep"


def test_purge_intermediate_second_rename_failure_restores_first_tree(
    tmp_path, monkeypatch
):
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    stage_d = _make_run(str(tmp_path), n=1)
    output_root = tmp_path / "run"
    stage_b_audio = output_root / "stage_b" / "audio"
    d2_audio = output_root / "stage_d" / "D2" / "audio"
    _make_old_tree(str(stage_b_audio))
    _stub_valid_cleanup_marker(monkeypatch, module, stage_d, tmp_path / "final")
    real_rename = module.os.rename

    def fail_second_tree(source, destination):
        if os.path.abspath(source) == os.path.abspath(d2_audio):
            raise OSError("injected second rename failure")
        return real_rename(source, destination)

    monkeypatch.setattr(module.os, "rename", fail_second_tree)
    assert module._purge_intermediate_audio(
        stage_d, min_age_hours=0, dry_run=False
    ) == 0
    assert stage_b_audio.is_dir()
    assert d2_audio.is_dir()
    assert not list((output_root / "stage_b").glob("*.quarantine"))


def test_purge_intermediate_rejects_source_path_replacement_toctou(
    tmp_path, monkeypatch
):
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    stage_d = _make_run(str(tmp_path), n=1)
    output_root = tmp_path / "run"
    stage_b_audio = output_root / "stage_b" / "audio"
    d2_audio = output_root / "stage_d" / "D2" / "audio"
    _make_old_tree(str(stage_b_audio))
    _stub_valid_cleanup_marker(monkeypatch, module, stage_d, tmp_path / "final")
    real_rename = module.os.rename

    def replace_source_after_first_rename(source, destination):
        result = real_rename(source, destination)
        if os.path.abspath(source) == os.path.abspath(stage_b_audio):
            os.makedirs(source)
            with open(os.path.join(source, "replacement.flac"), "wb") as f:
                f.write(b"do-not-delete")
        return result

    monkeypatch.setattr(module.os, "rename", replace_source_after_first_rename)
    assert module._purge_intermediate_audio(
        stage_d, min_age_hours=0, dry_run=False
    ) == 0
    assert (stage_b_audio / "replacement.flac").read_bytes() == b"do-not-delete"
    assert d2_audio.is_dir()
    quarantines = list((output_root / "stage_b").glob(".*.quarantine"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "x.flac").is_file()


def test_purge_intermediate_rejects_symlinked_final_asset(tmp_path, monkeypatch):
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    stage_d = _make_run(str(tmp_path), n=1)
    output_root = tmp_path / "run"
    stage_b_audio = output_root / "stage_b" / "audio"
    d2_audio = output_root / "stage_d" / "D2" / "audio"
    _make_old_tree(str(stage_b_audio))
    final_root = tmp_path / "final"
    _stub_valid_cleanup_marker(monkeypatch, module, stage_d, final_root)
    stats_path = final_root / "library_stats.json"
    stats_path.unlink()
    target = tmp_path / "outside_stats.json"
    target.write_text("{}", encoding="utf-8")
    stats_path.symlink_to(target)

    assert module._purge_intermediate_audio(
        stage_d, min_age_hours=0, dry_run=False
    ) == 0
    assert stage_b_audio.is_dir()
    assert d2_audio.is_dir()


def test_purge_intermediate_skips_missing_dirs(tmp_path):
    from noise_filter_pipeline.scripts.cleanup_pipeline_temp import _find_intermediate_audio

    assert _find_intermediate_audio(str(tmp_path), min_age_hours=24) == []


def test_general_residual_scan_prunes_unpromoted_d2_audio(tmp_path, monkeypatch):
    """无有效 marker 时通用递归扫描不得进入 D2/audio。"""
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    output_root = tmp_path / "run"
    d2_audio = output_root / "stage_d" / "D2" / "audio"
    _make_old_tree(str(d2_audio), fname=".x.flac.dead.flac")
    real_walk = module.os.walk

    def guarded_walk(root, *args, **kwargs):
        for item in real_walk(root, *args, **kwargs):
            assert os.path.realpath(item[0]) != os.path.realpath(d2_audio)
            yield item

    monkeypatch.setattr(module.os, "walk", guarded_walk)
    found = module._find_residuals(
        roots=set(), min_age_hours=24, recursive_roots={str(output_root)},
        excluded_subtrees={str(d2_audio)},
    )
    assert found == []


# ---------------------------------------------------------------------------
# promote 内容一致性 / 防护
# ---------------------------------------------------------------------------
def test_move_file_atomically_replaces_conflicting_content(tmp_path):
    """dst 已存在且内容不一致：用新 run 内容原子替换，不再先删旧 dst。"""
    from noise_filter_pipeline.scripts.promote_run_to_final import _move_file

    src = tmp_path / "src.flac"
    dst = tmp_path / "dst.flac"
    src.write_bytes(b"a" * 300000)
    dst.write_bytes(b"a" * 150000 + b"B" + b"a" * 149999)

    error = _move_file(str(src), str(dst), dry_run=False)
    assert error is None  # 原子替换成功
    assert not src.exists()
    assert dst.read_bytes() == b"a" * 300000


def test_move_file_dry_run_conflict_changes_nothing(tmp_path):
    """dry-run 时内容冲突只校验，不替换、不删除任何文件。"""
    from noise_filter_pipeline.scripts.promote_run_to_final import _move_file

    src = tmp_path / "src.flac"
    dst = tmp_path / "dst.flac"
    src.write_bytes(b"new-audio")
    dst.write_bytes(b"old-audio")

    error = _move_file(str(src), str(dst), dry_run=True)
    assert error is None
    assert src.exists()
    assert dst.read_bytes() == b"old-audio"


def test_cross_device_conflict_replaces_with_verified_copy(tmp_path, monkeypatch):
    """跨盘内容冲突：copy+完整校验后 os.replace 原子替换旧 dst。"""
    import errno
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    src = tmp_path / "src.flac"
    dst = tmp_path / "out" / "dst.flac"
    src.write_bytes(b"new-audio")
    dst.parent.mkdir()
    dst.write_bytes(b"old-audio")
    real_rename = module.os.rename

    def cross_device_only_for_initial_move(source, target, *args, **kwargs):
        if os.fspath(source) == str(src) and os.fspath(target) != "source":
            raise OSError(errno.EXDEV, "cross-device")
        return real_rename(source, target, *args, **kwargs)

    monkeypatch.setattr(module.os, "rename", cross_device_only_for_initial_move)
    error = module._move_file(str(src), str(dst), dry_run=False)
    assert error is None
    assert not src.exists()
    assert dst.read_bytes() == b"new-audio"


def test_cross_device_copy_verifies_full_content_before_unlink(tmp_path, monkeypatch):
    """跨盘拷贝若中间内容损坏但尺寸相同，必须保留 src 并删除坏 dst。"""
    import errno
    import shutil
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    src = tmp_path / "src.flac"
    dst = tmp_path / "out" / "dst.flac"
    src.write_bytes(b"a" * 300000)

    monkeypatch.setattr(module.os, "rename", lambda *_: (_ for _ in ()).throw(OSError(errno.EXDEV, "cross-device")))

    def corrupt_copy(src_path, dst_path):
        shutil.copyfile(src_path, dst_path)
        with open(dst_path, "r+b") as f:
            f.seek(150000)
            f.write(b"B")

    monkeypatch.setattr(module.shutil, "copy2", corrupt_copy)
    error = module._move_file(str(src), str(dst), dry_run=False)
    assert error and "校验失败" in error
    assert src.exists()
    assert not dst.exists()


def test_promote_rejects_overlapping_audio_roots(tmp_path):
    """相等、final 在 run 内、run 在 final 内三种重叠都必须拒绝。"""
    stage_d = _make_run(str(tmp_path / "equal"))
    assert promote_run(stage_d, os.path.join(stage_d, "D2")) == 1

    nested_stage_d = _make_run(str(tmp_path / "nested_final"))
    nested_final_root = os.path.join(nested_stage_d, "D2", "audio", "nested")
    assert promote_run(nested_stage_d, nested_final_root) == 1

    ancestor_final = tmp_path / "ancestor" / "final"
    stage_d_inside_final = _make_run(str(ancestor_final / "audio" / "container"))
    assert promote_run(stage_d_inside_final, str(ancestor_final)) == 1


def test_promote_rejects_final_root_equal_to_current_run_root(tmp_path):
    """final_root 不得等于 stage_d 的父目录，即使两侧 audio roots 不重叠。"""
    stage_d = _make_run(str(tmp_path))
    run_root = os.path.dirname(stage_d)

    assert promote_run(stage_d, run_root) == 1

    src = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    assert src.is_file()
    assert not (tmp_path / "run" / "manifests").exists()


def test_repromote_atomically_replaces_conflicting_dst(tmp_path):
    """同一 run 重建新音频后重新 promote：新内容原子替换旧 dst，不再产生缺失窗口。"""
    stage_d = _make_run(str(tmp_path), n=1)
    final_root = tmp_path / "final"
    assert promote_run(stage_d, str(final_root)) == 0
    marker = tmp_path / "run" / "stage_d" / ".promote_complete.json"
    assert marker.is_file()

    src = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    dst = final_root / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    _write_audio(str(src), b"new-audio")

    assert promote_run(stage_d, str(final_root)) == 0
    assert not src.exists()
    assert dst.read_bytes() == b"new-audio"
    assert marker.is_file()


def test_cross_device_copy_exception_leaves_no_published_dst(tmp_path, monkeypatch):
    """copy2 中途异常只能留下 src，不得暴露半成品正式 dst。"""
    import errno
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    src = tmp_path / "src.flac"
    dst = tmp_path / "out" / "dst.flac"
    src.write_bytes(b"complete")
    monkeypatch.setattr(
        module.os, "rename",
        lambda *_: (_ for _ in ()).throw(OSError(errno.EXDEV, "cross-device")),
    )

    def partial_then_fail(_src, copy_dst):
        with open(copy_dst, "wb") as f:
            f.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(module.shutil, "copy2", partial_then_fail)
    error = module._move_file(str(src), str(dst), dry_run=False)
    assert error and "跨盘拷贝失败" in error
    assert src.exists()
    assert not dst.exists()
    assert not list((tmp_path / "out").glob(".dst.flac.*.tmp"))


def test_promote_marker_binds_each_manifest_fingerprint(tmp_path):
    """修改任一 D2 manifest 都必须使当前 run marker 失效。"""
    from noise_filter_pipeline.scripts.promote_run_to_final import validate_promote_marker

    stage_d = _make_run(str(tmp_path), dataset_id="ds1", n=1)
    d2_dir = os.path.join(stage_d, "D2")
    ds2_audio = _write_audio(
        os.path.join(d2_dir, "audio", "ds2", "Rain", "shard_00000", "0000.flac")
    )
    write_manifest(
        iter([_passed_record("ds2__src__0", "ds2", ds2_audio)]),
        os.path.join(d2_dir, "d2_manifest_ds2.jsonl"),
    )
    _write_stage_d_completion(stage_d)
    assert promote_run(stage_d, str(tmp_path / "final")) == 0
    assert validate_promote_marker(stage_d)[0]

    manifest = os.path.join(d2_dir, "d2_manifest_ds1.jsonl")
    old_stat = os.stat(manifest)
    with open(manifest, "r+", encoding="utf-8") as f:
        content = f.read()
        f.seek(0)
        f.write(content.replace('"source_id": "src"', '"source_id": "SRC"'))
        f.truncate()
    os.utime(manifest, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert not validate_promote_marker(stage_d)[0]


def test_promote_marker_binds_library_stats_fingerprint(tmp_path):
    from noise_filter_pipeline.scripts.promote_run_to_final import validate_promote_marker

    stage_d = _make_run(str(tmp_path), n=1)
    final_root = tmp_path / "final"
    assert promote_run(stage_d, str(final_root)) == 0
    assert validate_promote_marker(stage_d)[0]
    (final_root / "library_stats.json").write_text("{}\n", encoding="utf-8")

    valid, reason = validate_promote_marker(stage_d)
    assert not valid
    assert "library_stats.json 已变化" in reason


def test_promote_marker_invalidated_by_new_remaining_audio(tmp_path):
    """成功 promote 后新增 D2/audio 文件，即使 manifest/stats 不变也必须失效。"""
    from noise_filter_pipeline.scripts.promote_run_to_final import validate_promote_marker

    stage_d = _make_run(str(tmp_path), n=1)
    assert promote_run(stage_d, str(tmp_path / "final")) == 0
    assert validate_promote_marker(stage_d)[0]

    leftover = tmp_path / "run" / "stage_d" / "D2" / "audio" / "unexpected.flac"
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_bytes(b"new")

    valid, reason = validate_promote_marker(stage_d)
    assert not valid
    assert "audio" in reason


def test_promote_audio_tree_fingerprint_fails_closed_on_walk_error(tmp_path, monkeypatch):
    """D2/audio 遍历错误时 marker 校验必须失败。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    stage_d = _make_run(str(tmp_path), n=1)
    assert module.promote_run(stage_d, str(tmp_path / "final")) == 0
    real_walk = module.os.walk

    def failing_walk(path, *args, onerror=None, **kwargs):
        if os.path.realpath(path) == os.path.realpath(os.path.join(stage_d, "D2", "audio")):
            assert onerror is not None
            onerror(PermissionError("walk denied"))
            return iter(())
        return real_walk(path, *args, onerror=onerror, **kwargs)

    monkeypatch.setattr(module.os, "walk", failing_walk)
    valid, reason = module.validate_promote_marker(stage_d)
    assert not valid
    assert "audio" in reason


def test_promote_marker_rejects_invalid_contract_fields(tmp_path):
    """数量或完成时间字段畸形时不得授权 cleanup。"""
    from noise_filter_pipeline.scripts.promote_run_to_final import validate_promote_marker

    stage_d = _make_run(str(tmp_path), n=1)
    assert promote_run(stage_d, str(tmp_path / "final")) == 0
    marker_path = os.path.join(stage_d, ".promote_complete.json")
    with open(marker_path, "r", encoding="utf-8") as f:
        marker = json.load(f)

    marker["dataset_count"] = "1"
    marker["created_at"] = "not-a-time"
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(marker, f)
    valid, reason = validate_promote_marker(stage_d)
    assert not valid
    assert "字段" in reason


@pytest.mark.parametrize(
    "legacy_field",
    [
        "admission_calibration_version",
        "admission_calibration_sha256",
        "calibration_approved",
        "calibration_status",
        "approved",
        "approval_blockers",
        "review_required",
        "approval_bundle",
        "approval_bundle_sha256",
        "attestation",
        "attestation_sha256",
        "threshold_hits",
        "content_kind",
        "margin",
        "quarantine",
        "reason_codes",
        "risk",
        "max_risk",
        "sustain",
    ],
)
def test_promote_marker_rejects_every_legacy_gate_field(tmp_path, legacy_field):
    """marker v4 不得因版本号有效而接纳任何旧门禁字段。"""
    from noise_filter_pipeline.scripts.cleanup_pipeline_temp import _promoted_audio_target
    from noise_filter_pipeline.scripts.promote_run_to_final import validate_promote_marker

    stage_d = _make_run(str(tmp_path), n=1)
    assert promote_run(stage_d, str(tmp_path / "final")) == 0
    marker_path = os.path.join(stage_d, ".promote_complete.json")
    marker = json.loads(open(marker_path, encoding="utf-8").read())
    marker[legacy_field] = False
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(marker, f)

    valid, reason = validate_promote_marker(stage_d)
    assert not valid
    assert legacy_field in reason
    target, cleanup_reason = _promoted_audio_target(stage_d)
    assert target is None
    assert legacy_field in cleanup_reason


@pytest.mark.parametrize("legacy_version", [1, 2, 3])
def test_promote_marker_rejects_legacy_versions(tmp_path, legacy_version):
    """marker v4 是唯一有效发布标记，旧版本必须 fail-closed。"""
    from noise_filter_pipeline.scripts.promote_run_to_final import validate_promote_marker

    stage_d = _make_run(str(tmp_path), n=1)
    assert promote_run(stage_d, str(tmp_path / "final")) == 0
    marker_path = os.path.join(stage_d, ".promote_complete.json")
    marker = json.loads(open(marker_path, encoding="utf-8").read())
    assert {
        key: marker[key]
        for key in (
            "version",
            "manifest_schema_version",
            "admission_policy_version",
            "evidence_contract_version",
        )
    } == {
        "version": 4,
        "manifest_schema_version": 6,
        "admission_policy_version": 4,
        "evidence_contract_version": 5,
    }
    assert "admission_calibration_version" not in marker
    marker["version"] = legacy_version
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(marker, f)

    valid, reason = validate_promote_marker(stage_d)
    assert not valid
    assert "版本不受支持" in reason


def test_move_file_read_error_preserves_existing_dst(tmp_path, monkeypatch):
    """比较读取失败不是内容不同，不得删除已发布 final 文件。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    src = tmp_path / "src.flac"
    dst = tmp_path / "dst.flac"
    src.write_bytes(b"new")
    dst.write_bytes(b"old")
    real_open = open

    def fail_src_open(path, *args, **kwargs):
        if os.fspath(path) == str(src) and args and args[0] == "rb":
            raise PermissionError("temporarily unreadable")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fail_src_open)
    error = module._move_file(str(src), str(dst), dry_run=False)
    assert error and "读取失败" in error
    assert src.exists()
    assert dst.read_bytes() == b"old"


def test_file_fingerprint_read_error_is_not_absent(tmp_path, monkeypatch):
    """存在但不可读的 manifest 必须报错，不能伪装成稳定 absent 指纹。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    path = tmp_path / "manifest.jsonl"
    path.write_text("{}\n")
    real_open = module.os.open

    def fail_open(target, *args, **kwargs):
        if os.fspath(target) == str(path):
            raise PermissionError("temporarily unreadable")
        return real_open(target, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", fail_open)
    with pytest.raises(OSError):
        module._file_fingerprint(str(path))


def test_file_fingerprint_rejects_symlink_leaf(tmp_path):
    """fingerprint 必须以 O_NOFOLLOW 拒绝 symlink leaf。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "marker.json"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink unsupported: {exc}")

    with pytest.raises(OSError):
        module._file_fingerprint(str(link))


def test_file_fingerprint_binds_visible_path_identity_after_read(tmp_path, monkeypatch):
    """读取中的 rename 替换必须因可见 path identity 不再绑定原 FD 而失败。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    path = tmp_path / "marker.json"
    replacement = tmp_path / "replacement.json"
    path.write_text('{"value": "old"}\n', encoding="utf-8")
    replacement.write_text('{"value": "new"}\n', encoding="utf-8")
    real_fstat = module.os.fstat
    calls = 0

    def replace_path_after_read(fd):
        nonlocal calls
        calls += 1
        info = real_fstat(fd)
        if calls == 2:
            os.replace(replacement, path)
        return info

    monkeypatch.setattr(module.os, "fstat", replace_path_after_read)
    with pytest.raises(ValueError, match="changed while fingerprinting"):
        module._file_fingerprint(str(path))


def test_llm_cache_interrupted_tmp_matches_cleanup_contract(tmp_path, monkeypatch):
    """LLM 原子写残留必须严格使用 <key>.json.<pid>.tmp 并被 cleanup 扫描。"""
    from noise_filter_pipeline.core import llm_cache as cache_module
    from noise_filter_pipeline.scripts.cleanup_pipeline_temp import _find_residuals

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    cache = cache_module.LLMResultCache(str(tmp_path / "cache"), model_name="model")
    key = cache._key(str(audio), "prompt", cache.model_name)
    observed = {}

    def fail_publish(source, destination, **kwargs):
        observed["source"] = source
        raise OSError("injected interrupted cache write")

    monkeypatch.setattr(cache_module.os, "rename", fail_publish)
    monkeypatch.setattr(cache_module.os, "unlink", lambda *args, **kwargs: None)
    cache.set(str(audio), "prompt", {"result": "noise"})

    expected_name = f"{key}.json.{os.getpid()}.tmp"
    assert observed["source"] == expected_name
    residual = cache.cache_dir / key[:2] / expected_name
    old = time.time() - 72 * 3600
    os.utime(residual, (old, old))
    assert residual in _find_residuals(
        set(), min_age_hours=24, recursive_roots={str(cache.cache_dir)}
    )


def test_residual_scan_skips_recursive_root_inside_excluded_tree(tmp_path):
    """cache_dir 自身落在 D2/audio 内时也必须整体跳过。"""
    from noise_filter_pipeline.scripts.cleanup_pipeline_temp import _find_residuals

    d2_audio = tmp_path / "stage_d" / "D2" / "audio"
    nested = d2_audio / "cache"
    _make_old_tree(str(nested), fname="x.json.1.tmp")
    found = _find_residuals(
        set(), 24, recursive_roots={str(nested)}, excluded_subtrees={str(d2_audio)}
    )
    assert found == []


def test_subtree_latest_mtime_fails_closed_on_child_stat_error(tmp_path, monkeypatch):
    """任一子项 stat 失败时目录年龄未知，必须返回 None。"""
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    root = tmp_path / "tree"
    child = root / "child.flac"
    _make_old_tree(str(root), fname=child.name)
    real_stat = module.os.stat

    def fail_child_stat(path, *args, **kwargs):
        if os.path.realpath(path) == os.path.realpath(child):
            raise PermissionError("denied")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "stat", fail_child_stat)
    assert module._subtree_latest_mtime(root) is None


def test_subtree_latest_mtime_fails_closed_on_walk_error(tmp_path, monkeypatch):
    """os.walk 遍历错误必须通过 onerror 使年龄判定失败。"""
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    root = tmp_path / "tree"
    _make_old_tree(str(root))

    def failing_walk(path, onerror=None, **kwargs):
        assert onerror is not None
        onerror(PermissionError("walk denied"))
        return iter(())

    monkeypatch.setattr(module.os, "walk", failing_walk)
    assert module._subtree_latest_mtime(root) is None


@pytest.mark.parametrize("relative", ["stage_b/audio", "stage_d/D2/audio"])
def test_remove_audio_revalidates_subtree_age_at_delete_boundary(tmp_path, monkeypatch, relative):
    """候选发现后子树出现新文件时，最终删除边界必须重新拒绝。"""
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    target = tmp_path / relative
    _make_old_tree(str(target))
    candidates = [target]
    (target / "new.flac").write_bytes(b"active")
    removed = []
    monkeypatch.setattr(
        module,
        "read_validated_promote_marker",
        lambda _stage_d: (True, "ok", {"final_stage": "D2"}),
    )
    monkeypatch.setattr(module.shutil, "rmtree", lambda path: removed.append(path))

    assert module._remove_residuals(
        candidates,
        dry_run=False,
        protected_d2_stage_d=str(tmp_path / "stage_d"),
        min_age_hours=24,
    ) == 0
    assert removed == []


def test_remove_file_revalidates_age_at_delete_boundary(tmp_path, monkeypatch):
    """旧文件成为候选后被 touch，unlink 前必须重新 stat 并拒绝删除。"""
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    target = tmp_path / "checkpoint.json.tmp"
    target.write_text("stale", encoding="utf-8")
    old = time.time() - 72 * 3600
    os.utime(target, (old, old))
    candidates = module._find_residuals(
        set(), min_age_hours=24, recursive_roots={str(tmp_path)}
    )
    assert candidates == [target]

    target.touch()
    unlinked = []
    monkeypatch.setattr(type(target), "unlink", lambda self: unlinked.append(self))

    assert module._remove_residuals(
        candidates, dry_run=False, min_age_hours=24
    ) == 0
    assert unlinked == []
    assert target.exists()


def test_remove_d2_audio_revalidates_marker_at_delete_boundary(tmp_path, monkeypatch):
    """发现候选后 marker 失效时，最终 rmtree 边界必须再次拒绝。"""
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as module

    stage_d = tmp_path / "stage_d"
    d2_audio = stage_d / "D2" / "audio"
    _make_old_tree(str(d2_audio))
    removed = []
    monkeypatch.setattr(
        module,
        "read_validated_promote_marker",
        lambda _stage_d: (False, "expired", None),
    )
    monkeypatch.setattr(module.shutil, "rmtree", lambda path: removed.append(path))

    assert module._remove_residuals(
        [d2_audio], dry_run=False, protected_d2_stage_d=str(stage_d)
    ) == 0
    assert removed == []


# ---------------------------------------------------------------------------
# 并发锁：同一 final_root 的并发 promote 必须互斥
# ---------------------------------------------------------------------------
def test_promote_lock_blocks_concurrent_promote(tmp_path):
    """锁被持有期间 promote 必须失败，释放后可成功。"""
    from noise_filter_pipeline.scripts.promote_run_to_final import (
        _acquire_promote_lock,
        _release_promote_lock,
    )

    stage_d = _make_run(str(tmp_path), n=1)
    final_root = tmp_path / "final"
    fd = _acquire_promote_lock(str(final_root))
    try:
        assert promote_run(stage_d, str(final_root)) == 1
    finally:
        _release_promote_lock(fd)
    assert promote_run(stage_d, str(final_root)) == 0


def test_promote_lock_scoped_to_final_root(tmp_path):
    """不同 final_root 的锁互不阻塞。"""
    from noise_filter_pipeline.scripts.promote_run_to_final import (
        _acquire_promote_lock,
        _release_promote_lock,
    )

    fd = _acquire_promote_lock(str(tmp_path / "final_a"))
    try:
        # 另一 final_root 的锁应可正常获取（不抛异常即通过）
        fd2 = _acquire_promote_lock(str(tmp_path / "final_b"))
        _release_promote_lock(fd2)
    finally:
        _release_promote_lock(fd)


def test_promote_lock_uses_realpath_identity_for_symlink_alias(tmp_path):
    """同一 final_root 的真实路径与 symlink alias 必须争用同一把锁。"""
    from noise_filter_pipeline.scripts.promote_run_to_final import (
        _acquire_promote_lock,
        _release_promote_lock,
    )

    final_root = tmp_path / "final"
    final_root.mkdir()
    alias = tmp_path / "final_alias"
    alias.symlink_to(final_root, target_is_directory=True)

    fd = _acquire_promote_lock(str(final_root))
    try:
        with pytest.raises(RuntimeError, match="promote 锁"):
            _acquire_promote_lock(str(alias))
    finally:
        _release_promote_lock(fd)


def test_promote_acquires_and_releases_lock_after_success(tmp_path):
    """promote 成功后锁被释放，后续 promote 不受影响。"""
    stage_d = _make_run(str(tmp_path), n=1)
    final_root = tmp_path / "final"
    assert promote_run(stage_d, str(final_root)) == 0
    # 幂等重跑应能再次拿到锁并成功
    assert promote_run(stage_d, str(final_root)) == 0


@pytest.mark.parametrize("lock_kind", ["symlink", "fifo"])
def test_lock_entrypoints_reject_unsafe_lock_files(tmp_path, monkeypatch, lock_kind):
    """run、promote、cleanup 锁遇到 symlink/特殊文件时必须 fail closed。"""
    from noise_filter_pipeline.core import run_lock as run_lock_module
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as cleanup_module
    from noise_filter_pipeline.scripts import promote_run_to_final as promote_module

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir(mode=0o700)
    lock_dir.chmod(0o700)
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(lock_dir))

    stage_d = tmp_path / "run" / "stage_d"
    stage_d.mkdir(parents=True)
    final_root = tmp_path / "final"
    output_root = tmp_path / "output"
    lock_paths = [
        run_lock_module.run_lock_path(str(stage_d)),
        promote_module._promote_lock_path(str(final_root)),
        run_lock_module.cleanup_lock_path(str(output_root)),
    ]

    for index, lock_path in enumerate(lock_paths):
        if lock_kind == "symlink":
            target = tmp_path / f"lock_target_{index}"
            target.write_text("not a lock", encoding="utf-8")
            os.symlink(target, lock_path)
            expected = "不得是符号链接"
        else:
            os.mkfifo(lock_path)
            expected = "必须是 regular file"

        with pytest.raises(RuntimeError, match=expected):
            if index == 0:
                run_lock_module.acquire_run_lock(str(stage_d))
            elif index == 1:
                promote_module._acquire_promote_lock(str(final_root))
            else:
                cleanup_module._acquire_lock(str(output_root))


def test_run_lock_uses_realpath_identity_for_symlink_alias(tmp_path):
    """同一 stage_d 的真实路径与 symlink alias 必须争用同一把 run 锁。"""
    from noise_filter_pipeline.core.run_lock import acquire_run_lock, release_run_lock

    stage_d = tmp_path / "run" / "stage_d"
    stage_d.mkdir(parents=True)
    alias = tmp_path / "stage_d_alias"
    alias.symlink_to(stage_d, target_is_directory=True)

    fd = acquire_run_lock(str(stage_d))
    try:
        with pytest.raises(RuntimeError, match="run 锁"):
            acquire_run_lock(str(alias))
    finally:
        release_run_lock(fd)


def test_stage_d_writer_cannot_replace_inputs_during_promote_preflight(tmp_path, monkeypatch):
    """promote 预检到发布期间，Stage D writer 不得进入输出写路径。"""
    from noise_filter_pipeline.core.pipeline_config import UnifiedPipelineConfig
    from noise_filter_pipeline.scripts import promote_run_to_final as promote_module
    from noise_filter_pipeline.stages.stage_d_enhancement import StageDOrchestrator

    stage_d = _make_run(str(tmp_path), n=1)
    final_root = tmp_path / "final"
    entered_preflight = threading.Event()
    release_preflight = threading.Event()
    real_preflight = promote_module._preflight_manifests

    def blocked_preflight(*args, **kwargs):
        entered_preflight.set()
        assert release_preflight.wait(timeout=5)
        return real_preflight(*args, **kwargs)

    monkeypatch.setattr(promote_module, "_preflight_manifests", blocked_preflight)
    result = []
    worker = threading.Thread(
        target=lambda: result.append(promote_module.promote_run(stage_d, str(final_root)))
    )
    worker.start()
    assert entered_preflight.wait(timeout=5)

    writer_entered = []
    orchestrator = StageDOrchestrator(
        UnifiedPipelineConfig(output_root=str(tmp_path / "run"), stage_d={"skip_d1": True})
    )
    monkeypatch.setattr(
        orchestrator,
        "_run_substage",
        lambda **_kwargs: writer_entered.append(True) or str(tmp_path / "unexpected"),
    )
    try:
        with pytest.raises(RuntimeError, match="run 锁"):
            orchestrator.run(
                input_manifest=os.path.join(stage_d, "D2", "d2_manifest_ds1.jsonl"),
                output_dir=stage_d,
                output_manifest=os.path.join(stage_d, "d_manifest.jsonl"),
            )
        assert writer_entered == []
    finally:
        release_preflight.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert result == [0]


def test_cleanup_marker_recheck_and_d2_delete_exclude_promote(tmp_path, monkeypatch):
    """cleanup 从 marker 复验到双树隔离期间必须排斥同 run promote。"""
    from noise_filter_pipeline.scripts import cleanup_pipeline_temp as cleanup_module
    from noise_filter_pipeline.scripts import promote_run_to_final as promote_module

    stage_d = _make_run(str(tmp_path), n=1)
    output_root = tmp_path / "run"
    d2_audio = output_root / "stage_d" / "D2" / "audio"
    stage_b_audio = output_root / "stage_b" / "audio"
    final_root = tmp_path / "final"
    _make_old_tree(str(d2_audio), fname="leftover.flac")
    _make_old_tree(str(stage_b_audio))
    _stub_valid_cleanup_marker(
        monkeypatch, cleanup_module, stage_d, final_root
    )

    entered_marker_check = threading.Event()
    release_marker_check = threading.Event()
    real_read_marker = cleanup_module.read_validated_promote_marker

    def blocked_valid_marker(locked_stage_d):
        entered_marker_check.set()
        assert release_marker_check.wait(timeout=5)
        return real_read_marker(locked_stage_d)

    monkeypatch.setattr(
        cleanup_module, "read_validated_promote_marker", blocked_valid_marker
    )
    cleanup_result = []
    worker = threading.Thread(
        target=lambda: cleanup_result.append(
            cleanup_module._purge_intermediate_audio(
                stage_d, min_age_hours=0, dry_run=False
            )
        )
    )
    worker.start()
    assert entered_marker_check.wait(timeout=5)
    try:
        assert promote_module.promote_run(stage_d, str(final_root)) == 1
        assert d2_audio.is_dir()
        assert stage_b_audio.is_dir()
    finally:
        release_marker_check.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert cleanup_result == [2]
    assert not d2_audio.exists()
    assert not stage_b_audio.exists()


# ---------------------------------------------------------------------------
# promote 可恢复事务与 marker v4 强绑定
# ---------------------------------------------------------------------------
def test_source_cleanup_preserves_path_replacement_after_fingerprint(
    tmp_path, monkeypatch
):
    """fingerprint 后 pathname 被替换时不得删除替换对象。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    source = tmp_path / "source.flac"
    source.write_bytes(b"expected")
    expected = module._file_fingerprint(str(source))
    displaced = tmp_path / "displaced.flac"
    replacement = b"replacement"
    real_open = module.os.open
    injected = False

    def replace_before_source_open(path, flags, *args, **kwargs):
        nonlocal injected
        if (
            path == source.name
            and kwargs.get("dir_fd") is not None
            and flags & os.O_NOFOLLOW
            and not injected
        ):
            injected = True
            source.rename(displaced)
            source.write_bytes(replacement)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", replace_before_source_open)

    with pytest.raises(RuntimeError, match="source 在删除前变化"):
        module._unlink_source_if_unchanged(
            str(source),
            expected,
            transaction_id="a" * 32,
        )

    assert source.read_bytes() == replacement
    assert displaced.read_bytes() == b"expected"
    assert not list(tmp_path.glob(".promote-source-*"))


def test_source_quarantine_restore_concurrent_real_link_race(tmp_path, monkeypatch):
    """两个恢复线程同时进入真实 os.link 时至多一个成功，且败者留证。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    source = tmp_path / "source.flac"
    quarantine_root = tmp_path / (".promote-source-" + "a" * 32 + "-000000")
    quarantine_root.mkdir(mode=0o700)
    (quarantine_root / "source").write_bytes(b"expected")
    expected = module._content_fingerprint(str(quarantine_root / "source"))
    journal_path = tmp_path / module.PROMOTE_JOURNAL_NAME
    journal_path.write_text("{}\n", encoding="utf-8")
    entered = threading.Barrier(2)
    results = []
    real_link = module.os.link

    def synchronized_real_link(*args, **kwargs):
        entered.wait(timeout=5)
        return real_link(*args, **kwargs)

    monkeypatch.setattr(module.os, "link", synchronized_real_link)

    def restore():
        try:
            module._restore_source_from_quarantine(
                str(quarantine_root), str(source), expected
            )
        except (OSError, RuntimeError, ValueError) as exc:
            results.append(exc)
        else:
            results.append(None)

    workers = [threading.Thread(target=restore) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
    assert all(not worker.is_alive() for worker in workers)
    assert sum(result is None for result in results) == 1
    assert sum(result is not None for result in results) == 1
    assert source.read_bytes() == b"expected"
    assert journal_path.exists()


@pytest.mark.parametrize(
    "link_error",
    [errno.EXDEV, errno.EPERM, errno.EACCES],
)
def test_source_quarantine_restore_link_errors_fail_closed(tmp_path, monkeypatch, link_error):
    """hard-link 失败时 source、quarantine 与 journal 均保留，后续 promote 被阻断。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    stage_d = _make_run(str(tmp_path), n=1)
    source = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    final_root = tmp_path / "final"
    quarantine_root = source.parent / (".promote-source-" + "a" * 32 + "-000000")
    quarantine_root.mkdir(mode=0o700)
    shutil.move(str(source), str(quarantine_root / "source"))
    expected_source = module._file_fingerprint(str(quarantine_root / "source"))
    expected_content = module._content_fingerprint(str(quarantine_root / "source"))
    journal_path = tmp_path / "run" / "stage_d" / module.PROMOTE_JOURNAL_NAME
    entry = {
        "source": str(source),
        "target": str(final_root / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"),
        "quarantine_root": str(quarantine_root),
        "source_fingerprint": expected_source,
        "new_content": expected_content,
        "old_content": {"exists": False, "size": 0, "sha256": ""},
        "staged": None,
        "backup": None,
        "source_restore_pending": False,
    }
    journal = {
        "version": module._JOURNAL_VERSION,
        "transaction_id": "a" * 32,
        "phase": "published",
        "stage_d_dir": os.path.realpath(stage_d),
        "final_root": os.path.realpath(final_root),
        "owned_roots": [],
        "audio": [entry],
        "targets": [entry],
    }
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    monkeypatch.setattr(
        module.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError(link_error, "link denied")),
    )

    with pytest.raises(RuntimeError, match="recovery failed"):
        module._recover_or_block(stage_d, str(final_root))
    assert not source.exists()
    assert quarantine_root.exists()
    assert (quarantine_root / "source").read_bytes() == b"fake-flac"
    assert journal_path.exists()
    assert module.promote_run(stage_d, str(final_root)) == 1
    assert journal_path.exists()


@pytest.mark.parametrize("operation", ["unlink", "rmdir"])
def test_source_quarantine_restore_cleanup_errors_fail_closed(tmp_path, monkeypatch, operation):
    """恢复后 unlink/rmdir 任一失败都必须保留 quarantine 证据与 journal。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    stage_d = _make_run(str(tmp_path), n=1)
    source = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    final_root = tmp_path / "final"
    quarantine_root = source.parent / (".promote-source-" + "a" * 32 + "-000000")
    quarantine_root.mkdir(mode=0o700)
    shutil.move(str(source), str(quarantine_root / "source"))
    expected_source = module._file_fingerprint(str(quarantine_root / "source"))
    expected_content = module._content_fingerprint(str(quarantine_root / "source"))
    journal_path = tmp_path / "run" / "stage_d" / module.PROMOTE_JOURNAL_NAME
    entry = {
        "source": str(source),
        "target": str(final_root / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"),
        "quarantine_root": str(quarantine_root),
        "source_fingerprint": expected_source,
        "new_content": expected_content,
        "old_content": {"exists": False, "size": 0, "sha256": ""},
        "staged": None,
        "backup": None,
        "source_restore_pending": False,
    }
    journal = {
        "version": module._JOURNAL_VERSION,
        "transaction_id": "a" * 32,
        "phase": "published",
        "stage_d_dir": os.path.realpath(stage_d),
        "final_root": os.path.realpath(final_root),
        "owned_roots": [],
        "audio": [entry],
        "targets": [entry],
    }
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    real_unlink = module.os.unlink
    real_rmdir = module.os.rmdir
    if operation == "unlink":
        monkeypatch.setattr(
            module.os,
            "unlink",
            lambda path, *args, **kwargs: (_ for _ in ()).throw(
                OSError(errno.EACCES, "unlink denied")
            ) if path == "source" else real_unlink(path, *args, **kwargs),
        )
    else:
        monkeypatch.setattr(
            module.os,
            "rmdir",
            lambda path, *args, **kwargs: (_ for _ in ()).throw(
                OSError(errno.EPERM, "rmdir denied")
            ) if path == quarantine_root.name else real_rmdir(path, *args, **kwargs),
        )
    with pytest.raises(RuntimeError, match="recovery failed"):
        module._recover_or_block(stage_d, str(final_root))
    assert source.read_bytes() == b"fake-flac"
    assert journal_path.exists()
    assert quarantine_root.exists()


def test_source_quarantine_is_journaled_and_recovery_cleans_it(tmp_path, monkeypatch):
    """source quarantine 已写入 journal 时，恢复成功后必须清理该目录。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    stage_d = _make_run(str(tmp_path), n=1)
    source = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    final_root = tmp_path / "final"
    quarantine_root = source.parent / (".promote-source-" + "a" * 32 + "-000000")
    quarantine_root.mkdir(mode=0o700)
    shutil.move(str(source), str(quarantine_root / "source"))
    expected_source = module._file_fingerprint(str(quarantine_root / "source"))
    expected_content = module._content_fingerprint(str(quarantine_root / "source"))
    journal_path = tmp_path / "run" / "stage_d" / module.PROMOTE_JOURNAL_NAME
    journal = {
        "version": module._JOURNAL_VERSION,
        "transaction_id": "a" * 32,
        "phase": "published",
        "stage_d_dir": os.path.realpath(stage_d),
        "final_root": os.path.realpath(final_root),
        "owned_roots": [],
        "audio": [],
        "targets": [],
    }
    entry = {
        "source": str(source),
        "target": str(final_root / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"),
        "quarantine_root": str(quarantine_root),
        "source_fingerprint": expected_source,
        "new_content": expected_content,
        "old_content": {"exists": False, "size": 0, "sha256": ""},
        "staged": None,
        "backup": None,
        "source_restore_pending": False,
    }
    journal["audio"].append(entry)
    journal["targets"].append(entry)
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    assert module._recover_or_block(stage_d, str(final_root)) is True
    assert source.read_bytes() == b"fake-flac"
    assert not quarantine_root.exists()
    assert not journal_path.exists()


def test_promote_manifest_commit_failure_restores_source_and_old_destination(tmp_path, monkeypatch):
    """音频已替换、manifest 提交失败时必须恢复 source 与旧 final 集合。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    stage_d = _make_run(str(tmp_path), n=1)
    source = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    source.write_bytes(b"new-audio")
    final_root = tmp_path / "final"
    destination = final_root / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    _write_audio(str(destination), b"old-audio")
    old_record = _passed_record("ds1__src__0", "ds1", str(destination))
    final_manifest = final_root / "manifests" / "ds1.jsonl"
    write_manifest(iter([old_record]), str(final_manifest))
    old_manifest = final_manifest.read_bytes()
    real_replace = module.os.replace
    injected = False

    def fail_manifest_once(source_path, target_path):
        nonlocal injected
        if os.fspath(target_path) == str(final_manifest) and not injected:
            injected = True
            raise OSError("injected manifest commit failure")
        return real_replace(source_path, target_path)

    monkeypatch.setattr(module, "os", patched_os(replace=fail_manifest_once))
    assert module.promote_run(stage_d, str(final_root)) == 1

    assert source.read_bytes() == b"new-audio"
    assert destination.read_bytes() == b"old-audio"
    assert final_manifest.read_bytes() == old_manifest
    assert not (final_root / "library_stats.json").exists()
    assert not (tmp_path / "run" / "stage_d" / ".promote_transaction.json").exists()
    assert not (tmp_path / "run" / "stage_d" / ".promote_complete.json").exists()


def test_unfinished_journal_is_recovered_and_blocks_current_promote(tmp_path, monkeypatch):
    """恢复失败遗留的 journal 在下次调用先恢复，并阻止同次继续发布。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    stage_d = _make_run(str(tmp_path), n=1)
    source = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    final_root = tmp_path / "final"
    final_manifest = final_root / "manifests" / "ds1.jsonl"
    real_replace = module.os.replace
    injected = False

    def fail_manifest_once(source_path, target_path):
        nonlocal injected
        if os.fspath(target_path) == str(final_manifest) and not injected:
            injected = True
            raise OSError("injected manifest commit failure")
        return real_replace(source_path, target_path)

    with monkeypatch.context() as patch:
        patch.setattr(module.os, "replace", fail_manifest_once)
        patch.setattr(
            module,
            "_restore_transaction",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("injected recovery failure")),
        )
        assert module.promote_run(stage_d, str(final_root)) == 1

    journal = tmp_path / "run" / "stage_d" / ".promote_transaction.json"
    assert journal.is_file()
    assert module.promote_run(stage_d, str(final_root)) == 1
    assert source.is_file()
    assert not journal.exists()
    assert not final_manifest.exists()
    assert module.promote_run(stage_d, str(final_root)) == 0


def test_unfinished_journal_recovers_after_two_partial_recovery_failures(
    tmp_path, monkeypatch
):
    """source 已恢复后 target 连续失败两次，后续仍应完成恢复。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    stage_d = _make_run(str(tmp_path), n=1)
    source = (
        tmp_path
        / "run"
        / "stage_d"
        / "D2"
        / "audio"
        / "ds1"
        / "Rain"
        / "shard_00000"
        / "0000.flac"
    )
    source.write_bytes(b"new-audio")
    final_root = tmp_path / "final"
    destination = (
        final_root
        / "audio"
        / "ds1"
        / "Rain"
        / "shard_00000"
        / "0000.flac"
    )
    _write_audio(str(destination), b"old-audio")
    final_manifest = final_root / "manifests" / "ds1.jsonl"
    old_record = _passed_record("ds1__src__0", "ds1", str(destination))
    write_manifest(iter([old_record]), str(final_manifest))
    old_manifest = final_manifest.read_bytes()
    marker_path = tmp_path / "run" / "stage_d" / ".promote_complete.json"
    journal_path = tmp_path / "run" / "stage_d" / ".promote_transaction.json"
    real_replace = module.os.replace
    marker_failed = False
    audio_restore_failures = 0

    def fail_marker_and_two_audio_restores(source_path, target_path):
        nonlocal marker_failed, audio_restore_failures
        source_text = os.fspath(source_path)
        target_text = os.fspath(target_path)
        if (
            target_text == str(marker_path)
            and ".promote-owned-" in source_text
            and not marker_failed
        ):
            marker_failed = True
            raise OSError("injected marker publish failure")
        if (
            target_text == str(destination)
            and os.path.basename(source_text).startswith("backup-")
            and audio_restore_failures < 2
        ):
            audio_restore_failures += 1
            raise OSError("injected audio restore failure")
        return real_replace(source_path, target_path)

    monkeypatch.setattr(module.os, "replace", fail_marker_and_two_audio_restores)

    assert module.promote_run(stage_d, str(final_root)) == 1
    assert journal_path.is_file()
    assert source.read_bytes() == b"new-audio"

    assert module.promote_run(stage_d, str(final_root)) == 1
    assert journal_path.is_file()
    assert audio_restore_failures == 2

    assert module.promote_run(stage_d, str(final_root)) == 1
    assert not journal_path.exists()
    assert source.read_bytes() == b"new-audio"
    assert destination.read_bytes() == b"old-audio"
    assert final_manifest.read_bytes() == old_manifest

    assert module.promote_run(stage_d, str(final_root)) == 0


def test_unfinished_journal_rejects_replaced_existing_source(tmp_path, monkeypatch):
    """恢复期间 source 被替换时不得用未知 bytes 继续或覆盖。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    stage_d = _make_run(str(tmp_path), n=1)
    source = (
        tmp_path
        / "run"
        / "stage_d"
        / "D2"
        / "audio"
        / "ds1"
        / "Rain"
        / "shard_00000"
        / "0000.flac"
    )
    final_root = tmp_path / "final"
    final_manifest = final_root / "manifests" / "ds1.jsonl"
    real_replace = module.os.replace
    injected = False

    def fail_manifest_once(source_path, target_path):
        nonlocal injected
        if os.fspath(target_path) == str(final_manifest) and not injected:
            injected = True
            raise OSError("injected manifest commit failure")
        return real_replace(source_path, target_path)

    with monkeypatch.context() as patch:
        patch.setattr(module.os, "replace", fail_manifest_once)
        patch.setattr(
            module,
            "_restore_transaction",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("injected recovery failure")),
        )
        assert module.promote_run(stage_d, str(final_root)) == 1

    journal = tmp_path / "run" / "stage_d" / ".promote_transaction.json"
    assert journal.is_file()
    source.write_bytes(b"concurrent-unknown-bytes")

    assert module.promote_run(stage_d, str(final_root)) == 1
    assert source.read_bytes() == b"concurrent-unknown-bytes"
    assert journal.is_file()


def test_unfinished_journal_invalidates_marker_and_cleanup_authorization(tmp_path):
    """任意未完成 journal 都必须使 marker 与 Stage D cleanup 授权失效。"""
    from noise_filter_pipeline.scripts.cleanup_pipeline_temp import _promoted_audio_target
    from noise_filter_pipeline.scripts.promote_run_to_final import validate_promote_marker

    stage_d = _make_run(str(tmp_path), n=1)
    final_root = tmp_path / "final"
    assert promote_run(stage_d, str(final_root)) == 0
    journal = tmp_path / "run" / "stage_d" / ".promote_transaction.json"
    journal.write_text("{}\n", encoding="utf-8")

    valid, reason = validate_promote_marker(stage_d)
    assert not valid
    assert "journal" in reason
    target, cleanup_reason = _promoted_audio_target(stage_d)
    assert target is None
    assert "journal" in cleanup_reason


def test_marker_v4_binds_published_audio_sha_and_record_identity(tmp_path):
    """发布音频 bytes 或 marker 内 record identity 被篡改均必须失效。"""
    from noise_filter_pipeline.scripts.promote_run_to_final import validate_promote_marker

    stage_d = _make_run(str(tmp_path), n=1)
    final_root = tmp_path / "final"
    assert promote_run(stage_d, str(final_root)) == 0
    marker_path = tmp_path / "run" / "stage_d" / ".promote_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    audio_path = marker["published_audio"][0]["path"]
    with open(audio_path, "ab") as output:
        output.write(b"tampered")
    valid, reason = validate_promote_marker(stage_d)
    assert not valid
    assert "发布音频" in reason

    with open(audio_path, "rb+") as output:
        output.seek(-len(b"tampered"), os.SEEK_END)
        output.truncate()
    marker["published_records"][0]["record_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    valid, reason = validate_promote_marker(stage_d)
    assert not valid
    assert "published record" in reason


def test_marker_v4_requires_frozen_parser_generator_and_tree_identities(tmp_path):
    """旧的不完整 v4 marker 不得因 version=4 被 cleanup 接纳。"""
    from noise_filter_pipeline.scripts.promote_run_to_final import validate_promote_marker

    stage_d = _make_run(str(tmp_path), n=1)
    assert promote_run(stage_d, str(tmp_path / "final")) == 0
    marker_path = tmp_path / "run" / "stage_d" / ".promote_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["parser_version"] == 10
    assert marker["generator_version"] == "6.1"
    for field in ("parser_version", "generator_version", "stage_b_tree_identity", "final_stage_tree_identity"):
        incomplete = dict(marker)
        incomplete.pop(field)
        marker_path.write_text(json.dumps(incomplete), encoding="utf-8")
        valid, _reason = validate_promote_marker(stage_d)
        assert not valid, field


def test_missing_source_with_arbitrary_destination_is_not_idempotent(tmp_path):
    """没有完整 marker/journal SHA 时，dst 存在不能替代缺失 source。"""
    stage_d = _make_run(str(tmp_path), n=1)
    source = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    final_root = tmp_path / "final"
    destination = final_root / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    _write_audio(str(destination), b"arbitrary-existing-bytes")
    source.unlink()

    assert promote_run(stage_d, str(final_root)) == 1
    assert destination.read_bytes() == b"arbitrary-existing-bytes"
    assert not (final_root / "manifests" / "ds1.jsonl").exists()
    assert not (final_root / "library_stats.json").exists()


def test_marker_publish_failure_rolls_back_stats_manifest_audio_and_source(tmp_path, monkeypatch):
    """marker 是最后一步；其发布失败必须回滚此前所有 destination/source。"""
    from noise_filter_pipeline.scripts import promote_run_to_final as module

    stage_d = _make_run(str(tmp_path), n=1)
    source = tmp_path / "run" / "stage_d" / "D2" / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac"
    original_source = source.read_bytes()
    final_root = tmp_path / "final"
    marker_path = tmp_path / "run" / "stage_d" / ".promote_complete.json"
    real_replace = module.os.replace

    def fail_marker_publish(source_path, target_path):
        if os.fspath(target_path) == str(marker_path) and ".promote-owned-" in os.fspath(source_path):
            raise OSError("injected marker publish failure")
        return real_replace(source_path, target_path)

    monkeypatch.setattr(module.os, "replace", fail_marker_publish)
    assert module.promote_run(stage_d, str(final_root)) == 1

    assert source.read_bytes() == original_source
    assert not (final_root / "audio" / "ds1" / "Rain" / "shard_00000" / "0000.flac").exists()
    assert not (final_root / "manifests" / "ds1.jsonl").exists()
    assert not (final_root / "library_stats.json").exists()
    assert not marker_path.exists()
    assert not (tmp_path / "run" / "stage_d" / ".promote_transaction.json").exists()


def test_marker_v4_binds_stage_b_and_final_stage_trees(tmp_path):
    """Stage B 或最终 Stage D 非音频元数据变化均使 cleanup marker 失效。"""
    from noise_filter_pipeline.scripts.promote_run_to_final import validate_promote_marker

    stage_d = _make_run(str(tmp_path), n=1)
    stage_b = tmp_path / "run" / "stage_b"
    stage_b.mkdir()
    (stage_b / "b_stats.json").write_text("{}\n", encoding="utf-8")
    final_root = tmp_path / "final"
    assert promote_run(stage_d, str(final_root)) == 0
    assert validate_promote_marker(stage_d)[0]

    (stage_b / "b_stats.json").write_text('{"tampered": true}\n', encoding="utf-8")
    valid, reason = validate_promote_marker(stage_d)
    assert not valid
    assert "Stage B" in reason
