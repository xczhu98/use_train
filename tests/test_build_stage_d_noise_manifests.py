"""Stage D strict noise JSONL schema v6 与 generator v6.1 发布契约测试。"""

from __future__ import annotations

import hashlib
import io
import json
import stat
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core.data.noise_admission import ADMISSION_POLICY_VERSION, load_canonical_voice_policy
from core.data.noise_admission_policy import (
    CLASSIFIER_LABEL_COUNT,
    EVIDENCE_CONTRACT_VERSION,
    SCORE_CONTRACT,
    decision_to_admission_extra,
    evaluate_noise_admission,
    load_admission_policy,
    parse_c2_evidence,
)
from core.data.simulation_utils import read_source_scp
from tools.prepare_data_tools.generators.noise.build_stage_d_noise_manifests import (
    GENERATOR_VERSION,
    NOISE_MANIFEST_SCHEMA_VERSION,
    generate,
    generate_split,
)

CANONICAL_POLICY_SHA = load_canonical_voice_policy()[1]
_POLICY = load_admission_policy()
FAKE_ARTIFACTS = {
    "voice_policy_sha256": CANONICAL_POLICY_SHA,
    "audiotag_label_dict_sha256": _POLICY.label_space_sha256,
    "audiotag_checkpoint_sha256": "2" * 64,
    "qwen_model_sha256": "3" * 64,
    "c1_prompt_sha256": "4" * 64,
}


def _write_audio(path: Path, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.ones(sr, dtype=np.float32) * 0.1, sr)


def _write_manifest(path: Path, dataset_id: str, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _stage(status: str = "passed", **extra) -> dict:
    return {"status": status, **extra}


def _mid_for_label(label: str, *, kind: str = "noise") -> str:
    return next(
        mid
        for mid, canonical_label in _POLICY.mid_to_label.items()
        if canonical_label == label and _POLICY.class_of(mid) == kind
    )


def _top5_evidence(hit_mids: list[str]) -> tuple[dict, dict]:
    """构造固定 top-5；完整 527 分数只存在于 C2 内部，不进入 manifest。"""
    mids = list(dict.fromkeys(hit_mids))
    for mid in _POLICY.index_to_mid.values():
        if len(mids) == 5:
            break
        if mid not in mids and _POLICY.class_of(mid) == "noise":
            mids.append(mid)
    probabilities = [1.0, 0.8, 0.6, 0.4, 0.2]
    labels = {
        "audioset_top5": [
            {
                "index": _POLICY.mid_to_index[mid],
                "mid": mid,
                "label": _POLICY.label_of(mid),
                "prob": prob,
            }
            for mid, prob in zip(mids, probabilities)
        ]
    }
    c2 = {
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "score_contract": SCORE_CONTRACT,
        "class_count": CLASSIFIER_LABEL_COUNT,
        "label_space_sha256": _POLICY.label_space_sha256,
    }
    return labels, c2


def _valid_record(
    dataset_id: str,
    record_id: str,
    path,
    category: str,
    *,
    c1_intelligibility: str = "noise",
    d2_status: str = "passed",
    c3_status: str = "passed",
    balance_category: str | None = None,
    admission_allowed: bool = True,
    policy_sha: str = CANONICAL_POLICY_SHA,
    artifacts: dict | None = None,
    execution_mode: dict | None = None,
    publishable: bool = True,
) -> dict:
    """构造固定 top-5 evidence，并用生产 evaluator 生成 C3 frozen contract。"""
    selected = (balance_category or "Noise") if c1_intelligibility == "babble" else category
    if c1_intelligibility == "babble":
        hit_mids = [_mid_for_label(selected), "/m/07qfr4h"]
    else:
        hit_mids = [_mid_for_label(selected)]
    labels, c2_extra = _top5_evidence(hit_mids)
    evidence = parse_c2_evidence(labels, _POLICY, c2_extra)
    decision = evaluate_noise_admission(
        c1_intelligibility=c1_intelligibility,
        evidence=evidence,
        policy=_POLICY,
    )
    c3_extra = decision_to_admission_extra(
        decision,
        execution_mode=execution_mode
        or {"c1": "production", "c2": "subprocess_daemon"},
        publishable=publishable,
        artifact_fingerprints=dict(artifacts or FAKE_ARTIFACTS),
    )
    c3_extra["admission_allowed"] = admission_allowed
    c3_extra["admission_policy_sha256"] = policy_sha
    return {
        "record_id": record_id,
        "dataset_id": dataset_id,
        "source_id": record_id,
        "chunk_id": "",
        "split": "train",
        "audio": {"path": str(path), "sr": 16000, "duration": 1.00003125},
        "labels": labels,
        "stages": {
            "C1": _stage(intelligibility=c1_intelligibility),
            "C2": _stage(**c2_extra),
            "C3": _stage(c3_status, **c3_extra),
            "D2": _stage(d2_status),
        },
        "rejected_reason": None,
        "provenance": {},
    }


def _expected_output(
    dataset_id: str,
    record_id: str,
    path,
    category: str,
    *,
    c1_intelligibility: str = "noise",
    balance_category: str | None = None,
    category_id: int = 0,
    roles: list[str] | None = None,
) -> dict:
    if c1_intelligibility == "babble":
        selected = balance_category or "Noise"
        decision = "accept_babble"
        hit_mids = [_mid_for_label(selected), "/m/07qfr4h"]
    else:
        selected = category
        decision = "accept_noise"
        hit_mids = [_mid_for_label(selected)]
    source = "noise_top5_hit"
    selected_mid = _mid_for_label(selected)
    labels, _ = _top5_evidence(hit_mids)
    top5_hits = [
        {**item, "kind": _POLICY.class_of(item["mid"])}
        for item in labels["audioset_top5"]
    ]
    return {
        "uid": record_id,
        "fs": 16000,
        "length": 16001,
        "path": str(path),
        "manifest_schema_version": NOISE_MANIFEST_SCHEMA_VERSION,
        "balance_category_mid": selected_mid,
        "balance_category": selected,
        "balance_category_source": source,
        "category_name": selected,
        "category_id": category_id,
        "admission_allowed": True,
        "admission_decision": decision,
        "admission_reason_codes": [
            "admitted_babble" if decision == "accept_babble" else "admitted_noise"
        ],
        "admission_policy_version": ADMISSION_POLICY_VERSION,
        "admission_policy_sha256": CANONICAL_POLICY_SHA,
        "verification": {
            "c1_status": "passed",
            "c1_intelligibility": c1_intelligibility,
            "c2_status": "passed",
            "c3_status": "passed",
            "d2_status": "passed",
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "score_contract": SCORE_CONTRACT,
            "class_count": CLASSIFIER_LABEL_COUNT,
            "label_space_sha256": _POLICY.label_space_sha256,
            "artifact_fingerprints": dict(FAKE_ARTIFACTS),
            "top5_hits": top5_hits,
        },
        "dataset_id": dataset_id,
        "roles": roles if roles is not None else ["noise"],
    }


def test_build_stage_d_noise_manifests_filters_status_and_writes_global_vocab(tmp_path):
    root = tmp_path / "stage_d"
    first_audio = root / "audio" / "first.flac"
    second_audio = root / "audio" / "second.flac"
    _write_audio(first_audio)
    _write_audio(second_audio)
    _write_manifest(
        root / "d_manifest_first.jsonl",
        "first",
        [
            _valid_record("first", "first-id", first_audio, "Rain"),
            _valid_record("first", "rejected", second_audio, "Bird", d2_status="rejected"),
        ],
    )
    _write_manifest(
        root / "d_manifest_second.jsonl",
        "second",
        [_valid_record("second", "second-id", second_audio, "Bird")],
    )

    output = tmp_path / "noise_sources.jsonl"
    vocab_path = tmp_path / "noise_categories.json"
    count, category_count, stats = generate(root, output, vocab_path)
    assert count == 2
    assert category_count == 2
    assert stats == {
        "total_categories": 2,
        "total_records": 2,
        "ordinary_noise_se_aec": 2,
        "ordinary_noise_tse": 2,
        "wind": 0,
        "music": 0,
    }

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records == [
        _expected_output("first", "first-id", first_audio, "Rain", category_id=1),
        _expected_output("second", "second-id", second_audio, "Bird", category_id=0),
    ]
    # duration=1.00003125, sr=16000 => 16000.5，生成端 HALF_UP 为 16001；
    # source reader 必须原样读回，不得重新按 binary float 计算或截断。
    _, _, _, lengths, metadata = read_source_scp(output, return_metadata=True)
    assert lengths == {"first-id": 16001, "second-id": 16001}
    assert metadata == {record["uid"]: record for record in records}
    assert json.loads(vocab_path.read_text(encoding="utf-8")) == {"Bird": 0, "Rain": 1}
    publication = json.loads(
        output.with_name(f"{output.name}.metadata.json").read_text(encoding="utf-8")
    )
    assert GENERATOR_VERSION == "6.1"
    assert publication["metadata_schema_version"] == 3
    assert publication["generator_version"] == GENERATOR_VERSION
    assert publication["manifest_schema_version"] == 6
    assert publication["manifest_summary"] == stats
    assert "split" not in publication["manifest_summary"]
    assert publication["artifacts"]["manifest"] == {
        "filename": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "size": output.stat().st_size,
    }


def test_build_stage_d_noise_manifests_does_not_overwrite_without_flag(tmp_path):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    _write_manifest(root / "d_manifest_set.jsonl", "set", [_valid_record("set", "id", audio, "Noise")])
    output = tmp_path / "noise.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        generate(root, output, vocab)
    assert output.read_text(encoding="utf-8") == "old\n"


def test_generate_reuses_only_matching_versioned_publication(tmp_path, monkeypatch):
    from tools.prepare_data_tools.generators.noise import build_stage_d_noise_manifests as generator

    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    _write_manifest(root / "d_manifest_set.jsonl", "set", [_valid_record("set", "id", audio, "Noise")])
    output = tmp_path / "noise.jsonl"
    vocab = tmp_path / "vocab.json"

    expected = generate(root, output, vocab)
    assert generate(root, output, vocab) == expected

    metadata_path = output.with_name(f"{output.name}.metadata.json")
    old_metadata = metadata_path.read_bytes()
    old_fingerprint = json.loads(old_metadata)["publication_fingerprint"]
    monkeypatch.setattr(generator, "GENERATOR_VERSION", "strict-v-next")
    with pytest.raises(FileExistsError, match="stale or incomplete"):
        generate(root, output, vocab)
    assert metadata_path.read_bytes() == old_metadata

    assert generate(root, output, vocab, overwrite=True) == expected
    new_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert new_metadata["generator_version"] == "strict-v-next"
    assert new_metadata["publication_fingerprint"] != old_fingerprint


def test_build_stage_d_noise_manifests_rejects_concurrent_run_lock(tmp_path):
    from tools.noise_filter_pipeline.core.run_lock import acquire_run_lock, release_run_lock

    root = tmp_path / "stage_d"
    root.mkdir()
    lock_fd = acquire_run_lock(str(root.resolve()))
    try:
        with pytest.raises(RuntimeError, match="run 锁"):
            generate(root, tmp_path / "noise.jsonl", tmp_path / "vocab.json")
    finally:
        release_run_lock(lock_fd)


def test_generator_json_writers_reject_nonfinite_numbers():
    from tools.prepare_data_tools.generators.noise.build_stage_d_noise_manifests import (
        _json_writer,
        _jsonl_writer,
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        _json_writer({"value": float("nan")})(io.StringIO())
    with pytest.raises(ValueError, match="Out of range float values"):
        _jsonl_writer([{"value": float("inf")}])(io.StringIO())


def test_generate_reuses_one_admission_context(tmp_path, monkeypatch):
    from core.data.noise_admission import NoiseAdmissionContext

    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    _write_manifest(root / "d_manifest_set.jsonl", "set", [_valid_record("set", "id", audio, "Noise")])
    context = NoiseAdmissionContext.load()
    load_count = 0

    def load_once(cls):
        nonlocal load_count
        load_count += 1
        return context

    monkeypatch.setattr(NoiseAdmissionContext, "load", classmethod(load_once))
    generate(root, tmp_path / "noise.jsonl", tmp_path / "vocab.json")
    assert load_count == 1


def test_generate_rolls_back_manifest_when_vocab_fsync_fails(tmp_path, monkeypatch):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    _write_manifest(root / "d_manifest_set.jsonl", "set", [_valid_record("set", "id", audio, "Noise")])
    output = tmp_path / "noise.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old manifest\n", encoding="utf-8")
    vocab.write_text('{"old":0}\n', encoding="utf-8")

    from core.utils import file_transaction

    real_fsync = file_transaction.os.fsync
    calls = 0

    def fail_vocab_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated vocab fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(file_transaction.os, "fsync", fail_vocab_fsync)
    with pytest.raises(OSError, match="simulated vocab fsync failure"):
        generate(root, output, vocab, overwrite=True)

    assert output.read_text(encoding="utf-8") == "old manifest\n"
    assert vocab.read_text(encoding="utf-8") == '{"old":0}\n'
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_generate_split_writes_reproducible_three_file_publication(tmp_path):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    records = [
        _valid_record("set", f"id-{index:02d}", audio, "Rain" if index % 2 else "Bird")
        for index in range(10)
    ]
    _write_manifest(root / "d_manifest_set.jsonl", "set", records)

    output_dir = tmp_path / "noise" / "filter"
    train_count, val_count, category_count, stats = generate_split(
        root, output_dir, val_ratio=0.2, seed=42
    )
    assert train_count == 8
    assert val_count == 2
    assert category_count == 2
    assert stats == {
        "total_categories": 2,
        "total_records": 10,
        "ordinary_noise_se_aec": 2,
        "ordinary_noise_tse": 2,
        "wind": 0,
        "music": 0,
    }
    train_path = output_dir / "train_noise_sources.jsonl"
    val_path = output_dir / "val_noise_sources.jsonl"
    train_records = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    val_records = [json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines()]

    assert {record["uid"] for record in train_records}.isdisjoint(
        {record["uid"] for record in val_records}
    )
    assert {record["uid"] for record in train_records + val_records} == {
        f"id-{index:02d}" for index in range(10)
    }
    records_by_category = {
        record["balance_category"]: record["category_id"]
        for record in train_records + val_records
    }
    assert records_by_category == {"Bird": 0, "Rain": 1}
    for record in train_records + val_records:
        assert record["category_name"] == record["balance_category"]
        assert set(record["roles"]) == {"noise"}
        assert record["manifest_schema_version"] == NOISE_MANIFEST_SCHEMA_VERSION == 6
        assert record["verification"]["artifact_fingerprints"] == FAKE_ARTIFACTS
        assert len(record["verification"]["top5_hits"]) == 5
        assert all(
            record["verification"][field] == "passed"
            for field in ("c1_status", "c2_status", "c3_status", "d2_status")
        )
    assert not (output_dir / "noise_categories.json").exists()
    assert not (output_dir / "noise_category_roles.json").exists()
    publication_path = output_dir / "noise_publication_metadata.json"
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    assert publication["metadata_schema_version"] == 3
    assert publication["generator_version"] == GENERATOR_VERSION
    assert publication["parameters"] == {"seed": 42, "val_ratio": 0.2}
    assert publication["manifest_summary"] == {
        **stats,
        "split": {"train": 8, "val": 2},
    }
    assert set(publication["artifacts"]) == {"train_manifest", "val_manifest"}
    assert publication["artifacts"]["train_manifest"]["sha256"] == hashlib.sha256(
        train_path.read_bytes()
    ).hexdigest()
    fingerprint_payload = {
        key: value
        for key, value in publication.items()
        if key != "publication_fingerprint"
    }
    canonical = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    assert publication["publication_fingerprint"] == hashlib.sha256(canonical).hexdigest()
    # sidecar 不改变 strict v6 record schema，既有 consumer 仍可原样读取。
    _, _, _, lengths, parsed_metadata = read_source_scp(train_path, return_metadata=True)
    assert lengths == {record["uid"]: record["length"] for record in train_records}
    assert parsed_metadata == {record["uid"]: record for record in train_records}

    second_output_dir = tmp_path / "noise-second" / "filter"
    assert generate_split(root, second_output_dir, val_ratio=0.2, seed=42)[:3] == (8, 2, 2)
    assert (second_output_dir / "train_noise_sources.jsonl").read_text(encoding="utf-8") == train_path.read_text(encoding="utf-8")
    assert (second_output_dir / "val_noise_sources.jsonl").read_text(encoding="utf-8") == val_path.read_text(encoding="utf-8")


def test_generate_split_rolls_back_all_outputs_when_publish_fails_midway(tmp_path, monkeypatch):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    records = [_valid_record("set", f"id-{index}", audio, "Noise") for index in range(4)]
    _write_manifest(root / "d_manifest_set.jsonl", "set", records)
    output_dir = tmp_path / "noise"
    output_dir.mkdir()
    publication_names = [
        "train_noise_sources.jsonl",
        "val_noise_sources.jsonl",
        "noise_publication_metadata.json",
    ]
    legacy_names = [
        "noise_pool_stats.json",
        "noise_categories.json",
        "noise_category_roles.json",
    ]
    old_contents = {}
    for index, name in enumerate(publication_names + legacy_names):
        old_contents[name] = f"old-{index}\n"
        (output_dir / name).write_text(old_contents[name], encoding="utf-8")

    from core.utils import file_transaction

    real_replace = file_transaction.os.replace
    publish_calls = 0

    def fail_metadata_publish(source_path, target_path, *args, **kwargs):
        nonlocal publish_calls
        if Path(source_path).suffix == ".tmp":
            publish_calls += 1
            if publish_calls == len(publication_names):
                raise OSError("simulated metadata publish failure")
        return real_replace(source_path, target_path, *args, **kwargs)

    monkeypatch.setattr(file_transaction.os, "replace", fail_metadata_publish)
    with pytest.raises(OSError, match="simulated metadata publish failure"):
        generate_split(root, output_dir, val_ratio=0.25, overwrite=True)

    assert {
        name: (output_dir / name).read_text(encoding="utf-8")
        for name in publication_names + legacy_names
    } == old_contents
    assert not list(output_dir.glob(".*.tmp"))
    assert not list(output_dir.glob(".*.bak"))


def test_generate_split_fsyncs_all_staging_files_before_first_replace(tmp_path, monkeypatch):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    records = [_valid_record("set", f"id-{index}", audio, "Noise") for index in range(4)]
    _write_manifest(root / "d_manifest_set.jsonl", "set", records)
    output_dir = tmp_path / "noise"

    from core.utils import file_transaction

    real_fsync = file_transaction.os.fsync
    real_replace = file_transaction.os.replace
    staging_fsyncs = 0
    first_replace_fsyncs = None

    def count_fsync(fd):
        nonlocal staging_fsyncs
        staging_fsyncs += 1
        return real_fsync(fd)

    def observe_first_replace(source_path, target_path, *args, **kwargs):
        nonlocal first_replace_fsyncs
        if Path(source_path).suffix == ".tmp" and first_replace_fsyncs is None:
            first_replace_fsyncs = staging_fsyncs
        return real_replace(source_path, target_path, *args, **kwargs)

    monkeypatch.setattr(file_transaction.os, "fsync", count_fsync)
    monkeypatch.setattr(file_transaction.os, "replace", observe_first_replace)
    generate_split(root, output_dir, val_ratio=0.25)

    assert first_replace_fsyncs == 3
    assert all((output_dir / name).is_file() for name in [
        "train_noise_sources.jsonl",
        "val_noise_sources.jsonl",
        "noise_publication_metadata.json",
    ])


def test_generate_split_overwrite_removes_all_regular_legacy_sidecars_after_publish(
    tmp_path, monkeypatch
):
    from tools.prepare_data_tools.generators.noise import build_stage_d_noise_manifests as generator

    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    records = [_valid_record("set", f"id-{index}", audio, "Noise") for index in range(4)]
    _write_manifest(root / "d_manifest_set.jsonl", "set", records)
    output_dir = tmp_path / "noise"
    output_dir.mkdir()
    legacy_publication = {
        "train_noise_sources.jsonl": "old train\n",
        "val_noise_sources.jsonl": "old val\n",
        "noise_categories.json": "{}\n",
        "noise_category_roles.json": "{}\n",
        "noise_publication_metadata.json": "{}\n",
        "noise_pool_stats.json": "legacy stats\n",
    }
    for name, content in legacy_publication.items():
        (output_dir / name).write_text(content, encoding="utf-8")
    legacy_names = [
        "noise_pool_stats.json",
        "noise_categories.json",
        "noise_category_roles.json",
    ]

    real_fsync = generator.os.fsync
    directory_fsyncs = 0

    def count_directory_fsync(fd):
        nonlocal directory_fsyncs
        if stat.S_ISDIR(generator.os.fstat(fd).st_mode):
            directory_fsyncs += 1
        return real_fsync(fd)

    monkeypatch.setattr(generator.os, "fsync", count_directory_fsync)
    generate_split(root, output_dir, val_ratio=0.25, overwrite=True)

    assert all(not (output_dir / name).exists() for name in legacy_names)
    assert directory_fsyncs == 2
    publication = json.loads(
        (output_dir / "noise_publication_metadata.json").read_text(encoding="utf-8")
    )
    assert publication["metadata_schema_version"] == 3
    assert set(publication["artifacts"]) == {"train_manifest", "val_manifest"}


def test_generate_split_rejects_legacy_sidecar_identity_change_before_cleanup(
    tmp_path, monkeypatch
):
    from tools.prepare_data_tools.generators.noise import build_stage_d_noise_manifests as generator

    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    _write_manifest(
        root / "d_manifest_set.jsonl",
        "set",
        [_valid_record("set", "id", audio, "Noise")],
    )
    output_dir = tmp_path / "noise"
    output_dir.mkdir()
    legacy_vocab = output_dir / "noise_categories.json"
    legacy_vocab.write_text("old vocab\n", encoding="utf-8")
    real_publish = generator._publish_or_reuse

    def publish_then_replace(*args, **kwargs):
        real_publish(*args, **kwargs)
        replacement = output_dir / "replacement.json"
        replacement.write_text("replacement\n", encoding="utf-8")
        replacement.replace(legacy_vocab)

    monkeypatch.setattr(generator, "_publish_or_reuse", publish_then_replace)
    with pytest.raises(RuntimeError, match="changed before cleanup"):
        generate_split(root, output_dir, val_ratio=0.0, overwrite=True)

    assert legacy_vocab.read_text(encoding="utf-8") == "replacement\n"
    assert (output_dir / "train_noise_sources.jsonl").is_file()
    assert (output_dir / "noise_publication_metadata.json").is_file()


@pytest.mark.parametrize(
    "legacy_name",
    ["noise_pool_stats.json", "noise_categories.json", "noise_category_roles.json"],
)
@pytest.mark.parametrize("leaf_kind", ["symlink", "directory"])
def test_generate_split_overwrite_rejects_unsafe_legacy_sidecar_leaf_before_publish(
    tmp_path, legacy_name, leaf_kind
):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    _write_manifest(
        root / "d_manifest_set.jsonl",
        "set",
        [_valid_record("set", "id", audio, "Noise")],
    )
    output_dir = tmp_path / "noise"
    output_dir.mkdir()
    legacy_path = output_dir / legacy_name
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    if leaf_kind == "symlink":
        legacy_path.symlink_to(sentinel)
    else:
        legacy_path.mkdir()

    with pytest.raises(RuntimeError, match="legacy publication leaf must be a regular file"):
        generate_split(root, output_dir, overwrite=True)

    assert legacy_path.is_symlink() if leaf_kind == "symlink" else legacy_path.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"
    assert not (output_dir / "train_noise_sources.jsonl").exists()
    assert not (output_dir / "noise_publication_metadata.json").exists()


def test_build_stage_d_noise_manifests_tags_wind_and_music_roles(tmp_path):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    _write_manifest(
        root / "d_manifest_set.jsonl",
        "set",
        [
            _valid_record("set", "wind-id", audio, "Wind"),
            _valid_record("set", "mic-wind-id", audio, "Wind noise (microphone)"),
            _valid_record("set", "music-id", audio, "Music"),
            _valid_record("set", "guitar-id", audio, "Guitar"),
            _valid_record("set", "noise-id", audio, "Traffic noise, roadway noise"),
        ],
    )
    output = tmp_path / "noise_sources.jsonl"
    vocab_path = tmp_path / "noise_categories.json"
    _, _, stats = generate(root, output, vocab_path)

    records = {json.loads(line)["uid"]: json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()}
    assert records["wind-id"]["roles"] == ["wind"]
    assert records["mic-wind-id"]["roles"] == ["wind"]
    assert records["music-id"]["roles"] == ["music"]
    assert records["guitar-id"]["roles"] == ["music"]
    assert records["noise-id"]["roles"] == ["noise"]
    assert stats == {
        "total_categories": 5,
        "total_records": 5,
        "ordinary_noise_se_aec": 5 - 2,
        "ordinary_noise_tse": 5 - 2 - 2,
        "wind": 2,
        "music": 2,
    }


def test_generate_split_does_not_overwrite_without_flag(tmp_path):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    _write_manifest(root / "d_manifest_set.jsonl", "set", [_valid_record("set", "id", audio, "Noise")])
    output_dir = tmp_path / "noise"
    output_dir.mkdir()
    train_path = output_dir / "train_noise_sources.jsonl"
    train_path.write_text("old\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        generate_split(root, output_dir)
    assert train_path.read_text(encoding="utf-8") == "old\n"


def test_generate_split_requires_overwrite_when_reusable_three_file_set_has_legacy_sidecar(
    tmp_path,
):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    _write_manifest(
        root / "d_manifest_set.jsonl",
        "set",
        [_valid_record("set", "id", audio, "Noise")],
    )
    output_dir = tmp_path / "noise"
    expected = generate_split(root, output_dir, val_ratio=0.0)
    legacy_vocab = output_dir / "noise_categories.json"
    legacy_vocab.write_text('{"Noise":0}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        generate_split(root, output_dir, val_ratio=0.0)

    assert legacy_vocab.read_text(encoding="utf-8") == '{"Noise":0}\n'
    assert generate_split(root, output_dir, val_ratio=0.0, overwrite=True) == expected
    assert not legacy_vocab.exists()


def test_generate_rejects_audio_path_outside_stage_d_root(tmp_path):
    root = tmp_path / "stage_d"
    root.mkdir()
    inside_audio = root / "audio.flac"
    outside_audio = tmp_path / "outside.flac"
    _write_audio(inside_audio)
    _write_audio(outside_audio)
    _write_manifest(
        root / "d_manifest_set.jsonl",
        "set",
        [_valid_record("set", "id", outside_audio, "Noise")],
    )
    output = tmp_path / "noise_sources.jsonl"
    vocab = tmp_path / "vocab.json"
    with pytest.raises(ValueError, match="escapes stage-d-root"):
        generate(root, output, vocab)


def test_generate_rejects_audio_symlink_inside_stage_d_root(tmp_path):
    root = tmp_path / "stage_d"
    root.mkdir()
    outside_audio = tmp_path / "outside.flac"
    _write_audio(outside_audio)
    symlink_audio = root / "audio.flac"
    symlink_audio.symlink_to(outside_audio)
    _write_manifest(
        root / "d_manifest_set.jsonl",
        "set",
        [_valid_record("set", "id", symlink_audio, "Noise")],
    )

    with pytest.raises(ValueError, match="symbolic link"):
        generate(root, tmp_path / "noise_sources.jsonl", tmp_path / "vocab.json")


def test_generate_rejects_audio_with_symlinked_ancestor_inside_stage_d_root(tmp_path):
    root = tmp_path / "stage_d"
    real_audio_dir = root / "real_audio"
    audio = real_audio_dir / "audio.flac"
    _write_audio(audio)
    symlink_dir = root / "audio"
    symlink_dir.symlink_to(real_audio_dir, target_is_directory=True)
    symlinked_audio = symlink_dir / audio.name
    _write_manifest(
        root / "d_manifest_set.jsonl",
        "set",
        [_valid_record("set", "id", symlinked_audio, "Noise")],
    )

    with pytest.raises(ValueError, match="stable non-symlink entries"):
        generate(root, tmp_path / "noise_sources.jsonl", tmp_path / "vocab.json")


def test_generate_split_val_ratio_zero_produces_empty_val(tmp_path):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    records = [_valid_record("set", f"id-{index:02d}", audio, "Noise") for index in range(5)]
    _write_manifest(root / "d_manifest_set.jsonl", "set", records)
    output_dir = tmp_path / "noise" / "filter"
    train_count, val_count, category_count, stats = generate_split(
        root, output_dir, val_ratio=0.0, seed=42
    )
    assert train_count == 5
    assert val_count == 0
    assert category_count == 1
    assert (output_dir / "val_noise_sources.jsonl").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record["stages"].pop("C1"),
        lambda record: record["stages"].__setitem__("C1", _stage("rejected")),
        lambda record: record["stages"]["C1"].pop("intelligibility"),
        lambda record: record["stages"].pop("C2"),
        lambda record: record["stages"]["C2"].__setitem__("evidence_contract_version", 4),
        lambda record: record["stages"]["C2"].__setitem__("class_count", 526),
        lambda record: record["labels"].pop("audioset_top5"),
        lambda record: record["labels"]["audioset_top5"][0].__setitem__("prob", float("nan")),
        lambda record: record["labels"].__setitem__("audioset_top5", []),
        lambda record: record["labels"]["audioset_top5"][0].__setitem__("prob", 0.5),
        lambda record: record["stages"].pop("C3"),
        lambda record: record["stages"].__setitem__("C3", _stage("rejected")),
        lambda record: record["stages"]["C3"].__setitem__("admission_allowed", False),
        lambda record: record["stages"]["C3"].pop("admission_decision"),
        lambda record: record["stages"]["C3"].pop("top5_hits"),
        lambda record: record["stages"]["C3"].__setitem__("content_kind", "speech"),
        lambda record: record["stages"]["C3"].__setitem__("quarantine", True),
        lambda record: record["stages"]["C3"].pop("balance_category_mid"),
        lambda record: record["stages"]["C3"].pop("balance_category_source"),
        lambda record: record["stages"]["C3"].pop("artifact_fingerprints"),
        lambda record: record["stages"]["C3"].pop("execution_mode"),
        lambda record: record["stages"]["C3"].pop("publishable"),
        lambda record: record["stages"].pop("D2"),
    ],
)
def test_generate_stage_d_rejects_stale_or_malformed_contract_and_preserves_output(
    tmp_path, mutate
):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    record = _valid_record("set", "id", audio, "Rain")
    mutate(record)
    _write_manifest(root / "d_manifest_set.jsonl", "set", [record])
    output = tmp_path / "noise.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old output\n", encoding="utf-8")

    with pytest.raises(ValueError):
        generate(root, output, vocab, overwrite=True)
    assert output.read_text(encoding="utf-8") == "old output\n"
    assert not vocab.exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("legacy_stage", ["C4", "C", "D"])
def test_generate_stage_d_rejects_legacy_admission_stages(tmp_path, legacy_stage):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    record = _valid_record("set", "id", audio, "Rain")
    record["stages"][legacy_stage] = dict(record["stages"]["C3"])
    _write_manifest(root / "d_manifest_set.jsonl", "set", [record])

    with pytest.raises(ValueError, match="legacy admission stages are forbidden"):
        generate(root, tmp_path / "noise.jsonl", tmp_path / "vocab.json")


def test_generate_stage_d_rejects_dry_run_with_valid_artifact_hashes(tmp_path):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    record = _valid_record(
        "set",
        "id",
        audio,
        "Rain",
        execution_mode={"c1": "dry_run", "c2": "subprocess_daemon"},
        publishable=False,
    )
    _write_manifest(root / "d_manifest_set.jsonl", "set", [record])
    output = tmp_path / "noise.jsonl"
    output.write_text("old output\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Stage C execution is not production-publishable"):
        generate(root, output, tmp_path / "vocab.json", overwrite=True)
    assert output.read_text(encoding="utf-8") == "old output\n"


def test_generate_stage_d_accept_babble_ignores_babble_mid_for_decision(tmp_path):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    _write_manifest(
        root / "d_manifest_set.jsonl",
        "set",
        [
            _valid_record(
                "set", "babble-id", audio, "Conversation", c1_intelligibility="babble"
            ),
            _valid_record("set", "noise-id", audio, "Rain"),
        ],
    )
    output = tmp_path / "noise_sources.jsonl"
    vocab_path = tmp_path / "noise_categories.json"
    count, category_count, _ = generate(root, output, vocab_path)
    assert (count, category_count) == (2, 2)

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records == [
        _expected_output(
            "set",
            "babble-id",
            audio,
            "Conversation",
            c1_intelligibility="babble",
            category_id=0,
        ),
        _expected_output("set", "noise-id", audio, "Rain", category_id=1),
    ]
    assert json.loads(vocab_path.read_text(encoding="utf-8")) == {"Noise": 0, "Rain": 1}


def test_generate_stage_d_accept_babble_uses_highest_noise_hit(tmp_path):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    record = _valid_record(
        "set",
        "babble-id",
        audio,
        "Conversation",
        c1_intelligibility="babble",
        balance_category="Music",
    )
    _write_manifest(root / "d_manifest_set.jsonl", "set", [record])
    output = tmp_path / "noise.jsonl"
    generate(root, output, tmp_path / "vocab.json")

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["admission_decision"] == "accept_babble"
    assert result["balance_category_mid"] == "/m/04rlf"
    assert result["balance_category"] == "Music"
    assert result["balance_category_source"] == "noise_top5_hit"
    assert result["roles"] == ["music"]


def test_generate_stage_d_rejects_speech_mid_in_top5(tmp_path):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    record = _valid_record(
        "set", "id", audio, "Conversation", c1_intelligibility="babble"
    )
    speech_mid = "/m/0261r1"
    item = record["labels"]["audioset_top5"][0]
    item.update(
        index=_POLICY.mid_to_index[speech_mid],
        mid=speech_mid,
        label=_POLICY.label_of(speech_mid),
    )
    _write_manifest(root / "d_manifest_set.jsonl", "set", [record])

    with pytest.raises(ValueError, match="does not match recomputed value"):
        generate(root, tmp_path / "noise.jsonl", tmp_path / "vocab.json")


def test_generate_stage_d_rejects_policy_hash_mismatch(tmp_path):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    record = _valid_record("set", "id", audio, "Rain", policy_sha="0" * 64)
    _write_manifest(root / "d_manifest_set.jsonl", "set", [record])

    with pytest.raises(ValueError, match="admission_policy_sha256.*does not match recomputed value"):
        generate(root, tmp_path / "noise.jsonl", tmp_path / "vocab.json")


def test_generate_stage_d_rejects_legacy_calibration_field(tmp_path):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    record = _valid_record("set", "id", audio, "Rain")
    record["stages"]["C3"]["admission_calibration_sha256"] = "0" * 64
    _write_manifest(root / "d_manifest_set.jsonl", "set", [record])

    with pytest.raises(ValueError, match="legacy admission fields are forbidden"):
        generate(root, tmp_path / "noise.jsonl", tmp_path / "vocab.json")


def test_generate_stage_d_rejects_cross_record_artifact_inconsistency(tmp_path):
    """同一批导出的候选必须共享 artifact fingerprints。"""
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    other_artifacts = {
        **FAKE_ARTIFACTS,
        "qwen_model_sha256": "f" * 64,
    }
    _write_manifest(
        root / "d_manifest_set.jsonl",
        "set",
        [
            _valid_record("set", "id-a", audio, "Rain"),
            _valid_record("set", "id-b", audio, "Bird", artifacts=other_artifacts),
        ],
    )

    with pytest.raises(ValueError, match="artifact fingerprints differ within one manifest"):
        generate(root, tmp_path / "noise.jsonl", tmp_path / "vocab.json")


def test_generate_stage_d_rejects_cross_manifest_artifact_inconsistency(tmp_path):
    """跨 manifest 候选同样必须共享 artifact fingerprints。"""
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    other_artifacts = {
        **FAKE_ARTIFACTS,
        "qwen_model_sha256": "f" * 64,
    }
    _write_manifest(
        root / "d_manifest_first.jsonl",
        "first",
        [_valid_record("first", "id-a", audio, "Rain")],
    )
    _write_manifest(
        root / "d_manifest_second.jsonl",
        "second",
        [_valid_record("second", "id-b", audio, "Bird", artifacts=other_artifacts)],
    )

    with pytest.raises(ValueError, match="artifact fingerprints differ within one manifest"):
        generate(root, tmp_path / "noise.jsonl", tmp_path / "vocab.json")


def test_generate_stage_d_output_fields_match_v6_contract_exactly(tmp_path):
    """输出字段必须与 strict schema v6 契约逐字段一致。"""
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    _write_manifest(
        root / "d_manifest_set.jsonl",
        "set",
        [_valid_record("set", "id", audio, "Traffic noise, roadway noise")],
    )
    output = tmp_path / "noise.jsonl"
    generate(root, output, tmp_path / "vocab.json")

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record == _expected_output("set", "id", audio, "Traffic noise, roadway noise", category_id=0)
    assert set(record) == {
        "uid",
        "fs",
        "length",
        "path",
        "manifest_schema_version",
        "balance_category_mid",
        "balance_category",
        "balance_category_source",
        "category_name",
        "category_id",
        "admission_allowed",
        "admission_decision",
        "admission_reason_codes",
        "admission_policy_version",
        "admission_policy_sha256",
        "verification",
        "dataset_id",
        "roles",
    }
    assert record["manifest_schema_version"] == 6
    assert set(record["verification"]) == {
        "c1_status",
        "c1_intelligibility",
        "c2_status",
        "c3_status",
        "d2_status",
        "evidence_contract_version",
        "score_contract",
        "class_count",
        "label_space_sha256",
        "artifact_fingerprints",
        "top5_hits",
    }


@pytest.mark.parametrize("blocked_operation", ["open", "read", "sha256"])
def test_generate_never_reads_or_hashes_audio_payloads(
    tmp_path, monkeypatch, blocked_operation
):
    from tools.prepare_data_tools.generators.noise import build_stage_d_noise_manifests as generator

    root = tmp_path / "stage_d"
    root.mkdir()
    audio_payloads = {
        root / "noise.wav": b"WAV payload must remain unopened",
        root / "noise.flac": b"FLAC payload must remain unopened",
        root / "room.rir": b"RIR payload must remain unopened",
    }
    for path, payload in audio_payloads.items():
        path.write_bytes(payload)
    _write_manifest(
        root / "d_manifest_set.jsonl",
        "set",
        [
            _valid_record("set", f"id-{index}", path, "Noise")
            for index, path in enumerate(audio_payloads)
        ],
    )

    audio_names = {path.name for path in audio_payloads}
    real_open = generator.os.open
    audio_fds: set[int] = set()

    def guarded_open(path, flags, *args, **kwargs):
        is_audio = Path(path).name in audio_names
        if blocked_operation == "open" and is_audio:
            raise AssertionError(f"audio leaf opened: {path}")
        descriptor = real_open(path, flags, *args, **kwargs)
        if is_audio:
            audio_fds.add(descriptor)
        return descriptor

    monkeypatch.setattr(generator.os, "open", guarded_open)
    if blocked_operation == "read":
        real_read = generator.os.read

        def guarded_read(descriptor, size):
            if descriptor in audio_fds:
                raise AssertionError(f"audio leaf read: fd={descriptor}")
            return real_read(descriptor, size)

        monkeypatch.setattr(generator.os, "read", guarded_read)
    elif blocked_operation == "sha256":
        real_sha256 = generator.hashlib.sha256
        forbidden_payloads = set(audio_payloads.values())

        class GuardedDigest:
            def __init__(self, payload=b""):
                if payload in forbidden_payloads:
                    raise AssertionError("audio payload hashed by sha256 constructor")
                self._digest = real_sha256(payload)

            def update(self, payload):
                if payload in forbidden_payloads:
                    raise AssertionError("audio payload hashed by sha256 update")
                return self._digest.update(payload)

            def hexdigest(self):
                return self._digest.hexdigest()

        monkeypatch.setattr(generator.hashlib, "sha256", GuardedDigest)

    count, category_count, _ = generate(
        root,
        tmp_path / f"noise-{blocked_operation}.jsonl",
        tmp_path / f"vocab-{blocked_operation}.json",
    )
    assert (count, category_count) == (3, 1)
    assert not audio_fds


def test_generate_rejects_audio_replacement_during_stat_identity(tmp_path, monkeypatch):
    from tools.prepare_data_tools.generators.noise import build_stage_d_noise_manifests as generator

    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    replacement = root / "replacement.flac"
    _write_audio(audio)
    _write_audio(replacement, sr=8000)
    _write_manifest(
        root / "d_manifest_set.jsonl",
        "set",
        [_valid_record("set", "id", audio, "Noise")],
    )
    real_stat = generator.os.stat
    replaced = False

    def replace_after_first_leaf_stat(path, *args, **kwargs):
        nonlocal replaced
        value = real_stat(path, *args, **kwargs)
        if not replaced and Path(path) == Path(audio.name) and "dir_fd" in kwargs:
            replaced = True
            replacement.replace(audio)
        return value

    monkeypatch.setattr(generator.os, "stat", replace_after_first_leaf_stat)
    with pytest.raises(
        ValueError,
        match="audio (?:ancestor )?identity changed during stat",
    ):
        generate(root, tmp_path / "noise.jsonl", tmp_path / "vocab.json")
    assert replaced


def test_generate_rejects_manifest_identity_change_before_publication(tmp_path, monkeypatch):
    from tools.prepare_data_tools.generators.noise import build_stage_d_noise_manifests as generator

    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    manifest = root / "d_manifest_set.jsonl"
    _write_audio(audio)
    _write_manifest(manifest, "set", [_valid_record("set", "id", audio, "Noise")])
    real_assert = generator._assert_input_snapshots

    def replace_manifest(stage_d_root, manifests, audios):
        replacement = manifest.with_suffix(".replacement")
        replacement.write_bytes(manifest.read_bytes())
        replacement.replace(manifest)
        return real_assert(stage_d_root, manifests, audios)

    monkeypatch.setattr(generator, "_assert_input_snapshots", replace_manifest)
    with pytest.raises(ValueError, match="manifest identity changed before publication"):
        generate(root, tmp_path / "noise.jsonl", tmp_path / "vocab.json")


def test_generate_rejects_audio_stat_change_before_publication(tmp_path, monkeypatch):
    from tools.prepare_data_tools.generators.noise import build_stage_d_noise_manifests as generator

    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    _write_manifest(
        root / "d_manifest_set.jsonl",
        "set",
        [_valid_record("set", "id", audio, "Noise")],
    )
    real_assert = generator._assert_input_snapshots

    def change_audio_stat(stage_d_root, manifests, audios):
        before = audio.stat()
        audio.chmod(stat.S_IMODE(before.st_mode) ^ stat.S_IXUSR)
        assert audio.stat().st_ctime_ns != before.st_ctime_ns
        return real_assert(stage_d_root, manifests, audios)

    monkeypatch.setattr(generator, "_assert_input_snapshots", change_audio_stat)
    with pytest.raises(ValueError, match="audio identity changed before publication"):
        generate(root, tmp_path / "noise.jsonl", tmp_path / "vocab.json")


def test_generate_rejects_audio_replacement_before_publication(tmp_path, monkeypatch):
    from tools.prepare_data_tools.generators.noise import build_stage_d_noise_manifests as generator

    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    _write_manifest(
        root / "d_manifest_set.jsonl",
        "set",
        [_valid_record("set", "id", audio, "Noise")],
    )
    real_assert = generator._assert_input_snapshots

    def replace_audio(stage_d_root, manifests, audios):
        replacement = root / "replacement.flac"
        _write_audio(replacement, sr=8000)
        replacement.replace(audio)
        return real_assert(stage_d_root, manifests, audios)

    monkeypatch.setattr(generator, "_assert_input_snapshots", replace_audio)
    with pytest.raises(ValueError, match="audio identity changed before publication"):
        generate(root, tmp_path / "noise.jsonl", tmp_path / "vocab.json")


def test_generate_rejects_unknown_audiorecord_field(tmp_path):
    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    record = _valid_record("set", "id", audio, "Noise")
    record["unknown_future_field"] = True
    _write_manifest(root / "d_manifest_set.jsonl", "set", [record])

    with pytest.raises(ValueError, match="invalid AudioRecord fields"):
        generate(root, tmp_path / "noise.jsonl", tmp_path / "vocab.json")


def test_generate_streams_large_stage_d_manifest_without_bytes_snapshot(
    tmp_path,
    monkeypatch,
):
    from tools.prepare_data_tools.generators.noise import build_stage_d_noise_manifests as generator

    root = tmp_path / "stage_d"
    audio = root / "audio.flac"
    _write_audio(audio)
    records = [
        _valid_record("set", f"id-{index:04d}", audio, "Noise")
        for index in range(1000)
    ]
    manifest = root / "d_manifest_set.jsonl"
    _write_manifest(manifest, "set", records)
    real_snapshot = generator._stable_file_snapshot

    def reject_manifest_bytes_snapshot(path, *, include_bytes, root=None):
        assert not (Path(path) == manifest and include_bytes)
        return real_snapshot(path, include_bytes=include_bytes, root=root)

    monkeypatch.setattr(generator, "_stable_file_snapshot", reject_manifest_bytes_snapshot)

    count, category_count, _ = generate(
        root,
        tmp_path / "noise.jsonl",
        tmp_path / "vocab.json",
    )

    assert count == 1000
    assert category_count == 1


pytestmark = [pytest.mark.unit]