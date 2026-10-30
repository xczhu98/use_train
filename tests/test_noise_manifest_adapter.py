"""Contract tests for strict-v6 final-manifest publication and materialization."""

from __future__ import annotations

import json
import multiprocessing
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from core.data.noise_admission import (
    NOISE_MANIFEST_SCHEMA_VERSION,
    load_canonical_voice_policy,
    sha256_path,
    validate_noise_manifest_records,
)
from core.data.noise_admission_policy import (
    ADMISSION_POLICY_VERSION,
    CLASSIFIER_LABEL_COUNT,
    EVIDENCE_CONTRACT_VERSION,
    LEGACY_C2_LABEL_FIELDS,
    LEGACY_C3_FIELDS,
    SCORE_CONTRACT,
    decision_to_admission_extra,
    evaluate_noise_admission,
    load_admission_policy,
    load_label_metadata,
    parse_c2_evidence,
)
from core.data.simulation_utils import materialize_final_noise_manifest, read_source_scp

_POLICY = load_admission_policy()
_LABEL_METADATA = load_label_metadata()
CANONICAL_POLICY_SHA = _POLICY.sha256
FAKE_ARTIFACTS = {
    "voice_policy_sha256": CANONICAL_POLICY_SHA,
    "audiotag_label_dict_sha256": _POLICY.label_space_sha256,
    "audiotag_checkpoint_sha256": "2" * 64,
    "qwen_model_sha256": "3" * 64,
    "c1_prompt_sha256": "4" * 64,
}
OTHER_ARTIFACTS = {**FAKE_ARTIFACTS, "qwen_model_sha256": "f" * 64}
_SAFE_NOISE_LABELS = ("Noise", "Rain", "Wind", "Music", "Bird")


def _final_manifest_path(noise_root: Path, dataset_id: str) -> Path:
    path = noise_root / "final" / "manifests" / f"{dataset_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _stage(status: str = "passed", **extra) -> dict:
    return {"status": status, **extra}


def _c2_labels(score_by_label: dict[str, float]) -> dict:
    """模拟完整 527 内部排序，但只持久化固定 audioset_top5。"""
    scores = [0.0] * CLASSIFIER_LABEL_COUNT
    for rank, label in enumerate(_SAFE_NOISE_LABELS):
        index = _LABEL_METADATA.index_of(_LABEL_METADATA.label_to_mid[label])
        scores[index] = 0.05 - rank * 0.01
    for label, score in score_by_label.items():
        scores[_LABEL_METADATA.index_of(_LABEL_METADATA.label_to_mid[label])] = score
    top_indices = sorted(
        range(CLASSIFIER_LABEL_COUNT), key=lambda index: (-scores[index], index)
    )[:5]
    return {
        "audioset_top5": [
            {
                "index": index,
                "mid": _POLICY.index_to_mid[index],
                "label": _POLICY.label_of(_POLICY.index_to_mid[index]),
                "prob": scores[index],
            }
            for index in top_indices
        ]
    }


def _c2_stage() -> dict:
    return _stage(
        evidence_contract_version=EVIDENCE_CONTRACT_VERSION,
        score_contract=SCORE_CONTRACT,
        class_count=CLASSIFIER_LABEL_COUNT,
        label_space_sha256=_POLICY.label_space_sha256,
    )


def _decision_for(c1_branch: str, score_by_label: dict[str, float]):
    labels = _c2_labels(score_by_label)
    c2 = _c2_stage()
    decision = evaluate_noise_admission(
        c1_intelligibility=c1_branch,
        evidence=parse_c2_evidence(labels, _POLICY, c2),
        policy=_POLICY,
    )
    return labels, c2, decision


def _valid_record(
    record_id: str,
    dataset_id: str,
    category_name: str,
    *,
    audio_path: str = "/audio/clip.wav",
    sr: int = 16000,
    duration: object = 1.0,
    c1_branch: str = "noise",
    balance_category: str | None = None,
    policy_sha: str = CANONICAL_POLICY_SHA,
    artifacts: dict | None = None,
    execution_mode: dict | None = None,
    publishable: bool = True,
    speech_veto: tuple[str | None, float | None] = (None, None),
    background_speech: tuple[str | None, float | None] = (None, None),
) -> dict:
    """构造 top5-only C2 证据，并由共享 C3 evaluator 生成准入字段。"""
    if c1_branch not in {"noise", "babble"}:
        raise AssertionError(f"unsupported C1 branch: {c1_branch!r}")
    score_by_label: dict[str, float] = {}
    if c1_branch == "noise":
        score_by_label[balance_category or category_name] = 1.0
    else:
        support_label, support_score = background_speech
        score_by_label[support_label or "Hubbub, speech noise, speech babble"] = (
            1.0 if support_score is None else support_score
        )
        if balance_category is not None:
            score_by_label[balance_category] = 1.0
    if speech_veto[0] is not None:
        score_by_label[speech_veto[0]] = speech_veto[1]
    labels, c2, decision = _decision_for(c1_branch, score_by_label)
    admission_extra = decision_to_admission_extra(
        decision,
        execution_mode=execution_mode or {"c1": "production", "c2": "subprocess_daemon"},
        publishable=publishable,
        artifact_fingerprints=dict(artifacts or FAKE_ARTIFACTS),
    )
    admission_extra["admission_policy_sha256"] = policy_sha
    return {
        "record_id": record_id,
        "dataset_id": dataset_id,
        "audio": {"path": audio_path, "sr": sr, "duration": duration},
        "labels": labels,
        "stages": {
            "C1": _stage(intelligibility=c1_branch),
            "C2": c2,
            "C3": _stage(**admission_extra),
            "D2": _stage(),
        },
    }


def _expected_output(
    record_id: str,
    dataset_id: str,
    category_name: str,
    *,
    audio_path: str = "/audio/clip.wav",
    sr: int = 16000,
    length: int = 16000,
    c1_branch: str = "noise",
    balance_category: str | None = None,
    category_id: int = 0,
) -> dict:
    scores = (
        {balance_category or category_name: 1.0}
        if c1_branch == "noise"
        else {
            "Hubbub, speech noise, speech babble": 1.0,
            **({balance_category: 1.0} if balance_category is not None else {}),
        }
    )
    _, _, admission = _decision_for(c1_branch, scores)
    return {
        "uid": record_id,
        "fs": sr,
        "length": length,
        "path": audio_path,
        "manifest_schema_version": NOISE_MANIFEST_SCHEMA_VERSION,
        "balance_category_mid": admission.balance_category_mid,
        "balance_category": admission.balance_category,
        "balance_category_source": admission.balance_category_source,
        "category_name": admission.balance_category,
        "category_id": category_id,
        "admission_allowed": True,
        "admission_decision": admission.decision,
        "admission_reason_codes": list(admission.reason_codes),
        "admission_policy_version": ADMISSION_POLICY_VERSION,
        "admission_policy_sha256": CANONICAL_POLICY_SHA,
        "verification": {
            "c1_status": "passed",
            "c1_intelligibility": c1_branch,
            "c2_status": "passed",
            "c3_status": "passed",
            "d2_status": "passed",
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "score_contract": SCORE_CONTRACT,
            "class_count": CLASSIFIER_LABEL_COUNT,
            "label_space_sha256": _POLICY.label_space_sha256,
            "artifact_fingerprints": dict(FAKE_ARTIFACTS),
            "top5_hits": [
                {
                    "index": hit.index,
                    "mid": hit.mid,
                    "label": hit.label,
                    "kind": hit.kind,
                    "prob": hit.prob,
                }
                for hit in admission.top5_hits
            ],
        },
        "dataset_id": dataset_id,
    }


def _fixture_audio_path(noise_root: Path, record_id: str, original_path: str) -> Path:
    return noise_root / "final" / "audio" / f"{record_id}-{Path(original_path).name}"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    output_records = json.loads(json.dumps(records))
    if path.parent.name == "manifests" and path.parent.parent.name == "final":
        noise_root = path.parent.parent.parent
        for record in output_records:
            audio = record.get("audio") if isinstance(record, dict) else None
            record_id = record.get("record_id") if isinstance(record, dict) else None
            original_path = audio.get("path") if isinstance(audio, dict) else None
            if isinstance(record_id, str) and isinstance(original_path, str) and original_path:
                fixture_path = _fixture_audio_path(noise_root, record_id, original_path)
                fixture_path.parent.mkdir(parents=True, exist_ok=True)
                fixture_path.write_bytes(b"audio fixture")
                audio["path"] = str(fixture_path)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in output_records),
        encoding="utf-8",
    )


def _inject_publish_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    **callbacks,
) -> None:
    """仅测试显式注入 path callback；生产 facade 不传这些参数。"""
    from core.data import simulation_manifest_transaction, simulation_utils

    def publish_with_callbacks(
        strict_temp_path: Path,
        output: Path,
        vocab_temp_path: Path,
        vocab_output: Path,
    ) -> None:
        simulation_manifest_transaction.publish_output_pair(
            strict_temp_path,
            output,
            vocab_temp_path,
            vocab_output,
            **callbacks,
        )

    monkeypatch.setattr(simulation_utils, "_publish_output_pair", publish_with_callbacks)


def _materialize_writer(
    noise_root: Path,
    source: Path,
    output: Path,
    vocab: Path,
    first_publish_done,
    resume_publish,
    result_queue,
    lock_attempted=None,
) -> None:
    """子进程 writer；可在首个目标 replace 后暂停以构造确定性交错。"""
    from core.data import simulation_manifest_transaction, simulation_utils

    real_replace = os.replace
    real_output_lock = simulation_utils._exclusive_output_set_lock
    publish_calls = 0

    @contextmanager
    def controlled_output_lock(targets):
        if lock_attempted is not None:
            lock_attempted.set()
        with real_output_lock(targets):
            yield

    def controlled_replace(source_path, target_path):
        nonlocal publish_calls
        result = real_replace(source_path, target_path)
        if Path(source_path).suffix == ".tmp":
            publish_calls += 1
            if publish_calls == 1:
                first_publish_done.set()
                if resume_publish is not None and not resume_publish.wait(timeout=10):
                    raise TimeoutError("timed out waiting to resume materialize publish")
        return result

    def controlled_publish(
        strict_temp_path: Path,
        output_path: Path,
        vocab_temp_path: Path,
        vocab_output: Path,
    ) -> None:
        simulation_manifest_transaction.publish_output_pair(
            strict_temp_path,
            output_path,
            vocab_temp_path,
            vocab_output,
            replace=controlled_replace,
        )

    simulation_utils._publish_output_pair = controlled_publish
    simulation_utils._exclusive_output_set_lock = controlled_output_lock
    try:
        materialize_final_noise_manifest(noise_root, [source], output, vocab)
    except Exception as exc:
        result_queue.put(("error", repr(exc)))
    else:
        result_queue.put(("ok", None))


def test_noise_manifest_adapter_reuses_one_admission_context(tmp_path, monkeypatch):
    from core.data.noise_admission import NoiseAdmissionContext

    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain")])
    context = NoiseAdmissionContext.load()
    load_count = 0

    def load_once(cls):
        nonlocal load_count
        load_count += 1
        return context

    monkeypatch.setattr(NoiseAdmissionContext, "load", classmethod(load_once))
    materialize_final_noise_manifest(
        noise_root,
        [source],
        tmp_path / "noise.jsonl",
        tmp_path / "vocab.json",
    )
    assert load_count == 1


def test_noise_manifest_adapter_stabilizes_vocab_and_roundtrips_strict_jsonl(tmp_path):
    noise_root = tmp_path / "noise"
    alpha = _final_manifest_path(noise_root, "alpha")
    zeta = _final_manifest_path(noise_root, "zeta")
    _write_jsonl(
        alpha,
        [
            _valid_record("rec-2", "alpha", "Rain", audio_path="/audio/rain.wav", duration=1.00003125),
            _valid_record("rec-1", "alpha", "Animal", audio_path="/audio/animal.wav", duration=0.5),
        ],
    )
    _write_jsonl(zeta, [_valid_record("rec-3", "zeta", "Bird", audio_path="/audio/bird.wav", sr=8000, duration=0.25)])
    output = tmp_path / "training" / "noise.jsonl"
    vocab_path = tmp_path / "training" / "noise_categories.json"

    vocab = materialize_final_noise_manifest(
        noise_root=noise_root,
        final_manifest_paths=[zeta, alpha],
        output_path=output,
        category_vocab_path=vocab_path,
    )

    assert vocab == {"Animal": 0, "Bird": 1, "Rain": 2}
    assert json.loads(vocab_path.read_text(encoding="utf-8")) == vocab
    by_fs, uids, flat, lengths, metadata = read_source_scp(output, return_metadata=True)
    assert uids[16000] == ["rec-2", "rec-1"]
    bird_path = str(_fixture_audio_path(noise_root, "rec-3", "/audio/bird.wav"))
    animal_path = str(_fixture_audio_path(noise_root, "rec-1", "/audio/animal.wav"))
    rain_path = str(_fixture_audio_path(noise_root, "rec-2", "/audio/rain.wav"))
    assert by_fs[8000] == {"rec-3": bird_path}
    assert flat["rec-1"] == animal_path
    # duration * sr is rounded with decimal ROUND_HALF_UP, never truncated.
    assert lengths == {"rec-2": 16001, "rec-1": 8000, "rec-3": 2000}
    assert metadata["rec-2"] == _expected_output(
        "rec-2", "alpha", "Rain", audio_path=rain_path, length=16001, category_id=2
    )


def test_noise_manifest_adapter_vocab_is_independent_of_input_file_order(tmp_path):
    noise_root = tmp_path / "noise"
    first = _final_manifest_path(noise_root, "first")
    second = _final_manifest_path(noise_root, "second")
    _write_jsonl(first, [_valid_record("one", "first", "Bird")])
    _write_jsonl(second, [_valid_record("two", "second", "Rain")])

    first_vocab = materialize_final_noise_manifest(
        noise_root=noise_root,
        final_manifest_paths=[first, second],
        output_path=tmp_path / "one.jsonl",
        category_vocab_path=tmp_path / "one-vocab.json",
    )
    second_vocab = materialize_final_noise_manifest(
        noise_root=noise_root,
        final_manifest_paths=[second, first],
        output_path=tmp_path / "two.jsonl",
        category_vocab_path=tmp_path / "two-vocab.json",
    )

    assert first_vocab == second_vocab == {"Bird": 0, "Rain": 1}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record["labels"].pop("audioset_top5"),
        lambda record: record["labels"]["audioset_top5"].pop(),
        lambda record: record["labels"]["audioset_top5"][0].__setitem__("prob", True),
        lambda record: record["labels"].__setitem__("audioset_scores", []),
        lambda record: record["audio"].__setitem__("sr", True),
        lambda record: record["audio"].__setitem__("duration", float("nan")),
        lambda record: record["audio"].__setitem__("path", ""),
    ],
)
def test_noise_manifest_adapter_rejects_invalid_final_audiorecord_fields_without_overwrite(tmp_path, mutate):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    record = _valid_record("stable-id", "set-a", "Rain")
    mutate(record)
    _write_jsonl(source, [record])
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old output\n", encoding="utf-8")
    vocab.write_text('{"old": 0}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert output.read_text(encoding="utf-8") == "old output\n"
    assert vocab.read_text(encoding="utf-8") == '{"old": 0}\n'
    assert not list(tmp_path.glob(".*.tmp"))


def test_noise_manifest_adapter_rejects_empty_final_manifest_without_overwrite(tmp_path):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    source.write_bytes(b"")
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old output\n", encoding="utf-8")
    vocab.write_text('{"old": 0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"set-a\.jsonl: final manifest contains no records"):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert output.read_text(encoding="utf-8") == "old output\n"
    assert vocab.read_text(encoding="utf-8") == '{"old": 0}\n'
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_noise_manifest_adapter_second_pass_failure_preserves_existing_targets(tmp_path, monkeypatch):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain")])
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old output\n", encoding="utf-8")
    vocab.write_text('{"old": 0}\n', encoding="utf-8")

    from core.data import simulation_utils

    real_builder = simulation_utils.build_validated_noise_record
    calls = 0

    def fail_second_pass(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("simulated second-pass failure")
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(simulation_utils, "build_validated_noise_record", fail_second_pass)
    with pytest.raises(ValueError, match="simulated second-pass failure"):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert output.read_text(encoding="utf-8") == "old output\n"
    assert vocab.read_text(encoding="utf-8") == '{"old": 0}\n'
    assert not list(tmp_path.glob(".*.tmp"))


def test_noise_manifest_adapter_production_publish_uses_dirfd_defaults(
    tmp_path,
    monkeypatch,
):
    from core.data import simulation_manifest_transaction

    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain")])
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    real_publish = simulation_manifest_transaction.publish_staged_files
    callback_kwargs = []

    def record_publish(staged, targets, **kwargs):
        callback_kwargs.append(kwargs)
        return real_publish(staged, targets, **kwargs)

    monkeypatch.setattr(
        simulation_manifest_transaction,
        "publish_staged_files",
        record_publish,
    )
    materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert callback_kwargs == [{}]
    assert output.is_file() and vocab.is_file()


def test_noise_manifest_adapter_rolls_back_both_targets_when_second_publish_replace_fails(tmp_path, monkeypatch):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain")])
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    old_output = "old output\n"
    old_vocab = '{"old": 0}\n'
    output.write_text(old_output, encoding="utf-8")
    vocab.write_text(old_vocab, encoding="utf-8")

    real_replace = os.replace
    publish_calls = 0

    def fail_second_publish(source_path, target_path):
        nonlocal publish_calls
        if Path(source_path).suffix == ".tmp":
            publish_calls += 1
            if publish_calls == 2:
                raise OSError("simulated second publish failure")
        return real_replace(source_path, target_path)

    _inject_publish_callbacks(monkeypatch, replace=fail_second_publish)
    with pytest.raises(OSError, match="simulated second publish failure"):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert output.read_text(encoding="utf-8") == old_output
    assert vocab.read_text(encoding="utf-8") == old_vocab
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_noise_manifest_adapter_serializes_writers_for_the_same_output_pair(tmp_path):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires fork to inject a process-local publish interleaving")

    noise_root = tmp_path / "noise"
    source_a = _final_manifest_path(noise_root, "writer-a")
    source_b = _final_manifest_path(noise_root, "writer-b")
    _write_jsonl(source_a, [_valid_record("from-a", "writer-a", "Rain")])
    _write_jsonl(source_b, [_valid_record("from-b", "writer-b", "Bird")])
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old output\n", encoding="utf-8")
    vocab.write_text('{"old":0}\n', encoding="utf-8")

    context = multiprocessing.get_context("fork")
    writer_a_published_one = context.Event()
    resume_writer_a = context.Event()
    writer_b_published_one = context.Event()
    writer_b_lock_attempted = context.Event()
    result_queue = context.Queue()
    writer_a = context.Process(
        target=_materialize_writer,
        args=(noise_root, source_a, output, vocab, writer_a_published_one, resume_writer_a, result_queue),
    )
    writer_b = context.Process(
        target=_materialize_writer,
        args=(
            noise_root,
            source_b,
            output,
            vocab,
            writer_b_published_one,
            None,
            result_queue,
            writer_b_lock_attempted,
        ),
    )

    writer_b_entered_publish_while_a_paused = False
    try:
        writer_a.start()
        assert writer_a_published_one.wait(timeout=10), "writer A did not reach its first publish"
        writer_b.start()
        assert writer_b_lock_attempted.wait(timeout=10), "writer B did not reach the output lock"
        writer_b_entered_publish_while_a_paused = writer_b_published_one.wait(timeout=1)
        resume_writer_a.set()
        writer_a.join(timeout=10)
        writer_b.join(timeout=10)
    finally:
        resume_writer_a.set()
        for writer in (writer_a, writer_b):
            if writer.is_alive():
                writer.terminate()
            if writer.pid is not None:
                writer.join(timeout=10)

    assert writer_a.exitcode == 0
    assert writer_b.exitcode == 0
    assert sorted(result_queue.get(timeout=2) for _ in range(2)) == [("ok", None), ("ok", None)]
    assert not writer_b_entered_publish_while_a_paused
    assert json.loads(output.read_text(encoding="utf-8"))["uid"] == "from-b"
    assert json.loads(vocab.read_text(encoding="utf-8")) == {"Bird": 0}
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_noise_manifest_adapter_rolls_back_and_fsyncs_when_commit_directory_fsync_fails(
    tmp_path,
    monkeypatch,
):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain")])
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    old_output = "old output\n"
    old_vocab = '{"old":0}\n'
    output.write_text(old_output, encoding="utf-8")
    vocab.write_text(old_vocab, encoding="utf-8")

    directory_fsync_calls = 0

    def fail_first_directory_fsync(targets):
        nonlocal directory_fsync_calls
        directory_fsync_calls += 1
        if directory_fsync_calls == 1:
            raise OSError("simulated commit directory fsync failure")
        for directory in {Path(target).parent for target in targets}:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    _inject_publish_callbacks(
        monkeypatch,
        fsync_directories=fail_first_directory_fsync,
    )
    with pytest.raises(OSError, match="simulated commit directory fsync failure"):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert directory_fsync_calls == 2
    assert output.read_text(encoding="utf-8") == old_output
    assert vocab.read_text(encoding="utf-8") == old_vocab
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_noise_manifest_adapter_retains_backup_when_rollback_fails(tmp_path, monkeypatch):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain")])
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    old_output = "old output\n"
    old_vocab = '{"old": 0}\n'
    output.write_text(old_output, encoding="utf-8")
    vocab.write_text(old_vocab, encoding="utf-8")

    real_replace = os.replace
    publish_calls = 0

    def fail_publish_and_output_rollback(source_path, target_path):
        nonlocal publish_calls
        source_path = Path(source_path)
        target_path = Path(target_path)
        if source_path.suffix == ".tmp":
            publish_calls += 1
            if publish_calls == 2:
                raise OSError("simulated second publish failure")
        if source_path.suffix == ".bak" and target_path == output:
            raise OSError("simulated output rollback failure")
        return real_replace(source_path, target_path)

    _inject_publish_callbacks(
        monkeypatch,
        replace=fail_publish_and_output_rollback,
    )
    with pytest.raises(RuntimeError, match="backup retained at"):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    backups = list(tmp_path.glob(".*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == old_output
    assert output.read_text(encoding="utf-8") != old_output
    assert vocab.read_text(encoding="utf-8") == old_vocab
    assert not list(tmp_path.glob(".*.tmp"))


def test_noise_manifest_adapter_cleans_created_backups_when_second_backup_fails(tmp_path, monkeypatch):
    """显式 backup callback 第二次失败时，已创建的 backup 必须被清理。"""
    from core.data import simulation_manifest_transaction

    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain")])
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    old_output = "old output\n"
    old_vocab = '{"old": 0}\n'
    output.write_text(old_output, encoding="utf-8")
    vocab.write_text(old_vocab, encoding="utf-8")

    real_backup = simulation_manifest_transaction.backup_existing_target
    backup_calls = 0

    def fail_second_backup(target):
        nonlocal backup_calls
        backup_calls += 1
        if backup_calls == 2:
            raise OSError("simulated second backup failure")
        return real_backup(target)

    _inject_publish_callbacks(monkeypatch, backup_existing=fail_second_backup)
    with pytest.raises(OSError, match="simulated second backup failure"):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    # 目标内容保持不变；tmp 与已创建的 bak 均无残留。
    assert output.read_text(encoding="utf-8") == old_output
    assert vocab.read_text(encoding="utf-8") == old_vocab
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_noise_manifest_adapter_preserves_targets_when_first_publish_replace_fails(tmp_path, monkeypatch):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain")])
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old output\n", encoding="utf-8")
    vocab.write_text('{"old": 0}\n', encoding="utf-8")

    real_replace = os.replace

    def fail_first_publish(source_path, target_path):
        if Path(source_path).suffix == ".tmp":
            raise OSError("simulated first publish failure")
        return real_replace(source_path, target_path)

    _inject_publish_callbacks(monkeypatch, replace=fail_first_publish)
    with pytest.raises(OSError, match="simulated first publish failure"):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert output.read_text(encoding="utf-8") == "old output\n"
    assert vocab.read_text(encoding="utf-8") == '{"old": 0}\n'
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


@pytest.mark.parametrize("alias_kind", ["dotdot", "symlink"])
def test_noise_manifest_adapter_rejects_physical_output_aliases_before_staging(tmp_path, alias_kind):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain")])
    output = tmp_path / "outputs" / "output.jsonl"
    output.parent.mkdir()
    if alias_kind == "dotdot":
        vocab = output.parent / "nested" / ".." / output.name
    else:
        vocab = tmp_path / "vocab-link.json"
        try:
            vocab.symlink_to(output)
        except OSError as exc:
            pytest.skip(f"symlinks unsupported: {exc}")

    with pytest.raises(ValueError, match="physical files"):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert not list(tmp_path.rglob(".*.tmp"))
    assert not list(tmp_path.rglob(".*.bak"))


def test_noise_manifest_adapter_rejects_second_pass_manifest_change_without_overwrite(tmp_path, monkeypatch):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain")])
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old output\n", encoding="utf-8")
    vocab.write_text('{"old": 0}\n', encoding="utf-8")

    from core.data import simulation_utils

    real_iter = simulation_utils._iter_final_audiorecords
    calls = 0

    def mutate_before_second_pass(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            # 第二遍读到不同的 audio 内容（其余字段一致），两遍摘要必须不一致。
            _write_jsonl(source, [_valid_record("id", "set-a", "Rain", audio_path="/audio/changed.wav")])
        yield from real_iter(path)

    monkeypatch.setattr(simulation_utils, "_iter_final_audiorecords", mutate_before_second_pass)
    with pytest.raises(ValueError, match=r"set-a\.jsonl: line 1: .*changed"):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert output.read_text(encoding="utf-8") == "old output\n"
    assert vocab.read_text(encoding="utf-8") == '{"old": 0}\n'
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_noise_manifest_adapter_rejects_second_pass_canonical_output_change(tmp_path, monkeypatch):
    from core.data import simulation_utils

    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(
        source,
        [_valid_record("id", "set-a", "Conversation", c1_branch="babble")],
    )
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old output\n", encoding="utf-8")
    vocab.write_text('{"old": 0}\n', encoding="utf-8")

    real_builder = simulation_utils.build_validated_noise_record
    calls = 0

    def change_babble_category_on_second_pass(*args, **kwargs):
        nonlocal calls
        calls += 1
        output_record = real_builder(*args, **kwargs)
        if calls == 2:
            output_record["balance_category_source"] = "generic_noise_fallback"
        return output_record

    monkeypatch.setattr(
        simulation_utils,
        "build_validated_noise_record",
        change_babble_category_on_second_pass,
    )

    with pytest.raises(ValueError, match="balance category must select the highest-ranked noise"):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert output.read_text(encoding="utf-8") == "old output\n"
    assert vocab.read_text(encoding="utf-8") == '{"old": 0}\n'
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_noise_manifest_adapter_rejects_second_pass_artifact_identity_change(tmp_path, monkeypatch):
    from core.data import simulation_utils

    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    records = [
        _valid_record("first", "set-a", "Rain"),
        _valid_record("second", "set-a", "Bird"),
    ]
    _write_jsonl(source, records)
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old output\n", encoding="utf-8")
    vocab.write_text('{"old": 0}\n', encoding="utf-8")

    real_iter = simulation_utils._iter_final_audiorecords
    calls = 0

    def replace_identity_before_second_pass(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            changed_records = [
                records[0],
                _valid_record("second", "set-a", "Bird", artifacts=OTHER_ARTIFACTS),
            ]
            for record in changed_records:
                original_path = record["audio"]["path"]
                record["audio"]["path"] = str(
                    _fixture_audio_path(noise_root, record["record_id"], original_path)
                )
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in changed_records),
                encoding="utf-8",
            )
        yield from real_iter(path)

    monkeypatch.setattr(simulation_utils, "_iter_final_audiorecords", replace_identity_before_second_pass)

    with pytest.raises(
        ValueError,
        match="admission identity changed between validation and write passes",
    ):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert output.read_text(encoding="utf-8") == "old output\n"
    assert vocab.read_text(encoding="utf-8") == '{"old": 0}\n'
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_noise_manifest_adapter_does_not_open_audio_payload_during_publication(
    tmp_path,
    monkeypatch,
):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain")])
    audio_path = Path(json.loads(source.read_text(encoding="utf-8"))["audio"]["path"])
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    real_open = Path.open

    def reject_audio_open(path, *args, **kwargs):
        if path == audio_path:
            raise AssertionError("audio payload must not be opened during publication")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_audio_open)

    materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert output.is_file()
    assert vocab.is_file()


@pytest.mark.parametrize("content", ["\n", "  \t \n"])
def test_noise_manifest_adapter_rejects_blank_final_jsonl_lines_with_location(tmp_path, content):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain")])
    with source.open("a", encoding="utf-8") as handle:
        handle.write(content)

    with pytest.raises(ValueError, match=r"set-a\.jsonl: line 2: blank JSONL line"):
        materialize_final_noise_manifest(noise_root, [source], tmp_path / "out.jsonl", tmp_path / "vocab.json")


def test_noise_manifest_adapter_uses_audioset_not_audio_path_for_category(tmp_path):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain", audio_path="/label/DefinitelyNotRain/file.wav")])

    materialize_final_noise_manifest(noise_root, [source], tmp_path / "output.jsonl", tmp_path / "vocab.json")

    record = json.loads((tmp_path / "output.jsonl").read_text(encoding="utf-8"))
    assert record["category_name"] == "Rain"
    assert record["balance_category"] == "Rain"
    assert record["category_id"] == 0


def test_noise_manifest_adapter_rejects_duplicate_record_id_and_strict_input(tmp_path):
    noise_root = tmp_path / "noise"
    first = _final_manifest_path(noise_root, "first")
    second = _final_manifest_path(noise_root, "second")
    _write_jsonl(first, [_valid_record("duplicate", "first", "Rain")])
    _write_jsonl(second, [_valid_record("duplicate", "second", "Bird")])

    with pytest.raises(ValueError, match="duplicate.*record_id"):
        materialize_final_noise_manifest(noise_root, [first, second], tmp_path / "out.jsonl", tmp_path / "vocab.json")

    _write_jsonl(first, [{"uid": "legacy", "fs": 16000, "length": 16000, "path": "/a.wav"}])
    with pytest.raises(ValueError, match="record_id"):
        materialize_final_noise_manifest(noise_root, [first], tmp_path / "out.jsonl", tmp_path / "vocab.json")


@pytest.mark.parametrize(
"candidate",
[
    lambda root: root / "final" / "manifests",
    lambda root: root / "final" / "manifests" / "set-a.json",
    lambda root: root / "other" / "set-a.jsonl",
    lambda root: root / "final" / "manifests" / "missing.jsonl",
],
)
def test_noise_manifest_adapter_rejects_paths_outside_explicit_final_layout(tmp_path, candidate):
    noise_root = tmp_path / "noise"
    _final_manifest_path(noise_root, "set-a")
    with pytest.raises(ValueError):
        materialize_final_noise_manifest(
            noise_root, [candidate(noise_root)], tmp_path / "out.jsonl", tmp_path / "vocab.json"
        )


def test_noise_manifest_adapter_rejects_dataset_id_that_disagrees_with_manifest_filename(tmp_path):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "other-set", "Rain")])

    with pytest.raises(ValueError, match="dataset_id.*filename"):
        materialize_final_noise_manifest(noise_root, [source], tmp_path / "out.jsonl", tmp_path / "vocab.json")


def test_noise_manifest_adapter_creates_output_parents_and_leaves_no_temp_files(tmp_path):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain")])
    output = tmp_path / "new" / "nested" / "out.jsonl"
    vocab = tmp_path / "new" / "nested" / "vocab.json"

    materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert output.is_file() and vocab.is_file()
    assert not list(output.parent.glob(".*.tmp"))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record["stages"].pop("C1"),
        lambda record: record["stages"].__setitem__("C1", _stage("rejected", intelligibility="noise")),
        lambda record: record["stages"]["C1"].__setitem__("intelligibility", "speech"),
        lambda record: record["stages"].pop("C2"),
        lambda record: record["stages"].pop("C3"),
        lambda record: record["stages"].__setitem__("C3", _stage("rejected")),
        lambda record: record["stages"]["C3"].__setitem__("admission_allowed", False),
        lambda record: record["stages"]["C3"].__setitem__("content_kind", "speech"),
        lambda record: record["stages"]["C3"].pop("execution_mode"),
        lambda record: record["stages"]["C3"].pop("publishable"),
        lambda record: record["stages"].pop("D2"),
    ],
)
def test_noise_manifest_adapter_rejects_missing_or_non_passed_stage_without_overwrite(tmp_path, mutate):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    record = _valid_record("id", "set-a", "Rain")
    mutate(record)
    _write_jsonl(source, [record])
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old output\n", encoding="utf-8")
    vocab.write_text('{"old": 0}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert output.read_text(encoding="utf-8") == "old output\n"
    assert vocab.read_text(encoding="utf-8") == '{"old": 0}\n'
    assert not list(tmp_path.glob(".*.tmp"))


def test_noise_manifest_adapter_rejects_dry_run_with_valid_artifact_hashes(tmp_path):
    """materialize 不得发布带合法 hash 的 C2 dry-run Stage C 证据。"""
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(
        source,
        [
            _valid_record(
                "id",
                "set-a",
                "Rain",
                execution_mode={"c1": "production", "c2": "dry_run"},
                publishable=False,
            )
        ],
    )
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old output\n", encoding="utf-8")
    vocab.write_text('{"old": 0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Stage C execution is not production-publishable"):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert output.read_text(encoding="utf-8") == "old output\n"
    assert vocab.read_text(encoding="utf-8") == '{"old": 0}\n'


@pytest.mark.parametrize(
    "record",
    [
        lambda: _valid_record(
            "id",
            "set-a",
            "Conversation",
            c1_branch="babble",
            speech_veto=("Babbling", 1.0),
        ),
        lambda: _valid_record(
            "id",
            "set-a",
            "Conversation",
            c1_branch="babble",
            speech_veto=("Singing", 1.0),
        ),
    ],
)
def test_noise_manifest_adapter_rejects_babble_speech_veto_evidence(tmp_path, record):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [record()])

    with pytest.raises(ValueError):
        materialize_final_noise_manifest(noise_root, [source], tmp_path / "out.jsonl", tmp_path / "vocab.json")


def test_noise_manifest_adapter_rejects_policy_hash_mismatch_without_overwrite(tmp_path):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain", policy_sha="0" * 64)])
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old output\n", encoding="utf-8")
    vocab.write_text('{"old": 0}\n', encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="admission_policy_sha256.*does not match recomputed value",
    ):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert output.read_text(encoding="utf-8") == "old output\n"
    assert vocab.read_text(encoding="utf-8") == '{"old": 0}\n'


def test_noise_manifest_adapter_rejects_voice_policy_fingerprint_mismatch_without_publish(tmp_path):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    artifacts = {**FAKE_ARTIFACTS, "voice_policy_sha256": "0" * 64}
    _write_jsonl(source, [_valid_record("id", "set-a", "Rain", artifacts=artifacts)])
    output = tmp_path / "output.jsonl"
    vocab = tmp_path / "vocab.json"
    output.write_text("old output\n", encoding="utf-8")
    vocab.write_text('{"old": 0}\n', encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            r"stages\.C3\.artifact_fingerprints\.voice_policy_sha256 "
            r"must equal current policy\.sha256"
        ),
    ):
        materialize_final_noise_manifest(noise_root, [source], output, vocab)

    assert output.read_text(encoding="utf-8") == "old output\n"
    assert vocab.read_text(encoding="utf-8") == '{"old": 0}\n'
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_noise_manifest_adapter_rejects_cross_manifest_artifact_identity_inconsistency(tmp_path):
    noise_root = tmp_path / "noise"
    first = _final_manifest_path(noise_root, "first")
    second = _final_manifest_path(noise_root, "second")
    _write_jsonl(first, [_valid_record("one", "first", "Rain")])
    _write_jsonl(second, [_valid_record("two", "second", "Bird", artifacts=OTHER_ARTIFACTS)])

    with pytest.raises(
        ValueError,
        match="noise admission identity differs from other final manifests",
    ):
        materialize_final_noise_manifest(
            noise_root, [first, second], tmp_path / "out.jsonl", tmp_path / "vocab.json"
        )


def test_noise_manifest_adapter_output_fields_match_v6_contract_exactly(tmp_path):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(source, [_valid_record("id", "set-a", "Traffic noise, roadway noise")])

    materialize_final_noise_manifest(noise_root, [source], tmp_path / "output.jsonl", tmp_path / "vocab.json")

    record = json.loads((tmp_path / "output.jsonl").read_text(encoding="utf-8"))
    assert record == _expected_output(
        "id",
        "set-a",
        "Traffic noise, roadway noise",
        audio_path=str(_fixture_audio_path(noise_root, "id", "/audio/clip.wav")),
    )
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
    }
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


def test_noise_manifest_adapter_babble_uses_highest_ranked_noise_hit(tmp_path):
    """Babble 不建独立 bucket，category 仍取 top5 最高排名 noise MID。"""
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(
        source,
        [
            _valid_record("babble-fallback", "set-a", "Conversation", c1_branch="babble"),
            _valid_record(
                "babble-music",
                "set-a",
                "Chatter",
                c1_branch="babble",
                balance_category="Music",
            ),
            _valid_record("noise", "set-a", "Rain"),
        ],
    )

    vocab = materialize_final_noise_manifest(
        noise_root, [source], tmp_path / "out.jsonl", tmp_path / "vocab.json"
    )

    assert vocab == {"Music": 0, "Noise": 1, "Rain": 2}
    records = {
        record["uid"]: record
        for record in (
            json.loads(line)
            for line in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    assert records["babble-fallback"] == _expected_output(
        "babble-fallback",
        "set-a",
        "Conversation",
        audio_path=str(_fixture_audio_path(noise_root, "babble-fallback", "/audio/clip.wav")),
        c1_branch="babble",
        category_id=1,
    )
    assert records["babble-music"] == _expected_output(
        "babble-music",
        "set-a",
        "Chatter",
        audio_path=str(_fixture_audio_path(noise_root, "babble-music", "/audio/clip.wav")),
        c1_branch="babble",
        balance_category="Music",
        category_id=0,
    )
    assert records["noise"] == _expected_output(
        "noise",
        "set-a",
        "Rain",
        audio_path=str(_fixture_audio_path(noise_root, "noise", "/audio/clip.wav")),
        category_id=2,
    )
    assert records["babble-fallback"]["balance_category"] == "Noise"
    assert records["babble-fallback"]["balance_category_source"] == "noise_top5_hit"
    assert records["babble-music"]["balance_category_source"] == "noise_top5_hit"


def test_noise_manifest_adapter_accepts_babble_music_and_wind(tmp_path):
    noise_root = tmp_path / "noise"
    source = _final_manifest_path(noise_root, "set-a")
    _write_jsonl(
        source,
        [
            _valid_record("babble", "set-a", "Conversation", c1_branch="babble"),
            _valid_record("music", "set-a", "Music"),
            _valid_record("wind", "set-a", "Wind"),
        ],
    )
    output = tmp_path / "output.jsonl"

    vocab = materialize_final_noise_manifest(
        noise_root, [source], output, tmp_path / "vocab.json"
    )

    assert vocab == {"Music": 0, "Noise": 1, "Wind": 2}
    records = {
        record["uid"]: record
        for record in (json.loads(line) for line in output.read_text(encoding="utf-8").splitlines())
    }
    assert records["babble"]["balance_category"] == "Noise"
    assert records["music"]["balance_category"] == "Music"
    assert records["wind"]["balance_category"] == "Wind"


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda record: record.__setitem__("unexpected", True), "unknown=.*unexpected"),
        (
            lambda record: record["verification"]["artifact_fingerprints"].__setitem__(
                "unexpected_sha256", "f" * 64
            ),
            "artifact_fingerprints.*unknown=.*unexpected_sha256",
        ),
    ],
)
def test_noise_manifest_runtime_rejects_unknown_v6_fields(mutate, match):
    record = _expected_output("id", "set-a", "Rain")
    mutate(record)

    with pytest.raises(ValueError, match=match):
        validate_noise_manifest_records({"id": record}, source_path="noise.jsonl")


def test_noise_manifest_runtime_rejects_schema_v4_and_c4_verification():
    record = _expected_output("id", "set-a", "Rain")
    record["manifest_schema_version"] = 4
    record["verification"]["c4_status"] = record["verification"].pop("c3_status")

    with pytest.raises(ValueError, match="manifest_schema_version must be exact int 6"):
        validate_noise_manifest_records({"id": record}, source_path="noise.jsonl")

    record["manifest_schema_version"] = 6
    with pytest.raises(ValueError, match=r"missing=.*c3_status.*unknown=.*c4_status"):
        validate_noise_manifest_records({"id": record}, source_path="noise.jsonl")


def test_noise_manifest_runtime_rejects_voice_policy_fingerprint_mismatch():
    record = _expected_output("id", "set-a", "Rain")
    record["verification"]["artifact_fingerprints"]["voice_policy_sha256"] = "0" * 64

    with pytest.raises(
        ValueError,
        match=(
            r"verification\.artifact_fingerprints\.voice_policy_sha256 "
            r"must equal admission_policy_sha256"
        ),
    ):
        validate_noise_manifest_records({"id": record}, source_path="noise.jsonl")


@pytest.mark.parametrize("dataset_id", [None, "", " set-a ", 123])
def test_noise_manifest_runtime_requires_nonempty_trimmed_string_dataset_id(dataset_id):
    record = _expected_output("id", "set-a", "Rain")
    if dataset_id is None:
        record.pop("dataset_id")
    else:
        record["dataset_id"] = dataset_id

    with pytest.raises(ValueError, match="dataset_id must be a non-empty trimmed string"):
        validate_noise_manifest_records({"id": record}, source_path="noise.jsonl")


def test_noise_manifest_runtime_rejects_category_name_mapping_to_multiple_ids():
    first = _expected_output("first", "set-a", "Rain", category_id=1)
    second = _expected_output("second", "set-a", "Rain", category_id=2)

    with pytest.raises(ValueError, match="category name maps to multiple category_ids"):
        validate_noise_manifest_records(
            {"first": first, "second": second}, source_path="noise.jsonl"
        )


@pytest.mark.parametrize(
    "category_name, roles, c1_branch",
    [
        ("Rain", ["noise"], "noise"),
        ("Music", ["music"], "noise"),
        ("Wind", ["wind"], "noise"),
        ("Noise", ["noise"], "babble"),
    ],
)
def test_noise_manifest_runtime_accepts_canonical_category_roles(
    category_name, roles, c1_branch
):
    record = _expected_output(
        "id",
        "set-a",
        category_name,
        c1_branch=c1_branch,
        balance_category=category_name,
    )
    record["roles"] = roles

    validate_noise_manifest_records({"id": record}, source_path="noise.jsonl")


@pytest.mark.parametrize("roles", [["wind"], ["noise", "wind"], "wind", None])
def test_noise_manifest_runtime_rejects_noncanonical_roles(roles):
    record = _expected_output("id", "set-a", "Rain")
    record["roles"] = roles

    with pytest.raises(ValueError, match="roles must equal canonical roles"):
        validate_noise_manifest_records({"id": record}, source_path="noise.jsonl")


def test_manifest_cache_reuses_unchanged_source(tmp_path, monkeypatch):
    from core.data import simulation_utils

    source = tmp_path / "noise.jsonl"
    _write_jsonl(source, [{"uid": "old", "fs": 16000, "length": 1, "path": "/a.wav"}])
    expected = read_source_scp(source, return_metadata=True)

    def fail_if_parsed(*args, **kwargs):
        raise AssertionError("unchanged manifest should use its cache")

    monkeypatch.setattr(simulation_utils, "_read_source_manifest_impl", fail_if_parsed)

    assert read_source_scp(source, return_metadata=True) == expected


def test_manifest_cache_is_versioned_json_with_validation_mode(tmp_path):
    from core.data import simulation_utils

    source = tmp_path / "noise.jsonl"
    _write_jsonl(source, [{"uid": "id", "fs": 16000, "length": 1, "path": "/a.wav"}])

    read_source_scp(source, return_metadata=True)

    cache_path = simulation_utils._manifest_cache_path(source)
    cache_obj = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache_path.suffix == ".json"
    assert cache_obj["_cache_version"] == "6"
    assert cache_obj["_validation_mode"] == "base"


def test_training_manifest_does_not_run_admission_and_reuses_base_cache(
    tmp_path,
    monkeypatch,
):
    from core.data import noise_admission, simulation_utils

    source = tmp_path / "strict.jsonl"
    record = _expected_output("id", "set-a", "Rain")
    _write_jsonl(source, [record])

    def unexpected_validation(*args, **kwargs):
        raise AssertionError("training runtime must not replay production admission")

    monkeypatch.setattr(
        noise_admission,
        "validate_noise_manifest_records",
        unexpected_validation,
    )

    first = simulation_utils.read_training_noise_manifest(source)
    second = simulation_utils.read_training_noise_manifest(source)

    assert first == second
    assert first.capabilities.strict_fields_complete is False
    cache_obj = json.loads(
        simulation_utils._manifest_cache_path(source).read_text(encoding="utf-8")
    )
    assert cache_obj["_validation_mode"] == "training-base-v1"


def test_manifest_cache_does_not_bind_pre_replacement_parse_to_post_replacement_identity(tmp_path, monkeypatch):
    from core.data import simulation_utils

    source = tmp_path / "noise.jsonl"
    original = {"uid": "old", "fs": 16000, "length": 1, "path": "/a.wav"}
    replacement = {"uid": "new", "fs": 16000, "length": 1, "path": "/b.wav"}
    _write_jsonl(source, [original])
    real_parser = simulation_utils._read_source_manifest_impl
    parse_calls = 0

    def replace_source_after_parse(path, return_metadata=True):
        nonlocal parse_calls
        parse_calls += 1
        result = real_parser(path, return_metadata=return_metadata)
        if parse_calls == 1:
            staged = tmp_path / "replacement.jsonl"
            _write_jsonl(staged, [replacement])
            os.replace(staged, source)
        return result

    monkeypatch.setattr(simulation_utils, "_read_source_manifest_impl", replace_source_after_parse)

    _, _, _, _, first_metadata = read_source_scp(source, return_metadata=True)
    _, _, _, _, second_metadata = read_source_scp(source, return_metadata=True)

    assert first_metadata == {"old": original}
    assert second_metadata == {"new": replacement}
    assert parse_calls == 2


def test_manifest_cache_invalidates_same_size_replacement_with_restored_mtime(tmp_path):
    source = tmp_path / "noise.jsonl"
    original = {"uid": "old", "fs": 16000, "length": 1, "path": "/a.wav"}
    replacement = {"uid": "new", "fs": 16000, "length": 1, "path": "/b.wav"}
    _write_jsonl(source, [original])
    original_stat = source.stat()
    _, _, _, _, original_metadata = read_source_scp(source, return_metadata=True)
    assert original_metadata == {"old": original}

    staged = tmp_path / "replacement.jsonl"
    _write_jsonl(staged, [replacement])
    os.utime(staged, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    os.replace(staged, source)
    replacement_stat = source.stat()
    assert replacement_stat.st_size == original_stat.st_size
    assert replacement_stat.st_mtime_ns == original_stat.st_mtime_ns
    assert replacement_stat.st_ino != original_stat.st_ino

    _, _, _, _, replacement_metadata = read_source_scp(source, return_metadata=True)

    assert replacement_metadata == {"new": replacement}


def test_manifest_cache_invalidates_legacy_identity_without_ctime(tmp_path, monkeypatch):
    from core.data import simulation_utils

    source = tmp_path / "noise.jsonl"
    record = {"uid": "id", "fs": 16000, "length": 1, "path": "/a.wav"}
    _write_jsonl(source, [record])
    read_source_scp(source, return_metadata=True)
    cache_obj = simulation_utils._manifest_cache_raw_load(
        simulation_utils._manifest_cache_path(source)
    )
    cache_obj.pop("_source_ctime_ns", None)
    monkeypatch.setattr(simulation_utils, "_manifest_cache_raw_load", lambda _: cache_obj)
    real_parser = simulation_utils._read_source_manifest_impl
    parse_calls = 0

    def count_parse_calls(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return real_parser(*args, **kwargs)

    monkeypatch.setattr(simulation_utils, "_read_source_manifest_impl", count_parse_calls)

    read_source_scp(source, return_metadata=True)

    assert parse_calls == 1


def test_sha256_path_observes_same_path_content_changes(tmp_path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"before")
    before = sha256_path(asset)

    asset.write_bytes(b"after")

    assert sha256_path(asset) != before


def test_sha256_path_rejects_symlink_entries(tmp_path):
    asset = tmp_path / "asset"
    asset.mkdir()
    target = asset / "target.bin"
    target.write_bytes(b"target")
    alias = asset / "alias.bin"
    try:
        alias.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unsupported: {exc}")

    with pytest.raises(ValueError, match="symlink"):
        sha256_path(asset)


def test_sha256_path_rejects_special_entries(tmp_path):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "regular.bin").write_bytes(b"regular")
    special = asset / "fifo"
    try:
        os.mkfifo(special)
    except (AttributeError, NotImplementedError, OSError) as exc:
        pytest.skip(f"special files unsupported: {exc}")

    with pytest.raises(ValueError, match="non-regular"):
        sha256_path(asset)


@pytest.mark.parametrize("metadata", [None, [], "metadata", 1])
def test_noise_manifest_runtime_rejects_non_mapping_metadata(metadata):
    with pytest.raises(ValueError, match="metadata must be an object"):
        validate_noise_manifest_records(metadata, source_path="noise.jsonl")


@pytest.mark.parametrize("metadata_key", [None, 1, " id "])
def test_noise_manifest_runtime_rejects_invalid_metadata_keys(metadata_key):
    record = _expected_output("id", "set-a", "Rain")

    with pytest.raises(ValueError, match="metadata key must be a non-empty trimmed string"):
        validate_noise_manifest_records({metadata_key: record}, source_path="noise.jsonl")


@pytest.mark.parametrize("record_uid", [None, "", " id ", 1])
def test_noise_manifest_runtime_requires_record_uid_type(record_uid):
    record = _expected_output("id", "set-a", "Rain")
    if record_uid is None:
        record.pop("uid")
    else:
        record["uid"] = record_uid

    with pytest.raises(ValueError, match="uid must be a non-empty trimmed string"):
        validate_noise_manifest_records({"id": record}, source_path="noise.jsonl")


def test_noise_manifest_runtime_requires_uid_to_match_metadata_key():
    record = _expected_output("id", "set-a", "Rain")
    record["uid"] = "other-id"

    with pytest.raises(ValueError, match="uid must equal metadata key"):
        validate_noise_manifest_records({"id": record}, source_path="noise.jsonl")


def test_noise_source_parser_unifies_jsonl_and_scp(tmp_path):
    from core.data.noise_manifest_adapter import read_noise_source_manifest

    record = {
        "uid": "noise one",
        "fs": 16000,
        "length": 32000,
        "path": "/audio/noise one.wav",
        "category_id": 3,
        "category_name": "Rain",
    }
    jsonl_path = tmp_path / "noise.jsonl"
    _write_jsonl(jsonl_path, [record])
    scp_path = tmp_path / "noise.scp"
    scp_path.write_text(
        "'noise one' 16000 32000 '/audio/noise one.wav'\n", encoding="utf-8"
    )
    json_path = tmp_path / "noise.json"
    json_path.write_text(json.dumps([record]), encoding="utf-8")

    parsed_jsonl = read_noise_source_manifest(jsonl_path)
    parsed_scp = read_noise_source_manifest(scp_path)

    for parsed in (parsed_jsonl, parsed_scp):
        assert parsed.parser_version == 10
        assert parsed.by_fs == {16000: {"noise one": "/audio/noise one.wav"}}
        assert parsed.uids == {16000: ["noise one"]}
        assert parsed.lengths == {"noise one": 32000}
    assert parsed_jsonl.capabilities.has_metadata is True
    assert parsed_jsonl.capabilities.category_complete is True
    assert parsed_jsonl.capabilities.strict_fields_complete is False
    assert parsed_scp.metadata == {}
    assert parsed_scp.capabilities.has_metadata is False
    assert parsed_scp.capabilities.category_complete is False
    with pytest.raises(ValueError, match="must use .scp .* or .jsonl"):
        read_noise_source_manifest(json_path)


@pytest.mark.parametrize(
    "legacy_field", sorted(LEGACY_C2_LABEL_FIELDS | LEGACY_C3_FIELDS)
)
@pytest.mark.parametrize("container", ["record", "labels"])
def test_noise_source_parser_keeps_base_reading_separate_from_strict_admission(
    tmp_path, legacy_field, container
):
    from core.data.noise_manifest_adapter import read_noise_source_manifest

    record = _expected_output("id", "set-a", "Rain")
    if container == "record":
        record[legacy_field] = True
    else:
        record["labels"] = {"original": "allowed", legacy_field: True}
    source = tmp_path / f"legacy-{container}-{legacy_field}.jsonl"
    _write_jsonl(source, [record])

    parsed = read_noise_source_manifest(source)

    assert parsed.capabilities.strict_fields_complete is False


def test_noise_source_parser_does_not_probe_strict_capability_by_default(tmp_path):
    from core.data.noise_manifest_adapter import read_noise_source_manifest

    record = _expected_output("id", "set-a", "Rain")
    record["labels"] = {"original": "upstream-source-label"}
    source = tmp_path / "strict-with-original.jsonl"
    _write_jsonl(source, [record])

    parsed = read_noise_source_manifest(source)

    assert parsed.capabilities.strict_fields_complete is False


def test_noise_source_parser_rejects_schema_v5_strict_capability(tmp_path):
    from core.data.noise_manifest_adapter import read_noise_source_manifest

    record = _expected_output("id", "set-a", "Rain")
    record["manifest_schema_version"] = 5
    source = tmp_path / "schema-v5.jsonl"
    _write_jsonl(source, [record])

    parsed = read_noise_source_manifest(source)

    assert parsed.capabilities.strict_fields_complete is False


def test_noise_source_json_array_is_rejected_by_suffix(tmp_path):
    from core.data.noise_manifest_adapter import read_noise_source_manifest

    source = tmp_path / "noise.json"
    source.write_text('[{"uid":"id"}]', encoding="utf-8")

    with pytest.raises(ValueError, match="must use .scp .* or .jsonl"):
        read_noise_source_manifest(source)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("uid", " id ", "uid must be a non-empty trimmed string"),
        ("path", " /audio/noise.wav ", "path must be a non-empty trimmed string"),
        ("fs", True, "fs must be an exact integer > 0"),
        ("fs", 0, "fs must be an exact integer > 0"),
        ("fs", 16000.0, "fs must be an exact integer > 0"),
        ("length", False, "length must be an exact integer > 0"),
        ("length", 0, "length must be an exact integer > 0"),
        ("length", 1.5, "length must be an exact integer > 0"),
    ],
)
def test_noise_source_jsonl_rejects_noncanonical_base_fields(
    tmp_path, field, value, message
):
    from core.data.noise_manifest_adapter import read_noise_source_manifest

    record = {"uid": "id", "fs": 16000, "length": 16000, "path": "/audio/noise.wav"}
    record[field] = value
    source = tmp_path / "noise.jsonl"
    _write_jsonl(source, [record])

    with pytest.raises(ValueError, match=message):
        read_noise_source_manifest(source)


def test_noise_source_parser_rejects_duplicate_uid_and_blank_jsonl_line(tmp_path):
    from core.data.noise_manifest_adapter import read_noise_source_manifest

    record = {"uid": "id", "fs": 16000, "length": 16000, "path": "/audio/noise.wav"}
    duplicate = tmp_path / "duplicate.jsonl"
    _write_jsonl(duplicate, [record, record])
    with pytest.raises(ValueError, match="duplicate uid"):
        read_noise_source_manifest(duplicate)

    blank = tmp_path / "blank.jsonl"
    blank.write_text(json.dumps(record) + "\n\n" + json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="blank JSONL line"):
        read_noise_source_manifest(blank)


@pytest.mark.parametrize("category_name", ["Human speech", "Speech"])
def test_noise_source_parser_does_not_apply_admission_label_filter(tmp_path, category_name):
    from core.data.noise_manifest_adapter import read_noise_source_manifest

    record = {
        "uid": "legacy",
        "fs": 16000,
        "length": 16000,
        "path": "/audio/legacy.wav",
        "category_id": 1,
        "category_name": category_name,
    }
    source = tmp_path / "legacy.jsonl"
    _write_jsonl(source, [record])

    parsed = read_noise_source_manifest(source)
    assert parsed.metadata["legacy"]["category_name"] == category_name
    with pytest.raises(ValueError, match="manifest_schema_version"):
        read_noise_source_manifest(source, production_admission=True)


def test_noise_source_snapshot_detects_manifest_path_replacement(tmp_path):
    from core.data.noise_manifest_adapter import (
        assert_file_identity,
        read_file_bytes_snapshot,
    )

    manifest = tmp_path / "noise.json"
    manifest.write_bytes(b"[]")
    snapshot = read_file_bytes_snapshot(manifest)
    assert snapshot.data == b"[]"
    assert len(snapshot.sha256) == 64
    manifest.write_bytes(b"[ ]")
    with pytest.raises(ValueError, match="file identity changed"):
        assert_file_identity(manifest, snapshot.identity)


def test_noise_source_jsonl_streams_without_whole_file_bytes_snapshot(
    tmp_path,
    monkeypatch,
):
    from core.data import noise_manifest_adapter

    source = tmp_path / "large.jsonl"
    records = [
        {"uid": f"id-{index}", "fs": 16000, "length": 1, "path": f"/{index}.wav"}
        for index in range(2000)
    ]
    _write_jsonl(source, records)

    def fail_bytes_snapshot(*args, **kwargs):
        raise AssertionError("JSONL parser must not read the whole file as bytes")

    monkeypatch.setattr(
        noise_manifest_adapter,
        "read_file_bytes_snapshot",
        fail_bytes_snapshot,
    )

    parsed = noise_manifest_adapter.read_noise_source_manifest(source)

    assert len(parsed.flat) == 2000
    assert parsed.flat["id-1999"] == "/1999.wav"


def test_noise_admission_context_is_reused_by_validator_and_core_builder(monkeypatch):
    from core.data.noise_admission import (
        NoiseAdmissionContext,
        build_validated_noise_record,
    )

    context = NoiseAdmissionContext.load()

    def unexpected_load(cls):
        raise AssertionError("context dependency files were reloaded")

    monkeypatch.setattr(NoiseAdmissionContext, "load", classmethod(unexpected_load))
    validate_noise_manifest_records(
        {"id": _expected_output("id", "set-a", "Rain")},
        source_path="noise.jsonl",
        context=context,
    )
    output = build_validated_noise_record(
        _valid_record("built", "set-a", "Rain", duration=0.00009375),
        location="source.jsonl:1",
        context=context,
    )
    assert output["uid"] == "built"
    assert output["length"] == 2
    assert output["admission_reason_codes"] == ["admitted_noise"]


def test_noise_source_parser_rejects_empty_production_manifest(tmp_path):
    suffix = ".jsonl"
    payload = ""
    from core.data.noise_manifest_adapter import read_noise_source_manifest

    source = tmp_path / f"empty{suffix}"
    source.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="production admission manifest must not be empty"):
        read_noise_source_manifest(source, production_admission=True)


def test_noise_source_parser_rejects_duplicate_object_keys_with_location(tmp_path):
    suffix = ".jsonl"
    payload = (
        '{"uid":"id","fs":16000,"length":1,"path":"/a.wav",'
        '"nested":{"kind":"noise","kind":"speech"}}\n'
    )
    message = r"line 1: duplicate JSON object key 'kind' at \$\['nested'\]"
    from core.data.noise_manifest_adapter import read_noise_source_manifest

    source = tmp_path / f"duplicate-key{suffix}"
    source.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_noise_source_manifest(source)


def test_strict_fields_complete_requires_actual_single_record_validation(tmp_path):
    from core.data.noise_manifest_adapter import read_noise_source_manifest

    record = _expected_output("id", "set-a", "Rain")
    record["admission_reason_codes"] = ["forged"]
    source = tmp_path / "forged-strict.jsonl"
    _write_jsonl(source, [record])

    parsed = read_noise_source_manifest(source)

    assert parsed.capabilities.strict_fields_complete is False
    with pytest.raises(ValueError, match="admission_reason_codes"):
        read_noise_source_manifest(source, production_admission=True)


def test_core_builder_preserves_labels_original_and_strict_validator_accepts_it():
    from core.data.noise_admission import build_validated_noise_record

    source_record = _valid_record("id", "set-a", "Rain")
    source_record["labels"]["original"] = {"source": "upstream"}

    output = build_validated_noise_record(source_record, location="source.jsonl: line 1")

    assert output["labels"] == {"original": {"source": "upstream"}}
    output["category_id"] = 0
    validate_noise_manifest_records({"id": output}, source_path="noise.jsonl")


def test_canonical_json_bytes_rejects_nonfinite_numbers():
    from core.data.noise_manifest_adapter import canonical_json_bytes

    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_bytes({"weight": float("nan")})


def test_noise_scp_double_quotes_keep_posix_backslashes_and_quoted_provenance(tmp_path):
    from core.data.noise_manifest_adapter import read_noise_source_manifest

    source = tmp_path / "quoted.scp"
    source.write_text(
        '"uid\\q" 16000 2 "123\\q path.wav"\n'
        'plain 16000 1 "123"\n',
        encoding="utf-8",
    )

    parsed = read_noise_source_manifest(source)

    assert parsed.flat["uid\\q"] == str(tmp_path / "123\\q path.wav")
    assert parsed.flat["plain"] == str(tmp_path / "123")
    assert parsed.lengths == {"uid\\q": 2, "plain": 1}


def test_noise_scp_rejects_bare_numeric_path(tmp_path):
    from core.data.noise_manifest_adapter import read_noise_source_manifest

    numeric_path = tmp_path / "numeric-path.scp"
    numeric_path.write_text("uid 16000 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected exactly four fields"):
        read_noise_source_manifest(numeric_path)


def test_noise_source_formats_reject_unreasonable_fs_and_length(tmp_path):
    from core.data.noise_manifest_adapter import read_noise_source_manifest

    for suffix, payload in (
        (".jsonl", '{"uid":"u","fs":768001,"length":1,"path":"a.wav"}\n'),
        (".jsonl", '{"uid":"u","fs":16000,"length":9007199254740992,"path":"a.wav"}\n'),
        (".scp", "u 768001 1 path.wav\n"),
    ):
        source = tmp_path / f"invalid{suffix}"
        source.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError, match="fs|length|expected uid"):
            read_noise_source_manifest(source)


pytestmark = [pytest.mark.unit]