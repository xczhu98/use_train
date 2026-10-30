"""v3 原子 artifact 目录的契约与篡改检测测试。"""

import gc
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from pathlib import Path

import pytest

import audio_filter_pipeline.utils.artifacts as artifacts_module
from audio_filter_pipeline.utils.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactIndex,
    ArtifactReader,
    ArtifactWriter,
    PipelineStep,
    drop_artifact,
    is_production_bound_artifact_index,
    iter_artifact_rows,
    open_artifact_for_read,
    open_artifact_reader,
)

pytestmark = pytest.mark.unit


CONFIG_DIGEST = "1" * 64
LINEAGE_DIGEST = "2" * 64
CONTRACT_FILES = {"rows.jsonl", "index.json", "_SUCCESS"}


def inspect_artifact_unbound(*args, **kwargs):
    return artifacts_module.inspect_artifact_unbound(*args, **kwargs)


def _artifact_dir(root, step="step1", dataset_id="dataset"):
    from audio_filter_pipeline.utils.path_utils import v3_artifact_dir

    return Path(v3_artifact_dir(str(root), step, dataset_id))


def _publish(root, rows=(), step="step1", dataset_id="dataset"):
    writer = ArtifactWriter(root, step, dataset_id, CONFIG_DIGEST, LINEAGE_DIGEST)
    for row in rows:
        writer.append(row)
    writer.finalize()
    return _open_bound(root, step, dataset_id)


def _open_bound(
    root,
    step="step1",
    dataset_id="dataset",
    *,
    expected_config_digest=CONFIG_DIGEST,
    expected_lineage_digest=LINEAGE_DIGEST,
):
    return open_artifact_for_read(
        root,
        step,
        dataset_id,
        expected_config_digest=expected_config_digest,
        expected_lineage_digest=expected_lineage_digest,
    )


def _private_staging_path(writer):
    return Path(os.readlink(f"/proc/self/fd/{writer._staging_fd}"))


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _snapshot_tree(root):
    snapshot = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        snapshot[relative] = ("dir", None) if path.is_dir() else ("file", path.read_bytes())
    return snapshot


def _rewrite_index(artifact_dir, **changes):
    index_path = artifact_dir / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload.update(changes)
    without_digest = {key: value for key, value in payload.items() if key != "artifact_digest"}
    payload["artifact_digest"] = hashlib.sha256(_canonical(without_digest)).hexdigest()
    index_path.write_bytes(_canonical(payload) + b"\n")
    (artifact_dir / "_SUCCESS").write_text(payload["artifact_digest"] + "\n", encoding="ascii")


def test_roundtrip_empty_artifact_has_exact_contract(tmp_path):
    index = _publish(tmp_path, step=PipelineStep.STEP0, dataset_id="数据 集")
    artifact_dir = _artifact_dir(tmp_path, "step0", "数据 集")

    assert ARTIFACT_SCHEMA_VERSION == 2
    assert artifact_dir.name == "%E6%95%B0%E6%8D%AE%20%E9%9B%86"
    assert {path.name for path in artifact_dir.iterdir()} == CONTRACT_FILES
    assert (artifact_dir / "rows.jsonl").read_bytes() == b""
    assert index.row_count == 0
    assert list(iter_artifact_rows(index)) == []


def test_nonempty_roundtrip_uses_deterministic_canonical_bytes(tmp_path):
    rows = [{"z": 1, "text": "中文", "nested": {"b": False, "a": None}}, {"n": 2.5}]
    index = _publish(tmp_path, rows)
    artifact_dir = _artifact_dir(tmp_path)
    expected_rows = (
        b'{"nested":{"a":null,"b":false},"text":"\xe4\xb8\xad\xe6\x96\x87","z":1}\n'
        b'{"n":2.5}\n'
    )
    expected_without_digest = {
        "config_digest": CONFIG_DIGEST,
        "dataset_id": "dataset",
        "layout_version": 3,
        "lineage_digest": LINEAGE_DIGEST,
        "row_count": 2,
        "rows_sha256": hashlib.sha256(expected_rows).hexdigest(),
        "schema_version": 2,
        "step": "step1",
    }
    expected_artifact_digest = hashlib.sha256(_canonical(expected_without_digest)).hexdigest()
    expected_index = dict(expected_without_digest, artifact_digest=expected_artifact_digest)

    assert (artifact_dir / "rows.jsonl").read_bytes() == expected_rows
    assert (artifact_dir / "index.json").read_bytes() == _canonical(expected_index) + b"\n"
    assert (artifact_dir / "_SUCCESS").read_bytes() == (expected_artifact_digest + "\n").encode()
    assert list(iter_artifact_rows(index)) == rows
    assert index.artifact_digest == expected_artifact_digest
    assert index.rows_path == str(artifact_dir / "rows.jsonl")


def test_independent_writes_produce_identical_contract_bytes(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _publish(first, [{"b": 2, "a": 1}])
    _publish(second, [{"a": 1, "b": 2}])

    for name in CONTRACT_FILES:
        assert (_artifact_dir(first) / name).read_bytes() == (_artifact_dir(second) / name).read_bytes()


@pytest.mark.parametrize("bad_row", [None, [1], "row", 3])
def test_append_rejects_non_mapping_rows(tmp_path, bad_row):
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    try:
        with pytest.raises(TypeError, match="mapping"):
            writer.append(bad_row)
    finally:
        writer.abort()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), object()])
def test_append_rejects_non_json_or_non_finite_values(tmp_path, bad_value):
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    try:
        with pytest.raises((TypeError, ValueError), match="JSON|finite"):
            writer.append({"value": bad_value})
    finally:
        writer.abort()


def test_finalize_publishes_verified_private_source_inode(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    final_dir = _artifact_dir(tmp_path)
    source_stat = os.fstat(writer._temp_fd)
    real_rename = artifacts._atomic_rename_noreplace_between
    observed = {}

    def checking_rename(source_fd, source_name, target_fd, target_name):
        observed["target"] = target_name
        assert not final_dir.exists()
        assert source_name.startswith(".artifact.sealed-")
        assert target_name == final_dir.name
        source = os.open(source_name, os.O_RDONLY, dir_fd=source_fd)
        try:
            assert set(os.listdir(source)) == CONTRACT_FILES
            assert (os.fstat(source).st_dev, os.fstat(source).st_ino) == (
                source_stat.st_dev,
                source_stat.st_ino,
            )
        finally:
            os.close(source)
        real_rename(source_fd, source_name, target_fd, target_name)

    monkeypatch.setattr(artifacts, "_atomic_rename_noreplace_between", checking_rename)
    writer.append({"id": 1})
    writer.finalize()

    final_stat = final_dir.stat()
    assert observed["target"] == final_dir.name
    assert (final_stat.st_dev, final_stat.st_ino) == (source_stat.st_dev, source_stat.st_ino)
    assert {path.name for path in final_dir.iterdir()} == CONTRACT_FILES


def test_abort_and_context_exception_remove_only_private_temp(tmp_path):
    sibling = tmp_path / "step1/sibling"
    sibling.mkdir(parents=True)
    (sibling / "keep").write_text("safe", encoding="utf-8")
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    temp_dir = _private_staging_path(writer)
    writer.abort()
    assert not temp_dir.exists()
    assert (sibling / "keep").read_text(encoding="utf-8") == "safe"

    with pytest.raises(RuntimeError, match="boom"):
        with ArtifactWriter(tmp_path, "step2", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST) as active:
            temp_dir = _private_staging_path(active)
            active.append({"id": 1})
            raise RuntimeError("boom")
    assert not temp_dir.exists()
    assert not _artifact_dir(tmp_path, "step2").exists()


def test_forgotten_writer_warns_and_converges_staging_and_fds(tmp_path):
    baseline_fds = len(os.listdir("/proc/self/fd"))
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    staging = _private_staging_path(writer)
    writer.append({"id": 1})

    with pytest.warns(ResourceWarning, match="unclosed ArtifactWriter"):
        del writer
        gc.collect()

    assert not staging.exists()
    assert not _artifact_dir(tmp_path).exists()
    assert len(os.listdir("/proc/self/fd")) == baseline_fds


def test_writer_destructor_never_reports_unraisable_exception(tmp_path, monkeypatch):
    baseline_fds = len(os.listdir("/proc/self/fd"))
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    staging = _private_staging_path(writer)
    observed = []
    cleanup_calls = 0
    real_cleanup = ArtifactWriter._cleanup_unpublished

    def record_unraisable(details):
        observed.append(details)

    def fail_cleanup_once(self):
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise OSError("injected destructor cleanup failure")
        return real_cleanup(self)

    monkeypatch.setattr(sys, "unraisablehook", record_unraisable)
    monkeypatch.setattr(ArtifactWriter, "_cleanup_unpublished", fail_cleanup_once)

    with pytest.warns(ResourceWarning, match="unclosed ArtifactWriter"):
        del writer
        gc.collect()

    assert observed == []
    assert cleanup_calls == 2
    assert not staging.exists()
    assert len(os.listdir("/proc/self/fd")) == baseline_fds


def test_existing_target_is_never_overwritten(tmp_path):
    target = _artifact_dir(tmp_path)
    target.mkdir(parents=True)
    sentinel = target / "partial"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not list(target.parent.glob(".dataset.stage-*"))


def test_finalize_atomically_refuses_target_created_after_writer_start(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    target = _artifact_dir(tmp_path)
    real_rename = artifacts._atomic_rename_noreplace_between

    def racing_rename(source_fd, source_name, target_fd, target_name):
        target.mkdir()
        (target / "competitor").write_text("keep", encoding="utf-8")
        real_rename(source_fd, source_name, target_fd, target_name)

    monkeypatch.setattr(artifacts, "_atomic_rename_noreplace_between", racing_rename)
    with pytest.raises(FileExistsError):
        writer.finalize()
    assert (target / "competitor").read_text(encoding="utf-8") == "keep"
    assert not list(target.parent.glob(".dataset.stage-*"))


@pytest.mark.parametrize("terminal_action", ["finalize", "abort"])
def test_writer_is_single_use(tmp_path, terminal_action):
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    getattr(writer, terminal_action)()

    with pytest.raises(RuntimeError, match="finalized|aborted"):
        writer.append({"id": 1})
    with pytest.raises(RuntimeError, match="finalized|aborted"):
        writer.finalize()


@pytest.mark.parametrize("missing", ["_SUCCESS", "index.json", "rows.jsonl"])
def test_read_rejects_missing_contract_file(tmp_path, missing):
    _publish(tmp_path)
    (_artifact_dir(tmp_path) / missing).unlink()

    with pytest.raises(ValueError, match="missing"):
        _open_bound(tmp_path, "step1", "dataset")


@pytest.mark.parametrize("raw", [b"not json", b"[]", b"null", b'{"schema_version":NaN}'])
def test_read_rejects_malformed_or_non_object_index(tmp_path, raw):
    _publish(tmp_path)
    (_artifact_dir(tmp_path) / "index.json").write_bytes(raw)

    with pytest.raises(ValueError, match="index"):
        _open_bound(tmp_path, "step1", "dataset")


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("schema_version", 3, "schema"),
        ("layout_version", 2, "layout"),
        ("step", "step2", "step"),
        ("dataset_id", "other", "dataset"),
        ("config_digest", "ABC", "config_digest"),
        ("lineage_digest", "g" * 64, "lineage_digest"),
        ("rows_sha256", "0" * 63, "rows_sha256"),
        ("artifact_digest", "0" * 63, "artifact_digest"),
        ("row_count", True, "row_count"),
        ("row_count", -1, "row_count"),
    ],
)
def test_read_rejects_index_contract_mismatch(tmp_path, field, value, error):
    _publish(tmp_path)
    artifact_dir = _artifact_dir(tmp_path)
    _rewrite_index(artifact_dir, **{field: value})
    if field == "artifact_digest":
        payload = json.loads((artifact_dir / "index.json").read_text(encoding="utf-8"))
        payload[field] = value
        (artifact_dir / "index.json").write_bytes(_canonical(payload) + b"\n")
        (artifact_dir / "_SUCCESS").write_text(value + "\n", encoding="ascii")

    with pytest.raises(ValueError, match=error):
        _open_bound(tmp_path, "step1", "dataset")


def test_schema1_container_is_rejected_by_schema2_reader(tmp_path):
    """schema-1 容器即使 marker/digest 一致也必须被 schema-2 reader 拒绝。

    把合法 index 改写为 ``schema_version: 1`` 并一致重算 digest/marker，opener
    必须在任何行读取前 fail closed（不做迁移/重写/自动覆盖）。
    """
    _publish(tmp_path)
    artifact_dir = _artifact_dir(tmp_path)
    payload = json.loads((artifact_dir / "index.json").read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    without_digest = {
        key: value for key, value in payload.items() if key != "artifact_digest"
    }
    payload["artifact_digest"] = hashlib.sha256(_canonical(without_digest)).hexdigest()
    (artifact_dir / "index.json").write_bytes(_canonical(payload) + b"\n")
    (artifact_dir / "_SUCCESS").write_text(
        payload["artifact_digest"] + "\n", encoding="ascii"
    )

    with pytest.raises(ValueError, match="schema"):
        _open_bound(tmp_path, "step1", "dataset")
    # 拒绝必须是无副作用的：contract 文件保持原样，不被改写或删除
    assert json.loads((artifact_dir / "index.json").read_text(encoding="utf-8"))[
        "schema_version"
    ] == 1


def test_read_rejects_marker_and_artifact_digest_mismatch(tmp_path):
    _publish(tmp_path)
    artifact_dir = _artifact_dir(tmp_path)
    (artifact_dir / "_SUCCESS").write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="marker"):
        _open_bound(tmp_path, "step1", "dataset")

    payload = json.loads((artifact_dir / "index.json").read_text(encoding="utf-8"))
    payload["artifact_digest"] = "0" * 64
    (artifact_dir / "index.json").write_bytes(_canonical(payload) + b"\n")
    (artifact_dir / "_SUCCESS").write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="artifact digest"):
        _open_bound(tmp_path, "step1", "dataset")


def test_read_checks_expected_config_and_lineage(tmp_path):
    _publish(tmp_path)

    with pytest.raises(ValueError, match="config digest"):
        _open_bound(
            tmp_path, "step1", "dataset", expected_config_digest="3" * 64,
        )
    with pytest.raises(ValueError, match="lineage digest"):
        _open_bound(
            tmp_path, "step1", "dataset", expected_lineage_digest="4" * 64,
        )


@pytest.mark.parametrize(
    "expected_config_digest,expected_lineage_digest,error",
    [
        ("3" * 64, LINEAGE_DIGEST, "config digest"),
        (CONFIG_DIGEST, "4" * 64, "lineage digest"),
    ],
)
def test_expectation_mismatch_rejects_before_rows_scan(
    tmp_path,
    monkeypatch,
    expected_config_digest,
    expected_lineage_digest,
    error,
):
    _publish(tmp_path, [{"id": 1}])

    def fail_if_rows_scanned(_descriptor):
        raise AssertionError("stale expectation must reject before rows scan")

    monkeypatch.setattr(artifacts_module, "_rows_hash_and_count_fd", fail_if_rows_scanned)
    with pytest.raises(ValueError, match=error):
        open_artifact_for_read(
            tmp_path,
            "step1",
            "dataset",
            expected_config_digest=expected_config_digest,
            expected_lineage_digest=expected_lineage_digest,
        )


def test_bound_reader_requires_both_digest_expectations(tmp_path):
    _publish(tmp_path)

    with pytest.raises(TypeError, match="expected_config_digest"):
        open_artifact_for_read(tmp_path, "step1", "dataset")
    with pytest.raises(TypeError, match="expected_lineage_digest"):
        open_artifact_for_read(
            tmp_path,
            "step1",
            "dataset",
            expected_config_digest=CONFIG_DIGEST,
        )


def test_unbound_inspection_is_explicit_and_integrity_checked(tmp_path):
    _publish(tmp_path, [{"id": 1}])

    index = inspect_artifact_unbound(tmp_path, "step1", "dataset")
    assert list(iter_artifact_rows(index)) == [{"id": 1}]

    Path(index.rows_path).write_bytes(b'{"id":2}\n')
    with pytest.raises(ValueError, match="rows hash"):
        inspect_artifact_unbound(tmp_path, "step1", "dataset")


def test_only_expectation_bound_openers_have_production_capability(tmp_path):
    _publish(tmp_path, [{"id": 1}])

    bound = _open_bound(tmp_path)
    unbound = inspect_artifact_unbound(tmp_path, "step1", "dataset")
    reader = open_artifact_reader(
        tmp_path,
        "step1",
        "dataset",
        expected_config_digest=CONFIG_DIGEST,
        expected_lineage_digest=LINEAGE_DIGEST,
    )
    try:
        assert is_production_bound_artifact_index(bound)
        assert is_production_bound_artifact_index(reader.index)
        assert not is_production_bound_artifact_index(unbound)
        assert list(iter_artifact_rows(unbound)) == [{"id": 1}]
        assert not is_production_bound_artifact_index(unbound)
    finally:
        reader.close()


def test_production_capability_cannot_be_forged_or_copied(tmp_path):
    bound = _publish(tmp_path, [{"id": 1}])
    forged = ArtifactIndex(
        **bound.serialized(),
        artifact_dir=bound.artifact_dir,
        output_root=bound.output_root,
    )

    assert "_production_bound" not in {item.name for item in fields(ArtifactIndex)}
    assert not is_production_bound_artifact_index(forged)
    assert not is_production_bound_artifact_index(replace(bound))
    vars(forged)["_production_bound"] = True
    assert not is_production_bound_artifact_index(forged)


@pytest.mark.parametrize("dataset_id", [1, True, "", "   "])
def test_artifact_public_apis_require_non_empty_exact_string_dataset_id(
    tmp_path, dataset_id,
):
    before = set(tmp_path.rglob("*"))

    with pytest.raises(ValueError, match="dataset_id.*non-empty string"):
        ArtifactWriter(
            tmp_path, "step1", dataset_id, CONFIG_DIGEST, LINEAGE_DIGEST,
        )
    with pytest.raises(ValueError, match="dataset_id.*non-empty string"):
        open_artifact_for_read(
            tmp_path,
            "step1",
            dataset_id,
            expected_config_digest=CONFIG_DIGEST,
            expected_lineage_digest=LINEAGE_DIGEST,
        )
    with pytest.raises(ValueError, match="dataset_id.*non-empty string"):
        inspect_artifact_unbound(tmp_path, "step1", dataset_id)

    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize(
    "transform",
    [
        lambda payload: json.dumps(payload, indent=2, ensure_ascii=False).encode() + b"\n",
        lambda payload: json.dumps(
            dict(reversed(list(payload.items()))),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode() + b"\n",
        lambda payload: _canonical(payload) + b"\r\n",
        lambda payload: _canonical(payload),
    ],
    ids=["pretty", "reordered", "crlf", "missing-lf"],
)
def test_open_rejects_noncanonical_index_bytes(tmp_path, transform):
    _publish(tmp_path)
    index_path = _artifact_dir(tmp_path) / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    index_path.write_bytes(transform(payload))

    with pytest.raises(ValueError, match="non-canonical index bytes"):
        _open_bound(tmp_path)


@pytest.mark.parametrize("mutation", ["pretty", "crlf", "missing-lf"])
def test_iterator_close_rejects_noncanonical_index_bytes(tmp_path, mutation):
    index = _publish(tmp_path, [{"id": 1}, {"id": 2}])
    iterator = iter_artifact_rows(index)
    assert next(iterator) == {"id": 1}
    index_path = _artifact_dir(tmp_path) / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    raw = {
        "pretty": json.dumps(payload, indent=2).encode() + b"\n",
        "crlf": _canonical(payload) + b"\r\n",
        "missing-lf": _canonical(payload),
    }[mutation]
    index_path.write_bytes(raw)

    with pytest.raises(ValueError, match="non-canonical index bytes"):
        iterator.close()


def test_removed_verified_index_boolean_helper_is_not_exposed():
    assert not hasattr(artifacts_module, "_is_verified_index")


def test_iterator_rejects_manually_constructed_index_even_with_valid_fields(tmp_path):
    index = _publish(tmp_path, [{"id": 1}])
    forged = ArtifactIndex(
        **index.serialized(),
        artifact_dir=index.artifact_dir,
        output_root=index.output_root,
    )

    with pytest.raises(ValueError, match="verified.*opener"):
        list(iter_artifact_rows(forged))


def test_verified_capability_is_not_an_artifact_index_constructor_field(tmp_path):
    index = _publish(tmp_path, [{"id": 1}])

    assert "_verification_token" not in {item.name for item in fields(ArtifactIndex)}
    with pytest.raises(TypeError, match="_verification_token"):
        ArtifactIndex(
            **index.serialized(),
            artifact_dir=index.artifact_dir,
            output_root=index.output_root,
            _verification_token=object(),
        )


def test_iterator_rejects_mutated_verified_index_identity(tmp_path):
    first = _publish(tmp_path / "first", [{"id": 1}], dataset_id="first")
    second = _publish(tmp_path / "second", [{"id": 2}], dataset_id="second")

    vars(first).update(vars(second))

    with pytest.raises(ValueError, match="verified.*opener"):
        list(iter_artifact_rows(first))


def test_iterator_rejects_dataclass_copy_of_verified_index(tmp_path):
    index = _publish(tmp_path, [{"id": 1}])
    forged = replace(index, row_count=2)

    with pytest.raises(ValueError, match="verified.*opener"):
        list(iter_artifact_rows(forged))


@pytest.mark.parametrize("mutation", ["pretty", "reordered", "crlf", "missing-lf"])
def test_iterator_snapshot_rejects_noncanonical_index_bytes(tmp_path, mutation):
    index = _publish(tmp_path, [{"id": 1}])
    index_path = _artifact_dir(tmp_path) / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    raw = {
        "pretty": json.dumps(payload, indent=2).encode() + b"\n",
        "reordered": json.dumps(
            dict(reversed(list(payload.items()))), separators=(",", ":"),
        ).encode() + b"\n",
        "crlf": _canonical(payload) + b"\r\n",
        "missing-lf": _canonical(payload),
    }[mutation]
    index_path.write_bytes(raw)

    with pytest.raises(ValueError, match="non-canonical index bytes"):
        list(iter_artifact_rows(index))


@pytest.mark.parametrize("mutation,error", [(b'{"id":2}\n', "rows hash"), (b'', "row count")])
def test_read_rejects_rows_hash_or_count_tampering(tmp_path, mutation, error):
    _publish(tmp_path, [{"id": 1}])
    artifact_dir = _artifact_dir(tmp_path)
    (artifact_dir / "rows.jsonl").write_bytes(mutation)
    if error == "row count":
        _rewrite_index(artifact_dir, rows_sha256=hashlib.sha256(mutation).hexdigest())

    with pytest.raises(ValueError, match=error):
        _open_bound(tmp_path, "step1", "dataset")


def test_read_rejects_extra_contract_file(tmp_path):
    _publish(tmp_path)
    (_artifact_dir(tmp_path) / "unexpected").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected"):
        _open_bound(tmp_path, "step1", "dataset")


@pytest.mark.parametrize("target", ["artifact", "rows", "index", "success"])
def test_read_rejects_symlinks_in_artifact_path(tmp_path, target):
    outside = tmp_path / "outside"
    outside.mkdir()
    _publish(outside)
    source = _artifact_dir(outside)
    expected = _artifact_dir(tmp_path)
    expected.parent.mkdir(parents=True)
    if target == "artifact":
        expected.symlink_to(source, target_is_directory=True)
    else:
        expected.mkdir()
        names = {"rows": "rows.jsonl", "index": "index.json", "success": "_SUCCESS"}
        symlink_name = names[target]
        for name in CONTRACT_FILES:
            destination = expected / name
            if name == symlink_name:
                destination.symlink_to(source / name)
            else:
                destination.write_bytes((source / name).read_bytes())

    with pytest.raises(ValueError, match="symlink"):
        _open_bound(tmp_path, "step1", "dataset")


def test_writer_rejects_symlinked_artifact_parent(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "out"
    root.mkdir()
    (root / "step1").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        ArtifactWriter(root, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("raw", [b"\n", b"not json\n", b"[]\n", b'{"x":NaN}\n'])
def test_open_rejects_invalid_rows_even_with_matching_index(tmp_path, raw):
    _publish(tmp_path, [{"id": 1}])
    artifact_dir = _artifact_dir(tmp_path)
    (artifact_dir / "rows.jsonl").write_bytes(raw)
    _rewrite_index(
        artifact_dir,
        rows_sha256=hashlib.sha256(raw).hexdigest(),
        row_count=raw.count(b"\n"),
    )

    with pytest.raises(ValueError, match="row"):
        _open_bound(tmp_path, "step1", "dataset")


def test_iterator_detects_content_change_during_iteration(tmp_path):
    index = _publish(tmp_path, [{"id": 1}, {"id": 2}])
    iterator = iter_artifact_rows(index)
    assert next(iterator) == {"id": 1}
    Path(index.rows_path).write_bytes(b'{"id":1}\n{"id":3}\n')

    with pytest.raises((RuntimeError, ValueError), match="changed|hash"):
        list(iterator)


def test_iterator_detects_count_change_after_open(tmp_path):
    index = _publish(tmp_path, [{"id": 1}])
    Path(index.rows_path).write_bytes(b'{"id":1}\n{"id":2}\n')

    with pytest.raises(ValueError, match="changed|count|hash"):
        list(iter_artifact_rows(index))


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("schema_version", True, "schema"),
        ("schema_version", 1.0, "schema"),
        ("layout_version", True, "layout"),
        ("layout_version", 3.0, "layout"),
    ],
)
def test_read_requires_exact_integer_types_for_versions(tmp_path, field, value, error):
    _publish(tmp_path)
    _rewrite_index(_artifact_dir(tmp_path), **{field: value})

    with pytest.raises(ValueError, match=error):
        _open_bound(tmp_path, "step1", "dataset")


@pytest.mark.parametrize(
    "row",
    [
        {1: "integer-key"},
        {"nested": {1: "integer-key"}},
        {"nested": [{"ok": 1}, {2: "integer-key"}]},
    ],
)
def test_append_rejects_non_string_json_object_keys_at_any_depth(tmp_path, row):
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    try:
        with pytest.raises(TypeError, match="key|string"):
            writer.append(row)
    finally:
        writer.abort()


def test_read_rejects_duplicate_index_key(tmp_path):
    _publish(tmp_path)
    index_path = _artifact_dir(tmp_path) / "index.json"
    raw = index_path.read_bytes().replace(
        b'"schema_version":2', b'"schema_version":2,"schema_version":2', 1,
    )
    index_path.write_bytes(raw)

    with pytest.raises(ValueError, match="duplicate|index"):
        _open_bound(tmp_path, "step1", "dataset")


def test_open_rejects_nested_duplicate_row_key(tmp_path):
    raw = b'{"nested":{"x":1,"x":2}}\n'
    _publish(tmp_path, [{"id": 1}])
    artifact_dir = _artifact_dir(tmp_path)
    (artifact_dir / "rows.jsonl").write_bytes(raw)
    _rewrite_index(
        artifact_dir,
        rows_sha256=hashlib.sha256(raw).hexdigest(),
        row_count=1,
    )

    with pytest.raises(ValueError, match="duplicate|row"):
        _open_bound(tmp_path, "step1", "dataset")


def test_iterator_revalidates_serialized_index_before_first_row(tmp_path):
    index = _publish(tmp_path, [{"id": 1}])
    _rewrite_index(_artifact_dir(tmp_path), config_digest="3" * 64)

    with pytest.raises(ValueError, match="index.*changed|changed.*index"):
        next(iter_artifact_rows(index))


def test_iterator_revalidates_index_and_marker_after_last_row(tmp_path):
    index = _publish(tmp_path, [{"id": 1}, {"id": 2}])
    iterator = iter_artifact_rows(index)
    assert next(iterator) == {"id": 1}
    _rewrite_index(_artifact_dir(tmp_path), config_digest="3" * 64)

    with pytest.raises(ValueError, match="index.*changed|marker.*changed|changed"):
        list(iterator)


def test_contract_files_are_opened_relative_to_fixed_artifact_dirfd(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    _publish(tmp_path, [{"id": 1}])
    real_open = artifacts.os.open
    contract_opens = []

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):
        if path in CONTRACT_FILES:
            contract_opens.append((path, flags, dir_fd))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "open", recording_open)
    index = _open_bound(tmp_path, "step1", "dataset")
    assert list(iter_artifact_rows(index)) == [{"id": 1}]

    assert {name for name, _flags, _dir_fd in contract_opens} == CONTRACT_FILES
    assert all(dir_fd is not None for _name, _flags, dir_fd in contract_opens)
    assert all(flags & os.O_NOFOLLOW for _name, flags, _dir_fd in contract_opens)
    assert all(flags & os.O_NONBLOCK for _name, flags, _dir_fd in contract_opens)


def test_parallel_writers_create_shared_parent_for_distinct_datasets(tmp_path):
    def publish(dataset_id):
        return _publish(tmp_path, [{"dataset": dataset_id}], dataset_id=dataset_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        indexes = list(pool.map(publish, [f"dataset-{number}" for number in range(16)]))

    assert len(indexes) == 16
    assert all(Path(index.artifact_dir).is_dir() for index in indexes)


def test_parallel_writers_same_target_publish_exactly_once(tmp_path):
    def publish(_number):
        try:
            _publish(tmp_path, [{"id": 1}])
            return "published"
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(publish, range(16)))

    assert outcomes.count("published") == 1
    assert outcomes.count("exists") == 15
    assert list(iter_artifact_rows(_open_bound(tmp_path, "step1", "dataset"))) == [
        {"id": 1}
    ]
    assert not list(_artifact_dir(tmp_path).parent.glob(".dataset.stage-*"))


class _FailingRowsProxy:
    def __init__(self, stream, *, fail_close=False, fail_flush=False):
        self._stream = stream
        self._fail_close = fail_close
        self._fail_flush = fail_flush

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def flush(self):
        if self._fail_flush:
            raise OSError("injected flush failure")
        return self._stream.flush()

    def close(self):
        self._stream.close()
        if self._fail_close:
            raise OSError("injected close failure")


def test_abort_close_failure_still_cleans_temp_and_terminates_writer(tmp_path):
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    temp_dir = _private_staging_path(writer)
    writer._rows = _FailingRowsProxy(writer._rows, fail_close=True)

    with pytest.raises(OSError, match="close failure"):
        writer.abort()

    assert not temp_dir.exists()
    assert not _artifact_dir(tmp_path).exists()
    assert writer.state == "cleanup_failed"
    assert writer.cleanup_error is not None
    with pytest.raises(RuntimeError, match="cleanup_failed"):
        writer.append({"id": 1})


def test_context_exception_is_not_replaced_by_close_failure(tmp_path):
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    temp_dir = _private_staging_path(writer)
    writer._rows = _FailingRowsProxy(writer._rows, fail_close=True)

    with pytest.raises(RuntimeError, match="context boom"):
        with writer:
            raise RuntimeError("context boom")

    assert not temp_dir.exists()
    assert writer.state == "cleanup_failed"
    assert writer.cleanup_error is not None
    with pytest.raises(RuntimeError, match="cleanup_failed"):
        writer.finalize()


def test_finalize_flush_failure_keeps_final_invisible_and_cleans_temp(tmp_path):
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    temp_dir = _private_staging_path(writer)
    writer._rows = _FailingRowsProxy(writer._rows, fail_flush=True)

    with pytest.raises(OSError, match="flush failure"):
        writer.finalize()

    assert not temp_dir.exists()
    assert not _artifact_dir(tmp_path).exists()
    with pytest.raises(RuntimeError, match="aborted"):
        writer.append({"id": 1})


def test_finalize_fsync_order(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    (tmp_path / "step1").mkdir()
    events = []
    real_fsync = artifacts.os.fsync
    real_rename = artifacts._atomic_rename_noreplace_between

    def recording_fsync(fd):
        events.append(("fsync", os.path.basename(os.readlink(f"/proc/self/fd/{fd}"))))
        return real_fsync(fd)

    def recording_rename(source_fd, source_name, target_fd, target_name):
        events.append(("rename", target_name))
        return real_rename(source_fd, source_name, target_fd, target_name)

    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    writer.append({"id": 1})
    monkeypatch.setattr(artifacts.os, "fsync", recording_fsync)
    monkeypatch.setattr(artifacts, "_atomic_rename_noreplace_between", recording_rename)
    writer.finalize()

    assert events == [
        ("fsync", "rows.jsonl"),
        ("fsync", "index.json"),
        ("fsync", "_SUCCESS"),
        ("fsync", "artifact"),
        ("rename", "dataset"),
        ("fsync", "step1"),
    ]


def test_failure_before_rename_keeps_final_invisible_and_cleans_temp(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    real_fsync = artifacts.os.fsync

    def fail_marker_fsync(fd):
        if os.path.basename(os.readlink(f"/proc/self/fd/{fd}")) == "_SUCCESS":
            raise OSError("injected marker fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(artifacts.os, "fsync", fail_marker_fsync)
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    temp_dir = _private_staging_path(writer)

    with pytest.raises(OSError, match="marker fsync failure"):
        writer.finalize()

    assert not _artifact_dir(tmp_path).exists()
    assert not temp_dir.exists()


def test_parent_fsync_failure_preserves_published_final_and_terminates_writer(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.utils.artifacts as artifacts

    real_fsync = artifacts.os.fsync

    def fail_parent_fsync(fd):
        if (
            os.path.basename(os.readlink(f"/proc/self/fd/{fd}")) == "step1"
            and _artifact_dir(tmp_path).exists()
        ):
            raise OSError("injected parent fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(artifacts.os, "fsync", fail_parent_fsync)
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)

    with pytest.raises(OSError, match="parent fsync failure"):
        writer.finalize()

    assert _artifact_dir(tmp_path).is_dir()
    assert {path.name for path in _artifact_dir(tmp_path).iterdir()} == CONTRACT_FILES
    assert list(iter_artifact_rows(_open_bound(tmp_path, "step1", "dataset"))) == []
    with pytest.raises(RuntimeError, match="publication_failed"):
        writer.append({"id": 1})
    with pytest.raises(RuntimeError, match="publication_failed"):
        writer.finalize()


def test_open_rejects_finite_syntax_that_overflows_float(tmp_path):
    raw = b'{"nested":{"value":1e999}}\n'
    _publish(tmp_path, [{"id": 1}])
    artifact_dir = _artifact_dir(tmp_path)
    (artifact_dir / "rows.jsonl").write_bytes(raw)
    _rewrite_index(artifact_dir, rows_sha256=hashlib.sha256(raw).hexdigest(), row_count=1)

    with pytest.raises(ValueError, match="finite|row"):
        _open_bound(tmp_path, "step1", "dataset")


@pytest.mark.parametrize("name", sorted(CONTRACT_FILES))
def test_read_rejects_fifo_contract_without_blocking(tmp_path, name):
    _publish(tmp_path)
    target = _artifact_dir(tmp_path) / name
    target.unlink()
    os.mkfifo(target)

    with pytest.raises(ValueError, match="regular"):
        _open_bound(tmp_path, "step1", "dataset")


def test_new_directory_chain_is_fsynced_before_writer_returns(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    root = tmp_path / "nested" / "output"
    real_fsync = artifacts.os.fsync
    fsynced = []

    def recording_fsync(fd):
        fsynced.append(os.readlink(f"/proc/self/fd/{fd}"))
        return real_fsync(fd)

    monkeypatch.setattr(artifacts.os, "fsync", recording_fsync)
    writer = ArtifactWriter(root, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    writer.abort()

    assert str(tmp_path) in fsynced
    assert str(tmp_path / "nested") in fsynced
    assert str(root) in fsynced
    # step1 是叶子目录，其目录项由 root 的 fsync 持久化；
    # 自身 fsync 会推迟到向其中写入 dataset 子目录时发生。


class _PartialWriteProxy:
    def __init__(self, stream):
        self._stream = stream

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def write(self, data):
        self._stream.write(data[:1])
        self._stream.flush()
        raise OSError("injected partial write failure")


def test_partial_append_failure_aborts_and_cleans_private_temp(tmp_path):
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    temp_dir = _private_staging_path(writer)
    writer._rows = _PartialWriteProxy(writer._rows)

    with pytest.raises(OSError, match="partial write failure"):
        writer.append({"id": 1})

    assert not temp_dir.exists()
    assert not _artifact_dir(tmp_path).exists()
    with pytest.raises(RuntimeError, match="aborted"):
        writer.finalize()


def test_descriptor_close_failure_still_attempts_all_closes(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    temp_fd = writer._temp_fd
    parent_fd = writer._parent_fd
    real_close = artifacts.os.close
    closed = []

    def failing_close(fd):
        closed.append(fd)
        real_close(fd)
        if fd == temp_fd:
            raise OSError("injected descriptor close failure")

    monkeypatch.setattr(artifacts.os, "close", failing_close)
    with pytest.raises(OSError, match="descriptor close failure"):
        writer.abort()

    assert temp_fd in closed
    assert parent_fd in closed
    assert writer._temp_fd is None
    assert writer._parent_fd is None


def test_fstat_failure_closes_new_contract_fd(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    _publish(tmp_path)
    baseline = len(os.listdir("/proc/self/fd"))
    real_fstat = artifacts.os.fstat

    def failing_fstat(fd):
        if os.path.basename(os.readlink(f"/proc/self/fd/{fd}")) == "index.json":
            raise OSError("injected fstat failure")
        return real_fstat(fd)

    monkeypatch.setattr(artifacts.os, "fstat", failing_fstat)
    with pytest.raises(OSError, match="fstat failure"):
        _open_bound(tmp_path, "step1", "dataset")

    assert len(os.listdir("/proc/self/fd")) == baseline


def test_published_parent_fsync_error_is_not_replaced_by_close_error(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    real_fsync = artifacts.os.fsync
    real_close = artifacts.os.close
    target = _artifact_dir(tmp_path)

    def failing_fsync(fd):
        if os.path.basename(os.readlink(f"/proc/self/fd/{fd}")) == "step1" and target.exists():
            raise OSError("original parent fsync failure")
        return real_fsync(fd)

    def failing_close(fd):
        path = os.readlink(f"/proc/self/fd/{fd}")
        real_close(fd)
        if os.path.basename(path).startswith(".dataset.stage-"):
            raise OSError("masking close failure")

    monkeypatch.setattr(artifacts.os, "fsync", failing_fsync)
    monkeypatch.setattr(artifacts.os, "close", failing_close)
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)

    with pytest.raises(OSError, match="original parent fsync failure"):
        writer.finalize()

    assert target.is_dir()
    with pytest.raises(RuntimeError, match="publication_failed"):
        writer.finalize()


def test_writer_does_not_expose_mutable_temp_path(tmp_path):
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    try:
        assert not hasattr(writer, "temp_dir")
    finally:
        writer.abort()


def test_finalize_rejects_extra_file_in_private_staging(tmp_path):
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    extra_fd = os.open(
        "unexpected",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=writer._temp_fd,
    )
    os.close(extra_fd)

    with pytest.raises(ValueError, match="unexpected"):
        writer.finalize()

    assert not _artifact_dir(tmp_path).exists()


def test_finalize_rejects_extra_subdirectory_in_private_staging(tmp_path):
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    os.mkdir("unexpected-dir", mode=0o700, dir_fd=writer._temp_fd)

    with pytest.raises(ValueError, match="unexpected"):
        writer.finalize()

    assert not _artifact_dir(tmp_path).exists()
    assert writer.cleanup_error is not None
    assert writer.state == "cleanup_failed"


def test_constructor_fdopen_failure_does_not_leak_descriptor(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    baseline = len(os.listdir("/proc/self/fd"))
    real_fdopen = artifacts.os.fdopen

    def failing_fdopen(fd, *args, **kwargs):
        if os.path.basename(os.readlink(f"/proc/self/fd/{fd}")) == "rows.jsonl":
            raise OSError("injected fdopen failure")
        return real_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(artifacts.os, "fdopen", failing_fdopen)
    with pytest.raises(OSError, match="fdopen failure"):
        ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)

    assert len(os.listdir("/proc/self/fd")) == baseline


def test_artifact_directory_parent_close_failure_closes_child_fd(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    _publish(tmp_path)
    baseline = len(os.listdir("/proc/self/fd"))
    real_close = artifacts.os.close
    injected = False

    def close_then_fail(fd):
        nonlocal injected
        path = os.readlink(f"/proc/self/fd/{fd}")
        real_close(fd)
        if not injected and os.path.basename(path) == "step1":
            injected = True
            raise OSError("injected parent close failure")

    monkeypatch.setattr(artifacts.os, "close", close_then_fail)
    with pytest.raises(OSError, match="parent close failure"):
        _open_bound(tmp_path, "step1", "dataset")
    monkeypatch.setattr(artifacts.os, "close", real_close)

    assert len(os.listdir("/proc/self/fd")) == baseline


@pytest.mark.parametrize(
    "raw",
    [
        b' {"id":1}\n',
        b'{"b":2,"a":1}\n',
        b'{"id":1}\r\n',
        b'{"id":1}',
        b'\n',
        b'not json\n',
        b'[]\n',
        b'{"x":NaN}\n',
    ],
)
def test_open_rejects_noncanonical_or_invalid_rows_with_matching_index(tmp_path, raw):
    _publish(tmp_path, [{"id": 1}])
    artifact_dir = _artifact_dir(tmp_path)
    (artifact_dir / "rows.jsonl").write_bytes(raw)
    _rewrite_index(
        artifact_dir,
        rows_sha256=hashlib.sha256(raw).hexdigest(),
        row_count=raw.count(b"\n"),
    )

    with pytest.raises(ValueError, match="row|canonical|newline"):
        _open_bound(tmp_path, "step1", "dataset")


def test_open_rejects_contract_entry_replaced_during_verification(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    _publish(tmp_path, [{"id": 1}])
    artifact_dir = _artifact_dir(tmp_path)
    real_read = artifacts.os.read
    replaced = False

    def replacing_read(fd, size):
        nonlocal replaced
        data = real_read(fd, size)
        if not replaced and os.path.basename(os.readlink(f"/proc/self/fd/{fd}")) == "index.json":
            replaced = True
            original = artifact_dir / "rows.jsonl"
            replacement = artifact_dir / "rows.replacement"
            replacement.write_bytes(original.read_bytes())
            os.replace(replacement, original)
        return data

    monkeypatch.setattr(artifacts.os, "read", replacing_read)
    with pytest.raises((RuntimeError, ValueError), match="changed|entry"):
        _open_bound(tmp_path, "step1", "dataset")


def test_open_bounds_marker_and_index_files(tmp_path):
    _publish(tmp_path)
    artifact_dir = _artifact_dir(tmp_path)
    (artifact_dir / "_SUCCESS").write_bytes(b"0" * 66)
    with pytest.raises(ValueError, match="marker|size"):
        _open_bound(tmp_path, "step1", "dataset")

    (artifact_dir / "index.json").write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="index|large|size"):
        _open_bound(tmp_path, "step1", "dataset")


def test_iterator_rejects_artifact_directory_renamed_during_iteration(tmp_path):
    index = _publish(tmp_path, [{"id": 1}, {"id": 2}])
    iterator = iter_artifact_rows(index)
    assert next(iterator) == {"id": 1}
    artifact_dir = _artifact_dir(tmp_path)
    moved = artifact_dir.with_name("moved")
    artifact_dir.rename(moved)

    with pytest.raises((RuntimeError, ValueError), match="changed|entry"):
        list(iterator)


def test_iterator_rejects_artifact_directory_rebound_to_symlink(tmp_path):
    index = _publish(tmp_path, [{"id": 1}, {"id": 2}])
    iterator = iter_artifact_rows(index)
    assert next(iterator) == {"id": 1}
    artifact_dir = _artifact_dir(tmp_path)
    moved = artifact_dir.with_name("moved")
    artifact_dir.rename(moved)
    artifact_dir.symlink_to(moved, target_is_directory=True)

    with pytest.raises((RuntimeError, ValueError, OSError), match="changed|entry|symlink"):
        list(iterator)


def test_iterator_rejects_forged_artifact_index_runtime_path(tmp_path):
    index = _publish(tmp_path, [{"id": 1}])
    forged = type(index)(
        **index.serialized(),
        artifact_dir=str(_artifact_dir(tmp_path).with_name("alternate")),
        output_root=index.output_root,
    )

    with pytest.raises(ValueError, match="verified.*opener"):
        list(iter_artifact_rows(forged))


def test_abort_retry_after_unlink_cleanup_failure(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    staging = _private_staging_path(writer)
    real_unlink = artifacts.os.unlink
    failed = False

    def fail_once(path, *, dir_fd=None):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected unlink failure")
        return real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "unlink", fail_once)
    with pytest.raises(OSError, match="unlink failure"):
        writer.abort()
    assert writer.state == "cleanup_failed"
    assert writer.cleanup_error is not None
    assert staging.exists()

    monkeypatch.setattr(artifacts.os, "unlink", real_unlink)
    writer.abort()
    assert writer.state == "aborted"
    assert not staging.exists()


def test_abort_retry_after_rmdir_cleanup_failure(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    staging = _private_staging_path(writer)
    real_rmdir = artifacts.os.rmdir
    failed = False

    def fail_once(path, *, dir_fd=None):
        nonlocal failed
        if not failed and path == "artifact":
            failed = True
            raise OSError("injected rmdir failure")
        return real_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "rmdir", fail_once)
    with pytest.raises(OSError, match="rmdir failure"):
        writer.abort()
    assert writer.state == "cleanup_failed"
    assert staging.exists()

    monkeypatch.setattr(artifacts.os, "rmdir", real_rmdir)
    writer.abort()
    assert writer.state == "aborted"
    assert not staging.exists()


def test_finalize_fails_closed_if_staging_parent_entry_is_replaced(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    source_stat = os.fstat(writer._temp_fd)
    staging_name = writer._staging_name
    step_parent = _artifact_dir(tmp_path).parent
    moved = step_parent / f"{staging_name}.moved"
    replacement = step_parent / staging_name
    real_rename = artifacts._atomic_rename_noreplace_between

    def replace_staging_then_publish(source_fd, source_name, target_fd, target_name):
        os.rename(staging_name, moved.name, src_dir_fd=target_fd, dst_dir_fd=target_fd)
        os.mkdir(staging_name, mode=0o700, dir_fd=target_fd)
        real_rename(source_fd, source_name, target_fd, target_name)

    monkeypatch.setattr(
        artifacts, "_atomic_rename_noreplace_between", replace_staging_then_publish,
    )
    with pytest.raises(RuntimeError, match="entry changed"):
        writer.finalize()

    final_stat = _artifact_dir(tmp_path).stat()
    assert (final_stat.st_dev, final_stat.st_ino) == (source_stat.st_dev, source_stat.st_ino)
    assert replacement.is_dir()
    assert writer.state == "publication_failed"


def test_open_rejects_same_size_in_place_change_after_rows_validation(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    _publish(tmp_path, [{"id": 1}])
    rows_path = _artifact_dir(tmp_path) / "rows.jsonl"
    real_validate = artifacts._rows_hash_and_count_fd

    def mutate_after_validation(fd):
        result = real_validate(fd)
        rows_path.write_bytes(b'{"id":2}\n')
        return result

    monkeypatch.setattr(artifacts, "_rows_hash_and_count_fd", mutate_after_validation)
    with pytest.raises((RuntimeError, ValueError), match="changed"):
        _open_bound(tmp_path, "step1", "dataset")


def test_reader_fdopen_failure_does_not_leak_descriptor(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    index = _publish(tmp_path, [{"id": 1}])
    baseline = len(os.listdir("/proc/self/fd"))
    real_fdopen = artifacts.os.fdopen

    def failing_fdopen(fd, *args, **kwargs):
        if args and args[0] == "rb":
            raise OSError("injected reader fdopen failure")
        return real_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(artifacts.os, "fdopen", failing_fdopen)
    with pytest.raises(OSError, match="reader fdopen failure"):
        list(iter_artifact_rows(index))
    monkeypatch.setattr(artifacts.os, "fdopen", real_fdopen)

    assert len(os.listdir("/proc/self/fd")) == baseline


def test_finalize_quarantines_swapped_inner_source_without_deleting_replacement(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.utils.artifacts as artifacts

    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    writer.append({"id": 1})
    staging = _private_staging_path(writer)
    step_parent = _artifact_dir(tmp_path).parent
    sibling = step_parent / "sibling"
    sibling.mkdir()
    (sibling / "keep").write_text("safe", encoding="utf-8")
    real_rename = artifacts._atomic_rename_noreplace_between
    moved_source = staging / "verified-source"
    replacement_snapshot = {}

    def swap_source_then_publish(source_fd, source_name, target_fd, target_name):
        if source_name.startswith(".artifact.sealed-"):
            os.rename(source_name, moved_source.name, src_dir_fd=source_fd, dst_dir_fd=source_fd)
            os.mkdir(source_name, mode=0o700, dir_fd=source_fd)
            replacement = staging / source_name
            for contract_name in CONTRACT_FILES:
                (replacement / contract_name).write_bytes(
                    (moved_source / contract_name).read_bytes()
                )
            (replacement / "sentinel.bin").write_bytes(b"unknown-owner-sentinel\x00\xff")
            (replacement / "nested/deeper").mkdir(parents=True)
            (replacement / "nested/deeper/payload.bin").write_bytes(b"nested-unknown-data")
            replacement_snapshot.update(_snapshot_tree(replacement))
        real_rename(source_fd, source_name, target_fd, target_name)

    monkeypatch.setattr(
        artifacts, "_atomic_rename_noreplace_between", swap_source_then_publish,
    )
    with pytest.raises(RuntimeError, match="changed|publication"):
        writer.finalize()

    final_dir = _artifact_dir(tmp_path)
    quarantines = list(step_parent.glob(".dataset.quarantine-*"))
    assert not final_dir.exists()
    with pytest.raises((ValueError, RuntimeError), match="missing|changed"):
        _open_bound(tmp_path, "step1", "dataset")
    assert len(quarantines) == 1
    assert _snapshot_tree(quarantines[0]) == replacement_snapshot
    assert moved_source.is_dir()
    assert (sibling / "keep").read_text(encoding="utf-8") == "safe"
    assert writer.state == "publication_failed"


def test_finalize_rejects_step_parent_rebind(tmp_path):
    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    writer.append({"id": 1})
    step_parent = _artifact_dir(tmp_path).parent
    moved_parent = step_parent.with_name("step1.moved")
    step_parent.rename(moved_parent)
    step_parent.mkdir()

    with pytest.raises(RuntimeError, match="entry changed"):
        writer.finalize()

    assert not _artifact_dir(tmp_path).exists()
    assert writer.state != "finalized"


def test_iterator_rejects_step_parent_rebind_during_iteration(tmp_path):
    index = _publish(tmp_path, [{"id": 1}, {"id": 2}])
    iterator = iter_artifact_rows(index)
    assert next(iterator) == {"id": 1}
    step_parent = _artifact_dir(tmp_path).parent
    moved_parent = step_parent.with_name("step1.moved")
    step_parent.rename(moved_parent)
    step_parent.mkdir()

    with pytest.raises(RuntimeError, match="entry changed"):
        list(iterator)


def test_constructor_staging_open_failure_cleans_orphan_and_fds(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    baseline = len(os.listdir("/proc/self/fd"))
    real_open_directory_at = artifacts._open_directory_at

    def fail_staging_open(parent_fd, name, *, create, sync_parent=False):
        if name.startswith(".dataset.stage-"):
            raise OSError("injected staging open failure")
        return real_open_directory_at(
            parent_fd, name, create=create, sync_parent=sync_parent,
        )

    monkeypatch.setattr(artifacts, "_open_directory_at", fail_staging_open)
    with pytest.raises(OSError, match="staging open failure"):
        ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)

    assert not list(_artifact_dir(tmp_path).parent.glob(".dataset.stage-*"))
    assert len(os.listdir("/proc/self/fd")) == baseline


def test_constructor_artifact_open_failure_cleans_orphan_and_fds(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    baseline = len(os.listdir("/proc/self/fd"))
    real_open_directory_at = artifacts._open_directory_at

    def fail_artifact_open(parent_fd, name, *, create, sync_parent=False):
        if name == "artifact":
            raise OSError("injected artifact open failure")
        return real_open_directory_at(
            parent_fd, name, create=create, sync_parent=sync_parent,
        )

    monkeypatch.setattr(artifacts, "_open_directory_at", fail_artifact_open)
    with pytest.raises(OSError, match="artifact open failure"):
        ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)

    assert not list(_artifact_dir(tmp_path).parent.glob(".dataset.stage-*"))
    assert len(os.listdir("/proc/self/fd")) == baseline


def test_abort_accepts_temp_rmdir_success_then_raise(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    staging = _private_staging_path(writer)
    real_rmdir = artifacts.os.rmdir
    injected = False

    def remove_then_raise(path, *, dir_fd=None):
        nonlocal injected
        result = real_rmdir(path, dir_fd=dir_fd)
        if not injected and path == "artifact":
            injected = True
            raise OSError("injected temp post-rmdir failure")
        return result

    monkeypatch.setattr(artifacts.os, "rmdir", remove_then_raise)
    writer.abort()

    assert writer.state == "aborted"
    assert not staging.exists()


def test_abort_accepts_staging_rmdir_success_then_raise(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    staging = _private_staging_path(writer)
    real_rmdir = artifacts.os.rmdir
    injected = False

    def remove_then_raise(path, *, dir_fd=None):
        nonlocal injected
        result = real_rmdir(path, dir_fd=dir_fd)
        if not injected and path.startswith(".dataset.stage-"):
            injected = True
            raise OSError("injected staging post-rmdir failure")
        return result

    monkeypatch.setattr(artifacts.os, "rmdir", remove_then_raise)
    writer.abort()

    assert writer.state == "aborted"
    assert not staging.exists()


def test_abort_stops_scanning_at_first_unexpected_entry(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    unexpected_fd = os.open(
        "unexpected", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600, dir_fd=writer._temp_fd,
    )
    os.close(unexpected_fd)
    real_scandir = artifacts.os.scandir
    consumed = []

    class FakeEntry:
        name = "unexpected"

    class EarlyStoppingEntries:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            consumed.append("next")
            if len(consumed) == 1:
                return FakeEntry()
            raise AssertionError("cleanup scanned past first unexpected entry")

    monkeypatch.setattr(artifacts.os, "scandir", lambda fd: EarlyStoppingEntries())
    with pytest.raises(ValueError, match="unexpected"):
        writer.abort()

    assert consumed == ["next"]
    assert writer.state == "cleanup_failed"
    assert os.stat("unexpected", dir_fd=writer._temp_fd)

    monkeypatch.setattr(artifacts.os, "scandir", real_scandir)
    os.unlink("unexpected", dir_fd=writer._temp_fd)
    writer.abort()


def test_open_integrity_error_is_not_replaced_by_close_error(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    _publish(tmp_path, [{"id": 1}])
    (_artifact_dir(tmp_path) / "rows.jsonl").write_bytes(b'{"id":2}\n')
    real_close = artifacts.os.close
    injected = False

    def close_then_raise(fd):
        nonlocal injected
        path = os.readlink(f"/proc/self/fd/{fd}")
        real_close(fd)
        if not injected and os.path.basename(path) in CONTRACT_FILES:
            injected = True
            raise OSError("injected contract close failure")

    monkeypatch.setattr(artifacts.os, "close", close_then_raise)
    with pytest.raises(ValueError, match="rows hash mismatch"):
        _open_bound(tmp_path, "step1", "dataset")


def test_iterator_rebind_error_is_not_replaced_by_close_error(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    index = _publish(tmp_path, [{"id": 1}, {"id": 2}])
    iterator = iter_artifact_rows(index)
    assert next(iterator) == {"id": 1}
    step_parent = _artifact_dir(tmp_path).parent
    step_parent.rename(step_parent.with_name("step1.moved"))
    step_parent.mkdir()
    real_close = artifacts.os.close
    injected = False

    def close_then_raise(fd):
        nonlocal injected
        path = os.readlink(f"/proc/self/fd/{fd}")
        real_close(fd)
        if not injected and os.path.basename(path) == "rows.jsonl":
            injected = True
            raise OSError("injected iterator close failure")

    monkeypatch.setattr(artifacts.os, "close", close_then_raise)
    with pytest.raises(RuntimeError, match="entry changed"):
        list(iterator)


def test_iterator_rejects_forged_serialized_index_fields(tmp_path):
    index = _publish(tmp_path, [{"id": 1}])
    forged_payload = index.serialized()
    forged_payload["row_count"] = 2
    forged = type(index)(
        **forged_payload,
        artifact_dir=index.artifact_dir,
        output_root=index.output_root,
    )

    with pytest.raises(ValueError, match="verified.*opener"):
        list(iter_artifact_rows(forged))


def test_finalize_preserves_exact_contract_replacement_in_quarantine(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.utils.artifacts as artifacts

    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    writer.append({"id": 1})
    staging = _private_staging_path(writer)
    step_parent = _artifact_dir(tmp_path).parent
    moved_source = staging / "verified-source"
    real_rename = artifacts._atomic_rename_noreplace_between
    replacement_snapshot = {}

    def swap_source_then_publish(source_fd, source_name, target_fd, target_name):
        if source_name.startswith(".artifact.sealed-"):
            os.rename(source_name, moved_source.name, src_dir_fd=source_fd, dst_dir_fd=source_fd)
            os.mkdir(source_name, mode=0o700, dir_fd=source_fd)
            replacement = staging / source_name
            for contract_name in CONTRACT_FILES:
                (replacement / contract_name).write_bytes(
                    (moved_source / contract_name).read_bytes()
                )
            replacement_snapshot.update(_snapshot_tree(replacement))
        real_rename(source_fd, source_name, target_fd, target_name)

    monkeypatch.setattr(
        artifacts, "_atomic_rename_noreplace_between", swap_source_then_publish,
    )
    with pytest.raises(RuntimeError, match="changed|publication"):
        writer.finalize()

    quarantines = list(step_parent.glob(".dataset.quarantine-*"))
    assert not _artifact_dir(tmp_path).exists()
    assert len(quarantines) == 1
    assert _snapshot_tree(quarantines[0]) == replacement_snapshot
    assert moved_source.is_dir()
    assert writer.state == "publication_failed"


def test_finalize_keeps_unknown_final_when_quarantine_rename_outcome_is_uncertain(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.utils.artifacts as artifacts

    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    writer.append({"id": 1})
    staging = _private_staging_path(writer)
    moved_source = staging / "verified-source"
    real_publish = artifacts._atomic_rename_noreplace_between
    real_renameat2 = artifacts._renameat2_noreplace

    def swap_source_then_publish(source_fd, source_name, target_fd, target_name):
        if source_name.startswith(".artifact.sealed-"):
            os.rename(source_name, moved_source.name, src_dir_fd=source_fd, dst_dir_fd=source_fd)
            os.mkdir(source_name, mode=0o700, dir_fd=source_fd)
            replacement = staging / source_name
            for contract_name in CONTRACT_FILES:
                (replacement / contract_name).write_bytes(
                    (moved_source / contract_name).read_bytes()
                )
            (replacement / "sentinel.bin").write_bytes(b"must-remain-at-final")
        real_publish(source_fd, source_name, target_fd, target_name)

    def fail_quarantine_rename(source_fd, source_name, target_fd, target_name):
        if source_name == "dataset" and target_name.startswith(".dataset.quarantine-"):
            raise OSError("injected uncertain quarantine rename")
        return real_renameat2(source_fd, source_name, target_fd, target_name)

    monkeypatch.setattr(
        artifacts, "_atomic_rename_noreplace_between", swap_source_then_publish,
    )
    monkeypatch.setattr(artifacts, "_renameat2_noreplace", fail_quarantine_rename)
    with pytest.raises(RuntimeError, match="changed|publication"):
        writer.finalize()

    final_dir = _artifact_dir(tmp_path)
    assert final_dir.is_dir()
    assert (final_dir / "sentinel.bin").read_bytes() == b"must-remain-at-final"
    assert not list(final_dir.parent.glob(".dataset.quarantine-*"))
    assert moved_source.is_dir()
    assert writer.state == "publication_failed"


def test_descriptor_bound_reader_scans_source_once_before_iteration(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.utils.artifacts as artifacts

    _publish(tmp_path, [{"id": 1}, {"id": 2}])
    scanner_calls = 0
    real_scanner = artifacts._scan_artifact_rows

    def recording_scanner(stream):
        nonlocal scanner_calls
        scanner_calls += 1
        yield from real_scanner(stream)

    monkeypatch.setattr(artifacts, "_scan_artifact_rows", recording_scanner)
    reader = open_artifact_reader(
        tmp_path,
        "step1",
        "dataset",
        expected_config_digest=CONFIG_DIGEST,
        expected_lineage_digest=LINEAGE_DIGEST,
    )

    assert isinstance(reader, ArtifactReader)
    assert reader.index.row_count == 2
    assert scanner_calls == 1
    assert list(reader) == [{"id": 1}, {"id": 2}]
    assert reader.closed
    assert scanner_calls == 1


def test_reader_close_discards_verified_spool_without_fd_leak(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    _publish(tmp_path, [{"id": 1}, {"id": 2}, {"id": 3}])
    baseline = len(os.listdir("/proc/self/fd"))
    scanner_calls = 0
    scanned_rows = 0
    real_scanner = artifacts._scan_artifact_rows

    def recording_scanner(stream):
        nonlocal scanner_calls, scanned_rows
        scanner_calls += 1
        for item in real_scanner(stream):
            scanned_rows += 1
            yield item

    monkeypatch.setattr(artifacts, "_scan_artifact_rows", recording_scanner)
    reader = open_artifact_reader(
        tmp_path,
        "step1",
        "dataset",
        expected_config_digest=CONFIG_DIGEST,
        expected_lineage_digest=LINEAGE_DIGEST,
    )
    assert scanner_calls == 1
    assert scanned_rows == 3
    assert next(reader) == {"id": 1}

    reader.close()

    assert reader.closed
    assert scanner_calls == 1
    assert scanned_rows == 3
    assert len(os.listdir("/proc/self/fd")) == baseline


@pytest.mark.parametrize(
    "contract_name,replacement",
    [
        ("rows.jsonl", b'{"id":1}\n{"id":3}\n'),
        ("index.json", b'{"tampered":true}\n'),
        ("_SUCCESS", b"0" * 64 + b"\n"),
    ],
)
def test_reader_close_rejects_contract_tamper_and_closes_fds(
    tmp_path, contract_name, replacement,
):
    _publish(tmp_path, [{"id": 1}, {"id": 2}])
    baseline = len(os.listdir("/proc/self/fd"))
    reader = open_artifact_reader(
        tmp_path,
        "step1",
        "dataset",
        expected_config_digest=CONFIG_DIGEST,
        expected_lineage_digest=LINEAGE_DIGEST,
    )
    assert next(reader) == {"id": 1}
    target = _artifact_dir(tmp_path) / contract_name
    substitute = target.with_name(f"{target.name}.tampered")
    substitute.write_bytes(replacement)
    os.replace(substitute, target)

    with pytest.raises((ValueError, RuntimeError), match="changed|hash|index|marker|rows"):
        reader.close()

    assert reader.closed
    assert len(os.listdir("/proc/self/fd")) == baseline


def test_reader_close_rejects_step_parent_rebind(tmp_path):
    _publish(tmp_path, [{"id": 1}, {"id": 2}])
    reader = open_artifact_reader(
        tmp_path,
        "step1",
        "dataset",
        expected_config_digest=CONFIG_DIGEST,
        expected_lineage_digest=LINEAGE_DIGEST,
    )
    assert next(reader) == {"id": 1}
    step_parent = _artifact_dir(tmp_path).parent
    step_parent.rename(step_parent.with_name("step1.moved"))
    step_parent.mkdir()

    with pytest.raises(RuntimeError, match="entry changed"):
        reader.close()


def test_reader_integrity_error_precedes_fd_close_error_without_leak(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.utils.artifacts as artifacts

    _publish(tmp_path, [{"id": 1}, {"id": 2}])
    baseline = len(os.listdir("/proc/self/fd"))
    reader = open_artifact_reader(
        tmp_path,
        "step1",
        "dataset",
        expected_config_digest=CONFIG_DIGEST,
        expected_lineage_digest=LINEAGE_DIGEST,
    )
    assert next(reader) == {"id": 1}
    (_artifact_dir(tmp_path) / "rows.jsonl").write_bytes(
        b'{"id":1}\n{"id":3}\n'
    )
    real_close = artifacts.os.close
    injected = False

    def close_then_raise(fd):
        nonlocal injected
        path = os.readlink(f"/proc/self/fd/{fd}")
        real_close(fd)
        if not injected and os.path.basename(path) == "rows.jsonl":
            injected = True
            raise OSError("injected reader fd close failure")

    monkeypatch.setattr(artifacts.os, "close", close_then_raise)
    with pytest.raises((ValueError, RuntimeError), match="changed|hash|rows") as caught:
        reader.close()

    assert any(
        "injected reader fd close failure" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert reader.closed
    assert len(os.listdir("/proc/self/fd")) == baseline


def test_reader_scanner_failure_closes_all_held_descriptors(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    _publish(tmp_path, [{"id": 1}, {"id": 2}])
    baseline = len(os.listdir("/proc/self/fd"))
    real_scanner = artifacts._scan_artifact_rows

    def failing_scanner(stream):
        scanner = real_scanner(stream)
        yield next(scanner)
        raise OSError("injected scanner fd failure")

    monkeypatch.setattr(artifacts, "_scan_artifact_rows", failing_scanner)
    with pytest.raises(OSError, match="scanner fd failure"):
        open_artifact_reader(
            tmp_path,
            "step1",
            "dataset",
            expected_config_digest=CONFIG_DIGEST,
            expected_lineage_digest=LINEAGE_DIGEST,
        )

    assert len(os.listdir("/proc/self/fd")) == baseline


def test_finalize_accepts_staging_rmdir_success_then_raise(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    writer = ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)
    real_rmdir = artifacts.os.rmdir
    injected = False

    def remove_then_raise(path, *, dir_fd=None):
        nonlocal injected
        result = real_rmdir(path, dir_fd=dir_fd)
        if not injected and path.startswith(".dataset.stage-"):
            injected = True
            raise OSError("injected finalize post-rmdir failure")
        return result

    monkeypatch.setattr(artifacts.os, "rmdir", remove_then_raise)
    index = writer.finalize()

    assert index.artifact_dir == str(_artifact_dir(tmp_path))
    assert writer.state == "finalized"


def test_constructor_cleanup_failure_still_closes_all_fds(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    baseline = len(os.listdir("/proc/self/fd"))
    real_open_directory_at = artifacts._open_directory_at
    real_rmdir = artifacts.os.rmdir

    def fail_staging_open(parent_fd, name, *, create, sync_parent=False):
        if name.startswith(".dataset.stage-"):
            raise OSError("original staging open failure")
        return real_open_directory_at(
            parent_fd, name, create=create, sync_parent=sync_parent,
        )

    def fail_staging_rmdir(path, *, dir_fd=None):
        if path.startswith(".dataset.stage-"):
            raise OSError("injected constructor cleanup failure")
        return real_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts, "_open_directory_at", fail_staging_open)
    monkeypatch.setattr(artifacts.os, "rmdir", fail_staging_rmdir)
    with pytest.raises(OSError, match="original staging open failure"):
        ArtifactWriter(tmp_path, "step1", "dataset", CONFIG_DIGEST, LINEAGE_DIGEST)

    assert len(os.listdir("/proc/self/fd")) == baseline


def test_contract_fstat_error_is_not_replaced_by_close_error(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    _publish(tmp_path)
    real_fstat = artifacts.os.fstat
    real_close = artifacts.os.close

    def fail_index_fstat(fd):
        if os.path.basename(os.readlink(f"/proc/self/fd/{fd}")) == "index.json":
            raise OSError("original contract fstat failure")
        return real_fstat(fd)

    def close_then_raise(fd):
        path = os.readlink(f"/proc/self/fd/{fd}")
        real_close(fd)
        if os.path.basename(path) == "index.json":
            raise OSError("masking contract close failure")

    monkeypatch.setattr(artifacts.os, "fstat", fail_index_fstat)
    monkeypatch.setattr(artifacts.os, "close", close_then_raise)
    with pytest.raises(OSError, match="original contract fstat failure"):
        _open_bound(tmp_path, "step1", "dataset")


def test_reader_fdopen_error_is_not_replaced_by_close_error(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    index = _publish(tmp_path, [{"id": 1}])
    real_fdopen = artifacts.os.fdopen
    real_close = artifacts.os.close
    failed_fd = None

    def fail_reader_fdopen(fd, *args, **kwargs):
        nonlocal failed_fd
        if args and args[0] == "rb":
            failed_fd = fd
            raise OSError("original reader fdopen failure")
        return real_fdopen(fd, *args, **kwargs)

    def close_then_raise(fd):
        real_close(fd)
        if fd == failed_fd:
            raise OSError("masking duplicate close failure")

    monkeypatch.setattr(artifacts.os, "fdopen", fail_reader_fdopen)
    monkeypatch.setattr(artifacts.os, "close", close_then_raise)
    with pytest.raises(OSError, match="original reader fdopen failure"):
        list(iter_artifact_rows(index))


def test_drop_artifact_removes_published_artifact_idempotently(tmp_path):
    index = _publish(tmp_path, [{"id": 1}], step="step4", dataset_id="alpha")
    artifact_dir = Path(index.artifact_dir)
    assert artifact_dir.is_dir()

    drop_artifact(tmp_path, "step4", "alpha")
    assert not artifact_dir.exists()
    assert not Path(tmp_path / "step4" / "alpha").exists()
    # 幂等：目录已不存在时重复调用仍静默成功。
    drop_artifact(tmp_path, "step4", "alpha")
    drop_artifact(tmp_path, "step4", "alpha")
    # 相邻 dataset 不受影响。
    other = _publish(tmp_path, [{"id": 2}], step="step4", dataset_id="beta")
    assert Path(other.artifact_dir).is_dir()


@pytest.mark.parametrize("dataset_id", ["数据 集", "name with spaces", "percent%name"])
def test_drop_artifact_roundtrips_percent_encoded_dataset_directory(
    tmp_path, dataset_id,
):
    index = _publish(tmp_path, [{"id": 1}], step="step4", dataset_id=dataset_id)
    artifact_dir = Path(index.artifact_dir)
    raw_dir = tmp_path / "step4" / dataset_id
    assert artifact_dir.is_dir()
    assert artifact_dir != raw_dir

    drop_artifact(tmp_path, "step4", dataset_id)

    assert not artifact_dir.exists()
    # 第二次必须只把 canonical encoded entry 的缺失视为幂等成功。
    drop_artifact(tmp_path, "step4", dataset_id)


def test_drop_artifact_accepts_pipeline_step_enum(tmp_path):
    index = _publish(tmp_path, [{"id": 1}], step="step5", dataset_id="alpha")
    drop_artifact(tmp_path, PipelineStep.STEP5, "alpha")
    assert not Path(index.artifact_dir).exists()


@pytest.mark.parametrize("rogue", [False, True], ids=["success", "rogue-refusal"])
def test_drop_artifact_success_and_rogue_refusal_do_not_grow_fds(
    tmp_path, rogue,
):
    index = _publish(tmp_path, [{"id": 1}], step="step6", dataset_id="fd-check")
    artifact_dir = Path(index.artifact_dir)
    if rogue:
        (artifact_dir / "rogue.txt").write_text("not a contract file", encoding="utf-8")
    baseline = len(os.listdir("/proc/self/fd"))

    if rogue:
        with pytest.raises(ValueError, match="unexpected artifact file"):
            drop_artifact(tmp_path, "step6", "fd-check")
        assert artifact_dir.is_dir()
    else:
        drop_artifact(tmp_path, "step6", "fd-check")
        assert not artifact_dir.exists()

    assert len(os.listdir("/proc/self/fd")) == baseline


@pytest.mark.parametrize("rogue", [False, True], ids=["success", "rogue-refusal"])
def test_drop_artifact_closes_artifact_and_parent_fds_once(
    tmp_path, monkeypatch, rogue,
):
    import audio_filter_pipeline.utils.artifacts as artifacts

    index = _publish(tmp_path, [{"id": 1}], step="step6", dataset_id="close-check")
    artifact_dir = Path(index.artifact_dir)
    if rogue:
        (artifact_dir / "rogue.txt").write_text("not a contract file", encoding="utf-8")
    real_open_artifact_directory = artifacts._open_artifact_directory
    real_close = artifacts.os.close
    opened = {}
    close_counts = {}

    def recording_open_artifact_directory(*args, **kwargs):
        parent_chain, artifact_fd = real_open_artifact_directory(*args, **kwargs)
        opened["artifact_fd"] = artifact_fd
        opened["parent_fds"] = tuple(parent_chain.descriptors)
        return parent_chain, artifact_fd

    def recording_close(fd):
        close_counts[fd] = close_counts.get(fd, 0) + 1
        return real_close(fd)

    monkeypatch.setattr(
        artifacts, "_open_artifact_directory", recording_open_artifact_directory,
    )
    monkeypatch.setattr(artifacts.os, "close", recording_close)
    if rogue:
        with pytest.raises(ValueError, match="unexpected artifact file"):
            drop_artifact(tmp_path, "step6", "close-check")
    else:
        drop_artifact(tmp_path, "step6", "close-check")

    owned_fds = (opened["artifact_fd"], *opened["parent_fds"])
    assert all(close_counts.get(fd) == 1 for fd in owned_fds)


def test_drop_artifact_notes_artifact_and_parent_close_failures(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.utils.artifacts as artifacts

    index = _publish(tmp_path, [{"id": 1}], step="step6", dataset_id="close-check")
    artifact_dir = Path(index.artifact_dir)
    (artifact_dir / "rogue.txt").write_text("not a contract file", encoding="utf-8")
    real_open_artifact_directory = artifacts._open_artifact_directory
    real_close = artifacts.os.close
    opened = {}

    def recording_open_artifact_directory(*args, **kwargs):
        parent_chain, artifact_fd = real_open_artifact_directory(*args, **kwargs)
        opened["artifact_fd"] = artifact_fd
        opened["parent_fds"] = tuple(parent_chain.descriptors[-2:])
        return parent_chain, artifact_fd

    def close_then_fail(fd):
        real_close(fd)
        if fd == opened.get("artifact_fd"):
            raise OSError("injected artifact close failure")
        if fd in opened.get("parent_fds", ()):
            raise OSError(f"injected parent close failure {fd}")

    monkeypatch.setattr(
        artifacts, "_open_artifact_directory", recording_open_artifact_directory,
    )
    monkeypatch.setattr(artifacts.os, "close", close_then_fail)
    with pytest.raises(ValueError, match="unexpected artifact file") as caught:
        drop_artifact(tmp_path, "step6", "close-check")

    notes = getattr(caught.value, "__notes__", ())
    assert any(
        "artifact fd close failed" in note
        and "injected artifact close failure" in note
        for note in notes
    )
    assert any(
        "directory chain close failed" in note
        and "injected parent close failure" in note
        for note in notes
    )
    assert any(
        "additional directory fd close failed" in note
        and "injected parent close failure" in note
        for note in notes
    )


def test_drop_artifact_fails_closed_on_unknown_entries(tmp_path):
    index = _publish(tmp_path, [{"id": 1}], step="step6", dataset_id="alpha")
    artifact_dir = Path(index.artifact_dir)
    (artifact_dir / "rogue.txt").write_text("not a contract file", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected artifact file"):
        drop_artifact(tmp_path, "step6", "alpha")
    # 拒绝删除后目录原样保留。
    assert artifact_dir.is_dir()
    assert (artifact_dir / "rogue.txt").read_text(encoding="utf-8") == "not a contract file"
    assert (artifact_dir / "rows.jsonl").is_file()
