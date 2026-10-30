"""配置快照必须绑定实际解析的字节，而非加载结束后的同名文件。"""

import hashlib

import pytest

import core.utils.config as config_utils
from core.utils.inference_artifacts import fingerprint_json

pytestmark = pytest.mark.unit


def test_snapshot_binds_actual_fd_bytes_when_base_path_is_replaced(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    replacement = tmp_path / "replacement.yaml"
    original = b"value: original\r\nitems: [1, 2]\r\n"
    base.write_bytes(original)
    child.write_text("base_config: base.yaml\ncopy: ${value}\n", encoding="utf-8")
    replacement.write_text("value: replaced\n", encoding="utf-8")
    real_load = config_utils.OmegaConf.load
    load_count = 0

    def replace_after_open(source):
        nonlocal load_count
        load_count += 1
        if load_count == 2:
            replacement.replace(base)
        return real_load(source)

    monkeypatch.setattr(config_utils.OmegaConf, "load", replace_after_open)
    snapshot = config_utils.load_merged_config_snapshot(child)
    assert snapshot.config == {"value": "original", "items": [1, 2], "copy": "original"}
    assert snapshot.file_identities[str(base)] == {
        "path": str(base), "size": len(original), "sha256": hashlib.sha256(original).hexdigest()
    }
    assert set(snapshot.file_identities) == {str(base), str(child)}
    assert snapshot.resolved_identity == {
        "algorithm": "sha256-json-v1", "sha256": fingerprint_json(snapshot.config)
    }
    assert base.read_text(encoding="utf-8") == "value: replaced\n"
    with pytest.raises(RuntimeError, match="source artifact changed"):
        snapshot.verify_current_sources()


@pytest.mark.parametrize("pop_base", [False, True])
def test_snapshot_full_chain_merge_and_views_are_isolated(tmp_path, pop_base):
    from core.utils.artifact_identity import SourceBinding

    root = tmp_path / "configs"
    nested = root / "task"
    nested.mkdir(parents=True)
    grandparent = root / "common.yaml"
    parent = root / "parent.yaml"
    right = root / "right.yaml"
    child = nested / "child.yaml"
    grandparent.write_text("width: 3\nitems: [1, 2]\nnested: {left: 1}\n", encoding="utf-8")
    parent.write_text("base_config: common.yaml\nnested: {parent: 2}\n", encoding="utf-8")
    right.write_text("base_config: common.yaml\nnested: {right: 3}\n", encoding="utf-8")
    child.write_text(
        "base_config: [../parent.yaml, ../right.yaml]\nitems: [9]\ncopy: ${width}\n",
        encoding="utf-8",
    )
    snapshot = config_utils.load_merged_config_snapshot(child, pop_base=pop_base)
    expected = {
        "width": 3, "items": [9], "nested": {"left": 1, "parent": 2, "right": 3}, "copy": 3,
    }
    if not pop_base:
        expected["base_config"] = ["../parent.yaml", "../right.yaml"]
    assert snapshot.config == expected == config_utils.load_merged_config(child, pop_base=pop_base)
    assert set(snapshot.file_identities) == {str(p) for p in (child, parent, right, grandparent)}
    snapshot.verify_current_sources()
    loaded_identity = snapshot.resolved_identity
    snapshot.config["nested"]["left"] = 999
    snapshot.config["items"].append(999)
    snapshot.file_identities[str(parent)]["sha256"] = "0" * 64
    snapshot.resolved_identity["sha256"] = "0" * 64
    assert snapshot.config == expected
    assert snapshot.resolved_identity == loaded_identity
    snapshot.verify_current_sources()
    identities = snapshot.file_identities
    binding = SourceBinding.bind(
        {path: path for path in identities},
        precomputed=identities, parameters={"resolved_config": loaded_identity},
    )
    binding.verify()
    grandparent.write_text("width: 4\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source artifact changed"):
        snapshot.verify_current_sources()
    with pytest.raises(RuntimeError, match="source artifact changed"):
        binding.verify()


@pytest.mark.parametrize("target_name", ["child.yaml", "base.yaml"])
@pytest.mark.parametrize("replace", [False, True])
def test_snapshot_rejects_post_load_mutation_of_every_source(tmp_path, target_name, replace):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("value: A\n", encoding="utf-8")
    child.write_text("base_config: base.yaml\n", encoding="utf-8")
    snapshot = config_utils.load_merged_config_snapshot(child)
    original_identity = snapshot.resolved_identity
    target = tmp_path / target_name
    if replace:
        replacement = tmp_path / "replacement.yaml"
        replacement.write_text("value: B\n", encoding="utf-8")
        replacement.replace(target)
    else:
        target.write_text("value: B\n", encoding="utf-8")
    assert snapshot.config == {"value": "A"}
    assert snapshot.resolved_identity == original_identity
    with pytest.raises(RuntimeError, match="source artifact changed"):
        snapshot.verify_current_sources()


def test_snapshot_resolved_identity_ignores_source_formatting(tmp_path):
    left = tmp_path / "left.yaml"
    right = tmp_path / "right.yaml"
    left.write_bytes(b"b: [1, 2]\na: 3\n")
    right.write_bytes(b"# same effective values\r\na: 3\r\nb: [1, 2]\r\n")
    a = config_utils.load_merged_config_snapshot(left)
    b = config_utils.load_merged_config_snapshot(right)
    assert a.resolved_identity == b.resolved_identity
    assert a.file_identities[str(left)]["sha256"] != b.file_identities[str(right)]["sha256"]


@pytest.mark.parametrize("case", ["absolute", "escape", "cycle", "hardlink_cycle", "symlink", "ancestor_symlink"])
def test_snapshot_preserves_path_and_cycle_guards(tmp_path, case):
    root = tmp_path / "configs"
    root.mkdir()
    child = root / "child.yaml"
    base = root / "base.yaml"
    base.write_text("value: 1\n", encoding="utf-8")
    if case == "absolute":
        child.write_text(f"base_config: {base}\n", encoding="utf-8")
    elif case == "escape":
        child.write_text("base_config: ../outside.yaml\n", encoding="utf-8")
    elif case == "cycle":
        child.write_text("base_config: child.yaml\n", encoding="utf-8")
    elif case == "hardlink_cycle":
        child.write_text("base_config: alias.yaml\n", encoding="utf-8")
        (root / "alias.yaml").hardlink_to(child)
    elif case == "symlink":
        (root / "alias.yaml").symlink_to(base)
        child.write_text("base_config: alias.yaml\n", encoding="utf-8")
    else:
        (root / "alias").symlink_to(root, target_is_directory=True)
        child.write_text("base_config: alias/base.yaml\n", encoding="utf-8")
    with pytest.raises((ValueError, OSError)):
        config_utils.load_merged_config_snapshot(child)


def test_snapshot_freezes_resolved_environment_interpolation(tmp_path, monkeypatch):
    source = tmp_path / "source.yaml"
    source.write_text("value: ${oc.env:CONFIG_SNAPSHOT_TEST_VALUE}\n", encoding="utf-8")
    monkeypatch.setenv("CONFIG_SNAPSHOT_TEST_VALUE", "original")
    snapshot = config_utils.load_merged_config_snapshot(source)
    monkeypatch.setenv("CONFIG_SNAPSHOT_TEST_VALUE", "changed")
    assert snapshot.config == {"value": "original"}
    assert config_utils.load_merged_config(source) == {"value": "changed"}
    snapshot.verify_current_sources()


def test_snapshot_rejects_conflicting_reads_of_shared_base(tmp_path, monkeypatch):
    source = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    source.write_text("value: A\n", encoding="utf-8")
    child.write_text("base_config: [base.yaml, base.yaml]\n", encoding="utf-8")
    real_load = config_utils.OmegaConf.load
    calls = 0

    def change_after_first_base_read(stream):
        nonlocal calls
        calls += 1
        result = real_load(stream)
        if calls == 2:
            source.write_text("value: B\n", encoding="utf-8")
        return result

    monkeypatch.setattr(config_utils.OmegaConf, "load", change_after_first_base_read)
    with pytest.raises(RuntimeError, match="changed between inherited reads"):
        config_utils.load_merged_config_snapshot(child)


def test_snapshot_rejects_mutation_during_fd_read(tmp_path, monkeypatch):
    source = tmp_path / "source.yaml"
    source.write_text("value: A\n", encoding="utf-8")
    real_fdopen = config_utils.os.fdopen

    def open_then_mutate_on_read(*args, **kwargs):
        stream = real_fdopen(*args, **kwargs)
        real_read = stream.read

        def read_then_mutate(*read_args, **read_kwargs):
            content = real_read(*read_args, **read_kwargs)
            source.write_text("value: B changed\n", encoding="utf-8")
            return content

        stream.read = read_then_mutate
        return stream

    monkeypatch.setattr(config_utils.os, "fdopen", open_then_mutate_on_read)
    with pytest.raises(RuntimeError, match="changed during read"):
        config_utils.load_merged_config_snapshot(source)
