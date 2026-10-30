"""
Tests for core.registry
"""
import os
import tempfile
import pytest

from noise_filter_pipeline.core.registry import DatasetRegistry, DatasetEntry

pytestmark = pytest.mark.pipeline


def test_registry_load_and_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "registry.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write("""
datasets:
  - id: esc50
    name: ESC50
    enabled: true
    type: directory
    path: /data/esc50
    input_pattern: "**/*.wav"
    label_source: mapping
    label_mapping: config/labels/esc50_to_audioset.yaml
    splits: [train, val, test]
    split_strategy: directory
    license: "CC BY-NC 3.0"
  - id: disabled_ds
    name: Disabled
    enabled: false
    type: directory
    path: /data/disabled
""")
        registry = DatasetRegistry(path)
        assert registry.has("esc50")
        assert not registry.has("nonexistent")
        enabled = registry.list_enabled()
        assert len(enabled) == 1
        assert enabled[0].id == "esc50"


def test_dataset_entry_validation():
    entry = DatasetEntry(id="test", name="Test", type="directory", path="/tmp")
    assert entry.id == "test"

    with pytest.raises(ValueError):
        DatasetEntry(id="bad", name="Bad", type="unsupported", path="/tmp")


@pytest.mark.parametrize("dataset_id", ["../escape", "nested/id", "bad__id", "with space"])
def test_dataset_entry_rejects_unsafe_id(dataset_id):
    with pytest.raises(ValueError, match="id"):
        DatasetEntry(id=dataset_id, name="Bad", path="/tmp")


def test_registry_rejects_duplicate_dataset_ids():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "registry.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write("""
datasets:
  - id: duplicate
    path: /opt/nonexistent/one
  - id: duplicate
    path: /opt/nonexistent/two
""")

        with pytest.raises(ValueError, match="重复"):
            DatasetRegistry(path)


def test_dataset_entry_rejects_unimplemented_filename_split_strategy():
    """split_strategy=filename 从未实现，应报 ValueError 并指明未实现。"""
    with pytest.raises(ValueError, match="未实现"):
        DatasetEntry(
            id="ds", name="DS", path="/tmp",
            splits=["train", "test"], split_strategy="filename",
        )


def test_dataset_entry_rejects_unknown_split_strategy():
    with pytest.raises(ValueError, match="split_strategy"):
        DatasetEntry(
            id="ds", name="DS", path="/tmp",
            splits=["train"], split_strategy="bogus",
        )


def test_dataset_entry_ratio_requires_matching_split_ratio_length():
    with pytest.raises(ValueError, match="split_ratio"):
        DatasetEntry(
            id="ds", name="DS", path="/tmp",
            splits=["train", "val", "test"], split_strategy="ratio",
            split_ratio=[0.8, 0.2],
        )


def test_dataset_entry_ratio_requires_split_ratio_present():
    with pytest.raises(ValueError, match="split_ratio"):
        DatasetEntry(
            id="ds", name="DS", path="/tmp",
            splits=["train", "test"], split_strategy="ratio",
            split_ratio=None,
        )


def test_dataset_entry_ratio_requires_split_ratio_sum_one():
    with pytest.raises(ValueError, match="之和"):
        DatasetEntry(
            id="ds", name="DS", path="/tmp",
            splits=["train", "test"], split_strategy="ratio",
            split_ratio=[0.7, 0.7],
        )


def test_dataset_entry_ratio_valid_config_accepted():
    entry = DatasetEntry(
        id="ds", name="DS", path="/tmp",
        splits=["train", "val", "test"], split_strategy="ratio",
        split_ratio=[0.8, 0.1, 0.1],
    )
    assert entry.split_strategy == "ratio"


def test_registry_resolve_path_absolute_passthrough():
    """绝对路径原样返回，保持既有行为不变。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "registry.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write("datasets: []\n")
        registry = DatasetRegistry(path)
        assert registry.resolve_path("/data/abs/path") == "/data/abs/path"


def test_registry_resolve_path_relative_uses_registry_dir():
    """相对路径按 registry 文件所在目录解析为绝对路径。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_dir = os.path.join(tmpdir, "sub", "reg")
        os.makedirs(reg_dir)
        path = os.path.join(reg_dir, "registry.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write("datasets: []\n")
        registry = DatasetRegistry(path)
        assert registry.resolve_path("data/x.wav") == os.path.join(reg_dir, "data", "x.wav")
        assert os.path.isabs(registry.resolve_path("data"))
