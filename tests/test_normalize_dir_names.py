from __future__ import annotations

import json

import pytest

from tools.ops.data.normalize_dir_names import (
    _normalize_name,
    _resolve_unique_name,
    main,
)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("My Folder!", "my_folder"),
        ("测试 (2024)!", "2024"),
        ("__a  b__", "a_b"),
        ("!!!", "dir"),
        ("Foo-Bar_123", "foo-bar_123"),
        ("  leading spaces  ", "leading_spaces"),
    ],
)
def test_normalize_name(name, expected):
    assert _normalize_name(name) == expected


def test_normalize_name_respects_options():
    assert _normalize_name("My Folder!", lowercase=False) == "My_Folder"
    assert _normalize_name("a b", replacement="-") == "a-b"


def test_resolve_unique_name(tmp_path):
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo_1").mkdir()
    assert _resolve_unique_name(tmp_path, "foo") == "foo_2"
    assert _resolve_unique_name(tmp_path, "bar") == "bar"


def test_main_dry_run_by_default(tmp_path):
    original = tmp_path / "My Folder!"
    original.mkdir()
    mapping = main([str(tmp_path)])
    assert original.exists()
    assert mapping
    assert any("My Folder!" in k for k in mapping)


def test_main_force_renames_directories(tmp_path):
    dirty = tmp_path / "My Folder!"
    dirty.mkdir()
    mapping = main([str(tmp_path), "--force"])
    assert not dirty.exists()
    assert (tmp_path / "my_folder").exists()
    assert mapping == {str(dirty): str(tmp_path / "my_folder")}


def test_main_conflict_dedup(tmp_path):
    a = tmp_path / "a b"
    a.mkdir()
    b = tmp_path / "a_b"
    b.mkdir()
    mapping = main([str(tmp_path), "--force"])
    names = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert names == {"a_b", "a_b_1"}
    assert len(mapping) == 1  # a_b 未改变，只有 a b 被重命名


def test_main_skip_hidden(tmp_path):
    hidden = tmp_path / ".hidden dir"
    hidden.mkdir()
    visible = tmp_path / "Visible Dir!"
    visible.mkdir()
    main([str(tmp_path), "--force"])
    assert hidden.exists()
    assert (tmp_path / "visible_dir").exists()


def test_main_no_skip_hidden(tmp_path):
    hidden = tmp_path / ".hidden dir"
    hidden.mkdir()
    main([str(tmp_path), "--force", "--no-skip-hidden"])
    assert not hidden.exists()
    assert (tmp_path / "hidden_dir").exists()


def test_main_output_mapping(tmp_path):
    dirty = tmp_path / "Dirty Name!"
    dirty.mkdir()
    mapping_path = tmp_path / "map.json"
    main([str(tmp_path), "--force", "--output-mapping", str(mapping_path)])
    assert mapping_path.exists()
    with open(mapping_path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved == {str(dirty): str(tmp_path / "dirty_name")}


def test_main_recursive_bottom_up(tmp_path):
    parent = tmp_path / "Parent Dir!"
    parent.mkdir()
    child = parent / "Child Dir!"
    child.mkdir()
    main([str(tmp_path), "--force"])
    assert (tmp_path / "parent_dir" / "child_dir").exists()

pytestmark = [pytest.mark.unit]
