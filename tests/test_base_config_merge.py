"""测试 YAML 配置继承（base_config）。"""

from __future__ import annotations
import pytest

import tempfile
from pathlib import Path

import core.utils.config as config_utils
from core.utils.config import load_merged_config


def test_single_base_config_merge():
    """单 base 配置：子配置覆盖父配置。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        base_path = root / "base.yaml"
        child_path = root / "child.yaml"

        base_path.write_text("model:\n  type: Base\n  args:\n    lr: 0.001\n", encoding="utf-8")
        child_path.write_text(
            "base_config: base.yaml\nmodel:\n  args:\n    hidden: 64\n", encoding="utf-8"
        )

        cfg = load_merged_config(child_path)
        assert cfg["model"]["type"] == "Base"
        assert cfg["model"]["args"]["lr"] == 0.001
        assert cfg["model"]["args"]["hidden"] == 64
        assert "base_config" not in cfg


def test_multiple_base_config_merge():
    """多个 base 配置：从左到右合并，子配置覆盖所有 base。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        base1 = root / "base1.yaml"
        base2 = root / "base2.yaml"
        child = root / "child.yaml"

        base1.write_text("a: 1\nb: 1\n", encoding="utf-8")
        base2.write_text("b: 2\nc: 2\n", encoding="utf-8")
        child.write_text(
            "base_config:\n  - base1.yaml\n  - base2.yaml\nc: 3\n", encoding="utf-8"
        )

        cfg = load_merged_config(child)
        assert cfg["a"] == 1
        assert cfg["b"] == 2  # base2 覆盖 base1
        assert cfg["c"] == 3  # child 覆盖 base2


def test_base_config_interpolation():
    """子配置可使用 base 配置中的值进行插值。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        base = root / "base.yaml"
        child = root / "child.yaml"

        base.write_text("common:\n  lr: 0.01\n", encoding="utf-8")
        child.write_text(
            "base_config: base.yaml\noptimizer:\n  lr: ${common.lr}\n", encoding="utf-8"
        )

        cfg = load_merged_config(child)
        assert cfg["optimizer"]["lr"] == 0.01


def test_multilevel_base_config_chain():
    """多层继承（child -> parent -> grandparent）：祖父配置字段不能被静默丢弃。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        grandparent = root / "grandparent.yaml"
        parent = root / "parent.yaml"
        child = root / "child.yaml"

        grandparent.write_text(
            "type: Grandparent\nlr: 0.001\ngrandparent_only: gp\nnested:\n  a: 1\n  b: 2\nitems: [1, 2, 3]\n",
            encoding="utf-8",
        )
        parent.write_text(
            "base_config: grandparent.yaml\ntype: Parent\nparent_only: p\nnested:\n  b: 20\n",
            encoding="utf-8",
        )
        child.write_text(
            "base_config: parent.yaml\ntype: Child\nitems: [9]\n", encoding="utf-8"
        )

        cfg = load_merged_config(child)
        assert cfg["type"] == "Child"
        assert cfg["lr"] == 0.001  # from grandparent, must not be dropped
        assert cfg["grandparent_only"] == "gp"
        assert cfg["parent_only"] == "p"
        assert cfg["nested"] == {"a": 1, "b": 20}
        assert cfg["items"] == [9]  # lists replace wholesale, not merge
        assert "base_config" not in cfg


def test_circular_base_config_raises():
    """循环继承必须 fail-fast，而不是无限递归或静默截断。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        a = root / "a.yaml"
        b = root / "b.yaml"

        a.write_text("base_config: b.yaml\na_val: 1\n", encoding="utf-8")
        b.write_text("base_config: a.yaml\nb_val: 2\n", encoding="utf-8")

        try:
            load_merged_config(a)
            assert False, "circular base_config should raise ValueError"
        except ValueError as e:
            assert "Circular" in str(e)


def test_base_config_rejects_absolute_path(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("value: 1\n", encoding="utf-8")
    child.write_text(f"base_config: {base}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be relative"):
        load_merged_config(child)


def test_base_config_allows_parent_reference_within_configs_root(tmp_path):
    root = tmp_path / "configs"
    nested = root / "task"
    nested.mkdir(parents=True)
    base = root / "base.yaml"
    child = nested / "child.yaml"
    base.write_text("value: 1\n", encoding="utf-8")
    child.write_text("base_config: ../base.yaml\nchild: true\n", encoding="utf-8")

    assert load_merged_config(child) == {"value": 1, "child": True}


def test_base_config_rejects_parent_escape(tmp_path):
    root = tmp_path / "configs"
    nested = root / "task"
    nested.mkdir(parents=True)
    outside = tmp_path / "outside.yaml"
    child = nested / "child.yaml"
    outside.write_text("value: 1\n", encoding="utf-8")
    child.write_text("base_config: ../../outside.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes config root"):
        load_merged_config(child)


def test_base_config_rejects_symlink_escape(tmp_path):
    root = tmp_path / "configs"
    root.mkdir()
    outside = tmp_path / "outside.yaml"
    link = root / "linked.yaml"
    child = root / "child.yaml"
    outside.write_text("value: 1\n", encoding="utf-8")
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unsupported: {exc}")
    child.write_text("base_config: linked.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="symbolic links"):
        load_merged_config(child)


def test_base_config_reads_bound_fd_if_path_is_replaced(tmp_path, monkeypatch):
    root = tmp_path / "configs"
    root.mkdir()
    base = root / "base.yaml"
    replacement = root / "replacement.yaml"
    child = root / "child.yaml"
    base.write_text("value: original\n", encoding="utf-8")
    replacement.write_text("value: attacker\n", encoding="utf-8")
    child.write_text("base_config: base.yaml\n", encoding="utf-8")
    real_load = config_utils.OmegaConf.load
    load_count = 0

    def replace_after_open(source):
        nonlocal load_count
        load_count += 1
        if load_count == 2:
            replacement.replace(base)
        return real_load(source)

    monkeypatch.setattr(config_utils.OmegaConf, "load", replace_after_open)

    assert load_merged_config(child)["value"] == "original"
    assert base.read_text(encoding="utf-8") == "value: attacker\n"


if __name__ == "__main__":
    test_single_base_config_merge()
    test_multiple_base_config_merge()
    test_base_config_interpolation()
    test_multilevel_base_config_chain()
    test_circular_base_config_raises()
    print("base_config merge tests passed.")

pytestmark = [pytest.mark.unit]
