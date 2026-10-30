"""噪声属性生成器的 manifest 读取契约测试。"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


@pytest.fixture()
def generator():
    """加载生成器模块，避免在测试收集阶段执行生成入口。"""
    return importlib.import_module("tools.ops.data.generate_noise_attributes")


def test_read_manifest_labels_streams_and_deduplicates(tmp_path: Path, generator):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"category_name": "Zebra"}),
                "",
                json.dumps({"category_name": "Alpha"}),
                json.dumps({"category_name": "Zebra"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert generator._read_manifest_labels(manifest) == ["Alpha", "Zebra"]


def test_empty_manifest_raises_system_exit(tmp_path: Path, monkeypatch, generator):
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "data/shared/noise/filter/v2/train_noise_sources.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("\n", encoding="utf-8")
    (tmp_path / "core/data").mkdir(parents=True)

    with pytest.raises(SystemExit, match="未解析到任何 category_name"):
        generator._main()


def test_missing_category_name_keeps_key_error_semantics(tmp_path: Path, generator):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"uid": "without-category"}) + "\n", encoding="utf-8")

    with pytest.raises(KeyError, match="category_name"):
        generator._read_manifest_labels(manifest)
