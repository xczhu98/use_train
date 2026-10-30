"""
Tests for validate_dataset_registry.py.
"""
import json
import os
import sys
import tempfile

import numpy as np
import soundfile as sf
import pytest

import noise_filter_pipeline.scripts.validate_dataset_registry as validate_module
from noise_filter_pipeline.scripts.validate_dataset_registry import main as validate_main
from noise_filter_pipeline.scripts.validate_dataset_registry import validate_registry

pytestmark = pytest.mark.pipeline


def _make_wav(path, duration=1.0, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav = np.random.randn(int(duration * sr)) * 0.1
    sf.write(path, wav, sr)


def test_validate_registry_pass():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "data")
        _make_wav(os.path.join(dataset_dir, "a.wav"))
        _make_wav(os.path.join(dataset_dir, "b.wav"))

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test
    name: Test
    enabled: true
    type: directory
    path: {dataset_dir}
    input_pattern: "**/*.wav"
    label_source: filename
    splits: [train]
    split_strategy: none
""")

        report = validate_registry(registry_path, tmpdir)
        assert report["all_passed"] is True
        ds = report["datasets"][0]
        assert ds["matched_files"] == 2


def test_validate_registry_missing_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write("""
datasets:
  - id: missing
    name: Missing
    enabled: true
    type: directory
    path: /nonexistent/dataset
    input_pattern: "**/*.wav"
    label_source: filename
    splits: [train]
    split_strategy: none
""")

        report = validate_registry(registry_path, tmpdir)
        assert report["all_passed"] is False
        ds = report["datasets"][0]
        assert ds["path_status"]["passed"] is False


def test_validate_registry_disabled_skipped():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write("""
datasets:
  - id: disabled
    name: Disabled
    enabled: false
    type: directory
    path: /nonexistent/dataset
    input_pattern: "**/*.wav"
    label_source: filename
    splits: [train]
    split_strategy: none
""")

        report = validate_registry(registry_path, tmpdir)
        assert report["all_passed"] is True
        ds = report["datasets"][0]
        assert ds["status"] == "skipped"


def test_validate_registry_relative_path_resolved_against_registry_dir():
    """条目相对路径应按 registry 文件所在目录解析（与 Stage A 同口径）。

    base_dir 故意指向另一个目录：若仍按 base_dir 解析会误判不存在。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_dir = os.path.join(tmpdir, "reg")
        os.makedirs(reg_dir)
        _make_wav(os.path.join(reg_dir, "data", "a.wav"))
        _make_wav(os.path.join(reg_dir, "data", "b.wav"))

        registry_path = os.path.join(reg_dir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write("""
datasets:
  - id: rel_ds
    name: Relative
    enabled: true
    type: directory
    path: data
    input_pattern: "**/*.wav"
    label_source: filename
    splits: [train]
""")

        other_base = os.path.join(tmpdir, "elsewhere")
        os.makedirs(other_base)
        report = validate_registry(registry_path, other_base)
        assert report["all_passed"] is True
        ds = report["datasets"][0]
        assert ds["matched_files"] == 2
        assert ds["path"] == os.path.join(reg_dir, "data")


def test_validate_registry_glob_filters_audio_extensions():
    """directory 匹配数应按 AUDIO_EXTS 过滤，与 Stage A 口径一致。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "data")
        _make_wav(os.path.join(dataset_dir, "a.wav"))
        # 非音频文件不应计入匹配数
        with open(os.path.join(dataset_dir, "notes.txt"), "w", encoding="utf-8") as f:
            f.write("not audio")

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: filter_ds
    name: Filter
    enabled: true
    type: directory
    path: {dataset_dir}
    input_pattern: "**/*"
    label_source: filename
    splits: [train]
""")

        report = validate_registry(registry_path, tmpdir)
        assert report["all_passed"] is True
        assert report["datasets"][0]["matched_files"] == 1


def test_validate_registry_ignores_label_mapping_when_source_not_mapping():
    """label_source 非 mapping 时，即使 label_mapping 指向不存在文件也不应失败。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "data")
        _make_wav(os.path.join(dataset_dir, "a.wav"))

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: stale_mapping
    name: Stale Mapping
    enabled: true
    type: directory
    path: {dataset_dir}
    input_pattern: "**/*.wav"
    label_source: filename
    label_mapping: /nonexistent/mapping.yaml
    splits: [train]
""")

        report = validate_registry(registry_path, tmpdir)
        assert report["all_passed"] is True
        ds = report["datasets"][0]
        assert ds["mapping_status"]["passed"] is True


def test_validate_registry_mapping_source_requires_existing_mapping_file():
    """label_source=mapping 且映射文件不存在时应失败。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "data")
        _make_wav(os.path.join(dataset_dir, "a.wav"))

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: missing_mapping
    name: Missing Mapping
    enabled: true
    type: directory
    path: {dataset_dir}
    input_pattern: "**/*.wav"
    label_source: mapping
    label_mapping: no_such_mapping.yaml
    splits: [train]
""")

        report = validate_registry(registry_path, tmpdir)
        assert report["all_passed"] is False
        ds = report["datasets"][0]
        assert ds["mapping_status"]["passed"] is False


def test_validate_registry_label_mapping_relative_to_registry_dir():
    """label_mapping 仅文件名时按 <registry 目录>/labels/ 解析（与 Stage A 一致）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_dir = os.path.join(tmpdir, "reg")
        os.makedirs(reg_dir)
        _make_wav(os.path.join(reg_dir, "data", "a.wav"))
        labels_dir = os.path.join(reg_dir, "labels")
        os.makedirs(labels_dir)
        with open(os.path.join(labels_dir, "m.yaml"), "w", encoding="utf-8") as f:
            f.write('mappings:\n  a: Dog\n')

        registry_path = os.path.join(reg_dir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write("""
datasets:
  - id: mapping_ds
    name: Mapping
    enabled: true
    type: directory
    path: data
    input_pattern: "**/*.wav"
    label_source: mapping
    label_mapping: m.yaml
    splits: [train]
""")

        report = validate_registry(registry_path, tmpdir)
        assert report["all_passed"] is True
        ds = report["datasets"][0]
        assert ds["mapping_status"]["passed"] is True


def test_validate_cli_without_overrides_uses_default_pipeline_registry(monkeypatch, capsys):
    """无显式参数时应从默认 pipeline.yaml 推导便携 noise.yaml。"""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_dataset_registry.py", "--json"],
    )
    monkeypatch.setattr(
        validate_module,
        "validate_registry",
        lambda registry_path, base_dir: {
            "registry_path": registry_path,
            "base_dir": base_dir,
            "all_passed": True,
            "datasets": [],
        },
    )

    assert validate_main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["registry_path"] == os.path.join(
        repo_root, "tools", "noise_filter_pipeline", "config", "noise.yaml"
    )


def test_validate_cli_config_resolves_repo_relative_registry_without_duplicate_tools(
    monkeypatch, capsys
):
    """--config 应按仓库根解析 registry_path，而不是重复拼接 tools/。"""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    monkeypatch.chdir(repo_root)
    config_path = "tools/noise_filter_pipeline/config/pipeline.yaml"
    expected_registry = os.path.join(
        repo_root, "tools", "noise_filter_pipeline", "config", "noise.yaml"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_dataset_registry.py", "--config", config_path, "--json"],
    )
    monkeypatch.setattr(
        validate_module,
        "validate_registry",
        lambda registry_path, base_dir: {
            "registry_path": registry_path,
            "base_dir": base_dir,
            "all_passed": True,
            "datasets": [],
        },
    )

    assert validate_main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["registry_path"] == expected_registry
    assert report["base_dir"] == repo_root
    assert "tools/tools/" not in report["registry_path"]


def test_validate_cli_explicit_registry_preserves_relative_and_absolute_paths(
    monkeypatch, capsys
):
    """显式 --registry 的仓库相对路径与绝对路径均按原参数使用。"""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    monkeypatch.chdir(repo_root)
    registry_rel = "tools/noise_filter_pipeline/config/noise.yaml"
    monkeypatch.setattr(
        validate_module,
        "validate_registry",
        lambda registry_path, base_dir: {
            "registry_path": registry_path,
            "base_dir": base_dir,
            "all_passed": True,
            "datasets": [],
        },
    )
    for registry_arg in (registry_rel, os.path.join(repo_root, registry_rel)):
        monkeypatch.setattr(
            sys,
            "argv",
            ["validate_dataset_registry.py", "--registry", registry_arg, "--json"],
        )

        assert validate_main() == 0
        report = json.loads(capsys.readouterr().out)
        assert report["registry_path"] == registry_arg


def test_validate_cli_explicit_missing_registry_fails_fast(monkeypatch, capsys, tmp_path):
    """显式错误 --registry 必须失败，不得静默改用默认文件。"""
    missing_registry = str(tmp_path / "missing-noise-registry.yaml")
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_dataset_registry.py", "--registry", missing_registry],
    )

    assert validate_main() == 2
    captured = capsys.readouterr()
    assert "注册表文件不存在" in captured.err
    assert missing_registry in captured.err
    assert captured.out == ""


def test_validate_cli_config_missing_registry_fails_fast(monkeypatch, capsys, tmp_path):
    """配置推导出的错误 registry 也必须原路径失败，不得回退默认文件。"""
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        "registry_path: tools/noise_filter_pipeline/config/missing-noise.yaml\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_dataset_registry.py", "--config", str(config_path)],
    )

    assert validate_main() == 2
    captured = capsys.readouterr()
    assert "注册表文件不存在" in captured.err
    assert "tools/noise_filter_pipeline/config/missing-noise.yaml" in captured.err
    assert captured.out == ""
