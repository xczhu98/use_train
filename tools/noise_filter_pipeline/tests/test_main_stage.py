"""
Integration tests for the canonical ``noise_filter_pipeline.cli`` stage entry point.
"""
import json
import os
import tempfile
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from noise_filter_pipeline.core.manifest_io import AudioRecord, count_manifest, write_manifest
from noise_filter_pipeline.core.pipeline_config import UnifiedPipelineConfig
from noise_filter_pipeline.core.stage_runner import StageRunner
from noise_filter_pipeline.cli.main import _parse_stage_list

pytestmark = pytest.mark.pipeline


def test_pipeline_config_defaults_to_current_version():
    config = UnifiedPipelineConfig()
    assert config.pipeline_version == "4.1.0"
    assert config.output_root == "/data_nvme0n1/data/noise_filter/"
    assert config.registry_path == "tools/noise_filter_pipeline/config/noise.yaml"
    from_dict = UnifiedPipelineConfig.from_dict({})
    assert from_dict.pipeline_version == "4.1.0"
    assert from_dict.output_root == "/data_nvme0n1/data/noise_filter/"
    assert from_dict.registry_path == "tools/noise_filter_pipeline/config/noise.yaml"


def _make_silent_wav(path, duration=1.0, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav = np.random.randn(int(duration * sr)) * 0.001
    sf.write(path, wav, sr)


@pytest.mark.subprocess
def test_main_stage_a():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "datasets", "test")
        _make_silent_wav(os.path.join(dataset_dir, "a.wav"), duration=2.0)
        _make_silent_wav(os.path.join(dataset_dir, "b.wav"), duration=2.0)

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

        config_path = os.path.join(tmpdir, "pipeline.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(f"""
output_root: {os.path.join(tmpdir, "out")}
registry_path: {registry_path}
pipeline_version: "2.1.0"
stages:
  - A
""")

        use_train_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(use_train_root, "tools")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "noise_filter_pipeline.cli",
                "--stage",
                "A",
                "--config",
                config_path,
            ],
            cwd=use_train_root,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

        a_dir = os.path.join(tmpdir, "out", "stage_a")
        # per-dataset 输出：每个数据集一个 a_manifest_<id>.jsonl（此处 1 个数据集）
        manifest_files = [
            f for f in os.listdir(a_dir)
            if f.startswith("a_manifest_") and f.endswith(".jsonl")
        ]
        assert len(manifest_files) == 1
        assert count_manifest(a_dir) == 2


@pytest.mark.parametrize("value", ["A,A", "A,C", "B,A", "E", ""])
def test_parse_stage_list_rejects_non_contiguous_or_duplicate_stages(value):
    with pytest.raises(ValueError):
        _parse_stage_list(value)


def test_parse_stage_list_accepts_contiguous_stages():
    assert _parse_stage_list("B,C,D") == ["B", "C", "D"]


def test_stage_runner_rejects_legacy_single_manifest_in_upstream_dir():
    """上游目录残留遗留单文件（a_manifest.jsonl）时必须显式拒绝，不自动拆分/删除。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = os.path.join(tmpdir, "out")
        stage_a = os.path.join(output_root, "stage_a")
        os.makedirs(stage_a)
        # 完成 marker
        with open(os.path.join(stage_a, ".complete"), "w", encoding="utf-8") as f:
            json.dump({"stage": "A"}, f)
        legacy = os.path.join(stage_a, "a_manifest.jsonl")
        write_manifest(iter([
            AudioRecord(record_id="ds1__a", dataset_id="ds1", source_id="ds1__a",
                        audio={"path": "/tmp/a.wav", "sr": 16000, "duration": 1.0}),
        ]), legacy)

        cfg = UnifiedPipelineConfig(output_root=output_root, stages=["A", "B", "C", "D"])

        for dataset_filter in (None, ["ds1"]):
            with pytest.raises(RuntimeError, match="遗留单文件"):
                StageRunner(cfg, dataset_filter=dataset_filter)._resolve_input_manifest(
                    "B", output_root, None,
                )

        assert os.path.exists(legacy), "拒绝路径不得迁移/删除遗留单文件"
        assert not os.path.exists(os.path.join(stage_a, "a_manifest_ds1.jsonl"))

        # 人工删除遗留单文件后放行
        os.remove(legacy)
        assert StageRunner(cfg)._resolve_input_manifest("B", output_root, None) == stage_a


def test_stage_runner_rejects_legacy_single_manifest_in_explicit_input_dir():
    """显式传入的输入目录同样受遗留单文件检查约束。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stage_b = os.path.join(tmpdir, "stage_b")
        os.makedirs(stage_b)
        legacy = os.path.join(stage_b, "b_manifest.jsonl")
        write_manifest(iter([
            AudioRecord(record_id="ds1__a", dataset_id="ds1", source_id="ds1__a",
                        audio={"path": "/tmp/a.wav", "sr": 16000, "duration": 1.0}),
        ]), legacy)

        cfg = UnifiedPipelineConfig(output_root=tmpdir, stages=["A", "B", "C", "D"])
        with pytest.raises(RuntimeError, match="遗留单文件"):
            StageRunner(cfg)._resolve_input_manifest("C", tmpdir, stage_b)
        assert os.path.exists(legacy)
