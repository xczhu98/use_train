"""
Tests for core.stage_runner
"""
import os
import tempfile
import pytest

import numpy as np
import soundfile as sf

from noise_filter_pipeline.core.pipeline_config import UnifiedPipelineConfig
from noise_filter_pipeline.core.stage_runner import STAGE_MODULES, StageRunner
from noise_filter_pipeline.core.manifest_io import count_manifest

pytestmark = pytest.mark.pipeline


class _FailingOrchestrator:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, **_kwargs):
        raise RuntimeError("模拟 stage 失败")


class _CapturingOrchestrator:
    """捕获 run 输入的自定义 orchestrator。

    selected_files 是实例属性：类属性跨用例共享属于隐式全局状态，
    用例间会互相污染。instances 仅为断言收集，每个用例通过
    monkeypatch.setattr 重置为独立列表。"""

    instances = []

    def __init__(self, *_args, **_kwargs):
        self.selected_files = []
        _CapturingOrchestrator.instances.append(self)

    def run(self, *, input_manifest, output_dir, **_kwargs):
        self.selected_files = sorted(os.listdir(input_manifest))
        return output_dir


def _make_silent_wav(path, duration=1.0, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav = np.random.randn(int(duration * sr)) * 0.001
    sf.write(path, wav, sr)


def test_stage_runner_stage_a():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "datasets", "esc50")
        _make_silent_wav(os.path.join(dataset_dir, "ocean.wav"), duration=2.0)
        _make_silent_wav(os.path.join(dataset_dir, "rain.wav"), duration=2.0)

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: esc50
    name: ESC50
    enabled: true
    type: directory
    path: {dataset_dir}
    input_pattern: "**/*.wav"
    label_source: filename
    splits: [train]
    split_strategy: none
""")

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            registry_path=registry_path,
            stages=["A"],
        )
        runner = StageRunner(config)
        manifests = runner.run_pipeline(os.path.join(tmpdir, "out"))

        assert "A" in manifests
        assert count_manifest(manifests["A"]) == 2


def test_stage_rerun_invalidates_complete_marker_before_processor_runs(tmp_path, monkeypatch):
    output_root = str(tmp_path / "out")
    stage_dir = tmp_path / "out" / "stage_a"
    stage_dir.mkdir(parents=True)
    marker = stage_dir / ".complete"
    marker.write_text('{"stage": "A"}', encoding="utf-8")
    cfg = UnifiedPipelineConfig(output_root=output_root, stages=["A"])

    import importlib
    real_import = importlib.import_module

    def fake_import(name):
        if name.endswith("stage_a_ingestion"):
            return type("Module", (), {"StageAOrchestrator": _FailingOrchestrator})
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(RuntimeError, match="模拟 stage 失败"):
        StageRunner(cfg).run_stage("A", output_root)
    assert not marker.exists()


@pytest.mark.parametrize(
    ("stage_name", "previous_stage", "orchestrator_attr"),
    [("B", "A", "StageBOrchestrator"), ("C", "B", "StageCOrchestrator"), ("D", "C", "StageDOrchestrator")],
)
def test_stage_runner_filters_directory_input_for_independent_stage(
    tmp_path, monkeypatch, stage_name, previous_stage, orchestrator_attr
):
    """独立运行 B/C/D 时 --dataset 只暴露所选 per-dataset manifest。"""
    output_root = str(tmp_path / "out")
    upstream_dir = tmp_path / "out" / f"stage_{previous_stage.lower()}"
    upstream_dir.mkdir(parents=True)
    (upstream_dir / ".complete").write_text(
        f'{{"stage": "{previous_stage}"}}', encoding="utf-8"
    )
    (upstream_dir / f"{previous_stage.lower()}_manifest_ds1.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    (upstream_dir / f"{previous_stage.lower()}_manifest_ds2.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    cfg = UnifiedPipelineConfig(output_root=output_root, stages=["A", "B", "C", "D"])

    import importlib
    real_import = importlib.import_module

    def fake_import(name):
        if name == STAGE_MODULES[stage_name]:
            return type("Module", (), {orchestrator_attr: _CapturingOrchestrator})
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    # 每个用例独立收集列表，避免 instances 类属性跨用例累积
    monkeypatch.setattr(_CapturingOrchestrator, "instances", [])
    StageRunner(cfg, dataset_filter=["ds1"]).run_stage(stage_name, output_root)

    assert _CapturingOrchestrator.instances[-1].selected_files == [
        f"{previous_stage.lower()}_manifest_ds1.jsonl"
    ]


def test_stage_runner_dataset_selection_passes_strict_input_snapshot(tmp_path):
    """选择目录中的 manifest 必须可直接通过严格输入快照校验。"""
    from noise_filter_pipeline.core.stage_orchestrator import StageOrchestratorBase

    upstream_dir = tmp_path / "stage_a"
    selected_dir = tmp_path / "selected"
    upstream_dir.mkdir()
    selected_dir.mkdir()
    source = upstream_dir / "a_manifest_ds1.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    cfg = UnifiedPipelineConfig(output_root=str(tmp_path / "out"), stages=["A", "B"])

    selected_path = StageRunner(cfg, dataset_filter=["ds1"])._select_dataset_manifest_dir(
        str(upstream_dir), "A", str(selected_dir)
    )
    selected_file = selected_dir / source.name
    assert selected_path == str(selected_dir)
    assert selected_file.is_file()
    assert not selected_file.is_symlink()

    snapshot, files = StageOrchestratorBase._input_snapshot(
        selected_path, include_files=True
    )
    assert [path.name for path in files] == [source.name]
    assert snapshot[0][0]["name"] == source.name
    assert source.read_text(encoding="utf-8") == "{}\n"


def test_stage_runner_rejects_missing_dataset_manifest_for_independent_stage(tmp_path):
    """独立 B/C/D 运行时，所选数据集缺少上游 manifest 必须立即失败。"""
    output_root = str(tmp_path / "out")
    stage_a = tmp_path / "out" / "stage_a"
    stage_a.mkdir(parents=True)
    (stage_a / ".complete").write_text('{"stage": "A"}', encoding="utf-8")
    (stage_a / "a_manifest_ds1.jsonl").write_text("{}\n", encoding="utf-8")
    cfg = UnifiedPipelineConfig(output_root=output_root, stages=["A", "B"])

    with pytest.raises(FileNotFoundError, match="ds2"):
        StageRunner(cfg, dataset_filter=["ds1", "ds2"]).run_stage("B", output_root)


def test_stage_runner_cli_input_error_keeps_complete_marker(tmp_path):
    """P3 回归：缺上游 manifest 或 --dataset 拼错等纯 CLI 输入错误，
    不得撤销本 stage 已有完好产物的 .complete 完成态。"""
    output_root = str(tmp_path / "out")
    stage_b = tmp_path / "out" / "stage_b"
    stage_b.mkdir(parents=True)
    marker = stage_b / ".complete"
    marker.write_text('{"stage": "B"}', encoding="utf-8")
    cfg = UnifiedPipelineConfig(output_root=output_root, stages=["A", "B"])

    # 缺上游 manifest：stage_a 目录不存在 -> 报错且 .complete 保留
    with pytest.raises(FileNotFoundError, match="缺少输入 manifest"):
        StageRunner(cfg).run_stage("B", output_root)
    assert marker.exists(), "纯输入错误不得删除已有 .complete"

    # --dataset 拼错：上游存在但无匹配数据集 -> 报错且 .complete 保留
    stage_a = tmp_path / "out" / "stage_a"
    stage_a.mkdir(parents=True)
    (stage_a / ".complete").write_text('{"stage": "A"}', encoding="utf-8")
    (stage_a / "a_manifest_ds1.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="ds_typo"):
        StageRunner(cfg, dataset_filter=["ds_typo"]).run_stage("B", output_root)
    assert marker.exists(), "--dataset 拼错不得删除已有 .complete"


def test_stage_runner_dataset_filter_rejects_single_file_input(tmp_path):
    """单文件输入 + --dataset：无法按数据集筛选，报错须明确指出输入不是目录，
    而不是误导性的"缺少已选数据集 manifest、上游实际包含: 无"。"""
    output_root = str(tmp_path / "out")
    (tmp_path / "out").mkdir()
    single = tmp_path / "some_manifest.jsonl"
    single.write_text("{}\n", encoding="utf-8")
    cfg = UnifiedPipelineConfig(output_root=output_root, stages=["A", "B"])

    with pytest.raises(FileNotFoundError, match="不是目录"):
        StageRunner(cfg, dataset_filter=["ds1"]).run_stage(
            "B", output_root, input_manifest=str(single)
        )


def test_stage_runner_dataset_selection_failure_cleans_tempdir(tmp_path, monkeypatch):
    """_select_dataset_manifest_dir raise 时 TemporaryDirectory 必须显式清理。"""
    import tempfile as tempfile_mod
    from noise_filter_pipeline.core import stage_runner as sr

    created = []
    real_td = tempfile_mod.TemporaryDirectory

    class _TrackingTemporaryDirectory:
        def __init__(self, **kwargs):
            self._inner = real_td(**kwargs)
            self.name = self._inner.name
            self.cleaned = False
            created.append(self)

        def cleanup(self):
            self.cleaned = True
            self._inner.cleanup()

    monkeypatch.setattr(sr.tempfile, "TemporaryDirectory", _TrackingTemporaryDirectory)

    output_root = str(tmp_path / "out")
    stage_a = tmp_path / "out" / "stage_a"
    stage_a.mkdir(parents=True)
    (stage_a / ".complete").write_text('{"stage": "A"}', encoding="utf-8")
    (stage_a / "a_manifest_ds1.jsonl").write_text("{}\n", encoding="utf-8")
    cfg = UnifiedPipelineConfig(output_root=output_root, stages=["A", "B"])

    with pytest.raises(FileNotFoundError, match="ds_typo"):
        StageRunner(cfg, dataset_filter=["ds_typo"]).run_stage("B", output_root)
    assert len(created) == 1
    assert created[0].cleaned, "选择失败时必须显式 cleanup 临时目录"


@pytest.mark.parametrize("stages", [["A", "C"], ["B", "D"], ["A", "A"]])
def test_stage_runner_rejects_non_contiguous_pipeline_plan(tmp_path, stages):
    """非 CLI 调用也不得绕过相邻上游 stage 的完成态校验。"""
    cfg = UnifiedPipelineConfig(
        output_root=str(tmp_path / "out"),
        stages=["A", "B", "C", "D"],
    )

    with pytest.raises(ValueError, match="严格连续且不重复"):
        StageRunner(cfg).run_pipeline(cfg.output_root, stages=stages)

    assert not (tmp_path / "out").exists(), "非法计划不得创建或发布任何 stage 输出"
