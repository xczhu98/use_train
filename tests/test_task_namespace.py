"""任务命名空间的实际路由、资源边界与仓库收录回归检查。"""

from pathlib import Path
import subprocess
import sys

import pytest

from core.data.constants import normalize_data_mode
from tasks.registry import get_task_for_data_mode, iter_tasks

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def test_task_discovery_and_cli_import_remain_lazy():
    script = """
import sys
import tasks
import train
assert len(tasks.iter_tasks()) == 5
for task in tasks.iter_tasks():
    assert task.model_registry_module not in sys.modules
    assert task.training_entrypoint.rsplit('.', 1)[0] not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


@pytest.mark.parametrize("descriptor", iter_tasks(), ids=lambda task: task.name)
def test_task_modes_and_source_visibility(descriptor):
    for mode in descriptor.data_modes:
        assert get_task_for_data_mode(f" {mode.upper()} ") == descriptor
        assert normalize_data_mode(mode).task == descriptor.name
    assert (ROOT / descriptor.config_root).is_dir()
    assert not (ROOT / descriptor.name).exists()
    sources = sorted((ROOT / "tasks" / descriptor.name).rglob("*.py"))
    assert sources
    checked = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=ROOT, input="\n".join(str(p.relative_to(ROOT)) for p in sources) + "\n",
        text=True, capture_output=True,
    )
    assert checked.returncode == 1, checked.stdout + checked.stderr


@pytest.mark.parametrize(
    ("mode", "task"),
    [
        ("offline", "se"),
        ("offline_aec", "aec"),
        ("offline_tse", "tse"),
        ("offline_ss", "ss"),
    ],
)
def test_public_data_mode_routes_to_declared_task(mode, task):
    descriptor = get_task_for_data_mode(f" {mode.upper()} ")
    assert descriptor.name == task
    assert normalize_data_mode(mode).task == descriptor.name
    assert mode in descriptor.data_modes


def test_actual_tse_source_boundary_resolves_inside_repository():
    from tasks.tse.inference import artifacts, extractor
    from tasks.tse.models.speaker import config
    from core.utils.inference_artifacts import collect_source_tree_files

    assert artifacts._PROJECT_ROOT == ROOT
    assert extractor._PROJECT_ROOT == ROOT
    assert config._REPO_ROOT == ROOT
    files = collect_source_tree_files(artifacts._PROJECT_ROOT, artifacts.IMPLEMENTATION_PATHS)
    assert files["source:tasks/tse/inference/artifacts.py"] == Path(artifacts.__file__)
    assert "source:tasks/registry.py" in files
    assert "source:core/utils/inference_artifacts.py" in files


def test_task_resource_files_stay_ignored():
    paths = [f"tasks/{task}/data/example{extension}"
             for task in ("se", "aec", "tse", "ss", "ssl")
             for extension in (".pt", ".wav", ".scp", ".json")]
    checked = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"], cwd=ROOT,
        input="\n".join(paths) + "\n", text=True, capture_output=True,
    )
    assert checked.returncode == 0, checked.stderr
    assert set(checked.stdout.splitlines()) == set(paths)


@pytest.mark.parametrize("module_name,class_name,expected", [
    ("tasks.se.data.dynamic_dataset", "DynamicMixingDataset", "se"),
    ("tasks.tse.data.dynamic_dataset", "TSEDynamicMixingDataset", "tse"),
    ("tasks.ss.data.dynamic_dataset", "SSDynamicMixingDataset", "ss"),
])
def test_simulation_trace_keeps_task_identity_after_namespace_move(module_name, class_name, expected):
    import importlib

    dataset_cls = getattr(importlib.import_module(module_name), class_name)
    dataset = dataset_cls.__new__(dataset_cls)
    assert dataset._simulation_trace_task() == expected


def test_ssl_imports_resolve_package_modules_not_shadowed_files():
    import tasks.ssl.data as data_package
    import tasks.ssl.trainer as trainer_package

    assert Path(data_package.__file__).parent.name == "data"
    assert Path(trainer_package.__file__).parent.name == "trainer"
    assert hasattr(data_package, "SSLBatch")
    assert hasattr(data_package, "prepare_ssl_scene")
    assert hasattr(trainer_package, "SSLTrainer")
