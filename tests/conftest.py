"""顶层测试的集中 marker 分类与源码检查 fixture。

现有 ``tests/`` 仍为平铺布局。这里集中标记已核实的 contract 测试，
以及会启动真实子进程、DDP worker 或 ONNX Runtime 的资源测试，不搬动测试文件。
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from pathlib import Path
import sys

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _restore_project_root_import_priority() -> None:
    """测试收集任务侧 inference 目录时仍优先解析根 inference.py。"""
    project_root = str(_PROJECT_ROOT)
    sys.path[:] = [entry for entry in sys.path if entry != project_root]
    sys.path.insert(0, project_root)
    shadow_dirs = {
        str(path)
        for path in _PROJECT_ROOT.glob("tasks/*/inference")
        if path.is_dir()
    }
    sys.path[:] = [entry for entry in sys.path if entry not in shadow_dirs]
    loaded = sys.modules.get("inference")
    loaded_path = getattr(loaded, "__file__", "") if loaded is not None else ""
    if loaded is not None and "tasks" in Path(loaded_path).parts and "inference" in Path(loaded_path).parts:
        sys.modules.pop("inference", None)


@pytest.fixture(autouse=True)
def restore_project_root_import_priority() -> None:
    """每个测试前恢复根模块优先级，避免任务 inference 包遮蔽 inference.py。"""
    _restore_project_root_import_priority()

_CONTRACT_TESTS = {
    "tests/test_configs.py": None,
}

_SUBPROCESS_TESTS = {
    "tests/test_ddp_val_padding_integration.py": {
        "test_ddp_val_padding_end_to_end",
    },
    "tests/test_eval_pipeline_provenance.py": {
        "test_default_paths_follow_explicit_environment_root",
        "test_default_paths_use_ignored_placeholder_without_machine_absolute_path",
    },
    "tests/test_generate_simulated_rirs.py": {"test_cli_smoke"},
    "tests/test_model_registry_contract.py": {
        "test_core_registry_facade_keeps_task_packages_lazy",
        "test_task_model_package_import_does_not_load_concrete_models",
    },
    "tests/test_noise_manifest_adapter.py": {
        "test_noise_manifest_adapter_serializes_writers_for_the_same_output_pair",
    },
    "tests/test_onnx_export_help.py": {
        "test_onnx_export_help",
    },
    "tests/test_onnx_export_verify_nan.py": {
        "test_verify_onnx_fail_closed_for_nan",
    },
    "tests/test_prepare_data_manifest_contracts.py": {
        "test_cli_json_output_and_failure_exit",
    },
    "tests/test_trainer_entry_scheduler_contract.py": {
        "test_acquisition_guard_unlock_failure_preserves_original_aggregated_error",
        "test_all_rank_consumer_acquisition_failure_preserves_concurrent_active_job",
        "test_consumer_atexit_cleans_last_snapshot",
        "test_consumer_atexit_releases_owner_transition_guard",
        "test_gloo_consumer_acquisition_failure_is_aggregated_and_cleaned",
        "test_gloo_consumer_completion_failure_is_aggregated_and_cleaned",
        "test_gloo_original_trainer_controls_resume_hooks_and_nested_aliases",
        "test_gloo_post_super_failure_preserves_peer_and_cleans_shared_snapshot",
        "test_gloo_resume_preview_accepts_rank_local_paths_with_same_content",
        "test_gloo_resume_preview_aggregates_rank_local_not_found",
        "test_gloo_resume_preview_aggregates_rank_local_parse_error",
        "test_gloo_resume_preview_resolves_each_rank_local_cwd",
        "test_gloo_resume_snapshot_survives_until_all_trainers_finish_loading",
        "test_gloo_resume_uses_rank_local_snapshots_with_matching_content",
        "test_gloo_sampler_stage_aggregates_rank_one_failure_before_cleanup",
        "test_gloo_sampler_stage_aggregates_rank_zero_failure_before_cleanup",
        "test_resume_consumer_os_exit_is_reclaimed_by_next_freeze",
        "test_resume_snapshot_scan_preserves_active_lease_and_reclaims_stale",
        "test_resume_snapshot_scan_reclaims_lease_left_by_abrupt_exit",
    },
    "tests/test_tse_trainer_enhancements.py": {
        "test_two_iteration_gloo_ddp_handles_rank_asymmetric_all_ignore",
        "test_gloo_external_speaker_failure_is_unified_without_hang",
        "test_gloo_external_speaker_identity_mismatch_is_unified_without_hang",
        "test_gloo_external_speaker_identity_sync_runs_once",
    },
    "tests/test_verify_3way_checkpoint_gate.py": {
        "test_verify_scripts_help",
    },
}

_DDP_TESTS = {
    "tests/test_ddp_collective_consistency.py": None,
    "tests/test_ddp_val_padding_integration.py": None,
    "tests/test_sampler_resume_determinism.py": {
        "test_gloo_validation_gate_precedes_collective_and_train_full_paths_succeed",
    },
    "tests/test_trainer_entry_scheduler_contract.py": {
        name
        for name in _SUBPROCESS_TESTS["tests/test_trainer_entry_scheduler_contract.py"]
        if "gloo" in name or "all_rank" in name
    }
    | {"test_acquisition_guard_unlock_failure_preserves_original_aggregated_error"},
    "tests/test_tse_trainer_enhancements.py": {
        "test_two_iteration_gloo_ddp_handles_rank_asymmetric_all_ignore",
        "test_gloo_external_speaker_failure_is_unified_without_hang",
        "test_gloo_external_speaker_identity_mismatch_is_unified_without_hang",
        "test_gloo_external_speaker_identity_sync_runs_once",
    },
}

_ORT_TESTS = {
    "tests/test_onnx_export_verify_nan.py": {
        "test_verify_onnx_fail_closed_for_nan",
    },
    "tests/test_onnx_runtime_spec.py": {
        "test_real_ort_streaming_abi_integration",
    },
}


def _matches(
    item: pytest.Item,
    mapping: dict[str, set[str] | None],
) -> bool:
    path = item.nodeid.partition("::")[0]
    if path not in mapping:
        return False
    names = mapping[path]
    test_name = getattr(item, "originalname", None) or item.name
    return names is None or test_name in names


def _add_marker(
    items: Iterable[pytest.Item],
    marker: str,
    mapping: dict[str, set[str] | None],
) -> None:
    for item in items:
        if _matches(item, mapping) and item.get_closest_marker(marker) is None:
            item.add_marker(getattr(pytest.mark, marker))


_DDP_SPAWN_TIMEOUT_SEC = 90
_DDP_SPAWN_REAP_TIMEOUT_SEC = 10


def spawn_ddp_workers(
    worker,
    *,
    args=(),
    nprocs: int = 2,
    timeout_sec: float = _DDP_SPAWN_TIMEOUT_SEC,
) -> None:
    """spawn DDP worker 并在超时内 join；挂死时终止全部子进程后让测试失败。

    默认 unit 套件没有 pytest-timeout 兜底，collective 序列不一致会让
    ``join=True`` 永久阻塞。``ProcessContext.join(timeout)`` 在任一进程退出时
    即返回，False 仅表示还有进程未结束，因此需要循环调用直到全部 join；
    超时先 terminate 再 kill 存活进程并 join 回收，不留僵尸进程。
    """
    import time

    import torch.multiprocessing as mp

    def drain(join_timeout: float) -> None:
        """回收已终止的子进程；join 会把 terminate 的退出码转成异常，此处不算失败。"""
        try:
            context.join(join_timeout)
        except Exception:
            pass

    context = mp.spawn(worker, args=tuple(args), nprocs=nprocs, join=False)
    deadline = time.monotonic() + timeout_sec
    while not context.join(max(0.0, deadline - time.monotonic())):
        if time.monotonic() < deadline:
            continue
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        drain(_DDP_SPAWN_REAP_TIMEOUT_SEC)
        for process in context.processes:
            if process.is_alive():
                process.kill()
        drain(_DDP_SPAWN_REAP_TIMEOUT_SEC)
        raise TimeoutError(
            f"{getattr(worker, '__name__', repr(worker))} 未在 "
            f"{timeout_sec:g}s 内结束，已终止全部子进程："
            "DDP collective 序列可能不一致（挂死回归）"
        )


@pytest.fixture()
def parse_python_ast() -> Callable[[str | Path], ast.Module]:
    """解析项目内 Python 文件，保留文件名以改善 SyntaxError 定位。"""

    def parse(path: str | Path) -> ast.Module:
        source_path = Path(path)
        if not source_path.is_absolute():
            source_path = _PROJECT_ROOT / source_path
        source_path = source_path.resolve()
        source_path.relative_to(_PROJECT_ROOT)
        return ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )

    return parse


@pytest.fixture()
def preserve_tensorboard_module_state():
    """恢复测试前的 TensorBoard 模块与 ``torch.utils`` 属性状态。"""
    module_name = "torch.utils.tensorboard"
    missing = object()
    module_before = sys.modules.get(module_name, missing)
    torch_utils = sys.modules.get("torch.utils")
    attribute_before = (
        getattr(torch_utils, "tensorboard", missing)
        if torch_utils is not None
        else missing
    )

    yield

    if module_before is missing:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = module_before

    torch_utils = sys.modules.get("torch.utils")
    if torch_utils is not None:
        if attribute_before is missing:
            if hasattr(torch_utils, "tensorboard"):
                delattr(torch_utils, "tensorboard")
        else:
            torch_utils.tensorboard = attribute_before


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """在默认 ``-m`` 过滤前补充语义与资源 marker。"""
    _restore_project_root_import_priority()
    _add_marker(items, "contract", _CONTRACT_TESTS)
    _add_marker(items, "subprocess", _SUBPROCESS_TESTS)
    _add_marker(items, "subprocess", _DDP_TESTS)
    _add_marker(items, "ddp", _DDP_TESTS)
    _add_marker(items, "ort", _ORT_TESTS)
