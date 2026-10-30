"""Generation-3 Step3 MOS 纯执行计划测试。"""

from __future__ import annotations

import builtins
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from audio_filter_pipeline.config import PipelineConfig
from audio_filter_pipeline.processors.stages_v3 import mos_execution
from audio_filter_pipeline.processors.stages_v3.mos_execution import (
    MosAssetSpec,
    MosExecutionPlan,
)


pytestmark = pytest.mark.unit


def _quality_profiles(scoreq: float = 4.0, utmos: float = 3.8):
    return {
        profile: {"thresholds": {"scoreq": scoreq, "utmos": utmos}}
        for profile in ("quality_balanced", "quality_high", "quality_premium")
    }


def _config(*, threshold: float = 4.0, **policy_overrides):
    global_policy = {
        "mos_models": ["scoreq", "utmos"],
        "thresholds": {"scoreq": threshold, "utmos": 3.8},
        "min_mos_models": 2,
        "mos_fusion": "min",
        "early_exit": None,
    }
    global_policy.update(policy_overrides)
    return PipelineConfig(
        pipeline_generation=3,
        output_root="/does/not/need/to/exist/output",
        datasets={
            "alpha": {
                "input_root": "/does/not/need/to/exist/input",
                "language": "en",
                "category": "read_speech",
                "quality_profile": "quality_high",
                "speaker": {"source": "none", "required": False},
            },
        },
        selection_policies={
            "global": global_policy,
            "quality_profiles": _quality_profiles(scoreq=threshold),
        },
        scoreq_model_dir="relative/scoreq-model",
        utmos_model_path="relative/utmos-model.pt",
        utmos_model_version="utmos-test",
        utmos_model_sha256="a" * 64,
        # 未选模型可携带不可用路径，planning 仍不得访问或纳入 assets。
        dnsmos_model_dir="/must/not/be/read/dnsmos",
        nisqa_model_path="/must/not/be/read/nisqa.tar",
        step3_gpu_ids=(3, 7),
        step3_worker_idle_timeout_seconds=123.5,
        step3_worker_hard_timeout_seconds=456.0,
        step3_worker_stderr_tail_bytes=2048,
        speaker_consistency={"enabled": False},
    )


MODEL_FIELDS = {
    "scoreq": ("scoreq_model_dir",),
    "urgent_pk": (
        "urgent_pk_checkpoint", "urgent_pk_source_dir", "urgent_pk_encoder",
        "urgent_pk_backbone", "urgent_pk_score_range",
        "urgent_pk_expected_sha256", "urgent_pk_expected_commit",
    ),
    "dnsmos": ("dnsmos_model_dir",),
    "nisqa": (
        "nisqa_model_path", "nisqa_model_version", "nisqa_model_sha256",
    ),
    "utmos": (
        "utmos_model_path", "utmos_model_version", "utmos_model_sha256",
    ),
}
MODEL_VALUES = {
    "scoreq_model_dir": "assets/scoreq",
    "urgent_pk_checkpoint": "assets/urgent/model.pt",
    "urgent_pk_source_dir": "assets/urgent/source",
    "urgent_pk_encoder": "wav2vec2",
    "urgent_pk_backbone": "cnn",
    "urgent_pk_score_range": {"min": 1.0, "max": 5.0},
    "urgent_pk_expected_sha256": "b" * 64,
    "urgent_pk_expected_commit": "0123456789abcdef",
    "dnsmos_model_dir": "assets/dnsmos",
    "nisqa_model_path": "assets/nisqa/model.tar",
    "nisqa_model_version": "nisqa-v2",
    "nisqa_model_sha256": "c" * 64,
    "utmos_model_path": "assets/utmos/model.pt",
    "utmos_model_version": "utmos-v1",
    "utmos_model_sha256": "d" * 64,
}
CPU_TOPOLOGY = {
    "kind": "single_process",
    "device": "cpu",
}
GPU_TOPOLOGY = {
    "kind": "gpu_process_pool",
    "device": "cuda",
    "gpu_ids": [3, 7],
    "worker_idle_timeout_seconds": 123.5,
    "worker_hard_timeout_seconds": 456.0,
    "worker_stderr_tail_bytes": 2048,
    "cpu_fallback": False,
}


class _GuardedConfigSource:
    """只开放当前选择所需字段，任何越界读取立即失败。"""

    def __init__(self, selected_models, values=None):
        self._selected_models = tuple(selected_models)
        self._values = dict(MODEL_VALUES if values is None else values)
        self._values.update({
            "step3_gpu_ids": (3, 7),
            "step3_worker_idle_timeout_seconds": 123.5,
            "step3_worker_hard_timeout_seconds": 456.0,
            "step3_worker_stderr_tail_bytes": 2048,
        })

    def selected_mos_models(self):
        return self._selected_models

    def __getattr__(self, name):
        if name in self._values:
            return self._values[name]
        raise AssertionError(f"MOS planning must not access config attribute {name!r}")


def _expected_assets(models):
    expected = {
        "scoreq": {
            "paths": {"model_dir": MODEL_VALUES["scoreq_model_dir"]},
            "settings": {},
        },
        "urgent_pk": {
            "paths": {
                "checkpoint": MODEL_VALUES["urgent_pk_checkpoint"],
                "source_dir": MODEL_VALUES["urgent_pk_source_dir"],
            },
            "settings": {
                "encoder": MODEL_VALUES["urgent_pk_encoder"],
                "backbone": MODEL_VALUES["urgent_pk_backbone"],
                "score_range": MODEL_VALUES["urgent_pk_score_range"],
                "expected_sha256": MODEL_VALUES["urgent_pk_expected_sha256"],
                "expected_commit": MODEL_VALUES["urgent_pk_expected_commit"],
            },
        },
        "dnsmos": {
            "paths": {"model_dir": MODEL_VALUES["dnsmos_model_dir"]},
            "settings": {},
        },
        "nisqa": {
            "paths": {"model_path": MODEL_VALUES["nisqa_model_path"]},
            "settings": {
                "model_version": MODEL_VALUES["nisqa_model_version"],
                "model_sha256": MODEL_VALUES["nisqa_model_sha256"],
            },
        },
        "utmos": {
            "paths": {"model_path": MODEL_VALUES["utmos_model_path"]},
            "settings": {
                "model_version": MODEL_VALUES["utmos_model_version"],
                "model_sha256": MODEL_VALUES["utmos_model_sha256"],
            },
        },
    }
    return {model: expected[model] for model in models}


def _asset_snapshots(plan):
    return {
        asset.model: {
            "paths": dict(asset.as_mapping()["paths"]),
            "settings": dict(asset.as_mapping()["settings"]),
        }
        for asset in plan.assets
    }


def _topology_snapshot(plan):
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in plan.topology.items()
    }


def _urgent_asset(*, score_range=MODEL_VALUES["urgent_pk_score_range"]):
    return MosAssetSpec(
        "urgent_pk",
        {
            "checkpoint": MODEL_VALUES["urgent_pk_checkpoint"],
            "source_dir": MODEL_VALUES["urgent_pk_source_dir"],
        },
        {
            "encoder": MODEL_VALUES["urgent_pk_encoder"],
            "backbone": MODEL_VALUES["urgent_pk_backbone"],
            "score_range": score_range,
            "expected_sha256": MODEL_VALUES["urgent_pk_expected_sha256"],
            "expected_commit": MODEL_VALUES["urgent_pk_expected_commit"],
        },
    )


def test_five_model_mapping_and_topology_are_exact():
    models = tuple(MODEL_FIELDS)
    plan = MosExecutionPlan.from_config(_GuardedConfigSource(models), use_gpu=True)

    assert plan.selected_models == models
    assert tuple(asset.model for asset in plan.assets) == models
    assert _asset_snapshots(plan) == _expected_assets(models)
    assert _topology_snapshot(plan) == GPU_TOPOLOGY
    assert "worker_timeout_seconds" not in plan.topology


@pytest.mark.parametrize("selected_model", tuple(MODEL_FIELDS))
def test_single_model_selection_accesses_only_that_models_attributes(selected_model):
    selected_fields = set(MODEL_FIELDS[selected_model])
    values = {
        field: value for field, value in MODEL_VALUES.items()
        if field in selected_fields
    }
    plan = MosExecutionPlan.from_config(
        _GuardedConfigSource((selected_model,), values),
    )

    assert plan.selected_models == (selected_model,)
    assert _asset_snapshots(plan) == _expected_assets((selected_model,))


@pytest.mark.parametrize(
    ("selected_model", "field"),
    tuple(
        (model, field)
        for model, fields in MODEL_FIELDS.items()
        for field in fields
        if field != "urgent_pk_score_range"
    ),
)
@pytest.mark.parametrize("blank_value", ("", " \t\n"), ids=("empty", "whitespace"))
def test_each_required_asset_string_field_rejects_blank_values(
    selected_model, field, blank_value,
):
    values = {
        name: MODEL_VALUES[name]
        for name in MODEL_FIELDS[selected_model]
    }
    values[field] = blank_value

    with pytest.raises((TypeError, ValueError)):
        MosExecutionPlan.from_config(
            _GuardedConfigSource((selected_model,), values),
        )


def test_planning_never_accesses_speaker_or_decision_attributes():
    plan = MosExecutionPlan.from_config(
        _GuardedConfigSource(("scoreq",), {
            "scoreq_model_dir": MODEL_VALUES["scoreq_model_dir"],
        }),
    )

    assert plan.assets[0].as_mapping() == {
        "model": "scoreq",
        "paths": {"model_dir": MODEL_VALUES["scoreq_model_dir"]},
        "settings": {},
    }


def test_gpu_topology_is_saved_without_starting_workers():
    plan = MosExecutionPlan.from_config(_config(), use_gpu=True)

    assert dict(plan.topology) == {
        **GPU_TOPOLOGY,
        "gpu_ids": (3, 7),
    }
    assert _topology_snapshot(plan) == GPU_TOPOLOGY
    assert "worker_timeout_seconds" not in plan.topology


@pytest.mark.parametrize(
    "overrides",
    ({"threshold": 4.5}, {"min_mos_models": 1}),
    ids=("thresholds", "quorum"),
)
def test_each_decision_field_is_independent_from_execution_plan(overrides):
    baseline = MosExecutionPlan.from_config(_config())
    changed = MosExecutionPlan.from_config(_config(**overrides))

    assert baseline == changed


def test_asset_model_order_and_scoring_settings_change_execution_plan():
    models = tuple(MODEL_FIELDS)
    baseline = MosExecutionPlan.from_config(_GuardedConfigSource(models))

    changed_path_values = dict(MODEL_VALUES, scoreq_model_dir="other/scoreq")
    changed_path = MosExecutionPlan.from_config(
        _GuardedConfigSource(models, changed_path_values),
    )
    reversed_order = MosExecutionPlan.from_config(
        _GuardedConfigSource(tuple(reversed(models))),
    )
    changed_range_values = dict(MODEL_VALUES)
    changed_range_values["urgent_pk_score_range"] = {"min": 0.0, "max": 1.0}
    changed_range = MosExecutionPlan.from_config(
        _GuardedConfigSource(models, changed_range_values),
    )

    assert baseline != changed_path
    assert baseline != reversed_order
    assert baseline != changed_range


def test_plan_and_asset_state_are_deeply_immutable_and_defensively_copied():
    score_range = {"min": 1.0, "max": 5.0}
    paths = {"checkpoint": "model.pt", "source_dir": "source"}
    settings = {
        "encoder": "wav2vec2",
        "backbone": "cnn",
        "score_range": score_range,
        "expected_sha256": "e" * 64,
        "expected_commit": "abcdef",
    }
    topology = dict(GPU_TOPOLOGY)
    asset = MosAssetSpec("urgent_pk", paths, settings)
    plan = MosExecutionPlan(("urgent_pk",), (asset,), topology)

    paths["checkpoint"] = "mutated.pt"
    score_range["min"] = 0.0
    topology["gpu_ids"].append(9)
    assert plan.assets[0].paths["checkpoint"] == "model.pt"
    assert dict(plan.assets[0].settings["score_range"]) == {
        "min": 1.0, "max": 5.0,
    }
    assert plan.topology["gpu_ids"] == (3, 7)

    with pytest.raises(TypeError):
        plan.assets[0].settings["score_range"]["min"] = 0.0
    with pytest.raises(TypeError):
        plan.topology["device"] = "cpu"

    snapshot = plan.assets[0].as_mapping()
    snapshot["settings"]["score_range"]["min"] = 0.0
    assert plan.assets[0].as_mapping()["settings"]["score_range"]["min"] == 1.0


def test_plan_snapshots_are_strict_json_values():
    plan = MosExecutionPlan.from_config(
        _GuardedConfigSource(tuple(MODEL_FIELDS)),
        use_gpu=True,
    )

    assets = [asset.as_mapping() for asset in plan.assets]
    topology = _topology_snapshot(plan)
    assert json.loads(json.dumps(assets, allow_nan=False)) == assets
    assert json.loads(json.dumps(topology, allow_nan=False)) == topology


@pytest.mark.parametrize(
    ("score_range", "expected_payload"),
    (
        ("mos_1_to_5", "mos_1_to_5"),
        ([1, 5], [1, 5]),
        ((1, 5), [1, 5]),
        ({"min": 1, "max": 5}, {"min": 1, "max": 5}),
    ),
    ids=("named-range", "list", "tuple", "exact-mapping"),
)
def test_urgent_score_range_legal_matrix(score_range, expected_payload):
    asset = _urgent_asset(score_range=score_range)
    plan = MosExecutionPlan(("urgent_pk",), (asset,), CPU_TOPOLOGY)

    assert plan.assets[0].as_mapping()["settings"]["score_range"] == expected_payload


@pytest.mark.parametrize(
    "score_range",
    (
        None,
        False,
        "",
        " \t\n",
        "unknown_range",
        {"minimum": 1, "max": 5},
        {"min": 1, "max": 5, "extra": 3},
        [],
        [1],
        [1, 5, 6],
        (),
        (1,),
        (1, 5, 6),
        ["not-a-number", 5],
        {"min": 1, "max": "not-a-number"},
        [float("nan"), 5],
        {"min": 1, "max": float("inf")},
        [5, 5],
        {"min": 5, "max": 1},
    ),
    ids=(
        "none", "bool", "empty-string", "blank-string", "unknown-string",
        "wrong-mapping-keys", "extra-mapping-key", "empty-list",
        "one-item-list", "three-item-list", "empty-tuple", "one-item-tuple",
        "three-item-tuple", "nonnumeric-list", "nonnumeric-mapping", "nan",
        "inf", "equal", "descending",
    ),
)
def test_urgent_score_range_illegal_matrix_fails_closed(score_range):
    with pytest.raises((TypeError, ValueError)):
        _urgent_asset(score_range=score_range)


@pytest.mark.parametrize(
    "bad_value",
    (b"bytes", {1: "non-string key"}),
)
def test_non_strict_json_asset_settings_fail_closed(bad_value):
    with pytest.raises((TypeError, ValueError)):
        _urgent_asset(score_range=bad_value)


@pytest.mark.parametrize(
    "topology",
    (
        {key: value for key, value in CPU_TOPOLOGY.items() if key != "device"},
        {**CPU_TOPOLOGY, "unexpected": True},
        {**CPU_TOPOLOGY, "kind": "thread_pool"},
        {**CPU_TOPOLOGY, "device": "cuda"},
        {**CPU_TOPOLOGY, "gpu_ids": [0]},
        {**GPU_TOPOLOGY, "gpu_ids": []},
        {**GPU_TOPOLOGY, "gpu_ids": [3, 3]},
        {**GPU_TOPOLOGY, "cpu_fallback": True},
        {**GPU_TOPOLOGY, "worker_idle_timeout_seconds": float("inf")},
        {**GPU_TOPOLOGY, "worker_hard_timeout_seconds": 0},
        {**GPU_TOPOLOGY, "worker_timeout_seconds": 123.5},
        {**GPU_TOPOLOGY, "worker_stderr_tail_bytes": 0},
    ),
)
def test_illegal_topology_fails_closed(topology):
    with pytest.raises((TypeError, ValueError)):
        MosExecutionPlan(
            ("scoreq",),
            (MosAssetSpec("scoreq", {"model_dir": "assets/scoreq"}),),
            topology,
        )


@pytest.mark.parametrize(
    ("model", "paths", "settings"),
    (
        ("scoreq", {}, {}),
        ("scoreq", {"model_dir": "assets/scoreq", "extra": "x"}, {}),
        ("dnsmos", {"model_dir": None}, {}),
        ("nisqa", {"model_path": "nisqa.tar"}, {"model_version": "v1"}),
        ("utmos", {"model_path": "utmos.pt"}, {
            "model_version": "v1", "model_sha256": "f" * 64, "extra": True,
        }),
        ("urgent_pk", {"checkpoint": "model.pt", "source_dir": "source"}, {
            "encoder": "wav2vec2", "backbone": "cnn",
            "score_range": {"minimum": 1.0, "maximum": 5.0},
            "expected_sha256": "f" * 64,
        }),
    ),
)
def test_missing_extra_or_non_string_asset_fields_fail_closed(
    model, paths, settings,
):
    with pytest.raises((TypeError, ValueError)):
        MosAssetSpec(model, paths, settings)


def test_missing_selected_model_config_field_fails_closed():
    class MissingVersionSource:
        nisqa_model_path = "nisqa.tar"
        nisqa_model_sha256 = "f" * 64
        step3_gpu_ids = ()
        step3_worker_idle_timeout_seconds = 10.0
        step3_worker_stderr_tail_bytes = 1024

        @staticmethod
        def selected_mos_models():
            return ("nisqa",)

    with pytest.raises((AttributeError, TypeError, ValueError)):
        MosExecutionPlan.from_config(MissingVersionSource())


def test_plan_rejects_noncanonical_public_constructor_inputs():
    asset = MosAssetSpec("scoreq", {"model_dir": "assets/scoreq"})
    with pytest.raises(TypeError):
        MosExecutionPlan(["scoreq"], (asset,), CPU_TOPOLOGY)
    with pytest.raises(TypeError):
        MosExecutionPlan(("scoreq",), [asset], CPU_TOPOLOGY)


def test_plan_rejects_assets_that_do_not_exactly_match_selection():
    with pytest.raises(ValueError, match="exactly match"):
        MosExecutionPlan(
            selected_models=("scoreq",),
            assets=(MosAssetSpec("dnsmos", {"model_dir": "unused"}),),
            topology=CPU_TOPOLOGY,
        )


def test_planning_does_not_call_guarded_file_hash_dynamic_import_or_popen_apis(
    monkeypatch,
):
    config_source = _GuardedConfigSource(("scoreq",), {
        "scoreq_model_dir": "opaque/model/path",
    })

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pure MOS planning performed a forbidden operation")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(os, "stat", forbidden)
        patch.setattr(os.path, "exists", forbidden)
        patch.setattr(hashlib, "sha256", forbidden)
        patch.setattr(importlib, "import_module", forbidden)
        patch.setattr(subprocess, "Popen", forbidden)

        plan = MosExecutionPlan.from_config(config_source, use_gpu=True)

    assert plan.selected_models == ("scoreq",)
    assert plan.assets[0].as_mapping()["paths"] == {
        "model_dir": "opaque/model/path",
    }


def test_module_source_imports_only_standard_library_and_mos_contract():
    source_path = Path(mos_execution.__file__)
    probe = textwrap.dedent(
        """
        import ast
        import sys
        from pathlib import Path

        source_path = Path(sys.argv[1])
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"), filename=str(source_path),
        )
        imported_roots = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_roots.update(
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        allowed = {
            "__future__", "math", "collections", "dataclasses", "types", "typing",
            "mos_contract",
        }
        unexpected = sorted(imported_roots - allowed)
        if unexpected:
            raise AssertionError(
                f"imports outside lightweight allowlist: {unexpected!r}"
            )
        print("OK")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(source_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "OK\n"


def test_module_execution_does_not_call_guarded_file_hash_dynamic_import_or_popen_apis():
    source_path = Path(mos_execution.__file__)
    contract_path = source_path.parents[1] / "mos_contract.py"
    probe = textwrap.dedent(
        """
        import builtins
        import hashlib
        import importlib
        import os
        import subprocess
        import sys
        import types
        from pathlib import Path

        source_path = Path(sys.argv[1])
        contract_path = Path(sys.argv[2])
        code = compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec")
        contract_code = compile(
            contract_path.read_text(encoding="utf-8"), str(contract_path), "exec",
        )

        def forbidden(*_args, **_kwargs):
            raise AssertionError("module execution performed forbidden side effect")

        builtins.open = forbidden
        os.stat = forbidden
        os.path.exists = forbidden
        hashlib.sha256 = forbidden
        importlib.import_module = forbidden
        subprocess.Popen = forbidden
        for name in ("isolated", "isolated.processors", "isolated.processors.stages_v3"):
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        contract = types.ModuleType("isolated.processors.mos_contract")
        contract.__file__ = str(contract_path)
        contract.__package__ = "isolated.processors"
        sys.modules[contract.__name__] = contract
        exec(contract_code, contract.__dict__)
        module = types.ModuleType("isolated.processors.stages_v3.mos_execution")
        module.__file__ = str(source_path)
        module.__package__ = "isolated.processors.stages_v3"
        sys.modules[module.__name__] = module
        exec(code, module.__dict__)
        print("OK")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(source_path), str(contract_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "OK\n"
