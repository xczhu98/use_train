"""轻量 canonical MOS contract 的 parity 与导入边界测试。"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from audio_filter_pipeline import config
from audio_filter_pipeline.processors import mos_contract
from audio_filter_pipeline.processors.stages_v3 import (
    mos_asset_preflight,
    mos_execution,
    mos_gpu_job,
    mos_pool,
    mos_runtime,
    select,
)
from audio_filter_pipeline.processors.workers import step3_mos_worker


pytestmark = pytest.mark.unit


EXPECTED_MODEL_ORDER = ("scoreq", "urgent_pk", "dnsmos", "nisqa", "utmos")
# 生产常量名暂保留 MOS_SHIPPED_MODELS，但其语义是 legacy/direct helper 默认，
# shipped YAML 的实际 route 由 selection_policies 决定。
EXPECTED_LEGACY_DIRECT_MODELS = EXPECTED_MODEL_ORDER[:3]
EXPECTED_REQUEST_OPTION_FIELDS = {
    "scoreq": "scoreq_options",
    "urgent_pk": "urgent_options",
    "dnsmos": "dnsmos_options",
    "nisqa": "nisqa_options",
    "utmos": "utmos_options",
}


def test_contract_model_order_thresholds_and_module_aliases_have_exact_parity():
    assert mos_contract.MOS_MODEL_ORDER == EXPECTED_MODEL_ORDER
    assert mos_contract.MOS_SHIPPED_MODELS == EXPECTED_LEGACY_DIRECT_MODELS
    assert tuple(mos_contract.MOS_THRESHOLD_DIMENSIONS) == EXPECTED_MODEL_ORDER
    assert mos_contract.MOS_THRESHOLD_DIMENSIONS == {
        "scoreq": (),
        "urgent_pk": (),
        "dnsmos": ("ovrl", "sig", "bak", "p808"),
        "nisqa": (),
        "utmos": (),
    }
    assert tuple(config._SELECTION_MOS_SPECS) == EXPECTED_MODEL_ORDER
    assert config._V3_LEGACY_DEFAULT_ATTRIBUTES["mos_types"] == list(
        EXPECTED_LEGACY_DIRECT_MODELS
    )
    assert {
        model: spec.validate_threshold
        for model, spec in config._SELECTION_MOS_SPECS.items()
    } == {
        model: dimensions or None
        for model, dimensions in mos_contract.MOS_THRESHOLD_DIMENSIONS.items()
    }
    assert mos_execution._SUPPORTED_MODELS is mos_contract.MOS_MODEL_ORDER
    assert mos_asset_preflight._LEGACY_MODEL_ORDER is mos_contract.MOS_MODEL_ORDER
    assert mos_runtime._MODEL_ORDER is mos_contract.MOS_MODEL_ORDER
    assert mos_pool._MODELS is mos_contract.MOS_SHIPPED_MODELS
    assert step3_mos_worker._MODELS is mos_contract.MOS_SHIPPED_MODELS
    assert select._SHIPPED_MOS_MODELS is mos_contract.MOS_SHIPPED_MODELS
    assert select._DNSMOS_DIMENSIONS is mos_contract.MOS_THRESHOLD_DIMENSIONS["dnsmos"]


def test_asset_runtime_and_provenance_field_contracts_have_exact_parity():
    assert tuple(mos_contract.MOS_ASSET_PATH_FIELDS) == EXPECTED_MODEL_ORDER
    assert tuple(mos_contract.MOS_ASSET_SETTING_FIELDS) == EXPECTED_MODEL_ORDER
    assert tuple(mos_contract.MOS_RUNTIME_OPTION_FIELDS) == EXPECTED_MODEL_ORDER
    assert tuple(mos_contract.MOS_ASSET_PROVENANCE_FIELDS) == EXPECTED_MODEL_ORDER
    assert mos_contract.MOS_ASSET_PROVENANCE_FIELDS["urgent_pk"] == {
        "checkpoint_sha256", "source_commit", "source_package_sha256",
    }
    assert dict(mos_contract.MOS_MODEL_REQUEST_OPTION_FIELDS) == (
        EXPECTED_REQUEST_OPTION_FIELDS
    )
    assert mos_execution._MODEL_SCHEMAS == {
        model: (
            mos_contract.MOS_ASSET_PATH_FIELDS[model],
            mos_contract.MOS_ASSET_SETTING_FIELDS[model],
        )
        for model in EXPECTED_MODEL_ORDER
    }
    assert mos_asset_preflight._MODEL_OPTION_FIELDS is (
        mos_contract.MOS_RUNTIME_OPTION_FIELDS
    )
    assert mos_asset_preflight._MODEL_PROVENANCE_FIELDS is (
        mos_contract.MOS_ASSET_PROVENANCE_FIELDS
    )
    assert mos_runtime._MODEL_OPTION_FIELDS is mos_contract.MOS_RUNTIME_OPTION_FIELDS
    assert mos_runtime._ASSET_PROVENANCE_MODEL_FIELDS is (
        mos_contract.MOS_ASSET_PROVENANCE_FIELDS
    )
    assert step3_mos_worker._OPTION_FIELDS is (
        mos_contract.MOS_MODEL_REQUEST_OPTION_FIELDS
    )
    assert mos_pool._TREE_BOUND_OPTION_FIELDS == {
        model: mos_contract.MOS_RUNTIME_OPTION_FIELDS[model]
        for model in ("scoreq", "dnsmos")
    }


def test_provider_value_validator_is_one_shared_lightweight_contract():
    assert mos_runtime.validate_mos_provider_attestation is (
        mos_contract.validate_mos_provider_attestation
    )
    assert mos_gpu_job.validate_mos_provider_attestation is (
        mos_contract.validate_mos_provider_attestation
    )
    contradictory = {
        "execution_mode": "cuda_only_ep",
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "device": "gpu",
    }

    with pytest.raises(ValueError, match="scoreq provider"):
        mos_contract.validate_mos_provider_attestation(
            "scoreq", contradictory, requested="cuda",
        )


@pytest.mark.parametrize(
    ("requested", "execution_mode", "providers", "device"),
    [
        ("cpu", "cpu_ep", ["CPUExecutionProvider"], "not-cpu"),
        ("cuda", "cuda_only_ep", ["CUDAExecutionProvider"], "not-cuda"),
        ("cuda", "cuda_only_ep", ["CUDAExecutionProvider"], "not-gpu"),
    ],
)
def test_scoreq_provider_rejects_negated_device_markers(
    requested, execution_mode, providers, device,
):
    with pytest.raises(ValueError, match="scoreq provider"):
        mos_contract.validate_mos_provider_attestation(
            "scoreq",
            {
                "execution_mode": execution_mode,
                "providers": providers,
                "device": device,
            },
            requested=requested,
        )


@pytest.mark.parametrize(
    ("requested", "device"),
    [
        ("cpu", "cpu"),
        ("cuda", "cuda"),
        ("cuda", "cuda:0"),
        ("cuda", "gpu"),
    ],
)
def test_scoreq_provider_accepts_canonical_device_markers(requested, device):
    providers = (
        ["CPUExecutionProvider"]
        if requested == "cpu" else ["CUDAExecutionProvider"]
    )
    execution_mode = "cpu_ep" if requested == "cpu" else "cuda_only_ep"
    value = {
        "execution_mode": execution_mode,
        "providers": providers,
        "device": device,
    }

    mos_contract.validate_mos_provider_attestation(
        "scoreq", value, requested=requested,
    )
    # 同一 attestation 用相反的 requested device 解释时必须被拒绝，
    # 否则上面的"接受"可能只是校验分支没被执行。
    with pytest.raises(ValueError, match="scoreq provider"):
        mos_contract.validate_mos_provider_attestation(
            "scoreq",
            value,
            requested="cuda" if requested == "cpu" else "cpu",
        )


def test_worker_protocol_binding_execution_policy_and_wire_fields_remain_exact():
    assert mos_contract.MOS_WORKER_PROTOCOL == (
        "audio_filter_pipeline.step3_mos_worker.v6"
    )
    assert dict(mos_contract.MOS_ASSET_LOAD_BINDING) == {"version": 3}
    assert dict(mos_contract.MOS_SCORING_EXECUTION_POLICY) == {
        "version": 1,
        "waveform": {
            "channel": "first", "dtype": "float32", "padding": "none",
        },
        "batching": {
            "max_rows": 32,
            "max_float32_bytes": 64 * 1024 * 1024,
            "decode_reservation": "interval_channels_plus_mono_float32",
            "cross_source": True,
            "flush": "tuple_change_or_limit_or_eof",
        },
        "source_session": {
            "reuse": "same_source_across_batches",
            "switch": "close_reverify_then_open",
        },
        "worker": {"physical_subshard_boundary": "flush"},
    }
    assert mos_pool._PROTOCOL == step3_mos_worker._PROTOCOL == (
        mos_contract.MOS_WORKER_PROTOCOL
    )
    assert mos_runtime.ASSET_LOAD_BINDING is mos_contract.MOS_ASSET_LOAD_BINDING
    assert mos_pool.ASSET_LOAD_BINDING is mos_contract.MOS_ASSET_LOAD_BINDING
    assert step3_mos_worker.ASSET_LOAD_BINDING is (
        mos_contract.MOS_ASSET_LOAD_BINDING
    )
    assert mos_runtime.SCORING_EXECUTION_POLICY is mos_contract.MOS_SCORING_EXECUTION_POLICY
    assert step3_mos_worker.SCORING_EXECUTION_POLICY is (
        mos_contract.MOS_SCORING_EXECUTION_POLICY
    )
    assert mos_contract.MOS_WORKER_REQUEST_FIELDS == {
        "protocol", "asset_load_binding", "scoring_execution_policy", "private_dir", "attestation_path",
        "subshards", "scoreq_options", "urgent_options", "dnsmos_options",
        "nisqa_options", "utmos_options",
    }
    assert mos_contract.MOS_WORKER_ORDINAL_SIDECAR_FIELDS == {
        "ordinal_sidecar_path", "ordinal_sidecar_sha256",
        "ordinal_sidecar_size_bytes",
    }
    assert mos_contract.MOS_WORKER_SUBSHARD_FIELDS == {
        "models", "input_path", "output_path", "row_count", "input_sha256",
        "start_ordinal", "end_ordinal", "ordinal_sidecar_path",
        "ordinal_sidecar_sha256", "ordinal_sidecar_size_bytes",
    }
    assert step3_mos_worker._ORDINAL_SIDECAR_FIELDS is (
        mos_contract.MOS_WORKER_ORDINAL_SIDECAR_FIELDS
    )
    assert mos_pool._ORDINAL_SIDECAR_FIELDS is (
        mos_contract.MOS_WORKER_ORDINAL_SIDECAR_FIELDS
    )
    assert mos_contract.MOS_WORKER_ATTESTATION_FIELDS == {
        "protocol", "asset_load_binding", "scoring_execution_policy", "worker_models",
        "model_tuple_attestations",
    }
    assert mos_contract.MOS_WORKER_MODEL_TUPLE_ATTESTATION_FIELDS == {
        "models", "attestation",
    }


def test_contract_containers_are_immutable_at_every_mapping_layer():
    mappings = (
        mos_contract.MOS_THRESHOLD_DIMENSIONS,
        mos_contract.MOS_ASSET_PATH_FIELDS,
        mos_contract.MOS_ASSET_SETTING_FIELDS,
        mos_contract.MOS_RUNTIME_OPTION_FIELDS,
        mos_contract.MOS_ASSET_PROVENANCE_FIELDS,
        mos_contract.MOS_MODEL_REQUEST_OPTION_FIELDS,
        mos_contract.MOS_SCORING_EXECUTION_POLICY,
        mos_contract.MOS_ASSET_LOAD_BINDING,
        mos_contract.MOS_PROVIDER_ATTESTATION_FIELDS,
    )
    for mapping in mappings:
        with pytest.raises(TypeError):
            mapping["scoreq"] = object()
    with pytest.raises(TypeError):
        mos_contract.MOS_PROVIDER_ATTESTATION_FIELDS["dnsmos"]["cuda"] = (
            frozenset()
        )
    with pytest.raises(AttributeError):
        mos_contract.MOS_RUNTIME_OPTION_FIELDS["scoreq"].add("extra")
    with pytest.raises(TypeError):
        mos_contract.MOS_SCORING_EXECUTION_POLICY["batching"]["cross_source"] = False


def test_contract_source_uses_only_the_standard_library():
    source_path = Path(mos_contract.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
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
    assert imported_roots <= {"__future__", "types"}


def test_contract_and_config_import_without_mos_or_torch():
    package_root = Path(__file__).resolve().parents[1]
    tools_root = package_root.parent
    repository_root = tools_root.parent
    probe = """
import importlib.abc
import sys

class BlockHeavyDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, _path, _target=None):
        if (
            fullname == "audio_filter_pipeline.processors.mos"
            or fullname == "torch"
            or fullname.startswith("torch.")
        ):
            raise AssertionError(f"forbidden eager import: {fullname}")
        return None

sys.meta_path.insert(0, BlockHeavyDependencies())
import audio_filter_pipeline.processors.mos_contract as contract
import audio_filter_pipeline.config as config
assert contract.MOS_MODEL_ORDER == tuple(config._SELECTION_MOS_SPECS)
assert "audio_filter_pipeline.processors.mos" not in sys.modules
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
print("OK")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        str(repository_root), str(tools_root),
    ))
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=package_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "OK\n"
