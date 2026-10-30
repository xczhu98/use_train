"""Phase 4C-2 simulation audit envelope 独立合同测试。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core.data.simulation_trace import SimulationTrace
from tools.verify.simulation_audit_contract import (
    AUDIT_FORMAT_LAYERS_MEASUREMENTS,
    AUDIT_FORMAT_ROLE_LOCAL_TRACE,
    AUDIT_FORMAT_TSE_STEM_DESCRIPTORS,
    CONTRACT_VERSION,
    TSE_FINAL_PUBLICATION_MEASURE_POINT,
    TSE_LAYER_A_MEASURE_POINT,
    TSE_SIR_PRE_RIR_MEASURE_POINT,
    TSE_STEM_UNAVAILABLE_REASON,
    SimulationAuditContractError,
    SimulationAuditEnvelope,
    build_layers_measurements_envelope,
    build_tse_trace_metadata_envelope,
    load_simulation_audit_envelope,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_contract_publishes_version_1_2():
    assert CONTRACT_VERSION == "1.4"


def _payload(task="se"):
    return {
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "task": task,
        "audit_format": (
            AUDIT_FORMAT_ROLE_LOCAL_TRACE
            if task == "tse"
            else AUDIT_FORMAT_LAYERS_MEASUREMENTS
        ),
        "stem_capability": {
            "available": task != "tse",
            "reason": TSE_STEM_UNAVAILABLE_REASON if task == "tse" else None,
        },
        "payload": {"layers": {"B": {}}, "measurements": {}},
    }


def test_se_aec_keep_task_local_layers_measurements_and_capability():
    se_layers = {"B": {"branch": "prob_target_noise"}}
    se_measurements = {"volume": {"status": "applied"}}
    aec_layers = {"C": {"need_echo": True}}
    aec_measurements = {"layer_d_full_rms": {"ser_db": 3.0}}
    se = build_layers_measurements_envelope(
        task="se", layers=se_layers, measurements=se_measurements, stems_available=True)
    aec = build_layers_measurements_envelope(
        task="aec", layers=aec_layers, measurements=aec_measurements,
        stems_available=False, stems_unavailable_reason="verifier_without_stems")
    assert se.to_dict()["payload"] == {"layers": se_layers, "measurements": se_measurements}
    assert aec.to_dict()["payload"] == {"layers": aec_layers, "measurements": aec_measurements}
    assert se.to_dict()["stem_capability"] == {"available": True, "reason": None}
    assert aec.to_dict()["stem_capability"] == {
        "available": False, "reason": "verifier_without_stems"}


def _tse_trace(*, capture_stems=False):
    trace = SimulationTrace(index=7, uid="target", task="tse", capture_stems=capture_stems)
    trace.replace_metadata("branch", "prob_target_noise")
    for role, uid in (("target", "t"), ("interference/0", "i"),
                      ("alternate_target/0", "i"), ("enrollment/0", "e"),
                      ("scene", "scene")):
        trace.source(role).replace_metadata("uid", uid)
    return trace


def test_tse_keeps_complete_role_local_metadata_and_never_fakes_stems():
    trace = _tse_trace()
    trace.seal()
    envelope = build_tse_trace_metadata_envelope(trace).to_dict()
    assert envelope["contract_version"] == CONTRACT_VERSION
    assert envelope["schema_version"] == 1
    assert envelope["stem_capability"] == {
        "available": False, "reason": TSE_STEM_UNAVAILABLE_REASON}
    assert envelope["payload"] == {"simulation_trace": trace.metadata_dict()}
    assert set(envelope["payload"]["simulation_trace"]["sources"]) == {
        "target", "interference/0", "alternate_target/0", "enrollment/0", "scene"}
    assert "stems" not in json.dumps(envelope)


@pytest.mark.parametrize(
    "legacy_contract_version",
    [None, "1.0", "1.1", "1.2", "1.3"],
    ids=["missing", "v1.0", "v1.1", "v1.2", "v1.3"],
)
def test_tse_legacy_contract_versions_are_rejected(legacy_contract_version):
    """contract_version 缺失或非当前版本（1.0–1.3）一律 fail-closed。"""
    trace = _tse_trace()
    trace.seal()
    legacy = {
        "schema_version": 1,
        "task": "tse",
        "audit_format": AUDIT_FORMAT_ROLE_LOCAL_TRACE,
        "stem_capability": {
            "available": False,
            "reason": TSE_STEM_UNAVAILABLE_REASON,
        },
        "payload": {"simulation_trace": trace.metadata_dict()},
    }
    if legacy_contract_version is not None:
        legacy["contract_version"] = legacy_contract_version
    with pytest.raises(SimulationAuditContractError):
        SimulationAuditEnvelope.from_dict(legacy)


def _record_tse_capture_stems(trace, *, alternate=False):
    scene = trace.source("scene")
    scene.replace_metadata(
        "final_time_transform", {"target_sample_rate": 16000, "output_length": 3}
    )
    scene.record_stem("final_mixture", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    scene.record_stem(
        "final_additive_noise_reference", np.array([0.0, 0.5, 0.0], dtype=np.float32)
    )
    target = trace.source("target")
    target.replace_metadata("layer_a", {"sample_rate": 16000, "samples_after_align": 3})
    target.record_stem("layer_a_output", np.array([0.5, 0.0, 0.0], dtype=np.float32))
    target.replace_metadata("final_track", {"sample_rate": 16000, "num_samples": 3})
    target.record_stem("final_target", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    enrollment = trace.source("enrollment/0")
    enrollment.replace_metadata("enrollment", {"sample_rate": 16000, "num_samples": 2})
    enrollment.record_stem("final_enrollment", np.array([1.0, 0.0], dtype=np.float32))
    if alternate:
        interference = trace.source("interference/0")
        interference.replace_metadata(
            "sir_scaled_pre_rir", {"sample_rate": 16000, "num_samples": 3}
        )
        interference.record_stem(
            "sir_scaled_pre_rir", np.array([0.0, 0.25, 0.0], dtype=np.float32)
        )
        alt = trace.source("alternate_target/0")
        alt.replace_metadata("final_track", {"sample_rate": 16000, "num_samples": 3})
        alt.record_stem("final_target", np.array([0.0, 1.0, 0.0], dtype=np.float32))
        alt_enrollment = trace.source("enrollment/1")
        alt_enrollment.replace_metadata(
            "enrollment", {"sample_rate": 16000, "num_samples": 2}
        )
        alt_enrollment.record_stem(
            "final_enrollment", np.array([0.0, 1.0], dtype=np.float32)
        )


def test_tse_stem_descriptors_are_json_only_and_hash_final_waveforms():
    trace = _tse_trace(capture_stems=True)
    _record_tse_capture_stems(trace, alternate=True)
    trace.seal()
    envelope = build_tse_trace_metadata_envelope(trace).to_dict()
    assert envelope["audit_format"] == AUDIT_FORMAT_TSE_STEM_DESCRIPTORS
    assert envelope["stem_capability"] == {"available": True, "reason": None}
    assert SimulationAuditEnvelope.from_dict(envelope).to_dict() == envelope
    descriptors = envelope["payload"]["stem_descriptors"]
    assert {(item["role"], item["stem"]) for item in descriptors} == {
        ("scene", "final_mixture"),
        ("scene", "final_additive_noise_reference"),
        ("target", "layer_a_output"),
        ("target", "final_target"),
        ("interference/0", "sir_scaled_pre_rir"),
        ("alternate_target/0", "final_target"),
        ("enrollment/0", "final_enrollment"),
        ("enrollment/1", "final_enrollment"),
    }
    scene = next(
        item for item in descriptors if (item["role"], item["stem"]) == ("scene", "final_mixture")
    )
    assert scene["measure_point"] == TSE_FINAL_PUBLICATION_MEASURE_POINT
    layer_a = next(item for item in descriptors if item["stem"] == "layer_a_output")
    interference = next(item for item in descriptors if item["role"] == "interference/0")
    assert layer_a["measure_point"] == TSE_LAYER_A_MEASURE_POINT
    assert interference["measure_point"] == TSE_SIR_PRE_RIR_MEASURE_POINT
    assert scene["num_samples"] == 3
    assert scene["shape"] == [3]
    assert scene["nbytes"] == 12
    assert scene["sha256"] == hashlib.sha256(
        np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
    ).hexdigest()
    serialized = json.dumps(envelope)
    assert "[1.0, 2.0, 3.0]" not in serialized
    assert "ndarray" not in serialized
    assert "b'" not in serialized


def test_tse_stem_descriptor_contract_fails_closed():
    with pytest.raises(SimulationAuditContractError, match="sealed"):
        build_tse_trace_metadata_envelope(_tse_trace())
    trace = _tse_trace(capture_stems=True)
    _record_tse_capture_stems(trace)
    trace.source("target").replace_stem("unexpected", np.ones(3, dtype=np.float32))
    trace.seal()
    with pytest.raises(SimulationAuditContractError, match="不在 TSE schema"):
        build_tse_trace_metadata_envelope(trace)


def test_tse_descriptor_contract_rejects_unpaired_published_alternate():
    trace = _tse_trace(capture_stems=True)
    _record_tse_capture_stems(trace, alternate=True)
    trace.seal()
    payload = build_tse_trace_metadata_envelope(trace).to_dict()
    descriptors = payload["payload"]["stem_descriptors"]
    descriptors[:] = [
        item
        for item in descriptors
        if (item["role"], item["stem"]) != ("enrollment/1", "final_enrollment")
    ]

    with pytest.raises(SimulationAuditContractError, match="发布配对不一致"):
        SimulationAuditEnvelope.from_dict(payload)


def _ss_trace(*, source_count=2):
    trace = SimulationTrace(index=3, uid="ss", task="ss", capture_stems=True)
    trace.replace_metadata(
        "scene",
        {
            "sample_rate": 16000,
            "num_samples": 3,
            "source_count": source_count,
        },
    )
    scene = trace.source("scene")
    scene.replace_metadata(
        "scene",
        {
            "sample_rate": 16000,
            "num_samples": 3,
            "source_count": source_count,
        },
    )
    scene_stems = {
        "mixture": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "background": np.zeros(3, dtype=np.float32),
    }
    for index in range(source_count):
        scene_stems[f"source_full_rir/{index}"] = np.ones(3, dtype=np.float32)
        scene_stems[f"target/{index}"] = np.ones(3, dtype=np.float32)
    for stem, waveform in scene_stems.items():
        scene.record_stem(stem, waveform)
    trace.seal()
    return trace


def test_ss_source_scene_requires_all_fixed_k_source_target_pairs():
    from tools.verify.simulation_audit_contract import build_ss_trace_metadata_envelope

    envelope = build_ss_trace_metadata_envelope(_ss_trace())
    assert envelope.audit_format == "source-scene-stems-v2"
    payload = envelope.to_dict()
    payload["payload"]["stem_descriptors"] = [
        item
        for item in payload["payload"]["stem_descriptors"]
        if not (item["role"] == "scene" and item["stem"] == "target/1")
    ]
    with pytest.raises(SimulationAuditContractError, match="一一配对"):
        SimulationAuditEnvelope.from_dict(payload)


@pytest.mark.parametrize(
    ("mutate_roles", "match"),
    [
        (
            {"alternate_target/0": "alternate_target/2", "enrollment/1": "enrollment/3"},
            "alternate_target/N 索引必须从 0 连续",
        ),
        ({"enrollment/1": "enrollment/2"}, "enrollment/N 索引必须从 0 连续"),
    ],
)
def test_tse_descriptor_contract_rejects_sparse_indexed_roles(mutate_roles, match):
    trace = _tse_trace(capture_stems=True)
    _record_tse_capture_stems(trace, alternate=True)
    trace.seal()
    payload = build_tse_trace_metadata_envelope(trace).to_dict()
    for descriptor in payload["payload"]["stem_descriptors"]:
        descriptor["role"] = mutate_roles.get(descriptor["role"], descriptor["role"])

    with pytest.raises(SimulationAuditContractError, match=match):
        SimulationAuditEnvelope.from_dict(payload)


@pytest.mark.parametrize(("change", "match"), [
    (lambda p: p.update(future=1), "字段不匹配"),
    (lambda p: p.update(schema_version=True), "exact int"),
    (lambda p: p.update(task="tse"), "audit_format"),
    (lambda p: p["stem_capability"].update(reason="bad"), "reason 必须是 null"),
    (lambda p: p["payload"].update(value=float("nan")), "有限 JSON-safe"),
    (lambda p: p["payload"].update(wave=np.zeros(2)), "有限 JSON-safe"),
])
def test_envelope_fails_closed(change, match):
    payload = _payload()
    change(payload)
    with pytest.raises(SimulationAuditContractError, match=match):
        SimulationAuditEnvelope.from_dict(payload)


def test_tse_capability_requires_stable_unavailable_reason():
    payload = _payload("tse")
    payload["stem_capability"]["reason"] = "generic"
    with pytest.raises(SimulationAuditContractError, match="对应合同版本"):
        SimulationAuditEnvelope.from_dict(payload)
    payload["stem_capability"] = {"available": True, "reason": None}
    with pytest.raises(SimulationAuditContractError, match="unavailable"):
        SimulationAuditEnvelope.from_dict(payload)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda item: item.update(role="interference/0"), "不在 TSE schema"),
        (lambda item: item.update(stem="final_noise"), "不在 TSE schema"),
        (lambda item: item.update(measure_point="post_mix"), "测量点不匹配"),
        (lambda item: item.update(dtype="float64"), "float32"),
        (lambda item: item.update(sha256="A" * 64), "小写十六进制"),
        (lambda item: item.update(num_samples=0), "正整数"),
        (lambda item: item.pop("shape"), "字段不匹配"),
        (lambda item: item.update(shape=True), "shape"),
        (lambda item: item.update(shape=[-1]), "shape"),
        (lambda item: item.update(shape=[3]), "shape"),
        (lambda item: item.pop("nbytes"), "字段不匹配"),
        (lambda item: item.update(nbytes=True), "正整数"),
        (lambda item: item.update(nbytes=-4), "正整数"),
        (lambda item: item.update(nbytes=12), "nbytes"),
    ],
)
def test_tse_descriptor_json_validation_rejects_schema_tampering(mutate, match):
    trace = _tse_trace(capture_stems=True)
    _record_tse_capture_stems(trace)
    trace.seal()
    payload = build_tse_trace_metadata_envelope(trace).to_dict()
    mutate(payload["payload"]["stem_descriptors"][0])
    with pytest.raises(SimulationAuditContractError, match=match):
        SimulationAuditEnvelope.from_dict(payload)


def test_round_trip_and_real_tmp_fixture(tmp_path):
    envelope = build_layers_measurements_envelope(
        task="aec", layers={"B": {"branch": "main"}}, measurements={"x": [1, 2.5]},
        stems_available=True)
    path = tmp_path / "audit.json"
    path.write_text(envelope.to_json(), encoding="utf-8")
    assert load_simulation_audit_envelope(path).to_dict() == envelope.to_dict()


@pytest.mark.subprocess
def test_cli_help_and_real_tmp_fixture(tmp_path):
    script = Path("tools/verify/simulation_audit_contract.py").resolve()
    before = set(tmp_path.iterdir())
    help_result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                                 text=True, capture_output=True, check=False)
    assert help_result.returncode == 0 and "usage:" in help_result.stdout
    assert set(tmp_path.iterdir()) == before
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    valid = subprocess.run([sys.executable, str(script), str(path), "--json"],
                           text=True, capture_output=True, check=False)
    assert valid.returncode == 0 and json.loads(valid.stdout) == _payload()
    path.write_text("{}", encoding="utf-8")
    invalid = subprocess.run([sys.executable, str(script), str(path)],
                             text=True, capture_output=True, check=False)
    assert invalid.returncode == 1 and "验证失败" in invalid.stderr
