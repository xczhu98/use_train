"""指定 streaming ORT CLI 的启动 runtime metadata 预检契约。"""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from onnx.runtime_spec import (
    CAGCRN_RUNTIME_METADATA_KEY,
    CAGCRN_RUNTIME_METADATA_VERSION,
    DBGNET_CACHE_ABI_VERSION,
    DBGNET_CACHE_SEMANTICS,
    DBGNET_MODEL_FINGERPRINT_ALGORITHM,
    DEEPVQE_RUNTIME_METADATA_KEY,
    DEEPVQE_RUNTIME_METADATA_VERSION,
    DEPLOYMENT_CONFIG_METADATA_KEY,
    RUNTIME_METADATA_KEY,
    RUNTIME_METADATA_VERSION,
    DBGNetRuntimeSpec,
    STFTRuntimeSpec,
    StreamingRuntimeSpec,
    dbgnet_cache_names,
    deployment_config_binding,
    get_runtime_profile,
)


pytestmark = pytest.mark.unit

_CAGCRN_CACHE_NAMES = tuple(f"cache_{index}" for index in range(40))

_STREAMING_CLIS = (
    ("onnx.infer.infer_streaming_gtcrn_ort", "gtcrn", "conv_cache"),
    ("onnx.infer.infer_streaming_cagcrn_ort", "cagcrn", "cache_0"),
    ("onnx.infer.infer_streaming_deepvqe_ort", "deepvqe", "cache"),
    ("onnx.infer.infer_streaming_fastenhancer_ort", "fastenhancer", "cache"),
    ("onnx.infer.infer_streaming_ulunas_ort", "ulunas", "cache"),
    ("onnx.infer.infer_streaming_ultra_dpc_ort", "ultra_dpc", "cache"),
)


def _deepvqe_config():
    return {
        "model": {"type": "DeepVQE", "args": {
            "variant": "small", "align_delay": 20, "compression": 0.3,
        }},
        "stft": {
            "n_fft": 320, "hop_length": 160, "win_length": 320,
            "window": "hann", "center": True,
        },
        "data": {"sample_rate": 16000},
        "aec_frontend": {"input_routing": "mic_farend"},
    }


def _runtime_metadata(model_id: str) -> dict[str, str]:
    stft = {
        "n_fft": None,
        "hop_length": None,
        "win_length": None,
        "sample_rate": None,
        "window": "hann",
        "center": None,
    }
    if model_id == "cagcrn":
        stft = {
            "n_fft": 512,
            "hop_length": 256,
            "win_length": 512,
            "sample_rate": 16000,
            "window": "hann",
            "center": False,
        }
    elif model_id == "deepvqe":
        stft = deployment_config_binding(_deepvqe_config(), model_id)["config"]["stft"]
    metadata = {
        RUNTIME_METADATA_KEY: json.dumps({
            "version": RUNTIME_METADATA_VERSION,
            "model_id": model_id,
            "stft": stft,
        })
    }
    if model_id == "cagcrn":
        metadata[CAGCRN_RUNTIME_METADATA_KEY] = json.dumps({
            "version": CAGCRN_RUNTIME_METADATA_VERSION,
            "delay_d": 6,
            "algorithmic_delay_frames": 0,
            "cache_abi_version": 2,
            "cache_names": list(_CAGCRN_CACHE_NAMES),
        })
    if model_id == "deepvqe":
        metadata[DEPLOYMENT_CONFIG_METADATA_KEY] = json.dumps(
            deployment_config_binding(_deepvqe_config(), model_id)
        )
        metadata[DEEPVQE_RUNTIME_METADATA_KEY] = json.dumps({
            "version": DEEPVQE_RUNTIME_METADATA_VERSION,
            "variant": "small",
            "compression": 0.3,
            "alignment_window_frames": 20,
            "alignment_history_frames": 19,
            "algorithmic_delay_frames": 0,
            "algorithmic_delay_samples": 0,
            "cache_names": ["cache"],
        })
    return metadata


class _FakeSession:
    def __init__(self, model_id: str, *, metadata: dict[str, str] | None = None):
        profiles = {
            "gtcrn": (("spec",), "enh", "gtcrn_cache_cfg", "conv_cache", "conv_cache_out"),
            "cagcrn": (("mic_spec", "ref_spec"), "enh_spec", "cagcrn_cache_cfg", "cache", "out_cache"),
            "deepvqe": (("mic_spec", "far_spec"), "enh_spec", "deepvqe_cache_cfg", "cache", "out_cache"),
            "fastenhancer": (("spec_in",), "spec_out", "fastenhancer_cache_cfg", "cache", "out_cache"),
            "ulunas": (("mix",), "enh", "ulunas_cache_cfg", "cache", "cache_out"),
            "ultra_dpc": (("mic_spec", "ref_spec", "err_spec"), "enh_spec", "ultra_dpc_cache_cfg", "cache", "out_cache"),
        }
        primary_inputs, primary_output, metadata_key, cache_input, cache_output = profiles[model_id]
        cache_inputs = _CAGCRN_CACHE_NAMES if model_id == "cagcrn" else (cache_input,)
        cache_outputs = (
            tuple(f"out_{name}" for name in cache_inputs)
            if model_id == "cagcrn" else (cache_output,)
        )
        frequency_bins = {"cagcrn": 257, "deepvqe": 161}.get(model_id, 1)
        self._inputs = [
            SimpleNamespace(name=name, shape=["batch", frequency_bins, 1, 2], type="tensor(float)")
            for name in primary_inputs
        ]
        self._inputs.extend(
            SimpleNamespace(name=name, shape=["batch", 2], type="tensor(float)")
            for name in cache_inputs
        )
        self._outputs = [
            SimpleNamespace(
                name=primary_output, shape=["batch", frequency_bins, 1, 2], type="tensor(float)"
            ),
            *(SimpleNamespace(name=name, shape=["batch", 2], type="tensor(float)") for name in cache_outputs),
        ]
        self._metadata = metadata if metadata is not None else {
            **_runtime_metadata(model_id),
            metadata_key: json.dumps({name: [1, 2] for name in cache_inputs}),
        }

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self._metadata)


@pytest.mark.parametrize(("module_name", "model_id", "cache_input"), _STREAMING_CLIS)
def test_streaming_cli_preflight_accepts_valid_runtime_metadata(
    module_name, model_id, cache_input
):
    module = importlib.import_module(module_name)
    result = module._preflight_onnx_session(_FakeSession(model_id))
    runner = result[0] if model_id == "cagcrn" else result

    assert runner.spec.profile.model_id == model_id
    expected_cache_names = _CAGCRN_CACHE_NAMES if model_id == "cagcrn" else (cache_input,)
    assert runner.spec.cache_input_names == expected_cache_names
    assert all(runner.state[name].shape == (1, 2) for name in expected_cache_names)
    if model_id == "cagcrn":
        cagcrn_spec = result[1]
        assert cagcrn_spec.delay_d == 6
        assert cagcrn_spec.algorithmic_delay_frames == 0
        assert runner.spec.stft == STFTRuntimeSpec(
            n_fft=512,
            hop_length=256,
            win_length=512,
            sample_rate=16000,
            window="hann",
            center=False,
        )


@pytest.mark.parametrize(("module_name", "model_id", "_cache_input"), _STREAMING_CLIS)
def test_streaming_cli_preflight_rejects_missing_cache_metadata(
    module_name, model_id, _cache_input
):
    module = importlib.import_module(module_name)
    metadata_key = f"{model_id}_cache_cfg"
    with pytest.raises(RuntimeError, match=metadata_key):
        module._preflight_onnx_session(
            _FakeSession(model_id, metadata=_runtime_metadata(model_id))
        )


@pytest.mark.parametrize(("module_name", "model_id", "cache_input"), _STREAMING_CLIS)
def test_streaming_cli_preflight_rejects_missing_runtime_metadata(
    module_name, model_id, cache_input
):
    module = importlib.import_module(module_name)
    cache_names = _CAGCRN_CACHE_NAMES if model_id == "cagcrn" else (cache_input,)
    metadata = {
        f"{model_id}_cache_cfg": json.dumps({name: [1, 2] for name in cache_names}),
    }
    with pytest.raises(RuntimeError, match=RUNTIME_METADATA_KEY):
        module._preflight_onnx_session(_FakeSession(model_id, metadata=metadata))


@pytest.mark.parametrize(("module_name", "model_id", "cache_input"), _STREAMING_CLIS)
def test_streaming_cli_preflight_rejects_cache_shape_drift(
    module_name, model_id, cache_input
):
    module = importlib.import_module(module_name)
    cache_names = _CAGCRN_CACHE_NAMES if model_id == "cagcrn" else (cache_input,)
    metadata = {
        **_runtime_metadata(model_id),
        f"{model_id}_cache_cfg": json.dumps({
            name: [1, 3] if name == cache_names[0] else [1, 2]
            for name in cache_names
        }),
    }
    with pytest.raises(RuntimeError, match="static axis"):
        module._preflight_onnx_session(_FakeSession(model_id, metadata=metadata))


@pytest.mark.parametrize("drift", ("v1", "extra_cache", "reordered_cache_names", "output_name"))
def test_cagcrn_preflight_rejects_legacy_or_noncanonical_abi(drift):
    module = importlib.import_module("onnx.infer.infer_streaming_cagcrn_ort")
    metadata = {
        **_runtime_metadata("cagcrn"),
        "cagcrn_cache_cfg": json.dumps({name: [1, 2] for name in _CAGCRN_CACHE_NAMES}),
    }
    session = _FakeSession("cagcrn", metadata=metadata)
    if drift == "v1":
        payload = json.loads(metadata[CAGCRN_RUNTIME_METADATA_KEY])
        payload["version"] = 1
        metadata[CAGCRN_RUNTIME_METADATA_KEY] = json.dumps(payload)
        message = "CAGCRN runtime metadata version"
    elif drift == "extra_cache":
        session._inputs.append(SimpleNamespace(name="cache_40", shape=["batch", 2], type="tensor(float)"))
        session._outputs.append(SimpleNamespace(name="out_cache_40", shape=["batch", 2], type="tensor(float)"))
        metadata["cagcrn_cache_cfg"] = json.dumps({
            **json.loads(metadata["cagcrn_cache_cfg"]), "cache_40": [1, 2],
        })
        message = "cache order/input ABI mismatch"
    elif drift == "reordered_cache_names":
        payload = json.loads(metadata[CAGCRN_RUNTIME_METADATA_KEY])
        payload["cache_names"] = list(reversed(payload["cache_names"]))
        metadata[CAGCRN_RUNTIME_METADATA_KEY] = json.dumps(payload)
        message = "cache schema mismatch"
    else:
        session._outputs[1].name = "out_cache_bad"
        message = "cache output ABI mismatch"
    with pytest.raises((RuntimeError, ValueError), match=message):
        module._preflight_onnx_session(session)


def test_cagcrn_cli_stft_defaults_to_metadata_and_rejects_conflicts():
    module = importlib.import_module("onnx.infer.infer_streaming_cagcrn_ort")
    artifact = STFTRuntimeSpec(
        n_fft=512,
        hop_length=256,
        win_length=512,
        sample_rate=16000,
        window="hann",
        center=False,
    )
    assert module._resolve_stft_parameters(artifact, {}) == artifact
    assert module._resolve_stft_parameters(
        artifact,
        {
            "n_fft": 512,
            "hop_length": 256,
            "win_length": 512,
            "sample_rate": 16000,
            "window": "hann",
            "center": False,
        },
    ) == artifact
    with pytest.raises(ValueError, match="n_fft.*metadata"):
        module._resolve_stft_parameters(artifact, {"n_fft": 320})


def test_deepvqe_runtime_parameters_use_artifact_or_matching_config_as_truth():
    module = importlib.import_module("onnx.infer.infer_streaming_deepvqe_ort")
    artifact_spec = module._preflight_onnx_session(_FakeSession("deepvqe")).spec
    expected = {
        "variant": "small",
        "compression": 0.3,
        "align_delay": 20,
        "n_fft": 320,
        "hop_length": 160,
        "win_length": 320,
        "sample_rate": 16000,
        "window": "hann",
        "center": True,
    }

    assert module._resolve_runtime_parameters(
        config=None, artifact_spec=artifact_spec, cli_values={}
    ) == expected
    assert module._resolve_runtime_parameters(
        config=_deepvqe_config(), artifact_spec=artifact_spec,
        cli_values={"compression": 0.3, "center": True},
    ) == expected


def test_deepvqe_runtime_parameters_reject_config_binding_and_cli_conflicts():
    module = importlib.import_module("onnx.infer.infer_streaming_deepvqe_ort")
    artifact_spec = module._preflight_onnx_session(_FakeSession("deepvqe")).spec
    conflicting_config = _deepvqe_config()
    conflicting_config["model"]["args"]["compression"] = 0.4

    with pytest.raises(ValueError, match="config does not match ONNX deployment metadata"):
        module._resolve_runtime_parameters(
            config=conflicting_config, artifact_spec=artifact_spec, cli_values={}
        )
    with pytest.raises(ValueError, match="--n-fft=.*conflicts"):
        module._resolve_runtime_parameters(
            config=None, artifact_spec=artifact_spec, cli_values={"n_fft": 512}
        )


class _TransactionalSession(_FakeSession):
    def __init__(self):
        super().__init__("deepvqe")
        self._inputs.append(SimpleNamespace(
            name="cache_b", shape=["batch", 2], type="tensor(float)"
        ))
        self._outputs.append(SimpleNamespace(
            name="out_cache_b", shape=["batch", 2], type="tensor(float)"
        ))
        self._metadata = {
            **_runtime_metadata("deepvqe"),
            "deepvqe_cache_cfg": json.dumps({"cache": [1, 2], "cache_b": [1, 2]}),
        }
        deepvqe_runtime = json.loads(
            self._metadata[DEEPVQE_RUNTIME_METADATA_KEY]
        )
        deepvqe_runtime["cache_names"] = ["cache", "cache_b"]
        self._metadata[DEEPVQE_RUNTIME_METADATA_KEY] = json.dumps(deepvqe_runtime)
        self.fail_second = False
        self.grow_dynamic = False

    def run(self, _output_names, feeds):
        next_a = feeds["cache"] + 1
        next_b = feeds["cache_b"] + 2
        if self.fail_second:
            next_b = next_b.astype(np.float64)
        return [feeds["mic_spec"].copy(), next_a, next_b]


class _DBGNetTransactionalSession:
    cache_names = dbgnet_cache_names(False)

    def __init__(self):
        self._inputs = [
            SimpleNamespace(name="mic_spec", shape=[1, 161, 1, 2], type="tensor(float)"),
            SimpleNamespace(name="far_spec", shape=[1, 161, 1, 2], type="tensor(float)"),
            *(SimpleNamespace(name=name, shape=[1, 2], type="tensor(float)") for name in self.cache_names),
        ]
        self._outputs = [
            SimpleNamespace(name="enh", shape=[1, 161, 1, 2], type="tensor(float)"),
            *(SimpleNamespace(name=f"{name}_out", shape=[1, 2], type="tensor(float)") for name in self.cache_names),
        ]
        self.fail_second = False

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, _output_names, feeds):
        next_caches = [feeds[name] + index + 1 for index, name in enumerate(self.cache_names)]
        if self.fail_second:
            next_caches[1] = next_caches[1].astype(np.float64)
        return [feeds["mic_spec"].copy(), *next_caches]


def _dbgnet_runtime_spec(*, sha256: str = "1" * 64) -> StreamingRuntimeSpec:
    cache_names = dbgnet_cache_names(False)
    return StreamingRuntimeSpec(
        profile=get_runtime_profile("dbgnet"),
        input_names=("mic_spec", "far_spec", *cache_names),
        output_names=("enh", *(f"{name}_out" for name in cache_names)),
        cache_shapes={name: (1, 2) for name in cache_names},
        dbgnet=DBGNetRuntimeSpec(
            use_state_detector=False,
            state_nclass=3,
            network_lookahead_frames=0,
            algorithmic_delay_frames=0,
            algorithmic_delay_samples=0,
            stft_lookahead_samples=0,
            frame_shift_samples=160,
            frame_shift_ms=10.0,
            cache_abi_version=DBGNET_CACHE_ABI_VERSION,
            cache_semantics=DBGNET_CACHE_SEMANTICS,
            cache_names=cache_names,
            model_fingerprint={
                "algorithm": DBGNET_MODEL_FINGERPRINT_ALGORITHM,
                "sha256": sha256,
                "tensor_count": 1,
                "byte_count": 4,
            },
        ),
    )


def test_dbgnet_versioned_snapshot_restores_same_abi_and_cold_resets():
    from onnx.infer.streaming_ort_runner import (
        StreamingORTFrameRunner,
        StreamingORTStateSnapshot,
    )

    runner = StreamingORTFrameRunner(
        _DBGNetTransactionalSession(), spec=_dbgnet_runtime_spec()
    )
    frame = np.zeros((1, 161, 1, 2), dtype=np.float32)
    runner.run_frame(frame, frame)
    snapshot = runner.state

    assert isinstance(snapshot, StreamingORTStateSnapshot)
    assert snapshot.model_identity == f"dbgnet:{'1' * 64}"
    assert snapshot.cache_abi_version == DBGNET_CACHE_ABI_VERSION
    assert snapshot.cache_semantics == DBGNET_CACHE_SEMANTICS
    assert tuple(snapshot.values) == dbgnet_cache_names(False)
    with pytest.raises(TypeError):
        snapshot["mic_in"] = np.zeros((1, 2), dtype=np.float32)

    exposed = runner.state
    with pytest.raises(ValueError, match="read-only"):
        exposed["mic_in"][...] = 99
    with pytest.raises(ValueError, match="WRITEABLE"):
        exposed["mic_in"].setflags(write=True)
    np.testing.assert_array_equal(runner.state["mic_in"], np.ones((1, 2)))

    runner.run_frame(frame, frame)
    runner.reset(snapshot)
    np.testing.assert_array_equal(runner.state["mic_in"], np.ones((1, 2)))
    np.testing.assert_array_equal(runner.state["far_in"], np.full((1, 2), 2.0))
    assert runner.last_outputs == {}

    runner.reset()
    assert all(np.count_nonzero(value) == 0 for value in runner.state.values())
    assert runner.last_outputs == {}


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"model_identity": "other"}, "model identity mismatch"),
        ({"cache_abi_version": DBGNET_CACHE_ABI_VERSION + 1}, "ABI version mismatch"),
        ({"cache_semantics": "other"}, "cache semantics mismatch"),
    ),
)
def test_dbgnet_versioned_snapshot_rejects_identity_or_abi_drift(changes, message):
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    runner = StreamingORTFrameRunner(
        _DBGNetTransactionalSession(), spec=_dbgnet_runtime_spec()
    )
    candidate = replace(runner.state, **changes)

    with pytest.raises(ValueError, match=message):
        runner.reset(candidate)


def test_dbgnet_snapshot_rejects_same_config_with_different_weights():
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    source = StreamingORTFrameRunner(
        _DBGNetTransactionalSession(), spec=_dbgnet_runtime_spec(sha256="1" * 64)
    )
    target = StreamingORTFrameRunner(
        _DBGNetTransactionalSession(), spec=_dbgnet_runtime_spec(sha256="2" * 64)
    )

    with pytest.raises(ValueError, match="model identity mismatch"):
        target.reset(source.state)


def test_dbgnet_snapshot_rejects_other_runner_with_spoofed_metadata():
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    spoofed_spec = _dbgnet_runtime_spec(sha256="1" * 64)
    source = StreamingORTFrameRunner(
        _DBGNetTransactionalSession(), spec=spoofed_spec
    )
    target = StreamingORTFrameRunner(
        _DBGNetTransactionalSession(), spec=spoofed_spec
    )

    with pytest.raises(ValueError, match="runner identity mismatch"):
        target.reset(source.state)


def test_dbgnet_runner_rejects_non_one_batch_size():
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    with pytest.raises(ValueError, match="static batch size 1"):
        StreamingORTFrameRunner(
            _DBGNetTransactionalSession(),
            spec=_dbgnet_runtime_spec(),
            batch_size=2,
        )


def test_dbgnet_runner_rejects_dynamic_session_batch_with_manual_spec():
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    session = _DBGNetTransactionalSession()
    for info in (*session._inputs, *session._outputs):
        info.shape[0] = "batch"

    with pytest.raises(RuntimeError, match="static batch size 1"):
        StreamingORTFrameRunner(session, spec=_dbgnet_runtime_spec())


def test_dbgnet_versioned_snapshot_rejects_bare_mapping():
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    runner = StreamingORTFrameRunner(
        _DBGNetTransactionalSession(), spec=_dbgnet_runtime_spec()
    )

    with pytest.raises(TypeError, match="versioned state snapshot"):
        runner.reset(dict(runner.state))


@pytest.mark.parametrize("case", ("order", "name"))
def test_dbgnet_versioned_snapshot_rejects_cache_name_or_order_drift(case):
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    runner = StreamingORTFrameRunner(
        _DBGNetTransactionalSession(), spec=_dbgnet_runtime_spec()
    )
    values = dict(runner.state)
    if case == "order":
        values = dict(reversed(tuple(values.items())))
    else:
        values["legacy_y1_in"] = values.pop("gy1_in")
    candidate = replace(runner.state, values=values)

    with pytest.raises(ValueError, match="state names/order mismatch"):
        runner.reset(candidate)


def test_dbgnet_failed_reset_does_not_commit_partial_cache():
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    runner = StreamingORTFrameRunner(
        _DBGNetTransactionalSession(), spec=_dbgnet_runtime_spec()
    )
    frame = np.zeros((1, 161, 1, 2), dtype=np.float32)
    runner.run_frame(frame, frame)
    previous = runner.state
    invalid_values = dict(previous)
    invalid_values["mic_in"] = np.full((1, 2), 7.0, dtype=np.float32)
    invalid_values["far_in"] = np.full((1, 2), 8.0, dtype=np.float64)
    invalid = replace(previous, values=invalid_values)

    with pytest.raises(ValueError, match="dtype mismatch"):
        runner.reset(invalid)

    for name, value in previous.items():
        np.testing.assert_array_equal(runner.state[name], value)


def test_dbgnet_failed_frame_does_not_commit_partial_cache():
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    session = _DBGNetTransactionalSession()
    runner = StreamingORTFrameRunner(session, spec=_dbgnet_runtime_spec())
    frame = np.zeros((1, 161, 1, 2), dtype=np.float32)
    runner.run_frame(frame, frame)
    previous = runner.state
    session.fail_second = True

    with pytest.raises(RuntimeError, match="dtype changed"):
        runner.run_frame(frame, frame)

    for name, value in previous.items():
        np.testing.assert_array_equal(runner.state[name], value)


def test_non_dbgnet_reset_keeps_plain_mapping_compatibility():
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    runner = StreamingORTFrameRunner(_TransactionalSession(), "deepvqe")
    state = {
        "cache": np.full((1, 2), 3.0, dtype=np.float32),
        "cache_b": np.full((1, 2), 4.0, dtype=np.float32),
    }

    runner.reset(state)

    assert type(runner.state) is dict
    for name, value in state.items():
        np.testing.assert_array_equal(runner.state[name], value)


def test_runner_state_snapshots_cold_reset_and_atomic_cache_commit():
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    session = _TransactionalSession()
    runner = StreamingORTFrameRunner(session, "deepvqe")
    frame = np.zeros((1, 161, 1, 2), dtype=np.float32)
    runner.run_frame(frame, frame)

    exposed = runner.state
    exposed["cache"][...] = 99
    assert np.all(runner.state["cache"] == 1)
    exposed_outputs = runner.last_outputs
    exposed_outputs["out_cache"][...] = 99
    assert np.all(runner.last_outputs["out_cache"] == 1)

    result = runner.run(((frame, frame),))
    assert type(result.final_state) is dict
    result.final_state["cache"][...] = 99
    assert np.all(runner.state["cache"] == 2)
    previous = runner.state
    session.fail_second = True
    with pytest.raises(RuntimeError, match="dtype changed"):
        runner.run_frame(frame, frame)
    for name, value in previous.items():
        np.testing.assert_array_equal(runner.state[name], value)

    runner.reset()
    assert all(np.count_nonzero(value) == 0 for value in runner.state.values())
    assert runner.last_outputs == {}
