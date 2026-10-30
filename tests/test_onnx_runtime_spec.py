"""StreamingRuntimeSpec 与 NumPy ORT frame runner 合同。"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner
from onnx.runtime_spec import (
    CAGCRN_RUNTIME_METADATA_KEY,
    CAGCRN_RUNTIME_METADATA_VERSION,
    DBGNET_CACHE_ABI_VERSION,
    DBGNET_CACHE_SEMANTICS,
    DBGNET_MODEL_FINGERPRINT_ALGORITHM,
    DBGNET_RUNTIME_METADATA_KEY,
    DBGNET_RUNTIME_METADATA_VERSION,
    DEEPVQE_RUNTIME_METADATA_KEY,
    DEEPVQE_RUNTIME_METADATA_VERSION,
    DEPLOYMENT_CONFIG_METADATA_KEY,
    RUNTIME_METADATA_KEY,
    RUNTIME_METADATA_VERSION,
    dbgnet_cache_names,
    dbgnet_model_fingerprint,
    deployment_config_binding,
    get_runtime_profile,
    normalize_model_id,
    runtime_spec_from_session,
    runtime_spec_from_wrapper,
)
from onnx.verify.verify_common import ThreeWayComparison, ThreeWayResult

pytestmark = pytest.mark.unit


class _TinyGTCRNWrapper:
    CACHE_KEYS = ("conv_cache",)

    def get_input_names(self):
        return ["spec", "conv_cache"]

    def get_output_names(self):
        return ["enh", "conv_cache_out"]

    def get_dynamic_axes(self):
        return {"spec": {0: "batch"}, "conv_cache": {0: "batch"}}

    def init_cache(self, batch_size, _device, _dtype):
        return {"conv_cache": torch.zeros(batch_size, 2)}


class _InitializeCacheFlatWrapper(_TinyGTCRNWrapper):
    def initialize_cache_flat(self, batch_size, _device, _dtype):
        return [torch.zeros(batch_size, 3)]

    def init_cache(self, _batch_size, _device, _dtype):
        raise AssertionError("initialize_cache_flat must be preferred")


def _deployment_config(model_id: str, *, use_state_detector: bool = True):
    model_type = "DeepVQE" if model_id == "deepvqe" else "DBGNet"
    model_args = (
        {"variant": "small", "align_delay": 20, "compression": 0.3}
        if model_id == "deepvqe"
        else {
            "sample_rate": 16000,
            "use_state_detector": use_state_detector,
            "state_nclass": 3,
        }
    )
    return {
        "model": {"type": model_type, "args": model_args},
        "stft": {
            "n_fft": 320, "hop_length": 160, "win_length": 320,
            "window": "hann", "center": True,
        },
        "data": {"sample_rate": 16000},
        "aec_frontend": {"input_routing": "mic_farend"},
    }


def _runtime_metadata(model_id="gtcrn"):
    stft = {
        "n_fft": None,
        "hop_length": None,
        "win_length": None,
        "sample_rate": None,
        "window": "hann",
        "center": None,
    }
    if model_id in {"deepvqe", "dbgnet"}:
        stft = deployment_config_binding(_deployment_config(model_id), model_id)["config"]["stft"]
    return json.dumps({
        "version": RUNTIME_METADATA_VERSION,
        "model_id": model_id,
        "stft": stft,
    })


class _FakeSession:
    def __init__(
        self,
        *,
        metadata='{"conv_cache":[1,2]}',
        runtime_metadata=None,
        include_runtime_metadata=True,
        nonfinite=False,
        cache_type="tensor(float)",
    ):
        self._metadata = metadata
        self._runtime_metadata = runtime_metadata or _runtime_metadata()
        self._include_runtime_metadata = include_runtime_metadata
        self._nonfinite = nonfinite
        self._cache_type = cache_type
        self.last_cache_dtype = None

    def get_inputs(self):
        return [
            SimpleNamespace(name="spec", shape=["batch", 1, 1, 2], type="tensor(float)"),
            SimpleNamespace(
                name="conv_cache", shape=["batch", 2], type=self._cache_type,
            ),
        ]

    def get_outputs(self):
        return [SimpleNamespace(name="enh"), SimpleNamespace(name="conv_cache_out")]

    def get_modelmeta(self):
        metadata = {"gtcrn_cache_cfg": self._metadata}
        if self._include_runtime_metadata:
            metadata[RUNTIME_METADATA_KEY] = self._runtime_metadata
        return SimpleNamespace(custom_metadata_map=metadata)

    def run(self, _names, feeds):
        cache = feeds["conv_cache"]
        self.last_cache_dtype = cache.dtype
        enhanced = feeds["spec"] + cache[:, :1, None, None]
        if self._nonfinite:
            enhanced = enhanced * np.nan
        return enhanced, cache + np.ones_like(cache)


def _test_dbgnet_fingerprint(sha256: str = "1" * 64) -> dict[str, object]:
    return {
        "algorithm": DBGNET_MODEL_FINGERPRINT_ALGORITHM,
        "sha256": sha256,
        "tensor_count": 1,
        "byte_count": 4,
    }


def _strict_aec_metadata(model_id: str, *, use_state_detector: bool = True):
    config = _deployment_config(model_id, use_state_detector=use_state_detector)
    cache_names = (
        dbgnet_cache_names(use_state_detector) if model_id == "dbgnet" else ("cache",)
    )
    metadata = {
        RUNTIME_METADATA_KEY: _runtime_metadata(model_id),
        DEPLOYMENT_CONFIG_METADATA_KEY: json.dumps(
            deployment_config_binding(config, model_id)
        ),
        f"{model_id}_cache_cfg": json.dumps({name: [1, 2] for name in cache_names}),
    }
    if model_id == "deepvqe":
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
    else:
        metadata[DBGNET_RUNTIME_METADATA_KEY] = json.dumps({
            "version": DBGNET_RUNTIME_METADATA_VERSION,
            "use_state_detector": use_state_detector,
            "state_nclass": 3,
            "network_lookahead_frames": 0,
            "algorithmic_delay_frames": 0,
            "algorithmic_delay_samples": 0,
            "stft_lookahead_samples": 160,
            "frame_shift_samples": 160,
            "frame_shift_ms": 10.0,
            "cache_abi_version": DBGNET_CACHE_ABI_VERSION,
            "cache_semantics": DBGNET_CACHE_SEMANTICS,
            "cache_names": list(cache_names),
            "model_fingerprint": _test_dbgnet_fingerprint(),
        })
    return metadata


class _StrictAECSession:
    _DEFAULT_BATCH = object()

    def __init__(
        self,
        model_id: str,
        *,
        metadata=None,
        frequency_bins: int = 161,
        primary_type: str = "tensor(float)",
        cache_output_shape=None,
        cache_output_type: str = "tensor(float)",
        state_shape=None,
        state_type: str = "tensor(float)",
        use_state_detector: bool = True,
        batch_dimension=_DEFAULT_BATCH,
    ):
        self.model_id = model_id
        self.metadata = metadata or _strict_aec_metadata(
            model_id, use_state_detector=use_state_detector
        )
        self.frequency_bins = frequency_bins
        self.primary_type = primary_type
        self.batch_dimension = (
            (1 if model_id == "dbgnet" else "batch")
            if batch_dimension is self._DEFAULT_BATCH else batch_dimension
        )
        self.cache_output_shape = cache_output_shape or [self.batch_dimension, 2]
        self.cache_output_type = cache_output_type
        self.state_shape = state_shape or [self.batch_dimension, 1, 3]
        self.state_type = state_type
        self.use_state_detector = use_state_detector
        self.cache_names = (
            dbgnet_cache_names(use_state_detector) if model_id == "dbgnet" else ("cache",)
        )

    def get_inputs(self):
        return [
            SimpleNamespace(name="mic_spec", shape=[self.batch_dimension, self.frequency_bins, 1, 2], type=self.primary_type),
            SimpleNamespace(name="far_spec", shape=[self.batch_dimension, self.frequency_bins, 1, 2], type=self.primary_type),
            *(SimpleNamespace(name=name, shape=[self.batch_dimension, 2], type="tensor(float)") for name in self.cache_names),
        ]

    def get_outputs(self):
        primary_name = "enh_spec" if self.model_id == "deepvqe" else "enh"
        outputs = [SimpleNamespace(
            name=primary_name,
            shape=[self.batch_dimension, self.frequency_bins, 1, 2],
            type=self.primary_type,
        )]
        if self.model_id == "dbgnet" and self.use_state_detector:
            outputs.append(SimpleNamespace(
                name="state_probs", shape=self.state_shape, type=self.state_type,
            ))
        outputs.extend(
            SimpleNamespace(
                name=(f"out_{name}" if self.model_id == "deepvqe" else f"{name}_out"),
                shape=(
                    self.cache_output_shape
                    if index == 0 else [self.batch_dimension, 2]
                ),
                type=self.cache_output_type if index == 0 else "tensor(float)",
            )
            for index, name in enumerate(self.cache_names)
        )
        return outputs

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self.metadata)


def test_runtime_spec_derives_cache_from_wrapper():
    spec = runtime_spec_from_wrapper(_TinyGTCRNWrapper(), "gtcrn")
    assert spec.input_names == ("spec", "conv_cache")
    assert spec.output_names == ("enh", "conv_cache_out")
    assert dict(spec.cache_shapes) == {"conv_cache": (1, 2)}
    assert spec.metadata["gtcrn_cache_cfg"] == {"conv_cache": [1, 2]}


def test_runtime_spec_prefers_initialize_cache_flat():
    spec = runtime_spec_from_wrapper(_InitializeCacheFlatWrapper(), "gtcrn")
    assert dict(spec.cache_shapes) == {"conv_cache": (1, 3)}
    assert spec.metadata["gtcrn_cache_cfg"] == {"conv_cache": [1, 3]}


def test_bsrnn_adapter_marks_b_times_k_cache_axis_flattened():
    class TinyBSRNNWrapper:
        def get_input_names(self):
            return ["frame_spec", "h_0", "c_0"]

        def get_output_names(self):
            return ["enhanced_spec", "h_0_out", "c_0_out"]

        def get_dynamic_axes(self):
            return {name: {0: "batch_size"} for name in self.get_input_names() + self.get_output_names()}

        def initialize_cache(self, batch_size, _device, _dtype):
            return [torch.zeros(1, batch_size * 8, 4), torch.zeros(1, batch_size * 8, 4)]

    spec = runtime_spec_from_wrapper(TinyBSRNNWrapper(), "bsrnn")
    assert dict(spec.dynamic_axes["h_0"]) == {1: "batch_flattened"}
    assert dict(spec.cache_shapes) == {"h_0": (1, 8, 4), "c_0": (1, 8, 4)}


def test_runtime_spec_rejects_unregistered_auxiliary_output():
    class UnexpectedAuxSession(_FakeSession):
        def get_outputs(self):
            return [
                SimpleNamespace(name="enh"),
                SimpleNamespace(name="state_probs"),
                SimpleNamespace(name="conv_cache_out"),
            ]

    with pytest.raises(ValueError, match="unregistered auxiliary outputs"):
        runtime_spec_from_session(UnexpectedAuxSession(), "gtcrn")


def test_dbgnet_wrapper_runtime_metadata_declares_static_batch_one_abi(monkeypatch):
    """DBGNet metadata 必须绑定 canonical cache 顺序与实际权重身份。"""
    from onnx.models.streaming_dbgnet import StreamingDBGNetFlat

    class _DummyStreamingDBGNet:
        CACHE_KEYS = dbgnet_cache_names(False)
        n_fft = 320
        hop_length = 160
        win_length = 320
        nfreqs = 161
        target_fs = 16000
        window = "hann"
        center = True
        use_state_detector = False
        state_nclass = 3
        network_lookahead_frames = 0
        algorithmic_delay_frames = 0
        algorithmic_delay = 0
        deployment_config = _deployment_config("dbgnet", use_state_detector=False)

        def __init__(self):
            self.dbgnet = torch.nn.Linear(1, 1)

        def init_cache(self, batch_size, device, dtype):
            return {
                name: torch.zeros(batch_size, 2, device=device, dtype=dtype)
                for name in self.CACHE_KEYS
            }

    inner = _DummyStreamingDBGNet()
    flat = StreamingDBGNetFlat(inner)
    spec = runtime_spec_from_wrapper(flat, "dbgnet")
    expected_shapes = {name: (1, 2) for name in inner.CACHE_KEYS}

    assert spec.profile.static_batch_size == 1
    assert dict(spec.dynamic_axes) == {}
    assert dict(spec.cache_shapes) == expected_shapes
    assert spec.metadata["dbgnet_cache_cfg"] == {
        name: [1, 2] for name in inner.CACHE_KEYS
    }
    assert spec.metadata[RUNTIME_METADATA_KEY]["model_id"] == "dbgnet"
    assert spec.metadata[DBGNET_RUNTIME_METADATA_KEY] == spec.dbgnet.to_metadata()
    assert spec.dbgnet.cache_abi_version == DBGNET_CACHE_ABI_VERSION
    assert spec.dbgnet.cache_semantics == DBGNET_CACHE_SEMANTICS
    assert spec.dbgnet.cache_names == inner.CACHE_KEYS
    assert dict(spec.dbgnet.model_fingerprint) == dbgnet_model_fingerprint(inner.dbgnet)

    serialized = {
        key: json.dumps(value, sort_keys=True, separators=(",", ":"))
        for key, value in spec.metadata.items()
    }
    assert tuple(json.loads(serialized["dbgnet_cache_cfg"])) != inner.CACHE_KEYS

    class _RoundtripSession:
        def get_inputs(self):
            return [
                SimpleNamespace(name="mic_spec", shape=[1, 161, 1, 2], type="tensor(float)"),
                SimpleNamespace(name="far_spec", shape=[1, 161, 1, 2], type="tensor(float)"),
                *(SimpleNamespace(name=name, shape=[1, 2], type="tensor(float)") for name in inner.CACHE_KEYS),
            ]

        def get_outputs(self):
            return [
                SimpleNamespace(name="enh", shape=[1, 161, 1, 2], type="tensor(float)"),
                *(SimpleNamespace(name=f"{name}_out", shape=[1, 2], type="tensor(float)") for name in inner.CACHE_KEYS),
            ]

        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map=serialized)

    session_spec = runtime_spec_from_session(_RoundtripSession(), "dbgnet")
    assert tuple(session_spec.cache_shapes) == inner.CACHE_KEYS
    assert dict(session_spec.cache_shapes) == expected_shapes

    from onnx.models.streaming_dbgnet import StreamingDBGNetFlat as _Flat
    from onnx import utils as onnx_utils

    monkeypatch.setattr(onnx_utils, "read_onnx_metadata", lambda _path: serialized)
    loaded = _Flat.load_cache_cfg_from_onnx("unused.onnx")
    assert tuple(loaded) == inner.CACHE_KEYS
    assert loaded == {name: [1, 2] for name in inner.CACHE_KEYS}

    legacy = json.loads(serialized[DBGNET_RUNTIME_METADATA_KEY])
    legacy["version"] = 1
    serialized[DBGNET_RUNTIME_METADATA_KEY] = json.dumps(legacy)
    with pytest.raises(RuntimeError, match="DBGNet runtime metadata version"):
        _Flat.load_cache_cfg_from_onnx("unused.onnx")
    with pytest.raises(ValueError, match="only supports batch_size=1"):
        flat.init_cache(2, "cpu", torch.float32)


def test_dbgnet_wrapper_contract_rejects_dynamic_batch_axis():
    from onnx.models.streaming_dbgnet import StreamingDBGNetFlat

    class _DynamicBatchFlat(StreamingDBGNetFlat):
        def get_dynamic_axes(self):
            return {
                name: {0: "batch"}
                for name in self.get_input_names() + self.get_output_names()
            }

    class _Inner:
        CACHE_KEYS = dbgnet_cache_names(False)
        n_fft = 320
        hop_length = 160
        win_length = 320
        nfreqs = 161
        target_fs = 16000
        window = "hann"
        center = True
        use_state_detector = False
        state_nclass = 3
        network_lookahead_frames = 0
        algorithmic_delay_frames = 0
        algorithmic_delay = 0
        deployment_config = _deployment_config("dbgnet", use_state_detector=False)
        dbgnet = torch.nn.Linear(1, 1)

        def init_cache(self, batch_size, device, dtype):
            return {
                name: torch.zeros(batch_size, 2, device=device, dtype=dtype)
                for name in self.CACHE_KEYS
            }

    with pytest.raises(ValueError, match="static batch size 1"):
        runtime_spec_from_wrapper(_DynamicBatchFlat(_Inner()), "dbgnet")


@pytest.mark.parametrize("batch_dimension", ("batch", None, 2))
def test_dbgnet_session_rejects_dynamic_or_non_one_batch_axis(batch_dimension):
    with pytest.raises(RuntimeError, match="static batch size 1"):
        runtime_spec_from_session(
            _StrictAECSession(
                "dbgnet",
                use_state_detector=False,
                batch_dimension=batch_dimension,
            ),
            "dbgnet",
        )


def test_dbgnet_session_rejects_dynamic_batch_axis_outside_axis_zero():
    session = _StrictAECSession("dbgnet", use_state_detector=False)
    inputs = session.get_inputs()
    inputs[10].shape = [1, "batch", 2]
    session.get_inputs = lambda: inputs

    with pytest.raises(RuntimeError, match="static batch size 1"):
        runtime_spec_from_session(session, "dbgnet")


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("v1", "DBGNet runtime metadata version"),
        ("missing_field", "metadata fields mismatch"),
        ("extra_field", "metadata fields mismatch"),
        ("cache_abi", "cache ABI version"),
        ("cache_semantics", "cache semantics"),
        ("fingerprint_missing", "fingerprint.*fields mismatch"),
        ("fingerprint_extra", "fingerprint.*fields mismatch"),
        ("fingerprint_sha", "fingerprint SHA-256"),
    ),
)
def test_dbgnet_runtime_metadata_rejects_schema_or_cache_contract_drift(
    case, message,
):
    metadata = _strict_aec_metadata("dbgnet", use_state_detector=False)
    payload = json.loads(metadata[DBGNET_RUNTIME_METADATA_KEY])
    if case == "v1":
        payload["version"] = 1
    elif case == "missing_field":
        payload.pop("cache_names")
    elif case == "extra_field":
        payload["unexpected"] = True
    elif case == "cache_abi":
        payload["cache_abi_version"] = DBGNET_CACHE_ABI_VERSION + 1
    elif case == "cache_semantics":
        payload["cache_semantics"] = "other"
    elif case == "fingerprint_missing":
        payload["model_fingerprint"].pop("byte_count")
    elif case == "fingerprint_extra":
        payload["model_fingerprint"]["unexpected"] = True
    else:
        payload["model_fingerprint"]["sha256"] = "not-a-sha256"
    metadata[DBGNET_RUNTIME_METADATA_KEY] = json.dumps(payload)

    with pytest.raises(RuntimeError, match=message):
        runtime_spec_from_session(
            _StrictAECSession(
                "dbgnet",
                metadata=metadata,
                use_state_detector=False,
            ),
            "dbgnet",
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("order", "cache schema mismatch"),
        ("name", "cache metadata/input ABI names mismatch"),
    ),
)
def test_dbgnet_runtime_metadata_rejects_cache_name_or_order_drift(case, message):
    metadata = _strict_aec_metadata("dbgnet", use_state_detector=False)
    payload = json.loads(metadata[DBGNET_RUNTIME_METADATA_KEY])
    if case == "order":
        payload["cache_names"] = list(reversed(payload["cache_names"]))
        metadata[DBGNET_RUNTIME_METADATA_KEY] = json.dumps(payload)
    else:
        config = json.loads(metadata["dbgnet_cache_cfg"])
        config["legacy_y1_in"] = config.pop("gy1_in")
        metadata["dbgnet_cache_cfg"] = json.dumps(config)

    with pytest.raises(RuntimeError, match=message):
        runtime_spec_from_session(
            _StrictAECSession(
                "dbgnet", metadata=metadata, use_state_detector=False,
            ),
            "dbgnet",
        )


@pytest.mark.parametrize("case", ("missing", "extra", "order", "legacy_yx"))
def test_dbgnet_session_rejects_graph_cache_schema_drift(case):
    session = _StrictAECSession("dbgnet", use_state_detector=False)
    cache_inputs = session.get_inputs()[2:]
    cache_outputs = session.get_outputs()[1:]
    if case == "missing":
        cache_inputs = cache_inputs[:-1]
        cache_outputs = cache_outputs[:-1]
    elif case == "extra":
        cache_inputs.append(SimpleNamespace(
            name="extra_cache", shape=[1, 2], type="tensor(float)"
        ))
        cache_outputs.append(SimpleNamespace(
            name="extra_cache_out", shape=[1, 2], type="tensor(float)"
        ))
    elif case == "order":
        cache_inputs[0], cache_inputs[1] = cache_inputs[1], cache_inputs[0]
        cache_outputs[0], cache_outputs[1] = cache_outputs[1], cache_outputs[0]
    else:
        cache_inputs[2].name = "y1_in"
        cache_outputs[2].name = "y1_in_out"
    session.get_inputs = lambda: [
        SimpleNamespace(name="mic_spec", shape=[1, 161, 1, 2], type="tensor(float)"),
        SimpleNamespace(name="far_spec", shape=[1, 161, 1, 2], type="tensor(float)"),
        *cache_inputs,
    ]
    session.get_outputs = lambda: [
        SimpleNamespace(name="enh", shape=[1, 161, 1, 2], type="tensor(float)"),
        *cache_outputs,
    ]

    with pytest.raises(RuntimeError, match="cache order/input ABI mismatch"):
        runtime_spec_from_session(session, "dbgnet")


def test_runtime_spec_accepts_dbgnet_state_probabilities():
    spec = runtime_spec_from_session(_StrictAECSession("dbgnet"), "dbgnet")
    expected_outputs = (
        "enh", "state_probs",
        *(f"{name}_out" for name in dbgnet_cache_names(True)),
    )
    assert spec.profile.auxiliary_outputs == ("state_probs",)
    assert spec.output_names == expected_outputs
    assert spec.dbgnet.state_nclass == 3
    assert spec.dbgnet.cache_names[-1] == "state_h"


@pytest.mark.parametrize(
    ("model_id", "metadata_key"),
    (("deepvqe", DEEPVQE_RUNTIME_METADATA_KEY), ("dbgnet", DBGNET_RUNTIME_METADATA_KEY)),
)
def test_strict_aec_artifacts_reject_missing_model_metadata(
    model_id, metadata_key,
):
    metadata = _strict_aec_metadata(model_id)
    metadata.pop(metadata_key)
    with pytest.raises(RuntimeError, match=metadata_key):
        runtime_spec_from_session(_StrictAECSession(model_id, metadata=metadata), model_id)


@pytest.mark.parametrize(
    ("model_id", "metadata_key", "version_message"),
    (
        ("deepvqe", DEEPVQE_RUNTIME_METADATA_KEY, "DeepVQE runtime metadata version"),
        ("dbgnet", DBGNET_RUNTIME_METADATA_KEY, "DBGNet runtime metadata version"),
    ),
)
def test_strict_aec_artifacts_reject_unknown_model_metadata_version(
    model_id, metadata_key, version_message,
):
    metadata = _strict_aec_metadata(model_id)
    payload = json.loads(metadata[metadata_key])
    payload["version"] = 999
    metadata[metadata_key] = json.dumps(payload)
    with pytest.raises(RuntimeError, match=version_message):
        runtime_spec_from_session(_StrictAECSession(model_id, metadata=metadata), model_id)


@pytest.mark.parametrize("model_id", ("deepvqe", "dbgnet"))
def test_strict_aec_artifacts_reject_legacy_deployment_binding_without_provenance(
    model_id,
):
    metadata = _strict_aec_metadata(model_id)
    binding = json.loads(metadata[DEPLOYMENT_CONFIG_METADATA_KEY])
    binding["version"] = 1
    binding.pop("resolved_config")
    metadata[DEPLOYMENT_CONFIG_METADATA_KEY] = json.dumps(binding)

    with pytest.raises(RuntimeError, match="metadata fields mismatch"):
        runtime_spec_from_session(_StrictAECSession(model_id, metadata=metadata), model_id)


@pytest.mark.parametrize("model_id", ("deepvqe", "dbgnet"))
def test_strict_aec_artifacts_reject_deployment_binding_sha_drift(model_id):
    metadata = _strict_aec_metadata(model_id)
    binding = json.loads(metadata[DEPLOYMENT_CONFIG_METADATA_KEY])
    binding["sha256"] = "0" * 64
    metadata[DEPLOYMENT_CONFIG_METADATA_KEY] = json.dumps(binding)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        runtime_spec_from_session(_StrictAECSession(model_id, metadata=metadata), model_id)


@pytest.mark.parametrize("model_id", ("deepvqe", "dbgnet"))
def test_strict_aec_artifacts_reject_incomplete_stft_metadata(model_id):
    metadata = _strict_aec_metadata(model_id)
    runtime = json.loads(metadata[RUNTIME_METADATA_KEY])
    runtime["stft"]["win_length"] = None
    metadata[RUNTIME_METADATA_KEY] = json.dumps(runtime)
    with pytest.raises(RuntimeError, match="deployment config STFT"):
        runtime_spec_from_session(_StrictAECSession(model_id, metadata=metadata), model_id)


@pytest.mark.parametrize("model_id", ("deepvqe", "dbgnet"))
def test_strict_aec_artifacts_reject_primary_graph_frequency_and_type_drift(model_id):
    with pytest.raises(RuntimeError, match="does not match STFT metadata"):
        runtime_spec_from_session(_StrictAECSession(model_id, frequency_bins=257), model_id)
    with pytest.raises(RuntimeError, match="must have type tensor\\(float\\)"):
        runtime_spec_from_session(
            _StrictAECSession(model_id, primary_type="tensor(double)"), model_id
        )


def test_strict_deepvqe_artifact_accepts_exported_symbolic_output_axes():
    """Torch exporter 可保留主输出符号轴，但输入频率轴仍必须静态绑定。"""
    session = _StrictAECSession("deepvqe")
    outputs = session.get_outputs()
    outputs[0].shape[1:] = [
        "Transposeenh_spec_dim_1",
        "Transposeenh_spec_dim_2",
        "Transposeenh_spec_dim_3",
    ]
    session.get_outputs = lambda: outputs

    spec = runtime_spec_from_session(session, "deepvqe")

    assert spec.output_names[0] == "enh_spec"


def test_strict_deepvqe_artifact_accepts_exported_symbolic_cache_output_axis():
    """DeepVQE cache 输出符号轴由 metadata 模板与首帧实际结果共同校验。"""
    session = _StrictAECSession(
        "deepvqe", cache_output_shape=["batch", "Unsqueezeout_dec4_res_dim_1"]
    )

    spec = runtime_spec_from_session(session, "deepvqe")

    assert spec.cache_shapes == {"cache": (1, 2)}


@pytest.mark.parametrize("model_id", ("deepvqe", "dbgnet"))
def test_strict_aec_artifacts_reject_cache_output_shape_and_type_drift(model_id):
    bad_shape = [1 if model_id == "dbgnet" else "batch", 3]
    with pytest.raises(RuntimeError, match="cache output shape"):
        runtime_spec_from_session(
            _StrictAECSession(model_id, cache_output_shape=bad_shape), model_id
        )
    with pytest.raises(RuntimeError, match="cache output type"):
        runtime_spec_from_session(
            _StrictAECSession(model_id, cache_output_type="tensor(double)"), model_id
        )


def test_dbgnet_artifact_rejects_state_output_shape_category_and_type_drift():
    with pytest.raises(RuntimeError, match="state_probs shape"):
        runtime_spec_from_session(
            _StrictAECSession("dbgnet", state_shape=[1, 1, 4]), "dbgnet"
        )
    with pytest.raises(RuntimeError, match="state_probs must have type"):
        runtime_spec_from_session(
            _StrictAECSession("dbgnet", state_type="tensor(double)"), "dbgnet"
        )


def test_numpy_runner_reset_rejects_invalid_state_shape_dtype_and_values():
    runner = StreamingORTFrameRunner(_FakeSession(), "gtcrn")
    with pytest.raises(ValueError, match="shape mismatch"):
        runner.reset({"conv_cache": np.zeros((1, 3), dtype=np.float32)})
    with pytest.raises(ValueError, match="dtype mismatch"):
        runner.reset({"conv_cache": np.zeros((1, 2), dtype=np.float64)})
    with pytest.raises(ValueError, match="non-finite"):
        runner.reset({"conv_cache": np.array([[0.0, np.nan]], dtype=np.float32)})


def test_numpy_runner_rejects_cache_output_dtype_change():
    class WrongCacheDtypeSession(_FakeSession):
        def run(self, names, feeds):
            enhanced, cache = super().run(names, feeds)
            return enhanced, cache.astype(np.float64)

    runner = StreamingORTFrameRunner(WrongCacheDtypeSession(), "gtcrn")
    with pytest.raises(RuntimeError, match="cache dtype changed"):
        runner.run_frame(np.ones((1, 1, 1, 2), dtype=np.float32))


def test_deepvqe_flat_wrapper_exposes_cache_initializer_for_runtime_spec():
    """DeepVQE 导出后写 metadata 必须能从 flat wrapper 推导 cache。"""
    from onnx.models.streaming_deepvqe import StreamingDeepVQEFlat

    class _DummyStreamingDeepVQE:
        CACHE_KEYS = ["cache"]
        n_fft = 320
        hop_length = 160
        win_length = 320
        target_fs = 16000
        window = "hann"
        center = True
        variant = "small"
        compression = 0.3
        alignment_window_frames = 20
        alignment_history_frames = 19
        algorithmic_delay_frames = 0
        algorithmic_delay = 0
        deployment_config = _deployment_config("deepvqe")

        def init_cache(self, batch, device, dtype):
            return {"cache": torch.zeros(batch, 2, device=device, dtype=dtype)}

    spec = runtime_spec_from_wrapper(
        StreamingDeepVQEFlat(_DummyStreamingDeepVQE()), "deepvqe"
    )
    assert spec.input_names == ("mic_spec", "far_spec", "cache")
    assert spec.output_names == ("enh_spec", "out_cache")
    assert dict(spec.cache_shapes) == {"cache": (1, 2)}
    assert spec.metadata["deepvqe_cache_cfg"] == {"cache": [1, 2]}


def test_deepvqe_export_metadata_roundtrip_preserves_graph_cache_order(monkeypatch):
    """JSON object 排序后仍必须由显式 cache_names 恢复 graph ABI 顺序。"""
    import onnx.export as export_module
    from onnx.models.streaming_deepvqe import StreamingDeepVQEFlat

    class _TwoCacheStreamingDeepVQE:
        CACHE_KEYS = ["z_cache", "a_cache"]
        n_fft = 320
        hop_length = 160
        win_length = 320
        target_fs = 16000
        window = "hann"
        center = True
        variant = "small"
        compression = 0.3
        alignment_window_frames = 20
        alignment_history_frames = 19
        algorithmic_delay_frames = 0
        algorithmic_delay = 0
        deployment_config = _deployment_config("deepvqe")

        def init_cache(self, batch, device, dtype):
            return {
                "z_cache": torch.zeros(batch, 2, device=device, dtype=dtype),
                "a_cache": torch.zeros(batch, 3, device=device, dtype=dtype),
            }

    serialized = {}

    def _serialize_like_writer(_path, metadata):
        serialized.update({
            key: json.dumps(value, sort_keys=True, separators=(",", ":"))
            for key, value in metadata.items()
        })

    flat = StreamingDeepVQEFlat(_TwoCacheStreamingDeepVQE())
    monkeypatch.setattr(export_module, "write_onnx_metadata", _serialize_like_writer)
    export_module.write_streaming_runtime_metadata("unused.onnx", flat)

    assert json.loads(serialized[DEEPVQE_RUNTIME_METADATA_KEY])["cache_names"] == [
        "z_cache", "a_cache",
    ]
    assert tuple(json.loads(serialized["deepvqe_cache_cfg"])) == (
        "a_cache", "z_cache",
    )

    class _RoundtripSession:
        def get_inputs(self):
            return [
                SimpleNamespace(name="mic_spec", shape=["batch", 161, 1, 2], type="tensor(float)"),
                SimpleNamespace(name="far_spec", shape=["batch", 161, 1, 2], type="tensor(float)"),
                SimpleNamespace(name="z_cache", shape=["batch", 2], type="tensor(float)"),
                SimpleNamespace(name="a_cache", shape=["batch", 3], type="tensor(float)"),
            ]

        def get_outputs(self):
            return [
                SimpleNamespace(name="enh_spec", shape=["batch", 161, 1, 2], type="tensor(float)"),
                SimpleNamespace(name="out_z_cache", shape=["batch", 2], type="tensor(float)"),
                SimpleNamespace(name="out_a_cache", shape=["batch", 3], type="tensor(float)"),
            ]

        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map=serialized)

    session_spec = runtime_spec_from_session(_RoundtripSession(), "deepvqe")
    assert tuple(session_spec.cache_shapes) == ("z_cache", "a_cache")
    assert dict(session_spec.cache_shapes) == {
        "z_cache": (1, 2), "a_cache": (1, 3),
    }


def test_deepvqe_runtime_metadata_rejects_cache_order_drift_and_legacy_v1():
    class _TwoCacheSession(_StrictAECSession):
        def get_inputs(self):
            return [
                *super().get_inputs(),
                SimpleNamespace(
                    name="cache_b", shape=["batch", 3], type="tensor(float)"
                ),
            ]

        def get_outputs(self):
            return [
                *super().get_outputs(),
                SimpleNamespace(
                    name="out_cache_b", shape=["batch", 3], type="tensor(float)"
                ),
            ]

    metadata = _strict_aec_metadata("deepvqe")
    metadata["deepvqe_cache_cfg"] = json.dumps({
        "cache": [1, 2], "cache_b": [1, 3],
    })
    payload = json.loads(metadata[DEEPVQE_RUNTIME_METADATA_KEY])

    payload["cache_names"] = ["cache_b", "cache"]
    metadata[DEEPVQE_RUNTIME_METADATA_KEY] = json.dumps(payload)
    with pytest.raises(RuntimeError, match="cache order/input ABI mismatch"):
        runtime_spec_from_session(
            _TwoCacheSession("deepvqe", metadata=metadata), "deepvqe"
        )

    payload.pop("cache_names")
    payload["version"] = 1
    metadata[DEEPVQE_RUNTIME_METADATA_KEY] = json.dumps(payload)
    with pytest.raises(RuntimeError, match="DeepVQE runtime metadata version"):
        runtime_spec_from_session(_TwoCacheSession("deepvqe", metadata=metadata), "deepvqe")


def test_deepvqe_wrapper_rejects_deployment_stft_drift_before_export():
    """wrapper 实际 STFT 与绑定配置不一致时，不得写出自相矛盾 metadata。"""
    from onnx.models.streaming_deepvqe import StreamingDeepVQEFlat

    class _DriftedStreamingDeepVQE:
        CACHE_KEYS = ["cache"]
        n_fft = 320
        hop_length = 160
        win_length = 320
        target_fs = 16000
        window = "hann"
        center = True
        variant = "small"
        compression = 0.3
        alignment_window_frames = 20
        alignment_history_frames = 19
        algorithmic_delay_frames = 0
        algorithmic_delay = 0
        deployment_config = _deployment_config("deepvqe")
        deployment_config["stft"] = {
            **deployment_config["stft"],
            "n_fft": 512,
            "win_length": 512,
        }

        def init_cache(self, batch, device, dtype):
            return {"cache": torch.zeros(batch, 2, device=device, dtype=dtype)}

    with pytest.raises(RuntimeError, match="deployment config STFT"):
        runtime_spec_from_wrapper(
            StreamingDeepVQEFlat(_DriftedStreamingDeepVQE()), "deepvqe"
        )


def test_localvqe_flat_wrapper_exposes_profile_and_metadata():
    from onnx.models.streaming_localvqe import StreamingLocalVQEFlat

    class _DummyStreamingLocalVQE:
        CACHE_KEYS = ("cache",)
        n_fft = 512
        hop_length = 256
        algorithmic_delay = 0

        def init_cache(self, batch, device, dtype):
            return {"cache": torch.zeros(batch, 2, device=device, dtype=dtype)}

    spec = runtime_spec_from_wrapper(StreamingLocalVQEFlat(_DummyStreamingLocalVQE()))
    assert spec.profile.model_id == "localvqe"
    assert spec.input_names == ("mic_spec", "far_spec", "cache")
    assert spec.output_names == ("enh_spec", "out_cache")
    assert dict(spec.cache_shapes) == {"cache": (1, 2)}
    assert spec.metadata["localvqe_cache_cfg"] == {"cache": [1, 2]}


def test_streaming_aliases_preserve_canonical_profiles():
    for alias, canonical in (
        ("streaming_dbgnet", "dbgnet"),
        ("streaming_bsrnn", "bsrnn"),
        ("streaming_localvqe", "localvqe"),
    ):
        assert normalize_model_id(alias) == canonical
        alias_profile = get_runtime_profile(alias)
        canonical_profile = get_runtime_profile(canonical)
        assert alias_profile == canonical_profile
        assert alias_profile.model_id == canonical


def test_runtime_spec_rejects_graph_metadata_shape_drift():
    with pytest.raises(RuntimeError, match="static axis"):
        runtime_spec_from_session(
            _FakeSession(metadata='{"conv_cache":[1,3]}'), "gtcrn"
        )


def test_runtime_spec_missing_runtime_metadata_fails_closed():
    session = _FakeSession(include_runtime_metadata=False)
    with pytest.raises(RuntimeError, match=RUNTIME_METADATA_KEY):
        runtime_spec_from_session(session, "gtcrn")
    with pytest.raises(RuntimeError, match=RUNTIME_METADATA_KEY):
        StreamingORTFrameRunner(session, "gtcrn")


def test_numpy_runner_keeps_and_updates_numpy_state():
    runner = StreamingORTFrameRunner(_FakeSession(), "gtcrn", batch_size=1)
    first = np.ones((1, 1, 1, 2), dtype=np.float32)
    result = runner.run([(first,), (first,)])
    assert all(isinstance(value, np.ndarray) for value in result.final_state.values())
    assert result.final_state["conv_cache"].tolist() == [[2.0, 2.0]]
    np.testing.assert_allclose(result.concatenate(), np.concatenate([first, first + 1], axis=2))


def test_numpy_runner_uses_ort_cache_input_dtype():
    session = _FakeSession(cache_type="tensor(int64)")
    runner = StreamingORTFrameRunner(session, "gtcrn", batch_size=1)
    frame = np.ones((1, 1, 1, 2), dtype=np.float32)
    runner.run_frame(frame)
    assert session.last_cache_dtype == np.dtype(np.int64)
    assert runner.state["conv_cache"].dtype == np.int64


def test_numpy_runner_rejects_nonfinite_primary_output():
    runner = StreamingORTFrameRunner(_FakeSession(nonfinite=True), "gtcrn")
    with pytest.raises(RuntimeError, match="non-finite"):
        runner.run_frame(np.ones((1, 1, 1, 2), dtype=np.float32))


def test_three_way_result_rejects_nonfinite_or_extra_gate_failure():
    passed = ThreeWayComparison("ok", 0.0, 0.0, 1, 1e-5, "PASS")
    nonfinite = ThreeWayComparison("nan", float("nan"), 0.0, 1, 1e-5, "PASS")
    assert ThreeWayResult(passed, passed, passed).all_pass
    assert not ThreeWayResult(passed, nonfinite, passed).all_pass
    assert not ThreeWayResult(
        passed, passed, passed, extra_gates={"metadata": False},
    ).all_pass


def test_cagcrn_versioned_runtime_metadata_roundtrip_and_legacy_rejection():
    cache_names = tuple(f"cache_{index}" for index in range(40))

    class TinyCAGCRNWrapper:
        n_fft = 320
        hop_length = 160
        win_length = 320
        target_fs = 16000
        window = "hann"
        center = False
        delay_d = 6
        algorithmic_delay_frames = 0

        def get_input_names(self):
            return ["mic_spec", "ref_spec", *cache_names]

        def get_output_names(self):
            return ["enh_spec", *(f"out_{name}" for name in cache_names)]

        def get_dynamic_axes(self):
            return {
                name: {0: "batch_size"}
                for name in self.get_input_names() + self.get_output_names()
            }

        def init_cache_flat(self, batch_size, _device, _dtype):
            return [torch.zeros(batch_size, 2) for _ in cache_names]

    wrapper_spec = runtime_spec_from_wrapper(TinyCAGCRNWrapper(), "cagcrn")
    assert wrapper_spec.stft.to_metadata() == {
        "n_fft": 320,
        "hop_length": 160,
        "win_length": 320,
        "sample_rate": 16000,
        "window": "hann",
        "center": False,
    }
    assert wrapper_spec.cagcrn.delay_d == 6
    assert wrapper_spec.cagcrn.algorithmic_delay_frames == 0
    assert wrapper_spec.cagcrn.cache_names == cache_names
    assert wrapper_spec.metadata[CAGCRN_RUNTIME_METADATA_KEY] == {
        "version": CAGCRN_RUNTIME_METADATA_VERSION,
        "delay_d": 6,
        "algorithmic_delay_frames": 0,
        "cache_abi_version": 2,
        "cache_names": list(cache_names),
    }

    class CAGCRNSession:
        _DEFAULT = object()

        def __init__(
            self,
            *,
            frequency_bins=161,
            output_shape=None,
            output_type="tensor(float)",
            cagcrn_metadata=_DEFAULT,
            cache_schema=cache_names,
        ):
            self.frequency_bins = frequency_bins
            self.output_shape = output_shape or ["batch_size", 2]
            self.output_type = output_type
            self.cache_schema = cache_schema
            self.cagcrn_metadata = (
                wrapper_spec.metadata[CAGCRN_RUNTIME_METADATA_KEY]
                if cagcrn_metadata is self._DEFAULT else cagcrn_metadata
            )

        def get_inputs(self):
            return [
                SimpleNamespace(name="mic_spec", shape=["batch_size", self.frequency_bins, 1, 2], type="tensor(float)"),
                SimpleNamespace(name="ref_spec", shape=["batch_size", self.frequency_bins, 1, 2], type="tensor(float)"),
                *(SimpleNamespace(name=name, shape=["batch_size", 2], type="tensor(float)") for name in self.cache_schema),
            ]

        def get_outputs(self):
            return [
                SimpleNamespace(name="enh_spec", shape=["batch_size", self.frequency_bins, 1, 2], type="tensor(float)"),
                *(SimpleNamespace(name=f"out_{name}", shape=self.output_shape, type=self.output_type) for name in self.cache_schema),
            ]

        def get_modelmeta(self):
            metadata = {
                RUNTIME_METADATA_KEY: json.dumps(wrapper_spec.metadata[RUNTIME_METADATA_KEY]),
                "cagcrn_cache_cfg": json.dumps({name: [1, 2] for name in self.cache_schema}),
            }
            if self.cagcrn_metadata is not None:
                metadata[CAGCRN_RUNTIME_METADATA_KEY] = json.dumps(self.cagcrn_metadata)
            return SimpleNamespace(custom_metadata_map=metadata)

    session_spec = runtime_spec_from_session(CAGCRNSession(), "cagcrn")
    assert session_spec.stft == wrapper_spec.stft
    assert session_spec.cagcrn == wrapper_spec.cagcrn
    assert session_spec.input_names[2:] == cache_names
    assert session_spec.output_names[1:] == tuple(f"out_{name}" for name in cache_names)

    with pytest.raises(RuntimeError, match=CAGCRN_RUNTIME_METADATA_KEY):
        runtime_spec_from_session(CAGCRNSession(cagcrn_metadata=None), "cagcrn")

    invalid_version = dict(wrapper_spec.metadata[CAGCRN_RUNTIME_METADATA_KEY])
    invalid_version["version"] = 1
    with pytest.raises(RuntimeError, match="CAGCRN runtime metadata version"):
        runtime_spec_from_session(CAGCRNSession(cagcrn_metadata=invalid_version), "cagcrn")

    invalid_delay = dict(wrapper_spec.metadata[CAGCRN_RUNTIME_METADATA_KEY])
    invalid_delay["delay_d"] = 0
    with pytest.raises(RuntimeError, match="delay_d"):
        runtime_spec_from_session(CAGCRNSession(cagcrn_metadata=invalid_delay), "cagcrn")

    invalid_delay = dict(wrapper_spec.metadata[CAGCRN_RUNTIME_METADATA_KEY])
    invalid_delay["algorithmic_delay_frames"] = 1
    with pytest.raises(RuntimeError, match="algorithmic delay"):
        runtime_spec_from_session(CAGCRNSession(cagcrn_metadata=invalid_delay), "cagcrn")

    invalid_cache_names = dict(wrapper_spec.metadata[CAGCRN_RUNTIME_METADATA_KEY])
    invalid_cache_names["cache_names"] = "cache_0"
    with pytest.raises(RuntimeError, match="cache schema mismatch"):
        runtime_spec_from_session(CAGCRNSession(cagcrn_metadata=invalid_cache_names), "cagcrn")

    with pytest.raises(RuntimeError, match="cache order/input ABI mismatch"):
        runtime_spec_from_session(CAGCRNSession(cache_schema=(*cache_names, "cache_40")), "cagcrn")
    with pytest.raises(RuntimeError, match="cache order/input ABI mismatch"):
        runtime_spec_from_session(CAGCRNSession(cache_schema=tuple(reversed(cache_names))), "cagcrn")
    with pytest.raises(RuntimeError, match="cache output.*shape"):
        runtime_spec_from_session(CAGCRNSession(output_shape=["batch_size", 3]), "cagcrn")
    with pytest.raises(RuntimeError, match="cache output.*type"):
        runtime_spec_from_session(CAGCRNSession(output_type="tensor(double)"), "cagcrn")
    with pytest.raises(RuntimeError, match="does not match STFT metadata"):
        runtime_spec_from_session(CAGCRNSession(frequency_bins=257), "cagcrn")


def test_versioned_stft_metadata_roundtrip_and_version_gate():
    class STFTWrapper(_TinyGTCRNWrapper):
        n_fft = 320
        hop_length = 160
        win_length = 320
        target_fs = 16000
        window = "hann"
        center = False

    wrapper_spec = runtime_spec_from_wrapper(STFTWrapper(), "gtcrn")
    runtime_metadata = wrapper_spec.metadata[RUNTIME_METADATA_KEY]
    session = _FakeSession()
    session.get_modelmeta = lambda: SimpleNamespace(
        custom_metadata_map={
            "gtcrn_cache_cfg": '{"conv_cache":[1,2]}',
            RUNTIME_METADATA_KEY: json.dumps(runtime_metadata),
        }
    )
    session_spec = runtime_spec_from_session(session, "gtcrn")
    assert session_spec.stft == wrapper_spec.stft

    invalid = dict(runtime_metadata)
    invalid["version"] = 999
    session.get_modelmeta = lambda: SimpleNamespace(
        custom_metadata_map={
            "gtcrn_cache_cfg": '{"conv_cache":[1,2]}',
            RUNTIME_METADATA_KEY: json.dumps(invalid),
        }
    )
    with pytest.raises(RuntimeError, match="Unsupported streaming runtime metadata version"):
        runtime_spec_from_session(session, "gtcrn")


def test_materialize_flattened_batch_axis_uses_batch_one_multiplier():
    from onnx.utils import materialize_cache_shapes

    infos = [SimpleNamespace(name="cache", shape=[1, "batch_flattened", 8])]
    assert materialize_cache_shapes({"cache": [1, 33, 8]}, infos, 4) == {
        "cache": [1, 132, 8]
    }


def test_pypi_onnx_scope_serializes_concurrent_threads(monkeypatch):
    import onnx.utils as utils

    external = SimpleNamespace(__file__="/site-packages/onnx/__init__.py")

    def fake_import_module(name):
        # 真实 import 一定把模块登记进 sys.modules；mock 必须保持同一不变量，
        # 否则 scope 的隔离复核会把「绑定缺失」误判为并发改写。
        utils.sys.modules[name] = external
        return external

    monkeypatch.setattr(utils.importlib, "import_module", fake_import_module)
    original_cache = dict(utils._PYPI_ONNX_MODULE_CACHE)
    utils._PYPI_ONNX_MODULE_CACHE.clear()
    active = 0
    maximum = 0
    state_lock = threading.Lock()

    def worker():
        nonlocal active, maximum
        with utils.pypi_onnx_scope():
            with state_lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert maximum == 1
    finally:
        utils._PYPI_ONNX_MODULE_CACHE.clear()
        utils._PYPI_ONNX_MODULE_CACHE.update(original_cache)


_HAS_EXTERNAL_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None


@pytest.mark.ort
@pytest.mark.integration
@pytest.mark.skipif(
    not (_HAS_EXTERNAL_ONNX and _HAS_ORT),
    reason="PyPI onnx and onnxruntime are required for real ORT ABI integration",
)
def test_real_ort_streaming_abi_integration(tmp_path):
    from onnx.utils import create_ort_session, export_onnx, write_onnx_metadata

    class TinyStreaming(torch.nn.Module):
        def forward(self, spec, state):
            return spec + state[:, :1, None, None], state + 1

    path = tmp_path / "tiny.onnx"
    export_onnx(
        TinyStreaming(),
        (torch.ones(1, 1, 1, 2), torch.zeros(1, 2)),
        str(path),
        input_names=["spec", "conv_cache"],
        output_names=["enh", "conv_cache_out"],
        dynamic_axes={
            "spec": {0: "batch"}, "enh": {0: "batch"},
            "conv_cache": {0: "batch"}, "conv_cache_out": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    write_onnx_metadata(
        path,
        {
            "gtcrn_cache_cfg": {"conv_cache": [1, 2]},
            RUNTIME_METADATA_KEY: json.loads(_runtime_metadata()),
        },
    )
    runner = StreamingORTFrameRunner(create_ort_session(path), "gtcrn")
    frame = np.ones((1, 1, 1, 2), dtype=np.float32)
    result = runner.run([(frame,), (frame,)])
    np.testing.assert_allclose(result.concatenate(), np.concatenate([frame, frame + 1], axis=2))


@pytest.mark.ort
@pytest.mark.integration
@pytest.mark.skipif(
    not (_HAS_EXTERNAL_ONNX and _HAS_ORT),
    reason="PyPI onnx and onnxruntime are required for DeepVQE metadata roundtrip",
)
def test_real_deepvqe_export_metadata_roundtrip_preserves_cache_order(tmp_path):
    """真实 metadata writer 按插入序保留 cache ABI 顺序；versioned cache_names 兜底旧资产。"""
    from onnx.export import write_streaming_runtime_metadata
    from onnx.models.streaming_deepvqe import StreamingDeepVQEFlat
    from onnx.utils import create_ort_session, export_onnx, read_onnx_metadata

    class _TinyDeepVQE(torch.nn.Module):
        CACHE_KEYS = ["z_cache", "a_cache"]
        n_fft = 320
        hop_length = 160
        win_length = 320
        target_fs = 16000
        window = "hann"
        center = True
        variant = "small"
        compression = 0.3
        alignment_window_frames = 20
        alignment_history_frames = 19
        algorithmic_delay_frames = 0
        algorithmic_delay = 0
        deployment_config = _deployment_config("deepvqe")

        def init_cache(self, batch, device, dtype):
            return {
                "z_cache": torch.zeros(batch, 2, device=device, dtype=dtype),
                "a_cache": torch.zeros(batch, 3, device=device, dtype=dtype),
            }

        def forward(self, mic_spec, far_spec, cache):
            return mic_spec + far_spec, {
                "z_cache": cache["z_cache"] + 1,
                "a_cache": cache["a_cache"] + 2,
            }

    flat = StreamingDeepVQEFlat(_TinyDeepVQE()).eval()
    cache = flat.init_cache_flat(1, "cpu", torch.float32)
    path = tmp_path / "tiny_deepvqe.onnx"
    export_onnx(
        flat,
        (
            torch.ones(1, 161, 1, 2),
            torch.ones(1, 161, 1, 2),
            *cache,
        ),
        str(path),
        input_names=flat.get_input_names(),
        output_names=flat.get_output_names(),
        dynamic_axes=flat.get_dynamic_axes(),
        opset_version=17,
        dynamo=False,
    )
    write_streaming_runtime_metadata(path, flat)
    metadata = read_onnx_metadata(path)
    session_spec = runtime_spec_from_session(
        create_ort_session(path), "deepvqe"
    )
    loaded_cache = StreamingDeepVQEFlat.load_cache_cfg_from_onnx(path)

    # writer 保持插入序：序列化后的 cache_cfg 键顺序即 ABI 顺序（非字母序）。
    assert tuple(json.loads(metadata["deepvqe_cache_cfg"])) == (
        "z_cache", "a_cache",
    )
    assert json.loads(metadata[DEEPVQE_RUNTIME_METADATA_KEY])["cache_names"] == [
        "z_cache", "a_cache",
    ]
    assert tuple(session_spec.cache_shapes) == ("z_cache", "a_cache")
    assert loaded_cache == {"z_cache": (1, 2), "a_cache": (1, 3)}
