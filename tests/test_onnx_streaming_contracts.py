"""ONNX streaming 执行边界回归合同。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner
from onnx.models.streaming_wrapper import StreamingAECInferenceWrapper
from onnx.runtime_spec import (
    RUNTIME_METADATA_KEY,
    RUNTIME_METADATA_VERSION,
    get_runtime_profile,
)


pytestmark = pytest.mark.unit


class _Session:
    def get_inputs(self):
        return [
            SimpleNamespace(name="spec", shape=[1, 1, 1, 2], type="tensor(float)"),
            SimpleNamespace(name="conv_cache", shape=[1, 2], type="tensor(float)"),
        ]

    def get_outputs(self):
        return [
            SimpleNamespace(name="enh"),
            SimpleNamespace(name="conv_cache_out"),
        ]

    def get_modelmeta(self):
        import json

        return SimpleNamespace(custom_metadata_map={
            RUNTIME_METADATA_KEY: json.dumps({
                "version": RUNTIME_METADATA_VERSION,
                "model_id": "gtcrn",
                "stft": {
                    "n_fft": None, "hop_length": None, "win_length": None,
                    "sample_rate": None, "window": "hann", "center": None,
                },
            }),
            "gtcrn_cache_cfg": json.dumps({"conv_cache": [1, 2]}),
        })

    def run(self, _names, feeds):
        return [feeds["spec"].copy(), feeds["conv_cache"] + 1]


def test_ort_runner_rejects_primary_input_dtype_rank_and_nonfinite():
    runner = StreamingORTFrameRunner(_Session(), "gtcrn")
    with pytest.raises(ValueError, match="dtype mismatch"):
        runner.run_frame(np.ones((1, 1, 1, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="rank mismatch"):
        runner.run_frame(np.ones((1, 1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        runner.run_frame(np.full((1, 1, 1, 2), np.nan, dtype=np.float32))


def test_ort_runner_rejects_nonfinite_auxiliary_output_before_state_commit():
    class _AuxiliarySession(_Session):
        def get_outputs(self):
            return [
                SimpleNamespace(name="enh"),
                SimpleNamespace(name="state_probs"),
                SimpleNamespace(name="conv_cache_out"),
            ]

        def run(self, _names, feeds):
            return [
                feeds["spec"].copy(),
                np.full((1, 1), np.nan, dtype=np.float32),
                feeds["conv_cache"] + 1,
            ]

    profile = get_runtime_profile("gtcrn")
    spec = __import__("onnx.runtime_spec", fromlist=["StreamingRuntimeSpec"]).StreamingRuntimeSpec(
        profile=profile.__class__(
            **{
                **profile.__dict__,
                "auxiliary_outputs": ("state_probs",),
                "allowed_auxiliary_outputs": frozenset({"state_probs"}),
            },
        ),
        input_names=("spec", "conv_cache"),
        output_names=("enh", "state_probs", "conv_cache_out"),
        cache_shapes={"conv_cache": (1, 2)},
    )
    runner = StreamingORTFrameRunner(_AuxiliarySession(), spec=spec)
    before = runner.state["conv_cache"].copy()
    with pytest.raises(RuntimeError, match="auxiliary output.*non-finite"):
        runner.run_frame(np.ones((1, 1, 1, 2), dtype=np.float32))
    np.testing.assert_array_equal(runner.state["conv_cache"], before)


def test_aec_wrapper_rejects_mismatched_or_nonfinite_waveforms():
    class _Stft(torch.nn.Module):
        n_fft = 4
        hop_length = 2

    class _Model(torch.nn.Module):
        hop_length = 2
        n_fft = 4

        def initialize_cache(self, *_args):
            return {}

        def forward_streaming(self, *_args):
            raise AssertionError("invalid inputs must fail before model execution")

    wrapper = StreamingAECInferenceWrapper(_Stft(), _Model())
    with pytest.raises(ValueError, match="matching batch/length"):
        wrapper.process(torch.ones(4), torch.ones(5))
    with pytest.raises(TypeError, match="floating-point"):
        wrapper.process(torch.ones(4, dtype=torch.int64), torch.ones(4, dtype=torch.int64))
    with pytest.raises(ValueError, match="finite"):
        wrapper.process(torch.full((4,), float("nan")), torch.ones(4))


@pytest.mark.parametrize(
    "validator",
    [
        pytest.param(
            __import__("onnx.verify.verify_3way_cagcrn", fromlist=["_positive_frames"])._positive_frames,
            id="cagcrn",
        ),
        pytest.param(
            __import__("onnx.verify.verify_3way_fastenhancer", fromlist=["_positive_frames"])._positive_frames,
            id="fastenhancer",
        ),
        pytest.param(
            __import__("onnx.verify.verify_3way_ultra_dpc", fromlist=["_positive_frames"])._positive_frames,
            id="ultra_dpc",
        ),
    ],
)
def test_three_way_verifiers_reject_nonpositive_frames(validator):
    with pytest.raises(Exception, match="positive integer"):
        validator("0")
    assert validator("1") == 1
