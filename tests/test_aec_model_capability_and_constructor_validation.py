"""AEC 模型 capability 与 state detector 构造参数边界回归测试。"""

from __future__ import annotations

import pytest

from tasks.aec.inference.aec_frontend import AECFrontendConfig
from tasks.aec.models.discriminative.ultra_dpc_aec import UltraDPCAEC
from tasks.aec.models.state_detector import (
    AECStateDetector,
    AECStateDetectorHead,
    AECStateFeatureExtractor,
)
from tasks.aec.trainer.aec_trainer import AECTrainer


pytestmark = pytest.mark.unit


@pytest.mark.unit
def test_ultra_dpc_allows_mic_farend_error_routing():
    trainer = object.__new__(AECTrainer)
    trainer.model = UltraDPCAEC(
        n_fft=16,
        hop_length=8,
        embed_dim=4,
        n_dp_blocks=1,
        n_heads=2,
        encoder_layers=1,
        decoder_layers=1,
    )
    trainer._aec_frontend_config = AECFrontendConfig.from_dict(
        {
            "linear_aec": {"backend": "pfdkf_torch"},
            "input_routing": "mic_farend_error",
        }
    )

    trainer._validate_aec_frontend_capabilities()

    assert "error_spectra" in UltraDPCAEC.aec_input_capabilities


@pytest.mark.parametrize(
    ("constructor", "kwargs"),
    [
        (AECStateFeatureExtractor, {"input_size": 2, "typo": 1}),
        (AECStateDetector, {"input_size": 2, "typo": 1}),
        (AECStateDetectorHead, {"feature_freq_sizes": [1], "typo": 1}),
    ],
)
def test_state_detector_constructors_reject_unknown_kwargs(constructor, kwargs):
    with pytest.raises(TypeError, match="unexpected keyword arguments.*typo"):
        constructor(**kwargs)


@pytest.mark.parametrize(
    ("constructor", "kwargs", "parameter"),
    [
        (AECStateFeatureExtractor, {"input_size": 0}, "input_size"),
        (AECStateFeatureExtractor, {"input_size": True}, "input_size"),
        (AECStateDetector, {"input_size": 0}, "input_size"),
        (AECStateDetector, {"input_size": 1, "hidden_size": 0}, "hidden_size"),
        (AECStateDetector, {"input_size": 1, "nclass": 0}, "nclass"),
        (AECStateDetectorHead, {"feature_freq_sizes": [0]}, r"feature_freq_sizes\[0\]"),
        (AECStateDetectorHead, {"feature_freq_sizes": [True]}, r"feature_freq_sizes\[0\]"),
        (AECStateDetectorHead, {"feature_freq_sizes": [1], "projection_size": 0}, "projection_size"),
    ],
)
def test_state_detector_constructors_reject_invalid_sizes(constructor, kwargs, parameter):
    with pytest.raises(ValueError, match=parameter):
        constructor(**kwargs)


@pytest.mark.parametrize(
    "constructor, kwargs",
    [
        (AECStateFeatureExtractor, {"input_size": 2, "spatial_layout": "invalid"}),
        (AECStateDetectorHead, {"feature_freq_sizes": [1], "spatial_layout": "invalid"}),
    ],
)
def test_state_detector_constructors_reject_invalid_spatial_layout(constructor, kwargs):
    with pytest.raises(ValueError, match="spatial_layout"):
        constructor(**kwargs)


def test_aec_native_multichannel_contract_requires_explicit_axes():
    from core.models.io_contract import ModelIOContract, TensorField
    from tasks.aec.model_io import validate_aec_multichannel_contract

    class _NativeAEC:
        supports_multichannel_input = True
        IO_CONTRACT = ModelIOContract(
            tasks=frozenset({"aec"}),
            inputs=(
                TensorField("mic_spectra", "spectrum", ("batch", "microphone", "frequency", "frame", "complex")),
                TensorField("farend_spectra", "spectrum", ("batch", "frequency", "frame", "complex")),
            ),
            outputs=(TensorField("nearend_estimate_spectra", "spectrum", ("batch", "frequency", "frame", "complex")),),
            forward_kind="estimate",
            frontend_owner="runtime",
        )

    from tasks.aec.model_io import validate_aec_multichannel_contract
    validate_aec_multichannel_contract(_NativeAEC())


def test_aec_declared_multichannel_but_mono_contract_fails():
    from core.models.io_contract import ModelIOContract, TensorField
    from tasks.aec.model_io import validate_aec_multichannel_contract

    class _FalseNativeAEC:
        supports_multichannel_input = True
        IO_CONTRACT = ModelIOContract(
            tasks=frozenset({"aec"}),
            inputs=(
                TensorField("mic_spectra", "spectrum", ("batch", "frequency", "frame", "complex")),
                TensorField("farend_spectra", "spectrum", ("batch", "frequency", "frame", "complex")),
            ),
            outputs=(TensorField("nearend_estimate_spectra", "spectrum", ("batch", "frequency", "frame", "complex")),),
            forward_kind="estimate",
            frontend_owner="runtime",
        )

    with pytest.raises(ValueError, match="microphone axis"):
        validate_aec_multichannel_contract(_FalseNativeAEC())
