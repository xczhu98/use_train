"""TSE trainer/inference 共享模型 I/O 边界回归测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from core.stft import StftIstftModule
from tasks.tse.inference.extractor import TSEExtractor
from tasks.tse.model_io import (
    build_tse_model_inputs,
    decode_tse_model_output,
    guard_tse_model_sample_rate,
)
from tasks.tse.models.discriminative.tse_bsrnn_spk import TSE_BSRNN_SPK
from tasks.tse.trainer.tse_trainer import TSETrainer

SR = 16000


class _IdentitySpectralModel(nn.Module):
    sample_rate = SR

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[dict] = []

    def forward(self, data: dict) -> dict:
        self.inputs.append({key: value for key, value in data.items()})
        data["target_estimate_spectra"] = data["mix_spectra"]
        return data


class _ShortWaveformModel(_IdentitySpectralModel):
    def forward(self, data: dict) -> dict:
        self.inputs.append({key: value for key, value in data.items()})
        data["target_estimate"] = data["mixture"][..., :-17]
        return data


class _CountingSTFT:
    def __init__(self) -> None:
        self.inner = StftIstftModule(n_fft=64, hop_length=16, win_length=64)
        self.forward_shapes: list[tuple[int, ...]] = []
        self.hop_length = self.inner.hop_length

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        self.forward_shapes.append(tuple(waveform.shape))
        return self.inner(waveform)

    def inverse(self, spectra: torch.Tensor, length: int) -> torch.Tensor:
        return self.inner.inverse(spectra, length)


class _CueAwareModel(_IdentitySpectralModel):
    def __init__(self, fields: set[str]) -> None:
        super().__init__()
        self.fields = frozenset(fields | {"mix_spectra"})

    def required_tse_input_fields(self) -> frozenset[str]:
        return self.fields


def _bare_extractor(
    model: nn.Module,
    stft: StftIstftModule,
    *,
    output_peak: float | None,
) -> TSEExtractor:
    extractor = object.__new__(TSEExtractor)
    extractor.device = torch.device("cpu")
    extractor.model = model
    extractor.stft = stft
    extractor.sample_rate = SR
    extractor.apply_vad = False
    extractor.output_peak = output_peak
    extractor._uses_preextracted_spk_embedding = False
    return extractor


def test_shared_builder_and_extractor_forward_are_numerically_identical():
    """同模型/STFT/PCM 下，共享 builder 与 inference 整句前向严格对齐。"""
    torch.manual_seed(0)
    mixture = torch.randn(1, 1024) * 0.1
    enroll = torch.randn(1, 768) * 0.1
    stft = StftIstftModule(n_fft=64, hop_length=16, win_length=64)
    model = _IdentitySpectralModel()
    extractor = _bare_extractor(model, stft, output_peak=None)

    shared_data = build_tse_model_inputs(
        model=model,
        stft=stft,
        mixture=mixture,
        enroll=enroll,
        sample_rate=SR,
    )
    shared_data = decode_tse_model_output(
        model(shared_data),
        stft=stft,
        reference_length=mixture.shape[-1],
    )
    infer_wav = extractor.extract_speech_from_pcm(
        mixture.squeeze(0).numpy(),
        SR,
        enroll.squeeze(0).numpy(),
        SR,
    )

    expected_keys = {
        "mixture",
        "mix_spectra",
        "enroll_spectra",
        "enroll_wav",
        "sample_rate",
    }
    assert set(model.inputs[0]) == expected_keys
    assert set(model.inputs[1]) == expected_keys
    np.testing.assert_allclose(
        infer_wav,
        shared_data["target_estimate"].squeeze(0).numpy(),
        atol=1e-6,
    )


def test_multi_target_builder_analyzes_mixture_once_and_aligns_rows():
    """多目标只分析 B 条 mixture 一次，再在谱域按 sample-major 展开。"""
    stft = _CountingSTFT()
    model = _CueAwareModel({"enroll_spectra", "enroll_spectra_lengths", "enroll_wav"})
    mixture = torch.randn(2, 256)
    enroll = torch.randn(3, 128)
    target = torch.randn(3, 256)
    lengths = torch.tensor([200, 240], dtype=torch.int32)
    enroll_lengths = torch.tensor([100, 110, 120], dtype=torch.int32)
    num_targets = torch.tensor([2, 1], dtype=torch.int32)
    embedding = torch.randn(3, 12)

    data = build_tse_model_inputs(
        model=model,
        stft=stft,
        mixture=mixture,
        enroll=enroll,
        target=target,
        lengths=lengths,
        enroll_lengths=enroll_lengths,
        spk_embedding=embedding,
        num_targets=num_targets,
        sample_rate=SR,
    )

    assert stft.forward_shapes == [(2, 256), (1, 100), (1, 110), (1, 120)]
    assert data["enroll_spectra_lengths"].tolist() == [7, 7, 8]
    assert data["mix_spectra"].shape[0] == 3
    assert data["mixture"].shape == target.shape
    assert data["lengths"].tolist() == [200, 200, 240]
    assert torch.count_nonzero(data["enroll_wav"][0, 100:]) == 0
    assert "spk_embedding" not in data


def test_multichannel_mixture_requires_native_model_contract():
    mixture = torch.randn(1, 2, 256)
    enroll = torch.randn(1, 128)
    stft = _CountingSTFT()
    with pytest.raises(ValueError, match="native model IO contract.*microphone axis"):
        build_tse_model_inputs(
            model=_IdentitySpectralModel(),
            stft=stft,
            mixture=mixture,
            enroll=enroll,
            sample_rate=SR,
        )

def test_existing_bsrnn_rejects_spatial_mixture_without_native_contract():
    model = TSE_BSRNN_SPK(
        sr=SR,
        feature_dim=8,
        num_repeat=1,
        n_fft=512,
        hop_length=128,
        win_length=512,
    ).eval()
    stft = StftIstftModule(n_fft=512, hop_length=128, win_length=512)
    with pytest.raises(ValueError, match="native model IO contract"):
        build_tse_model_inputs(
            model=model,
            stft=stft,
            mixture=torch.randn(1, 2, 1024),
            enroll=torch.randn(1, 128),
            sample_rate=SR,
        )


def test_bsrnn_rejects_removed_observation_adapter_config():
    with pytest.raises(TypeError, match="unexpected keyword arguments.*observation_adapter"):
        TSE_BSRNN_SPK(
            sr=SR,
            feature_dim=8,
            num_repeat=1,
            observation_adapter={"type": "reference", "reference_channel": 0},
        )


def test_extractor_keeps_mono_model_contract():
    model = _IdentitySpectralModel()
    stft = StftIstftModule(n_fft=64, hop_length=16, win_length=64)
    extractor = _bare_extractor(model, stft, output_peak=None)
    mixture = torch.randn(1, 1024)
    output = extractor.extract_speech_from_pcm(
        mixture.numpy(), SR, torch.randn(512).numpy(), SR
    )
    assert output.shape == (1024,)
    np.testing.assert_allclose(output, mixture[0].numpy(), atol=1e-5)


def test_cue_aware_builder_skips_unused_enrollment_stft():
    """仅 waveform cue 时不应为 enrollment 无条件执行 STFT。"""
    stft = _CountingSTFT()
    model = _CueAwareModel({"enroll_wav", "enroll_lengths"})
    mixture = torch.randn(2, 256)
    enroll = torch.randn(2, 128)
    enroll_lengths = torch.tensor([80, 128], dtype=torch.int32)

    data = build_tse_model_inputs(
        model=model,
        stft=stft,
        mixture=mixture,
        enroll=enroll,
        enroll_lengths=enroll_lengths,
        sample_rate=SR,
    )

    assert stft.forward_shapes == [(2, 256)]
    assert set(data) == {
        "mixture",
        "mix_spectra",
        "enroll_wav",
        "enroll_lengths",
        "sample_rate",
    }
    assert data["enroll_lengths"] is enroll_lengths


def test_cue_aware_builder_only_builds_preextracted_embedding_input():
    """非 joint spkemb 不生成 enrollment 谱，也不把波形喷入模型字段。"""
    stft = _CountingSTFT()
    model = _CueAwareModel({"spk_embedding"})
    embedding = torch.randn(1, 12)

    data = build_tse_model_inputs(
        model=model,
        stft=stft,
        mixture=torch.randn(1, 256),
        enroll=torch.randn(1, 128),
        spk_embedding=embedding,
        sample_rate=SR,
    )

    assert stft.forward_shapes == [(1, 256)]
    assert "enroll_spectra" not in data
    assert "enroll_wav" not in data
    assert data["spk_embedding"] is embedding


def test_extractor_direct_waveform_output_restores_length_and_peak():
    """共享 decoder 的兼容波形路径不破坏推理裁长与 peak normalization。"""
    torch.manual_seed(1)
    mixture = (torch.randn(1024) * 0.1).numpy()
    enroll = (torch.randn(512) * 0.1).numpy()
    stft = StftIstftModule(n_fft=64, hop_length=16, win_length=64)
    extractor = _bare_extractor(_ShortWaveformModel(), stft, output_peak=0.5)

    output = extractor.extract_speech_from_pcm(mixture, SR, enroll, SR)

    assert output.shape == mixture.shape
    assert np.isfinite(output).all()
    assert float(np.abs(output).max()) == pytest.approx(0.5, abs=1e-6)


def test_shared_sample_rate_guard_and_missing_output_fail_loudly():
    model = _IdentitySpectralModel()
    with pytest.raises(ValueError, match="model.sample_rate"):
        guard_tse_model_sample_rate(model, 8000)

    stft = StftIstftModule(n_fft=64, hop_length=16, win_length=64)
    with pytest.raises(RuntimeError, match="target_estimate_spectra.*target_estimate"):
        decode_tse_model_output(
            {"unrelated": torch.zeros(1)},
            stft=stft,
            reference_length=128,
        )


@pytest.mark.parametrize("tuple_length", [7, 8, 9])
def test_legacy_tuples_are_rejected_at_trainer_boundary(tuple_length):
    """7/8/9 元组均不得恢复为生产兼容入口。"""
    legacy = tuple(torch.zeros(1) for _ in range(tuple_length))
    with pytest.raises(TypeError, match="TSEBatch.*tuple batches are not accepted"):
        object.__new__(TSETrainer)._unpack_batch(legacy)


pytestmark = [pytest.mark.unit]


def test_tse_native_multichannel_contract_requires_actual_microphone_axis():
    from core.models.io_contract import ModelIOContract, TensorField
    from tasks.tse.model_io import validate_tse_multichannel_contract

    class _NativeTSE:
        supports_multichannel_input = True
        IO_CONTRACT = ModelIOContract(
            tasks=frozenset({"tse"}),
            inputs=(
                TensorField("mix_spectra", "spectrum", ("batch", "microphone", "frequency", "frame", "complex")),
                TensorField("enroll_spectra", "spectrum", ("batch", "frequency", "frame", "complex"), required=False),
            ),
            outputs=(TensorField("target_estimate_spectra", "spectrum", ("batch", "frequency", "frame", "complex")),),
            forward_kind="estimate",
            frontend_owner="runtime",
        )

    validate_tse_multichannel_contract(_NativeTSE())


def test_tse_declared_multichannel_but_mono_contract_fails():
    from core.models.io_contract import ModelIOContract, TensorField
    from tasks.tse.model_io import validate_tse_multichannel_contract

    class _FalseNativeTSE:
        supports_multichannel_input = True
        IO_CONTRACT = ModelIOContract(
            tasks=frozenset({"tse"}),
            inputs=(TensorField("mix_spectra", "spectrum", ("batch", "frequency", "frame", "complex")),),
            outputs=(TensorField("target_estimate_spectra", "spectrum", ("batch", "frequency", "frame", "complex")),),
            forward_kind="estimate",
            frontend_owner="runtime",
        )

    with pytest.raises(ValueError, match="microphone axis"):
        validate_tse_multichannel_contract(_FalseNativeTSE())
