"""TSE typed cue 与 waveform/spectrum ModelIOContract 装配测试。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from core.models.io_contract import ModelIOContract, TensorField
from core.stft import StftIstftModule
from tasks.tse.cue_contract import TSECueBatch, TSECueBatches
from tasks.tse.model_io import build_tse_model_inputs, decode_tse_model_output


SR = 16000


class _ContractModel(nn.Module):
    sample_rate = SR

    def __init__(
        self,
        inputs: tuple[TensorField, ...],
        *,
        output: TensorField | None = None,
        supports_multichannel: bool = False,
        ref_channel: int | None = None,
    ) -> None:
        super().__init__()
        self.IO_CONTRACT = ModelIOContract(
            tasks=frozenset({"tse"}),
            inputs=inputs,
            outputs=(
                output
                or TensorField(
                    "target_estimate", "waveform", ("batch", "time")
                ),
            ),
            forward_kind="estimate",
            frontend_owner="runtime",
        )
        self.supports_multichannel_input = supports_multichannel
        if ref_channel is not None:
            self.ref_channel = ref_channel


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


def _temporal_cue(
    representation: str,
    value: torch.Tensor,
    present: list[bool],
    lengths: list[int],
    *,
    timebase: torch.Tensor | None = None,
) -> TSECueBatch:
    return TSECueBatch(
        representation=representation,
        value=value,
        present=torch.tensor(present, dtype=torch.bool),
        lengths=torch.tensor(lengths, dtype=torch.int32),
        timebase=timebase,
    )


def test_no_enrollment_waveform_contract_does_not_require_stft():
    model = _ContractModel(
        (TensorField("mixture", "waveform", ("batch", "time")),)
    )
    mixture = torch.randn(2, 160)

    data = build_tse_model_inputs(
        model=model,
        stft=None,
        mixture=mixture,
        sample_rate=SR,
    )

    assert set(data) == {"mixture", "sample_rate"}
    assert data["mixture"] is mixture


def test_typed_cues_map_values_presence_lengths_and_timebases():
    inputs = (
        TensorField("mixture", "waveform", ("batch", "time")),
        TensorField("speaker_waveform", "waveform", ("batch", "time")),
        TensorField("speaker_waveform_present", "mask", ("batch",)),
        TensorField("speaker_waveform_lengths", "label", ("batch",)),
        TensorField("speaker_embedding", "embedding", ("batch", "embedding")),
        TensorField("speaker_embedding_present", "mask", ("batch",)),
        TensorField("spatial", "embedding", ("batch", "coordinate", "point")),
        TensorField("spatial_present", "mask", ("batch",)),
        TensorField("spatial_lengths", "label", ("batch",)),
        TensorField("spatial_timebase", "label", ("batch", "point")),
        TensorField("visual", "embedding", ("batch", "feature", "frame")),
        TensorField("visual_present", "mask", ("batch",)),
        TensorField("visual_lengths", "label", ("batch",)),
        TensorField("visual_timebase", "label", ("batch", "frame")),
        TensorField("textual", "label", ("batch", "token")),
        TensorField("textual_present", "mask", ("batch",)),
        TensorField("textual_lengths", "label", ("batch",)),
    )
    speaker_waveform = _temporal_cue(
        "waveform",
        torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        [True, False],
        [2, 0],
    )
    speaker_embedding = TSECueBatch(
        representation="embedding",
        value=torch.tensor([[0.1, 0.2], [0.0, 0.0]]),
        present=torch.tensor([True, False]),
        lengths=None,
        timebase=None,
    )
    spatial = _temporal_cue(
        "track",
        torch.tensor(
            [
                [[0.1, 0.2, 0.0], [0.3, 0.4, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ),
        [True, False],
        [2, 0],
        timebase=torch.tensor([[0, 160, 0], [0, 0, 0]], dtype=torch.int64),
    )
    visual = _temporal_cue(
        "features",
        torch.tensor(
            [
                [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ),
        [True, False],
        [2, 0],
        timebase=torch.tensor(
            [[0.0, 0.04, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
        ),
    )
    textual = _temporal_cue(
        "phoneme_ids",
        torch.tensor([[4, 5, 0], [0, 0, 0]], dtype=torch.int64),
        [True, False],
        [2, 0],
    )
    cues = TSECueBatches(
        speaker_waveform=speaker_waveform,
        speaker_embedding=speaker_embedding,
        spatial=spatial,
        visual=visual,
        textual=textual,
    )

    data = build_tse_model_inputs(
        model=_ContractModel(inputs),
        stft=None,
        mixture=torch.randn(2, 64),
        cues=cues,
        sample_rate=SR,
    )

    assert data["speaker_waveform"] is speaker_waveform.value
    assert data["speaker_waveform_present"].tolist() == [True, False]
    assert data["speaker_waveform_lengths"].tolist() == [2, 0]
    assert data["speaker_embedding"] is speaker_embedding.value
    assert data["spatial_timebase"] is spatial.timebase
    assert data["visual_lengths"].tolist() == [2, 0]
    assert data["textual"].tolist() == [[4, 5, 0], [0, 0, 0]]


def test_missing_cue_rows_require_declared_presence_field():
    model = _ContractModel(
        (
            TensorField("mixture", "waveform", ("batch", "time")),
            TensorField("textual", "label", ("batch", "token")),
            TensorField("textual_lengths", "label", ("batch",)),
        )
    )
    cues = TSECueBatches(
        textual=_temporal_cue(
            "phoneme_ids",
            torch.zeros(1, 3, dtype=torch.int64),
            [False],
            [0],
        )
    )

    with pytest.raises(ValueError, match="no presence field"):
        build_tse_model_inputs(
            model=model,
            stft=None,
            mixture=torch.randn(1, 64),
            cues=cues,
            sample_rate=SR,
        )


def test_enabled_unknown_cue_is_not_silently_ignored():
    model = _ContractModel(
        (TensorField("mixture", "waveform", ("batch", "time")),)
    )
    cues = TSECueBatches(
        spatial=TSECueBatch(
            representation="static",
            value=torch.tensor([[0.1, -0.2]]),
            present=torch.tensor([True]),
            lengths=None,
            timebase=None,
        )
    )

    with pytest.raises(ValueError, match="does not consume"):
        build_tse_model_inputs(
            model=model,
            stft=None,
            mixture=torch.randn(1, 64),
            cues=cues,
            sample_rate=SR,
        )


def test_required_cue_field_missing_but_optional_field_may_be_absent():
    required = _ContractModel(
        (
            TensorField("mixture", "waveform", ("batch", "time")),
            TensorField("textual", "label", ("batch", "token")),
        )
    )
    with pytest.raises(ValueError, match="missing required.*textual"):
        build_tse_model_inputs(
            model=required,
            stft=None,
            mixture=torch.randn(1, 64),
            sample_rate=SR,
        )

    optional = _ContractModel(
        (
            TensorField("mixture", "waveform", ("batch", "time")),
            TensorField(
                "textual", "label", ("batch", "token"), required=False
            ),
        )
    )
    data = build_tse_model_inputs(
        model=optional,
        stft=None,
        mixture=torch.randn(1, 64),
        sample_rate=SR,
    )
    assert "textual" not in data


@pytest.mark.parametrize(
    "field, message",
    [
        (
            TensorField("spatial_typo", "embedding", ("batch", "coordinate")),
            "unsupported TSE model input field",
        ),
        (
            TensorField("textual", "embedding", ("batch", "token")),
            "must use domain",
        ),
        (
            TensorField(
                "mixture", "waveform", ("batch", "microphone", "time")
            ),
            "reference-channel loss waveform",
        ),
    ],
)
def test_contract_rejects_unknown_fields_wrong_domains_and_mixture_mic_axis(
    field, message
):
    model = _ContractModel((field,))
    with pytest.raises(ValueError, match=message):
        build_tse_model_inputs(
            model=model,
            stft=None,
            mixture=torch.randn(1, 64),
            sample_rate=SR,
        )


def test_multichannel_spectrum_keeps_microphones_but_loss_uses_reference():
    model = _ContractModel(
        (
            TensorField(
                "mix_spectra",
                "spectrum",
                ("batch", "microphone", "frequency", "frame", "complex"),
            ),
        ),
        output=TensorField(
            "target_estimate_spectra",
            "spectrum",
            ("batch", "frequency", "frame", "complex"),
        ),
        supports_multichannel=True,
        ref_channel=1,
    )
    stft = _CountingSTFT()
    mixture = torch.randn(2, 3, 256)
    num_targets = torch.tensor([2, 1], dtype=torch.int32)

    data = build_tse_model_inputs(
        model=model,
        stft=stft,
        mixture=mixture,
        num_targets=num_targets,
        sample_rate=SR,
    )

    assert stft.forward_shapes == [(6, 256)]
    assert data["mix_spectra"].shape[:2] == (3, 3)
    expected_reference = torch.repeat_interleave(
        mixture[:, 1], num_targets.to(torch.int64), dim=0
    )
    torch.testing.assert_close(data["mixture"], expected_reference)


def test_multichannel_reference_channel_is_validated():
    model = _ContractModel(
        (
            TensorField(
                "mix_wav", "waveform", ("batch", "microphone", "time")
            ),
        ),
        supports_multichannel=True,
    )
    mixture = torch.randn(1, 2, 64)

    with pytest.raises(ValueError, match="explicit model/call reference_channel"):
        build_tse_model_inputs(
            model=model,
            stft=None,
            mixture=mixture,
            sample_rate=SR,
        )
    with pytest.raises(ValueError, match="must be in"):
        build_tse_model_inputs(
            model=model,
            stft=None,
            mixture=mixture,
            reference_channel=2,
            sample_rate=SR,
        )

    data = build_tse_model_inputs(
        model=model,
        stft=None,
        mixture=mixture,
        reference_channel=0,
        sample_rate=SR,
    )
    assert data["mix_wav"].shape == (1, 2, 64)
    assert data["mixture"].shape == (1, 64)


def test_waveform_decoder_without_stft_does_not_create_spectra():
    estimate = torch.randn(2, 48)
    data = decode_tse_model_output(
        {"target_estimate": estimate},
        stft=None,
        reference_length=64,
    )

    assert data["target_estimate"].shape == (2, 64)
    assert "target_estimate_spectra" not in data


def test_spectral_decoder_requires_stft():
    with pytest.raises(ValueError, match="requires a runtime STFT"):
        decode_tse_model_output(
            {"target_estimate_spectra": torch.zeros(1, 4, 4, 2)},
            stft=None,
            reference_length=64,
        )


pytestmark = [pytest.mark.unit]
