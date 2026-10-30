"""TSE spatial length-mask regressions."""

from __future__ import annotations

import pytest
import torch

from core.data.simulation.array_profiles import resolve_microphone_array
from core.models.io_contract import ModelIOContract, TensorField
from tasks.tse.cue_contract import TSECue, TSECueSpec, TSECueSpecs, TSECues
from tasks.tse.data.collate import make_tse_sample, make_tse_target, tse_collate_fn
from tasks.tse.model_io import build_tse_model_inputs

pytestmark = pytest.mark.unit


class _STFT:
    n_fft = 4
    hop_length = 2

    def __call__(self, waveform):
        return torch.zeros(waveform.shape[0], 3, 2, 2)


class _Model:
    sample_rate = 16000
    supports_multichannel_input = True
    reference_channel = 0
    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"tse"}),
        inputs=(
            TensorField(
                "mix_spectra",
                "spectrum",
                ("batch", "microphone", "frequency", "frame", "complex"),
            ),
            TensorField("speaker_waveform", "waveform", ("batch", "time")),
        ),
        outputs=(
            TensorField(
                "target_estimate_spectra",
                "spectrum",
                ("batch", "frequency", "frame", "complex"),
            ),
        ),
        forward_kind="estimate",
        frontend_owner="runtime",
    )


def test_short_spatial_mixture_masks_last_time_axis():
    mixture = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    cue_specs = TSECueSpecs(
        speaker_waveform=TSECueSpec("waveform", (3,))
    )
    sample = make_tse_sample(
        key="spatial-short",
        mixture=mixture,
        sr=16000,
        length=3,
        targets=[
            make_tse_target(
                target=torch.ones(5),
                cues=TSECues(
                    speaker_waveform=TSECue(
                        "waveform", torch.ones(3), True, 3, None
                    )
                ),
                speaker_id=None,
                spk_label=-1,
            )
        ],
        geometry=resolve_microphone_array(
            array_profile="linear_2ch_80mm", reference_channel=0
        ),
    )
    batch = tse_collate_fn([sample], cue_specs=cue_specs)

    assert batch.mixture.shape == (1, 2, 5)
    torch.testing.assert_close(batch.mixture[0, :, 3:], torch.zeros(2, 2))
    torch.testing.assert_close(batch.mixture[0, :, :3], mixture[:, :3])

    inputs = build_tse_model_inputs(
        model=_Model(),
        stft=_STFT(),
        mixture=batch.mixture,
        cues=batch.cues,
        target=batch.target,
        sample_rate=16000,
        lengths=batch.lengths,
        num_targets=batch.num_targets,
        reference_channel=batch.geometry[0].reference_channel,
    )
    torch.testing.assert_close(inputs["mixture"][0, 3:], torch.zeros(2))
    torch.testing.assert_close(inputs["mixture"][0, :3], mixture[0, :3])
    assert inputs["mix_spectra"].shape == (1, 2, 3, 2, 2)


def test_tse_model_io_keeps_enrollment_and_target_mono_for_spatial_mixture():
    cue_specs = TSECueSpecs(
        speaker_waveform=TSECueSpec("waveform", (4,))
    )
    sample = make_tse_sample(
        key="spatial-mono-cue",
        mixture=torch.zeros(2, 6),
        sr=16000,
        length=6,
        targets=[
            make_tse_target(
                target=torch.zeros(6),
                cues=TSECues(
                    speaker_waveform=TSECue(
                        "waveform", torch.zeros(4), True, 4, None
                    )
                ),
                speaker_id=None,
            )
        ],
        geometry=resolve_microphone_array(
            array_profile="linear_2ch_80mm", reference_channel=0
        ),
    )
    batch = tse_collate_fn([sample], cue_specs=cue_specs)
    inputs = build_tse_model_inputs(
        model=_Model(),
        stft=_STFT(),
        mixture=batch.mixture,
        target=batch.target,
        cues=batch.cues,
        sample_rate=16000,
        lengths=batch.lengths,
        num_targets=batch.num_targets,
        reference_channel=batch.geometry[0].reference_channel,
    )
    assert inputs["mixture"].shape == (1, 6)
    assert inputs["target"].shape == (1, 6)
    assert inputs["speaker_waveform"].shape == (1, 4)
    assert inputs["mix_spectra"].shape[1] == 2
