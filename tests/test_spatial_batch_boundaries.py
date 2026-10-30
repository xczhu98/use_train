"""Spatial batch shape and finite-value boundary regressions."""

from __future__ import annotations

import pytest
import torch

from core.data.collate import collate_fn
from core.data.simulation.array_profiles import (
    ArrayTopology,
    MicrophoneArrayProfile,
    ResolvedMicrophoneArray,
)
from core.data.source_separation_batch import SSBatch
from core.losses.pit import PITLoss
from tasks.aec.data.collate import aec_collate_fn
from tasks.tse.cue_contract import TSECueSpecs, TSECues
from tasks.tse.data.collate import make_tse_sample, make_tse_target, tse_collate_fn


pytestmark = pytest.mark.unit

_TSE_CUE_SPECS = TSECueSpecs()


def _tse_geometry() -> ResolvedMicrophoneArray:
    topology = ArrayTopology(((-0.04, 0.0, 0.0), (0.04, 0.0, 0.0)), spacing_m=0.08)
    return ResolvedMicrophoneArray(
        MicrophoneArrayProfile("tse-boundary-fixture", topology),
        reference_channel=0,
    )


def _tse_sample(mixture: torch.Tensor, length: int, *, spatial: bool) -> dict:
    return make_tse_sample(
        key=f"tse-boundary-{length}-{int(spatial)}",
        mixture=mixture,
        sr=16000,
        length=length,
        targets=[
            make_tse_target(
                target=torch.zeros(length),
                cues=TSECues(),
                speaker_id=None,
                spk_label=-1,
            )
        ],
        geometry=_tse_geometry() if spatial else None,
    )


def test_explicit_se_spatial_c2_t1_is_not_mono_column_vector():
    batch = collate_fn(
        [(torch.zeros(1), torch.ones(2, 1), 16000, 1)],
        allow_spatial_noisy=True,
    )

    assert batch.noisy.shape == (1, 2, 1)


def test_se_legacy_t_by_one_mono_layout_remains_supported():
    batch = collate_fn([(torch.zeros(2), torch.ones(2, 1), 16000, 2)])

    assert batch.noisy.shape == (1, 2)


def test_aec_spatial_c2_t1_and_legacy_mono_layouts():
    def sample(mic, length):
        return (torch.zeros(length), mic, torch.zeros(length), 16000, length, torch.zeros(length), True)

    spatial = aec_collate_fn([sample(torch.ones(2, 1), 1)])
    legacy = aec_collate_fn([sample(torch.ones(2, 1), 2)])

    assert spatial.mic.shape == (1, 2, 1)
    assert legacy.mic.shape == (1, 2)


def test_tse_spatial_c2_t1_and_legacy_mono_layouts():
    spatial = tse_collate_fn(
        [_tse_sample(torch.ones(2, 1), 1, spatial=True)],
        cue_specs=_TSE_CUE_SPECS,
    )
    legacy = tse_collate_fn(
        [_tse_sample(torch.ones(2, 1), 2, spatial=False)],
        cue_specs=_TSE_CUE_SPECS,
    )

    assert spatial.mixture.shape == (1, 2, 1)
    assert legacy.mixture.shape == (1, 1, 2)


def test_aec_and_tse_reject_nonfinite_spatial_observations():
    aec_sample = (
        torch.zeros(1),
        torch.tensor([[float("nan")], [0.0]]),
        torch.zeros(1),
        16000,
        1,
        torch.zeros(1),
        True,
    )
    tse_sample = _tse_sample(
        torch.tensor([[float("nan")], [0.0]]),
        1,
        spatial=True,
    )

    with pytest.raises(RuntimeError, match="NaN/Inf"):
        aec_collate_fn([aec_sample])
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        tse_collate_fn([tse_sample], cue_specs=_TSE_CUE_SPECS)


def _ss_batch(*, mixture, sources, lengths=None, source_mask=None, sr=None):
    batch, source_count, time = sources.shape[:3]
    return SSBatch(
        mixture=mixture,
        sources=sources,
        lengths=torch.tensor([time] * batch) if lengths is None else lengths,
        source_mask=(
            torch.ones(batch, source_count, dtype=torch.bool)
            if source_mask is None
            else source_mask
        ),
        sr=torch.tensor([16000] * batch) if sr is None else sr,
    )


@pytest.mark.parametrize(
    ("mixture", "sources", "message"),
    [
        (torch.zeros(0, 4), torch.zeros(0, 2, 4), "B must be positive"),
        (torch.zeros(1, 0, 4), torch.zeros(1, 2, 4), "C must be positive"),
        (torch.zeros(1, 2, 0), torch.zeros(1, 2, 0), "T must be positive"),
        (torch.zeros(1, 4), torch.zeros(1, 0, 4), "K must be positive"),
        (torch.zeros(1, 2, 4), torch.zeros(1, 2, 2, 4), r"shape \[B, K, T\]"),
        (torch.zeros(1, 4), torch.zeros(1, 2, 3), "aligned with mixture"),
    ],
)
def test_ss_batch_rejects_empty_axes_future_target_shape_and_misalignment(
    mixture, sources, message
):
    batch = _ss_batch(mixture=mixture, sources=sources)

    with pytest.raises((TypeError, ValueError), match=message):
        batch.validate()


@pytest.mark.parametrize("field", ["mixture", "sources"])
def test_ss_batch_rejects_nonfinite_inputs(field):
    mixture = torch.zeros(1, 4)
    sources = torch.zeros(1, 2, 4)
    (mixture if field == "mixture" else sources).flatten()[0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        _ss_batch(mixture=mixture, sources=sources).validate()


@pytest.mark.unit
def test_ss_batch_reports_nonfinite_waveform_before_other_batch_slices():
    mixture = torch.full((1, 4), float("nan"))
    sources = torch.zeros(1, 2, 4)
    batch = _ss_batch(
        mixture=mixture,
        sources=sources,
        lengths=torch.tensor([0]),
        source_mask=torch.zeros(1, 2, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="finite"):
        batch.validate()


def test_ss_batch_rejects_partial_source_mask_without_owning_loss_semantics():
    mixture = torch.zeros(1, 4)
    sources = torch.zeros(1, 2, 4)

    for lengths in (torch.tensor([0]), torch.tensor([5]), torch.tensor([4.0])):
        with pytest.raises((TypeError, ValueError), match="length"):
            _ss_batch(mixture=mixture, sources=sources, lengths=lengths).validate()

    with pytest.raises(ValueError, match="source_mask"):
        _ss_batch(
            mixture=mixture,
            sources=sources,
            source_mask=torch.zeros(1, 2, dtype=torch.bool),
        ).validate()
    with pytest.raises(TypeError, match="source_mask"):
        _ss_batch(
            mixture=mixture,
            sources=sources,
            source_mask=torch.ones(1, 2, dtype=torch.int64),
        ).validate()

    partial_mask_batch = _ss_batch(
        mixture=mixture,
        sources=sources,
        source_mask=torch.tensor([[True, False]]),
    )
    with pytest.raises(ValueError, match="source_mask"):
        partial_mask_batch.validate()
    with pytest.raises(ValueError, match="variable source count"):
        PITLoss(pairwise_loss="PairwiseL1Loss").compute(
            sources,
            sources,
            source_mask=partial_mask_batch.source_mask,
            lengths=partial_mask_batch.lengths,
        )
