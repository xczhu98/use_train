"""AEC 审计修复的回归测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tasks.aec.data.batch import AECBatch
from tasks.aec.data.dynamic_dataset import scale_signal_to_dBfs_rms
from tasks.aec.data.linear_aec_frontend import LinearAECFrontend
from tasks.aec.data.pfdkf import PFDKF
from tasks.aec.data.pfdkf_torch import PFDKFTorch
from tasks.aec.data.state_label import compute_aec_state_label
from tasks.aec.data.time_delay_jitter import _apply_transition_ramps
from tasks.aec.metrics.state_metrics import compute_state_counts, state_metrics_from_counts
from tasks.aec.trainer.aec_trainer import AECTrainer


class _RecordingStft:
    n_fft = 8
    hop_length = 4

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        self.inputs.append(waveform.clone())
        return waveform.clone()

    def inverse(self, spectra: torch.Tensor, length: int) -> torch.Tensor:
        return spectra[..., :length]


class _ReplacingModel(torch.nn.Module):
    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"nearend_estimate_spectra": data["mic_spectra"]}


def _batch() -> AECBatch:
    return AECBatch(
        nearend=torch.tensor([[1.0, 2.0, 3.0, 99.0, 99.0]]),
        mic=torch.tensor([[4.0, 5.0, 6.0, 99.0, 99.0]]),
        farend=torch.tensor([[7.0, 8.0, 9.0, 99.0, 99.0]]),
        fs=torch.tensor(16000, dtype=torch.int32),
        lengths=torch.tensor([3], dtype=torch.int32),
        error=torch.tensor([[1.0, 1.0, 1.0, 42.0, 42.0]]),
        is_pure_noise=torch.tensor([True]),
        state=torch.tensor([[0, 1]], dtype=torch.long),
    )


@pytest.mark.unit
def test_trainer_masks_tail_before_stft_and_restores_batch_metadata():
    trainer = object.__new__(AECTrainer)
    trainer.stft_module = _RecordingStft()
    trainer._aec_frontend_enabled = False
    trainer.model = _ReplacingModel()
    batch = _batch()

    data = trainer._forward_aec_batch(batch)

    assert all(torch.equal(value[0, 3:], torch.zeros(2)) for value in trainer.stft_module.inputs)
    assert torch.equal(batch.error[0, 3:], torch.zeros(2))
    assert data["lengths"] is batch.lengths
    assert data["is_pure_noise"] is batch.is_pure_noise
    assert data["state"] is batch.state


@pytest.mark.unit
def test_legacy_error_detection_ignores_declared_tail():
    batch = _batch()
    batch.error[:, :3] = 0.0
    batch.error_mask = None

    assert not AECTrainer._batch_has_valid_error(batch)


@pytest.mark.unit
def test_strict_batch_validation_raises_explicit_errors():
    trainer = object.__new__(AECTrainer)
    valid = torch.zeros(1, 4)
    lengths = torch.tensor([4])

    with pytest.raises(ValueError, match=r"must be \[B, T\]"):
        trainer._assert_batch_valid(valid[0], valid, valid, lengths)
    with pytest.raises(ValueError, match="Shape mismatch"):
        trainer._assert_batch_valid(valid, torch.zeros(1, 3), valid, lengths)
    with pytest.raises(ValueError, match="NaN/Inf"):
        trainer._assert_batch_valid(
            valid.fill_(float("nan")), torch.zeros(1, 4), torch.zeros(1, 4), lengths
        )


@pytest.mark.unit
def test_trainer_rejects_multilabel_state_batch_even_without_collate():
    trainer = object.__new__(AECTrainer)
    trainer.stft_module = _RecordingStft()
    batch = _batch()
    batch.state = torch.zeros(1, 2, 3, dtype=torch.int64)

    with pytest.raises(ValueError, match="mutually-exclusive class indices"):
        trainer._mask_state_padding(batch)


@pytest.mark.unit
def test_disabled_state_head_skips_metric_accumulators_and_reductions(monkeypatch):
    trainer = object.__new__(AECTrainer)
    trainer.device = torch.device("cpu")
    trainer.world_size = 2
    trainer._vm_max_erle = 0
    trainer.loss_fn = object()
    raw_model = type(
        "DisabledStateHead",
        (),
        {"use_state_detector": False, "state_nclass": 3},
    )()
    trainer._raw_model = lambda: raw_model
    monkeypatch.setattr(
        "tasks.aec.trainer.aec_trainer.dist.all_reduce",
        lambda *_args, **_kwargs: pytest.fail("disabled state metrics must not reduce"),
    )

    extra = trainer._init_val_extra_state()
    trainer._val_post_allreduce_extra(extra, val_x=0)

    assert extra["state_metrics_available"] is False
    assert extra["state_tp"] is None
    assert extra["state_metrics"] is None


@pytest.mark.unit
def test_state_metrics_reject_invalid_tensor_contracts():
    logits = torch.zeros(1, 2, 2)
    labels = torch.tensor([[0, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="NaN or Inf"):
        compute_state_counts(logits.fill_(float("nan")), labels)
    with pytest.raises(TypeError, match="integer dtype"):
        compute_state_counts(logits, labels.float())
    with pytest.raises(ValueError, match="non-negative"):
        state_metrics_from_counts(
            torch.tensor([-1], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )


@pytest.mark.unit
def test_state_label_rejects_invalid_framing_and_nonfinite_waveform():
    waveform = np.ones(32, dtype=np.float32)

    with pytest.raises(ValueError, match="positive integer"):
        compute_aec_state_label(waveform, waveform, hop_length=0)
    with pytest.raises(ValueError, match="finite numeric"):
        compute_aec_state_label(np.array([np.nan], dtype=np.float32), waveform[:1], 1)


@pytest.mark.unit
def test_rms_scaling_uses_float64_intermediate_power():
    signal = np.full(8, np.float32(1e20))

    scaled = scale_signal_to_dBfs_rms(signal, -20.0)

    assert np.isfinite(scaled).all()
    assert np.sqrt(np.mean(scaled.astype(np.float64) ** 2)) == pytest.approx(0.1, rel=1e-5)


@pytest.mark.unit
def test_transition_ramp_matches_overlapping_reference():
    delay = np.array([0.0, 0.0, 3.0, 3.0, 2.0, 2.0])
    result = _apply_transition_ramps(delay, [2, 4], transition_len=3)

    reference_rate = np.zeros(delay.size)
    rate = np.diff(delay, prepend=delay[0])
    for index in np.flatnonzero(rate):
        end = min(delay.size, index + 3)
        reference_rate[index:end] += rate[index] / (end - index)
    expected = delay[0] + np.cumsum(reference_rate)
    assert np.allclose(result, expected)


@pytest.mark.unit
@pytest.mark.parametrize("factory", [PFDKF, PFDKFTorch])
def test_pfdkf_rejects_invalid_overlap_save_shape(factory):
    with pytest.raises(ValueError, match=r"M == 2 \* hop"):
        factory(N=1, M=8, hop=3)


@pytest.mark.unit
def test_disabled_linear_frontend_validates_inputs_before_zero_output():
    frontend = LinearAECFrontend(None)

    with pytest.raises(ValueError, match="shape mismatch"):
        frontend.process(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        frontend.process(np.array([np.nan]), np.zeros(1))


pytestmark = [pytest.mark.unit]
