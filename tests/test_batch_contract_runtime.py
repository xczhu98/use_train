"""训练侧 BatchContract 的运行时边界。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from core.data.batch_contract import (
    batch_size_from_contract,
    is_batch_contract,
    move_batch_to_device,
)
from core.data.source_separation_batch import SSBatch
from core.trainer.metrics_trainer import MetricsCapableTrainer
from tasks.aec.data.batch import AECBatch
from tasks.se.data.batch import SEBatch
from tasks.ssl.data.batch import SSLBatch
from tasks.tse.cue_contract import TSECueBatches, TSECueSpecs
from tasks.tse.data.batch import TSEBatch


pytestmark = pytest.mark.unit


def _se_batch() -> SEBatch:
    return SEBatch(
        clean=torch.zeros(2, 8),
        noisy=torch.ones(2, 8),
        sr=torch.full((2,), 16000),
        lengths=torch.full((2,), 8),
    )


def _aec_batch() -> AECBatch:
    waveform = torch.zeros(2, 8)
    return AECBatch(
        nearend=waveform,
        mic=waveform.clone(),
        farend=waveform.clone(),
        fs=torch.full((2,), 16000),
        lengths=torch.full((2,), 8),
    )


def _tse_batch() -> TSEBatch:
    return TSEBatch(
        mixture=torch.zeros(2, 1, 8),
        target=torch.zeros(2, 8),
        sr=torch.tensor(16000, dtype=torch.int32),
        lengths=torch.full((2,), 8, dtype=torch.int32),
        num_targets=torch.ones(2, dtype=torch.int32),
        spk_labels=torch.full((2,), -1, dtype=torch.int64),
        target_present=torch.ones(2, dtype=torch.bool),
        cues=TSECueBatches(),
        cue_specs=TSECueSpecs(),
        keys=("a", "b"),
        speaker_ids=(None, None),
        geometry=(None, None),
        timebase=(None, None),
        provenance=(None, None),
    )


def _ss_batch() -> SSBatch:
    return SSBatch(
        mixture=torch.zeros(2, 8),
        sources=torch.zeros(2, 2, 8),
        lengths=torch.full((2,), 8),
        source_mask=torch.ones(2, 2, dtype=torch.bool),
        sr=torch.full((2,), 16000),
    )


def _ssl_batch() -> SSLBatch:
    return SSLBatch(
        features=torch.zeros(2, 2, 3, 4),
        target_dpipd=torch.zeros(2, 2, 3, 1, 1, 2),
        input_lengths=torch.full((2,), 2, dtype=torch.int64),
        output_lengths=torch.full((2,), 2, dtype=torch.int64),
        pair_index=torch.tensor([[[0, 1]], [[0, 1]]], dtype=torch.int64),
        pair_mask=torch.ones(2, 1, dtype=torch.bool),
        geometry=torch.zeros(2, 2, 3),
        frame_origin=torch.zeros(2, dtype=torch.int64),
        time_stride=1,
    )


@pytest.mark.parametrize(
    "factory, tensor_count",
    [
        (_se_batch, 4),
        (_aec_batch, 5),
        (_tse_batch, 7),
        (_ss_batch, 5),
        (_ssl_batch, 8),
    ],
    ids=["se", "aec", "tse", "ss", "ssl"],
)
def test_all_task_batches_use_the_shared_transfer_and_size_contract(
    factory, tensor_count, monkeypatch
):
    batch = factory()
    moved_tensors = []

    def record_to(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        assert torch.device(device) == torch.device("cpu")
        assert kwargs.get("non_blocking") is True
        moved_tensors.append(tensor)
        return tensor

    monkeypatch.setattr(torch.Tensor, "to", record_to)

    assert is_batch_contract(batch)
    assert batch_size_from_contract(batch) == 2

    moved = move_batch_to_device(batch, torch.device("cpu"), non_blocking=True)

    assert moved is not batch
    assert is_batch_contract(moved)
    assert batch_size_from_contract(moved) == 2
    assert len(moved_tensors) == tensor_count


class _InPlaceBatch:
    batch_size = 2

    def validate(self):
        pass

    def pin_memory(self):
        return self

    def to(self, device, *, non_blocking=False):
        self.device = torch.device(device)
        self.non_blocking = non_blocking
        return self


def test_in_place_contract_transfer_is_not_confused_with_a_miss():
    batch = _InPlaceBatch()

    moved = move_batch_to_device(batch, "cpu", non_blocking=True)

    assert moved is batch
    assert batch.device == torch.device("cpu")
    assert batch.non_blocking is True


class _WrongSizeBatch(_InPlaceBatch):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def to(self, device, *, non_blocking=False):
        return _WrongSizeBatch(1)


def test_contract_transfer_must_preserve_the_sample_axis():
    with pytest.raises(ValueError, match="changed batch_size"):
        move_batch_to_device(_WrongSizeBatch(2), "cpu")


class _WrongReturnBatch(_InPlaceBatch):
    batch_size = 2

    def to(self, device, *, non_blocking=False):
        return object()


def test_contract_transfer_must_return_another_contract():
    with pytest.raises(TypeError, match="must return a BatchContract"):
        move_batch_to_device(_WrongReturnBatch(), "cpu")


@dataclass
class _UncontractedBatch:
    clean: torch.Tensor


def test_trainer_rejects_uncontracted_dataclass_instead_of_reflective_transfer():
    trainer = SimpleNamespace(device=torch.device("cpu"))
    batch = _UncontractedBatch(clean=torch.zeros(2, 8))

    with pytest.raises(TypeError, match="must implement BatchContract"):
        MetricsCapableTrainer._batch_to_device(trainer, batch)
    with pytest.raises(ValueError, match="must implement BatchContract"):
        MetricsCapableTrainer._batch_size_from_batch(trainer, batch)


def test_trainer_keeps_the_explicit_gan_dict_path():
    trainer = SimpleNamespace(device=torch.device("cpu"))
    batch = {
        "clean": torch.zeros(2, 8),
        "noisy": torch.ones(2, 8),
        "fs": torch.full((2,), 16000),
        "lengths": torch.full((2,), 8),
    }

    moved = MetricsCapableTrainer._batch_to_device(trainer, batch)

    assert moved is batch
    assert MetricsCapableTrainer._batch_size_from_batch(trainer, batch) == 2
    assert torch.equal(MetricsCapableTrainer._get_batch_fs(trainer, batch), batch["fs"])

    with pytest.raises(ValueError, match="field 'clean'"):
        MetricsCapableTrainer._batch_size_from_batch(
            trainer, {"noisy": torch.ones(2, 8)}
        )


@pytest.mark.parametrize("container", [tuple, list])
@pytest.mark.parametrize("length", [4, 5, 7, 10, 11, 12, 13])
def test_se_and_aec_trainers_reject_positional_batches(container, length):
    from tasks.se.trainer.metrics_trainer import _unpack_se_batch
    from tasks.aec.trainer.aec_trainer import _unpack_aec_batch

    with pytest.raises(TypeError, match="requires an SEBatch"):
        _unpack_se_batch(container([None] * length))
    with pytest.raises(TypeError, match="requires an AECBatch"):
        _unpack_aec_batch(container([None] * length))


@pytest.mark.parametrize("factory", [_se_batch, _aec_batch])
@pytest.mark.parametrize("field,value", [
    ("lengths", torch.tensor([0, 8])),
    ("lengths", torch.tensor([8, 9])),
    ("lengths", torch.tensor([8.0, 8.0])),
    ("lengths", torch.tensor([[8, 8]])),
    ("is_pure_noise", torch.tensor([0, 1])),
    ("is_pure_echo", torch.tensor([True])),
])
def test_waveform_batches_validate_lengths_and_flags(factory, field, value):
    batch = factory()
    batch.validate()
    with pytest.raises((TypeError, ValueError), match=field):
        replace(batch, **{field: value}).validate()


@pytest.mark.parametrize("factory,field", [(_se_batch, "sr"), (_aec_batch, "fs")])
@pytest.mark.parametrize("value", [
    16000, torch.tensor(16000.0), torch.tensor([8000, 16000]),
    torch.tensor([-1, -1]), torch.tensor([[16000, 16000]]),
])
def test_waveform_batches_validate_sample_rate(factory, field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        replace(factory(), **{field: value}).validate()


@pytest.mark.parametrize("factory,field,value", [
    (_se_batch, "clean", torch.zeros(8)),
    (_se_batch, "noisy", torch.zeros(1, 8)),
    (_se_batch, "noisy", torch.zeros(2, 0, 8)),
    (_aec_batch, "mic", torch.zeros(2, 9)),
    (_aec_batch, "farend", torch.zeros(2, 1, 8)),
    (_aec_batch, "error", torch.zeros(2, 7)),
    (_aec_batch, "error_mask", torch.ones(2, dtype=torch.bool)),
    (_aec_batch, "state", torch.zeros(2, 3)),
    (_aec_batch, "state", torch.zeros(1, 3, dtype=torch.int64)),
    (_aec_batch, "state", torch.full((2, 3), -2, dtype=torch.int64)),
])
def test_waveform_batches_reject_malformed_optional_and_spatial_fields(factory, field, value):
    with pytest.raises((TypeError, ValueError)):
        replace(factory(), **{field: value}).validate()
