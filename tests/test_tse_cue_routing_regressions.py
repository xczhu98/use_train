from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset

from core.data.simulation.array_profiles import resolve_microphone_array
from core.losses.time_domain import L1WavLoss
from core.models.io_contract import ModelIOContract, TensorField
from core.stft import StftIstftModule
from tasks.tse.cue_contract import (
    TSECueBatch,
    TSECueBatches,
    TSECueSpec,
    TSECueSpecs,
)
from tasks.tse.data.batch import TSEBatch
from tasks.tse.models.discriminative.tse_bsrnn_spk import TSE_BSRNN_SPK
from tasks.tse.trainer.tse_trainer import (
    TSETrainer,
    validate_tse_training_dataset_cues,
)


SR = 16000
N_FFT = 512
HOP = 128
SAMPLES = 1024
ENROLL_SAMPLES = 640
EMBED_DIM = 192


class _BatchDataset(Dataset):
    def __init__(
        self,
        batch: TSEBatch,
        *,
        cue_guarantees: dict[str, bool] | None = None,
    ) -> None:
        self.batch = batch
        self.cue_specs = batch.cue_specs
        self.cue_guarantees = cue_guarantees or {}
        self.strict_load = False

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> TSEBatch:
        return self.batch


class _FakeSpeakerEncoder(nn.Module):
    inputs: list[torch.Tensor] = []
    fail_rank: int | None = None

    def __init__(self, *, pretrained: str, spk_args=None, **kwargs) -> None:
        super().__init__()
        self.embed_dim = int((spk_args or {}).get("embed_dim", EMBED_DIM))
        self.probe = nn.Parameter(torch.tensor(1.0))
        self.fbank = type("_Fbank", (), {"window_size": 4})()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        type(self).inputs.append(waveform.detach().clone())
        mean = waveform.mean(dim=1, keepdim=True)
        output = mean.repeat(1, self.embed_dim)
        if (
            self.fail_rank is not None
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_rank() == self.fail_rank
        ):
            return output[:, :-1]
        return output


class _NativeReferenceModel(nn.Module):
    supports_multichannel_input = True
    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"tse"}),
        inputs=(
            TensorField(
                "mix_spectra",
                "spectrum",
                ("batch", "microphone", "frequency", "frame", "complex"),
            ),
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

    def __init__(self, ref_channel: int | None = None) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.5))
        if ref_channel is not None:
            self.ref_channel = ref_channel

    def forward(self, data: dict) -> dict:
        data["target_estimate_spectra"] = data["mix_spectra"][:, 1] * self.gain
        return data



def _real_model() -> TSE_BSRNN_SPK:
    return TSE_BSRNN_SPK(
        sr=SR,
        feature_dim=8,
        num_repeat=1,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window="hann",
        center=True,
        joint_training=False,
        features={"spkemb": {"enabled": True}},
    )


def _batch(
    source: str,
    *,
    placeholders: tuple[int, ...] = (),
    rows: int = 2,
) -> TSEBatch:
    generator = torch.Generator().manual_seed(17)
    mixture = torch.randn(rows, 1, SAMPLES, generator=generator) * 0.05
    target = torch.randn(rows, SAMPLES, generator=generator) * 0.05
    spk_labels = torch.arange(rows, dtype=torch.int64)
    provenance: list[dict[str, bool]] = []
    for row in range(rows):
        is_placeholder = row in placeholders
        provenance.append({"placeholder": is_placeholder})
        if is_placeholder:
            mixture[row].zero_()
            target[row].zero_()
            spk_labels[row] = -1

    if source == "waveform":
        value = torch.randn(rows, ENROLL_SAMPLES, generator=generator) * 0.05
        present = torch.ones(rows, dtype=torch.bool)
        lengths = torch.full((rows,), ENROLL_SAMPLES, dtype=torch.int32)
        for row in placeholders:
            value[row].zero_()
            present[row] = False
            lengths[row] = 0
        cues = TSECueBatches(
            speaker_waveform=TSECueBatch(
                representation="waveform",
                value=value,
                present=present,
                lengths=lengths,
                timebase=None,
            )
        )
        cue_specs = TSECueSpecs(
            speaker_waveform=TSECueSpec("waveform", (ENROLL_SAMPLES,))
        )
    elif source == "embedding":
        value = torch.randn(rows, EMBED_DIM, generator=generator) * 0.05
        present = torch.ones(rows, dtype=torch.bool)
        for row in placeholders:
            value[row].zero_()
            present[row] = False
        cues = TSECueBatches(
            speaker_embedding=TSECueBatch(
                representation="embedding",
                value=value,
                present=present,
                lengths=None,
                timebase=None,
            )
        )
        cue_specs = TSECueSpecs(
            speaker_embedding=TSECueSpec("embedding", (EMBED_DIM,))
        )
    else:
        raise ValueError(source)

    result = TSEBatch(
        mixture=mixture,
        target=target,
        sr=torch.tensor(SR, dtype=torch.int32),
        lengths=torch.full((rows,), SAMPLES, dtype=torch.int32),
        num_targets=torch.ones(rows, dtype=torch.int32),
        spk_labels=spk_labels,
        target_present=torch.ones(rows, dtype=torch.bool),
        cues=cues,
        cue_specs=cue_specs,
        keys=tuple(f"sample-{row}" for row in range(rows)),
        speaker_ids=tuple(f"speaker-{row}" for row in range(rows)),
        geometry=(None,) * rows,
        timebase=(None,) * rows,
        provenance=tuple(provenance),
    )
    result.validate()
    return result


def _trainer(
    tmp_path: Path,
    batch: TSEBatch,
    *,
    val_batch: TSEBatch | None = None,
    checkpoint: Path | None = None,
    world_size: int = 1,
    rank: int = 0,
) -> TSETrainer:
    model = _real_model()
    model_args = {
        "joint_training": False,
        "features": {"spkemb": {"enabled": True, "embed_dim": EMBED_DIM}},
    }
    if checkpoint is not None:
        model_args["speaker_model"] = {
            "fbank": {"window_size": 4},
            "speaker_encoder": {
                "model": "fake",
                "pretrained": str(checkpoint),
                "spk_args": {"embed_dim": EMBED_DIM},
            },
        }
    config = {
        "model": {"type": "TSE_BSRNN_SPK", "args": model_args},
        "trainer": {
            "epochs": 1,
            "exp_path": str(tmp_path),
            "strict_checks": False,
            "validation_epoch_interval": 0,
        },
        "meta": {"seed": 7, "use_amp": False},
        "data": {"sample_rate": SR},
    }
    train_dataset = _BatchDataset(batch)
    validation_dataset = _BatchDataset(val_batch if val_batch is not None else batch)
    train_loader = DataLoader(train_dataset, batch_size=None)
    val_loader = DataLoader(validation_dataset, batch_size=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return TSETrainer(
        rank=rank,
        world_size=world_size,
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
        loss_fn=L1WavLoss(),
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        sample_rate=SR,
        stft_module=StftIstftModule(
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=N_FFT,
        ),
    )


@pytest.mark.unit
def test_real_tse_bsrnn_external_waveform_route_removes_consumed_cue(
    tmp_path, monkeypatch
):
    from tasks.tse.models.speaker import encoder as encoder_module

    _FakeSpeakerEncoder.inputs.clear()
    monkeypatch.setattr(encoder_module, "SpeakerEncoder", _FakeSpeakerEncoder)
    checkpoint = tmp_path / "speaker.pt"
    checkpoint.write_bytes(b"speaker-checkpoint")
    batch = _batch("waveform")
    trainer = _trainer(tmp_path, batch, checkpoint=checkpoint)

    loss, finite, _ = trainer._forward_and_compute_loss(
        batch.to(trainer.device), 0, 0, SR
    )

    assert finite and torch.isfinite(loss)
    assert trainer._uses_external_spk_encoder is True
    assert len(_FakeSpeakerEncoder.inputs) == 1
    assert trainer._raw_model().required_tse_input_fields() == frozenset(
        {"mix_spectra", "spk_embedding"}
    )


@pytest.mark.unit
def test_real_tse_bsrnn_direct_embedding_needs_no_external_checkpoint(tmp_path):
    batch = _batch("embedding")
    trainer = _trainer(tmp_path, batch)

    loss, finite, _ = trainer._forward_and_compute_loss(
        batch.to(trainer.device), 0, 0, SR
    )

    assert finite and torch.isfinite(loss)
    assert trainer._uses_external_spk_encoder is False
    assert trainer._spk_encoder_cfg is None
    assert trainer._spk_encoder is None


@pytest.mark.unit
@pytest.mark.parametrize("placeholders", [(1,), (0, 1)])
def test_direct_embedding_route_runs_load_failure_placeholders(
    tmp_path, placeholders
):
    batch = _batch("embedding", placeholders=placeholders)
    original_present = batch.cues.speaker_embedding.present.clone()
    trainer = _trainer(tmp_path, batch)

    loss, finite, _ = trainer._forward_and_compute_loss(
        batch.to(trainer.device), 0, 0, SR
    )

    assert finite and torch.isfinite(loss)
    assert torch.equal(batch.cues.speaker_embedding.present, original_present)


@pytest.mark.unit
def test_waveform_embedding_source_keeps_external_checkpoint_fail_closed(tmp_path):
    with pytest.raises((RuntimeError, ValueError), match="speaker_encoder.pretrained"):
        _trainer(tmp_path, _batch("waveform"))


@pytest.mark.unit
@pytest.mark.parametrize("placeholders", [(1,), (0, 1)])
def test_external_route_runs_mixed_and_pure_placeholders_without_mutating_cues(
    tmp_path, monkeypatch, placeholders
):
    from tasks.tse.models.speaker import encoder as encoder_module

    _FakeSpeakerEncoder.inputs.clear()
    monkeypatch.setattr(encoder_module, "SpeakerEncoder", _FakeSpeakerEncoder)
    checkpoint = tmp_path / "speaker.pt"
    checkpoint.write_bytes(b"speaker-checkpoint")
    batch = _batch("waveform", placeholders=placeholders)
    original_present = batch.cues.speaker_waveform.present.clone()
    original_lengths = batch.cues.speaker_waveform.lengths.clone()
    trainer = _trainer(tmp_path, batch, checkpoint=checkpoint)

    loss, finite, _ = trainer._forward_and_compute_loss(
        batch.to(trainer.device), 0, 0, SR
    )

    assert finite and torch.isfinite(loss)
    assert torch.equal(batch.cues.speaker_waveform.present, original_present)
    assert torch.equal(batch.cues.speaker_waveform.lengths, original_lengths)
    encoder_input = _FakeSpeakerEncoder.inputs[-1]
    for row in placeholders:
        assert torch.count_nonzero(encoder_input[row]) == 0
        assert encoder_input.shape[-1] == ENROLL_SAMPLES


@pytest.mark.unit
def test_real_missing_waveform_cue_is_not_treated_as_placeholder(tmp_path, monkeypatch):
    from tasks.tse.models.speaker import encoder as encoder_module

    monkeypatch.setattr(encoder_module, "SpeakerEncoder", _FakeSpeakerEncoder)
    checkpoint = tmp_path / "speaker.pt"
    checkpoint.write_bytes(b"speaker-checkpoint")
    batch = _batch("waveform")
    cue = batch.cues.speaker_waveform
    cue.value[0].zero_()
    cue.present[0] = False
    cue.lengths[0] = 0
    trainer = _trainer(tmp_path, batch, checkpoint=checkpoint)

    with pytest.raises(ValueError, match="provenance-declared placeholder"):
        trainer._forward_and_compute_loss(batch.to(trainer.device), 0, 0, SR)


@pytest.mark.unit
def test_constructor_rejects_ambiguous_or_non_guaranteed_required_cues():
    model = _real_model()
    both_specs = TSECueSpecs(
        speaker_waveform=TSECueSpec("waveform", (ENROLL_SAMPLES,)),
        speaker_embedding=TSECueSpec("embedding", (EMBED_DIM,)),
    )
    dataset = type(
        "_Dataset",
        (),
        {"cue_specs": both_specs, "cue_guarantees": {}},
    )()
    with pytest.raises(ValueError, match="both speaker waveform and embedding"):
        validate_tse_training_dataset_cues(model, dataset, {}, split="train")

    waveform_dataset = type(
        "_Dataset",
        (),
        {
            "cue_specs": TSECueSpecs(
                speaker_waveform=TSECueSpec("waveform", (ENROLL_SAMPLES,))
            ),
            "cue_guarantees": {"speaker_waveform": False},
        },
    )()
    with pytest.raises(ValueError, match="non-guaranteed real waveform"):
        validate_tse_training_dataset_cues(
            model,
            waveform_dataset,
            {"speaker_cue_source": "waveform"},
            split="train",
        )


@pytest.mark.unit
def test_geometry_reference_channel_reaches_builder_and_conflicts_fail():
    geometry = resolve_microphone_array(
        array_profile="linear_2ch_80mm",
        reference_channel=1,
    )
    mixture = torch.stack(
        (torch.full((SAMPLES,), 0.1), torch.full((SAMPLES,), 0.7)), dim=0
    ).unsqueeze(0)
    batch = TSEBatch(
        mixture=mixture,
        target=torch.zeros(1, SAMPLES),
        sr=torch.tensor(SR, dtype=torch.int32),
        lengths=torch.tensor([SAMPLES], dtype=torch.int32),
        num_targets=torch.ones(1, dtype=torch.int32),
        spk_labels=torch.full((1,), -1, dtype=torch.int64),
        target_present=torch.ones(1, dtype=torch.bool),
        cues=TSECueBatches(),
        cue_specs=TSECueSpecs(),
        keys=("sample",),
        speaker_ids=(None,),
        geometry=(geometry,),
        timebase=(None,),
        provenance=({"placeholder": False},),
    )
    batch.validate()
    trainer = object.__new__(TSETrainer)
    trainer.model = _NativeReferenceModel()
    trainer.sample_rate = SR
    trainer.stft_module = StftIstftModule(
        n_fft=N_FFT, hop_length=HOP, win_length=N_FFT
    )

    data = trainer._forward_tse_batch(batch)
    torch.testing.assert_close(data["mixture"], mixture[:, 1])

    trainer.model = _NativeReferenceModel(ref_channel=0)
    with pytest.raises(ValueError, match="conflicts with model reference channel"):
        trainer._forward_tse_batch(batch)


def _gloo_placeholder_worker(rank: int, root: str) -> None:
    torch.set_num_threads(1)
    torch.cuda.is_available = lambda: False
    root_path = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=(root_path / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        from tasks.tse.models.speaker import encoder as encoder_module

        encoder_module.SpeakerEncoder = _FakeSpeakerEncoder
        checkpoint = root_path / "speaker.pt"
        batch = _batch("waveform", placeholders=(0,) if rank == 1 else ())
        trainer = _trainer(
            root_path / f"rank-{rank}",
            batch,
            checkpoint=checkpoint,
            world_size=2,
            rank=rank,
        )
        device_batch = batch.to(trainer.device)
        loss, finite, _ = trainer._forward_and_compute_loss(
            device_batch, 0, 0, SR
        )
        loss.backward()
        cue = batch.cues.speaker_waveform
        result = {
            "rank": rank,
            "finite": finite,
            "loss": float(loss.detach()),
            "raw_present": bool(cue.present[0]),
            "raw_length": int(cue.lengths[0]),
        }
        (root_path / f"rank-{rank}.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.ddp
def test_gloo_rank_local_placeholder_keeps_collective_sequence(tmp_path):
    checkpoint = tmp_path / "speaker.pt"
    checkpoint.write_bytes(b"speaker-checkpoint")
    context = mp.spawn(
        _gloo_placeholder_worker,
        args=(str(tmp_path),),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 45.0
    while not context.join(max(0.0, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            raise TimeoutError("TSE placeholder Gloo worker did not finish")

    results = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert all(result["finite"] for result in results)
    assert results[1]["raw_present"] is False
    assert results[1]["raw_length"] == 0


def _run_trainer_split(
    trainer: TSETrainer,
    split: str,
    train_batch: TSEBatch,
    validation_batch: TSEBatch,
):
    if split == "train":
        return trainer._forward_and_compute_loss(
            train_batch.to(trainer.device), 0, 0, SR
        )[0]
    if split == "validation":
        return trainer._val_forward_loss(
            validation_batch, 0, 1
        )[0]
    raise ValueError(split)


def _gloo_split_source_worker(
    rank: int,
    root: str,
    train_source: str,
    validation_source: str,
    first_split: str,
    fail_rank: int | None,
) -> None:
    torch.set_num_threads(1)
    torch.cuda.is_available = lambda: False
    root_path = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=(root_path / "split-rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        from tasks.tse.models.speaker import encoder as encoder_module

        _FakeSpeakerEncoder.inputs.clear()
        _FakeSpeakerEncoder.fail_rank = fail_rank
        encoder_module.SpeakerEncoder = _FakeSpeakerEncoder
        checkpoint = root_path / "speaker.pt"
        train_batch = _batch(train_source)
        validation_batch = _batch(validation_source)
        trainer = _trainer(
            root_path / f"split-rank-{rank}",
            train_batch,
            val_batch=validation_batch,
            checkpoint=checkpoint,
            world_size=2,
            rank=rank,
        )
        second_split = "validation" if first_split == "train" else "train"
        first_loss = _run_trainer_split(
            trainer,
            first_split,
            train_batch,
            validation_batch,
        )
        if first_split == "train":
            first_loss.backward()
            trainer.optimizer.zero_grad(set_to_none=True)
        direct_state = {
            "encoder_is_none": trainer._spk_encoder is None,
            "identity_is_none": trainer._loaded_spk_checkpoint_identity is None,
            "identity_synced": trainer._spk_checkpoint_identity_synced,
            "first_finite": bool(torch.isfinite(first_loss).item()),
        }

        result: dict[str, object] = {"rank": rank, "direct": direct_state}
        try:
            second_loss = _run_trainer_split(
                trainer,
                second_split,
                train_batch,
                validation_batch,
            )
        except Exception as exc:
            result.update(
                status="error",
                error_type=type(exc).__name__,
                message=str(exc),
                identity_synced=trainer._spk_checkpoint_identity_synced,
            )
        else:
            result.update(
                status="success",
                second_finite=bool(torch.isfinite(second_loss).item()),
                encoder_loaded=trainer._spk_encoder is not None,
                identity_synced=trainer._spk_checkpoint_identity_synced,
            )
        (root_path / f"split-rank-{rank}.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        dist.barrier()
    finally:
        _FakeSpeakerEncoder.fail_rank = None
        dist.destroy_process_group()


def _spawn_split_source_case(
    tmp_path: Path,
    *,
    train_source: str,
    validation_source: str,
    first_split: str,
    fail_rank: int | None = None,
) -> list[dict]:
    (tmp_path / "speaker.pt").write_bytes(b"speaker-checkpoint")
    context = mp.spawn(
        _gloo_split_source_worker,
        args=(
            str(tmp_path),
            train_source,
            validation_source,
            first_split,
            fail_rank,
        ),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 45.0
    while not context.join(max(0.0, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            raise TimeoutError("TSE split-source Gloo worker did not finish")
    return [
        json.loads(
            (tmp_path / f"split-rank-{rank}.json").read_text(encoding="utf-8")
        )
        for rank in range(2)
    ]


@pytest.mark.unit
@pytest.mark.ddp
@pytest.mark.parametrize(
    ("train_source", "validation_source", "first_split"),
    [
        ("embedding", "waveform", "train"),
        ("waveform", "embedding", "validation"),
    ],
)
def test_gloo_split_source_plan_delays_external_identity_until_waveform_stage(
    tmp_path,
    train_source,
    validation_source,
    first_split,
):
    results = _spawn_split_source_case(
        tmp_path,
        train_source=train_source,
        validation_source=validation_source,
        first_split=first_split,
    )

    assert all(result["status"] == "success" for result in results)
    assert all(result["direct"]["first_finite"] for result in results)
    assert all(result["direct"]["encoder_is_none"] for result in results)
    assert all(result["direct"]["identity_is_none"] for result in results)
    assert all(not result["direct"]["identity_synced"] for result in results)
    assert all(result["second_finite"] for result in results)
    assert all(result["encoder_loaded"] for result in results)
    assert all(result["identity_synced"] for result in results)


@pytest.mark.unit
@pytest.mark.ddp
def test_gloo_waveform_stage_external_error_is_unified_after_direct_embedding(
    tmp_path,
):
    results = _spawn_split_source_case(
        tmp_path,
        train_source="waveform",
        validation_source="embedding",
        first_split="validation",
        fail_rank=1,
    )

    assert all(result["status"] == "error" for result in results)
    assert all(result["direct"]["encoder_is_none"] for result in results)
    messages = [result["message"] for result in results]
    assert messages[0] == messages[1]
    assert "rank 1" in messages[0]
    assert "dimension mismatch" in messages[0].lower()
    assert all(not result["identity_synced"] for result in results)
