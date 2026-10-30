"""SSL 输出错误必须在 loss 的有效帧归约之前跨 rank 同步。"""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from core.models.io_contract import ModelIOContract, TensorField
from tasks.ssl.data.batch import SSLBatch
from tasks.ssl.losses import SSLDPIPDPITLoss
from tasks.ssl.runtime import SSLRuntimeInputs
from tasks.ssl.trainer.ssl_trainer import SSLTrainer


class _PredictionModel(torch.nn.Module):
    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"ssl"}),
        inputs=(
            TensorField(
                "features",
                "embedding",
                ("batch", "frame", "frequency", "feature"),
            ),
        ),
        outputs=(
            TensorField(
                "dpipd",
                "spectrum",
                ("batch", "frame", "frequency", "track", "pair", "complex"),
                multiple=True,
            ),
        ),
        forward_kind="estimate",
        frontend_owner="model",
    )

    def __init__(self, failure=None):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(2.0))
        self.failure = failure

    def forward(self, features):
        if self.failure == "exception":
            raise ValueError("injected SSL forward failure")
        prediction = features.unsqueeze(-2).unsqueeze(-2) * self.gain
        if self.failure == "shape":
            return prediction[:, :-1]
        if self.failure == "nonfinite":
            return prediction * float("nan")
        return prediction


def _batch():
    return SSLBatch(
        features=torch.ones(1, 2, 3, 2),
        target_dpipd=torch.zeros(1, 2, 3, 1, 1, 2),
        input_lengths=torch.tensor([2]),
        output_lengths=torch.tensor([2]),
        pair_index=torch.tensor([[[0, 1]]], dtype=torch.int64),
        pair_mask=torch.ones(1, 1, dtype=torch.bool),
        geometry=torch.tensor([[[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]]),
        frame_origin=torch.tensor([0], dtype=torch.int64),
        time_stride=1,
        valid_output_mask=torch.ones(1, 2, dtype=torch.bool),
        doa=torch.zeros(1, 2, 1, 2),
        vad=torch.ones(1, 2, 1),
        sample_rates=torch.tensor([8000]),
        provenance=({"backend": "fixture"},),
        input_representation="reference_ipd",
        target_representation="dpipd_far_field",
        coordinate_frame="array_local",
    )


def _trainer(model, *, rank=0, world_size=1):
    trainer = object.__new__(SSLTrainer)
    trainer.model = model
    trainer.loss_fn = SSLDPIPDPITLoss(max_tracks=1)
    trainer.device = torch.device("cpu")
    trainer.sample_rate = 8000
    trainer.rank = rank
    trainer.world_size = world_size
    trainer._nccl_flow_log = lambda _phase: None
    return trainer


def _forward(trainer, phase):
    batch = _batch()
    if phase == "train":
        return trainer._forward_and_compute_loss(batch, 0, 0, 8000)[0]
    prepared = trainer._val_prepare_batch(batch, 0, 1)
    return trainer._val_forward_prepared(prepared, 0, 1)[0]


@pytest.mark.unit
def test_ssl_preserves_finite_unbounded_prediction_and_gradients():
    trainer = _trainer(_PredictionModel())
    prediction = trainer._forward_prediction(SSLRuntimeInputs.from_batch(_batch()))
    torch.testing.assert_close(prediction["dpipd"], torch.full_like(prediction["dpipd"], 2.0))
    train_loss = _forward(trainer, "train")
    train_loss.backward()
    assert trainer.model.gain.grad is not None
    assert torch.isfinite(trainer.model.gain.grad)
    with torch.no_grad():
        torch.testing.assert_close(_forward(trainer, "validation"), train_loss)


@pytest.mark.unit
@pytest.mark.parametrize("phase", ["train", "validation"])
@pytest.mark.parametrize("failure", ["exception", "shape", "nonfinite"])
def test_ssl_rejects_bad_prediction_before_entering_loss(phase, failure):
    trainer = _trainer(_PredictionModel(failure))

    def unexpected_loss(*_args, **_kwargs):
        pytest.fail("invalid model output reached the collective-bearing loss")

    trainer.loss_fn = unexpected_loss
    with pytest.raises(ValueError, match="SSL"):
        _forward(trainer, phase)


def _ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        for phase in ("train", "validation"):
            for failure in (None, "exception", "shape", "nonfinite"):
                model = DistributedDataParallel(
                    _PredictionModel(failure if rank == 1 else None)
                )
                trainer = _trainer(model, rank=rank, world_size=2)
                if failure is None:
                    if phase == "validation":
                        with torch.inference_mode():
                            loss = _forward(trainer, phase)
                    else:
                        loss = _forward(trainer, phase)
                        loss.backward()
                    torch.testing.assert_close(loss, torch.tensor(8.0))
                else:
                    with pytest.raises(RuntimeError, match="rank 1.*SSL"):
                        if phase == "validation":
                            with torch.inference_mode():
                                _forward(trainer, phase)
                        else:
                            _forward(trainer, phase)
    finally:
        dist.destroy_process_group()


@pytest.mark.ddp
@pytest.mark.subprocess
def test_ssl_two_rank_output_errors_fail_before_loss_collective(tmp_path):
    mp.spawn(
        _ddp_worker,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
