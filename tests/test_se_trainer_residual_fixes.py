"""SE trainer 残留审计修复的回归测试。"""

from __future__ import annotations

import contextlib
import signal
from types import SimpleNamespace

import pytest
import torch

from tasks.se.data.batch import SEBatch
from tasks.se.trainer.dual_input_loss_trainer import DualInputLossTrainer
from tasks.se.trainer.gan_trainer import GANTrainer
from tasks.se.trainer.meanflowse_trainer import MeanFlowSETrainer
from tasks.se.trainer import metricgan_trainer as metricgan_module
from tasks.se.trainer.metricgan_trainer import MetricGANCompositeTrainer
from tasks.se.trainer.metrics_trainer import SETrainerBase
from tasks.se.trainer.rt_semamba_distill_trainer import RTSEMambaDistillTrainer
from tasks.se.trainer.unipase_adapter_gan_trainer import UniPASEAdapterGANTrainer


pytestmark = pytest.mark.unit


def _raw_batch() -> SEBatch:
    clean = torch.tensor([[0.1, -0.1]])
    noisy = torch.tensor([[0.2, -0.2]])
    return SEBatch(clean, noisy, torch.tensor(16000), torch.tensor([2]))


def _configure_validation_stub(trainer, *, shutdown_signums: list[int]) -> list[tuple[float, int]]:
    reductions: list[tuple[float, int]] = []
    trainer.device = torch.device("cpu")
    trainer.val_loader = [_raw_batch()]
    trainer.use_amp = False
    trainer.amp_dtype = torch.float16
    trainer.max_val_batches = 0
    trainer.rank = 0
    trainer._model_accepts_fs = False
    trainer._ema_context = contextlib.nullcontext
    trainer._val_iter_is_ddp_padding = lambda _: False
    trainer._batch_fs_hz = lambda sr, _: int(torch.as_tensor(sr).item())
    trainer._sync_train_batch_exception = (
        lambda error, *, phase: (
            None if error is None else f"local {type(error).__name__}: {error}"
        )
    )

    def _sync_shutdown_signum() -> int:
        return shutdown_signums.pop(0)

    trainer._sync_shutdown_signum = _sync_shutdown_signum

    def _accumulate(loss, total, count, *_args, weight, **_kwargs):
        return total + loss.detach() * weight, count + weight, True

    trainer._accumulate_val_loss = _accumulate

    def _reduce(total, count):
        reductions.append((float(total.item()), count))
        return float(total.item()), float(count)

    trainer._allreduce_val_loss_sum_count = _reduce
    trainer._val_avg_and_curve_x = lambda *_: pytest.fail("aborted validation must not log")
    trainer._log_scalar = lambda *_: pytest.fail("aborted validation must not log")
    return reductions


def test_dual_input_training_rejects_non_scalar_loss():
    trainer = object.__new__(DualInputLossTrainer)
    trainer._model_accepts_fs = False
    trainer.model = lambda *_args, **_kwargs: torch.ones(2)
    batch = SimpleNamespace(
        noisy=torch.zeros(2, 3), clean=torch.zeros(2, 3), lengths=torch.tensor([3, 3])
    )

    with pytest.raises(RuntimeError, match="scalar loss tensor"):
        trainer._forward_and_compute_loss(batch, 1, 1, 16000)


def test_dual_input_validation_rejects_non_scalar_loss():
    trainer = object.__new__(DualInputLossTrainer)
    _configure_validation_stub(trainer, shutdown_signums=[0])
    trainer.model = lambda *_args, **_kwargs: torch.ones(1)

    with pytest.raises(RuntimeError, match="scalar loss tensor"):
        trainer._validation_epoch(1)


@pytest.mark.parametrize("trainer_type", [DualInputLossTrainer, RTSEMambaDistillTrainer])
def test_scalar_validation_shutdown_discards_partial_loss_after_synced_abort(trainer_type):
    trainer = object.__new__(trainer_type)
    reductions = _configure_validation_stub(
        trainer, shutdown_signums=[0, int(signal.SIGTERM)]
    )
    if trainer_type is DualInputLossTrainer:
        trainer.model = lambda *_args, **_kwargs: torch.tensor(3.0)
    else:
        trainer._distill_forward = lambda *_args: (torch.tensor(3.0), {})

    assert trainer._validation_epoch(1) is None
    # 前置边界完成一次 forward，terminal boundary 中止；归约仍固定执行且贡献被清零。
    assert reductions == [(0.0, 0)]


@pytest.mark.parametrize("trainer_type", [DualInputLossTrainer, RTSEMambaDistillTrainer])
def test_scalar_validation_syncs_rank_local_prepare_error_before_forward(trainer_type):
    trainer = object.__new__(trainer_type)
    _configure_validation_stub(trainer, shutdown_signums=[0])
    phases: list[str] = []
    trainer._prepare_scalar_validation_batch = lambda *_: (_ for _ in ()).throw(
        ValueError("bad local batch")
    )
    trainer.model = lambda *_args, **_kwargs: pytest.fail(
        "forward must not run after synchronized prepare failure"
    )

    def _sync(error, *, phase):
        phases.append(phase)
        assert isinstance(error, ValueError)
        return "rank 0 ValueError: bad local batch"

    trainer._sync_train_batch_exception = _sync
    with pytest.raises(RuntimeError, match="batch preparation failed.*rank 0"):
        trainer._validation_epoch(1)
    assert phases == ["validation_prepare_error"]


@pytest.mark.parametrize("trainer_type", [DualInputLossTrainer, RTSEMambaDistillTrainer])
def test_scalar_validation_syncs_remote_forward_error_at_fixed_phase(trainer_type):
    trainer = object.__new__(trainer_type)
    _configure_validation_stub(trainer, shutdown_signums=[0])
    phases: list[str] = []
    if trainer_type is DualInputLossTrainer:
        trainer.model = lambda *_args, **_kwargs: torch.tensor(1.0)
    else:
        trainer._distill_forward = lambda *_args: (torch.tensor(1.0), {})

    def _sync(error, *, phase):
        phases.append(phase)
        assert error is None
        if phase == "validation_forward_error":
            return "rank 1 RuntimeError: remote forward failed"
        return None

    trainer._sync_train_batch_exception = _sync
    with pytest.raises(RuntimeError, match="forward failed.*rank 1"):
        trainer._validation_epoch(1)
    assert phases == ["validation_prepare_error", "validation_forward_error"]


def test_metricgan_pesq_injection_rejects_sample_rate_mismatch():
    trainer = object.__new__(MetricGANCompositeTrainer)
    trainer.use_pesq_injection = True
    trainer.pesq_sr = 16000
    trainer._batch_fs_hz = lambda *_: 8000
    trainer._compute_batch_pesq = lambda *_: pytest.fail("must fail before PESQ")
    forward_dict = {
        "clean": torch.zeros(1, 4),
        "enhanced": torch.zeros(1, 4),
        "fs": torch.tensor(8000),
    }

    with pytest.raises(ValueError, match="requires batch fs=16000, got 8000"):
        trainer._get_discriminator_real_scores(forward_dict)


def test_metricgan_pesq_dependency_failure_precedes_parent_initialization(monkeypatch):
    def _missing_backend():
        raise RuntimeError("pesq missing")

    monkeypatch.setattr(metricgan_module, "_load_pesq_backend", _missing_backend)
    monkeypatch.setattr(
        GANTrainer,
        "__init__",
        lambda self, **kwargs: pytest.fail("parent initialization must not run"),
    )

    with pytest.raises(RuntimeError, match="pesq missing"):
        MetricGANCompositeTrainer(
            config={"trainer": {"use_pesq_injection": True}}
        )


def test_metricgan_pesq_scoring_failure_is_not_replaced_with_zero():
    trainer = object.__new__(MetricGANCompositeTrainer)
    trainer.pesq_sr = 16000

    def _failed_score(*_args):
        raise ValueError("too short")

    trainer._pesq_backend = _failed_score
    with pytest.raises(RuntimeError, match="sample 0.*16000 Hz"):
        trainer._compute_batch_pesq(
            torch.zeros(1, 320),
            torch.zeros(1, 320),
            torch.tensor([320]),
        )


def test_se_non_strict_sanitizes_clean_even_when_noisy_is_finite(caplog):
    trainer = object.__new__(DualInputLossTrainer)
    trainer._strict_checks = False
    trainer.rank = 0
    batch = SEBatch(
        clean=torch.tensor([[float("nan"), float("inf"), -float("inf"), 1.0]]),
        noisy=torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        sr=torch.tensor(16000),
        lengths=torch.tensor([4]),
    )

    with caplog.at_level("WARNING"):
        sanitized = trainer._sanitize_batch_inputs(batch, epoch=2, batch_idx=3)

    torch.testing.assert_close(
        sanitized.clean, torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    )
    torch.testing.assert_close(
        sanitized.noisy, torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    )
    assert "input fields clean" in caplog.text


@pytest.mark.parametrize(
    ("clean", "noisy", "lengths", "input_domain", "message"),
    [
        (torch.zeros(1, 2, 4), torch.zeros(1, 2, 4), torch.tensor([4]), "spectrum", "clean must be"),
        (torch.zeros(2, 4), torch.zeros(2, 3), torch.tensor([4, 4]), "spectrum", "shape mismatch"),
        (torch.zeros(2, 4), torch.zeros(2, 2, 4), torch.tensor([4, 4]), "waveform", "spectrum-domain"),
        (torch.zeros(2, 4), torch.zeros(2, 4), torch.tensor([4]), "spectrum", "lengths must be"),
        (torch.full((2, 4), float("nan")), torch.zeros(2, 4), torch.tensor([4, 4]), "spectrum", "NaN/Inf"),
    ],
)
def test_se_strict_batch_validation_raises_value_error(
    clean, noisy, lengths, input_domain, message
):
    """strict 校验必须 raise ValueError（python -O 不得使训练侧校验失效）。"""
    trainer = object.__new__(SETrainerBase)
    trainer.execution_plan = SimpleNamespace(input_domain=input_domain)

    with pytest.raises(ValueError, match=message):
        trainer._assert_batch_valid(clean, noisy, lengths)


def test_se_strict_train_forward_shape_mismatch_raises_value_error():
    trainer = object.__new__(SETrainerBase)
    trainer.device = torch.device("cpu")
    trainer.use_amp = False
    trainer.amp_dtype = torch.float16
    trainer._strict_checks = True
    trainer._use_lsnr_loss = False
    trainer._forward_model = lambda *_args, **_kwargs: (torch.zeros(2, 3), None, None)
    trainer._sanitize_model_output = lambda wav: wav
    batch = SEBatch(
        clean=torch.zeros(2, 4),
        noisy=torch.zeros(2, 4),
        sr=torch.tensor(16000),
        lengths=torch.tensor([4, 4]),
    )

    with pytest.raises(ValueError, match="Model output shape mismatch"):
        trainer._forward_and_compute_loss(batch, 0, 0, 16000)


def test_se_silence_placeholder_count_uses_row_energy_without_square_buffer():
    trainer = object.__new__(SETrainerBase)
    batch = SEBatch(
        clean=torch.cat([torch.zeros(2, 8), torch.ones(1, 8)]),
        noisy=torch.cat([torch.zeros(2, 8), torch.full((1, 8), 0.5)]),
        sr=torch.tensor(16000),
        lengths=torch.tensor([8, 8, 8]),
    )

    assert trainer._maybe_count_silence_placeholders(batch) == 2


def test_se_spatial_silence_placeholder_count_is_per_sample_not_per_channel():
    """spatial [B, C, T] 占位样本每个样本只计一次，任一通道非零即非占位。"""
    trainer = object.__new__(SETrainerBase)
    silent_channels = SEBatch(
        clean=torch.zeros(1, 8),
        noisy=torch.zeros(1, 2, 8),
        sr=torch.tensor(16000),
        lengths=torch.tensor([8]),
    )
    assert trainer._maybe_count_silence_placeholders(silent_channels) == 1

    non_silent = silent_channels.noisy.clone()
    non_silent[0, 1, 3] = 1.0
    mixed = SEBatch(
        clean=torch.zeros(1, 8),
        noisy=non_silent,
        sr=torch.tensor(16000),
        lengths=torch.tensor([8]),
    )
    assert trainer._maybe_count_silence_placeholders(mixed) == 0


def test_standard_se_validation_prepare_does_not_enter_model_forward():
    trainer = object.__new__(SETrainerBase)
    trainer.device = torch.device("cpu")
    trainer._strict_checks = False
    trainer._batch_fs_hz = lambda sr, _: int(torch.as_tensor(sr).item())
    trainer.model = lambda *_args, **_kwargs: pytest.fail(
        "prepare stage must not enter model forward"
    )

    prepared = trainer._val_prepare_batch(_raw_batch(), epoch=1, n_loops=1)

    se_batch, clean, noisy, lengths, fs_val = prepared
    assert isinstance(se_batch, SEBatch)
    torch.testing.assert_close(clean, _raw_batch().clean)
    torch.testing.assert_close(noisy, _raw_batch().noisy)
    torch.testing.assert_close(lengths, torch.tensor([2]))
    assert fs_val == 16000


@pytest.mark.parametrize("trainer_type", [SETrainerBase, MeanFlowSETrainer])
def test_se_validation_legacy_hook_composes_split_phases(trainer_type):
    trainer = object.__new__(trainer_type)
    calls: list[tuple[str, object]] = []

    def _prepare(raw_batch, epoch, n_loops):
        calls.append(("prepare", raw_batch))
        return "prepared"

    def _forward(prepared_batch, epoch, n_loops):
        calls.append(("forward", prepared_batch))
        return "result"

    trainer._val_prepare_batch = _prepare
    trainer._val_forward_prepared = _forward

    assert trainer._val_forward_loss("raw", 2, 3) == "result"
    assert calls == [("prepare", "raw"), ("forward", "prepared")]


@pytest.mark.parametrize(
    ("trainer_type", "output_domain"),
    [
        (DualInputLossTrainer, "scalar_loss"),
        (RTSEMambaDistillTrainer, "scalar_loss"),
        (UniPASEAdapterGANTrainer, "wavlm_feature"),
    ],
)
def test_special_se_trainers_reject_waveform_validation_metrics(
    trainer_type, output_domain
):
    trainer = object.__new__(trainer_type)
    with pytest.raises(ValueError, match=output_domain):
        SETrainerBase.__init__(
            trainer,
            config={"trainer": {"val_metrics": {"enabled": True}}},
        )


def test_meanflow_padding_batches_length_transfer_without_item(monkeypatch):
    trainer = object.__new__(MeanFlowSETrainer)
    trainer.recipe_num_samples = 6
    noisy = torch.tensor([[1.0, -2.0, 99.0], [3.0, 88.0, 88.0]])
    clean = torch.tensor([[0.5, -1.0, 77.0], [1.5, 66.0, 66.0]])

    def _forbid_item(self, *_args, **_kwargs):
        raise AssertionError("padding must batch-transfer lengths instead of per-row .item()")

    monkeypatch.setattr(torch.Tensor, "item", _forbid_item)
    padded_noisy, padded_clean, full_lengths = trainer._pad_training_pair(
        noisy, clean, torch.tensor([2, 1])
    )
    monkeypatch.undo()

    torch.testing.assert_close(
        padded_noisy,
        torch.tensor([[0.0, 0.0, 1.0, -2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        padded_clean,
        torch.tensor([[0.0, 0.0, 0.5, -1.0, 0.0, 0.0], [0.0, 0.0, 1.5, 0.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(full_lengths, torch.tensor([6, 6]))
