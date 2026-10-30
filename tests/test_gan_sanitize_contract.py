"""GAN trainer sanitize forward hook 契约测试。"""

from __future__ import annotations

import contextlib
import signal
from types import SimpleNamespace

import pytest
import torch

from tasks.se.trainer.gan_trainer import GANTrainer
from tasks.se.trainer.metricgan_trainer import MetricGANCompositeTrainer
from tasks.se.trainer.metrics_trainer import SETrainerBase


pytestmark = [pytest.mark.unit]


class _ToggleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class _HookContractTrainer(GANTrainer):
    """通过 __new__ 构造，仅保留 epoch loop 所需的轻量行为。"""

    def _forward_generator(self, batch_dict):
        self.events.append("forward")
        return {
            **batch_dict,
            "enhanced": torch.tensor([[2.0]]),
            "clean": torch.zeros(1, 1),
            "lengths": torch.ones(1, dtype=torch.int64),
        }

    def _sanitize_forward_output(self, forward_dict):
        self.events.append("sanitize")
        forward_dict["enhanced"] = forward_dict["enhanced"].clamp(-1.0, 1.0)
        return forward_dict

    def _extract_gan_mag_pha(self, fwd):
        self.events.append("extract")
        assert fwd["enhanced"].item() == 1.0
        value = torch.zeros(1, 1, 1)
        return value, value, value, value

    def _get_discriminator_real_scores(self, forward_dict):
        self.events.append("scores")
        assert forward_dict["enhanced"].item() == 1.0
        return None

    def _prepare_discriminator_losses(self, *args, **kwargs):
        return []

    def _prepare_generator_losses(self, *args, **kwargs):
        return []

    def _train_step_discriminator(self, *args, **kwargs):
        self.events.append("discriminator")
        assert self.events.index("sanitize") < self.events.index("discriminator")
        if getattr(self, "force_d_nonfinite", False):
            self._gan_window_d_finite = False
        values = getattr(self, "discriminator_loss_values", None)
        return torch.tensor([values.pop(0) if values else 0.0])

    def _train_step_generator(self, enhanced, *args, **kwargs):
        self.events.append("loss")
        assert enhanced.item() == 1.0
        assert self.events.index("sanitize") < self.events.index("loss")
        values = getattr(self, "generator_loss_values", None)
        return torch.tensor([values.pop(0) if values else 0.0])

    def _compute_validation_loss(self, forward_dict):
        self.events.append("validation_loss")
        assert forward_dict["enhanced"].item() == 1.0
        assert self.events.index("sanitize") < self.events.index("validation_loss")
        return torch.zeros(())


def _make_loop_stub(
    *, batches: list[tuple[int, object]] | None = None, accumulation_steps: int = 1
) -> _HookContractTrainer:
    trainer = _HookContractTrainer.__new__(_HookContractTrainer)
    trainer.events = []
    trainer.pair_calls = []
    trainer.device = torch.device("cpu")
    trainer.generator = _ToggleModel()
    trainer.discriminator = _ToggleModel()
    trainer.models = {"generator": trainer.generator, "discriminator": trainer.discriminator}
    for model in trainer.models.values():
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
    trainer._is_multi_model = True
    trainer.accumulation_steps = accumulation_steps
    trainer.world_size = 1
    trainer.rank = 1
    trainer.use_amp = False
    trainer.amp_dtype = torch.float16
    trainer.start_epoch = 1
    trainer._resume_skip_batches = 0
    trainer._train_log_loss_win = []
    trainer.n_steps = 0
    trainer.max_steps = None
    trainer.max_val_batches = 0
    trainer._vm_max_metrics = 0
    trainer._reset_output_sanitize_stats = lambda: None
    trainer._make_train_progress_bar = lambda epoch: batches or [(0, None)]
    trainer._prepare_batch = lambda raw, epoch, bi: ({}, 16000, 0)
    trainer._optimizer_step = lambda *args, **kwargs: True
    trainer._allreduce_bool_and = lambda value, phase="": value
    trainer._gradients_are_finite = lambda model_key=None: True
    def _paired_step(**kwargs):
        trainer.pair_calls.append(kwargs)
        results = getattr(trainer, "pair_results", None)
        if results:
            return results.pop(0)
        return kwargs.get("step_allowed", True)

    trainer._gan_optimizer_step = _paired_step
    trainer._gan_skip_optimizer_steps = lambda: False
    trainer._log_train_loss_after_step = lambda: None
    trainer._log_output_sanitize_summary = lambda *args: None
    trainer._advance_and_should_run_metrics = lambda: False
    trainer._ema_context = contextlib.nullcontext
    trainer._val_iter_is_ddp_padding = lambda n: False
    trainer._sync_train_batch_exception = lambda error, *, phase: None
    trainer._unpack_batch = lambda batch: {
        "clean": torch.zeros(1, 1),
        "lengths": torch.ones(1, dtype=torch.int64),
    }
    trainer._accumulate_val_loss = lambda loss, running, count, *args, **kwargs: (
        running + loss,
        count + 1,
        True,
    )
    trainer._val_avg_and_curve_x = lambda loss_sum, count, epoch: (loss_sum / count, epoch)
    trainer._run_validation_metrics_from_pairs = lambda *args, **kwargs: None
    trainer.val_loader = [None]
    return trainer


def test_train_epoch_dispatches_polymorphic_sanitize_before_discriminator_and_loss():
    trainer = _make_loop_stub()

    trainer._train_epoch(1)

    assert trainer.events == [
        "forward",
        "sanitize",
        "extract",
        "scores",
        "discriminator",
        "loss",
    ]


def test_validation_epoch_dispatches_same_polymorphic_sanitize_hook():
    trainer = _make_loop_stub()

    trainer._validation_epoch(1)

    assert trainer.events == ["forward", "sanitize", "validation_loss"]


def test_validation_adapter_preserves_gan_collective_and_finalize_order():
    trainer = _make_loop_stub()
    trainer.world_size = 2
    trainer.sample_rate = 16000
    trainer._vm_max_metrics = 1
    trainer._advance_and_should_run_metrics = lambda: True
    trainer._append_val_metric_pairs = (
        lambda *args: trainer.events.append("collect_pairs") or 1
    )
    trainer._allreduce_sum = lambda value, phase="": (
        trainer.events.append(phase) or value
    )
    trainer._run_validation_metrics_from_pairs = (
        lambda *args, **kwargs: trainer.events.append("metrics")
    )
    trainer._log_output_sanitize_summary = (
        lambda *args: trainer.events.append("sanitize_summary")
    )

    trainer._validation_epoch(1)

    assert trainer.events == [
        "forward",
        "sanitize",
        "validation_loss",
        "collect_pairs",
        "val_loss_allreduce",
        "val_count_allreduce",
        "metrics",
        "sanitize_summary",
    ]


class _SpectrumStub:
    def __init__(self):
        self.calls = 0

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.stack((wav, wav * 2.0), dim=-1)


class _BaseGANStub(GANTrainer):
    def _sanitize_model_output(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)


def test_gan_base_refreshes_enhanced_spec_before_discriminator_extract():
    trainer = _BaseGANStub.__new__(_BaseGANStub)
    trainer.stft_module = _SpectrumStub()
    fwd = {
        "enhanced": torch.tensor([[float("nan"), 2.0, -0.5]]),
        "enhanced_spec": torch.full((1, 3, 2), 99.0),
        "clean_spec": torch.zeros(1, 3, 2),
    }

    result = trainer._sanitize_forward_output(fwd)
    _, _, enhanced_mag, enhanced_pha = trainer._extract_gan_mag_pha(result)

    expected_spec = torch.stack((result["enhanced"], result["enhanced"] * 2.0), dim=-1)
    expected_mag = torch.sqrt(
        expected_spec[..., 0] ** 2 + expected_spec[..., 1] ** 2 + 1e-8
    )
    expected_pha = torch.atan2(expected_spec[..., 1], expected_spec[..., 0])
    assert trainer.stft_module.calls == 1
    assert torch.equal(result["enhanced_spec"], expected_spec)
    assert torch.allclose(enhanced_mag, expected_mag)
    assert torch.allclose(enhanced_pha, expected_pha)


class _MetricGANStub(MetricGANCompositeTrainer):
    def _sanitize_model_output(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)


def test_metricgan_recomputes_all_enhanced_spectral_derivatives_after_sanitize():
    trainer = _MetricGANStub.__new__(_MetricGANStub)
    trainer.stft_module = _SpectrumStub()
    trainer.composite_loss = SimpleNamespace(compress_factor=0.5)
    fwd = {
        "enhanced": torch.tensor([[float("nan"), 2.0, -0.5]]),
        "enhanced_spec": torch.full((1, 3, 2), 99.0),
        "enhanced_mag": torch.full((1, 3), 99.0),
        "enhanced_pha": torch.full((1, 3), 99.0),
    }

    result = trainer._sanitize_forward_output(fwd)

    expected_spec = torch.stack((result["enhanced"], result["enhanced"] * 2.0), dim=-1)
    # 与 trainer 官方对齐语义（mag_phase_stft(addeps=False)）一致：mag 直接用
    # torch.abs(complex)，pha 用 torch.angle(complex)，不再附加 sqrt(+1e-9) 与
    # atan2 的 eps 项——零谱处的压缩 mag 因此为 0.0 而非旧公式的 ~0.0056。
    expected_mag = torch.sqrt(
        expected_spec[..., 0] ** 2 + expected_spec[..., 1] ** 2
    ).pow(trainer.composite_loss.compress_factor)
    expected_pha = torch.atan2(expected_spec[..., 1], expected_spec[..., 0])
    assert result is fwd
    assert trainer.stft_module.calls == 1
    assert torch.equal(result["enhanced"], torch.tensor([[0.0, 1.0, -0.5]]))
    assert torch.equal(result["enhanced_spec"], expected_spec)
    assert torch.allclose(result["enhanced_mag"], expected_mag)
    assert torch.allclose(result["enhanced_pha"], expected_pha)


def _make_pair_step_stub(
    *, bad_model: str | None = None, amp_fp16: bool = False
):
    trainer = GANTrainer.__new__(GANTrainer)
    trainer.generator = _ToggleModel()
    trainer.discriminator = _ToggleModel()
    trainer.models = {"generator": trainer.generator, "discriminator": trainer.discriminator}
    trainer._is_multi_model = True
    trainer.world_size = 1
    trainer.rank = 0
    trainer.calls = []
    trainer._allreduce_bool_ands = lambda values, phase="": values

    def _prepare(key, *, skip_step=False):
        eligible = not skip_step and key != bad_model
        prepared = SimpleNamespace(
            key=key,
            eligible=eligible,
            amp_fp16=amp_fp16,
            reason=None if eligible else "unscaled gradients are non-finite",
            # AMP 溢出由显式标志判定，stub 必须与 PreparedOptimizerStep 字段一致。
            amp_nonfinite=amp_fp16 and not eligible,
        )
        trainer.calls.append(("prepare", key, {"skip_step": skip_step}))
        return prepared

    def _commit(prepared, **kwargs):
        trainer.calls.append(("commit", prepared.key, kwargs))
        return True

    def _discard(prepared, **kwargs):
        trainer.calls.append(("discard", prepared.key, kwargs))

    trainer._prepare_optimizer_step = _prepare
    trainer._commit_prepared_optimizer_step = _commit
    trainer._discard_prepared_optimizer_step = _discard
    return trainer


def test_gan_optimizer_step_prepares_both_then_commits_d_before_g():
    trainer = _make_pair_step_stub()

    assert trainer._gan_optimizer_step() is True
    assert trainer.calls == [
        ("prepare", "discriminator", {"skip_step": False}),
        ("prepare", "generator", {"skip_step": False}),
        (
            "commit",
            "discriminator",
            {"update_scaler": False, "increment_step": False},
        ),
        (
            "commit",
            "generator",
            {"update_scaler": True, "increment_step": True},
        ),
    ]


def test_gan_optimizer_step_rejects_amp_unscale_overflow_before_d_step():
    trainer = _make_pair_step_stub(bad_model="discriminator", amp_fp16=True)

    assert trainer._gan_optimizer_step() is False
    assert trainer.calls == [
        ("prepare", "discriminator", {"skip_step": False}),
        ("prepare", "generator", {"skip_step": False}),
        (
            "discard",
            "discriminator",
            {"update_scaler": False, "amp_overflow": True},
        ),
        (
            "discard",
            "generator",
            {"update_scaler": True, "amp_overflow": True},
        ),
    ]


@pytest.mark.parametrize("bad_model", ["discriminator", "generator"])
def test_gan_optimizer_step_skips_both_sides_when_one_is_nonfinite(bad_model):
    trainer = _make_pair_step_stub(bad_model=bad_model)

    assert trainer._gan_optimizer_step() is False
    assert trainer.calls == [
        ("prepare", "discriminator", {"skip_step": False}),
        ("prepare", "generator", {"skip_step": False}),
        (
            "discard",
            "discriminator",
            {"update_scaler": False, "amp_overflow": False},
        ),
        (
            "discard",
            "generator",
            {"update_scaler": True, "amp_overflow": False},
        ),
    ]


def test_gan_accumulation_boundary_uses_one_paired_step():
    trainer = _make_loop_stub(
        batches=[(0, None), (1, None)], accumulation_steps=2
    )
    callbacks = []

    trainer._train_epoch(1, step_callback=lambda *args: callbacks.append(args))

    assert trainer.pair_calls == [{"step_allowed": True}]
    assert len(callbacks) == 1


def test_gan_shutdown_drops_partial_accumulation_and_preserves_cursor():
    """停机发生在 accumulation 中间时不得提交 D/G step 或推进 cursor。"""
    trainer = _make_loop_stub(
        batches=[(0, None), (1, None), (2, None)], accumulation_steps=2
    )
    trainer.optimizers = {
        key: torch.optim.SGD(model.parameters(), lr=0.1)
        for key, model in trainer.models.items()
    }
    sync_calls = 0

    def _sync_shutdown_signum():
        nonlocal sync_calls
        sync_calls += 1
        return 0 if sync_calls == 1 else int(signal.SIGTERM)

    trainer._sync_shutdown_signum = _sync_shutdown_signum

    trainer._train_epoch(1)

    # 第一个完整 window 在首次边界同步后提交；第二次同步发生在 epoch-tail，
    # 此时仅丢弃 batch 2 所在的未提交 window，cursor 保持上一安全边界。
    assert trainer.pair_calls == [{"step_allowed": True}]
    assert trainer._last_completed_batch_cursor == 2
    assert trainer._train_epoch_completed is False
    assert all(
        parameter.grad is None
        for model in trainer.models.values()
        for parameter in model.parameters()
    )


def test_gan_epoch_tail_uses_paired_step_and_skips_nonfinite_window():
    trainer = _make_loop_stub(batches=[(0, None)], accumulation_steps=2)
    callbacks = []
    trainer._train_epoch(1, step_callback=lambda *args: callbacks.append(args))
    assert trainer.pair_calls == [{"step_allowed": True}]
    assert len(callbacks) == 1

    trainer = _make_loop_stub(batches=[(0, None)], accumulation_steps=2)
    trainer.force_d_nonfinite = True
    callbacks = []
    trainer._train_epoch(1, step_callback=lambda *args: callbacks.append(args))
    assert trainer.pair_calls == [{"step_allowed": False}]
    assert callbacks == []
    assert trainer._train_log_loss_win == []


def test_gan_epoch_loss_excludes_skipped_optimizer_window():
    trainer = _make_loop_stub(
        batches=[(0, None), (1, None)], accumulation_steps=1
    )
    trainer.generator_loss_values = [100.0, 3.0]
    trainer.discriminator_loss_values = [50.0, 2.0]
    trainer.pair_results = [False, True]

    avg_loss = trainer._train_epoch(1)

    assert avg_loss == pytest.approx(3.0)
    assert trainer._train_log_loss_win == [pytest.approx(3.0)]
    assert trainer.pair_calls == [
        {"step_allowed": True},
        {"step_allowed": True},
    ]


def test_gan_all_skipped_windows_return_zero_without_logging_loss():
    trainer = _make_loop_stub(batches=[(0, None)], accumulation_steps=1)
    trainer.generator_loss_values = [100.0]
    trainer.discriminator_loss_values = [50.0]
    trainer.pair_results = [False]

    assert trainer._train_epoch(1) == pytest.approx(0.0)
    assert trainer._train_log_loss_win == []


def test_gan_forward_error_repairs_persistent_state_before_synced_raise():
    """forward 失败放弃 graph 后必须立即修复被污染的持久状态。"""
    trainer = _make_loop_stub()
    repairs: list[str] = []
    trainer._repair_persistent_state_after_abandoned_graph = lambda: repairs.append(
        "repair"
    )
    trainer._forward_generator = lambda batch: (_ for _ in ()).throw(
        ValueError("invalid local forward")
    )

    with pytest.raises(ValueError, match="invalid local forward"):
        trainer._train_epoch(1)

    assert repairs == ["repair"]


def test_gan_pending_persistent_state_drops_paired_step_and_flushes_after_backward():
    """持久状态 pending 时本 window 不提交 D/G step，且 flush 必须晚于 backward。"""
    trainer = _make_loop_stub()
    flush_snapshots: list[list[str]] = []
    trainer._has_persistent_state_repair_pending = lambda: True
    trainer._flush_persistent_state_repair = lambda: flush_snapshots.append(
        list(trainer.events)
    )

    trainer._train_epoch(1)

    assert trainer.pair_calls == [{"step_allowed": False}]
    assert flush_snapshots == [
        ["forward", "sanitize", "extract", "scores", "discriminator", "loss"]
    ]


def test_gan_rejects_nonpositive_discriminator_steps(monkeypatch):
    def _base_init(self, **_kwargs):
        self._is_multi_model = True
        self._is_multi_optimizer = True
        self.models = {"generator": _ToggleModel(), "discriminator": _ToggleModel()}
        self.optimizers = {"generator": object(), "discriminator": object()}
        self.config = {"trainer": {"discriminator_steps": 0}}
        self.world_size = 1
        self.log_interval = 1

    monkeypatch.setattr(SETrainerBase, "__init__", _base_init)
    with pytest.raises(ValueError, match="discriminator_steps=0.*>= 1"):
        GANTrainer()
