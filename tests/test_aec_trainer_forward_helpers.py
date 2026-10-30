"""AEC trainer 前向 helper 与结构化 batch 边界回归测试。"""

from __future__ import annotations

import pytest
import torch

from tasks.aec.data.batch import AECBatch
from tasks.aec.inference.aec_frontend import AECDeploymentFrontend, AECFrontendConfig
from tasks.aec.models.discriminative.deepvqe import DeepVQE
from tasks.aec.trainer.aec_trainer import AECTrainer, _unpack_aec_batch


class _RecordingStft:
    n_fft = 8
    hop_length = 4

    def __init__(self, inverse_output: torch.Tensor | None = None) -> None:
        self.forward_inputs: list[torch.Tensor] = []
        self.inverse_calls: list[tuple[torch.Tensor, int]] = []
        self.inverse_output = inverse_output

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        self.forward_inputs.append(waveform.detach().clone())
        return waveform.clone()

    def inverse(self, spectra: torch.Tensor, length: int) -> torch.Tensor:
        self.inverse_calls.append((spectra, length))
        if self.inverse_output is not None:
            return self.inverse_output
        return spectra[..., :length]


class _FrontendStub:
    def __init__(
        self,
        *,
        aligned: torch.Tensor,
        error: torch.Tensor | None = None,
        error_mask: torch.Tensor | None = None,
    ) -> None:
        self.aligned = aligned
        self.error = error
        self.error_mask = error_mask
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []

    def process_batch(self, mic, farend, lengths, sample_rate):
        self.calls.append((mic.clone(), farend.clone(), lengths.clone(), sample_rate))
        mask = self.error_mask
        if mask is None:
            mask = torch.zeros(mic.shape[0], dtype=torch.bool, device=mic.device)
        return self.aligned, self.error, mask, {"input_routing": "stub"}


def _batch(*, error: torch.Tensor | None = None, error_mask: torch.Tensor | None = None) -> AECBatch:
    return AECBatch(
        nearend=torch.full((2, 6), 0.25),
        mic=torch.full((2, 6), 0.5),
        farend=torch.full((2, 6), -0.5),
        fs=torch.tensor(16000, dtype=torch.int32),
        lengths=torch.tensor([6, 5], dtype=torch.int32),
        error=error,
        error_mask=error_mask,
        is_pure_noise=torch.tensor([True, False]),
        is_pure_echo=torch.tensor([False, True]),
        is_pure_echo_noise=torch.tensor([False, False]),
        has_echo=torch.tensor([True, True]),
        state=torch.zeros(2, 3, dtype=torch.long),
    )


def _trainer(stft: _RecordingStft) -> AECTrainer:
    trainer = object.__new__(AECTrainer)
    trainer.device = torch.device("cpu")
    trainer.stft_module = stft
    trainer._aec_frontend_config = AECFrontendConfig()
    trainer._aec_frontend = AECDeploymentFrontend(trainer._aec_frontend_config)
    trainer._aec_frontend_enabled = False
    return trainer


def test_input_builder_preserves_aec_dict_contract():
    stft = _RecordingStft()
    trainer = _trainer(stft)
    batch = _batch()
    nontarget = torch.ones_like(batch.mic)

    data = trainer._build_aec_model_input(batch, nontarget=nontarget)

    assert set(data) == {
        "mic",
        "farend",
        "nearend",
        "nontarget",
        "lengths",
        "mic_spectra",
        "farend_spectra",
        "nearend_spectra",
        "nontarget_spectra",
        "stft_n_fft",
        "stft_hop_length",
    }
    # 标签只进入 loss/metrics 消费的 dict，不作为模型输入。
    assert not {"is_pure_noise", "is_pure_echo", "is_pure_echo_noise", "state"} & set(data)
    assert data["mic"] is batch.mic
    assert data["farend"] is batch.farend
    assert data["nearend"] is batch.nearend
    assert data["nontarget"] is nontarget
    assert data["lengths"] is batch.lengths
    assert int(data["stft_n_fft"].item()) == stft.n_fft
    assert int(data["stft_hop_length"].item()) == stft.hop_length
    assert len(stft.forward_inputs) == 4
    assert "has_echo" not in data
    assert "error_spectra" not in data
    assert "error_mask" not in data


@pytest.mark.unit
def test_strict_false_sanitizes_inputs_without_finite_scan(monkeypatch):
    trainer = object.__new__(AECTrainer)
    trainer._strict_checks = False
    batch = _batch()
    batch.mic[0, 0] = float("nan")
    batch.farend[0, 1] = float("inf")
    batch.nearend[0, 2] = -float("inf")
    monkeypatch.setattr(torch, "isfinite", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("finite scan must be skipped")
    ))

    sanitized = trainer._sanitize_batch_inputs(batch, epoch=1, batch_idx=0)

    assert sanitized is batch
    assert not torch.isnan(batch.mic).any()
    assert not torch.isnan(batch.farend).any()
    assert not torch.isnan(batch.nearend).any()
    assert batch.mic[0, 0] == 0
    assert batch.farend[0, 1] == 0
    assert batch.nearend[0, 2] == 0
    assert trainer._maybe_count_silence_placeholders(batch) == 0


def test_deepvqe_declares_only_required_spectra():
    assert DeepVQE.aec_input_capabilities == frozenset(
        {"mic_spectra", "farend_spectra"}
    )


def test_declared_capabilities_skip_unused_spectra():
    stft = _RecordingStft()
    trainer = _trainer(stft)

    class _InputStub:
        aec_input_capabilities = frozenset({"mic_spectra", "farend_spectra"})

    trainer.model = _InputStub()
    data = trainer._build_aec_model_input(_batch(), nontarget=torch.ones(2, 6))

    assert "mic_spectra" in data and "farend_spectra" in data
    assert "nearend_spectra" not in data
    assert "nontarget_spectra" not in data
    assert len(stft.forward_inputs) == 2


def test_undeclared_model_keeps_legacy_full_input_and_offline_error():
    stft = _RecordingStft()
    trainer = _trainer(stft)
    trainer.model = type("LegacyAEC", (), {})()
    batch = _batch(
        error=torch.ones(2, 6),
        error_mask=torch.tensor([True, False]),
    )

    model_mic, aligned, error, error_mask = trainer._run_aec_frontend(batch)
    data = trainer._build_aec_model_input(
        batch,
        model_mic=model_mic,
        aligned_farend=aligned,
        error=error,
        error_mask=error_mask,
        nontarget=torch.ones(2, 6),
    )

    assert "nearend_spectra" in data
    assert "nontarget_spectra" in data
    assert "error_spectra" in data
    assert torch.equal(data["error_mask"], torch.tensor([True, False]))
    assert torch.equal(data["error_spectra"][0], batch.error[0])
    assert torch.equal(data["error_spectra"][1], batch.mic[1])
    assert len(stft.forward_inputs) == 5


@pytest.mark.parametrize("local_count", [None, 3])
def test_bn_sanitize_count_always_sums_at_epoch_boundary(monkeypatch, caplog, local_count):
    trainer = _trainer(_RecordingStft())
    trainer.world_size = 2
    trainer.rank = 0
    if local_count is not None:
        trainer._aec_state_n_nonfinite = torch.tensor(local_count, dtype=torch.int64)
    calls = []

    def fake_all_reduce(tensor, *, op):
        calls.append((tensor.clone(), op))
        tensor.add_(5)

    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.dist.all_reduce", fake_all_reduce)
    with caplog.at_level("WARNING", logger="tasks.aec.trainer.aec_trainer"):
        trainer._log_output_sanitize_summary(epoch=2, tag="[Train]")

    assert len(calls) == 2
    assert all(op is torch.distributed.ReduceOp.SUM for _, op in calls)
    # 第一次是共享输出 sanitize 的固定 [non-finite, clamped] 归约；第二次是
    # AEC BatchNorm 持久状态计数归约，二者都必须在 epoch boundary 参与。
    assert calls[1][0].ndim == 0
    expected = 5 if local_count is None else local_count + 5
    assert f"{expected} non-finite elements globally" in caplog.text


def test_bn_sanitize_count_nonzero_rank_never_reports(monkeypatch, caplog):
    trainer = _trainer(_RecordingStft())
    trainer.world_size = 2
    trainer.rank = 1
    trainer._aec_state_n_nonfinite = torch.tensor(3, dtype=torch.int64)
    monkeypatch.setattr(
        "tasks.aec.trainer.aec_trainer.dist.all_reduce", lambda tensor, *, op: tensor.add_(5)
    )

    with caplog.at_level("WARNING", logger="tasks.aec.trainer.aec_trainer"):
        trainer._log_output_sanitize_summary(epoch=2, tag="[Train]")

    assert "BatchNorm running state" not in caplog.text


class _CorruptingBatchNormAEC(torch.nn.Module):
    def __init__(self, *, raise_after_corruption: bool = False) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(6)
        self.raise_after_corruption = raise_after_corruption

    def forward(self, data):
        # 模拟异常激活：BatchNorm 会把 NaN 统计写入持久 running buffers；
        # 输出同时经过该 BN，使训练路径 backward 必须消费其保存的统计版本。
        self.bn(torch.full_like(data["mic"], float("nan")))
        if self.raise_after_corruption:
            raise RuntimeError("synthetic AEC forward failure")
        data["nearend_estimate_spectra"] = self.bn(data["mic"])
        return data


@pytest.mark.parametrize("raise_after_corruption", [False, True])
def test_forward_defers_batchnorm_repair_until_after_backward(raise_after_corruption):
    trainer = _trainer(_RecordingStft())
    trainer.model = _CorruptingBatchNormAEC(
        raise_after_corruption=raise_after_corruption
    )
    trainer._strict_checks = False
    batch = _batch()
    bn = trainer.model.bn

    if raise_after_corruption:
        # forward 抛错路径不会再有 backward，应立即修复。
        with pytest.raises(RuntimeError, match="synthetic AEC forward failure"):
            trainer._forward_aec_batch(batch)
        assert torch.isfinite(bn.running_mean).all()
        assert torch.isfinite(bn.running_var).all()
        assert not trainer._has_persistent_state_repair_pending()
        assert trainer._aec_state_n_nonfinite.item() == 12
        return

    data = trainer._forward_aec_batch(batch)
    assert data["nearend_estimate"].shape == batch.nearend.shape
    # 训练路径（grad enabled）在 backward 前不得改写 running buffers：
    # 仍保持污染状态并挂起修复标记。
    assert not torch.isfinite(bn.running_mean).all()
    assert trainer._materialize_aec_state_repair_pending()
    assert trainer._aec_state_n_nonfinite.item() == 12
    # 经 BN 输出的 backward 不得报 inplace version 错误。
    data["nearend_estimate"].square().mean().backward()
    trainer._flush_persistent_state_repair()
    assert torch.isfinite(bn.running_mean).all()
    assert torch.isfinite(bn.running_var).all()
    assert not trainer._has_persistent_state_repair_pending()


def test_forward_repairs_batchnorm_state_immediately_without_grad():
    trainer = _trainer(_RecordingStft())
    trainer.model = _CorruptingBatchNormAEC()

    with torch.no_grad():
        trainer._forward_aec_batch(_batch())

    bn = trainer.model.bn
    assert torch.isfinite(bn.running_mean).all()
    assert torch.isfinite(bn.running_var).all()
    assert not trainer._has_persistent_state_repair_pending()
    assert trainer._aec_state_n_nonfinite.item() == 12


def test_loss_preparation_exception_repairs_abandoned_batchnorm_graph(monkeypatch):
    """forward 后、backward 前异常会放弃 graph，此时才允许原地修复 BN。"""
    trainer = _trainer(_RecordingStft())
    trainer.model = _CorruptingBatchNormAEC()
    trainer.loss_fn = lambda *args, **kwargs: torch.zeros((), requires_grad=True)
    trainer.log_interval = 100
    trainer._strict_checks = False
    monkeypatch.setattr(
        "tasks.aec.trainer.aec_trainer.sanitize_model_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic sanitize failure")
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic sanitize failure"):
        trainer._forward_and_compute_loss(_batch(), epoch=1, batch_idx=0, fs_val=16000)

    bn = trainer.model.bn
    assert torch.isfinite(bn.running_mean).all()
    assert torch.isfinite(bn.running_var).all()
    assert not trainer._has_persistent_state_repair_pending()
    assert getattr(trainer, "_aec_last_state_bad_count", None) is None


class _FiniteBatchNormAEC(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(6)

    def forward(self, data):
        data["nearend_estimate_spectra"] = self.bn(data["mic"])
        return data


@pytest.mark.unit
def test_forward_does_not_modify_finite_batchnorm_buffers_before_backward():
    trainer = _trainer(_RecordingStft())
    trainer.model = _FiniteBatchNormAEC()
    batch = _batch()

    data = trainer._forward_aec_batch(batch)
    data["nearend_estimate"].square().mean().backward()


@pytest.mark.unit
def test_disabled_frontend_is_strict_noop_without_backend_call():
    trainer = _trainer(_RecordingStft())
    trainer._aec_frontend.process_batch = lambda *_args, **_kwargs: pytest.fail(
        "disabled frontend must not run"
    )
    batch = _batch()

    model_mic, aligned, error, error_mask = trainer._run_aec_frontend(batch)

    assert model_mic is batch.mic
    assert aligned is batch.farend
    assert error is None
    assert error_mask is None


@pytest.mark.unit
def test_offline_error_compatibility_sanitizes_and_keeps_explicit_mask():
    error = torch.tensor(
        [[float("nan"), float("inf"), -float("inf"), 1.0, 0.0, 0.0], [0.0] * 6]
    )
    trainer = _trainer(_RecordingStft())
    batch = _batch(error=error, error_mask=torch.tensor([True, False]))

    _, _, resolved_error, error_mask = trainer._run_aec_frontend(batch)

    assert resolved_error is not None and torch.isfinite(resolved_error).all()
    assert torch.equal(resolved_error[0, :3], torch.zeros(3))
    assert torch.equal(error_mask, torch.tensor([True, False]))
    assert error_mask.dtype == torch.bool


@pytest.mark.unit
@pytest.mark.parametrize(
    "error_mask",
    [
        torch.tensor([1, 0], dtype=torch.int32),
        torch.tensor([[True], [False]]),
    ],
)
def test_offline_error_mask_requires_exact_per_sample_bool_contract(error_mask):
    trainer = _trainer(_RecordingStft())

    with pytest.raises((TypeError, ValueError), match="error_mask"):
        trainer._run_aec_frontend(
            _batch(error=torch.ones(2, 6), error_mask=error_mask)
        )


@pytest.mark.unit
def test_mixed_error_mask_uses_mic_fallback_before_stft():
    stft = _RecordingStft()
    trainer = _trainer(stft)
    error = torch.stack((torch.full((6,), 3.0), torch.zeros(6)))
    batch = _batch(error=error, error_mask=torch.tensor([True, False]))

    model_mic, aligned, resolved_error, error_mask = trainer._run_aec_frontend(batch)
    data = trainer._build_aec_model_input(
        batch,
        model_mic=model_mic,
        aligned_farend=aligned,
        error=resolved_error,
        error_mask=error_mask,
    )

    assert torch.equal(data["error_spectra"][0], error[0])
    assert torch.equal(data["error_spectra"][1], batch.mic[1])
    assert torch.equal(data["error_mask"], torch.tensor([True, False]))


@pytest.mark.unit
def test_legacy_validation_hook_delegates_to_two_phase_contract(monkeypatch):
    trainer = object.__new__(AECTrainer)
    raw_batch = object()
    prepared = (_batch(), 16000)
    expected = (object(), object(), object(), object(), 16000, object())
    calls = []

    def prepare(value, epoch, n_loops):
        calls.append(("prepare", value, epoch, n_loops))
        return prepared

    def forward(value, epoch, n_loops):
        calls.append(("forward", value, epoch, n_loops))
        return expected

    monkeypatch.setattr(trainer, "_val_prepare_batch", prepare)
    monkeypatch.setattr(trainer, "_val_forward_prepared", forward)

    assert trainer._val_forward_loss(raw_batch, epoch=2, n_loops=3) is expected
    assert calls == [
        ("prepare", raw_batch, 2, 3),
        ("forward", prepared, 2, 3),
    ]


@pytest.mark.unit
def test_offline_error_compatibility_uses_legacy_amplitude_mask():
    trainer = _trainer(_RecordingStft())
    error = torch.tensor([[0.0] * 6, [0.0, 0.0, 2e-6, 0.0, 0.0, 0.0]])

    _, _, resolved_error, error_mask = trainer._run_aec_frontend(
        _batch(error=error)
    )

    assert resolved_error is error
    assert torch.equal(error_mask, torch.tensor([False, True]))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("routing", "expected_mic_key", "expect_error_input"),
    [
        ("mic_farend", "original", False),
        ("mic_farend_error", "original", True),
        ("error_farend", "error", False),
    ],
)
def test_frontend_routing_is_explicit(routing, expected_mic_key, expect_error_input):
    trainer = _trainer(_RecordingStft())
    config_dict = {"input_routing": routing}
    if routing == "mic_farend":
        config_dict["delay_alignment"] = {"enabled": True}
    else:
        config_dict["linear_aec"] = {"backend": "pfdkf_torch"}
    trainer._aec_frontend_config = AECFrontendConfig.from_dict(config_dict)
    aligned = torch.full((2, 6), 2.0)
    runtime_error = torch.full((2, 6), 3.0) if routing != "mic_farend" else None
    runtime_mask = torch.tensor([True, True]) if runtime_error is not None else None
    frontend = _FrontendStub(
        aligned=aligned,
        error=runtime_error,
        error_mask=runtime_mask,
    )
    trainer._aec_frontend = frontend
    trainer._aec_frontend_enabled = True
    batch = _batch()

    model_mic, model_farend, error, error_mask = trainer._run_aec_frontend(batch)

    assert torch.equal(model_farend, aligned)
    expected_mic = batch.mic if expected_mic_key == "original" else runtime_error
    assert torch.equal(model_mic, expected_mic)
    assert (error is not None) is expect_error_input
    assert (error_mask is not None) is expect_error_input
    assert frontend.calls[0][3] == 16000


@pytest.mark.unit
def test_runtime_linear_aec_rejects_pre_generated_error():
    trainer = _trainer(_RecordingStft())
    trainer._aec_frontend_config = AECFrontendConfig.from_dict(
        {
            "linear_aec": {"backend": "pfdkf_torch"},
            "input_routing": "mic_farend_error",
        }
    )
    trainer._aec_frontend_enabled = True

    with pytest.raises(RuntimeError, match="conflicts with pre-generated"):
        trainer._run_aec_frontend(
            _batch(error=torch.ones(2, 6), error_mask=torch.tensor([True, True]))
        )


@pytest.mark.unit
def test_deepvqe_rejects_error_input_but_allows_error_farend_routing():
    trainer = _trainer(_RecordingStft())
    trainer.model = type(
        "DeepVQEStub",
        (),
        {"aec_input_capabilities": DeepVQE.aec_input_capabilities},
    )()
    trainer._aec_frontend_config = AECFrontendConfig.from_dict(
        {
            "linear_aec": {"backend": "pfdkf_torch"},
            "input_routing": "mic_farend_error",
        }
    )
    with pytest.raises(ValueError, match="requires model capability 'error_spectra'"):
        trainer._validate_aec_frontend_capabilities()

    trainer._aec_frontend_config = AECFrontendConfig.from_dict(
        {
            "linear_aec": {"backend": "pfdkf_torch"},
            "input_routing": "error_farend",
        }
    )
    trainer._validate_aec_frontend_capabilities()


@pytest.mark.parametrize("output_len", [4, 8])
def test_output_decoder_waveform_fallback_pad_or_trims_and_attaches_spectrum(output_len):
    trainer = _trainer(_RecordingStft())
    batch = _batch()
    output = torch.arange(2 * output_len, dtype=torch.float32).reshape(2, output_len)

    data = trainer._decode_aec_model_output({"nearend_estimate": output}, batch)

    assert data["nearend_estimate"].shape == batch.nearend.shape
    assert data["nearend_estimate_spectra"].shape == batch.nearend.shape
    expected_prefix = output[..., : min(output_len, batch.nearend.shape[-1])]
    assert torch.equal(data["nearend_estimate"][..., : expected_prefix.shape[-1]], expected_prefix)
    if output_len < batch.nearend.shape[-1]:
        assert torch.equal(
            data["nearend_estimate"][..., output_len:],
            torch.zeros(2, batch.nearend.shape[-1] - output_len),
        )


def test_output_decoder_spectrum_uses_mic_reference_length():
    spectra = torch.full((2, 6), 0.75)
    stft = _RecordingStft(inverse_output=spectra.clone())
    trainer = _trainer(stft)
    batch = _batch()

    data = trainer._decode_aec_model_output({"nearend_estimate_spectra": spectra}, batch)

    assert stft.inverse_calls == [(spectra, batch.mic.shape[-1])]
    assert torch.equal(data["nearend_estimate"], spectra)


def test_output_decoder_rejects_missing_model_output():
    trainer = _trainer(_RecordingStft())

    with pytest.raises(RuntimeError, match="must output either"):
        trainer._decode_aec_model_output({"mic": _batch().mic}, _batch())


def test_output_decoder_keeps_shape_guard_after_istft():
    short = torch.zeros(2, 5)
    trainer = _trainer(_RecordingStft(inverse_output=short))

    with pytest.raises(RuntimeError, match="AEC shape mismatch"):
        trainer._decode_aec_model_output(
            {"nearend_estimate_spectra": torch.zeros(2, 6)}, _batch()
        )


def test_forward_orchestrates_helpers_and_preserves_model_dict_keys():
    trainer = _trainer(_RecordingStft())

    class _Model:
        def __call__(self, data):
            data["model_aux"] = torch.tensor(1.0)
            data["nearend_estimate_spectra"] = data["nearend_spectra"]
            return data

    trainer.model = _Model()
    data = trainer._forward_aec_batch(_batch())

    assert "model_aux" in data
    assert "mic_spectra" in data
    assert "farend_spectra" in data
    assert "nearend_spectra" in data
    assert "nearend_estimate_spectra" in data
    assert "nearend_estimate" in data


def test_structured_batch_boundary_is_idempotent_and_rejects_8_9_tuples():
    batch = _batch()
    assert _unpack_aec_batch(batch) is batch
    for n_fields in (8, 9):
        with pytest.raises(TypeError, match="requires an AECBatch"):
            _unpack_aec_batch(tuple(range(n_fields)))


pytestmark = [pytest.mark.unit]
