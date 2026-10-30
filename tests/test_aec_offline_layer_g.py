"""AEC Trainer/Entry input/deployment frontend 接线回归测试。"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from tasks.aec.data.batch import AECBatch
from tasks.aec.inference.aec_frontend import AECFrontendConfig
from tasks.aec.trainer.aec_trainer import AECTrainer
from tasks.aec.trainer.entry import AECTrainingEntry
from core.trainer.entry import detect_task


def _full_offline_cfg(*, error_scp: str | None = None) -> dict:
    section = {
        "mic_scp": "mic.scp",
        "farend_scp": "farend.scp",
        "nearend_scp": "nearend.scp",
    }
    if error_scp is not None:
        section["error_scp"] = error_scp
    return {
        "meta": {},
        "stft": {"n_fft": 8, "hop_length": 4, "win_length": 8},
        "model": {"type": "DeepVQE"},
        "optimizer": {"type": "Adam"},
        "scheduler": {"type": "StepLR"},
        "trainer": {"type": "AECTrainer"},
        "loss": {"type": "L1WavLoss"},
        "data": {
            "mode": "offline_aec",
            "sample_rate": 16000,
            "dataloader": {"batch_size": 1, "num_workers": 0},
            "train": deepcopy(section),
            "validation": deepcopy(section),
        },
    }


@pytest.mark.unit
def test_entry_offline_error_scp_conflicts_with_runtime_linear_aec():
    cfg = _full_offline_cfg(error_scp="error.scp")
    cfg["aec_frontend"] = {
        "linear_aec": {"backend": "pfdkf_torch"},
        "input_routing": "error_farend",
    }

    with pytest.raises(ValueError, match=r"data\.train\.error_scp"):
        object.__new__(AECTrainingEntry)._validate_config(cfg)


@pytest.mark.unit
def test_entry_offline_without_error_accepts_configured_frontend():
    cfg = _full_offline_cfg()
    cfg["aec_frontend"] = {
        "delay_alignment": {"enabled": False},
        "input_routing": "mic_farend",
    }

    object.__new__(AECTrainingEntry)._validate_config(cfg)


@pytest.mark.unit
def test_detect_task_derives_laec_from_canonical_frontend():
    cfg = _full_offline_cfg()
    cfg["aec_frontend"] = {
        "linear_aec": {"backend": "pfdkf_torch"},
        "input_routing": "error_farend",
    }

    assert detect_task(cfg).has_laec is True

    cfg["linear_aec"] = {"backend": "pfdkf_torch"}
    with pytest.raises(ValueError, match="top-level linear_aec"):
        detect_task(cfg)


class _RecordingFrontend:
    def __init__(self, events: list[str] | None = None) -> None:
        self.calls: list[torch.Tensor] = []
        self.events = events

    def process_batch(self, mic, farend, lengths, sample_rate):
        del lengths, sample_rate
        if self.events is not None:
            self.events.append("aec_input_frontend")
        self.calls.append(mic.detach().clone())
        error = mic - farend
        return farend + 1.0, error, torch.ones(mic.shape[0], dtype=torch.bool), {}


class _RecordingSTFT:
    n_fft = 8
    hop_length = 4

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __call__(self, waveform):
        self.events.append("stft")
        return waveform

    def inverse(self, spectra, length):
        self.events.append("istft")
        return spectra[..., :length]


class _RecordingModel(torch.nn.Module):
    aec_input_capabilities = frozenset({"mic_spectra", "farend_spectra"})

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def forward(self, data):
        self.events.append("model")
        data["nearend_estimate_spectra"] = data["mic_spectra"]
        return data


def _raw_batch() -> AECBatch:
    return AECBatch(
        nearend=torch.zeros(1, 4),
        mic=torch.ones(1, 4),
        farend=torch.zeros(1, 4),
        fs=torch.tensor(16000),
        lengths=torch.tensor([4]),
        error=torch.zeros(1, 4),
        error_mask=torch.zeros(1, dtype=torch.bool),
        is_pure_noise=torch.zeros(1, dtype=torch.bool),
        is_pure_echo=torch.zeros(1, dtype=torch.bool),
        is_pure_echo_noise=torch.zeros(1, dtype=torch.bool),
        has_echo=torch.zeros(1, dtype=torch.bool),
    )


def _order_recording_trainer(events: list[str]) -> tuple[AECTrainer, _RecordingFrontend]:
    trainer = object.__new__(AECTrainer)
    trainer.device = torch.device("cpu")
    trainer.use_amp = False
    trainer.amp_dtype = torch.float32
    trainer._strict_checks = False
    trainer.log_interval = 100
    trainer._aec_persistent_state_repair_pending = False
    trainer._accumulate_output_sanitize = lambda *_args: None
    trainer._sanitize_model_output = lambda value: value
    trainer.stft_module = _RecordingSTFT(events)
    trainer.model = _RecordingModel(events)

    frontend = _RecordingFrontend(events)
    trainer._aec_frontend = frontend
    trainer._aec_frontend_config = AECFrontendConfig.from_dict(
        {
            "delay_alignment": {"enabled": True},
            "input_routing": "mic_farend",
        }
    )
    trainer._aec_frontend_enabled = True

    move_to_device = trainer._batch_to_device

    def recording_move_to_device(batch):
        events.append("device")
        return move_to_device(batch)

    trainer._batch_to_device = recording_move_to_device

    def loss_fn(estimate, target, lengths, **_kwargs):
        del lengths
        events.append("loss")
        return (estimate - target).abs().mean()

    trainer.loss_fn = loss_fn
    return trainer, frontend


@pytest.mark.unit
def test_prepare_batch_to_train_forward_orders_frontend_stft_model():
    events: list[str] = []
    trainer, frontend = _order_recording_trainer(events)

    batch, fs_val, _ = trainer._prepare_batch(
        _raw_batch(), epoch=0, batch_idx=1
    )
    trainer._forward_and_compute_loss(batch, epoch=0, batch_idx=1, fs_val=fs_val)

    assert events == [
        "device",
        "aec_input_frontend",
        "stft",
        "stft",
        "model",
        "istft",
        "loss",
    ]
    assert torch.equal(frontend.calls[0], torch.ones(1, 4))


@pytest.mark.unit
def test_offline_and_dynamic_batches_share_identical_frontend_result():
    frontend = _RecordingFrontend()
    trainer = object.__new__(AECTrainer)
    trainer._aec_frontend = frontend
    trainer._aec_frontend_config = AECFrontendConfig.from_dict(
        {
            "linear_aec": {"backend": "pfdkf_torch"},
            "input_routing": "mic_farend_error",
        }
    )
    trainer._aec_frontend_enabled = True
    trainer._batch_fs_hz = lambda *_args: 16000

    def batch() -> AECBatch:
        return AECBatch(
            nearend=torch.zeros(1, 4),
            mic=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            farend=torch.tensor([[0.5, 1.0, 1.5, 2.0]]),
            fs=torch.tensor(16000),
            lengths=torch.tensor([4]),
        )

    offline_result = trainer._run_aec_frontend(batch())
    dynamic_result = trainer._run_aec_frontend(batch())

    for offline_value, dynamic_value in zip(offline_result, dynamic_result):
        assert torch.equal(offline_value, dynamic_value)


@pytest.mark.unit
def test_inference_mode_frontend_outputs_are_cloned_for_training_autograd():
    class InferenceFrontend:
        @torch.inference_mode()
        def process_batch(self, mic, farend, lengths, sample_rate):
            del lengths, sample_rate
            error = mic - farend
            return farend + 1.0, error, torch.ones(mic.shape[0], dtype=torch.bool), {}

    trainer = object.__new__(AECTrainer)
    trainer._aec_frontend = InferenceFrontend()
    trainer._aec_frontend_config = AECFrontendConfig.from_dict(
        {
            "linear_aec": {"backend": "pfdkf_torch"},
            "input_routing": "error_farend",
        }
    )
    trainer._aec_frontend_enabled = True
    trainer._batch_fs_hz = lambda *_args: 16000
    batch = AECBatch(
        nearend=torch.zeros(1, 4),
        mic=torch.ones(1, 4),
        farend=torch.zeros(1, 4),
        fs=torch.tensor(16000),
        lengths=torch.tensor([4]),
    )

    model_mic, aligned, _, _ = trainer._run_aec_frontend(batch)
    weight = torch.nn.Parameter(torch.ones(()))
    loss = (model_mic * weight).sum() + (aligned * weight).sum()
    loss.backward()

    assert weight.grad is not None and torch.isfinite(weight.grad)


@pytest.mark.unit
def test_validation_entry_orders_device_frontend_stft_model():
    events: list[str] = []
    trainer, frontend = _order_recording_trainer(events)

    trainer._val_forward_loss(_raw_batch(), epoch=0, n_loops=0)

    assert events == [
        "device",
        "aec_input_frontend",
        "stft",
        "stft",
        "model",
        "istft",
        "loss",
    ]
    assert torch.equal(frontend.calls[0], torch.ones(1, 4))
