"""Tests for TSE trainer enhancements (tasks/tse/trainer/tse_trainer.py).

覆盖三项增强：
  - 多目标批（num_targets 非 None）：STFT 一次 + repeat_interleave 展开、
    loss 在 [sum(K_i)] 上与 target/lengths 对齐、val 路径同口径、
    sanitize/占位计数按多目标正确处理；
  - SSA（Self-estimated Speech Augmentation）：ssa_prob=0 不触发、=1 整批
    以 no_grad 预前向的 est0 替换 enroll、预提取模式跳过并 warning；
  - multi-task speaker CE：SpeakerCrossEntropyLoss 数值/ignore_index 语义、
    trainer 侧 spk_logits/spk_labels kwargs 接线与反传。

全部小尺寸 CPU 可跑（tiny dict 模型 + 合成 batch，不走真实 Dataset）。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from core.losses import CompositeLoss  # noqa: F401 — 触发 SpeakerCrossEntropyLoss 注册
from core.losses.speaker_ce import SpeakerCrossEntropyLoss
from core.stft import StftIstftModule
from core.utils.inference_artifacts import fingerprint_file
from core.utils.registry import LOSS_REGISTRY
from tasks.tse.cue_contract import (
    TSECue,
    TSECueBatch,
    TSECueBatches,
    TSECueSpec,
    TSECueSpecs,
    TSECues,
)
from tasks.tse.data.batch import TSEBatch
from tasks.tse.data.collate import make_tse_sample, make_tse_target, tse_collate_fn
from tasks.tse.objectives import resolve_dual_self_estimated_objective
from tasks.tse.trainer.tse_trainer import TSETrainer

SR = 16000
T_MIX = 4096
T_ENROLL = 1024


# ----------------------------------------------------------------------
# 测试替身：tiny dict 谱进谱出模型 / 简单波形 loss / 合成 TSE 数据集
# ----------------------------------------------------------------------
class _TinyTSEModel(nn.Module):
    """最小 dict 谱进谱出 TSE 模型：1x1 conv 透传 mix 谱 + 可选 spk 分类头。

    记录每次 forward 收到的 enroll_wav / mix_spectra 形状，供 SSA 与
    多目标展开的断言使用。
    """

    def __init__(self, n_spks: int | None = None) -> None:
        super().__init__()
        self.proj = nn.Conv2d(2, 2, kernel_size=1)
        self.n_spks = n_spks
        if n_spks is not None:
            # enroll_wav 的 4 维统计量 → 分类 logits（模拟句级 embedding 分类头）
            self.cls = nn.Linear(4, n_spks)
        self.enroll_seen: list[torch.Tensor] = []
        self.spk_embedding_seen: list[torch.Tensor] = []
        self.mix_spec_batch_sizes: list[int] = []
        self.data_lengths_seen: list[torch.Tensor] = []
        self.sample_rates_seen: list[int] = []
        self.mixtures_seen: list[torch.Tensor] = []
        self.estimate_spectra_seen: list[torch.Tensor] = []
        self.state_probe = nn.Identity()
        self.training_states_seen: list[tuple[bool, bool]] = []

    def required_tse_input_fields(self):
        return frozenset({"mix_spectra", "enroll_spectra", "enroll_spectra_lengths", "enroll_wav"})

    def forward(self, data: dict) -> dict:
        self.training_states_seen.append((self.training, self.state_probe.training))
        enroll_wav = data["enroll_wav"]
        self.enroll_seen.append(enroll_wav.detach().clone())
        if "spk_embedding" in data:
            self.spk_embedding_seen.append(data["spk_embedding"].detach().clone())
        self.mixtures_seen.append(data["mixture"].detach().clone())
        self.data_lengths_seen.append(data["lengths"].detach().clone())
        self.sample_rates_seen.append(data["sample_rate"])
        mix_spec = data["mix_spectra"]  # (B, F, T, 2)
        self.mix_spec_batch_sizes.append(mix_spec.shape[0])
        est = self.proj(mix_spec.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
        self.estimate_spectra_seen.append(est)
        data["target_estimate_spectra"] = est
        if self.n_spks is not None:
            stats = torch.stack(
                [
                    enroll_wav.mean(dim=1),
                    enroll_wav.std(dim=1),
                    enroll_wav.max(dim=1).values,
                    enroll_wav.min(dim=1).values,
                ],
                dim=1,
            )
            data["spk_logits"] = self.cls(stats)
        return data


class _L1Loss(nn.Module):
    def forward(self, enhanced, clean, lengths=None, **kwargs) -> torch.Tensor:
        return F.l1_loss(enhanced, clean)


class _RecordingL1Loss(_L1Loss):
    def __init__(self) -> None:
        super().__init__()
        self.lengths_seen: list[torch.Tensor] = []

    def forward(self, enhanced, clean, lengths=None, **kwargs) -> torch.Tensor:
        self.lengths_seen.append(lengths.detach().clone())
        return super().forward(enhanced, clean, lengths, **kwargs)


class _RecordingDualL1Loss(_RecordingL1Loss):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[torch.Tensor] = []

    def forward(self, enhanced, clean, lengths=None, **kwargs) -> torch.Tensor:
        value = super().forward(enhanced, clean, lengths, **kwargs)
        self.values.append(value)
        return value


class _TinyDualSelfEstimatedModel(_TinyTSEModel):
    supports_dual_self_estimated = True

    def __init__(self) -> None:
        super().__init__()
        self.enroll_requires_grad: list[bool] = []

    def forward(self, data: dict) -> dict:
        self.enroll_requires_grad.append(data["enroll_wav"].requires_grad)
        output = super().forward(data)
        call_scale = 1.0 + 0.5 * len(self.enroll_seen)
        estimate = output["target_estimate_spectra"] * call_scale
        output["target_estimate_spectra"] = estimate
        self.estimate_spectra_seen[-1] = estimate
        return output


class _RecordingCapabilityLoss(_RecordingL1Loss):
    REQUIRES_NOISY_REFERENCE = True
    REQUIRES_SAMPLE_RATE = True

    def __init__(self) -> None:
        super().__init__()
        self.noisy_seen: list[torch.Tensor] = []
        self.sample_rates_seen: list[int] = []
        self.extra_keys_seen: list[set[str]] = []

    def forward(
        self,
        enhanced,
        clean,
        lengths=None,
        *,
        noisy=None,
        sample_rate=None,
        **kwargs,
    ) -> torch.Tensor:
        assert noisy is not None
        assert sample_rate is not None
        self.noisy_seen.append(noisy.detach().clone())
        self.sample_rates_seen.append(sample_rate)
        self.extra_keys_seen.append(set(kwargs))
        return super().forward(enhanced, clean, lengths, **kwargs)


class _DirectWaveformModel(_TinyTSEModel):
    def __init__(self, output_delta: int) -> None:
        super().__init__()
        self.output_delta = output_delta

    def forward(self, data: dict) -> dict:
        self.enroll_seen.append(data["enroll_wav"].detach().clone())
        self.data_lengths_seen.append(data["lengths"].detach().clone())
        self.sample_rates_seen.append(data["sample_rate"])
        ref_len = data["mixture"].shape[-1]
        wav_len = max(1, ref_len + self.output_delta)
        data["target_estimate"] = torch.ones(
            data["mix_spectra"].shape[0], wav_len, device=data["mix_spectra"].device
        ) * self.proj.weight.mean()
        return data


class _FakeSpeakerEncoder(nn.Module):
    """可控的外部 speaker encoder 替身，避免测试加载真实 checkpoint。"""

    instances: list["_FakeSpeakerEncoder"] = []
    output_kind = "valid"

    def __init__(
        self,
        *,
        model: str,
        pretrained: str | None,
        freeze: bool | None,
        spk_args: dict | None,
        fbank_args: dict | None,
    ) -> None:
        super().__init__()
        self.constructor_kwargs = {
            "model": model,
            "pretrained": pretrained,
            "freeze": freeze,
            "spk_args": spk_args,
            "fbank_args": fbank_args,
        }
        self.embed_dim = int((spk_args or {}).get("embed_dim", 4))
        self.probe = nn.Parameter(torch.tensor(1.0))
        self.inputs: list[torch.Tensor] = []
        window_size = (fbank_args or {}).get("window_size")
        self.fbank = type("_FakeFbank", (), {"window_size": window_size})()
        self.__class__.instances.append(self)

    def forward(self, wav: torch.Tensor):
        self.inputs.append(wav.detach().clone())
        rows = wav.shape[0]
        kind = self.__class__.output_kind
        if kind == "waveform_mean":
            return wav.mean(dim=1, keepdim=True).repeat(1, self.embed_dim)
        if kind == "non_tensor":
            return None
        if kind == "integer":
            return torch.zeros(rows, self.embed_dim, dtype=torch.int64, device=wav.device)
        if kind == "rank":
            return torch.zeros(rows, device=wav.device)
        if kind == "rows":
            return torch.zeros(rows + 1, self.embed_dim, device=wav.device)
        if kind == "dim":
            return torch.zeros(rows, self.embed_dim + 1, device=wav.device)
        if kind == "nonfinite":
            output = torch.zeros(rows, self.embed_dim, device=wav.device)
            output[0, 0] = float("nan")
            return output
        if kind == "device":
            return torch.empty(rows, self.embed_dim, device="meta")
        return torch.arange(
            rows * self.embed_dim, dtype=torch.float32, device=wav.device
        ).reshape(rows, self.embed_dim)


class _MutatingFakeSpeakerEncoder(_FakeSpeakerEncoder):
    """模拟安全 loader 执行期间 checkpoint 被替换。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        Path(kwargs["pretrained"]).write_bytes(b"changed-during-load")


_SPEAKER_CUE_SPECS = TSECueSpecs(
    speaker_waveform=TSECueSpec("waveform", (T_ENROLL,))
)


def _speaker_waveform(batch: TSEBatch) -> TSECueBatch:
    cue = batch.cues.speaker_waveform
    assert cue is not None and cue.lengths is not None
    return cue


def _make_speaker_cue_batch(
    value: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> TSECueBatch:
    rows = value.shape[0]
    if lengths is None:
        lengths = torch.full(
            (rows,), value.shape[-1], dtype=torch.int32, device=value.device
        )
    return TSECueBatch(
        representation="waveform",
        value=value,
        present=torch.ones(rows, dtype=torch.bool, device=value.device),
        lengths=lengths,
        timebase=None,
    )


def _make_tse_batch(
    *,
    mixture: torch.Tensor,
    target: torch.Tensor,
    enroll: torch.Tensor,
    lengths: torch.Tensor,
    enroll_lengths: torch.Tensor,
    num_targets: torch.Tensor,
    spk_labels: torch.Tensor | None = None,
) -> TSEBatch:
    if mixture.ndim == 2:
        mixture = mixture.unsqueeze(1)
    rows = target.shape[0]
    device = mixture.device
    if spk_labels is None:
        spk_labels = torch.full((rows,), -1, dtype=torch.int64, device=device)
    cue_batch = TSECueBatch(
        representation="waveform",
        value=enroll,
        present=torch.ones(rows, dtype=torch.bool, device=device),
        lengths=enroll_lengths,
        timebase=None,
    )
    return TSEBatch(
        mixture=mixture,
        target=target,
        sr=torch.tensor(SR, dtype=torch.int32, device=device),
        lengths=lengths,
        num_targets=num_targets,
        spk_labels=spk_labels,
        target_present=torch.ones(rows, dtype=torch.bool, device=device),
        cues=TSECueBatches(speaker_waveform=cue_batch),
        cue_specs=TSECueSpecs(
            speaker_waveform=TSECueSpec("waveform", (enroll.shape[-1],))
        ),
        keys=tuple(f"sample-{index}" for index in range(mixture.shape[0])),
        speaker_ids=tuple(f"speaker-{index}" for index in range(rows)),
        geometry=(None,) * mixture.shape[0],
        timebase=(None,) * mixture.shape[0],
        provenance=(None,) * mixture.shape[0],
    )


class _SyntheticTSEDataset(Dataset):
    """产 canonical dict item，由 collate 生成唯一 ``TSEBatch``。"""

    cue_specs = _SPEAKER_CUE_SPECS

    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(100 + idx)
        mixture = torch.randn(T_MIX, generator=g) * 0.1
        target = torch.randn(T_MIX, generator=g) * 0.1
        enroll = torch.randn(T_ENROLL, generator=g) * 0.1
        cues = TSECues(
            speaker_waveform=TSECue(
                "waveform", enroll, True, T_ENROLL, None
            )
        )
        return make_tse_sample(
            key=f"sample-{idx}",
            mixture=mixture,
            sr=SR,
            length=T_MIX,
            targets=[
                make_tse_target(
                    target=target,
                    cues=cues,
                    speaker_id=f"speaker-{idx % 3}",
                    spk_label=idx % 3,
                )
            ],
        )


def _make_trainer(
    tmp_path,
    *,
    model: nn.Module | None = None,
    loss_fn: nn.Module | None = None,
    trainer_cfg: dict | None = None,
    model_args: dict | None = None,
) -> TSETrainer:
    model = model or _TinyTSEModel()
    loss_fn = loss_fn or _L1Loss()
    resolved_model_args = model_args or {"joint_training": True}
    spkemb = (resolved_model_args.get("features") or {}).get("spkemb") or {}
    if resolved_model_args.get("joint_training", True) is False and spkemb.get(
        "enabled", False
    ):
        model.required_tse_input_fields = lambda: frozenset(
            {"mix_spectra", "enroll_spectra", "enroll_wav", "spk_embedding"}
        )
    tc = {
        "epochs": 1,
        "gradient_clip_value": 5.0,
        "accumulation_steps": 1,
        "validation_epoch_interval": 0,
        "log_interval": 1,
        "exp_path": str(tmp_path),
        "strict_checks": False,
    }
    tc.update(trainer_cfg or {})
    config = {
        "model": {"type": "_TinyTSEModel", "args": resolved_model_args},
        "trainer": tc,
        "meta": {"seed": 42},
        "data": {"placeholder_num_samples": T_MIX},
    }
    ds = _SyntheticTSEDataset()
    dl = DataLoader(
        ds,
        batch_size=2,
        collate_fn=partial(tse_collate_fn, cue_specs=ds.cue_specs),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return TSETrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
        loss_fn=loss_fn,
        config=config,
        train_loader=dl,
        val_loader=dl,
        sample_rate=SR,
        stft_module=StftIstftModule(n_fft=512, hop_length=128, win_length=512),
    )


def _make_multi_target_batch(device, B: int = 2, K: int = 2) -> TSEBatch:
    """合成 sample-major 多目标 typed batch。"""
    g = torch.Generator().manual_seed(7)
    rows = B * K
    return _make_tse_batch(
        mixture=(torch.randn(B, T_MIX, generator=g) * 0.1).to(device),
        target=(torch.randn(rows, T_MIX, generator=g) * 0.1).to(device),
        enroll=(torch.randn(rows, T_ENROLL, generator=g) * 0.1).to(device),
        lengths=torch.full((B,), T_MIX, dtype=torch.int32, device=device),
        enroll_lengths=torch.full(
            (rows,), T_ENROLL, dtype=torch.int32, device=device
        ),
        spk_labels=torch.arange(rows, dtype=torch.int64, device=device) % 3,
        num_targets=torch.full((B,), K, dtype=torch.int32, device=device),
    )


def _external_spk_model_args(
    embed_dim: int = 4, *, pretrained: str = "fake-speaker.pt"
) -> dict:
    """返回 trainer 外部 SpeakerEncoder 测试所需的最小模型配置。"""
    return {
        "joint_training": False,
        "features": {"spkemb": {"enabled": True, "embed_dim": embed_dim}},
        "speaker_model": {
            "fbank": {"num_mel_bins": 80},
            "speaker_encoder": {
                "model": "fake-speaker",
                "pretrained": pretrained,
                "freeze": False,
                "spk_args": {"embed_dim": embed_dim},
            },
        },
    }


class TestExternalSpeakerEmbedding:
    """非 joint spkemb 的 trainer 接线、隔离和输出合同回归。"""

    def setup_method(self):
        _FakeSpeakerEncoder.instances.clear()
        _FakeSpeakerEncoder.output_kind = "valid"

    def _make_external_trainer(self, tmp_path, monkeypatch, *, embed_dim: int = 4):
        monkeypatch.setattr(
            "tasks.tse.models.speaker.encoder.SpeakerEncoder", _FakeSpeakerEncoder
        )
        pretrained = tmp_path / "fake-speaker.pt"
        pretrained.touch()
        return _make_trainer(
            tmp_path,
            model_args=_external_spk_model_args(
                embed_dim=embed_dim, pretrained=str(pretrained)
            ),
        )

    @staticmethod
    def _to_device(trainer, batch: TSEBatch) -> TSEBatch:
        return batch.to(trainer.device)

    def test_train_single_target_injects_frozen_external_embedding(
        self, tmp_path, monkeypatch
    ):
        trainer = self._make_external_trainer(tmp_path, monkeypatch)
        assert _FakeSpeakerEncoder.instances == []
        batch = self._to_device(trainer, next(iter(trainer.train_loader)))

        loss, was_finite, _ = trainer._forward_and_compute_loss(batch, 0, 0, SR)
        assert was_finite and torch.isfinite(loss)
        assert len(_FakeSpeakerEncoder.instances) == 1
        encoder = _FakeSpeakerEncoder.instances[0]
        assert encoder.constructor_kwargs["pretrained"] == str(
            (tmp_path / "fake-speaker.pt").resolve()
        )
        assert encoder.constructor_kwargs["freeze"] is True
        assert encoder.constructor_kwargs["spk_args"] == {"embed_dim": 4}
        assert encoder.constructor_kwargs["fbank_args"] == {"num_mel_bins": 80}
        assert torch.equal(encoder.inputs[-1], _speaker_waveform(batch).value)
        assert trainer.model.spk_embedding_seen[-1].shape == (2, 4)
        assert all(not parameter.requires_grad for parameter in encoder.parameters())
        assert not encoder.training
        assert all(module is not encoder for module in trainer.model.modules())
        optimizer_parameter_ids = {
            id(parameter)
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        assert all(id(parameter) not in optimizer_parameter_ids for parameter in encoder.parameters())

        loss.backward()
        assert encoder.probe.grad is None

    def test_external_encoder_rejects_checkpoint_changed_during_load(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "tasks.tse.models.speaker.encoder.SpeakerEncoder",
            _MutatingFakeSpeakerEncoder,
        )
        pretrained = tmp_path / "fake-speaker.pt"
        pretrained.write_bytes(b"before-load")
        trainer = _make_trainer(
            tmp_path,
            model_args=_external_spk_model_args(pretrained=str(pretrained)),
        )

        with pytest.raises(RuntimeError, match="loader 执行期间发生变化"):
            trainer._compute_spk_embedding(
                _make_speaker_cue_batch(
                    torch.randn(2, T_ENROLL, device=trainer.device)
                )
            )
        assert trainer._spk_encoder is None
        assert trainer._loaded_spk_checkpoint_identity is None

    def test_checkpoint_state_persists_external_speaker_identity_before_lazy_load(
        self, tmp_path, monkeypatch
    ):
        trainer = self._make_external_trainer(tmp_path, monkeypatch)
        checkpoint = Path(trainer._spk_encoder_cfg["pretrained"])

        state = trainer._build_state(0)

        assert state["tse_external_speaker_checkpoint_identity"] == fingerprint_file(
            checkpoint
        )
        assert trainer._spk_encoder is None

    def test_resume_rejects_changed_external_speaker_checkpoint(
        self, tmp_path, monkeypatch
    ):
        trainer = self._make_external_trainer(tmp_path, monkeypatch)
        checkpoint = Path(trainer._spk_encoder_cfg["pretrained"])
        expected = fingerprint_file(checkpoint)
        trainer._after_resume_checkpoint(
            {"tse_external_speaker_checkpoint_identity": expected}
        )
        trainer._validate_resumed_external_spk_checkpoint_identity()

        checkpoint.write_bytes(b"changed-after-checkpoint")
        with pytest.raises(RuntimeError, match="identity mismatch"):
            trainer._validate_resumed_external_spk_checkpoint_identity()

    def test_resume_external_mode_rejects_legacy_checkpoint_without_identity(
        self, tmp_path, monkeypatch
    ):
        trainer = self._make_external_trainer(tmp_path, monkeypatch)
        trainer._after_resume_checkpoint({})

        with pytest.raises(RuntimeError, match="missing external SpeakerEncoder"):
            trainer._validate_resumed_external_spk_checkpoint_identity()

    def test_external_embedding_is_independent_of_other_rows_padding(
        self, tmp_path, monkeypatch
    ):
        _FakeSpeakerEncoder.output_kind = "waveform_mean"
        trainer = self._make_external_trainer(tmp_path, monkeypatch)
        valid = torch.linspace(-1.0, 1.0, 240, device=trainer.device)

        standalone = trainer._compute_spk_embedding(
            _make_speaker_cue_batch(
                valid.unsqueeze(0),
                torch.tensor([240], dtype=torch.int32, device=trainer.device),
            )
        )
        padded_batch = torch.full((2, 700), 99.0, device=trainer.device)
        padded_batch[0] = torch.linspace(0.0, 2.0, 700, device=trainer.device)
        padded_batch[1, :240] = valid
        batched = trainer._compute_spk_embedding(
            _make_speaker_cue_batch(
                padded_batch,
                torch.tensor([700, 240], dtype=torch.int32, device=trainer.device),
            )
        )

        assert torch.equal(batched[1], standalone[0])
        encoder = _FakeSpeakerEncoder.instances[0]
        assert [input_.shape for input_ in encoder.inputs] == [
            torch.Size([1, 240]),
            torch.Size([1, 240]),
            torch.Size([1, 700]),
        ]

    def test_external_encoder_groups_lengths_and_only_pads_to_fbank_window(
        self, tmp_path, monkeypatch
    ):
        _FakeSpeakerEncoder.output_kind = "waveform_mean"
        trainer = self._make_external_trainer(tmp_path, monkeypatch)
        trainer._spk_encoder_cfg["fbank_args"]["window_size"] = 160
        enroll = torch.full((3, 300), 99.0, device=trainer.device)
        enroll[0, :80] = 1.0
        enroll[1, :200] = 2.0
        enroll[2, :80] = 3.0
        lengths = torch.tensor([80, 200, 80], dtype=torch.int32, device=trainer.device)

        embedding = trainer._compute_spk_embedding(
            _make_speaker_cue_batch(enroll, lengths)
        )

        encoder = _FakeSpeakerEncoder.instances[0]
        assert [input_.shape for input_ in encoder.inputs] == [
            torch.Size([2, 160]),
            torch.Size([1, 200]),
        ]
        assert torch.all(encoder.inputs[0][0, :80] == 1)
        assert torch.all(encoder.inputs[0][0, 80:] == 0)
        assert torch.all(encoder.inputs[0][1, :80] == 3)
        assert torch.all(encoder.inputs[0][1, 80:] == 0)
        assert torch.all(encoder.inputs[1] == 2)
        assert torch.allclose(
            embedding[:, 0],
            torch.tensor([0.5, 2.0, 1.5], device=trainer.device),
        )

    def test_external_embedding_rejects_sum_num_targets_row_mismatch(
        self, tmp_path, monkeypatch
    ):
        trainer = self._make_external_trainer(tmp_path, monkeypatch)
        enroll = torch.randn(2, T_ENROLL, device=trainer.device)
        num_targets = torch.tensor([1, 2], dtype=torch.int32, device=trainer.device)

        with pytest.raises(ValueError, match=r"sum\(num_targets\)"):
            trainer._compute_spk_embedding(
                _make_speaker_cue_batch(enroll),
                num_targets=num_targets,
                mixture_rows=2,
            )

    def test_train_multi_target_embedding_matches_expanded_enroll_rows(
        self, tmp_path, monkeypatch
    ):
        trainer = self._make_external_trainer(tmp_path, monkeypatch)
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)

        loss, was_finite, _ = trainer._forward_and_compute_loss(batch, 0, 0, SR)
        assert was_finite and torch.isfinite(loss)
        encoder = _FakeSpeakerEncoder.instances[0]
        assert torch.equal(encoder.inputs[-1], _speaker_waveform(batch).value)
        assert trainer.model.spk_embedding_seen[-1].shape == (4, 4)

    def test_validation_injects_multi_target_embedding(
        self, tmp_path, monkeypatch
    ):
        trainer = self._make_external_trainer(tmp_path, monkeypatch)
        batch = _make_multi_target_batch(torch.device("cpu"), B=2, K=2)

        loss, estimate, target, lengths, _, _ = trainer._val_forward_loss(
            batch, 0, 1
        )
        assert torch.isfinite(loss)
        assert estimate.shape == target.shape == (4, T_MIX)
        assert lengths.shape == (4,)
        encoder = _FakeSpeakerEncoder.instances[0]
        assert torch.equal(encoder.inputs[-1].cpu(), _speaker_waveform(batch).value.cpu())
        assert trainer.model.spk_embedding_seen[-1].shape == (4, 4)

    def test_external_encoder_is_not_constructed_for_joint_training(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "tasks.tse.models.speaker.encoder.SpeakerEncoder", _FakeSpeakerEncoder
        )
        trainer = _make_trainer(
            tmp_path,
            model_args={
                **_external_spk_model_args(),
                "joint_training": True,
            },
        )
        enroll = torch.randn(2, T_ENROLL, device=trainer.device)
        assert trainer._compute_spk_embedding(
            _make_speaker_cue_batch(enroll)
        ) is None
        assert _FakeSpeakerEncoder.instances == []

    @pytest.mark.parametrize(
        ("output_kind", "error", "match"),
        [
            ("non_tensor", TypeError, "must return a torch.Tensor"),
            ("integer", TypeError, "floating dtype"),
            ("rank", ValueError, "one sentence row per enroll row"),
            ("rows", ValueError, "one sentence row per enroll row"),
            ("dim", ValueError, "dimension mismatch"),
            ("nonfinite", ValueError, "NaN/Inf"),
            ("device", ValueError, "device mismatch"),
        ],
    )
    def test_invalid_external_embedding_fails_closed(
        self, tmp_path, monkeypatch, output_kind, error, match
    ):
        _FakeSpeakerEncoder.output_kind = output_kind
        trainer = self._make_external_trainer(tmp_path, monkeypatch)
        enroll = torch.randn(2, T_ENROLL, device=trainer.device)

        with pytest.raises(error, match=match):
            trainer._compute_spk_embedding(_make_speaker_cue_batch(enroll))


class _TinyDDPSpeakerModel(nn.Module):
    """双 iteration Gloo 合同用的最小 waveform + speaker classifier 模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(8, 8)
        self.classifier = nn.Linear(8, 3)

    def forward(self, mixture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enhanced = self.base(mixture)
        return enhanced, self.classifier(enhanced)


class _TinyDDPDualObjectiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(8, 8)
        self.condition = nn.Linear(1, 8)

    def forward(self, mixture: torch.Tensor, enroll: torch.Tensor) -> torch.Tensor:
        cue = self.condition(enroll.mean(dim=1, keepdim=True))
        return self.base(mixture) + cue


def _gloo_speaker_ce_worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        torch.manual_seed(100 + rank)
        model = nn.parallel.DistributedDataParallel(
            _TinyDDPSpeakerModel(), find_unused_parameters=False
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = CompositeLoss(
            components=[
                {"type": "L1WavLoss", "weight": 1.0},
                {"type": "SpeakerCrossEntropyLoss", "weight": 0.1},
            ]
        )
        # 直接复用真实 trainer wiring；rank 0 全 ignore、rank 1 有监督。
        trainer = object.__new__(TSETrainer)
        trainer.loss_fn = loss_fn
        labels = torch.tensor([-1, -1] if rank == 0 else [0, 1], dtype=torch.int64)
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            mixture = torch.randn(2, 8) + rank + iteration
            target = torch.zeros_like(mixture)
            enhanced, logits = model(mixture)
            spk_ce_kwargs = trainer._composite_spk_ce_kwargs(
                {"spk_logits": logits}, labels
            )
            loss = loss_fn(enhanced, target, **spk_ce_kwargs)
            loss.backward()
            assert model.module.base.weight.grad is not None
            assert model.module.classifier.weight.grad is not None
            optimizer.step()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _gloo_dual_self_estimated_worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(300 + rank)
        model = nn.parallel.DistributedDataParallel(
            _TinyDDPDualObjectiveModel(), find_unused_parameters=False
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = object.__new__(TSETrainer)
        trainer.rank = rank
        trainer.world_size = 2
        trainer.device = torch.device("cpu")
        trainer.model = model
        trainer.config = {
            "trainer": {
                "objectives": {
                    "dual_self_estimated": {
                        "enabled": True,
                        "original_loss_weight": 0.4,
                        "self_loss_weight": 0.6,
                    }
                }
            }
        }
        trainer._tse_model_input_fields = frozenset(
            {"mix_spectra", "enroll_spectra", "enroll_wav"}
        )
        trainer._tse_requires_speaker_waveform = True
        trainer._tse_requires_speaker_embedding = False
        trainer._speaker_cue_source_preference = None
        trainer._uses_external_spk_encoder = False
        trainer._spk_checkpoint_identity_synced = False
        trainer.loss_fn = CompositeLoss(
            components=[{"type": "L1WavLoss", "weight": 1.0}]
        )
        trainer.stft_module = None
        trainer._tse_joint_training = True
        trainer._tse_spkemb_enabled = False
        trainer._dual_self_estimated_loss_weights = resolve_dual_self_estimated_objective(
            trainer.config["trainer"]
        )
        trainer._maybe_ssa_cues = lambda batch, epoch, batch_idx: batch.cues
        trainer._accumulate_output_sanitize = lambda *args: None
        trainer._collect_step_diagnostics = lambda *args: {"out_energy": 1.0}

        def forward_tse_batch(batch, *, cues=None, **kwargs):
            resolved_cues = batch.cues if cues is None else cues
            speaker_waveform = resolved_cues.speaker_waveform
            assert speaker_waveform is not None
            mixture = batch.mixture[:, 0]
            return {
                "target_estimate": model(mixture, speaker_waveform.value),
                "mixture": mixture,
                "target": batch.target,
                "lengths": batch.lengths,
            }

        trainer._forward_tse_batch = forward_tse_batch
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            batch = _make_tse_batch(
                mixture=torch.randn(2, 8) + rank + iteration,
                target=torch.zeros(2, 8),
                enroll=torch.randn(2, 8),
                lengths=torch.full((2,), 8, dtype=torch.int32),
                enroll_lengths=torch.full((2,), 8, dtype=torch.int32),
                num_targets=torch.ones(2, dtype=torch.int32),
            )
            loss, finite, _ = trainer._forward_and_compute_loss(
                batch, iteration, 0, SR
            )
            assert finite
            loss.backward()
            assert all(parameter.grad is not None for parameter in model.parameters())
            optimizer.step()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _gloo_external_spk_failure_worker(
    rank: int, init_file: str, result_dir: str, checkpoint_path: str
) -> None:
    """一 rank 的外部 speaker 输出非法时，所有 rank 必须统一失败且不挂死。"""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        from tasks.tse.models.speaker import encoder as encoder_module

        _FakeSpeakerEncoder.output_kind = "dim" if rank == 0 else "valid"
        encoder_module.SpeakerEncoder = _FakeSpeakerEncoder
        trainer = object.__new__(TSETrainer)
        trainer.rank = rank
        trainer.world_size = 2
        trainer.device = torch.device("cpu")
        trainer._tse_joint_training = False
        trainer._tse_spkemb_enabled = True
        trainer._spk_encoder = None
        trainer._spk_embedding_dim = 4
        trainer._loaded_spk_checkpoint_identity = None
        trainer._spk_checkpoint_identity_synced = False
        trainer._spk_encoder_cfg = {
            "model": "fake-speaker",
            "pretrained": checkpoint_path,
            "spk_args": {"embed_dim": 4},
            "fbank_args": {"num_mel_bins": 80},
        }

        result: dict[str, str | int] = {"rank": rank}
        try:
            trainer._compute_spk_embedding(
                _make_speaker_cue_batch(torch.randn(2, T_ENROLL))
            )
        except Exception as exc:
            result["type"] = type(exc).__name__
            result["message"] = str(exc)
        else:
            result["type"] = "no-error"
            result["message"] = "embedding unexpectedly succeeded"
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        (Path(result_dir) / f"rank{rank}.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _gloo_external_spk_identity_worker(
    rank: int, init_file: str, result_dir: str, checkpoint_dir: str
) -> None:
    """校验各 rank identity，并记录连续两 batch 的 object collective 次数。"""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        from tasks.tse.models.speaker import encoder as encoder_module

        _FakeSpeakerEncoder.output_kind = "valid"
        encoder_module.SpeakerEncoder = _FakeSpeakerEncoder
        trainer = object.__new__(TSETrainer)
        trainer.rank = rank
        trainer.world_size = 2
        trainer.device = torch.device("cpu")
        trainer._tse_joint_training = False
        trainer._tse_spkemb_enabled = True
        trainer._spk_encoder = None
        trainer._spk_embedding_dim = 4
        trainer._loaded_spk_checkpoint_identity = None
        trainer._spk_checkpoint_identity_synced = False
        trainer._spk_encoder_cfg = {
            "model": "fake-speaker",
            "pretrained": str(Path(checkpoint_dir) / f"rank{rank}.pt"),
            "spk_args": {"embed_dim": 4},
            "fbank_args": {"num_mel_bins": 80},
        }

        object_collectives = 0
        original_all_gather_object = dist.all_gather_object

        def counted_all_gather_object(*args, **kwargs):
            nonlocal object_collectives
            object_collectives += 1
            return original_all_gather_object(*args, **kwargs)

        dist.all_gather_object = counted_all_gather_object
        result: dict[str, str | int] = {"rank": rank}
        try:
            trainer._compute_spk_embedding(
                _make_speaker_cue_batch(torch.randn(2, T_ENROLL))
            )
            trainer._compute_spk_embedding(
                _make_speaker_cue_batch(torch.randn(2, T_ENROLL))
            )
        except Exception as exc:
            result["type"] = type(exc).__name__
            result["message"] = str(exc)
        else:
            result["type"] = "success"
            result["message"] = "two batches succeeded"
        finally:
            dist.all_gather_object = original_all_gather_object
        result["object_collectives"] = object_collectives
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        (Path(result_dir) / f"rank{rank}.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


# ----------------------------------------------------------------------
# 通用 waveform loss capability
# ----------------------------------------------------------------------
class TestWaveformLossCapability:
    def test_single_target_receives_mixture_and_runtime_sr(self, tmp_path):
        loss_fn = _RecordingCapabilityLoss()
        trainer = _make_trainer(tmp_path, loss_fn=loss_fn)
        batch = next(iter(trainer.train_loader)).to(trainer.device)

        loss, was_finite, _ = trainer._forward_and_compute_loss(batch, 0, 0, SR)

        assert was_finite and torch.isfinite(loss)
        assert torch.equal(loss_fn.noisy_seen[-1], batch.mixture[:, 0])
        assert loss_fn.sample_rates_seen[-1] == SR
        assert loss_fn.extra_keys_seen[-1] == set()


# ----------------------------------------------------------------------
# 多目标批展开
# ----------------------------------------------------------------------
class TestMultiTargetBatch:
    def test_forward_expands_mix_spectra_once(self, tmp_path):
        """mixture 只算一次 STFT；模型收到的 mix_spectra 已展开为 [sum(K_i)]。"""
        trainer = _make_trainer(tmp_path)
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        loss, was_finite, _ = trainer._forward_and_compute_loss(batch, 0, 0, SR)
        assert was_finite and torch.isfinite(loss)
        # 一次前向，mix 谱行数 = sum(K_i) = 4（展开在谱域，波形不复制进 STFT）
        assert trainer.model.mix_spec_batch_sizes == [4]

    def test_forward_tse_batch_expands_waveform_for_loss(self, tmp_path):
        """data['mixture'] 同步展开，供 SS 混合等 loss kwargs 形状对齐。"""
        trainer = _make_trainer(tmp_path)
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        data = trainer._forward_tse_batch(batch)
        assert data["mixture"].shape == batch.target.shape
        assert data["target_estimate"].shape == batch.target.shape

    def test_train_and_validation_loss_lengths_are_per_target(self, tmp_path):
        loss_fn = _RecordingL1Loss()
        trainer = _make_trainer(tmp_path, loss_fn=loss_fn)
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        batch.lengths = torch.tensor([1000, 2000], dtype=torch.int32, device=trainer.device)

        trainer._forward_and_compute_loss(batch, 0, 0, SR)
        trainer._val_forward_loss(batch, 0, 1)

        expected = torch.tensor([1000, 1000, 2000, 2000], device=trainer.device)
        assert len(trainer.model.data_lengths_seen) == 2
        assert all(torch.equal(value, expected) for value in trainer.model.data_lengths_seen)
        assert len(loss_fn.lengths_seen) == 2
        assert all(torch.equal(value, expected) for value in loss_fn.lengths_seen)

    def test_validation_prepare_finishes_checks_and_masks_before_model_forward(
        self, tmp_path
    ):
        trainer = _make_trainer(tmp_path, trainer_cfg={"strict_checks": True})
        batch = _make_multi_target_batch(torch.device("cpu"), B=2, K=2)
        batch.lengths = torch.tensor([1000, 2000], dtype=torch.int32)
        _speaker_waveform(batch).lengths = torch.tensor([100, 200, 300, 400], dtype=torch.int32)
        batch.mixture.fill_(1)
        batch.target.fill_(1)
        _speaker_waveform(batch).value.fill_(1)

        prepared = trainer._val_prepare_batch(batch, 0, 1)

        assert trainer.model.mix_spec_batch_sizes == []
        assert prepared.fs_val == SR
        assert prepared.lengths_for_loss.tolist() == [1000, 1000, 2000, 2000]
        assert torch.all(prepared.batch.mixture[0, :, 1000:] == 0)
        assert torch.all(prepared.batch.target[1, 1000:] == 0)
        assert torch.all(_speaker_waveform(prepared.batch).value[3, 400:] == 0)

        result = trainer._val_forward_prepared(prepared, 0, 1)
        assert torch.isfinite(result[0])
        assert trainer.model.mix_spec_batch_sizes == [4]

    def test_train_capability_loss_receives_expanded_mixture_and_runtime_sr(
        self, tmp_path
    ):
        loss_fn = _RecordingCapabilityLoss()
        trainer = _make_trainer(tmp_path, loss_fn=loss_fn)
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)

        loss, was_finite, _ = trainer._forward_and_compute_loss(batch, 0, 0, SR)

        expected = torch.repeat_interleave(
            batch.mixture[:, 0], batch.num_targets.long(), dim=0
        )
        assert was_finite and torch.isfinite(loss)
        assert torch.equal(loss_fn.noisy_seen[-1], expected)
        assert loss_fn.sample_rates_seen[-1] == SR
        assert loss_fn.extra_keys_seen[-1] == set()

    def test_val_capability_loss_receives_expanded_mixture_and_runtime_sr(
        self, tmp_path
    ):
        loss_fn = _RecordingCapabilityLoss()
        trainer = _make_trainer(tmp_path, loss_fn=loss_fn)
        batch = _make_multi_target_batch(torch.device("cpu"), B=2, K=2)

        loss, _, _, _, fs_val, _ = trainer._val_forward_loss(batch, 0, 1)

        expected = torch.repeat_interleave(
            batch.mixture[:, 0], batch.num_targets.long(), dim=0
        )
        assert torch.isfinite(loss)
        assert torch.equal(loss_fn.noisy_seen[-1].cpu(), expected)
        assert loss_fn.sample_rates_seen[-1] == fs_val == SR
        assert loss_fn.extra_keys_seen[-1] == set()

    def test_train_defensively_zeroes_all_input_tails(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        batch.lengths = torch.tensor([1000, 2000], dtype=torch.int32, device=trainer.device)
        _speaker_waveform(batch).lengths = torch.tensor(
            [100, 200, 300, 400], dtype=torch.int32, device=trainer.device
        )
        batch.mixture.fill_(1)
        batch.target.fill_(1)
        _speaker_waveform(batch).value.fill_(1)
        sanitized = trainer._sanitize_batch_inputs(batch, 0, 0)
        assert torch.all(sanitized.mixture[0, :, 1000:] == 0)
        assert torch.all(sanitized.target[1, 1000:] == 0)
        assert torch.all(sanitized.target[2, 2000:] == 0)
        assert torch.all(_speaker_waveform(sanitized).value[0, 100:] == 0)
        assert torch.all(_speaker_waveform(sanitized).value[3, 400:] == 0)

    def test_loss_backward_flows(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        loss, was_finite, _ = trainer._forward_and_compute_loss(batch, 0, 0, SR)
        assert was_finite
        loss.backward()
        assert trainer.model.proj.weight.grad is not None
        assert torch.isfinite(trainer.model.proj.weight.grad).all()

    def test_val_forward_multi_target(self, tmp_path):
        """val 路径同口径展开：loss 有限，target/lengths 均按 per-target 对齐。"""
        trainer = _make_trainer(tmp_path)
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        loss, est, target, lengths, fs_val, _ = trainer._val_forward_loss(
            batch, 0, 1
        )
        assert torch.isfinite(loss)
        assert est.shape == target.shape == (4, T_MIX)
        assert lengths.shape[0] == 4

    def test_strict_assert_accepts_multi_target(self, tmp_path):
        trainer = _make_trainer(tmp_path, trainer_cfg={"strict_checks": True})
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        trainer._assert_batch_valid(batch)

    def test_strict_assert_rejects_row_mismatch(self, tmp_path):
        trainer = _make_trainer(tmp_path, trainer_cfg={"strict_checks": True})
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        batch.target = batch.target[:3]
        with pytest.raises(ValueError, match=r"spk_labels must have shape \[3\]"):
            trainer._assert_batch_valid(batch)

    def test_unpack_rejects_legacy_tuple(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        with pytest.raises(TypeError, match="tuple batches are not accepted"):
            trainer._unpack_batch((torch.zeros(1),) * 7)

    def test_placeholder_count_multi_target(self, tmp_path):
        """占位计数按 mixture 条数：一条全零 mixture（含其全部目标行）计 1。"""
        trainer = _make_trainer(tmp_path)
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        batch.mixture[0].zero_()
        batch.target[0].zero_()
        batch.target[1].zero_()  # mixture 0 的两个目标行
        assert trainer._maybe_count_silence_placeholders(batch) == 1


class TestStrictAndRuntimeContracts:
    def _assert_valid(self, trainer, batch, **overrides):
        speaker_waveform = replace(_speaker_waveform(batch))
        candidate = replace(
            batch,
            cues=replace(batch.cues, speaker_waveform=speaker_waveform),
        )
        for field, value in overrides.items():
            if field == "speaker_waveform_lengths":
                speaker_waveform.lengths = value
            else:
                setattr(candidate, field, value)
        trainer._assert_batch_valid(candidate)

    @pytest.mark.parametrize(
        ("field", "value", "error", "match"),
        [
            ("lengths", [1, 2], ValueError, "lengths must have shape"),
            ("lengths", torch.tensor([1.0, 2.0]), TypeError, "torch.int32"),
            ("lengths", torch.tensor([0, 2], dtype=torch.int32), ValueError, "in \\[1"),
            (
                "speaker_waveform_lengths",
                [1, 1, 1, 1],
                TypeError,
                "speaker_waveform lengths must be a tensor",
            ),
            (
                "speaker_waveform_lengths",
                torch.tensor([0, 1, 1, 1], dtype=torch.int32),
                ValueError,
                "positive for present rows",
            ),
            ("num_targets", [2, 2], ValueError, "num_targets must have shape"),
            ("num_targets", torch.tensor([2.0, 2.0]), TypeError, "torch.int32"),
            (
                "num_targets",
                torch.tensor([0, 4], dtype=torch.int32),
                ValueError,
                "positive values",
            ),
            ("spk_labels", [0, 1, 2, 3], ValueError, "spk_labels must have shape"),
            (
                "spk_labels",
                torch.tensor([0.0, 1.0, 2.0, 3.0]),
                TypeError,
                "torch.int64",
            ),
            ("spk_labels", torch.tensor([0, 1, 2]), ValueError, r"shape \[4\]"),
            ("spk_labels", torch.tensor([0, -2, 1, 2]), ValueError, ">= -1"),
            ("sr", torch.tensor(16000.0), TypeError, "torch.int32"),
            ("sr", torch.tensor(0, dtype=torch.int32), ValueError, "positive"),
        ],
    )
    def test_strict_checks_raise_explicit_metadata_errors(
        self, tmp_path, field, value, error, match
    ):
        trainer = _make_trainer(tmp_path, trainer_cfg={"strict_checks": True})
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        if isinstance(value, torch.Tensor):
            value = value.to(trainer.device)
        with pytest.raises(error, match=match):
            self._assert_valid(trainer, batch, **{field: value})

    def test_strict_check_order_follows_typed_batch_contract(self, tmp_path):
        trainer = _make_trainer(tmp_path, trainer_cfg={"strict_checks": True})
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        with pytest.raises(ValueError, match="sr must be positive"):
            self._assert_valid(
                trainer,
                batch,
                lengths=torch.tensor(
                    [0, 1], dtype=torch.int32, device=trainer.device
                ),
                num_targets=torch.tensor(
                    [0, 4], dtype=torch.int32, device=trainer.device
                ),
                sr=torch.tensor(0, dtype=torch.int32, device=trainer.device),
            )
        with pytest.raises(ValueError, match="num_targets must contain positive"):
            self._assert_valid(
                trainer,
                batch,
                num_targets=torch.tensor(
                    [0, 4], dtype=torch.int32, device=trainer.device
                ),
            )

    def test_legacy_model_rejects_missing_speaker_waveform_rows(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        speaker_waveform = _speaker_waveform(batch)
        speaker_waveform.present[0] = False
        speaker_waveform.lengths[0] = 0
        speaker_waveform.value[0].zero_()

        with pytest.raises(
            ValueError,
            match="missing speaker_waveform.*provenance-declared placeholder",
        ):
            trainer._prepare_speaker_conditioning(
                batch,
                batch.cues,
                split="train",
                phase="test_missing_real_cue",
            )

    def test_batch_to_device_recurses_into_typed_cues(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        batch = _make_multi_target_batch(torch.device("cpu"), B=2, K=2)

        moved = trainer._batch_to_device(batch)

        assert moved.mixture.device == trainer.device
        assert _speaker_waveform(moved).value.device == trainer.device
        assert _speaker_waveform(moved).lengths.device == trainer.device

    def test_model_sample_rate_guard_and_data_passthrough(self, tmp_path):
        model = _TinyTSEModel()
        model.sample_rate = SR
        trainer = _make_trainer(tmp_path, model=model)
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        trainer._forward_and_compute_loss(batch, 0, 0, SR)
        assert model.sample_rates_seen == [SR]
        with pytest.raises(ValueError, match="model.sample_rate"):
            trainer._forward_and_compute_loss(batch, 0, 1, 8000)

    @pytest.mark.parametrize("output_delta", [-73, 91])
    def test_direct_waveform_output_is_exact_ref_len_and_tail_zero(
        self, tmp_path, output_delta
    ):
        trainer = _make_trainer(
            tmp_path, model=_DirectWaveformModel(output_delta=output_delta)
        )
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        batch.lengths = torch.tensor([1000, 2000], dtype=torch.int32, device=trainer.device)
        data = trainer._forward_tse_batch(batch, sample_rate=SR)
        assert data["target_estimate"].shape == batch.target.shape
        assert torch.all(data["target_estimate"][0, 1000:] == 0)
        assert torch.all(data["target_estimate"][2, 2000:] == 0)

    def test_istft_output_is_exact_ref_len_and_tail_zero(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        batch.lengths = torch.tensor([1000, 2000], dtype=torch.int32, device=trainer.device)
        data = trainer._forward_tse_batch(batch, sample_rate=SR)
        assert data["target_estimate"].shape == batch.target.shape
        assert torch.all(data["target_estimate"][1, 1000:] == 0)
        assert torch.all(data["target_estimate"][3, 2000:] == 0)

    def test_spectral_output_is_preserved_when_lengths_are_present(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        batch.lengths = torch.tensor([1000, 2000], dtype=torch.int32, device=trainer.device)
        data = trainer._forward_tse_batch(batch, sample_rate=SR)
        assert data["target_estimate_spectra"] is trainer.model.estimate_spectra_seen[-1]

    def test_validation_defensively_zeroes_input_and_output_tails(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        batch = _make_multi_target_batch(torch.device("cpu"), B=2, K=2)
        batch.lengths = torch.tensor([1000, 2000], dtype=torch.int32)
        batch.mixture.fill_(1)
        batch.target.fill_(1)
        _, estimate, target, lengths, _, _ = trainer._val_forward_loss(
            batch, 0, 1
        )
        assert lengths.tolist() == [1000, 1000, 2000, 2000]
        assert torch.all(trainer.model.mixtures_seen[-1][0, 1000:] == 0)
        assert torch.all(target[1, 1000:] == 0)
        assert torch.all(estimate[3, 2000:] == 0)


# ----------------------------------------------------------------------
# SSA（Self-estimated Speech Augmentation）
# ----------------------------------------------------------------------
class TestSSA:
    def test_ssa_prob_zero_keeps_speaker_waveform(self, tmp_path):
        trainer = _make_trainer(tmp_path, trainer_cfg={"ssa_prob": 0.0})
        batch = next(iter(trainer.train_loader)).to(trainer.device)
        original = _speaker_waveform(batch).value
        loss, was_finite, _ = trainer._forward_and_compute_loss(batch, 0, 0, SR)
        assert was_finite
        assert len(trainer.model.enroll_seen) == 1
        assert torch.equal(trainer.model.enroll_seen[0], original)

    def test_ssa_prob_one_replaces_speaker_waveform(self, tmp_path):
        trainer = _make_trainer(tmp_path, trainer_cfg={"ssa_prob": 1.0})
        batch = next(iter(trainer.train_loader)).to(trainer.device)
        original = _speaker_waveform(batch).value
        loss, was_finite, _ = trainer._forward_and_compute_loss(batch, 0, 0, SR)
        assert was_finite
        assert len(trainer.model.enroll_seen) == 2
        first, second = trainer.model.enroll_seen
        assert torch.equal(first, original)
        assert second.shape[-1] == batch.mixture.shape[-1]
        assert not second.requires_grad
        loss.backward()
        assert trainer.model.proj.weight.grad is not None

    def test_ssa_enrollment_uses_per_target_mixture_lengths(self, tmp_path):
        trainer = _make_trainer(tmp_path, trainer_cfg={"ssa_prob": 1.0})
        batch = _make_multi_target_batch(trainer.device, B=2, K=2)
        batch.lengths = torch.tensor([1000, 2000], dtype=torch.int32, device=trainer.device)
        trainer._forward_and_compute_loss(batch, 0, 0, SR)
        ssa_enroll = trainer.model.enroll_seen[1]
        assert torch.all(ssa_enroll[0, 1000:] == 0)
        assert torch.all(ssa_enroll[1, 1000:] == 0)
        assert torch.all(ssa_enroll[2, 2000:] == 0)
        assert torch.all(ssa_enroll[3, 2000:] == 0)

    def test_ssa_deterministic_across_calls(self, tmp_path):
        t1 = _make_trainer(tmp_path / "a", trainer_cfg={"ssa_prob": 0.5})
        t2 = _make_trainer(tmp_path / "b", trainer_cfg={"ssa_prob": 0.5})
        source = next(iter(t1.train_loader))
        decisions = []
        for trainer, batch in (
            (t1, source.to(t1.device)),
            (t2, source.to(t2.device)),
        ):
            original = _speaker_waveform(batch).value
            resolved = trainer._maybe_ssa_cues(batch, 0, 0)
            speaker_waveform = resolved.speaker_waveform
            assert speaker_waveform is not None
            decisions.append(torch.equal(speaker_waveform.value, original))
        assert decisions[0] == decisions[1]

    def test_ssa_skipped_in_preextracted_mode(self, tmp_path, caplog):
        trainer = _make_trainer(
            tmp_path,
            trainer_cfg={"ssa_prob": 1.0},
            model_args={"joint_training": False},
        )
        batch = next(iter(trainer.train_loader)).to(trainer.device)
        with caplog.at_level(logging.WARNING):
            trainer._forward_and_compute_loss(batch, 0, 0, SR)
            trainer._forward_and_compute_loss(batch, 0, 1, SR)
        assert len(trainer.model.enroll_seen) == 2
        assert caplog.text.count("SSA 不生效") == 1

    def test_ssa_prob_out_of_range_raises(self, tmp_path):
        with pytest.raises(ValueError, match="ssa_prob"):
            _make_trainer(tmp_path, trainer_cfg={"ssa_prob": 1.5})

    @pytest.mark.parametrize("value", ["0.5", True, None])
    def test_ssa_prob_rejects_non_numeric_types(self, tmp_path, value):
        with pytest.raises(TypeError, match="ssa_prob"):
            _make_trainer(tmp_path, trainer_cfg={"ssa_prob": value})

    def test_ssa_est0_zero_energy_falls_back(self, tmp_path, caplog):
        trainer = _make_trainer(tmp_path, trainer_cfg={"ssa_prob": 1.0})
        batch = next(iter(trainer.train_loader)).to(trainer.device)
        original = _speaker_waveform(batch).value

        def _zero_forward(batch, **kw):
            return {"target_estimate": torch.zeros_like(batch.target)}

        trainer._forward_tse_batch = _zero_forward
        with caplog.at_level(logging.WARNING):
            resolved = trainer._maybe_ssa_cues(batch, 0, 0)
        speaker_waveform = resolved.speaker_waveform
        assert speaker_waveform is not None
        assert torch.equal(speaker_waveform.value, original)
        assert "energy collapsed" in caplog.text

    def test_ssa_est0_nan_sanitized(self, tmp_path):
        trainer = _make_trainer(tmp_path, trainer_cfg={"ssa_prob": 1.0})
        batch = next(iter(trainer.train_loader)).to(trainer.device)
        original = _speaker_waveform(batch).value

        def _nan_forward(batch, **kw):
            est = torch.randn_like(batch.target) * 0.1
            est[..., :100] = float("nan")
            return {"target_estimate": est}

        trainer._forward_tse_batch = _nan_forward
        resolved = trainer._maybe_ssa_cues(batch, 0, 0)
        speaker_waveform = resolved.speaker_waveform
        assert speaker_waveform is not None
        assert torch.isfinite(speaker_waveform.value).all()
        assert not torch.equal(speaker_waveform.value, original)


class TestDualSelfEstimatedObjective:
    def test_canonical_trainer_objective_accepts_configured_weights(self, tmp_path):
        trainer = _make_trainer(
            tmp_path,
            model=_TinyDualSelfEstimatedModel(),
            trainer_cfg={
                "objectives": {
                    "dual_self_estimated": {
                        "enabled": True,
                        "original_loss_weight": 0.4,
                        "self_loss_weight": 0.6,
                    }
                }
            },
        )
        assert trainer._dual_self_estimated_loss_weights == (0.4, 0.6)

    def test_canonical_trainer_objective_accepts_non_default_weights(self, tmp_path):
        trainer = _make_trainer(
            tmp_path,
            model=_TinyDualSelfEstimatedModel(),
            trainer_cfg={
                "objectives": {
                    "dual_self_estimated": {
                        "enabled": True,
                        "original_loss_weight": 0.3,
                        "self_loss_weight": 0.7,
                    }
                }
            },
        )
        assert trainer._dual_self_estimated_loss_weights == (0.3, 0.7)

    def test_disabled_canonical_objective_runs_single_forward(self, tmp_path):
        model = _TinyDualSelfEstimatedModel()
        trainer = _make_trainer(
            tmp_path,
            model=model,
            trainer_cfg={
                "objectives": {
                    "dual_self_estimated": {"enabled": False}
                }
            },
        )
        batch = next(iter(trainer.train_loader)).to(trainer.device)
        loss, was_finite, _ = trainer._forward_and_compute_loss(
            batch, epoch=0, batch_idx=0, fs_val=SR
        )
        assert was_finite
        assert trainer._dual_self_estimated_loss_weights is None
        assert len(model.enroll_seen) == 1
        loss.backward()

    def test_two_estimates_are_weighted_and_self_enrollment_is_detached(
        self, tmp_path
    ):
        model = _TinyDualSelfEstimatedModel()
        loss_fn = _RecordingDualL1Loss()
        trainer = _make_trainer(
            tmp_path,
            model=model,
            loss_fn=loss_fn,
            trainer_cfg={
                "objectives": {
                    "dual_self_estimated": {"enabled": True}
                }
            },
        )
        batch = next(iter(trainer.train_loader)).to(trainer.device)

        loss, was_finite, info = trainer._forward_and_compute_loss(
            batch, epoch=0, batch_idx=0, fs_val=SR
        )

        assert was_finite
        assert len(model.enroll_seen) == len(loss_fn.values) == 2
        torch.testing.assert_close(
            model.enroll_seen[1],
            trainer.stft_module.inverse(model.estimate_spectra_seen[0], T_MIX),
        )
        assert model.enroll_requires_grad == [False, False]
        torch.testing.assert_close(
            loss,
            0.4 * loss_fn.values[0] + 0.6 * loss_fn.values[1],
        )
        assert info["dual_original_loss_weight"] == pytest.approx(0.4)
        assert info["dual_self_loss_weight"] == pytest.approx(0.6)

        loss.backward()
        assert model.proj.weight.grad is not None
        assert model.proj.weight.grad.abs().sum() > 0

    def test_composite_breakdown_matches_weighted_dual_total(self, tmp_path):
        model = _TinyDualSelfEstimatedModel()
        loss_fn = CompositeLoss(
            components=[{"type": "L1WavLoss", "weight": 1.0, "name": "L1"}]
        )
        trainer = _make_trainer(
            tmp_path,
            model=model,
            loss_fn=loss_fn,
            trainer_cfg={
                "objectives": {
                    "dual_self_estimated": {"enabled": True}
                }
            },
        )
        batch = next(iter(trainer.train_loader)).to(trainer.device)

        loss, was_finite, _ = trainer._forward_and_compute_loss(
            batch, epoch=0, batch_idx=0, fs_val=SR
        )

        assert was_finite
        breakdown = loss_fn._breakdown_tensors
        assert len(breakdown) == 1
        decoded = [
            trainer.stft_module.inverse(spectra, T_MIX).clamp(-1.0, 1.0)
            for spectra in model.estimate_spectra_seen
        ]
        raw_losses = [F.l1_loss(estimate, batch.target) for estimate in decoded]
        assert not torch.allclose(raw_losses[0], raw_losses[1], rtol=0, atol=1e-5)
        expected = 0.4 * raw_losses[0] + 0.6 * raw_losses[1]
        torch.testing.assert_close(loss.detach(), expected)
        torch.testing.assert_close(breakdown[0][3], expected)

    def test_validation_keeps_single_original_enrollment_pass(self, tmp_path):
        model = _TinyDualSelfEstimatedModel()
        trainer = _make_trainer(tmp_path, model=model)

        loss, estimate, target, _, _, _ = trainer._val_forward_loss(
            next(iter(trainer.val_loader)), epoch=0, n_loops=1
        )

        assert torch.isfinite(loss)
        assert estimate.shape == target.shape
        assert len(model.enroll_seen) == 1

    def test_random_ssa_and_dual_objective_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="cannot be combined.*ssa_prob"):
            _make_trainer(
                tmp_path,
                model=_TinyDualSelfEstimatedModel(),
                trainer_cfg={
                    "ssa_prob": 0.5,
                    "objectives": {
                        "dual_self_estimated": {"enabled": True}
                    },
                },
            )

    def test_two_iteration_gloo_ddp_keeps_collective_sequence(self, tmp_path):
        if not dist.is_gloo_available():
            pytest.skip("PyTorch Gloo backend unavailable")
        mp.spawn(
            _gloo_dual_self_estimated_worker,
            args=(str(tmp_path / "dual_objective_init"),),
            nprocs=2,
            join=True,
        )


# ----------------------------------------------------------------------
# multi-task speaker CE
# ----------------------------------------------------------------------
class TestSpeakerCrossEntropyLoss:
    def test_registered_in_loss_registry(self):
        assert LOSS_REGISTRY.get("SpeakerCrossEntropyLoss") is SpeakerCrossEntropyLoss

    def test_matches_manual_ce(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 2, 1, 0])
        loss = SpeakerCrossEntropyLoss()(torch.zeros(4, 8), torch.zeros(4, 8),
                                         spk_logits=logits, spk_labels=labels)
        assert torch.allclose(loss, F.cross_entropy(logits, labels))

    def test_valid_label_bounds_use_one_cpu_transfer_without_scalar_items(
        self, monkeypatch
    ):
        """min/max 合并为一次 CPU transfer，禁止逐 scalar ``item()``。"""
        original_cpu = torch.Tensor.cpu
        cpu_calls = 0

        def counted_cpu(tensor, *args, **kwargs):
            nonlocal cpu_calls
            cpu_calls += 1
            return original_cpu(tensor, *args, **kwargs)

        def forbidden_item(tensor, *args):
            raise AssertionError("speaker CE label validation must not call Tensor.item()")

        monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
        monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, -1, 2, 1])

        loss = SpeakerCrossEntropyLoss()(
            torch.zeros(4, 8),
            torch.zeros(4, 8),
            spk_logits=logits,
            spk_labels=labels,
        )

        assert torch.isfinite(loss)
        assert cpu_calls == 1

    def test_ignore_index_skips_unknown(self):
        """spk_labels=-1（未知说话人）被 ignore：loss 等于有效子集上的 CE。"""
        torch.manual_seed(0)
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, -1, 2, -1])
        loss = SpeakerCrossEntropyLoss()(torch.zeros(4, 8), torch.zeros(4, 8),
                                         spk_logits=logits, spk_labels=labels)
        valid = labels != -1
        assert torch.allclose(loss, F.cross_entropy(logits[valid], labels[valid]))

    @pytest.mark.parametrize(
        "labels",
        [
            torch.tensor([0.5, 1.0], dtype=torch.float32),
            torch.tensor([False, True], dtype=torch.bool),
        ],
    )
    def test_rejects_non_integer_label_dtype_before_conversion(self, labels):
        with pytest.raises((TypeError, ValueError), match="spk_labels.*dtype"):
            SpeakerCrossEntropyLoss()(
                torch.zeros(2, 8),
                torch.zeros(2, 8),
                spk_logits=torch.randn(2, 3),
                spk_labels=labels,
            )

    def test_rejects_non_floating_logits_dtype(self):
        with pytest.raises((TypeError, ValueError), match="spk_logits.*floating.*dtype"):
            SpeakerCrossEntropyLoss()(
                torch.zeros(2, 8),
                torch.zeros(2, 8),
                spk_logits=torch.ones(2, 3, dtype=torch.int64),
                spk_labels=torch.tensor([0, 1], dtype=torch.int64),
            )

    def test_rejects_label_and_logit_device_mismatch(self):
        logits = torch.empty(2, 3, dtype=torch.float32, device="meta")
        labels = torch.tensor([0, 1], dtype=torch.int64, device="cpu")

        with pytest.raises(
            (TypeError, ValueError),
            match="spk_labels.*device.*cpu.*spk_logits.*meta",
        ):
            SpeakerCrossEntropyLoss()(
                torch.zeros(2, 8),
                torch.zeros(2, 8),
                spk_logits=logits,
                spk_labels=labels,
            )

    @pytest.mark.parametrize(
        ("ignore_index", "labels", "expected_range"),
        [
            (-1, [0, -2], "-2..0"),
            (-1, [0, 3], "0..3"),
            (-9, [-1, 1], "-1..1"),
        ],
    )
    def test_valid_labels_outside_logit_class_range_fail_fast(
        self, ignore_index, labels, expected_range
    ):
        logits = torch.randn(2, 3)

        with pytest.raises(
            ValueError,
            match=rf"label range {expected_range}.*C=3",
        ):
            SpeakerCrossEntropyLoss(ignore_index=ignore_index)(
                torch.zeros(2, 8),
                torch.zeros(2, 8),
                spk_logits=logits,
                spk_labels=torch.tensor(labels),
            )

    def test_all_unknown_returns_zero(self):
        logits = torch.randn(2, 3, requires_grad=True)
        labels = torch.tensor([-1, -1])
        loss = SpeakerCrossEntropyLoss()(torch.zeros(2, 8), torch.zeros(2, 8),
                                         spk_logits=logits, spk_labels=labels)
        assert loss.item() == 0.0
        loss.backward()  # 可反传（零梯度），不破坏 DDP grad hooks
        assert logits.grad is not None
        assert torch.equal(logits.grad, torch.zeros_like(logits))

    @pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
    def test_all_unknown_nonfinite_logits_return_exact_finite_zero(self, nonfinite):
        logits = torch.tensor(
            [[nonfinite, 1.0, -2.0], [3.0, nonfinite, 4.0]],
            requires_grad=True,
        )
        labels = torch.full((2,), -1, dtype=torch.int64)

        loss = SpeakerCrossEntropyLoss()(
            torch.zeros(2, 8),
            torch.zeros(2, 8),
            spk_logits=logits,
            spk_labels=labels,
        )

        assert torch.isfinite(loss)
        assert loss.item() == 0.0
        assert loss.dtype == logits.dtype
        assert loss.device == logits.device
        loss.backward()
        assert logits.grad is not None
        assert torch.equal(logits.grad, torch.zeros_like(logits))

    def test_missing_logits_and_labels_fail_fast(self):
        with pytest.raises(ValueError, match="SpeakerCrossEntropyLoss.*spk_logits.*spk_labels"):
            SpeakerCrossEntropyLoss()(torch.zeros(2, 8), torch.zeros(2, 8))

    def test_missing_labels_fail_fast_even_when_logits_require_grad(self):
        logits = torch.randn(2, 3, requires_grad=True)
        with pytest.raises(ValueError, match="spk_labels"):
            SpeakerCrossEntropyLoss()(
                torch.zeros(2, 8),
                torch.zeros(2, 8),
                spk_logits=logits,
            )


class TestMultiTaskWiring:
    def _composite(self) -> CompositeLoss:
        return CompositeLoss(
            components=[
                {"type": "SISNRLoss", "weight": 1.0},
                {"type": "SpeakerCrossEntropyLoss", "weight": 0.1},
            ]
        )

    def test_spk_logits_kwargs_reach_ce_component(self, tmp_path):
        """模型输出 spk_logits 时 CE 分量参与 loss 且梯度到达分类头。"""
        model = _TinyTSEModel(n_spks=3)
        trainer = _make_trainer(tmp_path, model=model, loss_fn=self._composite())
        batch = next(iter(trainer.train_loader)).to(trainer.device)
        loss, was_finite, _ = trainer._forward_and_compute_loss(batch, 0, 0, SR)
        assert was_finite
        loss.backward()
        assert model.cls.weight.grad is not None
        assert model.cls.weight.grad.abs().sum() > 0

    def test_missing_spk_labels_fails_fast_when_speaker_ce_enabled(self, tmp_path):
        """spk_labels=None（仅可能来自手工构造的 batch）遇非零 Speaker CE 必须 fail-fast。"""
        model = _TinyTSEModel(n_spks=3)
        trainer = _make_trainer(tmp_path, model=model, loss_fn=self._composite())

        with pytest.raises(
            ValueError,
            match=r"spk_labels is None.*SpeakerCrossEntropyLoss",
        ):
            trainer._composite_spk_ce_kwargs(
                {"spk_logits": torch.randn(2, 3)},
                None,
            )

    def test_no_spk_logits_fails_fast_when_speaker_ce_enabled(self, tmp_path):
        """启用非零 Speaker CE 时模型缺少 spk_logits 也不得静默跳过。"""
        trainer = _make_trainer(tmp_path, loss_fn=self._composite())
        with pytest.raises(ValueError, match="spk_logits.*SpeakerCrossEntropyLoss"):
            trainer._composite_spk_ce_kwargs({}, torch.tensor([0, 1]))

    def test_non_composite_loss_no_ce_kwargs(self, tmp_path):
        trainer = _make_trainer(tmp_path, loss_fn=_L1Loss())
        data = {"spk_logits": torch.randn(2, 3)}
        assert trainer._composite_spk_ce_kwargs(data, torch.tensor([0, 1])) == {}

    def test_two_iteration_gloo_ddp_handles_rank_asymmetric_all_ignore(self, tmp_path):
        """rank 0 全 ignore、rank 1 有监督时连续两轮 DDP 反传均完成。"""
        if not dist.is_gloo_available():
            pytest.skip("PyTorch Gloo backend unavailable")
        init_file = tmp_path / "gloo_init"
        mp.spawn(
            _gloo_speaker_ce_worker,
            args=(str(init_file),),
            nprocs=2,
            join=True,
        )

    def test_gloo_external_speaker_failure_is_unified_without_hang(self, tmp_path):
        """单 rank 外部 embedding 失败时，两 rank 应在有限时间内统一报错。"""
        if not dist.is_gloo_available():
            pytest.skip("PyTorch Gloo backend unavailable")
        init_file = tmp_path / "external_spk_init"
        result_dir = tmp_path / "external_spk_results"
        checkpoint = tmp_path / "fake-speaker.pt"
        checkpoint.write_bytes(b"same-checkpoint")
        ctx = mp.spawn(
            _gloo_external_spk_failure_worker,
            args=(str(init_file), str(result_dir), str(checkpoint)),
            nprocs=2,
            join=False,
        )
        deadline = time.monotonic() + 30.0
        while not ctx.join(max(0.0, deadline - time.monotonic())):
            if time.monotonic() >= deadline:
                for process in ctx.processes:
                    if process.is_alive():
                        process.terminate()
                raise TimeoutError(
                    "external speaker DDP failure worker did not finish; "
                    "collective sequence may have hung"
                )

        results = [
            json.loads((result_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
            for rank in range(2)
        ]
        assert all(result["type"] == "RuntimeError" for result in results)
        messages = [result["message"] for result in results]
        assert messages[0] == messages[1]
        assert "rank 0" in messages[0]
        assert "dimension mismatch" in messages[0].lower()

    def test_gloo_external_speaker_identity_mismatch_is_unified_without_hang(
        self, tmp_path
    ):
        """两 rank 加载不同合法 checkpoint 时应统一失败且不挂死。"""
        if not dist.is_gloo_available():
            pytest.skip("PyTorch Gloo backend unavailable")
        init_file = tmp_path / "external_spk_identity_init"
        result_dir = tmp_path / "external_spk_identity_results"
        checkpoint_dir = tmp_path / "external_spk_checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "rank0.pt").write_bytes(b"rank-zero-checkpoint")
        (checkpoint_dir / "rank1.pt").write_bytes(b"rank-one-checkpoint")
        ctx = mp.spawn(
            _gloo_external_spk_identity_worker,
            args=(str(init_file), str(result_dir), str(checkpoint_dir)),
            nprocs=2,
            join=False,
        )
        deadline = time.monotonic() + 30.0
        while not ctx.join(max(0.0, deadline - time.monotonic())):
            if time.monotonic() >= deadline:
                for process in ctx.processes:
                    if process.is_alive():
                        process.terminate()
                raise TimeoutError(
                    "external speaker identity worker did not finish; "
                    "collective sequence may have hung"
                )

        results = [
            json.loads((result_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
            for rank in range(2)
        ]
        assert all(result["type"] == "RuntimeError" for result in results)
        messages = [result["message"] for result in results]
        assert messages[0] == messages[1]
        assert "identity mismatch" in messages[0]
        assert "rank 0" in messages[0]
        assert "rank 1" in messages[0]
        assert all(result["object_collectives"] == 1 for result in results)

    def test_gloo_external_speaker_identity_sync_runs_once(self, tmp_path):
        """同内容异路径 checkpoint 连续两 batch 只执行一次 identity gather。"""
        if not dist.is_gloo_available():
            pytest.skip("PyTorch Gloo backend unavailable")
        init_file = tmp_path / "external_spk_identity_once_init"
        result_dir = tmp_path / "external_spk_identity_once_results"
        checkpoint_dir = tmp_path / "external_spk_identity_once_checkpoints"
        checkpoint_dir.mkdir()
        for rank in range(2):
            (checkpoint_dir / f"rank{rank}.pt").write_bytes(b"same-checkpoint")
        ctx = mp.spawn(
            _gloo_external_spk_identity_worker,
            args=(str(init_file), str(result_dir), str(checkpoint_dir)),
            nprocs=2,
            join=False,
        )
        deadline = time.monotonic() + 30.0
        while not ctx.join(max(0.0, deadline - time.monotonic())):
            if time.monotonic() >= deadline:
                for process in ctx.processes:
                    if process.is_alive():
                        process.terminate()
                raise TimeoutError(
                    "external speaker identity worker did not finish; "
                    "collective sequence may have hung"
                )

        results = [
            json.loads((result_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
            for rank in range(2)
        ]
        assert all(result["type"] == "success" for result in results)
        assert all(result["object_collectives"] == 1 for result in results)


def test_tse_throughput_counts_mixtures_and_target_rows_separately():
    trainer = object.__new__(TSETrainer)
    batch = _make_multi_target_batch(torch.device("cpu"), B=2, K=2)

    assert trainer._throughput_row_counts(batch) == (2, 4)


def test_tse_throughput_logs_explicit_rate_names():
    trainer = object.__new__(TSETrainer)
    logged = []
    trainer._log_scalar = lambda name, value, epoch: logged.append(
        (name, value, epoch)
    )

    trainer._log_task_throughput(12.5, 30.0, epoch=3)

    assert logged == [
        ("system/mixtures_per_sec", 12.5, 3),
        ("system/target_rows_per_sec", 30.0, 3),
    ]


pytestmark = [pytest.mark.unit]
