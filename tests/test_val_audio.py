"""PASE 家族 validation 验证音频写入器的 unit 合同测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tasks.se.trainer.val_audio import (
    ValAudioConfig,
    ValAudioSampleSettings,
    ValAudioWriter,
    parse_val_audio_config,
    preserve_rng_state,
    waveform_to_numpy,
)
from tasks.se.models.pase_family.common.utils import peak_normalize_to

pytestmark = pytest.mark.unit


def _writer(tmp_path, **overrides) -> ValAudioWriter:
    raw = {"enabled": True, **overrides}
    cfg = parse_val_audio_config(raw)
    return ValAudioWriter(
        cfg,
        exp_path=str(tmp_path / "exp"),
        rank=0,
        device=torch.device("cpu"),
        sample_rate=16000,
    )


class TestParseConfig:
    def test_defaults(self):
        cfg = parse_val_audio_config({"enabled": True})
        assert cfg == ValAudioConfig(
            enabled=True,
            output_dir=None,
            save_initial_epochs=10,
            save_every_n_epochs=10,
            initial_epoch_reference="start_epoch",
            max_batches=10,
            vocoder_ckpt_path=None,
            vocoder_kind="unipase",
            wavlm_vocoder_ckpt_path=None,
            wavlm_vocoder_kind="unipase",
            sample_settings=None,
        )

    def test_unknown_key_rejected(self):
        with pytest.raises(TypeError, match="Unknown trainer.val_audio"):
            parse_val_audio_config({"enabled": True, "max_batch": 5})

    def test_unknown_sample_setting_rejected(self):
        with pytest.raises(TypeError, match="sample_settings"):
            parse_val_audio_config(
                {
                    "enabled": True,
                    "sample_settings": {
                        "steps": 8,
                        "cfg_strength": 0.5,
                        "sway": -1.0,
                    },
                }
            )

    def test_missing_sample_setting_rejected(self):
        with pytest.raises(ValueError, match="cfg_strength|sway"):
            parse_val_audio_config(
                {"enabled": True, "sample_settings": {"steps": 8}}
            )

    @pytest.mark.parametrize(
        "raw, exc",
        [
            ({"enabled": 1}, TypeError),
            ({"enabled": True, "max_batches": 0}, ValueError),
            ({"enabled": True, "save_every_n_epochs": 0}, ValueError),
            ({"enabled": True, "initial_epoch_reference": "epoch0"}, ValueError),
            ({"enabled": True, "vocoder_kind": "pase"}, ValueError),
            ({"enabled": True, "output_dir": "  "}, TypeError),
            ({"enabled": True, "sample_settings": {"steps": 0, "cfg_strength": 0.5}}, ValueError),
            (
                {"enabled": True, "sample_settings": {"steps": 8, "cfg_strength": float("inf")}},
                ValueError,
            ),
        ],
    )
    def test_invalid_values_rejected(self, raw, exc):
        with pytest.raises(exc):
            parse_val_audio_config(raw)

    def test_disabled_is_zero_cost(self, tmp_path):
        cfg = parse_val_audio_config({"enabled": False})
        assert cfg.enabled is False
        assert (
            ValAudioWriter.from_config(
                {"trainer": {"exp_path": str(tmp_path), "val_audio": {"enabled": False}}},
                rank=0,
                device=torch.device("cpu"),
                sample_rate=16000,
            )
            is None
        )
        assert (
            ValAudioWriter.from_config(
                {"trainer": {}},
                rank=0,
                device=torch.device("cpu"),
                sample_rate=16000,
            )
            is None
        )

    def test_sample_settings_parsed(self):
        cfg = parse_val_audio_config(
            {
                "enabled": True,
                "sample_settings": {
                    "steps": 8,
                    "cfg_strength": 0.5,
                    "sway_sampling_coef": -1.0,
                },
            }
        )
        assert cfg.sample_settings == ValAudioSampleSettings(
            steps=8, cfg_strength=0.5, sway_sampling_coef=-1.0
        )
        assert cfg.sample_settings.as_kwargs() == {
            "steps": 8,
            "cfg_strength": 0.5,
            "sway_sampling_coef": -1.0,
        }


class TestShouldSave:
    def test_initial_window_and_periodic(self, tmp_path):
        writer = _writer(tmp_path, save_initial_epochs=10, max_batches=10)
        assert writer.should_save(1, 1, 1) is True
        assert writer.should_save(10, 1, 10) is True
        # 官方 1-based：epoch 10 命中初始窗口与 10%10==0；11 不命中。
        assert writer.should_save(11, 1, 1) is False
        assert writer.should_save(12, 1, 1) is False
        assert writer.should_save(19, 1, 1) is False
        assert writer.should_save(20, 1, 1) is True
        assert writer.should_save(21, 1, 1) is False
        assert writer.should_save(11, 1, 11) is False
        assert writer.should_save(20, 1, 0) is False

    def test_gap_zero_reference(self, tmp_path):
        writer = _writer(
            tmp_path,
            save_initial_epochs=10,
            initial_epoch_reference="zero",
            max_batches=5,
        )
        assert writer.should_save(1, 51, 1) is True
        assert writer.should_save(9, 51, 5) is True
        assert writer.should_save(10, 51, 1) is True
        assert writer.should_save(11, 51, 1) is False
        assert writer.should_save(20, 51, 1) is True
        # 续训 epoch 51 不落在 10 的倍数。
        assert writer.should_save(51, 51, 1) is False
        assert writer.should_save(60, 51, 1) is True

    def test_non_rank0_never_saves(self, tmp_path):
        cfg = parse_val_audio_config({"enabled": True})
        writer = ValAudioWriter(
            cfg,
            exp_path=str(tmp_path / "exp"),
            rank=1,
            device=torch.device("cpu"),
            sample_rate=16000,
        )
        assert writer.should_save(1, 1, 1) is False
        writer.save(1, 1, {"noisy": torch.zeros(8)})
        assert not writer.root.exists()


class TestSave:
    def test_naming_and_content(self, tmp_path):
        writer = _writer(tmp_path)
        noisy = torch.linspace(-0.5, 0.5, 160)
        enh = torch.linspace(-0.25, 0.25, 320)
        writer.save(
            3,
            2,
            {"noisy": noisy, "enh_epoch{epoch:03d}": enh},
        )
        root = tmp_path / "exp" / "samples"
        assert (root / "2_noisy.wav").exists()
        # 官方 1-based epoch：trainer epoch=3 -> 文件名 epoch003。
        assert (root / "2_enh_epoch003.wav").exists()
        data, sr = sf.read(root / "2_enh_epoch003.wav", dtype="float32")
        assert sr == 16000
        assert data.dtype == np.float32
        assert data.shape == (320,)
        np.testing.assert_allclose(data, enh.numpy(), atol=1e-4)

    def test_output_dir_override(self, tmp_path):
        writer = _writer(tmp_path, output_dir=str(tmp_path / "custom"))
        writer.save(1, 1, {"noisy": torch.zeros(8)})
        assert (tmp_path / "custom" / "1_noisy.wav").exists()

    def test_rank0_only_write(self, tmp_path):
        cfg = parse_val_audio_config({"enabled": True})
        writer = ValAudioWriter(
            cfg,
            exp_path=str(tmp_path / "exp"),
            rank=1,
            device=torch.device("cpu"),
            sample_rate=48000,
        )
        writer.save(1, 1, {"noisy": torch.zeros(8)})
        assert not (tmp_path / "exp" / "samples").exists()

    def test_multi_sample_batch_keeps_first_sample(self):
        data = waveform_to_numpy(torch.arange(16, dtype=torch.float32).reshape(2, 8))
        np.testing.assert_allclose(data, np.arange(8, dtype=np.float32))


class TestPeakNormalize:
    def test_peak_is_0_9(self):
        wav = torch.tensor([-2.0, 1.0, 0.5])
        out = peak_normalize_to(wav)
        assert torch.isclose(out.abs().amax(), torch.tensor(0.9))

    def test_zero_input_stays_finite(self):
        out = peak_normalize_to(torch.zeros(4))
        assert torch.isfinite(out).all()
        assert float(out.abs().amax()) == 0.0


class TestRngGuard:
    def test_rng_state_restored(self):
        torch.manual_seed(1234)
        before = torch.get_rng_state().clone()
        with preserve_rng_state(torch.device("cpu")):
            torch.randn(64)
        assert torch.equal(torch.get_rng_state(), before)


class TestStageRegistration:
    def test_gan_wrappers_declare_stage(self):
        from tasks.se.models.pase_family.gap_urgenet.gan_wrappers import (
            GAPURGENetPostNetGenerator,
        )
        from tasks.se.models.pase_family.unipase.gan_wrappers import (
            UniPASEAdapterGenerator,
            UniPASEPostNetGenerator,
            UniPASEVocoderGenerator,
        )

        assert UniPASEVocoderGenerator.val_audio_stage == "vocoder"
        assert UniPASEAdapterGenerator.val_audio_stage == "adapter"
        assert UniPASEPostNetGenerator.val_audio_stage == "postnet"
        assert GAPURGENetPostNetGenerator.val_audio_stage == "gap_postnet"
        assert UniPASEAdapterGenerator.val_audio_vocoder_roles == (
            "vocoder",
            "wavlm_vocoder",
        )

    def test_pase_and_gap_predictor_declare_stage(self):
        from tasks.se.models.pase_family.gap_urgenet.gan_wrappers import (
            GAPURGENetPredictor,
        )
        from tasks.se.models.pase_family.pase.model import PASE

        assert GAPURGENetPredictor.val_audio_stage == "predictor"
        assert GAPURGENetPredictor.val_audio_vocoder_roles == ()
        assert PASE.val_audio_sample_rate == 16000
        single = type("Stub", (), {"use_dual_stream": False})()
        dual = type("Stub", (), {"use_dual_stream": True})()
        assert PASE.val_audio_stage.fget(single) == "vocoder"
        assert PASE.val_audio_stage.fget(dual) == "pase_dual"

    def test_dual_input_models_declare_stage(self):
        from tasks.se.models.pase_family.common.dewavlm_distill import (
            DeWavLMDistillModel,
        )
        from tasks.se.models.pase_family.stupase.flow_train_wrapper import (
            StuPASEFlowTrainModel,
        )

        assert DeWavLMDistillModel.val_audio_stage == "distill"
        assert DeWavLMDistillModel.val_audio_vocoder_roles == ("vocoder",)
        assert StuPASEFlowTrainModel.val_audio_stage == "flow"
        assert StuPASEFlowTrainModel.val_audio_vocoder_roles == ("vocoder",)

    def test_vocoder_stage_uses_raw_forward_output(self):
        from tasks.se.models.pase_family.unipase.gan_wrappers import (
            UniPASEVocoderGenerator,
        )

        gen = UniPASEVocoderGenerator.__new__(UniPASEVocoderGenerator)
        clean = torch.linspace(-0.1, 0.1, 32)
        esti = torch.linspace(-0.2, 0.2, 32)
        items = gen.val_audio_outputs(
            clean,
            clean,
            None,
            fs=16000,
            raw_enhanced=esti,
        )
        assert set(items) == {"true", "esti_epoch{epoch:03d}"}
        assert torch.equal(items["true"], clean)
        assert torch.equal(items["esti_epoch{epoch:03d}"], esti)


# ---------------------------------------------------------------------------
# Trainer 钩子接线（tiny stand-in，不依赖真实 checkpoint）
# ---------------------------------------------------------------------------
class _TinyValDataset(Dataset):
    def __init__(self, n: int = 2, length: int = 128) -> None:
        self.n = n
        self.length = length

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        torch.manual_seed(7 + idx)
        clean = torch.randn(self.length) * 0.3
        noisy = clean + torch.randn(self.length) * 0.1
        return (clean, noisy, 16000, torch.tensor(self.length))


def _collate(items):
    from core.data.collate import collate_fn

    return collate_fn(items)


class _TinyDistillValModel(nn.Module):
    val_audio_stage = "distill"
    val_audio_vocoder_roles = ()

    def __init__(self) -> None:
        super().__init__()
        self.student = nn.Linear(64, 32)

    def forward(self, noisy, clean, lengths, **kwargs):
        return F.mse_loss(self.student(noisy[:, :64]), self.student(clean[:, :64]))

    def val_audio_outputs(self, noisy, clean, lengths=None, **kwargs):
        return {
            "noisy": noisy,
            "clean": clean,
            "enh_epoch{epoch:03d}": noisy * 0.5,
            "ref_epoch{epoch:03d}": clean * 0.5,
        }


class _ScalarPassthroughLoss(nn.Module):
    def forward(self, enhanced, clean, lengths=None, **kwargs):
        return enhanced


def _build_dual_trainer(exp_dir: Path):
    from tasks.se.trainer.dual_input_loss_trainer import DualInputLossTrainer

    model = _TinyDistillValModel()
    train_loader = DataLoader(_TinyValDataset(4), batch_size=2, collate_fn=_collate)
    val_loader = DataLoader(_TinyValDataset(2), batch_size=2, collate_fn=_collate)
    return DualInputLossTrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        scheduler=None,
        loss_fn=_ScalarPassthroughLoss(),
        config={
            "trainer": {
                "epochs": 1,
                "exp_path": str(exp_dir),
                "val_metrics": {"enabled": False},
                "val_audio": {"enabled": True, "max_batches": 1},
            },
            "meta": {"seed": 42, "allow_no_stft": True},
        },
        train_loader=train_loader,
        val_loader=val_loader,
        sample_rate=16000,
    )


def test_dual_input_validation_writes_official_names(tmp_path):
    trainer = _build_dual_trainer(tmp_path / "exp")
    trainer._validation_epoch(epoch=3)
    root = tmp_path / "exp" / "samples"
    assert (root / "1_noisy.wav").exists()
    assert (root / "1_clean.wav").exists()
    assert (root / "1_enh_epoch003.wav").exists()
    assert (root / "1_ref_epoch003.wav").exists()


class _TinySubDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(4, 1, kernel_size=3, padding=1)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        f1 = torch.relu(self.conv1(x))
        out = self.conv2(f1)
        return out.flatten(1), [f1, out]


class _TinyDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discs = nn.ModuleList([_TinySubDiscriminator(), _TinySubDiscriminator()])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discs:
            yr, fr = d(y)
            yg, fg = d(y_hat)
            y_d_rs.append(yr)
            y_d_gs.append(yg)
            fmap_rs.append(fr)
            fmap_gs.append(fg)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class _TinyVocoGenerator(nn.Module):
    val_audio_stage = "vocoder"
    val_audio_vocoder_roles = ()

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)

    def forward(self, noisy, lengths, *, fs=None):
        return self.conv(noisy.unsqueeze(1)).squeeze(1)

    def val_audio_outputs(self, noisy, clean, lengths=None, **kwargs):
        return {
            "true": clean,
            "esti_epoch{epoch:03d}": self.forward(noisy, lengths),
        }


class _TinyReconLoss(nn.Module):
    def forward(self, enhanced, clean, lengths=None, **kwargs):
        return F.mse_loss(enhanced, clean)


def _build_gan_trainer(exp_dir: Path):
    from tasks.se.trainer.pase_gan_trainer import PASEGANTrainer

    gen = _TinyVocoGenerator()
    disc = _TinyDiscriminator()
    loader = DataLoader(_TinyValDataset(2), batch_size=1, collate_fn=_collate)
    return PASEGANTrainer(
        rank=0,
        world_size=1,
        model={"generator": gen, "discriminator": disc},
        optimizer={
            "generator": torch.optim.Adam(gen.parameters(), lr=1e-3),
            "discriminator": torch.optim.Adam(disc.parameters(), lr=1e-3),
        },
        scheduler=None,
        loss_fn=_TinyReconLoss(),
        train_loader=loader,
        val_loader=loader,
        config={
            "trainer": {
                "exp_path": str(exp_dir),
                "epochs": 1,
                "discriminator_steps": 1,
                "generator_steps": 1,
                "adversarial_weight": 1.0,
                "lamda_adv": 2.0,
                "lamda_feat": 1.0,
                "val_metrics": {"enabled": False},
                "val_audio": {"enabled": True, "max_batches": 1},
            },
            "meta": {"allow_no_stft": True},
        },
        sample_rate=16000,
        stft_module=None,
    )


def test_gan_validation_writes_vocoder_stage_names(tmp_path):
    trainer = _build_gan_trainer(tmp_path / "exp")
    trainer._validation_epoch(epoch=1)
    root = tmp_path / "exp" / "samples"
    assert (root / "1_true.wav").exists()
    assert (root / "1_esti_epoch001.wav").exists()


def test_gan_val_audio_rejects_unsupported_generator(tmp_path):
    from tasks.se.trainer.pase_gan_trainer import PASEGANTrainer

    class _NoStageGen(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)

        def forward(self, noisy, lengths):
            return self.conv(noisy.unsqueeze(1)).squeeze(1)

    gen = _NoStageGen()
    disc = _TinyDiscriminator()
    loader = DataLoader(_TinyValDataset(2), batch_size=1, collate_fn=_collate)
    with pytest.raises(ValueError, match="does not support trainer.val_audio"):
        PASEGANTrainer(
            rank=0,
            world_size=1,
            model={"generator": gen, "discriminator": disc},
            optimizer={
                "generator": torch.optim.Adam(gen.parameters(), lr=1e-3),
                "discriminator": torch.optim.Adam(disc.parameters(), lr=1e-3),
            },
            scheduler=None,
            loss_fn=_TinyReconLoss(),
            train_loader=loader,
            val_loader=loader,
            config={
                "trainer": {
                    "exp_path": str(tmp_path / "exp"),
                    "epochs": 1,
                    "discriminator_steps": 1,
                    "generator_steps": 1,
                    "adversarial_weight": 1.0,
                    "val_metrics": {"enabled": False},
                    "val_audio": {"enabled": True},
                },
                "meta": {"allow_no_stft": True},
            },
            sample_rate=16000,
            stft_module=None,
        )


def _build_tiny_flow_model():
    from tasks.se.models.pase_family.common.wavlm.WavLM import WavLM
    from tasks.se.models.pase_family.stupase.flow_train_wrapper import (
        StuPASEFlowTrainModel,
    )

    model = StuPASEFlowTrainModel(
        encoder_output_layer=1,
        cfm_kwargs={
            "dit_kwargs": dict(
                dim=32,
                depth=1,
                heads=2,
                dim_head=16,
                ff_mult=2,
                mel_dim=20,
                text_in_dim=32,
                text_dim=16,
                conv_layers=1,
            ),
            "mel_spec_kwargs": dict(
                n_fft=256,
                hop_length=64,
                win_length=256,
                n_mel_channels=20,
                target_sample_rate=16000,
            ),
        },
    )
    cfg = model.encoder.cfg
    cfg.encoder_layers = 2
    cfg.encoder_embed_dim = 32
    cfg.encoder_ffn_embed_dim = 64
    cfg.encoder_attention_heads = 2
    model.encoder.wavlm = WavLM(cfg)
    for p in model.encoder.wavlm.parameters():
        p.requires_grad = False
    return model


def test_flow_validation_writes_peak_normalized_samples(tmp_path, monkeypatch):
    from tasks.se.models.pase_family.stupase.vocos_vocoder import VocosVocoder
    from tasks.se.trainer.dual_input_loss_trainer import DualInputLossTrainer

    def _fake_vocoder_loader(kind, ckpt_path, *, device):
        module = VocosVocoder(
            input_channels=20,
            dim=32,
            intermediate_dim=64,
            num_layers=1,
            n_fft=256,
            hop_length=64,
        )
        module.eval()
        return module.to(device)

    monkeypatch.setattr(
        "tasks.se.trainer.val_audio.load_published_vocoder", _fake_vocoder_loader
    )

    model = _build_tiny_flow_model()
    train_loader = DataLoader(
        _TinyValDataset(2, length=4000), batch_size=1, collate_fn=_collate
    )
    trainer = DualInputLossTrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=torch.optim.Adam(model.cfm.parameters(), lr=1e-3),
        scheduler=None,
        loss_fn=_ScalarPassthroughLoss(),
        config={
            "trainer": {
                "epochs": 1,
                "exp_path": str(tmp_path / "exp"),
                "val_metrics": {"enabled": False},
                "val_audio": {
                    "enabled": True,
                    "max_batches": 1,
                    "vocoder_ckpt_path": "unused.pt",
                    "vocoder_kind": "stupase",
                    "sample_settings": {
                        "steps": 2,
                        "cfg_strength": 0.5,
                        "sway_sampling_coef": -1.0,
                    },
                },
            },
            "meta": {"seed": 42, "allow_no_stft": True},
        },
        train_loader=train_loader,
        val_loader=train_loader,
        sample_rate=16000,
    )
    trainer._validation_epoch(epoch=5)
    root = tmp_path / "exp" / "samples"
    for name in ("1_noisy.wav", "1_clean.wav", "1_recon.wav", "1_enh_epoch005.wav"):
        assert (root / name).exists(), name
    for name in ("1_recon.wav", "1_enh_epoch005.wav"):
        data, sr = sf.read(root / name, dtype="float32")
        assert sr == 16000
        assert float(np.abs(data).max()) == pytest.approx(0.9, abs=2e-3)


# ---------------------------------------------------------------------------
# 模型侧 val_audio_outputs：用轻量 stub 验证官方中间量与布局，不加载大权重
# ---------------------------------------------------------------------------
class TestModelValAudioOutputs:
    def test_distill_outputs_use_student_features_and_vocoder(self):
        from types import SimpleNamespace

        from tasks.se.models.pase_family.common.dewavlm_distill import (
            DeWavLMDistillModel,
        )

        masks: list[bool] = []

        def student(wav, sr=None, mask=False, lengths=None):
            masks.append(bool(mask))
            return wav.unsqueeze(-1).expand(-1, -1, 8)

        def vocoder(feat):
            assert feat.shape[0] == 2 and feat.shape[1] == 8
            return feat.mean(dim=1)

        stub = SimpleNamespace(
            student=student, use_plc_frontend=True, val_audio_sample_rate=16000
        )
        noisy = torch.randn(2, 16)
        clean = torch.randn(2, 16)
        items = DeWavLMDistillModel.val_audio_outputs(
            stub,
            noisy,
            clean,
            None,
            fs=16000,
            vocoders={"vocoder": vocoder},
        )
        assert masks == [True, False]
        assert set(items) == {
            "noisy",
            "clean",
            "enh_epoch{epoch:03d}",
            "ref_epoch{epoch:03d}",
        }
        assert items["enh_epoch{epoch:03d}"].shape == (2, 16)

    def test_adapter_outputs_reconstruct_three_paths(self):
        from types import SimpleNamespace

        from tasks.se.models.pase_family.unipase.gan_wrappers import (
            UniPASEAdapterGenerator,
        )

        def encoder(wav, sr=None, mask=False, lengths=None):
            feat = wav.unsqueeze(-1).expand(-1, -1, 4)
            return feat, feat

        def decoder(feat):
            assert feat.shape[1] == 4
            return feat.mean(dim=1)

        stub = SimpleNamespace(
            enable_plc=True,
            decoder_ckpt_path="",
            wavlm_decoder_ckpt_path="",
            _val_vocoders={},
            encoder=encoder,
            val_audio_sample_rate=16000,
        )
        items = UniPASEAdapterGenerator.val_audio_outputs(
            stub,
            torch.randn(2, 16),
            torch.randn(2, 16),
            None,
            fs=16000,
            vocoders={"vocoder": decoder, "wavlm_vocoder": decoder},
            forward_dict={
                "target_feat": torch.randn(2, 16, 4),
                "enhanced": torch.randn(2, 16, 4),
            },
        )
        assert set(items) == {
            "noisy",
            "clean",
            "recon",
            "enh0",
            "enh_epoch{epoch:03d}",
        }
        for key in ("recon", "enh0", "enh_epoch{epoch:03d}"):
            assert items[key].shape == (2, 16)
            assert float(items[key].abs().amax()) == pytest.approx(0.9, abs=1e-5)

    def test_postnet_outputs_include_resampled_16k_branch(self):
        from types import SimpleNamespace

        from tasks.se.models.pase_family.unipase.gan_wrappers import (
            UniPASEPostNetGenerator,
        )

        def prefix(noisy, lengths, fs):
            return noisy, lengths, torch.ones(2, 16), torch.ones(2, 48)

        stub = SimpleNamespace(_frozen_prefix=prefix)
        items = UniPASEPostNetGenerator.val_audio_outputs(
            stub,
            torch.randn(2, 48),
            torch.randn(2, 48),
            None,
            fs=48000,
            raw_enhanced=torch.full((2, 48), 0.5),
        )
        assert set(items) == {"noisy", "clean", "enh_16k", "enh_epoch{epoch:03d}"}
        assert items["enh_16k"].shape == (2, 48)
        assert items["enh_epoch{epoch:03d}"].shape == (2, 48)

    def test_gap_outputs_include_pred_and_gen_branches(self):
        from types import SimpleNamespace

        from tasks.se.models.pase_family.gap_urgenet.gan_wrappers import (
            GAPURGENetPostNetGenerator,
        )

        def prefix(noisy, lengths, fs):
            return (
                noisy,
                lengths,
                torch.zeros(2, 16),
                torch.ones(2, 48),
                torch.full((2, 48), 2.0),
                torch.stack(
                    [torch.full((2, 48), 2.0), torch.ones(2, 48)], dim=1
                ),
            )

        stub = SimpleNamespace(_frozen_prefix=prefix)
        items = GAPURGENetPostNetGenerator.val_audio_outputs(
            stub,
            torch.randn(2, 48),
            torch.randn(2, 48),
            None,
            fs=48000,
            raw_enhanced=torch.full((2, 48), 0.25),
        )
        assert set(items) == {
            "noisy",
            "clean",
            "enh_pred_16k",
            "enh_gen_16k",
            "enh_epoch{epoch:03d}",
        }
        np.testing.assert_allclose(items["enh_pred_16k"].numpy(), 2.0, atol=1e-6)
        np.testing.assert_allclose(items["enh_gen_16k"].numpy(), 1.0, atol=1e-6)


def test_pase_p1_and_p3_outputs(tmp_path):
    from types import SimpleNamespace

    from tasks.se.models.pase_family.pase.model import PASE

    noisy = torch.linspace(-0.5, 0.5, 32)
    clean = noisy * 0.5
    enhanced = noisy * 0.9
    single = SimpleNamespace(use_dual_stream=False, val_audio_sample_rate=16000)
    dual = SimpleNamespace(use_dual_stream=True, val_audio_sample_rate=16000)

    p1 = PASE.val_audio_outputs(
        single, noisy, clean, None, fs=16000, raw_enhanced=enhanced
    )
    assert set(p1) == {"true", "esti_epoch{epoch:03d}"}
    assert torch.equal(p1["true"], clean)

    p3 = PASE.val_audio_outputs(
        dual, noisy, clean, None, fs=16000, raw_enhanced=enhanced
    )
    assert set(p3) == {"noisy", "clean", "enh_epoch{epoch:03d}"}


def test_gap_predictor_outputs_share_global_scale():
    from tasks.se.models.pase_family.gap_urgenet.gan_wrappers import (
        GAPURGENetPredictor,
    )

    class _TinyPredictor(nn.Module):
        def forward(self, wav, lengths=None, *, fs=None):
            return wav * 2.0

    model = _TinyPredictor()
    noisy = torch.full((1, 16), 0.25)
    clean = torch.full((1, 16), 0.5)
    items = GAPURGENetPredictor.val_audio_outputs(
        model, noisy, clean, None, fs=16000
    )
    assert set(items) == {"noisy", "clean", "enh_epoch{epoch:03d}"}
    peak = torch.stack([items["noisy"].abs().amax(), items["clean"].abs().amax(), items["enh_epoch{epoch:03d}"].abs().amax()]).amax()
    assert float(peak) == pytest.approx(0.9, abs=1e-6)


class _TinyPredictorValModel(nn.Module):
    val_audio_stage = "predictor"
    val_audio_vocoder_roles = ()

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)

    def forward(self, noisy, lengths=None, *, fs=None):
        return self.conv(noisy.unsqueeze(1)).squeeze(1)

    def val_audio_outputs(self, noisy, clean, lengths=None, *, fs=None, **kwargs):
        enhanced = self(noisy, lengths, fs=fs)
        peak = torch.stack(
            [noisy.abs().amax(), clean.abs().amax(), enhanced.abs().amax()]
        ).amax()
        scale = 0.9 / (peak + 1e-12)
        return {
            "noisy": noisy * scale,
            "clean": clean * scale,
            "enh_epoch{epoch:03d}": enhanced * scale,
        }


class _MseLoss(nn.Module):
    def forward(self, enhanced, clean, lengths=None, **kwargs):
        return F.mse_loss(enhanced, clean)


def test_set_trainer_validation_writes_predictor_audio(tmp_path):
    from tasks.se.trainer.se_trainer import SETrainer

    model = _TinyPredictorValModel()
    train_loader = DataLoader(_TinyValDataset(4), batch_size=2, collate_fn=_collate)
    val_loader = DataLoader(_TinyValDataset(2), batch_size=2, collate_fn=_collate)
    trainer = SETrainer(
        rank=0,
        world_size=1,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        scheduler=None,
        loss_fn=_MseLoss(),
        config={
            "trainer": {
                "epochs": 1,
                "exp_path": str(tmp_path / "exp"),
                "val_metrics": {"enabled": False},
                "val_audio": {"enabled": True, "max_batches": 1},
            },
            "meta": {"seed": 42, "allow_no_stft": True},
        },
        sample_rate=16000,
    )
    trainer._validation_epoch(epoch=3)
    root = tmp_path / "exp" / "samples"
    assert (root / "1_noisy.wav").exists()
    assert (root / "1_clean.wav").exists()
    assert (root / "1_enh_epoch003.wav").exists()


def test_multi_rate_audio_outputs_use_official_rate():
    """多采样率 batch：16k 阶段必须把 noisy/clean 重采样到 16k 再落盘。"""
    from types import SimpleNamespace

    from tasks.se.models.pase_family.common.dewavlm_distill import (
        DeWavLMDistillModel,
    )
    from tasks.se.models.pase_family.pase.model import PASE

    def student(wav, sr=None, mask=False, lengths=None):
        return wav.unsqueeze(-1).expand(-1, -1, 4)

    def vocoder(feat):
        return torch.zeros(feat.shape[0], 16)

    distill = SimpleNamespace(
        student=student, use_plc_frontend=False, val_audio_sample_rate=16000
    )
    items = DeWavLMDistillModel.val_audio_outputs(
        distill,
        torch.zeros(1, 48),
        torch.zeros(1, 48),
        None,
        fs=48000,
        vocoders={"vocoder": vocoder},
    )
    assert items["noisy"].shape[-1] == 16
    assert items["clean"].shape[-1] == 16

    dual = SimpleNamespace(use_dual_stream=True, val_audio_sample_rate=16000)
    p3 = PASE.val_audio_outputs(
        dual,
        torch.zeros(1, 48),
        torch.zeros(1, 48),
        None,
        fs=48000,
        raw_enhanced=torch.zeros(1, 16),
    )
    assert p3["noisy"].shape[-1] == 16
    assert p3["clean"].shape[-1] == 16
    assert p3["enh_epoch{epoch:03d}"].shape[-1] == 16


def test_postnet_val_audio_uses_forward_dict_enhanced():
    """固定输出率阶段：raw_enhanced 缺失时从 validation forward_dict 取输出。"""
    from types import SimpleNamespace

    from tasks.se.models.pase_family.unipase.gan_wrappers import (
        UniPASEPostNetGenerator,
    )

    calls = []

    def prefix(noisy, lengths, fs):
        calls.append(int(fs))
        return (
            torch.zeros(1, 48),
            lengths,
            torch.zeros(1, 16),
            torch.zeros(1, 48),
        )

    stub = SimpleNamespace(_frozen_prefix=prefix)
    items = UniPASEPostNetGenerator.val_audio_outputs(
        stub,
        torch.zeros(1, 48),
        torch.zeros(1, 48),
        None,
        fs=48000,
        forward_dict={"enhanced": torch.full((1, 48), 0.5)},
    )
    assert calls == [48000]
    assert items["enh_16k"].shape[-1] == 48
    np.testing.assert_allclose(
        items["enh_epoch{epoch:03d}"].numpy(), 0.5, atol=1e-6
    )


def test_stupase_p1_outputs_true_and_esti():
    from types import SimpleNamespace

    from tasks.se.models.pase_family.stupase.mel_gan_wrapper import (
        StuPASEMelVocoderGenerator,
    )

    stub = SimpleNamespace(val_audio_sample_rate=16000)
    clean = torch.linspace(-0.5, 0.5, 32)
    esti = torch.linspace(-0.25, 0.25, 32)
    items = StuPASEMelVocoderGenerator.val_audio_outputs(
        stub,
        clean,
        clean,
        None,
        fs=16000,
        raw_enhanced=esti,
    )
    assert set(items) == {"true", "esti_epoch{epoch:03d}"}
    assert torch.equal(items["true"], clean)
    assert torch.equal(items["esti_epoch{epoch:03d}"], esti)


def test_frozen_postnet_wrappers_keep_frozen_branches_eval():
    from tasks.se.models.pase_family.gap_urgenet.gan_wrappers import (
        GAPURGENetPostNetGenerator,
    )
    from tasks.se.models.pase_family.unipase.gan_wrappers import (
        UniPASEPostNetGenerator,
    )

    class _Stub:
        pass

    for cls in (UniPASEPostNetGenerator, GAPURGENetPostNetGenerator):
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.adapter = nn.Dropout(0.1)
        model.decoder = nn.Dropout(0.1)
        model.predictor = nn.Dropout(0.1)
        model.postnet = nn.Linear(2, 2)
        model.train()
        assert model.training is True
        assert model.postnet.training is True
        assert model.adapter.training is False
        assert model.decoder.training is False
        if cls is GAPURGENetPostNetGenerator:
            assert model.predictor.training is False


def test_output_clamp_can_be_disabled_for_pase_family():
    from core.trainer.metrics_trainer import MetricsCapableTrainer

    trainer = MetricsCapableTrainer.__new__(MetricsCapableTrainer)
    trainer.output_clamp = False
    trainer._accumulate_output_sanitize = lambda *args: None
    out = MetricsCapableTrainer._sanitize_model_output(
        trainer, torch.tensor([2.0, -3.0, float("nan")])
    )
    assert out[0].item() == 2.0
    assert out[1].item() == -3.0
    assert out[2].item() == 0.0
