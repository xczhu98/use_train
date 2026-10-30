"""TSE 训练期 speaker 句级 head 的 per-rank batch 门禁契约测试。

覆盖 ``tasks/tse/trainer/entry.py::_validate_tse_speaker_train_batch``：
joint 训练 + effective ``spkemb`` + 未冻结的句级 BatchNorm speaker encoder 时，
per-rank ``data.dataloader.batch_size`` 必须 >= 2，否则在校验边界 fail-fast，
而不是在 ECAPA 内部抛不可读的 BatchNorm 报错。
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import yaml

from tasks.tse.models.speaker.config import resolve_joint_sentence_batch_encoder
from tasks.tse.models.speaker.encoder import (
    SpeakerEncoder,
    train_sentence_batch_requires_two,
)
from tasks.tse.trainer.entry import TSETrainingEntry

pytestmark = pytest.mark.unit

_ECAPA = "ECAPA_TDNN_GLOB_c512"


def _offline_tse_cfg(*, batch_size: int = 1, **model_args) -> dict:
    """能通过 TSE entry 配置校验的 offline_tse 最小配置（train/val 均离线）。"""
    split = {
        "samples_jsonl": "placeholder/samples.jsonl",
        "cues_yaml": "placeholder/cues.yaml",
        "enabled_cues": ["audio"],
    }
    return {
        "data": {
            "mode": "offline_tse",
            "sample_rate": 16000,
            "train": dict(split),
            "validation": dict(split),
            "dataloader": {"batch_size": batch_size, "num_workers": 0},
        },
        "model": {"type": "TSE_BSRNN_SPK", "args": dict(model_args)},
        "optimizer": {"type": "Adam"},
        "scheduler": {"type": "ExponentialLR"},
        "trainer": {"type": "TSETrainer"},
        "loss": {
            "type": "CompositeLoss",
            "components": [{"type": "SISNRLoss", "weight": 1.0}],
        },
        "stft": {"n_fft": 512, "hop_length": 128, "win_length": 512},
    }


def _joint_spkemb(**speaker_encoder) -> dict:
    args = {
        "joint_training": True,
        "features": {"spkemb": {"enabled": True}},
    }
    if speaker_encoder:
        args["speaker_model"] = {"speaker_encoder": dict(speaker_encoder)}
    return args


class TestConfigBoundaryGate:
    def test_train_mode_batch1_sentence_head_is_the_reason_for_the_gate(self):
        """门禁根因：train 模式句级 ``BatchNorm1d`` 在 batch=1 时直接报错。"""
        encoder = SpeakerEncoder(model=_ECAPA, freeze=False).train()

        with pytest.raises(ValueError, match="more than 1 value per channel"):
            encoder(torch.randn(1, 16000))

    @pytest.mark.parametrize("with_distributed", [False, True])
    def test_batch1_joint_spkemb_trainable_ecapa_is_rejected(self, with_distributed):
        cfg = _offline_tse_cfg(**_joint_spkemb())
        if with_distributed:
            cfg["distributed"] = {"backend": "gloo", "find_unused_parameters": False}

        with pytest.raises(ValueError) as exc_info:
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

        message = str(exc_info.value)
        assert _ECAPA in message
        assert "features.spkemb" in message
        assert "batch_size=1" in message
        assert "per-rank" in message
        assert "world_size * batch_size" in message
        assert "batch_size >= 2" in message

    def test_error_message_carries_all_remediation_paths(self):
        with pytest.raises(ValueError) as exc_info:
            TSETrainingEntry("TSE_Test")._validate_config(
                _offline_tse_cfg(**_joint_spkemb())
            )

        message = str(exc_info.value)
        assert "data.dataloader.batch_size >= 2" in message
        assert "speaker_encoder.pretrained" in message
        assert "joint_training=false" in message
        assert "ecapa.py" in message

    def test_batch2_is_accepted(self):
        TSETrainingEntry("TSE_Test")._validate_config(
            _offline_tse_cfg(batch_size=2, **_joint_spkemb())
        )

    def test_batch1_with_frozen_encoder_is_accepted(self):
        """freeze: null + pretrained 自动冻结；句级 head 恒为 eval。"""
        TSETrainingEntry("TSE_Test")._validate_config(
            _offline_tse_cfg(**_joint_spkemb(pretrained="placeholder/avg_model.pt"))
        )

    def test_batch1_with_explicit_freeze_true_is_accepted(self):
        TSETrainingEntry("TSE_Test")._validate_config(
            _offline_tse_cfg(**_joint_spkemb(freeze=True))
        )

    def test_batch1_with_explicit_freeze_false_is_rejected(self):
        """显式解冻即使有 pretrained 也进入 train 模式，仍要求 batch >= 2。"""
        with pytest.raises(ValueError, match="per-rank"):
            TSETrainingEntry("TSE_Test")._validate_config(
                _offline_tse_cfg(
                    **_joint_spkemb(
                        pretrained="placeholder/avg_model.pt", freeze=False
                    )
                )
            )

    def test_batch1_in_preextracted_mode_is_accepted(self):
        """joint_training=false 走 trainer 外部冻结 encoder，不受约束。"""
        TSETrainingEntry("TSE_Test")._validate_config(
            _offline_tse_cfg(
                joint_training=False, features={"spkemb": {"enabled": True}}
            )
        )

    @pytest.mark.parametrize(
        "features",
        [
            {},
            {"spkemb": {"enabled": False}},
            {"context": {"enabled": True}},
            {"listen": {"enabled": True}, "usef": {"enabled": True}},
            {"tfmap": {"enabled": True, "type": "embedding"}},
        ],
    )
    def test_batch1_without_sentence_cue_is_accepted(self, features):
        """纯 BSRNN / 帧级 cue 不消费句级 head，batch=1 合法。"""
        TSETrainingEntry("TSE_Test")._validate_config(
            _offline_tse_cfg(joint_training=True, features=features)
        )

    def test_batch1_resnet34_default_is_accepted(self):
        """ResNet34 默认 two_emb_layer=False，句级路径无 BatchNorm1d。"""
        TSETrainingEntry("TSE_Test")._validate_config(
            _offline_tse_cfg(
                **_joint_spkemb(model="ResNet34", spk_args={"embed_dim": 256})
            )
        )

    def test_batch1_resnet34_two_emb_layer_is_rejected(self):
        with pytest.raises(ValueError, match="ResNet34"):
            TSETrainingEntry("TSE_Test")._validate_config(
                _offline_tse_cfg(
                    **_joint_spkemb(
                        model="ResNet34",
                        spk_args={"embed_dim": 256, "two_emb_layer": True},
                    )
                )
            )


class TestResolverContract:
    def test_ecapa_always_requires_two(self):
        assert train_sentence_batch_requires_two(_ECAPA) is True
        assert train_sentence_batch_requires_two(_ECAPA, {"emb_bn": True}) is True

    def test_resnet34_only_with_two_emb_layer(self):
        assert train_sentence_batch_requires_two("ResNet34") is False
        assert (
            train_sentence_batch_requires_two("ResNet34", {"two_emb_layer": True})
            is True
        )

    def test_unknown_model_fails_fast(self):
        with pytest.raises(ValueError, match="Unknown speaker model"):
            train_sentence_batch_requires_two("NotASpeakerModel")

    def test_resolver_returns_encoder_only_for_train_mode_sentence_head(self):
        assert (
            resolve_joint_sentence_batch_encoder(_joint_spkemb()) == _ECAPA
        )
        assert resolve_joint_sentence_batch_encoder(
            _joint_spkemb(pretrained="placeholder/avg_model.pt")
        ) is None
        assert resolve_joint_sentence_batch_encoder(
            {"joint_training": False, "features": {"spkemb": {"enabled": True}}}
        ) is None
        assert resolve_joint_sentence_batch_encoder(
            {"joint_training": True, "features": {"spkemb": {"enabled": False}}}
        ) is None

    def test_resolver_rejects_non_mapping_sections(self):
        with pytest.raises(TypeError, match=r"model\.args\.features must be a mapping"):
            resolve_joint_sentence_batch_encoder(
                {"joint_training": True, "features": []}
            )
        with pytest.raises(
            TypeError, match=r"speaker_model\.speaker_encoder must be a mapping"
        ):
            resolve_joint_sentence_batch_encoder(
                {
                    "joint_training": True,
                    "features": {"spkemb": {"enabled": True}},
                    "speaker_model": {"speaker_encoder": []},
                }
            )


class TestEvalAndInferencePath:
    def test_eval_mode_batch1_sentence_embedding_still_works(self):
        """推理/评估路径：eval 模式 BatchNorm 用 running stats，batch=1 可用。"""
        encoder = SpeakerEncoder(model=_ECAPA, freeze=False).eval()

        with torch.no_grad():
            embedding = encoder(torch.randn(1, 16000))

        assert tuple(embedding.shape) == (1, 192)

    def test_inference_style_frozen_encoder_batch1_works(self):
        """推理侧固定 freeze=True + eval 的构造方式不受训练门禁影响。"""
        encoder = SpeakerEncoder(model=_ECAPA, freeze=True)
        encoder.eval()

        with torch.no_grad():
            embedding = encoder(torch.randn(1, 16000))

        assert tuple(embedding.shape) == (1, 192)

    def test_training_gate_does_not_run_on_inference_imports(self):
        """门禁只存在于训练 entry 的配置校验，推理模块不引用它。"""
        import inspect

        import evaluate.eval_pipeline_tse as eval_pipeline
        import tasks.tse.inference.extractor as extractor

        for module in (eval_pipeline, extractor):
            source = inspect.getsource(module)
            assert "resolve_joint_sentence_batch_encoder" not in source
            assert "_validate_tse_speaker_train_batch" not in source


class TestRunStopsBeforeLoaders:
    def test_run_rejects_batch1_before_building_loaders(self, tmp_path, monkeypatch, caplog):
        """bootstrap 阶段 fail-fast：不构建 dataloader、不构建模型、不启动训练。"""
        cfg = _offline_tse_cfg(**_joint_spkemb())
        config_path = tmp_path / "tse_batch1.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        entry = TSETrainingEntry("TSE_Test")
        args = Namespace(
            config=str(config_path),
            log_level="INFO",
            log_file=None,
            strict=False,
            max_steps=None,
            exp_path=None,
        )
        monkeypatch.setattr(entry, "parse_args", lambda: args)
        monkeypatch.setattr(entry, "_ensure_free_master_port", lambda: None)
        monkeypatch.setattr(
            "core.trainer.entry._ensure_file_descriptor_limit", lambda **_kwargs: None
        )
        monkeypatch.setattr(
            "core.trainer.entry.detect_task",
            lambda _cfg: SimpleNamespace(
                task="tse",
                subtask="offline_tse",
                paradigm="discriminative",
                model_type="TSE_BSRNN_SPK",
                has_laec=False,
            ),
        )
        monkeypatch.setattr(entry, "_log_task_info", lambda _task_info: None)
        started: list[str] = []
        monkeypatch.setattr(
            entry, "build_loaders", lambda *_a, **_k: started.append("loaders")
        )
        monkeypatch.setattr(
            entry,
            "build_model_for_training",
            lambda *_a, **_k: started.append("model"),
        )

        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit) as exc_info:
                entry.run(config=cfg, config_path=config_path)

        assert exc_info.value.code == 1
        assert started == []
        assert any(
            "per-rank" in record.getMessage() and "batch_size=1" in record.getMessage()
            for record in caplog.records
        )
