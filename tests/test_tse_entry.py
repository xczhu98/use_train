"""Tests for TSE training entry (tasks/tse/trainer/entry.py + core detect_task 路由)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.trainer.entry import detect_task
from tasks.tse.cue_contract import TSECueSpecs
from tasks.tse.trainer.entry import TSETrainingEntry


def _minimal_tse_cfg() -> dict:
    """能通过 UnifiedTrainingEntry._validate_config 基类校验的最小 TSE 配置。"""
    return {
        "data": {
            "mode": "dynamic_tse",
            "train": {
                "speech_scp": "placeholder/train.jsonl",
                "spk2enroll_json": "placeholder/spk2enroll.json",
                "simulation_cfg": {
                    "prob_target_only": 1.0,
                }
            },
            "validation": {
                "mode": "dynamic_tse",
                "inherit_simulation": True,
            },
            "dataloader": {"batch_size": 1, "num_workers": 0},
        },
        "model": {"type": "TSE_BSRNN_SPK"},
        "optimizer": {"type": "Adam"},
        "scheduler": {"type": "LinearWarmupCosineAnnealing"},
        "trainer": {"type": "TSETrainer"},
        "loss": {"type": "CompositeLoss"},
        "stft": {"n_fft": 512, "hop_length": 128, "win_length": 512},
    }


class TestDetectTask:
    def test_dynamic_tse_routes_to_tse(self):
        cfg = detect_task(_minimal_tse_cfg())
        assert cfg.task == "tse"
        assert cfg.subtask == "dynamic_tse"
        assert cfg.paradigm == "discriminative"
        assert cfg.model_type == "TSE_BSRNN_SPK"
        assert cfg.logger_name == "TSE_Train"

    def test_se_and_aec_routing_unaffected(self):
        """TSE 分支为纯追加：dynamic / dynamic_aec 判定不变。"""
        se_cfg = detect_task({"data": {"mode": "dynamic"}, "model": {"type": "GTCRN"}})
        assert se_cfg.task == "se"
        aec_cfg = detect_task(
            {"data": {"mode": "dynamic_aec"}, "model": {"type": "DeepVQE"}}
        )
        assert aec_cfg.task == "aec"


class TestValidateConfig:
    def test_valid_config_passes(self):
        TSETrainingEntry("TSE_Test")._validate_config(_minimal_tse_cfg())

    def test_evaluation_only_config_is_rejected_by_training_entry(self):
        cfg = _minimal_tse_cfg()
        cfg["meta"] = {"evaluation_only": True}
        with pytest.raises(ValueError, match="evaluation_only.*不能用于训练"):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    @pytest.mark.parametrize(
        ("config_path", "invalid_value"),
        [
            ("model", "not-a-mapping"),
            ("model", None),
            ("model.args", []),
            ("model.args", "not-a-mapping"),
            ("model.args.speaker_model", []),
            ("model.args.speaker_model", "not-a-mapping"),
            ("model.args.speaker_model.fbank", []),
            ("model.args.speaker_model.fbank", "not-a-mapping"),
            ("model.args.speaker_model.speaker_encoder", []),
            ("model.args.speaker_model.speaker_encoder", "not-a-mapping"),
        ],
    )
    def test_rejects_non_mapping_model_sections_with_full_path(
        self, config_path, invalid_value
    ):
        cfg = _minimal_tse_cfg()
        if config_path == "model":
            cfg["model"] = invalid_value
        else:
            cfg["model"]["args"] = {
                "speaker_model": {
                    "fbank": {},
                    "speaker_encoder": {},
                }
            }
            target = cfg
            for key in config_path.split(".")[:-1]:
                target = target[key]
            target[config_path.rsplit(".", 1)[-1]] = invalid_value

        escaped_config_path = config_path.replace('.', r"\.")
        with pytest.raises(
            TypeError,
            match=rf"{escaped_config_path} must be a mapping",
        ):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    @pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
    def test_rejects_non_boolean_joint_training(self, value):
        cfg = _minimal_tse_cfg()
        cfg["model"]["args"] = {"joint_training": value}

        with pytest.raises(TypeError, match=r"model\.args\.joint_training.*bool"):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    @pytest.mark.parametrize("feature", ["listen", "usef", "tfmap", "context", "spkemb"])
    @pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
    def test_rejects_non_boolean_feature_enabled(self, feature, value):
        cfg = _minimal_tse_cfg()
        cfg["model"]["args"] = {
            "joint_training": True,
            "features": {feature: {"enabled": value}},
        }

        with pytest.raises(
            TypeError,
            match=rf"model\.args\.features\.{feature}\.enabled.*bool",
        ):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_rejects_non_mapping_feature_config(self):
        cfg = _minimal_tse_cfg()
        cfg["model"]["args"] = {
            "joint_training": True,
            "features": {"spkemb": "false"},
        }

        with pytest.raises(TypeError, match=r"features\.spkemb.*mapping"):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_rejects_non_mapping_features(self):
        cfg = _minimal_tse_cfg()
        cfg["model"]["args"] = {
            "joint_training": True,
            "features": "false",
        }

        with pytest.raises(TypeError, match=r"model\.args\.features.*mapping"):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_rejects_data_and_model_sample_rate_mismatch(self):
        cfg = _minimal_tse_cfg()
        cfg["data"]["sample_rate"] = 8000
        cfg["model"]["args"] = {"sr": 16000}

        with pytest.raises(ValueError, match=r"data\.sample_rate.*model\.args\.sr"):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    @pytest.mark.parametrize(
        ("path", "value", "error", "match"),
        [
            ("data", 16000.5, TypeError, r"data\.sample_rate"),
            ("model", True, TypeError, r"model\.args\.sr"),
            ("fbank", 0, ValueError, r"fbank\.sample_rate"),
        ],
    )
    def test_rejects_invalid_sample_rate_types(self, path, value, error, match):
        cfg = _minimal_tse_cfg()
        cfg["data"]["sample_rate"] = 16000
        cfg["model"]["args"] = {
            "sr": 16000,
            "speaker_model": {"fbank": {"sample_rate": 16000}},
        }
        if path == "data":
            cfg["data"]["sample_rate"] = value
        elif path == "model":
            cfg["model"]["args"]["sr"] = value
        else:
            cfg["model"]["args"]["speaker_model"]["fbank"]["sample_rate"] = value

        with pytest.raises(error, match=match):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_rejects_speaker_fbank_sample_rate_mismatch(self):
        cfg = _minimal_tse_cfg()
        cfg["data"]["sample_rate"] = 16000
        cfg["model"]["args"] = {
            "sr": 16000,
            "speaker_model": {"fbank": {"sample_rate": 8000}},
        }

        with pytest.raises(ValueError, match=r"fbank\.sample_rate.*model\.args\.sr"):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    @pytest.mark.parametrize("cue", [None, "listen", "usef", "tfmap"])
    def test_preextracted_mode_accepts_non_context_cues(self, cue):
        cfg = _minimal_tse_cfg()
        features = {} if cue is None else {cue: {"enabled": True}}
        cfg["model"]["args"] = {
            "joint_training": False,
            "features": features,
        }

        TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_preextracted_mode_rejects_context(self):
        cfg = _minimal_tse_cfg()
        cfg["model"]["args"] = {
            "joint_training": False,
            "features": {"context": {"enabled": True}},
        }

        with pytest.raises(ValueError, match=r"joint_training=False.*context"):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_preextracted_spkemb_accepts_existing_pretrained_file(self, tmp_path):
        pretrained = tmp_path / "speaker.pt"
        pretrained.touch()
        cfg = _minimal_tse_cfg()
        cfg["model"]["args"] = {
            "joint_training": False,
            "features": {"spkemb": {"enabled": True}},
            "speaker_model": {
                "speaker_encoder": {"pretrained": str(pretrained)},
            },
        }

        TSETrainingEntry("TSE_Test")._validate_config(cfg)

    @pytest.mark.parametrize(
        "pretrained",
        [None, "", "   ", 123, "missing.pt"],
    )
    def test_preextracted_spkemb_defers_checkpoint_to_dataset_source(
        self, pretrained
    ):
        cfg = _minimal_tse_cfg()
        cfg["model"]["args"] = {
            "joint_training": False,
            "features": {"spkemb": {"enabled": True}},
            "speaker_model": {
                "speaker_encoder": {"pretrained": pretrained},
            },
        }

        TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_joint_mode_keeps_existing_pretrained_semantics(self):
        cfg = _minimal_tse_cfg()
        # pretrained=None 表示不冻结内置 encoder，句级 ECAPA head 因此要求
        # per-rank batch >= 2；本用例只验证 pretrained 语义，故用 batch=2。
        cfg["data"]["dataloader"]["batch_size"] = 2
        cfg["model"]["args"] = {
            "joint_training": True,
            "features": {"spkemb": {"enabled": True}},
            "speaker_model": {
                "speaker_encoder": {"pretrained": None},
            },
        }

        TSETrainingEntry("TSE_Test")._validate_config(cfg)

    @pytest.mark.parametrize(
        "invalid_speech", [None, "", "   ", " train.jsonl", "train.jsonl ", 123]
    )
    def test_runtime_rejects_invalid_train_speech_scp(self, invalid_speech):
        cfg = _minimal_tse_cfg()
        cfg["data"]["train"]["speech_scp"] = invalid_speech

        with pytest.raises(ValueError, match=r"data\.train\.speech_scp"):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_runtime_rejects_missing_train_speech_scp(self):
        from tasks.tse.data import build_dataset as build_tse_dataset

        cfg = _minimal_tse_cfg()
        del cfg["data"]["train"]["speech_scp"]

        for validate in (
            lambda: TSETrainingEntry("TSE_Test")._validate_config(cfg),
            lambda: build_tse_dataset(cfg["data"], mode="train"),
        ):
            with pytest.raises(ValueError, match=r"data\.train\.speech_scp"):
                validate()

    def test_missing_spk2enroll_json_raises(self):
        cfg = _minimal_tse_cfg()
        del cfg["data"]["train"]["spk2enroll_json"]
        with pytest.raises(
            ValueError,
            match=r"data\.train\.spk2enroll_json",
        ):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_missing_simulation_cfg_raises(self):
        cfg = _minimal_tse_cfg()
        del cfg["data"]["train"]["simulation_cfg"]
        with pytest.raises(
            ValueError,
            match=r"data\.train\.simulation_cfg.*mapping",
        ):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_validation_no_inherit_requires_nested_enrollment(self):
        """inherit_simulation: false 时只要求 validation-local enrollment 资源。"""
        cfg = _minimal_tse_cfg()
        cfg["data"]["validation"] = {
            "inherit_simulation": False,
            "use_dynamic": True,
            "spk2enroll_json": "x",
        }
        TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_validation_inherit_true_skips_check(self):
        """validation 继承 train 仿真参数（默认）时不重复校验。"""
        cfg = _minimal_tse_cfg()
        cfg["data"]["validation"] = {"inherit_simulation": True, "use_dynamic": True}
        TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_validation_without_dynamic_marker_raises(self):
        """dynamic_tse 下 validation 段必须显式走动态仿真（TSE 无离线 val 路径）。"""
        cfg = _minimal_tse_cfg()
        cfg["data"]["validation"] = {"speech_scp": "placeholder/val.scp"}
        with pytest.raises(ValueError, match="offline validation"):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_missing_validation_config_raises_at_entry(self):
        """dynamic_tse 缺少 validation 时应在入口阶段拒绝，而非延迟到 builder。"""
        cfg = _minimal_tse_cfg()
        del cfg["data"]["validation"]
        with pytest.raises(ValueError, match=r"data\.validation"):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_validation_mode_dynamic_tse_accepted(self):
        """val_dynamic 判定与 core 对齐：TSE validation 接受 mode: dynamic_tse。"""
        cfg = _minimal_tse_cfg()
        cfg["data"]["validation"] = {"mode": "dynamic_tse", "inherit_simulation": True}
        TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_validation_mode_generic_dynamic_rejected(self):
        """通用 dynamic 标记不被 TSE entry 的 validation 动态判定接受。"""
        cfg = _minimal_tse_cfg()
        cfg["data"]["validation"] = {"mode": "dynamic", "inherit_simulation": True}
        with pytest.raises(ValueError, match="offline validation"):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)


class TestResolveDistributedCfg:
    def test_derives_find_unused_parameters_false(self):
        cfg = _minimal_tse_cfg()
        TSETrainingEntry._resolve_tse_distributed_cfg(cfg)
        assert cfg["distributed"]["find_unused_parameters"] is False

    def test_derives_false_for_all_cue_combinations(self):
        """当前所有合法 cue 组合下推导恒为 False（构造即用，无可训练但无梯度参数）。"""
        joint_feature_sets = (
            {},
            {"spkemb": {"enabled": True}},
            {"context": {"enabled": True}, "spkemb": {"enabled": True}},
            {"listen": {"enabled": True}, "usef": {"enabled": True}},
        )
        preextracted_feature_sets = (
            {},
            {"spkemb": {"enabled": True}},
            {"listen": {"enabled": True}, "usef": {"enabled": True}},
        )
        for joint, feature_sets in (
            (True, joint_feature_sets),
            (False, preextracted_feature_sets),
        ):
            for feats in feature_sets:
                cfg = _minimal_tse_cfg()
                cfg["model"]["args"] = {
                    "joint_training": joint,
                    "features": feats,
                }
                TSETrainingEntry._resolve_tse_distributed_cfg(cfg)
                assert cfg["distributed"]["find_unused_parameters"] is False

    def test_explicit_value_preserved(self):
        """用户显式配置优先保留（即使与推导不一致，仅 warning）。"""
        cfg = _minimal_tse_cfg()
        cfg["distributed"] = {"find_unused_parameters": True}
        TSETrainingEntry._resolve_tse_distributed_cfg(cfg)
        assert cfg["distributed"]["find_unused_parameters"] is True


class TestResolveSpkClassifierCfg:
    """multi-task speaker CE 配置推导（仿 AEC state detector 范例）。"""

    def _cfg_with_spk_ce(self) -> dict:
        cfg = _minimal_tse_cfg()
        cfg["loss"] = {
            "type": "CompositeLoss",
            "components": [
                {"type": "SISNRLoss", "weight": 1.0},
                {"type": "SpeakerCrossEntropyLoss", "weight": 0.1},
            ],
        }
        return cfg

    def test_derives_multi_task_and_spks_in_train_from_explicit_dataset_fact(self):
        """loss 含 SpeakerCrossEntropyLoss 时，由训练 identity 空间显式传入类别数。"""
        cfg = self._cfg_with_spk_ce()
        TSETrainingEntry._resolve_spk_classifier_cfg(cfg, speaker_class_count=7)
        assert cfg["model"]["args"]["multi_task"] is True
        assert cfg["model"]["args"]["spks_in_train"] == 7

    def test_no_spk_ce_derives_multi_task_false(self):
        entry = TSETrainingEntry("TSE_Test")
        cfg = _minimal_tse_cfg()
        entry._resolve_spk_classifier_cfg(cfg)
        assert cfg["model"]["args"]["multi_task"] is False
        assert "spks_in_train" not in cfg["model"]["args"]

    def test_auxiliary_only_spk_ce_is_rejected_before_model_mutation(self):
        entry = TSETrainingEntry("TSE_Test")
        cfg = _minimal_tse_cfg()
        cfg["loss"] = {
            "type": "CompositeLoss",
            "components": [{"type": "SpeakerCrossEntropyLoss", "weight": 0.1}],
        }
        original = copy.deepcopy(cfg)

        with pytest.raises(ValueError, match="auxiliary-only"):
            entry._resolve_spk_classifier_cfg(cfg)

        assert cfg == original

    def test_auxiliary_only_spk_ce_is_rejected_during_config_validation(self):
        """在构建 DDP loader 与模型前拒绝只有非零 speaker CE 的配置。"""
        cfg = _minimal_tse_cfg()
        cfg["loss"] = {
            "type": "CompositeLoss",
            "components": [{"type": "SpeakerCrossEntropyLoss", "weight": 0.1}],
        }
        original = copy.deepcopy(cfg)

        with pytest.raises(ValueError, match="auxiliary-only"):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

        assert cfg == original

    def test_primary_loss_with_spk_ce_passes_config_validation(self):
        TSETrainingEntry("TSE_Test")._validate_config(self._cfg_with_spk_ce())

    def test_zero_weight_spk_ce_passes_config_validation(self):
        cfg = self._cfg_with_spk_ce()
        cfg["loss"]["components"][1]["weight"] = 0.0

        TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_zero_weight_spk_ce_derives_multi_task_false(self):
        """SpeakerCrossEntropyLoss 分量 weight=0 时不在 loss 计算图内，
        multi_task 必须推导为 False（否则 classifier 参数无梯度路径，
        DDP unused-parameters 崩溃）。"""
        entry = TSETrainingEntry("TSE_Test")
        cfg = self._cfg_with_spk_ce()
        cfg["loss"]["components"][1]["weight"] = 0.0
        entry._resolve_spk_classifier_cfg(cfg)
        assert cfg["model"]["args"]["multi_task"] is False
        assert "spks_in_train" not in cfg["model"]["args"]

    def test_zero_weight_spk_ce_explicit_multi_task_true_raises(self):
        """weight=0 的 CE 分量 + 显式 multi_task=True 属冲突配置，应报错。"""
        entry = TSETrainingEntry("TSE_Test")
        cfg = self._cfg_with_spk_ce()
        cfg["loss"]["components"][1]["weight"] = 0.0
        cfg["model"]["args"] = {"multi_task": True}
        with pytest.raises(ValueError, match="multi_task"):
            entry._resolve_spk_classifier_cfg(cfg)

    def test_explicit_consistent_values_kept(self):
        cfg = self._cfg_with_spk_ce()
        cfg["model"]["args"] = {"multi_task": True, "spks_in_train": 7}
        TSETrainingEntry._resolve_spk_classifier_cfg(cfg, speaker_class_count=7)
        assert cfg["model"]["args"]["spks_in_train"] == 7

    def test_multi_task_conflict_raises(self):
        """显式 multi_task 与 loss 分量存在性不一致时报错（防静默错配）。"""
        entry = TSETrainingEntry("TSE_Test")
        cfg = self._cfg_with_spk_ce()
        cfg["model"]["args"] = {"multi_task": False}
        with pytest.raises(ValueError, match="multi_task"):
            entry._resolve_spk_classifier_cfg(cfg)

    def test_spks_in_train_conflict_raises(self):
        cfg = self._cfg_with_spk_ce()
        cfg["model"]["args"] = {"multi_task": True, "spks_in_train": 5}
        with pytest.raises(ValueError, match="spks_in_train"):
            TSETrainingEntry._resolve_spk_classifier_cfg(cfg, speaker_class_count=7)

    def test_missing_spk2idx_requires_explicit_spks_in_train(self):
        """未经 build_loaders（无 identity 推导源）时必须显式配置 spks_in_train。"""
        entry = TSETrainingEntry("TSE_Test")
        cfg = self._cfg_with_spk_ce()
        cfg["model"]["args"] = {"multi_task": True}
        with pytest.raises(ValueError, match="spks_in_train"):
            entry._resolve_spk_classifier_cfg(cfg)

    def test_training_model_adapter_derives_speaker_count_from_train_dataset(
        self, monkeypatch
    ):
        cfg = self._cfg_with_spk_ce()
        entry = TSETrainingEntry("TSE_Test")
        observed = {}

        class TrainDataset:
            speaker_identity_to_idx = {("set-a", "s1"): 0, ("set-b", "s2"): 1}

        monkeypatch.setattr(
            entry,
            "_build_model",
            lambda config, rank, *, speaker_class_count: observed.update(
                config=config,
                rank=rank,
                speaker_class_count=speaker_class_count,
            ) or "model",
        )

        assert entry.build_model_for_training(cfg, rank=3, train_dataset=TrainDataset()) == "model"
        assert observed["config"] is cfg
        assert observed["rank"] == 3
        assert observed["speaker_class_count"] == 2

    def test_direct_build_model_keeps_explicit_speaker_count(self, monkeypatch):
        cfg = self._cfg_with_spk_ce()
        cfg["model"]["args"] = {"multi_task": True, "spks_in_train": 2}
        entry = TSETrainingEntry("TSE_Test")
        observed = {}

        monkeypatch.setattr("tasks.tse.models.register_tse_models", lambda **_kwargs: None)
        monkeypatch.setattr(
            "core.models.build.build_model",
            lambda model_cfg, *, stft_cfg: observed.update(
                model_cfg=model_cfg,
                stft_cfg=stft_cfg,
            ) or type("Model", (), {"parameters": lambda self: []})(),
        )

        entry.build_model(cfg, rank=1)

        assert observed["model_cfg"]["args"]["spks_in_train"] == 2


class TestTargetsPerMixtureHoisting:
    """targets_per_mixture 经 TSE_DATASET_KEYS 提升到 Dataset 构造 kwargs。"""

    def test_key_in_tse_dataset_keys(self):
        from core.data.dataset_build import TSE_DATASET_KEYS

        assert "targets_per_mixture" in TSE_DATASET_KEYS

    def test_expand_hoists_targets_per_mixture(self):
        from core.data.dataset_build import expand_simulation_cfg_tse_dataset_keys

        section = {
            "speech_scp": "x.jsonl",
            "simulation_cfg": {"targets_per_mixture": 2, "snr_low": -5},
        }
        expand_simulation_cfg_tse_dataset_keys(section)
        assert section["targets_per_mixture"] == 2
        # 仿真键保留在 simulation_cfg 内，提升仅针对 Dataset 构造键
        assert "targets_per_mixture" not in section["simulation_cfg"]
        assert section["simulation_cfg"]["snr_low"] == -5


class TestTrainerRouting:
    def test_get_trainer_cls(self):
        from tasks.tse.trainer.tse_trainer import TSETrainer

        entry = TSETrainingEntry("TSE_Test")
        assert entry.get_trainer_cls() is TSETrainer


class TestTSESamplingWiring:
    """Task 6：TSE builder 的 speech 三 controller 接入与 noise 拒绝。"""

    class _DatasetSpy:
        calls: list[dict] = []

        def __init__(self, **kwargs):
            type(self).calls.append(kwargs)

        def set_epoch(self, *args, **kwargs):
            return None

    def _spy(self, monkeypatch):
        self._DatasetSpy.calls = []
        monkeypatch.setattr("tasks.tse.data.TSEDynamicMixingDataset", self._DatasetSpy)

    def _tse_cfg(self, sampling=None) -> dict:
        train = {
            "speech_scp": "speech.jsonl",
            "noise_scp": "noise.scp",
            "spk2enroll_json": "enroll.json",
            "simulation_cfg": {},
        }
        if sampling is not None:
            train["sampling"] = sampling
        return {
            "mode": "dynamic_tse",
            "sample_rate": 16000,
            "train": train,
            "validation": {"clean_list": "clean.list", "noisy_list": "noisy.list"},
        }

    def test_tse_train_passes_speech_sampling_cfg(self, monkeypatch):
        from core.data.sampling import SamplingConfig
        from tasks.tse.data import build_dataset as build_tse_dataset

        self._spy(monkeypatch)
        build_tse_dataset(
            self._tse_cfg(
                sampling={
                    "speech": {
                        "language_probs": {"zh": 1},
                        "speaker": {"max_samples_per_epoch": 7},
                    }
                }
            )
        )
        sampling_cfg = self._DatasetSpy.calls[-1]["sampling_cfg"]
        assert isinstance(sampling_cfg, SamplingConfig)
        assert sampling_cfg.speech.speaker.max_samples_per_epoch == 7

    def test_tse_train_passes_noise_sampling_cfg(self, monkeypatch):
        from core.data.sampling import SamplingConfig
        from tasks.tse.data import build_dataset as build_tse_dataset

        self._spy(monkeypatch)
        build_tse_dataset(
            self._tse_cfg(sampling={"noise": {"balance_categories": True}})
        )
        sampling_cfg = self._DatasetSpy.calls[-1]["sampling_cfg"]
        assert isinstance(sampling_cfg, SamplingConfig)
        assert sampling_cfg.noise.balance_categories is True

    def test_tse_train_disabled_noise_is_effective_empty(self, monkeypatch):
        from tasks.tse.data import build_dataset as build_tse_dataset

        self._spy(monkeypatch)
        build_tse_dataset(
            self._tse_cfg(sampling={"noise": {"balance_categories": False}})
        )
        assert "sampling_cfg" not in self._DatasetSpy.calls[-1]

    def test_tse_validation_strips_inherited_train_sampling(self, monkeypatch):
        from tasks.tse.data import build_dataset as build_tse_dataset

        self._spy(monkeypatch)
        cfg = self._tse_cfg(
            sampling={"speech": {"speaker": {"max_samples_per_epoch": 7}}}
        )
        cfg["train"]["spk2enroll_json"] = "enroll.json"
        cfg["train"]["simulation_cfg"] = {}
        cfg["validation"] = {
            "mode": "dynamic_tse",
            "speech_scp": "val_speech.jsonl",
            "simulation_cfg": {},
        }
        build_tse_dataset(cfg, mode="val")
        kwargs = self._DatasetSpy.calls[-1]
        assert "sampling" not in kwargs
        assert "sampling_cfg" not in kwargs

    def test_tse_validation_rejects_explicit_effective_sampling(self, monkeypatch):
        from tasks.tse.data import build_dataset as build_tse_dataset

        self._spy(monkeypatch)
        cfg = self._tse_cfg()
        cfg["validation"] = {
            "mode": "dynamic_tse",
            "speech_scp": "val_speech.jsonl",
            "simulation_cfg": {},
            "sampling": {"speech": {"speaker": {"max_samples_per_epoch": 7}}},
        }
        with pytest.raises(ValueError, match="validation.*sampling.*not supported"):
            build_tse_dataset(cfg, mode="val")
        assert self._DatasetSpy.calls == []

    def test_build_loaders_preserves_train_identity_space_for_model_handoff(self, monkeypatch):
        """训练 dataset 保留 identity mapping，供入口到模型的显式交接使用。"""

        class _IdentityStubDataset:
            speaker_identity_to_idx = {("set-a", "s1"): 0, ("set-a", "s2"): 1, ("set-b", "s1"): 2}
            balanced_speech_sampling_enabled = False
            cue_specs = TSECueSpecs()

            def get_srs(self):
                return [16000] * 8

            def get_source_length(self):
                return [16000] * 8

            def __len__(self):
                return 8

            def set_epoch(self, *args, **kwargs):
                return None

        monkeypatch.setattr(
            "tasks.tse.data.build_dataset", lambda *args, **kwargs: _IdentityStubDataset()
        )
        entry = TSETrainingEntry("TSE_Test")
        cfg = _minimal_tse_cfg()
        cfg["data"]["dataloader"]["persistent_workers"] = False
        cfg["data"]["validation"] = {
            "mode": "dynamic_tse",
            "speech_scp": "val.jsonl",
            "simulation_cfg": {},
        }
        result = entry.build_loaders(cfg, rank=0, local_rank=0, world_size=1)
        assert len(result[3].speaker_identity_to_idx) == 3


class TestTSEValidationResources:
    """Task 9：validation 资源 provenance 与 label 空间对齐。"""

    class _DatasetSpy:
        calls: list[dict] = []

        def __init__(self, **kwargs):
            type(self).calls.append(kwargs)

        def set_epoch(self, *args, **kwargs):
            return None

    def _spy(self, monkeypatch):
        self._DatasetSpy.calls = []
        monkeypatch.setattr("tasks.tse.data.TSEDynamicMixingDataset", self._DatasetSpy)

    def _cfg(
        self,
        *,
        train_speech="train.jsonl",
        train_sim=None,
        val_speech="val.jsonl",
        val_sim=None,
    ) -> dict:
        train_sim = train_sim if train_sim is not None else {
            "spk2enroll_json": "enroll.json",
        }
        train_enroll = train_sim.pop("spk2enroll_json", "enroll.json")
        train_sim = {"prob_target_only": 1.0, **train_sim}
        val_sim = val_sim if val_sim is not None else {}
        val_enroll = val_sim.pop("spk2enroll_json", None)
        return {
            "mode": "dynamic_tse",
            "sample_rate": 16000,
            "train": {
                "speech_scp": train_speech,
                "noise_scp": "noise.scp",
                "spk2enroll_json": train_enroll,
                "simulation_cfg": train_sim,
            },
            "validation": {
                "mode": "dynamic_tse",
                "speech_scp": val_speech,
                **({"spk2enroll_json": val_enroll} if val_enroll is not None else {}),
                "simulation_cfg": val_sim,
            },
        }

    def test_removed_utt2spk_key_fails_fast(self, monkeypatch):
        """已移除的 utt2spk 不得再通过 entry 或 Dataset builder。"""
        from tasks.tse.data import build_dataset as build_tse_dataset

        data_cfg = self._cfg(
            train_sim={
                "spk2enroll_json": "train_nested.json",
                "utt2spk": "legacy_utt2spk",
            }
        )
        entry_cfg = _minimal_tse_cfg()
        entry_cfg["data"].update(copy.deepcopy(data_cfg))
        for validate in (
            lambda: TSETrainingEntry("TSE_Test")._validate_config(entry_cfg),
            lambda: build_tse_dataset(data_cfg, mode="train"),
        ):
            self._spy(monkeypatch)
            with pytest.raises((TypeError, ValueError), match="utt2spk"):
                validate()
            assert self._DatasetSpy.calls == []

    @pytest.mark.parametrize("invalid_sim", [[], "bad", 7])
    def test_entry_and_builder_reject_explicit_non_mapping_validation_simulation_cfg(
        self, monkeypatch, invalid_sim
    ):
        """显式 section 类型错误不得按空 mapping 静默继承 train 资源。"""
        from tasks.tse.data import build_dataset as build_tse_dataset

        data_cfg = self._cfg(
            train_speech="train.jsonl",
            train_sim={"spk2enroll_json": "train_nested.json"},
            val_speech="val.jsonl",
        )
        data_cfg["validation"]["simulation_cfg"] = invalid_sim
        entry_cfg = _minimal_tse_cfg()
        entry_cfg["data"].update(copy.deepcopy(data_cfg))

        for validate in (
            lambda: TSETrainingEntry("TSE_Test")._validate_config(entry_cfg),
            lambda: build_tse_dataset(data_cfg, mode="val"),
        ):
            self._spy(monkeypatch)
            with pytest.raises(
                ValueError,
                match=r"data\.validation\.simulation_cfg.*mapping",
            ):
                validate()
            assert self._DatasetSpy.calls == []

    @pytest.mark.parametrize("simulation_cfg_state", ["absent", "none"])
    def test_entry_and_builder_allow_validation_simulation_cfg_inherit_states(
        self, monkeypatch, simulation_cfg_state
    ):
        """同 schema 且继承开启时 absent/None 延续既有 inherit 合同。"""
        from tasks.tse.data import build_dataset as build_tse_dataset

        data_cfg = self._cfg(
            train_speech="train.jsonl",
            train_sim={"spk2enroll_json": "train_nested.json"},
            val_speech="val.jsonl",
        )
        if simulation_cfg_state == "absent":
            data_cfg["validation"].pop("simulation_cfg")
        else:
            data_cfg["validation"]["simulation_cfg"] = None
        entry_cfg = _minimal_tse_cfg()
        entry_cfg["data"].update(copy.deepcopy(data_cfg))
        self._spy(monkeypatch)

        TSETrainingEntry("TSE_Test")._validate_config(entry_cfg)
        build_tse_dataset(data_cfg, mode="val")

        assert self._DatasetSpy.calls[-1]["spk2enroll_json"] == "train_nested.json"

    @pytest.mark.parametrize(
        "invalid_speech", [None, "", "   ", " val.jsonl", "val.jsonl ", 123]
    )
    def test_entry_and_builder_reject_explicit_invalid_validation_speech_scp(
        self, monkeypatch, invalid_speech
    ):
        """显式非法 manifest 不得按 key missing 回退到 train。"""
        from tasks.tse.data import build_dataset as build_tse_dataset

        data_cfg = self._cfg(
            train_speech="train.jsonl",
            train_sim={"spk2enroll_json": "train_nested.json"},
            val_speech=invalid_speech,
        )
        entry_cfg = _minimal_tse_cfg()
        entry_cfg["data"].update(copy.deepcopy(data_cfg))
        self._spy(monkeypatch)

        for validate in (
            lambda: TSETrainingEntry("TSE_Test")._validate_config(entry_cfg),
            lambda: build_tse_dataset(data_cfg, mode="val"),
        ):
            with pytest.raises(ValueError, match=r"data\.validation\.speech_scp"):
                validate()
        assert self._DatasetSpy.calls == []

    def test_missing_validation_speech_scp_inherits_train_manifest(
        self, monkeypatch
    ):
        """只有 validation key 缺失时才允许继承 train manifest。"""
        from tasks.tse.data import build_dataset as build_tse_dataset

        data_cfg = self._cfg(
            train_speech="train.jsonl",
            train_sim={"spk2enroll_json": "train_nested.json"},
            val_speech="unused.jsonl",
        )
        del data_cfg["validation"]["speech_scp"]
        entry_cfg = _minimal_tse_cfg()
        entry_cfg["data"].update(copy.deepcopy(data_cfg))
        self._spy(monkeypatch)

        TSETrainingEntry("TSE_Test")._validate_config(entry_cfg)
        build_tse_dataset(data_cfg, mode="val")

        assert self._DatasetSpy.calls[-1]["speech_scp"] == "train.jsonl"

    @pytest.mark.parametrize(
        ("inherit", "val_sim", "accepted"),
        [
            (True, {}, True),
            (False, {"spk2enroll_json": "val_nested.json"}, True),
            (False, {}, False),
            (False, {"spk2enroll_json": None}, False),
        ],
    )
    def test_strict_validation_resource_provenance_matches_entry_and_builder(
        self, monkeypatch, inherit, val_sim, accepted
    ):
        """strict JSONL 下验证 nested enrollment 的继承与本地覆盖。"""
        from tasks.tse.data import build_dataset as build_tse_dataset

        data_cfg = self._cfg(val_sim=copy.deepcopy(val_sim))
        data_cfg["validation"]["inherit_simulation"] = inherit
        entry_cfg = _minimal_tse_cfg()
        entry_cfg["data"].update(copy.deepcopy(data_cfg))

        for is_builder, validate in (
            (False, lambda: TSETrainingEntry("TSE_Test")._validate_config(entry_cfg)),
            (True, lambda: build_tse_dataset(data_cfg, mode="val")),
        ):
            self._spy(monkeypatch)
            if accepted:
                validate()
                if is_builder:
                    assert self._DatasetSpy.calls
            else:
                with pytest.raises((TypeError, ValueError)):
                    validate()
                assert self._DatasetSpy.calls == []

    def test_inherit_true_passes_validation_local_split_resources_to_dataset(
        self, monkeypatch
    ):
        """inherit_simulation=True 仍允许 validation 覆盖 split 级资源。"""
        from tasks.tse.data import build_dataset as build_tse_dataset

        data_cfg = self._cfg(val_speech="val.jsonl")
        data_cfg["validation"]["spk2enroll_json"] = "val_nested.json"
        data_cfg["validation"]["music_scp"] = "val_music.scp"
        data_cfg["validation"]["inherit_simulation"] = True
        self._spy(monkeypatch)

        build_tse_dataset(data_cfg, mode="val")

        kwargs = self._DatasetSpy.calls[-1]
        assert kwargs["spk2enroll_json"] == "val_nested.json"
        assert kwargs["music_scp"] == "val_music.scp"

    @pytest.mark.parametrize("speech_scp", ["train.scp", "train.json"])
    def test_entry_and_builder_reject_non_strict_train_manifest(
        self, monkeypatch, speech_scp
    ):
        """动态 TSE 在 entry 与 builder 边界一致拒绝旧 speech manifest。"""
        from tasks.tse.data import build_dataset as build_tse_dataset

        cfg = self._cfg(train_speech=speech_scp)
        entry_cfg = _minimal_tse_cfg()
        entry_cfg["data"].update(copy.deepcopy(cfg))
        self._spy(monkeypatch)
        for validate in (
            lambda: TSETrainingEntry("TSE_Test")._validate_config(entry_cfg),
            lambda: build_tse_dataset(cfg, mode="train"),
        ):
            with pytest.raises(ValueError, match="strict JSONL"):
                validate()
        assert self._DatasetSpy.calls == []

    def test_builder_rejects_non_strict_validation_manifest(self, monkeypatch):
        from tasks.tse.data import build_dataset as build_tse_dataset

        self._spy(monkeypatch)
        cfg = self._cfg(val_speech="val.scp")
        with pytest.raises(ValueError, match="strict JSONL"):
            build_tse_dataset(cfg, mode="val")
        assert self._DatasetSpy.calls == []

    def test_builder_rejects_removed_validation_utt2spk(self, monkeypatch):
        """validation 中已移除的 utt2spk 也必须快速失败。"""
        from tasks.tse.data import build_dataset as build_tse_dataset

        self._spy(monkeypatch)
        cfg = self._cfg(
            val_sim={
                "spk2enroll_json": "val_nested.json",
                "utt2spk": "val_utt2spk",
            }
        )
        with pytest.raises(ValueError, match="utt2spk"):
            build_tse_dataset(cfg, mode="val")
        assert self._DatasetSpy.calls == []

    @pytest.mark.parametrize("speech_scp", ["train.scp", "train.json"])
    def test_entry_rejects_non_strict_train_manifest(self, speech_scp):
        cfg = _minimal_tse_cfg()
        cfg["data"]["train"]["speech_scp"] = speech_scp
        with pytest.raises(ValueError, match="strict JSONL"):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_entry_rejects_non_strict_validation_manifest(self):
        cfg = _minimal_tse_cfg()
        cfg["data"]["validation"] = {
            "use_dynamic": True,
            "speech_scp": "val.scp",
            "simulation_cfg": {},
        }
        with pytest.raises(ValueError, match="strict JSONL"):
            TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_entry_allows_inherited_nested_enrollment(self):
        cfg = _minimal_tse_cfg()
        cfg["data"]["validation"] = {
            "use_dynamic": True,
            "speech_scp": "val.jsonl",
            "simulation_cfg": {},
        }
        TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_val_strict_inherit_false_requires_only_enroll_key(self):
        """inherit=False 按 effective manifest 判定：strict val 只需 val-local
        spk2enroll_json，entry 校验通过。"""
        cfg = _minimal_tse_cfg()
        cfg["data"]["train"]["speech_scp"] = "train.jsonl"
        cfg["data"]["train"]["spk2enroll_json"] = "enroll.json"
        cfg["data"]["train"]["simulation_cfg"] = {"prob_target_only": 1.0}
        cfg["data"]["validation"] = {
            "inherit_simulation": False,
            "use_dynamic": True,
            "speech_scp": "val.jsonl",
            "spk2enroll_json": "val_enroll.json",
        }
        TSETrainingEntry("TSE_Test")._validate_config(cfg)

    def test_entry_injects_train_label_mapping_into_validation(self, monkeypatch):
        """Task 9 Step 6：build_loaders 把 train identity map 显式注入 val dataset。"""
        train_map = {("set-a", "s1"): 0, ("set-b", "s1"): 1}

        class _TrainStub:
            speaker_identity_to_idx = train_map
            balanced_speech_sampling_enabled = False
            cue_specs = TSECueSpecs()

            def get_srs(self):
                return [16000] * 8

            def get_source_length(self):
                return [16000] * 8

            def __len__(self):
                return 8

            def set_epoch(self, *args, **kwargs):
                return None

        class _ValStub(_TrainStub):
            received = None

            def set_speaker_label_mapping(self, mapping):
                type(self).received = mapping

        def fake_build(data_cfg, mode="train", meta=None):
            return _TrainStub() if mode == "train" else _ValStub()

        monkeypatch.setattr("tasks.tse.data.build_dataset", fake_build)
        entry = TSETrainingEntry("TSE_Test")
        cfg = _minimal_tse_cfg()
        cfg["data"]["dataloader"]["persistent_workers"] = False
        cfg["data"]["validation"] = {
            "mode": "dynamic_tse",
            "speech_scp": "val.jsonl",
            "simulation_cfg": {},
        }
        entry.build_loaders(cfg, rank=0, local_rank=0, world_size=1)
        assert _ValStub.received == train_map

    def test_tse_num_per_epoch_wiring(self, monkeypatch):
        """Task 9 Step 3：num_per_epoch 只进 loader wiring（本测试断言 loader 侧；
        Dataset kwargs 侧由 test_sampling_builder_guards 的既有用例覆盖）。"""
        captured = {}

        def fake_common(self, cfg, rank, local_rank, world_size, **kwargs):
            captured.update(kwargs)
            return None

        class _Stub:
            balanced_speech_sampling_enabled = False
            cue_specs = TSECueSpecs()

            def set_epoch(self, *args, **kwargs):
                return None

        monkeypatch.setattr(
            TSETrainingEntry, "_build_loaders_common", fake_common
        )
        monkeypatch.setattr(
            "tasks.tse.data.build_dataset", lambda *args, **kwargs: _Stub()
        )
        entry = TSETrainingEntry("TSE_Test")
        cfg = _minimal_tse_cfg()
        cfg["data"]["train"]["num_per_epoch"] = 321
        cfg["data"]["validation"] = {
            "mode": "dynamic_tse",
            "speech_scp": "val.jsonl",
            "simulation_cfg": {},
            "num_per_epoch": 55,
        }
        entry.build_loaders(cfg, rank=0, local_rank=0, world_size=1)
        assert captured["train_num_per_epoch"] == 321
        assert captured["val_num_per_epoch"] == 55


class TestInjectTseSceneGeometry:
    """_inject_tse_scene_geometry：train/validation 阵列坐标与声速必须一致。"""

    @staticmethod
    def _geometry(positions):
        from core.data.simulation.array_profiles import (
            ArrayTopology,
            MicrophoneArrayProfile,
            ResolvedMicrophoneArray,
        )

        return ResolvedMicrophoneArray(
            MicrophoneArrayProfile("stub", ArrayTopology(tuple(positions))),
            reference_channel=0,
        )

    @classmethod
    def _dataset(cls, positions=None, *, speed=343.0):
        class _Stub:
            pass

        stub = _Stub()
        if positions is not None:
            stub.resolved_microphone_array = cls._geometry(positions)
        stub.spatial_speed_of_sound_mps = speed
        return stub

    @staticmethod
    def _cfg():
        from tasks.tse.models import register_tse_models

        register_tse_models(only="TSE_Modular")
        return {
            "model": {
                "type": "TSE_Modular",
                "args": {
                    "backbone": "bsrnn",
                    "separator": {},
                    "cues": {
                        "spatial": {
                            "representation": "static",
                            "features": {"ipd": {}, "sdf": {}},
                        }
                    },
                },
            }
        }

    _POSITIONS = ((0.02, 0.02, 0.0), (-0.02, 0.02, 0.0), (-0.02, -0.02, 0.0))

    def test_matching_validation_geometry_is_injected(self):
        from tasks.tse.trainer.entry import _inject_tse_scene_geometry

        cfg = self._cfg()
        _inject_tse_scene_geometry(
            cfg,
            self._dataset(self._POSITIONS),
            self._dataset(self._POSITIONS),
        )
        args = cfg["model"]["args"]
        assert args["mic_positions_m"] == tuple(
            tuple(float(axis) for axis in row) for row in self._POSITIONS
        )
        assert args["speed_of_sound_mps"] == 343.0

    def test_mismatched_validation_positions_are_rejected(self):
        from tasks.tse.trainer.entry import _inject_tse_scene_geometry

        shifted = tuple((x + 0.01, y, z) for x, y, z in self._POSITIONS)
        with pytest.raises(ValueError, match="train/validation scene mic_positions_m must match"):
            _inject_tse_scene_geometry(
                self._cfg(),
                self._dataset(self._POSITIONS),
                self._dataset(shifted),
            )

    def test_mismatched_validation_speed_is_rejected(self):
        from tasks.tse.trainer.entry import _inject_tse_scene_geometry

        with pytest.raises(ValueError, match="train/validation scene speed_of_sound_mps must match"):
            _inject_tse_scene_geometry(
                self._cfg(),
                self._dataset(self._POSITIONS, speed=343.0),
                self._dataset(self._POSITIONS, speed=340.0),
            )

    def test_explicit_args_conflicting_with_dataset_are_rejected(self):
        from tasks.tse.trainer.entry import _inject_tse_scene_geometry

        cfg = self._cfg()
        cfg["model"]["args"]["mic_positions_m"] = [[0.0, 0.0, 0.0]] * 3
        with pytest.raises(ValueError, match="model.args.mic_positions_m conflicts"):
            _inject_tse_scene_geometry(cfg, self._dataset(self._POSITIONS), None)


pytestmark = [pytest.mark.unit]
