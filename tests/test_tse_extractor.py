"""TSEExtractor 单测：随机初始化小模型 ckpt → 加载 → 短波形推理（CPU 可跑）。

覆盖：
- ``_require_existing_pretrained`` 相对路径解析（CWD 无关，先仓库根解析）与缺失即报错；
- 与 ``core.trainer`` 同构的 ckpt（``{"model", "resolved_config"}``）加载；
- ``extract_speech_from_pcm`` 输出长度与峰值归一（0.9）断言；
- 短于 STFT reflect padding 下限的输入补零后长度复原。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from core.inference import ArtifactBinding, FileInferenceItem, InferenceSpec, ReusePolicy, run_inference_report
from core.utils.inference_artifacts import fingerprint_file, fingerprint_json
from tasks.tse.inference import artifacts as tse_inference_artifacts
import evaluate.eval_pipeline_tse as eval_pipeline_tse
from tasks.tse.inference import TSEInferenceAdapter, publish_sidecars_from_report
from tasks.tse.inference.adapter import cue_declaration
from tasks.tse.inference.extractor import (
    TSEExtractor,
    _apply_ema_state_fail_closed,
    _load_model_state_fail_closed,
    _min_stft_input_len,
    _require_existing_pretrained,
)

_SR = 16000


def _small_cfg() -> dict:
    """CPU 可跑的小尺寸 TSE_BSRNN_SPK 配置（spkemb cue + 随机初始化 encoder）。"""
    return {
        "stft": {"n_fft": 512, "hop_length": 128, "win_length": 512, "window": "hann"},
        "model": {
            "type": "TSE_BSRNN_SPK",
            "args": {
                "sr": _SR,
                "feature_dim": 32,
                "num_repeat": 1,
                "causal": False,
                "joint_training": True,
                "features": {"spkemb": {"enabled": True}},
                "speaker_model": {"speaker_encoder": {"pretrained": None}},
            },
        },
        "data": {"sample_rate": _SR},
    }


def _save_random_ckpt(path: Path, cfg: dict | None = None) -> dict:
    """构建随机初始化小模型并保存与 trainer 同构的 ckpt，返回配置。"""
    from core.models.build import build_model
    from tasks.tse.models import register_tse_models

    cfg = cfg or _small_cfg()
    register_tse_models(only="TSE_BSRNN_SPK")
    torch.manual_seed(0)
    model = build_model(cfg["model"], stft_cfg=cfg["stft"]).eval()
    torch.save({"model": model.state_dict(), "resolved_config": cfg}, path)
    return cfg


def _save_speaker_encoder_ckpt(path: Path) -> Path:
    """保存与真实 SpeakerEncoder 严格匹配的最小本地测试 checkpoint。"""
    from tasks.tse.models.speaker.encoder import SpeakerEncoder

    encoder = SpeakerEncoder(pretrained=None)
    state = {
        f"module.{key}": value.detach().clone()
        for key, value in encoder.spk_model.state_dict().items()
    }
    torch.save(state, path)
    return path


def _run_public_inference(rows, extractor, wav_dir, mapping, snapshot, model, config=None):
    wav_dir = Path(wav_dir)
    expected = {}
    items = []
    for row in rows:
        uid = eval_pipeline_tse._make_uid(row)
        mixture = Path(mapping[row["mixture_utterance"]])
        enrollment = Path(mapping[row["enrolment_speakers_utterance"]])
        cue_files = {"speaker_waveform": enrollment}
        declaration = cue_declaration(cue_files, {})
        expected[uid] = snapshot.bind_inputs(
            {"mixture": mixture, **cue_files}, cues=declaration
        )
        items.append(FileInferenceItem(
            uid=uid,
            inputs={"mixture": mixture, "speaker_waveform": enrollment},
            output=wav_dir / f"{uid}.wav",
        ))
    spec = InferenceSpec(
        config=config or model,
        checkpoint=model,
        output_dir=wav_dir,
        items=items,
        artifacts=ArtifactBinding(
            files={"checkpoint": model, "config": config or model},
            parameters={"task": "tse", "snapshot": snapshot.meta},
        ),
        reuse=ReusePolicy(enabled=True),
        allow_partial=False,
    )
    report = run_inference_report(spec, TSEInferenceAdapter(extractor))
    publish_sidecars_from_report(report, expected, output_dir=wav_dir)
    by_uid = {result.uid: result for result in report.results}
    output_rows = []
    for row in rows:
        current = dict(row)
        result = by_uid[eval_pipeline_tse._make_uid(row)]
        current["tse_path"] = result.output_path
        current["inference_status"] = "ok" if result.status in {"ok", "skipped"} else "failed"
        current["inference_reused"] = result.status == "skipped"
        output_rows.append(current)
    return output_rows


@pytest.fixture(scope="module")
def extractor(tmp_path_factory) -> TSEExtractor:
    ckpt_path = tmp_path_factory.mktemp("ckpt") / "model.tar"
    _save_random_ckpt(ckpt_path)
    return TSEExtractor(str(ckpt_path), device="cpu")


@pytest.mark.unit
class TestPretrainedPathValidation:
    def test_missing_path_is_rejected(self):
        cfg = {"a": {"pretrained": "/nonexistent/xyz.pt"}}
        with pytest.raises(FileNotFoundError, match="pretrained"):
            _require_existing_pretrained(cfg)

    def test_missing_path_message_points_to_canonical_usage(self):
        cfg = {"a": {"pretrained": "/nonexistent/xyz.pt"}}
        with pytest.raises(FileNotFoundError, match="pretrained: ~"):
            _require_existing_pretrained(cfg)

    def test_existing_abs_path_kept(self, tmp_path: Path):
        f = tmp_path / "w.pt"
        f.write_bytes(b"x")
        cfg = {"a": {"pretrained": str(f)}}
        _require_existing_pretrained(cfg)

    def test_repo_relative_path_kept(self):
        # 相对仓库根存在的路径（不依赖 CWD）：仓库根解析后应通过
        _require_existing_pretrained({"a": {"pretrained": "requirements.txt"}})

    def test_non_pretrained_keys_untouched(self):
        cfg = {"a": {"pretrained": None, "other": "/nonexistent/keep.me"}}
        _require_existing_pretrained(cfg)


@pytest.mark.unit
class TestPCMConversion:
    @pytest.mark.parametrize(
        ("pcm", "expected"),
        [
            (np.array([-128, 0, 127], dtype=np.int8), [-1.0, 0.0, 127 / 128]),
            (torch.tensor([-32768, 0, 32767], dtype=torch.int16), [-1.0, 0.0, 32767 / 32768]),
        ],
    )
    def test_signed_integer_normalized_by_dtype_range(self, pcm, expected):
        out = TSEExtractor._to_mono_float32(pcm)
        assert out.dtype == torch.float32
        assert out.tolist() == pytest.approx(expected)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    @pytest.mark.parametrize(
        ("pcm", "expected"),
        [
            (np.array([0, 32768, 65535], dtype=np.uint16), [-1.0, 0.0, 32767 / 32768]),
            (torch.tensor([0, 128, 255], dtype=torch.uint8), [-1.0, 0.0, 127 / 128]),
        ],
    )
    def test_unsigned_integer_is_centered_and_normalized(self, pcm, expected):
        out = TSEExtractor._to_mono_float32(pcm)
        assert out.dtype == torch.float32
        assert out.tolist() == pytest.approx(expected)

    @pytest.mark.parametrize(
        "pcm",
        [
            np.array([True, False]),
            np.array([1 + 2j], dtype=np.complex64),
            torch.tensor([True, False]),
            torch.tensor([1 + 2j], dtype=torch.complex64),
        ],
    )
    def test_rejects_bool_and_complex(self, pcm):
        with pytest.raises(TypeError, match="不支持 dtype"):
            TSEExtractor._to_mono_float32(pcm)

    def test_float_amplitude_is_not_normalized(self):
        pcm = np.array([-2.5, 0.25, 3.0], dtype=np.float64)
        out = TSEExtractor._to_mono_float32(pcm)
        assert out.dtype == torch.float32
        assert out.tolist() == pytest.approx(pcm.tolist())


@pytest.mark.unit
class TestCheckpointFailClosed:
    def test_public_constructor_rejects_partial_loading_flag(self):
        with pytest.raises(ValueError, match="no longer supports partial"):
            TSEExtractor("unused.tar", device="cpu", allow_partial_load=True)

    def test_model_state_mismatch_does_not_partially_copy(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 2),
        )
        before = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        bad_state = {key: value.detach().clone() for key, value in before.items()}
        bad_state["0.weight"].fill_(99)
        bad_state.pop("1.bias")

        with pytest.raises(RuntimeError, match="missing"):
            _load_model_state_fail_closed(
                model, bad_state, checkpoint_label="test checkpoint"
            )

        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value)

    def test_ema_mismatch_does_not_copy_any_shadow_parameter(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 2),
        )
        parameters = list(model.parameters())
        before = [parameter.detach().clone() for parameter in parameters]
        shadows = [torch.full_like(parameter, 7.0) for parameter in parameters]
        shadows[-1] = torch.zeros(parameters[-1].numel() + 1)

        with pytest.raises(RuntimeError, match="EMA shadow parameter validation"):
            _apply_ema_state_fail_closed(
                model,
                {"shadow_params": shadows},
                checkpoint_label="test checkpoint",
            )

        for parameter, expected in zip(parameters, before):
            torch.testing.assert_close(parameter, expected)

    def test_valid_ema_state_replaces_all_model_parameters(self):
        model = torch.nn.Linear(3, 2)
        shadows = [
            torch.full_like(parameter, float(index + 1))
            for index, parameter in enumerate(model.parameters())
        ]

        _apply_ema_state_fail_closed(
            model,
            {"shadow_params": shadows},
            checkpoint_label="test checkpoint",
        )

        for parameter, expected in zip(model.parameters(), shadows):
            torch.testing.assert_close(parameter, expected)


@pytest.mark.unit
@pytest.mark.integration
class TestExtractSpeechFromPCM:
    def test_output_length_and_peak(self, extractor: TSEExtractor):
        rng = np.random.default_rng(0)
        mix = (0.1 * rng.standard_normal(_SR // 2)).astype(np.float32)
        enroll = (0.1 * rng.standard_normal(_SR // 4)).astype(np.float32)
        out = extractor.extract_speech_from_pcm(mix, _SR, enroll, _SR)
        assert out.dtype == np.float32
        assert out.ndim == 1
        assert len(out) == len(mix)
        assert np.all(np.isfinite(out))
        # 峰值归一化到 output_peak=0.9（全零输出守卫除外，随机输入不会触发）
        assert float(np.abs(out).max()) == pytest.approx(0.9, abs=1e-5)

    def test_short_input_padded_then_restored(self, extractor: TSEExtractor):
        # 短于 n_fft//2+1 的输入：内部补零，输出裁回原始长度
        min_len = _min_stft_input_len(extractor.stft.n_fft)
        rng = np.random.default_rng(1)
        mix = (0.1 * rng.standard_normal(min_len - 1)).astype(np.float32)
        enroll = (0.1 * rng.standard_normal(_SR // 4)).astype(np.float32)
        out = extractor.extract_speech_from_pcm(mix, _SR, enroll, _SR)
        assert len(out) == len(mix)

    def test_joint_short_enroll_is_padded_for_internal_speaker_fbank(
        self, extractor: TSEExtractor
    ):
        """joint 模式实际启用 speaker encoder 时也必须满足 fbank 窗长。"""
        mix = np.zeros(1024, dtype=np.float32)
        enroll = np.zeros(16, dtype=np.float32)

        out = extractor.extract_speech_from_pcm(mix, _SR, enroll, _SR)

        assert len(out) == len(mix)

    def test_resample_to_model_sr(self, extractor: TSEExtractor):
        # 输入 8k：重采样到模型 16k，输出长度按模型采样率折算
        rng = np.random.default_rng(2)
        mix8k = (0.1 * rng.standard_normal(4000)).astype(np.float32)
        enroll8k = (0.1 * rng.standard_normal(2000)).astype(np.float32)
        out = extractor.extract_speech_from_pcm(mix8k, 8000, enroll8k, 8000)
        assert len(out) == 8000

    def test_empty_input_rejected(self, extractor: TSEExtractor):
        enroll = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError, match="mix 为空"):
            extractor.extract_speech_from_pcm(
                np.zeros(0, dtype=np.float32), _SR, enroll, _SR)

    @pytest.mark.parametrize(
        ("sr_mix", "sr_enroll", "match"),
        [
            (0, _SR, "sr_mix"),
            (_SR, -1, "sr_enroll"),
            (8000.5, _SR, "sr_mix"),
            (_SR, True, "sr_enroll"),
        ],
    )
    def test_invalid_input_sample_rates_fail_explicitly(
        self, extractor: TSEExtractor, sr_mix, sr_enroll, match
    ):
        pcm = np.zeros(100, dtype=np.float32)
        with pytest.raises((TypeError, ValueError), match=match):
            extractor.extract_speech_from_pcm(pcm, sr_mix, pcm, sr_enroll)

    def test_passes_effective_sample_rate_to_model(
        self, extractor: TSEExtractor, monkeypatch
    ):
        seen = {}
        original_forward = extractor.model.forward

        def capture_sample_rate(data):
            seen["sample_rate"] = data.get("sample_rate")
            return original_forward(data)

        monkeypatch.setattr(extractor.model, "forward", capture_sample_rate)
        pcm = np.zeros(1024, dtype=np.float32)
        extractor.extract_speech_from_pcm(pcm, _SR, pcm, _SR)
        assert seen["sample_rate"] == _SR


@pytest.mark.unit
@pytest.mark.integration
class TestExtractorConfiguration:
    def test_explicit_relative_config_is_loaded_and_exposed_from_current_cwd(
        self, tmp_path: Path, monkeypatch
    ):
        work_dir = tmp_path / "caller"
        work_dir.mkdir()
        config_path = work_dir / "config.yaml"
        config_path.write_text(json.dumps(_small_cfg()), encoding="utf-8")
        ckpt_path = tmp_path / "model.tar"
        _save_random_ckpt(ckpt_path)
        monkeypatch.chdir(work_dir)

        extractor = TSEExtractor(
            str(ckpt_path), config_path="config.yaml", device="cpu"
        )

        assert extractor.config_path == str(config_path.resolve())
        assert extractor.loaded_config_identity["path"] == str(config_path.resolve())

    def test_model_dir_exposes_its_canonical_config_path(self, tmp_path: Path):
        model_dir = tmp_path / "model-dir"
        model_dir.mkdir()
        config_path = model_dir / "config.yaml"
        config_path.write_text(json.dumps(_small_cfg()), encoding="utf-8")
        _save_random_ckpt(model_dir / "avg_model.pt")

        extractor = TSEExtractor(str(model_dir), device="cpu")

        assert extractor.config_path == str(config_path.resolve())
        assert extractor.loaded_config_identity["path"] == str(config_path.resolve())
        assert extractor.model_checkpoint_path == str(
            (model_dir / "avg_model.pt").resolve()
        )
        assert extractor.loaded_model_checkpoint_identity == fingerprint_file(
            model_dir / "avg_model.pt"
        )

    @pytest.mark.parametrize("model_dir_mode", [False, True])
    def test_rejects_external_config_symlink(
        self, tmp_path: Path, model_dir_mode: bool
    ):
        real_config = tmp_path / "real-config.yaml"
        real_config.write_text(json.dumps(_small_cfg()), encoding="utf-8")
        try:
            if model_dir_mode:
                model_dir = tmp_path / "model-dir"
                model_dir.mkdir()
                _save_random_ckpt(model_dir / "avg_model.pt")
                linked_config = model_dir / "config.yaml"
            else:
                checkpoint = tmp_path / "model.tar"
                _save_random_ckpt(checkpoint)
                linked_config = tmp_path / "linked-config.yaml"
            linked_config.symlink_to(real_config)
        except OSError as exc:
            pytest.skip(f"symlinks unsupported: {exc}")

        source = model_dir if model_dir_mode else checkpoint
        with pytest.raises(ValueError, match="符号链接"):
            TSEExtractor(
                str(source),
                config_path=None if model_dir_mode else str(linked_config),
                device="cpu",
            )

    def test_checkpoint_embedded_config_exposes_no_external_path(self, tmp_path: Path):
        ckpt_path = tmp_path / "model.tar"
        cfg = _save_random_ckpt(ckpt_path)

        extractor = TSEExtractor(str(ckpt_path), device="cpu")

        assert extractor.config_path is None
        assert extractor.loaded_config_identity is None
        assert extractor.model_checkpoint_path == str(ckpt_path.resolve())
        assert extractor.loaded_model_checkpoint_identity == fingerprint_file(ckpt_path)
        assert extractor.resolved_config_identity == {
            "algorithm": "sha256-json-v1",
            "sha256": fingerprint_json(cfg),
        }

    def test_external_base_config_is_bound_through_resolved_identity(
        self, tmp_path: Path
    ):
        base_cfg = _small_cfg()
        child_cfg = {"base_config": "base.yaml", "data": {"sample_rate": _SR}}
        base_path = tmp_path / "base.yaml"
        child_path = tmp_path / "child.yaml"
        base_path.write_text(json.dumps(base_cfg), encoding="utf-8")
        child_path.write_text(json.dumps(child_cfg), encoding="utf-8")
        ckpt_path = tmp_path / "model.tar"
        _save_random_ckpt(ckpt_path, base_cfg)

        extractor = TSEExtractor(
            str(ckpt_path), config_path=str(child_path), device="cpu"
        )

        assert extractor.resolved_config_identity == {
            "algorithm": "sha256-json-v1",
            "sha256": fingerprint_json(base_cfg),
        }

    def test_model_checkpoint_change_during_safe_loader_fails_closed(
        self, tmp_path: Path, monkeypatch
    ):
        ckpt_path = tmp_path / "model.tar"
        _save_random_ckpt(ckpt_path)
        original_loader = TSEExtractor._torch_load

        def mutating_loader(path):
            loaded = original_loader(path)
            ckpt_path.write_bytes(b"changed-during-loader")
            return loaded

        monkeypatch.setattr(
            TSEExtractor, "_torch_load", staticmethod(mutating_loader)
        )

        with pytest.raises(RuntimeError, match="model checkpoint changed after identity capture"):
            TSEExtractor(str(ckpt_path), device="cpu")

    def test_external_config_change_during_load_fails_closed(
        self, tmp_path: Path, monkeypatch
    ):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(json.dumps(_small_cfg()), encoding="utf-8")
        ckpt_path = tmp_path / "model.tar"
        _save_random_ckpt(ckpt_path)
        from tasks.tse.inference import extractor as extractor_module

        original_loader = extractor_module.load_merged_config

        def mutating_loader(path):
            loaded = original_loader(path)
            config_path.write_text(json.dumps({**_small_cfg(), "changed": True}))
            return loaded

        monkeypatch.setattr(extractor_module, "load_merged_config", mutating_loader)

        with pytest.raises(RuntimeError, match="配置在加载期间发生变化"):
            TSEExtractor(str(ckpt_path), config_path=str(config_path), device="cpu")

    def test_rejects_zero_data_sample_rate_before_fallback(self, tmp_path: Path):
        cfg = _small_cfg()
        cfg["data"]["sample_rate"] = 0
        ckpt_path = tmp_path / "zero-sr.tar"
        _save_random_ckpt(ckpt_path, cfg)

        with pytest.raises(ValueError, match=r"effective sample_rate.*> 0"):
            TSEExtractor(str(ckpt_path), device="cpu")

    @pytest.mark.parametrize("sample_rate", [8000, 24000])
    def test_rejects_effective_sample_rate_mismatch(
        self, tmp_path: Path, sample_rate: int
    ):
        ckpt_path = tmp_path / "model.tar"
        _save_random_ckpt(ckpt_path)

        with pytest.raises(ValueError, match=r"effective sample_rate.*model\.args\.sr"):
            TSEExtractor(str(ckpt_path), device="cpu", sample_rate=sample_rate)

    @pytest.mark.parametrize("joint_training", [True, False])
    def test_missing_pretrained_path_is_never_silently_dropped(
        self, tmp_path: Path, joint_training: bool
    ):
        """缺失的 ``pretrained`` 路径不再静默置 None。

        joint 模式由主 checkpoint 提供内部 encoder，该训练遗留字段被显式清空；
        non-joint 模式缺少外部 speaker checkpoint 时 fail-closed。
        """
        cfg = _small_cfg()
        args = cfg["model"]["args"]
        args["joint_training"] = joint_training
        args["speaker_model"] = {
            "speaker_encoder": {"pretrained": "/nonexistent/train-time-spk.pt"}
        }
        ckpt_path = tmp_path / "model.tar"
        _save_random_ckpt(ckpt_path)
        config_path = tmp_path / "config.yaml"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")

        if joint_training:
            extractor = TSEExtractor(
                str(ckpt_path), config_path=str(config_path), device="cpu"
            )
            assert extractor.model is not None
        else:
            with pytest.raises(ValueError, match="must point to an existing file"):
                TSEExtractor(
                    str(ckpt_path), config_path=str(config_path), device="cpu"
                )

    def test_preextracted_mode_builds_configured_encoder(
        self, tmp_path: Path, monkeypatch
    ):
        cfg = _small_cfg()
        args = cfg["model"]["args"]
        args["joint_training"] = False
        speaker_ckpt = tmp_path / "missing-speaker-encoder.pt"
        torch.save({"state_dict": {}}, speaker_ckpt)
        args["speaker_model"] = {
            "fbank": {"sample_rate": _SR, "num_mel_bins": 80},
            "speaker_encoder": {
                "model": "ECAPA_TDNN_GLOB_c512",
                "pretrained": str(speaker_ckpt),
                "freeze": False,
                "spk_args": {"embed_dim": 192},
            },
        }
        ckpt_path = tmp_path / "preextracted.tar"
        _save_random_ckpt(ckpt_path, cfg)
        extractor = TSEExtractor(str(ckpt_path), device="cpu")
        calls = {}

        class FakeSpeakerEncoder(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                calls["kwargs"] = kwargs

            def forward(self, wav):
                calls["wav_shape"] = tuple(wav.shape)
                return torch.zeros(wav.shape[0], 192, device=wav.device)

        monkeypatch.setattr(
            "tasks.tse.models.speaker.encoder.SpeakerEncoder", FakeSpeakerEncoder
        )
        seen = {}
        original_forward = extractor.model.forward

        def capture_model_input(data):
            seen["spk_embedding"] = data["spk_embedding"].detach().clone()
            return original_forward(data)

        monkeypatch.setattr(extractor.model, "forward", capture_model_input)
        pcm = np.zeros(1024, dtype=np.float32)
        extractor.extract_speech_from_pcm(pcm, _SR, pcm, _SR)

        assert calls["kwargs"]["pretrained"] == str(speaker_ckpt.resolve())
        assert calls["kwargs"]["pretrained_state"] == {"state_dict": {}}
        assert calls["kwargs"]["freeze"] is True
        assert calls["kwargs"]["fbank_args"]["sample_rate"] == _SR
        assert calls["wav_shape"] == (1, 1024)
        assert extractor.prepare_external_speaker_encoder()["path"] == str(
            speaker_ckpt.resolve()
        )
        embedding = seen["spk_embedding"]
        assert embedding.shape == (1, 192)
        assert embedding.is_floating_point()
        assert torch.isfinite(embedding).all()

    def test_external_speaker_checkpoint_change_during_loader_fails_closed(
        self, tmp_path: Path, monkeypatch
    ):
        cfg = _small_cfg()
        args = cfg["model"]["args"]
        args["joint_training"] = False
        speaker_ckpt = tmp_path / "speaker-encoder.pt"
        torch.save({"state_dict": {}}, speaker_ckpt)
        args["speaker_model"] = {
            "fbank": {"sample_rate": _SR, "num_mel_bins": 80},
            "speaker_encoder": {
                "model": "fake-speaker",
                "pretrained": str(speaker_ckpt),
                "spk_args": {"embed_dim": 192},
            },
        }
        ckpt_path = tmp_path / "preextracted-toctou.tar"
        _save_random_ckpt(ckpt_path, cfg)
        extractor = TSEExtractor(str(ckpt_path), device="cpu")

        class MutatingSpeakerEncoder(torch.nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()
                speaker_ckpt.write_bytes(b"speaker-v2")

        monkeypatch.setattr(
            "tasks.tse.models.speaker.encoder.SpeakerEncoder", MutatingSpeakerEncoder
        )

        with pytest.raises(RuntimeError, match="speaker checkpoint changed after identity capture"):
            extractor.prepare_external_speaker_encoder()
        assert extractor._spk_encoder is None
        assert extractor._loaded_external_speaker_checkpoint_identity is None

    def test_preextracted_short_enroll_is_padded_for_speaker_fbank(
        self, tmp_path: Path, monkeypatch
    ):
        """非 joint 模式的短 enroll 也必须满足 speaker fbank 窗长。"""
        cfg = _small_cfg()
        args = cfg["model"]["args"]
        args["joint_training"] = False
        speaker_ckpt = tmp_path / "missing-speaker-encoder.pt"
        torch.save({"state_dict": {}}, speaker_ckpt)
        args["speaker_model"] = {
            "fbank": {"sample_rate": _SR, "num_mel_bins": 80},
            "speaker_encoder": {
                "model": "fake-speaker",
                "pretrained": str(speaker_ckpt),
                "spk_args": {"embed_dim": 192},
            },
        }
        ckpt_path = tmp_path / "preextracted-short.tar"
        _save_random_ckpt(ckpt_path, cfg)
        extractor = TSEExtractor(str(ckpt_path), device="cpu")
        calls = {}

        class FakeSpeakerEncoder(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.fbank = type("FakeFbank", (), {"window_size": 400})()

            def forward(self, wav):
                calls["wav_shape"] = tuple(wav.shape)
                if wav.shape[-1] < self.fbank.window_size:
                    raise ValueError("speaker fbank input is too short")
                return torch.zeros(wav.shape[0], 192, device=wav.device)

        monkeypatch.setattr(
            "tasks.tse.models.speaker.encoder.SpeakerEncoder", FakeSpeakerEncoder
        )
        mix = np.zeros(1024, dtype=np.float32)
        enroll = np.zeros(16, dtype=np.float32)
        out = extractor.extract_speech_from_pcm(mix, _SR, enroll, _SR)

        assert len(out) == len(mix)
        assert calls["wav_shape"] == (1, 400)


@pytest.mark.unit
@pytest.mark.integration
class TestInferenceArtifactSnapshot:
    def test_real_loader_output_then_inspection_reuse_and_dependency_rejection(
        self, tmp_path: Path
    ):
        ckpt_path = tmp_path / "model.tar"
        cfg = _save_random_ckpt(ckpt_path)
        config_path = tmp_path / "config.yaml"
        config_text = json.dumps(cfg)
        config_path.write_text(config_text, encoding="utf-8")
        mix = tmp_path / "mix.wav"
        enroll = tmp_path / "enroll.wav"
        signal = (0.05 * np.sin(np.linspace(0.0, 8.0 * np.pi, 1024))).astype(
            np.float32
        )
        sf.write(mix, signal, _SR)
        sf.write(enroll, signal[:512], _SR)
        row = {
            "dataset": "fixture",
            "mixture_utterance": "mix",
            "enrolment_speakers_utterance": "enroll",
        }
        mapping = {"mix": str(mix), "enroll": str(enroll)}
        params = {"device": "cpu", "apply_vad": False}
        extractor = TSEExtractor(
            str(ckpt_path), config_path=str(config_path), device="cpu"
        )
        loaded = tse_inference_artifacts.TSEInferenceArtifactSnapshot.from_loaded_extractor(
            extractor, params
        )
        assert loaded.mode == "loaded"
        assert loaded.meta["artifacts"]["model"] == extractor.loaded_model_checkpoint_identity
        assert loaded.meta["artifacts"]["speaker_checkpoint"] is None

        wav_dir = tmp_path / "eval" / "wav"
        produced = _run_public_inference([dict(row)], extractor, wav_dir, mapping, loaded, ckpt_path, config_path)
        output = Path(produced[0]["tse_path"])
        assert output.is_file()
        assert produced[0]["inference_reused"] is False

        inspected = tse_inference_artifacts.TSEInferenceArtifactSnapshot.inspect_for_skip(
            ckpt_path, config_path, params
        )
        assert inspected.mode == "inspection"
        declaration = cue_declaration({"speaker_waveform": enroll}, {})
        expected = inspected.bind_inputs(
            {"mixture": mix, "speaker_waveform": enroll}, cues=declaration
        )
        sidecar_path = output.with_name(output.stem + ".meta.json")
        assert inspected.matches_sidecar(
            tse_inference_artifacts.load_sidecar(sidecar_path), expected, output
        )

        reused = _run_public_inference([dict(row)], extractor, wav_dir, mapping, inspected, ckpt_path, config_path)
        assert reused[0]["inference_reused"] is True
        assert reused[0]["tse_path"] == str(output)

        ckpt_path.write_bytes(ckpt_path.read_bytes() + b"model-change")
        changed_model = tse_inference_artifacts.TSEInferenceArtifactSnapshot.inspect_for_skip(
            ckpt_path, config_path, params
        )
        assert not changed_model.matches_sidecar(
            tse_inference_artifacts.load_sidecar(sidecar_path),
            changed_model.bind_inputs(
                {"mixture": mix, "speaker_waveform": enroll}, cues=declaration
            ),
            output,
        )
        _save_random_ckpt(ckpt_path, cfg)

        config_path.write_text(
            json.dumps({**cfg, "artifact_change": True}), encoding="utf-8"
        )
        changed_config = tse_inference_artifacts.TSEInferenceArtifactSnapshot.inspect_for_skip(
            ckpt_path, config_path, params
        )
        assert not changed_config.matches_sidecar(
            tse_inference_artifacts.load_sidecar(sidecar_path),
            changed_config.bind_inputs({"mixture": mix, "speaker_waveform": enroll}),
            output,
        )
        config_path.write_text(config_text, encoding="utf-8")

        mix_stat = mix.stat()
        mix.touch()
        changed_input = tse_inference_artifacts.TSEInferenceArtifactSnapshot.inspect_for_skip(
            ckpt_path, config_path, params
        )
        assert mix.stat().st_mtime_ns >= mix_stat.st_mtime_ns
        assert not changed_input.matches_sidecar(
            tse_inference_artifacts.load_sidecar(sidecar_path),
            changed_input.bind_inputs({"mixture": mix, "speaker_waveform": enroll}),
            output,
        )

        output_stat = output.stat()
        output.touch()
        assert output.stat().st_mtime_ns >= output_stat.st_mtime_ns
        assert not inspected.matches_sidecar(
            tse_inference_artifacts.load_sidecar(sidecar_path), expected, output
        )

    def test_non_joint_speaker_loader_identity_matches_inspection_and_rejects_change(
        self, tmp_path: Path
    ):
        speaker_ckpt = _save_speaker_encoder_ckpt(tmp_path / "speaker.pt")
        cfg = _small_cfg()
        cfg["model"]["args"]["joint_training"] = False
        cfg["model"]["args"]["speaker_model"] = {
            "fbank": {"sample_rate": _SR, "num_mel_bins": 80},
            "speaker_encoder": {
                "model": "ECAPA_TDNN_GLOB_c512",
                "pretrained": str(speaker_ckpt),
                "spk_args": {"embed_dim": 192},
            },
        }
        ckpt_path = tmp_path / "model.tar"
        _save_random_ckpt(ckpt_path, cfg)
        config_path = tmp_path / "config.yaml"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
        params = {"device": "cpu", "apply_vad": False}

        extractor = TSEExtractor(
            str(ckpt_path), config_path=str(config_path), device="cpu"
        )
        loaded_snapshot = extractor.loaded_artifact_snapshot()
        assert loaded_snapshot["speaker_checkpoint_identity"] == fingerprint_file(
            speaker_ckpt
        )
        loaded = tse_inference_artifacts.TSEInferenceArtifactSnapshot.from_loaded_extractor(
            extractor, params
        )
        inspected = tse_inference_artifacts.TSEInferenceArtifactSnapshot.inspect_for_skip(
            ckpt_path, config_path, params
        )
        loaded_speaker = loaded.meta["artifacts"]["speaker_checkpoint"]
        assert loaded_speaker == fingerprint_file(speaker_ckpt)
        assert loaded_speaker == inspected.meta["artifacts"]["speaker_checkpoint"]
        assert loaded.meta["fingerprint"] == inspected.meta["fingerprint"]
        assert extractor._spk_encoder is not None

        mix = tmp_path / "mix.wav"
        enroll = tmp_path / "enroll.wav"
        output = tmp_path / "output.wav"
        sf.write(mix, np.zeros(512, dtype=np.float32), _SR)
        sf.write(enroll, np.zeros(512, dtype=np.float32), _SR)
        sf.write(output, np.zeros(512, dtype=np.float32), _SR)
        expected = loaded.bind_inputs({"mixture": mix, "speaker_waveform": enroll})
        sidecar_path = output.with_name(output.stem + ".meta.json")
        loaded.write_output_sidecar(sidecar_path, output, expected)
        assert inspected.matches_sidecar(
            tse_inference_artifacts.load_sidecar(sidecar_path),
            inspected.bind_inputs({"mixture": mix, "speaker_waveform": enroll}),
            output,
        )

        speaker_ckpt.write_bytes(speaker_ckpt.read_bytes() + b"speaker-change")
        changed = tse_inference_artifacts.TSEInferenceArtifactSnapshot.inspect_for_skip(
            ckpt_path, config_path, params
        )
        assert not changed.matches_sidecar(
            tse_inference_artifacts.load_sidecar(sidecar_path),
            changed.bind_inputs({"mixture": mix, "speaker_waveform": enroll}),
            output,
        )

    def test_skip_inspection_does_not_construct_extractor_or_speaker_encoder(
        self, tmp_path: Path, monkeypatch
    ):
        ckpt_path = tmp_path / "model.tar"
        _save_random_ckpt(ckpt_path)
        params = {"device": "cpu", "apply_vad": False}

        def unexpected_constructor(*_args, **_kwargs):
            raise AssertionError("skip inspection must not construct TSEExtractor")

        monkeypatch.setattr("tasks.tse.inference.TSEExtractor", unexpected_constructor)
        snapshot = tse_inference_artifacts.TSEInferenceArtifactSnapshot.inspect_for_skip(
            ckpt_path, None, params
        )

        assert snapshot.mode == "inspection"
        assert snapshot.meta["artifacts"]["speaker_checkpoint"] is None

    def test_main_skip_inference_hits_real_loader_artifact_without_extractor(
        self, tmp_path: Path, monkeypatch
    ):
        ckpt_path = tmp_path / "model.tar"
        cfg = _save_random_ckpt(ckpt_path)
        config_path = tmp_path / "config.yaml"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
        (tmp_path / "meta.csv").write_text("fixture\n", encoding="utf-8")
        mix = tmp_path / "mix.wav"
        enroll = tmp_path / "enroll.wav"
        signal = np.linspace(-0.05, 0.05, 1024, dtype=np.float32)
        sf.write(mix, signal, _SR)
        sf.write(enroll, signal[:512], _SR)
        row = {
            "dataset": "fixture",
            "mixture_utterance": "mix",
            "enrolment_speakers_utterance": "enroll",
        }
        mapping = {"mix": str(mix), "enroll": str(enroll)}
        params = {"device": "cpu", "apply_vad": False}
        dataset_identity = eval_pipeline_tse._build_dataset_identity(
            SimpleNamespace(
                meta=[str(tmp_path / "meta.csv")],
                mapping_csv=None,
                audio_root=None,
                timing_gt_json_base_dir=None,
            ),
            [row],
        )
        extractor = TSEExtractor(str(ckpt_path), config_path=str(config_path), device="cpu")
        snapshot = tse_inference_artifacts.TSEInferenceArtifactSnapshot.from_loaded_extractor(
            extractor, params, dataset_identity=dataset_identity
        )
        output_dir = tmp_path / "eval"
        _run_public_inference([dict(row)], extractor, output_dir / "wav", mapping, snapshot, ckpt_path, config_path)
        args = SimpleNamespace(
            metrics="",
            device="cpu",
            output_dir=str(output_dir),
            meta=[str(tmp_path / "meta.csv")],
            max_samples=None,
            mapping_csv=None,
            model=str(ckpt_path),
            config=str(config_path),
            apply_vad=False,
            skip_inference=True,
            audio_root=None,
            ref_root=None,
            allow_partial=False,
            _public_tse_cue_roles={"speaker_waveform": None},
        )

        def unexpected_constructor(*_args, **_kwargs):
            raise AssertionError("--skip_inference must not construct TSEExtractor")

        monkeypatch.setattr(eval_pipeline_tse, "parse_args", lambda: args)
        monkeypatch.setattr(eval_pipeline_tse, "_read_meta", lambda _paths: [dict(row)])
        monkeypatch.setattr(eval_pipeline_tse, "_load_mapping", lambda _path: mapping)
        monkeypatch.setattr("tasks.tse.inference.TSEExtractor", unexpected_constructor)

        assert eval_pipeline_tse.main() == 0
        manifest = json.loads(
            (output_dir / "tse_inference_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["ok"] == 1
        assert manifest["items"][0]["reused"] is True


@pytest.mark.unit
class TestTorchLoadFallback:
    def test_corrupt_file_raises_not_silent(self, tmp_path: Path):
        # 非 weights_only 类异常（文件损坏）必须直接抛出，不静默回退
        bad = tmp_path / "bad.tar"
        bad.write_bytes(b"not a checkpoint")
        with pytest.raises(Exception):
            TSEExtractor._torch_load(bad)

    def test_normal_ckpt_loads(self, tmp_path: Path):
        p = tmp_path / "ok.tar"
        torch.save({"model": {"w": torch.zeros(2)}}, p)
        assert "model" in TSEExtractor._torch_load(p)

pytestmark = [pytest.mark.unit, pytest.mark.contract]
