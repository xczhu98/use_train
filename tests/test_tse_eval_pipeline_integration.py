"""eval_pipeline_tse 的真实依赖与端到端 integration 测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.utils.inference_artifacts import fingerprint_file, stat_audio_identity
import evaluate.eval_pipeline_tse as pipeline


@pytest.mark.external
class TestWeSpeakerScorer:
    """wespeakerruntime 后端冒烟测试（依赖与模型文件缺失时跳过）。"""

    def test_same_wav_cosine_is_one(self, tmp_path: Path):
        pytest.importorskip("wespeakerruntime")
        model_dir = Path("third_party_scripts/wespeaker/en/model.onnx")
        if not model_dir.exists():
            pytest.skip("wespeaker en 模型未放置")
        import soundfile as sf

        sr = 16000
        t = np.arange(sr) / sr
        wav = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        wav_path = tmp_path / "a.wav"
        sf.write(str(wav_path), wav, sr)
        scorer = pipeline._WeSpeakerScorer(model_dir.parent.parent)
        assert scorer.cosine("en", str(wav_path), str(wav_path)) == pytest.approx(1.0)


def test_skip_inference_v5_rejects_changed_speaker_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = tmp_path / "model.tar"
    config = tmp_path / "config.yaml"
    speaker = tmp_path / "speaker.pt"
    mixture = tmp_path / "mixture.wav"
    enrollment = tmp_path / "enrollment.wav"
    meta = tmp_path / "meta.csv"
    output_dir = tmp_path / "eval"
    wav_dir = output_dir / "wav"
    torch.save({"model": {}, "resolved_config": {}}, model)
    speaker.write_bytes(b"speaker-v1")
    mixture.write_bytes(b"mixture")
    enrollment.write_bytes(b"enrollment")
    meta.write_text("fixture\n", encoding="utf-8")
    config.write_text(
        "model:\n"
        "  args:\n"
        "    joint_training: false\n"
        "    features:\n"
        "      spkemb:\n"
        "        enabled: true\n"
        "    speaker_model:\n"
        "      speaker_encoder:\n"
        f"        pretrained: {speaker}\n",
        encoding="utf-8",
    )
    row = {
        "dataset": "fixture",
        "mixture_utterance": "mix",
        "enrolment_speakers_utterance": "enroll",
    }
    mapping = {"mix": str(mixture), "enroll": str(enrollment)}
    loaded_config, effective_config, config_identity = (
        pipeline._load_model_config_for_artifacts(model, config)
    )
    resolved_speaker = pipeline._resolve_external_speaker_checkpoint_for_artifacts(
        model, config, strict=True, loaded_config=loaded_config
    )
    base_meta = pipeline._model_meta(
        str(model),
        effective_config,
        {"device": "cpu", "apply_vad": False},
        config_identity=config_identity,
        speaker_checkpoint=resolved_speaker,
        speaker_checkpoint_identity=fingerprint_file(resolved_speaker),
    )
    expected_meta = pipeline._bind_input_artifacts(
        base_meta, {"mixture": mixture, "speaker_waveform": enrollment}
    )
    output = wav_dir / f"{pipeline._out_name(pipeline._make_uid(row))}.wav"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"cached-output")
    sidecar = {**expected_meta, "output": stat_audio_identity(output)}
    output.with_name(output.stem + ".meta.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )

    speaker.write_bytes(b"speaker-v2")
    args = SimpleNamespace(
        metrics="",
        device="cpu",
        output_dir=str(output_dir),
        meta=[str(meta)],
        max_samples=None,
        mapping_csv=None,
        model=str(model),
        config=str(config),
        apply_vad=False,
        skip_inference=True,
        audio_root=None,
        ref_root=None,
        allow_partial=False,
        _public_tse_cue_roles={"speaker_waveform": None},
    )
    monkeypatch.setattr(pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(pipeline, "_read_meta", lambda _paths: [dict(row)])
    monkeypatch.setattr(pipeline, "_load_mapping", lambda _path: mapping)

    with pytest.raises(SystemExit, match="无法验证 1/1 个 artifact"):
        pipeline.main()

    manifest = json.loads(
        (output_dir / "tse_inference_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["ok"] == 0
    assert manifest["failed"] == 1


pytestmark = [pytest.mark.integration]
