"""TSE WeSep 兼容、timeline 权重与冻结 speaker embedding cache 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from core.data.simulation_configs import SimulationConfigs
from core.data.speech_scene import TIMELINE_MODES, place_scene_sources
from tasks.tse.models.speaker.encoder import SpeakerEncoder
from tools.prepare_data_tools.generators.tse import import_wesep_enrollment


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _write_wav(path: Path, *, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.linspace(-0.2, 0.2, sample_rate // 10, dtype=np.float32)
    sf.write(path, waveform, sample_rate, subtype="FLOAT")


def _write_cues_yaml(path: Path, *, key: str, resource: str = "audio.json") -> None:
    path.write_text(
        "\n".join(
            (
                "cues:",
                "  audio:",
                "    type: raw",
                "    guaranteed: true",
                "    scope: speaker",
                "    policy:",
                "      type: random",
                f"      key: {key}",
                f"      resource: {resource}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_import_wesep_speaker_cues_to_dynamic_nested_json(tmp_path: Path) -> None:
    enroll_a = tmp_path / "audio" / "enroll-a.wav"
    enroll_b = tmp_path / "audio" / "enroll-b.wav"
    _write_wav(enroll_a)
    _write_wav(enroll_b)

    resource = tmp_path / "cues" / "audio.json"
    resource.parent.mkdir()
    resource.write_text(
        json.dumps(
            {
                "speaker-a": [
                    {"utt_id": "a-0", "path": "../audio/enroll-a.wav"},
                    {"utt_id": "a-1", "path": "../audio/enroll-a.wav"},
                ],
                "speaker-b": ["../audio/enroll-b.wav"],
            }
        ),
        encoding="utf-8",
    )
    cues_yaml = tmp_path / "cues" / "cues.yaml"
    _write_cues_yaml(cues_yaml, key="spk_id")
    speech = tmp_path / "speech.jsonl"
    speech.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "uid": "a",
                        "dataset_id": "vox",
                        "speaker_id": "speaker-a",
                    }
                ),
                json.dumps(
                    {
                        "uid": "b",
                        "dataset_id": "vox",
                        "speaker_id": "speaker-b",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "published" / "spk2enroll.json"

    assert import_wesep_enrollment.main(
        [
            "--cues-yaml",
            str(cues_yaml),
            "--dataset-id",
            "vox",
            "--speech-jsonl",
            str(speech),
            "--output-json",
            str(output),
        ]
    ) == 0

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "vox": {
            "speaker-a": [str(enroll_a.resolve())],
            "speaker-b": [str(enroll_b.resolve())],
        }
    }


def test_import_wesep_rejects_mix_scoped_fixed_cues(tmp_path: Path) -> None:
    cues_yaml = tmp_path / "cues.yaml"
    _write_cues_yaml(cues_yaml, key="mix_spk_id")

    with pytest.raises(ValueError, match="mix_spk_id.*offline"):
        import_wesep_enrollment.main(
            [
                "--cues-yaml",
                str(cues_yaml),
                "--dataset-id",
                "vox",
                "--output-json",
                str(tmp_path / "out.json"),
            ]
        )


def test_timeline_mode_weights_select_only_enabled_mode() -> None:
    weights = {mode: 0.0 for mode in TIMELINE_MODES}
    weights["tail"] = 1.0
    cfg = SimulationConfigs.from_dict(
        {
            "prob_timeline": 1.0,
            "overlap_ratio_range": [0.5, 0.5],
            "timeline_mode_weights": weights,
        },
        strict=True,
    )

    placed, plans = place_scene_sources(
        (np.ones(100, dtype=np.float32), np.ones(100, dtype=np.float32)),
        cfg,
        np.random.default_rng(7),
    )

    assert plans[0]["mode"] == "tail"
    assert np.count_nonzero(placed[1][:50]) == 0
    assert np.count_nonzero(placed[1][50:]) == 50


def test_scene_timeline_rejects_multichannel_source_shape() -> None:
    cfg = SimulationConfigs.from_dict(
        {"prob_timeline": 1.0, "overlap_ratio_range": [0.5, 0.5]},
        strict=True,
    )
    with pytest.raises(ValueError, match=r"one-dimensional \[T\]"):
        place_scene_sources(
            (
                np.ones((2, 100), dtype=np.float32),
                np.ones(100, dtype=np.float32),
            ),
            cfg,
            np.random.default_rng(7),
        )


@pytest.mark.parametrize(
    "weights",
    [
        {"head": 1.0},
        {mode: 0.0 for mode in TIMELINE_MODES},
        {**{mode: 0.2 for mode in TIMELINE_MODES}, "unknown": 0.1},
        {**{mode: 0.2 for mode in TIMELINE_MODES}, "head": -0.1},
    ],
)
def test_timeline_mode_weights_fail_closed(weights: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="timeline_mode_weights"):
        SimulationConfigs.from_dict(
            {"timeline_mode_weights": weights},
            strict=True,
        )


def test_frozen_eval_speaker_encoder_caches_by_waveform_content() -> None:
    encoder = SpeakerEncoder(
        model="ResNet34",
        freeze=True,
        spk_args={"feat_dim": 8, "embed_dim": 4},
        fbank_args={"num_mel_bins": 8},
    ).eval()
    calls = 0
    original_forward = encoder.spk_model.forward

    def counted_forward(features):
        nonlocal calls
        calls += 1
        return original_forward(features)

    encoder.spk_model.forward = counted_forward
    waveform = torch.linspace(-0.2, 0.2, 1600).unsqueeze(0)

    with torch.no_grad():
        first = encoder(waveform)
        second = encoder(waveform.clone())

    assert calls == 1
    assert torch.equal(first, second)
    assert not any("cache" in key for key in encoder.state_dict())
    assert encoder.embedding_cache_info()["hits"] == 1
    assert encoder.embedding_cache_info()["misses"] == 1


def test_unfrozen_eval_speaker_encoder_does_not_use_embedding_cache() -> None:
    encoder = SpeakerEncoder(
        model="ResNet34",
        freeze=False,
        spk_args={"feat_dim": 8, "embed_dim": 4},
        fbank_args={"num_mel_bins": 8},
    ).eval()
    calls = 0
    original_forward = encoder.spk_model.forward

    def counted_forward(features):
        nonlocal calls
        calls += 1
        return original_forward(features)

    encoder.spk_model.forward = counted_forward
    waveform = torch.linspace(-0.2, 0.2, 1600).unsqueeze(0)

    with torch.no_grad():
        encoder(waveform)
        encoder(waveform.clone())

    assert calls == 2
    assert encoder.embedding_cache_info()["hits"] == 0


def test_enrollment_augmentation_operation_order_is_explicit() -> None:
    from tasks.tse.data.enrollment_augment import EnrollmentAugmentor

    augmentor = EnrollmentAugmentor.from_config(
        {
            "enabled": True,
            "prob_eq": 0.0,
            "target_duration_seconds": 0.1,
            "target_db": -12.0,
            "fit_policy": "fixed",
            "prob_reverb": 1.0,
            "prob_noise": 1.0,
            "snr_range": [8.0, 8.0],
        }
    )
    waveform = np.ones(1600, dtype=np.float32) * 0.05
    rir = np.zeros((1, 16), dtype=np.float32)
    rir[0, 0] = 1.0
    noise = np.ones((1, 1600), dtype=np.float32) * 0.01

    output, facts = augmentor.apply(
        waveform,
        16000,
        np.random.default_rng(9),
        rir_provider=lambda _sr, _rng: (rir, {"path": "rir.wav"}),
        noise_provider=lambda _sr, _length, _rng: (noise, {"path": "noise.wav"}),
    )

    assert output.shape == waveform.shape
    assert facts["operations"] == ["rms", "rir", "noise"]
    assert np.isfinite(output).all()
