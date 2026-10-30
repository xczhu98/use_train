"""Unit tests for dataset data-summary reporting used before training epochs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

from tasks.aec.data.dynamic_dataset import DynamicMixingDataset as AECDynamicMixingDataset
from core.data.dynamic_dataset_base import DataSummary, SourceSummary
from core.data.noise_admission import (
    NOISE_MANIFEST_SCHEMA_VERSION,
    load_canonical_voice_policy,
)
from core.data.noise_admission_policy import (
    ADMISSION_POLICY_VERSION,
    CLASSIFIER_LABEL_COUNT,
    EVIDENCE_CONTRACT_VERSION,
    SCORE_CONTRACT,
)
from tasks.se.data.dynamic_dataset import DynamicMixingDataset as SEDynamicMixingDataset
from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset

pytestmark = pytest.mark.unit


def _write_wav(path: Path, value: float, sr: int = 16000, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 1:
        data = np.full(sr, value, dtype=np.float32)
    else:
        data = np.full((sr, channels), value, dtype=np.float32)
    soundfile.write(path, data, sr)


def _write_scp(path: Path, uid: str, wav: Path, sr: int = 16000) -> None:
    path.write_text(f"{uid} {sr} {sr} {wav}\n", encoding="utf-8")


def _noise_record(
    uid: str,
    fs: int,
    wav: Path,
    category_id: int,
    category_name: str,
    dataset_id: str,
    roles: list[str] | None = None,
) -> dict:
    """构造 schema v6/top-5/no-threshold 噪声记录。"""
    policy = load_canonical_voice_policy()[0]
    mid = next(
        mid
        for mid, label in policy.mid_to_label.items()
        if label == category_name and policy.class_of(mid) == "noise"
    )
    mids = [mid]
    for candidate in policy.index_to_mid.values():
        if len(mids) == 5:
            break
        if candidate not in mids and policy.class_of(candidate) == "noise":
            mids.append(candidate)
    top5_hits = [
        {
            "index": policy.mid_to_index[candidate],
            "mid": candidate,
            "label": policy.label_of(candidate),
            "kind": "noise",
            "prob": 1.0 - rank * 0.1,
        }
        for rank, candidate in enumerate(mids)
    ]
    record = {
        "uid": uid,
        "fs": fs,
        "length": fs,
        "path": str(wav),
        "manifest_schema_version": NOISE_MANIFEST_SCHEMA_VERSION,
        "balance_category_mid": mid,
        "balance_category": category_name,
        "balance_category_source": "noise_top5_hit",
        "category_name": category_name,
        "category_id": category_id,
        "dataset_id": dataset_id,
        "admission_allowed": True,
        "admission_decision": "accept_noise",
        "admission_reason_codes": ["admitted_noise"],
        "admission_policy_version": ADMISSION_POLICY_VERSION,
        "admission_policy_sha256": policy.sha256,
        "verification": {
            "c1_status": "passed",
            "c1_intelligibility": "noise",
            "c2_status": "passed",
            "c3_status": "passed",
            "d2_status": "passed",
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "score_contract": SCORE_CONTRACT,
            "class_count": CLASSIFIER_LABEL_COUNT,
            "label_space_sha256": policy.label_space_sha256,
            "artifact_fingerprints": {
                "voice_policy_sha256": policy.sha256,
                "audiotag_label_dict_sha256": policy.label_space_sha256,
                "audiotag_checkpoint_sha256": "2" * 64,
                "qwen_model_sha256": "3" * 64,
                "c1_prompt_sha256": "4" * 64,
            },
            "top5_hits": top5_hits,
        },
    }
    if roles is not None:
        record["roles"] = roles
    return record


def test_se_data_summary(tmp_path: Path):
    speech_wav = tmp_path / "speech.wav"
    noise_wav = tmp_path / "noise.wav"
    wind_wav = tmp_path / "wind.wav"
    rir_wav = tmp_path / "rir.wav"
    _write_wav(speech_wav, 0.1)
    _write_wav(noise_wav, 0.2)
    _write_wav(wind_wav, 0.3)
    _write_wav(rir_wav, 0.4)

    speech_scp = tmp_path / "speech.scp"
    noise_scp = tmp_path / "noise.jsonl"
    wind_scp = tmp_path / "wind.scp"
    rir_scp = tmp_path / "rir.scp"
    _write_scp(speech_scp, "speech", speech_wav)
    noise_records = [
        _noise_record("noise-1", 16000, noise_wav, 0, "Traffic noise, roadway noise", "set-a"),
        _noise_record("noise-2", 16000, noise_wav, 1, "Wind", "set-b", roles=["wind"]),
    ]
    noise_scp.write_text(
        "\n".join(json.dumps(record) for record in noise_records) + "\n",
        encoding="utf-8",
    )
    _write_scp(wind_scp, "wind", wind_wav)
    _write_scp(rir_scp, "rir", rir_wav)

    ds = SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=str(wind_scp),
        simulation_cfg={
            "prob_reverberation": 0.0,
            "prob_target_only": 0.1,
            "prob_noise_only": 0.2,
        },
        filter_by_duration=False,
    )

    summary = ds.get_data_summary("train", n_batches=10, dataloader_config={"batch_size": 4})
    assert isinstance(summary, DataSummary)
    assert summary.split == "train"
    assert summary.n_samples == len(ds)
    assert summary.n_batches == 10

    # Speech / noise / wind / rir are all reported.
    assert set(summary.sources) == {"speech", "noise", "wind", "rir"}
    speech_summary = summary.sources["speech"]
    assert speech_summary.clips == 1
    assert speech_summary.total_hours == pytest.approx(1.0 / 3600.0)
    assert 16000 in speech_summary.sr_counts

    # Noise category distribution excludes the embedded wind role record.
    assert summary.noise_categories is not None
    categories = {name: count for name, count, _ in summary.noise_categories}
    assert "Traffic noise, roadway noise" in categories
    assert "Wind" not in categories

    # Scenario probabilities are surfaced when configured.
    assert summary.scenario_probs is not None
    assert summary.scenario_probs.get("prob_target_only") == pytest.approx(0.1)
    assert summary.scenario_probs.get("prob_noise_only") == pytest.approx(0.2)


def test_tse_data_summary(tmp_path: Path):
    speech_wav = tmp_path / "speech.wav"
    noise_wav = tmp_path / "noise.wav"
    music_wav = tmp_path / "music.wav"
    rir_wav = tmp_path / "rir.wav"
    _write_wav(speech_wav, 0.1)
    _write_wav(noise_wav, 0.2)
    _write_wav(music_wav, 0.3)
    _write_wav(rir_wav, 0.4)

    # Strict JSONL speech manifest with speaker metadata.
    speech_jsonl = tmp_path / "speech.jsonl"
    speech_records = [
        {
            "uid": "spk1-utt1",
            "fs": 16000,
            "length": 16000,
            "path": str(speech_wav),
            "dataset_id": "test-set",
            "speaker_id": "spk1",
        },
        {
            "uid": "spk2-utt1",
            "fs": 16000,
            "length": 16000,
            "path": str(speech_wav),
            "dataset_id": "test-set",
            "speaker_id": "spk2",
        },
    ]
    speech_jsonl.write_text(
        "\n".join(json.dumps(record) for record in speech_records) + "\n",
        encoding="utf-8",
    )

    # Strict JSONL noise manifest with an embedded music role record.
    noise_jsonl = tmp_path / "noise.jsonl"
    noise_records = [
        _noise_record("noise-1", 16000, noise_wav, 0, "Traffic noise, roadway noise", "set-a"),
        _noise_record("music-1", 16000, music_wav, 1, "Music", "set-b", roles=["music"]),
    ]
    noise_jsonl.write_text(
        "\n".join(json.dumps(record) for record in noise_records) + "\n",
        encoding="utf-8",
    )

    rir_scp = tmp_path / "rir.scp"
    _write_scp(rir_scp, "rir", rir_wav)

    spk2enroll_json = tmp_path / "spk2enroll.json"
    spk2enroll_json.write_text(
        json.dumps(
            {
                "test-set": {
                    "spk1": [str(speech_wav)],
                    "spk2": [str(speech_wav), str(speech_wav)],
                }
            }
        ),
        encoding="utf-8",
    )

    ds = TSEDynamicMixingDataset(
        speech_scp=str(speech_jsonl),
        noise_scp=str(noise_jsonl),
        rir_scp=str(rir_scp),
        spk2enroll_json=str(spk2enroll_json),
        simulation_cfg={
            "prob_reverberation": 0.0,
            "prob_target_only": 0.0,
            "prob_target_noise": 1.0,
            "prob_noise_only": 0.0,
            "prob_wind_noise_sidechain": 0.0,
            "music_ratio": 0.5,
        },
        filter_by_duration=False,
    )

    summary = ds.get_data_summary("train", n_batches=5, dataloader_config={"batch_size": 2})
    assert isinstance(summary, DataSummary)
    assert "music" in summary.sources
    assert summary.sources["music"].clips == 1
    assert summary.sources["noise"].clips == 1  # music record moved out of noise pool
    # 干扰人池与目标语音共享 speech 池，摘要中单独列出。
    assert "interference" in summary.sources
    assert summary.sources["interference"].clips == 2

    extra = summary.extra or {}
    assert extra.get("speakers") == 2
    assert extra.get("interference_identities") == 2
    enroll = extra.get("enroll_candidates", {})
    assert enroll.get("min") == 1
    assert enroll.get("max") == 2
    assert enroll.get("mean") == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("rir_normalize", "normalization_mode"),
    [(False, "raw"), (True, "shared_full_peak")],
)
def test_aec_data_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rir_normalize: bool,
    normalization_mode: str,
):
    speech_wav = tmp_path / "speech.wav"
    noise_wav = tmp_path / "noise.wav"
    wind_wav = tmp_path / "wind.wav"
    rir_wav = tmp_path / "rir.wav"
    stereo_wav = tmp_path / "stereo.wav"
    _write_wav(speech_wav, 0.1)
    _write_wav(noise_wav, 0.2)
    _write_wav(wind_wav, 0.3)
    _write_wav(rir_wav, 0.4)
    # AEC default convention: ch0=echo, ch1=farend.
    _write_wav(stereo_wav, 0.5, channels=2)

    speech_scp = tmp_path / "speech.scp"
    noise_scp = tmp_path / "noise.jsonl"
    wind_scp = tmp_path / "wind.scp"
    rir_scp = tmp_path / "rir.scp"
    _write_scp(speech_scp, "speech", speech_wav)
    noise_records = [
        _noise_record("noise-1", 16000, noise_wav, 0, "Traffic noise, roadway noise", "set-a"),
    ]
    noise_scp.write_text(
        "\n".join(json.dumps(record) for record in noise_records) + "\n",
        encoding="utf-8",
    )
    _write_scp(wind_scp, "wind", wind_wav)
    _write_scp(rir_scp, "rir", rir_wav)

    pair_json = tmp_path / "pairs.json"
    pair_json.write_text(
        json.dumps({"pairs": [str(stereo_wav)]}),
        encoding="utf-8",
    )

    ds = AECDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=str(wind_scp),
        real_farend_echo_pair_json=str(pair_json),
        simulation_cfg={
            "prob_target_only": 0.1,
            "prob_noise_only": 0.1,
            "prob_echo_only": 0.1,
            "prob_noise_echo_only": 0.1,
            "prob_target_noise": 0.2,
            "prob_target_echo": 0.2,
            "prob_target_noise_echo": 0.2,
            "snr_low": -3.5,
            "snr_high": 17.5,
            "ser_range": [-8.0, 12.0],
            "prob_reverberation": 0.35,
            "early_rir_sec": 0.08,
            "target_type": "dry",
            "rir_normalize": rir_normalize,
        },
        filter_by_duration=False,
    )

    monkeypatch.setattr(
        ds,
        "_get_rng",
        lambda *_args, **_kwargs: pytest.fail("data summary must not advance RNG"),
    )
    monkeypatch.setattr(
        soundfile,
        "read",
        lambda *_args, **_kwargs: pytest.fail("data summary must not read audio"),
    )
    summary = ds.get_data_summary("train", n_batches=3, dataloader_config={"batch_size": 2})
    assert isinstance(summary, DataSummary)
    assert set(summary.sources) == {"speech", "noise", "wind", "rir", "ref", "echo"}

    extra = summary.extra or {}
    assert extra.get("farend_echo_pairs") == 1
    assert extra.get("aec_ref_echo_stereo_wav") is True
    assert extra.get("aec_task") == "joint"
    assert extra.get("simulation_nominal") == {
        "snr_range": [-3.5, 17.5],
        "ser_range": [-8.0, 12.0],
        "prob_reverberation": 0.35,
        "early_rir_sec": 0.08,
        "target_type": "dry",
        "rir_normalize": rir_normalize,
        "normalization_mode": normalization_mode,
    }

    assert summary.scenario_probs is not None
    assert summary.scenario_probs.get("prob_echo_only") == pytest.approx(0.1)
