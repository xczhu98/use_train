"""Tests: realized 元数据（管线末端统一重测电平）。

覆盖：
1. 无 clip/rescale 样本 realized.ser_db ≈ c3_effective_ser_db（±0.01 dB）；
2. Layer D 共同 guard 样本 realized.erl_db == c3_erl_db − 20·log10(scale)；
3. Layer F fixed_rail 样本 mic_clip_frame_ratio > 0；
4. realized_final 与手工重算 final 波形一致。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile
import torch

from tasks.aec.data.dynamic_dataset import DynamicMixingDataset as AECDynamicMixingDataset
from tasks.aec.data.dynamic_dataset import _source_volume_calibration_export


def _write_scp(path: Path, entries: dict[str, Path]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for uid, audio_path in entries.items():
            info = soundfile.info(str(audio_path))
            f.write(f"{uid} {info.samplerate} {info.frames} {audio_path}\n")


def _make_impulse_rir(path: Path, sr: int) -> None:
    rir = np.zeros(sr, dtype=np.float32)
    rir[0] = 1.0
    soundfile.write(path, rir, sr)


def _rms_dbfs(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).ravel()
    return 20.0 * float(np.log10(np.sqrt(np.mean(a ** 2))))


@pytest.fixture
def make_paths(tmp_path: Path):
    """构造 near 与 echo 同相正弦的样本，幅度/相位完全可控。

    speech = farend = A·sin(300Hz)：echo = farend 内容（delay=0、同相），
    mic 峰值 = A·(1 + 10^(−ser/20))，可精确预测。
    """

    def _build(
        speech_amp: float,
        ser_db: float,
        observation_clip: bool = False,
        sim_overrides: dict | None = None,
    ):
        sr = 16000
        duration = 1.0

        speech_dir = tmp_path / f"speech_{speech_amp}_{ser_db}_{observation_clip}"
        noise_dir = tmp_path / "noise"
        rir_dir = tmp_path / "rir"
        wind_dir = tmp_path / "wind"
        aec_dir = tmp_path / "aec"
        for d in (speech_dir, noise_dir, rir_dir, wind_dir, aec_dir):
            d.mkdir(parents=True, exist_ok=True)

        t = np.arange(int(duration * sr)) / sr
        speech = (speech_amp * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
        noise = (0.001 * np.sin(2.0 * np.pi * 700.0 * t)).astype(np.float32)

        speech_wav = speech_dir / "spk1_001.wav"
        noise_wav = noise_dir / "noise_001.wav"
        rir_wav = rir_dir / "rir_001.wav"
        wind_wav = wind_dir / "wind_001.wav"
        farend_wav = aec_dir / "farend_001.wav"
        echo_wav = aec_dir / "echo_001.wav"
        soundfile.write(speech_wav, speech, sr, subtype="FLOAT")
        soundfile.write(noise_wav, noise, sr)
        _make_impulse_rir(rir_wav, sr)
        soundfile.write(wind_wav, noise, sr)
        # farend/echo 与 speech 同内容 → echo 与 near 同相叠加
        soundfile.write(farend_wav, speech, sr, subtype="FLOAT")
        soundfile.write(echo_wav, speech, sr, subtype="FLOAT")
        stereo_wav = aec_dir / "stereo_001.wav"
        soundfile.write(stereo_wav, np.stack([speech, speech], axis=1), sr, subtype="FLOAT")

        speech_scp = tmp_path / f"speech_{speech_amp}_{ser_db}.scp"
        noise_scp = tmp_path / "noise.scp"
        rir_scp = tmp_path / "rir.scp"
        wind_scp = tmp_path / "wind.scp"
        pair_json = tmp_path / f"pair_{speech_amp}_{ser_db}.json"
        _write_scp(speech_scp, {"spk1_001": speech_wav})
        _write_scp(noise_scp, {"noise_001": noise_wav})
        _write_scp(rir_scp, {"rir_001": rir_wav})
        _write_scp(wind_scp, {"wind_001": wind_wav})
        with pair_json.open("w", encoding="utf-8") as f:
            json.dump({"pairs": [str(stereo_wav)]}, f)

        simulation_cfg = {
            "prob_target_only": 0.0,
            "prob_noise_only": 0.0,
            "prob_echo_only": 0.0,
            "prob_noise_echo_only": 0.0,
            "prob_target_noise": 0.0,
            "prob_target_echo": 1.0,
            "prob_target_noise_echo": 0.0,
            "snr_low": 60.0,
            "snr_high": 60.0,
            "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": observation_clip,
            "prob_clip": float(observation_clip),
            "clip_variants": ["fixed_rail"],
            "clip_variant_probs": {"fixed_rail": 1.0},
            "clip_pre_drive_db_range": [12.0, 12.0],
            "prob_reverberation": 0.0,
            "use_high_pass": False,
            "ser_range": [ser_db, ser_db],
            # 本文件不测试 ERL configured range；覆盖完整 physical range，
            # 保持旧 fallback 下同一 beta draw 与波形。
            "erl_range": [-60.0, 60.0],
        }
        if sim_overrides:
            simulation_cfg.update(sim_overrides)

        return {
            "speech_scp": str(speech_scp),
            "noise_scp": str(noise_scp),
            "rir_scp": str(rir_scp),
            "windnoise_scp": str(wind_scp),
            "simulation_cfg": simulation_cfg,
            "real_farend_echo_pair_json": str(pair_json),
            "aec_ref_echo_random_delay_prob": 0.0,
            "max_duration_seconds": 1.0,
            "target_sr": sr,
            "task": "joint",
            "strict_load": True,
        }

    return _build


def _run_one(paths):
    ds = AECDynamicMixingDataset(**paths)
    ds.set_epoch(0, base_seed=123)
    ds.export_simulation_stems = True
    _ = ds[0]
    return ds


def test_no_clip_sample_realized_ser_matches_drawn(make_paths):
    """无 clip/rescale（mic 峰值 0.6）：realized.ser_db ≈ c3_effective_ser_db。"""
    paths = make_paths(speech_amp=0.4, ser_db=6.0)
    ds = _run_one(paths)
    by_layer = ds._last_simulation_stems["simulation_by_layer"]
    layer_d = by_layer["D"]
    assert set(layer_d) == {"stage_order", "normalize", "guard"}
    assert layer_d["stage_order"] == ("D-normalize", "D-guard")
    assert layer_d["normalize"] == {
        "mode": "off",
        "target_peak": None,
        "scale": 1.0,
        "preserves_decomposition": True,
    }
    guard_keys = {"roles", "threshold", "peak", "scale", "applied"}
    assert set(layer_d["guard"]) == {"mic_related", "farend"}
    mic_guard = layer_d["guard"]["mic_related"]
    assert set(mic_guard) == guard_keys
    assert mic_guard["applied"] is False
    assert mic_guard["roles"] == ["mic", "nearend", "echo", "noise"]
    assert mic_guard["threshold"] == float(np.float32(0.99))
    assert mic_guard["scale"] == 1.0
    farend_guard = layer_d["guard"]["farend"]
    assert set(farend_guard) == guard_keys
    assert farend_guard["roles"] == ["farend"]
    assert not ({"D-normalize", "D-guard", "aec_context"} & set(by_layer))
    trace = ds._last_simulation_trace
    trace_metadata = trace.metadata
    scene_metadata = trace.sources["scene"].metadata
    canonical_layer_d = {**layer_d, "stage_order": list(layer_d["stage_order"])}
    for metadata in (trace_metadata, scene_metadata):
        assert metadata["layer_d_facts"] == canonical_layer_d
        assert not ({"layer_d_normalize", "layer_d", "layer_d_normalize_facts"} & set(metadata))
        assert "farend_guard" not in metadata["aec_context"]
        json.dumps(dict(metadata), allow_nan=False)
    realized = by_layer["realized"]
    assert realized["measure_point"] == "post_dataset_post_mix"
    assert realized["ser_db"] == pytest.approx(
        by_layer["C"]["c3_effective_ser_db"], abs=0.01
    )
    assert realized["erl_db"] == pytest.approx(by_layer["C"]["c3_erl_db"], abs=0.01)
    assert "farend_rescaled" not in realized


def test_layer_d_common_rescale_shifts_realized_erl(make_paths):
    """D 共同 guard 后 ERL 反映五路耦合 bundle 的统一缩放。"""
    speech_amp = 1.2
    paths = make_paths(speech_amp=speech_amp, ser_db=20.0)
    ds = _run_one(paths)
    by_layer = ds._last_simulation_stems["simulation_by_layer"]
    assert by_layer["D"]["guard"]["mic_related"]["applied"] is True
    realized = by_layer["realized"]
    scale = by_layer["D"]["guard"]["mic_related"]["scale"]
    assert realized["erl_db"] == pytest.approx(by_layer["C"]["c3_erl_db"], abs=0.01)
    # echo 被等比压低 20·log10(scale) dB
    drawn_echo_db = by_layer["C"]["c3_far_db"] - by_layer["C"]["c3_erl_db"]
    assert realized["echo_dbfs_rms"] == pytest.approx(
        drawn_echo_db + 20.0 * np.log10(scale), abs=0.01
    )


def test_layer_f_fixed_rail_reports_clip_ratio(make_paths):
    """F fixed_rail + 12 dB pre-drive 产生可观测平台。"""
    paths = make_paths(speech_amp=0.8, ser_db=6.0, observation_clip=True)
    ds = _run_one(paths)
    by_layer = ds._last_simulation_stems["simulation_by_layer"]
    assert by_layer["F"]["cpu_applied_ops"] == ["clip"]
    clipping = by_layer["F"]["observation_clipping"]
    assert clipping["event"] is True
    assert clipping["applied"] is True
    assert clipping["actual_changed"] is True
    assert clipping["skipped"] is False
    realized = by_layer["realized"]
    assert realized["mic_clip_frame_ratio"] > 0.0
    assert realized["mic_peak_abs"] <= 1.0
    assert realized["decomposable_reason_codes"] == [
        "layer_f_nonlinear",
        "final_guard_observation_only_scale",
    ]


def test_realized_final_matches_manual_recompute(make_paths):
    """realized_final 与手工重算 final 波形一致。"""
    paths = make_paths(speech_amp=0.4, ser_db=6.0)
    ds = _run_one(paths)
    stems = ds._last_simulation_stems
    rf = stems["realized_final"]
    assert rf["measure_point"] == "final"
    final_clean = np.asarray(stems["final_clean"], dtype=np.float64).ravel()
    final_noisy = np.asarray(stems["final_noisy"], dtype=np.float64).ravel()
    assert rf["clean_dbfs_rms"] == pytest.approx(_rms_dbfs(final_clean), abs=1e-3)
    assert rf["mic_dbfs_rms"] == pytest.approx(_rms_dbfs(final_noisy), abs=1e-3)
    # 自洽：ser_db = clean − echo（echo 取自同步到最终窗口的 ctx）
    if rf["ser_db"] is not None:
        assert rf["ser_db"] == pytest.approx(
            rf["clean_dbfs_rms"] - rf["echo_dbfs_rms"], abs=1e-6
        )


def test_acoustic_audit_schema_and_realized_snr(make_paths):
    paths = make_paths(
        speech_amp=0.4,
        ser_db=6.0,
        sim_overrides={
            "prob_target_echo": 0.0,
            "prob_target_noise_echo": 1.0,
            "snr_low": 20.0,
            "snr_high": 20.0,
        },
    )
    ds = _run_one(paths)
    stems = ds._last_simulation_stems
    layer_c = stems["simulation_by_layer"]["C"]
    audit = layer_c["acoustic_audit"]

    assert set(audit) == {"candidate", "nominal", "calibration", "c0_full_early"}
    assert audit["candidate"]["primary_snr_db"] == pytest.approx(20.0)
    assert audit["nominal"]["selected_primary_snr_db"] == pytest.approx(20.0)
    # aggregate 重测会在缩放后的 noise 上重新做 active detection，允许门限引入小偏差。
    assert audit["calibration"]["layer_c_aggregate_realized_snr_db"] == pytest.approx(
        20.0, abs=0.35
    )
    source_audits = audit["calibration"]["source_level_audits"]
    assert source_audits["additive"][0]["source"] == "single"
    assert source_audits["additive"][0]["level_audit"]["name"] == "mixture.snr"
    snr_level_audit = audit["calibration"]["snr_level_audit"]
    assert snr_level_audit["calibration_realized_db"] == pytest.approx(20.0, abs=0.35)
    assert snr_level_audit["final_realized_db"] is None
    assert snr_level_audit["applicable"] is False
    assert snr_level_audit["reason"] \
        == "final_available_in_simulation_by_layer_realized"
    assert layer_c["source_volume_calibration"] == {
        "target_db": None,
        "target_requested_db": None,
        "reference_db_before": None,
        "realized_after_db": None,
        "gain": None,
        "gain_clipped": None,
        "status": "disabled",
        "reason": None,
        "forced_gain": False,
        "reference": None,
        "stage": None,
    }
    c0 = audit["c0_full_early"]
    assert c0["full_to_early_peak_ratio"] == pytest.approx(1.0)
    assert c0["full_to_early_rms_ratio"] == pytest.approx(1.0)

    # nominal 键是唯一的主 SNR 记录；误名为 mic-chain 的旧键与其 deprecated 标记已移除。
    assert layer_c["c1_c2_nominal_snr_db"] == pytest.approx(20.0)
    assert "c1_c2_mic_chain_snr_db" not in layer_c
    assert "deprecated_fields" not in layer_c
    assert layer_c["c1_c2_additive_snr_db_each_mix"] == [20.0]

    realized = stems["simulation_by_layer"]["realized"]
    realized_final = stems["realized_final"]
    level_audits = layer_c["level_audits"]
    assert set(level_audits) == {"snr", "ser", "erl"}
    assert level_audits["snr"]["candidate_drawn_db"] == pytest.approx(20.0)
    assert level_audits["snr"]["calibration_realized_db"] == pytest.approx(
        20.0, abs=0.35
    )
    assert level_audits["snr"]["final_realized_db"] == pytest.approx(
        realized["snr_db"]
    )
    assert level_audits["ser"]["calibration_realized_db"] == pytest.approx(
        layer_c["c3_effective_ser_db"], abs=0.01
    )
    assert level_audits["erl"]["calibration_realized_db"] == pytest.approx(
        layer_c["c3_erl_db"], abs=0.01
    )
    assert realized["snr_db"] == pytest.approx(20.0, abs=0.35)
    assert realized_final["snr_db"] == pytest.approx(20.0, abs=0.35)
    assert realized["reconstruction_residual"]["rms"] == pytest.approx(0.0, abs=1e-6)
    assert realized_final["reconstruction_residual"]["rms"] == pytest.approx(0.0, abs=1e-6)
    json.dumps(audit, allow_nan=False)
    json.dumps(realized, allow_nan=False)
    json.dumps(realized_final, allow_nan=False)


def test_post_rir_early_calibration_metadata_and_linear_ratios(make_paths):
    """固定 early target 后再标定 noise/echo，且 C-0 缓存保存 gain 后波形。"""
    target_db = -26.0
    paths = make_paths(
        speech_amp=0.4,
        ser_db=6.0,
        sim_overrides={
            "prob_target_echo": 0.0,
            "prob_target_noise_echo": 1.0,
            "prob_reverberation": 1.0,
            "early_rir_sec": 0.05,
            "target_type": "early",
            "source_volume_normalization_enabled": True,
            "source_volume_db_range": [target_db, target_db],
            "source_volume_use_rms": False,
            "snr_low": 20.0,
            "snr_high": 20.0,
            "erl_range": [10.0, 10.0],
            "far_db_range": [-40.0, 0.0],
        },
    )
    rir_path = Path(paths["rir_scp"]).read_text(encoding="utf-8").strip().split()[-1]
    rir = np.zeros(16000, dtype=np.float32)
    rir[0] = 1.0
    rir[1600] = 0.5  # 100 ms 晚反射：排除在 50 ms early target 之外。
    soundfile.write(rir_path, rir, 16000, subtype="FLOAT")

    ds = AECDynamicMixingDataset(**paths)
    ds.set_epoch(0, base_seed=123)
    ds.export_simulation_stems = True
    sample = ds[0]
    assert len(sample) == 11

    stems = ds._last_simulation_stems
    by_layer = stems["simulation_by_layer"]
    calibration = by_layer["C"]["source_volume_calibration"]
    assert {
        "target_db",
        "target_requested_db",
        "reference_db_before",
        "realized_after_db",
        "gain",
        "gain_clipped",
        "status",
        "reason",
        "forced_gain",
        "reference",
        "stage",
    } <= set(calibration)
    assert calibration["reference"] == "early_rir"
    assert calibration["stage"] == "post_rir_pre_noise_echo"
    for key in (
        "target_db",
        "target_requested_db",
        "reference_db_before",
        "realized_after_db",
        "gain",
    ):
        assert np.isfinite(calibration[key]), key
    assert calibration["target_db"] == pytest.approx(target_db)
    assert calibration["target_requested_db"] == pytest.approx(target_db)
    assert calibration["realized_after_db"] == pytest.approx(target_db, abs=1e-3)
    assert calibration["gain"] == pytest.approx(
        10.0
        ** (
            (calibration["realized_after_db"] - calibration["reference_db_before"])
            / 20.0
        )
    )
    assert calibration["gain_clipped"] is False
    assert calibration["status"] == "applied"
    assert calibration["forced_gain"] is False
    json.dumps(calibration, allow_nan=False)
    json.dumps(by_layer, allow_nan=False)

    # 仅从公开 export stems 恢复 C-0 full/early，不截获 ``_mix_acoustic`` 私有返回。
    noise = np.asarray(stems["layer_d_noise"], dtype=np.float64)
    echo = np.asarray(stems["layer_d_echo"], dtype=np.float64)
    farend = np.asarray(stems["farend_ref"], dtype=np.float64)
    full = np.asarray(stems["layer_d_noisy"], dtype=np.float64) - noise - echo
    early = np.asarray(stems["final_clean"], dtype=np.float64)
    assert np.max(np.abs(early)) == pytest.approx(10.0 ** (target_db / 20.0), abs=1e-5)

    c0_audit = by_layer["C"]["acoustic_audit"]["c0_full_early"]
    assert c0_audit["full_to_early_peak_ratio"] == pytest.approx(
        np.max(np.abs(full)) / np.max(np.abs(early)), rel=1e-6
    )
    assert c0_audit["full_to_early_rms_ratio"] == pytest.approx(
        np.sqrt(np.mean(full ** 2)) / np.sqrt(np.mean(early ** 2)), rel=1e-6
    )
    assert c0_audit["full_to_early_peak_ratio"] > 1.4

    # C-1 noise 与 C-aec echo 均在 early target 固定后定标；直接用 gain 后
    # C-0 early reference 重算，避免 joint full/early target mismatch 门控伪装比值。
    assert _rms_dbfs(early) - _rms_dbfs(noise) == pytest.approx(20.0, abs=0.35)
    assert _rms_dbfs(early) - _rms_dbfs(echo) == pytest.approx(6.0, abs=0.01)
    assert _rms_dbfs(farend) - _rms_dbfs(echo) == pytest.approx(10.0, abs=0.01)

    realized = by_layer["realized"]
    assert realized["decomposable"] is False
    assert realized["decomposable_reason_codes"] == ["joint_target_rir_mismatch"]
    assert realized["ser_db"] is None
    assert realized["snr_db"] is None
    assert realized["erl_db"] == pytest.approx(10.0, abs=0.01)
    assert by_layer["D"]["guard"]["mic_related"]["applied"] is False
    assert calibration["stage"] == "post_rir_pre_noise_echo"


def test_source_volume_calibration_reports_clipped_gain_via_public_export(make_paths):
    target_db = -100.0
    paths = make_paths(
        speech_amp=0.4,
        ser_db=6.0,
        sim_overrides={
            "source_volume_normalization_enabled": True,
            "source_volume_db_range": [target_db, target_db],
            "source_volume_use_rms": False,
        },
    )
    ds = _run_one(paths)

    calibration = ds._last_simulation_stems["simulation_by_layer"]["C"][
        "source_volume_calibration"
    ]
    assert calibration["target_requested_db"] == pytest.approx(target_db)
    assert calibration["gain"] == pytest.approx(1e-3)
    assert calibration["realized_after_db"] > target_db
    assert calibration["gain_clipped"] is True
    assert calibration["status"] == "applied"
    assert calibration["reason"] == "gain_clamped"
    json.dumps(calibration, allow_nan=False)


def test_source_volume_calibration_preserves_all_zero_skipped_via_public_export(
    make_paths,
):
    target_db = -24.0
    paths = make_paths(
        speech_amp=0.4,
        ser_db=6.0,
        sim_overrides={
            "source_volume_normalization_enabled": True,
            "source_volume_db_range": [target_db, target_db],
            "source_volume_use_rms": False,
        },
    )
    speech_path = Path(paths["speech_scp"]).read_text(encoding="utf-8").split()[-1]
    soundfile.write(speech_path, np.zeros(16000, dtype=np.float32), 16000, subtype="FLOAT")

    ds = _run_one(paths)
    calibration = ds._last_simulation_stems["simulation_by_layer"]["C"][
        "source_volume_calibration"
    ]

    assert calibration["target_db"] == pytest.approx(target_db)
    assert calibration["target_requested_db"] == pytest.approx(target_db)
    assert calibration["reference_db_before"] == pytest.approx(-240.0)
    assert calibration["realized_after_db"] == pytest.approx(-240.0)
    assert calibration["gain"] == pytest.approx(1e3)
    assert calibration["gain_clipped"] is True
    assert calibration["status"] == "skipped"
    assert calibration["reason"] == "reference_all_zero"
    json.dumps(calibration, allow_nan=False)


def test_source_volume_calibration_preserves_float32_safe_clamp_via_public_export(
    make_paths,
):
    target_db = 780.0
    paths = make_paths(
        speech_amp=0.4,
        ser_db=6.0,
        sim_overrides={
            "prob_target_only": 1.0,
            "prob_target_echo": 0.0,
            "source_volume_normalization_enabled": True,
            "source_volume_db_range": [target_db, target_db],
            "source_volume_use_rms": False,
        },
    )
    speech_path = Path(paths["speech_scp"]).read_text(encoding="utf-8").split()[-1]
    signal_peak = np.float32(np.finfo(np.float32).max / 2.0)
    speech = np.full(16000, signal_peak, dtype=np.float32)
    speech[1::2] *= -1.0
    soundfile.write(speech_path, speech, 16000, subtype="FLOAT")

    ds = _run_one(paths)
    calibration = ds._last_simulation_stems["simulation_by_layer"]["C"][
        "source_volume_calibration"
    ]

    expected_gain = float(np.finfo(np.float32).max) / float(signal_peak)
    assert calibration["target_db"] == pytest.approx(target_db)
    assert calibration["target_requested_db"] == pytest.approx(target_db)
    assert calibration["realized_after_db"] == pytest.approx(
        calibration["reference_db_before"] + 20.0 * np.log10(expected_gain)
    )
    assert calibration["gain"] == pytest.approx(expected_gain)
    assert 1e-3 < calibration["gain"] < 1e3
    assert calibration["gain_clipped"] is True
    assert calibration["status"] == "applied"
    assert calibration["reason"] == "shared_stem_float32_limit"
    json.dumps(calibration, allow_nan=False)


def test_source_volume_calibration_rejects_non_strict_bool_and_unknown_status():
    calibration = _source_volume_calibration_export(
        {
            "target_db": -24.0,
            "target_db_requested": -24.0,
            "reference_db_before": -30.0,
            "reference_db_after": -24.0,
            "gain": 2.0,
            "gain_clipped": 1,
            "forced_gain": "false",
            "status": "invented",
            "reason": 7,
        }
    )

    assert calibration["gain_clipped"] is None
    assert calibration["forced_gain"] is None
    assert calibration["status"] == "invalid"
    assert calibration["reason"] is None
    json.dumps(calibration, allow_nan=False)


def test_source_volume_calibration_legacy_sidecar_uses_fallbacks_only_when_missing():
    calibration = _source_volume_calibration_export(
        {
            "target_db": -24.0,
            "reference_db_before": -30.0,
            "gain": 10.0 ** (6.0 / 20.0),
            "forced_gain": False,
        }
    )

    assert calibration["target_db"] == pytest.approx(-24.0)
    assert calibration["target_requested_db"] == pytest.approx(-24.0)
    assert calibration["realized_after_db"] == pytest.approx(-24.0)
    assert calibration["gain_clipped"] is False
    assert calibration["status"] == "applied"
    json.dumps(calibration, allow_nan=False)


def test_source_volume_calibration_preserves_finite_post_augment_extensions():
    calibration = _source_volume_calibration_export(
        {
            "target_db": -24.0,
            "target_db_requested": -24.0,
            "reference_db_before": -30.0,
            "reference_db_after": -24.0,
            "gain": 2.0,
            "gain_clipped": False,
            "forced_gain": False,
            "status": "applied",
            "post_augment_schema_version": 2,
            "producer_extension": {
                "measurement": "active_rms",
                "bounds_db": [-45.0, -10.0],
                "invalid_probe": np.nan,
            },
        }
    )

    assert calibration["post_augment_schema_version"] == 2
    assert calibration["producer_extension"] == {
        "measurement": "active_rms",
        "bounds_db": [-45.0, -10.0],
        "invalid_probe": None,
    }
    json.dumps(calibration, allow_nan=False)


def test_export_only_audit_is_not_called_and_training_tuple_is_unchanged(make_paths):
    paths = make_paths(
        speech_amp=0.4,
        ser_db=6.0,
        sim_overrides={
            "prob_target_echo": 0.0,
            "prob_target_noise_echo": 1.0,
            "snr_low": 20.0,
            "snr_high": 20.0,
        },
    )
    ds = AECDynamicMixingDataset(**paths)
    ds.set_epoch(0, base_seed=123)

    with patch(
        "tasks.aec.data.dynamic_dataset._active_power_db_ratio",
        side_effect=AssertionError("export-only audit ran on training path"),
    ), patch(
        "tasks.aec.data.dynamic_dataset._peak_rms_ratio_audit",
        side_effect=AssertionError("export-only audit ran on training path"),
    ), patch(
        "tasks.aec.data.dynamic_dataset._measure_realized_levels",
        side_effect=AssertionError("export-only audit ran on training path"),
    ):
        sample = ds[0]

    assert len(sample) == 11
    nearend, mic, farend, sr, length, error, error_mask, *flags = sample
    assert all(isinstance(value, torch.Tensor) for value in (nearend, mic, farend, length, error))
    assert sr == 16000
    assert error_mask is False
    assert len(flags) == 4
    assert ds._last_simulation_stems is None

    ds.export_simulation_stems = True
    ds.set_epoch(0, base_seed=123)
    exported_sample = ds[0]
    assert len(exported_sample) == len(sample)
    for expected, actual in zip(sample, exported_sample):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        else:
            assert actual == expected


pytestmark = [pytest.mark.unit]
