"""Integration tests: aec_echo_clipping wired end-to-end through DynamicMixingDataset._run."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import ANY, patch

import numpy as np
import pytest
import soundfile

from tasks.aec.data.dynamic_dataset import DynamicMixingDataset as AECDynamicMixingDataset
from core.data.dynamic_dataset_base import LayerEOperationResult, LayerEResult


def _write_scp(path: Path, entries: dict[str, Path]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for uid, audio_path in entries.items():
            info = soundfile.info(str(audio_path))
            f.write(f"{uid} {info.samplerate} {info.frames} {audio_path}\n")


def _make_wav(path: Path, duration: float, sr: int, *, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    signal = np.sin(2.0 * np.pi * 300.0 * t).astype(np.float32)
    signal += 0.01 * rng.standard_normal(signal.shape).astype(np.float32)
    signal = np.clip(signal, -0.99, 0.99)
    soundfile.write(path, signal, sr)


def _make_impulse_rir(path: Path, sr: int) -> None:
    rir = np.zeros(sr, dtype=np.float32)
    rir[0] = 1.0
    soundfile.write(path, rir, sr)


def _mock_fixed_rail(data, _sr, _cfg, rng=None, facts_out=None):
    del rng
    source = np.asarray(data, dtype=np.float32)
    output = np.clip(source * np.float32(4.0), -1.0, 1.0).astype(np.float32)
    changed = int(np.count_nonzero(output != source))
    if facts_out is not None:
        facts_out.update(
            {
                "variant": "fixed_rail",
                "pre_drive_db": 12.0,
                "rail": 1.0,
                "pre_peak": float(np.max(np.abs(source))),
                "post_peak": float(np.max(np.abs(output))),
                "changed_sample_count": changed,
                "changed_sample_ratio": changed / source.size,
                "applied": changed > 0,
            }
        )
    return output


def _mock_identity_clip(data, _sr, _cfg, rng=None, facts_out=None):
    del rng
    source = np.asarray(data, dtype=np.float32)
    if facts_out is not None:
        facts_out.update(
            {
                "variant": "fixed_rail",
                "pre_drive_db": 0.0,
                "rail": 1.0,
                "pre_peak": float(np.max(np.abs(source))),
                "post_peak": float(np.max(np.abs(source))),
                "changed_sample_count": 0,
                "changed_sample_ratio": 0.0,
                "applied": False,
            }
        )
    return source


@pytest.fixture
def aec_paths(tmp_path: Path):
    """Build synthetic speech/noise/rir/wind/farend/echo files and return the
    matching DynamicMixingDataset constructor kwargs (minus aec_echo_clipping),
    so tests can vary just the aec_echo_clipping config, including ones
    expected to fail construction-time validation.
    """
    sr = 16000
    duration = 2.0

    speech_dir = tmp_path / "speech"
    noise_dir = tmp_path / "noise"
    rir_dir = tmp_path / "rir"
    wind_dir = tmp_path / "wind"
    aec_dir = tmp_path / "aec"
    for d in (speech_dir, noise_dir, rir_dir, wind_dir, aec_dir):
        d.mkdir()

    speech_wav = speech_dir / "spk1_001.wav"
    noise_wav = noise_dir / "noise_001.wav"
    rir_wav = rir_dir / "rir_001.wav"
    wind_wav = wind_dir / "wind_001.wav"
    farend_wav = aec_dir / "farend_001.wav"
    echo_wav = aec_dir / "echo_001.wav"

    _make_wav(speech_wav, duration, sr, seed=10)
    _make_wav(noise_wav, duration, sr, seed=20)
    _make_impulse_rir(rir_wav, sr)
    _make_wav(wind_wav, duration, sr, seed=30)
    _make_wav(farend_wav, duration, sr, seed=40)
    farend, _ = soundfile.read(farend_wav, dtype="float32")
    echo = np.zeros_like(farend)
    delay = int(0.05 * sr)
    echo[delay:] = farend[:-delay] * 0.3
    soundfile.write(echo_wav, echo, sr)
    stereo_wav = aec_dir / "stereo_001.wav"
    soundfile.write(stereo_wav, np.stack([echo, farend], axis=1), sr)

    speech_scp = tmp_path / "speech.scp"
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    wind_scp = tmp_path / "wind.scp"
    pair_json = tmp_path / "farend_echo.json"
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
        "snr_low": 5.0,
        "snr_high": 20.0,
        "prob_channel_enabled": False,
        "noisy_post_mix_degradations_enabled": True,
        "prob_clip": 1.0,
        "clip_variants": ["fixed_rail"],
        "clip_variant_probs": {"fixed_rail": 1.0},
        "clip_pre_drive_db_range": [12.0, 12.0],
        "prob_reverberation": 0.0,
        "ser_range": [-10.0, 10.0],
    }

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


@pytest.fixture
def echo_clip_dataset(aec_paths):
    """使用 canonical path-local 配置构造必触发 clipping 的 AEC Dataset。"""
    ds = AECDynamicMixingDataset(
        **aec_paths,
        aec_echo_clipping={
            "enabled": True,
            "prob": 1.0,
            "path_probs": {"post_ser_distortion": 1.0},
            "paths": {
                "post_ser_distortion": {
                    "variant_probs": {"hard": 1.0},
                    "peak_relative_threshold_range": [0.5, 0.5],
                },
                "pre_ser_overload": {
                    "variant_probs": {"fixed_rail": 1.0},
                    "overload_boost_db_range": [6.0, 6.0],
                    "rail_dbfs_range": [-3.0, -3.0],
                },
                "synced_ref_clip": {
                    "variant_probs": {"fixed_rail": 1.0},
                    "rail_dbfs_range": [-12.0, -12.0],
                },
            },
        },
    )
    ds.set_epoch(0, base_seed=456)
    ds.export_simulation_stems = True
    return ds


def test_canonical_path_local_config_is_wired(echo_clip_dataset):
    """构造期应保留三条路径各自的参数，不再展开为共享扁平字段。"""
    assert echo_clip_dataset._aec_echo_clipping_paths == {
        "post_ser_distortion": {
            "variant_probs": {"hard": 1.0},
            "peak_relative_threshold_range": (0.5, 0.5),
        },
        "pre_ser_overload": {
            "variant_probs": {"fixed_rail": 1.0},
            "rail_dbfs_range": (-3.0, -3.0),
            "overload_boost_db_range": (6.0, 6.0),
        },
        "synced_ref_clip": {
            "variant_probs": {"fixed_rail": 1.0},
            "rail_dbfs_range": (-12.0, -12.0),
        },
    }
    assert not hasattr(echo_clip_dataset, "_aec_echo_clipping_variant_probs")
    assert not hasattr(echo_clip_dataset, "_aec_echo_clipping_threshold_frac_range")


@pytest.mark.parametrize(
    "path_probs,expected_variant,expected_rail_dbfs",
    [
        ({"post_ser_distortion": 1.0}, "hard", None),
        ({"pre_ser_overload": 1.0}, "fixed_rail", -3.0),
        ({"synced_ref_clip": 1.0}, "fixed_rail", -12.0),
    ],
)
def test_each_path_runs_and_reports_real_changes(
    echo_clip_dataset, path_probs, expected_variant, expected_rail_dbfs
):
    echo_clip_dataset._aec_echo_clipping_path_probs = path_probs
    _ = echo_clip_dataset[0]
    audit = echo_clip_dataset._last_simulation_stems["simulation_by_layer"]["C"]["c3_echo_clipping"]
    assert audit["path"] == next(iter(path_probs))
    assert audit["variant"] == expected_variant
    assert audit["echo_applied"] is (audit["peak_after"] != audit["peak_before"])
    assert audit["applied"] is (audit["echo_applied"] or audit["farend_applied"])
    assert audit["any_applied"] is audit["applied"]
    assert audit["applied"] is True

    if audit["path"] == "post_ser_distortion":
        assert audit["threshold_frac"] == pytest.approx(0.5)
        assert audit["threshold_abs"] == pytest.approx(0.5 * audit["peak_before"])
        assert audit["rail_dbfs"] is None
    else:
        assert audit["rail_dbfs"] == pytest.approx(expected_rail_dbfs)
        assert audit["rail"] == pytest.approx(10.0 ** (expected_rail_dbfs / 20.0))
        assert audit["threshold_frac"] is None
    if audit["path"] == "pre_ser_overload":
        assert audit["boost_db"] == pytest.approx(6.0)
    if audit["path"] == "synced_ref_clip":
        assert audit["shared_variant"] == "fixed_rail"
        assert audit["farend_variant"] == "fixed_rail"


@pytest.mark.parametrize(
    "path_probs,expected_cp",
    [
        ({"post_ser_distortion": 1.0}, "playback_input_reference"),
        ({"pre_ser_overload": 1.0}, "playback_input_reference"),
        ({"synced_ref_clip": 1.0}, "shared_nonlinear_reference"),
    ],
)
def test_capture_point_label_follows_path(echo_clip_dataset, path_probs, expected_cp):
    """JAE-M11：capture point 语义标签随 clipping 路径正确标注（协议见 echo_clipping.py）。"""
    echo_clip_dataset._aec_echo_clipping_path_probs = path_probs
    _ = echo_clip_dataset[0]
    audit = echo_clip_dataset._last_simulation_stems["simulation_by_layer"]["C"]["c3_echo_clipping"]
    assert audit["capture_point"] == expected_cp


def test_capture_point_label_when_clipping_disabled(echo_clip_dataset):
    """JAE-M11：未触发 clipping 时 reference 为播放链输入侧参考（主语义）。"""
    echo_clip_dataset._aec_echo_clipping_enabled = False
    _ = echo_clip_dataset[0]
    audit = echo_clip_dataset._last_simulation_stems["simulation_by_layer"]["C"]["c3_echo_clipping"]
    assert audit["path"] is None
    assert audit["capture_point"] == "playback_input_reference"


def test_capture_point_of_mapping():
    """JAE-M11：路径→语义标签映射的单值断言（含 None 与未知路径回退）。"""
    from tasks.aec.data.echo_clipping import capture_point_of

    assert capture_point_of(None) == "playback_input_reference"
    assert capture_point_of("post_ser_distortion") == "playback_input_reference"
    assert capture_point_of("pre_ser_overload") == "playback_input_reference"
    assert capture_point_of("synced_ref_clip") == "shared_nonlinear_reference"


def test_pre_ser_overload_skips_random_ser_draw(echo_clip_dataset):
    echo_clip_dataset._aec_echo_clipping_path_probs = {"pre_ser_overload": 1.0}
    _ = echo_clip_dataset[0]
    audit = echo_clip_dataset._last_simulation_stems["simulation_by_layer"]["C"]
    assert audit["c3_ser_db_drawn"] is None  # SER draw was skipped, not merely unused
    assert audit["c3_ser_scaling_used"] is False
    assert audit["c3_echo_clipping"]["boost_db"] is not None
    assert audit["c3_echo_clipping"]["effective_ser_db"] is not None


def test_pre_ser_overload_real_silence_is_noop(echo_clip_dataset):
    """真实 overload 实现收到 silent echo 时不得捏造 boost、rail 或 applied。"""
    silence = np.zeros(16000, dtype=np.float32)
    echo_clip_dataset._aec_echo_clipping_path_probs = {"pre_ser_overload": 1.0}
    with patch.object(
        echo_clip_dataset,
        "_load_farend_echo_pair_aligned",
        return_value=(silence.copy(), silence.copy(), {}),
    ):
        _ = echo_clip_dataset[0]

    audit = echo_clip_dataset._last_simulation_stems["simulation_by_layer"]["C"]["c3_echo_clipping"]
    assert audit["path"] == "pre_ser_overload"
    assert audit["variant"] == "none"
    assert audit["boost_db"] is None
    assert audit["rail_dbfs"] is None
    assert audit["rail"] is None
    assert audit["peak_before"] == pytest.approx(0.0)
    assert audit["peak_after"] == pytest.approx(0.0)
    assert audit["rms_before"] == pytest.approx(0.0)
    assert audit["rms_after"] == pytest.approx(0.0)
    assert audit["echo_applied"] is False
    assert audit["farend_applied"] is False
    assert audit["any_applied"] is False
    assert audit["applied"] is False


@pytest.mark.parametrize(
    "echo_level,rail_hit",
    [(0.05, False), (0.5, True)],
    ids=["boost-only", "rail-hit"],
)
def test_pre_ser_overload_reports_boost_only_and_rail_hit_facts(
    echo_clip_dataset, echo_level, rail_hit
):
    """最终公开 facts 应区分仅 boost 改写与 boost 后命中绝对 rail。"""
    farend = np.full(16000, 0.1, dtype=np.float32)
    echo = np.full(16000, echo_level, dtype=np.float32)
    echo_clip_dataset._aec_echo_clipping_path_probs = {"pre_ser_overload": 1.0}
    with patch.object(
        echo_clip_dataset,
        "_load_farend_echo_pair_aligned",
        return_value=(farend, echo, {}),
    ):
        _ = echo_clip_dataset[0]

    audit = echo_clip_dataset._last_simulation_stems["simulation_by_layer"]["C"]["c3_echo_clipping"]
    rail = 10.0 ** (-3.0 / 20.0)
    boosted_peak = echo_level * (10.0 ** (6.0 / 20.0))
    assert audit["path"] == "pre_ser_overload"
    assert audit["variant"] == "fixed_rail"
    assert audit["boost_db"] == pytest.approx(6.0)
    assert audit["rail_dbfs"] == pytest.approx(-3.0)
    assert audit["rail"] == pytest.approx(rail)
    assert audit["peak_before"] == pytest.approx(echo_level)
    assert audit["peak_after"] == pytest.approx(min(boosted_peak, rail))
    assert (boosted_peak > audit["rail"]) is rail_hit
    assert audit["echo_applied"] is True
    assert audit["farend_applied"] is False
    assert audit["any_applied"] is True
    assert audit["applied"] is True
    echo_facts = audit["echo_rail_facts"]
    assert echo_facts["sample_count"] == 16000
    assert echo_facts["clipped_sample_count"] == (16000 if rail_hit else 0)
    assert echo_facts["clipped_sample_ratio"] == pytest.approx(1.0 if rail_hit else 0.0)
    assert echo_facts["rail_hit"] is rail_hit
    assert audit["farend_rail_facts"] is None
    assert audit["echo_rail_hit"] is rail_hit
    assert audit["farend_rail_hit"] is None
    assert audit["rail_hit"] is rail_hit
    assert audit["clipped_sample_count"] == (16000 if rail_hit else 0)
    assert audit["rail_sample_count"] == 16000
    assert audit["clipped_sample_ratio"] == pytest.approx(1.0 if rail_hit else 0.0)


def test_c_aec_echo_clip_can_coexist_with_layer_f_observation_clip(
    echo_clip_dataset,
):
    """C-aec 改写 echo 不再抑制 Layer F；F 仍负责 mic observation clipping。"""
    echo_clip_dataset._aec_echo_clipping_path_probs = {"post_ser_distortion": 1.0}
    with patch("core.data.post_augment.speaker_distortion") as speaker_distortion:
        speaker_distortion.side_effect = _mock_fixed_rail
        _ = echo_clip_dataset[0]

    speaker_distortion.assert_called_once()
    by_layer = echo_clip_dataset._last_simulation_stems["simulation_by_layer"]
    assert by_layer["C"]["c3_echo_clipping"]["echo_applied"] is True
    assert "clip" in by_layer["F"]["cpu_applied_ops"]
    assert float(np.max(np.abs(echo_clip_dataset._last_simulation_stems["final_noisy"]))) <= 1.0


def test_layer_e_actual_clipping_suppresses_layer_f(echo_clip_dataset):
    """Layer E 已实际 clipping 时，F 不得重复执行 observation clipping。"""

    def _layer_e_clipping(noisy, *_args, **_kwargs):
        waveform = np.clip(np.asarray(noisy), -0.5, 0.5).astype(np.float32)
        return LayerEResult(
            waveform=waveform,
            enabled=True,
            num_augmentations_drawn=1,
            selected=("clipping",),
            operations=(
                LayerEOperationResult("clipping", True, True, None),
            ),
            observation_changed=True,
        )

    echo_clip_dataset._apply_channel_degradation_result = _layer_e_clipping
    with patch("core.data.post_augment.speaker_distortion") as speaker_distortion:
        speaker_distortion.side_effect = _mock_identity_clip
        _ = echo_clip_dataset[0]

    speaker_distortion.assert_not_called()
    by_layer = echo_clip_dataset._last_simulation_stems["simulation_by_layer"]
    assert by_layer["E"]["applied"] == ["clipping"]
    assert "clip" not in by_layer["F"]["cpu_applied_ops"]


def test_silence_short_circuit_does_not_mark_applied(echo_clip_dataset):
    """C-aec 选中路径但未改 waveform 时，F observation clipping 仍正常执行。"""
    echo_clip_dataset._aec_echo_clipping_path_probs = {"post_ser_distortion": 1.0}
    with patch(
        "tasks.aec.data.dynamic_dataset.apply_peak_relative_clip",
        side_effect=lambda signal, *a, **k: (signal, "none"),
    ), patch("core.data.post_augment.speaker_distortion") as speaker_distortion:
        speaker_distortion.side_effect = _mock_identity_clip
        _ = echo_clip_dataset[0]
        speaker_distortion.assert_called_once()
    audit = echo_clip_dataset._last_simulation_stems["simulation_by_layer"]["C"]["c3_echo_clipping"]
    assert audit["path"] == "post_ser_distortion"
    assert audit["variant"] == "none"
    assert audit["applied"] is False


def test_construction_rejects_unknown_path_probs_key(aec_paths):
    with pytest.raises(ValueError, match="path_probs has unknown path"):
        AECDynamicMixingDataset(
            **aec_paths,
            aec_echo_clipping={"enabled": True, "path_probs": {"post_ser_distortian": 1.0}},
        )


def test_construction_rejects_unknown_variant_probs_key(aec_paths):
    with pytest.raises(ValueError, match="variant_probs has unknown variant"):
        AECDynamicMixingDataset(
            **aec_paths,
            aec_echo_clipping={
                "enabled": True,
                "paths": {
                    "post_ser_distortion": {"variant_probs": {"soft": 1.0}}
                },
            },
        )


def test_construction_rejects_negative_overload_boost_db(aec_paths):
    """负 gain 是衰减而不是 overload，必须在构造期拒绝。"""
    with pytest.raises(ValueError, match="overload_boost_db_range must be non-negative"):
        AECDynamicMixingDataset(
            **aec_paths,
            aec_echo_clipping={
                "enabled": True,
                "paths": {
                    "pre_ser_overload": {"overload_boost_db_range": [-6.0, 3.0]}
                },
            },
        )


def test_construction_rejects_zero_path_probs_weights(aec_paths):
    with pytest.raises(ValueError, match=r"path_probs weights must sum to (?:a finite value )?> 0"):
        AECDynamicMixingDataset(
            **aec_paths,
            aec_echo_clipping={"enabled": True, "path_probs": {"post_ser_distortion": 0.0}},
        )


def test_construction_rejects_zero_variant_probs_weights(aec_paths):
    with pytest.raises(
        ValueError,
        match=r"variant_probs weights must sum to (?:a finite value )?> 0",
    ):
        AECDynamicMixingDataset(
            **aec_paths,
            aec_echo_clipping={
                "enabled": True,
                "paths": {
                    "synced_ref_clip": {"variant_probs": {"fixed_rail": 0.0}}
                },
            },
        )


@pytest.mark.parametrize(
    "config,field",
    [
        ({"path_probs": {"post_ser_distortion": -1.0, "synced_ref_clip": 2.0}}, "path_probs"),
        ({"path_probs": {"post_ser_distortion": np.nan}}, "path_probs"),
        ({"path_probs": {"post_ser_distortion": np.inf}}, "path_probs"),
        (
            {"paths": {"post_ser_distortion": {"variant_probs": {"hard": -1.0, "tanh": 2.0}}}},
            "variant_probs",
        ),
        (
            {"paths": {"synced_ref_clip": {"variant_probs": {"fixed_rail": np.nan}}}},
            "variant_probs",
        ),
        (
            {"paths": {"synced_ref_clip": {"variant_probs": {"fixed_rail": np.inf}}}},
            "variant_probs",
        ),
    ],
)
def test_construction_rejects_nonfinite_or_negative_clip_weights(
    aec_paths, config, field
):
    with pytest.raises(ValueError, match=rf"{field} weights must be finite and non-negative"):
        AECDynamicMixingDataset(
            **aec_paths,
            aec_echo_clipping={"enabled": True, **config},
        )


@pytest.mark.parametrize(
    "path,field,value",
    [
        ("post_ser_distortion", "peak_relative_threshold_range", [0.5, np.nan]),
        ("post_ser_distortion", "peak_relative_threshold_range", [0.5, np.inf]),
        ("pre_ser_overload", "overload_boost_db_range", [3.0, np.nan]),
        ("pre_ser_overload", "overload_boost_db_range", [3.0, np.inf]),
        ("pre_ser_overload", "rail_dbfs_range", [-6.0, np.nan]),
        ("synced_ref_clip", "rail_dbfs_range", [-6.0, np.inf]),
    ],
)
def test_construction_rejects_nonfinite_clip_ranges(aec_paths, path, field, value):
    with pytest.raises(ValueError, match=rf"{field} values must be finite"):
        AECDynamicMixingDataset(
            **aec_paths,
            aec_echo_clipping={
                "enabled": True,
                "paths": {path: {field: value}},
            },
        )


def test_construction_rejects_positive_dbfs_threshold(aec_paths):
    """正 dBFS rail 位于 full scale 以上，必须拒绝。"""
    with pytest.raises(ValueError, match="rail_dbfs_range must be <= 0 dBFS"):
        AECDynamicMixingDataset(
            **aec_paths,
            aec_echo_clipping={
                "enabled": True,
                "paths": {"synced_ref_clip": {"rail_dbfs_range": [-1.0, 1.0]}},
            },
        )


def test_synced_ref_clip_reuses_absolute_rail_and_reports_facts(echo_clip_dataset):
    """synced 两路复用一次绝对 rail 抽样，并按真实 waveform 变化记录 facts。"""
    from tasks.aec.data.echo_clipping import (
        apply_absolute_rail_clip,
        sample_absolute_clip_params,
    )

    echo_clip_dataset._aec_echo_clipping_path_probs = {"synced_ref_clip": 1.0}
    recorded = []

    def _record_and_clip(signal, rail_range, variant_probs, rng, **kwargs):
        output = apply_absolute_rail_clip(
            signal, rail_range, variant_probs, rng, **kwargs
        )
        recorded.append(
            {
                "input": np.asarray(signal).copy(),
                "output": output[0].copy(),
                "rail_range": rail_range,
                "rail_dbfs": kwargs.get("rail_dbfs"),
                "returned_rail_dbfs": output[2],
                "variant": kwargs.get("variant"),
            }
        )
        return output

    with patch(
        "tasks.aec.data.dynamic_dataset.sample_absolute_clip_params",
        wraps=sample_absolute_clip_params,
    ) as sample_params, patch(
        "tasks.aec.data.dynamic_dataset.apply_absolute_rail_clip",
        side_effect=_record_and_clip,
    ):
        _ = echo_clip_dataset[0]

    audit = echo_clip_dataset._last_simulation_stems["simulation_by_layer"]["C"]["c3_echo_clipping"]
    sample_params.assert_called_once_with(
        (-12.0, -12.0), {"fixed_rail": 1.0}, ANY
    )
    assert len(recorded) == 2
    rail = 10.0 ** (-12.0 / 20.0)
    assert [item["rail_range"] for item in recorded] == [
        (-12.0, -12.0),
        (-12.0, -12.0),
    ]
    assert [item["rail_dbfs"] for item in recorded] == [-12.0, -12.0]
    assert [item["returned_rail_dbfs"] for item in recorded] == [-12.0, -12.0]
    assert [item["variant"] for item in recorded] == ["fixed_rail", "fixed_rail"]
    assert audit["path"] == "synced_ref_clip"
    assert audit["rail_dbfs"] == pytest.approx(-12.0)
    assert audit["rail"] == pytest.approx(rail)
    assert audit["threshold_frac"] is None
    assert audit["threshold_abs"] is None
    assert audit["shared_variant"] == "fixed_rail"
    assert audit["variant"] == "fixed_rail"
    assert audit["farend_variant"] == "fixed_rail"
    assert all(float(np.max(np.abs(item["output"]))) <= rail for item in recorded)
    assert audit["echo_applied"] is (not np.array_equal(recorded[0]["input"], recorded[0]["output"]))
    assert audit["farend_applied"] is (not np.array_equal(recorded[1]["input"], recorded[1]["output"]))
    assert audit["any_applied"] is (audit["echo_applied"] or audit["farend_applied"])
    assert audit["applied"] is audit["any_applied"]
    echo_facts = audit["echo_rail_facts"]
    farend_facts = audit["farend_rail_facts"]
    assert echo_facts["sample_count"] == len(recorded[0]["input"])
    assert farend_facts["sample_count"] == len(recorded[1]["input"])
    assert audit["echo_rail_hit"] is echo_facts["rail_hit"]
    assert audit["farend_rail_hit"] is farend_facts["rail_hit"]
    total_clipped = echo_facts["clipped_sample_count"] + farend_facts["clipped_sample_count"]
    total_samples = echo_facts["sample_count"] + farend_facts["sample_count"]
    assert audit["rail_hit"] is (echo_facts["rail_hit"] or farend_facts["rail_hit"])
    assert audit["clipped_sample_count"] == total_clipped
    assert audit["rail_sample_count"] == total_samples
    assert audit["clipped_sample_ratio"] == pytest.approx(total_clipped / total_samples)


def test_synced_farend_only_clip_does_not_gate_layer_f(echo_clip_dataset):
    """farend-only corruption 计入 any，但 Layer F 仍只由 echo applied 门控。"""
    echo_clip_dataset._aec_echo_clipping_path_probs = {"synced_ref_clip": 1.0}
    calls = 0

    def _farend_only(signal, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.asarray(signal).copy(), "none", kwargs["rail_dbfs"]
        return (
            np.clip(signal, -0.1, 0.1).astype(np.float32),
            "fixed_rail",
            kwargs["rail_dbfs"],
        )

    with patch(
        "tasks.aec.data.dynamic_dataset.apply_absolute_rail_clip",
        side_effect=_farend_only,
    ), patch("core.data.post_augment.speaker_distortion") as speaker_distortion:
        speaker_distortion.side_effect = _mock_identity_clip
        _ = echo_clip_dataset[0]

    speaker_distortion.assert_called_once()
    audit = echo_clip_dataset._last_simulation_stems["simulation_by_layer"]["C"]["c3_echo_clipping"]
    assert audit["echo_applied"] is False
    assert audit["farend_applied"] is True
    assert audit["any_applied"] is True
    assert audit["applied"] is True

pytestmark = [pytest.mark.integration]
