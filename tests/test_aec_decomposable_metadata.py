"""Tests: AEC realized decomposable 门控、C-3 colored noise 并入 noise
stem、随机延迟混合分布（_DELAY_TAIL_MIX_PROB）。

对应 diff 引入但此前零直接测试的三处逻辑（tasks/aec/data/dynamic_dataset.py）：
1. decomposable gating：Layer F fixed_rail 改变 observation 时，mic 相关
   SER/SNR 失效、独立 ERL 保留，decomposable=False 落盘。
2. C-3 colored noise 并入 noise stem：prob_colored_noise=1.0 时 noise stem
   能量/std 相对同种子未触发 run 抬升，decomposable 仍为 True。
   （colored noise 自原 Layer F-5 迁入 Layer C-3：仅在 need_c12 分支施加，
   从产生起即属 noise stem，不再经 Layer F post-mix。）
3. beta 模式随机延迟的混合分布 (1-π)·Beta(1,3)+π·Uniform[min,max]
   （π=_DELAY_TAIL_MIX_PROB=0.25）：固定种子下延迟尾部（均匀段）占比与
   π 同量级（纯 Beta(1,3) 尾部概率仅 0.1%，混合后 ≈2.6%）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

import tasks.aec.data.dynamic_dataset as dynamic_dataset_module
import tasks.aec.data.post_mix_adapter as post_mix_adapter_module
from tasks.aec.data.dynamic_dataset import (
    _DELAY_TAIL_MIX_PROB,
    DynamicMixingDataset as AECDynamicMixingDataset,
    _waveform_mismatch_reason,
    validate_aec_dataset_config,
)
from core.data.dynamic_dataset_base import LayerEOperationResult, LayerEResult


def _write_scp(path: Path, entries: dict[str, Path]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for uid, audio_path in entries.items():
            info = soundfile.info(str(audio_path))
            f.write(f"{uid} {info.samplerate} {info.frames} {audio_path}\n")


def _make_impulse_rir(path: Path, sr: int) -> None:
    rir = np.zeros(sr, dtype=np.float32)
    rir[0] = 1.0
    soundfile.write(path, rir, sr)


@pytest.fixture
def make_paths(tmp_path: Path):
    """构造 near 与 echo 同相正弦的样本（风格同 test_aec_realized_metadata）。

    sim_overrides 合并进 simulation_cfg，dataset_kwargs 追加到 dataset
    构造参数（如 state_detector_hop_length / 随机延迟配置）。
    """

    def _build(
        speech_amp: float = 0.4,
        ser_db: float = 6.0,
        observation_clip: bool = False,
        sim_overrides: dict | None = None,
        **dataset_kwargs,
    ):
        sr = 16000
        duration = 1.0

        speech_dir = tmp_path / "speech"
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
        soundfile.write(farend_wav, speech, sr, subtype="FLOAT")
        soundfile.write(echo_wav, speech, sr, subtype="FLOAT")
        stereo_wav = aec_dir / "stereo_001.wav"
        soundfile.write(stereo_wav, np.stack([speech, speech], axis=1), sr, subtype="FLOAT")

        speech_scp = tmp_path / "speech.scp"
        noise_scp = tmp_path / "noise.scp"
        rir_scp = tmp_path / "rir.scp"
        wind_scp = tmp_path / "wind.scp"
        pair_json = tmp_path / "pair.json"
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

        paths = {
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
        paths.update(dataset_kwargs)
        return paths

    return _build


def _run_one(paths):
    ds = AECDynamicMixingDataset(**paths)
    ds.set_epoch(0, base_seed=123)
    ds.export_simulation_stems = True
    sample = ds[0]
    return ds, sample


def test_layer_f_fixed_rail_marks_linear_ratios_non_decomposable(make_paths):
    """F fixed_rail 令 mic 比值失效，但独立 far/echo ERL 仍可审计。"""
    paths = make_paths(speech_amp=0.8, observation_clip=True)
    ds, _ = _run_one(paths)
    by_layer = ds._last_simulation_stems["simulation_by_layer"]
    layer_d = by_layer["D"]
    guard_keys = {"roles", "threshold", "peak", "scale", "applied"}
    assert layer_d["stage_order"] == ("D-normalize", "D-guard")
    assert set(layer_d["normalize"]) == {
        "mode",
        "target_peak",
        "scale",
        "preserves_decomposition",
    }
    assert set(layer_d["guard"]["mic_related"]) == guard_keys
    assert set(layer_d["guard"]["farend"]) == guard_keys
    assert "D-normalize" not in by_layer
    assert "aec_context" not in by_layer
    trace_metadata = ds._last_simulation_trace.metadata
    canonical_layer_d = trace_metadata["layer_d_facts"]
    assert canonical_layer_d == {
        **layer_d,
        "stage_order": list(layer_d["stage_order"]),
    }
    assert json.loads(json.dumps(canonical_layer_d)) == canonical_layer_d
    assert "layer_d_normalize" not in trace_metadata
    assert "layer_d" not in trace_metadata
    assert "farend_guard" not in trace_metadata["aec_context"]
    realized = by_layer["realized"]
    assert ds._last_layer_ef_decomposable is False
    assert realized["decomposable"] is False
    assert realized["decomposable_reason_codes"] == [
        "layer_f_nonlinear",
        "final_guard_observation_only_scale",
    ]
    assert realized["ser_db"] is None
    assert realized["ser_reason_code"] == "mic_not_decomposable"
    assert realized["snr_db"] is None
    assert realized["snr_applicable"] is False
    assert realized["reconstruction_residual"] == {
        "rms": None,
        "peak_abs": None,
        "applicable": False,
        "reason_code": "mic_not_decomposable",
    }
    assert realized["erl_db"] is not None
    assert realized["erl_applicable"] is True
    assert np.isfinite(realized["echo_dbfs_rms"])
    assert np.isfinite(realized["far_dbfs_rms"])


def test_no_clip_sample_is_decomposable(make_paths):
    """无 Layer E/F 非线性时，ser_db/erl_db 正常输出。"""
    paths = make_paths(speech_amp=0.4)
    ds, _ = _run_one(paths)
    realized = ds._last_simulation_stems["simulation_by_layer"]["realized"]
    assert ds._last_layer_ef_decomposable is True
    assert realized["decomposable"] is True
    assert realized["ser_db"] is not None
    assert realized["erl_db"] is not None


@pytest.mark.parametrize("target_type", ["early", "dry"])
def test_identity_rir_does_not_create_target_mismatch(make_paths, target_type):
    """RIR 虽被抽到，但 delta RIR 的 full/early/dry 实际一致，不应误报。"""
    paths = make_paths(
        sim_overrides={"prob_reverberation": 1.0, "target_type": target_type}
    )
    ds, _ = _run_one(paths)
    realized = ds._last_simulation_stems["simulation_by_layer"]["realized"]

    assert realized["decomposable"] is True
    assert realized["decomposable_reason_codes"] == []


def test_nonidentity_rir_target_mismatch_is_detected(make_paths):
    paths = make_paths(
        sim_overrides={"prob_reverberation": 1.0, "target_type": "early"}
    )
    rir_path = Path(paths["rir_scp"]).read_text(encoding="utf-8").strip().split()[-1]
    rir = np.zeros(16000, dtype=np.float32)
    rir[0] = 1.0
    rir[2000] = 0.5
    soundfile.write(rir_path, rir, 16000, subtype="FLOAT")

    ds, _ = _run_one(paths)
    realized = ds._last_simulation_stems["simulation_by_layer"]["realized"]

    assert realized["decomposable"] is False
    assert realized["decomposable_reason_codes"] == ["joint_target_rir_mismatch"]


def test_waveform_mismatch_distinguishes_gain_from_nonlinearity():
    reference = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
    assert _waveform_mismatch_reason(
        reference * 0.5,
        reference,
        nonlinear_reason="nonlinear",
        gain_reason="gain",
    ) == "gain"
    distorted = np.clip(reference, -0.4, 0.4)
    assert _waveform_mismatch_reason(
        distorted,
        reference,
        nonlinear_reason="nonlinear",
        gain_reason="gain",
    ) == "nonlinear"


def test_actual_sidechain_wind_is_branch_aware(make_paths):
    sim = {
        "prob_target_echo": 0.0,
        "prob_target_noise_echo": 1.0,
        "prob_wind_noise_sidechain": 1.0,
    }
    ds, _ = _run_one(make_paths(sim_overrides=sim))
    by_layer = ds._last_simulation_stems["simulation_by_layer"]

    assert by_layer["C"]["c1_c2_wind_sidechain_applied"] is True
    assert by_layer["realized"]["decomposable"] is False
    assert by_layer["realized"]["decomposable_reason_codes"] == [
        "sidechain_wind_nonlinear"
    ]


def test_layer_e_and_cpu_layer_f_are_fail_closed(make_paths, monkeypatch):
    ds = AECDynamicMixingDataset(
        **make_paths(
            sim_overrides={
                "noisy_post_mix_degradations_enabled": True,
                "prob_post_eq": 1.0,
            }
        )
    )
    ds.set_epoch(0, base_seed=123)
    ds.export_simulation_stems = True

    def _fake_layer_e(noisy, *args, **kwargs):
        return LayerEResult(
            waveform=noisy + 0.01,
            enabled=True,
            num_augmentations_drawn=1,
            selected=("bandwidth_limitation",),
            operations=(
                LayerEOperationResult(
                    "bandwidth_limitation", True, True, None
                ),
            ),
            observation_changed=True,
        )

    def _fake_layer_f(noisy, *args, **kwargs):
        kwargs["applied_ops_out"].append("post_eq")
        return noisy

    ds._apply_channel_degradation_result = _fake_layer_e
    monkeypatch.setattr(
        post_mix_adapter_module, "apply_post_augmentations", _fake_layer_f
    )
    _ = ds[0]
    realized = ds._last_simulation_stems["simulation_by_layer"]["realized"]
    assert realized["decomposable"] is False
    assert realized["decomposable_reason_codes"] == [
        "layer_e_nonlinear",
        "layer_f_nonlinear",
    ]
    layer_f = ds._last_simulation_stems["simulation_by_layer"]["F"]
    assert layer_f == {
        "noisy_post_mix_degradations_enabled": True,
        "cpu_applied_ops": ["post_eq"],
        "observation_clipping": {},
    }


def test_disabled_layer_f_skips_all_cpu_operations(make_paths):
    ds, _ = _run_one(
        make_paths(
            sim_overrides={
                "noisy_post_mix_degradations_enabled": False,
                "prob_post_eq": 1.0,
            }
        )
    )
    by_layer = ds._last_simulation_stems["simulation_by_layer"]

    assert by_layer["realized"]["decomposable"] is True
    assert by_layer["F"] == {
        "noisy_post_mix_degradations_enabled": False,
        "cpu_applied_ops": [],
        "observation_clipping": {},
    }


def test_enabled_layer_f_reports_cpu_operations(make_paths):
    ds, _ = _run_one(
        make_paths(
            sim_overrides={
                "noisy_post_mix_degradations_enabled": True,
                "prob_clip": 0.0,
                "prob_post_eq": 0.0,
            }
        )
    )
    by_layer = ds._last_simulation_stems["simulation_by_layer"]

    assert by_layer["realized"]["decomposable"] is True
    assert by_layer["F"]["noisy_post_mix_degradations_enabled"] is True
    assert by_layer["F"]["cpu_applied_ops"] == []
    assert by_layer["F"]["observation_clipping"] == {}


def test_selected_but_unchanged_layer_e_preserves_decomposability(make_paths):
    ds = AECDynamicMixingDataset(**make_paths())
    ds.set_epoch(0, base_seed=123)
    ds.export_simulation_stems = True

    def _fake_layer_e(noisy, *args, **kwargs):
        return LayerEResult(
            waveform=noisy,
            enabled=True,
            num_augmentations_drawn=1,
            selected=("codec",),
            operations=(
                LayerEOperationResult(
                    "codec", False, False, "dependency_unavailable"
                ),
            ),
            observation_changed=False,
        )

    ds._apply_channel_degradation_result = _fake_layer_e
    _ = ds[0]
    by_layer = ds._last_simulation_stems["simulation_by_layer"]

    assert by_layer["E"]["selected"] == ["codec"]
    assert by_layer["E"]["applied"] == []
    assert by_layer["E"]["skipped"] == ["codec"]
    assert by_layer["E"]["observation_changed"] is False
    assert by_layer["E"]["skip_codec_redundant_with_e"] is False
    assert by_layer["realized"]["decomposable"] is True
    assert by_layer["realized"]["decomposable_reason_codes"] == []


@pytest.mark.parametrize("branch", ["prob_echo_only", "prob_noise_echo_only"])
def test_pure_echo_routes_ignore_hidden_nearend_nonlinearity(make_paths, branch):
    sim = {
        "prob_target_only": 0.0,
        "prob_noise_only": 0.0,
        "prob_echo_only": float(branch == "prob_echo_only"),
        "prob_noise_echo_only": float(branch == "prob_noise_echo_only"),
        "prob_target_noise": 0.0,
        "prob_target_echo": 0.0,
        "prob_target_noise_echo": 0.0,
        "prob_reverberation": 1.0,
        "prob_wind_noise_sidechain": 1.0,
    }
    ds, _ = _run_one(make_paths(sim_overrides=sim))
    by_layer = ds._last_simulation_stems["simulation_by_layer"]
    realized = by_layer["realized"]

    assert realized["decomposable"] is True
    assert realized["decomposable_reason_codes"] == []
    assert realized["ser_db"] is None
    assert realized["ser_reason_code"] == "scenario_has_no_nearend"
    assert realized["erl_db"] is not None
    # SER draw 仍作为 latent calibration 记录保留。
    assert by_layer["C"]["c3_ser_db_drawn"] is not None
    assert by_layer["C"]["c3_ser_scaling_used"] is True


def test_c3_colored_noise_merges_into_noise_stem(make_paths):
    """C-3 prob_colored_noise=1.0（need_c12 分支）：noise stem 能量/std 抬升，
    且因 colored noise 属声学混合层（Layer C），decomposable 仍为 True。

    注：不再断言 state label 的 noise-present bit——检测器对噪声做峰值
    归一化（state_label._state_frame_activity_mask），电平不变；need_c12
    分支的 C-1 基线噪声恒非零，bit 在 off/on run 均置位，对 C-3 无区分
    度（迁移前的原 F-5 可作用于无基线噪声分支，该断言才有意义）。
    """
    # 基线噪声压到 120 dB（电平可忽略），分支切到 prob_target_noise_echo
    # （need_c12=True），C-3 才会施加
    base_sim = {
        "snr_low": 120.0,
        "snr_high": 120.0,
        "prob_target_echo": 0.0,
        "prob_target_noise_echo": 1.0,
    }
    paths_off = make_paths(sim_overrides=base_sim)
    ds_off, _ = _run_one(paths_off)

    sim_on = dict(
        base_sim,
        prob_colored_noise=1.0,
        colored_noise_snr_range=[10.0, 10.0],
    )
    paths_on = make_paths(sim_overrides=sim_on)
    ds_on, _ = _run_one(paths_on)

    # colored noise 在 Layer C 声学混合阶段并入 noise stem，不破坏线性分解
    assert ds_on._last_layer_ef_decomposable is True

    noise_off = np.asarray(ds_off._last_noise_additive_1d)
    noise_on = np.asarray(ds_on._last_noise_additive_1d)
    e_off = float(np.mean(noise_off ** 2))
    e_on = float(np.mean(noise_on ** 2))
    # 10 dB SNR 的 colored noise 远大于 120 dB 基线噪声
    assert e_on > e_off * 10.0
    # 回归：C-3 若把 2D clean_target((1,T)) 直接传给 colored_noise()，
    # len(data)==1 会退化为单点 DC 偏移——std 留在基线量级；真实 colored
    # noise 零均值、std≈rms（相对基线抬升 ~5 个数量级）
    assert float(np.std(noise_on)) > 10.0 * float(np.std(noise_off))


def test_random_delay_beta_mode_tail_mixture(make_paths):
    """混合延迟分布 (1-π)·Beta(1,3)+π·Uniform[min,max]：尾部占比与 π 同量级。"""
    paths = make_paths(
        aec_ref_echo_random_delay_prob=1.0,
        aec_ref_echo_random_delay_distribution="beta",
        aec_ref_echo_random_delay_beta_a=1.0,
        aec_ref_echo_random_delay_beta_b=3.0,
        aec_ref_echo_random_delay_uniform_prob=0.25,
        aec_ref_echo_delay_range_ms=[0.0, 100.0],
    )
    ds = AECDynamicMixingDataset(**paths)
    sr = 16000
    target_len = sr  # 1 s，远大于 max delay 100 ms
    farend = np.zeros(256, dtype=np.float32)
    echo = np.zeros(256, dtype=np.float32)
    rng = np.random.default_rng(0)
    n = 4000
    delays = np.array(
        [
            ds._apply_random_echo_delay(
                rng, farend, echo, target_len, sr
            )[2]
            for _ in range(n)
        ]
    )
    max_samp = 0.1 * sr
    # 尾部（>0.9·max 的均匀段）期望占比：
    # (1-π)·P(Beta(1,3)>0.9) + π·0.1 ≈ 0.75·0.001 + 0.025 ≈ 0.0258；
    # 纯 Beta(1,3) 仅 0.001 —— 必须与 π=0.25 同量级（容差放宽避免 flake）。
    tail_frac = float((delays > 0.9 * max_samp).mean())
    assert 0.01 < tail_frac < 0.05
    # 均匀分量同时给中高段曝光（期望 ≈0.75·0.125+0.25·0.5 ≈ 0.22）
    mid_frac = float((delays > 0.5 * max_samp).mean())
    assert mid_frac > 0.1
    assert _DELAY_TAIL_MIX_PROB == pytest.approx(0.25)


def test_random_delay_beta_custom_uniform_probability(make_paths):
    """会议配置的 Beta(1.5,15)+30% Uniform 应保留 VoIP 长延迟覆盖。"""
    paths = make_paths(
        aec_ref_echo_random_delay_prob=1.0,
        aec_ref_echo_random_delay_distribution="beta",
        aec_ref_echo_random_delay_beta_a=1.5,
        aec_ref_echo_random_delay_beta_b=15.0,
        aec_ref_echo_random_delay_uniform_prob=0.30,
        aec_ref_echo_delay_range_ms=[0.0, 1000.0],
    )
    ds = AECDynamicMixingDataset(**paths)
    sr = 16000
    target_len = 2 * sr
    farend = np.zeros(256, dtype=np.float32)
    echo = np.zeros(256, dtype=np.float32)
    rng = np.random.default_rng(1)
    delays_ms = np.asarray(
        [
            ds._apply_random_echo_delay(
                rng, farend, echo, target_len, sr
            )[2]
            * 1000.0
            / sr
            for _ in range(8000)
        ]
    )
    edges = np.asarray([0.0, 50.0, 100.0, 200.0, 500.0, 1000.0])
    fractions = np.asarray(
        [
            ((delays_ms >= lo) & (delays_ms < hi)).mean()
            for lo, hi in zip(edges[:-2], edges[1:-1])
        ]
        + [((delays_ms >= edges[-2]) & (delays_ms <= edges[-1])).mean()]
    )
    expected = np.asarray([0.248, 0.230, 0.227, 0.145, 0.150])
    np.testing.assert_allclose(fractions, expected, atol=0.025)


def test_aec_delay_range_rejects_invalid_values():
    for value in ([-100.0, -50.0], [50.0, 10.0], [0.0], [0.0, float("nan")]):
        with pytest.raises(ValueError, match="aec_ref_echo_delay_range_ms"):
            validate_aec_dataset_config(
                simulation_cfg={"prob_target_only": 1.0},
                aec_ref_echo_random_delay_prob=1.0,
                aec_ref_echo_delay_range_ms=value,
            )


def test_aec_delay_range_accepts_zero_to_990_ms():
    validate_aec_dataset_config(
        simulation_cfg={"prob_target_only": 1.0},
        aec_ref_echo_random_delay_prob=1.0,
        aec_ref_echo_delay_range_ms=[0.0, 990.0],
    )


def test_jitter_derives_caps_from_base_delay_range(
    make_paths, monkeypatch: pytest.MonkeyPatch
):
    """canonical 上限仍由基础范围派生。"""
    paths = make_paths(
        aec_ref_echo_random_delay_prob=1.0,
        aec_ref_echo_delay_range_ms=[100.0, 120.0],
        aec_time_delay_jitter={
            "enabled": True,
            "prob": 1.0,
            "jitter_type": "sudden_jump",
            "jitter_rate": [1000.0, 1000.0],
            "jump_ms_range": [100.0, 100.0],
            "min_segment_ms": 0.5,
            "max_relative_jitter_ms": 1.0,
        },
    )
    results = []
    original = dynamic_dataset_module.apply_time_delay_jitter

    def record_result(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(
        dynamic_dataset_module, "apply_time_delay_jitter", record_result
    )

    ds, _ = _run_one(paths)

    assert len(results) == 1
    result = results[0]
    max_total_samp = 120.0 / 1000.0 * 16000
    assert result.meta["max_relative_jitter_ms"] == pytest.approx(120.0)
    assert np.all(result.delay_gt >= 0.0)
    assert np.all(result.delay_gt <= max_total_samp + 1e-6)
    jitter_meta = ds._last_simulation_stems["simulation_by_layer"]["C"][
        "c3_disk_pair_load"
    ]["time_delay_jitter"]
    assert jitter_meta["max_relative_jitter_ms"] == pytest.approx(120.0)


def test_enabled_jitter_requires_base_delay_range():
    with pytest.raises(ValueError, match="aec_ref_echo_delay_range_ms explicitly"):
        validate_aec_dataset_config(
            simulation_cfg={"prob_target_only": 1.0},
            aec_time_delay_jitter={"enabled": True, "prob": 1.0},
        )


def test_jitter_with_zero_probability_does_not_require_base_delay_range():
    validate_aec_dataset_config(
        simulation_cfg={"prob_target_only": 1.0},
        aec_time_delay_jitter={"enabled": True, "prob": 0.0},
    )


@pytest.mark.parametrize(
    "jitter_config",
    [
        {"max_abs_delay_ms": 1.0},
        {"max_abs_delay_ms": 1.0, "max_relative_jitter_ms": 10.0},
    ],
    ids=["legacy_only", "legacy_and_canonical"],
)
def test_jitter_rejects_removed_legacy_cap_with_base_range(jitter_config):
    validate_aec_dataset_config(
        simulation_cfg={"prob_target_only": 1.0},
        aec_ref_echo_delay_range_ms=[0.0, 100.0],
        aec_time_delay_jitter={"max_relative_jitter_ms": 1.0},
    )
    with pytest.raises(ValueError) as exc_info:
        validate_aec_dataset_config(
            simulation_cfg={"prob_target_only": 1.0},
            aec_ref_echo_delay_range_ms=[0.0, 100.0],
            aec_time_delay_jitter=jitter_config,
        )
    message = str(exc_info.value)
    assert "max_abs_delay_ms" in message
    assert "max_relative_jitter_ms" in message


def test_jitter_gate_allows_effective_zero_base_delay(
    make_paths, monkeypatch: pytest.MonkeyPatch
):
    paths = make_paths(
        aec_ref_echo_random_delay_prob=0.0,
        aec_ref_echo_delay_range_ms=[0.0, 100.0],
        aec_time_delay_jitter={
            "enabled": True,
            "prob": 1.0,
            "jitter_type": "sudden_jump",
            "jitter_rate": [1000.0, 1000.0],
            "jump_ms_range": [5.0, 5.0],
            "min_segment_ms": 0.5,
        },
    )
    calls = []
    original = dynamic_dataset_module.apply_time_delay_jitter

    def record_call(*args, **kwargs):
        calls.append(float(args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        dynamic_dataset_module, "apply_time_delay_jitter", record_call
    )

    ds, _ = _run_one(paths)

    assert calls == [0.0]
    jitter_meta = ds._last_simulation_stems["simulation_by_layer"]["C"][
        "c3_disk_pair_load"
    ]["time_delay_jitter"]
    assert jitter_meta["applied"] is True


@pytest.mark.parametrize(
    ("random_delay_mode", "expected_jitter_base_delay"),
    [("echo_only", 320.0), ("sync", 0.0)],
)
def test_jitter_base_delay_tracks_relative_ref_echo_lag(
    make_paths,
    monkeypatch: pytest.MonkeyPatch,
    random_delay_mode,
    expected_jitter_base_delay,
):
    paths = make_paths(
        aec_ref_echo_random_delay_prob=1.0,
        aec_ref_echo_random_delay_mode=random_delay_mode,
        aec_ref_echo_delay_range_ms=[20.0, 20.0],
        aec_time_delay_jitter={
            "enabled": True,
            "prob": 1.0,
            "jitter_type": "sudden_jump",
            "jitter_rate": [1000.0, 1000.0],
            "jump_ms_range": [5.0, 5.0],
            "min_segment_ms": 0.5,
        },
    )
    calls = []
    original = dynamic_dataset_module.apply_time_delay_jitter

    def record_call(*args, **kwargs):
        calls.append(float(args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        dynamic_dataset_module, "apply_time_delay_jitter", record_call
    )

    ds, _ = _run_one(paths)

    assert calls == [expected_jitter_base_delay]
    c3_meta = ds._last_simulation_stems["simulation_by_layer"]["C"][
        "c3_disk_pair_load"
    ]
    assert c3_meta["random_delay_ms"] == pytest.approx(20.0)


@pytest.mark.parametrize("policy", ["skip", "log_only"])
def test_aec_dataset_rejects_removed_gcc_pad_policies(policy):
    with pytest.raises(ValueError, match="only supports 'apply'"):
        validate_aec_dataset_config(
            simulation_cfg={"prob_target_only": 1.0},
            aec_ref_echo_gcc_farend_pad_policy=policy,
        )


pytestmark = [pytest.mark.unit]
