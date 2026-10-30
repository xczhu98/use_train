"""Tests: SER 可行窗口抽取（C-aec 段，tasks/aec/data/dynamic_dataset.py）。

修复前：ser ~ U(ser_range)，echo_db = clamp(clean_db - ser, -60, 0)，
极端音量样本在钳位边界形成 effective-SER 质量堆积。
修复后：ser 在 [max(ser_lo, clean_db), min(ser_hi, clean_db + 60)] 窗口内
抽取，echo_db 恒落 [-60, 0]，c3_effective_ser_db 恒等于 c3_ser_db_drawn。

覆盖：
1. clean_db ≈ -50 dBFS（轻声）：抽取值落在收缩窗口内且 effective == drawn；
2. clean_db ≈ -10 dBFS（热声）：同上；
3. clean_db ≈ -30 dBFS（常规音量）：窗口退化为完整 ser_range，行为不变；
4. 钉死 ser_range（lo == hi）：回退旧行为，drawn == 钉值；
5. 钉值 < clean_db（窗口为空）：回退旧行为，clamp 仍生效。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

from tasks.aec.data.dynamic_dataset import DynamicMixingDataset as AECDynamicMixingDataset
from tasks.aec.data.erl_coupling import (
    ERL_RANGE_EMPTY_INTERSECTION,
    ERLCouplingContractError,
)

_N_SAMPLES = 30


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
    """构造单电平正弦样本；N 个 speech uid 指向同一 wav，靠 index RNG 分叉。

    speech = farend = A·sin(300Hz)，1 s 整周期 → clean_db = 20·log10(A/√2)。
    """

    def _build(
        speech_amp: float,
        ser_range: list[float],
        *,
        erl_range: list[float] | None = None,
        erl_distribution: str = "beta",
        strict_load: bool = True,
    ):
        sr = 16000
        duration = 1.0
        erl_range = erl_range or [-40.0, 60.0]
        tag = (
            f"{speech_amp}_{ser_range[0]}_{ser_range[1]}_"
            f"{erl_range[0]}_{erl_range[1]}_{erl_distribution}_{strict_load}"
        )

        speech_dir = tmp_path / f"speech_{tag}"
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

        speech_scp = tmp_path / f"speech_{tag}.scp"
        noise_scp = tmp_path / f"noise_{tag}.scp"
        rir_scp = tmp_path / f"rir_{tag}.scp"
        wind_scp = tmp_path / f"wind_{tag}.scp"
        pair_json = tmp_path / f"pair_{tag}.json"
        _write_scp(
            speech_scp,
            {f"spk1_{i + 1:03d}": speech_wav for i in range(_N_SAMPLES)},
        )
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
            "noisy_post_mix_degradations_enabled": False,
            "prob_reverberation": 0.0,
            "use_high_pass": False,
            "ser_range": ser_range,
            # SER 边界用例默认覆盖全部物理窗口，避免把 ERL 合同冲突
            # 混入本文件的 SER 职责；合同专项可通过 fixture 参数收紧。
            "erl_range": erl_range,
            "erl_distribution": erl_distribution,
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
            "strict_load": strict_load,
        }

    return _build


def _collect_c3(paths, n: int = _N_SAMPLES) -> list[dict]:
    ds = AECDynamicMixingDataset(**paths)
    ds.set_epoch(0, base_seed=123)
    ds.export_simulation_stems = True
    out = []
    for i in range(n):
        _ = ds[i]
        out.append(ds._last_simulation_stems["simulation_by_layer"]["C"])
    return out


def _assert_window(resamples: list[dict], ser_lo: float, ser_hi: float) -> None:
    """窗口语义断言：drawn ∈ [max(lo,clean), min(hi,clean+60)] 且 effective == drawn。"""
    for c3 in resamples:
        drawn = c3["c3_ser_db_drawn"]
        effective = c3["c3_effective_ser_db"]
        echo_db = c3["c3_far_db"] - c3["c3_erl_db"]
        clean_db = echo_db + drawn  # echo_db = clean_db - drawn 自洽反解
        assert effective == pytest.approx(drawn, abs=0.01)
        assert -60.0 <= echo_db <= 0.0
        assert drawn >= max(ser_lo, clean_db) - 1e-6
        assert drawn <= min(ser_hi, clean_db + 60.0) + 1e-6


def test_ser_window_shrinks_at_quiet_end(make_paths):
    """clean_db ≈ -50：窗口收缩为 [-20, 10]，旧行为下 25% 抽取会被钳位。"""
    amp = float(np.sqrt(2.0) * 10.0 ** (-50.0 / 20.0))
    paths = make_paths(speech_amp=amp, ser_range=[-20.0, 20.0])
    resamples = _collect_c3(paths)
    _assert_window(resamples, -20.0, 20.0)
    # 统计显著性：窗口上界 10 被实质压缩（旧行为下 30 样本全 ≤10 的概率 0.75^30）
    assert max(c3["c3_ser_db_drawn"] for c3 in resamples) <= 10.0 + 0.1


def test_ser_window_shrinks_at_hot_end(make_paths):
    """clean_db ≈ -10：窗口收缩为 [-10, 20]，旧行为下 25% 抽取会被钳位。"""
    amp = float(np.sqrt(2.0) * 10.0 ** (-10.0 / 20.0))
    paths = make_paths(speech_amp=amp, ser_range=[-20.0, 20.0])
    resamples = _collect_c3(paths)
    _assert_window(resamples, -20.0, 20.0)
    assert min(c3["c3_ser_db_drawn"] for c3 in resamples) >= -10.0 - 0.1


def test_ser_window_full_at_mid_level(make_paths):
    """clean_db ≈ -30：窗口 = 完整 [-20, 20]，与修复前行为一致。"""
    amp = float(np.sqrt(2.0) * 10.0 ** (-30.0 / 20.0))
    paths = make_paths(speech_amp=amp, ser_range=[-20.0, 20.0])
    resamples = _collect_c3(paths)
    _assert_window(resamples, -20.0, 20.0)


def test_pinned_ser_range_fallback(make_paths):
    """钉死 ser_range（lo == hi）：回退旧行为，drawn == 钉值且 effective == drawn。"""
    paths = make_paths(speech_amp=0.05, ser_range=[6.0, 6.0])
    for c3 in _collect_c3(paths):
        assert c3["c3_ser_db_drawn"] == pytest.approx(6.0, abs=1e-9)
        assert c3["c3_effective_ser_db"] == pytest.approx(6.0, abs=0.01)


def test_empty_window_falls_back_to_clamp(make_paths):
    """钉值 < clean_db（空窗口）：回退旧行为，echo_db 天花板钳位仍生效。

    amp=0.4 → clean_db ≈ -10.97；ser 钉 -15 → echo_db_raw = 4.03 → 钳到 0，
    effective SER = clean_db - 0 ≈ -10.97 ≠ drawn -15。
    """
    amp = 0.4
    paths = make_paths(speech_amp=amp, ser_range=[-15.0, -15.0])
    expected_clean_db = 20.0 * np.log10(amp / np.sqrt(2.0))
    for c3 in _collect_c3(paths):
        assert c3["c3_ser_db_drawn"] == pytest.approx(-15.0, abs=1e-9)
        assert c3["c3_effective_ser_db"] == pytest.approx(expected_clean_db, abs=0.01)


@pytest.mark.parametrize("distribution", ["beta", "uniform"])
def test_erl_range_audit_records_configured_physical_effective_selected(
    make_paths, distribution
):
    paths = make_paths(
        speech_amp=float(np.sqrt(2.0) * 10.0 ** (-30.0 / 20.0)),
        ser_range=[0.0, 0.0],
        erl_range=[-5.0, 5.0],
        erl_distribution=distribution,
    )
    c3 = _collect_c3(paths, n=1)[0]
    audit = c3["c3_erl_range_audit"]

    assert audit["configured_range_db"] == [-5.0, 5.0]
    assert audit["physical_range_db"] == pytest.approx([-10.0, 18.0], abs=0.02)
    assert audit["effective_range_db"] == [-5.0, 5.0]
    assert audit["selected_erl_db"] == pytest.approx(c3["c3_erl_db"], abs=1e-9)
    assert -5.0 <= audit["selected_erl_db"] <= 5.0
    json.dumps(audit, allow_nan=False)


def test_empty_erl_intersection_fails_closed_even_when_strict_load_false(make_paths):
    paths = make_paths(
        speech_amp=float(np.sqrt(2.0) * 10.0 ** (-20.0 / 20.0)),
        ser_range=[0.0, 0.0],
        erl_range=[30.0, 40.0],
        strict_load=False,
    )
    dataset = AECDynamicMixingDataset(**paths)
    dataset.set_epoch(0, base_seed=123)

    with pytest.raises(ERLCouplingContractError) as caught:
        _ = dataset[0]

    assert caught.value.reason == ERL_RANGE_EMPTY_INTERSECTION
    assert caught.value.metadata["effective_range_db"] is None
    json.dumps(caught.value.metadata, allow_nan=False)


pytestmark = [pytest.mark.unit]
