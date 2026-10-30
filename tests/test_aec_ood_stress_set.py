"""JAE-M03 OOD stress set 生成器测试：smoke + 回放一致性 + 场景语义抽查。

全合成源（零数据依赖），tmp_path 输出，总时长秒级。
"""
from __future__ import annotations
import pytest

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prepare_data_tools.generators.aec._ood_stress_scenarios import (  # noqa: E402
    SCENARIOS,
)
from tools.prepare_data_tools.generators.aec import generate_ood_stress_set as generator  # noqa: E402
from tools.prepare_data_tools.generators.aec.generate_ood_stress_set import (  # noqa: E402
    main as gen_main,
    parse_args,
)

SR = 16000
DURATION = 1.0  # smoke 用 1s，控制测试时长

# 场景名 → 事件 type 的固定映射
SCENARIO_EVENT_TYPE = {
    "far_end_stop_rir_tail": "far_end_stop",
    "mid_utterance_path_replace": "path_replace",
    "buffer_reset": "buffer_reset",
    "reference_missing": "reference_missing",
    "nearend_leak_into_reference": "nearend_leak",
    "reference_polarity_inversion": "polarity_inversion",
    "clock_async_resample": "clock_async",
    "codec_degrade": "codec_degrade",
    "agc_drc_limiter": "agc_drc",
}


def _assert_retired_publication(
    argv: list[str], out_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """AEC OOD GenerationManifest 发布已下线，任何输入都不得产生输出。"""
    assert gen_main(argv) == 1
    assert "GenerationManifest v1 publication 已下线" in capsys.readouterr().err
    assert not out_dir.exists()


# ----------------------------------------------------------------------
# 1. 生成器发布已下线；场景算子仍在本文件后半段直接测试。
# ----------------------------------------------------------------------
def test_ood_publication_is_retired_for_default_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    out_dir = tmp_path / "ood"
    _assert_retired_publication(
        [
            "--output-dir", str(out_dir),
            "--num-per-scenario", "1",
            "--seed", "0",
            "--duration-sec", str(DURATION),
            "--sr", str(SR),
        ],
        out_dir,
        capsys,
    )


def test_legacy_out_dir_alias_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--out-dir", "legacy-out"])


@pytest.mark.parametrize(
    "extra",
    [
        ["--num-per-scenario", "0"],
        ["--sr", "0"],
        ["--duration-sec", "nan"],
        ["--duration-sec", "inf"],
        ["--duration-sec", "0"],
    ],
)
def test_invalid_positive_parameters_fail_without_publication(tmp_path: Path, extra: list[str]):
    out_dir = tmp_path / "invalid"
    rc = gen_main(["--output-dir", str(out_dir), *extra])
    assert rc == 1
    assert not out_dir.exists()


def test_relative_quoted_scp_paths_resolve_from_scp_directory(tmp_path: Path):
    source_dir = tmp_path / "source dir"
    source_dir.mkdir()
    wav = source_dir / "input audio.wav"
    sf.write(wav, np.ones(200, dtype=np.float32) * 0.05, 8000)
    scp = source_dir / "inputs.scp"
    scp.write_text("utt 'input audio.wav'\n", encoding="utf-8")
    out_dir = tmp_path / "relative"

    rc = gen_main([
        "--output-dir", str(out_dir),
        "--farend-scp", str(scp),
        "--nearend-scp", str(scp),
        "--rir-scp", str(scp),
        "--scenarios", "reference_polarity_inversion",
        "--num-per-scenario", "1",
        "--duration-sec", "0.02",
        "--sr", "8000",
    ])

    assert rc == 1
    assert not out_dir.exists()


@pytest.mark.parametrize("suffix", [".wav", ".flac", ".rir"])
def test_audio_payload_fingerprint_helper_rejects_audio(tmp_path: Path, suffix: str):
    audio = tmp_path / f"source{suffix}"
    audio.write_bytes(b"must-not-be-read")
    with pytest.raises(ValueError, match="禁止对音频 payload"):
        generator._ordinary_file_fingerprint(audio)


@pytest.mark.parametrize("suffix", [".wav", ".flac", ".rir"])
def test_audio_payload_is_not_opened_for_identity_or_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
):
    audio = tmp_path / f"source{suffix}"
    sf.write(
        audio,
        np.ones(200, dtype=np.float32) * 0.05,
        8000,
        format="WAV",
    )
    scp = tmp_path / "source.scp"
    scp.write_text(f"utt {audio.name}\n", encoding="utf-8")
    out_dir = tmp_path / "no-audio-hash"
    real_open = generator.os.open
    audio_open_count = 0

    def guarded_open(path, flags, *args, **kwargs):
        nonlocal audio_open_count
        if Path(path) == audio:
            audio_open_count += 1
            raise AssertionError("audio payload must not be opened for identity/hash")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(generator.os, "open", guarded_open)
    assert gen_main([
        "--output-dir", str(out_dir),
        "--farend-scp", str(scp),
        "--nearend-scp", str(scp),
        "--rir-scp", str(scp),
        "--scenarios", "reference_polarity_inversion",
        "--num-per-scenario", "1",
        "--duration-sec", "0.02",
        "--sr", "8000",
    ]) == 1
    assert audio_open_count == 0
    assert not out_dir.exists()


def test_audio_source_symlink_is_rejected_without_publication(tmp_path: Path):
    real_wav = tmp_path / "real.wav"
    sf.write(real_wav, np.ones(200, dtype=np.float32) * 0.05, 8000)
    linked_wav = tmp_path / "linked.wav"
    linked_wav.symlink_to(real_wav)
    scp = tmp_path / "source.scp"
    scp.write_text(f"utt {linked_wav.name}\n", encoding="utf-8")
    out_dir = tmp_path / "symlink-source"

    assert gen_main([
        "--output-dir", str(out_dir),
        "--farend-scp", str(scp),
        "--scenarios", "reference_polarity_inversion",
        "--num-per-scenario", "1",
        "--duration-sec", "0.02",
        "--sr", "8000",
    ]) == 1
    assert not out_dir.exists()


def test_output_symlink_is_rejected_without_touching_referent(tmp_path: Path):
    referent = tmp_path / "referent"
    referent.mkdir()
    sentinel = referent / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    out_dir = tmp_path / "ood"
    out_dir.symlink_to(referent, target_is_directory=True)

    rc = gen_main([
        "--output-dir", str(out_dir),
        "--scenarios", "reference_polarity_inversion",
        "--num-per-scenario", "1",
        "--duration-sec", "0.02",
        "--sr", "8000",
    ])

    assert rc == 1
    assert out_dir.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(referent.iterdir()) == [sentinel]


@pytest.mark.parametrize(
    "scenario_arg",
    [
        "reference_polarity_inversion",
        "reference_missing,reference_polarity_inversion",
    ],
)
def test_ood_publication_is_retired_for_scenario_selections(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    scenario_arg: str,
):
    """发布退役后，场景选择不得绕过 fail-closed 边界。"""
    out_dir = tmp_path / "retired"
    _assert_retired_publication(
        [
            "--output-dir", str(out_dir),
            "--scenarios", scenario_arg,
            "--num-per-scenario", "1",
            "--duration-sec", "0.1",
            "--sr", "8000",
        ],
        out_dir,
        capsys,
    )


@pytest.mark.parametrize(
    ("scenario", "kwargs"),
    [
        ("far_end_stop_rir_tail", {"t1_frac": float("nan")}),
        ("buffer_reset", {"d1_ms": float("inf")}),
        ("reference_missing", {"win_sec": 0.0}),
        ("nearend_leak_into_reference", {"alpha": -1.0}),
        ("clock_async_resample", {"ppm": float("nan")}),
        ("codec_degrade", {"cutoff_hz": 0.0}),
        ("agc_drc_limiter", {"ratio": float("inf")}),
    ],
)
def test_scenario_numeric_overrides_must_be_finite_and_positive(scenario, kwargs):
    far, near, rir, sr = _make_inputs()
    with pytest.raises(ValueError):
        SCENARIOS[scenario](far, near, rir, sr, np.random.default_rng(0), **kwargs)


# ----------------------------------------------------------------------
# 3. 语义抽查（operator 级）
# ----------------------------------------------------------------------
def _make_inputs(duration: float = 1.0, sr: int = SR, seed: int = 123):
    """far=白噪声，near=白噪声，rir=delta（echo 退化为 far 本身/延迟版）。"""
    n = int(duration * sr)
    rng = np.random.default_rng(seed)
    far = (0.3 * rng.standard_normal(n)).astype(np.float32)
    near = (0.2 * rng.standard_normal(n)).astype(np.float32)
    rir = np.zeros(int(0.05 * sr), dtype=np.float32)
    rir[0] = 1.0
    return far, near, rir, sr


@pytest.mark.parametrize("scenario", ["mid_utterance_path_replace", "buffer_reset"])
@pytest.mark.parametrize(
    "t_frac",
    [float("nan"), float("inf"), float("-inf"), -1e-6, 0.0, 1.0, 1.0 + 1e-6],
    ids=["nan", "positive_inf", "negative_inf", "negative", "zero", "one", "above_one"],
)
def test_mid_scenario_t_frac_must_be_strictly_inside_unit_interval(scenario, t_frac):
    far, near, rir, sr = _make_inputs(duration=0.1)

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        SCENARIOS[scenario](
            far,
            near,
            rir,
            sr,
            np.random.default_rng(0),
            t_frac=t_frac,
            rir2_kind="time_reverse" if scenario == "mid_utterance_path_replace" else None,
        )


@pytest.mark.parametrize("scenario", ["mid_utterance_path_replace", "buffer_reset"])
@pytest.mark.parametrize("t_frac", [1e-6, 0.25, 0.999999], ids=["near_start", "middle", "near_end"])
def test_mid_scenario_event_time_stays_within_waveform(scenario, t_frac):
    far, near, rir, sr = _make_inputs(duration=0.1)
    kwargs = {"t_frac": t_frac}
    if scenario == "mid_utterance_path_replace":
        kwargs["rir2_kind"] = "time_reverse"

    out = SCENARIOS[scenario](far, near, rir, sr, np.random.default_rng(0), **kwargs)
    event = out["events"][0]
    event_time_sec = event["t_ms"] / 1000.0

    assert 0.0 <= event_time_sec <= len(near) / sr
    assert all(len(out[key]) == len(near) for key in ("mic", "farend", "nearend", "echo"))


def test_polarity_inversion_semantics():
    far, near, rir, sr = _make_inputs()
    out = SCENARIOS["reference_polarity_inversion"](
        far, near, rir, sr, np.random.default_rng(0))
    # farend 恰为未反转 far 的 -1 倍；echo 仍由 +far 卷积（delta RIR → far 本身）
    # （fftconvolve 为 float32 精度，容差放宽到 1e-5/1e-4）
    assert np.allclose(out["farend"], -far)
    assert np.allclose(out["echo"], far, rtol=1e-4, atol=1e-5)
    assert out["events"][0]["type"] == "polarity_inversion"


def test_reference_missing_semantics():
    far, near, rir, sr = _make_inputs()
    out = SCENARIOS["reference_missing"](far, near, rir, sr, np.random.default_rng(0))
    ev = out["events"][0]
    assert ev["type"] == "reference_missing"
    t0 = int(ev["t0_ms"] * sr / 1000.0)
    t1 = int(ev["t1_ms"] * sr / 1000.0)
    assert t1 > t0
    # 段内 farend 全零，段外保留原 far
    assert np.all(out["farend"][t0:t1] == 0.0)
    assert np.any(out["farend"][:t0] != 0.0)
    # mic / echo 不受影响（fftconvolve float32 尾差，容差放宽）
    assert np.allclose(out["echo"], far, rtol=1e-4, atol=1e-5)
    assert np.allclose(out["mic"], near + far, rtol=1e-4, atol=1e-5)


def test_buffer_reset_delay_change():
    far, near, rir, sr = _make_inputs()
    out = SCENARIOS["buffer_reset"](far, near, rir, sr, np.random.default_rng(0))
    ev = out["events"][0]
    assert ev["type"] == "buffer_reset"
    t_switch = int(ev["t_ms"] * sr / 1000.0)
    d1 = int(round(ev["d1_ms"] * sr / 1000.0))
    d2 = int(round(ev["d2_ms"] * sr / 1000.0))
    echo = out["echo"]

    def estimate_delay(start: int, end: int) -> int:
        """互相关粗验 echo[start:end] 相对同窗口 far 的延迟（采样数）。"""
        seg_e = echo[start:end]
        seg_f = far[start:end]
        corr = fftconvolve(seg_e, seg_f[::-1])
        return int(np.argmax(corr)) - (len(seg_f) - 1)

    lag_head = estimate_delay(0, t_switch)
    # 跳过切换点后 100ms，避开切换瞬态
    lag_tail = estimate_delay(t_switch + sr // 10, len(echo))
    assert abs(lag_head - d1) <= 2
    assert abs(lag_tail - d2) <= 2
    assert lag_head != lag_tail


@pytest.mark.parametrize(
    ("ppm", "kwargs"),
    [
        (5000.0, {"ppm": 5000.0, "sign": 1.0}),
        (-5000.0, {"ppm": -5000.0}),
    ],
)
def test_clock_async_resample_matches_legacy_numpy_recipe_byte_for_byte(ppm, kwargs):
    far, near, rir, sr = _make_inputs()
    expected_echo = fftconvolve(far, rir)[: far.size].astype(np.float32)
    ratio = 1.0 + ppm * 1e-6
    positions = np.arange(expected_echo.size, dtype=np.float64) * ratio
    valid = positions <= expected_echo.size - 1
    legacy = np.zeros(expected_echo.size, dtype=np.float32)
    legacy[valid] = np.interp(
        positions[valid],
        np.arange(expected_echo.size, dtype=np.float64),
        expected_echo.astype(np.float64),
    ).astype(np.float32)

    out = SCENARIOS["clock_async_resample"](
        far, near, rir, sr, np.random.default_rng(0), **kwargs
    )

    assert out["events"][0]["ppm"] == ppm
    np.testing.assert_array_equal(out["echo"], legacy)


def test_clock_async_resample_drift_direction():
    """clock_async：ppm 符号固定边界，互相关估计输出相对输入的全程延迟漂移方向与 ppm 符号一致。"""
    far, near, rir, sr = _make_inputs(duration=2.0)

    def measure_lag(echo_rs: np.ndarray, start: int, end: int) -> int:
        """互相关估计 echo_rs[start:end] 相对同窗口 far 的 lag（约定同 buffer_reset 测试）。"""
        seg_e = echo_rs[start:end]
        seg_f = far[start:end]
        corr = fftconvolve(seg_e, seg_f[::-1])
        return int(np.argmax(corr)) - (len(seg_f) - 1)

    # 实现约定 echo_rs[i] = echo[i*ratio]（ratio = 1 + ppm·1e-6，delta RIR 下 echo≈far）：
    # 正 ppm → 输出采样位置相对原信号以 +i·ppm·1e-6 累积漂移（越来越超前），
    # 窗口中心 c 处互相关 lag ≈ -c·ppm·1e-6，故 lag_head - lag_tail 与 ppm 同号。
    # 用 ±5000ppm（远大于默认 20~80ppm）让两窗口间隔 1.6 万样点累积 ~80 样点漂移，
    # 整样点分辨率的互相关可稳定测出；窗长 4000 内漂移涂抹 ±10 样点，幅度容差放宽到 20。
    for ppm, kw in ((5000.0, {"ppm": 5000.0, "sign": 1.0}), (-5000.0, {"ppm": -5000.0})):
        out = SCENARIOS["clock_async_resample"](
            far, near, rir, sr, np.random.default_rng(0), **kw)
        ev = out["events"][0]
        assert ev["type"] == "clock_async"
        # 显式传负 ppm 时符号不得被随机翻正（显式 sign 时以 sign 为准）
        assert ev["ppm"] == ppm
        lag_head = measure_lag(out["echo"], 4000, 8000)    # 窗口中心 6000
        lag_tail = measure_lag(out["echo"], 20000, 24000)  # 窗口中心 22000（避开尾部补零段）
        drift = lag_head - lag_tail  # 时钟位置累积漂移，方向应与 ppm 符号一致
        assert np.sign(drift) == np.sign(ppm), \
            f"ppm={ppm}: 漂移方向应与 ppm 符号一致，实际 drift={drift}"
        expected = (22000 - 6000) * abs(ppm) * 1e-6  # ≈80 样点
        assert abs(abs(drift) - expected) <= 20, \
            f"ppm={ppm}: |drift|={abs(drift)} 与预期 {expected:.0f} 偏差过大"


@pytest.mark.parametrize("rir2_kind", ["time_reverse", "circular_shift_25ms"])
def test_path_replace_crossfade_matches_existing_task_local_recipe_byte_for_byte(
    rir2_kind: str,
):
    """path crossfade 暂留 OOD local，固定输入下必须保持原 float32 recipe。"""
    far, near, rir, sr = _make_inputs(duration=0.1)
    t_frac = 0.5
    out = SCENARIOS["mid_utterance_path_replace"](
        far,
        near,
        rir,
        sr,
        np.random.default_rng(0),
        t_frac=t_frac,
        rir2_kind=rir2_kind,
    )

    n = far.size
    t_switch = int(t_frac * n)
    echo1 = fftconvolve(far, rir)[:n].astype(np.float32)
    if rir2_kind == "time_reverse":
        rir2 = rir[::-1].copy()
    else:
        rir2 = np.roll(rir, int(0.025 * sr))
    echo2 = fftconvolve(far, rir2)[:n].astype(np.float32)
    window = max(2, int(0.010 * sr))
    lo = max(0, t_switch - window // 2)
    hi = min(n, t_switch + window // 2)
    expected = echo1.copy()
    expected[t_switch:] = echo2[t_switch:]
    ramp = np.linspace(0.0, 1.0, hi - lo, dtype=np.float32)
    expected[lo:hi] = echo1[lo:hi] * (1.0 - ramp) + echo2[lo:hi] * ramp

    event = out["events"][0]
    assert event["type"] == "path_replace"
    assert event["rir2_kind"] == rir2_kind
    np.testing.assert_array_equal(out["echo"], expected)


def test_far_end_stop_rir_tail_decay():
    """far_end_stop_rir_tail：t1 后 echo 保留 RIR 尾衰减（非硬切零），且尾部能量递减。"""
    far, near, _, sr = _make_inputs(duration=1.0)
    n = len(far)
    # 50ms 指数衰减 RIR（T60≈50ms），确定性生成
    n_rir = int(0.05 * sr)
    rng_rir = np.random.default_rng(7)
    rir = (rng_rir.standard_normal(n_rir).astype(np.float32)
           * np.exp(-6.91 * np.arange(n_rir) / n_rir).astype(np.float32))
    rir[0] = 1.0

    t1_frac = 0.5  # 固定边界，事件时间可精确对齐
    out = SCENARIOS["far_end_stop_rir_tail"](
        far, near, rir, sr, np.random.default_rng(0), t1_frac=t1_frac)
    ev = out["events"][0]
    assert ev["type"] == "far_end_stop"
    t1 = int(t1_frac * n)
    assert abs(ev["t_ms"] - 1000.0 * t1 / sr) < 1e-6
    # farend 在 t1 后归零（场景定义）
    assert np.all(out["farend"][t1:] == 0.0)

    def seg_rms(seg: np.ndarray) -> float:
        return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))

    echo = out["echo"]
    pre = seg_rms(echo[t1 - 1600:t1])          # t1 前 100ms 稳态
    tail_early = seg_rms(echo[t1:t1 + 400])    # t1 后前 25ms（RIR 尾前半）
    tail_late = seg_rms(echo[t1 + 400:t1 + 800])  # RIR 尾后半
    # 非硬切：t1 后 echo 仍有明显余音（RIR 尾持续泄放）
    assert tail_early > 0.1 * pre
    # 随 RIR 尾衰减：尾后半能量显著低于前半
    assert tail_late < tail_early

pytestmark = [pytest.mark.unit, pytest.mark.integration]
