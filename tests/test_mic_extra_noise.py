"""Tests for mic_extra_noise / state label synchronization in AEC dynamic mixing."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch

from tasks.aec.data.dynamic_dataset import (
    DynamicMixingDataset,
    _load_stereo_farend_echo_impl,
    parse_aec_dynamic_pipeline,
    validate_aec_dataset_config,
)


SR = 16000

_VALID_AEC_SIMULATION_CFG = {
    "prob_target_only": 0.0,
    "prob_noise_only": 0.0,
    "prob_echo_only": 0.0,
    "prob_noise_echo_only": 0.0,
    "prob_target_noise": 0.0,
    "prob_target_echo": 0.0,
    "prob_target_noise_echo": 1.0,
}


@pytest.mark.parametrize(
    ("pipeline", "message"),
    [
        ("invalid", "mapping"),
        ({"mic_extra_noise": []}, "mapping"),
        ({"mic_extra_noise": {"enabled": "false"}}, "enabled"),
        ({"unknown": True}, "unknown"),
        ({"mic_extra_noise": {"unknown": True}}, "unknown"),
        ({"mic_extra_noise": {"snr_range": [2.0, 1.0]}}, "snr_range"),
        ({"mic_extra_noise": {"snr_range": [float("nan"), 1.0]}}, "snr_range"),
        ({"mic_extra_noise": {"snr_range": ["1", "2"]}}, "snr_range"),
    ],
)
def test_parse_aec_dynamic_pipeline_rejects_invalid_mic_extra_noise(pipeline, message):
    with pytest.raises(ValueError, match=message):
        parse_aec_dynamic_pipeline(pipeline)
    with pytest.raises(ValueError, match=message):
        validate_aec_dataset_config(
            simulation_cfg=_VALID_AEC_SIMULATION_CFG,
            aec_dynamic_pipeline=pipeline,
        )


def test_parse_aec_dynamic_pipeline_accepts_equal_snr_bounds_and_numpy_bool():
    pipeline = {"mic_extra_noise": {"enabled": np.bool_(True), "snr_range": [7, 7]}}

    parsed = parse_aec_dynamic_pipeline(pipeline)

    assert parsed == {
        "mic_extra_noise_enabled": True,
        "mic_extra_noise_snr_range": (7.0, 7.0),
    }
    validate_aec_dataset_config(
        simulation_cfg=_VALID_AEC_SIMULATION_CFG,
        aec_dynamic_pipeline=pipeline,
    )


def _make_dataset_stub(
    aec_pipe: dict[str, Any] | None = None,
    noise_paths: dict[str, str] | None = None,
) -> DynamicMixingDataset:
    """Create a bare-bones DynamicMixingDataset instance for unit testing."""
    ds = object.__new__(DynamicMixingDataset)
    ds._aec_pipe = aec_pipe or {"mic_extra_noise_enabled": False}
    ds.noise_flt = noise_paths or {}
    ds.noise_source = {SR: {uid: None for uid in ds.noise_flt}}
    ds.strict_load = True
    ds._epoch = 0
    ds._base_seed = 42
    return ds


def _write_noise_wav(path: str, duration: float = 1.0) -> None:
    samples = int(SR * duration)
    noise = np.random.randn(samples).astype(np.float32) * 0.1
    sf.write(path, noise, SR)


def test_mic_extra_noise_disabled_returns_none_extra():
    ds = _make_dataset_stub(aec_pipe={"mic_extra_noise_enabled": False})
    mic = np.ones(SR, dtype=np.float32)
    out, extra = ds._maybe_mic_extra_noise(mic, SR, 0, mic)
    assert out is mic
    assert extra is None


def test_mic_extra_noise_records_independent_rng_policy(monkeypatch):
    ds = _make_dataset_stub(
        aec_pipe={
            "mic_extra_noise_enabled": True,
            "mic_extra_noise_snr_range": (0.0, 0.0),
        },
        noise_paths={"first": "/tmp/first.wav", "second": "/tmp/second.wav"},
    )
    rng_calls: list[int] = []
    monkeypatch.setattr(
        ds,
        "_aec_rng_mic_extra_noise",
        lambda index: rng_calls.append(index) or np.random.default_rng(5),
    )
    monkeypatch.setattr(
        "tasks.aec.data.dynamic_dataset._load_audio_cached",
        lambda *_args: np.ones(SR * 2, dtype=np.float32),
    )

    mic = np.ones(SR, dtype=np.float32) * 0.1
    ds._maybe_mic_extra_noise(mic, SR, 7, mic)

    assert rng_calls == [7]
    assert ds._last_mic_extra_noise_policy.application_role.value == "aec_post_ef_mic_extra"
    assert ds._last_mic_extra_noise_policy.uid in {"first", "second"}
    assert ds._last_mic_extra_noise_policy.cache_provenance.value == "shared_audio_cache_unknown"


def test_mic_extra_result_ignores_stale_legacy_policy(monkeypatch):
    ds = _make_dataset_stub(
        aec_pipe={
            "mic_extra_noise_enabled": True,
            "mic_extra_noise_snr_range": (0.0, 0.0),
        },
        noise_paths={"current": "/tmp/current.wav"},
    )
    ds._last_mic_extra_noise_policy = "stale-invalid-policy"
    monkeypatch.setattr(ds, "_aec_rng_mic_extra_noise", lambda _index: np.random.default_rng(5))
    monkeypatch.setattr(
        "tasks.aec.data.dynamic_dataset._load_audio_cached",
        lambda *_args: np.ones(SR * 2, dtype=np.float32),
    )

    result = ds._maybe_mic_extra_noise_result(
        np.ones(SR, dtype=np.float32) * 0.1,
        SR,
        7,
        np.ones(SR, dtype=np.float32) * 0.1,
    )

    assert result.policy is not None
    assert result.policy.uid == "current"
    assert ds._last_mic_extra_noise_policy == "stale-invalid-policy"
    legacy_mic, legacy_extra = ds._maybe_mic_extra_noise(
        np.ones(SR, dtype=np.float32) * 0.1,
        SR,
        7,
        np.ones(SR, dtype=np.float32) * 0.1,
    )
    assert legacy_extra is not None
    assert legacy_mic.shape == result.mic.shape
    assert ds._last_mic_extra_noise_policy.uid == "current"


@pytest.mark.parametrize(
    ("fs_out", "noise_source", "expected_uid"),
    [
        (16000, {8000: {"low": None}, 16000: {"exact": None}, 48000: {"high": None}}, "exact"),
        (24000, {8000: {"low": None}, 48000: {"high": None}}, "high"),
        (96000, {48000: {"first": None}, 8000: {"second": None}}, "first"),
    ],
)
def test_mic_extra_noise_uses_prebuilt_ordinary_exact_higher_all_selector(
    monkeypatch, fs_out, noise_source, expected_uid
):
    class NoIterationNoisePaths(dict):
        def __iter__(self):
            raise AssertionError("mic-extra must not materialize tuple(self.noise_flt)")

    paths = NoIterationNoisePaths(
        {uid: f"/tmp/{uid}.wav" for by_fs in noise_source.values() for uid in by_fs}
    )
    ds = _make_dataset_stub(
        aec_pipe={
            "mic_extra_noise_enabled": True,
            "mic_extra_noise_snr_range": (0.0, 0.0),
        }
    )
    ds.noise_flt = paths
    ds.noise_source = noise_source

    class SelectorRng:
        def integers(self, low, high):
            assert low == 0
            return 0

        def uniform(self, low, high):
            return low

    monkeypatch.setattr(ds, "_aec_rng_mic_extra_noise", lambda _index: SelectorRng())
    monkeypatch.setattr(
        "tasks.aec.data.dynamic_dataset._load_audio_cached",
        lambda *_args: np.ones(SR * 2, dtype=np.float32),
    )

    mic = np.ones(SR, dtype=np.float32) * 0.1
    ds._maybe_mic_extra_noise(mic, fs_out, 7, mic)

    assert ds._last_mic_extra_noise_policy.uid == expected_uid
    assert ds._last_mic_extra_noise_policy.path == f"/tmp/{expected_uid}.wav"
    assert ds._last_mic_extra_noise_policy.family.value == "ordinary"


def test_mic_extra_noise_selector_cache_materializes_large_pool_once(monkeypatch):
    class CountingKeysDict(dict):
        def __init__(self, items):
            super().__init__(items)
            self.keys_calls = 0

        def keys(self):
            self.keys_calls += 1
            return super().keys()

    ordinary = CountingKeysDict((f"noise-{index}", None) for index in range(10_000))
    paths = {uid: f"/tmp/{uid}.wav" for uid in ordinary}
    ds = _make_dataset_stub(
        aec_pipe={
            "mic_extra_noise_enabled": True,
            "mic_extra_noise_snr_range": (0.0, 0.0),
        },
        noise_paths=paths,
    )
    ds.noise_source = {SR: ordinary}
    monkeypatch.setattr(
        "tasks.aec.data.dynamic_dataset._load_audio_cached",
        lambda *_args: np.ones(SR * 2, dtype=np.float32),
    )
    mic = np.ones(SR, dtype=np.float32) * 0.1

    ds._maybe_mic_extra_noise(mic, SR, 1, mic)
    ds._maybe_mic_extra_noise(mic, SR, 2, mic)

    assert ordinary.keys_calls == 1


def test_mic_extra_noise_enabled_returns_scaled_noise():
    with tempfile.TemporaryDirectory() as tmpdir:
        noise_path = str(Path(tmpdir) / "noise.wav")
        _write_noise_wav(noise_path, duration=2.0)

        ds = _make_dataset_stub(
            aec_pipe={
                "mic_extra_noise_enabled": True,
                "mic_extra_noise_snr_range": (5.0, 25.0),
            },
            noise_paths={"noise": noise_path},
        )
        mic = np.ones(SR, dtype=np.float32) * 0.1
        clean = np.ones(SR, dtype=np.float32) * 0.1
        out, extra = ds._maybe_mic_extra_noise(mic, SR, 0, clean)

        assert out.shape == mic.shape
        assert extra is not None
        assert extra.shape == mic.shape
        # The returned mic should be approximately speech + scaled noise.
        np.testing.assert_allclose(out, mic + extra, atol=1e-6)


def test_finalize_sample_does_not_change_acoustic_signals(monkeypatch):
    ds = _make_dataset_stub(
        aec_pipe={
            "mic_extra_noise_enabled": True,
            "mic_extra_noise_snr_range": (-20.0, -20.0),
        }
    )
    ds.state_detector_hop_length = None
    ds._last_simulation_stems = None
    monkeypatch.setattr(
        ds,
        "_maybe_mic_extra_noise",
        lambda *_args, **_kwargs: pytest.fail("_finalize_sample must not augment mic"),
    )

    mic = np.linspace(-0.8, 0.8, 32, dtype=np.float32)
    farend = np.linspace(0.3, -0.3, 32, dtype=np.float32)
    clean = np.linspace(-0.2, 0.2, 32, dtype=np.float32)
    echo = mic - clean
    out = ds._finalize_sample(mic, farend, clean, echo, SR, 0, "test")

    np.testing.assert_array_equal(out[0].numpy(), clean)
    np.testing.assert_array_equal(out[1].numpy(), mic)
    np.testing.assert_array_equal(out[2].numpy(), farend)


def _write_scp(path: Path, entries: dict[str, Path]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for uid, audio_path in entries.items():
            info = sf.info(str(audio_path))
            f.write(f"{uid} {info.samplerate} {info.frames} {audio_path}\n")


@pytest.mark.parametrize("task", ["joint", "aec"])
def test_mic_extra_noise_stays_in_final_stems_with_deferred_frontend(tmp_path, task):
    t = np.arange(SR, dtype=np.float32) / SR
    speech = (0.6 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    farend = (0.4 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    # 与 farend 使用不同频率，确保后续 echo 的增益归一化校验可检测声道反转。
    echo = (0.2 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    extra_source = (0.8 * np.sin(2 * np.pi * 900 * t)).astype(np.float32)
    impulse = np.zeros(SR, dtype=np.float32)
    impulse[0] = 1.0

    wavs = {}
    for name, audio in {
        "speech": speech,
        "noise": extra_source,
        "rir": impulse,
        "wind": extra_source,
        "farend": farend,
        "echo": echo,
    }.items():
        path = tmp_path / f"{name}.wav"
        sf.write(path, audio, SR, subtype="FLOAT")
        wavs[name] = path

    scps = {}
    for name in ("speech", "noise", "rir", "wind"):
        scp = tmp_path / f"{name}.scp"
        _write_scp(scp, {name: wavs[name]})
        scps[name] = scp
    pair_json = tmp_path / "pairs.json"
    # stereo manifest 的固定语义：ch0=echo，ch1=farend。保留独立 oracle，
    # 后续直接核验 Dataset 导出的 stem，而不是用 stem 自身构造期望值。
    expected_echo_ch0 = echo.copy()
    expected_farend_ch1 = farend.copy()
    stereo_wav = tmp_path / "stereo.wav"
    sf.write(
        stereo_wav,
        np.stack([expected_echo_ch0, expected_farend_ch1], axis=1),
        SR,
        subtype="FLOAT",
    )
    pair_json.write_text(
        json.dumps({"pairs": [str(stereo_wav)]}),
        encoding="utf-8",
    )
    loaded_farend, loaded_echo = _load_stereo_farend_echo_impl(
        str(stereo_wav), SR, True, ch_farend=1, ch_echo=0
    )
    np.testing.assert_allclose(loaded_echo, expected_echo_ch0, atol=1e-7)
    np.testing.assert_allclose(loaded_farend, expected_farend_ch1, atol=1e-7)

    ds = DynamicMixingDataset(
        speech_scp=str(scps["speech"]),
        noise_scp=str(scps["noise"]),
        rir_scp=str(scps["rir"]),
        windnoise_scp=str(scps["wind"]),
        real_farend_echo_pair_json=str(pair_json),
        target_sr=SR,
        max_duration_seconds=1.0,
        strict_load=True,
        task=task,
        aec_dynamic_pipeline={
            "mic_extra_noise": {"enabled": True, "snr_range": [-20.0, -20.0]}
        },
        simulation_cfg={
            "prob_target_only": 0.0,
            "prob_noise_only": 0.0,
            "prob_echo_only": 0.0,
            "prob_noise_echo_only": 0.0,
            "prob_target_noise": 0.0,
            "prob_target_echo": 1.0,
            "prob_target_noise_echo": 0.0,
            "prob_reverberation": 0.0,
            "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": False,
            "use_high_pass": False,
            "ser_range": [6.0, 6.0],
            # mic-extra 职责不约束 ERL；覆盖 physical range 以保持旧抽样。
            "erl_range": [-60.0, 60.0],
        },
    )
    ds.set_epoch(0, base_seed=123)
    ds.export_simulation_stems = True
    ds._last_mic_extra_noise_policy = "stale-invalid-policy"

    sample = ds[0]
    mic = sample[1].numpy()
    clean = sample[0].numpy()
    error = sample[5]
    err_mask = sample[6]
    stems = ds._last_simulation_stems
    final_noise = np.asarray(stems["final_noise"])
    final_echo = np.asarray(stems["final_echo"])
    length = int(sample[4])

    assert torch.count_nonzero(error) == 0
    assert err_mask is False
    # final_echo 必须源自 manifest ch0 的 echo；farend 则来自 ch1。echo 会按
    # 随机 ERL 缩放，故以独立 ch0 oracle 投影求增益后核验完整波形形状。
    echo_oracle = expected_echo_ch0[:length]
    echo_gain = float(
        np.dot(final_echo[:length], echo_oracle) / np.dot(echo_oracle, echo_oracle)
    )
    np.testing.assert_allclose(final_echo[:length], echo_gain * echo_oracle, atol=2e-5)
    farend_oracle = expected_farend_ch1[:length]
    farend_gain = float(
        np.dot(sample[2].numpy()[:length], farend_oracle)
        / np.dot(farend_oracle, farend_oracle)
    )
    np.testing.assert_allclose(
        sample[2].numpy()[:length], farend_gain * farend_oracle, atol=2e-5
    )
    expected_mic = clean[:length] + echo_gain * echo_oracle
    if task == "joint":
        expected_mic = expected_mic + final_noise[:length]
    threshold = float(np.float32(0.99))
    expected_peak = float(np.max(np.abs(expected_mic)))
    expected_scale = threshold / expected_peak if expected_peak > threshold else 1.0
    trace = ds._last_simulation_trace.sources["scene"].metadata
    assert trace["final_peak_guard"] == {
        "roles": ["mic"],
        "threshold": threshold,
        "peak": pytest.approx(expected_peak),
        "scale": pytest.approx(expected_scale),
        "applied": expected_peak > threshold,
    }
    final_guard_scale = trace["final_peak_guard"]["scale"]
    expected_mic = expected_mic * final_guard_scale
    np.testing.assert_allclose(mic[:length], expected_mic, atol=2e-5)
    np.testing.assert_allclose(ds._last_noise_additive_1d[:length], final_noise[:length])
    mic_extra = trace["aec_post_ef"]["mic_extra_noise"]
    assert trace["aec_post_ef"]["stage"] == "task-post-EF"
    assert mic_extra["family"] == "ordinary"
    assert mic_extra["application_role"] == "aec_post_ef_mic_extra"
    assert mic_extra["cache_provenance"] == "shared_audio_cache_unknown"
    assert mic_extra["uid"] == "noise"
    assert all(
        source["uid"] != "noise"
        for source in trace["acoustic_scene_plan"]["noise"]["sources"]
    )
    expected_reasons = ["task_post_ef_mic_extra_noise"]
    if final_guard_scale < 1.0:
        expected_reasons.append("final_guard_observation_only_scale")
    assert trace["aec_context"]["decomposable"] is False
    assert list(trace["aec_context"]["decomposable_reason_codes"]) == expected_reasons
    for realized in (stems["simulation_by_layer"]["realized"], stems["realized_final"]):
        assert realized["decomposable"] is False
        assert realized["decomposable_reason_codes"] == expected_reasons
        assert realized["ser_db"] is None
        assert realized["ser_reason_code"] == "mic_not_decomposable"
        assert realized["snr_db"] is None
        assert realized["snr_reason_code"] == "mic_not_decomposable"
        assert realized["reconstruction_residual"]["applicable"] is False
        assert realized["reconstruction_residual"]["reason_code"] \
            == "mic_not_decomposable"
        assert realized["erl_db"] is not None
        assert realized["erl_applicable"] is True
    assert float(np.max(np.abs(mic[:length]))) <= 0.99 + 1e-6
    assert float(np.max(np.abs(final_noise[:length]))) > 0.1

pytestmark = [pytest.mark.unit]
