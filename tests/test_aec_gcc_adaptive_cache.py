"""AEC GCC 自适应工作采样率与跨 epoch 缓存回归测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import numpy as np
import pytest
import soundfile
import torch

from tasks.aec.inference.gcc_phat_core import (
    estimate_gcc_tau_windowed,
    resolve_gcc_downsample,
)
from tasks.aec.data.dynamic_aec_build import (
    build_dynamic_aec_dataset,
    validate_aec_dynamic_aec_config,
)
from tasks.aec.data.dynamic_dataset import DynamicMixingDataset, validate_aec_dataset_config
from tasks.aec.data.time_delay_jitter import TimeDelayJitterConfig

pytestmark = pytest.mark.unit


def _delayed_pair(sr: int, delay_ms: float, duration_s: float = 2.0):
    rng = np.random.default_rng(17)
    n = int(sr * duration_s)
    delay = int(round(delay_ms * sr / 1000.0))
    far = (rng.standard_normal(n) * 0.2).astype(np.float32)
    echo = np.zeros_like(far)
    echo[delay:] = far[:-delay]
    return far, echo


def test_gcc_auto_downsample_targets_4khz_and_preserves_tau():
    """16k/32k 自动降采样均约为 4k，物理延迟保持一致。"""
    assert resolve_gcc_downsample(16000) == (4, 4000.0)
    assert resolve_gcc_downsample(32000) == (8, 4000.0)
    assert resolve_gcc_downsample(32000, downsample=4) == (4, 8000.0)
    # 非整倍数采用最近整数倍率（半数向上），且不做上采样。
    ds, work_sr = resolve_gcc_downsample(22050)
    assert ds == 6
    assert work_sr == pytest.approx(3675.0)

    taus = []
    for sr in (16000, 32000):
        far, echo = _delayed_pair(sr, 50.0)
        estimate = estimate_gcc_tau_windowed(
            far, echo, sr, max_tau_seconds=0.2
        )
        taus.append(estimate.tau_sec)
    assert taus[0] == pytest.approx(0.05, abs=0.001)
    assert taus[1] == pytest.approx(0.05, abs=0.001)
    assert taus[0] == pytest.approx(taus[1], abs=0.001)


def _build_cache_probe(
    tmp_path: Path,
    *,
    pair_rows: list[str] | None = None,
    strict_load: bool = True,
    **dataset_kwargs,
) -> DynamicMixingDataset:
    """构造只覆盖 GCC 对齐路径的最小 AEC Dataset。"""
    sr = 16000
    far, echo = _delayed_pair(sr, 50.0, duration_s=1.0)
    far_path = tmp_path / "far.wav"
    echo_path = tmp_path / "echo.wav"
    soundfile.write(far_path, far, sr)
    soundfile.write(echo_path, echo, sr)
    stereo_path = tmp_path / "stereo.wav"
    soundfile.write(stereo_path, np.stack([echo, far], axis=1), sr)

    speech_path = tmp_path / "speech.wav"
    noise_path = tmp_path / "noise.wav"
    rir_path = tmp_path / "rir.wav"
    soundfile.write(speech_path, np.ones(sr, np.float32) * 0.1, sr)
    soundfile.write(noise_path, np.ones(sr, np.float32) * 0.01, sr)
    rir = np.zeros(sr, np.float32)
    rir[0] = 1.0
    soundfile.write(rir_path, rir, sr)

    def write_scp(path: Path, uid: str, audio: Path) -> None:
        info = soundfile.info(str(audio))
        path.write_text(f"{uid} {sr} {info.frames} {audio}\n", encoding="utf-8")

    speech_scp = tmp_path / "speech.scp"
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    write_scp(speech_scp, "speech", speech_path)
    write_scp(noise_scp, "noise", noise_path)
    write_scp(rir_scp, "rir", rir_path)
    pair_json = tmp_path / "pairs.json"
    rows = pair_rows or [stereo_path.name]
    pair_json.write_text(json.dumps({"pairs": rows}), encoding="utf-8")
    cfg = {
        "prob_target_only": 0.0,
        "prob_noise_only": 0.0,
        "prob_echo_only": 1.0,
        "prob_noise_echo_only": 0.0,
        "prob_target_noise": 0.0,
        "prob_target_echo": 0.0,
        "prob_target_noise_echo": 0.0,
        "prob_reverberation": 0.0,
        "prob_wind_noise_sidechain": 0.0,
        "prob_channel_enabled": False,
        "noisy_post_mix_degradations_enabled": False,
    }
    return DynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        simulation_cfg=cfg,
        real_farend_echo_pair_json=str(pair_json),
        target_sr=sr,
        max_duration_seconds=0.5,
        strict_load=strict_load,
        **dataset_kwargs,
    )


def test_manifest_relative_pairs_resolve_from_manifest_parent(tmp_path):
    ds = _build_cache_probe(tmp_path)
    canonical = str((tmp_path / "stereo.wav").resolve())
    assert ds._farend_echo_pairs == [(canonical, canonical)]


def test_manifest_strict_mode_rejects_unusable_row(tmp_path):
    with pytest.raises(ValueError, match=r"manifest row 0.*unusable farend"):
        _build_cache_probe(
            tmp_path,
            pair_rows=["missing.wav", "stereo.wav"],
        )


def test_manifest_lenient_mode_keeps_branch_and_records_diagnostics(tmp_path):
    ds = _build_cache_probe(
        tmp_path,
        pair_rows=["missing.wav", "stereo.wav"],
        strict_load=False,
    )
    canonical = str((tmp_path / "stereo.wav").resolve())
    assert ds._farend_echo_pairs == [(canonical, canonical)]
    assert ds._aec_ref_echo_stereo_wav is True
    assert ds._farend_echo_manifest_diagnostics
    assert "manifest row 0" in ds._farend_echo_manifest_diagnostics[0]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("aec_ref_echo_gcc_bias_seconds", np.nan),
        ("aec_ref_echo_gcc_max_tau_seconds", np.inf),
        ("aec_ref_echo_gcc_max_pad_seconds", -1.0),
        ("aec_ref_echo_gcc_min_pad_samples", -1),
    ],
)
def test_gcc_delay_config_rejects_nonfinite_or_negative_values(name, value):
    with pytest.raises(ValueError):
        validate_aec_dataset_config(**{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("state_detector_near_threshold_db", np.nan),
        ("state_detector_echo_threshold_db", np.inf),
        ("state_detector_noise_threshold_db", -np.inf),
    ],
)
def test_state_detector_thresholds_reject_nonfinite_values(name, value):
    with pytest.raises(ValueError, match=name):
        validate_aec_dataset_config(
            simulation_cfg={"prob_target_only": 1.0},
            **{name: value},
        )


@pytest.mark.parametrize("value", ["false", "true"])
def test_fine_gcc_align_rejects_string_boolean(value):
    with pytest.raises(ValueError, match="aec_ref_echo_fine_gcc_align.*bool"):
        validate_aec_dataset_config(
            simulation_cfg={"prob_target_only": 1.0},
            aec_ref_echo_fine_gcc_align=value,
        )


def test_aec_constructor_forwards_valid_validator_fields(tmp_path):
    ds = _build_cache_probe(
        tmp_path,
        aec_ref_echo_fine_gcc_align=False,
        state_detector_near_threshold_db=-31.0,
        state_detector_echo_threshold_db=-47.5,
        state_detector_noise_threshold_db=-36,
    )

    assert ds._aec_ref_echo_fine_gcc_align is False
    assert ds.state_detector_near_threshold_db == -31.0
    assert ds.state_detector_echo_threshold_db == -47.5
    assert ds.state_detector_noise_threshold_db == -36.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("state_detector_near_threshold_db", np.nan),
        ("state_detector_echo_threshold_db", np.inf),
        ("state_detector_noise_threshold_db", -np.inf),
        ("aec_ref_echo_fine_gcc_align", "false"),
    ],
)
def test_aec_constructor_rejects_invalid_validator_fields(tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        _build_cache_probe(tmp_path, **{name: value})


def test_gcc_cache_survives_epoch_and_parameter_changes_miss(tmp_path, monkeypatch):
    """纯函数 GCC 结果跨 epoch 命中，任一 GCC 参数变化必须 miss。"""
    ds = _build_cache_probe(tmp_path)
    calls = {"n": 0}
    original = ds._estimate_gcc_tau_windowed

    def counted(self, wf, we, target_sr):
        calls["n"] += 1
        return original(wf, we, target_sr)

    ds._estimate_gcc_tau_windowed = MethodType(counted, ds)
    rng = np.random.default_rng(3)
    ds._load_farend_echo_pair_aligned(rng, 16000, 4000)
    assert calls["n"] == 1

    ds.set_epoch(1, base_seed=9)
    ds._load_farend_echo_pair_aligned(np.random.default_rng(3), 16000, 4000)
    assert calls["n"] == 1

    ds._aec_ref_echo_gcc_prominence_db += 1.0
    ds._load_farend_echo_pair_aligned(np.random.default_rng(3), 16000, 4000)
    assert calls["n"] == 2

    # stereo/audio cache 的 epoch 清理语义仍保留，GCC cache 不被清空。
    assert len(ds._gcc_cache) == 2
    monkeypatch.undo()


def test_gcc_estimation_failure_is_retried_and_success_is_cached(tmp_path):
    """瞬态 GCC 异常只影响当前调用，下一次必须重新估计并可缓存成功值。"""
    ds = _build_cache_probe(tmp_path)
    original = ds._estimate_gcc_tau_windowed
    calls = {"n": 0}

    def flaky(self, wf, we, target_sr):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient GCC failure")
        return original(wf, we, target_sr)

    ds._estimate_gcc_tau_windowed = MethodType(flaky, ds)
    ds._load_farend_echo_pair_aligned(np.random.default_rng(3), 16000, 4000)
    assert calls["n"] == 1
    assert not ds._gcc_cache

    ds._load_farend_echo_pair_aligned(np.random.default_rng(3), 16000, 4000)
    assert calls["n"] == 2
    assert len(ds._gcc_cache) == 1


def test_lenient_exhausted_echo_load_enters_sample_placeholder(tmp_path, monkeypatch):
    """普通 echo_only 加载耗尽时返回 placeholder，显式 forced 分支仍快速失败。"""
    ds = _build_cache_probe(tmp_path, strict_load=False)
    load_calls = {"n": 0}

    def silent_load(*_args, **_kwargs):
        load_calls["n"] += 1
        silence = np.zeros(16000, dtype=np.float32)
        return silence, silence

    silent_load.cache_clear = lambda: None
    monkeypatch.setattr(
        "tasks.aec.data.dynamic_dataset._load_stereo_farend_echo_cached_impl", silent_load
    )

    traced = ds.get_sample_with_trace(0)

    assert load_calls["n"] == 5  # 每次尝试从同一 stereo 文件加载两个声道。
    assert traced.trace.placeholder is True
    assert "all 5 attempts failed" in traced.trace.metadata["placeholder"]["reason"]
    assert traced.sample[10] is False
    assert traced.sample[8] is False
    assert torch.count_nonzero(traced.sample[1]) == 0
    assert torch.count_nonzero(traced.sample[2]) == 0

    with pytest.raises(RuntimeError, match="all 5 attempts failed"):
        ds.get_sample_with_trace(0, force_branch="prob_echo_only")
    assert load_calls["n"] == 10


def test_gcc_cache_hit_does_not_stat_or_resolve_inputs(tmp_path):
    """同一构造快照的 cache hit 不重复 stat/resolve manifest 或 pair 文件。"""
    ds = _build_cache_probe(tmp_path)
    ds._load_farend_echo_pair_aligned(np.random.default_rng(3), 16000, 4000)

    with patch("tasks.aec.data.dynamic_dataset.Path.stat", side_effect=AssertionError("stat on hit")), \
         patch("tasks.aec.data.dynamic_dataset.Path.resolve", side_effect=AssertionError("resolve on hit")):
        ds._load_farend_echo_pair_aligned(np.random.default_rng(3), 16000, 4000)


def test_manifest_strict_bad_schema_fails_and_lenient_only_diagnoses(tmp_path):
    rows = [{"farend": "far.wav", "echo": "echo.wav"}]
    with pytest.raises(ValueError, match=r'JSON "pairs"\[0\].*stereo path string'):
        _build_cache_probe(tmp_path, pair_rows=rows)
    with pytest.raises(ValueError, match=r'JSON "pairs"\[0\].*stereo path string'):
        _build_cache_probe(tmp_path, pair_rows=rows, strict_load=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ser_range": [0.0]}, "ser_range"),
        ({"ser_range": [0.0, np.nan]}, "finite"),
        ({"aec_ref_echo_random_delay_beta_a": np.inf}, "beta_a"),
        ({"aec_ref_echo_random_delay_beta_b": True}, "beta_b"),
        ({"aec_echo_device_eq": {"enabled": False, "probb": 0.2}}, "unknown keys"),
        ({"aec_echo_device_eq": {"enabled": True, "prob": np.nan}}, "finite"),
        (
            {
                "aec_echo_device_eq": {
                    "enabled": True,
                    "variant_probs": {"random_eq": -1.0},
                }
            },
            "non-negative",
        ),
        ({"aec_echo_clipping": {"enabled": False, "path_prob": {}}}, "unknown keys"),
    ],
)
def test_aec_peripheral_config_rejects_malformed_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate_aec_dataset_config(
            simulation_cfg={"prob_target_only": 1.0},
            **kwargs,
        )


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"enabled": "false"},
        {"prob": True},
        {"jitter_rate": [0.1, "0.5"]},
        {"jitter_type_probs": {"sudden_jump": np.nan}},
    ],
)
def test_time_delay_jitter_config_rejects_coercive_or_nonfinite_values(raw):
    with pytest.raises(ValueError):
        TimeDelayJitterConfig.from_dict(raw)


def _dynamic_build_cfg_with_typo() -> dict:
    section = {
        "real_farend_echo_pair_json": "dummy.json",
        "speech_scp": "speech.scp",
        "noise_scp": "noise.scp",
        "rir_scp": "rir.scp",
        "simulation_cfg": {"prob_target_only": 1.0},
        "ser_ragne": [-20.0, 20.0],
    }
    return {
        "data": {
            "mode": "dynamic_aec",
            "sample_rate": 16000,
            "train": section,
            "validation": dict(section),
        }
    }


def test_dynamic_aec_unknown_key_precheck_matches_validation_and_build():
    cfg = _dynamic_build_cfg_with_typo()
    with pytest.raises(ValueError, match=r"ser_ragne.*ser_range"):
        validate_aec_dynamic_aec_config(cfg)
    with pytest.raises(ValueError, match=r"ser_ragne.*ser_range"):
        build_dynamic_aec_dataset(cfg, "train")
