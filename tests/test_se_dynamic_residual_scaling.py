"""SE 动态混音 residual stem 的 peak normalization 与 Layer-D 回归测试。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import core.data.dynamic_dataset_base as dynamic_base
import core.data.simulation.background as background_mix
import tasks.se.data.post_mix_adapter as post_mix_adapter
from core.data.dynamic_dataset_base import (
    BaseDynamicMixingDataset,
    LayerEOperationResult,
    LayerEResult,
    MixAcousticOutput,
    PreparedSpeech,
)
from core.data.peak_normalization import PeakNormalizationResult
from core.data.simulation_configs import SimulationConfigs
from tasks.se.data.dynamic_dataset import DynamicMixingDataset
from tasks.se.data.post_mix_adapter import (
    ADDITIVE,
    OBSERVATION,
    RESIDUAL,
    SE_POST_MIX_STAGE_ORDER,
    TARGET,
    TaskPostMixAdapter,
    run_se_post_mix_pipeline,
)


pytestmark = [pytest.mark.unit]


def _validated_sim_cfg(**kwargs) -> SimulationConfigs:
    """stub dataset 的 sim_cfg 必须是已校验的 SimulationConfigs。"""
    return SimulationConfigs.from_dict(kwargs, strict=True)


def _run_with_acoustic_stems(
    noisy_speech: np.ndarray,
    noisy_after_rir: np.ndarray,
    peak_normalization_scale: float,
    peak_normalization_mode: str = "joint",
    *,
    branch_path: str = "prob_target_noise",
    layer_ef_inputs: list[np.ndarray] | None = None,
    layer_ef_scale: float = 1.0,
    reported_noisy_scale: float | None = None,
    legacy_sidecar: bool = True,
) -> dict[str, np.ndarray] | None:
    noisy_speech = np.asarray(noisy_speech, dtype=np.float32).reshape(1, -1)
    noisy_after_rir = np.asarray(noisy_after_rir, dtype=np.float32).reshape(1, -1)
    actual_len = noisy_speech.shape[-1]
    clean_target = np.full_like(noisy_speech, 0.25)
    additive_noise = np.full_like(noisy_speech, 0.125)

    dataset = DynamicMixingDataset.__new__(DynamicMixingDataset)
    dataset.sim_cfg = _validated_sim_cfg(
        prob_target_only=0.0,
        prob_noise_only=0.0,
        prob_target_noise=1.0,
        prob_channel_enabled=False,
        noisy_post_mix_degradations_enabled=False,
    )
    dataset.export_simulation_stems = legacy_sidecar
    dataset._last_simulation_stems = {} if legacy_sidecar else None
    dataset._max_duration_for_sr = lambda sr: actual_len
    dataset._get_rng = lambda index: np.random.default_rng(0)
    dataset.speech_preparation = SimpleNamespace(
        prepare_result=lambda *_args, **_kwargs: PreparedSpeech(
            speech_uid="speech",
            sample_rate=16000,
            speech_1d=clean_target.flatten(),
            speech_2d=clean_target,
            actual_len=actual_len,
            fixed_silence_canvas_active=False,
            speech_splice_export={},
            repeat_export={},
        )
    )
    dataset._select_branch = lambda choices, rng, **kwargs: (branch_path, None)
    acoustic = SimpleNamespace(
        noisy_speech=noisy_speech,
        clean_target=clean_target,
        noisy_after_rir=noisy_after_rir,
        additive_noise_2d=additive_noise,
        used_sidechain_wind=False,
        use_reverb=False,
        wn_snr=0.0,
        additive_snr_db_list=[],
        mix_two=False,
        want_sidechain=False,
        snr_primary_drawn_db=None,
        wind_noise_snr_candidate_db=None,
        source_volume_calibration={"stage": "post_rir_pre_noise_echo"},
        acoustic_audit={"colored_noise": None},
    )
    dataset._mix_acoustic = lambda *args: acoustic
    dataset._align_mix_acoustic_outputs = lambda *args: args[:4]
    dataset._guard_finite = lambda noisy, clean, uid, stage: (noisy, clean)

    def apply_peak_normalization_result(noisy, clean, noise, rng):
        noisy_scale = peak_normalization_scale if peak_normalization_mode == "joint" else 1.0
        if peak_normalization_mode == "separate" and np.any(
            np.asarray(noisy) != 0.0
        ):
            noisy_scale = peak_normalization_scale
        clean_scale = (
            peak_normalization_scale
            if peak_normalization_mode in {"joint", "separate", "single"}
            else 1.0
        )
        noise_scale = peak_normalization_scale if peak_normalization_mode == "joint" else 1.0
        waveform_noisy_scale = noisy_scale
        if reported_noisy_scale is not None:
            noisy_scale = reported_noisy_scale
        return PeakNormalizationResult(
            noisy=np.asarray(noisy * waveform_noisy_scale, dtype=np.float32),
            clean=np.asarray(clean * clean_scale, dtype=np.float32),
            noise=np.asarray(noise * noise_scale, dtype=np.float32),
            mode=peak_normalization_mode,
            target_peak=None if peak_normalization_mode == "off" else 0.75,
            noisy_scale=noisy_scale,
            clean_scale=clean_scale,
            noise_scale=noise_scale,
            preserves_decomposition=peak_normalization_mode in {"off", "joint"},
        )

    dataset._apply_peak_normalization_result = apply_peak_normalization_result

    def apply_layer_e(noisy, sr, length, rng, sidechain):
        if layer_ef_inputs is not None:
            layer_ef_inputs.append(np.asarray(noisy, dtype=np.float32).copy())
        return LayerEResult(
            noisy * layer_ef_scale,
            False,
            0,
            (),
            (),
            bool(layer_ef_scale != 1.0),
            None,
        )

    dataset._apply_channel_degradation_result = apply_layer_e

    dataset._run("speech", actual_len, 16000, index=0)
    return dataset._last_simulation_stems


def test_trace_less_run_without_layer_a_sidecar_keeps_stems_none():
    stems = _run_with_acoustic_stems(
        noisy_speech=np.array([0.5, -0.25, 0.0, 0.0]),
        noisy_after_rir=np.array([0.25, -0.125, 0.0, 0.0]),
        peak_normalization_scale=1.0,
        legacy_sidecar=False,
    )

    assert stems is None


def test_se_post_mix_stage_order_and_rng_cursor(monkeypatch):
    """固定 post-mix stage 顺序、波形与 RNG cursor。"""
    actual_len = 4
    dataset = DynamicMixingDataset.__new__(DynamicMixingDataset)
    dataset.sim_cfg = _validated_sim_cfg(
        prob_target_only=0.0,
        prob_noise_only=0.0,
        prob_target_noise=1.0,
        prob_channel_enabled=False,
        noisy_post_mix_degradations_enabled=False,
    )
    dataset._last_simulation_stems = {}
    dataset._max_duration_for_sr = lambda _sr: actual_len
    rng = np.random.default_rng(20260826)
    dataset._get_rng = lambda _index: rng
    clean = np.array([[0.25, -0.25, 0.125, 0.0]], dtype=np.float32)
    noisy = np.array([[0.5, -0.5, 0.25, 0.0]], dtype=np.float32)
    after_rir = np.array([[0.25, -0.25, 0.125, 0.0]], dtype=np.float32)
    additive = np.array([[0.125, -0.125, 0.0625, 0.0]], dtype=np.float32)
    dataset.speech_preparation = SimpleNamespace(
        prepare_result=lambda *_args, **_kwargs: PreparedSpeech(
            speech_uid="speech",
            sample_rate=16000,
            speech_1d=clean.ravel(),
            speech_2d=clean,
            actual_len=actual_len,
            fixed_silence_canvas_active=False,
            speech_splice_export={},
            repeat_export={},
        )
    )
    dataset._select_branch = lambda *_args, **_kwargs: ("prob_target_noise", None)
    dataset._mix_acoustic = lambda *_args: SimpleNamespace(
        noisy_speech=noisy,
        clean_target=clean,
        noisy_after_rir=after_rir,
        additive_noise_2d=additive,
        used_sidechain_wind=False,
        use_reverb=False,
        wn_snr=0.0,
        additive_snr_db_list=[],
        mix_two=False,
        want_sidechain=False,
        snr_primary_drawn_db=None,
        wind_noise_snr_candidate_db=None,
        source_volume_calibration=None,
        acoustic_audit=None,
    )
    dataset._align_mix_acoustic_outputs = lambda *args: args[:4]

    calls: list[str] = []

    def finite_guard(observation, target, _uid, label):
        calls.append("pre-finite" if label == "mix pipeline" else "post-finite")
        return observation, target

    def d_normalize(observation, target, noise_stem, stage_rng):
        calls.append("D-normalize")
        scale = float(stage_rng.uniform(0.5, 0.5))
        return PeakNormalizationResult(
            noisy=(observation * scale).astype(np.float32),
            clean=(target * scale).astype(np.float32),
            noise=(noise_stem * scale).astype(np.float32),
            mode="joint",
            target_peak=0.5,
            noisy_scale=scale,
            clean_scale=scale,
            noise_scale=scale,
            preserves_decomposition=True,
        )

    def layer_e(observation, _sr, _length, stage_rng, _sidechain):
        calls.append("E")
        return LayerEResult(observation, False, 0, (), (), False, None)

    def layer_f(observation, _sr, _cfg, **kwargs):
        calls.append("F")
        gain = np.float32(kwargs["rng"].uniform(0.75, 0.75))
        return observation * gain

    real_peak_guard = post_mix_adapter.apply_deterministic_peak_guard
    guard_calls = 0

    def record_peak_guard(waveforms):
        nonlocal guard_calls
        calls.append("D-guard" if guard_calls == 0 else "final-guard")
        guard_calls += 1
        return real_peak_guard(waveforms)

    dataset._guard_finite = finite_guard
    dataset._apply_peak_normalization_result = d_normalize
    dataset._apply_channel_degradation_result = layer_e
    monkeypatch.setattr(post_mix_adapter, "apply_deterministic_peak_guard", record_peak_guard)
    monkeypatch.setattr(post_mix_adapter, "apply_post_augmentations", layer_f)

    clean_out, noisy_out, sr_out = dataset._run("speech", actual_len, 16000, index=0)
    expected_rng = np.random.default_rng(20260826)
    expected_rng.uniform(0.5, 0.5)
    expected_rng.uniform(0.75, 0.75)

    assert calls == [
        "pre-finite",
        "D-normalize",
        "D-guard",
        "E",
        "F",
        "final-guard",
        "post-finite",
    ]
    assert rng.bit_generator.state == expected_rng.bit_generator.state
    assert sr_out == 16000
    np.testing.assert_array_equal(clean_out, clean.ravel() * np.float32(0.5))
    np.testing.assert_array_equal(noisy_out, noisy.ravel() * np.float32(0.375))
    np.testing.assert_array_equal(dataset._last_simulation_stems["final_clean"], clean_out)
    np.testing.assert_array_equal(dataset._last_simulation_stems["final_noisy"], noisy_out)


def test_se_post_mix_adapter_maps_roles_and_preserves_fixed_order(monkeypatch):
    """Adapter 纯测：四 role followers 与显式 A–F facts。"""
    calls: list[str] = []
    rng = np.random.default_rng(20260826)
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = _validated_sim_cfg(
        noisy_post_mix_degradations_enabled=True,
    )
    dataset._guard_finite = lambda observation, target, _uid, label: (
        calls.append("pre-finite" if label == "mix pipeline" else "post-finite")
        or (observation, target)
    )

    def d_normalize(observation, target, additive, stage_rng):
        calls.append("D-normalize")
        scale = float(stage_rng.uniform(0.5, 0.5))
        return PeakNormalizationResult(
            *(np.asarray(item * scale, dtype=np.float32) for item in (observation, target, additive)),
            mode="joint",
            target_peak=0.5,
            noisy_scale=scale,
            clean_scale=scale,
            noise_scale=scale,
            preserves_decomposition=True,
        )

    dataset._apply_peak_normalization_result = d_normalize

    real_peak_guard = post_mix_adapter.apply_deterministic_peak_guard
    guard_calls = 0

    def record_peak_guard(waveforms):
        nonlocal guard_calls
        calls.append("D-guard" if guard_calls == 0 else "final-guard")
        guard_calls += 1
        return real_peak_guard(waveforms)

    monkeypatch.setattr(post_mix_adapter, "apply_deterministic_peak_guard", record_peak_guard)

    def layer_e(observation, _sr, _length, stage_rng, _sidechain):
        calls.append("E")
        gain = np.float32(stage_rng.uniform(0.75, 0.75))
        waveform = observation * gain
        return LayerEResult(
            waveform,
            True,
            1,
            ("clipping",),
            (LayerEOperationResult("clipping", True, True),),
            True,
            None,
        )

    dataset._apply_channel_degradation_result = layer_e

    def layer_f(observation, _sr, layer_f_cfg, **kwargs):
        calls.append("F")
        assert kwargs["skip_clip"] is True
        assert layer_f_cfg.prob_clip == 0.0
        kwargs["applied_ops_out"].append("post_eq")
        gain = np.float32(kwargs["rng"].uniform(0.8, 0.8))
        return observation * gain

    monkeypatch.setattr(post_mix_adapter, "apply_post_augmentations", layer_f)
    adapter = TaskPostMixAdapter(
        dataset=dataset,
        rng=rng,
        sample_rate=16000,
        actual_len=4,
        speech_uid="speech",
        branch="prob_target_noise",
        used_sidechain_wind=False,
        metadata={"branch": "prob_target_noise"},
    )
    bundle = adapter.prepare_initial_bundle(
        observation=np.array([0.5, -0.5, 0.25, 0.0], dtype=np.float32),
        target=np.array([0.25, -0.25, 0.125, 0.0], dtype=np.float32),
        additive=np.array([0.125, -0.125, 0.0625, 0.0], dtype=np.float32),
        residual=np.array([0.25, -0.25, 0.125, 0.0], dtype=np.float32),
        metadata={"branch": "main"},
    )
    trace = run_se_post_mix_pipeline(adapter.state_from_bundle(bundle), adapter)
    result = adapter.build_result(trace)

    expected_stage_order = (
        "pre-finite",
        "D-normalize",
        "D-guard",
        "E",
        "F",
        "task-post-EF",
        "final-guard",
        "post-finite",
    )
    assert SE_POST_MIX_STAGE_ORDER == expected_stage_order
    assert trace.stage_order == expected_stage_order
    assert calls == [
        "pre-finite",
        "D-normalize",
        "D-guard",
        "E",
        "F",
        "final-guard",
        "post-finite",
    ]
    canonical_guard_keys = {
        "roles",
        "threshold",
        "peak",
        "scale",
        "applied",
    }
    assert result.layer_d_facts == {
        "stage_order": ("D-normalize", "D-guard"),
        "normalize": {
            "mode": "joint",
            "target_peak": 0.5,
            "noisy_scale": 0.5,
            "clean_scale": 0.5,
            "noise_scale": 0.5,
            "preserves_decomposition": True,
        },
        "guard": {
            "roles": [OBSERVATION, TARGET, ADDITIVE, RESIDUAL],
            "threshold": float(np.float32(0.99)),
            "peak": 0.25,
            "scale": 1.0,
            "applied": False,
        },
    }
    assert not hasattr(result, "layer_d_normalize_facts")
    assert set(result.layer_d_facts) == {"stage_order", "normalize", "guard"}
    assert set(result.layer_d_facts["guard"]) == canonical_guard_keys
    assert set(result.final_guard_facts) == canonical_guard_keys
    assert result.final_guard_facts["roles"] == [OBSERVATION]
    assert result.layer_f_applied_ops == ("post_eq",)
    assert result.layer_f_observation_changed is True
    assert result.layer_f_clipping_facts == {}
    assert result.layer_e.observation_changed is True
    assert result.bundle.metadata["branch"] == "prob_target_noise"
    np.testing.assert_array_equal(
        result.layer_d_bundle.require(RESIDUAL),
        np.array([0.125, -0.125, 0.0625, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.bundle.require(OBSERVATION),
        np.array([0.15, -0.15, 0.075, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.bundle.require(TARGET),
        np.array([0.125, -0.125, 0.0625, 0.0], dtype=np.float32),
    )
    assert set(result.bundle.tracks) == {OBSERVATION, TARGET, ADDITIVE, RESIDUAL}


def test_layer_f_skip_clip_requires_actual_layer_e_clipping(monkeypatch):
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = _validated_sim_cfg(
        noisy_post_mix_degradations_enabled=True,
    )
    adapter = TaskPostMixAdapter(
        dataset=dataset,
        rng=np.random.default_rng(29),
        sample_rate=16_000,
        actual_len=2,
        speech_uid="speech",
        branch="prob_target_noise",
        used_sidechain_wind=False,
    )
    state = adapter.state_from_bundle(
        adapter.make_initial_bundle(
            observation=np.array([0.5, -0.5], dtype=np.float32),
            target=np.array([0.25, -0.25], dtype=np.float32),
            additive=np.array([0.125, -0.125], dtype=np.float32),
            residual=np.array([0.125, -0.125], dtype=np.float32),
        )
    )
    adapter._layer_e = LayerEResult(
        state.waveforms[OBSERVATION],
        True,
        1,
        ("clipping",),
        (LayerEOperationResult("clipping", False, False, "no_change"),),
        False,
        None,
    )

    def layer_f(observation, _sr, layer_f_cfg, **kwargs):
        assert kwargs["skip_clip"] is False
        assert layer_f_cfg is adapter._simulation_view.shared.post_mix
        kwargs["applied_ops_out"].append("clip")
        kwargs["clipping_facts_out"].update({"applied": True})
        return observation.copy()

    monkeypatch.setattr(post_mix_adapter, "apply_post_augmentations", layer_f)
    result = adapter.apply_stage("F", state)

    assert result.applied is False
    assert adapter._layer_f_observation_changed is False
    assert adapter._layer_f_applied_ops == ()
    assert adapter._layer_f_clipping_facts == {}


def test_target_only_single_restores_equal_observation_before_layer_ef():
    layer_ef_inputs: list[np.ndarray] = []
    stems = _run_with_acoustic_stems(
        noisy_speech=np.array([0.5, -0.25, 0.0, 0.0]),
        noisy_after_rir=np.array([0.25, -0.125, 0.0, 0.0]),
        peak_normalization_scale=0.5,
        peak_normalization_mode="single",
        branch_path="prob_target_only",
        layer_ef_inputs=layer_ef_inputs,
        layer_ef_scale=0.5,
    )

    expected_clean = np.full(4, 0.125, dtype=np.float32)
    assert len(layer_ef_inputs) == 1
    np.testing.assert_array_equal(layer_ef_inputs[0], expected_clean)
    np.testing.assert_array_equal(stems["layer_d_noisy"], expected_clean)
    np.testing.assert_array_equal(stems["layer_d_clean"], expected_clean)
    np.testing.assert_array_equal(stems["final_clean"], expected_clean)
    np.testing.assert_array_equal(stems["final_noisy"], expected_clean * 0.5)
    np.testing.assert_array_equal(
        stems["layer_d_noise_residual_mixed"],
        np.array([0.125, -0.0625, 0.0, 0.0], dtype=np.float32),
    )


def test_se_simulation_by_layer_separates_branch_and_acoustic_metadata():
    stems = _run_with_acoustic_stems(
        noisy_speech=np.array([0.5, -0.25, 0.0, 0.0]),
        noisy_after_rir=np.array([0.25, -0.125, 0.0, 0.0]),
        peak_normalization_scale=1.0,
        peak_normalization_mode="off",
    )

    by_layer = stems["simulation_by_layer"]
    assert by_layer["B"] == {"branch": "prob_target_noise", "force_branch": None}
    assert by_layer["C"] == {
        "need_additive_noise": True,
        "c0_reverb_sampled": False,
        "source_volume_calibration": {"stage": "post_rir_pre_noise_echo"},
        "acoustic_audit": {"colored_noise": None},
        "c1_c2_snr_primary_drawn_db": None,
        "c1_c2_wind_noise_snr_candidate_drawn_db": None,
        "c1_c2_additive_snr_db_each_mix": [],
        "c1_c2_wind_sidechain_applied": False,
        "c1_c2_mix_two_segments_drawn": False,
        "c1_c2_want_sidechain_drawn": False,
    }
    assert by_layer["E"] == {
        "channel_augmentation_enabled": False,
        "num_augmentations_drawn": 0,
        "selected": [],
        "applied": [],
        "skipped": [],
        "operations": [],
        "observation_changed": False,
        "skip_reason": None,
        "augmentation_names_chosen": [],
        "skip_codec_redundant_with_e": False,
        "skip_clip_redundant_with_e": False,
    }
    assert by_layer["F"] == {
        "noisy_post_mix_degradations_enabled": False,
        "applied_ops": [],
        "observation_changed": False,
        "clipping_skipped_due_to_layer_e": False,
        "clipping_facts": {},
    }
    canonical_guard_keys = {"roles", "threshold", "peak", "scale", "applied"}
    assert "D-normalize" not in by_layer
    assert by_layer["D"] == {
        "stage_order": ("D-normalize", "D-guard"),
        "normalize": {
            "mode": "off",
            "target_peak": None,
            "noisy_scale": 1.0,
            "clean_scale": 1.0,
            "noise_scale": 1.0,
            "preserves_decomposition": True,
        },
        "guard": {
            "roles": [OBSERVATION, TARGET, ADDITIVE, RESIDUAL],
            "threshold": float(np.float32(0.99)),
            "peak": 0.5,
            "scale": 1.0,
            "applied": False,
        },
    }
    assert set(by_layer["D"]) == {"stage_order", "normalize", "guard"}
    assert set(by_layer["D"]["guard"]) == canonical_guard_keys
    assert set(by_layer["final_peak_normalize"]) == canonical_guard_keys
    assert by_layer["final_peak_normalize"]["roles"] == [OBSERVATION]


@pytest.mark.parametrize("export_stems", [False, True])
def test_mix_acoustic_preserves_draw_order_and_namedtuple_contract(monkeypatch, export_stems):
    draws = iter((0.1, 5.0, 25.0, 0.1, 0.1, 7.0))
    draw_log = []
    call_log = []

    def draw_random(_rng):
        value = next(draws)
        draw_log.append(("random", value))
        return value

    def draw_uniform(_rng, _low, _high):
        value = next(draws)
        draw_log.append(("uniform", value))
        return value

    monkeypatch.setattr(dynamic_base, "rng_random", draw_random)
    monkeypatch.setattr(dynamic_base, "rng_uniform", draw_uniform)
    # C-1/C-2/C-3 已提取为共享模块，两段仍必须共同满足既有 RNG 序列。
    monkeypatch.setattr(background_mix, "rng_random", draw_random)
    monkeypatch.setattr(background_mix, "rng_uniform", draw_uniform)
    monkeypatch.setattr(
        background_mix,
        "mix_noise",
        lambda noisy, sample, **kwargs: (
            noisy + sample,
            sample.astype(np.float32, copy=False),
        ),
    )

    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = _validated_sim_cfg(
        prob_reverberation=0.5,
        snr_low=0.0,
        snr_high=10.0,
        source_volume_normalization_enabled=False,
        reuse_noise=True,
        wind_noise_config={"wind_noise_snr_range": (20.0, 30.0)},
        noise_mix_prob=0.5,
        prob_wind_noise_sidechain=0.5,
    )
    dataset._noise_single_reuse_cache = None
    dataset.export_simulation_stems = export_stems
    dataset._mix_acoustic_rir = lambda speech, sr, rng, use_reverb: (
        call_log.append(("rir", use_reverb))
        or (speech + 1.0, speech + 2.0, speech + 3.0)
    )
    dataset._mix_acoustic_apply_wind = lambda noisy, reference, stem, *args: (
        call_log.append(("wind", args[3]))
        or (noisy + 4.0, stem + 4.0, True)
    )
    dataset._read_one_additive_noise_scp_only = lambda sr, duration, rng: (
        call_log.append(("single", sr, duration))
        or np.full((1, 4), 6.0, dtype=np.float32)
    )

    def apply_colored(noisy, reference, stem, snrs, sr, rng, audit):
        call_log.append(("colored", tuple(snrs), audit is not None))
        if audit is not None:
            audit["colored_noise"] = {"nominal_snr_db": 33.0}
        snrs.append(33.0)
        return noisy + 8.0, stem + 8.0

    dataset._mix_acoustic_colored_noise = apply_colored
    output = BaseDynamicMixingDataset._mix_acoustic(
        dataset,
        np.ones((1, 4), dtype=np.float32),
        16000,
        4,
        32000,
        np.random.default_rng(0),
        True,
    )

    assert draw_log == [
        ("random", 0.1),
        ("uniform", 5.0),
        ("uniform", 25.0),
        ("random", 0.1),
        ("random", 0.1),
        ("uniform", 7.0),
    ]
    expected_audit = (
        {
            "additive": [{"source": "ordinary_after_wind"}],
            "colored_noise": {"nominal_snr_db": 33.0},
        }
        if export_stems
        else None
    )
    assert call_log == [
        ("rir", True),
        ("wind", 25.0),
        ("single", 16000, 32000),
        ("colored", (7.0,), export_stems),
    ]
    assert isinstance(output, MixAcousticOutput)
    assert len(output) == 12
    unpacked = tuple(output)
    assert len(unpacked) == 12
    assert unpacked[-1] is True
    assert output.additive_snr_db_list == [7.0, 33.0]
    assert output.acoustic_audit == expected_audit
    assert output.snr_primary_drawn_db == 5.0
    assert output.wind_noise_snr_candidate_db == 25.0
    assert output.wn_snr == 25.0
    assert output.used_sidechain_wind is True
    assert output.mix_two is True
    assert output.want_sidechain is True
    assert output.use_reverb is True
    assert output.source_volume_calibration is None
    assert output.acoustic_scene_plan.noise.mode == "wind_plus_single"
    assert output.acoustic_scene_plan.source_gain.enabled is False
    assert output.acoustic_scene_plan.rir.enabled is False
    assert output.acoustic_scene_plan.to_dict()["num_samples"] == 4
    assert output.resolved_assets.noises == ()
    if export_stems:
        assert output.c0_audit_stems["source_volume_gain"] == 1.0
        np.testing.assert_array_equal(output.c0_audit_stems["full"], 2.0)
        np.testing.assert_array_equal(output.c0_audit_stems["early"], 3.0)
    else:
        assert output.c0_audit_stems is None


@pytest.mark.parametrize("volume_enabled", [False, True])
@pytest.mark.parametrize("invalid_stem", ["full", "early", "target"])
def test_c0_rejects_any_finite_stem_outside_float32_range(
    invalid_stem, volume_enabled
):
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = _validated_sim_cfg(
        prob_reverberation=1.0,
        snr_low=0.0,
        snr_high=0.0,
        source_volume_normalization_enabled=volume_enabled,
        source_volume_db_range=[-20.0, -20.0],
        source_volume_use_rms=False,
        source_volume_distribution="uniform",
        source_volume_beta_a=2.0,
        source_volume_beta_b=2.0,
    )
    dataset.export_simulation_stems = False
    normal = np.array([[1.0, -0.5, 0.25, 0.0]], dtype=np.float32)
    stems = {
        "full": normal * 2.0,
        "early": normal.copy(),
        "target": normal.copy(),
    }
    stems[invalid_stem] = np.array(
        [[1e308, -1e308, 0.0, 0.0]], dtype=np.float64
    )
    dataset._mix_acoustic_rir = lambda *args: (
        stems["full"].copy(),
        stems["early"].copy(),
        stems["target"].copy(),
    )

    with pytest.raises(
        ValueError, match="source volume stem exceeds the finite float32 range"
    ):
        dataset._mix_acoustic(
            normal,
            16000,
            normal.shape[1],
            normal.shape[1],
            np.random.default_rng(0),
            False,
            forced_source_volume_gain=2.0,
        )


def test_source_gain_is_applied_before_noise_and_excluded_from_residual(monkeypatch):
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = _validated_sim_cfg(
        prob_reverberation=1.0,
        snr_low=0.0,
        snr_high=0.0,
        source_volume_normalization_enabled=True,
        source_volume_db_range=[-20.0, -20.0],
        source_volume_use_rms=True,
        source_volume_distribution="uniform",
        source_volume_beta_a=2.0,
        source_volume_beta_b=2.0,
        reuse_noise=False,
        wind_noise_config={"wind_noise_snr_range": (0.0, 0.0)},
        noise_mix_prob=0.0,
        prob_wind_noise_sidechain=0.0,
        prob_colored_noise=0.0,
    )
    dataset.export_simulation_stems = False
    dataset._noise_single_reuse_cache = None
    dataset._mix_acoustic_rir = lambda speech, sr, rng, use_reverb: (
        speech * 4.0,
        speech * 2.0,
        speech * 2.0,
    )
    dataset._mix_acoustic_read_additive = lambda *args: np.full(
        (1, 4), 0.25, dtype=np.float32
    )
    observed = {}
    noise_component = np.full((1, 4), 0.5, dtype=np.float32)

    def mix_noise_after_gain(noisy, sample, **kwargs):
        observed["noisy"] = noisy.copy()
        observed["reference"] = kwargs["snr_reference"].copy()
        return noisy + noise_component, noise_component.copy()

    monkeypatch.setattr(background_mix, "mix_noise", mix_noise_after_gain)
    speech = np.array([[1.0, -0.5, 0.25, 0.0]], dtype=np.float32)
    output = dataset._mix_acoustic(
        speech,
        16000,
        speech.shape[1],
        speech.shape[1],
        np.random.default_rng(0),
        True,
        forced_source_volume_gain=3.0,
    )

    np.testing.assert_array_equal(observed["noisy"], speech * 12.0)
    np.testing.assert_array_equal(observed["reference"], speech * 6.0)
    np.testing.assert_array_equal(output.noisy_after_rir, speech * 12.0)
    np.testing.assert_array_equal(output.additive_noise_2d, noise_component)
    np.testing.assert_array_equal(
        output.noisy_speech - output.noisy_after_rir, noise_component
    )
    assert output.source_volume_calibration["gain"] == pytest.approx(3.0)


def test_forced_mix_consumes_source_volume_draw_and_aligns_following_noise_rng(
    monkeypatch,
):
    speech = np.array([[1.0, -0.5, 0.25, 0.0]], dtype=np.float32)

    def _run(forced_gain=None):
        dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
        dataset.sim_cfg = _validated_sim_cfg(
            prob_reverberation=0.5,
            snr_low=0.0,
            snr_high=10.0,
            source_volume_normalization_enabled=True,
            source_volume_db_range=[-35.0, -15.0],
            source_volume_use_rms=True,
            source_volume_distribution="uniform",
            source_volume_beta_a=2.0,
            source_volume_beta_b=2.0,
            reuse_noise=False,
            wind_noise_config={"wind_noise_snr_range": (10.0, 30.0)},
            noise_mix_prob=0.0,
            prob_wind_noise_sidechain=0.0,
        )
        dataset.export_simulation_stems = False
        dataset._noise_single_reuse_cache = None
        dataset._mix_acoustic_rir = lambda source, sr, rng, use_reverb: (
            source.copy(),
            source.copy(),
            source.copy(),
        )
        rng = np.random.default_rng(20260826)
        c0_states = []
        noise_states = []
        noise_draws = []

        dataset._record_post_rir_calibration = lambda *args: c0_states.append(
            rng.bit_generator.state.copy()
        )

        def _read_noise(mode, sr, max_dur_sr, noise_rng, reuse_noise):
            noise_states.append(noise_rng.bit_generator.state.copy())
            noise_draws.append(float(noise_rng.random()))
            return np.zeros_like(speech)

        dataset._mix_acoustic_read_additive = _read_noise
        dataset._mix_acoustic_colored_noise = lambda noisy, ref, stem, *args: (
            noisy,
            stem,
        )
        output = dataset._mix_acoustic(
            speech,
            16000,
            speech.shape[1],
            speech.shape[1],
            rng,
            True,
            forced_source_volume_gain=forced_gain,
        )
        return output, c0_states[0], noise_states[0], noise_draws[0], rng.bit_generator.state

    monkeypatch.setattr(
        dynamic_base,
        "mix_noise",
        lambda noisy, sample, **kwargs: (noisy.copy(), np.zeros_like(noisy)),
    )
    normal, normal_c0, normal_noise_state, normal_noise_draw, normal_final = _run()
    forced, forced_c0, forced_noise_state, forced_noise_draw, forced_final = _run(
        normal.source_volume_calibration["gain"]
    )

    assert normal.source_volume_calibration["forced_gain"] is False
    assert forced.source_volume_calibration["forced_gain"] is True
    assert forced_c0 == normal_c0
    assert forced_noise_state == normal_noise_state
    assert forced_noise_draw == normal_noise_draw
    assert forced_final == normal_final


def test_d_guard_scales_all_roles_after_d_normalize_and_draws_no_rng():
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = _validated_sim_cfg(
        peak_normalization_mode="off",
        peak_normalization_scale_range=[0.5, 0.95],
    )
    rng = np.random.default_rng(11)
    adapter = TaskPostMixAdapter(
        dataset=dataset,
        rng=rng,
        sample_rate=16_000,
        actual_len=2,
        speech_uid="speech",
        branch="prob_target_noise",
        used_sidechain_wind=False,
    )
    initial_state = adapter.state_from_bundle(
        adapter.make_initial_bundle(
            observation=np.array([2.0, -2.0], dtype=np.float32),
            target=np.array([3.0, -3.0], dtype=np.float32),
            additive=np.array([0.5, -0.5], dtype=np.float32),
            residual=np.array([0.25, -0.25], dtype=np.float32),
        )
    )
    state_after_d_normalize = adapter.apply_stage(
        "D-normalize", initial_state
    ).state
    cursor = rng.bit_generator.state.copy()

    state_after_d_guard = adapter.apply_stage(
        "D-guard", state_after_d_normalize
    ).state
    final = adapter.apply_stage("final-guard", state_after_d_guard).state

    scale = 0.99 / 3.0
    for role in (OBSERVATION, TARGET, ADDITIVE, RESIDUAL):
        np.testing.assert_allclose(
            state_after_d_guard.waveforms[role],
            state_after_d_normalize.waveforms[role] * scale,
            atol=1e-6,
        )
    assert adapter._layer_d_facts == {
        "stage_order": ("D-normalize", "D-guard"),
        "normalize": {
            "mode": "off",
            "target_peak": None,
            "noisy_scale": 1.0,
            "clean_scale": 1.0,
            "noise_scale": 1.0,
            "preserves_decomposition": True,
        },
        "guard": {
            "roles": [OBSERVATION, TARGET, ADDITIVE, RESIDUAL],
            "threshold": float(np.float32(0.99)),
            "peak": 3.0,
            "scale": pytest.approx(scale),
            "applied": True,
        },
    }
    np.testing.assert_array_equal(
        final.waveforms[OBSERVATION], state_after_d_guard.waveforms[OBSERVATION]
    )
    np.testing.assert_array_equal(
        final.waveforms[TARGET], state_after_d_guard.waveforms[TARGET]
    )
    assert rng.bit_generator.state == cursor


def test_final_peak_guard_scales_only_observation():
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = _validated_sim_cfg()
    adapter = TaskPostMixAdapter(
        dataset=dataset,
        rng=np.random.default_rng(17),
        sample_rate=16_000,
        actual_len=2,
        speech_uid="speech",
        branch="prob_target_noise",
        used_sidechain_wind=False,
    )
    state = adapter.state_from_bundle(
        adapter.make_initial_bundle(
            observation=np.array([2.0, -2.0], dtype=np.float32),
            target=np.array([3.0, -3.0], dtype=np.float32),
            additive=np.array([4.0, -4.0], dtype=np.float32),
            residual=np.array([5.0, -5.0], dtype=np.float32),
        )
    )

    final = adapter.apply_stage("final-guard", state).state

    np.testing.assert_allclose(final.waveforms[OBSERVATION], [0.99, -0.99])
    for role in (TARGET, ADDITIVE, RESIDUAL):
        np.testing.assert_array_equal(final.waveforms[role], state.waveforms[role])
    threshold = float(np.float32(0.99))
    assert adapter._final_guard_facts == {
        "roles": [OBSERVATION],
        "threshold": threshold,
        "peak": 2.0,
        "scale": pytest.approx(threshold / 2.0),
        "applied": True,
    }


def test_d_normalize_off_is_noop_and_preserves_se_rng():
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = _validated_sim_cfg(
        peak_normalization_mode="off",
        peak_normalization_scale_range=[0.5, 0.95],
    )
    rng = np.random.default_rng(19)
    adapter = TaskPostMixAdapter(
        dataset=dataset,
        rng=rng,
        sample_rate=16_000,
        actual_len=2,
        speech_uid="speech",
        branch="prob_target_noise",
        used_sidechain_wind=False,
    )
    state = adapter.state_from_bundle(
        adapter.make_initial_bundle(
            observation=np.array([0.5, -0.5], dtype=np.float32),
            target=np.array([0.75, -0.75], dtype=np.float32),
            additive=np.array([0.25, -0.25], dtype=np.float32),
            residual=np.array([0.125, -0.125], dtype=np.float32),
        )
    )
    cursor = rng.bit_generator.state.copy()

    result = adapter.apply_stage("D-normalize", state)

    assert result.applied is False
    assert result.state is state
    assert adapter._peak_normalization.mode == "off"
    assert rng.bit_generator.state == cursor


def test_d_guard_noop_preserves_se_rng_and_reports_not_applied():
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = _validated_sim_cfg(
        peak_normalization_mode="off",
        peak_normalization_scale_range=[0.5, 0.95],
    )
    rng = np.random.default_rng(23)
    adapter = TaskPostMixAdapter(
        dataset=dataset,
        rng=rng,
        sample_rate=16_000,
        actual_len=2,
        speech_uid="speech",
        branch="prob_target_noise",
        used_sidechain_wind=False,
    )
    initial_state = adapter.state_from_bundle(
        adapter.make_initial_bundle(
            observation=np.array([0.5, -0.5], dtype=np.float32),
            target=np.array([0.75, -0.75], dtype=np.float32),
            additive=np.array([0.25, -0.25], dtype=np.float32),
            residual=np.array([0.125, -0.125], dtype=np.float32),
        )
    )
    state_after_d_normalize = adapter.apply_stage(
        "D-normalize", initial_state
    ).state
    cursor = rng.bit_generator.state.copy()

    guarded = adapter.apply_stage("D-guard", state_after_d_normalize).state

    assert guarded is state_after_d_normalize
    assert adapter._layer_d_facts["guard"]["applied"] is False
    assert adapter._layer_d_facts["guard"]["scale"] == 1.0
    assert rng.bit_generator.state == cursor


def test_residual_uses_noisy_peak_normalization_and_layer_d_guard_scale():
    stems = _run_with_acoustic_stems(
        noisy_speech=np.array([4.0, -2.0, 1.0, 0.0]),
        noisy_after_rir=np.array([-2.0, 1.0, 0.0, 0.0]),
        peak_normalization_scale=0.5,
    )

    residual_after_peak_normalization = np.array([3.0, -1.5, 0.5, 0.0], dtype=np.float32)
    layer_d_scale = 0.99 / 3.0
    np.testing.assert_allclose(
        stems["layer_d_noise_residual_mixed"],
        residual_after_peak_normalization * layer_d_scale,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        stems["layer_d_noisy"],
        np.array([2.0, -1.0, 0.5, 0.0], dtype=np.float32) * layer_d_scale,
        rtol=1e-6,
        atol=1e-7,
    )


def test_residual_uses_reported_scale_without_peak_inference():
    stems = _run_with_acoustic_stems(
        noisy_speech=np.array([0.5, -0.25, 0.0, 0.0]),
        noisy_after_rir=np.zeros(4, dtype=np.float32),
        peak_normalization_scale=0.5,
        reported_noisy_scale=0.25,
    )

    np.testing.assert_array_equal(
        stems["layer_d_noise_residual_mixed"],
        np.array([0.125, -0.0625, 0.0, 0.0], dtype=np.float32),
    )


def test_zero_noisy_joint_recovers_common_scale_from_nonzero_clean_stem():
    stems = _run_with_acoustic_stems(
        noisy_speech=np.zeros(4, dtype=np.float32),
        noisy_after_rir=np.array([-0.5, 0.25, 0.0, 0.0], dtype=np.float32),
        peak_normalization_scale=0.5,
    )

    residual = stems["layer_d_noise_residual_mixed"]
    np.testing.assert_array_equal(
        residual,
        np.array([0.25, -0.125, 0.0, 0.0], dtype=np.float32),
    )
    assert np.isfinite(residual).all()


@pytest.mark.parametrize(
    ("mode", "expected_scale"),
    [("joint", 0.5), ("separate", 0.5), ("single", 1.0), ("off", 1.0)],
)
def test_residual_scale_follows_peak_normalization_mode(mode: str, expected_scale: float):
    stems = _run_with_acoustic_stems(
        noisy_speech=np.array([0.5, -0.25, 0.0, 0.0]),
        noisy_after_rir=np.zeros(4, dtype=np.float32),
        peak_normalization_scale=0.5,
        peak_normalization_mode=mode,
    )

    expected = np.array([0.5, -0.25, 0.0, 0.0], dtype=np.float32) * expected_scale
    np.testing.assert_allclose(stems["layer_d_noise_residual_mixed"], expected)


def test_zero_noisy_separate_uses_safe_identity_fallback():
    stems = _run_with_acoustic_stems(
        noisy_speech=np.zeros(4, dtype=np.float32),
        noisy_after_rir=np.array([-0.5, 0.25, 0.0, 0.0], dtype=np.float32),
        peak_normalization_scale=0.5,
        peak_normalization_mode="separate",
    )

    np.testing.assert_array_equal(
        stems["layer_d_noise_residual_mixed"],
        np.array([0.5, -0.25, 0.0, 0.0], dtype=np.float32),
    )
