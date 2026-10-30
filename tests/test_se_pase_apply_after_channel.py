"""SE PASE ``apply_after_channel`` 的 post-mix 执行顺序测试。

官方 UniPASE/GAP 在增强之后做 noisy/clean 逐信号 peak 归一化；
``pase_preprocess.apply_after_channel=True`` 必须把 PASE 归一化从 Layer D
推迟到 ``task-post-EF``，且默认 False 保持原有 Layer D 行为。
"""

from __future__ import annotations

import numpy as np
import pytest

from core.data.dynamic_dataset_base import (
    BaseDynamicMixingDataset,
    LayerEOperationResult,
    LayerEResult,
)
from core.data.simulation_configs import SimulationConfigs
from tasks.se.data.post_mix_adapter import (
    ADDITIVE,
    OBSERVATION,
    RESIDUAL,
    TARGET,
    TaskPostMixAdapter,
    run_se_post_mix_pipeline,
)

pytestmark = [pytest.mark.unit]

OBSERVATION_IN = np.array([0.8, -0.4, 0.2, 0.0], dtype=np.float32)
TARGET_IN = np.array([0.4, -0.2, 0.1, 0.0], dtype=np.float32)
ADDITIVE_IN = np.array([0.2, -0.1, 0.05, 0.0], dtype=np.float32)
RESIDUAL_IN = np.array([0.3, -0.15, 0.075, 0.0], dtype=np.float32)


def _peak(waveform: np.ndarray) -> float:
    return float(np.max(np.abs(waveform))) if waveform.size else 0.0


def _dataset(*, pase_preprocess: dict) -> BaseDynamicMixingDataset:
    """只带 sim_cfg / PASE 配置与 Layer E stub 的 dummy Dataset。"""
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimulationConfigs.from_dict(
        {
            "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": False,
        },
        strict=True,
    )
    dataset.pase_preprocess = pase_preprocess
    dataset._guard_finite = lambda observation, target, _uid, _label: (
        observation,
        target,
    )

    def layer_e(observation, _sr, _length, _rng, _sidechain):
        # Layer E 只改 observation；缩放 0.5 用于区分 D 阶段与 post-E 的 PASE。
        return LayerEResult(
            waveform=np.asarray(observation, dtype=np.float32) * np.float32(0.5),
            enabled=True,
            num_augmentations_drawn=1,
            selected=("codec",),
            operations=(LayerEOperationResult("codec", True, True),),
            observation_changed=True,
            skip_reason=None,
        )

    dataset._apply_channel_degradation_result = layer_e
    return dataset


def _run(
    *,
    pase_preprocess: dict,
    branch: str = "prob_target_noise",
    observation: np.ndarray = OBSERVATION_IN,
    target: np.ndarray = TARGET_IN,
):
    dataset = _dataset(pase_preprocess=pase_preprocess)
    adapter = TaskPostMixAdapter(
        dataset=dataset,
        rng=np.random.default_rng(0),
        sample_rate=16000,
        actual_len=observation.size,
        speech_uid="speech",
        branch=branch,
        used_sidechain_wind=False,
        metadata={"branch": branch},
    )
    bundle = adapter.prepare_initial_bundle(
        observation=observation,
        target=target,
        additive=ADDITIVE_IN,
        residual=RESIDUAL_IN,
    )
    trace = run_se_post_mix_pipeline(adapter.state_from_bundle(bundle), adapter)
    return adapter, trace, adapter.build_result(trace)


def _separate_pase(*, apply_after_channel: bool) -> dict:
    return {
        "peak_normalization_mode": "separate",
        "peak_normalization_scale_range": [0.5, 0.5],
        "apply_after_channel": apply_after_channel,
    }


def test_apply_after_channel_defers_pase_to_post_ef():
    """D 阶段保持公共 off，PASE 在 Layer E/F 之后按 separate 逐信号归一化。"""
    adapter, trace, result = _run(
        pase_preprocess=_separate_pase(apply_after_channel=True)
    )

    applied = {entry.stage: entry.applied for entry in trace.results}
    assert applied["D-normalize"] is False
    assert applied["task-post-EF"] is True

    # D 阶段波形不含 PASE 缩放：observation 仍是 Layer E 之前的原始峰值。
    np.testing.assert_array_equal(
        result.layer_d_bundle.require(OBSERVATION), OBSERVATION_IN
    )
    np.testing.assert_array_equal(result.layer_d_bundle.require(TARGET), TARGET_IN)

    # post-E 后 observation 经 Layer E(×0.5) 再归一化到 0.5；target 独立归一化到 0.5。
    assert _peak(result.bundle.require(OBSERVATION)) == pytest.approx(0.5)
    assert _peak(result.bundle.require(TARGET)) == pytest.approx(0.5)
    np.testing.assert_allclose(
        result.bundle.require(RESIDUAL),
        RESIDUAL_IN * np.float32(1.25),
        rtol=1e-6,
        atol=0.0,
    )

    normalize_facts = result.layer_d_facts["normalize"]
    assert normalize_facts["mode"] == "off"
    assert normalize_facts["pase_deferred"] == {
        "configured_mode": "separate",
        "apply_after_channel": True,
        "applied_at": "task-post-EF",
    }
    assert set(result.layer_d_facts) == {
        "stage_order",
        "normalize",
        "guard",
        "pase_post_e",
    }
    pase_facts = result.layer_d_facts["pase_post_e"]
    assert pase_facts["mode"] == "separate"
    assert pase_facts["target_peak"] == pytest.approx(0.5)
    assert pase_facts["noisy_scale"] == pytest.approx(1.25)
    assert pase_facts["clean_scale"] == pytest.approx(1.25)
    assert pase_facts["noise_scale"] == 1.0
    assert pase_facts["preserves_decomposition"] is False
    assert pase_facts["apply_after_channel"] is True
    assert adapter._peak_normalization.mode == "separate"


def test_default_apply_after_channel_keeps_pase_at_layer_d():
    """apply_after_channel=False（默认）保持既有 Layer D PASE 行为。"""
    _, trace, result = _run(
        pase_preprocess=_separate_pase(apply_after_channel=False),
    )

    applied = {entry.stage: entry.applied for entry in trace.results}
    assert applied["D-normalize"] is True
    assert applied["task-post-EF"] is False

    # Layer D 已把 observation/target 各自归一化到 0.5，Layer E 再缩放 0.5。
    assert _peak(result.layer_d_bundle.require(OBSERVATION)) == pytest.approx(0.5)
    assert _peak(result.layer_d_bundle.require(TARGET)) == pytest.approx(0.5)
    assert _peak(result.bundle.require(OBSERVATION)) == pytest.approx(0.25)
    assert _peak(result.bundle.require(TARGET)) == pytest.approx(0.5)

    assert result.layer_d_facts["normalize"]["mode"] == "separate"
    assert "pase_deferred" not in result.layer_d_facts["normalize"]
    assert "pase_post_e" not in result.layer_d_facts


def test_apply_after_channel_single_target_only_keeps_observation_target_aligned():
    """single + prob_target_only 在 post-E 仍以 clean_scale 对齐 observation/target。"""
    _, _, result = _run(
        pase_preprocess={
            "peak_normalization_mode": "single",
            "peak_normalization_scale_range": [0.5, 0.5],
            "apply_after_channel": True,
        },
        branch="prob_target_only",
        observation=TARGET_IN.copy(),
        target=TARGET_IN.copy(),
    )

    np.testing.assert_allclose(
        result.bundle.require(OBSERVATION),
        result.bundle.require(TARGET),
        rtol=0.0,
        atol=0.0,
    )
    assert _peak(result.bundle.require(OBSERVATION)) == pytest.approx(0.5)
    np.testing.assert_allclose(
        result.bundle.require(RESIDUAL),
        RESIDUAL_IN * np.float32(1.25),
        rtol=1e-6,
        atol=0.0,
    )
    pase_facts = result.layer_d_facts["pase_post_e"]
    assert pase_facts["mode"] == "single"
    assert pase_facts["noisy_scale"] == 1.0
    assert pase_facts["clean_scale"] == pytest.approx(1.25)


def test_apply_after_channel_without_pase_mode_keeps_pipeline_noop():
    """apply_after_channel 但未配置 PASE mode 时，post-E 不做归一化也不写事实。"""
    _, trace, result = _run(
        pase_preprocess={"apply_after_channel": True},
    )

    applied = {entry.stage: entry.applied for entry in trace.results}
    assert applied["D-normalize"] is False
    assert applied["task-post-EF"] is False
    assert result.layer_d_facts["normalize"]["mode"] == "off"
    assert "pase_deferred" not in result.layer_d_facts["normalize"]
    assert "pase_post_e" not in result.layer_d_facts
    # 只有 Layer E 的 0.5 缩放。
    assert _peak(result.bundle.require(OBSERVATION)) == pytest.approx(0.4)
    assert _peak(result.bundle.require(TARGET)) == pytest.approx(0.4)
