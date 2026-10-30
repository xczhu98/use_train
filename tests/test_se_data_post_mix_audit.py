"""SE post-mix 发布与审计门禁回归测试。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import tasks.se.data.post_mix_adapter as adapter_module
from core.data.dynamic_dataset_base import BaseDynamicMixingDataset, LayerEResult
from core.data.post_mix_pipeline import PostMixStageResult, PostMixState
from core.data.simulation_configs import SimulationConfigs
from core.data.waveform_bundle import WaveformBundle
from tasks.se.data.post_mix_adapter import (
    ADDITIVE,
    OBSERVATION,
    RESIDUAL,
    SE_POST_MIX_STAGE_ORDER,
    TARGET,
    TaskPostMixAdapter,
    run_se_post_mix_pipeline,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
def test_build_result_rejects_pipeline_without_post_finite():
    adapter = TaskPostMixAdapter.__new__(TaskPostMixAdapter)
    adapter._post_finite_completed = False

    with pytest.raises(RuntimeError, match="did not complete post-finite"):
        adapter.build_result(None)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("optional_op", ["drc", "codec"])
def test_layer_f_reports_optional_cpu_op_with_other_effect(monkeypatch, optional_op):
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimulationConfigs.from_dict(
        {"noisy_post_mix_degradations_enabled": True},
        strict=True,
    )
    adapter = TaskPostMixAdapter(
        dataset=dataset,
        rng=np.random.default_rng(17),
        sample_rate=16_000,
        actual_len=2,
        speech_uid="sample",
        branch="prob_target_noise",
        used_sidechain_wind=False,
    )
    state = adapter.state_from_bundle(
        WaveformBundle(
            {
                OBSERVATION: np.array([0.25, -0.25], dtype=np.float32),
                TARGET: np.array([0.2, -0.2], dtype=np.float32),
                ADDITIVE: np.array([0.05, -0.05], dtype=np.float32),
                RESIDUAL: np.array([0.1, -0.1], dtype=np.float32),
            }
        )
    )
    adapter._layer_e = LayerEResult(
        waveform=state.waveforms[OBSERVATION],
        enabled=False,
        num_augmentations_drawn=0,
        selected=(),
        operations=(),
        observation_changed=False,
    )

    def _layer_f(observation, *_args, **kwargs):
        kwargs["applied_ops_out"].extend((optional_op, "post_eq"))
        return np.asarray(observation, dtype=np.float32) * np.float32(0.5)

    monkeypatch.setattr(adapter_module, "apply_post_augmentations", _layer_f)

    result = adapter.apply_stage("F", state)

    assert result.applied is True
    assert adapter._layer_f_observation_changed is True
    assert adapter._layer_f_applied_ops == (optional_op, "post_eq")


@pytest.mark.unit
def test_noop_stage_cannot_change_waveform_role_keys():
    initial = PostMixState(
        {
            OBSERVATION: np.array([0.25, -0.25], dtype=np.float32),
            TARGET: np.array([0.2, -0.2], dtype=np.float32),
        }
    )

    class _BadNoopAdapter:
        def apply_stage(self, stage, state):
            if stage == SE_POST_MIX_STAGE_ORDER[0]:
                changed_state = PostMixState(
                    {OBSERVATION: state.waveforms[OBSERVATION]}
                )
                return PostMixStageResult(stage, changed_state, False)
            return PostMixStageResult(stage, state, False)

    with pytest.raises(ValueError, match="no-op stage 'pre-finite' changed waveforms"):
        run_se_post_mix_pipeline(initial, _BadNoopAdapter())
