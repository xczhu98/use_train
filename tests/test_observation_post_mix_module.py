from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import core.data.simulation as simulation

from core.data.simulation.observation_post_mix import (
    NonDecomposableObservationContract,
    ObservationPostMixRenderer,
    ObservationPostMixRequest,
    PostMixRoleContract,
    infer_post_mix_role_contract,
)
from core.data.waveform_bundle import WaveformBundle
from core.data.dynamic_dataset_base import LayerEResult

pytestmark = pytest.mark.unit


def _policy(*, enabled=False):
    cfg = SimpleNamespace(
        peak_normalization_mode="off",
        peak_normalization_scale_range=(0.9, 1.0),
        prob_channel_enabled=False,
        noisy_post_mix_degradations_enabled=enabled,
    )
    spatial = SimpleNamespace(
        spatial_enabled=False,
        spatial_layer_d_policy="shared",
        spatial_layer_e_policy="shared",
        spatial_layer_f_policy="shared",
        reference_channel=0,
        channels=1,
    )
    shared = SimpleNamespace(
        spatial=spatial,
        peak=SimpleNamespace(
            peak_normalization_mode="off",
            peak_normalization_scale_range=(0.9, 1.0),
        ),
        post_mix=SimpleNamespace(noisy_post_mix_degradations_enabled=enabled),
    )
    policy = SimpleNamespace(
        sim_cfg=cfg,
        _simulation_view=SimpleNamespace(shared=shared),
        used_sidechain_wind=False,
    )
    policy._apply_channel_degradation_result = lambda waveform, sample_rate, length, rng, sidechain: LayerEResult(
        waveform=np.asarray(waveform, dtype=np.float32),
        enabled=False,
        num_augmentations_drawn=0,
        selected=(),
        operations=(),
        observation_changed=False,
        skip_reason="test",
    )
    return policy


def _request(policy):
    return ObservationPostMixRequest(
        scene_bundle=WaveformBundle(
            {
                "mixture": np.full(8, 0.25, dtype=np.float32),
                "target/0": np.linspace(0.0, 0.7, 8, dtype=np.float32),
                "direct/0": np.linspace(0.7, 0.0, 8, dtype=np.float32),
            }
        ),
        sample_rate=16000,
        rng=np.random.default_rng(4),
        task_policy=policy,
        observation_roles=("mixture",),
        protected_roles=("target/0", "direct/0"),
    )


def test_observation_post_mix_uses_fixed_order_and_preserves_targets():
    result = ObservationPostMixRenderer().apply(_request(_policy()))

    assert result.trace.stage_order == (
        "pre-finite", "D-normalize", "D-guard", "E", "F", "task-post-EF", "final-guard", "post-finite"
    )
    assert result.facts["target_invariant_after_d"] is True
    assert result.facts["target_invariant_after_observation_stages"] is True
    assert result.facts["target_direct_invariant"] == result.facts[
        "target_invariant_after_observation_stages"
    ]
    assert result.facts["decomposable_final"] == result.facts[
        "decomposition_preserved"
    ]
    assert np.array_equal(result.initial_bundle.require("target/0"), result.bundle.require("target/0"))
    assert np.array_equal(result.initial_bundle.require("direct/0"), result.bundle.require("direct/0"))
    assert result.facts["observation_changed"] is False


def test_observation_post_mix_facts_use_d_state_with_compact_trace():
    policy = _policy()
    policy.sim_cfg.peak_normalization_mode = "joint"
    policy.sim_cfg.peak_normalization_scale_range = (0.5, 0.5)
    request = replace(_request(policy), retain_stage_states=False)

    result = ObservationPostMixRenderer().apply(request)

    assert result.trace.retains_stage_states is False
    assert result.facts["target_invariant_after_d"] is False
    assert result.facts["target_invariant_after_observation_stages"] is True
    assert result.facts["target_direct_invariant"] is False


def test_observation_post_mix_runs_observation_degradation_only():
    result = ObservationPostMixRenderer().apply(_request(_policy(enabled=False)))
    assert np.array_equal(result.initial_bundle.require("target/0"), result.bundle.require("target/0"))
    assert np.array_equal(result.initial_bundle.require("direct/0"), result.bundle.require("direct/0"))
    assert result.facts["stage_order"][-1] == "post-finite"


def test_spatial_additive_background_keeps_decomposition_through_d_e_f():
    policy = _policy()
    policy._simulation_view.shared.spatial.spatial_enabled = True
    policy._simulation_view.shared.spatial.channels = 2
    source = np.vstack(
        [
            np.linspace(-0.2, 0.2, 8),
            np.linspace(0.3, -0.3, 8),
        ]
    ).astype(np.float32)
    background = np.full((2, 8), 0.05, dtype=np.float32)
    target = np.linspace(0.0, 0.7, 8, dtype=np.float32)
    direct = np.linspace(0.7, 0.0, 8, dtype=np.float32)
    request = ObservationPostMixRequest(
        scene_bundle=WaveformBundle(
            {
                "mixture": source + background,
                "background": background,
                "source_full_rir/0": source,
                "target/0": target,
                "direct/0": direct,
            }
        ),
        sample_rate=16000,
        rng=np.random.default_rng(4),
        task_policy=policy,
        observation_roles=("mixture", "background"),
        protected_roles=("target/0", "direct/0"),
        amplitude_roles=("source_full_rir/0",),
    )

    result = ObservationPostMixRenderer().apply(request)

    assert result.facts["decomposable_final"] is True
    np.testing.assert_allclose(
        result.bundle.require("mixture"),
        result.bundle.require("source_full_rir/0") + result.bundle.require("background"),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(result.initial_bundle.require("target/0"), result.bundle.require("target/0"))
    np.testing.assert_array_equal(result.initial_bundle.require("direct/0"), result.bundle.require("direct/0"))


def test_non_decomposable_observation_contract_marks_sidechain_as_non_additive():
    policy = _policy()
    target = np.linspace(0.0, 0.7, 8, dtype=np.float32)
    control = np.full(8, 0.1, dtype=np.float32)
    contract = NonDecomposableObservationContract(
        observation_roles=("mixture",),
        control_roles=("wind_sidechain",),
        protected_roles=("target/0",),
        transform_id="wind_sidechain_compressor_v1",
        mutable_stage="C-2",
        role_shapes={"mixture": (8,), "wind_sidechain": (8,), "target/0": (8,)},
        timebase_identity="scene-timebase-1",
        rng_stream="background.wind_sidechain",
        provenance={"source_identity": "wind_scp:wind-1"},
    )
    request = ObservationPostMixRequest(
        scene_bundle=WaveformBundle(
            {
                "mixture": np.full(8, 0.25, dtype=np.float32),
                "wind_sidechain": control,
                "target/0": target,
            }
        ),
        sample_rate=16000,
        rng=np.random.default_rng(4),
        task_policy=policy,
        observation_roles=("mixture",),
        protected_roles=("target/0",),
        non_decomposable_contract=contract,
    )

    result = ObservationPostMixRenderer().apply(request)

    assert result.facts["decomposable_final"] is False
    assert result.facts["non_decomposable_observation"]["decomposable"] is False
    assert result.facts["non_decomposable_observation"]["control_roles"] == ["wind_sidechain"]
    np.testing.assert_array_equal(result.initial_bundle.require("target/0"), result.bundle.require("target/0"))


def test_non_decomposable_observation_contract_rejects_overlapping_or_missing_controls():
    with pytest.raises(ValueError, match="overlapping roles"):
        NonDecomposableObservationContract(
            observation_roles=("mixture",),
            control_roles=("mixture",),
            transform_id="wind_sidechain_compressor_v1",
            mutable_stage="C-2",
        )

    with pytest.raises(ValueError, match="mutable_stage"):
        NonDecomposableObservationContract(
            observation_roles=("mixture",),
            transform_id="wind_sidechain_compressor_v1",
            mutable_stage="C-9",
        )

    contract = NonDecomposableObservationContract(
        observation_roles=("mixture",),
        control_roles=("wind_sidechain",),
        protected_roles=("target/0",),
        transform_id="wind_sidechain_compressor_v1",
        mutable_stage="C-2",
    )
    with pytest.raises(KeyError, match="control_roles"):
        ObservationPostMixRequest(
            scene_bundle=WaveformBundle(
                {
                    "mixture": np.zeros(4, dtype=np.float32),
                    "target/0": np.zeros(4, dtype=np.float32),
                }
            ),
            sample_rate=16000,
            rng=np.random.default_rng(0),
            task_policy=_policy(),
            observation_roles=("mixture",),
            protected_roles=("target/0",),
            non_decomposable_contract=contract,
        )


def test_observation_post_mix_rejects_overlapping_roles():
    with pytest.raises(ValueError, match="overlap"):
        ObservationPostMixRequest(
            scene_bundle=WaveformBundle({"mixture": np.zeros(4, dtype=np.float32)}),
            sample_rate=16000,
            rng=np.random.default_rng(0),
            task_policy=_policy(),
            observation_roles=("mixture",),
            protected_roles=("mixture",),
        )


def test_post_mix_role_contract_normalizes_roles_and_is_recorded_in_facts():
    contract = PostMixRoleContract(
        observation_roles=("mixture",),
        protected_roles=("target/0",),
        derived_roles=("direct/0",),
        role_shapes={"mixture": (8,), "target/0": (8,), "direct/0": (8,)},
        mutable_stages={"mixture": ("E", "F")},
    )
    request = replace(
        _request(_policy()),
        protected_roles=("target/0",),
        role_contract=contract,
    )
    result = ObservationPostMixRenderer().apply(request)

    assert result.facts["derived_roles"] == ["direct/0"]
    assert result.facts["role_contract"]["mutable_stages"] == {
        "mixture": ["E", "F"]
    }


def test_post_mix_role_contract_rejects_conflicting_legacy_fields():
    contract = PostMixRoleContract(
        observation_roles=("mixture",),
        protected_roles=("target/0",),
    )
    with pytest.raises(ValueError, match="disagrees"):
        replace(
            _request(_policy()),
            observation_roles=("target/0",),
            role_contract=contract,
        )


def test_post_mix_role_contract_rejects_shape_mismatch():
    contract = PostMixRoleContract(
        observation_roles=("mixture",),
        protected_roles=("target/0",),
        role_shapes={"mixture": (4,), "target/0": (8,)},
    )
    with pytest.raises(ValueError, match=r"role_shapes\['mixture'\]"):
        ObservationPostMixRenderer().apply(
            replace(
                _request(_policy()),
                role_contract=contract,
                protected_roles=("target/0",),
            )
        )


def test_infer_post_mix_role_contract_discovers_task_scene_roles():
    bundle = WaveformBundle(
        {
            "mixture": np.zeros(4, dtype=np.float32),
            "background": np.zeros(4, dtype=np.float32),
            "source_full_rir/0": np.zeros(4, dtype=np.float32),
            "target/0": np.zeros(4, dtype=np.float32),
            "direct/0": np.zeros(4, dtype=np.float32),
            "cue/0": np.zeros(4, dtype=np.float32),
        }
    )

    contract = infer_post_mix_role_contract(bundle)

    assert contract.observation_roles == ("mixture", "background")
    assert contract.protected_roles == ("target/0", "direct/0")
    assert contract.amplitude_roles == ("source_full_rir/0",)


def test_infer_post_mix_role_contract_rejects_missing_observation_role():
    with pytest.raises(KeyError, match="observation_roles"):
        infer_post_mix_role_contract(
            WaveformBundle({"mixture": np.zeros(4, dtype=np.float32)})
        )


def test_post_mix_contract_helpers_are_public_simulation_seams():
    assert simulation.PostMixRoleContract is PostMixRoleContract
    assert simulation.NonDecomposableObservationContract is NonDecomposableObservationContract
    assert simulation.infer_post_mix_role_contract is infer_post_mix_role_contract


def test_post_mix_role_contract_enforces_declared_mutable_stages():
    policy = _policy()
    policy.sim_cfg.peak_normalization_mode = "joint"
    policy.sim_cfg.peak_normalization_scale_range = (0.5, 0.5)
    contract = PostMixRoleContract(
        observation_roles=("mixture",),
        protected_roles=("target/0", "direct/0"),
        mutable_stages={"mixture": ("E", "F")},
    )
    with pytest.raises(ValueError, match="mutable_stages"):
        ObservationPostMixRenderer().apply(
            replace(
                _request(policy),
                protected_roles=("target/0", "direct/0"),
                role_contract=contract,
            )
        )


def test_d_gain_couples_target_and_source_but_final_facts_use_bundle():
    policy = _policy()
    policy.sim_cfg.peak_normalization_mode = "joint"
    policy.sim_cfg.peak_normalization_scale_range = (0.25, 0.25)
    request = ObservationPostMixRequest(
        scene_bundle=WaveformBundle(
            {
                "mixture": np.full(4, 0.5, dtype=np.float32),
                "background": np.full(4, 0.25, dtype=np.float32),
                "source_full_rir/0": np.full(4, 0.25, dtype=np.float32),
                "target/0": np.full(4, 0.25, dtype=np.float32),
            }
        ),
        sample_rate=16000,
        rng=np.random.default_rng(0),
        task_policy=policy,
        observation_roles=("mixture", "background"),
        protected_roles=("target/0",),
        amplitude_roles=("source_full_rir/0",),
    )

    result = ObservationPostMixRenderer().apply(request)

    np.testing.assert_allclose(result.bundle.require("target/0"), 0.125)
    np.testing.assert_allclose(result.bundle.require("source_full_rir/0"), 0.125)
    assert result.facts["decomposable_final"] is True
    assert result.facts["layer_d_facts"]["normalize"]["preserves_decomposition"] is True
    assert set(result.facts["layer_e"]) >= {"observation_changed"}
    assert result.facts["layer_f"]["observation_changed"] is False
