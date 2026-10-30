"""AcousticScenePlan 与 typed replay Adapter 合同测试。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.data.acoustic_scene_plan import (
    AcousticActorPlan,
    AcousticReplayAdapter,
    AcousticScenePlan,
    AudioCropPlan,
    C0AssetReplayError,
    NoisePlan,
    NoiseSourcePlan,
    RIRPlan,
    ResolvedAssets,
    SourceGainPlan,
)
from core.data.acoustic_scene_replay import (
    AssetForcedReplayError,
    replay_target_acoustic_scene,
)


pytestmark = [pytest.mark.unit]


def test_build_spatial_scene_plan_is_task_neutral_and_preserves_actor_order():
    with pytest.raises(ValueError, match="position_m or trajectory"):
        AcousticActorPlan("speaker/a", "target")
    actors = (
        AcousticActorPlan("speaker/a", "target", position_m=(1., 1., 1.5), supervision_index=0),
        AcousticActorPlan("speaker/b", "interference", position_m=(2., 1., 1.5), supervision_index=1),
    )
    from dataclasses import replace
    plan = replace(_scene_plan(), actors=actors)
    assert tuple(actor.actor_uid for actor in plan.actors) == ("speaker/a", "speaker/b")
    assert plan.to_dict()["actors"][1]["supervision_index"] == 1


def _scene_plan() -> AcousticScenePlan:
    crop = AudioCropPlan(
        identity="noise:traffic",
        source_sample_rate=16_000,
        source_num_samples=48_000,
        start_sample=123,
        num_samples=16_000,
    )
    return AcousticScenePlan(
        sample_rate=16_000,
        num_samples=16_000,
        need_noise=True,
        rir=RIRPlan(
            enabled=True,
            uid="rir-1",
            path="rir/one.wav",
            moving=False,
            reuse_enabled=True,
            cache_hit=False,
            segment_sec=None,
            normalize=True,
            early_rir_sec=0.05,
            target_type="early",
        ),
        source_gain=SourceGainPlan(
            enabled=True,
            target_db=-24.0,
            gain=0.5,
            forced=False,
            reference="early_rir",
        ),
        noise=NoisePlan(
            enabled=True,
            mode="single",
            mix_ratio=None,
            sources=(
                NoiseSourcePlan(
                    family="ordinary",
                    uid="traffic",
                    path="noise/traffic.wav",
                    category_id=7,
                    category_name="Traffic noise",
                    crop=crop,
                    snr_db=12.0,
                    applied=True,
                ),
            ),
            colored_enabled=False,
            colored_type=None,
            colored_snr_db=None,
        ),
    )


def test_plan_is_frozen_json_safe_and_contains_no_waveform():
    plan = _scene_plan()

    with pytest.raises(FrozenInstanceError):
        plan.sample_rate = 8_000

    payload = plan.to_dict()
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert '"schema_version": 1' in encoded
    assert payload["rir"]["uid"] == "rir-1"
    assert payload["noise"]["sources"][0]["crop"]["start_sample"] == 123

    with pytest.raises(TypeError, match="JSON-safe|ndarray"):
        AcousticScenePlan(
            sample_rate=16_000,
            num_samples=4,
            need_noise=False,
            rir=RIRPlan.disabled(),
            source_gain=SourceGainPlan.disabled(),
            noise=NoisePlan.disabled(),
            identity={"waveform": np.ones(4, dtype=np.float32)},
        )


def test_resolved_assets_are_runtime_only_and_copy_waveforms():
    rir = np.arange(4, dtype=np.float32)
    noise = np.arange(6, dtype=np.float32).reshape(1, -1)
    assets = ResolvedAssets(rir=rir, noises=(noise,))

    rir[:] = -1.0
    noise[:] = -2.0
    np.testing.assert_array_equal(assets.rir, np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(
        assets.noises[0], np.arange(6, dtype=np.float32).reshape(1, -1)
    )
    assert not hasattr(assets, "to_dict")


def _assert_cache_equal(left, right):
    assert left[:1] == right[:1]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


@pytest.mark.parametrize(
    "cache",
    [
        None,
        (16_000, np.arange(5, dtype=np.float32)),
        (16_000, np.arange(6, dtype=np.float32).reshape(2, 3), 0.25, "moving"),
    ],
)
def test_typed_snapshot_capture_restore_deep_copies_rng_and_rir_cache(cache):
    rng = np.random.default_rng(91)
    owner = SimpleNamespace(_rir_reuse_cache=cache)
    expected_cache = (
        None
        if cache is None
        else (cache[0], cache[1].copy(), *cache[2:])
    )
    adapter = AcousticReplayAdapter(rng, owner)
    snapshot = adapter.capture()
    expected_first = int(rng.integers(0, 1_000_000))

    rng.integers(0, 1_000_000, size=10)
    if owner._rir_reuse_cache is not None:
        owner._rir_reuse_cache[1][...] = -99.0
    owner._rir_reuse_cache = None

    adapter.restore(snapshot)
    assert int(rng.integers(0, 1_000_000)) == expected_first
    if cache is None:
        assert owner._rir_reuse_cache is None
    else:
        _assert_cache_equal(owner._rir_reuse_cache, expected_cache)
        assert owner._rir_reuse_cache is not cache
        assert owner._rir_reuse_cache[1] is not cache[1]


def test_replay_returns_completion_snapshot_without_implicit_rollback():
    rng = np.random.default_rng(3)
    owner = SimpleNamespace(_rir_reuse_cache=None)
    adapter = AcousticReplayAdapter(rng, owner)
    start = adapter.capture()

    def operation():
        value = int(rng.integers(0, 100))
        owner._rir_reuse_cache = (
            16_000,
            np.array([value], dtype=np.float32),
        )
        return value

    first = adapter.replay(start, operation)
    rng.integers(0, 100, size=5)
    owner._rir_reuse_cache = None
    second = adapter.replay(start, operation)

    assert first.value == second.value
    assert first.completion.rng_state == second.completion.rng_state
    assert first.completion.rir_cache is not second.completion.rir_cache
    np.testing.assert_array_equal(
        first.completion.rir_cache.waveform,
        second.completion.rir_cache.waveform,
    )

    adapter.restore(first.completion)
    _assert_cache_equal(
        owner._rir_reuse_cache,
        (16_000, np.array([first.value], dtype=np.float32)),
    )


@pytest.mark.parametrize(
    "cache",
    [
        None,
        (16_000, np.arange(4, dtype=np.float32)),
        (16_000, np.arange(6, dtype=np.float32).reshape(2, 3), 0.25, "moving"),
    ],
)
def test_typed_replay_matches_legacy_waveform_rng_and_cache_completion(cache):
    """固定 seed 下 typed Adapter 与旧手写 restore 的三重状态完全一致。"""

    def run(use_adapter: bool):
        rng = np.random.default_rng(73)
        owner = SimpleNamespace(
            _rir_reuse_cache=(
                None
                if cache is None
                else (cache[0], cache[1].copy(), *cache[2:])
            )
        )

        def operation():
            waveform = rng.normal(size=8).astype(np.float32)
            owner._rir_reuse_cache = (
                16_000,
                waveform[:4].copy(),
                *(cache[2:] if cache is not None else ()),
            )
            return waveform

        if use_adapter:
            adapter = AcousticReplayAdapter(rng, owner)
            start = adapter.capture()
            result = adapter.replay(start, operation)
            adapter.restore(result.completion)
            waveform = result.value
        else:
            rng_state = rng.bit_generator.state.copy()
            cache_state = (
                None
                if owner._rir_reuse_cache is None
                else (
                    owner._rir_reuse_cache[0],
                    owner._rir_reuse_cache[1].copy(),
                    *owner._rir_reuse_cache[2:],
                )
            )
            rng.bit_generator.state = rng_state
            owner._rir_reuse_cache = cache_state
            waveform = operation()
        return waveform, rng.bit_generator.state, owner._rir_reuse_cache

    typed = run(True)
    legacy = run(False)
    np.testing.assert_array_equal(typed[0], legacy[0])
    assert typed[1] == legacy[1]
    _assert_cache_equal(typed[2], legacy[2])


def _asset_replay_scene(*, moving: bool = False, gain_enabled: bool = False):
    rir = (
        np.array([[1.0, 0.25], [0.5, 0.125]], dtype=np.float32)
        if moving
        else np.array([[1.0, 0.25]], dtype=np.float32)
    )
    plan = AcousticScenePlan(
        sample_rate=16_000,
        num_samples=8,
        need_noise=True,
        rir=RIRPlan(
            enabled=True,
            uid="rir-asset",
            path="rir/asset.wav",
            moving=moving,
            reuse_enabled=False,
            cache_hit=False,
            segment_sec=0.01 if moving else None,
            normalize=False,
            early_rir_sec=1.0,
            target_type="early",
        ),
        source_gain=SourceGainPlan(
            enabled=gain_enabled,
            target_db=None,
            gain=0.5 if gain_enabled else 1.0,
            forced=True,
            reference="early_rir" if gain_enabled else None,
        ),
        # formal mixture 的 C-1/C-2/C-3 记录可保留；replay 本身显式 need_noise=False。
        noise=NoisePlan.disabled(),
    )
    assets = ResolvedAssets(
        rir=rir,
        role="scene",
        sample_rate=16_000,
        num_samples=8,
    )
    return plan, assets


def _replay_target(plan, assets, **overrides):
    kwargs = {
        "scene_plan": plan,
        "resolved_assets": assets,
        "asset_role": "scene",
        "sample_rate": 16_000,
        "num_samples": 8,
        "need_noise": False,
        "source_volume_normalization_enabled": plan.source_gain.enabled,
        "source_volume_use_rms": True,
    }
    kwargs.update(overrides)
    return replay_target_acoustic_scene(
        np.array([[1.0, 0.0, -0.5, 0.0, 0.25, 0.0, 0.0, 0.0]], dtype=np.float32),
        **kwargs,
    )


def test_asset_forced_replay_public_helper_defaults_need_noise_to_false():
    plan, assets = _asset_replay_scene()
    target = np.array(
        [[1.0, 0.0, -0.5, 0.0, 0.25, 0.0, 0.0, 0.0]], dtype=np.float32
    )
    replayed = replay_target_acoustic_scene(
        target,
        scene_plan=plan,
        resolved_assets=assets,
        asset_role="scene",
        sample_rate=16_000,
        num_samples=8,
        source_volume_normalization_enabled=False,
        source_volume_use_rms=True,
    )
    explicit = _replay_target(plan, assets)
    np.testing.assert_array_equal(replayed.noisy_speech, explicit.noisy_speech)
    np.testing.assert_array_equal(replayed.mix_reference, explicit.mix_reference)
    np.testing.assert_array_equal(replayed.clean_target, explicit.clean_target)


def test_asset_forced_replay_public_helper_has_no_rng_io_or_cache_interface(monkeypatch):
    """公共 helper 仅消费已解析 plan/assets 与 target，不触碰外部状态。"""
    parameter_names = set(inspect.signature(replay_target_acoustic_scene).parameters)
    assert {"rng", "cache", "path", "owner"}.isdisjoint(parameter_names)
    assert C0AssetReplayError is AssetForcedReplayError

    def _forbidden_open(*_args, **_kwargs):
        raise AssertionError("asset-forced replay must not perform I/O")

    monkeypatch.setattr("builtins.open", _forbidden_open)
    plan, assets = _asset_replay_scene()
    replayed = _replay_target(plan, assets)
    expected = np.array([[1.0, 0.25, -0.5, -0.125, 0.25, 0.0625, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(replayed.noisy_speech, expected, atol=1e-6)
    np.testing.assert_allclose(replayed.mix_reference, expected, atol=1e-6)
    np.testing.assert_allclose(replayed.clean_target, expected, atol=1e-6)
    assert replayed.source_volume_calibration is None


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("role", "resolved asset role or length mismatch"),
        ("sample_rate", "resolved asset role or length mismatch"),
        ("length", "resolved asset role or length mismatch"),
        ("noise", "only supports need_noise=False"),
        ("missing_rir", "RIR plan and asset mismatch"),
        ("rir_shape", "RIR asset must be finite and non-empty"),
        ("moving", "RIR moving shape does not match plan"),
        ("gain_config", "source gain plan does not match config"),
        ("unsafe_gain", "forced source volume gain is unsafe for C-0 stems"),
        ("nonfinite_gain", "forced_gain must be finite and non-negative"),
    ],
)
def test_asset_forced_replay_helper_rejects_contract_corruption(mutation, match):
    """C-0 helper 对 assets/role/RIR/gain 合同一律不可恢复地拒绝。"""
    plan, assets = _asset_replay_scene(
        gain_enabled=mutation in {"unsafe_gain", "nonfinite_gain"}
    )
    kwargs = {}
    if mutation == "role":
        assets = ResolvedAssets(rir=assets.rir, role="other", sample_rate=16_000, num_samples=8)
    elif mutation == "sample_rate":
        assets = ResolvedAssets(rir=assets.rir, role="scene", sample_rate=8_000, num_samples=8)
    elif mutation == "length":
        assets = ResolvedAssets(rir=assets.rir, role="scene", sample_rate=16_000, num_samples=7)
    elif mutation == "noise":
        kwargs["need_noise"] = True
    elif mutation == "missing_rir":
        assets = ResolvedAssets(role="scene", sample_rate=16_000, num_samples=8)
    elif mutation == "rir_shape":
        assets = ResolvedAssets(rir=np.array([], dtype=np.float32), role="scene", sample_rate=16_000, num_samples=8)
    elif mutation == "moving":
        plan = AcousticScenePlan(
            sample_rate=plan.sample_rate,
            num_samples=plan.num_samples,
            need_noise=plan.need_noise,
            rir=RIRPlan(
                enabled=True, uid="rir-asset", path="rir/asset.wav", moving=True,
                reuse_enabled=False, cache_hit=False, segment_sec=0.01,
                normalize=False, early_rir_sec=1.0, target_type="early",
            ),
            source_gain=plan.source_gain,
            noise=plan.noise,
        )
    elif mutation == "gain_config":
        kwargs["source_volume_normalization_enabled"] = True
    elif mutation == "unsafe_gain":
        object.__setattr__(plan.source_gain, "gain", 1e39)
    elif mutation == "nonfinite_gain":
        object.__setattr__(plan.source_gain, "gain", float("inf"))
    with pytest.raises(AssetForcedReplayError, match=match):
        _replay_target(plan, assets, **kwargs)


def test_asset_forced_replay_normalizes_float64_target_to_canonical_float32():
    plan, assets = _asset_replay_scene()
    target_storage = np.zeros((1, 16), dtype=np.float64)
    target_storage[:, ::2] = [1.0, 0.0, -0.5, 0.0, 0.25, 0.0, 0.0, 0.0]
    target = target_storage[:, ::2]
    assert not target.flags.c_contiguous
    replayed = replay_target_acoustic_scene(
        target,
        scene_plan=plan,
        resolved_assets=assets,
        asset_role="scene",
        sample_rate=16_000,
        num_samples=8,
        source_volume_normalization_enabled=False,
        source_volume_use_rms=True,
    )
    expected = _replay_target(plan, assets)
    for actual, reference in (
        (replayed.noisy_speech, expected.noisy_speech),
        (replayed.mix_reference, expected.mix_reference),
        (replayed.clean_target, expected.clean_target),
    ):
        assert actual.dtype == np.float32
        assert actual.flags.c_contiguous
        np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
        float(np.finfo(np.float32).max) * 2.0,
    ],
)
def test_asset_forced_replay_rejects_nonfinite_or_unrepresentable_float64_target(value):
    plan, assets = _asset_replay_scene()
    target = np.zeros((1, 8), dtype=np.float64)
    target[0, 0] = value
    with pytest.raises(AssetForcedReplayError, match="finite|float32 range"):
        replay_target_acoustic_scene(
            target,
            scene_plan=plan,
            resolved_assets=assets,
            asset_role="scene",
            sample_rate=16_000,
            num_samples=8,
            source_volume_normalization_enabled=False,
            source_volume_use_rms=True,
        )


def test_asset_forced_replay_helper_supports_moving_rir():
    plan, assets = _asset_replay_scene(moving=True)
    replayed = _replay_target(plan, assets)
    assert replayed.noisy_speech.shape == (1, 8)
    assert np.isfinite(replayed.noisy_speech).all()


@pytest.mark.parametrize(
    "target",
    [
        np.array([["not-a-number"] * 8], dtype=object),
        np.array([["not-a-number"] * 8], dtype="U12"),
    ],
)
def test_asset_forced_replay_wraps_invalid_target_waveform_types(target):
    plan, assets = _asset_replay_scene()
    with pytest.raises(AssetForcedReplayError, match="target waveform"):
        replay_target_acoustic_scene(
            target,
            scene_plan=plan,
            resolved_assets=assets,
            asset_role="scene",
            sample_rate=16_000,
            num_samples=8,
            source_volume_normalization_enabled=False,
            source_volume_use_rms=True,
        )


@pytest.mark.parametrize(
    "rir",
    [
        np.array([["not-a-number", "also-bad"]], dtype=object),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.zeros((1, 1, 2), dtype=np.float32),
    ],
)
def test_asset_forced_replay_wraps_invalid_rir_inputs(rir):
    plan, _ = _asset_replay_scene()
    assets = SimpleNamespace(
        rir=rir, role="scene", sample_rate=16_000, num_samples=8
    )
    with pytest.raises(AssetForcedReplayError, match="RIR"):
        _replay_target(plan, assets)


def test_asset_forced_replay_does_not_wrap_internal_value_errors(monkeypatch):
    plan, assets = _asset_replay_scene()

    def _internal_failure(*_args, **_kwargs):
        raise ValueError("internal replay failure")

    monkeypatch.setattr(
        "core.data.acoustic_scene_replay.add_paired_reverberation",
        _internal_failure,
    )
    with pytest.raises(ValueError, match="internal replay failure"):
        _replay_target(plan, assets)


def test_scene_plan_accepts_shared_room_array_timebase_and_actor_identity():
    from core.data.simulation.array_profiles import get_array_profile
    from core.data.simulation.trajectory import TrajectoryTimebase, plan_linear_source_trajectory

    timebase = TrajectoryTimebase(16_000, 32, 8, 3)
    trajectory = plan_linear_source_trajectory(
        np.array([[0.5, 0.0, 1.2]], dtype=np.float32),
        np.array([[0.8, 0.0, 1.2]], dtype=np.float32),
        timebase,
    )
    actor = AcousticActorPlan(
        actor_uid="target/0",
        role="target",
        supervision_index=0,
        trajectory=trajectory,
    )
    plan = AcousticScenePlan(
        sample_rate=16_000,
        num_samples=32,
        need_noise=False,
        rir=RIRPlan.disabled(),
        source_gain=SourceGainPlan.disabled(),
        noise=NoisePlan.disabled(),
        room={"room_uid": "room-1", "rt60": 0.4},
        array=get_array_profile("linear_2ch_80mm").resolve(reference_channel=0),
        timebase=timebase,
        actors=(actor,),
    )

    payload = plan.to_dict()
    assert payload["room"]["room_uid"] == "room-1"
    assert payload["array"]["topology"]["channels"] == 2
    assert payload["timebase"]["frame_samples"] == [0, 8, 16]
    assert payload["actors"][0]["actor_uid"] == "target/0"
    assert payload["actors"][0]["trajectory"]["timebase"]["num_samples"] == 32


def test_scene_plan_rejects_duplicate_actor_identity_and_mismatched_timebase():
    from core.data.simulation.trajectory import TrajectoryTimebase

    timebase = TrajectoryTimebase(16_000, 32, 8, 3)
    other_timebase = TrajectoryTimebase(16_000, 32, 4, 3)
    actor = AcousticActorPlan(actor_uid="speaker", role="target", position_m=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="actor_uid"):
        AcousticScenePlan(
            sample_rate=16_000,
            num_samples=32,
            need_noise=False,
            rir=RIRPlan.disabled(),
            source_gain=SourceGainPlan.disabled(),
            noise=NoisePlan.disabled(),
            actors=(actor, actor),
        )
    from core.data.simulation.trajectory import SourceTrajectory

    moving_actor = AcousticActorPlan(
        actor_uid="moving",
        role="target",
        trajectory=SourceTrajectory(
            np.zeros((3, 1, 3), dtype=np.float32), other_timebase
        ),
    )
    with pytest.raises(ValueError, match="share the scene timebase"):
        AcousticScenePlan(
            sample_rate=16_000,
            num_samples=32,
            need_noise=False,
            rir=RIRPlan.disabled(),
            source_gain=SourceGainPlan.disabled(),
            noise=NoisePlan.disabled(),
            timebase=timebase,
            actors=(moving_actor,),
        )
