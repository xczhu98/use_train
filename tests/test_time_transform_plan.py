"""TimeTransformPlan 纯 Module Interface 合同测试。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect

import numpy as np
import pytest

from core.data.time_transform import (
    CanvasOp,
    CropOp,
    POLICY_FINAL_CENTER_CROP,
    POLICY_LAYER_A_RANDOM_CROP,
    POLICY_OFFLINE_RANDOM_CROP,
    ResampleOp,
    SceneTimeTransformResult,
    TileOp,
    TimeTransformPlan,
    WaveformBundle,
    apply_scene_time_transform,
    apply_time_transform,
    plan_time_transform,
)


pytestmark = [pytest.mark.unit]


def test_plan_is_frozen_and_preserves_policy_labels():
    expected = (
        POLICY_LAYER_A_RANDOM_CROP,
        POLICY_FINAL_CENTER_CROP,
        POLICY_OFFLINE_RANDOM_CROP,
    )
    plans = tuple(
        plan_time_transform(
            input_length=8,
            operations=(CropOp(start=index, output_length=6),),
            policy=policy,
        )
        for index, policy in enumerate(expected)
    )

    assert tuple(plan.policy for plan in plans) == expected
    assert all(plan.output_length == 6 for plan in plans)
    assert isinstance(plans[0].operations, tuple)
    with pytest.raises(FrozenInstanceError):
        plans[0].policy = "changed"
    with pytest.raises(FrozenInstanceError):
        plans[0].operations[0].start = 1


def test_crop_tile_canvas_apply_one_schedule_to_synchronized_tracks():
    clean = np.arange(8, dtype=np.float32)
    noisy = np.stack((clean + 10.0, clean + 20.0))
    bundle = WaveformBundle(
        {"clean": clean, "noisy": noisy},
        metadata={"sample_rate": 16_000},
    )
    plan = plan_time_transform(
        input_length=8,
        operations=(
            CropOp(start=2, output_length=4),
            TileOp(output_length=7, start=1, gap_length=2),
            CanvasOp(output_length=10, offset=2),
        ),
        policy=POLICY_OFFLINE_RANDOM_CROP,
    )

    result = apply_time_transform(plan, bundle)

    expected_clean = np.array([0, 0, 3, 4, 5, 0, 0, 2, 3, 0], dtype=np.float32)
    np.testing.assert_array_equal(result.tracks["clean"], expected_clean)
    np.testing.assert_array_equal(
        result.tracks["noisy"][0],
        np.where(expected_clean != 0, expected_clean + 10.0, 0.0),
    )
    np.testing.assert_array_equal(
        result.tracks["noisy"][1],
        np.where(expected_clean != 0, expected_clean + 20.0, 0.0),
    )
    assert result.tracks["noisy"].shape == (2, plan.output_length)
    assert plan.output_length == 10
    assert result.metadata == bundle.metadata
    assert all(track.dtype == np.float32 for track in result.tracks.values())
    assert all(not track.flags.writeable for track in result.tracks.values())


def test_bundle_owns_inputs_and_replay_is_deterministic_without_rng_interface():
    source = np.arange(5, dtype=np.float32)
    bundle = WaveformBundle({"speech": source})
    source[:] = -1.0
    plan = TimeTransformPlan(
        input_length=5,
        operations=(TileOp(output_length=9, start=3),),
        policy=POLICY_LAYER_A_RANDOM_CROP,
        required_tracks=("speech",),
    )

    first = apply_time_transform(plan, bundle)
    second = apply_time_transform(plan, bundle)

    np.testing.assert_array_equal(bundle.tracks["speech"], np.arange(5, dtype=np.float32))
    np.testing.assert_array_equal(first.tracks["speech"], second.tracks["speech"])
    assert "rng" not in inspect.signature(plan_time_transform).parameters
    assert "rng" not in inspect.signature(apply_time_transform).parameters


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: CropOp(start=-1, output_length=2), "non-negative"),
        (lambda: CropOp(start=0, output_length=0), "positive"),
        (lambda: TileOp(output_length=2, start=True), "non-negative"),
        (lambda: CanvasOp(output_length=2, offset=-1), "non-negative"),
        (lambda: ResampleOp(0, 16_000, 4), "positive"),
    ],
)
def test_operations_reject_invalid_integer_fields(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("input_length", "operations", "message"),
    [
        (4, (CropOp(start=2, output_length=3),), "exceeds"),
        (0, (TileOp(output_length=4),), "empty"),
        (3, (TileOp(output_length=4, start=3),), "tile period"),
        (4, (CanvasOp(output_length=5, offset=2),), "cannot place"),
    ],
)
def test_plan_fails_closed_on_impossible_geometry(input_length, operations, message):
    with pytest.raises(ValueError, match=message):
        TimeTransformPlan(input_length, operations, policy="test")


def test_required_tracks_and_input_length_fail_closed():
    waveform = np.ones(4, dtype=np.float32)
    plan = TimeTransformPlan(
        4,
        (),
        policy="identity",
        required_tracks=("clean", "noisy"),
    )
    with pytest.raises(ValueError, match="required tracks are missing"):
        apply_time_transform(plan, WaveformBundle({"clean": waveform}))

    wrong_length = WaveformBundle(
        {"clean": np.ones(3, dtype=np.float32), "noisy": np.ones(3, dtype=np.float32)}
    )
    with pytest.raises(ValueError, match="input length mismatch"):
        apply_time_transform(plan, wrong_length)


@pytest.mark.parametrize(
    ("waveform", "exception", "message"),
    [
        (np.ones(4, dtype=np.float64), TypeError, "float32"),
        (np.array([1.0, np.nan], dtype=np.float32), ValueError, "finite"),
        (np.array([1.0, np.inf], dtype=np.float32), ValueError, "finite"),
        (np.array(1.0, dtype=np.float32), ValueError, "time axis"),
    ],
)
def test_waveform_contract_rejects_invalid_arrays(waveform, exception, message):
    with pytest.raises(exception, match=message):
        WaveformBundle({"speech": waveform})


def test_resample_uses_injected_adapter_for_each_track_and_chains_exact_length():
    calls = []

    def fake_resample(waveform, source_sample_rate, target_sample_rate):
        calls.append((waveform.copy(), source_sample_rate, target_sample_rate))
        indices = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        return waveform[..., indices]

    bundle = WaveformBundle(
        {
            "clean": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "noisy": np.array([11.0, 12.0, 13.0], dtype=np.float32),
        }
    )
    plan = TimeTransformPlan(
        input_length=3,
        operations=(
            ResampleOp(8_000, 16_000, output_length=6),
            CropOp(start=1, output_length=4),
        ),
        policy="resample_then_crop",
    )

    result = apply_time_transform(plan, bundle, resample_adapter=fake_resample)

    assert [(source_sr, target_sr) for _, source_sr, target_sr in calls] == [
        (8_000, 16_000),
        (8_000, 16_000),
    ]
    np.testing.assert_array_equal(
        result.tracks["clean"],
        np.array([1.0, 2.0, 2.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.tracks["noisy"],
        np.array([11.0, 12.0, 12.0, 13.0], dtype=np.float32),
    )
    assert plan.output_length == 4


def test_resample_length_rounding_is_declared_by_adapter_contract():
    plan = TimeTransformPlan(
        input_length=3,
        operations=(ResampleOp(3, 2, output_length=3),),
        policy="adapter_rounding",
    )
    bundle = WaveformBundle({"speech": np.arange(3, dtype=np.float32)})

    result = apply_time_transform(
        plan,
        bundle,
        resample_adapter=lambda waveform, _source_sr, _target_sr: waveform.copy(),
    )

    assert result.tracks["speech"].shape[-1] == plan.output_length == 3


def test_resample_requires_adapter():
    plan = TimeTransformPlan(
        2,
        (ResampleOp(8_000, 16_000, output_length=4),),
        policy="resample",
    )
    bundle = WaveformBundle({"speech": np.ones(2, dtype=np.float32)})

    with pytest.raises(ValueError, match="requires an injected"):
        apply_time_transform(plan, bundle)


@pytest.mark.parametrize(
    ("adapter", "exception", "message"),
    [
        (lambda waveform, _source_sr, _target_sr: waveform.astype(np.float64), TypeError, "float32"),
        (lambda _waveform, _source_sr, _target_sr: np.ones(3, dtype=np.float32), ValueError, "length mismatch"),
        (
            lambda _waveform, _source_sr, _target_sr: np.ones((2, 4), dtype=np.float32),
            ValueError,
            "non-time axes",
        ),
        (
            lambda _waveform, _source_sr, _target_sr: np.array(
                [1.0, 2.0, 3.0, np.nan], dtype=np.float32
            ),
            ValueError,
            "non-finite",
        ),
    ],
)
def test_resample_adapter_output_contract_fails_closed(adapter, exception, message):
    plan = TimeTransformPlan(
        2,
        (ResampleOp(8_000, 16_000, output_length=4),),
        policy="resample",
    )
    bundle = WaveformBundle({"speech": np.ones(2, dtype=np.float32)})

    with pytest.raises(exception, match=message):
        apply_time_transform(plan, bundle, resample_adapter=adapter)


def test_scene_orchestration_resamples_and_applies_one_final_plan_to_every_role():
    class _FinalPlan:
        transform = TimeTransformPlan(
            6,
            (CropOp(start=1, output_length=4),),
            policy="final_test_crop",
        )
        output_length = 4

    calls = []

    def build_final_plan(speech, noisy, sample_rate, rng, **kwargs):
        calls.append((len(speech), len(noisy), sample_rate, kwargs))
        assert isinstance(rng, np.random.Generator)
        return _FinalPlan()

    def resample(waveform, source_sr, target_sr):
        assert (source_sr, target_sr) == (8_000, 16_000)
        return np.repeat(waveform, 2, axis=-1).astype(np.float32)

    result = apply_scene_time_transform(
        bundle=WaveformBundle(
            {
                "observation": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "target": np.array([11.0, 12.0, 13.0], dtype=np.float32),
            }
        ),
        source_sample_rate=8_000,
        target_sample_rate=16_000,
        rng=np.random.default_rng(12),
        fixed_silence_canvas_active=True,
        build_final_time_plan=build_final_plan,
        resample_adapter=resample,
        policy_prefix="se_target_sr",
    )

    assert isinstance(result, SceneTimeTransformResult)
    assert calls == [(6, 6, 16_000, {"fixed_silence_canvas_active": True})]
    assert result.plan.policy == "se_target_sr+final_test_crop"
    assert result.to_metadata()["required_tracks"] == ["observation", "target"]
    np.testing.assert_array_equal(
        result.bundle.require("observation"),
        np.array([1.0, 2.0, 2.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.bundle.require("target"),
        np.array([11.0, 12.0, 12.0, 13.0], dtype=np.float32),
    )
    assert "rng" in inspect.signature(apply_scene_time_transform).parameters
