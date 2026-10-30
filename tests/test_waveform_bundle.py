"""WaveformBundle 不变量与同步变换 Interface 的 unit 测试。"""
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from core.data.waveform_bundle import WaveformBundle

pytestmark = [pytest.mark.unit]

def _bundle() -> WaveformBundle:
    return WaveformBundle(
        {"speech": np.array([0.25, -0.5, 0.0], np.float32),
         "noise": np.array([[0.1, 0.0, -0.1]], np.float32)},
        {"sample_rate": 16_000, "tags": ["synthetic", True]},
    )

def test_constructor_copies_and_freezes_one_timeline():
    source, metadata = np.arange(3, dtype=np.float32), {"nested": ["original"]}
    bundle = WaveformBundle({"custom_role": source}, metadata)
    source[:] = -1
    metadata["nested"].append("changed")
    np.testing.assert_array_equal(bundle.require("custom_role"), np.arange(3, dtype=np.float32))
    assert bundle.num_samples == 3 and bundle.metadata["nested"] == ("original",)
    with pytest.raises(ValueError, match="read-only"):
        bundle.require("custom_role")[0] = 1
    with pytest.raises(ValueError):
        bundle.require("custom_role").setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        bundle.tracks = {}
    with pytest.raises(KeyError, match="required waveform role 'echo'"):
        bundle.require("echo")

@pytest.mark.parametrize("tracks,error,message", [
    ({}, ValueError, "at least one"), ({"x": [0.0]}, TypeError, "numpy.ndarray"),
    ({"x": np.ones(2, np.float64)}, TypeError, "float32"),
    ({"x": np.array(1, np.float32)}, ValueError, "time axis"),
    ({"x": np.array([np.inf], np.float32)}, ValueError, "finite"),
    ({"x": np.ones(2, np.float32), "y": np.ones(3, np.float32)}, ValueError, "timeline")])
def test_invalid_tracks_fail_closed(tracks, error, message):
    with pytest.raises(error, match=message):
        WaveformBundle(tracks)

@pytest.mark.parametrize("metadata,error", [
    ({"waveform": np.ones(1, np.float32)}, TypeError),
    ({"score": float("nan")}, ValueError), ({"unsupported": {1, 2}}, TypeError),
    ({1: "non-string key"}, TypeError)])
def test_metadata_is_json_safe_and_never_contains_ndarray(metadata, error):
    with pytest.raises(error):
        WaveformBundle({"x": np.ones(1, np.float32)}, metadata)

def test_replace_and_external_plan_mapping_revalidate_without_mutation():
    original = _bundle()
    replaced = original.replace(tracks={"target": np.ones((2, 3), np.float32)})
    seen = []
    mapped = original.map_tracks(lambda role, wav: (seen.append(role) or wav * np.float32(2)))
    assert set(original.tracks) == {"speech", "noise"}
    assert set(replaced.tracks) == {"target"} and seen == ["speech", "noise"]
    np.testing.assert_array_equal(mapped.require("speech"), original.require("speech") * 2)
    with pytest.raises(ValueError, match="broadcast/crop forbidden"):
        original.map_tracks(lambda _, wav: wav[..., :-1])
    with pytest.raises(TypeError, match="numpy.ndarray"):
        original.map_tracks(lambda _role, _wav: np.float32(1))

def test_uniform_gain_preserves_float32_and_fails_closed():
    bundle, gained = _bundle(), _bundle().apply_gain(0.5)
    for role, waveform in bundle.tracks.items():
        assert gained.require(role).dtype == np.float32
        np.testing.assert_array_equal(gained.require(role), waveform * np.float32(0.5))
    for invalid in (True, np.array([0.5], np.float32)):
        with pytest.raises(TypeError, match="real scalar"):
            bundle.apply_gain(invalid)
    for invalid in (float("nan"), float(np.finfo(np.float32).max) * 4):
        with pytest.raises(ValueError, match="finite"):
            bundle.apply_gain(invalid)
