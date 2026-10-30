import numpy as np
import pytest

from core.data.simulation.rng import SimulationRNG
from core.data.speech_scene import scene_rng


pytestmark = pytest.mark.unit


class _Dataset:
    _worker_base_seed = [17]
    _worker_epoch = [3]
    _simulation_split = "train"


def test_named_stream_preserves_existing_seed_mapping():
    dataset = _Dataset()
    named = SimulationRNG.for_task(dataset, 5, "ssl").stream("post_mix")
    legacy = scene_rng(dataset, 5, 300)

    np.testing.assert_array_equal(
        named.integers(0, 2**31, size=8),
        legacy.integers(0, 2**31, size=8),
    )


def test_source_offset_preserves_existing_source_stream_mapping():
    dataset = _Dataset()
    named = SimulationRNG.for_task(dataset, 5, "ssl").stream(
        "speech_preparation", offset=2
    )
    legacy = scene_rng(dataset, 5, 102)

    np.testing.assert_array_equal(
        named.integers(0, 2**31, size=8),
        legacy.integers(0, 2**31, size=8),
    )


def test_tse_enrollment_offset_preserves_existing_stream_mapping():
    dataset = _Dataset()
    named = SimulationRNG.for_task(dataset, 5, "tse").stream(
        "enrollment", offset=2
    )
    legacy = scene_rng(dataset, 5, 10002)

    np.testing.assert_array_equal(
        named.integers(0, 2**31, size=8),
        legacy.integers(0, 2**31, size=8),
    )


def test_tse_timeline_uses_a_dedicated_named_stream():
    dataset = _Dataset()
    named = SimulationRNG.for_task(dataset, 5, "tse")
    assert named.stream_facts()["streams"]["timeline"] == 303
    np.testing.assert_array_equal(
        named.stream("timeline").integers(0, 2**31, size=8),
        scene_rng(dataset, 5, 303).integers(0, 2**31, size=8),
    )


def test_unknown_task_or_stream_fails_fast():
    dataset = _Dataset()
    with pytest.raises(ValueError, match="no registered task"):
        SimulationRNG.for_task(dataset, 0, "aec")
    rng = SimulationRNG.for_task(dataset, 0, "ssl")
    with pytest.raises(KeyError, match="unknown ssl"):
        rng.stream("new_random_branch")
