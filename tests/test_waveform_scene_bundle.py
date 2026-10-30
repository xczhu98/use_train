"""WaveformSceneBundle 的角色合同和投影测试。"""

import numpy as np
import pytest

from core.data.waveform_bundle import WaveformBundle
from core.data.waveform_scene_bundle import (
    WaveformSceneBundle,
    WaveformSceneContract,
)


pytestmark = [pytest.mark.unit]


def _contract(**changes: object) -> WaveformSceneContract:
    values = {
        "required_roles": ("mixture", "background", "source_full_rir/0", "target/0"),
        "source_count": 1,
        "target_count": 1,
        "observation_role": "mixture",
        "decomposition_roles": ("background", "source_full_rir/0"),
        "declared_roles": (
            "mixture",
            "background",
            "source_full_rir/*",
            "target/*",
        ),
    }
    values.update(changes)
    return WaveformSceneContract(**values)


def _scene() -> WaveformSceneBundle:
    waveform = np.array([0.25, -0.5, 0.0], dtype=np.float32)
    return WaveformSceneBundle.from_tracks(
        {
            "mixture": waveform,
            "background": waveform * np.float32(0.1),
            "source_full_rir/0": waveform * np.float32(0.9),
            "target/0": waveform * np.float32(0.8),
        },
        metadata={"sample_rate": 16_000},
        contract=_contract(),
    )


def test_scene_contract_exposes_roles_counts_and_observation():
    scene = _scene()

    assert scene.roles == ("mixture", "background", "source_full_rir/0", "target/0")
    assert scene.source_roles == ("source_full_rir/0",)
    assert scene.target_roles == ("target/0",)
    assert scene.source_count == 1
    assert scene.target_count == 1
    assert scene.observation_role == "mixture"
    assert scene.decomposition_roles == ("background", "source_full_rir/0")
    assert scene.has("mixture")
    assert not scene.has("echo")
    np.testing.assert_array_equal(scene.require("target/0"), np.array([0.2, -0.4, 0.0], np.float32))


def test_scene_validates_absent_target_without_counting_it_as_target_source():
    scene = WaveformSceneBundle.from_tracks(
        {
            "mixture": np.ones(4, np.float32),
            "background": np.ones(4, np.float32),
            "source_full_rir/0": np.ones(4, np.float32),
            "target/absent": np.zeros(4, np.float32),
        },
        contract=WaveformSceneContract(
            required_roles=("mixture", "background", "target/absent"),
            source_count=1,
            target_count=0,
            observation_role="mixture",
            decomposition_roles=("background", "source_full_rir/0"),
            declared_roles=("mixture", "background", "source_full_rir/*", "target/absent"),
        ),
    )

    assert scene.source_count == 1
    assert scene.target_count == 0
    assert scene.target_roles == ()
    assert scene.has("target/absent")


@pytest.mark.parametrize(
    "tracks,contract,match",
    [
        (
            {"mixture": np.ones(3, np.float32), "source_full_rir/1": np.ones(3, np.float32)},
            WaveformSceneContract(source_count=2),
            "contiguous",
        ),
        (
            {"mixture": np.ones(3, np.float32), "target/unknown": np.ones(3, np.float32)},
            WaveformSceneContract(),
            "target role",
        ),
        (
            {"mixture": np.ones(3, np.float32), "echo": np.ones(3, np.float32)},
            WaveformSceneContract(declared_roles=("mixture", "background")),
            "undeclared",
        ),
    ],
)
def test_scene_contract_fails_closed(tracks, contract, match):
    with pytest.raises((KeyError, ValueError), match=match):
        WaveformSceneBundle.from_tracks(tracks, contract=contract)


def test_task_view_is_a_selected_immutable_snapshot():
    scene = _scene()
    view = scene.to_task_view(("mixture", "target/0"))

    assert tuple(view.tracks) == ("mixture", "target/0")
    assert view.metadata["sample_rate"] == 16_000
    with pytest.raises(ValueError, match="read-only"):
        view.require("mixture")[0] = 1.0
    with pytest.raises(KeyError, match="missing"):
        scene.to_task_view(("mixture", "echo"))


def test_scene_rewraps_waveform_bundle_snapshot_without_changing_contract():
    source = np.ones(3, np.float32)
    bundle = WaveformBundle({"mixture": source})
    scene = WaveformSceneBundle(
        bundle,
        WaveformSceneContract(observation_role="mixture", required_roles=("mixture",)),
    )
    source[:] = 0.0

    np.testing.assert_array_equal(scene.require("mixture"), np.ones(3, np.float32))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"source_count": -1}, "non-negative"),
        ({"target_count": True}, "non-negative integer"),
        ({"required_roles": ("mixture", "mixture")}, "duplicate"),
    ],
)
def test_contract_arguments_are_immutable_and_typed(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        WaveformSceneContract(**kwargs)
