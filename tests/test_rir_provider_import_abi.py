"""RIR provider ABI 与 simulation package import 边界测试。"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core.data.simulation.rir_provider import SpatialRIRProvider
from core.data.simulation.spatial_scene import SpatialSceneRenderer, SpatialSceneRequest


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "module",
    (
        "core.data.spatial",
        "core.data.simulation.scene_renderer",
        "core.data.dynamic_dataset_base",
    ),
)
def test_spatial_modules_import_without_cycle(module):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_sample_spatial_scene_forwards_canonical_source_uids_keyword():
    class Provider(SpatialRIRProvider):
        def get_rirs(self, sample_rate, source_count, rng, *, source_uids=(), normalize=True, distance_range=None):
            assert sample_rate == 16000
            assert source_count == 2
            assert isinstance(source_uids, tuple)
            assert source_uids == ("target", "interference")
            return np.ones((2, 1, 1), dtype=np.float32)

    result = Provider().sample_spatial_scene(
        16000,
        2,
        np.random.default_rng(0),
        source_uids=["target", "interference"],
    )
    np.testing.assert_array_equal(result, np.ones((2, 1, 1), dtype=np.float32))


def test_spatial_scene_accepts_scene_provider_and_forwards_normalize():
    seen = {}
    from dataclasses import replace
    from test_spatial_scene_module import _scene_plan
    from core.data.simulation.rir_provider import SpatialRIRResult, SpatialRIRTrackSet
    plan = replace(_scene_plan(actor_order=("target",)), num_samples=3)
    class Provider:
        def sample_scene_tracks(self, scene_plan, *, normalize, track_kinds):
            seen.update(source_uids=tuple(a.actor_uid for a in scene_plan.actors), normalize=normalize)
            return SpatialRIRTrackSet(full=SpatialRIRResult(
                np.ones((1, 2, 3), dtype=np.float32),
                {"positions_m": np.asarray(plan.array.positions_m).tolist()},
                {"scene_uid": plan.identity["scene_uid"], "room": dict(plan.room)}, "fixture",
            ))

    result = SpatialSceneRenderer().render(
        SpatialSceneRequest(
            sources=np.ones((1, 3), dtype=np.float32),
            sample_rate=16000,
            provider=Provider(),
            source_uids=("target",),
            track_kinds=("full",),
            rng=np.random.default_rng(0),
            normalize=False,
            scene_plan=plan,
        )
    )

    assert seen["source_uids"] == ("target",)
    assert seen["normalize"] is False
    assert result.full_tracks.shape == (1, 2, 3)
