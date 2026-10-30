import numpy as np
import pytest

from core.data.simulation import (
    MovingRIRCapabilityError,
    MovingSceneRIRRequest,
    SpatialRIRProvider,
    validate_moving_scene_rir_request,
)
from core.data.simulation.trajectory import TrajectoryTimebase, plan_linear_source_trajectory


def _request():
    timebase = TrajectoryTimebase(16000, 32, 16, 2)
    a = plan_linear_source_trajectory(
        np.array([[0.5, 1.0, 1.4]], dtype=np.float32),
        np.array([[0.7, 1.0, 1.4]], dtype=np.float32), timebase,
    )
    b = plan_linear_source_trajectory(
        np.array([[2.0, 1.0, 1.4]], dtype=np.float32),
        np.array([[1.8, 1.0, 1.4]], dtype=np.float32), timebase,
    )
    return MovingSceneRIRRequest(
        sample_rate=16000,
        num_samples=32,
        actor_trajectories={"nearend": a, "loudspeaker": b},
        room={"room_uid": "room-1"},
        array={"positions_m": [[0, 0, 0], [0.04, 0, 0]]},
        timebase=timebase,
    )


@pytest.mark.unit
def test_multi_actor_request_preserves_independent_tracks_and_shared_contract():
    request = _request()
    assert request.actor_count == 2
    assert request.microphone_count == 2
    assert request.source_uids == ("nearend", "loudspeaker")
    assert request.actor_trajectories["nearend"] is not request.actor_trajectories["loudspeaker"]
    assert validate_moving_scene_rir_request(request) is request


@pytest.mark.unit
def test_multi_actor_request_rejects_mismatched_timebase():
    request = _request()
    other = TrajectoryTimebase(16000, 32, 8, 2)
    trajectory = plan_linear_source_trajectory(
        np.zeros((1, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32), other,
    )
    with pytest.raises(ValueError, match="share one trajectory timebase"):
        MovingSceneRIRRequest(
            sample_rate=16000, num_samples=32,
            actor_trajectories={"nearend": request.actor_trajectories["nearend"], "loudspeaker": trajectory},
            room=request.room, array=request.array, timebase=request.timebase,
        )


@pytest.mark.unit
def test_default_provider_reports_capability_error_without_generating_data():
    request = _request()
    with pytest.raises(MovingRIRCapabilityError, match="independent actor trajectories"):
        SpatialRIRProvider().sample_moving_scene_tracks(request)


@pytest.mark.unit
def test_request_rejects_source_order_mismatch():
    request = _request()
    with pytest.raises(ValueError, match="source_uids"):
        MovingSceneRIRRequest(
            sample_rate=request.sample_rate, num_samples=request.num_samples,
            actor_trajectories=request.actor_trajectories, room=request.room,
            array=request.array, timebase=request.timebase,
            source_uids=("loudspeaker", "nearend"),
        )
