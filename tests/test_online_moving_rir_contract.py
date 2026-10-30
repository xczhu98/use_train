"""在线 moving RIR typed contract tests using a deterministic fake gpuRIR module."""

import numpy as np
import pytest

from core.data.simulation.gpurir_backend import (
    generate_moving_rir_group,
    generate_moving_rir_tracks,
)

pytestmark = pytest.mark.unit


class _FakeGpuRIR:
    calls = 0

    def beta_SabineEstimation(self, room, rt60):
        return np.ones(6, dtype=np.float32) * 0.8

    def att2t_SabineEstimator(self, diffuse_db, rt60):
        return 0.01

    def t2n(self, diffuse, room, c=343.0):
        return (2, 2, 2)

    def simulateRIR(self, *, room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs,
                    Tdiff, spkr_pattern, mic_pattern, orV_src, orV_rcv, c):
        type(self).calls += 1
        sources = np.asarray(pos_src, dtype=np.float32)
        mics = np.asarray(pos_rcv, dtype=np.float32)
        if mics.ndim == 1:
            mics = mics[None]
        kind_scale = 1.0 if tuple(nb_img) != (1, 1, 1) else 0.5
        frames = 8
        wave = np.zeros((sources.shape[0], mics.shape[0], frames), dtype=np.float32)
        for source in range(sources.shape[0]):
            wave[source, :, 0] = kind_scale * (1.0 + 0.1 * sources[source, 0])
        return wave


def _plan():
    return {
        "room_dim": [4.0, 4.0, 2.5],
        "source_loc": [[1.0, 1.0, 1.5]],
        "trajectory": [[[1.0, 1.0, 1.5]], [[1.2, 1.0, 1.5]], [[1.4, 1.0, 1.5]]],
        "trajectory_timebase": {
            "sample_rate": 16000,
            "num_samples": 64,
            "hop_samples": 16,
            "frame_count": 3,
            "start_sample": 0,
        },
        "mic_loc": [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]],
        "num_samples": 8,
        "sample_rate": 16000,
        "rt60": 0.2,
        "diffuse_db": 15.0,
        "sound_speed": 343.0,
        "group_uid": "moving-fixture",
    }


def test_moving_full_group_preserves_trajectory_and_shape():
    bank, facts = generate_moving_rir_group(_plan(), _FakeGpuRIR())
    assert bank.shape == (3, 1, 2, 8)
    assert facts["moving"] is True
    assert facts["track_kind"] == "full"
    assert len(facts["frame_facts"]) == 3
    np.testing.assert_allclose(facts["trajectory"], _plan()["trajectory"])


def test_moving_full_direct_tracks_share_scene_and_differ():
    tracks = generate_moving_rir_tracks(_plan(), _FakeGpuRIR())
    full, full_facts = tracks["full"]
    direct, direct_facts = tracks["direct"]
    assert full.shape == direct.shape == (3, 1, 2, 8)
    assert full_facts["track_kind"] == "full"
    assert direct_facts["track_kind"] == "direct"
    np.testing.assert_allclose(full_facts["trajectory"], direct_facts["trajectory"])
    assert not np.array_equal(full, direct)


def test_moving_plan_requires_source_loc_equal_to_first_frame():
    plan = _plan()
    plan["source_loc"] = [[0.0, 0.0, 1.5]]
    with pytest.raises(ValueError, match="source_loc"):
        generate_moving_rir_group(plan, _FakeGpuRIR())


def test_moving_generation_batches_trajectory_frames_into_fewer_backend_calls():
    plan = _plan()
    _FakeGpuRIR.calls = 0
    bank, _ = generate_moving_rir_group(plan, _FakeGpuRIR())
    assert bank.shape == (3, 1, 2, 8)
    assert _FakeGpuRIR.calls < len(plan["trajectory"])
