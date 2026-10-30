"""Scene sampler → provider → 真实 socket/worker → backend → renderer。"""
from dataclasses import replace

import numpy as np
import pytest

from core.data.simulation.rir_config import OnlineRIRConfig
from core.data.simulation.rir_provider import OnlineRIRProvider
from core.data.simulation.rir_service import close_online_rir_services
from core.data.simulation.scene_sampler import prepare_scene_plan
from core.data.simulation.spatial_scene import SpatialSceneRenderer, SpatialSceneRequest
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_views import project_simulation_view
from test_scene_level_rir_ipc import _FakeGpuRIR, _server


class _GeometryBackend(_FakeGpuRIR):
    def simulateRIR(self, *, pos_src, pos_rcv, **kwargs):
        length = int(round(kwargs["Tmax"] * kwargs["fs"]))
        distances = np.linalg.norm(np.asarray(pos_src)[:, None] - np.asarray(pos_rcv)[None], axis=-1)
        wave = np.zeros((*distances.shape, length), dtype=np.float32)
        for source, microphone in np.ndindex(distances.shape):
            arrival = min(length-1, int(round(distances[source, microphone] / kwargs["c"] * kwargs["fs"])))
            wave[source, microphone, arrival] = 1 / max(distances[source, microphone], .1)
            if kwargs["Tdiff"] < kwargs["Tmax"]:
                wave[source, microphone, min(arrival+8, length-1)] += .2
        return wave


def _backend_factory(_plan):
    return _GeometryBackend()


def _cfg():
    return SimulationConfigs.from_dict({
        "rir_backend": "online", "spatial_enabled": True,
        "array_profile": "linear_2ch_80mm", "reference_channel": 0,
        "gpurir_rt60_range_sec": [.2, .2],
        "gpurir_room_dim_range_m": [[4., 4.], [4., 4.], [3., 3.]],
        "gpurir_source_distance_range_m": [.7, 1.2],
        "gpurir_device": 0, "gpurir_timeout_sec": 60.,
    })


def _plan(cfg, provider, moving):
    return prepare_scene_plan(
        provider=provider, task="aec", sample_index=0, sample_rate=8000,
        num_samples=128, source_uids=("nearend", "loudspeaker"),
        spatial_view=project_simulation_view(cfg, "aec").shared.spatial,
        simulation_cfg=cfg, rng=np.random.default_rng(14), moving=moving,
        roles=("target", "loudspeaker"), stationary_actor_uids=("loudspeaker",),
        trajectory_points=3,
    )


def _render(provider, plan, moving):
    source = np.random.default_rng(1).normal(0, .01, (2, plan.num_samples)).astype(np.float32)
    return SpatialSceneRenderer().render(SpatialSceneRequest(
        sources=source, sample_rate=plan.sample_rate, provider=provider,
        source_uids=("nearend", "loudspeaker"), mode="moving" if moving else "static",
        scene_plan=plan, track_kinds=("full", "direct", "early"),
        num_samples=plan.num_samples, early_boundary=40, reference_channel=0,
    ))


@pytest.mark.unit
@pytest.mark.parametrize("moving", [False, True])
def test_provider_socket_roundtrip(tmp_path, moving):
    cfg = _cfg()
    with _server(tmp_path, _backend_factory) as client:
        provider = OnlineRIRProvider(OnlineRIRConfig.from_simulation(cfg), client)
        plan = _plan(cfg, provider, moving)
        plan = replace(plan, room={**plan.room, "rir_num_samples": 5000})
        before = plan.to_dict()
        result = _render(provider, plan, moving)
        assert plan.to_dict() == before
        assert result.full_tracks.shape == (2, 2, 128)
        assert result.observation.shape == (2, 128)
        assert result.provenance["scene_uid"] == plan.identity["scene_uid"]
        assert result.provenance["actor_order"] == ["nearend", "loudspeaker"]
        np.testing.assert_allclose(result.observation, result.full_tracks.sum(axis=0))
        if moving:
            assert result.timebase.num_samples == 128
            np.testing.assert_array_equal(result.trajectory.positions_m[:, 1],
                                          np.repeat(result.trajectory.positions_m[:1, 1], 3, axis=0))
        second = _render(provider, plan, moving)
        np.testing.assert_array_equal(result.full_tracks, second.full_tracks)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("moving", [False, True])
def test_real_cuda_scene_provider(moving):
    """不替换 provider、socket、spawn worker 或 gpuRIR，核对直达传播延迟。"""
    cfg = _cfg()
    provider = OnlineRIRProvider.from_simulation(cfg)
    try:
        plan = _plan(cfg, provider, moving)
        result = _render(provider, plan, moving)
        assert result.full_tracks.shape == (2, 2, 128)
        assert np.isfinite(result.observation).all()
        if moving:
            from core.data.simulation.rir_provider import MovingSceneRIRRequest
            request = MovingSceneRIRRequest(
                sample_rate=8000, num_samples=128, room=plan.to_dict()["room"],
                array={"positions_m": np.asarray(plan.array.positions_m).tolist(), "reference_channel": 0},
                actor_trajectories={a.actor_uid: a.trajectory for a in plan.actors},
                timebase=plan.timebase, source_uids=("nearend", "loudspeaker"),
                scene_uid=plan.identity["scene_uid"],
            )
            tracks = provider.sample_moving_scene_tracks(request)
            direct = tracks.direct[0]
            assert tracks.full.shape == (3, 2, 2, plan.room["rir_num_samples"])
            np.testing.assert_allclose(tracks.direct[:, 1], np.repeat(tracks.direct[:1, 1], 3, axis=0), atol=1e-7)
        else:
            tracks = provider.sample_scene_tracks(plan, track_kinds=("full", "direct"))
            direct = tracks.direct.canonical_rirs
        distance = np.linalg.norm(np.asarray([a.position_m for a in plan.actors])[:, None]
                                  - np.asarray(plan.array.positions_m)[None], axis=-1)
        np.testing.assert_allclose(abs(direct).argmax(axis=-1), distance / cfg.gpurir_sound_speed_mps * 8000,
                                   atol=2, rtol=0)
    finally:
        close_online_rir_services()


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("moving", [False, True])
def test_real_aec_dataset_scene(tmp_path, moving):
    """真实录制 pair fixture 经新 sampler/IPC/gpuRIR 完成 AEC observation。"""
    from test_aec_spatial_dynamic_branches import _spatial_aec_dataset
    dataset = _spatial_aec_dataset(tmp_path)
    for name in ("rir_backend", "gpurir_rt60_range_sec", "gpurir_room_dim_range_m",
                 "gpurir_source_distance_range_m", "gpurir_device", "gpurir_timeout_sec"):
        setattr(dataset.sim_cfg, name, getattr(_cfg(), name))
    dataset.sim_cfg.prob_moving_rir = float(moving)
    provider = OnlineRIRProvider.from_simulation(dataset.sim_cfg)
    dataset._spatial_rir_provider = dataset.spatial_rir_provider = provider
    try:
        result = dataset.get_sample_with_trace(0, force_branch="prob_target_noise_echo", capture_stems=True)
        nearend, mic, farend = result.sample[:3]
        assert nearend.ndim == farend.ndim == 1
        assert mic.shape == (2, len(nearend))
        mic_array = np.asarray(mic)
        assert np.isfinite(mic_array).all() and np.any(mic_array != 0)
    finally:
        close_online_rir_services()
