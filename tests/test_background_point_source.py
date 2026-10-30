"""背景点源 RIR 解析 seam（background_resolver）与 mix_background 点源分支的合同测试。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.data.acoustic_scene_plan import (
    AcousticActorPlan,
    AcousticScenePlan,
    NoisePlan,
    RIRPlan,
    SourceGainPlan,
)
from core.data.simulation.array_profiles import resolve_microphone_array
from core.data.simulation.background import mix_background
from core.data.simulation.background_resolver import (
    BACKGROUND_NOISE_ACTOR_ROLE,
    BACKGROUND_NOISE_ACTOR_UID,
    build_background_noise_plan,
    resolve_background_spec,
    resolve_point_source_provider,
    sample_background_noise_actor,
)
from core.data.simulation.point_source import (
    MovingPointSourceRIRProvider,
    PointSourceRIRProvider,
)
from core.data.simulation.role_policy import BackgroundPolicy, BackgroundSpec
from core.data.simulation.rir_provider import (
    MovingSpatialRIRTrackSet,
    SpatialRIRResult,
)
from core.data.simulation.trajectory import (
    TrajectoryTimebase,
    plan_linear_source_trajectory,
)
from core.data.simulation_configs import SimulationConfigs

pytestmark = [pytest.mark.unit]

_MIC_POSITIONS = [[1.0, 1.0, 1.5], [1.1, 1.0, 1.5]]
_ROOM = {
    "room_dim_m": [4.0, 5.0, 3.0],
    "rt60": 0.3,
    "sound_speed": 340.0,
    "coordinate_system": "cartesian_xyz_m",
    "diffuse_db": 15.0,
    "rir_num_samples": 512,
    "room_uid": "room-test",
}
_SCENE_UID = "se/sample/0/group/online-test"


def _make_cfg(**overrides) -> SimulationConfigs:
    values = {
        "rir_backend": "online",
        "spatial_enabled": True,
        "custom_geometry": {"positions_m": _MIC_POSITIONS},
        "reference_channel": 0,
        "prob_point_source_background": 1.0,
    }
    values.update(overrides)
    return SimulationConfigs.from_dict(values, strict=True)


def _make_scene_plan(*, moving=False, num_samples=8000) -> AcousticScenePlan:
    array = resolve_microphone_array(
        custom_geometry={"positions_m": _MIC_POSITIONS},
        reference_channel=0,
    )
    timebase = None
    actor = AcousticActorPlan(
        actor_uid="speech/0", role="target", position_m=(2.0, 2.5, 1.5)
    )
    if moving:
        timebase = TrajectoryTimebase(16000, num_samples, num_samples // 4, 4)
        trajectory = plan_linear_source_trajectory(
            np.array([[2.0, 2.5, 1.5]], dtype=np.float64),
            np.array([[2.2, 2.6, 1.5]], dtype=np.float64),
            timebase,
        )
        actor = AcousticActorPlan(
            actor_uid="speech/0", role="target", trajectory=trajectory
        )
    return AcousticScenePlan(
        sample_rate=16000,
        num_samples=num_samples,
        need_noise=False,
        rir=RIRPlan.disabled(),
        source_gain=SourceGainPlan.disabled(),
        noise=NoisePlan.disabled(),
        identity={
            "scene_uid": _SCENE_UID,
            "scene_instance_uid": f"{_SCENE_UID}/instance/abc",
        },
        room=_ROOM,
        array=array,
        timebase=timebase,
        actors=(actor,),
    )


class _FakeSceneProvider:
    """最小 scene provider：按 from_scene_plan 合同回显 plan 身份。"""

    def __init__(self, scene_plan, *, rir_length=64, moving=False):
        self._rir_length = rir_length
        self._moving = moving
        self.static_calls = 0
        self.moving_calls = 0

    def sample_scene_tracks(self, scene_plan, *, normalize=True, track_kinds=("full",)):
        self.static_calls += 1
        channels = len(scene_plan.array.positions_m)
        rng = np.random.default_rng(1234)
        rirs = (rng.normal(size=(1, channels, self._rir_length)) * 0.1).astype(np.float32)
        provenance = {
            "scene_uid": scene_plan.identity["scene_uid"],
            "actor_order": [actor.actor_uid for actor in scene_plan.actors],
            "source_order": [actor.actor_uid for actor in scene_plan.actors],
            "room": json.loads(json.dumps(scene_plan.to_dict()["room"])),
        }
        return SpatialRIRResult(
            rirs,
            {"positions_m": [list(p) for p in scene_plan.array.positions_m]},
            provenance,
            "fake-scene-v1",
        )

    def sample_moving_scene_tracks(self, request, *, normalize=True):
        self.moving_calls += 1
        frame_count = request.timebase.frame_count
        channels = len(request.array["positions_m"])
        rng = np.random.default_rng(5678)
        full = (
            rng.normal(size=(frame_count, 1, channels, self._rir_length)) * 0.1
        ).astype(np.float32)
        trajectory = np.concatenate(
            [traj.positions_m for traj in request.actor_trajectories.values()], axis=1
        )
        plan = {
            "scene_uid": request.scene_uid,
            "group_uid": request.scene_uid,
            "sample_rate": request.sample_rate,
            "num_samples": self._rir_length,
            "waveform_num_samples": request.timebase.num_samples,
            "room_dim": list(request.room["room_dim_m"]),
            "rt60": request.room["rt60"],
            "sound_speed": request.room["sound_speed"],
            "mic_loc": [list(p) for p in request.array["positions_m"]],
            "array": dict(request.array),
            "room": dict(request.room),
            "source_loc": trajectory[0].tolist(),
            "source_ids": list(request.source_uids),
            "trajectory": trajectory.tolist(),
            "trajectory_timebase": request.timebase.to_dict(),
            "actor_trajectories": {
                uid: traj.positions_m.tolist()
                for uid, traj in request.actor_trajectories.items()
            },
        }
        return MovingSpatialRIRTrackSet(
            full, full.copy(), plan, {"full": {"generator_version": "fake-moving-v1"}}
        )


# ---------------------------------------------------------------------------
# resolve_background_spec 门控与 rng 纪律
# ---------------------------------------------------------------------------


def test_gate_closed_keeps_default_policy_and_consumes_zero_rng():
    cfg = _make_cfg(prob_point_source_background=0.0)
    scene_plan = _make_scene_plan()
    rng = np.random.default_rng(7)
    spec = resolve_background_spec(
        cfg=cfg,
        scene_plan=scene_plan,
        provider=_FakeSceneProvider(scene_plan),
        rng=rng,
        sample_rate=16000,
        image_geometry=_MIC_POSITIONS,
        shared_scene=True,
        enabled=True,
    )
    assert spec.policy == BackgroundPolicy.DIFFUSE_COHERENCE
    control = np.random.default_rng(7)
    assert rng.random() == control.random()


def test_typed_background_view_is_consumed_for_point_source_gate():
    cfg = _make_cfg(prob_point_source_background=1.0)
    scene_plan = _make_scene_plan()
    rng = np.random.default_rng(17)
    view = type("BackgroundView", (), {"prob_point_source_background": 0.0})()
    spec = resolve_background_spec(
        cfg=cfg,
        background_view=view,
        scene_plan=scene_plan,
        provider=_FakeSceneProvider(scene_plan),
        rng=rng,
        sample_rate=16000,
        image_geometry=_MIC_POSITIONS,
        shared_scene=True,
        enabled=True,
    )
    assert spec.policy == BackgroundPolicy.DIFFUSE_COHERENCE
    control = np.random.default_rng(17)
    assert rng.random() == control.random()


def test_gate_closed_mono_observation_keeps_additive_default():
    cfg = _make_cfg(prob_point_source_background=1.0)
    scene_plan = _make_scene_plan()
    spec = resolve_background_spec(
        cfg=cfg,
        scene_plan=scene_plan,
        provider=_FakeSceneProvider(scene_plan),
        rng=np.random.default_rng(7),
        sample_rate=16000,
        image_geometry=None,
        enabled=True,
    )
    assert spec.policy == BackgroundPolicy.ADDITIVE_MONO


def test_disabled_sample_keeps_default_policy_and_consumes_zero_rng():
    cfg = _make_cfg(prob_point_source_background=1.0)
    scene_plan = _make_scene_plan()
    rng = np.random.default_rng(11)
    spec = resolve_background_spec(
        cfg=cfg,
        scene_plan=scene_plan,
        provider=_FakeSceneProvider(scene_plan),
        rng=rng,
        sample_rate=16000,
        image_geometry=_MIC_POSITIONS,
        enabled=False,
    )
    assert spec.policy == BackgroundPolicy.DIFFUSE_COHERENCE
    control = np.random.default_rng(11)
    assert rng.random() == control.random()


def test_gate_miss_consumes_exactly_one_draw():
    # seed 0 的首抽约 0.637 >= 0.5，门确定未命中。
    cfg = _make_cfg(prob_point_source_background=0.5)
    scene_plan = _make_scene_plan()
    rng = np.random.default_rng(0)
    spec = resolve_background_spec(
        cfg=cfg,
        scene_plan=scene_plan,
        provider=_FakeSceneProvider(scene_plan),
        rng=rng,
        sample_rate=16000,
        image_geometry=_MIC_POSITIONS,
        enabled=True,
    )
    assert spec.policy == BackgroundPolicy.DIFFUSE_COHERENCE
    control = np.random.default_rng(0)
    control.random()
    assert rng.random() == control.random()


def test_gate_open_builds_static_point_source_spec():
    cfg = _make_cfg(prob_point_source_background=1.0)
    scene_plan = _make_scene_plan()
    fake = _FakeSceneProvider(scene_plan)
    spec = resolve_background_spec(
        cfg=cfg,
        scene_plan=scene_plan,
        provider=fake,
        rng=np.random.default_rng(5),
        sample_rate=16000,
        image_geometry=_MIC_POSITIONS,
        shared_scene=True,
        enabled=True,
    )
    assert spec.policy == BackgroundPolicy.POINT_SOURCE_RIR
    assert spec.shared_scene is True
    assert isinstance(spec.point_source_provider, PointSourceRIRProvider)
    assert fake.static_calls == 1
    assert fake.moving_calls == 0
    provider = spec.point_source_provider
    assert provider.sample_rate == 16000
    assert provider.source_uid == BACKGROUND_NOISE_ACTOR_UID
    np.testing.assert_allclose(
        np.asarray(provider.geometry["positions_m"]), np.asarray(_MIC_POSITIONS)
    )


def test_gate_open_builds_moving_point_source_spec():
    cfg = _make_cfg(prob_point_source_background=1.0)
    scene_plan = _make_scene_plan(moving=True)
    fake = _FakeSceneProvider(scene_plan, moving=True)
    spec = resolve_background_spec(
        cfg=cfg,
        scene_plan=scene_plan,
        provider=fake,
        rng=np.random.default_rng(5),
        sample_rate=16000,
        image_geometry=_MIC_POSITIONS,
        shared_scene=True,
        enabled=True,
    )
    assert spec.policy == BackgroundPolicy.POINT_SOURCE_RIR
    assert isinstance(spec.point_source_provider, MovingPointSourceRIRProvider)
    assert fake.moving_calls == 1
    assert fake.static_calls == 0


def test_gate_open_requires_scene_provider():
    cfg = _make_cfg(prob_point_source_background=1.0)
    scene_plan = _make_scene_plan()
    with pytest.raises(TypeError, match="sample_scene_tracks"):
        resolve_background_spec(
            cfg=cfg,
            scene_plan=scene_plan,
            provider=object(),
            rng=np.random.default_rng(5),
            sample_rate=16000,
            image_geometry=_MIC_POSITIONS,
            enabled=True,
        )


# ---------------------------------------------------------------------------
# 噪声 actor 采样与派生 plan
# ---------------------------------------------------------------------------


def test_noise_actor_is_inside_room_and_separated_from_speech():
    cfg = _make_cfg()
    scene_plan = _make_scene_plan()
    actor = sample_background_noise_actor(
        scene_plan=scene_plan, cfg=cfg, rng=np.random.default_rng(17)
    )
    assert actor.actor_uid == BACKGROUND_NOISE_ACTOR_UID
    assert actor.role == BACKGROUND_NOISE_ACTOR_ROLE
    position = np.asarray(actor.position_m, dtype=np.float64)
    room = np.asarray(_ROOM["room_dim_m"], dtype=np.float64)
    assert np.all(position > 0.0) and np.all(position < room)
    speech = np.asarray([2.0, 2.5, 1.5])
    assert np.linalg.norm(position - speech) >= 0.3 - 1e-9


def test_noise_actor_replay_is_deterministic():
    cfg = _make_cfg()
    scene_plan = _make_scene_plan()
    first = sample_background_noise_actor(
        scene_plan=scene_plan, cfg=cfg, rng=np.random.default_rng(23)
    )
    second = sample_background_noise_actor(
        scene_plan=scene_plan, cfg=cfg, rng=np.random.default_rng(23)
    )
    np.testing.assert_allclose(first.position_m, second.position_m)


def test_moving_noise_actor_shares_scene_timebase():
    cfg = _make_cfg()
    scene_plan = _make_scene_plan(moving=True)
    actor = sample_background_noise_actor(
        scene_plan=scene_plan, cfg=cfg, rng=np.random.default_rng(29)
    )
    assert actor.trajectory is not None
    assert actor.trajectory.timebase == scene_plan.timebase
    assert actor.position_m is None
    room = np.asarray(_ROOM["room_dim_m"], dtype=np.float64)
    assert np.all(actor.trajectory.positions_m > 0.0)
    assert np.all(actor.trajectory.positions_m < room)


def test_noise_plan_derives_identity_and_clones_scene():
    scene_plan = _make_scene_plan()
    cfg = _make_cfg()
    noise_actor = sample_background_noise_actor(
        scene_plan=scene_plan, cfg=cfg, rng=np.random.default_rng(31)
    )
    noise_plan = build_background_noise_plan(
        scene_plan=scene_plan, noise_actor=noise_actor
    )
    assert noise_plan.actors == (noise_actor,)
    assert noise_plan.room == scene_plan.room
    assert noise_plan.array is scene_plan.array
    assert noise_plan.timebase is scene_plan.timebase
    assert noise_plan.identity["scene_uid"].startswith(f"{_SCENE_UID}/noise/")
    assert noise_plan.identity["background_noise"] is True
    again = build_background_noise_plan(
        scene_plan=scene_plan, noise_actor=noise_actor
    )
    assert again.identity["scene_uid"] == noise_plan.identity["scene_uid"]


def test_resolve_point_source_provider_dispatches_on_timebase():
    cfg = _make_cfg()
    static_plan = _make_scene_plan()
    static_actor = sample_background_noise_actor(
        scene_plan=static_plan, cfg=cfg, rng=np.random.default_rng(37)
    )
    static_noise = build_background_noise_plan(
        scene_plan=static_plan, noise_actor=static_actor
    )
    provider = resolve_point_source_provider(
        provider=_FakeSceneProvider(static_plan),
        noise_plan=static_noise,
        sample_rate=16000,
    )
    assert isinstance(provider, PointSourceRIRProvider)

    moving_plan = _make_scene_plan(moving=True)
    moving_actor = sample_background_noise_actor(
        scene_plan=moving_plan, cfg=cfg, rng=np.random.default_rng(37)
    )
    moving_noise = build_background_noise_plan(
        scene_plan=moving_plan, noise_actor=moving_actor
    )
    provider = resolve_point_source_provider(
        provider=_FakeSceneProvider(moving_plan, moving=True),
        noise_plan=moving_noise,
        sample_rate=16000,
    )
    assert isinstance(provider, MovingPointSourceRIRProvider)


# ---------------------------------------------------------------------------
# mix_background 点源消费分支（显式注入 seam）
# ---------------------------------------------------------------------------


def test_mix_background_point_source_branch_renders_and_scales(monkeypatch):
    from core.data.dynamic_dataset_base import BaseDynamicMixingDataset

    speech = np.linspace(-0.5, 0.5, 4000, dtype=np.float32)
    speech_2d = np.stack([speech, speech * 0.5]).astype(np.float32)
    noise = np.random.default_rng(99).normal(0.0, 0.1, 4000).astype(np.float32)

    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimulationConfigs.from_dict(
        {
            "noise_mix_prob": 0.0,
            "prob_colored_noise": 0.0,
            "snr_low": 0.0,
            "snr_high": 10.0,
            "snr_reference": "all_speech",
            "reuse_noise": False,
        },
        strict=True,
    )
    dataset.export_simulation_stems = False
    monkeypatch.setattr(
        dataset,
        "_read_one_additive_noise_scp_only_with_identity",
        lambda sr, max_dur, rng: ("noise_scp:fake", noise),
    )

    rirs = np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    point_source = PointSourceRIRProvider(
        rirs,
        {"positions_m": _MIC_POSITIONS},
        16000,
        BACKGROUND_NOISE_ACTOR_UID,
    )
    result = mix_background(
        dataset,
        speech_2d.copy(),
        speech_2d,
        16000,
        speech_2d.shape[1],
        speech_2d.shape[1],
        np.random.default_rng(0),
        True,
        5.0,
        image_geometry=_MIC_POSITIONS,
        background_spec=BackgroundSpec(
            BackgroundPolicy.POINT_SOURCE_RIR,
            geometry_bound=True,
            shared_scene=True,
            point_source_provider=point_source,
        ),
    )

    assert result.observation.shape == speech_2d.shape
    assert result.observation.dtype == np.float32
    assert result.additive.shape == speech_2d.shape
    assert not np.allclose(result.additive, 0.0)
    provenance = result.noise_image_provenance
    assert provenance["background_policy"] == "point_source_rir"
    assert provenance["projection"] == "point_source_rir"
    assert provenance["rir_applied"] is True
    assert provenance["background_shared_scene"] is True


# ---------------------------------------------------------------------------
# SE 数据集端到端：online moving scene + 点源背景门打开
# ---------------------------------------------------------------------------


class _FakeSceneClient:
    """scene-level fake：按 ``generate_scene`` wire 合同返回 full/direct 与元数据。"""

    def __init__(self):
        self.requests = []

    def generate_scene(self, scene, *, moving=False, normalize=True, track_kind=None):
        self.requests.append((scene, moving))
        actors = scene["actors"]
        source_count = len(actors)
        mic = np.asarray(scene["array"]["positions_m"], dtype=np.float32)
        rir_len = int(scene["room"]["rir_num_samples"])
        rng = np.random.default_rng(abs(hash(scene["scene_uid"])) % (2 ** 32))
        if moving:
            frames = int(scene["timebase"]["frame_count"])
            shape = (frames, source_count, mic.shape[0], rir_len)
            source_loc = [
                np.asarray(actor["trajectory"], dtype=np.float32)[0, 0].tolist()
                for actor in actors
            ]
            actor_trajectories = {
                actor["actor_uid"]: actor["trajectory"] for actor in actors
            }
        else:
            shape = (source_count, mic.shape[0], rir_len)
            source_loc = [actor["position_m"] for actor in actors]
            actor_trajectories = None
        names = ("full", "direct") if track_kind and "direct" in track_kind else ("full",)
        out = {}
        for name in names:
            wave = (rng.normal(size=shape) * 0.1).astype(np.float32)
            metadata = {
                "scene_uid": scene["scene_uid"],
                "actor_order": [actor["actor_uid"] for actor in actors],
                "track_kind": name,
                "generator_version": "fake-scene-v1",
                "rir_sha256": __import__("hashlib").sha256(wave.tobytes()).hexdigest(),
                "room_dim": scene["room"]["room_dim_m"],
                "mic_loc": mic.tolist(),
                "source_loc": source_loc,
                "sample_rate": scene["sample_rate"],
            }
            if moving:
                metadata["actor_trajectories"] = actor_trajectories
                metadata["trajectory_timebase"] = scene["timebase"]
            out[name] = (wave, metadata)
        return out


@pytest.mark.parametrize("moving", [False, True])
def test_se_dataset_point_source_background_end_to_end(tmp_path, monkeypatch, moving):
    import soundfile as sf

    from core.data.simulation.rir_config import OnlineRIRConfig
    from core.data.simulation.rir_provider import OnlineRIRProvider
    from tasks.se.data.dynamic_dataset import DynamicMixingDataset

    sample_rate = 8000
    waveform = (0.05 * np.sin(np.linspace(0.0, 3.0, 256))).astype(np.float32)
    for name in ("speech", "noise"):
        sf.write(tmp_path / f"{name}.wav", waveform, sample_rate, subtype="FLOAT")
    (tmp_path / "speech.scp").write_text(
        "u 8000 256 speech.wav\n", encoding="utf-8"
    )
    (tmp_path / "noise.scp").write_text("n 8000 256 noise.wav\n", encoding="utf-8")

    client = _FakeSceneClient()

    def _from_simulation(cls, cfg):
        return OnlineRIRProvider(OnlineRIRConfig.from_simulation(cfg), client)

    monkeypatch.setattr(
        OnlineRIRProvider, "from_simulation", classmethod(_from_simulation)
    )

    dataset = DynamicMixingDataset(
        str(tmp_path / "speech.scp"),
        str(tmp_path / "noise.scp"),
        None,
        max_duration_seconds=256 / sample_rate,
        filter_by_duration=False,
        simulation_cfg={
            "rir_backend": "online",
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "prob_reverberation": 1.0,
            "prob_moving_rir": 1.0 if moving else 0.0,
            "moving_source_speed_range_mps": [0.12, 0.2],
            "prob_target_only": 0.0,
            "prob_noise_only": 0.0,
            "prob_target_noise": 1.0,
            "prob_point_source_background": 1.0,
            "snr_low": 6.0,
            "snr_high": 6.0,
            "gpurir_rt60_range_sec": [0.1, 0.2],
            "source_volume_normalization_enabled": False,
            "peak_normalization_mode": "off",
            "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": False,
        },
    )

    traced = dataset.get_sample_with_trace(0)
    target, observation = traced.sample[:2]
    assert target.shape == (256,)
    assert observation.shape == (2, 256)
    facts = traced.trace.metadata_dict()["sources"]["scene"]["scene"]
    provenance = facts["noise_image_provenance"]
    assert provenance["background_policy"] == "point_source_rir"
    if moving:
        assert provenance["moving"] is True
    else:
        assert not provenance.get("moving", False)
    assert len(client.requests) == 2
    assert all(request_moving is moving for _, request_moving in client.requests)


def test_mix_background_point_source_rejects_geometry_mismatch(monkeypatch):
    from core.data.dynamic_dataset_base import BaseDynamicMixingDataset

    speech_2d = np.ones((2, 2000), dtype=np.float32) * 0.1
    noise = np.ones(2000, dtype=np.float32) * 0.05

    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimulationConfigs.from_dict(
        {
            "noise_mix_prob": 0.0,
            "prob_colored_noise": 0.0,
            "snr_low": 0.0,
            "snr_high": 10.0,
            "reuse_noise": False,
        },
        strict=True,
    )
    dataset.export_simulation_stems = False
    monkeypatch.setattr(
        dataset,
        "_read_one_additive_noise_scp_only_with_identity",
        lambda sr, max_dur, rng: ("noise_scp:fake", noise),
    )

    point_source = PointSourceRIRProvider(
        np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32),
        {"positions_m": _MIC_POSITIONS},
        16000,
        BACKGROUND_NOISE_ACTOR_UID,
    )
    with pytest.raises(ValueError, match="geometry does not match"):
        mix_background(
            dataset,
            speech_2d.copy(),
            speech_2d,
            16000,
            speech_2d.shape[1],
            speech_2d.shape[1],
            np.random.default_rng(0),
            True,
            5.0,
            image_geometry=[[5.0, 5.0, 1.5], [5.1, 5.0, 1.5]],
            background_spec=BackgroundSpec(
                BackgroundPolicy.POINT_SOURCE_RIR,
                geometry_bound=True,
                shared_scene=True,
                point_source_provider=point_source,
            ),
        )


def _spatial_wind_dataset(monkeypatch, cfg, wind, *, export=True):
    from core.data.dynamic_dataset_base import BaseDynamicMixingDataset

    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = cfg
    dataset.export_simulation_stems = export
    dataset._acoustic_plan_record = lambda: None
    monkeypatch.setattr(
        dataset,
        "_mix_acoustic_get_wind_sample",
        lambda sr, max_dur, rng, reuse: wind.copy(),
    )
    monkeypatch.setattr(
        dataset,
        "_mix_acoustic_colored_noise",
        lambda observation, reference, additive, snrs, sr, rng, audit: (
            observation,
            additive,
        ),
    )
    return dataset


def test_spatial_wind_background_is_additive_and_preserves_provenance(monkeypatch):
    geometry = [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]]
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "custom_geometry": {"positions_m": geometry},
            "reference_channel": 0,
            "prob_wind_noise_background": 1.0,
            "prob_wind_noise_sidechain": 0.0,
            "noise_mix_prob": 0.0,
            "prob_colored_noise": 0.0,
            "snr_low": 6.0,
            "snr_high": 6.0,
        },
        strict=True,
    )
    speech = np.vstack(
        [
            np.sin(np.linspace(0.0, 14.0, 1024)),
            np.cos(np.linspace(0.0, 11.0, 1024)),
        ]
    ).astype(np.float32)
    wind = np.sin(np.linspace(0.0, 31.0, 1024)).astype(np.float32)[None, :]
    dataset = _spatial_wind_dataset(monkeypatch, cfg, wind)

    result = mix_background(
        dataset,
        speech.copy(),
        speech.copy(),
        16000,
        speech.shape[1],
        speech.shape[1],
        np.random.default_rng(42),
        True,
        6.0,
        image_geometry=geometry,
        background_spec=BackgroundSpec(
            BackgroundPolicy.DIFFUSE_COHERENCE,
            geometry_bound=True,
            shared_scene=True,
        ),
    )

    assert result.used_wind_background is True
    assert result.used_sidechain_wind is False
    assert result.additive.shape == result.observation.shape == speech.shape
    np.testing.assert_allclose(result.observation, speech + result.additive, rtol=0.0, atol=1e-6)
    assert not np.array_equal(result.additive[0], result.additive[1])
    provenance = result.noise_image_provenance
    assert provenance["output_shape"] == [2, 1024]
    assert provenance["geometry"]["microphone_positions_m"] == geometry
    assert provenance["coherence_policy"] == "habets_spherical"
    assert provenance["source_identity"].startswith("noise_scp:inline:")
    assert provenance["snr_nominal_db"] == result.wind_snr
    assert provenance["wind_background_applied"] is True
    assert provenance["wind_background"]["shape"] == [2, 1024]
    assert provenance["wind_background"]["geometry_bound"] is True


def test_spatial_wind_background_seed_replay_is_byte_stable(monkeypatch):
    geometry = [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]]
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "custom_geometry": {"positions_m": geometry},
            "reference_channel": 0,
            "prob_wind_noise_background": 1.0,
            "noise_mix_prob": 0.0,
            "prob_colored_noise": 0.0,
            "snr_low": 6.0,
            "snr_high": 6.0,
        },
        strict=True,
    )
    speech = np.ones((2, 1024), dtype=np.float32) * 0.2
    wind = np.linspace(-0.4, 0.4, 1024, dtype=np.float32)[None, :]
    spec = BackgroundSpec(
        BackgroundPolicy.DIFFUSE_COHERENCE,
        geometry_bound=True,
        shared_scene=True,
    )
    first = mix_background(
        _spatial_wind_dataset(monkeypatch, cfg, wind),
        speech.copy(), speech.copy(), 16000, 1024, 1024,
        np.random.default_rng(123), True, 6.0,
        image_geometry=geometry, background_spec=spec,
    )
    second = mix_background(
        _spatial_wind_dataset(monkeypatch, cfg, wind),
        speech.copy(), speech.copy(), 16000, 1024, 1024,
        np.random.default_rng(123), True, 6.0,
        image_geometry=geometry, background_spec=spec,
    )
    np.testing.assert_array_equal(first.observation, second.observation)
    np.testing.assert_array_equal(first.additive, second.additive)
    assert first.noise_image_provenance == second.noise_image_provenance


def test_spatial_wind_background_rejects_mismatched_multichannel_asset(monkeypatch):
    geometry = [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]]
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "custom_geometry": {"positions_m": geometry},
            "reference_channel": 0,
            "prob_wind_noise_background": 1.0,
            "noise_mix_prob": 0.0,
            "prob_colored_noise": 0.0,
        },
        strict=True,
    )
    speech = np.ones((2, 256), dtype=np.float32) * 0.1
    wind = np.ones((3, 256), dtype=np.float32) * 0.2
    with pytest.raises(ValueError, match="expected geometry C=2"):
        mix_background(
            _spatial_wind_dataset(monkeypatch, cfg, wind),
            speech.copy(), speech.copy(), 16000, 256, 256,
            np.random.default_rng(5), True, 6.0,
            image_geometry=geometry,
            background_spec=BackgroundSpec(
                BackgroundPolicy.DIFFUSE_COHERENCE,
                geometry_bound=True,
                shared_scene=True,
            ),
        )


def test_moving_point_source_wind_background_keeps_geometry_and_timebase(monkeypatch):
    cfg = _make_cfg(prob_wind_noise_background=1.0)
    scene_plan = _make_scene_plan(moving=True, num_samples=8000)
    wind = np.sin(np.linspace(0.0, 31.0, scene_plan.num_samples)).astype(np.float32)[None, :]
    dataset = _spatial_wind_dataset(monkeypatch, cfg, wind)
    from core.data.simulation.background_resolver import resolve_and_mix_background

    speech = np.ones((2, scene_plan.num_samples), dtype=np.float32) * 0.1
    background, spec = resolve_and_mix_background(
        dataset=dataset,
        speech_image=speech,
        reference_image=speech,
        sample_rate=scene_plan.sample_rate,
        actual_length=scene_plan.num_samples,
        max_duration_samples=scene_plan.num_samples,
        rng=np.random.default_rng(7),
        need_noise=True,
        snr=6.0,
        scene_plan=scene_plan,
        provider=_FakeSceneProvider(scene_plan, moving=True),
        image_geometry=_MIC_POSITIONS,
        shared_scene=True,
    )

    assert spec.policy == BackgroundPolicy.POINT_SOURCE_RIR
    assert background.used_wind_background is True
    assert background.additive.shape == speech.shape
    provenance = background.noise_image_provenance
    assert provenance["projection"] == "point_source_moving_rir"
    assert provenance["moving"] is True
    assert provenance["timebase"]["num_samples"] == scene_plan.num_samples
    assert provenance["geometry"]["positions_m"] == _MIC_POSITIONS
    assert provenance["source_identity"] == BACKGROUND_NOISE_ACTOR_UID
    assert provenance["coherence_policy"] == "point_source_rir"


def test_point_source_provider_accepts_room_absolute_microphone_positions():
    geometry = {
        "positions_m": [[12.0, 12.0, 12.0], [12.1, 12.0, 12.0]],
    }
    provider = PointSourceRIRProvider(
        np.zeros((2, 16), dtype=np.float32),
        geometry,
        sample_rate=16000,
        source_uid="noise/0",
    )
    np.testing.assert_allclose(provider.geometry["positions_m"], geometry["positions_m"])
