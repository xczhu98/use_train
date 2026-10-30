"""离线 spatial moving scene（缺口 2）全链路单元测试。

覆盖：schema v3 manifest 校验、moving bank 加载、OfflineRIRProvider
moving 读取（hash/轨迹/timebase 复核）、prepare_scene_plan offline moving
plan、SpatialSceneRenderer moving 渲染端到端。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.data.simulation.scene_manifest import (
    OfflineSceneManifest,
    trajectory_digest,
)
from core.data.simulation.rir_provider import OfflineRIRProvider
from core.data.simulation.scene_sampler import prepare_scene_plan
from core.data.simulation.spatial_scene import (
    SpatialSceneRenderer,
    SpatialSceneRequest,
)
from core.data.spatial import load_offline_moving_rir_bank_with_metadata
from core.data.simulation_configs import SimulationConfigs


pytestmark = [pytest.mark.unit]

FS = 8000
FRAMES = 3
HOP = 100
ACTORS = ("target", "interference")
CHANNELS = ((1.0, 1.0, 1.5), (1.08, 1.0, 1.5))
ROOM = (4.0, 4.0, 3.0)
RIR_LENGTH = 32
NUM_SAMPLES = FRAMES * HOP + 50
GEOMETRY = {
    "positions_m": [list(row) for row in CHANNELS],
    "reference_channel": 0,
    "channel_order": ["mic_0", "mic_1"],
    "coordinate_system": "cartesian_xyz_m",
}


def _trajectories() -> dict:
    return {
        "target": [
            [[1.2, 1.3, 1.5]],
            [[1.3, 1.35, 1.5]],
            [[1.4, 1.4, 1.5]],
        ],
        "interference": [
            [[2.2, 2.3, 1.4]],
            [[2.25, 2.3, 1.4]],
            [[2.3, 2.3, 1.4]],
        ],
    }


def _timebase() -> dict:
    return {
        "sample_rate": FS,
        "num_samples": NUM_SAMPLES,
        "hop_samples": HOP,
        "frame_count": FRAMES,
        "start_sample": 0,
    }


def _write_moving_scene(tmp_path: Path) -> Path:
    rng = np.random.default_rng(19)
    bundle_dir = tmp_path / "scene_bundles"
    bundle_dir.mkdir()
    bundle_path = bundle_dir / "scene-test.npz"
    banks = {
        name: rng.standard_normal((FRAMES, len(ACTORS), len(CHANNELS), RIR_LENGTH)).astype(np.float32)
        for name in ("full", "direct", "early")
    }
    np.savez_compressed(bundle_path, **banks)
    trajectories = _trajectories()
    identities = {
        uid: {
            track: hashlib.sha256(
                np.ascontiguousarray(banks[track][:, actor_index], dtype="<f4").tobytes()
            ).hexdigest()
            for track in ("full", "direct", "early")
        }
        for actor_index, uid in enumerate(ACTORS)
    }
    manifest_dir = tmp_path / "scene_manifests"
    manifest_dir.mkdir()
    record = {
        "entries": {
            uid: {
                "bundle_path": "../scene_bundles/scene-test.npz",
                "actor_index": actor_index,
                "full": "full",
                "direct": "direct",
                "early": "early",
            }
            for actor_index, uid in enumerate(ACTORS)
        },
        "metadata": {
            "schema_version": 3,
            "moving": True,
            "scene_uid": "scene-test",
            "sample_rate": FS,
            "actor_order": list(ACTORS),
            "actor_roles": list(ACTORS),
            "actor_positions_m": {uid: trajectories[uid][0][0] for uid in ACTORS},
            "actor_trajectories": trajectories,
            "trajectory_timebase": _timebase(),
            "trajectory_sha256": trajectory_digest(trajectories, _timebase()),
            "room": {
                "room_uid": "room-test",
                "room_dim_m": list(ROOM),
                "rt60": 0.3,
                "sound_speed": 340.0,
                "rir_num_samples": RIR_LENGTH,
            },
            "geometry": dict(GEOMETRY),
            "generator_version": "se.scene-rir-v3.0",
            "track_identities": identities,
            "track_shapes": [FRAMES, len(ACTORS), len(CHANNELS), RIR_LENGTH],
        },
    }
    manifest_path = manifest_dir / "scene-test.json"
    manifest_path.write_text(json.dumps(record), encoding="utf-8")
    return manifest_path


@pytest.fixture()
def moving_manifest(tmp_path):
    return _write_moving_scene(tmp_path)


def _provider(manifest_path) -> OfflineRIRProvider:
    return OfflineRIRProvider(str(manifest_path), geometry=GEOMETRY)


def _moving_request(plan):
    from core.data.simulation.rir_provider import MovingSceneRIRRequest

    return MovingSceneRIRRequest(
        sample_rate=FS,
        num_samples=NUM_SAMPLES,
        actor_trajectories={actor.actor_uid: actor.trajectory for actor in plan.actors},
        room=plan.to_dict()["room"],
        array={
            "positions_m": np.asarray(plan.array.positions_m).tolist(),
            "reference_channel": plan.array.reference_channel,
        },
        timebase=plan.timebase,
        source_uids=tuple(actor.actor_uid for actor in plan.actors),
        scene_uid=plan.identity["scene_uid"],
        actor_roles=tuple(actor.role for actor in plan.actors),
    )


def _manual_moving_request() -> "MovingSceneRIRRequest":
    """不依赖 plan 构造 moving 请求（用于 static manifest 拒绝路径）。"""
    from core.data.simulation.rir_provider import MovingSceneRIRRequest
    from core.data.simulation.trajectory import SourceTrajectory, TrajectoryTimebase

    clock = TrajectoryTimebase(FS, NUM_SAMPLES, HOP, FRAMES, 0)
    trajectories = {
        uid: SourceTrajectory(np.asarray(points, dtype=np.float32), clock)
        for uid, points in _trajectories().items()
    }
    return MovingSceneRIRRequest(
        sample_rate=FS,
        num_samples=NUM_SAMPLES,
        actor_trajectories=trajectories,
        room={
            "room_dim_m": list(ROOM),
            "rt60": 0.3,
            "sound_speed": 340.0,
            "rir_num_samples": RIR_LENGTH,
        },
        array={"positions_m": [list(row) for row in CHANNELS], "reference_channel": 0},
        timebase=clock,
        source_uids=ACTORS,
        scene_uid="scene-test",
        actor_roles=ACTORS,
    )


def _offline_cfg() -> SimulationConfigs:
    return SimulationConfigs.from_dict(
        {
            "rir_backend": "offline",
            "spatial_enabled": True,
            "custom_geometry": {
                "positions_m": [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]
            },
            "reference_channel": 0,
        }
    )


def _spatial_view():
    return SimpleNamespace(
        microphone_positions=np.asarray([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]),
        reference_channel=0,
    )


def test_manifest_load_moving_ok(moving_manifest):
    manifest = OfflineSceneManifest.load(moving_manifest, require_typed=True)
    assert manifest.metadata["moving"] is True
    assert manifest.metadata["trajectory_sha256"] == trajectory_digest(
        _trajectories(), _timebase()
    )


def test_manifest_load_moving_missing_fields(tmp_path):
    path = _write_moving_scene(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["metadata"]["actor_trajectories"]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        OfflineSceneManifest.load(path, require_typed=True)


def test_manifest_load_moving_tampered_trajectory(tmp_path):
    path = _write_moving_scene(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["metadata"]["actor_trajectories"]["target"][1][0][0] += 0.25
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="trajectory_sha256 does not match"):
        OfflineSceneManifest.load(path, require_typed=True)


def test_manifest_load_moving_non_bool_rejected(tmp_path):
    path = _write_moving_scene(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["metadata"]["moving"] = "true"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata.moving must be a bool"):
        OfflineSceneManifest.load(path, require_typed=True)


def test_manifest_load_moving_trajectory_out_of_room(tmp_path):
    path = _write_moving_scene(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["metadata"]["actor_trajectories"]["target"][-1] = [[9.9, 9.9, 9.9]]
    record["metadata"]["trajectory_sha256"] = trajectory_digest(
        record["metadata"]["actor_trajectories"], record["metadata"]["trajectory_timebase"]
    )
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="leaves the scene room"):
        OfflineSceneManifest.load(path, require_typed=True)


def test_moving_bank_loader_roundtrip(moving_manifest):
    bank, metadata = load_offline_moving_rir_bank_with_metadata(
        moving_manifest, ACTORS, geometry=GEOMETRY, track="full"
    )
    assert bank.shape == (FRAMES, len(ACTORS), len(CHANNELS), RIR_LENGTH)
    assert metadata["moving"] is True
    expected = np.load(moving_manifest.parent.parent / "scene_bundles" / "scene-test.npz")["full"]
    np.testing.assert_array_equal(bank, expected)


def test_offline_provider_moving_tracks_roundtrip(moving_manifest):
    provider = _provider(moving_manifest)
    plan = prepare_scene_plan(
        provider=provider,
        task="se",
        sample_index=0,
        sample_rate=FS,
        num_samples=NUM_SAMPLES,
        source_uids=ACTORS,
        spatial_view=_spatial_view(),
        simulation_cfg=_offline_cfg(),
        rng=np.random.default_rng(0),
    )
    assert plan.timebase is not None
    tracks = provider.sample_moving_scene_tracks(_moving_request(plan), normalize=False)
    assert tracks.full.shape == (FRAMES, len(ACTORS), len(CHANNELS), RIR_LENGTH)
    assert tracks.direct.shape == tracks.full.shape
    expected = np.load(moving_manifest.parent.parent / "scene_bundles" / "scene-test.npz")["full"]
    np.testing.assert_array_equal(tracks.full, expected)
    assert tracks.track_facts["full"]["group_normalization_scale"] == 1.0


def test_offline_provider_moving_hash_mismatch(moving_manifest):
    bundle = moving_manifest.parent.parent / "scene_bundles" / "scene-test.npz"
    banks = dict(np.load(bundle))
    banks["full"] = banks["full"].copy()
    banks["full"][0, 0, 0, 0] += 1.0
    np.savez_compressed(bundle, **banks)
    provider = _provider(moving_manifest)
    plan = prepare_scene_plan(
        provider=provider,
        task="se",
        sample_index=0,
        sample_rate=FS,
        num_samples=NUM_SAMPLES,
        source_uids=ACTORS,
        spatial_view=_spatial_view(),
        simulation_cfg=_offline_cfg(),
        rng=np.random.default_rng(0),
    )
    with pytest.raises(ValueError, match="content hash mismatch"):
        provider.sample_moving_scene_tracks(_moving_request(plan))


def test_scene_generator_moving_requires_motion_enabled(tmp_path):
    """--moving 要求生成器配置 motion.enabled=true，否则在 GPU 导入前 fail-fast。"""
    import shutil

    from tools.prepare_data_tools.generators.se.generate_scene_rirs import (
        generate_scene_dataset,
    )

    src = Path("tools/prepare_data_tools/generators/se/rir_gpurir.yaml")
    config_path = tmp_path / "rir_gpurir.yaml"
    shutil.copy(src, config_path)
    with pytest.raises(ValueError, match="motion.enabled=true"):
        generate_scene_dataset(
            config_path=config_path,
            output_dir=tmp_path / "out",
            num_scenes=1,
            sample_rate=16000,
            moving=True,
        )


def test_scene_generator_resume_rejects_invalid_scenes_manifest(tmp_path, monkeypatch):
    """--resume 只接受可解析且 generator_version 一致的 scenes.jsonl，不静默跳过。"""
    import shutil

    from tools.prepare_data_tools.generators.se import generate_scene_rirs
    from tools.prepare_data_tools.generators.se.rir_backends import gpurir as gpurir_backend

    repo_root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "rir_gpurir.yaml"
    shutil.copy(
        repo_root / "tools/prepare_data_tools/generators/se/rir_gpurir.yaml", config_path
    )
    monkeypatch.setattr(
        generate_scene_rirs,
        "resolve_spatial_array",
        lambda *_args, **_kwargs: SimpleNamespace(channels=2),
    )
    monkeypatch.setattr(gpurir_backend.GPURIRBackend, "_ensure_import", lambda self: None)

    output = tmp_path / "out"
    output.mkdir()
    scenes = output / "scenes.jsonl"

    def _generate() -> Path:
        return generate_scene_rirs.generate_scene_dataset(
            config_path=config_path,
            output_dir=output,
            num_scenes=1,
            sample_rate=16000,
            resume=True,
        )

    scenes.write_text(
        json.dumps({
            "metadata": {
                "scene_uid": "a",
                "generator_version": generate_scene_rirs.GENERATOR_VERSION,
            }
        })
        + "\n{truncated\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="无法解析"):
        _generate()

    scenes.write_text(
        json.dumps({"metadata": {"scene_uid": "a", "generator_version": "se.scene-rir-v1.0"}})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generator_version mismatch"):
        _generate()


def test_offline_provider_moving_supports_actor_subset(moving_manifest):
    """与静态路径一致：请求 actor 是 manifest actor_order 的唯一子集即可。"""
    provider = _provider(moving_manifest)
    plan = prepare_scene_plan(
        provider=provider,
        task="se",
        sample_index=0,
        sample_rate=FS,
        num_samples=NUM_SAMPLES,
        source_uids=("target",),
        spatial_view=_spatial_view(),
        simulation_cfg=_offline_cfg(),
        rng=np.random.default_rng(0),
    )
    assert plan.timebase is not None
    tracks = provider.sample_moving_scene_tracks(_moving_request(plan), normalize=False)
    assert tracks.full.shape == (FRAMES, 1, len(CHANNELS), RIR_LENGTH)
    expected = np.load(moving_manifest.parent.parent / "scene_bundles" / "scene-test.npz")["full"]
    np.testing.assert_array_equal(tracks.full[:, :1], expected[:, :1])


def test_offline_provider_moving_rejects_static_manifest(moving_manifest):
    record = json.loads(moving_manifest.read_text(encoding="utf-8"))
    record["metadata"]["moving"] = False
    del record["metadata"]["actor_trajectories"]
    del record["metadata"]["trajectory_timebase"]
    del record["metadata"]["trajectory_sha256"]
    record["metadata"]["schema_version"] = 2
    moving_manifest.write_text(json.dumps(record), encoding="utf-8")
    provider = _provider(moving_manifest)
    with pytest.raises(ValueError, match="not a moving scene"):
        provider.sample_moving_scene_tracks(_manual_moving_request())


def test_prepare_scene_plan_offline_moving_attaches_trajectories(moving_manifest):
    provider = _provider(moving_manifest)
    plan = prepare_scene_plan(
        provider=provider,
        task="se",
        sample_index=0,
        sample_rate=FS,
        num_samples=NUM_SAMPLES,
        source_uids=ACTORS,
        spatial_view=_spatial_view(),
        simulation_cfg=_offline_cfg(),
        rng=np.random.default_rng(0),
    )
    assert plan.timebase is not None
    assert plan.timebase.frame_count == FRAMES
    for actor in plan.actors:
        assert actor.trajectory is not None
        assert actor.trajectory.positions_m.shape == (FRAMES, 1, 3)
    # scene_uid 保持 manifest 值，provider 才能按 scene_uid 解析。
    assert plan.identity["scene_uid"] == "scene-test"
    assert "/moving/" in plan.identity["scene_instance_uid"]


def test_prepare_scene_plan_offline_moving_rejects_short_waveform(moving_manifest):
    provider = _provider(moving_manifest)
    with pytest.raises(ValueError, match="exceeds requested waveform length"):
        prepare_scene_plan(
            provider=provider,
            task="se",
            sample_index=0,
            sample_rate=FS,
            num_samples=(FRAMES - 1) * HOP,  # 最后一个轨迹帧起点越界
            source_uids=ACTORS,
            spatial_view=_spatial_view(),
            simulation_cfg=_offline_cfg(),
            rng=np.random.default_rng(0),
        )


def test_spatial_scene_renderer_offline_moving(moving_manifest):
    provider = _provider(moving_manifest)
    cfg = _offline_cfg()
    plan = prepare_scene_plan(
        provider=provider,
        task="se",
        sample_index=0,
        sample_rate=FS,
        num_samples=NUM_SAMPLES,
        source_uids=ACTORS,
        spatial_view=_spatial_view(),
        simulation_cfg=cfg,
        rng=np.random.default_rng(0),
    )
    sources = np.stack(
        [
            np.sin(np.linspace(0, 8 * np.pi, NUM_SAMPLES)).astype(np.float32),
            np.cos(np.linspace(0, 6 * np.pi, NUM_SAMPLES)).astype(np.float32) * 0.5,
        ]
    )
    rendered = SpatialSceneRenderer().render(
        SpatialSceneRequest(
            sources=sources,
            sample_rate=FS,
            provider=provider,
            source_uids=ACTORS,
            mode="moving" if plan.timebase is not None else "static",
            track_kinds=("full", "direct", "early"),
            early_boundary=max(0, int(round(0.05 * FS))),
            reference_channel=0,
            rng=np.random.default_rng(1),
            num_samples=NUM_SAMPLES,
            scene_plan=plan,
        )
    )
    assert rendered.full_tracks.shape == (len(ACTORS), len(CHANNELS), NUM_SAMPLES)
    assert rendered.early_tracks.shape == rendered.full_tracks.shape
    assert rendered.direct_tracks is not None
    assert rendered.mixture.shape == (len(CHANNELS), NUM_SAMPLES)
    assert rendered.observation.shape == (len(CHANNELS), NUM_SAMPLES)
    # 轨迹 timebase 原样保留，供 SSL DOA 投影等下游消费。
    assert rendered.trajectory is not None
    assert rendered.timebase == plan.timebase
    assert np.allclose(rendered.trajectory.positions_m[0, 0], plan.actors[0].position_m)
