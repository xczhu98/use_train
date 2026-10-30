"""typed offline scene manifest 合同测试。"""
import json
import hashlib

import numpy as np
import pytest

from core.data.simulation.rir_provider import OfflineRIRProvider, build_spatial_rir_provider
from core.data.simulation.scene_manifest import OfflineSceneIndex
from tools.prepare_data_tools.generators.se.rir_output import save_scene_result

pytestmark = pytest.mark.unit


def _hash(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()


def _write_typed(tmp_path):
    values = {}
    entries = {}
    identities = {}
    for actor, scale in (("target", 1.0), ("interference", 0.5)):
        tracks = {}
        identities[actor] = {}
        for track in ("full", "direct", "early"):
            value = np.full((2, 4), scale + len(track), dtype=np.float32)
            filename = f"{actor}_{track}.npy"
            np.save(tmp_path / filename, value)
            tracks[track] = filename
            identities[actor][track] = _hash(value)
        entries[actor] = tracks
    manifest = tmp_path / "scene.json"
    manifest.write_text(json.dumps({
        "entries": entries,
        "metadata": {
            "schema_version": 1,
            "scene_uid": "scene-1",
            "actor_order": ["target", "interference"],
            "room": {"room_uid": "room-1"},
            "geometry": {"positions_m": [[0, 0, 0], [0.04, 0, 0]]},
            "generator_version": "fixture-v1",
            "track_identities": identities,
        },
    }), encoding="utf-8")
    return manifest


def test_typed_scene_manifest_loads_and_binds_identity(tmp_path):
    manifest = _write_typed(tmp_path)
    result = OfflineRIRProvider(
        manifest, geometry={"positions_m": [[0, 0, 0], [0.04, 0, 0]]}, require_scene_identity=True
    ).get_rirs(16000, 2, None, source_uids=("target", "interference"))
    assert result.provenance["scene_uid"] == "scene-1"
    assert result.provenance["actor_order"] == ["target", "interference"]
    assert result.generator_version == "fixture-v1"
    assert result.rirs.shape == (2, 2, 4)


def test_typed_scene_manifest_rejects_missing_identity(tmp_path):
    np.save(tmp_path / "a.npy", np.ones((2, 2), dtype=np.float32))
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"entries": {"a": "a.npy"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        OfflineRIRProvider(path, geometry={"positions_m": [[0, 0, 0], [0.04, 0, 0]]}, require_scene_identity=True)


def test_typed_scene_manifest_rejects_actor_order_and_geometry(tmp_path):
    manifest = _write_typed(tmp_path)
    provider = OfflineRIRProvider(manifest, geometry={"positions_m": [[0, 0, 0], [0.05, 0, 0]]}, require_scene_identity=True)
    with pytest.raises(ValueError, match="actor_order|geometry"):
        provider.get_rirs(16000, 2, None, source_uids=("interference", "target"))


def test_factory_requires_typed_scene_identity_for_new_spatial_path(tmp_path):
    np.save(tmp_path / "a.npy", np.ones((2, 2), dtype=np.float32))
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"entries": {"a": "a.npy"}}), encoding="utf-8")
    class Cfg:
        spatial_enabled = True
        rir_backend = "offline"
        array_profile = "linear_2ch_80mm"
        reference_channel = 0
    with pytest.raises(ValueError, match="identity"):
        build_spatial_rir_provider(Cfg(), rir_scp=path)


def test_scene_jsonl_index_resolves_generator_scene_manifests(tmp_path):
    shape = (2, 2, 4)
    scene = {
        "scene": {
            "scene_uid": "scene-index-0", "sample_rate": 16000, "rir_num_samples": 4,
            "room": {"room_uid": "room-index", "room_dim_m": [4, 4, 3], "rt60": .3},
            "array": {"positions_m": [[1, 1, 1], [1.04, 1, 1]], "reference_channel": 0,
                       "channel_order": ["mic_0", "mic_1"]},
            "actors": [
                {"actor_uid": "a", "role": "target", "position_m": [2, 2, 1.5]},
                {"actor_uid": "b", "role": "interference", "position_m": [3, 2, 1.5]},
            ], "generator_version": "fixture-v2",
        },
        "full": np.ones(shape, np.float32),
        "direct": np.ones(shape, np.float32) * .8,
        "early": np.ones(shape, np.float32) * .6,
    }
    from tools.prepare_data_tools.generators.se.rir_output import save_scene_result
    record = save_scene_result(scene, output_dir=tmp_path, index=0)
    index_path = tmp_path / "scenes.jsonl"
    index_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    index = OfflineSceneIndex.load(index_path)
    selected = index.select(("a", "b"), 0)
    assert selected.scene_uid == "scene-index-0"
    assert selected.metadata["sample_rate"] == 16000


def test_offline_provider_loads_generator_npz_bundle_through_scene_index(tmp_path):
    from core.data.acoustic_scene_plan import (
        AcousticActorPlan, AcousticScenePlan, NoisePlan, RIRPlan, SourceGainPlan,
    )
    from core.data.simulation.array_profiles import resolve_microphone_array
    from tools.prepare_data_tools.generators.se.rir_output import save_scene_result

    shape = (2, 2, 4)
    scene = {
        "scene": {
            "scene_uid": "scene-bundle-0", "sample_rate": 16000, "rir_num_samples": 4,
            "room": {"room_uid": "room-bundle", "room_dim_m": [4, 4, 3], "rt60": .3,
                     "sound_speed": 340.0, "diffuse_db": 15.0},
            "array": {"positions_m": [[1, 1, 1], [1.04, 1, 1]], "reference_channel": 0,
                      "channel_order": ["mic_0", "mic_1"], "coordinate_system": "cartesian_xyz_m"},
            "actors": [
                {"actor_uid": "a", "role": "target", "position_m": [2, 2, 1.5]},
                {"actor_uid": "b", "role": "interference", "position_m": [3, 2, 1.5]},
            ], "generator_version": "fixture-v2",
        },
        "full": np.ones(shape, np.float32),
        "direct": np.ones(shape, np.float32) * .8,
        "early": np.ones(shape, np.float32) * .6,
    }
    record = save_scene_result(scene, output_dir=tmp_path, index=0)
    index_path = tmp_path / "scenes.jsonl"
    index_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    plan = AcousticScenePlan(
        sample_rate=16000, num_samples=16, need_noise=False,
        rir=RIRPlan.disabled(), source_gain=SourceGainPlan.disabled(), noise=NoisePlan.disabled(),
        identity={"scene_uid": "scene-bundle-0", "sample_index": 0},
        room=scene["scene"]["room"],
        array=resolve_microphone_array(custom_geometry={"positions_m": scene["scene"]["array"]["positions_m"]}, reference_channel=0),
        actors=(
            AcousticActorPlan("a", "target", position_m=(2, 2, 1.5), supervision_index=0),
            AcousticActorPlan("b", "interference", position_m=(3, 2, 1.5), supervision_index=1),
        ),
    )
    provider = OfflineRIRProvider(index_path, geometry={"positions_m": scene["scene"]["array"]["positions_m"]}, require_scene_identity=True)
    tracks = provider.sample_scene_tracks(plan, track_kinds=("full", "direct", "early"))
    assert tracks.full.canonical_rirs.shape == shape
    np.testing.assert_array_equal(tracks.direct.canonical_rirs, scene["direct"])
    np.testing.assert_array_equal(tracks.early.canonical_rirs, scene["early"])
