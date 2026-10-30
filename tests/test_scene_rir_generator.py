"""SE 生成器的 mono legacy / multichannel scene 输出分界。"""

import numpy as np
import pytest

from tools.prepare_data_tools.generators.se.rir_config import GeneratorConfig
from tools.prepare_data_tools.generators.se.rir_plan import build_field_config
from tools.prepare_data_tools.generators.se.rir_output import save_scene_result
from tools.prepare_data_tools.generators.se.scene_generation import (
    generate_gpurir_scene,
    sample_spatial_scene,
)

pytestmark = pytest.mark.unit


def _config():
    return GeneratorConfig.from_mapping({
        "schema_version": 1,
        "profile": "fixture",
        "backend": "gpurir",
        "sample_rates_hz": [8000],
        "dataset": {"total_rirs": 1},
        "scene": {
            "room": {"length_range_m": [4, 4], "width_range_m": [4, 4], "height_range_m": [3, 3]},
            "geometry": {
                "array_profile": "linear_2ch_80mm",
                "reference_channel": 0,
                "channel_order": ["mic_0", "mic_1"],
            },
            "distance": {"near_range_m": [0.7, 0.7], "far_range_m": [1.2, 1.2]},
            "actors": {"roles": ["target", "interference"]},
        },
        "acoustics": {
            "rt60_range_sec": [0.2, 0.2],
            "sound_speed_mps": 340,
            "near_rir_length_samples": 64,
            "far_rir_length_samples": 64,
            "early_rir_sec": 0.01,
        },
        "motion": {"enabled": False},
        "execution": {"workers": 1, "shards": 1, "gpus": None},
        "output": {"format": "pkl", "write_metadata_json": False, "compute_sha256": True},
        "validation": {"enabled": True, "peak_threshold": 1e-6},
        "seed": 3,
    })


def test_spatial_generator_plan_has_absolute_array_and_actor_scene():
    config = _config()
    args = type("Args", (), config.to_runtime_namespace())()
    rir_cfg = build_field_config(args, "far", 8000, mode="static")
    plan = sample_spatial_scene(rir_cfg, 0, generator_version="fixture-v2")
    assert plan.channels == 2
    assert plan.actor_order == ("target", "interference")
    assert plan.room["rir_num_samples"] == 64
    assert len(plan.array["positions_m"]) == 2
    assert all(len(actor["position_m"]) == 3 for actor in plan.actors)
    assert plan.scene_uid.startswith("scene-")


def test_scene_output_publishes_full_direct_early_and_identity(tmp_path):
    config = _config()
    args = type("Args", (), config.to_runtime_namespace())()
    rir_cfg = build_field_config(args, "far", 8000, mode="static")
    plan = sample_spatial_scene(rir_cfg, 0, generator_version="fixture-v2")
    shape = (2, 2, 64)
    result = {
        "scene": plan.to_dict(),
        "full": np.ones(shape, dtype=np.float32),
        "direct": np.ones(shape, dtype=np.float32) * .8,
        "early": np.ones(shape, dtype=np.float32) * .6,
    }
    record = save_scene_result(result, output_dir=tmp_path, index=0, shards=1)
    assert record["metadata"]["schema_version"] == 2
    assert record["metadata"]["actor_order"] == ["target", "interference"]
    assert record["metadata"]["geometry"]["reference_channel"] == 0
    assert set(record["metadata"]["track_identities"]["target"]) == {"full", "direct", "early"}
    bundle_paths = {tracks["bundle_path"] for tracks in record["entries"].values()}
    assert len(bundle_paths) == 1
    assert (tmp_path / "scene_manifests" / next(iter(bundle_paths))).resolve().exists()
    scene_manifest = tmp_path / record["scene_manifest_path"]
    assert scene_manifest.is_file()
    assert scene_manifest.read_text(encoding="utf-8").strip().startswith("{")


def test_legacy_config_without_geometry_stays_out_of_scene_mode():
    config = GeneratorConfig.from_mapping({
        "schema_version": 1, "profile": "legacy", "backend": "gpurir",
        "sample_rates_hz": [8000], "dataset": {"total_rirs": 1},
    })
    assert config.array_profile is None
    assert config.custom_geometry is None
