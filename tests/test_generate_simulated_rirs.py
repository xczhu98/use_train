"""Tests for tools/prepare_data_tools/generators/se/generate_simulated_rirs.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data.resource_paths import resolve_manifest_resource_path  # noqa: E402
from tools.prepare_data_tools._common import parse_float_pair  # noqa: E402
from tools.prepare_data_tools.generators.se.generate_simulated_rirs import (  # noqa: E402
    _detect_gpus,
    _parse_gpu_token,
    _parse_ratio,
    _physical_gpu_token_map,
    _resolve_gpu_ids,
    main,
)
from tools.prepare_data_tools.generators.se.rir_backends import (  # noqa: E402
    FRAMRIRBackend,
    GPURIRBackend,
    RIRGeneratorBackend,
    get_backend,
    list_available_backends,
)
from tools.prepare_data_tools.generators.se.rir_backends.fram import (  # noqa: E402
    _isolated_random_state,
)
from tools.prepare_data_tools.generators.se.rir_config import (  # noqa: E402
    GeneratorConfig,
    MIN_SOURCE_MICROPHONE_DISTANCE,
    RIRConfig,
)
from tools.prepare_data_tools.generators.se.rir_output import make_uid, save_result  # noqa: E402
from tools.prepare_data_tools.generators.se.rir_plan import (  # noqa: E402
    build_field_config,
    derive_scene_seed,
    iter_gpu_tasks,
    iter_tasks,
    recommended_rir_length_samples,
    split_count,
)
from tools.prepare_data_tools.generators.se.rir_scene import (  # noqa: E402
    resolve_scene_seed,
    sample_moving_scene,
    sample_room_and_positions,
    sample_room_rt60,
)
from tools.prepare_data_tools.generators.se.rir_validation import validate_rirs  # noqa: E402
from tools.shared.pickle_io import load_pkl_file, save_pkl_file  # noqa: E402


_REAL_MAIN = main


def run_main(argv):
    """使用 canonical profile 运行 CLI 测试；产品代码不提供旧参数兼容。"""
    if "--config" not in argv:
        argv = [
            "--config",
            "tools/prepare_data_tools/generators/se/rir_gpurir.yaml",
            *argv,
        ]
    if "--sample-rate-hz" not in argv:
        argv = ["--sample-rate-hz", "8000", *argv]
    return _REAL_MAIN(argv)


@pytest.mark.unit
def test_parse_float_pair():
    assert parse_float_pair("0.01,0.3") == (0.01, 0.3)
    assert parse_float_pair(" 3 , 10 ") == (3.0, 10.0)
    with pytest.raises(ValueError):
        parse_float_pair("0.01")


@pytest.mark.unit
def test_detect_gpus_honors_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2, 3")
    assert _detect_gpus() == [2, 3]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert _detect_gpus() == []


@pytest.mark.unit
def test_gpu_tokens_support_uuid_mig_and_visible_slots(monkeypatch):
    assert _parse_gpu_token(" 2 ") == 2
    assert _parse_gpu_token("GPU-abcd") == "GPU-abcd"
    assert _parse_gpu_token("slot:1") == "slot:1"
    assert _parse_gpu_token("id:2") == "id:2"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcd,MIG-GPU-abcd/1/0")
    assert _detect_gpus() == ["GPU-abcd", "MIG-GPU-abcd/1/0"]
    # 设置 CVD 后裸整数表示可见 slot
    assert _resolve_gpu_ids([0, 1]) == ["GPU-abcd", "MIG-GPU-abcd/1/0"]
    assert _resolve_gpu_ids(["slot:1"]) == ["MIG-GPU-abcd/1/0"]
    assert _resolve_gpu_ids(["MIG-GPU-abcd/1/0"]) == ["MIG-GPU-abcd/1/0"]
    with pytest.raises(ValueError, match="unavailable"):
        _resolve_gpu_ids([2])


@pytest.mark.unit
def test_gpu_numeric_tokens_are_visible_slots_under_cvd(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,4")
    assert _resolve_gpu_ids([1]) == [4]
    assert _resolve_gpu_ids(["id:1"]) == [1]
    assert _resolve_gpu_ids(["slot:0"]) == [1]
    with pytest.raises(ValueError, match="unavailable"):
        _resolve_gpu_ids([2])


@pytest.mark.unit
def test_gpu_tokens_reject_duplicates_and_disabled_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with pytest.raises(ValueError, match="duplicates"):
        _resolve_gpu_ids([0, 0])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert _detect_gpus() == []
    assert _resolve_gpu_ids(None) == []


@pytest.mark.unit
def test_negative_gpu_tokens_are_rejected_even_with_cvd(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,4")
    with pytest.raises(ValueError, match="non-negative"):
        _resolve_gpu_ids([-1])
    with pytest.raises(ValueError, match="non-negative"):
        _resolve_gpu_ids([0, -2])


@pytest.mark.unit
def test_physical_gpu_token_map_links_mig_to_its_gpu(monkeypatch):
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    fake_outputs = {
        "query": "0, GPU-aaaa-0000\n1, GPU-bbbb-1111\n",
        # MIG 实例属于 GPU 0，但出现在 GPU 1 之后（nvidia-smi -L 顺序无保证）
        "list": (
            "GPU 0: NVIDIA A800 (UUID: GPU-aaaa-0000)\n"
            "GPU 1: NVIDIA A800 (UUID: GPU-bbbb-1111)\n"
            "MIG 3: MIG 1g.10gb (UUID: MIG-GPU-aaaa-0000/3/0)\n"
        ),
    }

    def fake_run(cmd, *args, **kwargs):
        stdout = fake_outputs["list"] if "-L" in cmd else fake_outputs["query"]
        return SimpleNamespace(stdout=stdout, returncode=0)

    monkeypatch.setattr(f"{generator}.subprocess.run", fake_run)
    token_map = _physical_gpu_token_map()
    assert token_map["GPU-aaaa-0000"] == 0
    assert token_map["GPU-bbbb-1111"] == 1
    assert token_map["MIG-GPU-aaaa-0000/3/0"] == 0


@pytest.mark.unit
def test_sampling_fails_closed_for_invalid_rt60_and_distance_ranges():
    invalid_rt60_cfg = RIRConfig(
        fs=8000,
        c=340.0,
        rt60_range=(0.01, 0.01),
        room_length_range=(1.0, 1.0),
        room_width_range=(1.0, 1.0),
        room_height_range=(1.0, 1.0),
        dist_range=(0.1, 0.2),
        n_samples=512,
        seed=42,
    )
    with pytest.raises(ValueError, match="alpha < 1"):
        sample_room_rt60(invalid_rt60_cfg, np.random.default_rng(42))

    invalid_distance_cfg = RIRConfig(
        fs=8000,
        c=340.0,
        rt60_range=(0.2, 0.3),
        room_length_range=(3.0, 3.0),
        room_width_range=(3.0, 3.0),
        room_height_range=(2.5, 2.5),
        dist_range=(10.0, 11.0),
        n_samples=512,
        seed=42,
    )
    with pytest.raises(ValueError, match="place microphone"):
        sample_room_and_positions(invalid_distance_cfg, 0)

    zero_distance_cfg = RIRConfig(
        fs=8000,
        c=340.0,
        rt60_range=(0.2, 0.3),
        room_length_range=(3.0, 3.0),
        room_width_range=(3.0, 3.0),
        room_height_range=(2.5, 2.5),
        dist_range=(0.0, 0.0),
        n_samples=512,
        seed=42,
    )
    with pytest.raises(ValueError, match="strictly positive"):
        sample_room_and_positions(zero_distance_cfg, 0)


@pytest.mark.unit
def test_microphone_directions_are_uniform_over_the_sphere():
    cfg = RIRConfig(
        fs=8000,
        c=340.0,
        rt60_range=(0.3, 0.3),
        room_length_range=(10.0, 10.0),
        room_width_range=(10.0, 10.0),
        room_height_range=(10.0, 10.0),
        dist_range=(0.2, 0.2),
        n_samples=512,
        seed=42,
    )
    z_components = []
    for idx in range(4000):
        _, source, mic, _, _ = sample_room_and_positions(cfg, idx)
        direction = (np.asarray(mic) - np.asarray(source)) / 0.2
        z_components.append(direction[2])

    z_components = np.asarray(z_components)
    assert abs(float(z_components.mean())) < 0.03
    assert 0.47 < float(np.mean(np.abs(z_components))) < 0.53


@pytest.mark.unit
def test_moving_trajectory_reflects_large_steps_without_stalling():
    cfg = RIRConfig(
        fs=8000,
        c=340.0,
        rt60_range=(0.3, 0.3),
        room_length_range=(3.0, 3.0),
        room_width_range=(3.0, 3.0),
        room_height_range=(2.5, 2.5),
        dist_range=(0.2, 0.2),
        n_samples=512,
        seed=42,
        moving=True,
        n_segments_range=(5, 5),
        segment_sec=10.0,
        speed_range=(10.0, 11.0),
        traj_type="linear",
    )

    room_dim, trajectory, _, _, _, speed = sample_moving_scene(cfg, 0)

    assert speed > 0.0
    assert np.all(trajectory > 0.099)
    assert np.all(trajectory < np.asarray(room_dim) - 0.099)
    assert np.all(np.linalg.norm(np.diff(trajectory, axis=0), axis=1) > 1e-4)


@pytest.mark.unit
def test_moving_trajectory_samples_segment_count_from_range():
    cfg = RIRConfig(
        fs=8000,
        c=340.0,
        rt60_range=(0.3, 0.3),
        room_length_range=(4.0, 4.0),
        room_width_range=(4.0, 4.0),
        room_height_range=(3.0, 3.0),
        dist_range=(0.2, 0.2),
        n_samples=512,
        seed=42,
        moving=True,
        n_segments_range=(3, 6),
        segment_sec=0.2,
        speed_range=(0.5, 1.5),
        traj_type="linear",
    )

    segment_counts = {
        sample_moving_scene(cfg, index)[1].shape[0] for index in range(24)
    }

    assert segment_counts <= {3, 4, 5, 6}
    assert len(segment_counts) > 1


@pytest.mark.unit
def test_fram_image_directions_cover_the_full_sphere():
    from tools.prepare_data_tools.generators.se.rir_backends import fram

    directions = fram._sample_uniform_unit_vectors_torch(8, 2048)
    norms = directions.square().sum(dim=-1).sqrt()

    assert directions.shape == (8, 2048, 3)
    assert float(directions[..., 2].min()) < -0.9
    assert float(directions[..., 2].max()) > 0.9
    assert abs(float(directions[..., 2].mean())) < 0.03
    assert np.allclose(norms.numpy(), 1.0, atol=1e-6)


@pytest.mark.unit
def test_fram_backend_passes_configured_sound_speed(monkeypatch):
    backend = FRAMRIRBackend()
    captured = {}
    monkeypatch.setattr(backend, "_ensure_import", lambda: (object(), object()))
    monkeypatch.setattr(
        "tools.prepare_data_tools.generators.se.rir_backends.fram.sample_room_and_positions",
        lambda *_args, **_kwargs: (
            [4.0, 5.0, 3.0],
            [1.0, 1.0, 1.0],
            [1.2, 1.0, 1.0],
            0.3,
            0.2,
        ),
    )

    def fake_fram_rir(**kwargs):
        captured.update(kwargs)
        return np.ones((1, 1, 16), dtype=np.float32), np.ones(
            (1, 1, 16), dtype=np.float32
        )

    monkeypatch.setattr(backend, "_fram_rir", fake_fram_rir)
    cfg = RIRConfig(
        fs=8000,
        c=300.0,
        rt60_range=(0.3, 0.3),
        room_length_range=(4.0, 4.0),
        room_width_range=(5.0, 5.0),
        room_height_range=(3.0, 3.0),
        dist_range=(0.2, 0.2),
        n_samples=16,
        seed=42,
    )

    backend.generate_one(0, cfg)

    assert captured["c"] == 300.0
    assert MIN_SOURCE_MICROPHONE_DISTANCE > 0.0


@pytest.mark.unit
def testmake_uid_encoding():
    uid = make_uid(
        method="fram",
        fs=16000,
        field="near",
        rt60=0.12345,
        dist=0.45678,
        room_dim=[4.123, 5.456, 3.789],
        idx=7,
    )
    assert uid.startswith("fram_fs16000_near_rt0.12_dist0.46_room4.1x5.5x3.8_00000007")
    assert "fs16000" in uid
    assert "near" in uid
    assert "rt0.12" in uid


@pytest.mark.unit
def test_backend_registry():
    names = ["rirgen", "fram", "gpurir"]
    for name in names:
        b = get_backend(name)
        assert b.name == name


@pytest.mark.unit
def test_list_available_backends():
    available = list_available_backends()
    if FRAMRIRBackend().is_available():
        assert "fram" in available


@pytest.mark.unit
def test_task_iterators_are_lazy_and_preserve_gpu_round_robin(tmp_path):
    cfg = RIRConfig(
        fs=8000,
        c=340.0,
        rt60_range=(0.1, 0.2),
        room_length_range=(3.0, 5.0),
        room_width_range=(3.0, 5.0),
        room_height_range=(2.5, 3.5),
        dist_range=(0.5, 1.0),
        n_samples=512,
        seed=42,
    )
    tasks = iter_tasks("fram", "near", 5, cfg, tmp_path, "wav", False, 2, start_index=10)
    assert iter(tasks) is tasks
    assert [task["idx"] for task in tasks] == [10, 11, 12, 13, 14]

    gpu_ids = [2, 5]
    gpu_tasks = [
        list(iter_gpu_tasks("far", 5, cfg, tmp_path, "wav", False, gpu_ids, slot, 2, 10))
        for slot in range(len(gpu_ids))
    ]
    assert [[task["idx"] for task in slot] for slot in gpu_tasks] == [[10, 12, 14], [11, 13]]
    assert [[task["gpu_id"] for task in slot] for slot in gpu_tasks] == [[2, 2, 2], [5, 5]]
    assert "gpu_seed" not in gpu_tasks[0][0]
    # 场景种子按 (seed, backend, run_mode, mode, idx) 派生，与 GPU 分配无关
    slot_seeds = {task["idx"]: task["cfg"]["scene_seed"] for slot in gpu_tasks for task in slot}
    assert len(slot_seeds) == 5
    assert len(set(slot_seeds.values())) == 5
    other_gpu_tasks = list(
        iter_gpu_tasks("far", 5, cfg, tmp_path, "wav", False, [7, 8], 0, 2, 10)
    )
    assert [task["cfg"]["scene_seed"] for task in other_gpu_tasks] == [
        task["cfg"]["scene_seed"] for task in gpu_tasks[0]
    ]
    assert all(task["cfg"]["seed"] == 42 for slot in gpu_tasks for task in slot)


@pytest.mark.unit
def test_scene_seed_is_stable_and_domain_separated():
    seed = derive_scene_seed(42, "gpurir", 16000, "near", "static", 0)
    assert derive_scene_seed(42, "gpurir", 16000, "near", "static", 0) == seed
    assert seed != derive_scene_seed(42, "gpurir", 16000, "near", "static", 1)
    assert seed != derive_scene_seed(42, "gpurir", 16000, "near", "moving", 0)
    assert seed != derive_scene_seed(42, "gpurir", 16000, "far", "static", 0)
    assert seed != derive_scene_seed(43, "gpurir", 16000, "near", "static", 0)
    # 不同采样率属于不同场景域
    assert seed != derive_scene_seed(42, "gpurir", 32000, "near", "static", 0)


@pytest.mark.unit
def testresolve_scene_seed_falls_back_and_stays_nonnegative():
    cfg = RIRConfig(
        fs=8000,
        c=340.0,
        rt60_range=(0.1, 0.2),
        room_length_range=(3.0, 5.0),
        room_width_range=(3.0, 5.0),
        room_height_range=(2.5, 3.5),
        dist_range=(0.5, 1.0),
        n_samples=512,
        seed=42,
    )
    assert resolve_scene_seed(cfg, 7) == 42 + 7
    scene_cfg = RIRConfig(
        fs=8000,
        c=340.0,
        rt60_range=(0.1, 0.2),
        room_length_range=(3.0, 5.0),
        room_width_range=(3.0, 5.0),
        room_height_range=(2.5, 3.5),
        dist_range=(0.5, 1.0),
        n_samples=512,
        seed=1,
        scene_seed=1 << 70,
    )
    assert 0 <= resolve_scene_seed(scene_cfg, 7) < (1 << 64)


@pytest.mark.unit
def testsample_room_and_positions_is_deterministic_per_scene_seed():
    kwargs = dict(
        fs=8000,
        c=340.0,
        rt60_range=(0.1, 0.2),
        room_length_range=(3.0, 5.0),
        room_width_range=(3.0, 5.0),
        room_height_range=(2.5, 3.5),
        dist_range=(0.5, 1.0),
        n_samples=512,
        seed=42,
    )
    cfg_a = RIRConfig(**kwargs)
    cfg_b = RIRConfig(**kwargs, scene_seed=987654321)
    first = sample_room_and_positions(cfg_a, 0)
    second = sample_room_and_positions(cfg_a, 0)
    assert first == second
    other = sample_room_and_positions(cfg_b, 0)
    assert (first[0], first[3]) != (other[0], other[3])


@pytest.mark.unit
def test_isolated_random_state_restores_global_state():
    torch = pytest.importorskip("torch")

    def states_close(a, b):
        for x, y in zip(a, b):
            if isinstance(x, np.ndarray):
                if not np.array_equal(x, y):
                    return False
            elif x != y:
                return False
        return True

    np.random.seed(1234)
    torch.manual_seed(5678)
    np_before = np.random.get_state()
    torch_before = torch.random.get_rng_state()
    with _isolated_random_state(999):
        drawn_np = np.random.randint(0, 1000, 5)
        drawn_torch = torch.randint(0, 1000, (5,))
    with _isolated_random_state(999):
        assert np.array_equal(np.random.randint(0, 1000, 5), drawn_np)
        assert torch.equal(torch.randint(0, 1000, (5,)), drawn_torch)
    assert states_close(np.random.get_state(), np_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)


@pytest.mark.unit
def test_ratio_parsing_and_count_splitting():
    assert _parse_ratio("7:3") == (7.0, 3.0)
    assert _parse_ratio(" 5 , 5 ") == (5.0, 5.0)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_ratio("7")
    with pytest.raises(ValueError):
        split_count(10, (0.0, 0.0))

    assert split_count(150000, (5.0, 5.0)) == (75000, 75000)
    assert split_count(75000, (7.0, 3.0)) == (52500, 22500)
    assert split_count(1, (1.0, 3.0)) == (0, 1)


@pytest.mark.unit
def test_field_configs_share_rt60_and_calculate_sample_lengths():
    args = SimpleNamespace(
        fs=16000,
        c=340.0,
        rt60_range=(0.2, 1.3),
        rt60_range_near=(0.2, 1.3),
        rt60_range_far=(0.2, 1.3),
        room_length_range=(3.0, 10.0),
        room_width_range=(3.0, 10.0),
        room_height_range=(2.5, 4.5),
        dist_range_near=(0.05, 0.5),
        dist_range_far=(0.1, 3.0),
        n_samples_near=21760,
        n_samples_far=21760,
        seed=42,
        moving=False,
        segments=8,
        segment_sec=0.5,
        speed_range=(0.5, 1.5),
        traj_type="both",
    )

    near_cfg = build_field_config(args, "near", 16000, mode="static")
    far_cfg = build_field_config(args, "far", 16000, mode="static")
    moving_cfg = build_field_config(args, "far", 16000, mode="moving")

    assert near_cfg.rt60_range == far_cfg.rt60_range == (0.2, 1.3)
    assert near_cfg.moving is False
    assert moving_cfg.moving is True
    assert near_cfg.n_samples == recommended_rir_length_samples(
        16000, (0.2, 1.3), (0.05, 0.5), 340.0
    )
    assert far_cfg.n_samples == recommended_rir_length_samples(
        16000, (0.2, 1.3), (0.1, 3.0), 340.0
    )


@pytest.mark.unit
def testsave_result_uses_static_moving_field_layout_and_auto_format(tmp_path):
    common_result = {
        "fs": 8000,
        "rt60": 0.5,
        "dist": 0.3,
        "room_dim": [4.0, 5.0, 3.0],
        "source_loc": [1.0, 2.0, 1.2],
        "mic_loc": [1.2, 2.0, 1.2],
    }
    static_result = {**common_result, "rir": np.array([0.0, 1.0, 0.0], dtype=np.float32)}
    static_entry = save_result(
        static_result,
        {
            "out_dir": str(tmp_path),
            "backend": "fram",
            "field": "near",
            "mode": "static",
            "format": "auto",
            "idx": 2,
            "shards": 2,
            "write_meta_json": True,
        },
    )
    assert static_entry is not None
    static_path = Path(static_entry[3])
    assert static_path.suffix == ".pkl"
    assert static_path.relative_to(tmp_path).parts[:5] == (
        "fram",
        "fs8000",
        "static",
        "near",
        "shard_0000",
    )
    static_meta = json.loads(static_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert static_meta["mode"] == "static"
    assert static_meta["field"] == "near"
    assert static_meta["moving"] is False

    moving_result = {
        **common_result,
        "rir": np.ones((2, 3), dtype=np.float32),
        "moving": True,
        "n_segments": 2,
        "segment_sec": 0.5,
        "trajectory": np.ones((2, 3), dtype=np.float32),
        "speed": 1.0,
    }
    moving_entry = save_result(
        moving_result,
        {
            "out_dir": str(tmp_path),
            "backend": "gpurir",
            "field": "far",
            "mode": "moving",
            "format": "auto",
            "idx": 3,
            "shards": 2,
            "write_meta_json": True,
        },
    )
    assert moving_entry is not None
    moving_path = Path(moving_entry[3])
    assert moving_path.suffix == ".pkl"
    assert moving_path.relative_to(tmp_path).parts[:5] == (
        "gpurir",
        "fs8000",
        "moving",
        "far",
        "shard_0001",
    )
    moving_meta = json.loads(moving_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert moving_meta["mode"] == "moving"
    assert moving_meta["field"] == "far"
    assert moving_meta["moving"] is True


@pytest.mark.unit
def testsave_result_supports_static_flac_and_pkl(tmp_path):
    import soundfile as sf

    common_result = {
        "fs": 8000,
        "rt60": 0.5,
        "dist": 0.3,
        "room_dim": [4.0, 5.0, 3.0],
        "source_loc": [1.0, 2.0, 1.2],
        "mic_loc": [1.2, 2.0, 1.2],
    }
    rir = np.array([0.0, 1.0, 0.5, -0.25], dtype=np.float32)

    flac_entry = save_result(
        {**common_result, "rir": rir},
        {
            "out_dir": str(tmp_path),
            "backend": "fram",
            "field": "near",
            "mode": "static",
            "format": "flac",
            "idx": 0,
            "shards": 1,
            "write_meta_json": False,
        },
    )
    assert flac_entry is not None
    flac_path = Path(flac_entry[3])
    assert flac_path.suffix == ".flac"
    loaded, fs = sf.read(flac_path, dtype="float32")
    assert fs == 8000
    assert np.abs(loaded - rir).max() < 1e-5

    pkl_entry = save_result(
        {**common_result, "rir": rir},
        {
            "out_dir": str(tmp_path),
            "backend": "fram",
            "field": "near",
            "mode": "static",
            "format": "pkl",
            "idx": 0,
            "shards": 1,
            "write_meta_json": False,
        },
    )
    assert pkl_entry is not None
    assert Path(pkl_entry[3]).suffix == ".pkl"


@pytest.mark.unit
def test_static_output_preserves_large_float_rirs_or_rejects_lossy_flac(tmp_path):
    import soundfile as sf

    common_result = {
        "fs": 8000,
        "rt60": 0.5,
        "dist": 0.3,
        "room_dim": [4.0, 5.0, 3.0],
        "source_loc": [1.0, 2.0, 1.2],
        "mic_loc": [1.2, 2.0, 1.2],
    }
    rir = np.array([0.0, 1.25, -1.5, 0.5], dtype=np.float32)

    def task(fmt, idx):
        return {
            "out_dir": str(tmp_path),
            "backend": "fram",
            "field": "near",
            "mode": "static",
            "format": fmt,
            "idx": idx,
            "shards": 1,
            "write_meta_json": False,
        }

    with pytest.raises(ValueError, match=r"FLAC.*\[-1, 1\]"):
        save_result({**common_result, "rir": rir}, task("flac", 0))

    wav_entry = save_result({**common_result, "rir": rir}, task("wav", 1))
    assert wav_entry is not None
    wav, wav_fs = sf.read(wav_entry[3], dtype="float32")
    assert wav_fs == 8000
    assert np.array_equal(wav, rir)

    pkl_entry = save_result({**common_result, "rir": rir}, task("pkl", 2))
    assert pkl_entry is not None
    payload = load_pkl_file(pkl_entry[3], purpose="verify generated RIR test output")
    assert isinstance(payload, dict)
    assert np.array_equal(payload["rir"], rir)


@pytest.mark.unit
def testvalidate_rirs_rejects_nonfinite_wrong_length_and_silent_segments(tmp_path):
    cases = {
        "nan": np.array([1.0, np.nan], dtype=np.float32),
        "inf": np.array([1.0, np.inf], dtype=np.float32),
        "wrong_length": np.ones(3, dtype=np.float32),
        "silent_segment": np.array([[1.0, 0.5], [0.0, 0.0]], dtype=np.float32),
    }
    entries = []
    for uid, rir in cases.items():
        path = tmp_path / f"{uid}.pkl"
        save_pkl_file(path, {"rir": rir, "fs": 16000})
        declared_frames = 2 if uid != "wrong_length" else 4
        entries.append((uid, 16000, declared_frames, str(path)))

    abnormal = validate_rirs(entries, threshold=1e-6)

    assert len(abnormal) == len(cases)
    assert any("nan" in item and "non-finite" in item for item in abnormal)
    assert any("inf" in item and "non-finite" in item for item in abnormal)
    assert any("wrong_length" in item and "frame count" in item for item in abnormal)
    assert any("silent_segment" in item and "segment" in item for item in abnormal)


class _ReportedEntries:
    def __init__(self, reported_count, entries):
        self.reported_count = reported_count
        self.entries = entries

    def __len__(self):
        return self.reported_count

    def __iter__(self):
        return iter(self.entries)


def _fake_result_for_call(kwargs):
    cfg = kwargs["cfg"]
    moving = kwargs["mode"] == "moving"
    result = {
        "rir": np.ones((2, 2) if moving else 2, dtype=np.float32),
        "fs": cfg.fs,
        "rt60": 0.5,
        "dist": 0.3,
        "room_dim": [4.0, 5.0, 3.0],
        "source_loc": [1.0, 2.0, 1.2],
        "mic_loc": [1.2, 2.0, 1.2],
    }
    if moving:
        result.update(
            {
                "moving": True,
                "n_segments": 2,
                "segment_sec": 0.5,
                "trajectory": np.ones((2, 3), dtype=np.float32),
                "speed": 1.0,
            }
        )
    entry = save_result(
        result,
        {
            "out_dir": str(kwargs["output_dir"]),
            "backend": kwargs.get("backend", "gpurir"),
            "field": kwargs["field"],
            "mode": kwargs["mode"],
            "format": kwargs["output_format"],
            "idx": 0,
            "shards": kwargs["shards"],
            "write_meta_json": kwargs["write_metadata_json"],
        },
    )
    assert entry is not None
    return entry


@pytest.mark.unit
def test_main_default_gpurir_uses_requested_counts_ratios_and_auto_formats(monkeypatch, tmp_path):
    # backend 已被替身替换；显式声明可见槽位，不继承宿主的 CPU-only 掩码。
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    calls = []
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(f"{generator}.list_available_backends", lambda: ["gpurir"])

    def fake_run_gpu_backend(**kwargs):
        calls.append(
            (
                kwargs["field"],
                kwargs["mode"],
                kwargs["n_rirs"],
                kwargs["output_format"],
                kwargs["cfg"].n_samples,
            )
        )
        entry = _fake_result_for_call(kwargs)
        return _ReportedEntries(kwargs["n_rirs"], [entry])

    monkeypatch.setattr("tools.prepare_data_tools.generators.se.rir_executor.run_gpu_backend", fake_run_gpu_backend)
    out_dir = tmp_path / "default_gpurir"

    assert run_main(
        [
            "--output-dir",
            str(out_dir),
            "--sample-rate-hz",
            "8000",
            "--field",
            "both",
            "--gpus",
            "0",
            "--shards",
            "1",
        ]
    ) == 0

    assert calls == [
        (
            "near",
            "static",
            50000,
            "pkl",
            recommended_rir_length_samples(8000, (0.1, 1.0), (0.2, 0.8), 340.0),
        ),
        (
            "far",
            "static",
            50000,
            "pkl",
            recommended_rir_length_samples(8000, (0.1, 1.0), (1.0, 3.0), 340.0),
        ),
    ]
    summary = json.loads((out_dir / "rirs_meta.json").read_text(encoding="utf-8"))
    assert summary["backend"] == "gpurir"
    assert summary["shared_rt60_range_sec"] == [0.1, 1.0]
    assert summary["ratio_weights"] == {
        "near_far": [5.0, 5.0],
        "static_moving": [1.0, 0.0],
    }
    assert summary["counts"]["requested"] == {
        "total": 100000,
        "near": 50000,
        "far": 50000,
        "static": 100000,
        "moving": 0,
    }
    assert summary["effective_formats"] == {"static": "pkl", "moving": "pkl"}
    assert len(summary["output_inventory"]) == 2
    assert {item["mode"] for item in summary["output_inventory"]} == {"static"}
    assert {item["field"] for item in summary["output_inventory"]} == {"near", "far"}

    scp_lines = (out_dir / "rirs.scp").read_text(encoding="utf-8").splitlines()
    assert len(scp_lines) == 2
    for line in scp_lines:
        parts = line.split()
        assert len(parts) == 4
        assert not Path(parts[3]).is_absolute()
        assert Path(resolve_manifest_resource_path(parts[3], out_dir / "rirs.scp")).is_file()


@pytest.mark.unit
@pytest.mark.parametrize("method", ["fram", "rirgen"])
def test_explicit_cpu_backend_is_static_only(monkeypatch, tmp_path, method):
    calls = []
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(f"{generator}.list_available_backends", lambda: [method])

    def fake_run_cpu_backend(**kwargs):
        calls.append(
            (kwargs["backend"], kwargs["field"], kwargs["mode"], kwargs["n_rirs"], kwargs["output_format"])
        )
        entry = _fake_result_for_call(kwargs)
        return _ReportedEntries(kwargs["n_rirs"], [entry])

    monkeypatch.setattr("tools.prepare_data_tools.generators.se.rir_executor.run_cpu_backend", fake_run_cpu_backend)
    out_dir = tmp_path / method

    assert run_main(
        [
            "--output-dir",
            str(out_dir),
            "--method",
            method,
            "--field",
            "both",
            "--total-rirs",
            "4",
            "--workers",
            "1",
        ]
    ) == 0

    assert calls == [
        (method, "near", "static", 2, "pkl"),
        (method, "far", "static", 2, "pkl"),
    ]
    summary = json.loads((out_dir / "rirs_meta.json").read_text(encoding="utf-8"))
    assert summary["counts"]["requested"] == {
        "total": 4,
        "near": 2,
        "far": 2,
        "static": 4,
        "moving": 0,
    }


@pytest.mark.unit
def test_static_flac_format_publishes_flac_manifest(monkeypatch, tmp_path):
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(f"{generator}.list_available_backends", lambda: ["fram"])

    def fake_run_cpu_backend(**kwargs):
        entries = []
        for i in range(1, int(kwargs["n_rirs"]) + 1):
            result = {
                "rir": np.ones(2, dtype=np.float32),
                "fs": 8000,
                "rt60": 0.5,
                "dist": 0.3,
                "room_dim": [4.0, 5.0, 3.0],
                "source_loc": [1.0, 2.0, 1.2],
                "mic_loc": [1.2, 2.0, 1.2],
            }
            entry = save_result(
                result,
                {
                    "out_dir": str(kwargs["output_dir"]),
                    "backend": "fram",
                    "field": "near",
                    "mode": "static",
                    "format": "flac",
                    "idx": i,
                    "shards": 1,
                    "write_meta_json": False,
                },
            )
            assert entry is not None
            entries.append(entry)
        return _ReportedEntries(kwargs["n_rirs"], entries)

    monkeypatch.setattr("tools.prepare_data_tools.generators.se.rir_executor.run_cpu_backend", fake_run_cpu_backend)
    out_dir = tmp_path / "flac_static"

    assert run_main(
        [
            "--output-dir",
            str(out_dir),
            "--method",
            "fram",
            "--field",
            "near",
            "--near-rirs",
            "2",
            "--format",
            "flac",
        ]
    ) == 0
    summary = json.loads((out_dir / "rirs_meta.json").read_text(encoding="utf-8"))
    assert summary["effective_formats"] == {"static": "flac", "moving": "pkl"}
    scp_lines = (out_dir / "rirs.scp").read_text(encoding="utf-8").splitlines()
    assert len(scp_lines) == 2
    for line in scp_lines:
        parts = line.split()
        assert parts[3].endswith(".flac")
        assert Path(resolve_manifest_resource_path(parts[3], out_dir / "rirs.scp")).is_file()


@pytest.mark.unit
@pytest.mark.parametrize(
    "args",
    [
        ["--sample-rate-hz", "0"],
        ["--workers", "0"],
        ["--shards", "0"],
        ["--near-rirs", "-1"],
        ["--total-rirs", "-1"],
        ["--static-moving-ratio", "0:0"],
        ["--near-far-ratio", "nan:1"],
        ["--rt60-range-sec", "0,1"],
        ["--rt60-range-sec", "1.0,0.5"],
        ["--rt60-range-sec", "0.2,1.3", "--rt60-range-sec", "0.2,1.0"],
        ["--dist-range-near", "-0.1,0.5"],
        ["--dist-range-far", "0.1,-3.0"],
        ["--room-length-range-m", "0,10"],
        ["--room-width-range-m", "-1,10"],
        ["--room-height-range-m", "0,4.5"],
        ["--sound-speed-mps", "nan"],
        ["--segment-duration-sec", "nan"],
        ["--speed-range-mps", "0.5,nan"],
        ["--validation-peak-threshold", "nan"],
    ],
)
def test_main_rejects_invalid_generation_parameters(tmp_path, args):
    with pytest.raises(SystemExit) as exc_info:
        run_main(["--output-dir", str(tmp_path / "out"), *args])
    assert exc_info.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.unit
def test_duplicate_sample_rates_are_rejected_before_backend_resolution(
    monkeypatch, tmp_path
):
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(
        f"{generator}.list_available_backends",
        lambda: pytest.fail("duplicate --sample-rate-hz must fail before backend resolution"),
    )

    with pytest.raises(SystemExit) as exc_info:
        run_main(
            [
                "--output-dir",
                str(tmp_path / "out"),
                "--method",
                "fram",
                "--sample-rate-hz",
                "16000",
                "16000",
            ]
        )

    assert exc_info.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.unit
def test_rir_generator_version_tracks_geometry_and_integrity_changes():
    from tools.prepare_data_tools.generators.se import generate_simulated_rirs

    assert generate_simulated_rirs.GENERATOR_VERSION_HISTORY[-2:] == ("1.11", "1.12")
    assert generate_simulated_rirs.GENERATOR_VERSION == "1.12"


@pytest.mark.unit
def test_rir_yaml_profiles_are_typed_and_unknown_keys_fail_fast():
    profile = GeneratorConfig.from_yaml(
        "tools/prepare_data_tools/generators/se/rir_gpurir.yaml"
    )
    assert profile.profile == "realistic_speech"
    assert profile.sample_rates_hz == (16000, 32000)
    assert profile.output_format == "pkl"
    assert profile.total_rirs == 100000
    assert profile.rt60_range_sec == (0.1, 1.0)
    assert profile.static_moving_ratio == (1.0, 0.0)
    assert profile.motion_enabled is False
    assert profile.n_segments_range == (3, 6)
    assert profile.shards == 10
    assert profile.near_distance_range_m == (0.2, 0.8)
    assert profile.backend_options.gpurir.diffuse_transition_db == 15.0

    with pytest.raises(ValueError, match="unknown config key"):
        GeneratorConfig.from_mapping({"unknown": 1})


@pytest.mark.unit
def test_typed_config_rejects_coercion_and_unsupported_directionality():
    empty = GeneratorConfig.from_mapping({})
    assert empty.far_distance_range_m == (0.8, 3.0)

    with pytest.raises(ValueError, match="must be a boolean"):
        GeneratorConfig.from_mapping({"motion": {"enabled": "false"}})
    with pytest.raises(ValueError, match="integer values"):
        GeneratorConfig.from_mapping(
            {"backend": {"name": "fram", "fram": {"n_image_range": [512.5, 2049]}}}
        )
    with pytest.raises(ValueError, match="orientation"):
        GeneratorConfig.from_mapping(
            {"backend": {"name": "gpurir", "gpurir": {"speaker_pattern": "card"}}}
        )
    directional = GeneratorConfig.from_mapping(
        {
            "backend": {
                "name": "gpurir",
                "gpurir": {
                    "speaker_pattern": "card",
                    "speaker_orientation": [2.0, 0.0, 0.0],
                },
            },
            "scene": {"distance": {"allow_overlap": True}},
        }
    )
    assert directional.backend_options.gpurir.speaker_orientation == (1.0, 0.0, 0.0)


@pytest.mark.unit
def test_dump_effective_config_uses_normalized_counts(capsys):
    assert _REAL_MAIN(
        [
            "--config",
            "tools/prepare_data_tools/generators/se/rir_gpurir.yaml",
            "--near-rirs",
            "2",
            "--dump-effective-config",
        ]
    ) == 0
    effective = json.loads(capsys.readouterr().out)
    assert (effective["total_rirs"], effective["near_rirs"], effective["far_rirs"]) == (
        2,
        2,
        0,
    )


@pytest.mark.unit
def test_config_file_runs_through_runtime_plan(monkeypatch, tmp_path):
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(f"{generator}.list_available_backends", lambda: ["fram"])
    out_dir = tmp_path / "config_adapter"

    assert run_main(
        [
            "--config",
            "tools/prepare_data_tools/generators/se/rir_gpurir.yaml",
            "--output-dir",
            str(out_dir),
            "--method",
            "fram",
            "--total-rirs",
            "0",
        ]
    ) == 0
    summary = json.loads((out_dir / "rirs_meta.json").read_text(encoding="utf-8"))
    assert summary["config"]["source"].endswith(
        "rir_gpurir.yaml"
    )
    assert summary["config"]["effective"]["output_dir"] == str(out_dir)


@pytest.mark.unit
def test_moving_all_uses_only_gpurir(monkeypatch, tmp_path):
    observed = []
    monkeypatch.setattr(
        "tools.prepare_data_tools.generators.se.generate_simulated_rirs.list_available_backends",
        lambda: ["rirgen", "fram", "gpurir"],
    )

    def fake_run_gpu_backend(**kwargs):
        observed.append(
            (kwargs["field"], kwargs["mode"], kwargs["n_rirs"], kwargs["output_format"])
        )
        return []

    monkeypatch.setattr(
        "tools.prepare_data_tools.generators.se.rir_executor.run_gpu_backend",
        fake_run_gpu_backend,
    )
    assert run_main(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--method",
            "all",
            "--motion-enabled",
            "--format",
            "pkl",
            "--field",
            "near",
            "--near-rirs",
            "0",
        ]
    ) == 0
    assert observed == []


@pytest.mark.unit
def test_static_all_skips_gpurir_when_auto_detection_finds_no_gpu(monkeypatch, tmp_path):
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    observed = []
    monkeypatch.setattr(
        f"{generator}.list_available_backends", lambda: ["fram", "gpurir"]
    )
    monkeypatch.setattr(f"{generator}._resolve_gpu_ids", lambda _requested: [])

    def fake_run_cpu_backend(**kwargs):
        observed.append(kwargs["backend"])
        return _ReportedEntries(kwargs["n_rirs"], [_fake_result_for_call(kwargs)])

    monkeypatch.setattr("tools.prepare_data_tools.generators.se.rir_executor.run_cpu_backend", fake_run_cpu_backend)
    monkeypatch.setattr(
        "tools.prepare_data_tools.generators.se.rir_executor.run_gpu_backend",
        lambda **_kwargs: pytest.fail("gpuRIR must be skipped when no GPU is available"),
    )
    out_dir = tmp_path / "all_cpu"

    assert run_main(
        [
            "--output-dir",
            str(out_dir),
            "--method",
            "all",
            "--field",
            "near",
            "--total-rirs",
            "1",
        ]
    ) == 0
    assert observed == ["fram"]
    summary = json.loads((out_dir / "rirs_meta.json").read_text(encoding="utf-8"))
    assert summary["statistics"]["by_backend"] == {"fram": 1}


@pytest.mark.unit
def test_gpurir_field_counts_keep_static_wav(monkeypatch, tmp_path):
    # 本例验证任务计数与输出格式，不实际启动 GPU backend。
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    calls = []
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(f"{generator}.list_available_backends", lambda: ["gpurir"])

    def fake_run_gpu_backend(**kwargs):
        calls.append(
            (kwargs["field"], kwargs["mode"], kwargs["n_rirs"], kwargs["output_format"])
        )
        return _ReportedEntries(kwargs["n_rirs"], [_fake_result_for_call(kwargs)])

    monkeypatch.setattr("tools.prepare_data_tools.generators.se.rir_executor.run_gpu_backend", fake_run_gpu_backend)
    out_dir = tmp_path / "gpurir_field_counts"

    assert run_main(
        [
            "--output-dir",
            str(out_dir),
            "--method",
            "gpurir",
            "--field",
            "far",
            "--far-rirs",
            "2",
            "--format",
            "wav",
            "--gpus",
            "0",
        ]
    ) == 0

    assert calls == [("far", "static", 2, "wav")]
    summary = json.loads((out_dir / "rirs_meta.json").read_text(encoding="utf-8"))
    assert summary["counts"]["requested"] == {
        "total": 2,
        "near": 0,
        "far": 2,
        "static": 2,
        "moving": 0,
    }
    assert summary["parameters"]["near_rirs"] == 0
    assert summary["parameters"]["far_rirs"] == 2
    assert summary["parameters"]["total_rirs"] == 2


@pytest.mark.unit
def test_removed_rt60_range_option_is_rejected(monkeypatch, tmp_path):
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(f"{generator}.list_available_backends", lambda: ["fram"])
    monkeypatch.setattr("tools.prepare_data_tools.generators.se.rir_executor.run_cpu_backend", lambda **_kwargs: [])

    with pytest.raises(SystemExit) as exc_info:
        run_main(
            [
                "--output-dir",
                str(tmp_path / "out"),
                "--method",
                "fram",
                "--field",
                "near",
                "--near-rirs",
                "0",
                "--rt60-range",
                "0.2,1.3",
            ]
        )
    assert exc_info.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.unit
def test_explicit_sample_length_shorter_than_recommendation_warns(
    monkeypatch, caplog, tmp_path
):
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(f"{generator}.list_available_backends", lambda: ["fram"])

    def fake_run_cpu_backend(**kwargs):
        return _ReportedEntries(kwargs["n_rirs"], [_fake_result_for_call(kwargs)])

    monkeypatch.setattr("tools.prepare_data_tools.generators.se.rir_executor.run_cpu_backend", fake_run_cpu_backend)
    caplog.set_level("WARNING", logger="generate_simulated_rirs")

    assert run_main(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--method",
            "fram",
            "--field",
            "near",
            "--near-rirs",
            "1",
            "--near-rir-length-samples",
            "256",
        ]
    ) == 0
    assert any("shorter than the recommended" in record.message for record in caplog.records)


@pytest.mark.unit
def test_worker_failure_does_not_publish_rir_indexes(monkeypatch, tmp_path):
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(f"{generator}.list_available_backends", lambda: ["fram"])
    monkeypatch.setattr("tools.prepare_data_tools.generators.se.rir_executor.run_cpu_backend", lambda **_kwargs: [])
    out_dir = tmp_path / "out"

    assert run_main(
        [
            "--output-dir",
            str(out_dir),
            "--method",
            "fram",
            "--field",
            "near",
            "--near-rirs",
            "1",
        ]
    ) == 1
    assert not (out_dir / "rirs.scp").exists()
    assert not (out_dir / "rirs_meta.json").exists()


@pytest.mark.unit
def test_duplicate_uids_do_not_publish_rir_indexes(monkeypatch, tmp_path):
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(f"{generator}.list_available_backends", lambda: ["fram"])

    def fake_run_cpu_backend(**kwargs):
        entry = _fake_result_for_call(kwargs)
        return [entry, entry]

    monkeypatch.setattr("tools.prepare_data_tools.generators.se.rir_executor.run_cpu_backend", fake_run_cpu_backend)
    out_dir = tmp_path / "duplicate_uids"

    assert run_main(
        [
            "--output-dir",
            str(out_dir),
            "--method",
            "fram",
            "--field",
            "near",
            "--total-rirs",
            "2",
        ]
    ) == 1
    assert not (out_dir / "rirs.scp").exists()
    assert not (out_dir / "rirs_meta.json").exists()


@pytest.mark.unit
def test_moving_params_only_validated_when_moving_tasks_exist(monkeypatch, tmp_path):
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(f"{generator}.list_available_backends", lambda: ["fram"])
    monkeypatch.setattr("tools.prepare_data_tools.generators.se.rir_executor.run_cpu_backend", lambda **_kwargs: [])

    # 纯静态任务传语义非法的移动参数（segs=1、sec=0、speed lo>hi）应正常通过
    static_out = tmp_path / "static_out"
    assert run_main(
        [
            "--output-dir",
            str(static_out),
            "--method",
            "fram",
            "--field",
            "near",
            "--near-rirs",
            "0",
            "--n-segments",
            "1",
            "--segment-duration-sec",
            "0",
            "--speed-range-mps",
            "1.0,0.5",
        ]
    ) == 0

    # 数值形式非法（nan）无条件拒绝，与是否静态无关
    with pytest.raises(SystemExit) as exc_info:
        run_main(
            [
                "--output-dir",
                str(tmp_path / "nan_out"),
                "--method",
                "fram",
                "--field",
                "near",
                "--near-rirs",
                "0",
                "--speed-range-mps",
                "0.5,nan",
            ]
        )
    assert exc_info.value.code == 2

        # 显式启用移动任务时，段数语义下限仍被拒绝。
    with pytest.raises(SystemExit) as exc_info:
        run_main(
            [
                "--output-dir",
                str(tmp_path / "moving_out"),
                "--method",
                "gpurir",
                "--field",
                "near",
                    "--total-rirs",
                    "4",
                    "--motion-enabled",
                    "--n-segments",
                "1",
            ]
        )
    assert exc_info.value.code == 2


@pytest.mark.unit
def test_sha256_inventory_skipped_by_default_and_enabled_by_flag(monkeypatch, tmp_path):
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(f"{generator}.list_available_backends", lambda: ["fram"])

    def fake_run_cpu_backend(**kwargs):
        return _ReportedEntries(1, [_fake_result_for_call(kwargs)])

    monkeypatch.setattr("tools.prepare_data_tools.generators.se.rir_executor.run_cpu_backend", fake_run_cpu_backend)

    # 默认跳过指纹
    out_dir = tmp_path / "default"
    assert run_main(
        [
            "--output-dir",
            str(out_dir),
            "--method",
            "fram",
            "--field",
            "near",
            "--near-rirs",
            "1",
        ]
    ) == 0
    summary = json.loads((out_dir / "rirs_meta.json").read_text(encoding="utf-8"))
    assert summary["compute_sha256"] is False
    assert all(item["sha256"] is None for item in summary["output_inventory"])

    # 显式 --inventory 开启指纹
    out_dir2 = tmp_path / "with_inventory"
    assert run_main(
        [
            "--output-dir",
            str(out_dir2),
            "--method",
            "fram",
            "--field",
            "near",
            "--near-rirs",
            "1",
            "--inventory",
        ]
    ) == 0
    summary2 = json.loads((out_dir2 / "rirs_meta.json").read_text(encoding="utf-8"))
    assert summary2["compute_sha256"] is True
    assert all(len(item["sha256"]) == 64 for item in summary2["output_inventory"])

    # --no-inventory 显式声明默认行为
    out_dir3 = tmp_path / "no_inventory"
    assert run_main(
        [
            "--output-dir",
            str(out_dir3),
            "--method",
            "fram",
            "--field",
            "near",
            "--near-rirs",
            "1",
            "--no-inventory",
        ]
    ) == 0
    summary3 = json.loads((out_dir3 / "rirs_meta.json").read_text(encoding="utf-8"))
    assert summary3["compute_sha256"] is False


@pytest.mark.unit
def test_multi_fs_generates_per_fs_layout_and_scps(monkeypatch, tmp_path):
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(f"{generator}.list_available_backends", lambda: ["fram"])
    calls = []

    def fake_run_cpu_backend(**kwargs):
        calls.append((kwargs["field"], kwargs["mode"], kwargs["cfg"].fs, kwargs["cfg"].n_samples))
        entry = _fake_result_for_call(kwargs)
        return _ReportedEntries(kwargs["n_rirs"], [entry])

    monkeypatch.setattr("tools.prepare_data_tools.generators.se.rir_executor.run_cpu_backend", fake_run_cpu_backend)
    out_dir = tmp_path / "multi_fs"

    assert run_main(
        [
            "--output-dir",
            str(out_dir),
            "--method",
            "fram",
            "--field",
            "both",
            "--total-rirs",
            "4",
            "--sample-rate-hz",
            "8000",
            "16000",
            "--workers",
            "1",
        ]
    ) == 0

    # 每个采样率各生成一套 near/far
    assert calls == [
        ("near", "static", 8000, recommended_rir_length_samples(8000, (0.1, 1.0), (0.2, 0.8), 340.0)),
        ("far", "static", 8000, recommended_rir_length_samples(8000, (0.1, 1.0), (1.0, 3.0), 340.0)),
        ("near", "static", 16000, recommended_rir_length_samples(16000, (0.1, 1.0), (0.2, 0.8), 340.0)),
        ("far", "static", 16000, recommended_rir_length_samples(16000, (0.1, 1.0), (1.0, 3.0), 340.0)),
    ]
    # 目录按采样率分层
    assert (out_dir / "fram" / "fs8000" / "static" / "near").is_dir()
    assert (out_dir / "fram" / "fs16000" / "static" / "near").is_dir()
    # 混合 scp + 每采样率独立 scp
    mixed = (out_dir / "rirs.scp").read_text(encoding="utf-8").splitlines()
    assert len(mixed) == 4
    fs8 = (out_dir / "rirs_fs8000.scp").read_text(encoding="utf-8").splitlines()
    fs16 = (out_dir / "rirs_fs16000.scp").read_text(encoding="utf-8").splitlines()
    assert len(fs8) == 2 and len(fs16) == 2
    assert all(line.split()[1] == "8000" for line in fs8)
    assert all(line.split()[1] == "16000" for line in fs16)
    summary = json.loads((out_dir / "rirs_meta.json").read_text(encoding="utf-8"))
    assert summary["sample_rates_hz"] == [8000, 16000]


@pytest.mark.unit
def test_validation_failure_blocks_manifest_publication(monkeypatch, tmp_path):
    """--validate 是严格发布门禁：异常 RIR 存在时不发布 rirs.scp / rirs_meta.json。"""
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(f"{generator}.list_available_backends", lambda: ["fram"])

    def fake_run_cpu_backend(**kwargs):
        # 生成一条全零（幅值过低）的 RIR
        result = {
            "rir": np.zeros(4, dtype=np.float32),
            "fs": 8000,
            "rt60": 0.3,
            "dist": 0.3,
            "room_dim": [4.0, 5.0, 3.0],
            "source_loc": [1.0, 2.0, 1.2],
            "mic_loc": [1.2, 2.0, 1.2],
        }
        entry = save_result(
            result,
            {
                "out_dir": str(kwargs["output_dir"]),
                "backend": "fram",
                "field": "near",
                "mode": "static",
                "format": "wav",
                "idx": 0,
                "shards": 1,
                "write_meta_json": False,
            },
        )
        assert entry is not None
        return _ReportedEntries(1, [entry])

    monkeypatch.setattr("tools.prepare_data_tools.generators.se.rir_executor.run_cpu_backend", fake_run_cpu_backend)
    out_dir = tmp_path / "out"

    assert run_main(
        [
            "--output-dir",
            str(out_dir),
            "--method",
            "fram",
            "--field",
            "near",
            "--near-rirs",
            "1",
            "--validate",
            "--validation-peak-threshold",
            "1e-6",
        ]
    ) == 1
    assert not (out_dir / "rirs.scp").exists()
    assert not (out_dir / "rirs_meta.json").exists()


@pytest.mark.unit
def test_nonempty_output_directory_is_rejected(monkeypatch, tmp_path):
    generator = "tools.prepare_data_tools.generators.se.generate_simulated_rirs"
    monkeypatch.setattr(f"{generator}.list_available_backends", lambda: ["fram"])
    monkeypatch.setattr(
        "tools.prepare_data_tools.generators.se.rir_executor.run_cpu_backend",
        lambda **_kwargs: pytest.fail("backend must not run for a non-empty output directory"),
    )
    out_dir = tmp_path / "existing"
    out_dir.mkdir()
    marker = out_dir / "existing.marker"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        run_main(
            [
                "--output-dir",
                str(out_dir),
                "--method",
                "fram",
                "--field",
                "near",
                "--near-rirs",
                "0",
            ]
        )
    assert exc_info.value.code == 1
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.external
@pytest.mark.skipif(not FRAMRIRBackend().is_available(), reason="FRAM-RIR not available")
def test_fram_backend_generate_one(tmp_path):
    backend = FRAMRIRBackend()
    cfg = RIRConfig(
        fs=8000,
        c=340.0,
        rt60_range=(0.1, 0.2),
        room_length_range=(3.0, 4.0),
        room_width_range=(3.0, 4.0),
        room_height_range=(2.5, 3.0),
        dist_range=(0.2, 0.4),
        n_samples=512,
        seed=42,
    )
    result = backend.generate_one(0, cfg)
    assert "rir" in result
    assert result["rir"].ndim == 1
    assert result["rir"].dtype == "float32"
    assert result["rir"].shape[0] == cfg.n_samples
    assert result["fs"] == cfg.fs
    assert cfg.rt60_range[0] <= result["rt60"] <= cfg.rt60_range[1]
    assert cfg.dist_range[0] <= result["dist"] <= cfg.dist_range[1]
    assert len(result["room_dim"]) == 3


@pytest.mark.external
@pytest.mark.skipif(not RIRGeneratorBackend().is_available(), reason="rir_generator not available")
def test_rirgen_backend_generate_one(tmp_path):
    backend = RIRGeneratorBackend()
    cfg = RIRConfig(
        fs=8000,
        c=340.0,
        rt60_range=(0.1, 0.2),
        room_length_range=(3.0, 4.0),
        room_width_range=(3.0, 4.0),
        room_height_range=(2.5, 3.0),
        dist_range=(0.5, 1.0),
        n_samples=512,
        seed=42,
    )
    result = backend.generate_one(0, cfg)
    assert result["rir"].ndim == 1
    assert result["rir"].shape[0] == cfg.n_samples


@pytest.mark.external
@pytest.mark.skipif(not GPURIRBackend().is_available(), reason="gpuRIR not available")
def test_gpurir_backend_generate_one(tmp_path):
    backend = GPURIRBackend()
    cfg = RIRConfig(
        fs=8000,
        c=340.0,
        rt60_range=(0.1, 0.2),
        room_length_range=(3.0, 4.0),
        room_width_range=(3.0, 4.0),
        room_height_range=(2.5, 3.0),
        dist_range=(0.5, 1.0),
        n_samples=512,
        seed=42,
    )
    result = backend.generate_one(0, cfg)
    assert result["rir"].ndim == 1
    assert result["rir"].shape[0] == cfg.n_samples


@pytest.mark.external
@pytest.mark.skipif(not FRAMRIRBackend().is_available(), reason="FRAM-RIR not available")
def test_cli_smoke(tmp_path):
    """端到端冒烟测试：生成少量 RIR 并验证 rirs.scp 格式。"""
    import subprocess

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "tools.prepare_data_tools.generators.se.generate_simulated_rirs",
        "--config",
        "tools/prepare_data_tools/generators/se/rir_fram.yaml",
        "--output-dir",
        str(out_dir),
        "--method",
        "fram",
        "--sample-rate-hz",
        "8000",
        "--field",
        "near",
        "--near-rirs",
        "2",
        "--near-rir-length-samples",
        "512",
        "--rt60-range-sec",
        "0.1,0.2",
        "--near-distance-range-m",
        "0.2,0.4",
        "--room-length-range-m",
        "3,4",
        "--room-width-range-m",
        "3,4",
        "--room-height-range-m",
        "2.5,3",
        "--workers",
        "2",
        "--seed",
        "42",
        "--validate",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}"

    scp_path = out_dir / "rirs.scp"
    assert scp_path.exists()

    lines = scp_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        parts = line.split()
        assert len(parts) == 4, f"SCP line must have 4 columns: {line}"
        uid, fs_str, n_samples_str, path = parts
        assert uid.startswith("fram_fs8000_near_")
        assert fs_str == "8000"
        assert int(n_samples_str) == 512
        assert not Path(path).is_absolute()
        assert Path(resolve_manifest_resource_path(path, scp_path)).is_file()

    meta_path = out_dir / "rirs_meta.json"
    assert meta_path.exists()
