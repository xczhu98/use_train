"""AEC spatial dynamic branch shape regressions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from tasks.aec.data.collate import aec_collate_fn
from tasks.aec.data.dynamic_dataset import DynamicMixingDataset
from core.data.simulation.rir_provider import MovingRIRCapabilityError

pytestmark = pytest.mark.unit


class SceneProvider(SimpleNamespace):
    def __call__(self, *args, **kwargs):
        raise AssertionError("AEC must not call the legacy provider seam")



@pytest.fixture(autouse=True)
def prepared_aec_scene(monkeypatch):
    """共享 sampler 接口替身：测试只控制场景事实，不替代波形渲染。"""
    from core.data.acoustic_scene_plan import (
        AcousticActorPlan, AcousticScenePlan, RIRPlan, SourceGainPlan, NoisePlan,
    )
    from core.data.simulation.array_profiles import resolve_microphone_array
    from core.data.simulation.trajectory import TrajectoryTimebase, plan_linear_source_trajectory
    from core.data.simulation import scene_sampler
    calls = []

    def prepare(**kwargs):
        calls.append(kwargs)
        clock = TrajectoryTimebase(kwargs["sample_rate"], kwargs["num_samples"],
                                   kwargs["num_samples"] // 4, 4) if kwargs["moving"] else None
        array = resolve_microphone_array(
            custom_geometry={"positions_m": [[2, 2, 1], [2.08, 2, 1]]},
            reference_channel=kwargs["spatial_view"].reference_channel,
        )
        actors = []
        for i, (uid, role) in enumerate(zip(kwargs["source_uids"], kwargs["roles"])):
            start = np.array([[1.0 + i, 3, 1.5]])
            end = start if uid in kwargs["stationary_actor_uids"] else start + [0.2, 0, 0]
            actors.append(AcousticActorPlan(
                actor_uid=uid, role=role, position_m=tuple(start[0]),
                trajectory=plan_linear_source_trajectory(start, end, clock) if clock else None,
            ))
        return AcousticScenePlan(
            sample_rate=kwargs["sample_rate"], num_samples=kwargs["num_samples"],
            need_noise=False, rir=RIRPlan.disabled(), source_gain=SourceGainPlan.disabled(),
            noise=NoisePlan.disabled(), identity={"scene_uid": "aec-fixture"},
            room={"room_uid": "room-fixture", "room_dim_m": [6, 6, 3], "rt60": 0.4},
            array=array, actors=tuple(actors), timebase=clock,
        )
    monkeypatch.setattr(scene_sampler, "prepare_scene_plan", prepare, raising=False)
    return calls


def _static_tracks(plan):
    from core.data.simulation.rir_provider import SpatialRIRResult, SpatialRIRTrackSet
    bank = np.repeat(np.asarray([[[1.0], [0.0]]], np.float32), len(plan.actors), axis=0)
    result = SpatialRIRResult(
        bank, {"positions_m": np.asarray(plan.array.positions_m).tolist(),
               "reference_channel": plan.array.reference_channel},
        {"scene_uid": plan.identity["scene_uid"], "room": dict(plan.room),
         "actor_order": [actor.actor_uid for actor in plan.actors]}, "fixture-v1",
    )
    return SpatialRIRTrackSet(full=result, early=result)


def _write_scp(path: Path, uid: str, audio: Path, sr: int) -> None:
    info = sf.info(str(audio))
    path.write_text(f"{uid} {sr} {info.frames} {audio}\n", encoding="utf-8")


def _spatial_aec_dataset(tmp_path: Path) -> DynamicMixingDataset:
    sr = 8000
    length = 4000
    time = np.arange(length, dtype=np.float32) / sr
    speech = 0.1 * np.sin(2 * np.pi * 300 * time)
    noise = np.full(length, 0.01, dtype=np.float32)
    farend = 0.1 * np.sin(2 * np.pi * 500 * time)
    echo = 0.2 * farend
    rir = np.zeros(16, dtype=np.float32)
    rir[0] = 1.0
    for name, waveform in {
        "speech": speech,
        "noise": noise,
        "rir": rir,
    }.items():
        sf.write(tmp_path / f"{name}.wav", waveform, sr)
    sf.write(tmp_path / "far_echo.wav", np.stack((echo, farend), axis=1), sr)
    _write_scp(tmp_path / "speech.scp", "speech", tmp_path / "speech.wav", sr)
    _write_scp(tmp_path / "noise.scp", "noise", tmp_path / "noise.wav", sr)
    _write_scp(tmp_path / "rir.scp", "rir", tmp_path / "rir.wav", sr)
    (tmp_path / "pairs.json").write_text(
        json.dumps({"pairs": [str(tmp_path / "far_echo.wav")]}), encoding="utf-8"
    )

    cfg = {
        "prob_target_only": 1.0 / 7.0,
        "prob_noise_only": 1.0 / 7.0,
        "prob_echo_only": 1.0 / 7.0,
        "prob_noise_echo_only": 1.0 / 7.0,
        "prob_target_noise": 1.0 / 7.0,
        "prob_target_echo": 1.0 / 7.0,
        "prob_target_noise_echo": 1.0 / 7.0,
        "prob_reverberation": 0.0,
        "prob_channel_enabled": False,
        "snr_low": 5.0,
        "snr_high": 5.0,
        "spatial_enabled": True,
        "array_profile": "linear_2ch_80mm",
        "reference_channel": 1,
    }
    provider = SceneProvider(sample_scene_tracks=lambda plan, **kwargs: _static_tracks(plan))
    dataset = DynamicMixingDataset(
        str(tmp_path / "speech.scp"),
        str(tmp_path / "noise.scp"),
        str(tmp_path / "rir.scp"),
        simulation_cfg=cfg,
        max_duration_seconds=0.25,
        target_sr=sr,
        filter_by_duration=False,
        strict_load=True,
        real_farend_echo_pair_json=str(tmp_path / "pairs.json"),
        spatial_rir_provider=provider,
    )
    assert dataset._spatial_rir_provider is provider
    return dataset


@pytest.mark.parametrize(
    "branch",
    (
        "prob_target_only",
        "prob_noise_only",
        "prob_echo_only",
        "prob_noise_echo_only",
        "prob_target_noise",
        "prob_target_echo",
        "prob_target_noise_echo",
    ),
)
def test_aec_spatial_dynamic_branches_keep_mic_multichannel(
    tmp_path: Path, branch: str
):
    dataset = _spatial_aec_dataset(tmp_path)
    sample = dataset.get_sample_with_trace(0, force_branch=branch).sample

    assert isinstance(sample, tuple)
    assert sample[0].ndim == 1
    assert sample[1].shape == (2, sample[0].shape[-1])
    assert sample[2].shape == sample[0].shape
    batch = aec_collate_fn([sample])
    assert batch.nearend.shape == (1, sample[0].shape[-1])
    assert batch.mic.shape == (1, 2, sample[0].shape[-1])
    assert batch.farend.shape == batch.nearend.shape
    assert torch.isfinite(batch.mic).all()
    assert not torch.equal(batch.mic[0, 0], batch.mic[0, 1])


def test_aec_spatial_mic_extra_noise_preserves_observation_layout(tmp_path: Path):
    dataset = _spatial_aec_dataset(tmp_path)
    dataset._aec_pipe["mic_extra_noise_enabled"] = True
    mic = np.zeros((2, 2000), dtype=np.float32)
    result = dataset._maybe_mic_extra_noise_result(
        mic, 8000, 0, np.ones(2000, dtype=np.float32) * 0.1
    )
    assert result.mic.shape == mic.shape
    assert result.extra_noise is not None
    assert result.extra_noise.shape == (2000,)


def test_aec_static_spatial_scene_requests_nearend_and_loudspeaker_together(
    tmp_path: Path,
):
    dataset = _spatial_aec_dataset(tmp_path)
    calls = []

    def sample(plan, *, normalize, track_kinds):
        calls.append((plan.sample_rate, len(plan.actors), tuple(a.actor_uid for a in plan.actors)))
        assert track_kinds == ("full", "early")
        return _static_tracks(plan)

    provider = SimpleNamespace(sample_scene_tracks=sample)
    dataset.spatial_rir_provider = provider
    dataset._spatial_rir_provider = provider
    dataset.get_sample_with_trace(0, force_branch="prob_target_noise_echo")

    assert calls == [(8000, 2, ("speech", "__echo__"))]


def test_aec_moving_spatial_echo_requests_independent_actor_trajectories(tmp_path: Path):
    dataset = _spatial_aec_dataset(tmp_path)
    dataset.sim_cfg.prob_moving_rir = 1.0
    calls = []

    def provider(request, *, normalize=True):
        calls.append((request, normalize))
        raise MovingRIRCapabilityError("backend capability unavailable")

    provider = SimpleNamespace(sample_moving_scene_tracks=provider)
    dataset.spatial_rir_provider = provider
    dataset._spatial_rir_provider = provider

    with pytest.raises(RuntimeError, match="backend capability unavailable") as error:
        dataset.get_sample_with_trace(0, force_branch="prob_target_noise_echo")
    assert isinstance(error.value.__cause__, MovingRIRCapabilityError)
    assert len(calls) == 1
    request, normalize = calls[0]
    assert normalize is True
    assert request.source_uids == ("speech", "__echo__")
    assert tuple(request.actor_trajectories) == ("speech", "__echo__")
    assert request.actor_trajectories["speech"] is not request.actor_trajectories["__echo__"]
    assert request.actor_trajectories["speech"].timebase == request.timebase
    assert request.actor_trajectories["__echo__"].timebase == request.timebase
    np.testing.assert_array_equal(
        request.actor_trajectories["speech"].timebase.frame_samples,
        request.timebase.frame_samples,
    )
    assert request.room["room_uid"] == "room-fixture"
    echo_positions = request.actor_trajectories["__echo__"].positions_m
    np.testing.assert_array_equal(echo_positions[0], echo_positions[-1])
    np.testing.assert_array_equal(request.array["positions_m"], [[2, 2, 1], [2.08, 2, 1]])
    assert request.microphone_count == 2


@pytest.mark.parametrize(
    ("branch", "context_key"),
    (
        ("prob_echo_only", "prob_echo_only"),
        ("prob_noise_echo_only", "prob_noise_echo_only"),
        ("prob_target_echo", "prob_target_echo_disk"),
        ("prob_target_noise_echo", "prob_target_noise_echo_disk"),
    ),
)
def test_aec_echo_projection_slices_spatial_mic_on_time_axis(
    branch: str, context_key: str
):
    dataset = object.__new__(DynamicMixingDataset)
    dataset._pure_branch_bool_tensors = lambda _branch: {}
    dataset._finalize_sample = lambda mic, *_args, **_kwargs: mic
    mic = np.arange(6, dtype=np.float32).reshape(3, 2)
    disk = {
        context_key: {
            "farend": np.ones(2, dtype=np.float32),
            "echo": np.ones(2, dtype=np.float32),
        }
    }

    projected = dataset._assemble_training_tuple(
        np.zeros(2, dtype=np.float32),
        mic,
        8000,
        0,
        "utt",
        disk,
        np.random.default_rng(0),
        np.zeros(2, dtype=np.float32),
        branch=branch,
    )

    np.testing.assert_array_equal(projected, mic)


@pytest.mark.parametrize("branch", ["prob_target_only", "prob_target_echo", "prob_noise_echo_only"])
def test_aec_sampler_called_once_with_scene_roles(tmp_path, prepared_aec_scene, branch):
    dataset = _spatial_aec_dataset(tmp_path)
    result = dataset.get_sample_with_trace(0, force_branch=branch, capture_stems=True)
    assert len(prepared_aec_scene) == 1
    request = prepared_aec_scene[0]
    has_echo = "echo" in branch
    assert request["source_uids"] == (("speech", "__echo__") if has_echo else ("speech",))
    assert request["roles"] == (("target", "loudspeaker") if has_echo else ("target",))
    assert request["stationary_actor_uids"] == (("__echo__",) if has_echo else ())
    plan = result.trace.sources["scene"].metadata["spatial_scene_plan"]
    assert plan["room"]["room_uid"] == "room-fixture"
    assert plan["actors"][0]["position_m"] == [1.0, 3.0, 1.5]


def test_aec_spatial_trace_publishes_spatial_plan_without_capture(tmp_path, prepared_aec_scene):
    dataset = _spatial_aec_dataset(tmp_path)
    run_output = dataset._run(
        "speech", 4000, 8000, index=0, force_branch="prob_target_only"
    )
    assert run_output.context.acoustic_scene_plan is None
    assert run_output.context.spatial_scene_plan["room"]["room_uid"] == "room-fixture"

    traced = dataset.get_sample_with_trace(0, force_branch="prob_target_only")
    source_metadata = traced.trace.sources["scene"].metadata
    assert source_metadata["spatial_scene_plan"]["room"]["room_uid"] == "room-fixture"
    assert "acoustic_scene_plan" not in source_metadata


def test_aec_static_uses_explicit_early_asset(tmp_path):
    from core.data.simulation.rir_provider import SpatialRIRResult, SpatialRIRTrackSet
    dataset = _spatial_aec_dataset(tmp_path)
    dataset.task = "joint"

    def sample(plan, **kwargs):
        tracks = _static_tracks(plan)
        # full 在参考通道非零，独立 early 资产为零，检测是否错误重截 full。
        full = np.ones((len(plan.actors), 2, 1), np.float32)
        return SpatialRIRTrackSet(
            full=SpatialRIRResult(full, tracks.geometry, tracks.provenance, "fixture-v1"),
            early=SpatialRIRResult(np.zeros_like(full), tracks.geometry, tracks.provenance, "fixture-v1"),
        )

    dataset._spatial_rir_provider = SceneProvider(sample_scene_tracks=sample)
    target, mic, *_ = dataset.get_sample_with_trace(0, force_branch="prob_target_echo").sample
    assert torch.count_nonzero(target) == 0
    assert torch.count_nonzero(mic) > 0


def test_aec_static_uses_real_online_scene_sampler(tmp_path, monkeypatch):
    # 恢复共享 sampler，仅替换 RIR 数值生成，验证 room/position 在 provider 前固定。
    monkeypatch.undo()
    dataset = _spatial_aec_dataset(tmp_path)
    dataset.sim_cfg.rir_backend = "online"
    calls = []

    def sample(plan, **kwargs):
        calls.append(plan)
        assert kwargs == {"track_kinds": ("full", "early"), "normalize": True}
        return _static_tracks(plan)

    dataset._spatial_rir_provider = SceneProvider(sample_scene_tracks=sample)
    dataset.get_sample_with_trace(0, force_branch="prob_target_echo")
    assert len(calls) == 1
    plan = calls[0]
    room = np.asarray(plan.room["room_dim_m"])
    for actor in plan.actors:
        assert np.all(np.asarray(actor.position_m) > 0)
        assert np.all(np.asarray(actor.position_m) < room)
    assert not np.array_equal(plan.actors[0].position_m, plan.actors[1].position_m)
    assert np.all(np.asarray(plan.array.positions_m) > 0)


def test_aec_offline_uses_manifest_scene_without_sampling(tmp_path, monkeypatch):
    monkeypatch.undo()
    from core.data.simulation import scene_sampler
    dataset = _spatial_aec_dataset(tmp_path)
    positions = np.asarray(dataset._simulation_view.shared.spatial.microphone_positions) + [2, 2, 1]
    manifest = SimpleNamespace(
        actor_order=("speech", "__echo__"), scene_uid="offline-aec-scene",
        metadata={"sample_rate": 8000, "actor_positions_m": {
            "speech": [1.0, 3.0, 1.5], "__echo__": [4.0, 3.0, 1.0],
        }},
        geometry={"positions_m": positions.tolist(), "reference_channel": 1},
        room={"room_uid": "offline-room", "room_dim_m": [6, 6, 3], "rt60": 0.5},
    )
    calls = []

    def no_random_scene(*args, **kwargs):
        raise AssertionError("offline AEC must not sample new room or actor positions")

    def sample(plan, **kwargs):
        calls.append(plan)
        assert kwargs["normalize"] is False
        return _static_tracks(plan)

    monkeypatch.setattr(scene_sampler, "sample_static_scene_plan", no_random_scene)
    dataset._spatial_rir_provider = SceneProvider(scene_manifest=manifest, sample_scene_tracks=sample)
    dataset.get_sample_with_trace(0, force_branch="prob_target_echo")
    assert len(calls) == 1
    assert calls[0].identity["scene_uid"] == manifest.scene_uid
    assert calls[0].to_dict()["room"] == manifest.room
    np.testing.assert_array_equal(calls[0].array.positions_m, positions)
    assert tuple(calls[0].actors[1].position_m) == (4.0, 3.0, 1.0)


@pytest.mark.parametrize("bad", ["room", "scene_uid", "actor_order"])
def test_aec_static_rejects_other_scene_tracks(tmp_path, bad):
    from core.data.simulation.rir_provider import SpatialRIRResult, SpatialRIRTrackSet
    dataset = _spatial_aec_dataset(tmp_path)

    def sample(plan, **kwargs):
        tracks = _static_tracks(plan)
        provenance = dict(tracks.provenance)
        provenance[bad] = {"room_uid": "other"} if bad == "room" else "other"
        full = SpatialRIRResult(tracks.full.canonical_rirs, tracks.geometry, provenance, "fixture-v1")
        return SpatialRIRTrackSet(full=full, early=full)

    dataset._spatial_rir_provider = SceneProvider(sample_scene_tracks=sample)
    with pytest.raises(RuntimeError, match="AEC static RIR"):
        dataset.get_sample_with_trace(0, force_branch="prob_target_echo")
