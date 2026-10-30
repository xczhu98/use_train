"""AEC moving actor 波形、能量标定及数据交付合同。"""

import numpy as np
import pytest
import torch
import tasks.aec.data.dynamic_dataset as aec_module

from test_aec_spatial_dynamic_branches import _spatial_aec_dataset, prepared_aec_scene
from types import SimpleNamespace
from core.data.simulation.rir_provider import MovingSpatialRIRTrackSet
from tasks.aec.data.collate import aec_collate_fn
from tasks.aec.data.dynamic_dataset import _render_aec_moving_actor

pytestmark = pytest.mark.unit


def _tracks(request):
    """使用 sampler 提供的房间绝对坐标，不再次平移。"""
    positions = np.concatenate([
        value.positions_m for value in request.actor_trajectories.values()
    ], axis=1)
    plan = {
        "room_dim": list(request.room["room_dim_m"]), "sample_rate": request.sample_rate,
        "num_samples": 16, "group_uid": "aec-moving-fixture",
        "scene_uid": request.scene_uid,
        "source_ids": list(request.source_uids),
        "trajectory": positions.tolist(), "source_loc": positions[0].tolist(),
        "trajectory_timebase": request.timebase.to_dict(),
        "mic_loc": np.array(request.array["positions_m"]).tolist(),
    }
    full = np.zeros((request.timebase.frame_count, request.actor_count, 2, 16), np.float32)
    for point in range(len(full)):
        full[point, 0, :, 0] = [0.5, 1.0]
        if request.actor_count > 1:
            full[point, 1, :, 2] = np.array([1.5, 0.75]) * (point + 1)
    direct = full.copy()
    full[..., 12] = 0.1
    return MovingSpatialRIRTrackSet(full, direct, plan, {"group_uid": plan["group_uid"]})


def _dataset(tmp_path):
    dataset = _spatial_aec_dataset(tmp_path)
    dataset.sim_cfg.prob_moving_rir = 1.0
    dataset.sim_cfg.early_rir_sec = 0.001
    dataset._apply_ser = True
    dataset._ser_low = dataset._ser_high = 6.0
    dataset._erl_min_db = dataset._erl_max_db = 3.0
    dataset._far_db_lo, dataset._far_db_hi = -60.0, 0.0
    calls = []

    def provider(request, *, normalize):
        tracks = _tracks(request)
        calls.append((request, tracks))
        return tracks

    dataset._spatial_rir_provider = SimpleNamespace(sample_moving_scene_tracks=provider)
    return dataset, calls


@pytest.mark.parametrize("branch", [
    "prob_target_only", "prob_noise_only", "prob_target_noise",
    "prob_echo_only", "prob_noise_echo_only", "prob_target_echo", "prob_target_noise_echo",
])
def test_moving_dataset_delivers_all_branches(tmp_path, branch):
    dataset, calls = _dataset(tmp_path)
    result = dataset.get_sample_with_trace(0, force_branch=branch, capture_stems=True)
    assert len(calls) == 1
    batch = aec_collate_fn([result.sample])
    assert batch.mic.shape == (1, 2, batch.nearend.shape[-1])
    assert batch.farend.shape == batch.nearend.shape
    assert torch.isfinite(batch.mic).all()
    scene = result.trace.sources["scene"]
    stems = scene.stems
    assert stems["moving_nearend_full_pre_mix"].shape == (2, 2000)
    assert not np.array_equal(stems["moving_nearend_full_pre_mix"], stems["moving_nearend_early_pre_mix"])
    if "echo" in branch:
        meta = scene.metadata["moving_scene"]
        assert meta["source_order"] == ["speech", "__echo__"]
        assert meta["echo_projection_gain"] > 0
        echo = stems["moving_echo_full_pre_mix"]
        assert echo.shape == (2, 2000)
        assert not np.array_equal(echo, stems["moving_echo_direct_pre_mix"])
        assert not np.array_equal(echo[0], echo[1])


def test_moving_echo_reference_preserves_ser_erl_and_observation(tmp_path):
    dataset, _ = _dataset(tmp_path)
    dataset.task = "aec"
    result = dataset.get_sample_with_trace(0, force_branch="prob_target_echo", capture_stems=True)
    scene = result.trace.sources["scene"]
    near, mic, far = (value.numpy() for value in result.sample[:3])
    echo = mic[1] - near
    power = lambda value: np.mean(value.astype(np.float64) ** 2)
    # SER 的既有标定参考是 early target；full target 仅用于 task=aec 交付。
    stems = scene.stems
    target = stems["moving_nearend_early_pre_mix"][1]
    pre_echo = stems["moving_echo_full_pre_mix"][1]
    assert 10 * np.log10(power(target) / power(pre_echo)) == pytest.approx(6.0, abs=0.01)
    assert 10 * np.log10(power(far) / power(echo)) == pytest.approx(3.0, abs=0.01)
    expected = stems["moving_nearend_full_pre_mix"] + stems["moving_echo_full_pre_mix"]
    # Layer D 只允许共同增益；混合必须保留两路独立空间图像。
    gain = np.sum(mic * expected) / np.sum(expected ** 2)
    np.testing.assert_allclose(mic, expected * gain, atol=2e-6)


@pytest.mark.parametrize("bad", ["order", "clock", "geometry", "trajectory", "scene", "room", "type"])
def test_moving_returned_tracks_must_match_request(tmp_path, bad):
    dataset, calls = _dataset(tmp_path)
    dataset.get_sample_with_trace(0, force_branch="prob_target_echo")
    request, tracks = calls[0]
    if bad == "order":
        tracks.plan["source_ids"].reverse()
    elif bad == "clock":
        tracks.plan["trajectory_timebase"]["num_samples"] += 1
    elif bad == "geometry":
        tracks.plan["mic_loc"][1][0] += 0.1
    elif bad == "scene":
        tracks.plan["scene_uid"] = "other-scene"
    elif bad == "room":
        tracks.plan["room_dim"][0] += 1
    elif bad == "trajectory":
        tracks.plan["trajectory"][1][1][0] += 0.1
    else:
        tracks = np.zeros((4, 2, 2, 16), np.float32)
    with pytest.raises((ValueError, TypeError), match="AEC moving"):
        _render_aec_moving_actor(
            tracks, request, np.ones(request.num_samples), "__echo__",
            early_boundary=8, reference_channel=1,
        )


@pytest.mark.parametrize("clip_path", [None, "post_ser_distortion", "pre_ser_overload", "synced_ref_clip"])
def test_moving_projection_preserves_processed_echo_power(tmp_path, monkeypatch, clip_path):
    dataset, _ = _dataset(tmp_path)
    dataset._aec_ref_echo_random_delay_prob = 1.0
    dataset._aec_ref_echo_min_delay_sec = dataset._aec_ref_echo_max_delay_sec = 0.01
    if clip_path:
        dataset._aec_echo_clipping_enabled = True
        dataset._aec_echo_clipping_prob = 1.0
        dataset._aec_echo_clipping_path_probs = {clip_path: 1.0}
    original = aec_module._render_aec_moving_actor
    rendered = {}

    def capture(tracks, request, waveform, actor_uid, **kwargs):
        result = original(tracks, request, waveform, actor_uid, **kwargs)
        rendered[actor_uid] = (np.asarray(waveform).copy(), result)
        return result

    monkeypatch.setattr(aec_module, "_render_aec_moving_actor", capture)
    result = dataset.get_sample_with_trace(0, force_branch="prob_target_echo", capture_stems=True)
    scene = result.trace.sources["scene"]
    waveform, raw = rendered["__echo__"]
    assert np.count_nonzero(waveform[:80]) == 0
    echo = scene.stems["moving_echo_full_pre_mix"]
    np.testing.assert_allclose(np.mean(echo[1] ** 2), np.mean(waveform ** 2), rtol=2e-6)
    gain = scene.metadata["moving_scene"]["echo_projection_gain"]
    for kind in ("full", "direct", "early"):
        np.testing.assert_allclose(
            scene.stems[f"moving_echo_{kind}_pre_mix"],
            getattr(raw, f"{kind}_tracks")[0] * gain, atol=1e-6,
        )


def test_moving_echo_empty_reference_fails_without_broadcast(tmp_path):
    dataset, _ = _dataset(tmp_path)

    def provider(request, *, normalize):
        tracks = _tracks(request)
        full, direct = tracks.full.copy(), tracks.direct.copy()
        full[:, 1, 1] = direct[:, 1, 1] = 0
        return MovingSpatialRIRTrackSet(full, direct, tracks.plan, tracks.track_facts)

    dataset._spatial_rir_provider = SimpleNamespace(sample_moving_scene_tracks=provider)
    with pytest.raises(RuntimeError, match="reference channel has no usable energy"):
        dataset.get_sample_with_trace(0, force_branch="prob_target_echo")


def test_actor_waveforms_match_independent_segment_convolution(tmp_path):
    dataset, calls = _dataset(tmp_path)
    dataset.get_sample_with_trace(0, force_branch="prob_target_echo")
    request, tracks = calls[0]
    source = np.zeros(request.num_samples, np.float32)
    source[request.timebase.frame_samples[1] - 1] = 1
    source[request.timebase.frame_samples[1]] = 2
    source[-1] = 3
    for actor_index, uid in enumerate(request.source_uids):
        result = _render_aec_moving_actor(
            tracks, request, source, uid, early_boundary=8, reference_channel=1,
        )
        for kind, bank in (("full", tracks.full), ("direct", tracks.direct), ("early", tracks.full[..., :8])):
            expected = np.zeros((2, request.num_samples), np.float32)
            for point, start in enumerate(request.timebase.frame_samples):
                stop = request.timebase.frame_samples[point + 1] if point + 1 < request.timebase.frame_count else len(source)
                for channel in range(2):
                    convolution = np.convolve(source[start:stop], bank[point, actor_index, channel])
                    count = min(len(convolution), len(source) - start)
                    expected[channel, start:start + count] += convolution[:count]
            np.testing.assert_allclose(getattr(result, f"{kind}_tracks")[0], expected, atol=1e-6)


def test_aec_moving_uses_real_sampler_with_stationary_echo(tmp_path, monkeypatch):
    monkeypatch.undo()
    dataset, calls = _dataset(tmp_path)
    dataset.sim_cfg.rir_backend = "online"
    dataset.get_sample_with_trace(0, force_branch="prob_target_echo")
    assert len(calls) == 1
    request, tracks = calls[0]
    assert request.actor_roles == ("target", "loudspeaker")
    echo = request.actor_trajectories["__echo__"].positions_m
    near = request.actor_trajectories["speech"].positions_m
    np.testing.assert_array_equal(echo[0], echo[-1])
    assert not np.array_equal(near[0], near[-1])
    np.testing.assert_array_equal(tracks.plan["mic_loc"], request.array["positions_m"])
    assert request.scene_uid
