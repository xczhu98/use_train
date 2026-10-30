"""数据仿真审计修复的回归测试。"""

from __future__ import annotations

import hashlib
import json
import logging

import numpy as np
import pytest
import scipy.signal

from core.data.simulation import scene_manifest, urgent_impl
from core.data.simulation.rir_service import OnlineRIRError, _prepare_tracks
from core.data.simulation.scene_manifest import OfflineSceneIndex
from core.data.simulation.urgent_impl import add_moving_reverberation
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_utils import align_to_max_duration

pytestmark = pytest.mark.unit


def _cfg_with(**updates):
    values = {"noisy_post_mix_degradations_enabled": False}
    values.update(updates)
    return SimulationConfigs.from_dict(values)


def test_moving_rir_constant_sequence_matches_static_convolution():
    speech = np.array([[1.0, -0.5, 0.25, 0.75, -0.25, 0.5]], dtype=np.float32)
    rir = np.array([1.0, 0.4, -0.2, 0.1], dtype=np.float32)
    actual = add_moving_reverberation(
        speech, np.stack([rir, rir]), segment_sec=0.2, fs=10, normalize_rir=False
    )
    expected = scipy.signal.convolve(speech, rir.reshape(1, -1), mode="full")[:, : speech.shape[1]]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_moving_rir_preserves_long_tail_across_segments():
    speech = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    rir = np.array([1.0, 0.0, 0.0, 0.75], dtype=np.float32)
    actual = add_moving_reverberation(
        speech, np.stack([rir, rir]), segment_sec=0.2, fs=10, normalize_rir=False
    )
    np.testing.assert_allclose(actual, np.array([[1.0, 0.0, 0.0, 0.75]], dtype=np.float32))


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.01, 1.01])
def test_clip_hard_portion_rejects_invalid_probability(value):
    with pytest.raises(ValueError, match="clip_hard_portion"):
        _cfg_with(clip_hard_portion=value)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("overlap_ratio_range", [-0.1, 0.2]),
        ("overlap_ratio_range", [0.2, 1.1]),
        ("spectral_holes_cutoff_freq", -1.0),
        ("spectral_holes_cutoff_freq", float("nan")),
        ("bass_boost_attenuation_min_db", 0.0),
    ],
)
def test_layer_f_domain_validation_rejects_invalid_values(key, value):
    with pytest.raises(ValueError, match=key):
        _cfg_with(**{key: value})


def test_align_to_max_duration_rejects_mismatched_tracks():
    with pytest.raises(ValueError, match="same length"):
        align_to_max_duration(np.zeros(4), np.zeros(3), 8)


def test_removed_moving_reverberation_overlap_sec_is_rejected():
    speech = np.arange(8, dtype=np.float32).reshape(1, -1) / 8.0
    rir_seq = np.stack([np.array([1.0, 0.3, 0.1], dtype=np.float32)] * 3)
    with pytest.raises(TypeError, match="overlap_sec"):
        add_moving_reverberation(
            speech, rir_seq, 0.2, 8, normalize_rir=False, overlap_sec=0.02
        )


def test_bandwidth_limitation_and_rir_start_sample_fail_fast_without_asserts():
    waveform = np.zeros((1, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="greater than fs_new"):
        urgent_impl.bandwidth_limitation(waveform, fs=16000, fs_new=32000)
    with pytest.raises(ValueError, match="level_ratio"):
        urgent_impl.get_rir_start_sample(np.zeros(8, dtype=np.float32), level_ratio=1.0)
    with pytest.raises(ValueError, match="Too many channels"):
        urgent_impl.get_rir_start_sample(np.zeros((20, 8), dtype=np.float32))


def test_codec_compression_rejects_unknown_format_and_encoder():
    waveform = np.zeros((1, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="Unsupported format"):
        urgent_impl.codec_compression_result(waveform, 16000, "wav")
    with pytest.raises(ValueError, match="Unsupported encoder"):
        urgent_impl.codec_compression_result(waveform, 16000, "mp3", encoder="aac")


def test_codec_failure_is_logged_with_traceback(monkeypatch, caplog):
    waveform = np.zeros((1, 8), dtype=np.float32)

    class _BrokenEffector:
        def __init__(self, **kwargs):
            pass

        def apply(self, samples, fs):
            raise RuntimeError("codec failed")

    monkeypatch.setattr(urgent_impl, "HAS_TORCHAUDIO_IO", True)
    monkeypatch.setattr(urgent_impl, "CodecConfig", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(urgent_impl, "AudioEffector", _BrokenEffector, raising=False)
    with caplog.at_level(logging.WARNING, logger=urgent_impl.__name__):
        result = urgent_impl.codec_compression_result(waveform, 16000, "mp3")
    assert result.reason == "codec_error" and result.observation_changed is False
    warnings = [record for record in caplog.records if "Codec compression failed" in record.message]
    assert warnings and warnings[0].exc_info is not None


def _moving_metadata(*, room_dim_m):
    trajectory = {actor: [[[1.0, 1.0, 1.0]]] for actor in ("a",)}
    metadata = {
        "schema_version": 3,
        "scene_uid": "scene-1",
        "actor_order": ["a"],
        "room": {"room_uid": "room-1"},
        "geometry": {"positions_m": [[1.0, 1.0, 1.0]], "reference_channel": 0, "channel_order": ["mic_0"]},
        "generator_version": "fixture",
        "track_identities": {"a": {"full": "full.npy", "direct": "direct.npy", "early": "early.npy"}},
        "sample_rate": 16000,
        "actor_positions_m": {"a": [1.0, 1.0, 1.0]},
        "moving": True,
        "actor_trajectories": trajectory,
        "trajectory_timebase": {
            "sample_rate": 16000, "num_samples": 16, "hop_samples": 16,
            "frame_count": 1, "start_sample": 0,
        },
    }
    metadata["trajectory_sha256"] = scene_manifest.trajectory_digest(
        trajectory, metadata["trajectory_timebase"]
    )
    if room_dim_m is not None:
        metadata["room"]["room_dim_m"] = list(room_dim_m)
    return metadata


def test_moving_manifest_requires_room_dim_m():
    with pytest.raises(ValueError, match="room_dim_m"):
        scene_manifest._validate_moving_metadata(_moving_metadata(room_dim_m=None))
    scene_manifest._validate_moving_metadata(_moving_metadata(room_dim_m=(4.0, 4.0, 3.0)))


def test_scene_index_rejects_manifest_path_escaping_index_directory(tmp_path):
    escaped = tmp_path.parent / "escaped_scene.json"
    escaped.write_text("{}", encoding="utf-8")
    index = tmp_path / "scenes.jsonl"
    index.write_text(
        json.dumps({"metadata": {"scene_uid": "s1"}, "scene_manifest_path": "../escaped_scene.json"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes the index directory"):
        OfflineSceneIndex.load(index)

    inside = tmp_path / "scene.json"
    inside.write_text("{}", encoding="utf-8")
    index.write_text(
        json.dumps({"metadata": {"scene_uid": "s1"}, "scene_manifest_path": "scene.json"}) + "\n",
        encoding="utf-8",
    )
    records = OfflineSceneIndex.load(index).records
    assert records["s1"]["scene_manifest_path"] == str(inside.resolve())


def test_scene_index_rejects_actor_roles_shorter_than_actor_order(tmp_path):
    index = tmp_path / "scenes.jsonl"
    index.write_text(
        json.dumps({
            "metadata": {"scene_uid": "s1", "actor_order": ["a", "b"], "actor_roles": ["target"]},
            "scene_manifest_path": "scene.json",
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="actor_roles must match actor_order"):
        OfflineSceneIndex.load(index)


def test_build_track_identity_reads_npy_once_and_rejects_empty_bundle(tmp_path, monkeypatch):
    value = np.arange(8, dtype=np.float32).reshape(2, 4)
    np.save(tmp_path / "track.npy", value)
    calls = []
    original_load = np.load

    def counting_load(*args, **kwargs):
        calls.append(args[0])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counting_load)
    identity = scene_manifest.build_track_identity(tmp_path / "track.npy")
    assert calls == [tmp_path / "track.npy"]
    monkeypatch.undo()

    expected = hashlib.sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()
    assert identity == expected
    np.savez(tmp_path / "bundle.npz", first=value)
    assert scene_manifest.build_track_identity(tmp_path / "bundle.npz") == expected
    np.savez(tmp_path / "empty.npz")
    with pytest.raises(ValueError, match="no arrays"):
        scene_manifest.build_track_identity(tmp_path / "empty.npz")


def _prepared_inputs(*, moving):
    shape = (4, 1, 2, 4) if moving else (1, 2, 4)
    plan = {"group_uid": "g", "num_samples": shape[-1], "trajectory_timebase": {"num_samples": 16}}
    plan["trajectory"] = [[[0.0, 0.0, 0.0]]]
    generated = {}
    for offset, name in enumerate(("full", "direct")):
        wave = np.arange(int(np.prod(shape)), dtype="<f4").reshape(shape) + offset * 100.0
        facts = {
            "group_uid": "g",
            "track_kind": name,
            "rir_sha256": hashlib.sha256(np.ascontiguousarray(wave, dtype="<f4").tobytes()).hexdigest(),
        }
        if moving:
            facts["trajectory_timebase"] = plan["trajectory_timebase"]
            facts["trajectory"] = plan["trajectory"]
        generated[name] = (wave, facts)
    return generated, plan, shape


@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("hit", [False, True])
def test_prepare_tracks_publishes_accurate_identity_on_cache_hit(moving, hit):
    generated, plan, shape = _prepared_inputs(moving=moving)
    track_kind = "moving_full_direct" if moving else "full_direct"
    result, infos = _prepare_tracks(generated, plan, track_kind, shape, hit)
    for name, (wave, facts) in result.items():
        digest = hashlib.sha256(np.ascontiguousarray(wave, dtype="<f4").tobytes()).hexdigest()
        assert facts["rir_sha256"] == digest
        assert facts["cache_hit"] is hit
        frames = wave if moving else (wave,)
        assert infos[name]["frame_hashes"] == [
            hashlib.sha256(np.ascontiguousarray(frame, dtype="<f4").tobytes()).hexdigest()
            for frame in frames
        ]


def test_prepare_tracks_rejects_mismatched_digest_on_generation():
    generated, plan, shape = _prepared_inputs(moving=False)
    generated["direct"] = (generated["direct"][0], {**generated["direct"][1], "rir_sha256": "0" * 64})
    with pytest.raises(OnlineRIRError, match="content identity"):
        _prepare_tracks(generated, plan, "full_direct", shape, False)


def test_generation_pool_keeps_in_flight_failure_when_worker_stop_fails(monkeypatch):
    """worker 回收失败不得覆盖 in-flight 的超时根因，池仍必须 fail-closed。"""
    from types import SimpleNamespace

    from core.data.simulation import rir_service

    class _Parent:
        def poll(self, timeout):
            return False

        def send_bytes(self, payload):
            pass

        def recv_bytes(self):
            raise AssertionError("no payload expected on the timeout path")

        def close(self):
            pass

    class _Child:
        def close(self):
            pass

    class _Process:
        pid = 4321

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

        def close(self):
            pass

    context = SimpleNamespace(
        Pipe=lambda duplex=False: (_Parent(), _Child()),
        Process=lambda target=None, args=(): _Process(),
    )
    monkeypatch.setattr(rir_service, "mp", SimpleNamespace(get_context=lambda method: context))

    def _unrecoverable(process):
        raise rir_service.OnlineRIRError("gpuRIR backend worker could not be stopped")

    monkeypatch.setattr(rir_service, "_stop_worker", _unrecoverable)
    pool = rir_service._GenerationPool(1)
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            pool.generate({"group_uid": "g"}, "full", (1, 4), 0.01)
    finally:
        monkeypatch.setattr(rir_service, "_stop_worker", lambda process: None)
        pool.close()
