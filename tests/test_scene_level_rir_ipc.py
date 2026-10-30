"""scene-level RIR IPC canonical request and multi-actor response contracts."""
import hashlib
import json
import socket
import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest

from core.data.simulation import rir_service as service


class _FakeGpuRIR:
    def beta_SabineEstimation(self, room, rt60):
        return np.ones(6, dtype=np.float32) * 0.5

    def att2t_SabineEstimator(self, diffuse_db, rt60):
        return 0.001

    def t2n(self, diffuse, room, c=343.0):
        return (1, 1, 1)

    def simulateRIR(self, *, pos_src, pos_rcv, **kwargs):
        sources = np.asarray(pos_src, dtype=np.float32)
        microphones = np.asarray(pos_rcv, dtype=np.float32)
        length = int(round(kwargs["Tmax"] * kwargs["fs"]))
        out = np.zeros((len(sources), len(microphones), length), dtype=np.float32)
        for source_index in range(len(sources)):
            for mic_index in range(len(microphones)):
                out[source_index, mic_index, 0] = 1.0 + source_index * 10 + mic_index
        return out


def _fake_backend_factory(_plan):
    return _FakeGpuRIR()


def _scene(moving=False):
    actors = []
    for index, uid in enumerate(("nearend", "loudspeaker")):
        actor = {"actor_uid": uid, "role": uid, "position_m": [1.0 + index, 1.0, 1.0]}
        if moving:
            actor["trajectory"] = [[[1.0 + index + frame * .01, 1.0, 1.0]] for frame in range(3)]
            actor["position_m"] = actor["trajectory"][0][0]
        actors.append(actor)
    scene = {"scene_uid": "scene-ipc", "sample_rate": 8000, "num_samples": 32,
             "room": {"room_dim_m": [4, 4, 3], "rt60": .3,
                      "diffuse_db": -40.0, "rir_num_samples": 48},
             "array": {"positions_m": [[2, 2, 1], [2.04, 2, 1]], "reference_channel": 0},
             "actors": actors}
    if moving:
        scene["timebase"] = {"sample_rate": 8000, "num_samples": 32, "hop_samples": 8, "frame_count": 3, "start_sample": 0}
    return scene


@contextmanager
def _server(tmp_path, backend_factory):
    address = str(tmp_path / "scene.sock")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(address); listener.listen(4)
        def run():
            service._serve_requests(listener, "token", None, 4, 10.0,
                                    backend_factory=backend_factory)
        thread = threading.Thread(target=run, daemon=True); thread.start()
        try:
            yield service.RIRClient(address, "token", 10.0)
        finally:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(address); service._send(sock, json.dumps({"token": "token", "action": "close"}).encode())
            thread.join(3)


@pytest.mark.unit
def test_canonical_scene_request_contains_actor_identity():
    result = service._canonical_scene_request("scene_static", _scene())
    assert result["plan"]["source_ids"] == ["nearend", "loudspeaker"]
    assert result["plan"]["scene_uid"] == "scene-ipc"
    with pytest.raises(service.OnlineRIRError, match="position_m"):
        bad = _scene(); bad["actors"][0].pop("position_m")
        service._canonical_scene_request("scene_static", bad)

@pytest.mark.unit
def test_canonical_moving_scene_requires_shared_timebase():
    with pytest.raises(service.OnlineRIRError, match="timebase"):
        service._canonical_scene_request("scene_moving", _scene(moving=True) | {"timebase": None})

@pytest.mark.unit
@pytest.mark.parametrize("fault", ["clock", "position", "microphone", "actor_shape"])
def test_scene_rejects_inconsistent_physics_before_socket(fault):
    scene = _scene(moving=True)
    if fault == "clock":
        scene["timebase"]["num_samples"] = 64
    elif fault == "position":
        scene["actors"][0]["trajectory"][1][0][0] = -1
    elif fault == "microphone":
        scene["array"]["positions_m"][0][0] = -1
    else:
        scene["actors"][0]["trajectory"].pop()
    with pytest.raises((ValueError, service.OnlineRIRError)):
        service._canonical_scene_request("scene_moving", scene)

@pytest.mark.unit
def test_client_scene_request_roundtrip_with_real_server_fake_gpurir(tmp_path):
    with _server(tmp_path, _fake_backend_factory) as client:
        result = client.generate_scene(_scene(), moving=False)
        assert result["full"][0].shape == (2, 2, 48)
        moving = client.generate_scene(_scene(moving=True), moving=True)
        assert moving["full"][0].shape == (3, 2, 2, 48)


@pytest.mark.unit
def test_client_scene_request_can_skip_unused_direct_track(tmp_path):
    with _server(tmp_path, _fake_backend_factory) as client:
        result = client.generate_scene(_scene(), moving=False, track_kind="full")
        assert set(result) == {"full"}
        assert result["full"][0].shape == (2, 2, 48)


@pytest.mark.unit
def test_client_moving_scene_request_can_skip_unused_direct_track(tmp_path):
    with _server(tmp_path, _fake_backend_factory) as client:
        result = client.generate_scene(_scene(moving=True), moving=True,
                                       track_kind="scene_moving_full")
        assert set(result) == {"full"}
        assert result["full"][0].shape == (3, 2, 2, 48)
