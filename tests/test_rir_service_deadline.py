"""gpuRIR 服务 watchdog 的真实进程 deadline 合同。"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import socket
import time

import numpy as np
import pytest

from core.data.simulation.gpurir_backend import validate_rir_plan_budget
from core.data.simulation.rir_service import (
    OnlineRIRError,
    RIRClient,
    _prepare_tracks,
    _request_shape,
    _send,
    _server,
)


def _slow_backend_factory(plan):
    class Backend:
        def beta_SabineEstimation(self, room, rt60):
            return np.full(6, 0.7, dtype=np.float32)

        def att2t_SabineEstimator(self, attenuation, rt60):
            return 0.01

        def t2n(self, duration, room, c):
            return (2, 2, 2)

        def simulateRIR(self, **kwargs):
            time.sleep(float(plan.get("test_delay_sec", 0.0)))
            sources = len(kwargs["pos_src"])
            channels = len(kwargs["pos_rcv"])
            length = int(plan["num_samples"])
            result = np.zeros((sources, channels, length), dtype=np.float32)
            result[:, :, 1] = 1.0
            return result

    return Backend()


def _plan(delay=0.0):
    return {
        "room_dim": [3.0, 3.0, 2.5],
        "source_loc": [[1.0, 1.0, 1.2]],
        "mic_loc": [1.5, 1.5, 1.2],
        "num_samples": 80,
        "sample_rate": 8000,
        "rt60": 0.2,
        "diffuse_db": 15.0,
        "sound_speed": 340.0,
        "group_uid": f"deadline-{delay}",
        "test_delay_sec": delay,
    }


# 时限断言以配置的 deadline 为基准，slack 只吸收进程回收与 IPC 拆除开销，
# 不随机器负载漂移成硬编码常量。
_DEADLINE_TIMEOUT_SEC = 3.0
_DEADLINE_SLACK_SEC = 2.0


@pytest.mark.unit
@pytest.mark.subprocess
def test_generation_deadline_kills_worker_and_service_remains_usable(tmp_path: Path):
    context = mp.get_context("spawn")
    address = str(tmp_path / "service.sock")
    token = "test-token"
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_server,
        args=(address, token, "", 2, 2, _DEADLINE_TIMEOUT_SEC, child, _slow_backend_factory),
    )
    process.start()
    child.close()
    try:
        assert parent.poll(2.0)
        assert parent.recv_bytes() == b"OK"
        client = RIRClient(address, token, _DEADLINE_TIMEOUT_SEC)
        started = time.monotonic()
        with pytest.raises(OnlineRIRError, match="timed out|deadline|terminated"):
            client.generate(_plan(delay=6.0))
        assert time.monotonic() - started < _DEADLINE_TIMEOUT_SEC + _DEADLINE_SLACK_SEC

        wave, facts = client.generate(_plan(delay=0.0))
        assert wave.shape == (1, 80)
        assert facts["group_uid"] == "deadline-0.0"
        assert facts["provenance"]["timeout_sec"] == pytest.approx(client.timeout)
        assert facts["provenance"]["requested_timeout_sec"] == pytest.approx(client.timeout)
        assert 0.0 < facts["provenance"]["deadline_timeout_sec"] < client.timeout

        tracks = client.generate_tracks(_plan(delay=0.0), track_kind="full_direct")
        assert set(tracks) == {"full", "direct"}
        assert tracks["full"][0].shape == tracks["direct"][0].shape == (1, 80)
        for name in ("full", "direct"):
            provenance = tracks[name][1]["provenance"]
            assert provenance["timeout_sec"] == pytest.approx(client.timeout)
            assert provenance["requested_timeout_sec"] == pytest.approx(client.timeout)
            assert 0.0 < provenance["deadline_timeout_sec"] < client.timeout

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            sock.connect(address)
            _send(sock, json.dumps({"token": token, "action": "close"}).encode())
        process.join(1.0)
        assert not process.is_alive()
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(1.0)


@pytest.mark.unit
def test_full_direct_budget_counts_both_requested_tracks():
    plan = {**_plan(), "num_samples": 4}
    assert validate_rir_plan_budget(plan, max_bytes=16, track_count=1) == 16
    with pytest.raises(MemoryError, match="tracks"):
        validate_rir_plan_budget(plan, max_bytes=16, track_count=2)


@pytest.mark.unit
def test_service_response_shape_hash_and_budget_guards_fail_closed():
    plan = _plan()
    expected = _request_shape(plan, "full")
    wave = np.zeros(expected, dtype="<f4")
    wave[0, 1] = 1.0
    facts = {
        **plan,
        "track_kind": "full",
        "rir_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
    }

    with pytest.raises(OnlineRIRError, match="shape/value"):
        _prepare_tracks(
            {"full": (wave[:, :-1], facts)}, plan, "full", expected, False
        )
    with pytest.raises(OnlineRIRError, match="content identity"):
        _prepare_tracks(
            {"full": (wave, {**facts, "rir_sha256": "0" * 64})},
            plan,
            "full",
            expected,
            False,
        )
    with pytest.raises(MemoryError, match="exceeds memory budget"):
        _request_shape({**plan, "num_samples": 70_000_000}, "full")
