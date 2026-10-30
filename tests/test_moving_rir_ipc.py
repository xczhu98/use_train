"""moving RIR 的真实本机 IPC unit 门禁；backend 使用 fake，不计入 CUDA 验证。"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import socket
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from core.data.simulation import gpurir_backend as backend
from core.data.simulation import rir_service as service



def _plan(moving=True):
    plan = {
        'group_uid': 'ipc-scene', 'sample_rate': 8000, 'num_samples': 256,
        'room_dim': [4, 4, 3], 'source_loc': [[1, 1, 1]],
        'mic_loc': [[2, 2, 1], [2.04, 2, 1]], 'source_ids': ['speaker-a'],
    }
    if moving:
        plan.update({
            'trajectory': [[[1 + index / 10, 1, 1]] for index in range(4)],
            'trajectory_timebase': {
                'sample_rate': 8000, 'num_samples': 8000, 'hop_samples': 2000,
                'frame_count': 4, 'start_sample': 0,
            },
        })
    return plan


def _generated(plan, moving):
    shape = (1, 2, plan['num_samples'])
    if moving:
        shape = (len(plan['trajectory']), *shape)
    result = {}
    for index, name in enumerate(('full', 'direct')):
        wave = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + index + 1) / 100
        facts = {
            'group_uid': plan['group_uid'], 'track_kind': name,
            'rir_sha256': hashlib.sha256(wave.tobytes()).hexdigest(),
        }
        if moving:
            facts.update({key: deepcopy(plan[key]) for key in ('trajectory', 'trajectory_timebase')})
        result[name] = (wave, facts)
    return result


@pytest.fixture
def fake_backend(monkeypatch):
    calls = []
    # 仅替换 backend；共享服务循环、JSON framing 和客户端均执行生产代码。
    def validate(plan, *, max_bytes, track_count=2):
        calls.append(('validate', max_bytes))
        return np.asarray(plan['trajectory'], dtype=np.float32)
    def moving(plan, gpu_rir, *, timeout):
        calls.append(('moving', timeout))
        return _generated(plan, True)
    def moving_full(plan, gpu_rir, *, timeout):
        calls.append(('moving_full', timeout))
        return _generated(plan, True)['full']
    def paired(plan, gpu_rir, *, timeout):
        calls.append(('paired', timeout))
        return _generated(plan, False)
    def full(plan, gpu_rir, *, timeout):
        calls.append(('full', timeout))
        return _generated(plan, False)['full']
    monkeypatch.setattr(backend, 'validate_moving_rir_plan', validate, raising=False)
    monkeypatch.setattr(backend, 'generate_moving_rir_tracks', moving)
    monkeypatch.setattr(backend, 'generate_moving_rir_group', moving_full)
    monkeypatch.setattr(backend, 'generate_rir_tracks', paired)
    monkeypatch.setattr(backend, 'generate_rir_group', full)
    return calls


@contextmanager
def _local_server(tmp_path, *, handler=None, timeout=2.0, gpu_rir=None):
    address = str(tmp_path / 'rir.sock')
    failures = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(address)
        listener.listen(8)
        def run():
            try:
                if handler is None:
                    service._serve_requests(listener, 'fixture', gpu_rir, 8, timeout)
                else:
                    conn, _ = listener.accept()
                    with conn:
                        request = json.loads(service._receive(conn, time.monotonic() + 2))
                        handler(conn, request)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except BaseException as exc:
                failures.append(exc)
        thread = threading.Thread(target=run, daemon=True)
        # 本组验证 wire/cache 协议，fake backend 必须留在本进程；spawn watchdog
        # 由独立 deadline tests 与真实 GPU gate 覆盖，不能依赖跨进程 monkeypatch。
        def generate_inline(plan, kind, remaining, _backend_factory, _requested_timeout_sec):
            if kind == 'moving_full_direct':
                return backend.generate_moving_rir_tracks(plan, gpu_rir, timeout=remaining)
            if kind == 'moving_full':
                return {'full': backend.generate_moving_rir_group(plan, gpu_rir, timeout=remaining)}
            if kind == 'full_direct':
                return backend.generate_rir_tracks(plan, gpu_rir, timeout=remaining)
            return {'full': backend.generate_rir_group(plan, gpu_rir, timeout=remaining)}
        patcher = patch.object(service, '_run_backend', side_effect=generate_inline)
        patcher.start()
        thread.start()
        try:
            yield service.RIRClient(address, 'fixture', timeout)
        finally:
            if handler is None and thread.is_alive():
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                    conn.settimeout(2)
                    conn.connect(address)
                    service._send(conn, json.dumps({'token': 'fixture', 'action': 'close'}).encode())
            thread.join(3)
            patcher.stop()
            assert not thread.is_alive(), 'IPC fixture failed to stop'
            (tmp_path / 'rir.sock').unlink(missing_ok=True)
            assert not failures, failures


@pytest.mark.unit
def test_moving_roundtrip_chunks_each_frame_and_isolates_cache(tmp_path, monkeypatch, fake_backend):
    monkeypatch.setattr(service, '_MAX_FRAME', 4096)
    sent = []
    original = service._send
    def observe(sock, payload, deadline=None):
        sent.append(len(payload))
        return original(sock, payload, deadline)
    monkeypatch.setattr(service, '_send', observe)
    with _local_server(tmp_path) as client:
        for plan, kind in (
            (_plan(False), 'full'),
            (_plan(False), 'full_direct'),
            (_plan(), 'moving_full_direct'),
            (_plan(), 'moving_full'),
        ):
            for hit in (False, True):
                result = client.generate(plan) if kind == 'full' else client.generate_tracks(plan, track_kind=kind)
                if kind == 'full':
                    assert result[1]['cache_hit'] is hit
                else:
                    expected = _generated(plan, kind.startswith('moving'))
                    for name, (wave, facts) in result.items():
                        np.testing.assert_array_equal(wave, expected[name][0])
                        assert facts['cache_hit'] is hit
        altered = _plan()
        altered['trajectory'][1][0][0] += 0.1
        changed = client.generate_tracks(altered, track_kind='moving_full_direct')
        assert not changed['full'][1]['cache_hit']
    assert [kind for kind, _ in fake_backend if kind != 'validate'] == [
        'full', 'paired', 'moving', 'moving_full', 'moving'
    ]
    assert all(size <= 4096 for size in sent)
    # 每次 moving 请求两轨四帧，不允许整轨 8192 bytes 一包发送。
    assert sent.count(2048) >= 24


@pytest.mark.parametrize('limit,frame_limit', [(16000, 4096), (256 * 1024 * 1024, 2047)])
@pytest.mark.unit
def test_budget_rejected_before_socket_and_backend(monkeypatch, fake_backend, limit, frame_limit):
    monkeypatch.setattr(service, '_MAX_TRACK_BYTES', limit)
    monkeypatch.setattr(service, '_MAX_FRAME', frame_limit)
    client = service.RIRClient('/does/not/exist', 'fixture', 1)
    with pytest.raises(service.OnlineRIRError, match='memory budget'):
        client.generate_tracks(_plan(), track_kind='moving_full_direct')
    assert fake_backend == [('validate', limit)]


@pytest.mark.unit
def test_moving_full_only_uses_single_track_budget():
    plan = _plan()
    with pytest.raises(MemoryError, match='trajectory frame/byte budget'):
        backend.validate_moving_rir_plan(plan, max_bytes=10_000, track_count=2)
    trajectory = backend.validate_moving_rir_plan(
        plan, max_bytes=10_000, track_count=1
    )
    assert trajectory.shape == (4, 1, 3)


@pytest.mark.unit
def test_backend_error_propagates_and_does_not_poison_cache(tmp_path, monkeypatch, fake_backend):
    valid = backend.generate_moving_rir_tracks
    def broken(*args, **kwargs):
        raise RuntimeError('backend trajectory rejected')
    monkeypatch.setattr(backend, 'generate_moving_rir_tracks', broken)
    with _local_server(tmp_path) as client:
        with pytest.raises(service.OnlineRIRError, match='backend trajectory rejected'):
            client.generate_tracks(_plan(), track_kind='moving_full_direct')
        monkeypatch.setattr(backend, 'generate_moving_rir_tracks', valid)
        assert not client.generate_tracks(_plan(), track_kind='moving_full_direct')['full'][1]['cache_hit']


def _peer_response(conn, request, change=None, *, delay=0):
    plan = request['plan']
    expected = (4, 1, 2, 256)
    waves, infos = service._prepare_tracks(_generated(plan, True), plan, 'moving_full_direct', expected, False)
    header = {'request_id': request['request_id'], 'track_kind': 'moving_full_direct', 'tracks': infos}
    if change:
        change(header, waves)
    service._send(conn, json.dumps(header).encode())
    for wave, _ in waves.values():
        for frame in wave:
            if delay:
                time.sleep(delay)
            service._send(conn, frame.tobytes())


@pytest.mark.parametrize('corruption,match', [
    ('shape', 'geometry'), ('group', 'group identity'), ('kind', 'track_kind'),
    ('header_kind', 'track_kind'), ('timebase', 'trajectory_timebase'),
    ('trajectory', 'trajectory mismatch'), ('whole_hash', 'content identity'),
    ('frame_hash', 'frame content'), ('missing', 'track set'),
    ('frame_count', 'hash count'), ('payload', 'frame content'),
    ('request_id', 'response identity'),
])
@pytest.mark.unit
def test_client_rejects_corrupt_responses(tmp_path, fake_backend, corruption, match):
    def change(header, waves):
        info = header['tracks']['direct']
        if corruption == 'shape': info['shape'][0] += 1
        if corruption == 'group': info['facts']['group_uid'] = 'other'
        if corruption == 'kind': info['facts']['track_kind'] = 'full'
        if corruption == 'header_kind': header['track_kind'] = 'full_direct'
        if corruption == 'timebase': info['facts']['trajectory_timebase']['num_samples'] = 256
        if corruption == 'trajectory': info['facts']['trajectory'][1][0][0] += 1
        if corruption == 'whole_hash': info['facts']['rir_sha256'] = '0' * 64
        if corruption == 'frame_hash': info['frame_hashes'][0] = '0' * 64
        if corruption == 'missing': del header['tracks']['direct']
        if corruption == 'frame_count': info['frame_hashes'].pop()
        if corruption == 'payload': waves['direct'][0][0, 0, 0, 0] += 1
        if corruption == 'request_id': header['request_id'] = 'other'
    with _local_server(tmp_path, handler=lambda conn, req: _peer_response(conn, req, change)) as client:
        with pytest.raises(service.OnlineRIRError, match=match):
            client.generate_tracks(_plan(), track_kind='moving_full_direct')


@pytest.mark.unit
def test_client_uses_one_deadline_across_trajectory_packets(tmp_path, fake_backend):
    with _local_server(tmp_path, handler=lambda conn, req: _peer_response(conn, req, delay=0.05)) as original:
        client = service.RIRClient(original.address, original.token, 0.12)
        with pytest.raises(service.OnlineRIRError, match='timed out'):
            client.generate_tracks(_plan(), track_kind='moving_full_direct')


@pytest.mark.unit
def test_static_request_cannot_silently_ignore_trajectory(fake_backend):
    client = service.RIRClient('/does/not/exist', 'fixture', 1)
    with pytest.raises(service.OnlineRIRError, match='trajectory requires'):
        client.generate_tracks(_plan())
    with pytest.raises(service.OnlineRIRError, match='unsupported'):
        client.generate_tracks(_plan(), track_kind='unknown')


class _FakeGpuRIR:
    """只替换第三方 CUDA 扩展，core backend 与 IPC 均执行真实代码。"""
    def beta_SabineEstimation(self, room, rt60):
        return np.full(6, 0.7)

    def att2t_SabineEstimator(self, attenuation, rt60):
        return rt60 / 4

    def t2n(self, duration, room, c):
        return [4, 4, 4]

    def simulateRIR(self, **kwargs):
        wave = np.zeros((len(kwargs['pos_src']), len(kwargs['pos_rcv']), 256), dtype=np.float32)
        wave[..., 8] = 1.0
        if tuple(kwargs['nb_img']) != (1, 1, 1):
            wave[..., 16] = 0.5
        return wave


@pytest.mark.unit
def test_real_backend_validator_and_generator_connect_to_ipc(tmp_path):
    plan = _plan()
    plan.update({'rt60': 0.15, 'diffuse_db': 15, 'sound_speed': 340})
    with _local_server(tmp_path, gpu_rir=_FakeGpuRIR()) as client:
        tracks = client.generate_tracks(plan, track_kind='moving_full_direct')
    for name, (wave, facts) in tracks.items():
        assert wave.shape == (4, 1, 2, 256)
        assert facts['trajectory_timebase']['num_samples'] == 8000
        assert facts['num_samples'] == 256
        assert facts['track_kind'] == name
        assert facts['rir_sha256'] == hashlib.sha256(wave.tobytes()).hexdigest()
    assert tracks['full'][1]['rir_sha256'] != tracks['direct'][1]['rir_sha256']


@pytest.mark.parametrize('field,value', [('frame_count', 3), ('sample_rate', 16000),
                                         ('hop_samples', 0), ('start_sample', True),
                                         ('num_samples', 256)])
@pytest.mark.unit
def test_timebase_validation_runs_before_connect(field, value):
    plan = _plan()
    plan['trajectory_timebase'][field] = value
    client = service.RIRClient('/does/not/exist', 'fixture', 1)
    with pytest.raises(service.OnlineRIRError, match='frame|sample|hop|integer'):
        client.generate_tracks(plan, track_kind='moving_full_direct')


@pytest.mark.unit
def test_full_only_legacy_mono_stays_compatible(tmp_path):
    plan = _plan(False)
    plan.update({'rt60': 0.15, 'diffuse_db': 15, 'sound_speed': 340, 'mic_loc': [2, 2, 1]})
    with _local_server(tmp_path, gpu_rir=_FakeGpuRIR()) as client:
        first, first_facts = client.generate(plan)
        second, second_facts = client.generate(plan)
    assert first.shape == (1, 256)
    np.testing.assert_array_equal(first, second)
    assert not first_facts['cache_hit']
    assert second_facts['cache_hit']


@pytest.mark.unit
def test_server_deadline_covers_backend_generation(tmp_path, monkeypatch, fake_backend):
    def slow(plan, gpu_rir, *, timeout):
        assert 0 < timeout <= 0.05
        time.sleep(0.08)
        return _generated(plan, True)
    monkeypatch.setattr(backend, 'generate_moving_rir_tracks', slow)
    with _local_server(tmp_path, timeout=0.05) as original:
        client = service.RIRClient(original.address, original.token, 0.5)
        with pytest.raises(service.OnlineRIRError, match='disconnected|timed out'):
            client.generate_tracks(_plan(), track_kind='moving_full_direct')


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize('sample_rate,channels', [(8000, 2), (16000, 4)])
def test_real_moving_provider_service_gpu_physics_hash_and_cache(sample_rate, channels):
    """真实 CUDA/spawn/IPC/provider 全链路，不使用 fake backend 或 skip 回退。"""
    from core.data.simulation.rir_provider import OnlineRIRProvider
    from core.data.simulation_configs import SimulationConfigs

    cfg = SimulationConfigs.from_dict({
        'rir_backend': 'online', 'spatial_enabled': True,
        'array_profile': 'linear_2ch_80mm' if channels == 2 else 'linear_4ch_40mm',
        'reference_channel': 0, 'gpurir_rt60_range_sec': [0.2, 0.2],
        'gpurir_room_dim_range_m': [[4., 4.], [4., 4.], [3., 3.]],
        'gpurir_source_distance_range_m': [0.7, 1.2],
        'gpurir_device': 0, 'gpurir_timeout_sec': 60.,
    }, strict=True)
    provider = OnlineRIRProvider.from_simulation(cfg)
    try:
        keys = ('speaker-b', 'speaker-a')
        # 同一服务依次消费 full-only、静态双轨和 moving 双轨。
        static = provider.sample_spatial_tracks(sample_rate, 2, np.random.default_rng(65), source_uids=keys, normalize=False)
        full_only = provider.sample_spatial_scene(sample_rate, 2, np.random.default_rng(65), source_uids=keys, normalize=False)
        assert static.full.facts['cache_hit'] is False
        assert full_only.facts['cache_hit'] is False
        kwargs = dict(num_samples=sample_rate, trajectory_points=4, source_uids=keys, normalize=False)
        moving = provider.sample_moving_spatial_tracks(sample_rate, 2, np.random.default_rng(65), **kwargs)
        plan = moving.plan
        assert moving.full.shape == moving.direct.shape == (4, 2, channels, plan['num_samples'])
        assert plan['trajectory_timebase']['num_samples'] == sample_rate
        assert plan['num_samples'] != plan['trajectory_timebase']['num_samples']
        assert plan['source_ids'] == list(keys)
        assert not np.array_equal(plan['trajectory'][0], plan['trajectory'][-1])
        for name in ('full', 'direct'):
            wave = getattr(moving, name)
            facts = moving.track_facts[name]
            assert np.isfinite(wave).all()
            assert facts['group_uid'] == plan['group_uid']
            assert facts['track_kind'] == name
            assert facts['generator_version'] == backend.GPURIR_GENERATOR_VERSION
            assert facts['trajectory_timebase'] == plan['trajectory_timebase']
            assert facts['cache_hit'] is False
            assert facts['rir_sha256'] == hashlib.sha256(wave.tobytes()).hexdigest()
            assert moving.facts['track_set'][name] == facts['rir_sha256']
        assert moving.track_facts['full']['rir_sha256'] != moving.track_facts['direct']['rir_sha256']
        mics = np.asarray(plan['mic_loc'])
        for index, positions in enumerate(plan['trajectory']):
            distance = np.linalg.norm(np.asarray(positions)[:, None] - mics[None], axis=-1)
            direct = moving.direct[index]
            peaks = np.abs(direct).argmax(axis=-1)
            np.testing.assert_allclose(peaks, distance * sample_rate / plan['sound_speed'], atol=2., rtol=0.)
            for source, channel in np.ndindex(peaks.shape):
                tail = direct[source, channel, peaks[source, channel] + int(sample_rate * .005):]
                assert np.sum(tail * tail) < np.sum(direct[source, channel] ** 2) * 1e-5
            frame = moving.track_facts['direct']['frame_facts'][index]
            assert frame['Tdiff'] == plan['num_samples'] / sample_rate
            assert frame['nb_img'] == [1, 1, 1]
        cached = provider.sample_moving_spatial_tracks(sample_rate, 2, np.random.default_rng(65), **kwargs)
        for name in ('full', 'direct'):
            assert cached.track_facts[name]['cache_hit'] is True
            np.testing.assert_array_equal(getattr(moving, name), getattr(cached, name))
        # 缓存不能把源音频时间轴不同的同几何请求当作相同 scene。
        changed = provider.sample_moving_spatial_tracks(
            sample_rate, 2, np.random.default_rng(65), **{**kwargs, 'num_samples': sample_rate * 2})
        assert changed.plan['group_uid'] != plan['group_uid']
        assert not changed.track_facts['full']['cache_hit']
        assert not changed.track_facts['direct']['cache_hit']
    finally:
        service.close_online_rir_services()


@pytest.mark.unit
@pytest.mark.parametrize('kind', ['full_direct', 'moving_full_direct'])
def test_backend_reversed_track_order_uses_canonical_wire_order(tmp_path, monkeypatch, fake_backend, kind):
    moving = kind == 'moving_full_direct'
    plan = _plan(moving)
    def reversed_backend(plan, gpu_rir, *, timeout):
        result = _generated(plan, moving)
        return {'direct': result['direct'], 'full': result['full']}
    function = 'generate_moving_rir_tracks' if moving else 'generate_rir_tracks'
    monkeypatch.setattr(backend, function, reversed_backend)
    # 发送层也必须独立于校验 helper 的字典插入序。
    prepare = service._prepare_tracks
    def reversed_prepare(*args, **kwargs):
        result, infos = prepare(*args, **kwargs)
        return {'direct': result['direct'], 'full': result['full']}, infos
    monkeypatch.setattr(service, '_prepare_tracks', reversed_prepare)
    with _local_server(tmp_path) as client:
        for hit in (False, True):
            result = client.generate_tracks(plan, track_kind=kind)
            for name, (wave, facts) in result.items():
                np.testing.assert_array_equal(wave, _generated(plan, moving)[name][0])
                assert facts['cache_hit'] is hit
