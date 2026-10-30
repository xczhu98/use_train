"""真实 gpuRIR full/direct、moving IPC 和 SSL 波形端到端门禁。"""

import numpy as np
import pytest

from core.data.simulation.rir_provider import OnlineRIRProvider
from core.data.simulation.rir_service import close_online_rir_services
from core.data.simulation_configs import SimulationConfigs

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _config_dict():
    return {
        'rir_backend': 'online', 'spatial_enabled': True,
        'array_profile': 'linear_2ch_80mm',
        'reference_channel': 0,
        'gpurir_rt60_range_sec': [0.2, 0.2],
        'gpurir_room_dim_range_m': [[4., 4.], [4., 4.], [3., 3.]],
        'gpurir_source_distance_range_m': [0.7, 1.2],
        'gpurir_device': 0, 'gpurir_timeout_sec': 60.,
    }


def _config():
    return SimulationConfigs.from_dict(_config_dict())


def _assert_direct_physics(direct, source_loc, mic_loc, sr, speed):
    distances = np.linalg.norm(np.asarray(source_loc)[:, None] - np.asarray(mic_loc)[None], axis=-1)
    peaks = np.abs(direct).argmax(axis=-1)
    np.testing.assert_allclose(peaks, distances * sr / speed, atol=2.0, rtol=0)
    # direct RIR 的窗化 sinc 只在到达时刻邻域内；5 ms 后不应有漫射尾。
    for k, c in np.ndindex(peaks.shape):
        tail = direct[k, c, peaks[k, c] + int(sr * .005):]
        assert np.sum(tail * tail) < np.sum(direct[k, c] ** 2) * 1e-5


def test_real_full_direct_delay_tail_and_cache():
    provider = OnlineRIRProvider.from_simulation(_config())
    try:
        result = provider.sample_spatial_tracks(16000, 2, np.random.default_rng(65))
        direct, full = result.direct.canonical_rirs, result.full.canonical_rirs
        assert full.shape == direct.shape and full.shape[:2] == (2, 2)
        assert np.isfinite(full).all() and np.isfinite(direct).all()
        _assert_direct_physics(direct, result.direct.facts['source_loc'], result.direct.facts['mic_loc'], 16000, 343.)
        assert result.direct.facts['nb_img'] == [1, 1, 1]
        assert result.direct.facts['Tdiff'] == result.direct.facts['num_samples'] / 16000
        assert result.full.content_hash != result.direct.content_hash
        cached = provider.sample_spatial_tracks(16000, 2, np.random.default_rng(65))
        np.testing.assert_array_equal(full, cached.full.canonical_rirs)
        np.testing.assert_array_equal(direct, cached.direct.canonical_rirs)
    finally:
        close_online_rir_services()


@pytest.mark.parametrize('points', [4, 50])
def test_real_moving_ipc_physics_cache_and_official_waveform(points):
    from core.data.simulation.moving_scene import render_moving_spatial_scene

    provider = OnlineRIRProvider.from_simulation(_config())
    try:
        result = provider.sample_moving_spatial_tracks(8000, 2, np.random.default_rng(86),
                                                     num_samples=4000, trajectory_points=points)
        assert result.full.shape[:3] == (points, 2, 2)
        assert result.full.shape == result.direct.shape
        for index in (0, points - 1):
            _assert_direct_physics(result.direct[index], result.plan['trajectory'][index],
                                   result.plan['mic_loc'], 8000, 343.)
        assert not np.allclose(result.direct[0], result.direct[-1])
        cached = provider.sample_moving_spatial_tracks(8000, 2, np.random.default_rng(86),
                                                     num_samples=4000, trajectory_points=points)
        assert all(cached.track_facts[name]['cache_hit'] for name in ('full', 'direct'))
        np.testing.assert_array_equal(result.full, cached.full)
        np.testing.assert_array_equal(result.direct, cached.direct)
        sources = np.random.default_rng(87).normal(size=(2, 4000)).astype(np.float32) * .02
        starts = np.arange(points) * result.plan['trajectory_timebase']['hop_samples']
        rendered = render_moving_spatial_scene(sources, result.full, direct_rirs=result.direct,
                                             frame_samples=starts, early_boundary=80, num_samples=4000)
        import gpuRIR
        for source in range(2):
            # 使用官方 CUDA 卷积独立复核共享 CPU renderer 的分段/尾部语义。
            for bank, actual in ((result.full, rendered.full_tracks), (result.direct, rendered.direct_tracks)):
                expected = gpuRIR.simulateTrajectory(sources[source], bank[:, source].copy(),
                                                    timestamps=starts / 8000., fs=8000)[:4000].T
                np.testing.assert_allclose(actual[source], expected, rtol=3e-4, atol=2e-6)
    finally:
        close_online_rir_services()


@pytest.mark.parametrize('points', [1, 4])
def test_real_ssl_dataset_moving_or_static_to_optimizer(tmp_path, points):
    import soundfile as sf
    import torch
    from tasks.ssl.data.dynamic_dataset import SSLDynamicMixingDataset
    from tasks.ssl.data.collate import ssl_collate_fn
    from tasks.ssl.trainer.entry import SSLTrainingEntry
    from tasks.ssl.models.ipdnet import FixedArrayIPDNet

    rate, length = 8000, 2000
    rows = []
    rng = np.random.default_rng(23)
    for index in range(2):
        path = tmp_path / f'speech-{index}.wav'
        sf.write(path, rng.normal(size=length).astype(np.float32) * .02, rate, subtype='FLOAT')
        rows.append(f'source_{index} {rate} {length} {path}')
    speech = tmp_path / 'speech.scp'
    speech.write_text('\n'.join(rows) + '\n')
    noise = tmp_path / 'noise.wav'
    sf.write(noise, rng.normal(size=length).astype(np.float32) * .01, rate, subtype='FLOAT')
    noise_scp = tmp_path / 'noise.scp'
    noise_scp.write_text(f'noise {rate} {length} {noise}\n')
    simulation = _config_dict()
    simulation.update(prob_reverberation=1., prob_channel_enabled=False,
                      prob_moving_rir=float(points > 1),
                      noisy_post_mix_degradations_enabled=True, prob_post_eq=0.,
                      prob_clip=1., clip_variants=['fixed_rail'], clip_variant_probs={'fixed_rail': 1.})
    cfg = {
        'data': {'sample_rate': rate, 'mode': 'dynamic_ssl', 'array_profile': 'linear_2ch_80mm',
                 'train': {'speech_scp': str(speech), 'noise_scp': str(noise_scp), 'tracks': 2,
                           'segment_seconds': length / rate,
                           'trajectory_points': points, 'simulation_cfg': simulation}},
        'stft': {'n_fft': 128, 'hop_length': 64, 'win_length': 128, 'center': False},
        'model': {'type': 'FixedArrayIPDNet', 'args': {'input_size': 1, 'max_track': 2}},
    }
    section = SSLTrainingEntry._dataset_config(cfg, 'train')
    SSLTrainingEntry._validate_split_resources('dynamic_ssl', section, 'train')
    try:
        dataset = SSLDynamicMixingDataset(**section)
        item = dataset[0]
        assert item['provenance']['direct_path_exact']
        assert torch.count_nonzero(item['target_dpipd']) > 0
        assert item['provenance']['scene']['post_mix_stage_order'][-1] == 'post-finite'
        if points == 1:
            assert item['source_positions'].shape == (2, 3)
        else:
            assert item['provenance']['rir']['moving']
            assert item['source_positions'].shape == (item['input_lengths'], 2, 3)
            trajectory = np.asarray(item['provenance']['rir']['trajectory'], dtype=np.float32)
            timebase = item['provenance']['rir']['trajectory_timebase']
            anchors = np.arange(points) * timebase['hop_samples']
            # center=False 的观测对应窗口中心，末端按轨迹最后一个锚点保持。
            centers = np.arange(item['input_lengths']) * 64 + 128 / 2
            expected_positions = np.empty((len(centers), 2, 3), dtype=np.float32)
            for track in range(2):
                for axis in range(3):
                    expected_positions[:, track, axis] = np.interp(centers, anchors, trajectory[:, track, axis])
            np.testing.assert_allclose(item['source_positions'].numpy(), expected_positions, atol=1e-6)
            assert not torch.allclose(item['source_positions'][0], item['source_positions'][-1])
            assert not torch.allclose(item['doa'][0], item['doa'][-1])
        batch = ssl_collate_fn([item]).pin_memory().to("cuda", non_blocking=True)
        model = FixedArrayIPDNet(input_size=1, hidden_size=4, max_track=2, n_fft=128).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        before = next(model.parameters()).detach().clone()
        loss = (model(batch.features) - batch.target_dpipd).square().mean()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        assert not torch.equal(before, next(model.parameters()))
        if points > 1:
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1,
                                                multiprocessing_context='spawn', collate_fn=ssl_collate_fn)
            iterator = iter(loader)
            try:
                worker_batch = next(iterator)
                torch.testing.assert_close(worker_batch.features[0], item['features'])
            finally:
                iterator._shutdown_workers()
    finally:
        close_online_rir_services()
