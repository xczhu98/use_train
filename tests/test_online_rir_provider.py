"""在线 RIR 几何、整组调用与受管理服务。真实 CUDA 用 integration+gpu 显式选择。"""
import json
from dataclasses import replace
from functools import partial
import numpy as np
import pytest
from test_multisource_scene_simulation import sources
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation.rir_config import OnlineRIRConfig
from core.data.simulation.rir_geometry import sample_rir_scene
from core.data.simulation.gpurir_backend import generate_rir_group
from core.data.simulation.rir_provider import OnlineRIRProvider
from core.data.simulation.rir_service import close_online_rir_services

@pytest.mark.unit
@pytest.mark.parametrize('count', [1, 2, 3, 8])
def test_group_geometry_is_shared_and_reproducible(count):
    cfg = OnlineRIRConfig.from_simulation(SimulationConfigs())
    a = sample_rir_scene(cfg, 16000, count, np.random.default_rng(6))
    b = sample_rir_scene(cfg, 16000, count, np.random.default_rng(6))
    assert a == b
    room = np.array(a['room_dim'])
    positions = np.array(a['source_loc'])
    mic = np.array(a['mic_loc'])
    assert positions.shape == (count, 3) and mic.shape == (3,)
    assert ((positions > cfg.margin) & (positions < room - cfg.margin)).all()
    distances = np.linalg.norm(positions - mic, axis=1)
    np.testing.assert_allclose(distances, a['dist'])
    assert np.all(distances >= cfg.distance_range[0]) and np.all(distances <= cfg.distance_range[1])
    for i in range(count):
        for j in range(i):
            assert np.linalg.norm(positions[i] - positions[j]) >= cfg.min_separation
    json.dumps(a, allow_nan=False)

@pytest.mark.unit
def test_gpu_backend_submits_all_sources_in_one_call():
    plan = sample_rir_scene(OnlineRIRConfig.from_simulation(SimulationConfigs()), 16000, 3, np.random.default_rng(7))

    class Fake:
        calls = []

        def beta_SabineEstimation(self, room, rt60):
            return np.full(6, 0.7)

        def att2t_SabineEstimator(self, attenuation, rt60):
            return rt60 / 4

        def t2n(self, time, room, c):
            return [4, 4, 4]

        def simulateRIR(self, **kw):
            self.calls.append(kw)
            out = np.zeros((len(kw['pos_src']), len(kw['pos_rcv']), plan['num_samples']), np.float32)
            out[:, 0, 10] = [1.0, 0.5, 0.2]
            return out
    backend = Fake()
    (wave, facts) = generate_rir_group(plan, backend)
    assert len(backend.calls) == 1 and backend.calls[0]['pos_src'].shape == (3, 3)
    assert backend.calls[0]['pos_rcv'].shape == (1, 3)
    assert wave.shape == (3, plan['num_samples']) and len(facts['beta']) == 6

@pytest.mark.unit
def test_fixed_array_backend_returns_source_microphone_bank():
    cfg = SimulationConfigs.from_dict({
        'rir_backend': 'online', 'spatial_enabled': True,
        'reference_channel': 0,
        'custom_geometry': {'positions_m': [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]]},
    }, strict=True)
    online = OnlineRIRConfig.from_simulation(cfg)
    plan = sample_rir_scene(online, 8000, 2, np.random.default_rng(8))
    assert np.asarray(plan['mic_loc']).shape == (2, 3)
    assert np.asarray(plan['microphone_offsets']).shape == (2, 3)

    class Fake:
        def beta_SabineEstimation(self, room, rt60): return np.full(6, 0.7)
        def att2t_SabineEstimator(self, attenuation, rt60): return rt60 / 4
        def t2n(self, time, room, c): return [4, 4, 4]
        def simulateRIR(self, **kw):
            out = np.zeros((2, 2, plan['num_samples']), np.float32)
            out[:, :, 10] = [[1.0, 0.8], [0.5, 0.4]]
            return out

    wave, facts = generate_rir_group(plan, Fake())
    assert wave.shape == (2, 2, plan['num_samples'])
    assert facts['provenance']['geometry'] == 'fixed_microphone_array'
    assert facts['provenance']['requested_shape'] == [2, 2, plan['num_samples']]
    assert facts['num_channels'] == 2

@pytest.mark.unit
def test_spatial_provider_validates_shape_and_preserves_provenance():
    cfg = SimulationConfigs.from_dict({
        'rir_backend': 'online', 'spatial_enabled': True,
        'reference_channel': 0,
        'custom_geometry': {'positions_m': [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]]},
    }, strict=True)
    online = OnlineRIRConfig.from_simulation(cfg)

    class Client:
        def generate(self, plan):
            return np.ones((2, 2, plan['num_samples']), np.float32), {**plan, 'rir_sha256': 'raw'}

    provider = OnlineRIRProvider(online, Client())
    wave, facts = provider.sample_spatial_scene(8000, 2, np.random.default_rng(9), normalize=True)
    assert wave.shape == (2, 2, wave.shape[-1])
    assert facts['provenance']['provider_contract'] == '[K,C,L]'
    assert np.isfinite(wave).all()

@pytest.mark.unit
@pytest.mark.parametrize('key,value', [('gpurir_rt60_range_sec', [0.0, 1.0]), ('gpurir_source_min_separation_m', 0.0), ('gpurir_room_dim_range_m', [[3.0, 4.0], [3.0, 4.0]]), ('gpurir_max_sources', True), ('gpurir_device', -1), ('gpurir_diffuse_transition_db', 60)])
def test_invalid_online_config_fails_before_cuda(key, value):
    with pytest.raises(ValueError):
        SimulationConfigs.from_dict({key: value}, strict=True)

@pytest.mark.unit
def test_tse_interference_only_budget_counts_only_interferers():
    from core.data.simulation.tse_scenarios import validate_tse_scenarios

    cfg = SimulationConfigs.from_dict(
        {
            'rir_backend': 'online',
            'gpurir_max_sources': 1,
            'num_interferers': {1: 1.0},
            'prob_interference_only': 1.0,
            'sir_range': [-5.0, 5.0],
        },
        strict=True,
    )
    validate_tse_scenarios(cfg)

@pytest.mark.unit
def test_joint_normalization_preserves_group_gain_ratio():

    class Client:

        def generate(self, plan):
            return (np.array([[1, 0], [0.5, 0]], np.float32), plan)
    cfg = OnlineRIRConfig.from_simulation(SimulationConfigs())
    (wave, facts) = OnlineRIRProvider(cfg, Client()).sample_scene(16000, 2, np.random.default_rng(1), normalize=True)
    assert wave[0, 0] / wave[1, 0] == 2

@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize('count', [1, 2, 3])
@pytest.mark.parametrize('sample_rate', [16000, 32000])
def test_real_gpurir_service_group_and_cache(count, sample_rate):
    cfg = SimulationConfigs.from_dict({'rir_backend': 'online', 'gpurir_room_dim_range_m': [[3.0, 4.0], [3.0, 4.0], [2.5, 3.0]], 'gpurir_rt60_range_sec': [0.15, 0.2]}, strict=True)
    provider = OnlineRIRProvider.from_simulation(cfg)
    try:
        (first, facts) = provider.sample_scene(sample_rate, count, np.random.default_rng(5), normalize=True)
        (second, again) = provider.sample_scene(sample_rate, count, np.random.default_rng(5), normalize=True)
        np.testing.assert_array_equal(first, second)
        assert first.shape[0] == count and np.isfinite(first).all()
        assert again['cache_hit'] is True and facts['group_uid'] == again['group_uid']
        assert len(facts['source_loc']) == count and len(facts['beta']) == 6
    finally:
        close_online_rir_services()

@pytest.mark.integration
@pytest.mark.gpu
def test_real_online_tse_and_se_dataset(sources):
    from test_tse_seven_scenes import dataset_for
    from tasks.se.data.dynamic_dataset import DynamicMixingDataset
    options = {'rir_backend': 'online', 'prob_reverberation': 1.0, 'gpurir_rt60_range_sec': [0.15, 0.2]}
    try:
        tse = dataset_for(sources, 'prob_target_interference_noise', simulation_cfg=options)
        traced = tse.get_sample_with_trace(0, capture_stems=True)
        scene = traced.trace.metadata_dict()['scene']
        assert len(scene['online_rir_scene']['source_loc']) == 3
        assert all((p['rir']['backend'] == 'online' for p in scene['source_plans']))
        assert np.isfinite(traced.sample["mixture"]).all()
        assert np.isfinite(traced.sample["targets"][0]["target"]).all()
        se = DynamicMixingDataset(sources['speech_scp'], sources['noise_scp'], None, target_sr=8000, max_duration_seconds=0.05, filter_by_duration=False, simulation_cfg={**options, 'prob_wind_noise_sidechain': 0.0, 'prob_channel_enabled': False})
        output = se.get_sample_with_trace(0, capture_stems=True)
        assert np.isfinite(output.sample[0]).all()
    finally:
        close_online_rir_services()

@pytest.mark.integration
@pytest.mark.gpu
def test_real_fixed_microphone_array_returns_k_c_l():
    cfg = SimulationConfigs.from_dict({
        'rir_backend': 'online', 'spatial_enabled': True,
        'reference_channel': 0,
        'custom_geometry': {'positions_m': [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]]},
        'gpurir_room_dim_range_m': [[3.0, 3.0], [3.0, 3.0], [2.5, 2.5]],
        'gpurir_rt60_range_sec': [0.15, 0.15],
        'gpurir_source_distance_range_m': [0.5, 0.5],
    }, strict=True)
    provider = OnlineRIRProvider.from_simulation(cfg)
    try:
        wave, facts = provider.sample_spatial_scene(8000, 2, np.random.default_rng(12), normalize=True)
        assert wave.ndim == 3 and wave.shape[:2] == (2, 2)
        assert np.isfinite(wave).all()
        assert facts['provenance']['requested_shape'][:2] == [2, 2]
    finally:
        close_online_rir_services()

@pytest.mark.integration
@pytest.mark.gpu
def test_online_tse_spawn_workers_and_epoch(sources):
    import torch
    from test_tse_seven_scenes import dataset_for
    from tasks.tse.data.collate import tse_collate_fn
    ds = dataset_for(sources, 'prob_target_interference_noise', simulation_cfg={'rir_backend': 'online', 'prob_reverberation': 1.0, 'gpurir_rt60_range_sec': [0.15, 0.2]})
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context='spawn',
        collate_fn=partial(tse_collate_fn, cue_specs=ds.cue_specs),
        timeout=30,
    )
    try:
        first = list(loader)[0].mixture
        torch.testing.assert_close(first[0], torch.from_numpy(ds[0]["mixture"]))
        ds.set_epoch(2, base_seed=31)
        second = list(loader)[0].mixture
        torch.testing.assert_close(second[0], torch.from_numpy(ds[0]["mixture"]))
        assert not torch.equal(first, second)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
        close_online_rir_services()

@pytest.mark.unit
def test_online_geometry_failure_cannot_become_placeholder():
    from core.data.simulation.rir_service import OnlineRIRError
    provider = OnlineRIRProvider(OnlineRIRConfig.from_simulation(SimulationConfigs()), None)
    with pytest.raises(OnlineRIRError, match='source_count'):
        provider.sample_scene(16000, 0, np.random.default_rng(0), normalize=True)

@pytest.mark.unit
def test_enrollment_distance_range_override_shapes_plan():
    captured = {}

    class Client:
        def generate(self, plan):
            captured.update(plan)
            wave = np.zeros((1, max(plan['num_samples'], 16)), np.float32)
            wave[0, 10] = 1.0
            return wave, {'rir_sha256': 'fake'}

    cfg = OnlineRIRConfig.from_simulation(SimulationConfigs())
    provider = OnlineRIRProvider(cfg, Client())
    (wave, facts) = provider.sample_scene(
        16000, 1, np.random.default_rng(3), normalize=True, distance_range=[0.05, 0.2]
    )
    assert wave.shape[0] == 1 and facts["rir_sha256"] != facts["raw_rir_sha256"]
    assert 0.05 <= captured["dist"][0] <= 0.2
    # 同一 rng 不传覆盖时应落回 config 默认距离范围 (0.2, 3.0)。
    (default_wave, _) = provider.sample_scene(
        16000, 1, np.random.default_rng(3), normalize=True
    )
    assert default_wave.shape[0] == 1

@pytest.mark.unit
def test_enrollment_distance_range_override_invalid_fails_closed():
    from core.data.simulation.rir_service import OnlineRIRError
    provider = OnlineRIRProvider(OnlineRIRConfig.from_simulation(SimulationConfigs()), None)
    with pytest.raises(OnlineRIRError, match='distance_range'):
        provider.sample_scene(
            16000, 1, np.random.default_rng(3), normalize=True, distance_range=[0.0, 0.2]
        )

@pytest.mark.unit
def test_enrollment_distance_range_field_validated():
    cfg = SimulationConfigs.from_dict(
        {'gpurir_enrollment_distance_range_m': [0.06, 0.18]}, strict=True
    )
    assert tuple(cfg.gpurir_enrollment_distance_range_m) == (0.06, 0.18)
    assert tuple(SimulationConfigs().gpurir_enrollment_distance_range_m) == (0.05, 0.2)
    with pytest.raises(ValueError):
        SimulationConfigs.from_dict(
            {'gpurir_enrollment_distance_range_m': [0.0, 0.2]}, strict=True
        )
    with pytest.raises(ValueError):
        SimulationConfigs.from_dict(
            {'gpurir_enrollment_distance_range_m': [0.3, 0.2]}, strict=True
        )


@pytest.mark.integration
@pytest.mark.gpu
def test_real_online_aec_and_ss_dataset(sources,tmp_path):
    import soundfile as sf
    from tasks.aec.data.dynamic_dataset import DynamicMixingDataset as AECDataset
    from tasks.ss.data.dynamic_dataset import SSDynamicMixingDataset
    pair=tmp_path/'pair.wav'
    sf.write(pair,np.zeros((400,2),np.float32),8000)
    pairs=tmp_path/'pairs.json';pairs.write_text(json.dumps({'pairs':[str(pair)]}))
    options={'rir_backend':'online','prob_reverberation':1.,
             'gpurir_rt60_range_sec':[.15,.2],'prob_wind_noise_sidechain':0.,
             'prob_channel_enabled':False}
    try:
        aec=AECDataset(speech_scp=sources['speech_scp'],noise_scp=sources['noise_scp'],rir_scp=None,
            real_farend_echo_pair_json=str(pairs),aec_ref_echo_fine_gcc_align=False,
            target_sr=8000,max_duration_seconds=.05,filter_by_duration=False,
            simulation_cfg={**options,'prob_target_noise':1.})
        trace=aec.get_sample_with_trace(0,capture_stems=True)
        assert np.isfinite(np.asarray(trace.sample[0])).all()
        ss=SSDynamicMixingDataset(sources['speech_scp'],num_sources=2,sample_rate=8000,
            segment_seconds=.05,simulation_cfg=options)
        trace=ss.get_sample_with_trace(0,capture_stems=True)
        assert len(trace.trace.metadata_dict()['scene']['online_rir_scene']['source_loc'])==2
        assert np.isfinite(np.asarray(trace.sample['mixture'])).all()
    finally:close_online_rir_services()

@pytest.mark.unit
def test_backend_direct_track_is_explicit_and_uses_direct_image_request():
    plan = sample_rir_scene(OnlineRIRConfig.from_simulation(SimulationConfigs()), 16000, 1, np.random.default_rng(41))
    class Fake:
        calls = []
        def beta_SabineEstimation(self, room, rt60): return np.full(6, 0.7)
        def att2t_SabineEstimator(self, attenuation, rt60): return rt60 / 4
        def t2n(self, time, room, c): return [4, 4, 4]
        def simulateRIR(self, **kw):
            self.calls.append(kw)
            out = np.zeros((1, 1, plan['num_samples']), np.float32)
            out[0, 0, 8 if tuple(kw['nb_img']) == (1, 1, 1) else 12] = 1.0
            return out
    backend = Fake()
    wave, facts = generate_rir_group(plan, backend, track_kind='direct')
    assert wave.shape == (1, plan['num_samples'])
    assert facts['track_kind'] == 'direct'
    assert tuple(backend.calls[0]['nb_img']) == (1, 1, 1)

@pytest.mark.unit
def test_online_full_direct_provider_fails_closed_without_grouped_client():
    cfg = SimulationConfigs.from_dict({
        'rir_backend': 'online', 'spatial_enabled': True,         'reference_channel': 0,
        'custom_geometry': {'positions_m': [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]]},
    }, strict=True)
    provider = OnlineRIRProvider(OnlineRIRConfig.from_simulation(cfg), object())
    with pytest.raises(Exception, match='full_direct|generate_tracks'):
        provider.sample_spatial_tracks(8000, 1, np.random.default_rng(42))

@pytest.mark.unit
def test_request_shape_supports_static_scene_full_direct():
    from core.data.simulation.rir_service import OnlineRIRError, _request_shape

    plan = {
        "group_uid": "g",
        "source_loc": [[1.0, 1.0, 1.5], [2.0, 2.0, 1.5]],
        "mic_loc": [[0.0, 0.0, 1.5], [0.1, 0.0, 1.5]],
        "num_samples": 64,
    }
    assert _request_shape(plan, "scene_static_full_direct") == (2, 2, 64)
    with pytest.raises(OnlineRIRError, match="trajectory"):
        _request_shape(
            {**plan, "trajectory_timebase": {}}, "scene_static_full_direct"
        )
    with pytest.raises(OnlineRIRError, match="mic geometry"):
        _request_shape({**plan, "mic_loc": [0.0, 0.0, 1.5]}, "scene_static_full_direct")
