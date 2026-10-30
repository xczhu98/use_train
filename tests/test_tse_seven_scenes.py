"""唯一 TSE 七路管线的公开行为验收。"""
import copy
import numpy as np
import pytest
import torch
from test_multisource_scene_simulation import sources
from core.data.simulation.tse_scenarios import TSE_SCENARIOS, REMOVED_TSE_KEYS
from core.data.simulation_configs import SimulationConfigs
from core.losses.time_domain import AbsentTargetLoss
from core.data.speech_scene import scene_rng
from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset
from tasks.tse.data.collate import tse_collate_fn
pytestmark = pytest.mark.unit

def dataset_for(sources, scenario, **options):
    sim = {s.probability_key: float(s.name == scenario) for s in TSE_SCENARIOS}
    sim.update(prob_reverberation=0.0, prob_wind_noise_sidechain=0.0, prob_channel_enabled=False, use_high_pass=False, num_interferers={2: 1.0}, sir_range=[0.0, 0.0])
    sim.update(options.pop('simulation_cfg', {}))
    resources = dict(sources)
    if sim.get('rir_backend') == 'online':
        resources['rir_scp'] = None
    return TSEDynamicMixingDataset(**resources, target_sr=8000, max_duration_seconds=0.05, filter_by_duration=False, enrollment_duration=0.05, simulation_cfg=sim, **options)


def test_tse_mono_timeline_uses_dedicated_named_stream(sources, monkeypatch):
    """mono TSE 也必须在 shared renderer 前使用 timeline 子流 303。"""
    import tasks.tse.data.scene_dataset as scene_dataset

    ds = dataset_for(
        sources,
        "prob_target_interference",
        simulation_cfg={
            "prob_timeline": 1.0,
            "overlap_ratio_range": [0.5, 0.5],
            "timeline_mode_weights": {
                name: float(name == "head")
                for name in ("head", "tail", "middle", "crossing", "containment")
            },
        },
    )
    original = scene_dataset.place_scene_sources
    seen = {"called": False}

    def wrapped(source_waveforms, temporal, rng):
        probe = np.random.default_rng()
        probe.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        expected = scene_rng(ds, 0, 303)
        assert probe.random() == expected.random()
        seen["called"] = True
        return original(source_waveforms, temporal, rng)

    monkeypatch.setattr(scene_dataset, "place_scene_sources", wrapped)
    trace = ds.get_sample_with_trace(0).trace
    assert seen["called"]
    assert trace.metadata_dict()["scene"]["timeline"]

@pytest.mark.parametrize('scenario', TSE_SCENARIOS, ids=lambda s: s.name)
def test_tse_seven_compositions(sources, scenario):
    ds = dataset_for(sources, scenario.name)
    traced = ds.get_sample_with_trace(0, capture_stems=True)
    sample = traced.sample
    row = sample["targets"][0]
    cue = row["cues"].speaker_waveform
    assert cue is not None and cue.value is not None
    mix = sample["mixture"][0]
    target = row["target"]
    enroll = cue.value
    assert mix.shape == target.shape == (400,) and enroll.shape == (400,)
    facts = traced.trace.metadata_dict()['scene']
    assert row["target_present"] is scenario.target
    assert (facts['target_present'], facts['interference_present'], facts['noise_present']) == (scenario.target, scenario.interference, scenario.noise)
    assert facts['num_speakers_in_scene'] == int(scenario.target) + 2 * int(scenario.interference)
    assert np.any(target) == scenario.target
    assert np.any(mix) and np.any(enroll)
    stems = traced.trace.get_source('scene').stems
    assert bool(np.any(stems['background'])) == scenario.noise
    assert bool(facts['sir']) == scenario.interference
    if scenario.interference and not scenario.target:
        assert facts['sir_reference_source'] == 'latent_target'
    if not scenario.target:
        predicted = torch.tensor(mix[None], requires_grad=True)
        AbsentTargetLoss()(predicted, torch.zeros_like(predicted)).backward()
        assert predicted.grad.abs().sum() > 0
    if not scenario.target and (not scenario.interference):
        assert facts['source_plans'] == []
        assert facts['snr_reference_track'] == 'latent_speech'
        assert facts['background_snr_values']

@pytest.mark.parametrize('scenario', TSE_SCENARIOS, ids=lambda s: s.name)
def test_tse_forcing_and_multitarget(sources, scenario):
    ds = dataset_for(sources, 'prob_target_only', targets_per_mixture=3)
    result = ds.get_sample_with_trace(0, force_branch=scenario.name).sample
    expected = 1 + (2 if scenario.interference else 0)
    assert len(result['targets']) == min(3, expected)
    assert result["targets"][0]["target_present"] is scenario.target
    assert all(
        target["target_present"] is True for target in result["targets"][1:]
    )
    batch = tse_collate_fn([result], cue_specs=ds.cue_specs)
    assert batch.num_targets.tolist() == [min(3, expected)]
    assert batch.target_present.tolist() == [
        scenario.target,
        *([True] * (min(3, expected) - 1)),
    ]

@pytest.mark.parametrize('key', sorted(REMOVED_TSE_KEYS))
@pytest.mark.parametrize('strict', [False, True])
def test_removed_tse_fields_never_fallback(key, strict):
    with pytest.raises(ValueError, match='removed'):
        SimulationConfigs.from_dict({key: 0}, strict=strict)

def test_tse_probabilities_and_identity_capacity(sources):
    with pytest.raises(ValueError, match='sum to 1'):
        dataset_for(sources, 'prob_target_only', simulation_cfg={'prob_target_noise': 0.1})
    ds = dataset_for(sources, 'prob_target_interference', simulation_cfg={'num_interferers': {3: 1.0}})
    with pytest.raises((ValueError, RuntimeError), match='distinct speakers'):
        ds[0]

def test_tse_scene_seed_and_enrollment_stream(sources):
    a = dataset_for(sources, 'prob_target_interference_noise')
    b = dataset_for(sources, 'prob_target_interference_noise', enrollment_augment={'enabled': True, 'prob_eq': 1.0, 'eq_db_range': [6, 6]})
    a.set_epoch(1, base_seed=42)
    b.set_epoch(1, base_seed=42)
    first = a[0]
    second = b[0]
    np.testing.assert_array_equal(first["mixture"], second["mixture"])
    np.testing.assert_array_equal(
        first["targets"][0]["target"], second["targets"][0]["target"]
    )
    first_cue = first["targets"][0]["cues"].speaker_waveform
    second_cue = second["targets"][0]["cues"].speaker_waveform
    assert first_cue is not None and second_cue is not None
    assert not np.array_equal(first_cue.value, second_cue.value)

@pytest.mark.parametrize('mode', ['head', 'tail', 'middle', 'crossing', 'containment'])
@pytest.mark.parametrize('ratio', [0.0, 0.25, 1.0])
def test_shared_tse_timeline_window(mode, ratio):
    from core.data.speech_scene import plan_overlap_window
    source = np.arange(1, 101, dtype=np.float32)
    plan = plan_overlap_window(100, 100, ratio=ratio, mode=mode, rng=np.random.default_rng(7))
    result = plan.apply(source)
    assert np.count_nonzero(result) == round(100 * ratio)
    if ratio:
        turn = plan.turns[0]
        np.testing.assert_array_equal(result[turn.scene_start:turn.scene_start + turn.length], source[turn.source_start:turn.source_start + turn.length])

def test_noise_only_does_not_start_online_gpu(sources, monkeypatch):
    from core.data.simulation.rir_provider import OnlineRIRProvider

    def forbidden(*args, **kwargs):
        raise AssertionError('noise-only must not start GPU')
    monkeypatch.setattr(OnlineRIRProvider, 'from_simulation', forbidden)
    ds = dataset_for(sources, 'prob_noise_only', simulation_cfg={'rir_backend': 'online', 'prob_reverberation': 1.0})
    assert ds._online_rir_provider is None
    assert ds.get_sample_with_trace(0).trace.metadata_dict()['scene']['source_plans'] == []

def test_seven_scenario_probability_distribution():
    from collections import Counter
    from core.data.simulation.tse_scenarios import select_tse_scenario
    weights = [0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.4]
    cfg = SimulationConfigs.from_dict({'sir_range': [-5, 5]})
    for scenario, probability in zip(TSE_SCENARIOS, weights):
        setattr(cfg, scenario.probability_key, probability)
    rng = np.random.default_rng(51)
    counts = Counter((select_tse_scenario(cfg, rng)[0].name for _ in range(10000)))
    for (s, p) in zip(TSE_SCENARIOS, weights):
        assert counts[s.name] / 10000 == pytest.approx(p, abs=0.02)

@pytest.mark.parametrize('scenario', TSE_SCENARIOS, ids=lambda s: s.name)
def test_tse_seven_scene_audit_envelope(sources, scenario):
    from tools.verify.simulation_audit_contract import build_tse_trace_metadata_envelope, SimulationAuditEnvelope
    ds = dataset_for(sources, scenario.name, targets_per_mixture=3)
    trace = ds.get_sample_with_trace(0, capture_stems=True).trace
    envelope = build_tse_trace_metadata_envelope(trace)
    assert envelope.audit_format == 'source-scene-stems-v2'
    assert envelope.contract_version == '1.4'
    descriptor_keys = {
        (item['role'], item['stem'])
        for item in envelope.payload['stem_descriptors']
    }
    if scenario.name == 'prob_noise_only':
        assert ('scene', 'target/absent') in descriptor_keys
        assert not any(
            role == 'scene' and (stem.startswith('source_full_rir/') or stem.startswith('target/'))
            for role, stem in descriptor_keys
            if stem != 'target/absent'
        )
    SimulationAuditEnvelope.from_dict(envelope.to_dict())


def test_tse_target_only_needs_no_background_pool(sources):
    ds=TSEDynamicMixingDataset(speech_scp=sources['speech_scp'],spk2enroll_json=sources['spk2enroll_json'],
        target_sr=8000,max_duration_seconds=.05,filter_by_duration=False,enrollment_duration=.05,
        simulation_cfg={'prob_target_only':1.,'prob_reverberation':0.,'prob_channel_enabled':False})
    assert np.any(ds[0]["targets"][0]["target"])


def test_tse_shortage_cannot_become_lenient_placeholder(sources):
    ds=dataset_for(sources,'prob_target_interference',strict_load=False,simulation_cfg={'num_interferers':{3:1.}})
    with pytest.raises(RuntimeError,match='distinct speakers'):
        ds[0]
