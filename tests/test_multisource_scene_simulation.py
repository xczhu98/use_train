"""多源共享仿真的跨层行为测试，覆盖真实波形、身份、静音监督和可复现性。"""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from core.data.post_augment import apply_post_augmentations
from core.data.simulation_configs import SimulationConfigs
from core.data.source_pool import parse_source_pool, sample_distinct_identities
from core.data.speech_scene import plan_overlap_window, scene_activity_facts
from core.losses.time_domain import AbsentTargetLoss
from tasks.ss.data import SSDynamicMixingDataset, ss_collate_fn
from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset, EnrollmentAugmentationProviderError

pytestmark = pytest.mark.unit


@pytest.fixture
def sources(tmp_path):
    sr = 8000
    rows, enrollment = [], {"fixture": {}}
    for index, (uid, speaker, length) in enumerate((('a', 'a', 240), ('a2', 'a', 600), ('b', 'b', 640), ('c', 'c', 500))):
        waveform = (0.1 * np.sin(np.arange(length) * (0.13 + index * 0.07))).astype(np.float32)
        path = tmp_path / f'{uid}.wav'
        sf.write(path, waveform, sr, subtype='FLOAT')
        rows.append(dict(uid=uid, path=str(path), fs=sr, length=length, language='en', dataset_id='fixture', speaker_id=speaker))
        enroll_path = tmp_path / f'{uid}_enroll.wav'
        sf.write(enroll_path, waveform, sr, subtype='FLOAT')
        enrollment['fixture'].setdefault(speaker, []).append(str(enroll_path))
    speech = tmp_path / 'speech.jsonl'
    speech.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    enroll = tmp_path / 'enroll.json'
    enroll.write_text(json.dumps(enrollment))
    noise = tmp_path / 'noise.wav'
    sf.write(noise, np.random.default_rng(9).normal(0, 0.03, 700).astype(np.float32), sr)
    noise_scp = tmp_path / 'noise.scp'
    noise_scp.write_text(f'noise {sr} 700 {noise}\n')
    rir = tmp_path / 'rir.wav'
    sf.write(rir, np.array([1, 0, 0.6], dtype=np.float32), sr, subtype='FLOAT')
    rir_scp = tmp_path / 'rir.scp'
    rir_scp.write_text(f'room {sr} 3 {rir}\n')
    return dict(speech_scp=str(speech), noise_scp=str(noise_scp), rir_scp=str(rir_scp), spk2enroll_json=str(enroll))


def tse_dataset(sources, **simulation):
    return TSEDynamicMixingDataset(**sources, target_sr=8000, max_duration_seconds=0.05,
        enrollment_duration=0.05, filter_by_duration=False,
        simulation_cfg={"rir_backend": "offline", "prob_reverberation": 1.0,
                        "prob_wind_noise_sidechain": 0.0, "prob_channel_enabled": False,
                        "use_high_pass": False, "num_interferers": {2: 1.}, "prob_target_only": 0., "prob_target_interference_noise": 1.,
                        "sir_range": [0., 0.], **simulation})


def test_ss_mixed_short_long_sources_use_shared_scene_and_close(sources):
    dataset = SSDynamicMixingDataset(sources['speech_scp'], num_sources=3, sample_rate=8000,
        segment_seconds=0.05, sir_range=[0, 0])
    traced = dataset.get_sample_with_trace(0, capture_stems=True)
    batch = ss_collate_fn([traced.sample])
    assert batch.sources.shape == (1, 3, 400)
    torch.testing.assert_close(batch.mixture, batch.sources.sum(dim=1))
    facts = traced.trace.metadata_dict()['scene']
    assert len(facts['source_plans']) == 3
    assert facts['post_mix_stage_order'][1:3] == ['D-normalize', 'D-guard']


def test_source_identity_is_optional_and_namespace_is_preserved(sources, tmp_path):
    records = parse_source_pool(sources['speech_scp'])
    assert records[0].identity == ('fixture', 'a')
    rows = [dict(uid=r.uid, path=str(r.path), fs=r.sample_rate, length=r.length) for r in records]
    anonymous = tmp_path / 'anonymous.jsonl'
    anonymous.write_text('\n'.join(json.dumps(row) for row in rows))
    dataset = SSDynamicMixingDataset(str(anonymous), num_sources=4, sample_rate=8000, segment_seconds=0.05)
    assert len(dataset[0]['sources']) == 4


def test_distinct_identity_sampler_fails_or_reduces_explicitly():
    groups = {('d', 'a'): ('a1', 'a2'), ('d', 'b'): ('b1',)}
    with pytest.raises(ValueError, match='distinct speakers'):
        sample_distinct_identities(groups, 3, np.random.default_rng(1))
    selected = sample_distinct_identities(groups, 3, np.random.default_rng(1), shortage='reduce')
    assert len(selected) == 2 and 'b1' in selected


def test_tse_independent_scene_calibrates_sir_after_rir(sources):
    dataset = tse_dataset(sources)
    traced = dataset.get_sample_with_trace(0, capture_stems=True)
    assert traced.sample["mixture"].shape == (1, 400)
    assert traced.sample["targets"][0]["target"].shape == (400,)
    facts = traced.trace.metadata_dict()['scene']
    assert facts['actual_interferers'] == 2
    assert len({tuple(identity) for identity in facts['source_identities']}) == 3
    assert facts['sir'][0]['realized_db'] == pytest.approx(0., abs=1e-5)
    assert facts['sir_measure_point'] == 'post_rir_pre_background'
    json.dumps(traced.trace.metadata_dict(), allow_nan=False)


def test_tse_target_absent_contains_other_speech_and_has_suppression_gradient(sources):
    dataset = tse_dataset(sources, prob_interference_noise_only=1., prob_target_interference_noise=0.)
    traced = dataset.get_sample_with_trace(0, capture_stems=True)
    row = traced.sample["targets"][0]
    assert np.count_nonzero(row["target"]) == 0
    assert row["target_present"] is False
    assert np.count_nonzero(traced.sample["mixture"]) > 0
    facts = traced.trace.metadata_dict()['scene']
    assert facts['target_present'] is False and 'a' not in facts['source_uids']
    assert facts['sir_reference_source'] == 'latent_target'
    assert facts['sir'][0]['reference'] == 'latent_target'
    prediction = torch.ones(1, 400, requires_grad=True)
    loss = AbsentTargetLoss()(prediction, torch.zeros_like(prediction), torch.tensor([200]))
    loss.backward()
    assert prediction.grad[0, :200].abs().sum() > 0
    assert prediction.grad[0, 200:].abs().sum() == 0


def test_enrollment_cannot_reuse_scene_recording(sources):
    dataset = tse_dataset(sources)
    dataset.spk2enroll[('fixture', 'a')] = (dataset.speech_flt['a'],)
    with pytest.raises(EnrollmentAugmentationProviderError, match='independent enrollment'):
        dataset[0]


def test_timeline_zero_overlap_has_no_samples_and_preserves_source():
    source = np.arange(10, dtype=np.float32)
    plan = plan_overlap_window(10, 20, ratio=0, mode='head', rng=np.random.default_rng(1))
    assert not plan.turns and np.count_nonzero(plan.apply(source)) == 0
    assert scene_activity_facts((np.ones(20), plan.apply(source)))['overlap_ratio'] == 0
    np.testing.assert_array_equal(source, np.arange(10))


def test_new_count_distribution_rejects_double_controls():
    with pytest.raises(ValueError, match='removed'):
        SimulationConfigs.from_dict({'interference_count_distribution': [0, 1], 'prob_interference': 1.}, strict=True)


def test_enrollment_augmentation_does_not_change_scene(sources, monkeypatch):
    plain = tse_dataset(sources)
    augmented = tse_dataset(sources)
    from tasks.tse.data.enrollment_augment import EnrollmentAugmentor
    monkeypatch.setattr(
        "core.data.augment.random_eq",
        lambda waveform, *_args, **_kwargs: (waveform + 0.001).astype(np.float32),
    )
    augmented.enrollment_augmentor = EnrollmentAugmentor.from_config({'enabled': True, 'prob_eq': 1., 'eq_db_range': [8., 8.]})
    first, second = plain[0], augmented[0]
    np.testing.assert_array_equal(first["mixture"], second["mixture"])
    np.testing.assert_array_equal(
        first["targets"][0]["target"], second["targets"][0]["target"]
    )
    first_cue = first["targets"][0]["cues"].speaker_waveform
    second_cue = second["targets"][0]["cues"].speaker_waveform
    assert first_cue is not None and second_cue is not None
    assert not np.array_equal(first_cue.value, second_cue.value)


def test_music_pool_shared_by_se_and_ss(sources):
    from tasks.se.data.dynamic_dataset import DynamicMixingDataset
    common = {key: value for key, value in sources.items() if key != 'spk2enroll_json'}
    se = DynamicMixingDataset(**common, music_scp=sources['noise_scp'], target_sr=8000,
        max_duration_seconds=0.05, filter_by_duration=False,
        simulation_cfg={'music_ratio': 1., 'prob_wind_noise_sidechain': 0., 'prob_channel_enabled': False})
    ss = SSDynamicMixingDataset(sources['speech_scp'], num_sources=2, sample_rate=8000,
        segment_seconds=0.05, music_scp=sources['noise_scp'], simulation_cfg={'music_ratio': 1.})
    assert se.get_sample_with_trace(0).sample[0].shape == (400,)
    assert ss[0]['mixture'].shape == (400,)


def test_se_direct_dataset_rejects_relative_timeline_config(sources):
    from tasks.se.data.dynamic_dataset import DynamicMixingDataset

    with pytest.raises(ValueError, match="TSE/SS/SSL multi-source-only"):
        DynamicMixingDataset(
            sources['speech_scp'],
            noise_scp=sources['noise_scp'],
            rir_scp=sources['rir_scp'],
            target_sr=8000,
            max_duration_seconds=0.05,
            filter_by_duration=False,
            simulation_cfg={"prob_timeline": 1.0},
        )


def test_ss_persistent_workers_observe_epoch_and_match_direct_samples(sources):
    dataset = SSDynamicMixingDataset(sources['speech_scp'], num_sources=2, sample_rate=8000, segment_seconds=0.05)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1,
                                        persistent_workers=True, collate_fn=ss_collate_fn,
                                        multiprocessing_context='spawn', timeout=30)
    try:
        first = next(iter(loader)).mixture
        torch.testing.assert_close(first[0], dataset[0]['mixture'])
        dataset.set_epoch(3, base_seed=71)
        second = next(iter(loader)).mixture
        torch.testing.assert_close(second[0], dataset[0]['mixture'])
        assert not torch.equal(first, second)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_full_rir_images_close_but_early_targets_do_not(sources, tmp_path):
    rir = np.zeros(31, dtype=np.float32)
    rir[0], rir[30] = 1.0, 0.7
    path = tmp_path / 'late.wav'
    sf.write(path, rir, 8000, subtype='FLOAT')
    manifest = tmp_path / 'late.scp'
    manifest.write_text(f'late 8000 31 {path}\n')
    dataset = SSDynamicMixingDataset(sources['speech_scp'], num_sources=2,
        sample_rate=8000, segment_seconds=0.05, rir_scp=str(manifest),
        simulation_cfg={'prob_reverberation': 1., 'early_rir_sec': 0.001})
    traced = dataset.get_sample_with_trace(0, capture_stems=True)
    stems = traced.trace.get_source('scene').stems
    np.testing.assert_allclose(stems['mixture'], stems['source_full_rir/0'] + stems['source_full_rir/1'], atol=1e-7)
    assert np.max(np.abs(stems['mixture'] - stems['target/0'] - stems['target/1'])) > 0.01


def test_observation_degradation_does_not_modify_source_targets(sources):
    from core.data.simulation_views import PostMixSimulationView
    from core.data.simulation.scene_renderer import finish_speech_scene, render_speech_scene
    from core.data.simulation.role_policy import AcousticActor, AcousticRole, RIRPolicy
    dataset = SSDynamicMixingDataset(sources['speech_scp'], num_sources=2, sample_rate=8000, segment_seconds=0.05,
        simulation_cfg={**{key: 0.0 for key in PostMixSimulationView.__annotations__ if key.startswith('prob_')},
                        'noisy_post_mix_degradations_enabled': True, 'prob_clip': 1.,
                        'clip_variants': ['fixed_rail'], 'clip_variant_probs': {'fixed_rail': 1.},
                        'clip_pre_drive_db_range': [30., 30.]})
    waves = (np.full(400, 0.2, dtype=np.float32), np.full(400, 0.1, dtype=np.float32))
    actors = tuple(
        AcousticActor(
            uid=f"source/{index}",
            role=AcousticRole.SOURCE_SPEECH,
            rir_policy=RIRPolicy.MONO_SOURCE,
            waveform=wave,
        )
        for index, wave in enumerate(waves)
    )
    scene = render_speech_scene(dataset, actors, 8000, np.random.default_rng(1), sir_range=[0,0], need_noise=False)
    final = finish_speech_scene(dataset, scene, 8000, np.random.default_rng(2))
    np.testing.assert_array_equal(final.bundle.require('target/0'), scene.bundle.require('target/0'))
    assert not np.array_equal(final.bundle.require('mixture'), scene.bundle.require('mixture'))


def test_cpu_layer_f_keeps_k_targets_and_changes_one_mixture(sources):
    dataset = SSDynamicMixingDataset(
        sources['speech_scp'], num_sources=2, sample_rate=8000, segment_seconds=0.05
    )
    batch = ss_collate_fn([dataset[0]])
    targets = batch.sources.clone()
    cfg = SimulationConfigs.from_dict(
        {
            'noisy_post_mix_degradations_enabled': True,
            'prob_clip': 1.0,
            'clip_variants': ['fixed_rail'],
            'clip_variant_probs': {'fixed_rail': 1.0},
            'clip_pre_drive_db_range': [30.0, 30.0],
        },
        strict=True,
    )
    mixture = batch.mixture[0].numpy()
    degraded = apply_post_augmentations(
        mixture.copy(), 8000, cfg, rng=np.random.default_rng(0)
    )
    assert degraded.shape == mixture.shape
    assert not np.array_equal(degraded, mixture)
    torch.testing.assert_close(batch.sources, targets)


def test_tse_multi_target_scene_preserves_existing_collate(sources):
    from tasks.tse.data.collate import tse_collate_fn
    dataset = tse_dataset(sources, prob_interference_noise_only=1., prob_target_interference_noise=0.)
    dataset.targets_per_mixture = 3
    sample = dataset[0]
    assert len(sample['targets']) == 3
    assert sample["targets"][0]["target_present"] is False
    assert all(row["target_present"] is True for row in sample["targets"][1:])
    batch = tse_collate_fn([sample], cue_specs=dataset.cue_specs)
    assert batch.mixture.shape[0] == 1 and batch.target.shape[0] == 3
    assert batch.num_targets.tolist() == [3]
    assert batch.target_present.tolist() == [False, True, True]


def test_prob_noise_only_scene_reports_no_present_speaker(sources):
    dataset = tse_dataset(sources, prob_noise_only=1., prob_target_interference_noise=0.)
    traced = dataset.get_sample_with_trace(0)
    facts = traced.trace.metadata_dict()['scene']
    assert facts['source_uids'] == [] and facts['num_speakers_in_scene'] == 0
    assert facts['activity']['active_source_count'] == 0
    assert facts['decomposable_final'] is True


PASE_DNS5_RIR_CONFIGS = {
    'se/urgent2026/pase/phase1_vocoder_single.yaml',
    'se/urgent2026/pase/phase2_dewavlm_distill.yaml',
    'se/urgent2026/pase/phase3_vocoder_dual_gan.yaml',
}


def test_all_config_rir_paths_match_published_manifests():
    from core.utils.config import load_merged_config
    root = Path(__file__).resolve().parents[1] / 'configs'
    checked = 0
    for task in ('se', 'aec', 'tse', 'ss'):
        for path in (root / task).rglob('*.yaml'):
            cfg = load_merged_config(path)
            data = cfg.get('data', {})
            rate = data.get('sample_rate')
            # 两份 48 kHz PostNet 保留原有 16 kHz 资产 + runtime 重采样策略。
            source_rate = 16000 if rate == 48000 else rate
            rel = path.relative_to(root).as_posix()
            def check(mapping, split, enrollment=False):
                nonlocal checked
                if not isinstance(mapping, dict):
                    return
                for key, value in mapping.items():
                    if key == 'rir_scp' and value is not None:
                        if rel in PASE_DNS5_RIR_CONFIGS:
                            # PASE released 配方使用官方 DNS5 RIR 清单，
                            # 不是共享 gpuRIR 清单；这是封闭复现例外。
                            expected = f'data/se/pase/manifests/dns5_rir_{split}.scp'
                        elif task == 'tse' and not enrollment:
                            expected = f'data/shared/rirs/gpurir_static/gpurir/fs{source_rate}/tse/{split}_rirs.scp'
                        elif task == 'tse' and enrollment:
                            expected = f'data/shared/rirs/gpurir_static/gpurir/fs{source_rate}/static/near/{split}_rirs.scp'
                        else:
                            expected = f'data/shared/rirs/gpurir_static/gpurir/fs{source_rate}/{split}_rirs.scp'
                        assert value == expected, str(path)
                        checked += 1
                    check(value, split, enrollment or key == 'enrollment_augment')
            check(data.get('train'), 'train')
            check(data.get('validation'), 'val')
    assert checked > 50


@pytest.mark.unit
def test_ss_layer_b_prob_target_only_vs_prob_target_noise(sources):
    from tools.verify.simulation_audit_contract import build_ss_trace_metadata_envelope

    common = {
        key: sources[key]
        for key in ("speech_scp", "noise_scp", "rir_scp")
    }
    common.update(num_sources=2, sample_rate=8000, segment_seconds=0.05)
    prob_target_only = SSDynamicMixingDataset(**common, simulation_cfg={
        "prob_target_only": 1.0, "prob_target_noise": 0.0, "prob_noise_only": 0.0,
    })
    prob_target_noise = SSDynamicMixingDataset(**common, simulation_cfg={
        "prob_target_only": 0.0, "prob_target_noise": 1.0, "prob_noise_only": 0.0,
    })
    only = prob_target_only.get_sample_with_trace(0, capture_stems=True)
    noisy = prob_target_noise.get_sample_with_trace(0, capture_stems=True)
    only_scene = only.trace.metadata_dict()["scene"]
    noisy_scene = noisy.trace.metadata_dict()["scene"]
    assert only_scene["branch"] == "prob_target_only"
    assert noisy_scene["branch"] == "prob_target_noise"
    assert only_scene["noise_present"] is False
    assert noisy_scene["noise_present"] is True
    torch.testing.assert_close(only.sample["mixture"], only.sample["sources"][0] + only.sample["sources"][1])
    assert not torch.equal(noisy.sample["mixture"], noisy.sample["sources"][0] + noisy.sample["sources"][1])
    envelope = build_ss_trace_metadata_envelope(noisy.trace)
    assert envelope.task == "ss"
    assert envelope.audit_format == "source-scene-stems-v2"
    assert len(envelope.payload["stem_descriptors"]) >= 4


@pytest.mark.unit
def test_ss_dynamic_duration_filter_and_sampling(sources):
    from tasks.ss.data.config import SSDatasetConfig

    cfg = {
        "meta": {"seed": 7},
        "data": {
            "mode": "dynamic_ss",
            "sample_rate": 8000,
            "strict_load": True,
            "train": {
                "speech_scp": sources["speech_scp"],
                "segment_seconds": 0.05,
                "filter_by_duration": True,
                "filter_min_duration": 0.05,
                "filter_max_duration": 1.0,
                "sampling": {"speech": {"speaker": {"max_samples_per_epoch": 4}}},
                "simulation_cfg": {"prob_target_only": 1.0, "prob_target_noise": 0.0, "prob_noise_only": 0.0},
            },
            "validation": {
                "speech_scp": sources["speech_scp"],
                "segment_seconds": 0.05,
                "filter_by_duration": False,
                "simulation_cfg": {"prob_target_only": 1.0, "prob_target_noise": 0.0, "prob_noise_only": 0.0},
            },
        },
        "ss": {"num_sources": 2, "sir_range": [-5.0, 5.0]},
    }
    resolved = SSDatasetConfig.from_config(cfg, "train")
    dataset = SSDynamicMixingDataset(
        **resolved.dynamic_kwargs,
        speech_scp=resolved.speech_scp,
        num_sources=resolved.num_sources,
        sample_rate=resolved.sample_rate,
        segment_seconds=resolved.segment_seconds,
        sir_range=resolved.sir_range,
        seed=resolved.seed,
    )
    assert len(dataset) == 3
    assert dataset._speech_sampling_config.speaker.max_samples_per_epoch == 4
