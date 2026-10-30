"""仿真重构端到端回归：参考传递、静音梯度、采样 anchor 与会话重置。"""

import numpy as np
import pytest
import torch

from test_multisource_scene_simulation import sources
from test_tse_seven_scenes import dataset_for
from core.data.errors import RecoverableResourceError
from core.data.simulation.scene_renderer import render_speech_scene
from core.data.simulation.role_policy import AcousticActor, AcousticRole, RIRPolicy
from core.losses.build import build_loss
from core.losses.pit import PITLoss
from core.losses.time_domain import AbsentTargetLoss
from core.trainer.metrics_trainer import _waveform_loss_kwargs
from tasks.ss.data import SSDynamicMixingDataset, ss_collate_fn

pytestmark = pytest.mark.unit


def test_ss_noise_only_delivers_k_zero_targets_and_valid_audit(sources):
    from tools.verify.simulation_audit_contract import build_ss_trace_metadata_envelope

    ds = SSDynamicMixingDataset(
        sources['speech_scp'], noise_scp=sources['noise_scp'], num_sources=2,
        sample_rate=8000, segment_seconds=.05,
        simulation_cfg={'prob_target_only': 0., 'prob_noise_only': 1.},
    )
    traced = ds.get_sample_with_trace(0, capture_stems=True)
    batch = ss_collate_fn([traced.sample])
    assert batch.sources.shape == (1, 2, 400)
    assert torch.count_nonzero(batch.sources) == 0
    assert torch.count_nonzero(batch.mixture) > 0
    assert batch.source_mask.all()
    assert traced.trace.metadata['scene']['scene_spec']['source_count'] == 0
    build_ss_trace_metadata_envelope(traced.trace)


def test_ss_pit_silence_has_suppression_gradient_and_ignores_padding():
    prediction = torch.full((1, 2, 16), .2, requires_grad=True)
    loss = PITLoss()(prediction, torch.zeros_like(prediction), lengths=torch.tensor([8]))
    loss.backward()
    assert torch.all(prediction.grad[..., :8] > 0)
    assert torch.count_nonzero(prediction.grad[..., 8:]) == 0


def test_ss_sampling_anchor_and_recoverable_placeholder(sources, monkeypatch):
    ds = SSDynamicMixingDataset(sources['speech_scp'], num_sources=2,
                                sample_rate=8000, segment_seconds=.05, strict_load=False)
    fs, position = ds._get_from_index(0)
    anchor = ds.speech_uids[fs][position]
    sample = ds[0]
    assert sample['source_ids'][0] == anchor
    assert len(set(sample['source_ids'])) == 2

    def broken(*args, **kwargs):
        raise RecoverableResourceError('injected source read failure')

    monkeypatch.setattr(ds, '_build_sample', broken)
    traced = ds.get_sample_with_trace(0)
    batch = ss_collate_fn([traced.sample])
    assert traced.trace.placeholder
    assert batch.sources.shape == (1, 2, 400)


@pytest.mark.parametrize('composite', [False, True])
@pytest.mark.parametrize('silent_row', [False, True])
def test_fw_sdr_reference_reaches_final_loss_and_backward(composite, silent_row):
    component = {'type': 'FWNegSISDRLoss', 'args': {'n_fft': 128, 'hop_length': 32}}
    cfg = {'type': 'CompositeLoss', 'components': [component]} if composite else component
    loss_fn = build_loss(cfg, sample_rate=16000)
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(2, 256, generator=generator)
    if silent_row:
        target[1] = 0
    observation = target + .1 * torch.randn(2, 256, generator=generator)
    prediction = (target + .05 * torch.randn(2, 256, generator=generator)).requires_grad_()
    kwargs = _waveform_loss_kwargs(loss_fn, observation, target, prediction, 16000)
    loss = loss_fn(prediction, target, torch.tensor([256, 200]), **kwargs)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
    assert torch.count_nonzero(prediction.grad[1, 200:]) == 0


@pytest.mark.parametrize('scene', ['prob_interference_only', 'prob_interference_noise_only'])
def test_tse_absent_interference_requires_sir_at_construction(sources, scene):
    with pytest.raises(ValueError, match='interference scenarios require sir_range'):
        dataset_for(sources, scene, simulation_cfg={'sir_range': None})


def test_tse_zero_timeline_is_explicitly_inactive(sources):
    ds = dataset_for(sources, 'prob_target_interference', simulation_cfg={
        'prob_timeline': 1., 'overlap_ratio_range': [0., 0.],
    })
    traced = ds.get_sample_with_trace(0, capture_stems=True)
    facts = traced.trace.metadata['scene']['reference_facts']
    assert facts['sir'] is None
    assert facts['sir_unavailable_reason'] == 'inactive_source'
    sample = traced.sample
    assert sample["mixture"].shape == (1, 400)
    assert np.isfinite(sample["mixture"]).all()
    assert sample["targets"][0]["target_present"] is True
    assert np.isfinite(sample["targets"][0]["target"]).all()


def test_primary_snr_audit_records_the_actual_reference(sources):
    ds = SSDynamicMixingDataset(sources['speech_scp'], noise_scp=sources['noise_scp'],
        num_sources=2, sample_rate=8000, segment_seconds=.05,
        simulation_cfg={'snr_reference': 'primary', 'source_volume_normalization_enabled': False})
    a = np.sin(np.arange(400) * .2).astype('float32')
    b = (2 * np.cos(np.arange(400) * .3)).astype('float32')
    scene = render_speech_scene(
        ds,
        (
            AcousticActor("a", AcousticRole.SOURCE_SPEECH, RIRPolicy.MONO_SOURCE, a),
            AcousticActor("b", AcousticRole.SOURCE_SPEECH, RIRPolicy.MONO_SOURCE, b),
        ),
        8000,
        np.random.default_rng(7),
        sir_range=[0, 0],
        sir_mode='none',
    )
    actual = scene.bundle.require('source_full_rir/0')
    assert scene.facts['reference_facts']['snr']['energy'] == pytest.approx(
        float(np.sum(actual.astype('float64') ** 2)))


def test_ort_enrollment_reset_discards_temporal_state():
    from onnx.infer.infer_streaming_tse_ort import TSEStreamingORTProcessor

    class Runner:
        def __init__(self):
            self.state = {'spkemb': np.array([1.], dtype='float32'),
                          'separator_00_h': np.array([9.], dtype='float32')}

        def reset(self, state=None):
            self.state = ({k: np.zeros_like(v) for k, v in self.state.items()}
                          if state is None else state)

    processor = TSEStreamingORTProcessor.__new__(TSEStreamingORTProcessor)
    processor.runner = Runner()
    processor._condition_names = ('spkemb',)
    processor.reset({'spkemb': np.array([2.], dtype='float32')})
    np.testing.assert_array_equal(processor.runner.state['spkemb'], [2.])
    np.testing.assert_array_equal(processor.runner.state['separator_00_h'], [0.])


def test_absent_loss_reports_true_global_reduction_mass():
    target = torch.tensor([[0., 0.], [1., 1.]])
    prediction = torch.tensor([[2., 2.], [1., 1.]], requires_grad=True)
    loss_fn = AbsentTargetLoss()
    mass0 = loss_fn.ddp_global_mean_mass(prediction[:1], target[:1])
    mass1 = loss_fn.ddp_global_mean_mass(prediction[1:], target[1:])
    assert mass0 == 1 and mass1 == 0
    global_loss = loss_fn(prediction, target)
    weighted_rank_mean = (loss_fn(prediction[:1], target[:1]) * mass0
                          + loss_fn(prediction[1:], target[1:]) * mass1) / (mass0 + mass1)
    torch.testing.assert_close(global_loss, weighted_rank_mean)


def test_tse_runtime_identity_changes_with_weights():
    from test_tse_streaming_deployment import _flat_model
    from onnx.runtime_spec import runtime_spec_from_wrapper, TSERuntimeSpec

    flat = _flat_model()
    first = runtime_spec_from_wrapper(flat, 'tse').tse
    assert TSERuntimeSpec.from_metadata(first.to_metadata()) == first
    with torch.no_grad():
        next(flat.model.parameters()).add_(.01)
    second = runtime_spec_from_wrapper(flat, 'tse').tse
    assert first.cache_names == second.cache_names
    assert first.model_sha256 != second.model_sha256


@pytest.mark.ort
def test_real_ort_reset_matches_new_session(tmp_path):
    pytest.importorskip('onnxruntime')
    from test_tse_streaming_deployment import _flat_model
    from onnx.export import write_streaming_runtime_metadata
    from onnx.utils import export_onnx, create_ort_session
    from onnx.infer.infer_streaming_tse_ort import TSEStreamingORTProcessor

    flat = _flat_model()
    frame = torch.randn(1, 257, 1, 2)
    cache = flat.init_cache_flat(1, 'cpu', torch.float32)
    path = tmp_path / 'reset.onnx'
    export_onnx(flat, (frame, *cache), str(path),
                input_names=flat.get_input_names(), output_names=flat.get_output_names(),
                dynamic_axes=flat.get_dynamic_axes(), opset_version=17, dynamo=False)
    write_streaming_runtime_metadata(str(path), flat)
    session = create_ort_session(path)
    first_condition = {'spkemb': torch.ones_like(cache[0])}
    next_condition = {'spkemb': torch.full_like(cache[0], .3)}
    reused = TSEStreamingORTProcessor(session, first_condition)
    fresh = TSEStreamingORTProcessor(session, next_condition)
    reused(frame)
    reused(frame * .7)
    reused.reset(next_condition)
    for value in (frame * .2, frame * -.4):
        torch.testing.assert_close(reused(value), fresh(value), rtol=1e-5, atol=2e-6)


def test_se_explicit_three_scene_distribution_is_not_ignored(sources):
    from tasks.se.data.dynamic_dataset import DynamicMixingDataset

    common = dict(speech_scp=sources['speech_scp'], noise_scp=sources['noise_scp'],
                  rir_scp=sources['rir_scp'], target_sr=8000,
                  max_duration_seconds=.05, filter_by_duration=False)
    with pytest.raises(ValueError, match='three scenario probabilities must sum to 1'):
        DynamicMixingDataset(**common, simulation_cfg={
            'prob_target_only': .2, 'prob_noise_only': .2, 'prob_target_noise': 0.,
        })
    ds = DynamicMixingDataset(**common, simulation_cfg={
        'prob_target_only': .2, 'prob_noise_only': .2, 'prob_target_noise': .6,
        'prob_wind_noise_sidechain': 0., 'prob_channel_enabled': False,
    })
    assert ds._scenario_probabilities() == {
        'prob_target_only': .2, 'prob_noise_only': .2, 'prob_target_noise': .6,
    }


@pytest.mark.parametrize('key', ['prob_clean_only', 'prob_nearend_echo', 'prob_speech_only'])
def test_removed_scene_keys_fail_even_with_nonstrict_schema(key):
    from core.data.simulation_configs import SimulationConfigs

    with pytest.raises(ValueError, match='removed'):
        SimulationConfigs.from_dict({key: 1.}, strict=False)


def test_ss_offline_lenient_rejects_nonfinite_and_uses_placeholder_length(tmp_path, monkeypatch):
    import json
    import soundfile as sf
    from tasks.ss.data import build_dataset

    wav = tmp_path / 'source.wav'
    sf.write(wav, np.ones(16, dtype='float32'), 8000, subtype='FLOAT')
    manifest = tmp_path / 'sources.jsonl'
    manifest.write_text(json.dumps({'uid': 'one', 'fs': 8000, 'length': 16,
        'mixture': str(wav), 'sources': [
            {'source_id': 'a', 'path': str(wav), 'length': 16},
            {'source_id': 'b', 'path': str(wav), 'length': 16},
        ]}) + '\n')
    cfg = {'data': {'mode': 'offline_ss', 'sample_rate': 8000, 'strict_load': False,
                    'placeholder_num_samples': 37, 'train': {'manifest': str(manifest)}},
           'ss': {'num_sources': 2}}
    ds = build_dataset(cfg)
    monkeypatch.setattr('tasks.ss.data.offline_dataset.load_audio',
                        lambda _: (torch.full((1, 16), float('nan')), 8000))
    with pytest.raises(ValueError, match='NaN/Inf'):
        ds[0]

    def missing(_):
        raise FileNotFoundError('injected missing audio')

    monkeypatch.setattr('tasks.ss.data.offline_dataset.load_audio', missing)
    sample = ds[0]
    assert sample['mixture'].shape == (37,)
    assert len(sample['sources']) == 2
