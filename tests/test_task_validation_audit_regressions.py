"""任务 validation 的真实 loss、输入合同与两 rank 故障回归。"""

from datetime import timedelta
from pathlib import Path
import time

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset

from core.data.simulation.array_profiles import resolve_microphone_array
from core.losses.composite import CompositeLoss
from core.losses.model_specific.deepfilternet import (
    DeepFilterNetLSNRComponent,
    DfAlphaLossComponent,
)
from core.models.io_contract import ModelIOContract, TensorField
from core.stft import StftIstftModule
from tasks.aec.data.batch import AECBatch
from tasks.aec.trainer.aec_trainer import AECTrainer
from tasks.se.data.batch import SEBatch
from tasks.se.trainer.dual_input_loss_trainer import DualInputLossTrainer
from tasks.se.trainer.gan_trainer import GANTrainer
from tasks.se.trainer.meanflowse_trainer import MeanFlowSETrainer
from tasks.se.trainer.se_trainer import SETrainer
from tasks.ssl.data.batch import SSLBatch
from tasks.ssl.losses import SSLSpatialMSELoss, ipd_mse_loss
from tasks.ssl.trainer.ssl_trainer import SSLTrainer
from tasks.tse.cue_contract import (
    TSECue,
    TSECueBatch,
    TSECueBatches,
    TSECueSpec,
    TSECueSpecs,
    TSECues,
)
from tasks.tse.data.batch import TSEBatch
from tasks.tse.data.collate import make_tse_sample, make_tse_target, tse_collate_fn
from tasks.tse.trainer.tse_trainer import TSETrainer


class _MSE(nn.Module):
    def forward(self, enhanced, clean, lengths=None, **kwargs):
        return (enhanced - clean).square().mean()


class _SEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.5))

    def forward(self, noisy, lengths=None):
        return noisy * self.gain


class _AuxModel(_SEModel):
    def __init__(self):
        super().__init__()
        self.lsnr = nn.Parameter(torch.tensor(2.0))
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.requests = []

    def forward(self, noisy, lengths=None, *, return_lsnr=False, return_alpha=False):
        self.requests.append((return_lsnr, return_alpha))
        enhanced = noisy * self.gain
        if return_lsnr:
            shape = (noisy.shape[0], (noisy.shape[-1] - 8) // 4 + 1)
            return enhanced, self.lsnr.expand(shape), self.alpha.expand(shape)
        return enhanced


class _ScalarModel(_SEModel):
    def forward(self, noisy, clean, lengths=None):
        return (noisy * self.gain - clean).square().mean()


class _SpatialModel(_SEModel):
    """声明真实 SSL 输出合同，保留原先的标量梯度测试语义。"""

    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({'ssl'}),
        inputs=(
            TensorField('features', 'embedding', ('batch', 'frame', 'frequency', 'feature')),
        ),
        outputs=(
            TensorField(
                'dpipd', 'spectrum',
                ('batch', 'frame', 'frequency', 'track', 'pair', 'complex'),
                multiple=True,
            ),
        ),
        forward_kind='estimate', frontend_owner='model',
    )

    def forward(self, features):
        return features.unsqueeze(-2).unsqueeze(-2) * self.gain


class _CustomSpatialLoss(nn.Module):
    SSL_SPATIAL_LOSS = True

    def forward(self, prediction, *, target_dpipd, output_lengths, doa, vad, **kwargs):
        return ipd_mse_loss(
            prediction,
            target_dpipd=target_dpipd,
            output_lengths=output_lengths,
            doa=doa,
            vad=vad,
            **kwargs,
        )


class _NativeAEC(_SEModel):
    supports_multichannel_input = True
    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({'aec'}),
        inputs=(
            TensorField('mic_spectra', 'spectrum', ('batch', 'microphone', 'frequency', 'frame', 'complex')),
            TensorField('farend_spectra', 'spectrum', ('batch', 'frequency', 'frame', 'complex')),
        ),
        outputs=(TensorField('nearend_estimate_spectra', 'spectrum', ('batch', 'frequency', 'frame', 'complex')),),
        forward_kind='estimate', frontend_owner='runtime',
    )

    def forward(self, data):
        self.mic_seen = data['mic'].detach().clone()
        self.spec_seen = data['mic_spectra'].detach().clone()
        return {'nearend_estimate_spectra': data['mic_spectra'][:, 1] * self.gain}


class _SpatialDataset(Dataset):
    def __init__(self, batch):
        self.batch = batch
        self.resolved_microphone_array = resolve_microphone_array(
            array_profile='linear_2ch_80mm', reference_channel=1,
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.batch


def _se_batch():
    clean = torch.full((2, 64), 0.1)
    noisy = torch.full((2, 64), 0.6)
    return SEBatch(clean, noisy, torch.full((2,), 16000), torch.tensor([64, 48]))


def _trainer(path, cls=SETrainer, *, model=None, loss=None, batch=None,
             stft=None, trainer_cfg=None, model_args=None, rank=0, world_size=1):
    model = _SEModel() if model is None else model
    batch = _se_batch() if batch is None else batch
    loader = DataLoader([batch], batch_size=None)
    if isinstance(model, dict):
        optim = {key: torch.optim.SGD(value.parameters(), lr=0.01) for key, value in model.items()}
        scheduler = {key: torch.optim.lr_scheduler.ConstantLR(value, factor=1.0) for key, value in optim.items()}
    else:
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optim, factor=1.0)
    config = {
        'model': {'type': 'AuditTinyModel', 'args': model_args or {}},
        'trainer': {
            'exp_path': str(path), 'epochs': 1, 'strict_checks': True,
            'validation_epoch_interval': 0, 'save_curve_interval': 0,
            'save_checkpoint_interval': 0, 'log_interval': 100,
            'val_metrics': {'enabled': False}, **(trainer_cfg or {}),
        },
        'meta': {'use_amp': False},
        'data': {'sample_rate': 16000, 'train': {'max_duration': 64}},
    }
    return cls(model=model, optimizer=optim, scheduler=scheduler,
               loss_fn=_MSE() if loss is None else loss, config=config,
               train_loader=loader, val_loader=loader, stft_module=stft,
               rank=rank, world_size=world_size, sample_rate=16000)


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


@pytest.mark.unit
def test_se_real_lsnr_alpha_components_match_training_and_validation(tmp_path, cpu):
    loss = CompositeLoss([
        {'type': 'DeepFilterNetLSNRComponent', 'weight': 0.2, 'args': {'frame_length': 8, 'hop_length': 4}},
        {'type': 'DfAlphaLossComponent', 'weight': 0.7, 'args': {'frame_length': 8, 'hop_length': 4}},
    ], strict=True)
    trainer = _trainer(tmp_path, model=_AuxModel(), loss=loss,
                       trainer_cfg={'use_lsnr_loss': True})
    batch = _se_batch()
    train_loss, finite, _ = trainer._forward_and_compute_loss(batch, 1, 0, 16000)
    train_loss.backward()
    assert finite and trainer.model.lsnr.grad.abs() > 0 and trainer.model.alpha.grad.abs() > 0
    with torch.inference_mode():
        val_loss = trainer._val_forward_loss(batch, 1, 1)[0]
        expected_lsnr = DeepFilterNetLSNRComponent(8, 4)(
            batch.noisy * 0.5, batch.clean, batch.lengths,
            predicted_lsnr=torch.full((2, 15), 2.0), noisy=batch.noisy,
        )
        expected_alpha = DfAlphaLossComponent(8, 4)(
            batch.noisy * 0.5, batch.clean, batch.lengths,
            predicted_alpha=torch.full((2, 15), 0.3), noisy=batch.noisy,
        )
    assert expected_lsnr > 0 and expected_alpha > 0
    torch.testing.assert_close(train_loss, expected_lsnr * 0.2 + expected_alpha * 0.7)
    torch.testing.assert_close(val_loss, train_loss)
    assert trainer.model.requests == [(True, True), (True, True)]


@pytest.mark.unit
def test_aec_metrics_use_nonzero_reference_without_projecting_model_input(tmp_path, cpu):
    mic = torch.stack((torch.full((2, 64), 0.1), torch.full((2, 64), 0.6)), dim=1)
    batch = AECBatch(nearend=torch.full((2, 64), 0.2), mic=mic,
                     farend=torch.full((2, 64), 0.3), fs=torch.tensor([16000, 16000]),
                     lengths=torch.tensor([64, 48]))
    trainer = _trainer(tmp_path, AECTrainer, model=_NativeAEC(), batch=batch,
                       stft=StftIstftModule(n_fft=16, hop_length=4, win_length=16))
    trainer.val_loader = DataLoader(_SpatialDataset(batch), batch_size=None)
    with torch.inference_mode():
        result = trainer._val_forward_loss(batch, 1, 1)
        context = result[-1]
        cached = trainer._materialize_val_waveforms(context)
    torch.testing.assert_close(trainer.model.mic_seen[0], mic[0])
    torch.testing.assert_close(trainer.model.mic_seen[1, :, :48], mic[1, :, :48])
    assert trainer.model.spec_seen.shape[:2] == (2, 2)
    assert context['mic'].shape == (2, 2, 64)
    np.testing.assert_array_equal(cached['mic'][0], mic[0, 1].numpy())
    np.testing.assert_array_equal(cached['mic'][1, :48], mic[1, 1, :48].numpy())
    assert trainer._materialize_val_waveforms(context) is cached
    info = trainer._collect_step_diagnostics(
        {'error_mask': torch.tensor([True, False])}, torch.zeros(2, 64), batch,
    )
    assert info['error_mask_ratio'] == 0.5


@pytest.mark.unit
@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('kind', ['se', 'dual', 'meanflow', 'gan', 'tse'])
def test_validation_input_sanitization_and_strict_semantics(tmp_path, cpu, kind, strict):
    batch = _se_batch()
    batch.clean[0, 2] = float('nan')
    batch.noisy[1, 3] = float('inf')
    classes = {'se': SETrainer, 'dual': DualInputLossTrainer,
               'meanflow': MeanFlowSETrainer, 'gan': GANTrainer, 'tse': TSETrainer}
    model = _ScalarModel() if kind == 'dual' else _SEModel()
    if kind == 'gan':
        model = {'generator': model, 'discriminator': _SEModel()}
    stft = StftIstftModule(n_fft=16, hop_length=4, win_length=16) if kind in ('meanflow', 'tse') else None
    trainer = _trainer(tmp_path, classes[kind], model=model, stft=stft,
                       trainer_cfg={'strict_checks': strict})
    if kind == 'tse':
        rows, samples = batch.clean.shape
        speaker_waveform = TSECueBatch(
            representation='waveform',
            value=batch.noisy.clone(),
            present=torch.ones(rows, dtype=torch.bool),
            lengths=batch.lengths.to(torch.int32),
            timebase=None,
        )
        raw = TSEBatch(
            mixture=batch.noisy.unsqueeze(1),
            target=batch.clean,
            sr=batch.sr[0].to(torch.int32),
            lengths=batch.lengths.to(torch.int32),
            num_targets=torch.ones(rows, dtype=torch.int32),
            spk_labels=torch.full((rows,), -1, dtype=torch.int64),
            target_present=torch.ones(rows, dtype=torch.bool),
            cues=TSECueBatches(speaker_waveform=speaker_waveform),
            cue_specs=TSECueSpecs(
                speaker_waveform=TSECueSpec('waveform', (samples,))
            ),
            keys=tuple(f'sample-{index}' for index in range(rows)),
            speaker_ids=tuple(f'speaker-{index}' for index in range(rows)),
            geometry=(None,) * rows,
            timebase=(None,) * rows,
            provenance=(None,) * rows,
        )
        operation = lambda: trainer._val_prepare_batch(raw, 1, 1)
    elif kind == 'gan':
        raw = batch
        trainer.val_loader = DataLoader(
            [raw], batch_size=None, collate_fn=lambda item: item
        )
        seen = []
        def forward(data):
            seen.append(data)
            return {**data, 'enhanced': data['noisy'] * 0.5}
        trainer._forward_generator = forward
        operation = lambda: trainer._validation_epoch(1)
    elif kind == 'dual':
        operation = lambda: trainer._prepare_scalar_validation_batch(batch, 1, 1)
    else:
        operation = lambda: trainer._val_prepare_batch(batch, 1, 1)
    if strict:
        with pytest.raises((AssertionError, ValueError), match='NaN|Inf|finite'):
            operation()
    else:
        prepared = operation()
        if kind == 'gan':
            tensors = (seen[0]['clean'], seen[0]['noisy'])
        elif kind == 'tse':
            prepared_speaker = prepared.batch.cues.speaker_waveform
            assert prepared_speaker is not None
            tensors = (
                prepared.batch.target,
                prepared.batch.mixture[:, 0],
                prepared_speaker.value,
            )
        elif kind == 'se':
            tensors = prepared[1:3]
        else:
            tensors = prepared[:2]
        assert all(torch.isfinite(value).all() for value in tensors)
        assert tensors[0][0, 2] == 0 and tensors[1][1, 3] == 0


@pytest.mark.unit
@pytest.mark.parametrize('error', [torch.cuda.OutOfMemoryError('oom'), RuntimeError('CUDA error: device-side assert triggered')])
def test_dual_fatal_validation_never_enters_object_collective(tmp_path, cpu, monkeypatch, error):
    trainer = _trainer(tmp_path, DualInputLossTrainer, model=_ScalarModel())
    trainer.world_size = 2
    def fail():
        raise error
    monkeypatch.setattr(trainer, '_sync_train_batch_exception',
                        lambda *args, **kwargs: pytest.fail('fatal CUDA entered object collective'))
    with pytest.raises(type(error), match=str(error)):
        trainer._run_synced_validation_phase(fail, phase='validation_forward_error', context='dual')


class _LossWrapper(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize('wrapped', [False, True])
def test_ssl_direct_constructor_rejects_nonspatial_loss(tmp_path, cpu, wrapped):
    loss = _LossWrapper(_MSE()) if wrapped else _MSE()
    with pytest.raises(ValueError, match='SSL_SPATIAL_LOSS'):
        _trainer(tmp_path, SSLTrainer, model=_SpatialModel(), loss=loss)


@pytest.mark.unit
def test_ssl_direct_constructor_rejects_waveform_metrics(tmp_path, cpu):
    with pytest.raises(ValueError, match='waveform trainer.val_metrics'):
        _trainer(tmp_path, SSLTrainer, model=_SpatialModel(),
                 loss=SSLSpatialMSELoss(allow_no_stft=True),
                 trainer_cfg={'val_metrics': {'enabled': True}})


@pytest.mark.unit
@pytest.mark.parametrize('loss', [SSLSpatialMSELoss(allow_no_stft=True), _CustomSpatialLoss()])
def test_ssl_direct_constructor_accepts_spatial_capability_and_runs(tmp_path, cpu, loss):
    batch = SSLBatch(
        features=torch.ones(1, 2, 3, 2),
        target_dpipd=torch.zeros(1, 2, 3, 1, 1, 2),
        input_lengths=torch.tensor([2]),
        output_lengths=torch.tensor([2]),
        pair_index=torch.tensor([[[0, 1]]]),
        pair_mask=torch.ones(1, 1, dtype=torch.bool),
        geometry=torch.tensor([[[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]]),
        frame_origin=torch.tensor([0]),
        time_stride=1,
        valid_output_mask=torch.ones(1, 2, dtype=torch.bool),
        doa=torch.zeros(1, 2, 1, 2),
        vad=torch.ones(1, 2, 1),
        sample_rates=torch.tensor([16000]),
        provenance=({'backend': 'fixture'},),
        input_representation="reference_ipd",
        target_representation="dpipd_far_field",
        coordinate_frame="array_local",
    )
    trainer = _trainer(tmp_path, SSLTrainer, model=_SpatialModel(), loss=loss, batch=batch)
    value = trainer._forward_and_compute_loss(batch, 1, 0, 16000)[0]
    value.backward()
    torch.testing.assert_close(trainer.model.gain.grad, torch.tensor(2.0))
    with torch.inference_mode():
        torch.testing.assert_close(trainer._val_forward_loss(batch, 1, 1)[0], value)


class _TSEModel(_SEModel):
    def required_tse_input_fields(self):
        # 外部 SpeakerEncoder 条件模型：与真实 spkemb 模型一样显式声明消费
        # spk_embedding，避免退回 legacy waveform enrollment 合同。
        return frozenset({'mix_spectra', 'spk_embedding'})

    def forward(self, data):
        self.embedding_seen = data['spk_embedding'].detach().clone()
        return {**data, 'target_estimate_spectra': data['mix_spectra'] * self.gain}


class _CheckpointSpeaker(nn.Module):
    def __init__(self, *, pretrained, **kwargs):
        super().__init__()
        from core.utils.safe_load import safe_torch_load
        self.embed_dim = 4
        self.projection = nn.Linear(1, 4, bias=False)
        self.load_state_dict(safe_torch_load(pretrained, purpose='task validation regression speaker fixture'))

    def forward(self, wave):
        return self.projection(wave.mean(dim=1, keepdim=True))


_TSE_SPEAKER_CUE_SPECS = TSECueSpecs(speaker_waveform=TSECueSpec('waveform', (64,)))


def _tse_val_loader():
    """假 val loader：canonical dict item 经 ``tse_collate_fn`` 产出真实 TSEBatch。"""
    clean = torch.full((2, 64), 0.1)
    noisy = torch.full((2, 64), 0.6)
    lengths = (64, 48)
    samples = [
        make_tse_sample(
            key=f'sample-{index}',
            mixture=noisy[index],
            sr=16000,
            length=lengths[index],
            targets=[
                make_tse_target(
                    target=clean[index],
                    cues=TSECues(speaker_waveform=TSECue(
                        representation='waveform', value=clean[index],
                        present=True, length=lengths[index], timebase=None,
                    )),
                    speaker_id=f'speaker-{index}',
                    spk_label=-1,
                    target_present=True,
                ),
            ],
        )
        for index in range(2)
    ]
    return DataLoader(
        samples, batch_size=2,
        collate_fn=lambda items: tse_collate_fn(items, cue_specs=_TSE_SPEAKER_CUE_SPECS),
    )


def _tse_worker(rank, root):
    torch.set_num_threads(1)
    torch.cuda.is_available = lambda: False
    root = Path(root)
    dist.init_process_group('gloo', init_method=(root / 'tse-rendezvous').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=25))
    try:
        from tasks.tse.models.speaker import encoder
        encoder.SpeakerEncoder = _CheckpointSpeaker
        checkpoint = root / 'speaker.pt'
        args = {'joint_training': False, 'features': {'spkemb': {'enabled': True, 'embed_dim': 4}},
                'speaker_model': {'speaker_encoder': {'pretrained': str(checkpoint), 'spk_args': {'embed_dim': 4}}}}
        trainer = _trainer(root / 'tse-exp', TSETrainer, model=_TSEModel(),
                           stft=StftIstftModule(n_fft=16, hop_length=4, win_length=16),
                           model_args=args, rank=rank, world_size=2)
        trainer.val_loader = _tse_val_loader()
        assert not trainer._spk_checkpoint_identity_synced
        value = trainer._validation_epoch(1)
        assert np.isfinite(value) and trainer._spk_checkpoint_identity_synced
        torch.testing.assert_close(trainer._raw_model().embedding_seen, torch.full((2, 4), 0.1))
        assert not any(p.requires_grad for p in trainer._spk_encoder.parameters())
        if rank == 1:
            trainer._spk_encoder.projection.weight.data.fill_(float('nan'))
        with pytest.raises(RuntimeError, match='rank 1.*NaN|rank 1.*finite'):
            trainer._validation_epoch(2)
        trainer._spk_encoder.projection.weight.data.fill_(1.0)
        trainer._spk_checkpoint_identity_synced = False
        if rank == 1:
            trainer._loaded_spk_checkpoint_identity = {**trainer._loaded_spk_checkpoint_identity, 'sha256': 'different'}
        with pytest.raises(RuntimeError, match='identity mismatch'):
            trainer._validation_epoch(3)
    finally:
        dist.destroy_process_group()


def _join_bounded(context, seconds=60):
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            if context.join(timeout=1):
                return
        pytest.fail('two-rank test exceeded bounded completion deadline')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)


@pytest.mark.ddp
@pytest.mark.subprocess
def test_tse_first_validation_external_identity_and_failures_two_real_ranks(tmp_path):
    torch.save({'projection.weight': torch.ones(4, 1)}, tmp_path / 'speaker.pt')
    _join_bounded(mp.spawn(_tse_worker, args=(str(tmp_path),), nprocs=2, join=False))


def _backward_worker(rank, root):
    torch.set_num_threads(1)
    torch.cuda.is_available = lambda: False
    root = Path(root)
    dist.init_process_group('gloo', init_method=(root / 'backward-rendezvous').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=25))
    from tasks.se.trainer.entry import SETrainingEntry
    trainer = _trainer(root / 'backward-exp', rank=rank, world_size=2)
    started = False
    original_gather = dist.all_gather_object
    def guarded_gather(*args, **kwargs):
        if started:
            # 触发即落盘：surfaced 的异常可能来自 peer，不能只靠 surfaced 内容判定守卫是否被绕过。
            (root / f'new-gather-{rank}').write_text('triggered')
            raise AssertionError('new object collective after reducer started')
        return original_gather(*args, **kwargs)
    dist.all_gather_object = guarded_gather
    def hook(state, bucket):
        nonlocal started
        started = True
        work = dist.all_reduce(bucket.buffer(), async_op=True)
        (root / f'reducer-{rank}').write_text('started')
        if rank == 1:
            raise RuntimeError('injected reducer backward failure')
        return work.get_future().then(lambda future: future.value()[0] / 2)
    trainer.model.register_comm_hook(None, hook)
    try:
        SETrainingEntry('SE_Audit')._run_trainer_main_loop(trainer, 2)
    except BaseException as error:
        # worker 侧独立记录注入根因，使该 rank 的失败不依赖 surfaced 是哪一个进程。
        (root / f'failure-{rank}').write_text(f'{type(error).__name__}: {error}')
        raise
    finally:
        (root / f'cleanup-{rank}').write_text(str(dist.is_initialized()))


@pytest.mark.ddp
@pytest.mark.subprocess
def test_backward_failure_exits_launcher_and_destroys_peers_without_new_gather(tmp_path, capfd):
    """rank-local 致命错误必须被 launcher 捕获并回收全部进程，且不得新增 object collective。

    surfaced 的是哪一个 rank 属于 ``mp.spawn`` 的 join 选择细节：``ProcessContext.join``
    在多个进程同时 ready 时按 ``connection.wait`` 的返回顺序遍历 ``ready``，只取第一个
    非零退出码，并 SIGTERM 其余进程（torch/multiprocessing/spawn.py ``for sentinel in
    ready:``）。本用例的 peer 会阻塞在自己的 all_reduce 上直到 gloo 超时，因此 surfaced
    的可能是 peer 的超时错误而不是注入失败的那个 rank —— 这不是产品保证。这里只要求注入
    根因出现在 surfaced 异常、worker stderr 日志或 worker 侧失败记录中，其余结构性断言
    （reducer-1 存在、cleanup-1 == 'False'、未触发 guarded gather、全部进程死亡、至少一个
    非零退出码）保持确定可验证。
    """
    context = mp.spawn(_backward_worker, args=(str(tmp_path),), nprocs=2, join=False)
    with pytest.raises(mp.ProcessRaisedException) as surfaced:
        _join_bounded(context)
    worker_log = capfd.readouterr()
    failure_record = ''.join(
        (tmp_path / f'failure-{rank}').read_text()
        for rank in range(2)
        if (tmp_path / f'failure-{rank}').exists()
    )
    visible_error = str(surfaced.value) + worker_log.err + worker_log.out + failure_record
    assert 'injected reducer backward failure' in visible_error, surfaced.value
    assert not list(tmp_path.glob('new-gather-*')), 'reducer 启动后仍出现 object collective'
    assert (tmp_path / 'reducer-1').exists()
    assert (tmp_path / 'cleanup-1').read_text() == 'False'
    assert all(not process.is_alive() for process in context.processes)
    assert any(process.exitcode != 0 for process in context.processes)
