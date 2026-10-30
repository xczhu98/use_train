"""真实谱 cue 在 runtime/model 边界的有效长度回归。"""

import pytest
import torch
import torch.nn.functional as F

from core.stft import StftIstftModule
from tasks.tse.model_io import build_tse_model_inputs
from tasks.tse.models.discriminative.tse_bsrnn_spk import TSE_BSRNN_SPK

def _model(cues=("tfmap",), causal=False):
    return TSE_BSRNN_SPK(
        feature_dim=8, num_repeat=1, causal=causal,
        n_fft=512, hop_length=128, win_length=512,
        features={name: {"enabled": name in cues}
                  for name in ("listen", "usef", "tfmap", "context", "spkemb")},
    ).eval()


def _inputs(model, mixture, enroll, lengths):
    return build_tse_model_inputs(
        model=model, stft=StftIstftModule(n_fft=512, hop_length=128, win_length=512),
        mixture=mixture, enroll=enroll, enroll_lengths=torch.tensor(lengths, dtype=torch.int32),
        sample_rate=16000,
    )


@pytest.mark.unit
def test_real_tfmap_640_sample_enrollment_is_invariant_to_16000_capacity():
    torch.manual_seed(29)
    model = _model()
    mixture, enroll = torch.randn(1, 1024), torch.randn(1, 640)
    with torch.no_grad():
        short = model(_inputs(model, mixture, enroll, [640]))["target_estimate_spectra"]
        padded = model(_inputs(model, mixture, F.pad(enroll, (0, 15360)), [640]))["target_estimate_spectra"]
    relative_l2 = (short - padded).norm() / short.norm()
    print(f"640->16000 relative L2: {relative_l2.item():.9g}")
    torch.testing.assert_close(short, padded, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("cues", [("listen",), ("usef",), ("tfmap",), ("listen", "usef", "tfmap")])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.unit
def test_mixed_enrollment_lengths_match_individual_forward_and_backward(cues, causal):
    torch.manual_seed(37)
    model = _model(cues, causal)
    reference = _model(cues, causal)
    reference.load_state_dict(model.state_dict(), strict=True)
    before = {key: (value.shape, value.dtype) for key, value in model.state_dict().items()}
    lengths = [640, 769, 640, 16000]
    mixture, enroll = torch.randn(4, 1024) * 0.1, torch.randn(4, 16000) * 0.1
    actual = model(_inputs(model, mixture, enroll, lengths))["target_estimate_spectra"]
    expected = torch.cat([reference(_inputs(reference, mixture[i:i+1], enroll[i:i+1, :length], [length]))["target_estimate_spectra"]
                          for i, length in enumerate(lengths)])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for parameter, ref in zip(model.parameters(), reference.parameters()):
        assert (parameter.grad is None) == (ref.grad is None)
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, ref.grad, rtol=1e-4, atol=2e-6)
    assert before == {key: (value.shape, value.dtype) for key, value in model.state_dict().items()}


@pytest.mark.parametrize("length", [1, 256, 257, 640, 641])
@pytest.mark.unit
def test_enrollment_spectrum_reflect_boundary_matches_independent_stft(length):
    model = _model()
    enroll = torch.randn(1, 16000)
    data = _inputs(model, torch.randn(1, 1024), enroll, [length])
    valid = F.pad(enroll[:, :length], (0, max(0, 257-length)))
    expected = StftIstftModule(n_fft=512, hop_length=128, win_length=512)(valid)
    assert data["enroll_spectra_lengths"].dtype == torch.int32
    assert data["enroll_spectra_lengths"].tolist() == [expected.shape[-2]]
    torch.testing.assert_close(data["enroll_spectra"], expected, rtol=0, atol=0)


@pytest.mark.unit
def test_tfmap_embedding_without_context_consumes_true_waveform_lengths():
    torch.manual_seed(91)
    model = TSE_BSRNN_SPK(
        feature_dim=8, num_repeat=1, n_fft=512, hop_length=128, win_length=512,
        features={"tfmap": {"enabled": True, "type": "embedding"}},
        speaker_model={"fbank": {"frame_length": 32, "frame_shift": 8}},
    ).eval()
    assert {"enroll_wav", "enroll_lengths"} <= model.required_tse_input_fields()
    mixture, enroll = torch.randn(3, 1024) * 0.1, torch.randn(3, 16000) * 0.1
    lengths = [1, 769, 640]
    actual = model(_inputs(model, mixture, enroll, lengths))["target_estimate_spectra"]
    with torch.no_grad():
        expected = torch.cat([model(_inputs(model, mixture[i:i+1], enroll[i:i+1, :length], [length]))["target_estimate_spectra"]
                              for i, length in enumerate(lengths)])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    actual.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("bad", [None, torch.tensor([0], dtype=torch.int32), torch.tensor([999], dtype=torch.int32), torch.tensor([6.0])])
@pytest.mark.unit
def test_direct_spectrum_requires_valid_frame_lengths(bad):
    model = _model()
    data = _inputs(model, torch.randn(1, 1024), torch.randn(1, 640), [640])
    if bad is None:
        del data["enroll_spectra_lengths"]
    else:
        data["enroll_spectra_lengths"] = bad
    with pytest.raises((ValueError, TypeError), match="enroll_spectra_lengths"):
        model(data)


def _gloo_length_groups_worker(rank, rendezvous):
    from datetime import timedelta
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(109)
        model = torch.nn.parallel.DistributedDataParallel(_model(("listen", "usef", "tfmap")))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        calls = []
        handle = model.register_forward_pre_hook(lambda *args: calls.append(1))
        for _ in range(2):
            optimizer.zero_grad()
            lengths = [640, 640] if rank == 0 else [640, 769]
            inputs = _inputs(model, torch.randn(2, 1024) * 0.1, torch.randn(2, 1024) * 0.1, lengths)
            result = model(inputs)["target_estimate_spectra"]
            result.square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
        handle.remove()
        assert len(calls) == 2
        weights = torch.cat([p.detach().flatten() for p in model.parameters()])
        other = weights.clone()
        dist.broadcast(other, src=0)
        torch.testing.assert_close(weights, other, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.ddp
def test_two_gloo_ranks_with_different_length_group_counts(tmp_path):
    torch.multiprocessing.spawn(_gloo_length_groups_worker,
                                args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


@pytest.mark.parametrize("length", [1, 256, 257, 640, 641])
@pytest.mark.unit
def test_real_trainer_and_pcm_extractor_agree_with_padded_cue_capacity(length):
    from tasks.tse.cue_contract import TSECue, TSECueSpec, TSECueSpecs, TSECues
    from tasks.tse.data.collate import make_tse_sample, make_tse_target, tse_collate_fn
    from tasks.tse.inference.extractor import TSEExtractor
    from tasks.tse.trainer.tse_trainer import TSETrainer
    torch.manual_seed(111)
    model = _model(("listen", "usef", "tfmap"))
    stft = StftIstftModule(n_fft=512, hop_length=128, win_length=512)
    mixture, enroll = torch.randn(1024) * 0.1, torch.randn(length) * 0.1
    sample = make_tse_sample(
        key="sample", mixture=mixture, sr=16000, length=1024,
        targets=[make_tse_target(target=mixture, speaker_id=None,
                 cues=TSECues(speaker_waveform=TSECue("waveform", enroll, True, length, None)))],
    )
    batch = tse_collate_fn([sample], cue_specs=TSECueSpecs(speaker_waveform=TSECueSpec("waveform", (16000,))))
    trainer = object.__new__(TSETrainer)
    trainer.model, trainer.stft_module, trainer.sample_rate = model, stft, 16000
    extractor = object.__new__(TSEExtractor)
    extractor.device, extractor.model, extractor.stft = torch.device("cpu"), model, stft
    extractor.sample_rate, extractor.apply_vad, extractor.output_peak = 16000, False, None
    extractor._uses_preextracted_spk_embedding = False
    with torch.no_grad():
        train_output = trainer._forward_tse_batch(batch)["target_estimate"]
        inference = extractor.extract_speech_from_pcm(mixture.numpy(), 16000, enroll.numpy(), 16000)
    torch.testing.assert_close(train_output[0], torch.from_numpy(inference), rtol=1e-5, atol=2e-6)


@pytest.mark.unit
def test_multi_target_analyzes_observation_once_and_batches_equal_cue_lengths():
    class CountingSTFT(StftIstftModule):
        def __init__(self):
            super().__init__(n_fft=512, hop_length=128, win_length=512)
            self.shapes = []

        def forward(self, waveform):
            self.shapes.append(tuple(waveform.shape))
            return super().forward(waveform)

    model = _model()
    stft = CountingSTFT()
    mixture, enroll = torch.randn(2, 1024) * 0.1, torch.randn(4, 16000) * 0.1
    lengths = [640, 769, 640, 16000]
    data = build_tse_model_inputs(model=model, stft=stft, mixture=mixture, enroll=enroll,
                                 enroll_lengths=torch.tensor(lengths, dtype=torch.int32),
                                 num_targets=torch.tensor([2, 2], dtype=torch.int32), sample_rate=16000)
    assert stft.shapes == [(2, 1024), (2, 640), (1, 769), (1, 16000)]
    batches = []
    handle = model.sep_model.separator.register_forward_pre_hook(lambda _, inputs: batches.append(inputs[0].shape[0]))
    with torch.no_grad():
        result = model(data)["target_estimate_spectra"]
    handle.remove()
    assert batches == [2, 1, 1]
    with torch.no_grad():
        expected = torch.cat([model(_inputs(model, mixture[i//2:i//2+1], enroll[i:i+1, :length], [length]))["target_estimate_spectra"]
                              for i, length in enumerate(lengths)])
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=2e-6)
