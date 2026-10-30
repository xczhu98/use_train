"""TSE 句级 speaker 前端的长度分桶单行组 BatchNorm 统计语义。

覆盖 ``tasks/tse/models/speaker/encoder.py::sentence_batch_norm_modules`` /
``sentence_batch_statistics_off`` 与
``tasks/tse/models/modules/speaker/spk_frontend.py::_encode_sentence_bucket``：
train 模式下按 enrollment 真实长度分桶后出现的单行组改用句级 BatchNorm running
statistics（逐行不变、不再触发 ``Expected more than 1 value per channel``），多行
同长度分组仍用 batch statistics 并更新 running stats；eval / 冻结 encoder / 预提取
embedding / 帧级路径零行为变化。全部使用真实 ECAPA-TDNN 与真实分桶逻辑。
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import torch

from tasks.tse.models.discriminative.tse_bsrnn_spk import TSE_BSRNN_SPK
from tasks.tse.models.modules.speaker.spk_frontend import (
    ContextFeature,
    SpeakerEmbFeature,
)
from tasks.tse.models.speaker.encoder import (
    SpeakerEncoder,
    sentence_batch_norm_modules,
    sentence_batch_norm_requires_two_rows,
    sentence_batch_statistics_off,
)

N_FFT = 512
HOP = 128
FREQ = N_FFT // 2 + 1
T_MIX = 8
_ECAPA = "ECAPA_TDNN_GLOB_c512"
SHORT = 3200
LONG = 4800


def _trainable_ecapa(**kwargs) -> SpeakerEncoder:
    """未冻结的真实 ECAPA（train 模式下句级 BN 要求 >= 2 行）。"""
    return SpeakerEncoder(model=_ECAPA, freeze=False, **kwargs)


def _spkemb(encoder: SpeakerEncoder, mix_dim: int = 16) -> SpeakerEmbFeature:
    return SpeakerEmbFeature(
        {"mix_dim": mix_dim, "fusion": "multiply"},
        encoder,
        embed_dim=encoder.embed_dim,
    )


def _lengths(*values: int) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.int32)


def _wav(rows: int, samples: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, samples, generator=generator) * 0.01


def _training_flags(module: torch.nn.Module) -> dict[str, bool]:
    return {name: child.training for name, child in module.named_modules()}


def _finite_grads(module: torch.nn.Module) -> list[torch.Tensor]:
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads, "no parameter received gradient"
    assert all(torch.isfinite(grad).all() for grad in grads)
    return grads


@pytest.mark.unit
class TestSentenceBatchNormLocation:
    """句级 BN 的架构事实（登记路径）与运行期定位。"""

    def test_ecapa_locates_only_the_sentence_head_bn(self):
        encoder = _trainable_ecapa()
        assert sentence_batch_norm_modules(encoder) == (encoder.spk_model.bn,)
        assert sentence_batch_norm_requires_two_rows(encoder) is True

    def test_ecapa_emb_bn_adds_a_second_sentence_bn(self):
        encoder = _trainable_ecapa(spk_args={"emb_bn": True})
        assert sentence_batch_norm_modules(encoder) == (
            encoder.spk_model.bn,
            encoder.spk_model.bn2,
        )

    def test_resnet34_sentence_bn_requires_two_emb_layer(self):
        single = SpeakerEncoder(model="ResNet34", freeze=False)
        assert sentence_batch_norm_modules(single) == ()
        assert sentence_batch_norm_requires_two_rows(single) is False

        double = SpeakerEncoder(
            model="ResNet34", freeze=False, spk_args={"two_emb_layer": True}
        )
        assert sentence_batch_norm_modules(double) == (double.spk_model.seg_bn_1,)

    def test_unknown_encoder_has_no_sentence_bn(self):
        class _CustomEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(4)

            def forward(self, wav):
                return wav.mean(dim=1, keepdim=True).repeat(1, 4)

        encoder = _CustomEncoder()
        assert sentence_batch_norm_modules(encoder) == ()
        assert sentence_batch_norm_requires_two_rows(encoder) is False


@pytest.mark.unit
class TestSentenceBatchStatisticsOff:
    """上下文管理器只切换句级 2-D BatchNorm，并可靠恢复。"""

    def test_flips_only_sentence_bn_and_restores(self):
        encoder = _trainable_ecapa().train()
        before = _training_flags(encoder)

        with sentence_batch_statistics_off(encoder):
            inside = _training_flags(encoder)

        assert _training_flags(encoder) == before
        assert {
            name for name, mode in inside.items() if mode != before[name]
        } == {"spk_model.bn"}
        assert inside["spk_model.bn"] is False
        assert inside["spk_model.layer1.bn"] is True
        assert inside["fbank"] is True

    def test_restores_sentence_bn_mode_on_exception(self):
        encoder = _trainable_ecapa().train()

        with pytest.raises(RuntimeError, match="boom"):
            with sentence_batch_statistics_off(encoder):
                assert encoder.spk_model.bn.training is False
                raise RuntimeError("boom")

        assert encoder.spk_model.bn.training is True

    def test_eval_encoder_stays_in_eval(self):
        encoder = _trainable_ecapa().eval()

        with sentence_batch_statistics_off(encoder):
            assert encoder.spk_model.bn.training is False

        assert encoder.spk_model.bn.training is False

    def test_encoder_without_sentence_bn_is_a_no_op(self):
        encoder = SpeakerEncoder(model="ResNet34", freeze=False).train()
        before = _training_flags(encoder)

        with sentence_batch_statistics_off(encoder) as modules:
            assert modules == ()
            assert _training_flags(encoder) == before

        assert _training_flags(encoder) == before


@pytest.mark.unit
class TestSingleRowLengthBucketTraining:
    """train 模式下长度分桶单行组的数值与统计语义（真实 ECAPA）。"""

    def test_heterogeneous_lengths_forward_backward_are_finite(self):
        encoder = _trainable_ecapa().train()
        feature = _spkemb(encoder)

        embedding = feature.compute(
            _wav(2, LONG, seed=0), enroll_lengths=_lengths(SHORT, LONG)
        )

        assert tuple(embedding.shape) == (2, encoder.embed_dim)
        assert torch.isfinite(embedding).all()
        embedding.square().mean().backward()
        _finite_grads(encoder)
        assert encoder.spk_model.bn.weight.grad.abs().sum() > 0

    def test_all_distinct_lengths_forward_backward_are_finite(self):
        encoder = _trainable_ecapa().train()
        feature = _spkemb(encoder)

        embedding = feature.compute(
            _wav(3, LONG, seed=1), enroll_lengths=_lengths(SHORT, 4000, LONG)
        )

        assert tuple(embedding.shape) == (3, encoder.embed_dim)
        embedding.square().mean().backward()
        _finite_grads(encoder)

    def test_single_row_embedding_is_unchanged_by_heterogeneous_batch(self):
        encoder = _trainable_ecapa().train()
        feature = _spkemb(encoder)
        wav = _wav(2, LONG, seed=2)

        batched = feature.compute(wav, enroll_lengths=_lengths(SHORT, LONG))
        standalone = feature.compute(wav[1:2, :LONG], enroll_lengths=_lengths(LONG))

        max_abs_diff = (batched[1] - standalone[0]).abs().max().item()
        assert max_abs_diff == 0.0
        torch.testing.assert_close(batched[1], standalone[0], rtol=0, atol=0)

    def test_single_row_group_does_not_update_running_stats(self):
        encoder = _trainable_ecapa().train()
        feature = _spkemb(encoder)
        sentence_bn = encoder.spk_model.bn
        before_mean = sentence_bn.running_mean.clone()
        before_tracked = sentence_bn.num_batches_tracked.clone()

        feature.compute(_wav(1, SHORT, seed=3), enroll_lengths=_lengths(SHORT))

        assert torch.equal(sentence_bn.running_mean, before_mean)
        assert torch.equal(sentence_bn.num_batches_tracked, before_tracked)

    def test_multi_row_group_uses_batch_statistics(self):
        encoder = _trainable_ecapa().train()
        feature = _spkemb(encoder)
        sentence_bn = encoder.spk_model.bn
        before_mean = sentence_bn.running_mean.clone()

        feature.compute(_wav(2, SHORT, seed=4), enroll_lengths=_lengths(SHORT, SHORT))

        assert not torch.equal(sentence_bn.running_mean, before_mean)
        assert int(sentence_bn.num_batches_tracked) == 1

    def test_mixed_batch_updates_only_the_multi_row_group(self):
        encoder = _trainable_ecapa().train()
        feature = _spkemb(encoder)
        sentence_bn = encoder.spk_model.bn

        feature.compute(
            _wav(3, LONG, seed=5), enroll_lengths=_lengths(SHORT, SHORT, LONG)
        )

        assert int(sentence_bn.num_batches_tracked) == 1

    def test_conv_bn_and_other_modules_keep_train_behavior(self):
        encoder = _trainable_ecapa().train()
        feature = _spkemb(encoder)
        conv_bn = encoder.spk_model.layer1.bn
        sentence_bn = encoder.spk_model.bn
        observed: dict[str, bool] = {}
        handle = conv_bn.register_forward_pre_hook(
            lambda module, _args: observed.update(
                conv=module.training, sentence=sentence_bn.training
            )
        )
        try:
            feature.compute(_wav(1, SHORT, seed=6), enroll_lengths=_lengths(SHORT))
        finally:
            handle.remove()

        assert observed == {"conv": True, "sentence": False}
        assert int(conv_bn.num_batches_tracked) == 1
        assert int(sentence_bn.num_batches_tracked) == 0


@pytest.mark.unit
class TestUnchangedPaths:
    """eval / 冻结 encoder / 帧级路径 / 模型级 joint 训练路径。"""

    def test_eval_mode_leaves_modules_and_running_stats_untouched(self):
        encoder = _trainable_ecapa().eval()
        feature = _spkemb(encoder)
        before = _training_flags(encoder)
        sentence_bn = encoder.spk_model.bn
        before_mean = sentence_bn.running_mean.clone()
        before_tracked = sentence_bn.num_batches_tracked.clone()

        with torch.no_grad():
            embedding = feature.compute(
                _wav(2, LONG, seed=7), enroll_lengths=_lengths(SHORT, LONG)
            )

        assert tuple(embedding.shape) == (2, encoder.embed_dim)
        assert _training_flags(encoder) == before
        assert torch.equal(sentence_bn.running_mean, before_mean)
        assert torch.equal(sentence_bn.num_batches_tracked, before_tracked)

    def test_frozen_encoder_in_train_mode_keeps_eval_sentence_bn(self):
        encoder = SpeakerEncoder(model=_ECAPA, freeze=True).train()
        assert encoder.spk_model.bn.training is False
        feature = _spkemb(encoder)
        before_tracked = encoder.spk_model.bn.num_batches_tracked.clone()

        with torch.no_grad():
            embedding = feature.compute(
                _wav(2, LONG, seed=8), enroll_lengths=_lengths(SHORT, LONG)
            )

        assert tuple(embedding.shape) == (2, encoder.embed_dim)
        assert encoder.spk_model.bn.training is False
        assert torch.equal(encoder.spk_model.bn.num_batches_tracked, before_tracked)

    def test_frame_level_context_path_never_touches_the_sentence_bn(self):
        torch.manual_seed(9)
        encoder = _trainable_ecapa().train()
        context = ContextFeature(
            {
                "embed_dim": encoder.frame_dim,
                "atten_dim": 16,
                "mix_dim": 8,
                "num_heads": 2,
                "band": 2,
                "fusion": "multiply",
            },
            encoder,
        )
        before = _training_flags(encoder)
        observed: dict[str, bool] = {}
        handle = encoder.spk_model.bn.register_forward_pre_hook(
            lambda module, _args: observed.update(sentence=module.training)
        )
        try:
            fused = context.compute_and_post(
                torch.randn(2, 2, 8, T_MIX),
                _wav(2, LONG, seed=10),
                _lengths(SHORT, LONG),
            )
        finally:
            handle.remove()

        assert tuple(fused.shape) == (2, 2, 8, T_MIX)
        assert torch.isfinite(fused).all()
        assert _training_flags(encoder) == before
        assert observed == {}

    def test_joint_model_forward_backward_with_heterogeneous_lengths(self):
        torch.manual_seed(11)
        model = TSE_BSRNN_SPK(
            sr=16000,
            feature_dim=32,
            num_repeat=2,
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=N_FFT,
            window="hann",
            center=True,
            joint_training=True,
            features={"spkemb": {"enabled": True}},
        )
        model.train()
        generator = torch.Generator().manual_seed(12)
        batch = {
            "mix_spectra": torch.randn(2, FREQ, T_MIX, 2, generator=generator),
            "enroll_wav": torch.randn(2, LONG, generator=generator) * 0.01,
            "enroll_lengths": _lengths(SHORT, LONG),
        }

        estimate = model(batch)["target_estimate_spectra"]

        assert tuple(estimate.shape) == (2, FREQ, T_MIX, 2)
        assert torch.isfinite(estimate).all()
        estimate.square().mean().backward()
        _finite_grads(model)
        assert model.spk_ft.encoder.spk_model.bn.weight.grad.abs().sum() > 0

    def test_preextracted_embedding_model_has_no_sentence_bn(self):
        torch.manual_seed(15)
        model = TSE_BSRNN_SPK(
            sr=16000,
            feature_dim=32,
            num_repeat=2,
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=N_FFT,
            window="hann",
            center=True,
            joint_training=False,
            features={"spkemb": {"enabled": True}},
        )
        assert not hasattr(model.spk_ft, "encoder")
        assert sentence_batch_norm_modules(model.spk_ft) == ()
        model.train()
        generator = torch.Generator().manual_seed(16)
        batch = {
            "mix_spectra": torch.randn(2, FREQ, T_MIX, 2, generator=generator),
            "spk_embedding": torch.randn(2, 192, generator=generator),
        }

        estimate = model(batch)["target_estimate_spectra"]

        assert tuple(estimate.shape) == (2, FREQ, T_MIX, 2)
        assert torch.isfinite(estimate).all()


def _gloo_sentence_bucket_worker(rank: int, rendezvous: str) -> None:
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(13)
        model = torch.nn.parallel.DistributedDataParallel(
            TSE_BSRNN_SPK(
                sr=16000,
                feature_dim=32,
                num_repeat=2,
                n_fft=N_FFT,
                hop_length=HOP,
                win_length=N_FFT,
                window="hann",
                center=True,
                joint_training=True,
                features={"spkemb": {"enabled": True}},
            )
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        lengths = (SHORT, LONG) if rank == 0 else (LONG, SHORT)
        for _ in range(2):
            optimizer.zero_grad()
            generator = torch.Generator().manual_seed(14)
            batch = {
                "mix_spectra": torch.randn(2, FREQ, T_MIX, 2, generator=generator),
                "enroll_wav": torch.randn(2, LONG, generator=generator) * 0.01,
                "enroll_lengths": _lengths(*lengths),
            }
            estimate = model(batch)["target_estimate_spectra"]
            estimate.square().mean().backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            optimizer.step()

        weights = torch.cat([p.detach().flatten() for p in model.parameters()])
        other = weights.clone()
        dist.broadcast(other, src=0)
        torch.testing.assert_close(weights, other, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.ddp
def test_two_gloo_ranks_train_on_single_row_length_buckets(tmp_path):
    """每个 rank 的 enrollment 长度互不相同（全为单行组）时 DDP 训练不崩。"""
    torch.multiprocessing.spawn(
        _gloo_sentence_bucket_worker,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=2,
        join=True,
    )
