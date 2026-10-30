"""Tests for tasks/tse/models/discriminative/tse_bsrnn_spk.py (TSE_BSRNN_SPK).

覆盖 dict 谱进谱出接口的 forward 形状（joint / 预提取双模式、各 speaker cue）、
以及构造期未知 kwargs / 非法配置的 fail-fast 行为。全部小尺寸 CPU 可跑。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tasks.tse.models.discriminative.tse_bsrnn_spk import TSE_BSRNN_SPK
from tasks.tse.models.modules.speaker.spk_frontend import (
    ContextFeature,
    ListenFeature,
    SpeakerEmbFeature,
    TFMapFeature,
    UsefFeature,
)
from tasks.tse.models.modules.speaker.usef import (
    AllHeadPReLULayerNormalization4DCF,
    LayerNormalization4DCF,
    USEF_attentionblock,
)
from tasks.tse.models.speaker.encoder import SpeakerEncoder, _can_cast_bn_counter

N_FFT = 512
HOP = 128
FREQ = N_FFT // 2 + 1  # 257
T_MIX = 8
T_ENROLL = 6
ENROLL_WAV_LEN = 16000  # 1s @16k


class _DummyMeta:
    """不可被 weights_only=True 安全反序列化的占位对象（测试 fail-closed）。"""


def _small_kwargs(**overrides) -> dict:
    """小尺寸模型构造参数（模拟框架经 model.args 注入的 stft 参数）。"""
    kw = dict(
        sr=16000,
        feature_dim=32,
        num_repeat=2,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window="hann",
        center=True,
    )
    kw.update(overrides)
    return kw


def _make_batch(with_enroll_spec: bool = True, with_enroll_wav: bool = True) -> dict:
    g = torch.Generator().manual_seed(0)
    batch = {
        "mix_spectra": torch.randn(1, FREQ, T_MIX, 2, generator=g),
    }
    if with_enroll_spec:
        batch["enroll_spectra"] = torch.randn(1, FREQ, T_ENROLL, 2, generator=g)
        batch["enroll_spectra_lengths"] = torch.tensor([T_ENROLL], dtype=torch.int32)
    if with_enroll_wav:
        batch["enroll_wav"] = torch.randn(1, ENROLL_WAV_LEN, generator=g) * 0.01
        batch["enroll_lengths"] = torch.tensor(
            [ENROLL_WAV_LEN], dtype=torch.int32
        )
    return batch


def _check_out(out: dict):
    assert "target_estimate_spectra" in out
    est = out["target_estimate_spectra"]
    assert est.shape == (1, FREQ, T_MIX, 2)
    assert torch.isfinite(est).all()


class TestSpeakerFrontendRegressions:
    class _PaddingSensitiveEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fbank = type("_Fbank", (), {"window_size": 4})()

        def forward(self, wav):
            return wav.mean(dim=1, keepdim=True).repeat(1, 4)

        def forward_frame_level(self, wav):
            return wav.unsqueeze(1).repeat(1, 4, 1)

    def test_spkemb_uses_true_lengths_instead_of_batch_padding(self):
        encoder = self._PaddingSensitiveEncoder()
        feature = SpeakerEmbFeature(
            {"mix_dim": 4, "fusion": "multiply"}, encoder, embed_dim=4
        )
        valid = torch.linspace(-1.0, 1.0, 6)
        standalone = feature.compute(
            valid.unsqueeze(0), enroll_lengths=torch.tensor([6])
        )
        padded = torch.full((2, 10), 99.0)
        padded[0] = torch.arange(10, dtype=torch.float32)
        padded[1, :6] = valid
        batched = feature.compute(
            padded, enroll_lengths=torch.tensor([10, 6])
        )
        torch.testing.assert_close(batched[1], standalone[0])

    def test_context_fusion_uses_true_lengths_instead_of_padded_frames(self):
        torch.manual_seed(3)
        encoder = self._PaddingSensitiveEncoder()
        feature = ContextFeature(
            {
                "embed_dim": 4,
                "atten_dim": 4,
                "mix_dim": 4,
                "num_heads": 1,
                "band": 1,
                "fusion": "multiply",
            },
            encoder,
        ).eval()
        valid = torch.linspace(-0.5, 0.5, 6)
        standalone_mix = torch.randn(1, 1, 4, 3)
        standalone = feature.compute_and_post(
            standalone_mix, valid.unsqueeze(0), torch.tensor([6])
        )
        padded = torch.full((2, 10), 99.0)
        padded[0] = torch.arange(10, dtype=torch.float32)
        padded[1, :6] = valid
        batch_mix = torch.cat((torch.randn(1, 1, 4, 3), standalone_mix), dim=0)
        batched = feature.compute_and_post(
            batch_mix, padded, torch.tensor([10, 6])
        )
        torch.testing.assert_close(batched[1], standalone[0])

    def test_tfmap_zero_norm_stays_finite(self):
        """全零 enroll/mix 不应在 TFMap 归一化处产生 NaN。"""
        feature = TFMapFeature({})
        out = feature.compute(torch.zeros(2, 5, 3), torch.zeros(2, 5, 4))
        assert torch.equal(out, torch.zeros_like(out))
        assert torch.isfinite(out).all()

    def test_listen_offset_is_explicit_per_call(self):
        """交错调用不共享可变 offset，每个结果使用自身返回值裁剪。"""
        feature = ListenFeature({"glue": 2, "hop": 1})
        first, first_offset = feature.compute(
            torch.full((1, 2), -1.0), torch.tensor([[10.0, 11.0]])
        )
        _, second_offset = feature.compute(
            torch.full((1, 5), -2.0), torch.tensor([[20.0]])
        )

        assert first_offset == 4
        assert second_offset == 7
        assert torch.equal(
            feature.post(first, first_offset), torch.tensor([[10.0, 11.0]])
        )
        assert not hasattr(feature, "offset")

    def test_causal_usef_time_conv_has_no_future_context(self):
        """未来帧变化不得影响 causal USEF 时间卷积的历史输出。"""
        feature = UsefFeature({
            "causal": True,
            "spec_dim": 2,
            "emb_dim": 4,
            "enc_dim": 5,
            "n_head": 2,
            "approx_qk_dim": 16,
            "t_ksize": 3,
        }).eval()
        g = torch.Generator().manual_seed(7)
        enroll = torch.randn(1, 2, 5, 4, generator=g)
        mix = torch.randn(1, 2, 5, 6, generator=g)
        changed = mix.clone()
        changed[..., 3:] += 100.0

        with torch.no_grad():
            _, encoded = feature.compute(enroll, mix)
            _, changed_encoded = feature.compute(enroll, changed)
        assert encoded.shape[-1] == mix.shape[-1]
        assert torch.equal(encoded[..., :3], changed_encoded[..., :3])

    @pytest.mark.parametrize(
        ("factory", "match"),
        [
            (lambda: USEF_attentionblock(4, 5, 2, 16, activation="relu"), "activation"),
            (lambda: USEF_attentionblock(5, 5, 2, 16), "divisible"),
            (lambda: LayerNormalization4DCF((2, 3, 4)), "channels, frequencies"),
            (
                lambda: AllHeadPReLULayerNormalization4DCF((2, 3)),
                "heads, channels, frequencies",
            ),
        ],
    )
    def test_usef_configuration_checks_are_explicit(self, factory, match):
        with pytest.raises(ValueError, match=match):
            factory()

    def test_usef_input_rank_check_is_explicit(self):
        layer = AllHeadPReLULayerNormalization4DCF((2, 2, 3))
        with pytest.raises(ValueError, match="4 dimensions"):
            layer(torch.zeros(1, 4, 3))


class TestForwardShapes:
    def test_joint_spkemb(self):
        """joint_training=True + spkemb cue：enroll_wav 内部提 embedding。"""
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=True,
                features={"spkemb": {"enabled": True}},
            )
        )
        model.eval()
        with torch.no_grad():
            _check_out(model(_make_batch()))

    def test_preextracted_spkemb(self):
        """joint_training=False + spkemb cue：用预提取 spk_embedding，不构造 encoder。"""
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=False,
                features={"spkemb": {"enabled": True}},
            )
        )
        assert not hasattr(model.spk_ft, "encoder")
        model.eval()
        batch = _make_batch(with_enroll_spec=False, with_enroll_wav=False)
        batch["spk_embedding"] = torch.randn(1, 192, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            _check_out(model(batch))

    def test_listen_cue(self):
        """listen cue：谱域前缀拼接，输出裁掉前缀后保持 [B, F, T, 2]。"""
        model = TSE_BSRNN_SPK(
            **_small_kwargs(features={"listen": {"enabled": True}})
        )
        model.eval()
        with torch.no_grad():
            _check_out(model(_make_batch()))

    def test_tfmap_cue(self):
        model = TSE_BSRNN_SPK(
            **_small_kwargs(features={"tfmap": {"enabled": True}})
        )
        model.eval()
        with torch.no_grad():
            _check_out(model(_make_batch()))

    def test_usef_cue(self):
        model = TSE_BSRNN_SPK(
            **_small_kwargs(features={"usef": {"enabled": True}})
        )
        model.eval()
        with torch.no_grad():
            _check_out(model(_make_batch()))

    def test_context_cue_joint(self):
        """context cue（帧级 embedding cross-attention），仅 joint 模式。"""
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=True,
                features={"context": {"enabled": True}},
            )
        )
        model.eval()
        with torch.no_grad():
            _check_out(model(_make_batch()))

    def test_all_cues_combined(self):
        """5 cue 全开（listen/usef/tfmap/context/spkemb）联合前向。"""
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=True,
                features={
                    "listen": {"enabled": True},
                    "usef": {"enabled": True},
                    "tfmap": {"enabled": True},
                    "context": {"enabled": True},
                    "spkemb": {"enabled": True},
                },
            )
        )
        model.eval()
        with torch.no_grad():
            _check_out(model(_make_batch()))

    def test_no_cue_passthrough(self):
        """全部 cue 关闭：纯 BSRNN 分离通路。"""
        model = TSE_BSRNN_SPK(**_small_kwargs())
        model.eval()
        with torch.no_grad():
            _check_out(model(_make_batch(with_enroll_spec=False, with_enroll_wav=False)))


class TestStatefulStreaming:
    """causal BSRNN 分块一致性与 enrollment 会话状态。"""

    @staticmethod
    def _model() -> TSE_BSRNN_SPK:
        return TSE_BSRNN_SPK(
            **_small_kwargs(
                feature_dim=16,
                num_repeat=1,
                causal=True,
                joint_training=False,
                features={"spkemb": {"enabled": True}},
            )
        ).eval()

    @staticmethod
    def _inputs():
        generator = torch.Generator().manual_seed(2026)
        mix = torch.randn(1, FREQ, 7, 2, generator=generator)
        embedding = torch.randn(1, 192, generator=generator)
        return mix, embedding

    def test_chunks_match_offline_and_carry_enrollment(self):
        model = self._model()
        mix, embedding = self._inputs()
        with torch.no_grad():
            offline = model(
                {"mix_spectra": mix, "spk_embedding": embedding}
            )["target_estimate_spectra"]
            state = model.init_streaming_state({"spk_embedding": embedding})
            first, state = model.forward_stream_chunk(
                {"mix_spectra": mix[:, :, :3]}, state
            )
            second, state = model.forward_stream_chunk(
                {"mix_spectra": mix[:, :, 3:]}, state
            )
        streamed = torch.cat(
            (first["target_estimate_spectra"], second["target_estimate_spectra"]),
            dim=2,
        )
        torch.testing.assert_close(streamed, offline, rtol=1e-5, atol=2e-6)
        assert state.enrollment["spkemb"].shape == (1, 192)

    def test_reset_clears_history_and_rebinds_enrollment(self):
        model = self._model()
        mix, embedding = self._inputs()
        replacement = -embedding
        with torch.no_grad():
            dirty = model.init_streaming_state({"spk_embedding": embedding})
            _, dirty = model.forward_stream_chunk({"mix_spectra": mix[:, :, :2]}, dirty)
            reset = model.reset_streaming_state(
                dirty, {"spk_embedding": replacement}
            )
            reset_out, _ = model.forward_stream_chunk({"mix_spectra": mix}, reset)
            fresh = model.init_streaming_state({"spk_embedding": replacement})
            fresh_out, _ = model.forward_stream_chunk({"mix_spectra": mix}, fresh)
        torch.testing.assert_close(
            reset_out["target_estimate_spectra"],
            fresh_out["target_estimate_spectra"],
            rtol=0,
            atol=0,
        )

    def test_chunk_rejects_enrollment_replacement_without_reset(self):
        model = self._model()
        mix, embedding = self._inputs()
        state = model.init_streaming_state({"spk_embedding": embedding})
        with pytest.raises(ValueError, match="reset_streaming_state"):
            model.forward_stream_chunk(
                {
                    "mix_spectra": mix[:, :, :1],
                    "spk_embedding": -embedding,
                },
                state,
            )

    def test_tfmap_context_chunks_match_offline(self):
        """官方 tfmap+context causal cue 也复用固定 enrollment 条件。"""
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                feature_dim=16,
                num_repeat=1,
                causal=True,
                joint_training=True,
                features={
                    "tfmap": {"enabled": True},
                    "context": {
                        "enabled": True,
                        "atten_dim": 16,
                        "mix_dim": 16,
                    },
                },
            )
        ).eval()
        generator = torch.Generator().manual_seed(2027)
        mix = torch.randn(1, FREQ, 5, 2, generator=generator)
        enroll_spec = torch.randn(1, FREQ, 4, 2, generator=generator)
        enroll_wav = torch.randn(1, ENROLL_WAV_LEN, generator=generator) * 0.01
        with torch.no_grad():
            offline = model(
                {
                    "mix_spectra": mix,
                    "enroll_spectra": enroll_spec,
                    "enroll_spectra_lengths": torch.tensor([4], dtype=torch.int32),
                    "enroll_wav": enroll_wav,
                    "enroll_lengths": torch.tensor([ENROLL_WAV_LEN]),
                }
            )["target_estimate_spectra"]
            state = model.init_streaming_state(
                {"enroll_spectra": enroll_spec, "enroll_wav": enroll_wav}
            )
            first, state = model.forward_stream_chunk(
                {"mix_spectra": mix[:, :, :2]}, state
            )
            second, state = model.forward_stream_chunk(
                {"mix_spectra": mix[:, :, 2:]}, state
            )
        streamed = torch.cat(
            (first["target_estimate_spectra"], second["target_estimate_spectra"]),
            dim=2,
        )
        torch.testing.assert_close(streamed, offline, rtol=1e-5, atol=2e-6)


class TestForwardValidation:
    def test_bad_mix_rank_raises(self):
        model = TSE_BSRNN_SPK(**_small_kwargs())
        with pytest.raises(ValueError, match=r"\[B, F, T, 2\]"):
            model({"mix_spectra": torch.randn(1, FREQ, T_MIX)})

    def test_bad_freq_raises(self):
        model = TSE_BSRNN_SPK(**_small_kwargs())
        with pytest.raises(ValueError, match="不一致"):
            model({"mix_spectra": torch.randn(1, FREQ + 1, T_MIX, 2)})

    def test_runtime_sample_rate_mismatch_raises(self):
        model = TSE_BSRNN_SPK(**_small_kwargs())
        assert model.sample_rate == 16000
        batch = _make_batch(with_enroll_spec=False, with_enroll_wav=False)
        batch["sample_rate"] = 8000
        with pytest.raises(ValueError, match="model.sample_rate=16000"):
            model(batch)

    @pytest.mark.parametrize(
        ("sample_rate", "error", "match"),
        [
            (torch.tensor([16000, 16000]), ValueError, "must be scalar"),
            (16000.0, TypeError, "integer scalar"),
            (True, TypeError, "integer scalar"),
        ],
    )
    def test_runtime_sample_rate_contract_is_explicit(
        self, sample_rate, error, match
    ):
        model = TSE_BSRNN_SPK(**_small_kwargs())
        batch = _make_batch(with_enroll_spec=False, with_enroll_wav=False)
        batch["sample_rate"] = sample_rate
        with pytest.raises(error, match=match):
            model(batch)

    def test_runtime_sample_rate_missing_remains_compatible(self):
        model = TSE_BSRNN_SPK(**_small_kwargs()).eval()
        batch = _make_batch(with_enroll_spec=False, with_enroll_wav=False)
        assert "sample_rate" not in batch
        with torch.no_grad():
            _check_out(model(batch))

    def test_listen_requires_enroll_spectra(self):
        model = TSE_BSRNN_SPK(
            **_small_kwargs(features={"listen": {"enabled": True}})
        )
        with pytest.raises(ValueError, match="enroll_spectra"):
            model(_make_batch(with_enroll_spec=False))

    def test_joint_spkemb_requires_enroll_wav(self):
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=True,
                features={"spkemb": {"enabled": True}},
            )
        )
        with pytest.raises(ValueError, match="enroll_wav"):
            model(_make_batch(with_enroll_wav=False))

    def test_joint_spkemb_requires_enroll_lengths(self):
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=True,
                features={"spkemb": {"enabled": True}},
            )
        )
        batch = _make_batch()
        batch.pop("enroll_lengths")
        with pytest.raises(ValueError, match="enroll_lengths"):
            model(batch)

    def test_instance_io_contract_tracks_enabled_cues(self):
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=True,
                features={
                    "listen": {"enabled": True},
                    "spkemb": {"enabled": True},
                },
                multi_task=True,
                spks_in_train=3,
            )
        )
        assert [field.name for field in model.IO_CONTRACT.inputs] == [
            "mix_spectra",
            "enroll_spectra",
            "enroll_spectra_lengths",
            "enroll_wav",
            "enroll_lengths",
        ]
        assert [field.name for field in model.IO_CONTRACT.outputs] == [
            "target_estimate_spectra",
            "spk_logits",
        ]

    def test_preextracted_requires_spk_embedding(self):
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=False,
                features={"spkemb": {"enabled": True}},
            )
        )
        with pytest.raises(ValueError, match="spk_embedding"):
            model(_make_batch(with_enroll_spec=False, with_enroll_wav=False))

    def test_bad_enroll_spectra_last_dim_raises(self):
        """enroll_spectra 缺末维 2（[B, F, T]）时 fail-loud。"""
        model = TSE_BSRNN_SPK(
            **_small_kwargs(features={"listen": {"enabled": True}})
        )
        batch = _make_batch()
        batch["enroll_spectra"] = torch.randn(1, FREQ, T_ENROLL)
        with pytest.raises(ValueError, match="enroll_spectra 应为"):
            model(batch)

    def test_bad_spk_embedding_shape_raises(self):
        """预提取 spk_embedding 形状畸形（embed_dim 不符）时 fail-loud。"""
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=False,
                features={"spkemb": {"enabled": True}},
            )
        )
        batch = _make_batch(with_enroll_spec=False, with_enroll_wav=False)
        batch["spk_embedding"] = torch.randn(1, 64)
        with pytest.raises(ValueError, match="spk_embedding 应为"):
            model(batch)

    @pytest.mark.parametrize("embedding_dtype", [torch.float16, torch.float64])
    def test_spk_embedding_dtype_must_match_mix(self, embedding_dtype):
        """预提取 embedding dtype 不一致时应在融合层前明确拒绝。"""
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=False,
                features={"spkemb": {"enabled": True}},
            )
        )
        batch = _make_batch(with_enroll_spec=False, with_enroll_wav=False)
        batch["spk_embedding"] = torch.randn(1, 192, dtype=embedding_dtype)

        with pytest.raises(TypeError, match="dtype 必须与 mix_spectra 一致") as exc:
            model(batch)

        message = str(exc.value)
        assert f"embedding={embedding_dtype}" in message
        assert f"mix_spectra={batch['mix_spectra'].dtype}" in message

    def test_spk_embedding_matching_dtype_is_accepted(self):
        """与 mix 一致的 floating dtype 保持原有预提取前向行为。"""
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=False,
                features={"spkemb": {"enabled": True}},
            )
        ).eval()
        batch = _make_batch(with_enroll_spec=False, with_enroll_wav=False)
        batch["spk_embedding"] = torch.randn(
            1,
            192,
            dtype=batch["mix_spectra"].dtype,
            generator=torch.Generator().manual_seed(1),
        )

        with torch.no_grad():
            out = model(batch)

        _check_out(out)
        assert out["target_estimate_spectra"].dtype == batch["mix_spectra"].dtype


class TestConstructionValidation:
    @pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
    def test_joint_training_requires_bool(self, value):
        with pytest.raises(TypeError, match=r"joint_training.*bool"):
            TSE_BSRNN_SPK(**_small_kwargs(joint_training=value))

    @pytest.mark.parametrize("feature", ["listen", "usef", "tfmap", "context", "spkemb"])
    @pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
    def test_feature_enabled_requires_bool(self, feature, value):
        with pytest.raises(
            TypeError,
            match=rf"features\.{feature}\.enabled.*bool",
        ):
            TSE_BSRNN_SPK(
                **_small_kwargs(features={feature: {"enabled": value}})
            )

    def test_unknown_kwargs_raise(self):
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            TSE_BSRNN_SPK(**_small_kwargs(bogus_arg=1))

    def test_unknown_feature_key_raise(self):
        with pytest.raises(KeyError, match="Unknown config key"):
            TSE_BSRNN_SPK(
                **_small_kwargs(features={"spkemb": {"enabled": True, "bogus": 1}})
            )

    def test_win_length_must_equal_n_fft(self):
        with pytest.raises(ValueError, match="win_length == n_fft"):
            TSE_BSRNN_SPK(**_small_kwargs(win_length=256))

    def test_window_must_be_hann(self):
        with pytest.raises(ValueError, match="window == 'hann'"):
            TSE_BSRNN_SPK(**_small_kwargs(window="hamming"))

    def test_center_must_be_true(self):
        with pytest.raises(ValueError, match="center=True"):
            TSE_BSRNN_SPK(**_small_kwargs(center=False))

    def test_context_requires_joint_training(self):
        with pytest.raises(ValueError, match="joint_training=False"):
            TSE_BSRNN_SPK(
                **_small_kwargs(
                    joint_training=False,
                    features={"context": {"enabled": True}},
                )
            )

    def test_tfmap_type_must_be_supported(self):
        with pytest.raises(ValueError, match="'spec' or 'embedding'"):
            TSE_BSRNN_SPK(
                **_small_kwargs(features={"tfmap": {"enabled": True, "type": "wav"}})
            )

    def test_speaker_fbank_sample_rate_must_match_top_level(self):
        with pytest.raises(ValueError, match="fbank.sample_rate=8000.*sr=16000"):
            TSE_BSRNN_SPK(
                **_small_kwargs(
                    features={"spkemb": {"enabled": True}},
                    speaker_model={"fbank": {"sample_rate": 8000}},
                )
            )

    @pytest.mark.parametrize(
        ("causal", "usef_causal"),
        [(True, False), (False, True)],
    )
    def test_usef_causal_must_match_top_level(self, causal, usef_causal):
        with pytest.raises(ValueError, match="features.usef.causal=.*顶层 causal="):
            TSE_BSRNN_SPK(
                **_small_kwargs(
                    causal=causal,
                    features={
                        "usef": {"enabled": True, "causal": usef_causal}
                    },
                )
            )


class TestMultiTaskSpeakerHead:
    """multi_task speaker 分类头：spk_logits 输出 / 构造校验 / 默认行为不变。"""

    def test_multi_task_outputs_spk_logits_joint(self):
        """joint + spkemb + multi_task：forward 返回 dict 追加 spk_logits [B, C]。"""
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=True,
                features={"spkemb": {"enabled": True}},
                multi_task=True,
                spks_in_train=3,
            )
        )
        model.eval()
        with torch.no_grad():
            out = model(_make_batch())
        _check_out(out)
        assert out["spk_logits"].shape == (1, 3)
        assert torch.isfinite(out["spk_logits"]).all()

    def test_multi_task_outputs_spk_logits_preextracted(self):
        """预提取模式：spk_logits 由 batch 的 spk_embedding 经分类头产生。"""
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=False,
                features={"spkemb": {"enabled": True}},
                multi_task=True,
                spks_in_train=5,
            )
        )
        model.eval()
        batch = _make_batch(with_enroll_spec=False, with_enroll_wav=False)
        batch["spk_embedding"] = torch.randn(1, 192, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            out = model(batch)
        assert out["spk_logits"].shape == (1, 5)

    def test_multi_task_logits_backward(self):
        """spk_logits 反传：分类头收到非零梯度（CE 辅助任务可训练）。

        eval 模式下反传（B=1 时 train 模式会触发 encoder 内 BatchNorm 的
        batch>1 限制，与本测试目标无关）。
        """
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=True,
                features={"spkemb": {"enabled": True}},
                multi_task=True,
                spks_in_train=3,
            )
        )
        model.eval()
        g = torch.Generator().manual_seed(0)
        batch = {
            "mix_spectra": torch.randn(2, FREQ, T_MIX, 2, generator=g),
            "enroll_spectra": torch.randn(2, FREQ, T_ENROLL, 2, generator=g),
            "enroll_wav": torch.randn(2, ENROLL_WAV_LEN, generator=g) * 0.01,
            "enroll_lengths": torch.full((2,), ENROLL_WAV_LEN, dtype=torch.int32),
        }
        out = model(batch)
        ce = torch.nn.functional.cross_entropy(
            out["spk_logits"], torch.tensor([1, 2])
        )
        ce.backward()
        assert model.spk_classifier.weight.grad is not None
        assert model.spk_classifier.weight.grad.abs().sum() > 0

    def test_multi_task_disabled_no_logits(self):
        """multi_task=False（默认）：无 spk_classifier、输出 dict 无 spk_logits 键。"""
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=True,
                features={"spkemb": {"enabled": True}},
            )
        )
        assert not hasattr(model, "spk_classifier")
        model.eval()
        with torch.no_grad():
            out = model(_make_batch())
        assert "spk_logits" not in out

    def test_multi_task_requires_spkemb_cue(self):
        with pytest.raises(ValueError, match="spkemb"):
            TSE_BSRNN_SPK(**_small_kwargs(multi_task=True, spks_in_train=3))

    def test_multi_task_requires_spks_in_train(self):
        with pytest.raises(ValueError, match="spks_in_train"):
            TSE_BSRNN_SPK(
                **_small_kwargs(
                    joint_training=True,
                    features={"spkemb": {"enabled": True}},
                    multi_task=True,
                )
            )


class TestLoadPretrained:
    """wesep 比赛 ckpt 兼容加载与安全校验（weights_only + fail-closed）。"""

    def _joint_kwargs(self) -> dict:
        return _small_kwargs(
            joint_training=True,
            features={"context": {"enabled": True}, "spkemb": {"enabled": True}},
        )

    def test_wesep_format_full_match(self, tmp_path):
        """合成 wesep 格式 state_dict 在显式官方兼容模式下全量命中。"""
        model = TSE_BSRNN_SPK(**self._joint_kwargs())
        state = {f"module.{k}": v for k, v in model.state_dict().items()}
        state["loss.weight"] = torch.randn(3)  # 官方训练头附加键
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save(state, ckpt_path)

        loaded = TSE_BSRNN_SPK.load_pretrained(
            str(ckpt_path), strict=False, **self._joint_kwargs()
        )
        loaded_state = loaded.state_dict()
        for k, v in model.state_dict().items():
            assert k in loaded_state
            assert torch.equal(loaded_state[k], v), f"键 {k} 加载后不一致"

    def test_official_training_archive_format(self, tmp_path):
        """官方训练存档格式 {"models": [state_dict], ...}：全量键命中。

        wesep average_model.py 产出的存档（models/optimizers/schedulers/scaler），
        load_pretrained 应取 models[0]（state_dict dict）加载。
        """
        model = TSE_BSRNN_SPK(**self._joint_kwargs())
        ckpt_path = tmp_path / "official.pt"
        torch.save(
            {
                "models": [dict(model.state_dict())],
                "optimizers": [],
                "schedulers": [],
                "scaler": {},
            },
            ckpt_path,
        )

        loaded = TSE_BSRNN_SPK.load_pretrained(
            str(ckpt_path), **self._joint_kwargs()
        )
        # 简化断言：数值等价键数 == 模型键数（全量命中）
        loaded_state = loaded.state_dict()
        model_state = model.state_dict()
        assert sum(
            torch.equal(loaded_state[k], model_state[k]) for k in model_state
        ) == len(model_state)

    def test_zero_match_raises(self, tmp_path):
        """ckpt 与模型 0 匹配时显式 raise（疑似配置不对应）。"""
        ckpt_path = tmp_path / "bad.pt"
        torch.save({"module.unrelated.weight": torch.randn(2, 2)}, ckpt_path)
        with pytest.raises(ValueError, match="0 匹配"):
            TSE_BSRNN_SPK.load_pretrained(str(ckpt_path), **self._joint_kwargs())

    @pytest.mark.parametrize("failure", ["missing", "unexpected", "shape"])
    def test_public_loader_fails_closed_before_copy(self, tmp_path, failure):
        """公共 loader 对不完整、额外或 shape 不符权重均须在复制前拒绝。"""
        model = TSE_BSRNN_SPK(**self._joint_kwargs())
        state = dict(model.state_dict())
        key = next(iter(state))
        if failure == "missing":
            state.pop(key)
            match = "缺失键"
        elif failure == "unexpected":
            state["optimizer.unexpected"] = torch.ones(1)
            match = "未知键"
        else:
            state[key] = torch.zeros(state[key].numel() + 1)
            match = "shape"
        ckpt_path = tmp_path / f"{failure}.pt"
        torch.save({"model": state}, ckpt_path)

        with pytest.raises(ValueError, match=match):
            TSE_BSRNN_SPK.load_pretrained(
                str(ckpt_path), strict=True, **self._joint_kwargs()
            )

    def test_official_compatibility_mode_keeps_non_strict_exception(self, tmp_path):
        """显式 strict=False 仅放宽官方附加键，不放宽 0-match/shape gate。"""
        model = TSE_BSRNN_SPK(**self._joint_kwargs())
        state = dict(model.state_dict())
        state["loss.weight"] = torch.ones(1)
        ckpt_path = tmp_path / "official_extra.pt"
        torch.save({"model": state}, ckpt_path)

        loaded = TSE_BSRNN_SPK.load_pretrained(
            str(ckpt_path), strict=False, **self._joint_kwargs()
        )
        for key, value in model.state_dict().items():
            assert torch.equal(loaded.state_dict()[key], value)

    @pytest.mark.parametrize("failure", ["zero", "shape"])
    def test_official_compatibility_mode_retains_hard_gates(self, tmp_path, failure):
        model = TSE_BSRNN_SPK(**self._joint_kwargs())
        if failure == "zero":
            state = {"unrelated.weight": torch.ones(1)}
            match = "0 匹配"
        else:
            state = dict(model.state_dict())
            key = next(iter(state))
            state[key] = torch.zeros(state[key].numel() + 1)
            match = "shape"
        ckpt_path = tmp_path / f"official_{failure}.pt"
        torch.save({"model": state}, ckpt_path)

        with pytest.raises(ValueError, match=match):
            TSE_BSRNN_SPK.load_pretrained(
                str(ckpt_path), strict=False, **self._joint_kwargs()
            )

    def test_unsafe_ckpt_fails_closed(self, tmp_path):
        """含自定义类等不可安全加载对象的 ckpt：safe_torch_load 直接抛错，不回退。"""
        model = TSE_BSRNN_SPK(**self._joint_kwargs())
        ckpt_path = tmp_path / "unsafe.pt"
        torch.save(
            {"model": dict(model.state_dict()), "meta": _DummyMeta()},
            ckpt_path,
        )
        with pytest.raises(RuntimeError, match="allow_unsafe=True"):
            TSE_BSRNN_SPK.load_pretrained(str(ckpt_path), **self._joint_kwargs())

    def test_safe_ckpt_loads_directly(self, tmp_path):
        """合法安全 ckpt（仅 tensor 权重）直接加载成功，不触发任何回退。"""
        model = TSE_BSRNN_SPK(**self._joint_kwargs())
        ckpt_path = tmp_path / "safe.pt"
        torch.save({"model": dict(model.state_dict())}, ckpt_path)
        loaded = TSE_BSRNN_SPK.load_pretrained(
            str(ckpt_path), **self._joint_kwargs()
        )
        for k, v in model.state_dict().items():
            assert torch.equal(loaded.state_dict()[k], v)

    def test_missing_file_raises(self, tmp_path):
        """非 weights_only 类异常（如文件缺失）不得被静默吞掉。"""
        with pytest.raises(FileNotFoundError):
            TSE_BSRNN_SPK.load_pretrained(
                str(tmp_path / "nope.pt"), **self._joint_kwargs()
            )


class TestVariantConfigs:
    """单一 Dynamic TSE YAML 的四种手工 variant 组合实测 build + forward。"""

    CONFIG_DIR = (
        Path(__file__).resolve().parent.parent
        / "configs" / "tse" / "dynamic"
    )

    @pytest.mark.parametrize(
        ("variant", "causal", "spkemb", "tfmap", "context"),
        [
            ("spk_emb_100", False, True, False, False),
            ("spk_emb_causal_100", True, True, False, False),
            ("tfmap_context_100", False, False, True, True),
            ("tfmap_context_causal_100", True, False, True, True),
        ],
    )
    def test_variant_switches_from_single_yaml(
        self, variant, causal, spkemb, tfmap, context
    ):
        """四种官方 cue/causal 组合均从同一主配置切换并可构造前向。"""
        from core.models.build import build_model
        from core.utils.config import load_merged_config
        import tasks.tse.models  # noqa: F401

        cfg = load_merged_config(self.CONFIG_DIR / "tse_bsrnn_spk.yaml")
        args = cfg["model"]["args"]
        args["causal"] = causal
        feats = args["features"]
        feats["spkemb"]["enabled"] = spkemb
        feats.setdefault("tfmap", {})["enabled"] = tfmap
        feats.setdefault("context", {})["enabled"] = context
        assert args["causal"] is causal, variant
        assert feats["spkemb"]["enabled"] is spkemb, variant
        assert feats["tfmap"]["enabled"] is tfmap, variant
        assert feats["context"]["enabled"] is context, variant
        # 本测试只验证单配置切换、模型构造和 forward；随机 speaker encoder
        # 已足够覆盖这些结构契约，不应依赖仓库外的比赛 checkpoint。
        cfg["model"]["args"]["speaker_model"]["speaker_encoder"]["pretrained"] = None
        model = build_model(cfg["model"], cfg.get("stft"))
        model.eval()
        with torch.no_grad():
            _check_out(model(_make_batch()))


class TestFrozenEncoder:
    """冻结 speaker encoder：model.train() 后 spk_model 恒为 eval。"""

    def _make_spk_ckpt(self, tmp_path) -> str:
        """合成 wespeaker 格式 ckpt（module. 前缀 + 显式允许的 loss. 附加键）。"""
        enc = SpeakerEncoder(pretrained=None)
        state = {f"module.{k}": v for k, v in enc.spk_model.state_dict().items()}
        state["loss.weight"] = torch.randn(3)
        ckpt_path = tmp_path / "spk.pt"
        torch.save(state, ckpt_path)
        return str(ckpt_path)

    def test_train_keeps_frozen_spk_model_eval(self, tmp_path):
        model = TSE_BSRNN_SPK(
            **_small_kwargs(
                joint_training=True,
                features={"spkemb": {"enabled": True}},
                speaker_model={
                    "speaker_encoder": {"pretrained": self._make_spk_ckpt(tmp_path)}
                },
            )
        )
        assert model.spk_ft.encoder.freeze  # 有预训练默认冻结
        model.train()
        assert model.training
        assert model.spk_ft.encoder.spk_model.training is False

    def test_encoder_zero_match_raises(self, tmp_path):
        """encoder ckpt 与模型 0 匹配时显式 raise。"""
        ckpt_path = tmp_path / "bad.pt"
        torch.save({"unrelated.weight": torch.randn(2)}, ckpt_path)
        with pytest.raises(ValueError, match="0 匹配"):
            SpeakerEncoder(pretrained=str(ckpt_path))

    def _write_encoder_state(self, tmp_path, mutate):
        enc = SpeakerEncoder(pretrained=None)
        state = {f"module.{k}": v.clone() for k, v in enc.spk_model.state_dict().items()}
        mutate(state)
        ckpt_path = tmp_path / "invalid_spk.pt"
        torch.save(state, ckpt_path)
        return ckpt_path

    def test_encoder_unknown_key_raises(self, tmp_path):
        """未声明的附加 namespace 不得被静默过滤。"""
        ckpt_path = self._write_encoder_state(
            tmp_path, lambda state: state.update({"optimizer.weight": torch.randn(1)})
        )
        with pytest.raises(ValueError, match="未知键"):
            SpeakerEncoder(pretrained=str(ckpt_path))

    def test_encoder_missing_key_raises(self, tmp_path):
        """缺失模型权重不得保留随机初始化。"""
        def remove_key(state):
            state.pop(next(iter(state)))

        ckpt_path = self._write_encoder_state(tmp_path, remove_key)
        with pytest.raises(ValueError, match="缺失键"):
            SpeakerEncoder(pretrained=str(ckpt_path))

    def test_encoder_shape_mismatch_raises(self, tmp_path):
        """已命中的权重 shape 不符时必须拒绝加载。"""
        def change_shape(state):
            state["module.linear.weight"] = torch.randn(1, 1)

        ckpt_path = self._write_encoder_state(tmp_path, change_shape)
        with pytest.raises(ValueError, match="shape"):
            SpeakerEncoder(pretrained=str(ckpt_path))

    def test_encoder_dtype_mismatch_raises(self, tmp_path):
        """已命中的权重 dtype 不符时必须拒绝加载。"""
        def change_dtype(state):
            state["module.linear.weight"] = state["module.linear.weight"].double()

        ckpt_path = self._write_encoder_state(tmp_path, change_dtype)
        with pytest.raises(ValueError, match="dtype"):
            SpeakerEncoder(pretrained=str(ckpt_path))

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, True),
            (42.0, True),
            (float("nan"), False),
            (float("inf"), False),
            (1.5, False),
            (-1.0, False),
            (float(2**63 - 2048), True),
            (float(2**63), False),
            (float(2**64), False),
        ],
    )
    def test_bn_counter_float_cast_boundaries(self, value, expected):
        """BN counter 转 int64 前必须有限、整数、非负并落在位宽范围内。"""
        actual = torch.tensor(value, dtype=torch.float64)
        assert _can_cast_bn_counter(actual, torch.int64) is expected


def _crossfuse_reference_forward(self, mix, emb):
    """CrossFuse 批量化前的串行实现（逐 band 一次 MHA），作为数值基准。"""
    spk_embeddings = []
    for i in range(mix.shape[1]):
        query = mix[:, i, :, :]
        query = self.Linear_mix(query.transpose(1, 2))
        key = self.Linear[i](emb.transpose(1, 2))
        value = key
        x, _ = self.multihead_attn(query, key, value)
        spk_embeddings.append(x.transpose(1, 2))
    return torch.stack(spk_embeddings, dim=1)


class TestCrossFuseBatched:
    """CrossFuse 4D 路径批量化（nband 次串行 MHA 合并为一次）的数值等价。"""

    def _make(self, seed=0, **kw):
        from tasks.tse.models.modules.fusion.speech import CrossFuse

        defaults = dict(embed_dim=16, atten_dim=8, mix_dim=8,
                        num_heads=2, nband=4, batch_first=True)
        defaults.update(kw)
        torch.manual_seed(seed)
        return CrossFuse(**defaults).eval()

    def _inputs(self, B, nband, mix_dim, T=5, Te=7, Fe=16, seed=1):
        g = torch.Generator().manual_seed(seed)
        mix = torch.randn(B, nband, mix_dim, T, generator=g)
        emb = torch.randn(B, Fe, Te, generator=g)
        return mix, emb

    def test_batched_matches_serial_float64(self):
        """float64 下批量实现与串行基准完全一致。"""
        for nband in (4, 32):
            m = self._make(nband=nband).double()
            mix, emb = self._inputs(2, nband, mix_dim=8)
            mix, emb = mix.double(), emb.double()
            with torch.no_grad():
                out_new = m(mix, emb)
                out_ref = _crossfuse_reference_forward(m, mix, emb)
            assert out_new.shape == out_ref.shape
            assert torch.equal(out_new, out_ref), (
                f"nband={nband} max|diff|="
                f"{(out_new - out_ref).abs().max().item():.3e}"
            )

    def test_batched_matches_serial_linear_mix(self):
        """mix_dim != atten_dim（Linear_mix 为真实 Linear）时同样等价。"""
        m = self._make(mix_dim=6, atten_dim=8).double()
        mix, emb = self._inputs(2, 4, mix_dim=6)
        mix, emb = mix.double(), emb.double()
        with torch.no_grad():
            out_new = m(mix, emb)
            out_ref = _crossfuse_reference_forward(m, mix, emb)
        assert torch.equal(out_new, out_ref)

    def test_batched_matches_serial_float32(self):
        m = self._make()
        mix, emb = self._inputs(2, 4, mix_dim=8)
        with torch.no_grad():
            out_new = m(mix, emb)
            out_ref = _crossfuse_reference_forward(m, mix, emb)
        assert torch.allclose(out_new, out_ref, atol=1e-6)

    def test_state_dict_keys_unchanged(self):
        """批量化不改模块结构：Linear.0..nband-1 键名保持（比赛 ckpt 兼容）。"""
        m = self._make(nband=32)
        keys = set(m.state_dict().keys())
        for i in range(32):
            assert f"Linear.{i}.weight" in keys
            assert f"Linear.{i}.bias" in keys
        assert "Linear_mix" not in keys  # mix_dim == atten_dim 时为 Identity
        assert "multihead_attn.in_proj_weight" in keys

pytestmark = [pytest.mark.unit]
