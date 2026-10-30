"""文本 KCE 前端（WeSep dc11cb18 移植）的模块级合同与数值测试。

覆盖离线 token cue → frontend 的 CPU forward/backward、缺失 mask、BOS/EOS/padding/
length 严格性、非法 token、checkpoint 注入与严格 key/shape 校验。这些测试只证明
frontend 模块本身可用，不代表 TSE_Modular 已接线、训练或部署通过。
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import yaml

from tasks.tse.cue_contract import (
    TSECue,
    TSECueSpec,
    TSECueSpecs,
    TSECues,
    collate_tse_cues,
)
from tasks.tse.models.modules.composition import CueComposer
from tasks.tse.models.modules.textual import (
    KCE,
    KCEWeightsError,
    TextualFrontend,
    TextualFrontendOutput,
)
from tasks.tse.models.modules.textual.kce import (
    DAEPhonemeTokenizer,
    validate_kce_model_config,
)


pytestmark = pytest.mark.unit


_MEL_BINS = 80
_WINDOW_SIZE = 400
_WINDOW_SHIFT = 160
# au_conv 两级 3x3 stride-2 卷积要求至少 7 帧 fbank，即 400 + 6 * 160 个样本。
_MIN_FRAMES = 7
_MIN_SAMPLES = _WINDOW_SIZE + (_MIN_FRAMES - 1) * _WINDOW_SHIFT
_MIX_DIM = 12
_NUM_STAGES = 3
_OUTPUT_DIM = 24
_TOKEN_CAPACITY = 8

# 小规模官方结构子集：参数布局、键名与深度和 DEFAULT_KCE_MODEL_CONFIG 一致，
# 只缩小维度，使 CPU 测试无需下载官方 checkpoint。
_MODEL_CONFIG = {
    "audio_net_config": {
        "num_mel_bins": _MEL_BINS,
        "input_trans": {"dim": [32, 32]},
        "transformer_config": {
            "size": 32,
            "self_att": "MultiHeadAtt",
            "self_att_config": {"n_head": 2, "n_feats": 32},
            "cross_att": "MultiHeadCrossAtt",
            "corss_att_config": {"n_head": 2, "n_feats": 32, "norm": "LayerNorm"},
            "feed_forward_config": {"dim": [32, 64, 32]},
            "conv_config": {"channels": 32, "kernel_size": 15},
        },
    },
    "kw_net_config": {
        "num_phn_token": 73,
        "input_trans": {"dim": [16, 16]},
        "transformer_config": {
            "size": 16,
            "self_att": "MultiHeadAtt",
            "self_att_config": {"n_head": 2, "n_feats": 16},
            "feed_forward_config": {"dim": [16, 32, 16]},
        },
    },
    "num_audio_block": 2,
    "num_kw_block": 1,
    "sv_net_config": {
        "num_classes": 10,
        "front_output_size": 32,
        "pooling_conf": {
            "type": "learnable_weights",
            "layer_indices": [0, 1],
            "use_softmax": True,
        },
    },
}


def _textual_config(**feature_overrides):
    feature = {
        "adapter": {"output_dim": _OUTPUT_DIM, "num_layers": 2},
        "model_config": _MODEL_CONFIG,
        "fusion": "FiLM",
    }
    feature.update(feature_overrides)
    return {"features": {"dae_kce": feature}}


def _kce_config(checkpoint=None, **overrides):
    config = {
        "adapter": {"output_dim": _OUTPUT_DIM, "num_layers": 2},
        "model_config": _MODEL_CONFIG,
    }
    config.update(overrides)
    if checkpoint is not None:
        config["checkpoint"] = str(checkpoint)
    return config


def _backbone_state():
    seed = KCE({"model_config": _MODEL_CONFIG}, defer_pretrained=True)
    return seed.model.state_dict()


@pytest.fixture(scope="module")
def backbone_state():
    return _backbone_state()


@pytest.fixture
def frontend(backbone_state):
    return TextualFrontend(
        _textual_config(),
        mix_dim=_MIX_DIM,
        num_stages=_NUM_STAGES,
        backbone_state_dict=backbone_state,
    )


def _token_cue(rows, capacity=_TOKEN_CAPACITY):
    tokens = torch.zeros((len(rows), capacity), dtype=torch.int64)
    lengths = torch.zeros(len(rows), dtype=torch.int32)
    present = torch.zeros(len(rows), dtype=torch.bool)
    for row, values in enumerate(rows):
        if values is None:
            continue
        present[row] = True
        lengths[row] = len(values)
        tokens[row, :len(values)] = torch.tensor(values, dtype=torch.int64)
    return tokens, present, lengths


def _mixture(rows, sample_lengths):
    generator = torch.Generator().manual_seed(11)
    return torch.randn(rows, max(sample_lengths), generator=generator) * 0.1


def _composer(output_dim, feature_dim=_MIX_DIM, nband=4):
    return CueComposer(
        feature_dim=feature_dim,
        nband=nband,
        num_blocks=1,
        spkemb={"fusion": "FiLM"},
        embedding_dim=output_dim,
    )


def _break_film_symmetry(module):
    """FiLM 初始化时 gamma/beta 全零（等价 identity），需要扰动才可观察融合效果。"""
    generator = torch.Generator().manual_seed(5)
    for parameter in module.parameters():
        with torch.no_grad():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)


def test_frontend_forward_and_backward_on_cpu(frontend):
    tokens, present, lengths = _token_cue([[3, 4, 5], [7, 8]])
    mixture = _mixture(2, [16000, 12000])
    mixture_lengths = torch.tensor([16000, 12000], dtype=torch.int32)

    output = frontend(mixture, tokens, present=present, textual_lengths=lengths,
                      mixture_lengths=mixture_lengths)

    assert isinstance(output, TextualFrontendOutput)
    assert output.cue.shape == (2, _OUTPUT_DIM)
    assert output.cue.dtype == torch.float32
    assert torch.isfinite(output.cue).all()
    assert output.present is present
    assert not torch.equal(output.cue[0], output.cue[1])

    # adapter 末尾是 LayerNorm，而 ``cue.sum()`` 对每个特征给出相同梯度，会被
    # LayerNorm 的 Jacobian 精确抵消（前面各层梯度恒为 0）；这里必须用逐特征
    # 不同的损失才能真正检验整条 adapter 的可训练性。
    output.cue.pow(2).sum().backward()
    assert all(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in frontend.dae_kce.kce.adapter.parameters()
    )
    assert all(parameter.grad is None for parameter in frontend.dae_kce.kce.model.parameters())


def test_frozen_backbone_keeps_adapter_trainable(frontend):
    kce = frontend.dae_kce.kce
    assert all(not parameter.requires_grad for parameter in kce.model.parameters())
    assert all(parameter.requires_grad for parameter in kce.adapter.parameters())
    frontend.train()
    assert kce.training is True
    assert kce.model.training is False


def test_batchnorm_eval_guard_rejects_training_backbone(frontend):
    tokens, present, lengths = _token_cue([[3, 4, 5]])
    mixture = _mixture(1, [16000])
    mixture_lengths = torch.tensor([16000], dtype=torch.int32)

    frontend.dae_kce.kce.model.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        frontend(mixture, tokens, present=present, textual_lengths=lengths,
                 mixture_lengths=mixture_lengths)


def test_missing_rows_are_zeroed_and_isolated(frontend):
    tokens, present, lengths = _token_cue([[3, 4, 5], None])
    mixture = _mixture(2, [16000, 16000])
    mixture_lengths = torch.tensor([16000, 9000], dtype=torch.int32)

    output = frontend(mixture, tokens, present=present, textual_lengths=lengths,
                      mixture_lengths=mixture_lengths)
    assert float(output.cue[1].abs().sum()) == 0.0
    assert torch.count_nonzero(output.cue[0]) > 0

    changed_tokens, _, _ = _token_cue([[3, 4, 5], None])
    changed_mixture = mixture.clone()
    changed_mixture[1] = torch.randn_like(changed_mixture[1])
    repeated = frontend(changed_mixture, changed_tokens, present=present,
                        textual_lengths=lengths, mixture_lengths=mixture_lengths)
    assert torch.equal(repeated.cue[0], output.cue[0])
    assert float(repeated.cue[1].abs().sum()) == 0.0


def test_offline_token_cue_batch_feeds_frontend(frontend):
    specs = TSECueSpecs(textual=TSECueSpec(representation="phoneme_ids", value_shape=(_TOKEN_CAPACITY,)))
    rows = [
        TSECues(textual=TSECue(
            representation="phoneme_ids",
            value=torch.tensor([3, 4, 5], dtype=torch.int64),
            present=True,
            length=3,
            timebase=None,
        )),
        TSECues(textual=TSECue(
            representation="phoneme_ids",
            value=None,
            present=False,
            length=0,
            timebase=None,
        )),
    ]
    batches = collate_tse_cues(rows, specs)
    textual = batches.textual
    assert textual is not None
    mixture = _mixture(2, [16000, 16000])

    output = frontend(
        mixture,
        textual.value,
        present=textual.present,
        textual_lengths=textual.lengths,
        mixture_lengths=torch.tensor([16000, 8000], dtype=torch.int32),
    )
    assert output.cue.shape == (2, _OUTPUT_DIM)
    assert float(output.cue[1].abs().sum()) == 0.0


def test_keyword_sequence_uses_bos_eos_and_lengths(frontend, monkeypatch):
    recorded = {}
    kce = frontend.dae_kce.kce

    def recorder(speech, speech_lengths, keyword, keyword_lengths):
        recorded["speech"] = speech
        recorded["speech_lengths"] = speech_lengths
        recorded["keyword"] = keyword
        recorded["keyword_lengths"] = keyword_lengths
        return torch.zeros((keyword.shape[0], kce.backbone_dim))

    monkeypatch.setattr(kce.model, "forward", recorder)
    tokens, present, lengths = _token_cue([[3, 0, 5], None])
    mixture = _mixture(2, [16000, 16000])
    mixture_lengths = torch.tensor([16000, 16000], dtype=torch.int32)

    frontend(mixture, tokens, present=present, textual_lengths=lengths,
             mixture_lengths=mixture_lengths)

    assert recorded["keyword"][0].tolist() == [71, 3, 5, 72]
    assert recorded["keyword"][1].tolist() == [71, 72, 0, 0]
    assert recorded["keyword_lengths"].tolist() == [4, 2]
    assert recorded["keyword"].dtype == torch.int64
    expected_frames = 1 + (16000 - _WINDOW_SIZE) // _WINDOW_SHIFT
    assert recorded["speech_lengths"].tolist() == [expected_frames, expected_frames]
    assert recorded["speech"].shape[1] == expected_frames
    assert recorded["speech"].shape[2] == _MEL_BINS


def test_fbank_uses_valid_mixture_length(frontend, monkeypatch):
    recorded = {}
    kce = frontend.dae_kce.kce
    monkeypatch.setattr(
        kce.model,
        "forward",
        lambda speech, speech_lengths, keyword, keyword_lengths: (
            recorded.update(speech_lengths=speech_lengths)
            or torch.zeros((keyword.shape[0], kce.backbone_dim))
        ),
    )
    tokens, present, lengths = _token_cue([[3, 4, 5], [6, 7]])
    mixture = _mixture(2, [16000, 16000])

    frontend(mixture, tokens, present=present, textual_lengths=lengths,
             mixture_lengths=torch.tensor([16000, 9000], dtype=torch.int32))

    assert recorded["speech_lengths"].tolist() == [
        1 + (16000 - _WINDOW_SIZE) // _WINDOW_SHIFT,
        1 + (9000 - _WINDOW_SIZE) // _WINDOW_SHIFT,
    ]


@pytest.mark.parametrize(
    "lengths, match",
    [
        ([0, 0], "textual_lengths"),
        ([9, 0], "textual_lengths"),
        ([3, 1], "textual_lengths"),
    ],
)
def test_textual_lengths_must_match_present_mask(frontend, lengths, match):
    tokens, present, _ = _token_cue([[3, 4, 5], None])
    mixture = _mixture(2, [16000, 16000])
    with pytest.raises(ValueError, match=match):
        frontend(mixture, tokens, present=present,
                 textual_lengths=torch.tensor(lengths, dtype=torch.int32),
                 mixture_lengths=torch.tensor([16000, 16000], dtype=torch.int32))


def test_tail_tokens_must_be_padding(frontend):
    tokens, present, lengths = _token_cue([[3, 4, 5], [6, 7]])
    tokens[0, 3] = 9
    mixture = _mixture(2, [16000, 16000])
    with pytest.raises(ValueError, match="padding_id"):
        frontend(mixture, tokens, present=present, textual_lengths=lengths,
                 mixture_lengths=torch.tensor([16000, 16000], dtype=torch.int32))


@pytest.mark.parametrize("bad_value", [73, 100, -1])
def test_out_of_range_phoneme_ids_rejected(frontend, bad_value):
    tokens, present, lengths = _token_cue([[3, 4, 5], [6, 7]])
    tokens[0, 1] = bad_value
    mixture = _mixture(2, [16000, 16000])
    with pytest.raises(ValueError, match="phoneme_ids"):
        frontend(mixture, tokens, present=present, textual_lengths=lengths,
                 mixture_lengths=torch.tensor([16000, 16000], dtype=torch.int32))


def test_non_int64_phoneme_ids_rejected(frontend):
    tokens, present, lengths = _token_cue([[3, 4, 5], [6, 7]])
    mixture = _mixture(2, [16000, 16000])
    with pytest.raises(TypeError, match="int64"):
        frontend(mixture, tokens.to(torch.int32), present=present, textual_lengths=lengths,
                 mixture_lengths=torch.tensor([16000, 16000], dtype=torch.int32))


def test_short_and_mismatched_inputs_rejected(frontend):
    tokens, present, lengths = _token_cue([[3, 4, 5]])
    with pytest.raises(ValueError, match="samples"):
        frontend(torch.zeros(1, _MIN_SAMPLES - 1), tokens, present=present,
                 textual_lengths=lengths,
                 mixture_lengths=torch.tensor([_MIN_SAMPLES - 1], dtype=torch.int32))
    with pytest.raises(ValueError, match="mixture_lengths"):
        frontend(torch.zeros(1, _MIN_SAMPLES), tokens, present=present,
                 textual_lengths=lengths,
                 mixture_lengths=torch.tensor([_MIN_SAMPLES - 1], dtype=torch.int32))


def test_conv_stack_minimum_geometry(frontend):
    kce = frontend.dae_kce.kce
    assert kce.minimum_fbank_frames == _MIN_FRAMES
    assert kce.minimum_samples == _MIN_SAMPLES

    tokens, present, lengths = _token_cue([[3, 4, 5]])
    mixture = torch.randn(1, _MIN_SAMPLES, generator=torch.Generator().manual_seed(7)) * 0.1
    output = frontend(mixture, tokens, present=present, textual_lengths=lengths,
                      mixture_lengths=torch.tensor([_MIN_SAMPLES], dtype=torch.int32))
    assert output.cue.shape == (1, _OUTPUT_DIM)
    assert torch.isfinite(output.cue).all()


def test_missing_backbone_weights_are_never_silent():
    with pytest.raises(KCEWeightsError, match="absent"):
        KCE({"model_config": _MODEL_CONFIG})

    deferred = KCE({"model_config": _MODEL_CONFIG}, defer_pretrained=True)
    assert deferred.weights_loaded is False
    tokens, present, lengths = _token_cue([[3, 4, 5]])
    with pytest.raises(KCEWeightsError, match="never loaded"):
        deferred(_mixture(1, [16000]), tokens, present=present, textual_lengths=lengths,
                 mixture_lengths=torch.tensor([16000], dtype=torch.int32))


def test_checkpoint_path_matches_injected_state_dict(tmp_path, backbone_state):
    path = tmp_path / "epoch_149.pt"
    torch.save({"model": backbone_state}, path)
    injected = TextualFrontend(_textual_config(), mix_dim=_MIX_DIM,
                               num_stages=_NUM_STAGES,
                               backbone_state_dict=backbone_state)
    loaded = TextualFrontend(_textual_config(), mix_dim=_MIX_DIM,
                             num_stages=_NUM_STAGES,
                             backbone_state_dict=_backbone_state())
    loaded.load_state_dict(injected.state_dict())

    tokens, present, lengths = _token_cue([[3, 4, 5], None])
    mixture = _mixture(2, [16000, 16000])
    mixture_lengths = torch.tensor([16000, 9000], dtype=torch.int32)
    assert torch.equal(
        injected(mixture, tokens, present=present, textual_lengths=lengths,
                 mixture_lengths=mixture_lengths).cue,
        loaded(mixture, tokens, present=present, textual_lengths=lengths,
               mixture_lengths=mixture_lengths).cue,
    )

    from_path = KCE(_kce_config(checkpoint=path))
    from_path.adapter.load_state_dict(injected.dae_kce.kce.adapter.state_dict())
    assert from_path.weights_loaded is True
    assert torch.equal(
        injected.dae_kce.kce(mixture, tokens, present=present, textual_lengths=lengths,
                             mixture_lengths=mixture_lengths),
        from_path(mixture, tokens, present=present, textual_lengths=lengths,
                  mixture_lengths=mixture_lengths),
    )


def test_checkpoint_key_and_shape_mismatch_rejected(backbone_state):
    missing = dict(backbone_state)
    missing.pop(sorted(missing)[0])
    with pytest.raises(KCEWeightsError, match="missing"):
        KCE({"model_config": _MODEL_CONFIG}, backbone_state_dict=missing)

    extra = dict(backbone_state)
    extra["unexpected.weight"] = torch.zeros(1)
    with pytest.raises(KCEWeightsError, match="unexpected"):
        KCE({"model_config": _MODEL_CONFIG}, backbone_state_dict=extra)

    reshaped = dict(backbone_state)
    key = sorted(reshaped)[0]
    reshaped[key] = reshaped[key][:1]
    with pytest.raises(KCEWeightsError, match="shape_or_type"):
        KCE({"model_config": _MODEL_CONFIG}, backbone_state_dict=reshaped)


def test_outer_checkpoint_without_backbone_fills_from_external_checkpoint(tmp_path):
    path = tmp_path / "epoch_149.pt"
    torch.save({"model": _backbone_state()}, path)
    reference = KCE(_kce_config(checkpoint=path))

    outer = KCE(_kce_config(checkpoint=path), defer_pretrained=True)
    outer.load_state_dict(
        {"adapter.2.weight": outer.adapter[2].weight.detach().clone()}, strict=False
    )
    outer.adapter.load_state_dict(reference.adapter.state_dict())
    assert outer.weights_loaded is True

    tokens, present, lengths = _token_cue([[3, 4, 5], None])
    mixture = _mixture(2, [16000, 16000])
    mixture_lengths = torch.tensor([16000, 16000], dtype=torch.int32)
    assert torch.equal(
        outer(mixture, tokens, present=present, textual_lengths=lengths,
              mixture_lengths=mixture_lengths),
        reference(mixture, tokens, present=present, textual_lengths=lengths,
                  mixture_lengths=mixture_lengths),
    )


def test_outer_checkpoint_without_external_checkpoint_fails():
    outer = KCE({"model_config": _MODEL_CONFIG}, defer_pretrained=True)
    with pytest.raises(KCEWeightsError, match="absent"):
        outer.load_state_dict(
            {"adapter.0.weight": outer.adapter[0].weight.detach().clone()}, strict=False
        )


def test_sample_rate_is_strictly_16k(backbone_state):
    with pytest.raises(ValueError, match="16000"):
        TextualFrontend(_textual_config(), mix_dim=_MIX_DIM, num_stages=_NUM_STAGES,
                        sample_rate=8000, backbone_state_dict=backbone_state)
    with pytest.raises(ValueError, match="sample_frequency"):
        KCE(
            {"model_config": _MODEL_CONFIG,
             "fbank": {"num_mel_bins": _MEL_BINS, "frame_length": 25.0,
                       "frame_shift": 10.0, "sample_frequency": 8000.0}},
            backbone_state_dict=backbone_state,
        )


def test_fbank_must_match_backbone_mel_bins(backbone_state):
    with pytest.raises(ValueError, match="num_mel_bins"):
        KCE(
            {"model_config": _MODEL_CONFIG,
             "fbank": {"num_mel_bins": 40, "frame_length": 25.0,
                       "frame_shift": 10.0, "sample_frequency": 16000.0}},
            backbone_state_dict=backbone_state,
        )
    with pytest.raises(ValueError, match="unknown Kaldi"):
        KCE(
            {"model_config": _MODEL_CONFIG,
             "fbank": {"num_mel_bins": _MEL_BINS, "frame_length": 25.0,
                       "frame_shift": 10.0, "sample_frequency": 16000.0,
                       "n_mels": 80}},
            backbone_state_dict=backbone_state,
        )


def test_unknown_config_keys_rejected(backbone_state):
    with pytest.raises(KeyError, match="Unknown config key"):
        TextualFrontend({**_textual_config(), "extra": 1}, mix_dim=_MIX_DIM,
                        num_stages=_NUM_STAGES, backbone_state_dict=backbone_state)
    with pytest.raises(KeyError, match="Unknown config key"):
        TextualFrontend({"features": {"kce_v2": {}}}, mix_dim=_MIX_DIM,
                        num_stages=_NUM_STAGES, backbone_state_dict=backbone_state)
    with pytest.raises(KeyError, match="Unknown config key"):
        TextualFrontend(_textual_config(prettrained="x"), mix_dim=_MIX_DIM,
                        num_stages=_NUM_STAGES, backbone_state_dict=backbone_state)
    with pytest.raises(ValueError, match="unknown keys"):
        validate_kce_model_config({**_MODEL_CONFIG, "num_audio_blcok": 8})
    with pytest.raises(ValueError, match="unknown keys"):
        KCE({"model_config": _MODEL_CONFIG, "pretrained": "x"},
            backbone_state_dict=backbone_state)


def test_official_dae_sidecars_load_with_inference_schema(tmp_path, backbone_state):
    model_config = deepcopy(_MODEL_CONFIG)
    model_config["audio_net_config"].pop("num_mel_bins")
    model_config["audio_net_config"]["input_trans"]["norm"] = "LayerNorm"
    audio_transformer = model_config["audio_net_config"]["transformer_config"]
    audio_transformer["conv_config"].pop("kernel_size")
    audio_transformer["feed_forward_config"]["norm"] = "LayerNorm"
    model_config["kw_net_config"]["input_trans"]["dim"] = 16
    model_config["kw_net_config"]["input_trans"]["norm"] = "LayerNorm"
    model_config["kw_net_config"]["transformer_config"]["feed_forward_config"][
        "norm"
    ] = "LayerNorm"
    model_config["kw_net_config"]["num_bpe_token"] = 5005
    model_config.update({
        "decoder_net_config": {"training_only": True},
        "eos": 5004,
        "num_doecoder_block": 4,
        "sos": 2,
        "loss_weight": [0.3, 0.1, 0.3],
    })

    checkpoint = tmp_path / "epoch_149.pt"
    torch.save({"model": backbone_state}, checkpoint)
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(model_config))
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({
        "speech_config": {
            "feats_config": {
                "frame_length": 25,
                "frame_shift": 10,
                "num_mel_bins": _MEL_BINS,
            },
        },
    }))

    kce = KCE({"checkpoint": str(checkpoint)}, sample_rate=16000)
    assert kce.weights_loaded
    assert len(kce.model.state_dict()) == len(backbone_state)
    assert kce.num_phn_token == model_config["kw_net_config"]["num_phn_token"]
    assert kce.fbank_config["sample_frequency"] == 16000.0


def test_disabled_feature_and_bad_geometry_rejected(backbone_state):
    with pytest.raises(ValueError, match="enabled"):
        TextualFrontend(_textual_config(enabled=False), mix_dim=_MIX_DIM,
                        num_stages=_NUM_STAGES, backbone_state_dict=backbone_state)
    with pytest.raises(ValueError, match="mix_dim"):
        TextualFrontend(_textual_config(), mix_dim=0, num_stages=_NUM_STAGES,
                        backbone_state_dict=backbone_state)
    with pytest.raises(ValueError, match="num_stages"):
        TextualFrontend(_textual_config(), mix_dim=_MIX_DIM, num_stages=0,
                        backbone_state_dict=backbone_state)


def test_multi_fuse_false_builds_single_fusion(backbone_state):
    frontend = TextualFrontend(_textual_config(multi_fuse=False), mix_dim=_MIX_DIM,
                               num_stages=_NUM_STAGES,
                               backbone_state_dict=backbone_state)
    assert frontend.multi_fuse is False
    assert len(frontend.dae_kce.fusions) == 1
    assert frontend.dae_kce.config["num_stages"] == _NUM_STAGES


def test_post_fuses_each_stage_and_gates_missing_rows(frontend):
    _break_film_symmetry(frontend.dae_kce.fusions)
    tokens, present, lengths = _token_cue([[3, 4, 5], None])
    mixture = _mixture(2, [16000, 16000])
    output = frontend(mixture, tokens, present=present, textual_lengths=lengths,
                      mixture_lengths=torch.tensor([16000, 16000], dtype=torch.int32))
    latent = torch.randn(2, 4, _MIX_DIM, 7, generator=torch.Generator().manual_seed(3))

    for stage in range(_NUM_STAGES):
        fused = frontend.post(latent, output, stage)
        assert fused.shape == latent.shape
        assert not torch.equal(fused[0], latent[0])
        assert torch.equal(fused[1], latent[1])

    with pytest.raises(ValueError, match="stage"):
        frontend.post(latent, output, _NUM_STAGES)
    with pytest.raises(TypeError, match="TextualFrontendOutput"):
        frontend.post(latent, output.cue, 0)


def test_cue_is_composer_broadcast_compatible(frontend):
    tokens, present, lengths = _token_cue([[3, 4, 5], None])
    mixture = _mixture(2, [16000, 16000])
    output = frontend(mixture, tokens, present=present, textual_lengths=lengths,
                      mixture_lengths=torch.tensor([16000, 16000], dtype=torch.int32))
    latent = torch.randn(2, 4, _MIX_DIM, 7, generator=torch.Generator().manual_seed(4))

    composer = _composer(frontend.output_dim)
    _break_film_symmetry(composer.speaker_fusion)
    fused = composer.fuse_latent(latent, present=output.present, embedding=output.cue)
    assert fused.shape == latent.shape
    assert not torch.equal(fused[0], latent[0])
    assert torch.equal(fused[1], latent[1])


def test_tokenizer_normalize_and_lexicon_path(tmp_path):
    phoneme_map = tmp_path / "phoneme2int.txt"
    phoneme_map.write_text("SIL 0\nB 1\nAH0 2\nK 3\n", encoding="utf-8")
    lexicon = tmp_path / "word2lexicon.txt"
    lexicon.write_text("back 1 2 3\n", encoding="utf-8")

    tokenizer = DAEPhonemeTokenizer(str(phoneme_map), str(lexicon))
    assert tokenizer.normalize("  Back, please! ") == "back please"
    normalized, words, labels = tokenizer.encode_words("Back")
    assert normalized == "back"
    assert words == ["back"]
    assert labels == [[1, 2, 3]]
    assert tokenizer.encode("back") == [1, 2, 3]


def test_build_tokenizer_requires_configuration(frontend, tmp_path):
    with pytest.raises(RuntimeError, match="tokenizer"):
        frontend.build_tokenizer()

    phoneme_map = tmp_path / "phoneme2int.txt"
    phoneme_map.write_text("SIL 0\nB 1\n", encoding="utf-8")
    configured = TextualFrontend(
        _textual_config(tokenizer={"phoneme_map": str(phoneme_map)}),
        mix_dim=_MIX_DIM,
        num_stages=_NUM_STAGES,
        backbone_state_dict=_backbone_state(),
    )
    assert isinstance(configured.build_tokenizer(), DAEPhonemeTokenizer)
