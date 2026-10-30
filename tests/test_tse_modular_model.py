"""Modular 的真实特性、唯一参数 owner 与 runtime STFT 集成。"""

import copy

import pytest
import torch

from core.models.build import build_model
from core.stft import StftIstftModule as STFT
from tasks.tse.cue_contract import TSECueBatch, TSECueBatches
from tasks.tse.model_io import build_tse_model_inputs, decode_tse_model_output
from tasks.tse.models.discriminative.tse_modular import TSE_Modular
from tasks.tse.models.modules.composition import CueComposer, gate_fusion
from tasks.tse.models.modules.textual.kce import KCE, KCEWeightsError

pytestmark = pytest.mark.unit

# 小规模官方结构子集：键名与深度和 DEFAULT_KCE_MODEL_CONFIG 一致，只缩小维度，
# 使 CPU 测试无需下载官方 DAE-TSE checkpoint。
_KCE_MODEL_CONFIG = {
    "audio_net_config": {
        "num_mel_bins": 80,
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
_TEXTUAL_TOKENS = torch.tensor([[3, 4, 5, 0, 0], [7, 8, 0, 0, 0]])
_VISUAL_VALUE = torch.tensor([
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
])


@pytest.fixture(autouse=True)
def _threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def kce_checkpoint(tmp_path_factory):
    """官方结构的随机权重 checkpoint；缺真实 DAE-TSE 资源时仍可跑真实模块。"""
    path = tmp_path_factory.mktemp("tse_modular_kce") / "epoch_149.pt"
    seed = KCE({"model_config": _KCE_MODEL_CONFIG}, defer_pretrained=True)
    torch.save({"model": seed.model.state_dict()}, path)
    return str(path)


def _args(speaker=(), spatial=(), *, embedding=False, track=False, missing=False):
    cues = {}
    if speaker:
        features = {name: {} for name in speaker}
        if "usef" in features:
            features["usef"] = {"latent_dim": 4, "num_heads": 2, "approx_qk_dim": 8}
        cues["speaker"] = {"representation": "embedding" if embedding else "waveform", "features": features}
        if embedding:
            cues["speaker"]["encoder"] = {"embedding_dim": 8}
    if spatial:
        cues["spatial"] = {"representation": "track" if track else "static", "features": {name: {} for name in spatial}}
    return {
        "backbone": "bsrnn", "separator": {"feature_dim": 8, "num_repeat": 2, "band_width": [16, 17]},
        "cues": cues, "composition": {"allow_all_missing": missing, **({"usef_mode": "append"} if "usef" in speaker else {})},
        "sample_rate": 16000, "n_fft": 64, "hop_length": 16, "win_length": 64,
        "n_channel": 2 if spatial else 1, "ref_channel": 1 if spatial else 0,
        **({"mic_positions_m": [[0, 0, 0], [0.05, 0, 0]], "speed_of_sound_mps": 340.0} if spatial else {}),
    }


def _textual_args(checkpoint, *, fusion=None, multi_fuse=None, num_repeat=2, missing=False):
    feature = {
        "checkpoint": checkpoint,
        "model_config": _KCE_MODEL_CONFIG,
        "adapter": {"output_dim": 8, "num_layers": 1},
    }
    if fusion is not None:
        feature["fusion"] = fusion
    if multi_fuse is not None:
        feature["multi_fuse"] = multi_fuse
    args = _args(missing=missing)
    args["separator"]["num_repeat"] = num_repeat
    args["cues"] = {"textual": {"representation": "phoneme_ids", "features": {"dae_kce": feature}}}
    return args


def _visual_args(*, fusion=None, multi_fuse=None, adapter=None, num_repeat=2, missing=False):
    feature = {"adapter": {"type": "direct", "input_dim": 4, "output_dim": 4} if adapter is None else adapter}
    if fusion is not None:
        feature["fusion"] = fusion
    if multi_fuse is not None:
        feature["multi_fuse"] = multi_fuse
    args = _args(missing=missing)
    args["separator"]["num_repeat"] = num_repeat
    args["cues"] = {"visual": {"representation": "features", "features": {"muse_visual": feature}}}
    return args


def _inputs(model, present=(True, True)):
    present = torch.tensor(present)
    result = {"mix_spectra": torch.randn(2, model.n_channel, 33, 6, 2) if model.n_channel > 1 else torch.randn(2, 33, 6, 2)}
    names = {field.name for field in model.IO_CONTRACT.inputs}
    if "speaker_waveform_present" in names:
        result["speaker_waveform_present"] = present
    if "enroll_spectra" in names:
        result.update(enroll_spectra=torch.randn(2, 33, 5, 2), enroll_spectra_lengths=torch.tensor([5, 3]) * present)
    if "speaker_waveform" in names:
        result.update(speaker_waveform=torch.randn(2, 1200), speaker_waveform_lengths=torch.tensor([1200, 900]) * present)
    if "mix_wav" in names:
        result.update(mix_wav=torch.randn(2, 512), lengths=torch.tensor([512, 512], dtype=torch.int32))
    if "speaker_embedding" in names:
        result.update(speaker_embedding=torch.randn(2, model.embedding_dim), speaker_embedding_present=present)
    if "spatial" in names:
        result.update(spatial=torch.zeros(2, 2), spatial_present=torch.ones(2, dtype=torch.bool))
        if "spatial_timebase" in names:
            result.update(spatial=torch.zeros(2, 2, 2), spatial_lengths=torch.tensor([2, 2], dtype=torch.int32), spatial_timebase=torch.tensor([[0, 48], [0, 48]]))
    if "textual" in names:
        result.update(_textual_inputs(present=present))
    if "visual" in names:
        result.update(_visual_inputs(present=present))
    return result


def _textual_inputs(*, present=(True, True), tokens=None, waveform=None, frames=96):
    present = torch.tensor(present)
    tokens = _TEXTUAL_TOKENS.clone() if tokens is None else tokens
    tokens[~present] = 0
    return {
        "mix_spectra": torch.randn(2, 33, frames, 2),
        "mix_wav": torch.randn(2, 16000) if waveform is None else waveform,
        "lengths": torch.tensor([16000, 12000], dtype=torch.int32),
        "textual": tokens,
        "textual_present": present,
        "textual_lengths": torch.tensor(
            [int((row != 0).sum()) for row in tokens], dtype=torch.int32
        ) * present,
    }


def _visual_inputs(*, present=(True, True), value=None, frames=6, timebase=None, lengths=None):
    present = torch.tensor(present)
    value = _VISUAL_VALUE.clone() if value is None else value
    if timebase is None:
        timebase = torch.tensor(
            [[0.0, 0.003, 0.006], [0.0, 0.003, 0.0]], dtype=torch.float64
        )
        timebase[~present] = 0.0
    return {
        "mix_spectra": torch.randn(2, 33, frames, 2),
        "visual": value,
        "visual_present": present,
        "visual_lengths": (
            torch.tensor([3, 2], dtype=torch.int32) if lengths is None else lengths
        ) * present,
        "visual_timebase": timebase,
    }


def _fused(model, data):
    estimate = model(data)["target_estimate_spectra"]
    estimate.square().mean().backward()
    return estimate


def _grads(module):
    return [parameter.grad for parameter in module.parameters() if parameter.requires_grad]


@pytest.mark.parametrize("feature", ["tfmap", "usef", "context", "spkemb"])
@pytest.mark.parametrize("missing", [False, True])
def test_speaker_features_forward_backward_fixed_graph(feature, missing):
    model = TSE_Modular(**_args((feature,), missing=missing)).train()
    data = _inputs(model, (False, False) if missing else (True, True))
    estimate = model(data)["target_estimate_spectra"]
    assert estimate.shape == (2, 33, 6, 2)
    estimate.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.requires_grad)
    if model.speaker_encoder is not None:
        assert not model.speaker_encoder.training
        assert not model.speaker_encoder.freeze
        gradients = [parameter.grad for parameter in model.speaker_encoder.parameters() if parameter.requires_grad]
        assert gradients
        if missing:
            assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)
        else:
            assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_tfmap_embedding_reuses_frame_condition_and_backpropagates():
    args = _args(("tfmap", "context"))
    args["cues"]["speaker"]["features"]["tfmap"] = {"type": "embedding"}
    args["cues"]["speaker"]["features"]["context"].update(
        {"source": "tfmap_embedding", "injection": "latent", "attention_dim": 4, "num_heads": 1}
    )
    args["cues"]["speaker"]["encoder"] = {
        "name": "ECAPA_TDNN_GLOB_c512",
        "fbank": {
            "num_mel_bins": 8,
            "frame_length_ms": 4.0,
            "frame_shift_ms": 1.0,
            "dither": 0.0,
        },
    }
    model = TSE_Modular(**args).train()
    data = {
        "mix_spectra": torch.randn(2, 33, 33, 2),
        "mix_wav": torch.randn(2, 512),
        "lengths": torch.full((2,), 512, dtype=torch.int32),
        "speaker_waveform_present": torch.ones(2, dtype=torch.bool),
        "speaker_waveform": torch.randn(2, 256),
        "speaker_waveform_lengths": torch.full((2,), 256, dtype=torch.int32),
        "enroll_spectra": torch.randn(2, 33, 17, 2),
        "enroll_spectra_lengths": torch.full((2,), 17, dtype=torch.int32),
    }
    calls = []
    original = model.speaker_encoder.forward_frame_level

    def counted(waveform):
        calls.append(tuple(waveform.shape))
        return original(waveform)

    model.speaker_encoder.forward_frame_level = counted
    output = model(data)["target_estimate_spectra"]
    output.square().mean().backward()
    assert output.shape == data["mix_spectra"].shape
    assert calls == [(1, 576), (1, 320), (1, 576), (1, 320)]
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.speaker_encoder.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize("features", [("cdf",), ("sdf",), ("ipd", "cdf", "delta_stft"), ("cyc_doaemb",), ("initstate_emb",)])
@pytest.mark.parametrize("track", [False, True])
def test_spatial_features_forward_backward(features, track):
    model = TSE_Modular(**_args(spatial=features, track=track))
    estimate = model(_inputs(model))["target_estimate_spectra"]
    estimate.square().mean().backward()
    assert estimate.shape == (2, 33, 6, 2)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.requires_grad)


@pytest.mark.parametrize("features", [("tfmap", "context"), ("usef", "context"), ("context", "spkemb", "tfmap", "usef")])
def test_joint_cues_and_unique_parameter_owners(features):
    model = TSE_Modular(**_args(features, ("ipd", "cdf", "sdf", "delta_stft", "cyc_doaemb", "initstate_emb")))
    data = _inputs(model, (False, True))
    data["spatial_present"] = torch.tensor([True, False])
    model(data)["target_estimate_spectra"].square().mean().backward()
    named = list(model.named_parameters(remove_duplicate=False))
    assert len(named) == len({id(parameter) for _, parameter in named})
    assert all(parameter.grad is not None for _, parameter in named if parameter.requires_grad)
    assert all(not name.startswith("composer.separator") for name, _ in named)
    if "usef" in features:
        assert model.usef.usef_con2v[1].in_channels == 2
        assert model.separator.input_dim == 4 + 5 + 8 + int("tfmap" in features)


@pytest.mark.parametrize("fusion", ["additive", "concat", "multiply", "FiLM"])
def test_final_fusion_gate_is_identity_for_missing_rows(fusion):
    composer = CueComposer(feature_dim=8, nband=2, num_blocks=2, spkemb={"fusion": fusion}, embedding_dim=4, cyclic_fusion=fusion)
    latent = torch.randn(2, 2, 8, 5)
    presence = torch.tensor([True, False])
    result = composer.fuse_latent(latent, present=presence, embedding=torch.randn(2, 4))
    torch.testing.assert_close(result[1], latent[1], rtol=0, atol=0)
    for conditioner in composer.block_conditioners(presence):
        result = conditioner(latent, torch.randn(2, 1, 8, 5))
        torch.testing.assert_close(result[1], latent[1], rtol=0, atol=0)


def test_missing_direction_keeps_observation_features_and_ignores_placeholder():
    model = TSE_Modular(**_args(("tfmap",), ("ipd", "cdf", "sdf", "delta_stft"))).eval()
    data = _inputs(model)
    data["spatial_present"][:] = False
    captured = []
    hook = model.separator.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    first = model(data)["target_estimate_spectra"]
    data["spatial"][:] = float("nan")
    second = model(data)["target_estimate_spectra"]
    hook.remove()
    torch.testing.assert_close(first, second)
    assert captured[0][:, 4].abs().sum() > 0
    assert torch.count_nonzero(captured[0][:, 5:7]) == 0
    assert captured[0][:, 7:9].abs().sum() > 0


def test_missing_speaker_and_padding_do_not_pollute_bn_or_output():
    model = TSE_Modular(**_args(("context", "spkemb", "tfmap", "usef"), ("cdf",))).train()
    data = _inputs(model, (True, False))
    before = {name: value.clone() for name, value in model.speaker_encoder.named_buffers()}
    first = model(data)["target_estimate_spectra"]
    data["speaker_waveform"][1] = float("nan")
    data["enroll_spectra"][1] = float("nan")
    second = model(data)["target_estimate_spectra"]
    torch.testing.assert_close(first, second)
    for name, value in model.speaker_encoder.named_buffers():
        torch.testing.assert_close(before[name], value, rtol=0, atol=0)


def test_all_missing_is_explicit_and_listen_not_implemented():
    model = TSE_Modular(**_args(("tfmap",)))
    with pytest.raises(ValueError, match="all cues are missing"):
        model(_inputs(model, (False, False)))
    with pytest.raises(NotImplementedError):
        TSE_Modular(**_args(("listen",)))


@pytest.mark.parametrize("update,exception", [
    ({"mystery": 1}, TypeError), ({"ref_channel": 2}, ValueError),
    ({"mic_positions_m": [[0, 0, 0]]}, ValueError),
    ({"speed_of_sound_mps": 0}, ValueError),
    ({"separator": {"n_fft": 64}}, ValueError),
])
def test_constructor_rejects_unknown_and_conflicting_geometry(update, exception):
    args = _args(spatial=("cdf",))
    args.update(update)
    with pytest.raises(exception):
        TSE_Modular(**args)


def test_runtime_build_stft_forward_decode_backward_and_strict_reload():
    args = _args(("tfmap",), ("cdf",))
    stft_config = {key: args.pop(key) for key in ("n_fft", "hop_length", "win_length")}
    stft = STFT(**stft_config)
    model = build_model({"type": "TSE_Modular", "args": args}, stft_cfg=stft_config)
    cue = TSECueBatches(
        speaker_waveform=TSECueBatch("waveform", torch.randn(2, 160), torch.tensor([True, False]), torch.tensor([140, 0], dtype=torch.int32), None),
        spatial=TSECueBatch("static", torch.zeros(2, 2), torch.tensor([False, True]), None, None),
    )
    data = build_tse_model_inputs(model=model, stft=stft, mixture=torch.randn(2, 2, 256), sample_rate=16000, cues=cue)
    result = decode_tse_model_output(model(data), stft=stft, reference_length=256)
    assert result["target_estimate"].shape == (2, 256)
    result["target_estimate"].square().mean().backward()
    other = build_model({"type": "TSE_Modular", "args": args}, stft_cfg=stft_config)
    other.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    torch.testing.assert_close(other(data)["target_estimate_spectra"], result["target_estimate_spectra"])


def test_runtime_we_sep_speaker_tfmap_context_cue_batch_to_loss_backward():
    """验证 WeSep 风格 speaker cue 组合经过 runtime I/O 合同进入反向图。"""
    args = _args(("tfmap", "context"))
    args["cues"]["speaker"]["features"]["tfmap"] = {"type": "embedding"}
    args["cues"]["speaker"]["features"]["context"] = {
        "source": "tfmap_embedding",
        "injection": "latent",
        "attention_dim": 4,
        "num_heads": 1,
        "fusion": "multiply",
    }
    args["cues"]["speaker"]["encoder"] = {
        "fbank": {
            "num_mel_bins": 8,
            "frame_length_ms": 4.0,
            "frame_shift_ms": 1.0,
            "dither": 0.0,
        }
    }
    stft_config = {key: args.pop(key) for key in ("n_fft", "hop_length", "win_length")}
    stft = STFT(**stft_config)
    model = build_model({"type": "TSE_Modular", "args": args}, stft_cfg=stft_config).train()
    cue = TSECueBatches(
        speaker_waveform=TSECueBatch(
            "waveform",
            torch.randn(2, 256),
            torch.ones(2, dtype=torch.bool),
            torch.full((2,), 256, dtype=torch.int32),
            None,
        )
    )
    data = build_tse_model_inputs(
        model=model,
        stft=stft,
        mixture=torch.randn(2, 256),
        sample_rate=16000,
        cues=cue,
    )
    output = decode_tse_model_output(
        model(data), stft=stft, reference_length=256
    )["target_estimate"]
    output.square().mean().backward()
    assert output.shape == (2, 256)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_embedding_only_contract_uses_exact_runtime_fields():
    model = TSE_Modular(**_args(("spkemb",), embedding=True))
    assert {field.name for field in model.IO_CONTRACT.inputs} == {
        "mix_spectra", "speaker_embedding", "speaker_embedding_present",
    }
    model(_inputs(model))["target_estimate_spectra"].square().mean().backward()
    assert not hasattr(TSE_Modular, "IO_CONTRACT")
    assert TSE_Modular.supports_multichannel_input
    assert not model.supports_multichannel_input


def test_separator_is_the_only_public_parameter_name():
    args = _args(("tfmap",))
    args["sep"] = args.pop("separator")
    with pytest.raises(TypeError, match="unexpected keyword argument 'sep'"):
        TSE_Modular(**args)


@pytest.mark.parametrize("freeze", [None, 0, 1, "true", []])
def test_encoder_freeze_requires_exact_bool(freeze):
    args = _args(("context",))
    args["cues"]["speaker"]["encoder"] = {"freeze": freeze}
    with pytest.raises(TypeError, match="encoder.freeze must be a bool"):
        TSE_Modular(**args)


def test_freezing_without_checkpoint_is_rejected():
    args = _args(("context",))
    args["cues"]["speaker"]["encoder"] = {"freeze": True}
    with pytest.raises(ValueError, match="freeze=True requires checkpoint"):
        TSE_Modular(**args)


@pytest.mark.parametrize("freeze", [None, True, False])
def test_checkpoint_freeze_policy_preserves_gradients_and_bn(tmp_path, freeze):
    args = _args(("context", "spkemb"), missing=True)
    source = TSE_Modular(**args)
    checkpoint = tmp_path / "speaker.pt"
    torch.save(source.speaker_encoder.spk_model.state_dict(), checkpoint)
    try:
        options = {"checkpoint": str(checkpoint)}
        if freeze is not None:
            options["freeze"] = freeze
        args["cues"]["speaker"]["encoder"] = options
        model = TSE_Modular(**args).train()
        encoder = model.speaker_encoder
        frozen = freeze is not False
        assert encoder.freeze is frozen
        assert all(parameter.requires_grad is not frozen for parameter in encoder.parameters())
        assert all(not module.training for module in encoder.modules())
        before = {name: value.clone() for name, value in encoder.named_buffers()}
        for presence in ((True, False), (False, False)):
            model.zero_grad(set_to_none=True)
            data = _inputs(model, presence)
            context, embedding = model._speaker_encoding(data, data["speaker_waveform_present"])
            assert context[0].requires_grad is not frozen
            assert embedding.requires_grad is not frozen
            model(data)["target_estimate_spectra"].square().mean().backward()
            parameters = list(encoder.parameters())
            if frozen:
                assert all(parameter.grad is None for parameter in parameters)
            else:
                assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
                if any(presence):
                    assert any(torch.count_nonzero(parameter.grad) > 0 for parameter in parameters)
                else:
                    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in parameters)
            for name, value in encoder.named_buffers():
                torch.testing.assert_close(before[name], value, rtol=0, atol=0)
        reloaded = TSE_Modular(**args)
        reloaded.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(reloaded(data)["target_estimate_spectra"], model(data)["target_estimate_spectra"])
    finally:
        checkpoint.unlink()


def test_modular_speaker_encoder_loads_prefixed_full_tse_checkpoint(tmp_path):
    args = _args(("context",))
    source = TSE_Modular(**args)
    checkpoint = tmp_path / "full_tse.pt"
    torch.save(
        {
            f"spk_ft.encoder.spk_model.{name}": value
            for name, value in source.speaker_encoder.spk_model.state_dict().items()
        },
        checkpoint,
    )
    args["cues"]["speaker"]["encoder"] = {
        "checkpoint": str(checkpoint),
        "pretrained_prefix": "spk_ft.encoder.spk_model.",
        "freeze": True,
    }
    loaded = TSE_Modular(**args)
    assert loaded.speaker_encoder.freeze is True
    for name, value in source.speaker_encoder.spk_model.state_dict().items():
        torch.testing.assert_close(
            loaded.speaker_encoder.spk_model.state_dict()[name], value
        )


# --- textual：KCE 句级条件 ---------------------------------------------------


def test_textual_kce_forward_backward_freeze_and_bn(kce_checkpoint):
    model = TSE_Modular(**_textual_args(kce_checkpoint, fusion="additive")).train()
    kce = model.textual_frontend.dae_kce.kce
    buffers = {name: value.clone() for name, value in kce.model.named_buffers()}
    assert buffers
    assert all(not parameter.requires_grad for parameter in kce.model.parameters())

    estimate = _fused(model, _inputs(model))

    assert estimate.shape == (2, 33, 96, 2)
    assert kce.model.training is False
    assert model.textual_multi_fuse is True
    assert len(model.textual_frontend.dae_kce.fusions) == model.separator.num_blocks
    assert all(parameter.grad is None for parameter in kce.model.parameters())
    for module in (kce.adapter, model.textual_frontend.dae_kce.fusions):
        grads = _grads(module)
        assert grads
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(float(g.abs().sum()) > 0 for g in grads)
    for name, value in kce.model.named_buffers():
        torch.testing.assert_close(buffers[name], value, rtol=0, atol=0)
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_textual_cue_modulates_its_own_row_only(kce_checkpoint):
    model = TSE_Modular(
        **_textual_args(kce_checkpoint, fusion="additive", missing=True)
    ).train()
    data = _textual_inputs(present=(True, False))
    first = model(data)["target_estimate_spectra"]

    changed_tokens = _TEXTUAL_TOKENS.clone()
    changed_tokens[0] = torch.tensor([11, 12, 13, 0, 0])
    other = _textual_inputs(present=(True, False), tokens=changed_tokens)
    other["mix_spectra"] = data["mix_spectra"]
    other["mix_wav"][1] = torch.randn(16000)
    second = model(other)["target_estimate_spectra"]

    assert not torch.equal(first[0], second[0])
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)


@pytest.mark.parametrize(("multi_fuse", "expected"), [(True, 3), (False, 1)])
def test_textual_multi_fuse_stage_count_follows_separator_blocks(
    kce_checkpoint, multi_fuse, expected
):
    model = TSE_Modular(
        **_textual_args(
            kce_checkpoint, fusion="concat", multi_fuse=multi_fuse, num_repeat=3
        )
    ).train()
    fusions = model.textual_frontend.dae_kce.fusions

    assert model.textual_multi_fuse is multi_fuse
    assert model.separator.num_blocks == 3
    assert len(fusions) == expected

    _fused(model, _inputs(model))

    grads = _grads(fusions)
    assert grads
    assert all(g is not None and float(g.abs().sum()) > 0 for g in grads)


def test_textual_missing_checkpoint_and_wrong_sample_rate_fail_fast():
    args = _textual_args("unused")
    del args["cues"]["textual"]["features"]["dae_kce"]["checkpoint"]
    with pytest.raises(KCEWeightsError):
        TSE_Modular(**args)

    args = _textual_args("unused")
    args["sample_rate"] = 8000
    with pytest.raises(ValueError, match="sample_rate=16000"):
        TSE_Modular(**args)


def test_textual_contract_fields_and_required_roles(kce_checkpoint):
    model = TSE_Modular(**_textual_args(kce_checkpoint))

    assert {field.name for field in model.IO_CONTRACT.inputs} == {
        "mix_spectra", "mix_wav", "lengths", "textual", "textual_present",
        "textual_lengths",
    }

    data = _inputs(model)
    del data["mix_wav"]
    with pytest.raises(ValueError, match="missing required TSE_Modular inputs"):
        model(data)

    visual = TSECueBatches(
        visual=TSECueBatch(
            "features",
            torch.randn(2, 4, 3),
            torch.tensor([True, True]),
            torch.tensor([3, 3], dtype=torch.int32),
            torch.zeros(2, 3, dtype=torch.float64),
        )
    )
    with pytest.raises(ValueError, match="does not consume it"):
        build_tse_model_inputs(
            model=model,
            stft=STFT(n_fft=64, hop_length=16, win_length=64),
            mixture=torch.randn(2, 16000),
            sample_rate=16000,
            cues=visual,
        )


# --- visual：MuSE 帧级条件 --------------------------------------------------


def test_visual_frame_condition_is_aligned_to_the_audio_grid():
    model = TSE_Modular(
        **_visual_args(fusion="additive", multi_fuse=False, missing=True)
    ).eval()
    captured = []
    hook = model.composer.visual_fusion.register_forward_pre_hook(
        lambda _module, args: captured.append(args[1].detach().clone())
    )
    data = _visual_inputs(present=(True, False), frames=6)
    estimate = model(data)["target_estimate_spectra"]
    hook.remove()

    assert estimate.shape == (2, 33, 6, 2)
    # hop/sr = 0.001s：音频帧 0..5 → 视频帧索引 0,0,0,1,1,1
    assert captured[0].shape == (2, 1, 4, 6)
    torch.testing.assert_close(
        captured[0][0, 0, 0], torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]), rtol=0, atol=0
    )
    assert bool((captured[0][1] == 0).all())


def test_visual_forward_backward_and_missing_row_isolation():
    model = TSE_Modular(
        **_visual_args(fusion="additive", multi_fuse=False, missing=True)
    ).train()
    data = _visual_inputs(present=(True, False))
    estimate = model(data)["target_estimate_spectra"]
    estimate.square().mean().backward()

    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert any(
        float(parameter.grad.abs().sum()) > 0
        for parameter in model.composer.visual_fusion.parameters()
    )

    placeholder = _visual_inputs(present=(True, False))
    placeholder["mix_spectra"] = data["mix_spectra"]
    placeholder["visual"][1] = torch.randn(4, 3)
    torch.testing.assert_close(
        model(placeholder)["target_estimate_spectra"][1], estimate[1], rtol=0, atol=0
    )

    changed = _visual_inputs(present=(True, False))
    changed["mix_spectra"] = data["mix_spectra"]
    changed["visual"][0] = changed["visual"][0] + 1.0
    assert not torch.equal(model(changed)["target_estimate_spectra"][0], estimate[0])


def test_visual_missing_rows_do_not_pollute_adapter_bn():
    adapter = {"type": "muse", "input_dim": 4, "output_dim": 4, "num_layers": 1}
    model = TSE_Modular(
        **_visual_args(adapter=adapter, fusion="additive", missing=True)
    ).train()
    frontend = model.visual_frontend
    assert frontend.adapter.training is False
    before = {name: value.clone() for name, value in frontend.adapter.named_buffers()}
    assert before

    model(_visual_inputs(present=(True, False)))

    for name, value in frontend.adapter.named_buffers():
        torch.testing.assert_close(before[name], value, rtol=0, atol=0)


def test_visual_multi_fuse_hooks_every_separator_block():
    model = TSE_Modular(
        **_visual_args(fusion="concat", multi_fuse=True, num_repeat=3)
    ).train()
    stages = model.composer.visual_stages

    assert model.visual_multi_fuse is True
    assert model.separator.num_blocks == 3
    assert len(stages) == 3
    assert not hasattr(model.composer, "visual_fusion")

    _fused(model, _visual_inputs())

    grads = _grads(stages)
    assert grads
    assert all(g is not None and float(g.abs().sum()) > 0 for g in grads)


def test_visual_single_fuse_uses_the_latent_level_hook():
    model = TSE_Modular(**_visual_args(fusion="concat", multi_fuse=False)).train()

    assert model.visual_multi_fuse is False
    assert not hasattr(model.composer, "visual_stages")

    _fused(model, _visual_inputs())

    grads = _grads(model.composer.visual_fusion)
    assert grads
    assert all(g is not None and float(g.abs().sum()) > 0 for g in grads)


def test_visual_contract_fields_roles_and_raw_video_gate():
    model = TSE_Modular(**_visual_args())

    assert {field.name for field in model.IO_CONTRACT.inputs} == {
        "mix_spectra", "visual", "visual_present", "visual_lengths", "visual_timebase",
    }

    data = _visual_inputs()
    del data["visual"]
    with pytest.raises(ValueError, match="missing required TSE_Modular inputs"):
        model(data)

    raw = _visual_args()
    raw["cues"]["visual"]["representation"] = "raw"
    with pytest.raises(NotImplementedError, match="raw video"):
        TSE_Modular(**raw)

    textual = TSECueBatches(
        textual=TSECueBatch(
            "phoneme_ids",
            _TEXTUAL_TOKENS,
            torch.tensor([True, True]),
            torch.tensor([5, 4], dtype=torch.int32),
            None,
        )
    )
    with pytest.raises(ValueError, match="does not consume it"):
        build_tse_model_inputs(
            model=model,
            stft=STFT(n_fft=64, hop_length=16, win_length=64),
            mixture=torch.randn(2, 256),
            sample_rate=16000,
            cues=textual,
        )


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (
            lambda data: data.update(
                visual_timebase=torch.zeros(2, 4, dtype=torch.float64)
            ),
            ValueError,
            "aligned with visual",
        ),
        (
            lambda data: data.update(
                visual_timebase=torch.zeros(2, 3, dtype=torch.int64)
            ),
            ValueError,
            "floating",
        ),
        (
            lambda data: data.update(visual_lengths=torch.tensor([4, 2], dtype=torch.int32)),
            ValueError,
            "visual_lengths exceeds capacity",
        ),
        (
            lambda data: data.update(visual_present=torch.tensor([True, False])),
            ValueError,
            "zero exactly for missing rows",
        ),
    ],
)
def test_visual_timebase_lengths_and_presence_are_validated(mutate, error, match):
    model = TSE_Modular(**_visual_args(missing=True))
    data = _visual_inputs()
    mutate(data)

    with pytest.raises(error, match=match):
        model(data)


def test_visual_all_missing_requires_the_explicit_research_opt_in():
    model = TSE_Modular(**_visual_args())
    data = _visual_inputs()
    data.update(
        visual_present=torch.tensor([False, False]),
        visual_lengths=torch.zeros(2, dtype=torch.int32),
        visual_timebase=torch.zeros(2, 3, dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="all cues are missing"):
        model(data)


# --- composer 的 block 级注入 hook ------------------------------------------


def test_block_injection_composes_only_one_cue_source():
    composer = CueComposer(
        feature_dim=8, nband=2, num_blocks=2, cyclic_fusion="concat",
        visual={"fusion": "concat", "multi_fuse": True}, visual_dim=4,
    )
    present = torch.tensor([True, True])
    spatial = (torch.randn(2, 1, 8, 5), present)
    visual = (torch.randn(2, 4, 5), present)

    assert CueComposer(feature_dim=8, nband=2, num_blocks=2).block_injection() == (
        None, None
    )
    with pytest.raises(ValueError, match="only one per-block cue source"):
        composer.block_injection(spatial=spatial, visual=visual)

    conditioners, conditions = composer.block_injection(visual=visual)
    assert len(conditioners) == 2
    assert conditions.shape == (2, 1, 4, 5)

    with pytest.raises(ValueError, match="fusion layers"):
        composer.block_injection(
            textual_stages=((composer.visual_stages[0],), torch.randn(2, 6), present)
        )

    single = CueComposer(
        feature_dim=8, nband=2, num_blocks=2, visual={"multi_fuse": False}, visual_dim=4
    )
    with pytest.raises(ValueError, match="multi_fuse=True"):
        single.block_injection(visual=visual)


def test_block_injection_and_fuse_latent_keep_missing_rows_unfused():
    composer = CueComposer(
        feature_dim=8, nband=2, num_blocks=1,
        visual={"fusion": "concat", "multi_fuse": True}, visual_dim=4,
    )
    present = torch.tensor([True, False])
    latent = torch.randn(2, 2, 8, 5)
    condition = torch.randn(2, 4, 5)
    conditioners, conditions = composer.block_injection(visual=(condition, present))

    fused = conditioners[0](latent, conditions)

    assert not torch.equal(fused[0], latent[0])
    torch.testing.assert_close(fused[1], latent[1], rtol=0, atol=0)
    torch.testing.assert_close(
        fused,
        gate_fusion(
            latent, CueComposer._fuse(composer.visual_stages[0], latent, conditions), present
        ),
        rtol=0,
        atol=0,
    )

    single = CueComposer(
        feature_dim=8, nband=2, num_blocks=1,
        visual={"fusion": "concat", "multi_fuse": False}, visual_dim=4,
    )
    at_latent = single.fuse_latent(latent, visual=(condition, present))
    assert not torch.equal(at_latent[0], latent[0])
    torch.testing.assert_close(at_latent[1], latent[1], rtol=0, atol=0)
    with pytest.raises(ValueError, match="per-block fusion"):
        composer.fuse_latent(latent, visual=(condition, present))


# --- registry / build --------------------------------------------------------


def test_registry_registers_modular_for_the_new_cue_contracts():
    from core.utils.registry import MODEL_REGISTRY
    from tasks.tse.models import TSE_MODEL_TYPES, register_tse_models

    register_tse_models(only="TSE_Modular")

    assert "TSE_Modular" in TSE_MODEL_TYPES
    assert MODEL_REGISTRY.get("TSE_Modular") is TSE_Modular


@pytest.mark.parametrize("cue", ["textual", "visual"])
def test_runtime_build_and_strict_reload_for_textual_and_visual(kce_checkpoint, cue):
    args = _textual_args(kce_checkpoint) if cue == "textual" else _visual_args()
    stft_config = {key: args.pop(key) for key in ("n_fft", "hop_length", "win_length")}
    stft = STFT(**stft_config)
    model = build_model({"type": "TSE_Modular", "args": args}, stft_cfg=stft_config)
    if cue == "textual":
        cues = TSECueBatches(
            textual=TSECueBatch(
                "phoneme_ids",
                _TEXTUAL_TOKENS,
                torch.tensor([True, True]),
                torch.tensor([5, 4], dtype=torch.int32),
                None,
            )
        )
        mixture = torch.randn(2, 16000)
    else:
        cues = TSECueBatches(
            visual=TSECueBatch(
                "features",
                _VISUAL_VALUE,
                torch.tensor([True, True]),
                torch.tensor([3, 2], dtype=torch.int32),
                torch.tensor(
                    [[0.0, 0.003, 0.006], [0.0, 0.003, 0.0]], dtype=torch.float64
                ),
            )
        )
        mixture = torch.randn(2, 256)

    data = build_tse_model_inputs(
        model=model, stft=stft, mixture=mixture, sample_rate=16000, cues=cues
    )
    result = decode_tse_model_output(
        model(data), stft=stft, reference_length=mixture.shape[-1]
    )

    assert result["target_estimate"].shape == (2, mixture.shape[-1])
    result["target_estimate"].square().mean().backward()

    other = build_model({"type": "TSE_Modular", "args": args}, stft_cfg=stft_config)
    other.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    torch.testing.assert_close(
        other(data)["target_estimate_spectra"], result["target_estimate_spectra"]
    )


def test_speaker_context_block_fusion_runs_each_separator_block():
    args = _args(("context",))
    args["cues"]["speaker"]["features"]["context"].update(
        {"injection": "block", "attention_dim": 4, "num_heads": 1, "fusion": "multiply"}
    )
    model = TSE_Modular(**args).train()
    data = _inputs(model)
    output = model(data)["target_estimate_spectra"]
    output.square().mean().backward()
    assert output.shape == data["mix_spectra"].shape
    stages = model.composer.context_stages
    assert len(stages) == model.separator.num_blocks
    assert all(
        any(parameter.grad is not None for parameter in stage.parameters())
        for stage in stages
    )
