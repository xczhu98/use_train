"""PASE family checkpoint 与 UniPASE 推理构造契约。"""

from pathlib import Path

import pytest
import torch
from torch import nn

import inference
from evaluate import checkpoint_loading
from evaluate.inference_components import configured_checkpoint_files
from tasks.se.models.pase_family.common import utils as common_utils
from tasks.se.models.pase_family.unipase import unipase_model

pytestmark = [pytest.mark.unit]


class _TinyArtifact(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


def test_artifact_loader_expands_user_and_uses_safe_loader(monkeypatch):
    captured = {}

    def fake_safe_load(path, *, map_location):
        captured["path"] = path
        captured["map_location"] = map_location
        return {"cfg": {}, "model": {"weight": torch.ones(1)}}

    monkeypatch.setattr(common_utils, "safe_torch_load", fake_safe_load)
    model = common_utils.load_artifact_module(
        _TinyArtifact, "~/checkpoints/artifact.pt", frozen=True
    )

    assert captured == {
        "path": Path.home() / "checkpoints" / "artifact.pt",
        "map_location": "cpu",
    }
    assert model.weight.item() == pytest.approx(1.0)
    assert model.training is False
    assert model.weight.requires_grad is False


@pytest.mark.parametrize(
    "relative_path",
    [
        "tasks/se/models/pase_family/common/utils.py",
        "tasks/se/models/pase_family/common/wavlm_frontend.py",
        "tasks/se/models/pase_family/common/vocoder_backend.py",
        "tasks/se/models/pase_family/stupase/flow/cfm.py",
        "tasks/se/models/pase_family/stupase/vocos_vocoder.py",
    ],
)
def test_pase_checkpoint_modules_have_no_bare_torch_load(relative_path):
    source = (Path(__file__).resolve().parent.parent / relative_path).read_text(
        encoding="utf-8"
    )
    assert "torch.load(" not in source
    # 直接走 safe_torch_load，或委托统一 artifact loader（其内部使用 safe_torch_load）
    assert "safe_torch_load" in source or "load_artifact_module" in source


def test_artifact_loader_fails_closed_on_missing_cfg_key(monkeypatch):
    monkeypatch.setattr(
        common_utils, "safe_torch_load", lambda path, *, map_location: {"model": {}}
    )
    with pytest.raises(RuntimeError, match="缺少 'cfg'"):
        common_utils.load_artifact_module(_TinyArtifact, "~/checkpoints/artifact.pt")


def test_stupase_artifact_loaders_use_the_unified_loader(monkeypatch):
    """StuPASE CFM/Vocos 加载器统一走 load_artifact_module：缺 key 与多余 key 都 fail closed。"""
    from tasks.se.models.pase_family.stupase.flow.cfm import CFM
    from tasks.se.models.pase_family.stupase.vocos_vocoder import VocosVocoder

    vocoder_cfg = dict(
        input_channels=32,
        dim=32,
        intermediate_dim=64,
        num_layers=1,
        num_res=2,
        num_attn=1,
        n_fft=16,
        hop_length=4,
    )
    vocoder_state = VocosVocoder(**vocoder_cfg).state_dict()
    cfm_cfg = dict(
        dit_kwargs=dict(
            dim=16, depth=1, heads=2, dim_head=8, ff_mult=2, dropout=0.0,
            mel_dim=4, text_in_dim=4, text_dim=4,
        ),
        mel_spec_kwargs=dict(
            n_fft=32, hop_length=8, win_length=32, n_mel_channels=8, target_sample_rate=16000
        ),
    )
    cfm_state = CFM(**cfm_cfg).state_dict()

    monkeypatch.setattr(
        common_utils,
        "safe_torch_load",
        lambda path, *, map_location: {"cfg": vocoder_cfg, "model": vocoder_state},
    )
    vocoder = VocosVocoder.from_pretrained("~/Vocoder_Mel-16k.pt")
    assert vocoder.training is False
    assert all(not p.requires_grad for p in vocoder.parameters())

    monkeypatch.setattr(
        common_utils,
        "safe_torch_load",
        lambda path, *, map_location: {"cfg": cfm_cfg, "generator": cfm_state},
    )
    assert CFM.from_pretrained("~/CFM.pt").training is False

    # 缺 "cfg"/"model" 等 key 时给 fail-closed 文案，而不是裸 KeyError
    monkeypatch.setattr(
        common_utils, "safe_torch_load", lambda path, *, map_location: {"cfg": vocoder_cfg}
    )
    with pytest.raises(RuntimeError, match="'generator' 也无 'model'"):
        VocosVocoder.from_pretrained("~/Vocoder_Mel-16k.pt")

    # 多余 key 即报错（strict=False 也要 fail closed）
    monkeypatch.setattr(
        common_utils,
        "safe_torch_load",
        lambda path, *, map_location: {
            "cfg": vocoder_cfg,
            "model": {**vocoder_state, "unexpected.weight": torch.zeros(1)},
        },
    )
    with pytest.raises(RuntimeError, match="state_dict 不匹配"):
        VocosVocoder.from_pretrained("~/Vocoder_Mel-16k.pt")


def _patch_unipase_components(monkeypatch, postnet_calls):
    monkeypatch.setattr(
        unipase_model,
        "WavLM_feat_PLC",
        lambda **kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        unipase_model.VocosAdapter,
        "from_pretrained",
        lambda path: nn.Identity(),
    )
    monkeypatch.setattr(
        unipase_model.VocosVocoder,
        "from_pretrained",
        lambda path: nn.Identity(),
    )

    def fake_postnet(path):
        postnet_calls.append(path)
        return nn.Identity()

    monkeypatch.setattr(unipase_model.TFGridNet, "from_pretrained", fake_postnet)


def test_unipase_16k_does_not_load_postnet(monkeypatch):
    postnet_calls = []
    _patch_unipase_components(monkeypatch, postnet_calls)

    model = unipase_model.UniPASE(
        dewavlm_ckpt_path="dewavlm.pt",
        adapter_ckpt_path="adapter.pt",
        vocoder_ckpt_path="vocoder.pt",
        postnet_ckpt_path="postnet.pt",
        sr_out=16000,
    )

    assert model.postnet is None
    assert postnet_calls == []


def test_unipase_48k_requires_explicit_postnet_checkpoint():
    with pytest.raises(ValueError, match="postnet_ckpt_path"):
        unipase_model.UniPASE(sr_out=48000)


@pytest.mark.parametrize("sr_out", [True, 16000.5, "16000"])
def test_unipase_rejects_non_integer_sr_out(sr_out):
    with pytest.raises(TypeError, match="sr_out must be a positive integer"):
        unipase_model.UniPASE(sr_out=sr_out)


@pytest.mark.parametrize("sr_out", [0, -1])
def test_unipase_rejects_nonpositive_sr_out(sr_out):
    with pytest.raises(ValueError, match="sr_out must be > 0"):
        unipase_model.UniPASE(sr_out=sr_out)


def test_unipase_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="unexpected keyword arguments.*typo"):
        unipase_model.UniPASE(typo=True)


def test_configured_checkpoint_files_expand_user():
    files = configured_checkpoint_files(
        {"args": {"encoder_ckpt_path": "~/checkpoints/encoder.pt"}}
    )
    assert files == {
        "config.args.encoder_ckpt_path": Path.home()
        / "checkpoints"
        / "encoder.pt"
    }


def test_load_model_uses_fd_bound_safe_loader(monkeypatch, tmp_path):
    loaded = {}
    model = nn.Linear(1, 1)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    monkeypatch.setattr(
        "core.models.build.build_model", lambda cfg, stft_cfg=None: model
    )

    def fake_safe_load(stream, **kwargs):
        loaded["stream_is_readable"] = stream.readable()
        loaded["stream_is_seekable"] = stream.seekable()
        loaded["kwargs"] = kwargs
        return {"model": model.state_dict()}

    monkeypatch.setattr(
        checkpoint_loading.safe_load_module, "safe_torch_load", fake_safe_load
    )
    loaded_model = inference._load_model(
        {"model": {"type": "GTCRN"}}, checkpoint, torch.device("cpu")
    )

    assert loaded_model is model
    assert loaded["stream_is_readable"] is True
    assert loaded["stream_is_seekable"] is True
    assert loaded["kwargs"] == {
        "map_location": "cpu",
        "purpose": "inference model checkpoint (safe mode)",
    }


def test_load_model_rejects_missing_checkpoint_before_unsafe_torch_load(monkeypatch, tmp_path):
    model = nn.Linear(1, 1)
    checkpoint = tmp_path / "missing.pt"
    monkeypatch.setattr(
        "core.models.build.build_model", lambda cfg, stft_cfg=None: model
    )

    def fail_safe_load(*args, **kwargs):
        raise AssertionError("safe loader must not receive an unopened checkpoint")

    def fail_torch_load(*args, **kwargs):
        raise AssertionError("inference must not call torch.load directly")

    monkeypatch.setattr(
        checkpoint_loading.safe_load_module, "safe_torch_load", fail_safe_load
    )
    monkeypatch.setattr(inference.torch, "load", fail_torch_load)

    with pytest.raises(RuntimeError, match="无法安全打开 checkpoint"):
        inference._load_model(
            {"model": {"type": "GTCRN"}}, checkpoint, torch.device("cpu")
        )


def test_wavlm_conv_feature_layers_rejects_code_execution():
    """checkpoint 的 conv_feature_layers 只允许受限表达式，禁止 eval 任意代码。"""
    from tasks.se.models.pase_family.common.wavlm.WavLM import parse_conv_feature_layers

    parsed = parse_conv_feature_layers("[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2")
    assert len(parsed) == 7
    assert parsed[-1] == (512, 2, 2)

    for hostile in (
        "__import__('os').system('true')",
        "[open('/etc/passwd').read()]",
        "globals()",
    ):
        with pytest.raises(ValueError):
            parse_conv_feature_layers(hostile)

    with pytest.raises(ValueError):
        parse_conv_feature_layers([(512, 10), (512, 2, 2)])
