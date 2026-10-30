"""Tests for core.losses.build.build_loss STFT default injection."""

import copy
import logging
import subprocess
import sys

import pytest
import torch

from core.losses.build import _inject_sample_rate
from core.stft import build_stft_istft
from core.utils.registry import LOSS_REGISTRY
from core.losses import build_loss


@LOSS_REGISTRY.register()
class _StftConfigProbeLoss(torch.nn.Module):
    def __init__(
        self,
        n_fft=None,
        hop_length=None,
        win_length=None,
        window=None,
        center=None,
        normalized=None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.normalized = normalized


@LOSS_REGISTRY.register()
class _MatchingContextProbeLoss(_StftConfigProbeLoss):
    REQUIRES_MATCHING_STFT_CONTEXT = True

    def __init__(
        self,
        sample_rate=48000,
        n_fft=None,
        hop_length=None,
        win_length=None,
        window=None,
        center=None,
        normalized=None,
    ):
        super().__init__(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            normalized=normalized,
        )
        self.sample_rate = sample_rate


def test_inject_sample_rate_uses_fs_alias_without_registry_registration():
    class _FsOnlyLoss:
        def __init__(self, fs):
            self.fs = fs

    args = _inject_sample_rate({}, 32000, _FsOnlyLoss)

    assert args["fs"] == 32000


def test_inject_sample_rate_does_not_use_var_keyword_as_sampling_rate_contract():
    class _KwargsOnlyLoss:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    assert _inject_sample_rate({}, 32000, _KwargsOnlyLoss) == {}


def test_build_loss_injects_same_stft_context_into_gtcrn_pair():
    cfg = {
        "type": "CompositeLoss",
        "components": [
            {
                "type": "GTCRNCompressedSpectralLoss",
                "weight": 1.0,
                "args": {"compress_factor": 0.3},
            },
            {"type": "GTCRNLogRatioLoss", "weight": 1.0},
            {
                "type": "FWNegSISDRLoss",
                "weight": 0.2,
                "args": {"n_bins": 18, "scale": "mel", "weights_type": "log_snr_speech_norm"},
            },
            {"type": "StateCrossEntropyLoss", "weight": 0.001},
        ],
    }
    original = copy.deepcopy(cfg)
    stft_cfg = {
        "n_fft": 640,
        "hop_length": 320,
        "win_length": 640,
        "window": "hann",
        "center": True,
        "normalized": False,
    }
    loss = build_loss(cfg, stft_cfg=stft_cfg)

    assert cfg == original
    spectral, ratio, fwsdr, _ = loss.losses
    for component in (spectral, ratio):
        assert (component.n_fft, component.hop_length, component.win_length) == (
            640,
            320,
            640,
        )
        assert component.window_name == "hann"
        assert component.center is True
        assert component.normalized is False
    assert fwsdr.n_fft == 640
    assert fwsdr.hop_length == 320


@pytest.mark.parametrize(
    ("n_fft", "hop_length", "win_length"),
    [(320, 160, 320), (640, 320, 640), (2048, 512, 2048)],
)
def test_build_loss_gtcrn_pair_supports_non_512_stft_geometry(
    n_fft, hop_length, win_length
):
    loss = build_loss(
        {
            "type": "CompositeLoss",
            "components": [
                {"type": "GTCRNCompressedSpectralLoss", "weight": 1.0},
                {"type": "GTCRNLogRatioLoss", "weight": 1.0},
            ],
        },
        stft_cfg={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "window": "hann",
            "center": True,
            "normalized": False,
        },
    )

    for component in loss.losses:
        assert (component.n_fft, component.hop_length, component.win_length) == (
            n_fft,
            hop_length,
            win_length,
        )
        assert component.window_name == "hann"
        assert component.center is True
        assert component.normalized is False


@pytest.mark.parametrize(
    ("stft_cfg", "window_name", "center", "normalized"),
    [
        (
            {
                "n_fft": 640,
                "hop_length": 320,
                "win_length": 640,
                "window": "hamming",
                "center": True,
                "normalized": False,
            },
            "hamming",
            True,
            False,
        ),
        (
            {
                "n_fft": 640,
                "hop_length": 320,
                "win_length": 640,
                "window": "hamming",
                "center": False,
                "normalized": False,
            },
            "hamming",
            False,
            False,
        ),
        (
            {
                "n_fft": 640,
                "hop_length": 320,
                "win_length": 640,
                "window": "hann",
                "center": True,
                "normalized": True,
            },
            "hann",
            True,
            True,
        ),
    ],
    ids=("hamming", "center-false-hamming", "normalized-true"),
)
def test_build_loss_direct_gtcrn_ratio_accepts_extended_stft_context(
    stft_cfg, window_name, center, normalized
):
    """hamming、center=False（框架允许时）与 normalized=True 均为合法继承。"""
    loss = build_loss({"type": "GTCRNLogRatioLoss"}, stft_cfg=stft_cfg)

    assert (loss.n_fft, loss.hop_length, loss.win_length) == (640, 320, 640)
    assert loss.window_name == window_name
    assert loss.center is center
    assert loss.normalized is normalized


@pytest.mark.parametrize(
    ("stft_cfg", "match"),
    [
        (
            {
                "n_fft": 640,
                "hop_length": 320,
                "win_length": 640,
                "window": "hann",
                "center": False,
            },
            "torch.istft NOLA",
        ),
        (
            {"n_fft": 640, "hop_length": 480, "win_length": 320},
            r"hop_length <= win_length <= n_fft",
        ),
        (
            {"n_fft": 320, "hop_length": 160, "win_length": 640},
            r"hop_length <= win_length <= n_fft",
        ),
    ],
    ids=("hann-center-false", "hop-exceeds-win", "win-exceeds-nfft"),
)
def test_build_loss_direct_gtcrn_ratio_rejects_invalid_stft_geometry(stft_cfg, match):
    """hann+center=False、hop>win、win>n_fft 仍在构造期 fail-fast。"""
    with pytest.raises(ValueError, match=match):
        build_loss({"type": "GTCRNLogRatioLoss"}, stft_cfg=stft_cfg)


def test_build_loss_gtcrn_rejects_explicit_args_conflicting_with_stft_context():
    """GTCRN 组件声明 REQUIRES_MATCHING_STFT_CONTEXT：显式几何与上下文冲突时 fail-fast。"""
    cfg = {
        "type": "CompositeLoss",
        "components": [
            {
                "type": "GTCRNCompressedSpectralLoss",
                "weight": 1.0,
                "args": {"n_fft": 400, "hop_length": 200, "win_length": 400},
            },
        ],
    }
    with pytest.raises(
        ValueError, match=r"GTCRNCompressedSpectralLoss.*context conflict"
    ):
        build_loss(cfg, stft_cfg={"n_fft": 512, "hop_length": 256, "win_length": 512})


@pytest.mark.parametrize(
    ("loss_type", "extra_context"),
    [
        ("_StftConfigProbeLoss", {}),
        ("MultiBandRelativeContrastiveLoss", {"sample_rate": 16000}),
    ],
)
def test_build_loss_injects_effective_model_stft_defaults(loss_type, extra_context):
    stft_cfg = {"n_fft": 512, "hop_length": 128}
    model_stft = build_stft_istft(stft_cfg)
    loss = build_loss({"type": loss_type}, stft_cfg=stft_cfg, **extra_context)

    assert loss.n_fft == model_stft.n_fft
    assert loss.hop_length == model_stft.hop_length
    assert loss.win_length == model_stft.win_length
    loss_window = loss.window_name if hasattr(loss, "window_name") else loss.window
    assert loss_window == model_stft.window_name
    assert loss.center is model_stft.center


@pytest.mark.parametrize(
    ("loss_type", "extra_context"),
    [
        ("_StftConfigProbeLoss", {}),
        ("MultiBandRelativeContrastiveLoss", {"sample_rate": 16000}),
    ],
)
def test_build_loss_injects_explicit_model_stft_values(loss_type, extra_context):
    stft_cfg = {
        "n_fft": 640,
        "hop_length": 160,
        "win_length": 640,
        "window": "hamming",
        "center": False,
    }
    model_stft = build_stft_istft(stft_cfg)
    loss = build_loss(
        {"type": loss_type}, stft_cfg=stft_cfg, **extra_context
    )

    assert (loss.n_fft, loss.hop_length, loss.win_length) == (
        model_stft.n_fft,
        model_stft.hop_length,
        model_stft.win_length,
    )
    loss_window = loss.window_name if hasattr(loss, "window_name") else loss.window
    assert loss_window == model_stft.window_name
    assert loss.center is model_stft.center


def test_build_loss_generic_stft_probe_keeps_explicit_override_semantics():
    loss = build_loss(
        {
            "type": "_StftConfigProbeLoss",
            "args": {
                "n_fft": 400,
                "hop_length": 100,
                "win_length": 400,
                "window": "hann",
                "center": True,
            },
        },
        stft_cfg={
            "n_fft": 640,
            "hop_length": 160,
            "win_length": 640,
            "window": "hamming",
            "center": False,
        },
    )

    assert (loss.n_fft, loss.hop_length, loss.win_length) == (400, 100, 400)
    assert loss.window == "hann"
    assert loss.center is True


def test_build_loss_consumes_matching_context_capability_without_class_special_case():
    loss = build_loss(
        {"type": "_MatchingContextProbeLoss"},
        stft_cfg={"n_fft": 640, "hop_length": 160},
        sample_rate=32000,
    )

    assert loss.sample_rate == 32000
    assert (loss.n_fft, loss.hop_length, loss.win_length) == (640, 160, 640)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("sample_rate", 16000),
        ("n_fft", 400),
        ("hop_length", 100),
        ("win_length", 400),
        ("window", "hann"),
        ("center", True),
    ],
)
def test_build_loss_matching_context_capability_rejects_conflicts(key, value):
    with pytest.raises(
        ValueError,
        match=rf"_MatchingContextProbeLoss.*{key}.*conflict",
    ):
        build_loss(
            {"type": "_MatchingContextProbeLoss", "args": {key: value}},
            stft_cfg={
                "n_fft": 640,
                "hop_length": 160,
                "win_length": 640,
                "window": "hamming",
                "center": False,
            },
            sample_rate=32000,
        )


def test_build_loss_composite_consumes_matching_context_capability():
    with pytest.raises(
        ValueError,
        match=r"_MatchingContextProbeLoss.*hop_length.*conflict",
    ):
        build_loss(
            {
                "type": "CompositeLoss",
                "components": [
                    {
                        "type": "_MatchingContextProbeLoss",
                        "args": {"hop_length": 80},
                    }
                ],
            },
            stft_cfg={"n_fft": 320, "hop_length": 160},
            sample_rate=32000,
        )


def test_build_loss_matching_context_capability_accepts_matching_explicit_args():
    loss = build_loss(
        {
            "type": "_MatchingContextProbeLoss",
            "args": {
                "n_fft": 640,
                "hop_length": 160,
                "win_length": 640,
                "window": "hamming",
                "center": False,
                "sample_rate": 32000,
            },
        },
        stft_cfg={
            "n_fft": 640,
            "hop_length": 160,
            "win_length": 640,
            "window": "hamming",
            "center": False,
        },
        sample_rate=32000,
    )

    assert loss.sample_rate == 32000
    assert (loss.n_fft, loss.hop_length, loss.win_length) == (640, 160, 640)
    assert loss.window == "hamming"
    assert loss.center is False


def test_build_loss_without_stft_cfg_uses_class_defaults():
    cfg = {
        "type": "CompositeLoss",
        "components": [
            {"type": "GTCRNCompressedSpectralLoss", "weight": 1.0},
        ],
    }
    loss = build_loss(cfg)
    spectral = loss.losses[0]
    assert spectral.n_fft == 512
    assert spectral.hop_length == 256
    assert spectral.win_length == 512


def test_build_loss_warns_on_explicit_center_mismatch(caplog):
    """未声明 REQUIRES_MATCHING_STFT_CONTEXT 的组件显式 center 失配时仅告警并保留。"""
    cfg = {
        "type": "CompositeLoss",
        "components": [
            {"type": "_StftConfigProbeLoss", "weight": 1.0, "args": {"center": False}},
        ],
    }
    with caplog.at_level(logging.WARNING, logger="core.losses.build"):
        loss = build_loss(cfg, stft_cfg={"n_fft": 512, "hop_length": 256, "win_length": 512})
    assert "_StftConfigProbeLoss explicitly uses center=False" in caplog.text
    assert loss.losses[0].center is False


@pytest.mark.parametrize(
    ("unknown_key", "suggestion"),
    [("component", "components"), ("enable_ss_undesired_energi", "enable_ss_undesired_energy")],
)
def test_build_loss_rejects_unknown_composite_top_level_key_with_hint(
    unknown_key, suggestion
):
    cfg = {
        "type": "CompositeLoss",
        "components": [{"type": "SISNRLoss"}],
        unknown_key: True,
    }

    with pytest.raises(KeyError, match=rf"{unknown_key!r}.*did you mean {suggestion!r}"):
        build_loss(cfg)


def test_build_loss_rejects_unknown_component_key_with_hint():
    cfg = {
        "type": "CompositeLoss",
        "components": [{"type": "SISNRLoss", "weights": 1.0}],
    }

    with pytest.raises(KeyError, match=r"components\[0\].*'weights'.*did you mean 'weight'"):
        build_loss(cfg)


def test_build_loss_rejects_unknown_single_loss_top_level_key_with_hint():
    cfg = {"type": "SISNRLoss", "arg": {}}

    with pytest.raises(KeyError, match=r"Single loss config.*'arg'.*did you mean 'args'"):
        build_loss(cfg)


@pytest.mark.parametrize("args", [[], "zero_mean=true", 1, False])
def test_build_loss_requires_single_loss_args_dict(args):
    with pytest.raises(TypeError, match=r"Single loss config\.args must be a dict or null"):
        build_loss({"type": "SISNRLoss", "args": args})


@pytest.mark.parametrize(
    "cfg",
    [
        {"type": "SISNRLoss", "args": None},
        {
            "type": "CompositeLoss",
            "components": [{"type": "SISNRLoss", "args": None}],
        },
    ],
)
def test_build_loss_keeps_legacy_null_args_compatibility(cfg):
    loss = build_loss(cfg)
    inner = loss.losses[0] if hasattr(loss, "losses") else loss
    assert inner.__class__.__name__ == "SISNRLoss"


@pytest.mark.parametrize("stft_cfg", [[], "n_fft=512", 512, False])
def test_build_loss_requires_stft_context_dict(stft_cfg):
    with pytest.raises(TypeError, match="stft_cfg must be dict or null"):
        build_loss({"type": "SISNRLoss"}, stft_cfg=stft_cfg)


def test_build_loss_rejects_unknown_stft_context_key_with_hint():
    with pytest.raises(KeyError, match=r"stft_cfg.*'n_fftt'.*did you mean 'n_fft'"):
        build_loss({"type": "SISNRLoss"}, stft_cfg={"n_fftt": 512})


@pytest.mark.parametrize("components", ["not-a-list", {"type": "SISNRLoss"}, ("SISNRLoss",)])
def test_build_loss_requires_components_list(components):
    with pytest.raises(TypeError, match="CompositeLoss 'components' must be a list"):
        build_loss({"type": "CompositeLoss", "components": components})


def test_build_loss_requires_component_dict():
    with pytest.raises(TypeError, match=r"components\[0\] must be a dict"):
        build_loss({"type": "CompositeLoss", "components": ["SISNRLoss"]})


@pytest.mark.parametrize("args", [[], "n_fft=512", 512])
def test_build_loss_requires_component_args_dict(args):
    cfg = {
        "type": "CompositeLoss",
        "components": [{"type": "SISNRLoss", "args": args}],
    }

    with pytest.raises(TypeError, match=r"components\[0\]\.args must be a dict or null"):
        build_loss(cfg)


def test_build_loss_injects_sample_rate_into_composite_fw_loss():
    cfg = {
        "type": "CompositeLoss",
        "components": [
            {
                "type": "FWNegSISDRLoss",
                "weight": 1.0,
                "args": {"n_bins": 18, "scale": "mel"},
            },
        ],
    }
    loss = build_loss(cfg, sample_rate=32000)
    assert loss.losses[0].sr == 32000


def test_build_loss_injects_sample_rate_into_single_fw_loss():
    loss = build_loss(
        {"type": "FWNegSISDRLoss", "args": {"n_bins": 18, "scale": "mel"}},
        sample_rate=32000,
    )
    assert loss.sr == 32000


def test_build_loss_injects_same_named_sample_rate_parameter():
    loss = build_loss(
        {"type": "DeepFilterNetMultiResSpecLoss", "args": {"fft_sizes": (320,)}},
        sample_rate=32000,
    )
    assert loss.sample_rate == 32000


def test_build_loss_injects_sample_rate_into_pase_sampling_rate_alias():
    loss = build_loss({"type": "PaseMultiScaleMelLoss"}, sample_rate=32000)

    assert loss.sampling_rate == 32000


def test_build_loss_pase_sampling_rate_injection_keeps_loss_formula():
    args = {"n_mels": [8], "window_lengths": [32]}
    built = build_loss(
        {"type": "PaseMultiScaleMelLoss", "args": args}, sample_rate=32000
    )
    direct = LOSS_REGISTRY.build("PaseMultiScaleMelLoss", sampling_rate=32000, **args)
    enhanced = torch.randn(1, 1, 128)
    clean = torch.randn_like(enhanced)

    assert torch.equal(built(enhanced, clean), direct(enhanced, clean))


@pytest.mark.parametrize(
    ("cfg", "attribute"),
    [
        ({"type": "FWNegSISDRLoss", "args": {"sr": 16000}}, "sr"),
        (
            {"type": "DeepFilterNetMultiResSpecLoss", "args": {"sample_rate": 16000}},
            "sample_rate",
        ),
        (
            {"type": "PaseMultiScaleMelLoss", "args": {"sampling_rate": 16000}},
            "sampling_rate",
        ),
    ],
)
def test_build_loss_explicit_sampling_rate_args_take_priority(cfg, attribute):
    loss = build_loss(cfg, sample_rate=32000)
    assert getattr(loss, attribute) == 16000


@pytest.mark.parametrize(
    ("loss_type", "attribute", "default"),
    [
        ("FWNegSISDRLoss", "sr", 16000),
        ("DeepFilterNetMultiResSpecLoss", "sample_rate", 48000),
        ("PaseMultiScaleMelLoss", "sampling_rate", 16000),
    ],
)
def test_build_loss_without_sample_rate_context_keeps_class_default(
    loss_type, attribute, default
):
    loss = build_loss({"type": loss_type})
    assert getattr(loss, attribute) == default


@pytest.mark.parametrize("sample_rate", [0, -16000, True, 32000.0])
def test_build_loss_rejects_invalid_sample_rate(sample_rate):
    with pytest.raises(ValueError, match="sample_rate"):
        build_loss({"type": "FWNegSISDRLoss"}, sample_rate=sample_rate)


@pytest.mark.parametrize(
    ("loss_type", "sample_rate_arg"),
    [
        ("FWNegSISDRLoss", "sr"),
        ("DeepFilterNetMultiResSpecLoss", "sample_rate"),
        ("PaseMultiScaleMelLoss", "sampling_rate"),
    ],
)
@pytest.mark.parametrize("sample_rate", [0, -16000, True, 32000.0])
def test_build_loss_rejects_invalid_explicit_sample_rate(sample_rate, loss_type, sample_rate_arg):
    with pytest.raises(ValueError, match=f"loss args\\.{sample_rate_arg}"):
        build_loss({"type": loss_type, "args": {sample_rate_arg: sample_rate}})


def test_build_loss_passes_sample_rate_to_nested_deepfilternet_enhancement_loss():
    loss = build_loss(
        {
            "type": "DeepFilterNetFullLossWithLSNR",
            "args": {
                "enhancement_loss_type": "FWNegSISDRLoss",
                "enhancement_loss_args": {"n_bins": 18, "scale": "mel"},
            },
        },
        sample_rate=32000,
    )
    assert loss.enhancement_loss.sr == 32000


def test_build_loss_follows_stft_geometry():
    """仅验证 stft_cfg 注入的 STFT 几何参数，不验证 sample_rate 上下文。"""
    cases = [
        (320, 160, 320),
        (480, 240, 480),
        (640, 320, 640),
        (960, 480, 960),
    ]
    for n_fft, hop, win in cases:
        cfg = {"type": "GTCRNCompressedSpectralLoss", "args": {"compress_factor": 0.3, "lamda_ri": 30.0, "lamda_mag": 70.0}}
        loss = build_loss(cfg, stft_cfg={"n_fft": n_fft, "hop_length": hop, "win_length": win})
        assert loss.n_fft == n_fft, "n_fft mismatch"
        assert loss.hop_length == hop, "hop mismatch"
        assert loss.win_length == win, "win mismatch"


def test_build_loss_single_resolution_follows_stft():
    """单分辨率 STFT loss（MagnitudePlusComplexL1）同样随 stft 注入。"""
    cfg = {"type": "MagnitudePlusComplexL1Loss"}
    loss = build_loss(cfg, stft_cfg={"n_fft": 480, "hop_length": 240, "win_length": 480})
    assert loss.n_fft == 480
    assert loss.hop_length == 240
    assert loss.win_length == 480


def test_build_loss_injects_window_from_stft():
    """window 随 stft 块注入；一致显式 args 保留，冲突显式 args fail-fast。"""
    cfg = {"type": "GTCRNCompressedSpectralLoss", "args": {"compress_factor": 0.3}}
    loss = build_loss(cfg, stft_cfg={"n_fft": 640, "hop_length": 320, "win_length": 640, "window": "hamming"})
    assert loss._window.shape[0] == 640
    # 注入 hamming 窗应与 torch.hamming_window 一致，而非默认 hann
    assert torch.equal(loss._window, torch.hamming_window(640))

    # 显式 window 与 stft 上下文一致时保留
    cfg2 = {"type": "GTCRNCompressedSpectralLoss", "args": {"window": "hamming"}}
    loss2 = build_loss(cfg2, stft_cfg={"n_fft": 640, "hop_length": 320, "win_length": 640, "window": "hamming"})
    assert torch.equal(loss2._window, torch.hamming_window(640))

    # REQUIRES_MATCHING_STFT_CONTEXT：显式 window 与上下文冲突时构造期 fail-fast
    cfg3 = {"type": "GTCRNCompressedSpectralLoss", "args": {"window": "hann"}}
    with pytest.raises(ValueError, match="window context conflict"):
        build_loss(cfg3, stft_cfg={"n_fft": 640, "hop_length": 320, "win_length": 640, "window": "hamming"})


def test_deepfilternet_full_loss_passes_stft_cfg():
    """DeepFilterNetFullLossWithLSNR 内部 build_loss 透传 stft_cfg（嵌套路径不丢 stft 默认）。"""
    from core.losses.model_specific.deepfilternet import DeepFilterNetFullLossWithLSNR

    loss = DeepFilterNetFullLossWithLSNR(
        enhancement_loss_type="GTCRNCompressedSpectralLoss",
        enhancement_loss_args={"compress_factor": 0.3},
        stft_cfg={"n_fft": 960, "hop_length": 480, "win_length": 960},
    )
    enh = loss.enhancement_loss
    assert enh.n_fft == 960
    assert enh.hop_length == 480
    assert enh.win_length == 960

    # 不传 stft_cfg 时保持原默认（SISNRLoss 为时域 loss，无 stft 属性）
    default = DeepFilterNetFullLossWithLSNR()
    assert default.enhancement_loss is not None


def test_build_loss_metricgan_composite_registered():
    """MPSEnetCompositeLoss 已注册且可由 YAML 形状的 loss 块经 build_loss 构造。

    MetricGAN 4 个生产 YAML 的 loss.type 已改为 MPSEnetCompositeLoss，entry 的
    build_loss 必须能构造（签名非标准 forward_dict，SIGNATURE 标注仅供 registry
    遍历跳过，此处仅验证注册与构造）。
    """
    from core.losses.model_specific.metricgan import MPSEnetCompositeLoss

    loss = build_loss(
        {
            "type": "MPSEnetCompositeLoss",
            "args": {
                "weight_mag": 0.9,
                "weight_pha": 0.3,
                "weight_com": 0.1,
                "weight_time": 0.2,
                "weight_stft": 0.1,
                "stft_n_fft": 400,
                "stft_hop_length": 100,
                "stft_win_length": 400,
                "compress_factor": 0.3,
            },
        },
        stft_cfg={"n_fft": 400, "hop_length": 100, "win_length": 400, "window": "hamming"},
    )
    assert isinstance(loss, MPSEnetCompositeLoss)
    assert loss.SIGNATURE == "forward_dict"
    # window 由 stft 块注入（显式 args 未声明 window）
    assert loss.window.shape[0] == 400
    assert torch.equal(loss.window, torch.hamming_window(400))


def test_scalar_passthrough_registers_without_pase_optional_dependencies():
    """PASE 重依赖导入失败时，轻量占位 Loss 仍必须独立注册。"""
    code = r'''
import builtins

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "core.losses.pase_family":
        raise ImportError("simulated optional dependency failure")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
from core.losses import build_loss
loss = build_loss({"type": "ScalarPassthroughLoss"})
assert type(loss).__name__ == "ScalarPassthroughLoss"
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


_MR_LOSS_TYPES = ("MRSIRelCspecL2Loss", "MRSpecSuppressPenaltyLoss")
_MR_STFT_SCALE_FACTORS = [0.125, 0.25, 0.5, 1, 2, 4]


@pytest.mark.parametrize("loss_type", _MR_LOSS_TYPES)
@pytest.mark.parametrize(
    ("win_length", "hop_length", "expected_windows", "expected_hops"),
    [
        (320, 160, [40, 80, 160, 320, 640, 1280], [20, 40, 80, 160, 320, 640]),
        (640, 320, [80, 160, 320, 640, 1280, 2560], [40, 80, 160, 320, 640, 1280]),
    ],
)
def test_build_loss_resolves_frame_relative_mr_schedule(
    loss_type, win_length, hop_length, expected_windows, expected_hops
):
    cfg = {
        "type": loss_type,
        "args": {"stft_scale_factors": list(_MR_STFT_SCALE_FACTORS)},
    }
    original = copy.deepcopy(cfg)
    stft_cfg = {"win_length": win_length, "hop_length": hop_length}

    loss = build_loss(cfg, stft_cfg=stft_cfg)

    assert list(loss.win_lengths) == expected_windows
    assert list(loss.hop_lengths) == expected_hops
    assert cfg == original
    assert stft_cfg == {"win_length": win_length, "hop_length": hop_length}


@pytest.mark.parametrize("loss_type", _MR_LOSS_TYPES)
def test_build_loss_keeps_default_legacy_mr_schedule(loss_type):
    loss = build_loss({"type": loss_type})

    assert list(loss.win_lengths) == [64, 128, 256, 512, 1024, 2048]
    assert list(loss.hop_lengths) == [32, 64, 128, 256, 512, 1024]


@pytest.mark.parametrize("loss_type", _MR_LOSS_TYPES)
def test_build_loss_explicit_legacy_mr_schedule_ignores_stft_context(loss_type):
    cfg = {
        "type": loss_type,
        "args": {"fft_power_min": 7, "fft_power_max": 9},
    }
    original = copy.deepcopy(cfg)

    loss = build_loss(
        cfg,
        stft_cfg={"win_length": 640, "hop_length": 320},
    )

    assert list(loss.win_lengths) == [128, 256, 512]
    assert list(loss.hop_lengths) == [64, 128, 256]
    assert cfg == original


@pytest.mark.parametrize("loss_type", _MR_LOSS_TYPES)
def test_build_loss_rejects_frame_relative_and_legacy_mr_schedule_conflict(loss_type):
    cfg = {
        "type": loss_type,
        "args": {
            "stft_scale_factors": list(_MR_STFT_SCALE_FACTORS),
            "fft_power_min": 6,
            "fft_power_max": 11,
        },
    }

    with pytest.raises(ValueError, match="conflict"):
        build_loss(cfg, stft_cfg={"win_length": 320, "hop_length": 160})


@pytest.mark.parametrize("loss_type", _MR_LOSS_TYPES)
@pytest.mark.parametrize(
    "stft_cfg",
    [None, {"win_length": 320}, {"hop_length": 160}],
)
def test_build_loss_frame_relative_mr_schedule_requires_win_and_hop(loss_type, stft_cfg):
    cfg = {
        "type": loss_type,
        "args": {"stft_scale_factors": list(_MR_STFT_SCALE_FACTORS)},
    }

    with pytest.raises(ValueError, match="win_length.*hop_length"):
        build_loss(cfg, stft_cfg=stft_cfg)


@pytest.mark.parametrize("loss_type", _MR_LOSS_TYPES)
def test_build_loss_rejects_non_integer_frame_relative_sample_count(loss_type):
    cfg = {
        "type": loss_type,
        "args": {"stft_scale_factors": [0.125]},
    }

    with pytest.raises(ValueError, match="integer sample"):
        build_loss(cfg, stft_cfg={"win_length": 300, "hop_length": 120})


@pytest.mark.parametrize("loss_type", _MR_LOSS_TYPES)
@pytest.mark.parametrize(
    "factors",
    [[], [0], [-0.5], [1, 0.5], [1, 1], [True], [float("inf")], [float("nan")], "1,2", 1],
)
def test_build_loss_rejects_invalid_frame_relative_mr_factors(loss_type, factors):
    cfg = {
        "type": loss_type,
        "args": {"stft_scale_factors": factors},
    }

    with pytest.raises((TypeError, ValueError)):
        build_loss(cfg, stft_cfg={"win_length": 320, "hop_length": 160})


@pytest.mark.unit
def test_build_loss_mr_suppress_penalty_reuses_one_shot_scale_factor_generator():
    scale_factors = (factor for factor in _MR_STFT_SCALE_FACTORS)
    loss = build_loss(
        {
            "type": "MRSIRelCspecL2SuppressPenaltyLoss",
            "args": {"stft_scale_factors": scale_factors},
        },
        stft_cfg={"win_length": 320, "hop_length": 160},
    )
    expected_fft_sizes = [40, 80, 160, 320, 640, 1280]
    expected_hop_sizes = [20, 40, 80, 160, 320, 640]

    assert list(loss.mr_loss.fft_sizes) == expected_fft_sizes
    assert list(loss.mr_loss.hop_sizes) == expected_hop_sizes
    assert list(loss.penalty_loss.fft_sizes) == expected_fft_sizes
    assert list(loss.penalty_loss.hop_sizes) == expected_hop_sizes


def _single_res_mrstft_loss():
    from core.losses.freq_domain import MultiResSTFTLoss

    return MultiResSTFTLoss(
        fft_sizes=(64,), hop_sizes=(16,), win_lengths=(64,)
    )


def _single_res_log_mag(enhanced, target):
    window = torch.hann_window(64)
    kwargs = {
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "window": window,
        "center": True,
        "return_complex": True,
    }
    enhanced_mag = torch.stft(enhanced.float(), **kwargs).abs().clamp(min=1e-8)
    target_mag = torch.stft(target.float(), **kwargs).abs().clamp(min=1e-8)
    return (enhanced_mag.log() - target_mag.log()).abs().mean(dim=(1, 2))


@pytest.mark.parametrize("target_scale", [0.0, 1e-12])
def test_multi_res_stft_zero_target_uses_only_log_magnitude_error(target_scale):
    generator = torch.Generator().manual_seed(11)
    enhanced = torch.randn(2, 320, generator=generator).requires_grad_(True)
    target = target_scale * torch.randn(2, 320, generator=generator)

    actual = _single_res_mrstft_loss()(enhanced, target)
    log_mag = _single_res_log_mag(enhanced, target)

    # target=0/近零时标准 SC 无定义，因此只保留一次 log-mag residual suppression。
    assert torch.allclose(actual, log_mag.mean(), rtol=1e-6, atol=1e-7)
    assert actual < 100.0
    actual.backward()
    assert torch.isfinite(enhanced.grad).all()
    assert torch.count_nonzero(enhanced.grad) > 0


def test_multi_res_stft_normal_target_matches_per_sample_standard_formula():
    generator = torch.Generator().manual_seed(12)
    target = torch.randn(2, 320, generator=generator)
    enhanced = target + 0.1 * torch.randn(2, 320, generator=generator)
    window = torch.hann_window(64)
    kwargs = {
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "window": window,
        "center": True,
        "return_complex": True,
    }
    enhanced_mag = torch.stft(enhanced, **kwargs).abs().clamp(min=1e-8)
    target_mag = torch.stft(target, **kwargs).abs().clamp(min=1e-8)
    sc = torch.linalg.vector_norm(
        target_mag - enhanced_mag, dim=(1, 2)
    ) / torch.linalg.vector_norm(target_mag, dim=(1, 2))
    expected = (sc + _single_res_log_mag(enhanced, target)).mean()

    actual = _single_res_mrstft_loss()(enhanced, target)

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_multi_res_stft_mixed_batch_reduces_per_sample_and_is_padding_invariant():
    generator = torch.Generator().manual_seed(13)
    normal_target = torch.randn(320, generator=generator)
    normal_enhanced = normal_target + 0.05 * torch.randn(320, generator=generator)
    pure_target = torch.zeros(224)
    pure_enhanced = 0.2 * torch.randn(224, generator=generator)
    loss_fn = _single_res_mrstft_loss()
    expected = (
        loss_fn(normal_enhanced[None, :], normal_target[None, :])
        + loss_fn(pure_enhanced[None, :], pure_target[None, :])
    ) / 2.0

    for total_length in (384, 640):
        enhanced = torch.randn(2, total_length, generator=generator)
        target = torch.randn(2, total_length, generator=generator)
        enhanced[0, :320] = normal_enhanced
        target[0, :320] = normal_target
        enhanced[1, :224] = pure_enhanced
        target[1, :224] = pure_target

        actual = loss_fn(enhanced, target, torch.tensor([320, 224]))

        assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)


pytestmark = [pytest.mark.unit]
