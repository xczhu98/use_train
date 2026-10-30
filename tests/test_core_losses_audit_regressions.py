"""core/losses 审计问题的独立回归测试。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from core.losses.composite import CompositeLoss
from core.losses.model_specific.deepfilternet import (
    DeepFilterNetLSNRComponent,
    DfAlphaLossComponent,
    DeepFilterNetMultiResSpecLoss,
)
from core.losses.pase_family import PaseMultiScaleMelLoss
from core.losses.perceptual_lib import AuralossSDRLoss
from core.losses.perceptual_metric import PESQLoss, STOILoss
from core.losses.plcpa_asymmetric import PLCPAAsymmetricLoss
from core.utils.registry import LOSS_REGISTRY


@pytest.mark.parametrize(
    ("loss_type", "kwargs", "expected"),
    [
        (
            "StateCrossEntropyLoss",
            {
                "state_probs": torch.tensor([[[2.0, -1.0]]], requires_grad=True),
                "state": torch.tensor([[0]]),
            },
            "state_probs",
        ),
        (
            "SpeakerCrossEntropyLoss",
            {
                "spk_logits": torch.tensor([[2.0, -1.0]], requires_grad=True),
                "spk_labels": torch.tensor([0]),
            },
            "spk_logits",
        ),
    ],
)
def test_composite_mean_mass_matches_two_rank_global_mean(monkeypatch, loss_type, kwargs, expected):
    """DDP 两 rank 有效标签数不等时，local mean 必须按全局质量缩放。"""
    from core.losses import composite as composite_module

    calls = 0

    def fake_sum(local):
        nonlocal calls
        calls += 1
        # 第一次为 mean mass：本 rank 一个有效标签，另一 rank 三个；第二次
        # 是 Composite 的失败状态同步，保持单 rank 避免触发真实 all_gather。
        if calls == 1:
            return local + local.new_tensor(3.0), 2
        return local, 1

    monkeypatch.setattr(composite_module, "_distributed_sum_scalar", fake_sum)
    loss_fn = CompositeLoss(components=[{"type": loss_type, "weight": 1.0}])
    enhanced = torch.randn(1, 8, requires_grad=True)
    clean = torch.randn_like(enhanced)
    raw = loss_fn.losses[0](enhanced, clean, **kwargs)
    actual = loss_fn(enhanced, clean, **kwargs)

    # DDP 最终平均两 rank 梯度；本 rank 目标权重应为 1 / (1 + 3)。
    torch.testing.assert_close(actual, raw * 0.5)
    actual.backward()
    assert kwargs[expected].grad is not None


@pytest.mark.parametrize("loss_cls", [PESQLoss, STOILoss])
def test_missing_perceptual_metric_returns_graph_connected_zero(loss_cls):
    """allow_missing 的零值仍必须连接 enhanced 图。"""
    loss_fn = loss_cls(allow_missing=True)
    loss_fn.available = False
    enhanced = torch.randn(2, 32, requires_grad=True)
    out = loss_fn(enhanced, torch.randn_like(enhanced))
    assert out.item() == 0.0 and out.requires_grad
    out.backward()
    assert enhanced.grad is not None


class _AuralossSubsetProbe(nn.Module):
    """替代 auraloss 内核，记录 Composite 传入的 speech 子集大小。"""

    def __init__(self):
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, enhanced, clean):
        self.batch_sizes.append(enhanced.shape[0])
        return (enhanced - clean).square().mean()


@LOSS_REGISTRY.register("_AuditAuralossSDR")
class _AuditAuralossSDR(AuralossSDRLoss):
    """无需 optional dependency 的 AuralossSDR Composite 代理。"""

    def __init__(self):
        nn.Module.__init__(self)
        self.available = True
        self.loss_fn = _AuralossSubsetProbe()


def test_auraloss_sdr_excludes_pure_reference_free_samples():
    """AuralossSDRLoss 的 capability 必须使 pure 样本不进入 SDR。"""
    assert AuralossSDRLoss.REQUIRES_SPEECH_REFERENCE is True
    loss_fn = CompositeLoss(components=[{"type": "_AuditAuralossSDR", "weight": 1.0}])
    enhanced = torch.randn(3, 64, requires_grad=True)
    clean = torch.randn_like(enhanced)
    out = loss_fn(
        enhanced,
        clean,
        is_pure_noise=torch.tensor([True, False, True]),
    )
    out.backward()
    assert loss_fn.losses[0].loss_fn.batch_sizes == [1]


def test_deepfilternet_max_freq_keeps_the_last_allowed_bin(monkeypatch):
    """max_freq_hz 必须包含频率恰好等于上界的 one-sided bin。"""
    loss_fn = DeepFilterNetMultiResSpecLoss(
        fft_sizes=(64,),
        hop_factor=0.5,
        sample_rate=64,
        max_freq_hz=1.0,
    )
    enhanced_mag = torch.zeros(1, 33, 1)
    clean_mag = enhanced_mag.clone()
    clean_mag[:, 1] = 2.0  # k=1 的频率正好是 1 Hz。
    values = iter((enhanced_mag, clean_mag))
    monkeypatch.setattr(loss_fn, "_stft_mag", lambda *args: next(values))

    actual = loss_fn._one_resolution(torch.zeros(1, 64), torch.zeros(1, 64), 64, None)
    torch.testing.assert_close(actual, torch.tensor(1.0))


def test_plcpa_noncentered_rejects_short_valid_length_and_masks_tail():
    """center=False 不得将短有效段的 padding 伪装为 STFT 帧。"""
    loss_fn = PLCPAAsymmetricLoss(n_fft=64, hop_length=16, win_length=64, center=False)
    short = torch.randn(1, 96, requires_grad=True)
    with pytest.raises(ValueError, match="valid length"):
        loss_fn(short, torch.randn_like(short), torch.tensor([63]))

    enhanced = torch.randn(1, 128, requires_grad=True)
    clean = torch.randn_like(enhanced)
    out = loss_fn(enhanced, clean, torch.tensor([64]))
    out.backward()
    assert torch.count_nonzero(enhanced.grad[:, 64:]) == 0


@pytest.mark.parametrize(
    "component, kwargs, message",
    [
        (DeepFilterNetLSNRComponent(), {}, "predicted_lsnr"),
        (DeepFilterNetLSNRComponent(), {"predicted_lsnr": torch.randn(1, 2)}, "noise or noisy"),
        (DfAlphaLossComponent(), {}, "predicted_alpha"),
        (DfAlphaLossComponent(), {"predicted_alpha": torch.randn(1, 2)}, "noise or noisy"),
    ],
)
def test_deepfilternet_components_fail_fast_for_missing_auxiliary_inputs(component, kwargs, message):
    """非零辅助分量缺模型输出或噪声参考时不得静默退化为零。"""
    enhanced = torch.randn(1, 640)
    clean = torch.randn_like(enhanced)
    with pytest.raises(ValueError, match=message):
        component(enhanced, clean, **kwargs)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DeepFilterNetMultiResSpecLoss(fft_sizes=()),
        lambda: DeepFilterNetMultiResSpecLoss(fft_sizes=(64,), max_freq_hz=0.0),
        lambda: PaseMultiScaleMelLoss(n_mels=[8], window_lengths=[64, 128]),
        lambda: PaseMultiScaleMelLoss(n_mels=[], window_lengths=[]),
    ],
)
def test_multiscale_loss_schedules_fail_fast(factory):
    """多尺度 schedule 不得依靠 zip 截断或空列表除零。"""
    with pytest.raises(ValueError):
        factory()


pytestmark = [pytest.mark.unit]


def test_strict_loss_wrapper_raises_explicit_errors_instead_of_assertions():
    """strict 门禁必须在 -O 下同样生效，不能依赖会被剥离的 assert。"""
    from core.losses.build import _StrictLossWrapper

    class _ScalarL1(nn.Module):
        def forward(self, enhanced, clean, lengths=None):
            return (enhanced - clean).abs().mean()

    class _SeparationL1(nn.Module):
        SEPARATION_LOSS = True

        def forward(self, enhanced, clean, lengths=None, source_mask=None):
            return (enhanced - clean).square().mean()

    wrapper = _StrictLossWrapper(_ScalarL1())
    with pytest.raises(ValueError, match=r"enhanced must be \[B, T\]"):
        wrapper(torch.ones(2, 8, 1), torch.ones(2, 8, 1))
    with pytest.raises(TypeError, match="integer dtype"):
        wrapper(torch.ones(2, 8), torch.ones(2, 8), lengths=torch.tensor([8.0, 8.0]))
    with pytest.raises(ValueError, match="lengths out of range"):
        wrapper(torch.ones(2, 8), torch.ones(2, 8), lengths=torch.tensor([9, 8]))

    separation = _StrictLossWrapper(_SeparationL1())
    with pytest.raises(TypeError, match="source_mask must be bool"):
        separation(
            torch.ones(2, 3, 8),
            torch.ones(2, 3, 8),
            source_mask=torch.ones(2, 3, dtype=torch.long),
        )


def test_strict_loss_wrapper_caches_breakdown_materialization():
    """分项 float 物化只在首次消费时发生，同一份 tensor 不重复 .item()。"""
    from core.losses.build import _StrictLossWrapper

    wrapper = _StrictLossWrapper(CompositeLoss(components=[{"type": "L1WavLoss", "weight": 1.0}]))
    wrapper(torch.ones(2, 8), torch.zeros(2, 8))
    first = wrapper._last_breakdown
    assert first is not None and len(first) == 1
    assert wrapper._last_breakdown is first

    wrapper(torch.full((2, 8), 2.0), torch.zeros(2, 8))
    assert wrapper._last_breakdown is not first


def test_metricgan_composite_loss_rejects_unknown_window():
    """窗口名拼写错误不得静默回退 Hann。"""
    from core.losses.model_specific.metricgan import MPSEnetCompositeLoss

    with pytest.raises(ValueError, match="window must be one of"):
        MPSEnetCompositeLoss(
            stft_n_fft=64, stft_hop_length=16, stft_win_length=64, window="hanning"
        )


def test_state_ce_rejects_unsigned_labels():
    """ignore_index 是负数 sentinel，无符号标签无法表达 padding。"""
    from core.losses.state_ce import StateCrossEntropyLoss

    loss = StateCrossEntropyLoss()
    state_probs = torch.randn(1, 2, 3)
    with pytest.raises(TypeError, match="signed integer"):
        loss(None, None, state_probs=state_probs, state=torch.zeros(1, 2, dtype=torch.uint8))
    assert torch.isfinite(
        loss(None, None, state_probs=state_probs, state=torch.zeros(1, 2, dtype=torch.int64))
    )


def test_suppress_penalty_rejects_unconsumed_align_se_train():
    """suppress-only 类不读取 align_se_train，非默认值必须 fail-fast。"""
    from core.losses.se_train_mr import (
        MRSIRelCspecL2SuppressPenaltyLoss,
        MRSpecSuppressPenaltyLoss,
    )

    with pytest.raises(ValueError, match="align_se_train"):
        MRSpecSuppressPenaltyLoss(
            fft_power_min=6, fft_power_max=7, align_se_train=False
        )

    combined = MRSIRelCspecL2SuppressPenaltyLoss(
        fft_power_min=6, fft_power_max=7, align_se_train=False
    )
    assert combined.align_se_train is False
    assert combined.penalty_loss.align_se_train is True


def test_utmos_aware_loss_import_does_not_leave_source_dir_on_sys_path(monkeypatch, tmp_path):
    """vendored SpeechMOS 的导入路径只在导入窗口内临时提升。"""
    import sys
    import types

    from core.losses import pase_family

    class _TinyUTMOS(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

    module_names = (
        "speechmos",
        "speechmos.utmos22",
        "speechmos.utmos22.strong",
        "speechmos.utmos22.strong.model",
    )
    modules = [types.ModuleType(name) for name in module_names]
    modules[-1].UTMOS22Strong = _TinyUTMOS
    modules[0].utmos22 = modules[1]
    modules[1].strong = modules[2]
    modules[2].model = modules[3]
    for name, module in zip(module_names, modules):
        monkeypatch.setitem(sys.modules, name, module)

    (tmp_path / "SpeechMOS" / "speechmos").mkdir(parents=True)
    (tmp_path / "utmos22_strong_step7459_v1.pt").write_bytes(b"")
    monkeypatch.setattr(
        pase_family,
        "safe_torch_load",
        lambda path, **kwargs: {"weight": torch.ones(1)},
    )

    sys_path_before = list(sys.path)
    loss = pase_family._UTMOSAwareLoss(str(tmp_path))

    assert loss.model.weight.item() == pytest.approx(1.0)
    assert sys.path == sys_path_before
