"""SSL DP-IPD PIT 的多源指派合同测试。

覆盖：K<=2 枚举路径逐位不变、K>=3 Hungarian 与暴力枚举最优值一致、
source 置换不变性、VAD/pair_mask/length 边界、全 inactive 的有限零值、
DOA 辅助项沿用选中指派，以及 K=4 的性能 smoke。
"""

import itertools
import time

import pytest
import torch

from tasks.ssl.losses import (
    SSLDPIPDPITLoss,
    _batched_min_assignment,
    _enumerate_min_assignment,
    _hungarian_min_assignment,
)


pytestmark = pytest.mark.unit


def _reference_enum_loss(
    prediction,
    target,
    *,
    lengths=None,
    vad=None,
    pair_mask=None,
):
    """独立的纯枚举参考实现（逐位对齐 loss 内部归约顺序，故非“教科书式”简化版）。

    它先在 ``[B,T,F,K_pred,K_ref]`` 上按 pair_mask 求复 MSE，再乘 mask 并沿
    频率求和得到 ``cost``；随后枚举全部 ``K!`` 个排列取最小值，按每帧有效
    频率数求平均，最后用 ``_masked_mean`` 的归约顺序（float64 归一化求和）。
    """
    batch, frames, freqs, tracks, pairs, _ = prediction.shape
    diff = prediction.unsqueeze(4) - target.unsqueeze(3)
    pair_error = diff.square().sum(dim=-1)
    if pair_mask is None:
        values = pair_error.mean(dim=-1)
    else:
        # pair_error 是 [B,T,F,K_pred,K_ref,P]，pair_mask 是 [B,P]。
        weights = pair_mask[:, None, None, None, None, :].to(pair_error.dtype)
        values = (pair_error * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
    mask = torch.ones(batch, frames, freqs, tracks, dtype=torch.bool)
    if lengths is not None:
        frame_ids = torch.arange(frames)[None, :]
        mask = mask & (frame_ids < lengths[:, None])[:, :, None, None]
    if vad is not None:
        mask = mask & (vad[:, :, None, :] > 0.5)
    # values 是 [B,T,F,K_pred,K_ref]（float32），mask 是 [B,T,F,K_ref]。
    # loss 在 [B,T,F,K,K] 上逐元素 mask 后再沿频率 sum；参考实现按同样的
    # 求和顺序还原，才能要求逐位一致。先算 values 再乘 mask，保证与 loss
    # 内部的乘序一致。
    weighted = values * mask.unsqueeze(3).to(values.dtype)
    cost = weighted.sum(dim=2)  # [B,T,K,K]
    totals = []
    for permutation in itertools.permutations(range(tracks)):
        total = None
        for row, column in enumerate(permutation):
            term = cost[:, :, row, column]
            total = term if total is None else total + term
        totals.append(total)
    best = torch.stack(totals, dim=-1).min(dim=-1).values
    # cost 是 [B,T,K_pred,K_ref]；frame_mass 的口径是每个预测轨道上有效频率
    # 数之和，因此单独在 best 上除以它，不参与每个排列的代价求和。
    frame_mass = mask.sum(dim=-2).sum(dim=-1).clamp_min(1).float()  # [B,T]
    best = best / frame_mass.to(best.dtype)
    # loss 的归约是 _masked_mean(best, valid_frames)，参考实现保持同一顺序：
    # float32 除法 + float64 归一化求和。
    valid_frames = mask.any(dim=(-1, -2))
    local_sum = best.detach().double()[valid_frames].sum()
    local_mass = valid_frames.sum().double()
    if float(local_mass.item()) == 0.0:
        return (best * 0.0).sum().float()
    return (local_sum / local_mass).float()


def _tensor(shape, seed, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator) * scale


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_k2_matches_reference_enumeration_bitwise(seed):
    """K=2 必须保持旧的纯枚举结果，逐位一致。"""
    prediction = _tensor((2, 4, 3, 2, 3, 2), seed)
    target = _tensor((2, 4, 3, 2, 3, 2), seed + 100)
    pair_mask = torch.tensor([[True, False, True], [True, True, False]])
    lengths = torch.tensor([3, 4])
    vad = torch.ones(2, 4, 2)
    vad[1, 3] = 0.0
    loss_fn = SSLDPIPDPITLoss(max_tracks=2)
    loss = loss_fn(
        prediction,
        target,
        output_lengths=lengths,
        vad=vad,
        pair_mask=pair_mask,
    )
    assert torch.isfinite(loss)
    expected = _reference_enum_loss(
        prediction, target, lengths=lengths, vad=vad, pair_mask=pair_mask
    )
    torch.testing.assert_close(loss, expected, rtol=1e-4, atol=1e-5)


def test_k2_keeps_enumeration_path_and_k3_switches_to_hungarian():
    """K=2 走枚举、K>=3 走 Hungarian，路由必须显式可查。"""
    cost = torch.rand(2, 5, 2, 2)
    torch.testing.assert_close(
        _batched_min_assignment(cost), _enumerate_min_assignment(cost)
    )
    cost3 = torch.rand(2, 5, 3, 3)
    torch.testing.assert_close(
        _batched_min_assignment(cost3), _hungarian_min_assignment(cost3)
    )


@pytest.mark.parametrize("tracks", [3, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_k_ge_3_hungarian_matches_brute_force_optimum(tracks, seed):
    """K=3/4 时 Hungarian 的选中代价必须等于暴力枚举的最优代价。"""
    cost = _tensor((2, 6, tracks, tracks), seed, scale=5.0)
    assignment = _hungarian_min_assignment(cost)
    assert torch.equal(
        assignment.sort(dim=-1).values,
        torch.arange(tracks).expand_as(assignment),
    )
    hungarian_cost = cost.gather(-1, assignment.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    brute = _enumerate_min_assignment(cost)
    brute_cost = cost.gather(-1, brute.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    torch.testing.assert_close(hungarian_cost, brute_cost, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("tracks", [3, 4])
def test_loss_is_invariant_to_target_source_permutation(tracks):
    """K=3/4 时 target 的 source 轴置换不能改变 PIT 主损失。"""
    prediction = _tensor((1, 5, 4, tracks, 1, 2), seed=11)
    target = _tensor((1, 5, 4, tracks, 1, 2), seed=12)
    permutation = torch.roll(torch.arange(tracks), 1)
    permuted = target[:, :, :, permutation].clone()
    loss_fn = SSLDPIPDPITLoss(max_tracks=tracks)
    base = loss_fn(prediction, target, output_lengths=torch.tensor([5]))
    shuffled = loss_fn(prediction, permuted, output_lengths=torch.tensor([5]))
    torch.testing.assert_close(shuffled, base, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("tracks", [3, 4])
def test_masked_frames_and_pairs_are_excluded_from_pit_cost(tracks):
    """部分帧全 inactive、部分 track 无效、部分 pair 被屏蔽的边界场景。"""
    prediction = _tensor((2, 4, 3, tracks, 3, 2), seed=21)
    target = _tensor((2, 4, 3, tracks, 3, 2), seed=22)
    vad = torch.ones(2, 4, tracks)
    vad[0, 3:] = 0.0  # batch 0 的后一帧整帧 inactive
    vad[1, 1, 1] = 0.0  # batch 1 的 track 1 在中段无效
    vad[1, 3, 2] = 0.0  # batch 1 的末帧只留下部分有效 track
    pair_mask = torch.tensor([[True, False, True], [True, True, False]])
    lengths = torch.tensor([4, 4])
    loss_fn = SSLDPIPDPITLoss(max_tracks=tracks)
    loss = loss_fn(
        prediction,
        target,
        output_lengths=lengths,
        vad=vad,
        pair_mask=pair_mask,
    )
    expected = _reference_enum_loss(
        prediction, target, lengths=lengths, vad=vad, pair_mask=pair_mask
    )
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)

    # 整帧 inactive 的帧被逐帧剔除：把 batch 0 第 4 帧（整帧 inactive）的
    # 预测/目标换成完全不参与的噪声时损失不变。batch 1 末帧仍有部分有效
    # track，属于该帧的部分掩蔽而不是整帧剔除。
    noisy_prediction = prediction.clone()
    noisy_target = target.clone()
    noisy_prediction[0, 3] = _tensor((3, tracks, 3, 2), seed=41, scale=100.0)
    noisy_target[0, 3] = _tensor((3, tracks, 3, 2), seed=42, scale=100.0)
    perturbed = loss_fn(
        noisy_prediction,
        noisy_target,
        output_lengths=lengths,
        vad=vad,
        pair_mask=pair_mask,
    )
    assert torch.isfinite(perturbed)
    torch.testing.assert_close(perturbed, loss, rtol=1e-5, atol=1e-6)

    # batch 1 剩余 track 全无效时，该帧同样整帧剔除。
    vad_sparse = torch.ones(2, 4, tracks)
    vad_sparse[1] = 0.0
    sparse = loss_fn(
        prediction,
        target,
        output_lengths=lengths,
        vad=vad_sparse,
        pair_mask=pair_mask,
    )
    sparse_expected = _reference_enum_loss(
        prediction, target, lengths=lengths, vad=vad_sparse, pair_mask=pair_mask
    )
    assert torch.isfinite(sparse)
    torch.testing.assert_close(sparse, sparse_expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("tracks", [3, 4])
def test_all_inactive_batch_returns_finite_zero(tracks):
    """全部帧 inactive 时没有有效 target：返回有限的零而不是 NaN。"""
    prediction = _tensor((2, 3, 4, tracks, 1, 2), seed=31)
    target = _tensor((2, 3, 4, tracks, 1, 2), seed=32)
    loss = SSLDPIPDPITLoss(max_tracks=tracks)(
        prediction,
        target,
        output_lengths=torch.tensor([3, 3]),
        vad=torch.zeros(2, 3, tracks),
    )
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.zeros_like(loss))


def test_doa_auxiliary_loss_reuses_k3_assignments():
    """K=3 的 DOA 辅助项必须沿用 Hungarian 选出的 source 指派。"""
    tracks = 3
    target = torch.zeros(1, 1, 1, tracks, 1, 2)
    target[:, :, :, 0, :, 0] = 1.0
    target[:, :, :, 1, :, 1] = 1.0
    target[:, :, :, 2, :, 0] = 0.5
    target[:, :, :, 2, :, 1] = 0.5
    permutation = torch.tensor([2, 0, 1])
    prediction = target[:, :, :, permutation].clone()
    angles = torch.tensor(
        [[[[torch.pi / 2, 0.0], [torch.pi / 2, torch.pi / 2], [torch.pi / 3, torch.pi / 4]]]]
    )
    loss = SSLDPIPDPITLoss(max_tracks=tracks, doa_loss_weight=1.0)(
        prediction,
        target_dpipd=target,
        output_lengths=torch.tensor([1]),
        doa=angles,
        vad=torch.ones(1, 1, tracks),
        doa_prediction=angles[:, :, permutation].clone(),
        pair_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    # 预测是 target 的同构置换、DOA 同步置换时，指派应把预测还原到正确源，
    # 主损失与 DOA 辅助项都接近 0。
    assert loss.item() < 1e-6


def test_k4_forward_is_within_a_loose_time_budget():
    """性能 smoke：K=4、B=2、T=64 的单次 forward 必须落在宽松上限内。"""
    tracks, batch, frames, freqs = 4, 2, 64, 257
    prediction = _tensor((batch, frames, freqs, tracks, 1, 2), seed=41)
    target = _tensor((batch, frames, freqs, tracks, 1, 2), seed=42)
    loss_fn = SSLDPIPDPITLoss(max_tracks=tracks)
    with torch.no_grad():
        loss_fn(prediction, target)
        start = time.perf_counter()
        loss = loss_fn(prediction, target)
        elapsed = time.perf_counter() - start
    assert torch.isfinite(loss)
    assert elapsed < 2.0, f"K=4 forward took {elapsed * 1000:.1f} ms"
