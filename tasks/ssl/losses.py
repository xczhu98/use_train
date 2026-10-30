"""SSL losses."""

from __future__ import annotations

import itertools
import math
from numbers import Real

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from core.utils.registry import LOSS_REGISTRY


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_INPUT_REPRESENTATIONS = frozenset({
    "multichannel_stft_real_imag",
    "reference_ipd",
})
_TARGET_REPRESENTATIONS = frozenset({
    "dpipd_far_field",
    "dpipd_near_field",
})


def _contains_boolean(value) -> bool:
    """递归拒绝配置列表中的 bool，避免混合数值列表把 True 转成 1.0。"""
    if type(value) is bool:
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    return False


def _global_mass(local_mass: torch.Tensor) -> tuple[torch.Tensor, int]:
    if not (dist.is_available() and dist.is_initialized()):
        return local_mass, 1
    world_size = dist.get_world_size()
    if world_size <= 1:
        return local_mass, world_size
    global_mass = local_mass.detach().clone()
    dist.all_reduce(global_mass, op=dist.ReduceOp.SUM)
    return global_mass, world_size


def _all_ranks_finite(value: torch.Tensor) -> bool:
    """在 reduction 后同步有限性，避免单 rank 产生 Inf 后 collective 失步。"""
    finite = torch.tensor(
        int(bool(torch.isfinite(value).all().item())),
        dtype=torch.int32,
        device=value.device,
    )
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    values32 = values.float()
    if mask is None:
        local_sum = values32.sum()
        local_mass = values32.new_tensor(values32.numel(), dtype=torch.float32)
    else:
        if mask.dtype is not torch.bool or mask.ndim > values.ndim:
            raise ValueError("SSL loss mask must be a broadcastable bool tensor")
        expanded = mask.reshape(mask.shape + (1,) * (values.ndim - mask.ndim))
        try:
            valid = expanded.expand_as(values)
        except RuntimeError as exc:
            raise ValueError("SSL loss mask is not broadcastable to prediction") from exc
        local_sum = values32.masked_select(valid).sum()
        local_mass = valid.sum().to(dtype=torch.float32)
    if not _all_ranks_finite(local_sum):
        raise ValueError("SSL loss reduction must be finite")
    global_mass, world_size = _global_mass(local_mass)
    if not torch.isfinite(global_mass).item() or global_mass.item() < 0:
        raise ValueError("SSL loss valid mass must be finite and non-negative")
    if global_mass.item() == 0:
        # 不能用 values.sum()*0：有限元素的总和可能先溢出为 Inf，Inf*0 会变成 NaN。
        return values.reshape(-1)[0] * 0.0
    # DDP averages parameter gradients across ranks; scale each local sum to the
    # global valid-element mean before backward so unequal valid frame counts do
    # not change the objective.
    result = local_sum * (float(world_size) / global_mass.to(local_sum.dtype))
    if not _all_ranks_finite(result):
        raise ValueError("SSL loss reduction must be finite")
    return result


def _validate_named_targets(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    doa: torch.Tensor | None,
    vad: torch.Tensor | None,
) -> None:
    """校验具名空间监督；当前 loss 的优化目标明确只有 IPD MSE。"""
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("SSL DPIPD prediction must be a torch.Tensor")
    if not torch.is_floating_point(prediction) or not torch.isfinite(prediction).all():
        raise ValueError("SSL DPIPD prediction must be finite floating-point")
    if prediction.ndim != 6 or prediction.shape[-1] != 2:
        raise ValueError("SSL DPIPD prediction must have shape [B,T,F,K,P,2]")
    if target is not None:
        if not isinstance(target, torch.Tensor):
            raise TypeError("SSL DPIPD target must be a torch.Tensor")
        if not torch.is_floating_point(target) or not torch.isfinite(target).all():
            raise ValueError("SSL DPIPD target must be finite floating-point")
        if target.shape != prediction.shape:
            raise ValueError(
                "SSL prediction/target must share B,T,F and have identical "
                "shape [B,T,F,K,P,2]"
            )
    batch, frames, frequencies, tracks, pairs, _ = prediction.shape
    if batch < 1 or frames < 1 or frequencies < 1:
        raise ValueError("SSL DPIPD prediction must contain at least one batch, frame, and frequency")
    if tracks < 1 or pairs < 1:
        raise ValueError("SSL DPIPD prediction must contain at least one track and pair")
    if doa is not None:
        if (
            not isinstance(doa, torch.Tensor)
            or doa.shape != (batch, frames, tracks, 2)
            or not torch.isfinite(doa).all()
        ):
            raise ValueError("SSL doa must have shape [B,T,K,2] and be finite")
    if vad is not None:
        if (
            not isinstance(vad, torch.Tensor)
            or vad.shape != (batch, frames, tracks)
            or not torch.isfinite(vad).all()
        ):
            raise ValueError("SSL vad must have shape [B,T,K] and be finite")
        if torch.any((vad < 0.0) | (vad > 1.0)):
            raise ValueError("SSL vad must be in [0, 1]")


def _doa_unit_vector(angles: torch.Tensor) -> torch.Tensor:
    """把 canonical ``(polar, azimuth)`` 转成三维单位向量。

    用单位向量计算 DOA 辅助项可以避免 azimuth 在 ``-π/π`` 处产生人为
    跳变；它等价于同时约束 sin/cos，并保留极角的球面几何。
    """
    if angles.ndim < 1 or angles.shape[-1] != 2:
        raise ValueError("SSL doa angles must end with (polar, azimuth)")
    polar, azimuth = angles.unbind(dim=-1)
    return torch.stack(
        (
            polar.sin() * azimuth.cos(),
            polar.sin() * azimuth.sin(),
            polar.cos(),
        ),
        dim=-1,
    )


def _circular_doa_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """计算不受 azimuth wrap-around 影响的球面 DOA 误差。"""
    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("SSL DOA prediction/target must share [...,2] shape")
    values = (_doa_unit_vector(prediction) - _doa_unit_vector(target)).square()
    return _masked_mean(values, mask)


def ipd_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor | None = None,
    output_lengths=None,
    *,
    target_dpipd: torch.Tensor | None = None,
    input_lengths=None,
    doa: torch.Tensor | None = None,
    vad: torch.Tensor | None = None,
    pair_mask: torch.Tensor | None = None,
    pair_index: torch.Tensor | None = None,
    geometry: torch.Tensor | None = None,
    source_distances: torch.Tensor | None = None,
    frame_origin: torch.Tensor | None = None,
    time_stride: int | None = None,
    input_representation: str | None = None,
    target_representation: str | None = None,
    coordinate_frame: str | None = None,
) -> torch.Tensor:
    if isinstance(prediction, dict):
        if {"prediction", "ipd_prediction"}.intersection(prediction):
            raise ValueError(
                "SSL spatial loss rejects legacy prediction fields; use canonical 'dpipd'"
            )
        if "dpipd" not in prediction:
            raise ValueError("SSL spatial loss requires canonical 'dpipd' output")
        prediction = prediction["dpipd"]
    if target is not None and target_dpipd is not None:
        raise ValueError("SSL loss received both positional target and named target_dpipd")
    target = target if target is not None else target_dpipd
    if target is None:
        raise ValueError("SSL loss requires named target_dpipd")
    if not isinstance(target, torch.Tensor):
        raise TypeError("SSL DPIPD target must be a torch.Tensor")
    if input_lengths is not None and output_lengths is None:
        raise ValueError("SSL loss requires output_lengths when input_lengths is provided")
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("SSL DPIPD prediction must be a torch.Tensor")
    if prediction.shape != target.shape:
        raise ValueError(f"SSL prediction/target shape mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}")
    if prediction.ndim != 6 or prediction.shape[-1] != 2:
        raise ValueError("SSL prediction/target must have shape [B,T,F,K,P,2]")
    _validate_named_targets(prediction, target, doa, vad)
    _validate_length_metadata(
        input_lengths,
        batch=prediction.shape[0],
        frames=None,
        name="input_lengths",
    )
    _validate_length_metadata(
        output_lengths,
        batch=prediction.shape[0],
        frames=prediction.shape[1],
        name="output_lengths",
    )
    if any(value is not None for value in (
        pair_index, pair_mask, geometry, source_distances, frame_origin, time_stride,
        input_representation, target_representation, coordinate_frame,
    )):
        _validate_spatial_metadata(
            target,
            pair_index=pair_index,
            pair_mask=pair_mask,
            geometry=geometry,
            source_distances=source_distances,
            output_lengths=output_lengths,
            frame_origin=frame_origin,
            time_stride=time_stride,
            input_representation=input_representation,
            target_representation=target_representation,
            coordinate_frame=coordinate_frame,
        )
    if pair_mask is not None:
        if not torch.is_tensor(pair_mask) or pair_mask.dtype is not torch.bool:
            raise TypeError("SSL pair_mask must be a bool tensor")
        if pair_mask.shape != (prediction.shape[0], prediction.shape[4]):
            raise ValueError("SSL pair_mask must have shape [B,P]")
    mask = None
    if output_lengths is not None:
        if output_lengths.ndim != 1 or output_lengths.shape[0] != prediction.shape[0]:
            raise ValueError("SSL output_lengths must have shape (B,)")
        frame_ids = torch.arange(prediction.shape[1], device=prediction.device)[None, :]
        mask = (
            frame_ids < output_lengths.to(device=prediction.device)[:, None]
        )[:, :, None, None, None]
    if vad is not None:
        vad_mask = (vad.to(device=prediction.device) > 0.5)[:, :, None, :, None]
        mask = vad_mask if mask is None else mask & vad_mask
    values = prediction.sub(target).square().sum(dim=-1)
    if pair_mask is None:
        values = values.mean(dim=-1)
    else:
        pair_weights = pair_mask.to(device=values.device, dtype=values.dtype)
        pair_weights = pair_weights[:, None, None, None, :]
        values = (values * pair_weights).sum(dim=-1)
        values = values / pair_weights.sum(dim=-1).clamp_min(1.0)
    if not torch.isfinite(values).all():
        raise ValueError("SSL DPIPD MSE must be finite")
    return _masked_mean(values, mask[..., 0] if mask is not None else None)


def _frame_mask(lengths, frames: int, batch: int, device):
    if lengths is None:
        return torch.ones(batch, frames, dtype=torch.bool, device=device)
    if not torch.is_tensor(lengths) or lengths.ndim != 1 or lengths.shape[0] != batch:
        raise ValueError("SSL lengths must have shape [B]")
    if lengths.dtype not in _INTEGER_DTYPES:
        raise TypeError("SSL lengths must use an integer dtype")
    if bool(((lengths < 1) | (lengths > frames)).any().item()):
        raise ValueError(f"SSL lengths must be in [1, {frames}]")
    return torch.arange(frames, device=device)[None, :] < lengths.to(device=device)[:, None]


def _validate_length_metadata(
    lengths,
    *,
    batch: int,
    frames: int | None,
    name: str,
) -> None:
    """校验输入/输出帧长度，避免浮点或越界值被静默截断。"""
    if lengths is None:
        return
    if not torch.is_tensor(lengths) or lengths.ndim != 1 or lengths.shape[0] != batch:
        raise ValueError(f"SSL {name} must have shape [B]")
    if lengths.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"SSL {name} must use an integer dtype")
    if bool((lengths < 1).any().item()):
        raise ValueError(f"SSL {name} must contain positive values")
    if frames is not None and bool((lengths > frames).any().item()):
        raise ValueError(f"SSL {name} must be within [1, {frames}]")


def _validate_spatial_metadata(
    target: torch.Tensor,
    *,
    pair_index=None,
    pair_mask=None,
    geometry=None,
    source_distances=None,
    output_lengths=None,
    frame_origin=None,
    time_stride=None,
    input_representation=None,
    target_representation=None,
    coordinate_frame=None,
) -> None:
    """校验 mapping 所需的几何和时间字段，不在 loss 内重新推导它们。"""
    if target.ndim != 6:
        raise ValueError("SSL target_dpipd must have shape [B,T,F,K,P,2]")
    batch, frames, _, tracks, pairs, _ = target.shape
    if input_representation is not None and input_representation not in _INPUT_REPRESENTATIONS:
        raise ValueError(
            "SSL input_representation must be multichannel_stft_real_imag or reference_ipd"
        )
    if target_representation is not None and target_representation not in _TARGET_REPRESENTATIONS:
        raise ValueError(
            "SSL target_representation must be dpipd_far_field or dpipd_near_field"
        )
    if geometry is not None:
        if (
            not torch.is_tensor(geometry)
            or geometry.ndim != 3
            or geometry.shape[0] != batch
            or geometry.shape[1] < 2
            or geometry.shape[-1] != 3
        ):
            raise ValueError("SSL geometry must have shape [B,C,3]")
        if not torch.isfinite(geometry).all():
            raise ValueError("SSL geometry must be finite")
    if pair_index is not None:
        if not torch.is_tensor(pair_index) or pair_index.ndim != 3:
            raise ValueError("SSL pair_index must have shape [B,P,2]")
        if pair_index.shape != (batch, pairs, 2) or pair_index.dtype not in _INTEGER_DTYPES:
            raise ValueError("SSL pair_index must have integer shape [B,P,2]")
        if bool((pair_index[..., 0] == pair_index[..., 1]).any().item()):
            raise ValueError("SSL pair_index cannot contain self pairs")
        if geometry is not None:
            channels = geometry.shape[1]
            if channels < 2 or bool((pair_index < 0).any().item()) or bool((pair_index >= channels).any().item()):
                raise ValueError("SSL pair_index must index geometry channels")
    if pair_mask is not None and (
        not torch.is_tensor(pair_mask)
        or pair_mask.dtype is not torch.bool
        or pair_mask.shape != (batch, pairs)
    ):
        raise ValueError("SSL pair_mask must have bool shape [B,P]")
    if pair_mask is not None and not pair_mask.any(dim=1).all():
        raise ValueError("SSL pair_mask must keep at least one pair per item")
    if source_distances is not None:
        if not torch.is_tensor(source_distances):
            raise TypeError("SSL source_distances must be a torch.Tensor")
        if source_distances.shape != (batch, frames, tracks):
            raise ValueError("SSL source_distances must have shape [B,T,K]")
        if not torch.isfinite(source_distances).all():
            raise ValueError("SSL source_distances must be finite and strictly positive")
        valid_frames = _frame_mask(output_lengths, frames, batch, source_distances.device)
        if bool((source_distances[valid_frames] <= 0).any().item()):
            raise ValueError("SSL source_distances must be strictly positive on valid frames")
        padding = ~valid_frames
        if bool((source_distances[padding] != 0).any().item()):
            raise ValueError("SSL source_distances must be zero beyond output_lengths")
        if target_representation != "dpipd_near_field":
            raise ValueError(
                "SSL source_distances require target_representation=dpipd_near_field"
            )
    elif target_representation == "dpipd_near_field":
        raise ValueError(
            "SSL near-field targets require source_distances [B,T,K]"
        )
    if frame_origin is not None:
        if (
            not torch.is_tensor(frame_origin)
            or frame_origin.shape != (batch,)
            or frame_origin.dtype not in _INTEGER_DTYPES
        ):
            raise ValueError("SSL frame_origin must have integer shape [B]")
        if bool((frame_origin < 0).any().item()):
            raise ValueError("SSL frame_origin must be non-negative")
    if time_stride is not None and (type(time_stride) is not int or time_stride < 1):
        raise ValueError("SSL time_stride must be an exact positive integer")
    for name, value in (
        ("input_representation", input_representation),
        ("target_representation", target_representation),
        ("coordinate_frame", coordinate_frame),
    ):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"SSL {name} must be a non-empty string")


def _pairwise_complex_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    lengths=None,
    vad=None,
    pair_mask=None,
):
    """返回逐帧、逐预测轨道、逐目标轨道的复数 MSE。"""
    if prediction.ndim != 6 or target.ndim != 6:
        raise ValueError("SSL DPIPD tensors must be [B,T,F,K,P,2]")
    if prediction.shape[-1] != 2 or target.shape[-1] != 2:
        raise ValueError("SSL DPIPD tensors must use real/imag layout")
    if prediction.shape[:3] != target.shape[:3]:
        raise ValueError("SSL prediction and target must share B,T,F")
    b, frames, freqs = prediction.shape[:3]
    pred = prediction
    ref = target
    if pred.shape[3] < 1 or ref.shape[3] < 1 or pred.shape[4] < 1:
        raise ValueError("SSL DPIPD tensors must contain at least one track and pair")
    if pred.shape[4] != ref.shape[4]:
        raise ValueError("SSL prediction/target pair dimensions must agree")
    # [B,T,F,K_pred,K_ref,P,2] -> scalar MSE per track pair.
    diff = pred.unsqueeze(4) - ref.unsqueeze(3)
    pair_error = diff.square().sum(dim=-1)
    if pair_mask is None:
        values = pair_error.mean(dim=-1)
    else:
        pm = pair_mask.to(device=prediction.device) if torch.is_tensor(pair_mask) else None
        if pm is None or pm.dtype is not torch.bool:
            raise TypeError("SSL pair_mask must be a bool tensor")
        if pm.shape != (b, pred.shape[4]):
            raise ValueError("SSL pair_mask must have shape [B,P]")
        if not pm.any(dim=1).all():
            raise ValueError("SSL pair_mask must keep at least one pair per item")
        pair_weights = pm[:, None, None, None, None, :].to(pair_error.dtype)
        values = (pair_error * pair_weights).sum(dim=-1)
        values = values / pair_weights.sum(dim=-1).clamp_min(1.0)
    if not torch.isfinite(values).all():
        raise ValueError("SSL DPIPD pairwise MSE must be finite")
    frame = _frame_mask(lengths, frames, b, prediction.device)
    mask = frame[:, :, None, None].expand(b, frames, 1, ref.shape[3])
    if vad is not None:
        if vad.shape != (b, frames, ref.shape[3]):
            raise ValueError("SSL vad must have shape [B,T,K]")
        if not torch.isfinite(vad).all() or bool(((vad < 0) | (vad > 1)).any().item()):
            raise ValueError("SSL vad must be finite and in [0, 1]")
        mask = mask & (vad.to(device=prediction.device)[:, :, None, :] > 0.5)
    return values, mask


def _enumerate_min_assignment(cost_matrix: torch.Tensor) -> torch.Tensor:
    """枚举全部 ``K!`` 个排列求每个 ``[B,T]`` frame 的最小一一匹配。

    仅用于 ``K <= 2``（``2! = 2`` 时枚举比 Python 级 Hungarian 更快，且语义
    完全一致）。返回值 ``assignment[b, t, row]`` 是该 frame 中第 ``row`` 个
    预测轨道对齐到的目标轨道下标，与 :func:`_hungarian_min_assignment` 同构。
    """
    if cost_matrix.ndim != 4 or cost_matrix.shape[-1] != cost_matrix.shape[-2]:
        raise ValueError("SSL assignment cost must have shape [B,T,K,K]")
    batch, frames, tracks, _ = cost_matrix.shape
    if tracks < 1:
        raise ValueError("SSL assignment cost must contain at least one track")
    if not torch.isfinite(cost_matrix).all():
        raise ValueError("SSL assignment cost must be finite")
    # cost 的最后两轴是 (prediction track, target track)；先取出每个预测轨道的
    # 对角行 [B,T,K,K]，再按排列 gather 目标轨道并沿预测轨道求和，得到
    # [B,T,n_perms] 的排列代价。
    diagonal = cost_matrix[
        ..., torch.arange(tracks, device=cost_matrix.device), :
    ]
    totals = []
    for permutation in itertools.permutations(range(tracks)):
        terms = [diagonal[..., row, column] for row, column in enumerate(permutation)]
        totals.append(torch.stack(terms, dim=0).sum(dim=0))
    stacked = torch.stack(totals, dim=-1)
    best = stacked.argmin(dim=-1)
    return torch.as_tensor(
        list(itertools.permutations(range(tracks))), dtype=torch.long, device=cost_matrix.device
    )[best]


def _hungarian_min_assignment(cost_matrix: torch.Tensor) -> torch.Tensor:
    """用 Hungarian 算法求每个 ``[B,T]`` frame 的最小一一匹配。

    ``K >= 3`` 时排列枚举（``K!``）在 ``max_tracks=8`` 下会变成
    每帧 40320 次代价计算，必须换成 ``O(K^3)`` 的
    ``scipy.optimize.linear_sum_assignment``。求解在 detached 的 float64
    cost 上进行，只产生硬指派；调用方随后用原始 cost gather，因此选择本身
    不需要梯度，而被选中的预测项仍保留完整反向图（与旧 ``min`` 语义一致）。

    无效帧（全 inactive）不参与均值，但指派仍被求解，保证返回值在每个
    ``[B,T]`` 位置都有合法排列，DOA 辅助项可以直接 gather 使用。
    """
    from scipy.optimize import linear_sum_assignment

    if cost_matrix.ndim != 4 or cost_matrix.shape[-1] != cost_matrix.shape[-2]:
        raise ValueError("SSL assignment cost must have shape [B,T,K,K]")
    batch, frames, tracks, _ = cost_matrix.shape
    if tracks < 1:
        raise ValueError("SSL assignment cost must contain at least one track")
    if not torch.isfinite(cost_matrix).all():
        raise ValueError("SSL assignment cost must be finite")
    # scipy 只接受 CPU host 内存的 ndarray。无效帧已经在调用方通过 VAD/长度
    # mask 变成全零代价，因此这里无需按数值大小猜测“不可行”帧；有限但很大的
    # 合法代价仍必须交给 Hungarian，不能被替换成零矩阵破坏一一匹配。
    flat_host = cost_matrix.detach().to(device="cpu", dtype=torch.float64).reshape(-1, tracks, tracks)
    active = range(flat_host.shape[0])
    assignment_host = torch.zeros(
        (flat_host.shape[0], tracks), dtype=torch.long, device="cpu"
    )
    if flat_host.shape[0]:
        # linear_sum_assignment 不接受三维批量输入，只能按 frame 串行调用；
        # 训练中实测仍远快于 K>=6 的 K! 枚举。
        for row in active:
            rows, columns = linear_sum_assignment(flat_host[row].numpy())
            if len(rows) != tracks or len(set(columns.tolist())) != tracks:
                raise RuntimeError("SSL Hungarian assignment did not return a permutation")
            # scipy 以行下标升序返回配对，必须 scatter 回 rows，不能按位置写入。
            assignment_host[row, torch.as_tensor(rows, dtype=torch.long)] = torch.as_tensor(
                columns, dtype=torch.long
            )
    return assignment_host.to(device=cost_matrix.device).reshape(batch, frames, tracks)


def _batched_min_assignment(cost_matrix: torch.Tensor) -> torch.Tensor:
    """按轨道数选择最小值匹配求解器：``K <= 2`` 枚举，``K >= 3`` Hungarian。

    两者都返回 ``[B,T,K]`` 的硬指派，``assignment[b, t, row]`` 是第 ``row``
    个预测轨道对齐到的目标轨道。复杂度：``K <= 2`` 为 ``O(K! K)``，
    ``K >= 3`` 为 ``O(K^3)`` 每帧（无 ``K!`` 项）。
    """
    if cost_matrix.ndim != 4 or cost_matrix.shape[-1] != cost_matrix.shape[-2]:
        raise ValueError("SSL assignment cost must have shape [B,T,K,K]")
    tracks = cost_matrix.shape[-1]
    if tracks < 1:
        raise ValueError("SSL assignment cost must contain at least one track")
    if tracks <= 2:
        return _enumerate_min_assignment(cost_matrix)
    return _hungarian_min_assignment(cost_matrix)


def _pit_frame_objective(values, mask, *, max_tracks):
    """从 canonical pairwise cost 计算逐帧 PIT 代价、指派与有效帧。"""
    tracks = values.shape[-1]
    if tracks != values.shape[-2] or tracks > max_tracks:
        raise ValueError("SSL DPIPD prediction/target track dimensions are invalid")
    weighted_values = values * mask.unsqueeze(2).to(dtype=values.dtype)
    cost_matrix = weighted_values.sum(dim=2)
    assignment = _batched_min_assignment(cost_matrix)
    selected_cost = cost_matrix.gather(-1, assignment.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    frame_mass = mask.expand(
        values.shape[0], values.shape[1], values.shape[2], tracks
    ).sum(dim=(-1, -2)).clamp_min(1)
    best = selected_cost / frame_mass.to(dtype=selected_cost.dtype)
    valid_frames = mask.any(dim=(-1, -2))
    return best, assignment, valid_frames


@LOSS_REGISTRY.register("SSLDPIPDPITLoss")
class SSLDPIPDPITLoss(nn.Module):
    """FN-SSL 的逐帧 DP-IPD 复数 MSE 与 permutation-invariant 目标。

    source 指派在 cost 矩阵上按帧求解：``K <= 2`` 枚举全部排列，``K >= 3``
    使用 Hungarian。指派只在 detached cost 上求解，gather 回
    原始 cost 的项仍可微。

    复杂度与实测（Python 3.10 / torch 2.4 / scipy 1.15 / CPU，``F=257``、
    ``B=2``、``T=64``，暖机后取多次平均）：

    - ``K=4``：Hungarian 路径约 6 ms/forward（旧 ``K!`` 枚举约 6 ms）；
    - ``K=6``：Hungarian 路径约 9 ms/forward（旧枚举约 48 ms）；
    - ``K=8``：Hungarian 路径约 14 ms/forward（旧枚举约 3175 ms）。
    """

    SSL_SPATIAL_LOSS = True
    SSL_DPI_PIT_LOSS = True
    NAMED_TARGET_FIELDS = (
        "target_dpipd", "doa", "vad", "input_lengths", "output_lengths", "pair_mask", "pair_index",
        "geometry", "source_distances", "frame_origin", "time_stride",
    )
    DOA_OBJECTIVE_IMPLEMENTED = True

    def __init__(self, *, doa_loss_weight=0.0, max_tracks=8, **kwargs):
        super().__init__()
        if kwargs:
            raise TypeError(f"unexpected SSL DPIPD loss arguments: {sorted(kwargs)}")
        if isinstance(doa_loss_weight, bool) or not isinstance(doa_loss_weight, (int, float)):
            raise TypeError("doa_loss_weight must be numeric")
        if not math.isfinite(float(doa_loss_weight)) or float(doa_loss_weight) < 0:
            raise ValueError("doa_loss_weight must be finite and non-negative")
        if type(max_tracks) is not int or max_tracks < 1:
            raise ValueError("max_tracks must be a positive integer")
        self.doa_loss_weight = float(doa_loss_weight)
        self.max_tracks = max_tracks

    def validate_contract(self, prediction, **targets) -> None:
        """在同步输出预检中检查已启用的 DOA 辅助头和目标。"""
        if self.doa_loss_weight == 0.0:
            return
        fields = prediction if isinstance(prediction, dict) else {"dpipd": prediction}
        target = targets.get("target_dpipd", targets.get("target"))
        doa = targets.get("doa")
        vad = targets.get("vad")
        if not isinstance(fields, dict) or "dpipd" not in fields:
            raise ValueError("SSL DPIPD loss requires canonical 'dpipd' output")
        if "doa_vector" not in fields or target is None or doa is None:
            raise ValueError(
                "doa_loss_weight requires doa_vector output, target_dpipd and doa targets"
            )
        _validate_named_targets(fields["dpipd"], target, doa, vad)
        doa_prediction = fields["doa_vector"]
        if (
            not isinstance(doa_prediction, torch.Tensor)
            or not torch.is_floating_point(doa_prediction)
            or not torch.isfinite(doa_prediction).all()
            or doa_prediction.shape != doa.shape
        ):
            raise ValueError("SSL doa_vector output must be finite [B,T,K,2] matching doa")

    def forward(
        self,
        prediction,
        target=None,
        output_lengths=None,
        *,
        target_dpipd=None,
        input_lengths=None,
        doa=None,
        vad=None,
        pair_mask=None,
        pair_index=None,
        geometry=None,
        source_distances=None,
        frame_origin=None,
        time_stride=None,
        input_representation=None,
        target_representation=None,
        coordinate_frame=None,
        doa_prediction=None,
        return_assignment=False,
    ):
        if type(return_assignment) is not bool:
            raise TypeError("return_assignment must be a bool")
        if isinstance(prediction, dict):
            if {"prediction", "ipd_prediction"}.intersection(prediction):
                raise ValueError(
                    "SSL DPIPD loss rejects legacy prediction fields; use canonical 'dpipd'"
                )
            if "dpipd" not in prediction:
                raise ValueError("SSL DPIPD loss requires canonical 'dpipd' output")
            doa_prediction = prediction.get("doa_prediction", doa_prediction)
            prediction = prediction["dpipd"]
        if target is not None and target_dpipd is not None:
            raise ValueError("SSL DPIPD loss received both positional and named target")
        target = target if target is not None else target_dpipd
        if prediction is None or target is None:
            raise ValueError("SSL DPIPD PIT loss requires prediction and target_dpipd")
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("SSL DPIPD prediction must be a torch.Tensor")
        if not isinstance(target, torch.Tensor):
            raise TypeError("SSL DPIPD target must be a torch.Tensor")
        if input_lengths is not None and output_lengths is None:
            raise ValueError(
                "SSL DPIPD loss requires output_lengths when input_lengths is provided"
            )
        if prediction.ndim != 6 or prediction.shape[-1] != 2:
            raise ValueError(
                "SSL DPIPD prediction must have shape [B,T,F,K,P,2] "
                "with real/imag layout"
            )
        if target.ndim != 6 or target.shape[-1] != 2:
            raise ValueError("SSL DPIPD target must use real/imag layout")
        if prediction.shape[:3] != target.shape[:3]:
            raise ValueError(
                "SSL DPIPD prediction/target must share B,T,F and have identical shape"
            )
        if prediction.shape[3] != target.shape[3]:
            raise ValueError("SSL DPIPD prediction/target track dimensions are invalid")
        if prediction.shape[4] != target.shape[4]:
            raise ValueError("SSL DPIPD prediction/target pair dimensions are invalid")
        _validate_named_targets(prediction, target, doa, vad)
        _validate_length_metadata(
            input_lengths,
            batch=prediction.shape[0],
            frames=None,
            name="input_lengths",
        )
        _validate_length_metadata(
            output_lengths,
            batch=prediction.shape[0],
            frames=prediction.shape[1],
            name="output_lengths",
        )
        _validate_spatial_metadata(
            target,
            pair_index=pair_index,
            pair_mask=pair_mask,
            geometry=geometry,
            source_distances=source_distances,
            output_lengths=output_lengths,
            frame_origin=frame_origin,
            time_stride=time_stride,
            input_representation=input_representation,
            target_representation=target_representation,
            coordinate_frame=coordinate_frame,
        )
        values, mask = _pairwise_complex_mse(
            prediction, target, lengths=output_lengths, vad=vad, pair_mask=pair_mask
        )
        tracks = values.shape[-1]
        # target VAD mask 只依赖目标轨道，因此所有 assignment 的 frame mass
        # 相同；先沿频率聚合成 KxK cost，K<=2 枚举、K>=3 Hungarian。
        best, assignment, valid_frames = _pit_frame_objective(
            values, mask, max_tracks=self.max_tracks
        )
        # 无任何有效 target（全 inactive 或超出长度）的帧整帧剔除，不参与均值分母。
        base = _masked_mean(best, valid_frames)
        if self.doa_loss_weight:
            if doa_prediction is None or doa is None:
                raise ValueError("doa_loss_weight requires doa_prediction and doa targets")
            if (
                not isinstance(doa_prediction, torch.Tensor)
                or not torch.is_floating_point(doa_prediction)
                or not torch.isfinite(doa_prediction).all()
            ):
                raise ValueError("SSL DOA prediction must be finite floating-point")
            if doa_prediction.shape != doa.shape:
                raise ValueError("SSL DOA prediction/target shape mismatch")
            # DOA 辅助项必须沿用同一 assignment。assignment 的每一行是
            # prediction track 对应的 target track，故把 target DOA gather 到预测顺序。
            aligned_doa = doa.gather(
                2,
                assignment.to(device=doa.device).unsqueeze(-1).expand(
                    -1, -1, -1, doa.shape[-1]
                ),
            )
            doa_mask = _frame_mask(
                output_lengths, doa.shape[1], doa.shape[0], doa.device
            )
            doa_mask = doa_mask[:, :, None].expand(
                doa.shape[0], doa.shape[1], tracks
            )
            if vad is not None:
                aligned_vad = vad.to(device=doa.device).gather(
                    2, assignment.to(device=doa.device)
                )
                doa_mask = doa_mask & (aligned_vad > 0.5)
            base = base + self.doa_loss_weight * _circular_doa_loss(
                doa_prediction, aligned_doa, doa_mask
            )
        if return_assignment:
            return base, assignment, valid_frames
        return base


@LOSS_REGISTRY.register("SSLDOACompositeLoss")
class SSLDOACompositeLoss(nn.Module):
    """复用 DP-IPD PIT 指派的 SSL 多项监督损失。

    默认只计算 canonical DP-IPD PIT，与 ``SSLDPIPDPITLoss`` 保持相同数值语义。
    所有辅助目标都使用 DP-IPD PIT 得到的逐帧预测轨道到目标轨道映射；不会为
    DOA、VAD 或 range 单独匹配。DOA角使用 canonical ``(polar, azimuth)`` 弧度，
    即便字段名沿用历史 ``doa_vector``，它仍是 ``[B,T,K,2]`` 两角格式。
    """

    SSL_SPATIAL_LOSS = True
    SSL_DPI_PIT_LOSS = True
    SSL_COMPOSITE_LOSS = True
    NAMED_TARGET_FIELDS = SSLDPIPDPITLoss.NAMED_TARGET_FIELDS
    DOA_OBJECTIVE_IMPLEMENTED = True

    def __init__(
        self,
        *,
        dpipd_weight=1.0,
        doa_classification_weight=0.0,
        doa_regression_weight=0.0,
        vad_weight=0.0,
        range_weight=0.0,
        doa_class_bins=None,
        max_tracks=8,
    ):
        super().__init__()
        weights = {}
        for name, value in (
            ("dpipd_weight", dpipd_weight),
            ("doa_classification_weight", doa_classification_weight),
            ("doa_regression_weight", doa_regression_weight),
            ("vad_weight", vad_weight),
            ("range_weight", range_weight),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            weights[name] = numeric
        if not any(value > 0.0 for value in weights.values()):
            raise ValueError("SSLDOACompositeLoss requires at least one positive loss weight")
        if weights["doa_classification_weight"] and weights["doa_regression_weight"]:
            raise ValueError(
                "DOA classification and regression are mutually exclusive contracts"
            )
        if type(max_tracks) is not int or max_tracks < 1:
            raise ValueError("max_tracks must be an exact positive integer")

        bins = None
        if doa_class_bins is not None:
            if _contains_boolean(doa_class_bins):
                raise TypeError("doa_class_bins must not contain boolean values")
            raw_bins = torch.as_tensor(doa_class_bins)
            if raw_bins.dtype == torch.bool or raw_bins.is_complex():
                raise TypeError("doa_class_bins must contain real numeric angles")
            bins = raw_bins.to(dtype=torch.float32)
            if bins.ndim != 2 or bins.shape[1] != 2 or bins.shape[0] < 2:
                raise ValueError("doa_class_bins must have shape [Q,2] with Q >= 2")
            if not torch.isfinite(bins).all():
                raise ValueError("doa_class_bins must contain finite angles")
            if bool(((bins[:, 0] < 0.0) | (bins[:, 0] > math.pi)).any().item()):
                raise ValueError("doa_class_bins polar angles must be in [0, pi]")
            if bool(((bins[:, 1] < -math.pi) | (bins[:, 1] > math.pi)).any().item()):
                raise ValueError("doa_class_bins azimuth angles must be in [-pi, pi]")
            bin_vectors = _doa_unit_vector(bins)
            similarity = bin_vectors @ bin_vectors.transpose(0, 1)
            duplicates = torch.triu(similarity >= 1.0 - 1e-6, diagonal=1)
            if bool(duplicates.any().item()):
                raise ValueError("doa_class_bins must represent distinct spherical directions")
            if weights["doa_classification_weight"] == 0.0:
                raise ValueError(
                    "doa_class_bins is only valid when doa_classification_weight > 0"
                )
        elif weights["doa_classification_weight"] > 0.0:
            raise ValueError(
                "doa_classification_weight requires explicit doa_class_bins [Q,2]"
            )

        self.weights = weights
        self.max_tracks = max_tracks
        self.dpipd_loss = SSLDPIPDPITLoss(
            doa_loss_weight=0.0,
            max_tracks=max_tracks,
        )
        self.register_buffer(
            "doa_class_bins",
            None if bins is None else bins.detach().clone(),
            persistent=False,
        )

    @staticmethod
    def _prediction_fields(prediction):
        if not isinstance(prediction, dict):
            return {"dpipd": prediction}
        legacy = {"prediction", "ipd_prediction"}.intersection(prediction)
        if legacy:
            raise ValueError(
                "SSL composite loss rejects legacy prediction fields; use canonical 'dpipd'"
            )
        allowed = {"dpipd", "doa_logits", "doa_vector", "vad_logits", "range"}
        unknown = set(prediction) - allowed
        if unknown:
            raise ValueError(f"SSL composite loss received unknown prediction fields: {sorted(unknown)}")
        if "dpipd" not in prediction:
            raise ValueError("SSL composite loss requires canonical 'dpipd' output")
        return prediction

    def validate_contract(self, prediction, **targets) -> None:
        """在 loss collective 前检查启用的 head 和对应 supervision 是否齐全。"""
        fields = self._prediction_fields(prediction)
        if targets.get("target_dpipd") is None:
            raise ValueError("SSLDOACompositeLoss requires target_dpipd")
        target_dpipd = targets["target_dpipd"]
        _validate_named_targets(
            fields["dpipd"], target_dpipd, targets.get("doa"), targets.get("vad")
        )
        required = (
            ("doa_classification_weight", "doa_logits", "doa"),
            ("doa_regression_weight", "doa_vector", "doa"),
            ("vad_weight", "vad_logits", "vad"),
            ("range_weight", "range", "source_distances"),
        )
        for weight_name, prediction_name, target_name in required:
            if self.weights[weight_name] == 0.0:
                continue
            if fields.get(prediction_name) is None:
                raise ValueError(
                    f"{weight_name} requires model output {prediction_name!r}"
                )
            if targets.get(target_name) is None:
                raise ValueError(f"{weight_name} requires target {target_name!r}")
        if any(
            self.weights[name] > 0.0
            for name in (
                "doa_classification_weight",
                "doa_regression_weight",
                "range_weight",
            )
        ) and targets.get("vad") is None:
            raise ValueError("SSL DOA/range objectives require source VAD targets")
        if self.weights["range_weight"] > 0.0:
            if targets.get("target_representation") != "dpipd_near_field":
                raise ValueError(
                    "range_weight requires target_representation='dpipd_near_field'"
                )
        _validate_spatial_metadata(
            target_dpipd,
            pair_index=targets.get("pair_index"),
            pair_mask=targets.get("pair_mask"),
            geometry=targets.get("geometry"),
            source_distances=targets.get("source_distances"),
            output_lengths=targets.get("output_lengths"),
            frame_origin=targets.get("frame_origin"),
            time_stride=targets.get("time_stride"),
            input_representation=targets.get("input_representation"),
            target_representation=targets.get("target_representation"),
            coordinate_frame=targets.get("coordinate_frame"),
        )
        batch, frames, _, tracks, _, _ = target_dpipd.shape
        if self.weights["doa_classification_weight"] > 0.0:
            logits = fields["doa_logits"]
            if not torch.is_tensor(logits):
                raise TypeError("doa_logits must be a torch.Tensor")
            if logits.ndim != 4 or tuple(logits.shape[:3]) != (batch, frames, tracks):
                raise ValueError("doa_logits must have shape [B,T,K,Q]")
            if logits.shape[-1] != self.doa_class_bins.shape[0]:
                raise ValueError(
                    "doa_logits class dimension must match doa_class_bins"
                )
            if not torch.is_floating_point(logits) or not torch.isfinite(logits).all():
                raise ValueError("doa_logits must be finite floating-point")
        expected_shapes = {
            "doa_vector": (batch, frames, tracks, 2),
            "vad_logits": (batch, frames, tracks),
            "range": (batch, frames, tracks),
        }
        for weight_name, output_name in (
            ("doa_regression_weight", "doa_vector"),
            ("vad_weight", "vad_logits"),
            ("range_weight", "range"),
        ):
            if self.weights[weight_name] == 0.0:
                continue
            output = fields[output_name]
            if not torch.is_tensor(output):
                raise TypeError(f"{output_name} must be a torch.Tensor")
            if tuple(output.shape) != expected_shapes[output_name]:
                raise ValueError(
                    f"{output_name} must have shape {expected_shapes[output_name]}"
                )
            if not torch.is_floating_point(output) or not torch.isfinite(output).all():
                raise ValueError(f"{output_name} must be finite floating-point")
            if output_name == "range":
                valid_frames = _frame_mask(
                    targets.get("output_lengths"),
                    frames,
                    batch,
                    output.device,
                )
                valid_slots = valid_frames[:, :, None].expand_as(output)
                if bool((output[valid_slots] <= 0.0).any().item()):
                    raise ValueError(
                        "range must be strictly positive for every slot on valid output frames"
                    )

    def forward(
        self,
        prediction,
        target=None,
        output_lengths=None,
        **kwargs,
    ):
        if target is not None and "target_dpipd" in kwargs:
            raise ValueError("SSL composite loss received both positional and named target")
        if target is not None:
            kwargs["target_dpipd"] = target
        if output_lengths is not None:
            if "output_lengths" in kwargs:
                raise ValueError("SSL composite loss received output_lengths twice")
            kwargs["output_lengths"] = output_lengths
        allowed = set(SSLDPIPDPITLoss.NAMED_TARGET_FIELDS) | {
            "input_representation",
            "target_representation",
            "coordinate_frame",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise TypeError(f"unexpected SSL composite loss inputs: {sorted(unknown)}")
        fields = self._prediction_fields(prediction)
        self.validate_contract(fields, **kwargs)
        dpipd_loss, assignment, _ = self.dpipd_loss(
            fields["dpipd"], return_assignment=True, **kwargs
        )
        result = self.weights["dpipd_weight"] * dpipd_loss
        target_dpipd = kwargs.get("target_dpipd")
        batch, frames, _, tracks, _, _ = target_dpipd.shape
        output_lengths = kwargs.get("output_lengths")
        frame_mask = _frame_mask(output_lengths, frames, batch, target_dpipd.device)
        pair_mask = kwargs.get("pair_mask")
        if pair_mask is None:
            pair_valid = torch.ones(batch, dtype=torch.bool, device=target_dpipd.device)
        else:
            pair_valid = pair_mask.to(device=target_dpipd.device).any(dim=1)
        valid_mask = frame_mask & pair_valid[:, None]

        target_vad = kwargs.get("vad")
        aligned_vad = None
        if target_vad is not None:
            aligned_vad = target_vad.to(device=assignment.device).gather(2, assignment)
        source_mask = valid_mask[:, :, None].expand(batch, frames, tracks)
        active_mask = source_mask
        if aligned_vad is not None:
            active_mask = active_mask & (aligned_vad > 0.5)

        doa = kwargs.get("doa")
        if self.weights["doa_classification_weight"] > 0.0:
            aligned_doa = doa.to(device=assignment.device).gather(
                2, assignment.unsqueeze(-1).expand(-1, -1, -1, 2)
            )
            target_vectors = _doa_unit_vector(aligned_doa)
            bins = self.doa_class_bins.to(device=target_vectors.device, dtype=target_vectors.dtype)
            bin_vectors = _doa_unit_vector(bins)
            class_target = torch.einsum("btkd,qd->btkq", target_vectors, bin_vectors).argmax(-1)
            logits = fields["doa_logits"]
            if logits.shape[:3] != (batch, frames, tracks) or logits.shape[-1] != bins.shape[0]:
                raise ValueError("doa_logits must have shape [B,T,K,Q] matching doa_class_bins")
            if not torch.is_floating_point(logits) or not torch.isfinite(logits).all():
                raise ValueError("doa_logits must be finite floating-point")
            values = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                class_target.reshape(-1),
                reduction="none",
            ).reshape(batch, frames, tracks)
            result = result + self.weights["doa_classification_weight"] * _masked_mean(
                values, active_mask
            )

        if self.weights["doa_regression_weight"] > 0.0:
            aligned_doa = doa.to(device=assignment.device).gather(
                2, assignment.unsqueeze(-1).expand(-1, -1, -1, 2)
            )
            doa_prediction = fields["doa_vector"]
            if doa_prediction.shape != aligned_doa.shape:
                raise ValueError("doa_vector must have canonical [B,T,K,2] angle shape")
            if not torch.is_floating_point(doa_prediction) or not torch.isfinite(doa_prediction).all():
                raise ValueError("doa_vector must be finite floating-point")
            result = result + self.weights["doa_regression_weight"] * _circular_doa_loss(
                doa_prediction,
                aligned_doa,
                active_mask,
            )

        if self.weights["vad_weight"] > 0.0:
            logits = fields["vad_logits"]
            if logits.shape != (batch, frames, tracks):
                raise ValueError("vad_logits must have shape [B,T,K]")
            if not torch.is_floating_point(logits) or not torch.isfinite(logits).all():
                raise ValueError("vad_logits must be finite floating-point")
            aligned_vad = target_vad.to(device=logits.device).gather(2, assignment.to(logits.device))
            values = F.binary_cross_entropy_with_logits(
                logits, aligned_vad.to(dtype=logits.dtype), reduction="none"
            )
            result = result + self.weights["vad_weight"] * _masked_mean(values, source_mask)

        if self.weights["range_weight"] > 0.0:
            predicted_range = fields["range"]
            target_range = kwargs["source_distances"].to(device=assignment.device).gather(
                2, assignment
            )
            if predicted_range.shape != (batch, frames, tracks):
                raise ValueError("range must have shape [B,T,K]")
            if not torch.is_floating_point(predicted_range) or not torch.isfinite(predicted_range).all():
                raise ValueError("range must be finite floating-point in metres")
            safe_prediction = torch.where(
                active_mask, predicted_range, torch.ones_like(predicted_range)
            )
            safe_target = torch.where(
                active_mask, target_range.to(predicted_range), torch.ones_like(predicted_range)
            )
            values = F.smooth_l1_loss(
                safe_prediction.log(), safe_target.log(), reduction="none"
            )
            result = result + self.weights["range_weight"] * _masked_mean(values, active_mask)

        if not torch.isfinite(result).all():
            raise ValueError("SSL composite loss must be finite")
        return result


@LOSS_REGISTRY.register("SSLSpatialMSELoss")
class SSLSpatialMSELoss(nn.Module):
    """VAD-gated IPD MSE；DOA 只校验，当前没有 DOA prediction head。"""

    SSL_SPATIAL_LOSS = True
    NAMED_TARGET_FIELDS = (
        "target_dpipd", "doa", "vad", "input_lengths", "output_lengths", "pair_mask", "pair_index",
        "geometry", "source_distances", "frame_origin", "time_stride",
    )
    DOA_OBJECTIVE_IMPLEMENTED = False

    def __init__(self, *, n_fft=None, hop_length=None, win_length=None, window=None,
                 center=None, normalized=None, allow_no_stft=False, doa_loss_weight=0.0):
        super().__init__()
        if not isinstance(allow_no_stft, bool):
            raise TypeError("allow_no_stft must be a bool")
        if not allow_no_stft:
            raise ValueError("SSLSpatialMSELoss does not use an STFT; set allow_no_stft=true")
        if isinstance(doa_loss_weight, bool) or not isinstance(doa_loss_weight, (int, float)):
            raise TypeError("doa_loss_weight must be numeric")
        if float(doa_loss_weight) != 0.0:
            raise NotImplementedError("SSL DOA loss requires a model DOA prediction head")
        self.stft = {
            key: value for key, value in {
                "n_fft": n_fft, "hop_length": hop_length, "win_length": win_length,
                "window": window, "center": center, "normalized": normalized,
            }.items() if value is not None
        }
        self.allow_no_stft = allow_no_stft

    def forward(
        self,
        prediction,
        target=None,
        output_lengths=None,
        *,
        target_dpipd=None,
        input_lengths=None,
        doa=None,
        vad=None,
        pair_mask=None,
        pair_index=None,
        geometry=None,
        source_distances=None,
        frame_origin=None,
        time_stride=None,
        input_representation=None,
        target_representation=None,
        coordinate_frame=None,
    ):
        return ipd_mse_loss(
            prediction,
            target,
            output_lengths,
            target_dpipd=target_dpipd,
            input_lengths=input_lengths,
            doa=doa,
            vad=vad,
            pair_mask=pair_mask,
            pair_index=pair_index,
            geometry=geometry,
            source_distances=source_distances,
            frame_origin=frame_origin,
            time_stride=time_stride,
            input_representation=input_representation,
            target_representation=target_representation,
            coordinate_frame=coordinate_frame,
        )
