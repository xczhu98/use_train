"""SE 与 AEC 的样本元组到结构化 Batch 的 collate 合同。

**SE / offline / dynamic（tuple）**
  ``__getitem__`` 返回 4/7 元组；共享 ``collate_fn`` 输出 SEBatch。

**AEC（tuple）**
  ``aec_collate_fn`` 接受 7/11/12 元组（7=offline 无 flags，11=标准，
  12=含帧级 ``state`` label），输出已校验的 AECBatch。
  ``fs`` 是标量 ``int32`` tensor，``lengths`` 是 ``[B]`` ``int32`` tensor。
"""

from __future__ import annotations

import logging
import math
from numbers import Real
from typing import Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


from core.data.collate import _to_1d_wav, _to_spatial_noisy
from tasks.aec.data.batch import AECBatch


# Re-export SE collate_fn to avoid duplication; AEC-specific logic is in aec_collate_fn below.
from core.data.collate import collate_fn


def _aec_silent_placeholder_batch(
    batch_size: int,
    placeholder_samples: int,
    *,
    placeholder_sr: int = 16000,
    with_state: bool = False,
) -> AECBatch:
    """Return a silent placeholder batch when every sample in a batch failed to load.

    """
    T = max(1, int(placeholder_samples))
    nearend = torch.zeros(batch_size, T, dtype=torch.float32)
    mic = torch.zeros(batch_size, T, dtype=torch.float32)
    farend = torch.zeros(batch_size, T, dtype=torch.float32)
    error = torch.zeros(batch_size, T, dtype=torch.float32)
    lengths = torch.full((batch_size,), T, dtype=torch.int32)
    # 与正常 collate 路径一致：fs 为标量 tensor
    fs = torch.tensor(int(placeholder_sr), dtype=torch.int32)
    z_mask = torch.zeros(batch_size, dtype=torch.bool)
    pn = torch.zeros(batch_size, dtype=torch.bool)
    pe = torch.zeros(batch_size, dtype=torch.bool)
    pen = torch.zeros(batch_size, dtype=torch.bool)
    he = torch.zeros(batch_size, dtype=torch.bool)
    out = AECBatch(
        nearend=nearend, mic=mic, farend=farend, fs=fs, lengths=lengths,
        error=error, error_mask=z_mask, is_pure_noise=pn, is_pure_echo=pe,
        is_pure_echo_noise=pen, has_echo=he,
        state=torch.full((batch_size, 1), -1, dtype=torch.int64) if with_state else None,
    )
    out.validate()
    return out


def _validate_state_label(v: object, *, idx: int) -> torch.Tensor:
    """校验帧级 state 为互斥类别索引 ``(T,) int64``。"""
    if not isinstance(v, torch.Tensor):
        raise ValueError(
            f"batch[{idx}] state must be a torch.Tensor, got {type(v)}"
        )
    if v.dim() == 1 and v.dtype == torch.int64:
        return v
    raise ValueError(
        f"batch[{idx}] state must be int64 (T,) mutually-exclusive class "
        f"indices; multi-label attributes are unsupported, got "
        f"shape={tuple(v.shape)}, dtype={v.dtype}"
    )


def _one_integer_scalar(
    v: object, *, name: str, idx: int, require_positive: bool = True
) -> int:
    """校验采样率和长度是精确整数标量，禁止截断小数。"""
    try:
        scalar = torch.as_tensor(v)
    except (OverflowError, RuntimeError, TypeError, ValueError) as exc:
        raise TypeError(f"batch[{idx}] {name} must be an integer scalar") from exc
    if scalar.numel() != 1 or scalar.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            f"batch[{idx}] {name} must be an integer scalar, "
            f"got shape={tuple(scalar.shape)}, dtype={scalar.dtype}"
        )
    result = int(scalar.item())
    if require_positive and result < 1:
        raise ValueError(f"batch[{idx}] {name} must be >= 1, got {result}")
    return result


def _one_bool_flag(v: object, *, name: str, idx: int) -> bool:
    """校验 bool 字段只接受标量 bool 或有限数值 0/1。"""
    try:
        flag = torch.as_tensor(v)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"batch[{idx}] {name} flag must be a strict bool or finite numeric 0/1"
        ) from exc
    if flag.dim() != 0:
        raise TypeError(
            f"batch[{idx}] {name} flag must be scalar, got shape {tuple(flag.shape)}"
        )
    value = flag.item()
    if isinstance(value, bool):
        return value
    if (
        isinstance(value, Real)
        and math.isfinite(float(value))
        and value in (0, 1)
    ):
        return bool(value)
    raise ValueError(
        f"batch[{idx}] {name} flag must be a strict bool or finite numeric 0/1"
    )


def aec_collate_fn(
    batch: list,
    *,
    placeholder_sr: int = 16000,
    placeholder_samples: int = 16000,
    raise_on_all_failed: bool = False,
    expect_state: bool = False,
    check_finite: bool = True,
) -> AECBatch:
    """Pad AEC tuples; spatial ``mic`` is preserved as ``[B,C,T_max]``.

    Nearend, farend, error and target-facing outputs remain mono.

    Expected per-sample tuple (current format):

    .. code-block:: text

        (nearend, mic, farend, fs, length,
         error, error_mask,
         is_pure_noise, is_pure_echo, is_pure_echo_noise, has_echo
         [, state])

    ``state`` at index 11 is an optional frame-level state label for models such as
    DBGNet. It must be a 1-D ``torch.int64`` tensor of shape ``(T,)`` containing
    mutually-exclusive class indices; padding value is ``-1``. Multi-label
    attributes are rejected because the current head, loss and metrics all use
    softmax class-index semantics. If any sample has ``state``, every sample in
    that batch must use the 12-tuple schema; missing supervision is rejected.

    Supported tuple lengths:
      - 7-tuple: offline/precomputed-residual capability without pure-branch flags.
      - 11-tuple: 7-tuple with pure-branch flags
        (is_pure_noise/is_pure_echo/is_pure_echo_noise) and ``has_echo``
        at index 10 (current dynamic format).
      - 12-tuple: 11-tuple with an optional frame-level ``state`` label
        at index 11 for models such as DBGNet.

    All other lengths (including legacy 5/6/8/9/10 and the legacy 13-tuple
    with trailing ``clean_vad_mask``) are explicitly rejected instead of
    being silently truncated or ignored.

    输出已校验的 AECBatch；``fs`` 为标量 ``torch.int32``，
    ``lengths`` 为 ``[B]`` ``torch.int32``，可选 ``state`` 为 ``[B,T_state]``。
    ``error`` is an optional time-domain residual supplied by an offline track or
    an upstream runtime frontend and is padded identically to the other waveforms.
    ``error_mask`` is ``True`` when the sample carries a real residual and
    ``False`` for zero-filled fallbacks (e.g. SE-only branches).
    ``fs`` and ``length`` must be exact positive integer scalars; boolean fields
    accept only scalar bool or finite numeric 0/1 values.
    ``has_echo`` is ``True`` when the sample contains a disk-manifest echo
    component; used by the trainer for precise ERLE sample collection.
    ``expect_state`` 仅为 all-None fallback 提供无样本可推断时的 schema 预期：
    启用时返回全 ``-1`` 的第 12 位 state，关闭时保持旧 11-tuple。

    部分 None 样本直接拒绝，不缩小 batch；这避免 DDP 各 rank 的有效样本数
    不一致。单卡的全 None fallback 保留旧兼容语义，DDP 由 Entry 强制 fail-fast。

    ``check_finite`` 控制逐样本四轨 ``isfinite`` 扫描：默认开启，为自定义
    Dataset 保留 NaN/Inf 兜底；当上游 Dataset 已在 ``strict_load=True`` 下校验
    过波形时，入口可关闭它以省掉 collate 热路径上的 O(T) 扫描。
    """
    _ = placeholder_sr
    n_in = len(batch)
    none_count = sum(s is None for s in batch)
    if 0 < none_count < n_in:
        raise RuntimeError(
            f"aec_collate_fn: {none_count}/{n_in} samples were None; refusing to "
            "shrink a training batch because rank-local filtering breaks DDP "
            "sample weighting. Fix data paths or use dataset placeholder handling."
        )
    batch = [] if none_count == n_in else list(batch)

    # 预扫描 tuple/state schema，避免无效 state 被静默当作缺失。
    state_samples: list[torch.Tensor] = []
    state_presence: list[bool] = []
    for i, sample in enumerate(batch):
        if not isinstance(sample, tuple):
            raise TypeError(f"batch[{i}] must be tuple, got {type(sample)}")
        if len(sample) not in (7, 11, 12):
            raise ValueError(
                f"batch[{i}] tuple length must be one of 7/11/12, got {len(sample)}. "
                f"Legacy 5/6/10/13-tuples, 8/9-tuples and unknown extensions are "
                f"rejected instead of being silently truncated or ignored. Ensure "
                f"the offline dataset returns 7-tuple and DynamicMixingDataset "
                f"returns 11/12-tuple with explicit error_mask."
            )
        has_state = len(sample) == 12
        state_presence.append(has_state)
        if has_state:
            state_samples.append(_validate_state_label(sample[11], idx=i))

    if any(state_presence) and not all(state_presence):
        missing = [i for i, present in enumerate(state_presence) if not present]
        raise ValueError(
            "AEC batch mixes samples with and without frame-level state labels; "
            f"12-tuple state is required for every sample when enabled (missing at {missing})"
        )

    has_state_input = bool(state_samples)

    if len(batch) == 0:
        if raise_on_all_failed:
            raise RuntimeError(
                f"aec_collate_fn: all {n_in} samples in batch were None (load/__getitem__ failure). "
                "Fix data paths or set data.strict_load / check worker logs."
            )
        bsz = max(1, int(n_in))
        logger.warning(
            "aec_collate_fn: all samples None — returning silent placeholder [%d, %d]",
            bsz,
            placeholder_samples,
        )
        return _aec_silent_placeholder_batch(
            bsz,
            placeholder_samples,
            placeholder_sr=placeholder_sr,
            with_state=expect_state,
        )

    rows: list[tuple] = []
    states: list[torch.Tensor] = []
    for i, s in enumerate(batch):
        # 新合同下长度仅 7/11/12：pure flags 与 has_echo 同时存在（11/12）或同时缺席（7）。
        has_flags = len(s) >= 11
        # 按长度守卫，避免访问不存在的 has_echo 位。
        has_he = len(s) >= 11
        nearend = _to_1d_wav(s[0], name="nearend", idx=i)
        farend = _to_1d_wav(s[2], name="farend", idx=i)
        fs_val = _one_integer_scalar(s[3], name="fs", idx=i)
        length = _one_integer_scalar(
            s[4], name="length", idx=i, require_positive=False
        )
        mic = _to_spatial_noisy(
            s[1],
            name="mic",
            idx=i,
            declared_length=length,
            keep_single_channel=False,
        )

        t_nearend = int(nearend.shape[-1])
        t_mic = int(mic.shape[-1])
        t_farend = int(farend.shape[-1])
        if t_nearend != t_mic or t_farend != t_mic:
            raise RuntimeError(
                f"batch[{i}] length mismatch: mic T={t_mic}, nearend T={t_nearend}, "
                f"farend T={t_farend}"
            )
        if t_mic <= 0:
            raise RuntimeError(f"batch[{i}] has empty waveforms")

        err = s[5]
        if not isinstance(err, torch.Tensor):
            raise TypeError(f"batch[{i}] error must be torch.Tensor, got {type(err)}")
        err = _to_1d_wav(err, name="error", idx=i)
        t_error = int(err.shape[-1])
        if t_error != t_mic:
            raise RuntimeError(
                f"batch[{i}] length mismatch: mic T={t_mic}, error T={t_error}"
            )
        if length < 1 or length > t_mic:
            raise RuntimeError(
                f"batch[{i}] invalid declared length={length}; expected 1 <= declared "
                f"length <= waveform length={t_mic}"
            )
        if check_finite and not all(
            torch.isfinite(value).all() for value in (nearend, mic, farend, err)
        ):
            raise RuntimeError(
                f"batch[{i}] contains NaN/Inf (strict_load=True 时应在 Dataset 已拦截; "
                "检查自定义 Dataset)"
            )

        err_mask = _one_bool_flag(s[6], name="error_mask", idx=i)
        pn = (
            _one_bool_flag(s[7], name="is_pure_noise", idx=i)
            if has_flags
            else False
        )
        pe = (
            _one_bool_flag(s[8], name="is_pure_echo", idx=i)
            if has_flags
            else False
        )
        pen = (
            _one_bool_flag(s[9], name="is_pure_echo_noise", idx=i)
            if has_flags
            else False
        )
        he = (
            _one_bool_flag(s[10], name="has_echo", idx=i)
            if has_he
            else False
        )

        rows.append((nearend, mic, farend, fs_val, length, err, err_mask, pn, pe, pen, he))

        if has_state_input:
            states.append(s[11])

    # 每个样本内部四轨等长；不同样本可预填充到不同物理长度，统一补到
    # batch 最大长度，有效区间由 declared lengths 指定。
    mx = max(int(r[1].shape[-1]) for r in rows)
    mic_layouts = {tuple(r[1].shape[:-1]) for r in rows}
    if len(mic_layouts) != 1:
        raise RuntimeError(
            f"AEC microphone channel layouts must match within a batch, got {mic_layouts}"
        )
    lengths = torch.tensor([r[4] for r in rows], dtype=torch.int32)
    fs_rows = [r[3] for r in rows]
    if not all(f == fs_rows[0] for f in fs_rows):
        raise RuntimeError(f"Sample rates must match within a batch, got {set(fs_rows)}")

    def _stack_pad(idx: int) -> torch.Tensor:
        return torch.stack([F.pad(r[idx], (0, mx - r[idx].shape[-1])) for r in rows])

    out = AECBatch(
        nearend=_stack_pad(0),
        mic=_stack_pad(1),
        farend=_stack_pad(2),
        fs=torch.tensor(fs_rows[0], dtype=torch.int32),
        lengths=lengths,
        error=_stack_pad(5),
        error_mask=torch.tensor([r[6] for r in rows], dtype=torch.bool),
        is_pure_noise=torch.tensor([r[7] for r in rows], dtype=torch.bool),
        is_pure_echo=torch.tensor([r[8] for r in rows], dtype=torch.bool),
        is_pure_echo_noise=torch.tensor([r[9] for r in rows], dtype=torch.bool),
        has_echo=torch.tensor([r[10] for r in rows], dtype=torch.bool),
    )

    if has_state_input:
        max_state_len = max(int(st.shape[-1]) for st in states)
        padded_states = torch.stack([
            F.pad(st, (0, max_state_len - st.shape[-1]), value=-1)
            for st in states
        ])
        out.state = padded_states

    out.validate()
    return out
