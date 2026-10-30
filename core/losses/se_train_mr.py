"""se_train 式多分辨率损失：MR 复谱 + suppress penalty。

注册名 3 个：MRSIRelCspecL2Loss、MRSpecSuppressPenaltyLoss、
MRSIRelCspecL2SuppressPenaltyLoss。时域 SIL1WavSeTrainLoss 位于 time_domain 模块。
要点: 默认窗为 2^6..2^11、hop=窗长/2；也可按顶层 win/hop 比例缩放。
``align_se_train`` 只改变 MRSI 在 ``lengths is None`` 时的 clean-power 低能量
归一化；有 ``lengths`` 时不生效。suppress-only 类不消费它，非默认值 fail-fast。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real

import torch
import torch.nn as nn
from core.losses.stft_cache import cached_arange, stft_real_view
from core.losses.utils import reflect_extend_at_lengths
from core.utils.registry import LOSS_REGISTRY


def _trim_consistent_stft_input(x: torch.Tensor, lengths: torch.Tensor, pad: int) -> torch.Tensor:
    """时域 mask(lengths 外置零) + 有效边界 reflect 延展, 使 STFT 结果与
    trim 到 lengths 后 ``torch.stft(center=True, pad_mode='reflect')`` 逐点一致
    (JAE-H02 padding 不变)。

    x: [B, T]; lengths: [B]; pad = n_fft // 2。调用 reflect 延展前固定追加
    pad 个零，确保 centered STFT 总有完整右侧余量，且极短输入也长于 reflect pad。
    原输入 padding 区梯度恒为 0(mask 置零, 延展只 gather 有效区)。

    与 utils.reflect_extend_at_lengths 的差异: 本函数先做时域 mask(模型输出的
    padding 区可能非零, 必须置零)并预留右侧余量，后者要求调用方保证这两点。
    """
    lengths = lengths.to(device=x.device, dtype=torch.long)
    t_idx = cached_arange(x.shape[1], x.device).unsqueeze(0)
    x = x * (t_idx < lengths.unsqueeze(1)).to(x.dtype)
    x = torch.cat((x, x.new_zeros(x.shape[0], pad)), dim=1)
    return reflect_extend_at_lengths(x, lengths, pad)


def _resolve_stft_schedule(
    fft_power_min: int | None,
    fft_power_max: int | None,
    *,
    stft_scale_factors: Iterable[Real] | None,
    win_length: int | None,
    hop_length: int | None,
) -> tuple[tuple[int, ...], tuple[int, ...], int | None, int | None]:
    """解析 legacy power 或基于顶层 STFT 参数缩放的精确分辨率表。

    ``stft_scale_factors`` 未提供时使用 legacy power schedule，并忽略由
    ``build_loss`` 注入的 ``win_length`` / ``hop_length`` 上下文参数。
    """
    if stft_scale_factors is None:
        power_min = 6 if fft_power_min is None else fft_power_min
        power_max = 11 if fft_power_max is None else fft_power_max
        if (
            isinstance(power_min, bool)
            or not isinstance(power_min, int)
            or isinstance(power_max, bool)
            or not isinstance(power_max, int)
        ):
            raise TypeError("fft_power_min and fft_power_max must be integers")
        if power_min < 1 or power_max < power_min:
            raise ValueError("fft powers must satisfy 1 <= fft_power_min <= fft_power_max")
        fft_sizes = tuple(2**power for power in range(power_min, power_max + 1))
        return fft_sizes, tuple(size // 2 for size in fft_sizes), power_min, power_max

    if fft_power_min is not None or fft_power_max is not None:
        raise ValueError("stft_scale_factors conflicts with explicit fft_power_min/fft_power_max")
    if win_length is None or hop_length is None:
        raise ValueError("stft_scale_factors requires both win_length and hop_length")
    if (
        isinstance(win_length, bool)
        or not isinstance(win_length, int)
        or win_length <= 0
        or isinstance(hop_length, bool)
        or not isinstance(hop_length, int)
        or hop_length <= 0
    ):
        raise ValueError("win_length and hop_length must be positive integers")
    if isinstance(stft_scale_factors, (str, bytes)):
        raise TypeError("stft_scale_factors must be an iterable of finite numbers")
    try:
        factors = tuple(stft_scale_factors)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("stft_scale_factors must be an iterable of finite numbers") from exc
    if not factors:
        raise ValueError("stft_scale_factors must not be empty")

    normalized_factors: list[float] = []
    for factor in factors:
        if isinstance(factor, bool) or not isinstance(factor, Real):
            raise TypeError("stft_scale_factors must contain finite numbers")
        try:
            value = float(factor)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("stft_scale_factors must contain finite numbers") from exc
        if not math.isfinite(value):
            raise ValueError("stft_scale_factors must contain only finite numbers")
        if value <= 0:
            raise ValueError("stft_scale_factors must be positive")
        if normalized_factors and value <= normalized_factors[-1]:
            raise ValueError("stft_scale_factors must be strictly increasing")
        normalized_factors.append(value)

    def _scale_exact(base: int, factor: float, name: str) -> int:
        try:
            scaled = base * factor
        except OverflowError as exc:
            raise ValueError(
                f"{name} * factor {factor!r} must be a positive integer sample count"
            ) from exc
        if not math.isfinite(scaled) or not scaled.is_integer() or scaled <= 0:
            raise ValueError(f"{name} * factor {factor!r} must be a positive integer sample count")
        return int(scaled)

    fft_sizes = tuple(_scale_exact(win_length, factor, "win_length") for factor in normalized_factors)
    hop_sizes = tuple(_scale_exact(hop_length, factor, "hop_length") for factor in normalized_factors)
    return fft_sizes, hop_sizes, None, None


@LOSS_REGISTRY.register("MRSIRelCspecL2Loss")
class MRSIRelCspecL2Loss(nn.Module):
    """Multi-resolution SI + relative complex-spectrum L2 with optional frequency weighting.

    ``align_se_train=True`` 仅在 ``lengths is None`` 时匹配 se_train
    ``MR_SI_REL_CSPEC_L2_Loss`` 的 clean-power 归一化：只把 ``<1e-20``
    的均值替换为 0.5。设为 False 时同一全网格均值会 clamp 到最低 0.5。
    提供 ``lengths`` 后统一改为有效帧 masked mean 并 clamp 到 0.5，
    ``align_se_train`` 不再影响结果（JAE-H02 padding 不变）。
    """

    def __init__(
        self,
        fft_power_min: int | None = None,
        fft_power_max: int | None = None,
        alpha_rel_L2: float = 1.0,
        use_frequency_weight: bool = True,
        align_se_train: bool = True,
        *,
        stft_scale_factors: Iterable[Real] | None = None,
        win_length: int | None = None,
        hop_length: int | None = None,
    ):
        super().__init__()
        self.fft_sizes, self.hop_sizes, self.fft_power_min, self.fft_power_max = (
            _resolve_stft_schedule(
                fft_power_min,
                fft_power_max,
                stft_scale_factors=stft_scale_factors,
                win_length=win_length,
                hop_length=hop_length,
            )
        )
        self.K = len(self.fft_sizes)
        self.alpha_rel_L2 = alpha_rel_L2
        self.use_frequency_weight = use_frequency_weight
        self.align_se_train = bool(align_se_train)

        self.win_lengths = self.fft_sizes
        self.hop_lengths = self.hop_sizes
        for resolution_idx, wl in enumerate(self.fft_sizes):
            self.register_buffer(f"win_{resolution_idx}", torch.hann_window(wl), persistent=False)
            if use_frequency_weight:
                fw = 2 * torch.cos(torch.linspace(0, math.pi / 2, wl // 2 + 1) - math.pi / 16) + 0.5
                self.register_buffer(
                    f"fw_{resolution_idx}", fw.unsqueeze(-1).unsqueeze(0), persistent=False
                )

    def _one_resolution(self, enhanced: torch.Tensor, clean: torch.Tensor,
                        lengths: torch.Tensor | None, k_idx: int,
                        enh_stft: torch.Tensor | None = None,
                        cln_stft: torch.Tensor | None = None) -> torch.Tensor:
        wl, hl = self.fft_sizes[k_idx], self.hop_sizes[k_idx]
        win = getattr(self, f"win_{k_idx}")

        if lengths is not None:
            # JAE-H02: lengths 存在时, 传入的 stft_cache 是在原始 padded 波形上计算的
            # (边界帧依赖 batch 最大长度, 且模型输出的 padding 区可能非零), 不满足
            # padding 不变性, 故在 trim 一致的波形上重算本分辨率 STFT。
            enh_stft = stft_real_view(_trim_consistent_stft_input(enhanced, lengths, wl // 2), wl, hl, win)
            cln_stft = stft_real_view(_trim_consistent_stft_input(clean, lengths, wl // 2), wl, hl, win)
        else:
            if enh_stft is None:
                enh_stft = stft_real_view(enhanced, wl, hl, win)  # (B, F, T, 2)
            if cln_stft is None:
                cln_stft = stft_real_view(clean, wl, hl, win)

        diff_sq = ((enh_stft - cln_stft) ** 2).sum(dim=-1)  # (B, F, T)
        cln_pow = (cln_stft ** 2).sum(dim=-1)

        # Build (B, F, T) mask matching se_train's cur_spec_mask exactly
        if lengths is not None:
            from core.losses.utils import compute_spec_frame_lengths
            spec_lens = compute_spec_frame_lengths(lengths, n_fft=wl, win_length=wl, hop_length=hl)
            t_max = diff_sq.shape[2]
            t_idx = cached_arange(t_max, diff_sq.device).unsqueeze(0)
            t_mask = (t_idx < spec_lens.unsqueeze(1)).unsqueeze(1)  # (B, 1, T)
            mask = t_mask.expand_as(diff_sq).float()  # (B, F, T)
        else:
            mask = diff_sq.new_ones(diff_sq.shape)

        n_valid = mask.sum().clamp(1)

        if self.align_se_train and lengths is None:
            batch_mean = cln_pow.mean(dim=(1, 2), keepdim=True)
            batch_mean = torch.where(
                batch_mean < 1e-20, batch_mean.new_tensor(0.5), batch_mean
            )
        else:
            # JAE-H02: 提供 lengths 时按有效帧 masked mean(公式同 align_se_train=False),
            # padding 帧不参与, 避免 batch_mean 被稀释。
            batch_mean = (cln_pow * mask).sum(dim=(1, 2), keepdim=True) / mask.sum(
                dim=(1, 2), keepdim=True
            ).clamp(1)
            batch_mean = batch_mean.clamp(min=0.5)

        if self.use_frequency_weight:
            fw = getattr(self, f"fw_{k_idx}")
            si_cspec = (mask * fw * diff_sq / (batch_mean + 1e-12)).sum() / n_valid
            rel_cspec = (mask * fw * diff_sq / (cln_pow + 0.05)).sum() / n_valid
        else:
            si_cspec = (mask * diff_sq / (batch_mean + 1e-12)).sum() / n_valid
            rel_cspec = (mask * diff_sq / (cln_pow + 0.05)).sum() / n_valid

        return si_cspec + self.alpha_rel_L2 * rel_cspec

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        stft_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        if stft_cache is not None:
            if len(stft_cache) != self.K:
                raise ValueError(
                    f"stft_cache length {len(stft_cache)} != K={self.K} "
                    f"(fft_power {self.fft_power_min}-{self.fft_power_max})"
                )
            total = sum(
                self._one_resolution(
                    enhanced, clean, lengths, i,
                    stft_cache[i][0], stft_cache[i][1],
                )
                for i in range(self.K)
            )
        else:
            total = sum(self._one_resolution(enhanced, clean, lengths, i) for i in range(self.K))
        return total / self.K


@LOSS_REGISTRY.register("MRSpecSuppressPenaltyLoss")
class MRSpecSuppressPenaltyLoss(nn.Module):
    """Multi-resolution mean of relu(|enh| - |clean|) on STFT magnitude (se_train suppress term only).

    与 CompositeLoss 合用时，若要表达外部配方的 ``alpha_MR × alpha_penalty``，
    应由 component ``weight`` 显式给出；本类不会读取这两个外部系数。
    ``lengths is None`` 时始终做全网格 mean，提供 ``lengths`` 时始终做有效帧
    masked mean（JAE-H02 padding 不变）。``align_se_train`` 仅保留为构造兼容位，
    当前 forward 在两个分支中都不读取它，因此非默认值 ``false`` 在构造期被拒绝。
    """

    def __init__(
        self,
        fft_power_min: int | None = None,
        fft_power_max: int | None = None,
        align_se_train: bool = True,
        *,
        stft_scale_factors: Iterable[Real] | None = None,
        win_length: int | None = None,
        hop_length: int | None = None,
    ):
        super().__init__()
        self.fft_sizes, self.hop_sizes, self.fft_power_min, self.fft_power_max = (
            _resolve_stft_schedule(
                fft_power_min,
                fft_power_max,
                stft_scale_factors=stft_scale_factors,
                win_length=win_length,
                hop_length=hop_length,
            )
        )
        self.K = len(self.fft_sizes)
        if not align_se_train:
            # 该类的 forward 两个分支都不读取此开关：非默认值必须显式拒绝，
            # 否则配置里写 false 会静默无效。
            raise ValueError(
                "MRSpecSuppressPenaltyLoss 不消费 align_se_train=False："
                "该参数仅为构造兼容位，请移除该键或使用默认 true"
            )
        self.align_se_train = True
        self.win_lengths = self.fft_sizes
        self.hop_lengths = self.hop_sizes
        for resolution_idx, wl in enumerate(self.fft_sizes):
            self.register_buffer(f"win_{resolution_idx}", torch.hann_window(wl), persistent=False)

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        stft_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        if stft_cache is not None and len(stft_cache) != self.K:
            raise ValueError(
                f"stft_cache length {len(stft_cache)} != K={self.K} "
                f"(fft_power {self.fft_power_min}-{self.fft_power_max})"
            )
        total_pen = enhanced.new_tensor(0.0)
        for i in range(self.K):
            wl, hl = self.fft_sizes[i], self.hop_sizes[i]
            win = getattr(self, f"win_{i}")
            if lengths is not None:
                # JAE-H02: 同 MRSIRelCspecL2Loss, lengths 存在时在 trim 一致的波形上
                # 重算 STFT(stft_cache 是原始 padded 波形的, 不满足 padding 不变)。
                enh_stft = stft_real_view(_trim_consistent_stft_input(enhanced, lengths, wl // 2), wl, hl, win)
                cln_stft = stft_real_view(_trim_consistent_stft_input(clean, lengths, wl // 2), wl, hl, win)
            elif stft_cache is not None:
                enh_stft, cln_stft = stft_cache[i]
            else:
                enh_stft = stft_real_view(enhanced, wl, hl, win)
                cln_stft = stft_real_view(clean, wl, hl, win)
            enh_mag = torch.linalg.norm(enh_stft, dim=-1)
            cln_mag = torch.linalg.norm(cln_stft, dim=-1)
            penalty = torch.relu(enh_mag - cln_mag)
            if lengths is None:
                total_pen = total_pen + penalty.mean()
            else:
                # JAE-H02: 提供 lengths 时一律按有效帧 masked mean,
                # align_se_train 不再强制全网格。
                from core.losses.utils import compute_spec_frame_lengths
                spec_lens = compute_spec_frame_lengths(lengths, n_fft=wl, win_length=wl, hop_length=hl)
                t_max = penalty.shape[2]
                t_idx = cached_arange(t_max, penalty.device).unsqueeze(0)
                pen_mask = (t_idx < spec_lens.unsqueeze(1)).unsqueeze(1).expand_as(penalty).float()
                total_pen = total_pen + (penalty * pen_mask).sum() / pen_mask.sum().clamp(1)
        return total_pen / self.K


@LOSS_REGISTRY.register("MRSIRelCspecL2SuppressPenaltyLoss")
class MRSIRelCspecL2SuppressPenaltyLoss(nn.Module):
    """MR SI+REL cspec L2 with suppress penalty (penalise enhanced > clean).

    ``align_se_train=True`` 且 ``lengths is None`` 时，MR 主项采用 se_train
    ``MR_SI_REL_CSPEC_L2_SuppressPenalty_Loss`` 的 clean-power 低能量替换语义；
    suppress 项无论该开关取值都保持全网格 mean。

    penalty 项完全委托 ``MRSpecSuppressPenaltyLoss``：传入 ``lengths`` 时，
    两个子项都在 trim 一致波形上重算 STFT 并做有效帧 masked mean
    （JAE-H02 padding 不变），此时 ``align_se_train`` 对两项都不生效；
    ``lengths=None`` 时组合类内部缓存并复用同一组 STFT pair。
    """

    def __init__(
        self,
        fft_power_min: int | None = None,
        fft_power_max: int | None = None,
        alpha_rel_L2: float = 1.0,
        use_frequency_weight: bool = True,
        alpha_penalty: float = 1.0,
        align_se_train: bool = True,
        *,
        stft_scale_factors: Iterable[Real] | None = None,
        win_length: int | None = None,
        hop_length: int | None = None,
    ):
        super().__init__()
        if stft_scale_factors is not None and not isinstance(stft_scale_factors, (list, tuple)):
            stft_scale_factors = tuple(stft_scale_factors)
        schedule_kwargs = {
            "stft_scale_factors": stft_scale_factors,
            "win_length": win_length,
            "hop_length": hop_length,
        }
        self.mr_loss = MRSIRelCspecL2Loss(
            fft_power_min,
            fft_power_max,
            alpha_rel_L2,
            use_frequency_weight,
            align_se_train,
            **schedule_kwargs,
        )
        # penalty 子项不消费 align_se_train：此处不转发，避免被其构造边界拒绝
        self.penalty_loss = MRSpecSuppressPenaltyLoss(
            fft_power_min,
            fft_power_max,
            **schedule_kwargs,
        )
        self.alpha_penalty = alpha_penalty
        self.align_se_train = bool(align_se_train)
        self.fft_sizes = self.mr_loss.fft_sizes
        self.hop_sizes = self.mr_loss.hop_sizes
        self.win_lengths = self.fft_sizes
        self.hop_lengths = self.hop_sizes
        self.K = self.mr_loss.K

    def forward(
        self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        if lengths is not None:
            # JAE-H02: 外层 padded STFT 不满足 padding 不变(边界帧依赖 batch
            # 最大长度), mr 项与 penalty 项各自在 trim 一致波形上重算, 此处
            # 不再重复计算外层 STFT。
            total_loss = sum(
                self.mr_loss._one_resolution(enhanced, clean, lengths, i)
                for i in range(self.K)
            )
            total_pen = self.penalty_loss(enhanced, clean, lengths)
        else:
            # lengths=None: 外层 STFT 同时供 mr 项与 penalty 复用。
            stft_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
            total_loss = enhanced.new_tensor(0.0)
            for i in range(self.K):
                wl = self.fft_sizes[i]
                hl = self.hop_sizes[i]
                win = getattr(self.mr_loss, f"win_{i}")
                enh_stft = stft_real_view(enhanced, wl, hl, win)
                cln_stft = stft_real_view(clean, wl, hl, win)
                stft_pairs.append((enh_stft, cln_stft))
                total_loss = total_loss + self.mr_loss._one_resolution(
                    enhanced, clean, lengths, i, enh_stft=enh_stft, cln_stft=cln_stft
                )
            total_pen = self.penalty_loss(
                enhanced, clean, None, stft_cache=stft_pairs
            )
        return total_loss / self.K + self.alpha_penalty * total_pen
