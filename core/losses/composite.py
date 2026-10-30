"""CompositeLoss：多 loss 线性组合。说明见 docs/guides/loss/Loss_Configuration_Guide.md「基础概念 → 组合 Loss (CompositeLoss)」小节。

YAML 示例:
  loss:
    type: CompositeLoss
    components:
      - { type: SIL1Loss, weight: 1.0 }
      - { type: MultiResFreqLoss, weight: 0.5, args: { fft_sizes: [256, 512, 1024] } }
      - { type: MRSIRelCspecL2Loss, weight: 0.5, name: "MR-SI-Rel" }  # 可选 name 用于日志可读

Final loss = sum(weight_i * loss_i(enhanced, clean, lengths[, **kwargs])).
``weight`` 可为任意有限 float（含负数和零），但至少一项必须非零；不会归一化，也未限制到 [0, 1] 或非负区间。
仅对 forward 签名含 **kwargs 的组件传入 kwargs（如 predicted_lsnr、noisy），避免其他组件收到多余参数报错。

**is_pure_noise / is_pure_echo / is_pure_echo_noise**（可选 kwargs）：均为 ``(B,)`` bool。任一为 ``True`` 的样本视为**无有效近端语音参考**；三掩码按位或后，声明 ``REQUIRES_SPEECH_REFERENCE`` capability 的组件（当前包括 PESQLoss、STOILoss、SISNRLoss、FWNegSISDRLoss、TAPLoss〔含 AcousticLoss 别名〕和 GTCRNLogRatioLoss）仅在非 pure 子集上计算。兼容仅传 ``is_pure_noise`` 的旧配置。其余分量（含 MultiBandRelativeContrastiveLoss、MRSIRelCspecL2Loss、主损失、MR-SpecSuppress、StateCrossEntropyLoss 等）不按该 metadata 排除，仍使用全 batch；三者均未传入时不执行 pure 子集门控。DDP 下每个静态 speech-reference 分量固定执行一次 reduction-mass ``all_reduce``，将各 rank 局部 mean 缩放为全局 mean；默认 mass 为有效样本数，按帧/元素归约的组件应实现 ``speech_reference_reduction_mass(clean, lengths) -> [B]``，TAPLoss 兼容路径使用有效帧数。任意组件还可声明 ``ddp_global_mean_mass(enhanced, clean, lengths, **kwargs) -> scalar``：CompositeLoss 同样固定执行一次 SUM，并以 ``local_loss × world_size × local_mass / global_mass`` 重标定其 local mean（State/Speaker CE 使用此协议）。零有效质量仍返回 graph-connected zero。权重为零的 capability 分量不调用 valid/mass callback 或 forward，但所有 rank 仍执行固定零质量 collective。组件若有额外样本 guard，应实现 ``speech_reference_valid_mask(clean, lengths) -> [B] bool``；GTCRNLogRatioLoss、SISNRLoss、FWNegSISDRLoss 的既有 clean-energy guard 由兼容路径识别。结构、metadata、SS、共享 STFT cache 和 energy 预处理先在固定 preflight 中跨 rank 同步；preflight 使用 ``inference_mode(False)`` 的 ``all_gather_object``，异常在任何组件内部 collective 前由所有 rank 一致抛出。组件 forward 异常与 strict 输出 NaN/Inf 则延迟到固定内部 collective 序列结束后同步抛出；不可恢复 CUDA 异常立即传播。

**enable_ss_undesired_energy**（顶层默认 ``false``；各 ``components[]`` 可写 ``enable_ss_undesired_energy: true/false``，省略则跟随顶层）：为 ``true`` 的普通子项按样本能量混合两路监督
``a_i * L_i(enhanced, clean) + (1-a_i) * L_i(undesired_enhanced, undesired_ref)``，其中
``a_i = E_clean / (E_clean + E_undesired_ref + eps)``（有效长度内能量）。**SE/AEC trainer** 注入 ``undesired_ref = noisy - clean``（AEC：``mic - nearend``）、``undesired_enhanced = noisy - enhanced``（AEC：``mic - nearend_estimate``），模型只需单路 ``enhanced``；若手工传入 kwargs 亦可覆盖。未给 ``undesired_ref`` 时可由 ``noisy - clean`` 推导。只要传入任一 pure metadata，上述语音参考依赖项就优先走非 pure 子集分支，不执行 SS 混合；StateCrossEntropyLoss 不属于 pure 排除集合，也没有通用自动 SS 排除。TAPLoss 的 clean-speech estimator 不适用于 undesired/noise residual，effective SS 为 ``true`` 时在构造期直接拒绝；使用顶层 SS 时必须给 TAP component 显式覆盖 ``false``。MRSIRelCspecL2Loss 是否执行 SS 只取决于顶层或 component 级开关，不受 pure metadata 影响。开启 SS 的 MR 子项走逐样本前向，不使用 MR STFT 共享缓存。

训练时：SETrainer/AECTrainer 在 **epoch 结束**时打印并记录各分量的 **加权值 weight×raw**（与总 loss 同尺度）：train/loss_comp/<name>、val/loss_comp/<name>，以及 train_val_curve.csv / train_val_loss.png 子图。文档无对应小节，详见 docs/guides/loss/Loss_Configuration_Guide.md。

**MR-STFT 共享缓存**（任务无关，鸭子类型协议，见 ``_mr_cache_signature``）：构造期按 YAML 顺序选择**第一对**符合条件的 loss 组件；两者须暴露顺序一致的 canonical ``fft_sizes``/``hop_sizes`` 属性，且 ``forward()`` 显式接受 ``stft_cache``。仅这对组件会在 ``lengths is None`` 且两者均未开启 SS 时复用由 ``core.losses.stft_cache.compute_mr_stft_shared_cache`` 计算的一份多分辨率 STFT；后续即使还有同 schedule 组件，也不会加入该 cache pair。旧组件可暴露 ``fft_power_min``/``fft_power_max``，协议会将其转换为 power-of-two、half-hop schedule。协议不要求具体 loss 类名，但 cache signature 只表达精确几何与约定布局，不包含 ``align_se_train``。总 loss 在 float32 中累加，避免 bf16/fp16 与权重混加数值不稳。
"""

from __future__ import annotations

import inspect
import math
import torch
import torch.distributed as dist
import torch.nn as nn
from core.losses.constants import LOSS_DISPLAY_NAMES  # noqa: F401 (re-exported for callers)
from core.losses.stft_cache import compute_mr_stft_shared_cache
from core.utils.ddp_error import is_unrecoverable_cuda_exception
from core.utils.registry import LOSS_REGISTRY

# 纯噪/纯回声时不参与：依赖干净语音参考的语音/感知辅助项与抗过衰减项。
# MRSI 适用于所有 AEC 场景；其 SS 行为只由 enable_ss_undesired_energy 决定。
# 类名兼容集合只保留仍依赖类名 fallback 的 TAPLoss；PESQLoss/STOILoss/
# SISNRLoss/FWNegSISDRLoss 已显式声明 REQUIRES_SPEECH_REFERENCE capability。
_SPEECH_REF_DEPENDENT_CLASS_NAMES = frozenset(
    {
        "TAPLoss",
    }
)


class StrictFiniteLossError(RuntimeError):
    """Strict loss 数值检查发现 NaN/Inf；不表示结构、类型或 shape 错误。"""


def _graph_connected_zero(reference: torch.Tensor) -> torch.Tensor:
    """构造与 ``reference`` 图连接、且对非有限输入仍恒为 0 的标量零。

    先 ``nan_to_num`` 再乘零：直接 ``(x * 0.0).sum()`` 在 ``x`` 含 NaN/Inf 时
    会得到 NaN，与「乘零后归约恒为 0」的意图不符（strict=False 无输入检查）。
    """
    return (torch.nan_to_num(reference) * 0.0).sum()


def _requires_speech_reference(loss: nn.Module) -> bool:
    """优先使用显式 capability，旧 loss 回退到类名兼容集合。"""
    capability = getattr(loss, "REQUIRES_SPEECH_REFERENCE", None)
    if capability is not None:
        return bool(capability)
    return type(loss).__name__ in _SPEECH_REF_DEPENDENT_CLASS_NAMES


def _speech_reference_valid_mask(
    loss: nn.Module,
    clean: torch.Tensor,
    lengths: torch.Tensor | None,
) -> torch.Tensor:
    """返回组件实际参与 batch mean 的样本 mask。

    自带额外 guard 的新组件应实现
    ``speech_reference_valid_mask(clean, lengths) -> [B] bool``。未实现时默认
    metadata 选中的样本均有效；GTCRNLogRatioLoss、SISNRLoss 与
    FWNegSISDRLoss 的既有 clean-energy guard 在此保留兼容实现。
    """
    callback = getattr(loss, "speech_reference_valid_mask", None)
    if callback is not None:
        valid = callback(clean, lengths)
        if not torch.is_tensor(valid) or valid.shape != (clean.shape[0],):
            raise ValueError(
                "speech_reference_valid_mask() must return a [B] Tensor, got "
                f"{type(valid).__name__} with shape "
                f"{getattr(valid, 'shape', None)}"
            )
        return valid.to(device=clean.device, dtype=torch.bool)

    class_name = type(loss).__name__
    if lengths is not None:
        time = torch.arange(clean.shape[-1], device=clean.device)
        sample_mask = time.unsqueeze(0) < lengths.unsqueeze(1)
        masked_clean = clean * sample_mask
    else:
        sample_mask = None
        masked_clean = clean

    if class_name == "SISNRLoss":
        if lengths is None:
            centered = clean - clean.mean(dim=-1, keepdim=True)
        else:
            count = lengths.to(clean.dtype).unsqueeze(-1).clamp(min=1.0)
            centered = (masked_clean - masked_clean.sum(dim=-1, keepdim=True) / count)
            centered = centered * sample_mask
        return centered.square().sum(dim=-1) > float(getattr(loss, "EPS", 1e-8))
    if class_name == "FWNegSISDRLoss":
        if lengths is None:
            energy = clean.square().mean(dim=-1)
        else:
            count = lengths.to(clean.dtype).clamp(min=1.0)
            energy = masked_clean.square().sum(dim=-1) / count
        return energy > float(getattr(loss, "eps", 1e-8))
    if class_name == "GTCRNLogRatioLoss":
        return masked_clean.square().sum(dim=-1) > float(getattr(loss, "eps", 1e-8))
    return torch.ones(clean.shape[0], device=clean.device, dtype=torch.bool)


def _speech_reference_reduction_mass(
    loss: nn.Module,
    clean: torch.Tensor,
    lengths: torch.Tensor | None,
) -> torch.Tensor:
    """返回每条样本在组件局部 mean denominator 中的 reduction mass。

    默认契约是逐样本 mean，因此每条有效样本质量为 1。按有效帧、有效元素等
    归约的组件必须实现
    ``speech_reference_reduction_mass(clean, lengths) -> [B]``。TAPLoss 的旧
    实现按所有有效帧与固定 25 个参数求 mean；固定参数维会在比例中约掉，因此
    兼容路径只返回 ``center=True`` STFT 的有效帧数。
    """
    callback = getattr(loss, "speech_reference_reduction_mass", None)
    if callback is not None:
        mass = callback(clean, lengths)
    elif type(loss).__name__ == "TAPLoss":
        sample_lengths = (
            lengths
            if lengths is not None
            else torch.full(
                (clean.shape[0],),
                clean.shape[-1],
                device=clean.device,
                dtype=torch.long,
            )
        )
        mass = torch.div(
            sample_lengths,
            int(getattr(loss, "hop_length", 160)),
            rounding_mode="floor",
        ) + 1
    else:
        mass = torch.ones(clean.shape[0], device=clean.device, dtype=torch.float32)

    if not torch.is_tensor(mass) or mass.shape != (clean.shape[0],):
        raise ValueError(
            "speech_reference_reduction_mass() must return a [B] Tensor, got "
            f"{type(mass).__name__} with shape {getattr(mass, 'shape', None)}"
        )
    mass = mass.to(device=clean.device, dtype=torch.float32)
    if not bool(torch.isfinite(mass).all().item()) or bool((mass < 0).any().item()):
        raise ValueError("speech_reference_reduction_mass() must be finite and non-negative")
    return mass


def _distributed_sum_scalar(local: torch.Tensor) -> tuple[torch.Tensor, int]:
    """固定执行 scalar SUM；validation inference mode 下也使用普通 tensor。"""
    if not (dist.is_available() and dist.is_initialized()):
        return local, 1
    world_size = dist.get_world_size()
    if world_size <= 1:
        return local, world_size
    with torch.inference_mode(False):
        total = torch.zeros((), device=local.device, dtype=local.dtype)
        total.copy_(local)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total, world_size


def _synchronize_preflight_error(local_error: Exception | None) -> None:
    """在任何组件内部 collective 前同步结构/准备阶段异常。"""
    if not (dist.is_available() and dist.is_initialized()):
        if local_error is not None:
            raise local_error
        return
    world_size = dist.get_world_size()
    if world_size <= 1:
        if local_error is not None:
            raise local_error
        return
    try:
        dist.get_backend()
    except (RuntimeError, ValueError):
        if local_error is not None:
            raise local_error
        return
    local_detail = (
        local_error is not None,
        isinstance(local_error, StrictFiniteLossError),
        f"rank {dist.get_rank()}: {type(local_error).__name__}: {local_error}"
        if local_error is not None
        else None,
    )
    details: list[tuple[bool, bool, str | None]] = [None] * world_size
    with torch.inference_mode(False):
        dist.all_gather_object(details, local_detail)
    if not any(detail[0] for detail in details):
        return
    messages = [detail[2] for detail in details if detail[2] is not None]
    message = "; ".join(messages)
    if all(detail[1] for detail in details if detail[0]):
        raise StrictFiniteLossError(message) from local_error
    raise RuntimeError(message) from local_error


def _ddp_global_mean_mass(
    loss: nn.Module,
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    lengths: torch.Tensor | None,
    kwargs: dict,
) -> torch.Tensor:
    """调用组件的全局 mean 分母 callback，并校验标量质量契约。"""
    callback = getattr(loss, "ddp_global_mean_mass", None)
    if callback is None:
        raise TypeError("loss does not define ddp_global_mean_mass")
    mass = callback(
        enhanced=enhanced,
        clean=clean,
        lengths=lengths,
        **kwargs,
    )
    if not torch.is_tensor(mass) or mass.numel() != 1:
        raise ValueError(
            "ddp_global_mean_mass() must return a scalar Tensor, got "
            f"{type(mass).__name__} with shape {getattr(mass, 'shape', None)}"
        )
    mass = mass.to(device=enhanced.device, dtype=torch.float32).reshape(())
    if not bool(torch.isfinite(mass).item()) or bool((mass < 0).item()):
        raise ValueError("ddp_global_mean_mass() must be finite and non-negative")
    return mass


def _strict_finite_input_error(
    named_inputs: list[tuple[str, object]],
) -> StrictFiniteLossError | None:
    """返回首个浮点/复数输入的有限性错误，交由固定 DDP 协议同步。"""
    for name, value in named_inputs:
        if not torch.is_tensor(value):
            continue
        if not (torch.is_floating_point(value) or torch.is_complex(value)):
            continue
        if not bool(torch.isfinite(value).all().item()):
            return StrictFiniteLossError(
                f"CompositeLoss strict mode: '{name}' has NaN/Inf"
            )
    return None


def _accepts_var_keyword(fn) -> bool:
    """True if fn.forward accepts **kwargs."""
    sig = inspect.signature(fn.forward)
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _mr_cache_signature(
    loss: nn.Module,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Exact ordered STFT geometry if ``loss`` can consume a shared cache.

    Canonical consumers expose ``fft_sizes`` and ``hop_sizes``. Legacy duck-typed
    consumers exposing ``fft_power_min``/``fft_power_max`` are converted to the
    equivalent power-of-two, half-hop schedule. In either case ``forward()`` must
    explicitly accept ``stft_cache``.

    The cache convention remains Hann, ``win_length = n_fft = fft_size``,
    ``center=True``, ``onesided=True``, FP32, with ``view_as_real`` layout
    ``(B, F, T, 2)``. Geometry and resolution order are both part of the signature.
    Consumers using another window, mode, or layout must not accept ``stft_cache``.

    NOTE (JAE-H02): the cache is computed on the raw padded waveforms; when
    ``lengths`` is not None, consumers MUST ignore the shared cache and recompute
    per-resolution STFTs on their own trim-consistent waveforms (padding
    invariance), which is why ``CompositeLoss.forward`` skips building the
    cache entirely in that case.
    """
    try:
        sig = inspect.signature(loss.forward)
    except (TypeError, ValueError):
        return None
    if "stft_cache" not in sig.parameters:
        return None

    fft_sizes = getattr(loss, "fft_sizes", None)
    hop_sizes = getattr(loss, "hop_sizes", None)
    if fft_sizes is not None or hop_sizes is not None:
        if fft_sizes is None or hop_sizes is None:
            return None
        try:
            exact_fft_sizes = tuple(int(size) for size in fft_sizes)
            exact_hop_sizes = tuple(int(size) for size in hop_sizes)
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        fmin = getattr(loss, "fft_power_min", None)
        fmax = getattr(loss, "fft_power_max", None)
        if fmin is None or fmax is None:
            return None
        try:
            powers = range(int(fmin), int(fmax) + 1)
            exact_fft_sizes = tuple(2**power for power in powers)
        except (TypeError, ValueError, OverflowError):
            return None
        exact_hop_sizes = tuple(size // 2 for size in exact_fft_sizes)

    if (
        not exact_fft_sizes
        or len(exact_fft_sizes) != len(exact_hop_sizes)
        or any(size <= 0 for size in exact_fft_sizes)
        or any(hop <= 0 for hop in exact_hop_sizes)
    ):
        return None
    return exact_fft_sizes, exact_hop_sizes


def _subset_kwargs_for_batch(speech_idx: torch.Tensor, batch_b: int, kwargs: dict) -> dict:
    """Index-select tensor kwargs along dim 0 when shape[0]==B."""
    if not kwargs:
        return {}
    out: dict = {}
    for k, v in kwargs.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == batch_b:
            out[k] = v.index_select(0, speech_idx)
        else:
            out[k] = v
    return out


def _subset_kwargs_for_index(kwargs: dict, b: int, batch_b: int) -> dict:
    """Take sample ``b`` from batch-sized tensor kwargs (shape[0]==batch_b)."""
    if not kwargs:
        return {}
    out: dict = {}
    for k, v in kwargs.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == batch_b:
            out[k] = v[b : b + 1]
        else:
            out[k] = v
    return out


def _ss_energy_a(
    clean: torch.Tensor,
    undesired_ref: torch.Tensor,
    lengths: torch.Tensor | None,
) -> torch.Tensor:
    """Per-sample mixing weight a[b] in (0,1), shape [B]."""
    bsz, t = clean.shape
    device = clean.device
    dtype = clean.dtype
    if lengths is not None:
        t_idx = torch.arange(t, device=device, dtype=torch.long).unsqueeze(0)
        mask = (t_idx < lengths.unsqueeze(1).clamp(min=0)).to(dtype)
        n = mask.sum(dim=1).clamp(min=1.0)
        e_tar = ((clean * clean) * mask).sum(dim=1)
        e_non = ((undesired_ref * undesired_ref) * mask).sum(dim=1)
    else:
        e_tar = (clean * clean).sum(dim=1)
        e_non = (undesired_ref * undesired_ref).sum(dim=1)
    return e_tar / (e_tar + e_non + 1e-8)


class CompositeLoss(nn.Module):
    """Weighted sum of multiple registered losses. Forwards **kwargs only to components that accept them.

    After each forward(), sets self._last_breakdown: list of
    (log_key, weight, raw_value, weighted_value) for logging.
    ``log_key`` is ``yaml_{index:03d}::{type}`` (YAML component order + LOSS_REGISTRY type).
    Human-readable label remains in ``_component_names`` / ``_component_types``.
    """

    def __init__(
        self,
        components: list[dict],
        strict: bool = False,
        enable_ss_undesired_energy: bool = False,
    ):
        super().__init__()
        if not components:
            raise ValueError("components must be a non-empty list")
        self.strict = bool(strict)
        self._global_ss = bool(enable_ss_undesired_energy)
        self.losses = nn.ModuleList()
        weights = []
        self._component_names: list[str] = []
        self._component_types: list[str] = []
        self._component_log_keys: list[str] = []
        self._effective_ss: list[bool] = []
        for i, comp in enumerate(components):
            if "type" not in comp:
                raise KeyError(f"Each component must contain 'type', got: {comp}")
            # Use `or {}` to handle `args: null` in YAML (which returns None, not {})
            args = comp.get("args") or {}
            typ = comp["type"]
            loss_cls = LOSS_REGISTRY.get(typ)
            ov = comp.get("enable_ss_undesired_energy")
            effective_ss = self._global_ss if ov is None else bool(ov)
            if effective_ss and getattr(
                loss_cls, "SUPPORTS_SS_UNDESIRED_ENERGY", True
            ) is False:
                raise ValueError(
                    f"CompositeLoss component[{i}] {typ} does not support "
                    "enable_ss_undesired_energy=True; its clean-speech reference "
                    "estimator cannot supervise undesired/noise residuals. Set this "
                    "component's enable_ss_undesired_energy to false."
                )
            self.losses.append(loss_cls(**args))
            weights.append(float(comp.get("weight", 1.0)))
            self._component_types.append(typ)
            self._component_names.append(comp.get("name") or typ)
            # Stable TB / CSV key: YAML order + registry type (survives duplicate display names)
            self._component_log_keys.append(f"yaml_{i:03d}::{typ}")
            self._effective_ss.append(effective_ss)
        # 权重校验：NaN/Inf 权重会静默污染总 loss；全零权重则无任何监督信号。
        if not all(math.isfinite(w) for w in weights):
            raise ValueError(f"CompositeLoss weights must all be finite, got {weights}")
        if not any(w != 0.0 for w in weights):
            raise ValueError("CompositeLoss 需要一个非零权重")
        self.register_buffer("weights", torch.tensor(weights), persistent=False)
        # 权重为静态 python float 列表：热路径判断直接用，避免每 micro-batch
        # 对 GPU buffer 做 .item() 同步。
        self._weights_float = weights
        self._accepts_kwargs = [_accepts_var_keyword(fn) for fn in self.losses]
        # 分项日志先存 detached tensor，仅在 _last_breakdown 被消费时物化为
        # float（惰性 .item()），避免每个 micro-batch ~4 次 GPU 同步。
        self._breakdown_tensors: list[tuple[str, float, torch.Tensor, torch.Tensor]] | None = None
        self._breakdown_cache: list[tuple[str, float, float, float]] | None = None
        self._mr_cache_si, self._mr_cache_sj = (
            CompositeLoss._find_mr_cache_pair(self.losses)
        )

    @property
    def _last_breakdown(self) -> list[tuple[str, float, float, float]] | None:
        """分项日志 (log_key, weight, raw, weighted)，惰性物化。

        forward() 内只保留 detached tensor；首次被 trainer 日志 / epoch 汇总
        消费时才 .item() 同步并缓存结果，对外语义与原来一致（float 元组列表）。
        """
        tensors = self._breakdown_tensors
        if tensors is None:
            return None
        if self._breakdown_cache is None:
            self._breakdown_cache = [
                (key, w, float(raw.item()), float(weighted.item()))
                for key, w, raw, weighted in tensors
            ]
        return self._breakdown_cache

    @property
    def any_ss_enabled(self) -> bool:
        """True if any component uses SS (energy-weighted target / undesired) mixing."""
        return any(self._effective_ss)

    @staticmethod
    def _find_mr_cache_pair(losses: nn.ModuleList) -> tuple[int, int]:
        """First (i, j) pair of components eligible to share one MR-STFT cache.

        Task/class-agnostic (see ``_mr_cache_signature``): scan in YAML order and
        return only the first two distinct components with matching exact ordered
        ``(fft_sizes, hop_sizes)`` geometry and a ``stft_cache``-accepting forward(),
        regardless of concrete class. Later matching components are not paired.
        """
        sigs = [_mr_cache_signature(m) for m in losses]
        for i, sig_i in enumerate(sigs):
            if sig_i is None:
                continue
            for j in range(i + 1, len(losses)):
                if sigs[j] == sig_i:
                    return i, j
        return -1, -1

    def _forward_one(
        self,
        i: int,
        fn: nn.Module,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None,
        kwargs: dict,
        stft_cache: list | None,
        use_cache: bool,
    ) -> torch.Tensor:
        """Single component forward; ``use_cache`` only for the MR-STFT-cache-eligible pair."""
        if self._accepts_kwargs[i]:
            component_kwargs = kwargs
        elif bool(getattr(fn, "SUPPORTS_DIRECT_SPECTRA", False)):
            component_kwargs = {
                key: kwargs[key]
                for key in ("enhanced_spectra", "target_spectra")
                if key in kwargs
            }
        else:
            component_kwargs = {}
        if use_cache and stft_cache is not None:
            return fn(
                enhanced,
                clean,
                lengths,
                stft_cache=stft_cache,
                **component_kwargs,
            )
        return fn(enhanced, clean, lengths, **component_kwargs)

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        device = getattr(enhanced, "device", None)
        if not isinstance(device, torch.device):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bsz = 0
        undesired_enhanced = None
        undesired_ref = None
        speech_idx: torch.Tensor | None = None
        ss_needed = False
        stft_cache: list | None = None
        a_weights: torch.Tensor | None = None
        si, sj = self._mr_cache_si, self._mr_cache_sj
        preflight_error: Exception | None = None

        try:
            if not torch.is_tensor(enhanced) or not torch.is_tensor(clean):
                raise TypeError("CompositeLoss enhanced and clean must be tensors")
            bsz = enhanced.shape[0]
            device = enhanced.device
            if enhanced.ndim < 1 or clean.ndim < 1:
                raise ValueError("CompositeLoss enhanced and clean must have a batch dimension")
            if enhanced.shape != clean.shape:
                raise ValueError(
                    "CompositeLoss enhanced and clean must have identical shapes, got "
                    f"{tuple(enhanced.shape)} and {tuple(clean.shape)}"
                )
            if lengths is not None and (
                not torch.is_tensor(lengths)
                or lengths.ndim != 1
                or lengths.shape[0] != bsz
            ):
                raise ValueError(
                    f"CompositeLoss lengths must have shape ({bsz},), got "
                    f"{getattr(lengths, 'shape', None)}"
                )

            undesired_enhanced = kwargs.pop("undesired_enhanced", None)
            undesired_ref = kwargs.pop("undesired_ref", None)
            has_direct_spectra = any(
                key in kwargs for key in ("enhanced_spectra", "target_spectra")
            )
            if has_direct_spectra and any(
                weight != 0.0
                and self._effective_ss[index]
                and bool(getattr(component, "SUPPORTS_DIRECT_SPECTRA", False))
                for index, (component, weight) in enumerate(
                    zip(self.losses, self._weights_float)
                )
            ):
                raise ValueError(
                    "direct-spectrum loss does not support enable_ss_undesired_energy: "
                    "undesired prediction/reference spectra are not available"
                )

            def _as_bool_1d(name: str, t: torch.Tensor | None) -> torch.Tensor:
                if t is None:
                    return torch.zeros(bsz, device=device, dtype=torch.bool)
                if not torch.is_tensor(t):
                    raise TypeError(
                        f"{name} must be a [B] Tensor, got {type(t).__name__}"
                    )
                if t.dim() != 1 or t.shape[0] != bsz:
                    raise ValueError(
                        f"{name} must be a [B] Tensor, got shape {tuple(t.shape)} for B={bsz}"
                    )
                return t.to(device=device, dtype=torch.bool)

            is_pure_noise = kwargs.pop("is_pure_noise", None)
            is_pure_echo = kwargs.pop("is_pure_echo", None)
            is_pure_echo_noise = kwargs.pop("is_pure_echo_noise", None)
            if is_pure_noise is None and is_pure_echo is None and is_pure_echo_noise is None:
                speech_idx = None
            else:
                pn = _as_bool_1d("is_pure_noise", is_pure_noise)
                pe = _as_bool_1d("is_pure_echo", is_pure_echo)
                pen = _as_bool_1d("is_pure_echo_noise", is_pure_echo_noise)
                speech_idx = (~(pn | pe | pen)).nonzero(as_tuple=True)[0]

            ss_needed = any(
                self._effective_ss[j] and self._weights_float[j] != 0.0
                for j in range(len(self.losses))
            )
            if ss_needed:
                if undesired_ref is None:
                    noisy_kw = kwargs.get("noisy")
                    if noisy_kw is not None:
                        undesired_ref = noisy_kw - clean
                if undesired_enhanced is None or undesired_ref is None:
                    raise ValueError(
                        "CompositeLoss SS mixing (enable_ss_undesired_energy) requires "
                        "undesired_enhanced and undesired_ref tensors, or pass noisy=... "
                        "to derive undesired_ref = noisy - clean."
                    )
                if (
                    not torch.is_tensor(undesired_enhanced)
                    or not torch.is_tensor(undesired_ref)
                    or undesired_enhanced.shape != enhanced.shape
                    or undesired_ref.shape != clean.shape
                ):
                    raise ValueError(
                        "undesired_enhanced / undesired_ref must match enhanced/clean shape "
                        f"{tuple(enhanced.shape)}."
                    )

            if self.strict:
                preflight_error = _strict_finite_input_error(
                    [
                        ("enhanced", enhanced),
                        ("clean", clean),
                        ("undesired_enhanced", undesired_enhanced),
                        ("undesired_ref", undesired_ref),
                        *kwargs.items(),
                    ]
                )

            mr_ss = (
                (si >= 0 and self._effective_ss[si])
                or (sj >= 0 and self._effective_ss[sj])
            )
            if (
                preflight_error is None
                and si >= 0
                and not mr_ss
                and lengths is None
            ):
                schedule = _mr_cache_signature(self.losses[si])
                if schedule is not None:
                    fft_sizes, hop_sizes = schedule
                    stft_cache = compute_mr_stft_shared_cache(
                        enhanced, clean, fft_sizes, hop_sizes
                    )
            if preflight_error is None and ss_needed:
                a_weights = _ss_energy_a(clean, undesired_ref, lengths).float()
        except Exception as exc:
            if is_unrecoverable_cuda_exception(exc):
                raise
            preflight_error = exc

        _synchronize_preflight_error(preflight_error)
        deferred_error: Exception | None = None
        total = torch.zeros((), device=device, dtype=torch.float32)
        w32 = self.weights.float()
        raw_values: list[torch.Tensor] = []
        # 门控跳过/异常组件的图连接零：整个 forward 只物化一次。
        zero = _graph_connected_zero(enhanced)
        for i, fn in enumerate(self.losses):
            # 每个静态 capability 分量固定执行其所需的 mass SUM；零权重或本
            # rank 已失败时只贡献零质量，禁止调用 callback，collective 序列仍不变。
            speech_ref_component = _requires_speech_reference(fn)
            mean_mass_component = callable(getattr(fn, "ddp_global_mean_mass", None))
            component_enabled = self._weights_float[i] != 0.0
            valid = torch.zeros(bsz, device=device, dtype=torch.bool)
            per_sample_mass = torch.zeros(bsz, device=device, dtype=torch.float32)
            local_mass = torch.zeros((), device=device, dtype=torch.float32)
            if speech_ref_component:
                if component_enabled and deferred_error is None:
                    try:
                        valid = _speech_reference_valid_mask(fn, clean, lengths)
                        if speech_idx is not None:
                            metadata_valid = torch.zeros(
                                bsz, device=device, dtype=torch.bool
                            )
                            metadata_valid[speech_idx] = True
                            valid = valid & metadata_valid
                        per_sample_mass = _speech_reference_reduction_mass(
                            fn, clean, lengths
                        )
                        local_mass = (per_sample_mass * valid).sum()
                    except Exception as exc:
                        if is_unrecoverable_cuda_exception(exc):
                            raise
                        deferred_error = deferred_error or exc
                        valid.zero_()
                        local_mass.zero_()
                global_mass, world_size = _distributed_sum_scalar(local_mass)
            else:
                global_mass = local_mass
                world_size = 1

            mean_local_mass = torch.zeros((), device=device, dtype=torch.float32)
            if mean_mass_component:
                if component_enabled and deferred_error is None:
                    try:
                        mean_local_mass = _ddp_global_mean_mass(
                            fn, enhanced, clean, lengths, kwargs
                        )
                    except Exception as exc:
                        if is_unrecoverable_cuda_exception(exc):
                            raise
                        deferred_error = deferred_error or exc
                        mean_local_mass.zero_()
                mean_global_mass, mean_world_size = _distributed_sum_scalar(
                    mean_local_mass
                )
            else:
                mean_global_mass = mean_local_mass
                mean_world_size = 1

            if deferred_error is not None:
                raw_values.append(local_mass.new_zeros(()))
                continue
            if not component_enabled:
                raw_values.append(zero.float())
                continue

            try:
                component_kwargs = dict(kwargs)
                if undesired_ref is not None and bool(getattr(fn, "REQUIRES_UNDESIRED_REFERENCE", False)):
                    component_kwargs["undesired_ref"] = undesired_ref
                # 无 pure metadata 时保留 speech 分量既有 SS mixing 语义；mass
                # collective 已按静态 capability 执行，但 SS 分支不使用该归一化。
                if speech_ref_component and (
                    speech_idx is not None or not self._effective_ss[i]
                ):
                    effective_idx = (valid & (per_sample_mass > 0)).nonzero(
                        as_tuple=True
                    )[0]
                    # mass 非负，effective_idx 为空等价于 local_mass == 0；改用
                    # numel()（shape 属性）判断，省掉每 micro-batch 一次 host 同步。
                    if effective_idx.numel() == 0:
                        cur = zero
                    else:
                        if effective_idx.numel() < bsz:
                            e = enhanced.index_select(0, effective_idx)
                            c = clean.index_select(0, effective_idx)
                            le = (
                                lengths.index_select(0, effective_idx)
                                if lengths is not None
                                else None
                            )
                            skw = _subset_kwargs_for_batch(effective_idx, bsz, component_kwargs)
                        else:
                            e, c, le = enhanced, clean, lengths
                            skw = component_kwargs
                        cur = self._forward_one(
                            i,
                            fn,
                            e,
                            c,
                            le,
                            skw,
                            None,
                            False,
                        )
                        scale = local_mass * float(world_size) / global_mass
                        cur = cur * scale.to(cur.dtype)
                else:
                    use_mr_cache = (
                        stft_cache is not None
                        and (i == si or i == sj)
                        and (not self._effective_ss[i])
                    )
                    if self._effective_ss[i] and ss_needed and a_weights is not None:
                        assert undesired_enhanced is not None and undesired_ref is not None
                        acc = torch.zeros((), device=device, dtype=torch.float32)
                        for b in range(bsz):
                            skw_b = _subset_kwargs_for_index(component_kwargs, b, bsz)
                            lt = self._forward_one(
                                i,
                                fn,
                                enhanced[b : b + 1],
                                clean[b : b + 1],
                                lengths[b : b + 1] if lengths is not None else None,
                                skw_b,
                                None,
                                False,
                            )
                            skw_u = dict(skw_b)
                            if bool(getattr(fn, "REQUIRES_UNDESIRED_REFERENCE", False)):
                                skw_u["undesired_ref"] = clean[b : b + 1]
                            lu = self._forward_one(
                                i,
                                fn,
                                undesired_enhanced[b : b + 1],
                                undesired_ref[b : b + 1],
                                lengths[b : b + 1] if lengths is not None else None,
                                skw_u,
                                None,
                                False,
                            )
                            ab = a_weights[b]
                            acc = acc + (
                                ab * lt.float() + (1.0 - ab) * lu.float()
                            )
                        cur = acc / float(bsz)
                    else:
                        cur = self._forward_one(
                            i,
                            fn,
                            enhanced,
                            clean,
                            lengths,
                            component_kwargs,
                            stft_cache,
                            use_mr_cache,
                        )
                if mean_mass_component:
                    # 全零质量时 local 质量同样为 0，scale 自然为 0，得到与
                    # cur * 0.0 相同的图连接零；用 where 兜住 0/0 即可，无需
                    # 每个 micro-batch 一次 (mean_global_mass == 0).item() 同步。
                    safe_global_mass = torch.where(
                        mean_global_mass == 0,
                        torch.ones_like(mean_global_mass),
                        mean_global_mass,
                    )
                    scale = (
                        mean_local_mass * float(mean_world_size) / safe_global_mass
                    )
                    cur = cur * scale.to(cur.dtype)
                if self.strict and not bool(torch.isfinite(cur).item()):
                    raise StrictFiniteLossError(
                        f"CompositeLoss component[{i}] returned NaN/Inf"
                    )
            except Exception as exc:
                if is_unrecoverable_cuda_exception(exc):
                    raise
                deferred_error = exc
                cur = zero

            raw_values.append(cur)
            total = total + w32[i] * cur.float()

        # Composite 自身固定执行一次最终状态同步。单 rank 组件异常/strict NaN
        # 先延迟到所有后续 speech-reference mass collective 完成，再让所有 rank
        # 以相同错误退出，避免与 Trainer strict-finite collective 串台。
        local_failed = torch.tensor(
            deferred_error is not None,
            device=device,
            dtype=torch.int32,
        )
        failed_sum, world_size = _distributed_sum_scalar(local_failed)
        if bool(failed_sum.item()):
            if world_size <= 1 and deferred_error is not None:
                raise deferred_error
            local_detail = None
            if deferred_error is not None:
                local_detail = (
                    isinstance(deferred_error, StrictFiniteLossError),
                    f"rank {dist.get_rank() if world_size > 1 else 0}: "
                    f"{type(deferred_error).__name__}: {deferred_error}",
                )
            details = [local_detail]
            if world_size > 1:
                details = [None] * world_size
                with torch.inference_mode(False):
                    dist.all_gather_object(details, local_detail)
            messages = [detail[1] for detail in details if detail is not None]
            message = "; ".join(messages)
            if all(detail[0] for detail in details if detail is not None):
                raise StrictFiniteLossError(message) from deferred_error
            raise RuntimeError(message) from deferred_error

        # 只存 detached tensor，float 物化延迟到 _last_breakdown 被消费时。
        self._breakdown_tensors = [
            (
                self._component_log_keys[i],
                self._weights_float[i],
                raw_values[i].detach(),
                (w32[i] * raw_values[i]).detach(),
            )
            for i in range(len(self.losses))
        ]
        self._breakdown_cache = None
        # 所有活动分量均被门控跳过时，total 不连模型输出图；
        # 注意 all-pure batch 只有在活动分量都依赖非零 speech reference 时
        # 才会触发此兜底；MBRCL/MR-STFT 等 full-batch 分量不会因此跳过。
        # 注入图保留零兜底，保证 DDP hook 与优化器 backward 正常
        # （与 SISNRLoss (0.0*snr).sum() 同范式）；加法形式保留原值，
        # 输入本身不 requires_grad 的纯数值场景（测试/推理）不受影响。
        # 先置零非有限元素再乘零：fp16 AMP 下 sum 先做可能溢出为 inf，
        # inf*0=NaN；详见 _graph_connected_zero。
        if not total.requires_grad:
            total = total + _graph_connected_zero(enhanced)
        return total
