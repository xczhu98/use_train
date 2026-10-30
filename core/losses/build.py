"""Loss builder and strict wrapper.

Build loss instance from config dict. CompositeLoss is handled specially.
"""

from __future__ import annotations

import copy
import difflib
import inspect
import logging
from typing import Any

import torch
import torch.nn as nn

from core.losses.composite import CompositeLoss, StrictFiniteLossError
from core.stft import (
    LOSS_STFT_CONFIG_KEYS,
    inject_stft_defaults,
    resolve_stft_config,
)

logger = logging.getLogger(__name__)


class _StrictLossWrapper(nn.Module):
    """Defensive wrapper: validate loss I/O and enforce finite scalar output."""

    STRICT_FINITE_CHECKS = True

    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss
        self._breakdown_cache: list[tuple[str, float, float, float]] | None = None

    @property
    def _last_breakdown(self):
        """惰性物化内层 CompositeLoss 的分项（仅在日志/验证消费时 .item()）。

        与 ``CompositeLoss._last_breakdown`` 同语义：首次消费时物化并缓存，
        避免同一份 detached tensor 在每个 val batch 反复触发 GPU 同步。
        """
        tensors = getattr(self, "_breakdown_tensors", None)
        if tensors is None:
            return None
        if self._breakdown_cache is None:
            self._breakdown_cache = [
                (k, w, float(r.item()), float(wt.item())) for k, w, r, wt in tensors
            ]
        return self._breakdown_cache

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # strict 门禁一律显式 raise：assert 在 python -O / PYTHONOPTIMIZE=1 下
        # 会被整体剥离，strict 契约不能依赖它。
        # SSL spatial style: (prediction, target, lengths).  DP-IPD keeps an
        # explicit pair axis, so the canonical shape is [B,T,F,K,P,2].
        if len(args) >= 2 and torch.is_tensor(args[0]) and torch.is_tensor(args[1]) \
                and bool(getattr(self.loss, "SSL_SPATIAL_LOSS", False)):
            prediction, target = args[0], args[1]
            lengths = args[2] if len(args) >= 3 else kwargs.get("lengths")
            if prediction.shape != target.shape:
                raise ValueError(
                    f"SSL prediction/target shape mismatch: {tuple(prediction.shape)} != "
                    f"{tuple(target.shape)}"
                )
            if prediction.ndim != 6 or prediction.shape[-1] != 2:
                raise ValueError(
                    "SSL tensors must be [B, T, F, K, P, 2] (canonical DP-IPD), "
                    f"got {tuple(prediction.shape)}"
                )
            if lengths is not None:
                if not torch.is_tensor(lengths):
                    raise TypeError("lengths must be a Tensor")
                if lengths.ndim != 1 or lengths.shape[0] != prediction.shape[0]:
                    raise ValueError(
                        f"lengths must be [B], got {tuple(lengths.shape)}"
                    )
                if lengths.dtype not in (
                    torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
                ):
                    raise TypeError(f"lengths must be integer dtype, got {lengths.dtype}")
                if not (
                    bool((lengths >= 1).all())
                    and bool((lengths <= prediction.shape[1]).all())
                ):
                    raise ValueError(
                        f"lengths out of range [1, {prediction.shape[1]}], got {lengths.tolist()}"
                    )
            if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                raise StrictFiniteLossError("SSL strict loss input has NaN/Inf")
        # SE style: (enhanced, clean, lengths)
        elif len(args) >= 2 and torch.is_tensor(args[0]) and torch.is_tensor(args[1]):
            enhanced = args[0]
            clean = args[1]
            lengths = args[2] if len(args) >= 3 else kwargs.get("lengths")
            if bool(getattr(self.loss, "SEPARATION_LOSS", False)):
                if enhanced.dim() != 3:
                    raise ValueError(
                        f"enhanced must be [B, K, T], got {tuple(enhanced.shape)}"
                    )
                if clean.dim() != 3:
                    raise ValueError(
                        f"clean must be [B, K, T], got {tuple(clean.shape)}"
                    )
                if enhanced.shape != clean.shape:
                    raise ValueError(
                        f"enhanced/clean shape mismatch: {tuple(enhanced.shape)} vs "
                        f"{tuple(clean.shape)}"
                    )
                if lengths is not None:
                    if not torch.is_tensor(lengths):
                        raise TypeError("lengths must be a Tensor")
                    if lengths.dim() != 1 or lengths.shape[0] != enhanced.shape[0]:
                        raise ValueError(
                            f"lengths must be [B], got {tuple(lengths.shape)}"
                        )
                    if lengths.dtype not in (
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                        torch.uint8,
                    ):
                        raise TypeError(
                            f"lengths must be integer dtype, got {lengths.dtype}"
                        )
                    time = enhanced.shape[-1]
                    if not (
                        bool((lengths >= 1).all()) and bool((lengths <= time).all())
                    ):
                        raise ValueError(
                            f"lengths out of range [1, {time}], got {lengths.tolist()}"
                        )
                source_mask = kwargs.get("source_mask")
                if source_mask is not None:
                    if not torch.is_tensor(source_mask):
                        raise TypeError("source_mask must be a Tensor")
                    if source_mask.shape != enhanced.shape[:2]:
                        raise ValueError(
                            "source_mask must match enhanced [B, K], got "
                            f"{tuple(source_mask.shape)}"
                        )
                    if source_mask.dtype is not torch.bool:
                        raise TypeError(
                            f"source_mask must be bool, got {source_mask.dtype}"
                        )
                if not torch.isfinite(enhanced).all() or not torch.isfinite(clean).all():
                    raise StrictFiniteLossError(
                        "SS strict loss input has NaN/Inf"
                    )
                out = self.loss(*args, **kwargs)
                if not torch.is_tensor(out):
                    raise TypeError("loss output must be torch.Tensor")
                if out.ndim != 0:
                    raise ValueError(
                        f"loss must be scalar tensor, got shape {tuple(out.shape)}"
                    )
                if not torch.isfinite(out):
                    raise StrictFiniteLossError("loss output is NaN/Inf under strict mode")
                return out
            if enhanced.dim() != 2:
                raise ValueError(
                    f"enhanced must be [B, T], got {tuple(enhanced.shape)}"
                )
            if clean.dim() != 2:
                raise ValueError(f"clean must be [B, T], got {tuple(clean.shape)}")
            if enhanced.shape != clean.shape:
                raise ValueError(
                    f"enhanced/clean shape mismatch: {tuple(enhanced.shape)} vs "
                    f"{tuple(clean.shape)}"
                )
            if lengths is not None:
                if not torch.is_tensor(lengths):
                    raise TypeError("lengths must be a Tensor")
                if lengths.dim() != 1 or lengths.shape[0] != enhanced.shape[0]:
                    raise ValueError(
                        f"lengths must be [B], got {tuple(lengths.shape)}"
                    )
                # 防御校验：lengths 必须是整型且值域 1 <= lengths <= T
                if lengths.dtype not in (
                    torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
                ):
                    raise TypeError(f"lengths must be integer dtype, got {lengths.dtype}")
                T = enhanced.shape[1]
                if not (bool((lengths >= 1).all()) and bool((lengths <= T).all())):
                    raise ValueError(
                        f"lengths out of range [1, {T}], got {lengths.tolist()}"
                    )
        # AEC style: (batch_dict,)
        elif len(args) >= 1 and isinstance(args[0], dict):
            batch = args[0]
            required = ("mic", "nearend", "nearend_estimate")
            missing = [k for k in required if k not in batch]
            if missing:
                raise KeyError(f"AEC loss strict mode: missing keys {missing}")
            for k in required:
                t = batch[k]
                if not torch.is_tensor(t):
                    raise TypeError(f"AEC loss strict mode: '{k}' must be tensor")
                if not torch.isfinite(t).all():
                    raise StrictFiniteLossError(
                        f"AEC loss strict mode: '{k}' has NaN/Inf"
                    )
        out = self.loss(*args, **kwargs)
        # 透传 detached 分项 tensor；_last_breakdown 改为惰性 property，
        # 训练热路径（每 micro-batch 读取）不再触发 .item() 同步（review P2-1）
        if hasattr(self.loss, "_breakdown_tensors"):
            self._breakdown_tensors = getattr(self.loss, "_breakdown_tensors", None)
        # 分项已更新，作废上一次物化的 float 缓存
        self._breakdown_cache = None
        if not torch.is_tensor(out):
            raise TypeError("loss output must be torch.Tensor")
        if out.ndim != 0:
            raise ValueError(f"loss must be scalar tensor, got shape {tuple(out.shape)}")
        if not torch.isfinite(out):
            raise StrictFiniteLossError("loss output is NaN/Inf under strict mode")
        return out


def _loss_accepts_param(loss_cls: type, param: str) -> bool:
    """Return True if ``loss_cls.__init__`` has an explicit keyword param (not ``**kwargs``)."""
    try:
        sig = inspect.signature(loss_cls.__init__)
    except (ValueError, TypeError):
        return False
    p = sig.parameters.get(param)
    if p is None:
        return False
    return p.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    )


_SAMPLE_RATE_ALIASES = ("sample_rate", "sampling_rate", "sr", "fs")
_COMPOSITE_TOP_LEVEL_KEYS = frozenset(
    {"type", "components", "strict", "enable_ss_undesired_energy"}
)
_COMPOSITE_COMPONENT_KEYS = frozenset(
    {"type", "weight", "args", "name", "enable_ss_undesired_energy"}
)
_SINGLE_LOSS_TOP_LEVEL_KEYS = frozenset({"type", "args", "strict"})


def _reject_unknown_keys(
    mapping: dict,
    *,
    allowed: frozenset[str],
    source: str,
) -> None:
    """拒绝结构层未知键，并为常见拼写错误提供最近匹配提示。"""
    for key in mapping:
        if key in allowed:
            continue
        matches = (
            difflib.get_close_matches(key, sorted(allowed), n=1, cutoff=0.6)
            if isinstance(key, str)
            else []
        )
        hint = f"; did you mean {matches[0]!r}?" if matches else ""
        raise KeyError(f"{source} contains unknown key {key!r}{hint}")


def _validate_composite_config(cfg: dict) -> list[dict]:
    """校验 CompositeLoss 外围 YAML 结构，不介入子 loss 参数或数学语义。"""
    _reject_unknown_keys(
        cfg,
        allowed=_COMPOSITE_TOP_LEVEL_KEYS,
        source="CompositeLoss config",
    )
    components = cfg.get("components")
    if components is None:
        raise ValueError(
            "CompositeLoss requires 'components' field with sub-loss definitions"
        )
    if not isinstance(components, list):
        raise TypeError(
            "CompositeLoss 'components' must be a list, "
            f"got {type(components).__name__}"
        )
    if not components:
        raise ValueError(
            "CompositeLoss requires 'components' field with sub-loss definitions"
        )
    for index, component in enumerate(components):
        source = f"CompositeLoss components[{index}]"
        if not isinstance(component, dict):
            raise TypeError(
                f"{source} must be a dict, got {type(component).__name__}"
            )
        _reject_unknown_keys(
            component,
            allowed=_COMPOSITE_COMPONENT_KEYS,
            source=source,
        )
        args = component.get("args")
        if args is not None and not isinstance(args, dict):
            raise TypeError(
                f"{source}.args must be a dict or null, got {type(args).__name__}"
            )
    return copy.deepcopy(components)


def _validate_single_loss_config(cfg: dict) -> dict[str, Any]:
    """校验普通 loss 的外围 YAML 结构并返回独立构造参数。"""
    _reject_unknown_keys(
        cfg,
        allowed=_SINGLE_LOSS_TOP_LEVEL_KEYS,
        source="Single loss config",
    )
    args = cfg.get("args")
    if args is not None and not isinstance(args, dict):
        raise TypeError(
            "Single loss config.args must be a dict or null, "
            f"got {type(args).__name__}"
        )
    return dict(args or {})


def _validate_sample_rate(value: Any, *, source: str) -> int:
    """校验采样率上下文或显式构造参数为正整数。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{source} must be a positive integer, got {value!r}")
    return value


def _inject_sample_rate(
    args: dict[str, Any],
    sample_rate: int | None,
    loss_cls: type,
) -> dict[str, Any]:
    """按统一别名规则向 loss 构造参数注入采样率上下文。

    显式 ``args`` 中任一采样率别名通常优先于外部上下文；声明
    ``REQUIRES_MATCHING_STFT_CONTEXT`` 的 loss 必须与非空上下文一致。没有显式
    别名时，只按 ``sample_rate``、``sampling_rate``、``sr``、``fs`` 的顺序
    注入第一个显式声明的构造参数；仅有 ``**kwargs`` 不构成注入契约。
    """
    explicit_aliases = [key for key in _SAMPLE_RATE_ALIASES if key in args]
    for key in explicit_aliases:
        explicit_value = _validate_sample_rate(args[key], source=f"loss args.{key}")
        if (
            sample_rate is not None
            and getattr(loss_cls, "REQUIRES_MATCHING_STFT_CONTEXT", False)
            and explicit_value != sample_rate
        ):
            raise ValueError(
                f"{loss_cls.__name__} {key} context conflict: loss args.{key}="
                f"{explicit_value}, build_loss sample_rate={sample_rate}"
            )
    required_sample_rate = getattr(loss_cls, "REQUIRED_SAMPLE_RATE", None)
    if sample_rate is not None and required_sample_rate is not None:
        if sample_rate != required_sample_rate:
            raise ValueError(
                f"{loss_cls.__name__} requires sample_rate={required_sample_rate}, "
                f"but build_loss received sample_rate={sample_rate}"
            )
    if explicit_aliases or sample_rate is None:
        return args
    for key in _SAMPLE_RATE_ALIASES:
        if _loss_accepts_param(loss_cls, key):
            out = dict(args)
            out[key] = sample_rate
            return out
    return args


def _inject_stft_defaults(
    args: dict[str, Any],
    stft_cfg: dict[str, Any] | None,
    loss_cls: type,
) -> dict[str, Any]:
    """Return args with applicable framework STFT defaults injected.

    Loss classes with fixed pretrained-estimator geometry may declare
    ``USE_FRAMEWORK_STFT_DEFAULTS = False``; their explicit ``args`` remain intact,
    but the model top-level STFT context is never injected. Other classes receive
    only explicitly accepted ``LOSS_STFT_CONFIG_KEYS`` through ``setdefault``.
    """
    if not stft_cfg or getattr(loss_cls, "USE_FRAMEWORK_STFT_DEFAULTS", True) is False:
        return args
    accepted = tuple(
        key for key in LOSS_STFT_CONFIG_KEYS if _loss_accepts_param(loss_cls, key)
    )
    if getattr(loss_cls, "REQUIRES_MATCHING_STFT_CONTEXT", False):
        for key in accepted:
            if key not in args or key not in stft_cfg:
                continue
            explicit_value = args[key]
            context_value = stft_cfg[key]
            if key == "window":
                if isinstance(explicit_value, str):
                    explicit_value = explicit_value.lower()
                if isinstance(context_value, str):
                    context_value = context_value.lower()
            if explicit_value != context_value:
                raise ValueError(
                    f"{loss_cls.__name__} {key} context conflict: loss args.{key}="
                    f"{args[key]!r}, effective model stft.{key}={context_value!r}"
                )
    return inject_stft_defaults(args, stft_cfg, keys=accepted)


def _stft_center(stft_cfg: dict[str, Any] | None) -> bool:
    """Effective center value used by the framework STFT module."""
    if not stft_cfg:
        return True
    return bool(stft_cfg.get("center", True))


def build_loss(
    cfg: dict,
    stft_cfg: dict[str, Any] | None = None,
    *,
    sample_rate: int | None = None,
):
    """Build loss instance from config.

    When ``stft_cfg`` is provided (top-level YAML ``stft`` block), its effective
    ``n_fft`` / ``hop_length`` / ``win_length`` / ``window`` / ``center`` /
    ``normalized`` values are defaulted into each STFT-aware loss component via ``setdefault``
    (explicit ``args`` still win), except classes declaring
    ``REQUIRES_MATCHING_STFT_CONTEXT``: their explicit STFT args and construction
    sample rate must match non-null model context. Fixed-estimator losses may opt
    out through ``USE_FRAMEWORK_STFT_DEFAULTS = False``. ``sample_rate`` is optional
    construction-time context kept backward-compatible with the historical
    two-argument call. It is validated as a positive integer, checked against a
    loss's optional ``REQUIRED_SAMPLE_RATE``, then injected only into the first
    explicitly declared constructor alias in canonical order: ``sample_rate``,
    ``sampling_rate``, ``sr``, ``fs``. Any explicit alias in ``args`` wins after
    the context check;
    external context never injects a second alias. The same rules apply to
    composite components.

    CompositeLoss requires additional 'components' field:
      loss:
        type: CompositeLoss
        components:
          - type: SISNRLoss
            weight: 1.0
          - type: MultiResSTFTLoss
            weight: 0.5
    """
    from core.utils.registry import LOSS_REGISTRY

    if not isinstance(cfg, dict):
        raise TypeError(f"loss config must be dict, got {type(cfg)}")
    if stft_cfg is not None:
        if not isinstance(stft_cfg, dict):
            raise TypeError(
                "stft_cfg must be dict or null, "
                f"got {type(stft_cfg).__name__}"
            )
        _reject_unknown_keys(
            stft_cfg,
            allowed=frozenset(LOSS_STFT_CONFIG_KEYS),
            source="stft_cfg",
        )
        if "n_fft" in stft_cfg and "hop_length" in stft_cfg:
            stft_cfg = resolve_stft_config(stft_cfg)
    if sample_rate is not None:
        sample_rate = _validate_sample_rate(sample_rate, source="sample_rate")
    loss_type = cfg.get("type", "")
    strict = bool(cfg.get("strict", False))

    if loss_type == "CompositeLoss":
        # 校验并深拷贝结构，后续上下文注入不得修改调用方配置。
        components = _validate_composite_config(cfg)
        stft_center_value = _stft_center(stft_cfg)

        for comp in components:
            typ = comp.get("type")
            if not typ:
                raise KeyError(f"Each component must contain 'type', got: {comp}")
            loss_cls = LOSS_REGISTRY.get(typ)
            args = dict(comp.get("args") or {})
            explicit_center = args.get("center") if "center" in args else None
            args = _inject_sample_rate(args, sample_rate, loss_cls)
            args = _inject_stft_defaults(args, stft_cfg, loss_cls)
            comp["args"] = args
            if explicit_center is not None and explicit_center != stft_center_value:
                logger.warning(
                    "Loss component %s explicitly uses center=%s while the framework "
                    "STFT uses stft.center=%s (default True). This can cause frame "
                    "count / padding mismatch. Consider aligning them.",
                    typ,
                    explicit_center,
                    stft_center_value,
                )

        enable_ss = bool(cfg.get("enable_ss_undesired_energy", False))
        loss_obj = CompositeLoss(
            components,
            strict=strict,
            enable_ss_undesired_energy=enable_ss,
        )
    else:
        args = _validate_single_loss_config(cfg)
        loss_cls = LOSS_REGISTRY.get(loss_type)
        explicit_center = args.get("center") if "center" in args else None
        args = _inject_sample_rate(args, sample_rate, loss_cls)
        args = _inject_stft_defaults(args, stft_cfg, loss_cls)
        if explicit_center is not None and stft_cfg is not None:
            stft_center_value = _stft_center(stft_cfg)
            if explicit_center != stft_center_value:
                logger.warning(
                    "Loss %s explicitly uses center=%s while the framework STFT uses "
                    "stft.center=%s. This can cause frame count / padding mismatch.",
                    loss_type,
                    explicit_center,
                    stft_center_value,
                )
        loss_obj = LOSS_REGISTRY.build(loss_type, **args)
    return _StrictLossWrapper(loss_obj) if strict else loss_obj
