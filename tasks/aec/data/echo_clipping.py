"""Optional C-aec echo distortion: three mutually exclusive non-linear paths.

Design: docs/guides/aec/AEC_Data_Simulation.md

These are stateless functions with no dependency on ``DynamicMixingDataset``.
``tasks/aec/data/dynamic_dataset.py`` wires the C-aec block as: load aligned farend /
echo -> optional echo-device EQ -> select clipping path -> SER-primary level
selection and ERL coupling (``far_db = echo_db + erl_db``, constrained by
``far_db_range`` and applied to the farend reference) -> optional
``pre_ser_overload`` -> random ref/echo delay -> optional time-delay jitter ->
``post_ser_distortion`` or ``synced_ref_clip``. ``post_ser_distortion`` uses a
peak-relative threshold; ``pre_ser_overload`` and ``synced_ref_clip`` use an
absolute dBFS rail. ``pre_ser_overload`` skips SER and ERL/far_db coupling, so
its echo level is set by overload boost while the farend remains at its loaded
level. There is no ``farend_random_volume`` stage.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np

from core.data.post_augment import sox_clip
from core.data.simulation_utils import rng_uniform, weighted_choice

# C-aec canonical schema 的路径与各路径合法 variant。post 使用峰值相对阈值；
# pre/synced 使用绝对 dBFS rail，因此硬截幅名称显式写作 fixed_rail。
ECHO_CLIP_PATHS = frozenset({"post_ser_distortion", "pre_ser_overload", "synced_ref_clip"})
ECHO_CLIP_VARIANTS = frozenset({"hard", "tanh"})
ECHO_ABSOLUTE_CLIP_VARIANTS = frozenset({"fixed_rail", "tanh"})

_DEFAULT_PATH_PROBS = {
    "post_ser_distortion": 0.5,
    "pre_ser_overload": 0.3,
    "synced_ref_clip": 0.2,
}
_DEFAULT_PATH_CONFIGS: dict[str, dict[str, Any]] = {
    "post_ser_distortion": {
        "variant_probs": {"hard": 0.5, "tanh": 0.5},
        "peak_relative_threshold_range": (0.5, 0.95),
    },
    "pre_ser_overload": {
        "variant_probs": {"fixed_rail": 0.5, "tanh": 0.5},
        "overload_boost_db_range": (3.0, 18.0),
        "rail_dbfs_range": (-6.0, -1.0),
    },
    "synced_ref_clip": {
        "variant_probs": {"fixed_rail": 0.5, "tanh": 0.5},
        "rail_dbfs_range": (-6.0, -1.0),
    },
}
# Signals whose peak is below this are treated as silence / near-silence and
# short-circuit the distortion functions unchanged.
_SILENCE_PEAK_THRESHOLD: float = 1e-8

# 模型 reference 的采集点由 echo clipping 路径决定。
CAPTURE_POINT_PLAYBACK_INPUT = "playback_input_reference"
CAPTURE_POINT_SHARED_NONLINEAR = "shared_nonlinear_reference"


def capture_point_of(path: str | None) -> str:
    """echo clipping 路径 → capture point 语义标签（JAE-M11，见上方协议）。"""
    if path == "synced_ref_clip":
        return CAPTURE_POINT_SHARED_NONLINEAR
    return CAPTURE_POINT_PLAYBACK_INPUT


def _validate_weights(name: str, weights: dict[str, float]) -> None:
    """在进入 ``weighted_choice`` 前校验有限、非负且总和有限非零。"""
    values = np.asarray([float(value) for value in weights.values()], dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"{name} weights must be finite and non-negative, got {weights}")
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            f"{name} weights must sum to a finite value > 0, got {weights}"
        )


def _validate_range(
    name: str,
    value_range: tuple[float, float],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_open: bool = False,
) -> tuple[float, float]:
    """校验二元有限 range，并返回 float bounds。"""
    if len(value_range) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value_range!r}")
    lo, hi = float(value_range[0]), float(value_range[1])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"{name} values must be finite, got {value_range!r}")
    if lo > hi:
        raise ValueError(f"{name} must satisfy lo <= hi, got {(lo, hi)!r}")
    if minimum is not None and (lo <= minimum if minimum_open else lo < minimum):
        op = ">" if minimum_open else ">="
        raise ValueError(f"{name} lower bound must be {op} {minimum}, got {lo}")
    if maximum is not None and hi > maximum:
        raise ValueError(f"{name} upper bound must be <= {maximum}, got {hi}")
    return lo, hi


def _db_to_linear(name: str, db: float, *, require_float32: bool) -> float:
    """安全执行 dB→linear，拒绝下溢、上溢及 float32 不可表示的结果。"""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        linear = float(np.power(10.0, float(db) / 20.0))
    if require_float32:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            linear_float32 = np.float32(linear)
        if (
            not np.isfinite(linear)
            or linear <= 0.0
            or not np.isfinite(linear_float32)
            or linear_float32 <= 0.0
        ):
            kind = "linear rail" if "rail" in name else "gain"
            raise ValueError(
                f"{name} {kind} must be finite, positive, and representable as "
                f"float32, got {db!r} dB"
            )
    elif not np.isfinite(linear) or linear <= 0.0:
        raise ValueError(
            f"{name} must convert to a finite positive linear rail, got {db!r} dB"
        )
    return linear


def _strict_weights(
    name: str,
    raw: Any,
    *,
    allowed: frozenset[str],
    unknown_kind: str = "variant",
) -> dict[str, float]:
    """规范化权重 mapping，并拒绝 bool、未知 key 与非有限值。"""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be a mapping")
    weights: dict[str, float] = {}
    for raw_key, raw_value in raw.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{name} keys must be strings, got {raw_key!r}")
        key = raw_key
        if key not in allowed:
            raise ValueError(
                f"{name} has unknown {unknown_kind}(s) {[str(raw_key)]}; "
                f"expected subset of {sorted(allowed)}"
            )
        if (
            isinstance(raw_value, (bool, np.bool_))
            or not isinstance(raw_value, Real)
        ):
            raise ValueError(
                f"{name} weights must be finite and non-negative, got {raw!r}"
            )
        value = float(raw_value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name} weights must be finite and non-negative, got {raw!r}"
            )
        weights[key] = weights.get(key, 0.0) + value
    _validate_weights(name, weights)
    return weights


def _strict_range(
    name: str,
    raw: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_open: bool = False,
) -> tuple[float, float]:
    """规范化配置 range，避免字符串、bool 等值被隐式接受。"""
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {raw!r}")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in raw
    ):
        raise ValueError(f"{name} values must be finite, got {raw!r}")
    return _validate_range(
        name,
        (float(raw[0]), float(raw[1])),
        minimum=minimum,
        maximum=maximum,
        minimum_open=minimum_open,
    )


def normalize_echo_clipping_config(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """返回唯一 canonical C-aec echo-clipping 配置。

    Canonical 形式将参数放在对应路径下：
    ``paths.post_ser_distortion.{variant_probs,peak_relative_threshold_range}``、
    ``paths.pre_ser_overload.{variant_probs,overload_boost_db_range,rail_dbfs_range}``、
    ``paths.synced_ref_clip.{variant_probs,rail_dbfs_range}``。
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("aec_echo_clipping must be a mapping")
    cfg = dict(raw)
    unknown = set(cfg) - {"enabled", "prob", "path_probs", "paths"}
    if unknown:
        raise ValueError(
            f"aec_echo_clipping contains unknown keys: {sorted(map(str, unknown))!r}"
        )

    enabled = cfg.get("enabled", False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError(f"aec_echo_clipping.enabled must be a boolean, got {enabled!r}")
    prob_raw = cfg.get("prob", 0.25)
    if isinstance(prob_raw, (bool, np.bool_)) or not isinstance(prob_raw, Real):
        raise ValueError(
            f"aec_echo_clipping.prob must be a finite probability, got {prob_raw!r}"
        )
    prob = float(prob_raw)
    if not np.isfinite(prob) or not 0.0 <= prob <= 1.0:
        raise ValueError(f"aec_echo_clipping.prob must be in [0, 1], got {prob_raw!r}")

    path_probs = _strict_weights(
        "aec_echo_clipping.path_probs",
        cfg.get("path_probs", _DEFAULT_PATH_PROBS),
        allowed=ECHO_CLIP_PATHS,
        unknown_kind="path",
    )
    paths_raw = cfg.get("paths", {})
    if not isinstance(paths_raw, Mapping):
        raise ValueError("aec_echo_clipping.paths must be a mapping")
    unknown_paths = set(paths_raw) - ECHO_CLIP_PATHS
    if unknown_paths:
        raise ValueError(
            "aec_echo_clipping.paths has unknown path(s) "
            f"{sorted(map(str, unknown_paths))!r}; expected subset of {sorted(ECHO_CLIP_PATHS)}"
        )

    paths: dict[str, dict[str, Any]] = {}
    for path in sorted(ECHO_CLIP_PATHS):
        path_raw = paths_raw.get(path, {})
        if not isinstance(path_raw, Mapping):
            raise ValueError(f"aec_echo_clipping.paths.{path} must be a mapping")
        local = dict(path_raw)
        defaults = _DEFAULT_PATH_CONFIGS[path]
        allowed_keys = set(defaults)
        unknown_local = set(local) - allowed_keys
        if unknown_local:
            raise ValueError(
                f"aec_echo_clipping.paths.{path} contains unknown keys: "
                f"{sorted(map(str, unknown_local))!r}"
            )

        if path == "post_ser_distortion":
            variant_raw = local.get("variant_probs", defaults["variant_probs"])
            paths[path] = {
                "variant_probs": _strict_weights(
                    f"aec_echo_clipping.paths.{path}.variant_probs",
                    variant_raw,
                    allowed=ECHO_CLIP_VARIANTS,
                ),
                "peak_relative_threshold_range": _strict_range(
                    f"aec_echo_clipping.paths.{path}.peak_relative_threshold_range",
                    local.get(
                        "peak_relative_threshold_range",
                        defaults["peak_relative_threshold_range"],
                    ),
                    minimum=0.0,
                    maximum=1.0,
                    minimum_open=True,
                ),
            }
            continue

        variant_raw = local.get("variant_probs", defaults["variant_probs"])
        rail_name = f"aec_echo_clipping.paths.{path}.rail_dbfs_range"
        rail_range = _strict_range(
            rail_name,
            local.get("rail_dbfs_range", defaults["rail_dbfs_range"]),
        )
        if rail_range[1] > 0.0:
            raise ValueError(f"{rail_name} must be <= 0 dBFS, got {rail_range!r}")
        for rail_dbfs in rail_range:
            _db_to_linear(rail_name, rail_dbfs, require_float32=True)
        path_cfg = {
            "variant_probs": _strict_weights(
                f"aec_echo_clipping.paths.{path}.variant_probs",
                variant_raw,
                allowed=ECHO_ABSOLUTE_CLIP_VARIANTS,
            ),
            "rail_dbfs_range": rail_range,
        }
        if path == "pre_ser_overload":
            boost_name = f"aec_echo_clipping.paths.{path}.overload_boost_db_range"
            boost_range = _strict_range(
                boost_name,
                local.get("overload_boost_db_range", defaults["overload_boost_db_range"]),
            )
            if boost_range[0] < 0.0:
                raise ValueError(
                    f"{boost_name} must be non-negative, got {boost_range!r}"
                )
            for boost_db in boost_range:
                _db_to_linear(boost_name, boost_db, require_float32=True)
            path_cfg["overload_boost_db_range"] = boost_range
        paths[path] = path_cfg

    return {
        "enabled": bool(enabled),
        "prob": prob,
        "path_probs": path_probs,
        "paths": paths,
    }


def select_echo_clip_path(
    rng: np.random.Generator, prob: float, path_probs: dict[str, float]
) -> str | None:
    """Decide whether to trigger echo clipping and, if so, which path.

    Returns ``None`` when not triggered (probability ``1 - prob``), otherwise
    one of ``path_probs``' keys, chosen by weight.
    """
    prob = float(prob)
    if not np.isfinite(prob) or not 0.0 <= prob <= 1.0:
        raise ValueError(f"echo clipping prob must be finite and in [0, 1], got {prob!r}")
    _validate_weights("path_probs", path_probs)
    if prob <= 0.0 or float(rng.random()) >= prob:
        return None
    return weighted_choice(rng, path_probs)


def sample_peak_relative_clip_params(
    threshold_frac_range: tuple[float, float],
    variant_probs: dict[str, float],
    rng: np.random.Generator,
) -> tuple[float, str]:
    """一次抽取峰值相对截幅的 fraction 与 variant（先 fraction 后 variant）。"""
    lo, hi = _validate_range(
        "threshold_frac_range",
        threshold_frac_range,
        minimum=0.0,
        maximum=1.0,
        minimum_open=True,
    )
    _validate_weights("variant_probs", variant_probs)
    threshold_frac = rng_uniform(rng, lo, hi)
    variant = weighted_choice(rng, variant_probs)
    return threshold_frac, variant


def apply_peak_relative_clip(
    signal: np.ndarray,
    threshold_frac_range: tuple[float, float],
    variant_probs: dict[str, float],
    rng: np.random.Generator,
    *,
    threshold_frac: float | None = None,
    variant: str | None = None,
) -> tuple[np.ndarray, str]:
    """Clip *signal* at a threshold relative to its own peak.

    Threshold = ``signal``'s own peak * a value drawn from
    ``threshold_frac_range``. This (rather than a fixed dBFS threshold) keeps
    the clip probability stable regardless of upstream SER / volume scaling.
    ``threshold_frac`` and ``variant`` may be pre-sampled and reused across
    signals; omitted values are drawn in the fixed order (fraction, variant).

    Supports variants "hard" (``np.clip``) and "tanh" (``core.data.post_augment
    .sox_clip``, converting the linear threshold to dB). "soft" is
    intentionally not supported: ``post_augment.soft_clip`` normalizes to its
    own peak internally and ignores any external threshold.

    Returns ``(distorted_signal, variant)``. When ``signal``'s peak is below
    1e-8 (silence), valid pre-sampled parameters consume no RNG and the function
    returns ``(signal.copy(), "none")`` unchanged. Invalid explicitly supplied
    parameters are rejected before the silence short-circuit.
    """
    lo, hi = _validate_range(
        "threshold_frac_range",
        threshold_frac_range,
        minimum=0.0,
        maximum=1.0,
        minimum_open=True,
    )
    if threshold_frac is not None:
        threshold_frac = float(threshold_frac)
        if not np.isfinite(threshold_frac) or not lo <= threshold_frac <= hi:
            raise ValueError(
                "apply_peak_relative_clip: threshold_frac "
                f"{threshold_frac!r} outside threshold_frac_range {threshold_frac_range!r}"
            )
    if variant is not None and variant not in ECHO_CLIP_VARIANTS:
        raise ValueError(
            f"apply_peak_relative_clip: unknown variant {variant!r}; expected one of {sorted(ECHO_CLIP_VARIANTS)}"
        )
    if threshold_frac is None or variant is None:
        _validate_weights("variant_probs", variant_probs)

    signal = np.asarray(signal, dtype=np.float32)
    if not np.isfinite(signal).all():
        raise ValueError("apply_peak_relative_clip: signal contains non-finite values")
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak < _SILENCE_PEAK_THRESHOLD:
        return signal.copy(), "none"

    if threshold_frac is None and variant is None:
        threshold_frac, variant = sample_peak_relative_clip_params(
            (lo, hi), variant_probs, rng
        )
    else:
        if threshold_frac is None:
            threshold_frac = rng_uniform(rng, lo, hi)
        if variant is None:
            variant = weighted_choice(rng, variant_probs)

    threshold = peak * threshold_frac
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(
            "apply_peak_relative_clip: threshold must be finite and positive"
        )
    if variant == "hard":
        out = np.clip(signal, -threshold, threshold).astype(np.float32)
    elif variant == "tanh":
        with np.errstate(divide="ignore", invalid="ignore"):
            threshold_db = 20.0 * float(np.log10(threshold))
        if not np.isfinite(threshold_db):
            raise ValueError(
                "apply_peak_relative_clip: threshold_db must be finite"
            )
        _db_to_linear(
            "apply_peak_relative_clip: threshold_db",
            threshold_db,
            require_float32=True,
        )
        out = sox_clip(signal, threshold_db).astype(np.float32)
    else:
        raise ValueError(
            f"apply_peak_relative_clip: unknown variant {variant!r}; expected one of {sorted(ECHO_CLIP_VARIANTS)}"
        )
    if not np.isfinite(out).all():
        raise ValueError("apply_peak_relative_clip produced non-finite output")
    return out, variant


def sample_absolute_clip_params(
    rail_dbfs_range: tuple[float, float],
    variant_probs: dict[str, float],
    rng: np.random.Generator,
) -> tuple[float, str]:
    """抽取绝对 dBFS rail 与 ``fixed_rail``/``tanh`` variant。"""
    lo, hi = _validate_range("rail_dbfs_range", rail_dbfs_range, maximum=0.0)
    for bound in (lo, hi):
        _db_to_linear("sample_absolute_clip_params: rail_dbfs", bound, require_float32=True)
    _validate_weights("variant_probs", variant_probs)
    unknown = set(variant_probs) - ECHO_ABSOLUTE_CLIP_VARIANTS
    if unknown:
        raise ValueError(
            f"sample_absolute_clip_params: unknown variant {sorted(unknown)!r}; "
            f"expected one of {sorted(ECHO_ABSOLUTE_CLIP_VARIANTS)}"
        )
    rail_dbfs = rng_uniform(rng, lo, hi)
    variant = weighted_choice(rng, variant_probs)
    return float(rail_dbfs), variant


def summarize_absolute_rail_facts(
    echo_facts: Mapping[str, Any] | None,
    farend_facts: Mapping[str, Any] | None,
) -> dict[str, bool | int | float | None]:
    """汇总 absolute rail 的 echo/farend 两路事实；缺失路保持 ``None``。"""
    required = {"rail_hit", "clipped_sample_count", "sample_count"}
    valid_echo = echo_facts if echo_facts is not None and required.issubset(echo_facts) else None
    valid_farend = (
        farend_facts
        if farend_facts is not None and required.issubset(farend_facts)
        else None
    )
    rows = [facts for facts in (valid_echo, valid_farend) if facts is not None]
    clipped_count = (
        sum(int(facts["clipped_sample_count"]) for facts in rows) if rows else None
    )
    sample_count = sum(int(facts["sample_count"]) for facts in rows) if rows else None
    return {
        "echo_rail_hit": bool(valid_echo["rail_hit"]) if valid_echo is not None else None,
        "farend_rail_hit": (
            bool(valid_farend["rail_hit"]) if valid_farend is not None else None
        ),
        "rail_hit": any(bool(facts["rail_hit"]) for facts in rows) if rows else None,
        "clipped_sample_count": clipped_count,
        "rail_sample_count": sample_count,
        "clipped_sample_ratio": (
            float(clipped_count / sample_count)
            if sample_count
            else (0.0 if rows else None)
        ),
    }


def apply_absolute_rail_clip(
    signal: np.ndarray,
    rail_dbfs_range: tuple[float, float],
    variant_probs: dict[str, float],
    rng: np.random.Generator,
    *,
    rail_dbfs: float | None = None,
    variant: str | None = None,
    facts_out: dict[str, Any] | None = None,
) -> tuple[np.ndarray, str, float | None]:
    """按绝对 dBFS rail 截幅，并可记录输入真实越 rail 事实。"""
    lo, hi = _validate_range("rail_dbfs_range", rail_dbfs_range, maximum=0.0)
    for bound in (lo, hi):
        _db_to_linear(
            "apply_absolute_rail_clip: rail_dbfs",
            bound,
            require_float32=True,
        )
    if rail_dbfs is not None:
        rail_dbfs = float(rail_dbfs)
        if not np.isfinite(rail_dbfs) or not lo <= rail_dbfs <= hi:
            raise ValueError(
                f"apply_absolute_rail_clip: rail_dbfs {rail_dbfs!r} outside "
                f"rail_dbfs_range {rail_dbfs_range!r}"
            )
    if variant is not None and variant not in ECHO_ABSOLUTE_CLIP_VARIANTS:
        raise ValueError(
            f"apply_absolute_rail_clip: unknown variant {variant!r}; "
            f"expected one of {sorted(ECHO_ABSOLUTE_CLIP_VARIANTS)}"
        )
    if rail_dbfs is None or variant is None:
        _validate_weights("variant_probs", variant_probs)

    source = np.asarray(signal, dtype=np.float32)
    if not np.isfinite(source).all():
        raise ValueError("apply_absolute_rail_clip: signal contains non-finite values")
    sample_count = int(source.size)
    peak = float(np.max(np.abs(source))) if source.size else 0.0
    if peak < _SILENCE_PEAK_THRESHOLD:
        if facts_out is not None:
            facts_out.update(
                {
                    "sample_count": sample_count,
                    "clipped_sample_count": 0,
                    "clipped_sample_ratio": 0.0,
                    "rail_hit": False,
                }
            )
        return source.copy(), "none", rail_dbfs
    if rail_dbfs is None:
        rail_dbfs = float(rng_uniform(rng, lo, hi))
    if variant is None:
        variant = weighted_choice(rng, variant_probs)
    if variant not in ECHO_ABSOLUTE_CLIP_VARIANTS:
        raise ValueError(
            f"apply_absolute_rail_clip: unknown variant {variant!r}; "
            f"expected one of {sorted(ECHO_ABSOLUTE_CLIP_VARIANTS)}"
        )

    rail = _db_to_linear(
        "apply_absolute_rail_clip: rail_dbfs",
        rail_dbfs,
        require_float32=True,
    )
    clipped_sample_count = int(np.count_nonzero(np.abs(source) > rail))
    if facts_out is not None:
        facts_out.update(
            {
                "sample_count": sample_count,
                "clipped_sample_count": clipped_sample_count,
                "clipped_sample_ratio": (
                    float(clipped_sample_count / sample_count) if sample_count else 0.0
                ),
                "rail_hit": bool(clipped_sample_count > 0),
            }
        )
    if variant == "fixed_rail":
        out = np.clip(source, -rail, rail)
    else:
        out = sox_clip(source, rail_dbfs)
    out = np.asarray(out, dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError("apply_absolute_rail_clip produced non-finite output")
    return out, variant, rail_dbfs


def apply_playback_overload(
    echo_raw: np.ndarray,
    boost_db_range: tuple[float, float],
    dbfs_threshold_range: tuple[float, float],
    variant_probs: dict[str, float],
    rng: np.random.Generator,
    *,
    facts_out: dict[str, Any] | None = None,
) -> tuple[np.ndarray, float | None, str]:
    """先提升 echo，再按绝对 dBFS rail 截幅。

    Canonical variant 为 ``fixed_rail``/``tanh``；直接调用时仍接受旧 ``hard``
    别名。``facts_out`` 记录真实 waveform 变化、绝对 rail 与 boost 事实。
    """
    boost_lo, boost_hi = _validate_range(
        "boost_db_range", boost_db_range, minimum=0.0
    )
    rail_lo, rail_hi = _validate_range(
        "dbfs_threshold_range", dbfs_threshold_range, maximum=0.0
    )
    for bound in (boost_lo, boost_hi):
        _db_to_linear("apply_playback_overload: boost_db", bound, require_float32=True)
    for bound in (rail_lo, rail_hi):
        _db_to_linear(
            "apply_playback_overload: rail_dbfs",
            bound,
            require_float32=True,
        )
    if "hard" in variant_probs and "fixed_rail" in variant_probs:
        raise ValueError(
            "apply_playback_overload: variant_probs must not contain both "
            "hard and fixed_rail"
        )
    uses_legacy_hard = "hard" in variant_probs
    normalized_variants = {
        ("fixed_rail" if key == "hard" else key): value
        for key, value in variant_probs.items()
    }
    _validate_weights("variant_probs", normalized_variants)
    unknown = set(normalized_variants) - ECHO_ABSOLUTE_CLIP_VARIANTS
    if unknown:
        raise ValueError(
            f"apply_playback_overload: unknown variant {sorted(unknown)!r}; "
            f"expected one of {sorted(ECHO_ABSOLUTE_CLIP_VARIANTS)}"
        )

    source = np.asarray(echo_raw, dtype=np.float32)
    if not np.isfinite(source).all():
        raise ValueError("apply_playback_overload: echo_raw contains non-finite values")
    peak = float(np.max(np.abs(source))) if source.size else 0.0
    if peak < _SILENCE_PEAK_THRESHOLD:
        if facts_out is not None:
            facts_out.update(
                {
                    "applied": False,
                    "boost_db": None,
                    "rail_dbfs": None,
                    "rail": None,
                    "sample_count": int(source.size),
                    "clipped_sample_count": 0,
                    "clipped_sample_ratio": 0.0,
                    "rail_hit": False,
                }
            )
        return source.copy(), None, "none"
    boost_db = float(rng_uniform(rng, boost_lo, boost_hi))
    boost_gain = _db_to_linear(
        "apply_playback_overload: boost_db",
        boost_db,
        require_float32=True,
    )
    boosted = (source * boost_gain).astype(np.float32)
    if not np.isfinite(boosted).all():
        raise ValueError("apply_playback_overload produced non-finite boosted signal")
    variant = weighted_choice(rng, normalized_variants)
    rail_dbfs = float(rng_uniform(rng, rail_lo, rail_hi))
    rail_facts: dict[str, Any] = {}
    out, applied_variant, _ = apply_absolute_rail_clip(
        boosted,
        (rail_lo, rail_hi),
        normalized_variants,
        rng,
        rail_dbfs=rail_dbfs,
        variant=variant,
        facts_out=rail_facts,
    )
    if not np.isfinite(out).all():
        raise ValueError("apply_playback_overload produced non-finite output")
    applied = not np.array_equal(out, source)
    if facts_out is not None:
        facts_out.update(rail_facts)
        facts_out.update(
            {
                "applied": applied,
                "boost_db": boost_db,
                "rail_dbfs": rail_dbfs,
                "rail": _db_to_linear(
                    "apply_playback_overload: rail_dbfs",
                    rail_dbfs,
                    require_float32=True,
                ),
            }
        )
    returned_variant = (
        "hard"
        if uses_legacy_hard and applied_variant == "fixed_rail"
        else applied_variant
    )
    return out, boost_db, returned_variant


def ser_db_from_power_ratio(clean_speech_1d: np.ndarray, echo_1d: np.ndarray) -> float:
    """Compute the achieved SER (dB) of ``echo_1d`` against ``clean_speech_1d``.

    Inverse of ``tasks.aec.data.dynamic_dataset.scale_echo_for_ser_db``'s convention:
    negative dB means the echo is louder than the clean speech. Audit-only —
    does not scale anything.
    """
    s = np.asarray(clean_speech_1d, dtype=np.float32)
    e = np.asarray(echo_1d, dtype=np.float32)
    power_s = max(float(np.mean(s ** 2)), 1e-10)
    power_e = max(float(np.mean(e ** 2)), 1e-10)
    return float(20.0 * np.log10(np.sqrt(power_s) / np.sqrt(power_e)))
