"""Strict configuration primitives for dynamic balanced sampling."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from core.data import sampling_projector as _sampling_projector
from core.data import sampling_schema as _sampling_schema
from core.data.sampling_schema import (
    DEFAULT_ALPHA,
    DEFAULT_DIFFICULTY,
    DatasetSamplingConfig,
    LanguageSamplingConfig,
    NoiseSamplingConfig,
    SamplingConfig,
    SpeakerSamplingConfig,
    SpeechSamplingConfig,
    _canonicalize_dataset_config,
    _finite_number,
    _normalise_probs,
    _parse_fraction,
    _parse_speech,
    _probs_tuple,
    _validate_dataset_bounds,
    _validate_speaker_config,
    _validated_language_probs,
    difficulty_key,
)
from core.data.speech_quota_solver import (
    SolverBudget,
    SolverLeaf,
    SolverMargins,
    SolverStats,
    _leaf_key_sort_key,
    solve_leaf_quotas,
)


# Protect construction-time residue DP.  2M transitions keeps temporary Python
# mappings comfortably below typical training-process memory while accepting
# ordinary multi-SR virtual epochs; callers can reduce epoch quota, SR classes,
# or DDP granularity when their request exceeds this explicit bound.
_SR_DP_STATE_BUDGET = 250_000
_SR_DP_TRANSITION_BUDGET = 2_000_000
# Dict spreads, distance scans, and stable-key comparisons materialize quota
# vectors.  Count their quota-entry work separately from logical DP edges.
_SR_DP_MATERIALIZATION_BUDGET = 8_000_000


class _PlannerBudgetExceeded(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class SpeechLeafMetadata:
    """联合 speech planner 的原子 leaf 输入。

    leaf key 固定为 ``(language, dataset_id, canonical_speaker_or_none,
    batching_sr)``；``uid_indices`` 是 Dataset external index 空间内的稳定
    下标（planner 只用其容量，不在此选择具体 UID）。``effective_seconds``
    必须是正 ``Fraction``。
    """

    language: str
    dataset_id: str
    speaker: tuple[str, str] | None
    batching_sr: int
    uid_indices: tuple[int, ...]
    effective_seconds: Fraction


@dataclass(frozen=True)
class SpeechPlan:
    """Immutable quota-only plan for one independently sampled speech epoch."""

    requested: int
    effective_requested: int
    actual: int
    language_quotas: Mapping[str, int]
    global_dataset_quotas: Mapping[str, int]
    cell_quotas: Mapping[tuple[str, str, int], int]
    leaf_quotas: Mapping[tuple[str, str, tuple[str, str] | None, int], int]
    speaker_quotas: Mapping[tuple[str, str], int]
    sr_quotas: Mapping[int, int]
    summary: Mapping[str, Any]


def make_speech_plan(
    leaves: Iterable[Any],
    *,
    requested: int,
    world_size: int,
    batch_size: int,
    drop_last: bool = False,
    sampling_config: SpeechSamplingConfig | SamplingConfig | None = None,
    language_probs: Mapping[str, Real] | None = None,
    dataset_alpha: Real | None = None,
    max_dataset_ratio: Real | Fraction | None = None,
    budget: SolverBudget = SolverBudget(),
) -> SpeechPlan:
    """Build stable bounded quotas without selecting any UID or dataset index.

    输入可以是 ``SpeechLeafMetadata`` leaf 序列（新接口），也可以是旧式
    cell（带 ``language/dataset_id/batching_sr/uids/effective_seconds``
    属性的对象）；两者不得混用。未启用 speaker controller 时走 legacy
    fast path（既有分配管线，结果与旧实现逐位一致），其 SR residue 失败
    候选由联合 solver 接管后才允许 shrink；启用 speaker 时全路径由联合
    solver 求解（无 legacy 结果需要保持）。
    """
    _positive_int(requested, "requested")
    _positive_int(world_size, "world_size")
    _positive_int(batch_size, "batch_size")
    if type(drop_last) is not bool:
        raise ValueError("drop_last must be bool")
    if not isinstance(budget, SolverBudget):
        raise ValueError("budget must be a SolverBudget")
    config = _normalize_speech_inputs(
        sampling_config,
        language_probs=language_probs,
        dataset_alpha=dataset_alpha,
        max_dataset_ratio=max_dataset_ratio,
    )
    language_cfg = config.language if config is not None else None
    dataset_cfg = config.dataset if config is not None else None
    speaker_cfg = config.speaker if config is not None else None
    probs = (
        dict(language_cfg.probs) if language_cfg is not None else None
    )
    alpha = float(dataset_cfg.alpha) if dataset_cfg is not None else None
    ratio = dataset_cfg.max_ratio if dataset_cfg is not None else None
    speaker_cap = (
        speaker_cfg.max_samples_per_epoch if speaker_cfg is not None else None
    )
    if probs is not None and speaker_cap is None:
        probs = {key: float(value) for key, value in probs.items()}

    leaf_entries = _prepare_leaf_entries(list(leaves))
    eligible = [
        entry
        for entry in leaf_entries
        if probs is None or probs.get(entry["language"], 0.0) > 0
    ]
    if speaker_cap is not None and any(
        entry["speaker"] is None for entry in eligible
    ):
        # design §3：启用 speaker 时每个 eligible primary anchor 都必须可
        # 解析 canonical identity；None-speaker leaf 会绕过 absolute cap
        raise ValueError(
            "speaker sampling requires every eligible leaf to carry speaker "
            "metadata (canonical (dataset_id, speaker_id) identity)"
        )
    if probs is not None:
        present = {entry["language"] for entry in eligible}
        for language, probability in probs.items():
            if probability > 0 and language not in present:
                raise ValueError(f"positive-probability language {language!r} has no eligible cells")
    # 未启用 speaker 时 canonical speaker 统一为 None（按对外三元合并
    # leaf），不创建冗余 speaker 状态（design §3 末尾 / §4.2.8 fast path）
    canonical = (
        eligible if speaker_cap is not None else _merge_leaf_entries(eligible)
    )
    legacy_entries = [
        _legacy_entry(entry) for entry in _merge_leaf_entries(canonical)
    ]

    raw_capacity = sum(entry["capacity"] for entry in canonical)
    speaker_capacity = None
    if speaker_cap is None:
        effective_requested = min(requested, raw_capacity)
    else:
        identity_capacities = defaultdict(int)
        for entry in canonical:
            identity_capacities[entry["speaker"]] += entry["capacity"]
        speaker_capacity = sum(
            min(capacity, speaker_cap)
            for capacity in identity_capacities.values()
        )
        effective_requested = min(requested, raw_capacity, speaker_capacity)
    granularity = world_size * batch_size if drop_last else world_size
    projected = effective_requested - effective_requested % granularity

    controllers = (
        "language+dataset" if probs is not None and alpha is not None
        else "language" if probs is not None
        else "dataset" if alpha is not None
        else "none"
    )
    if speaker_cap is not None:
        controllers = f"{controllers}+speaker" if controllers != "none" else "speaker"
    search_context = _prepare_search_context(legacy_entries, probs, alpha)
    search_projected = _project_transport_actual(
        search_context, probs, ratio, projected, granularity
    )
    if speaker_cap is not None:
        search_projected = _project_speaker_margin_actual(
            canonical,
            search_context,
            probs,
            ratio,
            speaker_cap,
            search_projected,
            granularity,
        )
    if projected > 0 and search_projected == 0:
        # 保留一个最小 aligned candidate，维持稳定失败阶段诊断，并作为
        # projection 证书的运行时安全兜底。
        search_projected = granularity
    solver_stats = SolverStats(0, 0, 0)
    candidates = {}
    candidates_tried = 0
    last_failure = (
        "speaker_capacity"
        if projected == 0
        and speaker_capacity is not None
        and speaker_capacity < min(requested, raw_capacity)
        else "none"
    )
    selected = None
    try:
        for actual in range(search_projected, 0, -granularity):
            candidates_tried += 1
            if speaker_cap is None:
                trial = _allocate_controlled(
                    legacy_entries, actual, probs, alpha, ratio, granularity,
                    search_context,
                )
                ok, payload, aux = trial
                if ok:
                    selected = ("legacy", actual, payload)
                    _record_candidate(candidates, actual, None)
                    break
                reason = _FAILURE_REASON_CODES.get(payload, payload)
                if payload in ("sr_residue", "sr_granularity"):
                    references = {
                        (language, dataset, None, sr): Fraction(quota)
                        for (language, dataset, sr), quota in aux["cell_quotas"].items()
                    }
                    attempt = _solve_candidate(
                        canonical, actual,
                        aux["language_quotas"], aux["dataset_quotas"], None,
                        granularity, budget, solver_stats, references,
                    )
                    solver_stats = attempt.stats
                    if attempt.result is not None:
                        selected = ("solver", actual, attempt.result, aux)
                        _record_candidate(candidates, actual, None)
                        break
                _record_candidate(candidates, actual, reason)
                last_failure = reason
            else:
                reason, result, aux = _speaker_candidate(
                    canonical, actual, probs, alpha, ratio, speaker_cap,
                    granularity, search_context, budget, solver_stats,
                )
                solver_stats = aux["stats"]
                if result is not None:
                    selected = ("solver", actual, result, aux)
                    _record_candidate(candidates, actual, None)
                    break
                _record_candidate(candidates, actual, reason)
                last_failure = reason
    except _PlannerBudgetExceeded as exc:
        raise ValueError(
            f"{exc} (controllers={controllers}, projected={projected}, "
            f"granularity={granularity}, candidates_tried={candidates_tried}, "
            f"last_failure={exc.stage})"
        ) from None
    if selected is None:
        raise ValueError(
            "infeasible speech quota plan: no positive actual satisfies capacities "
            f"and DDP granularity (controllers={controllers}, projected={projected}, "
            f"granularity={granularity}, candidates_tried={candidates_tried}, "
            f"last_failure={last_failure})"
        )
    return _build_plan(
        selected=selected,
        canonical=canonical,
        all_legacy_entries=[
            _legacy_entry(entry) for entry in _merge_leaf_entries(leaf_entries)
        ],
        requested=requested,
        effective_requested=effective_requested,
        granularity=granularity,
        ratio=ratio,
        context=search_context,
        solver_stats=solver_stats,
        candidates=candidates,
        speaker_cap=speaker_cap,
    )


# fast-path 内部分阶段 reason 到细化 reason code 的映射（Step 6 冻结集合：
# language_capacity / global_dataset_cap / transport_structure /
# speaker_capacity / sr_residue；budget stage 名保持不变）
_FAILURE_REASON_CODES = {
    "language_marginal": "language_capacity",
    "dataset_marginal": "global_dataset_cap",
    "joint_transport": "transport_structure",
    "cell_allocation": "transport_structure",
    "sr_residue": "sr_residue",
    "sr_granularity": "sr_residue",
}


def _record_candidate(candidates, actual, reason):
    """按有限 reason 集合压缩候选摘要，搜索次数不受此限制。"""
    entry = candidates.get(reason)
    if entry is None:
        candidates[reason] = {
            "actual": actual,
            "lowest_actual": actual,
            "count": 1,
            "reason": reason,
        }
        return
    entry["lowest_actual"] = min(entry["lowest_actual"], actual)
    entry["count"] += 1


def _leaf_entry_sort_key(entry):
    """stable leaf order：委托 solver 模块的唯一排序契约实现。"""
    return _leaf_key_sort_key(entry["key"])


def _prepare_leaf_entries(items):
    """统一 leaf 输入：``SpeechLeafMetadata`` 直接校验；legacy cell 经
    ``_prepare_cells`` 转换（speaker 一律 None）。两类输入不得混用。"""
    if not items:
        return _leaf_entries_from_cells(items)
    leaf_flags = [isinstance(item, SpeechLeafMetadata) for item in items]
    if any(leaf_flags) and not all(leaf_flags):
        raise ValueError(
            "mixed SpeechLeafMetadata and legacy cell inputs are not supported"
        )
    if all(leaf_flags):
        return _prepare_leaves(items)
    return _leaf_entries_from_cells(items)


def _leaf_entries_from_cells(cells):
    prepared = _prepare_cells(cells)
    return [
        {
            "key": (entry["language"], entry["dataset"], None, entry["sr"]),
            "language": entry["language"],
            "dataset": entry["dataset"],
            "speaker": None,
            "sr": entry["sr"],
            "capacity": entry["capacity"],
            "seconds": entry["seconds"],
            "seconds_fraction": Fraction(str(entry["seconds"])),
        }
        for entry in prepared
    ]


def _prepare_leaves(leaves):
    """校验 ``SpeechLeafMetadata`` 序列并转为内部 leaf entry（含排序）。"""
    result = []
    seen_keys = set()
    seen_indices = set()
    for index, leaf in enumerate(leaves):
        if not isinstance(leaf.language, str) or not leaf.language.strip():
            raise ValueError(f"leaf {index}: language must be a non-empty string")
        if not isinstance(leaf.dataset_id, str) or not leaf.dataset_id.strip():
            raise ValueError(f"leaf {index}: dataset_id must be a non-empty string")
        speaker = leaf.speaker
        if speaker is not None:
            if not (
                isinstance(speaker, tuple)
                and len(speaker) == 2
                and all(isinstance(part, str) and part.strip() for part in speaker)
            ):
                raise ValueError(
                    f"leaf {index}: speaker must be a canonical "
                    "(dataset_id, speaker_id) tuple or None"
                )
            if speaker[0] != leaf.dataset_id:
                raise ValueError(
                    f"leaf {index}: speaker dataset {speaker[0]!r} "
                    f"must match dataset_id {leaf.dataset_id!r}"
                )
        sr = leaf.batching_sr
        if type(sr) is bool or not isinstance(sr, Integral) or sr <= 0:
            raise ValueError(f"leaf {index}: batching_sr must be a positive int")
        indices = leaf.uid_indices
        if not isinstance(indices, (tuple, list)) or not indices:
            raise ValueError(f"leaf {index}: uid_indices must be a non-empty sequence")
        if any(type(item) is bool or not isinstance(item, Integral) or item < 0 for item in indices):
            raise ValueError(f"leaf {index}: uid_indices must be non-negative ints")
        if len(set(indices)) != len(indices):
            raise ValueError(f"leaf {index}: duplicate uid_index within leaf")
        if seen_indices.intersection(indices):
            raise ValueError(f"leaf {index}: duplicate uid_index across leaves")
        seconds = leaf.effective_seconds
        if not isinstance(seconds, Fraction) or seconds <= 0:
            raise ValueError(
                f"leaf {index}: effective_seconds must be a positive Fraction"
            )
        try:
            seconds_float = float(seconds)
        except OverflowError:
            seconds_float = math.inf
        if not math.isfinite(seconds_float) or seconds_float <= 0:
            raise ValueError(
                f"leaf {index}: effective_seconds is outside the supported float range"
            )
        key = (leaf.language, leaf.dataset_id, speaker, int(sr))
        if key in seen_keys:
            raise ValueError(f"leaf {index}: duplicate leaf key {key!r}")
        seen_keys.add(key)
        seen_indices.update(indices)
        result.append(
            {
                "key": key,
                "language": leaf.language,
                "dataset": leaf.dataset_id,
                "speaker": speaker,
                "sr": int(sr),
                "capacity": len(indices),
                "seconds": seconds_float,
                "seconds_fraction": seconds,
            }
        )
    result.sort(key=_leaf_entry_sort_key)
    return result


def _supported_float(value, context):
    """在 public 数值边界把 exact 值转换为有限 float。"""
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        result = math.inf
    if not math.isfinite(result) or (value != 0 and result == 0.0):
        raise ValueError(f"{context} is outside the supported float range")
    return result


def _merge_leaf_entries(entries):
    """按对外三元合并 leaf（speaker 统一为 None）；输入须已按 leaf key 排序。"""
    merged = {}
    for entry in entries:
        language, dataset, _speaker, sr = entry["key"]
        triple = (language, dataset, sr)
        existing = merged.get(triple)
        if existing is None:
            merged[triple] = {
                "key": (language, dataset, None, sr),
                "language": language,
                "dataset": dataset,
                "speaker": None,
                "sr": sr,
                "capacity": entry["capacity"],
                "seconds": entry["seconds"],
                "seconds_fraction": entry["seconds_fraction"],
            }
        else:
            existing["capacity"] += entry["capacity"]
            existing["seconds_fraction"] += entry["seconds_fraction"]
            existing["seconds"] = _supported_float(
                existing["seconds_fraction"],
                f"cell {triple!r} effective_seconds",
            )
    return [merged[triple] for triple in sorted(merged)]


def _legacy_entry(entry):
    """merged leaf entry → legacy 三元 entry（fast path 管线输入）。"""
    language, dataset, _speaker, sr = entry["key"]
    return {
        "key": (language, dataset, sr),
        "language": language,
        "dataset": dataset,
        "sr": sr,
        "capacity": entry["capacity"],
        "seconds": entry["seconds"],
    }


def _solve_candidate(
    leaves, actual, language_quotas, dataset_quotas, speaker_cap,
    granularity, budget, stats, references,
):
    """单候选的联合 solver 调用（cumulative stats 跨候选累计）。"""
    margins = SolverMargins(
        language=(
            tuple(sorted(language_quotas.items())) if language_quotas else None
        ),
        dataset=(
            tuple(sorted(dataset_quotas.items())) if dataset_quotas else None
        ),
    )
    solver_leaves = [
        SolverLeaf(
            key=entry["key"],
            capacity=entry["capacity"],
            reference=references[entry["key"]],
        )
        for entry in leaves
    ]
    return solve_leaf_quotas(
        solver_leaves,
        actual=actual,
        margins=margins,
        speaker_cap=speaker_cap,
        sr_modulus=granularity,
        budget=budget,
        cumulative_stats=stats,
    )


def _speaker_candidate(
    leaves, actual, probs, alpha, ratio, speaker_cap,
    granularity, context, budget, stats,
):
    """speaker 启用路径的单候选求解：分阶段归因 preflight + 联合 solver。

    返回 ``(reason, result, aux)``；``reason is None`` 时 ``result`` 为
    ``SolverResult``。reason 细化：language_capacity / global_dataset_cap /
    transport_structure / speaker_capacity / sr_residue。
    """
    language_quotas = None
    if probs is not None:
        language_quotas = _bounded_hamilton(
            {key: value for key, value in probs.items() if value > 0},
            context["language_capacities"],
            actual,
        )
        if language_quotas is None:
            return "language_capacity", None, {"stats": stats}
    dataset_quotas = None
    if alpha is not None:
        dataset_caps = {
            key: _dataset_cap(context["dataset_capacities"][key], ratio, actual)
            for key in context["by_dataset"]
        }
        dataset_quotas = _bounded_hamilton(
            context["dataset_weights"], dataset_caps, actual
        )
        if dataset_quotas is None:
            return "global_dataset_cap", None, {"stats": stats}
        if probs is not None and _allocate_joint_transport(
            context["pair_capacities"], language_quotas, dataset_quotas
        ) is None:
            return "transport_structure", None, {"stats": stats}
    # speaker 总量 preflight：每个 identity 最多提供 min(总 capacity, cap)
    identity_capacity = defaultdict(int)
    none_capacity = 0
    for entry in leaves:
        if entry["speaker"] is None:
            none_capacity += entry["capacity"]
        else:
            identity_capacity[entry["speaker"]] += entry["capacity"]
    slack_total = sum(min(total, speaker_cap) for total in identity_capacity.values())
    if slack_total + none_capacity < actual:
        return "speaker_capacity", None, {"stats": stats}
    total_capacity = sum(entry["capacity"] for entry in leaves)
    # 自然目标：按 eligible UID capacity 比例（design §4.2.3）
    references = {
        entry["key"]: Fraction(actual) * entry["capacity"] / total_capacity
        for entry in leaves
    }
    attempt = _solve_candidate(
        leaves, actual, language_quotas, dataset_quotas, speaker_cap,
        granularity, budget, stats, references,
    )
    if attempt.result is None:
        # 判别探针：放松 SR 模约束（sr_modulus=1）重解。language/dataset/
        # transport/aggregate 均已 preflight（pair 级与 leaf 级对二维 margin
        # 问题等价），仍不可行说明失败来自 speaker 维度；可行则确为 SR
        # residue 冲突。探针计入同一累计 stats，且预算耗尽同样原样上抛。
        probe = _solve_candidate(
            leaves, actual, language_quotas, dataset_quotas, speaker_cap,
            1, budget, attempt.stats, references,
        )
        reason = "sr_residue" if probe.result is not None else "speaker_capacity"
        return reason, None, {"stats": probe.stats}
    aux = {
        "stats": attempt.stats,
        "language_quotas": language_quotas,
        "dataset_quotas": dataset_quotas,
    }
    return None, attempt.result, aux


def _margin_details(context, actual, ratio, language_quotas, dataset_quotas):
    """solver 路径的 language/dataset summary 明细（与 legacy 数值同式）。"""
    language_capacities = context["language_capacities"]
    language_details = (
        {
            key: {
                "target": language_quotas[key],
                "quota": language_quotas[key],
                "capacity": language_capacities[key],
            }
            for key in sorted(language_quotas)
        }
        if language_quotas
        else {}
    )
    dataset_details = {}
    if dataset_quotas:
        dataset_weights = context["dataset_weights"]
        dataset_capacities = context["dataset_capacities"]
        dataset_caps = {
            key: _dataset_cap(dataset_capacities[key], ratio, actual)
            for key in context["by_dataset"]
        }
        weight_scale = max(dataset_weights.values(), default=1.0)
        scaled_weight_total = math.fsum(
            weight / weight_scale for weight in dataset_weights.values()
        )
        dataset_details = {
            key: {
                "effective_seconds": context["dataset_seconds"][key],
                "scaled_effective_seconds": context["scaled_dataset_seconds"][key],
                "effective_seconds_scale": context["seconds_scale"],
                "weight": min(
                    dataset_weights[key] * context["display_weight_factor"],
                    math.nextafter(math.inf, 0.0),
                ),
                "scaled_weight": dataset_weights[key],
                "real_target": actual * (dataset_weights[key] / weight_scale) / scaled_weight_total,
                "cap": dataset_caps[key],
                "quota": dataset_quotas[key],
                "capacity": dataset_capacities[key],
            }
            for key in sorted(dataset_quotas)
        }
    return language_details, dataset_details


def _build_plan(
    *, selected, canonical, all_legacy_entries, requested, effective_requested,
    granularity, ratio, context, solver_stats, candidates, speaker_cap,
):
    """把 fast-path 或 solver 结果聚合成 public ``SpeechPlan``（含 summary）。"""
    mode, actual = selected[0], selected[1]
    solver_used = mode == "solver"
    if mode == "legacy":
        (
            cell_quotas,
            language_quotas,
            dataset_quotas,
            sr_quotas,
            language_details,
            dataset_details,
            compensation,
        ) = selected[2]
        leaf_quotas = {
            (language, dataset, None, sr): quota
            for (language, dataset, sr), quota in cell_quotas.items()
        }
        speaker_quotas = {}
        speaker_info = None
    else:
        result, aux = selected[2], selected[3]
        leaf_quotas = {key: quota for key, quota in result.leaf_quotas}
        cell_totals = defaultdict(int)
        sr_totals = defaultdict(int)
        speaker_usage = defaultdict(int)
        for (language, dataset, speaker, sr), quota in result.leaf_quotas:
            cell_totals[(language, dataset, sr)] += quota
            sr_totals[sr] += quota
            if speaker is not None:
                speaker_usage[speaker] += quota
        cell_quotas = {key: cell_totals[key] for key in sorted(cell_totals)}
        sr_quotas = {entry["sr"]: 0 for entry in canonical}
        for sr, quota in sr_totals.items():
            sr_quotas[sr] += quota
        language_quotas = aux["language_quotas"] or {}
        dataset_quotas = aux["dataset_quotas"] or {}
        language_details, dataset_details = _margin_details(
            context, actual, ratio,
            aux["language_quotas"], aux["dataset_quotas"],
        )
        if speaker_cap is None:
            # fast-path SR 兜底：compensation = pre-SR 分配 → solver 结果 diff
            pre = aux["cell_quotas"]
            compensation = tuple(
                {"cell": key, "from": pre.get(key, 0), "to": cell_quotas.get(key, 0)}
                for key in sorted(set(pre) | set(cell_quotas))
                if pre.get(key, 0) != cell_quotas.get(key, 0)
            )
            speaker_quotas = {}
            speaker_info = None
        else:
            compensation = ()
            speaker_quotas = {
                speaker: speaker_usage.get(speaker, 0)
                for speaker in sorted(
                    {
                        entry["speaker"]
                        for entry in canonical
                        if entry["speaker"] is not None
                    }
                )
            }
            leaf_quota_map = dict(result.leaf_quotas)
            estimated_primary_seconds = sum(
                (
                    entry["seconds_fraction"]
                    * Fraction(leaf_quota_map.get(entry["key"], 0), entry["capacity"])
                    for entry in canonical
                    if entry["speaker"] is not None
                ),
                Fraction(),
            )
            speaker_info = {
                "cap": speaker_cap,
                "usage": speaker_quotas,
                "binding": {
                    speaker: quota == speaker_cap
                    for speaker, quota in speaker_quotas.items()
                },
                "primary_samples": sum(speaker_usage.values()),
                "estimated_primary_seconds": _supported_float(
                    estimated_primary_seconds,
                    "speaker summary estimated_primary_seconds",
                ),
            }
    summary = {
        "requested": requested,
        "effective_requested": effective_requested,
        "actual": actual,
        "shrink_reason": None if actual == effective_requested else "ddp_or_capacity_projection",
        "controllers": {
            "language_enabled": context["language_enabled"],
            "dataset_enabled": context["dataset_enabled"],
        },
        "sr_compensation": compensation,
        "languages": language_details,
        "datasets": dataset_details,
        "sample_rates": {
            sr: {
                "quota": sr_quotas.get(sr, 0),
                "capacity": sum(
                    entry["capacity"] for entry in all_legacy_entries if entry["sr"] == sr
                ),
                "granularity": granularity,
            }
            for sr in sorted({entry["sr"] for entry in all_legacy_entries})
        },
        "joint_solver": {
            "version": 2,
            "fast_path": speaker_cap is None,
            "solver_used": solver_used,
            "controllers": {
                "language_enabled": context["language_enabled"],
                "dataset_enabled": context["dataset_enabled"],
                "speaker_enabled": speaker_cap is not None,
            },
            "solver_stats": {
                "states": solver_stats.states,
                "transitions": solver_stats.transitions,
                "materialized": solver_stats.materialized,
            },
            "candidates": [
                candidates[reason]
                for reason in sorted(
                    candidates, key=lambda value: (value is None, repr(value))
                )
            ],
            "speaker": speaker_info,
        },
    }
    return SpeechPlan(
        requested=requested,
        effective_requested=effective_requested,
        actual=actual,
        language_quotas=_frozen(language_quotas),
        global_dataset_quotas=_frozen(dataset_quotas),
        cell_quotas=_frozen(cell_quotas),
        leaf_quotas=_frozen(leaf_quotas),
        speaker_quotas=_frozen(speaker_quotas),
        sr_quotas=_frozen(sr_quotas),
        summary=_frozen(summary),
    )


def _speech_config(value: SpeechSamplingConfig | SamplingConfig | None) -> SpeechSamplingConfig:
    if isinstance(value, SamplingConfig):
        value = value.speech
    if value is None:
        return SpeechSamplingConfig()
    if not isinstance(value, SpeechSamplingConfig):
        raise ValueError("sampling_config must be SpeechSamplingConfig or SamplingConfig")
    return value


def _normalize_speech_inputs(
    sampling_config: SamplingConfig | SpeechSamplingConfig | None,
    *,
    language_probs: Mapping[str, Real] | None = None,
    dataset_alpha: Real | None = None,
    max_dataset_ratio: Real | Fraction | None = None,
) -> SpeechSamplingConfig | None:
    """兼容 API 边界：direct overrides 统一归一化为 nested schema。

    direct 参数（language_probs/dataset_alpha/max_dataset_ratio）只允许
    在没有对应 speech config 的旧调用中出现；任何 direct 参数与
    ``sampling_config.speech`` 同时存在均抛错。config 路径的
    ``language.probs`` 严格校验（键集合、非负、有限、和为 1），dataset
    子块规范化为 ``Fraction`` 并校验范围；direct 路径沿用旧 flat 语义：
    language_probs 严格校验并归一化，dataset 缺省值（alpha=1/2、
    ratio=3/10）在此补齐。public planner 无参数调用恢复历史 language+dataset
    默认；显式 ``SamplingConfig().speech is None`` 仍表示禁用 balanced sampling。
    返回统一 nested ``SpeechSamplingConfig``，planner core 不分支判断来源。
    """
    direct_enabled = any(
        argument is not None
        for argument in (language_probs, dataset_alpha, max_dataset_ratio)
    )
    if (
        isinstance(sampling_config, SamplingConfig)
        and sampling_config.speech is None
        and not direct_enabled
    ):
        return None
    config = _speech_config(sampling_config)
    wrapper_types = (
        ("language", config.language, LanguageSamplingConfig),
        ("dataset", config.dataset, DatasetSamplingConfig),
        ("speaker", config.speaker, SpeakerSamplingConfig),
    )
    for field, value, expected_type in wrapper_types:
        if value is not None and type(value) is not expected_type:
            raise ValueError(
                f"speech.{field} must be a {expected_type.__name__} or None"
            )
    if config.has_controllers and direct_enabled:
        raise ValueError(
            "direct speech overrides cannot be combined with sampling_config.speech"
        )
    if config.has_controllers:
        if config.language is not None:
            _validated_language_probs(config.language.probs)
        if config.speaker is not None:
            _validate_speaker_config(config.speaker)
        dataset = (
            _canonicalize_dataset_config(config.dataset)
            if config.dataset is not None
            else None
        )
        return SpeechSamplingConfig(
            language=config.language,
            dataset=dataset,
            speaker=config.speaker,
        )
    if not direct_enabled:
        if sampling_config is not None:
            return None
        return SpeechSamplingConfig(
            language=LanguageSamplingConfig((
                ("en", Fraction(0)),
                ("other", Fraction(0)),
                ("zh", Fraction(1)),
            )),
            dataset=DatasetSamplingConfig(Fraction(1, 2), Fraction(3, 10)),
        )

    language = None
    if language_probs is not None:
        language = LanguageSamplingConfig(
            probs=_probs_tuple(_normalise_probs(language_probs))
        )
    dataset = None
    if dataset_alpha is not None or max_dataset_ratio is not None:
        alpha = (
            _parse_fraction(dataset_alpha, "dataset_alpha")
            if dataset_alpha is not None
            else Fraction(1, 2)
        )
        max_ratio = (
            _parse_fraction(max_dataset_ratio, "max_dataset_ratio")
            if max_dataset_ratio is not None
            else Fraction(3, 10)
        )
        _validate_dataset_bounds(alpha, max_ratio, "dataset_alpha", "max_dataset_ratio")
        dataset = DatasetSamplingConfig(alpha=alpha, max_ratio=max_ratio)
    return SpeechSamplingConfig(language=language, dataset=dataset)


def _group_entries(entries, key_fn):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[key_fn(entry)].append(entry)
    return dict(grouped)


_exact_weight_distribution = _sampling_projector._exact_weight_distribution
_fraction_ceiling = _sampling_projector._fraction_ceiling
_floor_sum_unsigned = _sampling_projector._floor_sum_unsigned
_floor_sum_prefix = _sampling_projector._floor_sum_prefix
_floor_sum_range = _sampling_projector._floor_sum_range
_max_dataset_segment_candidate = _sampling_projector._max_dataset_segment_candidate


def _prepare_search_context(entries, probs, alpha):
    by_language = _group_entries(entries, lambda entry: entry["language"])
    by_dataset = _group_entries(entries, lambda entry: entry["dataset"])
    by_pair = _group_entries(
        entries, lambda entry: (entry["language"], entry["dataset"])
    )

    def capacities(groups):
        return {
            key: sum(entry["capacity"] for entry in values)
            for key, values in groups.items()
        }

    def cell_weights(groups):
        return {
            key: {entry["key"]: entry["seconds"] for entry in values}
            for key, values in groups.items()
        }

    def cell_capacities(groups):
        return {
            key: {entry["key"]: entry["capacity"] for entry in values}
            for key, values in groups.items()
        }

    seconds_scale = max((entry["seconds"] for entry in entries), default=1.0)
    scaled_dataset_seconds = {
        key: math.fsum(entry["seconds"] / seconds_scale for entry in values)
        for key, values in by_dataset.items()
    }
    dataset_seconds = {}
    for key, values in by_dataset.items():
        try:
            seconds = math.fsum(entry["seconds"] for entry in values)
        except OverflowError:
            seconds = math.nextafter(math.inf, 0.0)
        dataset_seconds[key] = seconds
    dataset_weights = (
        {
            key: 1.0 if alpha == 0 else seconds ** alpha
            for key, seconds in scaled_dataset_seconds.items()
        }
        if alpha is not None else {}
    )
    display_weight_factor = (
        1.0 if alpha in (None, 0) else seconds_scale ** alpha
    )
    return {
        "by_language": by_language,
        "by_dataset": by_dataset,
        "by_pair": by_pair,
        "language_capacities": capacities(by_language),
        "dataset_capacities": capacities(by_dataset),
        "pair_capacities": capacities(by_pair),
        "language_cell_weights": cell_weights(by_language),
        "language_cell_capacities": cell_capacities(by_language),
        "dataset_cell_weights": cell_weights(by_dataset),
        "dataset_cell_capacities": cell_capacities(by_dataset),
        "pair_cell_weights": cell_weights(by_pair),
        "pair_cell_capacities": cell_capacities(by_pair),
        "dataset_seconds": dataset_seconds,
        "scaled_dataset_seconds": scaled_dataset_seconds,
        "seconds_scale": seconds_scale,
        "dataset_weights": dataset_weights,
        "display_weight_factor": display_weight_factor,
        "language_enabled": probs is not None,
        "dataset_enabled": alpha is not None,
        "srs": sorted({entry["sr"] for entry in entries}),
    }


def _allocate_margin(target, weights, capacities, cell_weights, cell_capacities):
    quotas = _bounded_hamilton(weights, capacities, target)
    if quotas is None:
        return None
    cell_quotas = {}
    for key, quota in sorted(quotas.items()):
        assigned = _bounded_hamilton(
            cell_weights[key], cell_capacities[key], quota
        )
        if assigned is None:
            return None
        cell_quotas.update(assigned)
    return quotas, cell_quotas


def _allocate_joint_transport(pair_capacity, language_quotas, dataset_quotas):
    if sum(language_quotas.values()) != sum(dataset_quotas.values()):
        return None

    source, sink = ("source",), ("sink",)
    residual = defaultdict(dict)

    def add_edge(left, right, capacity):
        residual[left][right] = capacity
        residual[right].setdefault(left, 0)

    for language, quota in sorted(language_quotas.items()):
        add_edge(source, ("language", language), quota)
    for (language, dataset), capacity in sorted(pair_capacity.items()):
        add_edge(("language", language), ("dataset", dataset), capacity)
    for dataset, quota in sorted(dataset_quotas.items()):
        add_edge(("dataset", dataset), sink, quota)

    total_flow = 0
    target = sum(language_quotas.values())
    while total_flow < target:
        parent = {source: None}
        queue = [source]
        for node in queue:
            neighbours = residual[node]
            for neighbour in sorted(neighbours, key=repr):
                if residual[node][neighbour] > 0 and neighbour not in parent:
                    parent[neighbour] = node
                    queue.append(neighbour)
                    if neighbour == sink:
                        break
            if sink in parent:
                break
        if sink not in parent:
            return None
        amount = target - total_flow
        node = sink
        while parent[node] is not None:
            amount = min(amount, residual[parent[node]][node])
            node = parent[node]
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            residual[previous][node] -= amount
            residual[node][previous] += amount
            node = previous
        total_flow += amount

    return {
        pair: capacity - residual[("language", pair[0])][("dataset", pair[1])]
        for pair, capacity in sorted(pair_capacity.items())
    }


def _allocate_controlled(
    entries, actual, probs, alpha, ratio, granularity, context=None
):
    """legacy fast-path 分配管线。

    返回 ``(ok, payload, aux)``：成功时 payload 为聚合结果元组；失败时
    payload 为内部分阶段 reason 字符串，SR 类失败（sr_residue /
    sr_granularity）的 aux 携带 pre-SR cell 分配与两组 margin，供联合
    solver 兜底构造 reference。
    """
    context = context or _prepare_search_context(entries, probs, alpha)
    language_enabled = context["language_enabled"]
    dataset_enabled = context["dataset_enabled"]
    by_language = context["by_language"]
    by_dataset = context["by_dataset"]
    language_capacities = context["language_capacities"]
    dataset_capacities = context["dataset_capacities"]

    language_quotas = {}
    if language_enabled:
        language_quotas = _bounded_hamilton(
            {key: value for key, value in probs.items() if value > 0},
            language_capacities,
            actual,
        )
        if language_quotas is None:
            return False, "language_marginal", {}

    dataset_quotas = {}
    dataset_weights = context["dataset_weights"]
    dataset_caps = {}
    if dataset_enabled:
        dataset_caps = {
            key: _dataset_cap(dataset_capacities[key], ratio, actual)
            for key in by_dataset
        }
        dataset_quotas = _bounded_hamilton(dataset_weights, dataset_caps, actual)
        if dataset_quotas is None:
            return False, "dataset_marginal", {}

    if language_enabled and dataset_enabled:
        pair_quotas = _allocate_joint_transport(
            context["pair_capacities"], language_quotas, dataset_quotas
        )
        if pair_quotas is None:
            return False, "joint_transport", {}
        groups = context["by_pair"]
        cell_quotas = {}
        for pair, quota in sorted(pair_quotas.items()):
            assigned = _bounded_hamilton(
                context["pair_cell_weights"][pair],
                context["pair_cell_capacities"][pair],
                quota,
            )
            if assigned is None:
                return False, "cell_allocation", {}
            cell_quotas.update(assigned)
        allocated = (pair_quotas, cell_quotas)
    elif language_enabled:
        allocated = _allocate_margin(
            actual, {key: value for key, value in probs.items() if value > 0},
            language_capacities,
            context["language_cell_weights"], context["language_cell_capacities"],
        )
        groups = by_language
    elif dataset_enabled:
        allocated = _allocate_margin(
            actual, dataset_weights, dataset_caps,
            context["dataset_cell_weights"], context["dataset_cell_capacities"],
        )
        groups = by_dataset
    else:
        return False, "cell_allocation", {}
    if allocated is None:
        return False, "cell_allocation", {}
    _, cell_quotas = allocated
    sr_aux = {
        "cell_quotas": dict(cell_quotas),
        "language_quotas": language_quotas or None,
        "dataset_quotas": dataset_quotas or None,
    }
    if len(context["srs"]) <= 1:
        compensation = ()
    else:
        compensation = _compensate_sr(groups, cell_quotas, granularity)
    if compensation is None:
        return False, "sr_residue", sr_aux
    sr_quotas = defaultdict(int)
    for entry in entries:
        sr_quotas[entry["sr"]] += cell_quotas[entry["key"]]
    if any(quota % granularity for quota in sr_quotas.values()):
        return False, "sr_granularity", sr_aux

    language_details = {
        key: {
            "target": language_quotas[key],
            "quota": language_quotas[key],
            "capacity": language_capacities[key],
        }
        for key in sorted(language_quotas)
    }
    weight_scale = max(dataset_weights.values(), default=1.0)
    scaled_weight_total = math.fsum(
        weight / weight_scale for weight in dataset_weights.values()
    )
    dataset_details = {
        key: {
            "effective_seconds": context["dataset_seconds"][key],
            "scaled_effective_seconds": context["scaled_dataset_seconds"][key],
            "effective_seconds_scale": context["seconds_scale"],
            "weight": min(
                dataset_weights[key] * context["display_weight_factor"],
                math.nextafter(math.inf, 0.0),
            ),
            "scaled_weight": dataset_weights[key],
            "real_target": actual * (dataset_weights[key] / weight_scale) / scaled_weight_total,
            "cap": dataset_caps[key],
            "quota": dataset_quotas[key],
            "capacity": dataset_capacities[key],
        }
        for key in sorted(dataset_quotas)
    }
    return True, (
        cell_quotas, language_quotas, dataset_quotas, dict(sr_quotas),
        language_details, dataset_details, compensation,
    ), {}


def _prepare_cells(cells: Iterable[Any]) -> list[dict[str, Any]]:
    result, keys, seen_uids = [], set(), set()
    for index, cell in enumerate(cells):
        try:
            language, dataset, sr, uids = cell.language, cell.dataset_id, cell.batching_sr, cell.uids
        except AttributeError as exc:
            raise ValueError(f"cell {index}: missing required sampling field") from exc
        if not isinstance(language, str) or not language.strip():
            raise ValueError(f"cell {index}: language must be a non-empty string")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError(f"cell {index}: dataset_id must be a non-empty string")
        if type(sr) is bool or not isinstance(sr, Integral) or sr <= 0:
            raise ValueError(f"cell {index}: batching_sr must be a positive int")
        if not isinstance(uids, (tuple, list)) or not uids:
            raise ValueError(f"cell {index}: uids must be a non-empty sequence")
        if any(not isinstance(uid, str) or not uid.strip() for uid in uids):
            raise ValueError(f"cell {index}: every uid must be a non-empty string")
        if len(set(uids)) != len(uids):
            raise ValueError(f"cell {index}: duplicate uid within cell")
        if seen_uids.intersection(uids):
            raise ValueError(f"cell {index}: duplicate uid across cells")
        key = (language, dataset, int(sr))
        if key in keys:
            raise ValueError(f"cell {index}: duplicate cell key {key!r}")
        try:
            seconds = _finite_number(cell.effective_seconds, "effective_seconds")
        except ValueError as exc:
            raise ValueError(f"cell {index}: {exc}") from exc
        if seconds <= 0:
            raise ValueError(f"cell {index}: effective_seconds must be positive")
        keys.add(key)
        seen_uids.update(uids)
        result.append({"key": key, "language": language, "dataset": dataset, "sr": int(sr), "capacity": len(uids), "seconds": seconds})
    return sorted(result, key=lambda entry: entry["key"])


def _bounded_hamilton(weights: Mapping[Any, float], capacities: Mapping[Any, int], total: int) -> dict[Any, int] | None:
    if total > sum(capacities.get(key, 0) for key in weights):
        return None
    exact = all(isinstance(value, Fraction) for value in weights.values())
    quotas = {key: 0 for key in sorted(weights)}
    remaining = total
    active = set(weights)
    while remaining and active:
        if exact:
            denominator = sum((weights[key] for key in active), Fraction())
            if denominator <= 0:
                return None
            ideals = {
                key: Fraction(remaining) * weights[key] / denominator
                for key in active
            }
            floor = lambda value: value.numerator // value.denominator
        else:
            weight_scale = max(weights[key] for key in active)
            if weight_scale <= 0:
                return None
            scaled = {key: weights[key] / weight_scale for key in active}
            denominator = math.fsum(scaled.values())
            ideals = {
                key: remaining * (scaled[key] / denominator)
                for key in active
            }
            floor = math.floor
        floors = {
            key: min(capacities[key] - quotas[key], int(floor(ideals[key])))
            for key in active
        }
        added = sum(floors.values())
        for key, amount in floors.items():
            quotas[key] += amount
        remaining -= added
        if remaining == 0:
            break
        eligible = sorted(
            (key for key in active if quotas[key] < capacities[key]),
            key=lambda key: (-(ideals[key] - floor(ideals[key])), key),
        )
        if not eligible:
            return None
        for key in eligible:
            if remaining == 0:
                break
            quotas[key] += 1
            remaining -= 1
        active = {key for key in active if quotas[key] < capacities[key]}
    return quotas if remaining == 0 else None


def _dataset_cap(capacity, ratio, target):
    """Cap a dataset using exact Fraction ceiling arithmetic."""
    scaled = ratio.numerator * target
    ceiling = (scaled + ratio.denominator - 1) // ratio.denominator
    return min(capacity, ceiling)


def _compensate_sr(dataset_entries, cell_quotas, granularity):
    """Solve exact SR congruences by dataset-local residue DP.

    Each dataset exposes every reachable vector of SR residues for its fixed
    quota.  A deterministic DP combines those vectors modulo ``granularity``.
    State count is bounded by ``granularity ** (num_srs - 1)`` per layer; it is
    independent of UID capacity and never enumerates sample selections.
    """
    srs = sorted({entry["sr"] for entries in dataset_entries.values() for entry in entries})
    if len(srs) <= 1:
        return ()
    _check_sr_dp_budget(dataset_entries, cell_quotas, len(srs), granularity)
    partial = {tuple(0 for _ in srs[:-1]): {}}
    for dataset_key in sorted(dataset_entries):
        options = _dataset_sr_options(dataset_entries[dataset_key], cell_quotas, srs, granularity)
        if not options:
            return None
        next_partial = {}
        for state, selected in partial.items():
            for residues, quotas in options.items():
                merged = tuple((left + right) % granularity for left, right in zip(state, residues))
                candidate = {**selected, **quotas}
                if merged not in next_partial or _quota_distance(candidate, cell_quotas) < _quota_distance(next_partial[merged], cell_quotas):
                    next_partial[merged] = candidate
        partial = next_partial
    result = partial.get(tuple(0 for _ in srs[:-1]))
    if result is None:
        return None
    before = dict(cell_quotas)
    cell_quotas.update(result)
    return tuple(
        {"cell": key, "from": before[key], "to": quota}
        for key, quota in sorted(result.items()) if before[key] != quota
    )


def _estimate_sr_dp_cost(dataset_entries, target, num_srs, granularity, initial=None):
    residue_states = granularity ** max(0, num_srs - 1)
    states = transitions = materialized_entries = 0
    processed_cells = 0
    for entries in dataset_entries.values():
        group_target = (
            sum(initial[entry["key"]] for entry in entries)
            if initial is not None else target
        )
        cell_count = len(entries)
        local_states = (group_target + 1) * residue_states
        local_transitions = local_states * sum(
            min(entry["capacity"], group_target) + 1 for entry in entries
        )
        states += local_states + residue_states
        transitions += local_transitions + residue_states * residue_states
        materialized_entries += local_states * cell_count + local_transitions * cell_count
        merge_edges = residue_states * residue_states
        vector_size = processed_cells + cell_count
        materialized_entries += merge_edges * vector_size * 3 + residue_states * vector_size
        processed_cells += cell_count
    return {
        "states": states,
        "transitions": transitions,
        "materialized_entries": materialized_entries,
    }


def _check_sr_dp_budget(dataset_entries, initial, num_srs, granularity):
    """Preflight logical and materialized costs before residue DP allocations."""
    estimate = _estimate_sr_dp_cost(
        dataset_entries, 0, num_srs, granularity, initial=initial
    )
    states = estimate["states"]
    transitions = estimate["transitions"]
    materialized_entries = estimate["materialized_entries"]
    if (
        states > _SR_DP_STATE_BUDGET
        or transitions > _SR_DP_TRANSITION_BUDGET
        or materialized_entries > _SR_DP_MATERIALIZATION_BUDGET
    ):
        raise _PlannerBudgetExceeded(
            "sr_residue_budget",
            "SR-residue planner budget exceeded "
            f"(estimated states={states}, transitions={transitions}, "
            f"materialized_entries={materialized_entries}); "
            "reduce epoch quota, reduce SR categories, or lower DDP granularity",
        )


def _quota_distance(candidate, initial):
    return sum(abs(quota - initial[key]) for key, quota in candidate.items()), tuple(sorted(candidate.items()))


def _dataset_sr_options(entries, initial, srs, granularity):
    """Return a canonical quota vector for every reachable dataset SR residue."""
    target = sum(initial[entry["key"]] for entry in entries)
    states = {(0, tuple(0 for _ in srs[:-1])): {}}
    for entry in sorted(entries, key=lambda item: item["key"]):
        next_states = {}
        sr_index = srs.index(entry["sr"])
        for (used, residues), selected in states.items():
            upper = min(entry["capacity"], target - used)
            for quota in range(upper + 1):
                updated = list(residues)
                if sr_index < len(updated):
                    updated[sr_index] = (updated[sr_index] + quota) % granularity
                marker = (used + quota, tuple(updated))
                if marker not in next_states:
                    next_states[marker] = {**selected, entry["key"]: quota}
        states = next_states
    return {residues: quotas for (used, residues), quotas in states.items() if used == target}


def _positive_int(value: Any, field: str) -> None:
    if type(value) is bool or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{field} must be a positive int")


def _frozen(value: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """Recursively freeze public plan mappings so callers cannot mutate a plan."""
    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType({key: freeze(value) for key, value in item.items()})
        if isinstance(item, list):
            return tuple(freeze(value) for value in item)
        return item

    return freeze(value)


_normalise_probs = _sampling_schema._normalise_probs
_probs_tuple = _sampling_schema._probs_tuple
_validated_language_probs = _sampling_schema._validated_language_probs


def _project_dataset_total_cap(capacities, ratio, projected, granularity):
    return _sampling_projector._project_dataset_total_cap(
        capacities,
        ratio,
        projected,
        granularity,
        fraction_ceiling=_fraction_ceiling,
    )


_project_transport_actual = _sampling_projector._project_transport_actual
_initial_hamilton_floor_bound = _sampling_projector._initial_hamilton_floor_bound
_project_speaker_margin_actual = _sampling_projector._project_speaker_margin_actual


__all__ = [
    "DatasetSamplingConfig",
    "LanguageSamplingConfig",
    "NoiseSamplingConfig",
    "SamplingConfig",
    "SpeakerSamplingConfig",
    "SpeechLeafMetadata",
    "SpeechPlan",
    "SpeechSamplingConfig",
    "make_speech_plan",
]
