#!/usr/bin/env python3
"""只读审计动态数据集的噪声 source sampling；不调用 ``__getitem__`` 或解码波形。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.utils.config import load_merged_config  # noqa: E402


def _category(dataset, uid: str | None) -> str:
    if uid is None:
        return "unavailable"
    category_id = getattr(dataset, "_noise_category_id_by_uid", {}).get(uid)
    return str(getattr(dataset, "_noise_category_name_by_id", {}).get(category_id, "unavailable"))


def _category_map_complete(dataset) -> bool:
    checker = getattr(dataset, "_ordinary_noise_categories_complete", None)
    if checker is not None:
        return bool(checker())
    ordinary_uids = {uid for entries in dataset.noise_source.values() for uid in entries}
    category_uids = set(getattr(dataset, "_noise_category_id_by_uid", {}))
    return bool(ordinary_uids) and ordinary_uids <= category_uids


def _ordinary_sources(dataset, sr: int, rng, dual: bool) -> list[tuple[str, str]]:
    if hasattr(dataset, "_select_tse_dual_sources") and dual:
        return [(family, uid) for family, uid, _ in dataset._select_tse_dual_sources(sr, rng)]
    if hasattr(dataset, "_select_tse_additive_source"):
        source = dataset._select_tse_additive_source(sr, rng)
        return [(source[0], source[1])] if source else []
    balanced = bool(getattr(dataset, "balanced_noise_sampling_enabled", False))
    first = (
        dataset._select_balanced_additive_noise_uid(sr, rng)
        if balanced
        else dataset._select_sample(sr, dataset.noise_source, rng=rng)
    )
    if first is None:
        return []
    out = [("ordinary", first)]
    if not dual:
        return out
    category_by_uid = getattr(dataset, "_noise_category_id_by_uid", {})
    if _category_map_complete(dataset):
        category_id = category_by_uid[first]
        second = (
            dataset._select_balanced_additive_noise_uid(
                sr, rng, exclude_category_id=category_id, exclude_uid=first
            )
            if balanced
            else dataset._select_natural_different_category_uid(
                sr, rng, exclude_category_id=category_id, exclude_uid=first
            )
        )
    else:
        second = dataset._select_sample(sr, dataset.noise_source, rng=rng)
    if second is not None:
        out.append(("ordinary", second))
    return out


DEFAULT_ALPHA_SWEEP = (0.5, 0.7, 0.85, 1.0)


def _natural_category_candidates(
    dataset,
    sr: int,
    *,
    exclude_category_id: int | None = None,
    exclude_uid: str | None = None,
) -> tuple[str | None, dict[int, tuple[str, ...]]]:
    """镜像 natural 首源 ``_select_sample`` 的 exact → first higher → all。"""
    source = dataset.noise_source
    category_by_uid = dataset._noise_category_id_by_uid
    higher = [fs for fs in sorted(source) if fs > int(sr) and source[fs]]
    tiers = (
        ("exact", [int(sr)] if int(sr) in source else []),
        ("higher", higher[:1]),
        ("all", sorted(source)),
    )
    for tier, sample_rates in tiers:
        grouped: dict[int, set[str]] = {}
        for fs in sample_rates:
            for uid in source[fs]:
                category_id = category_by_uid[uid]
                if category_id == exclude_category_id or uid == exclude_uid:
                    continue
                grouped.setdefault(category_id, set()).add(uid)
        candidates = {
            category_id: tuple(sorted(uids))
            for category_id, uids in sorted(grouped.items())
            if uids
        }
        if candidates:
            return tier, candidates
    return None, {}


def _selector_theory(dataset, sr: int, mode: str) -> dict[str, Any]:
    """报告 runtime helper 在当前 processing SR 的理论候选与 share。"""
    if not _category_map_complete(dataset):
        return {"availability": "unavailable", "tier": None, "categories": {}}
    if mode == "temperature":
        tier, candidates, weights = dataset._temperature_candidates_and_weights(sr)
    else:
        if mode == "category":
            tier, candidates = dataset._noise_category_candidates_for_tier(sr)
        else:
            tier, candidates = _natural_category_candidates(dataset, sr)
        weights = {
            category_id: 1.0 if mode == "category" else float(len(uids))
            for category_id, uids in candidates.items()
        }
    total_weight = float(sum(weights.values()))
    names = getattr(dataset, "_noise_category_name_by_id", {})
    categories = {}
    for category_id, uids in sorted(candidates.items()):
        weight = float(weights[category_id])
        share = weight / total_weight if total_weight else 0.0
        name = str(names.get(category_id, category_id))
        categories[name] = {
            "category_id": category_id,
            "candidate_count": len(uids),
            "weight": weight,
            "share": share,
            "uid_share": share / len(uids),
        }
    return {"availability": "available", "tier": tier, "categories": categories}


def _natural_second_candidates(
    dataset, sr: int, *, exclude_category_id: int
) -> tuple[str | None, dict[int, tuple[str, ...]]]:
    """复用 runtime natural view，返回排除首类后的第二源候选。"""
    specs = getattr(dataset, "_noise_natural_tier_specs", None)
    view = getattr(dataset, "_noise_natural_tier_view", None)
    category_by_uid = dataset._noise_category_id_by_uid
    if specs is None or view is None:
        source = dataset.noise_source
        fallback_specs = (
            ("exact", (int(sr),) if int(sr) in source else ()),
            ("higher", tuple(fs for fs in sorted(source) if fs > int(sr))),
            ("all", tuple(sorted(source))),
        )
        for tier, sample_rates in fallback_specs:
            uids = tuple(
                uid
                for fs in sample_rates
                for uid in source[fs]
                if category_by_uid[uid] != exclude_category_id
            )
            if uids:
                grouped: dict[int, list[str]] = {}
                for uid in uids:
                    grouped.setdefault(category_by_uid[uid], []).append(uid)
                return tier, {
                    category_id: tuple(category_uids)
                    for category_id, category_uids in grouped.items()
                }
        return None, {}
    for tier, sample_rates in specs(sr):
        uids = view(tier, sample_rates, exclude_category_id=exclude_category_id)
        if not uids:
            continue
        grouped: dict[int, list[str]] = {}
        for uid in uids:
            grouped.setdefault(category_by_uid[uid], []).append(uid)
        return tier, {
            category_id: tuple(category_uids)
            for category_id, category_uids in grouped.items()
        }
    return None, {}


def _natural_second_source_theory(dataset, sr: int) -> dict[str, Any]:
    """按 natural 首源可能类别报告 runtime 第二源的条件分布。"""
    if not _category_map_complete(dataset):
        return {"availability": "unavailable", "by_first_category": {}}
    source = dataset.noise_source
    if int(sr) in source and source[int(sr)]:
        first_tier, first_uids = "exact", tuple(source[int(sr)])
    else:
        higher = [fs for fs in sorted(source) if fs > int(sr) and source[fs]]
        if higher:
            first_tier, first_uids = "higher", tuple(source[higher[0]])
        else:
            first_tier = "all"
            first_uids = tuple(uid for entries in source.values() for uid in entries)
    category_by_uid = dataset._noise_category_id_by_uid
    names = getattr(dataset, "_noise_category_name_by_id", {})
    first_categories = sorted({category_by_uid[uid] for uid in first_uids})
    conditioned: dict[str, Any] = {}
    for first_category_id in first_categories:
        tier, candidates = _natural_second_candidates(
            dataset, sr, exclude_category_id=first_category_id
        )
        total = sum(len(uids) for uids in candidates.values())
        categories = {}
        for category_id, uids in sorted(candidates.items()):
            name = str(names.get(category_id, category_id))
            categories[name] = {
                "category_id": category_id,
                "candidate_count": len(uids),
                "category_share": len(uids) / total if total else 0.0,
                "uid_share": 1.0 / total if total else 0.0,
            }
        first_name = str(names.get(first_category_id, first_category_id))
        conditioned[first_name] = {
            "excluded_category_id": first_category_id,
            "tier": tier,
            "categories": categories,
        }
    return {
        "availability": "available",
        "first_tier": first_tier,
        "first_candidate_count": len(first_uids),
        "by_first_category": conditioned,
    }


def _alpha_key(alpha: float) -> str:
    return f"{alpha:g}"


def _alpha_sweep_theory(
    dataset,
    sr: int,
    *,
    selection_budget: int,
    alphas: tuple[float, ...],
) -> dict[str, Any]:
    """仅用 tier-local metadata 估算 temperature alpha 的复用风险。"""
    if not _category_map_complete(dataset):
        return {"availability": "unavailable", "reason": "category metadata incomplete"}
    tier, candidates = dataset._noise_category_candidates_for_tier(sr)
    difficulties = getattr(dataset, "_noise_category_weight_by_id", {})
    missing = sorted(category_id for category_id in candidates if category_id not in difficulties)
    if missing:
        return {
            "availability": "unavailable",
            "reason": f"difficulty unavailable for category_id={missing}",
        }

    normalized_difficulties = {
        category_id: float(difficulties[category_id]) for category_id in candidates
    }
    invalid = sorted(
        category_id
        for category_id, difficulty in normalized_difficulties.items()
        if not math.isfinite(difficulty) or difficulty <= 0.0
    )
    if invalid:
        raise ValueError(
            f"difficulty must be finite and positive for category_id={invalid}"
        )
    names = getattr(dataset, "_noise_category_name_by_id", {})
    difficulty_total = float(sum(normalized_difficulties.values()))
    sweep: dict[str, Any] = {}
    for alpha in alphas:
        weights = {
            category_id: (len(uids) ** alpha) * normalized_difficulties[category_id]
            for category_id, uids in candidates.items()
        }
        total_weight = float(sum(weights.values()))
        categories: dict[str, Any] = {}
        difficulty_exposure: dict[str, dict[str, Any]] = {}
        expected_unique = 0.0
        max_expected_uid_count = 0.0
        rare_count = min((len(uids) for uids in candidates.values()), default=0)
        rare_reuse: dict[str, Any] = {}
        for category_id, uids in sorted(candidates.items()):
            count = len(uids)
            difficulty = normalized_difficulties[category_id]
            category_share = weights[category_id] / total_weight if total_weight else 0.0
            uid_share = category_share / count
            expected_category_count = selection_budget * category_share
            expected_uid_count = selection_budget * uid_share
            if selection_budget <= 0 or uid_share <= 0.0:
                category_expected_unique = 0.0
            elif uid_share >= 1.0:
                category_expected_unique = 1.0
            else:
                category_expected_unique = count * -math.expm1(
                    selection_budget * math.log1p(-uid_share)
                )
            category_expected_repeats = expected_category_count - category_expected_unique
            expected_unique += category_expected_unique
            max_expected_uid_count = max(max_expected_uid_count, expected_uid_count)
            name = str(names.get(category_id, category_id))
            raw_difficulty_prior_share = (
                difficulty / difficulty_total if difficulty_total else 0.0
            )
            item = {
                "category_id": category_id,
                "candidate_count": count,
                "difficulty": difficulty,
                "raw_difficulty_prior_share": raw_difficulty_prior_share,
                "difficulty_share": raw_difficulty_prior_share,
                "weight": weights[category_id],
                "category_share": category_share,
                "uid_share": uid_share,
                "expected_category_count": expected_category_count,
                "expected_uid_count": expected_uid_count,
                "expected_unique": category_expected_unique,
                "expected_repeats": category_expected_repeats,
            }
            categories[name] = item
            difficulty_key = _alpha_key(difficulty)
            exposure = difficulty_exposure.setdefault(
                difficulty_key,
                {
                    "difficulty": difficulty,
                    "category_count": 0,
                    "candidate_uid_count": 0,
                    "selection_exposure_share": 0.0,
                    "expected_selection_count": 0.0,
                    "expected_unique_uid_count": 0.0,
                },
            )
            exposure["category_count"] += 1
            exposure["candidate_uid_count"] += count
            exposure["selection_exposure_share"] += category_share
            exposure["expected_selection_count"] += expected_category_count
            exposure["expected_unique_uid_count"] += category_expected_unique
            if count == rare_count:
                rare_reuse[name] = {
                    "candidate_count": count,
                    "expected_uid_count": expected_uid_count,
                    "expected_unique": category_expected_unique,
                    "expected_repeats": category_expected_repeats,
                }
        for exposure in difficulty_exposure.values():
            exposure["expected_unique_uid_exposure_share"] = (
                exposure["expected_unique_uid_count"] / expected_unique
                if expected_unique
                else 0.0
            )
        sweep[_alpha_key(alpha)] = {
            "categories": categories,
            "difficulty_exposure_by_value": difficulty_exposure,
            "deprecated_fields": {
                "categories.*.difficulty_share": "use raw_difficulty_prior_share; this is not alpha-dependent exposure"
            },
            "expected_unique": expected_unique,
            "expected_repeats": selection_budget - expected_unique,
            "rare_reuse": rare_reuse,
            "max_expected_uid_count": max_expected_uid_count,
        }
    return {
        "availability": "available",
        "tier": tier,
        "selection_budget": selection_budget,
        "formula": "category_weight = tier_uid_count ** alpha * difficulty",
        "alpha_one_includes_difficulty": True,
        "alphas": sweep,
    }


def audit_dataset(
    dataset,
    *,
    draws: int,
    seed: int,
    sample_rate: int | None = None,
    alphas: tuple[float, ...] = DEFAULT_ALPHA_SWEEP,
) -> dict[str, Any]:
    """按 runtime 分支和 selector 做固定 seed 抽样，不读取任何音频。"""
    if draws <= 0:
        raise ValueError("draws must be positive")
    normalized_alphas = tuple(dict.fromkeys(float(alpha) for alpha in alphas))
    if not normalized_alphas or any(
        not math.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0
        for alpha in normalized_alphas
    ):
        raise ValueError("alphas must contain finite values in (0, 1]")
    if sample_rate is not None:
        effective_sr = sample_rate
    else:
        effective_sr = getattr(dataset, "target_sr", 0)
        if not effective_sr:
            if not dataset.noise_source:
                raise ValueError("noise_source must not be empty")
            effective_sr = min(dataset.noise_source)
    if isinstance(effective_sr, bool) or not isinstance(effective_sr, (int, np.integer)):
        raise ValueError("sample_rate must be a positive integer")
    sr = int(effective_sr)
    if sr <= 0:
        raise ValueError("sample_rate must be a positive integer")
    rng = np.random.default_rng(seed)
    cfg = dataset.sim_cfg
    category_available = _category_map_complete(dataset)
    if category_available:
        _alpha_sweep_theory(
            dataset,
            sr,
            selection_budget=0,
            alphas=normalized_alphas,
        )
    counts = {
        key: Counter()
        for key in ("sr_tier", "source_count", "first_category", "second_category", "family")
    }
    uid_histogram: Counter[str] = Counter()
    same_category_violation = 0
    for _ in range(draws):
        mix_two = rng.random() < float(getattr(cfg, "noise_mix_prob", 0.0))
        want_wind = rng.random() < float(getattr(cfg, "prob_wind_noise_sidechain", 0.0))
        sources: list[tuple[str, str]] = []
        if want_wind and getattr(dataset, "wind_noises", {}):
            wind_uid = dataset._select_sample(sr, dataset.wind_noises, rng=rng)
            if wind_uid is not None:
                sources.append(("wind", wind_uid))
            if mix_two:
                sources.extend(_ordinary_sources(dataset, sr, rng, False))
        else:
            sources = _ordinary_sources(dataset, sr, rng, mix_two)
        counts["source_count"]["dual" if len(sources) > 1 else "single"] += 1
        for family, uid in sources:
            counts["family"][family] += 1
            uid_histogram[f"{family}:{uid}"] += 1
            pool = (
                dataset.music_source
                if family == "music"
                else dataset.wind_noises
                if family == "wind"
                else dataset.noise_source
            )
            native_sr = next((fs for fs, entries in pool.items() if uid in entries), None)
            tier = (
                "exact"
                if native_sr == sr
                else "higher"
                if native_sr is not None and native_sr > sr
                else "all"
            )
            counts["sr_tier"][tier] += 1
        categories = [
            _category(dataset, uid)
            if family == "ordinary" and category_available
            else "unavailable"
            if family == "ordinary"
            else family
            for family, uid in sources
        ]
        if categories:
            counts["first_category"][categories[0]] += 1
        if len(categories) > 1:
            counts["second_category"][categories[1]] += 1
            if category_available and categories[0] == categories[1] not in ("music", "wind"):
                same_category_violation += 1
    noise_cfg = getattr(dataset, "_noise_sampling_config", None)
    mode = getattr(noise_cfg, "balance_mode", "natural") if noise_cfg is not None else "natural"
    unique = len(uid_histogram)
    selections = sum(uid_histogram.values())
    ordinary_selections = int(counts["family"]["ordinary"])
    return {
        "draws": draws,
        "seed": seed,
        "processing_sr": sr,
        "effective_mode": mode,
        "alpha": getattr(noise_cfg, "alpha", None),
        "category_available": category_available,
        "selector_theory": _selector_theory(dataset, sr, mode),
        "natural_second_source_theory": _natural_second_source_theory(dataset, sr),
        "alpha_sweep": _alpha_sweep_theory(
            dataset,
            sr,
            selection_budget=ordinary_selections,
            alphas=normalized_alphas,
        ),
        **{key: dict(sorted(value.items())) for key, value in counts.items()},
        "same_category_violation": same_category_violation,
        "uid_coverage": {
            "unique": unique,
            "selections": selections,
            "repeats": selections - unique,
            "max_repeat": max(uid_histogram.values(), default=0),
            "mean_repeat": selections / unique if unique else 0.0,
            "selection_histogram": dict(sorted(uid_histogram.items())),
        },
    }


def _build_dataset(config: dict, split: str):
    mode = str(config.get("data", {}).get("mode", "")).lower()
    if "tse" in mode:
        from tasks.tse.data import build_dataset
    elif "aec" in mode:
        from tasks.aec.data import build_dataset
    else:
        from tasks.se.data import build_dataset
    return build_dataset(config["data"], "train" if split == "train" else "val", meta=config.get("meta"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="merged YAML config（支持 base_config）")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=None, help="可选 processing SR")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHA_SWEEP,
        help="纯元数据 sweep 的 alpha 列表（默认：0.5 0.7 0.85 1.0）",
    )
    parser.add_argument(
        "--json-out",
        "--json",
        dest="json_out",
        type=Path,
        default=None,
        help="可选 JSON 摘要输出路径（--json 为兼容别名）",
    )
    args = parser.parse_args()
    config_path = str(Path(args.config).resolve())
    report = audit_dataset(
        _build_dataset(load_merged_config(args.config), args.split),
        draws=args.draws,
        seed=args.seed,
        sample_rate=args.sample_rate,
        alphas=tuple(args.alphas),
    )
    report["source"] = {"config_path": config_path, "split": args.split}
    text = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
