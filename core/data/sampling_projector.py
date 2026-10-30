"""Pure quota upper-bound projection math for speech sampling."""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction


def _exact_weight_distribution(weights):
    """把 exact/float 权重按其 planner 归一化路径转为保守分析比例。"""
    positive = {key: value for key, value in weights.items() if value > 0}
    if not positive:
        return {}
    if all(isinstance(value, Fraction) for value in positive.values()):
        total = sum(positive.values(), Fraction())
        return {key: value / total for key, value in positive.items()}
    scale = max(positive.values())
    scaled = {key: value / scale for key, value in positive.items()}
    denominator = math.fsum(scaled.values())
    return {
        key: Fraction.from_float(scaled[key] / denominator)
        for key in positive
    }


def _fraction_ceiling(value):
    return (value.numerator + value.denominator - 1) // value.denominator


def _floor_sum_unsigned(count, modulus, slope, offset):
    """计算 ``Σ_{0<=i<count} floor((slope*i+offset)/modulus)``。"""
    total = 0
    while True:
        if slope >= modulus:
            total += (count - 1) * count * (slope // modulus) // 2
            slope %= modulus
        if offset >= modulus:
            total += count * (offset // modulus)
            offset %= modulus
        highest = slope * count + offset
        if highest < modulus:
            return total
        count = highest // modulus
        offset = highest % modulus
        modulus, slope = slope, modulus


def _floor_sum_prefix(count, modulus, slope, offset):
    """支持 signed slope/offset 的 linear floor-sum 前缀。"""
    slope_quotient, slope = divmod(slope, modulus)
    offset_quotient, offset = divmod(offset, modulus)
    return (
        slope_quotient * count * (count - 1) // 2
        + offset_quotient * count
        + _floor_sum_unsigned(count, modulus, slope, offset)
    )


def _floor_sum_range(left, right, modulus, slope, offset):
    if left > right:
        return 0
    return _floor_sum_prefix(
        right + 1, modulus, slope, offset
    ) - _floor_sum_prefix(left, modulus, slope, offset)


def _max_dataset_segment_candidate(
    saturated_sum,
    uncapped_count,
    ratio_numerator,
    ratio_denominator,
    granularity,
    lower_index,
    upper_index,
):
    """精确求固定 physical-saturation 分段内的最大 aligned index。"""
    if lower_index > upper_index:
        return None
    if uncapped_count == 0:
        candidate = min(upper_index, saturated_sum // granularity)
        return candidate if candidate >= lower_index else None
    if uncapped_count * ratio_numerator >= ratio_denominator:
        return upper_index

    scaled_slope = ratio_numerator * granularity
    deficit_slope = granularity * (
        ratio_denominator - uncapped_count * ratio_numerator
    )
    residue_limit = (
        saturated_sum * ratio_denominator
        + uncapped_count * (ratio_denominator - 1)
    )
    always_feasible = saturated_sum * ratio_denominator // deficit_slope
    last_possible = residue_limit // deficit_slope

    def feasible_count(left, right):
        if left > right:
            return 0
        count = 0
        certain_right = min(right, always_feasible)
        if left <= certain_right:
            count += certain_right - left + 1
        middle_left = max(left, always_feasible + 1)
        middle_right = min(right, last_possible)
        if middle_left <= middle_right:
            count += _floor_sum_range(
                middle_left,
                middle_right,
                ratio_denominator,
                scaled_slope,
                ratio_denominator - 1,
            ) - _floor_sum_range(
                middle_left,
                middle_right,
                uncapped_count,
                granularity,
                -saturated_sum - 1,
            )
        return count

    if feasible_count(lower_index, upper_index) == 0:
        return None
    low = lower_index
    high = upper_index + 1
    while low + 1 < high:
        middle = (low + high) // 2
        if feasible_count(middle, upper_index) > 0:
            low = middle
        else:
            high = middle
    return low


def _project_dataset_total_cap(
    capacities, ratio, projected, granularity, *, fraction_ceiling=_fraction_ceiling
):
    """返回 ``A <= Σ min(C_d, ceil(rA))`` 的最大 aligned 解。"""
    if projected <= 0:
        return 0
    capacities = tuple(sorted(capacities))
    top_index = projected // granularity
    numerator = ratio.numerator
    denominator = ratio.denominator
    if numerator == denominator:
        return granularity * min(
            top_index, sum(capacities) // granularity
        )

    scaled_slope = numerator * granularity
    max_quota = fraction_ceiling(ratio * (top_index * granularity))
    regimes = []
    saturated_count = 0
    saturated_sum = 0
    while (
        saturated_count < len(capacities)
        and capacities[saturated_count] <= 0
    ):
        saturated_sum += capacities[saturated_count]
        saturated_count += 1
    quota_lower = 0
    while quota_lower <= max_quota:
        next_capacity = (
            capacities[saturated_count]
            if saturated_count < len(capacities)
            else max_quota + 1
        )
        quota_upper = min(max_quota, next_capacity - 1)
        if quota_lower <= quota_upper:
            regimes.append((
                quota_lower,
                quota_upper,
                saturated_sum,
                len(capacities) - saturated_count,
            ))
        if next_capacity > max_quota:
            break
        while (
            saturated_count < len(capacities)
            and capacities[saturated_count] == next_capacity
        ):
            saturated_sum += capacities[saturated_count]
            saturated_count += 1
        quota_lower = next_capacity

    for quota_lower, quota_upper, saturated_sum, uncapped_count in reversed(regimes):
        lower_index = max(
            0,
            denominator * (quota_lower - 1) // scaled_slope + 1,
        )
        upper_index = min(
            top_index,
            denominator * quota_upper // scaled_slope,
        )
        candidate = _max_dataset_segment_candidate(
            saturated_sum,
            uncapped_count,
            numerator,
            denominator,
            granularity,
            lower_index,
            upper_index,
        )
        if candidate is not None:
            return candidate * granularity
    return 0


def _project_transport_actual(context, probs, ratio, projected, granularity):
    """用可证明的 dataset/transport 必要条件收紧候选上界。"""
    if not context["dataset_enabled"] or projected <= 0:
        return projected
    projected = _project_dataset_total_cap(
        tuple(context["dataset_capacities"].values()),
        ratio,
        projected,
        granularity,
    )
    if probs is None or projected <= 0:
        return projected
    language_weights = {key: value for key, value in probs.items() if value > 0}
    dataset_weights = context["dataset_weights"]
    language_share = _exact_weight_distribution(language_weights)
    dataset_share = _exact_weight_distribution(dataset_weights)
    dataset_unbounded = projected <= 2**52 and all(
        context["dataset_capacities"].get(key, 0)
        >= _fraction_ceiling(dataset_share.get(key, Fraction()) * projected) + 2
        and ratio > dataset_share.get(key, Fraction())
        for key in dataset_weights
    )

    languages = tuple(sorted(language_share))
    upper = projected
    pair_capacities = context["pair_capacities"]
    all_languages = set(languages)
    for mask in range(1, 1 << len(languages)):
        subset = {
            language
            for index, language in enumerate(languages)
            if mask & (1 << index)
        }
        neighbours = {
            dataset
            for language, dataset in pair_capacities
            if language in subset and pair_capacities[(language, dataset)] > 0
        }
        language_total = sum(
            (language_share[language] for language in subset), Fraction()
        )
        lower_bounds = [(
            Fraction(1),
            Fraction(sum(
                context["language_capacities"].get(language, 0)
                for language in all_languages - subset
            )),
            1,
        )]
        if all(
            context["language_capacities"].get(language, 0)
            >= _fraction_ceiling(language_share[language] * projected)
            for language in subset
        ):
            lower_bounds.append((language_total, Fraction(len(subset)), 1))
        upper_bounds = [
            (
                Fraction(0),
                Fraction(sum(
                    context["dataset_capacities"][dataset]
                    for dataset in neighbours
                )),
                1,
            ),
            (len(neighbours) * ratio, Fraction(len(neighbours)), 1),
        ]
        if dataset_unbounded:
            upper_bounds.append((
                sum(
                    (dataset_share.get(dataset, Fraction()) for dataset in neighbours),
                    Fraction(),
                ),
                Fraction(2 * len(neighbours)),
                1,
            ))

        slope = Fraction()
        constant = Fraction()
        for dataset in sorted(context["dataset_capacities"]):
            edge_capacity = sum(
                pair_capacities.get((language, dataset), 0)
                for language in subset
            )
            quota_bounds = [
                (Fraction(0), Fraction(context["dataset_capacities"][dataset])),
                (ratio, Fraction(1)),
            ]
            if dataset_unbounded:
                quota_bounds.append(
                    (dataset_share.get(dataset, Fraction()), Fraction(2))
                )
            quota_slope, quota_constant = min(
                quota_bounds,
                key=lambda bound: bound[0] * projected + bound[1],
            )
            if quota_slope * projected + quota_constant <= edge_capacity:
                slope += quota_slope
                constant += quota_constant
            else:
                constant += edge_capacity
        upper_bounds.append((slope, constant, 1))

        for lower_slope, lower_constant, lower_threshold in lower_bounds:
            for upper_slope, upper_constant, upper_threshold in upper_bounds:
                threshold = max(lower_threshold, upper_threshold)
                deficit = lower_slope - upper_slope
                bound_constant = lower_constant + upper_constant
                if min(
                    deficit * threshold - bound_constant,
                    deficit * upper - bound_constant,
                ) > 0:
                    upper = min(upper, threshold - 1)
                    continue
                if deficit <= 0:
                    continue
                cut_upper = bound_constant / deficit
                certified_upper = max(
                    cut_upper.numerator // cut_upper.denominator,
                    threshold - 1,
                )
                upper = min(upper, certified_upper)
    upper -= upper % granularity
    if 0 < upper < projected:
        return _project_transport_actual(context, probs, ratio, upper, granularity)
    return upper


def _initial_hamilton_floor_bound(weights, key, capacity, projected):
    """求首轮 Hamilton floor 不超过 ``capacity`` 的最大 candidate。"""
    if all(isinstance(value, Fraction) for value in weights.values()):
        denominator = sum(weights.values(), Fraction())
        share = weights[key] / denominator
        return min(projected, _fraction_ceiling(Fraction(capacity + 1) / share) - 1)

    scale = max(weights.values())
    scaled = {item: value / scale for item, value in weights.items()}
    denominator = math.fsum(scaled.values())
    share = scaled[key] / denominator
    low, high = 0, projected + 1
    while low + 1 < high:
        middle = (low + high) // 2
        try:
            within = math.floor(middle * share) <= capacity
        except OverflowError:
            within = False
        if within:
            low = middle
        else:
            high = middle
    return low


def _project_speaker_margin_actual(
    leaves, context, probs, ratio, speaker_cap, projected, granularity
):
    """用 speaker-adjusted language/dataset 边际容量收紧候选上界。"""
    if projected <= 0 or (probs is None and not context["dataset_enabled"]):
        return projected

    axis_speaker_capacities = {
        "language": defaultdict(lambda: defaultdict(int)),
        "dataset": defaultdict(lambda: defaultdict(int)),
    }
    for entry in leaves:
        identity = entry["speaker"]
        axis_speaker_capacities["language"][entry["language"]][identity] += entry["capacity"]
        axis_speaker_capacities["dataset"][entry["dataset"]][identity] += entry["capacity"]

    def adjusted(axis, key):
        return sum(
            min(capacity, speaker_cap)
            for capacity in axis_speaker_capacities[axis][key].values()
        )

    upper = projected
    if probs is not None:
        weights = {key: value for key, value in probs.items() if value > 0}
        for key in sorted(weights):
            physical = context["language_capacities"][key]
            speaker_capacity = adjusted("language", key)
            if speaker_capacity < physical:
                upper = min(
                    upper,
                    _initial_hamilton_floor_bound(
                        weights, key, speaker_capacity, upper
                    ),
                )

    if context["dataset_enabled"]:
        weights = context["dataset_weights"]
        for key in sorted(weights):
            physical = context["dataset_capacities"][key]
            speaker_capacity = adjusted("dataset", key)
            if speaker_capacity < physical:
                ratio_bound = speaker_capacity * ratio.denominator // ratio.numerator
                weight_bound = _initial_hamilton_floor_bound(
                    weights, key, speaker_capacity, upper
                )
                upper = min(upper, max(ratio_bound, weight_bound))

    return upper - upper % granularity


__all__ = [
    "_fraction_ceiling",
    "_project_dataset_total_cap",
    "_project_speaker_margin_actual",
    "_project_transport_actual",
]
