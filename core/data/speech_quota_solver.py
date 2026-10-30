"""联合 speech quota solver：在 primitive leaf 上求解独立 controller。

memoized residual-state DFS/DP（Task 3 实现，预算内穷尽所有未剪枝可行
状态并返回稳定最优解）：

- 只消费 primitive leaf（``SolverLeaf``）+ 可选 exact margins（``SolverMargins``）；
  本模块**禁止** import sampling 模块（防循环依赖），仅依赖标准库。
- 输入 canonicalization：按 ``SolverLeafKey`` 升序稳定排序并校验全部输入，
  非法输入（重复 key、负 capacity、margin sum 不等 actual、margin 未完整
  覆盖轴内全部 leaf key、unknown margin key、非正 ``sr_modulus`` / speaker
  cap / budget、非有限或负 reference 等）一律 ValueError fail-fast。
- 搜索：以 residual state 为 memo key 的迭代 DFS（显式栈，无递归深度
  风险）；只施加可由硬约束直接推出的**必要条件**剪枝（suffix 容量界、
  exact-margin 下界、speaker 松弛总量、SR residue 可对齐性），因此预算内
  等价于穷举全部可行 assignment，memo 保证每个 state 只展开一次。
- speaker-cap residual max-flow oracle（仅仍有 binding speaker 时）：总容量
  不超过 cap 的 identity 不进入 state；按 last position 建 O(leaves + tracked
  speakers) expiry schedule，child state 增量移除已从 suffix 消失的 identity
  usage。通过 cheap 剪枝且 value-memo miss 的 child state 补一次
  必要条件最大流判定（language→leaf→speaker→dataset 分层网络），flow
  不可行 ⟹ 真不可行，可安全剪枝；flow 可行不代表真可行（不表达 SR 模
  约束，dataset 层经 speaker 节点混流为松弛）。判定结果按 state memo 化，
  且**禁止**把 max-flow 的流转换为 leaf quotas。
- 预算计数规则（跨候选累计，``cumulative_stats`` 作为上一候选累计值进入）：
  - ``states``        每个首次展开的 memo state +1（含 terminal state；
                     memo hit 不复查、不计数）；
  - ``transitions``   每次为当前 leaf 枚举一个 quota 值 +1——**含**必要条件
                     剪枝判死后跳过的决策尝试（先计数后判死，与字面口径一致）；
  - ``materialized``  每个 terminal state 首次展开 +len(leaves)。
  任一计数超过对应 budget 上限立即抛 ``SolverBudgetExceeded``，绝不返回
  ``None`` 让上层 shrink。
- ``SolverAttempt.result is None`` 只表示当前候选数学不可行。
- objective：``(sum(abs(Fraction(q) - leaf.reference)), tuple(quotas))``，
  stable leaf order（``SolverLeafKey`` 升序，``None`` speaker 按 ``("", "")``
  参与比较、恒位于具名 identity 之前）下字典序最小者胜。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

CanonicalSpeaker = tuple[str, str]
SolverLeafKey = tuple[str, str, CanonicalSpeaker | None, int]


def _leaf_key_sort_key(key):
    """stable leaf order 的排序键：``SolverLeafKey`` 升序，``None`` speaker
    按 ``("", "")`` 参与比较（恒位于所有具名 canonical identity 之前）。

    ``key`` 直接作为 tuple 比较在 ``None`` 与 ``CanonicalSpeaker`` 混存时
    会抛 TypeError，因此拆成全字符串的 5 元组。这是全仓唯一的排序契约
    实现；sampling / dynamic_dataset_base 的 entry/leaf 排序均委托到此。
    """
    language, dataset_id, speaker, sr = key
    speaker_sort = speaker if speaker is not None else ("", "")
    return (language, dataset_id, speaker_sort[0], speaker_sort[1], sr)


def _leaf_sort_key(leaf: SolverLeaf):
    """``_leaf_key_sort_key`` 的 ``SolverLeaf`` 适配。"""
    return _leaf_key_sort_key(leaf.key)


@dataclass(frozen=True, order=True)
class SolverLeaf:
    key: SolverLeafKey
    capacity: int
    reference: Fraction


@dataclass(frozen=True)
class SolverMargins:
    language: tuple[tuple[str, int], ...] | None
    dataset: tuple[tuple[str, int], ...] | None


@dataclass(frozen=True)
class SolverBudget:
    max_states: int = 250_000
    max_transitions: int = 2_000_000
    max_materialized: int = 8_000_000


@dataclass(frozen=True)
class SolverStats:
    states: int
    transitions: int
    materialized: int


@dataclass(frozen=True)
class SolverResult:
    leaf_quotas: tuple[tuple[SolverLeafKey, int], ...]
    objective: tuple[Fraction, tuple[int, ...]]


@dataclass(frozen=True)
class SolverAttempt:
    result: SolverResult | None
    stats: SolverStats


class SolverBudgetExceeded(RuntimeError):
    pass


def solve_leaf_quotas(
    leaves: Sequence[SolverLeaf],
    *,
    actual: int,
    margins: SolverMargins,
    speaker_cap: int | None,
    sr_modulus: int,
    budget: SolverBudget,
    cumulative_stats: SolverStats | None = None,
) -> SolverAttempt:
    """在稳定排序后的 leaves 上求解满足全部硬约束的 objective 最优 quota。

    hard constraints：``sum(leaf_quota) == actual``、leaf capacity、
    language/dataset exact margins（启用轴逐项精确命中且完整覆盖轴内全部
    leaf key）、speaker absolute upper cap（``speaker_cap`` 非 None 时每个
    canonical identity 聚合 <= cap）、每个 ``batching_sr`` 聚合
    ``% sr_modulus == 0``。
    """
    ordered = _validate_and_canonicalize(
        leaves, actual, margins, speaker_cap, sr_modulus, budget, cumulative_stats
    )
    counts = {
        "states": cumulative_stats.states if cumulative_stats is not None else 0,
        "transitions": cumulative_stats.transitions if cumulative_stats is not None else 0,
        "materialized": cumulative_stats.materialized if cumulative_stats is not None else 0,
    }
    best = _search(ordered, actual, margins, speaker_cap, sr_modulus, budget, counts)
    result = None
    if best is not None:
        objective_value, quota_vector = best
        result = SolverResult(
            leaf_quotas=tuple(
                (leaf.key, quota)
                for leaf, quota in zip(ordered, quota_vector)
            ),
            objective=objective_value,
        )
    return SolverAttempt(result=result, stats=SolverStats(**counts))


def _validate_and_canonicalize(
    leaves, actual, margins, speaker_cap, sr_modulus, budget, cumulative_stats
):
    """输入校验并返回按 ``SolverLeafKey`` 升序的稳定 leaf 序列。"""
    if type(actual) is not int or actual < 0:
        raise ValueError("actual must be a non-negative int")
    if not isinstance(margins, SolverMargins):
        raise ValueError("margins must be a SolverMargins")
    if speaker_cap is not None and (
        type(speaker_cap) is not int or speaker_cap <= 0
    ):
        raise ValueError("speaker_cap must be None or a positive int")
    if type(sr_modulus) is not int or sr_modulus <= 0:
        raise ValueError("sr_modulus must be a positive int")
    if not isinstance(budget, SolverBudget):
        raise ValueError("budget must be a SolverBudget")
    for field in ("max_states", "max_transitions", "max_materialized"):
        value = getattr(budget, field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} must be a positive int")
    if cumulative_stats is not None:
        if not isinstance(cumulative_stats, SolverStats):
            raise ValueError("cumulative_stats must be a SolverStats")
        for field in ("states", "transitions", "materialized"):
            value = getattr(cumulative_stats, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"cumulative_stats.{field} must be a non-negative int")

    _validate_margins_shape(margins.language, "language")
    _validate_margins_shape(margins.dataset, "dataset")

    ordered = []
    seen = set()
    for index, leaf in enumerate(leaves):
        if not isinstance(leaf, SolverLeaf):
            raise ValueError(f"leaf {index} must be a SolverLeaf")
        _validate_leaf_key(leaf.key, index)
        if type(leaf.capacity) is not int or leaf.capacity < 0:
            raise ValueError(f"leaf {index}: capacity must be a non-negative int")
        if not isinstance(leaf.reference, Fraction) or leaf.reference < 0:
            raise ValueError(
                f"leaf {index}: reference must be a non-negative Fraction"
            )
        if leaf.key in seen:
            raise ValueError(f"duplicate leaf key {leaf.key!r}")
        seen.add(leaf.key)
        ordered.append(leaf)
    ordered.sort(key=_leaf_sort_key)

    _validate_margins_keys_and_sum(
        margins.language, actual, "language",
        {leaf.key[0] for leaf in ordered},
    )
    _validate_margins_keys_and_sum(
        margins.dataset, actual, "dataset",
        {leaf.key[1] for leaf in ordered},
    )
    return ordered


def _validate_leaf_key(key, index):
    """校验 key 结构：``(language, dataset_id, canonical_speaker_or_none,
    batching_sr)``；canonical speaker 必须是 ``(dataset_id, speaker_id)``
    且与 leaf 的 dataset_id 一致（跨 dataset 同裸 speaker 因此是不同 identity）。"""
    if not isinstance(key, tuple) or len(key) != 4:
        raise ValueError(
            f"leaf {index}: key must be "
            "(language, dataset_id, canonical_speaker_or_none, batching_sr)"
        )
    language, dataset_id, speaker, sr = key
    if not isinstance(language, str) or not language:
        raise ValueError(f"leaf {index}: key[0] language must be a non-empty string")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError(
            f"leaf {index}: key[1] dataset_id must be a non-empty string"
        )
    if speaker is not None:
        if not (
            isinstance(speaker, tuple)
            and len(speaker) == 2
            and all(isinstance(part, str) and part for part in speaker)
        ):
            raise ValueError(
                f"leaf {index}: key[2] must be a canonical "
                "(dataset_id, speaker_id) tuple or None"
            )
        if speaker[0] != dataset_id:
            raise ValueError(
                f"leaf {index}: canonical speaker dataset {speaker[0]!r} "
                f"must match key[1] dataset_id {dataset_id!r}"
            )
    if type(sr) is not int or sr <= 0:
        raise ValueError(f"leaf {index}: key[3] batching_sr must be a positive int")


def _validate_margins_shape(margins, axis):
    """``None`` = 轴未启用；启用时必须是非空 ``(key, quota)`` 元组。"""
    if margins is None:
        return
    if not isinstance(margins, tuple) or not margins:
        raise ValueError(
            f"margins.{axis} must be None or a non-empty tuple of (key, quota) pairs"
        )
    for index, pair in enumerate(margins):
        field = f"margins.{axis}[{index}]"
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"{field} must be a (key, quota) tuple pair")
        key, quota = pair
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field}[0] must be a non-empty string")
        if type(quota) is not int or quota < 0:
            raise ValueError(f"{field}[1] quota must be a non-negative int")


def _validate_margins_keys_and_sum(margins, actual, axis, known):
    """margin key 必须命中 leaf 空间、无重复，并**完整覆盖** leaves 中该轴
    出现的全部 key（缺失 key 属于 contract error）；所有 entry 的 quota 之和
    必须等于 actual。"""
    if margins is None:
        return
    total = 0
    seen = set()
    margin_keys = set()
    for key, quota in margins:
        if key not in known:
            raise ValueError(
                f"margins.{axis} contains unknown key {key!r} with no eligible leaf"
            )
        if key in seen:
            raise ValueError(f"margins.{axis} contains duplicate key {key!r}")
        seen.add(key)
        margin_keys.add(key)
        total += quota
    missing = sorted(known - margin_keys)
    if missing:
        raise ValueError(
            f"margins.{axis} must cover all leaf {axis} keys; missing: {missing}"
        )
    if total != actual:
        raise ValueError(
            f"margins.{axis} quotas must sum to actual ({total} != {actual})"
        )


def _quota_order(lower, upper, reference):
    """按 ``(abs(Fraction(q) - reference), q)`` 升序惰性枚举闭区间。

    枚举顺序只提升尽快找到 incumbent，不改变完整性（预算内所有可达
    transition 都会被尝试）；O(1) 内存，避免大 capacity 下为每个 DFS
    frame 预建完整 quota 列表。
    """
    floor_q = reference.numerator // reference.denominator
    frac = reference - floor_q
    center = floor_q if frac <= Fraction(1, 2) else floor_q + 1
    center = min(max(center, lower), upper)
    yield center
    # frac == 0 时同距 tie 由较小 q 优先（lo 先）；0 < frac <= 1/2 时 hi
    # 严格更近；frac > 1/2 时 center 已上移、lo 严格更近。
    lo_first = frac == 0 or frac > Fraction(1, 2)
    for k in range(1, max(center - lower, upper - center) + 1):
        lo, hi = center - k, center + k
        first, second = (lo, hi) if lo_first else (hi, lo)
        if lower <= first <= upper:
            yield first
        if lower <= second <= upper:
            yield second


# DFS frame 的 list 槽位（显式栈迭代实现，禁止无界递归：leaf 数可超 1000）
_STATE, _POS, _REM, _REM_LANG, _REM_DS, _SPK_USED, _SLACK, _RESIDUES, \
    _GEN, _PENDING_Q, _BEST = range(11)


def _search(ordered, actual, margins, speaker_cap, sr_modulus, budget, counts):
    """memoized residual-state DFS/DP：预算内穷尽所有未剪枝可行状态并返回
    全局最优（稳定 tie-break）。

    memo key（冻结契约）::

        (position, remaining_total,
         language_remaining | None,   # sorted margin-key order
         dataset_remaining | None,    # sorted margin-key order
         speaker_used | None,         # 当前 suffix-active binding identity 的
                                      # canonical (identity, used) tuple
         residues_except_last_sr)     # sorted SR 中除最后一个外，各自 quota
                                      # 总和 % sr_modulus

    最后 SR 不存 state：``actual % sr_modulus == 0`` 且其余 residue 全零时，
    最后 SR 总量由总量守恒自动满足模约束（root 预检直接排除
    ``actual % sr_modulus != 0`` 的必然不可行候选）。

    完整性论证：DFS 对每个首次展开的 state 枚举当前 leaf 在必要闭区间
    ``lower..upper`` 内的全部 quota；lower 只由后缀总容量和启用轴的后缀
    exact capacity 推出，upper 由 leaf capacity、remaining_total、启用轴
    exact remaining、binding speaker 剩余 cap 推出。其余剪枝同样都是硬约束
    的**必要条件**——suffix 容量界、speaker 剩余 cap 总量下界、非最后 SR
    residue 在后缀容量内的可对齐性；
    必要条件不会删除任何可行 assignment，memo 又保证相同 residual state
    的最优 suffix 只计算一次，因此未耗尽预算时等价于穷举全部可行
    assignment 并返回 objective 全局最优。任何 budget counter 在增量前
    检查上限（见 ``_bump``）：超限立即抛 ``SolverBudgetExceeded``，不返回
    incumbent、不返回 ``None``，不允许调用方 shrink。
    """
    n = len(ordered)
    capacities = [leaf.capacity for leaf in ordered]
    references = [leaf.reference for leaf in ordered]
    languages = [leaf.key[0] for leaf in ordered]
    dataset_ids = [leaf.key[1] for leaf in ordered]
    speakers = [leaf.key[2] for leaf in ordered]
    srs = [leaf.key[3] for leaf in ordered]

    # exact-margin 轴索引（sorted margin-key order；校验已保证完整覆盖）
    lang_keys = (
        sorted(key for key, _quota in margins.language)
        if margins.language is not None
        else None
    )
    lang_index = (
        {key: i for i, key in enumerate(lang_keys)}
        if lang_keys is not None
        else None
    )
    ds_keys = (
        sorted(key for key, _quota in margins.dataset)
        if margins.dataset is not None
        else None
    )
    ds_index = (
        {key: i for i, key in enumerate(ds_keys)} if ds_keys is not None else None
    )
    # 仅跟踪总容量严格超过 cap、因而真正可能 binding 的 speaker。其余
    # canonical identity 的 cap 永远无法触发，不进入 memo state 或 max-flow。
    # expiry schedule 中每个 tracked identity 仅出现一次；child state 在越过
    # last_pos 后增量删除 usage，避免按 position 预物化全部 active identity。
    tracked_speakers = None
    speaker_expiries = None
    if speaker_cap is not None:
        speaker_capacities = {}
        speaker_last_pos = {}
        for pos, (identity, capacity) in enumerate(zip(speakers, capacities)):
            if identity is None:
                continue
            speaker_capacities[identity] = (
                speaker_capacities.get(identity, 0) + capacity
            )
            speaker_last_pos[identity] = pos
        tracked_speakers = tuple(sorted(
            identity
            for identity, capacity in speaker_capacities.items()
            if capacity > speaker_cap
        ))
        if tracked_speakers:
            expiries = [[] for _ in range(n + 1)]
            for identity in tracked_speakers:
                expiries[speaker_last_pos[identity] + 1].append(identity)
            speaker_expiries = tuple(tuple(identities) for identities in expiries)
        else:
            tracked_speakers = None
    # SR residue 只跟踪 sorted SR 中除最后一个之外的全部 SR
    sr_values = sorted(set(srs))
    tracked_srs = sr_values[:-1]
    sr_index = {sr: i for i, sr in enumerate(tracked_srs)}

    # suffix 容量预计算：``suffix_*[i]`` 表示区间 ``[i:]``（**含**位置 i）
    # 的容量——root 预检读 ``suffix_*[0]``（全部 leaf），transition 剪枝读
    # ``suffix_*[child_pos]``（当前 leaf 之后的后缀）。未启用的 exact 轴
    # 不分配对应表（保持 None）。
    suffix_total = [0] * (n + 1)
    nlang = len(lang_keys) if lang_keys is not None else 0
    nds = len(ds_keys) if ds_keys is not None else 0
    suffix_lang = (
        [[0] * nlang for _ in range(n + 1)] if lang_keys is not None else None
    )
    suffix_ds = (
        [[0] * nds for _ in range(n + 1)] if ds_keys is not None else None
    )
    suffix_sr = [[0] * len(tracked_srs) for _ in range(n + 1)]
    suffix_untracked_spk = [0] * (n + 1)
    tracked_speaker_set = set(tracked_speakers or ())
    for i in range(n - 1, -1, -1):
        suffix_total[i] = suffix_total[i + 1] + capacities[i]
        if suffix_lang is not None:
            suffix_lang[i] = suffix_lang[i + 1].copy()
            suffix_lang[i][lang_index[languages[i]]] += capacities[i]
        if suffix_ds is not None:
            suffix_ds[i] = suffix_ds[i + 1].copy()
            suffix_ds[i][ds_index[dataset_ids[i]]] += capacities[i]
        if tracked_srs:
            suffix_sr[i] = suffix_sr[i + 1].copy()
            sr_slot = sr_index.get(srs[i])
            if sr_slot is not None:
                suffix_sr[i][sr_slot] += capacities[i]
        suffix_untracked_spk[i] = suffix_untracked_spk[i + 1] + (
            capacities[i] if speakers[i] not in tracked_speaker_set else 0
        )

    # root residual state；speaker usage 只携带当前 suffix 仍会出现的 binding
    # identity，并以 canonical identity 排序避免动态 tuple slot 漂移。
    rem_lang0 = (
        tuple(dict(margins.language)[key] for key in lang_keys)
        if lang_keys is not None
        else None
    )
    rem_ds0 = (
        tuple(dict(margins.dataset)[key] for key in ds_keys)
        if ds_keys is not None
        else None
    )
    spk_used0 = (
        tuple((identity, 0) for identity in tracked_speakers)
        if tracked_speakers is not None
        else None
    )
    slack0 = (
        speaker_cap * len(spk_used0) if spk_used0 is not None else None
    )
    residues0 = tuple(0 for _ in tracked_srs)

    # root 必要条件预检（不满足即数学不可行，不消耗任何计数）
    if actual % sr_modulus != 0:
        return None
    if actual > suffix_total[0]:
        return None
    if rem_lang0 is not None and any(
        rem_lang0[k] > suffix_lang[0][k] for k in range(nlang)
    ):
        return None
    if rem_ds0 is not None and any(
        rem_ds0[k] > suffix_ds[0][k] for k in range(nds)
    ):
        return None
    if slack0 is not None and actual > slack0 + suffix_untracked_spk[0]:
        return None

    root_state = (0, actual, rem_lang0, rem_ds0, spk_used0, residues0)
    memo = {}
    oracle_memo = {}
    no_result = object()
    child_result = no_result
    stack = [
        [root_state, 0, actual, rem_lang0, rem_ds0, spk_used0, slack0,
         residues0, None, None, None]
    ]

    while stack:
        frame = stack[-1]
        pos = frame[_POS]
        if frame[_GEN] is None:
            # 首次展开该 memo state（push 前已确认 memo miss）
            _bump(counts, budget, "states", 1)
            if pos == n:
                # terminal state：构造完整 candidate quota vector 计 n 次
                # materialized；不变量保证残量全零，仍显式复核兜底
                _bump(counts, budget, "materialized", n)
                feasible = frame[_REM] == 0
                if feasible and frame[_REM_LANG] is not None:
                    feasible = all(v == 0 for v in frame[_REM_LANG])
                if feasible and frame[_REM_DS] is not None:
                    feasible = all(v == 0 for v in frame[_REM_DS])
                if feasible:
                    feasible = all(v == 0 for v in frame[_RESIDUES])
                value = (Fraction(0), ()) if feasible else None
                memo[frame[_STATE]] = value
                stack.pop()
                child_result = value
                continue
            child_pos = pos + 1
            lower = max(0, frame[_REM] - suffix_total[child_pos])
            upper = min(capacities[pos], frame[_REM])
            if frame[_REM_LANG] is not None:
                slot = lang_index[languages[pos]]
                lower = max(
                    lower,
                    frame[_REM_LANG][slot] - suffix_lang[child_pos][slot],
                )
                upper = min(upper, frame[_REM_LANG][slot])
            if frame[_REM_DS] is not None:
                slot = ds_index[dataset_ids[pos]]
                lower = max(
                    lower,
                    frame[_REM_DS][slot] - suffix_ds[child_pos][slot],
                )
                upper = min(upper, frame[_REM_DS][slot])
            if frame[_SPK_USED] is not None and speakers[pos] in tracked_speaker_set:
                used = dict(frame[_SPK_USED])[speakers[pos]]
                upper = min(upper, speaker_cap - used)
            frame[_GEN] = (
                _quota_order(lower, upper, references[pos])
                if lower <= upper
                else iter(())
            )
            frame[_BEST] = None
        elif child_result is not no_result:
            # 子 state 刚完成：把 pending quota 与子 suffix 合并进 incumbent
            if child_result is not None:
                candidate = (
                    abs(Fraction(frame[_PENDING_Q]) - references[frame[_POS]])
                    + child_result[0],
                    (frame[_PENDING_Q],) + child_result[1],
                )
                if frame[_BEST] is None or candidate < frame[_BEST]:
                    frame[_BEST] = candidate
            child_result = no_result

        try:
            quota = next(frame[_GEN])
        except StopIteration:
            # 全部 quota 已尝试：该 state 最优 suffix 定稿并 memo 化
            memo[frame[_STATE]] = frame[_BEST]
            stack.pop()
            child_result = frame[_BEST]
            continue
        # 决策尝试先行计数：必要条件剪枝判死跳过的分支同样计一次 transition
        _bump(counts, budget, "transitions", 1)

        child_pos = pos + 1
        rem_after = frame[_REM] - quota
        if rem_after > suffix_total[child_pos]:
            continue
        rem_lang_after = frame[_REM_LANG]
        if rem_lang_after is not None:
            slot = lang_index[languages[pos]]
            updated = list(rem_lang_after)
            updated[slot] -= quota
            if updated[slot] > suffix_lang[child_pos][slot]:
                continue
            rem_lang_after = tuple(updated)
        rem_ds_after = frame[_REM_DS]
        if rem_ds_after is not None:
            slot = ds_index[dataset_ids[pos]]
            updated = list(rem_ds_after)
            updated[slot] -= quota
            if updated[slot] > suffix_ds[child_pos][slot]:
                continue
            rem_ds_after = tuple(updated)
        spk_used_after = frame[_SPK_USED]
        slack_after = frame[_SLACK]
        if speaker_expiries is not None:
            usage = dict(spk_used_after)
            if speakers[pos] in tracked_speaker_set:
                usage[speakers[pos]] += quota
                slack_after -= quota
            expired = speaker_expiries[child_pos]
            if expired:
                expired_set = set(expired)
                slack_after -= sum(
                    speaker_cap - usage[identity] for identity in expired
                )
                spk_used_after = tuple(
                    (identity, used)
                    for identity, used in spk_used_after
                    if identity not in expired_set
                )
            elif speakers[pos] in tracked_speaker_set:
                spk_used_after = tuple(
                    (identity, usage[identity])
                    for identity, _used in spk_used_after
                )
            # 弱必要条件（cheap pre-filter）：剩余量必须能被仍活跃 binding
            # speaker 的剩余 cap + 不受 speaker cap 约束的后缀容量吸收。
            if rem_after > slack_after + suffix_untracked_spk[child_pos]:
                continue
        residues_after = frame[_RESIDUES]
        if tracked_srs:
            sr_slot = sr_index.get(srs[pos])
            if sr_slot is not None:
                updated = list(residues_after)
                updated[sr_slot] = (updated[sr_slot] + quota) % sr_modulus
                residues_after = tuple(updated)
            # 每个 tracked SR：当前 residue 非零时，后缀必须至少提供到下一
            # 个模倍数的最小增量（suffix 容量 0 是其实例）
            pruned = False
            for slot in range(len(tracked_srs)):
                needed = (-residues_after[slot]) % sr_modulus
                if suffix_sr[child_pos][slot] < needed:
                    pruned = True
                    break
            if pruned:
                continue

        child_state = (
            child_pos, rem_after, rem_lang_after, rem_ds_after,
            spk_used_after, residues_after,
        )
        child_value = memo.get(child_state, no_result)
        if child_value is no_result:
            if spk_used_after:
                # 该 oracle 是必要条件剪枝，不表达 SR 模约束；true 不能直接
                # 构造或接受解。仅仍有 binding speaker、cheap 剪枝通过且
                # value-memo miss 时调用；判定结果按 state memo 化。
                feasible = oracle_memo.get(child_state)
                if feasible is None:
                    feasible = _speaker_flow_feasible(
                        position=child_pos,
                        capacities=capacities,
                        languages=languages,
                        dataset_ids=dataset_ids,
                        speakers=speakers,
                        tracked_speakers=tracked_speaker_set,
                        rem_total=rem_after,
                        rem_lang=rem_lang_after,
                        lang_index=lang_index,
                        rem_ds=rem_ds_after,
                        ds_index=ds_index,
                        spk_used=spk_used_after,
                        speaker_cap=speaker_cap,
                    )
                    oracle_memo[child_state] = feasible
                if not feasible:
                    continue
            frame[_PENDING_Q] = quota
            stack.append(
                [child_state, child_pos, rem_after, rem_lang_after,
                 rem_ds_after, spk_used_after, slack_after, residues_after,
                 None, None, None]
            )
        elif child_value is not None:
            candidate = (
                abs(Fraction(quota) - references[pos]) + child_value[0],
                (quota,) + child_value[1],
            )
            if frame[_BEST] is None or candidate < frame[_BEST]:
                frame[_BEST] = candidate

    root_best = memo[root_state]
    if root_best is None:
        return None
    deviation, quota_vector = root_best
    return (deviation, quota_vector), quota_vector


def _dinic_reaches(num_nodes, edges, source, sink, target):
    """整数容量最大流是否能达到 ``target``（Dinic；流量达到 target 提前停止）。

    ``edges`` 为 ``(u, v, capacity)`` 三元组；``cap <= 0`` 的边直接跳过
    （等效于不存在）。增广路径长度受分层数（约 7 层）限制，递归安全。
    """
    graph = [[] for _ in range(num_nodes)]  # 每项 [to, cap, reverse_index]
    for u, v, cap in edges:
        if cap <= 0:
            continue
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0, len(graph[u]) - 1])
    flow = 0
    while flow < target:
        level = [-1] * num_nodes
        level[source] = 0
        queue = [source]
        for u in queue:
            for to, cap, _rev in graph[u]:
                if cap > 0 and level[to] < 0:
                    level[to] = level[u] + 1
                    queue.append(to)
        if level[sink] < 0:
            break
        it = [0] * num_nodes

        def _augment(u, pushed):
            if u == sink:
                return pushed
            arcs = graph[u]
            while it[u] < len(arcs):
                edge = arcs[it[u]]
                to, cap, rev = edge
                if cap > 0 and level[to] == level[u] + 1:
                    pushed_now = _augment(to, min(pushed, cap))
                    if pushed_now:
                        edge[1] -= pushed_now
                        graph[to][rev][1] += pushed_now
                        return pushed_now
                it[u] += 1
            return 0

        while flow < target:
            pushed = _augment(source, target - flow)
            if not pushed:
                break
            flow += pushed
    return flow >= target


def _speaker_flow_feasible(
    *, position, capacities, languages, dataset_ids, speakers, tracked_speakers,
    rem_total, rem_lang, lang_index, rem_ds, ds_index,
    spk_used, speaker_cap,
):
    """speaker-cap residual max-flow oracle（必要条件剪枝）。

    分层网络：source → language 节点（cap = language remaining；轴未启用时
    source 直连 leaf，即 unconstrained aggregate side）→ leaf 节点对
    （node-split，cap = leaf capacity）→ binding speaker 节点对（cap = 剩余
    cap；非 binding / None-speaker leaf 绕过该层）→ dataset 节点（cap =
    dataset remaining；轴未启用时直连 sink）→ sink。流量达到
    ``rem_total`` 即判可行。

    language→leaf、leaf→speaker 边按 leaf 自身归属铺设（精确）；speaker→
    dataset 边按 suffix 内存在的 (speaker, dataset) 对铺设，flow 在 speaker
    节点混流，因此 **dataset 层是松弛**。真可行 ⟹ flow 可行（每单位沿自身
    leaf/speaker/dataset 路径即可构造流），故 flow 不可行 ⟹ 真不可行，可
    安全剪枝；flow 可行不代表真可行。

    该 oracle 是必要条件剪枝，不表达 SR 模约束；true 不能直接构造或接受解。
    禁止将 max-flow 的第一条流转换为最终 leaf quotas。
    """
    if rem_total == 0:
        return True
    suffix = [i for i in range(position, len(capacities)) if capacities[i] > 0]
    if not suffix:
        return False

    source = 0
    next_id = 1
    lang_node = {}
    if rem_lang is not None:
        for slot in range(len(rem_lang)):
            lang_node[slot] = next_id
            next_id += 1
    leaf_in = {}
    leaf_out = {}
    for i in suffix:
        leaf_in[i] = next_id
        leaf_out[i] = next_id + 1
        next_id += 2
    suffix_speakers = sorted(
        {speakers[i] for i in suffix if speakers[i] in tracked_speakers}
    )
    spk_in = {}
    spk_out = {}
    for identity in suffix_speakers:
        spk_in[identity] = next_id
        spk_out[identity] = next_id + 1
        next_id += 2
    ds_node = {}
    if rem_ds is not None:
        for slot in range(len(rem_ds)):
            ds_node[slot] = next_id
            next_id += 1
    sink = next_id
    num_nodes = sink + 1

    edges = []
    if rem_lang is not None:
        for slot, node in lang_node.items():
            edges.append((source, node, rem_lang[slot]))
    for i in suffix:
        if rem_lang is not None:
            edges.append(
                (lang_node[lang_index[languages[i]]], leaf_in[i], rem_total)
            )
        else:
            edges.append((source, leaf_in[i], capacities[i]))
        # node-split：leaf 容量约束
        edges.append((leaf_in[i], leaf_out[i], capacities[i]))
        identity = speakers[i]
        if identity in tracked_speakers:
            edges.append((leaf_out[i], spk_in[identity], rem_total))
        elif rem_ds is not None:
            # 非 binding / None-speaker leaf 直达自己的 dataset（不混流，精确）
            edges.append(
                (leaf_out[i], ds_node[ds_index[dataset_ids[i]]], rem_total)
            )
        else:
            edges.append((leaf_out[i], sink, capacities[i]))
    usage = dict(spk_used)
    for identity in suffix_speakers:
        slack = speaker_cap - usage[identity]
        # node-split：speaker 剩余 cap 约束（slack <= 0 的边被 Dinic 跳过）
        edges.append((spk_in[identity], spk_out[identity], slack))
        if rem_ds is not None:
            for dataset in sorted(
                {dataset_ids[i] for i in suffix if speakers[i] == identity}
            ):
                edges.append(
                    (spk_out[identity], ds_node[ds_index[dataset]], rem_total)
                )
        else:
            edges.append((spk_out[identity], sink, slack))
    if rem_ds is not None:
        for slot, node in ds_node.items():
            edges.append((node, sink, rem_ds[slot]))
    return _dinic_reaches(num_nodes, edges, source, sink, rem_total)


def _bump(counts, budget, field, amount):
    """按累计值计数并做预算上限检查（超出即 fail-fast）。"""
    limit = getattr(budget, f"max_{field}")
    value = counts[field] + amount
    if value > limit:
        raise SolverBudgetExceeded(
            f"{field} budget exceeded: {value} > {limit} "
            f"(cumulative states={counts['states']}, "
            f"transitions={counts['transitions']}, "
            f"materialized={counts['materialized']})"
        )
    counts[field] = value


__all__ = [
    "CanonicalSpeaker",
    "SolverAttempt",
    "SolverBudget",
    "SolverBudgetExceeded",
    "SolverLeaf",
    "SolverLeafKey",
    "SolverMargins",
    "SolverResult",
    "SolverStats",
    "solve_leaf_quotas",
]
