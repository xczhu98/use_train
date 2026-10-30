"""Exhaustive oracle tests for the independent speech quota solver.

Task 2 冻结合同：
- ``SolverLeaf`` / ``SolverMargins`` / ``SolverBudget`` / ``SolverStats`` /
  ``SolverResult`` / ``SolverAttempt`` / ``SolverBudgetExceeded`` /
  ``solve_leaf_quotas(...)`` 的精确 immutable contracts；
- test-only exhaustive oracle（leaf<=7、每 leaf capacity<=2、actual<=6）作为
  production solver 的独立参考实现，8 种 controller 组合 × 固定 seed 小图
  逐图比对可行性（``result is None`` 一致）与最佳解（objective 字典序一致）。
- solver 模块禁止 import ``core.data.sampling``（防循环依赖）。
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from pathlib import Path
import random

import pytest

from core.data.speech_quota_solver import (
    SolverAttempt,
    SolverBudget,
    SolverBudgetExceeded,
    SolverLeaf,
    SolverMargins,
    SolverResult,
    SolverStats,
    solve_leaf_quotas,
)


def test_solver_does_not_import_sampling():
    source = Path("core/data/speech_quota_solver.py").read_text()
    assert "core.data.sampling" not in source
    assert "from .sampling" not in source


# ---------------------------------------------------------------------------
# Step 3: exhaustive oracle（与 production solver 语义一致的独立参考实现）
# ---------------------------------------------------------------------------


def exact_margins_hold(leaves, quotas, margins):
    """language/dataset exact margin 逐轴逐项校验。

    ``margins.language`` 为 ``None`` 表示该轴未启用；启用时每个
    ``(key, target)`` 都必须在对应维（language 名 / dataset_id）上精确命中。
    """
    if margins.language is not None:
        for language, target in margins.language:
            total = sum(
                quota
                for leaf, quota in zip(leaves, quotas)
                if leaf.key[0] == language
            )
            if total != target:
                return False
    if margins.dataset is not None:
        for dataset, target in margins.dataset:
            total = sum(
                quota
                for leaf, quota in zip(leaves, quotas)
                if leaf.key[1] == dataset
            )
            if total != target:
                return False
    return True


def speaker_cap_holds(leaves, quotas, speaker_cap):
    """canonical speaker identity ``(dataset_id, speaker_id)`` 的 absolute upper cap。

    ``speaker_cap`` 为 ``None`` 时该约束不生效；``None`` speaker 身份的 leaf
    不参与任何 speaker 聚合。同裸 speaker 名跨 dataset 视为不同 identity。
    """
    if speaker_cap is None:
        return True
    usage = {}
    for leaf, quota in zip(leaves, quotas):
        if leaf.key[2] is not None:
            usage[leaf.key[2]] = usage.get(leaf.key[2], 0) + quota
    return all(used <= speaker_cap for used in usage.values())


def sr_moduli_hold(leaves, quotas, sr_modulus):
    """每个 ``batching_sr`` 的总 quota 必须满足 DDP granularity 模约束。"""
    totals = {}
    for leaf, quota in zip(leaves, quotas):
        totals[leaf.key[3]] = totals.get(leaf.key[3], 0) + quota
    return all(total % sr_modulus == 0 for total in totals.values())


def objective(leaves, quotas):
    """production objective：第一项为相对 reference 的绝对偏差和（精确
    Fraction 运算），第二项为 stable leaf order 下的 quota vector，字典序
    较小者胜。"""
    return (
        sum(abs(Fraction(quota) - leaf.reference) for leaf, quota in zip(leaves, quotas)),
        tuple(quotas),
    )


def exhaustive_solution(leaves, actual, margins, speaker_cap, sr_modulus):
    assert len(leaves) <= 7
    assert max((leaf.capacity for leaf in leaves), default=0) <= 2
    assert actual <= 6
    feasible = []
    for quotas in itertools.product(*(range(leaf.capacity + 1) for leaf in leaves)):
        if sum(quotas) != actual:
            continue
        if not exact_margins_hold(leaves, quotas, margins):
            continue
        if not speaker_cap_holds(leaves, quotas, speaker_cap):
            continue
        if not sr_moduli_hold(leaves, quotas, sr_modulus):
            continue
        feasible.append((objective(leaves, quotas), quotas))
    return min(feasible, default=None)


def normalize(result):
    """把 oracle ``(objective, quotas)`` 与 ``SolverResult`` 归一化为同一可比值。"""
    if result is None:
        return None
    if isinstance(result, SolverResult):
        return (result.objective, tuple(quota for _key, quota in result.leaf_quotas))
    return (result[0], tuple(result[1]))


# ---------------------------------------------------------------------------
# Step 4: 8 种 controller 组合 × 固定 seed 小图的 oracle matrix
# ---------------------------------------------------------------------------


def _leaf(language, dataset, speaker, sr, capacity, reference):
    return SolverLeaf((language, dataset, speaker, sr), capacity, reference)


def make_graph(rng):
    """随机小图：<=7 leaves、每 leaf capacity<=2；必含两个 SR、同 identity
    跨 language、跨 dataset 同裸 speaker（``("a","s1")`` / ``("b","s1")``）。

    固定核心 4 leaves 保证三类结构特征，随机补 0..3 个候选并随机化
    capacity（0..2）与 reference，最后按键去重防重复 key。
    """
    core = [
        _leaf("zh", "a", ("a", "s1"), 16000, 2, Fraction(3, 2)),
        _leaf("en", "a", ("a", "s1"), 48000, 2, Fraction(1, 2)),
        _leaf("zh", "b", ("b", "s1"), 16000, 1, Fraction(2, 1)),
        _leaf("en", "b", None, 48000, 1, Fraction(1, 1)),
    ]
    candidates = [
        _leaf("zh", "c", ("c", "s2"), 48000, 2, Fraction(1, 3)),
        _leaf("en", "c", ("c", "s2"), 16000, 1, Fraction(5, 2)),
        _leaf("other", "a", ("a", "s3"), 16000, 2, Fraction(0)),
        _leaf("zh", "a", ("a", "s4"), 48000, 1, Fraction(3, 1)),
        _leaf("other", "b", None, 48000, 2, Fraction(1, 2)),
        _leaf("en", "a", None, 16000, 2, Fraction(4, 3)),
        _leaf("zh", "c", None, 16000, 1, Fraction(2, 1)),
    ]
    rng.shuffle(candidates)
    selected = list(core)
    seen = {leaf.key for leaf in selected}
    for candidate in candidates:
        if len(selected) >= 7:
            break
        if candidate.key in seen:
            continue
        seen.add(candidate.key)
        selected.append(candidate)
    randomized = []
    for leaf in selected:
        capacity = rng.randint(0, 2)
        reference = Fraction(rng.randint(0, 6), rng.choice([1, 2, 3]))
        randomized.append(SolverLeaf(leaf.key, capacity, reference))
    return randomized


def random_partition(rng, total, groups):
    """把 ``total`` 随机拆成 ``groups`` 个非负整数（逐单位投掷，和恒为 total）。"""
    values = [0] * groups
    for _ in range(total):
        values[rng.randrange(groups)] += 1
    return tuple(values)


def make_margins(rng, leaves, actual, language_on, dataset_on):
    language = dataset = None
    if language_on:
        keys = sorted({leaf.key[0] for leaf in leaves})
        language = tuple(
            zip(keys, random_partition(rng, actual, len(keys)))
        )
    if dataset_on:
        keys = sorted({leaf.key[1] for leaf in leaves})
        dataset = tuple(
            zip(keys, random_partition(rng, actual, len(keys)))
        )
    return SolverMargins(language=language, dataset=dataset)


def _stable_sort_key(leaf):
    """stable leaf order 排序键（与 solver 契约一致）：``SolverLeafKey``
    升序，``None`` speaker 按 ``("", "")`` 参与比较。oracle 与 solver 两端
    独立实现同一规则，矩阵测试强制两者一致。"""
    language, dataset_id, speaker, sr = leaf.key
    speaker_sort = speaker if speaker is not None else ("", "")
    return (language, dataset_id, speaker_sort[0], speaker_sort[1], sr)


def solve_with_defaults(
    leaves, actual, margins, speaker_cap, sr_modulus, cumulative_stats=None
):
    return solve_leaf_quotas(
        leaves,
        actual=actual,
        margins=margins,
        speaker_cap=speaker_cap,
        sr_modulus=sr_modulus,
        budget=SolverBudget(),
        cumulative_stats=cumulative_stats,
    )


@pytest.mark.parametrize("language_on", [False, True])
@pytest.mark.parametrize("dataset_on", [False, True])
@pytest.mark.parametrize("speaker_on", [False, True])
def test_oracle_matrix_matches_exhaustive_solution(
    language_on, dataset_on, speaker_on
):
    combo = (
        (1 if language_on else 0)
        | (2 if dataset_on else 0)
        | (4 if speaker_on else 0)
    )
    for seed in range(25):
        rng = random.Random(combo * 1000 + seed)
        leaves = sorted(make_graph(rng), key=_stable_sort_key)
        actual = rng.randint(0, 6)
        margins = make_margins(rng, leaves, actual, language_on, dataset_on)
        speaker_cap = rng.randint(1, 3) if speaker_on else None
        sr_modulus = rng.choice([1, 2, 3])

        expected = exhaustive_solution(
            leaves, actual, margins, speaker_cap, sr_modulus
        )
        attempt = solve_with_defaults(
            leaves, actual, margins, speaker_cap, sr_modulus
        )
        assert attempt.result is not None or expected is None
        assert normalize(attempt.result) == normalize(expected)
        if attempt.result is not None:
            assert [key for key, _quota in attempt.result.leaf_quotas] == [
                leaf.key for leaf in leaves
            ]
        assert isinstance(attempt.stats, SolverStats)


def test_stats_cumulate_across_calls():
    leaves = [
        SolverLeaf(("zh", "a", None, 16000), 2, Fraction(1)),
        SolverLeaf(("en", "b", None, 48000), 2, Fraction(1)),
    ]
    first = solve_with_defaults(
        leaves, actual=2, margins=SolverMargins(None, None),
        speaker_cap=None, sr_modulus=1,
    )
    second = solve_with_defaults(
        leaves, actual=2, margins=SolverMargins(None, None),
        speaker_cap=None, sr_modulus=1,
        cumulative_stats=first.stats,
    )
    assert second.stats.states >= first.stats.states
    assert second.stats.transitions >= first.stats.transitions
    assert second.stats.materialized >= first.stats.materialized


def test_empty_leaves_with_positive_actual_are_infeasible():
    """I1：空 leaves + actual>0 必须返回 ``result is None``（sum 硬校验兜底
    根调用即叶节点的场景），与 exhaustive oracle 一致。"""
    margins = SolverMargins(None, None)
    attempt = solve_with_defaults([], actual=5, margins=margins, speaker_cap=None, sr_modulus=1)
    assert attempt.result is None
    assert exhaustive_solution([], 5, margins, None, 1) is None


def test_empty_leaves_with_zero_actual_are_feasible():
    """actual=0 且无叶子应返回空配额可行解，与 exhaustive oracle 一致。"""
    margins = SolverMargins(None, None)
    attempt = solve_with_defaults([], actual=0, margins=margins, speaker_cap=None, sr_modulus=1)
    assert attempt.result is not None
    assert attempt.result.leaf_quotas == ()
    assert attempt.result.objective == (Fraction(0), ())
    assert normalize(attempt.result) == normalize(
        exhaustive_solution([], 0, margins, None, 1)
    )


def test_budget_counts_match_frozen_rule():
    """I2：冻结计数口径的精确数值断言（固定小图，memoized DP 口径）。

    图：capacity [2, 1]、actual=1、无 margin、sr_modulus=1（单一 SR，
    tracked residue 为空）。memoized DFS 首次展开 4 个 state：
    root (pos0, rem1) → (pos1, rem0) → (pos2, rem0) terminal，以及
    (pos1, rem1)；terminal (pos2, rem0) 第二次到达时为 memo hit 不计。
    transitions = 4：root 枚举 2 个 quota、(pos1, rem0) 枚举 1 个、
    (pos1, rem1) 的 suffix lower bound 排除必死 q=0，仅枚举 q=1。
    materialized = 2：唯一 terminal state 首次展开 +len(leaves)=2。最佳解
    (0, 1)，objective 1。
    """
    leaves = [
        SolverLeaf(("zh", "a", None, 16000), 2, Fraction(1)),
        SolverLeaf(("zh", "b", None, 16000), 1, Fraction(1)),
    ]
    attempt = solve_with_defaults(
        leaves, actual=1, margins=SolverMargins(None, None),
        speaker_cap=None, sr_modulus=1,
    )
    assert attempt.stats == SolverStats(states=4, transitions=4, materialized=2)
    assert attempt.result is not None
    assert attempt.result.objective == (Fraction(1), (0, 1))


# ---------------------------------------------------------------------------
# Task 3 Step 1: alternate-flow / speaker / residue 定向 fixtures
# ---------------------------------------------------------------------------


def test_alternate_flow_sr_residue_finds_cycle_solution():
    """A：首个 language-dataset flow 的 SR residue 非零，cycle alternate flow 可行。

    2x2 transportation 结构（zh/en × a/b，margin 各 1），reference 偏好
    (zh,a)+(en,b) 的 flow，但该 flow 使 48000 SR 总量为 1（% 2 != 0）
    不可行；solver 必须回退到偏差更大的 alternate flow (en,a)+(zh,b)。
    stable leaf order: (en,a),(en,b),(zh,a),(zh,b)。
    """
    leaves = [
        SolverLeaf(("zh", "a", None, 48000), 1, Fraction(1)),
        SolverLeaf(("zh", "b", None, 16000), 1, Fraction(0)),
        SolverLeaf(("en", "a", None, 16000), 1, Fraction(0)),
        SolverLeaf(("en", "b", None, 16000), 1, Fraction(1)),
    ]
    margins = SolverMargins(
        language=(("en", 1), ("zh", 1)), dataset=(("a", 1), ("b", 1))
    )
    ordered = sorted(leaves, key=_stable_sort_key)
    expected = exhaustive_solution(ordered, 2, margins, None, 2)
    attempt = solve_with_defaults(
        leaves, actual=2, margins=margins, speaker_cap=None, sr_modulus=2
    )
    assert attempt.result is not None
    assert normalize(attempt.result) == normalize(expected)
    assert attempt.result.objective == (Fraction(4), (1, 0, 0, 1))


def test_speaker_cap_accumulates_across_languages():
    """B：language margins 可行，但同 canonical speaker 跨 language 累计超 cap。

    ("a","s1") 同时出现在 zh/en 两个 leaf；cap=1 迫使 reference 最优
    （speaker 用量 3）被放弃。stable leaf order 下最优 tie-break 为
    (0,1,2,0)（en=None / en=s1 / zh=None / zh=s1）。
    """
    leaves = [
        SolverLeaf(("zh", "a", ("a", "s1"), 16000), 2, Fraction(2)),
        SolverLeaf(("en", "a", ("a", "s1"), 16000), 2, Fraction(2)),
        SolverLeaf(("zh", "a", None, 16000), 2, Fraction(0)),
        SolverLeaf(("en", "a", None, 16000), 2, Fraction(0)),
    ]
    margins = SolverMargins(language=(("en", 1), ("zh", 2)), dataset=None)
    ordered = sorted(leaves, key=_stable_sort_key)
    expected = exhaustive_solution(ordered, 3, margins, 1, 1)
    attempt = solve_with_defaults(
        leaves, actual=3, margins=margins, speaker_cap=1, sr_modulus=1
    )
    assert attempt.result is not None
    assert normalize(attempt.result) == normalize(expected)
    assert attempt.result.objective == (Fraction(5), (0, 1, 2, 0))


def test_last_sr_residue_implied_by_total_and_other_residues():
    """C：最后一个 SR 的 residue 由 actual 与前 N-1 个 SR residue 隐含。

    actual=4、m=2：16000 总量必须为偶；48000（sorted 最后）的 residue 不
    进入 state，由 actual % m == 0 与其他 SR residue 全零隐含为 0。
    最优解 (1,1,2) 必须被找到（防止对最后 SR 过度剪枝/错误建状态）。
    """
    leaves = [
        SolverLeaf(("zh", "a", None, 16000), 2, Fraction(1)),
        SolverLeaf(("zh", "b", None, 16000), 2, Fraction(1)),
        SolverLeaf(("zh", "c", None, 48000), 2, Fraction(2)),
    ]
    margins = SolverMargins(None, None)
    ordered = sorted(leaves, key=_stable_sort_key)
    expected = exhaustive_solution(ordered, 4, margins, None, 2)
    attempt = solve_with_defaults(
        leaves, actual=4, margins=margins, speaker_cap=None, sr_modulus=2
    )
    assert attempt.result is not None
    assert normalize(attempt.result) == normalize(expected)
    assert attempt.result.objective == (Fraction(0), (1, 1, 2))


def test_speaker_flow_oracle_prunes_misaligned_capacity():
    """D：cheap 剪枝全部通过、只有 max-flow oracle 能判死的 speaker 错位。

    en 需求 3 只能经 speaker ("b","s2")（cap 2 → 最多供 2）；zh 需求 2 有
    ("a","s1") 与 None-speaker leaf 兜底。aggregate slack（4）+ None 后缀
    容量（1）== actual（5），per-language suffix 容量也逐项达标——cheap
    必要条件全部通过；但 en 的 reachable slack 只有 2 < 3，max-flow 不可行。

    stable leaf order 下 language suffix lower bound 在枚举前排除 q=0，仅枚举
    q=2/q=1，两个 child 均被 oracle 判死。因此精确计数 states=1（仅 root
    展开）、transitions=2、materialized=0——该断言同时证明 lower bound 和
    oracle 均生效（cheap-only 实现会展开 child 子树后才返回 None，states
    必然 > 1）。
    """
    leaves = [
        SolverLeaf(("en", "b", ("b", "s2"), 16000), 2, Fraction(2)),
        SolverLeaf(("en", "b", ("b", "s2"), 48000), 2, Fraction(2)),
        SolverLeaf(("zh", "a", ("a", "s1"), 16000), 2, Fraction(2)),
        SolverLeaf(("zh", "d", None, 16000), 1, Fraction(1)),
    ]
    margins = SolverMargins(language=(("en", 3), ("zh", 2)), dataset=None)
    ordered = sorted(leaves, key=_stable_sort_key)
    assert exhaustive_solution(ordered, 5, margins, 2, 1) is None
    attempt = solve_with_defaults(
        leaves, actual=5, margins=margins, speaker_cap=2, sr_modulus=1
    )
    assert attempt.result is None
    assert attempt.stats == SolverStats(states=1, transitions=2, materialized=0)


def test_quota_order_matches_sorted_reference():
    """quota 枚举顺序 helper：与 ``sorted(key=(|q-ref|, q))`` 完全一致。

    枚举顺序只影响 incumbent 发现先后，不影响完整性；但该 helper 声称复现
    精确顺序，直接锁定行为防回归。
    """
    from core.data.speech_quota_solver import _quota_order

    for numerator in range(0, 9):
        for denominator in (1, 2, 3, 5):
            reference = Fraction(numerator, denominator)
            for upper in range(0, 6):
                for lower in range(upper + 1):
                    expected = sorted(
                        range(lower, upper + 1),
                        key=lambda q: (abs(Fraction(q) - reference), q),
                    )
                    assert list(_quota_order(lower, upper, reference)) == expected


def test_budget_exhaustion_raises_instead_of_returning_none():
    leaves = [
        SolverLeaf(("zh", "a", None, 16000), 2, Fraction(1)),
        SolverLeaf(("zh", "b", None, 16000), 2, Fraction(1)),
    ]
    with pytest.raises(SolverBudgetExceeded, match="states budget exceeded"):
        solve_leaf_quotas(
            leaves,
            actual=2,
            margins=SolverMargins(None, None),
            speaker_cap=None,
            sr_modulus=1,
            budget=SolverBudget(max_states=1),
        )


def test_solver_uses_suffix_lower_bound_before_counting_transitions():
    """后缀容量推出唯一 quota 时，只尝试该 quota，不计必死 transition。"""
    leaves = [
        SolverLeaf(("zh", "a", None, 16000), 100, Fraction(0)),
    ]
    attempt = solve_leaf_quotas(
        leaves,
        actual=100,
        margins=SolverMargins(None, None),
        speaker_cap=None,
        sr_modulus=1,
        budget=SolverBudget(max_transitions=1),
    )

    assert attempt.result is not None
    assert attempt.result.leaf_quotas == ((leaves[0].key, 100),)
    assert attempt.stats.transitions == 1


def test_dataset_suffix_lower_bound_applies_before_counting_transitions():
    """dataset-only exact lower 必须在枚举前推出 a leaf 的唯一 quota=100。

    后缀总容量仍为 100，所以 total lower=0；language axis 关闭。只有 dataset
    a 的 remaining=100 且其后缀 capacity=0 能推出 lower=100。两个 leaf 各
    产生一个真实 transition，因此 2 是 public solver 可成功的最小预算。
    """
    leaves = [
        SolverLeaf(("zh", "a", None, 16000), 100, Fraction(0)),
        SolverLeaf(("zh", "b", None, 16000), 100, Fraction(0)),
    ]
    attempt = solve_leaf_quotas(
        leaves,
        actual=100,
        margins=SolverMargins(
            language=None,
            dataset=(("a", 100), ("b", 0)),
        ),
        speaker_cap=None,
        sr_modulus=1,
        budget=SolverBudget(max_transitions=2),
    )

    assert attempt.result is not None
    assert attempt.result.objective == (Fraction(100), (100, 0))
    assert attempt.stats.transitions == 2


def test_non_binding_independent_speakers_do_not_expand_exponential_state():
    """各 speaker 总容量不超过 cap 时，不应把独立 0/1 历史带入 memo key。"""
    leaves = [
        SolverLeaf(("zh", f"d{index:02d}", (f"d{index:02d}", "s"), 16000), 1, Fraction(1, 2))
        for index in range(19)
    ]
    expected_keys = tuple(leaf.key for leaf in leaves)

    for candidate_leaves in (leaves, list(reversed(leaves))):
        attempt = solve_leaf_quotas(
            candidate_leaves,
            actual=9,
            margins=SolverMargins(None, None),
            speaker_cap=1,
            sr_modulus=1,
            budget=SolverBudget(),
        )
        assert attempt.result is not None
        assert attempt.result.leaf_quotas == tuple(
            (key, 0 if index < 10 else 1)
            for index, key in enumerate(expected_keys)
        )
        assert attempt.stats.states <= SolverBudget().max_states
        assert attempt.stats.transitions <= SolverBudget().max_transitions


def test_binding_speakers_leave_memo_state_after_their_suffix():
    """binding identity 在第二个相邻 leaf 后必须退出后续 memo key。

    每个 identity 总 capacity=2 > cap=1，确保进入 tracked usage；19 个已结束
    identity 若永久留在 state，会保留指数级 0/1 历史并超过默认 state budget。
    """
    leaves = [
        SolverLeaf(
            ("zh", f"d{index:02d}", (f"d{index:02d}", "s"), sr),
            1,
            Fraction(1, 2),
        )
        for index in range(19)
        for sr in (16000, 48000)
    ]
    expected_quotas = (0, 0) * 10 + (0, 1) * 9
    expected_keys = tuple(leaf.key for leaf in leaves)

    for candidate_leaves in (leaves, list(reversed(leaves))):
        attempt = solve_leaf_quotas(
            candidate_leaves,
            actual=9,
            margins=SolverMargins(None, None),
            speaker_cap=1,
            sr_modulus=1,
            budget=SolverBudget(),
        )
        assert attempt.result is not None
        assert attempt.result.objective == (Fraction(19), expected_quotas)
        assert attempt.result.leaf_quotas == tuple(zip(expected_keys, expected_quotas))
        assert attempt.stats == SolverStats(
            states=318, transitions=514, materialized=38
        )


def test_overlapping_binding_intervals_do_not_materialize_quadratic_active_tuples(
    monkeypatch,
):
    """高度重叠 interval 的 speaker 预处理只物化 O(speaker_count) identity。

    所有 identity 在排序前半首次出现、后半最后出现。旧 active_speakers 会为
    每个 position 物化全部仍活跃 identity，产生 Θ(n²) canonical identity slot。
    用 tuple spy 计结构物化，不依赖 wall-clock；actual 超总容量使搜索在 root
    预检返回，计数仅覆盖预算前预处理。
    """
    import builtins

    speaker_count = 128
    leaves = [
        SolverLeaf(
            (language, f"d{index:03d}", (f"d{index:03d}", "s"), 16000),
            1,
            Fraction(0),
        )
        for language in ("a", "z")
        for index in range(speaker_count)
    ]

    class CountingTupleMeta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, builtins.tuple)

    class CountingTuple(builtins.tuple, metaclass=CountingTupleMeta):
        identity_slots = 0

        def __new__(cls, iterable=()):
            value = builtins.tuple(iterable)
            if value and all(
                isinstance(item, builtins.tuple)
                and len(item) == 2
                and all(isinstance(part, str) for part in item)
                for item in value
            ):
                cls.identity_slots += len(value)
            return builtins.tuple.__new__(cls, value)

    monkeypatch.setitem(solve_leaf_quotas.__globals__, "tuple", CountingTuple)
    attempt = solve_leaf_quotas(
        leaves,
        actual=2 * speaker_count + 1,
        margins=SolverMargins(None, None),
        speaker_cap=1,
        sr_modulus=1,
        budget=SolverBudget(),
    )

    assert attempt.result is None
    assert attempt.stats == SolverStats(states=0, transitions=0, materialized=0)
    assert CountingTuple.identity_slots <= 2 * speaker_count


# ---------------------------------------------------------------------------
# Task 3 Step 2: 预算跨候选累计 + 三类计数器 fail-fast（含 used/limit）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("budget", "match"),
    [
        (SolverBudget(max_states=3), r"states budget exceeded: 4 > 3"),
        (SolverBudget(max_transitions=3), r"transitions budget exceeded: 4 > 3"),
        (SolverBudget(max_materialized=1), r"materialized budget exceeded: 2 > 1"),
    ],
)
def test_each_budget_counter_fails_fast_with_used_and_limit(budget, match):
    """三类 budget counter 各自触发 fail-fast，错误信息包含 used/limit。

    固定小图（capacity [2, 1]、actual=1、sr_modulus=1，memoized 口径下精确
    计数 states=4 / transitions=4 / materialized=2），把对应上限设为
    实际值 -1 即在该计数器首次超限时抛出。
    """
    leaves = [
        SolverLeaf(("zh", "a", None, 16000), 2, Fraction(1)),
        SolverLeaf(("zh", "b", None, 16000), 1, Fraction(1)),
    ]
    with pytest.raises(SolverBudgetExceeded, match=match):
        solve_leaf_quotas(
            leaves,
            actual=1,
            margins=SolverMargins(None, None),
            speaker_cap=None,
            sr_modulus=1,
            budget=budget,
        )


def test_budget_is_cumulative_across_candidate_actuals():
    """预算计数跨候选（跨 ``solve_leaf_quotas`` 调用）累计，超限 fail-fast。

    固定图（3 leaves capacity 2、单 SR、无 margin）memoized 口径下：
    actual=2 计数 (8, 12, 3)、actual=1 增量 (6, 7, 3)、actual=0 增量
    (4, 3, 3)。前两次调用累计 transitions=19，第三次调用第 3 个
    transition 使累计达 22 > 21 触发 fail-fast；states/materialized 上限
    宽松，保证是 transitions 先超限。
    """
    leaves = [SolverLeaf(("zh", c, None, 16000), 2, Fraction(1)) for c in "abc"]
    budget = SolverBudget(max_states=40, max_transitions=21, max_materialized=100)
    stats = None
    for actual in (2, 1):
        attempt = solve_leaf_quotas(
            leaves,
            actual=actual,
            margins=SolverMargins(None, None),
            speaker_cap=None,
            sr_modulus=1,
            budget=budget,
            cumulative_stats=stats,
        )
        stats = attempt.stats
    assert stats == SolverStats(states=14, transitions=19, materialized=6)
    with pytest.raises(
        SolverBudgetExceeded, match=r"transitions budget exceeded: 22 > 21"
    ):
        solve_leaf_quotas(
            leaves,
            actual=0,
            margins=SolverMargins(None, None),
            speaker_cap=None,
            sr_modulus=1,
            budget=budget,
            cumulative_stats=stats,
        )


# ---------------------------------------------------------------------------
# Step 5: contract validation 失败测试
# ---------------------------------------------------------------------------


def leaf(key, capacity=1, reference=Fraction(1)):
    return SolverLeaf(key, capacity, reference)


def reject(match, **kwargs):
    defaults = dict(
        leaves=(leaf(("zh", "a", None, 16000)),),
        actual=1,
        margins=SolverMargins(None, None),
        speaker_cap=None,
        sr_modulus=1,
        budget=SolverBudget(),
    )
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        solve_leaf_quotas(**defaults)


@pytest.mark.parametrize(
    "leaves",
    [
        [leaf(("zh", "a", None, 16000)), leaf(("zh", "a", None, 16000))],
        [leaf(("zh", "a", ("a", "s1"), 16000)), leaf(("zh", "a", ("a", "s1"), 16000))],
    ],
)
def test_duplicate_leaf_keys_are_rejected(leaves):
    reject("duplicate leaf key", leaves=leaves)


@pytest.mark.parametrize(
    "leaves",
    [
        [leaf(("zh", "a", None, 16000), capacity=-1)],
        [leaf(("zh", "a", None, 16000), capacity=True)],
        [leaf(("zh", "a", None, 16000), capacity=1.5)],
    ],
)
def test_invalid_leaf_capacities_are_rejected(leaves):
    reject("capacity must be a non-negative int", leaves=leaves)


@pytest.mark.parametrize("actual", [-1, True, 1.5])
def test_negative_or_non_int_actual_is_rejected(actual):
    reject("actual must be a non-negative int", actual=actual)


@pytest.mark.parametrize(
    ("leaves", "margins"),
    [
        (
            (leaf(("zh", "a", None, 16000)),),
            SolverMargins(language=(("zh", 2),), dataset=None),
        ),
        (
            (leaf(("zh", "a", None, 16000)),),
            SolverMargins(language=None, dataset=(("a", 2),)),
        ),
        (
            (leaf(("zh", "a", None, 16000)),),
            SolverMargins(language=(("zh", 0),), dataset=(("a", 1),)),
        ),
        (
            (leaf(("zh", "a", None, 16000)), leaf(("en", "b", None, 16000))),
            SolverMargins(language=(("zh", 1), ("en", 1)), dataset=(("a", 2),)),
        ),
    ],
)
def test_margin_sum_mismatching_actual_is_rejected(leaves, margins):
    reject("must sum to actual", leaves=leaves, margins=margins)


@pytest.mark.parametrize(
    "margins",
    [
        SolverMargins(language=(("fr", 1),), dataset=None),
        SolverMargins(language=None, dataset=(("unknown", 1),)),
    ],
)
def test_unknown_margin_keys_are_rejected(margins):
    reject("unknown key", margins=margins)


@pytest.mark.parametrize(
    "margins",
    [
        SolverMargins(language=(), dataset=None),
        SolverMargins(language=None, dataset=()),
        SolverMargins(language=(("zh", 1), ("zh", 0)), dataset=None),
    ],
)
def test_empty_or_duplicate_margin_entries_are_rejected(margins):
    reject("non-empty tuple|duplicate key", margins=margins)


@pytest.mark.parametrize("axis", ["language", "dataset"])
@pytest.mark.parametrize(
    "pair",
    [
        None,
        ("zh",),
        ("zh", 1, "extra"),
        ["zh", 1],
        (("nested", "key"), 1),
    ],
)
def test_malformed_margin_pairs_are_fieldized_value_errors(axis, pair):
    margins = SolverMargins(
        language=(pair,) if axis == "language" else None,
        dataset=(pair,) if axis == "dataset" else None,
    )

    reject(rf"margins\.{axis}\[0\]", margins=margins)


def test_incomplete_margin_coverage_is_rejected():
    """M5 冻结语义：enabled 的 exact margin 必须完整覆盖 leaves 中该轴出现
    的全部 key；缺失 key 属于 contract error，立即 ValueError。"""
    leaves = (
        leaf(("zh", "a", None, 16000)),
        leaf(("en", "b", None, 16000)),
    )
    reject(
        "margins.language must cover all leaf language keys; missing: \\['en'\\]",
        leaves=leaves,
        margins=SolverMargins(language=(("zh", 1),), dataset=None),
    )
    reject(
        "margins.dataset must cover all leaf dataset keys; missing: \\['b'\\]",
        leaves=leaves,
        margins=SolverMargins(language=None, dataset=(("a", 1),)),
    )


@pytest.mark.parametrize("sr_modulus", [0, -1, True, 1.5])
def test_non_positive_sr_modulus_is_rejected(sr_modulus):
    reject("sr_modulus must be a positive int", sr_modulus=sr_modulus)


@pytest.mark.parametrize("speaker_cap", [0, -1, True, 1.5])
def test_non_positive_speaker_cap_is_rejected(speaker_cap):
    reject(
        "speaker_cap must be None or a positive int",
        speaker_cap=speaker_cap,
    )


@pytest.mark.parametrize(
    "budget",
    [
        SolverBudget(max_states=0),
        SolverBudget(max_transitions=-1),
        SolverBudget(max_materialized=0),
    ],
)
def test_non_positive_budget_is_rejected(budget):
    reject("must be a positive int", budget=budget)


@pytest.mark.parametrize(
    "reference",
    [Fraction(-1), Fraction(-1, 2), float("inf"), float("nan")],
)
def test_non_finite_or_negative_references_are_rejected(reference):
    reject(
        "reference must be a non-negative Fraction",
        leaves=(leaf(("zh", "a", None, 16000), reference=reference),),
    )


def test_malformed_leaf_keys_are_rejected():
    reject(
        "batching_sr must be a positive int",
        leaves=(leaf(("zh", "a", None, 0)),),
    )
    reject(
        "batching_sr must be a positive int",
        leaves=(leaf(("zh", "a", None, True)),),
    )
    reject(
        "language must be a non-empty string",
        leaves=(leaf(("", "a", None, 16000)),),
    )
    reject(
        "dataset_id must be a non-empty string",
        leaves=(leaf(("zh", "", None, 16000)),),
    )
    reject(
        "canonical",
        leaves=(leaf(("zh", "a", "s1", 16000)),),
    )
    reject(
        "must match key",
        leaves=(leaf(("zh", "a", ("other", "s1"), 16000)),),
    )
    reject(
        "must be a SolverLeaf",
        leaves=("not-a-leaf",),
    )


pytestmark = [pytest.mark.unit]
