"""``simulation_by_layer`` 的 producer↔consumer 合同测试。

生产端（SE/AEC dynamic dataset）把 ``final_peak_normalize`` / ``realized`` 两个聚合
事实与 A–F 层放在同一 dict；消费者（verify 工具等）通过
``core.data.simulation_layer_view`` 统一拆分。本文件锁定 canonical 键集合与
提升/回落语义，producer 或 consumer 单侧漂移时先在这里变红。
"""

from __future__ import annotations

import pytest

from core.data.simulation_layer_view import (
    AGGREGATE_FACT_KEYS,
    LAYER_KEYS,
    canonical_simulation_view,
    resolve_aggregate_facts,
    split_simulation_by_layer,
)

pytestmark = [pytest.mark.unit]


def _producer_by_layer() -> dict:
    """镜像生产端形状（tasks/se|aec/data/dynamic_dataset.py 的 by_layer 构造）。"""
    return {
        "A": {"speech_uid": "utt"},
        "B": {"branch": "prob_target_noise", "force_branch": "prob_target_noise"},
        "C": {"need_additive_noise": True},
        "D": {"stage_order": ("D-normalize", "D-guard")},
        "E": {},
        "F": {},
        "final_peak_normalize": {"applied": False, "scale": 1.0},
        "realized": {"measure_point": "post_layer_f"},
    }


def test_canonical_key_sets_are_layers_plus_aggregate_facts():
    assert set(LAYER_KEYS) == {"A", "B", "C", "D", "E", "F"}
    assert set(AGGREGATE_FACT_KEYS) == {"final_peak_normalize", "realized"}
    assert set(_producer_by_layer()) == set(LAYER_KEYS) | set(AGGREGATE_FACT_KEYS)


def test_split_promotes_aggregate_facts_and_rejects_unknown_keys():
    layers, aggregates = split_simulation_by_layer(_producer_by_layer())
    assert set(layers) == set(LAYER_KEYS)
    assert set(aggregates) == set(AGGREGATE_FACT_KEYS)
    with pytest.raises(RuntimeError, match="must contain only Layers A-F"):
        split_simulation_by_layer({**_producer_by_layer(), "G": {}})
    with pytest.raises(RuntimeError, match="must be a mapping"):
        split_simulation_by_layer(None)  # type: ignore[arg-type]


def test_resolve_prefers_nested_and_falls_back_to_top_level():
    nested = {"realized": {"measure_point": "nested"}}
    resolved = resolve_aggregate_facts(nested, {"realized": {"measure_point": "top"}})
    assert resolved["realized"]["measure_point"] == "nested"

    fallback = resolve_aggregate_facts({}, {"realized": {"measure_point": "top"}})
    assert fallback["realized"]["measure_point"] == "top"

    with pytest.raises(RuntimeError, match="must be a mapping"):
        resolve_aggregate_facts({}, {"realized": "not-a-mapping"})


def test_canonical_view_promotes_facts_to_top_level():
    stems = {"simulation_by_layer": _producer_by_layer(), "layer_d_noisy": "x"}
    layers, aggregates, canonical = canonical_simulation_view(stems)

    assert set(layers) == set(LAYER_KEYS)
    assert canonical["realized"] == aggregates["realized"]
    assert canonical["final_peak_normalize"] == aggregates["final_peak_normalize"]
    assert canonical["layer_d_noisy"] == "x"
    # 原始 by_layer 保留在副本中，consumer 必须使用返回的层视图做 A–F 校验
    assert canonical["simulation_by_layer"] is stems["simulation_by_layer"]
