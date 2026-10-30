"""声明式 spatial/temporal capability 矩阵（单一事实源）。

背景能力在 spatial（多通道）模式下的任务开放清单集中登记在本模块；
capability 门禁（``validate_spatial_background_capability``）、配置期校验
（``tests/test_configs.py``）与文档（``configs/README.md``「空间能力矩阵」）
消费同一张表，避免代码、测试与文档三处漂移。

矩阵规则：

- ``spatial_tasks`` 非空：该 feature 在 spatial 模式下仅对列出的任务开放；
  其余任务命中 ``trigger`` 即 fail-fast（配置期与 runtime 报同一信息），
  要求显式关闭（概率置 0）或走 mono 路径，不做静默降级。
- ``trigger`` 默认语义为「配置值 > 0」；特殊语义（如 ``target_type``）用
  自定义谓词表达。
- 概率为 0 时 ``trigger`` 不命中，capability 行零成本、零随机数消耗。

新增 spatial 背景能力的接入 checklist（三件套，缺一不可）：

1. **target protection**：背景只进入 observation，``target``/``direct``
   监督轨迹与 SNR 标定不变；
2. **facts/audit**：渲染 policy、SNR、类型与 RNG 抽取顺序进入 scene
   facts / acoustic audit，可事后核对；
3. **测试**：本矩阵登记行 + spatial 渲染集成测试（含 target protection
   断言）+ 概率为 0 的 seeded replay 零随机数消耗测试。

多源 relative timeline 使用同一份 capability 事实源；单源 SE/AEC 不因
``SimulationConfigs`` 能解析字段就静默接受该配置。画布长度仍属于 split-level
Dataset lifecycle，不在本矩阵中重复声明。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SUPPORTED_SPATIAL_TASKS: frozenset[str] = frozenset({"se", "aec", "tse", "ss", "ssl"})


@dataclass(frozen=True)
class SpatialFeatureCapability:
    """单条 spatial 背景能力声明。

    ``config_key`` 为 ``SimulationConfigs`` canonical 键；``trigger`` 判定该
    样本配置是否激活此能力；``spatial_tasks`` 为 spatial 模式下开放的任务集合。
    """

    feature: str
    config_key: str
    spatial_tasks: frozenset[str]
    trigger: Callable[[Any], bool] | None = None

    def is_triggered(self, sim_cfg: Any) -> bool:
        if self.trigger is not None:
            return bool(self.trigger(sim_cfg))
        try:
            return float(getattr(sim_cfg, self.config_key, 0.0)) > 0.0
        except (TypeError, ValueError):
            return False


@dataclass(frozen=True)
class TemporalFeatureCapability:
    """单条动态时间布局能力声明。"""

    feature: str
    config_key: str
    dynamic_tasks: frozenset[str]

    def is_triggered(self, sim_cfg: Any) -> bool:
        try:
            return float(getattr(sim_cfg, self.config_key, 0.0)) > 0.0
        except (TypeError, ValueError):
            return False


def _trigger_rts(sim_cfg: Any) -> bool:
    return getattr(sim_cfg, "target_type", "early") == "rts"


# spatial 模式任务开放矩阵。新增能力按 docstring 的接入 checklist 完成后，
# 在对应行追加任务即可放开；不要绕过矩阵在别处手写任务判断。
SPATIAL_FEATURE_CAPABILITIES: tuple[SpatialFeatureCapability, ...] = (
    SpatialFeatureCapability(
        feature="colored_noise",
        config_key="prob_colored_noise",
        # C-3 彩色噪声 spatial 渲染（BackgroundSpec 链）：全部五任务开放。
        spatial_tasks=frozenset({"se", "aec", "tse", "ss", "ssl"}),
    ),
    SpatialFeatureCapability(
        feature="wind_sidechain",
        config_key="prob_wind_noise_sidechain",
        # 非加性 compressor 仅 SSL/DOA 经 NonDecomposableObservationContract 开放。
        spatial_tasks=frozenset({"ssl"}),
    ),
    SpatialFeatureCapability(
        feature="rts_target",
        config_key="target_type",
        spatial_tasks=frozenset({"se"}),
        trigger=_trigger_rts,
    ),
)


TEMPORAL_FEATURE_CAPABILITIES: tuple[TemporalFeatureCapability, ...] = (
    TemporalFeatureCapability(
        feature="relative_timeline",
        config_key="prob_timeline",
        # SSL 已经通过 shared spatial adapter 消费 source timeline；保持与
        # TSE/SS 一样的 scene-level source window 语义。
        dynamic_tasks=frozenset({"tse", "ss", "ssl"}),
    ),
)


def triggered_spatial_features(task: str, sim_cfg: Any) -> list[str]:
    """返回该任务在 spatial 模式下被配置激活、但能力矩阵未开放的功能名。"""
    if not getattr(sim_cfg, "spatial_enabled", False) or task not in SUPPORTED_SPATIAL_TASKS:
        return []
    return [
        capability.feature
        for capability in SPATIAL_FEATURE_CAPABILITIES
        if capability.is_triggered(sim_cfg) and task not in capability.spatial_tasks
    ]


def triggered_temporal_features(task: str, sim_cfg: Any) -> list[str]:
    """返回该任务运行时激活但未开放的动态时间布局能力。"""
    return [
        capability.feature
        for capability in TEMPORAL_FEATURE_CAPABILITIES
        if capability.is_triggered(sim_cfg) and task not in capability.dynamic_tasks
    ]


__all__ = [
    "SPATIAL_FEATURE_CAPABILITIES",
    "SUPPORTED_SPATIAL_TASKS",
    "SpatialFeatureCapability",
    "TEMPORAL_FEATURE_CAPABILITIES",
    "TemporalFeatureCapability",
    "triggered_spatial_features",
    "triggered_temporal_features",
]
