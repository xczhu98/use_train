"""Spatial scene executor seam。

``SpatialSceneRenderer`` 负责具体的 RIR/trajectory/waveform 实现；本模块只提供
一个稳定、类型化的执行边界，供任务侧提交 ``SpatialSceneRequest``。这样任务策略
可以在不依赖 renderer 私有实现的情况下替换或观测执行器，同时保持现有渲染顺序、
随机数流和输出对象完全不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.data.simulation.spatial_scene import (
    SpatialSceneRenderer,
    SpatialSceneRequest,
    SpatialSceneResult,
)


class SpatialSceneRendererLike(Protocol):
    """SpatialSceneExecutor 所需的最小 renderer 合同。"""

    def render(self, request: SpatialSceneRequest) -> SpatialSceneResult:
        """根据 typed request 生成统一的 spatial scene result。"""


@dataclass(frozen=True, slots=True)
class SpatialSceneExecutor:
    """提交 ``SpatialSceneRequest`` 的稳定执行 seam。

    默认使用 ``SpatialSceneRenderer``。注入 renderer 只用于测试或受控扩展；executor
    不修改 request，不复制数组，也不额外申请或消费随机数。
    """

    renderer: SpatialSceneRendererLike = field(default_factory=SpatialSceneRenderer)

    def execute(self, request: SpatialSceneRequest) -> SpatialSceneResult:
        """执行一个 spatial scene request，并验证 renderer 返回类型。"""
        if not isinstance(request, SpatialSceneRequest):
            raise TypeError("execute expects a SpatialSceneRequest")
        result = self.renderer.render(request)
        if not isinstance(result, SpatialSceneResult):
            raise TypeError("spatial scene renderer must return SpatialSceneResult")
        return result


__all__ = ["SpatialSceneExecutor", "SpatialSceneRendererLike"]
