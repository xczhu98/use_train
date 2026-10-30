"""动态仿真的 speech-only 资源上下文。

``SSLDynamicMixingDataset`` 复用 speech Layer A、时间画布、几何配置和 spatial
RIR provider；若配置显式提供 noise/wind 清单，背景链仍需读取这些资源。它不应
因为复用历史 Dataset 基类而初始化通用 AcousticRenderer/ScenarioSampler 或 music
路由。本模块提供一个显式资源上下文：复用基类已经稳定的 manifest、长度、缓存和
epoch 生命周期，同时通过 ``speech_only`` profile 关闭与 SSL 无关的通用执行器。

该上下文不是新的任务 Dataset，也不改变其他任务的 ``BaseDynamicMixingDataset``
默认 ``full`` profile。调用方应通过明确的代理方法消费它，避免把 context 的
私有状态扩散到任务合同。
"""

from __future__ import annotations

from typing import Any

from core.data.dynamic_dataset_base import BaseDynamicMixingDataset


class DynamicSpeechResourceContext(BaseDynamicMixingDataset):
    """为 feature-domain 动态任务提供 speech-only 共享资源。

    ``simulation_task`` 只用于配置合同和 ``project_simulation_view`` 的任务
    选择；SSL 自己的 task-specific 配置校验在 context 构造完成后执行，避免
    context 反向依赖任务 Dataset。
    """

    _resource_mode = "speech_only"

    def __init__(self, *args: Any, simulation_task: str = "ssl", **kwargs: Any) -> None:
        if not isinstance(simulation_task, str) or not simulation_task.strip():
            raise ValueError("simulation_task must be a non-empty string")
        self._simulation_task = simulation_task.strip()
        super().__init__(*args, **kwargs)

    def _validate_simulation_cfg(self, cfg) -> None:
        """上下文只负责共享合同；任务特有门禁由拥有者执行。"""

    def prepare_clean_speech_result(self, dataset, *args: Any, **kwargs: Any):
        """把 Layer A 执行显式绑定到拥有者 Dataset。"""

        return self.speech_preparation.prepare_result(dataset, *args, **kwargs)


__all__ = ["DynamicSpeechResourceContext"]
