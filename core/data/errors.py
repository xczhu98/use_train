"""动态/离线数据管线共享错误分类。"""

from __future__ import annotations


class RecoverableResourceError(RuntimeError):
    """资源暂时不可读，可在 ``strict_load=False`` 下转为 placeholder。"""


class NonRecoverableSimulationError(RuntimeError):
    """仿真合同、身份、后端或模型输入错误，禁止静默占位。"""


class SimulationContractError(NonRecoverableSimulationError, ValueError):
    """配置、场景、身份和 asset replay 合同错误。"""


def is_recoverable_resource_error(exc: BaseException) -> bool:
    """只将显式资源错误和标准 I/O 错误视为 lenient 可恢复错误。"""
    return isinstance(exc, (RecoverableResourceError, OSError))


__all__ = [
    "RecoverableResourceError",
    "NonRecoverableSimulationError",
    "SimulationContractError",
    "is_recoverable_resource_error",
]
