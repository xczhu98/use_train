"""任务 batch 的公共设备搬运合同。

任务 collate 仍然保留各自的字段和校验语义；本模块只定义训练骨架需要的
最小生命周期接口，避免 ``MetricsCapableTrainer`` 通过 dataclass 反射猜测
哪些字段需要搬运。

五任务 collate 均返回实现 ``BatchContract`` 的结构化对象，并调用
``validate()``；DataLoader 通过 ``pin_memory()`` 锁页，公共训练循环通过
``move_batch_to_device`` 搬运。任务保留各自字段名，锁页和搬运后必须保留
对象类型、元数据与 ``batch_size``。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class BatchContract(Protocol):
    """训练侧结构化 batch 的最小公共接口。

    ``batch_size`` 必须由 batch 自己根据 canonical 主轴返回，避免公共 trainer
    根据 ``clean``/``mixture`` 等任务字段名进行启发式推断。``to`` 应返回设备
    搬运后的 batch（可为新对象），并保留非 tensor 元数据。
    """

    @property
    def batch_size(self) -> int:
        """返回 canonical 主样本数。"""

    def validate(self) -> None:
        """校验任务定义的字段、形状和元数据合同。"""

    def pin_memory(self) -> "BatchContract":
        """返回锁页后的 batch，覆盖全部 tensor 字段。"""

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "BatchContract":
        """把 batch 内的 tensor 搬到目标设备。"""


def is_batch_contract(value: object) -> bool:
    """判断对象是否实现了公共 batch 合同。

    使用显式属性检查而不是依赖所有任务 import，保证 ``core.trainer`` 不需要
    反向依赖任何 task 包。
    """

    return isinstance(value, BatchContract)


def move_batch_to_device(
    batch: Any,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> Any:
    """按 batch 合同搬运 tensor；非合同对象原样返回。

    dict 仍由调用方处理：GAN 等特殊路径使用 dict 作为显式合同，不能让本
    协议把 dict 的字段语义隐式扩展为公共 batch API。合同实现可以原地搬运并
    返回自身，因此调用方必须先用 ``is_batch_contract`` 判断是否命中。
    """

    if is_batch_contract(batch):
        expected_size = batch_size_from_contract(batch)
        moved = batch.to(device=device, non_blocking=non_blocking)
        if not is_batch_contract(moved):
            raise TypeError(
                f"{type(batch).__name__}.to() must return a BatchContract, "
                f"got {type(moved).__name__}"
            )
        actual_size = batch_size_from_contract(moved)
        if actual_size != expected_size:
            raise ValueError(
                f"{type(batch).__name__}.to() changed batch_size from "
                f"{expected_size} to {actual_size}"
            )
        return moved
    return batch


def batch_size_from_contract(batch: Any) -> int | None:
    """读取结构化 batch 的主样本数；非合同对象返回 ``None``。"""

    if not is_batch_contract(batch):
        return None
    size = batch.batch_size
    if type(size) is not int or size < 1:
        raise ValueError(
            f"batch contract {type(batch).__name__}.batch_size must be a positive int, "
            f"got {size!r}"
        )
    return size


def move_mapping_tensors(
    batch: Mapping[str, Any],
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> dict[str, Any]:
    """搬运显式 dict batch 的 tensor 值。

    该 helper 保持旧 GAN/dict 路径的可变 dict 语义，并且只处理顶层 tensor，
    不递归猜测嵌套字段；需要嵌套搬运的任务应实现 ``BatchContract.to``。
    """

    # 旧 MetricsCapableTrainer 路径原地更新 dict；保留这一点，避免调用方在
    # batch 之外持有同一 dict 时观察到新的对象身份。
    out = batch if isinstance(batch, dict) else dict(batch)
    for key, value in out.items():
        if torch.is_tensor(value):
            out[key] = value.to(device=device, non_blocking=non_blocking)
    return out


__all__ = [
    "BatchContract",
    "batch_size_from_contract",
    "is_batch_contract",
    "move_batch_to_device",
    "move_mapping_tensors",
]
