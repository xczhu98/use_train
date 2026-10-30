"""配置边界的严格整数校验共享实现。

统一此前散落于 entry_components / val_audio / dataset_build / dynamic_dataset_base /
simulation_configs / pase_family.common.utils 的近复制校验：拒绝 bool、浮点、
字符串与其他非 ``numbers.Integral`` 类型，报错消息携带完整配置键名。
"""

import numbers


def require_positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    """返回 ``int(value)``；非整数或 ``< minimum`` 时抛 ValueError。

    ``minimum=1`` 为正整数口径；``minimum=0`` 用于非负整数口径。
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an exact positive integer, got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(
            f"{name} must be an exact positive integer >= {minimum}, got {value!r}"
        )
    return value
