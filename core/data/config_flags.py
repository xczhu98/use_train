"""Shared config-flag resolution utilities for SE/AEC dataset construction.

Both tasks/se/data/__init__.py and tasks/aec/data/__init__.py used to independently define
identical copies of these functions (dict-layer scanning for strict_load /
placeholder_num_samples flags). Consolidated here since the logic has no
task-specific content.
"""

from __future__ import annotations

from collections.abc import Mapping

from core.data.constants import DEFAULT_PLACEHOLDER_NUM_SAMPLES


def config_layers_for_data_flags(cfg: dict) -> list:
    """从内到外扫描可写 strict_load / placeholder 的配置层。

    支持：仅 data 段、整包 config（data 层优先于根 config）。旧 ``data.data`` 内层
    已退役，出现即 fail-fast。
    """
    layers: list = []
    d = cfg.get("data")
    if isinstance(d, dict):
        if "data" in d:
            raise ValueError(
                "data.data is no longer supported; write strict_load / "
                "placeholder_num_samples directly under data"
            )
        layers.append(d)
    layers.append(cfg)
    return layers


def _resolve_bool_flag(cfg: dict, key: str, *, default: bool) -> bool:
    """按共享层级解析 exact bool，拒绝 YAML 字符串与数值真值。"""
    found = False
    value = None
    for layer in config_layers_for_data_flags(cfg):
        if key in layer:
            value = layer[key]
            found = True
            break
    if not found:
        return default
    if type(value) is not bool:
        raise ValueError(f"data.{key} must be a bool, got {value!r}")
    return value


def resolve_strict_load(cfg: dict) -> bool:
    """默认 True。键可写在 data 或根 config（data 覆盖根 config）。"""
    return _resolve_bool_flag(cfg, "strict_load", default=True)


def require_strict_training_load(cfg: dict) -> None:
    """拒绝训练入口使用会生成静音占位的宽松加载模式。

    ``strict_load=False`` 仍由 Dataset 构造器保留给离线诊断、数据清单排查和
    推理工具；统一训练入口不能消费这类 Dataset，因为动态/离线失败会被转换为
    全零样本，而现有 batch 合同没有逐样本有效性 mask，Trainer 无法从 loss 中
    安全排除它们。该门禁只读取 canonical 的根 ``config`` / ``data`` 层，和
    ``resolve_strict_load`` 使用同一优先级，避免训练与数据构造解析出不同值。
    """
    if resolve_strict_load(cfg) is not True:
        raise ValueError(
            "training requires data.strict_load=True: strict_load=False may emit "
            "silence placeholders without a validity mask and cannot enter loss; "
            "use a diagnostic/inference data path for lenient loading"
        )
    # split 内的 strict_load 是 Dataset builder 的 metadata，不参与公共解析；
    # 显式写入 False 仍必须在训练边界报错，避免用户以为它开启了安全的局部
    # lenient 模式，而实际却得到一个未生效或会污染验证 loss 的配置。
    data_cfg = cfg.get("data") if isinstance(cfg, Mapping) else None
    if isinstance(data_cfg, Mapping):
        for split in ("train", "validation"):
            section = data_cfg.get(split)
            if not isinstance(section, Mapping) or "strict_load" not in section:
                continue
            value = section["strict_load"]
            if type(value) is not bool:
                raise ValueError(
                    f"data.{split}.strict_load must be a bool, got {value!r}"
                )
            if value is False:
                raise ValueError(
                    f"training requires data.{split}.strict_load=True: "
                    "split-level strict_load=False may emit silence placeholders "
                    "without a validity mask"
                )


def resolve_placeholder_num_samples(cfg: dict) -> int:
    found = False
    value = None
    for layer in config_layers_for_data_flags(cfg):
        if "placeholder_num_samples" in layer:
            value = layer["placeholder_num_samples"]
            found = True
            break
    if not found:
        return DEFAULT_PLACEHOLDER_NUM_SAMPLES
    if type(value) is not int or value <= 0:
        raise ValueError(
            "data.placeholder_num_samples must be an exact int > 0, "
            f"got {value!r}"
        )
    return value


def _reject_retired_simulation_cfg_strict(value: object, path: str = "config") -> None:
    """旧 ``simulation_cfg_strict`` 宽容开关已退役，任何配置位置都必须 fail-fast。"""
    if isinstance(value, Mapping):
        if "simulation_cfg_strict" in value:
            raise ValueError(
                f"{path}.simulation_cfg_strict is no longer supported; unknown "
                "simulation_cfg keys always fail fast — remove the key"
            )
        for key, item in value.items():
            _reject_retired_simulation_cfg_strict(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_retired_simulation_cfg_strict(item, f"{path}[{index}]")


def resolve_simulation_cfg_strict(cfg: dict) -> bool:
    """``SimulationConfigs.from_dict(..., strict=True)``：未知键一律立即报错。

    旧的宽容逃生开关 ``data.simulation_cfg_strict`` 已退役：该键出现在任何配置层级
    都会 fail-fast，因此本函数恒返回 True。
    """
    _reject_retired_simulation_cfg_strict(cfg)
    return True
