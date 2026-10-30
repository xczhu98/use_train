"""统一解析动态仿真的 SNR/SIR/SER/ERL reference。

这里的函数是纯解析器：只读取调用方已经渲染好的 waveform，不执行混音、
缩放或 RIR。返回值是可直接交给 ``json.dumps`` 的 Python 字典。

``None`` 严格表示该场景不适用该 reference；当场景声明该 reference 应该
存在、但输入缺失、形状不一致、含非有限值或能量为零时，抛出
``InvalidReferenceError``，避免把数据错误伪装成“不适用”。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from core.data.simulation.scene_spec import SceneSpec


class InvalidReferenceError(ValueError):
    """reference 按场景应该存在，但输入不满足解析合同。"""


def resolve_snr_reference(
    scene: SceneSpec,
    rendered_sources: Any,
    latent_references: Any,
) -> dict[str, Any] | None:
    """解析背景 SNR 的能量 reference。

    有实际 speech source 时优先聚合 ``rendered_sources``；speech-free 场景
    使用 ``latent_references``，例如 TSE/SS 纯背景场景中的 latent clean
    speech。没有 background 时返回 ``None``。实际渲染源和 latent reference
    的 measure point 分开记录，避免两者被误认为同一种物理信号。
    """
    _require_scene(scene)
    if not scene.background_present:
        return None

    if scene.source_count > 0:
        if _collection_is_empty(rendered_sources):
            raise InvalidReferenceError(
                "background scene with speech sources requires rendered_sources"
            )
        waveform = _aggregate_collection(rendered_sources, "rendered_sources")
        return _fact("rendered_sources", "post_rir_pre_background", waveform)

    if not _collection_is_empty(rendered_sources):
        raise InvalidReferenceError(
            "speech-free scene must not provide rendered speech sources"
        )
    if _collection_is_empty(latent_references):
        raise InvalidReferenceError(
            "speech-free background scene requires latent_references"
        )
    waveform = _aggregate_collection(latent_references, "latent_reference")
    return _fact("latent_reference", "latent_reference", waveform)


def resolve_sir_reference(
    scene: SceneSpec,
    target_full_rir: Any,
    interference_full_rirs: Any,
) -> dict[str, Any] | None:
    """解析目标 speech 与 interference speech 的 full-RIR SIR reference。

    ``interference_present`` 为 false 时返回 ``None``。当场景包含
    interference 时，``target_full_rir`` 可以是实际 target 或 latent target
    的 full-RIR track，但必须显式提供；``interference_full_rirs`` 会聚合为
    denominator reference。
    """
    _require_scene(scene)
    if not scene.interference_present:
        return None
    target = _single_waveform(target_full_rir, "target_full_rir")
    if _collection_is_empty(interference_full_rirs):
        raise InvalidReferenceError(
            "interference scene requires at least one interference_full_rir"
        )
    interference = _aggregate_collection(
        interference_full_rirs, "interference_full_rirs"
    )
    return {
        "target": _fact("target", "post_rir_pre_background", target),
        "interference": _fact("interference", "post_rir_pre_background", interference),
    }


def resolve_ser_reference(nearend_image: Any, echo_image: Any) -> dict[str, Any] | None:
    """解析 AEC near-end/echo 的 SER reference。

    两个输入都为 ``None`` 表示当前分支没有 SER 关系，返回 ``None``；只
    缺少一侧属于无效输入并抛错。AEC 的 echo-only 分支应传入
    ``(None, echo_image)``，因为 near-end 缺席而不是一个非法的零 waveform。
    """
    if nearend_image is None and echo_image is None:
        return None
    if nearend_image is None or echo_image is None:
        raise InvalidReferenceError("SER requires both nearend_image and echo_image")
    nearend = _single_waveform(nearend_image, "nearend_image")
    echo = _single_waveform(echo_image, "echo_image")
    _require_same_shape(nearend, echo, "nearend_image", "echo_image")
    return {
        "nearend": _fact("nearend", "post_rir_pre_echo_mix", nearend),
        "echo": _fact("echo", "post_rir_pre_echo_mix", echo),
    }


def resolve_erl_reference(farend: Any, echo: Any) -> dict[str, Any] | None:
    """解析 AEC far-end/reference 与 echo 的 ERL 两侧 reference。"""
    if farend is None and echo is None:
        return None
    if farend is None or echo is None:
        raise InvalidReferenceError("ERL requires both farend and echo")
    far = _single_waveform(farend, "farend")
    echo_waveform = _single_waveform(echo, "echo")
    _require_same_shape(far, echo_waveform, "farend", "echo")
    return {
        "farend": _fact("farend", "post_rir_pre_mic_mix", far),
        "echo": _fact("echo", "post_rir_pre_mic_mix", echo_waveform),
    }


def _require_scene(scene: SceneSpec) -> None:
    if not isinstance(scene, SceneSpec):
        raise TypeError("scene must be a SceneSpec")


def _collection_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, Mapping):
        return len(value) == 0
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


def _single_waveform(value: Any, role: str) -> np.ndarray:
    if value is None:
        raise InvalidReferenceError(f"{role} is missing")
    array = _coerce_waveform(value, role)
    _validate_energy(array, role)
    return array


def _aggregate_collection(value: Any, role: str) -> np.ndarray:
    items = _collection_items(value, role)
    if not items:
        raise InvalidReferenceError(f"{role} is empty")
    arrays = [_coerce_waveform(item, f"{role}[{index}]") for index, item in enumerate(items)]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays[1:]):
        raise InvalidReferenceError(f"{role} waveforms must share shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        aggregate = np.sum(np.stack(arrays, axis=0), axis=0, dtype=np.float64)
    _validate_energy(aggregate, role)
    return aggregate


def _collection_items(value: Any, role: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    raise InvalidReferenceError(f"{role} must be a waveform, sequence, or mapping")


def _coerce_waveform(value: Any, role: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:  # numpy conversion errors need a stable public type
        raise InvalidReferenceError(f"{role} cannot be converted to an array") from exc
    if array.ndim == 0 or array.size == 0:
        raise InvalidReferenceError(f"{role} must be a non-empty waveform")
    if not np.issubdtype(array.dtype, np.number):
        raise InvalidReferenceError(f"{role} must contain numeric samples")
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidReferenceError(f"{role} must contain real-valued samples") from exc
    if not np.isfinite(array).all():
        raise InvalidReferenceError(f"{role} must contain only finite samples")
    return array


def _validate_energy(waveform: np.ndarray, role: str) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        energy = float(np.sum(np.square(waveform), dtype=np.float64))
    if not np.isfinite(energy):
        raise InvalidReferenceError(f"{role} energy must be finite")
    if energy <= 0.0:
        raise InvalidReferenceError(f"{role} energy must be positive")
    return energy


def _fact(role: str, measure_point: str, waveform: np.ndarray) -> dict[str, Any]:
    return {
        "role": role,
        "measure_point": measure_point,
        "energy": _validate_energy(waveform, role),
    }


def _require_same_shape(
    first: np.ndarray,
    second: np.ndarray,
    first_role: str,
    second_role: str,
) -> None:
    if first.shape != second.shape:
        raise InvalidReferenceError(
            f"{first_role} and {second_role} must share shape, got "
            f"{first.shape} and {second.shape}"
        )


__all__ = [
    "InvalidReferenceError",
    "resolve_snr_reference",
    "resolve_sir_reference",
    "resolve_ser_reference",
    "resolve_erl_reference",
]
