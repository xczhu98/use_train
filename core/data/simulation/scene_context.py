"""场景渲染所需的显式运行时上下文与资源端口。

``scene_renderer`` 和 ``background`` 位于通用仿真层，不应直接依赖具体
Dataset 的私有字段。该模块把 Dataset 适配收敛到一个边界对象；渲染器内部
只消费 ``SceneRenderContext``/``AcousticRuntimePort``，旧的 Dataset 调用方式
仍由 ``coerce_scene_render_context`` 保留，用于渐进迁移任务侧调用方。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Protocol, runtime_checkable


@runtime_checkable
class AcousticRuntimePort(Protocol):
    """Layer B/C 与 Layer E 所需的最小声学运行时端口。"""

    @property
    def config(self) -> Any: ...

    @property
    def simulation_view(self) -> Any: ...

    @property
    def strict_load(self) -> bool: ...

    @property
    def export_simulation_stems(self) -> bool: ...

    @property
    def online_rir_provider(self) -> Any: ...

    @property
    def last_wind_source_identity(self) -> str | None: ...

    def acoustic_plan_record(self) -> dict[str, Any] | None: ...

    def mix_acoustic(self, *args, **kwargs) -> Any: ...

    def apply_channel_degradation(self, *args, **kwargs) -> Any: ...

    def get_wind_sample(self, *args, **kwargs) -> Any: ...

    def apply_wind(self, *args, **kwargs) -> Any: ...

    def read_additive_noise(self, *args, **kwargs) -> Any: ...

    def read_additive_noise_with_identity(self, *args, **kwargs) -> Any: ...

    def read_additive_mode(self, *args, **kwargs) -> Any: ...

    def apply_colored_noise(self, *args, **kwargs) -> Any: ...

    def reuse_noise_source_sidecar(self, *args, **kwargs) -> Any: ...

    def get_noise_single_reuse_cache(self) -> Any: ...

    def set_noise_single_reuse_cache(self, value: Any) -> None: ...

    @contextmanager
    def online_rir_source(self, value: Any) -> Iterator[None]: ...

    @contextmanager
    def active_acoustic_plan_record(self, value: dict[str, Any]) -> Iterator[None]: ...


@runtime_checkable
class BackgroundRuntimePort(AcousticRuntimePort, Protocol):
    """背景混合专用端口别名，便于未来独立 Layer C 资源实现。"""


class DatasetAcousticRuntimeAdapter:
    """把现有 Dataset 私有实现适配为显式声学运行时端口。

    适配器是唯一允许触碰旧 Dataset 私有字段的位置。它不改变字段语义，
    也不缓存配置，因此 Dataset 更新其配置或 scene recorder 后仍能保持旧
    的生命周期和 RNG 顺序。
    """

    def __init__(self, owner: Any):
        """绑定调用方明确传入的资源 owner。

        组合型任务应在任务边界显式把资源 context 传入本适配器；这里不再
        探测 owner 的 ``_resource_context``，避免隐式改变 acoustic plan、缓存
        和 provenance 的写入对象。
        """

        if owner is None:
            raise TypeError("owner must be an explicit dataset or resource context")
        self._owner = owner

    @property
    def config(self) -> Any:
        return self._owner.sim_cfg

    @property
    def simulation_view(self) -> Any:
        return getattr(self._owner, "_simulation_view", None)

    @property
    def strict_load(self) -> bool:
        return bool(getattr(self._owner, "strict_load", True))

    @property
    def export_simulation_stems(self) -> bool:
        return bool(getattr(self._owner, "export_simulation_stems", False))

    @property
    def online_rir_provider(self) -> Any:
        return getattr(self._owner, "_online_rir_provider", None)

    @property
    def last_wind_source_identity(self) -> str | None:
        value = getattr(self._owner, "_last_wind_source_identity", None)
        return str(value) if value is not None else None

    def acoustic_plan_record(self) -> dict[str, Any] | None:
        reader = getattr(self._owner, "_acoustic_plan_record", None)
        if callable(reader):
            return reader()
        record = getattr(self._owner, "_active_acoustic_plan_record", None)
        return record if isinstance(record, dict) else None

    def mix_acoustic(self, *args, **kwargs) -> Any:
        return self._owner._mix_acoustic(*args, **kwargs)

    def apply_channel_degradation(self, *args, **kwargs) -> Any:
        return self._owner._apply_channel_degradation_result(*args, **kwargs)

    def get_wind_sample(self, *args, **kwargs) -> Any:
        return self._owner._mix_acoustic_get_wind_sample(*args, **kwargs)

    def apply_wind(self, *args, **kwargs) -> Any:
        return self._owner._mix_acoustic_apply_wind(*args, **kwargs)

    def read_additive_noise(self, *args, **kwargs) -> Any:
        return self._owner._read_one_additive_noise_scp_only(*args, **kwargs)

    def read_additive_noise_with_identity(self, *args, **kwargs) -> Any:
        return self._owner._read_one_additive_noise_scp_only_with_identity(*args, **kwargs)

    def read_additive_mode(self, *args, **kwargs) -> Any:
        return self._owner._mix_acoustic_read_additive(*args, **kwargs)

    def apply_colored_noise(self, *args, **kwargs) -> Any:
        return self._owner._mix_acoustic_colored_noise(*args, **kwargs)

    def reuse_noise_source_sidecar(self, *args, **kwargs) -> Any:
        return self._owner._reuse_noise_source_sidecar(*args, **kwargs)

    def get_noise_single_reuse_cache(self) -> Any:
        return getattr(self._owner, "_noise_single_reuse_cache", None)

    def set_noise_single_reuse_cache(self, value: Any) -> None:
        self._owner._noise_single_reuse_cache = value

    @contextmanager
    def online_rir_source(self, value: Any) -> Iterator[None]:
        missing = object()
        previous = getattr(self._owner, "_online_rir_source", missing)
        self._owner._online_rir_source = value
        try:
            yield
        finally:
            if previous is missing:
                state = getattr(self._owner, "__dict__", None)
                if isinstance(state, dict):
                    state.pop("_online_rir_source", None)
                else:
                    try:
                        delattr(self._owner, "_online_rir_source")
                    except AttributeError:
                        pass
            else:
                self._owner._online_rir_source = previous

    @contextmanager
    def active_acoustic_plan_record(self, value: dict[str, Any]) -> Iterator[None]:
        missing = object()
        previous = getattr(self._owner, "_active_acoustic_plan_record", missing)
        self._owner._active_acoustic_plan_record = value
        try:
            yield
        finally:
            if previous is missing:
                state = getattr(self._owner, "__dict__", None)
                if isinstance(state, dict):
                    state.pop("_active_acoustic_plan_record", None)
                else:
                    try:
                        delattr(self._owner, "_active_acoustic_plan_record")
                    except AttributeError:
                        pass
            else:
                self._owner._active_acoustic_plan_record = previous


@dataclass(frozen=True)
class SceneRenderContext:
    """场景渲染上下文：配置视图与声学资源端口的只读组合。"""

    acoustic: AcousticRuntimePort

    @property
    def config(self) -> Any:
        return self.acoustic.config

    @property
    def simulation_view(self) -> Any:
        return self.acoustic.simulation_view

    @property
    def strict_load(self) -> bool:
        return self.acoustic.strict_load

    # ObservationPostMixRequest 旧版仍使用这些字段名；保留只读投影，迁移后
    # 任务 policy 可直接传入 context，不再把 Dataset 传回通用仿真层。
    @property
    def sim_cfg(self) -> Any:
        return self.config

    @property
    def _simulation_view(self) -> Any:
        return self.simulation_view

    def _apply_channel_degradation_result(self, *args, **kwargs) -> Any:
        return self.acoustic.apply_channel_degradation(*args, **kwargs)

    @classmethod
    def from_runtime(cls, runtime: AcousticRuntimePort) -> "SceneRenderContext":
        if not isinstance(runtime, AcousticRuntimePort):
            raise TypeError("runtime must implement AcousticRuntimePort")
        return cls(runtime)


def coerce_scene_render_context(value: Any) -> SceneRenderContext:
    """把已注入 context、runtime 或旧 Dataset 统一为 context。"""
    if isinstance(value, SceneRenderContext):
        return value
    if isinstance(value, AcousticRuntimePort):
        return SceneRenderContext.from_runtime(value)
    return SceneRenderContext.from_runtime(DatasetAcousticRuntimeAdapter(value))


def coerce_background_runtime(value: Any) -> BackgroundRuntimePort:
    """背景函数的兼容入口，返回显式 ``BackgroundRuntimePort``。"""
    context = coerce_scene_render_context(value)
    runtime = context.acoustic
    if not isinstance(runtime, BackgroundRuntimePort):
        raise TypeError("background runtime does not implement BackgroundRuntimePort")
    return runtime


__all__ = [
    "AcousticRuntimePort",
    "BackgroundRuntimePort",
    "DatasetAcousticRuntimeAdapter",
    "SceneRenderContext",
    "coerce_background_runtime",
    "coerce_scene_render_context",
]
