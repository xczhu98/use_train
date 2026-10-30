"""TSE cue frontend 注册表与惰性构造入口。

注册表只保存稳定的模块路径和能力声明，不在任务包导入时加载可选 frontend。
模型配置通过 ``frontend`` 字段选择注册项；省略该字段时由配置解析器填入与
cue kind 同名的 canonical frontend。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class CueFrontendRegistration:
    """一个 cue frontend 的配置边界。"""

    name: str
    cue_kind: str
    module: str
    class_name: str
    representations: frozenset[str]
    features: frozenset[str]


_REGISTRATIONS = {
    "speaker": CueFrontendRegistration(
        name="speaker",
        cue_kind="speaker",
        module="tasks.tse.models.modules.speaker.spk_frontend",
        class_name="SpeakerFrontend",
        representations=frozenset({"waveform", "embedding"}),
        features=frozenset({"listen", "usef", "tfmap", "context", "spkemb", "spex_plus"}),
    ),
    "spatial": CueFrontendRegistration(
        name="spatial",
        cue_kind="spatial",
        module="tasks.tse.models.modules.spatial.spatial_frontend",
        class_name="SpatialFrontend",
        representations=frozenset({"static", "track"}),
        features=frozenset({"ipd", "cdf", "sdf", "delta_stft", "cyc_doaemb", "initstate_emb"}),
    ),
    "visual": CueFrontendRegistration(
        name="visual",
        cue_kind="visual",
        module="tasks.tse.models.modules.visual.visual_frontend",
        class_name="VisualFrontend",
        representations=frozenset({"features", "raw"}),
        features=frozenset({"muse_visual"}),
    ),
    "textual": CueFrontendRegistration(
        name="textual",
        cue_kind="textual",
        module="tasks.tse.models.modules.textual.textual_frontend",
        class_name="TextualFrontend",
        representations=frozenset({"phoneme_ids"}),
        features=frozenset({"dae_kce"}),
    ),
}
FRONTEND_REGISTRY = MappingProxyType(_REGISTRATIONS)


def get_cue_frontend(name: str) -> CueFrontendRegistration:
    """返回注册项，并在配置边界给出稳定错误。"""

    try:
        return FRONTEND_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown TSE cue frontend {name!r}; "
            f"known frontends: {sorted(FRONTEND_REGISTRY)}"
        ) from None


def resolve_cue_frontend(cue_kind: str, frontend: object | None) -> str:
    """解析 cue 的 frontend 名称并校验 cue kind/能力归属。"""

    name = cue_kind if frontend is None else frontend
    if type(name) is not str or not name:
        raise TypeError(f"cues.{cue_kind}.frontend must be a non-empty string")
    registration = get_cue_frontend(name)
    if registration.cue_kind != cue_kind:
        raise ValueError(
            f"cues.{cue_kind}.frontend={name!r} belongs to cue kind "
            f"{registration.cue_kind!r}"
        )
    return name


def validate_cue_frontend_capabilities(
    name: str, *, representation: str, features: tuple[str, ...]
) -> None:
    """校验注册 frontend 声明的 representation/feature 能力。"""

    registration = get_cue_frontend(name)
    if representation not in registration.representations:
        raise ValueError(
            f"frontend {name!r} does not support representation {representation!r}; "
            f"supported: {sorted(registration.representations)}"
        )
    unknown = sorted(set(features) - set(registration.features))
    if unknown:
        raise ValueError(
            f"frontend {name!r} does not support features: {unknown}"
        )


def load_cue_frontend(name: str):
    """惰性加载注册 frontend class。"""

    registration = get_cue_frontend(name)
    module = importlib.import_module(registration.module)
    try:
        return getattr(module, registration.class_name)
    except AttributeError:
        raise RuntimeError(
            f"registered frontend {name!r} has no "
            f"{registration.class_name} in {registration.module}"
        ) from None


def build_cue_frontend(name: str, *args: Any, **kwargs: Any):
    """按注册项构造 frontend；参数合同由具体 frontend 继续严格校验。"""

    return load_cue_frontend(name)(*args, **kwargs)


def iter_cue_frontends() -> tuple[CueFrontendRegistration, ...]:
    """返回稳定排序的注册项，供配置检查和文档工具使用。"""

    return tuple(FRONTEND_REGISTRY[name] for name in sorted(FRONTEND_REGISTRY))


__all__ = [
    "CueFrontendRegistration",
    "FRONTEND_REGISTRY",
    "build_cue_frontend",
    "get_cue_frontend",
    "iter_cue_frontends",
    "load_cue_frontend",
    "resolve_cue_frontend",
    "validate_cue_frontend_capabilities",
]
