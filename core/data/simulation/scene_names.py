"""四任务 Layer B canonical scene ID 合同。

场景名本身就是配置键和运行时 branch ID，不设名称映射层。这里仅保留各任务允许的
``prob_*`` 集合，供 trace 边界校验。
"""

from __future__ import annotations

SE_SCENE_NAMES = frozenset({
    "prob_target_only", "prob_noise_only", "prob_target_noise",
})

AEC_SCENE_ORDER = (
    "prob_target_only", "prob_noise_only", "prob_echo_only",
    "prob_noise_echo_only", "prob_target_noise", "prob_target_echo",
    "prob_target_noise_echo",
)
AEC_SCENE_NAMES = frozenset(AEC_SCENE_ORDER)
AEC_ECHO_SCENES = frozenset({
    "prob_echo_only", "prob_noise_echo_only", "prob_target_echo", "prob_target_noise_echo",
})

TSE_SCENE_NAMES = frozenset({
    "prob_target_only", "prob_interference_only", "prob_noise_only",
    "prob_interference_noise_only", "prob_target_noise",
    "prob_target_interference", "prob_target_interference_noise",
})

SS_SCENE_NAMES = frozenset({
    "prob_target_only", "prob_target_noise", "prob_noise_only",
})


def validate_scene_name(task: str, scene_name: str) -> str:
    """校验并原样返回任务的 canonical ``prob_*`` scene ID。"""
    tables = {"se": SE_SCENE_NAMES, "aec": AEC_SCENE_NAMES,
              "tse": TSE_SCENE_NAMES, "ss": SS_SCENE_NAMES}
    try:
        allowed = tables[task]
    except KeyError as exc:
        raise ValueError(f"unsupported simulation task: {task!r}") from exc
    if scene_name not in allowed:
        raise ValueError(f"unsupported {task} Layer B scene: {scene_name!r}")
    return scene_name


__all__ = [
    "SE_SCENE_NAMES", "AEC_SCENE_NAMES", "TSE_SCENE_NAMES", "SS_SCENE_NAMES",
    "validate_scene_name",
    "AEC_SCENE_ORDER", "AEC_ECHO_SCENES",
]
