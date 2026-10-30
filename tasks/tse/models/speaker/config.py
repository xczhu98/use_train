"""TSE speaker 配置的严格解析工具。

这里集中定义 TSE speaker cue 的布尔值、外部 checkpoint 路径边界，以及训练期句级
speaker head 的 batch 约束判定，避免训练、推理与评估分别按 CWD 或 Python
truthiness 解释同一份 YAML。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[4]


class _Missing:
    pass


_MISSING = _Missing()


def require_bool(value: Any, path: str) -> bool:
    """读取一个显式 bool；拒绝字符串、数值和其它 truthiness 值。"""
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a bool, got {value!r}")
    return value


def config_bool(
    mapping: Mapping[str, Any], key: str, default: bool, path: str
) -> bool:
    """读取 mapping 中的 bool；缺失时使用默认值，显式值必须为 bool。"""
    if key not in mapping:
        return default
    return require_bool(mapping[key], f"{path}.{key}")


def resolve_existing_file(
    value: Any,
    path: str,
    *,
    repo_root: str | Path = _REPO_ROOT,
) -> Path:
    """将仓库根相对路径解析为绝对路径，并要求其为已存在文件。"""
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a non-empty string, got {value!r}")
    if not value or value != value.strip():
        raise ValueError(
            f"{path} must be a non-empty string without leading/trailing whitespace"
        )
    try:
        resolved = Path(value).expanduser()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{path} is not a valid path: {value!r}") from exc
    if not resolved.is_absolute():
        resolved = Path(repo_root) / resolved
    if not resolved.is_file():
        raise ValueError(
            f"{path} must point to an existing file, got {value!r} "
            f"(resolved to {resolved})"
        )
    return resolved.resolve()


def resolve_external_speaker_checkpoint(
    model_args: Mapping[str, Any],
    *,
    repo_root: str | Path = _REPO_ROOT,
) -> Path | None:
    """解析非 joint + ``spkemb`` 模式所需的外部 speaker checkpoint。

    joint 模式由 TSE 模型内部持有 speaker encoder，不返回外部 artifact；未启用
    ``spkemb`` 时也不需要句级外部 encoder。只有真正需要预提取 embedding 的配置
    才会强制要求 checkpoint 文件存在。
    """
    if not isinstance(model_args, Mapping):
        raise TypeError("model.args must be a mapping")
    joint_training = config_bool(model_args, "joint_training", True, "model.args")
    features = model_args.get("features")
    if features is None:
        features = {}
    if not isinstance(features, Mapping):
        raise TypeError("model.args.features must be a mapping")
    spkemb = features.get("spkemb")
    if spkemb is None:
        spkemb = {}
    if not isinstance(spkemb, Mapping):
        raise TypeError("model.args.features.spkemb must be a mapping")
    spkemb_enabled = config_bool(
        spkemb, "enabled", False, "model.args.features.spkemb"
    )
    if joint_training or not spkemb_enabled:
        return None

    speaker_model = model_args.get("speaker_model")
    if speaker_model is None:
        speaker_model = {}
    if not isinstance(speaker_model, Mapping):
        raise TypeError("model.args.speaker_model must be a mapping")
    speaker_encoder = speaker_model.get("speaker_encoder")
    if speaker_encoder is None:
        speaker_encoder = {}
    if not isinstance(speaker_encoder, Mapping):
        raise TypeError("model.args.speaker_model.speaker_encoder must be a mapping")
    path_key = "model.args.speaker_model.speaker_encoder.pretrained"
    if "pretrained" not in speaker_encoder:
        raise ValueError(
            f"{path_key} must be a non-empty string without leading/trailing whitespace"
        )
    return resolve_existing_file(
        speaker_encoder["pretrained"], path_key, repo_root=repo_root
    )


def resolved_model_bool(
    model_args: Mapping[str, Any], key: str, default: bool
) -> bool:
    """读取 ``model.args.<key>`` 的严格 bool。"""
    return config_bool(model_args, key, default, "model.args")


def resolve_joint_sentence_batch_encoder(model_args: Mapping[str, Any]) -> str | None:
    """返回 joint 训练中句级 speaker head 要求 per-forward batch >= 2 的 encoder 名。

    只有下列条件同时成立才返回模型名，否则返回 ``None``：

    - ``joint_training=true``：speaker encoder 由 TSE 模型持有并进入 ``model.train()``；
      ``joint_training=false`` 走 trainer 外部 encoder，它恒为 ``freeze=True`` + eval；
    - ``features.spkemb.enabled=true``：只有 ``spkemb`` cue 调用句级
      ``SpeakerEncoder.forward``；
    - encoder 未冻结：``speaker_encoder.freeze`` 显式为 false，或缺省且未提供
      ``pretrained``（``SpeakerEncoder`` 的默认语义是"有预训练则冻结"）。冻结后
      ``spk_model`` 恒为 eval，句级 BatchNorm 使用 running stats，batch=1 合法；
    - 该模型的句级 head 含 ``nn.BatchNorm1d``
      （``train_sentence_batch_requires_two``）。

    ``context`` 与 ``tfmap``（spec / embedding）只走帧级 ``forward_frame_level``，
    不在此约束内。
    """
    if not isinstance(model_args, Mapping):
        raise TypeError("model.args must be a mapping")
    if not config_bool(model_args, "joint_training", True, "model.args"):
        return None

    features = model_args.get("features")
    if features is None:
        features = {}
    if not isinstance(features, Mapping):
        raise TypeError("model.args.features must be a mapping")
    spkemb = features.get("spkemb")
    if spkemb is None:
        spkemb = {}
    if not isinstance(spkemb, Mapping):
        raise TypeError("model.args.features.spkemb must be a mapping")

    if not config_bool(spkemb, "enabled", False, "model.args.features.spkemb"):
        return None

    speaker_model = model_args.get("speaker_model")
    if speaker_model is None:
        speaker_model = {}
    if not isinstance(speaker_model, Mapping):
        raise TypeError("model.args.speaker_model must be a mapping")
    speaker_encoder = speaker_model.get("speaker_encoder")
    if speaker_encoder is None:
        speaker_encoder = {}
    if not isinstance(speaker_encoder, Mapping):
        raise TypeError("model.args.speaker_model.speaker_encoder must be a mapping")

    model_name = speaker_encoder.get("model", "ECAPA_TDNN_GLOB_c512")
    if not isinstance(model_name, str) or not model_name:
        raise TypeError(
            "model.args.speaker_model.speaker_encoder.model must be a non-empty string, "
            f"got {model_name!r}"
        )
    spk_args = speaker_encoder.get("spk_args")
    if spk_args is None:
        spk_args = {}
    if not isinstance(spk_args, Mapping):
        raise TypeError(
            "model.args.speaker_model.speaker_encoder.spk_args must be a mapping"
        )

    freeze = speaker_encoder.get("freeze")
    if freeze is None:
        # 与 SpeakerEncoder 一致：freeze 缺省表示"有预训练则冻结"。
        frozen = speaker_encoder.get("pretrained") is not None
    else:
        frozen = require_bool(
            freeze, "model.args.speaker_model.speaker_encoder.freeze"
        )
    if frozen:
        return None

    from tasks.tse.models.speaker.encoder import train_sentence_batch_requires_two

    return (
        model_name if train_sentence_batch_requires_two(model_name, spk_args) else None
    )


__all__ = [
    "config_bool",
    "require_bool",
    "resolve_existing_file",
    "resolve_external_speaker_checkpoint",
    "resolve_joint_sentence_batch_encoder",
    "resolved_model_bool",
]
