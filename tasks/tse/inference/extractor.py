"""TSEExtractor — TSE 整句推理接口（与 REAL-TSE-Challenge 比赛 Extractor 同语义）.

语义对齐 wesep-real-tse ``wesep/cli/extractor.py::Extractor``：

- 输入 mix/enroll 波形（文件路径或 PCM），内部重采样到模型采样率（默认 16k）；
- 可选 silero-VAD 裁掉 enroll 首尾静音（``apply_vad``，环境无 silero-vad 时
  自动跳过并 warning，不引入硬依赖）；
- 整句（whole-utterance）推理：通过 ``tasks.tse.model_io`` 与 trainer 共用
  STFT 输入字段、sample-rate guard、dict 谱进谱出模型与 iSTFT 输出对齐；
- 输出峰值归一化到 0.9（比赛默认 ``output_norm=True``）。

与比赛实现的差异（use_train 谱进谱出架构）：

- 模型内部 STFT/iSTFT 已剥离，推理侧自建 :class:`StftIstftModule`（无 trainer）；
- joint_training=True 时 speaker encoder 在模型内部（enroll_wav 直接进模型）；
  joint_training=False（预提取 embedding）时由本类按需懒构建独立
  :class:`SpeakerEncoder` 从 enroll 提 ``spk_embedding``。

模型加载支持两种来源：

1. **use_train 训练 checkpoint**（``.tar``/``.pth``，含 ``config``/``resolved_config``
   与 ``model`` state_dict，见 ``core/trainer/base_trainer.py::_build_state``）；
2. **use_train 布局 model_dir**（目录内含 ``config.yaml`` + ``avg_model.pt``，
   其中 ``config.yaml`` 必须是含 ``model.type`` 与顶层 ``stft`` 块的
   use_train 配置；``config_path`` 显式指定时优先使用该配置）。
   也兼容直接传 ckpt 文件 + ``config_path`` 显式指定配置。注意：wesep 比赛的平铺
   ``config.yaml``（无 ``model.type``/``stft`` 块的原始比赛格式）**暂不支持**。

Usage::

    from tasks.tse.inference import TSEExtractor

    ext = TSEExtractor("exp/tse/.../checkpoints/<run-id>/best_model.tar", device="cuda:0")
    wav = ext.extract_speech("mix.wav", "enroll.wav")   # np.float32, 16k, 峰值 0.9

Author: Xincheng Zhu
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import stat

import numpy as np
import torch
import torchaudio

from core.stft import build_stft_istft
from core.utils.config import load_merged_config
from core.utils.inference_artifacts import fingerprint_file, fingerprint_json
from core.utils.safe_load import safe_torch_load

import tasks.tse.models  # noqa: F401 — 注册 TSE lazy-load map
from tasks.tse.cue_contract import TSECueBatch, TSECueBatches
from tasks.tse.inference.artifacts import (
    load_bound_audio,
    load_immutable_checkpoint,
    speaker_frontend_identity,
    verify_file_identity,
)
from tasks.tse.model_io import (
    _resolve_tse_model_input_contract,
    build_tse_model_inputs,
    decode_tse_model_output,
    resolve_tse_model_input_fields,
)
from tasks.tse.models import register_tse_models
from tasks.tse.models.speaker.config import (
    config_bool,
    require_bool,
    resolve_external_speaker_checkpoint,
)

logger = logging.getLogger(__name__)

# 从 tasks/tse/inference 目录向上三级到仓库根，作为相对路径解析的兜底基准。
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# streaming/ONNX 目前只覆盖 speaker 条件；其余 cue 模态在文件推理里可用，但
# 不进入 streaming/ONNX 发布路径，任何请求都必须显式失败而不是假接。
STREAMING_SUPPORTED_CUE_ROLES = frozenset({"speaker_waveform", "speaker_embedding"})

_ROLE_VALUE_FIELDS: dict[str, frozenset[str]] = {
    "speaker_waveform": frozenset({"enroll_spectra", "enroll_wav", "speaker_waveform"}),
    "speaker_embedding": frozenset({"spk_embedding", "speaker_embedding"}),
    "spatial": frozenset({"spatial"}),
    "visual": frozenset({"visual"}),
    "textual": frozenset({"textual"}),
}
_ROLE_FIELDS: dict[str, frozenset[str]] = {
    "speaker_waveform": frozenset(
        {
            "enroll_spectra",
            "enroll_spectra_lengths",
            "enroll_wav",
            "enroll_lengths",
            "enroll_present",
            "speaker_waveform",
            "speaker_waveform_lengths",
            "speaker_waveform_present",
        }
    ),
    "speaker_embedding": frozenset(
        {
            "spk_embedding",
            "spk_embedding_present",
            "speaker_embedding",
            "speaker_embedding_present",
        }
    ),
    "spatial": frozenset(
        {"spatial", "spatial_present", "spatial_lengths", "spatial_timebase"}
    ),
    "visual": frozenset(
        {"visual", "visual_present", "visual_lengths", "visual_timebase"}
    ),
    "textual": frozenset({"textual", "textual_present", "textual_lengths"}),
}
_VISUAL_RAW_AXES = frozenset({"channel", "height", "width"})


@dataclass(frozen=True)
class TSECueRole:
    """模型 IO 合同声明的一个 cue role。"""

    role: str
    representation: str
    fields: frozenset[str]
    row_rank: int
    required: bool


@dataclass(frozen=True)
class TSECueInput:
    """单条 utterance 的一个已读入 cue；value 含 batch 行轴。"""

    role: str
    representation: str
    value: torch.Tensor
    length: int | None
    timebase: torch.Tensor | None

    def __post_init__(self) -> None:
        if self.role not in _ROLE_VALUE_FIELDS:
            raise ValueError(f"unknown TSE cue role: {self.role!r}")
        if not isinstance(self.value, torch.Tensor) or self.value.ndim < 2:
            raise ValueError(
                f"cue {self.role!r} value must be a tensor with a batch axis"
            )


def _cue_representation(role: str, field: Any) -> str:
    axes = field.axes
    if role == "speaker_waveform":
        return "waveform"
    if role == "speaker_embedding":
        return "embedding"
    if role == "spatial":
        return "track" if "point" in axes else "static"
    if role == "visual":
        return "raw" if _VISUAL_RAW_AXES.intersection(axes) else "features"
    if role == "textual":
        return "phoneme_ids"
    raise ValueError(f"unknown TSE cue role: {role!r}")


def declared_cue_roles(model: Any) -> dict[str, TSECueRole]:
    """解析模型 IO 合同声明的 cue role、representation 与每行维度。"""
    fields = {field.name: field for field in _resolve_tse_model_input_contract(model)}
    declared: dict[str, TSECueRole] = {}
    for role, value_fields in _ROLE_VALUE_FIELDS.items():
        value_field = next(
            (fields[name] for name in sorted(value_fields) if name in fields), None
        )
        if value_field is None:
            continue
        declared[role] = TSECueRole(
            role=role,
            representation=_cue_representation(role, value_field),
            fields=frozenset(name for name in _ROLE_FIELDS[role] if name in fields),
            row_rank=len(value_field.axes) - 1,
            required=bool(value_field.required),
        )
    return declared


def verify_model_checkpoint_paths(model: Any) -> tuple[dict[str, Any], ...]:
    """复核模型构造后全部 ``*_ckpt_path`` 指向已固化内容身份的普通文件。

    waveform-native 前端（如 ``TSE_Modular`` 的 ``speaker_encoder_ckpt_path``）
    在构造时自行加载 checkpoint；这里在构造后重新绑定其内容身份，禁止出现
    未被 artifact 绑定的前端权重。
    """
    identities: list[dict[str, Any]] = []
    for name in sorted(dir(model)):
        if not name.endswith("_ckpt_path"):
            continue
        value = getattr(model, name)
        if value is None:
            continue
        if not isinstance(value, (str, Path)):
            raise TypeError(f"model.{name} must be a path string or None")
        identity = fingerprint_file(value)
        identities.append({"attribute": name, **identity})
    return tuple(identities)


# STFT center=True reflect padding 要求输入长度 > n_fft//2（与 inference.py 口径一致）
def _min_stft_input_len(n_fft: int) -> int:
    return int(n_fft) // 2 + 1


def _positive_sample_rate(value, name: str) -> int:
    """解析正整数采样率，不接受 bool、浮点截断或字符串转换。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _load_external_config(path: str | Path) -> tuple[dict, Path, dict]:
    """按当前 CWD 解析 canonical 配置，并复核 loader 前后的内容身份。"""
    configured = Path(path).expanduser()
    candidate = Path(os.path.abspath(configured))
    current = Path(candidate.anchor)
    try:
        for part in candidate.parts[1:]:
            current /= part
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"模型配置路径不允许符号链接: {configured}")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"模型配置不是普通文件: {configured}") from exc
    if not candidate.is_file() or stat.S_ISLNK(candidate.lstat().st_mode):
        raise FileNotFoundError(f"模型配置不是普通文件: {candidate}")
    before = fingerprint_file(candidate)
    config = load_merged_config(before["path"])
    after = fingerprint_file(before["path"])
    if before != after:
        raise RuntimeError(f"模型配置在加载期间发生变化: {candidate}")
    return config, Path(after["path"]), after


def _require_existing_pretrained(cfg, *, path: str = "model") -> None:
    """递归校验模型配置中的 ``pretrained`` 路径必须存在，缺失即报错。

    推理权重来自 checkpoint 本身；配置里显式声明的 ``pretrained`` 在推理机上
    不存在时必须显式失败，不得静默置 None。需要忽略训练遗留路径的 joint 模式
    由调用方在进入本校验前显式清空内部 speaker encoder 的 ``pretrained``。
    """
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            if key == "pretrained" and isinstance(value, str) and value:
                # 依次检查调用目录和仓库根相对路径。
                candidate = Path(value)
                if candidate.exists() or (
                    not candidate.is_absolute() and (_PROJECT_ROOT / candidate).exists()
                ):
                    continue
                raise FileNotFoundError(
                    f"{path}.pretrained 指向的文件不存在: {value}；"
                    "请提供正确路径，或按 canonical 用法显式写 `pretrained: ~`，"
                    "不会静默忽略该配置。"
                )
            _require_existing_pretrained(value, path=f"{path}.{key}")
    elif isinstance(cfg, list):
        for index, item in enumerate(cfg):
            _require_existing_pretrained(item, path=f"{path}[{index}]")


def _load_model_state_fail_closed(
    model: torch.nn.Module,
    state: Mapping,
    *,
    checkpoint_label: str,
) -> None:
    """在复制任何参数前完整校验模型 state，并执行 strict load。"""
    if not isinstance(state, Mapping):
        raise TypeError(
            f"{checkpoint_label} model state must be a mapping, "
            f"got {type(state).__name__}"
        )
    adapted: dict[str, torch.Tensor] = {}
    duplicates: list[str] = []
    invalid_keys: list[str] = []
    for raw_key, value in state.items():
        if not isinstance(raw_key, str):
            invalid_keys.append(repr(raw_key))
            continue
        key = raw_key[len("module."):] if raw_key.startswith("module.") else raw_key
        if key in adapted:
            duplicates.append(key)
        else:
            adapted[key] = value

    expected = model.state_dict()
    missing = sorted(set(expected) - set(adapted))
    unexpected = sorted(set(adapted) - set(expected))
    mismatches: list[str] = []
    for key in sorted(set(expected) & set(adapted)):
        actual = adapted[key]
        target = expected[key]
        if not isinstance(actual, torch.Tensor):
            mismatches.append(f"{key}: expected Tensor, got {type(actual).__name__}")
        elif actual.shape != target.shape:
            mismatches.append(
                f"{key}: shape {tuple(actual.shape)} != {tuple(target.shape)}"
            )
        elif actual.dtype != target.dtype:
            mismatches.append(f"{key}: dtype {actual.dtype} != {target.dtype}")
    if invalid_keys or duplicates or missing or unexpected or mismatches:
        details = []
        if invalid_keys:
            details.append(f"invalid keys={invalid_keys[:5]}")
        if duplicates:
            details.append(f"duplicate keys={sorted(set(duplicates))[:5]}")
        if missing:
            details.append(f"missing={missing[:5]} ({len(missing)} total)")
        if unexpected:
            details.append(
                f"unexpected={unexpected[:5]} ({len(unexpected)} total)"
            )
        if mismatches:
            details.append(f"tensor mismatches={mismatches[:5]}")
        raise RuntimeError(
            f"{checkpoint_label} is incompatible with the TSE model: "
            + "; ".join(details)
        )
    model.load_state_dict(adapted, strict=True)


def _apply_ema_state_fail_closed(
    model: torch.nn.Module,
    ema_state: Mapping,
    *,
    checkpoint_label: str,
) -> None:
    """完整校验 EMA shadow 后再复制，禁止依赖缺失时静默退回 raw 权重。"""
    if not isinstance(ema_state, Mapping):
        raise TypeError(
            f"{checkpoint_label} EMA state must be a mapping, "
            f"got {type(ema_state).__name__}"
        )
    shadow_params = ema_state.get("shadow_params")
    if not isinstance(shadow_params, (list, tuple)):
        raise RuntimeError(
            f"{checkpoint_label} EMA state must contain shadow_params list/tuple"
        )
    parameters = list(model.parameters())
    if len(shadow_params) != len(parameters):
        raise RuntimeError(
            f"{checkpoint_label} EMA shadow_params count {len(shadow_params)} "
            f"does not match model parameter count {len(parameters)}"
        )
    mismatches: list[str] = []
    for index, (shadow, parameter) in enumerate(zip(shadow_params, parameters)):
        if not isinstance(shadow, torch.Tensor):
            mismatches.append(
                f"[{index}]: expected Tensor, got {type(shadow).__name__}"
            )
        elif shadow.shape != parameter.shape:
            mismatches.append(
                f"[{index}]: shape {tuple(shadow.shape)} != {tuple(parameter.shape)}"
            )
        elif shadow.dtype != parameter.dtype:
            mismatches.append(
                f"[{index}]: dtype {shadow.dtype} != {parameter.dtype}"
            )
    if mismatches:
        raise RuntimeError(
            f"{checkpoint_label} EMA shadow parameter validation failed: "
            + "; ".join(mismatches[:5])
        )
    with torch.no_grad():
        for parameter, shadow in zip(parameters, shadow_params):
            parameter.copy_(shadow.to(device=parameter.device))


class TSEExtractor:
    """TSE 整句推理器（比赛 Extractor 同语义，谱进谱出架构）。

    Args:
        model_dir_or_ckpt: use_train 训练 checkpoint 路径，或 use_train 布局
            model_dir（含 ``config.yaml`` + ``avg_model.pt``；wesep 平铺
            config 格式暂不支持，见模块 docstring）。
        config_path: 显式配置 YAML；给定后优先于 checkpoint 内嵌配置，
            相对路径严格按调用时 CWD 解析；未给定时 model_dir 读取
            ``<dir>/config.yaml``，checkpoint 文件读取内嵌配置。
        device: 推理设备（无 CUDA 时自动回退 cpu）。
        sample_rate: 模型采样率；None 时取 ``data.sample_rate``，缺省 16000。
        apply_vad: 是否用 silero-VAD 裁 enroll 静音（未安装 silero-vad 时
            自动跳过，仅 warning）。
        output_peak: 输出峰值归一目标（比赛默认 0.9；None/<=0 关闭归一化）。
        allow_partial_load: 已废弃；传 True 会立即报错。推理模型始终 strict、
            fail-closed 加载，禁止保留部分随机初始化参数。
    """

    def __init__(
        self,
        model_dir_or_ckpt: str,
        config_path: str | None = None,
        device: str = "cuda:0",
        sample_rate: int | None = None,
        apply_vad: bool = False,
        output_peak: float | None = 0.9,
        allow_partial_load: bool = False,
    ):
        if type(allow_partial_load) is not bool:
            raise TypeError("allow_partial_load must be bool")
        if allow_partial_load:
            raise ValueError(
                "TSEExtractor no longer supports partial checkpoint loading"
            )
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.output_peak = output_peak
        self.apply_vad = apply_vad
        self._vad = None
        self._spk_encoder = None  # 预提取 embedding 模式下懒构建
        self._loaded_external_speaker_checkpoint_identity: dict | None = None
        self._loaded_model_checkpoint_identity: dict | None = None

        src = Path(model_dir_or_ckpt).expanduser()
        ckpt: dict | None = None
        external_config_path: Path | None = None
        config_identity: dict | None = None
        if src.is_dir():
            # 显式配置与 model_dir 配置按各自路径解析。
            config_candidate = (
                Path(config_path).expanduser() if config_path is not None
                else src / "config.yaml"
            )
            ckpt_path = src / "avg_model.pt"
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"model_dir 模式要求 checkpoint 存在: {ckpt_path}")
            cfg, external_config_path, config_identity = _load_external_config(
                config_candidate
            )
        else:
            ckpt_path = src
            if config_path is not None:
                cfg, external_config_path, config_identity = _load_external_config(
                    config_path
                )
            else:
                ckpt, self._loaded_model_checkpoint_identity = (
                    self._load_model_checkpoint(ckpt_path)
                )
                cfg = ckpt.get("resolved_config") or ckpt.get("config")
                if not isinstance(cfg, dict):
                    raise ValueError(
                        f"checkpoint {ckpt_path} 内不含 config/resolved_config，"
                        "请用 config_path 显式指定模型配置"
                    )
        self._config_path = (
            str(external_config_path) if external_config_path is not None else None
        )
        self._config_identity = config_identity
        self._resolved_config_identity = {
            "algorithm": "sha256-json-v1",
            "sha256": fingerprint_json(cfg),
        }

        # ---- 采样率：显式参数 > data.sample_rate > 16000，且须与模型一致 ----
        raw_model_cfg = cfg.get("model", {})
        model_sr = _positive_sample_rate(
            (raw_model_cfg.get("args") or {}).get("sr", 16000), "model.args.sr"
        )
        cfg_sr = (cfg.get("data") or {}).get("sample_rate")
        effective_sample_rate = (
            sample_rate
            if sample_rate is not None
            else (cfg_sr if cfg_sr is not None else 16000)
        )
        self.sample_rate = _positive_sample_rate(
            effective_sample_rate, "effective sample_rate"
        )
        if self.sample_rate != model_sr:
            raise ValueError(
                f"effective sample_rate={self.sample_rate} 与 "
                f"model.args.sr={model_sr} 不一致"
            )

        # ---- 外部 speaker checkpoint 必须在模型构造前 fail-closed 解析 ----
        raw_model_args = raw_model_cfg.get("args") or {}
        if not isinstance(raw_model_args, Mapping):
            raise TypeError("model.args must be a mapping")
        external_speaker_checkpoint = resolve_external_speaker_checkpoint(
            raw_model_args, repo_root=_PROJECT_ROOT
        )

        # ---- 构建模型（stft 参数经 build_model setdefault 注入 model.args）----
        model_cfg = copy.deepcopy(raw_model_cfg)
        if config_bool(raw_model_args, "joint_training", True, "model.args"):
            # 完整主 checkpoint 已包含内部 speaker encoder。推理构造阶段禁止先
            # 读取训练配置遗留的独立 speaker checkpoint，否则该无关资产可能在
            # 主模型权重复制前阻断官方 checkpoint。
            model_args = model_cfg.get("args") or {}
            speaker_model = model_args.get("speaker_model") or {}
            speaker_encoder = speaker_model.get("speaker_encoder") or {}
            if isinstance(speaker_encoder, dict):
                speaker_encoder["pretrained"] = None
        _require_existing_pretrained(model_cfg)
        stft_cfg = cfg.get("stft")
        if not stft_cfg:
            raise ValueError("配置缺少顶层 'stft' 块（谱进谱出模型必需）")
        model_type = model_cfg.get("type")
        if model_type:
            register_tse_models(only=str(model_type))
        from core.models.build import build_model

        self.model = build_model(model_cfg, stft_cfg=stft_cfg).to(self.device).eval()

        # ---- 加载权重，并保留实际 safe loader 消费的 checkpoint 身份 ----
        if ckpt is None:
            ckpt, self._loaded_model_checkpoint_identity = (
                self._load_model_checkpoint(ckpt_path)
            )
        if self._loaded_model_checkpoint_identity is None:
            raise RuntimeError("主模型 checkpoint 已加载但缺少内容身份")
        self._model_checkpoint_path = self._loaded_model_checkpoint_identity["path"]
        sd = (
            ckpt.get("model")
            or ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt
        )
        if sd is ckpt and isinstance(ckpt.get("models"), list) and ckpt["models"]:
            first = ckpt["models"][0]
            sd = first.state_dict() if hasattr(first, "state_dict") else first
        _load_model_state_fail_closed(
            self.model,
            sd,
            checkpoint_label=str(ckpt_path),
        )
        logger.info("TSEExtractor 加载权重: %s", ckpt_path)

        # EMA 权重存在时必须完整应用；任何结构不兼容都禁止退回 raw 权重。
        if "ema" in ckpt:
            _apply_ema_state_fail_closed(
                self.model,
                ckpt["ema"],
                checkpoint_label=str(ckpt_path),
            )
            logger.info("已应用 EMA 权重")

        # ---- 推理侧 STFT/iSTFT（无 trainer，按顶层 stft 块自建）----
        self.stft = build_stft_istft(stft_cfg).to(self.device)
        speaker_frontend_cfg = getattr(getattr(self.model, "spk_ft", None), "config", {})
        self._speaker_model_cfg = speaker_frontend_cfg.get("speaker_model", {})
        raw_speaker_model_cfg = raw_model_args.get("speaker_model") or {}
        if not isinstance(raw_speaker_model_cfg, Mapping):
            raise TypeError("model.args.speaker_model must be a mapping")
        raw_speaker_encoder_cfg = raw_speaker_model_cfg.get("speaker_encoder") or {}
        if not isinstance(raw_speaker_encoder_cfg, Mapping):
            raise TypeError("model.args.speaker_model.speaker_encoder must be a mapping")
        self._spk_pretrained = raw_speaker_encoder_cfg.get("pretrained")
        if self._spk_pretrained is not None and not isinstance(
            self._spk_pretrained, str
        ):
            raise TypeError(
                "model.args.speaker_model.speaker_encoder.pretrained must be a string or None"
            )
        resolved_joint_training = require_bool(
            getattr(self.model, "joint_training", True),
            "model.joint_training",
        )
        resolved_features = getattr(self.model, "features_config", {})
        if not isinstance(resolved_features, Mapping):
            raise TypeError("model.features_config must be a mapping")
        resolved_spkemb = resolved_features.get("spkemb")
        if resolved_spkemb is None:
            resolved_spkemb = {}
        if not isinstance(resolved_spkemb, Mapping):
            raise TypeError("model.features_config.spkemb must be a mapping")
        self._uses_preextracted_spk_embedding = (
            not resolved_joint_training
            and config_bool(
                resolved_spkemb,
                "enabled",
                False,
                "model.features_config.spkemb",
            )
        )
        self._external_speaker_encoder_pretrained = (
            str(external_speaker_checkpoint)
            if self._uses_preextracted_spk_embedding
            and external_speaker_checkpoint is not None
            else None
        )
        if self._uses_preextracted_spk_embedding:
            if self._external_speaker_encoder_pretrained is None:
                raise ValueError(
                    "非 joint spkemb 模式缺少已解析的外部 SpeakerEncoder checkpoint"
                )
        # STFT reflect padding 的最小输入长度。
        self._min_enroll_input_len = _min_stft_input_len(self.stft.n_fft)

        # ---- speaker 前端身份与模型自持 checkpoint 的构造后复核 ----
        self._speaker_frontend = speaker_frontend_identity(cfg, repo_root=_PROJECT_ROOT)
        self._model_checkpoint_paths = verify_model_checkpoint_paths(self.model)
        self._verify_frontend_checkpoint_identity()

    @property
    def config_path(self) -> str | None:
        """返回本次实际加载的 canonical 外部配置路径；内嵌配置返回 None。"""
        return self._config_path

    @property
    def loaded_config_identity(self) -> dict | None:
        """返回 loader 前后复核一致的入口外部配置内容身份。"""
        return dict(self._config_identity) if self._config_identity is not None else None

    @property
    def resolved_config_identity(self) -> dict:
        """返回模型实际消费的合并配置稳定身份。"""
        return dict(self._resolved_config_identity)

    @property
    def model_checkpoint_path(self) -> str:
        """返回实际 safe loader 消费的主模型 checkpoint 绝对路径。"""
        return self._model_checkpoint_path

    @property
    def loaded_model_checkpoint_identity(self) -> dict:
        """返回 loader 前后复核一致的主模型 checkpoint 内容身份。"""
        return dict(self._loaded_model_checkpoint_identity)

    @property
    def speaker_encoder_pretrained(self) -> str | None:
        """返回配置中的 SpeakerEncoder 路径，供 spksim 的 ECAPA 后备使用。"""
        return self._spk_pretrained

    @property
    def external_speaker_encoder_pretrained(self) -> str | None:
        """返回非 joint ``spkemb`` 实际使用的绝对 checkpoint 路径。"""
        return self._external_speaker_encoder_pretrained

    @property
    def speaker_frontend(self) -> dict | None:
        """返回配置派生的 speaker 前端身份（checkpoint/fbank/preprocess）。"""
        return dict(self._speaker_frontend) if self._speaker_frontend is not None else None

    @property
    def model_checkpoint_paths(self) -> tuple[dict, ...]:
        """返回模型构造后复核过的全部 ``*_ckpt_path`` 内容身份。"""
        return tuple(dict(identity) for identity in self._model_checkpoint_paths)

    def declared_cue_roles(self) -> dict[str, TSECueRole]:
        """返回模型 IO 合同声明的 cue role；未声明的 role 一律不可请求。"""
        return declared_cue_roles(self.model)

    def _verify_frontend_checkpoint_identity(self) -> None:
        """模型自持 checkpoint 必须与配置派生身份一致，否则 fail-closed。"""
        frontend = self._speaker_frontend
        declared = None if frontend is None else frontend.get("checkpoint")
        for identity in self._model_checkpoint_paths:
            if declared is None:
                raise RuntimeError(
                    f"模型 {identity['attribute']} 消费了未被配置绑定的 checkpoint: "
                    f"{identity['path']}"
                )
            if (
                identity["path"] != declared["path"]
                or identity["sha256"] != declared["sha256"]
            ):
                raise RuntimeError(
                    f"模型 {identity['attribute']} 与配置声明的 speaker 前端 "
                    f"checkpoint 不一致: {identity['path']}"
                )

    def prepare_external_speaker_encoder(self) -> dict | None:
        """构造实际 encoder，并返回 loader 前后复核一致的 checkpoint 身份。"""
        if not self._uses_preextracted_spk_embedding:
            return None
        self._get_spk_encoder()
        identity = self._loaded_external_speaker_checkpoint_identity
        if identity is None:
            raise RuntimeError("外部 SpeakerEncoder 已构造但缺少已加载 checkpoint 身份")
        return dict(identity)

    def loaded_artifact_snapshot(self) -> dict:
        """返回本次实际 loader 已消费的 TSE 推理依赖身份。"""
        speaker_identity = self.prepare_external_speaker_encoder()
        return {
            "model_checkpoint": self.model_checkpoint_path,
            "model_checkpoint_identity": self.loaded_model_checkpoint_identity,
            "config_path": self.config_path,
            "config_identity": self.loaded_config_identity,
            "resolved_config_identity": self.resolved_config_identity,
            "speaker_encoder_pretrained": self.speaker_encoder_pretrained,
            "speaker_checkpoint": self.external_speaker_encoder_pretrained,
            "speaker_checkpoint_identity": speaker_identity,
            "speaker_frontend": self.speaker_frontend,
            "model_checkpoint_paths": self.model_checkpoint_paths,
        }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _torch_load(path) -> dict:
        """以 safe_torch_load 安全模式加载（weights_only，fail-closed）。

        含自定义类等不可安全反序列化内容的 ckpt 直接报错，需要
        ``allow_unsafe=True`` 显式信任来源；不静默回退。
        """
        return safe_torch_load(path, map_location="cpu")

    @classmethod
    def _load_model_checkpoint(cls, path) -> tuple[dict, dict]:
        """从 immutable bytes 加载主模型 checkpoint 并返回同源身份。"""
        return load_immutable_checkpoint(
            path, loader=cls._torch_load, label="model checkpoint"
        )

    @staticmethod
    def _to_observation_float32(pcm) -> torch.Tensor:
        """PCM 转 observation float32，保留显式 channel-first mixture。"""
        if isinstance(pcm, np.ndarray):
            if pcm.dtype.kind not in ("i", "u", "f"):
                raise TypeError(f"pcm 不支持 dtype {pcm.dtype}")
            pcm = torch.from_numpy(pcm)
        elif not isinstance(pcm, torch.Tensor):
            raise TypeError(f"pcm 应为 np.ndarray 或 torch.Tensor，got {type(pcm)!r}")

        if pcm.dtype == torch.bool or pcm.is_complex():
            raise TypeError(f"pcm 不支持 dtype {pcm.dtype}")
        if pcm.is_floating_point():
            pcm = pcm.float()
        else:
            try:
                info = torch.iinfo(pcm.dtype)
            except TypeError as exc:
                raise TypeError(f"pcm 不支持 dtype {pcm.dtype}") from exc
            if info.min < 0:
                scale = float(max(abs(info.min), abs(info.max)))
                pcm = pcm.float() / scale
            else:
                midpoint = float(info.max // 2 + 1)
                pcm = (pcm.float() - midpoint) / midpoint
                pcm = pcm.clamp(max=1.0 - torch.finfo(torch.float32).eps)

        if pcm.dim() not in (1, 2):
            raise ValueError(f"pcm 应为 1-D 或 [C, T]，got shape {tuple(pcm.shape)}")
        return pcm.contiguous()

    @classmethod
    def _to_mono_float32(cls, pcm) -> torch.Tensor:
        """PCM 转单声道 float32；enrollment 只接受单通道波形。"""
        pcm = cls._to_observation_float32(pcm)
        if pcm.dim() == 2:
            if pcm.shape[0] != 1:
                raise ValueError(
                    f"enrollment must be single-channel [T], got {tuple(pcm.shape)}"
                )
            return pcm[0]
        return pcm

    def _load_vad(self):
        """懒加载 silero-VAD；未安装时返回 None（调用方降级跳过）。"""
        if self._vad is not None:
            return self._vad
        try:
            from silero_vad import load_silero_vad
        except ImportError:
            logger.warning(
                "apply_vad=True 但未安装 silero-vad，跳过 enroll 静音裁剪"
                "（可选安装: pip install silero-vad）"
            )
            return None
        self._vad = load_silero_vad()
        return self._vad

    def _trim_enroll_silence(self, enroll: torch.Tensor, sr: int) -> torch.Tensor:
        """silero-VAD 裁掉 enroll 静音段并拼接语音段（对齐比赛 Extractor 语义）。

        与比赛实现的差异：比赛在全静音时返回 None 丢弃该样本；此处保留原 enroll
        并 warning（批量评测流水线中不丢行）。
        """
        vad = self._load_vad()
        if vad is None:
            return enroll
        from silero_vad import get_speech_timestamps

        wav16 = enroll
        if sr != 16000:
            wav16 = torchaudio.functional.resample(enroll, sr, 16000)
        segments = get_speech_timestamps(wav16, vad, return_seconds=False)
        if not segments:
            logger.warning("enroll 全静音（VAD 无语音段），保留原始 enroll")
            return enroll
        parts = []
        for seg in segments:
            s = int(seg["start"] * sr / 16000)
            e = int(seg["end"] * sr / 16000)
            parts.append(enroll[s:e])
        return torch.cat(parts) if parts else enroll

    def _get_spk_encoder(self):
        """预提取 embedding 模式（joint_training=False）下懒构建 SpeakerEncoder。"""
        if self._spk_encoder is not None:
            return self._spk_encoder
        if not isinstance(
            self._external_speaker_encoder_pretrained, str
        ) or not self._external_speaker_encoder_pretrained:
            raise RuntimeError(
                "预提取 speaker embedding 模式必须持有已解析的 checkpoint 路径"
            )
        pretrained_path = Path(self._external_speaker_encoder_pretrained)
        if not pretrained_path.is_absolute() or not pretrained_path.is_file():
            raise RuntimeError(
                "预提取 speaker embedding checkpoint 必须是已存在的绝对路径，"
                f"got {self._external_speaker_encoder_pretrained!r}"
            )
        speaker_model_cfg = self._speaker_model_cfg
        spk_cfg = speaker_model_cfg.get("speaker_encoder", {})
        from tasks.tse.models.speaker.encoder import SpeakerEncoder

        checkpoint, identity = load_immutable_checkpoint(
            pretrained_path,
            loader=lambda stream: safe_torch_load(stream, map_location="cpu"),
            label="speaker checkpoint",
        )
        encoder = SpeakerEncoder(
            model=spk_cfg.get("model", "ECAPA_TDNN_GLOB_c512"),
            pretrained=str(pretrained_path),
            freeze=True,
            spk_args=spk_cfg.get("spk_args"),
            fbank_args=speaker_model_cfg.get("fbank"),
            pretrained_state=checkpoint,
            pretrained_prefix=spk_cfg.get("pretrained_prefix"),
        ).to(self.device).eval()
        verify_file_identity(identity, label="speaker checkpoint")
        self._spk_encoder = encoder
        self._loaded_external_speaker_checkpoint_identity = identity
        return self._spk_encoder

    # ------------------------------------------------------------------
    # 推理接口（比赛 Extractor 同语义）
    # ------------------------------------------------------------------
    def extract_speech(self, mix_path: str, enroll_path: str) -> np.ndarray:
        """从拒绝 symlink 的稳定文件描述符读取并提取目标说话人语音。"""
        return self.extract_speech_bound(mix_path, enroll_path, None)

    def extract_speech_bound(
        self,
        mix_path: str,
        enroll_path: str,
        input_identities: Mapping[str, Mapping[str, object]] | None,
    ) -> np.ndarray:
        """读取与可选 artifact identity 绑定的两个输入并完成推理。"""
        expected_mix = (
            input_identities.get("mixture") if input_identities is not None else None
        )
        expected_enroll = (
            input_identities.get("enrollment") if input_identities is not None else None
        )
        loaded_mix, _ = load_bound_audio(
            mix_path, torchaudio.load, expected_mix
        )
        loaded_enroll, _ = load_bound_audio(
            enroll_path, torchaudio.load, expected_enroll
        )
        pcm_mix, sr_mix = loaded_mix
        pcm_enroll, sr_enroll = loaded_enroll
        return self.extract_speech_from_pcm(pcm_mix, sr_mix, pcm_enroll, sr_enroll)

    @torch.inference_mode()
    def extract_speech_from_pcm(
        self,
        pcm_mix,
        sr_mix: int,
        pcm_enroll,
        sr_enroll: int,
    ) -> np.ndarray:
        """从 PCM 提取目标说话人语音（整句推理）。

        ⚠️ 整句推理**无时长上限**：中间激活（谱/LSTM 状态）随时长线性增长，
        超长输入（数十分钟以上）有 OOM 风险，请在调用侧自行切段。

        Args:
            pcm_mix: 混合语音（np.ndarray 或 torch.Tensor，1-D 或 [C, T]）。
            sr_mix: 混合语音采样率。
            pcm_enroll: 单通道注册音（1-D 或 ``[1, T]``）。
            sr_enroll: 注册音采样率。

        Returns:
            1-D np.float32 波形（长度与重采样后的 mix 一致，峰值归一）。
        """
        sr_mix = _positive_sample_rate(sr_mix, "sr_mix")
        sr_enroll = _positive_sample_rate(sr_enroll, "sr_enroll")
        mix = self._to_observation_float32(pcm_mix)
        enroll = self._to_mono_float32(pcm_enroll)
        if mix.numel() == 0:
            raise ValueError("mix 为空（0 样本）")
        if enroll.numel() == 0:
            raise ValueError("enroll 为空（0 样本）")

        # ---- 可选 VAD 裁 enroll 静音（原始采样率）----
        if self.apply_vad:
            enroll = self._trim_enroll_silence(enroll, sr_enroll)

        # ---- 重采样到模型采样率 ----
        if sr_mix != self.sample_rate:
            mix = torchaudio.functional.resample(mix, sr_mix, self.sample_rate)
        if sr_enroll != self.sample_rate:
            enroll = torchaudio.functional.resample(
                enroll, sr_enroll, self.sample_rate
            )
        enroll_valid_len = int(enroll.shape[-1])

        # ---- STFT 最小长度守卫（reflect padding 硬性下限，右侧补零）----
        min_len = _min_stft_input_len(self.stft.n_fft)
        orig_len = mix.shape[-1]
        model_input_fields = resolve_tse_model_input_fields(self.model)
        enroll_min_len = self._speaker_min_length(model_input_fields, min_len)
        if orig_len < min_len:
            mix = torch.nn.functional.pad(mix, (0, min_len - orig_len))
        if enroll.shape[-1] < enroll_min_len:
            enroll = torch.nn.functional.pad(
                enroll, (0, enroll_min_len - enroll.shape[-1])
            )

        mix = mix.unsqueeze(0).to(self.device)
        enroll = enroll.unsqueeze(0).to(self.device)

        # ---- 共享 TSE dict 谱进谱出边界（与 trainer 同一字段与对齐语义）----
        spk_embedding = None
        if self._uses_preextracted_spk_embedding:
            # 仅 spkemb cue 需要句级预提取 embedding；其他 cue 直接使用 enroll 谱。
            spk_encoder = self._get_spk_encoder()
            spk_embedding = self._validate_spk_embedding(
                spk_encoder(enroll), spk_encoder
            )
        return self._run_model(
            mix,
            enroll=enroll,
            enroll_lengths=(
                torch.tensor(
                    [enroll_valid_len],
                    dtype=torch.int64,
                    device=self.device,
                )
                if {"enroll_lengths", "enroll_spectra", "enroll_wav"}.intersection(
                    model_input_fields
                )
                else None
            ),
            spk_embedding=spk_embedding,
            orig_len=orig_len,
        )

    def _speaker_min_length(self, model_input_fields, min_len: int) -> int:
        """enroll 波形进入模型前必须满足的最小长度（STFT 下限 + fbank 窗口）。"""
        if self._uses_preextracted_spk_embedding:
            spk_encoder = self._get_spk_encoder()
        else:
            # joint 模式仅在启用 context/spkemb 时使用内部 encoder。
            speaker_frontend = getattr(self.model, "spk_ft", None)
            spk_encoder = getattr(speaker_frontend, "encoder", None)
        enroll_min_len = (
            getattr(self, "_min_enroll_input_len", min_len)
            if "enroll_spectra" in model_input_fields
            else 1
        )
        if spk_encoder is not None:
            fbank = getattr(spk_encoder, "fbank", None)
            window_size = getattr(fbank, "window_size", None)
            if window_size is not None:
                if (
                    isinstance(window_size, bool)
                    or not isinstance(window_size, int)
                    or window_size <= 0
                ):
                    raise RuntimeError(
                        "SpeakerEncoder fbank.window_size must be a positive integer, "
                        f"got {window_size!r}"
                    )
                enroll_min_len = max(enroll_min_len, window_size)
        if (
            isinstance(enroll_min_len, bool)
            or not isinstance(enroll_min_len, int)
            or enroll_min_len < 1
        ):
            raise RuntimeError(
                "extractor enrollment input length invariant is invalid: "
                f"min_stft={min_len}, min_enroll={enroll_min_len!r}"
            )
        return enroll_min_len

    def _validate_spk_embedding(self, spk_embedding, spk_encoder) -> torch.Tensor:
        """校验句级 speaker embedding 的类型、device、维度与有限性。"""
        if not isinstance(spk_embedding, torch.Tensor):
            raise TypeError(
                "SpeakerEncoder must return a torch.Tensor sentence embedding, "
                f"got {type(spk_embedding).__name__}"
            )
        if not spk_embedding.is_floating_point():
            raise TypeError(
                "SpeakerEncoder embedding must have a floating dtype, "
                f"got {spk_embedding.dtype}"
            )
        if spk_embedding.device != self.device:
            raise ValueError(
                "SpeakerEncoder embedding device mismatch: "
                f"embedding={spk_embedding.device}, extractor={self.device}"
            )
        if spk_embedding.dim() != 2 or spk_embedding.shape[0] != 1:
            raise ValueError(
                "SpeakerEncoder embedding must have one sentence row per enroll row: "
                f"expected (1, *), got {tuple(spk_embedding.shape)}"
            )
        expected_embed_dim = getattr(spk_encoder, "embed_dim", None)
        if expected_embed_dim is None:
            feature_cfg = getattr(self.model, "features_config", {})
            if isinstance(feature_cfg, Mapping):
                spk_cfg = feature_cfg.get("spkemb") or {}
                if isinstance(spk_cfg, Mapping):
                    expected_embed_dim = spk_cfg.get("embed_dim")
        if expected_embed_dim is not None:
            if (
                isinstance(expected_embed_dim, bool)
                or not isinstance(expected_embed_dim, int)
                or expected_embed_dim <= 0
            ):
                raise ValueError(
                    "SpeakerEncoder embedding dimension configuration must be a "
                    f"positive integer, got {expected_embed_dim!r}"
                )
            if spk_embedding.shape[1] != expected_embed_dim:
                raise ValueError(
                    "SpeakerEncoder embedding dimension mismatch: "
                    f"expected {expected_embed_dim}, got {spk_embedding.shape[1]}"
                )
        if not bool(torch.isfinite(spk_embedding).all().item()):
            raise ValueError("SpeakerEncoder embedding contains NaN/Inf")
        return spk_embedding.detach()

    def _run_model(
        self,
        mix: torch.Tensor,
        *,
        cues: TSECueBatches | None = None,
        enroll: torch.Tensor | None = None,
        enroll_lengths: torch.Tensor | None = None,
        spk_embedding: torch.Tensor | None = None,
        orig_len: int,
    ) -> np.ndarray:
        """按共享 TSE 边界跑模型、解码并对齐参考长度，最后峰值归一。"""
        data = build_tse_model_inputs(
            model=self.model,
            stft=self.stft,
            mixture=mix,
            cues=cues,
            enroll=enroll,
            enroll_lengths=enroll_lengths,
            spk_embedding=spk_embedding,
            sample_rate=self.sample_rate,
        )
        data = decode_tse_model_output(
            self.model(data),
            stft=self.stft,
            reference_length=mix.shape[-1],
        )
        est_wav = data["target_estimate"][..., :orig_len].squeeze(0).float().cpu()

        # ---- 峰值归一化（比赛默认 0.9；全零输出守卫）----
        if self.output_peak is not None and self.output_peak > 0:
            peak = est_wav.abs().max()
            if peak > 0:
                est_wav = est_wav / peak * float(self.output_peak)
        return est_wav.numpy().astype(np.float32, copy=False)

    def prepare_speaker_waveform_cue(
        self,
        pcm_enroll,
        sr_enroll: int,
    ) -> TSECueInput:
        """把注册音 PCM 转成 speaker_waveform cue（重采样、VAD、最小长度补齐）。

        这是构造 speaker_waveform cue 的唯一支持入口：只有 extractor 知道模型的
        STFT 下限与 speaker encoder fbank 窗口。
        """
        sr_enroll = _positive_sample_rate(sr_enroll, "sr_enroll")
        enroll = self._to_mono_float32(pcm_enroll)
        if enroll.numel() == 0:
            raise ValueError("enroll 为空（0 样本）")
        if self.apply_vad:
            enroll = self._trim_enroll_silence(enroll, sr_enroll)
        if sr_enroll != self.sample_rate:
            enroll = torchaudio.functional.resample(
                enroll, sr_enroll, self.sample_rate
            )
        valid_len = int(enroll.shape[-1])
        model_input_fields = resolve_tse_model_input_fields(self.model)
        min_len = _min_stft_input_len(self.stft.n_fft)
        enroll_min_len = self._speaker_min_length(model_input_fields, min_len)
        if enroll.shape[-1] < enroll_min_len:
            enroll = torch.nn.functional.pad(
                enroll, (0, enroll_min_len - enroll.shape[-1])
            )
        return TSECueInput(
            role="speaker_waveform",
            representation="waveform",
            value=enroll.unsqueeze(0).to(self.device),
            length=valid_len,
            timebase=None,
        )

    def _cue_batches(
        self,
        cue_inputs: Mapping[str, TSECueInput],
        roles: Mapping[str, TSECueRole],
    ) -> TSECueBatches:
        """把已读入的单行 cue 组装成共享 typed cue batch。"""
        batches: dict[str, TSECueBatch | None] = {}
        for role in (
            "speaker_waveform",
            "speaker_embedding",
            "spatial",
            "visual",
            "textual",
        ):
            declared = roles.get(role)
            cue = cue_inputs.get(role)
            if cue is None:
                batches[role] = None
                continue
            assert declared is not None
            if cue.representation != declared.representation:
                raise ValueError(
                    f"cue {role!r} representation {cue.representation!r} does not match "
                    f"model contract {declared.representation!r}"
                )
            if cue.value.ndim - 1 != declared.row_rank:
                raise ValueError(
                    f"cue {role!r} must have {declared.row_rank} value dimensions, "
                    f"got shape {tuple(cue.value.shape)}"
                )
            value = cue.value.to(self.device)
            present = torch.ones(value.shape[0], dtype=torch.bool, device=value.device)
            if declared.representation in {"static", "embedding"}:
                if cue.length is not None or cue.timebase is not None:
                    raise ValueError(
                        f"static cue {role!r} must not carry length or timebase"
                    )
                lengths = None
                timebase = None
            else:
                if cue.length is None or cue.length < 1:
                    raise ValueError(f"temporal cue {role!r} requires a positive length")
                lengths = torch.tensor(
                    [int(cue.length)], dtype=torch.int32, device=value.device
                )
                timebase = (
                    None
                    if cue.timebase is None
                    else cue.timebase.to(self.device)
                )
                if declared.representation in {"track", "raw", "features"} and (
                    timebase is None
                ):
                    raise ValueError(
                        f"cue {role!r}/{declared.representation} requires a timebase"
                    )
            batches[role] = TSECueBatch(
                representation=cue.representation,
                value=value,
                present=present,
                lengths=lengths,
                timebase=timebase,
            )
        return TSECueBatches(**batches)

    @torch.inference_mode()
    def extract_speech_with_cues(
        self,
        pcm_mix,
        sr_mix: int,
        cues: Mapping[str, TSECueInput],
    ) -> np.ndarray:
        """按模型声明的 cue role 做整句推理。

        ``cues`` 由文件推理层从实际 cue 文件与 inline cue JSON 读入；模型合同
        声明的每个 role 都必须提供，未声明的 role 一律拒绝。这里不做文件
        I/O、manifest、sidecar 或 reuse 判定。

        ⚠️ 整句推理**无时长上限**：超长输入有 OOM 风险，请在调用侧自行切段。
        """
        sr_mix = _positive_sample_rate(sr_mix, "sr_mix")
        mix = self._to_observation_float32(pcm_mix)
        if mix.numel() == 0:
            raise ValueError("mix 为空（0 样本）")
        if not isinstance(cues, Mapping):
            raise TypeError("cues must be a mapping of role to TSECueInput")
        roles = self.declared_cue_roles()
        unknown = sorted(set(cues) - set(roles))
        if unknown:
            raise ValueError(
                f"model.IO_CONTRACT does not consume cue roles: {unknown}"
            )
        missing = sorted(
            role for role, spec in roles.items() if spec.required and role not in cues
        )
        if missing:
            raise ValueError(f"missing required TSE cue roles: {missing}")
        for role, cue in cues.items():
            if not isinstance(cue, TSECueInput):
                raise TypeError(f"cue {role!r} must be a TSECueInput")
            if cue.role != role:
                raise ValueError(
                    f"cue mapping key {role!r} does not match cue.role {cue.role!r}"
                )

        if sr_mix != self.sample_rate:
            mix = torchaudio.functional.resample(mix, sr_mix, self.sample_rate)
        orig_len = mix.shape[-1]
        min_len = _min_stft_input_len(self.stft.n_fft)
        model_input_fields = resolve_tse_model_input_fields(self.model)
        speaker_cue = cues.get("speaker_waveform")
        if speaker_cue is not None and speaker_cue.length is None:
            raise ValueError("speaker_waveform cue requires its valid sample length")
        if orig_len < min_len:
            mix = torch.nn.functional.pad(mix, (0, min_len - orig_len))
        mix = mix.unsqueeze(0).to(self.device)

        spk_embedding = None
        if speaker_cue is not None and self._uses_preextracted_spk_embedding:
            spk_encoder = self._get_spk_encoder()
            spk_embedding = self._validate_spk_embedding(
                spk_encoder(speaker_cue.value.to(self.device)), spk_encoder
            )
        batches = self._cue_batches(cues, roles)
        if "enroll_spectra" in model_input_fields:
            speaker = batches.speaker_waveform
            if speaker is not None:
                minimum = self._speaker_min_length(model_input_fields, min_len)
                if speaker.value.shape[-1] < minimum:
                    raise ValueError(
                        "speaker_waveform cue is shorter than the model frontend "
                        f"minimum {minimum}"
                    )
        return self._run_model(
            mix, cues=batches, spk_embedding=spk_embedding, orig_len=orig_len
        )


def load_model_local(model_dir: str, device: str = "cuda:0", **kwargs) -> TSEExtractor:
    """比赛 ``wesep.load_model_local`` 同款入口：从 model_dir 构建 TSEExtractor。"""
    return TSEExtractor(model_dir, device=device, **kwargs)
