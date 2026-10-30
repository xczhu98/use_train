"""PASE 家族 validation 验证音频的公共写入器。

对齐官方各阶段 ``_validation_epoch`` 中的 ``sf.write`` 行为（仅 rank 0、仅在
validation 循环内、按 epoch/step 条件触发）。本模块只负责：

- ``trainer.val_audio`` 的严格解析（未知键 TypeError，非法类型/取值 ValueError）；
- 触发条件（官方 ``epoch < start_epoch + 10 or epoch % 10 == 0`` 与 ``step``
  上限；官方与本框架 epoch 均为 1-based；
  GAP Phase 2 的 ``epoch < 10`` 由 ``initial_epoch_reference=zero`` 表达）；
- 目录创建、峰值归一化辅助、``soundfile.write(float32, sr)``；
- 按 ``kind`` 懒加载各阶段冻结 vocoder（PASE/StuPASE/UniPASE 发布 artifact）。

阶段专属的中间波形由模型/生成器包装器的 ``val_audio_outputs()`` 计算，本模块
不复制任何阶段前向逻辑。文件名中的 epoch 与官方一致使用 1-based epoch；uid
使用 validation 循环的 1-based step（官方 ``train_flow.py`` 的
``uid = f"{step}"`` 口径），SEBatch 不携带数据集 id，这是与官方其它阶段
「用 ``info['id'][0]``」的唯一命名差异。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
import torch

from core.utils.validation import require_positive_int
from tasks.se.models.pase_family.common.utils import peak_normalize_to

logger = logging.getLogger(__name__)

_VAL_AUDIO_KEYS = frozenset(
    {
        "enabled",
        "output_dir",
        "save_initial_epochs",
        "save_every_n_epochs",
        "initial_epoch_reference",
        "max_batches",
        "vocoder_ckpt_path",
        "vocoder_kind",
        "wavlm_vocoder_ckpt_path",
        "wavlm_vocoder_kind",
        "sample_settings",
    }
)
_SAMPLE_SETTING_KEYS = frozenset({"steps", "cfg_strength", "sway_sampling_coef"})
_INITIAL_EPOCH_REFERENCES = ("start_epoch", "zero")
_VOCODER_KINDS = ("stupase", "unipase")


@dataclass(frozen=True)
class ValAudioSampleSettings:
    """``CFM.sample`` 的官方 ``sample_settings``（StuPASE Phase 2）。"""

    steps: int
    cfg_strength: float
    sway_sampling_coef: float | None

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "cfg_strength": self.cfg_strength,
            "sway_sampling_coef": self.sway_sampling_coef,
        }


@dataclass(frozen=True)
class ValAudioConfig:
    """已校验的 ``trainer.val_audio`` 配置。"""

    enabled: bool
    output_dir: str | None
    save_initial_epochs: int
    save_every_n_epochs: int
    initial_epoch_reference: str
    max_batches: int
    vocoder_ckpt_path: str | None
    vocoder_kind: str
    wavlm_vocoder_ckpt_path: str | None
    wavlm_vocoder_kind: str
    sample_settings: ValAudioSampleSettings | None


def _require_bool(raw: Mapping[str, Any], key: str) -> bool:
    value = raw[key]
    if type(value) is not bool:
        raise TypeError(f"trainer.val_audio.{key} must be a boolean")
    return value


def _require_int(raw: Mapping[str, Any], key: str, *, minimum: int) -> int:
    return require_positive_int(raw[key], f"trainer.val_audio.{key}", minimum=minimum)


def _require_str(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"trainer.val_audio.{key} must be a non-empty string or null")
    return value


def _parse_sample_settings(raw: Any) -> ValAudioSampleSettings:
    if not isinstance(raw, Mapping):
        raise TypeError("trainer.val_audio.sample_settings must be a mapping")
    unknown = sorted(set(raw) - _SAMPLE_SETTING_KEYS)
    if unknown:
        raise TypeError(
            f"Unknown trainer.val_audio.sample_settings keys: {unknown}"
        )
    for key in ("steps", "cfg_strength"):
        if key not in raw:
            raise ValueError(
                f"trainer.val_audio.sample_settings.{key} is required"
            )
    steps = _require_int(raw, "steps", minimum=1)
    cfg_strength = raw["cfg_strength"]
    if isinstance(cfg_strength, bool) or not isinstance(cfg_strength, (int, float)):
        raise TypeError("trainer.val_audio.sample_settings.cfg_strength must be a number")
    if not math.isfinite(float(cfg_strength)):
        raise ValueError("trainer.val_audio.sample_settings.cfg_strength must be finite")
    sway = raw.get("sway_sampling_coef")
    if sway is not None:
        if isinstance(sway, bool) or not isinstance(sway, (int, float)):
            raise TypeError(
                "trainer.val_audio.sample_settings.sway_sampling_coef must be a number or null"
            )
        if not math.isfinite(float(sway)):
            raise ValueError(
                "trainer.val_audio.sample_settings.sway_sampling_coef must be finite"
            )
        sway = float(sway)
    return ValAudioSampleSettings(
        steps=steps,
        cfg_strength=float(cfg_strength),
        sway_sampling_coef=sway,
    )


def parse_val_audio_config(raw: Any) -> ValAudioConfig:
    """严格解析 ``trainer.val_audio``；未知键直接 TypeError。"""
    if not isinstance(raw, Mapping):
        raise TypeError("trainer.val_audio must be a mapping")
    unknown = sorted(set(raw) - _VAL_AUDIO_KEYS)
    if unknown:
        raise TypeError(f"Unknown trainer.val_audio config keys: {unknown}")

    enabled = _require_bool(raw, "enabled") if "enabled" in raw else False
    output_dir = _require_str(raw, "output_dir") if "output_dir" in raw else None
    save_initial_epochs = (
        _require_int(raw, "save_initial_epochs", minimum=0)
        if "save_initial_epochs" in raw
        else 10
    )
    save_every_n_epochs = (
        _require_int(raw, "save_every_n_epochs", minimum=1)
        if "save_every_n_epochs" in raw
        else 10
    )
    initial_epoch_reference = raw.get("initial_epoch_reference", "start_epoch")
    if initial_epoch_reference not in _INITIAL_EPOCH_REFERENCES:
        raise ValueError(
            "trainer.val_audio.initial_epoch_reference must be one of "
            f"{list(_INITIAL_EPOCH_REFERENCES)}, got {initial_epoch_reference!r}"
        )
    max_batches = (
        _require_int(raw, "max_batches", minimum=1)
        if "max_batches" in raw
        else 10
    )
    vocoder_ckpt_path = (
        _require_str(raw, "vocoder_ckpt_path") if "vocoder_ckpt_path" in raw else None
    )
    vocoder_kind = raw.get("vocoder_kind", "unipase")
    if vocoder_kind not in _VOCODER_KINDS:
        raise ValueError(
            "trainer.val_audio.vocoder_kind must be one of "
            f"{list(_VOCODER_KINDS)}, got {vocoder_kind!r}"
        )
    wavlm_vocoder_ckpt_path = (
        _require_str(raw, "wavlm_vocoder_ckpt_path")
        if "wavlm_vocoder_ckpt_path" in raw
        else None
    )
    wavlm_vocoder_kind = raw.get("wavlm_vocoder_kind", "unipase")
    if wavlm_vocoder_kind not in _VOCODER_KINDS:
        raise ValueError(
            "trainer.val_audio.wavlm_vocoder_kind must be one of "
            f"{list(_VOCODER_KINDS)}, got {wavlm_vocoder_kind!r}"
        )
    sample_settings = (
        _parse_sample_settings(raw["sample_settings"])
        if "sample_settings" in raw
        else None
    )
    return ValAudioConfig(
        enabled=enabled,
        output_dir=output_dir,
        save_initial_epochs=save_initial_epochs,
        save_every_n_epochs=save_every_n_epochs,
        initial_epoch_reference=initial_epoch_reference,
        max_batches=max_batches,
        vocoder_ckpt_path=vocoder_ckpt_path,
        vocoder_kind=vocoder_kind,
        wavlm_vocoder_ckpt_path=wavlm_vocoder_ckpt_path,
        wavlm_vocoder_kind=wavlm_vocoder_kind,
        sample_settings=sample_settings,
    )


def load_published_vocoder(kind: str, ckpt_path: str, *, device: torch.device) -> torch.nn.Module:
    """按官方发布 artifact 的类别加载冻结 vocoder。

    - ``stupase``：StuPASE ``VocosVocoder``（log-mag 裁剪），用于 StuPASE Phase 0
      的 ``Vocoder_L24`` 与 Phase 2 的 ``Vocoder_Mel-16k``；
    - ``unipase``：UniPASE ``VocosVocoder``，用于 Stage 1/3 的
      ``Vocoder_WavLM-L24`` 与 ``Vocoder_DWO-L1``。

    输入布局统一为 ``(B, D, T)``。加载走 ``from_pretrained`` 的
    ``weights_only`` + key 匹配 fail-closed 语义。
    """
    if kind == "stupase":
        from tasks.se.models.pase_family.stupase.vocos_vocoder import (
            VocosVocoder as _Vocoder,
        )
    elif kind == "unipase":
        from tasks.se.models.pase_family.unipase.vocoder import (
            VocosVocoder as _Vocoder,
        )
    else:
        raise ValueError(
            f"unsupported val_audio vocoder kind {kind!r}; expected one of {list(_VOCODER_KINDS)}"
        )
    module = _Vocoder.from_pretrained(ckpt_path, frozen=True)
    module.eval()
    return module.to(device)


def align_last_dim(wav: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Pad or crop the last dimension to ``n_samples``（官方 ``align_length``）。"""
    if wav.shape[-1] < n_samples:
        wav = torch.nn.functional.pad(wav, (0, n_samples - wav.shape[-1]))
    elif wav.shape[-1] > n_samples:
        wav = wav[..., :n_samples]
    return wav


def waveform_to_numpy(wav: torch.Tensor | np.ndarray) -> np.ndarray:
    """将单条波形转为 float32 1-D numpy。

    官方 ``info['id'][0]`` 口径：batch 维只保存第 0 条；2-D 输入的其余样本
    被丢弃（validation 配置均使用 ``val_batch_size: 1``）。更高的维度直接拒绝。
    """
    if torch.is_tensor(wav):
        data = wav.detach().cpu().float().numpy()
    else:
        data = np.asarray(wav, dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[0]
    if data.ndim != 1:
        raise ValueError(f"val audio waveform must be 1-D, got shape {data.shape}")
    return np.ascontiguousarray(data, dtype=np.float32)


@contextmanager
def preserve_rng_state(device: torch.device) -> Iterator[None]:
    """保存/恢复全局 RNG，使验证采样不改变训练侧随机流。"""
    cpu_state = torch.get_rng_state()
    cuda_state = None
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state(device)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


class ValAudioWriter:
    """rank-0 validation 音频写入器；``enabled=false`` 时不会构造。"""

    def __init__(
        self,
        config: ValAudioConfig,
        *,
        exp_path: str,
        rank: int,
        device: torch.device,
        sample_rate: int,
    ) -> None:
        if not config.enabled:
            raise ValueError("ValAudioWriter requires an enabled ValAudioConfig")
        self.config = config
        self.rank = int(rank)
        self.device = torch.device(device)
        self.sample_rate = int(sample_rate)
        self.root = Path(config.output_dir).expanduser() if config.output_dir else (
            Path(exp_path) / "samples"
        )
        # plain dict：冻结 vocoder 不是模型子模块，不得进入 state_dict / optimizer。
        self._vocoders: dict[str, torch.nn.Module] = {}

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        rank: int,
        device: torch.device,
        sample_rate: int,
    ) -> "ValAudioWriter | None":
        """从完整训练配置构造；未声明或 ``enabled=false`` 时返回 None。"""
        trainer_cfg = config.get("trainer") or {}
        raw = trainer_cfg.get("val_audio")
        if raw is None:
            return None
        parsed = parse_val_audio_config(raw)
        if not parsed.enabled:
            return None
        exp_path = trainer_cfg.get("exp_path") or "exp/default"
        return cls(
            parsed,
            exp_path=str(exp_path),
            rank=rank,
            device=device,
            sample_rate=sample_rate,
        )

    # ------------------------------------------------------------------
    # 触发条件
    # ------------------------------------------------------------------
    def should_save(self, epoch: int, start_epoch: int, step: int) -> bool:
        """官方条件：``(epoch < base + N or epoch % N == 0) and step <= max_batches``。

        本框架与官方训练循环的 epoch 都是 1-based（``start_epoch=1`` 为首个训练
        epoch，resume 后为 checkpoint epoch + 1），因此这里直接使用 epoch，
        不再做 0-based 换算。官方训练前的 ``_validation_epoch(0)`` 若启用，会以
        epoch=0 调用本方法，从而写出 ``epoch000`` 参考音频。
        """
        if self.rank != 0:
            return False
        if step <= 0 or step > self.config.max_batches:
            return False
        epoch = int(epoch)
        if epoch < 0:
            return False
        base = (
            int(start_epoch)
            if self.config.initial_epoch_reference == "start_epoch"
            else 0
        )
        return (
            epoch < base + self.config.save_initial_epochs
            or epoch % self.config.save_every_n_epochs == 0
        )

    # ------------------------------------------------------------------
    # 冻结 vocoder
    # ------------------------------------------------------------------
    def _role_path(self, role: str) -> str | None:
        if role == "vocoder":
            return self.config.vocoder_ckpt_path
        if role == "wavlm_vocoder":
            return self.config.wavlm_vocoder_ckpt_path
        raise ValueError(f"unknown val_audio vocoder role {role!r}")

    def _role_kind(self, role: str) -> str:
        if role == "vocoder":
            return self.config.vocoder_kind
        if role == "wavlm_vocoder":
            return self.config.wavlm_vocoder_kind
        raise ValueError(f"unknown val_audio vocoder role {role!r}")

    def validate_roles(self, roles: Any) -> None:
        """训练启动即校验各阶段必需的 vocoder 路径，缺失 fail-fast。"""
        for role in roles:
            if not self._role_path(role):
                raise ValueError(
                    f"trainer.val_audio requires the {role!r} checkpoint path for "
                    f"this stage; set trainer.val_audio.{role}_ckpt_path"
                )

    def load_vocoders(self, roles: Any) -> dict[str, torch.nn.Module]:
        out: dict[str, torch.nn.Module] = {}
        for role in roles:
            path = self._role_path(role)
            if not path:
                raise ValueError(
                    f"trainer.val_audio requires the {role!r} checkpoint path for this stage"
                )
            module = self._vocoders.get(role)
            if module is None:
                module = load_published_vocoder(
                    self._role_kind(role), path, device=self.device
                )
                self._vocoders[role] = module
            out[role] = module
        return out

    def capture(
        self,
        raw_model: torch.nn.Module,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None,
        *,
        fs: int,
        epoch: int,
        step: int,
        start_epoch: int,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """触发判定、vocoder 加载、RNG 保护、阶段前向与写盘的统一编排。

        trainer 只负责解包出原始模型并提供 batch 事实；``model_kwargs``
        透传阶段专属参数（如 GAN 阶段的 ``forward_dict``/``raw_enhanced``）。
        输出采样率依次取 ``val_audio_sample_rate`` / ``output_sample_rate`` /
        batch ``fs``。
        """
        if not self.should_save(epoch, start_epoch, step):
            return
        vocoders = self.load_vocoders(getattr(raw_model, "val_audio_vocoder_roles", ()))
        kwargs = dict(model_kwargs or {})
        kwargs["vocoders"] = vocoders
        kwargs["sample_settings"] = (
            None
            if self.config.sample_settings is None
            else self.config.sample_settings.as_kwargs()
        )
        with preserve_rng_state(self.device):
            items = raw_model.val_audio_outputs(noisy, clean, lengths, fs=fs, **kwargs)
        out_sr = (
            getattr(raw_model, "val_audio_sample_rate", None)
            or getattr(raw_model, "output_sample_rate", None)
            or fs
        )
        self.save(epoch, step, items, sample_rate=int(out_sr))

    # ------------------------------------------------------------------
    # 写盘
    # ------------------------------------------------------------------
    def save(
        self,
        epoch: int,
        step: int,
        items: Mapping[str, torch.Tensor | np.ndarray],
        *,
        sample_rate: int | None = None,
    ) -> None:
        """写入 ``{step}_{template}.wav``；仅 rank 0 且 items 非空时落盘。

        文件名里的 ``epoch`` 与官方一致，直接使用 1-based trainer epoch；
        ``step`` 为 validation 循环的 1-based 序号。
        """
        if self.rank != 0 or not items:
            return
        sr = int(self.sample_rate if sample_rate is None else sample_rate)
        epoch_label = int(epoch)
        self.root.mkdir(parents=True, exist_ok=True)
        for template, waveform in items.items():
            name = f"{int(step)}_{template.format(epoch=epoch_label)}.wav"
            data = waveform_to_numpy(waveform)
            sf.write(self.root / name, data, sr)
        logger.info(
            "[ValAudio] epoch %s step %s: wrote %d file(s) to %s",
            epoch,
            step,
            len(items),
            self.root,
        )
