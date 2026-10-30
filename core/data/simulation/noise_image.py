"""将 ``noise_scp`` 波形统一转换为 observation-side ``[C,T]`` image。

公开数据配置只需要提供 ``noise_scp``。本模块不推断噪声的物理类别：
mono 输入使用已有的 geometry-aware renderer，或由调用方显式选择 broadcast
退化；已经是多通道的输入只做严格的阵列通道校验，不再次经过 RIR。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from core.data.simulation.spatial_background import render_diffuse_background
from core.data.simulation.energy_ratio import EnergyRatioScaler


@dataclass(frozen=True, slots=True)
class NoiseImage:
    """统一的 observation-side noise image 及其来源事实。"""

    waveform: np.ndarray
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = np.asarray(self.waveform)
        if value.dtype != np.float32 or value.ndim != 2:
            raise TypeError("noise image must be a float32 ndarray with shape [C,T]")
        if value.shape[0] < 1 or value.shape[1] < 1:
            raise ValueError("noise image must have positive [C,T] dimensions")
        if not np.isfinite(value).all():
            raise ValueError("noise image contains NaN or Inf")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("noise image provenance must be a mapping")
        object.__setattr__(self, "waveform", np.ascontiguousarray(value))
        object.__setattr__(self, "provenance", dict(self.provenance))


def _positions_from_geometry(geometry: Any) -> np.ndarray:
    if isinstance(geometry, Mapping):
        positions = geometry.get("positions_m", geometry.get("mic_loc"))
        if positions is None:
            raise ValueError("geometry mapping must provide positions_m")
    else:
        positions = getattr(geometry, "positions_m", geometry)
    value = np.asarray(positions, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or value.shape[0] < 1:
        raise ValueError("geometry must provide microphone positions with shape [C,3]")
    return value


def _source_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()


def adapt_noise_waveform(
    noise: np.ndarray,
    *,
    geometry: Any,
    sample_rate: int,
    rng: np.random.Generator | None = None,
    mono_policy: str = "geometry_aware",
    coherence_policy: str = "habets_spherical",
    reference_channel: int | None = None,
    provenance: Mapping[str, Any] | None = None,
    source_identity: str | None = None,
) -> NoiseImage:
    """把 ``noise_scp`` 读取结果转换成 ``[C,T]``。

    ``mono_policy`` 默认调用 Habets spherical coherence renderer；``broadcast`` 是
    明确记录的退化策略。多通道输入必须已经匹配 geometry，不允许再次卷积或
    隐式裁剪/广播。
    """
    value = np.asarray(noise)
    if value.dtype != np.float32:
        value = value.astype(np.float32)
    if value.ndim not in (1, 2):
        raise ValueError("noise waveform must have shape [T] or [C,T]")
    positions = _positions_from_geometry(geometry)
    channels = int(positions.shape[0])
    base = dict(provenance or {})
    base.update({"input_shape": list(value.shape), "input_channels": 1 if value.ndim == 1 else int(value.shape[0])})
    base["source_identity"] = source_identity or ("noise_scp:inline:" + _source_hash(value))
    base["target_channels"] = channels
    base["sample_rate"] = int(sample_rate)
    base["geometry"] = {
        "microphone_positions_m": positions.tolist(),
    }
    base["coherence_policy"] = str(coherence_policy)

    if value.ndim == 2 and value.shape[0] == 1:
        # C=1 geometry 时保持 mono；阵列 geometry 时把 [1,T] 视为 mono source。
        if channels == 1:
            image = np.ascontiguousarray(value, dtype=np.float32)
            base["projection"] = "mono_passthrough"
            base["output_shape"] = list(image.shape)
            base["rir_applied"] = False
            return NoiseImage(image, base)
        value = value[0]
        base["input_shape"] = [int(value.shape[0])]
        base["input_channels"] = 1

    if value.ndim == 2:
        if value.shape[0] != channels:
            raise ValueError(f"noise_scp multichannel input has C={value.shape[0]}, expected geometry C={channels}")
        image = np.ascontiguousarray(value, dtype=np.float32)
        base["projection"] = "direct_multichannel"
    else:
        if value.shape[0] < 1:
            raise ValueError("noise waveform must be non-empty")
        if channels == 1 and value.ndim == 1:
            image = value[None, :].copy()
            base["projection"] = "mono_passthrough"
            base["output_shape"] = list(image.shape)
            base["rir_applied"] = False
            return NoiseImage(image, base)
        if mono_policy == "geometry_aware":
            if rng is None:
                raise ValueError("rng is required for mono_policy='geometry_aware'")
            result = render_diffuse_background(
                value, positions, int(sample_rate), rng,
                coherence_policy=coherence_policy,
                reference_channel=0 if reference_channel is None else int(reference_channel),
                source_identity=source_identity,
                provenance=base,
            )
            image = result.waveform
            base.update(result.provenance)
            base["projection"] = "geometry_aware"
        elif mono_policy == "broadcast":
            image = np.broadcast_to(value[None, :], (channels, value.shape[0])).copy()
            base["projection"] = "broadcast"
        else:
            raise ValueError("mono_policy must be 'geometry_aware' or 'broadcast'")
    base["output_shape"] = list(image.shape)
    base["rir_applied"] = False
    return NoiseImage(image, base)


def scale_noise_image(
    speech_image: np.ndarray,
    noise_image: NoiseImage | np.ndarray,
    *,
    snr_db: float,
    provenance: Mapping[str, Any] | None = None,
) -> NoiseImage:
    """对完整 ``[C,T]`` image 只计算一次 gain。

    SNR 采用全部通道、全部时间样本的 ``full_sum_energy``；因此非对称阵列
    不会按 reference channel 分别缩放，也不会在 flatten 前后重复缩放。
    """
    speech = np.asarray(speech_image, dtype=np.float32)
    source = noise_image.waveform if isinstance(noise_image, NoiseImage) else np.asarray(noise_image, dtype=np.float32)
    if speech.ndim != 2 or source.ndim != 2 or speech.shape != source.shape:
        raise ValueError("speech and noise images must have identical [C,T] shapes")
    try:
        snr_value = float(snr_db)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"snr_db must be finite, got {snr_db!r}") from exc
    if not np.isfinite(snr_value):
        raise ValueError(f"snr_db must be finite, got {snr_db!r}")
    if not np.isfinite(speech).all() or not np.isfinite(source).all():
        raise ValueError("speech and noise images must be finite")
    numerator = EnergyRatioScaler.measure(speech, policy="full_sum_energy")
    denominator = EnergyRatioScaler.measure(source, policy="full_sum_energy")
    if numerator <= 1e-10 or denominator <= 1e-10:
        facts = dict(getattr(noise_image, "provenance", {}) or {})
        facts.update(dict(provenance or {}))
        facts.update({
            "snr_policy": "full_image_energy",
            "snr_reference": facts.get("snr_reference", "full_image_speech"),
            "snr_nominal_db": snr_value,
            "snr_realized_db": None,
            "snr_gain": None,
            "snr_applicable": False,
            "snr_reason": "silent_reference" if numerator <= 1e-10 else "silent_noise",
        })
        return NoiseImage(np.ascontiguousarray(source, dtype=np.float32), facts)
    ratio = EnergyRatioScaler.power_ratio(
        numerator, denominator, snr_value, policy="full_sum_energy",
        denominator_floor=1e-10, gain_cap=1e4,
    )
    facts = dict(getattr(noise_image, "provenance", {}) or {})
    facts.update(dict(provenance or {}))
    facts.update({
        "snr_policy": "full_image_energy",
        "snr_reference": facts.get("snr_reference", "full_image_speech"),
        "snr_nominal_db": snr_value,
        "snr_realized_db": float(ratio.realized_db),
        "snr_gain": float(ratio.gain),
    })
    return NoiseImage((source * np.float32(ratio.gain)).astype(np.float32), facts)


__all__ = ["NoiseImage", "adapt_noise_waveform", "scale_noise_image"]
