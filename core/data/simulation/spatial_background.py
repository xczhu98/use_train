"""多通道背景噪声合同与空间 diffuse-field 渲染。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpatialBackground:
    """显式的 ``[C,T]`` observation-side additive background。

    ``source`` 是面向审计的简短来源字段；资产合同使用
    ``source_identity``、``coherence_policy``、``sample_rate``、``geometry``、
    ``generator_version``、``provenance`` 和 ``content_hash``。``physics_status``
    明确区分 ``exact_asset`` 与通用空间近似，避免把近似结果宣称为 Noise92
    物理复现。
    """

    waveform: np.ndarray
    field: str = "diffuse"
    source: str | None = None
    source_identity: str | None = None
    coherence_policy: str = "unspecified"
    sample_rate: int | None = None
    geometry: Mapping[str, Any] | None = None
    generator_version: str = "unknown"
    provenance: Mapping[str, Any] | None = None
    content_hash: str | None = None
    asset_kind: str = "generated"
    physics_status: str = "approximate"

    def __post_init__(self) -> None:
        value = np.asarray(self.waveform)
        if value.dtype != np.float32 or value.ndim != 2:
            raise TypeError("spatial background must be a float32 ndarray with shape [C,T]")
        if value.shape[0] < 1 or value.shape[1] < 1:
            raise ValueError("spatial background must have positive [C,T] dimensions")
        if not np.isfinite(value).all():
            raise ValueError("spatial background contains NaN or Inf")
        if self.field != "diffuse":
            raise ValueError("unsupported spatial background field; expected 'diffuse'")
        if self.source is not None and (not isinstance(self.source, str) or not self.source.strip()):
            raise ValueError("source must be a non-empty string or None")
        source_identity = self.source_identity or self.source
        if not isinstance(source_identity, str) or not source_identity.strip():
            raise ValueError("source_identity must be a non-empty string")
        if self.sample_rate is not None and (type(self.sample_rate) is not int or self.sample_rate <= 0):
            raise ValueError("sample_rate must be a positive int or None")
        if not isinstance(self.coherence_policy, str) or not self.coherence_policy.strip():
            raise ValueError("coherence_policy must be a non-empty string")
        if not isinstance(self.generator_version, str) or not self.generator_version.strip():
            raise ValueError("generator_version must be a non-empty string")
        if self.geometry is not None and not isinstance(self.geometry, Mapping):
            raise TypeError("geometry must be a mapping or None")
        if self.provenance is not None and not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping or None")
        if self.asset_kind not in {"generated", "noise92", "external"}:
            raise ValueError("asset_kind must be 'generated', 'noise92', or 'external'")
        if self.physics_status not in {"approximate", "exact_asset"}:
            raise ValueError("physics_status must be 'approximate' or 'exact_asset'")
        digest = hashlib.sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()
        if self.content_hash is not None and self.content_hash != digest:
            raise ValueError("spatial background content_hash does not match waveform bytes")
        object.__setattr__(self, "waveform", np.ascontiguousarray(value, dtype=np.float32))
        object.__setattr__(self, "source_identity", source_identity.strip())
        object.__setattr__(self, "source", self.source or source_identity.strip())
        object.__setattr__(self, "geometry", dict(self.geometry or {}))
        object.__setattr__(self, "provenance", dict(self.provenance or {}))
        object.__setattr__(self, "content_hash", digest)
        if self.asset_kind == "noise92":
            if self.physics_status != "exact_asset":
                raise ValueError("Noise92 assets must declare physics_status='exact_asset'")
            if self.sample_rate is None:
                raise ValueError("Noise92 assets require sample_rate")
            if not self.geometry:
                raise ValueError("Noise92 assets require geometry")

    @property
    def is_exact_asset(self) -> bool:
        return self.physics_status == "exact_asset"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "source": self.source,
            "source_identity": self.source_identity,
            "coherence_policy": self.coherence_policy,
            "sample_rate": self.sample_rate,
            "geometry": dict(self.geometry or {}),
            "generator_version": self.generator_version,
            "provenance": dict(self.provenance or {}),
            "content_hash": self.content_hash,
            "asset_kind": self.asset_kind,
            "physics_status": self.physics_status,
            "channels": int(self.waveform.shape[0]),
            "num_samples": int(self.waveform.shape[1]),
        }


def validate_spatial_background(background: np.ndarray, *, num_channels: int, length: int) -> np.ndarray:
    """校验空间背景，禁止把 mono additive noise 静默广播为阵列噪声。"""
    value = np.asarray(background)
    if value.dtype != np.float32 or value.ndim != 2:
        raise TypeError("spatial background must be an explicit float32 [C,T] array")
    if value.shape != (int(num_channels), int(length)):
        raise ValueError(
            f"spatial background shape must be {(int(num_channels), int(length))}, got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError("spatial background contains NaN or Inf")
    return value


def validate_spatial_background_capability(task: str, sim_cfg: Any) -> None:
    """Reject spatial branches that are not yet covered by task contracts.

    任务开放清单以 ``core/data/simulation/capabilities.SPATIAL_FEATURE_CAPABILITIES``
    为单一事实源（见该模块 docstring 的接入 checklist）；此处只负责把命中且未开放
    的 feature 转成 fail-fast 错误，配置期（tests/test_configs.py）与 runtime
    （view 投影）报同一信息。
    """
    from core.data.simulation.capabilities import (
        SPATIAL_FEATURE_CAPABILITIES,
        SUPPORTED_SPATIAL_TASKS,
    )

    if not getattr(sim_cfg, "spatial_enabled", False) or task not in SUPPORTED_SPATIAL_TASKS:
        return
    unsupported = [
        "target_type='rts'"
        if capability.feature == "rts_target"
        else capability.config_key
        for capability in SPATIAL_FEATURE_CAPABILITIES
        if capability.is_triggered(sim_cfg) and task not in capability.spatial_tasks
    ]
    if unsupported:
        raise ValueError(
            f"{task} spatial background does not preserve {', '.join(unsupported)}; "
            "set these probabilities to 0 or use a supported mono simulation path"
        )


@dataclass(frozen=True)
class SpatialBackgroundAssetSpec:
    """不解码音频的空间背景 manifest 记录。"""

    path: str
    source_identity: str
    sample_rate: int
    geometry: Mapping[str, Any]
    coherence_policy: str
    generator_version: str
    provenance: Mapping[str, Any]
    content_hash: str | None = None
    asset_kind: str = "external"
    physics_status: str = "approximate"


def parse_spatial_background_manifest_record(record: Mapping[str, Any], *, location: str = "record") -> SpatialBackgroundAssetSpec:
    """解析空间背景资产记录；Noise92 缺少 exact 合同时 fail-fast。"""
    if not isinstance(record, Mapping):
        raise TypeError(f"{location}: spatial background record must be a mapping")
    required = ("path", "source_identity", "sample_rate", "geometry", "coherence_policy", "generator_version", "provenance")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"{location}: spatial background record missing {missing!r}")
    asset_kind = record.get("asset_kind", "external")
    physics_status = record.get("physics_status", "approximate")
    if asset_kind == "noise92" and physics_status != "exact_asset":
        raise ValueError(f"{location}: Noise92 asset must declare physics_status='exact_asset'")
    if not isinstance(record["path"], str) or not record["path"].strip():
        raise ValueError(f"{location}: path must be a non-empty string")
    if type(record["sample_rate"]) is not int or record["sample_rate"] <= 0:
        raise ValueError(f"{location}: sample_rate must be a positive int")
    if not isinstance(record["geometry"], Mapping) or not record["geometry"]:
        raise ValueError(f"{location}: geometry must be a non-empty mapping")
    for key in ("source_identity", "coherence_policy", "generator_version"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{location}: {key} must be a non-empty string")
    if not isinstance(record["provenance"], Mapping) or not record["provenance"]:
        raise ValueError(f"{location}: provenance must be a non-empty mapping")
    content_hash = record.get("content_hash")
    if content_hash is not None and (not isinstance(content_hash, str) or len(content_hash) != 64):
        raise ValueError(f"{location}: content_hash must be a SHA-256 hex string")
    if asset_kind not in {"external", "noise92"}:
        raise ValueError(f"{location}: unsupported asset_kind {asset_kind!r}")
    if physics_status not in {"approximate", "exact_asset"}:
        raise ValueError(f"{location}: unsupported physics_status {physics_status!r}")
    return SpatialBackgroundAssetSpec(
        path=record["path"].strip(), source_identity=record["source_identity"].strip(),
        sample_rate=record["sample_rate"], geometry=dict(record["geometry"]),
        coherence_policy=record["coherence_policy"].strip(),
        generator_version=record["generator_version"].strip(),
        provenance=dict(record["provenance"]), content_hash=content_hash,
        asset_kind=asset_kind, physics_status=physics_status,
    )


def _spherical_coherence(
    positions: np.ndarray, sample_rate: int, n_fft: int, sound_speed_mps: float
) -> np.ndarray:
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1
    )
    return np.sinc(
        2.0 * frequencies[:, None, None] * distances[None, :, :] / sound_speed_mps
    )


def _coherence_factors(coherence: np.ndarray) -> np.ndarray:
    factors = np.empty(coherence.shape, dtype=np.complex128)
    for index, matrix in enumerate(coherence):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        factors[index] = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]
    return factors


def _independent_spectral_sources(
    source_spectrum: np.ndarray,
    channels: int,
    rng: np.random.Generator,
    n_fft: int,
) -> np.ndarray:
    bins, frames = source_spectrum.shape
    magnitude = np.abs(source_spectrum)
    phase = rng.uniform(-np.pi, np.pi, size=(channels, bins, frames))
    independent = magnitude[None] * np.exp(1j * phase)
    independent[:, 0, :] = magnitude[0][None] * rng.choice(
        (-1.0, 1.0), size=(channels, frames)
    )
    if n_fft % 2 == 0:
        independent[:, -1, :] = magnitude[-1][None] * rng.choice(
            (-1.0, 1.0), size=(channels, frames)
        )
    return independent


def render_diffuse_background(
    mono: np.ndarray,
    microphone_positions: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    sound_speed_mps: float = 343.0,
    source_identity: str | None = None,
    coherence_policy: str = "habets_spherical",
    provenance: Mapping[str, Any] | None = None,
    generator_version: str = "spatial-diffuse-habets-spherical-v1",
    reference_channel: int = 0,
) -> SpatialBackground:
    """用 Habets spherical sinc coherence 将 mono noise 渲染为 ``[C,T]``。"""
    from scipy.signal import istft, stft

    source = np.asarray(mono, dtype=np.float32).reshape(-1)
    positions = np.asarray(microphone_positions, dtype=np.float64)
    if source.size == 0:
        raise ValueError("diffuse background source must be non-empty")
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 1:
        raise ValueError("microphone_positions must have shape [C,3]")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive exact int")
    if not np.isfinite(sound_speed_mps) or float(sound_speed_mps) <= 0:
        raise ValueError("sound_speed_mps must be positive and finite")
    if coherence_policy != "habets_spherical":
        raise ValueError("coherence_policy must be 'habets_spherical'")
    if not isinstance(reference_channel, (int, np.integer)) or isinstance(reference_channel, bool):
        raise TypeError("reference_channel must be an integer")
    if not 0 <= int(reference_channel) < positions.shape[0]:
        raise ValueError("reference_channel must be in [0, C)")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if not np.isfinite(source).all() or not np.isfinite(positions).all():
        raise ValueError("diffuse background input and geometry must be finite")

    channels = int(positions.shape[0])
    output = np.zeros((channels, source.size), dtype=np.float32)
    if channels == 1:
        output[0] = source
    else:
        n_fft = min(512, 1 << max(5, source.size.bit_length() - 1))
        stft_input = np.pad(source, (0, max(0, n_fft - source.size)))
        hop = n_fft // 4
        _, _, source_spectrum = stft(
            stft_input,
            fs=sample_rate,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            boundary="zeros",
            padded=True,
        )
        independent = _independent_spectral_sources(
            source_spectrum, channels, rng, n_fft
        )
        coherence = _spherical_coherence(
            positions, sample_rate, n_fft, float(sound_speed_mps)
        )
        factors = _coherence_factors(coherence)
        output_spectrum = np.einsum(
            "fck,kft->cft", factors, independent, optimize=True
        )
        for channel in range(channels):
            _, waveform = istft(
                output_spectrum[channel],
                fs=sample_rate,
                window="hann",
                nperseg=n_fft,
                noverlap=n_fft - hop,
                nfft=n_fft,
                input_onesided=True,
                boundary=True,
            )
            if waveform.size < source.size:
                waveform = np.pad(waveform, (0, source.size - waveform.size))
            output[channel] = waveform[: source.size].astype(np.float32)

    rms_in = float(np.sqrt(np.mean(source.astype(np.float64) ** 2)))
    reference_rms = float(
        np.sqrt(np.mean(output[int(reference_channel)].astype(np.float64) ** 2))
    )
    if rms_in > 1e-8:
        if reference_rms <= 1e-8:
            raise ValueError(
                "diffuse rendering produced a silent reference channel; "
                "cannot satisfy reference-channel SNR"
            )
        output *= np.float32(rms_in / reference_rms)
    source_hash = hashlib.sha256(np.ascontiguousarray(source, dtype="<f4").tobytes()).hexdigest()
    return SpatialBackground(
        np.ascontiguousarray(output, dtype=np.float32),
        source="inline:" + source_hash,
        source_identity=source_identity or ("inline:" + source_hash),
        coherence_policy=coherence_policy,
        sample_rate=sample_rate,
        geometry={
            "microphone_positions_m": positions.tolist(),
            "sound_speed_mps": float(sound_speed_mps),
        },
        generator_version=generator_version,
        provenance={
            **dict(provenance or {}),
            "spatial_model": "habets_spherical_sinc",
            "approximation": True,
        },
        asset_kind="generated",
        physics_status="approximate",
    )


__all__ = [
    "SpatialBackground",
    "SpatialBackgroundAssetSpec",
    "parse_spatial_background_manifest_record",
    "validate_spatial_background",
    "render_diffuse_background",
]
