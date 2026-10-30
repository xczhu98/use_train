"""流式 ONNX 的单一运行时 ABI 契约与 wrapper adapter。

export、ORT inference 与 three-way verifier 都应从本模块取得模型 I/O、
STFT 和 cache metadata 契约，避免各入口分别维护同一组名称和 shape。
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Mapping, Sequence

import torch


RUNTIME_METADATA_KEY = "streaming_runtime_spec"
RUNTIME_METADATA_VERSION = 1
CAGCRN_RUNTIME_METADATA_KEY = "cagcrn_runtime_spec"
CAGCRN_RUNTIME_METADATA_VERSION = 2
CAGCRN_CACHE_ABI_VERSION = 2
CAGCRN_CACHE_NAMES = tuple(f"cache_{index}" for index in range(40))
# CAGCRN delay_d 随模型 align_delay 配置变化。
CAGCRN_MAX_DELAY_D = 4096
DEEPVQE_RUNTIME_METADATA_KEY = "deepvqe_runtime_spec"
DEEPVQE_RUNTIME_METADATA_VERSION = 2
DBGNET_RUNTIME_METADATA_KEY = "dbgnet_runtime_spec"
DBGNET_RUNTIME_METADATA_VERSION = 3
DBGNET_CACHE_ABI_VERSION = 2
DBGNET_CACHE_SEMANTICS = "recursive_guidance_v1"
DBGNET_MODEL_FINGERPRINT_ALGORITHM = "sha256-state-dict-framed-v1"
DBGNET_BASE_CACHE_NAMES = (
    "mic_in", "far_in", "gy1_in", "gy2_in", "gy3_in",
    "gx1_in", "gx2_in", "gx3_in", "ftg_h1", "ftg_h2",
    "ftg_h3", "ftg_h4", "tgru_h", "d4_spill", "d3_spill",
    "d2_spill", "d1_spill",
)
DEPLOYMENT_CONFIG_METADATA_KEY = "deployment_config_binding"
DEPLOYMENT_CONFIG_METADATA_VERSION = 2
RESOLVED_CONFIG_FINGERPRINT_ALGORITHM = "sha256-json-v1"
TSE_RUNTIME_METADATA_KEY = "tse_runtime_spec"
TSE_RUNTIME_METADATA_VERSION = 2
TSE_CACHE_ABI_VERSION = 1
TSE_CACHE_SEMANTICS = "fixed-enrollment-loopback-v1"
ONLINE_SPATIALNET_RUNTIME_METADATA_KEY = "online_spatialnet_runtime_spec"
ONLINE_SPATIALNET_RUNTIME_METADATA_VERSION = 1
# v2：mhsa 分支改为增量 K/V cache，ret 分支为 recurrent 状态，mamba 分支为 conv/ssm state。
ONLINE_SPATIALNET_CACHE_ABI_VERSION = 2
ONLINE_SPATIALNET_CACHE_SEMANTICS = "causal-branch-state-v2"


def _require_metadata_fields(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise RuntimeError(
            f"{label} metadata fields mismatch: expected={sorted(expected)}, got={got}"
        )
    return value


def _require_metadata_int(
    value: object,
    name: str,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(
            f"{label} metadata {name} must be an integer >= {minimum}, got {value!r}"
        )
    if maximum is not None and value > maximum:
        raise RuntimeError(
            f"{label} metadata {name} must be an integer <= {maximum}, got {value!r}"
        )
    return value


@dataclass(frozen=True)
class CAGCRNRuntimeSpec:
    """CAGCRN 的历史窗口、零输出延迟与 canonical 40-cache ABI 契约。"""

    delay_d: int
    algorithmic_delay_frames: int
    cache_abi_version: int
    cache_names: tuple[str, ...]

    def __post_init__(self) -> None:
        delay_d = self.delay_d
        algorithmic_delay_frames = self.algorithmic_delay_frames
        if (
            isinstance(delay_d, bool)
            or not isinstance(delay_d, int)
            or delay_d < 1
            or delay_d > CAGCRN_MAX_DELAY_D
        ):
            raise ValueError(
                f"CAGCRN delay_d must be an integer in [1, {CAGCRN_MAX_DELAY_D}], "
                f"got {delay_d!r}"
            )
        if (
            isinstance(algorithmic_delay_frames, bool)
            or not isinstance(algorithmic_delay_frames, int)
            or algorithmic_delay_frames != 0
        ):
            raise ValueError(
                "CAGCRN streaming output algorithmic delay must be zero, "
                f"got {algorithmic_delay_frames!r}"
            )
        if self.cache_abi_version != CAGCRN_CACHE_ABI_VERSION:
            raise ValueError(
                f"Unsupported CAGCRN cache ABI version: {self.cache_abi_version!r}"
            )
        if self.cache_names != CAGCRN_CACHE_NAMES:
            raise ValueError(
                "CAGCRN cache schema mismatch: "
                f"expected={CAGCRN_CACHE_NAMES}, got={self.cache_names}"
            )

    def to_metadata(self) -> dict[str, object]:
        return {
            "version": CAGCRN_RUNTIME_METADATA_VERSION,
            "delay_d": self.delay_d,
            "algorithmic_delay_frames": self.algorithmic_delay_frames,
            "cache_abi_version": self.cache_abi_version,
            "cache_names": list(self.cache_names),
        }

    @classmethod
    def from_metadata(cls, value: object) -> "CAGCRNRuntimeSpec":
        fields = {
            "version", "delay_d", "algorithmic_delay_frames", "cache_abi_version",
            "cache_names",
        }
        value = _require_metadata_fields(value, fields, "CAGCRN runtime")
        if value["version"] != CAGCRN_RUNTIME_METADATA_VERSION:
            raise RuntimeError(
                f"Unsupported CAGCRN runtime metadata version: {value['version']!r}"
            )
        delay_d = _require_metadata_int(
            value["delay_d"], "delay_d", "CAGCRN runtime",
            minimum=1, maximum=CAGCRN_MAX_DELAY_D,
        )
        algorithmic_delay_frames = _require_metadata_int(
            value["algorithmic_delay_frames"],
            "algorithmic_delay_frames",
            "CAGCRN runtime",
        )
        if algorithmic_delay_frames != 0:
            raise RuntimeError(
                "CAGCRN streaming output algorithmic delay must be zero, "
                f"got {algorithmic_delay_frames!r}"
            )
        cache_abi_version = _require_metadata_int(
            value["cache_abi_version"], "cache_abi_version", "CAGCRN runtime", minimum=1,
        )
        if cache_abi_version != CAGCRN_CACHE_ABI_VERSION:
            raise RuntimeError(
                f"Unsupported CAGCRN cache ABI version: {cache_abi_version!r}"
            )
        cache_names_value = value["cache_names"]
        if (
            not isinstance(cache_names_value, list)
            or tuple(cache_names_value) != CAGCRN_CACHE_NAMES
        ):
            raise RuntimeError(
                "CAGCRN runtime metadata cache schema mismatch: "
                f"expected={CAGCRN_CACHE_NAMES}, got={cache_names_value!r}"
            )
        return cls(
            delay_d=delay_d,
            algorithmic_delay_frames=algorithmic_delay_frames,
            cache_abi_version=cache_abi_version,
            cache_names=CAGCRN_CACHE_NAMES,
        )


@dataclass(frozen=True)
class STFTRuntimeSpec:
    """图外 STFT 参数；``None`` 表示模型未声明该项。"""

    n_fft: int | None = None
    hop_length: int | None = None
    win_length: int | None = None
    sample_rate: int | None = None
    window: str = "hann"
    center: bool | None = None

    def to_metadata(self) -> dict[str, object]:
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "sample_rate": self.sample_rate,
            "window": self.window,
            "center": self.center,
        }

    @classmethod
    def from_metadata(cls, value: object) -> "STFTRuntimeSpec":
        if not isinstance(value, dict):
            raise RuntimeError("streaming runtime STFT metadata must be an object")
        expected = {"n_fft", "hop_length", "win_length", "sample_rate", "window", "center"}
        if set(value) != expected:
            raise RuntimeError(
                f"streaming runtime STFT metadata fields mismatch: expected={sorted(expected)}, "
                f"got={sorted(value)}"
            )
        integers: dict[str, int | None] = {}
        for name in ("n_fft", "hop_length", "win_length", "sample_rate"):
            item = value[name]
            if item is not None and (isinstance(item, bool) or not isinstance(item, int) or item <= 0):
                raise RuntimeError(f"streaming runtime STFT {name} must be a positive integer or null")
            integers[name] = item
        window = value["window"]
        if not isinstance(window, str) or not window:
            raise RuntimeError("streaming runtime STFT window must be a non-empty string")
        center = value["center"]
        if center is not None and not isinstance(center, bool):
            raise RuntimeError("streaming runtime STFT center must be boolean or null")
        return cls(**integers, window=window, center=center)

    def require_complete(self, *, label: str = "streaming runtime") -> "STFTRuntimeSpec":
        """拒绝无法在图外重建与训练一致前端的 STFT metadata。"""
        required = ("n_fft", "hop_length", "win_length", "sample_rate", "center")
        missing = tuple(name for name in required if getattr(self, name) is None)
        if missing:
            raise RuntimeError(f"{label} STFT metadata is incomplete: {missing}")
        if self.win_length > self.n_fft:
            raise RuntimeError(
                f"{label} STFT win_length exceeds n_fft: {self.win_length} > {self.n_fft}"
            )
        if self.window != "hann":
            raise RuntimeError(f"{label} STFT window must be 'hann', got {self.window!r}")
        return self


@dataclass(frozen=True)
class TSERuntimeSpec:
    """TSE enrollment 绑定、cache 顺序与 waveform 前端延迟合同。"""

    cache_abi_version: int
    cache_semantics: str
    condition_names: tuple[str, ...]
    cache_names: tuple[str, ...]
    separator_layers: int
    nband: int
    separator_hidden_size: int
    analysis_lookahead_samples: int
    waveform_output_delay_samples: int
    model_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.model_sha256, str) or len(self.model_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.model_sha256)):
            raise ValueError("TSE model_sha256 must be a lowercase SHA-256")
        if self.cache_abi_version != TSE_CACHE_ABI_VERSION:
            raise ValueError(
                f"Unsupported TSE cache ABI version: {self.cache_abi_version!r}"
            )
        if self.cache_semantics != TSE_CACHE_SEMANTICS:
            raise ValueError(f"Unsupported TSE cache semantics: {self.cache_semantics!r}")
        if len(set(self.condition_names)) != len(self.condition_names):
            raise ValueError("TSE condition_names must be unique")
        if len(set(self.cache_names)) != len(self.cache_names):
            raise ValueError("TSE cache_names must be unique")
        if self.cache_names[: len(self.condition_names)] != self.condition_names:
            raise ValueError("TSE condition caches must be the cache ABI prefix")
        for name, value in (
            ("separator_layers", self.separator_layers),
            ("nband", self.nband),
            ("separator_hidden_size", self.separator_hidden_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"TSE {name} must be a positive integer")
        for name, value in (
            ("analysis_lookahead_samples", self.analysis_lookahead_samples),
            ("waveform_output_delay_samples", self.waveform_output_delay_samples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"TSE {name} must be a non-negative integer")
        if self.waveform_output_delay_samples < self.analysis_lookahead_samples:
            raise ValueError("TSE waveform delay cannot be smaller than STFT lookahead")

    def to_metadata(self) -> dict[str, object]:
        return {
            "version": TSE_RUNTIME_METADATA_VERSION,
            "cache_abi_version": self.cache_abi_version,
            "cache_semantics": self.cache_semantics,
            "condition_names": list(self.condition_names),
            "cache_names": list(self.cache_names),
            "separator_layers": self.separator_layers,
            "nband": self.nband,
            "separator_hidden_size": self.separator_hidden_size,
            "analysis_lookahead_samples": self.analysis_lookahead_samples,
            "waveform_output_delay_samples": self.waveform_output_delay_samples,
            "model_sha256": self.model_sha256,
        }

    @classmethod
    def from_metadata(cls, value: object) -> "TSERuntimeSpec":
        fields = {
            "version", "cache_abi_version", "cache_semantics",
            "condition_names", "cache_names", "separator_layers", "nband",
            "separator_hidden_size", "analysis_lookahead_samples",
            "waveform_output_delay_samples",
            "model_sha256",
        }
        value = _require_metadata_fields(value, fields, "TSE runtime")
        if value["version"] != TSE_RUNTIME_METADATA_VERSION:
            raise RuntimeError(
                f"Unsupported TSE runtime metadata version: {value['version']!r}"
            )
        for name in ("condition_names", "cache_names"):
            names = value[name]
            if (
                not isinstance(names, list)
                or any(not isinstance(item, str) or not item for item in names)
            ):
                raise RuntimeError(f"TSE runtime metadata {name} must be a string list")
        return cls(
            cache_abi_version=_require_metadata_int(
                value["cache_abi_version"], "cache_abi_version", "TSE runtime", minimum=1
            ),
            cache_semantics=value["cache_semantics"],
            condition_names=tuple(value["condition_names"]),
            cache_names=tuple(value["cache_names"]),
            separator_layers=_require_metadata_int(
                value["separator_layers"], "separator_layers", "TSE runtime", minimum=1
            ),
            nband=_require_metadata_int(value["nband"], "nband", "TSE runtime", minimum=1),
            separator_hidden_size=_require_metadata_int(
                value["separator_hidden_size"],
                "separator_hidden_size",
                "TSE runtime",
                minimum=1,
            ),
            analysis_lookahead_samples=_require_metadata_int(
                value["analysis_lookahead_samples"],
                "analysis_lookahead_samples",
                "TSE runtime",
            ),
            waveform_output_delay_samples=_require_metadata_int(
                value["waveform_output_delay_samples"],
                "waveform_output_delay_samples",
                "TSE runtime",
            ),
            model_sha256=value["model_sha256"],
        )


@dataclass(frozen=True)
class OnlineSpatialNetRuntimeSpec:
    """OnlineSpatialNet 的版本化 cache 顺序与多通道 observation ABI。"""

    cache_abi_version: int
    cache_semantics: str
    n_channel: int
    n_fft: int
    cache_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.cache_abi_version != ONLINE_SPATIALNET_CACHE_ABI_VERSION:
            raise ValueError(
                f"Unsupported OnlineSpatialNet cache ABI version: {self.cache_abi_version!r}"
            )
        if self.cache_semantics != ONLINE_SPATIALNET_CACHE_SEMANTICS:
            raise ValueError(
                f"Unsupported OnlineSpatialNet cache semantics: {self.cache_semantics!r}"
            )
        for name, value in (("n_channel", self.n_channel), ("n_fft", self.n_fft)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"OnlineSpatialNet {name} must be a positive integer")
        if self.n_fft < 2:
            raise ValueError("OnlineSpatialNet n_fft must be >= 2")
        if not self.cache_names or len(set(self.cache_names)) != len(self.cache_names):
            raise ValueError("OnlineSpatialNet cache_names must be non-empty and unique")

    def to_metadata(self) -> dict[str, object]:
        return {
            "version": ONLINE_SPATIALNET_RUNTIME_METADATA_VERSION,
            "cache_abi_version": self.cache_abi_version,
            "cache_semantics": self.cache_semantics,
            "n_channel": self.n_channel,
            "n_fft": self.n_fft,
            "cache_names": list(self.cache_names),
        }

    @classmethod
    def from_metadata(cls, value: object) -> "OnlineSpatialNetRuntimeSpec":
        fields = {
            "version", "cache_abi_version", "cache_semantics", "n_channel",
            "n_fft", "cache_names",
        }
        value = _require_metadata_fields(value, fields, "OnlineSpatialNet runtime")
        if value["version"] != ONLINE_SPATIALNET_RUNTIME_METADATA_VERSION:
            raise RuntimeError(
                "Unsupported OnlineSpatialNet runtime metadata version: "
                f"{value['version']!r}"
            )
        names = value["cache_names"]
        if not isinstance(names, list) or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise RuntimeError("OnlineSpatialNet runtime cache_names must be a string list")
        return cls(
            cache_abi_version=_require_metadata_int(
                value["cache_abi_version"], "cache_abi_version", "OnlineSpatialNet runtime", minimum=1
            ),
            cache_semantics=value["cache_semantics"],
            n_channel=_require_metadata_int(
                value["n_channel"], "n_channel", "OnlineSpatialNet runtime", minimum=1
            ),
            n_fft=_require_metadata_int(
                value["n_fft"], "n_fft", "OnlineSpatialNet runtime", minimum=2
            ),
            cache_names=tuple(names),
        )


@dataclass(frozen=True)
class DeepVQERuntimeSpec:
    """DeepVQE 对齐历史、零输出 lookahead 与 cache 顺序部署契约。"""

    variant: str
    compression: float
    alignment_window_frames: int
    alignment_history_frames: int
    algorithmic_delay_frames: int
    algorithmic_delay_samples: int
    cache_names: tuple[str, ...]

    def to_metadata(self) -> dict[str, object]:
        return {
            "version": DEEPVQE_RUNTIME_METADATA_VERSION,
            **{name: value for name, value in self.__dict__.items() if name != "cache_names"},
            "cache_names": list(self.cache_names),
        }

    @classmethod
    def from_metadata(cls, value: object) -> "DeepVQERuntimeSpec":
        if not isinstance(value, dict):
            raise RuntimeError("DeepVQE runtime metadata must be an object")
        if value.get("version") != DEEPVQE_RUNTIME_METADATA_VERSION:
            raise RuntimeError(
                f"Unsupported DeepVQE runtime metadata version: {value.get('version')!r}"
            )
        fields = {
            "version", "variant", "compression", "alignment_window_frames",
            "alignment_history_frames", "algorithmic_delay_frames",
            "algorithmic_delay_samples", "cache_names",
        }
        value = _require_metadata_fields(value, fields, "DeepVQE runtime")
        variant = value["variant"]
        if variant not in {"base", "small"}:
            raise RuntimeError(f"DeepVQE runtime metadata variant is invalid: {variant!r}")
        compression = value["compression"]
        if isinstance(compression, bool) or not isinstance(compression, (int, float)):
            raise RuntimeError("DeepVQE runtime metadata compression must be finite and positive")
        compression = float(compression)
        if not math.isfinite(compression) or compression <= 0:
            raise RuntimeError("DeepVQE runtime metadata compression must be finite and positive")
        window = _require_metadata_int(
            value["alignment_window_frames"], "alignment_window_frames",
            "DeepVQE runtime", minimum=1,
        )
        history = _require_metadata_int(
            value["alignment_history_frames"], "alignment_history_frames",
            "DeepVQE runtime",
        )
        delay_frames = _require_metadata_int(
            value["algorithmic_delay_frames"], "algorithmic_delay_frames",
            "DeepVQE runtime",
        )
        delay_samples = _require_metadata_int(
            value["algorithmic_delay_samples"], "algorithmic_delay_samples",
            "DeepVQE runtime",
        )
        cache_names_value = value["cache_names"]
        if (
            not isinstance(cache_names_value, list)
            or not cache_names_value
            or any(not isinstance(name, str) or not name for name in cache_names_value)
            or len(set(cache_names_value)) != len(cache_names_value)
        ):
            raise RuntimeError(
                "DeepVQE runtime metadata cache_names must be a non-empty unique string list"
            )
        if history != window - 1:
            raise RuntimeError("DeepVQE alignment history must equal window minus one")
        if delay_frames != 0 or delay_samples != 0:
            raise RuntimeError("DeepVQE streaming output algorithmic delay must be zero")
        return cls(
            variant, compression, window, history, delay_frames, delay_samples,
            tuple(cache_names_value),
        )


def dbgnet_cache_names(use_state_detector: bool) -> tuple[str, ...]:
    """返回唯一受支持的 DBGNet cache ABI 顺序。"""
    if not isinstance(use_state_detector, bool):
        raise TypeError("use_state_detector must be boolean")
    return (*DBGNET_BASE_CACHE_NAMES, "state_h") if use_state_detector else DBGNET_BASE_CACHE_NAMES


def _sha256_frame(digest, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def dbgnet_model_fingerprint(model: torch.nn.Module) -> dict[str, object]:
    """按 key/dtype/shape/raw-bytes 长度分帧计算 DBGNet 内容身份。"""
    return _model_state_fingerprint(model, label="DBGNet")


def _model_state_fingerprint(model: torch.nn.Module, *, label: str) -> dict[str, object]:
    """绑定实际加载后的全部参数和 buffer，独立于 checkpoint 容器格式。"""
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{label} model fingerprint requires a torch.nn.Module")
    state = model.state_dict()
    digest = hashlib.sha256()
    byte_count = 0
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise TypeError(f"{label} state entry {key!r} must be a strided tensor")
        value = tensor.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        _sha256_frame(digest, key.encode("utf-8"))
        _sha256_frame(digest, str(value.dtype).encode("ascii"))
        _sha256_frame(
            digest,
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"),
        )
        _sha256_frame(digest, raw)
        byte_count += len(raw)
    return {
        "algorithm": DBGNET_MODEL_FINGERPRINT_ALGORITHM,
        "sha256": digest.hexdigest(),
        "tensor_count": len(state),
        "byte_count": byte_count,
    }


def _parse_dbgnet_model_fingerprint(value: object) -> dict[str, object]:
    value = _require_metadata_fields(
        value, {"algorithm", "sha256", "tensor_count", "byte_count"},
        "DBGNet model fingerprint",
    )
    if value["algorithm"] != DBGNET_MODEL_FINGERPRINT_ALGORITHM:
        raise RuntimeError(
            "Unsupported DBGNet model fingerprint algorithm: "
            f"{value['algorithm']!r}"
        )
    sha256 = value["sha256"]
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(char not in "0123456789abcdef" for char in sha256)
    ):
        raise RuntimeError("DBGNet model fingerprint SHA-256 is invalid")
    tensor_count = _require_metadata_int(
        value["tensor_count"], "tensor_count", "DBGNet model fingerprint", minimum=1
    )
    byte_count = _require_metadata_int(
        value["byte_count"], "byte_count", "DBGNet model fingerprint", minimum=1
    )
    return {
        "algorithm": value["algorithm"],
        "sha256": sha256,
        "tensor_count": tensor_count,
        "byte_count": byte_count,
    }


@dataclass(frozen=True)
class DBGNetRuntimeSpec:
    """DBGNet state-head、帧时序、cache 与模型内容身份部署契约。"""

    use_state_detector: bool
    state_nclass: int
    network_lookahead_frames: int
    algorithmic_delay_frames: int
    algorithmic_delay_samples: int
    stft_lookahead_samples: int
    frame_shift_samples: int
    frame_shift_ms: float
    cache_abi_version: int
    cache_semantics: str
    cache_names: tuple[str, ...]
    model_fingerprint: Mapping[str, object]

    def __post_init__(self) -> None:
        expected = dbgnet_cache_names(self.use_state_detector)
        if self.cache_names != expected:
            raise ValueError(
                f"DBGNet cache schema mismatch: expected {expected}, got {self.cache_names}"
            )
        fingerprint = _parse_dbgnet_model_fingerprint(dict(self.model_fingerprint))
        object.__setattr__(self, "model_fingerprint", MappingProxyType(fingerprint))

    @property
    def model_identity(self) -> str:
        return f"dbgnet:{self.model_fingerprint['sha256']}"

    def to_metadata(self) -> dict[str, object]:
        return {
            "version": DBGNET_RUNTIME_METADATA_VERSION,
            "use_state_detector": self.use_state_detector,
            "state_nclass": self.state_nclass,
            "network_lookahead_frames": self.network_lookahead_frames,
            "algorithmic_delay_frames": self.algorithmic_delay_frames,
            "algorithmic_delay_samples": self.algorithmic_delay_samples,
            "stft_lookahead_samples": self.stft_lookahead_samples,
            "frame_shift_samples": self.frame_shift_samples,
            "frame_shift_ms": self.frame_shift_ms,
            "cache_abi_version": self.cache_abi_version,
            "cache_semantics": self.cache_semantics,
            "cache_names": list(self.cache_names),
            "model_fingerprint": dict(self.model_fingerprint),
        }

    @classmethod
    def from_metadata(cls, value: object) -> "DBGNetRuntimeSpec":
        if not isinstance(value, dict) or value.get("version") != DBGNET_RUNTIME_METADATA_VERSION:
            version = value.get("version") if isinstance(value, dict) else None
            raise RuntimeError(
                f"Unsupported DBGNet runtime metadata version: {version!r}"
            )
        fields = {
            "version", "use_state_detector", "state_nclass",
            "network_lookahead_frames", "algorithmic_delay_frames",
            "algorithmic_delay_samples", "stft_lookahead_samples",
            "frame_shift_samples", "frame_shift_ms", "cache_abi_version",
            "cache_semantics", "cache_names", "model_fingerprint",
        }
        value = _require_metadata_fields(value, fields, "DBGNet runtime")
        use_state = value["use_state_detector"]
        if not isinstance(use_state, bool):
            raise RuntimeError("DBGNet runtime metadata use_state_detector must be boolean")
        integer_names = (
            "state_nclass", "network_lookahead_frames", "algorithmic_delay_frames",
            "algorithmic_delay_samples", "stft_lookahead_samples", "frame_shift_samples",
        )
        parsed = {
            name: _require_metadata_int(
                value[name], name, "DBGNet runtime",
                minimum=1 if name in {"state_nclass", "frame_shift_samples"} else 0,
            )
            for name in integer_names
        }
        cache_abi_version = _require_metadata_int(
            value["cache_abi_version"], "cache_abi_version", "DBGNet runtime", minimum=1
        )
        if cache_abi_version != DBGNET_CACHE_ABI_VERSION:
            raise RuntimeError(
                f"Unsupported DBGNet cache ABI version: {cache_abi_version!r}"
            )
        cache_semantics = value["cache_semantics"]
        if cache_semantics != DBGNET_CACHE_SEMANTICS:
            raise RuntimeError(f"Unsupported DBGNet cache semantics: {cache_semantics!r}")
        cache_names_value = value["cache_names"]
        if not isinstance(cache_names_value, list):
            raise RuntimeError("DBGNet runtime metadata cache_names must be a list")
        expected_cache_names = dbgnet_cache_names(use_state)
        if tuple(cache_names_value) != expected_cache_names:
            raise RuntimeError(
                "DBGNet runtime metadata cache schema mismatch: "
                f"expected={expected_cache_names}, got={tuple(cache_names_value)}"
            )
        fingerprint = _parse_dbgnet_model_fingerprint(value["model_fingerprint"])
        frame_shift_ms = value["frame_shift_ms"]
        if isinstance(frame_shift_ms, bool) or not isinstance(frame_shift_ms, (int, float)):
            raise RuntimeError("DBGNet runtime metadata frame_shift_ms must be finite and positive")
        frame_shift_ms = float(frame_shift_ms)
        if not math.isfinite(frame_shift_ms) or frame_shift_ms <= 0:
            raise RuntimeError("DBGNet runtime metadata frame_shift_ms must be finite and positive")
        if parsed["network_lookahead_frames"] != 0:
            raise RuntimeError("DBGNet network lookahead must be zero")
        if parsed["algorithmic_delay_frames"] != 0 or parsed["algorithmic_delay_samples"] != 0:
            raise RuntimeError("DBGNet streaming output algorithmic delay must be zero")
        return cls(
            use_state_detector=use_state,
            **parsed,
            frame_shift_ms=frame_shift_ms,
            cache_abi_version=cache_abi_version,
            cache_semantics=cache_semantics,
            cache_names=expected_cache_names,
            model_fingerprint=fingerprint,
        )


def _canonical_config_sha256(config: Mapping[str, object], *, label: str) -> str:
    try:
        canonical = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains non-canonical values") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def deployment_config_binding(config: Mapping[str, object], model_id: str) -> dict[str, object]:
    """绑定完整 resolved config 指纹及部署语义子集。"""
    resolved_id = normalize_model_id(model_id)
    if resolved_id not in {"deepvqe", "dbgnet"}:
        raise ValueError(f"Deployment config binding is unsupported for {resolved_id!r}")
    model_cfg = config.get("model")
    stft_cfg = config.get("stft")
    data_cfg = config.get("data")
    if not isinstance(model_cfg, Mapping) or not isinstance(stft_cfg, Mapping):
        raise ValueError("Deployment config requires model and stft mappings")
    if not isinstance(data_cfg, Mapping):
        raise ValueError("Deployment config requires a data mapping")
    expected_type = "DeepVQE" if resolved_id == "deepvqe" else "DBGNet"
    if model_cfg.get("type") != expected_type:
        raise ValueError(
            f"{resolved_id} deployment config requires model.type={expected_type}"
        )
    stft = {
        "n_fft": stft_cfg.get("n_fft"),
        "hop_length": stft_cfg.get("hop_length"),
        "win_length": stft_cfg.get("win_length"),
        "sample_rate": data_cfg.get("sample_rate"),
        "window": stft_cfg.get("window", "hann"),
        "center": stft_cfg.get("center", True),
    }
    STFTRuntimeSpec.from_metadata(stft)
    if any(stft[name] is None for name in ("n_fft", "hop_length", "win_length", "sample_rate")):
        raise ValueError("Deployment config STFT fields must be complete")
    model_args = dict(model_cfg.get("args") or {})
    if resolved_id == "dbgnet":
        model_sample_rate = model_args.get("sample_rate", stft["sample_rate"])
        if model_sample_rate != stft["sample_rate"]:
            raise ValueError("DBGNet model/data sample_rate mismatch")
        model_args["sample_rate"] = model_sample_rate
    payload = {
        "model_id": resolved_id,
        "stft": stft,
        "model": {"type": expected_type, "args": model_args},
        "aec_frontend": dict(config.get("aec_frontend") or {}),
    }
    resolved_config = {
        "algorithm": RESOLVED_CONFIG_FINGERPRINT_ALGORITHM,
        "sha256": _canonical_config_sha256(config, label="Resolved config"),
    }
    return {
        "version": DEPLOYMENT_CONFIG_METADATA_VERSION,
        "sha256": _canonical_config_sha256(
            {"resolved_config": resolved_config, "config": payload},
            label="Deployment config binding",
        ),
        "resolved_config": resolved_config,
        "config": payload,
    }


def parse_deployment_config_binding(value: object, model_id: str) -> dict[str, object]:
    """严格校验 artifact 中的部署配置、resolved config 指纹及摘要。"""
    value = _require_metadata_fields(
        value,
        {"version", "sha256", "resolved_config", "config"},
        "deployment config binding",
    )
    if value["version"] != DEPLOYMENT_CONFIG_METADATA_VERSION:
        raise RuntimeError(
            f"Unsupported deployment config metadata version: {value['version']!r}"
        )
    resolved_config = _require_metadata_fields(
        value["resolved_config"], {"algorithm", "sha256"},
        "resolved config fingerprint",
    )
    if resolved_config["algorithm"] != RESOLVED_CONFIG_FINGERPRINT_ALGORITHM:
        raise RuntimeError(
            "Unsupported resolved config fingerprint algorithm: "
            f"{resolved_config['algorithm']!r}"
        )
    resolved_sha256 = resolved_config["sha256"]
    if (
        not isinstance(resolved_sha256, str)
        or len(resolved_sha256) != 64
        or any(char not in "0123456789abcdef" for char in resolved_sha256)
    ):
        raise RuntimeError("Resolved config fingerprint SHA-256 is invalid")
    config = value["config"]
    if not isinstance(config, dict) or config.get("model_id") != normalize_model_id(model_id):
        raise RuntimeError("Deployment config binding model_id mismatch")
    # 完整 resolved config 不写入 ONNX；这里只重建并复核部署语义子集摘要。
    reconstructed = {
        "model": config.get("model"),
        "stft": config.get("stft"),
        "data": {"sample_rate": (config.get("stft") or {}).get("sample_rate")},
        "aec_frontend": config.get("aec_frontend"),
    }
    expected_config = deployment_config_binding(reconstructed, model_id)["config"]
    expected_sha256 = _canonical_config_sha256(
        {"resolved_config": dict(resolved_config), "config": expected_config},
        label="Deployment config binding",
    )
    if value["sha256"] != expected_sha256 or config != expected_config:
        raise RuntimeError("Deployment config binding SHA-256 mismatch")
    return value


@dataclass(frozen=True)
class StreamingRuntimeProfile:
    """某类模型稳定的主 I/O 与 metadata 约定。"""

    model_id: str
    primary_inputs: tuple[str, ...]
    primary_output: str
    metadata_key: str | None
    cache_output_style: str = "out_prefix"
    auxiliary_outputs: tuple[str, ...] = ()
    allowed_cache_dynamic_axes: frozenset[str] = frozenset()
    allowed_auxiliary_outputs: frozenset[str] = frozenset()
    static_batch_size: int | None = None

    def cache_output_name(self, cache_input: str) -> str:
        if self.cache_output_style == "out_prefix":
            return f"out_{cache_input}"
        if self.cache_output_style == "out_suffix":
            return f"{cache_input}_out"
        raise ValueError(f"Unsupported cache output style: {self.cache_output_style!r}")


@dataclass(frozen=True)
class StreamingRuntimeSpec:
    """一个具体 streaming artifact 的完整、可校验 ABI。"""

    profile: StreamingRuntimeProfile
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    cache_shapes: Mapping[str, tuple[int, ...]]
    dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)
    stft: STFTRuntimeSpec = field(default_factory=STFTRuntimeSpec)
    cagcrn: CAGCRNRuntimeSpec | None = None
    deepvqe: DeepVQERuntimeSpec | None = None
    dbgnet: DBGNetRuntimeSpec | None = None
    tse: TSERuntimeSpec | None = None
    online_spatialnet: OnlineSpatialNetRuntimeSpec | None = None
    deployment_config: Mapping[str, object] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cache_shapes = {
            str(name): tuple(int(dim) for dim in shape)
            for name, shape in self.cache_shapes.items()
        }
        dynamic_axes = {
            str(name): MappingProxyType(dict(axes))
            for name, axes in self.dynamic_axes.items()
        }
        object.__setattr__(self, "cache_shapes", MappingProxyType(cache_shapes))
        object.__setattr__(self, "dynamic_axes", MappingProxyType(dynamic_axes))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        self.validate()

    @property
    def cache_input_names(self) -> tuple[str, ...]:
        return self.input_names[len(self.profile.primary_inputs):]

    @property
    def cache_output_names(self) -> tuple[str, ...]:
        start = 1 + len(self.profile.auxiliary_outputs)
        return self.output_names[start:]

    def validate(self) -> None:
        if len(set(self.input_names)) != len(self.input_names):
            raise ValueError(f"Duplicate streaming input names: {self.input_names}")
        if len(set(self.output_names)) != len(self.output_names):
            raise ValueError(f"Duplicate streaming output names: {self.output_names}")
        if self.input_names[: len(self.profile.primary_inputs)] != self.profile.primary_inputs:
            raise ValueError(
                f"{self.profile.model_id} primary input ABI mismatch: "
                f"expected {self.profile.primary_inputs}, got {self.input_names}"
            )
        auxiliary_outputs = self.profile.auxiliary_outputs
        unexpected_auxiliary = tuple(
            name
            for name in auxiliary_outputs
            if name not in self.profile.allowed_auxiliary_outputs
        )
        if unexpected_auxiliary:
            raise ValueError(
                f"{self.profile.model_id} has unregistered auxiliary outputs: "
                f"{unexpected_auxiliary}; allowed={tuple(sorted(self.profile.allowed_auxiliary_outputs))}"
            )
        if len(set(auxiliary_outputs)) != len(auxiliary_outputs):
            raise ValueError(f"Duplicate auxiliary output names: {auxiliary_outputs}")
        expected_prefix = (self.profile.primary_output, *auxiliary_outputs)
        if self.output_names[: len(expected_prefix)] != expected_prefix:
            raise ValueError(
                f"{self.profile.model_id} primary output ABI mismatch: "
                f"expected {expected_prefix}, got {self.output_names}"
            )
        cache_inputs = self.cache_input_names
        expected_cache_outputs = tuple(
            self.profile.cache_output_name(name) for name in cache_inputs
        )
        if self.cache_output_names != expected_cache_outputs:
            raise ValueError(
                f"{self.profile.model_id} cache output ABI mismatch: "
                f"expected {expected_cache_outputs}, got {self.cache_output_names}"
            )
        if tuple(self.cache_shapes) != cache_inputs:
            raise ValueError(
                f"{self.profile.model_id} cache config order/names mismatch: "
                f"expected {cache_inputs}, got {tuple(self.cache_shapes)}"
            )
        for name, shape in self.cache_shapes.items():
            if not shape or any(dim < 0 for dim in shape):
                raise ValueError(f"Invalid cache shape for {name!r}: {shape!r}")
        canonical_profile = get_runtime_profile(self.profile.model_id)
        if self.profile.static_batch_size != canonical_profile.static_batch_size:
            raise ValueError(
                f"{self.profile.model_id} static batch contract differs from "
                "the canonical runtime profile"
            )
        static_batch_size = canonical_profile.static_batch_size
        if static_batch_size is not None:
            if self.dynamic_axes:
                raise ValueError(
                    f"{self.profile.model_id} formal ONNX contract requires static "
                    f"batch size {static_batch_size}; dynamic axes are forbidden"
                )
            for name, shape in self.cache_shapes.items():
                if shape[0] != static_batch_size:
                    raise ValueError(
                        f"{self.profile.model_id} cache {name!r} requires static batch "
                        f"size {static_batch_size}, got {shape[0]}"
                    )


_PROFILES = {
    "gtcrn": StreamingRuntimeProfile(
        "gtcrn", ("spec",), "enh", "gtcrn_cache_cfg", "out_suffix"
    ),
    "cagcrn": StreamingRuntimeProfile(
        "cagcrn", ("mic_spec", "ref_spec"), "enh_spec", "cagcrn_cache_cfg",
        allowed_cache_dynamic_axes=frozenset({"time"}),
    ),
    "deepvqe": StreamingRuntimeProfile(
        "deepvqe", ("mic_spec", "far_spec"), "enh_spec", "deepvqe_cache_cfg"
    ),
    "localvqe": StreamingRuntimeProfile(
        "localvqe", ("mic_spec", "far_spec"), "enh_spec", "localvqe_cache_cfg"
    ),
    "fastenhancer": StreamingRuntimeProfile(
        "fastenhancer", ("spec_in",), "spec_out", "fastenhancer_cache_cfg"
    ),
    "ulunas": StreamingRuntimeProfile(
        "ulunas", ("mix",), "enh", "ulunas_cache_cfg", "out_suffix"
    ),
    "ultra_dpc": StreamingRuntimeProfile(
        "ultra_dpc", ("mic_spec", "ref_spec", "err_spec"), "enh_spec",
        "ultra_dpc_cache_cfg",
    ),
    "dbgnet": StreamingRuntimeProfile(
        "dbgnet", ("mic_spec", "far_spec"), "enh", "dbgnet_cache_cfg",
        "out_suffix",
        allowed_auxiliary_outputs=frozenset({"state_probs"}),
        static_batch_size=1,
    ),
    "bsrnn": StreamingRuntimeProfile(
        "bsrnn", ("frame_spec",), "enhanced_spec", "bsrnn_cache_cfg", "out_suffix"
    ),
    "tse": StreamingRuntimeProfile(
        "tse", ("mix_frame",), "enhanced_frame", "tse_cache_cfg", "out_suffix",
        allowed_cache_dynamic_axes=frozenset({"enrollment_time"}),
    ),
    "online_spatialnet": StreamingRuntimeProfile(
        "online_spatialnet", ("mixture_spec",), "enhanced_spec",
        "online_spatialnet_cache_cfg", "out_suffix",
    ),
}

_ALIASES = {
    "streaming_gtcrn": "gtcrn",
    "streaming_cagcrn": "cagcrn",
    "streaming_deepvqe": "deepvqe",
    "streaming_localvqe": "localvqe",
    "streaming_fastenhancer": "fastenhancer",
    "streaming_ulunas": "ulunas",
    "streaming_ultra_dpc": "ultra_dpc",
    "streaming_dbgnet": "dbgnet",
    "streaming_bsrnn": "bsrnn",
    "streaming_tse": "tse",
    "streaming_online_spatialnet": "online_spatialnet",
    "online_spatial": "online_spatialnet",
}

_WRAPPER_IDS = {
    "StreamingGTCRN": "gtcrn",
    "StreamingCAGCRNFlat": "cagcrn",
    "StreamingDeepVQEFlat": "deepvqe",
    "StreamingLocalVQEFlat": "localvqe",
    "StreamingFastEnhancerFlat": "fastenhancer",
    "StreamingULUNASFlat": "ulunas",
    "StreamingUltraDPCAECFlat": "ultra_dpc",
    "StreamingDBGNetFlat": "dbgnet",
    "StreamingONNXWrapper": "bsrnn",
    "StreamingTSEFlat": "tse",
    "StreamingOnlineSpatialNetFlat": "online_spatialnet",
}


def normalize_model_id(model_id: str) -> str:
    value = str(model_id).strip().lower().replace("-", "_")
    value = _ALIASES.get(value, value)
    if value not in _PROFILES:
        raise ValueError(f"Unsupported streaming ONNX model id: {model_id!r}")
    return value


def get_runtime_profile(model_id: str) -> StreamingRuntimeProfile:
    """返回稳定 profile；未知模型 fail-fast。"""
    return _PROFILES[normalize_model_id(model_id)]


def infer_model_id(wrapper) -> str:
    """从 flat wrapper 类型解析已注册模型 id。"""
    try:
        return _WRAPPER_IDS[type(wrapper).__name__]
    except KeyError as exc:
        raise ValueError(
            f"No StreamingRuntimeSpec adapter for wrapper {type(wrapper).__name__}"
        ) from exc


def _runtime_metadata_payload(
    model_id: str,
    stft: STFTRuntimeSpec,
) -> dict[str, object]:
    return {
        "version": RUNTIME_METADATA_VERSION,
        "model_id": model_id,
        "stft": stft.to_metadata(),
    }


def _parse_runtime_metadata(
    metadata: Mapping[str, str],
    model_id: str,
) -> STFTRuntimeSpec:
    raw = metadata.get(RUNTIME_METADATA_KEY)
    if raw is None:
        raise RuntimeError(
            f"ONNX metadata is missing required {RUNTIME_METADATA_KEY!r} for {model_id}; "
            "legacy artifacts must be re-exported"
        )
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"ONNX metadata {RUNTIME_METADATA_KEY!r} is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"version", "model_id", "stft"}:
        raise RuntimeError(f"ONNX metadata {RUNTIME_METADATA_KEY!r} has invalid fields")
    if value["version"] != RUNTIME_METADATA_VERSION:
        raise RuntimeError(
            f"Unsupported streaming runtime metadata version: {value['version']!r}"
        )
    if value["model_id"] != model_id:
        raise RuntimeError(
            f"streaming runtime metadata model_id {value['model_id']!r} "
            f"does not match {model_id!r}"
        )
    return STFTRuntimeSpec.from_metadata(value["stft"])


def _parse_json_metadata(metadata: Mapping[str, str], key: str) -> object:
    raw = metadata.get(key)
    if raw is None:
        raise RuntimeError(
            f"ONNX metadata is missing required {key!r}; legacy artifacts must be re-exported"
        )
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"ONNX metadata {key!r} is not valid JSON") from exc


def _parse_cagcrn_runtime_metadata(
    metadata: Mapping[str, str],
) -> CAGCRNRuntimeSpec:
    return CAGCRNRuntimeSpec.from_metadata(
        _parse_json_metadata(metadata, CAGCRN_RUNTIME_METADATA_KEY)
    )


def _stft_source(wrapper):
    current = wrapper
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "n_fft", None) is not None:
            return current
        nested = getattr(current, "inner", getattr(current, "model", None))
        if nested is None:
            break
        current = nested
    return wrapper


def _init_cache_tensors(wrapper, batch_size: int = 1) -> tuple[torch.Tensor, ...]:
    device = torch.device("cpu")
    dtype = torch.float32
    for method_name in (
        "initialize_cache_flat",
        "init_cache_flat",
        "initialize_cache",
        "init_cache",
    ):
        method = getattr(wrapper, method_name, None)
        if callable(method):
            values = method(batch_size, device, dtype)
            break
    else:
        raise TypeError(f"{type(wrapper).__name__} has no cache initializer")
    if isinstance(values, dict):
        keys = getattr(wrapper, "CACHE_KEYS", tuple(values))
        if tuple(values) != tuple(keys):
            raise ValueError(
                f"{type(wrapper).__name__} cache dictionary order/names mismatch: "
                f"expected={tuple(keys)}, got={tuple(values)}"
            )
        values = tuple(values[key] for key in keys)
    return tuple(values)


def _dbgnet_model_source(wrapper) -> torch.nn.Module:
    current = wrapper
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        candidate = getattr(current, "dbgnet", None)
        if isinstance(candidate, torch.nn.Module):
            return candidate
        nested = getattr(current, "inner", None)
        if nested is None:
            break
        current = nested
    raise TypeError("DBGNet wrapper does not expose its underlying dbgnet module")


def runtime_spec_from_wrapper(
    wrapper,
    model_id: str | None = None,
    *,
    auxiliary_outputs: Sequence[str] | None = None,
) -> StreamingRuntimeSpec:
    """从 wrapper 权威接口推导完整 batch-one ABI 和 metadata。"""
    resolved_id = normalize_model_id(model_id or infer_model_id(wrapper))
    base_profile = get_runtime_profile(resolved_id)
    interface_methods = ("get_input_names", "get_output_names", "get_dynamic_axes")
    missing_methods = tuple(
        name for name in interface_methods if not callable(getattr(wrapper, name, None))
    )
    if missing_methods:
        raise TypeError(
            f"{type(wrapper).__name__} is missing authoritative streaming ABI methods: "
            f"{missing_methods}"
        )
    input_names = tuple(wrapper.get_input_names())
    output_names = tuple(wrapper.get_output_names())
    dynamic_axes = {
        name: dict(axes) for name, axes in wrapper.get_dynamic_axes().items()
    }
    if resolved_id == "bsrnn":
        for name in input_names[1:] + output_names[1:]:
            dynamic_axes[name] = {1: "batch_flattened"}
    if resolved_id == "dbgnet":
        cache_count = max(0, len(input_names) - len(base_profile.primary_inputs))
        actual_aux = tuple(output_names[1:len(output_names) - cache_count])
        profile = StreamingRuntimeProfile(
            **{
                **base_profile.__dict__,
                "auxiliary_outputs": (
                    tuple(auxiliary_outputs)
                    if auxiliary_outputs is not None
                    else actual_aux
                ),
            }
        )
    else:
        profile = base_profile
    cache_names = input_names[len(profile.primary_inputs):]
    if resolved_id == "dbgnet":
        expected_cache_names = dbgnet_cache_names(bool(getattr(wrapper, "use_state_detector", False)))
        if cache_names != expected_cache_names:
            raise ValueError(
                "DBGNet wrapper cache schema mismatch: "
                f"expected={expected_cache_names}, got={cache_names}"
            )
    cache_tensors = _init_cache_tensors(wrapper)
    if len(cache_names) != len(cache_tensors):
        raise ValueError(
            f"{resolved_id} wrapper cache count mismatch: names={len(cache_names)}, "
            f"tensors={len(cache_tensors)}"
        )
    cache_shapes = {
        name: tuple(int(dim) for dim in value.shape)
        for name, value in zip(cache_names, cache_tensors)
    }
    model = _stft_source(wrapper)
    stft = STFTRuntimeSpec(
        n_fft=getattr(model, "n_fft", None),
        hop_length=getattr(model, "hop_length", None),
        win_length=getattr(model, "win_length", None),
        sample_rate=getattr(model, "target_fs", None),
        window=str(getattr(model, "window", "hann")),
        center=getattr(model, "center", None),
    )
    metadata: dict[str, object] = {
        RUNTIME_METADATA_KEY: _runtime_metadata_payload(resolved_id, stft)
    }
    if profile.metadata_key:
        metadata[profile.metadata_key] = {name: list(shape) for name, shape in cache_shapes.items()}
    cagcrn = None
    deepvqe = None
    dbgnet = None
    tse = None
    online_spatialnet = None
    deployment_config = None
    if resolved_id == "online_spatialnet":
        online_model = getattr(wrapper, "model", model)
        online_spatialnet = OnlineSpatialNetRuntimeSpec(
            cache_abi_version=int(getattr(online_model, "streaming_spatial_abi_version", 0)),
            cache_semantics=ONLINE_SPATIALNET_CACHE_SEMANTICS,
            n_channel=int(model.n_channel),
            n_fft=int(model.n_fft),
            cache_names=cache_names,
        )
        metadata[ONLINE_SPATIALNET_RUNTIME_METADATA_KEY] = online_spatialnet.to_metadata()
    if resolved_id == "cagcrn":
        cagcrn = CAGCRNRuntimeSpec(
            delay_d=int(wrapper.delay_d),
            algorithmic_delay_frames=int(wrapper.algorithmic_delay_frames),
            cache_abi_version=CAGCRN_CACHE_ABI_VERSION,
            cache_names=cache_names,
        )
        metadata[CAGCRN_RUNTIME_METADATA_KEY] = cagcrn.to_metadata()
    if resolved_id in {"deepvqe", "dbgnet"}:
        required_stft = {
            "n_fft": stft.n_fft, "hop_length": stft.hop_length,
            "win_length": stft.win_length, "sample_rate": stft.sample_rate,
            "center": stft.center,
        }
        missing = tuple(name for name, value in required_stft.items() if value is None)
        if missing:
            raise ValueError(f"{resolved_id} wrapper STFT contract is incomplete: {missing}")
        raw_config = getattr(model, "deployment_config", None)
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"{resolved_id} wrapper requires deployment_config")
        deployment_config = deployment_config_binding(raw_config, resolved_id)
        metadata[DEPLOYMENT_CONFIG_METADATA_KEY] = deployment_config
    if resolved_id == "deepvqe":
        deepvqe = DeepVQERuntimeSpec(
            variant=str(model.variant),
            compression=float(model.compression),
            alignment_window_frames=int(model.alignment_window_frames),
            alignment_history_frames=int(model.alignment_history_frames),
            algorithmic_delay_frames=int(model.algorithmic_delay_frames),
            algorithmic_delay_samples=int(model.algorithmic_delay_frames) * int(stft.hop_length),
            cache_names=cache_names,
        )
        metadata[DEEPVQE_RUNTIME_METADATA_KEY] = deepvqe.to_metadata()
    if resolved_id == "dbgnet":
        stft_lookahead = int(stft.n_fft // 2) if stft.center else 0
        dbgnet = DBGNetRuntimeSpec(
            use_state_detector=bool(model.use_state_detector),
            state_nclass=int(model.state_nclass),
            network_lookahead_frames=int(model.network_lookahead_frames),
            algorithmic_delay_frames=int(model.algorithmic_delay_frames),
            algorithmic_delay_samples=int(model.algorithmic_delay_frames) * int(stft.hop_length),
            stft_lookahead_samples=stft_lookahead,
            frame_shift_samples=int(stft.hop_length),
            frame_shift_ms=1000.0 * int(stft.hop_length) / int(stft.sample_rate),
            cache_abi_version=DBGNET_CACHE_ABI_VERSION,
            cache_semantics=DBGNET_CACHE_SEMANTICS,
            cache_names=cache_names,
            model_fingerprint=dbgnet_model_fingerprint(_dbgnet_model_source(wrapper)),
        )
        metadata[DBGNET_RUNTIME_METADATA_KEY] = dbgnet.to_metadata()
    if resolved_id == "tse":
        stft.require_complete(label="TSE runtime")
        tse = TSERuntimeSpec(
            cache_abi_version=int(wrapper.CACHE_ABI_VERSION),
            cache_semantics=str(wrapper.CACHE_SEMANTICS),
            condition_names=tuple(wrapper.condition_names),
            cache_names=cache_names,
            separator_layers=int(wrapper.separator_layers),
            nband=int(wrapper.nband),
            separator_hidden_size=int(wrapper.separator_hidden_size),
            analysis_lookahead_samples=int(wrapper.n_fft // 2),
            waveform_output_delay_samples=int(wrapper.waveform_output_delay_samples),
            model_sha256=str(_model_state_fingerprint(wrapper.model, label="TSE")["sha256"]),
        )
        metadata[TSE_RUNTIME_METADATA_KEY] = tse.to_metadata()
    if deployment_config is not None:
        _validate_deployment_contract(
            resolved_id,
            deployment_config,
            stft,
            deepvqe=deepvqe,
            dbgnet=dbgnet,
        )
    return StreamingRuntimeSpec(
        profile=profile,
        input_names=input_names,
        output_names=output_names,
        cache_shapes=cache_shapes,
        dynamic_axes=dynamic_axes,
        stft=stft,
        cagcrn=cagcrn,
        deepvqe=deepvqe,
        dbgnet=dbgnet,
        tse=tse,
        online_spatialnet=online_spatialnet,
        deployment_config=deployment_config,
        metadata=metadata,
    )


def _validate_cache_output_graph(
    *,
    profile: StreamingRuntimeProfile,
    cache_names: tuple[str, ...],
    cache_config: Mapping[str, Sequence[int]],
    input_infos,
    output_infos,
) -> None:
    """在首帧执行前校验 cache output 的 shape/type 与输入 ABI 一致。"""
    from onnx.utils import materialize_cache_shapes

    cache_input_infos = list(input_infos)[len(profile.primary_inputs):]
    cache_output_infos = list(output_infos)[-len(cache_names):] if cache_names else []
    if len(cache_output_infos) != len(cache_names):
        raise RuntimeError(f"{profile.model_id} cache output count does not match inputs")
    output_proxies = []
    for cache_name, input_info, output_info in zip(
        cache_names, cache_input_infos, cache_output_infos
    ):
        input_type = getattr(input_info, "type", None)
        output_type = getattr(output_info, "type", None)
        if input_type is None or output_type is None or output_type != input_type:
            raise RuntimeError(
                f"{profile.model_id} cache output type mismatch for {cache_name!r}: "
                f"input={input_type!r}, output={output_type!r}"
            )
        output_shape = getattr(output_info, "shape", None)
        if output_shape is None:
            raise RuntimeError(
                f"{profile.model_id} cache output shape is missing for {cache_name!r}"
            )
        if profile.model_id == "deepvqe":
            template = cache_config[cache_name]
            if len(output_shape) != len(template):
                raise RuntimeError(
                    f"deepvqe cache output rank mismatch for {cache_name!r}: "
                    f"graph={output_shape!r}, metadata={list(template)!r}"
                )
            output_shape = [
                expected if isinstance(actual, str) and actual else actual
                for actual, expected in zip(output_shape, template)
            ]
        output_proxies.append(SimpleNamespace(name=cache_name, shape=output_shape))
    try:
        materialize_cache_shapes(
            {name: list(shape) for name, shape in cache_config.items()},
            output_proxies,
            batch_size=1,
            allowed_dynamic_axes=profile.allowed_cache_dynamic_axes,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"{profile.model_id} cache output shape does not match metadata/input ABI: {exc}"
        ) from exc


def validate_static_batch_graph(
    profile: StreamingRuntimeProfile,
    input_infos,
    output_infos,
) -> None:
    """按 canonical profile 拒绝动态或非合同 batch 轴。"""
    expected = profile.static_batch_size
    if expected is None:
        return
    dynamic_batch_names = {"batch", "batch_size", "batch_flattened"}
    for info in (*input_infos, *output_infos):
        shape = getattr(info, "shape", None)
        actual = shape[0] if shape else None
        dynamic_batch_axes = tuple(
            axis for axis, dimension in enumerate(shape or ())
            if dimension is None or (
                isinstance(dimension, str)
                and dimension.lower() in dynamic_batch_names
            )
        )
        if dynamic_batch_axes or isinstance(actual, bool) or actual != expected:
            raise RuntimeError(
                f"{profile.model_id} graph tensor {info.name!r} requires static "
                f"batch size {expected}, got shape={shape!r}"
            )


def _validate_aec_graph_stft(
    model_id: str,
    stft: STFTRuntimeSpec,
    input_infos,
    output_infos,
    dbgnet: DBGNetRuntimeSpec | None = None,
) -> None:
    """在首帧前绑定双输入 AEC 图、完整 STFT 与可选 state 输出。"""
    required = {
        "n_fft": stft.n_fft, "hop_length": stft.hop_length,
        "win_length": stft.win_length, "sample_rate": stft.sample_rate,
        "center": stft.center,
    }
    missing = tuple(name for name, value in required.items() if value is None)
    if missing:
        raise RuntimeError(f"{model_id} runtime STFT metadata is incomplete: {missing}")
    if stft.win_length > stft.n_fft:
        raise RuntimeError(
            f"{model_id} runtime STFT win_length exceeds n_fft: "
            f"{stft.win_length} > {stft.n_fft}"
        )
    if stft.window != "hann":
        raise RuntimeError(
            f"{model_id} runtime STFT window must be 'hann', got {stft.window!r}"
        )
    expected_shape = (stft.n_fft // 2 + 1, 1, 2)
    graph_tensors = (
        *((info, False) for info in list(input_infos)[:2]),
        *((info, True) for info in list(output_infos)[:1]),
    )
    for info, is_primary_output in graph_tensors:
        if getattr(info, "type", None) != "tensor(float)":
            raise RuntimeError(
                f"{model_id} graph tensor {info.name!r} must have type tensor(float)"
            )
        graph_shape = getattr(info, "shape", None)
        if graph_shape is None or len(graph_shape) != 4:
            raise RuntimeError(f"{model_id} graph tensor {info.name!r} must have rank 4")
        for axis, expected in zip((1, 2, 3), expected_shape):
            actual = graph_shape[axis]
            symbolic_deepvqe_output_axis = (
                model_id == "deepvqe"
                and is_primary_output
                and isinstance(actual, str)
                and bool(actual)
            )
            if actual != expected and not symbolic_deepvqe_output_axis:
                raise RuntimeError(
                    f"{model_id} graph tensor {info.name!r} axis {axis}="
                    f"{actual!r} does not match STFT metadata {expected}"
                )
    if dbgnet is None:
        return
    output_by_name = {info.name: info for info in output_infos}
    state_info = output_by_name.get("state_probs")
    if dbgnet.use_state_detector != (state_info is not None):
        raise RuntimeError("DBGNet state detector metadata/output ABI mismatch")
    if state_info is None:
        return
    if getattr(state_info, "type", None) != "tensor(float)":
        raise RuntimeError("DBGNet state_probs must have type tensor(float)")
    shape = getattr(state_info, "shape", None)
    if shape is None or len(shape) != 3:
        raise RuntimeError("DBGNet state_probs must have rank 3")
    if shape[1] != 1 or shape[2] != dbgnet.state_nclass:
        raise RuntimeError(
            "DBGNet state_probs shape does not match runtime metadata: "
            f"shape={shape!r}, state_nclass={dbgnet.state_nclass}"
        )


def _validate_deployment_contract(
    model_id: str,
    binding: Mapping[str, object],
    stft: STFTRuntimeSpec,
    *,
    deepvqe: DeepVQERuntimeSpec | None = None,
    dbgnet: DBGNetRuntimeSpec | None = None,
) -> None:
    config = binding["config"]
    if config["stft"] != stft.to_metadata():
        raise RuntimeError(f"{model_id} deployment config STFT does not match runtime metadata")
    args = config["model"]["args"]
    if deepvqe is not None:
        expected = (
            args.get("variant", "base"), float(args.get("compression", 0.3)),
            int(args.get("align_delay", 100)),
        )
        actual = (
            deepvqe.variant, deepvqe.compression, deepvqe.alignment_window_frames,
        )
        if actual != expected:
            raise RuntimeError(
                f"DeepVQE deployment config/model metadata mismatch: expected={expected}, "
                f"got={actual}"
            )
    if dbgnet is not None:
        if bool(args.get("use_state_detector", False)) != dbgnet.use_state_detector:
            raise RuntimeError("DBGNet deployment config use_state_detector mismatch")
        if int(args.get("state_nclass", 3)) != dbgnet.state_nclass:
            raise RuntimeError("DBGNet deployment config state_nclass mismatch")
        expected_stft_lookahead = stft.n_fft // 2 if stft.center else 0
        expected_frame_ms = 1000.0 * stft.hop_length / stft.sample_rate
        if (
            dbgnet.stft_lookahead_samples != expected_stft_lookahead
            or dbgnet.frame_shift_samples != stft.hop_length
            or not math.isclose(dbgnet.frame_shift_ms, expected_frame_ms, abs_tol=1e-12)
        ):
            raise RuntimeError("DBGNet runtime timing does not match STFT metadata")


def _validate_cagcrn_graph_stft(stft: STFTRuntimeSpec, input_infos, output_infos) -> None:
    """绑定 CAGCRN 图的单帧谱 shape 与完整 STFT metadata。"""
    required = {
        "n_fft": stft.n_fft,
        "hop_length": stft.hop_length,
        "win_length": stft.win_length,
        "sample_rate": stft.sample_rate,
        "center": stft.center,
    }
    missing = tuple(name for name, value in required.items() if value is None)
    if missing:
        raise RuntimeError(f"CAGCRN runtime STFT metadata is incomplete: {missing}")
    if stft.win_length != stft.n_fft:
        raise RuntimeError(
            f"CAGCRN runtime STFT win_length must equal n_fft, got "
            f"{stft.win_length} and {stft.n_fft}"
        )
    if stft.window != "hann":
        raise RuntimeError(f"CAGCRN runtime STFT window must be 'hann', got {stft.window!r}")
    expected_shape = (stft.n_fft // 2 + 1, 1, 2)
    for info in (*list(input_infos)[:2], list(output_infos)[0]):
        graph_shape = getattr(info, "shape", None)
        if graph_shape is None or len(graph_shape) != 4:
            raise RuntimeError(f"CAGCRN graph tensor {info.name!r} must have rank 4")
        for axis, expected in zip((1, 2, 3), expected_shape):
            if graph_shape[axis] != expected:
                raise RuntimeError(
                    f"CAGCRN graph tensor {info.name!r} axis {axis}={graph_shape[axis]!r} "
                    f"does not match STFT metadata {expected}"
                )


def _validate_tse_graph_stft(stft: STFTRuntimeSpec, input_infos, output_infos) -> None:
    """绑定 TSE 单帧 pair-real 频谱与训练 STFT 几何。"""
    stft.require_complete(label="TSE runtime")
    if stft.win_length != stft.n_fft or stft.center is not True:
        raise RuntimeError(
            "TSE runtime requires win_length == n_fft and center=True"
        )
    expected = (stft.n_fft // 2 + 1, 1, 2)
    for info, is_output in (
        (list(input_infos)[0], False),
        (list(output_infos)[0], True),
    ):
        if getattr(info, "type", None) != "tensor(float)":
            raise RuntimeError(f"TSE graph tensor {info.name!r} must be tensor(float)")
        shape = getattr(info, "shape", None)
        if shape is None or len(shape) != 4:
            raise RuntimeError(f"TSE graph tensor {info.name!r} must have rank 4")
        for axis, dimension in zip((1, 2, 3), expected):
            actual = shape[axis]
            symbolic_output_axis = is_output and isinstance(actual, str) and bool(actual)
            if actual != dimension and not symbolic_output_axis:
                raise RuntimeError(
                    f"TSE graph tensor {info.name!r} axis {axis}={actual!r} "
                    f"does not match {dimension}"
                )


def runtime_spec_from_session(session, model_id: str) -> StreamingRuntimeSpec:
    """从 ORT graph 与完整版本化 metadata 构造并严格校验 runtime spec。"""
    from onnx.utils import materialize_cache_shapes, parse_onnx_cache_config

    profile = get_runtime_profile(model_id)
    input_infos = list(session.get_inputs())
    output_infos = list(session.get_outputs())
    input_names = tuple(info.name for info in input_infos)
    output_names = tuple(info.name for info in output_infos)
    validate_static_batch_graph(profile, input_infos, output_infos)
    cache_names = input_names[len(profile.primary_inputs):]
    metadata = dict(session.get_modelmeta().custom_metadata_map or {})
    stft = _parse_runtime_metadata(metadata, profile.model_id)
    deepvqe = (
        DeepVQERuntimeSpec.from_metadata(
            _parse_json_metadata(metadata, DEEPVQE_RUNTIME_METADATA_KEY)
        )
        if profile.model_id == "deepvqe" else None
    )
    dbgnet = (
        DBGNetRuntimeSpec.from_metadata(
            _parse_json_metadata(metadata, DBGNET_RUNTIME_METADATA_KEY)
        )
        if profile.model_id == "dbgnet" else None
    )
    tse = (
        TSERuntimeSpec.from_metadata(
            _parse_json_metadata(metadata, TSE_RUNTIME_METADATA_KEY)
        )
        if profile.model_id == "tse" else None
    )
    online_spatialnet = (
        OnlineSpatialNetRuntimeSpec.from_metadata(
            _parse_json_metadata(metadata, ONLINE_SPATIALNET_RUNTIME_METADATA_KEY)
        )
        if profile.model_id == "online_spatialnet" else None
    )
    if profile.metadata_key is None:
        raise RuntimeError(f"{profile.model_id} has no registered cache metadata key")
    parsed = parse_onnx_cache_config(metadata, profile.metadata_key)
    cagcrn = (
        _parse_cagcrn_runtime_metadata(metadata)
        if profile.model_id == "cagcrn" else None
    )
    versioned_cache_names = (
        cagcrn.cache_names if cagcrn is not None
        else deepvqe.cache_names if deepvqe is not None
        else dbgnet.cache_names if dbgnet is not None
        else tse.cache_names if tse is not None
        else online_spatialnet.cache_names if online_spatialnet is not None
        else None
    )
    if versioned_cache_names is not None:
        if versioned_cache_names != cache_names:
            raise RuntimeError(
                f"{profile.model_id} cache order/input ABI mismatch: "
                f"metadata={versioned_cache_names}, inputs={cache_names}"
            )
        if set(parsed) != set(cache_names):
            raise RuntimeError(
                f"{profile.model_id} cache metadata/input ABI names mismatch: "
                f"metadata={tuple(parsed)}, inputs={cache_names}"
            )
        parsed = {name: parsed[name] for name in versioned_cache_names}
    elif tuple(parsed) != cache_names:
        raise RuntimeError(
            f"{profile.model_id} cache metadata/input ABI mismatch: "
            f"metadata={tuple(parsed)}, inputs={cache_names}"
        )
    materialized_cache = materialize_cache_shapes(
        parsed,
        input_infos[len(profile.primary_inputs):],
        batch_size=1,
        allowed_dynamic_axes=profile.allowed_cache_dynamic_axes,
    )
    if online_spatialnet is not None:
        primary_shape = getattr(input_infos[0], "shape", None)
        expected_freqs = online_spatialnet.n_fft // 2 + 1
        if (
            primary_shape is None
            or len(primary_shape) != 5
            or primary_shape[1] != online_spatialnet.n_channel
            or primary_shape[2] != expected_freqs
            or primary_shape[3] != 1
            or primary_shape[4] != 2
        ):
            raise RuntimeError(
                "OnlineSpatialNet primary input shape does not match versioned runtime metadata: "
                f"shape={primary_shape!r}"
            )
    deployment_config = None
    if profile.model_id == "cagcrn":
        _validate_cagcrn_graph_stft(stft, input_infos, output_infos)
    if profile.model_id == "tse":
        _validate_tse_graph_stft(stft, input_infos, output_infos)
    if profile.model_id in {"deepvqe", "dbgnet"}:
        deployment_config = parse_deployment_config_binding(
            _parse_json_metadata(metadata, DEPLOYMENT_CONFIG_METADATA_KEY),
            profile.model_id,
        )
        _validate_deployment_contract(
            profile.model_id, deployment_config, stft,
            deepvqe=deepvqe, dbgnet=dbgnet,
        )
        _validate_aec_graph_stft(
            profile.model_id, stft, input_infos, output_infos, dbgnet
        )
    if profile.model_id in {"cagcrn", "deepvqe", "dbgnet", "tse"}:
        _validate_cache_output_graph(
            profile=profile,
            cache_names=cache_names,
            cache_config=parsed,
            input_infos=input_infos,
            output_infos=output_infos,
        )
    auxiliary_count = len(output_names) - 1 - len(cache_names)
    if auxiliary_count < 0:
        raise RuntimeError(f"{profile.model_id} ONNX output count is invalid")
    actual_profile = StreamingRuntimeProfile(
        **{
            **profile.__dict__,
            "auxiliary_outputs": tuple(output_names[1:1 + auxiliary_count]),
        }
    )
    return StreamingRuntimeSpec(
        profile=actual_profile,
        input_names=input_names,
        output_names=output_names,
        cache_shapes={name: tuple(materialized_cache[name]) for name in cache_names},
        stft=stft,
        cagcrn=cagcrn,
        deepvqe=deepvqe,
        dbgnet=dbgnet,
        tse=tse,
        online_spatialnet=online_spatialnet,
        deployment_config=deployment_config,
        metadata=metadata,
    )
