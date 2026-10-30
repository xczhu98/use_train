"""FN-SSL/IPDnet 动态语音配方的任务侧合同。

官方仓库把语音清理、片段抽样和多源重叠写在 Dataset 内部。本模块把这些
语义登记成有限配方，避免把任意字符串当成 ``simulation_profile``，也避免
把 WebRTC VAD 依赖污染到共享数据层。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


SSL_RECIPE_VERSION = "ssl-official-recipe-v1"


@dataclass(frozen=True)
class SSLSimulationRecipe:
    """一个已登记的 SSL simulation profile。

    ``segment_sampling`` 描述 source manifest 上的抽样语义；实际音频加载和
    splice 仍由 shared ``SpeechPreparation`` 完成。``overlap_policy`` 只作用于
    source waveform，不能修改 direct target 或 scene geometry。
    """

    profile: str
    version: str
    vad_cleaning: str
    segment_sampling: str
    overlap_policy: str
    source_duration_seconds: float
    official_data_domain: str
    official_source_count: int | None = None
    simulation_not_official: bool = False

    def to_provenance(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "version": self.version,
            "vad_cleaning": self.vad_cleaning,
            "segment_sampling": self.segment_sampling,
            "overlap_policy": self.overlap_policy,
            "source_duration_seconds": self.source_duration_seconds,
            "official_data_domain": self.official_data_domain,
            "official_source_count": self.official_source_count,
            "simulation_not_official": self.simulation_not_official,
            "target_contract": "per_source_dpipd_pit",
        }


# FN-SSL 官方仿真和 IPDnet 官方仿真使用不同的持续时间；IPDnet2 的官方
# 训练域是 RealMAN 录音，故这里只登记其动态链的明确“无语音清理”边界。
_RECIPES = {
    "fnssl_official": SSLSimulationRecipe(
        profile="fnssl_official",
        version=f"{SSL_RECIPE_VERSION}:fnssl",
        vad_cleaning="webrtcvad_progressive",
        segment_sampling="chapter_random",
        overlap_policy="none",
        source_duration_seconds=4.79,
        official_data_domain="LibriSpeech+Noise92 synthetic gpuRIR",
        official_source_count=1,
    ),
    "ipdnet_official": SSLSimulationRecipe(
        profile="ipdnet_official",
        version=f"{SSL_RECIPE_VERSION}:ipdnet",
        vad_cleaning="webrtcvad_progressive",
        segment_sampling="chapter_random",
        overlap_policy="random_zero_mask_train",
        source_duration_seconds=4.5,
        official_data_domain="LibriSpeech+Noise92 synthetic gpuRIR",
        official_source_count=1,
    ),
    "ipdnet2_official": SSLSimulationRecipe(
        profile="ipdnet2_official",
        version=f"{SSL_RECIPE_VERSION}:ipdnet2",
        vad_cleaning="none",
        segment_sampling="recording",
        overlap_policy="none",
        source_duration_seconds=4.0,
        official_data_domain="RealMAN recordings; dynamic path is a local approximation",
        official_source_count=None,
        simulation_not_official=True,
    ),
}

SSL_SIMULATION_RECIPES: Mapping[str, SSLSimulationRecipe] = MappingProxyType(_RECIPES)
SSL_SIMULATION_PROFILES = frozenset(_RECIPES)
_VAD_CLEANING = frozenset({"none", "webrtcvad_progressive"})
_SEGMENT_SAMPLING = frozenset({"manifest_random", "chapter_random", "recording"})
_OVERLAP_POLICIES = frozenset({"none", "random_zero_mask_train"})


def _new_webrtc_vad():
    """加载 WebRTC VAD；兼容旧 wheel 对已移除 ``pkg_resources`` 的引用。"""

    try:
        import webrtcvad

        return webrtcvad.Vad(), False
    except ModuleNotFoundError as exc:
        if exc.name != "pkg_resources":
            raise ImportError(
                "FN-SSL official speech recipe requires optional dependency webrtcvad"
            ) from exc
        try:
            import _webrtcvad
        except ImportError as inner_exc:  # pragma: no cover - wheel-specific fallback
            raise ImportError(
                "FN-SSL official speech recipe requires optional dependency webrtcvad"
            ) from inner_exc

        class _CompatVad:
            def __init__(self):
                self._vad = _webrtcvad.create()
                _webrtcvad.init(self._vad)

            def set_mode(self, mode):
                _webrtcvad.set_mode(self._vad, mode)

            def is_speech(self, buf, sample_rate):
                return _webrtcvad.process(
                    self._vad, sample_rate, buf, int(len(buf) / 2)
                )

        return _CompatVad(), True


def resolve_ssl_recipe(
    simulation_profile: str | None,
    *,
    vad_cleaning: str | None = None,
    segment_sampling: str | None = None,
    overlap_policy: str | None = None,
) -> SSLSimulationRecipe | None:
    """解析登记的 profile，并校验 split-level 覆盖值。

    配方字段是 recipe 的一部分；显式给出的值必须与登记 recipe 一致，避免
    只改一个字段后仍声称官方复现。没有 profile 时不得单独提供 recipe 字段。
    """

    overrides = {
        "vad_cleaning": vad_cleaning,
        "segment_sampling": segment_sampling,
        "overlap_policy": overlap_policy,
    }
    if simulation_profile is None:
        if any(value is not None for value in overrides.values()):
            raise ValueError(
                "SSL vad_cleaning/segment_sampling/overlap_policy require a registered "
                "simulation_profile"
            )
        return None
    if not isinstance(simulation_profile, str) or not simulation_profile.strip():
        raise ValueError("SSL simulation_profile must be a registered non-empty string")
    profile = simulation_profile.strip()
    recipe = SSL_SIMULATION_RECIPES.get(profile)
    if recipe is None:
        raise ValueError(
            "unsupported SSL simulation_profile "
            f"{profile!r}; expected one of {sorted(SSL_SIMULATION_PROFILES)!r}"
        )
    allowed = {
        "vad_cleaning": _VAD_CLEANING,
        "segment_sampling": _SEGMENT_SAMPLING,
        "overlap_policy": _OVERLAP_POLICIES,
    }
    for name, value in overrides.items():
        if value is None:
            continue
        if not isinstance(value, str) or value not in allowed[name]:
            raise ValueError(
                f"SSL {name} must be one of {sorted(allowed[name])!r}"
            )
        expected = getattr(recipe, name)
        if value != expected:
            raise ValueError(
                f"SSL {name}={value!r} does not match registered profile "
                f"{profile!r} ({expected!r})"
            )
    return recipe


def clean_speech_with_webrtcvad(
    speech: np.ndarray,
    sample_rate: int,
    *,
    retain_ratio: float = 0.66,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """按官方 progressive aggressiveness 清理语音并返回 sample-level VAD。

    upstream FN-SSL/IPDnet 依次尝试 mode 3、2、1；当清理后保留比例低于
    ``0.66`` 时降低 aggressiveness。``webrtcvad`` 在函数内导入，缺失时只
    阻断选择该 recipe 的数据路径。
    """

    if type(sample_rate) is not int or sample_rate not in {8000, 16000, 32000, 48000}:
        raise ValueError("WebRTC VAD requires sample_rate in {8000,16000,32000,48000}")
    if not isinstance(retain_ratio, (int, float)) or not 0.0 < float(retain_ratio) <= 1.0:
        raise ValueError("retain_ratio must be in (0,1]")
    values = np.asarray(speech, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("SSL WebRTC VAD expects a mono 1-D waveform")
    if not np.isfinite(values).all():
        raise ValueError("SSL WebRTC VAD input must be finite")
    frame_length = int(round(0.01 * sample_rate))
    frame_count = len(values) // frame_length
    masks: np.ndarray | None = None
    selected_mode: int | None = None
    attempted_modes: list[int] = []
    for mode in (3, 2, 1):
        attempted_modes.append(mode)
        vad, compat_import = _new_webrtc_vad()
        vad.set_mode(mode)
        mask = np.zeros_like(values, dtype=np.float32)
        for frame_index in range(frame_count):
            start = frame_index * frame_length
            frame = values[start : start + frame_length]
            frame_bytes = (frame * 32767).astype(np.int16).tobytes()
            mask[start : start + frame_length] = float(
                vad.is_speech(frame_bytes, sample_rate)
            )
        cleaned = values * mask
        masks = mask
        selected_mode = mode
        if values.size == 0 or float(np.count_nonzero(cleaned)) >= values.size * float(retain_ratio):
            break
    assert masks is not None and selected_mode is not None
    cleaned = (values * masks).astype(np.float32, copy=False)
    facts = {
        "backend": "webrtcvad",
        "algorithm": "progressive_aggressiveness",
        "attempted_modes": attempted_modes,
        "selected_mode": selected_mode,
        "frame_ms": 10,
        "frame_samples": frame_length,
        "frame_count": frame_count,
        "retain_ratio_threshold": float(retain_ratio),
        "active_ratio": float(np.count_nonzero(cleaned) / values.size) if values.size else 0.0,
        "compat_pkg_resources_import": bool(compat_import),
    }
    return cleaned, masks, facts


def apply_overlap_policy(
    speech: np.ndarray,
    *,
    policy: str,
    split: str,
    tracks: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """执行 IPDnet 官方 train-only 0–2 秒 source overlap mask。"""

    if policy not in _OVERLAP_POLICIES:
        raise ValueError(f"unsupported SSL overlap_policy {policy!r}")
    values = np.asarray(speech, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("SSL overlap policy expects a mono 1-D waveform")
    facts: dict[str, Any] = {
        "policy": policy,
        "split": split,
        "applied": False,
        "start_sample": 0,
        "mask_samples": 0,
    }
    if policy == "none" or split != "train" or tracks <= 1:
        return values, facts
    if rng.random() <= 0.8:
        return values, facts
    max_mask = min(int(2 * sample_rate), len(values))
    mask_samples = int(rng.random() * max_mask)
    start = int(rng.integers(0, len(values) - mask_samples + 1))
    masked = values.copy()
    masked[start : start + mask_samples] = 0.0
    facts.update({"applied": True, "start_sample": start, "mask_samples": mask_samples})
    return masked, facts


__all__ = [
    "SSL_RECIPE_VERSION",
    "SSLSimulationRecipe",
    "SSL_SIMULATION_PROFILES",
    "SSL_SIMULATION_RECIPES",
    "apply_overlap_policy",
    "clean_speech_with_webrtcvad",
    "resolve_ssl_recipe",
]
