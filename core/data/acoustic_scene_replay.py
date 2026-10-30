"""不含 RNG、I/O 或 cache 的 C-0 asset-forced replay helper。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.data.post_augment import (
    scale_source_volume_stem,
    source_volume_normalize,
    source_volume_shared_max_abs,
)
from core.data.errors import SimulationContractError
from core.data.simulation.engine import (
    add_moving_reverberation,
    add_paired_reverberation,
    add_reverberation,
    estimate_early_rir,
)


class AssetForcedReplayError(SimulationContractError):
    """asset-forced C-0 replay 的不可恢复 runtime 契约错误。"""


@dataclass(frozen=True)
class TargetAcousticReplay:
    """纯 C-0 replay 的 full、early 与 target stems 及标定 sidecar。"""

    noisy_speech: np.ndarray
    mix_reference: np.ndarray
    clean_target: np.ndarray
    source_volume_calibration: dict[str, Any] | None


def _raise_contract_error(message: str, exc: Exception | None = None) -> None:
    if exc is None:
        raise AssetForcedReplayError(message)
    raise AssetForcedReplayError(message) from exc


def _validate_target_waveform(
    target_waveform: np.ndarray,
    *,
    sample_rate: int,
    num_samples: int,
) -> np.ndarray:
    if type(sample_rate) is not int or sample_rate <= 0:
        _raise_contract_error("C-0 asset replay sample rate must be a positive integer")
    if type(num_samples) is not int or num_samples < 0:
        _raise_contract_error("C-0 asset replay length must be a non-negative integer")
    try:
        waveform = np.asarray(target_waveform)
    except (TypeError, ValueError, OverflowError) as exc:
        _raise_contract_error("C-0 asset replay target waveform must be an array", exc)
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        _raise_contract_error(
            "C-0 asset replay target waveform must have mono shape (1, T)"
        )
    if waveform.shape[1] != num_samples:
        _raise_contract_error("C-0 asset replay target waveform length mismatch")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            waveform_32 = np.asarray(waveform, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        _raise_contract_error(
            "C-0 asset replay target waveform must contain numeric samples", exc
        )
    if not np.isfinite(waveform_32).all():
        _raise_contract_error(
            "C-0 asset replay target waveform must contain only finite float32 samples"
        )
    return np.ascontiguousarray(waveform_32)


def _validate_scene_contract(
    scene_plan: Any,
    resolved_assets: Any,
    *,
    asset_role: str,
    sample_rate: int,
    num_samples: int,
) -> np.ndarray | None:
    if not isinstance(asset_role, str) or not asset_role:
        _raise_contract_error("C-0 asset replay requires a non-empty asset role")
    if (
        getattr(scene_plan, "sample_rate", None) != sample_rate
        or getattr(scene_plan, "num_samples", None) != num_samples
    ):
        _raise_contract_error(
            "C-0 asset replay scene plan sample rate or length mismatch"
        )
    if (
        getattr(resolved_assets, "role", None) != asset_role
        or getattr(resolved_assets, "sample_rate", None) != sample_rate
        or getattr(resolved_assets, "num_samples", None) != num_samples
    ):
        _raise_contract_error("C-0 asset replay resolved asset role or length mismatch")
    # formal mixture 的 plan/assets 可携带完整 C-1/C-2/C-3 审计；C-0 replay
    # 只消费其中的 RIR/source gain。调用层负责保证本次 replay 的 need_noise=False。
    rir_plan = getattr(scene_plan, "rir", None)
    if rir_plan is None:
        _raise_contract_error("C-0 asset replay requires an RIR plan")
    rir_asset = getattr(resolved_assets, "rir", None)
    if bool(getattr(rir_plan, "enabled", False)) != (rir_asset is not None):
        _raise_contract_error("C-0 asset replay RIR plan and asset mismatch")
    if not bool(getattr(rir_plan, "enabled", False)):
        if bool(getattr(rir_plan, "moving", False)):
            _raise_contract_error("C-0 asset replay disabled RIR plan must be static")
        return None

    early_rir_sec = getattr(rir_plan, "early_rir_sec", None)
    try:
        early_rir_value = float(early_rir_sec)
    except (TypeError, ValueError, OverflowError) as exc:
        _raise_contract_error(
            "C-0 asset replay RIR early duration must be numeric", exc
        )
    if not np.isfinite(early_rir_value) or early_rir_value < 0.0:
        _raise_contract_error(
            "C-0 asset replay RIR early duration must be finite and non-negative"
        )

    try:
        rir = np.asarray(rir_asset, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        _raise_contract_error(
            "C-0 asset replay RIR asset must contain numeric samples", exc
        )
    if (
        rir.ndim != 2
        or rir.shape[0] <= 0
        or rir.shape[-1] <= 0
        or not np.isfinite(rir).all()
    ):
        _raise_contract_error("C-0 asset replay RIR asset must be finite and non-empty")
    moving = rir.shape[0] > 1
    if moving != bool(getattr(rir_plan, "moving", False)):
        _raise_contract_error("C-0 asset replay RIR moving shape does not match plan")
    if moving:
        segment_sec = getattr(rir_plan, "segment_sec", None)
        try:
            segment_value = float(segment_sec)
        except (TypeError, ValueError, OverflowError) as exc:
            _raise_contract_error(
                "C-0 asset replay moving RIR requires finite segment", exc
            )
        if not np.isfinite(segment_value) or segment_value <= 0.0:
            _raise_contract_error(
                "C-0 asset replay moving RIR requires finite segment"
            )
    return np.ascontiguousarray(rir)


def replay_target_acoustic_scene(
    target_waveform: np.ndarray,
    *,
    scene_plan: Any,
    resolved_assets: Any,
    asset_role: str,
    sample_rate: int,
    num_samples: int,
    need_noise: bool = False,
    source_volume_normalization_enabled: bool,
    source_volume_use_rms: bool,
) -> TargetAcousticReplay:
    """对已解析 C-0 asset 执行目标 replay，不抽样且不接触外部状态。

    调用者必须先完成 formal mixture；该 helper 只接受其 plan/assets 和目标
    waveform，不持有 dataset、RNG、路径或 cache，因此不会影响 formal completion。
    RIR 仅在同步调用期间只读借用，不保存调用者数组；输出不与资产共享内存。
    need_noise 默认为 False；显式传入 True 会按合同拒绝。
    """
    if need_noise:
        _raise_contract_error("C-0 asset replay only supports need_noise=False")
    waveform = _validate_target_waveform(
        target_waveform, sample_rate=sample_rate, num_samples=num_samples
    )
    rir = _validate_scene_contract(
        scene_plan,
        resolved_assets,
        asset_role=asset_role,
        sample_rate=sample_rate,
        num_samples=num_samples,
    )
    source_gain = getattr(scene_plan, "source_gain", None)
    if source_gain is None or bool(source_volume_normalization_enabled) != bool(
        getattr(source_gain, "enabled", False)
    ):
        _raise_contract_error("C-0 asset replay source gain plan does not match config")
    try:
        forced_gain = float(source_gain.gain)
    except (TypeError, ValueError, OverflowError) as exc:
        _raise_contract_error("C-0 asset replay source gain must be numeric", exc)
    if not np.isfinite(forced_gain) or forced_gain < 0.0:
        _raise_contract_error("C-0 asset replay forced_gain must be finite and non-negative")

    if rir is None:
        noisy_speech = waveform.copy()
        mix_reference = waveform.copy()
        clean_target = waveform.copy()
    else:
        rir_plan = scene_plan.rir
        rir_normalize = bool(rir_plan.normalize)
        early_rir_sec = float(rir_plan.early_rir_sec)
        target_type = str(rir_plan.target_type)
        moving = bool(rir_plan.moving)
        if moving:
            segment_sec = float(rir_plan.segment_sec)
            full_sequence_peak = (
                float(np.max(np.abs(rir))) if rir_normalize else None
            )
            noisy_speech = add_moving_reverberation(
                waveform,
                rir,
                segment_sec,
                sample_rate,
                normalize_rir=rir_normalize,
                normalization_peak=full_sequence_peak,
            )
            early_rir = estimate_early_rir(
                rir, early_rir_sec=early_rir_sec, fs=sample_rate
            )
            mix_reference = add_moving_reverberation(
                waveform,
                early_rir,
                segment_sec,
                sample_rate,
                normalize_rir=rir_normalize,
                normalization_peak=full_sequence_peak,
            )
        elif rir.ndim == 2 and rir.shape[0] == 1:
            noisy_speech, mix_reference = add_paired_reverberation(
                waveform,
                rir,
                early_rir_sec=early_rir_sec,
                fs=sample_rate,
                normalize_rir=rir_normalize,
            )
        else:
            noisy_speech = add_reverberation(
                waveform, rir, normalize_rir=rir_normalize
            )
            early_rir = estimate_early_rir(
                rir, early_rir_sec=early_rir_sec, fs=sample_rate
            )
            mix_reference = add_reverberation(
                waveform, early_rir, normalize_rir=rir_normalize
            )
        clean_target = waveform.copy() if target_type == "dry" else mix_reference

    calibration: dict[str, Any] | None = None
    if bool(source_volume_normalization_enabled):
        calibration = {}
        shared_stem_max_abs = source_volume_shared_max_abs(
            noisy_speech, mix_reference, clean_target
        )
        if shared_stem_max_abs > 0.0:
            safe_gain = float(np.finfo(np.float32).max) / shared_stem_max_abs
            if forced_gain > safe_gain:
                _raise_contract_error(
                    "C-0 asset replay forced source volume gain is unsafe for C-0 stems"
                )
        scaled_reference, _ = source_volume_normalize(
            mix_reference,
            use_rms=bool(source_volume_use_rms),
            sr=sample_rate,
            forced_gain=forced_gain,
            shared_stem_max_abs=shared_stem_max_abs,
            calibration_out=calibration,
        )
        calibration["stage"] = "post_rir_pre_noise_echo"
        gain = float(calibration["gain"])
        noisy_speech = scale_source_volume_stem(
            noisy_speech, gain, stem_name="full_rir"
        )
        mix_reference = scaled_reference
        clean_target = scale_source_volume_stem(
            clean_target, gain, stem_name="clean_target"
        )
    # 公共 helper 的输出统一为可直接交给后续 STFT/TSE 路径的 canonical buffer。
    noisy_speech = np.ascontiguousarray(np.asarray(noisy_speech, dtype=np.float32))
    mix_reference = np.ascontiguousarray(np.asarray(mix_reference, dtype=np.float32))
    clean_target = np.ascontiguousarray(np.asarray(clean_target, dtype=np.float32))
    return TargetAcousticReplay(
        noisy_speech=noisy_speech,
        mix_reference=mix_reference,
        clean_target=clean_target,
        source_volume_calibration=calibration,
    )


__all__ = [
    "AssetForcedReplayError",
    "TargetAcousticReplay",
    "replay_target_acoustic_scene",
]
