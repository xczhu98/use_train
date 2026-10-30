"""Online dynamic mixing dataset for AEC (DynamicMixingDataset).

Shared near-end I/O (resampling, LRU cache, alignment) lives in
``core.data.simulation_audio_utils``. Shared simulation utilities (RNG wrappers,
SCP parsers, active RMS) live in ``core.data.simulation_utils``.

This module retains AEC-specific paths: stereo ref/echo loading,
JSON farend/echo pair manifests, GCC-PHAT alignment, SER scaling,
random echo delay, and the ``mic``/``farend``/``nearend`` tuple assembly
for ``aec_collate_fn`` / ``AECTrainer``. With explicit ``spatial_rir_provider``
opt-in, ``mic`` is ``[C,T]`` while farend and nearend remain ``[T]``.

Six-layer simulation pipeline (``_run``, aligned with YAML ``simulation_cfg``):

  Layer A  Clean-speech pre-processing (clean only)
           A-1 Speech splice
           A-2 Speed / pitch / EQ / silence-gap augmentations
           A-3 模式 A crop/tile；模式 B 单 utterance 固定 scene canvas
           A-4 High-pass filter (70 Hz filtfilt)
  Layer B  Scenario branch selection (after Layer A, before acoustic mixing)
           One uniform draw selects among seven mutually-exclusive branches:
           prob_target_only, prob_noise_only, prob_echo_only, prob_noise_echo_only,
           prob_target_noise, prob_target_echo, prob_target_noise_echo.
           The seven ``prob_*`` values must sum to exactly 1.0.
  Layer C  Acoustic mixing (conditional sub-layers)
           C-0 RIR convolution (always runs)
           C-0b 以 early reference 标定 source volume，统一缩放 full/early/target
           C-1 Additive noise + C-2 wind noise + C-3 colored noise（按 branch 生效）
           C-aec Manifest ref/echo + GCC-PHAT alignment + SER scaling + random echo delay
  Layer D  Post-mix amplitude conditioning（固定 normalize → guard）
           D-normalize Joint peak normalization（mic 为 anchor，全部 AEC roles 同比随动）
           D-guard Deterministic peak guard（mic/nearend/echo/noise 共同缩放，farend 独立 guard）
  Layer E  URGENT2026 channel degradation (bandwidth / clipping / codec / packet loss)
           Applied **only** to noisy.
  Layer F  Post-mix distortion simulation (14 degradations, noisy only)；仅当 Layer E
           clipping 实际 applied 时跳过 F clipping，其他 F 操作保持执行

完整 post-mix 顺序：pre-finite -> D-normalize -> D-guard -> E -> F ->
task-post-EF -> final-guard -> post-finite。整体执行顺序为 A -> B -> C -> post-mix。

Training target: After Layer E/F, ``noisy`` (model input) is intentionally **not**
identical to the linear acoustic generation without E/F. The model learns to recover
the clean ``clean_target`` (from Layer B/C) from a degraded observation.
"""

from __future__ import annotations

from core.data.dynamic_dataset_base import (
    BaseDynamicMixingDataset,
    DataSummary,
    LayerEResult,
    SampleBuildResult,
    SourceSummary,
    signal_active_rms_dbfs,
    signal_rms_dbfs,
)

import copy
import json
import logging
import multiprocessing as mp
import os
import random
import time
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile
import torch
from scipy.signal import filtfilt
from core.data.simulation.urgent_impl import fast_filtfilt_fir

from tasks.aec.data.audio_path_utils import usable_audio_path
from tasks.aec.data.echo_clipping import (
    apply_absolute_rail_clip,
    apply_peak_relative_clip,
    apply_playback_overload,
    capture_point_of,
    normalize_echo_clipping_config,
    sample_absolute_clip_params,
    sample_peak_relative_clip_params,
    select_echo_clip_path,
    ser_db_from_power_ratio,
    summarize_absolute_rail_facts,
)
from tasks.aec.data.echo_device_eq import (
    ECHO_DEVICE_EQ_VARIANTS,
    apply_echo_device_eq,
    select_echo_device_eq_variant,
)
from tasks.aec.data.erl_coupling import (
    ERLCouplingContractError,
    resolve_erl_coupling_range,
)
from tasks.aec.data.post_mix_adapter import (
    ECHO as POST_MIX_ECHO,
    FAREND as POST_MIX_FAREND,
    MIC as POST_MIX_MIC,
    NEAREND as POST_MIX_NEAREND,
    NOISE as POST_MIX_NOISE,
    AECPostMixContext,
    AECPostMixResult,
    MicExtraNoiseResult,
    TaskPostMixAdapter,
    apply_scene_time_transform,
    project_training_tuple,
)
from core.data.acoustic_scene_plan import _thaw_json
from core.data.constants import DEFAULT_PLACEHOLDER_NUM_SAMPLES
from core.data.noise_roles import category_balance_name
from core.data.noise_source_policy import (
    NoiseApplicationRole,
    NoiseCacheProvenance,
    NoiseCategory,
    NoiseSourceFamily,
    NoiseSourcePolicy,
)
from core.data.simulation.channel_effects import (
    aggregate_layer_e_results,
    dispatch_observation_channels,
)
from core.data.simulation.energy_ratio import EnergyRatioScaler
from core.data.simulation.level_audit import LevelAudit
from core.data.post_mix_pipeline import PostMixContext, PostMixEngine
from core.data.simulation_trace import SimulationTrace
from core.data.waveform_bundle import WaveformBundle
from tasks.aec.data.state_label import (
    compute_aec_state_label,
    compute_aec_state_scenario_label,
)
from tasks.aec.inference.gcc_phat_core import (
    GccAlignEstimate,
    clamp_gcc_delay_samples,
    estimate_gcc_tau_windowed,
    resolve_gcc_downsample,
    trajectory_tau_at,
)
from core.data.errors import RecoverableResourceError
from tasks.aec.data.time_delay_jitter import (
    TimeDelayJitterConfig,
    apply_time_delay_jitter,
)
from core.data.simulation_audio_utils import (
    _AUDIO_CACHE_MAXSIZE,
    _ByteLRUCache,
    _align_with_window,
    _load_audio_cached,
    _load_audio_cached_impl,  # only for cache_clear() in set_epoch and normalized AEC pair paths
    _normalize_audio_cache_key,
    _read_cached_2d,
    _resample_audio,
)
from core.data.simulation_utils import (
    align_speech_with_silence_pad,
    fast_active_rms,
    rng_choice,
    rng_randint,
    rng_sample,
    rng_uniform,
    rng_uniform_or_beta,
    read_source_scp,
)
from core.data.augment import apply_speech_augmentations
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_views import simulation_view_for
from core.data.simulation.engine import mix_noise
from core.data.simulation.urgent_impl import detect_non_silence
from core.data.simulation.scene_spec import SceneSpec
from core.data.simulation.scene_names import validate_scene_name
from core.data.spatial import render_spatial_scene
from core.data.simulation.moving_scene import render_moving_spatial_scene
from core.data.simulation import scene_sampler
from core.data.simulation.rir_provider import (
    MovingSceneRIRRequest, MovingSpatialRIRTrackSet, SpatialRIRTrackSet,
)
from core.data.simulation.role_policy import AcousticActor, AcousticRole, RIRPolicy
from core.data.simulation.trajectory import (
    TrajectoryTimebase,
)

logger = logging.getLogger(__name__)


def _render_aec_moving_actor(tracks, request, waveform, actor_uid, *, early_boundary, reference_channel):
    """消费已请求的 moving bank，按 actor 渲染 full/direct/early，不再次采样。

    request 已固定房间绝对坐标；backend 不得再次平移或重采样。
    RIR 长度 L 与 waveform 长度 T 独立。
    """
    if not isinstance(tracks, MovingSpatialRIRTrackSet):
        raise TypeError("AEC moving provider must return MovingSpatialRIRTrackSet")
    plan = tracks.plan
    if tuple(plan.get("source_ids", ())) != request.source_uids:
        raise ValueError("AEC moving source order does not match request")
    if not plan.get("group_uid"):
        raise ValueError("AEC moving tracks require group_uid")
    if plan.get("scene_uid") != request.scene_uid:
        raise ValueError("AEC moving scene identity does not match request")
    if not np.allclose(plan["room_dim"], request.room["room_dim_m"], rtol=0, atol=1e-6):
        raise ValueError("AEC moving room does not match request")
    raw_clock = plan.get("trajectory_timebase", {})
    clock = TrajectoryTimebase(**{
        key: raw_clock[key] for key in
        ("sample_rate", "num_samples", "hop_samples", "frame_count", "start_sample")
        if key in raw_clock
    })
    if clock != request.timebase:
        raise ValueError("AEC moving timebase does not match request")
    microphones = np.asarray(plan["mic_loc"], dtype=np.float32)
    expected_mics = np.asarray(request.array["positions_m"], dtype=np.float32)
    if microphones.shape != expected_mics.shape:
        raise ValueError("AEC moving microphone geometry does not match request")
    positions = np.concatenate([
        trajectory.positions_m for trajectory in request.actor_trajectories.values()
    ], axis=1)
    if (np.shape(plan["trajectory"]) != positions.shape
            or not np.allclose(microphones, expected_mics, rtol=0, atol=1e-6)
            or not np.allclose(plan["trajectory"], positions, rtol=0, atol=1e-6)):
        raise ValueError("AEC moving actor trajectories/geometry do not match request")
    actor_index = request.source_uids.index(actor_uid)
    source = np.asarray(waveform, dtype=np.float32).reshape(1, -1)
    if source.shape[-1] != request.num_samples:
        raise ValueError("AEC moving waveform length does not match timebase")
    return render_moving_spatial_scene(
        source, tracks.full[:, actor_index:actor_index + 1],
        direct_rirs=tracks.direct[:, actor_index:actor_index + 1],
        frame_samples=clock.frame_samples, early_boundary=early_boundary,
        reference_channel=reference_channel, num_samples=request.num_samples,
    )


def _aec_pure_branch_flags(branch: str | None) -> tuple[bool, bool, bool, bool]:
    """返回 AEC canonical Layer-B branch 的纯场景标志。"""
    if branch == "prob_noise_only":
        return True, False, False, False
    if branch == "prob_echo_only":
        return False, True, False, True
    if branch == "prob_noise_echo_only":
        return False, False, True, True
    if branch == "prob_target_noise":
        return False, False, False, False
    if branch == "prob_target_only":
        return False, False, False, False
    return False, False, False, branch in {
        "prob_target_echo",
        "prob_target_noise_echo",
    }

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Exported waveform stem keys kept aligned with the model waveforms.
_LAYER_D_STEM_KEYS: tuple[str, ...] = (
    "layer_d_noisy",
    "layer_d_clean",
    "layer_d_noise",
    "layer_d_echo",
    "farend_ref",
    "final_noise",
)

_DELAY_TAIL_MIX_PROB: float = 0.25


def _build_aec_delivery_bundle(
    post_mix_result: AECPostMixResult,
    *,
    capture_stems: bool,
) -> WaveformBundle:
    """把 post-mix 结果投影为 scene-time 阶段消费的 delivery bundle。

    Layer-D 导出轨道只在 trace 请求捕获 stems 时附加；基础五个 AEC role
    与原有 ``_build_sample`` 路径保持相同。该 helper 不读取或推进随机数。
    """
    delivery_tracks = dict(post_mix_result.bundle.tracks)
    if capture_stems:
        layer_d_state = next(
            result.state
            for result in post_mix_result.trace.results
            if result.stage == "D-guard"
        )
        layer_d_stems = {
            "layer_d_noisy": layer_d_state.waveforms[POST_MIX_MIC],
            "layer_d_clean": layer_d_state.waveforms[POST_MIX_NEAREND],
            "layer_d_noise": layer_d_state.waveforms[POST_MIX_NOISE],
            "layer_d_echo": layer_d_state.waveforms[POST_MIX_ECHO],
            "farend_ref": layer_d_state.waveforms[POST_MIX_FAREND],
            "final_noise": post_mix_result.bundle.require(POST_MIX_NOISE),
        }
        for stem_key, waveform in layer_d_stems.items():
            delivery_tracks[f"export/{stem_key}"] = np.asarray(
                waveform, dtype=np.float32
            )
    return WaveformBundle(
        delivery_tracks, metadata=post_mix_result.bundle.metadata
    )



# =====================================================================
# AEC-specific helpers
# =====================================================================

def _load_stereo_farend_echo_impl(
    path: str,
    sr: int,
    strict: bool,
    ch_farend: int,
    ch_echo: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uncached core: load a multi-channel file and extract farend/echo channels.

    *path* must already be normalized (see ``load_stereo_farend_echo_cached``).
    ch_farend / ch_echo are **0-based** channel indices.
    Default convention for same-path stereo rows: **ch0 = echo**, **ch1 = farend**.
    """
    retry_delays = (0.02, 0.05)
    last_error: BaseException | None = None
    for attempt in range(len(retry_delays) + 1):
        try:
            data, orig_sr = soundfile.read(path, dtype="float32", always_2d=True)
            break
        except Exception as exc:
            last_error = exc
            if attempt >= len(retry_delays):
                raise RecoverableResourceError(
                    f"Failed to read stereo ref/echo file after {attempt + 1} attempts: {path}"
                ) from exc
            time.sleep(retry_delays[attempt])
    else:  # pragma: no cover - 循环必定 break 或 raise。
        raise RuntimeError(f"Failed to read stereo ref/echo file: {path}") from last_error

    nch = int(data.shape[1])
    need = max(int(ch_farend), int(ch_echo)) + 1
    if nch < need:
        raise RuntimeError(
            f"Stereo AEC file has {nch} channel(s), need at least {need} "
            f"for farend_ch={ch_farend}, echo_ch={ch_echo}: {path}"
        )

    wf = np.ascontiguousarray(data[:, ch_farend], dtype=np.float32)
    we = np.ascontiguousarray(data[:, ch_echo], dtype=np.float32)

    if orig_sr != sr:
        wf = _resample_audio(wf, orig_sr, sr).astype(np.float32, copy=False)
        we = _resample_audio(we, orig_sr, sr).astype(np.float32, copy=False)

    if strict and (not np.isfinite(wf).all() or not np.isfinite(we).all()):
        raise ValueError(f"Non-finite samples after load/resample: {path}")
    if not strict:
        if not np.isfinite(wf).all():
            wf = np.nan_to_num(wf, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(we).all():
            we = np.nan_to_num(we, nan=0.0, posinf=0.0, neginf=0.0)
    return wf, we


# AEC stereo farend/echo 缓存字节预算（默认 256 MiB；0 = 禁用缓存）
_AEC_STEREO_CACHE_BYTES = max(
    0,
    int(os.environ.get("USE_TRAIN_AEC_STEREO_CACHE_BYTES", str(256 * 1024 * 1024))),
)
_AEC_STEREO_CACHE = _ByteLRUCache(
    _AEC_STEREO_CACHE_BYTES, max(128, _AUDIO_CACHE_MAXSIZE // 2)
)


def _load_stereo_farend_echo_cached_impl(
    path: str,
    sr: int,
    strict: bool,
    ch_farend: int,
    ch_echo: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Byte-budget LRU wrapper around ``_load_stereo_farend_echo_impl``.

    Entry size = ``farend.nbytes + echo.nbytes``; budget env
    ``USE_TRAIN_AEC_STEREO_CACHE_BYTES`` (default 256 MiB per worker). Keeps the
    historical ``cache_clear()`` attribute so epoch-reset call sites stay unchanged.
    """
    key = (path, int(sr), bool(strict), int(ch_farend), int(ch_echo))
    cached = _AEC_STEREO_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        value = _load_stereo_farend_echo_impl(
            path, sr, strict, ch_farend, ch_echo
        )
    except Exception as exc:
        if strict:
            raise
        logging.getLogger(__name__).warning(
            "Failed to load stereo ref/echo (%s), using uncached silence placeholders",
            path,
            exc_info=exc,
        )
        placeholder = np.zeros(max(int(sr) * 2, 1), dtype=np.float32)
        return placeholder, placeholder.copy()
    _AEC_STEREO_CACHE.put(key, value)
    return value


def _clear_aec_stereo_cache() -> None:
    """Clear the stereo cache (dynamic lookup, monkeypatch-safe)."""
    _AEC_STEREO_CACHE.clear()


_load_stereo_farend_echo_cached_impl.cache_clear = _clear_aec_stereo_cache  # type: ignore[attr-defined]


def load_stereo_farend_echo_cached(
    path: str,
    sr: int,
    strict: bool,
    ch_farend: int,
    ch_echo: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a multi-channel file and return ``(farend_1d, echo_1d)`` at *sr*.

    Cached separately from mono ``_load_audio_cached`` so stereo paths are not
    collapsed to ch0. Paths are normalized before LRU lookup.
    """
    return _load_stereo_farend_echo_cached_impl(
        _normalize_audio_cache_key(path), sr, strict, ch_farend, ch_echo
    )


def scale_signal_to_dBfs_rms(
    signal: np.ndarray,
    target_db: float,
) -> np.ndarray:
    """Scale *signal* so its RMS level equals *target_db* dBFS.

    ``target_db`` is dB relative to full-scale (±1.0), referenced to RMS power.
    For example, ``target_db = −26`` produces a signal whose RMS = 10**(−26/20) ≈ 0.05.
    """
    raw = np.asarray(signal)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError("scale_signal_to_dBfs_rms requires a real numeric signal")
    raw64 = raw.astype(np.float64, copy=False).ravel()
    float32_max = float(np.finfo(np.float32).max)
    raw_peak = float(np.abs(raw64).max()) if raw64.size else 0.0
    if not np.isfinite(raw64).all() or raw_peak > float32_max:
        raise ValueError(
            "scale_signal_to_dBfs_rms requires finite values representable in float32"
        )
    sig = raw64.astype(np.float32)
    # float32 平方在高振幅输入下会先溢出；功率统计必须提升精度后再缩放。
    with np.errstate(over="ignore", invalid="ignore"):
        power = float(np.mean(raw64 * raw64))
    if not np.isfinite(power):
        raise ValueError("scale_signal_to_dBfs_rms input power must be finite")
    if power < 1e-12:
        return np.zeros_like(sig, dtype=np.float32)
    scale = EnergyRatioScaler.target_rms_gain(power, target_db)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = raw64 * scale
    scaled_peak = float(np.abs(scaled).max()) if scaled.size else 0.0
    if not np.isfinite(scaled).all() or scaled_peak > float32_max:
        raise ValueError("scale_signal_to_dBfs_rms produced non-float32-finite output")
    return scaled.astype(np.float32)


def _finite_power_db_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> float | None:
    """返回全段功率比 dB；任一侧为空、静音或非有限时返回 ``None``。"""
    num = np.asarray(numerator, dtype=np.float64).ravel()
    den = np.asarray(denominator, dtype=np.float64).ravel()
    length = min(num.size, den.size)
    if length <= 0:
        return None
    try:
        num_power = EnergyRatioScaler.measure(
            num[:length], policy="full_mean_power"
        )
        den_power = EnergyRatioScaler.measure(
            den[:length], policy="full_mean_power"
        )
    except ValueError:
        return None
    if num_power <= 1e-12 or den_power <= 1e-12:
        return None
    value = 10.0 * float(np.log10(num_power / den_power))
    return value if np.isfinite(value) else None


def _active_power_db_ratio(
    reference: np.ndarray,
    noise: np.ndarray,
) -> float | None:
    """复用 ``mix_noise`` 的各自 active-region 口径重测 aggregate SNR。"""
    ref = np.asarray(reference, dtype=np.float32).reshape(1, -1)
    nz = np.asarray(noise, dtype=np.float32).reshape(1, -1)
    length = min(ref.shape[1], nz.shape[1])
    if length <= 0:
        return None
    ref = ref[:, :length]
    nz = nz[:, :length]
    ref_active = ref[:, detect_non_silence(ref)]
    noise_active = nz[:, detect_non_silence(nz)]
    if ref_active.size <= 0 or noise_active.size <= 0:
        return None
    return _finite_power_db_ratio(ref_active, noise_active)


def _source_volume_calibration_export(raw: dict | None) -> dict[str, Any]:
    """规范化 C-0b sidecar，并保留 post_augment 的权威扩展字段。"""
    stable: dict[str, Any] = {
        "target_db": None,
        "target_requested_db": None,
        "reference_db_before": None,
        "realized_after_db": None,
        "gain": None,
        "gain_clipped": None,
        "status": "disabled",
        "reason": None,
        "forced_gain": False,
        "reference": None,
        "stage": None,
    }
    if not isinstance(raw, dict):
        return stable

    def json_finite(value: Any) -> Any:
        """递归转换 producer 扩展为 JSON-safe 值，非有限数统一写 null。"""
        if value is None or isinstance(value, (str, bool, np.bool_)):
            return bool(value) if isinstance(value, np.bool_) else value
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            result = float(value)
            return result if np.isfinite(result) else None
        if isinstance(value, dict):
            return {str(key): json_finite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            items = value.tolist() if isinstance(value, np.ndarray) else value
            return [json_finite(item) for item in items]
        return None

    stable.update({str(key): json_finite(value) for key, value in raw.items()})

    def finite_float(value: Any) -> float | None:
        if value is None or isinstance(value, (bool, np.bool_)):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if np.isfinite(result) else None

    def strict_bool(value: Any) -> bool | None:
        return bool(value) if isinstance(value, (bool, np.bool_)) else None

    if "target_db_requested" in raw:
        target_requested = finite_float(raw.get("target_db_requested"))
    elif "target_requested_db" in raw:
        target_requested = finite_float(raw.get("target_requested_db"))
    else:
        target_requested = finite_float(raw.get("target_db"))
    # ``target_db`` 是公开旧别名；canonical requested 值始终以 producer 新键优先。
    target_db = target_requested

    reference_before = finite_float(raw.get("reference_db_before"))
    gain = finite_float(raw.get("gain"))
    if "reference_db_after" in raw:
        realized_after = finite_float(raw.get("reference_db_after"))
    elif "realized_after_db" in raw:
        realized_after = finite_float(raw.get("realized_after_db"))
    else:
        realized_after = (
            reference_before + 20.0 * float(np.log10(gain))
            if reference_before is not None and gain is not None and gain > 0.0
            else None
        )
        if realized_after is not None and not np.isfinite(realized_after):
            realized_after = None

    forced_gain = (
        strict_bool(raw.get("forced_gain")) if "forced_gain" in raw else False
    )
    valid = (
        target_requested is not None
        and reference_before is not None
        and gain is not None
        and gain > 0.0
        and realized_after is not None
    )
    if "gain_clipped" in raw:
        gain_clipped = strict_bool(raw.get("gain_clipped"))
    else:
        gain_at_limit = (
            gain is not None
            and (
                np.isclose(gain, 1e-3, rtol=0.0, atol=1e-12)
                or np.isclose(gain, 1e3, rtol=0.0, atol=1e-9)
            )
        )
        target_missed = (
            valid
            and not np.isclose(realized_after, target_requested, rtol=0.0, atol=1e-3)
        )
        if valid and forced_gain is False:
            gain_clipped = bool(gain_at_limit and target_missed)
        elif valid and forced_gain is True:
            gain_clipped = False
        else:
            gain_clipped = None

    known_statuses = {"applied", "skipped", "gain_clipped", "disabled", "invalid"}
    if "status" in raw:
        raw_status = raw.get("status")
        status = raw_status if raw_status in known_statuses else "invalid"
    elif not valid:
        status = "invalid"
    else:
        status = "gain_clipped" if gain_clipped else "applied"

    raw_reason = raw.get("reason")
    reason = raw_reason if raw_reason is None or isinstance(raw_reason, str) else None
    stable.update(
        {
            "target_db": target_db,
            "target_requested_db": target_requested,
            "reference_db_before": reference_before,
            "realized_after_db": realized_after,
            "gain": gain,
            "gain_clipped": gain_clipped,
            "status": status,
            "reason": reason,
            "forced_gain": forced_gain,
            "reference": raw.get("reference") if isinstance(raw.get("reference"), str) else None,
            "stage": raw.get("stage") if isinstance(raw.get("stage"), str) else None,
        }
    )
    return stable


def _peak_rms_ratio_audit(
    full: np.ndarray | None,
    early: np.ndarray | None,
) -> dict[str, float | str | None]:
    """输出 C-0 full/early 峰值与 RMS 比率，静音时用 reason code 封闭。"""
    if full is None or early is None:
        return {
            "full_peak_abs": None,
            "early_peak_abs": None,
            "full_to_early_peak_ratio": None,
            "full_to_early_peak_db": None,
            "full_rms": None,
            "early_rms": None,
            "full_to_early_rms_ratio": None,
            "full_to_early_rms_db": None,
            "reason_code": "c0_stems_unavailable",
        }
    full_arr = np.asarray(full, dtype=np.float64).ravel()
    early_arr = np.asarray(early, dtype=np.float64).ravel()
    length = min(full_arr.size, early_arr.size)
    if length <= 0:
        return _peak_rms_ratio_audit(None, None)
    full_arr = full_arr[:length]
    early_arr = early_arr[:length]
    full_peak = float(np.max(np.abs(full_arr)))
    early_peak = float(np.max(np.abs(early_arr)))
    full_rms = float(np.sqrt(np.mean(full_arr ** 2)))
    early_rms = float(np.sqrt(np.mean(early_arr ** 2)))
    valid = early_peak > 1e-12 and early_rms > 1e-12
    peak_ratio = float(full_peak / early_peak) if valid else None
    rms_ratio = float(full_rms / early_rms) if valid else None
    return {
        "full_peak_abs": full_peak,
        "early_peak_abs": early_peak,
        "full_to_early_peak_ratio": peak_ratio,
        "full_to_early_peak_db": (
            20.0 * float(np.log10(peak_ratio)) if peak_ratio and peak_ratio > 0.0 else None
        ),
        "full_rms": full_rms,
        "early_rms": early_rms,
        "full_to_early_rms_ratio": rms_ratio,
        "full_to_early_rms_db": (
            20.0 * float(np.log10(rms_ratio)) if rms_ratio and rms_ratio > 0.0 else None
        ),
        "reason_code": None if valid else "early_stem_silent",
    }


def _waveform_mismatch_reason(
    observed: np.ndarray,
    reference: np.ndarray,
    *,
    nonlinear_reason: str,
    gain_reason: str,
) -> str | None:
    """区分实际非线性形变与单一增益失配；一致时返回 ``None``。"""
    obs = np.asarray(observed, dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()
    length = min(obs.size, ref.size)
    if length <= 0 or not np.isfinite(obs[:length]).all() or not np.isfinite(ref[:length]).all():
        return nonlinear_reason
    obs = obs[:length]
    ref = ref[:length]
    if np.allclose(obs, ref, rtol=1e-6, atol=1e-7):
        return None
    ref_power = float(np.dot(ref, ref))
    if ref_power <= 1e-12:
        return nonlinear_reason
    best_gain = float(np.dot(obs, ref) / ref_power)
    residual = obs - best_gain * ref
    if np.allclose(residual, 0.0, rtol=0.0, atol=1e-7):
        return gain_reason
    return nonlinear_reason


def _reconstruction_reference_for_branch(
    branch: str,
    task: str,
    clean: np.ndarray,
    echo: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    """按实际 Layer-B 组成构建审计参考，不进入训练 tuple。"""
    cl = np.asarray(clean, dtype=np.float32).ravel()
    ec = np.asarray(echo, dtype=np.float32).ravel()
    nz = np.asarray(noise, dtype=np.float32).ravel()
    length = min(cl.size, ec.size, nz.size)
    cl, ec, nz = cl[:length], ec[:length], nz[:length]
    if branch == "prob_echo_only":
        return ec.copy()
    if branch == "prob_noise_echo_only":
        return (cl + ec).astype(np.float32) if task == "aec" else (nz + ec).astype(np.float32)
    if branch == "prob_target_echo":
        return (cl + ec).astype(np.float32)
    if branch == "prob_target_noise_echo":
        return (cl + ec).astype(np.float32) if task == "aec" else (cl + nz + ec).astype(np.float32)
    if branch == "prob_noise_only":
        return cl.copy() if task == "aec" else nz.copy()
    if branch == "prob_target_noise":
        return cl.copy() if task == "aec" else (cl + nz).astype(np.float32)
    return cl.copy()


def _measure_realized_levels(
    clean: np.ndarray,
    noisy: np.ndarray,
    echo: np.ndarray | None,
    farend: np.ndarray | None,
    noise: np.ndarray | None,
    *,
    measure_point: str = "post_dataset_post_mix",
    decomposable: bool | None = None,
    decomposable_reason_codes: list[str] | tuple[str, ...] | None = None,
    branch: str | None = None,
    task: str = "joint",
    reconstruction_reference: np.ndarray | None = None,
) -> dict:
    """在 export 路径重测 realized 电平，并显式标记指标适用性。

    SER 只对含近端的 echo 分支有定义；pure echo/noise_echo 保留 latent SER
    draw，但 realized SER 为 N/A。ERL 仅依赖实际 farend/echo stem，因此不随
    mic 的可分解性一起失效。SNR 和 reconstruction residual 在不可分解时
    fail-closed，避免把 pre-E/F stem 比值误称为最终观测指标。AEC 任务的
    training target 包含透传噪声，SER/SNR 必须先减去 noise stem，恢复实际
    nearend component，不能把 target/noise 或 target/echo 比值误标成声学指标。
    """
    mic = np.asarray(noisy, dtype=np.float32).ravel()
    cl = np.asarray(clean, dtype=np.float32).ravel()
    nz = (
        np.asarray(noise, dtype=np.float32).ravel()
        if noise is not None
        else np.zeros(0, dtype=np.float32)
    )
    ratio_reference = cl
    ratio_reference_name = "training_target"
    if task == "aec":
        ratio_reference_name = "nearend_component"
        if nz.size == cl.size:
            ratio_reference = (cl - nz).astype(np.float32, copy=False)
        elif nz.size:
            ratio_reference = np.zeros(0, dtype=np.float32)
    reason_codes = [str(code) for code in (decomposable_reason_codes or [])]
    out: dict = {
        "clean_dbfs_rms": float(signal_rms_dbfs(cl)),
        "mic_dbfs_rms": float(signal_rms_dbfs(mic)),
        "noise_dbfs_rms": float(signal_rms_dbfs(nz)),
        "clean_active_dbfs": float(signal_active_rms_dbfs(cl)),
        "mic_active_dbfs": float(signal_active_rms_dbfs(mic)),
        "mic_peak_abs": float(np.max(np.abs(mic))) if mic.size else 0.0,
        "mic_clip_frame_ratio": (
            float(np.mean(np.abs(mic) >= 1.0 - 0.01)) if mic.size else 0.0
        ),
        "mic_near_fullscale_ratio": (
            float(np.mean(np.abs(mic) > 0.99)) if mic.size else 0.0
        ),
        "measure_point": str(measure_point),
        "decomposable": decomposable,
        "decomposable_reason_codes": reason_codes,
        "branch": branch,
        "task": str(task),
        "ratio_reference": ratio_reference_name,
    }

    echo_arr = np.asarray(echo, dtype=np.float32).ravel() if echo is not None else None
    has_echo = (
        echo_arr is not None
        and echo_arr.size > 0
        and float(np.mean(echo_arr.astype(np.float64) ** 2)) > 1e-12
    )
    far_arr = (
        np.asarray(farend, dtype=np.float32).ravel()
        if farend is not None
        else np.zeros(0, dtype=np.float32)
    )
    has_farend = (
        far_arr.size > 0
        and float(np.mean(far_arr.astype(np.float64) ** 2)) > 1e-12
    )
    pure_echo_branch = branch in ("prob_echo_only", "prob_noise_echo_only")
    nearend_present = (
        ratio_reference.size > 0
        and float(np.mean(ratio_reference.astype(np.float64) ** 2)) > 1e-12
    )
    noise_present = (
        nz.size > 0 and float(np.mean(nz.astype(np.float64) ** 2)) > 1e-12
    )

    if not has_echo:
        ser_reason = "echo_stem_unavailable"
    elif pure_echo_branch or not nearend_present:
        ser_reason = "scenario_has_no_nearend"
    elif decomposable is False:
        ser_reason = "mic_not_decomposable"
    else:
        ser_reason = None
    ser_db = (
        _finite_power_db_ratio(ratio_reference, echo_arr)
        if ser_reason is None and echo_arr is not None
        else None
    )

    if not has_echo:
        erl_reason = "echo_stem_unavailable"
    elif not has_farend:
        erl_reason = "farend_stem_unavailable"
    else:
        erl_reason = None
    erl_db = (
        _finite_power_db_ratio(far_arr, echo_arr)
        if erl_reason is None and echo_arr is not None
        else None
    )

    if not nearend_present:
        snr_reason = "scenario_has_no_nearend"
    elif not noise_present:
        snr_reason = "noise_stem_unavailable"
    elif decomposable is False:
        snr_reason = "mic_not_decomposable"
    else:
        snr_reason = None
    realized_snr_db = (
        _finite_power_db_ratio(ratio_reference, nz)
        if snr_reason is None
        else None
    )

    out.update(
        {
            "echo_dbfs_rms": float(signal_rms_dbfs(echo_arr)) if has_echo else None,
            "far_dbfs_rms": float(signal_rms_dbfs(far_arr)) if has_echo and has_farend else None,
            "ser_db": ser_db,
            "ser_applicable": ser_reason is None,
            "ser_reason_code": ser_reason,
            "erl_db": erl_db,
            "erl_applicable": erl_reason is None,
            "erl_reason_code": erl_reason,
            "snr_db": realized_snr_db,
            "snr_applicable": snr_reason is None,
            "snr_reason_code": snr_reason,
            "level_audits": {
                "snr": LevelAudit(
                    name="aec.snr",
                    candidate_drawn_db=None,
                    selected_nominal_db=None,
                    calibration_realized_db=None,
                    final_realized_db=realized_snr_db,
                    numerator=ratio_reference_name,
                    denominator="noise_stem",
                    policy="full_mean_power",
                    calibration_measure_point="not_available_in_final_measurement",
                    final_measure_point=str(measure_point),
                    applicable=snr_reason is None,
                    reason=snr_reason,
                ).to_dict(),
                "ser": LevelAudit(
                    name="aec.ser",
                    candidate_drawn_db=None,
                    selected_nominal_db=None,
                    calibration_realized_db=None,
                    final_realized_db=ser_db,
                    numerator=ratio_reference_name,
                    denominator="echo_stem",
                    policy="full_mean_power",
                    calibration_measure_point="not_available_in_final_measurement",
                    final_measure_point=str(measure_point),
                    applicable=ser_reason is None,
                    reason=ser_reason,
                ).to_dict(),
                "erl": LevelAudit(
                    name="aec.erl",
                    candidate_drawn_db=None,
                    selected_nominal_db=None,
                    calibration_realized_db=None,
                    final_realized_db=erl_db,
                    numerator="farend_stem",
                    denominator="echo_stem",
                    policy="full_mean_power",
                    calibration_measure_point="not_available_in_final_measurement",
                    final_measure_point=str(measure_point),
                    applicable=erl_reason is None,
                    reason=erl_reason,
                ).to_dict(),
            },
            "echo_active_dbfs": (
                float(signal_active_rms_dbfs(echo_arr)) if has_echo else None
            ),
            "farend_peak_abs": (
                float(np.max(np.abs(far_arr))) if has_echo and far_arr.size else None
            ),
        }
    )

    residual_reason: str | None = None
    residual_rms: float | None = None
    residual_peak: float | None = None
    if reconstruction_reference is None:
        residual_reason = "reconstruction_reference_unavailable"
    elif decomposable is False:
        residual_reason = "mic_not_decomposable"
    else:
        ref = np.asarray(reconstruction_reference, dtype=np.float32).ravel()
        length = min(mic.size, ref.size)
        if length <= 0:
            residual_reason = "reconstruction_reference_unavailable"
        else:
            residual = mic[:length].astype(np.float64) - ref[:length].astype(np.float64)
            residual_rms = float(np.sqrt(np.mean(residual ** 2)))
            residual_peak = float(np.max(np.abs(residual)))
    out["reconstruction_residual"] = {
        "rms": residual_rms,
        "peak_abs": residual_peak,
        "applicable": residual_reason is None,
        "reason_code": residual_reason,
    }
    return out


def parse_aec_dynamic_pipeline(raw: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Parse and strictly validate the ``aec_dynamic_pipeline`` subsection.

    Currently only ``mic_extra_noise`` is recognized.
    """
    out: dict[str, Any] = {
        "mic_extra_noise_enabled": False,
        "mic_extra_noise_snr_range": (5.0, 25.0),
    }
    if raw is None:
        return out
    if not isinstance(raw, Mapping):
        raise ValueError("aec_dynamic_pipeline must be a mapping")
    unknown_top_level = set(raw) - {"mic_extra_noise"}
    if unknown_top_level:
        raise ValueError(
            "aec_dynamic_pipeline contains unknown keys: "
            f"{sorted(map(str, unknown_top_level))!r}"
        )
    if not raw:
        return out

    if "mic_extra_noise" not in raw:
        return out
    mex = raw["mic_extra_noise"]
    if not isinstance(mex, Mapping):
        raise ValueError("aec_dynamic_pipeline.mic_extra_noise must be a mapping")
    unknown_mic_extra_noise = set(mex) - {"enabled", "snr_range"}
    if unknown_mic_extra_noise:
        raise ValueError(
            "aec_dynamic_pipeline.mic_extra_noise contains unknown keys: "
            f"{sorted(map(str, unknown_mic_extra_noise))!r}"
        )

    if "enabled" in mex:
        enabled = mex["enabled"]
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError(
                "aec_dynamic_pipeline.mic_extra_noise.enabled must be a boolean"
            )
        out["mic_extra_noise_enabled"] = bool(enabled)

    if "snr_range" in mex:
        snr_range = mex["snr_range"]
        if not isinstance(snr_range, (list, tuple)) or len(snr_range) != 2:
            raise ValueError(
                "aec_dynamic_pipeline.mic_extra_noise.snr_range must be a two-item list or tuple"
            )
        lo_raw, hi_raw = snr_range
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in (lo_raw, hi_raw)
        ):
            raise ValueError(
                "aec_dynamic_pipeline.mic_extra_noise.snr_range values must be finite real numbers"
            )
        try:
            lo, hi = float(lo_raw), float(hi_raw)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                "aec_dynamic_pipeline.mic_extra_noise.snr_range values must be finite real numbers"
            ) from exc
        if not np.isfinite((lo, hi)).all():
            raise ValueError(
                "aec_dynamic_pipeline.mic_extra_noise.snr_range values must be finite real numbers"
            )
        if lo > hi:
            raise ValueError(
                "aec_dynamic_pipeline.mic_extra_noise.snr_range must satisfy lo <= hi"
            )
        out["mic_extra_noise_snr_range"] = (lo, hi)
    return out


# ---------------------------------------------------------------------------
# JSON farend/echo pair manifest parsers
# ---------------------------------------------------------------------------

def _parse_farend_echo_pairs_from_json_obj(obj: Any) -> list[tuple[str, str]]:
    """Parse the canonical stereo echo-pair manifest schema.

    Only ``{"pairs": ["stereo.wav"], "meta": {...}}`` is accepted.  A path
    string maps to ``(path, path)``; stereo channel extraction remains the
    runtime contract (ch0=echo, ch1=farend).
    """
    if not isinstance(obj, dict):
        raise ValueError(
            "JSON farend/echo manifest root must be an object with 'pairs'"
        )
    unknown_root = set(obj) - {"pairs", "meta"}
    if unknown_root:
        raise ValueError(
            "JSON root object contains unknown keys: "
            f"{sorted(map(str, unknown_root))!r}"
        )
    if "meta" in obj and not isinstance(obj["meta"], dict):
        raise ValueError("JSON root meta must be an object when provided")
    if "pairs" not in obj:
        raise ValueError('JSON root object must contain "pairs"')
    pairs = obj["pairs"]
    if not isinstance(pairs, list):
        raise ValueError('JSON: "pairs" must be a list')

    out: list[tuple[str, str]] = []
    for index, item in enumerate(pairs):
        if not isinstance(item, str):
            raise ValueError(
                f'JSON "pairs"[{index}] must be a non-empty stereo path string'
            )
        path = item.strip()
        if not path:
            raise ValueError(
                f'JSON "pairs"[{index}] must be a non-empty stereo path string'
            )
        out.append((path, path))
    return out


def parse_real_farend_echo_pair_file(path: Path) -> list[tuple[str, str]]:
    """Load canonical stereo ref/echo paths from a JSON manifest."""
    if path.suffix.lower() != ".json":
        raise ValueError(
            f"real_farend_echo_pair_json must be a .json manifest: {path}"
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc
    pairs = _parse_farend_echo_pairs_from_json_obj(obj)
    if not pairs:
        raise ValueError(f"JSON manifest produced no valid stereo paths: {path}")
    return pairs


# =====================================================================
# 共享配置校验（配置检查路径与真实构建路径共用）
# =====================================================================

def _check_aec_scenario_probs(cfg: SimulationConfigs) -> None:
    """AEC 七个 Layer-B 场景概率各自 ∈ [0, 1] 且之和 = 1。"""
    def _prob_unit_interval(name: str, raw: float) -> float:
        v = float(raw)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"simulation_cfg {name} must be in [0.0, 1.0], got {raw!r}")
        return v

    scenario = simulation_view_for(cfg, task="aec").scenario
    _pc = float(scenario.prob_target_only)
    _pn = float(scenario.prob_noise_only)
    _pe = float(scenario.prob_echo_only)
    _pne = float(scenario.prob_noise_echo_only)
    _pnn = float(scenario.prob_target_noise)
    _pne2 = float(scenario.prob_target_echo)
    _pnne = float(scenario.prob_target_noise_echo)
    p_clean = _prob_unit_interval("prob_target_only", _pc)
    p_noise = _prob_unit_interval("prob_noise_only", _pn)
    p_echo = _prob_unit_interval("prob_echo_only", _pe)
    p_noise_echo = _prob_unit_interval("prob_noise_echo_only", _pne)
    p_nn = _prob_unit_interval("prob_target_noise", _pnn)
    p_ne = _prob_unit_interval("prob_target_echo", _pne2)
    p_nne = _prob_unit_interval("prob_target_noise_echo", _pnne)
    _sum_all = p_clean + p_noise + p_echo + p_noise_echo + p_nn + p_ne + p_nne
    if abs(_sum_all - 1.0) > 1e-9:
        raise ValueError(
            f"AEC scenario probs must sum to 1.0 (got {_sum_all:.8f}). "
            f"clean={p_clean:.6f}, noise={p_noise:.6f}, echo={p_echo:.6f}, "
            f"noise_echo={p_noise_echo:.6f}, ne_noise={p_nn:.6f}, "
            f"ne_echo={p_ne:.6f}, ne_noise_echo={p_nne:.6f}"
        )


def _resolve_aec_time_delay_jitter_config(
    aec_time_delay_jitter: Mapping[str, Any] | None,
    aec_ref_echo_delay_range_ms: Optional[list],
) -> TimeDelayJitterConfig:
    """从基础随机延时范围派生 jitter 的相对与总延时上限。

    ``max_relative_jitter_ms`` 限制相对基础延时的 jitter 偏移；运行时的
    ``max_total_delay_ms`` 仍将 ``base_delay + jitter`` 裁剪到
    ``[0, delay_range_max]``。存在 ``aec_ref_echo_delay_range_ms`` 时，其最大值
    是两项上限的唯一预算，故覆盖 YAML 中的 canonical 手工值，避免漂移。
    """
    if aec_time_delay_jitter is not None and not isinstance(
        aec_time_delay_jitter, Mapping
    ):
        raise ValueError(
            "aec_time_delay_jitter must be a mapping, "
            f"got {type(aec_time_delay_jitter).__name__}"
        )
    jitter_cfg = dict(aec_time_delay_jitter or {})
    legacy_key = "max_abs_delay_ms"
    canonical_key = "max_relative_jitter_ms"
    if legacy_key in jitter_cfg:
        raise ValueError(
            f"aec_time_delay_jitter.{legacy_key} was removed; use "
            f"aec_time_delay_jitter.{canonical_key} instead."
        )
    if aec_ref_echo_delay_range_ms is not None:
        jitter_cfg[canonical_key] = float(aec_ref_echo_delay_range_ms[1])
    return TimeDelayJitterConfig.from_dict(jitter_cfg)


def validate_aec_dataset_config(
    *,
    simulation_cfg: Optional[dict] = None,
    aec_dynamic_pipeline: Optional[Mapping[str, Any]] = None,
    task: str = "joint",
    ser_range: Optional[list] = None,
    aec_ref_echo_gcc_farend_pad_policy: Optional[str] = None,
    aec_ref_echo_fine_gcc_align: bool = True,
    aec_ref_echo_gcc_bias_seconds: float = 0.001,
    aec_ref_echo_gcc_max_tau_seconds: float = 0.2,
    aec_ref_echo_gcc_max_pad_seconds: float = 0.2,
    aec_ref_echo_gcc_min_pad_samples: int = 0,
    aec_ref_echo_gcc_peak_threshold: float = 0.05,
    aec_ref_echo_gcc_prominence_db: float = 6.0,
    aec_ref_echo_gcc_tv_iqr_ms: float = 10.0,
    aec_ref_echo_gcc_max_windows: int = 8,
    aec_ref_echo_gcc_downsample: Optional[int] = None,
    aec_ref_echo_random_delay_prob: float = 0.0,
    aec_ref_echo_random_delay_mode: str = "echo_only",
    aec_ref_echo_random_delay_distribution: str = "uniform",
    aec_ref_echo_random_delay_beta_a: float = 1.0,
    aec_ref_echo_random_delay_beta_b: float = 3.0,
    aec_ref_echo_random_delay_uniform_prob: float = _DELAY_TAIL_MIX_PROB,
    aec_ref_echo_delay_range_ms: Optional[list] = None,
    aec_time_delay_jitter: dict | None = None,
    aec_echo_device_eq: dict | None = None,
    aec_echo_clipping: dict | None = None,
    state_detector_hop_length: int | None = None,
    state_detector_win_length: int | None = None,
    state_detector_near_threshold_db: float = -40.0,
    state_detector_echo_threshold_db: float = -50.0,
    state_detector_noise_threshold_db: float = -40.0,
    state_detector_hangover_frames: int = 1,
    state_detector_absolute_floor: float = 1e-4,
    state_detector_pre_emphasis: float = 0.97,
    state_detector_label_type: str = "scenario",
    state_detector_smoothing_window_frames: int = 1,
    simulation_cfg_strict: bool = True,
    **_ignored,
) -> None:
    """校验 AEC ``DynamicMixingDataset`` 的配置（纯配置，不触碰文件系统）。

    “只做配置检查”的路径（``AECTrainingEntry._validate_config`` →
    ``tasks.aec.data.dynamic_aec_build.validate_aec_dynamic_aec_config``）与真正构建
    dataset 的路径（``DynamicMixingDataset.__init__``）调用**同一个函数**，
    消除“配置检查通过但构建失败”（或反之）的鸿沟。参数默认值与
    ``DynamicMixingDataset.__init__`` 保持一致。
    """
    retired_aec_frontend = {
        "aec_stream_align": "aec_frontend.delay_alignment.enabled",
        "linear_aec": "aec_frontend.linear_aec",
    }
    for key, replacement in retired_aec_frontend.items():
        if key in _ignored:
            raise ValueError(
                f"{key} is no longer supported by DynamicMixingDataset; configure "
                f"top-level {replacement} instead. The AEC input frontend runs only in Trainer/Inference."
            )

    if (
        isinstance(aec_time_delay_jitter, Mapping)
        and "max_abs_delay_ms" in aec_time_delay_jitter
    ):
        raise ValueError(
            "aec_time_delay_jitter.max_abs_delay_ms was removed; use "
            "aec_time_delay_jitter.max_relative_jitter_ms instead."
        )

    # ---- aec_dynamic_pipeline：与 Dataset 构造共享唯一解析器 ----
    parse_aec_dynamic_pipeline(aec_dynamic_pipeline)

    # ---- simulation_cfg：废弃键 fail-fast + SimulationConfigs 统一校验 + 场景概率和=1 ----
    sim_in = dict(simulation_cfg) if simulation_cfg else {}
    for key, replacement in retired_aec_frontend.items():
        if key in sim_in:
            value = sim_in.pop(key)
            raise ValueError(
                f"simulation_cfg.{key} is no longer supported by DynamicMixingDataset; "
                f"configure top-level {replacement} instead. The AEC input frontend runs only in Trainer/Inference."
            )
    if sim_in.pop("aec_scenario_probs", None) is not None:
        raise ValueError(
            "simulation_cfg.aec_scenario_probs was removed. Use prob_* keys directly."
        )
    if sim_in.pop("prob_pure_echo", None) is not None:
        raise ValueError(
            "simulation_cfg.prob_pure_echo was removed. Use prob_echo_only instead."
        )
    sim = SimulationConfigs.from_dict(sim_in, strict=simulation_cfg_strict)
    _check_aec_scenario_probs(sim)

    # ---- task ----
    _task = str(task).strip().lower() if task is not None else "joint"
    if _task not in ("aec", "joint"):
        raise ValueError(f"task must be 'aec' or 'joint', got {task!r}")

    if type(aec_ref_echo_fine_gcc_align) is not bool:
        raise ValueError(
            "aec_ref_echo_fine_gcc_align must be a bool, "
            f"got {aec_ref_echo_fine_gcc_align!r}"
        )

    state_detector_thresholds = {
        "state_detector_near_threshold_db": state_detector_near_threshold_db,
        "state_detector_echo_threshold_db": state_detector_echo_threshold_db,
        "state_detector_noise_threshold_db": state_detector_noise_threshold_db,
    }
    invalid_state_detector_thresholds = []
    for name, value in state_detector_thresholds.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            invalid_state_detector_thresholds.append(name)
            continue
        try:
            finite = np.isfinite(float(value))
        except (OverflowError, TypeError, ValueError):
            finite = False
        if not finite:
            invalid_state_detector_thresholds.append(name)
    if invalid_state_detector_thresholds:
        raise ValueError(
            "AEC state detector thresholds must be finite real numbers: "
            + ", ".join(invalid_state_detector_thresholds)
        )

    # ---- 帧级 state detector 标签参数 ----
    if state_detector_hop_length is not None and int(state_detector_hop_length) <= 0:
        raise ValueError(
            f"state_detector_hop_length must be positive, got {state_detector_hop_length!r}"
        )
    if state_detector_win_length is not None and int(state_detector_win_length) <= 0:
        raise ValueError(
            f"state_detector_win_length must be positive, got {state_detector_win_length!r}"
        )
    if int(state_detector_hangover_frames) < 0:
        raise ValueError(
            f"state_detector_hangover_frames must be >= 0, got {state_detector_hangover_frames!r}"
        )
    if float(state_detector_absolute_floor) < 0:
        raise ValueError(
            f"state_detector_absolute_floor must be >= 0, got {state_detector_absolute_floor!r}"
        )
    if not (0.0 <= float(state_detector_pre_emphasis) < 1.0):
        raise ValueError(
            f"state_detector_pre_emphasis must be in [0.0, 1.0), got {state_detector_pre_emphasis!r}"
        )
    _label_type = str(state_detector_label_type).strip().lower()
    if _label_type not in ("scenario", "fst_dt_nst"):
        raise ValueError(
            f"state_detector_label_type must be 'scenario' or 'fst_dt_nst', "
            f"got {state_detector_label_type!r}"
        )
    if int(state_detector_smoothing_window_frames) < 0:
        raise ValueError(
            f"state_detector_smoothing_window_frames must be >= 0, "
            f"got {state_detector_smoothing_window_frames!r}"
        )

    # ---- SER bounds（构造参数 ser_range 覆盖 simulation_cfg.ser_range）----
    if ser_range is not None:
        if not isinstance(ser_range, (list, tuple)) or len(ser_range) != 2:
            raise ValueError(
                f"ser_range must be a two-element range, got {ser_range!r}"
            )
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in ser_range
        ):
            raise ValueError(f"ser_range values must be finite real numbers, got {ser_range!r}")
        ser_override = (float(ser_range[0]), float(ser_range[1]))
        if not np.isfinite(ser_override).all():
            raise ValueError(f"ser_range values must be finite, got {ser_range!r}")
    else:
        ser_override = None
    effective_ser = ser_override if ser_override is not None else sim.ser_range
    if effective_ser is not None:
        lo, hi = float(effective_ser[0]), float(effective_ser[1])
        if lo > hi:
            raise ValueError(f"ser_range lo ({lo}) must be <= hi ({hi}).")

    # ---- far_db / ERL 耦合（erl_* 取值域已由 SimulationConfigs 校验）----
    _far_db_range = getattr(sim, "far_db_range", None)
    if _far_db_range is None or len(_far_db_range) != 2:
        _far_lo, _far_hi = -40.0, -12.0
    else:
        _far_lo, _far_hi = float(_far_db_range[0]), float(_far_db_range[1])
    if _far_lo >= _far_hi:
        raise ValueError(
            f"far_db_range lo ({_far_lo}) must be < hi ({_far_hi})"
        )

    # ---- ref/echo 对齐与随机延迟 ----
    if aec_ref_echo_gcc_farend_pad_policy is not None:
        _farend_pol = str(aec_ref_echo_gcc_farend_pad_policy).strip().lower()
        if _farend_pol != "apply":
            raise ValueError(
                "aec_ref_echo_gcc_farend_pad_policy only supports 'apply'; "
                "use aec_ref_echo_fine_gcc_align: false to disable GCC alignment, "
                f"got {_farend_pol!r}"
            )
    # ---- 窗式 GCC 精对齐（windowed：head+tail 快路径 + 全量扫描升级）----
    gcc_numeric = {
        "aec_ref_echo_gcc_bias_seconds": aec_ref_echo_gcc_bias_seconds,
        "aec_ref_echo_gcc_max_tau_seconds": aec_ref_echo_gcc_max_tau_seconds,
        "aec_ref_echo_gcc_max_pad_seconds": aec_ref_echo_gcc_max_pad_seconds,
        "aec_ref_echo_gcc_min_pad_samples": aec_ref_echo_gcc_min_pad_samples,
        "aec_ref_echo_gcc_peak_threshold": aec_ref_echo_gcc_peak_threshold,
        "aec_ref_echo_gcc_prominence_db": aec_ref_echo_gcc_prominence_db,
        "aec_ref_echo_gcc_tv_iqr_ms": aec_ref_echo_gcc_tv_iqr_ms,
        "aec_ref_echo_gcc_max_windows": aec_ref_echo_gcc_max_windows,
    }
    invalid_gcc = [
        name for name, value in gcc_numeric.items() if not np.isfinite(float(value))
    ]
    if invalid_gcc:
        raise ValueError("AEC GCC parameters must be finite: " + ", ".join(invalid_gcc))
    if float(aec_ref_echo_gcc_max_tau_seconds) < 0.0:
        raise ValueError("aec_ref_echo_gcc_max_tau_seconds must be >= 0")
    if float(aec_ref_echo_gcc_max_pad_seconds) < 0.0:
        raise ValueError("aec_ref_echo_gcc_max_pad_seconds must be >= 0")
    if int(aec_ref_echo_gcc_min_pad_samples) < 0:
        raise ValueError("aec_ref_echo_gcc_min_pad_samples must be >= 0")
    if not (0.0 < float(aec_ref_echo_gcc_peak_threshold) <= 1.0):
        raise ValueError(
            f"aec_ref_echo_gcc_peak_threshold must be in (0.0, 1.0], "
            f"got {aec_ref_echo_gcc_peak_threshold!r}"
        )
    if float(aec_ref_echo_gcc_prominence_db) < 0.0:
        raise ValueError(
            f"aec_ref_echo_gcc_prominence_db must be >= 0, "
            f"got {aec_ref_echo_gcc_prominence_db!r}"
        )
    if float(aec_ref_echo_gcc_tv_iqr_ms) <= 0.0:
        raise ValueError(
            f"aec_ref_echo_gcc_tv_iqr_ms must be > 0, got {aec_ref_echo_gcc_tv_iqr_ms!r}"
        )
    if int(aec_ref_echo_gcc_max_windows) < 1:
        raise ValueError(
            f"aec_ref_echo_gcc_max_windows must be >= 1, got {aec_ref_echo_gcc_max_windows!r}"
        )
    if aec_ref_echo_gcc_downsample is not None and int(aec_ref_echo_gcc_downsample) < 1:
        raise ValueError(
            f"aec_ref_echo_gcc_downsample must be >= 1, got {aec_ref_echo_gcc_downsample!r}"
        )
    if not isinstance(aec_ref_echo_random_delay_mode, str):
        raise ValueError(
            "aec_ref_echo_random_delay_mode must be a string, "
            f"got {aec_ref_echo_random_delay_mode!r}"
        )
    _mode = aec_ref_echo_random_delay_mode.strip().lower()
    if _mode not in ("echo_only", "sync"):
        raise ValueError(
            f"aec_ref_echo_random_delay_mode must be 'echo_only' or 'sync', got {_mode!r}"
        )
    if not isinstance(aec_ref_echo_random_delay_distribution, str):
        raise ValueError(
            "aec_ref_echo_random_delay_distribution must be a string, "
            f"got {aec_ref_echo_random_delay_distribution!r}"
        )
    _dist = aec_ref_echo_random_delay_distribution.strip().lower()
    if _dist not in ("uniform", "beta"):
        raise ValueError(
            f"aec_ref_echo_random_delay_distribution must be 'uniform' or 'beta', got {_dist!r}"
        )
    beta_values = {
        "aec_ref_echo_random_delay_beta_a": aec_ref_echo_random_delay_beta_a,
        "aec_ref_echo_random_delay_beta_b": aec_ref_echo_random_delay_beta_b,
    }
    normalized_beta = {}
    for name, value in beta_values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite positive real number, got {value!r}")
        normalized = float(value)
        if not np.isfinite(normalized) or normalized <= 0.0:
            raise ValueError(f"{name} must be a finite positive real number, got {value!r}")
        normalized_beta[name] = normalized
    _beta_a = normalized_beta["aec_ref_echo_random_delay_beta_a"]
    _beta_b = normalized_beta["aec_ref_echo_random_delay_beta_b"]

    delay_probabilities = {
        "aec_ref_echo_random_delay_uniform_prob": aec_ref_echo_random_delay_uniform_prob,
        "aec_ref_echo_random_delay_prob": aec_ref_echo_random_delay_prob,
    }
    normalized_delay_probabilities = {}
    for name, value in delay_probabilities.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite probability in [0.0, 1.0], got {value!r}")
        normalized = float(value)
        if not np.isfinite(normalized) or not (0.0 <= normalized <= 1.0):
            raise ValueError(f"{name} must be a finite probability in [0.0, 1.0], got {value!r}")
        normalized_delay_probabilities[name] = normalized
    _uniform_prob = normalized_delay_probabilities[
        "aec_ref_echo_random_delay_uniform_prob"
    ]
    _random_delay_prob = normalized_delay_probabilities[
        "aec_ref_echo_random_delay_prob"
    ]
    if aec_ref_echo_delay_range_ms is None:
        _delay_lo = _delay_hi = None
    else:
        if not (
            isinstance(aec_ref_echo_delay_range_ms, (list, tuple))
            and len(aec_ref_echo_delay_range_ms) == 2
        ):
            raise ValueError(
                "aec_ref_echo_delay_range_ms must be a two-element range, "
                f"got {aec_ref_echo_delay_range_ms!r}"
            )
        _delay_lo = float(aec_ref_echo_delay_range_ms[0])
        _delay_hi = float(aec_ref_echo_delay_range_ms[1])
        if not (np.isfinite(_delay_lo) and np.isfinite(_delay_hi)):
            raise ValueError(
                "aec_ref_echo_delay_range_ms values must be finite, "
                f"got {aec_ref_echo_delay_range_ms!r}"
            )
        if _delay_lo < 0.0 or _delay_hi < 0.0:
            raise ValueError(
                "aec_ref_echo_delay_range_ms values must be >= 0, "
                f"got {aec_ref_echo_delay_range_ms!r}"
            )
        if _delay_lo > _delay_hi:
            raise ValueError(
                "aec_ref_echo_delay_range_ms lo must be <= hi, "
                f"got {aec_ref_echo_delay_range_ms!r}"
            )
    if float(aec_ref_echo_random_delay_prob) > 0.0:
        if _delay_lo is None or _delay_hi is None:
            raise ValueError(
                "When aec_ref_echo_random_delay_prob > 0, set aec_ref_echo_delay_range_ms explicitly."
            )
        _min_ms = _delay_lo
        _max_ms = _delay_hi
    else:
        _min_ms = 0.0 if _delay_lo is None else _delay_lo
        _max_ms = 0.0 if _delay_hi is None else _delay_hi
    if _min_ms > _max_ms:
        raise ValueError(
            f"aec_ref_echo_delay_range_ms lo ({_min_ms}) must be <= hi ({_max_ms})."
        )
    _legacy_jump_keys = (
        "aec_ref_echo_delay_jump_prob",
        "aec_ref_echo_delay_jump_max_ms",
        "aec_ref_echo_delay_jump_negative_prob",
        "aec_ref_echo_delay_jump_min_dwell_ms",
        "aec_ref_echo_delay_jump_insert_mode",
    )
    _found_legacy = [
        k for k in _legacy_jump_keys
        if k in sim_in or _ignored.get(k) not in (None, "")
    ]
    if _found_legacy:
        raise ValueError(
            f"Legacy delay-jump key(s) removed: {_found_legacy}. "
            "Use aec_time_delay_jitter with jitter_type='sudden_jump' instead."
        )

    # ---- 完整时间抖动模块（解析即校验；默认关闭）----
    _tdj = _resolve_aec_time_delay_jitter_config(
        aec_time_delay_jitter, aec_ref_echo_delay_range_ms
    )
    if _tdj.enabled and _tdj.prob > 0.0 and aec_ref_echo_delay_range_ms is None:
        raise ValueError(
            "When aec_time_delay_jitter is enabled with prob > 0, set "
            "aec_ref_echo_delay_range_ms explicitly."
        )

    # ---- echo device EQ（C-aec）----
    if aec_echo_device_eq is not None and not isinstance(aec_echo_device_eq, Mapping):
        raise ValueError("aec_echo_device_eq must be a mapping")
    _eq_cfg = dict(aec_echo_device_eq or {})
    _eq_known_keys = {
        "enabled",
        "prob",
        "variant_probs",
        "random_eq_db_range",
        "bass_boost_highpass_cutoff_range",
        "bass_boost_attenuation_min_db",
    }
    _eq_unknown = set(_eq_cfg) - _eq_known_keys
    if _eq_unknown:
        raise ValueError(
            f"aec_echo_device_eq contains unknown keys: {sorted(map(str, _eq_unknown))!r}"
        )
    _eq_enabled = _eq_cfg.get("enabled", False)
    if not isinstance(_eq_enabled, (bool, np.bool_)):
        raise ValueError(
            f"aec_echo_device_eq.enabled must be a boolean, got {_eq_enabled!r}"
        )

    def _eq_finite_real(name: str, value: Any) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(
                f"aec_echo_device_eq.{name} must be a finite real number, got {value!r}"
            )
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(
                f"aec_echo_device_eq.{name} must be finite, got {value!r}"
            )
        return result

    _eq_prob = _eq_finite_real("prob", _eq_cfg.get("prob", 0.3))
    if not 0.0 <= _eq_prob <= 1.0:
        raise ValueError(f"aec_echo_device_eq.prob must be in [0, 1], got {_eq_prob}")
    _eq_variant_probs_raw = _eq_cfg.get(
        "variant_probs", {"random_eq": 0.6, "bass_boost": 0.4}
    )
    if not isinstance(_eq_variant_probs_raw, Mapping) or not _eq_variant_probs_raw:
        raise ValueError("aec_echo_device_eq.variant_probs must be a non-empty mapping")
    _eq_variant_probs = dict(_eq_variant_probs_raw)
    _bad_eq_variants = set(_eq_variant_probs) - ECHO_DEVICE_EQ_VARIANTS
    if _bad_eq_variants:
        raise ValueError(
            f"aec_echo_device_eq.variant_probs has unknown variant(s) {sorted(_bad_eq_variants)}; "
            f"expected subset of {sorted(ECHO_DEVICE_EQ_VARIANTS)}"
        )
    _eq_weights = np.asarray(
        [
            _eq_finite_real(f"variant_probs[{key!r}]", value)
            for key, value in _eq_variant_probs.items()
        ],
        dtype=np.float64,
    )
    if np.any(_eq_weights < 0.0):
        raise ValueError(
            "aec_echo_device_eq.variant_probs weights must be non-negative, "
            f"got {_eq_variant_probs}"
        )
    if float(_eq_weights.sum()) <= 0.0:
        raise ValueError(
            "aec_echo_device_eq.variant_probs weights must sum to > 0, "
            f"got {_eq_variant_probs}"
        )

    def _eq_range(name: str, default: list[float]) -> tuple[float, float]:
        value = _eq_cfg.get(name, default)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                f"aec_echo_device_eq.{name} must be a two-element range, got {value!r}"
            )
        pair = (
            _eq_finite_real(f"{name}[0]", value[0]),
            _eq_finite_real(f"{name}[1]", value[1]),
        )
        if pair[0] > pair[1]:
            raise ValueError(
                f"aec_echo_device_eq.{name} must satisfy lo <= hi, got {pair}"
            )
        return pair

    _req = _eq_range("random_eq_db_range", [-6.0, 6.0])
    _bbc = _eq_range("bass_boost_highpass_cutoff_range", [500.0, 2000.0])
    if _bbc[0] <= 0.0:
        raise ValueError(
            "aec_echo_device_eq.bass_boost_highpass_cutoff_range lower bound "
            f"must be positive, got {_bbc[0]}"
        )
    _eq_att = _eq_finite_real(
        "bass_boost_attenuation_min_db",
        _eq_cfg.get("bass_boost_attenuation_min_db", -25.0),
    )
    if _eq_att > 0.0:
        raise ValueError(
            "aec_echo_device_eq.bass_boost_attenuation_min_db must be <= 0 dB "
            "(it is an attenuation, not a boost), got "
            f"{_eq_att}"
        )

    normalize_echo_clipping_config(aec_echo_clipping)


# =====================================================================
# Dataset
# =====================================================================


@dataclass(frozen=True)
class AECRunContext:
    """一次 AEC 仿真的显式内部传值，不依赖 mutable ``_last_*``。"""

    branch: str
    noise_additive_1d: np.ndarray
    decomposable: bool
    decomposable_reason_codes: tuple[str, ...]
    c0_audit_stems: dict[str, Any] | None
    acoustic_audit: dict[str, Any] | None
    # mono acoustic plan 仅用于 legacy C=1；空间 C-0 使用 spatial_scene_plan。
    acoustic_scene_plan: dict[str, Any] | None
    spatial_scene_plan: dict[str, Any] | None = None


@dataclass(frozen=True)
class AECRunOutput:
    clean: np.ndarray
    noisy: np.ndarray
    sample_rate: int
    aec_disk: dict[str, dict[str, np.ndarray]] | None
    context: AECRunContext
    post_mix: AECPostMixResult


class DynamicMixingDataset(BaseDynamicMixingDataset):
    """AEC on-the-fly mixing: nearend via full augmentation + JSON farend/echo pairs.

    ``real_farend_echo_pair_json`` is **required** (path to a **``.json``** manifest).
    Default: stereo dual-channel WAV — each entry is one file;
    **ch0 = echo**, **ch1 = farend (reference)**. Channel order is fixed by convention.

    ``__getitem__`` returns an **11-tuple**
    ``(nearend, mic, farend, fs, length, error, error_mask,
    is_pure_noise, is_pure_echo, is_pure_echo_noise, has_echo)`` —
    启用帧级 state label（``state_detector_hop_length`` 设置）时为 12-tuple
    （末尾追加 ``state``），供 ``aec_collate_fn`` / ``AECTrainer`` 消费。

    Layer-B scenario branches (7 mutually-exclusive options, sum == 1.0):
      - prob_target_only, prob_noise_only, prob_echo_only, prob_noise_echo_only,
      - prob_target_noise, prob_target_echo, prob_target_noise_echo.

    C-aec (AEC echo) reads disk ref/echo pairs and applies SER scaling when enabled.
    Double-talk branches (prob_target_echo / prob_target_noise_echo) reuse the C-aec echo
    context to avoid a second disk read.

    空间 static/moving 均由 prepare_scene_plan 一次固定 near-end 与按需 echo
    的房间、阵列和位置；echo actor 默认静止。磁盘 echo 的再渲染是录制
    echo 空间增强，不代表从干净 far-end 合成播放链。noise 只走 image adapter。
    """

    _simulation_task = "aec"

    @staticmethod
    def _restore_layer_d_stage_order(metadata: Mapping[str, Any]) -> None:
        """恢复 AEC canonical Layer-D 顺序字段的 tuple 契约。"""
        layer_d = metadata.get("layer_d_facts")
        if isinstance(layer_d, dict):
            layer_d["stage_order"] = ("D-normalize", "D-guard")
        simulation_by_layer = metadata.get("simulation_by_layer")
        if isinstance(simulation_by_layer, dict):
            layer_d = simulation_by_layer.get("D")
            if isinstance(layer_d, dict):
                layer_d["stage_order"] = ("D-normalize", "D-guard")

    def _legacy_stems_from_trace(self, trace: SimulationTrace) -> dict | None:
        """仅在 legacy stems 投影中恢复 AEC Layer-D tuple 字段。"""
        legacy = super()._legacy_stems_from_trace(trace)
        if legacy is not None:
            self._restore_layer_d_stage_order(legacy)
        return legacy

    def _managed_legacy_last_defaults(self) -> dict[str, Any]:
        defaults = super()._managed_legacy_last_defaults()
        defaults.update(
            {
                "_last_noise_additive_1d": None,
                "_last_layer_ef_decomposable": None,
                "_last_decomposable_reason_codes": [],
                "_last_c0_audit_stems": None,
                "_last_mic_extra_noise_policy": None,
            }
        )
        return defaults

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        speech_scp: str,
        noise_scp: str,
        rir_scp: str,
        windnoise_scp: Optional[str] = None,
        speech_length_file: Optional[str] = None,
        simulation_cfg: Optional[dict] = None,
        strict_load: bool = True,
        max_duration: int = 240000,
        max_duration_seconds: Optional[float] = None,
        repeat_to_max_duration: bool = False,
        repeat_tile_silence_gap: bool = False,
        repeat_tile_silence_gap_range: Optional[list] = None,
        filter_by_duration: bool = True,
        filter_min_duration: float = 0.5,
        filter_max_duration: float = 30.0,
        target_sr: Optional[int] = None,
        placeholder_num_samples: int = DEFAULT_PLACEHOLDER_NUM_SAMPLES,
        export_simulation_stems: bool = False,
        simulation_cfg_strict: bool = True,
        sampling_cfg: Any = None,
        *,
        music_scp: Optional[str] = None,
        # AEC-specific args
        real_farend_echo_pair_json: str,
        aec_dynamic_pipeline: Optional[dict] = None,
        ser_range: Optional[list] = None,
        # Ref/echo alignment & delay
        aec_ref_echo_fine_gcc_align: bool = True,
        aec_ref_echo_gcc_bias_seconds: float = 0.001,
        aec_ref_echo_gcc_max_tau_seconds: float = 0.2,
        aec_ref_echo_gcc_max_pad_seconds: float = 0.2,
        aec_ref_echo_gcc_farend_pad_policy: Optional[str] = None,
        aec_ref_echo_gcc_min_pad_samples: int = 0,
        # 窗式 GCC 精对齐（windowed：head+tail 快路径 + 全量扫描升级）
        aec_ref_echo_gcc_peak_threshold: float = 0.05,
        aec_ref_echo_gcc_prominence_db: float = 6.0,
        aec_ref_echo_gcc_tv_iqr_ms: float = 10.0,
        aec_ref_echo_gcc_max_windows: int = 8,
        aec_ref_echo_gcc_downsample: Optional[int] = None,
        aec_ref_echo_random_delay_prob: float = 0.0,
        aec_ref_echo_random_delay_mode: str = "echo_only",
        aec_ref_echo_random_delay_distribution: str = "uniform",
        aec_ref_echo_random_delay_beta_a: float = 1.0,
        aec_ref_echo_random_delay_beta_b: float = 3.0,
        aec_ref_echo_random_delay_uniform_prob: float = _DELAY_TAIL_MIX_PROB,
        aec_ref_echo_delay_range_ms: Optional[list] = None,
        # Optional full time-delay jitter module (sudden_jump / drift / periodic / mixed).
        aec_time_delay_jitter: dict | None = None,
        # Optional C-aec echo device EQ (playback speaker frequency coloring).
        # Runs before aec_echo_clipping -- see tasks/aec/data/echo_device_eq.py.
        # Pass a dict, e.g. {"enabled": true, "prob": 0.2, ...}.
        aec_echo_device_eq: dict | None = None,
        # Optional C-aec echo non-linear distortion (clipping / playback overload).
        # Pass a dict, e.g. {"enabled": true, "prob": 0.25, ...}.
        # See docs/guides/aec/AEC_Data_Simulation.md
        aec_echo_clipping: dict | None = None,
        task: str = "joint",
        state_detector_hop_length: int | None = None,
        state_detector_win_length: int | None = None,
        state_detector_near_threshold_db: float = -40.0,
        state_detector_echo_threshold_db: float = -50.0,
        state_detector_noise_threshold_db: float = -40.0,
        state_detector_hangover_frames: int = 1,
        state_detector_absolute_floor: float = 1e-4,
        state_detector_pre_emphasis: float = 0.97,
        state_detector_label_type: str = "scenario",
        state_detector_smoothing_window_frames: int = 1,
        spatial_rir_provider=None,
        **retired_kwargs: Any,
    ):
        retired_aec_frontend = {
            "aec_stream_align": "aec_frontend.delay_alignment.enabled",
            "linear_aec": "aec_frontend.linear_aec",
        }
        for key, replacement in retired_aec_frontend.items():
            if key in retired_kwargs:
                raise ValueError(
                    f"{key} is no longer supported by DynamicMixingDataset; configure "
                    f"top-level {replacement} instead. The AEC input frontend runs only in Trainer/Inference."
                )
        if retired_kwargs:
            raise TypeError(
                "DynamicMixingDataset got unexpected keyword argument(s): "
                f"{sorted(retired_kwargs)}"
            )

        validate_aec_dataset_config(
            simulation_cfg=simulation_cfg,
            aec_dynamic_pipeline=aec_dynamic_pipeline,
            task=task,
            ser_range=ser_range,
            aec_ref_echo_fine_gcc_align=aec_ref_echo_fine_gcc_align,
            aec_ref_echo_gcc_farend_pad_policy=aec_ref_echo_gcc_farend_pad_policy,
            aec_ref_echo_gcc_bias_seconds=aec_ref_echo_gcc_bias_seconds,
            aec_ref_echo_gcc_max_tau_seconds=aec_ref_echo_gcc_max_tau_seconds,
            aec_ref_echo_gcc_max_pad_seconds=aec_ref_echo_gcc_max_pad_seconds,
            aec_ref_echo_gcc_min_pad_samples=aec_ref_echo_gcc_min_pad_samples,
            aec_ref_echo_gcc_peak_threshold=aec_ref_echo_gcc_peak_threshold,
            aec_ref_echo_gcc_prominence_db=aec_ref_echo_gcc_prominence_db,
            aec_ref_echo_gcc_tv_iqr_ms=aec_ref_echo_gcc_tv_iqr_ms,
            aec_ref_echo_gcc_max_windows=aec_ref_echo_gcc_max_windows,
            aec_ref_echo_gcc_downsample=aec_ref_echo_gcc_downsample,
            aec_ref_echo_random_delay_prob=aec_ref_echo_random_delay_prob,
            aec_ref_echo_random_delay_mode=aec_ref_echo_random_delay_mode,
            aec_ref_echo_random_delay_distribution=aec_ref_echo_random_delay_distribution,
            aec_ref_echo_random_delay_beta_a=aec_ref_echo_random_delay_beta_a,
            aec_ref_echo_random_delay_beta_b=aec_ref_echo_random_delay_beta_b,
            aec_ref_echo_random_delay_uniform_prob=aec_ref_echo_random_delay_uniform_prob,
            aec_ref_echo_delay_range_ms=aec_ref_echo_delay_range_ms,
            aec_time_delay_jitter=aec_time_delay_jitter,
            aec_echo_device_eq=aec_echo_device_eq,
            aec_echo_clipping=aec_echo_clipping,
            state_detector_hop_length=state_detector_hop_length,
            state_detector_win_length=state_detector_win_length,
            state_detector_near_threshold_db=state_detector_near_threshold_db,
            state_detector_echo_threshold_db=state_detector_echo_threshold_db,
            state_detector_noise_threshold_db=state_detector_noise_threshold_db,
            state_detector_hangover_frames=state_detector_hangover_frames,
            state_detector_absolute_floor=state_detector_absolute_floor,
            state_detector_pre_emphasis=state_detector_pre_emphasis,
            state_detector_label_type=state_detector_label_type,
            state_detector_smoothing_window_frames=state_detector_smoothing_window_frames,
            simulation_cfg_strict=simulation_cfg_strict,
        )

        # Preprocess simulation_cfg: remove legacy / AEC-specific keys before
        # the shared base class parses it（合法性已由 validate_aec_dataset_config 校验）。
        _sim_in = dict(simulation_cfg) if simulation_cfg else {}
        _sim_in.pop("aec_scenario_probs", None)
        _sim_in.pop("prob_pure_echo", None)

        super().__init__(
            speech_scp=speech_scp,
            noise_scp=noise_scp,
            rir_scp=rir_scp,
            windnoise_scp=windnoise_scp,
            music_scp=music_scp,
            speech_length_file=speech_length_file,
            simulation_cfg=_sim_in,
            strict_load=strict_load,
            max_duration=max_duration,
            max_duration_seconds=max_duration_seconds,
            repeat_to_max_duration=repeat_to_max_duration,
            repeat_tile_silence_gap=repeat_tile_silence_gap,
            repeat_tile_silence_gap_range=repeat_tile_silence_gap_range,
            filter_by_duration=filter_by_duration,
            filter_min_duration=filter_min_duration,
            filter_max_duration=filter_max_duration,
            target_sr=target_sr,
            placeholder_num_samples=placeholder_num_samples,
            export_simulation_stems=export_simulation_stems,
            simulation_cfg_strict=simulation_cfg_strict,
            sampling_cfg=sampling_cfg,
            spatial_rir_provider=spatial_rir_provider,
        )

        cfg = self.sim_cfg
        self.spatial_rir_provider = (
            self._spatial_rir_provider
            if cfg.spatial_enabled
            else spatial_rir_provider
        )

        # AEC pipeline config
        if not str(real_farend_echo_pair_json).strip():
            raise ValueError("real_farend_echo_pair_json is required (non-empty path to a .json manifest).")
        self._aec_pipe = parse_aec_dynamic_pipeline(aec_dynamic_pipeline)

        # Task mode: "joint" (AEC+SE, default) or "aec" (echo cancellation only)
        # 取值合法性已由 validate_aec_dataset_config 校验。
        self.task = str(task).strip().lower() if task is not None else "joint"
        logger.info("AEC: task=%s", self.task)

        # Frame-level state detector label generation (optional; e.g. DBGNet).
        # 取值合法性已由 validate_aec_dataset_config 校验。
        self.state_detector_hop_length = (
            int(state_detector_hop_length) if state_detector_hop_length is not None else None
        )
        self.state_detector_win_length = (
            int(state_detector_win_length) if state_detector_win_length is not None else None
        )
        self.state_detector_near_threshold_db = float(state_detector_near_threshold_db)
        self.state_detector_echo_threshold_db = float(state_detector_echo_threshold_db)
        self.state_detector_hangover_frames = int(state_detector_hangover_frames)
        self.state_detector_absolute_floor = float(state_detector_absolute_floor)
        self.state_detector_pre_emphasis = float(state_detector_pre_emphasis)
        self.state_detector_noise_threshold_db = float(state_detector_noise_threshold_db)
        self.state_detector_label_type = str(state_detector_label_type).strip().lower()
        self.state_detector_smoothing_window_frames = int(state_detector_smoothing_window_frames)

        pp = Path(real_farend_echo_pair_json).expanduser()
        if not pp.is_file():
            raise ValueError(f"real_farend_echo_pair_json must exist: {pp}")
        manifest_diagnostics: list[str] = []
        raw_pairs = parse_real_farend_echo_pair_file(pp)

        # 相对路径必须相对 manifest parent 解析，再执行路径判定。
        manifest_parent = pp.resolve(strict=False).parent

        def _canonical_manifest_path(path: str) -> str:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = manifest_parent / candidate
            try:
                return str(candidate.resolve(strict=False))
            except OSError:
                return os.path.normpath(os.fspath(candidate))

        canonical_pairs = [
            (_canonical_manifest_path(a), _canonical_manifest_path(b))
            for a, b in raw_pairs
        ]
        _path_usable_cache: dict[str, bool] = {}

        def _usable(path: str) -> bool:
            return _path_usable_cache.setdefault(path, usable_audio_path(path))

        valid_pairs: list[tuple[str, str]] = []
        for row_index, (farend_path, echo_path) in enumerate(canonical_pairs):
            unusable_roles = [
                role
                for role, path in (("farend", farend_path), ("echo", echo_path))
                if not _usable(path)
            ]
            if unusable_roles:
                message = (
                    f"manifest row {row_index} has unusable {','.join(unusable_roles)} "
                    f"resource(s): farend={farend_path!r}, echo={echo_path!r}"
                )
                if self.strict_load:
                    raise ValueError(message)
                manifest_diagnostics.append(message)
                continue
            valid_pairs.append((farend_path, echo_path))
        self._farend_echo_pairs = valid_pairs
        self._farend_echo_manifest_diagnostics = tuple(manifest_diagnostics)
        if manifest_diagnostics:
            logger.warning(
                "AEC manifest %s skipped %d invalid row(s) in lenient mode: %s",
                pp,
                len(manifest_diagnostics),
                "; ".join(manifest_diagnostics),
            )
        if not self._farend_echo_pairs:
            raise ValueError(f"No valid ref/echo rows in {pp}")

        self._aec_ref_echo_stereo_wav = True
        self._aec_stereo_echo_channel = 0
        self._aec_stereo_farend_channel = 1
        logger.info(
            "AEC: canonical stereo manifest rows=%d | farend=ch%d echo=ch%d",
            len(self._farend_echo_pairs),
            self._aec_stereo_farend_channel,
            self._aec_stereo_echo_channel,
        )

        # SER bounds（取值合法性已由 validate_aec_dataset_config 校验）
        _ser_override_lo, _ser_override_hi = (None, None) if ser_range is None else (ser_range[0], ser_range[1])
        _cfg_ser_lo, _cfg_ser_hi = (None, None) if cfg.ser_range is None else (cfg.ser_range[0], cfg.ser_range[1])
        eff_ser_lo = float(_ser_override_lo) if _ser_override_lo is not None else _cfg_ser_lo
        eff_ser_hi = float(_ser_override_hi) if _ser_override_hi is not None else _cfg_ser_hi
        if eff_ser_lo is None or eff_ser_hi is None:
            self._apply_ser = False
            self._ser_low = 0.0
            self._ser_high = 0.0
        else:
            lo, hi = float(eff_ser_lo), float(eff_ser_hi)
            self._apply_ser = True
            self._ser_low = lo
            self._ser_high = hi
            logger.info("AEC: SER scaling enabled, ser_db ~ U[%.2f, %.2f]", lo, hi)

        # ERL-based far-end <-> echo coupling。
        # ERL = far_db - echo_db (standard definition: positive = echo quieter).
        # During training, SER is the primary control; ERL guarantees physical coupling.
        # far_db_range defines the physical bounds for the reference signal (dBFS RMS).
        # 取值合法性已由 validate_aec_dataset_config / SimulationConfigs 校验。

        _far_db_range = getattr(cfg, "far_db_range", None)
        if _far_db_range is None or len(_far_db_range) != 2:
            self._far_db_lo = -40.0
            self._far_db_hi = -12.0
        else:
            self._far_db_lo = float(_far_db_range[0])
            self._far_db_hi = float(_far_db_range[1])

        _erl_range = getattr(cfg, "erl_range", None)
        if _erl_range is None or len(_erl_range) != 2:
            self._erl_min_db = -20.0
            self._erl_max_db = 40.0
        else:
            self._erl_min_db = float(_erl_range[0])
            self._erl_max_db = float(_erl_range[1])
        self._erl_distribution = str(getattr(cfg, "erl_distribution", "beta"))
        self._erl_beta_a = float(getattr(cfg, "erl_beta_a", 1.5))
        self._erl_beta_b = float(getattr(cfg, "erl_beta_b", 1.5))
        self._use_erl_coupling = True
        logger.info(
            "AEC: ERL coupling enabled, erl_db ~ %s[%.1f, %.1f] dB (beta a=%.2f b=%.2f)",
            self._erl_distribution, self._erl_min_db, self._erl_max_db,
            self._erl_beta_a, self._erl_beta_b,
        )

        # Ref/echo alignment settings
        _sr_eff = int(self.target_sr) if self.target_sr is not None else 16000
        self._aec_ref_echo_fine_gcc_align = bool(aec_ref_echo_fine_gcc_align)
        self._aec_ref_echo_gcc_bias_seconds = float(aec_ref_echo_gcc_bias_seconds)
        self._aec_ref_echo_gcc_max_tau_seconds = float(max(0.0, aec_ref_echo_gcc_max_tau_seconds))
        self._aec_ref_echo_gcc_max_pad_seconds = float(max(0.0, aec_ref_echo_gcc_max_pad_seconds))

        self._aec_ref_echo_gcc_min_pad_samples = max(0, int(aec_ref_echo_gcc_min_pad_samples))
        # 窗式 GCC 精对齐参数（取值合法性已由 validate_aec_dataset_config 校验）
        self._aec_ref_echo_gcc_peak_threshold = float(aec_ref_echo_gcc_peak_threshold)
        self._aec_ref_echo_gcc_prominence_db = float(aec_ref_echo_gcc_prominence_db)
        self._aec_ref_echo_gcc_tv_iqr_ms = float(aec_ref_echo_gcc_tv_iqr_ms)
        self._aec_ref_echo_gcc_max_windows = max(1, int(aec_ref_echo_gcc_max_windows))
        # None 表示按 target_sr 自动选择最接近 4kHz 的倍率；显式整数则直接使用该倍率。
        self._aec_ref_echo_gcc_downsample = (
            None
            if aec_ref_echo_gcc_downsample is None
            else max(1, int(aec_ref_echo_gcc_downsample))
        )

        # Random echo delay（取值合法性已由 validate_aec_dataset_config 校验）
        _mode = (aec_ref_echo_random_delay_mode or "echo_only").strip().lower()
        self._aec_ref_echo_random_delay_mode = _mode

        _dist = (aec_ref_echo_random_delay_distribution or "uniform").strip().lower()
        self._aec_ref_echo_random_delay_distribution = _dist
        self._aec_ref_echo_random_delay_beta_a = float(aec_ref_echo_random_delay_beta_a)
        self._aec_ref_echo_random_delay_beta_b = float(aec_ref_echo_random_delay_beta_b)
        self._aec_ref_echo_random_delay_uniform_prob = float(
            aec_ref_echo_random_delay_uniform_prob
        )

        self._aec_ref_echo_random_delay_prob = float(aec_ref_echo_random_delay_prob)
        _delay_lo = None if aec_ref_echo_delay_range_ms is None else aec_ref_echo_delay_range_ms[0]
        _delay_hi = None if aec_ref_echo_delay_range_ms is None else aec_ref_echo_delay_range_ms[1]
        if self._aec_ref_echo_random_delay_prob > 0.0:
            _min_ms = float(max(0.0, _delay_lo))
            _max_ms = float(max(0.0, _delay_hi))
        else:
            _min_ms = 0.0 if _delay_lo is None else float(max(0.0, _delay_lo))
            _max_ms = 0.0 if _delay_hi is None else float(max(0.0, _delay_hi))
        self._aec_ref_echo_min_delay_sec = _min_ms / 1000.0
        self._aec_ref_echo_max_delay_sec = _max_ms / 1000.0

        # 时间抖动上限由基础随机延时范围派生。
        self._time_delay_jitter_cfg = _resolve_aec_time_delay_jitter_config(
            aec_time_delay_jitter, aec_ref_echo_delay_range_ms
        )

        _dist_str = (
            f"U[{_min_ms:.1f}ms,{_max_ms:.1f}ms]"
            if _dist == "uniform"
            # beta 分支实际为可配置的 Beta/Uniform 混合，日志如实反映。
            else f"mix({1.0 - self._aec_ref_echo_random_delay_uniform_prob:.2f}·Beta({self._aec_ref_echo_random_delay_beta_a:.1f},{self._aec_ref_echo_random_delay_beta_b:.1f})"
                 f"+{self._aec_ref_echo_random_delay_uniform_prob:.2f}·U)[{_min_ms:.1f}ms,{_max_ms:.1f}ms]"
        )
        logger.info(
            "AEC: ref/echo post-load — fine_gcc=%s windowed "
            "(max_tau=%.3fs, max_pad=%.3fs, min_pad=%d, "
            "peak_thr=%.3f, prominence=%.1fdB, tv_iqr=%.1fms, max_win=%d, ds=%s), "
            "random_echo_delay mode=%s p=%.2f dist=%s @sr=%d",
            self._aec_ref_echo_fine_gcc_align,
            self._aec_ref_echo_gcc_max_tau_seconds,
            self._aec_ref_echo_gcc_max_pad_seconds,
            self._aec_ref_echo_gcc_min_pad_samples,
            self._aec_ref_echo_gcc_peak_threshold,
            self._aec_ref_echo_gcc_prominence_db,
            self._aec_ref_echo_gcc_tv_iqr_ms,
            self._aec_ref_echo_gcc_max_windows,
            self._aec_ref_echo_gcc_downsample,
            self._aec_ref_echo_random_delay_mode,
            self._aec_ref_echo_random_delay_prob,
            _dist_str,
            _sr_eff,
        )

        # Echo device EQ (C-aec, before SER / aec_echo_clipping): simulates the
        # playback speaker's linear frequency response. Echo-only (never
        # farend). See tasks/aec/data/echo_device_eq.py for the physical rationale
        # and why no Layer F skip/coordination is needed (unlike clipping).
        # 取值合法性已由 validate_aec_dataset_config 校验。
        _eq_cfg = dict(aec_echo_device_eq) if aec_echo_device_eq else {}
        self._aec_echo_device_eq_enabled = bool(_eq_cfg.get("enabled", False))
        self._aec_echo_device_eq_prob = float(_eq_cfg.get("prob", 0.3))
        self._aec_echo_device_eq_variant_probs = dict(
            _eq_cfg.get("variant_probs", {"random_eq": 0.6, "bass_boost": 0.4})
        )
        _req = _eq_cfg.get("random_eq_db_range", [-6.0, 6.0])
        self._aec_echo_device_eq_random_eq_db_range = (float(_req[0]), float(_req[1]))
        _bbc = _eq_cfg.get("bass_boost_highpass_cutoff_range", [500.0, 2000.0])
        self._aec_echo_device_eq_bass_boost_highpass_cutoff_range = (float(_bbc[0]), float(_bbc[1]))
        self._aec_echo_device_eq_bass_boost_attenuation_min_db = float(
            _eq_cfg.get("bass_boost_attenuation_min_db", -25.0)
        )
        if self._aec_echo_device_eq_enabled:
            logger.info(
                "AEC: echo device EQ enabled (prob=%.2f, variants=%s)",
                self._aec_echo_device_eq_prob,
                self._aec_echo_device_eq_variant_probs,
            )

        # 公共门控在顶层，算法参数按 path 配置，禁止混用。
        _clip_cfg = normalize_echo_clipping_config(aec_echo_clipping)
        self._aec_echo_clipping_enabled = _clip_cfg["enabled"]
        self._aec_echo_clipping_prob = _clip_cfg["prob"]
        self._aec_echo_clipping_path_probs = dict(_clip_cfg["path_probs"])
        self._aec_echo_clipping_paths = copy.deepcopy(_clip_cfg["paths"])
        if self._aec_echo_clipping_enabled:
            logger.info(
                "AEC: echo clipping enabled (prob=%.2f, paths=%s)",
                self._aec_echo_clipping_prob,
                self._aec_echo_clipping_path_probs,
            )

        # GCC LRU cache 绑定构造期 manifest/pair 输入快照。
        self._gcc_cache: OrderedDict[tuple, GccAlignEstimate] = OrderedDict()
        self._gcc_cache_max = 4096
        # worker 内 GCC 估计失败计数：保持"不丢样本"语义，但让静默降级可见。
        self._gcc_estimation_failures = 0
        try:
            _manifest_stat = pp.stat()
            self._gcc_manifest_identity = (
                str(pp.resolve()), int(_manifest_stat.st_size), int(_manifest_stat.st_mtime_ns)
            )
        except OSError:
            self._gcc_manifest_identity = (str(pp.resolve()), None, None)
        self._gcc_pair_identities = tuple(
            (self._gcc_path_identity(far_path), self._gcc_path_identity(echo_path))
            for far_path, echo_path in self._farend_echo_pairs
        )

    @staticmethod
    def _gcc_path_identity(path: str) -> tuple[str, int | None, int | None, int | None]:
        """在数据集构造期捕获 GCC pair 文件的路径与 stat 身份。"""
        pp = Path(path).expanduser()
        try:
            st = pp.stat()
            return (str(pp.resolve()), int(st.st_size), int(st.st_mtime_ns), int(st.st_ino))
        except OSError:
            return (str(pp.resolve()), None, None, None)

    def _gcc_cache_key(
        self, fi: int, far_path: str, echo_path: str, min_av: int, target_sr: int
    ) -> tuple:
        """用构造期输入快照、pair index 和完整 GCC 参数构造缓存 key。"""
        effective_ds, work_sr = resolve_gcc_downsample(
            int(target_sr), self._aec_ref_echo_gcc_downsample
        )
        pair_identity = self._gcc_pair_identities[int(fi)]
        return (
            "gcc-v3",
            self._gcc_manifest_identity,
            int(fi),
            pair_identity,
            int(min_av),
            int(target_sr),
            int(effective_ds),
            float(work_sr),
            4,  # GCC 插值倍率
            float(self._aec_ref_echo_gcc_max_tau_seconds),
            float(self._aec_ref_echo_gcc_bias_seconds),
            float(self._aec_ref_echo_gcc_max_pad_seconds),
            int(self._aec_ref_echo_gcc_min_pad_samples),
            str(getattr(self, "_aec_ref_echo_gcc_farend_pad_policy", "apply")),
            True,  # AEC 只搜索非负延迟
            float(self._aec_ref_echo_gcc_peak_threshold),
            float(self._aec_ref_echo_gcc_prominence_db),
            float(self._aec_ref_echo_gcc_tv_iqr_ms),
            int(self._aec_ref_echo_gcc_max_windows),
            bool(self._aec_ref_echo_stereo_wav),
            int(self._aec_stereo_farend_channel),
            int(self._aec_stereo_echo_channel),
        )

    def _validate_simulation_cfg(self, cfg: SimulationConfigs) -> None:
        """Validate AEC scenario probabilities（与 validate_aec_dataset_config 共用实现）。"""
        _check_aec_scenario_probs(cfg)

    def _reference_observation(self, waveform: np.ndarray) -> np.ndarray:
        """把空间 observation 投影到配置 reference channel 的 mono target。"""
        value = np.asarray(waveform, dtype=np.float32)
        if value.ndim == 1:
            return value
        if value.ndim == 2 and value.shape[0] == self._simulation_view.shared.spatial.channels:
            return value[self.sim_cfg.reference_channel]
        raise ValueError(f"observation must be [T] or [C,T], got {value.shape}")

    def _mix_acoustic_rir(
        self,
        speech_2d: np.ndarray,
        sr: int,
        rng: np.random.Generator,
        use_reverb: bool,
        *,
        forced_rir_plan=None,
        forced_rir_asset=None,
    ):
        """空间 opt-in 时跳过 mono RIR，C-0 由 supplied source×mic seam 完成。"""
        if self.sim_cfg.spatial_enabled and forced_rir_plan is None and forced_rir_asset is None:
            return speech_2d.copy(), speech_2d.copy(), speech_2d.copy()
        return super()._mix_acoustic_rir(
            speech_2d, sr, rng, use_reverb,
            forced_rir_plan=forced_rir_plan, forced_rir_asset=forced_rir_asset,
        )

    def _apply_channel_degradation_result(
        self,
        noisy_1d: np.ndarray,
        sr: int,
        actual_len: int,
        rng: np.random.Generator,
        used_sidechain_wind: bool,
    ) -> LayerEResult:
        """Layer E 只作用于 observation，并按空间 policy 重放 channel RNG。"""
        observation = np.asarray(noisy_1d, dtype=np.float32)
        if observation.ndim != 2:
            return super()._apply_channel_degradation_result(
                observation, sr, actual_len, rng, used_sidechain_wind
            )
        spatial = simulation_view_for(self, task="aec").shared.spatial
        if not spatial.spatial_enabled:
            raise ValueError("AEC channel-first observation requires spatial_enabled=True")
        dispatch = dispatch_observation_channels(
            observation,
            policy=spatial.spatial_layer_e_policy,
            rng=rng,
            apply_channel=lambda channel, channel_rng, _index: super(
                DynamicMixingDataset, self
            )._apply_channel_degradation_result(
                channel, sr, actual_len, channel_rng, used_sidechain_wind
            ),
        )
        return aggregate_layer_e_results(dispatch)

    def _is_nonrecoverable_sample_error(self, exc: Exception) -> bool:
        """ERL 场景合同错误不得被 ``strict_load=False`` 转为 placeholder。"""
        return isinstance(exc, ERLCouplingContractError)

    def _clear_extra_epoch_caches(self) -> None:
        """清理 stereo cache；纯函数 GCC 结果跨 epoch 保留并由 key 做失效。"""
        _load_stereo_farend_echo_cached_impl.cache_clear()

    def _clear_extra_worker_caches(self) -> None:
        """Clear AEC-specific caches inside a persistent DataLoader worker."""
        _load_stereo_farend_echo_cached_impl.cache_clear()

    def _record_post_rir_calibration(
        self,
        full_rir: np.ndarray,
        early_reference: np.ndarray,
        gain: float,
    ) -> None:
        """仅在 export 模式缓存 gain 后的 C-0 full/early 审计波形。"""
        if self.export_simulation_stems:
            self._last_c0_audit_stems = {
                "full": np.asarray(full_rir, dtype=np.float32).ravel().copy(),
                "early": np.asarray(early_reference, dtype=np.float32).ravel().copy(),
                "source_volume_gain": float(gain),
            }
        else:
            self._last_c0_audit_stems = None

    def _compute_echo_ref_summaries(
        self,
    ) -> tuple[SourceSummary, SourceSummary] | None:
        """根据 `_farend_echo_pairs` 计算 ref / echo 两个池的时长统计。

        对 unique 路径只调用一次 ``soundfile.info``，避免大清单重复 I/O。
        无法读取的文件跳过（不进入时长统计，但 clip 计数仍包含该 pair）。
        """
        if not self._farend_echo_pairs:
            empty = SourceSummary(
                clips=0,
                total_hours=None,
                min_hours=None,
                max_hours=None,
                mean_hours=None,
                sr_counts={},
            )
            return empty, empty

        unique_paths: set[str] = set()
        for far_path, echo_path in self._farend_echo_pairs:
            unique_paths.add(far_path)
            unique_paths.add(echo_path)

        path_info: dict[str, tuple[int, int]] = {}
        for p in unique_paths:
            try:
                info = soundfile.info(p)
                path_info[p] = (info.frames, info.samplerate)
            except Exception as exc:
                # 摘要只做统计：单文件不可读时跳过该条并保留路径上下文。
                logger.warning(
                    "AEC farend/echo summary skipped unreadable file %s: %s", p, exc
                )

        def _summary(role_path_getter) -> SourceSummary:
            seconds_list: list[float] = []
            sr_counts: Counter = Counter()
            for far_path, echo_path in self._farend_echo_pairs:
                path = role_path_getter(far_path, echo_path)
                info = path_info.get(path)
                if info is None:
                    continue
                frames, sr = info
                seconds_list.append(frames / float(sr))
                sr_counts[sr] += 1
            if not seconds_list:
                return SourceSummary(
                    clips=len(self._farend_echo_pairs),
                    total_hours=None,
                    min_hours=None,
                    max_hours=None,
                    mean_hours=None,
                    sr_counts=dict(sr_counts),
                )
            arr = np.asarray(seconds_list, dtype=np.float64) / 3600.0
            return SourceSummary(
                clips=len(self._farend_echo_pairs),
                total_hours=float(arr.sum()),
                min_hours=float(arr.min()),
                max_hours=float(arr.max()),
                mean_hours=float(arr.mean()),
                sr_counts=dict(sr_counts),
            )

        return _summary(lambda far, echo: far), _summary(lambda far, echo: echo)

    def get_data_summary(
        self,
        split: str,
        n_batches: int,
        dataloader_config: dict[str, Any],
    ) -> DataSummary:
        """AEC 数据分布报告：追加 ref/echo 统计与纯配置仿真摘要。"""
        summary = super().get_data_summary(split, n_batches, dataloader_config)
        sources = dict(summary.sources)
        ref_summary, echo_summary = self._compute_echo_ref_summaries()
        if ref_summary is not None:
            sources["ref"] = ref_summary
            sources["echo"] = echo_summary

        cfg = self.sim_cfg
        rir_normalize = bool(cfg.rir_normalize)
        extra: dict[str, Any] = dict(summary.extra or {})
        extra["farend_echo_pairs"] = len(self._farend_echo_pairs)
        extra["aec_ref_echo_stereo_wav"] = bool(self._aec_ref_echo_stereo_wav)
        extra["aec_task"] = self.task
        extra["simulation_nominal"] = {
            "snr_range": [float(cfg.snr_low), float(cfg.snr_high)],
            "ser_range": (
                [float(self._ser_low), float(self._ser_high)]
                if self._apply_ser
                else None
            ),
            "prob_reverberation": float(cfg.prob_reverberation),
            "early_rir_sec": float(cfg.early_rir_sec),
            "target_type": str(cfg.target_type),
            "rir_normalize": rir_normalize,
            "normalization_mode": "shared_full_peak" if rir_normalize else "raw",
        }
        return DataSummary(
            split=summary.split,
            n_samples=summary.n_samples,
            n_batches=summary.n_batches,
            sources=sources,
            dataloader_config=summary.dataloader_config,
            noise_categories=summary.noise_categories,
            scenario_probs=summary.scenario_probs,
            extra=extra,
        )

    def _aec_rng_b3(self, index: int | None) -> np.random.Generator:
        """Independent RNG for Layer B-3 (farend/echo, GCC, SER).

        Isolated from the main RNG to avoid coupling with the near-end degradation chain.
        """
        tag = 0xAECB300D
        worker_info = torch.utils.data.get_worker_info()
        epoch_token = int(worker_info.seed) if worker_info is not None else int(self._epoch)
        # persistent worker 下必须折入 shared epoch，避免随机流跨 epoch 冻结。
        shared_epoch = getattr(self, "_worker_epoch", None)
        epoch_extra = int(shared_epoch[0]) if shared_epoch is not None else int(getattr(self, "_epoch", 0))
        entropy = [self._base_seed, epoch_token, epoch_extra, tag]
        if index is not None:
            entropy.append(int(index))
        return np.random.default_rng(np.random.SeedSequence(entropy))


    def _aec_rng_mic_extra_noise(self, index: int | None) -> np.random.Generator:
        """Independent RNG for the mic extra-noise augmentation.

        Isolated from the near-end degradation chain so its draws do not
        couple with the main ``_get_rng(index)`` stream.
        """
        tag = 0xAECE7A0D
        worker_info = torch.utils.data.get_worker_info()
        epoch_token = int(worker_info.seed) if worker_info is not None else int(self._epoch)
        # 折入 shared epoch，避免随机流跨 epoch 冻结。
        shared_epoch = getattr(self, "_worker_epoch", None)
        epoch_extra = int(shared_epoch[0]) if shared_epoch is not None else int(getattr(self, "_epoch", 0))
        entropy = [self._base_seed, epoch_token, epoch_extra, tag]
        if index is not None:
            entropy.append(int(index))
        return np.random.default_rng(np.random.SeedSequence(entropy))


    # ------------------------------------------------------------------
    # Channel augmentation parameter generation (Layer E)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Export helpers (for verify_simulation tools)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Layer E: channel degradation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # C-aec: Load and align farend/echo pair from disk
    # ------------------------------------------------------------------
    def _clamp_gcc_tau_samples(self, tau_sec: float, target_sr: int) -> int:
        """tau(秒) → 样点数，套用 bias / max_pad / min_pad 钳制。

        constant 前端补零与时变 offset-crop 共用同一钳制，保证两种路径的
        pad 语义一致。
        """
        return clamp_gcc_delay_samples(
            tau_sec,
            target_sr,
            bias_seconds=self._aec_ref_echo_gcc_bias_seconds,
            max_delay_seconds=self._aec_ref_echo_gcc_max_pad_seconds,
            min_delay_samples=self._aec_ref_echo_gcc_min_pad_samples,
        )

    def _estimate_gcc_tau_windowed(
        self, wf: np.ndarray, we: np.ndarray, target_sr: int
    ) -> GccAlignEstimate:
        """窗式 GCC 延迟估计：委托给共享内核 ``tasks.aec.inference.gcc_phat_core``。

        head+tail @4k 快路径，双窗分歧或全灭时升级全量扫描（可识别时变
        延迟并保留稀疏轨迹）。估计过程只使用 numpy（FFT / resample_poly），
        **不消耗任何 rng**，保证 __getitem__ 的 rng 消耗序列稳定。
        """
        return estimate_gcc_tau_windowed(
            wf,
            we,
            target_sr,
            downsample=self._aec_ref_echo_gcc_downsample,
            max_tau_seconds=self._aec_ref_echo_gcc_max_tau_seconds,
            peak_threshold=self._aec_ref_echo_gcc_peak_threshold,
            prominence_db=self._aec_ref_echo_gcc_prominence_db,
            tv_iqr_ms=self._aec_ref_echo_gcc_tv_iqr_ms,
            max_windows=self._aec_ref_echo_gcc_max_windows,
        )

    def _load_farend_echo_pair_aligned(
        self,
        rng: np.random.Generator,
        target_sr: int,
        target_len: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Load, align, and crop/pad one farend/echo pair.

        五次加载均失败时始终抛错；``strict_load=False`` 由 Dataset 外层统一转换为
        invalid placeholder，禁止把静音加载结果继续标记成正常 echo 分支。
        """
        max_retries = 5
        log = logging.getLogger(__name__)

        for attempt in range(max_retries):
            fi = int(rng.integers(0, len(self._farend_echo_pairs)))
            far_path, echo_path = self._farend_echo_pairs[fi]

            if self._aec_ref_echo_stereo_wav:
                if far_path != echo_path:
                    raise RuntimeError(
                        f"internal: stereo mode expects duplicate path, got {far_path!r} vs {echo_path!r}"
                    )
                # pair paths are canonicalized during construction; call the impl
                # directly so cache hits do not resolve/stat the path again.
                wf, we = _load_stereo_farend_echo_cached_impl(
                    far_path,
                    int(target_sr),
                    self.strict_load,
                    self._aec_stereo_farend_channel,
                    self._aec_stereo_echo_channel,
                )
                wf = np.asarray(wf, dtype=np.float32).ravel()
                we = np.asarray(we, dtype=np.float32).ravel()
            else:
                wf = _load_audio_cached_impl(
                    far_path, target_sr, self.strict_load
                ).ravel().astype(np.float32)
                we = _load_audio_cached_impl(
                    echo_path, target_sr, self.strict_load
                ).ravel().astype(np.float32)

            min_av = min(int(wf.size), int(we.size))
            if min_av <= 0:
                log.warning(
                    "AEC load_farend_echo: attempt %d/%d empty file (fi=%d %s), retrying...",
                    attempt + 1,
                    max_retries,
                    fi,
                    far_path,
                )
                continue

            wf = wf[:min_av].astype(np.float32, copy=False)
            we = we[:min_av].astype(np.float32, copy=False)

            # Guard against silent farend or echo channel (data quality issue)
            wf_max = float(np.max(np.abs(wf)))
            we_max = float(np.max(np.abs(we)))
            if wf_max < 1e-6 or we_max < 1e-6:
                log.warning(
                    "AEC load_farend_echo: attempt %d/%d silent channel "
                    "(farend_max=%.2e echo_max=%.2e fi=%d %s), retrying...",
                    attempt + 1,
                    max_retries,
                    wf_max,
                    we_max,
                    fi,
                    far_path,
                )
                continue

            # Fine align (GCC-PHAT)
            gcc_est: GccAlignEstimate | None = None
            tv_trajectory: tuple | None = None
            applied_tau_samp = 0
            if (
                self._aec_ref_echo_fine_gcc_align
                and min_av > 256
                and np.isfinite(wf).all()
                and np.isfinite(we).all()
            ):
                we_full = np.asarray(we, dtype=np.float32)
                wf_full = np.asarray(wf, dtype=np.float32)

                # --- GCC-PHAT LRU cache (value 为 GccAlignEstimate) ---
                _gcc_key = self._gcc_cache_key(
                    fi, far_path, echo_path, min_av, int(target_sr)
                )
                gcc_est = self._gcc_cache.get(_gcc_key)
                if gcc_est is not None:
                    self._gcc_cache.move_to_end(_gcc_key)
                else:
                    _gcc_cacheable = True
                    try:
                        gcc_est = self._estimate_gcc_tau_windowed(
                            wf_full, we_full, int(target_sr)
                        )
                    except Exception as _gcc_exc:
                        # 估计失败不丢样本：当前调用 tau=0；瞬态错误不得进入缓存。
                        _gcc_cacheable = False
                        self._gcc_estimation_failures += 1
                        # 逐样本 debug 之外，在首次与每 100 次失败时给 worker 级可见提示。
                        if (
                            self._gcc_estimation_failures == 1
                            or self._gcc_estimation_failures % 100 == 0
                        ):
                            logger.warning(
                                "GCC-PHAT fine-align failed %d time(s) in this worker; "
                                "falling back to tau=0 for affected samples: %s",
                                self._gcc_estimation_failures,
                                _gcc_exc,
                            )
                        else:
                            logger.debug("GCC-PHAT fine-align failed (ignored): %s", _gcc_exc)
                        gcc_est = GccAlignEstimate(
                            tau_sec=0.0,
                            trajectory=None,
                            time_varying=False,
                            low_confidence=True,
                            mode="estimation_failed",
                            peak=0.0,
                            prominence=0.0,
                        )
                    if _gcc_cacheable:
                        # 有界 LRU：每次只淘汰最旧项，容量恒定为 4096。
                        if len(self._gcc_cache) >= self._gcc_cache_max:
                            self._gcc_cache.popitem(last=False)
                        self._gcc_cache[_gcc_key] = gcc_est
                # ------------------------------------------------------

                tau_sec = gcc_est.tau_sec
                if gcc_est.time_varying and gcc_est.trajectory:
                    # 时变对：不做整段前端补零，改在 crop 阶段按 crop 起点查轨迹做
                    # offset-crop（见下方裁剪块）。
                    tv_trajectory = gcc_est.trajectory
                if tv_trajectory is None and np.isfinite(tau_sec):
                    tau_samp = self._clamp_gcc_tau_samples(tau_sec, int(target_sr))
                    if tau_samp > 0:
                        # Echo lags farend by tau_samp (tau_sec > 0 means echo
                        # arrives after farend). Pad ref front by tau_samp so that
                        # ref and echo are aligned at the overlap.
                        # After random_echo_delay prepends zeros to echo, ref will
                        # lead mic by d_samp (0-120ms). Speex handles delay=0
                        # fine (ERLE ~24dB), so no extra frame_size margin needed.
                        pad = tau_samp
                        wf = np.concatenate(
                            [np.zeros(pad, dtype=np.float32), wf_full]
                        )
                        we = we_full
                        min_av = min(int(wf.size), int(we.size))
                        wf = wf[:min_av].astype(np.float32, copy=False)
                        we = we[:min_av].astype(np.float32, copy=False)
                        applied_tau_samp = int(tau_samp)

            crop_start: int | None = None
            pad_samples: int | None = None
            if min_av >= target_len:
                st = int(rng.integers(0, min_av - target_len + 1))
                crop_start = int(st)
                we = we[st : st + target_len].astype(np.float32, copy=True)
                if tv_trajectory is not None:
                    # 时变对 offset-crop：echo 取 [st, st+T]，farend 取
                    # [st-tau(st), st-tau(st)+T]（负区间补零）；tau(st) 为轨迹上
                    # 距 st 最近的窗值。注意此处 wf 未做整段补零，时间轴与
                    # 原始文件一致。
                    tau_local = trajectory_tau_at(tv_trajectory, st / float(target_sr))
                    applied_tau_samp = self._clamp_gcc_tau_samples(tau_local, int(target_sr))
                    f_start = st - applied_tau_samp
                    f_end = f_start + target_len
                    if f_start >= 0:
                        wf = wf[f_start:f_end].astype(np.float32, copy=True)
                    else:
                        wf = np.concatenate(
                            [np.zeros(-f_start, dtype=np.float32), wf[:f_end]]
                        ).astype(np.float32, copy=False)
                else:
                    wf = wf[st : st + target_len].astype(np.float32, copy=True)
            else:
                if tv_trajectory is not None:
                    # 短文件（不足 target_len）无法 offset-crop：退化为按文件级
                    # 中值 tau 前端补零 farend，再尾部补零。
                    applied_tau_samp = self._clamp_gcc_tau_samples(tau_sec, int(target_sr))
                    # 前端补零不能超过剩余可用长度，否则后续 np.pad 会收到负数。
                    available_room = max(0, target_len - int(wf.size))
                    applied_tau_samp = min(applied_tau_samp, available_room)
                    if applied_tau_samp > 0:
                        wf = np.concatenate(
                            [np.zeros(applied_tau_samp, dtype=np.float32), wf]
                        ).astype(np.float32, copy=False)
                    wf = np.pad(wf, (0, target_len - int(wf.size))).astype(np.float32)
                    pad = target_len - min_av
                    pad_samples = int(pad)
                    we = np.pad(we, (0, pad)).astype(np.float32)
                else:
                    pad = target_len - min_av
                    pad_samples = int(pad)
                    wf = np.pad(wf, (0, pad)).astype(np.float32)
                    we = np.pad(we, (0, pad)).astype(np.float32)

            load_meta: dict[str, Any] = {
                "pair_manifest_index": int(fi),
                "farend_path": str(far_path),
                "echo_path": str(echo_path),
                "crop_start_samples": crop_start,
                "zero_pad_end_samples": pad_samples,
                # GCC 精对齐元数据（jitter 模块 delay_gt 雏形；仅进 meta dict）
                "gcc_tau_ms": (
                    float(applied_tau_samp) / float(target_sr) * 1000.0
                    if gcc_est is not None
                    else None
                ),
                "gcc_peak": float(gcc_est.peak) if gcc_est is not None else None,
                "gcc_prominence": float(gcc_est.prominence) if gcc_est is not None else None,
                "gcc_time_varying": bool(gcc_est.time_varying) if gcc_est is not None else False,
                "gcc_low_confidence": bool(gcc_est.low_confidence) if gcc_est is not None else False,
                "gcc_mode": gcc_est.mode if gcc_est is not None else "disabled",
            }
            return wf, we, load_meta

        # 重试耗尽必须上抛，避免生成伪正常的静音回声样本。
        log.error(
            "AEC load_farend_echo: all %d attempts failed",
            max_retries,
        )
        raise RecoverableResourceError(
            f"AEC load_farend_echo: all {max_retries} attempts failed; "
            "marking the sample invalid."
        )

    # ------------------------------------------------------------------
    # Random echo delay
    # ------------------------------------------------------------------
    def _apply_random_echo_delay(
        self,
        rng: np.random.Generator,
        farend: np.ndarray,
        echo: np.ndarray,
        target_len: int,
        target_sr: int,
    ):
        """Apply random leading delay to echo (and optionally farend).

        Modes:
          - ``echo_only``: prepend zeros only to echo (ref->echo delay varies).
          - ``sync``: prepend the same pad to both channels (bulk delay).

        返回 ``(wf, we, d_samp)``（d_samp 为实际施加的前导延迟样点数，未施加为 0），
        供调用侧记录 ``random_delay_ms`` 元数据。
        """
        wf = np.asarray(farend, dtype=np.float32).ravel()
        we = np.asarray(echo, dtype=np.float32).ravel()

        if (
            self._aec_ref_echo_random_delay_prob <= 0.0
            or float(rng.random()) >= self._aec_ref_echo_random_delay_prob
            or self._aec_ref_echo_max_delay_sec <= 0.0
            or target_len <= 0
        ):
            return (wf, we, 0)

        min_samp = self._aec_ref_echo_min_delay_sec * float(target_sr)
        max_samp = self._aec_ref_echo_max_delay_sec * float(target_sr)
        if self._aec_ref_echo_random_delay_distribution == "beta":
            # 混合 Beta 与 Uniform 采样延时。
            if float(rng.random()) < self._aec_ref_echo_random_delay_uniform_prob:
                d_samp = int(rng.uniform(min_samp, max_samp))
            else:
                # Sample normalized delay from Beta(a, b), then scale to [min_samp, max_samp].
                x = float(rng.beta(self._aec_ref_echo_random_delay_beta_a, self._aec_ref_echo_random_delay_beta_b))
                d_samp = int(min_samp + x * (max_samp - min_samp))
        else:
            d_samp = int(rng.uniform(min_samp, max_samp))
        d_samp = max(0, min(d_samp, target_len - 1))
        if d_samp <= 0:
            return (wf, we, 0)

        pad = np.zeros(d_samp, dtype=np.float32)
        if self._aec_ref_echo_random_delay_mode == "sync":
            wf_d = np.concatenate([pad, wf])
            we_d = np.concatenate([pad, we])
            if wf_d.size < target_len:
                wf_d = np.pad(wf_d, (0, target_len - wf_d.size))
                we_d = np.pad(we_d, (0, target_len - we_d.size))
            else:
                wf_d = wf_d[:target_len]
                we_d = we_d[:target_len]
            return (wf_d, we_d, d_samp)

        we_d = np.concatenate([pad, we])[:target_len]
        if wf.size >= target_len:
            wf_d = wf[:target_len]
        else:
            wf_d = np.pad(wf, (0, target_len - wf.size))
        return (wf_d, we_d, d_samp)

    # ------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _align_four(
        mic: np.ndarray,
        farend: np.ndarray,
        clean_nearend: np.ndarray,
        echo: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """沿最终时间轴截断四个 AEC waveform，保留 mic 的 channel 轴。"""
        mic_array = np.asarray(mic)
        lens = (
            int(mic_array.shape[-1]),
            int(np.asarray(farend).shape[-1]),
            int(np.asarray(clean_nearend).shape[-1]),
            int(np.asarray(echo).shape[-1]),
        )
        t = min(lens)
        if t <= 0:
            mic_empty = np.zeros_like(mic_array[..., :0], dtype=np.float32)
            z = np.zeros(0, dtype=np.float32)
            return mic_empty, z, z, z
        if max(lens) != t:
            logger.debug(
                "_align_four: length mismatch mic=%d far=%d ne=%d echo=%d -> truncate to %d",
                *lens, t,
            )
        return (
            mic_array[..., :t].astype(np.float32, copy=False),
            np.asarray(farend)[..., :t].astype(np.float32, copy=False),
            np.asarray(clean_nearend)[..., :t].astype(np.float32, copy=False),
            np.asarray(echo)[..., :t].astype(np.float32, copy=False),
        )

    def _crop_aec_disk_context(
        self,
        ctx: dict[str, dict[str, np.ndarray]] | None,
        start: int,
        length: int,
    ) -> dict[str, dict[str, np.ndarray]] | None:
        """Crop 1-D context arrays inside an AEC disk context to a window."""
        if ctx is None:
            return None
        # 防御上游异常值，避免负 start 或零/负 length 导致非预期切片或 padding
        start = max(0, int(start))
        length = max(1, int(length))
        cropped: dict[str, dict[str, np.ndarray]] = {}
        for role, role_ctx in ctx.items():
            if not isinstance(role_ctx, dict):
                cropped[role] = role_ctx
                continue
            new_ctx = dict(role_ctx)
            for key in ("farend", "echo"):
                arr = new_ctx.get(key)
                if arr is None:
                    continue
                arr = np.asarray(arr, dtype=np.float32).ravel()
                if arr.size >= start + length:
                    new_ctx[key] = arr[start : start + length].copy()
                elif arr.size < length:
                    new_ctx[key] = np.pad(
                        arr, (0, length - arr.size), mode="constant"
                    ).astype(np.float32)
                else:
                    # start + length > arr.size >= length：从 start 取到末尾并 padding
                    new_ctx[key] = np.pad(
                        arr[start:], (0, start + length - arr.size), mode="constant"
                    ).astype(np.float32)
            cropped[role] = new_ctx
        return cropped

    def _crop_output_context(
        self,
        ctx: dict[str, dict[str, np.ndarray]] | None,
        start: int,
        length: int,
    ) -> dict[str, dict[str, np.ndarray]] | None:
        """Bridge base-class alignment to AEC farend/echo context cropping."""
        return self._crop_aec_disk_context(ctx, start, length)

    def _repeat_ctx_extra_arrays(
        self,
        ctx: dict[str, dict[str, np.ndarray]] | None,
        ref_len: int,
    ) -> tuple[list[np.ndarray], Any | None]:
        """把 ctx 中的 farend/echo 波形交给基类与 clean/noisy 同步 tile。

        repeat_to_max_duration 触发时，far/echo 必须与 mic 走同一份 tile/gap
        时间表，否则下游只能按最短 ctx 截断（echo 分支达不到目标时长且错位）。
        """
        if ctx is None:
            return [], None
        entries: list[tuple[str, str]] = []
        arrays: list[np.ndarray] = []
        for role, role_ctx in ctx.items():
            if not isinstance(role_ctx, dict):
                continue
            for key in ("farend", "echo"):
                arr = role_ctx.get(key)
                if arr is None:
                    continue
                a = np.asarray(arr, dtype=np.float32).ravel()
                # 复用同一 tile 时间表的前提是与 clean/noisy 等长；防御性对齐
                if a.size != ref_len:
                    a = (
                        np.pad(a, (0, ref_len - a.size), mode="constant").astype(np.float32)
                        if a.size < ref_len
                        else a[:ref_len].copy()
                    )
                entries.append((role, key))
                arrays.append(a)
        if not arrays:
            return [], None

        def _rebuild(tiled: list[np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
            new_ctx = {
                role: (dict(rc) if isinstance(rc, dict) else rc) for role, rc in ctx.items()
            }
            for (role, key), arr in zip(entries, tiled):
                new_ctx[role][key] = arr
            return new_ctx

        return arrays, _rebuild

    # ------------------------------------------------------------------
    # Dict assembly & finalization
    # ------------------------------------------------------------------
    def _pure_branch_bool_tensors(
        self, branch: str | None = None
    ) -> dict[str, torch.Tensor]:
        """返回 branch flags；无显式 branch 时兼容读取 last-completed Adapter。"""
        if branch is None:
            branch = getattr(self, "_last_run_branch", None)
        pn, pe, pen, has_echo = _aec_pure_branch_flags(branch)
        return {
            "is_pure_noise": torch.tensor(pn, dtype=torch.bool),
            "is_pure_echo": torch.tensor(pe, dtype=torch.bool),
            "is_pure_echo_noise": torch.tensor(pen, dtype=torch.bool),
            "has_echo": torch.tensor(has_echo, dtype=torch.bool),
        }

    def _compute_state_label_tensor(
        self,
        nearend: np.ndarray,
        echo: np.ndarray,
        noise: np.ndarray | None,
    ) -> torch.Tensor:
        """Return the frame-level state label according to ``state_detector_label_type``.

        Supports:
          - ``scenario``: 1-D int64 ``(T,)`` with 8 mutually-exclusive classes.
          - ``fst_dt_nst``: 1-D int64 ``(T,)`` with FST=0 / DT=1 / NST=2.
        """
        hop = self.state_detector_hop_length
        assert hop is not None
        if self.state_detector_label_type == "scenario":
            arr = compute_aec_state_scenario_label(
                nearend,
                echo,
                noise,
                hop,
                self.state_detector_win_length,
                near_threshold_db=self.state_detector_near_threshold_db,
                echo_threshold_db=self.state_detector_echo_threshold_db,
                noise_threshold_db=self.state_detector_noise_threshold_db,
                hangover_frames=self.state_detector_hangover_frames,
                absolute_floor_after_norm=self.state_detector_absolute_floor,
                pre_emphasis=self.state_detector_pre_emphasis,
                smoothing_window_frames=self.state_detector_smoothing_window_frames,
            )
            return torch.from_numpy(arr.copy())
        # "fst_dt_nst": original DBGNet 3-class label.
        arr = compute_aec_state_label(
            nearend,
            echo,
            hop,
            self.state_detector_win_length,
            near_threshold_db=self.state_detector_near_threshold_db,
            echo_threshold_db=self.state_detector_echo_threshold_db,
            hangover_frames=self.state_detector_hangover_frames,
            absolute_floor_after_norm=self.state_detector_absolute_floor,
            pre_emphasis=self.state_detector_pre_emphasis,
            smoothing_window_frames=self.state_detector_smoothing_window_frames,
        )
        return torch.from_numpy(arr.copy())

    def _pack_se_only(
        self,
        clean_nearend: np.ndarray,
        noisy_nearend: np.ndarray,
        fs_out: int,
        uid: str,
        pure_flags: dict | None = None,
        noise: np.ndarray | None = None,
    ) -> tuple:
        """Pack a non-AEC branch (prob_target_only / prob_noise_only / prob_target_noise) into a tuple.

        ``noise`` is the additive noise stem used for state label generation.
        It may be ``None`` for branches without additive noise (e.g. prob_target_only).
        """
        mic = np.asarray(noisy_nearend, dtype=np.float32).copy()
        if (
            mic.ndim == 2
            and mic.shape[0] == 1
            and not getattr(getattr(self, "sim_cfg", None), "spatial_enabled", False)
        ):
            mic = mic[0]
        clean = np.asarray(clean_nearend, dtype=np.float32).reshape(-1).copy()
        t = min(mic.shape[-1], clean.size)
        mic = mic[..., :t]
        clean = clean[:t]
        farend = np.zeros(t, dtype=np.float32)
        if pure_flags is None:
            pure_flags = {}
        out = (
            torch.from_numpy(clean.copy()),       # nearend  (clean)
            torch.from_numpy(mic.copy()),         # mic      (noisy)
            torch.from_numpy(farend.copy()),      # farend   (zero placeholder)
            fs_out,
            torch.tensor(t, dtype=torch.int32),
            torch.zeros(t, dtype=torch.float32),  # error (zero-filled for SE-only branches)
            False,                                # error_mask (False = no real LAEC output)
            pure_flags.get("is_pure_noise", False),
            pure_flags.get("is_pure_echo", False),
            pure_flags.get("is_pure_echo_noise", False),
            pure_flags.get("has_echo", False),
        )
        if self.state_detector_hop_length is not None:
            state_label = self._compute_state_label_tensor(
                clean,
                np.zeros(t, dtype=np.float32),  # no echo in SE-only branches
                noise[:t] if noise is not None else None,
            )
            out = out + (state_label,)
        return out

    def _finalize_sample(
        self,
        mic: np.ndarray,
        farend: np.ndarray,
        clean_nearend: np.ndarray,
        echo: np.ndarray,
        fs_out: int,
        index: int,
        uid: str,
        pure_flags: dict | None = None,
        noise: np.ndarray | None = None,
    ) -> tuple:
        """Finalize a sample tuple without changing its acoustic signals.

        ``echo`` is used only for length alignment, not returned.
        ``noise`` is the final additive noise stem used for state label generation.
        动态 Dataset 固定返回零 ``error`` 与 ``False`` mask；runtime LinearAEC
        仅由 Trainer 在 batch waveform 上执行。
        When ``state_detector_hop_length`` is set, a frame-level state label
        (8-class scenario or 3-class FST/DT/NST) is computed and appended to the
        returned tuple for models such as DBGNet.
        """
        mic = np.asarray(mic, dtype=np.float32)
        if (
            mic.ndim == 2
            and mic.shape[0] == 1
            and not getattr(getattr(self, "sim_cfg", None), "spatial_enabled", False)
        ):
            mic = mic[0]
        farend = np.asarray(farend, dtype=np.float32).reshape(-1)
        clean_nearend = np.asarray(clean_nearend, dtype=np.float32).reshape(-1)
        t_end = min(
            mic.shape[-1],
            farend.size,
            clean_nearend.size,
            int(np.asarray(echo, dtype=np.float32).ravel().size),
        )
        mic = mic[..., :t_end]
        farend = farend[:t_end]
        clean_nearend = clean_nearend[:t_end]
        c32 = clean_nearend.astype(np.float32, copy=False)

        err = torch.zeros(t_end, dtype=torch.float32)

        if pure_flags is None:
            pure_flags = {}
        out = (
            torch.from_numpy(c32.copy()),                       # nearend
            torch.from_numpy(mic.copy()),                       # mic
            torch.from_numpy(farend.astype(np.float32, copy=True)),  # farend
            fs_out,
            torch.tensor(t_end, dtype=torch.int32),
            err,                                                # error time-domain zero placeholder
            False,                                              # dynamic Dataset never emits a runtime LAEC output
            pure_flags.get("is_pure_noise", False),
            pure_flags.get("is_pure_echo", False),
            pure_flags.get("is_pure_echo_noise", False),
            pure_flags.get("has_echo", False),
        )
        if self.state_detector_hop_length is not None:
            state_label = self._compute_state_label_tensor(
                clean_nearend[:t_end],
                np.asarray(echo, dtype=np.float32).ravel()[:t_end],
                noise[:t_end] if noise is not None else None,
            )
            out = out + (state_label,)
        return out

    def _maybe_mic_extra_noise_result(
        self,
        mic: np.ndarray,
        fs_out: int,
        index: int,
        clean_reference: np.ndarray,
    ) -> MicExtraNoiseResult:
        """Build explicit task-post-EF mic-extra waveforms and source policy."""
        if not self._aec_pipe["mic_extra_noise_enabled"]:
            return MicExtraNoiseResult(mic, None, None)
        rng = self._aec_rng_mic_extra_noise(index)
        # 复用基类 ordinary source selector：exact → 首个 higher → all。其 lazy
        # per-FS view 避免对 ``noise_flt`` 进行每样本全池 tuple 物化。
        uid = self._select_sample(fs_out, self.noise_source, rng=rng)
        if uid is None:
            return MicExtraNoiseResult(mic, None, None)
        p = self.noise_flt[uid]
        metadata = getattr(self, "_noise_manifest_metadata", {}).get(uid, {})
        policy = NoiseSourcePolicy(
            family=NoiseSourceFamily.ORDINARY,
            category=NoiseCategory(
                category_id=metadata.get("category_id"),
                category_name=(category_balance_name(metadata) if metadata else None),
            ),
            application_role=NoiseApplicationRole.AEC_POST_EF_MIC_EXTRA,
            # `_load_audio_cached` 不公开命中位；如实保留 unknown，而非伪造 hit。
            cache_provenance=NoiseCacheProvenance.SHARED_AUDIO_CACHE_UNKNOWN,
            uid=uid,
            path=p,
        )
        mic_array = np.asarray(mic, dtype=np.float32)
        if mic_array.ndim not in (1, 2) or mic_array.shape[-1] < 1:
            raise ValueError(
                f"AEC mic-extra input must be [T] or [C,T], got {mic_array.shape}"
            )
        L = int(mic_array.shape[-1])
        nw = _load_audio_cached(p, fs_out, self.strict_load).ravel().astype(np.float32)
        if nw.size == 0:
            # 空噪声在平铺时静默产出空数组，只会在下游 mix_noise 广播处报错且无上下文。
            raise RuntimeError(
                "AEC mic-extra noise is empty after load/resample: "
                f"path={p} uid={uid} target_sr={fs_out} strict_load={self.strict_load}"
            )
        if len(nw) >= L:
            st = int(rng.integers(0, len(nw) - L + 1))
            nw = nw[st : st + L].copy()
        else:
            reps = (L + len(nw) - 1) // len(nw)
            nw = np.tile(nw, reps)[:L].astype(np.float32)
        lo, hi = self._aec_pipe["mic_extra_noise_snr_range"]
        snr = float(rng.uniform(lo, hi))
        cref = np.asarray(clean_reference, dtype=np.float32).ravel()
        if cref.size >= L:
            cref = cref[:L]
        else:
            cref = np.pad(cref, (0, L - int(cref.size))).astype(np.float32, copy=False)
        clean_2d = cref.reshape(1, -1)
        reference_mic = (
            mic_array.reshape(1, -1)
            if mic_array.ndim == 1
            else mic_array[self.sim_cfg.reference_channel : self.sim_cfg.reference_channel + 1]
        )
        _, scaled_noise = mix_noise(
            reference_mic, nw.reshape(1, -1), snr=snr, snr_reference=clean_2d
        )
        scaled_noise = scaled_noise.reshape(-1).astype(np.float32, copy=False)
        out = mic_array + scaled_noise
        return MicExtraNoiseResult(
            out.astype(np.float32, copy=False), scaled_noise, policy
        )

    def _maybe_mic_extra_noise(
        self,
        mic: np.ndarray,
        fs_out: int,
        index: int,
        clean_reference: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """历史直调 Adapter，保留二元组与 policy sidecar。"""
        result = self._maybe_mic_extra_noise_result(
            mic, fs_out, index, clean_reference
        )
        self._last_mic_extra_noise_policy = result.policy
        return result.mic, result.extra_noise

    # ------------------------------------------------------------------
    # Full simulation pipeline (_run)
    # ------------------------------------------------------------------
    def _run(
        self,
        speech_uid: str,
        speech_length: int,
        sr: int,
        index: int | None = None,
        force_branch: str | None = None,
        rng: np.random.Generator | None = None,
        *,
        source_trace=None,
    ) -> AECRunOutput:
        """Execute the full six-layer AEC dataset simulation and return explicit context.

        ``AECDeploymentFrontend`` belongs to the AEC model input/deployment path;
        Trainer/Inference runs it before STFT, outside Dataset simulation.

        moving 路径消费一次 provider 请求返回的多 actor bank，near-end 与
        echo 各自渲染 full/direct/early。echo 仍来自录制 pair，经既有
        EQ、SER/ERL、delay/clip 后作空间增强；共同 image gain 保持增强前后
        参考通道功率，不抹掉 delay/clip 的电平变化。trace 中 moving stems
        明确属于 post-mix 与 delivery 时间变换之前，不作为最终训练输出。
        """
        cfg = self.sim_cfg
        max_dur_sr = self._max_duration_for_sr(sr)
        if rng is None:
            rng = self._get_rng(index)
        prob_target_only = float(getattr(cfg, "prob_target_only", 0.0))

        # ========== Layer A: clean-speech pre-processing ==========
        prepared_speech = self._prepare_clean_speech_result(
            speech_uid,
            sr,
            max_dur_sr,
            rng,
            source_trace=(
                source_trace
                if source_trace is not None and source_trace.capture_stems
                else None
            ),
            record_layer_a_metadata=False,
        )
        speech_1d = prepared_speech.speech_1d
        speech_2d = prepared_speech.speech_2d
        actual_len = prepared_speech.actual_len

        # Layer A 的显式结果是唯一事实来源；不从 `_last_*` 回读。
        repeat_export = prepared_speech.repeat_export
        len_before_align = int(repeat_export.get("samples_before_repeat_step", actual_len))
        len_after_align = int(repeat_export.get("samples_after_repeat_step", actual_len))
        a3_alignment_mode = str(repeat_export.get("align_mode", "off"))
        a3_fixed_canvas_applied = a3_alignment_mode == "silence_fixed_canvas"
        a3_tile_applied = a3_alignment_mode == "tile_to_max"

        # ========== Layer B: scenario branch selection ==========
        p_clean = float(prob_target_only)
        p_noise = float(getattr(cfg, "prob_noise_only", 0.0))
        p_echo = float(getattr(cfg, "prob_echo_only", 0.0))
        p_noise_echo = float(getattr(cfg, "prob_noise_echo_only", 0.0))
        p_nn = float(getattr(cfg, "prob_target_noise", 0.0))
        p_ne = float(getattr(cfg, "prob_target_echo", 0.0))
        p_nne = float(getattr(cfg, "prob_target_noise_echo", 0.0))

        lc_branch, branch_sample_u = self._select_branch(
            [
                ("prob_target_only", p_clean),
                ("prob_noise_only", p_noise),
                ("prob_echo_only", p_echo),
                ("prob_noise_echo_only", p_noise_echo),
                ("prob_target_noise", p_nn),
                ("prob_target_echo", p_ne),
                ("prob_target_noise_echo", p_nne),
            ],
            rng,
            force_branch=force_branch,
            normalize_force=_normalize_force_branch,
        )

        need_c12, need_c3 = _layer_c_subflags(lc_branch)

        # ========== Layer C: acoustic mixing ==========
        ac = self._mix_acoustic(speech_2d, sr, actual_len, max_dur_sr, rng,
                                need_c12 and not cfg.spatial_enabled)

        # Align lengths after mixing
        (
            noisy_speech,
            clean_target,
            noisy_after_rir,
            additive_noise_2d,
        ) = self._align_mix_acoustic_outputs(
            ac.noisy_speech, ac.clean_target, ac.noisy_after_rir,
            ac.additive_noise_2d, actual_len, sr
        )

        noise_additive_1d = additive_noise_2d.flatten().astype(np.float32)
        spatial_mic: np.ndarray | None = None
        spatial_echo: np.ndarray | None = None
        spatial_noise: np.ndarray | None = None
        spatial_echo_rirs: np.ndarray | None = None
        background_spec = None
        moving_tracks = None
        spatial_scene_plan: dict[str, Any] | None = None
        if cfg.spatial_enabled:
            source_gain = float(
                (ac.source_volume_calibration or {}).get("gain", 1.0)
            )
            source_for_spatial = (
                np.asarray(speech_2d, dtype=np.float32) * np.float32(source_gain)
            )
            spatial_rng = self._aec_rng_b3(index)
            moving = bool(
                cfg.prob_moving_rir > 0.0
                and spatial_rng.random() < cfg.prob_moving_rir
            )
            provider = self._spatial_rir_provider
            source_uids = (speech_uid, "__echo__") if need_c3 else (speech_uid,)
            scene_plan = scene_sampler.prepare_scene_plan(
                provider=provider, task="aec", sample_index=index,
                sample_rate=int(sr), num_samples=int(actual_len),
                source_uids=source_uids,
                spatial_view=self._simulation_view.shared.spatial,
                simulation_cfg=cfg, rng=spatial_rng, moving=moving,
                trajectory_points=4, max_displacement_m=0.25,
                roles=("target", "loudspeaker") if need_c3 else ("target",),
                stationary_actor_uids=("__echo__",) if need_c3 else (),
            )
            if tuple(actor.actor_uid for actor in scene_plan.actors) != source_uids:
                raise ValueError("AEC scene actor order does not match requested sources")
            if scene_plan.array.reference_channel != cfg.reference_channel:
                raise ValueError("AEC scene reference channel does not match configuration")
            spatial_scene_plan = scene_plan.to_dict()
            if source_trace is not None:
                # 空间 C-0 的 provenance 与 stems capture 无关；capture 关闭时也必须发布真实 scene plan。
                source_trace.replace_metadata("spatial_scene_plan", spatial_scene_plan)
            if scene_plan.timebase is not None:
                moving_request = MovingSceneRIRRequest(
                    sample_rate=int(sr), num_samples=int(actual_len),
                    actor_trajectories={actor.actor_uid: actor.trajectory for actor in scene_plan.actors},
                    room=scene_plan.to_dict()["room"],
                    array={"positions_m": scene_plan.array.positions_m,
                           "reference_channel": scene_plan.array.reference_channel},
                    timebase=scene_plan.timebase, source_uids=source_uids,
                    scene_uid=scene_plan.identity["scene_uid"],
                    actor_roles=tuple(actor.role for actor in scene_plan.actors),
                )
                moving_tracks = provider.sample_moving_scene_tracks(moving_request, normalize=True)
                spatial = _render_aec_moving_actor(
                    moving_tracks, moving_request, source_for_spatial, speech_uid,
                    early_boundary=max(0, int(round(cfg.early_rir_sec * sr))),
                    reference_channel=cfg.reference_channel,
                )
                if source_trace is not None and source_trace.capture_stems:
                    source_trace.replace_metadata("moving_scene", {
                        "scene_uid": scene_plan.identity["scene_uid"],
                        "group_uid": moving_tracks.plan["group_uid"],
                        "source_order": list(moving_request.source_uids),
                        "trajectory_timebase": moving_request.timebase.to_dict(),
                        "trajectory": np.asarray(moving_tracks.plan["trajectory"]).tolist(),
                        "mic_loc": np.asarray(moving_tracks.plan["mic_loc"]).tolist(),
                        "room_dim": np.asarray(moving_tracks.plan["room_dim"]).tolist(),
                        "track_identity": moving_tracks.facts["track_set"],
                        "early_boundary": max(0, int(round(cfg.early_rir_sec * sr))),
                        "reference_channel": cfg.reference_channel,
                        "level_reference": "reference_channel",
                        "stem_stage": "pre_mix_pre_delivery_time_transform",
                    })
                    for kind in ("full", "direct", "early"):
                        source_trace.replace_stem(
                            f"moving_nearend_{kind}_pre_mix", getattr(spatial, f"{kind}_tracks")[0]
                        )
            else:
                tracks = provider.sample_scene_tracks(
                    scene_plan, track_kinds=("full", "early"),
                    normalize=cfg.rir_backend == "online",
                )
                if not isinstance(tracks, SpatialRIRTrackSet):
                    raise TypeError("AEC static provider must return SpatialRIRTrackSet")
                if tracks.early is None and cfg.rir_backend == "offline":
                    raise ValueError("AEC offline static provider must return requested early track")
                spatial_rirs_all = tracks.full.canonical_rirs
                if spatial_rirs_all.shape[:2] != (len(source_uids), scene_plan.array.channels):
                    raise ValueError("AEC static RIR source/channel axes do not match scene plan")
                if tuple(tracks.provenance.get("actor_order", ())) != source_uids:
                    raise ValueError("AEC static RIR actor order does not match scene plan")
                if tracks.provenance.get("scene_uid") != scene_plan.identity["scene_uid"]:
                    raise ValueError("AEC static RIR scene identity does not match scene plan")
                # 直接比较规范化后的 dict：json 字符串相等会把 1 与 1.0 误判为不一致。
                if _thaw_json(tracks.provenance.get("room")) != _thaw_json(scene_plan.room):
                    raise ValueError("AEC static RIR room does not match scene plan")
                if not np.allclose(tracks.geometry["positions_m"], scene_plan.array.positions_m,
                                   rtol=0, atol=1e-6):
                    raise ValueError("AEC static RIR geometry does not match scene plan")
                if need_c3:
                    spatial_echo_rirs = spatial_rirs_all[1:2].copy()
                spatial = render_spatial_scene(
                    source_for_spatial, spatial_rirs_all[:1],
                    early_boundary=0, reference_channel=cfg.reference_channel,
                    num_samples=actual_len,
                )
                # 离线 early 可以是独立资产；不得从 full 重新截断替代。
                if tracks.early is not None:
                    early_rirs = tracks.early.canonical_rirs[:1]
                else:
                    # online 只生成 full/direct；early 按配置窗口显式派生。
                    early_rirs = spatial_rirs_all[:1].copy()
                    early_rirs[..., max(0, int(round(cfg.early_rir_sec * sr))):] = 0
                early_image = render_spatial_scene(
                    source_for_spatial, early_rirs, early_boundary=early_rirs.shape[-1],
                    reference_channel=cfg.reference_channel, num_samples=actual_len,
                )
                spatial = replace(spatial, early_tracks=early_image.full_tracks)
            spatial_mic = spatial.mixture.copy()
            target_observation = (
                source_for_spatial[0].copy()
                if cfg.target_type == "dry"
                else spatial.early_tracks[0, cfg.reference_channel].copy()
            )
            noisy_after_rir = spatial.reference_tracks[0].copy()
            from core.data.simulation.background_resolver import resolve_and_mix_background
            background, background_spec = resolve_and_mix_background(
                dataset=self,
                speech_image=spatial_mic,
                reference_image=spatial_mic,
                sample_rate=sr,
                actual_length=actual_len,
                max_duration_samples=max_dur_sr,
                rng=spatial_rng,
                need_noise=need_c12,
                snr=(
                    float(spatial_rng.uniform(cfg.snr_low, cfg.snr_high))
                    if need_c12 else None
                ),
                scene_plan=scene_plan,
                provider=provider,
                image_geometry=scene_plan.array.positions_m,
                shared_scene=True,
            )
            spatial_noise = background.additive
            noise_additive_1d = spatial_noise[cfg.reference_channel].copy()
            spatial_mic = background.observation
            noisy_speech = spatial_mic
            clean_target = target_observation
            # 更新旧 task audit 视图，保留 C0 的额外属性，不再次读取或标定 noise。
            acoustic_values = ac._asdict()
            acoustic_values.update(
                noisy_speech=spatial_mic, clean_target=clean_target,
                noisy_after_rir=noisy_after_rir, additive_noise_2d=spatial_noise,
                additive_snr_db_list=background.snr_values,
                used_sidechain_wind=background.used_sidechain_wind,
                mix_two=background.mix_two, want_sidechain=background.want_sidechain,
                wn_snr=background.wind_snr,
                snr_primary_drawn_db=background.primary_snr,
                wind_noise_snr_candidate_db=background.wind_candidate_snr,
            )
            updated_ac = type(ac)(**acoustic_values)
            updated_ac.__dict__.update(ac.__dict__)
            updated_ac.acoustic_audit = background.audit
            ac = updated_ac
            if source_trace is not None:
                source_trace.replace_metadata("noise_image", background.noise_image_provenance)

        # 只在 export 模式比较全波形，区分“抽到 sidechain/reverb”与“实际破坏
        # joint target 分解”。低于 compressor threshold 的 wind 或 delta RIR
        # 不应仅因配置门控就被误报为非线性/target mismatch。
        _joint_nearend_branch = self.task == "joint" and lc_branch in (
            "prob_target_noise",
            "prob_target_echo",
            "prob_target_noise_echo",
        )
        _joint_target_rir_mismatch = False
        _sidechain_wind_mismatch_reason: str | None = None
        if self.export_simulation_stems and _joint_nearend_branch:
            _rir_mic = np.asarray(noisy_after_rir, dtype=np.float32).ravel()
            _target = np.asarray(clean_target, dtype=np.float32).ravel()
            _length = min(_rir_mic.size, _target.size)
            if (ac.use_reverb or moving_tracks is not None) and _length > 0:
                _joint_target_rir_mismatch = not np.allclose(
                    _rir_mic[:_length], _target[:_length], rtol=1e-6, atol=1e-7
                )
            if ac.used_sidechain_wind:
                _mixed = np.asarray(noisy_speech, dtype=np.float32).ravel()
                _length = min(_mixed.size, _rir_mic.size, noise_additive_1d.size)
                if _length > 0:
                    _wind_recovered_nearend = (
                        _mixed[:_length] - noise_additive_1d[:_length]
                    )
                    _sidechain_wind_mismatch_reason = _waveform_mismatch_reason(
                        _wind_recovered_nearend,
                        _rir_mic[:_length],
                        nonlinear_reason="sidechain_wind_nonlinear",
                        gain_reason="sidechain_wind_gain_mismatch",
                    )

        # 所有全波形统计仅在 export 模式执行。calibration 始终使用当前
        # physical noise stem；尤其 sidechain wind 不从 mic-clean 反推。
        _aec_acoustic_audit: dict[str, Any] | None = None
        _source_volume_calibration: dict[str, Any] | None = None
        _aggregate_snr: float | None = None
        if self.export_simulation_stems:
            _source_volume_calibration = _source_volume_calibration_export(
                ac.source_volume_calibration
            )
            _c0_stems = ac.c0_audit_stems or {}
            _c0_early = _c0_stems.get("early")
            _aggregate_snr = (
                _active_power_db_ratio(_c0_early, noise_additive_1d)
                if need_c12 and _c0_early is not None
                else None
            )
            _aec_acoustic_audit = {
                "candidate": {
                    "primary_snr_db": ac.snr_primary_drawn_db,
                    "wind_snr_db": ac.wind_noise_snr_candidate_db,
                    "wind_sidechain_requested": bool(ac.want_sidechain),
                },
                "nominal": {
                    "selected_primary_snr_db": float(ac.wn_snr) if need_c12 else None,
                    "additive_snr_db_each_mix": [
                        float(value) for value in ac.additive_snr_db_list
                    ],
                    "wind_sidechain_applied": bool(ac.used_sidechain_wind),
                },
                "calibration": {
                    "layer_c_aggregate_realized_snr_db": _aggregate_snr,
                    "layer_c_aggregate_reason_code": (
                        None if _aggregate_snr is not None
                        else "noise_not_required" if not need_c12
                        else "aggregate_power_unavailable"
                    ),
                    "source_level_audits": ac.acoustic_audit,
                    "snr_level_audit": LevelAudit(
                        name="aec.snr",
                        candidate_drawn_db=ac.snr_primary_drawn_db,
                        selected_nominal_db=(float(ac.wn_snr) if need_c12 else None),
                        calibration_realized_db=_aggregate_snr,
                        final_realized_db=None,
                        numerator="c0_early_active",
                        denominator="aggregate_additive_noise_active",
                        policy="active_mean_power",
                        calibration_measure_point="post_noise_mix_pre_c3_echo",
                        final_measure_point="see_simulation_by_layer.realized",
                        applicable=False,
                        reason="final_available_in_simulation_by_layer_realized",
                    ).to_dict(),
                    "colored_noise": (
                        (ac.acoustic_audit or {}).get("colored_noise")
                        if ac.acoustic_audit is not None
                        else None
                    ),
                },
                "c0_full_early": _peak_rms_ratio_audit(
                    _c0_stems.get("full"), _c0_early
                ),
            }

        # C-aec: manifest ref/echo (only when need_c3)
        ser_db_this_sample: float | None = None
        c3_load_meta: dict[str, Any] | None = None
        echo_eq_variant: str | None = None
        echo_clip_path: str | None = None
        echo_clip_variant: str | None = None
        farend_clip_variant: str | None = None
        echo_clip_threshold_frac: float | None = None
        echo_clip_threshold_abs: float | None = None
        echo_clip_rail: float | None = None
        echo_clip_rail_dbfs: float | None = None
        echo_clip_shared_variant: str | None = None
        echo_clip_boost_db: float | None = None
        echo_clip_effective_ser_db: float | None = None
        echo_clip_peak_before: float | None = None
        echo_clip_peak_after: float | None = None
        echo_clip_rms_before: float | None = None
        echo_clip_rms_after: float | None = None
        echo_clip_echo_applied = False
        echo_clip_farend_applied = False
        echo_clip_any_applied = False
        echo_clip_echo_rail_facts: dict[str, Any] | None = None
        echo_clip_farend_rail_facts: dict[str, Any] | None = None
        echo_clip_echo_rail_hit: bool | None = None
        echo_clip_farend_rail_hit: bool | None = None
        echo_clip_rail_hit: bool | None = None
        echo_clip_clipped_sample_count: int | None = None
        echo_clip_rail_sample_count: int | None = None
        echo_clip_clipped_sample_ratio: float | None = None
        _effective_ser: float | None = None
        _erl_db_record: float | None = None
        _far_db_record: float | None = None
        _erl_range_audit: dict[str, object] | None = None
        _ser_calibration_db: float | None = None
        _erl_calibration_db: float | None = None

        if need_c3:
            rng_b3 = self._aec_rng_b3(index)
            farend_1d, echo_raw_1d, c3_load_meta = self._load_farend_echo_pair_aligned(
                rng_b3, sr, actual_len
            )
            farend_1d = np.asarray(farend_1d, dtype=np.float32).ravel()
            echo_raw_1d = np.asarray(echo_raw_1d, dtype=np.float32).ravel()
            if echo_raw_1d.size != actual_len:
                echo_raw_1d = (
                    np.pad(echo_raw_1d, (0, actual_len - echo_raw_1d.size)).astype(np.float32)
                    if echo_raw_1d.size < actual_len
                    else echo_raw_1d[:actual_len].copy()
                )
            if farend_1d.size != actual_len:
                farend_1d = (
                    np.pad(farend_1d, (0, actual_len - farend_1d.size)).astype(np.float32)
                    if farend_1d.size < actual_len
                    else farend_1d[:actual_len].copy()
                )
            if self._aec_echo_device_eq_enabled:
                echo_eq_variant = select_echo_device_eq_variant(
                    rng_b3, self._aec_echo_device_eq_prob, self._aec_echo_device_eq_variant_probs
                )
                if echo_eq_variant is not None:
                    echo_raw_1d = apply_echo_device_eq(
                        echo_raw_1d,
                        sr,
                        echo_eq_variant,
                        self._aec_echo_device_eq_random_eq_db_range,
                        self._aec_echo_device_eq_bass_boost_highpass_cutoff_range,
                        self._aec_echo_device_eq_bass_boost_attenuation_min_db,
                        rng_b3,
                    )
            if self._aec_echo_clipping_enabled:
                echo_clip_path = select_echo_clip_path(
                    rng_b3, self._aec_echo_clipping_prob, self._aec_echo_clipping_path_probs
                )

            # ── ERL-based coupling: SER (primary) → echo_db → ERL → far_db ──
            # ERL = far_db - echo_db (standard: positive = echo quieter).
            # Physical constraint: far_db = echo_db + ERL ∈ far_db_range.
            # pre_ser_overload 路径契约要求跳过 SER 随机抽取：echo 电平由
            # 后面的 boost 增益自然决定，farend 保持加载电平（不经过 ERL 耦合）。
            if getattr(self, "_apply_ser", False) and echo_clip_path != "pre_ser_overload":
                # Compute clean RMS dBFS for SER reference
                clean_power = float(np.mean(np.asarray(
                    clean_target, dtype=np.float32
                ).ravel() ** 2))
                clean_db = max(10.0 * float(np.log10(max(clean_power, 1e-12))), -60.0)

                # 在 echo_db 的可行区间内采样 SER，避免钳位边界堆积。
                ser_lo_eff = max(self._ser_low, clean_db)
                ser_hi_eff = min(self._ser_high, clean_db + 60.0)
                if ser_lo_eff < ser_hi_eff:
                    ser_db_this_sample = float(rng_uniform(rng_b3, ser_lo_eff, ser_hi_eff))
                else:
                    ser_db_this_sample = float(rng_uniform(rng_b3, self._ser_low, self._ser_high))

                # Echo level from SER; clamp to feasible digital range.
                # 窗口抽取下该钳位为永假守卫，仅在上述回退路径生效。
                echo_db_raw = clean_db - ser_db_this_sample
                echo_db = float(max(min(echo_db_raw, 0.0), -60.0))
                _effective_ser = clean_db - echo_db  # may differ from ser_db if clamped

                # ERL 配置与当前物理可行区间必须有交集。
                erl_ranges = resolve_erl_coupling_range(
                    configured_range=(self._erl_min_db, self._erl_max_db),
                    far_db_range=(self._far_db_lo, self._far_db_hi),
                    echo_db=echo_db,
                )
                erl_lo, erl_hi = erl_ranges.effective
                if self._erl_distribution == "beta":
                    erl_u = float(rng_b3.beta(self._erl_beta_a, self._erl_beta_b))
                    erl_db = erl_lo + erl_u * (erl_hi - erl_lo)
                else:
                    erl_db = float(rng_uniform(rng_b3, erl_lo, erl_hi))
                _erl_range_audit = erl_ranges.to_audit(erl_db)

                # Far-end level from physical constraint
                far_db = echo_db + erl_db  # guaranteed ∈ [far_db_lo, far_db_hi]

                # Scale echo and far-end
                echo_after_ser_1d = scale_signal_to_dBfs_rms(echo_raw_1d, echo_db)
                farend_1d = scale_signal_to_dBfs_rms(farend_1d, far_db)

                # Derived values for audit；只在 export 路径执行全波形重测。
                _erl_db_record = float(erl_db)
                _far_db_record = float(far_db)
                if self.export_simulation_stems:
                    _ser_calibration_db = _finite_power_db_ratio(
                        np.asarray(clean_target, dtype=np.float32).ravel(),
                        echo_after_ser_1d,
                    )
                    _erl_calibration_db = _finite_power_db_ratio(
                        farend_1d, echo_after_ser_1d
                    )
            else:
                # 未抽 SER：_apply_ser 关闭，或 pre_ser_overload 路径跳过抽取。
                ser_db_this_sample = None
                echo_after_ser_1d = echo_raw_1d.copy()
                _effective_ser = None
                _erl_db_record = None
                _far_db_record = None

            # ── Echo clipping paths (on top of SER-scaled echo) ──
            if echo_clip_path == "pre_ser_overload":
                # 此路径未抽 SER；boost 后使用 path-local 绝对 dBFS rail。
                pre_cfg = self._aec_echo_clipping_paths["pre_ser_overload"]
                echo_before_clip = echo_after_ser_1d.copy()
                echo_clip_peak_before = float(np.max(np.abs(echo_before_clip)))
                echo_clip_rms_before = float(np.sqrt(np.mean(echo_before_clip ** 2)))
                overload_facts: dict[str, Any] = {}
                echo_after_ser_1d, echo_clip_boost_db, echo_clip_variant = apply_playback_overload(
                    echo_before_clip,
                    pre_cfg["overload_boost_db_range"],
                    pre_cfg["rail_dbfs_range"],
                    pre_cfg["variant_probs"],
                    rng_b3,
                    facts_out=overload_facts,
                )
                echo_clip_rail_dbfs = overload_facts.get("rail_dbfs")
                echo_clip_rail = overload_facts.get("rail")
                echo_clip_echo_rail_facts = overload_facts
                echo_clip_echo_applied = not np.array_equal(
                    echo_after_ser_1d, echo_before_clip
                )
                echo_clip_peak_after = float(np.max(np.abs(echo_after_ser_1d)))
                echo_clip_rms_after = float(np.sqrt(np.mean(echo_after_ser_1d ** 2)))
                if echo_clip_boost_db is not None:
                    echo_clip_effective_ser_db = ser_db_from_power_ratio(
                        clean_target.flatten().astype(np.float32), echo_after_ser_1d
                    )

            # Apply random echo delay BEFORE mixing into noisy_1d so that all
            # echo-involved branches (prob_echo_only / prob_noise_echo_only / prob_target_echo /
            # prob_target_noise_echo) process the delayed echo through Layer D/E/F.
            (
                farend_1d,
                echo_after_ser_1d,
                _echo_delay_samp,
            ) = self._apply_random_echo_delay(
                rng_b3, farend_1d, echo_after_ser_1d, actual_len, sr,
            )
            if c3_load_meta is not None:
                # 记录实际施加的随机前导延迟（ms，base d0 语义）。
                c3_load_meta["random_delay_ms"] = (
                    float(_echo_delay_samp) / float(sr) * 1000.0
                )

            # apply_time_delay_jitter 仅在内部做一次概率门控；不要此处预门控。
            if self._time_delay_jitter_cfg.enabled:
                # mic=None 避免分支耦合。
                _jitter_base_delay_samp = (
                    0.0
                    if self._aec_ref_echo_random_delay_mode == "sync"
                    else float(_echo_delay_samp)
                )
                _tdj = apply_time_delay_jitter(
                    None,
                    echo_after_ser_1d[:actual_len].astype(np.float32),
                    _jitter_base_delay_samp,
                    sample_rate=sr,
                    rng=rng_b3,
                    config=self._time_delay_jitter_cfg,
                    max_total_delay_ms=self._aec_ref_echo_max_delay_sec * 1000.0,
                )
                echo_after_ser_1d = _tdj.echo_jittered
                if c3_load_meta is not None:
                    c3_load_meta["time_delay_jitter"] = _tdj.meta

            if echo_clip_path == "post_ser_distortion":
                post_cfg = self._aec_echo_clipping_paths["post_ser_distortion"]
                echo_before_clip = echo_after_ser_1d.copy()
                if echo_before_clip.size and float(np.max(np.abs(echo_before_clip))) >= 1e-8:
                    echo_clip_threshold_frac, sampled_variant = sample_peak_relative_clip_params(
                        post_cfg["peak_relative_threshold_range"],
                        post_cfg["variant_probs"],
                        rng_b3,
                    )
                else:
                    sampled_variant = None
                (
                    echo_after_ser_1d,
                    echo_clip_variant,
                    echo_clip_peak_before,
                    echo_clip_rms_before,
                    echo_clip_peak_after,
                    echo_clip_rms_after,
                ) = _apply_peak_relative_echo(
                    echo_before_clip,
                    post_cfg["peak_relative_threshold_range"],
                    post_cfg["variant_probs"],
                    rng_b3,
                    threshold_frac=echo_clip_threshold_frac,
                    variant=sampled_variant,
                )
                echo_clip_threshold_abs = (
                    echo_clip_peak_before * echo_clip_threshold_frac
                    if echo_clip_threshold_frac is not None
                    else None
                )
                echo_clip_echo_applied = not np.array_equal(
                    echo_after_ser_1d, echo_before_clip
                )
            elif echo_clip_path == "synced_ref_clip":
                synced_cfg = self._aec_echo_clipping_paths["synced_ref_clip"]
                echo_clip_rail_dbfs, echo_clip_shared_variant = sample_absolute_clip_params(
                    synced_cfg["rail_dbfs_range"], synced_cfg["variant_probs"], rng_b3
                )
                echo_clip_rail = float(
                    10.0 ** (echo_clip_rail_dbfs / 20.0)
                )
                echo_before_clip = echo_after_ser_1d.copy()
                farend_before_clip = farend_1d.copy()
                echo_clip_peak_before = float(np.max(np.abs(echo_before_clip)))
                echo_clip_rms_before = float(np.sqrt(np.mean(echo_before_clip ** 2)))
                echo_clip_echo_rail_facts = {}
                echo_clip_farend_rail_facts = {}
                echo_after_ser_1d, echo_clip_variant, _ = apply_absolute_rail_clip(
                    echo_before_clip,
                    synced_cfg["rail_dbfs_range"],
                    synced_cfg["variant_probs"],
                    rng_b3,
                    rail_dbfs=echo_clip_rail_dbfs,
                    variant=echo_clip_shared_variant,
                    facts_out=echo_clip_echo_rail_facts,
                )
                farend_1d, farend_clip_variant, _ = apply_absolute_rail_clip(
                    farend_before_clip,
                    synced_cfg["rail_dbfs_range"],
                    synced_cfg["variant_probs"],
                    rng_b3,
                    rail_dbfs=echo_clip_rail_dbfs,
                    variant=echo_clip_shared_variant,
                    facts_out=echo_clip_farend_rail_facts,
                )
                echo_clip_peak_after = float(np.max(np.abs(echo_after_ser_1d)))
                echo_clip_rms_after = float(np.sqrt(np.mean(echo_after_ser_1d ** 2)))
                echo_clip_echo_applied = not np.array_equal(
                    echo_after_ser_1d, echo_before_clip
                )
                echo_clip_farend_applied = not np.array_equal(
                    farend_1d, farend_before_clip
                )
            # ``applied`` 只取决于真实 waveform 变化，不由选中路径或 variant 推断。
            echo_clip_any_applied = (
                echo_clip_echo_applied or echo_clip_farend_applied
            )
            # absolute 路径按两路输入分别记录越 rail，并以样本数加权汇总。
            rail_facts_summary = summarize_absolute_rail_facts(
                echo_clip_echo_rail_facts,
                echo_clip_farend_rail_facts,
            )
            echo_clip_echo_rail_hit = rail_facts_summary["echo_rail_hit"]
            echo_clip_farend_rail_hit = rail_facts_summary["farend_rail_hit"]
            echo_clip_rail_hit = rail_facts_summary["rail_hit"]
            echo_clip_clipped_sample_count = rail_facts_summary[
                "clipped_sample_count"
            ]
            echo_clip_rail_sample_count = rail_facts_summary["rail_sample_count"]
            echo_clip_clipped_sample_ratio = rail_facts_summary[
                "clipped_sample_ratio"
            ]
            # Layer C-aec-b: far-end level is already set by ERL coupling above.
        else:
            _z = np.zeros(actual_len, dtype=np.float32)
            farend_1d = _z.copy()
            echo_raw_1d = _z.copy()
            echo_after_ser_1d = _z.copy()
            c3_load_meta = None

        if spatial_mic is not None and need_c3:
            if moving_tracks is not None:
                echo_render = _render_aec_moving_actor(
                    moving_tracks, moving_request, echo_after_ser_1d, "__echo__",
                    early_boundary=max(0, int(round(cfg.early_rir_sec * sr))),
                    reference_channel=cfg.reference_channel,
                )
                # 参考通道保持 C-aec 已标定及 delay/clip 后的功率；所有通道与
                # full/direct/early 使用同一个 gain，保留空间幅度关系。
                before_power = float(np.mean(echo_after_ser_1d.astype(np.float64) ** 2))
                after_power = float(np.mean(echo_render.reference_tracks[0].astype(np.float64) ** 2))
                if before_power > 0 and after_power <= 1e-20:
                    raise ValueError("AEC moving echo reference channel has no usable energy")
                echo_gain = float(np.sqrt(before_power / after_power)) if before_power > 0 else 1.0
                spatial_echo = echo_render.mixture * np.float32(echo_gain)
                echo_after_ser_1d = spatial_echo[cfg.reference_channel].copy()
                if source_trace is not None and source_trace.capture_stems:
                    source_trace.replace_metadata("moving_scene", {
                        **source_trace.metadata["moving_scene"],
                        "echo_projection_gain": echo_gain,
                        "echo_projection": "recorded_echo_spatial_augmentation",
                    })
                    for kind in ("full", "direct", "early"):
                        source_trace.replace_stem(
                            f"moving_echo_{kind}_pre_mix",
                            getattr(echo_render, f"{kind}_tracks")[0] * np.float32(echo_gain),
                        )
            else:
                if spatial_echo_rirs is None:
                    raise RuntimeError("static AEC spatial echo RIR was not prepared")
                echo_rirs = spatial_echo_rirs
                spatial_echo = render_spatial_scene(
                    echo_after_ser_1d.reshape(1, -1), echo_rirs,
                    early_boundary=echo_rirs.shape[-1],
                    reference_channel=cfg.reference_channel,
                    num_samples=actual_len,
                ).mixture

        actor_rir_policy = (
            RIRPolicy.SPATIAL_SOURCE if cfg.spatial_enabled else RIRPolicy.MONO_SOURCE
        )
        acoustic_actors = [
            AcousticActor(
                uid=speech_uid,
                role=AcousticRole.NEAREND,
                rir_policy=actor_rir_policy,
                waveform=np.asarray(speech_1d, dtype=np.float32),
            )
        ]
        if need_c3:
            acoustic_actors.extend(
                (
                    AcousticActor(
                        uid="__echo__",
                        role=AcousticRole.LOUDSPEAKER_ECHO,
                        rir_policy=actor_rir_policy,
                        waveform=np.asarray(echo_after_ser_1d, dtype=np.float32),
                    ),
                    AcousticActor(
                        uid="__farend__",
                        role=AcousticRole.DIGITAL_REFERENCE,
                        rir_policy=RIRPolicy.NONE,
                        waveform=np.asarray(farend_1d, dtype=np.float32),
                    ),
                )
            )
        acoustic_actor_facts = [actor.to_dict() for actor in acoustic_actors]
        if source_trace is not None:
            source_trace.replace_metadata("acoustic_actors", acoustic_actor_facts)

        aec_disk: dict[str, dict[str, np.ndarray]] | None = None
        echo_ctx: dict[str, Any] = {
            "farend": farend_1d.astype(np.float32, copy=True),
            "echo": echo_after_ser_1d.astype(np.float32, copy=True),
            # 把 C-aec 加载元数据带入 aec_disk，方便 _finalize_sample / 分析工具
            # 在不改 tuple 的前提下获取。注意：time_delay_jitter 的 delay_gt/
            # event_mask 数组不进 meta（调用后即弃），meta 只含标量与事件列表。
            "c3_meta": c3_load_meta,
        }

        # Branch-specific assignment
        # task="aec": remove echo only; everything else passes through.
        # task="joint": target = clean_target (remove echo + noise + reverb)
        _task_aec = self.task == "aec"

        if lc_branch == "prob_target_only":
            noisy_1d = (
                spatial_mic.copy()
                if spatial_mic is not None
                else np.asarray(clean_target, dtype=np.float32).copy()
            )
            clean_1d = np.asarray(clean_target, dtype=np.float32).copy()
            branch_path = "prob_target_only"
        elif lc_branch == "prob_noise_only":
            noisy_1d = (
                spatial_noise.copy()
                if spatial_noise is not None
                else noise_additive_1d.copy()
            )
            clean_1d = (
                noise_additive_1d.copy()
                if _task_aec
                else np.zeros(actual_len, dtype=np.float32)
            )
            branch_path = "prob_noise_only"
        elif lc_branch == "prob_echo_only":
            noisy_1d = (
                spatial_echo.copy()
                if spatial_echo is not None
                else echo_after_ser_1d.copy()
            )
            clean_1d = np.zeros(actual_len, dtype=np.float32)
            aec_disk = {"prob_echo_only": echo_ctx}
            branch_path = "prob_echo_only"
        elif lc_branch == "prob_noise_echo_only":
            noisy_1d = (
                spatial_noise + spatial_echo
                if spatial_noise is not None and spatial_echo is not None
                else noise_additive_1d + echo_after_ser_1d
            ).astype(np.float32)
            clean_1d = (
                noise_additive_1d.copy()
                if _task_aec
                else np.zeros(actual_len, dtype=np.float32)
            )
            aec_disk = {"prob_noise_echo_only": echo_ctx}
            branch_path = "prob_noise_echo_only"
        elif lc_branch == "prob_target_noise":
            noisy_1d = np.asarray(noisy_speech, dtype=np.float32).copy()
            clean_1d = (
                self._reference_observation(noisy_speech).copy()
                if _task_aec
                else np.asarray(clean_target, dtype=np.float32).copy()
            )
            branch_path = "prob_target_noise"
        elif lc_branch == "prob_target_echo":
            # Layer C-aec: echo mixed into mic at acoustic mixing stage (before D/E/F)
            noisy_1d = (
                np.asarray(noisy_speech)
                + (spatial_echo if spatial_echo is not None else echo_after_ser_1d)
            ).astype(np.float32)
            clean_1d = (
                self._reference_observation(noisy_speech).copy()
                if _task_aec
                else np.asarray(clean_target, dtype=np.float32).copy()
            )
            branch_path = "prob_target_echo"
            aec_disk = {"prob_target_echo_disk": echo_ctx}
        elif lc_branch == "prob_target_noise_echo":
            # Layer C-aec: echo mixed into mic at acoustic mixing stage (before D/E/F)
            noisy_1d = (
                np.asarray(noisy_speech)
                + (spatial_echo if spatial_echo is not None else echo_after_ser_1d)
            ).astype(np.float32)
            clean_1d = (
                self._reference_observation(noisy_speech).copy()
                if _task_aec
                else np.asarray(clean_target, dtype=np.float32).copy()
            )
            branch_path = "prob_target_noise_echo"
            aec_disk = {"prob_target_noise_echo_disk": echo_ctx}
        else:
            raise RuntimeError(f"internal: unknown lc_branch {lc_branch!r}")

        # ===== Shared post-mix template: pre-finite -> D-normalize -> D-guard
        #       -> E -> F -> task-post-EF -> final-guard -> post-finite =====
        # Layer D normalization 仅支持 off/joint：mic 为 anchor，全部 roles 同比
        # 随动；随后 guard 对 mic/nearend/echo/noise/farend 五路 coupled bundle
        # 共同缩放，保持 echo 与 farend 的比例关系。
        post_mix_context = AECPostMixContext.from_aec_disk(
            branch=branch_path,
            task=self.task,
            aec_disk=aec_disk,
            used_sidechain_wind=ac.used_sidechain_wind,
            joint_target_rir_mismatch=_joint_target_rir_mismatch,
            sidechain_wind_mismatch_reason=_sidechain_wind_mismatch_reason,
            index=index,
        )
        post_mix_adapter = TaskPostMixAdapter(
            dataset=self,
            rng=rng,
            sample_rate=sr,
            actual_len=actual_len,
            speech_uid=speech_uid,
            context=post_mix_context,
            metadata={"branch": branch_path, "sample_rate": sr},
        )
        initial_post_mix_bundle = post_mix_adapter.prepare_initial_bundle(
            nearend=clean_1d,
            mic=noisy_1d,
            farend=farend_1d,
            echo=echo_after_ser_1d,
            noise=noise_additive_1d,
        )
        post_mix_runtime = PostMixEngine().run_bundle(
            initial_post_mix_bundle,
            post_mix_adapter,
            context=PostMixContext(
                {"branch": branch_path, "sample_rate": int(sr)}
            ),
            retain_stage_states=bool(
                source_trace is not None and source_trace.capture_stems
            ),
        )
        post_mix_result = post_mix_adapter.build_result(post_mix_runtime.trace)
        clean_1d = post_mix_result.bundle.require(POST_MIX_NEAREND).copy()
        noisy_1d = post_mix_result.bundle.require(POST_MIX_MIC).copy()
        farend_1d = post_mix_result.bundle.require(POST_MIX_FAREND).copy()
        echo_after_ser_1d = post_mix_result.bundle.require(POST_MIX_ECHO).copy()
        noise_additive_1d = post_mix_result.bundle.require(POST_MIX_NOISE).copy()
        aec_disk = post_mix_result.aec_disk
        layer_d_facts = copy.deepcopy(dict(post_mix_result.layer_d_facts))
        layer_e_result = post_mix_result.layer_e
        layer_e_metadata = layer_e_result.to_metadata()
        layer_e_metadata["channel_facts"] = [
            dict(facts) for facts in post_mix_result.layer_e_channel_facts
        ]
        _layer_f_ops = list(post_mix_result.layer_f_applied_ops)
        decomposable = post_mix_result.decomposable
        decomposable_reason_codes = post_mix_result.decomposable_reason_codes
        if source_trace is None:
            self._last_mic_extra_noise_policy = post_mix_result.mic_extra_noise_policy
        if source_trace is not None and source_trace.capture_stems:
            _realized_levels = _measure_realized_levels(
                clean_1d,
                self._reference_observation(noisy_1d),
                echo_after_ser_1d,
                farend_1d,
                noise_additive_1d,
                measure_point="post_dataset_post_mix",
                decomposable=decomposable,
                decomposable_reason_codes=decomposable_reason_codes,
                branch=lc_branch,
                task=self.task,
                reconstruction_reference=_reconstruction_reference_for_branch(
                    lc_branch,
                    self.task,
                    clean_1d,
                    echo_after_ser_1d,
                    noise_additive_1d,
                ),
            )
            simulation_by_layer = {
                "A": {
                    "speech_uid": speech_uid,
                    # 精确 crop 起点当前未记录；用 null 明确未知，避免伪造为 0。
                    "a3_crop_start_samples": None,
                    "a3_max_duration_samples": int(max_dur_sr) if max_dur_sr else None,
                    "a3_mode": (
                        "fixed_silence_canvas" if a3_fixed_canvas_applied else "standard"
                    ),
                    "a3_alignment_mode": a3_alignment_mode,
                    "a3_tile_applied": bool(a3_tile_applied),
                    "a3_fixed_canvas_applied": bool(a3_fixed_canvas_applied),
                    "a3_repeat_to_max_duration_enabled": bool(
                        self.repeat_to_max_duration
                    ),
                    "a3_silence_duration_policy": (
                        "fixed_canvas_remainder"
                        if a3_fixed_canvas_applied
                        else "not_applied"
                    ),
                    "repeat_to_max_duration_applied": bool(a3_tile_applied),
                    "samples_before_repeat": int(len_before_align),
                    "samples_after_repeat_before_hp": int(len_after_align),
                    "samples_after_A": int(actual_len),
                    "high_pass_70hz_applied": bool(getattr(cfg, "use_high_pass", True)),
                },
                "B": {
                    "branch": lc_branch,
                    "force_branch": force_branch,
                    "uniform_u_for_branch_sample": branch_sample_u,
                },
                "C": {
                    "need_additive_noise": bool(need_c12),
                    "need_echo_manifest": bool(need_c3),
                    "background_policy": (
                        background_spec.to_dict()
                        if background_spec is not None else None
                    ),
                    "acoustic_actors": acoustic_actor_facts,
                    "c0_reverb_sampled": bool(ac.use_reverb),
                    "source_volume_calibration": _source_volume_calibration,
                    "acoustic_audit": _aec_acoustic_audit,
                    "level_audits": {
                        "snr": LevelAudit(
                            name="aec.snr",
                            candidate_drawn_db=ac.snr_primary_drawn_db,
                            selected_nominal_db=(float(ac.wn_snr) if need_c12 else None),
                            calibration_realized_db=_aggregate_snr,
                            final_realized_db=_realized_levels["snr_db"],
                            numerator=_realized_levels["ratio_reference"],
                            denominator="noise_stem",
                            policy="active_mean_power_calibration/full_mean_power_final",
                            calibration_measure_point="post_noise_mix_pre_c3_echo",
                            final_measure_point="post_dataset_post_mix",
                            applicable=bool(_realized_levels["snr_applicable"]),
                            reason=_realized_levels["snr_reason_code"],
                        ).to_dict(),
                        "ser": LevelAudit(
                            name="aec.ser",
                            candidate_drawn_db=ser_db_this_sample,
                            selected_nominal_db=_effective_ser,
                            calibration_realized_db=_ser_calibration_db,
                            final_realized_db=_realized_levels["ser_db"],
                            numerator=_realized_levels["ratio_reference"],
                            denominator="echo_stem",
                            policy="full_mean_power",
                            calibration_measure_point="post_ser_erl_scaling_pre_delay_clip",
                            final_measure_point="post_dataset_post_mix",
                            applicable=bool(_realized_levels["ser_applicable"]),
                            reason=_realized_levels["ser_reason_code"],
                        ).to_dict(),
                        "erl": LevelAudit(
                            name="aec.erl",
                            candidate_drawn_db=_erl_db_record,
                            selected_nominal_db=_erl_db_record,
                            calibration_realized_db=_erl_calibration_db,
                            final_realized_db=_realized_levels["erl_db"],
                            numerator="farend_stem",
                            denominator="echo_stem",
                            policy="full_mean_power",
                            calibration_measure_point="post_ser_erl_scaling_pre_delay_clip",
                            final_measure_point="post_dataset_post_mix",
                            applicable=bool(_realized_levels["erl_applicable"]),
                            reason=_realized_levels["erl_reason_code"],
                            observations={"farend_target_dbfs": _far_db_record},
                        ).to_dict(),
                    },
                    "c1_c2_nominal_snr_db": float(ac.wn_snr) if need_c12 else None,
                    "c1_c2_snr_primary_drawn_db": ac.snr_primary_drawn_db,
                    "c1_c2_wind_noise_snr_candidate_drawn_db": ac.wind_noise_snr_candidate_db,
                    "c1_c2_additive_snr_db_each_mix": [float(x) for x in ac.additive_snr_db_list],
                    "c1_c2_wind_sidechain_applied": bool(ac.used_sidechain_wind),
                    "c1_c2_mix_two_segments_drawn": bool(ac.mix_two),
                    "c1_c2_want_sidechain_drawn": bool(ac.want_sidechain),
                    "c3_ser_db_drawn": ser_db_this_sample,
                    "c3_ser_scaling_used": bool(ser_db_this_sample is not None),
                    "c3_erl_db": _erl_db_record,       # ERL coupling: far − echo (dB)
                    "c3_far_db": _far_db_record,       # derived far-end level (dBFS RMS)
                    "c3_effective_ser_db": _effective_ser,  # actual SER after echo_db clamp
                    "c3_erl_range_audit": _erl_range_audit,
                    "c3_disk_pair_load": c3_load_meta,
                    "c3_echo_device_eq": {
                        "applied": bool(echo_eq_variant is not None),
                        "variant": echo_eq_variant,
                    },
                    "c3_echo_clipping": {
                        "applied": bool(echo_clip_any_applied),
                        "echo_applied": bool(echo_clip_echo_applied),
                        "farend_applied": bool(echo_clip_farend_applied),
                        "any_applied": bool(echo_clip_any_applied),
                        "path": echo_clip_path,
                        # 模型 reference 的采集点。
                        "capture_point": capture_point_of(echo_clip_path),
                        "variant": echo_clip_variant,
                        "farend_variant": farend_clip_variant,
                        "threshold_frac": echo_clip_threshold_frac,
                        "threshold_abs": echo_clip_threshold_abs,
                        "rail": echo_clip_rail,
                        "rail_dbfs": echo_clip_rail_dbfs,
                        "echo_rail_facts": echo_clip_echo_rail_facts,
                        "farend_rail_facts": echo_clip_farend_rail_facts,
                        "echo_rail_hit": echo_clip_echo_rail_hit,
                        "farend_rail_hit": echo_clip_farend_rail_hit,
                        "rail_hit": echo_clip_rail_hit,
                        "clipped_sample_count": echo_clip_clipped_sample_count,
                        "rail_sample_count": echo_clip_rail_sample_count,
                        "clipped_sample_ratio": echo_clip_clipped_sample_ratio,
                        "shared_variant": echo_clip_shared_variant,
                        "boost_db": echo_clip_boost_db,
                        "effective_ser_db": echo_clip_effective_ser_db,
                        "peak_before": echo_clip_peak_before,
                        "peak_after": echo_clip_peak_after,
                        "rms_before": echo_clip_rms_before,
                        "rms_after": echo_clip_rms_after,
                    },
                },
                "D": layer_d_facts,
                "E": layer_e_metadata,
                "F": {
                    "noisy_post_mix_degradations_enabled": bool(
                        getattr(cfg, "noisy_post_mix_degradations_enabled", False)
                    ),
                    "cpu_applied_ops": list(_layer_f_ops),
                    "observation_clipping": dict(
                        post_mix_result.layer_f_clipping_facts
                    ),
                },
                "final_peak_normalize": dict(post_mix_result.final_guard_facts),
                # Dataset post-mix 完成、最终裁剪前审计。
                # SER/ERL/SNR 独立判定 applicability，residual 只审计、不进 tuple。
                "realized": _realized_levels,
            }
            source_trace.replace_metadata("simulation_by_layer", simulation_by_layer)
            source_trace.replace_metadata("branch", branch_path)

        return AECRunOutput(
            clean=clean_1d,
            noisy=noisy_1d,
            sample_rate=sr,
            aec_disk=aec_disk,
            context=AECRunContext(
                branch=branch_path,
                noise_additive_1d=noise_additive_1d.copy(),
                decomposable=decomposable,
                decomposable_reason_codes=decomposable_reason_codes,
                c0_audit_stems=ac.c0_audit_stems,
                acoustic_audit=_aec_acoustic_audit,
                acoustic_scene_plan=(
                    None if cfg.spatial_enabled else ac.acoustic_scene_plan.to_dict()
                ),
                spatial_scene_plan=spatial_scene_plan,
            ),
            post_mix=post_mix_result,
        )

    def _build_sample(
        self,
        index: int,
        uid: str,
        slen: int,
        work_sr: int,
        force_branch: str | None = None,
        *,
        trace: SimulationTrace,
    ) -> SampleBuildResult:
        """运行 AEC 仿真并返回 sample 与调用方提供的 trace。"""
        rng = self._get_rng(index)
        scene_trace = trace.source("scene")
        run_output = self._run(
            uid,
            slen,
            work_sr,
            index=index,
            force_branch=force_branch,
            rng=rng,
            source_trace=scene_trace,
        )
        run_context = run_output.context
        post_mix_result = run_output.post_mix
        layer_d_facts = copy.deepcopy(dict(post_mix_result.layer_d_facts))
        sr_before = int(run_output.sample_rate)

        # 所有 export stems 在 delivery bundle 中同步 time transform。
        delivery_bundle = _build_aec_delivery_bundle(
            post_mix_result,
            capture_stems=trace.capture_stems,
        )
        scene_time_result = apply_scene_time_transform(
            dataset=self,
            bundle=delivery_bundle,
            source_sample_rate=sr_before,
            rng=rng,
            fixed_silence_canvas_active=bool(
                getattr(self, "_fixed_silence_canvas_active", False)
            ),
        )
        final_bundle = scene_time_result.bundle
        sp = final_bundle.require(POST_MIX_NEAREND).copy()
        nz = final_bundle.require(POST_MIX_MIC).copy()
        noise_additive_1d = final_bundle.require(POST_MIX_NOISE).copy()
        fs_a = scene_time_result.sample_rate
        L_a = scene_time_result.plan.output_length
        resampled = sr_before != int(fs_a)
        ctx_a = post_mix_result.context.build_aec_disk(
            final_bundle,
            resampled_from=(sr_before if resampled else None),
        )

        len_pre_final = scene_time_result.plan.input_length
        if scene_time_result.plan.operations and resampled:
            len_pre_final = scene_time_result.plan.operations[0].output_length
        if trace.capture_stems:
            for stem_key in _LAYER_D_STEM_KEYS:
                role = f"export/{stem_key}"
                if role in final_bundle.tracks:
                    scene_trace.replace_stem(stem_key, final_bundle.require(role))
            repeat_export = scene_trace.metadata.get("repeat_export")
            if isinstance(repeat_export, dict):
                repeat_export = dict(repeat_export)
                if resampled:
                    repeat_export["sample_rate"] = int(fs_a)
                    repeat_export["max_duration_samples"] = int(self._max_duration_for_sr(fs_a))
                    repeat_export["samples_after_repeat_step"] = int(len_pre_final)
                self._enrich_repeat_export_final(
                    repeat_export, fs_a, int(len_pre_final), L_a
                )
                scene_trace.replace_metadata("repeat_export", repeat_export)

        scene_trace.replace_metadata("branch", run_context.branch)
        scene_trace.replace_metadata("scene_name", validate_scene_name("aec", run_context.branch))
        branch = run_context.branch
        scene_trace.replace_metadata(
            "scene_spec",
            SceneSpec(
                scene_name=validate_scene_name("aec", branch),
                target_present=branch in {
                    "prob_target_only", "prob_target_noise", "prob_target_echo", "prob_target_noise_echo"
                },
                interference_present=False,
                background_present=branch in {
                    "prob_noise_only", "prob_noise_echo_only", "prob_target_noise", "prob_target_noise_echo"
                },
                echo_present=branch in {
                    "prob_echo_only", "prob_noise_echo_only", "prob_target_echo", "prob_target_noise_echo"
                },
                source_count=1 if branch in {
                    "prob_target_only", "prob_target_noise", "prob_target_echo", "prob_target_noise_echo"
                } else 0,
                task="aec",
            ).to_dict(),
        )
        if run_context.spatial_scene_plan is not None:
            scene_trace.replace_metadata(
                "spatial_scene_plan", run_context.spatial_scene_plan
            )
        elif run_context.acoustic_scene_plan is not None:
            scene_trace.replace_metadata(
                "acoustic_scene_plan", run_context.acoustic_scene_plan
            )
        aec_context_facts = {
            "decomposable": run_context.decomposable,
            "decomposable_reason_codes": run_context.decomposable_reason_codes,
            "acoustic_audit": run_context.acoustic_audit,
        }
        scene_trace.replace_metadata("aec_context", copy.deepcopy(aec_context_facts))
        scene_trace.replace_metadata(
            "aec_post_ef",
            {
                "stage": "task-post-EF",
                "mic_extra_noise": (
                    post_mix_result.mic_extra_noise_policy.to_dict()
                    if post_mix_result.mic_extra_noise_policy is not None
                    else None
                ),
            },
        )
        post_mix_stage_order = list(post_mix_result.trace.stage_order)
        post_mix_stage_applied = {
            result.stage: result.applied for result in post_mix_result.trace.results
        }
        scene_time_facts = scene_time_result.to_metadata()
        scene_trace.replace_metadata("post_mix_stage_order", post_mix_stage_order)
        scene_trace.replace_metadata(
            "post_mix_role_contract", post_mix_result.role_contract.to_dict()
        )
        scene_trace.replace_metadata(
            "post_mix_stage_applied", post_mix_stage_applied
        )
        scene_trace.replace_metadata("layer_d_facts", copy.deepcopy(layer_d_facts))
        scene_trace.replace_metadata("layer_e", post_mix_result.layer_e.to_metadata())
        scene_trace.replace_metadata(
            "layer_f",
            {
                "applied_ops": list(post_mix_result.layer_f_applied_ops),
                "observation_clipping": dict(
                    post_mix_result.layer_f_clipping_facts
                ),
            },
        )
        scene_trace.replace_metadata(
            "final_peak_guard", dict(post_mix_result.final_guard_facts)
        )
        scene_trace.replace_metadata("final_time_transform", scene_time_facts)
        if trace.capture_stems:
            _rf_echo = None
            _rf_farend = None
            if ctx_a:
                for _role_ctx in ctx_a.values():
                    if isinstance(_role_ctx, dict) and _role_ctx.get("echo") is not None:
                        _rf_echo = _role_ctx.get("echo")
                        _rf_farend = _role_ctx.get("farend")
                        break
            _final_echo_arr = (
                np.asarray(_rf_echo, dtype=np.float32).ravel()
                if _rf_echo is not None
                else np.zeros(L_a, dtype=np.float32)
            )
            realized_final = _measure_realized_levels(
                sp,
                self._reference_observation(nz),
                _rf_echo,
                _rf_farend,
                noise_additive_1d,
                measure_point="final",
                decomposable=run_context.decomposable,
                decomposable_reason_codes=run_context.decomposable_reason_codes,
                branch=run_context.branch,
                task=self.task,
                reconstruction_reference=_reconstruction_reference_for_branch(
                    run_context.branch,
                    self.task,
                    sp,
                    _final_echo_arr,
                    noise_additive_1d,
                ),
            )
            scene_trace.replace_metadata("realized_final", realized_final)
            scene_trace.replace_stem("final_clean", sp)
            scene_trace.replace_stem("final_noisy", nz)
            scene_trace.replace_stem("final_noise", noise_additive_1d)
            if _rf_echo is not None:
                scene_trace.replace_stem("final_echo", _rf_echo)
        trace.replace_metadata("branch", run_context.branch)
        trace.replace_metadata("post_mix_stage_order", post_mix_stage_order)
        trace.replace_metadata(
            "post_mix_role_contract", post_mix_result.role_contract.to_dict()
        )
        trace.replace_metadata("aec_context", copy.deepcopy(aec_context_facts))
        trace.replace_metadata("layer_d_facts", copy.deepcopy(layer_d_facts))
        trace.replace_metadata("final_time_transform", scene_time_facts)
        trace.set_legacy_snapshot("_last_run_branch", run_context.branch)
        trace.set_legacy_snapshot("_last_noise_additive_1d", noise_additive_1d.copy())
        trace.set_legacy_snapshot("_last_layer_e_result", post_mix_result.layer_e)
        trace.set_legacy_snapshot(
            "_last_layer_f_applied_ops", list(post_mix_result.layer_f_applied_ops)
        )
        trace.set_legacy_snapshot("_last_layer_ef_decomposable", run_context.decomposable)
        trace.set_legacy_snapshot(
            "_last_decomposable_reason_codes",
            list(run_context.decomposable_reason_codes),
        )
        trace.set_legacy_snapshot("_last_c0_audit_stems", run_context.c0_audit_stems)
        trace.set_legacy_snapshot(
            "_last_mic_extra_noise_policy", post_mix_result.mic_extra_noise_policy
        )
        sample = project_training_tuple(
            dataset=self, bundle=final_bundle, sample_rate=fs_a, index=index,
            uid=uid, rng=rng, context=post_mix_result.context,
            resampled_from=(sr_before if resampled else None),
        )
        if trace.capture_stems and ctx_a:
            disk = next(iter(ctx_a.values()))
            scene_trace.replace_metadata("aec_disk_consistency", {
                "farend_matches_output": bool(np.array_equal(disk["farend"], sample[2])),
                "echo_matches_final_stem": bool(np.array_equal(disk["echo"], scene_trace.stems["final_echo"])),
                "length_matches_output": len(disk["echo"]) == len(disk["farend"]) == int(sample[4]),
            })
        return SampleBuildResult(sample=sample, trace=trace)


    def _placeholder_sample(
        self, work_sr: int, index: int, uid: str
    ) -> tuple:
        """Return a silence placeholder when mixing fails."""
        self._last_run_branch = None
        mic, _ = self._make_observation_placeholder(work_sr)
        silence = np.zeros(mic.shape[-1], dtype=np.float32)
        return self._finalize_sample(
            mic, silence.copy(), silence.copy(), silence.copy(),
            work_sr, index, uid, None,
        )


    # ------------------------------------------------------------------
    # Training dict assembly
    # ------------------------------------------------------------------
    def _assemble_training_tuple(
        self,
        clean_nearend: np.ndarray,
        noisy_nearend: np.ndarray,
        fs_out: int,
        index: int,
        uid: str,
        aec_disk_ctx: dict[str, dict[str, np.ndarray]] | None,
        rng: np.random.Generator,
        noise_additive_1d: np.ndarray,
        *,
        branch: str | None = None,
    ) -> tuple:
        """Assemble the final training tuple from near-end and AEC disk context.

        Branches without echo (prob_target_only, prob_noise_only, prob_target_noise) are packed
        as SE-only. Echo-only and double-talk branches use the disk context.
        ``noise_additive_1d`` is passed through to state label generation.
        """
        br = branch if branch is not None else getattr(self, "_last_run_branch", None)
        pure_flags = self._pure_branch_bool_tensors(br)
        if br in ("prob_target_only", "prob_noise_only", "prob_target_noise"):
            return self._pack_se_only(
                clean_nearend, noisy_nearend, fs_out, uid, pure_flags, noise_additive_1d
            )

        # prob_noise_echo_only branch
        ne_ctx = (aec_disk_ctx or {}).get("prob_noise_echo_only") if aec_disk_ctx else None
        if br == "prob_noise_echo_only" and ne_ctx is not None:
            n = np.asarray(noisy_nearend, dtype=np.float32)
            c = np.asarray(clean_nearend, dtype=np.float32).ravel()
            t = min(
                n.shape[-1],
                int(c.size),
                int(np.asarray(ne_ctx["farend"], dtype=np.float32).ravel().size),
                int(np.asarray(ne_ctx["echo"], dtype=np.float32).ravel().size),
            )
            mic = n[..., :t].copy()
            farend = np.asarray(ne_ctx["farend"], dtype=np.float32).ravel()[:t].copy()
            echo = np.asarray(ne_ctx["echo"], dtype=np.float32).ravel()[:t].copy()
            target = c[:t].copy()
            return self._finalize_sample(
                mic, farend, target, echo, fs_out, index, uid, pure_flags,
                noise=noise_additive_1d,
            )

        # prob_echo_only branch
        c64 = np.asarray(clean_nearend, dtype=np.float64).ravel()
        n64 = np.asarray(noisy_nearend, dtype=np.float64).ravel()
        tmin = min(c64.size, n64.size)
        near_pow = float(np.mean(c64[:tmin] ** 2) + np.mean(n64[:tmin] ** 2)) if tmin > 0 else 0.0
        pure_echo_route = br == "prob_echo_only"
        if near_pow < 1e-20 and not pure_echo_route:
            log_a = logging.getLogger(__name__)
            log_a.warning(
                "AEC assemble: near-end energy < 1e-20 but branch=%r — continuing with standard assembly; "
                "index=%s uid=%s",
                br,
                index,
                uid,
            )

        if pure_echo_route:
            eo_ctx = (aec_disk_ctx or {}).get("prob_echo_only") if aec_disk_ctx else None
            if eo_ctx is not None:
                ctx = eo_ctx
                n = np.asarray(noisy_nearend, dtype=np.float32)
                c = np.asarray(clean_nearend, dtype=np.float32).ravel()
                t = min(
                    n.shape[-1],
                    int(c.size),
                    int(np.asarray(ctx["farend"], dtype=np.float32).ravel().size),
                    int(np.asarray(ctx["echo"], dtype=np.float32).ravel().size),
                )
                farend = np.asarray(ctx["farend"], dtype=np.float32).ravel()[:t].copy()
                echo = np.asarray(ctx["echo"], dtype=np.float32).ravel()[:t].copy()
                target = c[:t].copy()
                mic = n[..., :t].astype(np.float32, copy=True)
                return self._finalize_sample(
                    mic, farend, target, echo, fs_out, index, uid, pure_flags,
                    noise=noise_additive_1d,
                )
            # Fallback should never happen: _run already loads C-aec for need_c3 branches.
            # Re-loading would use a *different* echo file, breaking mic/echo consistency.
            raise RuntimeError(
                f"AEC prob_echo_only fallback triggered but aec_disk_ctx['prob_echo_only'] is None. "
                f"This means C-aec did not populate the expected context. "
                f"index={index} uid={uid}."
            )

        # prob_target_echo / prob_target_noise_echo branches
        if br in ("prob_target_echo", "prob_target_noise_echo"):
            if aec_disk_ctx:
                nd_ctx = aec_disk_ctx.get(
                    "prob_target_noise_echo_disk" if br == "prob_target_noise_echo" else "prob_target_echo_disk"
                )
            else:
                nd_ctx = None
            if nd_ctx is not None:
                n = np.asarray(noisy_nearend, dtype=np.float32)
                c = np.asarray(clean_nearend, dtype=np.float32).ravel()
                fa = np.asarray(nd_ctx["farend"], dtype=np.float32).ravel()
                eco = np.asarray(nd_ctx["echo"], dtype=np.float32).ravel()
                t = min(int(n.shape[-1]), int(c.size), int(fa.size), int(eco.size))
                if t <= 0:
                    # 空交集无法产出任何有效样本；静默返回 1 样点零 tuple 会污染训练数据。
                    raise RuntimeError(
                        f"AEC {br} branch has empty waveform intersection "
                        f"(t={t}); mic={int(n.shape[-1])}, nearend={int(c.size)}, "
                        f"farend={int(fa.size)}, echo={int(eco.size)}. "
                        f"index={index} uid={uid}."
                    )
                # n already contains echo (mixed in _run after C-aec with delay applied there).
                # Do NOT add echo again — that would double the echo.
                mic = n[..., :t].astype(np.float32, copy=True)
                farend = fa[:t].astype(np.float32, copy=True)
                echo = eco[:t].astype(np.float32, copy=True)
                mic, farend, c_out, echo = self._align_four(
                    mic, farend, c.astype(np.float32, copy=False), echo
                )
                return self._finalize_sample(
                    mic, farend, c_out, echo, fs_out, index, uid, pure_flags,
                    noise=noise_additive_1d,
                )
            # Fallback should never happen: _run already loads C-aec for need_c3 branches.
            raise RuntimeError(
                f"AEC {br} fallback triggered but aec_disk_ctx['{br}_disk'] is None. "
                f"C-aec context missing for a branch that requires it. "
                f"index={index} uid={uid}."
            )

        # Fallback: unknown branch — this indicates a logic error in _run branch selection
        raise RuntimeError(
            f"AEC _assemble_training_tuple: unknown branch {br!r}. "
            f"Expected one of the seven Layer-B branches. index={index} uid={uid}."
        )


# =====================================================================
# Branch helpers
# =====================================================================

def _normalize_force_branch(force_branch: str) -> str:
    """校验 force_branch 为 AEC 七路 canonical branch 名，非法值报 ValueError。"""
    fb = str(force_branch)
    allowed = {
        "prob_target_only", "prob_noise_only", "prob_echo_only",
        "prob_noise_echo_only", "prob_target_noise", "prob_target_echo",
        "prob_target_noise_echo",
    }
    if fb in allowed:
        return fb
    raise ValueError(
        f"force_branch must be one of {sorted(allowed)}, got {force_branch!r}"
    )


def _layer_c_subflags(br: str) -> tuple[bool, bool]:
    """Return (need_c12, need_c3) for a given branch.

    need_c12: additive noise + wind noise + colored noise (C-1/C-2/C-3)
    need_c3: manifest echo + SER (C-aec)
    C-0 (RIR) always runs.

    命名注意：本函数及导出元数据 ``c3_ser_*`` 键中的 "c3" 均指 C-aec；
    文档/配置注释中的 "C-3 彩色噪声" 是另一命名域（Layer C 第三子层），
    两者含义不同，勿混。
    """
    flags: dict[str, tuple[bool, bool]] = {
        "prob_target_only": (False, False),
        "prob_noise_only": (True, False),
        "prob_echo_only": (False, True),
        "prob_noise_echo_only": (True, True),
        "prob_target_noise": (True, False),
        "prob_target_echo": (False, True),
        "prob_target_noise_echo": (True, True),
    }
    return flags[br]


def _apply_peak_relative_echo(
    echo: np.ndarray,
    threshold_frac_range: tuple[float, float],
    variant_probs: dict[str, float],
    rng: np.random.Generator,
    *,
    threshold_frac: float | None = None,
    variant: str | None = None,
) -> tuple[np.ndarray, str | None, float, float, float, float]:
    """对 echo 截幅并返回审计值；可复用预抽取的同步参数。"""
    peak_before = float(np.max(np.abs(echo)))
    rms_before = float(np.sqrt(np.mean(echo ** 2)))
    echo_after, variant = apply_peak_relative_clip(
        echo,
        threshold_frac_range,
        variant_probs,
        rng,
        threshold_frac=threshold_frac,
        variant=variant,
    )
    peak_after = float(np.max(np.abs(echo_after)))
    rms_after = float(np.sqrt(np.mean(echo_after ** 2)))
    return echo_after, variant, peak_before, rms_before, peak_after, rms_after
