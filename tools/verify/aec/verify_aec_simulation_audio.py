#!/usr/bin/env python3
"""验证 AEC 动态仿真（``tasks.aec.data.dynamic_dataset.DynamicMixingDataset``）各 ``force_branch`` 场景波形。

对每种场景通过 Dataset 验证入口固定 ``force_branch``，写出与 **collate 后送模型** 一致的三路波形，
以及 Layer D 阶段快照，便于对照 mic / nearend(target) / farend 与 D 内同增益分量。

与训练一致：使用 ``build_dynamic_aec_dataset`` + ``meta.seed`` + ``set_epoch(0, base_seed)``；
验证副本额外强制 ``strict_load=True`` 和 stems 导出，并在写 WAV 前执行 schema/finite/branch 硬门禁。

默认 **每个场景导出多条**（``--num-samples``，默认 10 条），使用连续数据集下标 ``index, index+1, …``；
每条 ``meta.json`` 含 ``simulation_by_layer``（A–F 分层记录**本样本仿真实际取值**，非 YAML 范围）；AEC frontend facts 单独记录，不属于仿真层。

**送给模型的三路**（与 ``aec_collate_fn`` 一致）：``for_model_mic.wav``、``for_model_nearend_target.wav``、
``for_model_farend.wav``。

**Layer D（post-mix amplitude conditioning，E 前）**：``layer_D_*.wav`` 是按固定顺序执行
D-normalize → D-guard 后的快照；D-guard 对所有相关 stems 使用同一安全增益。
Dataset 仿真顺序为 pre-finite → D-normalize → D-guard → E → F → task-post-EF → final-guard → post-finite；
AECDeploymentFrontend 在 Trainer/Inference 的 STFT 前单独运行，不属于 Dataset 仿真。observation/device clipping 的权威位置是 Layer F，final-guard 仅作确定性安全网；
源音量标定仍位于 Layer C-0b（post-RIR/pre-noise-echo）。

输出目录结构::

    <output_dir>/
      README.txt
      summary.json
      simulation_cfg_bounds.json   # 可选：YAML 边界快照（与单条 meta 解耦）
      01_prob_target_only/
        sample_000_idx0/
          for_model_mic.wav
          for_model_nearend_target.wav
          for_model_farend.wav
          layer_D_*.wav
          meta.json
        ...
      ...

Usage::

    cd use_train
    python tools/verify/aec/verify_aec_simulation_audio.py \\
        -C configs/aec/microsoft_challenge/cagcrn/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml \\
        -o exp/verify_aec_audio \\
        --index 0 \\
        --num-samples 10

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# prepare_data_tools -> tools -> use_train
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.data.simulation_layer_view import canonical_simulation_view  # noqa: E402
from tasks.aec.data.dynamic_aec_build import build_dynamic_aec_dataset  # noqa: E402
from tools.verify._audio_verify_utils import (  # noqa: E402
    _align_audio_1d_to_len,
    _compute_active_rms,
    _make_stereo_left_right,
    _rms,
    _save_wav,
    _to_dbfs,
    _to_numpy_f32,
)
from tools.prepare_data_tools._common import json_safe  # noqa: E402
from tools.verify.aec.deep_check_aec_simulation import (  # noqa: E402
    _get_validated_forced_branch_sample,
    _prepare_verification_config,
    _unpack_dataset_sample as _unpack_verified_dataset_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("verify_aec_simulation_audio")


def _json_finite(value: Any) -> Any:
    """转成 JSON 原生类型，并把递归出现的 NaN/Inf 规范化为 null。"""
    safe = json_safe(value)
    if isinstance(safe, dict):
        return {str(key): _json_finite(child) for key, child in safe.items()}
    if isinstance(safe, list):
        return [_json_finite(child) for child in safe]
    if isinstance(safe, float) and not np.isfinite(safe):
        return None
    return safe


# Layer D 阶段导出的 stem 键名与输出 wav 文件名映射
_LAYER_D_WAV_EXPORTS: list[tuple[str, str]] = [
    ("layer_d_noisy", "layer_D_noisy.wav"),
    ("layer_d_clean", "layer_D_clean.wav"),
    ("layer_d_noise", "layer_D_noise_additive.wav"),
    ("layer_d_echo", "layer_D_echo_after_ser.wav"),
    ("farend_ref", "layer_D_farend.wav"),
]

# AEC 七场景（与 Layer B 分支一一对应）
# 子目录名前缀用数字排序，便于浏览。
SCENARIOS: list[tuple[str, str, str]] = [
    ("01_prob_target_only", "prob_target_only", "prob_target_only：仅近端；farend=0；mic≈target"),
    ("02_prob_noise_only", "prob_noise_only", "prob_noise_only：仅加性噪声；joint target=0，aec target=透传噪声"),
    ("03_prob_echo_only", "prob_echo_only", "prob_echo_only：仅磁盘回声；target=0；mic 以回声为主"),
    ("04_prob_noise_echo_only", "prob_noise_echo_only", "prob_noise_echo_only：加性噪声+回声；joint target=0，aec target=透传噪声"),
    ("05_prob_target_noise", "prob_target_noise", "prob_target_noise：近端+噪；无清单回声"),
    ("06_prob_target_echo", "prob_target_echo", "prob_target_echo：近端+回声；双讲组包"),
    ("07_prob_target_noise_echo", "prob_target_noise_echo", "prob_target_noise_echo：近端+噪+回声；双讲组包"),
]


def _stems_dict(st: dict[str, Any] | None) -> dict[str, Any] | None:
    """仅摘要 ``final_*`` / ``layer_d_*`` 与 JSON 元数据。"""
    if not st:
        return None
    _wave_keys = (
        "final_noisy",
        "final_clean",
        "layer_d_noisy",
        "layer_d_clean",
        "layer_d_noise",
        "layer_d_echo",
        "farend_ref",
    )
    out: dict[str, Any] = {}
    for k, v in st.items():
        if k in ("simulation_by_layer", "speech_splice_export", "repeat_export"):
            out[k] = v
        elif k in _wave_keys and isinstance(v, np.ndarray):
            out[k] = {"shape": list(v.shape), "rms": _rms(v)}
        elif k in _wave_keys:
            out[k] = str(type(v))
    return out


def _curated_simulation_cfg_bounds(ds: Any) -> dict[str, Any]:
    """从 ``DynamicMixingDataset.sim_cfg`` 抽取与 SNR/SER/音量/Layer B 相关的配置边界（非本样本随机值）。"""
    c = getattr(ds, "sim_cfg", None)
    if c is None:
        return {}

    def gv(name: str, default: Any = None) -> Any:
        return getattr(c, name, default)

    keys_scalar = [
        "prob_target_only",
        "prob_noise_only",
        "prob_echo_only",
        "prob_noise_echo_only",
        "prob_target_noise",
        "prob_target_echo",
        "prob_target_noise_echo",
        "prob_reverberation",
        "snr_low",
        "snr_high",
        "prob_wind_noise_sidechain",
        "noise_mix_prob",
        "prob_channel_enabled",
        "noisy_post_mix_degradations_enabled",
        "ser_range",
    ]
    out: dict[str, Any] = {}
    for k in keys_scalar:
        if not hasattr(c, k):
            continue
        out[k] = gv(k)

    if hasattr(ds, "_ser_low") and hasattr(ds, "_ser_high"):
        out["ser_effective_bounds_db"] = [float(ds._ser_low), float(ds._ser_high)]
        out["ser_scaling_active"] = bool(getattr(ds, "_apply_ser", False))

    wn = gv("wind_noise_config", None)
    if isinstance(wn, dict):
        out["wind_noise_snr_bounds_db"] = [float(v) for v in wn.get("wind_noise_snr_range", [0.0, 0.0])]

    return json_safe(out)


def _unpack_dataset_sample(sample: Any) -> tuple[Any, Any, Any, Any, Any]:
    """复用 deep-check 的 AEC tuple 兼容契约。"""
    return _unpack_verified_dataset_sample(sample)


def _as_mapping(value: Any) -> dict[str, Any]:
    """将 sidecar mapping 规范化为普通 dict；未知形状按空映射处理。"""
    return dict(value) if isinstance(value, dict) else {}


_CANONICAL_GUARD_KEYS = {"roles", "threshold", "peak", "scale", "applied"}
_CANONICAL_D_NORMALIZE_KEYS = {
    "mode",
    "target_peak",
    "scale",
    "preserves_decomposition",
}


def _require_exact_fact_keys(
    value: Any, *, expected: set[str], path: str
) -> dict[str, Any]:
    """要求事实 mapping 严格匹配 canonical 键集合。"""
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise RuntimeError(
            f"{path} must contain exactly {sorted(expected)}; got {sorted(actual)}"
        )
    return value


def _validated_layer_d_facts(by_layer: dict[str, Any]) -> dict[str, Any]:
    """只读取 AEC Layer D facts，并确保仿真层仅覆盖 A-F。"""
    allowed_layers = {"A", "B", "C", "D", "E", "F"}
    unexpected = sorted(set(by_layer) - allowed_layers)
    if unexpected:
        raise RuntimeError(
            "simulation_by_layer must contain only Layers A-F; "
            f"unexpected={unexpected}"
        )
    layer_d = _require_exact_fact_keys(
        by_layer.get("D"),
        expected={"stage_order", "normalize", "guard"},
        path="simulation_by_layer.D",
    )
    if tuple(layer_d["stage_order"]) != ("D-normalize", "D-guard"):
        raise RuntimeError(
            "simulation_by_layer.D.stage_order must be "
            "('D-normalize', 'D-guard')"
        )
    normalize = _require_exact_fact_keys(
        layer_d["normalize"],
        expected=_CANONICAL_D_NORMALIZE_KEYS,
        path="simulation_by_layer.D.normalize",
    )
    guard = _require_exact_fact_keys(
        layer_d["guard"],
        expected={"mic_related", "farend"},
        path="simulation_by_layer.D.guard",
    )
    mic_related = _require_exact_fact_keys(
        guard["mic_related"],
        expected=_CANONICAL_GUARD_KEYS,
        path="simulation_by_layer.D.guard.mic_related",
    )
    _require_exact_fact_keys(
        guard["farend"],
        expected=_CANONICAL_GUARD_KEYS,
        path="simulation_by_layer.D.guard.farend",
    )
    return layer_d


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    """返回第一个存在的键值（保留显式 False/0/None）。"""
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _bool_fact(mapping: dict[str, Any], *keys: str, default: bool = False) -> bool:
    value = _first_present(mapping, *keys)
    return default if value is None else bool(value)


def _float_fact(mapping: dict[str, Any], *keys: str) -> float | None:
    value = _first_present(mapping, *keys)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _int_fact(mapping: dict[str, Any], key: str) -> int | None:
    """读取非负整数统计；旧 sidecar 缺字段或非法值时返回 None。"""
    value = mapping.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if result < 0:
        return None
    try:
        if float(value) != float(result):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return result


def _layer_e_clipping_facts(layer_e: dict[str, Any]) -> dict[str, bool]:
    """提取 Layer E channel clipping 的 selected/applied/changed 事实。"""
    selected = {str(item) for item in layer_e.get("selected", [])}
    applied = {str(item) for item in layer_e.get("applied", [])}
    operation = {}
    for item in layer_e.get("operations", []):
        if not isinstance(item, dict) or str(item.get("name")) != "clipping":
            continue
        operation = item
        break
    event = "clipping" in selected or _bool_fact(operation, "selected")
    operation_applied = "clipping" in applied or _bool_fact(operation, "applied")
    actual_changed = _bool_fact(
        operation,
        "actual_changed",
        "observation_changed",
        "waveform_changed",
        default=operation_applied,
    )
    return {
        "event": bool(event),
        "applied": bool(operation_applied),
        "actual_changed": bool(actual_changed),
    }


def _observation_clipping_audit(
    by_layer: dict[str, Any],
    *,
    sample_rate: int | None = None,
    final_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """归一化 AEC C/D/E/F 的当前审计事实。

    Layer F clipping 可位于 ``observation_clipping``、``clipping``
    或 F 顶层；这里优先读取嵌套事实，同时兼容迁移期的 ``cpu_applied_ops``。
    rail 命中统计只读取 runtime 权威字段；旧 sidecar 缺字段时保持 ``None``，
    连续命中时长仅由 ``longest_clipped_run_samples`` 和 sample rate 推导。
    """
    layer_c = _as_mapping(by_layer.get("C"))
    c_clip = _as_mapping(layer_c.get("c3_echo_clipping"))
    layer_d_facts = _validated_layer_d_facts(by_layer)
    layer_d_guard = layer_d_facts["guard"]["mic_related"]
    layer_e = _as_mapping(by_layer.get("E"))
    layer_f = _as_mapping(by_layer.get("F"))
    final_guard = _as_mapping(final_guard)
    f_clip = _as_mapping(layer_f.get("observation_clipping"))
    if not f_clip:
        f_clip = _as_mapping(layer_f.get("clipping"))
    if not f_clip:
        f_clip = _as_mapping(layer_f.get("clipping_facts"))
    if not f_clip:
        f_clip = layer_f

    c_applied = _bool_fact(c_clip, "any_applied", "applied")
    c_echo_changed = _bool_fact(c_clip, "echo_applied", default=c_applied)
    e_facts = _layer_e_clipping_facts(layer_e)
    f_ops = {str(item) for item in layer_f.get("cpu_applied_ops", [])}
    f_event = _bool_fact(f_clip, "event", "selected", "triggered")
    if not f_event:
        f_event = _first_present(f_clip, "variant", "mode", "kind") is not None
    if not f_event:
        f_event = bool({"clip", "fixed_rail", "observation_clipping"} & f_ops)
    f_applied = _bool_fact(f_clip, "applied", default=f_event)
    f_changed = _bool_fact(
        f_clip,
        "actual_changed",
        "observation_changed",
        "waveform_changed",
        default=f_applied,
    )
    pre_drive_db = _float_fact(f_clip, "pre_drive_db", "drive_db")
    pre_drive = _float_fact(f_clip, "pre_drive", "pre_drive_gain", "drive")
    if pre_drive is None and pre_drive_db is not None:
        pre_drive = float(10.0 ** (pre_drive_db / 20.0))
    peak_before = _float_fact(f_clip, "peak_before", "pre_peak", "input_peak")
    peak_after_pre_drive = _float_fact(
        f_clip, "peak_after_pre_drive", "pre_clip_peak", "driven_peak"
    )
    if peak_after_pre_drive is None and peak_before is not None and pre_drive is not None:
        peak_after_pre_drive = float(peak_before * pre_drive)

    clipped_sample_count = _int_fact(f_clip, "clipped_sample_count")
    clipped_frame_count = _int_fact(f_clip, "clipped_frame_count")
    longest_clipped_run_samples = _int_fact(f_clip, "longest_clipped_run_samples")
    valid_sample_rate = (
        int(sample_rate)
        if isinstance(sample_rate, (int, np.integer)) and int(sample_rate) > 0
        else None
    )
    longest_clipped_run_ms = (
        1000.0 * longest_clipped_run_samples / valid_sample_rate
        if longest_clipped_run_samples is not None and valid_sample_rate is not None
        else None
    )

    layer_f_facts = {
        "event": bool(f_event),
        "applied": bool(f_applied),
        "actual_changed": bool(f_changed),
        "mode": _first_present(f_clip, "mode", "variant", "kind"),
        "rail": _float_fact(f_clip, "rail", "rail_abs", "threshold"),
        "pre_drive": pre_drive,
        "pre_drive_db": pre_drive_db,
        "peak_before": peak_before,
        "peak_after_pre_drive": peak_after_pre_drive,
        "peak_after": _float_fact(
            f_clip, "peak_after", "post_peak", "output_peak"
        ),
        "changed_sample_ratio": _float_fact(f_clip, "changed_sample_ratio"),
        "clipped_sample_count": clipped_sample_count,
        "clipped_sample_ratio": _float_fact(f_clip, "clipped_sample_ratio"),
        "clipped_frame_count": clipped_frame_count,
        "clipped_frame_ratio": _float_fact(f_clip, "clipped_frame_ratio"),
        "longest_clipped_run_samples": longest_clipped_run_samples,
        "longest_clipped_run_ms": longest_clipped_run_ms,
    }
    return {
        "layer_c_echo_clipping": {
            "event": bool(c_clip.get("path") is not None),
            "actual_changed": bool(c_applied),
            "echo_changed": bool(c_echo_changed),
            "path": c_clip.get("path"),
            "variant": c_clip.get("variant"),
        },
        "layer_d_facts": dict(layer_d_facts),
        "layer_e_channel_clipping": e_facts,
        "layer_f_observation_clipping": layer_f_facts,
        "final_guard": {
            "applied": _bool_fact(final_guard, "applied"),
            "scale": _float_fact(final_guard, "scale"),
        },
        "cooccurrence": {
            "c_and_e_actual": bool(c_applied and e_facts["actual_changed"]),
            "c_and_f_actual": bool(c_applied and f_changed),
            "e_and_f_actual": bool(e_facts["actual_changed"] and f_changed),
            "c_e_f_actual": bool(c_applied and e_facts["actual_changed"] and f_changed),
        },
    }


def _aggregate_clipping_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合小批验证样本的 C/E/F clipping 事件和 realized 统计。"""
    total = len(audits)
    keys = (
        "c_event",
        "c_actual",
        "d_guard_applied",
        "e_event",
        "e_actual",
        "f_event",
        "f_actual",
        "final_guard_applied",
        "c_and_e_actual",
        "c_and_f_actual",
        "e_and_f_actual",
        "c_e_f_actual",
    )
    counts = {key: 0 for key in keys}
    realized_metrics: dict[str, list[float]] = {
        "clipped_sample_count": [],
        "clipped_sample_ratio": [],
        "clipped_frame_count": [],
        "clipped_frame_ratio": [],
        "longest_clipped_run_samples": [],
        "longest_clipped_run_ms": [],
    }
    for audit in audits:
        c = _as_mapping(audit.get("layer_c_echo_clipping"))
        layer_d = _as_mapping(audit.get("layer_d_facts"))
        d_guard = _as_mapping(_as_mapping(layer_d.get("guard")).get("mic_related"))
        e = _as_mapping(audit.get("layer_e_channel_clipping"))
        f = _as_mapping(audit.get("layer_f_observation_clipping"))
        final_guard = _as_mapping(audit.get("final_guard"))
        co = _as_mapping(audit.get("cooccurrence"))
        flags = {
            "c_event": c.get("event"),
            "c_actual": c.get("actual_changed"),
            "d_guard_applied": d_guard.get("applied"),
            "e_event": e.get("event"),
            "e_actual": e.get("actual_changed"),
            "f_event": f.get("event"),
            "f_actual": f.get("actual_changed"),
            "final_guard_applied": final_guard.get("applied"),
            "c_and_e_actual": co.get("c_and_e_actual"),
            "c_and_f_actual": co.get("c_and_f_actual"),
            "e_and_f_actual": co.get("e_and_f_actual"),
            "c_e_f_actual": co.get("c_e_f_actual"),
        }
        for key, value in flags.items():
            counts[key] += int(bool(value))
        for key in realized_metrics:
            value = _float_fact(f, key)
            if value is not None:
                realized_metrics[key].append(value)

    return {
        "num_samples": total,
        "counts": counts,
        "rates": {
            key: (float(value) / total if total else None)
            for key, value in counts.items()
        },
        "layer_f_realized_means": {
            key: (float(np.mean(values)) if values else None)
            for key, values in realized_metrics.items()
        },
    }


def _finite_full_rms_ratio_db(
    numerator: Any, denominator: Any
) -> tuple[float | None, str | None]:
    """计算 Layer-D 全波形 RMS 比；静音或缺失时保留 N/A 原因。"""
    if numerator is None or denominator is None:
        return None, "layer_d_stem_unavailable"
    num = np.asarray(numerator, dtype=np.float64).ravel()
    den = np.asarray(denominator, dtype=np.float64).ravel()
    length = min(num.size, den.size)
    if length <= 0:
        return None, "layer_d_stem_unavailable"
    num_power = float(np.mean(num[:length] ** 2))
    den_power = float(np.mean(den[:length] ** 2))
    if not np.isfinite(num_power) or not np.isfinite(den_power):
        return None, "layer_d_power_non_finite"
    if num_power <= 1e-12 or den_power <= 1e-12:
        return None, "layer_d_stem_silent"
    return 10.0 * float(np.log10(num_power / den_power)), None


def _audit_measurement_points(
    stems: dict[str, Any], by_layer: dict[str, Any], *, task: str = "joint"
) -> dict[str, Any]:
    """按新 schema 提取互不混用的审计测量点与 Layer C-0b source-volume calibration（post-RIR/pre-noise-echo，以 early RIR reference 为基准）。"""
    layer_c = by_layer.get("C") if isinstance(by_layer.get("C"), dict) else {}
    audit = (
        layer_c.get("acoustic_audit")
        if isinstance(layer_c.get("acoustic_audit"), dict)
        else {}
    )
    calibration = (
        audit.get("calibration")
        if isinstance(audit.get("calibration"), dict)
        else {}
    )
    colored = (
        calibration.get("colored_noise")
        if isinstance(calibration.get("colored_noise"), dict)
        else None
    )
    source_volume = layer_c.get("source_volume_calibration")
    post_f = stems.get("realized")
    final = stems.get("realized_final")
    frontend_facts = stems.get("aec_frontend_facts")
    if frontend_facts is not None and not isinstance(frontend_facts, dict):
        raise RuntimeError("aec_frontend_facts must be a mapping")

    layer_d_clean = stems.get("layer_d_clean")
    layer_d_noise = stems.get("layer_d_noise")
    physical_nearend = layer_d_clean
    physical_reason = None
    if str(task).strip().lower() == "aec":
        if layer_d_clean is None or layer_d_noise is None:
            physical_nearend = None
            physical_reason = "aec_physical_nearend_stem_unavailable"
        else:
            clean_arr = np.asarray(layer_d_clean, dtype=np.float32).ravel()
            noise_arr = np.asarray(layer_d_noise, dtype=np.float32).ravel()
            length = min(clean_arr.size, noise_arr.size)
            physical_nearend = (clean_arr[:length] - noise_arr[:length]).astype(
                np.float32, copy=False
            )

    if physical_reason is None:
        snr_db, snr_reason = _finite_full_rms_ratio_db(
            physical_nearend, layer_d_noise
        )
        ser_db, ser_reason = _finite_full_rms_ratio_db(
            physical_nearend, stems.get("layer_d_echo")
        )
    else:
        snr_db = ser_db = None
        snr_reason = ser_reason = physical_reason
    return {
        "layer_d_full_rms": {
            "snr_db": snr_db,
            "snr_reason_code": snr_reason,
            "ser_db": ser_db,
            "ser_reason_code": ser_reason,
        },
        "colored_calibration": dict(colored) if colored is not None else None,
        "layer_c_aggregate": {
            "snr_db": calibration.get("layer_c_aggregate_realized_snr_db"),
            "reason_code": calibration.get("layer_c_aggregate_reason_code"),
        },
        "source_volume_calibration": (
            dict(source_volume) if isinstance(source_volume, dict) else None
        ),
        "post_f_realized": dict(post_f) if isinstance(post_f, dict) else None,
        "realized_final": dict(final) if isinstance(final, dict) else None,
        "aec_frontend_facts": dict(frontend_facts) if isinstance(frontend_facts, dict) else None,
    }


def _export_one_scenario_sample(
    ds: Any,
    d_out: Path,
    dataset_index: int,
    branch: str,
    zh: str,
    export_stems_subdir: bool,
    data_cfg: dict[str, Any],
    simulation_cfg_bounds: dict[str, Any],
) -> dict[str, Any]:
    """导出单个场景单条样本的波形与元数据。

    Args:
        ds: AEC DynamicMixingDataset 实例。
        d_out: 该样本的输出目录。
        dataset_index: 数据集下标。
        branch: force_branch 值（如 "prob_target_noise_echo"）。
        zh: 中文场景说明。
        export_stems_subdir: 是否额外导出 stems/ 子目录。
        data_cfg: data 配置块（用于取 sample_rate 等默认值）。

    Returns:
        该样本的 meta dict（会被写入 meta.json）。
    """
    sample, st, branches = _get_validated_forced_branch_sample(
        ds, dataset_index, branch
    )
    nearend_t, mic_t, farend_t, fs_raw, _lengths = _unpack_dataset_sample(sample)
    mic = _to_numpy_f32(mic_t)
    farend = _to_numpy_f32(farend_t)
    nearend = _to_numpy_f32(nearend_t)
    fs = int(fs_raw.item()) if isinstance(fs_raw, torch.Tensor) else int(fs_raw)
    by_layer, aggregate_facts, canonical_stems = canonical_simulation_view(st)
    _validated_layer_d_facts(by_layer)
    task = str(getattr(ds, "task", "joint")).strip().lower()
    measurement_points = _audit_measurement_points(canonical_stems, by_layer, task=task)
    clipping_audit = _observation_clipping_audit(
        by_layer, sample_rate=fs, final_guard=aggregate_facts.get("final_peak_normalize")
    )

    # forced sample 通过全部硬门禁后才允许创建目录或写任何产物。
    d_out.mkdir(parents=True, exist_ok=True)
    _save_wav(d_out / "for_model_mic.wav", mic, fs)
    _save_wav(d_out / "for_model_farend.wav", farend, fs)
    _save_wav(d_out / "for_model_nearend_target.wav", nearend, fs)

    n_mic = int(mic.size)

    # 导出 Layer D 阶段快照
    for stem_key, wav_name in _LAYER_D_WAV_EXPORTS:
        raw = st.get(stem_key)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float32).ravel()
        if arr.size == 0:
            continue
        _save_wav(d_out / wav_name, _align_audio_1d_to_len(arr, n_mic), fs)

    meta: dict[str, Any] = {
        "dataset_index": int(dataset_index),
        "force_branch": branch,
        "task": task,
        "description_zh": zh,
        "sample_rate": fs,
        "last_run_branch": branches.get("dataset_actual"),
        "simulation_by_layer": json_safe(by_layer),
        "realized": json_safe(aggregate_facts.get("realized")),
        "realized_final": json_safe(st.get("realized_final")),
        "aec_frontend_facts": json_safe(st.get("aec_frontend_facts")),
        "audit_measurement_points": json_safe(measurement_points),
        "clipping_audit": json_safe(clipping_audit),
        "rms": {
            "mic_for_model": _rms(mic),
            "farend_for_model": _rms(farend),
            "nearend_target_for_model": _rms(nearend),
            "layer_d_noisy": _rms(_align_audio_1d_to_len(st.get("layer_d_noisy"), n_mic)),
            "layer_d_clean": _rms(_align_audio_1d_to_len(st.get("layer_d_clean"), n_mic)),
            "layer_d_noise": _rms(_align_audio_1d_to_len(st.get("layer_d_noise"), n_mic)),
            "layer_d_echo": _rms(_align_audio_1d_to_len(st.get("layer_d_echo"), n_mic)),
            "farend_ref": _rms(_align_audio_1d_to_len(st.get("farend_ref"), n_mic)),
        },
        "max_abs": {
            "mic": float(np.max(np.abs(mic)) if mic.size else 0.0),
            "farend": float(np.max(np.abs(farend)) if farend.size else 0.0),
            "nearend": float(np.max(np.abs(nearend)) if nearend.size else 0.0),
        },
    }
    meta["simulation_stems_summary"] = _stems_dict(st)

    # 不同测量点保持独立：Layer-D 是 full-RMS，Layer-C aggregate 不是单分量。
    validation: dict[str, Any] = {
        "scenario_match": all(value == branch for value in branches.values()),
        "branches": json_safe(branches),
        "checks": [],
        "measurement_points": json_safe(measurement_points),
    }

    ld_clean = st.get("layer_d_clean")
    ld_noisy = st.get("layer_d_noisy")
    ld_noise = st.get("layer_d_noise")
    ld_echo = st.get("layer_d_echo")
    ld_farend = st.get("farend_ref")

    # 活跃段音量（dBFS）
    if ld_clean is not None:
        validation["clean_active_dbfs"] = round(_to_dbfs(_compute_active_rms(ld_clean)), 2)
    if ld_noisy is not None:
        validation["noisy_active_dbfs"] = round(_to_dbfs(_compute_active_rms(ld_noisy)), 2)
    if ld_echo is not None:
        validation["echo_active_dbfs"] = round(_to_dbfs(_compute_active_rms(ld_echo)), 2)
    if ld_farend is not None:
        validation["farend_active_dbfs"] = round(_to_dbfs(_compute_active_rms(ld_farend)), 2)

    layer_d_full_rms = measurement_points["layer_d_full_rms"]
    validation["snr_actual_db"] = layer_d_full_rms["snr_db"]
    validation["snr_reason_code"] = layer_d_full_rms["snr_reason_code"]
    validation["ser_actual_db"] = layer_d_full_rms["ser_db"]
    validation["ser_reason_code"] = layer_d_full_rms["ser_reason_code"]

    # SER 对应单一 echo stem；aggregate SNR 不与单分量配置范围比较。
    ser_actual = layer_d_full_rms["ser_db"]
    ser_bounds = simulation_cfg_bounds.get("ser_effective_bounds_db")
    if ser_actual is not None and ser_bounds is not None:
        validation["ser_expected_range"] = [float(ser_bounds[0]), float(ser_bounds[1])]
        if not (ser_bounds[0] - 5 <= ser_actual <= ser_bounds[1] + 5):
            validation["checks"].append(
                f"Layer-D full-RMS SER {ser_actual:.1f} dB outside expected range {ser_bounds}"
            )

    # 场景语义验证
    if branch == "prob_target_only":
        if nearend.size and mic.size:
            diff = float(np.max(np.abs(nearend - mic)))
            validation["prob_target_only_max_diff"] = round(diff, 6)
            if diff > 1e-4:
                validation["checks"].append(f"prob_target_only: mic != nearend, max_diff={diff:.2e}")
    elif branch == "prob_noise_only":
        ne_db = _to_dbfs(_compute_active_rms(nearend))
        validation["nearend_target_dbfs"] = round(ne_db, 2)
        if task == "aec":
            t = min(nearend.size, mic.size)
            diff = float(np.max(np.abs(nearend[:t] - mic[:t]))) if t else 0.0
            validation["noise_only_target_mic_max_diff"] = round(diff, 6)
            if diff > 1e-4:
                validation["checks"].append(
                    f"noise_only task=aec: mic != passthrough-noise target, max_diff={diff:.2e}"
                )
        elif ne_db > -40:
            validation["checks"].append(
                f"noise_only: nearend target energy {ne_db:.1f} dBFS > -40, expected near silence"
            )
    elif branch == "prob_echo_only":
        ne_db = _to_dbfs(_compute_active_rms(nearend))
        validation["nearend_target_dbfs"] = round(ne_db, 2)
        if ne_db > -40:
            validation["checks"].append(
                f"echo_only: nearend target energy {ne_db:.1f} dBFS > -40, expected near silence"
            )
    elif branch in ("prob_target_echo", "prob_target_noise_echo"):
        if ld_echo is not None:
            echo_rms = _rms(ld_echo)
            mic_rms = _rms(mic)
            validation["echo_to_mic_ratio_db"] = round(
                20 * np.log10(max(echo_rms, 1e-12) / max(mic_rms, 1e-12)), 2
            )
        # Critical: farend must not be empty in double-talk branches
        farend_max = float(np.max(np.abs(farend)) if farend.size else 0.0)
        validation["farend_max_abs"] = round(farend_max, 6)
        if farend_max < 1e-6:
            validation["checks"].append(
                f"CRITICAL: farend is empty (max_abs={farend_max:.2e}) in {branch} branch"
            )
        # Also verify echo is present in mic (mic should differ from nearend)
        if nearend.size and mic.size:
            t = min(nearend.size, mic.size)
            diff_rms = np.sqrt(np.mean((mic[:t] - nearend[:t]) ** 2))
            validation["mic_minus_nearend_rms"] = round(float(diff_rms), 6)
            if diff_rms < 1e-4:
                validation["checks"].append(
                    f"WARNING: mic ≈ nearend (diff_rms={diff_rms:.2e}), echo may be missing"
                )

    meta["validation"] = validation

    # ------------------------------------------------------------------
    # 试听辅助：生成立体声对比文件（左耳 = 参考，右耳 = 观测/干扰）
    # ------------------------------------------------------------------
    # 左 = 训练 target（joint 为 clean，aec 可含透传噪声），右 = mic。
    stereo_cn = _make_stereo_left_right(nearend, mic)
    _save_wav(d_out / "for_listen_cleanL_noisyR.wav", stereo_cn.T, fs)
    # 左 = farend, 右 = echo —— 含回声场景可用
    if ld_farend is not None and ld_echo is not None:
        stereo_fe = _make_stereo_left_right(ld_farend, ld_echo)
        _save_wav(d_out / "for_listen_farendL_echoR.wav", stereo_fe.T, fs)

    # 可选：导出完整 stems 子目录
    if export_stems_subdir and st:
        st2 = st
        stem_dir = d_out / "stems"
        stem_dir.mkdir(exist_ok=True)
        if st2.get("final_noisy") is not None:
            _save_wav(stem_dir / "final_noisy_for_model.wav", st2["final_noisy"], fs)
        if st2.get("final_clean") is not None:
            _save_wav(stem_dir / "final_nearend_target_for_model.wav", st2["final_clean"], fs)
        for stem_key, wav_name in _LAYER_D_WAV_EXPORTS:
            if st2.get(stem_key) is not None:
                _save_wav(
                    stem_dir / wav_name,
                    _align_audio_1d_to_len(st2[stem_key], n_mic),
                    fs,
                )
        sbl = st2.get("simulation_by_layer")
        if sbl:
            (stem_dir / "simulation_by_layer.json").write_text(
                json.dumps(
                    _json_finite(sbl), indent=2, ensure_ascii=False, allow_nan=False
                ),
                encoding="utf-8",
            )

    (d_out / "meta.json").write_text(
        json.dumps(_json_finite(meta), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Export AEC simulation waveforms per force_branch scenario."
    )
    p.add_argument("-C", "--config", required=True, help="训练用 YAML（含 data.mode=dynamic_aec）")
    p.add_argument("-o", "--output-dir", required=True, help="输出根目录")
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="起始数据集样本下标；与训练同一清单顺序；第 k 条使用下标 index+k",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="每个场景导出的条数（连续占用 index .. index+num_samples-1）",
    )
    p.add_argument(
        "--export-stems",
        action="store_true",
        help="在各条样本子目录下额外写 stems/（final_*、layer_D_*、simulation_by_layer.json）",
    )
    args = p.parse_args(argv)

    num_samples = int(args.num_samples)
    if num_samples < 1:
        logger.error("--num-samples must be >= 1")
        return 1
    idx0 = int(args.index)
    if idx0 < 0:
        logger.error("--index must be >= 0")
        return 1

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        logger.error("Config not found: %s", cfg_path)
        return 1
    out_root = Path(args.output_dir).resolve()

    from core.utils.config import load_merged_config

    cfg = _prepare_verification_config(load_merged_config(str(cfg_path)))
    data_cfg = cfg.get("data") or {}

    ds = build_dynamic_aec_dataset(cfg, "train")
    base_seed = int((cfg.get("meta") or {}).get("seed", 0))
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(0, base_seed=base_seed)

    dataset_len = len(ds)
    if idx0 >= dataset_len:
        logger.error("index=%s out of range [0, %s)", idx0, dataset_len)
        return 1
    last_idx = idx0 + num_samples - 1
    if last_idx >= dataset_len:
        logger.error(
            "index=%s with num_samples=%s needs indices up to %s, but dataset len=%s",
            idx0,
            num_samples,
            last_idx,
            dataset_len,
        )
        return 1

    out_root.mkdir(parents=True, exist_ok=True)
    simulation_cfg_bounds = _curated_simulation_cfg_bounds(ds)
    (out_root / "simulation_cfg_bounds.json").write_text(
        json.dumps(
            _json_finite(simulation_cfg_bounds),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "config": str(cfg_path),
        "index_start": idx0,
        "num_samples_per_scenario": num_samples,
        "base_seed": base_seed,
        "task": str(getattr(ds, "task", "joint")).strip().lower(),
        "dataset_len": len(ds),
        "export_simulation_stems": True,
        "export_stems_subdir": bool(args.export_stems),
        "simulation_cfg_bounds": simulation_cfg_bounds,
        "scenarios": [],
    }

    all_clipping_audits: list[dict[str, Any]] = []
    readme_lines = [
        "AEC DynamicMixingDataset 场景导出",
        f"config: {cfg_path}",
        f"起始 index: {idx0} | 每场景条数: {num_samples} | 下标范围 [{idx0}, {idx0 + num_samples - 1}]",
        f"base_seed: {base_seed}",
        f"task: {summary['task']}",
        "",
        "目录结构：各场景 / sample_XXX_idxY / 下为 wav 与 meta.json。",
        "meta.json 字段：",
        "  - simulation_by_layer['D']：固定 stage_order、normalize 与 AEC mic_related/farend 双 guard 的唯一聚合 facts；仅允许 A-F。",
        "根目录 simulation_cfg_bounds.json：YAML 边界快照（可选对照）。",
        "",
        "for_model_*.wav：与 collate 后送模型的 mic、nearend(target)、farend 一致。",
        "layer_D_*.wav：Layer D（post-mix amplitude conditioning）按 D-normalize → D-guard 固定顺序执行后、Layer E 前的快照。",
        "Dataset 仿真顺序：pre-finite → D-normalize → D-guard → E → F → task-post-EF → final-guard → post-finite。",
        "AECDeploymentFrontend facts 单独记录；Trainer/Inference 在 STFT 前运行，不属于 simulation_by_layer。",
        "for_listen_cleanL_noisyR.wav：立体声，左耳=训练 target（aec 可含透传噪声），右耳=mic。",
        "for_listen_farendL_echoR.wav：立体声，左耳=farend，右耳=echo，验证回声延迟与 SER。",
        "stems/ 仅当 --export-stems：final_* 与 layer_D_* 副本及 simulation_by_layer.json。",
        "",
        "验证项（meta.json -> validation）：",
        "  - scenario_match/branches：requested、Dataset actual、Layer B、post-F 与 realized_final 必须一致。",
        "  - measurement_points：区分 Layer-D full-RMS、colored calibration、Layer-C aggregate、post-F realized 与 realized_final。",
        "  - clipping_audit：分别报告 C-aec、E channel clipping、F observation clipping 的 event/actual changed；F rail 命中 count/ratio 直接取 runtime，连续命中 ms 由 samples/sample_rate 推导，旧 sidecar 缺字段为 None。",
        "  - stems/schema/finite/non-placeholder 为写文件前硬门禁；不可分解指标保留 None/reason_code。",
        "  - checks：仅适用的单分量指标超出范围时列出告警；aggregate 不与单分量范围比较。",
        "",
    ]

    def _fmt_dbfs(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.1f}"

    for subdir, branch, zh in SCENARIOS:
        scenario_entry: dict[str, Any] = {
            "subdir": subdir,
            "branch": branch,
            "samples": [],
        }
        d_scenario = out_root / subdir
        for k in range(num_samples):
            dataset_idx = idx0 + k
            sample_dir_name = f"sample_{k:03d}_idx{dataset_idx}"
            d_out = d_scenario / sample_dir_name
            meta = _export_one_scenario_sample(
                ds,
                d_out,
                dataset_idx,
                branch,
                zh,
                bool(args.export_stems),
                data_cfg,
                simulation_cfg_bounds,
            )
            scenario_entry["samples"].append({"dir": sample_dir_name, "meta": meta})
            clipping = meta.get("clipping_audit") or {}
            all_clipping_audits.append(clipping)
            readme_lines.append(f"[{subdir}/{sample_dir_name}] {branch} index={dataset_idx}")
            points = meta.get("audit_measurement_points") or {}
            layer_d = points.get("layer_d_full_rms") or {}
            aggregate = points.get("layer_c_aggregate") or {}
            colored = points.get("colored_calibration") or {}
            source_volume = points.get("source_volume_calibration") or {}
            post_f = points.get("post_f_realized") or {}
            final = points.get("realized_final") or {}
            readme_lines.append(
                f"  Layer C aggregate SNR={aggregate.get('snr_db')} "
                f"reason={aggregate.get('reason_code')} | colored={colored.get('color')} "
                f"colored calibration SNR={colored.get('calibration_realized_snr_db')}"
            )
            readme_lines.append(
                f"  source volume: requested={source_volume.get('target_requested_db')} "
                f"realized={source_volume.get('realized_after_db')} "
                f"gain={source_volume.get('gain')} clipped={source_volume.get('gain_clipped')} "
                f"status={source_volume.get('status')} reason={source_volume.get('reason')}"
            )
            layer_d_facts = clipping.get("layer_d_facts") or {}
            d_normalize = layer_d_facts.get("normalize") or {}
            d_guards = layer_d_facts.get("guard") or {}
            mic_related_guard = d_guards.get("mic_related") or {}
            farend_guard = d_guards.get("farend") or {}
            final_guard = clipping.get("final_guard") or {}
            readme_lines.append(
                f"  D-normalize: mode={d_normalize.get('mode')} "
                f"target_peak={d_normalize.get('target_peak')} "
                f"scale={d_normalize.get('scale')}"
            )
            readme_lines.append(
                f"  D-guard mic_related: applied={mic_related_guard.get('applied')} "
                f"scale={mic_related_guard.get('scale')} | "
                f"farend: applied={farend_guard.get('applied')} "
                f"scale={farend_guard.get('scale')} | "
                f"full-RMS SNR={layer_d.get('snr_db')} SER={layer_d.get('ser_db')}"
            )
            readme_lines.append(
                f"  final-guard: applied={final_guard.get('applied')} "
                f"scale={final_guard.get('scale')}"
            )
            f_clip = clipping.get("layer_f_observation_clipping") or {}
            co = clipping.get("cooccurrence") or {}
            readme_lines.append(
                f"  Layer F observation clip: event={f_clip.get('event')} changed={f_clip.get('actual_changed')} "
                f"mode={f_clip.get('mode')} rail={f_clip.get('rail')} pre_drive={f_clip.get('pre_drive')} "
                f"pre_peak={f_clip.get('peak_after_pre_drive')} post_peak={f_clip.get('peak_after')} "
                f"rail_sample_count={f_clip.get('clipped_sample_count')} "
                f"rail_sample_ratio={f_clip.get('clipped_sample_ratio')} "
                f"rail_frame_count={f_clip.get('clipped_frame_count')} "
                f"rail_frame_ratio={f_clip.get('clipped_frame_ratio')} "
                f"longest_rail_run_samples={f_clip.get('longest_clipped_run_samples')} "
                f"longest_rail_run_ms={f_clip.get('longest_clipped_run_ms')}"
            )
            readme_lines.append(
                f"  clipping cooccurrence: C+E={co.get('c_and_e_actual')} "
                f"C+F={co.get('c_and_f_actual')} E+F={co.get('e_and_f_actual')} "
                f"C+E+F={co.get('c_e_f_actual')}"
            )
            readme_lines.append(
                f"  post-F realized: SNR={post_f.get('snr_db')} SER={post_f.get('ser_db')} "
                f"decomposable={post_f.get('decomposable')}"
            )
            readme_lines.append(
                f"  realized_final: SNR={final.get('snr_db')} SER={final.get('ser_db')} "
                f"decomposable={final.get('decomposable')}"
            )
            readme_lines.append(
                f"  RMS mic={meta['rms']['mic_for_model']:.6f} | farend={meta['rms']['farend_for_model']:.6f} | "
                f"nearend(target)={meta['rms']['nearend_target_for_model']:.6f}"
            )
            val = meta.get("validation") or {}
            if val.get("clean_active_dbfs") is not None:
                readme_lines.append(
                    f"  Active dBFS clean={_fmt_dbfs(val.get('clean_active_dbfs'))} "
                    f"noisy={_fmt_dbfs(val.get('noisy_active_dbfs'))} "
                    f"echo={_fmt_dbfs(val.get('echo_active_dbfs'))}"
                )
            checks = val.get("checks", [])
            if checks:
                for chk in checks:
                    readme_lines.append(f"  [CHECK] {chk}")
            else:
                readme_lines.append("  [OK] All checks passed")
            readme_lines.append("")

        scenario_audits = [
            item["meta"].get("clipping_audit", {})
            for item in scenario_entry["samples"]
        ]
        scenario_entry["clipping_summary"] = _aggregate_clipping_audits(
            scenario_audits
        )
        summary["scenarios"].append(scenario_entry)
        readme_lines.append("")

    summary["clipping_summary"] = _aggregate_clipping_audits(all_clipping_audits)
    readme_lines.append(
        "全样本 clipping 汇总："
        + json.dumps(
            _json_finite(summary["clipping_summary"]),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    (out_root / "README.txt").write_text("\n".join(readme_lines), encoding="utf-8")
    (out_root / "summary.json").write_text(
        json.dumps(
            _json_finite(summary), indent=2, ensure_ascii=False, allow_nan=False
        ),
        encoding="utf-8",
    )
    logger.info(
        "Wrote %d scenarios x %d samples under %s",
        len(SCENARIOS),
        num_samples,
        out_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
