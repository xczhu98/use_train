#!/usr/bin/env python3
"""Deep-check AEC simulation pipeline for energy/SNR/SER/consistency/peak anomalies.

验证副本强制 ``strict_load=True`` 并导出 stems；任何 tuple/schema/finite、静音
placeholder 或 forced-branch 审计不一致都会在报告写出前 fail-closed。
报告写出后，``anomalies`` 非空即返回非零退出码（报告保留，供定位问题）。

Usage:
    python deep_check_aec_simulation.py -C configs/aec/microsoft_challenge/cagcrn/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml \
        --num-samples 50 --output report.json
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tasks.aec.data.dynamic_aec_build import build_dynamic_aec_dataset
from core.data.simulation.scene_names import AEC_SCENE_ORDER, AEC_ECHO_SCENES
from core.data.simulation_layer_view import canonical_simulation_view


def rms_db(x: np.ndarray) -> float:
    rms = math.sqrt(float(np.mean(x.astype(np.float64) ** 2)) + 1e-15)
    return 20.0 * math.log10(rms)


_SUPPORTED_AEC_TUPLE_LENGTHS = (5, 6, 7, 10, 11, 12, 13)
_REQUIRED_SIMULATION_LAYERS = ("A", "B", "C", "D", "E", "F")
_REQUIRED_STEM_ARRAYS = (
    "layer_d_noisy",
    "layer_d_clean",
    "layer_d_noise",
    "layer_d_echo",
    "farend_ref",
    "final_noise",
    "final_clean",
    "final_noisy",
)
_ECHO_BRANCHES = AEC_ECHO_SCENES
_TRACE_BRANCH_MISSING = object()


def _unpack_dataset_sample(sample):
    """消费 AEC dataset tuple 的稳定前五项，显式拒绝歧义长度。"""
    length = len(sample) if isinstance(sample, tuple) else "N/A"
    if not isinstance(sample, tuple) or length not in _SUPPORTED_AEC_TUPLE_LENGTHS:
        raise RuntimeError(
            "expected AEC dataset tuple with length 5/6/7/10/11/12/13, "
            f"got {type(sample).__name__} with len={length}"
        )
    return sample[0], sample[1], sample[2], sample[3], sample[4]


def _prepare_verification_config(cfg):
    """复制 merged config，并只强化实际生效的 dynamic section。"""
    verified = copy.deepcopy(cfg)
    data_cfg = verified.get("data")
    if not isinstance(data_cfg, dict):
        data_cfg = {}
        verified["data"] = data_cfg

    # resolve_strict_load 扫描 data 与根 config（data 优先），两层都显式强化。
    data_cfg["strict_load"] = True
    verified["strict_load"] = True

    section = data_cfg.get("train")
    section = dict(section) if isinstance(section, dict) else {}
    section["export_simulation_stems"] = True
    data_cfg["train"] = section
    return verified


def _scalar_int(value, field: str, *, index: int) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(
            f"AEC forced sample idx={index} {field} must be scalar, shape={array.shape}"
        )
    try:
        scalar = float(array.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"AEC forced sample idx={index} {field} must be a finite integer"
        ) from exc
    if not math.isfinite(scalar) or not scalar.is_integer():
        raise RuntimeError(
            f"AEC forced sample idx={index} {field} must be a finite integer, got {scalar!r}"
        )
    return int(scalar)


def _require_branch(value, expected: str, field: str, index: int) -> None:
    if value != expected:
        raise RuntimeError(
            f"AEC forced sample idx={index} {field}={value!r}, "
            f"expected branch={expected!r}"
        )


def _validate_forced_branch_sample(
    ds,
    index: int,
    branch: str,
    sample,
    stems,
    *,
    actual_branch=_TRACE_BRANCH_MISSING,
):
    """对验证样本执行 branch、schema、finite 与 placeholder 硬门禁。"""
    near_raw, mic_raw, far_raw, fs_raw, length_raw = _unpack_dataset_sample(sample)
    fs = _scalar_int(fs_raw, "fs", index=index)
    length = _scalar_int(length_raw, "lengths", index=index)
    if fs <= 0 or length <= 0:
        raise RuntimeError(
            f"AEC forced sample idx={index} requires fs>0 and lengths>0, "
            f"got fs={fs}, lengths={length}"
        )

    waveforms = {}
    for name, raw in (("nearend", near_raw), ("mic", mic_raw), ("farend", far_raw)):
        arr = _to_arr(raw)
        if arr is None or arr.size < length:
            size = None if arr is None else arr.size
            raise RuntimeError(
                f"AEC forced sample idx={index} {name} size={size}, expected>={length}"
            )
        arr = arr[:length]
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"AEC forced sample idx={index} {name} contains NaN/Inf")
        waveforms[name] = arr
    if not any(np.any(arr != 0.0) for arr in waveforms.values()):
        raise RuntimeError(
            f"AEC forced sample idx={index} branch={branch} is a silence placeholder"
        )

    if not isinstance(stems, dict):
        raise RuntimeError(f"AEC forced sample idx={index} stems missing or invalid")
    if not isinstance(stems.get("simulation_by_layer"), dict):
        raise RuntimeError(f"AEC forced sample idx={index} missing simulation_by_layer")
    # 统一 canonical 视图：A–F 层与聚合事实（内嵌优先、顶层回落），未知键 fail-closed。
    by_layer, aggregate_facts, _ = canonical_simulation_view(stems)
    missing_layers = [
        name for name in _REQUIRED_SIMULATION_LAYERS
        if not isinstance(by_layer.get(name), dict)
    ]
    if missing_layers:
        raise RuntimeError(
            f"AEC forced sample idx={index} simulation schema missing={missing_layers}"
        )
    realized_final = stems.get("realized_final")
    if not isinstance(realized_final, dict):
        raise RuntimeError(f"AEC forced sample idx={index} missing realized_final")
    realized = aggregate_facts.get("realized")
    if not isinstance(realized, dict):
        raise RuntimeError(f"AEC forced sample idx={index} missing realized")

    layer_b = by_layer["B"]
    branches = {
        "requested": branch,
        "dataset_actual": (
            actual_branch
            if actual_branch is not _TRACE_BRANCH_MISSING
            else getattr(ds, "_last_run_branch", None)
        ),
        "layer_b_force": layer_b.get("force_branch"),
        "layer_b_actual": layer_b.get("branch"),
        "post_f_realized": realized.get("branch"),
        "realized_final": realized_final.get("branch"),
    }
    for field, value in branches.items():
        if field != "requested":
            _require_branch(value, branch, field, index)

    for stem_name in _REQUIRED_STEM_ARRAYS:
        stem = stems.get(stem_name)
        if not isinstance(stem, np.ndarray) or stem.size < length:
            size = stem.size if isinstance(stem, np.ndarray) else None
            raise RuntimeError(
                f"AEC forced sample idx={index} stem {stem_name} size={size}, "
                f"expected ndarray size>={length}"
            )
        if not np.all(np.isfinite(stem.ravel()[:length])):
            raise RuntimeError(
                f"AEC forced sample idx={index} stem {stem_name} contains NaN/Inf"
            )

    final_echo = stems.get("final_echo")
    if branch in _ECHO_BRANCHES:
        if not isinstance(final_echo, np.ndarray) or final_echo.size < length:
            raise RuntimeError(
                f"AEC forced sample idx={index} branch={branch} missing valid final_echo"
            )
        final_echo = final_echo.ravel()[:length]
        if not np.all(np.isfinite(final_echo)) or not np.any(final_echo != 0.0):
            raise RuntimeError(
                f"AEC forced sample idx={index} branch={branch} final_echo is invalid"
            )
    elif final_echo is not None:
        raise RuntimeError(
            f"AEC forced sample idx={index} branch={branch} contains stale final_echo"
        )
    return branches


def _get_validated_forced_branch_sample(ds, index: int, branch: str):
    """经原子 Dataset seam 取样并执行硬门禁。"""
    get_traced = getattr(ds, "get_sample_with_trace", None)
    if callable(get_traced):
        traced = get_traced(index, force_branch=branch, capture_stems=True)
        sample = traced.sample
        stems = traced.trace.legacy_stems("scene")
        actual_branch = traced.trace.metadata.get("branch")
    else:
        ds._last_simulation_stems = None
        ds._last_run_branch = None
        sample = ds._getitem_simulation_verify(index, force_branch=branch)
        stems = getattr(ds, "_last_simulation_stems", None)
        actual_branch = getattr(ds, "_last_run_branch", None)
    branches = _validate_forced_branch_sample(
        ds,
        index,
        branch,
        sample,
        stems,
        actual_branch=actual_branch,
    )
    return sample, stems, branches


def _get_forced_branch_sample(ds, index: int, branch: str):
    """兼容旧调用方：返回已通过硬门禁的 Dataset sample。"""
    sample, _stems, _branches = _get_validated_forced_branch_sample(ds, index, branch)
    return sample


def _to_arr(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).ravel()


def _full_rms_ratio_db(numerator, denominator):
    """Layer-D 全波形 RMS 比；缺失、静音或非有限时返回 None/reason。"""
    num = _to_arr(numerator)
    den = _to_arr(denominator)
    if num is None or den is None:
        return None, "layer_d_stem_unavailable"
    length = min(num.size, den.size)
    if length <= 0:
        return None, "layer_d_stem_unavailable"
    num_power = float(np.mean(num[:length] ** 2))
    den_power = float(np.mean(den[:length] ** 2))
    if not np.isfinite(num_power) or not np.isfinite(den_power):
        return None, "layer_d_power_non_finite"
    if num_power <= 1e-12 or den_power <= 1e-12:
        return None, "layer_d_stem_silent"
    return 10.0 * math.log10(num_power / den_power), None


def _canonical_audit_record(stems, branch, *, task="joint"):
    """从 ``simulation_by_layer.C.acoustic_audit`` 与 realized 节点读取审计。"""
    by_layer = stems.get("simulation_by_layer", {}) if isinstance(stems, dict) else {}
    layer_c = by_layer.get("C", {}) if isinstance(by_layer.get("C"), dict) else {}
    audit = (
        layer_c.get("acoustic_audit", {})
        if isinstance(layer_c.get("acoustic_audit"), dict)
        else {}
    )
    calibration = (
        audit.get("calibration", {})
        if isinstance(audit.get("calibration"), dict)
        else {}
    )
    colored = (
        calibration.get("colored_noise")
        if isinstance(calibration.get("colored_noise"), dict)
        else None
    )
    c0 = audit.get("c0_full_early", {})
    c0 = c0 if isinstance(c0, dict) else {}
    source_volume = layer_c.get("source_volume_calibration", {})
    source_volume = source_volume if isinstance(source_volume, dict) else {}
    post_f = stems.get("realized", {}) if isinstance(stems, dict) else {}
    final = stems.get("realized_final", {}) if isinstance(stems, dict) else {}

    layer_d_clean = stems.get("layer_d_clean")
    layer_d_noise = stems.get("layer_d_noise")
    physical_nearend = layer_d_clean
    physical_reason = None
    if str(task).strip().lower() == "aec":
        clean_arr = _to_arr(layer_d_clean)
        noise_arr = _to_arr(layer_d_noise)
        if clean_arr is None or noise_arr is None:
            physical_nearend = None
            physical_reason = "aec_physical_nearend_stem_unavailable"
        else:
            length = min(clean_arr.size, noise_arr.size)
            physical_nearend = clean_arr[:length] - noise_arr[:length]

    if physical_reason is None:
        snr_db, snr_reason = _full_rms_ratio_db(physical_nearend, layer_d_noise)
        ser_db, ser_reason = _full_rms_ratio_db(
            physical_nearend, stems.get("layer_d_echo")
        )
    else:
        snr_db = ser_db = None
        snr_reason = ser_reason = physical_reason

    record = {
        "branch": branch,
        "task": str(task).strip().lower(),
        "layer_d_full_rms_snr_db": snr_db,
        "layer_d_full_rms_snr_reason_code": snr_reason,
        "layer_d_full_rms_ser_db": ser_db,
        "layer_d_full_rms_ser_reason_code": ser_reason,
        "c0_full_to_early_rms_db": c0.get("full_to_early_rms_db"),
        "c0_full_to_early_reason_code": c0.get("reason_code"),
        "colored_noise_color": colored.get("color") if colored else None,
        "colored_calibration_snr_db": (
            colored.get("calibration_realized_snr_db") if colored else None
        ),
        "colored_calibration_reason_code": (
            None if colored else "colored_noise_not_applied"
        ),
        "layer_c_aggregate_snr_db": calibration.get(
            "layer_c_aggregate_realized_snr_db"
        ),
        "layer_c_aggregate_reason_code": calibration.get(
            "layer_c_aggregate_reason_code"
        ),
        "source_volume_target_requested_db": source_volume.get(
            "target_requested_db"
        ),
        "source_volume_realized_after_db": source_volume.get("realized_after_db"),
        "source_volume_gain": source_volume.get("gain"),
        "source_volume_gain_clipped": source_volume.get("gain_clipped"),
        "source_volume_status": source_volume.get("status"),
        "source_volume_reason": source_volume.get("reason"),
    }
    for prefix, realized in (("post_f", post_f), ("realized_final", final)):
        realized = realized if isinstance(realized, dict) else {}
        for metric in ("snr", "ser", "erl"):
            record[f"{prefix}_{metric}_db"] = realized.get(f"{metric}_db")
            record[f"{prefix}_{metric}_reason_code"] = realized.get(
                f"{metric}_reason_code"
            )
        record[f"{prefix}_decomposable"] = realized.get("decomposable")
        reasons = realized.get("decomposable_reason_codes")
        record[f"{prefix}_decomposable_reason_codes"] = (
            list(reasons) if isinstance(reasons, (list, tuple)) else []
        )
        residual = realized.get("reconstruction_residual")
        residual = residual if isinstance(residual, dict) else {}
        record[f"{prefix}_reconstruction_rms"] = residual.get("rms")
        record[f"{prefix}_reconstruction_reason_code"] = residual.get("reason_code")
    return record


def _far_db_range_from_merged_config(cfg):
    """从 merged train config 读取显式 far_db_range；缺失或无效时不猜默认值。"""
    data_cfg = cfg.get("data") if isinstance(cfg, dict) else None
    if not isinstance(data_cfg, dict):
        return None
    train_cfg = data_cfg.get("train")
    if not isinstance(train_cfg, dict):
        return None
    simulation_cfg = train_cfg.get("simulation_cfg")
    if not isinstance(simulation_cfg, dict):
        return None
    raw_range = simulation_cfg.get("far_db_range")
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
        return None
    lo = _finite_float_or_none(raw_range[0])
    hi = _finite_float_or_none(raw_range[1])
    if lo is None or hi is None or lo > hi:
        return None
    return lo, hi


def _c3_far_db_range_gate(stems, far_db_range):
    """返回 C-3 nominal farend 电平及其配置范围 gate 适用性。"""
    by_layer = stems.get("simulation_by_layer", {}) if isinstance(stems, dict) else {}
    layer_c = by_layer.get("C", {}) if isinstance(by_layer.get("C"), dict) else {}
    clipping = layer_c.get("c3_echo_clipping", {})
    clipping = clipping if isinstance(clipping, dict) else {}
    if clipping.get("path") == "pre_ser_overload":
        return None, False, "pre_ser_overload_bypasses_ser_erl_coupling"
    if layer_c.get("c3_ser_scaling_used") is not True:
        return None, False, "ser_erl_coupling_not_applied"

    nominal_far_db = _finite_float_or_none(layer_c.get("c3_far_db"))
    if nominal_far_db is None:
        return None, False, "c3_far_db_nominal_unavailable"
    if far_db_range is None:
        return nominal_far_db, False, "far_db_range_unavailable"
    return nominal_far_db, True, None


def _finite_closed_range(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    lo = _finite_float_or_none(value[0])
    hi = _finite_float_or_none(value[1])
    if lo is None or hi is None or lo > hi:
        return None
    return lo, hi


def _c3_erl_range_gate(stems):
    """校验 producer 导出的 ERL configured/physical/effective/selected 合同。"""
    result = {
        "configured_range_db": None,
        "physical_range_db": None,
        "effective_range_db": None,
        "selected_erl_db": None,
        "applicable": False,
        "passed": None,
        "reason_code": None,
    }
    by_layer = stems.get("simulation_by_layer", {}) if isinstance(stems, dict) else {}
    layer_c = by_layer.get("C", {}) if isinstance(by_layer.get("C"), dict) else {}
    clipping = layer_c.get("c3_echo_clipping", {})
    clipping = clipping if isinstance(clipping, dict) else {}
    if clipping.get("path") == "pre_ser_overload":
        result["reason_code"] = "pre_ser_overload_bypasses_ser_erl_coupling"
        return result
    if layer_c.get("c3_ser_scaling_used") is not True:
        result["reason_code"] = "ser_erl_coupling_not_applied"
        return result

    audit = layer_c.get("c3_erl_range_audit")
    if not isinstance(audit, dict):
        result["reason_code"] = "legacy_erl_range_audit_unavailable"
        return result

    configured = _finite_closed_range(audit.get("configured_range_db"))
    physical = _finite_closed_range(audit.get("physical_range_db"))
    effective = _finite_closed_range(audit.get("effective_range_db"))
    selected = _finite_float_or_none(audit.get("selected_erl_db"))
    result.update(
        {
            "configured_range_db": list(configured) if configured else None,
            "physical_range_db": list(physical) if physical else None,
            "effective_range_db": list(effective) if effective else None,
            "selected_erl_db": selected,
            "applicable": True,
            "passed": False,
        }
    )
    if configured is None or physical is None or effective is None or selected is None:
        result["reason_code"] = "erl_range_audit_invalid"
        return result

    expected = (max(configured[0], physical[0]), min(configured[1], physical[1]))
    if expected[0] > expected[1]:
        result["reason_code"] = "erl_range_audit_empty_intersection"
        return result
    if not np.allclose(effective, expected, rtol=0.0, atol=1e-9):
        result["reason_code"] = "erl_effective_range_mismatch"
        return result
    if not all(lo <= selected <= hi for lo, hi in (configured, physical, effective)):
        result["reason_code"] = "erl_selected_outside_range"
        return result

    result["passed"] = True
    return result


def check_branch(
    ds,
    branch: str,
    n_samples: int,
    rng_seed: int = 42,
    *,
    far_db_range=None,
):
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, len(ds), size=n_samples)
    records = []
    anomalies = []

    task = str(getattr(ds, "task", "joint")).strip().lower()
    for idx in indices:
        sample, stems, branches = _get_validated_forced_branch_sample(
            ds, int(idx), branch
        )
        nearend_t, mic_t, farend_t, _fs_raw, _lengths = _unpack_dataset_sample(sample)
        mic = _to_arr(mic_t)
        nearend = _to_arr(nearend_t)
        farend = _to_arr(farend_t)
        rec = {
            "index": int(idx),
            "scenario_match": all(
                value == branch for value in branches.values()
            ),
            "branches": branches,
            **_canonical_audit_record(stems, branch, task=task),
            "mic_rms_db": rms_db(mic),
            "nearend_rms_db": rms_db(nearend),
            "farend_rms_db": rms_db(farend) if np.max(np.abs(farend)) > 1e-10 else None,
            "mic_peak": float(np.max(np.abs(mic))),
            "nearend_peak": float(np.max(np.abs(nearend))),
            "farend_peak": float(np.max(np.abs(farend))),
        }

        layer_d_noisy = _to_arr(stems.get("layer_d_noisy"))
        layer_d_clean = _to_arr(stems.get("layer_d_clean"))
        layer_d_noise = _to_arr(stems.get("layer_d_noise"))
        layer_d_echo = _to_arr(stems.get("layer_d_echo"))
        farend_ref = _to_arr(stems.get("farend_ref"))
        for key, value in (
            ("layer_d_noisy_rms_db", layer_d_noisy),
            ("layer_d_clean_rms_db", layer_d_clean),
            ("layer_d_noise_rms_db", layer_d_noise),
            ("layer_d_echo_rms_db", layer_d_echo),
            ("layer_d_actual_farend_rms_db", farend_ref),
        ):
            rec[key] = (
                rms_db(value)
                if value is not None and np.max(np.abs(value)) > 1e-10
                else None
            )
        # Layer-D actual 可能已受同步 clipping 或 D-guard 改动，仅用于诊断。
        rec["layer_d_actual_farend_rms_gate_role"] = "diagnostic_only"
        rec["farend_ref_rms_db"] = rec["layer_d_actual_farend_rms_db"]

        nominal_far_db, range_applicable, range_reason = _c3_far_db_range_gate(
            stems, far_db_range
        )
        expected_range = (
            [float(far_db_range[0]), float(far_db_range[1])]
            if far_db_range is not None
            else None
        )
        rec["c3_far_db_nominal"] = nominal_far_db
        rec["c3_far_db_expected_range_db"] = expected_range
        rec["c3_far_db_range_applicable"] = range_applicable
        rec["c3_far_db_range_reason_code"] = range_reason
        rec["farend_ref_expected_range_db"] = expected_range
        rec["farend_ref_range_applicable"] = range_applicable
        rec["farend_ref_range_reason_code"] = range_reason
        rec["farend_ref_range_gate_basis"] = "c3_far_db_nominal"

        erl_gate = _c3_erl_range_gate(stems)
        for key, value in erl_gate.items():
            rec[f"c3_erl_range_{key}"] = value
        if erl_gate["applicable"] and not erl_gate["passed"]:
            anomalies.append(
                f"[{branch} idx={idx}] ERL range contract failed: "
                f"{erl_gate['reason_code']}"
            )

        # canonical 元数据判定不可分解时，保留 None/reason，不自行重建波形。
        for prefix in ("post_f", "realized_final"):
            residual = rec.get(f"{prefix}_reconstruction_rms")
            reason = rec.get(f"{prefix}_reconstruction_reason_code")
            if reason is None and residual is not None and residual > 1e-4:
                anomalies.append(
                    f"[{branch} idx={idx}] {prefix} reconstruction_rms={residual:.2e} > 1e-4"
                )

        for name, arr in (("mic", mic), ("nearend", nearend), ("farend", farend)):
            peak = float(np.max(np.abs(arr)))
            if peak > 1.0:
                anomalies.append(f"[{branch} idx={idx}] {name} peak={peak:.4f} > 1.0")

        expected_silent = {
            "prob_target_only": (layer_d_noise, layer_d_echo),
            "prob_noise_only": (
                (layer_d_echo,) if task == "aec" else (layer_d_clean, layer_d_echo)
            ),
            "prob_echo_only": (layer_d_clean, layer_d_noise),
            "prob_noise_echo_only": (() if task == "aec" else (layer_d_clean,)),
        }
        for stem in expected_silent.get(branch, ()):
            if stem is not None and np.max(np.abs(stem)) > 1e-5:
                anomalies.append(
                    f"[{branch} idx={idx}] expected-silent stem peak={np.max(np.abs(stem)):.2e}"
                )

        if range_applicable:
            lo, hi = far_db_range
            if not (lo <= nominal_far_db <= hi):
                anomalies.append(
                    f"[{branch} idx={idx}] nominal c3_far_db={nominal_far_db:.1f} dB "
                    f"outside configured [{lo:g}, {hi:g}]"
                )
        if layer_d_noisy is not None and np.max(np.abs(layer_d_noisy)) < 1e-10:
            anomalies.append(f"[{branch} idx={idx}] layer_d_noisy is all zeros")

        # aggregate SNR 是多分量总体，只做统计，绝不与单分量范围比较。
        records.append(rec)

    return records, anomalies


_AUDIT_VALUE_KEYS = (
    "layer_d_full_rms_snr_db",
    "layer_d_full_rms_ser_db",
    "c3_far_db_nominal",
    "layer_d_actual_farend_rms_db",
    "c0_full_to_early_rms_db",
    "colored_calibration_snr_db",
    "layer_c_aggregate_snr_db",
    "source_volume_target_requested_db",
    "source_volume_realized_after_db",
    "source_volume_gain",
    "post_f_snr_db",
    "post_f_ser_db",
    "post_f_erl_db",
    "post_f_reconstruction_rms",
    "realized_final_snr_db",
    "realized_final_ser_db",
    "realized_final_erl_db",
    "realized_final_reconstruction_rms",
)


def _finite_float_or_none(value):
    """把标量审计值规范化为有限 Python float。"""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _json_finite(value):
    """把 NumPy 类型转成 JSON 原生类型，并把 NaN/Inf 规范化为 null。"""
    if isinstance(value, dict):
        return {str(key): _json_finite(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_finite(value.tolist())
    if isinstance(value, np.generic):
        return _json_finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_stats(values):
    finite = [value for raw in values if (value := _finite_float_or_none(raw)) is not None]
    if not finite:
        return None
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def summarize(records):
    """汇总审计覆盖率、有限值统计，以及 ERL gate/color/branch 计数。"""
    total = len(records)
    metrics = {}
    for key in _AUDIT_VALUE_KEYS:
        values = [record.get(key) for record in records]
        present_count = sum(value is not None for value in values)
        finite_count = sum(_finite_float_or_none(value) is not None for value in values)
        missing_count = total - present_count
        invalid_count = present_count - finite_count
        metrics[key] = {
            "total": total,
            "present": present_count,
            "finite": finite_count,
            "invalid": invalid_count,
            "missing": missing_count,
            "audited": present_count,
            "none": missing_count,
            "stats": _finite_stats(values),
        }

    def _counts(key, none_label):
        counts = {}
        for record in records:
            value = record.get(key)
            label = (
                none_label if value is None
                else "true" if value is True
                else "false" if value is False
                else str(value)
            )
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    return {
        "total": total,
        "audited": sum(
            any(record.get(key) is not None for key in _AUDIT_VALUE_KEYS)
            for record in records
        ),
        "none": sum(
            not any(record.get(key) is not None for key in _AUDIT_VALUE_KEYS)
            for record in records
        ),
        "metrics": metrics,
        "source_volume_status_counts": _counts("source_volume_status", "missing"),
        "source_volume_reason_counts": _counts("source_volume_reason", "none"),
        "source_volume_gain_clipped_counts": _counts(
            "source_volume_gain_clipped", "missing"
        ),
        "c3_erl_range_gate_status_counts": _counts(
            "c3_erl_range_passed", "na"
        ),
        "c3_erl_range_gate_reason_counts": _counts(
            "c3_erl_range_reason_code", "none"
        ),
        "color_counts": _counts("colored_noise_color", "none"),
        "branch_counts": _counts("branch", "none"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output", type=str, default="/tmp/aec_deep_check_report.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from core.utils.config import load_merged_config

    source_cfg = load_merged_config(str(args.config))
    far_db_range = _far_db_range_from_merged_config(source_cfg)
    cfg = _prepare_verification_config(source_cfg)

    ds = build_dynamic_aec_dataset(cfg, "train")
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(0, base_seed=int(args.seed))

    branches = AEC_SCENE_ORDER

    all_records = []
    all_anomalies = []
    branch_stats = {}

    for branch in branches:
        print(f"\n>>> Checking branch: {branch} (n={args.num_samples})")
        recs, anoms = check_branch(
            ds,
            branch,
            args.num_samples,
            rng_seed=args.seed,
            far_db_range=far_db_range,
        )
        all_records.extend(recs)
        all_anomalies.extend(anoms)
        branch_stats[branch] = summarize(recs)
        st = branch_stats[branch]
        print(
            f"  audit coverage: total={st['total']} audited={st['audited']} "
            f"none={st['none']} colors={st['color_counts']}"
        )

    report = {
        "config": args.config,
        "seed": int(args.seed),
        "task": str(getattr(ds, "task", "joint")).strip().lower(),
        "c3_far_db_expected_range_db": (
            list(far_db_range) if far_db_range is not None else None
        ),
        "c3_far_db_nominal_gate_status": (
            "available" if far_db_range is not None else "range unavailable"
        ),
        "farend_ref_expected_range_db": (
            list(far_db_range) if far_db_range is not None else None
        ),
        "farend_ref_range_status": (
            "available" if far_db_range is not None else "range unavailable"
        ),
        "farend_ref_range_gate_basis": "c3_far_db_nominal",
        "layer_d_actual_farend_rms_gate_role": "diagnostic_only",
        "num_samples_per_branch": args.num_samples,
        "total_samples": len(all_records),
        "total_anomalies": len(all_anomalies),
        "anomalies": all_anomalies,
        "audit_summary": summarize(all_records),
        "branch_stats": branch_stats,
    }

    out_path = Path(args.output)
    out_path.write_text(
        json.dumps(
            _json_finite(report), indent=2, ensure_ascii=False, allow_nan=False
        )
    )
    print(f"\n=== REPORT ===")
    print(f"Total anomalies: {len(all_anomalies)}")
    if all_anomalies:
        print("First 10 anomalies:")
        for a in all_anomalies[:10]:
            print(f"  {a}")
    print(f"\nFull report written to: {out_path}")
    # 审计发现必须反映到退出码，否则 CI/脚本调用方会把异常样本读成通过
    if all_anomalies:
        print(f"[FAIL] {len(all_anomalies)} anomalies detected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
