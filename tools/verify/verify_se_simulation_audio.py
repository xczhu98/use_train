#!/usr/bin/env python3
"""验证 SE 动态仿真（``tasks.se.data.dynamic_dataset.DynamicMixingDataset``）各 ``force_branch`` 场景波形。

对每种场景固定 ``force_branch`` 调用 ``get_sample_with_trace``，原子获取训练同构的
``clean`` / ``noisy`` 及（可选）仿真中间 stem，便于对照「观测 noisy / 监督 clean」是否与文档一致。

与训练一致：使用 ``tasks.se.data.build_dataset`` + ``meta.seed`` + ``set_epoch(0, base_seed)``。

默认 **每个场景导出多条**（``--num-samples``，默认 10 条），使用连续数据集下标 ``index, index+1, …``；
每条 ``meta.json`` 含 ``simulation_by_layer``（A–F 分层记录**本样本仿真实际取值**）。

**送给模型的两路**（与训练 ``__getitem__`` 返回一致）：``for_model_noisy.wav``、``for_model_clean_target.wav``。
SE 单通道增强**无远端参考**（无 ``farend``；AEC 管线见 ``verify_aec_simulation_audio.py``）。

**Layer D（post-mix amplitude conditioning，E 前）**：``layer_D_*.wav`` 是按固定顺序执行
D-normalize → D-guard 后的快照（含 ``layer_d_noise_*`` 辅助量）；D-guard 触发时所有相关 stems
使用同一安全增益。Dataset 仿真顺序为 pre-finite → D-normalize → D-guard → E → F → task-post-EF →
final-guard → post-finite。源音量标定仍位于 Layer C-0b（post-RIR/pre-noise-echo）；
final-guard 不回写这些 Layer-D 快照。

输出目录结构::

    <output_dir>/
      README.txt
      summary.json
      simulation_cfg_bounds.json
      01_prob_target_only/
        sample_000_idx0/
          for_model_clean_target.wav
          for_model_noisy.wav
          layer_D_*.wav
          meta.json
        ...
      ...

Usage::

    cd use_train
    python tools/verify/verify_se_simulation_audio.py \\
        -C configs/se/dns_challenge3/gtcrn_causal_offline.yaml \\
        -o exp/verify_se_audio \\
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
import soundfile as sf
import torch
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.data.simulation_layer_view import canonical_simulation_view  # noqa: E402
from tasks.se.data import build_dataset  # noqa: E402
from tools.verify._audio_verify_utils import (  # noqa: E402
    _align_audio_1d_to_len,
    _compute_active_rms,
    _compute_snr_db,
    _make_stereo_left_right,
    _rms,
    _save_wav,
    _to_dbfs,
    _to_numpy_f32,
)
from tools.prepare_data_tools._common import json_safe  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("verify_se_simulation_audio")


# canonical Layer-D stem 键名与既有输出 wav 文件名映射。
_LAYER_D_WAV_EXPORTS: list[tuple[str, str]] = [
    ("layer_d_noisy", "layer_D_noisy.wav"),
    ("layer_d_clean", "layer_D_clean.wav"),
    ("layer_d_noise_additive", "layer_D_noise_additive.wav"),
]


def _validated_layer_d_stems(stems: dict[str, Any]) -> dict[str, np.ndarray]:
    """读取 canonical Layer-D stems；缺失或非法时立即失败。"""
    validated: dict[str, np.ndarray] = {}
    for stem_key, _ in _LAYER_D_WAV_EXPORTS:
        if stem_key not in stems:
            raise RuntimeError(f"missing required simulation stem: {stem_key}")
        arr = np.asarray(stems[stem_key], dtype=np.float32).ravel()
        if arr.size == 0:
            raise RuntimeError(f"required simulation stem is empty: {stem_key}")
        if not np.isfinite(arr).all():
            raise RuntimeError(f"required simulation stem is non-finite: {stem_key}")
        validated[stem_key] = arr
    return validated


def _validated_layer_d_facts(by_layer: dict[str, Any]) -> dict[str, Any]:
    """只接受 SE 嵌套的 canonical Layer D 聚合事实，并限制仿真层为 A-F。"""
    unexpected = sorted(set(by_layer) - {"A", "B", "C", "D", "E", "F"})
    if unexpected:
        raise RuntimeError(
            "simulation_by_layer must contain only Layers A-F; "
            f"unexpected={unexpected}"
        )
    normalize_keys = {
        "mode",
        "target_peak",
        "noisy_scale",
        "clean_scale",
        "noise_scale",
        "preserves_decomposition",
    }
    # apply_after_channel=True 时 PASE 归一化推迟到 task-post-EF：
    # D.normalize 额外记录 pase_deferred，D 顶层额外记录 pase_post_e。
    optional_normalize_keys = {"pase_deferred"}
    optional_layer_d_keys = {"pase_post_e"}
    guard_keys = {"roles", "threshold", "peak", "scale", "applied"}
    layer_d = by_layer.get("D")
    layer_d_keys = {"stage_order", "normalize", "guard"}
    if not isinstance(layer_d, dict) or set(layer_d) - optional_layer_d_keys != layer_d_keys:
        raise RuntimeError(
            "simulation_by_layer.D must contain exactly "
            f"{sorted(layer_d_keys)} plus optional {sorted(optional_layer_d_keys)}"
        )
    if tuple(layer_d["stage_order"]) != ("D-normalize", "D-guard"):
        raise RuntimeError(
            "simulation_by_layer.D.stage_order must be "
            "('D-normalize', 'D-guard')"
        )
    normalize = layer_d["normalize"]
    if (
        not isinstance(normalize, dict)
        or set(normalize) - optional_normalize_keys != normalize_keys
    ):
        raise RuntimeError(
            "simulation_by_layer.D.normalize must contain exactly "
            f"{sorted(normalize_keys)} plus optional {sorted(optional_normalize_keys)}"
        )
    guard = layer_d["guard"]
    if not isinstance(guard, dict) or set(guard) != guard_keys:
        raise RuntimeError(
            "simulation_by_layer.D.guard must contain exactly "
            f"{sorted(guard_keys)}"
        )
    return layer_d


# (子目录名, force_branch, 中文说明)
# 注：SE 只有 3 个场景（prob_target_only / prob_noise_only / prob_target_noise），与 AEC 的 7 场景不同。
SCENARIOS: list[tuple[str, str, str]] = [
    ("01_prob_target_only", "prob_target_only", "prob_target_only：noisy=clean=近端目标；随后仍过 D/E/F"),
    ("02_prob_noise_only", "prob_noise_only", "prob_noise_only：noisy=加性噪声 stem；clean=0"),
    ("03_prob_target_noise", "prob_target_noise", "prob_target_noise：noisy=Layer B 全混合；clean=clean_target（early RIR reference）"),
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
        "layer_d_noise_additive",
    )
    out: dict[str, Any] = {}
    for k, v in st.items():
        if k in ("simulation_by_layer", "sim_meta", "speech_splice_export", "repeat_export"):
            out[k] = v
        elif k in _wave_keys and isinstance(v, np.ndarray):
            out[k] = {"shape": list(v.shape), "rms": _rms(v)}
        elif k in _wave_keys:
            out[k] = str(type(v))
    return out


def _curated_simulation_cfg_bounds(ds: Any) -> dict[str, Any]:
    """从 ``DynamicMixingDataset.sim_cfg`` 抽取与 SNR/音量/Layer B 相关的配置边界（非本样本随机值）。"""
    c = getattr(ds, "sim_cfg", None)
    if c is None:
        return {}

    def gv(name: str, default: Any = None) -> Any:
        return getattr(c, name, default)

    keys_scalar = [
        "prob_target_only",
        "prob_noise_only",
        "prob_reverberation",
        "snr_low",
        "snr_high",
        "prob_wind_noise_sidechain",
        "noise_mix_prob",
        "prob_channel_enabled",
        "noisy_post_mix_degradations_enabled",
    ]
    out: dict[str, Any] = {}
    for k in keys_scalar:
        if not hasattr(c, k):
            continue
        out[k] = gv(k)

    wn = gv("wind_noise_config", None)
    if isinstance(wn, dict):
        out["wind_noise_snr_bounds_db"] = [float(v) for v in wn.get("wind_noise_snr_range", [0.0, 0.0])]

    return json_safe(out)


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
        ds: SE DynamicMixingDataset 实例。
        d_out: 该样本的输出目录。
        dataset_index: 数据集下标。
        branch: force_branch 值（如 "prob_target_only"）。
        zh: 中文场景说明。
        export_stems_subdir: 是否额外导出 stems/ 子目录。
        data_cfg: data 配置块（用于取 sample_rate 等默认值）。

    Returns:
        该样本的 meta dict（会被写入 meta.json）。
    """
    # SE sample 保持 7 元组；stems/branch 与该 sample 由 trace 原子绑定。
    get_traced = getattr(ds, "get_sample_with_trace", None)
    if callable(get_traced):
        traced = get_traced(
            dataset_index, force_branch=branch, capture_stems=True
        )
        sample = traced.sample
        st = traced.trace.legacy_stems("scene") or {}
        actual_branch = traced.trace.metadata.get("branch")
    else:
        ds._last_simulation_stems = None
        ds._last_run_branch = None
        sample = ds._getitem_simulation_verify(dataset_index, force_branch=branch)
        st = getattr(ds, "_last_simulation_stems", None) or {}
        actual_branch = getattr(ds, "_last_run_branch", None)
    clean_t, noisy_t, fs_raw, _L_raw, pn, pe, pen = sample
    clean = _to_numpy_f32(clean_t)
    noisy = _to_numpy_f32(noisy_t)
    fs = int(fs_raw.item()) if isinstance(fs_raw, torch.Tensor) else int(fs_raw)

    # 任何 artifact 写入前先校验 canonical Layer-D stems/facts。
    layer_d_stems = _validated_layer_d_stems(st)
    by_layer, aggregate_facts, _ = canonical_simulation_view(st)
    _validated_layer_d_facts(by_layer)
    final_guard = aggregate_facts.get("final_peak_normalize")
    n_ref = int(max(clean.size, noisy.size))
    aligned_layer_d_stems = {
        key: _align_audio_1d_to_len(value, n_ref)
        for key, value in layer_d_stems.items()
    }

    d_out.mkdir(parents=True, exist_ok=True)
    _save_wav(d_out / "for_model_clean_target.wav", clean, fs)
    _save_wav(d_out / "for_model_noisy.wav", noisy, fs)
    for stem_key, wav_name in _LAYER_D_WAV_EXPORTS:
        _save_wav(d_out / wav_name, aligned_layer_d_stems[stem_key], fs)

    meta: dict[str, Any] = {
        "dataset_index": int(dataset_index),
        "force_branch": branch,
        "description_zh": zh,
        "sample_rate": fs,
        "last_run_branch": actual_branch,
        "simulation_by_layer": json_safe(by_layer),
        "final_peak_normalize": json_safe(final_guard),
        "realized": json_safe(aggregate_facts.get("realized")),
        "sim_meta": json_safe(st.get("sim_meta") or {}),
        "rms": {
            "noisy_for_model": _rms(noisy),
            "clean_target_for_model": _rms(clean),
            "noisy_Layer_D": _rms(aligned_layer_d_stems["layer_d_noisy"]),
            "clean_Layer_D": _rms(aligned_layer_d_stems["layer_d_clean"]),
        },
        "max_abs": {
            "noisy": float(np.max(np.abs(noisy)) if noisy.size else 0.0),
            "clean": float(np.max(np.abs(clean)) if clean.size else 0.0),
        },
        "pure_flags": {
            "is_pure_noise": bool(pn.item()) if torch.is_tensor(pn) else bool(pn),
            "is_pure_echo": bool(pe.item()) if torch.is_tensor(pe) else bool(pe),
            "is_pure_echo_noise": bool(pen.item()) if torch.is_tensor(pen) else bool(pen),
        },
        "simulation_stems_summary": _stems_dict(st),
    }

    # validation：与 AEC 验证脚本同名的事实 + checks。所有数值都来自本样本
    # 实际波形与已记录的仿真事实，checks 只在这些事实异常时列出告警。
    ld_clean = aligned_layer_d_stems["layer_d_clean"]
    ld_noise = aligned_layer_d_stems["layer_d_noise_additive"]
    clean_ref = _align_audio_1d_to_len(clean, n_ref)
    noisy_ref = _align_audio_1d_to_len(noisy, n_ref)
    sim_meta = st.get("sim_meta") or {}
    d_normalize = (by_layer.get("D") or {}).get("normalize") or {}

    noisy_max_abs = float(np.max(np.abs(noisy_ref))) if noisy_ref.size else 0.0
    noise_max_abs = float(np.max(np.abs(ld_noise))) if ld_noise.size else 0.0
    validation: dict[str, Any] = {
        "checks": [],
        "clean_active_dbfs": round(_to_dbfs(_compute_active_rms(clean_ref)), 2),
        "noisy_active_dbfs": round(_to_dbfs(_compute_active_rms(noisy_ref)), 2),
        "noise_active_dbfs": round(_to_dbfs(_compute_active_rms(ld_noise)), 2),
        "noisy_max_abs": round(noisy_max_abs, 6),
        "noisy_clip_ratio": (
            round(float(np.mean(np.abs(noisy_ref) >= 1.0)), 6) if noisy_ref.size else 0.0
        ),
    }

    for name, arr in (("clean", clean_ref), ("noisy", noisy_ref)):
        if not np.isfinite(arr).all():
            validation["checks"].append(f"{name} contains non-finite samples")
    if noisy_max_abs > 1.0 + 1e-6:
        validation["checks"].append(
            f"noisy peak {noisy_max_abs:.4f} exceeds full scale (clipping)"
        )

    # SNR：Layer D clean vs 加性噪声 stem。drawn SNR 等值比较只在单一加性 mix、
    # 无风噪 sidechain 且 D-normalize 保持分解时成立；多 mix/sidechain 只会额外
    # 降低 SNR，此时仅记录实测值。
    mix_snrs = [float(x) for x in (sim_meta.get("additive_snr_mix_db") or [])]
    # clean 静音（prob_noise_only 的 target=0）时 SNR 无定义，不能与 drawn SNR 比较
    if noise_max_abs > 0.0 and _rms(ld_clean) > 1e-9:
        snr_actual_db = _compute_snr_db(ld_clean, ld_noise)
        validation["snr_actual_db"] = round(snr_actual_db, 2)
        if (
            len(mix_snrs) == 1
            and not bool(sim_meta.get("wind_sidechain"))
            and bool(d_normalize.get("preserves_decomposition"))
        ):
            validation["snr_drawn_db"] = round(mix_snrs[0], 2)
            if abs(snr_actual_db - mix_snrs[0]) > 5.0:
                validation["checks"].append(
                    f"Layer-D SNR {snr_actual_db:.1f} dB deviates from drawn SNR "
                    f"{mix_snrs[0]:.1f} dB by more than 5 dB"
                )
    else:
        validation["snr_actual_db"] = None

    # 场景语义：target_only 无加性噪声；noise_only 的 target 为静音；
    # target_noise 必须同时具备 target 与加性噪声
    if branch == "prob_target_only":
        validation["layer_d_noise_max_abs"] = round(noise_max_abs, 6)
        if noise_max_abs > 1e-6:
            validation["checks"].append(
                "prob_target_only: additive noise stem is non-empty "
                f"(max_abs={noise_max_abs:.2e})"
            )
    elif branch == "prob_noise_only":
        clean_max_abs = float(np.max(np.abs(clean_ref))) if clean_ref.size else 0.0
        validation["clean_max_abs"] = round(clean_max_abs, 6)
        if clean_max_abs > 1e-6:
            validation["checks"].append(
                "prob_noise_only: clean target is non-silent "
                f"(max_abs={clean_max_abs:.2e})"
            )
    elif branch == "prob_target_noise" and noise_max_abs <= 1e-6:
        validation["checks"].append("prob_target_noise: additive noise stem is empty")

    meta["validation"] = validation

    # 可选：导出完整 stems 子目录（final_* + layer_D_* + simulation_by_layer.json）
    if export_stems_subdir and st:
        st2 = st
        stem_dir = d_out / "stems"
        stem_dir.mkdir(exist_ok=True)
        if st2.get("final_noisy") is not None:
            _save_wav(stem_dir / "final_noisy_for_model.wav", st2["final_noisy"], fs)
        if st2.get("final_clean") is not None:
            _save_wav(stem_dir / "final_clean_target_for_model.wav", st2["final_clean"], fs)
        for stem_key, wav_name in _LAYER_D_WAV_EXPORTS:
            _save_wav(stem_dir / wav_name, aligned_layer_d_stems[stem_key], fs)
        sbl = st2.get("simulation_by_layer") or {}
        if sbl:
            (stem_dir / "simulation_by_layer.json").write_text(
                json.dumps(json_safe(sbl), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    (d_out / "meta.json").write_text(
        json.dumps(json_safe(meta), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return meta


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export SE DynamicMixingDataset waveforms per force_branch scenario."
    )
    p.add_argument("-C", "--config", required=True, help="训练用 YAML（含 data.mode=dynamic）")
    p.add_argument("-o", "--output-dir", required=True, help="输出根目录")
    p.add_argument("--index", type=int, default=0, help="起始数据集样本下标")
    p.add_argument("--num-samples", type=int, default=10, help="每个场景导出条数")
    p.add_argument(
        "--export-stems",
        action="store_true",
        help="在各条样本子目录下额外写 stems/",
    )
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        logger.error("Config not found: %s", cfg_path)
        return 1
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    num_samples = int(args.num_samples)
    if num_samples < 1:
        logger.error("--num-samples must be >= 1")
        return 1

    from core.utils.config import load_merged_config

    cfg = load_merged_config(str(cfg_path))

    data_cfg = cfg.get("data") or {}
    if str(data_cfg.get("mode", "")).lower() != "dynamic":
        logger.error("data.mode must be 'dynamic' for this tool, got %r", data_cfg.get("mode"))
        return 1

    train_block = dict(data_cfg.get("train") or {})
    train_block["export_simulation_stems"] = True
    data_cfg = dict(data_cfg)
    data_cfg["train"] = train_block
    cfg["data"] = data_cfg

    ds = build_dataset(cfg, "train")
    base_seed = int((cfg.get("meta") or {}).get("seed", 0))
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(0, base_seed=base_seed)

    idx0 = int(args.index)
    if idx0 < 0 or idx0 >= len(ds):
        logger.error("index=%s out of range [0, %s)", idx0, len(ds))
        return 1
    last_idx = idx0 + num_samples - 1
    if last_idx >= len(ds):
        logger.error(
            "index=%s with num_samples=%s needs indices up to %s, but dataset len=%s",
            idx0,
            num_samples,
            last_idx,
            len(ds),
        )
        return 1

    simulation_cfg_bounds = _curated_simulation_cfg_bounds(ds)
    (out_root / "simulation_cfg_bounds.json").write_text(
        json.dumps(simulation_cfg_bounds, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "config": str(cfg_path),
        "index_start": idx0,
        "num_samples_per_scenario": num_samples,
        "base_seed": base_seed,
        "dataset_len": len(ds),
        "export_simulation_stems": True,
        "export_stems_subdir": bool(args.export_stems),
        "simulation_cfg_bounds": simulation_cfg_bounds,
        "scenarios": [],
    }

    readme_lines = [
        "SE DynamicMixingDataset 场景导出",
        f"config: {cfg_path}",
        f"起始 index: {idx0} | 每场景条数: {num_samples}",
        f"base_seed: {base_seed}",
        "",
        "for_model_*.wav：与训练 __getitem__ 返回 (clean, noisy) 一致；无 farend。",
        "layer_D_*.wav：Layer D（post-mix amplitude conditioning）按 D-normalize → D-guard 固定顺序执行后、E 前的相关 stem 快照；若 D-guard 生效，所有相关 stem 使用同一安全增益。",
        "Dataset 仿真顺序：pre-finite → D-normalize → D-guard → E → F → task-post-EF → final-guard → post-finite。",
        "final-guard：task-post-EF 后仅保护 observation；不回写 layer_D_*.wav。",
        "",
        "验证项（meta.json -> validation）：",
        "  - clean/noisy/noise_active_dbfs：活跃段音量。",
        "  - noisy_max_abs / noisy_clip_ratio：最终 observation 的峰值与触顶比例。",
        "  - snr_actual_db：从 Layer D clean vs 加性噪声计算的实测 SNR；当该样本只有单一",
        "    加性 mix、无风噪 sidechain 且 D-normalize 保持分解时额外给出 snr_drawn_db 并比较。",
        "  - checks：非有限值 / 削波 / SNR 偏差 >5 dB / 场景语义异常（target_only 含加性噪声、",
        "    noise_only 的 target 非静音、target_noise 无加性噪声）会在此列出告警。",
        "",
    ]

    for subdir, branch, zh in SCENARIOS:
        scenario_entry: dict[str, Any] = {"subdir": subdir, "branch": branch, "samples": []}
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
            readme_lines.append(f"[{subdir}/{sample_dir_name}] {branch} index={dataset_idx}")
            layer = meta.get("simulation_by_layer") or {}
            b = layer.get("B") or {}
            c = layer.get("C") or {}
            layer_d_facts = layer.get("D") or {}
            d_normalize = layer_d_facts.get("normalize") or {}
            d_guard = layer_d_facts.get("guard") or {}
            final_guard = meta.get("final_peak_normalize") or {}
            readme_lines.append(
                f"  Layer B: branch={b.get('branch')} force={b.get('force_branch')}"
            )
            readme_lines.append(
                f"  Layer C: reverb={c.get('c0_reverb_sampled')} "
                f"snr_primary={c.get('c1_c2_snr_primary_drawn_db')} "
                f"wind_applied={c.get('c1_c2_wind_sidechain_applied')}"
            )
            r = meta.get("realized") or layer.get("realized") or {}
            readme_lines.append(
                f"  D-normalize: mode={d_normalize.get('mode')} "
                f"target_peak={d_normalize.get('target_peak')} "
                f"noisy_scale={d_normalize.get('noisy_scale')}"
            )
            readme_lines.append(
                f"  D-guard: applied={d_guard.get('applied')} "
                f"scale={d_guard.get('scale')}"
            )
            readme_lines.append(
                "  final-guard: "
                f"applied={final_guard.get('applied')} scale={final_guard.get('scale')}"
            )
            readme_lines.append(
                f"  Realized: mic_dbfs_rms={r.get('mic_dbfs_rms')} "
                f"clean_dbfs_rms={r.get('clean_dbfs_rms')} "
                f"mic_clip_frame_ratio={r.get('mic_clip_frame_ratio')}"
            )
            readme_lines.append(
                f"  RMS noisy={meta['rms']['noisy_for_model']:.6f} | clean={meta['rms']['clean_target_for_model']:.6f}"
            )
            val = meta.get("validation")
            if not isinstance(val, dict):
                # validation 事实缺失时不得断言全部通过
                readme_lines.append("  [WARN] validation metadata unavailable")
            else:
                if val.get("snr_actual_db") is not None:
                    readme_lines.append(f"  Validation SNR={val['snr_actual_db']:.1f} dB")
                if all(
                    val.get(key) is not None
                    for key in ("clean_active_dbfs", "noisy_active_dbfs", "noise_active_dbfs")
                ):
                    readme_lines.append(
                        f"  Active dBFS clean={val['clean_active_dbfs']:.1f} "
                        f"noisy={val['noisy_active_dbfs']:.1f} "
                        f"noise={val['noise_active_dbfs']:.1f}"
                    )
                checks = val.get("checks", [])
                if checks:
                    for chk in checks:
                        readme_lines.append(f"  [CHECK] {chk}")
                else:
                    readme_lines.append("  [OK] All checks passed")
            readme_lines.append("")

        summary["scenarios"].append(scenario_entry)
        readme_lines.append("")

    (out_root / "README.txt").write_text("\n".join(readme_lines), encoding="utf-8")
    (out_root / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False),
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
