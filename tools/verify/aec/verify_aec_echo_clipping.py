#!/usr/bin/env python3
"""验证 aec_echo_clipping：统计路径选择、实际改写、绝对 rail 命中及采样点削波率，
检查 aec_disk 与最终输出一致性，并比较 echo peak/RMS、FST/DT/NST 状态占比。
绝对 rail 统计只计有对应事实的路径；相对截幅或未触发样本不冒充零命中。

对应设计文档 §7 "验证思路" 与
docs/guides/aec/AEC_Data_Simulation.md。

Usage::

    cd use_train
    python tools/verify/aec/verify_aec_echo_clipping.py \\
        -C configs/aec/microsoft_challenge/dbgnet/dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml \\
        -o exp/verify_echo_clipping \\
        --num-samples 500
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tasks.aec.data.dynamic_aec_build import build_dynamic_aec_dataset  # noqa: E402
from tasks.aec.data.state_label import STATE_LABEL_DT, STATE_LABEL_FST, STATE_LABEL_NST  # noqa: E402
from core.utils.config import load_merged_config  # noqa: E402


def _build_dataset(cfg_path: str, *, echo_clip_enabled: bool):
    cfg = load_merged_config(cfg_path)
    sim_cfg = cfg["data"]["train"]["simulation_cfg"]
    # load_merged_config 已完成分层 YAML 到 runtime 字段的规范化。
    for name in ("target_only", "noise_only", "echo_only", "noise_echo_only",
                 "target_noise", "target_echo", "target_noise_echo"):
        sim_cfg[f"prob_{name}"] = float(name == "target_echo")
    sim_cfg["aec_echo_clipping"]["enabled"] = echo_clip_enabled
    # ``stft:`` is a top-level YAML key (not nested under ``model:``); state
    # detector frame params are derived from it (see comment in each AEC
    # config's data.train block: "state_detector_hop_length / ... are derived
    # from the top-level stft block").
    stft_cfg = cfg.get("stft")
    ds = build_dynamic_aec_dataset(cfg, "train", stft_cfg=stft_cfg)
    # Force the 3-class FST/DT/NST label (rather than the 8-class "scenario"
    # label, which is the constructor default) so STATE_LABEL_FST/DT/NST apply.
    ds.state_detector_label_type = "fst_dt_nst"
    seed = int(cfg.get("meta", {}).get("seed", 0))
    ds.set_epoch(0, base_seed=seed)
    ds.export_simulation_stems = True
    return ds


def _state_label_proportions(ds: Any, stems: dict[str, Any]) -> dict[str, float]:
    """Recompute state label from already-exported post-Layer-D stems, via the
    same ``_compute_state_label_tensor`` the dataset itself uses.

    ``stems`` 应由调用方从同一次 ``get_sample_with_trace`` 返回值提取，避免
    sample 与 mutable sidecar 串线。
    """
    nearend = stems.get("layer_d_clean")
    echo = stems.get("layer_d_echo")
    if nearend is None or echo is None:
        return {}
    noise = stems.get("layer_d_noise")
    label = ds._compute_state_label_tensor(nearend, echo, noise)
    arr = label.numpy().ravel()
    total = max(len(arr), 1)
    return {
        "fst": float((arr == STATE_LABEL_FST).sum()) / total,
        "dt": float((arr == STATE_LABEL_DT).sum()) / total,
        "nst": float((arr == STATE_LABEL_NST).sum()) / total,
    }


def _summarize_clipping(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """实际改写率以全部样本为分母；rail-hit 仅以有绝对 rail 事实的样本为分母。"""
    summary: dict[str, Any] = {"samples": len(rows)}
    for field in ("echo_applied", "farend_applied", "any_applied",
                  "echo_rail_hit", "farend_rail_hit", "rail_hit"):
        if field.endswith("_applied") and any(row[field] is None for row in rows):
            raise ValueError(f"missing clipping boolean fact: {field}")
        values = [row[field] for row in rows if row[field] is not None]
        if any(type(value) is not bool for value in values):
            raise ValueError(f"invalid clipping boolean fact: {field}")
        summary[field] = {"count": sum(values), "observed": len(values),
                          "rate": sum(values) / len(values) if values else None}
    if any((row["clipped_sample_count"] is None) != (row["rail_sample_count"] is None) for row in rows):
        raise ValueError("clipping sample counts must be both present or both absent")
    rail_rows = [row for row in rows if row["rail_sample_count"] is not None]
    for row in rail_rows:
        count, total = row["clipped_sample_count"], row["rail_sample_count"]
        if type(count) is not int or type(total) is not int or not 0 <= count <= total:
            raise ValueError("invalid clipping sample counts")
    count = sum(row["clipped_sample_count"] for row in rail_rows)
    total = sum(row["rail_sample_count"] for row in rail_rows)
    summary["absolute_rail_samples"] = {
        "clipped": count, "total": total, "ratio": count / total if total else None,
    }
    return summary


def run(cfg_path: str, out_dir: Path, num_samples: int) -> None:
    # 触发率以 num_samples 为分母；0 或负数会产出全空的"通过"报告
    if type(num_samples) is not int or num_samples <= 0:
        raise ValueError(f"num_samples must be a positive int, got {num_samples!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}

    for tag, enabled in (("echo_clipping_off", False), ("echo_clipping_on", True)):
        ds = _build_dataset(cfg_path, echo_clip_enabled=enabled)
        path_counter: Counter[str] = Counter()
        cp_counter: Counter[str] = Counter()
        peak_deltas: list[float] = []
        rms_deltas: list[float] = []
        state_props: list[dict[str, float]] = []
        clipping_rows: list[dict[str, Any]] = []
        disk_checks: Counter[str] = Counter()

        for i in range(num_samples):
            traced = ds.get_sample_with_trace(
                i, force_branch="prob_target_echo", capture_stems=True
            )
            stems = traced.trace.legacy_stems("scene") or {}
            layer_c = (stems.get("simulation_by_layer") or {}).get("C", {})
            audit = layer_c.get("c3_echo_clipping") or {}
            clipping_rows.append(audit)
            disk = traced.trace.metadata_dict()["sources"]["scene"]["aec_disk_consistency"]
            if not disk or not all(value is True for value in disk.values()):
                raise ValueError(f"sample {i}: aec_disk consistency failed: {disk}")
            disk_checks.update(disk)
            path = audit.get("path") or "none"
            path_counter[path] += 1
            # JAE-M11：capture point 语义桶命中率（协议见 tasks/aec/data/echo_clipping.py）
            cp_counter[audit.get("capture_point") or "unknown"] += 1
            if audit.get("peak_before") is not None and audit.get("peak_after") is not None:
                peak_deltas.append(audit["peak_after"] - audit["peak_before"])
            if audit.get("rms_before") is not None and audit.get("rms_after") is not None:
                rms_deltas.append(audit["rms_after"] - audit["rms_before"])
            props = _state_label_proportions(ds, stems)
            if props:
                state_props.append(props)

        report[tag] = {
            "scenario": "prob_target_echo",
            "clipping": _summarize_clipping(clipping_rows),
            "path_clipping": {
                path: _summarize_clipping([row for row in clipping_rows if (row["path"] or "none") == path])
                for path in path_counter
            },
            "aec_disk_consistency": {"checked_samples": num_samples, "passed": dict(disk_checks)},
            "path_trigger_counts": dict(path_counter),
            "path_trigger_rate": {k: v / num_samples for k, v in path_counter.items()},
            "capture_point_counts": dict(cp_counter),
            "capture_point_rate": {k: v / num_samples for k, v in cp_counter.items()},
            "echo_peak_delta_mean": float(np.mean(peak_deltas)) if peak_deltas else None,
            "echo_rms_delta_mean": float(np.mean(rms_deltas)) if rms_deltas else None,
            "state_label_mean_proportions": (
                {
                    k: float(np.mean([p[k] for p in state_props]))
                    for k in ("fst", "dt", "nst")
                }
                if state_props
                else None
            ),
        }

    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-C", "--config", required=True, help="AEC training YAML config path")
    p.add_argument("-o", "--output-dir", required=True, help="Output directory for report.json")
    p.add_argument("--num-samples", type=int, default=200)
    args = p.parse_args()
    run(args.config, Path(args.output_dir), args.num_samples)


if __name__ == "__main__":
    main()
