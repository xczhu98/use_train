#!/usr/bin/env python3
"""三路精度对齐验证：离线 PyTorch vs 流式 PyTorch vs 导出的 ONNX Runtime (CAGCRN).

不提供 checkpoint 时仅用于随机权重的手工结构验证。流程：
1. 构建离线 CAGCRN + 流式 StreamingCAGCRN，拷贝权重
2. 导出 StreamingCAGCRNFlat ONNX
3. 逐帧对比三路输出

``--save-onnx`` 正式发布要求 config、checkpoint、float32、未放宽的预注册阈值，
并且三路检查全部 PASS。

Usage:
    python onnx/verify/verify_3way_cagcrn.py -C configs/aec/.../cagcrn.yaml \
        --checkpoint exp/aec/.../best_model.tar [--frames 80]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


from onnx.utils import export_onnx
from onnx.verify.manifest import runtime_contract_from_spec
from onnx.verify.verify_common import (
    OwnedOnnxLifecycle as _OwnedOnnxLifecycle,
    VerifiedOnnxArtifact as _VerifiedOnnxArtifact,
    owned_onnx_cleanup,
)


def publish_gate_passed(statuses, *, missing_in, missing_out) -> bool:
    """A/B/C 与导出图 IO 完整性必须同时通过。"""
    return not missing_in and not missing_out and all(status == "PASS" for status in statuses)


def aligned_comparison_slices(total_frames: int, warmup_frames: int, algorithmic_delay_frames: int):
    """返回离线与流式输出的对齐 slice 及有效比较帧数。"""
    if total_frames < 0 or warmup_frames < 0 or algorithmic_delay_frames < 0:
        raise ValueError("frame counts must be non-negative")
    offline_start = max(0, warmup_frames - algorithmic_delay_frames)
    offline_end = max(offline_start, total_frames - algorithmic_delay_frames)
    stream_start = warmup_frames
    stream_end = total_frames
    valid_frames = max(0, min(offline_end - offline_start, stream_end - stream_start))
    return slice(offline_start, offline_end), slice(stream_start, stream_end), valid_frames


def comparison_status(mse: float, threshold: float, valid_frames: int) -> str:
    """仅在存在有效帧且 MSE 有限时允许比较项通过。"""
    return (
        "PASS"
        if valid_frames > 0 and np.isfinite(mse) and mse < threshold
        else "FAIL"
    )


def _positive_frames(value: str) -> int:
    """argparse 使用的正帧数校验，避免空比较被误当作验证。"""
    frames = int(value)
    if frames <= 0:
        raise argparse.ArgumentTypeError("--frames must be a positive integer")
    return frames


def _run(lifecycle: _OwnedOnnxLifecycle) -> int:
    from tasks.aec.models.discriminative.cagcrn import CAGCRN
    from onnx.export import write_streaming_runtime_metadata
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner
    from onnx.models.streaming_cagcrn import StreamingCAGCRN, StreamingCAGCRNFlat
    from onnx.verify.verify_common import (
        load_checkpoint_strict,
        print_kept_onnx_artifact,
        publish_verified_onnx,
        release_results_pass,
        require_release_preflight,
        require_strict_runtime_spec,
    )

    # 用户决策：只验证 fp32/fp16，int8 不在精度验证范围内（不提供该档位）。
    _default_thresh = {"float32": 1e-5, "float16": 1e-3}
    ap = argparse.ArgumentParser(description="三路精度对齐：离线 PT / 流式 PT / 流式 ORT (CAGCRN)")
    ap.add_argument("--config", "-C", type=Path, default=None,
                    help="训练 YAML；正式发布时必填并作为 STFT 参数真源")
    ap.add_argument("--checkpoint", type=Path, default=None, help="可选 checkpoint (.tar / .pth)")
    ap.add_argument("--frames", type=_positive_frames, default=80, help="正整数 STFT 帧数")
    ap.add_argument("--inference-dtype", default="float32",
                    choices=list(_default_thresh), help="推理精度档位")
    ap.add_argument("--mse-threshold", type=float, default=None,
                    help="PASS/FAIL 阈值（默认按 dtype 自适应）")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--save-onnx", type=Path, default=None,
                    help="全部三路检查通过后发布 ONNX 到此路径")
    ns = ap.parse_args()
    if ns.mse_threshold is None:
        ns.mse_threshold = _default_thresh[ns.inference_dtype]
    require_release_preflight(
        save_onnx=ns.save_onnx,
        checkpoint=ns.checkpoint,
        config=ns.config,
        require_config=True,
        inference_dtype=ns.inference_dtype,
        thresholds={"mse_threshold": (ns.mse_threshold, _default_thresh["float32"])},
    )

    n_fft = 512
    hop = 256
    win_length = 512
    window = "hann"
    center = True
    sample_rate = 16000
    if ns.config is not None:
        from core.utils.config import load_merged_config

        cfg = load_merged_config(str(ns.config))
        model_cfg = cfg.get("model", {})
        if model_cfg.get("type") != "CAGCRN":
            raise ValueError(
                f"CAGCRN verifier requires model.type=CAGCRN, got {model_cfg.get('type')!r}"
            )
        stft_cfg = cfg.get("stft", {}) or {}
        n_fft = int(stft_cfg["n_fft"])
        hop = int(stft_cfg["hop_length"])
        win_length = int(stft_cfg.get("win_length", n_fft))
        window = str(stft_cfg.get("window", "hann"))
        center = bool(stft_cfg.get("center", True))
        sample_rate = int((cfg.get("data", {}) or {}).get("sample_rate", sample_rate))

    dev = torch.device(ns.device)
    F_bins = n_fft // 2 + 1  # 257
    T = ns.frames
    warmup_frames = 20

    # ---- 1. 构建模型 ----
    base = CAGCRN(
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_length,
        window=window,
        center=center,
        sample_rate=sample_rate,
    ).eval().to(dev)
    source_binding = None
    if ns.checkpoint:
        # checkpoint key 或参数量不一致时必须失败，禁止使用随机权重比对。
        source_binding = load_checkpoint_strict(
            base,
            ns.checkpoint,
            return_source_binding=True,
            source_files={"config": ns.config},
            model_id="cagcrn",
        )

    stream = StreamingCAGCRN(base).eval().to(dev)
    flat = StreamingCAGCRNFlat(stream).eval()
    inner = flat.model
    algorithmic_delay_frames = flat.algorithmic_delay_frames

    print(
        f"Config: n_fft={n_fft}, hop={hop}, win_length={win_length}, "
        f"window={window}, center={center}, sample_rate={sample_rate}, "
        f"freq_bins={F_bins}, warmup_frames={warmup_frames}, "
        f"delay_d={flat.delay_d}, "
        f"algorithmic_delay_frames={algorithmic_delay_frames}, T={T}"
    )

    # ---- 2. 导出 ONNX（临时文件）----
    # ``mkstemp`` 原子创建临时文件，避免 TOCTOU 竞态。
    _fd, onnx_path = tempfile.mkstemp(suffix=".onnx")
    lifecycle.own(onnx_path)
    os.close(_fd)
    in_names = flat.get_input_names()
    out_names = flat.get_output_names()

    cache_flat = flat.init_cache_flat(1, str(dev), torch.float32)
    mic_dummy = torch.randn(1, F_bins, 1, 2, device=dev, dtype=torch.float32)
    ref_dummy = torch.randn(1, F_bins, 1, 2, device=dev, dtype=torch.float32)
    dummy_inputs = (mic_dummy, ref_dummy) + tuple(cache_flat)

    print(f"\nExporting ONNX to {onnx_path} ...")
    export_onnx(
        flat, dummy_inputs, onnx_path,
        input_names=in_names, output_names=out_names,
        dynamic_axes=flat.get_dynamic_axes(),
        opset_version=ns.opset, do_constant_folding=True, verbose=False, dynamo=False,
    )
    write_streaming_runtime_metadata(onnx_path, flat)
    print(f"  ONNX exported ({Path(onnx_path).stat().st_size / 1024 / 1024:.1f} MB)")
    print(
        "  Saved complete STFT/cache ABI and versioned CAGCRN runtime metadata "
        f"(canonical 40-cache ABI, delay_d={flat.delay_d}, "
        f"algorithmic_delay_frames={flat.algorithmic_delay_frames})"
    )

    from onnx.runtime_spec import runtime_spec_from_wrapper

    wrapper_spec = runtime_spec_from_wrapper(flat, "cagcrn")
    onnx_artifact = _VerifiedOnnxArtifact(onnx_path)
    sess = onnx_artifact.create_session()
    runtime_spec = require_strict_runtime_spec(sess, "cagcrn")
    if (
        runtime_spec.input_names != wrapper_spec.input_names
        or runtime_spec.output_names != wrapper_spec.output_names
        or runtime_spec.stft != wrapper_spec.stft
        or runtime_spec.cagcrn != wrapper_spec.cagcrn
        or runtime_spec.cache_shapes != wrapper_spec.cache_shapes
    ):
        raise RuntimeError("CAGCRN exported graph/runtime metadata drifted from wrapper spec")
    print(f"  ONNX inputs: {len(runtime_spec.input_names)} (expected {len(in_names)})")
    print(f"  ONNX outputs: {len(runtime_spec.output_names)} (expected {len(out_names)})")
    ort_runner = StreamingORTFrameRunner(sess, spec=runtime_spec)
    missing_in = []
    missing_out = []

    # ---- 3. 生成测试数据 ----
    torch.manual_seed(42)
    mic_full = torch.randn(1, F_bins, T, 2, device=dev, dtype=torch.float32)
    ref_full = torch.randn(1, F_bins, T, 2, device=dev, dtype=torch.float32)

    # ---- 4. 离线 PyTorch ----
    with torch.inference_mode():
        y_off = base._net(mic_full, ref_full)

    # ---- 5. 逐帧：流式 PT + 流式 ORT（同步推进）----
    cache_pt = inner.init_cache(1, str(dev), mic_full.dtype)

    pt_frames = []
    ort_frames = []
    max_pt_ort_mse = float("nan")
    worst_pt_ort_step = -1
    n_ort_compared = 0
    frame_metrics_finite = True

    for t in range(T):
        mic_f = mic_full[:, :, t:t+1, :]
        far_f = ref_full[:, :, t:t+1, :]

        # -- 流式 PT --
        with torch.inference_mode():
            pt_enh, cache_pt = inner(mic_f, far_f, cache_pt)
        pt_frames.append(pt_enh)

        # -- 流式 ORT --
        ort_value, _ = ort_runner.run_frame(mic_f, far_f)
        ort_enh = torch.from_numpy(ort_value).to(device=dev, dtype=mic_full.dtype)
        ort_frames.append(ort_enh)

        # PT vs ORT 逐帧差异；任一非有限指标都阻断 B 项门禁。
        mse_e = float((pt_enh - ort_enh).pow(2).mean().item())
        mse_finite = bool(np.isfinite(mse_e))
        frame_metrics_finite = frame_metrics_finite and mse_finite
        n_ort_compared += 1
        if mse_finite and (not np.isfinite(max_pt_ort_mse) or mse_e > max_pt_ort_mse):
            max_pt_ort_mse = mse_e
            worst_pt_ort_step = t

    y_pt = torch.cat(pt_frames, dim=2)
    y_ort = torch.cat(ort_frames, dim=2)

    # ---- 6. 计算三路 MSE ----
    off_slice_idx, stream_slice_idx, valid_frames = aligned_comparison_slices(
        T, warmup_frames, algorithmic_delay_frames,
    )
    off_slice = y_off[:, :, off_slice_idx]
    pt_slice = y_pt[:, :, stream_slice_idx]
    ort_slice = y_ort[:, :, stream_slice_idx]
    if valid_frames <= 0:
        print(f"FAIL: no valid comparison frames (T={T}, warmup_frames={warmup_frames})")
    mse_off_pt = float((off_slice - pt_slice).pow(2).mean().item()) if valid_frames else float("nan")
    mx_off_pt = float((off_slice - pt_slice).abs().max().item()) if valid_frames else float("nan")

    # (B) 流式 PT vs 流式 ORT (逐帧已统计 max_pt_ort_mse)

    # (C) 离线 PT vs 流式 ORT
    mse_off_ort = float((off_slice - ort_slice).pow(2).mean().item()) if valid_frames else float("nan")
    mx_off_ort = float((off_slice - ort_slice).abs().max().item()) if valid_frames else float("nan")

    # ---- 7. 汇总 ----
    thr = ns.mse_threshold

    status_A = comparison_status(mse_off_pt, thr, valid_frames)
    status_B = (
        "PASS"
        if n_ort_compared > 0 and frame_metrics_finite
        and np.isfinite(max_pt_ort_mse) and max_pt_ort_mse < thr
        else "FAIL"
    )
    status_C = comparison_status(mse_off_ort, thr, valid_frames)

    print("\n" + "=" * 70)
    print("  三路精度对齐汇总 (CAGCRN)")
    print("=" * 70)
    print(f"  T={T}, warmup_frames={warmup_frames}, algorithmic_delay_frames={algorithmic_delay_frames}")
    print(f"  验证口径: narrow synthetic | frames={T}, valid_compare_frames={valid_frames}, resets=0, tail_samples=0 (单次 cold start)")
    print(f"  MSE 阈值: {thr:g}")
    print()
    print(f"  (A) 离线 PT vs 流式 PT  (offline {off_slice_idx}, stream {stream_slice_idx}):")
    print(f"      MSE={mse_off_pt:.6e}  max_abs={mx_off_pt:.6e}  [{status_A}]")
    print()
    print(f"  (B) 流式 PT vs 流式 ORT (逐帧, {n_ort_compared} frames):")
    print(f"      max_frame_MSE={max_pt_ort_mse:.6e}  worst_step={worst_pt_ort_step}  [{status_B}]")
    print()
    print(f"  (C) 离线 PT vs 流式 ORT  (offline {off_slice_idx}, stream {stream_slice_idx}):")
    print(f"      MSE={mse_off_ort:.6e}  max_abs={mx_off_ort:.6e}  [{status_C}]")
    print("=" * 70)

    all_pass = publish_gate_passed(
        [status_A, status_B, status_C], missing_in=missing_in, missing_out=missing_out,
    ) and release_results_pass(
        [status_A, status_B, status_C],
        compared_counts=[valid_frames, n_ort_compared, valid_frames],
        metrics=[mse_off_pt, max_pt_ort_mse, mse_off_ort],
        thresholds=[thr, thr, thr],
    )
    print(f"  总结: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 70)

    if ns.save_onnx:
        published = onnx_artifact.publish(
            ns.save_onnx, all_pass=all_pass, publisher=publish_verified_onnx,
            manifest={
                "model_id": "cagcrn",
                "source_files": {"checkpoint": ns.checkpoint, "config": ns.config},
                "source_binding": source_binding,
                "runtime_contract": runtime_contract_from_spec(runtime_spec),
                "verification": {"statuses": [status_A, status_B, status_C]},
            },
            require_manifest=True,
        )
        if published:
            size_mb = published["size_bytes"] / 1024 / 1024
            print(f"  ONNX artifact kept: {published['path']} ({size_mb:.1f} MB, "
                  f"sha256={published['sha256']})")
    else:
        # 手工验证模式保留临时 ONNX，供审计复核。
        print_kept_onnx_artifact(onnx_path)
        lifecycle.keep()
    onnx_artifact.close()
    return 0 if all_pass else 1


def main() -> int:
    with owned_onnx_cleanup() as lifecycle:
        return _run(lifecycle)


if __name__ == "__main__":
    raise SystemExit(main())
