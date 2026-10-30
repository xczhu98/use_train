#!/usr/bin/env python3
"""三路精度对齐验证：离线 PyTorch vs 流式 PyTorch vs 导出的 ONNX Runtime (FastEnhancer).

不提供 checkpoint 时仅用于随机权重的手工结构验证。``--save-onnx`` 正式发布要求
checkpoint、float32、未放宽的预注册阈值，并且三路检查全部 PASS。

Usage:
    python onnx/verify/verify_3way_fastenhancer.py \
        --variant B --n-fft 512 --hop-size 256 --frames 80 \
        [--checkpoint path/to/checkpoint.pth]
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


def _positive_frames(value: str) -> int:
    """argparse 使用的正帧数校验，避免空比较被误当作验证。"""
    frames = int(value)
    if frames <= 0:
        raise argparse.ArgumentTypeError("--frames must be a positive integer")
    return frames


def _run(lifecycle: _OwnedOnnxLifecycle) -> int:
    from tasks.se.models.discriminative.fastenhancer import FastEnhancer
    from onnx.models.streaming_fastenhancer import StreamingFastEnhancer, StreamingFastEnhancerFlat
    from onnx.export import write_streaming_runtime_metadata
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner
    from onnx.verify import verify_common
    from onnx.verify.verify_common import (
        release_results_pass,
        require_release_preflight,
        require_strict_runtime_spec,
    )

    # 用户决策：只验证 fp32/fp16，int8 不在精度验证范围内（不提供该档位）。
    _default_thresh = {"float32": 1e-5, "float16": 1e-3}
    ap = argparse.ArgumentParser(description="三路精度对齐：离线 PT / 流式 PT / 流式 ORT (FastEnhancer)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="可选 checkpoint (.pth / .tar)")
    ap.add_argument("--config", type=Path, default=None, help="发布时绑定的 resolved config")
    ap.add_argument("--variant", type=str, default="B", choices=["T", "B", "S", "M", "L"])
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--hop-size", type=int, default=256)
    ap.add_argument("--win-size", type=int, default=512)
    ap.add_argument("--frames", type=_positive_frames, default=80, help="正整数 STFT 帧数")
    ap.add_argument("--inference-dtype", default="float32", choices=list(_default_thresh), help="推理精度档位")
    ap.add_argument("--mse-threshold", type=float, default=None, help="PASS/FAIL 阈值（默认按 dtype 自适应）")
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

    dev = torch.device(ns.device)
    n_fft = ns.n_fft
    hop = ns.hop_size
    win = ns.win_size
    F_bins = n_fft // 2 + 1
    T = ns.frames

    # ---- 1. 构建模型 ----
    base = FastEnhancer(variant=ns.variant, n_fft=n_fft).eval().to(dev)
    source_binding = None
    if ns.checkpoint:
        try:
            source_binding = verify_common.load_checkpoint_strict(
                base,
                ns.checkpoint,
                return_source_binding=True,
                source_files={"config": ns.config},
                model_id="fastenhancer",
            )
        except TypeError as exc:
            if "unexpected keyword" not in str(exc):
                raise
            verify_common.load_checkpoint_strict(base, ns.checkpoint)

    stream = StreamingFastEnhancer(base).eval().to(dev)
    # 流式 wrapper 的计算不依赖图外 STFT；导出 metadata 必须绑定验证实际前端。
    stream.hop_length = hop
    stream.win_length = win
    flat = StreamingFastEnhancerFlat(stream).eval()
    inner = flat.inner

    print(f"Config: variant={ns.variant}, n_fft={n_fft}, hop={hop}, win={win}, F={F_bins}, T={T}")

    # ---- 2. 生成测试数据（spectrum） ----
    torch.manual_seed(42)
    spec = torch.randn(1, F_bins, T, 2, device=dev, dtype=torch.float32)

    # ---- 3. 导出 ONNX（临时文件） ----
    # ``mkstemp`` 原子创建临时文件，避免 TOCTOU 竞态。
    _fd, onnx_path = tempfile.mkstemp(suffix=".onnx")
    lifecycle.own(onnx_path)
    os.close(_fd)
    in_names = flat.get_input_names()
    out_names = flat.get_output_names()
    cache_keys = in_names[1:]

    cache_export = inner.init_cache(1, dev, spec.dtype)
    spec_dummy = spec[:, :, 0:1, :]
    dummy_inputs = (spec_dummy,) + tuple(cache_export[k] for k in cache_keys)

    print(f"\nExporting ONNX to {onnx_path} ...")
    export_onnx(
        flat, dummy_inputs, onnx_path,
        input_names=in_names, output_names=out_names,
        dynamic_axes=flat.get_dynamic_axes(),
        opset_version=ns.opset, do_constant_folding=True, verbose=False, dynamo=False,
    )
    write_streaming_runtime_metadata(onnx_path, flat)
    print(f"  ONNX exported ({Path(onnx_path).stat().st_size / 1024 / 1024:.1f} MB)")

    onnx_artifact = _VerifiedOnnxArtifact(onnx_path)
    sess = onnx_artifact.create_session()
    runtime_spec = require_strict_runtime_spec(sess, "fastenhancer")
    ort_runner = StreamingORTFrameRunner(sess, spec=runtime_spec)

    # ---- 4. 离线 PyTorch ----
    with torch.inference_mode():
        y_off = base(spec)

    # ---- 5. 逐帧：流式 PT + 流式 ORT（同步推进） ----
    cache_pt = inner.init_cache(1, dev, spec.dtype)

    pt_frames = []
    ort_frames = []
    max_pt_ort_mse = float("nan")
    worst_pt_ort_step = -1
    n_ort_compared = 0
    frame_metrics_finite = True

    for t in range(T):
        sp = spec[:, :, t:t + 1, :]

        # -- 流式 PT --
        with torch.inference_mode():
            pt_enh, cache_pt = inner(sp, cache_pt)
        pt_frames.append(pt_enh)

        # -- 流式 ORT --
        ort_value, _ = ort_runner.run_frame(sp)
        ort_enh = torch.from_numpy(ort_value).to(device=dev, dtype=spec.dtype)
        ort_frames.append(ort_enh)

        # PT vs ORT 逐帧差异；非有限指标直接阻断 B 项门禁。
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
    # FastEnhancer 无 lookahead，但 center=True STFT 引入 n_fft//2 的边界效应。
    # 公平比较：跳过前/后各若干帧。
    skip = n_fft // hop  # ~2 frames for n_fft=512, hop=256
    skip = max(skip, 1)

    off_slice = y_off[:, :, skip:T - skip] if T > 2 * skip else y_off
    pt_slice = y_pt[:, :, skip:T - skip] if T > 2 * skip else y_pt
    ort_slice = y_ort[:, :, skip:T - skip] if T > 2 * skip else y_ort

    valid_compare_frames = T if T <= 2 * skip else T - 2 * skip
    mse_off_pt = float((off_slice - pt_slice).pow(2).mean().item()) if valid_compare_frames > 0 else float("nan")
    mx_off_pt = float((off_slice - pt_slice).abs().max().item()) if valid_compare_frames > 0 else float("nan")

    mse_off_ort = float((off_slice - ort_slice).pow(2).mean().item()) if valid_compare_frames > 0 else float("nan")
    mx_off_ort = float((off_slice - ort_slice).abs().max().item()) if valid_compare_frames > 0 else float("nan")

    # ---- 7. 汇总 ----
    thr = ns.mse_threshold

    status_A = (
        "PASS"
        if valid_compare_frames > 0 and np.isfinite(mse_off_pt) and mse_off_pt < thr
        else "FAIL"
    )
    status_B = (
        "PASS"
        if n_ort_compared > 0 and frame_metrics_finite
        and np.isfinite(max_pt_ort_mse) and max_pt_ort_mse < thr
        else "FAIL"
    )
    status_C = (
        "PASS"
        if valid_compare_frames > 0 and np.isfinite(mse_off_ort) and mse_off_ort < thr
        else "FAIL"
    )

    print("\n" + "=" * 70)
    print("  三路精度对齐汇总 (FastEnhancer)")
    print("=" * 70)
    print(f"  T={T}, skip_frames={skip}, variant={ns.variant}")
    print(f"  MSE 阈值: {thr:g}")
    print()
    print(f"  (A) 离线 PT vs 流式 PT  (frames [{skip}, {T - skip})):")
    print(f"      MSE={mse_off_pt:.6e}  max_abs={mx_off_pt:.6e}  [{status_A}]")
    print()
    print(f"  (B) 流式 PT vs 流式 ORT (逐帧, {n_ort_compared} frames):")
    print(f"      max_frame_MSE={max_pt_ort_mse:.6e}  worst_step={worst_pt_ort_step}  [{status_B}]")
    print()
    print(f"  (C) 离线 PT vs 流式 ORT (frames [{skip}, {T - skip})):")
    print(f"      MSE={mse_off_ort:.6e}  max_abs={mx_off_ort:.6e}  [{status_C}]")
    print("=" * 70)

    all_pass = release_results_pass(
        [status_A, status_B, status_C],
        compared_counts=[valid_compare_frames, n_ort_compared, valid_compare_frames],
        metrics=[mse_off_pt, max_pt_ort_mse, mse_off_ort],
        thresholds=[thr, thr, thr],
    )
    print(f"  总结: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 70)

    if ns.save_onnx:
        published = onnx_artifact.publish(
            ns.save_onnx, all_pass=all_pass,
            manifest={
                "model_id": "fastenhancer",
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
    onnx_artifact.close()
    return 0 if all_pass else 1


def main() -> int:
    with owned_onnx_cleanup() as lifecycle:
        return _run(lifecycle)


if __name__ == "__main__":
    raise SystemExit(main())
