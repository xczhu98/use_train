#!/usr/bin/env python3
"""三路精度对齐验证：离线 PyTorch vs 流式 PyTorch vs 导出的 ONNX Runtime (Ultra-DPC-AEC).

Usage:
    python onnx/verify/verify_3way_ultra_dpc.py \
        --frames 80 \
        [--checkpoint path/to/checkpoint.tar] \
        [--freq-ratio 2] [--time-skip 1]

压缩配置说明：
  - ``--freq-ratio > 1``：流式 wrapper 原生支持频率压缩（cache 的频点数改为 band 数），
    离线/流式/ORT 三路可比，与未压缩配置同一判定口径。
  - ``--time-skip > 1``：流式 wrapper 维护历史帧、token 相位与最近解压特征，
    只在分组边界提交 DPT cache，组内复用同相位特征；离线整段与逐帧流式
    的 token 分组边界仍不等价。因此手工验证跳过 A/C，仅执行 B（流式 PT vs
    流式 ORT）；三路不完整时 ``--save-onnx`` 必定被阻断，退出码也非零，
    避免调用方把「只验证了 B」当成三路全过。
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
    from tasks.aec.models.discriminative.ultra_dpc_aec import UltraDPCAEC
    from onnx.export import write_streaming_runtime_metadata
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner
    from onnx.models.streaming_ultra_dpc import StreamingUltraDPCAEC, StreamingUltraDPCAECFlat
    from onnx.verify.verify_common import (
        load_checkpoint_strict,
        release_results_pass,
        require_release_preflight,
        require_strict_runtime_spec,
    )

    # 用户决策：只验证 fp32/fp16，int8 不在精度验证范围内（不提供该档位）。
    _default_thresh = {"float32": 1e-5, "float16": 1e-3}
    ap = argparse.ArgumentParser(description="三路精度对齐：离线 PT / 流式 PT / 流式 ORT (Ultra-DPC-AEC)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="可选 checkpoint (.pth / .tar)")
    ap.add_argument("--config", type=Path, default=None, help="发布时绑定的 resolved config")
    ap.add_argument("--n-fft", type=int, default=320)
    ap.add_argument("--hop-size", type=int, default=160)
    ap.add_argument("--win-size", type=int, default=320)
    ap.add_argument("--frames", type=_positive_frames, default=80, help="正整数 STFT 帧数")
    ap.add_argument("--freq-ratio", type=int, default=1,
                    help="频率压缩倍率（1=未压缩；>1 走 FreqCompression 流式路径）")
    ap.add_argument("--time-skip", type=int, default=1,
                    help="时间压缩跳帧（1=未压缩；>1 使用有状态分组 cache，仅对比 PT/ORT）")
    ap.add_argument("--inference-dtype", default="float32", choices=list(_default_thresh))
    ap.add_argument("--mse-threshold", type=float, default=None)
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
    base = UltraDPCAEC(
        n_fft=n_fft, hop_length=hop, embed_dim=48, n_dp_blocks=4,
        encoder_layers=2, decoder_layers=2, gru_dim=32,
        freq_ratio=ns.freq_ratio, time_skip=ns.time_skip,
    ).eval().to(dev)
    source_binding = None
    if ns.checkpoint:
        # checkpoint key 或参数量不一致时必须失败，禁止验证随机初始化模型。
        try:
            source_binding = load_checkpoint_strict(
                base,
                ns.checkpoint,
                return_source_binding=True,
                source_files={"config": ns.config},
                model_id="ultra_dpc",
            )
        except TypeError as exc:
            if "unexpected keyword" not in str(exc):
                raise
            load_checkpoint_strict(base, ns.checkpoint)

    stream = StreamingUltraDPCAEC(base).eval().to(dev)
    flat = StreamingUltraDPCAECFlat(stream).eval()

    print(f"Config: n_fft={n_fft}, hop={hop}, win={win}, F={F_bins}, T={T}, "
          f"freq_ratio={ns.freq_ratio}, time_skip={ns.time_skip}")

    # ---- 2. 生成测试数据（spectrum） ----
    torch.manual_seed(42)
    mic_spec = torch.randn(1, F_bins, T, 2, device=dev, dtype=torch.float32)
    ref_spec = torch.randn(1, F_bins, T, 2, device=dev, dtype=torch.float32)
    err_spec = torch.randn(1, F_bins, T, 2, device=dev, dtype=torch.float32)

    # ---- 3. 导出 ONNX（临时文件） ----
    # ``mkstemp`` 原子创建临时文件，避免 TOCTOU 竞态。
    _fd, onnx_path = tempfile.mkstemp(suffix=".onnx")
    lifecycle.own(onnx_path)
    os.close(_fd)
    in_names = flat.get_input_names()
    out_names = flat.get_output_names()

    dummy_mic = mic_spec[:, :, 0:1, :]
    dummy_ref = ref_spec[:, :, 0:1, :]
    dummy_err = err_spec[:, :, 0:1, :]
    flat_cache = flat.init_cache_flat(1, dev, mic_spec.dtype)
    dummy_inputs = (dummy_mic, dummy_ref, dummy_err) + tuple(flat_cache)

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
    runtime_spec = require_strict_runtime_spec(sess, "ultra_dpc")
    ort_runner = StreamingORTFrameRunner(sess, spec=runtime_spec)

    # ---- 4. 离线 PyTorch ----
    with torch.inference_mode():
        y_off = base.forward_spectra(mic_spec, ref_spec, err_spec)

    # ---- 5. 逐帧：流式 PT + 流式 ORT（同步推进） ----
    cache_pt = stream.init_cache(1, dev, mic_spec.dtype)

    pt_frames = []
    ort_frames = []
    max_pt_ort_mse = float("nan")
    worst_pt_ort_step = -1
    n_ort_compared = 0
    frame_metrics_finite = True

    for t in range(T):
        mic_t = mic_spec[:, :, t:t + 1, :]
        ref_t = ref_spec[:, :, t:t + 1, :]
        err_t = err_spec[:, :, t:t + 1, :]

        # -- 流式 PT --
        with torch.inference_mode():
            pt_enh, cache_pt = stream(mic_t, ref_t, err_t, cache_pt)
        pt_frames.append(pt_enh)

        # -- 流式 ORT --
        ort_value, _ = ort_runner.run_frame(mic_t, ref_t, err_t)
        ort_enh = torch.from_numpy(ort_value).to(device=dev, dtype=mic_spec.dtype)
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
    skip = n_fft // hop  # ~2 frames for n_fft=320, hop=160
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

    # 时间压缩的流式路径是显式逐帧回退：每帧独立左补零成单个 token，
    # 与离线整段按 time_skip 分组 token 不等价，因此只对 PT/ORT 流式两路判门。
    skip_offline = ns.time_skip > 1
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
    if skip_offline:
        status_A = status_C = "SKIP"

    print("\n" + "=" * 70)
    print("  三路精度对齐汇总 (Ultra-DPC-AEC)")
    print("=" * 70)
    print(f"  T={T}, skip_frames={skip}")
    print(f"  MSE 阈值: {thr:g}")
    print()
    if skip_offline:
        print("  (A/C) 离线对比跳过：time_skip>1 时流式为逐帧回退，")
        print("        与离线 token 分组不等价，仅执行 (B)。")
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

    all_pass = status_B == "PASS" and all(s in ("PASS", "SKIP") for s in [status_A, status_C])
    # SKIP 表示该路根本没有比较：只有 A/B/C 全部 PASS 才允许退出码 0。
    three_way_pass = all_pass and not skip_offline
    release_pass = release_results_pass(
        [status_A, status_B, status_C],
        compared_counts=[valid_compare_frames, n_ort_compared, valid_compare_frames],
        metrics=[mse_off_pt, max_pt_ort_mse, mse_off_ort],
        thresholds=[thr, thr, thr],
    )
    print(f"  总结: {'ALL PASS' if three_way_pass else 'SOME FAILED'}")
    if skip_offline:
        print("  注意: (A)/(C) 为 SKIP，本次运行不构成三路门禁，退出码非零。")
    if ns.save_onnx and not release_pass:
        print("  发布门禁: FAIL（A/B/C 必须全部 PASS；SKIP 不可发布）")
    print("=" * 70)

    if ns.save_onnx:
        published = onnx_artifact.publish(
            ns.save_onnx, all_pass=release_pass,
            manifest={
                "model_id": "ultra_dpc",
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
    return 0 if (release_pass if ns.save_onnx else three_way_pass) else 1


def main() -> int:
    with owned_onnx_cleanup() as lifecycle:
        return _run(lifecycle)


if __name__ == "__main__":
    raise SystemExit(main())
