#!/usr/bin/env python3
"""三路精度对齐验证：离线 PyTorch vs 流式 PyTorch vs 导出的 ONNX Runtime (DeepVQE).

不提供 checkpoint 时仅用于随机权重的手工结构验证。流程：
1. 构建离线 DeepVQE + 流式 StreamingDeepVQE，共享权重
2. 导出 StreamingDeepVQEFlat ONNX（临时文件，验完**保留**并输出路径 + SHA256，供审计复核）
3. 分段逐帧对比三路输出

``--save-onnx`` 正式发布要求 checkpoint、其训练 resolved config provenance、float32、
未放宽的预注册阈值，并且 widened 三路、cold-start/reset 全帧 A/C 与 waveform
reconstruction 门禁全部 PASS；训练配置优先发现 checkpoint 同目录 ``resolved_config.yaml``。

默认 widened 验证口径：≥1000 连续帧 + 2 次中途 reset。稳态 A/C 跳过每段
align_delay 帧，cold-start/reset 门禁覆盖全部帧。

``--inference-dtype`` 为 float16/bfloat16 时，流式 PT 使用对应低精度；离线 PT 与 ORT
保持 fp32 参考锚点。默认 float32 为发布级门禁。

Usage:
    python onnx/verify/verify_3way_deepvqe.py [--variant base] [--n-fft 512] [--align-delay 25] \
        [--frames 1200] [--resets 2] [--checkpoint exp/aec/.../best_model.tar]
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

# ---- 预注册 PASS/FAIL 阈值（float32）----
# 阈值是发布门禁的一部分，禁止按当前结果事后放宽。
PREREG_MSE_THRESHOLD_F32 = 1e-9
PREREG_MAX_ABS_THRESHOLD_F32 = 5e-4

# 非 float32 档位只验证 fp16/bf16。流式 PT 使用低精度，离线 PT 与 ORT 保持
# fp32 锚点；宽松阈值只防止严重回归，严格发布门禁使用 float32。
_LOOSE_DTYPE_THRESH = {"float16": (1e-3, 5e-2), "bfloat16": (1e-2, 2e-1)}

# widened 口径下限（JAE-M10）：连续帧数 ≥1000 且中途 reset ≥2 次。
MIN_WIDENED_FRAMES = 1000
MIN_WIDENED_RESETS = 2


def gate_status(
    mse: float,
    max_abs: float,
    mse_thr: float,
    max_abs_thr: float,
    valid_frames: int = 1,
) -> str:
    """MSE 与 max_abs 同时有限且低于阈值才 PASS（JAE-M10 双指标 gate）。"""
    return (
        "PASS"
        if valid_frames > 0
        and np.isfinite(mse)
        and np.isfinite(max_abs)
        and mse < mse_thr
        and max_abs < max_abs_thr
        else "FAIL"
    )


def publish_gate_passed(statuses, *, widened: bool, save_requested: bool) -> bool:
    """发布模式要求 widened；手工验证仍仅沿用既有数值 gate。"""
    numeric_pass = all(status == "PASS" for status in statuses)
    return numeric_pass and (widened or not save_requested)


def cold_start_gate_passed(
    status_a: str,
    status_c: str,
    *,
    compared_frames: int,
    expected_frames: int,
) -> bool:
    """cold-start/reset 后全部帧的 A/C 均完整通过才放行。"""
    return (
        status_a == "PASS"
        and status_c == "PASS"
        and expected_frames > 0
        and compared_frames == expected_frames
    )


def verification_gate_passed(
    statuses,
    *,
    widened: bool,
    save_requested: bool,
    numeric_gate: bool,
    cold_start_pass: bool,
    waveform_pass: bool,
) -> bool:
    """三路、cold-start 与 waveform 门禁缺一不可。"""
    return (
        numeric_gate
        and cold_start_pass
        and waveform_pass
        and publish_gate_passed(
            statuses,
            widened=widened,
            save_requested=save_requested,
        )
    )


def waveform_reconstruction_passed(
    spectra: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    center: bool,
) -> bool:
    """验证 ORT 输出谱满足当前 STFT 合同且可有限值 iSTFT 重建。"""
    try:
        reconstructed = torch.istft(
            torch.view_as_complex(spectra.contiguous()),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
        )
    except RuntimeError:
        return False
    return bool(reconstructed.numel() > 0 and torch.isfinite(reconstructed).all().item())


def require_matching_runtime_contract(runtime_spec, wrapper_spec) -> None:
    """artifact 必须与本次 config 构建的 wrapper 完整 ABI 一致。"""
    checks = {
        "input names": (runtime_spec.input_names, wrapper_spec.input_names),
        "output names": (runtime_spec.output_names, wrapper_spec.output_names),
        "cache shapes": (dict(runtime_spec.cache_shapes), dict(wrapper_spec.cache_shapes)),
        "STFT metadata": (runtime_spec.stft, wrapper_spec.stft),
        "DeepVQE metadata": (runtime_spec.deepvqe, wrapper_spec.deepvqe),
        "deployment metadata": (
            dict(runtime_spec.deployment_config or {}),
            dict(wrapper_spec.deployment_config or {}),
        ),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise RuntimeError(
            "DeepVQE ONNX runtime contract does not match config-built wrapper: "
            + ", ".join(mismatches)
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="三路精度对齐：离线 PT / 流式 PT / 流式 ORT (DeepVQE)")
    ap.add_argument("--config", "-C", type=Path, default=None,
                    help="训练 YAML；checkpoint 模式默认发现同目录 resolved_config.yaml，"
                         "显式传入时必须与 checkpoint 训练配置指纹一致")
    ap.add_argument("--checkpoint", type=Path, default=None, help="可选 checkpoint (.pth / .tar)")
    ap.add_argument("--variant", type=str, default="base", choices=["base", "small"])
    ap.add_argument("--n-fft", type=_positive_int, default=512)
    ap.add_argument("--hop-length", type=_positive_int, default=256)
    ap.add_argument("--win-length", type=_positive_int, default=None)
    ap.add_argument("--sample-rate", type=_positive_int, default=16000)
    ap.add_argument("--window", type=str, default="hann")
    ap.add_argument("--align-delay", type=_positive_int, default=25)
    ap.add_argument("--frames", type=_positive_int, default=1200,
                    help=f"STFT 帧数（widened 口径要求 ≥{MIN_WIDENED_FRAMES}）")
    ap.add_argument("--resets", type=_nonnegative_int, default=2,
                    help=f"流式 cache 中途 reset 次数（widened 口径要求 ≥{MIN_WIDENED_RESETS}）；"
                         "每次 reset 后 PT/ORT cache 同时冷启动，分段继续比对")
    ap.add_argument("--center", action="store_true",
                    help="STFT center=True (训练默认); 省略则 strict causal center=False")
    ap.add_argument("--inference-dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"], help="推理精度档位（int8 不在验证范围）")
    ap.add_argument("--mse-threshold", type=float, default=None,
                    help="PASS/FAIL MSE 阈值（默认用脚本内预注册常量，见文件头注释）")
    ap.add_argument("--max-abs-threshold", type=float, default=None,
                    help="PASS/FAIL max_abs 阈值（默认用脚本内预注册常量，见文件头注释）")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--save-onnx", type=Path, default=None,
                    help="全部三路检查通过后发布 ONNX 到此路径")
    return ap


def _run(lifecycle: _OwnedOnnxLifecycle) -> int:
    from tasks.aec.models.discriminative.deepvqe import DeepVQE
    from onnx.export import write_streaming_runtime_metadata
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner
    from onnx.models.streaming_deepvqe import StreamingDeepVQE, StreamingDeepVQEFlat
    from onnx.verify.verify_common import (
        load_checkpoint_strict,
        load_checkpoint_training_config,
        print_kept_onnx_artifact,
        publish_verified_onnx,
        release_results_pass,
        require_checkpoint_config_provenance,
        require_release_preflight,
        require_strict_runtime_spec,
    )

    ns = build_arg_parser().parse_args()
    model_compression = 0.3
    cfg = None
    if ns.config is not None:
        from core.utils.config import load_merged_config

        cfg = load_merged_config(str(ns.config))
    elif ns.checkpoint is not None:
        cfg = load_checkpoint_training_config(ns.checkpoint, "deepvqe")
    if cfg is not None:
        model_cfg = cfg.get("model", {})
        if model_cfg.get("type") != "DeepVQE":
            raise ValueError(
                f"DeepVQE verifier requires model.type=DeepVQE, got {model_cfg.get('type')!r}"
            )
        model_args = model_cfg.get("args", {}) or {}
        stft_cfg = cfg.get("stft", {}) or {}
        ns.variant = model_args.get("variant", ns.variant)
        ns.align_delay = int(model_args.get("align_delay", ns.align_delay))
        model_compression = float(model_args.get("compression", model_compression))
        ns.n_fft = int(stft_cfg["n_fft"])
        ns.hop_length = int(stft_cfg["hop_length"])
        ns.win_length = int(stft_cfg["win_length"])
        ns.window = str(stft_cfg.get("window", "hann"))
        ns.center = bool(stft_cfg.get("center", True))
        ns.sample_rate = int((cfg.get("data", {}) or {})["sample_rate"])
    if ns.win_length is None:
        ns.win_length = ns.n_fft
    if cfg is None:
        cfg = {
            "model": {"type": "DeepVQE", "args": {
                "variant": ns.variant,
                "align_delay": ns.align_delay,
                "compression": model_compression,
            }},
            "stft": {
                "n_fft": ns.n_fft,
                "hop_length": ns.hop_length,
                "win_length": ns.win_length,
                "window": ns.window,
                "center": ns.center,
            },
            "data": {"sample_rate": ns.sample_rate},
            "aec_frontend": {"input_routing": "mic_farend"},
        }

    # ORT 会话硬编码 CPUExecutionProvider，且 float32 预注册阈值为 CPU 口径：
    # PT 跑在 GPU 时会引入 PT-on-GPU vs ORT-on-CPU 的设备差，可能超出 MSE 1e-9 阈值。
    if ns.device != "cpu" and ns.inference_dtype == "float32":
        print(f"WARNING: --device {ns.device} 与 ORT CPUExecutionProvider 存在设备差，"
              "float32 预注册阈值（CPU 口径）可能误判 FAIL；建议 --device cpu")
    if ns.inference_dtype == "float32":
        default_mse_thr, default_abs_thr = PREREG_MSE_THRESHOLD_F32, PREREG_MAX_ABS_THRESHOLD_F32
    else:
        default_mse_thr, default_abs_thr = _LOOSE_DTYPE_THRESH[ns.inference_dtype]
    mse_thr = ns.mse_threshold if ns.mse_threshold is not None else default_mse_thr
    abs_thr = ns.max_abs_threshold if ns.max_abs_threshold is not None else default_abs_thr
    require_release_preflight(
        save_onnx=ns.save_onnx,
        checkpoint=ns.checkpoint,
        config=cfg,
        require_config=True,
        inference_dtype=ns.inference_dtype,
        thresholds={
            "mse_threshold": (mse_thr, PREREG_MSE_THRESHOLD_F32),
            "max_abs_threshold": (abs_thr, PREREG_MAX_ABS_THRESHOLD_F32),
        },
    )
    if ns.checkpoint is not None:
        require_checkpoint_config_provenance(ns.checkpoint, cfg, "deepvqe")

    dev = torch.device(ns.device)
    n_fft = ns.n_fft
    hop = ns.hop_length
    win = ns.win_length
    T = ns.frames
    ad = ns.align_delay
    n_resets = max(0, ns.resets)

    # ---- 1. 构建模型 ----
    base = DeepVQE(
        variant=ns.variant,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        align_delay=ns.align_delay,
        compression=model_compression,
    ).eval().to(dev)
    source_binding = None
    if ns.checkpoint:
        # checkpoint key 或参数量不一致时必须失败，禁止使用随机权重比对。
        source_binding = load_checkpoint_strict(
            base,
            ns.checkpoint,
            return_source_binding=True,
            source_files={"config": ns.config},
            model_id="deepvqe",
        )

    stream = StreamingDeepVQE(
        base,
        win_length=win,
        target_fs=ns.sample_rate,
        window=ns.window,
        center=ns.center,
        deployment_config=cfg,
    ).eval().to(dev)
    flat = StreamingDeepVQEFlat(stream).eval()
    inner = flat.inner

    print(f"Config: variant={ns.variant}, n_fft={n_fft}, hop={hop}, align_delay={ad}, "
          f"T={T}, resets={n_resets}, inference_dtype={ns.inference_dtype}")

    # ---- 2. 生成测试数据（波形 → STFT）----
    torch.manual_seed(42)
    N = (T - 1) * hop + n_fft
    mic_wav = torch.randn(1, N, device=dev)
    far_wav = torch.randn(1, N, device=dev)

    if ns.window != "hann":
        raise ValueError(f"DeepVQE verifier requires window='hann', got {ns.window!r}")
    window = torch.hann_window(win, device=dev)
    mic_spec = torch.view_as_real(torch.stft(
        mic_wav, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=ns.center,
    ))
    far_spec = torch.view_as_real(torch.stft(
        far_wav, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=ns.center,
    ))
    # 参与比对的帧数固定为 T（center=True 时裁掉首尾 padding 引入的多余帧）
    T_actual = min(T, mic_spec.shape[2])
    mic_spec = mic_spec[:, :, :T_actual, :]
    far_spec = far_spec[:, :, :T_actual, :]

    # ---- 3. 导出 ONNX（临时文件，验完保留并输出 SHA256）----
    # ``mkstemp`` 原子创建临时文件，避免 TOCTOU 竞态。
    _fd, onnx_path = tempfile.mkstemp(suffix=".onnx")
    lifecycle.own(onnx_path)
    os.close(_fd)
    in_names = flat.get_input_names()
    out_names = flat.get_output_names()
    cache_keys = in_names[2:]  # skip mic_spec, far_spec

    cache_export = inner.init_cache(1, dev, mic_spec.dtype)
    mic_dummy = mic_spec[:, :, 0:1, :]
    far_dummy = far_spec[:, :, 0:1, :]
    dummy_inputs = (mic_dummy, far_dummy) + tuple(cache_export[k] for k in cache_keys)

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
    runtime_spec = require_strict_runtime_spec(sess, "deepvqe")
    from onnx.runtime_spec import runtime_spec_from_wrapper

    wrapper_spec = runtime_spec_from_wrapper(flat, "deepvqe")
    require_matching_runtime_contract(runtime_spec, wrapper_spec)
    ort_runner = StreamingORTFrameRunner(sess, spec=runtime_spec)

    # ---- 3.5 低精度档位：构建独立的低精度流式 PT 副本 ----
    # 流式与离线模型共享参数；必须创建独立副本以保持离线参考为 fp32。
    if ns.inference_dtype != "float32":
        spec_dtype = getattr(torch, ns.inference_dtype)
        base_lp = DeepVQE(
            variant=ns.variant,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            align_delay=ns.align_delay,
            compression=model_compression,
        ).eval().to(dev)
        base_lp.load_state_dict(base.state_dict())
        stream_pt = StreamingDeepVQE(
            base_lp,
            win_length=win,
            target_fs=ns.sample_rate,
            window=ns.window,
            center=ns.center,
            deployment_config=cfg,
        ).eval().to(spec_dtype)
    else:
        stream_pt = stream
        spec_dtype = torch.float32

    # ---- 4. 分段逐帧：流式 PT + 流式 ORT（同步推进，段首同时 reset）----
    # 每段独立重置三路状态；离线路径必须按段前向，才能与流式状态对齐。
    n_seg = n_resets + 1
    bounds = sorted({round(i * T_actual / n_seg) for i in range(n_seg + 1)})
    bounds[0], bounds[-1] = 0, T_actual

    sum_se_off_pt = 0.0   # 跨段累加稳态平方误差（最后统一归一化成 MSE）
    sum_se_off_ort = 0.0
    n_cmp_off = 0         # 跳过 align_delay 后参与稳态比对的元素数
    max_abs_off_pt = 0.0
    max_abs_off_ort = 0.0
    sum_se_cold_pt = 0.0  # 含每段 cold-start/reset 后全部帧的独立硬门禁
    sum_se_cold_ort = 0.0
    n_cmp_cold = 0
    cold_frames_compared = 0
    max_abs_cold_pt = 0.0
    max_abs_cold_ort = 0.0
    ort_segments = []
    max_pt_ort_mse = float("nan")
    max_pt_ort_abs = float("nan")
    worst_pt_ort_step = -1
    n_ort_compared = 0
    frame_metrics_finite = True

    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        with torch.inference_mode():
            seg_off = base.forward_spectra(mic_spec[:, :, b0:b1], far_spec[:, :, b0:b1])

        # PT 与 ORT cache 必须同步重置为零状态。
        cache_pt = stream_pt.init_cache(1, dev, spec_dtype)
        ort_runner.reset()

        seg_pt_frames = []
        seg_ort_frames = []
        for t in range(b0, b1):
            mic_f = mic_spec[:, :, t:t+1, :]
            far_f = far_spec[:, :, t:t+1, :]

            # -- 流式 PT（fp16 档位逐帧转 half；fp32 档位 .to 为 no-op）--
            with torch.inference_mode():
                pt_enh, cache_pt = stream_pt(mic_f.to(spec_dtype), far_f.to(spec_dtype), cache_pt)
            seg_pt_frames.append(pt_enh)

            # -- 流式 ORT --
            ort_value, _ = ort_runner.run_frame(mic_f, far_f)
            ort_enh = torch.from_numpy(ort_value).to(device=dev, dtype=mic_spec.dtype)
            seg_ort_frames.append(ort_enh)

            # PT vs ORT 逐帧差异（MSE + max_abs 双指标）；非有限值阻断 B 项门禁。
            mse_e = float((pt_enh - ort_enh).pow(2).mean().item())
            abs_e = float((pt_enh - ort_enh).abs().max().item())
            frame_finite = bool(np.isfinite(mse_e) and np.isfinite(abs_e))
            frame_metrics_finite = frame_metrics_finite and frame_finite
            n_ort_compared += 1
            if frame_finite and (not np.isfinite(max_pt_ort_mse) or mse_e > max_pt_ort_mse):
                max_pt_ort_mse = mse_e
                worst_pt_ort_step = t
            if frame_finite and (not np.isfinite(max_pt_ort_abs) or abs_e > max_pt_ort_abs):
                max_pt_ort_abs = abs_e

        seg_pt = torch.cat(seg_pt_frames, dim=2)
        seg_ort = torch.cat(seg_ort_frames, dim=2)
        ort_segments.append(seg_ort)

        # cold-start/reset 门禁必须覆盖全部帧，不能由稳态跳帧口径替代。
        cold_pt_diff = seg_off - seg_pt
        cold_ort_diff = seg_off - seg_ort
        sum_se_cold_pt += float(cold_pt_diff.pow(2).sum().item())
        sum_se_cold_ort += float(cold_ort_diff.pow(2).sum().item())
        n_cmp_cold += seg_off.numel()
        cold_frames_compared += b1 - b0
        max_abs_cold_pt = max(max_abs_cold_pt, float(cold_pt_diff.abs().max().item()))
        max_abs_cold_ort = max(max_abs_cold_ort, float(cold_ort_diff.abs().max().item()))

        # 稳态 A/C 每段跳过 ``align_delay`` 帧。
        if b1 - b0 > ad:
            off_s = seg_off[:, :, ad:]
            pt_s = seg_pt[:, :, ad:]
            ort_s = seg_ort[:, :, ad:]
            sum_se_off_pt += float((off_s - pt_s).pow(2).sum().item())
            sum_se_off_ort += float((off_s - ort_s).pow(2).sum().item())
            n_cmp_off += off_s.numel()
            max_abs_off_pt = max(max_abs_off_pt, float((off_s - pt_s).abs().max().item()))
            max_abs_off_ort = max(max_abs_off_ort, float((off_s - ort_s).abs().max().item()))

    # ---- 5. 汇总三路指标 ----
    # (A) 离线 PT vs 流式 PT（跨段稳态聚合）
    mse_off_pt = sum_se_off_pt / n_cmp_off if n_cmp_off else float("nan")
    mx_off_pt = max_abs_off_pt if n_cmp_off else float("nan")

    # (B) 流式 PT vs 流式 ORT（逐帧 max，含 reset 后所有帧）

    # (C) 离线 PT vs 流式 ORT（跨段稳态聚合）
    mse_off_ort = sum_se_off_ort / n_cmp_off if n_cmp_off else float("nan")
    mx_off_ort = max_abs_off_ort if n_cmp_off else float("nan")

    status_A = gate_status(mse_off_pt, mx_off_pt, mse_thr, abs_thr, n_cmp_off)
    status_B = gate_status(
        max_pt_ort_mse, max_pt_ort_abs, mse_thr, abs_thr,
        n_ort_compared if frame_metrics_finite else 0,
    )
    status_C = gate_status(mse_off_ort, mx_off_ort, mse_thr, abs_thr, n_cmp_off)
    mse_cold_pt = sum_se_cold_pt / n_cmp_cold if n_cmp_cold else float("nan")
    mse_cold_ort = sum_se_cold_ort / n_cmp_cold if n_cmp_cold else float("nan")
    status_cold_A = gate_status(
        mse_cold_pt, max_abs_cold_pt, mse_thr, abs_thr, n_cmp_cold,
    )
    status_cold_C = gate_status(
        mse_cold_ort, max_abs_cold_ort, mse_thr, abs_thr, n_cmp_cold,
    )
    cold_start_pass = cold_start_gate_passed(
        status_cold_A,
        status_cold_C,
        compared_frames=cold_frames_compared,
        expected_frames=T_actual,
    )
    y_ort = torch.cat(ort_segments, dim=2)
    waveform_pass = waveform_reconstruction_passed(
        y_ort,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=ns.center,
    )

    # off-vs-stream 有效比较元素为零时，A/C 不得判 PASS。
    vacuous_off_cmp = n_cmp_off == 0
    if vacuous_off_cmp:
        status_A = status_C = "FAIL"

    # 标注 narrow 或 widened 验证口径。
    widened = T_actual >= MIN_WIDENED_FRAMES and n_resets >= MIN_WIDENED_RESETS
    scope = "widened" if widened else "narrow"

    print("\n" + "=" * 70)
    print("  三路精度对齐汇总 (DeepVQE)")
    print("=" * 70)
    print(f"  验证口径: {scope} synthetic | frames={T_actual}, resets={n_resets} "
          f"(segments={len(bounds) - 1}), center={ns.center}")
    if ns.inference_dtype != "float32":
        print(f"  精度档位: {ns.inference_dtype} —— 流式 PT 全链路 {ns.inference_dtype}；"
              "离线 PT / ORT 保持 fp32 参考锚点")
        print("    (A)/(B) 度量低精度流式 vs fp32 参考的偏差；(C) 为纯 fp32 对齐项"
              "（宽松阈值仅防 gross breakage，严格门禁归 float32 档位）")
    if not widened:
        print(f"  WARNING: 当前为 narrow 口径（frames<{MIN_WIDENED_FRAMES} 或 "
              f"resets<{MIN_WIDENED_RESETS}），结果不可作为发布级门禁")
    if vacuous_off_cmp:
        print("  WARNING: off-vs-stream 有效比对元素为 0（段长 ≤ align_delay），(A)/(C) 强制判 FAIL")
    print(f"  align_delay={ad}（稳态 A/C 每段跳过 {ad} 帧；cold-start gate 覆盖全部帧）")
    print(f"  阈值（预注册常量，见文件头注释）: MSE<{mse_thr:g}, max_abs<{abs_thr:g}")
    print()
    print(f"  (A) 离线 PT vs 流式 PT  (跨 {len(bounds) - 1} 段稳态聚合):")
    print(f"      MSE={mse_off_pt:.6e}  max_abs={mx_off_pt:.6e}  [{status_A}]")
    print()
    print(f"  (B) 流式 PT vs 流式 ORT (逐帧, {n_ort_compared} frames):")
    print(f"      max_frame_MSE={max_pt_ort_mse:.6e}  max_frame_abs={max_pt_ort_abs:.6e}  "
          f"worst_step={worst_pt_ort_step}  [{status_B}]")
    print()
    print(f"  (C) 离线 PT vs 流式 ORT (跨 {len(bounds) - 1} 段稳态聚合):")
    print(f"      MSE={mse_off_ort:.6e}  max_abs={mx_off_ort:.6e}  [{status_C}]")
    print("=" * 70)
    print(f"  Cold-start/reset A gate (all {cold_frames_compared}/{T_actual} frames): "
          f"MSE={mse_cold_pt:.6e} max_abs={max_abs_cold_pt:.6e} [{status_cold_A}]")
    print(f"  Cold-start/reset C gate (all {cold_frames_compared}/{T_actual} frames): "
          f"MSE={mse_cold_ort:.6e} max_abs={max_abs_cold_ort:.6e} [{status_cold_C}]")
    print(f"  Waveform reconstruction gate: {'PASS' if waveform_pass else 'FAIL'}")

    numeric_gate = release_results_pass(
        [status_A, status_B, status_C],
        compared_counts=[n_cmp_off, n_ort_compared, n_cmp_off],
        metrics=[mse_off_pt, max_pt_ort_mse, mse_off_ort],
        thresholds=[mse_thr, mse_thr, mse_thr],
    )
    all_pass = verification_gate_passed(
        [status_A, status_B, status_C],
        widened=widened,
        save_requested=ns.save_onnx is not None,
        numeric_gate=numeric_gate,
        cold_start_pass=cold_start_pass,
        waveform_pass=waveform_pass,
    )
    print(f"  总结: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 70)

    if ns.save_onnx:
        published = onnx_artifact.publish(
            ns.save_onnx, all_pass=all_pass, publisher=publish_verified_onnx,
            manifest={
                "model_id": "deepvqe",
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
        # 手工验证模式保留临时 ONNX 供复核。
        print_kept_onnx_artifact(onnx_path)
        lifecycle.keep()
    onnx_artifact.close()
    return 0 if all_pass else 1


def main() -> int:
    with owned_onnx_cleanup() as lifecycle:
        return _run(lifecycle)


if __name__ == "__main__":
    raise SystemExit(main())
