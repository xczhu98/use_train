#!/usr/bin/env python3
"""统一流式 ONNX 导出入口。

实际 direct mode 支持 ``streaming_gtcrn``、``streaming_deepvqe``、
``streaming_localvqe``、``streaming_cagcrn``、``streaming_fastenhancer``、
``streaming_ulunas``、``streaming_ultra_dpc``、``streaming_tse`` 和
``streaming_online_spatialnet``；config-based 路径支持因果 BSRNN wrapper 与 DBGNet。
模型保持 spec-in/spec-out，STFT/iSTFT 位于图外。

导出 dummy cache 由模型实际提供的 ``init_cache()``、``init_cache_flat()``、
``initialize_cache()`` 或 ``initialize_cache_flat()`` 接口推导；支持 metadata 的路径经
``write_streaming_runtime_metadata()`` 写入 ONNX。``--output``
只允许写入 ``runs/onnx_debug`` 下的 debug artifact，可用 ``--smoke`` 执行一次 PT/ORT
数值冒烟。generic exporter 不具备正式发布资格；``--save-onnx`` 永远拒绝；正式发布
必须由模型专属 ``verify_3way`` 脚本完成完整
三路门禁后调用 ``publish_verified_onnx``。OnlineSpatialNet 在专属 verifier 落地前只能
导出 debug artifact。

Usage:
    # GTCRN debug export；可选单步 PT/ORT smoke
    python onnx/export.py \
        --model streaming_gtcrn \
        --checkpoint /path/to/checkpoints/best_model.tar \
        --output runs/onnx_debug/gtcrn_streaming.onnx --smoke

    # LocalVQE direct debug export：config 必须声明 model.type=LocalVQE
    python onnx/export.py \
        --model streaming_localvqe \
        -C configs/aec/microsoft_challenge/localvqe/localvqe_16k_20ms_delay80ms_loss_hybrid_fwsdr.yaml \
        --checkpoint /path/to/checkpoints/best_model.tar \
        --output runs/onnx_debug/localvqe_streaming.onnx --smoke

    # 因果 BSRNN debug export：仅 config-based wrapper 路径
    python onnx/export.py \
        -C configs/se/urgent2026/usr_bsrnn_causal_urgent26_offline_hybrid.yaml \
        --checkpoint /path/to/checkpoints/best_model.tar \
        --output runs/onnx_debug/bsrnn_streaming.onnx
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Callable

# Add project root to sys.path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import numpy as np
from core.utils.config import load_merged_config
from onnx.runtime_spec import runtime_spec_from_wrapper
from onnx.utils import (
    create_ort_session,
    export_onnx,
    pypi_onnx_scope,
    write_onnx_metadata,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_onnx")


# direct streaming exporter registry。模型 wrapper 只需注册自己的构造函数，
# ``export_model`` 不再维护一条随模型数增长的分支链。注册表只保存导出
# ABI 已由 wrapper 自身公开的函数；通用 exporter 仍保留原有 debug-only 语义。
_STREAMING_EXPORTERS: dict[
    str, Callable[[torch.nn.Module, str, int], tuple[torch.nn.Module, tuple]]
] = {}


def register_streaming_exporter(*model_types: str):
    """注册 direct streaming wrapper 的导出函数。

    注册发生在模块加载阶段，重复模型名或空模型名均直接报错，避免插件
    静默覆盖既有 ABI。函数签名保持 ``(model, output_path, opset)``，返回值
    与旧版 ``export_model`` 分支完全一致。
    """
    if not model_types or any(not isinstance(name, str) or not name for name in model_types):
        raise ValueError("streaming exporter registration requires non-empty model names")

    def decorator(
        exporter: Callable[[torch.nn.Module, str, int], tuple[torch.nn.Module, tuple]]
    ):
        for model_type in model_types:
            if model_type in _STREAMING_EXPORTERS:
                raise RuntimeError(f"Duplicate streaming exporter registration: {model_type}")
            _STREAMING_EXPORTERS[model_type] = exporter
        return exporter

    return decorator


def parse_args():
    p = argparse.ArgumentParser(description="Streaming ONNX export（spec-in/spec-out；STFT/iSTFT 位于图外）")
    p.add_argument("-C", "--config", default=None, help="训练配置；direct mode 仅在对应模型需要参数时使用")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "direct mode：streaming_gtcrn/streaming_deepvqe/streaming_localvqe/"
            "streaming_cagcrn/streaming_fastenhancer/streaming_ulunas/"
            "streaming_ultra_dpc/streaming_tse/streaming_online_spatialnet；"
            "未指定时走 config-based BSRNN/DBGNet 路径。"
        ),
    )
    p.add_argument(
        "--tse-enrollment-samples",
        type=int,
        default=48000,
        help="streaming_tse dummy enrollment length used only to derive/export dynamic cache ABI",
    )
    p.add_argument("--variant", type=str, default=None, choices=["T", "B", "S", "M", "L"],
                    help="FastEnhancer variant；优先级高于 config 和 checkpoint 文件名推断。")
    p.add_argument("--freq-ratio", type=int, default=1,
                    help="Ultra-DPC-AEC only：频率压缩倍率（1=未压缩；>1 走 TrainMel 压缩，"
                         "流式 wrapper 原生支持，cache 频点数按 band 数推导）。")
    p.add_argument("--time-skip", type=int, default=1,
                    help="Ultra-DPC-AEC only：时间压缩跳帧（1=未压缩；>1 激活 skip prediction + PostNet，"
                         "流式为逐帧离线回退，PT/ORT 一致但不等价于离线 token 分组）。")
    p.add_argument("--checkpoint", default=None, help="Training weights (random weights if not provided)")
    output_group = p.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output", "-o", default="runs/onnx_debug/model.onnx",
        help="Debug ONNX path under runs/onnx_debug (never a release target)",
    )
    output_group.add_argument(
        "--save-onnx",
        dest="save_onnx",
        default=None,
        help="Compatibility-only: rejected; publish with a dedicated verify_3way script",
    )
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    smoke_group = p.add_mutually_exclusive_group()
    smoke_group.add_argument(
        "--smoke", "--verify", dest="verify", action="store_true",
        help="Run one-step PT/ORT smoke after debug export (not a release gate)",
    )
    smoke_group.add_argument(
        "--no-smoke", "--no-verify", dest="verify", action="store_false",
        help="Skip the optional one-step PT/ORT smoke (default)",
    )
    p.set_defaults(verify=False)
    p.add_argument("--simplify", action="store_true", help="Simplify model with onnxsim")
    p.add_argument("--verify-streaming", action="store_true",
                    help="Additional verification: streaming vs full-sequence consistency")
    p.add_argument("--mse-threshold", type=float, default=1e-4,
                    help="Verification MSE threshold (default 1e-4)")
    p.add_argument("--export-state-detector", action="store_true",
                    help="DBGNet only: export state detector auxiliary output state_probs "
                         "(default: off).")
    return p.parse_args()


def build_streaming_model(
    cfg: dict,
    export_state_detector: bool = False,
) -> torch.nn.Module:
    """Build streaming model. stft_cfg ensures n_fft/hop_length/win_length match config."""
    from tasks.se.models import build_model
    from onnx.models.streaming_bsrnn import StreamingBSRNNWrapper
    from onnx.models.streaming_dbgnet import StreamingDBGNet, StreamingDBGNetFlat
    from tasks.aec.models import register_aec_models

    # AEC models (DBGNet, DeepVQE, ...) are not registered by tasks.se.models.
    register_aec_models()

    stft_cfg = cfg.get("stft")
    model_cfg = cfg.get("model", {})
    # GAN-style config: export generator only
    if isinstance(model_cfg, dict) and "generator" in model_cfg:
        model = build_model(model_cfg["generator"], stft_cfg=stft_cfg)
        model_type = model_cfg["generator"].get("type", "")
    else:
        model = build_model(model_cfg, stft_cfg=stft_cfg)
        model_type = model_cfg.get("type", "")

    # DBGNet: wrap training model into streaming flat wrapper.
    if model_type == "DBGNet":
        stft_cfg = stft_cfg or {}
        data_cfg = cfg.get("data", {}) or {}
        if int(data_cfg.get("sample_rate", model.sample_rate)) != int(model.sample_rate):
            raise ValueError("DBGNet model/data sample_rate mismatch")
        export_use_state = bool(export_state_detector and model.use_state_detector)
        deployment_cfg = copy.deepcopy(cfg)
        deployment_cfg.setdefault("model", {}).setdefault("args", {})[
            "use_state_detector"
        ] = export_use_state
        streaming_net = StreamingDBGNet(
            model,
            window=str(stft_cfg.get("window", "hann")),
            center=bool(stft_cfg.get("center", True)),
            deployment_config=deployment_cfg,
        )
        # 关闭状态检测辅助输出时同步收紧 cache/output ABI 与 metadata binding。
        streaming_net.use_state_detector = export_use_state
        return StreamingDBGNetFlat(streaming_net)

    # Causal BSRNN: training model has no streaming API; wrap for onnx.
    if model_type == "BSRNN":
        if not getattr(getattr(model, "bsrnn", None), "causal", False):
            raise ValueError("Streaming BSRNN requires causal=True in model args.")
        model = StreamingBSRNNWrapper(model, target_fs=getattr(model, "target_fs", 48000))

    is_dbgnet_flat = isinstance(model, StreamingDBGNetFlat)
    has_flat_spec = hasattr(model, "forward_streaming_flat_spec") and hasattr(model, "initialize_cache_flat")
    has_streaming = hasattr(model, "forward_streaming") and hasattr(model, "initialize_cache")
    if not (is_dbgnet_flat or has_flat_spec or has_streaming):
        raise ValueError(
            f"Model {model_type} does not support streaming inference. "
            f"Required: ('forward_streaming_flat_spec' + 'initialize_cache_flat') OR "
            f"('forward_streaming' + 'initialize_cache'). Supported: BSRNN (causal) + wrapper, GTCRN, DBGNet."
        )

    return model


def _resolve_checkpoint_target(model: torch.nn.Module) -> torch.nn.Module:
    """按受控 wrapper 边递归定位训练模型，并拒绝循环引用。"""
    from onnx.models.streaming_bsrnn import StreamingBSRNNWrapper
    from onnx.models.streaming_dbgnet import StreamingDBGNetFlat
    from onnx.models.streaming_tse import StreamingTSEFlat

    target = model
    visited: set[int] = set()
    while True:
        identity = id(target)
        if identity in visited:
            raise RuntimeError("ONNX checkpoint target resolver detected a wrapper cycle")
        visited.add(identity)

        if isinstance(target, torch.nn.parallel.DistributedDataParallel):
            target = target.module
        elif hasattr(target, "_orig_mod"):
            target = target._orig_mod
        elif isinstance(target, StreamingDBGNetFlat):
            target = target.inner.dbgnet
        elif isinstance(target, StreamingBSRNNWrapper):
            target = target.model
        elif isinstance(target, StreamingTSEFlat):
            target = target.model
        else:
            return target
        if not isinstance(target, torch.nn.Module):
            raise RuntimeError("ONNX checkpoint wrapper target must be a torch.nn.Module")


def load_checkpoint(
    model,
    ckpt_path,
    *,
    family: str = "single_model",
):
    """Load a checkpoint with explicit family and fail-closed integrity checks.

    ``safe_torch_load`` is always used in safe (weights_only) mode.
    ``family`` prevents single-model and GAN generator checkpoints from being
    guessed across formats; ambiguous single-model wrapper keys are rejected.
    """
    from core.utils.safe_load import safe_torch_load
    from evaluate.checkpoint_loading import CheckpointFamily, apply_checkpoint

    ckpt = safe_torch_load(ckpt_path)
    try:
        checkpoint_family = CheckpointFamily(family)
    except ValueError as exc:
        raise ValueError(f"Unsupported ONNX checkpoint family: {family!r}") from exc

    target = _resolve_checkpoint_target(model)
    apply_checkpoint(
        target,
        ckpt,
        family=checkpoint_family,
        context=f"checkpoint {ckpt_path}",
    )
    if "ema" in ckpt:
        logger.info("Applied EMA weights: %s", ckpt_path)
    logger.info("Loaded weights: %s", ckpt_path)


def export_model(model, output_path: str, opset: int = 17):
    """Export streaming model to ONNX.

    1. Wrapper-based: BSRNN spec-in/spec-out (forward_streaming_flat_spec + cache).
    2. Direct: StreamingGTCRN with native streaming interface.
    """
    model_type = type(model).__name__

    # direct wrapper 统一走 registry；未注册模型仍沿用下面的通用 BSRNN wrapper。
    direct_exporter = _STREAMING_EXPORTERS.get(model_type)
    if direct_exporter is not None:
        logger.info("Detected %s - using direct ONNX export", model_type)
        return direct_exporter(model, output_path, opset)

    # Default: wrapper-based export (BSRNN + StreamingBSRNNWrapper, spec-in/spec-out)
    from onnx.models.streaming_wrapper import StreamingONNXWrapper

    wrapper = StreamingONNXWrapper(model)
    wrapper.eval()

    batch_size = 1
    cache = wrapper.initialize_cache(batch_size, "cpu", torch.float32)
    F = model.n_fft // 2 + 1
    frame_spec = torch.randn(batch_size, 1, F, 2)

    runtime_spec = runtime_spec_from_wrapper(wrapper)
    input_names = list(runtime_spec.input_names)
    output_names = list(runtime_spec.output_names)
    dynamic_axes = {
        name: dict(axes) for name, axes in runtime_spec.dynamic_axes.items()
    }

    dummy_inputs = (frame_spec, *cache)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting ONNX: opset={opset}, inputs={len(input_names)}, "
                f"outputs={len(output_names)}")
    logger.info(f"  frame_spec shape: {frame_spec.shape}")
    logger.info(f"  cache tensors: {len(cache)}")
    for i, c in enumerate(cache):
        logger.info(f"    cache[{i}] ({input_names[i+1]}): {c.shape}")

    export_onnx(
        wrapper,
        dummy_inputs,
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info(f"ONNX saved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return wrapper, dummy_inputs


def _export_streaming_direct(
    model: torch.nn.Module,
    output_path: str,
    opset: int,
    *,
    dummy_inputs: tuple,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict,
    label: str,
    note: str = "",
) -> tuple[torch.nn.Module, tuple]:
    """Direct ONNX export helper，供全部 streaming flat 模型共用。

    收敛多个高度重复的 ``_export_streaming_*`` 函数（gtcrn / deepvqe / localvqe /
    cagcrn / fastenhancer / ulunas / ultra_dpc / dbgnet）：统一执行 ``torch.onnx.export``
    + 输入/cache shape 日志 + 大小日志；各模型类型只负责构造自己的 dummy inputs
    与 input/output 名。cache shapes 一律由模型 ``init_cache`` 推导，禁止在导出端
    硬编码维度（AGENTS.md 红线）。
    """
    model.eval()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_spec = runtime_spec_from_wrapper(model)
    if tuple(input_names) != runtime_spec.input_names or tuple(output_names) != runtime_spec.output_names:
        raise RuntimeError(
            f"{runtime_spec.profile.model_id} exporter ABI drift: caller inputs={input_names}, "
            f"outputs={output_names}; runtime spec inputs={runtime_spec.input_names}, "
            f"outputs={runtime_spec.output_names}"
        )
    if dict(dynamic_axes) != {name: dict(axes) for name, axes in runtime_spec.dynamic_axes.items()}:
        raise RuntimeError(f"{runtime_spec.profile.model_id} exporter dynamic_axes drift")

    logger.info(f"Exporting {label} ONNX: opset={opset}")
    for i, name in enumerate(runtime_spec.input_names):
        if i < len(dummy_inputs):
            logger.info(f"  {name}: {tuple(dummy_inputs[i].shape)}")

    export_onnx(
        model,
        dummy_inputs,
        str(out_path),
        input_names=list(runtime_spec.input_names),
        output_names=list(runtime_spec.output_names),
        dynamic_axes={name: dict(axes) for name, axes in runtime_spec.dynamic_axes.items()},
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
        dynamo=False,
    )
    logger.info(f"ONNX saved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    if note:
        logger.info(note)
    return model, dummy_inputs


@register_streaming_exporter("StreamingGTCRN")
def _export_streaming_gtcrn(model, output_path: str, opset: int):
    """StreamingGTCRN 直接导出 ONNX（ABI 全部由 wrapper 推导）。"""
    batch_size = 1
    device = torch.device("cpu")
    dtype = torch.float32
    runtime_spec = runtime_spec_from_wrapper(model)
    n_fft = runtime_spec.stft.n_fft
    if n_fft is None:
        raise RuntimeError("StreamingGTCRN runtime spec must declare n_fft")
    n_freq = n_fft // 2 + 1

    cache = model.init_cache(batch_size, device, dtype)
    if not isinstance(cache, dict):
        raise TypeError("StreamingGTCRN init_cache() must return a cache mapping")
    try:
        cache_tensors = tuple(cache[name] for name in runtime_spec.cache_input_names)
    except KeyError as exc:
        raise RuntimeError(
            f"StreamingGTCRN cache initializer is missing {exc.args[0]!r}"
        ) from exc
    spec = torch.randn(batch_size, n_freq, 1, 2, device=device, dtype=dtype)

    return _export_streaming_direct(
        model,
        output_path,
        opset,
        dummy_inputs=(spec, *cache_tensors),
        input_names=list(runtime_spec.input_names),
        output_names=list(runtime_spec.output_names),
        dynamic_axes={
            name: dict(axes) for name, axes in runtime_spec.dynamic_axes.items()
        },
        label="StreamingGTCRN",
    )


@register_streaming_exporter("StreamingDeepVQEFlat", "StreamingLocalVQEFlat")
def _export_streaming_dual_input_aec_flat(model, output_path: str, opset: int):
    """导出 mic/far 双输入 flat wrapper；dummy/cache ABI 全部由 wrapper 推导。"""
    inner = getattr(model, "inner", None)
    if inner is None:
        raise RuntimeError(
            f"dual-input AEC export expects model.inner, got {type(model).__name__}"
        )
    batch_size = 1
    device = torch.device("cpu")
    dtype = torch.float32
    runtime_spec = runtime_spec_from_wrapper(model)
    n_fft = runtime_spec.stft.n_fft
    if n_fft is None:
        raise RuntimeError(f"{type(model).__name__} runtime spec must declare n_fft")
    n_freq = n_fft // 2 + 1
    mic_spec = torch.randn(batch_size, n_freq, 1, 2, device=device, dtype=dtype)
    far_spec = torch.randn(batch_size, n_freq, 1, 2, device=device, dtype=dtype)
    cache_tensors = model.init_cache_flat(batch_size, device, dtype)

    return _export_streaming_direct(
        model,
        output_path,
        opset,
        dummy_inputs=(mic_spec, far_spec, *cache_tensors),
        input_names=list(runtime_spec.input_names),
        output_names=list(runtime_spec.output_names),
        dynamic_axes={
            name: dict(axes) for name, axes in runtime_spec.dynamic_axes.items()
        },
        label=type(model).__name__,
    )


@register_streaming_exporter("StreamingCAGCRNFlat")
def _export_streaming_cagcrn(model, output_path: str, opset: int):
    """StreamingCAGCRNFlat 直接导出 ONNX。"""
    batch_size = 1
    device = "cpu"
    dtype = torch.float32

    n_fft = getattr(getattr(model, "model", None), "n_fft", 512)
    n_freq = n_fft // 2 + 1

    mic_spec = torch.randn(batch_size, n_freq, 1, 2, device=device, dtype=dtype)
    ref_spec = torch.randn(batch_size, n_freq, 1, 2, device=device, dtype=dtype)
    cache_tensors = model.init_cache_flat(batch_size, device, dtype)
    dummy_inputs = (mic_spec, ref_spec) + tuple(cache_tensors)

    return _export_streaming_direct(
        model,
        output_path,
        opset,
        dummy_inputs=dummy_inputs,
        input_names=model.get_input_names(),
        output_names=model.get_output_names(),
        dynamic_axes=model.get_dynamic_axes(),
        label="StreamingCAGCRNFlat",
    )


@register_streaming_exporter("StreamingFastEnhancerFlat")
def _export_streaming_fastenhancer(model, output_path: str, opset: int):
    """Direct ONNX export for StreamingFastEnhancerFlat (spec-in / spec-out, frame-by-frame)."""
    inner = model.inner
    batch_size = 1
    device = "cpu"
    dtype = torch.float32

    n_fft = inner.n_fft
    n_freq = n_fft // 2 + 1

    spec = torch.randn(batch_size, n_freq, 1, 2, device=device, dtype=dtype)
    cache = inner.init_cache(batch_size, torch.device(device), dtype)
    dummy_inputs = (spec,) + tuple(cache[k] for k in inner.CACHE_KEYS)

    return _export_streaming_direct(
        model,
        output_path,
        opset,
        dummy_inputs=dummy_inputs,
        input_names=model.get_input_names(),
        output_names=model.get_output_names(),
        dynamic_axes=model.get_dynamic_axes(),
        label="StreamingFastEnhancerFlat",
    )


@register_streaming_exporter("StreamingULUNASFlat")
def _export_streaming_ulunas(model, output_path: str, opset: int = 17):
    """Direct ONNX export for ``StreamingULUNASFlat`` using official ABI caches."""
    batch_size = 1
    device = torch.device("cpu")
    dtype = torch.float32
    n_freq = model.inner.n_fft // 2 + 1
    spec = torch.randn(batch_size, n_freq, 1, 2, device=device, dtype=dtype)
    cache = model.init_cache_flat(batch_size, device, dtype)
    return _export_streaming_direct(
        model,
        output_path,
        opset,
        dummy_inputs=(spec, *cache),
        input_names=model.get_input_names(),
        output_names=model.get_output_names(),
        dynamic_axes=model.get_dynamic_axes(),
        label="StreamingULUNASFlat",
    )


@register_streaming_exporter("StreamingUltraDPCAECFlat")
def _export_streaming_ultra_dpc(model, output_path: str, opset: int = 17):
    """Direct ONNX export for StreamingUltraDPCAECFlat (mic/ref/err spec-in, enhanced spec-out)."""
    inner = model.inner
    batch_size = 1
    device = "cpu"
    dtype = torch.float32

    n_fft = inner.n_fft
    n_freq = n_fft // 2 + 1

    mic = torch.randn(batch_size, n_freq, 1, 2, device=device, dtype=dtype)
    ref = torch.randn(batch_size, n_freq, 1, 2, device=device, dtype=dtype)
    err = torch.randn(batch_size, n_freq, 1, 2, device=device, dtype=dtype)
    cache_tensors = model.init_cache_flat(batch_size, torch.device(device), dtype)
    dummy_inputs = (mic, ref, err) + tuple(cache_tensors)

    return _export_streaming_direct(
        model,
        output_path,
        opset,
        dummy_inputs=dummy_inputs,
        input_names=model.get_input_names(),
        output_names=model.get_output_names(),
        dynamic_axes=model.get_dynamic_axes(),
        label="StreamingUltraDPCAECFlat",
    )


@register_streaming_exporter("StreamingDBGNetFlat")
def _export_streaming_dbgnet(
    model: StreamingDBGNetFlat,
    output_path: str,
    opset: int,
):
    """Direct ONNX export for ``StreamingDBGNetFlat`` (mic/ref spec-in, enh/state-out)."""
    batch_size = 1
    device = "cpu"
    dtype = torch.float32

    n_freq = model.nfreqs
    mic_spec = torch.randn(batch_size, n_freq, 1, 2, device=device, dtype=dtype)
    far_spec = torch.randn(batch_size, n_freq, 1, 2, device=device, dtype=dtype)
    cache = model.init_cache(batch_size, torch.device(device), dtype)
    dummy_inputs = (mic_spec, far_spec) + cache

    input_names = model.get_input_names()
    output_names = model.get_output_names()
    dynamic_axes = model.get_dynamic_axes()

    return _export_streaming_direct(
        model,
        output_path,
        opset,
        dummy_inputs=dummy_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        label="StreamingDBGNetFlat",
    )


@register_streaming_exporter("StreamingTSEFlat")
def _export_streaming_tse(model, output_path: str, opset: int = 17):
    """导出 TSE 单谱帧 + enrollment loopback + temporal cache flat ABI。"""
    batch_size = 1
    device = torch.device("cpu")
    dtype = torch.float32
    frame = torch.randn(
        batch_size, model.n_fft // 2 + 1, 1, 2, device=device, dtype=dtype
    )
    cache = model.init_cache_flat(batch_size, device, dtype)
    return _export_streaming_direct(
        model,
        output_path,
        opset,
        dummy_inputs=(frame, *cache),
        input_names=model.get_input_names(),
        output_names=model.get_output_names(),
        dynamic_axes=model.get_dynamic_axes(),
        label="StreamingTSEFlat",
    )


@register_streaming_exporter("StreamingOnlineSpatialNetFlat")
def _export_streaming_online_spatialnet(model, output_path: str, opset: int = 17):
    """导出真实 OnlineSpatialNet backend 的多通道 observation flat ABI。"""
    batch_size = 1
    device = torch.device("cpu")
    dtype = torch.float32
    runtime_spec = runtime_spec_from_wrapper(model, "online_spatialnet")
    if runtime_spec.stft.n_fft is None:
        raise RuntimeError("OnlineSpatialNet runtime spec must declare n_fft")
    frame = torch.randn(
        batch_size,
        model.n_channel,
        runtime_spec.stft.n_fft // 2 + 1,
        1,
        2,
        device=device,
        dtype=dtype,
    )
    cache = model.init_cache_flat(batch_size, device, dtype)
    return _export_streaming_direct(
        model,
        output_path,
        opset,
        dummy_inputs=(frame, *cache),
        input_names=list(runtime_spec.input_names),
        output_names=list(runtime_spec.output_names),
        dynamic_axes={
            name: dict(axes) for name, axes in runtime_spec.dynamic_axes.items()
        },
        label="StreamingOnlineSpatialNetFlat",
    )


def verify_onnx(onnx_path: str, model_or_wrapper, dummy_inputs, mse_threshold: float):
    """Verify ONNX model output consistency with PyTorch."""
    try:
        sess = create_ort_session(onnx_path)
    except RuntimeError as exc:
        logger.error("ONNX Runtime verification unavailable: %s", exc)
        return False

    # Wrapper and direct streaming models must both expose the authoritative ABI.
    if not all(
        callable(getattr(model_or_wrapper, name, None))
        for name in ("get_input_names", "get_output_names")
    ):
        logger.error(
            "Streaming model %s does not expose get_input_names/get_output_names",
            type(model_or_wrapper).__name__,
        )
        return False
    input_names = list(model_or_wrapper.get_input_names())
    output_names = list(model_or_wrapper.get_output_names())

    actual_input_names = [value.name for value in sess.get_inputs()]
    actual_output_names = [value.name for value in sess.get_outputs()]
    if actual_input_names != input_names or actual_output_names != output_names:
        logger.error(
            "ONNX ABI mismatch: expected inputs=%s, outputs=%s; got inputs=%s, outputs=%s",
            input_names,
            output_names,
            actual_input_names,
            actual_output_names,
        )
        return False
    if len(dummy_inputs) != len(input_names):
        logger.error(
            "ONNX verification input count mismatch: expected %d tensors, got %d",
            len(input_names),
            len(dummy_inputs),
        )
        return False

    model_or_wrapper.eval()
    with torch.inference_mode():
        pt_outputs = model_or_wrapper(*dummy_inputs)
    if isinstance(pt_outputs, torch.Tensor):
        pt_outputs = (pt_outputs,)
    else:
        pt_outputs = tuple(pt_outputs)
    if len(pt_outputs) != len(output_names):
        logger.error(
            "PyTorch output count mismatch: expected %d tensors, got %d",
            len(output_names),
            len(pt_outputs),
        )
        return False

    feeds = {
        name: value.detach().numpy()
        for name, value in zip(input_names, dummy_inputs)
    }
    ort_outputs = sess.run(None, feeds)
    if len(ort_outputs) != len(output_names):
        logger.error(
            "ONNX output count mismatch: expected %d tensors, got %d",
            len(output_names),
            len(ort_outputs),
        )
        return False

    logger.info("Precision verification:")
    all_pass = True
    empty_cache_outputs = {
        f"{name}_out" for name in input_names[1:]
    } if type(model_or_wrapper).__name__ == "StreamingOnlineSpatialNetFlat" else set()
    for i, name in enumerate(output_names):
        pt_val = pt_outputs[i].detach().numpy()
        ort_val = ort_outputs[i]
        if pt_val.shape != ort_val.shape:
            logger.error(
                "ONNX output shape mismatch for %s: PyTorch=%s, ONNX=%s",
                name,
                pt_val.shape,
                ort_val.shape,
            )
            all_pass = False
            continue
        if pt_val.size == 0:
            if name in empty_cache_outputs and ort_val.size == 0:
                logger.info("  %s: exact empty cache shape=%s [PASS]", name, pt_val.shape)
                continue
            logger.error("ONNX output is unexpectedly empty for %s: shape=%s", name, pt_val.shape)
            all_pass = False
            continue
        if not np.all(np.isfinite(pt_val)) or not np.all(np.isfinite(ort_val)):
            logger.error(
                "ONNX output contains non-finite values for %s: shape=%s",
                name,
                pt_val.shape,
            )
            all_pass = False
            continue
        mse = np.mean((pt_val - ort_val) ** 2)
        max_diff = np.max(np.abs(pt_val - ort_val))
        # MSE 必须为有限值，否则验证失败。
        mse_ok = bool(np.isfinite(mse)) and mse < mse_threshold
        status = "PASS" if mse_ok else "FAIL"
        if not mse_ok:
            all_pass = False
        logger.info(f"  {name}: MSE={mse:.2e}, max_diff={max_diff:.2e} [{status}]")

    if all_pass:
        logger.info("Verification passed: all outputs MSE < %.0e", mse_threshold)
    else:
        logger.error("Verification failed: some outputs exceed MSE threshold")
    return all_pass


def verify_streaming_consistency(model, mse_threshold: float):
    """Verify consistency between streaming and full-sequence inference (STFT outside model)."""
    from onnx.models.streaming_wrapper import StreamingInferenceWrapper
    from core.stft import build_stft_istft

    model.eval()
    stft_cfg = {
        "n_fft": model.n_fft,
        "hop_length": model.hop_length,
        "win_length": getattr(model, "win_length", model.n_fft),
        "window": "hann",
    }
    stft_module = build_stft_istft(stft_cfg)
    wrapper = StreamingInferenceWrapper(stft_module, model, device="cpu")

    test_len = getattr(model, "target_fs", 16000) * 2
    wav = torch.randn(1, test_len)

    result = wrapper.verify_consistency(wav, mse_tol=mse_threshold)
    logger.info("Streaming consistency verification:")
    logger.info(f"  MSE: {result['mse']:.2e}")
    logger.info(f"  max_abs_diff: {result['max_abs_diff']:.2e}")
    logger.info(f"  Result: {'PASS' if result['pass'] else 'FAIL'}")
    return result["pass"]


def simplify_onnx(onnx_path: str):
    """Simplify model with onnxsim inside the external ONNX import scope."""
    try:
        with pypi_onnx_scope() as onnx:
            from onnxsim import simplify

            model = onnx.load(str(onnx_path))
            model_simp, check = simplify(model)
            if check:
                onnx.save(model_simp, str(onnx_path))
                size_mb = Path(onnx_path).stat().st_size / 1024 / 1024
                logger.info("onnxsim simplification done: %s (%.1f MB)", onnx_path, size_mb)
            else:
                logger.warning("onnxsim simplification failed, keeping original model")
    except ImportError:
        logger.warning("onnxsim not installed, skipping simplification. pip install onnxsim")


def write_streaming_runtime_metadata(onnx_path: str, wrapper_or_model) -> dict[str, object]:
    """从 ``StreamingRuntimeSpec`` 一次写入模型的完整稳定 metadata。"""
    runtime_spec = runtime_spec_from_wrapper(wrapper_or_model)
    metadata = dict(runtime_spec.metadata)
    if metadata:
        write_onnx_metadata(onnx_path, metadata)
        logger.info(
            "Embedded streaming runtime metadata %s (isolated subprocess)",
            sorted(metadata),
        )
    return metadata


def _verify_or_exit(args, onnx_path: str, wrapper_or_model, dummy_inputs) -> None:
    """精度验证门控：对指定 ONNX 产物执行 verify，失败即 ``raise SystemExit(1)``。

    统一门控：验证失败时不打印 Done、以非零码退出。
    """
    try:
        ok = verify_onnx(onnx_path, wrapper_or_model, dummy_inputs, args.mse_threshold)
        if ok and type(wrapper_or_model).__name__ == "StreamingOnlineSpatialNetFlat":
            from onnx.runtime_spec import runtime_spec_from_session

            runtime_spec_from_session(
                create_ort_session(onnx_path), "online_spatialnet"
            )
    except Exception as e:
        # 验证异常同样视为失败。
        logger.error("Verification failed: %s: %s", type(e).__name__, e)
        raise SystemExit(1)
    if not ok:
        logger.error("Verification failed, aborting export.")
        raise SystemExit(1)


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


class _OwnedOnnxArtifact(os.PathLike):
    """由稳定 parent dirfd 与初始 inode 共同锚定的 owned ONNX 临时文件。"""

    def __init__(self, parent_fd: int, file_fd: int, name: str, display_path: Path):
        self._parent_fd = parent_fd
        self._file_fd = file_fd
        self._owner_pid = os.getpid()
        self.name = name
        self.display_path = display_path

    def __fspath__(self) -> str:
        if self._parent_fd < 0:
            raise RuntimeError("owned ONNX artifact is closed")
        # metadata helper 子进程必须通过持有 dirfd 的父进程访问临时产物。
        return f"/proc/{self._owner_pid}/fd/{self._parent_fd}/{self.name}"

    def _same_identity(self, actual: os.stat_result) -> bool:
        expected = os.fstat(self._file_fd)
        return (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)

    def close(self) -> None:
        if self._parent_fd < 0:
            return
        quarantine = f".cleanup_{secrets.token_hex(12)}.onnx"
        try:
            try:
                os.replace(
                    self.name,
                    quarantine,
                    src_dir_fd=self._parent_fd,
                    dst_dir_fd=self._parent_fd,
                )
            except FileNotFoundError:
                pass
            else:
                actual = os.stat(quarantine, dir_fd=self._parent_fd, follow_symlinks=False)
                if self._same_identity(actual):
                    os.unlink(quarantine, dir_fd=self._parent_fd)
                else:
                    os.replace(
                        quarantine,
                        self.name,
                        src_dir_fd=self._parent_fd,
                        dst_dir_fd=self._parent_fd,
                    )
        finally:
            os.close(self._file_fd)
            os.close(self._parent_fd)
            self._file_fd = -1
            self._parent_fd = -1


class _DebugOutputTarget:
    """在 precheck 时固定最终 parent dirfd 的 debug 输出目标。"""

    def __init__(self, parent_fd: int, name: str, display_path: Path, *, require_new: bool):
        self.parent_fd = parent_fd
        self.name = name
        self.display_path = display_path
        self.require_new = require_new

    def close(self) -> None:
        if self.parent_fd >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1


def _open_or_create_child_dir(parent_fd: int, name: str) -> int:
    """只通过 dirfd 创建并打开一级真实目录，拒绝 symlink。"""
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError(f"Invalid debug directory component: {name!r}")
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
        os.close(child_fd)
        raise NotADirectoryError(name)
    return child_fd


def _open_debug_root(*, include_tmp: bool = False) -> tuple[int, Path]:
    """从 project root 逐级 ``O_NOFOLLOW`` 打开 debug 根。"""
    project_root = Path(_project_root).absolute()
    current_fd = os.open(project_root, _DIRECTORY_FLAGS)
    try:
        for component in ("runs", "onnx_debug") + ((".tmp",) if include_tmp else ()):
            next_fd = _open_or_create_child_dir(current_fd, component)
            os.close(current_fd)
            current_fd = next_fd
    except Exception:
        os.close(current_fd)
        raise
    suffix = ("runs", "onnx_debug", ".tmp") if include_tmp else ("runs", "onnx_debug")
    return current_fd, project_root.joinpath(*suffix)


def _create_owned_onnx_artifact() -> _OwnedOnnxArtifact:
    """在稳定且无 symlink 的 debug ``.tmp`` 下创建 owned 文件。"""
    parent_fd, owned_root = _open_debug_root(include_tmp=True)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW
    try:
        for _ in range(128):
            name = f".export_{secrets.token_hex(12)}.onnx"
            try:
                file_fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
            except FileExistsError:
                continue
            return _OwnedOnnxArtifact(parent_fd, file_fd, name, owned_root / name)
    except Exception:
        os.close(parent_fd)
        raise
    os.close(parent_fd)
    raise RuntimeError("Unable to allocate a unique owned ONNX artifact")


def _prepare_debug_output_target(
    target: str | Path,
    *,
    require_new: bool = False,
) -> _DebugOutputTarget:
    """将目标限制在 debug 根内，并在返回前固定最终 parent dirfd。"""
    candidate = Path(target).expanduser()
    if not candidate.is_absolute():
        candidate = Path(_project_root) / candidate
    candidate = Path(os.path.abspath(candidate))
    debug_fd, debug_root = _open_debug_root()
    try:
        relative = candidate.relative_to(debug_root)
        if not relative.parts or relative.suffix.lower() != ".onnx":
            raise ValueError
    except ValueError as exc:
        os.close(debug_fd)
        raise ValueError(
            f"Debug ONNX output must be a .onnx file under {debug_root}"
        ) from exc

    current_fd = debug_fd
    try:
        for component in relative.parts[:-1]:
            next_fd = _open_or_create_child_dir(current_fd, component)
            os.close(current_fd)
            current_fd = next_fd
        name = relative.parts[-1]
        if require_new:
            try:
                os.stat(name, dir_fd=current_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError(f"Debug ONNX target already exists: {candidate}")
        return _DebugOutputTarget(current_fd, name, candidate, require_new=require_new)
    except Exception:
        os.close(current_fd)
        raise


def _copy_fd(source_fd: int, target_fd: int) -> None:
    os.lseek(source_fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(source_fd, 1024 * 1024)
        if not chunk:
            break
        view = memoryview(chunk)
        while view:
            written = os.write(target_fd, view)
            view = view[written:]


def _publish_debug_onnx(
    source: str | Path | _OwnedOnnxArtifact,
    target: _DebugOutputTarget,
) -> None:
    """仅通过 precheck 固定的 dirfd 原子发布 debug artifact。"""
    if not isinstance(target, _DebugOutputTarget) or target.parent_fd < 0:
        raise TypeError("debug target must be prepared with _prepare_debug_output_target")

    source_fd = os.open(source, os.O_RDONLY | _FILE_NOFOLLOW)
    staging_fd = -1
    staging_name = ""
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("debug ONNX source must be a regular file")
        if isinstance(source, _OwnedOnnxArtifact) and not source._same_identity(source_stat):
            raise RuntimeError("owned debug ONNX artifact identity changed before publication")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW
        for _ in range(128):
            staging_name = f".{target.name}.{secrets.token_hex(12)}.tmp"
            try:
                staging_fd = os.open(staging_name, flags, 0o600, dir_fd=target.parent_fd)
            except FileExistsError:
                continue
            break
        else:
            raise RuntimeError("Unable to allocate debug ONNX staging file")
        _copy_fd(source_fd, staging_fd)
        os.fsync(staging_fd)
        os.close(staging_fd)
        staging_fd = -1
        if target.require_new:
            os.link(
                staging_name,
                target.name,
                src_dir_fd=target.parent_fd,
                dst_dir_fd=target.parent_fd,
                follow_symlinks=False,
            )
            os.unlink(staging_name, dir_fd=target.parent_fd)
            staging_name = ""
        else:
            os.replace(
                staging_name,
                target.name,
                src_dir_fd=target.parent_fd,
                dst_dir_fd=target.parent_fd,
            )
            staging_name = ""
        os.fsync(target.parent_fd)
    finally:
        if staging_fd >= 0:
            os.close(staging_fd)
        os.close(source_fd)
        if staging_name:
            try:
                os.unlink(staging_name, dir_fd=target.parent_fd)
            except FileNotFoundError:
                pass
        if isinstance(source, _OwnedOnnxArtifact):
            source.close()


def create_owned_onnx_artifact() -> _OwnedOnnxArtifact:
    """公共入口：在 debug ``.tmp`` 下创建 owned ONNX 临时文件。"""
    return _create_owned_onnx_artifact()


def prepare_debug_output_target(
    target: str | Path,
    *,
    require_new: bool = False,
) -> _DebugOutputTarget:
    """公共入口：把 debug 输出限制在 debug 根内并固定最终 parent dirfd。"""
    return _prepare_debug_output_target(target, require_new=require_new)


def publish_debug_onnx(
    source: str | Path | _OwnedOnnxArtifact,
    target: _DebugOutputTarget,
) -> None:
    """公共入口：仅通过 precheck 固定的 dirfd 原子发布 debug artifact。"""
    return _publish_debug_onnx(source, target)


def _dedicated_verifier_hint(model: str | None, config: str | None) -> str:
    """返回 generic 发布拒绝消息中的专属 verifier 提示。"""
    normalized = (model or "").lower().removeprefix("streaming_").replace("-", "_")
    if normalized in {"online_spatialnet", "online_spatial"}:
        return (
            "no dedicated OnlineSpatialNet verifier is implemented yet; "
            "keep using generic --output debug artifacts only"
        )
    dedicated = {
        "cagcrn", "dbgnet", "deepvqe", "fastenhancer", "gtcrn", "ultra_dpc",
        "tse", "ulunas",
    }
    if normalized in dedicated:
        return f"onnx/verify/verify_3way_{normalized}.py"
    if config:
        return "the model-specific onnx/verify/verify_3way_*.py"
    return "a model-specific onnx/verify/verify_3way_*.py"


def main():
    args = parse_args()

    save_onnx = getattr(args, "save_onnx", None)
    release_requested = save_onnx is not None
    if release_requested:
        verifier = _dedicated_verifier_hint(args.model, args.config)
        if verifier.startswith("no dedicated"):
            raise ValueError(
                "generic onnx/export.py --save-onnx cannot publish release artifacts; "
                f"{verifier}"
            )
        raise ValueError(
            "generic onnx/export.py --save-onnx cannot publish release artifacts; "
            f"run {verifier} with --checkpoint and --save-onnx so dedicated A/B/C "
            "verification can call publish_verified_onnx"
        )
    if not args.config and not args.model:
        raise ValueError("Either --config or --model must be provided")
    config_required_models = {
        "streaming_localvqe", "streaming_deepvqe", "streaming_tse",
        "streaming_online_spatialnet", "online_spatialnet", "online_spatial",
    }
    if (args.model or "").lower() in config_required_models and not args.config:
        raise ValueError(
            f"{args.model.lower()} requires --config so architecture, STFT, "
            "sample-rate, and frontend semantics cannot drift from the checkpoint"
        )
    if args.output is None:
        args.output = "runs/onnx_debug/model.onnx"

    streaming_verification_passed = True

    # Mode 1: Direct model type (e.g., StreamingGTCRN)
    if args.model:
        logger.info(f"Direct model export mode: {args.model}")
        if args.model.lower() == "streaming_gtcrn":
            from onnx.models.streaming_gtcrn import StreamingGTCRN, convert_gtcrn_to_streaming
            model = StreamingGTCRN().eval()
            
            if args.checkpoint:
                # Load GTCRN checkpoint and convert weights with fail-closed checks.
                from tasks.se.models.discriminative.gtcrn import GTCRN
                gtcrn_model = GTCRN()
                load_checkpoint(gtcrn_model, args.checkpoint)

                # Convert to streaming
                convert_gtcrn_to_streaming(model, gtcrn_model)
                logger.info("Converted GTCRN weights to StreamingGTCRN")
        elif args.model.lower() == "streaming_deepvqe":
            from core.stft import inject_stft_defaults
            from tasks.aec.models.discriminative.deepvqe import DeepVQE
            from onnx.models.streaming_deepvqe import StreamingDeepVQE, StreamingDeepVQEFlat

            cfg = load_merged_config(args.config)
            model_cfg = cfg.get("model", {}) or {}
            if model_cfg.get("type") != "DeepVQE":
                raise ValueError(
                    "streaming_deepvqe expects model.type=DeepVQE, got "
                    f"{model_cfg.get('type')!r}"
                )
            stft_cfg = cfg.get("stft", {}) or {}
            data_cfg = cfg.get("data", {}) or {}
            model_args = inject_stft_defaults(
                dict(model_cfg.get("args", {}) or {}), stft_cfg
            )
            base_model = DeepVQE(**model_args).eval()
            if args.checkpoint:
                load_checkpoint(base_model, args.checkpoint)
            stream = StreamingDeepVQE(
                base_model,
                win_length=int(stft_cfg["win_length"]),
                target_fs=int(data_cfg["sample_rate"]),
                window=str(stft_cfg.get("window", "hann")),
                center=bool(stft_cfg.get("center", True)),
                deployment_config=cfg,
            ).eval()
            model = StreamingDeepVQEFlat(stream).eval()
            logger.info(
                "Built config-bound StreamingDeepVQEFlat(%s, n_fft=%d, hop=%d, "
                "align=%d, compression=%s)",
                stream.variant, stream.n_fft, stream.hop_length,
                stream.alignment_window_frames, stream.compression,
            )
        elif args.model.lower() == "streaming_localvqe":
            from core.stft import inject_stft_defaults
            from tasks.aec.models.discriminative.localvqe import LocalVQE
            from onnx.models.streaming_localvqe import (
                StreamingLocalVQE,
                StreamingLocalVQEFlat,
            )

            model_args = {}
            target_fs = 16000
            if args.config:
                cfg = load_merged_config(args.config)
                model_cfg = cfg.get("model", {})
                if model_cfg.get("type") != "LocalVQE":
                    raise ValueError(
                        "streaming_localvqe expects a config with model.type "
                        f"LocalVQE, got {model_cfg.get('type')!r}"
                    )
                model_args = inject_stft_defaults(
                    dict(model_cfg.get("args", {}) or {}),
                    cfg.get("stft", {}) or {},
                )
                target_fs = int(cfg.get("data", {}).get("sample_rate", target_fs))
            base_model = LocalVQE(**model_args).eval()
            if args.checkpoint:
                load_checkpoint(base_model, args.checkpoint)
            stream = StreamingLocalVQE(base_model, target_fs=target_fs).eval()
            model = StreamingLocalVQEFlat(stream).eval()
            logger.info(
                "Built StreamingLocalVQEFlat from LocalVQE checkpoint/config "
                f"(n_fft={stream.n_fft}, hop={stream.hop_length}, "
                f"arch_version={stream.arch_version}, delay=0 samples)"
            )
        elif args.model.lower() == "streaming_cagcrn":
            from tasks.aec.models.discriminative.cagcrn import CAGCRN
            from onnx.models.streaming_cagcrn import StreamingCAGCRN, StreamingCAGCRNFlat

            base_model = CAGCRN().eval()
            if args.checkpoint:
                load_checkpoint(base_model, args.checkpoint)
            stream = StreamingCAGCRN(base_model).eval()
            model = StreamingCAGCRNFlat(stream).eval()
            logger.info("Built StreamingCAGCRNFlat from CAGCRN checkpoint")
        elif args.model.lower() == "streaming_fastenhancer":
            from tasks.se.models.discriminative.fastenhancer import FastEnhancer
            from onnx.models.streaming_fastenhancer import StreamingFastEnhancer, StreamingFastEnhancerFlat

            # variant 优先级：显式参数、config、checkpoint 文件名。
            variant = args.variant
            n_fft = 512
            if args.config:
                cfg = load_merged_config(args.config)
                model_cfg = cfg.get("model", {})
                variant = variant or model_cfg.get("args", {}).get("variant")
                n_fft = cfg.get("stft", {}).get("n_fft", n_fft)
            if variant is None and args.checkpoint and "fastenhancer_" in str(args.checkpoint).lower():
                name = str(args.checkpoint).lower()
                for v in ["t", "b", "s", "m", "l"]:
                    if f"fastenhancer_{v}" in name or f"fastenhancer{v}" in name:
                        variant = v.upper()
                        break
            variant = (variant or "B").upper()
            base_model = FastEnhancer(variant=variant, n_fft=n_fft).eval()
            if args.checkpoint:
                load_checkpoint(base_model, args.checkpoint)
            stft_cfg = cfg.get("stft", {}) if args.config else {}
            stream = StreamingFastEnhancer(
                base_model,
                hop_length=int(stft_cfg.get("hop_length", 256)),
                win_length=int(stft_cfg.get("win_length", n_fft)),
                sample_rate=int((cfg.get("data", {}) or {}).get("sample_rate", 16000))
                if args.config else 16000,
                window=str(stft_cfg.get("window", "hann")),
                center=bool(stft_cfg.get("center", True)),
            ).eval()
            model = StreamingFastEnhancerFlat(stream).eval()
            logger.info(
                f"Built StreamingFastEnhancerFlat from FastEnhancer checkpoint "
                f"(variant={variant}, n_fft={n_fft})"
            )
        elif args.model.lower() == "streaming_ulunas":
            from tasks.se.models.discriminative.ulunas import ULUNAS
            from onnx.models.streaming_ulunas import StreamingULUNAS, StreamingULUNASFlat

            base_model = ULUNAS(n_fft=512).eval()
            if args.checkpoint:
                load_checkpoint(base_model, args.checkpoint)
            stream = StreamingULUNAS(base_model).eval()
            model = StreamingULUNASFlat(stream).eval()
            logger.info("Built StreamingULUNASFlat from ULUNAS checkpoint")
        elif args.model.lower() == "streaming_ultra_dpc":
            from tasks.aec.models.discriminative.ultra_dpc_aec import UltraDPCAEC
            from onnx.models.streaming_ultra_dpc import StreamingUltraDPCAEC, StreamingUltraDPCAECFlat

            base_model = UltraDPCAEC(n_fft=320, hop_length=160, embed_dim=48, n_dp_blocks=4,
                                     encoder_layers=2, decoder_layers=2, gru_dim=32,
                                     freq_ratio=args.freq_ratio, time_skip=args.time_skip).eval()
            if args.checkpoint:
                load_checkpoint(base_model, args.checkpoint)
            stream = StreamingUltraDPCAEC(base_model).eval()
            model = StreamingUltraDPCAECFlat(stream).eval()
            logger.info(
                f"Built StreamingUltraDPCAECFlat from UltraDPCAEC checkpoint "
                f"(freq_ratio={args.freq_ratio}, time_skip={args.time_skip})"
            )
        elif args.model.lower() in {
            "streaming_online_spatialnet", "online_spatialnet", "online_spatial",
        }:
            from onnx.models.streaming_online_spatialnet import StreamingOnlineSpatialNetFlat
            from tasks.se.models import build_model

            cfg = load_merged_config(args.config)
            model_cfg = copy.deepcopy(cfg.get("model", {}) or {})
            if model_cfg.get("type") != "OnlineSpatialNet_SE":
                raise ValueError(
                    "streaming_online_spatialnet expects model.type=OnlineSpatialNet_SE, got "
                    f"{model_cfg.get('type')!r}"
                )
            attention = str((model_cfg.get("args") or {}).get("attention") or "")
            if attention.strip().lower().startswith("mamba"):
                raise ValueError(
                    "streaming_online_spatialnet cannot export the mamba branch: the "
                    "selective scan relies on mamba_ssm/Triton kernels without an ONNX ABI; "
                    "use attention='mhsa(scope)' or attention='ret(factor,share_qk)'"
                )
            base_model = build_model(model_cfg, stft_cfg=cfg.get("stft")).eval()
            if args.checkpoint:
                load_checkpoint(base_model, args.checkpoint)
            model = StreamingOnlineSpatialNetFlat(base_model).eval()
            logger.info(
                "Built StreamingOnlineSpatialNetFlat(C=%d, n_fft=%d, caches=%d)",
                model.n_channel,
                model.n_fft,
                len(model.CACHE_KEYS),
            )
        elif args.model.lower() == "streaming_tse":
            from core.models.build import build_model
            from onnx.models.streaming_tse import StreamingTSEFlat
            from tasks.tse.models import register_tse_models

            cfg = load_merged_config(args.config)
            model_cfg = copy.deepcopy(cfg.get("model", {}) or {})
            if model_cfg.get("type") != "TSE_BSRNN_SPK":
                raise ValueError(
                    "streaming_tse expects model.type=TSE_BSRNN_SPK, got "
                    f"{model_cfg.get('type')!r}"
                )
            if args.checkpoint:
                speaker_cfg = (
                    (model_cfg.get("args") or {}).get("speaker_model") or {}
                ).get("speaker_encoder") or {}
                if isinstance(speaker_cfg, dict):
                    speaker_cfg["pretrained"] = None
            register_tse_models(only="TSE_BSRNN_SPK")
            base_model = build_model(model_cfg, stft_cfg=cfg.get("stft")).eval()
            if not getattr(base_model, "causal", False):
                raise ValueError("streaming_tse requires model.args.causal=true")
            if args.checkpoint:
                load_checkpoint(base_model, args.checkpoint)
            model = StreamingTSEFlat(
                base_model,
                enrollment_samples=args.tse_enrollment_samples,
            ).eval()
            logger.info(
                "Built StreamingTSEFlat(cues=%s, caches=%d, delay=%d samples)",
                tuple(
                    name for name, value in base_model.features_config.items()
                    if value["enabled"]
                ),
                len(model.CACHE_KEYS),
                model.waveform_output_delay_samples,
            )
        else:
            raise ValueError(f"Unsupported model type for direct export: {args.model}")
        
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Model: {args.model} | {n_params:.2f}M params")
        
    # Mode 2: Config-based (wrapper export)
    else:
        cfg = load_merged_config(args.config)
        model = build_streaming_model(
            cfg,
            export_state_detector=args.export_state_detector,
        )
        if args.checkpoint:
            model_cfg = cfg.get("model", {})
            checkpoint_family = (
                "gan_generator"
                if isinstance(model_cfg, dict) and "generator" in model_cfg
                else "single_model"
            )
            load_checkpoint(model, args.checkpoint, family=checkpoint_family)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Model: {cfg['model']['type']} | {n_params:.2f}M params | "
                    f"hop={model.hop_length}, n_fft={model.n_fft}, "
                    f"delay={model.algorithmic_delay} samples "
                    f"({model.algorithmic_delay / model.target_fs * 1000:.1f} ms)")

        if args.verify_streaming:
            streaming_verification_passed = bool(
                verify_streaming_consistency(model, args.mse_threshold)
            )
            if not streaming_verification_passed:
                logger.error("Streaming consistency verification failed; aborting export.")
                raise SystemExit(1)

    debug_target = None if release_requested else _prepare_debug_output_target(args.output)
    try:
        owned_path = _create_owned_onnx_artifact()
        export_path = os.fspath(owned_path)
        try:
            # 正式验证路径的所有中间操作都只接触 owned 临时文件。
            wrapper_or_model, dummy_inputs = export_model(model, export_path, args.opset)

            if args.simplify:
                simplify_onnx(export_path)

            # metadata 必须在 simplify 之后写入：onnxsim 重存模型时可能丢弃
            # metadata_props，先写会导致产物缺少 runtime ABI。
            write_streaming_runtime_metadata(export_path, wrapper_or_model)

            if args.verify or release_requested:
                _verify_or_exit(args, export_path, wrapper_or_model, dummy_inputs)
                if not release_requested:
                    logger.info("One-step PT/ORT smoke passed (debug-only, not a release gate).")

            # generic exporter 只产出 debug artifact；正式发布必须走对应
            # onnx/verify/verify_3way_*.py 的三路门禁，禁止在此直接调用 publisher。
            _publish_debug_onnx(owned_path, debug_target)
            logger.info("Debug/general ONNX exported to: %s", debug_target.display_path)
            logger.info("Done.")
        finally:
            owned_path.close()
    finally:
        if debug_target is not None:
            debug_target.close()


if __name__ == "__main__":
    main()
