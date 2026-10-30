#!/usr/bin/env python3
"""NBSS SpatialNet / OnlineSpatialNet 论文口径 Params 与 FLOPs 复核（只读）。

加载 ``configs/{se,ss}/spatial`` 下的论文对齐配置，按配置自身的 ``data.sample_rate`` 与顶层
``stft`` 构建模型，并在 4 s / 8 kHz / 6 通道口径下测量：

- Params：``sum(p.numel() for p in model.parameters())``；
- FLOPs：``torch.utils.flop_counter.FlopCounterMode`` 前向。在线分支按逐帧流式（论文
  inference 模式）的逐帧代价 × 帧数；离线 SpatialNet 走整段前向。

论文报告值（Table I，8 kHz / 6 ch / 4 s）：SpatialNet 1.2 M / 23.1 G/s；
oSpatialNet-MSA 1.2 M / 23.1 G/s；oSpatialNet-Ret 1.4 M / 20.4 G/s；
oSpatialNet-Mamba 1.4 M / 18.4 G/s。

口径说明：``torch.stft`` 不计入 FlopCounterMode（已核实为 0），因此数值只含网络前向；本环境
的 FlopCounterMode 不登记 SDPA，且整段 ``nn.MultiheadAttention`` 路径计得偏低（16.84 G/s），
离线 SpatialNet 的对照以同实现逐帧口径（23.08 G/s）为准，脚本会把两者都打印出来。

本脚本只复核 Params 与 FLOPs：论文没有报告 RTF、内存或设备级延迟，RTF 只能作为本仓工程
指标由 ``tools/model_tools/audit_metrics.py`` 测量（整段口径），流式部署成本需另行测逐帧
``forward_streaming``；两者在 CPU 上可相差约一个数量级，原因见
``tasks/se/models/modules/online_spatialnet_impl/backbone.py`` 的类 docstring。
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import torch
from torch.utils.flop_counter import FlopCounterMode

from core.utils.config import load_merged_config

PAPER_VALUES = {
    # label: (paper params M, paper FLOPs G/s)
    "spatialnet_dynamic": (1.2, 23.1),
    "online_spatialnet_dynamic_mhsa": (1.2, 23.1),
    "online_spatialnet_dynamic_ret": (1.4, 20.4),
    "online_spatialnet_dynamic_mamba": (1.4, 18.4),
}

DEFAULT_CONFIGS = (
    "configs/se/spatial/spatialnet_dynamic.yaml",
    "configs/se/spatial/online_spatialnet_dynamic_mhsa.yaml",
    "configs/se/spatial/online_spatialnet_dynamic_ret.yaml",
    "configs/se/spatial/online_spatialnet_dynamic_mamba.yaml",
    "configs/ss/spatial/online_spatialnet_dynamic_mhsa.yaml",
    "configs/ss/spatial/online_spatialnet_dynamic_ret.yaml",
    "configs/ss/spatial/online_spatialnet_dynamic_mamba.yaml",
)


def _build_model(config_path: str, cfg: dict):
    from core.models.build import build_model as build_shared_model
    from tasks.se.models import build_model as build_se_model
    from tasks.ss.models import build_model as build_ss_model

    model_cfg = cfg["model"]
    if model_cfg.get("type") in {"SpatialNet", "OnlineSpatialNet"}:
        num_sources = (cfg.get("ss") or {}).get("num_sources")
        return build_ss_model(model_cfg, stft_cfg=cfg.get("stft"), num_sources=num_sources)
    if "generator" in model_cfg:
        raise ValueError(f"{config_path}: alignment check does not cover GAN configs")
    return build_se_model(model_cfg, stft_cfg=cfg.get("stft")) if model_cfg.get("type") in {
        "SpatialNet_SE", "OnlineSpatialNet_SE",
    } else build_shared_model(model_cfg, stft_cfg=cfg.get("stft"))


def _frames(config: dict, seconds: float = 4.0) -> int:
    stft_cfg = config.get("stft") or {}
    sample_rate = int((config.get("data") or {}).get("sample_rate"))
    hop = int(stft_cfg.get("hop_length"))
    return int(round(seconds * sample_rate / hop)) + 1


def _flops(fn) -> float:
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return counter.get_total_flops() / 1e9


def _measure(config_path: str, seconds: float = 4.0) -> dict:
    cfg = load_merged_config(config_path)
    model = _build_model(config_path, cfg).eval()
    params = sum(parameter.numel() for parameter in model.parameters())
    stft_cfg = cfg.get("stft") or {}
    n_fft = int(stft_cfg["n_fft"])
    n_channel = int(model.n_channel)
    freqs = n_fft // 2 + 1
    frames = _frames(cfg, seconds)

    if callable(getattr(model, "init_cache", None)) and callable(getattr(model, "forward_streaming", None)):
        frame = torch.zeros(1, n_channel, freqs, 1, 2)
        cache = model.init_cache(1, torch.device("cpu"), torch.float32)
        per_frame = _flops(lambda: model.forward_streaming(frame, cache))
        streaming = per_frame * frames
        spectrum = torch.zeros(1, n_channel, freqs, frames, 2)
        sequence = _flops(lambda: model(spectrum))
    else:
        spectrum = torch.zeros(1, n_channel, freqs, frames, 2)
        sequence = _flops(lambda: model(spectrum))
        streaming = None
    return {
        "params": params,
        "streaming_gflops": streaming,
        "sequence_gflops": sequence,
        "frames": frames,
        "seconds": seconds,
        "n_channel": n_channel,
        "n_fft": n_fft,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NBSS SpatialNet/OnlineSpatialNet 论文口径对齐复核")
    parser.add_argument("--config", "-C", action="append", default=None, help="只复核指定 YAML（可重复）")
    parser.add_argument("--tolerance", type=float, default=0.10, help="Params/FLOPs 允许相对偏差，默认 0.10")
    args = parser.parse_args()

    configs = tuple(args.config) if args.config else DEFAULT_CONFIGS
    failures = 0
    print(f"{'config':52s} {'params(M)':>10s} {'paper':>6s} {'FLOPs(G/s)':>11s} {'paper':>6s} {'dev':>7s}")
    for config_path in configs:
        label = os.path.splitext(os.path.basename(config_path))[0]
        try:
            result = _measure(config_path)
        except Exception as exc:  # noqa: BLE001 - CLI 汇总
            print(f"{config_path:52s} FAILED: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        params_m = result["params"] / 1e6
        streaming = result["streaming_gflops"]
        sequence = result["sequence_gflops"]
        total_flops = streaming if streaming is not None else sequence
        paper_params, paper_flops = PAPER_VALUES.get(label, (None, None))
        flops_per_second = total_flops / result["seconds"] if total_flops is not None else None
        params_dev = abs(params_m - paper_params) / paper_params if paper_params else 0.0
        flops_dev = abs(flops_per_second - paper_flops) / paper_flops if paper_flops else 0.0
        status = "ok"
        if paper_params is not None and params_dev > args.tolerance:
            status = "params>tol"
            failures += 1
        if streaming is not None and paper_flops is not None and flops_dev > args.tolerance:
            status = "flops>tol"
            failures += 1
        detail = "" if streaming is not None else f" (整段 {sequence:.2f} G, 逐帧口径不可用)"
        print(
            f"{config_path:52s} {params_m:10.4f} {paper_params if paper_params else '-':>6} "
            f"{flops_per_second:11.2f} {paper_flops if paper_flops else '-':>6} "
            f"p{params_dev:+.1%}/f{flops_dev:+.1%} {status}{detail}"
        )
    print(f"[SUMMARY] configs={len(configs)} failures={failures} tolerance={args.tolerance:.0%}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
