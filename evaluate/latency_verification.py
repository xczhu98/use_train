#!/usr/bin/env python3
"""算法延迟/因果一致性验证（扰动法）—— TSE / SE / AEC 三任务统一入口。

核心 :func:`estimate_latency` 移植自官方 ``REAL-TSE-Challenge/utils/
latency_verification.py``，扰动形式、检测阈值、默认扰动点与打印口径逐行对齐
官方实现；调用约定保持官方签名 ``model_fn(x, aux) -> tuple``（``x`` 为主输入
波形、``aux`` 为第二路输入波形，返回 tuple 第一元素为增强波形）。

三任务适配器（统一"扰动只加在主输入 x 上，第二路 aux 固定"的语义）：

- **TSE**：``x``=mixture、``aux``=enrollment（官方原生语义），经
  :class:`TSEExtractor` 完成 waveform → STFT → model → iSTFT → waveform；
- **SE**：单输入任务，``aux`` 占位忽略；模型按配置有无顶层 ``stft`` 块走
  谱进谱出（胶水与 TSE 相同）或波形端到端两条路径；
- **AEC**：``x``=mic、``aux``=far-end ref（只扰动 mic、ref 固定，测对 mic
  的未来依赖）；dict 谱进谱出（``mic_spectra``/``farend_spectra`` →
  ``nearend_estimate_spectra``）。

⚠️ 通用注意（三任务均适用）：

- 测量前必须关闭输出峰值归一化（峰值归一是全局缩放，扰动改变峰值会把
  缩放泄漏到扰动点之前的样本，使响应时间虚降为 0）；
- 官方前提为**确定性无状态推理**；带内部 cache 的流式包装
  （``onnx/models/streaming_*``，接口异构）不在本 CLI 范围，若自行包装
  须在 y1/y2 两次前向之间用 ``init_cache()`` 重置状态。本 CLI 的适配器
  均为无状态全句前向，因果模型的测量值与流式部署版一致；
- 刻意算法延迟（如 DBGNet/DeepVQE 的 ``delay``）会使响应出现在扰动点
  **之后**，future dependency 为负——属"输出缓冲"，按官方口径与
  look-ahead 分开报告；
- 帧/chunk 系统测量值与名义延迟差一个 hop/chunk 属正常；报成绩时应
  使用默认扰动设置与阈值（threshold=1e-5），并另行报告含全部
  look-ahead、chunk 累积与输出缓冲的完整算法延迟。

Usage::

    # TSE（use_train checkpoint / model_dir；官方 wesep ckpt 配 --config）
    python -m evaluate.latency_verification --task tse \\
        --model data/tse/REAL-TSE/baseline/spk_emb_causal_100/avg_model.pt \\
        --config configs/tse/inference/tse_bsrnn_spk_realt_baseline_causal_infer.yaml

    # SE（checkpoint 内嵌配置，或 --config 显式指定）
    python -m evaluate.latency_verification --task se \\
        --model exp/se/<exp>/checkpoints/<ts>/best_model.tar

    # AEC（aux 自动作为 far-end ref，只扰动 mic）
    python -m evaluate.latency_verification --task aec \\
        --model exp/aec/<exp>/checkpoints/<ts>/best_model.tar
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import logging
import math
from pathlib import Path
from typing import Sequence

import torch

logger = logging.getLogger(__name__)

# 官方默认扰动点：整数秒与非对齐位置混合
_OFFICIAL_PERTURB_STARTS = (1.000, 2.000, 1.234, 2.346, 3.459)
_OFFICIAL_THRESHOLD = 1e-5


# ======================================================================
# 核心：官方 estimate_latency 逐行等价移植（任务无关）
# ======================================================================
@contextlib.contextmanager
def _temporary_torch_seed(seed: int | None):
    """块内应用 ``torch.manual_seed(seed)``，退出时恢复调用方全局 RNG 状态。"""
    if seed is None:
        yield
        return
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def estimate_latency(
    model_fn,
    fs: int = 16000,
    signal_len: int = 8,
    enroll_len: int = 5,
    perturb_starts=_OFFICIAL_PERTURB_STARTS,
    threshold: float = _OFFICIAL_THRESHOLD,
    seed: int | None = None,
):
    """扰动法估计模型未来依赖（官方 ``estimate_latency`` 逐行等价移植）。

    假设推理过程确定（无随机采样、无残留内部状态；如有内部状态须重置）。
    扰动只加在主输入 ``x`` 上，第二路输入（enroll/ref/占位）保持固定。

    Args:
        model_fn: 官方签名可调用对象 ``model_fn(x, aux) -> tuple``，
            ``x``/``aux`` 为 ``[1, T]`` 波形，返回 tuple 第一元素为
            增强波形。
        fs: 采样率。
        signal_len: 测试混合信号时长（秒）。
        enroll_len: 第二路输入时长（秒；enroll/ref，SE 为占位）。
        perturb_starts: 扰动起始点（秒），报成绩必须用官方默认值。
        threshold: 响应检测阈值，报成绩必须用官方默认 1e-5（更小会对
            浮点噪声与硬件差异敏感）。
        seed: 随机种子；None 与官方行为一致（不设种子）。

    Returns:
        list[float | None]：各扰动点的未来依赖估计；无响应为 ``None``。
    """
    if not isinstance(fs, int) or isinstance(fs, bool) or fs <= 0:
        raise ValueError(f"fs must be a positive integer, got {fs!r}")
    if not isinstance(signal_len, int) or isinstance(signal_len, bool) or signal_len <= 0:
        raise ValueError(f"signal_len must be a positive integer, got {signal_len!r}")
    if not isinstance(enroll_len, int) or isinstance(enroll_len, bool) or enroll_len <= 0:
        raise ValueError(f"enroll_len must be a positive integer, got {enroll_len!r}")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"threshold must be a positive finite number, got {threshold!r}")
    starts = tuple(perturb_starts)
    if not starts:
        raise ValueError("perturb_starts must not be empty")
    signal_samples = fs * signal_len
    for perturb_start in starts:
        if not isinstance(perturb_start, (int, float)) or not math.isfinite(perturb_start):
            raise ValueError(f"perturb_start must be finite, got {perturb_start!r}")
        start = int(perturb_start * fs)
        if start < 0 or start >= signal_samples:
            raise ValueError(
                f"perturb_start {perturb_start!r}s is outside signal duration {signal_len}s"
            )
    with _temporary_torch_seed(seed), torch.no_grad():
        # 随机测试输入与第二路输入
        x = torch.randn(1, signal_samples).clamp_(-1, 1)
        enroll = torch.randn(1, fs * enroll_len).clamp_(-1, 1)
        y1 = model_fn(x, enroll)[0]
        results: list[float | None] = []

        for perturb_start in starts:
            x2 = x.clone()
            start = int(perturb_start * fs)
            # 官方固定扰动形式：对未来所有样本加 0.5 * randn
            x2[..., start:] += 0.5 * torch.randn_like(x2[..., start:])
            y2 = model_fn(x2, enroll)[0]

            response_mask = (y1 - y2).abs() > threshold
            if not bool(response_mask.any()):
                logger.warning("Perturbation @ %.3fs produced no response", perturb_start)
                print(f"Perturbation @ {perturb_start:.3f}s | Response @ N/A | no response")
                results.append(None)
                continue
            response_index = int(response_mask.reshape(-1).nonzero(as_tuple=False)[0].item())
            response_time = response_index / fs
            # 未来依赖 = 扰动点 - 响应点；> 0 表示未来输入影响当前输出
            future_dependency = perturb_start - response_time
            print(
                f"Perturbation @ {perturb_start:.3f}s | "
                f"Response @ {response_time:.3f}s | "
                f"Future dependency = {future_dependency * 1000:.1f} ms"
            )
            results.append(future_dependency)

        valid_results = [value for value in results if value is not None]
        if valid_results:
            print(
                "\nSummary:\n"
                f"Min  : {min(valid_results) * 1000:.1f} ms\n"
                f"Mean : {sum(valid_results) / len(valid_results) * 1000:.1f} ms\n"
                f"Max  : {max(valid_results) * 1000:.1f} ms\n"
                f"No response: {len(results) - len(valid_results)}"
            )
        else:
            print("\nSummary:\nNo perturbation response detected.")

        print(
            "\nNote: The values above are intended for causality and "
            "latency-consistency verification only. Participants must "
            "separately report the total algorithmic latency of their "
            "systems, including any look-ahead, chunk accumulation, "
            "and output buffering."
        )

        return results


def summarize_future_dependency(
    results: list[float | None],
    *,
    fs: int,
    threshold: float,
    perturb_starts: Sequence[float],
) -> dict:
    """将未来依赖测量结果转换为适合机器读取的毫秒摘要，无响应保留为 ``None``。"""
    values_ms = [None if value is None else value * 1000 for value in results]
    valid_values = [value for value in values_ms if value is not None]
    return {
        "metric_kind": "future_dependency",
        "unit": "ms",
        "min_future_dependency_ms": min(valid_values) if valid_values else None,
        "mean_future_dependency_ms": (
            sum(valid_values) / len(valid_values) if valid_values else None
        ),
        "max_future_dependency_ms": max(valid_values) if valid_values else None,
        "values_ms": values_ms,
        "sample_rate": fs,
        "threshold": threshold,
        "perturb_starts_s": list(perturb_starts),
        "n_perturbations": len(results),
    }


# ======================================================================
# 适配器：统一包装为官方 model_fn(x, aux) -> (y,) 签名
# ======================================================================
def build_tse_model_fn(extractor):
    """TSE：把 :class:`TSEExtractor` 包装成官方签名（x=mix，aux=enroll）。

    要求 extractor 以 ``output_peak=None`` 构建（关闭峰值归一，见模块
    docstring）；内部经 ``extract_speech_from_pcm`` 完成
    waveform → STFT → model → iSTFT → waveform。
    """
    if extractor.output_peak is not None and extractor.output_peak > 0:
        raise ValueError(
            "latency 验证要求 output_peak=None（峰值归一会把扰动经全局缩放 "
            "泄漏到扰动点之前，破坏测量）"
        )
    sr = extractor.sample_rate

    def model_fn(x, enroll):
        # x/enroll: torch [1, T] @ sr（官方约定，CPU tensor）
        y = extractor.extract_speech_from_pcm(x[0], sr, enroll[0], sr)
        return (torch.from_numpy(y).unsqueeze(0),)

    return model_fn



def build_se_model_fn(model, stft_mod, device: torch.device, sample_rate: int):
    """SE：单输入适配器（x=noisy，aux 占位忽略）。

    配置含顶层 ``stft`` 块时走谱进谱出（STFT → model(spec) → iSTFT），
    否则波形端到端（model(wav)）；forward 声明 ``fs`` 参数时注入采样率。
    """
    accepts_fs = "fs" in inspect.signature(model.forward).parameters

    def model_fn(x, _aux):
        wav = x.to(device).float()
        if stft_mod is not None:
            spec = stft_mod(wav)
            out = model(spec, fs=sample_rate) if accepts_fs else model(spec)
            y = stft_mod.inverse(out, wav.shape[-1])
        else:
            out = model(wav, fs=sample_rate) if accepts_fs else model(wav)
            y = out
        # 输出长度守卫：裁到与输入等长（diff 要求 y1/y2 形状一致）
        return (y[..., : x.shape[-1]].detach().float().cpu(),)

    return model_fn


def build_aec_model_fn(model, stft_mod, device: torch.device):
    """AEC：双输入适配器（x=mic，aux=far-end ref；只扰动 mic）。

    dict 谱进谱出：``mic_spectra``/``farend_spectra`` →
    ``nearend_estimate_spectra`` → iSTFT。要求配置含顶层 ``stft`` 块。
    """
    if stft_mod is None:
        raise ValueError("AEC 模型为 dict 谱进谱出接口，配置必须含顶层 stft 块")

    def model_fn(x, ref):
        mic = x.to(device).float()
        far = ref.to(device).float()
        batch = {"mic_spectra": stft_mod(mic), "farend_spectra": stft_mod(far)}
        out = model(batch)
        est = out["nearend_estimate_spectra"]
        y = stft_mod.inverse(est, mic.shape[-1])
        return (y[..., : x.shape[-1]].detach().float().cpu(),)

    return model_fn


# ======================================================================
# SE/AEC 通用 checkpoint 加载（TSE 走 TSEExtractor，不用此路径）
# ======================================================================
def load_se_aec_model(ckpt_path: str, config_path: str | None, device: torch.device):
    """加载 SE/AEC 模型（checkpoint 内嵌配置或显式 --config）。

    ``data.mode`` 是任务推导的唯一来源，配置未声明时直接报错，不做默认回退。

    Returns:
        (model, stft_mod, sample_rate)：stft_mod 为 None 表示无顶层 stft 块
        （SE 波形端到端模型）；sample_rate 取 data.sample_rate，缺省 16000。
    """
    from core.models.build import build_model
    from core.models import register_task_models
    from core.data.constants import normalize_data_mode
    from core.stft import build_stft_istft
    from core.utils.config import load_merged_config
    from core.utils.inference_artifacts import fingerprint_file
    from evaluate.checkpoint_loading import (
        CheckpointFamily,
        apply_checkpoint,
        load_checkpoint_payload,
    )

    checkpoint_fingerprint = fingerprint_file(ckpt_path)
    _resolved_checkpoint, ckpt = load_checkpoint_payload(
        ckpt_path,
        map_location="cpu",
        purpose="latency verification model checkpoint",
        expected_fingerprint=checkpoint_fingerprint,
    )
    if config_path is not None:
        cfg = load_merged_config(config_path)
    else:
        cfg = ckpt.get("resolved_config") or ckpt.get("config")
        if not isinstance(cfg, dict):
            raise ValueError(
                f"checkpoint {ckpt_path} 内不含 config/resolved_config，"
                "请用 --config 显式指定模型配置"
            )

    stft_cfg = cfg.get("stft")
    model_cfg = cfg["model"]
    is_gan = isinstance(model_cfg, dict) and "generator" in model_cfg
    build_cfg = model_cfg["generator"] if is_gan else model_cfg
    data_mode = (cfg.get("data") or {}).get("mode")
    if data_mode is None:
        source = config_path if config_path is not None else ckpt_path
        raise ValueError(
            f"{source} 未声明 data.mode；latency_verification 现在按 data.mode "
            "推导 SE/AEC 任务，请用 --config 指定显式声明 data.mode 的配置"
        )
    task = normalize_data_mode(data_mode).task
    if task not in {"se", "aec"}:
        raise ValueError(
            f"latency_verification SE/AEC loader received unsupported task {task!r}"
        )
    model_type = build_cfg.get("type") if isinstance(build_cfg, dict) else None
    if model_type:
        try:
            register_task_models(task, only=str(model_type))
        except KeyError:
            # 自定义 ``model.path``/测试替身可能使用不在内置 lazy map 的 type；
            # 让原始 build_model 路径继续给出最终错误，而不是提前改变兼容语义。
            pass
    model = build_model(build_cfg, stft_cfg=stft_cfg).to(device).eval()
    family = (
        CheckpointFamily.GAN_GENERATOR
        if is_gan
        else CheckpointFamily.SINGLE_MODEL
    )
    apply_checkpoint(
        model,
        ckpt,
        family=family,
        context=f"latency checkpoint {ckpt_path}",
    )
    logger.info("加载 %s 权重: %s", family.value, ckpt_path)

    stft_mod = build_stft_istft(stft_cfg).to(device) if stft_cfg else None
    sample_rate = int((cfg.get("data") or {}).get("sample_rate") or 16000)
    return model, stft_mod, sample_rate


# ======================================================================
# 统一 CLI
# ======================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="算法延迟/因果一致性验证（扰动法，官方 latency_verification 移植版；"
                    "TSE/SE/AEC 三任务统一入口）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", required=True, choices=["tse", "se", "aec"],
                        help="任务类型：tse（x=mix, aux=enroll）/ se（单输入，aux 忽略）"
                             "/ aec（x=mic, aux=far-end ref）")
    parser.add_argument("--model", required=True,
                        help="use_train checkpoint（.tar）或 model_dir；"
                             "TSE 官方 wesep ckpt 需配合 --config 使用")
    parser.add_argument("--config", default=None,
                        help="显式模型配置 YAML（优先于 checkpoint 内嵌配置）")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--signal_len", type=int, default=8, help="测试信号时长（秒）")
    parser.add_argument("--aux_len", type=int, default=5,
                        help="第二路输入时长（秒）：tse=enroll / aec=ref / se 忽略")
    parser.add_argument("--threshold", type=float, default=_OFFICIAL_THRESHOLD,
                        help="响应检测阈值（报成绩须用官方默认 1e-5）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（官方默认不设；需要可复现结果时指定）")
    parser.add_argument("--json-output", default=None,
                        help="将结构化 future dependency 指标写入该 JSON 文件")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.task == "tse":
        from tasks.tse.inference import TSEExtractor

        extractor = TSEExtractor(
            args.model,
            config_path=args.config,
            device=str(device),
            apply_vad=False,
            output_peak=None,  # 强制关闭峰值归一（见模块 docstring 原因说明）
        )
        model_fn = build_tse_model_fn(extractor)
        fs = extractor.sample_rate
    else:
        # 导入任务模型包，触发 MODEL_REGISTRY lazy map 注册（build_model 据此
        # 惰性加载模型类；tse 路径由 TSEExtractor 内部处理）
        if args.task == "se":
            import tasks.se.models  # noqa: F401
        else:
            import tasks.aec.models  # noqa: F401
        model, stft_mod, fs = load_se_aec_model(args.model, args.config, device)
        if args.task == "se":
            model_fn = build_se_model_fn(model, stft_mod, device, fs)
        else:
            model_fn = build_aec_model_fn(model, stft_mod, device)

    # AEC 的 mic/ref 是同步等长录音（模型按帧对齐两路），ref 强制与 mic 同长；
    # TSE 的 enroll 可短于 mix（官方语义）；SE 忽略 aux
    aux_len = args.aux_len
    if args.task == "aec":
        if args.aux_len != args.signal_len:
            logger.info("AEC ref 与 mic 同步等长：aux_len 覆写为 signal_len=%ds",
                        args.signal_len)
        aux_len = args.signal_len

    results = estimate_latency(
        model_fn,
        fs=fs,
        signal_len=args.signal_len,
        enroll_len=aux_len,
        threshold=args.threshold,
        seed=args.seed,
    )
    if args.json_output:
        summary = summarize_future_dependency(
            results,
            fs=fs,
            threshold=args.threshold,
            perturb_starts=_OFFICIAL_PERTURB_STARTS,
        )
        summary.update(
            task=args.task,
            model=args.model,
            config=args.config,
            device=str(device),
        )
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
