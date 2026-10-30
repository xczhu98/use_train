"""Shared helpers for PASE / StuPASE models."""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

from core.utils.safe_load import safe_torch_load
from core.utils.validation import require_positive_int

logger = logging.getLogger(__name__)


def align_waveform_length(y: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Pad or crop a waveform's last dimension to exactly ``n_samples``."""
    if y.shape[-1] < n_samples:
        y = F.pad(y, (0, n_samples - y.shape[-1]))
    elif y.shape[-1] > n_samples:
        y = y[..., :n_samples]
    return y


def peak_normalize_to(wav: torch.Tensor, peak: float = 0.9, eps: float = 1e-8) -> torch.Tensor:
    """把单条波形峰值归一到 ``peak``（官方 validation 样例的 ``/ (max(abs) + 1e-8) * 0.9``）。"""
    return wav / (wav.abs().amax() + eps) * peak


def scale_waveform_to_reference_peak(
    waveform: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """把 ``waveform`` 的逐样本峰值缩放到 ``reference`` 的逐样本峰值。

    对齐官方 StuPASE / UniPASE ``inference/inference.py`` 的
    ``y / (y.abs().amax(-1) + eps) * x.abs().amax(-1)``：batch 内每个样本
    使用自己的峰值，不跨样本共享。峰值取 0 时结果保持有限值（输出为 0），
    不会产生 NaN。
    """
    out_peak = waveform.abs().amax(dim=-1, keepdim=True)
    ref_peak = reference.abs().amax(dim=-1, keepdim=True)
    return waveform / (out_peak + eps) * ref_peak


def scale_lengths_to(
    lengths: torch.Tensor | None,
    in_samples: int,
    out_samples: int,
) -> torch.Tensor | None:
    """把按 ``in_samples`` 计的有效长度线性缩放到 ``out_samples`` 时间基。

    用于 resample 之后把 ``(B,)`` 有效样本数换算到新长度；统计用途下允许
    末尾 1–2 个样本的取整误差。``lengths is None`` 时返回 ``None``。
    """
    if lengths is None:
        return None
    ratio = float(out_samples) / float(max(in_samples, 1))
    scaled = lengths.to(torch.float64) * ratio
    return scaled.round().clamp(min=0, max=out_samples).to(lengths.dtype)


def resample_waveform_and_lengths(
    waveform: torch.Tensor,
    lengths: torch.Tensor | None,
    orig_sr: int,
    target_sr: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """按 orig_sr→target_sr 重采样波形，并把 (B,) lengths 换算到新时间基。

    固定输出率 wrapper 在训练时接收 batch 实际采样率 ``fs`` 后统一走这里：
    ``orig_sr == target_sr`` 时原样返回（不触发任何重采样，保证单采样率
    训练路径逐位不变）；否则调用 ``torchaudio.functional.resample``，并用
    ``scale_lengths_to`` 把有效长度换算到新时间基。``lengths is None`` 时
    返回 ``None``。
    """
    orig_sr = require_positive_int(orig_sr, "orig_sr")
    target_sr = require_positive_int(target_sr, "target_sr")
    if orig_sr == target_sr:
        return waveform, lengths
    resampled = torchaudio.functional.resample(
        waveform, orig_freq=orig_sr, new_freq=target_sr
    )
    return resampled, scale_lengths_to(lengths, waveform.shape[-1], resampled.shape[-1])


def masked_time_stats(
    x: torch.Tensor,
    lengths: torch.Tensor | None = None,
    *,
    dim: int | tuple[int, ...] = 1,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按有效长度计算时间轴均值与无偏标准差。

    batch 右侧补零会改变全序列统计量，使同一条 utterance 的输出随同批最长
    样本变化；传入 ``(B,)`` 的 ``lengths`` 后，统计只覆盖每个样本的有效区。
    ``lengths is None`` 时退化为全序列统计；``dim`` 可为
    单个时间维（如 ``(B, T)`` 的 ``1``）或通道+时间维元组（如 ``(B, C, T)``
    的 ``(1, 2)``），时间维必须是最后一个统计维。
    """
    if lengths is None:
        mean = torch.mean(x, dim=dim, keepdim=True)
        std = torch.std(x, dim=dim, keepdim=True)
        return mean, std + eps
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    lengths = lengths.to(device=x.device, dtype=torch.long).clamp(
        min=0, max=x.shape[-1]
    )
    idx = torch.arange(x.shape[-1], device=x.device)
    mask = (idx.unsqueeze(0) < lengths.unsqueeze(1)).to(x.dtype)
    view = [1] * x.dim()
    view[0] = x.shape[0]
    view[-1] = x.shape[-1]
    mask = mask.view(view)
    # 计数必须与 torch.mean/std(dim=...) 的元素计数一致：多通道时按 C × 有效帧计。
    count = mask.expand_as(x).sum(dim=dims, keepdim=True).clamp(min=1)
    mean = (x * mask).sum(dim=dims, keepdim=True) / count
    centered = (x - mean) * mask
    var = (centered * centered).sum(dim=dims, keepdim=True)
    var = var / (count - 1).clamp(min=1)
    return mean, torch.sqrt(var) + eps


def hann_window_buffer(module: torch.nn.Module, n_fft: int, device) -> torch.Tensor:
    """返回挂在 ``module`` 上的 non-persistent hann 窗 buffer。

    窗按 ``n_fft`` 缓存成 buffer（Predictor 的窗长随 ``sr`` 变化），随模块
    ``.to()`` 一起迁移；只在 buffer 不在 ``device`` 时迁移一次，避免每次前向
    重建窗（非 persistent，不进入 ``state_dict``，不影响 checkpoint 严格匹配）。
    """
    name = f"_hann_window_{int(n_fft)}"
    window = getattr(module, name, None)
    if window is None:
        window = torch.hann_window(int(n_fft))
        module.register_buffer(name, window, persistent=False)
    if window.device != device:
        module.register_buffer(name, window.to(device), persistent=False)
        window = getattr(module, name)
    return window


def load_artifact_module(cls, ckpt_path: str, frozen: bool = True):
    """加载 ``{cfg, generator|model}`` 形式的发布 artifact（UniPASE 各模块共用）。

    - ``weights_only=True``：官方权重为纯 tensor + plain dict，杜绝 pickle 反序列化
      风险（权重经网络下载，路径由 YAML 注入）。
    - 官方实现用 ``strict=False``，这里改为输出 key 匹配报告，任何
      missing/unexpected 都 fail closed，避免静默错载。
    """
    ckpt_path = Path(ckpt_path).expanduser()
    logger.info("[%s] Loading from: %s", cls.__name__, ckpt_path)
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    if "cfg" not in ckpt:
        raise RuntimeError(
            f"[{cls.__name__}.from_pretrained] {ckpt_path}: checkpoint 缺少 'cfg' key。"
        )
    model = cls(**ckpt["cfg"])

    if "generator" in ckpt:
        model_dict = ckpt["generator"]
    elif "model" in ckpt:
        model_dict = ckpt["model"]
    else:
        raise RuntimeError(
            f"[{cls.__name__}.from_pretrained] {ckpt_path}: "
            f"checkpoint 既无 'generator' 也无 'model' key。"
        )

    missing, unexpected = model.load_state_dict(model_dict, strict=False)
    n_sd = len(model_dict)
    matched = n_sd - len(unexpected)
    logger.info(
        "[%s] key match report: matched=%d/%d, missing=%d, unexpected=%d",
        cls.__name__, matched, n_sd, len(missing), len(unexpected),
    )
    if matched == 0 or missing or unexpected:
        raise RuntimeError(
            f"[{cls.__name__}.from_pretrained] {ckpt_path}: state_dict 不匹配 "
            f"(matched={matched}/{n_sd}, missing={len(missing)}, unexpected={len(unexpected)})；"
            f"missing[:5]={missing[:5]}, unexpected[:5]={unexpected[:5]}"
        )

    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model
