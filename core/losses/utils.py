"""共用工具函数：STFT/幅度/相位计算，避免重复实现。

所有 loss 模块均可 import 这些工具函数。
"""

from __future__ import annotations

import torch


def resolve_waveform_lengths(
    x: torch.Tensor,
    lengths: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """统一 lengths 并返回有效 sample mask。"""
    if lengths is None:
        lengths = torch.full(
            (x.shape[0],), x.shape[1], device=x.device, dtype=torch.long
        )
    else:
        lengths = lengths.to(device=x.device, dtype=torch.long)
    sample_index = torch.arange(x.shape[1], device=x.device)
    sample_mask = sample_index.unsqueeze(0) < lengths.unsqueeze(1)
    return lengths, sample_mask


def prepare_centered_stft_inputs(
    *signals: torch.Tensor,
    lengths: torch.Tensor | None,
    n_fft: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """为 ``center=True`` STFT 准备 trim 等价输入。

    所有波形先按同一有效 sample mask 清除 padding，再固定预留右侧 reflect
    余量并从有效边界延展。这样短于 ``n_fft // 2`` 的输入也可执行 STFT，且
    trim、轻量右 padding 和完整 batch padding 共享同一边界语义。
    """
    if not signals:
        raise ValueError("prepare_centered_stft_inputs requires at least one signal")
    reference = signals[0]
    resolved_lengths, sample_mask = resolve_waveform_lengths(reference, lengths)
    prepared = []
    for signal in signals:
        if signal.shape != reference.shape:
            raise ValueError(
                "centered STFT signals must have identical shapes; "
                f"got {reference.shape} and {signal.shape}"
            )
        signal, _ = prepare_stft_waveform_at_lengths(
            signal, resolved_lengths, n_fft=n_fft, center=True
        )
        prepared.append(signal)
    return tuple(prepared), resolved_lengths, sample_mask


def stft_magnitude(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    center: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算 STFT 幅度谱。
    
    Args:
        x: (B, T) 时域波形
        n_fft: FFT 点数
        hop_length: 帧移
        win_length: 窗长
        window: 窗函数 tensor
        center: 是否中心填充
        eps: 最小值 clamp
        
    Returns:
        (B, F, T) 幅度谱，F = n_fft // 2 + 1
    """
    spec = torch.stft(
        x.float(),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
        onesided=True,
    )
    return spec.abs().clamp(min=eps)


def stft_complex(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    center: bool = True,
) -> torch.Tensor:
    """计算 STFT 复谱。
    
    Args:
        x: (B, T) 时域波形
        n_fft: FFT 点数
        hop_length: 帧移
        win_length: 窗长
        window: 窗函数 tensor
        center: 是否中心填充
        
    Returns:
        (B, F, T) 复谱，complex64/128
    """
    return torch.stft(
        x.float(),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
        onesided=True,
    )


def stft_real(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    center: bool = True,
) -> torch.Tensor:
    """计算 STFT 并转为实数形式 (real, imag)。
    
    Args:
        x: (B, T) 时域波形
        n_fft: FFT 点数
        hop_length: 帧移
        win_length: 窗长
        window: 窗函数 tensor
        center: 是否中心填充
        
    Returns:
        (B, F, T, 2) 实数形式，最后维度 [real, imag]
    """
    spec = stft_complex(x, n_fft, hop_length, win_length, window, center)
    return torch.view_as_real(spec)


def compute_spec_frame_lengths(
    sample_lengths: torch.Tensor,
    n_fft: int,
    win_length: int,
    hop_length: int,
    center: bool = True,
) -> torch.Tensor:
    """根据波形 sample 长度计算 STFT 帧数。

    Args:
        sample_lengths: (B,) 每条波形的有效样本数
        n_fft: FFT 点数（torch.stft 在 win_length < n_fft 时把窗补零到 n_fft,
               帧数由 n_fft 决定, 故公式基于 n_fft 而非 win_length）
        win_length: 窗长（帧数计算不使用该参数）
        hop_length: 帧移
        center: 是否使用 center padding（默认 True）。
                True:  1 + (L + 2·pad - n_fft) // hop, pad = n_fft // 2
                False: 1 + (L - n_fft) // hop

    Returns:
        (B,) 每条的有效帧数
    """
    if center:
        pad = n_fft // 2
        L_eff = sample_lengths.long() + 2 * pad
        return torch.clamp(1 + (L_eff - n_fft) // hop_length, min=1).int()
    L_eff = sample_lengths.long()
    # center=False 时长度不足 n_fft 不存在任何实际 STFT 帧；不得将其
    # 伪装成一帧并把右 padding 纳入谱损失。
    return torch.clamp(1 + (L_eff - n_fft) // hop_length, min=0).int()


def reflect_extend_at_lengths(x: torch.Tensor, lengths: torch.Tensor, pad: int) -> torch.Tensor:
    """在每条样本有效边界处补 reflect 内容, 等效于 trim 到 lengths 后
    ``torch.stft(center=True, pad_mode='reflect')`` 的边界填充(JAE-H02)。

    补零波形直接 STFT 时, 最后一个有效帧的窗尾覆盖的是零; 而未补零(trim)波形
    STFT 时该位置是 reflect 延展的有效信号, 两者不一致 → loss 随 batch 最大长度
    变化。此处按 torch reflect 右填充定义补齐；当有效长度不大于 pad 时按无端点
    周期反射继续映射，长度为 1 时退化为复制唯一样本。
    x: [B, T], padding 区需已置零且应预留至少 pad 个样本; lengths: [B];
    pad = n_fft // 2。padding 区梯度恒为 0(gather 只取有效区)。

    se_train_mr._trim_consistent_stft_input 是带时域 mask 的变体(mask + 本函数)。
    """
    if pad <= 0:
        return x
    lengths = lengths.to(device=x.device, dtype=torch.long)
    t = torch.arange(x.shape[1], device=x.device)
    ext_mask = (t.unsqueeze(0) >= lengths.unsqueeze(1)) & (
        t.unsqueeze(0) < lengths.unsqueeze(1) + pad
    )
    if not ext_mask.any():
        return x

    period = 2 * (lengths.unsqueeze(1) - 1)
    reflected = torch.remainder(t.unsqueeze(0), period.clamp(min=1))
    src = torch.where(reflected < lengths.unsqueeze(1), reflected, period - reflected)
    src = src.clamp(min=0, max=x.shape[1] - 1)
    ext = torch.gather(x, 1, src)
    return torch.where(ext_mask, ext, x)


def prepare_stft_waveform_at_lengths(
    x: torch.Tensor,
    lengths: torch.Tensor | None,
    *,
    n_fft: int,
    center: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """按真实有效边界准备 STFT 输入，避免 batch 右 padding 污染边界帧。"""
    stft_lengths = lengths
    pad = n_fft // 2 if center else 0
    if stft_lengths is None and center and x.shape[1] <= pad:
        stft_lengths = torch.full(
            (x.shape[0],),
            x.shape[1],
            dtype=torch.long,
            device=x.device,
        )
    if stft_lengths is None:
        return x, None

    stft_lengths = stft_lengths.to(device=x.device, dtype=torch.long)
    time = torch.arange(x.shape[1], device=x.device)
    x = x * (time[None, :] < stft_lengths[:, None])
    if pad > 0:
        x = torch.cat((x, x.new_zeros(x.shape[0], pad)), dim=1)
        x = reflect_extend_at_lengths(x, stft_lengths, pad)
    return x, stft_lengths


def create_frame_mask(
    lengths: torch.Tensor,
    n_frames: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    device: torch.device,
    center: bool = True,
) -> torch.Tensor:
    """创建 STFT 帧级别掩码 (B, T)。
    
    Args:
        lengths: (B,) 波形有效样本数
        n_frames: STFT 总帧数
        n_fft: FFT 点数
        win_length: 窗长
        hop_length: 帧移
        device: 设备
        center: 是否使用 center padding（默认 True）
        
    Returns:
        (B, T) bool 掩码，True 表示有效帧
    """
    frame_lens = compute_spec_frame_lengths(lengths, n_fft, win_length, hop_length, center=center)
    t = torch.arange(n_frames, device=device, dtype=lengths.dtype)
    return t[None, :] < frame_lens[:, None]
