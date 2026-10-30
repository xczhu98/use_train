"""GTCRN 官方 loss.py 公式的可组合损失组件。

``GTCRNCompressedSpectralLoss`` 计算 30/70 加权的压缩 RI 与 magnitude；
``GTCRNLogRatioLoss`` 计算尺度投影后的 ``-log10`` ratio，不是乘 10 的 dB。
两者拆分注册，由顶层 ``CompositeLoss`` 组合。
"""

import math
from numbers import Real

import torch
import torch.nn as nn

from core.losses.utils import (
    compute_spec_frame_lengths,
    prepare_stft_waveform_at_lengths,
)
from core.stft import resolve_stft_config
from core.utils.registry import LOSS_REGISTRY


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_positive_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an exact positive int, got {value!r}")
    if value <= 0:
        raise ValueError(f"{field} must be positive, got {value!r}")
    return value


def _validate_finite_real(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def _build_stft_window(window_name: str, win_length: int) -> torch.Tensor:
    """构造与框架 ``StftIstftModule`` 相同的普通 analysis/synthesis 窗。

    analysis/synthesis 同窗成比例是 ``torch.istft`` 包络归一化下
    一致谱精确重建的条件；非法窗名应已被 ``resolve_stft_config`` 拦截。
    """
    if window_name == "hann":
        return torch.hann_window(win_length)
    if window_name == "hamming":
        return torch.hamming_window(win_length)
    raise ValueError(
        f"window '{window_name}' not supported; use hann or hamming"
    )


def _validate_waveform_inputs(
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    lengths: torch.Tensor | None,
) -> None:
    """校验绕过 loss builder 直接调用时的公共输入契约。"""
    if not torch.is_tensor(enhanced):
        raise TypeError("enhanced must be a torch.Tensor")
    if enhanced.ndim != 2:
        raise ValueError(f"enhanced must be [B, T], got {tuple(enhanced.shape)}")
    if not enhanced.is_floating_point():
        raise TypeError(f"enhanced must be floating point, got {enhanced.dtype}")
    if enhanced.shape[0] < 1 or enhanced.shape[1] < 1:
        raise ValueError(f"enhanced must have non-empty B and T dimensions, got {tuple(enhanced.shape)}")

    if not torch.is_tensor(clean):
        raise TypeError("clean must be a torch.Tensor")
    if clean.ndim != 2:
        raise ValueError(f"clean must be [B, T], got {tuple(clean.shape)}")
    if clean.shape != enhanced.shape:
        raise ValueError(
            f"enhanced/clean shape mismatch: {tuple(enhanced.shape)} vs {tuple(clean.shape)}"
        )
    if clean.device != enhanced.device:
        raise ValueError(f"clean must be on device {enhanced.device}, got {clean.device}")
    if not clean.is_floating_point():
        raise TypeError(f"clean must be floating point, got {clean.dtype}")
    if clean.dtype != enhanced.dtype:
        raise ValueError(f"clean must use dtype {enhanced.dtype}, got {clean.dtype}")
    finite_inputs = torch.isfinite(enhanced).all() & torch.isfinite(clean).all()
    if not bool(finite_inputs.item()):
        raise ValueError("enhanced and clean must contain only finite values")

    if lengths is None:
        return
    if not torch.is_tensor(lengths):
        raise TypeError("lengths must be a torch.Tensor")
    if lengths.ndim != 1 or lengths.shape[0] != enhanced.shape[0]:
        raise ValueError(
            f"lengths must be [B], got {tuple(lengths.shape)} for B={enhanced.shape[0]}"
        )
    if lengths.device != enhanced.device:
        raise ValueError(f"lengths must be on device {enhanced.device}, got {lengths.device}")
    if lengths.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"lengths must use integer dtype, got {lengths.dtype}")
    if bool(((lengths < 1) | (lengths > enhanced.shape[1])).any().item()):
        raise ValueError(f"lengths must be in [1, {enhanced.shape[1]}]")


def _as_complex_spectrum(
    value: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    n_fft: int,
) -> torch.Tensor:
    """校验 framework 谱布局并规范为 ``[B, F, N]`` complex64。"""
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.is_complex():
        spectrum = value
    elif value.ndim == 4 and value.shape[-1] == 2 and value.is_floating_point():
        spectrum = torch.view_as_complex(value.float().contiguous())
    else:
        raise ValueError(
            f"{name} must be complex [B, F, N] or real [B, F, N, 2], "
            f"got shape {tuple(value.shape)} and dtype {value.dtype}"
        )
    expected_bins = n_fft // 2 + 1
    if spectrum.ndim != 3 or spectrum.shape[0] != batch_size:
        raise ValueError(
            f"{name} must have batch size {batch_size}, got {tuple(spectrum.shape)}"
        )
    if spectrum.shape[1] != expected_bins:
        raise ValueError(
            f"{name} frequency bins must match n_fft={n_fft} "
            f"({expected_bins} bins), got {spectrum.shape[1]}"
        )
    if not torch.isfinite(spectrum).all():
        raise ValueError(f"{name} must contain only finite values")
    return spectrum.to(dtype=torch.complex64)


@LOSS_REGISTRY.register("GTCRNCompressedSpectralLoss")
class GTCRNCompressedSpectralLoss(nn.Module):
    """GTCRN 压缩 RI 与 magnitude 损失，对完整 batch 生效。

    谱进谱出模型可通过 ``enhanced_spectra`` / ``target_spectra`` 直接传入
    framework STFT 复谱，避免 iSTFT→STFT 把模型预测投影到一致谱子空间。两者均未
    提供时保留 waveform adapter；只提供一侧则 fail-fast。
    """

    SUPPORTS_DIRECT_SPECTRA = True
    REQUIRES_MATCHING_STFT_CONTEXT = True

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        compress_factor: float = 0.3,
        lamda_ri: float = 30.0,
        lamda_mag: float = 70.0,
        center: bool = True,
        normalized: bool = False,
        window: str = "hann",
    ):
        super().__init__()
        n_fft = _validate_positive_int(n_fft, field="n_fft")
        hop_length = _validate_positive_int(hop_length, field="hop_length")
        win_length = _validate_positive_int(win_length, field="win_length")
        compress_factor = _validate_finite_real(
            compress_factor, field="compress_factor"
        )
        lamda_ri = _validate_finite_real(lamda_ri, field="lamda_ri")
        lamda_mag = _validate_finite_real(lamda_mag, field="lamda_mag")
        if type(center) is not bool:
            raise TypeError(f"center must be bool, got {center!r}")
        if type(normalized) is not bool:
            raise TypeError(f"normalized must be bool, got {normalized!r}")
        if type(window) is not str:
            raise TypeError(f"window must be str, got {window!r}")
        if win_length > n_fft:
            raise ValueError("win_length must not exceed n_fft")
        if not (0.0 < compress_factor <= 1.0):
            raise ValueError("compress_factor must be in (0, 1]")
        if lamda_ri < 0.0:
            raise ValueError("lamda_ri must be non-negative")
        if lamda_mag < 0.0:
            raise ValueError("lamda_mag must be non-negative")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.c = compress_factor
        self.lamda_ri = lamda_ri
        self.lamda_mag = lamda_mag
        self.center = center
        self.normalized = normalized
        wname = window.lower()
        self.window_name = wname
        self.register_buffer(
            "_window", _build_stft_window(wname, win_length), persistent=False
        )

    def _stft_kw(self) -> dict:
        return dict(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window,
            center=self.center,
            normalized=self.normalized,
            onesided=True,
            return_complex=True,
        )

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        enhanced_spectra: torch.Tensor | None = None,
        target_spectra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_waveform_inputs(enhanced, clean, lengths)
        direct_spectra = enhanced_spectra is not None or target_spectra is not None
        if (enhanced_spectra is None) != (target_spectra is None):
            raise ValueError(
                "enhanced_spectra and target_spectra must be provided together"
            )
        if not self.center and (
            enhanced.shape[1] < self.n_fft
            or (lengths is not None and bool((lengths < self.n_fft).any().item()))
        ):
            raise ValueError(
                "center=False requires every waveform length to be at least "
                f"n_fft={self.n_fft}"
            )

        enhanced = enhanced.float()
        clean = clean.float()
        if direct_spectra:
            assert enhanced_spectra is not None and target_spectra is not None
            ps = _as_complex_spectrum(
                enhanced_spectra,
                name="enhanced_spectra",
                batch_size=enhanced.shape[0],
                n_fft=self.n_fft,
            )
            ts = _as_complex_spectrum(
                target_spectra,
                name="target_spectra",
                batch_size=enhanced.shape[0],
                n_fft=self.n_fft,
            )
            if ps.device != enhanced.device or ts.device != enhanced.device:
                raise ValueError("direct spectra must be on the waveform device")
            if ps.shape != ts.shape:
                raise ValueError(
                    "enhanced_spectra/target_spectra shape mismatch: "
                    f"{tuple(ps.shape)} vs {tuple(ts.shape)}"
                )
            spectral_lengths = lengths
        else:
            enhanced_input, spectral_lengths = prepare_stft_waveform_at_lengths(
                enhanced,
                lengths,
                n_fft=self.n_fft,
                center=self.center,
            )
            clean_input, _ = prepare_stft_waveform_at_lengths(
                clean,
                lengths,
                n_fft=self.n_fft,
                center=self.center,
            )
            ps = torch.stft(enhanced_input, **self._stft_kw())
            ts = torch.stft(clean_input, **self._stft_kw())
        # 与官方 GTCRN 一致：固定 1e-12 置于 sqrt 内，零点附近保持平滑。
        pm = torch.sqrt(ps.real.square() + ps.imag.square() + 1e-12)
        tm = torch.sqrt(ts.real.square() + ts.imag.square() + 1e-12)
        pc = ps / pm ** (1.0 - self.c)
        tc = ts / tm ** (1.0 - self.c)

        if spectral_lengths is not None:
            n_frames = pc.shape[-1]
            frame_len = compute_spec_frame_lengths(
                spectral_lengths,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=self.center,
            ).to(device=pc.device)
            t = torch.arange(n_frames, device=pc.device)
            frame_mask = (t[None, :] < frame_len[:, None].clamp(min=1)).to(
                pc.real.dtype
            ).unsqueeze(1)
            denom = (frame_mask.sum() * pc.shape[1]).clamp(min=1.0)
            ri_loss = (((pc.real - tc.real) ** 2) * frame_mask).sum() / denom
            imag_loss = (((pc.imag - tc.imag) ** 2) * frame_mask).sum() / denom
            mag_loss = (((pm ** self.c - tm ** self.c) ** 2) * frame_mask).sum() / denom
        else:
            ri_loss = torch.mean((pc.real - tc.real) ** 2)
            imag_loss = torch.mean((pc.imag - tc.imag) ** 2)
            mag_loss = torch.mean((pm ** self.c - tm ** self.c) ** 2)

        return self.lamda_ri * (ri_loss + imag_loss) + self.lamda_mag * mag_loss


@LOSS_REGISTRY.register("GTCRNLogRatioLoss")
class GTCRNLogRatioLoss(nn.Module):
    """GTCRN 尺度投影 log-ratio，仅适用于有效 speech reference。

    loss 公式与官方 GTCRN 逐项对齐；直谱路径的 iSTFT 继承模型顶层 STFT
    的窗函数与几何配置（analysis/synthesis 同窗），``torch.istft`` 自带的
    窗包络归一化保证一致谱精确重建，不做 loss 私有 sqrt-Hann；
    未提供直谱时保留 waveform adapter。
    """

    REQUIRES_SPEECH_REFERENCE = True
    SUPPORTS_DIRECT_SPECTRA = True
    REQUIRES_MATCHING_STFT_CONTEXT = True

    def __init__(
        self,
        eps: float = 1e-8,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        center: bool = True,
        normalized: bool = False,
        window: str = "hann",
    ):
        super().__init__()
        eps = _validate_finite_real(eps, field="eps")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        # 继承框架顶层 STFT 契约：NOLA/几何校验与窗名规范化交给
        # resolve_stft_config，非法组合（如 hann center=False、hop>win）
        # 在构造期 fail-fast；合法组合（hamming、center=False、normalized=True）放行。
        params = resolve_stft_config(
            {
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": win_length,
                "window": window,
                "center": center,
                "normalized": normalized,
            }
        )
        if params["n_fft"] < 2:
            raise ValueError("n_fft must be at least 2")

        self.eps = eps
        self.n_fft = params["n_fft"]
        self.hop_length = params["hop_length"]
        self.win_length = params["win_length"]
        self.center = params["center"]
        self.normalized = params["normalized"]
        self.window_name = params["window"]
        # 合成窗与框架 analysis 窗相同，依赖 istft 包络归一化精确重建。
        self.register_buffer(
            "_window",
            _build_stft_window(self.window_name, self.win_length),
            persistent=False,
        )

    def _projected_log_ratio(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """逐样本执行官方投影乘除顺序，并返回 loss 与有效参考 mask。"""
        target_energy = torch.sum(target.square(), dim=-1, keepdim=True)
        valid = target_energy.squeeze(-1) > self.eps
        projection = (
            torch.sum(target * prediction, dim=-1, keepdim=True) * target
        ) / (target_energy + self.eps)
        residual = prediction - projection
        ratio = torch.sum(projection.square(), dim=-1) / (
            torch.sum(residual.square(), dim=-1) + self.eps
        )
        return -torch.log10(ratio + self.eps), valid

    def _direct_spectrum_log_ratio(
        self,
        predicted_spectra: torch.Tensor,
        target_spectra: torch.Tensor,
        frame_lengths: torch.Tensor,
        sample_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """按有效 frame count 分组重建波形，可选施加有效长度掩码后取样本均值。

        提供 ``sample_lengths`` 时，对每组重建波形施加
        ``t < sample_lengths`` 时间掩码（右 padding 区梯度为零），
        只统计有效覆盖区；聚合保持 per-sample mean（valid 样本均值）。
        """
        total = (predicted_spectra.real * 0.0).sum()
        valid_count = torch.zeros(
            (), device=predicted_spectra.device, dtype=predicted_spectra.real.dtype
        )
        synthesis_window = self._window.to(
            device=predicted_spectra.device,
            dtype=predicted_spectra.real.dtype,
        )
        for frame_count in torch.unique(frame_lengths, sorted=True).tolist():
            frame_count = int(frame_count)
            if self.center:
                output_length = self.hop_length * (frame_count - 1) + self.n_fft % 2
            else:
                output_length = self.n_fft + self.hop_length * (frame_count - 1)
            if output_length < 1:
                # 仅 center=True、偶数 n_fft、单帧时出现：重建波形恒为空，
                # 这些样本目标能量为零，交由 clean-energy guard 判 invalid。
                continue
            indices = (frame_lengths == frame_count).nonzero(as_tuple=True)[0]
            predicted_group = predicted_spectra.index_select(0, indices)[
                ..., :frame_count
            ]
            target_group = target_spectra.index_select(0, indices)[..., :frame_count]
            predicted_waveform = torch.istft(
                predicted_group,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=synthesis_window,
                center=self.center,
                normalized=self.normalized,
                onesided=True,
                length=None,
            )
            target_waveform = torch.istft(
                target_group,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=synthesis_window,
                center=self.center,
                normalized=self.normalized,
                onesided=True,
                length=None,
            )
            if sample_lengths is not None:
                group_lengths = sample_lengths.index_select(0, indices)
                t = torch.arange(output_length, device=predicted_waveform.device)
                time_mask = (
                    t[None, :] < group_lengths.clamp(max=output_length)[:, None]
                ).to(predicted_waveform.dtype)
                predicted_waveform = predicted_waveform * time_mask
                target_waveform = target_waveform * time_mask
            group_loss, valid = self._projected_log_ratio(
                predicted_waveform, target_waveform
            )
            valid_f = valid.to(group_loss.dtype)
            total = total + (group_loss * valid_f).sum()
            valid_count = valid_count + valid_f.sum()
        return total / valid_count.clamp(min=1.0)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        enhanced_spectra: torch.Tensor | None = None,
        target_spectra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_waveform_inputs(enhanced, clean, lengths)
        if (enhanced_spectra is None) != (target_spectra is None):
            raise ValueError(
                "enhanced_spectra and target_spectra must be provided together"
            )

        enhanced = enhanced.float()
        clean = clean.float()
        if enhanced_spectra is not None and target_spectra is not None:
            predicted_spectra = _as_complex_spectrum(
                enhanced_spectra,
                name="enhanced_spectra",
                batch_size=enhanced.shape[0],
                n_fft=self.n_fft,
            )
            reference_spectra = _as_complex_spectrum(
                target_spectra,
                name="target_spectra",
                batch_size=enhanced.shape[0],
                n_fft=self.n_fft,
            )
            if (
                predicted_spectra.device != enhanced.device
                or reference_spectra.device != enhanced.device
            ):
                raise ValueError("direct spectra must be on the waveform device")
            if predicted_spectra.shape != reference_spectra.shape:
                raise ValueError(
                    "enhanced_spectra/target_spectra shape mismatch: "
                    f"{tuple(predicted_spectra.shape)} vs "
                    f"{tuple(reference_spectra.shape)}"
                )
            if lengths is None:
                frame_lengths = torch.full(
                    (enhanced.shape[0],),
                    predicted_spectra.shape[-1],
                    device=enhanced.device,
                    dtype=torch.long,
                )
                sample_lengths = None
            else:
                frame_lengths = compute_spec_frame_lengths(
                    lengths,
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    hop_length=self.hop_length,
                    center=self.center,
                ).to(device=enhanced.device, dtype=torch.long)
                sample_lengths = lengths
            if bool((frame_lengths > predicted_spectra.shape[-1]).any().item()):
                raise ValueError(
                    "direct spectrum context does not contain all frames required "
                    "to cover the effective waveform lengths"
                )
            return self._direct_spectrum_log_ratio(
                predicted_spectra, reference_spectra, frame_lengths, sample_lengths
            )

        if lengths is not None:
            time_mask = (
                torch.arange(clean.shape[1], device=clean.device)[None, :]
                < lengths[:, None]
            )
            enhanced = enhanced * time_mask
            clean = clean * time_mask
        log_ratio, valid = self._projected_log_ratio(enhanced, clean)

        # 不分支读取 valid.any()，全无效 batch 仍返回连接 enhanced 图的零。
        valid_f = valid.to(log_ratio.dtype)
        return (log_ratio * valid_f).sum() / valid_f.sum().clamp(min=1.0)
