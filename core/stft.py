"""Framework-level STFT/iSTFT module for train/val.

Used by the trainer to convert waveform <-> spectrum so that models
receive spectrum and return spectrum (no STFT/iSTFT inside models).

Config comes from YAML top-level "stft" block: n_fft, hop_length, win_length,
window, center, normalized. STFT and iSTFT use the same parameters (same block).
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

# 全量有限性检查会物化整块布尔张量并强制 GPU→CPU 同步，默认关闭；需要显式
# 逐元素验证时设置 USE_TRAIN_STFT_VALIDATE_FINITE=1，或在进程内改写该开关。
VALIDATE_FINITE_INPUTS = (
    os.environ.get("USE_TRAIN_STFT_VALIDATE_FINITE", "").strip().lower()
    in {"1", "true", "yes", "on"}
)


def _get_window(name: str, length: int, device: torch.device) -> torch.Tensor:
    if name == "hann":
        return torch.hann_window(length, device=device)
    if name == "hamming":
        return torch.hamming_window(length, device=device)
    raise ValueError(f"Unknown window '{name}'. Supported: hann, hamming.")


def _validate_positive_int(value: Any, *, field: str) -> int:
    """校验 STFT 样本数为精确正整数，禁止隐式数值转换。"""
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"stft.{field} must be an exact positive int, got {value!r}"
        )
    return value


def _require_finite(tensor: torch.Tensor, message: str) -> None:
    """可选的全量有限性门禁；仅在显式开启时做同步检查。"""
    if VALIDATE_FINITE_INPUTS and not torch.isfinite(tensor).all().item():
        raise ValueError(message)


def _validate_waveform_input(wav: Any) -> torch.Tensor:
    """校验 STFT 波形输入，避免底层算子隐式转换或产生不透明错误。"""
    if not isinstance(wav, torch.Tensor):
        raise TypeError(f"stft.forward waveform must be a torch.Tensor, got {type(wav).__name__}")
    if wav.ndim != 2:
        raise ValueError(
            f"stft.forward waveform must have shape (B, T), got {tuple(wav.shape)}"
        )
    if wav.shape[0] <= 0 or wav.shape[1] <= 0:
        raise ValueError(
            f"stft.forward waveform must have non-empty batch and time dimensions, got {tuple(wav.shape)}"
        )
    if not wav.is_floating_point():
        raise TypeError(
            f"stft.forward waveform must have a floating-point dtype, got {wav.dtype}"
        )
    _require_finite(wav, "stft.forward waveform must contain only finite values")
    return wav


def _validate_spectrum_input(spec: Any, *, n_fft: int) -> torch.Tensor:
    """校验 iSTFT 频谱输入的布局、dtype 与数值范围。"""
    if not isinstance(spec, torch.Tensor):
        raise TypeError(f"stft.inverse spectrum must be a torch.Tensor, got {type(spec).__name__}")

    expected_bins = n_fft // 2 + 1
    if spec.is_complex():
        valid_layout = spec.ndim == 3
        frame_count = spec.shape[-1] if valid_layout else 0
    else:
        valid_layout = spec.ndim == 4 and spec.shape[-1] == 2
        frame_count = spec.shape[-2] if valid_layout else 0
    if not valid_layout:
        raise ValueError(
            "stft.inverse spectrum must have shape (B, F, T) complex or "
            f"(B, F, T, 2) real/imag, got {tuple(spec.shape)}"
        )
    if spec.shape[0] <= 0 or spec.shape[1] != expected_bins or frame_count <= 0:
        raise ValueError(
            "stft.inverse spectrum must have a non-empty batch/frame dimension and "
            f"F={expected_bins}, got {tuple(spec.shape)}"
        )
    if not (spec.is_complex() or spec.is_floating_point()):
        raise TypeError(
            "stft.inverse spectrum must have a floating-point or complex dtype, "
            f"got {spec.dtype}"
        )
    _require_finite(spec, "stft.inverse spectrum must contain only finite values")
    return spec


def _validate_stft_parameters(
    *,
    n_fft: Any,
    hop_length: Any,
    win_length: Any,
    window: Any,
    center: Any,
    normalized: Any = False,
) -> dict[str, Any]:
    """静态校验共享 STFT/iSTFT 几何与受支持窗口的 NOLA 契约。"""
    n_fft = _validate_positive_int(n_fft, field="n_fft")
    hop_length = _validate_positive_int(hop_length, field="hop_length")
    win_length = _validate_positive_int(win_length, field="win_length")
    if not hop_length <= win_length <= n_fft:
        raise ValueError(
            "stft geometry must satisfy hop_length <= win_length <= n_fft; "
            f"got hop_length={hop_length}, win_length={win_length}, n_fft={n_fft}"
        )

    if type(window) is not str or window.lower() not in {"hann", "hamming"}:
        raise ValueError(
            "stft.window must be one of: hann, hamming; "
            f"got {window!r}"
        )
    window = window.lower()
    if type(center) is not bool:
        raise ValueError(f"stft.center must be a bool, got {center!r}")
    if type(normalized) is not bool:
        raise ValueError(f"stft.normalized must be a bool, got {normalized!r}")

    # torch.stft 会把短窗零填充到 n_fft；无中心时该零边界无法由其他帧覆盖。
    if not center and win_length < n_fft:
        raise ValueError(
            "stft geometry/window/center does not satisfy torch.istft NOLA: "
            "center=False requires win_length == n_fft because PyTorch pads "
            f"short windows with zeros; got win_length={win_length}, n_fft={n_fft}"
        )
    # periodic Hann 的首样本严格为零，无中心时波形左边界必然不可重建。
    if not center and window == "hann":
        raise ValueError(
            "stft geometry/window/center does not satisfy torch.istft NOLA: "
            "PyTorch periodic Hann is zero at the left boundary with center=False"
        )
    # 对每个 hop residue 累加无限帧稳态窗平方；其最小值就是内部 NOLA 下界。
    window_tensor = _get_window(window, win_length, torch.device("cpu"))
    overlap_envelope = torch.zeros(hop_length)
    overlap_envelope.scatter_add_(
        0,
        torch.arange(win_length).remainder(hop_length),
        window_tensor.square(),
    )
    minimum_envelope = float(overlap_envelope.min())
    if minimum_envelope <= 1e-11:
        raise ValueError(
            "stft geometry/window/center does not satisfy torch.istft NOLA "
            "numerical threshold: minimum overlap envelope must be > 1e-11; "
            f"got {minimum_envelope:.6g} for n_fft={n_fft}, "
            f"hop_length={hop_length}, win_length={win_length}, "
            f"window={window!r}, center={center}"
        )

    return {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "window": window,
        "center": center,
        "normalized": normalized,
    }


class StftIstftModule(nn.Module):
    """Single module with forward (wav -> spec) and inverse (spec -> wav).

    Both use the same n_fft, hop_length, win_length, window, center, and normalized.
    For legal ``center=False`` geometries, forward right-pads to a complete frame
    boundary so inverse may safely crop back to the caller's original length.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
    ):
        super().__init__()
        params = _validate_stft_parameters(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            normalized=normalized,
        )
        self.n_fft = params["n_fft"]
        self.hop_length = params["hop_length"]
        self.win_length = params["win_length"]
        self.window_name = params["window"]
        self.center = params["center"]
        self.normalized = params["normalized"]
        w = _get_window(self.window_name, self.win_length, torch.device("cpu"))
        stable_support = torch.nonzero(w.square() > 1e-11).flatten()
        self._window_support_end = (
            (self.n_fft - self.win_length) // 2 + int(stable_support[-1])
        )
        self.register_buffer("_win_buf", w, persistent=False)

    def _right_pad_noncentered_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        """将无中心 STFT 输入补至完整帧边界，保证 iSTFT 可裁回原始长度。"""
        if self.center:
            return wav
        samples = wav.shape[-1]
        minimum_frames = max(samples, self.n_fft)
        padded_length = self.n_fft + (
            (minimum_frames - self.n_fft + self.hop_length - 1) // self.hop_length
        ) * self.hop_length
        if padded_length == samples:
            return wav
        return torch.nn.functional.pad(wav, (0, padded_length - samples))

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Waveform (B, T) -> complex spectrum (B, F, T), returned as (B, F, T, 2) real."""
        wav = _validate_waveform_input(wav)
        wav = self._right_pad_noncentered_waveform(wav)
        window = self._win_buf.to(wav.device)
        spec = torch.stft(
            wav.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            return_complex=True,
            onesided=True,
        )
        return torch.view_as_real(spec)

    def _maximum_inverse_length(self, frame_count: int) -> int:
        """返回实际 synthesis frames 中满足 PyTorch NOLA 阈值的右边界。"""
        left_trim = self.n_fft // 2 if self.center else 0
        return (
            (frame_count - 1) * self.hop_length
            + self._window_support_end
            + 1
            - left_trim
        )

    @torch.amp.autocast("cuda", enabled=False)
    def inverse(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """Spectrum (B, F, T, 2) or complex (B, F, T) -> waveform (B, T)."""
        spec = _validate_spectrum_input(spec, n_fft=self.n_fft)
        length = _validate_positive_int(length, field="inverse_length")
        frame_count = spec.shape[-1] if spec.is_complex() else spec.shape[-2]
        maximum_length = self._maximum_inverse_length(frame_count)
        if length > maximum_length:
            raise ValueError(
                "stft.inverse requested length="
                f"{length} exceeds maximum covered length={maximum_length} for "
                f"frame_count={frame_count}; refusing silent zero-padding at the tail"
            )

        window = self._win_buf.to(spec.device)
        if spec.is_complex():
            spec_c = spec
        else:
            spec_c = torch.view_as_complex(spec.float().contiguous())
        return torch.istft(
            spec_c,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            length=length,
            onesided=True,
        )


def resolve_stft_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """解析并严格校验顶层 STFT 配置，返回框架实际使用的有效参数。"""
    if not cfg:
        raise KeyError("[stft] config is required for spectrum-in/spectrum-out pipeline.")
    n_fft = cfg["n_fft"]
    return _validate_stft_parameters(
        n_fft=n_fft,
        hop_length=cfg["hop_length"],
        win_length=cfg.get("win_length", n_fft),
        window=cfg.get("window", "hann"),
        center=cfg.get("center", True),
        normalized=cfg.get("normalized", False),
    )


def build_stft_istft(cfg: dict[str, Any]) -> StftIstftModule:
    """Build StftIstftModule from top-level 'stft' config.

    cfg must contain: n_fft, hop_length, win_length, window (optional, default hann).
    """
    return StftIstftModule(**resolve_stft_config(cfg))


# 模型构造只消费其既有五项；loss context 还需匹配 framework normalized 语义。
STFT_CONFIG_KEYS = ("n_fft", "hop_length", "win_length", "window", "center")
LOSS_STFT_CONFIG_KEYS = STFT_CONFIG_KEYS + ("normalized",)


def inject_stft_defaults(
    args: dict[str, Any],
    stft_cfg: dict[str, Any] | None = None,
    keys: tuple[str, ...] = STFT_CONFIG_KEYS,
) -> dict[str, Any]:
    """Return a copy of ``args`` with STFT defaults from ``stft_cfg`` applied via setdefault.

    Only keys that are explicitly declared (non-None) in ``stft_cfg`` are
    injected; explicit values already in ``args`` always win. Shared by
    ``build_model`` and ``build_loss`` (the latter passes a per-class filtered
    ``keys`` so only parameters a loss class actually accepts are injected).
    """
    if not stft_cfg:
        return dict(args)
    out = dict(args)
    for key in keys:
        value = stft_cfg.get(key)
        if value is not None:
            out.setdefault(key, value)
    return out
