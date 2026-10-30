"""Universal streaming inference wrapper (Streaming Inference Wrapper).

Models are spectrum-in/spectrum-out; STFT/iSTFT is done outside the model.

  1. PyTorch: StreamingInferenceWrapper(stft_module, model) does wav -> frame STFT -> model -> iSTFT -> OLA.
  2. ONNX:    StreamingONNXWrapper wraps spec-in/spec-out (frame_spec + cache -> enhanced_spec + cache).

Usage example (PyTorch):
    stft_module = build_stft_istft(cfg["stft"])
    wrapper = StreamingInferenceWrapper(stft_module, model, device)
    enhanced = wrapper.process(noisy_wav)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def _frame_stft(stft_module: nn.Module, wav_frame: torch.Tensor) -> torch.Tensor:
    """(B, n_fft) -> (B, 1, F, 2) for one frame."""
    spec = stft_module(wav_frame)
    return spec.permute(0, 2, 1, 3)


def _frame_istft(stft_module: nn.Module, frame_spec: torch.Tensor, length: int) -> torch.Tensor:
    """(B, 1, F, 2) -> (B, length) for one frame."""
    spec = frame_spec.permute(0, 2, 1, 3)
    return stft_module.inverse(spec, length)


class StreamingInferenceWrapper:
    """Streaming wrapper: STFT outside model, frame-by-frame model(spec) -> iSTFT + OLA.

    Args:
        stft_module: STFT/iSTFT module (same n_fft/hop as model).
        model: Model with initialize_cache() and forward_streaming(frame_spec, cache).
        device: Inference device.
    """

    def __init__(
        self,
        stft_module: nn.Module,
        model: nn.Module,
        device: str | torch.device = "cpu",
    ):
        self.stft_module = stft_module.to(device)
        self.model = model
        self.device = torch.device(device)
        if not hasattr(model, "forward_streaming"):
            raise ValueError(
                f"{type(model).__name__} does not support streaming. "
                "Model must implement initialize_cache() and forward_streaming(frame_spec, cache)."
            )
        self._hop = getattr(model, "hop_length", getattr(stft_module, "hop_length"))
        self._n_fft = getattr(stft_module, "n_fft", None) or getattr(model, "n_fft")
        self._win_buf = getattr(stft_module, "_win_buf", None)

    @torch.inference_mode()
    def process(self, wav: torch.Tensor) -> torch.Tensor:
        """Process full waveform: chunk by hop, STFT -> model(frame_spec) -> iSTFT -> overlap-add."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self.device)
        B, T = wav.shape
        if T == 0:
            return wav.clone()
        hop = self._hop
        n_fft = self._n_fft

        pad_len = (hop - (T - n_fft) % hop) % hop if T >= n_fft else n_fft - T
        if pad_len > 0:
            wav = torch.nn.functional.pad(wav, (0, pad_len))
        T_pad = wav.shape[-1]
        num_frames = max(0, (T_pad - n_fft) // hop + 1)
        if num_frames == 0:
            return wav[:, :T].clone()

        cache = self.model.initialize_cache(B, self.device, wav.dtype)
        out_len = (num_frames - 1) * hop + n_fft
        output = torch.zeros(B, out_len, device=wav.device, dtype=wav.dtype)
        window = self._win_buf
        if window is not None:
            window = window.to(wav.device)
            if window.shape[0] < n_fft:
                window = torch.nn.functional.pad(window, (0, n_fft - window.shape[0]))
            elif window.shape[0] > n_fft:
                window = window[:n_fft]
            window_sum = torch.zeros(out_len, device=wav.device, dtype=wav.dtype)
        else:
            window_sum = torch.ones(out_len, device=wav.device, dtype=wav.dtype)

        for i in range(num_frames):
            start = i * hop
            wav_frame = wav[:, start : start + n_fft]
            if wav_frame.shape[-1] < n_fft:
                wav_frame = torch.nn.functional.pad(wav_frame, (0, n_fft - wav_frame.shape[-1]))
            frame_spec = _frame_stft(self.stft_module, wav_frame)
            out_spec, cache = self.model.forward_streaming(frame_spec, cache)
            out_frame = _frame_istft(self.stft_module, out_spec, n_fft)
            output[:, start : start + n_fft] += out_frame * (window if window is not None else 1.0)
            if window is not None:
                window_sum[start : start + n_fft] += window

        if window is not None:
            window_sum = window_sum.clamp(min=1e-8)
            output = output / window_sum

        delay = getattr(self.model, "algorithmic_delay", 0)
        enhanced = output[:, delay : delay + T]
        if enhanced.shape[-1] < T:
            enhanced = torch.nn.functional.pad(enhanced, (0, T - enhanced.shape[-1]))
        return enhanced

    def verify_consistency(
        self, wav: torch.Tensor,
        mse_tol: float = 1e-4,
        max_abs_tol: float = 1e-3,
    ) -> dict:
        """Compare streaming vs full: full = STFT -> model(spec) -> iSTFT."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self.device)
        self.model.eval()
        T = wav.shape[-1]

        with torch.inference_mode():
            full_spec = self.stft_module(wav)
            full_out_spec = self.model(full_spec)
            if isinstance(full_out_spec, (tuple, list)):
                full_out_spec = full_out_spec[0]
            full_out = self.stft_module.inverse(full_out_spec, T)
            stream_out = self.process(wav)

        min_len = min(full_out.shape[-1], stream_out.shape[-1])
        full_out = full_out[..., :min_len]
        stream_out = stream_out[..., :min_len]

        mse = torch.mean((full_out - stream_out) ** 2).item()
        max_diff = torch.max(torch.abs(full_out - stream_out)).item()

        return {
            "mse": mse,
            "max_abs_diff": max_diff,
            "pass": mse < mse_tol and max_diff < max_abs_tol,
        }


class StreamingAECInferenceWrapper:
    """Streaming wrapper for dual-input AEC models (mic + far-end).

    流式模型应来自 onnx/ 下对应 AEC 流式封装；训练用谱域模型不提供流式 API 时需先转为 onnx 流式实现。

    Model must implement initialize_cache(B, device, dtype) and
    forward_streaming(mic_spec, far_spec, cache) -> (enh_spec, cache).
    STFT/iSTFT is done outside the model; process() runs frame-by-frame OLA.
    """

    def __init__(
        self,
        stft_module: nn.Module,
        model: nn.Module,
        device: str | torch.device = "cpu",
    ):
        self.stft_module = stft_module.to(device)
        self.model = model
        self.device = torch.device(device)
        if not hasattr(model, "forward_streaming") or not hasattr(model, "initialize_cache"):
            raise ValueError(
                f"{type(model).__name__} must implement initialize_cache() and "
                "forward_streaming(mic_spec, far_spec, cache)."
            )
        self._hop = getattr(model, "hop_length", getattr(stft_module, "hop_length"))
        self._n_fft = getattr(stft_module, "n_fft", None) or getattr(model, "n_fft")
        self._win_buf = getattr(stft_module, "_win_buf", None)

    @torch.inference_mode()
    def process(self, mic_wav: torch.Tensor, far_wav: torch.Tensor) -> torch.Tensor:
        """处理严格匹配的双路浮点 waveform，避免错配 AEC 帧进入模型。"""
        if not isinstance(mic_wav, torch.Tensor) or not isinstance(far_wav, torch.Tensor):
            raise TypeError("mic_wav and far_wav must be torch.Tensor instances")
        if mic_wav.dim() == 1:
            mic_wav = mic_wav.unsqueeze(0)
        if far_wav.dim() == 1:
            far_wav = far_wav.unsqueeze(0)
        if mic_wav.dim() != 2 or far_wav.dim() != 2:
            raise ValueError(
                f"mic_wav and far_wav must be rank-1 or rank-2, got "
                f"{mic_wav.shape} and {far_wav.shape}"
            )
        if not mic_wav.is_floating_point() or not far_wav.is_floating_point():
            raise TypeError("mic_wav and far_wav must use floating-point dtypes")
        if mic_wav.shape != far_wav.shape:
            raise ValueError(
                f"mic_wav and far_wav must have matching batch/length, got "
                f"{mic_wav.shape} and {far_wav.shape}"
            )
        if mic_wav.dtype != far_wav.dtype:
            raise ValueError(
                f"mic_wav and far_wav must have matching dtype, got "
                f"{mic_wav.dtype} and {far_wav.dtype}"
            )
        if not torch.isfinite(mic_wav).all() or not torch.isfinite(far_wav).all():
            raise ValueError("mic_wav and far_wav must contain only finite values")
        mic_wav = mic_wav.to(self.device)
        far_wav = far_wav.to(self.device)
        B, T = mic_wav.shape
        if T == 0:
            return mic_wav.clone()
        hop = self._hop
        n_fft = self._n_fft
        pad_len = (hop - (T - n_fft) % hop) % hop if T >= n_fft else n_fft - T
        if pad_len > 0:
            mic_wav = torch.nn.functional.pad(mic_wav, (0, pad_len))
            far_wav = torch.nn.functional.pad(far_wav, (0, pad_len))
        T_pad = mic_wav.shape[-1]
        num_frames = max(0, (T_pad - n_fft) // hop + 1)
        if num_frames == 0:
            return mic_wav[:, :T].clone()

        cache = self.model.initialize_cache(B, self.device, mic_wav.dtype)
        out_len = (num_frames - 1) * hop + n_fft
        output = torch.zeros(B, out_len, device=mic_wav.device, dtype=mic_wav.dtype)
        window = self._win_buf
        if window is not None:
            window = window.to(mic_wav.device)
            if window.shape[0] < n_fft:
                window = torch.nn.functional.pad(window, (0, n_fft - window.shape[0]))
            elif window.shape[0] > n_fft:
                window = window[:n_fft]
            window_sum = torch.zeros(out_len, device=mic_wav.device, dtype=mic_wav.dtype)
        else:
            window_sum = torch.ones(out_len, device=mic_wav.device, dtype=mic_wav.dtype)

        for i in range(num_frames):
            start = i * hop
            mic_frame = mic_wav[:, start : start + n_fft]
            far_frame = far_wav[:, start : start + n_fft]
            if mic_frame.shape[-1] < n_fft:
                mic_frame = torch.nn.functional.pad(mic_frame, (0, n_fft - mic_frame.shape[-1]))
                far_frame = torch.nn.functional.pad(far_frame, (0, n_fft - far_frame.shape[-1]))
            mic_spec = _frame_stft(self.stft_module, mic_frame)   # (B, 1, F, 2)
            far_spec = _frame_stft(self.stft_module, far_frame)   # (B, 1, F, 2)
            mic_spec = mic_spec.permute(0, 2, 1, 3)   # (B, F, 1, 2) for AEC model
            far_spec = far_spec.permute(0, 2, 1, 3)
            out_spec, cache = self.model.forward_streaming(mic_spec, far_spec, cache)  # (B, F, 1, 2)
            out_frame = self.stft_module.inverse(out_spec, n_fft)
            output[:, start : start + n_fft] += out_frame * (window if window is not None else 1.0)
            if window is not None:
                window_sum[start : start + n_fft] += window

        if window is not None:
            window_sum = window_sum.clamp(min=1e-8)
            output = output / window_sum

        delay = getattr(self.model, "algorithmic_delay", 0)
        enhanced = output[:, delay : delay + T]
        if enhanced.shape[-1] < T:
            enhanced = torch.nn.functional.pad(enhanced, (0, T - enhanced.shape[-1]))
        return enhanced


class StreamingONNXWrapper(nn.Module):
    """ONNX-exportable spec-in/spec-out streaming wrapper.

    STFT/iSTFT is done by the application; this wrapper only exposes
    frame_spec + LSTM cache -> enhanced_spec + cache.

    forward(frame_spec, h_0, c_0, ..., h_{L-1}, c_{L-1})
    -> (enhanced_spec, h_0_out, c_0_out, ..., h_{L-1}_out, c_{L-1}_out)

    Args:
        model: Streaming BSRNN with initialize_cache_flat / forward_streaming_flat_spec.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, frame_spec: torch.Tensor, *cache: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.model.forward_streaming_flat_spec(frame_spec, *cache)

    def initialize_cache(
        self, batch_size: int = 1, device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> List[torch.Tensor]:
        return self.model.initialize_cache_flat(batch_size, device, dtype)

    def get_input_names(self) -> List[str]:
        names = ["frame_spec"]
        for i in range(self.model.bsrnn.num_layer):
            names.extend([f"h_{i}", f"c_{i}"])
        return names

    def get_output_names(self) -> List[str]:
        names = ["enhanced_spec"]
        for i in range(self.model.bsrnn.num_layer):
            names.extend([f"h_{i}_out", f"c_{i}_out"])
        return names

    def get_dynamic_axes(self) -> dict:
        axes = {}
        for name in self.get_input_names() + self.get_output_names():
            axes[name] = {0: "batch_size"}
        return axes
