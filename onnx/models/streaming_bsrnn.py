"""Streaming BSRNN wrapper for inference and ONNX export.

Wraps a training BSRNN (spectrum-in/spectrum-out) and exposes
initialize_cache() and forward_streaming(frame_spec, cache) so that
StreamingInferenceWrapper and onnx/export.py can use it. Streaming logic
lives here; models/bsrnn is training-only.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class StreamingBSRNNWrapper(nn.Module):
    """Wraps BSRNN (from tasks.se.models.bsrnn) and adds streaming API.

    Expects model to have .bsrnn (core) with causal=True. Uses core's
    band_split, norm_time, rnn_time, fc_time, norm_freq, rnn_freq, fc_freq,
    mask_decoder to run one frame with LSTM cache.
    """

    def __init__(self, model: nn.Module, target_fs: int = 48000):
        super().__init__()
        self.model = model
        self.target_fs = target_fs
        core = getattr(model, "bsrnn", None)
        if core is None:
            raise ValueError("StreamingBSRNNWrapper expects model.bsrnn (BSRNN core)")
        if not getattr(core, "causal", True):
            raise ValueError("Streaming BSRNN requires causal=True")
        self._core = core
        self.n_fft = getattr(model, "n_fft", 2048)
        self.hop_length = getattr(model, "hop_length", 512)

    @property
    def bsrnn(self):
        """Expose core for StreamingONNXWrapper.get_input_names (num_layer)."""
        return self._core

    @property
    def algorithmic_delay(self) -> int:
        return self.n_fft - self.hop_length

    def _initialize_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        K = self._core.num_subbands
        hdim = self._core.hidden_dim
        return [
            (
                torch.zeros(1, batch_size * K, hdim, device=device, dtype=dtype),
                torch.zeros(1, batch_size * K, hdim, device=device, dtype=dtype),
            )
            for _ in range(self._core.num_layer)
        ]

    def initialize_cache(
        self,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        dev = torch.device(device)
        lstm = self._initialize_cache(batch_size, dev, dtype)
        return {"lstm_cache": lstm}

    def forward_streaming(
        self, frame_spec: torch.Tensor, cache: dict, fs: int | None = None
    ) -> Tuple[torch.Tensor, dict]:
        """One STFT frame in, enhanced frame out; cache updated."""
        if fs is None:
            fs = self.target_fs
        core = self._core
        z = core.band_split(frame_spec, fs=fs)
        B, N, _, K = z.shape
        skip = z
        cache_list = cache["lstm_cache"]
        cache_new: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i in range(core.num_layer):
            out = core.norm_time[i](skip)
            out = out.transpose(1, 3).reshape(B * K, 1, N)
            out, hc = core.rnn_time[i](out, cache_list[i])
            cache_new.append(hc)
            out = core.fc_time[i](out)
            out = out.reshape(B, K, 1, N).transpose(1, 3)
            skip = skip + out

            out = core.norm_freq[i](skip)
            out = out.permute(0, 2, 3, 1).contiguous().reshape(B, K, N)
            out, _ = core.rnn_freq[i](out)
            out = core.fc_freq[i](out)
            out = out.reshape(B, 1, K, N).permute(0, 3, 1, 2).contiguous()
            skip = skip + out

        m, r = core.mask_decoder(skip)
        m = torch.view_as_complex(m)
        r = torch.view_as_complex(r)
        x_c = torch.view_as_complex(frame_spec)
        m = m[..., : x_c.size(-1)]
        r = r[..., : x_c.size(-1)]
        out = torch.view_as_real(m * x_c.unsqueeze(1) + r)
        return out[:, 0], {"lstm_cache": cache_new}

    def initialize_cache_flat(
        self,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> list:
        lstm = self._initialize_cache(batch_size, torch.device(device), dtype)
        flat = []
        for h, c in lstm:
            flat.extend([h, c])
        return flat

    def forward_streaming_flat_spec(
        self, frame_spec: torch.Tensor, *lstm_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """ONNX-friendly: real-valued mask, flat cache in/out."""
        core = self._core
        n = core.num_layer * 2
        if len(lstm_flat) != n:
            raise ValueError(f"Expected {n} cache tensors, got {len(lstm_flat)}")
        cache_list = [(lstm_flat[i], lstm_flat[i + 1]) for i in range(0, n, 2)]

        z = core.band_split(frame_spec, fs=self.target_fs)
        B, N, _, K = z.shape
        skip = z
        cache_new: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i in range(core.num_layer):
            out = core.norm_time[i](skip)
            out = out.transpose(1, 3).reshape(B * K, 1, N)
            out, hc = core.rnn_time[i](out, cache_list[i])
            cache_new.append(hc)
            out = core.fc_time[i](out)
            out = out.reshape(B, K, 1, N).transpose(1, 3)
            skip = skip + out

            out = core.norm_freq[i](skip)
            out = out.permute(0, 2, 3, 1).contiguous().reshape(B, K, N)
            out, _ = core.rnn_freq[i](out)
            out = core.fc_freq[i](out)
            out = out.reshape(B, 1, K, N).permute(0, 3, 1, 2).contiguous()
            skip = skip + out

        m, r = core.mask_decoder(skip)
        F_dim = frame_spec.size(2)
        m = m[..., :F_dim, :]
        r = r[..., :F_dim, :]
        x_r = frame_spec[..., 0].unsqueeze(2)
        x_i = frame_spec[..., 1].unsqueeze(2)
        m_r, m_i = m[..., 0], m[..., 1]
        out_r = m_r * x_r - m_i * x_i + r[..., 0]
        out_i = m_r * x_i + m_i * x_r + r[..., 1]
        out = torch.stack([out_r, out_i], dim=-1)
        out = out[:, 0]

        out_flat = []
        for h, c in cache_new:
            out_flat.extend([h, c])
        return (out, *out_flat)
