"""Streaming FastEnhancer wrapper for ONNX export and frame-by-frame inference.

Reference: https://github.com/aask1357/fastenhancer
Official ONNX: spec-in / spec-out with GRU hidden-state caches per RNNFormerBlock.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from tasks.se.models.discriminative.fastenhancer import FastEnhancer


class StreamingFastEnhancer(nn.Module):
    """Frame-by-frame streaming wrapper around FastEnhancer.

    Processes one STFT frame at a time (B, F, 1, 2) → (B, F, 1, 2).
    Maintains GRU hidden states for each RNNFormerBlock as cache.

    The base model's ``_forward_spectrum`` strips the Nyquist bin internally,
    so the ONNX input/output keeps the full ``n_fft//2+1`` freq bins.
    """

    def __init__(
        self,
        base_model: FastEnhancer,
        *,
        hop_length: int = 256,
        win_length: int | None = None,
        sample_rate: int = 16000,
        window: str = "hann",
        center: bool = True,
    ):
        super().__init__()
        if hop_length <= 0 or sample_rate <= 0:
            raise ValueError("hop_length and sample_rate must be positive")
        if win_length is not None and (win_length <= 0 or win_length > base_model.n_fft):
            raise ValueError("win_length must be positive and no greater than n_fft")
        if window != "hann":
            raise ValueError(f"FastEnhancer supports only window='hann', got {window!r}")
        self.base = base_model
        self.hop_length = int(hop_length)
        self.win_length = int(win_length if win_length is not None else base_model.n_fft)
        self.target_fs = int(sample_rate)
        self.window = window
        self.center = bool(center)
        self.n_blocks = len(base_model.rf_block)
        self._cache_keys = [f"cache_{i}" for i in range(self.n_blocks)]

    @property
    def CACHE_KEYS(self) -> list[str]:
        return self._cache_keys

    @property
    def n_fft(self) -> int:
        return self.base.n_fft

    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> dict[str, Tensor]:
        """Return zero-initialized caches for all RNNFormerBlocks.

        Cache shape per block: (1, rf_freq * batch_size, rf_ch).
        When batch_size==1 this matches the official (1, rf_freq, rf_ch).
        """
        cache: dict[str, Tensor] = {}
        for i, block in enumerate(self.base.rf_block):
            freq = block.freq
            ch = block.channels
            cache[f"cache_{i}"] = torch.zeros(1, freq * batch_size, ch, device=device, dtype=dtype)
        return cache

    def forward(self, spec_frame: Tensor, cache: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Process a single STFT frame.

        Args:
            spec_frame: (B, F, 1, 2) real-valued spectrum with Nyquist.
            cache: dict of GRU hidden states, one per RNNFormerBlock.

        Returns:
            enh_frame: (B, F, 1, 2) enhanced spectrum with Nyquist.
            new_cache: updated GRU hidden states.
        """
        if spec_frame.dim() != 4 or spec_frame.shape[2] != 1:
            raise ValueError(f"spec_frame must be (B, F, 1, 2), got {spec_frame.shape}")

        B, FREQ, _, _ = spec_frame.shape
        c = self.base.input_compression
        eps = self.base._eps

        # -- 1. Strip Nyquist (same as base model _forward_spectrum) --
        spec_frame = spec_frame[:, :-1, :, :]  # (B, FREQ-1, 1, 2)

        # -- 2. Magnitude compression --
        mag = torch.linalg.norm(spec_frame, dim=-1, keepdim=True).clamp(min=eps)
        spec_c = spec_frame * mag.pow(c - 1.0)

        # -- 3. Reshape for encoder: (B, 2, FREQ-1) --
        x = spec_c.permute(0, 2, 3, 1).reshape(B, 2, FREQ - 1)

        # -- 4. Encoder --
        x = self.base.enc_pre(x)
        enc_outs = [x]
        for enc in self.base.encoder:
            x = enc(x)
            enc_outs.append(x)

        # -- 5. RNNFormer pre --
        x = self.base.rf_pre(x)
        _, C, _F = x.shape
        x = x.view(1, B, _F, C)  # (TIME=1, BATCH, FREQ, CH)

        # -- 6. RNNFormer blocks with cache --
        new_cache: dict[str, Tensor] = {}
        for i, block in enumerate(self.base.rf_block):
            h = cache[f"cache_{i}"]
            x, h_new = block(x, h)
            new_cache[f"cache_{i}"] = h_new

        # -- 7. RNNFormer post --
        x = x.view(B, C, _F)
        x = self.base.rf_post(x)

        # -- 8. Decoder with skip connections --
        for dec in self.base.decoder:
            x = torch.cat([x, enc_outs.pop()], dim=1)
            x = dec(x)
        x = torch.cat([x, enc_outs.pop()], dim=1)
        x = self.base.dec_post(x)

        # -- 9. Reshape mask and apply to compressed spectrum (real-only ops for ONNX) --
        mask = x.reshape(B, 1, 2, FREQ - 1).permute(0, 3, 1, 2)  # (B, FREQ-1, 1, 2)

        # Complex multiplication: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
        real = spec_c[..., 0] * mask[..., 0] - spec_c[..., 1] * mask[..., 1]
        imag = spec_c[..., 0] * mask[..., 1] + spec_c[..., 1] * mask[..., 0]
        out_c = torch.stack([real, imag], dim=-1)  # (B, FREQ-1, 1, 2)

        # Power decompression
        mag = torch.linalg.norm(out_c, dim=-1, keepdim=True).clamp(min=eps)
        out_c = out_c * mag.pow(1.0 / c - 1.0)

        # -- 10. Pad Nyquist back --
        out = F.pad(out_c, (0, 0, 0, 0, 0, 1))  # (B, FREQ, 1, 2)
        return out, new_cache


class StreamingFastEnhancerFlat(nn.Module):
    """Flat wrapper: expands cache dict -> separate ONNX inputs/outputs.

    ONNX I/O naming matches the official convention:
      inputs:  ["spec_in", "cache_in_0", "cache_in_1", ...]
      outputs: ["spec_out", "cache_out_0", "cache_out_1", ...]
    """

    def __init__(self, streaming_model: StreamingFastEnhancer):
        super().__init__()
        self.inner = streaming_model
        self.n_fft = streaming_model.n_fft
        self.hop_length = streaming_model.hop_length
        self.win_length = streaming_model.win_length
        self.target_fs = streaming_model.target_fs
        self.window = streaming_model.window
        self.center = streaming_model.center
        self._keys = streaming_model.CACHE_KEYS

    def forward(self, spec_in: Tensor, *cache_tensors: Tensor) -> tuple[Tensor, ...]:
        cache = {k: v for k, v in zip(self._keys, cache_tensors)}
        enh, new_cache = self.inner(spec_in, cache)
        return (enh,) + tuple(new_cache[k] for k in self._keys)

    def get_input_names(self) -> list[str]:
        return ["spec_in"] + self._keys

    def get_output_names(self) -> list[str]:
        return ["spec_out"] + [f"out_{k}" for k in self._keys]

    def get_dynamic_axes(self) -> dict[str, dict[int, str]]:
        # spec 的 batch 位于 axis 0；GRU cache 展平为
        # (1, freq * batch, channels)，其 batch 位于 axis 1。
        axes: dict[str, dict[int, str]] = {
            "spec_in": {0: "batch_size"},
            "spec_out": {0: "batch_size"},
        }
        for name in self._keys:
            axes[name] = {1: "batch_flattened"}
            axes[f"out_{name}"] = {1: "batch_flattened"}
        return axes

    def init_cache_flat(self, batch_size: int, device: torch.device | str, dtype: torch.dtype) -> list[Tensor]:
        cache = self.inner.init_cache(batch_size, torch.device(device), dtype)
        return [cache[k] for k in self._keys]

    @staticmethod
    def load_cache_cfg_from_onnx(onnx_path: str) -> dict[str, list[int]]:
        """Read cache shapes from ONNX metadata."""
        import json
        from onnx.utils import read_onnx_metadata

        metadata = read_onnx_metadata(onnx_path)
        if "fastenhancer_cache_cfg" in metadata:
            return {
                key: list(value)
                for key, value in json.loads(metadata["fastenhancer_cache_cfg"]).items()
            }
        raise ValueError(f"ONNX {onnx_path} missing 'fastenhancer_cache_cfg' metadata.")
