"""Streaming Ultra-DPC-AEC wrapper for ONNX export and frame-by-frame inference.

Reference: Chen et al., "Ultra Dual-Path Compression For Joint Echo Cancellation And
           Noise Suppression", ICASSP 2023.

Frame-by-frame causal streaming with explicit caches for:
  - Sub-band linear attention KV states (per block)
  - GRU hidden state (first block only)

Input : mic_spec, ref_spec, err_spec  (B, F, 1, 2)  one STFT frame
Output: enh_spec                        (B, F, 1, 2)
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from tasks.aec.models.discriminative.ultra_dpc_aec import UltraDPCAEC, DualPathTransformerBlock, LinearAttention


# ---------------------------------------------------------------------------
# StreamLinearAttention
# ---------------------------------------------------------------------------
# Causal linear attention via recurrent KV state.
#   kv_state_t = kv_state_{t-1} + phi(k_t)^T * v_t
#   k_state_t  = k_state_{t-1} + phi(k_t)
#   o_t = phi(q_t) @ kv_state_t / (phi(q_t) @ k_state_t + eps)
#
# For sub-band path each freq bin is an independent sequence.
# ---------------------------------------------------------------------------

class StreamLinearAttention(nn.Module):
    """Streaming wrapper for causal LinearAttention."""

    def __init__(self, attn: LinearAttention):
        super().__init__()
        self.attn = attn
        self.dim = attn.dim
        self.n_heads = attn.n_heads
        self.head_dim = attn.head_dim
        self.eps = attn.eps

    def init_cache(
        self, batch_size: int, n_seqs: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[Tensor, Tensor]:
        """Return zero-initialized KV state and K state.

        Args:
            n_seqs: number of independent sequences (e.g., B*F for sub-band).
        """
        kv_state = torch.zeros(
            batch_size * n_seqs, self.n_heads, self.head_dim, self.head_dim,
            device=device, dtype=dtype,
        )
        k_state = torch.zeros(
            batch_size * n_seqs, self.n_heads, self.head_dim,
            device=device, dtype=dtype,
        )
        return kv_state, k_state

    def forward(
        self, x: Tensor, kv_state: Tensor, k_state: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Process one step.

        Args:
            x: (n_seqs, 1, D)
            kv_state: (n_seqs, n_heads, head_dim, head_dim)
            k_state:  (n_seqs, n_heads, head_dim)

        Returns:
            out: (n_seqs, 1, D)
            new_kv_state, new_k_state
        """
        n_seqs, L, D = x.shape
        assert L == 1, "Streaming LinearAttention only supports single step"

        qkv = self.attn.qkv(x).reshape(n_seqs, L, 3, self.n_heads, self.head_dim)
        q = qkv[:, 0, 0]  # (n_seqs, n_heads, head_dim)
        k = qkv[:, 0, 1]
        v = qkv[:, 0, 2]

        q = self.attn._feature_map(q)
        k = self.attn._feature_map(k)

        # Update recurrent states
        kv = torch.einsum("nhd,nhe->nhde", k, v)          # (n_seqs, n_heads, d, d)
        kv_state = kv_state + kv
        k_state = k_state + k

        # Compute output
        num = torch.einsum("nhd,nhde->nhe", q, kv_state)  # (n_seqs, n_heads, d)
        den = torch.einsum("nhd,nhd->nh", q, k_state).clamp_min(self.eps)
        out = num / den.unsqueeze(-1)                     # (n_seqs, n_heads, d)
        out = out.reshape(n_seqs, D)
        out = self.attn.out(out).unsqueeze(1)             # (n_seqs, 1, D)
        return out, kv_state, k_state


# ---------------------------------------------------------------------------
# StreamDualPathBlock
# ---------------------------------------------------------------------------

class StreamDualPathBlock(nn.Module):
    """Streaming dual-path block.

    Sub-band attention is causal (with KV cache).
    Full-band attention is bidirectional but per-frame independent (no cache).
    GRU hidden state cached for the first block only.
    """

    def __init__(self, block: DualPathTransformerBlock):
        super().__init__()
        self.block = block
        self.use_gru = block.use_gru
        self.dim = block.dim
        gru = getattr(block, "subband_gru", None)
        self.gru_dim = gru.hidden_size if gru is not None else None

        self.stream_attn = StreamLinearAttention(block.subband_attn)

    def init_cache(
        self, batch_size: int, freq_dim: int, device: torch.device, dtype: torch.dtype
    ) -> Dict[str, Tensor]:
        cache: Dict[str, Tensor] = {}
        if self.stream_attn is not None:
            kv_state, k_state = self.stream_attn.init_cache(batch_size, freq_dim, device, dtype)
            cache["kv_state"] = kv_state
            cache["k_state"] = k_state
        if self.use_gru:
            cache["gru_h"] = torch.zeros(
                1, batch_size * freq_dim, self.gru_dim, device=device, dtype=dtype
            )
        return cache

    def forward(
        self, x: Tensor, cache: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Args:
            x: (B, T=1, F, E)
            cache: dict with "kv_state", "k_state", optionally "gru_h"

        Returns:
            x: (B, 1, F, E)
            new_cache
        """
        B, T, F, E = x.shape
        assert T == 1
        new_cache: Dict[str, Tensor] = {}

        # --- Sub-band path (temporal, causal) ---
        x_sb = x.permute(0, 2, 1, 3).reshape(B * F, T, E)  # (B*F, 1, E)
        x_sb = self.block.subband_norm1(x_sb)

        attn_out, new_kv, new_k = self.stream_attn(
            x_sb, cache["kv_state"], cache["k_state"]
        )
        new_cache["kv_state"] = new_kv
        new_cache["k_state"] = new_k
        x_sb = x_sb + attn_out

        if self.use_gru:
            gru_out, gru_h_new = self.block.subband_gru(x_sb, cache["gru_h"])
            x_sb = x_sb + self.block.subband_proj(gru_out)
            new_cache["gru_h"] = gru_h_new

        x_sb = x_sb.reshape(B, F, T, E).permute(0, 2, 1, 3)  # (B, 1, F, E)
        x = x + x_sb

        # --- Full-band path (spectral, bidirectional, per-frame independent) ---
        x_fb = x.reshape(B * T, F, E)  # (B, F, E)
        x_fb = self.block.fullband_norm(x_fb)
        x_fb = x_fb + self.block.fullband_attn(x_fb)
        x_fb = x_fb.reshape(B, T, F, E)
        x = x + x_fb

        return x, new_cache


# ---------------------------------------------------------------------------
# StreamingUltraDPCAEC
# ---------------------------------------------------------------------------

class StreamingUltraDPCAEC(nn.Module):
    """Frame-by-frame streaming Ultra-DPC-AEC.

    Processes one STFT frame at a time:
      inputs:  mic_spec, ref_spec, err_spec  (B, F, 1, 2)
      output:  enh_spec                      (B, F, 1, 2)

    未压缩路径逐帧推进 causal DPT 状态；时间压缩路径维护历史帧、稀疏
    token 相位与当前解压特征，只在分组边界推进 DPT 状态，组内复用特征。
    PostNet 的单向 GRU hidden 每帧推进，因此逐帧结果与同相位整段路径一致。
    """

    def __init__(self, base_model: UltraDPCAEC):
        super().__init__()
        self.base = base_model
        self.compressed = base_model.freq_comp is not None or base_model.time_comp is not None

        self.n_fft = base_model.n_fft
        self.hop_length = base_model.hop_length
        self.embed_dim = base_model.embed_dim
        self.n_signals = base_model.n_signals
        self.in_ch = base_model.n_signals * 2

        self.stream_dp_blocks = nn.ModuleList([
            StreamDualPathBlock(b) for b in base_model.dp_blocks
        ])

    def init_cache(
        self, batch_size: int = 1, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32
    ) -> Dict[str, Tensor | Dict[str, Tensor]]:
        device = torch.device(device)
        cache: Dict[str, Tensor | Dict[str, Tensor]] = {}
        n_freq = self.n_fft // 2 + 1
        if self.base.freq_comp is not None:
            n_freq = self.base.freq_comp.n_bands
        for i, block in enumerate(self.stream_dp_blocks):
            cache[f"block_{i}"] = block.init_cache(batch_size, n_freq, device, dtype)
        if self.base.time_comp is not None:
            freq_bins = self.n_fft // 2 + 1
            cache["time_history"] = torch.zeros(
                batch_size, self.in_ch, self.base.time_skip - 1, freq_bins,
                device=device, dtype=dtype,
            )
            cache["time_feature"] = torch.zeros(
                batch_size, self.in_ch, 1, freq_bins, device=device, dtype=dtype,
            )
            cache["time_phase"] = torch.zeros(1, device=device, dtype=torch.long)
            if self.base.postnet is not None:
                cache["postnet_h"] = torch.zeros(
                    1, batch_size, self.base.postnet.n_bands, device=device, dtype=dtype,
                )
        return cache

    def forward(
        self,
        mic_spec: Tensor,
        ref_spec: Tensor,
        err_spec: Tensor,
        cache: Dict[str, Tensor | Dict[str, Tensor]],
    ) -> Tuple[Tensor, Dict[str, Tensor | Dict[str, Tensor]]]:
        """处理一个 STFT 帧并返回完整下一状态。"""
        batch, freq_bins, frames, _ = mic_spec.shape
        if frames != 1 or mic_spec.shape != ref_spec.shape or mic_spec.shape != err_spec.shape:
            raise ValueError("StreamingUltraDPCAEC expects matching (B,F,1,2) spectra")

        raw = torch.stack([mic_spec, ref_spec, err_spec], dim=1)
        raw = raw.permute(0, 1, 4, 3, 2).reshape(batch, self.in_ch, 1, freq_bins)
        new_cache: Dict[str, Tensor | Dict[str, Tensor]] = {}
        boundary = None
        if self.base.time_comp is not None:
            history = cache["time_history"]
            grouped = torch.cat([history, raw], dim=2)
            x = grouped.reshape(batch, self.in_ch * self.base.time_skip, 1, freq_bins)
            x = self.base.time_comp.proj(x)
            boundary = cache["time_phase"] == 0
            new_cache["time_history"] = grouped[:, :, 1:, :]
            new_cache["time_phase"] = (cache["time_phase"] + 1) % self.base.time_skip
        else:
            x = raw

        if self.base.freq_comp is not None:
            x = self.base.freq_comp(x)
        x = self.base.encoder(x.permute(0, 1, 3, 2).contiguous())
        x = x.permute(0, 3, 2, 1).contiguous()

        for i, block in enumerate(self.stream_dp_blocks):
            old_block = cache[f"block_{i}"]
            x, candidate = block(x, old_block)
            if boundary is None:
                new_cache[f"block_{i}"] = candidate
            else:
                new_cache[f"block_{i}"] = {
                    key: torch.where(boundary, value, old_block[key])
                    for key, value in candidate.items()
                }

        x = self.base.decoder(x.permute(0, 3, 2, 1).contiguous())
        x = x.permute(0, 1, 3, 2).contiguous()
        if self.base.freq_decomp is not None:
            x = self.base.freq_decomp(x)
        if self.base.time_decomp is not None:
            x = self.base.time_decomp.proj(x)
            x = torch.where(boundary, x, cache["time_feature"])
            new_cache["time_feature"] = x

        enhanced = self.base._apply_complex_masks(x, mic_spec, ref_spec, err_spec)
        postnet_h = cache.get("postnet_h")
        enhanced, postnet_h = self.base._apply_postnet(
            enhanced, mic_spec, err_spec, postnet_h
        )
        if postnet_h is not None:
            new_cache["postnet_h"] = postnet_h
        return enhanced, new_cache


# ---------------------------------------------------------------------------
# Flat wrapper for ONNX export
# ---------------------------------------------------------------------------

class StreamingUltraDPCAECFlat(nn.Module):
    """Flat cache wrapper for ONNX export.

    ONNX inputs : [mic_spec, ref_spec, err_spec, cache_0_kv, cache_0_k, ...]
    ONNX outputs: [enh_spec, out_cache_0_kv, out_cache_0_k, ...]
    """

    def __init__(self, streaming_model: StreamingUltraDPCAEC):
        super().__init__()
        self.inner = streaming_model
        self.n_blocks = len(streaming_model.stream_dp_blocks)

        # Build cache key list
        self._cache_keys: List[str] = []
        for i in range(self.n_blocks):
            self._cache_keys.append(f"block_{i}_kv")
            self._cache_keys.append(f"block_{i}_k")
            if streaming_model.stream_dp_blocks[i].use_gru:
                self._cache_keys.append(f"block_{i}_gru_h")
        self._extra_cache_keys: List[str] = []
        if streaming_model.base.time_comp is not None:
            self._extra_cache_keys.extend(["time_history", "time_feature", "time_phase"])
            if streaming_model.base.postnet is not None:
                self._extra_cache_keys.append("postnet_h")
            self._cache_keys.extend(self._extra_cache_keys)

    def _cache_to_flat(
        self, cache: Dict[str, Tensor | Dict[str, Tensor]]
    ) -> List[Tensor]:
        flat: List[Tensor] = []
        for i in range(self.n_blocks):
            c = cache[f"block_{i}"]
            flat.append(c["kv_state"])
            flat.append(c["k_state"])
            if self.inner.stream_dp_blocks[i].use_gru:
                flat.append(c["gru_h"])
        flat.extend(cache[key] for key in self._extra_cache_keys)
        return flat

    def _flat_to_cache(
        self, flat: List[Tensor]
    ) -> Dict[str, Tensor | Dict[str, Tensor]]:
        cache: Dict[str, Tensor | Dict[str, Tensor]] = {}
        idx = 0
        for i in range(self.n_blocks):
            c: Dict[str, Tensor] = {}
            c["kv_state"] = flat[idx]; idx += 1
            c["k_state"] = flat[idx]; idx += 1
            if self.inner.stream_dp_blocks[i].use_gru:
                c["gru_h"] = flat[idx]; idx += 1
            cache[f"block_{i}"] = c
        for key in self._extra_cache_keys:
            cache[key] = flat[idx]
            idx += 1
        return cache

    def forward(
        self, mic_spec: Tensor, ref_spec: Tensor, err_spec: Tensor, *flat_cache: Tensor
    ) -> Tuple[Tensor, ...]:
        cache = self._flat_to_cache(list(flat_cache))
        enh, new_cache = self.inner(mic_spec, ref_spec, err_spec, cache)
        new_flat = self._cache_to_flat(new_cache)
        return (enh,) + tuple(new_flat)

    def init_cache_flat(
        self, batch_size: int = 1, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32
    ) -> List[Tensor]:
        cache = self.inner.init_cache(batch_size, device, dtype)
        return self._cache_to_flat(cache)

    def get_input_names(self) -> List[str]:
        names = ["mic_spec", "ref_spec", "err_spec"]
        names += self._cache_keys
        return names

    def get_output_names(self) -> List[str]:
        names = ["enh_spec"]
        names += [f"out_{k}" for k in self._cache_keys]
        return names

    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        axes: Dict[str, Dict[int, str]] = {
            name: {0: "batch_size"}
            for name in self.get_input_names()[:3] + self.get_output_names()[:1]
        }
        for name in self._cache_keys:
            if name == "time_phase":
                continue
            # DPT KV/K 把 B*F 展在 axis 0；GRU hidden 的 batch 在 axis 1。
            axis = 1 if name.endswith("_gru_h") or name == "postnet_h" else 0
            label = "batch_flattened" if name.startswith("block_") else "batch_size"
            axes[name] = {axis: label}
            axes[f"out_{name}"] = {axis: label}
        return axes

    @staticmethod
    def load_cache_cfg_from_onnx(onnx_path: str) -> Dict[str, List[int]]:
        """Read cache shapes from ONNX metadata."""
        import json
        from onnx.utils import read_onnx_metadata

        metadata = read_onnx_metadata(onnx_path)
        if "ultra_dpc_cache_cfg" in metadata:
            return {
                key: list(value)
                for key, value in json.loads(metadata["ultra_dpc_cache_cfg"]).items()
            }
        raise ValueError(f"ONNX {onnx_path} missing 'ultra_dpc_cache_cfg' metadata.")
