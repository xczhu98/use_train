"""StreamingDeepVQE: frame-by-frame streaming DeepVQE (PyTorch + ONNX-exportable)."""

from __future__ import annotations

import sys
from pathlib import Path
_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.aec.models.discriminative.deepvqe import (
    DeepVQE,
    _AlignBlock,
    _Bottleneck,
    _CCM,
    _DecBlock,
    _EncBlock,
    _ResBlock,
    _SubPixelConv2d,
)


class _StreamConvCache(nn.Module):
    def __init__(self, conv: nn.Conv2d, time_history: int, freq_pad: int):
        super().__init__()
        self.conv = conv
        self.time_history = time_history
        self.freq_pad = freq_pad

    def forward(self, x: torch.Tensor, cache: torch.Tensor):
        inp = torch.cat([cache, x], dim=2)
        if self.freq_pad > 0:
            inp = F.pad(inp, (self.freq_pad, self.freq_pad, 0, 0))
        out = self.conv(inp)
        new_cache = torch.cat([cache[:, :, 1:, :], x], dim=2)
        return out, new_cache


class _StreamResBlock(nn.Module):
    def __init__(self, res: _ResBlock):
        super().__init__()
        self.conv = _StreamConvCache(res.net[1], time_history=3, freq_pad=1)
        self.bn = res.net[2]
        self.act = res.net[3]

    def forward(self, x: torch.Tensor, cache: torch.Tensor):
        out, new_cache = self.conv(x, cache)
        out = self.act(self.bn(out))
        return out + x, new_cache


class _StreamEncBlock(nn.Module):
    def __init__(self, block: _EncBlock):
        super().__init__()
        self.conv = _StreamConvCache(block.conv, time_history=3, freq_pad=1)
        self.bn = block.bn
        self.act = block.act
        self.res = _StreamResBlock(block.res) if block.res is not None else None

    def forward(self, x: torch.Tensor, conv_cache: torch.Tensor, res_cache):
        out, new_conv_cache = self.conv(x, conv_cache)
        out = self.act(self.bn(out))
        if self.res is not None:
            out, new_res_cache = self.res(out, res_cache)
        else:
            new_res_cache = None
        return out, new_conv_cache, new_res_cache


class _StreamAlignBlock(nn.Module):
    def __init__(self, align: _AlignBlock):
        super().__init__()
        self.pconv_mic = align.pconv_mic
        self.pconv_ref = align.pconv_ref
        self.delay = align.unfold[1].kernel_size[0]
        self.far_ch = align.far_ch
        self.attn_conv = _StreamConvCache(align.conv[1], time_history=4, freq_pad=1)

    def forward(self, x_mic, x_ref, far_cache, v_cache, k_cache):
        B = x_mic.shape[0]
        F_dim = x_mic.shape[3]
        Q = self.pconv_mic(x_mic)
        # K window: use pre-computed pconv_ref history (k_cache) to match offline
        # zero-padding on K (not on x_ref).
        K_current = self.pconv_ref(x_ref)
        K_window = torch.cat([k_cache, K_current], dim=2)
        Ku = K_window.view(B, K_window.shape[1], self.delay, 1, F_dim)
        Ku = Ku.permute(0, 1, 3, 2, 4).contiguous()
        V = torch.sum(Q.unsqueeze(-2) * Ku, dim=-1)
        V, new_v_cache = self.attn_conv(V, v_cache)
        A = torch.softmax(V, dim=-1).unsqueeze(-1)
        # y window: use raw far history (far_cache) for weighted sum
        y_window = torch.cat([far_cache, x_ref], dim=2)
        y_window = y_window.view(B, self.far_ch, self.delay, 1, F_dim)
        y_window = y_window.permute(0, 1, 3, 2, 4).contiguous()
        out = torch.sum(y_window * A, dim=-2)
        new_far_cache = torch.cat([far_cache[:, :, 1:, :], x_ref], dim=2)
        new_k_cache = torch.cat([k_cache[:, :, 1:, :], K_current], dim=2)
        return out, new_far_cache, new_v_cache, new_k_cache


class _StreamSubPixelConv2d(nn.Module):
    def __init__(self, deconv: _SubPixelConv2d):
        super().__init__()
        self.conv = _StreamConvCache(deconv.conv, time_history=3, freq_pad=1)

    def forward(self, x: torch.Tensor, cache: torch.Tensor):
        y, new_cache = self.conv(x, cache)
        B, C, T, F = y.shape
        return (
            y.view(B, C // 2, 2, T, F)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(B, C // 2, T, F * 2),
            new_cache,
        )


class _StreamDecBlock(nn.Module):
    def __init__(self, block: _DecBlock):
        super().__init__()
        self.skip_conv = block.skip_conv
        self.deconv = _StreamSubPixelConv2d(block.deconv)
        self.bn = block.bn
        self.act = block.act
        self.is_last = block.is_last
        self.res = _StreamResBlock(block.res) if block.res is not None else None

    def forward(self, x, x_enc, deconv_cache, res_cache):
        y = x + self.skip_conv(x_enc)
        if self.res is not None:
            y, new_res_cache = self.res(y, res_cache)
        else:
            new_res_cache = None
        y, new_deconv_cache = self.deconv(y, deconv_cache)
        if not self.is_last:
            y = self.act(self.bn(y))
        return y, new_deconv_cache, new_res_cache


class _StreamBottleneck(nn.Module):
    def __init__(self, bottle: _Bottleneck):
        super().__init__()
        self.gru = bottle.gru
        self.fc = bottle.fc

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        B, C, T, F = x.shape
        assert T == 1
        y = x.permute(0, 2, 1, 3).reshape(B, T, C * F)
        # h comes as (B, 1, hidden) from init_cache; GRU expects (1, B, hidden)
        h = h.transpose(0, 1).contiguous()
        y, h_new = self.gru(y, h)
        h_new = h_new.transpose(0, 1).contiguous()
        y = self.fc(y)
        return y.reshape(B, T, C, F).permute(0, 2, 1, 3), h_new


class _StreamCCM(nn.Module):
    def __init__(self, ccm: _CCM):
        super().__init__()
        self.register_buffer("v", ccm.v)

    def forward(self, m, x, mic_cache):
        B = m.shape[0]
        T = m.shape[2]
        F_dim = m.shape[3]
        m = m.view(B, 3, 9, T, F_dim)
        H_r = torch.sum(self.v[0][None, :, None, None, None] * m, dim=1)
        H_i = torch.sum(self.v[1][None, :, None, None, None] * m, dim=1)
        M_r = H_r.view(B, 3, 3, T, F_dim)
        M_i = H_i.view(B, 3, 3, T, F_dim)
        x_in = x.permute(0, 3, 2, 1).contiguous()
        x_window = torch.cat([mic_cache, x_in], dim=2)
        x_window = F.pad(x_window, (1, 1, 0, 0))
        xu = F.unfold(x_window, kernel_size=(3, 3))
        xu = xu.view(B, 2, 3, 3, T, F_dim)
        enh_r = torch.sum(M_r * xu[:, 0] - M_i * xu[:, 1], dim=(1, 2))
        enh_i = torch.sum(M_r * xu[:, 1] + M_i * xu[:, 0], dim=(1, 2))
        out = torch.stack([enh_r, enh_i], dim=3)
        out = out.transpose(1, 2).contiguous()
        new_mic_cache = torch.cat([mic_cache[:, :, 1:, :], x_in], dim=2)
        return out, new_mic_cache


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _freq_dims(n_fft: int, n_enc: int) -> List[int]:
    f = [n_fft // 2 + 1]
    for _ in range(n_enc):
        f.append((f[-1] - 1) // 2 + 1)
    return f


def _make_cache_config(model: DeepVQE) -> Dict[str, Tuple[int, ...]]:
    """Return {cache_key: (C, time_history_or_delay, F)} for each cache tensor.

    Only includes residual cache entries for blocks that actually have a residual
    connection — blocks without ``res`` return ``None`` from their streaming wrapper,
    which would cause ONNX output count mismatch.
    """
    n_fft = model.n_fft
    variant = model._variant
    delay = model.align.unfold[1].kernel_size[0]
    n_enc = 5 if variant == "base" else 4
    freqs = _freq_dims(n_fft, n_enc)

    cfg: Dict[str, Tuple[int, ...]] = {}

    # Helper: add conv cache; add res cache only if block.res exists
    def _add_enc(name: str, in_ch: int, out_ch: int, f_in: int, f_out: int, block):
        cfg[f"{name}_conv"] = (in_ch, 3, f_in)
        if block.res is not None:
            cfg[f"{name}_res"] = (out_ch, 3, f_out)

    def _add_dec(name: str, in_ch: int, f_in: int, block):
        if block.res is not None:
            cfg[f"{name}_res"] = (in_ch, 3, f_in)
        cfg[f"{name}_deconv"] = (in_ch, 3, f_in)

    if variant == "base":
        _add_enc("enc1", 2, 64, freqs[0], freqs[1], model.enc1)
        _add_enc("enc2", 64, 128, freqs[1], freqs[2], model.enc2)
        _add_enc("enc3", 256, 128, freqs[2], freqs[3], model.enc3)
        _add_enc("enc4", 128, 128, freqs[3], freqs[4], model.enc4)
        _add_enc("enc5", 128, 128, freqs[4], freqs[5], model.enc5)
        _add_dec("dec5", 128, freqs[5], model.dec5)
        _add_dec("dec4", 128, freqs[4], model.dec4)
        _add_dec("dec3", 128, freqs[3], model.dec3)
        _add_dec("dec2", 128, freqs[2], model.dec2)
        _add_dec("dec1", 64, freqs[1], model.dec1)
    else:
        _add_enc("enc1", 2, 16, freqs[0], freqs[1], model.enc1)
        _add_enc("enc2", 16, 40, freqs[1], freqs[2], model.enc2)
        _add_enc("enc3", 64, 56, freqs[2], freqs[3], model.enc3)
        _add_enc("enc4", 56, 24, freqs[3], freqs[4], model.enc4)
        _add_dec("dec4", 24, freqs[4], model.dec4)
        _add_dec("dec3", 40, freqs[3], model.dec3)
        _add_dec("dec2", 32, freqs[2], model.dec2)
        _add_dec("dec1", 32, freqs[1], model.dec1)

    # Far encoder
    if variant == "base":
        _add_enc("enc_far1", 2, 32, freqs[0], freqs[1], model.enc_far1)
        _add_enc("enc_far2", 32, 128, freqs[1], freqs[2], model.enc_far2)
        far_ch = 128
    else:
        _add_enc("enc_far1", 2, 8, freqs[0], freqs[1], model.enc_far1)
        _add_enc("enc_far2", 8, 24, freqs[1], freqs[2], model.enc_far2)
        far_ch = 24

    # Align
    cfg["far_cache"] = (far_ch, delay - 1, freqs[2])
    cfg["align_v"] = (far_ch, 4, delay)
    cfg["align_k"] = (far_ch, delay - 1, freqs[2])

    # Bottleneck
    bottle_freq = freqs[-1]
    bottle_hid = 64 * bottle_freq if variant == "base" else 12 * bottle_freq
    cfg["gru_h"] = (1, bottle_hid)

    # CCM
    cfg["ccm_mic"] = (2, 2, freqs[0])

    return cfg


# ---------------------------------------------------------------------------
# StreamingDeepVQE
# ---------------------------------------------------------------------------


class StreamingDeepVQE(nn.Module):
    """Frame-by-frame streaming DeepVQE.

    Input per step:
      - mic_spec: ``(B, F, 1, 2)``
      - far_spec: ``(B, F, 1, 2)``
      - cache:    dict from ``init_cache()``

    Output per step:
      - enh_spec: ``(B, F, 1, 2)``
      - cache:    updated dict
    """

    def __init__(
        self,
        base_model: DeepVQE,
        *,
        win_length: int | None = None,
        target_fs: int | None = None,
        window: str = "hann",
        center: bool | None = None,
        deployment_config: dict | None = None,
    ):
        super().__init__()
        self.n_fft = int(base_model.n_fft)
        self.hop_length = int(base_model.hop_length)
        self.win_length = int(win_length if win_length is not None else self.n_fft)
        self.target_fs = None if target_fs is None else int(target_fs)
        self.window = str(window)
        self.center = center
        self.variant = base_model._variant
        self.compression = float(base_model.fe.c)
        self.deployment_config = deployment_config
        self.fe = base_model.fe

        # Mic encoder
        self.enc1 = _StreamEncBlock(base_model.enc1)
        self.enc2 = _StreamEncBlock(base_model.enc2)
        self.enc3 = _StreamEncBlock(base_model.enc3)
        self.enc4 = _StreamEncBlock(base_model.enc4)
        self.enc5 = _StreamEncBlock(base_model.enc5) if base_model.enc5 is not None else None

        # Far encoder
        self.enc_far1 = _StreamEncBlock(base_model.enc_far1)
        self.enc_far2 = _StreamEncBlock(base_model.enc_far2)

        # Align
        self.align = _StreamAlignBlock(base_model.align)

        # Bottleneck
        self.bottle = _StreamBottleneck(base_model.bottle)

        # Decoder
        self.dec5 = _StreamDecBlock(base_model.dec5) if base_model.dec5 is not None else None
        self.dec4 = _StreamDecBlock(base_model.dec4)
        self.dec3 = _StreamDecBlock(base_model.dec3)
        self.dec2 = _StreamDecBlock(base_model.dec2)
        self.dec1 = _StreamDecBlock(base_model.dec1)

        # CCM
        self.ccm = _StreamCCM(base_model.ccm)

        # Cache config。alignment window 仅回看 far 历史，不造成主输出 lookahead。
        self._cache_cfg = _make_cache_config(base_model)
        self.CACHE_KEYS = list(self._cache_cfg.keys())
        self.alignment_window_frames = int(base_model.align.unfold[1].kernel_size[0])
        self.alignment_history_frames = self.alignment_window_frames - 1
        self.algorithmic_delay_frames = 0
        self.algorithmic_delay = 0

    def init_cache(self, batch: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        cache: Dict[str, torch.Tensor] = {}
        for k, (C, *rest) in self._cache_cfg.items():
            shape = (batch, C, *rest)
            cache[k] = torch.zeros(shape, device=device, dtype=dtype)
        return cache

    def forward(
        self,
        mic_spec: torch.Tensor,
        far_spec: torch.Tensor,
        cache: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if mic_spec.dim() != 4 or mic_spec.shape[2] != 1:
            raise ValueError(f"mic_spec must be (B, F, 1, 2), got {mic_spec.shape}")
        if far_spec.dim() != 4 or far_spec.shape[2] != 1:
            raise ValueError(f"far_spec must be (B, F, 1, 2), got {far_spec.shape}")

        # Feature extraction
        e0_mic = self.fe(mic_spec)   # (B, 2, 1, F)
        e0_far = self.fe(far_spec)   # (B, 2, 1, F)

        # Mic encoder
        e1, c1, r1 = self.enc1(e0_mic, cache["enc1_conv"], cache.get("enc1_res"))
        e2, c2, r2 = self.enc2(e1, cache["enc2_conv"], cache.get("enc2_res"))

        # Far encoder
        ef1, cf1, rf1 = self.enc_far1(e0_far, cache["enc_far1_conv"], cache.get("enc_far1_res"))
        ef2, cf2, rf2 = self.enc_far2(ef1, cache["enc_far2_conv"], cache.get("enc_far2_res"))

        # Align
        aligned, c_far, c_v, c_k = self.align(e2, ef2, cache["far_cache"], cache["align_v"], cache["align_k"])
        merged = torch.cat([e2, aligned], dim=1)

        # Shared encoder
        e3, c3, r3 = self.enc3(merged, cache["enc3_conv"], cache.get("enc3_res"))
        e4, c4, r4 = self.enc4(e3, cache["enc4_conv"], cache.get("enc4_res"))

        # Bottleneck + decoder
        if self.enc5 is not None:
            e5, c5, r5 = self.enc5(e4, cache["enc5_conv"], cache.get("enc5_res"))
            bn, h_new = self.bottle(e5, cache["gru_h"])
            d5, cd5, cr5 = self.dec5(bn, e5, cache["dec5_deconv"], cache.get("dec5_res"))
            d5 = d5[..., : e4.shape[-1]]
            d4, cd4, cr4 = self.dec4(d5, e4, cache["dec4_deconv"], cache.get("dec4_res"))
        else:
            bn, h_new = self.bottle(e4, cache["gru_h"])
            d4, cd4, cr4 = self.dec4(bn, e4, cache["dec4_deconv"], cache.get("dec4_res"))

        d4 = d4[..., : e3.shape[-1]]
        d3, cd3, cr3 = self.dec3(d4, e3, cache["dec3_deconv"], cache.get("dec3_res"))
        d3 = d3[..., : e2.shape[-1]]
        d2, cd2, cr2 = self.dec2(d3, e2, cache["dec2_deconv"], cache.get("dec2_res"))
        d2 = d2[..., : e1.shape[-1]]
        d1, cd1, cr1 = self.dec1(d2, e1, cache["dec1_deconv"], cache.get("dec1_res"))
        d1 = d1[..., : e0_mic.shape[-1]]

        # CCM
        enh, ccm_cache = self.ccm(d1, mic_spec, cache["ccm_mic"])

        # Assemble new cache — only include keys present in CACHE_KEYS
        keys = set(self.CACHE_KEYS)
        new_cache: Dict[str, torch.Tensor] = {}
        new_cache["enc1_conv"] = c1
        if "enc1_res" in keys:
            new_cache["enc1_res"] = r1
        new_cache["enc2_conv"] = c2
        if "enc2_res" in keys:
            new_cache["enc2_res"] = r2
        new_cache["enc3_conv"] = c3
        if "enc3_res" in keys:
            new_cache["enc3_res"] = r3
        new_cache["enc4_conv"] = c4
        if "enc4_res" in keys:
            new_cache["enc4_res"] = r4
        if self.enc5 is not None:
            new_cache["enc5_conv"] = c5
            if "enc5_res" in keys:
                new_cache["enc5_res"] = r5
        new_cache["enc_far1_conv"] = cf1
        if "enc_far1_res" in keys:
            new_cache["enc_far1_res"] = rf1
        new_cache["enc_far2_conv"] = cf2
        if "enc_far2_res" in keys:
            new_cache["enc_far2_res"] = rf2
        new_cache["far_cache"] = c_far
        new_cache["align_v"] = c_v
        new_cache["align_k"] = c_k
        new_cache["gru_h"] = h_new
        if self.enc5 is not None:
            new_cache["dec5_deconv"] = cd5
            if "dec5_res" in keys:
                new_cache["dec5_res"] = cr5
        new_cache["dec4_deconv"] = cd4
        if "dec4_res" in keys:
            new_cache["dec4_res"] = cr4
        new_cache["dec3_deconv"] = cd3
        if "dec3_res" in keys:
            new_cache["dec3_res"] = cr3
        new_cache["dec2_deconv"] = cd2
        if "dec2_res" in keys:
            new_cache["dec2_res"] = cr2
        new_cache["dec1_deconv"] = cd1
        if "dec1_res" in keys:
            new_cache["dec1_res"] = cr1
        new_cache["ccm_mic"] = ccm_cache

        return enh, new_cache


# ---------------------------------------------------------------------------
# Flat wrapper for ONNX export
# ---------------------------------------------------------------------------


class StreamingDeepVQEFlat(nn.Module):
    """Flat wrapper: expands cache dict -> separate ONNX inputs/outputs."""

    def __init__(self, streaming_model: StreamingDeepVQE):
        super().__init__()
        self.inner = streaming_model
        for name in (
            "n_fft", "hop_length", "win_length", "target_fs", "window", "center",
            "variant", "compression", "alignment_window_frames",
            "alignment_history_frames", "algorithmic_delay_frames",
            "algorithmic_delay", "deployment_config",
        ):
            setattr(self, name, getattr(streaming_model, name))
        self._keys = list(streaming_model.CACHE_KEYS)

    def init_cache_flat(
        self, batch_size: int, device: torch.device | str, dtype: torch.dtype
    ) -> list[torch.Tensor]:
        """按 flat wrapper 的 cache key 顺序构造 ONNX dummy cache。"""
        cache = self.inner.init_cache(batch_size, torch.device(device), dtype)
        return [cache[key] for key in self._keys]

    def forward(self, mic_spec: torch.Tensor, far_spec: torch.Tensor, *cache_tensors: torch.Tensor):
        cache = {k: v for k, v in zip(self._keys, cache_tensors)}
        enh, new_cache = self.inner(mic_spec, far_spec, cache)
        return (enh,) + tuple(new_cache[k] for k in self._keys)

    def get_input_names(self) -> List[str]:
        return ["mic_spec", "far_spec"] + self._keys

    def get_output_names(self) -> List[str]:
        return ["enh_spec"] + [f"out_{k}" for k in self._keys]

    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        axes: Dict[str, Dict[int, str]] = {}
        for name in self.get_input_names() + self.get_output_names():
            axes[name] = {0: "batch_size"}
        return axes

    def get_cache_config(self) -> Dict[str, list[int]]:
        """返回 metadata 使用的权威 batch-one cache ABI。"""
        cache = self.inner.init_cache(1, torch.device("cpu"), torch.float32)
        return {key: list(cache[key].shape) for key in self._keys}

    @staticmethod
    def load_cache_cfg_from_onnx(onnx_path: str) -> Dict[str, Tuple[int, ...]]:
        """按版本化 runtime metadata 顺序读取 DeepVQE cache ABI。"""
        import json

        from onnx.runtime_spec import (
            DEEPVQE_RUNTIME_METADATA_KEY,
            DeepVQERuntimeSpec,
        )
        from onnx.utils import parse_onnx_cache_config, read_onnx_metadata

        metadata = read_onnx_metadata(onnx_path)
        raw_runtime = metadata.get(DEEPVQE_RUNTIME_METADATA_KEY)
        if raw_runtime is None:
            raise RuntimeError(
                f"ONNX metadata is missing required {DEEPVQE_RUNTIME_METADATA_KEY!r}; "
                "legacy DeepVQE artifacts must be re-exported"
            )
        try:
            runtime = DeepVQERuntimeSpec.from_metadata(json.loads(raw_runtime))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"ONNX metadata {DEEPVQE_RUNTIME_METADATA_KEY!r} is not valid JSON"
            ) from exc
        parsed = parse_onnx_cache_config(metadata, "deepvqe_cache_cfg")
        if set(parsed) != set(runtime.cache_names):
            raise RuntimeError(
                "DeepVQE cache metadata names do not match versioned cache_names: "
                f"cache_cfg={tuple(parsed)}, cache_names={runtime.cache_names}"
            )
        return {key: tuple(parsed[key]) for key in runtime.cache_names}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--dur", type=float, default=1.0)
    ap.add_argument("--variant", type=str, default="base")
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--align-delay", type=int, default=25)
    ap.add_argument("--device", type=str, default="cpu")
    ns = ap.parse_args()

    device = torch.device(ns.device)
    T = int(ns.dur * ns.sr / 256)  # hop=256
    n_freq = ns.n_fft // 2 + 1

    base = DeepVQE(variant=ns.variant, n_fft=ns.n_fft, align_delay=ns.align_delay).eval().to(device)
    stream = StreamingDeepVQE(base).eval().to(device)

    mic = torch.randn(1, n_freq, T, 2, device=device)
    far = torch.randn(1, n_freq, T, 2, device=device)

    # Offline forward
    with torch.inference_mode():
        off = base.forward_spectra(mic, far)

    # Streaming forward
    cache = stream.init_cache(batch=1, device=device, dtype=mic.dtype)
    outs = []
    with torch.inference_mode():
        for t in range(T):
            enh, cache = stream(mic[:, :, t : t + 1, :], far[:, :, t : t + 1, :], cache)
            outs.append(enh)
    stream_out = torch.cat(outs, dim=2)

    # Compare (allow small diff due to BN running stats vs batch stats in training mode)
    diff = (off - stream_out).abs().max().item()
    mse = ((off - stream_out) ** 2).mean().item()
    print(f"Max abs diff: {diff:.6f}  MSE: {mse:.8f}")
    print("PASS" if diff < 2e-2 and mse < 1e-3 else "FAIL (may need eval mode or larger tolerance)")
