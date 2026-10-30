"""StreamingCAGCRN: frame-by-frame CAGCRN (PyTorch + ONNX-exportable).

端到端模型层严格因果：TFAG 的最大历史感受野为 8 帧，DGConvBlock 使用因果左
padding，TFGRU 通过 hidden state 累积历史，CATA 则维护配置大小的 farend K/V
滑动窗口。每输入一帧立即输出同索引结果，模型层 algorithmic lookahead 为 0，
尾部无需额外零帧 flush。图外 STFT 是否引入 lookahead 由 ``center`` 单独决定。

流式 wrapper 使用 canonical 40 个 flat cache 张量；TFAG 与 mic 帧缓存为 9 帧。
"""

from __future__ import annotations

import sys
from pathlib import Path
_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.aec.models.discriminative.cagcrn import (
    CAGCRN,
    ConvBlock,
    DGConvBlock,
    Encoder,
    TFGRU,
    TFAG,
    Decoder,
    ERB,
    CATA,
    CAGCRN_ALGORITHMIC_DELAY_FRAMES,
)


# ---------------------------------------------------------------------------
# StreamDGConvBlock
# ---------------------------------------------------------------------------
# DGConvBlock 的 depth_conv 在时间维度上做因果左 pad：
#   pad_size = (kernel_size[0]-1) * dilation[0] = 2 * dilation
#   向后看（历史依赖）= pad_size 帧，向前看 = 0 帧
# 各 Encoder/Decoder 中 dilation 取值：1→2 帧, 2→4 帧, 4→8 帧
# ---------------------------------------------------------------------------

class StreamDGConvBlock(nn.Module):
    def __init__(self, blk: DGConvBlock):
        super().__init__()
        self.blk = blk
        self.pad_size = blk.pad_size

    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype, out_channels: int, freq_dim: int) -> torch.Tensor:
        return torch.zeros(batch_size, out_channels, self.pad_size, freq_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = x
        p1 = self.blk.point_act(self.blk.point_bn1(self.blk.point_conv1(x)))
        if cache.numel() == 0:
            cache = torch.zeros(x.shape[0], p1.shape[1], self.pad_size, p1.shape[3], device=p1.device, dtype=p1.dtype)
        inp = torch.cat([cache, p1], dim=2)
        if inp.shape[2] < self.pad_size + 1:
            inp = F.pad(inp, [0, 0, self.pad_size + 1 - inp.shape[2], 0])
        out = self.blk.depth_act(self.blk.depth_bn(self.blk.depth_conv(inp)))
        out = self.blk.point_bn2(self.blk.point_conv2(out))
        new_cache = inp[:, :, -self.pad_size:, :]
        return out + x1, new_cache


# ---------------------------------------------------------------------------
# StreamCATA
# ---------------------------------------------------------------------------
# CATA 是严格因果的（向前看 0 帧），k_cache/v_cache 各保持 d-1 帧。
# ---------------------------------------------------------------------------

class StreamCATA(nn.Module):
    def __init__(self, cata: CATA):
        super().__init__()
        self.cata = cata
        self.d = int(cata.delay_frames)

    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # 零填充使 ONNX trace 始终经过 ``cat`` 分支。
        return (
            torch.zeros(batch_size, 12, self.d - 1, 65, device=device, dtype=dtype),
            torch.zeros(batch_size, 12, self.d - 1, 65, device=device, dtype=dtype),
        )

    def forward(self, mic: torch.Tensor, ref: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.d

        Q_mic = self.cata.pconv_mic(mic)
        K_mic = self.cata.pconv_mic2(Q_mic)
        V_mic = mic

        attn_scores_mic = torch.matmul(Q_mic.permute(0, 2, 3, 1), K_mic.permute(0, 2, 1, 3))
        attn_scores_mic = F.softmax(attn_scores_mic / (Q_mic.shape[1] ** 0.5), dim=-1)
        mic_out = torch.matmul(attn_scores_mic, V_mic.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        K_ref = self.cata.pconv_ref(ref)
        # 用 pad、concat、slice 和 unsqueeze 避免 ONNX trace 中的条件分支。
        k_padded = F.pad(k_cache, (0, 0, d - 1, 0))
        k_full = torch.cat([k_padded, K_ref], dim=2)
        k_full = k_full[:, :, -d:, :]  # (B, 12, d, F)
        ku = k_full.unsqueeze(2)  # (B, 12, 1, d, F)

        ref_score = Q_mic.unsqueeze(-2) * ku
        # 论文公式沿 sliding delay/window 轴归一化，而不是沿频率轴。
        ref_score = F.softmax(ref_score, dim=-2)

        v_padded = F.pad(v_cache, (0, 0, d - 1, 0))
        v_full = torch.cat([v_padded, ref], dim=2)
        v_full = v_full[:, :, -d:, :]  # (B, 12, d, F)
        vu = v_full.unsqueeze(2)  # (B, 12, 1, d, F)
        ref_out = torch.sum(vu * ref_score, dim=-2)

        combined = torch.cat([mic_out, ref_out], dim=1)
        out = self.cata.fusion(combined)

        new_k_cache = k_full[:, :, 1:, :]
        new_v_cache = v_full[:, :, 1:, :]
        return out, mic_out, new_k_cache, new_v_cache


# ---------------------------------------------------------------------------
# StreamEncoder
# ---------------------------------------------------------------------------
# Encoder 包含 mic/ref 的 pre_conv、CATA、mix/mic 的后续 conv。
# 端到端有限历史由 DGConv 链与 CATA 窗口共同决定：
#   mic 端 DGConv = mic_pre(2) + mic_convs(2+4+8) = 16 帧
#   ref 端另含 CATA 的 d-1 帧候选历史；TFGRU hidden state 则累积完整历史
# en_outs_buf 缓存 8 层输出，每层最多 5 帧，供 Decoder skip 使用；这是 skip 历史缓存，
# 不同于 TFAG 覆盖当前帧与此前 8 帧的 9-frame buffer。
# ---------------------------------------------------------------------------

class StreamEncoder(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder
        self.stream_align = StreamCATA(encoder.align)
        self.stream_mic_pre = [self._wrap(m) for m in encoder.mic_convs_pre]
        self.stream_ref_pre = [self._wrap(m) for m in encoder.ref_convs_pre]
        self.stream_mix = [self._wrap(m) for m in encoder.mix_convs]
        self.stream_mic = [self._wrap(m) for m in encoder.mic_convs]

    @staticmethod
    def _wrap(m):
        return StreamDGConvBlock(m) if isinstance(m, DGConvBlock) else m

    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        c: Dict[str, Any] = {}
        _z1 = torch.zeros(batch_size, device=device, dtype=dtype)
        c["mic_pre_caches"] = [m.init_cache(batch_size, device, dtype, m.blk.point_conv1.out_channels, 65) if isinstance(m, StreamDGConvBlock) else _z1 for m in self.stream_mic_pre]
        c["ref_pre_caches"] = [m.init_cache(batch_size, device, dtype, m.blk.point_conv1.out_channels, 65) if isinstance(m, StreamDGConvBlock) else _z1 for m in self.stream_ref_pre]
        c["mix_caches"] = [m.init_cache(batch_size, device, dtype, m.blk.point_conv1.out_channels, 33) if isinstance(m, StreamDGConvBlock) else _z1 for m in self.stream_mix]
        c["mic_caches"] = [m.init_cache(batch_size, device, dtype, m.blk.point_conv1.out_channels, 33) if isinstance(m, StreamDGConvBlock) else _z1 for m in self.stream_mic]
        c["k_cache"], c["v_cache"] = self.stream_align.init_cache(batch_size, device, dtype)
        # 使用零长度时间维，使 ``torch.cat`` 无需 Python 条件分支。
        c["en_outs_buf"] = [
            torch.zeros(batch_size, 12, 0, 65, device=device, dtype=dtype),
            torch.zeros(batch_size, 12, 0, 65, device=device, dtype=dtype),
            torch.zeros(batch_size, 24, 0, 33, device=device, dtype=dtype),
            torch.zeros(batch_size, 24, 0, 33, device=device, dtype=dtype),
            torch.zeros(batch_size, 24, 0, 33, device=device, dtype=dtype),
            torch.zeros(batch_size, 24, 0, 33, device=device, dtype=dtype),
            torch.zeros(batch_size, 24, 0, 33, device=device, dtype=dtype),
            torch.zeros(batch_size, 24, 0, 33, device=device, dtype=dtype),
        ]
        return c

    def forward(self, mic: torch.Tensor, ref: torch.Tensor, cache: Dict[str, Any]) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, Dict[str, Any]]:
        mic = self.encoder._extract_subband_features(mic, self.encoder.unfold_mic)
        ref = self.encoder._extract_subband_features(ref, self.encoder.unfold_ref)

        en_outs = []
        for i, m in enumerate(self.stream_mic_pre):
            if isinstance(m, StreamDGConvBlock):
                mic, cache["mic_pre_caches"][i] = m(mic, cache["mic_pre_caches"][i])
            else:
                mic = m(mic)
            ref_m = self.stream_ref_pre[i]
            if isinstance(ref_m, StreamDGConvBlock):
                ref, cache["ref_pre_caches"][i] = ref_m(ref, cache["ref_pre_caches"][i])
            else:
                ref = ref_m(ref)
            en_outs.append(mic)

        mix, _, cache["k_cache"], cache["v_cache"] = self.stream_align(mic, ref, cache["k_cache"], cache["v_cache"])

        for i, m in enumerate(self.stream_mix):
            if isinstance(m, StreamDGConvBlock):
                mix, cache["mix_caches"][i] = m(mix, cache["mix_caches"][i])
            else:
                mix = m(mix)
            en_outs.append(mix)

        for i, m in enumerate(self.stream_mic):
            if isinstance(m, StreamDGConvBlock):
                mic, cache["mic_caches"][i] = m(mic, cache["mic_caches"][i])
            else:
                mic = m(mic)

        buf = cache["en_outs_buf"]
        new_buf = []
        for i, eo in enumerate(en_outs):
            merged = torch.cat([buf[i], eo], dim=2)
            merged = merged[:, :, -5:, :]
            new_buf.append(merged)
        cache["en_outs_buf"] = new_buf
        return mix, en_outs, mic, cache


# ---------------------------------------------------------------------------
# StreamTFGRU
# ---------------------------------------------------------------------------
# TFGRU 由 intra_gru（帧内）和 inter_gru（帧间）组成。
# inter_gru 沿时间轴传播 hidden state，严格因果：
#   向后看 = 理论无限（hidden state 累积全部历史），向前看 = 0 帧
# 缓存为单个 hidden state 张量 (1, B*width, hidden_size)，非帧缓存。
# ---------------------------------------------------------------------------

class StreamTFGRU(nn.Module):
    def __init__(self, tfgru: TFGRU):
        super().__init__()
        self.tfgru = tfgru
        self.input_size = tfgru.input_size
        self.width = tfgru.width
        self.hidden_size = tfgru.hidden_size

    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(1, batch_size * self.width, self.hidden_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> Tuple[torch.Tensor, torch.Tensor]:
        x4 = x.permute(0, 2, 3, 1)
        b, t, w, c = x4.shape
        intra_x = x4.reshape(b * t, w, c)
        intra_x, _ = self.tfgru._grouped_gru_forward(intra_x, self.tfgru.intra_gru)
        intra_x = self.tfgru.intra_fc(intra_x)
        intra_x = intra_x.reshape(b, t, w, self.input_size)
        intra_x = self.tfgru.intra_norm(intra_x)
        intra_out = x4 + intra_x
        x2 = intra_out.permute(0, 2, 1, 3).reshape(b * w, t, c)
        inter_x, inter_h = self.tfgru._grouped_gru_forward(x2, self.tfgru.inter_gru, h)
        inter_x = self.tfgru.inter_fc(inter_x)
        inter_x = inter_x.reshape(b, w, t, self.input_size).permute(0, 2, 1, 3).contiguous()
        inter_x = self.tfgru.inter_norm(inter_x)
        inter_out = intra_out + inter_x
        out = inter_out.permute(0, 3, 1, 2)
        return out, inter_h


# ---------------------------------------------------------------------------
# StreamTFAG
# ---------------------------------------------------------------------------
# TFAG 两条 temporal scale 均为因果卷积：最大历史为 dilation=2 的 8 帧。
# 9 帧 buffer 覆盖当前帧及此前 8 帧，输出始终选择 newest frame。
# ---------------------------------------------------------------------------

class StreamTFAG(nn.Module):
    def __init__(self, tfag: TFAG):
        super().__init__()
        self.tfag = tfag
        self.algorithmic_delay_frames = CAGCRN_ALGORITHMIC_DELAY_FRAMES
        self.history_frames = 8
        self.buffer_frames = self.history_frames + 1

    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        c1 = torch.zeros(batch_size, 24, self.buffer_frames, 33, device=device, dtype=dtype)
        c2 = torch.zeros(batch_size, 24, self.buffer_frames, 33, device=device, dtype=dtype)
        return c1, c2

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = torch.cat([c1[:, :, 1:, :], x1], dim=2)
        c2 = torch.cat([c2[:, :, 1:, :], x2], dim=2)
        mix = c1 + c2
        temp_feats = [conv(mix) for conv in self.tfag.temporal_scales]
        freq_feats = [conv(mix) for conv in self.tfag.freq_scales]
        target_size = (self.buffer_frames, 33)
        temp_feats = [F.interpolate(feat, size=target_size, mode="nearest") for feat in temp_feats]
        freq_feats = [F.interpolate(feat, size=target_size, mode="nearest") for feat in freq_feats]
        multi_scale_feats = torch.cat(temp_feats + freq_feats, dim=1)
        gate = self.tfag.gate_generator(multi_scale_feats)
        out = gate * c1 + (1 - gate) * c2
        out = self.tfag.fusion(out)
        out = out[:, :, -1:, :]
        return out, c1, c2


# ---------------------------------------------------------------------------
# StreamDecoder
# ---------------------------------------------------------------------------
# Decoder 由若干 ConvBlock + DGConvBlock 组成，全部因果。
# 最大向后看来自 de_convs[0] (dilation=4)：8 帧。
# 每层 skip-connection 通过 en_outs_buf 获取历史输出。
# ---------------------------------------------------------------------------

class StreamDecoder(nn.Module):
    def __init__(self, decoder: Decoder):
        super().__init__()
        self.decoder = decoder
        self.stream_de_convs = [self._wrap(m) for m in decoder.de_convs]
        self.de_conv_freq_dims = [33, 33, 33, 65]

    @staticmethod
    def _wrap(m):
        return StreamDGConvBlock(m) if isinstance(m, DGConvBlock) else m

    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        c: Dict[str, Any] = {}
        c["de_conv_caches"] = []
        dg_idx = 0
        for i, m in enumerate(self.stream_de_convs):
            if isinstance(m, StreamDGConvBlock):
                fd = self.de_conv_freq_dims[dg_idx]
                c["de_conv_caches"].append(m.init_cache(batch_size, device, dtype, m.blk.point_conv1.out_channels, fd))
                dg_idx += 1
            else:
                c["de_conv_caches"].append(
                    torch.zeros(batch_size, device=device, dtype=dtype)
                )
        return c

    def forward(self, x: torch.Tensor, en_outs: List[torch.Tensor], cache: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        for i, m in enumerate(self.stream_de_convs):
            x = x + en_outs[len(self.stream_de_convs) - 1 - i]
            if isinstance(m, StreamDGConvBlock):
                x, cache["de_conv_caches"][i] = m(x, cache["de_conv_caches"][i])
            else:
                x = m(x)
        return x, cache


# ---------------------------------------------------------------------------
# StreamingCAGCRN
# ---------------------------------------------------------------------------
# 端到端流式 CAGCRN：输入 (B, F, 1, 2) 后立即输出同索引帧。
# 信号流：mic_spec/ref_spec → ERB → Encoder → TFGRU1/2/3 → TFAG → Decoder → Mask
# 9-frame TFAG 与 mic/ref buffer 仅保存历史；输出和 skip 均选择 newest frame。
# ---------------------------------------------------------------------------

class StreamingCAGCRN(nn.Module):
    def __init__(self, cagcrn: CAGCRN):
        super().__init__()
        self.net = cagcrn._net
        self.n_fft = cagcrn.n_fft
        self.hop_length = cagcrn.hop_length
        self.win_length = cagcrn.win_length
        self.window = cagcrn.window
        self.center = cagcrn.center
        self.target_fs = cagcrn.target_fs
        self.erb = self.net.erb
        self.encoder = StreamEncoder(self.net.encoder)
        self.tfgru1 = StreamTFGRU(self.net.tfgru1)
        self.tfgru2 = StreamTFGRU(self.net.tfgru2)
        self.tfgru3 = StreamTFGRU(self.net.tfgru3)
        self.tfag = StreamTFAG(self.net.multi_scale_gate)
        self.decoder = StreamDecoder(self.net.decoder)
        self.mask = self.net.mask

    def init_cache(self, batch_size: int = 1, device: str = "cpu", dtype=torch.float32) -> Dict[str, Any]:
        cache: Dict[str, Any] = {}
        cache["encoder"] = self.encoder.init_cache(batch_size, device, dtype)
        cache["tfgru1_h"] = self.tfgru1.init_cache(batch_size, device, dtype)
        cache["tfgru2_h"] = self.tfgru2.init_cache(batch_size, device, dtype)
        cache["tfgru3_h"] = self.tfgru3.init_cache(batch_size, device, dtype)
        cache["tfag_c1"], cache["tfag_c2"] = self.tfag.init_cache(batch_size, device, dtype)
        cache["decoder"] = self.decoder.init_cache(batch_size, device, dtype)
        freq_bins = self.n_fft // 2 + 1
        delay_buffer_frames = self.tfag.buffer_frames
        cache["mic_delay"] = torch.zeros(
            batch_size, freq_bins, delay_buffer_frames, 2, device=device, dtype=dtype
        )
        return cache

    def forward(self, mic_spec: torch.Tensor, ref_spec: torch.Tensor, cache: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        mic_real = mic_spec[..., 0].permute(0, 2, 1)
        mic_imag = mic_spec[..., 1].permute(0, 2, 1)
        mic_mag = torch.sqrt(mic_real**2 + mic_imag**2 + 1e-12)
        mic_feat = torch.stack([mic_mag, mic_real, mic_imag], dim=1)
        ref_real = ref_spec[..., 0].permute(0, 2, 1)
        ref_imag = ref_spec[..., 1].permute(0, 2, 1)
        ref_mag = torch.sqrt(ref_real**2 + ref_imag**2 + 1e-12)
        ref_feat = torch.stack([ref_mag, ref_real, ref_imag], dim=1)

        mic_feat = self.erb.bm(mic_feat)
        ref_feat = self.erb.bm(ref_feat)

        mix, _, mic_out, cache["encoder"] = self.encoder(mic_feat, ref_feat, cache["encoder"])

        mix, cache["tfgru1_h"] = self.tfgru1(mix, cache["tfgru1_h"])
        mix, cache["tfgru2_h"] = self.tfgru2(mix, cache["tfgru2_h"])
        mic_out, cache["tfgru3_h"] = self.tfgru3(mic_out, cache["tfgru3_h"])

        tfag_out, cache["tfag_c1"], cache["tfag_c2"] = self.tfag(mix, mic_out, cache["tfag_c1"], cache["tfag_c2"])

        mic_delay = torch.cat([cache["mic_delay"][:, :, 1:, :], mic_spec], dim=2)
        cache["mic_delay"] = mic_delay

        spec_ori = mic_delay[:, :, -1:, :]

        en_outs_buf = cache["encoder"]["en_outs_buf"]
        dec_en_outs = [eo[:, :, -1:, :] for eo in en_outs_buf]

        dec_out, cache["decoder"] = self.decoder(tfag_out, dec_en_outs, cache["decoder"])

        m = self.erb.bs(dec_out)
        spec_enh = self.mask(m, spec_ori.permute(0, 3, 2, 1))
        spec_enh = spec_enh.permute(0, 3, 2, 1)
        return spec_enh, cache


# ---------------------------------------------------------------------------
# StreamingCAGCRNFlat
# ---------------------------------------------------------------------------
# 将 StreamingCAGCRN 的嵌套 cache dict 扁平化为 canonical 40 个张量，便于 ONNX 导出。
# 40 个张量构成：
#   mic_pre(2) + ref_pre(2) + mix(6) + mic(6) + k(1) + v(1) + en_outs(8)
#   + tfgru_h(3) + tfag_c(2) + de_conv(8) + mic_delay(1) = 40
# 调用方式：out, *new_flat = flat(mic, ref, *flat_cache)
# ---------------------------------------------------------------------------

class StreamingCAGCRNFlat(nn.Module):
    def __init__(self, streaming_cagcrn: StreamingCAGCRN):
        super().__init__()
        self.model = streaming_cagcrn

    def init_cache_flat(self, batch_size: int = 1, device: str = "cpu", dtype=torch.float32) -> List[torch.Tensor]:
        cache = self.model.init_cache(batch_size, device, dtype)
        return self._cache_to_flat(cache)

    def get_cache_config(self) -> Dict[str, List[int]]:
        """返回完整 batch-one cache ABI；新 artifact 必须写入 metadata。"""
        cache = self.init_cache_flat(1, "cpu", torch.float32)
        return {
            name: list(value.shape)
            for name, value in zip(self.get_input_names()[2:], cache)
        }

    @property
    def delay_d(self) -> int:
        """返回配置的 CATA 历史窗口，独立于算法输出延迟。"""
        return int(self.model.encoder.stream_align.d)

    @property
    def algorithmic_delay_frames(self) -> int:
        """返回 causal TFAG 的算法输出延迟帧数。"""
        return int(self.model.tfag.algorithmic_delay_frames)

    def _cache_to_flat(self, cache: Dict[str, Any]) -> List[torch.Tensor]:
        enc = cache["encoder"]
        flat = []
        for key in ["mic_pre_caches", "ref_pre_caches", "mix_caches", "mic_caches"]:
            for c in enc[key]:
                flat.append(c)
        flat.append(enc["k_cache"])
        flat.append(enc["v_cache"])
        for eo in enc["en_outs_buf"]:
            flat.append(eo)
        flat.append(cache["tfgru1_h"])
        flat.append(cache["tfgru2_h"])
        flat.append(cache["tfgru3_h"])
        flat.append(cache["tfag_c1"])
        flat.append(cache["tfag_c2"])
        for c in cache["decoder"]["de_conv_caches"]:
            flat.append(c)
        flat.append(cache["mic_delay"])
        return flat

    def _flat_to_cache(self, flat: List[torch.Tensor]) -> Dict[str, Any]:
        cache: Dict[str, Any] = {}
        enc: Dict[str, Any] = {}
        idx = 0
        for key in ["mic_pre_caches", "ref_pre_caches", "mix_caches", "mic_caches"]:
            attr_name = "stream_" + key.replace("_caches", "")
            n = len(getattr(self.model.encoder, attr_name))
            enc[key] = []
            for _ in range(n):
                enc[key].append(flat[idx])
                idx += 1
        enc["k_cache"] = flat[idx]; idx += 1
        enc["v_cache"] = flat[idx]; idx += 1
        enc["en_outs_buf"] = flat[idx:idx+8]; idx += 8
        cache["encoder"] = enc
        cache["tfgru1_h"] = flat[idx]; idx += 1
        cache["tfgru2_h"] = flat[idx]; idx += 1
        cache["tfgru3_h"] = flat[idx]; idx += 1
        cache["tfag_c1"] = flat[idx]; idx += 1
        cache["tfag_c2"] = flat[idx]; idx += 1
        cache["decoder"] = {}
        cache["decoder"]["de_conv_caches"] = []
        for _ in range(len(self.model.decoder.stream_de_convs)):
            cache["decoder"]["de_conv_caches"].append(flat[idx])
            idx += 1
        cache["mic_delay"] = flat[idx]; idx += 1
        if idx != len(flat):
            raise ValueError(f"CAGCRN flat cache count mismatch: expected {idx}, got {len(flat)}")
        return cache

    def forward(self, mic_spec: torch.Tensor, ref_spec: torch.Tensor, *flat_cache: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        cache = self._flat_to_cache(list(flat_cache))
        out, new_cache = self.model(mic_spec, ref_spec, cache)
        new_flat = self._cache_to_flat(new_cache)
        return (out, *new_flat)

    def get_input_names(self) -> List[str]:
        names = ["mic_spec", "ref_spec"]
        names += [f"cache_{i}" for i in range(40)]
        return names

    def get_output_names(self) -> List[str]:
        names = ["enh_spec"]
        names += [f"out_cache_{i}" for i in range(40)]
        return names

    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        axes = {
            "mic_spec": {0: "batch_size"},
            "ref_spec": {0: "batch_size"},
            "enh_spec": {0: "batch_size"},
        }
        for i in range(40):
            axes[f"cache_{i}"] = {0: "batch_size"}
            axes[f"out_cache_{i}"] = {0: "batch_size"}
        # en_outs_buf caches (index 18-25) have dynamic time dim (grows 0→5 then capped)
        for i in range(18, 26):
            axes[f"cache_{i}"][2] = "time"
            axes[f"out_cache_{i}"][2] = "time"
        # TFGRU hidden states flatten B*width on axis 1; axis 0 is the GRU layer.
        for i in range(26, 29):
            axes[f"cache_{i}"] = {1: "batch_flattened"}
            axes[f"out_cache_{i}"] = {1: "batch_flattened"}
        return axes
