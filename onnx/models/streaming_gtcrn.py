"""StreamingGTCRN: ONNX-friendly streaming implementation.

This module provides a pure STFT-domain streaming interface for GTCRN, designed
specifically for ONNX export and deployment. STFT/iSTFT remain outside this model:
callers pass one complex STFT frame and perform waveform conversion themselves.
This implementation:
  - Operates on STFT frames (B, F, T=1, 2)
  - Uses explicit tensor caches (no dict/list) for ONNX compatibility
  - Implements StreamConv2d / StreamConvTranspose2d for causal temporal convolution
  - Follows the gtcrn-main/stream architecture for maximum inference speed

Architecture:
  Input: (B, F=257, T=1, 2) complex STFT frame
  -> ERB + SFE
  -> StreamEncoder (ConvBlocks + StreamGTConvBlocks)
  -> DPGRNN x2 (with inter-RNN state)
  -> StreamDecoder (StreamGTConvBlocks + ConvBlocks)
  -> iERB + Mask
  Output: (B, F=257, T=1, 2) enhanced complex STFT frame

Caches (all tensors for ONNX):
  - conv_cache: (2, B, C, T_buf, F) encoder/decoder temporal buffers
  - tra_cache:  (2, 3, 1, B, C) encoder/decoder TRA GRU states
  - inter_cache: (2, 1, B*F, C) DPGRNN inter-RNN states

Usage:
    # 1. Train a standard GTCRN model and convert it through the supported exporter.
    python onnx/export.py --model streaming_gtcrn \\
        --checkpoint /path/to/best_model.tar --output runs/onnx_debug/gtcrn_stream.onnx

    # 2. For Python callers, derive cache shapes from the streaming model itself.
    device = torch.device("cpu")
    stream_model = StreamingGTCRN().to(device).eval()
    cache = stream_model.init_cache(batch_size=1, device=device, dtype=torch.float32)
    spec_input = torch.randn(1, stream_model.n_fft // 2 + 1, 1, 2, device=device)

    # 3. Pass the frame and cache tensors to the model; waveform conversion stays outside.
    spec_output, *_ = stream_model(
        spec_input, cache["conv_cache"], cache["tra_cache"], cache["inter_cache"]
    )

Reference:
    gtcrn-main/stream/gtcrn_stream.py
    https://github.com/Xiaobin-Rong/gtcrn

Author: Xincheng Zhu
"""

# When run directly, add project root to sys.path
if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

import os
from typing import Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.utils.registry import MODEL_REGISTRY


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GTCRN_DEMO_DEBUG_ROOT = _PROJECT_ROOT / "runs" / "onnx_debug"
_GTCRN_RELEASE_ROOT = _PROJECT_ROOT / "onnx" / "onnx_model"


def _validate_demo_output_path(output_path: str | Path) -> Path:
    """校验 demo 只创建新的 debug ONNX，不触碰正式发布资产。"""
    try:
        candidate = Path(output_path).expanduser()
        if not candidate.is_absolute():
            candidate = _PROJECT_ROOT / candidate
        resolved = candidate.resolve(strict=False)
        debug_root = _GTCRN_DEMO_DEBUG_ROOT.resolve(strict=False)
        release_root = _GTCRN_RELEASE_ROOT.resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid --onnx_out path: {output_path!r}") from exc

    try:
        resolved.relative_to(release_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "--onnx_out is debug-only and cannot write under the ONNX release root "
            f"{release_root}"
        )

    try:
        resolved.relative_to(debug_root)
    except ValueError as exc:
        raise ValueError(
            "--onnx_out must be a new .onnx file under the explicit debug directory "
            f"{debug_root}"
        ) from exc

    if resolved == debug_root or resolved.suffix.lower() != ".onnx":
        raise ValueError(
            "--onnx_out must name a .onnx file under the explicit debug directory "
            f"{debug_root}"
        )
    if candidate.exists() or candidate.is_symlink():
        raise ValueError(
            f"--onnx_out target already exists or is a symlink: {candidate}"
        )
    if candidate.parent.exists() and not candidate.parent.is_dir():
        raise ValueError(f"--onnx_out parent is not a directory: {candidate.parent}")
    if candidate.parent.exists() or candidate.parent.is_symlink():
        try:
            candidate.parent.resolve(strict=True).relative_to(debug_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                "--onnx_out parent must resolve under the explicit debug directory "
                f"{debug_root}"
            ) from exc

    return resolved


def _export_demo_onnx_artifacts(stream_model, output_path: str | Path) -> tuple[Path, Path | None]:
    """经 owned debug 临时文件与稳定 publisher 导出 demo 主/简化产物。"""
    from onnx.export import (
        create_owned_onnx_artifact,
        export_model,
        prepare_debug_output_target,
        publish_debug_onnx,
        write_streaming_runtime_metadata,
    )

    target_path = _validate_demo_output_path(output_path)
    main_target = prepare_debug_output_target(target_path, require_new=True)
    main_owned = create_owned_onnx_artifact()
    simple_target = None
    simple_owned = None
    simple_path = None
    try:
        export_model(stream_model, os.fspath(main_owned), opset=17)
        write_streaming_runtime_metadata(os.fspath(main_owned), stream_model)

        # 简化同样只写 owned 文件；目标目录直到安全 publisher 才接触。
        try:
            from onnx.utils import pypi_onnx_scope

            with pypi_onnx_scope() as onnx:
                from onnxsim import simplify

                onnx_model = onnx.load(os.fspath(main_owned))
                model_simp, check = simplify(onnx_model)
                if check:
                    simple_path = target_path.with_stem(target_path.stem + "_simple")
                    simple_target = prepare_debug_output_target(
                        simple_path,
                        require_new=True,
                    )
                    simple_owned = create_owned_onnx_artifact()
                    onnx.save(model_simp, os.fspath(simple_owned))
                    # onnxsim 重存模型时可能丢弃 metadata_props，简化产物必须重写
                    # runtime metadata，否则下游 ORT 预检会因缺 ABI 直接拒绝。
                    write_streaming_runtime_metadata(os.fspath(simple_owned), stream_model)
        except ImportError:
            print("  (onnx-simplifier not installed, skipping simplification)")

        publish_debug_onnx(main_owned, main_target)
        if simple_owned is not None and simple_target is not None:
            publish_debug_onnx(simple_owned, simple_target)
        return target_path, simple_path
    finally:
        main_owned.close()
        main_target.close()
        if simple_owned is not None:
            simple_owned.close()
        if simple_target is not None:
            simple_target.close()


# =============================================================================
#  Streaming Convolution Primitives
# =============================================================================

class StreamConv2d(nn.Module):
    """Causal 2D convolution with explicit temporal buffer for streaming.

    Prepends a temporal buffer (size: kernel_t - 1) to the input before
    convolution, then updates the buffer. Ensures strict causality for ONNX.
    Requires padding=(0, F_pad) to only pad freq dimension.

    Args:
        Same as nn.Conv2d, except padding[0] MUST be 0 for causality.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        if isinstance(padding, int):
            self.T_pad, self.F_pad = padding, padding
        else:
            self.T_pad, self.F_pad = padding
        assert self.T_pad == 0, "T_pad must be 0 for causal streaming"

        self.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride, padding, dilation, groups, bias)

    def forward(self, x, cache):
        """
        Args:
            x: (B, C, 1, F) — single time frame
            cache: (B, C, T_buf, F) — temporal buffer

        Returns:
            out: (B, C, 1, F) — output frame
            cache_new: (B, C, T_buf, F) — updated buffer
        """
        inp = torch.cat([cache, x], dim=2)  # (B, C, T_buf+1, F)
        out = self.Conv2d(inp)
        cache_new = inp[:, :, 1:, :]  # shift: keep last T_buf frames
        return out, cache_new


class StreamConvTranspose2d(nn.Module):
    """Causal ConvTranspose2d for streaming via Conv2d with flipped weights.

    Implements ConvTranspose2d using Conv2d + weight reversal for faster ONNX
    inference (gtcrn-main/stream Version 2). Requires padding=(0, F_pad) and
    stride=(1, F_stride).

    Args:
        Same as nn.ConvTranspose2d, except:
          - padding[0] must be 0
          - stride[0] must be 1
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.T_size, self.F_size = kernel_size, kernel_size
        else:
            self.T_size, self.F_size = kernel_size
        if isinstance(stride, int):
            self.T_stride, self.F_stride = stride, stride
        else:
            self.T_stride, self.F_stride = stride
        assert self.T_stride == 1, "T_stride must be 1 for streaming"
        if isinstance(padding, int):
            self.T_pad, self.F_pad = padding, padding
        else:
            self.T_pad, self.F_pad = padding
        assert self.T_pad == 0, "T_pad must be 0 for causality"
        if isinstance(dilation, int):
            self.T_dilation, self.F_dilation = dilation, dilation
        else:
            self.T_dilation, self.F_dilation = dilation

        # Implement ConvTranspose2d using Conv2d with weight-time reversal
        self.ConvTranspose2d = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=(self.T_stride, 1),
            padding=(self.T_pad, 0),
            dilation=dilation, groups=groups, bias=bias
        )

    def forward(self, x, cache):
        """
        Args:
            x: (B, C, 1, F)
            cache: (B, C, T_buf, F)

        Returns:
            out: (B, C, 1, F_out)
            cache_new: (B, C, T_buf, F)
        """
        inp = torch.cat([cache, x], dim=2)  # (B, C, T_buf+1, F)
        cache_new = inp[:, :, 1:, :]
        B, C, T, F_dim = inp.shape

        # Upsampling for freq dimension
        if self.F_stride > 1:
            # (B,C,T,F) -> (B,C,T,F*F_stride) via zero-interleave
            inp = torch.cat(
                [inp[:, :, :, :, None],
                 torch.zeros([B, C, T, F_dim, self.F_stride - 1], device=inp.device, dtype=inp.dtype)],
                dim=-1
            ).reshape([B, C, T, -1])
            left_pad = self.F_stride - 1
            if self.F_size > 1:
                if left_pad <= self.F_size - 1:
                    pad_l = (self.F_size - 1) * self.F_dilation - self.F_pad
                    pad_r = (self.F_size - 1) * self.F_dilation - self.F_pad - left_pad
                    inp = F.pad(inp, [pad_l, pad_r, 0, 0])
                else:
                    raise NotImplementedError("left_pad > F_size - 1 not supported")
            else:
                raise NotImplementedError("F_size == 1 with F_stride > 1 not supported")
        else:
            # F_stride == 1
            pad_val = (self.F_size - 1) * self.F_dilation - self.F_pad
            inp = F.pad(inp, [pad_val, pad_val])

        out = self.ConvTranspose2d(inp)
        return out, cache_new


# =============================================================================
#  Streaming Architecture Components
# =============================================================================

class ERB(nn.Module):
    """ERB frequency grouping (identical to gtcrn.py)."""
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self._erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    @staticmethod
    def _hz2erb(freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    @staticmethod
    def _erb2hz(erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    def _erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft, high_lim, fs):
        low_lim = erb_subband_1 / nfft * fs
        erb_low, erb_high = self._hz2erb(low_lim), self._hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self._erb2hz(erb_points) / fs * nfft).astype(np.int32)
        f = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)
        f[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            f[i+1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12) / (bins[i+1] - bins[i] + 1e-12)
            f[i+1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i+2]) + 1e-12) / (bins[i+2] - bins[i+1] + 1e-12)
        f[-1, bins[-2]:bins[-1]+1] = 1 - f[-2, bins[-2]:bins[-1]+1]
        f = f[:, erb_subband_1:]
        return torch.from_numpy(np.abs(f))

    def bm(self, x):
        """Band Mapping: (B, C, T, F) -> (B, C, T, F_erb)."""
        return torch.cat([x[..., :self.erb_subband_1], self.erb_fc(x[..., self.erb_subband_1:])], dim=-1)

    def bs(self, x_erb):
        """Band Synthesis: (B, C, T, F_erb) -> (B, C, T, F)."""
        return torch.cat([x_erb[..., :self.erb_subband_1], self.ierb_fc(x_erb[..., self.erb_subband_1:])], dim=-1)


class SFE(nn.Module):
    """Shallow Feature Extraction (identical to gtcrn.py)."""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))

    def forward(self, x):
        return self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])


class StreamTRA(nn.Module):
    """Streaming Temporal Recurrent Attention.

    Differs from TRA in gtcrn.py: forward signature explicitly includes h_cache
    for ONNX compatibility (no default None, must pass state).
    """
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x, h_cache):
        """
        Args:
            x: (B, C, T=1, F)
            h_cache: (1, B, C*2) GRU hidden state

        Returns:
            out: (B, C, T=1, F)
            h_cache_new: (1, B, C*2)
        """
        zt = torch.mean(x.pow(2), dim=-1)  # (B, C, T=1)
        at, h_cache_new = self.att_gru(zt.transpose(1, 2), h_cache)
        at = self.att_fc(at).transpose(1, 2)
        at = self.att_act(at).unsqueeze(-1)
        return x * at, h_cache_new


class ConvBlock(nn.Module):
    """Standard Conv / ConvTranspose + BN + activation (same as gtcrn.py)."""
    def __init__(self, in_ch, out_ch, kernel, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        C = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = C(in_ch, out_ch, kernel, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class StreamGTConvBlock(nn.Module):
    """Streaming Grouped Temporal Conv Block with explicit state management."""
    def __init__(self, in_ch, hid, kernel, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        C = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        StreamC = StreamConvTranspose2d if use_deconv else StreamConv2d

        self.sfe = SFE(3, 1)
        self.pc1 = C(in_ch // 2 * 3, hid, 1)
        self.bn1 = nn.BatchNorm2d(hid)
        self.act1 = nn.PReLU()
        self.dc = StreamC(hid, hid, kernel, stride=stride, padding=padding, dilation=dilation, groups=hid)
        self.dbn = nn.BatchNorm2d(hid)
        self.dact = nn.PReLU()
        self.pc2 = C(hid, in_ch // 2, 1)
        self.bn2 = nn.BatchNorm2d(in_ch // 2)
        self.tra = StreamTRA(in_ch // 2)

    def shuffle(self, x1, x2):
        """Channel shuffle: [x1, x2] -> interleave."""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()
        return x.view(x.shape[0], -1, x.shape[3], x.shape[4])

    def forward(self, x, conv_cache, tra_cache):
        """
        Args:
            x: (B, C, T=1, F)
            conv_cache: (B, hid, T_buf, F)
            tra_cache: (1, B, C*2) TRA GRU hidden

        Returns:
            out: (B, C, T=1, F)
            conv_cache_new: (B, hid, T_buf, F)
            tra_cache_new: (1, B, C*2)
        """
        x1, x2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]
        h1 = self.act1(self.bn1(self.pc1(self.sfe(x1))))
        h1, conv_cache_new = self.dc(h1, conv_cache)
        h1 = self.dact(self.dbn(h1))
        h1 = self.bn2(self.pc2(h1))
        h1, tra_cache_new = self.tra(h1, tra_cache)
        out = self.shuffle(h1, x2)
        return out, conv_cache_new, tra_cache_new


class GRNN(nn.Module):
    """Grouped GRU (same as gtcrn.py)."""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        if h is None:
            d = 2 if self.bidirectional else 1
            h = torch.zeros(self.num_layers * d, x.shape[0], self.hidden_size,
                            device=x.device, dtype=x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        h1, h2 = torch.chunk(h, 2, dim=-1)
        y1, h1 = self.rnn1(x1, h1.contiguous())
        y2, h2 = self.rnn2(x2, h2.contiguous())
        return torch.cat([y1, y2], dim=-1), torch.cat([h1, h2], dim=-1)


class DPGRNN(nn.Module):
    """Streaming Dual-Path Grouped RNN (matches gtcrn-main/stream)."""
    def __init__(self, input_size, width, hidden_size):
        super().__init__()
        self.width = width
        self.hidden_size = hidden_size
        self.intra_rnn = GRNN(input_size, hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
        self.inter_rnn = GRNN(input_size, hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x, inter_cache):
        """
        Args:
            x: (B, C, T=1, F)
            inter_cache: (1, B*F, hidden_size) inter-RNN hidden

        Returns:
            out: (B, C, T=1, F)
            inter_cache_new: (1, B*F, hidden_size)
        """
        x = x.permute(0, 2, 3, 1)  # (B, T=1, F, C)
        B, T, _F, C = x.shape
        # Intra: bidirectional over freq
        intra_x = self.intra_fc(self.intra_rnn(x.reshape(B * T, _F, C))[0])
        intra_x = self.intra_ln(intra_x.reshape(B, T, self.width, self.hidden_size))
        intra_out = x + intra_x
        # Inter: unidirectional over time with state
        x2 = intra_out.permute(0, 2, 1, 3).reshape(B * _F, T, C)
        inter_x, inter_cache_new = self.inter_rnn(x2, inter_cache)
        # 与离线 DPGRNN 对齐：inter RNN 输出必须经过 inter_fc 投影；
        # 缺失该层会导致流式与离线数值发散。
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(B, self.width, T, self.hidden_size).permute(0, 2, 1, 3)
        out = (intra_out + self.inter_ln(inter_x)).permute(0, 3, 1, 2)
        return out, inter_cache_new


class StreamEncoder(nn.Module):
    """Streaming Encoder: 2 ConvBlocks + 3 StreamGTConvBlocks."""
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(9, 16, (1, 5), (1, 2), (0, 2)),
            ConvBlock(16, 16, (1, 5), (1, 2), (0, 2), groups=2),
            StreamGTConvBlock(16, 16, (3, 3), (1, 1), (0, 1), (1, 1)),
            StreamGTConvBlock(16, 16, (3, 3), (1, 1), (0, 1), (2, 1)),
            StreamGTConvBlock(16, 16, (3, 3), (1, 1), (0, 1), (5, 1)),
        ])

    def forward(self, x, conv_cache, tra_cache):
        """
        Args:
            x: (B, 9, T=1, F)
            conv_cache: (B, C, T_buf, F) stacked for 3 GTConvBlocks
            tra_cache: (3, 1, B, C*2) stacked TRA states

        Returns:
            x: (B, 16, T=1, F_reduced)
            en_outs: list of 5 skip connections
            conv_cache_new: updated conv buffers
            tra_cache_new: updated TRA states
        """
        en_outs = []
        # Layers 0,1: ConvBlocks (no state)
        for i in range(2):
            x = self.en_convs[i](x)
            en_outs.append(x)

        # Layers 2,3,4: StreamGTConvBlocks (with state)
        # Buffer sizes: (kT-1)*dilation = (3-1)*{1,2,5} = {2,4,10}
        x, conv_cache[:, :, :2, :], tra_cache[0] = self.en_convs[2](x, conv_cache[:, :, :2, :], tra_cache[0])
        en_outs.append(x)
        x, conv_cache[:, :, 2:6, :], tra_cache[1] = self.en_convs[3](x, conv_cache[:, :, 2:6, :], tra_cache[1])
        en_outs.append(x)
        x, conv_cache[:, :, 6:16, :], tra_cache[2] = self.en_convs[4](x, conv_cache[:, :, 6:16, :], tra_cache[2])
        en_outs.append(x)

        return x, en_outs, conv_cache, tra_cache


class StreamDecoder(nn.Module):
    """Streaming Decoder: 3 StreamGTConvBlocks + 2 ConvBlocks."""
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            StreamGTConvBlock(16, 16, (3, 3), (1, 1), (0, 1), (5, 1), use_deconv=True),
            StreamGTConvBlock(16, 16, (3, 3), (1, 1), (0, 1), (2, 1), use_deconv=True),
            StreamGTConvBlock(16, 16, (3, 3), (1, 1), (0, 1), (1, 1), use_deconv=True),
            ConvBlock(16, 16, (1, 5), (1, 2), (0, 2), groups=2, use_deconv=True),
            ConvBlock(16, 2, (1, 5), (1, 2), (0, 2), use_deconv=True, is_last=True),
        ])

    def forward(self, x, en_outs, conv_cache, tra_cache):
        """
        Args:
            x: (B, 16, T=1, F)
            en_outs: list of 5 skip connections from encoder
            conv_cache: (B, C, T_buf, F) stacked for 3 GTConvBlocks
            tra_cache: (3, 1, B, C*2) stacked TRA states

        Returns:
            x: (B, 2, T=1, F_full) mask
            conv_cache_new: updated conv buffers
            tra_cache_new: updated TRA states
        """
        # Layers 0,1,2: StreamGTConvBlocks (reverse order of encoder)
        x, conv_cache[:, :, 6:16, :], tra_cache[0] = self.de_convs[0](x + en_outs[4], conv_cache[:, :, 6:16, :], tra_cache[0])
        x, conv_cache[:, :, 2:6, :], tra_cache[1] = self.de_convs[1](x + en_outs[3], conv_cache[:, :, 2:6, :], tra_cache[1])
        x, conv_cache[:, :, :2, :], tra_cache[2] = self.de_convs[2](x + en_outs[2], conv_cache[:, :, :2, :], tra_cache[2])

        # Layers 3,4: ConvBlocks (no state)
        x = self.de_convs[3](x + en_outs[1])
        x = self.de_convs[4](x + en_outs[0])
        return x, conv_cache, tra_cache


class Mask(nn.Module):
    """Complex Ratio Mask (same as gtcrn.py)."""
    def forward(self, mask, spec):
        sr = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        si = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
        return torch.stack([sr, si], dim=1)


# =============================================================================
#  Main Streaming Model
# =============================================================================

@MODEL_REGISTRY.register("StreamingGTCRN")
class StreamingGTCRN(nn.Module):
    """ONNX-friendly streaming GTCRN.

    Operates on single STFT frames (B, F=257, T=1, 2), returns enhanced frame
    with updated caches. All state is explicit tensors (no dict/list) for ONNX.

    Args (same as GTCRN):
        n_fft: FFT size (default 512)
        hop_length: STFT hop (default 256)
        win_length: STFT window length (default 512)
    """
    def __init__(
        self,
        n_fft=512,
        hop_length=256,
        win_length=512,
        target_fs=16000,
        window="hann",
        center=True,
    ):
        super().__init__()
        if window != "hann":
            raise ValueError(f"StreamingGTCRN supports only window='hann', got {window!r}")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.target_fs = target_fs
        self.window = window
        self.center = bool(center)
        self.erb = ERB(65, 64, nfft=n_fft)
        self.sfe = SFE(3, 1)
        self.encoder = StreamEncoder()
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)
        self.decoder = StreamDecoder()
        self.mask = Mask()

        # Cache shape metadata (for ONNX export to avoid hardcoded dims)
        self._enc_ch = int(self.encoder.en_convs[0].conv.out_channels)
        self._dpgrnn_f = int(self.dpgrnn1.width)
        self._dpgrnn_h = int(self.dpgrnn1.hidden_size)
        # Temporal buffer: sum of (kT-1)*dilation for each StreamGTConvBlock
        self._conv_T = 0
        for m in list(self.encoder.en_convs) + list(self.decoder.de_convs):
            if hasattr(m, "dc") and hasattr(m.dc, "T_size"):
                self._conv_T += (int(m.dc.T_size) - 1) * int(m.dc.T_dilation)

    CACHE_KEYS = ("conv_cache", "tra_cache", "inter_cache")

    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Return correctly-shaped zero-filled cache tensors for ONNX export."""
        C, F, H, T = self._enc_ch, self._dpgrnn_f, self._dpgrnn_h, self._conv_T
        return {
            "conv_cache": torch.zeros(2, batch_size, C, T, F, device=device, dtype=dtype),
            "tra_cache": torch.zeros(2, 3, 1, batch_size, C, device=device, dtype=dtype),
            "inter_cache": torch.zeros(2, 1, batch_size * F, H, device=device, dtype=dtype),
        }

    def get_input_names(self) -> list[str]:
        return ["spec", *self.CACHE_KEYS]

    def get_output_names(self) -> list[str]:
        return ["enh", *(f"{name}_out" for name in self.CACHE_KEYS)]

    def get_dynamic_axes(self) -> dict[str, dict[int, str]]:
        return {
            "spec": {0: "batch"},
            "enh": {0: "batch"},
            "conv_cache": {1: "batch"},
            "conv_cache_out": {1: "batch"},
            "tra_cache": {3: "batch"},
            "tra_cache_out": {3: "batch"},
            "inter_cache": {2: "batch_flattened"},
            "inter_cache_out": {2: "batch_flattened"},
        }

    def get_cache_config(self) -> dict[str, list[int]]:
        cache = self.init_cache(1, torch.device("cpu"), torch.float32)
        return {name: list(cache[name].shape) for name in self.CACHE_KEYS}

    def forward(self, spec, conv_cache, tra_cache, inter_cache):
        """
        Args:
            spec: (B, F=257, T=1, 2) complex STFT frame [real, imag]
            conv_cache: (2, B, C=16, T_buf=16, F=33) [enc_cache, dec_cache]
            tra_cache: (2, 3, 1, B, C=16) [enc_tra, dec_tra]
            inter_cache: (2, 1, B*F=33, C=16) [dpgrnn1_h, dpgrnn2_h]

        Returns:
            spec_enh: (B, F=257, T=1, 2) enhanced complex STFT frame
            conv_cache_out: updated conv_cache
            tra_cache_out: updated tra_cache
            inter_cache_out: updated inter_cache
        """
        spec_ref = spec  # (B, F, T=1, 2)

        # Extract features: mag, real, imag
        spec_real = spec[..., 0].permute(0, 2, 1)  # (B, T=1, F)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real ** 2 + spec_imag ** 2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B, 3, T=1, F=257)

        # ERB + SFE
        feat = self.erb.bm(feat)  # (B, 3, T=1, F=129)
        feat = self.sfe(feat)     # (B, 9, T=1, F=129)

        # Encoder
        feat, en_outs, conv_cache[0], tra_cache[0] = self.encoder(feat, conv_cache[0], tra_cache[0])

        # DPGRNN x2
        feat, inter_cache[0] = self.dpgrnn1(feat, inter_cache[0])
        feat, inter_cache[1] = self.dpgrnn2(feat, inter_cache[1])

        # Decoder
        m_feat, conv_cache[1], tra_cache[1] = self.decoder(feat, en_outs, conv_cache[1], tra_cache[1])

        # iERB + Mask
        m = self.erb.bs(m_feat)
        spec_enh = self.mask(m, spec_ref.permute(0, 3, 2, 1))  # (B, 2, T=1, F)
        spec_enh = spec_enh.permute(0, 3, 2, 1)  # (B, F, T=1, 2)

        return spec_enh, conv_cache, tra_cache, inter_cache


# =============================================================================
#  Weight Conversion Utility
# =============================================================================

def convert_gtcrn_to_streaming(stream_model: StreamingGTCRN, gtcrn_model) -> None:
    """Convert trained GTCRN weights to StreamingGTCRN.

    Handles:
      - Direct weight copy for ERB, SFE, DPGRNN, non-streaming convs
      - Wraps streaming conv weights: Conv2d.weight -> dc.Conv2d.weight
      - Flips ConvTranspose2d weights for StreamConvTranspose2d (Version 2)

    Args:
        stream_model: target StreamingGTCRN instance
        gtcrn_model: source GTCRN model (from gtcrn.py)
    """
    # StreamingGTCRN 沿用官方 en_convs/de_convs 容器命名，框架 GTCRN 已统一为
    # layers；其余段（pc1/bn1/act1/dc/dbn/dact/pc2/bn2 等缩写）两者一致，
    # 因此只需重命名这两个容器段即可对齐 src/dst key。
    src_state = {
        k.replace("encoder.layers.", "encoder.en_convs.").replace(
            "decoder.layers.", "decoder.de_convs."
        ): v
        for k, v in gtcrn_model.state_dict().items()
    }
    dst_state = stream_model.state_dict()

    for key in dst_state.keys():
        # Direct copy (ERB, SFE, DPGRNN linear/GRU layers)
        if key in src_state:
            dst_state[key] = src_state[key]
        # Streaming Conv2d: dc.Conv2d.weight <- dc.weight (encoder GTConvBlocks)
        elif key.replace('dc.Conv2d.', 'dc.') in src_state:
            dst_state[key] = src_state[key.replace('dc.Conv2d.', 'dc.')]
        # Streaming ConvTranspose2d (Version 2): flip weights
        elif key.replace('dc.ConvTranspose2d.', 'dc.') in src_state:
            src_key = key.replace('dc.ConvTranspose2d.', 'dc.')
            if key.endswith('weight'):
                w = src_state[src_key]
                # ConvTranspose2d -> Conv2d: flip time & freq dims, possibly permute
                if dst_state[key].shape != w.shape:
                    # Transpose in/out channels
                    w = w.permute(1, 0, 2, 3)
                dst_state[key] = torch.flip(w, dims=[-2, -1])
            else:
                dst_state[key] = src_state[src_key]
        else:
            # Fallback: try removing '.Conv2d' or '.ConvTranspose2d' prefix
            alt_key = key.replace('.Conv2d.', '.').replace('.ConvTranspose2d.', '.')
            if alt_key in src_state:
                if 'ConvTranspose2d' in key and key.endswith('weight'):
                    w = src_state[alt_key]
                    if dst_state[key].shape != w.shape:
                        w = w.permute(1, 0, 2, 3)
                    dst_state[key] = torch.flip(w, dims=[-2, -1])
                else:
                    dst_state[key] = src_state[alt_key]
            else:
                raise ValueError(f"Cannot map StreamingGTCRN key: {key}")

    stream_model.load_state_dict(dst_state)


# =============================================================================
#  Test / Demo
# =============================================================================

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="StreamingGTCRN ONNX export & test")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained GTCRN checkpoint")
    parser.add_argument(
        "--onnx_out",
        type=str,
        default="runs/onnx_debug/gtcrn_stream.onnx",
        help="New debug ONNX file under runs/onnx_debug (release targets are rejected)",
    )
    parser.add_argument("--test_frames", type=int, default=100, help="Number of frames for latency test")
    args = parser.parse_args()
    if args.test_frames <= 0:
        parser.error("--test_frames must be positive")
    _validate_demo_output_path(args.onnx_out)

    device = torch.device("cpu")
    stream_model = StreamingGTCRN().to(device).eval()

    # checkpoint 通过统一的安全、严格加载入口读取。
    if args.checkpoint:
        from onnx.export import load_checkpoint
        from tasks.se.models.discriminative.gtcrn import GTCRN

        gtcrn = GTCRN(n_fft=stream_model.n_fft).to(device).eval()
        load_checkpoint(gtcrn, args.checkpoint)
        convert_gtcrn_to_streaming(stream_model, gtcrn)
        print(f"✓ Loaded weights from {args.checkpoint}")

    # cache 形状由模型接口推导，避免 demo 与模型实现分叉。
    batch_size = 1
    cache = stream_model.init_cache(batch_size, device, torch.float32)
    cache_values = tuple(cache[key] for key in stream_model.CACHE_KEYS)
    n_freq = stream_model.n_fft // 2 + 1

    spec_in = torch.randn(batch_size, n_freq, 1, 2, device=device)
    with torch.no_grad():
        spec_out, *cache_values = stream_model(spec_in, *cache_values)
    print(f"✓ Forward pass: {spec_in.shape} -> {spec_out.shape}")

    onnx_path, simp_path = _export_demo_onnx_artifacts(stream_model, args.onnx_out)
    print(f"✓ Exported to {onnx_path}")
    if simp_path is not None:
        print(f"✓ Simplified ONNX saved to {simp_path}")

    # 延迟 benchmark 使用同一套模型推导的 cache。
    print(f"\nLatency benchmark ({args.test_frames} frames):")
    cache_values = tuple(
        stream_model.init_cache(batch_size, device, torch.float32)[key].clone()
        for key in stream_model.CACHE_KEYS
    )
    times = []
    for _ in range(args.test_frames):
        spec_bench = torch.randn(batch_size, n_freq, 1, 2, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _, *cache_values = stream_model(spec_bench, *cache_values)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    print(f"  Mean: {sum(times) / len(times):.2f} ms")
    print(f"  Max:  {max(times):.2f} ms")
    print(f"  Min:  {min(times):.2f} ms")
    print(f"  RTF:  {sum(times) / len(times) / 16:.3f}  (hop=256 samples @ 16kHz = 16ms)")
