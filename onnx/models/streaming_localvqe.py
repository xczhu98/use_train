"""StreamingLocalVQE：LocalVQE 的逐帧流式、ONNX 可导出 wrapper。"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.aec.models.discriminative.localvqe import (
    LocalVQE,
    _AlignBlock,
    _CCM,
    _DecoderBlock,
    _EncoderBlock,
    _ResidualBlock,
    _S4DBottleneck,
    _SubpixelConv2d,
    _compute_freq_progression,
)


class _StreamingCausalConv(nn.Module):
    """对已归一化的单帧输入执行因果卷积并更新历史。"""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        self.time_history = int(conv.kernel_size[0]) - 1
        kernel_freq = int(conv.kernel_size[1])
        self.freq_left = (kernel_freq - 1) // 2
        self.freq_right = kernel_freq - 1 - self.freq_left

    def forward(self, x: torch.Tensor, cache: torch.Tensor):
        window = torch.cat([cache, x], dim=2)
        window = F.pad(window, (self.freq_left, self.freq_right, 0, 0))
        output = self.conv(window)
        if self.time_history:
            new_cache = torch.cat([cache[:, :, 1:, :], x], dim=2)
        else:
            new_cache = cache
        return output, new_cache


class _StreamingResidualBlock(nn.Module):
    def __init__(self, block: _ResidualBlock):
        super().__init__()
        if block.arch_version not in (2, 3):
            raise ValueError("StreamingLocalVQE only supports arch_version 2 or 3")
        self.norm = block.norm
        self.conv = _StreamingCausalConv(block.conv)
        self.act = block.act

    def forward(self, x: torch.Tensor, cache: torch.Tensor):
        normalized = self.norm(x)
        residual, new_cache = self.conv(normalized, cache)
        return self.act(residual) + x, new_cache


class _StreamingEncoderBlock(nn.Module):
    def __init__(self, block: _EncoderBlock):
        super().__init__()
        if block.arch_version not in (2, 3):
            raise ValueError("StreamingLocalVQE only supports arch_version 2 or 3")
        self.norm = block.norm
        self.conv = _StreamingCausalConv(block.conv)
        self.act = block.act
        self.resblock = _StreamingResidualBlock(block.resblock)

    def forward(
        self,
        x: torch.Tensor,
        conv_cache: torch.Tensor,
        res_cache: torch.Tensor,
    ):
        # cache 必须保存 norm 后历史，才能逐帧严格对齐。
        normalized = self.norm(x)
        output, new_conv_cache = self.conv(normalized, conv_cache)
        output = self.act(output)
        output, new_res_cache = self.resblock(output, res_cache)
        return output, new_conv_cache, new_res_cache


class _StreamingSubpixelConv2d(nn.Module):
    def __init__(self, block: _SubpixelConv2d):
        super().__init__()
        if block.arch_version not in (2, 3):
            raise ValueError("StreamingLocalVQE only supports arch_version 2 or 3")
        self.norm = block.norm
        self.conv = _StreamingCausalConv(block.conv)

    def forward(self, x: torch.Tensor, cache: torch.Tensor):
        normalized = self.norm(x)
        output, new_cache = self.conv(normalized, cache)
        batch, doubled_channels, time, frequency = output.shape
        channels = doubled_channels // 2
        # 严格匹配 base 的 rearrange("b (r c) t f -> b c t (r f)", r=2)。
        output = (
            output.reshape(batch, 2, channels, time, frequency)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .reshape(batch, channels, time, frequency * 2)
        )
        return output, new_cache


class _StreamingDecoderBlock(nn.Module):
    def __init__(self, block: _DecoderBlock):
        super().__init__()
        if block.arch_version not in (2, 3):
            raise ValueError("StreamingLocalVQE only supports arch_version 2 or 3")
        self.skip_norm = block.skip_norm
        self.skip_conv = block.skip_conv
        self.resblock = _StreamingResidualBlock(block.resblock)
        self.deconv = _StreamingSubpixelConv2d(block.deconv)
        self.is_last = block.is_last
        if not self.is_last:
            self.act = block.act

    def forward(
        self,
        x: torch.Tensor,
        encoder: torch.Tensor,
        res_cache: torch.Tensor,
        deconv_cache: torch.Tensor,
    ):
        output = x + self.skip_conv(self.skip_norm(encoder))
        output, new_res_cache = self.resblock(output, res_cache)
        output, new_deconv_cache = self.deconv(output, deconv_cache)
        if not self.is_last:
            output = self.act(output)
        return output, new_res_cache, new_deconv_cache


class _StreamingAlignBlock(nn.Module):
    def __init__(self, block: _AlignBlock):
        super().__init__()
        self.in_channels = block.in_channels
        self.hidden_channels = block.hidden_channels
        self.delay = block.dmax
        self.pconv_mic = block.pconv_mic
        self.pconv_ref = block.pconv_ref
        self.attention_conv = _StreamingCausalConv(block.conv[1])
        self.register_buffer("temperature", block.temperature)

    def forward(
        self,
        mic: torch.Tensor,
        far: torch.Tensor,
        far_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_cache: torch.Tensor,
    ):
        batch, _, _, frequency = mic.shape
        query = self.pconv_mic(mic)
        key_current = self.pconv_ref(far)
        key_window = torch.cat([key_cache, key_current], dim=2)
        keys = key_window.reshape(batch, self.hidden_channels, self.delay, 1, frequency)
        keys = keys.permute(0, 1, 3, 2, 4).contiguous()
        values = torch.sum(query.unsqueeze(-2) * keys, dim=-1) / math.sqrt(frequency)
        values, new_value_cache = self.attention_conv(values, value_cache)
        attention = torch.softmax(values / self.temperature, dim=-1).unsqueeze(-1)

        far_window = torch.cat([far_cache, far], dim=2)
        far_window = far_window.reshape(batch, self.in_channels, self.delay, 1, frequency)
        far_window = far_window.permute(0, 1, 3, 2, 4).contiguous()
        aligned = torch.sum(far_window * attention, dim=-2)

        if self.delay > 1:
            new_far_cache = torch.cat([far_cache[:, :, 1:, :], far], dim=2)
            new_key_cache = torch.cat([key_cache[:, :, 1:, :], key_current], dim=2)
        else:
            new_far_cache = far_cache
            new_key_cache = key_cache
        return aligned, new_far_cache, new_value_cache, new_key_cache


class _StreamingS4DBottleneck(nn.Module):
    def __init__(self, block: _S4DBottleneck):
        super().__init__()
        self.input_size = block.input_size
        self.hidden_size = block.hidden_size
        self.input_proj = block.input_proj
        self.output_proj = block.output_proj
        self.A_log_rate = block.A_log_rate
        self.A_theta = block.A_theta
        self.B_real = block.B_real
        self.B_imag = block.B_imag
        self.C_real = block.C_real
        self.C_imag = block.C_imag
        self.D = block.D

    def forward(
        self,
        x: torch.Tensor,
        state_real: torch.Tensor,
        state_imag: torch.Tensor,
    ):
        batch, channels, time, frequency = x.shape
        if time != 1:
            raise ValueError(f"S4D streaming input must contain one frame, got T={time}")
        flattened = x.permute(0, 2, 1, 3).reshape(batch, 1, channels * frequency)
        projected = self.input_proj(flattened).squeeze(1).float()

        radius = torch.exp(-F.softplus(self.A_log_rate).clamp(min=0.01)).float()
        a_real = radius * torch.cos(self.A_theta.float())
        a_imag = radius * torch.sin(self.A_theta.float())
        new_real = (
            a_real * state_real - a_imag * state_imag + self.B_real.float() * projected
        )
        new_imag = (
            a_real * state_imag + a_imag * state_real + self.B_imag.float() * projected
        )
        hidden = self.C_real.float() * new_real - self.C_imag.float() * new_imag
        output = self.output_proj(hidden.unsqueeze(1)) + self.D * flattened
        output = output.reshape(batch, 1, channels, frequency).permute(0, 2, 1, 3)
        return output, new_real, new_imag


class _StreamingCCM(nn.Module):
    def __init__(self, block: _CCM):
        super().__init__()
        self.register_buffer("v_real", block.v_real)
        self.register_buffer("v_imag", block.v_imag)

    def forward(self, mask: torch.Tensor, mic: torch.Tensor, cache: torch.Tensor):
        batch, _, time, frequency = mask.shape
        basis = mask.reshape(batch, 3, 9, time, frequency)
        mask_real = torch.sum(
            self.v_real[None, :, None, None, None] * basis, dim=1
        ).reshape(batch, 3, 3, time, frequency)
        mask_imag = torch.sum(
            self.v_imag[None, :, None, None, None] * basis, dim=1
        ).reshape(batch, 3, 3, time, frequency)

        mic_current = mic.permute(0, 3, 2, 1).contiguous()
        mic_window = torch.cat([cache, mic_current], dim=2)
        unfolded = F.unfold(F.pad(mic_window, (1, 1, 0, 0)), kernel_size=(3, 3))
        unfolded = unfolded.reshape(batch, 2, 3, 3, time, frequency)
        enhanced_real = torch.sum(
            mask_real * unfolded[:, 0] - mask_imag * unfolded[:, 1], dim=(1, 2)
        )
        enhanced_imag = torch.sum(
            mask_real * unfolded[:, 1] + mask_imag * unfolded[:, 0], dim=(1, 2)
        )
        enhanced = torch.stack([enhanced_real, enhanced_imag], dim=-1)
        enhanced = enhanced.transpose(1, 2).contiguous()
        new_cache = torch.cat([cache[:, :, 1:, :], mic_current], dim=2)
        return enhanced, new_cache


class StreamingLocalVQE(nn.Module):
    """逐帧 LocalVQE；仅接受 FP32 ``(B, 257, 1, 2)`` 双输入。"""

    def __init__(self, base_model: LocalVQE, target_fs: int = 16000):
        super().__init__()
        if base_model.arch_version not in (2, 3):
            raise ValueError("StreamingLocalVQE only supports arch_version 2 or 3")
        if base_model.n_freqs != 257:
            raise ValueError(
                f"StreamingLocalVQE currently requires F=257 (n_fft=512), got F={base_model.n_freqs}"
            )
        self.n_fft = base_model.n_fft
        self.hop_length = base_model.hop_length
        self.win_length = base_model.win_length
        self.window = base_model.window
        self.center = base_model.center
        self.target_fs = int(target_fs)
        self.n_freqs = base_model.n_freqs
        self.arch_version = base_model.arch_version
        self.algorithmic_delay = 0
        self.fe_mic = base_model.fe_mic
        self.fe_ref = base_model.fe_ref

        self.mic_enc1 = _StreamingEncoderBlock(base_model.mic_enc1)
        self.mic_enc2 = _StreamingEncoderBlock(base_model.mic_enc2)
        self.mic_enc3 = _StreamingEncoderBlock(base_model.mic_enc3)
        self.mic_enc4 = _StreamingEncoderBlock(base_model.mic_enc4)
        self.mic_enc5 = _StreamingEncoderBlock(base_model.mic_enc5)
        self.far_enc1 = _StreamingEncoderBlock(base_model.far_enc1)
        self.far_enc2 = _StreamingEncoderBlock(base_model.far_enc2)
        self.align = _StreamingAlignBlock(base_model.align)
        self.bottleneck = _StreamingS4DBottleneck(base_model.bottleneck)
        self.dec5 = _StreamingDecoderBlock(base_model.dec5)
        self.dec4 = _StreamingDecoderBlock(base_model.dec4)
        self.dec3 = _StreamingDecoderBlock(base_model.dec3)
        self.dec2 = _StreamingDecoderBlock(base_model.dec2)
        self.dec1 = _StreamingDecoderBlock(base_model.dec1)
        self.ccm = _StreamingCCM(base_model.ccm)

        self._cache_cfg = self._make_cache_config(base_model)
        self.CACHE_KEYS = tuple(self._cache_cfg)

    @staticmethod
    def _history(conv: nn.Conv2d) -> int:
        return int(conv.kernel_size[0]) - 1

    def _make_cache_config(self, base: LocalVQE) -> Dict[str, Tuple[int, ...]]:
        kernel = tuple(int(value) for value in base.mic_enc1.conv.kernel_size)
        frequencies = _compute_freq_progression(base.n_freqs, kernel)
        cfg: Dict[str, Tuple[int, ...]] = {}

        def add_encoder(name: str, block: _EncoderBlock, input_frequency: int, output_frequency: int):
            cfg[f"{name}_conv"] = (
                block.conv.in_channels,
                self._history(block.conv),
                input_frequency,
            )
            cfg[f"{name}_res"] = (
                block.resblock.conv.in_channels,
                self._history(block.resblock.conv),
                output_frequency,
            )

        add_encoder("mic_enc1", base.mic_enc1, frequencies[0], frequencies[1])
        add_encoder("mic_enc2", base.mic_enc2, frequencies[1], frequencies[2])
        add_encoder("mic_enc3", base.mic_enc3, frequencies[2], frequencies[3])
        add_encoder("mic_enc4", base.mic_enc4, frequencies[3], frequencies[4])
        add_encoder("mic_enc5", base.mic_enc5, frequencies[4], frequencies[5])
        add_encoder("far_enc1", base.far_enc1, frequencies[0], frequencies[1])
        add_encoder("far_enc2", base.far_enc2, frequencies[1], frequencies[2])

        cfg["align_far"] = (base.align.in_channels, base.align.dmax - 1, frequencies[2])
        cfg["align_v"] = (
            base.align.hidden_channels,
            self._history(base.align.conv[1]),
            base.align.dmax,
        )
        cfg["align_k"] = (base.align.hidden_channels, base.align.dmax - 1, frequencies[2])
        cfg["s4d_real"] = (base.bottleneck.hidden_size,)
        cfg["s4d_imag"] = (base.bottleneck.hidden_size,)

        for name, block, frequency in (
            ("dec5", base.dec5, frequencies[5]),
            ("dec4", base.dec4, frequencies[4]),
            ("dec3", base.dec3, frequencies[3]),
            ("dec2", base.dec2, frequencies[2]),
            ("dec1", base.dec1, frequencies[1]),
        ):
            cfg[f"{name}_res"] = (
                block.resblock.conv.in_channels,
                self._history(block.resblock.conv),
                frequency,
            )
            cfg[f"{name}_deconv"] = (
                block.deconv.conv.in_channels,
                self._history(block.deconv.conv),
                frequency,
            )
        ccm_kernel = base.ccm.unfold[1].kernel_size
        cfg["ccm_mic"] = (2, int(ccm_kernel[0]) - 1, frequencies[0])
        return cfg

    def init_cache(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        if dtype != torch.float32:
            raise ValueError(f"StreamingLocalVQE cache dtype must be torch.float32, got {dtype}")
        return {
            name: torch.zeros((batch, *shape), device=device, dtype=torch.float32)
            for name, shape in self._cache_cfg.items()
        }

    def _validate_inputs(self, mic_spec: torch.Tensor, far_spec: torch.Tensor) -> None:
        expected = ("B", self.n_freqs, 1, 2)
        if mic_spec.ndim != 4 or mic_spec.shape[1:] != (self.n_freqs, 1, 2):
            raise ValueError(f"mic_spec must be {expected}, got {tuple(mic_spec.shape)}")
        if far_spec.ndim != 4 or far_spec.shape[1:] != (self.n_freqs, 1, 2):
            raise ValueError(f"far_spec must be {expected}, got {tuple(far_spec.shape)}")
        if far_spec.shape != mic_spec.shape:
            raise ValueError(
                f"far_spec shape must exactly match mic_spec, got {tuple(far_spec.shape)} and {tuple(mic_spec.shape)}"
            )
        if mic_spec.dtype != torch.float32 or far_spec.dtype != torch.float32:
            raise TypeError(
                f"StreamingLocalVQE only supports FP32 inputs, got {mic_spec.dtype} and {far_spec.dtype}"
            )

    def forward(
        self,
        mic_spec: torch.Tensor,
        far_spec: torch.Tensor,
        cache: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._validate_inputs(mic_spec, far_spec)

        mic = self.fe_mic(mic_spec)
        far = self.fe_ref(far_spec)
        mic1, mic1_conv, mic1_res = self.mic_enc1(
            mic, cache["mic_enc1_conv"], cache["mic_enc1_res"]
        )
        mic2, mic2_conv, mic2_res = self.mic_enc2(
            mic1, cache["mic_enc2_conv"], cache["mic_enc2_res"]
        )
        far1, far1_conv, far1_res = self.far_enc1(
            far, cache["far_enc1_conv"], cache["far_enc1_res"]
        )
        far2, far2_conv, far2_res = self.far_enc2(
            far1, cache["far_enc2_conv"], cache["far_enc2_res"]
        )
        aligned, align_far, align_v, align_k = self.align(
            mic2,
            far2,
            cache["align_far"],
            cache["align_v"],
            cache["align_k"],
        )
        mic3, mic3_conv, mic3_res = self.mic_enc3(
            torch.cat([mic2, aligned], dim=1),
            cache["mic_enc3_conv"],
            cache["mic_enc3_res"],
        )
        mic4, mic4_conv, mic4_res = self.mic_enc4(
            mic3, cache["mic_enc4_conv"], cache["mic_enc4_res"]
        )
        mic5, mic5_conv, mic5_res = self.mic_enc5(
            mic4, cache["mic_enc5_conv"], cache["mic_enc5_res"]
        )
        bottleneck, s4d_real, s4d_imag = self.bottleneck(
            mic5, cache["s4d_real"], cache["s4d_imag"]
        )
        dec5, dec5_res, dec5_deconv = self.dec5(
            bottleneck, mic5, cache["dec5_res"], cache["dec5_deconv"]
        )
        dec5 = dec5[..., : mic4.shape[-1]]
        dec4, dec4_res, dec4_deconv = self.dec4(
            dec5, mic4, cache["dec4_res"], cache["dec4_deconv"]
        )
        dec4 = dec4[..., : mic3.shape[-1]]
        dec3, dec3_res, dec3_deconv = self.dec3(
            dec4, mic3, cache["dec3_res"], cache["dec3_deconv"]
        )
        dec3 = dec3[..., : mic2.shape[-1]]
        dec2, dec2_res, dec2_deconv = self.dec2(
            dec3, mic2, cache["dec2_res"], cache["dec2_deconv"]
        )
        dec2 = dec2[..., : mic1.shape[-1]]
        mask, dec1_res, dec1_deconv = self.dec1(
            dec2, mic1, cache["dec1_res"], cache["dec1_deconv"]
        )
        mask = mask[..., : mic.shape[-1]]
        enhanced, ccm_mic = self.ccm(mask, mic_spec, cache["ccm_mic"])

        new_cache = {
            "mic_enc1_conv": mic1_conv,
            "mic_enc1_res": mic1_res,
            "mic_enc2_conv": mic2_conv,
            "mic_enc2_res": mic2_res,
            "mic_enc3_conv": mic3_conv,
            "mic_enc3_res": mic3_res,
            "mic_enc4_conv": mic4_conv,
            "mic_enc4_res": mic4_res,
            "mic_enc5_conv": mic5_conv,
            "mic_enc5_res": mic5_res,
            "far_enc1_conv": far1_conv,
            "far_enc1_res": far1_res,
            "far_enc2_conv": far2_conv,
            "far_enc2_res": far2_res,
            "align_far": align_far,
            "align_v": align_v,
            "align_k": align_k,
            "s4d_real": s4d_real,
            "s4d_imag": s4d_imag,
            "dec5_res": dec5_res,
            "dec5_deconv": dec5_deconv,
            "dec4_res": dec4_res,
            "dec4_deconv": dec4_deconv,
            "dec3_res": dec3_res,
            "dec3_deconv": dec3_deconv,
            "dec2_res": dec2_res,
            "dec2_deconv": dec2_deconv,
            "dec1_res": dec1_res,
            "dec1_deconv": dec1_deconv,
            "ccm_mic": ccm_mic,
        }
        return enhanced, new_cache


class StreamingLocalVQEFlat(nn.Module):
    """将 cache dict 展平为稳定的 ONNX 输入输出 ABI。"""

    def __init__(self, streaming_model: StreamingLocalVQE):
        super().__init__()
        self.inner = streaming_model
        self.n_fft = streaming_model.n_fft
        self.hop_length = streaming_model.hop_length
        self.win_length = getattr(streaming_model, "win_length", self.n_fft)
        self.window = getattr(streaming_model, "window", "hann")
        self.center = getattr(streaming_model, "center", True)
        self.target_fs = getattr(streaming_model, "target_fs", 16000)
        self.algorithmic_delay = streaming_model.algorithmic_delay
        self._keys = list(streaming_model.CACHE_KEYS)

    def init_cache_flat(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        cache = self.inner.init_cache(batch_size, torch.device(device), dtype)
        return [cache[name] for name in self._keys]

    def forward(
        self,
        mic_spec: torch.Tensor,
        far_spec: torch.Tensor,
        *cache_tensors: torch.Tensor,
    ):
        if len(cache_tensors) != len(self._keys):
            raise ValueError(
                f"StreamingLocalVQEFlat expects {len(self._keys)} cache tensors, got {len(cache_tensors)}"
            )
        cache = dict(zip(self._keys, cache_tensors))
        enhanced, new_cache = self.inner(mic_spec, far_spec, cache)
        return (enhanced,) + tuple(new_cache[name] for name in self._keys)

    def get_input_names(self) -> List[str]:
        return ["mic_spec", "far_spec", *self._keys]

    def get_output_names(self) -> List[str]:
        return ["enh_spec", *[f"out_{name}" for name in self._keys]]

    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {
            name: {0: "batch_size"}
            for name in self.get_input_names() + self.get_output_names()
        }

    def get_cache_config(self) -> Dict[str, List[int]]:
        cache = self.init_cache_flat(1, torch.device("cpu"), torch.float32)
        return {name: list(value.shape) for name, value in zip(self._keys, cache)}

    @staticmethod
    def load_cache_cfg_from_onnx(onnx_path: str) -> Dict[str, Tuple[int, ...]]:
        from onnx.utils import parse_onnx_cache_config, read_onnx_metadata

        parsed = parse_onnx_cache_config(
            read_onnx_metadata(onnx_path), "localvqe_cache_cfg"
        )
        return {name: tuple(shape) for name, shape in parsed.items()}
