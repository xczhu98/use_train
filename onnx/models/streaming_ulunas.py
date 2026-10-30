"""UL-UNAS 官方逐帧流式 ONNX wrapper。

实现忠实对应官方 ``ulunas_onnx/stream/ulunas_stream.py``：输入为单帧复谱
``[B, 257, 1, 2]``，三个扁平 cache 的顺序、尺寸和 ONNX I/O 名称均与官方一致。
"""

from __future__ import annotations

import json
from math import prod
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from tasks.se.models.discriminative.ulunas import DPGRNN, ULUNAS, XConvBlock, XDWSBlock, XMBBlocks


class StreamingULUNAS(nn.Module):
    """将默认 ULUNAS 转换为官方 ABI 的单帧流式执行路径。

    wrapper 仅复用已完成权重加载的 ``base_model``，不承担 checkpoint 加载职责；
    非官方默认 NAS 架构会被拒绝，避免将不匹配的 cache ABI 导出为可部署模型。
    """

    CONV_CACHE_SHAPES = [
        (1, 2, 129),
        (24, 1, 65),
        (24, 1, 33),
        (24, 1, 33),
        (12, 1, 33),
        (12, 2, 65),
    ]
    TFA_CACHE_HIDDEN = [24, 48, 48, 64, 32, 64, 48, 48, 24, 2]
    INTER_CACHE_SHAPES = [(33, 16), (33, 16)]
    CACHE_KEYS = ("conv_cache", "tfa_cache", "inter_cache")

    def __init__(self, base_model: ULUNAS):
        super().__init__()
        self._validate_official_default(base_model)
        self.n_fft = base_model.n_fft
        self.erb = base_model.erb
        self.encoder = base_model.encoder
        self.dpgrnn = base_model.dpgrnn
        self.decoder = base_model.decoder

    @classmethod
    def _validate_official_default(cls, base_model: ULUNAS) -> None:
        """拒绝无法保证官方 cache ABI 的模型结构。"""
        if not isinstance(base_model, ULUNAS):
            raise TypeError(f"base_model must be ULUNAS, got {type(base_model).__name__}")
        if base_model.n_fft != 512:
            raise ValueError(f"Official streaming ULUNAS requires n_fft=512, got {base_model.n_fft}")

        enc = list(base_model.encoder.en_convs)
        dec = list(base_model.decoder.de_convs)
        expected_types = [XConvBlock, XMBBlocks, XDWSBlock, XMBBlocks, XDWSBlock]
        if len(enc) != 5 or len(dec) != 5 or [type(x) for x in enc] != expected_types:
            raise ValueError("Official streaming ULUNAS requires the default five-block encoder NAS structure")
        if [type(x) for x in dec] != [XDWSBlock, XMBBlocks, XDWSBlock, XMBBlocks, XConvBlock]:
            raise ValueError("Official streaming ULUNAS requires the default five-block decoder NAS structure")
        if len(base_model.dpgrnn) != 2 or not all(isinstance(x, DPGRNN) for x in base_model.dpgrnn):
            raise ValueError("Official streaming ULUNAS requires exactly two DPGRNN blocks")
        if [x.input_size for x in base_model.dpgrnn] != [16, 16] or [x.width for x in base_model.dpgrnn] != [33, 33]:
            raise ValueError("Official streaming ULUNAS requires the default DPGRNN channels and width")

    @classmethod
    def init_cache(
        cls,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        """按官方扁平 ABI 创建零 cache，batch 位始终是 axis 0。"""
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        conv_size = sum(prod(shape) for shape in cls.CONV_CACHE_SHAPES)
        tfa_size = sum(cls.TFA_CACHE_HIDDEN)
        inter_size = sum(prod(shape) for shape in cls.INTER_CACHE_SHAPES)
        return {
            "conv_cache": torch.zeros(batch_size, conv_size, device=device, dtype=dtype),
            "tfa_cache": torch.zeros(batch_size, tfa_size, device=device, dtype=dtype),
            "inter_cache": torch.zeros(batch_size, inter_size, device=device, dtype=dtype),
        }

    @classmethod
    def _unpack_conv_cache(cls, cache: Tensor) -> list[Tensor]:
        batch_size = cache.shape[0]
        tensors: list[Tensor] = []
        offset = 0
        for shape in cls.CONV_CACHE_SHAPES:
            size = prod(shape)
            tensors.append(cache[:, offset : offset + size].reshape(batch_size, *shape))
            offset += size
        return tensors

    @staticmethod
    def _pack_conv_cache(caches: list[Tensor]) -> Tensor:
        return torch.cat([cache.reshape(cache.shape[0], -1) for cache in caches], dim=1)

    @classmethod
    def _unpack_tfa_cache(cls, cache: Tensor) -> list[Tensor]:
        batch_size = cache.shape[0]
        tensors: list[Tensor] = []
        offset = 0
        for hidden in cls.TFA_CACHE_HIDDEN:
            tensors.append(cache[:, offset : offset + hidden].reshape(1, batch_size, hidden))
            offset += hidden
        return tensors

    @staticmethod
    def _pack_tfa_cache(caches: list[Tensor]) -> Tensor:
        return torch.cat([cache.reshape(cache.shape[1], -1) for cache in caches], dim=1)

    @classmethod
    def _unpack_inter_cache(cls, cache: Tensor) -> list[Tensor]:
        batch_size = cache.shape[0]
        tensors: list[Tensor] = []
        offset = 0
        for shape in cls.INTER_CACHE_SHAPES:
            size = prod(shape)
            tensors.append(cache[:, offset : offset + size].reshape(batch_size, *shape))
            offset += size
        return tensors

    @staticmethod
    def _pack_inter_cache(caches: list[Tensor]) -> Tensor:
        return torch.cat([cache.reshape(cache.shape[0], -1) for cache in caches], dim=1)

    @staticmethod
    def _stream_temporal_conv(x: Tensor, conv: nn.Module, cache: Tensor) -> tuple[Tensor, Tensor]:
        """执行官方因果 temporal cache 卷积，转置卷积仅保留当前帧。"""
        inp = torch.cat([cache, x], dim=2)
        if isinstance(conv, nn.ConvTranspose2d):
            kernel_time = conv.kernel_size[0]
            inp_padded = torch.nn.functional.pad(inp, (0, 0, kernel_time - 1, 0))
            output = conv(inp_padded)[:, :, -1:, :]
        else:
            output = conv(inp)
        return output, inp[:, :, 1:, :]

    @staticmethod
    def _stream_ctfa(ctfa: nn.Module, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """显式推进 cTFA 的单向时间 GRU hidden。"""
        zt = torch.mean(x.pow(2), dim=-1)
        at, hidden = ctfa.ta_gru(zt.transpose(1, 2), hidden)
        at = torch.sigmoid(ctfa.ta_fc(at).transpose(1, 2))
        af = torch.sigmoid(ctfa.fa(x))
        return at[..., None] * x * af[:, None], hidden

    def _stream_xconv(
        self, block: XConvBlock, x: Tensor, conv_cache: Tensor, tfa_cache: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        x, conv_cache = self._stream_temporal_conv(x, block.ops[1], conv_cache)
        x = block.ops[2](x)
        x = block.ops[3](x)
        x, tfa_cache = self._stream_ctfa(block.ops[4], x, tfa_cache)
        return block.ops[5](x), conv_cache, tfa_cache

    def _stream_xdws(
        self, block: XDWSBlock, x: Tensor, tfa_cache: Tensor, conv_cache: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        x = block.pconv(x)
        if conv_cache is None:
            x = block.dconv[0](x)
            x = block.dconv[1](x)
        else:
            x, conv_cache = self._stream_temporal_conv(x, block.dconv[1], conv_cache)
        x = block.dconv[2](x)
        x = block.dconv[3](x)
        x, tfa_cache = self._stream_ctfa(block.dconv[4], x, tfa_cache)
        return x, conv_cache, tfa_cache

    def _stream_xmb(
        self, block: XMBBlocks, x: Tensor, tfa_cache: Tensor, conv_cache: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        residual = x
        x = block.pconv1(x)
        if conv_cache is None:
            x = block.dconv(x)
        else:
            x, conv_cache = self._stream_temporal_conv(x, block.dconv[1], conv_cache)
            x = block.dconv[2](x)
            x = block.dconv[3](x)
        x = block.pconv2[0](x)
        x = block.pconv2[1](x)
        x, tfa_cache = self._stream_ctfa(block.pconv2[2], x, tfa_cache)
        if x.shape == residual.shape:
            x = x + residual
        return block.shuffle(x), conv_cache, tfa_cache

    @staticmethod
    def _stream_dpgrnn(block: DPGRNN, x: Tensor, inter_cache: Tensor) -> tuple[Tensor, Tensor]:
        """显式推进官方 DPGRNN inter（时间）GRU hidden。"""
        x = x.permute(0, 2, 3, 1)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        intra_x = block.intra_rnn(intra_x)[0]
        intra_x = block.intra_fc(intra_x)
        intra_x = intra_x.reshape(x.shape[0], -1, block.width, block.input_size)
        intra_x = block.intra_ln(intra_x)
        intra_out = x + intra_x

        x = intra_out.permute(0, 2, 1, 3)
        batch_size = x.shape[0]
        inter_x = x.reshape(batch_size * x.shape[1], x.shape[2], x.shape[3])
        # 官方 B=1 cache 为 (33, 16)；flat ABI 的 batch 位在 axis 0，
        # 因此 B>1 时将每个样本的 33 个频带 state 展开为 GRU batch 维。
        inter_hidden = inter_cache.reshape(1, batch_size * block.width, block.input_size)
        inter_x, inter_hidden = block.inter_rnn(inter_x, inter_hidden)
        inter_cache = inter_hidden.reshape(batch_size, block.width, block.input_size)
        inter_x = block.inter_fc(inter_x)
        inter_x = inter_x.reshape(batch_size, block.width, -1, block.input_size).permute(0, 2, 1, 3)
        inter_x = block.inter_ln(inter_x)
        return (intra_out + inter_x).permute(0, 3, 1, 2), inter_cache

    def forward(self, spec: Tensor, cache: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """处理一帧官方 ABI 复谱，返回增强谱和更新后的 dict cache。"""
        if spec.ndim != 4 or tuple(spec.shape[1:]) != (257, 1, 2):
            raise ValueError(f"spec must have shape [B, 257, 1, 2], got {tuple(spec.shape)}")
        missing = set(self.CACHE_KEYS).difference(cache)
        if missing:
            raise ValueError(f"cache missing keys: {sorted(missing)}")
        batch_size = spec.shape[0]
        expected = self.init_cache(batch_size, spec.device, spec.dtype)
        for key in self.CACHE_KEYS:
            if tuple(cache[key].shape) != tuple(expected[key].shape):
                raise ValueError(f"{key} must have shape {tuple(expected[key].shape)}, got {tuple(cache[key].shape)}")

        spec_2ch = spec.permute(0, 3, 2, 1)
        feat = torch.log10(torch.linalg.vector_norm(spec_2ch, dim=1, keepdim=True).clamp(1e-12))
        feat = self.erb.bm(feat)

        conv_e0, conv_e1, conv_e2, conv_d2, conv_d3, conv_d4 = self._unpack_conv_cache(cache["conv_cache"])
        tfa = self._unpack_tfa_cache(cache["tfa_cache"])
        inter_0, inter_1 = self._unpack_inter_cache(cache["inter_cache"])

        enc_outs: list[Tensor] = []
        feat, conv_e0, tfa[0] = self._stream_xconv(self.encoder.en_convs[0], feat, conv_e0, tfa[0])
        enc_outs.append(feat)
        feat, conv_e1, tfa[1] = self._stream_xmb(self.encoder.en_convs[1], feat, tfa[1], conv_e1)
        enc_outs.append(feat)
        feat, conv_e2, tfa[2] = self._stream_xdws(self.encoder.en_convs[2], feat, tfa[2], conv_e2)
        enc_outs.append(feat)
        feat, _, tfa[3] = self._stream_xmb(self.encoder.en_convs[3], feat, tfa[3])
        enc_outs.append(feat)
        feat, _, tfa[4] = self._stream_xdws(self.encoder.en_convs[4], feat, tfa[4])
        enc_outs.append(feat)

        feat, inter_0 = self._stream_dpgrnn(self.dpgrnn[0], feat, inter_0)
        feat, inter_1 = self._stream_dpgrnn(self.dpgrnn[1], feat, inter_1)

        feat, _, tfa[5] = self._stream_xdws(self.decoder.de_convs[0], feat + enc_outs[4], tfa[5])
        feat, _, tfa[6] = self._stream_xmb(self.decoder.de_convs[1], feat + enc_outs[3], tfa[6])
        feat, conv_d2, tfa[7] = self._stream_xdws(self.decoder.de_convs[2], feat + enc_outs[2], tfa[7], conv_d2)
        feat, conv_d3, tfa[8] = self._stream_xmb(self.decoder.de_convs[3], feat + enc_outs[1], tfa[8], conv_d3)
        feat, conv_d4, tfa[9] = self._stream_xconv(self.decoder.de_convs[4], feat + enc_outs[0], conv_d4, tfa[9])

        mask = self.erb.bs(torch.sigmoid(feat))
        enhanced = (spec_2ch * mask).permute(0, 3, 2, 1)
        new_cache = {
            "conv_cache": self._pack_conv_cache([conv_e0, conv_e1, conv_e2, conv_d2, conv_d3, conv_d4]),
            "tfa_cache": self._pack_tfa_cache(tfa),
            "inter_cache": self._pack_inter_cache([inter_0, inter_1]),
        }
        return enhanced, new_cache


class StreamingULUNASFlat(nn.Module):
    """官方 ONNX flat ABI：mix/cache 输入，enh/cache_out 输出。"""

    INPUT_NAMES = ["mix", "conv_cache", "tfa_cache", "inter_cache"]
    OUTPUT_NAMES = ["enh", "conv_cache_out", "tfa_cache_out", "inter_cache_out"]

    def __init__(self, streaming_model: StreamingULUNAS):
        super().__init__()
        self.inner = streaming_model

    def forward(self, mix: Tensor, conv_cache: Tensor, tfa_cache: Tensor, inter_cache: Tensor) -> tuple[Tensor, ...]:
        enhanced, new_cache = self.inner(
            mix,
            {"conv_cache": conv_cache, "tfa_cache": tfa_cache, "inter_cache": inter_cache},
        )
        return enhanced, new_cache["conv_cache"], new_cache["tfa_cache"], new_cache["inter_cache"]

    def init_cache_flat(self, batch_size: int, device: torch.device | str, dtype: torch.dtype) -> list[Tensor]:
        cache = self.inner.init_cache(batch_size, device, dtype)
        return [cache[key] for key in self.inner.CACHE_KEYS]

    @classmethod
    def get_input_names(cls) -> list[str]:
        return list(cls.INPUT_NAMES)

    @classmethod
    def get_output_names(cls) -> list[str]:
        return list(cls.OUTPUT_NAMES)

    @classmethod
    def get_dynamic_axes(cls) -> dict[str, dict[int, str]]:
        return {name: {0: "batch"} for name in cls.INPUT_NAMES + cls.OUTPUT_NAMES}

    @staticmethod
    def load_cache_cfg_from_onnx(onnx_path: str | Path) -> dict[str, list[int]]:
        """读取导出端写入的 ``ulunas_cache_cfg`` metadata。"""
        from onnx.utils import read_onnx_metadata

        metadata = read_onnx_metadata(onnx_path)
        if "ulunas_cache_cfg" in metadata:
            return {
                key: list(shape)
                for key, shape in json.loads(metadata["ulunas_cache_cfg"]).items()
            }
        raise ValueError(f"ONNX {onnx_path} missing 'ulunas_cache_cfg' metadata.")
