"""OnlineSpatialNet 的共享因果骨干（SE/SS 任务模型复用）。

移植自 NBSS ``models/arch/OnlineSpatialNet.py`` 与 ``models/arch/base/retention.py``，
保留上游 ``attention`` 开关语义：

  - ``mhsa(scope)``            有限窗口因果 MHSA，逐帧增量 K/V cache；
  - ``ret(factor[,share_qk])`` 多尺度 Retention，显式 recurrent 状态；
  - ``mamba(d_state,d_conv)``  selective SSM，同时替换窄带 MHSA 与 T-ConvFFN。

本模块只负责 ``[B, C, F, T, 2]`` 谱张量到 ``[B, F, T, output_dim]`` 的映射与 cache
生命周期；任务级 IO 合同、几何参数与 STFT 边界由 ``tasks.se`` / ``tasks.ss`` 的模型负责。
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tasks.se.models.modules.online_spatialnet_impl.mamba import CausalMamba, mamba_available
from tasks.se.models.modules.online_spatialnet_impl.retention import MultiScaleRetention, RetNetRelPos


class _GroupLinear(nn.Module):
    def __init__(self, features: int, groups: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, features, features))
        self.bias = nn.Parameter(torch.zeros(groups, features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("...gf,gkf->...gk", x, self.weight) + self.bias


class _SeqLastNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, -1)).transpose(1, -1)


class _ChannelGroupNorm(nn.Module):
    """分组归一化；``seq_last=True`` 时按时间帧独立统计，避免跨帧池化破坏因果性。"""

    def __init__(self, groups: int, channels: int, seq_last: bool) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.seq_last = seq_last

    def forward(self, x: Tensor) -> Tensor:
        if not self.seq_last:
            return self.norm(x.transpose(1, -1)).transpose(1, -1)
        batch, channels = x.shape[0], x.shape[1]
        value = x.transpose(1, -1).reshape(-1, channels, 1)
        value = self.norm(value).reshape(batch, -1, channels).transpose(1, -1)
        return value


class _CausalTemporalConv(nn.Module):
    """无前瞻的因果时域卷积；整段前向左 padding，逐帧使用显式 history。"""

    def __init__(self, channels: int, kernel_size: int, groups: int) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("causal temporal kernel_size must be a positive integer")
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=groups)
        self.history = kernel_size - 1

    def forward_sequence(self, x: Tensor) -> Tensor:
        if not self.history:
            return self.conv(x)
        return self.conv(F.pad(x, (self.history, 0)))

    def forward(self, x: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
        if history.shape[:-1] != x.shape[:-1] or history.shape[-1] != self.history:
            raise ValueError("causal temporal convolution history shape mismatch")
        if not self.history:
            return self.conv(x), x[..., :0]
        padded = torch.cat((history, x), dim=-1)
        return self.conv(padded), padded[..., -self.history :]


def _window_causal_mask(scope: int, length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """整段前向使用的窗口因果 mask：只允许当前帧与最近 ``scope - 1`` 帧。

    屏蔽值使用 ``finfo(dtype).min`` 而不是 ``-inf``：长序列下注意力 kernel 按块做在线
    softmax，整块被 ``-inf`` 屏蔽时会出现 ``-inf - (-inf)`` 而产生 NaN（实测 T≥1000 时
    ``-inf`` 版本输出 NaN，``finfo.min`` 版本保持有限）。
    """
    index = torch.arange(length, device=device)
    distance = index[:, None] - index[None, :]
    mask = torch.zeros(length, length, device=device, dtype=dtype)
    return mask.masked_fill((distance < 0) | (distance >= scope), torch.finfo(dtype).min)


class _IncrementalAttention(nn.Module):
    """有限窗口因果 MHSA：逐帧只计算新 query，对缓存 K/V 做精确注意力。"""

    def __init__(self, hidden: int, heads: int, scope: int) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden dimension must divide attention heads")
        self.scope = scope
        self.heads = heads
        self.head_dim = hidden // heads
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)

    @property
    def embed_dim(self) -> int:
        return self.attention.embed_dim

    def cache_names(self, index: int) -> tuple[str, ...]:
        return (f"layer{index}_key", f"layer{index}_value")

    def init_cache(
        self, batch: int, freqs: int, device: torch.device, dtype: torch.dtype, index: int
    ) -> OrderedDict[str, Tensor]:
        shape = (batch, freqs, self.heads, self.scope, self.head_dim)
        key_name, value_name = self.cache_names(index)
        return OrderedDict(
            (
                (key_name, torch.zeros(shape, device=device, dtype=dtype)),
                (value_name, torch.zeros(shape, device=device, dtype=dtype)),
            )
        )

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        weight = self.attention.in_proj_weight
        bias = self.attention.in_proj_bias
        qkv = F.linear(x, weight, bias)
        query, key, value = qkv.chunk(3, dim=-1)
        batch, freqs, frames, _ = query.shape

        def _heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(batch, freqs, frames, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)

        return _heads(query), _heads(key), _heads(value)

    def forward_sequence(self, x: Tensor) -> Tensor:
        batch, freqs, frames, hidden = x.shape
        flat = x.reshape(batch * freqs, frames, hidden)
        mask = _window_causal_mask(self.scope, frames, x.device, x.dtype)
        output, _ = self.attention(flat, flat, flat, need_weights=False, attn_mask=mask)
        return output.reshape(batch, freqs, frames, hidden)

    def _warmup_mask(self, batch: int, frame_index: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
        """首 ``scope - 1`` 帧只允许看到已产生帧；缓存右对齐，早于 window 的槽位被屏蔽。"""
        window = torch.clamp(frame_index.reshape(-1, 1) + 1.0, max=float(self.scope))
        positions = torch.arange(self.scope, device=device, dtype=dtype).reshape(1, -1)
        valid = positions >= (self.scope - window)
        mask = torch.zeros(batch, 1, 1, 1, self.scope, device=device, dtype=dtype)
        return mask.masked_fill(~valid.reshape(batch, 1, 1, 1, self.scope), float("-inf"))

    def forward_frame(
        self, x: Tensor, keys: Tensor, values: Tensor, frame_index: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, freqs, _, hidden = x.shape
        query, key, value = self._project(x)
        keys = torch.cat((keys, key), dim=3)[..., -self.scope :, :]
        values = torch.cat((values, value), dim=3)[..., -self.scope :, :]
        mask = self._warmup_mask(batch, frame_index, x.device, x.dtype)
        attended = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
        attended = attended.permute(0, 1, 3, 2, 4).reshape(batch, freqs, 1, hidden)
        return self.attention.out_proj(attended), keys, values


class _RetentionBranch(nn.Module):
    """多尺度 Retention：整段走 chunkwise，逐帧走显式 recurrent 状态。"""

    def __init__(self, hidden: int, heads: int, value_factor: int, share_qk: bool, decay, rope: bool) -> None:
        super().__init__()
        self.retention = MultiScaleRetention(
            embed_dim=hidden, num_heads=heads, value_factor=value_factor, share_qk=share_qk
        )
        self.rel_pos = RetNetRelPos(embed_dim=hidden, num_heads=heads, decay=decay)
        self.rope = rope
        self.heads = heads

    def cache_names(self, index: int) -> tuple[str, ...]:
        return (f"layer{index}_ret_kv", f"layer{index}_ret_scale")

    def init_cache(
        self, batch: int, freqs: int, device: torch.device, dtype: torch.dtype, index: int
    ) -> OrderedDict[str, Tensor]:
        kv_name, scale_name = self.cache_names(index)
        return OrderedDict(
            (
                (
                    kv_name,
                    torch.zeros(
                        batch * freqs,
                        self.heads,
                        self.retention.head_dim,
                        self.retention.key_dim,
                        device=device,
                        dtype=dtype,
                    ),
                ),
                (scale_name, torch.ones(batch * freqs, self.heads, 1, 1, device=device, dtype=dtype)),
            )
        )

    def forward_sequence(self, x: Tensor) -> Tensor:
        """整段前向与流式共用同一 recurrent 递推，保证训练/推理数值一致。"""
        batch, freqs, frames, hidden = x.shape
        state = self.retention.init_incremental_state(batch * freqs, x.device, x.dtype)
        outputs = []
        for index in range(frames):
            frame_index = torch.full((batch, 1), float(index), device=x.device, dtype=x.dtype)
            output, state["prev_key_value"], state["scale"] = self.forward_frame(
                x[:, :, index : index + 1], state["prev_key_value"], state["scale"], frame_index
            )
            outputs.append(output)
        if not outputs:
            raise ValueError("retention branch requires at least one frame")
        return torch.cat(outputs, dim=2)

    def forward_frame(
        self, x: Tensor, kv: Tensor, scale: Tensor, frame_index: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, freqs, _, hidden = x.shape
        flat = x.reshape(batch * freqs, 1, hidden)
        if self.rope:
            if frame_index is None:
                raise ValueError("rope retention branch requires the frame index counter")
            rel_pos = self.rel_pos.recurrent(frame_index.reshape(-1)[0])
        else:
            rel_pos = self.rel_pos.recurrent()
        state = {"prev_key_value": kv, "scale": scale}
        output = self.retention.forward(flat, rel_pos=rel_pos, incremental_state=state, rope=self.rope)
        return output.reshape(batch, freqs, 1, hidden), state["prev_key_value"], state["scale"]


class _MambaBranch(nn.Module):
    """窄带 selective SSM：整段走并行/参考扫描，逐帧走 tensor-only state。"""

    def __init__(self, hidden: int, d_state: int, d_conv: int) -> None:
        super().__init__()
        if not mamba_available():
            raise ImportError(
                "OnlineSpatialNet attention='mamba(...)' requires optional dependency mamba_ssm"
            )
        self.mixer = CausalMamba(hidden, d_state, d_conv)

    def cache_names(self, index: int, tag: str) -> tuple[str, ...]:
        return (f"layer{index}_{tag}_conv", f"layer{index}_{tag}_ssm")

    def init_cache(
        self, batch: int, freqs: int, device: torch.device, dtype: torch.dtype, index: int, tag: str
    ) -> OrderedDict[str, Tensor]:
        conv_name, ssm_name = self.cache_names(index, tag)
        return OrderedDict(
            (
                (
                    conv_name,
                    torch.zeros(batch * freqs, self.mixer.inner_dim, self.mixer.d_conv, device=device, dtype=dtype),
                ),
                (
                    ssm_name,
                    torch.zeros(batch * freqs, self.mixer.inner_dim, self.mixer.d_state, device=device, dtype=dtype),
                ),
            )
        )

    def forward_sequence(self, x: Tensor) -> Tensor:
        batch, freqs, frames, hidden = x.shape
        output = self.mixer.forward(x.reshape(batch * freqs, frames, hidden))
        return output.reshape(batch, freqs, frames, hidden)

    def forward_frame(self, x: Tensor, conv_state: Tensor, ssm_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, freqs, _, hidden = x.shape
        output, conv_state, ssm_state = self.mixer.step(x.reshape(batch * freqs, 1, hidden), conv_state, ssm_state)
        return output.reshape(batch, freqs, 1, hidden), conv_state, ssm_state


class _CausalLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        ffn: int,
        squeeze: int,
        freqs: int,
        heads: int,
        attention_branch: str,
        attention_scope: int | None,
        kernel_size: Sequence[int],
        conv_groups: Sequence[int],
        norms: Sequence[str],
        full: nn.Module | None,
        retention_args: Mapping[str, object],
        mamba_args: Mapping[str, object],
    ) -> None:
        super().__init__()
        fk, tk = (int(value) for value in kernel_size)
        fg, tg = (int(value) for value in conv_groups)
        if hidden % fg or ffn % tg or hidden % heads:
            raise ValueError("hidden/ffn dimensions must divide groups and attention heads")
        self.attention_branch = attention_branch
        self.attention_scope = attention_scope
        self.fconv1 = nn.Sequential(
            self._norm(norms[3], hidden, fg, True),
            nn.Conv1d(hidden, hidden, fk, padding="same", groups=fg),
            nn.PReLU(hidden),
        )
        self.norm_full = self._norm(norms[5], hidden, fg, False)
        self.squeeze = nn.Sequential(nn.Conv1d(hidden, squeeze, 1), nn.SiLU())
        self.full = full if full is not None else _GroupLinear(freqs, squeeze)
        self.unsqueeze = nn.Sequential(nn.Conv1d(squeeze, hidden, 1), nn.SiLU())
        self.fconv2 = nn.Sequential(
            self._norm(norms[4], hidden, fg, True),
            nn.Conv1d(hidden, hidden, fk, padding="same", groups=fg),
            nn.PReLU(hidden),
        )
        self.norm_attention = self._norm(norms[0], hidden, tg, False)
        if attention_branch == "msa":
            self.narrow: nn.Module = _IncrementalAttention(hidden, heads, int(attention_scope))
        elif attention_branch == "ret":
            self.narrow = _RetentionBranch(hidden, heads, **retention_args)  # type: ignore[arg-type]
        else:
            self.narrow = _MambaBranch(hidden, **mamba_args)  # type: ignore[arg-type]

        if attention_branch == "mamba":
            self.norm_ffn = self._norm(norms[1], hidden, tg, False)
            self.ffn_mamba: nn.Module | None = _MambaBranch(hidden, **mamba_args)  # type: ignore[arg-type]
            self.temporal = None
        else:
            self.norm_ffn = self._norm(norms[1], hidden, tg, True)
            self.ffn_in = nn.Sequential(nn.Conv1d(hidden, ffn, 1), nn.SiLU())
            self.temporal = nn.ModuleList([_CausalTemporalConv(ffn, tk, tg) for _ in range(3)])
            self.ffn_norm = self._norm(norms[2], ffn, tg, True)
            self.ffn_out = nn.Conv1d(ffn, hidden, 1)
            self.ffn_mamba = None

    @staticmethod
    def _norm(kind: str, channels: int, groups: int, seq_last: bool) -> nn.Module:
        if kind.upper() == "LN":
            return _SeqLastNorm(channels) if seq_last else nn.LayerNorm(channels)
        if kind.upper() == "GN":
            return _ChannelGroupNorm(groups, channels, seq_last)
        raise ValueError(f"unsupported OnlineSpatialNet normalization: {kind!r}")

    @staticmethod
    def _frequency_conv(block: nn.Module, x: Tensor) -> Tensor:
        batch, freqs, frames, hidden = x.shape
        value = x.permute(0, 2, 3, 1).reshape(batch * frames, hidden, freqs)
        value = block(value)
        return value.reshape(batch, frames, hidden, freqs).permute(0, 3, 1, 2)

    def _full(self, x: Tensor) -> Tensor:
        batch, freqs, frames, hidden = x.shape
        value = self.norm_full(x).permute(0, 2, 3, 1).reshape(batch * frames, hidden, freqs)
        value = self.unsqueeze(self.full(self.squeeze(value)))
        return value.reshape(batch, frames, hidden, freqs).permute(0, 3, 1, 2)

    def cache_names(self, index: int) -> tuple[str, ...]:
        if self.attention_branch == "mamba":
            return (*self.narrow.cache_names(index, "attn"), *self.ffn_mamba.cache_names(index, "ffn"))
        return (
            *self.narrow.cache_names(index),
            *(f"layer{index}_temporal{conv_index}" for conv_index in range(3)),
        )

    def init_cache(
        self, batch: int, freqs: int, device: torch.device, dtype: torch.dtype, index: int
    ) -> OrderedDict[str, Tensor]:
        if self.attention_branch == "mamba":
            cache = OrderedDict(self.narrow.init_cache(batch, freqs, device, dtype, index, "attn"))
            cache.update(self.ffn_mamba.init_cache(batch, freqs, device, dtype, index, "ffn"))
        else:
            cache = OrderedDict(self.narrow.init_cache(batch, freqs, device, dtype, index))
            for conv_index, conv in enumerate(self.temporal):
                cache[f"layer{index}_temporal{conv_index}"] = torch.zeros(
                    batch, freqs, self.ffn_in[0].out_channels, conv.history, device=device, dtype=dtype
                )
        return cache

    def cache_shape(self, name: str, batch: int, freqs: int, index: int) -> tuple[int, ...]:
        """按 cache 名称派生期望形状；初始化、校验与 ONNX ABI 共用同一推导。"""
        if name not in self.cache_names(index):
            raise ValueError(f"unknown OnlineSpatialNet cache name {name!r}")
        if self.attention_branch == "mamba":
            mixer = self.narrow.mixer
            if name.endswith("_conv"):
                return (batch * freqs, mixer.inner_dim, mixer.d_conv)
            return (batch * freqs, mixer.inner_dim, mixer.d_state)
        if name in self.narrow.cache_names(index):
            if self.attention_branch == "msa":
                return (batch, freqs, self.narrow.heads, self.narrow.scope, self.narrow.head_dim)
            retention = self.narrow.retention
            if name.endswith("_scale"):
                return (batch * freqs, self.narrow.heads, 1, 1)
            return (batch * freqs, self.narrow.heads, retention.head_dim, retention.key_dim)
        conv_index = int(name.rsplit("temporal", 1)[-1])
        conv = self.temporal[conv_index]
        return (batch, freqs, self.ffn_in[0].out_channels, conv.history)

    def forward_sequence(self, x: Tensor) -> Tensor:
        x = x + self._frequency_conv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._frequency_conv(self.fconv2, x)
        x = x + self.narrow.forward_sequence(self.norm_attention(x))
        if self.attention_branch == "mamba":
            return x + self.ffn_mamba.forward_sequence(self.norm_ffn(x))
        batch, freqs, frames, hidden = x.shape
        value = self.norm_ffn(x.transpose(-1, -2).reshape(batch * freqs, hidden, frames))
        value = self.ffn_in(value)
        for conv_index, conv in enumerate(self.temporal):
            value = conv.forward_sequence(value)
            if conv_index == 1:
                value = self.ffn_norm(value)
            value = F.silu(value)
        value = self.ffn_out(value).reshape(batch, freqs, hidden, frames).transpose(-1, -2)
        return x + value

    def forward_frame(self, x: Tensor, cache: Mapping[str, Tensor], index: int, frame_index: Tensor):
        names = self.cache_names(index)
        x = x + self._frequency_conv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._frequency_conv(self.fconv2, x)

        batch, freqs, frames, hidden = x.shape
        if frames != 1:
            raise ValueError("causal OnlineSpatialNet backend processes one frame per step")
        next_cache = OrderedDict(cache)
        normalised = self.norm_attention(x)
        if self.attention_branch == "msa":
            attended, key, value = self.narrow.forward_frame(
                normalised, cache[names[0]], cache[names[1]], frame_index
            )
            next_cache[names[0]] = key
            next_cache[names[1]] = value
        elif self.attention_branch == "ret":
            attended, kv, scale = self.narrow.forward_frame(
                normalised, cache[names[0]], cache[names[1]], frame_index
            )
            next_cache[names[0]] = kv
            next_cache[names[1]] = scale
        else:
            attended, conv_state, ssm_state = self.narrow.forward_frame(
                normalised, cache[names[0]], cache[names[1]]
            )
            next_cache[names[0]] = conv_state
            next_cache[names[1]] = ssm_state
        x = x + attended

        if self.attention_branch == "mamba":
            ffn_out, conv_state, ssm_state = self.ffn_mamba.forward_frame(
                self.norm_ffn(x), cache[names[2]], cache[names[3]]
            )
            next_cache[names[2]] = conv_state
            next_cache[names[3]] = ssm_state
            return x + ffn_out, next_cache

        value = self.ffn_in(self.norm_ffn(x.transpose(-1, -2).reshape(batch * freqs, hidden, 1)))
        for conv_index, (cache_name, conv) in enumerate(zip(names[2:], self.temporal)):
            cache_value = next_cache[cache_name]
            if conv.history:
                history = cache_value.reshape(batch * freqs, value.shape[1], conv.history)
                value, history = conv(value, history)
                next_cache[cache_name] = history.reshape(batch, freqs, value.shape[1], conv.history)
            else:
                value, _ = conv(value, value[..., :0])
            if conv_index == 1:
                value = self.ffn_norm(value)
            value = F.silu(value)
        value = self.ffn_out(value).reshape(batch, freqs, hidden, 1).transpose(-1, -2)
        return x + value, next_cache


def resolve_attention(
    attention: str | None,
    attention_scope: int | None,
    num_heads: int,
    decay,
    rope: bool,
) -> dict:
    """解析上游语义的 ``attention`` 字符串，返回分支参数。"""
    if attention is None:
        attention = f"mhsa({attention_scope or 251})"
    if not isinstance(attention, str) or not attention.strip():
        raise TypeError("OnlineSpatialNet attention must be a non-empty string")
    text = attention.strip().lower()
    if type(rope) is not bool:
        raise TypeError("OnlineSpatialNet rope must be an exact bool")
    if type(attention_scope) is not int and attention_scope is not None:
        raise TypeError("OnlineSpatialNet attention_scope must be an int or None")

    if text.startswith("mhsa(") and text.endswith(")"):
        body = text[5:-1]
        if body == "inf":
            raise ValueError("OnlineSpatialNet causal cache requires finite mhsa attention scope")
        if not body.isdigit() or int(body) < 1:
            raise ValueError("OnlineSpatialNet mhsa scope must be a positive integer")
        scope = int(body)
        if attention_scope is not None and attention_scope != scope:
            raise ValueError("attention_scope conflicts with attention")
        if rope:
            raise ValueError("OnlineSpatialNet rope is only supported by the ret branch")
        if decay is not None:
            raise ValueError("OnlineSpatialNet decay is only used by the ret branch")
        return {"branch": "msa", "scope": scope}

    if text.startswith("ret(") and text.endswith(")"):
        parts = [part.strip() for part in text[4:-1].split(",")]
        if not parts or not parts[0].isdigit() or int(parts[0]) < 1 or len(parts) > 2:
            raise ValueError("OnlineSpatialNet ret attention must be ret(factor[,share_qk])")
        value_factor = int(parts[0])
        if len(parts) == 1:
            share_qk = not rope
        elif parts[1] == "share_qk":
            share_qk = True
        elif parts[1] == "not_share_qk":
            share_qk = False
        else:
            raise ValueError("OnlineSpatialNet ret attention must be ret(factor[,share_qk])")
        if isinstance(decay, Sequence) and not isinstance(decay, (str, bytes)):
            if len(list(decay)) != num_heads:
                raise ValueError("OnlineSpatialNet ret decay length must equal num_heads")
        return {
            "branch": "ret",
            "retention_args": {
                "value_factor": value_factor,
                "share_qk": share_qk,
                "decay": decay,
                "rope": rope,
            },
        }

    if text.startswith("mamba(") and text.endswith(")"):
        if attention_scope is not None:
            raise ValueError("OnlineSpatialNet attention_scope is only used by the mhsa branch")
        if rope:
            raise ValueError("OnlineSpatialNet rope is only supported by the ret branch")
        if decay is not None:
            raise ValueError("OnlineSpatialNet decay is only used by the ret branch")
        parts = [part.strip() for part in text[6:-1].split(",")]
        if len(parts) != 2 or not all(part.isdigit() and int(part) > 0 for part in parts):
            raise ValueError("OnlineSpatialNet mamba attention must be mamba(d_state,d_conv)")
        if not mamba_available():
            raise RuntimeError(
                "OnlineSpatialNet mamba attention requires optional mamba_ssm; "
                "the supported backends are attention='mhsa(scope)' and attention='ret(factor,share_qk)'"
            )
        return {"branch": "mamba", "mamba_args": {"d_state": int(parts[0]), "d_conv": int(parts[1])}}

    raise ValueError(
        "OnlineSpatialNet attention must be mhsa(scope), ret(factor,share_qk) or mamba(d_state,d_conv), "
        f"got {attention!r}"
    )




class OnlineSpatialBackbone(nn.Module):
    """encoder + 交错 causal layers + decoder；整段与逐帧两条前向共享参数。

    整段前向（训练/离线推理）与 ``forward_streaming``（逐帧 ONNX ABI）对同一权重给出
    一致结果：``mhsa`` 走窗口 mask 注意力与增量 K/V，``ret`` 与 ``mamba`` 走显式状态递推。

    两条路径的计算量与调用开销差异很大：``forward_sequence`` 把时间轴折进 batch（频域卷积
    按 ``[B*T, H, F]``、因果卷积左 padding 一次跑完、mhsa 一次带窗口 mask 的注意力），而
    ``forward_streaming`` 每帧调用一次小算子并更新 cache。实测 8 kHz/F=129/4 s（251 帧）：
    整段 3,815 次 aten 调用，逐帧累计 989,853 次（≈3,944 次/帧，与整段一次前向相当），
    且逐帧 top-ops 以 ``as_strided``/``empty``/``view`` 等形状与分配操作为主。因此逐帧 RTF
    明显高于整段（CPU 单进程：mhsa 0.20 vs 3.02、ret 0.68 vs 2.43、mamba 1.49 vs 2.30），
    逐帧数字才是流式部署口径；整段数字只代表训练/离线前向。
    """

    def __init__(
        self,
        *,
        n_channel: int,
        num_freqs: int,
        output_dim: int,
        num_layers: int,
        dim_hidden: int,
        dim_ffn: int,
        dim_squeeze: int,
        num_heads: int,
        attention_spec: str | None,
        attention_scope: int | None,
        decay,
        rope: bool,
        encoder_kernel_size: int = 5,
        kernel_size: Sequence[int] = (5, 3),
        conv_groups: Sequence[int] = (8, 8),
        norms: Sequence[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
        dropout: Sequence[float] = (0, 0, 0),
        full_share: int = 0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if type(n_channel) is not int or n_channel < 1:
            raise ValueError("n_channel must be a positive integer")
        if type(num_freqs) is not int or num_freqs < 1:
            raise ValueError("num_freqs must be a positive integer")
        if type(output_dim) is not int or output_dim < 1:
            raise ValueError("output_dim must be a positive integer")
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        if type(full_share) is not int or not 0 <= full_share < num_layers:
            raise ValueError("full_share must select a layer in the configured stack")
        if type(encoder_kernel_size) is not int or encoder_kernel_size < 1 or encoder_kernel_size % 2 == 0:
            raise ValueError("encoder_kernel_size must be a positive odd integer")
        if len(kernel_size) != 2 or len(conv_groups) != 2:
            raise ValueError("kernel_size and conv_groups must contain two values")
        if any(type(value) is not int or value < 1 or value % 2 == 0 for value in kernel_size):
            raise ValueError("kernel_size values must be positive odd integers")
        if any(type(value) is not int or value < 1 for value in conv_groups):
            raise ValueError("conv_groups values must be positive exact integers")
        if len(norms) != 6:
            raise ValueError("norms must contain six values")
        if any(str(norm).upper() not in {"LN", "GN"} for norm in norms):
            raise ValueError("OnlineSpatialNet supports LN and GN normalization names")
        if len(dropout) != 3 or any(
            isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value < 1
            for value in dropout
        ):
            raise ValueError("OnlineSpatialNet dropout must contain three values in [0, 1)")
        if any(value != 0 for value in dropout):
            raise ValueError("OnlineSpatialNet streaming backend does not support non-zero dropout")
        if type(gradient_checkpointing) is not bool:
            raise TypeError("gradient_checkpointing must be an exact bool")

        resolved = resolve_attention(attention_spec, attention_scope, num_heads, decay, rope)
        self.gradient_checkpointing = gradient_checkpointing
        self.attention_branch = resolved["branch"]
        self.attention_spec = attention_spec if attention_spec is not None else f"mhsa({attention_scope or 251})"
        self.rope = rope
        self.n_channel = n_channel
        self.num_freqs = num_freqs
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_history = encoder_kernel_size - 1
        self.encoder = nn.Conv1d(n_channel * 2, dim_hidden, encoder_kernel_size, padding=0)
        layers = []
        shared = None
        for index in range(num_layers):
            layer = _CausalLayer(
                dim_hidden,
                dim_ffn,
                dim_squeeze,
                num_freqs,
                num_heads,
                self.attention_branch,
                resolved.get("scope"),
                kernel_size,
                conv_groups,
                norms,
                shared if index > full_share else None,
                resolved.get("retention_args", {}),
                resolved.get("mamba_args", {}),
            )
            if index >= full_share:
                shared = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.decoder = nn.Linear(dim_hidden, output_dim)
        self.needs_frame_index = self.attention_branch == "msa" or (
            self.attention_branch == "ret" and bool(rope)
        )
        self.CACHE_KEYS = self._cache_keys()

    def _cache_keys(self) -> tuple[str, ...]:
        keys = ["encoder"]
        if self.needs_frame_index:
            keys.append("frame_index")
        for index, layer in enumerate(self.layers):
            keys.extend(layer.cache_names(index))
        return tuple(keys)

    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> OrderedDict[str, Tensor]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        cache: OrderedDict[str, Tensor] = OrderedDict()
        cache["encoder"] = torch.zeros(
            batch_size, self.num_freqs, self.n_channel * 2, self.encoder_history, device=device, dtype=dtype
        )
        if self.needs_frame_index:
            cache["frame_index"] = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        for index, layer in enumerate(self.layers):
            cache.update(layer.init_cache(batch_size, self.num_freqs, device, dtype, index))
        return cache

    def expected_cache_shapes(self, batch: int) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {
            "encoder": (batch, self.num_freqs, self.n_channel * 2, self.encoder_history),
        }
        if self.needs_frame_index:
            shapes["frame_index"] = (batch, 1)
        for index, layer in enumerate(self.layers):
            for name in layer.cache_names(index):
                shapes[name] = layer.cache_shape(name, batch, self.num_freqs, index)
        return shapes

    def validate_cache(self, cache: Mapping[str, Tensor], batch: int, device: torch.device, dtype: torch.dtype) -> None:
        if not isinstance(cache, Mapping):
            raise TypeError("OnlineSpatialNet cache must be a mapping")
        if tuple(cache) != self.CACHE_KEYS:
            raise ValueError(
                f"OnlineSpatialNet cache names/order mismatch: expected={self.CACHE_KEYS}, got={tuple(cache)}"
            )
        expected_shapes = self.expected_cache_shapes(batch)
        for name in self.CACHE_KEYS:
            value = cache[name]
            if not isinstance(value, Tensor) or value.device != device or value.dtype != dtype:
                raise ValueError(f"OnlineSpatialNet cache {name!r} must match input device and dtype")
            if tuple(value.shape) != expected_shapes[name]:
                raise ValueError(
                    f"OnlineSpatialNet cache {name!r} shape mismatch: "
                    f"expected={expected_shapes[name]}, got={tuple(value.shape)}"
                )

    def forward_streaming(self, mixture_spec: Tensor, cache: Mapping[str, Tensor]):
        """单帧流式前向：``mixture_spec`` 为 ``[B, C, F, 1, 2]``，返回 (谱, 新 cache)。"""
        if mixture_spec.ndim != 5 or mixture_spec.shape[-1] != 2:
            raise ValueError("OnlineSpatialNet expects [B,C,F,T,2] spectrum input")
        if mixture_spec.shape[1] != self.n_channel or mixture_spec.shape[2] != self.num_freqs or mixture_spec.shape[3] != 1:
            raise ValueError("OnlineSpatialNet streaming input shape does not match configured C/F and one-frame ABI")
        self.validate_cache(cache, mixture_spec.shape[0], mixture_spec.device, mixture_spec.dtype)
        batch, _, freqs, _, _ = mixture_spec.shape
        value = mixture_spec.permute(0, 2, 3, 1, 4).reshape(batch, freqs, 1, self.n_channel * 2)
        value = value.permute(0, 1, 3, 2).reshape(batch * freqs, self.n_channel * 2, 1)
        encoder_cache = cache["encoder"]
        next_cache = OrderedDict(cache)
        if self.encoder_history:
            history = encoder_cache.reshape(batch * freqs, self.n_channel * 2, self.encoder_history)
            encoded_input = torch.cat((history, value), dim=-1)
            encoder_tail = encoded_input[..., -self.encoder_history :]
            next_cache["encoder"] = encoder_tail.reshape(
                batch, freqs, self.n_channel * 2, self.encoder_history
            )
        else:
            encoded_input = value
            next_cache["encoder"] = encoder_cache
        value = self.encoder(encoded_input).transpose(1, 2).reshape(batch, freqs, 1, -1)
        frame_index = cache["frame_index"] if self.needs_frame_index else None
        for index, layer in enumerate(self.layers):
            value, next_cache = layer.forward_frame(value, next_cache, index, frame_index)
        if self.needs_frame_index:
            next_cache["frame_index"] = frame_index + 1.0
        return self.decoder(value), next_cache

    def _run_layer(self, layer: nn.Module, value: Tensor) -> Tensor:
        """按需对单层做 activation recomputation（长片段训练省显存，约 +50% 计算）。

        必须用 ``use_reentrant=False``：``full_share`` 会把 ``layers[full_share].full``
        共享给后续所有层，同一参数出现在多个 checkpoint 段中；reentrant 版本在 backward
        时对每个段重放子图，DDP 的 per-parameter ready hook 被多次触发，第一个 step 即抛
        "Expected to mark a variable ready only once"（DDP 仅支持非 reentrant
        checkpoint 跨段复用参数）。实测两种模式显存基本持平：mhsa T=2001（32 s）/批 2/
        F=257/C=8 峰值 19.2 vs 19.5 GiB；ret T=1001 19.9 vs 20.1 GiB。层前向无 RNG
        （dropout 固定为 0）且输入为单个张量，满足 checkpoint 的使用前提。
        """
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(layer.forward_sequence, value, use_reentrant=False)
        return layer.forward_sequence(value)

    def forward_sequence(self, mixture_spec: Tensor) -> Tensor:
        """整段前向：``mixture_spec`` 为 ``[B, C, F, T, 2]``，返回 ``[B, F, T, output_dim]``。"""
        if mixture_spec.ndim != 5 or mixture_spec.shape[-1] != 2:
            raise ValueError("OnlineSpatialNet expects [B,C,F,T,2] spectrum input")
        if mixture_spec.shape[1] != self.n_channel or mixture_spec.shape[2] != self.num_freqs:
            raise ValueError("OnlineSpatialNet input shape does not match configured C/F")
        if mixture_spec.shape[3] < 1:
            raise ValueError("OnlineSpatialNet requires at least one frame")
        batch, channels, freqs, frames, _ = mixture_spec.shape
        value = mixture_spec.float().permute(0, 2, 3, 1, 4).reshape(batch, freqs, frames, channels * 2)
        flat = value.reshape(batch * freqs, frames, channels * 2).transpose(1, 2)
        if self.encoder_history:
            flat = F.pad(flat, (self.encoder_history, 0))
        value = self.encoder(flat).transpose(1, 2).reshape(batch, freqs, frames, -1)
        for layer in self.layers:
            value = self._run_layer(layer, value)
        return self.decoder(value)


__all__ = [
    "OnlineSpatialBackbone",
    "resolve_attention",
]
