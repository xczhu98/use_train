"""MultiScaleRetention（RetNet）窄带时序模块，移植自 NBSS。

来源：<https://github.com/Audio-WestlakeU/NBSS> 的 ``models/arch/base/retention.py``
（MIT License，Copyright (c) 2022 Microsoft，Copyright (c) 2023 quancs@westlake.edu.cn）。
本模块按本仓类型标注与中文注释约定重写，只保留 OnlineSpatialNet 使用的 recurrent 路径
（显式增量状态、训练与流式共用），去掉上游仅用于 look-ahead 的前向分支与 parallel /
chunkwise 两种整段前向；单步数学语义与上游保持一致。

取舍说明：整段前向也按逐帧 recurrent 递推实现，因此训练/推理数值一致，代价是比上游的
parallel/chunkwise 实现慢（每层每帧一次小算子）；FLOPs 口径不受影响（逐帧代价 × 帧数，
实测 8 kHz/F=129/4 s 为 20.42 G/s，与论文 20.4 G/s 一致）。
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """非仿射或逐元素仿射的 RMSNorm，与上游 retention 使用的形式一致。"""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


def _normalize_decays(
    decay: int | bool | Sequence[int] | Sequence[float] | None,
    num_heads: int,
) -> list[float]:
    """把上游 ``decay`` 语义（False / 序列 / 整数）解析为每个 head 的衰减系数。"""
    if decay is False:
        return [1.0] * num_heads
    if isinstance(decay, Sequence) and not isinstance(decay, (str, bytes)):
        values = list(decay)
        if not values:
            raise ValueError("RetNet decay sequence must not be empty")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise ValueError("RetNet decay sequence must contain real numbers")
        if isinstance(values[0], float):
            if any(value <= 0 or value > 1 for value in values):
                raise ValueError("RetNet float decay values must lie in (0, 1]")
            return [float(value) for value in values]
        if any(not float(value).is_integer() or int(value) <= 1 for value in values):
            raise ValueError("RetNet integer decay values must be ints > 1")
        return [1.0 - 2.0 ** (-int(value)) for value in values]
    if decay is None or decay is True:
        decay = 5
    if isinstance(decay, bool) or not isinstance(decay, int) or decay <= 0:
        raise ValueError("RetNet integer decay must be a positive int")
    return [1.0 - 2.0 ** (-decay - index) for index in range(num_heads)]


class RetNetRelPos(nn.Module):
    """RetNet 的单步 recurrent 相对位置/衰减。

    上游还提供 parallel 与 chunkwise 两种整段前向使用的 mask 构造；本仓的 OnlineSpatialNet
    只使用 recurrent 形式（训练与流式共用同一路径，保证两者一致），因此这里只保留该路径
    需要的 ``angle``/``decay`` buffer 与 ``recurrent()`` 输出。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        decay: int | bool | Sequence[int] | Sequence[float] | None = 5,
    ) -> None:
        super().__init__()
        if type(embed_dim) is not int or embed_dim <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be a positive int divisible by num_heads")
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.decays = _normalize_decays(decay, num_heads)
        self.register_buffer("angle", angle)
        self.register_buffer("decay", torch.log(torch.tensor(self.decays, dtype=torch.float)))

    def recurrent(self, frame_index: Tensor | None = None):
        """返回 recurrent 前向使用的 ``((sin, cos), decay)``；sin/cos 仅在 rope 启用时参与计算。"""
        if frame_index is None:
            sin = cos = None
        else:
            # 行内形如 [1, key_dim]，供 theta_shift 的 cos[:tgt_len] 切片对单帧广播。
            angle = self.angle.reshape(1, -1) * (frame_index - 1.0)
            sin = torch.sin(angle)
            cos = torch.cos(angle)
        return ((sin, cos), self.decay.exp())

    def extra_repr(self) -> str:
        efflen = [1 / (1 - d) if d < 1 else float("inf") for d in self.decays]
        return f"decays={self.decays} -> effective len={efflen}"


def rotate_every_two(x: Tensor) -> Tensor:
    """RoPE 的成对旋转；上游对 4D 张量实现，这里保留同样的语义。"""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def theta_shift(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    slen = x.shape[-2]
    return (x * cos[:slen]) + (rotate_every_two(x) * sin[:slen])


def _get_activation_fn(activation: str):
    if activation == "swish":
        return F.silu
    if activation == "gelu":
        return F.gelu
    raise NotImplementedError(f"unsupported retention gating activation: {activation!r}")


class MultiScaleRetention(nn.Module):
    """多尺度 Retention：parallel / chunkwise / recurrent 三种前向。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        value_factor: int = 2,
        gate_fn: str = "swish",
        share_qk: bool = False,
    ) -> None:
        super().__init__()
        value_dim = embed_dim * value_factor
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = self.value_dim // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.share_qk = share_qk
        self.gate_fn = _get_activation_fn(activation=str(gate_fn))

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False) if not share_qk else None
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.out_proj = nn.Linear(value_dim, embed_dim, bias=False)
        self.group_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-2.5)
        if self.k_proj is not None:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2**-1)

    def recurrent_forward(self, qr: Tensor, kr: Tensor, v: Tensor, decay: Tensor, incremental_state: dict) -> Tensor:
        bsz = v.size(0)
        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v  # [bsz, heads, head_dim, key_dim]
        head_decay = decay.reshape(1, self.num_heads, 1, 1)
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            scale = prev_scale * head_decay + 1
            # 上游此处把 [batch, heads, 1, 1] 的缩放写成 view(num_heads, 1, 1)，仅在 batch=1 时成立；
            # 这里按 head 维广播，保持同一数学语义并支持任意 batch×frequency。
            kv = prev_kv * (prev_scale.sqrt() * head_decay / scale.sqrt()) + kv / scale.sqrt()
        else:
            scale = torch.ones(
                bsz, self.num_heads, 1, 1, device=kv.device, dtype=kv.dtype
            )

        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale
        return torch.sum(qr * kv, dim=3)

    def forward(self, x: Tensor, rel_pos, incremental_state: dict, rope: bool = False) -> Tensor:
        """recurrent 前向；``rel_pos`` 为 ``RetNetRelPos.recurrent()`` 的输出。"""
        bsz, tgt_len, _ = x.size()
        (sin, cos), decay = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x) if not self.share_qk else None
        v = self.v_proj(x)
        g = self.g_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        if not self.share_qk:
            k = k * self.scaling
            k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        else:
            k = q

        qr = theta_shift(q, sin, cos) if rope else q
        kr = theta_shift(k, sin, cos) if rope else k

        output = self.recurrent_forward(qr, kr, v, decay, incremental_state)
        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)
        output = self.gate_fn(g) * output
        return self.out_proj(output)

    def init_incremental_state(self, batch: int, device, dtype) -> dict:
        """recurrent 状态：kv 形状 [batch, heads, head_dim, key_dim]，与 recurrent_forward 一致。"""
        return {
            "prev_key_value": torch.zeros(
                batch, self.num_heads, self.head_dim, self.key_dim, device=device, dtype=dtype
            ),
            "scale": torch.ones(batch, self.num_heads, 1, 1, device=device, dtype=dtype),
        }


__all__ = ["MultiScaleRetention", "RMSNorm", "RetNetRelPos", "rotate_every_two", "theta_shift"]
