"""mamba_ssm ``Mamba`` 的窄带分支封装：整段前向 + tensor-only 单帧 step。

上游 NBSS 在 ``attention='mamba(d_state,d_conv)'`` 时用 mamba_ssm 的 Mamba 替换窄带
MHSA，并默认同时替换 T-ConvFFN；本模块保留该约定，并把流式状态显式化为
``conv_state``/``ssm_state`` 两个张量，供 OnlineSpatialNet 的 flat cache ABI 使用。

``mamba_ssm`` 是可选依赖：仅 mamba 分支需要，缺少时在构造边界的报错信息里说明；
单帧 step 不依赖 Triton/causal_conv1d，走与本仓 ``rt_semamba_impl`` 相同的纯 PyTorch
参考更新，保证 CPU 可测。性能上要注意两条路径的差别：CUDA 整段前向走官方 selective-scan
内核，而 CPU/无内核环境与逐帧 step 走 Python 级 T 循环，慢得多（实测 8 kHz/F=129/4 s 的
CPU RTF：整段 1.49 vs 逐帧 2.30），因此 CPU 上的 mamba 数字不代表 GPU 成本。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:  # pragma: no cover - 由分支构造边界的错误信息覆盖
    from mamba_ssm.modules.mamba_simple import Mamba
except (ImportError, OSError) as exc:  # pragma: no cover
    Mamba = None
    _MAMBA_IMPORT_ERROR: Exception | None = exc
else:
    _MAMBA_IMPORT_ERROR = None


def mamba_available() -> bool:
    """mamba_ssm 是否可用；不可用时由 ``CausalMamba`` 给出明确报错。"""
    return Mamba is not None


class CausalMamba(nn.Module):
    """mamba_ssm ``Mamba`` 的薄封装。

    state 形状与官方 ``allocate_inference_cache`` 一致：
    ``conv_state [N, d_model*expand, d_conv]``、``ssm_state [N, d_model*expand, d_state]``。
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int) -> None:
        super().__init__()
        if Mamba is None:  # pragma: no cover - 依赖缺失路径
            raise ImportError(
                "OnlineSpatialNet attention='mamba(...)' requires optional dependency mamba_ssm"
            ) from _MAMBA_IMPORT_ERROR
        for name, value in (("d_model", d_model), ("d_state", d_state), ("d_conv", d_conv)):
            if type(value) is not int or value < 1:
                raise ValueError(f"CausalMamba {name} must be a positive int")
        self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, layer_idx=0)

    @property
    def inner_dim(self) -> int:
        return self.mixer.d_model * self.mixer.expand

    @property
    def d_state(self) -> int:
        return self.mixer.d_state

    @property
    def d_conv(self) -> int:
        return self.mixer.d_conv

    def forward(self, x: Tensor) -> Tensor:
        """整段前向；``x`` 形如 ``[N, T, d_model]``。

        CUDA 且官方扩展可用时走 mamba_ssm 的并行内核；否则退回本模块的纯 PyTorch
        逐帧递推，保证 CPU 测试、无内核环境和流式 step 使用同一套数学语义。
        """
        if x.is_cuda and mamba_available():
            return self.mixer(x)
        return self.forward_reference(x)

    def forward_reference(self, x: Tensor) -> Tensor:
        """纯 PyTorch 选择性扫描；与 ``step`` 语义一致，用于 CPU 与内核不可用环境。"""
        if x.ndim != 3:
            raise ValueError("CausalMamba sequence input must have shape [N, T, d_model]")
        batch, seq_len, _ = x.shape
        conv_state, ssm_state = self.init_state(batch, x.device, x.dtype)
        outputs = []
        for index in range(seq_len):
            frame, conv_state, ssm_state = self.step(x[:, index : index + 1, :], conv_state, ssm_state)
            outputs.append(frame)
        if not outputs:
            raise ValueError("CausalMamba requires at least one frame")
        return torch.cat(outputs, dim=1)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        conv_state = torch.zeros(batch, self.inner_dim, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch, self.inner_dim, self.d_state, device=device, dtype=dtype)
        return conv_state, ssm_state

    def step(self, x: Tensor, conv_state: Tensor, ssm_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """单帧更新；不原地修改传入 state，返回 ``(y, next_conv_state, next_ssm_state)``。"""
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("CausalMamba streaming step expects a single token [N, 1, d_model]")
        mixer = self.mixer
        dtype = x.dtype
        xz = mixer.in_proj(x.squeeze(1))
        x_inner, z = xz.chunk(2, dim=-1)

        next_conv = torch.cat((conv_state[:, :, 1:], x_inner.unsqueeze(-1)), dim=-1)
        conv_out = torch.sum(next_conv * mixer.conv1d.weight.squeeze(1), dim=-1)
        if mixer.conv1d.bias is not None:
            conv_out = conv_out + mixer.conv1d.bias
        conv_out = mixer.act(conv_out).to(dtype=dtype)

        x_db = mixer.x_proj(conv_out)
        dt, b_state, c_state = torch.split(
            x_db, [mixer.dt_rank, mixer.d_state, mixer.d_state], dim=-1
        )
        dt = F.linear(dt, mixer.dt_proj.weight)
        transition = -torch.exp(mixer.A_log.float())
        dt = F.softplus(dt + mixer.dt_proj.bias.to(dtype=dt.dtype))
        discrete_a = torch.exp(torch.einsum("bd,dn->bdn", dt, transition))
        discrete_b = torch.einsum("bd,bn->bdn", dt, b_state)
        next_ssm = ssm_state * discrete_a + conv_out.unsqueeze(-1) * discrete_b
        y = torch.einsum("bdn,bn->bd", next_ssm.to(conv_out.dtype), c_state)
        y = (y + mixer.D.to(conv_out.dtype) * conv_out) * mixer.act(z)
        return mixer.out_proj(y).unsqueeze(1), next_conv, next_ssm

    def extra_repr(self) -> str:
        return f"d_model={self.mixer.d_model}, d_state={self.d_state}, d_conv={self.d_conv}"


__all__ = ["CausalMamba", "mamba_available"]
