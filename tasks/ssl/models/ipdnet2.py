"""IPDnet2 官方 ``OnlineSpatialNet``（oSpatialNet + Mamba）结构接入。

上游参考实现为 FN-SSL 仓库的
``IPDnet2/IPDnet2.py``、``IPDnet2/arch/base/{norm,non_linear,linear_group}.py``，
训练侧构造超参与数据装配见 ``IPDnet2/run_IPDnet2.py``。本模块只移植**模型结构**，
不移植官方数据 recipe（RealMAN 真实录音、``utils_`` 的 forgetting norm 构造器等）。

官方结构与本地对应
------------------

- 骨干：``CausalConv1d`` encoder（kernel=5, look_ahead=0）→ ``num_layers`` 个
  ``SpatialNetLayer`` → ``FreqInverse`` → ``Linear(dim_output, dim_output)`` 解码；
  官方 ``run_IPDnet2.py`` 以 ``dim_input=10, dim_output=16, num_layers=8,
  dim_hidden=96, num_heads=4, dim_squeeze=8, num_freqs=256, conv_groups=(8,8),
  norms=["LN","LN","GN","LN","LN","LN"], attention='mamba(16,4)',
  fre_compression_ratio=16, time_compression_ratio=5, time_compression_layer=0``
  实例化（``ipdnet2_small``）。
- ``attention`` 解析沿用官方字符串格式（官方按 ``attention[6:-1].split(',')``，
  仅对 ``'mamba(16,4)'`` 正确）；本模块按 ``attention[6:-1]`` 与官方完全一致，
  并要求前缀必须是 ``mamba``，非法字符串在构造期 ``ValueError``。
- 每层两个序列槽位：``mhsa`` 槽与 ``tconvffn`` 槽，官方在 ``attention='mamba(...)'``
  下两个槽位都是同一个 ``Mamba(d_model=dim_hidden, d_state, d_conv)``。
  官方 ``Mamba(...)`` 调用未传 ``layer_idx``，因此**不会**落盘 ``layer_idx``；
  本实现同样不传，保持与官方 checkpoint 的 ``state_dict`` key 完全一致
  （``SpatialNetLayer`` 只保留 ``mhsa`` 名称，不额外注册任何 buffer）。
- 频率压缩全部发生在 layer 0 内部：``AvgPool2d(1,2)`` 把 ``num_freqs`` 折半
  （256→128，即官方传给 layer 0 的 ``num_freqs//2``），``AvgPool2d(1,8)`` 再折
  八分之一（128→16 = ``num_freqs // fre_compression_ratio``）；layer ≥ 1 的
  ``full`` 线性层作用在 16 个压缩频率上。输出端 ``FreqInverse`` 把 16 个压缩
  频率逆回 ``num_freqs``。
- 时间压缩：``time_compression_layer``（官方 0）那一层之后接
  ``AvgPool2d(time_compression_ratio, 1)``，只压缩时间轴，频率轴不变。

输入输出边界（框架 canonical）
-----------------------------

- 输入：``multichannel_stft_real_imag`` 的 canonical 形状 ``[B,T,F,2C]``，其中
  ``F = n_fft//2 + 1``（由 Dataset 侧 ``exclude_dc`` 决定是否已删除 DC）。
  框架特征末维的 real/imag 分块顺序为 ``[c0_real, c0_imag, c1_real, c1_imag, ...]``
  （``tasks/ssl/data/projection.py::_raw_stft_real_imag`` 的 ``view_as_real``
  结果），而官方输入是 ``[B, 2C, F, T]``，其分块顺序为**先全部 real 通道、
  再全部 imag 通道**（``run_IPDnet2.py`` 的 ``torch.cat((real, imag), dim=1)``）。
  两者不等价，本模块按通道显式重排（``_official_layout``），不做隐式 view。
- 官方 ``fre_range_used = range(1, n_fft//2 + 1)`` 表示删 DC 后只喂
  ``num_freqs = n_fft//2`` 个频点；因此 ``num_freqs`` 必须与删除 DC 后的特征
  频率轴一致，否则构造期/前向期 fail-fast。
- 幅度归一化：官方在 Dataset 侧（``run_IPDnet2.py::data_preprocess`` 调用
  ``utils_.forgetting_norm``）做 forgetting normalization，并把 real/imag 同除
  该因子。本模块支持模块内等价实现（``forgetting_norm_sample_length``，``None``
  表示关闭），递推式与官方逐行一致：``alpha=(L-1)/(L+1)``，前 ``L`` 帧使用
  ``min((t-1)/(t+1), alpha)``、之后固定 ``alpha``，递推量是同一帧上 ``C*F`` 个
  幅度值的均值；本地等价实现见 ``tasks/ssl/models/ipdnet_official.py::forgetting_norm``。
  官方在训练/推理都启用该归一化（推理路径同样调用 ``data_preprocess``），
  所以本实现不做 train/eval 分支。**默认 ``sample_length=249`` 取自官方
  ``IPDnet2/RecordData.py`` 的调用值，它对应官方 4.5 s 片段与官方 STFT 帧率
  （win_shift_ratio=0.625、fs=16000 → hop=320 样本）；本地 hop_length 不同时
  249 不是等效时间常数**；该数值只写入本地配置/provenance，不构成官方指标
  可复现声明。
- 输出：``FreqInverse`` 输出 ``[B, dim_output, T_out, num_freqs]``，第 3 维是
  ``C-1`` 个 reference-to-all pair、第 4 维是频率；官方随后把 real/imag 沿频率维
  拼接为 ``2F`` 并 reshape 成 ``[B,T,2F,C-1,K]``（real 块在前、imag 块在后）。
  本模块拆开该拼接并转置回框架 canonical ``[B,T_out,F,K,P,2]``（末维 2 =
  复数实虚部），另乘 ``output_scale`` 把官方 ``tanh`` 输出从 ``(-1,1)`` 映射到
  ``[-π,π]`` 的相位域（默认 1.0 = 纯官方输出，不隐式归一化）。
- ``T_out = T // time_compression_ratio``（官方池化后向下取整），与框架
  ``time_stride`` 必须一致；``SpatialNetLayer`` 内部的 ``Conv1d``（``padding='same'``）
  与 ``FreqInverse`` 的逐频带拷贝都要求 ``T`` 能被时间压缩比整除。

与官方实现的已知偏离（逐条登记，不隐藏）
----------------------------------------

**tensor 级忠实度已用 golden 对照实测**：以官方 ``IPDnet2.py``/``arch/base/*.py``
逐行转写的参考实现在 GPU 上做 ``state_dict`` 严格加载（``strict=True``，
0 missing / 0 unexpected）后，``num_layers=8, dim_hidden=96, num_freqs=256``
（官方 ``run_IPDnet2.py`` 配置）与小型对照配置的 ``encoder``、``layers``、
``FreqInverse``、``decoder`` 及最终输出均为**逐元素 0.0 误差**。下列偏离是
移除官方 bug/死代码后的等价或超集行为，不影响上述对照配置的数值。

1. ``SpatialNetLayer`` 的 ``tconvffn`` 槽在上游支持 ``Mamba`` 或由
   ``CausalConv1d``/``PReLU``/``Conv1d`` 组成的序列模块（``_tconvffn`` 遍历
   模块列表）；本实现只保留官方 ``attention='mamba(...)'`` 实际使用的两个
   ``Mamba`` 槽位。``arrangement='alternating'``（测试用，非官方配置）另行
   提供 ``MultiheadAttention`` + T-Conv/FFN 版本。
2. 上游 ``GroupBatchNorm``（``GBN``）的分组统计在 ``affine=False`` 时引用未定义
   的局部变量，并在 ``seq_last=True`` 下对维度语义有额外假设，两条路径在现代
   PyTorch 上不可靠。本实现提供等价的 ``GBN``/``GBNShare``（只实现
   ``seq_last=False`` 的 ``[B,H,F,T]`` 路径，其余组合 fail-fast），但默认
   ``norms`` 与官方保持一致，不含 ``GBN``。
3. 官方两个 ``PReLU(dim_hidden)`` 依赖 ``nn.PReLU`` 作用在 ``dim=1`` 的隐式
   语义（``[B*T,H,F]`` 下作用轴是 ``H`` 且不跨频率共享权重）。本实现用
   ``_prelu`` 在显式轴上复现同一语义，参数名保持官方 ``weight``，避免依赖帧数
   与 ``dim_hidden`` 的巧合相等。
4. 官方 ``FreqInverse`` 用逐频带 ``for i in range(nfilters)`` 拷贝，要求
   ``T_out % (num_freqs // fre_compression_ratio) == 0``；本实现改成一次性
   reshape（数学等价、无该整除要求），输出与官方逐元素一致。
5. 官方 ``OnlineSpatialNet`` 的 ``_mamba(inference=True)`` 逐帧推理分支依赖
   mamba_ssm 的 ``InferenceParams``（以及需要 ``chunkwise_recurrent`` 的遗留
   attention 路径）；``get_causal_mask`` 的 mask 只喂给 ``_tsa``，在本实现采用
   的 mamba 槽位上被忽略。本实现只移植并行前向，不移植该逐帧推理分支，因此
   **本模块不声明 streaming 支持**。
6. 官方 ``arch/base/retention.py``（``MultiScaleRetention``/``RetNetRelPos``）是
   未使用的遗留死代码，官方 checkpoint 也只有 326 个 mamba 相关键；本实现不移植。
7. 上游 ``PPReLU`` 之外的 ``linear_group.py``（``LinearGroup``/
   ``Conv1dGroup``/``LinearGroupSharedWeight``）在官方 ``attention='mamba(...)'``
   配置下不被实例化；本实现不移植这些未使用类。
8. 官方 ``FreqInverse`` 的 ``tanh`` 输出范围是 ``(-1,1)``，官方未做单位圆
   归一化；本实现同样不做，另提供 ``output_scale`` 让调用方显式把输出映射到
   ``[-π,π]`` 相位域（默认 ``1.0`` 即纯官方输出）。
9. 官方在 ``run_IPDnet2.py::data_preprocess`` 用 ``forgetting_norm`` 对 real/imag
   做幅度归一化；本模块把等价递推（``forgetting_norm_sample_length``）内置，
   便于在不改 Dataset 的情况下验证结构。两者不能同时启用——若数据侧已归一化，
   应把 ``forgetting_norm_sample_length`` 设为 ``None`` 避免重复归一化。

``mamba_ssm`` 依赖
------------------

上游是 ``try: from mamba_ssm import Mamba except: Mamba = None``，缺失时在建模期
以 ``TypeError`` 崩溃，实质是硬依赖且没有 GRU/LSTM 回退。本实现改为
**optional import + 构造期显式 ``ImportError``**：导入本模块本身不触碰
``mamba_ssm``，只有实例化 ``IPDNet2`` 才报错并给出安装提示。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from core.models.io_contract import ModelIOContract, TensorField
from core.utils.registry import MODEL_REGISTRY
from tasks.ssl.models.ipdnet import _PairwiseIPDModel
from tasks.ssl.model_io import ModelSpec


_NORMS = frozenset({"LN", "BN", "GN", "GBN", "GBNShare", "gLN"})
_ARRANGEMENTS = frozenset({"mamba_only", "alternating"})


class _LayerNorm(nn.LayerNorm):
    """官方 ``LayerNorm``：``seq_last=True`` 时在倒数第 2 轴做归一化。

    官方实现对 ``[B,F,T,H]`` 直接 ``transpose(-1, 1)``，等价于只转置倒数两轴，
    归一化轴落在 ``T`` 上；官方 checkpoint 的 ``weight`` 形状为 ``[dim_hidden]``。
    """

    def __init__(self, seq_last: bool, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, x: Tensor) -> Tensor:
        if self.seq_last:
            x = x.transpose(-1, -2)
        out = super().forward(x)
        if self.seq_last:
            out = out.transpose(-1, -2)
        return out


class _BatchNorm1d(nn.Module):
    """官方 ``BatchNorm1d``：``seq_last=False`` 时在倒数第 1 轴做归一化。"""

    def __init__(self, seq_last: bool, **kwargs: Any) -> None:
        super().__init__()
        self.seq_last = seq_last
        self.bn = nn.BatchNorm1d(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if not self.seq_last:
            x = x.transpose(-1, -2)
        out = self.bn(x)
        if not self.seq_last:
            out = out.transpose(-1, -2)
        return out


class _GroupNorm(nn.GroupNorm):
    """官方 ``GroupNorm``：``seq_last=False`` 时在倒数第 1 轴做归一化。"""

    def __init__(self, seq_last: bool, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, x: Tensor) -> Tensor:
        if not self.seq_last:
            x = x.transpose(-1, -2)
        out = super().forward(x)
        if not self.seq_last:
            out = out.transpose(-1, -2)
        return out


class _GlobalLayerNorm(nn.Module):
    """convtasnet 风格的 gLN，权重可广播到 ``[B, ..., dim]``。"""

    def __init__(self, dim_hidden: int, seq_last: bool, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim_hidden = dim_hidden
        self.seq_last = seq_last
        self.eps = eps
        shape = (dim_hidden, 1) if seq_last else (dim_hidden,)
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x: Tensor) -> Tensor:
        if self.seq_last:
            dims = tuple(range(1, x.ndim - 1))
        else:
            dims = tuple(range(x.ndim - 1))
        var, mean = torch.var_mean(x, dim=dims, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return out * self.weight + self.bias


class _GroupBatchNorm(nn.Module):
    """官方 ``GroupBatchNorm`` 的可用等价实现。

    上游原版在 ``affine=False`` 时引用未定义局部变量，并在 ``seq_last=True``
    下对 ``[B,H,Seq]`` 的 dims 语义有额外假设；本实现只实现
    ``seq_last=False`` 的 ``[B,H,F,T]`` 路径（``dims_norm`` 显式给出时按显式
    维度归一化），并在其它组合上 fail-fast，不静默退化。
    """

    def __init__(
        self,
        dim_hidden: int,
        group_size: int | None = None,
        *,
        share_along_sequence_dim: bool = False,
        seq_last: bool = False,
        affine: bool = True,
        eps: float = 1e-5,
        dims_norm: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if seq_last:
            raise ValueError("GroupBatchNorm seq_last=True is not implemented")
        if dims_norm is not None:
            raise NotImplementedError("GroupBatchNorm dims_norm is not implemented")
        if affine:
            self.weight = nn.Parameter(torch.ones(dim_hidden))
            self.bias = nn.Parameter(torch.zeros(dim_hidden))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.dim_hidden = dim_hidden
        self.group_size = group_size
        self.eps = eps
        self.affine = affine
        self.seq_last = seq_last
        self.share_along_sequence_dim = share_along_sequence_dim

    def forward(self, x: Tensor, group_size: int | None = None) -> Tensor:
        size = self.group_size if self.group_size is not None else group_size
        if size is None:
            raise ValueError("GroupBatchNorm requires a group size")
        original = x.shape
        if x.ndim == 4:
            batch, _, _, _ = x.shape
            x = x.reshape(batch // size, size, *x.shape[1:])
        elif x.ndim != 3:
            raise ValueError("GroupBatchNorm expects [B,H,F,T] or [B,H,F]")
        if self.share_along_sequence_dim:
            var, mean = torch.var_mean(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
        else:
            var, mean = torch.var_mean(x, dim=(1, 3), unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            out = out * self.weight + self.bias
        return out.reshape(original)


def _new_norm(
    norm_type: str,
    dim_hidden: int,
    *,
    seq_last: bool,
    num_groups: int | None = None,
    group_size: int | None = None,
) -> nn.Module:
    key = norm_type if norm_type == "GBNShare" else norm_type.upper()
    if key == "LN":
        return _LayerNorm(seq_last=seq_last, normalized_shape=dim_hidden)
    if key == "BN":
        return _BatchNorm1d(seq_last=seq_last, num_features=dim_hidden)
    if key == "GN":
        if num_groups is None:
            raise ValueError("GroupNorm requires num_groups (conv_groups[1])")
        return _GroupNorm(seq_last=seq_last, num_groups=num_groups, num_channels=dim_hidden)
    if key == "GBN":
        return _GroupBatchNorm(
            dim_hidden, group_size=group_size, seq_last=seq_last,
            share_along_sequence_dim=False,
        )
    if key == "GBNSHARE":
        return _GroupBatchNorm(
            dim_hidden, group_size=group_size, seq_last=seq_last,
            share_along_sequence_dim=True,
        )
    if key == "GLN":
        return _GlobalLayerNorm(dim_hidden, seq_last=seq_last)
    raise ValueError(f"unknown norm type: {norm_type!r}")


def _prelu(dim: int, channels: int) -> nn.Module:
    """官方 ``new_non_linear('PReLU', dim_hidden, seq_last)`` 的等价实现。

    官方 ``PReLU(dim=-1, seq_last=True)`` 直接返回 ``nn.PReLU(dim_hidden)``，
    即作用在 ``dim=1``（H 轴）；``dim=-1``（seq_last=False）时官方先转置到
    dim=1 再转回。两种情形归一为「在 p.dim 指定的轴上按 H 逐通道激活」。
    """
    class _PReLU(nn.Module):
        # 参数名与官方 ``nn.PReLU`` 一致（``weight``），保证官方 checkpoint 可加载。
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.full((channels,), 0.25))

        def forward(self, value: Tensor) -> Tensor:
            shape = [1] * value.dim()
            shape[dim] = channels
            return torch.where(value > 0, value, self.weight.view(shape) * value)

    return _PReLU()


class CausalConv1d(nn.Conv1d):
    """官方 ``CausalConv1d``：左 pad ``kernel-1-look_ahead``、右 pad ``look_ahead``。

    只移植并行前向（``state=None``）分支；逐帧 streaming 分支（依赖
    ``state`` 字典缓存左上下文）不在本轮范围内，调用时显式报错。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        look_ahead: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.look_ahead = look_ahead
        if type(look_ahead) is not int or look_ahead < 0 or look_ahead > kernel_size - 1:
            raise ValueError("look_ahead must be an integer in [0, kernel_size-1]")

    def forward(self, x: Tensor, state: dict[int, Any] | None = None) -> Tensor:  # type: ignore[override]
        if state is not None:
            raise NotImplementedError("CausalConv1d streaming state is not implemented")
        x = F.pad(x, (self.kernel_size[0] - 1 - self.look_ahead, self.look_ahead))
        return super().forward(x)


class _SpatialNetLayer(nn.Module):
    """官方 ``SpatialNetLayer``：fconv1 → (layer0 频率折半) → full → fconv2 → 序列槽。

    序列槽位语义由 ``attention`` 决定：``mamba(...)`` 时 ``mhsa`` 与 ``tconvffn``
    都是 ``Mamba``；``alternating`` 时 ``mhsa`` 用官方 ``MultiheadAttention``
    （``need_weights=True``，对 ``[B,F,T,H]`` 直接做帧注意力），``tconvffn`` 用
    官方 ``_tconvffn`` 的 T-Conv/FFN 形态（``CausalConv1d`` + ``PReLU`` +
    ``Conv1d(1x1)``）。上游在非 mamba 配置下还会退到
    ``MultiScaleRetention``（未使用的遗留死代码），本实现不移植。
    """

    def __init__(
        self,
        *,
        dim_hidden: int,
        dim_squeeze: int,
        num_freqs: int,
        arrangement: str,
        mamba_d_state: int,
        mamba_d_conv: int,
        num_heads: int,
        dropout: tuple[float, float, float] = (0.0, 0.0, 0.0),
        kernel_size: tuple[int, int] = (5, 3),
        conv_groups: tuple[int, int] = (8, 8),
        norms: tuple[str, ...] = ("LN", "LN", "GN", "LN", "LN", "LN"),
        is_first: bool = False,
    ) -> None:
        super().__init__()
        if type(is_first) is not bool:
            raise TypeError("SpatialNetLayer is_first must be a bool")
        f_conv_groups, t_conv_groups = conv_groups
        f_kernel_size = kernel_size[0]
        self.dim_hidden = dim_hidden
        self.dim_squeeze = dim_squeeze
        self.num_freqs = num_freqs
        self.is_first = is_first
        self.attention = arrangement

        self.fconv1 = nn.ModuleList([
            _new_norm(
                norms[3], dim_hidden, seq_last=True,
                group_size=None, num_groups=f_conv_groups,
            ),
            self._fpath_conv(dim_hidden, dim_hidden, f_kernel_size, f_conv_groups),
            _prelu(1, dim_hidden),
        ])
        self.norm_full = _new_norm(
            norms[5], dim_hidden, seq_last=False,
            group_size=None, num_groups=f_conv_groups,
        )
        self.squeeze = nn.Sequential(
            nn.Conv1d(dim_hidden, dim_squeeze, kernel_size=1), nn.SiLU()
        )
        self.full = nn.Linear(num_freqs, num_freqs)
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.unsqueeze = nn.Sequential(
            nn.Conv1d(dim_squeeze, dim_hidden, kernel_size=1), nn.SiLU()
        )
        self.fconv2 = nn.ModuleList([
            _new_norm(
                norms[4], dim_hidden, seq_last=True,
                group_size=None, num_groups=f_conv_groups,
            ),
            self._fpath_conv(dim_hidden, dim_hidden, f_kernel_size, f_conv_groups),
            _prelu(1, dim_hidden),
        ])
        self.fre_compress_first = nn.AvgPool2d(kernel_size=(1, 2))
        self.fre_compress_second = nn.AvgPool2d(kernel_size=(1, 8))

        self.norm_mhsa = _new_norm(
            norms[0], dim_hidden, seq_last=False,
            group_size=None, num_groups=t_conv_groups,
        )
        self.dropout_mhsa = nn.Dropout(dropout[0])
        self.norm_tconvffn = _new_norm(
            norms[1], dim_hidden, seq_last=False,
            group_size=None, num_groups=t_conv_groups,
        )
        self.dropout_tconvffn = nn.Dropout(dropout[1])

        mamba_cls = _require_mamba()
        self.mhsa: nn.Module
        self.tconvffn: nn.Module
        if arrangement == "mamba_only":
            # 官方 Mamba 调用不传 layer_idx，落盘键不含 layer_idx。
            self.mhsa = mamba_cls(
                d_model=dim_hidden, d_state=mamba_d_state, d_conv=mamba_d_conv
            )
            self.tconvffn = mamba_cls(
                d_model=dim_hidden, d_state=mamba_d_state, d_conv=mamba_d_conv
            )
        else:
            if dim_hidden % num_heads:
                raise ValueError("dim_hidden must be divisible by num_heads for frame attention")
            self.mhsa = nn.MultiheadAttention(
                embed_dim=dim_hidden, num_heads=num_heads, batch_first=True
            )
            self.tconvffn = nn.Sequential(
                _new_norm(norms[2], dim_hidden, seq_last=True, num_groups=t_conv_groups),
                CausalConv1d(dim_hidden, dim_hidden, kernel_size[1], groups=t_conv_groups),
                _prelu(1, dim_hidden),
                nn.Conv1d(dim_hidden, dim_hidden, kernel_size=1),
            )

    @staticmethod
    def _fpath_conv(
        in_channels: int, out_channels: int, kernel_size: int, groups: int
    ) -> nn.Module:
        """官方 fconv 卷积：``padding='same'`` + ``padding_mode='zeros'``，分组=H。

        频率轴卷积**不**因果化：官方在 ``[B*T,H,F]`` 上对 ``F`` 做对称 same
        padding，本实现保持同一语义（``padding_mode='zeros'`` 在现代 torch 中
        与默认行为等价）。
        """
        if in_channels % groups or out_channels % groups:
            raise ValueError(
                "fconv groups must divide dim_hidden: official conv_groups=(8,8) "
                "requires dim_hidden % 8 == 0"
            )
        return nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            groups=groups,
            padding="same",
            padding_mode="zeros",
        )

    def _fconv(self, module_list: nn.ModuleList, x: Tensor, sequence: int) -> Tensor:
        """官方 ``_fconv``：``[B,F,T,H]`` → ``[B*T,H,F]``，在频率轴 ``F`` 上卷积。

        ``LayerNorm(seq_last=True)`` 在该布局下 ``transpose(-1, 1)`` 得
        ``[B*T,F,H]``，归一化轴是 H；``Conv1d(groups=...)`` 的分组轴同样是 H。
        官方把 ``T`` 合并进 batch，因此频率卷积**不跨帧共享**。
        """
        batch, freqs, frames, hidden = x.shape
        value = x.permute(0, 2, 3, 1).reshape(batch * frames, hidden, freqs)
        for module in module_list:
            if isinstance(module, _GroupBatchNorm):
                value = module(value, group_size=sequence)
            else:
                value = module(value)
        value = value.reshape(batch, frames, hidden, freqs)
        return value.permute(0, 3, 1, 2)

    def _full(self, x: Tensor) -> Tensor:
        """官方 full-band 槽：逐频点通道压缩 + 全带频率线性混合（偏离点 2）。"""
        batch, freqs, frames, hidden = x.shape
        value = self.norm_full(x)
        value = value.permute(0, 2, 3, 1).reshape(batch * frames, hidden, freqs)
        value = self.squeeze(value)
        if self.dropout_full is not None:
            value = value.reshape(batch, frames, -1, freqs)
            value = value.transpose(1, 3)
            value = self.dropout_full(value)
            value = value.transpose(1, 3)
            value = value.reshape(batch * frames, -1, freqs)
        value = self.full(value)
        value = self.unsqueeze(value)
        value = value.reshape(batch, frames, hidden, freqs)
        return value.permute(0, 3, 1, 2)

    def _mamba(self, x: Tensor, module: nn.Module, norm: nn.Module, dropout: nn.Module) -> Tensor:
        batch = x.shape[0]
        value = norm(x)
        value = value.reshape(batch * x.shape[1], x.shape[2], x.shape[3])
        value = module(value)
        value = value.reshape(*x.shape)
        return dropout(value)

    def _frame_attention(
        self, x: Tensor, att_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        batch = x.shape[0]
        value = self.norm_mhsa(x)
        value = value.reshape(batch * x.shape[1], x.shape[2], x.shape[3])
        value, weights = self.mhsa(
            value, value, value, attn_mask=att_mask, need_weights=True,
            average_attn_weights=False,
        )
        value = value.reshape(*x.shape)
        return self.dropout_mhsa(value), weights

    def _tconvffn(self, x: Tensor) -> Tensor:
        batch, freqs, frames, hidden = x.shape
        value = x.transpose(-1, -2).reshape(batch * freqs, hidden, frames)
        for module in self.tconvffn:
            if isinstance(module, _GroupNorm):
                value = value.reshape(batch, freqs, -1, frames)
                value = value.transpose(1, -1).reshape(batch * frames, -1, freqs)
                value = module(value)
                value = value.reshape(batch, frames, -1, freqs)
                value = value.transpose(1, -1).reshape(batch * freqs, -1, frames)
            else:
                value = module(value)
        value = value.reshape(batch, freqs, hidden, frames)
        return self.dropout_tconvffn(value.transpose(-1, -2))

    def forward(
        self,
        x: Tensor,
        att_mask: Tensor | None = None,
        *,
        return_attn_score: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        x = x + self._fconv(self.fconv1, x, x.shape[2])
        if self.is_first:
            x = self.fre_compress_first(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self._full(x)
        x = x + self._fconv(self.fconv2, x, x.shape[2])
        if self.is_first:
            x = self.fre_compress_second(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attention = None
        if self.attention == "mamba_only":
            x = x + self._mamba(x, self.mhsa, self.norm_mhsa, self.dropout_mhsa)
        else:
            value, attention = self._frame_attention(x, att_mask)
            x = x + value
        if self.attention == "mamba_only":
            x = x + self._mamba(x, self.tconvffn, self.norm_tconvffn, self.dropout_tconvffn)
        else:
            x = x + self._tconvffn(x)
        return x, attention


class _FreqInverse(nn.Module):
    """官方 ``FreqInverse``：把压缩频率维调回 ``nfreq`` 并做 ``tanh``。

    官方实现保留逐频带拷贝，因此要求 ``time_step`` 能被
    ``nfreq // compression_ratio`` 整除；本实现改为一次性 reshape
    （数学等价且无该整除要求），并在 ``nfreq % compression_ratio == 0`` 上
    fail-fast。``tanh`` 后的输出仍是官方 ``[B, out_dim, T, nfreq]`` 布局。
    """

    def __init__(self, nfreq: int, compression_ratio: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        if nfreq % compression_ratio:
            raise ValueError("num_freqs must be divisible by fre_compression_ratio")
        self.nfreq = nfreq
        self.compression_ratio = compression_ratio
        self.out_dim = out_dim
        self.nfilters = nfreq // compression_ratio
        self.trans2 = nn.Conv1d(hidden_dim, compression_ratio * out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """``[B,H,T,nfilters]`` → ``[B,out_dim,T,nfreq]``（tanh 后）。

        ``trans2`` 是 1x1 ``Conv1d``，与官方一样在 ``[B,H,...]`` 的通道轴 ``H``
        上作用，因此需要把压缩频带轴放到卷积的序列轴上：官方逐频带
        ``self.trans2(x[:,:,:,i])`` 等价于一次性处理 ``[B,H,T*nfilters]``。
        """
        if x.ndim != 4:
            raise ValueError("FreqInverse expects [B,H,T,nfilters]")
        if x.shape[-1] != self.nfilters:
            raise ValueError(
                f"FreqInverse expects {self.nfilters} compressed frequency bands, got {x.shape[-1]}"
            )
        batch, hidden, frames, bands = x.shape
        value = self.trans2(x.reshape(batch, hidden, frames * bands))
        # 逐频带去卷积：每个压缩频带产生 compression_ratio 个相邻频率点，
        # 官方按 band 顺序沿频率维拼接，故 band 在前、ratio 在后。
        value = value.reshape(
            batch, self.out_dim, self.compression_ratio, frames, bands
        )
        value = value.permute(0, 1, 3, 4, 2).reshape(
            batch, self.out_dim, frames, self.nfreq
        )
        return value.contiguous().tanh()


def _require_mamba():
    """optional import guard：缺失 ``mamba_ssm`` 时构造期显式报错。"""
    try:
        from mamba_ssm import Mamba
    except ImportError as exc:  # pragma: no cover - 依赖存在时不触发
        raise ImportError(
            "IPDNet2 requires the optional dependency 'mamba-ssm' "
            "(and its 'causal-conv1d' runtime); install with "
            "`pip install mamba-ssm causal-conv1d`. "
            "The official oSpatialNet backbone has no GRU/LSTM fallback."
        ) from exc
    return Mamba


@MODEL_REGISTRY.register("IPDNet2")
class IPDNet2(_PairwiseIPDModel):
    """IPDnet2 官方 ``OnlineSpatialNet``（oSpatialNet + Mamba）结构。

    结构、输入输出边界与偏离点见模块 docstring。本类不消费
    ``source_distances``：官方 near-field 合同只把声源距离用于 target 生成，
    推理侧由 ``near_field_doa_range_grid`` mapping 在模型外完成解码。
    """

    MODEL_TYPE = "IPDNet2"
    MODEL_SPEC = ModelSpec(
        model_type=MODEL_TYPE,
        feature_contract="multichannel_stft_real_imag",
        target_contract="dpipd_pair",
        pair_layout="reference_to_all",
        geometry_mode="variable_array",
        field_mode="near_field",
        reduction="mamba_time_pool",
        causal=True,
        mapping="near_field_doa_range_grid",
        target_layout="trailing_ri",
    )
    IO_CONTRACT = _PairwiseIPDModel.IO_CONTRACT

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 96,
        max_track: int = 2,
        n_fft: int = 512,
        stride: int = 5,
        causal: bool = True,
        num_freqs: int | None = None,
        num_layers: int = 8,
        num_heads: int = 4,
        dim_squeeze: int = 8,
        encoder_kernel_size: int = 5,
        kernel_size: tuple[int, int] = (5, 3),
        conv_groups: tuple[int, int] = (8, 8),
        norms: tuple[str, ...] = ("LN", "LN", "GN", "LN", "LN", "LN"),
        attention: str = "mamba(16,4)",
        dropout: tuple[float, float, float] = (0.0, 0.0, 0.0),
        fre_compression_ratio: int = 16,
        time_compression_ratio: int = 5,
        time_compression_layer: int = 0,
        arrangement: str = "mamba_only",
        mamba_d_state: int | None = None,
        mamba_d_conv: int | None = None,
        reference_channel: int = 0,
        output_scale: float = 1.0,
        output_chunk_frames: int = 4000,
        forgetting_norm_sample_length: int | None = 249,
        forgetting_norm_eps: float = 1e-6,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str | None = None,
        center: bool | None = None,
        exclude_dc: bool | None = None,
        **kwargs: Any,
    ) -> None:
        # 只继承 ``_PairwiseIPDModel`` 的 IO_CONTRACT 与子类约定，不继承它的
        # GRU 代理骨干；这里直接初始化 ``nn.Module``，避免注册多余的
        # ``temporal``/``head`` 参数污染与官方 checkpoint 的 state_dict 对齐。
        nn.Module.__init__(self)
        if kwargs:
            raise TypeError(f"unexpected model arguments: {sorted(kwargs)}")
        for name, value, minimum in (
            ("input_size", input_size, 1),
            ("hidden_size", hidden_size, 2),
            ("max_track", max_track, 1),
            ("n_fft", n_fft, 2),
            ("stride", stride, 1),
            ("num_layers", num_layers, 1),
            ("num_heads", num_heads, 1),
            ("dim_squeeze", dim_squeeze, 1),
            ("encoder_kernel_size", encoder_kernel_size, 1),
            ("fre_compression_ratio", fre_compression_ratio, 1),
            ("time_compression_ratio", time_compression_ratio, 1),
            ("output_chunk_frames", output_chunk_frames, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an exact integer >= {minimum}")
        if type(causal) is not bool or not causal:
            raise ValueError("IPDNet2 official OnlineSpatialNet is causal; set causal=true")
        if type(stride) is not int or stride != time_compression_ratio:
            raise ValueError(
                "stride must equal time_compression_ratio: the official time pooling "
                "downsamples frames by AvgPool2d(time_compression_ratio, 1)"
            )
        if output_chunk_frames % time_compression_ratio:
            raise ValueError(
                "output_chunk_frames must be divisible by time_compression_ratio: "
                "chunked IPDNet2 inference cannot silently drop a partial pool window"
            )
        if not isinstance(time_compression_layer, int) or not 0 <= time_compression_layer < num_layers:
            raise ValueError("time_compression_layer must be an integer in [0, num_layers)")
        if not isinstance(kernel_size, (tuple, list)) or len(kernel_size) != 2:
            raise ValueError("kernel_size must be a (frequency, time) pair")
        if not isinstance(conv_groups, (tuple, list)) or len(conv_groups) != 2:
            raise ValueError("conv_groups must be a (frequency, time) pair")
        if not isinstance(norms, (tuple, list)) or len(norms) != 6:
            raise ValueError("norms must contain the six official norm slots")
        for norm in norms:
            if norm not in _NORMS:
                raise ValueError(f"unsupported norm type: {norm!r}")
        if not isinstance(dropout, (tuple, list)) or len(dropout) != 3:
            raise ValueError("dropout must be a (attention, tconvffn, full) triple")
        for value in dropout:
            if not 0.0 <= float(value) < 1.0:
                raise ValueError("dropout entries must be in [0,1)")
        if type(reference_channel) is not int or reference_channel < 0:
            raise ValueError("reference_channel must be a non-negative integer")
        if isinstance(output_scale, bool) or not isinstance(output_scale, (int, float)):
            raise TypeError("output_scale must be a finite number")
        if not math.isfinite(float(output_scale)):
            raise ValueError("output_scale must be finite")
        if forgetting_norm_sample_length is not None:
            if type(forgetting_norm_sample_length) is not int or forgetting_norm_sample_length < 1:
                raise ValueError("forgetting_norm_sample_length must be a positive integer or None")
        if isinstance(forgetting_norm_eps, bool) or not isinstance(forgetting_norm_eps, (int, float)):
            raise TypeError("forgetting_norm_eps must be a finite non-negative number")
        if not math.isfinite(float(forgetting_norm_eps)) or float(forgetting_norm_eps) < 0.0:
            raise ValueError("forgetting_norm_eps must be finite and non-negative")
        if hop_length is not None and (type(hop_length) is not int or hop_length < 1):
            raise ValueError("hop_length must be a positive exact integer")
        if win_length is not None and (type(win_length) is not int or not 1 <= win_length <= n_fft):
            raise ValueError("win_length must satisfy 1 <= win_length <= n_fft")
        if window is not None and window not in {"hann", "hamming"}:
            raise ValueError("window must be hann or hamming")
        if center is not None and type(center) is not bool:
            raise TypeError("center must be a bool")
        if exclude_dc is not None and type(exclude_dc) is not bool:
            raise TypeError("exclude_dc must be a bool")
        if arrangement not in _ARRANGEMENTS:
            raise ValueError(f"arrangement must be one of {sorted(_ARRANGEMENTS)!r}")

        # 官方 attention 字符串：attention[6:-1].split(',') -> (d_state, d_conv)。
        parsed = _parse_attention(attention)
        if parsed is None:
            raise ValueError(
                "attention must use the official 'mamba(d_state,d_conv)' format"
            )
        attention_state, attention_conv = parsed
        if attention_state < 1 or attention_conv < 1:
            raise ValueError(
                "mamba d_state and d_conv must be positive integers"
            )
        if mamba_d_state is not None and mamba_d_state != attention_state:
            raise ValueError("mamba_d_state conflicts with the attention string")
        if mamba_d_conv is not None and mamba_d_conv != attention_conv:
            raise ValueError("mamba_d_conv conflicts with the attention string")
        mamba_d_state = attention_state
        mamba_d_conv = attention_conv
        if arrangement == "alternating" and mamba_d_state <= 0:
            raise ValueError("mamba d_state must be a positive integer")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_track = max_track
        self.n_fft = n_fft
        self.stride = stride
        self.causal = causal
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.fre_compression_ratio = fre_compression_ratio
        self.time_compression_ratio = time_compression_ratio
        self.time_compression_layer = time_compression_layer
        self.arrangement = arrangement
        self.attention = attention
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.reference_channel = reference_channel
        self.output_scale = float(output_scale)
        self.output_chunk_frames = output_chunk_frames
        self.forgetting_norm_sample_length = forgetting_norm_sample_length
        self.forgetting_norm_eps = float(forgetting_norm_eps)
        self.hop_length, self.win_length = hop_length, win_length
        self.window, self.center = window, center
        self.exclude_dc = exclude_dc
        self.norms = tuple(norms)

        # 官方 dim_input = 2C = 2*(input_size+1)，dim_output = 2*(C-1)*max_track。
        self.num_mics = input_size + 1
        self.dim_input = 2 * self.num_mics
        self.dim_output = 2 * input_size * max_track
        # 官方 run_IPDnet2.py 的 num_freqs 是删 DC 后的频点数；
        # n_fft//2+1 是未删 DC 的完整频点数，作为非 exclude_dc 的默认值。
        self.num_freqs = n_fft // 2 if num_freqs is None else num_freqs
        # 完整（可能含 DC）频率轴的点数。
        self.full_freqs = n_fft // 2 + 1
        if type(self.num_freqs) is not int or self.num_freqs < 1:
            raise ValueError("num_freqs must be a positive exact integer")
        if exclude_dc is False and self.num_freqs != self.full_freqs:
            raise ValueError(
                "exclude_dc=false requires num_freqs == n_fft//2+1; "
                "the configured frequency axis is DC-excluded"
            )
        if exclude_dc is True and self.num_freqs != self.full_freqs - 1:
            raise ValueError(
                "exclude_dc=true requires num_freqs == n_fft//2; "
                "the configured frequency axis must contain exactly the non-DC bins"
            )
        # 官方 layer0 的 AvgPool2d(1,2) 要求折半前频率数为偶数，AvgPool2d(1,8)
        # 要求折半后频率数能被 8 整除；FreqInverse 把 8 个压缩频带逆回
        # 8*fre_compression_ratio 个频点，因此 num_freqs 必须严格等于
        # (num_freqs//2//8) * fre_compression_ratio。默认 num_freqs=n_fft//2
        # （删除 DC，官方口径）时三条约束自动成立。
        if self.num_freqs % 2:
            raise ValueError("num_freqs must be even for the official layer0 AvgPool2d(1,2)")
        self.freq_after_fold2 = self.num_freqs // 2
        if self.freq_after_fold2 % 8:
            raise ValueError(
                "num_freqs//2 must be divisible by 8 for the official layer0 AvgPool2d(1,8)"
            )
        self.freq_after_fold8 = self.freq_after_fold2 // 8
        if self.freq_after_fold8 * fre_compression_ratio != self.num_freqs:
            raise ValueError(
                "num_freqs must equal (num_freqs//2//8) * fre_compression_ratio: the "
                "official FreqInverse expands 8 compressed bands back to "
                "8*fre_compression_ratio frequency bins"
            )

        self.encoder = CausalConv1d(
            self.dim_input, hidden_size, encoder_kernel_size, look_ahead=0
        )
        layers = []
        for index in range(num_layers):
            layers.append(
                _SpatialNetLayer(
                    dim_hidden=hidden_size,
                    dim_squeeze=dim_squeeze,
                    num_freqs=(self.freq_after_fold2 if index == 0 else self.freq_after_fold8),
                    arrangement=arrangement,
                    mamba_d_state=mamba_d_state,
                    mamba_d_conv=mamba_d_conv,
                    num_heads=num_heads,
                    dropout=tuple(float(value) for value in dropout),
                    kernel_size=(int(kernel_size[0]), int(kernel_size[1])),
                    conv_groups=(int(conv_groups[0]), int(conv_groups[1])),
                    norms=tuple(norms),
                    is_first=(index == 0),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.freq_inverse = _FreqInverse(
            nfreq=self.num_freqs,
            compression_ratio=fre_compression_ratio,
            hidden_dim=hidden_size,
            out_dim=self.dim_output,
        )
        self.decoder = nn.Linear(in_features=self.dim_output, out_features=self.dim_output)
        self.time_pooling = nn.AvgPool2d(
            kernel_size=(time_compression_ratio, 1)
        )

    # ------------------------------------------------------------------ 输入边界
    def _official_layout(self, value: Tensor) -> Tensor:
        """框架 canonical ``[B,T,F,2C]`` → 官方 ``[B,2C,F,T]``。

        框架特征末维顺序是 ``[c0_re, c0_im, c1_re, c1_im, ...]``；官方是
        ``[re(c0), re(c1), ..., im(c0), im(c1), ...]``（先全部 real 再全部 imag）。
        这里先把显式 ``reference_channel`` 放到通道首位，再按通道显式重排，
        禁止用 view/reshape 隐式解释。
        """
        if value.ndim == 5 and value.shape[-1] == 2:
            mic = value.shape[3]
            value = value.reshape(value.shape[0], value.shape[1], value.shape[2], mic * 2)
        if value.ndim != 4:
            raise ValueError("multichannel STFT input must have shape [B,T,F,2C]")
        batch, frames, freqs, features = value.shape
        if features != self.dim_input:
            raise ValueError(
                f"multichannel STFT feature dimension {features} != configured 2C={self.dim_input}"
            )
        if freqs not in {self.full_freqs, self.num_freqs}:
            raise ValueError(
                f"multichannel STFT frequency bins {freqs} must be "
                f"{self.num_freqs} (exclude_dc) or {self.full_freqs} (full band)"
            )
        channels = features // 2
        if not 0 <= self.reference_channel < channels:
            raise ValueError("reference_channel is outside the multichannel STFT channels")
        value = value.reshape(batch, frames, freqs, channels, 2)
        if self.reference_channel:
            order = [
                self.reference_channel,
                *[index for index in range(channels) if index != self.reference_channel],
            ]
            value = value[..., order, :]
        real = value[..., 0]
        imag = value[..., 1]
        official = torch.cat((real, imag), dim=-1)  # [B,T,F,2C] with re-block then im-block
        return official.permute(0, 3, 2, 1).contiguous()

    def _drop_dc(self, value: Tensor) -> Tensor:
        """官方 ``fre_range_used=range(1, n_fft//2+1)``：删 DC 并校验频点数。"""
        freqs = value.shape[2]
        if freqs == self.full_freqs and freqs != self.num_freqs:
            # 官方输入从不含 DC；canonical 多通道特征默认含 DC。
            value = value[:, :, 1:, :]
            freqs = value.shape[2]
        if freqs != self.num_freqs:
            raise ValueError(
                f"IPDnet2 expects {self.num_freqs} frequency bins after DC removal, got {freqs}"
            )
        if self.exclude_dc is False and self.num_freqs == self.full_freqs:
            raise ValueError(
                "exclude_dc=false requires num_freqs == n_fft//2+1 and keeps DC; "
                "the official IPDnet2 recipe removes DC"
            )
        return value

    def _forgetting_normalize(self, value: Tensor) -> Tensor:
        """官方 forgetting normalization：real/imag 同除递推幅度因子。

        与官方 ``utils_.forgetting_norm`` 逐行一致（本地等价实现见
        ``tasks/ssl/models/ipdnet_official.py::forgetting_norm``）：
        ``alpha = (L-1)/(L+1)``，前 ``L`` 帧使用 ``min((t-1)/(t+1), alpha)``、
        之后固定 ``alpha``；递推量是同一帧上 ``C*F`` 个幅度值的均值。
        官方把 real/imag 都除以该因子（推理路径同样启用），故本实现不做
        train/eval 分支，也**不**做单位圆归一化。
        """
        length = self.forgetting_norm_sample_length
        if length is None:
            return value
        alpha = (length - 1) / (length + 1)
        magnitude = value.abs().reshape(value.shape[0], -1, value.shape[-1])
        running = torch.zeros(
            value.shape[0], 1, dtype=value.dtype, device=value.device
        )
        collected = []
        for frame in range(value.shape[-1]):
            local = min((frame - 1) / (frame + 1), alpha) if frame < length else alpha
            running = local * running + (1.0 - local) * magnitude[:, :, frame].mean(
                dim=1, keepdim=True
            )
            collected.append(running)
        norm = torch.stack(collected, dim=-1).reshape(
            value.shape[0], 1, 1, value.shape[-1]
        )
        return value / (norm + self.forgetting_norm_eps)

    # ----------------------------------------------------------------- 前向计算
    def _encode(self, value: Tensor) -> Tensor:
        """官方 encoder：``[B,2C,F,T]`` → ``[B,F,T,H]``。

        官方 ``OnlineSpatialNet.forward`` 先做 ``x = x.permute(0,2,3,1)`` 再
        ``x.reshape(B*F, T, H0).permute(0,2,1)`` 喂给 ``CausalConv1d``。对输入
        ``[B,2C,F,T]``（官方注释 ``[Batch,Freq,Time,Feature]``）而言，该 permute
        得到 ``[B,F,2C,T]``，展平后 encoder 看到的是
        ``[B*F, 2C, T]``：**频率进 batch、序列轴仍是 T、通道是 2C**。
        这里按同一语义显式转置，不做会打乱 (F,T) 的隐式 view。
        """
        batch, channels, freqs, frames = value.shape
        flat = value.permute(0, 2, 3, 1).reshape(batch * freqs, frames, channels)
        encoded = self.encoder(flat.permute(0, 2, 1))
        return encoded.permute(0, 2, 1).reshape(
            batch, freqs, frames, encoded.shape[1]
        )

    def _forward_layers(self, x: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x, _ = layer(x, None, return_attn_score=False)
            if index == self.time_compression_layer:
                batch, freqs, frames, hidden = x.shape
                x = x.reshape(batch * freqs, frames, hidden)
                x = self.time_pooling(x)
                x = x.reshape(batch, freqs, x.shape[1], hidden)
        return x

    def _decode(self, x: Tensor) -> Tensor:
        """``[B,F_c,T_out,H]`` → ``[B,nfreq,T_out,dim_output]``（官方 tanh 输出）。

        官方顺序：``x.permute(0,3,2,1)`` = ``[B,H,T_out,F_c]`` → ``FreqInverse``
        （在 ``H`` 通道上卷积、输出 ``[B,out_dim,T_out,nfreq]``，并已 ``tanh``）→
        ``permute(0,3,2,1)`` = ``[B,nfreq,T_out,out_dim]`` → ``decoder``。
        """
        x = x.permute(0, 3, 2, 1)
        x = self.freq_inverse(x)
        x = x.permute(0, 3, 2, 1)
        return self.decoder(x)


    def _reshape_output(self, value: Tensor) -> Tensor:
        """官方输出 axes 顺序 → 框架 canonical ``[B,T_out,F,K,P,2]``。

        进入本函数时 ``value`` 是官方 post-decoder 张量
        ``[B,num_freqs,T_out,dim_output]``。官方 ``OnlineSpatialNet.forward``
        逐字输出 reshape 为：

        ```text
        x.permute(0,2,1,3).reshape(B,T_out,F,2,-1).permute(0,1,3,2,4)
        x.reshape(B,T_out,2,F*2,-1).permute(0,1,3,4,2)
        ```

        结果即官方 ``[B,T_out,2F,P,K]``（``2F`` 与 P、K 均为官方轴序；golden
        trace 已逐元素核对该 3 步与官方 ``forward`` 完全一致）。框架 canonical
        要求 ``[B,T_out,F,K,P,2]``，末维 2 = 复数实部、虚部，因此这里先复刻官方
        3 步 reshape 得到 ``[B,T_out,2F,P,K]``，再把 ``2F`` 拆成
        ``(ri=2, F)``、把 ``(P,K)`` 转成 ``(K,P)``，末维固定为实虚部。
        """
        if value.ndim != 4:
            raise ValueError("decoded IPDnet2 output must have shape [B,F,T,dim_output]")
        batch, freqs, frames, dim_output = value.shape
        if freqs != self.num_freqs:
            raise ValueError(
                f"decoded IPDnet2 frequency bins {freqs} != configured num_freqs "
                f"{self.num_freqs}"
            )
        pairs = self.input_size
        tracks = self.max_track
        if dim_output != 2 * tracks * pairs:
            raise ValueError(
                f"decoded IPDnet2 feature dim {dim_output} != 2*max_track*input_size"
            )
        # 官方 3 步 reshape，逐字复刻。
        official = value.permute(0, 2, 1, 3).reshape(
            batch, frames, freqs, 2, -1
        ).permute(0, 1, 3, 2, 4)
        official = official.reshape(batch, frames, 2, freqs * 2, -1).permute(
            0, 1, 3, 4, 2
        )
        # official: [B,T_out,2F,P,K] -> 拆 2F 为 (ri,F) -> [B,T_out,ri,F,P,K]
        split = official.reshape(batch, frames, 2, freqs, pairs, tracks)
        # -> [B,T_out,F,K,P,ri]（末维 = 实部、虚部）
        canonical = split.permute(0, 1, 3, 5, 4, 2)
        return (canonical * self.output_scale).contiguous()

    def forward(
        self,
        multichannel_stft_real_imag: Tensor | None = None,
        *,
        features: Tensor | None = None,
    ) -> Tensor:
        if multichannel_stft_real_imag is None:
            if features is None:
                raise TypeError("missing required input: multichannel_stft_real_imag")
            multichannel_stft_real_imag = features
        elif features is not None:
            raise ValueError("provide only multichannel_stft_real_imag or features")
        value = multichannel_stft_real_imag
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError("multichannel STFT real/imag input must be finite floating point")
        official = self._official_layout(value)
        official = self._forgetting_normalize(official)
        official = self._drop_dc(official)
        frames = official.shape[-1]
        if frames % self.time_compression_ratio:
            raise ValueError(
                f"input frame count {frames} must be divisible by "
                f"time_compression_ratio={self.time_compression_ratio}"
            )
        chunk = self.output_chunk_frames
        outputs = []
        for start in range(0, frames, chunk):
            piece = official[:, :, :, start : start + chunk]
            if piece.shape[-1] % self.time_compression_ratio:
                piece = piece[:, :, :, : piece.shape[-1] - (piece.shape[-1] % self.time_compression_ratio)]
            if piece.shape[-1] == 0:
                continue
            encoded = self._encode(piece)
            decoded = self._decode(self._forward_layers(encoded))
            outputs.append(self._reshape_output(decoded))
        if not outputs:
            raise ValueError("IPDnet2 received an empty time axis")
        return torch.cat(outputs, dim=1).contiguous()


def _parse_attention(attention: str) -> tuple[int, int] | None:
    """按官方 ``attention[6:-1].split(',')`` 解析 ``mamba(d_state,d_conv)``。"""
    if not isinstance(attention, str) or len(attention) < 8 or attention[5] != "(":
        return None
    if not attention.lower().startswith("mamba("):
        return None
    if attention[-1] != ")":
        return None
    parts = attention[6:-1].split(",")
    if len(parts) != 2:
        return None
    try:
        state, conv = int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        return None
    return state, conv
