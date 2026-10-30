"""官方 IPDnet fixed / variable 阵列结构接入。

本模块是 Audio-WestlakeU/FN-SSL 仓库 ``IPDnet/FixedAarryIPDnet.py``（fixed）
与 ``IPDnet/VariableArrayIPDnet.py``（variable）两个结构的忠实移植，替代
``tasks/ssl/models/ipdnet.py`` 中原本只抛 ``NotImplementedError`` 的合同占位类。
移植以 raw 原文为准（核对来源见文件末「官方来源与核对记录」）。

官方结构事实（与本地实现的对应关系）
------------------------------------

``IPDnet = FNblock × 2 + CausCnnBlock``：

- ``FNblock``：full-band 双向 LSTM（沿频率轴，恒双向）+ narrow-band LSTM
  （沿时间轴，方向性由 ``is_online`` 决定）+ 两条 skip 连接；
- ``CausCnnBlock``：3 × Conv2d(3×3) + 输出左移因果截断 + AvgPool2d((1,3)) /
  ((1,4))，时间压缩共 12 倍，``tanh`` 输出。

online / offline 的真实区分点是 narrow-band LSTM 的方向性
（``bidirectional = not is_online``）与归一化（On＝forgetting normalization，
Off＝全局均值），**不是** ``CausCnnBlock``：官方把 causal convolution 写死在两个
runner 共用的 ``CausCnnBlock`` 里，且 ``runIPDnetOff.py`` 也只是把
``IPDnet(...)`` 用默认参数构造，因此它表达的仍是 causal 卷积。本模块如实保留该
结构：``CausCnnBlock`` 始终因果，online/offline 只切换 narrow-band LSTM 方向。

输入通道顺序（关键适配点）
--------------------------

框架 canonical 输入为 ``features [B,T,F,2C]``
（``multichannel_stft_real_imag``）。``tasks/ssl/data/projection.py::_raw_stft_real_imag``
由 ``torch.view_as_real`` 产生，因此最后一维是**逐麦交错的 ``(real, imag)``**：
``[m0_re, m0_im, m1_re, m1_im, ...]``。

官方 runner（``IPDnet/runIPDnetOn.py::data_preprocess``）构造的却是
「real 通道块在前、imag 通道块在后」的 ``[r_0..r_{C-1}, i_0..i_{C-1}]``，再交给
模型；模型内部按 ``is_first`` 的 ``input_size`` 直接消费该排布。论文式 (10) 写作
逐麦交错的 ``[ℛ(X_1),ℐ(X_1),…,ℛ(X_M),ℐ(X_M)]``，与发布代码的实际排布不同。
**fixed 模型以发布代码为准**（它带有官方 runner，是决定权重的实际路径），在模型
入口显式把 canonical 的逐麦交错重排为官方的 real 块 / imag 块，并与
``_OfficialFNSSL._pair_batch`` 使用同一约定。

**variable 模型没有官方 runner**，因此 pair 内 4 通道的排布在原始代码里没有唯一
事实来源：论文式 (14) 写作逐麦交错 ``[Re(X_1), ℐ(X_1), Re(X_m), Im(X_m)]``，
而 ``VariableArrayIPDnet.py`` 的 ``__main__`` 注释写作 ``2-channel[imag/real]``。
本模块以论文原文为准把默认值设为逐麦交错，并用构造参数
``pair_channel_order`` 显式开放另一种排布，不允许静默假设。

输出布局（官方 → canonical 的转换关系）
----------------------------------------

官方 fixed 输出（``FixedAarryIPDnet.py::IPDnet.forward``）经

``conv(x).permute(0,3,2,1).reshape(nb,nt2,nf,2,-1).permute(0,1,3,2,4)``
``.reshape(nb,nt2,2,nf*2,-1).permute(0,1,3,4,2)``

得到 ``[B, T/12, 2F, K*P/2, 2]``。该 ``2F`` 轴**不是**逐频的 ``(real, imag)``
配对，而是把卷积输出通道按 ``(k 外层, p 中层, ridx 内层)`` 打包后，与频率轴做了
一次 row-major 展开（经单元测试逐元素确认的对应关系见
``_official_to_canonical``）。因此本模块不做「先 reshape 再猜轴」的处理，而是
**完整执行官方那条 reshape 链**得到官方张量，再做一次有据可查、可逐元素验证的
轴置换，转成框架 canonical ``[B, T_out, F, K, P, 2]``（末尾 2 = 复数实虚部）。

时间轴
------

官方 ``CausCnnBlock`` 的时间压缩固定 12 倍
（``Conv2d(3,3)`` 的 ``padding=(1,2)`` + 左移截断 + ``AvgPool((1,3))`` +
``AvgPool((1,4))``）。构造参数 ``stride`` 必须等于 12，输出时间轴为
``T_in // 12``，与 ``FNSSL`` 的既有做法一致。已逐元素验证：只要 ``T_in >= 12``，
输出时间轴恰为 ``T_in // 12``（官方池化组按首帧左对齐，与本框架 projection 的
左对齐平均池化时间基准一致）。

上游缺陷与本地偏离（必须登记）
------------------------------

1. **variable 上游不可直接运行**：``VariableArrayIPDnet.py::FNblock_mean`` 把
   ``narrLstm`` 声明为 ``input_size=2*full_hidden_size+add_skip_dim``（hidden=128
   时为 132），但其输入来自 ``linear1``，而 ``linear1`` 的输出宽度是
   ``hidden_size``（128）。任意 ``hidden_size`` 下两者都不相等（已实际执行上游
   原始文件复现：``RuntimeError: input.size(-1) must be equal to input_size.
   Expected 132, got 128``）。若改用「把 ``linear1`` 输出改成 132」的另一种修法，
   则 ``block_2`` 的 ``fullLstm(input_size=hidden_size)`` 又会拒绝 132 宽的输入，
   故唯一自洽的修复是让 ``narrLstm`` 消费 ``hidden_size``。本模块采用该修复，
   其余部分与上游逐行对应。

2. **``K=1`` 不是官方可运行配置**：官方 reshape 链要求 ``P * K`` 为偶数
   （``P = C - 1``；``C=2, K=1`` 时 ``cnn_out_dim=2``，``reshape(..., 2, nf*2, -1)``
   无法整除），上游也从未以 ``K=1`` 运行过。本模块在构造期对 fixed 显式拒绝不满足
   该整除条件的 ``max_track``，而不是静默改变输出语义。``K=1`` 的 smoke 由 native
   ``FixedArrayIPDNet`` 承担；官方 variable 则把 ``K=2`` 硬编码进 ``cnn_out_dim=4``。

3. **无 geometry 输入**：与官方一致，两个模型都不接收 geometry；DOA 映射在模型外
   完成。variable 的 pair-sharing 只对**外层 pair 批**做均值并广播，因此不依赖
   阵列拓扑，也不依赖它自己那一对麦克风的几何。模型内不做任何 geometry 校验——
   pair 到麦克风的对应关系由调用方通过 ``pair_index`` 显式给出，``reference_channel``
   只用于范围校验，不参与配对。

4. **offline 分块推理**：fixed 模型通过 ``forward(..., offline_inference=True)``
   显式启用官方 ``n_seg`` 分块路径；该模式只允许 ``is_online=False`` 且 eval 状态。
   输入特征先由调用方按完整 observation 完成配置的归一化，再按 ``n_seg`` 右侧补零、
   分块前向、按时间顺序拼接并裁回 ``T//12`` 帧。为保证每块对应整数个输出帧，
   ``n_seg`` 必须是 12 的正整数倍。``offline_chunk_batch_size`` 限制每次前向的
   分块总数（跨输入 batch，默认 1）；无梯度推理时中间激活只覆盖当前小批，整段
   输入与输出仍随录音长度增长。默认 ``offline_inference=False`` 保留原有整段路径。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import Tensor, nn

from core.models.io_contract import ModelIOContract, TensorField
from core.utils.registry import MODEL_REGISTRY
from tasks.ssl.model_io import ModelSpec
from tasks.ssl.models.ipdnet import _PairwiseIPDModel


# 官方 CausCnnBlock 的时间压缩倍率：AvgPool((1,3)) 与 AvgPool((1,4))。
_OFFICIAL_TIME_STRIDE = 12

# 官方 CausCnnBlock 的卷积核/步长/填充（FixedAarryIPDnet.py 默认值）。
_CNN_KERNEL = (3, 3)
_CNN_STRIDE = (1, 1)
_CNN_PADDING = (1, 2)

# variable 模型 pair 内 4 通道的排布。官方仓库**没有** variable 的 runner，
# 因此这一项在原始代码里没有唯一事实来源：论文式 (14) 写作逐麦交错的
# ``[Re(X_1), Im(X_1), Re(X_m), Im(X_m)]``，而 ``VariableArrayIPDnet.py`` 的
# ``__main__`` 注释写作 ``2-channel[imag/real]``。本模块以论文（更权威的原文）
# 为准，默认取 ``per_mic_interleaved``，并把另一种排布显式开放给配置。
_VARIABLE_PAIR_CHANNEL_ORDER_INTERLEAVED = "per_mic_interleaved"
_VARIABLE_PAIR_CHANNEL_ORDER_BLOCK = "real_block_then_imag_block"
_VARIABLE_PAIR_CHANNEL_ORDERS = frozenset(
    {_VARIABLE_PAIR_CHANNEL_ORDER_INTERLEAVED, _VARIABLE_PAIR_CHANNEL_ORDER_BLOCK}
)
_VARIABLE_PAIR_CHANNEL_ORDER_DEFAULT = _VARIABLE_PAIR_CHANNEL_ORDER_INTERLEAVED


def _validate_int(value: Any, *, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def forgetting_norm(value: Tensor, sample_length: int = 298) -> Tensor:
    """官方 ``utils_.forgetting_norm``：沿时间轴的递推幅度归一化因子。

    与官方实现逐行对应：``alpha=(sample_length-1)/(sample_length+1)``，前
    ``sample_length`` 帧内使用 ``min((t-1)/(t+1), alpha)``，之后固定使用
    ``alpha``。输入 ``[B,C,F,T]``，返回 ``[B,1,1,T]``。

    注意：该函数只产生归一化因子，不修改特征本身。官方在 runner 里用它生成
    ``mean_value``，再把 real / imag 分别除以它。ON/OFF 的归一化差异（forgetting
    递推 vs 全局均值）由调用方选择，本函数只实现 online 的递推形式。
    """
    if value.ndim != 4:
        raise ValueError("forgetting_norm input must have shape [B,C,F,T]")
    if type(sample_length) is not int or sample_length < 1:
        raise ValueError("sample_length must be an exact integer >= 1")
    batch, channels, frequencies, frames = value.shape
    flat = value.reshape(batch, channels * frequencies, frames)
    alpha = (sample_length - 1) / (sample_length + 1)
    mu = torch.zeros(batch, 1, dtype=value.dtype, device=value.device)
    collected = []
    for frame in range(frames):
        if frame < sample_length:
            local = min((frame - 1) / (frame + 1), alpha)
        else:
            local = alpha
        mu = local * mu + (1.0 - local) * flat[:, :, frame].mean(dim=1, keepdim=True)
        collected.append(mu)
    return torch.stack(collected, dim=-1).reshape(batch, 1, 1, frames)


class _OfficialFNBlock(nn.Module):
    """官方 ``FixedAarryIPDnet.FNblock``：full-band + narrow-band LSTM 融合块。

    与官方逐行对应，仅补充形状校验。``is_first`` 决定 full-band LSTM 是否额外
    消费 ``add_skip_dim`` 宽的输入；narrow-band LSTM 的方向恒为
    ``not is_online``。输入 ``[N,T,F,nc]``。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        *,
        dropout: float = 0.2,
        add_skip_dim: int = 4,
        is_online: bool = False,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % 2:
            raise ValueError("hidden_size must be even for FN full/narrow fusion")
        if type(is_first) is not bool or type(is_online) is not bool:
            raise TypeError("is_first and is_online must be bool")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.full_hidden_size = hidden_size // 2
        self.add_skip_dim = add_skip_dim
        self.is_first = is_first
        self.is_online = is_online
        self.narr_hidden_size = hidden_size if is_online else hidden_size // 2
        self.dropout_full = nn.Dropout(p=float(dropout))
        self.dropout_narr = nn.Dropout(p=float(dropout))
        full_input = input_size if is_first else input_size + add_skip_dim
        self.full_lstm = nn.LSTM(
            input_size=full_input,
            hidden_size=self.full_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.narr_lstm = nn.LSTM(
            input_size=2 * self.full_hidden_size + add_skip_dim,
            hidden_size=self.narr_hidden_size,
            batch_first=True,
            bidirectional=not is_online,
        )

    def forward(self, value: Tensor, fb_skip: Tensor, nb_skip: Tensor) -> Tensor:
        if value.ndim != 4:
            raise ValueError("FN block input must have shape [N,T,F,D]")
        batch, frames, frequencies, _ = value.shape
        full = value.reshape(batch * frames, frequencies, -1)
        full, _ = self.full_lstm(full)
        full = self.dropout_full(full)
        full = torch.cat((full, fb_skip), dim=-1)
        narrow = full.view(batch, frames, frequencies, -1).permute(0, 2, 1, 3)
        narrow = narrow.reshape(batch * frequencies, frames, -1)
        narrow, _ = self.narr_lstm(narrow)
        narrow = self.dropout_narr(narrow)
        narrow = torch.cat((narrow, nb_skip), dim=-1)
        return narrow.view(batch, frequencies, frames, -1).permute(0, 2, 1, 3)


class _OfficialFNBlockMean(nn.Module):
    """官方 ``VariableArrayIPDnet.FNblock_mean``：带 pair 维均值广播的融合块。

    与官方逐行对应的部分是：full-band LSTM 沿频率轴、narrow-band LSTM 沿时间轴、
    每段之后对**外层 pair 批**求均值并广播回每个 pair、再与 ``skip`` 拼接后过
    ``Linear`` + ``PReLU``。

    登记的上游缺陷与本地修复见模块 docstring 第 1 条：官方把 ``narr_lstm`` 声明为
    ``2*full_hidden_size+add_skip_dim``，但实际输入来自 ``linear1``（宽度
    ``hidden_size``）。本实现让 ``narr_lstm`` 消费 ``hidden_size``。这是唯一
    自洽的取值，因为它同时保证 ``block_2.full_lstm(input_size=hidden_size)``
    能接受上一层输出。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        *,
        dropout: float = 0.2,
        add_skip_dim: int = 4,
        is_online: bool = False,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % 2:
            raise ValueError("hidden_size must be even for FN full/narrow fusion")
        if type(is_first) is not bool or type(is_online) is not bool:
            raise TypeError("is_first and is_online must be bool")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.full_hidden_size = hidden_size // 2
        self.add_skip_dim = add_skip_dim
        self.is_first = is_first
        self.is_online = is_online
        self.narr_hidden_size = hidden_size if is_online else hidden_size // 2
        self.linear1 = nn.Linear(2 * hidden_size + add_skip_dim, hidden_size)
        self.linear2 = nn.Linear(2 * hidden_size + add_skip_dim, hidden_size)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.dropout_full = nn.Dropout(p=float(dropout))
        self.dropout_narr = nn.Dropout(p=float(dropout))
        # 官方两个分支都用 ``input_size``（不做 add_skip_dim 区分），如实保留。
        self.full_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.full_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.narr_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=self.narr_hidden_size,
            batch_first=True,
            bidirectional=not is_online,
        )

    @staticmethod
    def _pair_mean(value: Tensor) -> Tensor:
        """对外层 pair 批求均值并广播回每个 pair（官方 pair-sharing）。"""
        return value.mean(dim=0, keepdim=True).expand_as(value)

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        if value.ndim != 4:
            raise ValueError("FN mean block input must have shape [P,T,F,D]")
        batch, frames, frequencies, _ = value.shape
        full = value.reshape(batch * frames, frequencies, -1)
        full, _ = self.full_lstm(full)
        full = self.dropout_full(full)
        full = full.view(batch, frames, frequencies, -1)
        narrow = torch.cat((full, self._pair_mean(full), skip), dim=-1)
        narrow = narrow.permute(0, 2, 1, 3).reshape(batch * frequencies, frames, -1)
        narrow = self.relu1(self.linear1(narrow))
        narrow, _ = self.narr_lstm(narrow)
        narrow = self.dropout_narr(narrow)
        narrow = narrow.view(batch, frequencies, frames, -1).permute(0, 2, 1, 3)
        fused = torch.cat((narrow, self._pair_mean(narrow), skip), dim=-1)
        return self.relu2(self.linear2(fused))


class _OfficialCausCnnBlock(nn.Module):
    """官方 ``CausCnnBlock``：3 × causal Conv2d + 两个时间平均池化。

    逐行对应官方实现。``padding=(1,2)`` 配合每层输出左移截断
    ``out[:, :, :, :-pad[1]]`` 实现时间轴因果；``AvgPool2d((1,3))`` 与
    ``AvgPool2d((1,4))`` 合计 12 倍时间压缩；最后一层 ``tanh``。

    输入 ``[N,C,Nf,T]``，输出 ``[N,out_dim,Nf,T//12]``。
    """

    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        *,
        cnn_hidden_dim: int = 128,
        kernel: tuple[int, int] = _CNN_KERNEL,
        stride: tuple[int, int] = _CNN_STRIDE,
        padding: tuple[int, int] = _CNN_PADDING,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inp_dim, cnn_hidden_dim, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.conv3 = nn.Conv2d(cnn_hidden_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 3))
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.pad = padding
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4:
            raise ValueError("CausCnnBlock input must have shape [N,C,Nf,T]")
        out = self.conv1(value)
        out = self.relu(out)
        out = out[:, :, :, : -self.pad[1]]
        out = self.pooling1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out[:, :, :, : -self.pad[1]]
        out = self.pooling2(out)
        out = self.conv3(out)
        out = out[:, :, :, : -self.pad[1]]
        return self.tanh(out)


def _canonical_stft_pair_input(
    value: Tensor,
    *,
    channels: int,
    reference_channel: int,
    num_freqs: int | None,
    exclude_dc: bool,
    name: str,
) -> Tensor:
    """把框架 canonical ``[B,T,F,2C]`` 转成官方 ``[N,T,F,2*C]`` 的实虚块排布。

    框架 ``features`` 的最后一维来自 ``projection._raw_stft_real_imag`` 的
    ``torch.view_as_real``，是**逐麦交错**的 ``(real, imag)``。官方 runner 的
    排布是「real 通道块 + imag 通道块」。这里先把 canonical 还原为 ``[B,T,F,C,2]``，
    再显式重排成官方顺序，避免任何隐式假设。

    官方 fixed 模型的 pair 轴以 reference microphone 为首；因此这里把显式
    ``reference_channel`` 放到第 0 个位置，再按原始通道顺序保留其余麦克风。
    """
    if value.ndim == 4:
        if value.shape[-1] != 2 * channels:
            raise ValueError(f"{name} must have last dimension 2*C = {2 * channels}")
        value = value.reshape(value.shape[0], value.shape[1], value.shape[2], channels, 2)
    elif value.ndim == 5:
        if value.shape[-1] != 2:
            raise ValueError(f"{name} last dimension must be 2 for real/imag layout")
        if value.shape[3] != channels:
            raise ValueError(f"{name} must contain exactly {channels} microphones")
    else:
        raise ValueError(f"{name} must have shape [B,T,F,2C] or [B,T,F,C,2]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point real/imag values")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    if type(reference_channel) is not int or not 0 <= reference_channel < channels:
        raise ValueError(
            f"reference_channel must be in [0, {channels}) for {name}"
        )
    if reference_channel:
        order = [reference_channel, *[index for index in range(channels) if index != reference_channel]]
        value = value[..., order, :]
    if value.shape[2] != num_freqs:
        raise ValueError(
            f"{name} frequency bins {value.shape[2]} != configured {num_freqs}"
        )
    if exclude_dc:
        # exclude_dc 由 projection 负责真正删除频点；此处只做一致性校验，不重复删除。
        raise ValueError(
            "official IPDnet consumes the raw multichannel STFT including DC; "
            "configure projection with exclude_dc=false"
        )
    # 官方排布：[全部 real 通道, 全部 imag 通道]。
    real = value[..., 0]
    imag = value[..., 1]
    return torch.cat((real, imag), dim=-1).contiguous()


def _fixed_reshape_to_official(output: Tensor, *, tracks: int, pairs: int) -> Tensor:
    """执行官方 fixed 的 reshape 链并返回官方布局张量。

    输入 ``[B, cnn_out_dim, F, T2]``，官方链为::

        permute(0,3,2,1) -> reshape(B,T2,F,2,-1) -> permute(0,1,3,2,4)
        -> reshape(B,T2,2,F*2,-1) -> permute(0,1,3,4,2)

    中间两步要求 ``cnn_out_dim = 2*P*K`` 满足 ``(P*K) % 2 == 0``；不满足时官方
    代码本身会抛 ``RuntimeError``（见模块 docstring 第 2 条），这里在调用前已由
    构造参数校验拦截。
    """
    batch, cnn_out_dim, frequencies, frames = output.shape
    value = output.permute(0, 3, 2, 1).reshape(batch, frames, frequencies, 2, -1)
    value = value.permute(0, 1, 3, 2, 4).reshape(batch, frames, 2, frequencies * 2, -1)
    return value.permute(0, 1, 3, 4, 2)


def _official_to_canonical(official: Tensor, *, tracks: int, pairs: int) -> Tensor:
    """官方 fixed 输出 → 框架 canonical ``[B,T,F,K,P,2]``。

    官方输出是 ``[B, T2, 2F, pairs*tracks/2, 2]``。后三轴 ``(2F, pairs*tracks/2, 2)``
    合并起来恰好是卷积输出通道与频率轴的**同一次 row-major 展开**（官方把
    ``(F, 2, P*K)`` reshape 成 ``(2, 2F, P*K/2)``）。因此还原不是「逐个
    ``(k,p,ridx,f)`` 猜位置」，而是精确地逆着那条 reshape 链走一遍：

    ``[B,T2,2F,P*K/2,2]`` → ``[B,T2,2,2F,P*K/2]`` → ``[B,T2,2,F,P*K]``
    → ``[B,T2,F,2,P*K]`` → ``[B,T2,F,2,P,K]`` → ``[B,T2,F,K,P,2]``。

    最后一步的 ``P*K → (P,K)`` 与 ``(2→ridx)`` 分组顺序（通道打包为
    ``c = ((p*K + k)*2 + ridx)``）由 ``tests/test_ssl_ipdnet_official.py`` 的
    逐元素置换测试固定：测试给每个 ``(p,k,ridx,f)`` 打上互不相同的记号值，验证
    它落在 ``out[B,T2,f,k,p,ridx]``。
    """
    batch, frames, double_freqs, grouped, two = official.shape
    frequencies = double_freqs // 2
    if two != 2 or grouped * two != pairs * tracks:
        raise ValueError(
            "official IPDnet output axes do not match the configured "
            f"K={tracks}, P={pairs}: got {tuple(official.shape)}"
        )
    # 逆最后一步 permute(0,1,3,4,2)。
    value = official.permute(0, 1, 4, 2, 3)
    # 逆 reshape(B,T2,2,F*2,-1)：把 (2F, P*K/2) 合回 (F, P*K)。
    value = value.reshape(batch, frames, 2, frequencies, pairs * tracks)
    # 逆 permute(0,1,3,2,4)：回到 reshape(B,T2,F,2,-1) 的输出布局。
    value = value.permute(0, 1, 3, 2, 4)
    # 逆 reshape(B,T2,F,2,-1)：把 P*K 拆成 (P, K)（通道按 p 外层、k 中层、ridx 内层打包）。
    value = value.reshape(batch, frames, frequencies, pairs, tracks, 2)
    # -> canonical [B, T2, F, K, P, 2]
    return value.permute(0, 1, 2, 4, 3, 5).contiguous()


def _variable_reshape_to_official(
    output: Tensor, *, batch: int, pairs: int, tracks: int
) -> Tensor:
    """执行官方 variable 的 reshape 链并返回官方布局张量。

    输入 ``[B*P, 4, F, T2]``（官方 conv 输出），官方链为::

        permute(0,3,2,1) -> reshape(B*P,T2,F,2,-1) -> permute(0,1,3,2,4)
        -> reshape(1,B*P,T2,-1,F*2) -> permute(0,2,4,1,3)

    输出 ``[1, T2, 2F, P*B, 2]``。该链把频率轴与 ``(2 re/im × K)`` 通道轴做了
    一次 row-major 展开，因此 ``2F`` 轴并非逐频 ``(real, imag)`` 配对；还原方式见
    ``_variable_official_to_canonical``。
    """
    pair_batch, channels, frequencies, frames = output.shape
    if pair_batch != batch * pairs:
        raise ValueError("official variable output pair-batch axis does not match B*P")
    # 官方 channel 打包为 (re/im × K)：必须与硬编码 cnn_out_dim=4 一致。
    if channels != 2 * tracks:
        raise ValueError(
            "official variable IPDnet channel axis must be 2*K "
            f"(hardcoded cnn_out_dim=4 for K=2), got {channels}"
        )
    value = output.permute(0, 3, 2, 1).reshape(pair_batch, frames, frequencies, 2, -1)
    value = value.permute(0, 1, 3, 2, 4)
    value = value.reshape(1, pair_batch, frames, -1, frequencies * 2)
    return value.permute(0, 2, 4, 1, 3)


def _variable_official_to_canonical(
    official: Tensor, *, batch: int, pairs: int, tracks: int
) -> Tensor:
    """官方 variable 输出 → 框架 canonical ``[B,T,F,K,P,2]``。

    官方输出 ``[1, T2, 2F, B*P, 2]`` 的轴语义（经单元测试逐元素确认）：

    - ``2F`` 轴按 ``2*f + k`` 展开（``k`` 内层、频点 ``f`` 外层）；
    - ``B*P`` 轴按 ``(b, p)`` 展开（``b`` 外层、pair ``p`` 内层）；
    - 最后一轴是实虚部 ``ridx``。

    conv 通道打包为 ``c = ridx * K + k``，即 ``cnn_out_dim = 2*K``（官方硬编码
    4，对应 K=2）。该对应关系由 ``tests/test_ssl_ipdnet_official.py`` 的逐元素
    置换测试固定。
    """
    one, frames, double_freqs, pair_batch, two = official.shape
    if one != 1 or two != 2 or pair_batch != batch * pairs:
        raise ValueError(
            "official variable IPDnet output axes do not match the expected "
            f"[1, T2, 2F, B*P, 2] with B={batch}, P={pairs}: got {tuple(official.shape)}"
        )
    frequencies = double_freqs // 2
    if frequencies * tracks != double_freqs:
        raise ValueError(
            f"official variable IPDnet 2F axis {double_freqs} != 2*F with F={frequencies}, K={tracks}"
        )
    # [1, T2, F, K, B, P, 2]：先拆 2F -> (F, K)，再拆 B*P -> (B, P)。
    value = official.reshape(1, frames, frequencies, tracks, batch, pairs, 2)
    # -> canonical [B, T2, F, K, P, 2]
    return value[0].permute(3, 0, 1, 2, 4, 5).contiguous()


class _OfficialIPDnetFixed(nn.Module):
    """官方 fixed-array IPDnet 的结构实现（``FixedAarryIPDnet.IPDnet``）。"""

    MODEL_TYPE = "IPDNetFixed"
    MODEL_SPEC = ModelSpec(
        model_type=MODEL_TYPE,
        feature_contract="multichannel_stft_real_imag",
        target_contract="dpipd_pair",
        pair_layout="reference_to_all",
        geometry_mode="fixed_array",
        field_mode="far_field",
        reduction="causal_conv_pool",
        causal=True,
        mapping="far_field_grid",
        target_layout="trailing_ri",
    )
    IO_CONTRACT = _PairwiseIPDModel.IO_CONTRACT

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        max_track: int = 2,
        n_fft: int = 512,
        stride: int = _OFFICIAL_TIME_STRIDE,
        causal: bool = True,
        dropout: float = 0.2,
        cnn_hidden_dim: int = 128,
        reference_channel: int = 0,
        n_seg: int = 312,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str | None = None,
        center: bool | None = None,
        exclude_dc: bool = False,
        offline_chunk_batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            raise TypeError(f"unexpected model arguments: {sorted(kwargs)}")
        _validate_int(input_size, name="input_size", minimum=4)
        _validate_int(hidden_size, name="hidden_size", minimum=2)
        _validate_int(max_track, name="max_track", minimum=1)
        _validate_int(n_fft, name="n_fft", minimum=2)
        _validate_int(stride, name="stride", minimum=1)
        _validate_int(cnn_hidden_dim, name="cnn_hidden_dim", minimum=1)
        _validate_int(n_seg, name="n_seg", minimum=1)
        _validate_int(offline_chunk_batch_size, name="offline_chunk_batch_size", minimum=1)
        if hidden_size % 2:
            raise ValueError("hidden_size must be even for FN full/narrow fusion")
        if input_size % 2:
            raise ValueError(
                "official IPDnet input_size must be 2*C (an even number of real/imag channels)"
            )
        if stride != _OFFICIAL_TIME_STRIDE:
            raise ValueError(
                "official IPDnet causal convolution fixes the time stride at 12"
            )
        if type(causal) is not bool:
            raise TypeError("causal must be a bool")
        # 官方 CausCnnBlock 恒因果：causal=False 只切换 narrow-band LSTM 到
        # is_online=False（双向、hidden_size//2），不会让卷积变成非因果。
        if type(reference_channel) is not int or reference_channel < 0:
            raise ValueError("reference_channel must be a non-negative integer")
        if hop_length is not None and (type(hop_length) is not int or hop_length < 1):
            raise ValueError("hop_length must be a positive exact integer")
        if win_length is not None and (type(win_length) is not int or not 1 <= win_length <= n_fft):
            raise ValueError("win_length must satisfy 1 <= win_length <= n_fft")
        if window is not None and window not in {"hann", "hamming"}:
            raise ValueError("window must be hann or hamming")
        if center is not None and type(center) is not bool:
            raise TypeError("center must be a bool")
        if exclude_dc is not False:
            raise ValueError("official IPDnet consumes the raw STFT including DC; set exclude_dc=false")
        if type(dropout) is not float and type(dropout) is not int:
            raise TypeError("dropout must be a float")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.pairs = input_size // 2 - 1
        if self.pairs < 1:
            raise ValueError("official IPDnet requires at least two microphones (input_size >= 4)")
        # 官方 reshape 链要求 (P*K) 为偶数；不满足时上游会在 reshape 处崩溃。
        if (self.pairs * max_track) % 2:
            raise ValueError(
                "official IPDnet output reshape requires (pairs * max_track) to be even; "
                f"got pairs={self.pairs}, max_track={max_track}"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_track = max_track
        self.n_fft = n_fft
        self.stride = stride
        self.causal = causal
        # 保留类级默认 online 合同，同时让实例声明实际使用的变体。
        self.MODEL_SPEC = replace(type(self).MODEL_SPEC, causal=causal)
        self.reference_channel = reference_channel
        self.n_seg = n_seg
        self.offline_chunk_batch_size = offline_chunk_batch_size
        self.hop_length, self.win_length = hop_length, win_length
        self.window, self.center = window, center
        self.exclude_dc = exclude_dc
        self.num_freqs = n_fft // 2 + 1
        self.add_skip_dim = input_size
        # ``causal`` 是本地公开参数对官方 ``is_online`` 的显式映射；不提供
        # 同名兼容别名，避免两个开关漂移。CausCnnBlock 仍始终因果。
        self.is_online = causal
        self.block_1 = _OfficialFNBlock(
            input_size, hidden_size, dropout=dropout, add_skip_dim=input_size,
            is_online=self.is_online, is_first=True,
        )
        self.block_2 = _OfficialFNBlock(
            hidden_size, hidden_size, dropout=dropout, add_skip_dim=input_size,
            is_online=self.is_online, is_first=False,
        )
        # 官方 ``cnn_out_dim = 2 * ((input_size // 2) - 1) * max_track``，其中
        # ``input_size // 2 - 1`` 即 reference-to-all 的 pair 数 ``P = C - 1``。
        self.cnn_out_dim = 2 * self.pairs * max_track
        self.cnn_inp_dim = hidden_size + input_size
        self.conv = _OfficialCausCnnBlock(
            self.cnn_inp_dim, self.cnn_out_dim, cnn_hidden_dim=cnn_hidden_dim,
        )

    @property
    def channels(self) -> int:
        """官方 ``input_size`` 对应的麦克风数（``input_size = 2C``）。"""
        return self.input_size // 2

    def forward(
        self,
        multichannel_stft_real_imag: Tensor | None = None,
        *,
        features: Tensor | None = None,
        offline_inference: bool = False,
    ) -> Tensor:
        if type(offline_inference) is not bool:
            raise TypeError("offline_inference must be a bool")
        if multichannel_stft_real_imag is None:
            if features is None:
                raise TypeError("missing required input: multichannel_stft_real_imag")
            multichannel_stft_real_imag = features
        elif features is not None:
            raise ValueError("provide only multichannel_stft_real_imag or features")
        if offline_inference:
            if self.is_online:
                raise ValueError(
                    "offline_inference requires IPDNetFixed configured with causal=false"
                )
            if self.training:
                raise RuntimeError(
                    "offline_inference is evaluation-only; call model.eval() first"
                )
        # canonical [B,T,F,2C] -> 官方 [B,T,F,2C]（real 块 + imag 块）
        value = _canonical_stft_pair_input(
            multichannel_stft_real_imag,
            channels=self.channels,
            reference_channel=self.reference_channel,
            num_freqs=self.num_freqs,
            exclude_dc=self.exclude_dc,
            name="multichannel_stft_real_imag",
        )
        if not offline_inference:
            return self._forward_core(value)

        batch, frames, frequencies, channels = value.shape
        if self.n_seg % _OFFICIAL_TIME_STRIDE:
            raise ValueError(
                "n_seg must be a positive multiple of the official time stride "
                f"({_OFFICIAL_TIME_STRIDE}) when offline_inference is enabled, got {self.n_seg}"
            )
        if frames < _OFFICIAL_TIME_STRIDE:
            # 官方路径会先补齐到 n_seg，再按原始 T//12 裁剪；不足一个
            # stride 的输入因此合法地产生零个输出帧。
            return value.new_empty(
                batch, 0, frequencies, self.max_track, self.pairs, 2
            )
        segment_count = (frames + self.n_seg - 1) // self.n_seg
        padded_frames = segment_count * self.n_seg
        pad_frames = padded_frames - frames
        if pad_frames:
            # 特征在整段 observation 上已完成 global_mean / 其他配置归一化；
            # padding 只补零，不能按 chunk 重新估计归一化因子。
            padding = value.new_zeros(batch, pad_frames, frequencies, channels)
            value = torch.cat((value, padding), dim=1)
        segments = value.reshape(
            batch, segment_count, self.n_seg, frequencies, channels
        ).reshape(batch * segment_count, self.n_seg, frequencies, channels)
        # flatten 顺序保持为 batch → segment；每次只计算有限个分块的中间激活。
        chunk_outputs = [
            self._forward_core(chunk)
            for chunk in segments.split(self.offline_chunk_batch_size, dim=0)
        ]
        chunk_output = torch.cat(chunk_outputs, dim=0)
        chunk_output = chunk_output.reshape(
            batch,
            segment_count,
            self.n_seg // _OFFICIAL_TIME_STRIDE,
            frequencies,
            self.max_track,
            self.pairs,
            2,
        )
        output = chunk_output.flatten(1, 2)
        return output[:, : frames // _OFFICIAL_TIME_STRIDE]

    def _forward_core(self, value: Tensor) -> Tensor:
        """执行单个连续时间段的官方 FN + causal CNN 前向核心。"""
        # 模型入口把 [B,T,F,2C] 转成官方 forward 期望的 [nb, nc=2C, nf, nt]。
        value = value.permute(0, 3, 2, 1).contiguous()
        x = value.permute(0, 3, 2, 1)
        batch, frames, frequencies, channels = x.shape
        if channels != self.input_size:
            raise ValueError(
                f"official IPDnet expects {self.input_size} real/imag channels, got {channels}"
            )
        if frames < _OFFICIAL_TIME_STRIDE:
            raise ValueError(
                "official IPDnet causal convolution requires at least "
                f"{_OFFICIAL_TIME_STRIDE} input frames, got {frames}"
            )
        fb_skip = x.reshape(batch * frames, frequencies, channels)
        nb_skip = x.permute(0, 2, 1, 3).reshape(batch * frequencies, frames, channels)
        x = self.block_1(x, fb_skip=fb_skip, nb_skip=nb_skip)
        x = self.block_2(x, fb_skip=fb_skip, nb_skip=nb_skip)
        batch, frames, frequencies, _ = x.shape
        x = x.permute(0, 3, 2, 1)
        output_frames = frames // self.stride
        out = self.conv(x)
        if out.shape[-1] != output_frames:
            raise ValueError(
                "official IPDnet causal convolution produced an unexpected time axis: "
                f"{out.shape[-1]} != {output_frames}"
            )
        official = _fixed_reshape_to_official(out, tracks=self.max_track, pairs=self.pairs)
        return _official_to_canonical(official, tracks=self.max_track, pairs=self.pairs)


@MODEL_REGISTRY.register("IPDNetFixed")
class IPDNetFixed(_OfficialIPDnetFixed):
    """官方 fixed-array IPDnet（``FixedAarryIPDnet.IPDnet``）的框架接入。

    结构、输入通道排布、时间压缩与输出布局转换见模块 docstring。离线评估可显式
    调用 ``forward(features, offline_inference=True)`` 启用 ``n_seg`` 分块；训练态、
    online 变体与启用分块时的非 12 倍数段长均 fail-fast。输入归一化由投影层按完整
    observation 完成，模型不会在各 chunk 内重新估计。``offline_chunk_batch_size``
    默认为 1，只限制离线推理单次处理的分块总数，不改变模型参数或 checkpoint 布局。
    """


class _OfficialIPDnetVariable(nn.Module):
    """官方 variable-array IPDnet（``VariableArrayIPDnet.VariableIPDnet``）。"""

    MODEL_TYPE = "IPDNetVariable"
    MODEL_SPEC = ModelSpec(
        model_type=MODEL_TYPE,
        feature_contract="multichannel_stft_real_imag",
        target_contract="dpipd_pair",
        # 官方 variable 结构本身只消费 pair-batched 2 麦输入；框架侧把 canonical
        # reference-to-all 通道在模型内部展开成 pair 批，再对 pair 维做 mean-sharing。
        # 输出仍还原为 canonical reference-to-all 的 P 轴，故 pair_layout 保持
        # reference_to_all；这正是官方 train/test 使用的 pair 排布。
        pair_layout="reference_to_all",
        geometry_mode="variable_array",
        field_mode="far_field",
        reduction="causal_conv_pool",
        causal=True,
        mapping="far_field_grid",
        target_layout="trailing_ri",
    )
    IO_CONTRACT = _PairwiseIPDModel.IO_CONTRACT

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        max_track: int = 2,
        n_fft: int = 512,
        stride: int = _OFFICIAL_TIME_STRIDE,
        causal: bool = True,
        dropout: float = 0.2,
        cnn_hidden_dim: int = 128,
        reference_channel: int = 0,
        pair_channel_order: str = _VARIABLE_PAIR_CHANNEL_ORDER_DEFAULT,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str | None = None,
        center: bool | None = None,
        exclude_dc: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            raise TypeError(f"unexpected model arguments: {sorted(kwargs)}")
        _validate_int(input_size, name="input_size", minimum=4)
        _validate_int(hidden_size, name="hidden_size", minimum=2)
        _validate_int(max_track, name="max_track", minimum=1)
        _validate_int(n_fft, name="n_fft", minimum=2)
        _validate_int(stride, name="stride", minimum=1)
        _validate_int(cnn_hidden_dim, name="cnn_hidden_dim", minimum=1)
        if pair_channel_order not in _VARIABLE_PAIR_CHANNEL_ORDERS:
            raise ValueError(
                "pair_channel_order must be one of "
                f"{sorted(_VARIABLE_PAIR_CHANNEL_ORDERS)!r}"
            )
        if hidden_size % 2:
            raise ValueError("hidden_size must be even for FN full/narrow fusion")
        if input_size != 4:
            raise ValueError(
                "official variable IPDnet consumes one microphone pair per network input "
                "(input_size=4 = 2 mics x real/imag)"
            )
        if stride != _OFFICIAL_TIME_STRIDE:
            raise ValueError(
                "official IPDnet causal convolution fixes the time stride at 12"
            )
        if type(causal) is not bool:
            raise TypeError("causal must be a bool")
        # CausCnnBlock 在 online/offline 两个官方变体中都保持因果；causal=False
        # 只切换 narrow-band LSTM 的 is_online=False 双向路径。
        if type(reference_channel) is not int or reference_channel < 0:
            raise ValueError("reference_channel must be a non-negative integer")
        if hop_length is not None and (type(hop_length) is not int or hop_length < 1):
            raise ValueError("hop_length must be a positive exact integer")
        if win_length is not None and (type(win_length) is not int or not 1 <= win_length <= n_fft):
            raise ValueError("win_length must satisfy 1 <= win_length <= n_fft")
        if window is not None and window not in {"hann", "hamming"}:
            raise ValueError("window must be hann or hamming")
        if center is not None and type(center) is not bool:
            raise TypeError("center must be a bool")
        if exclude_dc is not False:
            raise ValueError("official IPDnet consumes the raw STFT including DC; set exclude_dc=false")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        # 官方 variable 的 cnn_out_dim 硬编码为 4（2 speaker × 2 re/im），
        # 即 K 固定为 2、每个 pair 一对 re/im。见模块 docstring。
        if max_track != 2:
            raise ValueError(
                "official variable IPDnet hardcodes cnn_out_dim=4 (2 sources x real/imag); "
                "max_track must be 2"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_track = max_track
        self.n_fft = n_fft
        self.stride = stride
        self.causal = causal
        # 保留类级默认 online 合同，同时让实例声明实际使用的变体。
        self.MODEL_SPEC = replace(type(self).MODEL_SPEC, causal=causal)
        self.reference_channel = reference_channel
        self.pair_channel_order = pair_channel_order
        self.hop_length, self.win_length = hop_length, win_length
        self.window, self.center = window, center
        self.exclude_dc = exclude_dc
        self.num_freqs = n_fft // 2 + 1
        self.is_online = causal
        self.block_1 = _OfficialFNBlockMean(
            input_size, hidden_size, dropout=dropout, add_skip_dim=input_size,
            is_online=self.is_online, is_first=True,
        )
        self.block_2 = _OfficialFNBlockMean(
            hidden_size, hidden_size, dropout=dropout, add_skip_dim=input_size,
            is_online=self.is_online, is_first=False,
        )
        self.cnn_out_dim = 4
        self.cnn_inp_dim = hidden_size
        self.conv = _OfficialCausCnnBlock(
            self.cnn_inp_dim, self.cnn_out_dim, cnn_hidden_dim=cnn_hidden_dim,
        )

    def _pair_batch(self, value: Tensor, pair_index: Tensor, channels: int) -> Tensor:
        """按 ``pair_index`` 把 canonical 通道装配成官方 pair 批。

        官方 variable 的输入是 ``[P, 4, F, T]``：每个 pair 由 reference 与该 pair
        的另一通道组成，且 real/imag 按官方排布（real 块 + imag 块）拼接成 4 通道。
        这里不做任何按 pair 的 padding：``pair_index`` 中的每一对都必须真实存在。
        """
        if pair_index.ndim == 2:
            pair_index = pair_index.unsqueeze(0)
        if pair_index.ndim != 3 or pair_index.shape[-1] != 2:
            raise ValueError("pair_index must have shape [B,P,2] or [P,2]")
        batch = value.shape[0]
        if pair_index.shape[0] not in {1, batch}:
            raise ValueError("pair_index batch dimension must be 1 or match features B")
        if pair_index.dtype not in (torch.int64, torch.int32, torch.int16):
            raise TypeError("pair_index must use an integer dtype")
        pairs = pair_index.shape[1]
        if pairs < 1:
            raise ValueError("pair_index must contain at least one pair")
        if bool((pair_index < 0).any().item()) or bool((pair_index >= channels).any().item()):
            raise ValueError("pair_index contains an out-of-range microphone")
        if bool((pair_index[..., 0] == pair_index[..., 1]).any().item()):
            raise ValueError("pair_index cannot contain self pairs")
        first_channel = pair_index[..., 0].long()
        second_channel = pair_index[..., 1].long()
        real = value[..., 0]
        imag = value[..., 1]
        frames, frequencies = real.shape[1], real.shape[2]
        # ``real``/``imag`` 形状为 [B,T,F,C]，通道轴在最末。把 (T,F) 合并后，
        # 用 [B, T*F, P] 的索引在通道轴上按 pair 取两个通道。
        real_flat = real.reshape(batch, frames * frequencies, channels)
        imag_flat = imag.reshape(batch, frames * frequencies, channels)
        first_index = first_channel.unsqueeze(1).expand(batch, frames * frequencies, pairs)
        second_index = second_channel.unsqueeze(1).expand(batch, frames * frequencies, pairs)
        first_real = real_flat.gather(2, first_index).reshape(batch, frames, frequencies, pairs)
        second_real = real_flat.gather(2, second_index).reshape(batch, frames, frequencies, pairs)
        first_imag = imag_flat.gather(2, first_index).reshape(batch, frames, frequencies, pairs)
        second_imag = imag_flat.gather(2, second_index).reshape(batch, frames, frequencies, pairs)
        # 逐麦交错：pair 内 4 通道为 [Re(X_ref), Im(X_ref), Re(X_other), Im(X_other)]。
        # 依据论文式 (14) ``x_{1,m} = [Re(X_1), Im(X_1), Re(X_m), Im(X_m)]``，而非
        # fixed runner 的 real 块 / imag 块排布。官方 variable 无 runner，两种排布
        # 由 ``pair_channel_order`` 显式选择（见模块 docstring）。
        if self.pair_channel_order == _VARIABLE_PAIR_CHANNEL_ORDER_INTERLEAVED:
            stack = torch.stack((first_real, first_imag, second_real, second_imag), dim=-1)
        else:
            stack = torch.stack((first_real, second_real, first_imag, second_imag), dim=-1)
        # [B,T,F,P,4] -> [B,P,T,F,4]；官方 pair 批把 (B,P) 合并成一个前导轴
        # （前向只认 pair 维，mean-sharing 也在该轴上发生）。
        return stack.permute(0, 3, 1, 2, 4).reshape(
            batch * pairs, frames, frequencies, 4
        ).contiguous()

    def forward(
        self,
        multichannel_stft_real_imag: Tensor | None = None,
        *,
        features: Tensor | None = None,
        pair_index: Tensor | None = None,
    ) -> Tensor:
        if multichannel_stft_real_imag is None:
            if features is None:
                raise TypeError("missing required input: multichannel_stft_real_imag")
            multichannel_stft_real_imag = features
        elif features is not None:
            raise ValueError("provide only multichannel_stft_real_imag or features")
        if pair_index is None:
            raise TypeError("official variable IPDnet requires pair_index with shape [B,P,2]")
        value = multichannel_stft_real_imag
        if value.ndim == 4:
            if value.shape[-1] % 2:
                raise ValueError("multichannel_stft_real_imag last dimension must be 2*C")
            channels = value.shape[-1] // 2
            value = value.reshape(value.shape[0], value.shape[1], value.shape[2], channels, 2)
        elif value.ndim == 5:
            if value.shape[-1] != 2:
                raise ValueError("multichannel_stft_real_imag last dimension must be 2")
            channels = value.shape[3]
        else:
            raise ValueError("multichannel_stft_real_imag must have shape [B,T,F,2C] or [B,T,F,C,2]")
        if not value.is_floating_point():
            raise TypeError("multichannel_stft_real_imag must be floating point")
        if not torch.isfinite(value).all():
            raise ValueError("multichannel_stft_real_imag must be finite")
        if value.shape[2] != self.num_freqs:
            raise ValueError(
                f"multichannel_stft_real_imag frequency bins {value.shape[2]} != configured {self.num_freqs}"
            )
        if channels < 2:
            raise ValueError("official variable IPDnet requires at least two microphones")
        if not 0 <= self.reference_channel < channels:
            raise ValueError("reference_channel is outside the microphone dimension")
        batch, frames = value.shape[0], value.shape[1]
        if frames < _OFFICIAL_TIME_STRIDE:
            raise ValueError(
                "official IPDnet causal convolution requires at least "
                f"{_OFFICIAL_TIME_STRIDE} input frames, got {frames}"
            )
        pairs = self._pair_batch(value, pair_index, channels)
        # ``_pair_batch`` 返回 [B*P, T, F, 4]：即官方 forward 中
        # ``x = x.permute(0,3,2,1)`` 之后的 FN 块输入，同一张量同时充当 skip。
        blocks_input = pairs
        skip = blocks_input
        hidden = self.block_1(blocks_input, skip=skip)
        hidden = self.block_2(hidden, skip=skip)
        pair_batch, time, freq, _ = hidden.shape
        # CausCnnBlock 消费 [N, 4, F, T]。
        x = hidden.permute(0, 3, 2, 1).contiguous()
        out = self.conv(x)
        output_frames = time // self.stride
        if out.shape[-1] != output_frames:
            raise ValueError(
                "official variable IPDnet causal convolution produced an unexpected time axis: "
                f"{out.shape[-1]} != {output_frames}"
            )
        num_pairs = pair_batch // batch
        official = _variable_reshape_to_official(
            out, batch=batch, pairs=num_pairs, tracks=self.max_track
        )
        return _variable_official_to_canonical(
            official, batch=batch, pairs=num_pairs, tracks=self.max_track
        )


@MODEL_REGISTRY.register("IPDNetVariable")
class IPDNetVariable(_OfficialIPDnetVariable):
    """官方 variable-array IPDnet（``VariableArrayIPDnet.VariableIPDnet``）的框架接入。

    pair-batched 适配与 pair 维 mean-sharing 见模块 docstring。
    """


__all__ = [
    "IPDNetFixed",
    "IPDNetVariable",
    "forgetting_norm",
]
