"""BSRNN separator。

谱张量使用复数 ``(B, F, T)`` 或实虚堆叠 ``(B, 2, F, T)``；框架
``[B, F, T, 2]`` 的转换由 ``tasks.tse.models.discriminative.tse_bsrnn_spk`` 完成。

Author: Xincheng Zhu
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tasks.tse.models.modules.common.norm import select_norm


class BandSplit(nn.Module):
    """按子带表切分频谱。"""

    def __init__(self, band_width):
        super().__init__()
        self.band_width = band_width

    def forward(self, spec):
        """
        spec: (B, F, T) 复数谱 或 (B, 2, F, T) 实虚堆叠谱
        return:
           subband_spec: list of (B, 2, BW, T) 或 (B, BW, T)
        """
        num_dims = spec.dim()
        if num_dims not in (3, 4):
            raise ValueError(
                f"BandSplit only supports 3D (B, F, T) or 4D (B, 2, F, T) input, "
                f"got shape {tuple(spec.shape)}"
            )

        subband_spec = []
        idx = 0
        for bw in self.band_width:
            if num_dims == 4:
                subband_spec.append(spec[:, :, idx:idx + bw].contiguous())
            elif num_dims == 3:
                subband_spec.append(spec[:, idx:idx + bw])
            idx += bw
        return subband_spec


class SubbandNorm(nn.Module):
    """子带归一化 + 1x1 conv 投影到 feature_dim。"""

    def __init__(self,
                 band_width,
                 spec_dim,
                 nband,
                 feature_dim,
                 norm_type='GN',
                 eps=torch.finfo(torch.float32).eps):
        super().__init__()
        self.band_width = band_width
        self.spec_dim = spec_dim
        self.nband = nband
        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(
                    select_norm(norm_type, band_width[i] * spec_dim, eps),
                    nn.Conv1d(band_width[i] * spec_dim, feature_dim, 1),
                ))

    def forward(self, subband_spec):
        """
        subband_spec: list of (B, spec_dim, BW, T)
        return:
            subband_feature: (B, nband, feat, T)
        """
        B = subband_spec[0].shape[0]
        subband_feature = []
        for i, bn in enumerate(self.BN):
            x = subband_spec[i].view(B, self.band_width[i] * self.spec_dim, -1)
            subband_feature.append(bn(x))
        subband_feature = torch.stack(subband_feature, 1)
        return subband_feature


class BandMasker(nn.Module):
    """子带复数掩码估计（glu 掩码 × 混合谱）。"""

    def __init__(self,
                 band_width,
                 nband,
                 feature_dim,
                 norm_type,
                 nspk=1,
                 eps=torch.finfo(torch.float32).eps):
        super().__init__()
        self.band_width = band_width
        self.nspk = nspk
        self.mask = nn.ModuleList([])
        for i in range(nband):
            self.mask.append(
                nn.Sequential(
                    select_norm(norm_type, feature_dim, eps),
                    nn.Conv1d(feature_dim, feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(feature_dim * 4, feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(feature_dim * 4, band_width[i] * 4 * nspk, 1),
                ))

    def forward(self, sep_output, subband_mix_spec):
        """
        sep_output: (B, nband, feat, T)
        subband_mix_spec: list of (B, BW, T) complex
        return:
            est_spec_RI: (B, 2, S, F, T)
        """
        sep_subband_R = []  # real
        sep_subband_I = []  # imag
        for i, mask_func in enumerate(self.mask):
            this_output = mask_func(sep_output[:, i])  # (B, S*4*BW, T)
            Bnch, _, T = this_output.shape
            BW = self.band_width[i]

            # reshape → (B, S, 2, 2, BW, T)
            this_output = this_output.view(Bnch, self.nspk, 2, 2, BW, T)
            # shape → (B, S, 2, BW, T)
            this_mask = this_output[:, :, 0] * torch.sigmoid(this_output[:, :,
                                                                         1])
            # split real/imag → (B, S, BW, T)
            this_mask_real = this_mask[:, :, 0]
            this_mask_imag = this_mask[:, :, 1]

            # subband_mix_spec[i] shape: (B, BW, T)
            mix_real = subband_mix_spec[i].real.unsqueeze(1)  # → (B,1,BW,T)
            mix_imag = subband_mix_spec[i].imag.unsqueeze(1)

            # 复数掩码：
            est_real = mix_real * this_mask_real - mix_imag * this_mask_imag
            est_imag = mix_real * this_mask_imag + mix_imag * this_mask_real
            sep_subband_R.append(est_real)
            sep_subband_I.append(est_imag)

        # 频率维拼接
        est_R = torch.cat(sep_subband_R, dim=2)  # (B, S, F, T)
        est_I = torch.cat(sep_subband_I, dim=2)  # (B, S, F, T)
        est_spec_RI = torch.stack([est_R, est_I], dim=1)  # (B, 2, S, F, T)
        return est_spec_RI


class ResRNN(nn.Module):
    """残差 RNN：norm → LSTM → 线性投影 + 残差。"""

    def __init__(self,
                 input_size,
                 hidden_size,
                 bidirectional=True,
                 norm_type='GN',
                 eps=None):
        super(ResRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.eps = torch.finfo(torch.float32).eps if eps is None else eps
        norm_eps = 1e-5 if eps is None else eps

        self.norm = select_norm(norm_type, input_size, eps=norm_eps)
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # 线性投影层
        self.proj = nn.Linear(hidden_size * (int(bidirectional) + 1),
                              input_size)

    def forward(self, input, state=None):
        # input shape: batch, dim, seq
        output, _ = self.forward_stream(input, state)
        return output

    def forward_stream(self, input, state):
        """携带 LSTM ``(h, c)`` 处理一个连续时间 chunk。"""
        normalized = self.norm(input).transpose(1, 2).contiguous()
        rnn_output, next_state = self.rnn(normalized, state)
        rnn_output = self.proj(rnn_output.contiguous().view(
            -1, rnn_output.shape[2])).view(input.shape[0], input.shape[2],
                                           input.shape[1])
        return input + rnn_output.transpose(1, 2).contiguous(), next_state


class BSNet(nn.Module):
    """时间 ResRNN + 频率 ResRNN 双路模块。causal=True 时时间维单向。"""

    def __init__(
        self,
        in_channel,
        nband=7,
        causal=False,
        norm_type='GN',
        eps=None,
    ):
        super(BSNet, self).__init__()

        self.nband = nband
        self.feature_dim = in_channel // nband
        self.band_rnn = ResRNN(self.feature_dim,
                               self.feature_dim * 2,
                               bidirectional=not causal,
                               norm_type=norm_type,
                               eps=eps)
        self.band_comm = ResRNN(self.feature_dim,
                                self.feature_dim * 2,
                                bidirectional=True,
                                norm_type='GN',
                                eps=eps)

    @staticmethod
    def _expand_initial_state(
        state,
        *,
        name,
        input,
        batch_size,
        hidden_size,
        repeats,
        directions,
    ):
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError(f"initial_state[{name!r}] 必须是 (h0, c0) tuple")
        expanded = []
        for state_name, value in zip(("h0", "c0"), state):
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"initial_state[{name!r}].{state_name} 必须是 Tensor"
                )
            if tuple(value.shape) != (batch_size, hidden_size):
                raise ValueError(
                    f"initial_state[{name!r}].{state_name} 应为 "
                    f"[{batch_size}, {hidden_size}]，got {tuple(value.shape)}"
                )
            if value.device != input.device or value.dtype != input.dtype:
                raise TypeError(
                    f"initial_state[{name!r}].{state_name} 必须与 input "
                    "使用相同 device 和 dtype"
                )
            expanded.append(
                value.repeat_interleave(repeats, dim=0)
                .unsqueeze(0)
                .expand(directions, -1, -1)
                .contiguous()
            )
        return tuple(expanded)

    def forward(self, input, initial_state=None):
        B, N, T = input.shape
        band_state = None
        communication_state = None
        if initial_state is not None:
            if not isinstance(initial_state, dict):
                raise TypeError("BSNet initial_state 必须是 dict 或 None")
            unknown = sorted(set(initial_state) - {"band", "communication"})
            if unknown:
                raise ValueError(f"BSNet initial_state 包含未知键：{unknown}")
            if "band" in initial_state:
                band_state = self._expand_initial_state(
                    initial_state["band"],
                    name="band",
                    input=input,
                    batch_size=B,
                    hidden_size=self.band_rnn.hidden_size,
                    repeats=self.nband,
                    directions=2 if self.band_rnn.bidirectional else 1,
                )
            if "communication" in initial_state:
                communication_state = self._expand_initial_state(
                    initial_state["communication"],
                    name="communication",
                    input=input,
                    batch_size=B,
                    hidden_size=self.band_comm.hidden_size,
                    repeats=T,
                    directions=2 if self.band_comm.bidirectional else 1,
                )

        band_output = self.band_rnn(
            input.view(B * self.nband, self.feature_dim, -1),
            band_state,
        ).view(B, self.nband, -1, T)
        band_output = (band_output.permute(0, 3, 2, 1).contiguous().view(
            B * T, -1, self.nband))
        output = (self.band_comm(band_output, communication_state).view(
            B, T, -1, self.nband).permute(0, 3, 2, 1).contiguous())
        return output.view(B, N, T)

    def forward_stream(self, input, state):
        """按 chunk 更新时间 RNN state；频带 RNN 仍逐帧独立计算。"""
        B, N, T = input.shape
        band_output, next_state = self.band_rnn.forward_stream(
            input.view(B * self.nband, self.feature_dim, -1), state
        )
        band_output = band_output.view(B, self.nband, -1, T)

        band_output = (band_output.permute(0, 3, 2, 1).contiguous().view(
            B * T, -1, self.nband))
        output = (self.band_comm(band_output).view(
            B, T, -1, self.nband).permute(0, 3, 2, 1).contiguous())
        return output.view(B, N, T), next_state


class BSRNN_Separator(nn.Module):
    """num_repeat 个 BSNet 堆叠。"""

    def __init__(
        self,
        nband=7,
        num_repeat=6,
        feature_dim=128,
        causal=False,
        norm_type='GN',
        eps=None,
    ):
        super(BSRNN_Separator, self).__init__()
        self.nband = nband
        self.feature_dim = feature_dim
        self.separation = nn.ModuleList([])
        for _ in range(num_repeat):
            self.separation.append(
                BSNet(nband * feature_dim, nband, causal, norm_type, eps=eps))

    @property
    def num_blocks(self):
        return len(self.separation)

    def forward_block(self, x, index, initial_state=None):
        """运行一个 BSNet block。"""
        if not 0 <= index < self.num_blocks:
            raise IndexError(f"BSNet block index out of range: {index}")
        batch_size = x.shape[0]
        x = x.view(batch_size, self.nband * self.feature_dim, -1)
        x = self.separation[index](x, initial_state=initial_state)
        return x.view(batch_size, self.nband, self.feature_dim, -1)

    def forward(self, x, initial_states=None):
        """
        x: [B, nband, feature_dim, T]
        initial_states: 单个 state dict 或每个 BSNet block 一个 state dict
        out: [B, nband, feature_dim, T]
        """
        if isinstance(initial_states, (list, tuple)) and len(initial_states) != self.num_blocks:
            raise ValueError(
                f"BSRNN initial_states 数量应为 {self.num_blocks}，"
                f"got {len(initial_states)}"
            )
        for index in range(self.num_blocks):
            state = (
                initial_states[index]
                if isinstance(initial_states, (list, tuple))
                else initial_states
            )
            x = self.forward_block(x, index, initial_state=state)
        return x

    def forward_stream(self, x, states):
        """携带每个 BSNet 的时间 LSTM state 处理连续 chunk。"""
        if len(states) != len(self.separation):
            raise ValueError(
                f"BSRNN streaming state 数量应为 {len(self.separation)}，"
                f"got {len(states)}"
            )
        batch_size = x.shape[0]
        x = x.view(batch_size, self.nband * self.feature_dim, -1)
        next_states = []
        for sep, state in zip(self.separation, states):
            x, next_state = sep.forward_stream(x, state)
            next_states.append(next_state)
        x = x.view(batch_size, self.nband, self.feature_dim, -1)
        return x, next_states

    def forward_conditioned(
        self, x, fusion_layers, condition, condition_transform=None
    ):
        """在每个 BSNet 前用独立层融合固定 speaker condition。"""
        output, _ = self.forward_stream_conditioned(
            x,
            [None] * len(self.separation),
            fusion_layers,
            condition,
            condition_transform,
        )
        return output

    def forward_stream_conditioned(
        self,
        x,
        states,
        fusion_layers,
        condition,
        condition_transform=None,
    ):
        """``forward_stream`` 的 block-level speaker fusion 版本。"""
        if len(states) != len(self.separation):
            raise ValueError(
                f"BSRNN streaming state 数量应为 {len(self.separation)}，"
                f"got {len(states)}"
            )
        if len(fusion_layers) != len(self.separation):
            raise ValueError(
                f"multi_fuse 层数量应为 {len(self.separation)}，"
                f"got {len(fusion_layers)}"
            )
        batch_size = x.shape[0]
        next_states = []
        for fusion, sep, state in zip(fusion_layers, self.separation, states):
            current_condition = (
                condition_transform(x, condition)
                if condition_transform is not None
                else condition
            )
            x = fusion(x, current_condition)
            x = x.view(batch_size, self.nband * self.feature_dim, -1)
            x, next_state = sep.forward_stream(x, state)
            next_states.append(next_state)
            x = x.view(batch_size, self.nband, self.feature_dim, -1)
        return x, next_states


class BSRNN(nn.Module):
    """BSRNN 频带分离主体（不含 STFT/iSTFT，谱由外部喂入）。

    子模块命名（``band_split / subband_norm / separator / band_masker``）与
    wesep 原实现一致，官方 checkpoint 的 ``sep_model.*`` 键可原样加载。
    """

    def __init__(
            self,
            sr=16000,
            win=512,
            stride=128,
            feature_dim=128,
            num_repeat=6,
            causal=False,
            nspk=1,  # 多说话人分离输出数（TSE 固定为 1）
            spec_dim=2,  # TSE cue 拼接后的谱通道数，用于 self.subband_norm
    ):
        super().__init__()

        for name, value in (("sr", sr), ("win", win), ("stride", stride), ("feature_dim", feature_dim), ("num_repeat", num_repeat), ("nspk", nspk), ("spec_dim", spec_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if win % 2:
            raise ValueError(f"win must be even, got {win}")
        if stride > win:
            raise ValueError(f"stride must not exceed win, got {stride} > {win}")

        self.sr = sr
        self.win = win
        self.stride = stride
        self.feature_dim = feature_dim
        self.nspk = nspk
        self.eps = torch.finfo(torch.float32).eps

        norm_type = "cLN" if causal else "GN"

        # 子带表（16kHz / win=512 下 32 子带）：
        # 0-1.5k (100Hz) × 15, 1.5k-3.5k (200Hz) × 10, 3.5k-6k (500Hz) × 5,
        # 6k-8k (2kHz) × 1，余量归入最后一个子带
        enc_dim = win // 2 + 1
        bandwidth_100 = int(np.floor(100 / (sr / 2.0) * enc_dim))
        bandwidth_200 = int(np.floor(200 / (sr / 2.0) * enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.0) * enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sr / 2.0) * enc_dim))

        band_width = [bandwidth_100] * 15
        band_width += [bandwidth_200] * 10
        band_width += [bandwidth_500] * 5
        band_width += [bandwidth_2k] * 1
        band_width.append(enc_dim - int(np.sum(band_width)))
        invalid_band_widths = [
            (index, width) for index, width in enumerate(band_width) if width <= 0
        ]
        if invalid_band_widths:
            raise ValueError(
                f"BandSplit 子带表包含非正宽度：sr={sr}, win={win}, "
                f"invalid={invalid_band_widths}，请检查 sr/n_fft 组合"
            )
        nband = len(band_width)
        self.nband = nband

        self.band_split = BandSplit(band_width)
        self.subband_norm = SubbandNorm(
            band_width=band_width,
            spec_dim=spec_dim,
            nband=nband,
            feature_dim=feature_dim,
            norm_type=norm_type,
        )
        self.separator = BSRNN_Separator(
            nband=nband,
            num_repeat=num_repeat,
            feature_dim=feature_dim,
            causal=causal,
            norm_type=norm_type,
        )
        self.band_masker = BandMasker(
            band_width=band_width,
            nband=nband,
            feature_dim=feature_dim,
            norm_type=norm_type,
            nspk=nspk,
        )
