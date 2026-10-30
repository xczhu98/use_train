# Copyright espnet
# Adapted under Apache-2.0
# 算法基础来源：https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnetv3_separator.py
# 直接移植来源：https://github.com/Xiaobin-Rong/UniPASE 的
# ``models/postnet/tfgrid_cws_res.py``。
# 本仓训练 recipe：``configs/se/urgent2026/pase/unipase/unipase_phase4_postnet_gan.yaml``。
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torchaudio.functional import resample

from ...common.utils import hann_window_buffer, masked_time_stats

logger = logging.getLogger(__name__)

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class TFGridNet(nn.Module):
    """Offline TFGridNetV3 (UniPASE PostNet, 48kHz 宽带扩展).

    Compared with TFGridNet, TFGridNetV3 speeds up the code by vectorizing
    multiple heads in self-attention, and better dealing with Deconv1D in each
    intra- and inter-block when emb_ks == emb_hs.

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in TASLP, 2023.

    STFT 配置硬编码在官方实现中：n_fft = int(48000*0.032) = 1536，hop = 768 @48kHz。
    官方发布配置见 configs/cfg_train_postnet.yaml:11-17（n_layers=5, emb_dim=48,
    lstm_hidden_units=100, attn_n_head=4, attn_qk_output_channel=2）。

    注意 train/infer 的有意不一致（保留官方行为）：
      - forward()（训练）：全时域残差，output = (output + x_ref) * std_；
      - infer()（推理）：软门控残差——<=7.2kHz 完全保留输入谱，7.2~8kHz 为
        800Hz 过渡带，仅 >8kHz 频段叠加网络输出的残差谱，避免破坏低频内容。

    Args:
        n_srcs: number of output sources/speakers.
        n_imics: number of microphones channels (only single-channel supported).
        n_bands: number of sub-bands for BandSplit.
        n_layers: number of TFGridNetV3 blocks.
        lstm_hidden_units: number of hidden units in LSTM.
        attn_n_head: number of heads in self-attention.
        attn_qk_output_channel: output channels of point-wise conv2d for key/query.
        emb_dim: embedding dimension.
        emb_ks: kernel size for unfolding and deconv1D.
        emb_hs: hop size for unfolding and deconv1D.
        activation: activation function (only "prelu" supported).
        eps: small epsilon for normalization layers.
    """

    def __init__(
        self,
        n_srcs=1,
        n_imics=1,
        n_bands=3,
        n_layers=5,
        lstm_hidden_units=100,
        attn_n_head=4,
        attn_qk_output_channel=2,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics

        self.bs = BandSplit(n_bands=n_bands)
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics * n_bands, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetV3Block(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    qk_output_channel=attn_qk_output_channel,
                    activation=activation,
                    eps=eps,
                )
            )

        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2 * n_bands, ks, padding=padding)

        # STFT/iSTFT 共用窗：构造期注册为 non-persistent buffer（48k/32ms 固定配置）
        hann_window_buffer(self, int(48000 * 0.032), torch.device("cpu"))

    def forward(self, x, sr_in=48000, lengths=None):
        """训练用前向（官方 :123-163）：全时域残差。

        Args:
            x: (B, C, T) 或 (B, T)，有效带宽 8kHz 的输入信号。
            lengths: (B,) 有效样本数；传入时归一化统计只覆盖有效区，
                避免 batch 右侧补零改变同一条输入的输出。
        Returns:
            output: (B, T)，与输入同采样率，output = (output + x_ref) * std_。
        """
        n_samples = x.shape[-1]
        if self.n_imics != 1:
            raise ValueError(
                f"TFGridNet (UniPASE PostNet) 仅支持单通道输入，got n_imics={self.n_imics}"
            )

        x = x.squeeze(1)
        mean_, std_ = masked_time_stats(x, lengths, dim=1)
        x = (x - mean_) / std_
        x_ref = x.clone()

        if sr_in != 48000:
            x = resample(x, sr_in, 48000)

        nfft = int(48000 * 0.032)
        nhop = nfft // 2
        window = hann_window_buffer(self, nfft, x.device)

        input = torch.stft(x, nfft, nhop, nfft, window, return_complex=True)
        input = torch.view_as_real(input)  # (B,F,T,2)
        batch = input.permute(0, 3, 2, 1)  # [B, 2, T, F]

        batch = self.bs(batch)  # [B, 6, T, F=257]
        batch = self.conv(batch)  # [B, -1, T, F]

        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]

        batch = self.bs.inverse(batch)  # [B, 2, T, F]
        batch = batch.permute(0, 3, 2, 1)  # [B, F, T, 2]

        output = torch.istft(
            batch[..., 0] + 1j * batch[..., 1], nfft, nhop, nfft, window
        )

        if sr_in != 48000:
            output = resample(output, 48000, sr_in)
        # F.pad 负值即裁剪：输出比 n_samples 长时静默截断（对齐官方；review P3-14）
        output = torch.nn.functional.pad(output, (0, n_samples - output.shape[-1]))

        output = (output + x_ref) * std_

        return output

    @classmethod
    def from_pretrained(cls, ckpt_path: str, frozen: bool = True) -> "TFGridNet":
        """Load from a UniPASE postnet checkpoint.

        checkpoint 需含 'cfg' 以及 'generator' 或 'model' 权重（官方 :166-187）。
        加载语义统一走 tasks.se.models.pase_family.common.utils.load_artifact_module
        （weights_only=True + key 匹配报告，任何 missing/unexpected 都 fail closed）。
        """
        from ...common.utils import load_artifact_module

        return load_artifact_module(cls, ckpt_path, frozen=frozen)

    @torch.inference_mode()
    def infer(self, x, sr_in=16000, sr_out=None, lengths=None):
        """推理用前向（官方 :190-262）：软门控残差（与 forward 的训练行为有意不同）。

        <= cutoff_bw - transition_bw (7.2kHz) 的频段完全保留输入谱（alpha=1），
        7.2~8kHz 为过渡带（alpha 线性衰减到 0），仅 >8kHz 频段叠加网络输出的
        残差谱，实现 8k -> 24kHz 的宽带扩展而不破坏低频。

        Args:
            x: (B, C, T) 或 (B, T)，有效带宽 8kHz 的输入信号。
            sr_in: 输入采样率。
            sr_out: 输出采样率，None 时等于 sr_in。
            lengths: (B,) 有效样本数；传入时归一化统计只覆盖有效区。
        Returns:
            output: (B, T)，sr_out 采样率的全带信号。
        """
        n_samples = x.shape[-1]
        if self.n_imics != 1:
            raise ValueError(
                f"TFGridNet (UniPASE PostNet) 仅支持单通道输入，got n_imics={self.n_imics}"
            )

        if sr_out is None:
            sr_out = sr_in

        if x.ndim == 3:
            x = x.squeeze(1)
        mean_, std_ = masked_time_stats(x, lengths, dim=1)
        x = (x - mean_) / std_

        if sr_in != 48000:
            x = resample(x, sr_in, 48000)

        nfft = int(48000 * 0.032)
        nhop = nfft // 2
        window = hann_window_buffer(self, nfft, x.device)

        input = torch.stft(x, nfft, nhop, nfft, window, return_complex=True)
        input = torch.view_as_real(input)  # (B,F,T,2)
        spec_ref = input.clone()  # (B,F=769,T,2)
        batch = input.permute(0, 3, 2, 1)  # [B, 2, T, F]

        batch = self.bs(batch)  # [B, 6, T, F=257]
        batch = self.conv(batch)  # [B, -1, T, F]

        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]

        batch = self.bs.inverse(batch)  # [B, 2, T, F]
        batch = batch.permute(0, 3, 2, 1)  # [B, F, T, 2]

        freqs = torch.arange(spec_ref.shape[1], device=batch.device) * 48000 / nfft
        cutoff_bw = 8000     # cutoff frequency
        transition_bw = 800  # transition bandwidth

        alpha = torch.zeros_like(freqs)
        alpha[freqs <= cutoff_bw - transition_bw] = 1.0
        mask = (freqs > cutoff_bw - transition_bw) & (freqs <= cutoff_bw)
        alpha[mask] = (cutoff_bw - freqs[mask]) / transition_bw
        alpha = alpha[None, :, None, None]
        batch_out = spec_ref + (1 - alpha) * batch

        output = torch.istft(
            batch_out[..., 0] + 1j * batch_out[..., 1], nfft, nhop, nfft, window
        )

        if sr_out != 48000:
            output = resample(output, orig_freq=48000, new_freq=sr_out)

        if sr_out == sr_in:
            out_samples = n_samples
        else:
            out_samples = int(n_samples * sr_out / sr_in)

        if output.shape[-1] < out_samples:
            output = torch.nn.functional.pad(output, (0, out_samples - output.shape[-1]))
        else:
            output = output[..., :out_samples]

        output = output * std_

        return output

    @property
    def num_spk(self):
        return self.n_srcs


class BandSplit(nn.Module):
    """
    Split bands for 48 kHz spectrogram
    """

    def __init__(self, n_bands=3):
        super().__init__()
        self.n_bands = n_bands

    def forward(self, x):
        """x: (B,C,T,F)"""
        n_freqs = x.shape[-1]
        if n_freqs != 769:
            raise ValueError(
                f"BandSplit 绑定 48kHz/32ms 配置（769 频点），got n_freqs={n_freqs}"
            )
        sub_nfreqs = n_freqs // self.n_bands + 1
        x_bands = []
        for i in range(self.n_bands):
            x_bands.append(x[..., i * (sub_nfreqs - 1): i * (sub_nfreqs - 1) + sub_nfreqs])

        return torch.cat(x_bands, dim=1).contiguous()

    def inverse(self, x_bands):
        """x_bands: (B,C*n,T,F)"""
        if x_bands.shape[1] != self.n_bands * 2:
            raise ValueError(
                f"BandSplit.inverse 期望 {self.n_bands * 2} 通道，got {x_bands.shape[1]}"
            )
        x = [x_bands[:, :2]]
        for i in range(1, self.n_bands):
            x.append(x_bands[:, 2 * i: 2 * (i + 1), :, 1:])

        return torch.cat(x, dim=-1).contiguous()


class GridNetV3Block(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        hidden_channels,
        n_head=4,
        qk_output_channel=4,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()
        if activation != "prelu":
            raise ValueError(f"GridNetV3Block only supports activation='prelu', got {activation!r}")

        in_channels = emb_dim * emb_ks

        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        # use constant E not to be dependent on the number of frequency bins
        E = qk_output_channel
        if emb_dim % n_head != 0:
            raise ValueError(f"emb_dim ({emb_dim}) 必须能整除 n_head ({n_head})")

        self.add_module("attn_conv_Q", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_Q",
            AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps),
        )

        self.add_module("attn_conv_K", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_K",
            AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps),
        )

        self.add_module(
            "attn_conv_V", nn.Conv2d(emb_dim, n_head * emb_dim // n_head, 1)
        )
        self.add_module(
            "attn_norm_V",
            AllHeadPReLULayerNormalization4DC(
                (n_head, emb_dim // n_head), eps=eps
            ),
        )

        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                # 官方为 espnet get_layer(activation)()；activation 已 assert 为
                # "prelu"，get_layer("prelu") 即 torch.nn.PReLU，此处直接实例化
                nn.PReLU(),
                LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """GridNetV2Block Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape

        olp = self.emb_ks - self.emb_hs
        T = (
            math.ceil((old_T + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )
        Q = (
            math.ceil((old_Q + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )

        x = x.permute(0, 2, 3, 1)  # [B, old_T, old_Q, C]
        x = F.pad(x, (0, 0, olp, Q - old_Q - olp, olp, T - old_T - olp))  # [B, T, Q, C]

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, T, Q, C]
        if self.emb_ks == self.emb_hs:
            intra_rnn = intra_rnn.view([B * T, -1, self.emb_ks * C])  # [BT, Q//I, I*C]
            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, Q//I, H]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, Q//I, I*C]
            intra_rnn = intra_rnn.view([B, T, Q, C])
        else:
            intra_rnn = intra_rnn.view([B * T, Q, C])  # [BT, Q, C]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, C, Q]
            intra_rnn = F.unfold(
                intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BT, C*I, -1]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*I]

            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]

            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2, -1)  # [B, T, Q, C]
        intra_rnn = intra_rnn + input_  # [B, T, Q, C]

        intra_rnn = intra_rnn.transpose(1, 2)  # [B, Q, T, C]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, Q, T, C]
        if self.emb_ks == self.emb_hs:
            inter_rnn = inter_rnn.view([B * Q, -1, self.emb_ks * C])  # [BQ, T//I, I*C]
            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, T//I, H]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, T//I, I*C]
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = inter_rnn.view(B * Q, T, C)  # [BQ, T, C]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, C, T]
            inter_rnn = F.unfold(
                inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BQ, C*I, -1]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, -1, C*I]

            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, -1, H]

            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, H, -1]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, C, T]
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2, -1)  # [B, Q, T, C]
        inter_rnn = inter_rnn + input_  # [B, Q, T, C]

        inter_rnn = inter_rnn.permute(0, 3, 2, 1)  # [B, C, T, Q]

        inter_rnn = inter_rnn[..., olp: olp + old_T, olp: olp + old_Q]
        batch = inter_rnn

        Q = self["attn_norm_Q"](self["attn_conv_Q"](batch))  # [B, n_head, C, T, Q]
        K = self["attn_norm_K"](self["attn_conv_K"](batch))  # [B, n_head, C, T, Q]
        V = self["attn_norm_V"](self["attn_conv_V"](batch))  # [B, n_head, C, T, Q]
        Q = Q.view(-1, *Q.shape[2:])  # [B*n_head, C, T, Q]
        K = K.view(-1, *K.shape[2:])  # [B*n_head, C, T, Q]
        V = V.view(-1, *V.shape[2:])  # [B*n_head, C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]

        K = K.transpose(2, 3)
        K = K.contiguous().view([B * self.n_head, -1, old_T])  # [B', C*Q, T]

        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.contiguous().view(
            [B, self.n_head * emb_dim, old_T, old_Q]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out


class LayerNormalization(nn.Module):
    def __init__(self, input_dim, dim=1, total_dim=4, eps=1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        param_size = [1 if ii != self.dim else input_dim for ii in range(total_dim)]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x):
        if x.ndim - 1 < self.dim:
            raise ValueError(
                f"Expect x to have {self.dim + 1} dimensions, but got {x.ndim}"
            )
        if x.dtype in HALF_PRECISION_DTYPES:
            dtype = x.dtype
            x = x.float()
        else:
            dtype = None
        mu_ = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat.to(dtype=dtype) if dtype else x_hat


class AllHeadPReLULayerNormalization4DC(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        if len(input_dimension) != 2:
            raise ValueError(f"input_dimension 必须是 (H, E) 二元组，got {input_dimension!r}")
        H, E = input_dimension
        param_size = [1, H, E, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"AllHeadPReLULayerNormalization4DC 期望 4D 输入，got {x.ndim}D")
        B, _, T, F = x.shape
        x = x.view([B, self.H, self.E, T, F])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2,)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x
