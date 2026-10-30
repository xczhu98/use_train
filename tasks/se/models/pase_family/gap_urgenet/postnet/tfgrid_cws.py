# Copyright espnet
# Adapted under Apache-2.0
# 算法基础来源：https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnetv3_separator.py
# 直接移植来源：https://github.com/Xiaobin-Rong/gap-urgenet 的
# ``models/postnet/tfgrid_cws.py:23-198``。
# 本仓训练 recipe：
# ``configs/se/urgent2026/pase/gap_urgenet/gap_urgenet_phase2_postnet_gan.yaml``。
# 官方发布配置见 gap-urgenet/configs/cfg_train_postnet.yaml:17-23
# （postnet_config：n_imics=2, n_layers=5, lstm_hidden_units=100, attn_n_head=4,
# attn_qk_output_channel=2, emb_dim=96；n_srcs=1/n_bands=3/emb_ks=4/emb_hs=1 用
# 类默认值；发布 PostNet.pt 参数量 4.78M）。

import torch
import torch.nn as nn
from torchaudio.functional import resample

from ...common.utils import hann_window_buffer, masked_time_stats
from ...unipase.postnet.tfgrid_cws_res import BandSplit, GridNetV3Block


def multi_channel_stft(x, n_fft, hop_length, win_length, window, onesided=True):
    """多通道 STFT（官方 :23-31）。x: (batch, channels, n_samples) -> (B, C, F, T) complex。"""
    bs, ch = x.shape[0], x.shape[1]
    x = x.view(bs * ch, -1)  # (B*C, L)

    X = torch.stft(x, n_fft, hop_length, win_length, window, onesided=onesided, return_complex=True)  # (B*C, F, T)
    X = X.view(bs, ch, X.shape[1], X.shape[2])  # (B,C,F,T), complex

    return X


class TFGridNet(nn.Module):
    """Offline TFGridNetV3 (GAP-URGENet PostNet, 48kHz 双通道融合).

    Compared with TFGridNet, TFGridNetV3 speeds up the code by vectorizing
    multiple heads in self-attention, and better dealing with Deconv1D in each
    intra- and inter-block when emb_ks == emb_hs.

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in TASLP, 2023.

    STFT 配置硬编码在官方实现中：n_fft = int(48000*0.032) = 1536，hop = 768 @48kHz。
    BandSplit 的 assert n_freqs == 769 即绑定该配置（48k / 32ms 专用，769 =
    1536//2+1，3 个 257 频点重叠子带）。

    输入为预测支路与生成支路上采样到 48k 后的双通道混合 (B, 2, T)。
    ``legacy_centered`` 执行 (x-mean)/std，``paper_variance`` 仅执行 x/std；
    两者输出均乘回同一 std。

    Args:
        n_srcs: number of output sources/speakers.
        n_imics: number of microphones channels (GAP-URGENet 发布配置为 2).
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
        normalization_mode="legacy_centered",
    ):
        super().__init__()
        if normalization_mode not in {"legacy_centered", "paper_variance"}:
            raise ValueError(
                "normalization_mode must be 'legacy_centered' or 'paper_variance', "
                f"got {normalization_mode!r}"
            )
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        self.normalization_mode = normalization_mode

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
        """官方 :121-170。x: (B, C, T)（C = n_imics）。Returns: (B, T)，与输入同长同采样率。

        ``lengths`` 为 ``(B,)`` 有效样本数：传入时归一化统计只覆盖有效区，
        避免 batch 右侧补零改变同一条输入的输出。
        """
        n_samples = x.shape[-1]

        if self.n_imics == 1:
            x = x.squeeze(1)
            stat_dim = 1
        else:
            if x.ndim != 3:
                raise ValueError(
                    f"n_imics>1 时输入必须是 (B, C, T)，got {tuple(x.shape)}"
                )
            stat_dim = (1, 2)
        mean_, std_ = masked_time_stats(x, lengths, dim=stat_dim)
        if self.normalization_mode == "legacy_centered":
            x = x - mean_
        x = x / std_

        if sr_in != 48000:  # e.g., 44100
            x = resample(x, sr_in, 48000)

        nfft = int(48000 * 0.032)
        nhop = nfft // 2
        window = hann_window_buffer(self, nfft, x.device)

        if self.n_imics == 1:
            input = torch.stft(x, nfft, nhop, nfft, window, return_complex=True)
            input = torch.view_as_real(input)  # (B,F,T,2)
            batch = input.permute(0, 3, 2, 1)  # [B, 2, T, F]
        else:
            input = multi_channel_stft(x, nfft, nhop, nfft, window)  # (B,C,F,T), complex
            input = torch.view_as_real(input)  # (B,C,F,T,2)
            # 官方 :149 为 einops rearrange('b c f t ri -> b (c ri) t f')；
            # 等价为 permute 到 (B,C,ri,T,F) 后 reshape 合并 (C,ri) 两维
            bs, ch = input.shape[0], input.shape[1]
            batch = input.permute(0, 1, 4, 3, 2).reshape(bs, ch * 2, input.shape[3], input.shape[2])

        batch = self.bs(batch)  # [B, C*2*n_bands, T, F=257]
        batch = self.conv(batch)  # [B, -1, T, F]

        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        batch = self.deconv(batch)  # [B, n_srcs*2*n_bands, T, F]

        batch = self.bs.inverse(batch)  # [B, 2, T, F]
        batch = batch.permute(0, 3, 2, 1)  # [B, F, T, 2]

        output = torch.istft(
            batch[..., 0] + 1j * batch[..., 1], nfft, nhop, nfft, window
        )

        if sr_in != 48000:
            output = resample(output, 48000, sr_in, lowpass_filter_width=64)
        # F.pad 负值即裁剪：输出比 n_samples 长时静默截断（对齐官方）
        output = torch.nn.functional.pad(output, (0, n_samples - output.shape[-1]))

        # n_imics==1 时 std_ 的 shape 无法与 (B, L) 输出广播，GAP-URGENet 必须使用 n_imics=2。
        output = output * std_.squeeze(1)

        return output

    @classmethod
    def from_pretrained(cls, ckpt_path: str, frozen: bool = True) -> "TFGridNet":
        """Load from a GAP-URGENet postnet checkpoint.

        checkpoint 需含 'cfg' 以及 'generator' 或 'model' 权重（官方 :172-193）。
        加载语义统一走 tasks.se.models.pase_family.common.utils.load_artifact_module
        （weights_only=True + key 匹配报告，任何 missing/unexpected 都 fail closed）。
        """
        from ...common.utils import load_artifact_module

        return load_artifact_module(cls, ckpt_path, frozen=frozen)

    @property
    def num_spk(self):
        return self.n_srcs
