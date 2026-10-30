# Copyright espnet
# Adapted under Apache-2.0
# 算法基础来源：https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnetv3_separator.py
# 直接移植来源：https://github.com/Xiaobin-Rong/gap-urgenet 的
# ``models/predictor/tfgridnet.py:21-169``。
# 本仓训练 recipe：
# ``configs/se/urgent2026/pase/gap_urgenet/gap_urgenet_phase1_predictor.yaml``。
#
# 与官方的差异（均为去依赖化/复用，无数值影响）：
#   1. 去掉 espnet 依赖：官方文件 import AbsSeparator / get_layer / OmegaConf 但
#      TFGridNet 本身继承 nn.Module、未实际使用它们，直接删除；
#   2. GridNetV3Block / LayerNormalization / AllHeadPReLULayerNormalization4DC
#      与官方 models/postnet/tfgrid_cws.py 及本框架
#      unipase/postnet/tfgrid_cws_res.py 中的同名类逐行比对确认数值行为一致，
#      直接 import 复用（其中 espnet get_layer("prelu") 已替换为 nn.PReLU()，
#      见该文件 attn_concat_proj 处注释）；
#   3. from_pretrained 统一走 tasks.se.models.pase_family.common.utils.load_artifact_module
#      （weights_only=True + key 匹配报告，任何 missing/unexpected 都 fail closed），
#      替换官方 strict=False 的裸 torch.load。
#
# 官方发布配置见 gap-urgenet/configs/cfg_train_predictor.yaml:1-15
# （predictor_config：fft_len=0.032, hop_len=0.016, win_len=0.032, n_srcs=1,
# n_imics=1, n_layers=10, lstm_hidden_units=200, attn_n_head=8,
# attn_qk_output_channel=4, emb_dim=96, emb_ks=4, emb_hs=1, activation=prelu,
# eps=1e-5；发布 Predictor.pt 参数量 22.09M）。

import torch
import torch.nn as nn

from ...common.utils import hann_window_buffer, masked_time_stats
from ...unipase.postnet.tfgrid_cws_res import GridNetV3Block


class TFGridNet(nn.Module):
    """Offline TFGridNetV3 (GAP-URGENet Predictor, 16kHz 单通道复数谱直接回归).

    Compared with TFGridNet, TFGridNetV3 speeds up the code by vectorizing
    multiple heads in self-attention, and better dealing with Deconv1D in each
    intra- and inter-block when emb_ks == emb_hs.

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in TASLP, 2023.

    STFT 按时长比例参数化：fft_len/hop_len/win_len（秒）乘以 sr 得采样点数，
    16k 下即 n_fft=512 / hop=256 / win=512（hann 窗）。注意官方发布配置
    emb_ks=4 != emb_hs=1，GridNetV3Block 走 unfold/ConvTranspose1d 分支。

    与 UniPASE PostNet 变体的区别：无 BandSplit、无残差/软门控，输入逐
    utterance (x-mean)/(std+1e-12) 归一化，网络直接回归复数谱（无 mask 无
    压缩），istft 后乘回 std（官方 :120-145）。

    Args:
        fft_len: STFT 窗长（秒），16k 下 0.032s = 512 点。
        hop_len: STFT 帧移（秒），16k 下 0.016s = 256 点。
        win_len: STFT 分析窗长（秒），16k 下 0.032s = 512 点。
        n_srcs: number of output sources/speakers.
        n_imics: number of microphone channels (Predictor 仅支持单通道).
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
        fft_len=0.032,
        hop_len=0.016,
        win_len=0.032,
        n_srcs=1,
        n_imics=1,
        n_layers=6,
        lstm_hidden_units=200,
        attn_n_head=4,
        attn_qk_output_channel=2,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
    ):
        super().__init__()
        self.fft_len = fft_len
        self.hop_len = hop_len
        self.win_len = win_len
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
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

        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)

        # 默认 16kHz（Predictor 的官方工作采样率）窗构造期注册；其他 sr 由
        # hann_window_buffer 在首次前向时按长度补注册
        hann_window_buffer(self, int(16000 * self.win_len), torch.device("cpu"))

    def forward(self, x, sr=16000, lengths=None):
        """官方 :113-145。x: (B, T) 或 (B, 1, T)。Returns: (B, T)，与输入同长。

        ``lengths`` 为 ``(B,)`` 有效样本数：传入时归一化统计只覆盖有效区，
        避免 batch 右侧补零改变同一条 utterance 的输出。
        """
        if self.n_imics != 1:
            raise ValueError(
                f"TFGridNet (GAP-URGENet Predictor) 仅支持单通道输入，got n_imics={self.n_imics}"
            )
        if x.ndim == 3:
            x = x.squeeze(1)
        n_samples = x.shape[-1]

        # 逐 utterance 方差归一化（官方 NOTES：模型在归一化输入上训练）
        mean_, std_ = masked_time_stats(x, lengths, dim=1)
        x = (x - mean_) / std_

        nfft = int(sr * self.fft_len)
        nhop = int(sr * self.hop_len)
        nwin = int(sr * self.win_len)
        input = torch.stft(
            x, nfft, nhop, nwin, hann_window_buffer(self, nwin, x.device), return_complex=True
        )
        input = torch.view_as_real(input)  # (B, F, T, 2)

        batch = input.permute(0, 3, 2, 1)  # [B, 2, T, F]
        batch = self.conv(batch)  # [B, -1, T, F]

        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]

        batch = batch.permute(0, 3, 2, 1)  # [B, F, T, 2]

        # 复数谱直接回归（无 mask 无压缩），istft 合成窗用 nfft 长 hann（官方 :140）
        output = torch.istft(
            batch[..., 0] + 1j * batch[..., 1], nfft, nhop, nfft,
            hann_window_buffer(self, nfft, x.device),
        )
        output = output * std_

        # F.pad 负值即裁剪：输出比 n_samples 长时静默截断（对齐官方）
        output = torch.nn.functional.pad(output, (0, n_samples - output.shape[-1]))

        return output

    @classmethod
    def from_pretrained(cls, ckpt_path: str, frozen: bool = True) -> "TFGridNet":
        """Load from a GAP-URGENet predictor checkpoint.

        checkpoint 需含 'cfg' 以及 'generator' 或 'model' 权重（官方 :148-169）。
        加载语义统一走 tasks.se.models.pase_family.common.utils.load_artifact_module
        （weights_only=True + key 匹配报告，任何 missing/unexpected 都 fail closed）。
        """
        from ...common.utils import load_artifact_module

        return load_artifact_module(cls, ckpt_path, frozen=frozen)

    @property
    def num_spk(self):
        return self.n_srcs
