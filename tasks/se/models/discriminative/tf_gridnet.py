"""TF-GridNet: Time-Frequency GridNet for Speech Enhancement (频谱进/频谱出).

论文: "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
      Speaker Separation", Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim,
      S. Watanabe, ICASSP 2023.
      及扩展: "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech
      Separation", TASLP 2023.
参考实现:
      - ESPnet TFGridNetV3: https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnetv3_separator.py
      - X-TF-GridNet: https://github.com/HaoFengyuan/X-TF-GridNet
      - se-scaling: https://github.com/Emrys365/se-scaling

模块分区（速查）:
  - LayerNormalization4D: 可沿任意维度的 LayerNorm（论文中各处 normalization）
  - AllHeadPReLULayerNorm4DC: 多头 PReLU + LayerNorm（自注意力 Q/K/V 后）
  - GridNetBlock: Intra-frame BLSTM（频率维） + Sub-band BLSTM（时间维） + Multi-head Self-Attention（论文 Figure 2）
  - TFGridNet: Conv2d embedding → N × GridNetBlock → ConvTranspose2d 解码（论文 Figure 1 完整架构）

谱域接口:
  输入 noisy_spec (B, F, T, 2)；输出 enhanced_spec (B, F, T, 2)。
  STFT/iSTFT 由训练框架完成。

脚本: python tasks/se/models/tf_gridnet.py [--sr 16000] [--dur 1.0] [--device cpu]
"""

# When run directly, add project root to sys.path
if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from tasks.se.models.amp_utils import fp32_forward
from tasks.se.models.base_model import BaseSEModel
from core.utils.registry import MODEL_REGISTRY


# =============================================================================
# LayerNormalization — 可沿任意维度的 LayerNorm (ESPnet TFGridNetV3 同款)
# =============================================================================


class LayerNormalization4D(nn.Module):
    """LayerNorm along a specified dim in a 4-D tensor.

    Args:
        input_dim: Size of the normalization dimension.
        dim: Which dimension to normalize (supports negative indexing).
        total_dim: Total number of dimensions (default 4).
        eps: Epsilon for numerical stability.
    """

    def __init__(self, input_dim: int, dim: int = 1, total_dim: int = 4, eps: float = 1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        param_size = [1 if ii != self.dim else input_dim for ii in range(total_dim)]
        self.gamma = Parameter(torch.ones(*param_size))
        self.beta = Parameter(torch.zeros(*param_size))
        self.eps = eps

    @fp32_forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim - 1 < self.dim:
            raise ValueError(f"Expected at least {self.dim + 1} dims, got {x.ndim}")
        dtype = x.dtype
        if dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        mu = x.mean(dim=self.dim, keepdim=True)
        std = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x = ((x - mu) / std) * self.gamma + self.beta
        return x.to(dtype) if dtype in (torch.float16, torch.bfloat16) else x


# =============================================================================
# AllHeadPReLULayerNorm4DC — 多头 PReLU + 逐头 LayerNorm (Q/K/V 投影后)
# =============================================================================


class AllHeadPReLULayerNorm4DC(nn.Module):
    """Per-head PReLU activation + LayerNorm for multi-head attention projections.

    Input: (B, H*E, T, F) -> reshape to (B, H, E, T, F) -> PReLU -> LN along E dim.

    Args:
        input_dimension: (n_heads, E_per_head) tuple.
        eps: Epsilon for LayerNorm.
    """

    def __init__(self, input_dimension: tuple[int, int], eps: float = 1e-5):
        super().__init__()
        H, E = input_dimension
        param_size = [1, H, E, 1, 1]
        self.gamma = Parameter(torch.ones(*param_size))
        self.beta = Parameter(torch.zeros(*param_size))
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H*E, T, F) -> (B, H, E, T, F)."""
        B, _, T, F = x.shape
        x = x.view(B, self.H, self.E, T, F)
        x = self.act(x)
        mu = x.mean(dim=2, keepdim=True)
        std = torch.sqrt(x.var(dim=2, unbiased=False, keepdim=True) + self.eps)
        x = ((x - mu) / std) * self.gamma + self.beta
        return x


# =============================================================================
# GridNetBlock — 频域 BLSTM + 时域 BLSTM + 多头自注意力 (论文 Figure 2)
# =============================================================================


class GridNetBlock(nn.Module):
    """Single TF-GridNet block: intra-BLSTM (freq) + inter-BLSTM (time) + self-attention.

    论文 Figure 2 结构:
      1. Intra-frame BLSTM: 对每帧的频率维序列建模 (Full-band modeling)
      2. Sub-band BLSTM: 对每个频率 bin 的时间维序列建模 (Sub-band modeling)
      3. Multi-head self-attention: 时间维全局注意力

    Args:
        emb_dim: Embedding dimension (D).
        emb_ks: Unfold kernel size for sub-band grouping.
        emb_hs: Unfold hop size for sub-band grouping.
        hidden_channels: LSTM hidden units.
        n_head: Number of attention heads.
        qk_output_channel: Q/K projection output channels per head (E).
        activation: Activation function name (default 'prelu').
        eps: LayerNorm epsilon.
    """

    def __init__(
        self,
        emb_dim: int,
        emb_ks: int,
        emb_hs: int,
        hidden_channels: int,
        n_head: int = 4,
        qk_output_channel: int = 4,
        activation: str = "prelu",
        eps: float = 1e-5,
    ):
        super().__init__()
        in_channels = emb_dim * emb_ks

        # --- Intra-frame BLSTM (frequency dimension) ---
        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True,
        )
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs,
            )

        # --- Sub-band BLSTM (time dimension) ---
        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True,
        )
        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs,
            )

        # --- Multi-head Self-Attention (time dimension global attention) ---
        E = qk_output_channel
        if n_head < 1 or emb_dim % n_head != 0:
            raise ValueError(
                f"emb_dim ({emb_dim}) must be divisible by n_head ({n_head})"
            )
        self.attn_conv_Q = nn.Conv2d(emb_dim, n_head * E, 1)
        self.attn_norm_Q = AllHeadPReLULayerNorm4DC((n_head, E), eps=eps)
        self.attn_conv_K = nn.Conv2d(emb_dim, n_head * E, 1)
        self.attn_norm_K = AllHeadPReLULayerNorm4DC((n_head, E), eps=eps)
        self.attn_conv_V = nn.Conv2d(emb_dim, n_head * (emb_dim // n_head), 1)
        self.attn_norm_V = AllHeadPReLULayerNorm4DC((n_head, emb_dim // n_head), eps=eps)
        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1),
            nn.PReLU(),
            LayerNormalization4D(emb_dim, dim=-3, total_dim=4, eps=eps),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def _rnn_process(self, x, norm, rnn, linear, B_outer, L_outer, L_inner, C):
        """Shared logic for intra/inter RNN with unfold/fold."""
        input_ = x
        x = norm(x)  # (B_outer, L_inner, C)

        if self.emb_ks == self.emb_hs:
            # Simple reshape: each emb_ks consecutive elements grouped
            x = x.view(B_outer, -1, self.emb_ks * C)  # (B_outer, L_inner//ks, ks*C)
            x, _ = rnn(x)
            x = linear(x)  # (B_outer, L_inner//ks, ks*C)
            x = x.view(-1, L_inner, C)
        else:
            # Unfold-based sub-band grouping
            x = x.view(B_outer, L_inner, C)
            x = x.transpose(1, 2)  # (B_outer, C, L_inner)
            x = F.unfold(x[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))  # (B_outer, C*ks, -1)
            x = x.transpose(1, 2)  # (B_outer, -1, C*ks)
            x, _ = rnn(x)
            x = x.transpose(1, 2)  # (B_outer, H*2, -1)
            x = linear(x)  # (B_outer, C, L_inner)
            x = x.view(-1, C, L_inner).transpose(1, 2)  # (B_outer, L_inner, C)

        return x + input_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, Q) — C=emb_dim, T=time frames, Q=freq bins.

        Returns:
            (B, C, T, Q)
        """
        B, C, old_T, old_Q = x.shape

        # Pad to align with emb_ks/emb_hs unfold
        olp = self.emb_ks - self.emb_hs
        T = (math.ceil((old_T + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks)
        Q = (math.ceil((old_Q + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks)

        x = x.permute(0, 2, 3, 1)  # (B, old_T, old_Q, C)
        x = F.pad(x, (0, 0, olp, Q - old_Q - olp, olp, T - old_T - olp))  # (B, T, Q, C)

        # --- Intra-frame BLSTM (across frequency Q, per frame) ---
        intra_rnn = self._rnn_process(
            x.reshape(B * T, Q, C), self.intra_norm, self.intra_rnn,
            self.intra_linear, B * T, T, Q, C,
        )
        intra_rnn = intra_rnn.view(B, T, Q, C)

        # --- Sub-band BLSTM (across time T, per freq bin) ---
        intra_rnn = intra_rnn.transpose(1, 2)  # (B, Q, T, C)
        inter_rnn = self._rnn_process(
            intra_rnn.reshape(B * Q, T, C), self.inter_norm, self.inter_rnn,
            self.inter_linear, B * Q, Q, T, C,
        )
        inter_rnn = inter_rnn.view(B, Q, T, C)
        inter_rnn = inter_rnn.permute(0, 3, 2, 1)  # (B, C, T, Q)

        # Remove padding
        inter_rnn = inter_rnn[..., olp:olp + old_T, olp:olp + old_Q]
        batch = inter_rnn

        # --- Multi-head Self-Attention (time dimension) ---
        Qa = self.attn_norm_Q(self.attn_conv_Q(batch))  # (B, H, E, T, Q)
        Ka = self.attn_norm_K(self.attn_conv_K(batch))
        Va = self.attn_norm_V(self.attn_conv_V(batch))
        Qa = Qa.view(-1, *Qa.shape[2:])  # (B*H, E, T, Q)
        Ka = Ka.view(-1, *Ka.shape[2:])
        Va = Va.view(-1, *Va.shape[2:])

        # Flatten (E, Q) and compute attention over T
        Qa = Qa.transpose(1, 2).flatten(start_dim=2)  # (B*H, T, E*Q)
        Ka = Ka.transpose(2, 3).contiguous().view(Qa.shape[0], -1, old_T)  # (B*H, E*Q, T)
        old_shape = Va.transpose(1, 2).shape  # (B*H, T, E_v, Q)
        Va = Va.transpose(1, 2).flatten(start_dim=2)  # (B*H, T, E_v*Q)

        emb_dim_attn = Qa.shape[-1]
        attn = torch.matmul(Qa, Ka) / (emb_dim_attn ** 0.5)  # (B*H, T, T)
        attn = F.softmax(attn, dim=-1)
        Va = torch.matmul(attn, Va)  # (B*H, T, E_v*Q)
        Va = Va.reshape(old_shape).transpose(1, 2)  # (B*H, E_v, T, Q)
        E_v = Va.shape[1]

        batch = Va.contiguous().view(B, self.n_head * E_v, old_T, old_Q)
        batch = self.attn_concat_proj(batch)  # (B, C, T, Q)

        return batch + inter_rnn


# =============================================================================
# TFGridNet — 完整模型 (论文 Figure 1)
# =============================================================================


@MODEL_REGISTRY.register("TFGridNet")
class TFGridNet(BaseSEModel):
    """TF-GridNet for speech enhancement (spectrum-in / spectrum-out).

    Input:  noisy_spec (B, F, T, 2) from framework STFT.
    Output: enhanced_spec (B, F, T, 2).

    Architecture (论文 Figure 1):
      1. Conv2d: (B, 2, T, F) -> (B, emb_dim, T, F)
      2. N × GridNetBlock (intra-BLSTM + inter-BLSTM + Self-Attention)
      3. ConvTranspose2d: (B, emb_dim, T, F) -> (B, 2, T, F) complex mapping

    For speech enhancement (single source), output is direct complex spectrum estimation
    (not masking, following the original paper's approach of predicting clean complex spec).

    Args:
        n_fft: FFT size (injected from YAML stft block). Default 512.
        n_layers: Number of GridNet blocks. Default 6.
        lstm_hidden_units: LSTM hidden size per direction. Default 192.
        attn_n_head: Attention heads. Default 4.
        attn_qk_output_channel: Attention Q/K channels per head. Default 4.
        emb_dim: Embedding channels. Default 48.
        emb_ks: Unfold kernel size. Default 4.
        emb_hs: Unfold hop size. Default 1.
        eps: LayerNorm epsilon. Default 1e-5.
    """

    def __init__(
        self,
        n_fft: int = 512,
        n_layers: int = 6,
        lstm_hidden_units: int = 192,
        attn_n_head: int = 4,
        attn_qk_output_channel: int = 4,
        emb_dim: int = 48,
        emb_ks: int = 4,
        emb_hs: int = 1,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        for key in ("hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"TFGridNet got unexpected keyword arguments: {sorted(kwargs)}")
        self.n_fft = n_fft
        self.n_layers = n_layers

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)

        # Encoder: 2-ch (real+imag) -> emb_dim channels
        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
            nn.PReLU(),
        )

        # GridNet blocks
        self.blocks = nn.ModuleList([
            GridNetBlock(
                emb_dim=emb_dim,
                emb_ks=emb_ks,
                emb_hs=emb_hs,
                hidden_channels=lstm_hidden_units,
                n_head=attn_n_head,
                qk_output_channel=attn_qk_output_channel,
                eps=eps,
            )
            for _ in range(n_layers)
        ])

        # Decoder: emb_dim -> 2-ch (real+imag), single source for SE
        self.deconv = nn.ConvTranspose2d(emb_dim, 2, ks, padding=padding)

    @fp32_forward
    def forward(self, noisy_spec: torch.Tensor, lengths=None, **kw) -> torch.Tensor:
        """noisy_spec: (B, F, T, 2). Returns enhanced_spec (B, F, T, 2)."""
        if noisy_spec.dim() != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(f"Expected [B, F, T, 2], got {tuple(noisy_spec.shape)}")
        if noisy_spec.shape[1] != self.n_fft // 2 + 1:
            raise ValueError(f"Expected F={self.n_fft // 2 + 1}, got F={noisy_spec.shape[1]}")
        if noisy_spec.shape[0] == 0 or noisy_spec.shape[2] == 0:
            raise ValueError("TFGridNet requires non-empty batch and time dimensions")
        if not noisy_spec.is_floating_point():
            raise TypeError("TFGridNet noisy_spec must use a floating-point dtype")
        noisy_spec = noisy_spec.float()

        # (B, F, T, 2) -> (B, 2, T, F)
        feature = noisy_spec.permute(0, 3, 2, 1)

        # Encoder
        batch = self.conv(feature)  # (B, emb_dim, T, F)

        # GridNet blocks
        for block in self.blocks:
            batch = block(batch)

        # Decoder
        batch = self.deconv(batch)  # (B, 2, T, F)

        # (B, 2, T, F) -> (B, F, T, 2)
        enhanced = batch.permute(0, 3, 2, 1)

        return enhanced


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 验证本模型参数量与算力。运行: python tasks/se/models/tf_gridnet.py [--sr 16000] [--dur 1.0] [--device cpu]
    # -------------------------------------------------------------------------
    from core.utils.model_profile import profile_model, make_profile_parser
    from core.stft import build_stft_istft

    args = make_profile_parser(default_sr=16000).parse_args()

    STFT_CFG = {
        "n_fft": 512,
        "hop_length": 256,
        "win_length": 512,
        "window": "hann",
    }
    SAMPLE_RATE = args.sr
    MODEL_ARGS = {
        "n_fft": STFT_CFG["n_fft"],
        "n_layers": 6,
        "lstm_hidden_units": 192,
        "emb_dim": 48,
        "emb_ks": 4,
        "emb_hs": 1,
        "attn_n_head": 4,
        "attn_qk_output_channel": 4,
    }

    model = TFGridNet(**MODEL_ARGS)
    stft_module = build_stft_istft(STFT_CFG)

    class _W(nn.Module):
        def __init__(self, m, stft_m):
            super().__init__()
            self._m, self._stft = m, stft_m
        def forward(self, wav):
            spec = self._stft(wav)
            return self._stft.inverse(self._m(spec), wav.shape[-1])

    wrapped = _W(model, stft_module)
    print("\n[Config] STFT:", STFT_CFG, "| sample_rate:", SAMPLE_RATE, "| model args:", MODEL_ARGS)
    profile_model(
        wrapped,
        sample_rate=SAMPLE_RATE,
        duration=args.dur,
        batch_size=args.batch,
        device=args.device,
        repeats=args.repeats,
        label="TFGridNet",
    )
