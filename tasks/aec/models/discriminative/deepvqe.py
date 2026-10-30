"""DeepVQE：由论文结构派生的 AEC 模型，并非作者实现的严格复现。

Reference: DeepVQE: Real Time Deep Voice Quality Enhancement for Joint Acoustic Echo
Cancellation, Noise Suppression and Dereverberation (Microsoft, arXiv:2306.03177).
Unofficial single-input repo: https://github.com/Xiaobin-Rong/deepvqe

实现参考论文第 2 节的主要模块：
- Feature extraction: power-law compressed complex spectra (§2).
- Encoder: mic branch 5 blocks (base) / 4 blocks (small), far branch 2 blocks; kernel 4×3, stride 1×2; causal padding (§2 Encoder).
- Residual block: Y = X + ELU(BatchNorm(Conv2D(X))), kernel 4×3, stride 1 (Eq. 1).
- Alignment block: Q/K point-wise, unfold on time, conv 5×3 with causal padding, softmax, weighted sum (§2 Alignment block).
- Bottleneck: flatten to (T, c·f), GRU, linear projection, reshape (§2 Bottleneck).
- Decoder: skip block (1×1 conv), residual block, sub-pixel conv 4×3, BN, ELU (§2 Decoder, Skip block, Sub-pixel).
- CCM: v = (1, -1/2+j√3/2, -1/2-j√3/2), Eq. 2; then (m+1)×(2n+1) conv, Eq. 3, causal.

窗口、STFT/iSTFT、工程化 causal padding、小模型宽度和输入校验均以仓库实现为准；不能据此声称逐项严格复现论文或非官方仓库。AEC 模型不继承 BaseSEModel（双输入 mic + farend），训练使用 AECTrainer。

训练态与严格因果边界：
  残差块使用 ``BatchNorm2d``，训练时在时间维与 batch 维上统计；论文 §2 的 Encoder/Decoder
  路径在工程上常按 **整段** 频谱图处理，**不等价**于仅因果卷积的流式逐帧系统。推理时
  一般为 **eval + running stats**。若产品要求严格因果与 train/eval 一致统计，需替换
  BN 并单独验证；当前实现保留论文派生结构所用的 BN。

Usage:
    model:
      type: DeepVQE
      args:
        variant: "base"    # base / small
        n_fft: 512

Run directly to test params and compute:
  python -m tasks.aec.models.discriminative.deepvqe [--sr 16000] [--dur 1.0] [--device cpu]


Author: Xincheng Zhu
"""

from __future__ import annotations
import sys as _sys, os as _os
if __name__ == "__main__":
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))


import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.utils.registry import MODEL_REGISTRY


# ──────────────────────────────────────────────────────────────
# AEC building blocks
# ──────────────────────────────────────────────────────────────
class _FE(nn.Module):
    """Feature extraction: power-law compressed complex spectra (paper §2)."""

    def __init__(self, c: float = 0.3):
        super().__init__()
        self.c = c

    def forward(self, x: Tensor) -> Tensor:
        """(B, F, T, 2) → (B, 2, T, F) for encoder conv input."""
        mag = torch.sqrt(x[..., 0:1].pow(2) + x[..., 1:2].pow(2) + 1e-12)
        # Power-law compression: x_compressed = x / |x|^(1-c) = |x|^c * exp(j*phase)
        x = x / (mag.pow(1 - self.c) + 1e-12)
        # Layout: (B, F, T, 2) → (B, 2, T, F)  [channels-first for Conv2d]
        return x.permute(0, 3, 2, 1).contiguous()


class _ResBlock(nn.Module):
    """Residual block: Y = X + ELU(BatchNorm(Conv2D(X))), kernel 4×3, stride 1, causal padding (paper Eq. 1)."""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ZeroPad2d([1, 1, 3, 0]),
            nn.Conv2d(ch, ch, (4, 3)),
            nn.BatchNorm2d(ch),
            nn.ELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + x


class _AlignBlock(nn.Module):
    """Alignment block (paper §2): Q/K from point-wise conv, unfold K on time, dot product → Z,
    conv kernel 5×3, stride 1, softmax on delay axis, weighted sum for XF. Paper: padding 3×1;
    we use causal padding (left=4, right=0 on time; 1,1 on freq) so output shape matches input.

    NOTE (JAE-L01，已知 off-by-one，刻意不改代码)：delay=N 时 unfold pad N-1、
    候选窗为当前帧及前 N-1 帧（回看 0..N-1 帧），最大回看 (N-1)×帧移。
    例：delay=100、hop=10ms → 最大回看 990ms，**不含** 1000ms 端点。
    为保持与已发布 checkpoint 及 exporter/cache 的形状契约，语义保持现状；
    配置侧如需恰好覆盖端点 max_delay，须自行 +1 帧
    （见 configs/aec/microsoft_challenge/deepvqe/deepvqe_*.yaml 注释）。
    """

    def __init__(self, in_ch: int, far_ch: int, hid_ch: int, delay: int = 100):
        super().__init__()
        # NOTE: the weighted-sum step reuses K's shape to view unfolded x_ref,
        # so far_ch must equal hid_ch for the reshape to be valid.
        if far_ch != hid_ch:
            raise ValueError(
                f"_AlignBlock requires far_ch == hid_ch, got far_ch={far_ch} hid_ch={hid_ch}"
            )
        self.far_ch = far_ch
        self.pconv_mic = nn.Conv2d(in_ch, hid_ch, 1)
        self.pconv_ref = nn.Conv2d(far_ch, hid_ch, 1)
        self.unfold = nn.Sequential(nn.ZeroPad2d([0, 0, delay - 1, 0]), nn.Unfold((delay, 1)))
        self.conv = nn.Sequential(nn.ZeroPad2d([1, 1, 4, 0]), nn.Conv2d(hid_ch, 1, (5, 3)))

    def forward(self, x_mic: Tensor, x_ref: Tensor) -> Tensor:
        # The attention dot-product (sum over freq bins) can overflow in FP16
        # when delay × freq product is large.  Force FP32 only when needed.
        _need_fp32 = (x_mic.dtype != torch.float32)
        if _need_fp32:
            x_mic = x_mic.float()
            x_ref = x_ref.float()
        Q = self.pconv_mic(x_mic)   # (B, hid, T, F)
        K = self.pconv_ref(x_ref)   # (B, hid, T, F)

        # Unfold K along time to gather delay candidates: (B, hid, T, delay, F)
        Ku = self.unfold(K).view(K.shape[0], K.shape[1], -1, K.shape[2], K.shape[3]) \
            .permute(0, 1, 3, 2, 4).contiguous()

        # Dot-product similarity over delay axis → attention logits → (B, 1, T, F) via conv
        V = torch.sum(Q.unsqueeze(-2) * Ku, dim=-1)  # (B, hid, T, delay)
        V = self.conv(V)                              # (B, 1, T, delay)
        A = torch.softmax(V, dim=-1).unsqueeze(-1)    # (B, 1, T, delay, 1)

        # Weighted sum of delayed far-end frames using the attention weights
        y = self.unfold(x_ref).view(K.shape[0], K.shape[1], -1, K.shape[2], K.shape[3]) \
            .permute(0, 1, 3, 2, 4).contiguous()     # (B, far_ch, T, delay, F)
        return torch.sum(y * A, dim=-2)               # (B, far_ch, T, F)


class _EncBlock(nn.Module):
    """Encoding block (paper §2): downsampling conv 4×3 stride 1×2, BN, ELU, optional residual block; causal padding."""

    def __init__(self, in_ch: int, out_ch: int, use_res: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, (4, 3), stride=(1, 2))
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ELU()
        self.res = _ResBlock(out_ch) if use_res else None

    def forward(self, x: Tensor) -> Tensor:
        # Causal padding: (freq_left=1, freq_right=1, time_past=3, time_future=0)
        x = F.pad(x, (1, 1, 3, 0))
        x = self.act(self.bn(self.conv(x)))  # stride (1,2) halves freq dim
        if self.res:
            x = self.res(x)
        return x


class _SubPixelConv2d(nn.Module):
    """Sub-pixel convolution (paper §2): conv with 2c filters → reshape to Y ∈ R^{c×t×2f}; kernel 4×3, causal."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])
        self.conv = nn.Conv2d(in_ch, out_ch * 2, (4, 3))

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(self.pad(x))      # (B, out_ch*2, T, F)
        B, C, T, F = y.shape
        # Pixel-shuffle on freq axis: interleave the 2 sub-channels to double freq resolution
        # (B, out_ch*2, T, F) → (B, out_ch, 2, T, F) → (B, out_ch, T, F*2)
        return y.view(B, C // 2, 2, T, F).permute(0, 1, 3, 4, 2).contiguous().view(B, C // 2, T, F * 2)


class _DecBlock(nn.Module):
    """Decoding block (paper §2): skip block (1×1 conv, then sum), residual block, sub-pixel conv, BN, ELU (omit in last)."""

    def __init__(self, in_ch: int, out_ch: int, is_last: bool = False, use_res: bool = True,
                 skip_ch: int | None = None):
        super().__init__()
        skip_ch = skip_ch if skip_ch is not None else in_ch
        self.skip_conv = nn.Conv2d(skip_ch, in_ch, 1)
        self.deconv = _SubPixelConv2d(in_ch, out_ch)
        # BN+ELU are omitted on the final decoder block (is_last=True).
        # Do NOT create them at all to avoid unused-parameter errors in DDP.
        self.bn = nn.BatchNorm2d(out_ch) if not is_last else None
        self.act = nn.ELU() if not is_last else None
        self.is_last = is_last
        self.res = _ResBlock(in_ch) if use_res else None

    def forward(self, x: Tensor, x_enc: Tensor) -> Tensor:
        # Skip connection: align encoder feature channels and add to decoder input
        y = x + self.skip_conv(x_enc)
        if self.res:
            y = self.res(y)
        y = self.deconv(y)           # sub-pixel conv → doubles freq dim
        if not self.is_last:
            y = self.act(self.bn(y)) # BN+ELU omitted on the final decoder block
        return y


class _Bottleneck(nn.Module):
    """Bottleneck (paper §2): flatten X to (T, c·f), GRU, linear projection, reshape back to (c×t×f)."""

    def __init__(self, in_size: int, hid_size: int):
        super().__init__()
        self.gru = nn.GRU(in_size, hid_size, batch_first=True)
        self.fc = nn.Linear(hid_size, in_size)

    def forward(self, x: Tensor) -> Tensor:
        B, C, T, F = x.shape
        # Flatten spatial dims for GRU: (B, C, T, F) → (B, T, C*F)
        y = x.permute(0, 2, 1, 3).reshape(B, T, C * F)
        y = self.fc(self.gru(y)[0])  # GRU then project back to input size
        # Restore spatial layout: (B, T, C, F) → (B, C, T, F)
        return y.reshape(B, T, C, F).permute(0, 2, 1, 3)


class _CCM(nn.Module):
    """Complex convolving mask (paper §2): v = (1, -1/2+j√3/2, -1/2-j√3/2), Eq. 2; then (m+1)×(2n+1)=3×3 conv, Eq. 3, causal."""

    def __init__(self):
        super().__init__()
        v = torch.tensor([
            [1.0, -0.5, -0.5],
            [0.0, math.sqrt(3) / 2, -math.sqrt(3) / 2],
        ])
        self.register_buffer("v", v, persistent=False)
        self.unfold = nn.Sequential(nn.ZeroPad2d([1, 1, 2, 0]), nn.Unfold(kernel_size=(3, 3)))

    def forward(self, m: Tensor, x: Tensor) -> Tensor:
        """m: (B,27,T,F) decoder output mask; x: (B,F,T,2) mic spectrum → (B,F,T,2) enhanced."""
        # The complex 3×3 convolution can overflow/underflow in FP16 when
        # magnitude varies widely across mask coefficients.  Force FP32 only when needed.
        _need_fp32 = (m.dtype != torch.float32)
        if _need_fp32:
            m = m.float()
            x = x.float()
        # Step 1 (Eq. 2): decompose m into 3 rotated complex masks using v
        # m has 27 = 3 × 3 × 3 coefficients: 3 rotations × 3×3 spatial filter
        m = m.view(m.shape[0], 3, -1, m.shape[2], m.shape[3])  # (B, 3, 9, T, F)
        H_r = torch.sum(self.v[0][None, :, None, None, None] * m, dim=1)  # real part
        H_i = torch.sum(self.v[1][None, :, None, None, None] * m, dim=1)  # imag part
        # Reshape to 2D conv kernels: (B, 3, 3, T, F)
        M_r = H_r.view(H_r.shape[0], 3, 3, H_r.shape[2], H_r.shape[3])
        M_i = H_i.view(H_i.shape[0], 3, 3, H_i.shape[2], H_i.shape[3])

        # Step 2 (Eq. 3): apply the 3×3 complex filter to mic spectrum (causal in time)
        x_in = x.permute(0, 3, 2, 1).contiguous()  # (B, 2, T, F)
        xu = self.unfold(x_in)  # gather 3×3 freq-time neighbourhood
        F_dim = x_in.shape[-1]
        xu = xu.view(x_in.shape[0], 2, 3, 3, -1, F_dim)  # (B, 2, 3, 3, T, F)

        # Complex multiplication: (M_r + j*M_i) * (x_r + j*x_i)
        enh_r = torch.sum(M_r * xu[:, 0] - M_i * xu[:, 1], dim=(1, 2))  # real part
        enh_i = torch.sum(M_r * xu[:, 1] + M_i * xu[:, 0], dim=(1, 2))  # imag part
        # Stack and restore layout: (B, T, F, 2) → (B, F, T, 2)
        return torch.stack([enh_r, enh_i], dim=3).transpose(1, 2).contiguous()


# ──────────────────────────────────────────────────────────────
# DeepVQE model
# ──────────────────────────────────────────────────────────────
@MODEL_REGISTRY.register("DeepVQE")
class DeepVQE(nn.Module):
    """DeepVQE Acoustic Echo Cancellation model.

    Spectrum-in / spectrum-out: accepts only ``mic_spectra`` and
    ``farend_spectra`` (B, F, T, 2), as declared by
    :attr:`aec_input_capabilities`. STFT/iSTFT and window are done by the
    trainer (YAML ``stft`` block); the model does not consume target or
    auxiliary spectra.

    Note:
        The auxiliary frame-level state detector has been removed from DeepVQE.
        Use DBGNet if state classification supervision is required.

    Args:
        variant: "base" (8M params) or "small" (smaller).
        use_state_detector / state_nclass: DBGNet 专属的 state detector 字段，
            DeepVQE 不支持状态分类，显式传入即报错。
    """

    aec_input_capabilities = frozenset({"mic_spectra", "farend_spectra"})

    def __init__(
        self,
        variant: str = "base",
        align_delay: int = 100,
        compression: float = 0.3,
        **kwargs,
    ):
        # align_delay = d_max (paper §3: "dmax = 100, equivalent to 1 second").
        # With the default n_fft=320/hop_length=160 (20ms/10ms @ 16kHz), 100 frames = 1.0 s, matching the paper.
        super().__init__()
        self.n_fft = kwargs.pop("n_fft", 320)
        self.hop_length = kwargs.pop("hop_length", 160)
        kwargs.pop("win_length", None)
        if variant not in {"base", "small"}:
            raise ValueError(f"Unknown DeepVQE variant: {variant}")
        if isinstance(self.n_fft, bool) or not isinstance(self.n_fft, int) or self.n_fft < 2:
            raise ValueError(f"DeepVQE n_fft must be an integer >= 2, got {self.n_fft!r}")
        if (
            isinstance(self.hop_length, bool)
            or not isinstance(self.hop_length, int)
            or self.hop_length < 1
        ):
            raise ValueError(
                f"DeepVQE hop_length must be a positive integer, got {self.hop_length!r}"
            )
        if isinstance(align_delay, bool) or not isinstance(align_delay, int) or align_delay < 1:
            raise ValueError(
                f"DeepVQE align_delay must be a positive integer, got {align_delay!r}"
            )
        if isinstance(compression, bool) or not isinstance(compression, (int, float)):
            raise ValueError(
                f"DeepVQE compression must be numeric in (0, 1], got {compression!r}"
            )
        compression = float(compression)
        if not math.isfinite(compression) or not 0.0 < compression <= 1.0:
            raise ValueError(
                f"DeepVQE compression must be in (0, 1], got {compression!r}"
            )
        # STFT 由 trainer 执行，模型不消费 window/center。
        kwargs.pop("window", None)
        kwargs.pop("center", None)

        # DeepVQE 没有逐帧 state detector：这两个 DBGNet 专属字段一律显式拒绝，
        # 不得静默容忍 False/0（AEC entry 也不会为 non-DBGNet 注入它们）。
        if "use_state_detector" in kwargs:
            raise ValueError(
                "DeepVQE does not support use_state_detector. "
                "Use DBGNet if an auxiliary state detector is required."
            )
        if "state_nclass" in kwargs:
            raise ValueError(
                "DeepVQE does not support state_nclass. "
                "Use DBGNet if an auxiliary state detector is required."
            )

        # 未知配置键必须报错，避免 YAML 拼写错误静默回退。
        if kwargs:
            raise TypeError(
                f"DeepVQE got unexpected keyword arguments: {sorted(kwargs)}"
            )

        self.fe = _FE(c=compression)

        if variant == "base":
            self._build_base(align_delay)
        elif variant == "small":
            self._build_small(align_delay)
        else:
            raise ValueError(f"Unknown DeepVQE variant: {variant}")

        self.ccm = _CCM()
        self._variant = variant

    def _build_base(self, align_delay: int = 100):
        """Full DeepVQE (paper §2, ~7.5M params).

        Architecture:
          Mic branch:  enc1(2→64) → enc2(64→128) → [concat aligned far] →
                       enc3(256→128) → enc4(128→128) → enc5(128→128)
          Far branch:  enc_far1(2→32) → enc_far2(32→128) → align → merge at enc3
          Bottleneck:  GRU(128*F → 64*F) → Linear
          Decoder:     dec5→dec4→dec3→dec2→dec1(→27 CCM coeffs)
        """
        # ── Mic encoder ──────────────────────────────────────────────
        self.enc1 = _EncBlock(2, 64)      # (B, 2,  T, F) → (B, 64,  T, F/2)
        self.enc2 = _EncBlock(64, 128)    # (B, 64, T, F/2) → (B, 128, T, F/4)
        # enc3 receives concatenation of mic-enc2 (128ch) and aligned far-enc2 (128ch)
        self.enc3 = _EncBlock(256, 128)   # (B, 256, T, F/4) → (B, 128, T, F/8)
        self.enc4 = _EncBlock(128, 128)   # (B, 128, T, F/8) → (B, 128, T, F/16)
        self.enc5 = _EncBlock(128, 128)   # (B, 128, T, F/16) → (B, 128, T, F/32)

        # ── Far-end encoder + alignment ──────────────────────────────
        self.enc_far1 = _EncBlock(2, 32)          # (B, 2,  T, F) → (B, 32,  T, F/2)
        self.enc_far2 = _EncBlock(32, 128)         # (B, 32, T, F/2) → (B, 128, T, F/4)
        self.align = _AlignBlock(128, 128, 128, delay=align_delay)  # aligns far to mic

        # ── Bottleneck ───────────────────────────────────────────────
        # Freq dimension after 5× stride-2 downsampling: F_n = ceil(F_{n-1} / 2)
        # e.g. n_fft=320 → 161→81→41→21→11→6; n_fft=512 → 257→...→9
        n_fft_bins = self.n_fft // 2 + 1
        enc5_freq = n_fft_bins
        for _ in range(5):
            enc5_freq = (enc5_freq - 1) // 2 + 1
        self.bottle = _Bottleneck(128 * enc5_freq, 64 * enc5_freq)

        # ── Decoder (skip connections from encoder) ───────────────────
        self.dec5 = _DecBlock(128, 128)           # skip from enc5
        self.dec4 = _DecBlock(128, 128)           # skip from enc4
        self.dec3 = _DecBlock(128, 128)           # skip from enc3
        self.dec2 = _DecBlock(128, 64)            # skip from enc2
        self.dec1 = _DecBlock(64, 27, is_last=True)  # skip from enc1 → 27 CCM coeffs, no BN+ELU

    def _build_small(self, align_delay: int = 100):
        """DeepVQE-S (~0.59M params @ n_fft=480; ~0.40M @ n_fft=320), per paper Section 3.

        Architecture (paper §3):
          Mic branch:  4 blocks with 16, 40, 56, 24 filters (ResBlock in deeper layers)
          Far branch:  2 blocks with 8, 24 filters
          Alignment:   at enc2 level; merged 40+24=64 ch fed into enc3
          Bottleneck:  GRU hidden = input/2 (paper: "linear projection allows reduction of hidden units")
          Decoder:     4 blocks with 40, 32, 32, 27 filters (ResBlock in dec4/dec3/dec2)
        """
        # ── Mic encoder (ResBlock enabled in deeper layers for gradient flow) ─
        self.enc1 = _EncBlock(2, 16, use_res=False)   # (B, 2,  T, F)   → (B, 16, T, F/2)
        self.enc2 = _EncBlock(16, 40, use_res=False)  # (B, 16, T, F/2) → (B, 40, T, F/4)
        # enc3 input = mic_enc2 (40ch) + aligned far_enc2 (24ch) = 64ch
        self.enc3 = _EncBlock(64, 56, use_res=True)   # (B, 64, T, F/4) → (B, 56, T, F/8)
        self.enc4 = _EncBlock(56, 24, use_res=True)   # (B, 56, T, F/8) → (B, 24, T, F/16)

        # ── Far-end encoder + alignment ──────────────────────────────
        self.enc_far1 = _EncBlock(2, 8, use_res=False)   # (B, 2, T, F)   → (B, 8,  T, F/2)
        self.enc_far2 = _EncBlock(8, 24, use_res=False)  # (B, 8, T, F/2) → (B, 24, T, F/4)
        # AlignBlock: query from mic_enc2 (40ch), key/value from far_enc2 (24ch), hidden 24ch
        self.align = _AlignBlock(40, 24, 24, delay=align_delay)

        # ── Bottleneck ───────────────────────────────────────────────
        # Freq dimension after 4× stride-2 downsampling
        # e.g. n_fft=320 → 161→81→41→21→11; n_fft=512 → 257→...→17
        n_fft_bins = self.n_fft // 2 + 1
        enc4_freq = n_fft_bins
        for _ in range(4):
            enc4_freq = (enc4_freq - 1) // 2 + 1
        self.bottle = _Bottleneck(24 * enc4_freq, 12 * enc4_freq)

        # ── Decoder (skip_ch = corresponding encoder output channels) ─
        self.dec4 = _DecBlock(24, 40, use_res=True, skip_ch=24)            # skip from enc4
        self.dec3 = _DecBlock(40, 32, skip_ch=56)                          # skip from enc3
        self.dec2 = _DecBlock(32, 32, skip_ch=40)                          # skip from enc2
        self.dec1 = _DecBlock(32, 27, is_last=True, use_res=False, skip_ch=16)  # skip from enc1

        # enc5/dec5 not used in small variant (forward() checks for None)
        self.enc5 = None
        self.dec5 = None

    def _validate_spectra_inputs(self, mic_spec: Tensor, far_spec: Tensor) -> None:
        """在卷积前校验双路谱的完整接口合同。"""
        for name, value in (("mic_spec", mic_spec), ("far_spec", far_spec)):
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"DeepVQE {name} must be a torch.Tensor, got {type(value).__name__}"
                )
            if value.ndim != 4 or value.shape[-1] != 2:
                raise ValueError(
                    f"DeepVQE {name} must have shape [B, F, T, 2], got {tuple(value.shape)}"
                )
            if not value.is_floating_point():
                raise TypeError(
                    f"DeepVQE {name} must have floating dtype, got {value.dtype}"
                )
        if mic_spec.shape != far_spec.shape:
            raise ValueError(
                "DeepVQE mic_spec and far_spec shapes must match, got "
                f"{tuple(mic_spec.shape)} and {tuple(far_spec.shape)}"
            )
        if mic_spec.dtype != far_spec.dtype:
            raise TypeError(
                "DeepVQE mic_spec and far_spec dtypes must match, got "
                f"{mic_spec.dtype} and {far_spec.dtype}"
            )
        if mic_spec.device != far_spec.device:
            raise ValueError(
                "DeepVQE mic_spec and far_spec devices must match, got "
                f"{mic_spec.device} and {far_spec.device}"
            )
        expected_bins = self.n_fft // 2 + 1
        if mic_spec.shape[1] != expected_bins:
            raise ValueError(
                f"DeepVQE expected F=n_fft//2+1={expected_bins}, got {mic_spec.shape[1]}"
            )

    def forward_spectra(self, mic_spec: Tensor, far_spec: Tensor) -> Tensor:
        """Spectra-domain forward pass.

        Args:
            mic_spec: (B, F, T, 2) microphone spectrum (real/imag).
            far_spec:  (B, F, T, 2) far-end reference spectrum.
        Returns:
            nearend_est: (B, F, T, 2) estimated near-end (speech) spectrum.
        """
        self._validate_spectra_inputs(mic_spec, far_spec)
        # ── Feature extraction ────────────────────────────────────────
        e0_mic = self.fe(mic_spec)   # power-law compressed, layout → (B, 2, T, F)
        e0_far = self.fe(far_spec)

        # ── Mic encoder (lower two blocks before alignment) ───────────
        e1 = self.enc1(e0_mic)   # (B, 64,  T, F/2)
        e2 = self.enc2(e1)       # (B, 128, T, F/4)

        # ── Far encoder + alignment ───────────────────────────────────
        ef1 = self.enc_far1(e0_far)  # (B, 32,  T, F/2)
        ef2 = self.enc_far2(ef1)     # (B, 128, T, F/4)
        aligned = self.align(e2, ef2)         # delay-compensated far features
        merged = torch.cat([e2, aligned], dim=1)  # (B, 256, T, F/4)

        # ── Shared encoder path ───────────────────────────────────────
        e3 = self.enc3(merged)   # (B, 128, T, F/8)
        e4 = self.enc4(e3)       # (B, 128, T, F/16)

        # ── Bottleneck + decoder (base has enc5/dec5, small does not) ─
        if self.enc5 is not None:
            e5 = self.enc5(e4)              # (B, 128, T, F/32)
            bn = self.bottle(e5)
            d5 = self.dec5(bn, e5)[..., :e4.shape[-1]]
            d4 = self.dec4(d5, e4)[..., :e3.shape[-1]]
        else:
            bn = self.bottle(e4)
            d4 = self.dec4(bn, e4)[..., :e3.shape[-1]]

        # [..,:enc_output_freq] trims any off-by-one from sub-pixel upsampling
        d3 = self.dec3(d4, e3)[..., :e2.shape[-1]]
        d2 = self.dec2(d3, e2)[..., :e1.shape[-1]]
        d1 = self.dec1(d2, e1)[..., :e0_mic.shape[-1]]  # (B, 27, T, F)

        # ── CCM: produce enhanced near-end spectrum ───────────────────
        return self.ccm(d1, mic_spec)   # complex convolution mask

    def forward(self, batch: dict) -> dict:
        """Dict-based forward (compatible with AEC training loop).

        Expected keys: mic_spectra, farend_spectra (B, F, T, 2).
        Sets: nearend_estimate_spectra.
        """
        if not isinstance(batch, dict):
            raise TypeError(
                f"DeepVQE forward expects a dict batch, got {type(batch).__name__}"
            )
        missing = self.aec_input_capabilities.difference(batch)
        if missing:
            raise KeyError(f"DeepVQE batch is missing required keys: {sorted(missing)}")
        batch["nearend_estimate_spectra"] = self.forward_spectra(
            batch["mic_spectra"], batch["farend_spectra"]
        )
        return batch


if __name__ == "__main__":
    from core.utils.model_profile import profile_model, make_profile_parser

    args = make_profile_parser(default_sr=16000).parse_args()

    model = DeepVQE(variant="base")
    profile_model(model, sample_rate=args.sr, duration=args.dur,
                  batch_size=args.batch, device=args.device, repeats=args.repeats,
                  dual_input=True,
                  label="DeepVQE-base (AEC, mic+far dual-input)")
