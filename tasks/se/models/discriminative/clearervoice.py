"""ClearerVoice-Studio SE models — **framework STFT only** (spectrum in / spectrum out).

Trainer supplies ``noisy_spec`` as ``(B, F, T, 2)`` (``core.stft.StftIstftModule``, ``center=True``) and
performs iSTFT on the returned tensor of the same shape.

YAML must include a top-level ``stft`` block; ``build_model(..., stft_cfg=cfg['stft'])`` injects
``n_fft`` / ``hop_length`` / ``win_length`` / ``window`` into ``model.args``。三个 ClearerVoice
封装类的构造函数**只接受通用键**（与框架 STFT 一致）；旧键 ``win_len`` / ``win_inc`` /
``fft_len`` / ``win_type`` 已移除，会作为未知参数 fail-fast。

  - **FRCRN_SE_16K**: 典型 ``n_fft == win_length``（默认 640）、``hop_length``（320），
    ``window: hann``（映射上游 ``hanning``）。
  - **MossFormer2_SE_48K**: e.g. ``n_fft: 1920``, ``hop_length: 384``, ``win_length: 1920``, ``window: hamming``;
    ``data.sample_rate: 48000``.

**Checkpoint note:** FRCRN 仍受 **ConvSTFT vs 框架 torch.stft** 差异影响；MossFormer2 的 MaskNet 输入已与官方同为
**Kaldi fbank+Δ+ΔΔ**，但 fbank 的波形来自对 ``noisy_spec`` 的 **iSTFT**（与训练端 ``StftIstftModule`` 一致），
若与官方 ``center=False`` 的 STFT 混用权重，仍可能需微调。

Upstream: https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/speech_enhancement

--------------------------------------------------------------------
与官方逻辑对齐（审计摘要）
--------------------------------------------------------------------
- **FRCRN**：掩膜乘法与 ``DCCRN.apply_mask`` 前两步一致；输入由官方 **ConvSTFT** 改为框架
  ``torch.stft`` 的 ``(B,F,T,2)``，**窗/中心填充与卷积 STFT 不完全相同**，加载官方权重可能需微调。
- **MossFormer2**：先 **iSTFT**（与 ``core.stft`` 同参）还原波形，再 **Kaldi fbank+Δ+ΔΔ**（与官方 ``compute_fbank`` 一致）；
  Kaldi 帧数与框架 STFT 帧数可能略有差异，对 mask 沿时间维 **线性插值** 对齐到谱轴。
- **训练**：``SETrainer`` 为 wav→STFT→模型→iSTFT→**时域 loss**。若沿用 GTCRN-style 目标，需在
  ``CompositeLoss.components`` 中显式配置 ``GTCRNCompressedSpectralLoss`` 与 ``GTCRNLogRatioLoss``；
  压缩谱项的窗函数与 STFT 几何由顶层 ``stft`` 自动注入。
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tasks.se.models.amp_utils import fp32_forward
from tasks.se.models.base_model import BaseSEModel
from tasks.se.models.discriminative.clearervoice_dsp import (
    interpolate_mask_time,
    kaldi_fbank_plus_deltas,
)
from core.stft import _get_window
from core.utils.registry import MODEL_REGISTRY


def _frcrn_args(
    win_len: int,
    win_inc: int,
    fft_len: int,
    win_type: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        win_len=win_len,
        win_inc=win_inc,
        fft_len=fft_len,
        win_type=win_type,
    )


def _map_framework_stft_kwargs(
    win_len: int,
    win_inc: int,
    fft_len: int,
    win_type: str,
    *,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str | None = None,
) -> tuple[int, int, int, str]:
    if hop_length is not None:
        win_inc = hop_length
    if win_length is not None:
        win_len = win_length
    if n_fft is not None:
        fft_len = n_fft
    if window is not None:
        w = str(window).lower()
        if w == "hann":
            win_type = "hanning"
        else:
            win_type = str(window)
    return win_len, win_inc, fft_len, win_type


def _resolve_stft_kwargs(
    *,
    n_fft: int,
    hop_length: int,
    win_length: int | None,
    window: str | None,
    default_window: str,
) -> tuple[int, int, int, str]:
    """由框架通用键 ``n_fft`` / ``hop_length`` / ``win_length`` / ``window`` 解析内部 STFT 几何。

    旧键（``win_len`` / ``win_inc`` / ``fft_len`` / ``win_type``）不再兼容：它们会留在
    ``**kwargs`` 中，由构造函数按未知参数 fail-fast。
    """
    win_len = win_length if win_length is not None else n_fft
    return _map_framework_stft_kwargs(
        int(win_len),
        int(hop_length),
        int(n_fft),
        str(window if window is not None else default_window),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )


@MODEL_REGISTRY.register("FRCRN_SE_16K")
class FRCRN_SE_16K(BaseSEModel):
    """FRCRN 16 kHz — U-Net 复数掩膜；仅处理 ``(B, F, T, 2)``（无内部 STFT）。"""

    def __init__(
        self,
        n_fft: int = 640,
        hop_length: int = 320,
        win_length: int | None = None,
        window: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        win_len, win_inc, fft_len, win_type = _resolve_stft_kwargs(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            default_window="hanning",
        )
        if kwargs:
            raise TypeError(f"FRCRN_SE_16K got unexpected keyword arguments: {sorted(kwargs)}")
        from tasks.se.models.modules.clearervoice_impl.frcrn.frcrn import FRCRN_SE_16K as _FRCRNCore

        self.n_fft = fft_len
        self._net = _FRCRNCore(_frcrn_args(win_len, win_inc, fft_len, win_type))

    @fp32_forward
    def forward(self, noisy_spec: torch.Tensor, lengths: torch.Tensor | None = None, **kw: object) -> torch.Tensor:
        if kw:
            raise TypeError(f"FRCRN_SE_16K.forward got unexpected keyword arguments: {sorted(kw)}")
        if noisy_spec.dim() != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(f"Expected [B, F, T, 2], got {tuple(noisy_spec.shape)}")
        if noisy_spec.shape[1] != self.n_fft // 2 + 1:
            raise ValueError(f"Expected F={self.n_fft // 2 + 1}, got F={noisy_spec.shape[1]}")
        if not noisy_spec.is_floating_point():
            raise TypeError("FRCRN_SE_16K noisy_spec must use a floating-point dtype")
        return self._net.model.forward_enhance_spec(noisy_spec.float())


@MODEL_REGISTRY.register("MossFormer2_SE_48K")
class MossFormer2_SE_48K(BaseSEModel):
    """MossFormer2 PSM：iSTFT 后提取 **Kaldi fbank+Δ+ΔΔ**；dither 仅训练时启用。"""

    def __init__(
        self,
        sampling_rate: int = 48000,
        n_fft: int = 1920,
        hop_length: int = 384,
        win_length: int | None = None,
        window: str | None = None,
        num_mels: int = 60,
        num_blocks: int = 24,
        kaldi_dither: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__()
        win_len, win_inc, fft_len, win_type = _resolve_stft_kwargs(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            default_window="hamming",
        )
        if kwargs:
            raise TypeError(f"MossFormer2_SE_48K got unexpected keyword arguments: {sorted(kwargs)}")
        from tasks.se.models.modules.clearervoice_impl.mossformer2.mossformer2 import MossFormer_MaskNet

        self.sampling_rate = int(sampling_rate)
        self._n_fft = int(fft_len)
        self._hop_length = int(win_inc)
        self._win_length = int(win_len)
        self._win_type_kaldi = win_type
        self._kaldi_dither = float(kaldi_dither)
        self._feat_dim = fft_len // 2 + 1
        self._num_mels = int(num_mels)
        in_ch = num_mels * 3
        self.masknet = MossFormer_MaskNet(
            in_channels=in_ch,
            out_channels=512,
            out_channels_final=self._feat_dim,
            num_blocks=num_blocks,
            num_spks=2,
        )
        _wt = win_type.lower()
        _wname = "hann" if _wt in ("hanning", "hann") else "hamming"
        w = _get_window(_wname, self._win_length, torch.device("cpu"))
        self.register_buffer("_win_buf", w, persistent=False)

    @fp32_forward
    def forward(self, noisy_spec: torch.Tensor, lengths: torch.Tensor | None = None, **kw: object) -> torch.Tensor:
        if kw:
            raise TypeError(f"MossFormer2_SE_48K.forward got unexpected keyword arguments: {sorted(kw)}")
        if noisy_spec.dim() != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(f"Expected [B, F, T, 2], got {tuple(noisy_spec.shape)}")
        if not noisy_spec.is_floating_point():
            raise TypeError("MossFormer2_SE_48K noisy_spec must use a floating-point dtype")
        ns = noisy_spec.float()
        b, f, tstft, _ = ns.shape
        if f != self._feat_dim:
            raise ValueError(
                f"noisy_spec freq F={f} != n_fft//2+1={self._feat_dim} (check stft.n_fft vs model)"
            )
        win = self._win_buf.to(ns.device)
        fb_rows: list[torch.Tensor] = []
        max_rows = 0
        for bi in range(b):
            if lengths is not None:
                out_len = int(lengths[bi].item())
            else:
                out_len = max((tstft - 1) * self._hop_length + self._n_fft, self._n_fft)
            spec_c = torch.view_as_complex(ns[bi].contiguous())
            wav = torch.istft(
                spec_c,
                n_fft=self._n_fft,
                hop_length=self._hop_length,
                win_length=self._win_length,
                window=win,
                center=True,
                length=out_len,
                onesided=True,
            )
            fb = kaldi_fbank_plus_deltas(
                wav,
                sampling_rate=self.sampling_rate,
                win_len_samples=self._win_length,
                hop_samples=self._hop_length,
                num_mels=self._num_mels,
                win_type=self._win_type_kaldi,
                dither=self._kaldi_dither if self.training else 0.0,
            )
            fb_rows.append(fb)
            max_rows = max(max_rows, fb.shape[0])
        feat = ns.new_zeros(b, max_rows, self._num_mels * 3)
        for bi, fb in enumerate(fb_rows):
            feat[bi, : fb.shape[0]] = fb
        x_in = feat.transpose(1, 2)
        mask_bt = self.masknet(x_in)
        m = mask_bt.transpose(1, 2)
        if m.size(-1) != tstft:
            m = interpolate_mask_time(m, tstft)
        return ns * m.unsqueeze(-1)
