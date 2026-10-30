"""Top-level UniPASE inference model: DeWavLM-Omni -> VocosAdapter -> VocosVocoder -> PostNet.

Ported from ``unipase/models/unipase.py:15-69``，组合方式与数据流完全对齐官方：

    encoder (WavLM_feat_PLC, output_layer=[1,24])
        -> noisy_feat_a (L1 声学特征), enh_feat_p (L24 语义特征)
    adapter(noisy_feat_a, enh_feat_p) -> enh_feat_a
    vocoder(enh_feat_a.transpose(1, 2)) -> 16kHz 波形
    sr_out > 16000 时: postnet.infer() 做 8k->24k 宽带扩展
    重采样/pad-trim 到 n_samples * sr_out / sr_in
    峰值归一化到输入峰值（官方 inference/inference.py:84-85）

本模型为纯推理级联（子模块全部 frozen、@torch.no_grad），注册到
MODEL_REGISTRY 仅供 eval/inference 流水线使用。
"""

import logging

import torch
import torchaudio

from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.base_model import BaseSEModel
from tasks.se.models.pase_family.common.utils import (
    align_waveform_length,
    scale_lengths_to,
)

from .adapter import VocosAdapter
from .postnet import TFGridNet
from .vocoder import VocosVocoder
from .wavlm_frontend_plc import WavLM_feat_PLC

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register("UniPASE")
class UniPASE(BaseSEModel):
    """UniPASE generative speech enhancement model (inference-only).

    Args:
        dewavlm_ckpt_path: Path to DeWavLM-Omni.pt (带 PLC 的 WavLM encoder).
        adapter_ckpt_path: Path to Adapter.pt. 为空则不构造（forward 时报错）。
        vocoder_ckpt_path: Path to Vocoder_DWO-L1.pt. 为空则不构造。
        postnet_ckpt_path: Path to PostNet.pt。sr_out<=16000 时忽略且不加载；
            sr_out>16000 时必须显式提供。
        sr_out: 正整数输出采样率（默认 16000，不加载 PostNet；>16000 时走 PostNet 宽带扩展）。
        enable_plc: 是否启用丢包补偿（PLC）掩码。
    """

    def __init__(
        self,
        dewavlm_ckpt_path: str = "",
        adapter_ckpt_path: str = "",
        vocoder_ckpt_path: str = "",
        postnet_ckpt_path: str = "",
        sr_out: int = 16000,
        enable_plc: bool = True,
        **kwargs,
    ):
        for key in ("n_fft", "hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(
                f"UniPASE got unexpected keyword arguments: {sorted(kwargs)}"
            )
        if isinstance(sr_out, bool) or not isinstance(sr_out, int):
            raise TypeError(
                f"UniPASE sr_out must be a positive integer, got {sr_out!r}"
            )
        if sr_out <= 0:
            raise ValueError(f"UniPASE sr_out must be > 0, got {sr_out}")
        if sr_out > 16000 and not postnet_ckpt_path:
            raise ValueError(
                "UniPASE sr_out > 16000 时必须显式提供 postnet_ckpt_path。"
            )

        super().__init__()
        self.sr_out = sr_out
        self.enable_plc = enable_plc

        self.encoder = WavLM_feat_PLC(
            wavlm_ckpt_path=dewavlm_ckpt_path,
            output_layer=[1, 24],
            load_pretrained=bool(dewavlm_ckpt_path),
            frozen=True,
            enable_plc=enable_plc,
        )
        # encoder 必须 fail-closed，避免空路径退化为随机初始化的 WavLM-Large。
        self._encoder_pretrained = bool(dewavlm_ckpt_path)
        # 缺少 checkpoint 时不构造，forward 时 fail-closed。
        self.adapter = VocosAdapter.from_pretrained(adapter_ckpt_path) if adapter_ckpt_path else None
        self.decoder = VocosVocoder.from_pretrained(vocoder_ckpt_path) if vocoder_ckpt_path else None
        self.postnet = (
            TFGridNet.from_pretrained(postnet_ckpt_path) if sr_out > 16000 else None
        )

    @torch.no_grad()
    def forward(
        self,
        noisy_wav: torch.Tensor,
        lengths: torch.Tensor = None,
        *,
        fs=None,
        normalize_output: bool = True,
    ) -> torch.Tensor:
        """Args: noisy_wav (B, L) or (B, 1, L)，采样率 fs（None 视为 16000）。
        normalize_output: 是否将输出峰值归一化到输入峰值（官方 inference.py 行为）；
            长音频分块推理时应传 False，由外层在拼接后做一次全局归一化
            （对齐官方 inference_long.py 语义）。
        Returns: enhanced_wav (B, L')，L' = n_samples * sr_out / sr_in。"""
        if not self._encoder_pretrained:
            raise RuntimeError(
                "UniPASE.encoder 未加载预训练权重：需提供 dewavlm_ckpt_path 指向 "
                "DeWavLM-Omni artifact（否则 encoder 为随机初始化，输出无意义）。"
            )
        if self.adapter is None or self.decoder is None:
            raise RuntimeError(
                "UniPASE.adapter/decoder 未加载：需提供 adapter_ckpt_path 与 "
                "vocoder_ckpt_path 指向 UniPASE artifact。"
            )

        x = noisy_wav
        if x.ndim == 3:
            x = x.squeeze(1)

        n_samples = x.shape[-1]
        sr_in = int(fs) if fs is not None else 16000
        sr_out = self.sr_out

        noisy_feat_a, enh_feat_p = self.encoder(
            x, sr=sr_in, mask=self.enable_plc, lengths=lengths
        )
        enh_feat_a = self.adapter(noisy_feat_a, enh_feat_p)

        enh_wav_16k = self.decoder(enh_feat_a.transpose(1, 2))  # (B, T)

        if sr_out < 16000:
            enh_wav = torchaudio.functional.resample(enh_wav_16k, orig_freq=16000, new_freq=sr_out)
        elif sr_out == 16000:
            enh_wav = enh_wav_16k
        else:
            if self.postnet is None:
                raise RuntimeError(
                    "UniPASE.postnet 未加载：sr_out > 16000 需要提供 postnet_ckpt_path。"
                )
            lengths_16k = scale_lengths_to(lengths, x.shape[-1], enh_wav_16k.shape[-1])
            enh_wav = self.postnet.infer(
                enh_wav_16k, sr_in=16000, sr_out=sr_out, lengths=lengths_16k
            )

        if sr_out == sr_in:
            out_samples = n_samples
        else:
            out_samples = int(n_samples * sr_out / sr_in)
        y = align_waveform_length(enh_wav, out_samples)

        if normalize_output:
            # 官方 inference/inference.py:84-85：把输出峰值归一化到输入峰值
            in_peak = x.abs().amax(dim=-1, keepdim=True)
            y = y / (y.abs().amax(dim=-1, keepdim=True) + 1e-8) * in_peak

        return y
