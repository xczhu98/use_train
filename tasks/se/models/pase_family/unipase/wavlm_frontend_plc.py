"""WavLM_feat_PLC: UniPASE 的 WavLM 前端，带丢包检测（PLC）掩码路径。

Ported from ``unipase/models/wavlm/feature_extractor_plc.py``。

复用本框架 ``wavlm_frontend.WavLM_feat``（重采样到 16k、pad_to_multiple、
逐层 layer_norm eps=1e-6、{cfg, model} checkpoint 加载），仅新增 PLC 路径：
  - ``detect_packet_loss_indices()``（官方 :98-139，20ms 包、zero_ratio>=0.99、
    阈值 1e-7）；
  - ``enable_plc=True`` 且 ``mask=True`` 时，先在 16k pad 后波形上检测丢包区域，
    把 mask_indices 传给 ``extract_features``，由 ``apply_mask`` 用 mask_emb
    替换丢包区域的 CNN 特征（官方 :74-78）。

逐层 LayerNorm 沿每个样本的有效帧 ``feat_i.shape`` 归一化；该行为与官方
UniPASE 的 ``feat.shape[1:]`` utterance 归一化在无右侧 padding 时等价，并且避免
同批样本、排列或右侧 padding 影响表示。
"""

import logging
from typing import List, Union

import torch
import torchaudio

from ..common.wavlm_frontend import WavLM_feat

logger = logging.getLogger(__name__)


class WavLM_feat_PLC(WavLM_feat):
    """WavLM feature extractor with packet-loss-concealment (PLC) masking.

    Args:
        wavlm_ckpt_path: Path to the (De)WavLM checkpoint (e.g., DeWavLM-Omni.pt).
        output_layer: Which transformer layer(s) to return (e.g., [1, 24]).
        load_pretrained: Whether to load weights from ``wavlm_ckpt_path``.
        frozen: If True, set requires_grad=False for all WavLM parameters.
        enable_plc: If True, ``forward(..., mask=True)`` 会先检测丢包区域并把
            mask_indices 传入 extract_features；否则退化为父类行为。
    """

    def __init__(
        self,
        wavlm_ckpt_path: str = "",
        output_layer: Union[int, List[int]] = 24,
        load_pretrained: bool = True,
        frozen: bool = True,
        enable_plc: bool = False,
        feature_norm: str = "per_sample",
    ):
        super().__init__(
            wavlm_ckpt_path=wavlm_ckpt_path,
            output_layer=output_layer,
            load_pretrained=load_pretrained,
            frozen=frozen,
            feature_norm=feature_norm,
        )
        self.enable_plc = enable_plc

    def forward(
        self,
        wav: torch.Tensor,
        sr: int = None,
        mask: bool = False,
        mask_indices=None,
        lengths: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Extract WavLM features, optionally with PLC masking.

        Args / Returns: 与父类 ``WavLM_feat.forward`` 一致。
        """
        if mask and self.enable_plc and mask_indices is None:
            # 丢包检测必须在 16k、pad 后的波形上执行。
            if wav.ndim == 3:
                wav = wav.squeeze(1)
            if sr is not None and sr != 16000:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
                if lengths is not None:
                    # lengths 常为 collate 产出的 int32；48 kHz × 4 s = 192000，
                    # 192000 * 16000 = 3.07e9 会溢出 int32，必须先升到 int64。
                    lengths = torch.div(
                        torch.as_tensor(
                            lengths, device=wav.device, dtype=torch.long
                        )
                        * 16000
                        + sr
                        - 1,
                        sr,
                        rounding_mode="floor",
                    )
            wav = self.pad_to_multiple(wav)
            mask_indices = detect_packet_loss_indices(wav, fs=16000)
            if mask_indices is None:
                # 短音频无丢包索引时关闭 mask，避免进入随机掩码分支。
                return super().forward(wav, sr=16000, mask=False, lengths=lengths)
            return super().forward(
                wav, sr=16000, mask=True, mask_indices=mask_indices, lengths=lengths
            )
        return super().forward(wav, sr=sr, mask=mask, mask_indices=mask_indices, lengths=lengths)


def detect_packet_loss_indices(
    audio_batch: torch.Tensor,
    fs: int,
    packet_duration_ms: int = 20,
    threshold: float = 1e-7,
    min_zero_ratio: float = 0.99,
):
    """Detect packet loss regions (官方 feature_extractor_plc.py:98-139 原样移植).

    Args:
        audio_batch (tensor): shape (B, L)
        fs (int): sampling rate
        packet_duration_ms (int): packet duration in milliseconds
        threshold (float): threshold to consider a value as zero
        min_zero_ratio (float): minimum ratio of near-zero samples to consider a packet lost

    Returns:
        Tensor: (B, num_packets) bool mask，True 表示该包丢失；音频短于一个包时返回 None。
        20ms @16k = 320 samples，恰好等于 WavLM CNN 的下采样率，因此该 mask 可直接
        作为 apply_mask 的帧级 mask_indices 使用。
    """
    B, L = audio_batch.shape
    packet_len = int(fs * packet_duration_ms / 1000)
    num_packets = L // packet_len
    if num_packets == 0:
        return None
    # Truncate extra samples and reshape to (B, num_packets, packet_len)
    audio_batch = audio_batch[:, : num_packets * packet_len]
    packets = audio_batch.view(B, num_packets, packet_len)
    # Compute the ratio of samples below the threshold for each packet (B, num_packets)
    zero_ratio = (packets.abs() < threshold).float().mean(dim=2)
    # Obtain mask indicating lost packets
    loss_mask = zero_ratio >= min_zero_ratio  # (B, num_packets)

    return loss_mask
