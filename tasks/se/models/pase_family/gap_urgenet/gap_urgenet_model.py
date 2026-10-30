"""Top-level GAP-URGENet inference model: 生成支路 + 预测支路 + PostNet 融合。

Ported from ``gap-urgenet/models/gap_urgenet.py:16-94``，组合方式与数据流完全
对齐官方（官方 :49-77）：

    生成支路 (16k): encoder (WavLM_feat_PLC, output_layer=[1,24])
        -> noisy_feat_a (L1 声学特征), enh_feat_p (L24 语义特征)
        -> adapter(noisy_feat_a, enh_feat_p) -> enh_feat_a
        -> vocoder(enh_feat_a.transpose(1, 2)) -> 16kHz 波形 -> 上采样 48k
    预测支路 (16k): 输入重采样到 16k -> predictor -> 16kHz 波形 -> 上采样 48k
    融合     (48k): 长度对齐后 stack(dim=1) 成 (B, 2, L) -> postnet(sr_in=48000)
    输出: 重采样回 sr_out -> pad-trim 到 n_samples * sr_out / sr_in
        -> 峰值归一化到输入峰值（官方 inference/inference.py:84-85）

本模型为纯推理级联（子模块全部 frozen、@torch.no_grad），注册到
MODEL_REGISTRY 仅供 eval/inference 流水线使用。
"""

import math

import torch
from torchaudio.functional import resample

from core.utils.registry import MODEL_REGISTRY
from tasks.se.models.base_model import BaseSEModel

from ..common.utils import align_waveform_length, scale_lengths_to
from ..unipase.adapter import VocosAdapter
from ..unipase.vocoder import VocosVocoder
from ..unipase.wavlm_frontend_plc import WavLM_feat_PLC
from .postnet import TFGridNet as PostNet
from .predictor import TFGridNet as Predictor


@MODEL_REGISTRY.register("GAP_URGENet")
class GAP_URGENet(BaseSEModel):
    """GAP-URGENet generative-and-predictive speech enhancement model (inference-only).

    Args:
        dewavlm_ckpt_path: Path to DeWavLM-Omni.pt (带 PLC 的 WavLM encoder).
        adapter_ckpt_path: Path to Adapter.pt. 为空则不构造（forward 时报错）。
        vocoder_ckpt_path: Path to Vocoder.pt. 为空则不构造。
        predictor_ckpt_path: Path to Predictor.pt. 为空则不构造。
        postnet_ckpt_path: Path to PostNet.pt. 为空则不构造。
        sr_out: 输出采样率（默认 None，等于输入采样率）。
        enable_plc: 是否启用丢包补偿（PLC）掩码。
        max_forward_seconds: 单次前向允许的最长音频秒数（默认 60s），超过即报错，
            长音频需走 inference.py --chunk_seconds 分块推理。
    """

    def __init__(
        self,
        dewavlm_ckpt_path: str = "",
        adapter_ckpt_path: str = "",
        vocoder_ckpt_path: str = "",
        predictor_ckpt_path: str = "",
        postnet_ckpt_path: str = "",
        sr_out: int = None,
        enable_plc: bool = True,
        max_forward_seconds: float = 60.0,
        **kwargs,
    ):
        super().__init__()
        for key in ("n_fft", "hop_length", "win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"GAP_URGENet got unexpected keyword arguments: {sorted(kwargs)}")
        if isinstance(max_forward_seconds, bool) or not isinstance(max_forward_seconds, (int, float)):
            raise TypeError(
                f"max_forward_seconds must be a positive number, got {max_forward_seconds!r}"
            )
        if not math.isfinite(float(max_forward_seconds)) or float(max_forward_seconds) <= 0:
            raise ValueError(
                f"max_forward_seconds must be finite and > 0, got {max_forward_seconds!r}"
            )
        self.sr_out = sr_out
        self.enable_plc = enable_plc
        self.max_forward_seconds = float(max_forward_seconds)

        self.encoder = WavLM_feat_PLC(
            wavlm_ckpt_path=dewavlm_ckpt_path,
            output_layer=[1, 24],
            load_pretrained=bool(dewavlm_ckpt_path),
            frozen=True,
            enable_plc=enable_plc,
        )
        # encoder 必须 fail-closed，避免空路径退化为随机初始化的 WavLM-Large。
        self._encoder_pretrained = bool(dewavlm_ckpt_path)
        # 与 StuPASE/UniPASE 一致：缺 ckpt 时不构造，forward 时 fail closed
        self.adapter = VocosAdapter.from_pretrained(adapter_ckpt_path) if adapter_ckpt_path else None
        self.decoder = VocosVocoder.from_pretrained(vocoder_ckpt_path) if vocoder_ckpt_path else None
        self.predictor = Predictor.from_pretrained(predictor_ckpt_path) if predictor_ckpt_path else None
        self.postnet = PostNet.from_pretrained(postnet_ckpt_path) if postnet_ckpt_path else None

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
            长音频分块推理时应传 False，由外层在拼接后做一次全局归一化。
        Returns: enhanced_wav (B, L')，L' = n_samples * sr_out / sr_in。

        长音频防护：PostNet/Predictor 的 GridNetV3Block 自注意力为 T×T 全连接
        （48k 下每分钟约 225MB/层显存），单条输入的安全上限约 60s（A800-80G
        实测 180s 整条约需 35GB+，多模型同进程易 OOM）。输入时长超过
        ``max_forward_seconds``（默认 60s）时直接报错，不再只告警；长音频请走
        框架 inference.py 的分块语义（--chunk_seconds，中点拼接，对齐官方
        inference_long.py）。"""
        if not self._encoder_pretrained:
            raise RuntimeError(
                "GAP_URGENet.encoder 未加载预训练权重：需提供 dewavlm_ckpt_path 指向 "
                "DeWavLM-Omni artifact（否则 encoder 为随机初始化，输出无意义）。"
            )
        if self.adapter is None or self.decoder is None:
            raise RuntimeError(
                "GAP_URGENet.adapter/decoder 未加载：需提供 adapter_ckpt_path 与 "
                "vocoder_ckpt_path 指向 GAP-URGENet artifact。"
            )
        if self.predictor is None or self.postnet is None:
            raise RuntimeError(
                "GAP_URGENet.predictor/postnet 未加载：需提供 predictor_ckpt_path 与 "
                "postnet_ckpt_path 指向 GAP-URGENet artifact。"
            )

        x = noisy_wav
        if x.ndim == 3:
            x = x.squeeze(1)  # (B, L)

        n_samples = x.shape[-1]
        sr_in = int(fs) if fs is not None else 16000
        sr_out = self.sr_out or sr_in

        if n_samples > self.max_forward_seconds * sr_in:
            # GridNetV3Block 自注意力 T×T，显存随音频长度平方增长（见 docstring
            # 「长音频防护」），单条长音频会直接 OOM 掉共享推理进程 -> 显式拒绝
            raise RuntimeError(
                f"GAP_URGENet 单次前向输入 {n_samples / sr_in:.1f}s 超过 "
                f"max_forward_seconds={self.max_forward_seconds}s 上限；"
                "长音频请用 inference.py --chunk_seconds 分块推理，或显式提高 "
                "model.args.max_forward_seconds。"
            )

        # -------------------
        # Generative branch（官方 :52-55）
        # -------------------
        noisy_feat_a, enh_feat_p = self.encoder(
            x, sr=sr_in, mask=self.enable_plc, lengths=lengths
        )
        enh_feat_a = self.adapter(noisy_feat_a, enh_feat_p)
        enh_gen_16k = self.decoder(enh_feat_a.transpose(1, 2))  # (B, T)
        enh_gen_48k = resample(enh_gen_16k, orig_freq=16000, new_freq=48000)

        # -------------------
        # Predictive branch（官方 :60-63）
        # -------------------
        x_16k = x if sr_in == 16000 else resample(x, orig_freq=sr_in, new_freq=16000)
        # 有效长度换算到各支路时间基，避免 batch 右侧补零参与归一化统计
        lengths_16k = scale_lengths_to(lengths, x.shape[-1], x_16k.shape[-1])
        enh_pred_16k = self.predictor(x_16k, sr=16000, lengths=lengths_16k)
        enh_pred_48k = resample(enh_pred_16k, orig_freq=16000, new_freq=48000)

        # -------------------
        # PostNet fusion（官方 :68-74）
        # -------------------
        # 官方把 gen 支路 pad/trim 到 pred 支路长度后 stack 成双通道
        enh_gen_48k = align_waveform_length(enh_gen_48k, enh_pred_48k.shape[-1])
        enh_2ch = torch.stack([enh_pred_48k, enh_gen_48k], dim=1)  # (B, 2, L)
        lengths_48k = scale_lengths_to(lengths, x.shape[-1], enh_2ch.shape[-1])
        enh_fuse = self.postnet(enh_2ch, sr_in=48000, lengths=lengths_48k)

        # -------------------
        # Resample（官方 :79-92）
        # -------------------
        y = enh_fuse if sr_out == 48000 else resample(enh_fuse, orig_freq=48000, new_freq=sr_out)

        if sr_out == sr_in:
            out_samples = n_samples
        else:
            out_samples = int(n_samples * sr_out / sr_in)
        y = align_waveform_length(y, out_samples)

        if normalize_output:
            # 官方 inference/inference.py:84-85：把输出峰值归一化到输入峰值
            in_peak = x.abs().amax(dim=-1, keepdim=True)
            y = y / (y.abs().amax(dim=-1, keepdim=True) + 1e-8) * in_peak

        return y
