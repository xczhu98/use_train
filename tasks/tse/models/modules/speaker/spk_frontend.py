"""Speaker frontend：5 种可插拔 speaker cue。

配置采用严格合并，未知键直接报错。

句级 cue（``spkemb``）按 enrollment 真实长度分桶编码；train 模式下长度互不相同的
batch 会出现单行组，句级 head 的 ``nn.BatchNorm1d`` 此时改用 running statistics
（见 :func:`tasks.tse.models.speaker.encoder.sentence_batch_statistics_off`），保持
逐行不变且不触发 ``Expected more than 1 value per channel when training``。

Author: Xincheng Zhu
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.tse.models.modules.common.norm import select_norm
from tasks.tse.models.modules.fusion.speech import SpeakerFuseLayer, CrossFuse
from tasks.tse.models.modules.speaker.usef import USEF_attentionblock
from tasks.tse.models.speaker.encoder import (
    SpeakerEncoder,
    sentence_batch_norm_requires_two_rows,
    sentence_batch_statistics_off,
)

logger = logging.getLogger(__name__)

_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


@dataclass(frozen=True)
class SpeakerConditionOutput:
    """Modular speaker 条件；帧级 context 按行保留真实长度，不携带参数模块。

    ``spectral_cue`` 为可选 ``[R,1,F,T]``，``frame_context`` 的每项为
    ``[1,D,Te]``，``sentence_embedding`` 为可选 ``[R,D]``。
    ``present`` 门控最终融合；缺失行仍计算有限占位以保持参数计算图。
    """

    present: torch.Tensor | None
    spectral_cue: torch.Tensor | None = None
    frame_context: tuple[torch.Tensor, ...] | None = None
    sentence_embedding: torch.Tensor | None = None
    usef: tuple[torch.Tensor, torch.Tensor] | None = None


def _length_bucketed_waveforms(
    encoder: SpeakerEncoder,
    enroll_wav: torch.Tensor,
    enroll_lengths: torch.Tensor,
):
    """按真实 enrollment 长度分组，避免补零进入 speaker encoder。"""
    if not isinstance(enroll_wav, torch.Tensor) or enroll_wav.ndim != 2:
        raise ValueError("enroll_wav must be a [B, T] tensor")
    if not isinstance(enroll_lengths, torch.Tensor):
        raise TypeError("enroll_lengths must be a torch.Tensor")
    if enroll_lengths.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"enroll_lengths must have integer dtype, got {enroll_lengths.dtype}")
    if enroll_lengths.ndim != 1 or enroll_lengths.shape[0] != enroll_wav.shape[0]:
        raise ValueError(
            f"enroll_lengths must have shape [{enroll_wav.shape[0]}], "
            f"got {tuple(enroll_lengths.shape)}"
        )
    if enroll_lengths.device != enroll_wav.device:
        raise ValueError("enroll_lengths must use the same device as enroll_wav")
    if bool(
        ((enroll_lengths < 1) | (enroll_lengths > enroll_wav.shape[-1])).any().item()
    ):
        raise ValueError(
            f"enroll_lengths must be in [1, {enroll_wav.shape[-1]}], "
            f"got {enroll_lengths.tolist()}"
        )

    window_size = getattr(getattr(encoder, "fbank", None), "window_size", None)
    if window_size is not None and (
        isinstance(window_size, bool)
        or not isinstance(window_size, int)
        or window_size <= 0
    ):
        raise ValueError(
            "SpeakerEncoder fbank.window_size must be a positive integer, "
            f"got {window_size!r}"
        )

    for valid_length in torch.unique(enroll_lengths, sorted=True).tolist():
        row_indices = torch.nonzero(
            enroll_lengths == valid_length, as_tuple=False
        ).flatten()
        cropped = enroll_wav.index_select(0, row_indices)[..., :valid_length]
        encoded_length = max(valid_length, window_size or 0)
        if encoded_length > valid_length:
            cropped = F.pad(cropped, (0, encoded_length - valid_length))
        yield row_indices, cropped


def _restore_bucketed_rows(
    current: torch.Tensor | None,
    row_indices: torch.Tensor,
    values: torch.Tensor,
    total_rows: int,
) -> torch.Tensor:
    """把分桶结果按原始 batch 行号恢复，同时保留 autograd 图。"""
    if values.ndim < 2 or values.shape[0] != row_indices.numel():
        raise ValueError(
            "speaker feature must return one row per enrollment row, "
            f"got {tuple(values.shape)} for {row_indices.numel()} rows"
        )
    if current is None:
        current = values.new_zeros((total_rows, *values.shape[1:]))
    elif current.shape[1:] != values.shape[1:]:
        raise ValueError("speaker feature shape changed across length buckets")
    return current.index_copy(0, row_indices, values)


def _encode_sentence_bucket(
    encoder: SpeakerEncoder, cropped: torch.Tensor
) -> torch.Tensor:
    """句级编码单个长度分组；单行组在 train 模式下改用 running statistics。

    句级 head 的 ``nn.BatchNorm1d`` 作用在池化后的 2-D ``(B, dim)`` 上，train 模式
    要求每个通道至少 2 个样本。长度分组按真实 enrollment 长度切分，batch 内长度互不
    相同时必然出现单行组；此时临时使用 BatchNorm running statistics，使每行输出仍只
    依赖自身（逐行不变），而不是把整批补零到最大长度或裁剪到批内最小长度。多行同长度
    分组仍使用 batch statistics 并更新 running stats；eval 模式、冻结 encoder 与非
    句级 BN 模型（含 ``TSE_Modular`` 与自定义 encoder）零行为变化。
    """
    if (
        cropped.shape[0] < 2
        and encoder.training
        and sentence_batch_norm_requires_two_rows(encoder)
    ):
        with sentence_batch_statistics_off(encoder):
            return encoder(cropped)
    return encoder(cropped)


class BaseSpeakerFeature(nn.Module):

    def compute(self, enroll, mix=None):
        """返回该 cue 特征的计算结果。"""
        raise NotImplementedError

    def post(self, mix_repr, feat_repr):
        """返回融合后的 mix_repr。"""
        raise NotImplementedError


class ListenFeature(BaseSpeakerFeature):
    """listen cue：谱域前缀拼接。

    将 enroll 谱 + ``glue`` 对应的零谱帧拼到 mix 谱之前，分离后裁掉前缀帧。
    """

    def __init__(self, config):
        super().__init__()
        # glue 以采样点为单位配置，按 hop 换算为零谱帧数。
        if config["glue"] % config["hop"] != 0:
            logger.warning(
                "listen.glue=%d 不是 hop=%d 的整数倍，glue_frames 向下取整为 %d"
                "（实际 glue=%d 采样点）",
                config["glue"], config["hop"], config["glue"] // config["hop"],
                (config["glue"] // config["hop"]) * config["hop"],
            )
        self.glue_frames = config["glue"] // config["hop"]

    def compute(self, enroll, mix):
        """
        enroll: (..., T_e)  mix: (..., T)（实虚堆叠谱 (B,2,F,T) 或复数谱 (B,F,T)）
        return:
            x: (..., T_e + glue_frames + T)
            offset: 本次调用对应的前缀帧数
        """
        glue = torch.zeros(
            *enroll.shape[:-1], self.glue_frames,
            device=mix.device, dtype=mix.dtype,
        )
        prefix = torch.cat([enroll, glue], dim=-1)
        return torch.cat([prefix, mix], dim=-1), prefix.shape[-1]

    def post(self, mix_repr, offset, mix_len=None):
        """按本次 ``compute`` 返回的 offset 裁掉前缀帧。

        mix_repr: (..., T_total)；mix_len: 可选，期望保留的 mix 帧数。
        """
        out = mix_repr[..., offset:]
        if mix_len is not None:
            out = out[..., :mix_len]
        return out


class TFMapFeature(BaseSpeakerFeature):
    """tfmap cue：enroll 幅度谱对 mix 幅度谱的 TF 注意力图（作为第 3 通道拼接）。"""

    def __init__(self, config):
        super().__init__()
        # 仅支持谱域 TF-Map

    def compute(self, enroll, mix):
        """
        enroll: (B, F, T_e) 幅度谱
        mix: (B, F, T) 幅度谱
        return:
            tf_map: (B, F, T)
        """
        if mix.ndim != 3 or enroll.ndim != 3 or mix.shape[0] != enroll.shape[0]:
            raise ValueError("TFMapFeature expects mix/enroll [B, F, T] with matching batches")
        if mix.shape[-1] < 1 or enroll.shape[-1] < 1:
            raise ValueError("TFMapFeature requires non-empty mix and enrollment time axes")
        if mix.device != enroll.device or mix.dtype != enroll.dtype:
            raise ValueError("TFMapFeature mix and enrollment must use the same device and dtype")
        elements = mix.shape[0] * mix.shape[-1] * enroll.shape[-1]
        if elements > 16_000_000:
            raise ValueError(
                f"TFMapFeature attention has {elements} elements (>16,000,000); "
                "chunk the mixture or shorten enrollment"
            )
        mix_mag = F.normalize(mix, p=2, dim=1)
        enroll_mag = F.normalize(enroll, p=2, dim=1)

        mix_mag = mix_mag.permute(0, 2, 1).contiguous()
        att_scores = torch.matmul(mix_mag, enroll_mag)
        att_weights = F.softmax(att_scores, dim=-1)
        enroll_mag = enroll_mag.permute(0, 2, 1).contiguous()
        tf_map = torch.matmul(att_weights, enroll_mag)
        tf_map = tf_map.permute(0, 2, 1).contiguous()

        tf_map = F.normalize(tf_map, p=2, dim=1)
        # 恢复 tfmap 特征的能量
        tf_map = (torch.sum(mix * tf_map, dim=1, keepdim=True) * tf_map)
        return tf_map

    def post(self, mix_repr, feat_repr):
        """
        mix_repr: (B, 2, F, T)
        feat_repr: (B, 1, F, T)
        return:
            (B, 3, F, T)
        """
        return torch.cat([mix_repr, feat_repr], dim=1)


class EmbeddingTFMapFeature(TFMapFeature):
    """按 WeSep BSRNN_Feats 的 frame-embedding attention 生成 TF map。"""

    def __init__(self, config, encoder: SpeakerEncoder):
        super().__init__(config)
        self.encoder = encoder
        self.n_fft = int(config["n_fft"])

    def _frame_feature(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.encode_frames(self.encoder, waveform, self.n_fft)

    @staticmethod
    def encode_frames(encoder: SpeakerEncoder, waveform: torch.Tensor, n_fft: int) -> torch.Tensor:
        """复用帧编码计算；encoder 只由顶模持有，不在 Modular 中重复注册。"""
        pad = n_fft // 2
        if waveform.ndim != 2 or waveform.shape[-1] <= pad:
            raise ValueError(
                f"tfmap_emb waveform must be [B,T] with T>{pad}, got "
                f"{tuple(waveform.shape)}"
            )
        padded = F.pad(waveform.unsqueeze(1), (pad, pad), mode="reflect")
        return encoder.forward_frame_level(padded.squeeze(1))

    def compute(
        self,
        enroll: torch.Tensor,
        mix: torch.Tensor,
        *,
        enroll_wav: torch.Tensor,
        mix_wav: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(tf_map, enrollment_frame_feature)``，供 cross-fusion 复用。"""
        if mix.ndim != 3 or enroll.ndim != 3 or mix.shape[0] != enroll.shape[0]:
            raise ValueError(
                "EmbeddingTFMapFeature expects mix/enroll [B,F,T] with matching batches"
            )
        mix_frame = self._frame_feature(mix_wav)
        enroll_frame = self._frame_feature(enroll_wav)
        return self.from_frame_features(enroll, mix, enroll_frame, mix_frame), enroll_frame

    @staticmethod
    def from_frame_features(enroll, mix, enroll_frame, mix_frame):
        """帧特征决定注意力，enrollment 幅度谱提供 value；两路模型共用此核。"""
        if mix_frame.shape[-1] != mix.shape[-1]:
            raise ValueError(
                "tfmap_emb mix frame count does not match centered STFT: "
                f"embedding={mix_frame.shape[-1]}, spectrum={mix.shape[-1]}"
            )
        if enroll_frame.shape[-1] != enroll.shape[-1]:
            raise ValueError(
                "tfmap_emb enrollment frame count does not match centered STFT: "
                f"embedding={enroll_frame.shape[-1]}, spectrum={enroll.shape[-1]}"
            )
        elements = mix.shape[0] * mix.shape[-1] * enroll.shape[-1]
        if elements > 16_000_000:
            raise ValueError(
                f"EmbeddingTFMapFeature attention has {elements} elements "
                "(>16,000,000); chunk the mixture or shorten enrollment"
            )

        mix_query = F.normalize(mix_frame, p=2, dim=1).transpose(1, 2)
        enroll_key = F.normalize(enroll_frame, p=2, dim=1)
        attention = F.softmax(torch.matmul(mix_query, enroll_key), dim=-1)
        tf_map = torch.matmul(attention, enroll.transpose(1, 2)).transpose(1, 2)
        tf_map = F.normalize(tf_map, p=2, dim=1)
        tf_map = torch.sum(mix * tf_map, dim=1, keepdim=True) * tf_map
        return tf_map


class UsefFeature(BaseSpeakerFeature):
    """usef cue：USEF 跨谱注意力（enroll/mix 谱编码后做注意力，拼接为输入通道）。"""

    def __init__(self, config, eps=torch.finfo(torch.float32).eps):
        super().__init__()

        self.causal = config["causal"]
        norm_type = "LN" if self.causal else "GN"

        ks = (config["t_ksize"], 3)
        padding = (config["t_ksize"] // 2, 1)
        self.time_left_padding = config["t_ksize"] - 1

        if not self.causal:
            # === 非因果：共享编码器 ===
            self.usef_con2v = nn.Sequential(
                select_norm(norm_type, config["spec_dim"], eps),
                nn.Conv2d(config["spec_dim"],
                          config["emb_dim"],
                          ks,
                          padding=padding),
            )
        else:
            self.usef_con2v = nn.Sequential(
                select_norm(norm_type, (config["spec_dim"], config["enc_dim"]),
                            eps),
                nn.Conv2d(config["spec_dim"],
                          config["emb_dim"],
                          ks,
                          padding=(0, 1)),
            )
        self.usef_att = USEF_attentionblock(
            emb_dim=config["emb_dim"],
            n_freqs=config["enc_dim"],
            n_head=config["n_head"],
            approx_qk_dim=config["approx_qk_dim"],
        )

    def compute(self, enroll, mix):
        """
        enroll: (B, 2, F, T_e)
        mix: (B, 2, F, T)
        return:
            enroll: (B, emb_dim, F, T)
            mix: (B, emb_dim, F, T)
        """
        mix = mix.permute(0, 1, 3, 2).contiguous()  # B, 2, T, F
        enroll = enroll.permute(0, 1, 3, 2).contiguous()  # B, 2, T_e, F

        if not self.causal:
            mix = self.usef_con2v(mix)  # B, emb_dim, T, F
            enroll = self.usef_con2v(enroll)
        else:
            mix = mix.permute(0, 2, 1, 3).contiguous()  # B, T, 2, F
            mix = self.usef_con2v[0](mix)  # LN over 2, F
            mix = mix.permute(0, 2, 1, 3).contiguous()  # B, 2, T, F
            mix = F.pad(mix, (0, 0, self.time_left_padding, 0))
            mix = self.usef_con2v[1](mix)  # 严格左 padding 的时间卷积

            enroll = enroll.permute(0, 2, 1, 3).contiguous()
            enroll = self.usef_con2v[0](enroll)
            enroll = enroll.permute(0, 2, 1, 3).contiguous()
            enroll = F.pad(enroll, (0, 0, self.time_left_padding, 0))
            enroll = self.usef_con2v[1](enroll)

        enroll = self.usef_att(mix, enroll)  # B, emb_dim, T, F

        enroll = enroll.permute(0, 1, 3, 2).contiguous()  # B, emb_dim, F, T
        mix = mix.permute(0, 1, 3, 2).contiguous()  # B, emb_dim, F, T

        return enroll, mix

    def post(self, mix_repr, feat_repr):
        return torch.cat([mix_repr, feat_repr], dim=1)


class ContextFeature(BaseSpeakerFeature):
    """context cue：帧级 speaker 特征 + cross-attention 融合。"""

    def __init__(self, conf_context, encoder: SpeakerEncoder):
        super().__init__()
        # 共享 SpeakerEncoder 实例。
        self.encoder = encoder
        self.attenFuse = CrossFuse(
            embed_dim=conf_context['embed_dim'],
            atten_dim=conf_context['atten_dim'],
            mix_dim=conf_context['mix_dim'],
            num_heads=conf_context['num_heads'],
            nband=conf_context['band'],
            batch_first=True)
        self.fusionLayer = SpeakerFuseLayer(
            embed_dim=conf_context['atten_dim'],
            feat_dim=conf_context['mix_dim'],
            fuse_type=conf_context['fusion'],
        )

    def build_fusion_layer(self) -> SpeakerFuseLayer:
        """为 separator block 构造独立的 context 融合参数。"""
        return SpeakerFuseLayer(
            embed_dim=self.fusionLayer.embed_dim,
            feat_dim=self.fusionLayer.feat_dim,
            fuse_type=self.fusionLayer.fuse_type,
        )

    def compute(self, enroll_wav, mix=None):
        """
        enroll_wav: (B, T_e) 波形
        return:
            emb: (B, F_e, T_e) 帧级 speaker 特征
        """
        return self.encoder.forward_frame_level(enroll_wav)

    def attend(self, mix_repr, emb):
        """仅计算当前 separator 表征对应的 cross-attention condition。"""
        return self.attenFuse(mix_repr, emb)

    def post(self, mix_repr, emb, fusion_layer=None):
        """
        mix_repr: (B, band, F, T)
        emb: (B, F_e, T_e)
        """
        emb = self.attend(mix_repr, emb)  # (B, band, atten_dim, T)
        layer = self.fusionLayer if fusion_layer is None else fusion_layer
        mix_repr = layer(mix_repr, emb)
        return mix_repr

    def compute_and_post(self, mix_repr, enroll_wav, enroll_lengths=None):
        """按有效长度完成帧级编码和融合，不让 padding 进入 cross-attention。"""
        if enroll_lengths is None:
            return self.post(mix_repr, self.compute(enroll_wav))
        output = None
        for row_indices, cropped in _length_bucketed_waveforms(
            self.encoder, enroll_wav, enroll_lengths
        ):
            embedding = self.compute(cropped)
            fused = self.post(mix_repr.index_select(0, row_indices), embedding)
            output = _restore_bucketed_rows(
                output, row_indices, fused, enroll_wav.shape[0]
            )
        if output is None:
            raise RuntimeError("context cue received no enrollment rows")
        return output


class SpeakerEmbFeature(BaseSpeakerFeature):
    """spkemb cue：句级 speaker embedding + SpeakerFuseLayer 融合。"""

    def __init__(self, conf_emb, encoder: SpeakerEncoder | None, embed_dim: int):
        super().__init__()
        # 预提取模式下为 None，embedding 由 batch 提供。
        self.encoder = encoder
        self.fusionLayer = SpeakerFuseLayer(
            embed_dim=embed_dim,
            feat_dim=conf_emb['mix_dim'],
            fuse_type=conf_emb['fusion'],
        )

    def compute(self, enroll_wav, mix=None, enroll_lengths=None):
        """句级 embedding；给 ``enroll_lengths`` 时按真实长度分桶编码。

        enroll_wav: (B, T_e) 波形
        return:
            emb: (B, F_e) 句级 embedding

        train 模式下每个长度分组的行数 < 2 时（batch 内长度互不相同必然如此），该组
        的句级 BatchNorm 使用 running statistics，保持逐行不变且不触发
        ``Expected more than 1 value per channel when training``；多行同长度分组仍
        使用 batch statistics。
        """
        if self.encoder is None:
            raise RuntimeError(
                "SpeakerEmbFeature.compute 需要 joint_training=True；"
                "预提取模式请直接使用 batch 中的 spk_embedding"
            )
        if enroll_lengths is None:
            return self.encoder(enroll_wav)
        output = None
        for row_indices, cropped in _length_bucketed_waveforms(
            self.encoder, enroll_wav, enroll_lengths
        ):
            embedding = _encode_sentence_bucket(self.encoder, cropped)
            output = _restore_bucketed_rows(
                output, row_indices, embedding, enroll_wav.shape[0]
            )
        if output is None:
            raise RuntimeError("spkemb cue received no enrollment rows")
        return output

    def post(self, mix_repr, emb):
        return self.fusionLayer(mix_repr, emb)

    def build_fusion_layer(self) -> SpeakerFuseLayer:
        """为 separator block 构造独立参数的同型融合层。"""
        return SpeakerFuseLayer(
            embed_dim=self.fusionLayer.embed_dim,
            feat_dim=self.fusionLayer.feat_dim,
            fuse_type=self.fusionLayer.fuse_type,
        )


class SpeakerFrontend(nn.Module):
    """speaker cue 前端：按配置创建各 cue 模块（config 需已完成默认值合并）。"""

    def __init__(self, config: dict, joint_training: bool = True,
                 hop_length: int = 128, n_fft: int = 512):
        super().__init__()
        self.config = config
        self.joint_training = joint_training

        feats = self.config['features']
        spk_model_cfg = self.config['speaker_model']

        # ===== 共享 SpeakerEncoder（context + spkemb，仅 joint 模式）=====
        embedding_tfmap = (
            feats['tfmap']['enabled'] and feats['tfmap']['type'] == 'embedding'
        )
        if joint_training and (feats['context']['enabled']
                               or feats['spkemb']['enabled']
                               or embedding_tfmap):
            enc_cfg = spk_model_cfg['speaker_encoder']
            self.encoder = SpeakerEncoder(
                model=enc_cfg['model'],
                pretrained=enc_cfg.get('pretrained'),
                freeze=enc_cfg.get('freeze'),
                spk_args=enc_cfg.get('spk_args'),
                fbank_args=spk_model_cfg['fbank'],
                pretrained_prefix=enc_cfg.get('pretrained_prefix'),
            )
        shared_encoder = getattr(self, 'encoder', None)

        if feats['listen']['enabled']:
            self.listen = ListenFeature({**feats['listen'], 'hop': hop_length})

        if feats['usef']['enabled']:
            self.usef = UsefFeature(feats['usef'])

        if feats['tfmap']['enabled']:
            if embedding_tfmap:
                if shared_encoder is None:
                    raise ValueError("tfmap type='embedding' requires joint_training=true")
                self.tfmap = EmbeddingTFMapFeature(
                    {**feats['tfmap'], 'n_fft': n_fft}, shared_encoder
                )
            else:
                self.tfmap = TFMapFeature(feats['tfmap'])

        if feats['context']['enabled']:
            if shared_encoder is None:
                raise ValueError(
                    "context cue 需要帧级 speaker embedding，仅支持 "
                    "joint_training=True（预提取模式请改用 spkemb cue）"
                )
            self.context = ContextFeature(feats['context'], shared_encoder)

        if feats['spkemb']['enabled']:
            self.spkemb = SpeakerEmbFeature(
                feats['spkemb'],
                encoder=shared_encoder,
                embed_dim=feats['spkemb']['embed_dim'],
            )
