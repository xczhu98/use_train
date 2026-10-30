"""模块化 BSRNN TSE：runtime 谱进谱出，speaker/spatial/visual/textual 按启用特性声明合同。

内部 speaker encoder 默认有 checkpoint 时冻结，否则学习参数；显式 freeze=True
必须提供 checkpoint。异构固定图下整个 encoder 始终 eval，BN/Dropout 不更新状态，
非冻结路径仍保留梯度；context-only 静态冻结未消费的句级 head。其 fbank 是独立
声纹前端，不负责 separator STFT。USEF append 为本地扩展；listen 尚未接线。
speaker ``tfmap.type='embedding'`` 复用同一次 ECAPA frame-level 输出，同时生成
TFMap 和 context 条件；该路径固定为 mono、centered non-causal BSRNN，fbank 的
frame_length/frame_shift 必须与 ``n_fft/hop_length`` 对齐。speaker context/spkemb
支持显式 ``injection=latent|block``，block 路径由 ``CueComposer`` 逐 separator block
消费。

textual cue 消费 ``mix_wav`` 与 ``phoneme_ids``，由 ``TextualFrontend`` 计算句级
KCE cue：融合层由 frontend 持有（单 owner），``multi_fuse=False`` 在 latent 级注入，
``multi_fuse=True`` 在每个 separator block 前注入；KCE backbone 始终 eval 且冻结，
只有 adapter 与 fusion 可训练，缺 checkpoint 在构造期直接失败。visual cue 消费
数据层给出的帧级特征与 ``visual_timebase``，按音频帧网格对齐后由 ``CueComposer``
持有的帧级融合层注入；MuSE encoder 只在 raw video 路径使用，raw video 输入当前
仍未接线。textual/visual 都是本仓库本地接线：组合名沿用上游注册名，但不承诺
WeSep upstream checkpoint 的键结构或参数布局兼容。新模型拥有独立参数布局，
不承诺既有 TSE 模型 checkpoint 的键兼容。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.models.io_contract import ModelIOContract, TensorField
from core.utils.registry import MODEL_REGISTRY
from tasks.tse.models.modular_spec import resolve_modular_spec
from tasks.tse.models.modules.composition import CueComposer, gate
from tasks.tse.models.modules.frontend_registry import build_cue_frontend
from tasks.tse.models.modules.separator.adapters import BSRNNSeparatorAdapter
from tasks.tse.models.modules.speaker.spk_frontend import (
    EmbeddingTFMapFeature,
    SpeakerConditionOutput,
    TFMapFeature,
    UsefFeature,
)
from tasks.tse.models.speaker.encoder import SpeakerEncoder


@MODEL_REGISTRY.register("TSE_Modular")
class TSE_Modular(nn.Module):
    """模块化 BSRNN TSE 顶模。

    该模型维护独立的参数布局，不实现 multi-task speaker 分类头，因此显式声明
    ``supports_multi_task = False``，入口不得向其注入 ``multi_task`` /
    ``spks_in_train``。
    """

    supports_multichannel_input = True
    supports_multi_task = False
    supports_streaming = False
    supports_ssa = False
    requires_find_unused_parameters = False

    def __init__(
        self, *, backbone: str, separator: Mapping[str, object], cues: Mapping[str, object],
        composition: Mapping[str, object] | None = None,
        sample_rate: int = 16000, n_fft: int = 512, hop_length: int = 128,
        win_length: int = 512, window: str = "hann", center: bool = True,
        n_channel: int = 1, ref_channel: int = 0,
        mic_positions_m: Sequence[Sequence[float]] | None = None,
        speed_of_sound_mps: float | None = None,
    ) -> None:
        super().__init__()
        self.spec = resolve_modular_spec(
            backbone=backbone, separator=separator, cues=cues, composition=composition,
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center,
            n_channel=n_channel, ref_channel=ref_channel, require_implemented=True,
        )
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.n_channel = n_channel
        self.ref_channel = ref_channel
        self.supports_multichannel_input = n_channel > 1
        self.mic_positions_m = self._geometry(mic_positions_m)
        if speed_of_sound_mps is not None and (
            type(speed_of_sound_mps) not in (int, float)
            or not math.isfinite(speed_of_sound_mps) or speed_of_sound_mps <= 0
        ):
            raise ValueError("speed_of_sound_mps must be a finite positive number")
        self.speed_of_sound_mps = speed_of_sound_mps
        speaker = self.spec.cues.get("speaker")
        spatial = self.spec.cues.get("spatial")
        if spatial is None and (mic_positions_m is not None or speed_of_sound_mps is not None):
            raise ValueError("mic_positions_m/speed_of_sound_mps require spatial cues")
        self.speaker_features = {} if speaker is None else speaker.options["features"]
        self.embedding_tfmap = self.speaker_features.get("tfmap", {}).get("type") == "embedding"
        feature_dim = separator.get("feature_dim", 128)
        num_blocks = separator.get("num_repeat", 6)
        self.spatial_frontend = None
        spectral_channels = 0
        if spatial is not None:
            self.spatial_frontend = build_cue_frontend(
                spatial.frontend,
                representation=spatial.representation, features=spatial.options["features"],
                pairs=spatial.options["pairs"], num_mics=n_channel,
                sample_rate=sample_rate, n_fft=n_fft,
                mic_positions_m=self.mic_positions_m, speed_of_sound_mps=speed_of_sound_mps,
                mix_dim=feature_dim, num_layers=num_blocks,
            )
            spectral_channels = self.spatial_frontend.spectral_channels
        input_dim = 2 * n_channel
        if "usef" in self.speaker_features:
            config = self.speaker_features["usef"]
            latent_dim = config.get("latent_dim", 128)
            self.usef = UsefFeature({
                "causal": separator.get("causal", False), "spec_dim": 2,
                "enc_dim": n_fft // 2 + 1, "emb_dim": latent_dim,
                "t_ksize": config.get("time_kernel_size", 1),
                "n_head": config.get("num_heads", 4),
                "approx_qk_dim": config.get("approx_qk_dim", 512),
            })
            input_dim = (input_dim if self.spec.composition["usef_mode"] == "append" else 0) + 2 * latent_dim
        if "tfmap" in self.speaker_features:
            input_dim += 1
        self.separator = BSRNNSeparatorAdapter(
            sample_rate=sample_rate, n_fft=n_fft,
            input_dim=input_dim + spectral_channels, **dict(self.spec.separator),
        )
        self.speaker_encoder = None
        self.speaker_encoder_ckpt_path = None
        self.speaker_fbank_config = None
        encoder_options = {} if speaker is None else dict(speaker.options.get("encoder", {}))
        encoder_name = encoder_options.get("name", "ECAPA_TDNN_GLOB_c512")
        encoder_prefix = encoder_options.get("pretrained_prefix")
        embedding_dim = encoder_options.get("embedding_dim", 256 if encoder_name == "ResNet34" else 192)
        needs_encoder = speaker is not None and speaker.representation == "waveform" and bool(
            {"context", "spkemb"}.intersection(self.speaker_features)
            or (
                "tfmap" in self.speaker_features
                and self.speaker_features["tfmap"].get("type", "spec") == "embedding"
            )
        )
        if needs_encoder:
            self.speaker_encoder_ckpt_path = encoder_options.get("checkpoint")
            fbank = encoder_options.get("fbank", {})
            mel_bins = fbank.get("num_mel_bins", encoder_options.get("feature_dim", 80))
            if mel_bins != encoder_options.get("feature_dim", mel_bins):
                raise ValueError("speaker encoder feature_dim conflicts with fbank.num_mel_bins")
            self.speaker_fbank_config = {
                "sample_rate": sample_rate, "num_mel_bins": mel_bins,
                "frame_shift": fbank.get("frame_shift_ms", hop_length * 1000.0 / sample_rate if self.embedding_tfmap else 10),
                "frame_length": fbank.get("frame_length_ms", n_fft * 1000.0 / sample_rate if self.embedding_tfmap else 25),
                "dither": fbank.get("dither", 0.0),
            }
            encoder_args = {"embed_dim": embedding_dim, "feat_dim": mel_bins}
            if "pooling" in encoder_options:
                encoder_args["pooling_func"] = encoder_options["pooling"]
            self.speaker_encoder = SpeakerEncoder(
                model=encoder_name, pretrained=self.speaker_encoder_ckpt_path,
                pretrained_prefix=encoder_prefix,
                freeze=encoder_options.get("freeze", self.speaker_encoder_ckpt_path is not None),
                spk_args=encoder_args, fbank_args=self.speaker_fbank_config,
            ).eval()
            if self.embedding_tfmap and (
                self.speaker_encoder.fbank.window_size != n_fft
                or self.speaker_encoder.fbank.window_shift != hop_length
            ):
                raise ValueError("tfmap embedding fbank must match STFT window/hop in samples")
            if not self.speaker_encoder.freeze and "spkemb" not in self.speaker_features:
                unused_heads = (
                    ("conv", "pool", "bn", "linear", "bn2")
                    if encoder_name == "ECAPA_TDNN_GLOB_c512"
                    else ("pool", "seg_1", "seg_bn_1", "seg_2")
                )
                for name in unused_heads:
                    getattr(self.speaker_encoder.spk_model, name).requires_grad_(False)
        if "tfmap" in self.speaker_features:
            tfmap_config = dict(self.speaker_features["tfmap"])
            if not self.embedding_tfmap:
                self.tfmap = TFMapFeature(tfmap_config)
        context = self.speaker_features.get("context")
        context_dim = self.speaker_encoder.frame_dim if self.speaker_encoder is not None else None
        if context is not None and context.get("embedding_dim", context_dim) != context_dim:
            raise ValueError("context.embedding_dim conflicts with speaker encoder frame_dim")
        textual = self.spec.cues.get("textual")
        visual = self.spec.cues.get("visual")
        self.textual_frontend = None
        self.visual_frontend = None
        self.textual_multi_fuse = False
        self.visual_multi_fuse = False
        if textual is not None:
            self.textual_frontend = build_cue_frontend(
                textual.frontend,
                {"features": {"dae_kce": dict(textual.options["features"]["dae_kce"])}},
                mix_dim=feature_dim, num_stages=self.separator.num_blocks,
                sample_rate=sample_rate,
            )
            self.textual_multi_fuse = self.textual_frontend.multi_fuse
        visual_fusion_options = None
        if visual is not None:
            muse = dict(visual.options["features"]["muse_visual"])
            visual_fusion_options = {
                key: muse.pop(key) for key in ("fusion", "multi_fuse") if key in muse
            }
            self.visual_frontend = build_cue_frontend(
                visual.frontend,
                representation=visual.representation, features={"muse_visual": muse},
                sample_rate=sample_rate, hop_length=hop_length,
            )
            self.visual_multi_fuse = visual_fusion_options.get("multi_fuse", True)
        self.composer = CueComposer(
            feature_dim=feature_dim, nband=self.separator.nband, num_blocks=num_blocks,
            context=context, context_dim=context_dim,
            spkemb=self.speaker_features.get("spkemb"), embedding_dim=embedding_dim,
            cyclic_fusion=(self.spatial_frontend.cyc_doaemb.fusion
                           if self.spatial_frontend is not None and hasattr(self.spatial_frontend, "cyc_doaemb")
                           else None),
            visual=visual_fusion_options,
            visual_dim=None if self.visual_frontend is None else self.visual_frontend.output_dim,
        )
        self.embedding_dim = embedding_dim
        self.IO_CONTRACT = self._input_contract()

    def _geometry(self, positions):
        if positions is None:
            return None
        value = torch.as_tensor(positions, dtype=torch.float64)
        if value.shape != (self.n_channel, 3) or not torch.isfinite(value).all():
            raise ValueError(f"mic_positions_m must be finite [{self.n_channel},3]")
        return tuple(tuple(row) for row in value.tolist())

    def train(self, mode: bool = True):
        super().train(mode)
        if self.speaker_encoder is not None:
            self.speaker_encoder.eval()
        return self

    def _input_contract(self):
        axes = ("batch", "microphone", "frequency", "frame", "complex") if self.n_channel > 1 else (
            "batch", "frequency", "frame", "complex",
        )
        fields = [TensorField("mix_spectra", "spectrum", axes)]
        speaker = self.spec.cues.get("speaker")
        if speaker is not None:
            if speaker.representation == "embedding":
                fields.extend((
                    TensorField("speaker_embedding", "embedding", ("batch", "embedding")),
                    TensorField("speaker_embedding_present", "mask", ("batch",)),
                ))
            else:
                fields.append(TensorField("speaker_waveform_present", "mask", ("batch",)))
                if {"usef", "tfmap"}.intersection(self.speaker_features):
                    fields.extend((
                        TensorField("enroll_spectra", "spectrum", ("batch", "frequency", "frame", "complex")),
                        TensorField("enroll_spectra_lengths", "label", ("batch",)),
                    ))
                if (
                    "tfmap" in self.speaker_features
                    and self.speaker_features["tfmap"].get("type", "spec") == "embedding"
                ):
                    fields.extend(
                        (
                            TensorField("mix_wav", "waveform", ("batch", "time")),
                            TensorField("lengths", "label", ("batch",), required=False),
                        )
                    )
                if self.speaker_encoder is not None:
                    fields.extend((
                        TensorField("speaker_waveform", "waveform", ("batch", "time")),
                        TensorField("speaker_waveform_lengths", "label", ("batch",)),
                    ))
        spatial = self.spec.cues.get("spatial")
        if spatial is not None:
            fields.extend((
                TensorField("spatial", "embedding", ("batch", "angle", "point") if spatial.representation == "track" else ("batch", "angle")),
                TensorField("spatial_present", "mask", ("batch",)),
            ))
            if spatial.representation == "track":
                fields.extend((
                    TensorField("spatial_lengths", "label", ("batch",)),
                    TensorField("spatial_timebase", "label", ("batch", "point")),
                ))
        if self.textual_frontend is not None:
            fields.extend((
                TensorField("mix_wav", "waveform", ("batch", "time")),
                TensorField("lengths", "label", ("batch",), required=False),
                TensorField("textual", "label", ("batch", "token")),
                TensorField("textual_present", "mask", ("batch",)),
                TensorField("textual_lengths", "label", ("batch",)),
            ))
        if self.visual_frontend is not None:
            fields.extend((
                TensorField("visual", "embedding", ("batch", "dimension", "point")),
                TensorField("visual_present", "mask", ("batch",)),
                TensorField("visual_lengths", "label", ("batch",)),
                TensorField("visual_timebase", "label", ("batch", "point")),
            ))
        return ModelIOContract(
            tasks=frozenset({"tse"}), inputs=tuple(fields),
            outputs=(TensorField("target_estimate_spectra", "spectrum", ("batch", "frequency", "frame", "complex")),),
            forward_kind="estimate", frontend_owner="runtime",
        )

    @staticmethod
    def _presence(data, name, reference):
        value = data[name]
        if not isinstance(value, Tensor) or value.dtype != torch.bool or value.shape != (reference.shape[0],):
            raise ValueError(f"{name} must be bool [R]")
        if value.device != reference.device:
            raise ValueError(f"{name} must use mixture device")
        return value

    @staticmethod
    def _cue_lengths(value, present, capacity, name):
        """校验整数长度向量并返回原 tensor；缺失行必须精确为 0。"""
        if not isinstance(value, Tensor) or value.dtype not in (torch.int32, torch.int64) or value.shape != present.shape:
            raise ValueError(f"{name} must be integer [R]")
        if value.device != present.device or bool(((value < 0) | (value > capacity)).any()):
            raise ValueError(f"{name} exceeds capacity or uses wrong device")
        if bool(((value == 0) != ~present).any()):
            raise ValueError(f"{name} must be zero exactly for missing rows")
        return value

    @staticmethod
    def _lengths(value, present, capacity, name):
        return TSE_Modular._cue_lengths(value, present, capacity, name).tolist()

    @staticmethod
    def _mixture_lengths(data, waveform, rows):
        """textual KCE 的 mixture 有效样点数；缺 ``lengths`` 时按整段消费。"""
        value = data.get("lengths")
        if value is None:
            return torch.full(
                (rows,), waveform.shape[-1], dtype=torch.int32, device=waveform.device,
            )
        if (
            not isinstance(value, Tensor)
            or value.dtype not in (torch.int32, torch.int64)
            or value.shape != (rows,)
        ):
            raise ValueError("lengths must be an integer [R] tensor aligned with mix_wav")
        if value.device != waveform.device or bool(
            ((value < 1) | (value > waveform.shape[-1])).any()
        ):
            raise ValueError("lengths must be within [1, T] of mix_wav")
        return value

    def _speaker_spectra(self, data, reference, present):
        enroll = data["enroll_spectra"]
        if enroll.ndim != 4 or enroll.shape[:2] != reference.shape[:2] or enroll.shape[-1] != 2:
            raise ValueError("enroll_spectra must be [R,F,T,2] aligned with mixture")
        if enroll.device != reference.device or enroll.dtype != reference.real.dtype:
            raise ValueError("enroll_spectra must match mixture dtype/device")
        lengths = self._lengths(data["enroll_spectra_lengths"], present, enroll.shape[-2], "enroll_spectra_lengths")
        enroll = gate(enroll, present)
        usef, tfmap = [], []
        for row, length in enumerate(lengths):
            spectrum = torch.view_as_complex(enroll[row:row + 1, :, :max(1, length)].contiguous())
            mix = reference[row:row + 1]
            if hasattr(self, "usef"):
                usef.append(self.usef.compute(
                    torch.stack((spectrum.real, spectrum.imag), dim=1),
                    torch.stack((mix.real, mix.imag), dim=1),
                ))
            if hasattr(self, "tfmap"):
                tfmap.append(self.tfmap.compute(spectrum.abs(), mix.abs()).unsqueeze(1))
        return (
            tuple(torch.cat([value[index] for value in usef]) for index in range(2)) if usef else None,
            torch.cat(tfmap) if tfmap else None,
        )

    def _speaker_encoding(self, data, present):
        waveform = data["speaker_waveform"]
        if waveform.ndim != 2 or waveform.shape[0] != present.shape[0] or waveform.device != present.device:
            raise ValueError("speaker_waveform must be [R,T] on the mixture device")
        lengths = self._lengths(data["speaker_waveform_lengths"], present, waveform.shape[-1], "speaker_waveform_lengths")
        waveform = gate(waveform, present)
        context, embeddings = [], []
        minimum = self.speaker_encoder.fbank.window_size + self.speaker_encoder.fbank.window_shift
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.speaker_encoder.freeze):
            for row, length in enumerate(lengths):
                value = waveform[row:row + 1, :length]
                value = F.pad(value, (0, max(0, minimum - length)))
                if "context" in self.speaker_features:
                    context.append(self.speaker_encoder.forward_frame_level(value))
                if "spkemb" in self.speaker_features:
                    embeddings.append(self.speaker_encoder(value))
        return context or None, torch.cat(embeddings) if embeddings else None

    def _embedding_speaker_condition(self, data, reference, present) -> SpeakerConditionOutput:
        """每个有效时间窗只计算一次 mix/enroll 帧编码，并把同一 context 交给融合层。

        缺失行使用有限的最短占位参与计算，最终按 presence 门控；encoder 的参数
        只有 ``self.speaker_encoder`` 一个 owner。这里不执行 STFT。
        """
        mix_wav = data["mix_wav"]
        waveform = data["speaker_waveform"]
        enroll = data["enroll_spectra"]
        rows, bins, frames = reference.shape
        for name, value in (("mix_wav", mix_wav), ("speaker_waveform", waveform)):
            if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[0] != rows:
                raise ValueError(f"{name} must be [R,T]")
            if value.device != reference.device or value.dtype != reference.real.dtype:
                raise ValueError(f"{name} must match mixture dtype/device")
        if enroll.ndim != 4 or enroll.shape[:2] != (rows, bins) or enroll.shape[-1] != 2:
            raise ValueError("enroll_spectra must be [R,F,Te,2]")
        if enroll.device != reference.device or enroll.dtype != reference.real.dtype:
            raise ValueError("enroll_spectra must match mixture dtype/device")
        lengths = self._lengths(data["speaker_waveform_lengths"], present, waveform.shape[-1], "speaker_waveform_lengths")
        spectrum_lengths = self._lengths(data["enroll_spectra_lengths"], present, enroll.shape[-2], "enroll_spectra_lengths")
        mixture_lengths = self._mixture_lengths(data, mix_wav, rows).tolist()
        if 1 + mix_wav.shape[-1] // self.hop_length != frames:
            raise ValueError("tfmap embedding mix_wav frame count does not match centered STFT")
        waveform = gate(waveform, present)
        enroll = gate(enroll, present)
        tfmaps, contexts, embeddings = [], [], []
        minimum = self.n_fft // 2 + 1
        for row, (samples, spectral_length, mix_samples) in enumerate(zip(lengths, spectrum_lengths, mixture_lengths)):
            enroll_samples = max(minimum, samples)
            enroll_frames = 1 + enroll_samples // self.hop_length
            if samples and spectral_length != enroll_frames:
                raise ValueError("tfmap embedding enrollment frame count does not match waveform length")
            enroll_wav = F.pad(waveform[row:row + 1, :samples], (0, enroll_samples - samples))
            mix_samples_padded = max(minimum, mix_samples)
            mix_value = F.pad(mix_wav[row:row + 1, :mix_samples], (0, mix_samples_padded - mix_samples))
            mix_frames = 1 + mix_samples_padded // self.hop_length
            if mix_frames > frames:
                raise ValueError("tfmap embedding mixture is shorter than the runtime STFT minimum")
            if not bool(torch.isfinite(enroll_wav).all()) or not bool(torch.isfinite(mix_value).all()):
                raise ValueError("tfmap embedding waveforms must be finite within valid lengths")
            with torch.set_grad_enabled(torch.is_grad_enabled() and not self.speaker_encoder.freeze):
                mix_feature = EmbeddingTFMapFeature.encode_frames(self.speaker_encoder, mix_value, self.n_fft)
                context = EmbeddingTFMapFeature.encode_frames(self.speaker_encoder, enroll_wav, self.n_fft)
                if "spkemb" in self.speaker_features:
                    sentence_input = F.pad(enroll_wav, (0, max(0, self.n_fft + self.hop_length - enroll_samples)))
                    embeddings.append(self.speaker_encoder(sentence_input))
            # 缺失谱的有效帧数为零；占位仅维护 shape 和固定图，不能伪造 present。
            enroll_mag = (
                torch.view_as_complex(enroll[row:row + 1, :, :spectral_length].contiguous()).abs()
                if samples else reference.real.new_zeros((1, bins, enroll_frames))
            )
            value = EmbeddingTFMapFeature.from_frame_features(
                enroll_mag, reference[row:row + 1, :, :mix_frames].abs(), context, mix_feature,
            )
            tfmaps.append(F.pad(value.unsqueeze(1), (0, frames - mix_frames)))
            contexts.append(context)
        return SpeakerConditionOutput(
            present=present, spectral_cue=gate(torch.cat(tfmaps), present),
            frame_context=tuple(contexts),
            sentence_embedding=torch.cat(embeddings) if embeddings else None,
        )

    def forward(self, data: dict) -> dict:
        missing = {field.name for field in self.IO_CONTRACT.inputs if field.required} - data.keys()
        if missing:
            raise ValueError(f"missing required TSE_Modular inputs: {sorted(missing)}")
        raw = data["mix_spectra"]
        expected_ndim = 5 if self.n_channel > 1 else 4
        if raw.ndim != expected_ndim or raw.shape[-1] != 2 or raw.shape[-3] != self.n_fft // 2 + 1:
            raise ValueError("mix_spectra conflicts with model frequency/microphone contract")
        if self.n_channel > 1 and raw.shape[1] != self.n_channel:
            raise ValueError("mix_spectra microphone count conflicts with n_channel")
        mixture = torch.view_as_complex(raw.contiguous())
        if self.n_channel == 1:
            mixture = mixture.unsqueeze(1)
        reference = mixture[:, self.ref_channel]
        speaker = self.spec.cues.get("speaker")
        speaker_present = None
        presences = []
        if speaker is not None:
            name = "speaker_embedding_present" if speaker.representation == "embedding" else "speaker_waveform_present"
            speaker_present = self._presence(data, name, reference)
            presences.append(speaker_present)
        spatial_present = None
        if self.spatial_frontend is not None:
            spatial_present = self._presence(data, "spatial_present", reference)
            presences.append(spatial_present)
        textual_present = None
        if self.textual_frontend is not None:
            textual_present = self._presence(data, "textual_present", reference)
            presences.append(textual_present)
        visual_present = None
        if self.visual_frontend is not None:
            visual_present = self._presence(data, "visual_present", reference)
            presences.append(visual_present)
        if not self.spec.composition["allow_all_missing"] and bool((~torch.stack(presences).any(0)).any()):
            raise ValueError("all cues are missing for a target; enable composition.allow_all_missing only for explicit research")
        if self.embedding_tfmap:
            speaker_condition = self._embedding_speaker_condition(data, reference, speaker_present)
        else:
            usef, tfmap, context, embedding = None, None, None, None
            if {"usef", "tfmap"}.intersection(self.speaker_features):
                usef, tfmap = self._speaker_spectra(data, reference, speaker_present)
            if self.speaker_encoder is not None:
                context, embedding = self._speaker_encoding(data, speaker_present)
            elif speaker is not None and speaker.representation == "embedding":
                embedding = data["speaker_embedding"]
                if embedding.shape != (reference.shape[0], self.embedding_dim):
                    raise ValueError(f"speaker_embedding must be [R,{self.embedding_dim}]")
                embedding = gate(embedding, speaker_present)
            speaker_condition = SpeakerConditionOutput(
                present=speaker_present, spectral_cue=tfmap, usef=usef,
                frame_context=tuple(context) if context is not None else None,
                sentence_embedding=embedding,
            )
        spatial = None
        if self.spatial_frontend is not None:
            frame_indices = torch.arange(reference.shape[-1], device=reference.device) * self.hop_length
            if not self.center:
                frame_indices = frame_indices + self.n_fft // 2
            spatial = self.spatial_frontend(
                mixture, spatial_cue=gate(data["spatial"], spatial_present), present=spatial_present,
                frame_sample_indices=frame_indices,
                sample_anchors=data.get("spatial_timebase"), lengths=data.get("spatial_lengths"),
            )
        textual_output = self._textual_cue(data, reference, textual_present)
        visual_embedding = self._visual_condition(data, reference, visual_present)
        spectral_repr = self.composer.spectral(
            mixture, spatial=None if spatial is None else spatial.spectral,
            usef=speaker_condition.usef, usef_mode=self.spec.composition["usef_mode"],
            tfmap=speaker_condition.spectral_cue, speaker_present=speaker_present,
        )

        def latent_fusion(latent):
            latent = self.composer.fuse_latent(
                latent,
                present=speaker_present,
                context=(
                    speaker_condition.frame_context
                    if self.speaker_features.get("context", {}).get("injection", "latent") == "latent"
                    else None
                ),
                embedding=(
                    speaker_condition.sentence_embedding
                    if self.speaker_features.get("spkemb", {}).get("injection", "latent") == "latent"
                    else None
                ),
                visual=None if visual_embedding is None or self.visual_multi_fuse else (
                    visual_embedding, visual_present,
                ),
            )
            if textual_output is not None and not self.textual_multi_fuse:
                latent = self.textual_frontend.post(latent, textual_output, 0)
            return latent

        conditioners, conditions = self.composer.block_injection(
            speaker=(
                speaker_condition.frame_context,
                speaker_condition.sentence_embedding,
                speaker_present,
            ),
            spatial=(spatial.cyclic_embedding, spatial_present)
            if spatial is not None and len(self.composer.cyclic_fusions) else None,
            visual=(visual_embedding, visual_present)
            if visual_embedding is not None and self.visual_multi_fuse else None,
            textual_stages=(self.textual_frontend.dae_kce.fusions, textual_output.cue, textual_present)
            if textual_output is not None and self.textual_multi_fuse else None,
        )
        estimate = self.separator(
            spectral_repr, reference,
            latent_fusion=latent_fusion,
            block_conditioners=conditioners,
            block_conditions=conditions,
            initial_states=None if spatial is None else spatial.initial_states,
        )
        return {**data, "target_estimate_spectra": estimate}

    def _textual_cue(self, data, reference, present):
        """用 mixture 波形与 phoneme id 计算句级 KCE cue。"""
        if self.textual_frontend is None:
            return None
        waveform = data["mix_wav"]
        if waveform.ndim != 2 or waveform.shape[0] != reference.shape[0]:
            raise ValueError("mix_wav must be [R,T] aligned with mix_spectra")
        if waveform.device != reference.device or waveform.dtype != reference.real.dtype:
            raise ValueError("mix_wav must match mixture dtype/device")
        phoneme_ids = data["textual"]
        if phoneme_ids.ndim != 2 or phoneme_ids.shape[0] != reference.shape[0]:
            raise ValueError("textual must be [R,L] phoneme ids aligned with mix_spectra")
        if phoneme_ids.device != reference.device:
            raise ValueError("textual must use the mixture device")
        return self.textual_frontend(
            waveform, phoneme_ids, present=present,
            textual_lengths=self._cue_lengths(
                data["textual_lengths"], present, phoneme_ids.shape[1], "textual_lengths",
            ),
            mixture_lengths=self._mixture_lengths(data, waveform, reference.shape[0]),
        )

    def _visual_condition(self, data, reference, present):
        """把帧级视觉 cue 按 ``visual_timebase`` 对齐到音频帧网格。"""
        if self.visual_frontend is None:
            return None
        value = data["visual"]
        if value.ndim < 2 or value.shape[0] != reference.shape[0]:
            raise ValueError("visual must have one row per target")
        if value.device != reference.device:
            raise ValueError("visual must use the mixture device")
        lengths = self._cue_lengths(
            data["visual_lengths"], present, value.shape[-1], "visual_lengths",
        )
        timebase = data["visual_timebase"]
        if (
            not isinstance(timebase, Tensor)
            or not timebase.dtype.is_floating_point
            or tuple(timebase.shape) != (reference.shape[0], value.shape[-1])
        ):
            raise ValueError("visual_timebase must be a floating [R,V] tensor aligned with visual")
        if timebase.device != reference.device:
            raise ValueError("visual_timebase must use the mixture device")
        return self.visual_frontend(
            value, present=present, num_frames=reference.shape[-1],
            frame_times=timebase, lengths=lengths,
        ).embedding
