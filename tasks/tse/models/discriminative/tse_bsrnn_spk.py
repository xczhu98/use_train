"""TSE_BSRNN_SPK 目标说话人提取模型（dict 谱进谱出接口，mono mixture）。

该模型当前是 mono-only：它接受 ``mix_spectra [B,F,T,2]``。空间 mixture
``[B,C,F,T,2]`` 必须使用显式声明并实际实现 microphone axis 的 native spatial
TSE 模型；本模型不会隐式选择 reference channel。

::

    forward({"mix_spectra":    [B, F, T,  2],   # 混合谱（view_as_real 格式）
             "enroll_spectra": [B, F, Te, 2],   # 注册音谱（listen/usef/tfmap 用）
             "enroll_spectra_lengths": [B],    # int32 有效帧数（整句输入必填）
             "enroll_lengths": [B],            # waveform cue 的真实样点数
             "enroll_wav":     [B, Te_wav],     # 注册音波形（joint 模式提 embedding）
             "spk_embedding":  [B, 192]})       # 预提取 embedding（joint_training=False）
      -> {"target_estimate_spectra": [B, F, T, 2]}   # multi_task=true 时另含
                                                     # "spk_logits": [B, spks_in_train]

谱参数由 ``model.args`` 注入并在构造时校验。``causal=True`` 使用固化的
enrollment cue、USEF 左上下文和时间 LSTM state；切换 enrollment 必须通过
``reset_streaming_state`` 清空全部历史。

:meth:`load_pretrained` 在复制前校验 0-match、missing/unexpected keys 及 tensor
shape。公共加载默认严格；``strict=False`` **仅限第三方官方比赛 checkpoint**，且不放宽
安全加载、shape 或 0-match gate。

Usage:
    model:
      type: TSE_BSRNN_SPK
      args:
        joint_training: true
        features:
          spkemb: {enabled: true}
        speaker_model:
          speaker_encoder:
            pretrained: tasks/tse/speaker_embedding_models/ecapa_tdnn_glob_c512/avg_model.pt

Author: Xincheng Zhu
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch
import torch.nn as nn

from core.models.io_contract import ModelIOContract, TensorField
from core.utils.registry import MODEL_REGISTRY
from core.utils.safe_load import safe_torch_load
from tasks.tse.model_io import TSEStreamingState
from tasks.tse.models.modules.common.deep_update import deep_update
from tasks.tse.models.modules.separator.bsrnn import BSRNN
from tasks.tse.models.speaker.config import config_bool, require_bool
from tasks.tse.models.modules.frontend_registry import build_cue_frontend

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register("TSE_BSRNN_SPK")
class TSE_BSRNN_SPK(nn.Module):
    """TSE_BSRNN_SPK：BandSplit-RNN 目标说话人提取（谱进谱出）。

    Args:
        sr: 采样率（决定 BandSplit 子带表与 fbank）。
        feature_dim: BSRNN 特征维（每子带）。
        num_repeat: BSNet 堆叠数。
        causal: 因果变体（每个时间帧沿通道做 cLN + 时间维单向 LSTM；
            cLN 不跨时间统计）。
        joint_training: True 时模型内含 SpeakerEncoder，从 ``enroll_wav`` 提
            embedding（默认，与比赛 ckpt 对齐）；False 时仅在启用 spkemb cue
            时需要预提取的 ``spk_embedding``，context 不支持；listen/usef/tfmap
            仍需有效的 ``enroll_spectra``。
        features: 5 种 speaker cue 的配置 dict（listen/usef/tfmap/context/spkemb），
            经严格 deep_update 合并（未知键报错）。
        speaker_model: fbank + speaker encoder 配置 dict。
        multi_task: True 时在句级 embedding（spkemb cue 产物）后接 speaker
            分类头 ``nn.Linear(embed_dim, spks_in_train)``，forward 返回 dict
            追加 ``spk_logits``（供 ``SpeakerCrossEntropyLoss`` 多任务辅助）；
            False（默认）时无分类头、无 ``spk_logits`` 键，行为与之前一致。
        spks_in_train: 训练集说话人总数（分类头输出维）；multi_task=True
            必填（``TSETrainingEntry`` 可从 dataset ``speaker_identity_to_idx`` 大小自动注入）。
    """

    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"tse"}),
        inputs=(
            TensorField(
                "mix_spectra",
                "spectrum",
                ("batch", "frequency", "frame", "complex"),
            ),
            TensorField(
                "enroll_spectra",
                "spectrum",
                ("batch", "frequency", "frame", "complex"),
                required=False,
            ),
            TensorField("enroll_spectra_lengths", "label", ("batch",), required=False),
            TensorField(
                "enroll_wav",
                "waveform",
                ("batch", "time"),
                required=False,
            ),
            TensorField(
                "spk_embedding",
                "embedding",
                ("batch", "embedding"),
                required=False,
            ),
        ),
        outputs=(
            TensorField(
                "target_estimate_spectra",
                "spectrum",
                ("batch", "frequency", "frame", "complex"),
            ),
            TensorField(
                "spk_logits",
                "logits",
                ("batch", "speaker_class"),
                required=False,
            ),
        ),
        forward_kind="estimate",
        frontend_owner="runtime",
    )

    def __init__(
        self,
        sr: int = 16000,
        feature_dim: int = 128,
        num_repeat: int = 6,
        causal: bool = False,
        joint_training: bool = True,
        features: dict | None = None,
        speaker_model: dict | None = None,
        multi_task: bool = False,
        spks_in_train: int | None = None,
        multi_fuse: bool = False,
        **kwargs,
    ):
        super().__init__()
        # 框架经 model.args setdefault 注入的 stft 参数：pop 并校验。
        for name, value in (("sr", sr), ("feature_dim", feature_dim), ("num_repeat", num_repeat)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self.sample_rate = sr
        self.n_fft = kwargs.pop("n_fft", 512)
        self.hop_length = kwargs.pop("hop_length", 128)
        win_length = kwargs.pop("win_length", None)
        for name, value in (("n_fft", self.n_fft), ("hop_length", self.hop_length)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.n_fft % 2:
            raise ValueError(f"n_fft must be even, got {self.n_fft}")
        if self.hop_length > self.n_fft:
            raise ValueError(
                f"hop_length must not exceed n_fft, got {self.hop_length} > {self.n_fft}"
            )
        if win_length is not None and (
            isinstance(win_length, bool) or not isinstance(win_length, int) or win_length < 1
        ):
            raise ValueError(f"win_length must be a positive integer or None, got {win_length!r}")
        # window/center 由 trainer 侧 stft_module 消费，模型本身不使用，
        # 但与 win_length 同标准校验：比赛 ckpt 的 STFT 条件固定为 hann/center=True
        window = kwargs.pop("window", "hann")
        center = require_bool(kwargs.pop("center", True), "center")
        causal = require_bool(causal, "causal")
        joint_training = require_bool(joint_training, "joint_training")
        multi_task = require_bool(multi_task, "multi_task")
        multi_fuse = require_bool(multi_fuse, "multi_fuse")
        if kwargs:
            raise TypeError(
                f"TSE_BSRNN_SPK got unexpected keyword arguments: {sorted(kwargs)}"
            )
        if win_length is not None and win_length != self.n_fft:
            raise ValueError(
                f"TSE_BSRNN_SPK requires win_length == n_fft (got "
                f"win_length={win_length}, n_fft={self.n_fft})：原实现 "
                f"STFT 窗长等于 n_fft，BandSplit 子带表据此推导"
            )
        if window != "hann":
            raise ValueError(
                f"TSE_BSRNN_SPK requires window == 'hann' (got {window!r})："
                f"比赛 checkpoint 的 STFT 条件固定为 hann 窗"
            )
        if not center:
            raise ValueError(
                "TSE_BSRNN_SPK requires center=True（比赛 checkpoint 的 STFT "
                "条件固定为 center=True）"
            )
        self.joint_training = joint_training
        self.causal = causal
        self.multi_fuse = multi_fuse
        # ===== speaker cue 默认值（与原实现对齐，全部默认关闭）=====
        feat_defaults = {
            "listen": {"enabled": False, "glue": 512},
            "usef": {
                "enabled": False,
                "causal": causal,
                "spec_dim": 2,
                "emb_dim": feature_dim // 2,
                "enc_dim": self.n_fft // 2 + 1,
                "approx_qk_dim": 512,
                "n_head": 4,
                "t_ksize": 3,
            },
            "tfmap": {"enabled": False, "type": "spec"},
            "context": {
                "enabled": False,
                "embed_dim": 512,
                "atten_dim": feature_dim,
                "num_heads": 2,
                "fusion": "multiply",
                "mix_dim": feature_dim,
                "band": 1,  # 构造 sep_model 后改写为 nband
            },
            "spkemb": {
                "enabled": False,
                "fusion": "multiply",
                "mix_dim": feature_dim,
                "embed_dim": 192,  # 构造前由 spk_args 改写
            },
        }
        spk_defaults = {
            "fbank": {
                "num_mel_bins": 80,
                "frame_shift": 10,
                "frame_length": 25,
                "dither": 0.0,
                "sample_rate": sr,
                "norm_mean": True,  # 句级 mean-CMN（wesep 默认）
                "norm_var": False,
            },
            "speaker_encoder": {
                "model": "ECAPA_TDNN_GLOB_c512",
                "pretrained": None,  # ckpt 路径必须 YAML 注入（禁硬编码）
                "pretrained_prefix": None,
                "freeze": None,  # None=有预训练则冻结
                "spk_args": {"embed_dim": 192, "feat_dim": 80, "pooling_func": "ASTP"},
            },
        }
        if features is None:
            feature_overrides = {}
        elif not isinstance(features, Mapping):
            raise TypeError("features must be a mapping")
        else:
            feature_overrides = {}
            for feature_name, feature_cfg in features.items():
                if not isinstance(feature_cfg, Mapping):
                    raise TypeError(
                        f"features.{feature_name} must be a mapping"
                    )
                feature_cfg = dict(feature_cfg)
                feature_path = f"features.{feature_name}"
                config_bool(feature_cfg, "enabled", False, feature_path)
                if feature_name == "usef" and "causal" in feature_cfg:
                    require_bool(feature_cfg["causal"], f"{feature_path}.causal")
                feature_overrides[feature_name] = feature_cfg
        feat_cfg = deep_update(feat_defaults, feature_overrides)
        if speaker_model is None:
            speaker_overrides = {}
        elif not isinstance(speaker_model, Mapping):
            raise TypeError("speaker_model must be a mapping")
        else:
            speaker_overrides = dict(speaker_model)
            fbank_overrides = speaker_overrides.get("fbank")
            if isinstance(fbank_overrides, Mapping):
                fbank_overrides = dict(fbank_overrides)
                for key in ("norm_mean", "norm_var"):
                    if key in fbank_overrides:
                        require_bool(
                            fbank_overrides[key],
                            f"speaker_model.fbank.{key}",
                        )
                speaker_overrides["fbank"] = fbank_overrides
            encoder_overrides = speaker_overrides.get("speaker_encoder")
            if isinstance(encoder_overrides, Mapping):
                encoder_overrides = dict(encoder_overrides)
                prefix = encoder_overrides.get("pretrained_prefix")
                if prefix is not None and (
                    not isinstance(prefix, str)
                    or not prefix
                    or prefix != prefix.strip()
                ):
                    raise ValueError(
                        "speaker_model.speaker_encoder.pretrained_prefix must be "
                        "a non-empty trimmed string or null"
                    )
                if "freeze" in encoder_overrides and encoder_overrides["freeze"] is not None:
                    require_bool(
                        encoder_overrides["freeze"],
                        "speaker_model.speaker_encoder.freeze",
                    )
                speaker_overrides["speaker_encoder"] = encoder_overrides
        spk_cfg = deep_update(spk_defaults, speaker_overrides)
        self.features_config = feat_cfg
        # 双路监督由 Trainer 配置；模型只声明当前结构是否具备所需条件路径。
        self.supports_dual_self_estimated = joint_training and feat_cfg["spkemb"]["enabled"]

        # ===== 配置校验 =====
        if int(spk_cfg["fbank"]["sample_rate"]) != self.sample_rate:
            raise ValueError(
                f"speaker_model.fbank.sample_rate="
                f"{spk_cfg['fbank']['sample_rate']} 与顶层 sr={self.sample_rate} 不一致"
            )
        if not joint_training and feat_cfg["context"]["enabled"]:
            raise ValueError(
                "joint_training=False（预提取 embedding）不支持 context cue"
                "（需要帧级 embedding），请改用 spkemb cue"
            )
        if feat_cfg["tfmap"]["enabled"] and feat_cfg["tfmap"]["type"] not in {
            "spec",
            "embedding",
        }:
            raise ValueError("tfmap.type must be 'spec' or 'embedding'")
        if feat_cfg["tfmap"]["type"] == "embedding" and not joint_training:
            raise ValueError("tfmap type='embedding' requires joint_training=true")
        if feat_cfg["tfmap"]["type"] == "embedding" and causal:
            raise ValueError(
                "tfmap type='embedding' does not support causal mode without "
                "a mixture-waveform chunk contract"
            )
        if feat_cfg["usef"]["enabled"]:
            if feat_cfg["usef"]["causal"] != causal:
                raise ValueError(
                    f"features.usef.causal={feat_cfg['usef']['causal']} 与顶层 "
                    f"causal={causal} 不一致"
                )
            exp_enc_dim = self.n_fft // 2 + 1
            if feat_cfg["usef"]["enc_dim"] != exp_enc_dim:
                raise ValueError(
                    f"usef.enc_dim={feat_cfg['usef']['enc_dim']} 与 n_fft 推导的 "
                    f"F={exp_enc_dim} 不一致"
                )

        # ===== separator（spec_dim 由启用的 cue 决定输入通道数）=====
        spec_dim = 2
        if feat_cfg["usef"]["enabled"]:
            spec_dim = feat_cfg["usef"]["emb_dim"] * 2
        if feat_cfg["tfmap"]["enabled"]:
            spec_dim = spec_dim + 1
        self.sep_model = BSRNN(
            sr=sr,
            win=self.n_fft,
            stride=self.hop_length,
            feature_dim=feature_dim,
            num_repeat=num_repeat,
            causal=causal,
            nspk=1,
            spec_dim=spec_dim,
        )

        # ===== speaker frontend =====
        feat_cfg["context"]["band"] = self.sep_model.nband
        feat_cfg["spkemb"]["embed_dim"] = spk_cfg["speaker_encoder"]["spk_args"][
            "embed_dim"]
        self.spk_ft = build_cue_frontend(
            "speaker",
            {"features": feat_cfg, "speaker_model": spk_cfg},
            joint_training=joint_training,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )
        if self.multi_fuse:
            active_fuse_features = [
                name
                for name in ("context", "spkemb")
                if feat_cfg[name]["enabled"]
            ]
            if len(active_fuse_features) != 1:
                raise ValueError(
                    "multi_fuse=true requires exactly one of context/spkemb"
                )
            self.multi_fuse_feature = active_fuse_features[0]
            if (
                self.multi_fuse_feature == "context"
                and feat_cfg["tfmap"]["type"] != "embedding"
            ):
                raise ValueError(
                    "context multi_fuse currently requires tfmap.type='embedding'"
                )
            feature = getattr(self.spk_ft, self.multi_fuse_feature)
            self.multi_fuse_layers = nn.ModuleList(
                feature.build_fusion_layer()
                for _ in range(num_repeat)
            )

        # ===== multi-task speaker 分类头（可选；句级 embedding → 说话人 logits）=====
        self.multi_task = multi_task
        if self.multi_task:
            if not feat_cfg["spkemb"]["enabled"]:
                raise ValueError(
                    "multi_task=true 需要启用 spkemb cue：分类头接在句级 "
                    "speaker embedding（spkemb 产物）之后"
                )
            if spks_in_train is None or int(spks_in_train) < 2:
                raise ValueError(
                    f"multi_task=true 需要 spks_in_train >= 2，got {spks_in_train!r}"
                    "（TSETrainingEntry 可从 dataset speaker_identity_to_idx 大小自动注入）"
                )
            self.spk_classifier = nn.Linear(
                feat_cfg["spkemb"]["embed_dim"], int(spks_in_train)
            )
        # 实例合同绑定当前 cue/joint 配置；不改变任何参数模块或 checkpoint 键。
        self.IO_CONTRACT = self._build_io_contract()

    def _build_io_contract(self) -> ModelIOContract:
        """构造当前模型实例真正需要的 cue-resolved I/O 合同。"""
        inputs = [
            TensorField(
                "mix_spectra",
                "spectrum",
                ("batch", "frequency", "frame", "complex"),
            )
        ]
        if any(
            self.features_config[name]["enabled"]
            for name in ("listen", "usef", "tfmap")
        ):
            inputs.append(
                TensorField(
                    "enroll_spectra",
                    "spectrum",
                    ("batch", "frequency", "frame", "complex"),
                )
            )
            inputs.append(TensorField("enroll_spectra_lengths", "label", ("batch",)))
        waveform_cue = self.joint_training and (
            any(self.features_config[name]["enabled"] for name in ("context", "spkemb"))
            or (self.features_config["tfmap"]["enabled"]
                and self.features_config["tfmap"]["type"] == "embedding")
        )
        if waveform_cue:
            inputs.extend(
                (
                    TensorField("enroll_wav", "waveform", ("batch", "time")),
                    TensorField("enroll_lengths", "label", ("batch",)),
                )
            )
        if not self.joint_training and self.features_config["spkemb"]["enabled"]:
            inputs.append(
                TensorField(
                    "spk_embedding",
                    "embedding",
                    ("batch", "embedding"),
                )
            )

        outputs = [
            TensorField(
                "target_estimate_spectra",
                "spectrum",
                ("batch", "frequency", "frame", "complex"),
            )
        ]
        if self.multi_task:
            outputs.append(
                TensorField("spk_logits", "logits", ("batch", "speaker_class"))
            )
        return ModelIOContract(
            tasks=frozenset({"tse"}),
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            forward_kind="estimate",
            frontend_owner="runtime",
        )

    def required_tse_input_fields(self) -> frozenset[str]:
        """返回当前 cue 配置在普通整句前向中必须提供的模型字段。"""
        return frozenset(field.name for field in self.IO_CONTRACT.inputs)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        """单次外层 forward 内按有效谱/波形长度分组，保持 DDP 调用一致。"""
        if "enroll_spectra_lengths" not in self.required_tse_input_fields():
            return self._forward_equal_length(batch)
        mix = batch.get("mix_spectra")
        spectrum = batch.get("enroll_spectra")
        if not isinstance(mix, torch.Tensor) or mix.ndim != 4 or mix.shape[-1] != 2:
            raise ValueError("mix_spectra must be [B,F,T,2]")
        if mix.shape[0] < 1 or mix.shape[-2] < 1:
            raise ValueError("mix_spectra requires B>=1 and T>=1")
        if not isinstance(spectrum, torch.Tensor) or spectrum.ndim != 4:
            raise ValueError("spectral cue 的 enroll_spectra 应为 [B,F,T,2]")
        rows = mix.shape[0]
        if spectrum.shape[:2] != mix.shape[:2] or spectrum.shape[-1] != 2:
            raise ValueError("enroll_spectra must match mix_spectra batch and frequency")
        if spectrum.device != mix.device or spectrum.dtype != mix.dtype:
            raise ValueError("enroll_spectra must use the same device and dtype as mix_spectra")
        if not bool(torch.isfinite(spectrum).all().item()):
            raise ValueError("enroll_spectra cannot contain NaN/Inf")
        sizes = batch.get("enroll_spectra_lengths")
        if not isinstance(sizes, torch.Tensor) or sizes.shape != (rows,):
            raise ValueError(f"enroll_spectra_lengths must have shape [{rows}]")
        if sizes.dtype != torch.int32:
            raise TypeError("enroll_spectra_lengths must use torch.int32")
        if sizes.device != mix.device:
            raise ValueError("enroll_spectra_lengths must use the same device as mix_spectra")
        if bool(((sizes < 1) | (sizes > spectrum.shape[-2])).any().item()):
            raise ValueError("enroll_spectra_lengths must be positive and within spectrum capacity")
        keys = sizes.detach().cpu().tolist()
        waveform_lengths = None
        if "enroll_lengths" in self.required_tse_input_fields():
            waveform = batch.get("enroll_wav")
            waveform_lengths = batch.get("enroll_lengths")
            if not isinstance(waveform, torch.Tensor) or waveform.ndim != 2 or waveform.shape[0] != rows:
                raise ValueError("enroll_wav must be [B,T]")
            if waveform.device != mix.device or waveform.dtype != mix.dtype or not bool(torch.isfinite(waveform).all().item()):
                raise ValueError("enroll_wav must be finite and use the same device and dtype as mix_spectra")
            if not isinstance(waveform_lengths, torch.Tensor) or waveform_lengths.shape != (rows,):
                raise ValueError("enroll_lengths must have one value per row")
            if waveform_lengths.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise TypeError("enroll_lengths must have integer dtype")
            if waveform_lengths.device != mix.device or bool(((waveform_lengths < 1) | (waveform_lengths > waveform.shape[-1])).any().item()):
                raise ValueError("enroll_lengths must use mixture device and be within waveform capacity")
            keys = list(zip(keys, waveform_lengths.detach().cpu().tolist()))
        groups = {}
        for row, key in enumerate(keys):
            groups.setdefault(key, []).append(row)
        outputs, indices = [], []
        for key, row_ids in groups.items():
            index = torch.tensor(row_ids, device=mix.device)
            group = dict(batch) if len(groups) == 1 else {
                name: value.index_select(0, index)
                if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == rows
                else value for name, value in batch.items()
            }
            frames = key[0] if isinstance(key, tuple) else key
            group["enroll_spectra"] = group["enroll_spectra"][..., :frames, :]
            if waveform_lengths is not None:
                samples = key[1]
                waveform = group["enroll_wav"][..., :samples]
                embedding_tfmap = self.features_config["tfmap"]["enabled"] and self.features_config["tfmap"]["type"] == "embedding"
                minimum = self.n_fft // 2 + 1 if embedding_tfmap else int(getattr(self.spk_ft.encoder.fbank, "window_size", 1))
                group["enroll_wav"] = torch.nn.functional.pad(waveform, (0, max(0, minimum - samples)))
            output = self._forward_equal_length(group)
            if len(groups) == 1:
                return {**batch, **{field.name: output[field.name] for field in self.IO_CONTRACT.outputs}}
            outputs.append(output)
            indices.append(index)
        order = torch.cat(indices).argsort()
        result = dict(batch)
        for field in self.IO_CONTRACT.outputs:
            result[field.name] = torch.cat([output[field.name] for output in outputs]).index_select(0, order)
        return result

    def _forward_equal_length(self, batch: dict) -> dict:
        """计算一个无谱尾 padding 的等长组；不新增模块或 checkpoint 参数。"""
        runtime_sample_rate = batch.get("sample_rate")
        if runtime_sample_rate is not None:
            if isinstance(runtime_sample_rate, torch.Tensor):
                if runtime_sample_rate.numel() != 1:
                    raise ValueError(
                        "TSE batch sample_rate must be scalar, got shape "
                        f"{tuple(runtime_sample_rate.shape)}"
                    )
                runtime_sample_rate = runtime_sample_rate.item()
            if isinstance(runtime_sample_rate, bool) or not isinstance(
                runtime_sample_rate, int
            ):
                raise TypeError(
                    "TSE batch sample_rate must be an integer scalar, got "
                    f"{type(runtime_sample_rate).__name__}"
                )
            if runtime_sample_rate != self.sample_rate:
                raise ValueError(
                    f"TSE batch sample rate {runtime_sample_rate} does not match "
                    f"model.sample_rate={self.sample_rate}"
                )

        mix_spec = batch["mix_spectra"]
        if not isinstance(mix_spec, torch.Tensor) or mix_spec.dim() != 4 or mix_spec.shape[-1] != 2:
            raise ValueError(f"mix_spectra 应为 [B, F, T, 2]，got {getattr(mix_spec, 'shape', None)}")
        B, F, T, _ = mix_spec.shape
        if B < 1 or T < 1:
            raise ValueError(f"mix_spectra requires B>=1 and T>=1, got {tuple(mix_spec.shape)}")
        if not mix_spec.is_floating_point() or not bool(torch.isfinite(mix_spec).all().item()):
            raise ValueError("mix_spectra must be finite floating-point values")
        if F != self.n_fft // 2 + 1:
            raise ValueError(
                f"mix_spectra 的 F={F} 与 n_fft={self.n_fft} 推导的 "
                f"{self.n_fft // 2 + 1} 不一致"
            )
        enroll_spec = batch.get("enroll_spectra")
        enroll_wav = batch.get("enroll_wav")
        enroll_lengths = batch.get("enroll_lengths")
        feats = self.features_config

        # [B, F, T, 2] → 复数谱 (B, F, T) 与实虚堆叠谱 (B, 2, F, T)（原实现内部格式）
        spec = torch.complex(mix_spec[..., 0], mix_spec[..., 1])
        spec_RI = mix_spec.permute(0, 3, 1, 2).contiguous()

        enroll_RI = (
            enroll_spec.permute(0, 3, 1, 2).contiguous()
            if any(feats[name]["enabled"] for name in ("listen", "usef", "tfmap"))
            else None
        )

        def _require_enroll_wav(cue: str):
            if not isinstance(enroll_wav, torch.Tensor):
                raise ValueError(
                    f"joint_training 模式下 {cue} cue 需要 batch 提供 enroll_wav"
                )
            if enroll_wav.ndim != 2 or enroll_wav.shape[0] != B or enroll_wav.shape[1] < 1:
                raise ValueError(f"enroll_wav requires [B={B}, T_e>=1], got {tuple(enroll_wav.shape)}")
            if enroll_wav.device != mix_spec.device or enroll_wav.dtype != mix_spec.dtype:
                raise ValueError("enroll_wav must use the same device and dtype as mix_spectra")
            if not enroll_wav.is_floating_point() or not bool(torch.isfinite(enroll_wav).all().item()):
                raise ValueError("enroll_wav must contain finite floating-point values")
            return enroll_wav

        def _require_enroll_lengths(cue: str):
            if not isinstance(enroll_lengths, torch.Tensor):
                raise ValueError(
                    f"joint_training 模式下 {cue} cue 需要 batch 提供 enroll_lengths"
                )
            integer_dtypes = {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            if enroll_lengths.dtype not in integer_dtypes:
                raise TypeError(
                    f"enroll_lengths must have integer dtype, got {enroll_lengths.dtype}"
                )
            if enroll_lengths.ndim != 1 or enroll_lengths.shape[0] != B:
                raise ValueError(
                    f"enroll_lengths must have shape [{B}], got "
                    f"{tuple(enroll_lengths.shape)}"
                )
            if enroll_lengths.device != mix_spec.device:
                raise ValueError(
                    "enroll_lengths must use the same device as mix_spectra"
                )
            if bool(
                ((enroll_lengths < 1) | (enroll_lengths > enroll_wav.shape[-1]))
                .any()
                .item()
            ):
                raise ValueError(
                    f"enroll_lengths must be in [1, {enroll_wav.shape[-1]}], "
                    f"got {enroll_lengths.tolist()}"
                )
            return enroll_lengths

        # ---- C0. listen：谱域前缀拼接（enroll 谱 + glue 零谱帧 → mix 谱前）----
        listen_offset = None
        if feats["listen"]["enabled"]:
            spec_RI, listen_offset = self.spk_ft.listen.compute(enroll_RI, spec_RI)
            spec = torch.complex(spec_RI[:, 0], spec_RI[:, 1])

        # ---- C1. usef：USEF 跨谱注意力，输出拼接为输入通道 ----
        if feats["usef"]["enabled"]:
            enroll_usef, mix_usef = self.spk_ft.usef.compute(enroll_RI, spec_RI)
            spec_RI = self.spk_ft.usef.post(mix_usef, enroll_usef)

        # ---- C2. tfmap：TF 注意力图拼为第 3 通道 ----
        tfmap_context = None
        if feats["tfmap"]["enabled"]:
            enroll_mag = torch.abs(torch.complex(enroll_RI[:, 0], enroll_RI[:, 1]))
            if feats["tfmap"]["type"] == "embedding":
                mix_wav = batch.get("mixture")
                if (
                    not isinstance(mix_wav, torch.Tensor)
                    or mix_wav.ndim != 2
                    or mix_wav.shape[0] != B
                ):
                    raise ValueError(
                        "tfmap type='embedding' requires mixture waveform [B,T]"
                    )
                wav = _require_enroll_wav("tfmap")
                enroll_tfmap, tfmap_context = self.spk_ft.tfmap.compute(
                    enroll_mag,
                    torch.abs(spec),
                    enroll_wav=wav,
                    mix_wav=mix_wav,
                )
            else:
                enroll_tfmap = self.spk_ft.tfmap.compute(
                    enroll_mag, torch.abs(spec)
                )
            spec_RI = self.spk_ft.tfmap.post(spec_RI, enroll_tfmap.unsqueeze(1))

        # ---- S2/S3. BandSplit + SubbandNorm ----
        subband_spec = self.sep_model.band_split(spec_RI)
        subband_mix_spec = self.sep_model.band_split(spec)
        subband_feature = self.sep_model.subband_norm(subband_spec)

        # ---- C3. context：帧级 speaker 特征 cross-attention 融合 ----
        if feats["context"]["enabled"]:
            if tfmap_context is not None:
                enroll_context = tfmap_context
                if not self.multi_fuse:
                    subband_feature = self.spk_ft.context.post(
                        subband_feature, enroll_context
                    )
            else:
                wav = _require_enroll_wav("context")
                wav_lengths = _require_enroll_lengths("context")
                subband_feature = self.spk_ft.context.compute_and_post(
                    subband_feature, wav, wav_lengths
                )

        # ---- C4. spkemb：句级 embedding 融合 ----
        spk_logits: torch.Tensor | None = None
        if feats["spkemb"]["enabled"]:
            if self.joint_training:
                wav = _require_enroll_wav("spkemb")
                wav_lengths = _require_enroll_lengths("spkemb")
                enroll_emb = self.spk_ft.spkemb.compute(
                    wav, enroll_lengths=wav_lengths
                )  # (B, F_e)
            else:
                enroll_emb = batch.get("spk_embedding")
                if enroll_emb is None:
                    raise ValueError(
                        "joint_training=False 时 spkemb cue 需要 batch 提供 "
                        "预提取的 spk_embedding [B, embed_dim]"
                    )
                exp_dim = feats["spkemb"]["embed_dim"]
                if not isinstance(enroll_emb, torch.Tensor):
                    raise TypeError(
                        "spk_embedding 必须是 torch.Tensor，got "
                        f"{type(enroll_emb).__name__}"
                    )
                if enroll_emb.dim() != 2:
                    raise ValueError(
                        f"spk_embedding 应为 [B, {exp_dim}]，got "
                        f"{enroll_emb.shape}"
                    )
                if enroll_emb.shape[0] != B:
                    raise ValueError(
                        "spk_embedding 行数必须与 mix_spectra 的 batch size 一致："
                        f"expected B={B}, got {enroll_emb.shape[0]}"
                    )
                if enroll_emb.shape[1] != exp_dim:
                    raise ValueError(
                        f"spk_embedding 应为 [B, {exp_dim}]，got "
                        f"{enroll_emb.shape}"
                    )
                if enroll_emb.device != mix_spec.device:
                    raise ValueError(
                        "spk_embedding device 必须与 mix_spectra 一致："
                        f"embedding={enroll_emb.device}, mix_spectra={mix_spec.device}"
                    )
                if not enroll_emb.is_floating_point():
                    raise TypeError(
                        "spk_embedding 必须使用 floating dtype，got "
                        f"{enroll_emb.dtype}"
                    )
                if enroll_emb.dtype != mix_spec.dtype:
                    raise TypeError(
                        "spk_embedding dtype 必须与 mix_spectra 一致："
                        f"embedding={enroll_emb.dtype}, "
                        f"mix_spectra={mix_spec.dtype}"
                    )
                if not bool(torch.isfinite(enroll_emb).all().item()):
                    raise ValueError("spk_embedding 不能包含 NaN/Inf")
            # multi-task：句级 embedding 接 speaker 分类头，logits 供
            # SpeakerCrossEntropyLoss 辅助任务（trainer 侧按 batch.spk_labels 接线）
            if self.multi_task:
                spk_logits = self.spk_classifier(enroll_emb)  # (B, spks_in_train)
            enroll_emb = enroll_emb.unsqueeze(1).unsqueeze(3)  # (B, 1, F_e, 1)
            if not self.multi_fuse:
                subband_feature = self.spk_ft.spkemb.post(
                    subband_feature, enroll_emb
                )

        # ---- S4/S5. 分离 + 复数掩码 ----
        if self.multi_fuse:
            if self.multi_fuse_feature == "spkemb":
                condition = enroll_emb
                condition_transform = None
            else:
                condition = enroll_context
                condition_transform = self.spk_ft.context.attend
            sep_output = self.sep_model.separator.forward_conditioned(
                subband_feature,
                self.multi_fuse_layers,
                condition,
                condition_transform,
            )
        else:
            sep_output = self.sep_model.separator(subband_feature)
        est_spec_RI = self.sep_model.band_masker(sep_output,
                                                 subband_mix_spec)  # (B, 2, 1, F, T)
        est = torch.complex(est_spec_RI[:, 0, 0], est_spec_RI[:, 1, 0])  # (B, F, T)

        # ---- C0 后处理：裁掉 listen 前缀帧 ----
        if listen_offset is not None:
            est = self.spk_ft.listen.post(est, listen_offset, mix_len=T)

        # 输出不就地写回入参 dict：``build_tse_model_inputs`` 的返回值同时承载
        # 输出字段，就地改写会让「输入」与「输出」混在同一可变对象里。
        outputs = {
            "target_estimate_spectra": torch.view_as_real(
                est.contiguous()),  # (B, F, T, 2)
        }
        if spk_logits is not None:
            outputs["spk_logits"] = spk_logits
        return {**batch, **outputs}

    def init_streaming_state(self, enrollment: Mapping[str, torch.Tensor]) -> TSEStreamingState:
        """初始化一个绑定 enrollment 的 causal 流式会话。"""
        if not self.causal:
            raise ValueError("TSE_BSRNN_SPK streaming requires causal=True")
        if not isinstance(enrollment, Mapping):
            raise TypeError("streaming enrollment must be a mapping")
        condition: dict[str, torch.Tensor] = {}
        enroll_spec = enrollment.get("enroll_spectra")
        enroll_wav = enrollment.get("enroll_wav")
        spk_embedding = enrollment.get("spk_embedding")
        batch_size = None
        parameter = next(self.parameters())
        if any(self.features_config[name]["enabled"] for name in ("listen", "usef", "tfmap")):
            if (
                not isinstance(enroll_spec, torch.Tensor)
                or enroll_spec.ndim != 4
                or enroll_spec.shape[-1] != 2
                or enroll_spec.shape[1] != self.n_fft // 2 + 1
                or enroll_spec.shape[0] < 1
                or enroll_spec.shape[2] < 1
            ):
                raise ValueError("streaming spectral cues require finite enroll_spectra [B,F,T>=1,2]")
            condition["enroll_spectra"] = enroll_spec
            batch_size = enroll_spec.shape[0]
        if any(self.features_config[name]["enabled"] for name in ("context", "spkemb")):
            if self.joint_training:
                if (
                    not isinstance(enroll_wav, torch.Tensor)
                    or enroll_wav.ndim != 2
                    or enroll_wav.shape[0] < 1
                    or enroll_wav.shape[1] < 1
                ):
                    raise ValueError("joint streaming cues require finite enroll_wav [B,T>=1]")
                condition["enroll_wav"] = enroll_wav
                batch_size = enroll_wav.shape[0] if batch_size is None else batch_size
            elif self.features_config["spkemb"]["enabled"]:
                exp_dim = self.features_config["spkemb"]["embed_dim"]
                if (
                    not isinstance(spk_embedding, torch.Tensor)
                    or spk_embedding.ndim != 2
                    or spk_embedding.shape[0] < 1
                    or spk_embedding.shape[1] != exp_dim
                ):
                    raise ValueError(
                        f"preextracted streaming spkemb requires [B, {exp_dim}]"
                    )
                condition["spk_embedding"] = spk_embedding
                batch_size = spk_embedding.shape[0] if batch_size is None else batch_size
        if batch_size is None:
            raw_batch_size = enrollment.get("batch_size")
            if isinstance(raw_batch_size, bool) or not isinstance(raw_batch_size, int) or raw_batch_size < 1:
                raise ValueError("streaming without cues requires positive integer enrollment.batch_size")
            batch_size = raw_batch_size
        for name, value in condition.items():
            if value.shape[0] != batch_size:
                raise ValueError(f"streaming enrollment batch mismatch at {name}")
            if value.device != parameter.device or value.dtype != parameter.dtype:
                raise ValueError(
                    f"streaming enrollment {name} must match model device/dtype "
                    f"{parameter.device}/{parameter.dtype}"
                )
            if not value.is_floating_point() or not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"streaming enrollment {name} must be finite floating-point")

        if self.features_config["context"]["enabled"]:
            condition["context"] = self.spk_ft.context.compute(condition["enroll_wav"])
        if self.features_config["spkemb"]["enabled"]:
            embedding = (
                self.spk_ft.spkemb.compute(condition["enroll_wav"])
                if self.joint_training
                else condition["spk_embedding"]
            )
            condition["spkemb"] = embedding
            if self.multi_task:
                condition["spk_logits"] = self.spk_classifier(embedding)

        hidden_size = self.sep_model.separator.feature_dim * 2
        rows = batch_size * self.sep_model.nband
        separator_states = tuple(
            (
                torch.zeros(1, rows, hidden_size, device=parameter.device, dtype=parameter.dtype),
                torch.zeros(1, rows, hidden_size, device=parameter.device, dtype=parameter.dtype),
            )
            for _ in self.sep_model.separator.separation
        )
        usef_history = None
        if self.features_config["usef"]["enabled"]:
            left = self.spk_ft.usef.time_left_padding
            usef_history = torch.zeros(
                batch_size,
                2,
                left,
                self.n_fft // 2 + 1,
                device=parameter.device,
                dtype=parameter.dtype,
            )
        state = TSEStreamingState(
            batch_size=batch_size,
            enrollment=condition,
            separator_states=separator_states,
            usef_history=usef_history,
        )
        if self.features_config["listen"]["enabled"]:
            enroll_ri = condition["enroll_spectra"].permute(0, 3, 1, 2).contiguous()
            glue = enroll_ri.new_zeros(
                *enroll_ri.shape[:-1], self.spk_ft.listen.glue_frames
            )
            prefix = torch.cat((enroll_ri, glue), dim=-1).permute(0, 2, 3, 1)
            _, state = self.forward_stream_chunk({"mix_spectra": prefix}, state)
        return state

    def reset_streaming_state(
        self,
        state: TSEStreamingState,
        enrollment: Mapping[str, torch.Tensor] | None = None,
    ) -> TSEStreamingState:
        """清空全部时序历史；省略 enrollment 时复用当前目标条件。"""
        source = state.enrollment if enrollment is None else enrollment
        if enrollment is None:
            source = {
                key: value
                for key, value in source.items()
                if key in {"enroll_spectra", "enroll_wav", "spk_embedding", "batch_size"}
            }
            if not source:
                source = {"batch_size": state.batch_size}
        return self.init_streaming_state(source)

    def _stream_usef(
        self,
        enroll_ri: torch.Tensor,
        mix_ri: torch.Tensor,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.spk_ft.usef
        mix = mix_ri.permute(0, 1, 3, 2).contiguous()
        mix = mix.permute(0, 2, 1, 3).contiguous()
        mix = feature.usef_con2v[0](mix).permute(0, 2, 1, 3).contiguous()
        merged = torch.cat((history, mix), dim=2)
        encoded_mix = feature.usef_con2v[1](merged)
        if feature.time_left_padding:
            next_history = merged[:, :, -feature.time_left_padding:].contiguous()
        else:
            next_history = merged[:, :, :0].contiguous()

        enroll = enroll_ri.permute(0, 1, 3, 2).contiguous()
        enroll = enroll.permute(0, 2, 1, 3).contiguous()
        enroll = feature.usef_con2v[0](enroll).permute(0, 2, 1, 3).contiguous()
        enroll = torch.nn.functional.pad(
            enroll, (0, 0, feature.time_left_padding, 0)
        )
        encoded_enroll = feature.usef_con2v[1](enroll)
        attended = feature.usef_att(encoded_mix, encoded_enroll)
        return (
            attended.permute(0, 1, 3, 2).contiguous(),
            encoded_mix.permute(0, 1, 3, 2).contiguous(),
            next_history,
        )

    def forward_stream_chunk(
        self,
        batch: dict,
        state: TSEStreamingState,
    ) -> tuple[dict, TSEStreamingState]:
        """处理连续 causal 谱 chunk，并携带固定 enrollment 与 LSTM state。"""
        if not isinstance(state, TSEStreamingState):
            raise TypeError("state must be TSEStreamingState")
        forbidden = {"enroll_spectra", "enroll_wav", "spk_embedding"} & set(batch)
        if forbidden:
            raise ValueError(
                "streaming enrollment is bound to state; use reset_streaming_state "
                f"before replacing {sorted(forbidden)}"
            )
        mix_spec = batch.get("mix_spectra")
        if not isinstance(mix_spec, torch.Tensor) or mix_spec.ndim != 4:
            raise ValueError("streaming mix_spectra must be [B,F,T,2]")
        if mix_spec.shape[0] != state.batch_size or mix_spec.shape[1] != self.n_fft // 2 + 1:
            raise ValueError("streaming mix_spectra batch/frequency mismatch")
        if mix_spec.shape[-1] != 2 or mix_spec.shape[2] < 1:
            raise ValueError("streaming mix_spectra requires T>=1 and real/imag size 2")
        parameter = next(self.parameters())
        if mix_spec.device != parameter.device or mix_spec.dtype != parameter.dtype:
            raise ValueError("streaming mix_spectra must match model device and dtype")
        if not mix_spec.is_floating_point() or not bool(torch.isfinite(mix_spec).all().item()):
            raise ValueError("streaming mix_spectra must contain finite floating-point values")
        if len(state.separator_states) != len(self.sep_model.separator.separation):
            raise ValueError("streaming separator state count does not match model")
        runtime_sample_rate = batch.get("sample_rate", self.sample_rate)
        if isinstance(runtime_sample_rate, torch.Tensor):
            if runtime_sample_rate.numel() != 1:
                raise ValueError(
                    "streaming sample_rate must be scalar, got shape "
                    f"{tuple(runtime_sample_rate.shape)}"
                )
            runtime_sample_rate = runtime_sample_rate.item()
        if isinstance(runtime_sample_rate, bool) or not isinstance(
            runtime_sample_rate, int
        ):
            raise TypeError(
                "streaming sample_rate must be an integer scalar, got "
                f"{type(runtime_sample_rate).__name__}"
            )
        if runtime_sample_rate != self.sample_rate:
            raise ValueError("streaming sample_rate does not match model.sample_rate")

        condition = state.enrollment
        spec = torch.view_as_complex(mix_spec.contiguous())
        spec_ri = mix_spec.permute(0, 3, 1, 2).contiguous()
        next_history = state.usef_history
        if self.features_config["usef"]["enabled"]:
            enroll_ri = condition["enroll_spectra"].permute(0, 3, 1, 2).contiguous()
            enroll_usef, mix_usef, next_history = self._stream_usef(
                enroll_ri, spec_ri, state.usef_history
            )
            spec_ri = self.spk_ft.usef.post(mix_usef, enroll_usef)
        if self.features_config["tfmap"]["enabled"]:
            enroll = condition["enroll_spectra"]
            enroll_mag = torch.abs(torch.view_as_complex(enroll.contiguous()))
            tfmap = self.spk_ft.tfmap.compute(enroll_mag, torch.abs(spec))
            spec_ri = self.spk_ft.tfmap.post(spec_ri, tfmap.unsqueeze(1))

        subband_mix = self.sep_model.band_split(spec)
        subband = self.sep_model.band_split(spec_ri)
        features = self.sep_model.subband_norm(subband)
        if self.features_config["context"]["enabled"]:
            features = self.spk_ft.context.post(features, condition["context"])
        if self.features_config["spkemb"]["enabled"]:
            embedding = condition["spkemb"].unsqueeze(1).unsqueeze(3)
            if not self.multi_fuse:
                features = self.spk_ft.spkemb.post(features, embedding)
        if self.multi_fuse:
            separated, next_separator = (
                self.sep_model.separator.forward_stream_conditioned(
                    features,
                    state.separator_states,
                    self.multi_fuse_layers,
                    embedding,
                )
            )
        else:
            separated, next_separator = self.sep_model.separator.forward_stream(
                features, state.separator_states
            )
        estimated_ri = self.sep_model.band_masker(separated, subband_mix)
        estimated = torch.complex(estimated_ri[:, 0, 0], estimated_ri[:, 1, 0])
        output = dict(batch)
        output["target_estimate_spectra"] = torch.view_as_real(estimated.contiguous())
        if "spk_logits" in condition:
            output["spk_logits"] = condition["spk_logits"]
        next_state = TSEStreamingState(
            batch_size=state.batch_size,
            enrollment=condition,
            separator_states=tuple(next_separator),
            usef_history=next_history,
        )
        return output, next_state

    def forward_stream_frame(
        self, batch: dict, state: TSEStreamingState
    ) -> tuple[dict, TSEStreamingState]:
        """单谱帧流式前向。"""
        mix_spec = batch.get("mix_spectra")
        if not isinstance(mix_spec, torch.Tensor) or mix_spec.shape[2] != 1:
            raise ValueError("forward_stream_frame expects one spectrum frame")
        return self.forward_stream_chunk(batch, state)

    # ------------------------------------------------------------------
    # 比赛 checkpoint 兼容加载
    # ------------------------------------------------------------------
    @classmethod
    def load_pretrained(
        cls,
        ckpt_path: str,
        strict: bool = True,
        map_location: str = "cpu",
        **model_kwargs,
    ) -> "TSE_BSRNN_SPK":
        """安全加载 wesep-real-tse checkpoint（如 ``spk_emb_100``）。

        默认 ``strict=True`` 是公共 loader 的 fail-closed 路径：所有模型键必须
        完整且不得包含额外键。第三方官方比赛 checkpoint 如确有训练头附加键或
        共享 encoder 重复注册键差异，调用方可显式传 ``strict=False``；该例外仍
        使用 ``safe_torch_load`` 安全模式，并在复制前强制 0-match 与 tensor
        shape 校验。
        """
        model = cls(**model_kwargs)
        ckpt = safe_torch_load(
            ckpt_path,
            map_location=map_location,
            purpose="load official REAL-TSE BSRNN model-only checkpoint",
        )
        state = ckpt
        if isinstance(ckpt, Mapping):
            for wrapper_key in ("state_dict", "model"):
                if wrapper_key in ckpt and isinstance(ckpt[wrapper_key], Mapping):
                    state = ckpt[wrapper_key]
                    break
            else:
                models = ckpt.get("models")
                if isinstance(models, list) and models:
                    state = models[0]
        if not isinstance(state, Mapping):
            raise ValueError(
                f"load_pretrained: ckpt {ckpt_path} 的权重主体必须是 mapping，"
                f"got {type(state).__name__}"
            )

        adapted: dict[str, torch.Tensor] = {}
        duplicate: list[str] = []
        invalid: list[str] = []
        for raw_key, value in state.items():
            if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
                invalid.append(repr(raw_key))
                continue
            key = raw_key[len("module."):] if raw_key.startswith("module.") else raw_key
            if key in adapted:
                duplicate.append(key)
            else:
                adapted[key] = value

        model_state = model.state_dict()
        model_keys = set(model_state)
        matched = model_keys & set(adapted)
        if not matched:
            raise ValueError(
                f"load_pretrained: ckpt {ckpt_path} 与模型 {len(model_keys)} 个"
                "权重键 0 匹配，疑似模型配置或 ckpt 不对应"
            )
        missing = sorted(model_keys - set(adapted))
        unexpected = sorted(set(adapted) - model_keys)
        shape_mismatch = sorted(
            key for key in matched if adapted[key].shape != model_state[key].shape
        )
        dtype_mismatch = sorted(
            key for key in matched if adapted[key].dtype != model_state[key].dtype
        )
        # strict=False 只兼容 REAL-TSE 官方训练头；绝不允许模型参数缺失，避免
        # 返回含随机初始化层的 extractor。严格模式必须拒绝所有额外键。
        official_ignored_prefixes = ("loss.", "criterion.", "metrics.")
        allowed_unexpected = (
            [key for key in unexpected if key.startswith(official_ignored_prefixes)]
            if not strict
            else []
        )
        rejected_unexpected = sorted(set(unexpected) - set(allowed_unexpected))
        if duplicate or invalid or shape_mismatch or dtype_mismatch or missing or rejected_unexpected:
            details = []
            if missing:
                details.append(f"缺失键={missing}")
            if rejected_unexpected:
                details.append(f"未知键={rejected_unexpected}")
            if duplicate:
                details.append(f"重复键={sorted(set(duplicate))}")
            if invalid:
                details.append(f"非 tensor/非法键={invalid}")
            if shape_mismatch:
                details.append("shape 不匹配=" + repr(shape_mismatch))
            if dtype_mismatch:
                details.append("dtype 不匹配=" + repr(dtype_mismatch))
            raise ValueError(
                f"load_pretrained: ckpt {ckpt_path} 兼容性校验失败：" + "; ".join(details)
            )

        loadable = {key: adapted[key] for key in model_keys}
        incompatible = model.load_state_dict(loadable, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "load_pretrained: internal strict load unexpectedly left incompatible keys: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        if allowed_unexpected:
            logger.info("load_pretrained 官方兼容模式忽略训练头键: %s", allowed_unexpected)
        logger.info(
            "load_pretrained 加载 %s（%d/%d 键匹配，strict=%s）",
            ckpt_path,
            len(matched),
            len(model_keys),
            strict,
        )
        return model
