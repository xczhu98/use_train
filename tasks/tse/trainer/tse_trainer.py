"""Target speaker extraction trainer for validated ``TSEBatch`` inputs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from core.losses.composite import CompositeLoss
from core.models.execution_plan import resolve_trainer_execution_plan
from core.utils.inference_artifacts import fingerprint_file
from core.trainer.metrics_trainer import (
    MetricsCapableTrainer,
    _direct_spectral_loss_kwargs,
    _waveform_loss_kwargs,
    sanitize_model_output,
)
from tasks.tse.cue_contract import (
    CUE_SLOTS,
    TSECueBatch,
    TSECueBatches,
    TSECueSpecs,
)
from tasks.tse.data.batch import TSEBatch
from tasks.tse.objectives import (
    resolve_dual_self_estimated_objective,
    validate_dual_objective_model,
    validate_tse_objective_config,
)
from tasks.tse.models.speaker.config import (
    config_bool,
    require_bool,
    resolve_external_speaker_checkpoint,
)
from tasks.tse.model_io import (
    build_tse_model_inputs,
    decode_tse_model_output,
    mask_waveform_tail,
    per_target_lengths,
    resolve_tse_model_input_fields,
)

logger = logging.getLogger(__name__)

_EXTERNAL_SPEAKER_IDENTITY_KEY = "tse_external_speaker_checkpoint_identity"
_SPEAKER_WAVEFORM_FIELDS = frozenset(
    {
        "enroll_spectra",
        "enroll_wav",
        "enroll_lengths",
        "speaker_waveform",
        "speaker_waveform_lengths",
    }
)
_SPEAKER_EMBEDDING_FIELDS = frozenset({"spk_embedding", "speaker_embedding"})
_SPEAKER_CUE_SOURCES = frozenset({"waveform", "embedding"})


def _speaker_cue_source_preference(trainer_cfg: Mapping | None) -> str | None:
    trainer_cfg = trainer_cfg or {}
    if not isinstance(trainer_cfg, Mapping):
        raise TypeError("trainer must be a mapping")
    source = trainer_cfg.get("speaker_cue_source")
    if source is None:
        return None
    if not isinstance(source, str) or source not in _SPEAKER_CUE_SOURCES:
        raise ValueError(
            "trainer.speaker_cue_source must be 'waveform' or 'embedding'"
        )
    return source


def resolve_tse_dataset_speaker_cue_source(
    model: Any,
    cue_specs: TSECueSpecs,
    trainer_cfg: Mapping | None,
    *,
    split: str,
) -> str | None:
    """按模型合同与 split cue specs 解析唯一 speaker cue source。"""
    if not isinstance(cue_specs, TSECueSpecs):
        raise TypeError(f"{split} dataset cue_specs must be TSECueSpecs")
    fields = resolve_tse_model_input_fields(model)
    requires_waveform = bool(fields & _SPEAKER_WAVEFORM_FIELDS)
    requires_embedding = bool(fields & _SPEAKER_EMBEDDING_FIELDS)
    if not requires_waveform and not requires_embedding:
        return None

    has_waveform = cue_specs.speaker_waveform is not None
    has_embedding = cue_specs.speaker_embedding is not None
    preference = _speaker_cue_source_preference(trainer_cfg)
    if has_waveform and has_embedding and preference is None:
        raise ValueError(
            f"{split} dataset enables both speaker waveform and embedding cues; "
            "set trainer.speaker_cue_source explicitly"
        )
    if preference == "waveform" and not has_waveform:
        raise ValueError(
            f"trainer.speaker_cue_source='waveform' but {split} dataset does not "
            "declare speaker_waveform"
        )
    if preference == "embedding" and not has_embedding:
        raise ValueError(
            f"trainer.speaker_cue_source='embedding' but {split} dataset does not "
            "declare speaker_embedding"
        )
    if requires_waveform and not has_waveform:
        raise ValueError(
            f"configured TSE model requires speaker waveform audio, but {split} "
            "dataset cue_specs does not declare speaker_waveform"
        )

    if requires_embedding:
        source = preference
        if source is None:
            if has_embedding:
                source = "embedding"
            elif has_waveform:
                source = "waveform"
            else:
                raise ValueError(
                    f"configured TSE model requires a speaker embedding, but {split} "
                    "dataset declares neither speaker_embedding nor speaker_waveform"
                )
        return source

    if preference not in (None, "waveform"):
        raise ValueError(
            f"configured TSE model requires waveform audio; {split} cannot select "
            "speaker embedding as its source"
        )
    return "waveform"


def _dataset_cue_is_guaranteed(dataset: Any, source: str) -> bool:
    guarantees = getattr(dataset, "cue_guarantees", None)
    slot = "speaker_waveform" if source == "waveform" else "speaker_embedding"
    if isinstance(guarantees, Mapping) and slot in guarantees:
        value = guarantees[slot]
        if type(value) is not bool:
            raise TypeError(f"dataset cue_guarantees[{slot!r}] must be bool")
        return value

    records = getattr(dataset, "records", None)
    if records is None:
        return True
    declared: list[bool] = []
    for record in records:
        for target in getattr(record, "targets", ()):
            cue_getter = getattr(target, "cue", None)
            if not callable(cue_getter):
                continue
            try:
                cue_record = cue_getter("audio")
            except (KeyError, ValueError):
                continue
            spec = getattr(cue_record, "spec", None)
            cue_type = getattr(spec, "cue_type", None)
            cue_format = getattr(spec, "cue_format", None)
            matches = (
                source == "waveform" and (cue_type, cue_format) == ("wav", "waveform")
            ) or (
                source == "embedding"
                and (cue_type, cue_format) == ("npy", "spk_embedding")
            )
            if matches:
                declared.append(getattr(spec, "guaranteed", True) is True)
    return all(declared) if declared else True


def validate_tse_training_dataset_cues(
    model: Any,
    dataset: Any,
    trainer_cfg: Mapping | None,
    *,
    split: str,
) -> str | None:
    """构造期拒绝模型合同与 dataset cue 声明不兼容的训练。"""
    cue_specs = getattr(dataset, "cue_specs", None)
    if cue_specs is None:
        return None
    source = resolve_tse_dataset_speaker_cue_source(
        model,
        cue_specs,
        trainer_cfg,
        split=split,
    )
    fields = resolve_tse_model_input_fields(model)
    required_sources = set()
    if fields & _SPEAKER_WAVEFORM_FIELDS:
        required_sources.add("waveform")
    if fields & _SPEAKER_EMBEDDING_FIELDS and source is not None:
        required_sources.add(source)
    for required_source in required_sources:
        if not _dataset_cue_is_guaranteed(dataset, required_source):
            raise ValueError(
                f"{split} dataset declares non-guaranteed real {required_source} "
                "speaker cues, but the configured model has no missing-cue presence input; "
                "only provenance-validated load-failure placeholders may omit the cue"
            )
    return source


@dataclass(frozen=True)
class _PreparedTSEValidationBatch:
    """已通过同步 prepare 门禁、可直接进入 TSE forward/loss 的 batch。"""

    batch: TSEBatch
    lengths_for_loss: torch.Tensor
    fs_val: int


def _unpack_tse_batch(batch: Any) -> TSEBatch:
    """只接受 canonical typed batch，旧 tuple 在 trainer 边界显式拒绝。"""
    if not isinstance(batch, TSEBatch):
        raise TypeError(
            "TSE trainer requires TSEBatch from tse_collate_fn; tuple batches are not accepted"
        )
    return batch


class TSETrainer(MetricsCapableTrainer):
    """只消费 validated ``TSEBatch`` 的目标说话人提取 trainer。

    typed cues 经 ``build_tse_model_inputs`` 按模型 I/O 合同装配；旧 speaker 模型
    仍获得原有 ``enroll_spectra`` / ``enroll_wav`` 字段，但 trainer 不再接受或
    构造 tuple enrollment 合同。
    """

    def __init__(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        sample_rate: int = 16000,
        stft_module: Any = None,
        **kw: Any,
    ):
        validate_tse_objective_config(kw.get("config", {}))
        super().__init__(
            train_loader=train_loader,
            val_loader=val_loader,
            sample_rate=sample_rate,
            stft_module=stft_module,
            **kw,
        )

        raw_model = self._raw_model()
        self.execution_plan = resolve_trainer_execution_plan(
            self.config,
            raw_model,
            task="tse",
            sample_rate=self.sample_rate,
            dataset_contract=getattr(self.train_dataset, "contract", None),
            stft_module=self.stft_module,
        )
        self._tse_model_input_fields = resolve_tse_model_input_fields(raw_model)
        self._tse_requires_speaker_waveform = bool(
            self._tse_model_input_fields & _SPEAKER_WAVEFORM_FIELDS
        )
        self._tse_requires_speaker_embedding = bool(
            self._tse_model_input_fields & _SPEAKER_EMBEDDING_FIELDS
        )
        trainer_cfg = (
            self.config.get("trainer", {}) if isinstance(self.config, dict) else {}
        )
        self._speaker_cue_source_preference = _speaker_cue_source_preference(
            trainer_cfg
        )

        # ── TSE speaker embedding 配置 ──
        # 非 joint + spkemb 时，外部 encoder 必须保持 trainer 的独立状态：
        # 它在首次 train/val 前向时懒构造，不注册进 TSE model，因此不会进入
        # compile/DDP/主 optimizer。具体输出校验在 _compute_spk_embedding 完成。
        raw_model_args = (
            self.config.get("model", {}).get("args", {})
            if isinstance(self.config, dict) else {}
        )
        self._tse_model_args = {} if raw_model_args is None else raw_model_args
        if not isinstance(self._tse_model_args, Mapping):
            raise TypeError("model.args must be a mapping")
        config_joint_training = config_bool(
            self._tse_model_args,
            "joint_training",
            True,
            "model.args",
        )
        configured_features = self._tse_model_args.get("features")
        if configured_features is None:
            configured_features = {}
        if not isinstance(configured_features, Mapping):
            raise TypeError("model.args.features must be a mapping")
        for feature_name, feature_cfg in configured_features.items():
            if not isinstance(feature_cfg, Mapping):
                raise TypeError(
                    f"model.args.features.{feature_name} must be a mapping"
                )
            config_bool(
                feature_cfg,
                "enabled",
                False,
                f"model.args.features.{feature_name}",
            )

        resolved_joint_training = getattr(raw_model, "joint_training", None)
        if resolved_joint_training is not None:
            resolved_joint_training = require_bool(
                resolved_joint_training,
                "model.joint_training",
            )
            if resolved_joint_training != config_joint_training:
                raise ValueError(
                    "model.joint_training 与 model.args.joint_training 不一致"
                )
        self._tse_joint_training = (
            config_joint_training
            if resolved_joint_training is None
            else resolved_joint_training
        )
        resolved_features = getattr(raw_model, "features_config", None)
        if resolved_features is None:
            resolved_features = configured_features
        if not isinstance(resolved_features, Mapping):
            raise TypeError("model.features_config must be a mapping")
        resolved_spkemb = resolved_features.get("spkemb")
        if resolved_spkemb is None:
            resolved_spkemb = {}
        if not isinstance(resolved_spkemb, Mapping):
            raise TypeError("model.args.features.spkemb must be a mapping")
        self._tse_spkemb_enabled = config_bool(
            resolved_spkemb,
            "enabled",
            False,
            "model.args.features.spkemb",
        )
        train_dataset = self.train_dataset or getattr(self.train_loader, "dataset", None)
        val_dataset = getattr(self.val_loader, "dataset", None)
        declared_sources = set()
        source_plan: dict[str, str | None] = {}
        for split, dataset in (("train", train_dataset), ("validation", val_dataset)):
            if dataset is None or getattr(dataset, "cue_specs", None) is None:
                continue
            source = validate_tse_training_dataset_cues(
                raw_model,
                dataset,
                trainer_cfg,
                split=split,
            )
            source_plan[split] = source
            if source is not None:
                declared_sources.add(source)
        if self._tse_requires_speaker_embedding and not declared_sources:
            declared_sources.add(self._speaker_cue_source_preference or "waveform")
        fallback_source = next(iter(declared_sources)) if len(declared_sources) == 1 else None
        for split in ("train", "validation"):
            source_plan.setdefault(split, fallback_source)
        self._speaker_cue_source_plan = source_plan
        self._declared_speaker_cue_sources = frozenset(declared_sources)
        self._uses_external_spk_encoder = (
            self._tse_requires_speaker_embedding
            and "waveform" in self._declared_speaker_cue_sources
        )

        self._spk_encoder = None
        self._spk_encoder_cfg: dict[str, Any] | None = None
        self._spk_embedding_dim: int | None = None
        self._loaded_spk_checkpoint_identity: dict[str, Any] | None = None
        self._spk_checkpoint_identity_synced = False
        if self._uses_external_spk_encoder:
            self._configure_external_spk_encoder(raw_model)
        self._validate_resumed_external_spk_checkpoint_identity()

        # ── SSA（Self-estimated Speech Augmentation）配置 ──
        # SSA 只消费 trainer.ssa_prob，缺省关闭。
        _tc = trainer_cfg
        _raw_ssa_prob = _tc.get("ssa_prob", 0.0)
        if isinstance(_raw_ssa_prob, bool) or not isinstance(
            _raw_ssa_prob, (int, float)
        ):
            raise TypeError(
                f"ssa_prob must be a number in [0, 1], got {_raw_ssa_prob!r}"
            )
        self._ssa_prob = float(_raw_ssa_prob)
        if not 0.0 <= self._ssa_prob <= 1.0:
            raise ValueError(f"ssa_prob must be in [0, 1], got {self._ssa_prob}")
        if self._ssa_prob > 0.0 and not self._tse_requires_speaker_waveform:
            raise ValueError(
                "trainer.ssa_prob requires a model that consumes speaker waveform cues"
            )
        # SSA 只在 joint_training=True 时替换 typed speaker waveform cue；
        # 预提取模式下 _maybe_ssa_cues 会 warning 一次后恒跳过。
        self._ssa_joint_training = self._tse_joint_training
        self._ssa_skip_warned = False

        dual_weights = resolve_dual_self_estimated_objective(trainer_cfg)
        validate_dual_objective_model(raw_model, dual_weights)
        self._dual_self_estimated_loss_weights = dual_weights
        if dual_weights is not None:
            if not self._tse_joint_training:
                raise ValueError(
                    "dual self-estimated objective requires joint_training=true"
                )
            if not self._tse_requires_speaker_waveform:
                raise ValueError(
                    "dual self-estimated objective requires a model that consumes "
                    "speaker waveform cues"
                )
            if self._ssa_prob > 0.0:
                raise ValueError(
                    "dual self-estimated objective cannot be combined with trainer.ssa_prob"
                )

    def _configure_external_spk_encoder(self, raw_model: torch.nn.Module) -> None:
        """解析非 joint ``spkemb`` 的独立 SpeakerEncoder 配置。

        该 encoder 只保存在 trainer 实例上，而不是 TSE 模型的子模块；因此
        BaseTrainer 后续的 compile/DDP 包装和主 optimizer 都不会触碰它。这里
        只做配置解析，实际构造延迟到首个 train/val batch，便于未使用该 cue
        的运行不加载额外模型。
        """
        model_config = getattr(getattr(raw_model, "spk_ft", None), "config", {})
        if not isinstance(model_config, dict):
            model_config = {}
        speaker_model = model_config.get("speaker_model")
        if not isinstance(speaker_model, dict):
            speaker_model = self._tse_model_args.get("speaker_model") or {}
        if not isinstance(speaker_model, dict):
            raise TypeError("model.args.speaker_model must be a mapping")

        # pretrained 的权威来源必须是原始 YAML，而不是模型构造后可能被
        # deep_update/defaults 改写的 resolved config；统一 helper 解析仓库根相对
        # 路径并要求文件存在，避免随机 SpeakerEncoder 进入训练。
        pretrained_path = resolve_external_speaker_checkpoint(self._tse_model_args)
        if pretrained_path is None:
            raise RuntimeError(
                "external SpeakerEncoder is required for non-joint spkemb mode"
            )
        speaker_encoder = speaker_model.get("speaker_encoder")
        if not isinstance(speaker_encoder, dict):
            speaker_encoder = {}
        features = getattr(raw_model, "features_config", None)
        if not isinstance(features, dict):
            features = self._tse_model_args.get("features") or {}
        spkemb = features.get("spkemb") or {}
        if not isinstance(spkemb, dict):
            raise TypeError("model.args.features.spkemb must be a mapping")
        spk_args = speaker_encoder.get("spk_args") or {}
        if not isinstance(spk_args, dict):
            raise TypeError(
                "model.args.speaker_model.speaker_encoder.spk_args must be a mapping"
            )
        embed_dim = spkemb.get("embed_dim", spk_args.get("embed_dim", 192))
        if isinstance(embed_dim, bool) or not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(
                "features.spkemb.embed_dim must be a positive integer, "
                f"got {embed_dim!r}"
            )
        fbank_args = speaker_model.get("fbank") or {}
        if not isinstance(fbank_args, dict):
            raise TypeError("model.args.speaker_model.fbank must be a mapping")

        self._spk_embedding_dim = int(embed_dim)
        self._spk_encoder_cfg = {
            "model": speaker_encoder.get("model", "ECAPA_TDNN_GLOB_c512"),
            # SpeakerEncoder 内部按 Path 读取；统一传仓库根解析后的绝对路径，
            # 避免训练进程改变 CWD 后相对 YAML 路径失效。
            "pretrained": str(pretrained_path),
            "pretrained_prefix": speaker_encoder.get("pretrained_prefix"),
            "spk_args": dict(spk_args),
            "fbank_args": dict(fbank_args),
        }

    def _get_external_spk_encoder(self) -> torch.nn.Module:
        """懒构造并冻结非 joint ``spkemb`` 使用的独立 SpeakerEncoder。"""
        if self._spk_encoder is not None:
            return self._spk_encoder
        if self._spk_encoder_cfg is None:
            raise RuntimeError("external SpeakerEncoder configuration was not initialized")

        from tasks.tse.models.speaker.encoder import SpeakerEncoder

        cfg = self._spk_encoder_cfg
        checkpoint_path = cfg["pretrained"]
        identity_before = fingerprint_file(checkpoint_path)
        encoder_kwargs = {
            "model": cfg["model"],
            "pretrained": checkpoint_path,
            # 外部 encoder 永远只作固定特征提取器，不能接受 YAML 的解冻请求。
            "freeze": True,
            "spk_args": cfg["spk_args"],
            "fbank_args": cfg["fbank_args"],
        }
        pretrained_prefix = cfg.get("pretrained_prefix")
        if pretrained_prefix is not None:
            encoder_kwargs["pretrained_prefix"] = pretrained_prefix
        encoder = SpeakerEncoder(**encoder_kwargs)
        identity_after = fingerprint_file(checkpoint_path)
        if identity_before != identity_after:
            raise RuntimeError(
                "SpeakerEncoder checkpoint 在 loader 执行期间发生变化: "
                f"{checkpoint_path}"
            )
        encoder_embed_dim = getattr(encoder, "embed_dim", None)
        if encoder_embed_dim is not None and (
            isinstance(encoder_embed_dim, bool)
            or not isinstance(encoder_embed_dim, int)
            or encoder_embed_dim != self._spk_embedding_dim
        ):
            raise ValueError(
                "SpeakerEncoder embed_dim mismatch: "
                f"expected {self._spk_embedding_dim}, got {encoder_embed_dim!r}"
            )
        encoder.to(self.device)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        self._spk_encoder = encoder
        self._loaded_spk_checkpoint_identity = identity_after
        return encoder

    def _sync_external_spk_checkpoint_identity(self) -> None:
        """首次成功加载后校验所有 DDP rank 的 checkpoint 内容一致。"""
        if getattr(self, "_spk_checkpoint_identity_synced", False):
            return
        identity = getattr(self, "_loaded_spk_checkpoint_identity", None)
        if identity is None:
            raise RuntimeError("SpeakerEncoder 已加载但缺少 checkpoint identity")

        identities: list[dict[str, Any] | None] = [None] * self.world_size
        with torch.inference_mode(False):
            dist.all_gather_object(identities, identity)
        content_identities = {
            (item["size"], item["sha256"])
            for item in identities
            if isinstance(item, dict)
        }
        if len(content_identities) != 1 or any(item is None for item in identities):
            details = "; ".join(
                f"rank {rank}: {item!r}" for rank, item in enumerate(identities)
            )
            raise RuntimeError(
                "TSE external SpeakerEncoder checkpoint identity mismatch across "
                f"DDP ranks: {details}"
            )
        self._spk_checkpoint_identity_synced = True

    @staticmethod
    def _same_checkpoint_content(left: Mapping, right: Mapping) -> bool:
        """只按不可变内容身份比较，允许 checkpoint 在机器间迁移路径。"""
        return (
            left.get("size") == right.get("size")
            and left.get("sha256") == right.get("sha256")
        )

    def _external_spk_checkpoint_identity_for_state(self) -> dict | None:
        """返回保存时绑定的外部 SpeakerEncoder 内容身份。"""
        if not getattr(self, "_uses_external_spk_encoder", False):
            return None
        cfg = getattr(self, "_spk_encoder_cfg", None)
        if not isinstance(cfg, Mapping):
            raise RuntimeError("external SpeakerEncoder configuration is missing")
        current = fingerprint_file(cfg["pretrained"])
        loaded = getattr(self, "_loaded_spk_checkpoint_identity", None)
        if loaded is not None and not self._same_checkpoint_content(loaded, current):
            raise RuntimeError(
                "external SpeakerEncoder checkpoint changed after it was loaded"
            )
        return dict(loaded if loaded is not None else current)

    def _build_state(
        self,
        epoch: int,
        *,
        rng_states_by_rank: list[dict] | None = None,
        **extra,
    ) -> dict:
        """在通用训练状态中持久化外部 SpeakerEncoder artifact identity。"""
        if _EXTERNAL_SPEAKER_IDENTITY_KEY in extra:
            raise TypeError(
                f"{_EXTERNAL_SPEAKER_IDENTITY_KEY} is managed by TSETrainer"
            )
        extra[_EXTERNAL_SPEAKER_IDENTITY_KEY] = (
            self._external_spk_checkpoint_identity_for_state()
        )
        return super()._build_state(
            epoch,
            rng_states_by_rank=rng_states_by_rank,
            **extra,
        )

    def _after_resume_checkpoint(self, ckpt: dict) -> None:
        """暂存 resume checkpoint 绑定的 speaker identity，待 TSE 初始化后校验。"""
        super()._after_resume_checkpoint(ckpt)
        self._resumed_external_spk_identity_present = (
            _EXTERNAL_SPEAKER_IDENTITY_KEY in ckpt
        )
        self._resumed_external_spk_identity = ckpt.get(
            _EXTERNAL_SPEAKER_IDENTITY_KEY
        )

    def _validate_resumed_external_spk_checkpoint_identity(self) -> None:
        """在首次前向前校验 resume 与当前外部 SpeakerEncoder 内容一致。"""
        if not getattr(self, "_resumed_external_spk_identity_present", False):
            # 没有执行 resume 时不做检查；执行过 resume 但 checkpoint 缺字段，
            # external 模式必须 fail-closed。
            if not hasattr(self, "_resumed_external_spk_identity"):
                return
        expected = getattr(self, "_resumed_external_spk_identity", None)
        uses_external = self._uses_external_spk_encoder
        if uses_external:
            if not isinstance(expected, Mapping):
                raise RuntimeError(
                    "resume checkpoint is missing external SpeakerEncoder artifact identity"
                )
            current = self._external_spk_checkpoint_identity_for_state()
            if current is None or not self._same_checkpoint_content(expected, current):
                raise RuntimeError(
                    "resume external SpeakerEncoder checkpoint identity mismatch: "
                    f"checkpoint={dict(expected)!r}, current={current!r}"
                )
        elif expected is not None:
            raise RuntimeError(
                "resume checkpoint requires an external SpeakerEncoder, but the "
                "current TSE configuration does not use one"
            )

    @staticmethod
    def _required_speaker_waveform(
        cues: TSECueBatches,
        *,
        context: str,
    ) -> TSECueBatch:
        cue = cues.speaker_waveform
        if cue is None:
            raise ValueError(f"{context} requires the typed speaker_waveform cue")
        if cue.lengths is None:
            raise ValueError(f"{context} requires speaker_waveform lengths")
        if not bool(cue.present.all().item()):
            raise ValueError(
                f"{context} requires speaker_waveform to be present for every target row"
            )
        return cue

    @staticmethod
    def _target_sample_indices(batch: TSEBatch) -> torch.Tensor:
        sample_indices = torch.arange(
            batch.mixture.shape[0], device=batch.num_targets.device
        )
        return torch.repeat_interleave(
            sample_indices, batch.num_targets.long(), output_size=batch.target.shape[0],
        )

    def _validate_placeholder_missing_rows(
        self,
        batch: TSEBatch,
        missing: torch.Tensor,
        *,
        cue_name: str,
    ) -> None:
        if tuple(missing.shape) != (batch.target.shape[0],):
            raise ValueError(f"{cue_name} missing mask must align with target rows")
        target_sample_indices = self._target_sample_indices(batch)
        samples = batch.mixture.shape[0]
        declared = torch.tensor([
            isinstance(value, Mapping) and value.get("placeholder") is True
            for value in batch.provenance
        ], dtype=torch.bool, device=batch.mixture.device)
        invalid_target = (batch.target != 0).any(dim=1) | (batch.spk_labels != -1)
        bad_targets = torch.zeros(samples, dtype=torch.int64, device=missing.device)
        bad_targets.scatter_add_(0, target_sample_indices, invalid_target.long())
        missing_samples = torch.zeros_like(bad_targets)
        missing_samples.scatter_add_(0, target_sample_indices, missing.long())
        invalid_sample = ~declared | (batch.mixture != 0).flatten(1).any(dim=1) | (bad_targets > 0)
        if bool(((missing_samples > 0) & invalid_sample).any().item()):
            raise ValueError(
                f"missing {cue_name} is only valid for a provenance-declared "
                "placeholder with zero mixture/targets and spk_label=-1"
            )

    def _speaker_waveform_execution_cues(
        self,
        batch: TSEBatch,
        cues: TSECueBatches,
        *,
        context: str,
    ) -> TSECueBatches:
        cue = cues.speaker_waveform
        if cue is None:
            raise ValueError(f"{context} requires the typed speaker_waveform cue")
        if cue.lengths is None:
            raise ValueError(f"{context} requires speaker_waveform lengths")
        missing = ~cue.present
        self._validate_placeholder_missing_rows(
            batch,
            missing,
            cue_name="speaker_waveform",
        )
        if not bool(missing.any().item()):
            return cues
        value = cue.value.clone()
        value[missing] = 0
        present = cue.present.clone()
        present[missing] = True
        lengths = cue.lengths.clone()
        lengths[missing] = int(value.shape[-1])
        return replace(
            cues,
            speaker_waveform=replace(
                cue,
                value=value,
                present=present,
                lengths=lengths,
            ),
        )

    def _speaker_embedding_execution_cues(
        self,
        batch: TSEBatch,
        cues: TSECueBatches,
    ) -> TSECueBatches:
        cue = cues.speaker_embedding
        if cue is None:
            raise ValueError("configured TSE model requires speaker_embedding")
        missing = ~cue.present
        self._validate_placeholder_missing_rows(
            batch,
            missing,
            cue_name="speaker_embedding",
        )
        if not bool(missing.any().item()):
            return cues
        value = cue.value.clone()
        value[missing] = 0
        present = cue.present.clone()
        present[missing] = True
        return replace(
            cues,
            speaker_embedding=replace(cue, value=value, present=present),
        )

    def _prepare_speaker_conditioning_local(
        self,
        batch: TSEBatch,
        cues: TSECueBatches,
        *,
        source: str | None,
        split: str,
    ) -> tuple[TSECueBatches, torch.Tensor | None]:
        runtime_source = resolve_tse_dataset_speaker_cue_source(
            self._raw_model(),
            batch.cue_specs,
            self.config.get("trainer", {}),
            split=f"{split} runtime batch",
        )
        if runtime_source != source:
            raise ValueError(
                f"{split} speaker cue source changed after construction: "
                f"planned={source!r}, runtime={runtime_source!r}"
            )
        resolved_cues = cues
        needs_waveform_view = self._tse_requires_speaker_waveform or (
            self._tse_requires_speaker_embedding and source == "waveform"
        )
        if needs_waveform_view:
            resolved_cues = self._speaker_waveform_execution_cues(
                batch,
                resolved_cues,
                context="configured TSE speaker conditioning",
            )

        spk_embedding = None
        if self._tse_requires_speaker_embedding:
            if source == "waveform":
                waveform = self._required_speaker_waveform(
                    resolved_cues,
                    context="external SpeakerEncoder",
                )
                spk_embedding = self._compute_spk_embedding_local(
                    waveform,
                    num_targets=batch.num_targets,
                    mixture_rows=batch.mixture.shape[0],
                )
            elif source == "embedding":
                resolved_cues = self._speaker_embedding_execution_cues(
                    batch,
                    resolved_cues,
                )
                embedding_cue = resolved_cues.speaker_embedding
                assert embedding_cue is not None
                if "spk_embedding" in self._tse_model_input_fields:
                    spk_embedding = embedding_cue.value
                    resolved_cues = replace(
                        resolved_cues,
                        speaker_embedding=None,
                    )
            else:
                raise ValueError("speaker embedding source was not resolved")

        if not self._tse_requires_speaker_waveform:
            resolved_cues = replace(resolved_cues, speaker_waveform=None)
        if "speaker_embedding" not in self._tse_model_input_fields:
            resolved_cues = replace(resolved_cues, speaker_embedding=None)
        return resolved_cues, spk_embedding

    def _run_synced_speaker_precondition(
        self,
        operation,
        *,
        phase: str,
    ):
        distributed = (
            getattr(self, "world_size", 1) > 1
            and dist.is_available()
            and dist.is_initialized()
        )
        if not distributed:
            return operation()

        local_error: Exception | None = None
        result = None
        try:
            result = operation()
        except Exception as exc:
            local_error = exc
        all_ok = self._allreduce_bool_and(local_error is None, phase=phase)
        if not all_ok:
            local_error_text = (
                f"rank {self.rank}: {type(local_error).__name__}: {local_error}"
                if local_error is not None
                else None
            )
            error_texts: list[str | None] = [None] * self.world_size
            with torch.inference_mode(False):
                dist.all_gather_object(error_texts, local_error_text)
            failures = [text for text in error_texts if text is not None]
            message = "TSE speaker cue precondition failed on DDP rank(s): " + "; ".join(
                failures
            )
            raise RuntimeError(message) from local_error
        if local_error is not None:
            raise local_error
        return result

    def _prepare_speaker_conditioning(
        self,
        batch: TSEBatch,
        cues: TSECueBatches,
        *,
        phase: str,
        split: str | None = None,
    ) -> tuple[TSECueBatches, torch.Tensor | None]:
        if split is None:
            split = "validation" if "validation" in phase else "train"
        source_plan = getattr(self, "_speaker_cue_source_plan", {})
        if split in source_plan:
            source = source_plan[split]
        elif self._tse_requires_speaker_embedding:
            declared = getattr(self, "_declared_speaker_cue_sources", frozenset())
            source = next(iter(declared)) if len(declared) == 1 else None
        elif self._tse_requires_speaker_waveform:
            source = "waveform"
        else:
            source = None
        result = self._run_synced_speaker_precondition(
            lambda: self._prepare_speaker_conditioning_local(
                batch,
                cues,
                source=source,
                split=split,
            ),
            phase=phase,
        )
        uses_external_this_split = (
            self._tse_requires_speaker_embedding and source == "waveform"
        )
        if (
            uses_external_this_split
            and self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
            and not self._spk_checkpoint_identity_synced
        ):
            self._sync_external_spk_checkpoint_identity()
        return result

    def _compute_spk_embedding(
        self,
        speaker_waveform: TSECueBatch,
        *,
        num_targets: torch.Tensor | None = None,
        mixture_rows: int | None = None,
    ) -> torch.Tensor | None:
        """从 typed speaker waveform cue 生成外部句级 embedding。"""
        embedding = self._run_synced_speaker_precondition(
            lambda: self._compute_spk_embedding_local(
                speaker_waveform,
                num_targets=num_targets,
                mixture_rows=mixture_rows,
            ),
            phase="tse_external_spk_embedding",
        )
        uses_external = getattr(self, "_uses_external_spk_encoder", None)
        if uses_external is None:
            uses_external = (
                not getattr(self, "_tse_joint_training", True)
                and getattr(self, "_tse_spkemb_enabled", False)
            )
        if (
            uses_external
            and getattr(self, "world_size", 1) > 1
            and dist.is_available()
            and dist.is_initialized()
            and not getattr(self, "_spk_checkpoint_identity_synced", False)
        ):
            self._sync_external_spk_checkpoint_identity()
        return embedding

    def _compute_spk_embedding_local(
        self,
        speaker_waveform: TSECueBatch,
        *,
        num_targets: torch.Tensor | None = None,
        mixture_rows: int | None = None,
    ) -> torch.Tensor | None:
        """按 typed cue 的有效长度分组执行冻结 SpeakerEncoder。"""
        uses_external = getattr(self, "_uses_external_spk_encoder", None)
        if uses_external is None:
            uses_external = (
                not getattr(self, "_tse_joint_training", True)
                and getattr(self, "_tse_spkemb_enabled", False)
            )
        if not uses_external:
            return None
        if not isinstance(speaker_waveform, TSECueBatch):
            raise TypeError("speaker_waveform must be a TSECueBatch")
        if speaker_waveform.representation != "waveform":
            raise ValueError("speaker_waveform cue must use waveform representation")
        if speaker_waveform.lengths is None:
            raise ValueError("speaker_waveform cue requires lengths")
        if not bool(speaker_waveform.present.all().item()):
            raise ValueError(
                "external SpeakerEncoder requires speaker_waveform for every target row"
            )
        enroll = speaker_waveform.value
        enroll_lengths = speaker_waveform.lengths
        if enroll.dim() != 2:
            raise ValueError(
                "speaker_waveform.value must be [rows, T], got "
                f"{tuple(enroll.shape)}"
            )

        integer_dtypes = (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        )
        if mixture_rows is not None:
            if (
                isinstance(mixture_rows, bool)
                or not isinstance(mixture_rows, int)
                or mixture_rows < 1
            ):
                raise ValueError(
                    f"mixture_rows must be a positive integer, got {mixture_rows!r}"
                )

        expected_rows = enroll.shape[0]
        if num_targets is not None:
            if not isinstance(num_targets, torch.Tensor):
                raise TypeError(
                    f"num_targets must be a torch.Tensor, got {type(num_targets).__name__}"
                )
            if num_targets.dtype not in integer_dtypes:
                raise TypeError(
                    f"num_targets must have integer dtype, got {num_targets.dtype}"
                )
            if num_targets.dim() != 1:
                raise ValueError(
                    f"num_targets must be [B], got {tuple(num_targets.shape)}"
                )
            if mixture_rows is not None and num_targets.shape[0] != mixture_rows:
                raise ValueError(
                    f"num_targets rows={num_targets.shape[0]} do not match "
                    f"mixture_rows={mixture_rows}"
                )
            if bool((num_targets < 1).any().item()):
                raise ValueError(
                    f"num_targets must be >= 1, got {num_targets.tolist()}"
                )
            expected_rows = int(num_targets.long().sum().item())
        elif mixture_rows is not None:
            expected_rows = mixture_rows

        if enroll.shape[0] != expected_rows:
            expected_desc = (
                f"sum(num_targets)={expected_rows}"
                if num_targets is not None
                else f"mixture_rows={expected_rows}"
            )
            raise ValueError(
                f"enroll rows={enroll.shape[0]} do not match {expected_desc}"
            )

        if enroll_lengths is not None:
            if not isinstance(enroll_lengths, torch.Tensor):
                raise TypeError(
                    "enroll_lengths must be a torch.Tensor for speaker embedding, "
                    f"got {type(enroll_lengths).__name__}"
                )
            if enroll_lengths.dtype not in integer_dtypes:
                raise TypeError(
                    f"enroll_lengths must have integer dtype, got {enroll_lengths.dtype}"
                )
            if enroll_lengths.dim() != 1 or enroll_lengths.shape[0] != enroll.shape[0]:
                raise ValueError(
                    f"enroll_lengths must have {enroll.shape[0]} rows, got "
                    f"{tuple(enroll_lengths.shape)}"
                )
            if bool(
                ((enroll_lengths < 1) | (enroll_lengths > enroll.shape[-1])).any().item()
            ):
                raise ValueError(
                    f"enroll_lengths must be in [1, {enroll.shape[-1]}], got "
                    f"{enroll_lengths.tolist()}"
                )

        if enroll.device != self.device:
            raise ValueError(
                "TSE enroll device mismatch: "
                f"enroll={enroll.device}, trainer={self.device}"
            )
        encoder = self._get_external_spk_encoder()
        # SpeakerEncoder 的 fbank 使用 snip_edges=True，短于单帧窗口会直接报错。
        # 每行必须先裁到真实有效长度，避免 ECAPA 的统计池化依赖同 batch 最长
        # enrollment；只有短于单帧窗口时才在右侧补到该最小长度。
        fbank = getattr(encoder, "fbank", None)
        window_size = getattr(fbank, "window_size", None)
        if window_size is not None and (
            isinstance(window_size, bool)
            or not isinstance(window_size, int)
            or window_size <= 0
        ):
            raise ValueError(
                "SpeakerEncoder fbank.window_size must be a positive integer, "
                f"got {window_size!r}"
            )

        # encoder 是冻结的独立特征提取器；每次使用前保持 eval，并显式关闭
        # autocast，避免 AMP 改变 fbank/ECAPA 的句级 embedding 数值。
        # 使用 no_grad 而非 inference_mode，避免 detached embedding 作为主模型
        # 输入时携带 inference tensor 的版本限制。
        encoder.eval()
        if enroll_lengths is None:
            enroll_for_encoder = enroll
            if window_size is not None and enroll.shape[-1] < window_size:
                enroll_for_encoder = torch.nn.functional.pad(
                    enroll, (0, window_size - enroll.shape[-1])
                )
            encoder_batches = [
                (torch.arange(expected_rows, device=enroll.device), enroll_for_encoder)
            ]
        else:
            encoder_batches = []
            for valid_length in torch.unique(enroll_lengths, sorted=True).tolist():
                row_indices = torch.nonzero(
                    enroll_lengths == valid_length, as_tuple=False
                ).flatten()
                cropped = enroll.index_select(0, row_indices)[..., :valid_length]
                encoded_length = max(valid_length, window_size or 0)
                if encoded_length > valid_length:
                    cropped = torch.nn.functional.pad(
                        cropped, (0, encoded_length - valid_length)
                    )
                encoder_batches.append((row_indices, cropped))

        embedding = None
        with torch.no_grad():
            for row_indices, enroll_for_encoder in encoder_batches:
                with torch.autocast(
                    device_type=enroll_for_encoder.device.type, enabled=False
                ):
                    group_embedding = encoder(enroll_for_encoder.float())
                if not isinstance(group_embedding, torch.Tensor):
                    raise TypeError(
                        "SpeakerEncoder must return a torch.Tensor sentence embedding, "
                        f"got {type(group_embedding).__name__}"
                    )
                if group_embedding.ndim != 2 or group_embedding.shape[0] != row_indices.numel():
                    raise ValueError(
                        "SpeakerEncoder embedding must have one sentence row per enroll row: "
                        f"got {tuple(group_embedding.shape)} for {row_indices.numel()} rows"
                    )
                if embedding is None:
                    embedding = group_embedding.new_empty(
                        (expected_rows, group_embedding.shape[1])
                    )
                elif group_embedding.shape[1] != embedding.shape[1]:
                    raise ValueError("SpeakerEncoder returned inconsistent dimensions across buckets")
                # 按原始行号直接写回，避免 cat + argsort 的额外大 tensor 分配。
                embedding.index_copy_(0, row_indices, group_embedding)
        if embedding is None:
            raise RuntimeError("SpeakerEncoder received no enrollment rows")
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(
                "SpeakerEncoder must return a torch.Tensor sentence embedding, "
                f"got {type(embedding).__name__}"
            )
        if not embedding.is_floating_point():
            raise TypeError(
                "SpeakerEncoder embedding must have a floating dtype, "
                f"got {embedding.dtype}"
            )
        if embedding.device != self.device:
            raise ValueError(
                "SpeakerEncoder embedding device mismatch: "
                f"embedding={embedding.device}, trainer={self.device}"
            )
        if embedding.dim() != 2 or embedding.shape[0] != expected_rows:
            raise ValueError(
                "SpeakerEncoder embedding must have one sentence row per enroll row: "
                f"expected ({expected_rows}, {self._spk_embedding_dim}), "
                f"got {tuple(embedding.shape)}"
            )
        if embedding.shape[1] != self._spk_embedding_dim:
            raise ValueError(
                "SpeakerEncoder embedding dimension mismatch: "
                f"expected {self._spk_embedding_dim}, got {embedding.shape[1]}"
            )
        if not bool(torch.isfinite(embedding).all().item()):
            raise ValueError("SpeakerEncoder embedding contains NaN/Inf")
        return embedding.detach()

    _mask_waveform_tail = staticmethod(mask_waveform_tail)
    _per_target_lengths = staticmethod(per_target_lengths)

    def _forward_tse_batch(
        self,
        batch: TSEBatch,
        *,
        cues: TSECueBatches | None = None,
        spk_embedding: torch.Tensor | None = None,
        sample_rate: int | None = None,
    ) -> dict[str, Any]:
        """通过共享 TSE 模型边界执行 typed batch 的前向与解码。"""
        if not isinstance(batch, TSEBatch):
            raise TypeError("_forward_tse_batch requires TSEBatch")
        stft = self.stft_module
        if stft is None:
            raise RuntimeError(
                "TSETrainer requires stft_module. Set top-level 'stft' in YAML "
                "or pass a built module from train.py."
            )
        if sample_rate is None:
            sample_rate = int(self.sample_rate)
        resolved_cues = batch.cues if cues is None else cues
        geometry = batch.geometry[0] if batch.geometry else None
        reference_channel = (
            geometry.reference_channel if geometry is not None else None
        )

        data = build_tse_model_inputs(
            model=self._raw_model(),
            stft=stft,
            mixture=batch.mixture,
            target=batch.target,
            lengths=batch.lengths,
            spk_embedding=spk_embedding,
            num_targets=batch.num_targets,
            sample_rate=sample_rate,
            cues=resolved_cues,
            reference_channel=reference_channel,
        )
        data = self.model(data)
        model_enhanced_spectra = data.get("target_estimate_spectra")
        data = decode_tse_model_output(
            data,
            stft=stft,
            reference_length=batch.mixture.shape[-1],
            target=data.get("target"),
            target_lengths=data.get("lengths"),
        )
        if model_enhanced_spectra is not None:
            data["loss_enhanced_spectra"] = model_enhanced_spectra
        return data

    def _assert_batch_valid(self, batch: TSEBatch) -> None:
        if not isinstance(batch, TSEBatch):
            raise TypeError("TSE strict checks require TSEBatch")
        speaker_waveform = batch.cues.speaker_waveform
        if speaker_waveform is not None and not isinstance(
            speaker_waveform.lengths, torch.Tensor
        ):
            raise TypeError("speaker_waveform lengths must be a tensor")
        batch.validate()
        if bool((batch.spk_labels < -1).any().item()):
            raise ValueError("spk_labels must be >= -1")

    # ------------------------------------------------------------------
    # Template-method hooks (unified in core.trainer.metrics_trainer)
    # ------------------------------------------------------------------

    _train_epoch_tail_label = "TSE"

    def _unpack_batch(self, raw_batch: Any) -> TSEBatch:
        """在 trainer 入口拒绝旧 tuple，只接受 validated ``TSEBatch``。"""
        return _unpack_tse_batch(raw_batch)

    def _batch_to_device(self, batch: TSEBatch) -> TSEBatch:
        if not isinstance(batch, TSEBatch):
            raise TypeError("TSE device transfer requires TSEBatch")
        return batch.to(self.device, non_blocking=True)

    def _batch_size_from_batch(self, batch: TSEBatch) -> int:
        # 基类的属性名列表（clean/noisy/nearend/mic）不含 TSE 的 mixture，按 hook 覆写。
        return int(batch.mixture.shape[0])

    def _throughput_row_counts(self, batch: TSEBatch) -> tuple[int, int]:
        """分别统计输入 mixture 与展开后的 target row。"""
        return int(batch.mixture.shape[0]), int(batch.target.shape[0])

    def _log_task_throughput(
        self, primary_rate: float, task_row_rate: float, epoch: int
    ) -> None:
        self._log_scalar("system/mixtures_per_sec", primary_rate, epoch)
        self._log_scalar("system/target_rows_per_sec", task_row_rate, epoch)
        logger.info(
            f"  [E{epoch}] TSE throughput: mixtures/s={primary_rate:.1f} "
            f"target-rows/s={task_row_rate:.1f}"
        )

    def _maybe_count_silence_placeholders(self, batch: TSEBatch) -> int:
        # 占位样本是逐字段全零波形（见 Dataset._placeholder_sample），长度随
        # max_duration 配置变化，lengths 判据不可靠；改为能量判据：
        # mixture + target 同时零能量才是占位样本（照 AEC 的 mic+nearend 判据）。
        energy_mix = batch.mixture.square().reshape(batch.mixture.shape[0], -1).sum(dim=1)
        energy_tgt = batch.target.square().reshape(batch.target.shape[0], -1).sum(dim=1)
        if batch.num_targets is not None:
            # 多目标批：energy_tgt 为 per-target 行（[sum(K_i)]），按
            # num_targets 分段取 max 归并回 per-mixture 判据（任一目标行
            # 非零即非占位），保持计数口径与单目标批一致（按 mixture 条数）。
            k_list = batch.num_targets.long().tolist()
            rows_per_mix: list[torch.Tensor] = []
            start = 0
            for k in k_list:
                rows_per_mix.append(energy_tgt[start : start + k].max())
                start += k
            energy_tgt = torch.stack(rows_per_mix)
        return int(((energy_mix < 1e-10) & (energy_tgt < 1e-10)).sum().item())

    def _canonicalize_batch_tails(self, batch: TSEBatch) -> TSEBatch:
        """在 trainer 边界再次按 metadata 清零波形无效尾部。"""
        target_lengths = self._per_target_lengths(batch.lengths, batch.num_targets)
        batch.mixture = self._mask_waveform_tail(batch.mixture, batch.lengths)
        batch.target = self._mask_waveform_tail(batch.target, target_lengths)
        speaker_waveform = batch.cues.speaker_waveform
        if speaker_waveform is not None:
            if speaker_waveform.lengths is None:
                raise ValueError("speaker_waveform cue requires lengths")
            safe_lengths = speaker_waveform.lengths.clamp_min(1)
            masked = self._mask_waveform_tail(speaker_waveform.value, safe_lengths)
            speaker_waveform.value = masked.masked_fill(
                ~speaker_waveform.present.unsqueeze(1), 0
            )
        return batch

    @staticmethod
    def _floating_cue_tensors(batch: TSEBatch):
        for slot in CUE_SLOTS:
            cue = getattr(batch.cues, slot)
            if cue is None:
                continue
            if cue.value.is_floating_point():
                yield cue, "value", cue.value
            if cue.timebase is not None and cue.timebase.is_floating_point():
                yield cue, "timebase", cue.timebase

    def _sanitize_batch_inputs(
        self, batch: TSEBatch, epoch: int, batch_idx: int
    ) -> TSEBatch:
        cue_tensors = list(self._floating_cue_tensors(batch))
        has_nonfinite = not bool(torch.stack([
            torch.isfinite(value).all()
            for value in (batch.mixture, batch.target, *(value for _, _, value in cue_tensors))
        ]).all().item())
        if has_nonfinite and self._strict_checks:
            self._assert_batch_valid(batch)
        if has_nonfinite:
            if self.rank == 0:
                logger.warning(
                    f"[E{epoch} B{batch_idx}] NaN/Inf in TSE batch inputs, zeroed out"
                )
            batch.mixture = torch.nan_to_num(
                batch.mixture, nan=0.0, posinf=0.0, neginf=0.0
            )
            batch.target = torch.nan_to_num(
                batch.target, nan=0.0, posinf=0.0, neginf=0.0
            )
            for cue, field_name, value in cue_tensors:
                setattr(
                    cue,
                    field_name,
                    torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0),
                )
        batch = self._canonicalize_batch_tails(batch)
        if self._strict_checks:
            self._assert_batch_valid(batch)
        return batch

    @staticmethod
    def _replace_speaker_waveform(
        cues: TSECueBatches,
        value: torch.Tensor,
        lengths: torch.Tensor,
    ) -> TSECueBatches:
        original = cues.speaker_waveform
        if original is None:
            raise ValueError("speaker_waveform cue is required for waveform replacement")
        replacement = replace(
            original,
            value=value,
            present=torch.ones(value.shape[0], dtype=torch.bool, device=value.device),
            lengths=lengths.to(device=value.device, dtype=torch.int32),
            timebase=None,
        )
        return replace(cues, speaker_waveform=replacement)

    def _maybe_ssa_cues(
        self,
        batch: TSEBatch,
        epoch: int,
        batch_idx: int,
    ) -> TSECueBatches:
        """命中 SSA 时用自估计目标替换 typed speaker waveform cue。"""
        if self._ssa_prob <= 0.0:
            return batch.cues
        if not self._ssa_joint_training:
            if not self._ssa_skip_warned and self.rank == 0:
                logger.warning(
                    "ssa_prob=%g 但 joint_training=False（预提取 embedding）："
                    "SSA 不生效，全程使用原 speaker_waveform cue",
                    self._ssa_prob,
                )
                self._ssa_skip_warned = True
            return batch.cues
        base_seed = int((self.config.get("meta") or {}).get("seed", 42))
        gen = torch.Generator().manual_seed(
            base_seed + epoch * 10_000_000 + batch_idx + 486_187_739
        )
        if float(torch.rand((), generator=gen)) >= self._ssa_prob:
            return batch.cues
        execution_cues, spk_embedding = self._prepare_speaker_conditioning(
            batch,
            batch.cues,
            phase="tse_ssa_speaker_precondition",
        )
        training_states = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            with torch.no_grad():
                est0 = self._forward_tse_batch(
                    batch,
                    cues=execution_cues,
                    spk_embedding=spk_embedding,
                    sample_rate=self._batch_fs_hz(batch.sr, batch.mixture.shape[0]),
                )["target_estimate"]
        finally:
            for module, was_training in training_states:
                module.training = was_training
        est0 = torch.nan_to_num(est0, nan=0.0, posinf=0.0, neginf=0.0)
        est0_lengths = self._per_target_lengths(batch.lengths, batch.num_targets)
        est0 = self._mask_waveform_tail(est0, est0_lengths)
        row_mean_square = est0.square().sum(dim=-1) / est0_lengths.to(est0.dtype)
        if bool((row_mean_square < 1e-10).any().item()):
            if self.rank == 0:
                logger.warning(
                    f"[E{epoch} B{batch_idx}] SSA pre-forward est0 energy collapsed "
                    "for at least one row (mean square < 1e-10); falling back to original cue"
                )
            return batch.cues
        return self._replace_speaker_waveform(batch.cues, est0, est0_lengths)

    def _composite_spk_ce_kwargs(
        self,
        data: dict[str, Any],
        spk_labels: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """multi-task speaker CE 接线（照 AEC state detector 的 kwargs 传递范例）。

        loss 含非零权重 ``SpeakerCrossEntropyLoss`` 分量时，必须同时有模型
        ``spk_logits`` 与 batch ``spk_labels``，缺一即 fail-fast。collate 输出恒含
        ``spk_labels``，不能与启用的 multi-task CE 静默组合。全 -1 label（未知
        说话人）是合法输入，由 loss 的 ``ignore_index`` 路径处理。
        无有效 CE 分量时返回空 kwargs，避免把无关张量喷进纯波形 loss。
        """
        inner = self._inner_loss_fn()
        if not isinstance(inner, CompositeLoss):
            return {}
        active_spk_ce = any(
            typ == "SpeakerCrossEntropyLoss" and weight != 0.0
            for typ, weight in zip(
                getattr(inner, "_component_types", []),
                getattr(inner, "_weights_float", []),
            )
        )
        if not active_spk_ce:
            return {}
        if spk_labels is None:
            raise ValueError(
                "TSE batch spk_labels is None, but a non-zero "
                "SpeakerCrossEntropyLoss component is enabled (collate always "
                "emits spk_labels; None indicates a hand-constructed TSEBatch)"
            )
        if "spk_logits" not in data:
            raise ValueError(
                "model output is missing spk_logits required by the enabled non-zero "
                "SpeakerCrossEntropyLoss component"
            )
        return {"spk_logits": data["spk_logits"], "spk_labels": spk_labels}

    def _compute_tse_supervision_loss(
        self,
        data: dict[str, Any],
        target_estimate: torch.Tensor,
        spk_labels: torch.Tensor | None,
        fs_val: int,
    ) -> torch.Tensor:
        """按现有 TSE loss capability 计算单路监督。"""
        mixture_for_loss = data["mixture"]
        target_for_loss = data["target"]
        lengths_for_loss = data["lengths"]
        loss_kwargs = _waveform_loss_kwargs(
            self.loss_fn,
            mixture_for_loss,
            target_for_loss,
            target_estimate,
            fs_val,
        )
        loss_kwargs.update(self._composite_spk_ce_kwargs(data, spk_labels))
        loss_kwargs.update(
            _direct_spectral_loss_kwargs(
                self.loss_fn,
                getattr(self, "stft_module", None),
                target_for_loss,
                data.get("loss_enhanced_spectra"),
                lengths=lengths_for_loss,
            )
        )
        return self.loss_fn(
            target_estimate,
            target_for_loss,
            lengths_for_loss,
            **loss_kwargs,
        )

    def _loss_breakdown_owner(self) -> Any:
        """返回实际持有 CompositeLoss breakdown 状态的对象。"""
        inner = self._inner_loss_fn()
        if hasattr(inner, "_breakdown_tensors"):
            return inner
        return self.loss_fn

    def _merge_dual_loss_breakdowns(
        self,
        original: list | None,
        self_estimated: list | None,
        weights: tuple[float, float],
    ) -> None:
        """把两次 CompositeLoss breakdown 合并为真实 dual objective 口径。"""
        if original is None or self_estimated is None:
            return
        if len(original) != len(self_estimated):
            raise RuntimeError("dual loss breakdown component count changed between passes")
        merged = []
        for left, right in zip(original, self_estimated):
            if left[:2] != right[:2]:
                raise RuntimeError(
                    "dual loss breakdown component identity changed between passes"
                )
            name, component_weight = left[:2]
            raw = weights[0] * left[2] + weights[1] * right[2]
            weighted = weights[0] * left[3] + weights[1] * right[3]
            merged.append((name, component_weight, raw, weighted))
        owner = self._loss_breakdown_owner()
        owner._breakdown_tensors = merged
        if hasattr(owner, "_breakdown_cache"):
            owner._breakdown_cache = None

    def _forward_and_compute_loss(
        self,
        batch: TSEBatch,
        epoch: int,
        batch_idx: int,
        fs_val: int,
    ) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        cues_used = self._maybe_ssa_cues(batch, epoch, batch_idx)
        execution_cues, spk_embedding = self._prepare_speaker_conditioning(
            batch,
            cues_used,
            phase="tse_train_speaker_precondition",
        )
        data = self._forward_tse_batch(
            batch,
            cues=execution_cues,
            spk_embedding=spk_embedding,
            sample_rate=fs_val,
        )
        target_estimate = data["target_estimate"]
        target_estimate, _n_nf, _n_cl = sanitize_model_output(target_estimate)
        self._accumulate_output_sanitize(_n_nf, _n_cl)
        loss = self._compute_tse_supervision_loss(
            data, target_estimate, batch.spk_labels, fs_val
        )
        breakdown_owner = self._loss_breakdown_owner()
        original_breakdown = getattr(breakdown_owner, "_breakdown_tensors", None)

        dual_weights = getattr(self, "_dual_self_estimated_loss_weights", None)
        dual_info: dict[str, float] = {}
        if dual_weights is not None:
            self_lengths = data["lengths"]
            self_cues = self._replace_speaker_waveform(
                batch.cues,
                target_estimate.detach(),
                self_lengths,
            )
            self_execution_cues, self_spk_embedding = (
                self._prepare_speaker_conditioning(
                    batch,
                    self_cues,
                    phase="tse_dual_speaker_precondition",
                )
            )
            self_data = self._forward_tse_batch(
                batch,
                cues=self_execution_cues,
                spk_embedding=self_spk_embedding,
                sample_rate=fs_val,
            )
            self_estimate, self_nonfinite, self_clamped = sanitize_model_output(
                self_data["target_estimate"]
            )
            self._accumulate_output_sanitize(self_nonfinite, self_clamped)
            self_loss = self._compute_tse_supervision_loss(
                self_data, self_estimate, batch.spk_labels, fs_val
            )
            self_breakdown = getattr(
                self._loss_breakdown_owner(), "_breakdown_tensors", None
            )
            loss = dual_weights[0] * loss + dual_weights[1] * self_loss
            self._merge_dual_loss_breakdowns(
                original_breakdown, self_breakdown, dual_weights
            )
            data = self_data
            target_estimate = self_estimate
            dual_info = {
                "dual_original_loss_weight": dual_weights[0],
                "dual_self_loss_weight": dual_weights[1],
            }
        loss_was_finite = bool(torch.isfinite(loss).item())
        info = self._collect_step_diagnostics(data, target_estimate, batch)
        info.update(dual_info)
        info["loss_was_finite"] = loss_was_finite
        if not loss_was_finite:
            est_safe = torch.nan_to_num(
                target_estimate.detach(), nan=0.0, posinf=0.0, neginf=0.0
            )
            info["enhanced_safe_min"] = float(est_safe.min().item())
            info["enhanced_safe_max"] = float(est_safe.max().item())
        return loss, loss_was_finite, info

    def _collect_step_diagnostics(
        self,
        data: dict[str, Any],
        target_estimate: torch.Tensor,
        batch: TSEBatch,
    ) -> dict[str, Any]:
        """Compute lightweight scalar diagnostics; avoids storing full tensors in loss_info."""
        info: dict[str, Any] = {"out_energy": float(target_estimate.abs().mean().item())}
        if info["out_energy"] >= 1e-5:
            return info
        with torch.no_grad():
            info["mixture_energy"] = float(batch.mixture.abs().mean().item())
            info["target_energy"] = float(batch.target.abs().mean().item())
            speaker_waveform = batch.cues.speaker_waveform
            info["speaker_waveform_energy"] = (
                float(speaker_waveform.value.abs().mean().item())
                if speaker_waveform is not None
                else float("nan")
            )
        return info

    def _maybe_log_step_diagnostics(
        self,
        batch: TSEBatch,
        loss_info: dict[str, Any],
        epoch: int,
        batch_idx: int,
    ) -> None:
        if self.rank != 0:
            return
        out_energy = loss_info.get("out_energy")
        if out_energy is None or out_energy >= 1e-5:
            return
        logger.warning(
            f"[E{epoch} B{batch_idx}] Output energy collapsed: {out_energy:.2e}. "
            f"InputE: mixture={loss_info.get('mixture_energy'):.2e} "
            f"target={loss_info.get('target_energy'):.2e} "
            f"speaker_waveform={loss_info.get('speaker_waveform_energy'):.2e}. "
            f"Check speaker-cue wiring and placeholder ratio."
        )

    def _val_loss_weight(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        ctx: Any,
    ) -> int:
        """TSE validation loss 按展开后的 target rows 加权。"""
        return int(clean.shape[0])

    def _val_prepare_batch(
        self, raw_batch: Any, epoch: int, n_loops: int
    ) -> _PreparedTSEValidationBatch:
        """validation 复用训练侧 typed batch 搬运、校验与尾部清零。"""
        batch = self._batch_to_device(_unpack_tse_batch(raw_batch))
        batch = self._sanitize_batch_inputs(batch, epoch, n_loops)
        fs_val = self._batch_fs_hz(batch.sr, batch.mixture.shape[0])
        lengths_for_loss = self._per_target_lengths(
            batch.lengths, batch.num_targets
        )
        return _PreparedTSEValidationBatch(
            batch=batch,
            lengths_for_loss=lengths_for_loss,
            fs_val=fs_val,
        )

    def _val_forward_prepared(
        self,
        prepared_batch: _PreparedTSEValidationBatch,
        epoch: int,
        n_loops: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Any]:
        """仅在 prepare 异常已跨 rank 同步后执行模型前向与 loss。"""
        if not isinstance(prepared_batch, _PreparedTSEValidationBatch):
            raise TypeError(
                "TSE _val_forward_prepared requires _PreparedTSEValidationBatch"
            )
        batch = prepared_batch.batch
        lengths_for_loss = prepared_batch.lengths_for_loss
        fs_val = prepared_batch.fs_val

        def _forward_and_build_loss_kwargs():
            with torch.amp.autocast(
                self.device.type, enabled=self.use_amp, dtype=self.amp_dtype
            ):
                execution_cues, spk_embedding = self._prepare_speaker_conditioning(
                    batch,
                    batch.cues,
                    phase="tse_validation_speaker_precondition",
                )
                data = self._forward_tse_batch(
                    batch,
                    cues=execution_cues,
                    spk_embedding=spk_embedding,
                    sample_rate=fs_val,
                )
                target_estimate = self._sanitize_model_output(
                    data["target_estimate"]
                )
                if self._strict_checks and target_estimate.shape != batch.target.shape:
                    raise ValueError(
                        "Model output shape mismatch: "
                        f"target_estimate={tuple(target_estimate.shape)} "
                        f"target={tuple(batch.target.shape)}"
                    )
                loss_kwargs = _waveform_loss_kwargs(
                    self.loss_fn,
                    data["mixture"],
                    batch.target,
                    target_estimate,
                    fs_val,
                )
                loss_kwargs.update(
                    self._composite_spk_ce_kwargs(data, batch.spk_labels)
                )
                loss_kwargs.update(
                    _direct_spectral_loss_kwargs(
                        self.loss_fn,
                        getattr(self, "stft_module", None),
                        batch.target,
                        data.get("loss_enhanced_spectra"),
                        lengths=lengths_for_loss,
                    )
                )
                return data, target_estimate, loss_kwargs

        data, target_estimate, loss_kwargs = self._run_synced_validation_subphase(
            _forward_and_build_loss_kwargs,
            phase="validation_model_output_error",
            context="TSE validation model/output preflight failed",
        )
        with torch.amp.autocast(
            self.device.type, enabled=self.use_amp, dtype=self.amp_dtype
        ):
            loss = self.loss_fn(
                target_estimate,
                batch.target,
                lengths_for_loss,
                **loss_kwargs,
            )
        return (
            loss,
            target_estimate,
            batch.target,
            lengths_for_loss,
            fs_val,
            None,
        )

    def _val_forward_loss(
        self, raw_batch: Any, epoch: int, n_loops: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Any]:
        """顺序委托 typed prepare 与 forward_prepared hooks。"""
        prepared = self._val_prepare_batch(raw_batch, epoch, n_loops)
        return self._val_forward_prepared(prepared, epoch, n_loops)
