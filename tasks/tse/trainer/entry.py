#!/usr/bin/env python3
"""TSE training entry point.

Hosts ``TSETrainingEntry``, the TSE-specific concrete subclass of the
task-agnostic ``core.trainer.entry.UnifiedTrainingEntry``（照
``tasks/aec/trainer/entry.py::AECTrainingEntry`` 模式）。
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from core.data.constants import DATA_MODE_DYNAMIC_TSE, DATA_MODE_OFFLINE_TSE
from core.data.dataset_build import (
    validate_tse_train_resource_paths,
    validate_tse_validation_resource_provenance,
)
from core.trainer.entry import (
    UnifiedTrainingEntry,
    detect_task,
    set_spawn_start_method,
)
from tasks.tse.models.speaker.config import (
    config_bool,
    require_bool,
    resolve_external_speaker_checkpoint,
    resolve_joint_sentence_batch_encoder,
)
from tasks.tse.objectives import (
    resolve_dual_self_estimated_objective,
    validate_dual_objective_model,
    validate_tse_objective_config,
)


# Re-export detect_task / set_spawn_start_method so callers can import everything
# they need from either ``core.trainer.entry`` or ``tasks.tse.trainer.entry``.
__all__ = ["TSETrainingEntry", "detect_task", "set_spawn_start_method"]


def _tse_dataset_is_multichannel(dataset) -> bool:
    """从已解析 geometry 判断 TSE dataset 是否真正有 microphone 轴。"""
    from core.models.io_contract import resolve_dataset_microphone_array
    geometry = resolve_dataset_microphone_array(dataset)
    return geometry is not None and geometry.channels > 1


def _tse_model_supports_multi_task(model_type) -> bool:
    """模型类以 ``supports_multi_task`` 声明是否有 speaker 分类头。

    模型未声明该属性时默认按 ``True`` 处理；新增模型（如维护独立参数布局的
    ``TSE_Modular``）应显式声明 ``False``，避免入口把 ``multi_task`` /
    ``spks_in_train`` 注入到不接受它们的构造签名。
    """
    from core.utils.registry import MODEL_REGISTRY

    if not isinstance(model_type, str) or not model_type:
        return True
    if model_type not in MODEL_REGISTRY:
        return True
    return bool(getattr(MODEL_REGISTRY.get(model_type), "supports_multi_task", True))


def _inject_tse_scene_geometry(cfg: dict, dataset, validation_dataset=None) -> None:
    """在 TSE 模型构造前注入实际阵列坐标与 scene 声速。"""
    import inspect
    from core.models.build import resolve_model_class
    from core.models.io_contract import resolve_dataset_microphone_array

    geometry = resolve_dataset_microphone_array(dataset)
    if geometry is None:
        return
    model_cfg = cfg.get("model", {})
    if "generator" in model_cfg:
        model_cfg = model_cfg["generator"]
    model_cls = resolve_model_class(model_cfg)
    parameters = inspect.signature(model_cls).parameters
    args = dict(model_cfg.get("args") or {})
    spatial_requested = isinstance(args.get("cues"), Mapping) and "spatial" in args["cues"]
    if spatial_requested and "mic_positions_m" in parameters:
        expected_positions = tuple(tuple(float(value) for value in row) for row in geometry.positions_m)
        if validation_dataset is not None:
            validation_geometry = resolve_dataset_microphone_array(validation_dataset)
            if validation_geometry is not None:
                validation_positions = tuple(
                    tuple(float(value) for value in row)
                    for row in validation_geometry.positions_m
                )
                if validation_positions != expected_positions:
                    raise ValueError(
                        "train/validation scene mic_positions_m must match model geometry"
                    )
        if "mic_positions_m" in args and tuple(tuple(float(value) for value in row) for row in args["mic_positions_m"]) != expected_positions:
            raise ValueError("model.args.mic_positions_m conflicts with actual dataset scene geometry")
        args["mic_positions_m"] = expected_positions
    if spatial_requested and "speed_of_sound_mps" in parameters:
        speeds = [getattr(dataset, "spatial_speed_of_sound_mps", None)]
        if validation_dataset is not None:
            speeds.append(getattr(validation_dataset, "spatial_speed_of_sound_mps", None))
        known = [float(speed) for speed in speeds if speed is not None]
        if not known:
            raise ValueError("spatial TSE model requires actual scene speed_of_sound_mps")
        if any(speed != known[0] for speed in known[1:]):
            raise ValueError("train/validation scene speed_of_sound_mps must match model geometry")
        if "speed_of_sound_mps" in args and float(args["speed_of_sound_mps"]) != known[0]:
            raise ValueError("model.args.speed_of_sound_mps conflicts with actual scene geometry")
        args["speed_of_sound_mps"] = known[0]
    model_cfg["args"] = args


def _positive_config_int(value, path: str) -> int:
    """解析正整数配置，拒绝 bool、截断浮点数与隐式字符串转换。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be a positive integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{path} must be > 0, got {value}")
    return value


def _config_mapping(value, path: str) -> Mapping:
    """要求配置 section 为 mapping，并在错误中保留完整路径。"""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping, got {value!r}")
    return value


def _optional_config_mapping(value, path: str) -> Mapping:
    """将缺失语义的 ``None`` 归一为空 mapping，并拒绝其它畸形 section。"""
    if value is None:
        return {}
    return _config_mapping(value, path)


def _validate_tse_model_mappings(cfg: Mapping) -> tuple[Mapping, Mapping]:
    """在读取 TSE model 子项前校验完整的 mapping 配置路径。"""
    if "model" not in cfg:
        return {}, {}
    model_cfg = _config_mapping(cfg["model"], "model")
    model_args = _optional_config_mapping(model_cfg.get("args"), "model.args")
    speaker_model = _optional_config_mapping(
        model_args.get("speaker_model"), "model.args.speaker_model"
    )
    fbank = _optional_config_mapping(
        speaker_model.get("fbank"), "model.args.speaker_model.fbank"
    )
    _optional_config_mapping(
        speaker_model.get("speaker_encoder"),
        "model.args.speaker_model.speaker_encoder",
    )
    return model_args, fbank


def _validate_tse_boolean_config(model_args: Mapping) -> tuple[bool, Mapping]:
    """校验 TSE 的模式开关与 speaker cue 开关，并返回已确认的值。"""
    if not isinstance(model_args, Mapping):
        raise TypeError("model.args must be a mapping")

    joint_training = config_bool(
        model_args, "joint_training", True, "model.args"
    )
    features = model_args.get("features")
    if features is None:
        features = {}
    if not isinstance(features, Mapping):
        raise TypeError("model.args.features must be a mapping")

    for name, feature_cfg in features.items():
        if not isinstance(feature_cfg, Mapping):
            raise TypeError(f"model.args.features.{name} must be a mapping")
        config_bool(
            feature_cfg,
            "enabled",
            False,
            f"model.args.features.{name}",
        )
    return joint_training, features


def _validate_preextracted_tse_mode(model_args: Mapping) -> None:
    """校验非 joint TSE 的 cue 能力；source/ckpt 在 dataset 构造后解析。"""
    _, features = _validate_tse_boolean_config(model_args)

    def _feature_enabled(name: str) -> bool:
        feature_cfg = features.get(name)
        if feature_cfg is None:
            return False
        return config_bool(
            feature_cfg,
            "enabled",
            False,
            f"model.args.features.{name}",
        )

    if _feature_enabled("context"):
        raise ValueError(
            "model.args.joint_training=False 不支持 "
            "model.args.features.context.enabled=true：context 需要帧级 speaker embedding"
        )


def _per_rank_train_batch_size(cfg: Mapping) -> int | None:
    """读取 ``data.dataloader.batch_size``（per-rank）；无法解析时返回 ``None``。

    该值就是每个 rank 自己的 batch（``core.trainer.entry_components`` 按它构造
    ``GroupedBatchSampler``），不是 ``world_size * batch_size`` 的全局 batch。
    """
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, Mapping):
        return None
    dataloader = data_cfg.get("dataloader")
    if not isinstance(dataloader, Mapping):
        return None
    raw = dataloader.get("batch_size")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _validate_tse_speaker_train_batch(cfg: Mapping) -> None:
    """拒绝 train 模式下必然在 speaker encoder 内部崩溃的 batch=1 配置。

    joint 训练且 effective ``features.spkemb`` 生效时，TSE 模型内置的 speaker
    encoder 句级 head 随 ``model.train()`` 进入 train 模式；句级 head 含
    ``nn.BatchNorm1d``，其输入是池化后的 ``(per-forward enrollment 行数, dim)``，
    没有额外样本轴，train 模式要求每个通道至少 2 个样本。batch=1 意味着整批只有
    一个 enrollment 行，永远无法形成多行分组，因此在配置边界显式 fail-fast（早于
    DDP 初始化与 dataloader 构建）。

    冻结 encoder、``joint_training=false``（外部冻结 encoder）、未启用 ``spkemb``、
    帧级 cue（``context`` / ``tfmap``）以及非句级 BatchNorm 的 speaker 模型都不受
    影响；推理/评估路径不经过本校验，eval 模式下 BatchNorm 使用 running stats，
    batch=1 继续可用。

    变长 enrollment 造成的单行分组已由句级 BN 的 running statistics 路径处理（
    ``tasks/tse/models/speaker/encoder.py::sentence_batch_statistics_off``）：单个
    长度分组只有一行时该组临时改用 running stats，每行输出仍只依赖自身，不再触发
    ``Expected more than 1 value per channel when training``。单行分组不更新
    running stats，所以本校验保持不放宽：per-rank ``batch_size >= 2`` 是必要条件，
    避免出现整批只有一个 enrollment 行的退化配置。
    """
    per_rank_batch = _per_rank_train_batch_size(cfg)
    if per_rank_batch is None or per_rank_batch >= 2:
        return
    model_cfg = cfg.get("model")
    model_cfg = model_cfg if isinstance(model_cfg, Mapping) else {}
    model_args = model_cfg.get("args")
    model_args = model_args if isinstance(model_args, Mapping) else {}
    encoder_model = resolve_joint_sentence_batch_encoder(model_args)
    if encoder_model is None:
        return
    raise ValueError(
        "TSE joint training + features.spkemb 下，模型内置 speaker encoder "
        f"{encoder_model!r} 的句级 head 需要 >= 2 行样本轴：句级 head 的 "
        "nn.BatchNorm1d（ECAPA 为 "
        "tasks/tse/models/speaker/ecapa.py::ECAPA_TDNN.forward 的 self.bn，"
        "ResNet34 为 two_emb_layer=True 时的 resnet.py::ResNet.seg_bn_1）在 train "
        "模式要求每个通道至少 2 个样本，而它的输入是 "
        "(per-forward enrollment 行数, dim)，没有额外样本轴。变长 enrollment 产生"
        "的单行分组会改用 running statistics（tasks/tse/models/speaker/encoder.py::"
        "sentence_batch_statistics_off），但整批只有一行时永远没有多行分组。当前 "
        f"data.dataloader.batch_size={per_rank_batch} 是 per-rank batch size"
        "（每个 rank 自己的 batch，不是 world_size * batch_size 的全局 batch），"
        "DDP 下不会因为全局 batch 更大而满足该要求。修复方式（任选其一）："
        "(1) 设置 data.dataloader.batch_size >= 2（每个 rank 都 >= 2）；"
        "(2) 冻结内置 encoder：提供 model.args.speaker_model.speaker_encoder."
        "pretrained 并保持 freeze: null/true（冻结后句级 head 恒为 eval，使用 "
        "BatchNorm running stats，batch=1 合法）；"
        "(3) 改用预提取 embedding：model.args.joint_training=false。"
    )


def _validate_spk_ce_loss_components(cfg: Mapping) -> bool:
    """拒绝仅有非零 SpeakerCrossEntropyLoss 的 CompositeLoss 配置。"""
    loss_cfg = cfg.get("loss", {})
    components = (
        loss_cfg.get("components") if loss_cfg.get("type") == "CompositeLoss" else []
    )
    has_spk_ce = any(
        isinstance(component, dict)
        and component.get("type") == "SpeakerCrossEntropyLoss"
        and float(component.get("weight", 1.0)) != 0.0
        for component in (components or [])
    )
    has_primary_loss = any(
        isinstance(component, dict)
        and component.get("type") != "SpeakerCrossEntropyLoss"
        and float(component.get("weight", 1.0)) != 0.0
        for component in (components or [])
    )
    if has_spk_ce and not has_primary_loss:
        raise ValueError(
            "SpeakerCrossEntropyLoss is auxiliary-only and requires at least one "
            "non-zero non-SpeakerCrossEntropyLoss CompositeLoss component."
        )
    return has_spk_ce


def _reject_evaluation_only_config(cfg: Mapping) -> None:
    """拒绝把仅评测配置送入 TSE 训练路径。"""
    meta = cfg.get("meta")
    if isinstance(meta, Mapping) and meta.get("evaluation_only") is True:
        raise ValueError(
            "meta.evaluation_only=true 配置只能用于评测/数据合同审计，不能用于训练"
        )


class TSETrainingEntry(UnifiedTrainingEntry):
    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------
    def _validate_config(self, cfg: dict) -> None:
        """Fail-fast validation of required config keys.

        TSE 额外校验 dynamic_tse dataset 配置：
        ``data.train.spk2enroll_json`` 必须给出 nested enrollment JSON，且
        train/validation 的 ``speech_scp`` 仅接受带 ``dataset_id`` / ``speaker_id``
        metadata 的 strict JSONL，identity 直接由 JSONL 构建。validation ``speech_scp``
        只有 key 缺失时继承 train，显式
        无效值会按 ``data.validation.speech_scp`` fail-fast；动态仿真且
        ``inherit_simulation: false`` 时同样校验 data.validation 本地 enrollment 资源。
        另外，``data.mode=dynamic_tse`` 时必须提供非空的 validation 段并显式
        走动态仿真（``mode: dynamic_tse`` 或 ``use_dynamic: true``）：
        dynamic_tse 无离线 val 路径，缺此项会落入 core 的离线验证分支而在
        build_dataset 时崩溃（fail-fast 在校验期拦截）。validation 段仍可按既有
        inheritance 语义省略 speech/resource 字段。``offline_tse`` 模式的 train/val
        均走 ``TSEOfflineDataset``，上述动态资源校验整体跳过。

        另外在配置边界拒绝退化的 speaker 训练配置：joint 训练 + effective
        ``features.spkemb`` + 未冻结的内置 speaker encoder 时，句级 head 的
        ``nn.BatchNorm1d`` 需要 >= 2 行样本轴，故 per-rank
        ``data.dataloader.batch_size`` 必须 >= 2；变长 enrollment 产生的单行分组由
        句级 BN running statistics 处理（见
        :func:`_validate_tse_speaker_train_batch`）。
        """
        _reject_evaluation_only_config(cfg)
        # 先校验嵌套 section，确保畸形配置报告完整路径。
        model_args, fbank_cfg = _validate_tse_model_mappings(cfg)
        validate_tse_objective_config(cfg)
        super()._validate_config(cfg)
        # 在 DDP 初始化及 loader/model 预检前拒绝 auxiliary-only 配置。
        _validate_spk_ce_loss_components(cfg)
        # 同上：在构建 dataloader 之前拒绝 train 模式必然崩溃的 batch=1 speaker 配置。
        _validate_tse_speaker_train_batch(cfg)
        data_cfg = cfg.get("data", {})
        data_mode = str(data_cfg.get("mode") or "").strip().lower()
        train_cfg = data_cfg.get("train")
        if train_cfg is None:
            train_cfg = {}
        elif not isinstance(train_cfg, Mapping):
            raise ValueError("data.train must be a mapping")
        simulation_cfg = train_cfg.get("simulation_cfg")
        if data_mode == DATA_MODE_DYNAMIC_TSE:
            if not isinstance(simulation_cfg, Mapping):
                raise ValueError("data.train.simulation_cfg must be a mapping")
            simulation = simulation_cfg
        else:
            simulation = {}
        if data_mode == DATA_MODE_DYNAMIC_TSE and any(
            simulation.get(key, 0) > 0 for key in (
                "prob_interference_only", "prob_noise_only", "prob_interference_noise_only",
            )
        ):
            components = cfg.get("loss", {}).get("components") or [cfg.get("loss", {})]
            if not any(component.get("type") in {"AbsentTargetLoss", "L1WavLoss"}
                       and float(component.get("weight", 1.0)) > 0 for component in components):
                raise ValueError("target-absent TSE scenarios require non-zero AbsentTargetLoss or L1WavLoss")
        joint_training, _ = _validate_tse_boolean_config(model_args)
        data_sr = _positive_config_int(
            data_cfg.get("sample_rate", 16000), "data.sample_rate"
        )
        sr_key = "sample_rate" if cfg.get("model", {}).get("type") == "TSE_Modular" else "sr"
        model_sr = _positive_config_int(
            model_args.get(sr_key, 16000), f"model.args.{sr_key}"
        )
        if data_sr != model_sr:
            raise ValueError(
                f"data.sample_rate={data_sr} 与 model.args.{sr_key}={model_sr} 不一致"
            )
        if "sample_rate" in fbank_cfg:
            fbank_sr = _positive_config_int(
                fbank_cfg["sample_rate"],
                "model.args.speaker_model.fbank.sample_rate",
            )
            if fbank_sr != model_sr:
                raise ValueError(
                    f"model.args.speaker_model.fbank.sample_rate={fbank_sr} 与 "
                    f"model.args.sr={model_sr}/data.sample_rate={data_sr} 不一致"
                )
        if not joint_training:
            _validate_preextracted_tse_mode(model_args)

        if data_mode != DATA_MODE_DYNAMIC_TSE:
            return
        from core.data.simulation_config_contract import validate_simulation_config_contract

        validate_simulation_config_contract(
            task="tse",
            split="train",
            train_section=train_cfg,
        )
        validate_tse_train_resource_paths(train_cfg)
        from core.data.dataset_build import project_flat_simulation_cfg
        from core.data.simulation_configs import SimulationConfigs
        from core.data.simulation.tse_scenarios import validate_tse_scenarios

        projected = project_flat_simulation_cfg(train_cfg.get("simulation_cfg") or {}, task="tse")
        validate_tse_scenarios(SimulationConfigs.from_dict(projected.simulation, strict=True))

        val_cfg = data_cfg.get("validation")
        if not isinstance(val_cfg, Mapping) or not val_cfg:
            raise ValueError(
                f"data.mode={data_mode} requires a non-empty data.validation mapping "
                "for TSE dynamic validation"
            )
        # 与 core.data.dataset_build 的 use_dynamic_val 判定对齐；TSE dynamic
        # validation 只接受任务专用 dynamic_tse 或显式 use_dynamic 标记。
        val_mode = str(val_cfg.get("mode") or "").strip().lower()
        val_dynamic = (
            val_mode == DATA_MODE_DYNAMIC_TSE
            or val_cfg.get("use_dynamic") is True
        )
        if val_cfg and not val_dynamic:
            raise ValueError(
                f"data.mode={data_mode} requires data.validation to use dynamic "
                "simulation (set 'mode: dynamic_tse' or "
                "'use_dynamic: true'); TSE has no offline validation path."
            )
        if val_dynamic:
            if "simulation_cfg" in val_cfg and val_cfg["simulation_cfg"] is not None and not isinstance(
                val_cfg["simulation_cfg"], Mapping
            ):
                raise ValueError("data.validation.simulation_cfg must be a mapping")
            validate_tse_validation_resource_provenance(val_cfg, train_cfg)

    # ------------------------------------------------------------------
    # distributed.find_unused_parameters 配置推导（仿 AEC _resolve_state_detector_cfg 范例）
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_tse_distributed_cfg(cfg: dict) -> None:
        """按模型 cue 配置推导 ``distributed.find_unused_parameters``（in-place setdefault）。

        推导链（TSE_BSRNN_SPK / SpeakerFrontend 的构造逻辑）：

        - speaker encoder（``spk_ft.encoder``）只在 ``joint_training=True`` 且
          ``context.enabled`` 或 ``spkemb.enabled`` 时才被构造；构造后必然在每次
          前向中被调用（``context.compute`` / ``spkemb.compute``），不存在
          "构造但不用"的参数。
        - 冻结 encoder（``freeze=True``，默认"有预训练则冻结"）：参数
          ``requires_grad=False``，本就不在 DDP 的 expected-grad 集合内，
          前向被使用也不会触发 unused-parameters 报错。
        - 未冻结 encoder：参数在前向计算图中，backward 必收到梯度。
        - ``joint_training=False``：模型内部 speaker encoder 不构造；``context`` cue
          在入口阶段拒绝，``spkemb`` cue 使用 trainer 外部冻结 encoder 生成的
          ``spk_embedding``，其余 listen/usef/tfmap cue 仍由模型正常参与前向。
        - 其余 cue 模块（listen/usef/tfmap）仅在 enabled 时构造且构造即用；
          ``sep_model``（BSRNN，单采样率）全参数参与前向。
        - multi-task speaker 分类头（``multi_task=True`` 时构造）：``spk_logits``
          每次前向必产出并送入 ``SpeakerCrossEntropyLoss`` 分量；非零 CE 启用时
          trainer 对缺 ``spk_labels`` 的 batch（collate 输出恒含 spk_labels，
          缺失即手工构造的 ``TSEBatch``）或缺 ``spk_logits`` 的模型输出
          fail-fast，不会静默断图；CE ``weight == 0`` 时
          ``_resolve_spk_classifier_cfg`` 把 ``multi_task`` 推导为 ``False``
          （分类头不构造），故不存在"分类头在图外"的悬空参数。

        合法配置在入口阶段已拒绝 auxiliary-only ``SpeakerCrossEntropyLoss``，
        因此分类头一旦构造便始终有非零 CE 梯度路径。当前所有合法配置下均
        不存在"可训练但无梯度路径"的参数，推导结果恒为 ``False``。函数保留 cue 配置读取仅用于冲突 warning 的
        诊断输出；未来若新增"构造但按 batch 条件绕过前向"的子模块，推导
        规则的扩展点在此。用户显式配置的值优先保留，仅在与推导冲突时打 warning。
        """
        model_args = (cfg.get("model") or {}).get("args") or {}
        joint, feats = _validate_tse_boolean_config(model_args)

        def _feature_enabled(name: str) -> bool:
            feature_cfg = feats.get(name)
            if feature_cfg is None:
                return False
            return config_bool(
                feature_cfg,
                "enabled",
                False,
                f"model.args.features.{name}",
            )

        context_on = _feature_enabled("context")
        spkemb_on = _feature_enabled("spkemb")
        # 所有可训练参数均在前向中使用，推导为 False。
        need_find_unused = False

        dist_cfg = cfg.setdefault("distributed", {})
        if "find_unused_parameters" not in dist_cfg:
            dist_cfg["find_unused_parameters"] = need_find_unused
        elif require_bool(
            dist_cfg["find_unused_parameters"],
            "distributed.find_unused_parameters",
        ) != need_find_unused:
            import logging

            logging.getLogger(__name__).warning(
                "distributed.find_unused_parameters=%s 与 TSE cue 配置推导结果 "
                "(%s) 不一致，保留显式配置。joint_training=%s, context=%s, spkemb=%s",
                dist_cfg["find_unused_parameters"], need_find_unused,
                joint, context_on, spkemb_on,
            )

    # ------------------------------------------------------------------
    # multi-task speaker 分类配置推导（仿 AEC _resolve_state_detector_cfg 范例）
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_spk_classifier_cfg(
        cfg: dict,
        *,
        speaker_class_count: int | None = None,
    ) -> None:
        """按 loss 与训练 identity 映射推导 ``model.args.multi_task``。

        ``speaker_class_count`` 仅由训练启动路径的 train dataset
        ``speaker_identity_to_idx`` 派生。直接调用 ``build_model`` 时不提供该值，
        则 multi-task 模型必须在配置中显式给出 ``spks_in_train``。
        """
        has_spk_ce = _validate_spk_ce_loss_components(cfg)

        args = cfg.setdefault("model", {}).setdefault("args", {})
        model_type = (cfg.get("model") or {}).get("type")
        if not _tse_model_supports_multi_task(model_type):
            if has_spk_ce:
                raise ValueError(
                    f"model.type={model_type} 没有 speaker 分类头，不能与 "
                    "SpeakerCrossEntropyLoss 组合；请移除该 loss 分量，或改用 "
                    "支持 multi-task CE 的 TSE speaker cue 模型。"
                )
            unsupported = sorted(set(args) & {"multi_task", "spks_in_train"})
            if unsupported:
                raise ValueError(
                    f"model.type={model_type} 不接受 multi-task speaker 分类参数："
                    f"{unsupported}"
                )
            return
        if "multi_task" not in args:
            args["multi_task"] = has_spk_ce
        else:
            configured_multi_task = require_bool(
                args["multi_task"], "model.args.multi_task"
            )
            if configured_multi_task != has_spk_ce:
                raise ValueError(
                    f"model.args.multi_task={args['multi_task']} 与 loss 中 "
                    f"SpeakerCrossEntropyLoss 的存在性（has_spk_ce={has_spk_ce}）不一致"
                )
            args["multi_task"] = configured_multi_task

        if not args["multi_task"]:
            return
        if speaker_class_count is not None and speaker_class_count < 2:
            raise ValueError(
                "train dataset speaker_identity_to_idx must contain at least two "
                f"speakers for multi_task=true, got {speaker_class_count}"
            )
        if "spks_in_train" not in args:
            if speaker_class_count is None:
                raise ValueError(
                    "multi_task=true requires model.args.spks_in_train when build_model "
                    "is called outside the training startup path"
                )
            args["spks_in_train"] = speaker_class_count
        elif (
            speaker_class_count is not None
            and int(args["spks_in_train"]) != speaker_class_count
        ):
            raise ValueError(
                f"model.args.spks_in_train={args['spks_in_train']} conflicts with train "
                f"dataset speaker_identity_to_idx size {speaker_class_count}"
            )

    def build_loaders(self, cfg: dict, rank: int, local_rank: int, world_size: int):
        _reject_evaluation_only_config(cfg)
        from tasks.task_api import load_task_data_api

        data_api = load_task_data_api("tse")

        data_cfg = cfg["data"]
        data_mode = (data_cfg.get("mode") or "").strip().lower()
        if data_mode not in (
            DATA_MODE_DYNAMIC_TSE,
            DATA_MODE_OFFLINE_TSE,
        ):
            raise ValueError(
                "TSE training only supports data.mode 'dynamic_tse' / "
                f"'offline_tse' (got {data_mode!r})."
            )
        if (data_cfg.get("train") or {}).get("sources"):
            raise ValueError(
                "data.train.sources is not supported for TSE dynamic training; "
                "use a single data.train block."
            )

        train_ds = data_api.build_dataset(data_cfg, mode="train")
        val_ds = data_api.build_dataset(data_cfg, mode="val", meta=cfg.get("meta"))

        # validation 复用 train identity→label 映射，未知 identity 使用 -1。
        set_label_mapping = getattr(val_ds, "set_speaker_label_mapping", None)
        train_identity_map = getattr(train_ds, "speaker_identity_to_idx", None)
        if train_identity_map is not None and callable(set_label_mapping):
            val_ds.set_speaker_label_mapping(train_identity_map)
        self._tse_validation_dataset = val_ds

        train_collate_fn = partial(data_api.collate, cue_specs=train_ds.cue_specs)
        val_collate_fn = partial(data_api.collate, cue_specs=val_ds.cue_specs)
        return self._build_loaders_common(
            cfg, rank, local_rank, world_size,
            train_ds=train_ds,
            val_ds=val_ds,
            train_collate_fn=train_collate_fn,
            val_collate_fn=val_collate_fn,
            train_num_per_epoch=data_cfg.get("train", {}).get("num_per_epoch"),
            val_num_per_epoch=data_cfg.get("validation", {}).get("num_per_epoch"),
        )

    def build_model_for_training(
        self,
        cfg: dict,
        rank: int,
        *,
        train_dataset,
    ) -> "torch.nn.Module":
        _reject_evaluation_only_config(cfg)
        identity_to_idx = getattr(train_dataset, "speaker_identity_to_idx", None)
        speaker_class_count = (
            len(identity_to_idx) if identity_to_idx is not None else None
        )
        import tasks.tse.models
        from core.models.io_contract import inject_dataset_model_geometry
        inject_dataset_model_geometry(cfg, train_dataset)
        _inject_tse_scene_geometry(
            cfg,
            train_dataset,
            getattr(self, "_tse_validation_dataset", None),
        )
        model = self._build_model(
            cfg,
            rank,
            speaker_class_count=speaker_class_count,
        )
        from core.models.io_contract import validate_spatial_model_geometry
        from tasks.tse.model_io import (
            resolve_tse_model_input_fields,
            validate_tse_multichannel_contract,
        )
        from tasks.tse.trainer.tse_trainer import validate_tse_training_dataset_cues

        trainer_cfg = cfg.get("trainer", {})
        sources = {
            validate_tse_training_dataset_cues(
                model,
                train_dataset,
                trainer_cfg,
                split="train",
            )
        }
        validation_dataset = getattr(self, "_tse_validation_dataset", None)
        if validation_dataset is not None:
            sources.add(
                validate_tse_training_dataset_cues(
                    model,
                    validation_dataset,
                    trainer_cfg,
                    split="validation",
                )
            )
        model_fields = resolve_tse_model_input_fields(model)
        if "waveform" in sources and model_fields & {"spk_embedding", "speaker_embedding"}:
            resolve_external_speaker_checkpoint((cfg.get("model") or {}).get("args") or {})

        validate_spatial_model_geometry(model, train_dataset)
        if _tse_dataset_is_multichannel(train_dataset):
            validate_tse_multichannel_contract(model)
        return model

    def build_model(self, cfg: dict, rank: int) -> "torch.nn.Module":
        return self._build_model(cfg, rank, speaker_class_count=None)

    def _build_model(
        self,
        cfg: dict,
        rank: int,
        *,
        speaker_class_count: int | None,
    ) -> "torch.nn.Module":
        import tasks.tse.models as tse_models
        from core.models.build import build_model
        import core.losses  # noqa: F401 -- registers shared loss classes before
        # run_training() calls
        # core.losses.build.build_loss().

        model_cfg = cfg.get("model", {})
        stft_cfg = cfg.get("stft")

        # 先注册模型类，再按类级 ``supports_multi_task`` 能力推导分类头参数注入。
        _mt = model_cfg.get("type")
        if _mt:
            tse_models.register_tse_models(only=str(_mt))
        else:
            tse_models.register_tse_models()

        self._resolve_spk_classifier_cfg(
            cfg,
            speaker_class_count=speaker_class_count,
        )
        self._resolve_tse_distributed_cfg(cfg)
        model = build_model(model_cfg, stft_cfg=stft_cfg)
        validate_dual_objective_model(
            model, resolve_dual_self_estimated_objective(cfg.get("trainer", {}))
        )
        if rank == 0:
            model_desc = model_cfg.get("type") or model_cfg.get("path", "model")
            self.logger.info(f"Model: {model_desc} | {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
        return model

    def get_trainer_cls(self):
        from tasks.tse.trainer.tse_trainer import TSETrainer

        return TSETrainer
