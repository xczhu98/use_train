"""任务插件的静态描述。"""

from __future__ import annotations

from dataclasses import dataclass

from core.data.constants import (
    DATA_MODE_DYNAMIC, DATA_MODE_OFFLINE,
    DATA_MODE_DYNAMIC_AEC, DATA_MODE_OFFLINE_AEC,
    DATA_MODE_DYNAMIC_TSE, DATA_MODE_OFFLINE_TSE,
    DATA_MODE_DYNAMIC_SS, DATA_MODE_OFFLINE_SS,
    DATA_MODE_DYNAMIC_SSL,
)


@dataclass(frozen=True)
class TaskDescriptor:
    """描述任务级路由信息与能力概况。

    ``supports_streaming`` 和 ``supports_external_stft`` 表示任务是否提供相应
    能力的路径，不代表每个模型都支持；具体模型与输入组合仍由 IO contract
    和任务 adapter 在对应边界校验。
    """

    name: str
    data_modes: frozenset[str]
    training_entrypoint: str
    model_registry_module: str
    model_registrar: str
    config_root: str
    supports_streaming: bool
    supports_external_stft: bool
    file_inference_adapter_factory: str | None
    root_file_inference_error: str
    trainer_objective_validator: str | None = None
    data_module: str | None = None
    data_builder: str | None = "build_dataset"
    collate_function: str | None = None
    dataset_validator: str | None = None


TASK_DESCRIPTORS = (
    TaskDescriptor(
        name="se",
        data_modes=frozenset({DATA_MODE_DYNAMIC, DATA_MODE_OFFLINE}),
        training_entrypoint="tasks.se.trainer.entry.SETrainingEntry",
        model_registry_module="tasks.se.models",
        model_registrar="register_se_models",
        config_root="configs/se",
        supports_streaming=True,
        supports_external_stft=True,
        file_inference_adapter_factory="tasks.se.inference.adapter:build_inference_adapter",
        root_file_inference_error="",
        data_module="tasks.se.data",
        collate_function="collate_fn",
    ),
    TaskDescriptor(
        name="aec",
        data_modes=frozenset({DATA_MODE_DYNAMIC_AEC, DATA_MODE_OFFLINE_AEC}),
        training_entrypoint="tasks.aec.trainer.entry.AECTrainingEntry",
        model_registry_module="tasks.aec.models",
        model_registrar="register_aec_models",
        config_root="configs/aec",
        supports_streaming=True,
        supports_external_stft=True,
        file_inference_adapter_factory="tasks.aec.inference.adapter:build_inference_adapter",
        root_file_inference_error="",
        data_module="tasks.aec.data",
        collate_function="tasks.aec.data.collate:aec_collate_fn",
    ),
    TaskDescriptor(
        name="tse",
        trainer_objective_validator="tasks.tse.objectives:validate_tse_objective_config",
        data_modes=frozenset({DATA_MODE_DYNAMIC_TSE, DATA_MODE_OFFLINE_TSE}),
        training_entrypoint="tasks.tse.trainer.entry.TSETrainingEntry",
        model_registry_module="tasks.tse.models",
        model_registrar="register_tse_models",
        config_root="configs/tse",
        supports_streaming=True,
        supports_external_stft=True,
        file_inference_adapter_factory="tasks.tse.inference.adapter:build_inference_adapter",
        root_file_inference_error="",
        data_module="tasks.tse.data",
        collate_function="tasks.tse.data.collate:tse_collate_fn",
        dataset_validator="tasks.tse.data.interface:validate_tse_dataset_interface",
    ),
    TaskDescriptor(
        name="ss",
        data_modes=frozenset({DATA_MODE_DYNAMIC_SS, DATA_MODE_OFFLINE_SS}),
        training_entrypoint="tasks.ss.trainer.entry.SSTrainingEntry",
        model_registry_module="tasks.ss.models",
        model_registrar="register_ss_models",
        config_root="configs/ss",
        supports_streaming=False,
        supports_external_stft=True,
        file_inference_adapter_factory=None,
        root_file_inference_error=(
            "SS root file inference is unavailable; use tasks.ss.inference.SSExtractor "
            "or SSInferenceAdapter with a resolved SS execution plan."
        ),
        data_module="tasks.ss.data",
        collate_function="ss_collate_fn",
    ),
    TaskDescriptor(
        name="ssl",
        data_modes=frozenset({DATA_MODE_DYNAMIC_SSL}),
        training_entrypoint="tasks.ssl.trainer.entry.SSLTrainingEntry",
        model_registry_module="tasks.ssl.models",
        model_registrar="register_ssl_models",
        config_root="configs/ssl",
        supports_streaming=False,
        supports_external_stft=False,
        file_inference_adapter_factory=None,
        root_file_inference_error=(
            "SSL root file inference is unavailable; SSL exposes dynamic scene "
            "training and contract APIs through tasks.ssl, not a public audio-file runner."
        ),
        data_module="tasks.ssl.data",
        data_builder=None,
        collate_function="ssl_collate_fn",
    ),
)
