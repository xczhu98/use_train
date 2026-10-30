"""TSE Data module: dataset construction, collate, batch contracts.

TSE (Target Speaker Extraction) 支持两种数据模式：

- ``data.mode: dynamic_tse``：在线动态仿真
  （``TSEDynamicMixingDataset``），七路 T/I/N 场景使用 core 逐源声学与 D–F；RIR 后标定 SIR，注册音独立处理。启用空间 observation 时由 scene geometry 生成 static/track per-target spatial cue，并同步提供 geometry、timebase 和 provenance；这是数据接线，不等于所有模型的 spatial 前向或 streaming 已完成。
- ``data.mode: offline_tse``：离线 WeSep manifest（``TSEOfflineDataset``），
  显式选择 manifest/path/target/mix/cue schema，支持无 cue、speaker waveform、
  speaker embedding 和 NPY spatial cue，并保留全部目标与多文件通道；visual/textual
  cue 仍在 Dataset 构造期 fail-fast。

两种模式均产出 canonical dict item，经 ``tse_collate_fn`` 形成 validated
``TSEBatch``；Dataset 的 ``cue_specs`` 是 ``DatasetContract`` typed cue 字段的唯一
来源，geometry/timebase/provenance 按 mixture 保留为 host metadata，不暴露 enrollment tuple。

TSE 动态模式只接受含 ``dataset_id`` / ``speaker_id`` 的 strict JSONL speech
manifest。``spk2enroll_json`` 属于 ``data.train`` /
``data.validation`` 数据资源字段；专属 Dataset 构造参数（
``enrollment_duration`` / ``enrollment_db`` /
``enroll_policy`` / ``enrollment_augment`` / ``targets_per_mixture``）写在
``simulation_cfg`` 顶层。由
``expand_simulation_cfg_tse_dataset_keys`` 提升，与 AEC ref/echo 键同一机制。

验证集动态构建、时长过滤、strict_load / placeholder 语义与 SE 一致，
见 ``core.data.dataset_build.build_dataset``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from tasks.tse.cue_contract import (  # noqa: F401
    TSECue,
    TSECueBatch,
    TSECueBatches,
    TSECueSpec,
    TSECueSpecs,
    TSECues,
)
from tasks.tse.data.batch import TSEBatch  # noqa: F401
from tasks.tse.data.collate import (  # noqa: F401
    make_tse_sample,
    make_tse_target,
    tse_collate_fn,
)
from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset  # noqa: F401

if TYPE_CHECKING:
    from torch.utils.data import Dataset

from core.data.constants import normalize_data_mode
from core.data.dataset_contract import (
    attach_dataset_contract,
    build_tse_dataset_contract,
    infer_dataset_sample_rate,
)
from core.data.dataset_build import (
    VALIDATION_DYNAMIC_DATA_SOURCE_KEYS,
    DatasetBuildSpec,
    OfflineDatasetRequest,
    SamplingCapabilities,
    build_dataset as _build_dataset_impl,
    expand_simulation_cfg_layer_a,
    expand_simulation_cfg_tse_dataset_keys,
    validate_tse_train_resource_paths,
)
from tasks.tse.data.offline_dataset import TSEOfflineDataset  # noqa: F401
from tasks.tse.data.offline_dataset import build_offline_tse_dataset


def _build_offline_tse(request: OfflineDatasetRequest) -> TSEOfflineDataset:
    return TSEOfflineDataset(
        **request.section,
        seed=request.seed,
        target_sr=request.target_sr,
        strict_load=request.strict_load,
        simulation_cfg_strict=request.simulation_cfg_strict,
        placeholder_num_samples=request.placeholder_num_samples,
    )


def _tse_build_spec() -> DatasetBuildSpec:
    """按调用时绑定 Dataset 类，保留测试/工具替换构造器的能力。"""
    return DatasetBuildSpec(
        task="tse",
        dynamic_dataset_cls=TSEDynamicMixingDataset,
        offline_dataset_factory=_build_offline_tse,
        simulation_cfg_expanders=(
            expand_simulation_cfg_layer_a,
            expand_simulation_cfg_tse_dataset_keys,
        ),
        simulation_cfg_strict_from_cfg=True,
        sampling_capabilities=SamplingCapabilities(
            speech_language=True,
            speech_dataset=True,
            speech_speaker=True,
            noise=True,
        ),
        validate_dynamic_train=validate_tse_train_resource_paths,
        validation_data_source_keys=VALIDATION_DYNAMIC_DATA_SOURCE_KEYS
        + ("spk2enroll_json",),
    )


def build_dataset(cfg: dict, mode: str = "train", meta: Optional[dict] = None) -> "Dataset":
    """Build TSE dataset from config（薄封装）。

    ``normalize_data_mode`` 统一解析公开 mode；动态路径由显式 TSE spec
    描述 Dataset、simulation key 提升、strict schema 与 sampling 能力。
    ``offline_tse`` 仍实例化独立 ``TSEOfflineDataset``，不改变输出格式。
    """
    plane = cfg.get("data") if isinstance(cfg.get("data"), dict) else cfg
    normalized = normalize_data_mode(plane.get("mode"))
    # TSE 只接受任务专用 mode；generic dynamic 由 SE 拥有，不在此入口消费。
    if normalized.task != "tse":
        raise ValueError(
            f"data.mode {normalized.task_mode!r} belongs to task "
            f"{normalized.task!r}, not 'tse'"
        )
    dataset = _build_dataset_impl(cfg, mode=mode, meta=meta, spec=_tse_build_spec())
    targets_per_mixture = int(getattr(dataset, "targets_per_mixture", 1))
    attach_dataset_contract(
        dataset,
        build_tse_dataset_contract(
            infer_dataset_sample_rate(dataset, cfg),
            variable_targets=targets_per_mixture > 1,
            cue_specs=getattr(dataset, "cue_specs", None),
        ),
    )
    return dataset
