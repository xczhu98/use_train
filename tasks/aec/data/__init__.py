"""AEC Data module: dataset construction, collate, samplers.

与 ``tasks/se/data/__init__.py`` 共用 ``core.data.dataset_build.build_dataset`` 主体，
仅 AEC 专用接线不同（``dynamic_aec`` mode, farend/echo pair manifests, LAEC error stems）。

AEC（Acoustic Echo Cancellation）数据模式（由 ``config.data.mode`` 指定）：
  - ``dynamic_aec``：使用 farend/echo pair 资源的在线动态仿真，复用 C-aec 与共享 A–F 层
  - ``offline_aec``：读取预混合 mic/farend/nearend 记录，使用 cap-only 长度语义并支持 spatial mic

验证集（``data.validation``）：
  - Offline：validation mode 为 ``offline_aec``（或显式 AEC offline contract）时，从
    mic/farend/nearend manifest 构造 ``AECOfflineDataset``，使用 cap-only 长度语义。
  - Dynamic：validation mode 为 ``dynamic_aec``（或 ``use_dynamic: true``）时，构造
    ``DynamicMixingDataset``。仿真参数：
    - inherit_simulation: true (default): same simulation as training, only data-source keys
      (speech_scp, noise_scp, rir_scp, windnoise_scp, ...) can be overridden in validation.
    - inherit_simulation: false: validation may override any simulation param; unspecified
      keys stay from data.train; simulation_cfg is deep-merged so you can override e.g.
      only snr_low/snr_high for val.

风噪（``windnoise_scp``）来源可选：当 ``noise_scp`` 是分类 JSONL 时，按记录内嵌的
``roles``（无该字段时回退 ``category_name``）提取 ``wind`` 角色，包括 ``Wind`` /
``Wind noise (microphone)`` 等类别；若显式提供 ``windnoise_scp``，两者合并，内嵌
风噪条目同时从普通加性噪声池剔除。省略 ``windnoise_scp`` 且 ``noise_scp`` 不是
分类 JSONL 时，仅当 ``prob_wind_noise_sidechain > 0`` 才在构造期拒绝；风噪未启用
时，未开启噪声类别均衡的普通严格四列 SCP 等非分类清单仍可使用。side-chain
启用时，显式清单与内嵌角色合并后的风噪池必须非空；分类 JSONL 无风噪角色，或
显式清单为空时，同样在构造期拒绝。噪声类别均衡另有独立约束：ordinary pool
每条记录必须具有非负 ``category_id`` 和非空 ``balance_category``（兼容
``category_name``）；该训练约束不要求 Noise Filter strict manifest schema v6。

当前 ``AECTrainingEntry`` 主路径按 ``data.mode`` 直接使用
``build_dynamic_aec_dataset`` / ``build_offline_aec_dataset``。公开的
``tasks.aec.data.build_dataset`` 仍是受支持的通用/legacy 兼容 API，供历史调用方与审计工具
构建 AEC 数据集；它委托共享 builder 并注入 AEC 专用 ``DatasetBuildSpec``，不得因训练
入口不直接调用而删除。
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

from core.data.constants import DEFAULT_PLACEHOLDER_NUM_SAMPLES  # noqa: F401
from core.data.dataset_build import data_plane_for_build
from core.data.dataset_contract import (
    attach_dataset_contract,
    build_aec_dataset_contract,
    infer_dataset_sample_rate,
)
from core.data.dataset_build import expand_simulation_cfg_aec_ref_echo_dataset_keys
from core.data.dataset_build import expand_simulation_cfg_layer_a
from tasks.aec.data.dynamic_dataset import DynamicMixingDataset  # noqa: F401
from core.data.offline_dataset import OfflineDataset  # noqa: F401
from tasks.aec.data.offline_dataset import AECOfflineDataset  # noqa: F401
from tasks.aec.data.offline_dataset import build_offline_aec_dataset  # noqa: F401
from core.data.collate import collate_fn  # noqa: F401
from core.data.sampler import GroupedBatchSampler  # noqa: F401

if TYPE_CHECKING:
    from torch.utils.data import Dataset

from core.data.dataset_build import (
    VALIDATION_DYNAMIC_DATA_SOURCE_KEYS as _VALIDATION_DYNAMIC_DATA_SOURCE_KEYS,
    VALIDATION_META_KEYS as _VALIDATION_META_KEYS,
    OFFLINE_ONLY_KEYS as _OFFLINE_ONLY_KEYS,
    DatasetBuildSpec,
    OfflineDatasetRequest,
    build_dataset as _build_dataset_impl,
    deep_merge as _deep_merge,
    apply_duration_filter_config as _apply_duration_filter_config,
    get_train_section as _get_train_section,
)


from core.data.config_flags import (
    config_layers_for_data_flags as _config_layers_for_data_flags,
    resolve_placeholder_num_samples as _resolve_placeholder_num_samples,
    resolve_strict_load as _resolve_strict_load,
)


def _build_offline_aec(request: OfflineDatasetRequest) -> AECOfflineDataset:
    return AECOfflineDataset(
        **request.section,
        target_sr=request.target_sr,
        strict_load=request.strict_load,
        simulation_cfg_strict=request.simulation_cfg_strict,
        placeholder_num_samples=request.placeholder_num_samples,
        seed=request.seed,
    )


_AEC_BUILD_SPEC = DatasetBuildSpec(
    task="aec",
    dynamic_dataset_cls=DynamicMixingDataset,
    offline_dataset_factory=_build_offline_aec,
    simulation_cfg_expanders=(
        expand_simulation_cfg_layer_a,
        expand_simulation_cfg_aec_ref_echo_dataset_keys,
    ),
    simulation_cfg_strict_from_cfg=True,
)


def build_dataset(cfg: dict, mode: str = "train", meta: Optional[dict] = None) -> Dataset:
    """Build an AEC dataset through the retained compatibility API.

    ``AECTrainingEntry`` uses the mode-specific builders directly, but external callers may
    continue to import this function.  It delegates to the shared implementation with the
    AEC-specific ``_AEC_BUILD_SPEC`` and is not a removal candidate.

    meta: optional dict (e.g. cfg["meta"]) used when mode='val' and validation is dynamic,
    to set DynamicMixingDataset base_seed (meta.get("seed", 0)); caller should pass it so
    val seed matches experiment seed when cfg is only the data section.

    When mode='val':
      - If validation uses dynamic (data.mode==dynamic and validation.mode=='dynamic' or use_dynamic: true),
        builds DynamicMixingDataset. When validation.inherit_simulation is true (default), only data-source
        keys are overridden; when false, validation may override any param (incl. simulation_cfg), rest from
        data.train (simulation_cfg deep-merged).
      - Otherwise builds OfflineDataset from cfg['validation']; clean_scp/noisy_scp must both
        use strict four-column ``uid fs length path`` records.
    When mode='train', selects dynamic or offline based on cfg['mode']; training config is read from
    data.train.

    Duration filter (optional): 在 data.train / data.validation 中配置
    filter_min_duration、filter_max_duration（单位：秒）时，会在 Dataset 初始化解析 clean 阶段
    过滤过短/过长的音频；未配置则不过滤。单条样本长度上限用 max_duration_seconds（秒）配置。
    Filter 使用 SCP 内嵌的原始采样率与长度，先于重采样执行。

    Sample rate:
      - data.sample_rate → 模型训练目标采样率 target_sr（动态/离线读入后统一到该 SR）。
      - 离线轨清单内的 `fs` 表示音频磁盘采样率。

    strict_load / placeholder_num_samples:
      可写在 data 顶层；整包 config 时亦可写在根上（data 覆盖根 config）。旧 ``data.data``
      内层已退役，出现即 fail-fast。
    strict_load (default True):
      - True: 文件读失败、波形 NaN/Inf、动态混合失败 → 抛错（推荐）。
      - False: 离线可 fallback 另一条后再静音占位；动态混合失败返回静音占位（旧 DDP 行为）。
    """
    spec = (
        _AEC_BUILD_SPEC
        if DynamicMixingDataset is _AEC_BUILD_SPEC.dynamic_dataset_cls
        else replace(_AEC_BUILD_SPEC, dynamic_dataset_cls=DynamicMixingDataset)
    )
    dataset = _build_dataset_impl(cfg, mode=mode, meta=meta, spec=spec)
    attach_dataset_contract(
        dataset,
        build_aec_dataset_contract(infer_dataset_sample_rate(dataset, cfg)),
    )
    return dataset


# train_aec / custom builders: same duration-filter helper as build_dataset
apply_duration_filter_config = _apply_duration_filter_config

from tasks.aec.data.dynamic_aec_build import build_dynamic_aec_dataset  # noqa: E402
