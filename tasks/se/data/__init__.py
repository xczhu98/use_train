"""SE Data module: dataset construction, collate, samplers.

SE (Speech Enhancement) data modes (via config data.mode):
  - dynamic: online dynamic mixing (DynamicMixingDataset), real-time synthesis of noisy audio
  - offline: offline pre-mixed (OfflineDataset), uses pre-generated clean/noisy pairs

Validation (data.validation):
  - Offline: when validation has no mode/use_dynamic, or mode == "offline", build OfflineDataset
    from strict four-column clean_scp / noisy_scp plus max_duration_seconds etc.
  - Dynamic: when data.mode is dynamic and validation.mode == "dynamic" (or use_dynamic: true),
    build DynamicMixingDataset. Simulation params:
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

For AEC training data (``dynamic_aec`` / ``tasks.aec.data.dynamic_dataset.DynamicMixingDataset``), see ``tasks.aec.data``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional

from core.data.collate import collate_fn  # noqa: F401
from core.data.constants import DEFAULT_PLACEHOLDER_NUM_SAMPLES  # noqa: F401
from core.data.dataset_contract import (
    attach_dataset_contract,
    build_se_dataset_contract,
    infer_dataset_sample_rate,
)
from core.data.offline_dataset import OfflineDataset  # noqa: F401
from core.data.sampler import GroupedBatchSampler  # noqa: F401
from tasks.se.data.dynamic_dataset import DynamicMixingDataset  # noqa: F401

if TYPE_CHECKING:
    from torch.utils.data import Dataset

from core.data.dataset_build import (
    VALIDATION_DYNAMIC_DATA_SOURCE_KEYS as _VALIDATION_DYNAMIC_DATA_SOURCE_KEYS,
    VALIDATION_META_KEYS as _VALIDATION_META_KEYS,
    OFFLINE_ONLY_KEYS as _OFFLINE_ONLY_KEYS,
    AEC_REF_ECHO_DATASET_KEYS as _AEC_REF_ECHO_DATASET_KEYS,
    DatasetBuildSpec,
    build_dataset as _build_dataset_impl,
    deep_merge as _deep_merge,
    apply_duration_filter_config as _apply_duration_filter_config,
    expand_simulation_cfg_layer_a,
    expand_simulation_cfg_se_pase_dataset_keys,
    expand_simulation_cfg_aec_ref_echo_dataset_keys,
    data_plane_for_build as _data_plane_for_build,
    get_train_section as _get_train_section,
)


from core.data.config_flags import (
    config_layers_for_data_flags as _config_layers_for_data_flags,
    resolve_placeholder_num_samples as _resolve_placeholder_num_samples,
    resolve_simulation_cfg_strict,
    resolve_strict_load as _resolve_strict_load,
)


def _se_build_spec() -> DatasetBuildSpec:
    """按调用时绑定 Dataset 类，保留测试/工具替换构造器的能力。"""
    return DatasetBuildSpec(
        task="se",
        dynamic_dataset_cls=DynamicMixingDataset,
        simulation_cfg_expanders=(
            expand_simulation_cfg_layer_a,
            expand_simulation_cfg_se_pase_dataset_keys,
        ),
        simulation_cfg_strict_from_cfg=True,
    )


def build_dataset(cfg: dict, mode: str = "train", meta: Optional[dict] = None) -> Dataset:
    """Build SE dataset from config.

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
    过滤过短/过长的音频；未配置则不过滤。动态模式 Layer A 仿真参数与 B–F 均在
    ``data.train.simulation_cfg`` 顶层；
    ``max_duration_*`` / ``repeat_*`` 必须直接写在 data.train/data.validation；
    builder 不再从 simulation_cfg 隐式提升 Dataset lifecycle 参数。
    Filter 使用 SCP 内嵌的原始采样率与长度，先于重采样执行。

    Sample rate:
      - data.sample_rate → 模型训练目标采样率 target_sr（动态/离线读入后统一到该 SR）。

    simulation_cfg:
      ``SimulationConfigs.from_dict(..., strict=True)`` 恒为唯一模式，拼写错误的
      ``simulation_cfg`` 键一律抛错而非静默忽略；旧宽容开关 ``data.simulation_cfg_strict``
      已退役，配置中出现即 fail-fast。

    strict_load / placeholder_num_samples:
      可写在 data 顶层；整包 config 时亦可写在根上（data 覆盖根 config）。旧 ``data.data``
      内层已退役，出现即 fail-fast。
    strict_load (default True):
      - True: 文件读失败、波形 NaN/Inf、动态混合失败 → 抛错（推荐）。
      - False: 离线可 fallback 另一条后再静音占位；动态混合失败返回静音占位（旧 DDP 行为）。
    """
    dataset = _build_dataset_impl(cfg, mode=mode, meta=meta, spec=_se_build_spec())
    attach_dataset_contract(
        dataset,
        build_se_dataset_contract(infer_dataset_sample_rate(dataset, cfg)),
    )
    return dataset


# 与 ``_resolve_*`` 等价，供 ``train_aec`` / ``tasks.aec.data.dynamic_aec_build`` 等显式引用（整包 cfg 或仅 data 段均可）。
resolve_strict_load = _resolve_strict_load
resolve_placeholder_num_samples = _resolve_placeholder_num_samples
# 供 ``tasks.aec.data.build_dataset`` 与 SE 共用整包 YAML / 仅 data 段两种入口。
data_plane_for_build = _data_plane_for_build

# train_aec / custom builders: same duration-filter helper as build_dataset
apply_duration_filter_config = _apply_duration_filter_config
