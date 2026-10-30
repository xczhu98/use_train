"""构建 ``data.mode: dynamic_aec`` 的 ``DynamicMixingDataset``（供 ``train_aec`` 使用）。

``strict_load`` / ``placeholder_num_samples`` 与 ``tasks.se.data.build_dataset`` 相同：通过
``resolve_strict_load`` / ``resolve_placeholder_num_samples`` 扫描 ``data`` 与根 config（data 优先）。
"""

from __future__ import annotations

import copy
import difflib
from dataclasses import replace
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

from tasks.aec.data.dynamic_dataset import DynamicMixingDataset, validate_aec_dataset_config
from core.data.constants import normalize_data_mode
from core.data.dataset_contract import (
    attach_dataset_contract,
    build_aec_dataset_contract,
    infer_dataset_sample_rate,
)
from core.data.dataset_build import (
    VALIDATION_DYNAMIC_DATA_SOURCE_KEYS,
    DatasetBuildSpec,
    SamplingCapabilities,
    expand_simulation_cfg_aec_ref_echo_dataset_keys,
    expand_simulation_cfg_layer_a,
    instantiate_dynamic_dataset,
    resolve_dynamic_dataset_build_request,
)

if TYPE_CHECKING:
    from torch.utils.data import Dataset

# AEC 的动态 speech manifest 支持 language/dataset/speaker 联合控制；speaker
# 使用 canonical ``(dataset_id, speaker_id)`` identity。noise 控制保持独立。
_AEC_SAMPLING_CAPABILITIES = SamplingCapabilities(
    speech_language=True,
    speech_dataset=True,
    speech_speaker=True,
    noise=True,
)

_AEC_RESOLVE_SPEC = DatasetBuildSpec(
    task="aec",
    dynamic_dataset_cls=DynamicMixingDataset,
    simulation_cfg_expanders=(
        expand_simulation_cfg_layer_a,
        expand_simulation_cfg_aec_ref_echo_dataset_keys,
    ),
    simulation_cfg_strict_from_cfg=True,
    sampling_capabilities=_AEC_SAMPLING_CAPABILITIES,
    dynamic_validation_default=True,
    validation_data_source_keys=VALIDATION_DYNAMIC_DATA_SOURCE_KEYS
    + ("real_farend_echo_pair_json",),
)


def _param_names(callable_obj) -> set[str]:
    """Return the set of named parameters for a callable, excluding self/args/kwargs."""
    return {
        p.name
        for p in inspect.signature(callable_obj).parameters.values()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and p.name != "self"
    }


# Schema caches: validation path and real construction path should agree on what
# keys are valid.  We compare resolved kwargs against the union of parameters
# accepted by the validation function and the Dataset constructor so that
# Dataset-only keys (e.g. real_farend_echo_pair_json, strict_load) are not
# reported as typos.
_VALIDATE_PARAMS = _param_names(validate_aec_dataset_config)
_DATASET_PARAMS = _param_names(DynamicMixingDataset.__init__)
try:
    from core.data.dynamic_dataset_base import BaseDynamicMixingDataset

    _DATASET_PARAMS |= _param_names(BaseDynamicMixingDataset.__init__)
except Exception:  # pragma: no cover - best-effort fallback
    pass
_ALL_KNOWN_PARAMS = _VALIDATE_PARAMS | _DATASET_PARAMS


def _close_match(key: str, candidates: set[str], cutoff: float = 0.8) -> str | None:
    m = difflib.get_close_matches(key, candidates, n=1, cutoff=cutoff)
    return m[0] if m else None


def _validate_resolved_dataset_kwargs(ds_kw: dict, split: str) -> None:
    """确保配置预检与 ``DynamicMixingDataset`` 真实构造接受同一组键。"""
    unknown = sorted(set(ds_kw) - _ALL_KNOWN_PARAMS)
    if not unknown:
        return
    hints = []
    for key in unknown:
        match = _close_match(key, _ALL_KNOWN_PARAMS)
        hints.append(f"{key!r} (did you mean {match!r}?)" if match else repr(key))
    raise ValueError(
        f"data.{split} contains unknown DynamicMixingDataset key(s): "
        + ", ".join(hints)
    )


def _aec_yaml_value_is_set(v: object) -> bool:
    """非空配置值：视为用户有意设置；与 ``simulation_cfg`` 内废弃键 fail-fast 一致。"""
    if v is None:
        return False
    if isinstance(v, str) and not v.strip():
        return False
    if isinstance(v, (list, tuple, dict, set)) and len(v) == 0:
        return False
    return True


def _raise_removed_aec_yaml_key(
    key: str,
    value: object,
    *,
    split: str,
    hint: str,
) -> None:
    """与 ``DynamicMixingDataset`` 内 ``aec_scenario_probs`` / ``prob_pure_echo`` 相同策略：明确报错。"""
    raise ValueError(
        f"data.{split} key {key!r} was removed and is no longer supported. "
        f"Your value was {value!r}. {hint}"
    )


def _reject_retired_aec_frontend_dataset_keys(section: dict, split: str) -> None:
    """Dataset 侧旧 AEC frontend 入口一律拒绝，避免绕过 Entry/Trainer。"""
    retired = {
        "aec_stream_align": (
            "请改用顶层 aec_frontend.delay_alignment.enabled；AEC input/deployment frontend 仅由 Trainer/Inference 执行。"
        ),
        "linear_aec": (
            "请改用顶层 aec_frontend.linear_aec；AEC input/deployment frontend 仅由 Trainer/Inference 执行。"
        ),
    }
    simulation_cfg = section.get("simulation_cfg")
    for key, hint in retired.items():
        if key in section:
            value = section.pop(key)
            _raise_removed_aec_yaml_key(key, value, split=split, hint=hint)
        if isinstance(simulation_cfg, dict) and key in simulation_cfg:
            value = simulation_cfg.pop(key)
            raise ValueError(
                f"data.{split}.simulation_cfg.{key} was removed and is no longer supported. "
                f"Your value was {value!r}. {hint}"
            )


def _pop_legacy_aec_data_keys(section: dict, split: str) -> None:
    """旧版路径/清单键名：非空值则抛 ``ValueError``（与 simulation_cfg 废弃键一致，fail-fast）。"""
    legacy_hints = (
        (
            "farend_scp",
            "Farend/ref 列表请用 real_farend_echo_pair_json（stereo 路径字符串）与 speech_scp 等当前字段。",
        ),
        (
            "real_echo_manifest_csv",
            "请改用 real_farend_echo_pair_json（JSON manifest，见 DynamicMixingDataset 文档）。",
        ),
        (
            "real_mic_scp",
            "近端/干净语音请用 speech_scp；回声对请用 real_farend_echo_pair_json。",
        ),
        (
            "real_farend_scp",
            "请用 real_farend_echo_pair_json 描述 farend/echo 对，勿单独使用 real_farend_scp。",
        ),
        (
            "real_echo_scp",
            "请用 real_farend_echo_pair_json 描述回声参考对。",
        ),
        (
            "real_farend_echo_pair_scp",
            "该旧配置别名已移除；请使用 real_farend_echo_pair_json。",
        ),
        (
            "aec_farend_echo_dual_file",
            "双文件 pair schema 已移除；manifest 的 pairs 每项必须是 stereo 文件路径字符串。",
        ),
        (
            "real_nearend_scp",
            "近端请用 speech_scp；若需真实对见 real_farend_echo_pair_json。",
        ),
        (
            "use_real_farend_echo_pair_prob",
            "请用 simulation_cfg 中 prob_echo_only / prob_target_echo / prob_target_noise_echo 等场景概率控制回声比例。",
        ),
        (
            "real_echo_prob",
            "请用 simulation_cfg 中的场景概率（prob_echo_only、prob_noise_echo_only、prob_target_echo、prob_target_noise_echo）控制回声训练比例。",
        ),
    )
    for legacy, hint in legacy_hints:
        if legacy not in section:
            continue
        val = section.pop(legacy)
        _raise_removed_aec_yaml_key(legacy, val, split=split, hint=hint)


def _pop_meta_keys_for_dataset_kwargs(section: dict) -> None:
    """从 section 去掉仅用于 YAML/构建期、不传入 Dataset 的键（静默）。"""
    for k in (
        "return_noise",
        "collate_fn",
        "sources",
        "weight",
        "inherit_train_simulation",
        "inherit_simulation",
        "num_per_epoch",
        "mode",
        "data_root",
        "meta_csv",
        "manifest_csv",
        "nearend_dir",
        "farend_dir",
        "real_doubletalk",
        "aec_add_echo_prob",
        "aec_echo_cancel_training_prob",
        "real_farend_echo_pair_json",
    ):
        section.pop(k, None)


def _pop_deprecated_top_level_aec_keys(section: dict, split: str) -> None:
    """顶层曾用名：Dataset 构造函数不再接收；非空则 ``ValueError``。"""
    deprecated = {
        "aec_pure_echo_prob": (
            "请用 simulation_cfg（如 prob_echo_only、ser_*、Layer B 七项 prob_*）表达纯回声/回声场景；"
            "顶层 aec_pure_echo_prob 已废弃。"
        ),
        "aec_echo_simulation": (
            "请将回声仿真参数写入 simulation_cfg 或 aec_dynamic_pipeline；顶层 aec_echo_simulation 已废弃。"
        ),
        "aec_ref_echo_stereo_wav": (
            "立体声 ref/echo 固定由 canonical manifest 路径读取；该顶层开关已废弃。"
        ),
        "aec_stereo_echo_channel": (
            "声道索引由数据集根据 manifest 解析；顶层 aec_stereo_echo_channel 已废弃。"
        ),
        "aec_stereo_farend_channel": (
            "声道索引由数据集根据 manifest 解析；顶层 aec_stereo_farend_channel 已废弃。"
        ),
    }
    for key, hint in deprecated.items():
        if key not in section:
            continue
        val = section.pop(key)
        if _aec_yaml_value_is_set(val):
            _raise_removed_aec_yaml_key(key, val, split=split, hint=hint)


def _to_plain_dict(obj) -> dict:
    """Best-effort conversion to a plain, recursively-converted dict.

    ``build_dynamic_aec_dataset`` expects ``cfg`` to already be a plain dict
    (normally guaranteed by ``entry.py``'s ``OmegaConf.to_container(user_cfg,
    resolve=True)`` at load time). Calling ``OmegaConf.to_container()``
    unconditionally here breaks that normal path (it raises on plain dict
    input -- see 0a14b3d, which reverted an earlier unconditional attempt).
    Guard with an isinstance check instead, so callers that bypass entry.py
    and pass a raw ``DictConfig``/``ListConfig`` section still get a fully
    converted dict. Plain mappings are deep-copied because validation removes
    retired keys before raising and must never mutate the caller's config.
    """
    if obj is None:
        return {}
    if isinstance(obj, DictConfig) or OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    return copy.deepcopy(dict(obj))


def _require_dynamic_aec_mode(data_cfg: dict) -> None:
    """AEC specialized API 不接受共享 SE 的 generic dynamic/offline mode。"""
    normalized = normalize_data_mode(data_cfg.get("mode"))
    if normalized.task != "aec":
        raise ValueError(
            f"data.mode {normalized.task_mode!r} belongs to task "
            f"{normalized.task!r}, not 'aec'"
        )
    if normalized.storage != "dynamic":
        raise ValueError(
            f"data.mode {normalized.task_mode!r} is not a dynamic AEC mode"
        )


def resolve_aec_dataset_kwargs(
    cfg: dict, split: str, stft_cfg: dict | None = None
) -> dict:
    """解析 ``data.{split}`` section 为 ``DynamicMixingDataset`` 构造 kwargs。

    纯配置解析，不触碰文件系统（不读 scp / 不检查 manifest 文件是否存在）。
    ``build_dynamic_aec_dataset``（真实构建）与 ``validate_aec_dynamic_aec_config``
    （只做配置检查）共用本函数，保证两条路径看到同一份解析结果。
    """
    cfg = _to_plain_dict(cfg)
    data_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else cfg
    _require_dynamic_aec_mode(data_cfg)
    _reject_retired_aec_frontend_dataset_keys(data_cfg, "data")
    request = resolve_dynamic_dataset_build_request(
        cfg,
        split,
        spec=_AEC_RESOLVE_SPEC,
    )
    ds_kw = dict(request.dataset_kwargs)

    _reject_retired_aec_frontend_dataset_keys(ds_kw, split)
    _pop_legacy_aec_data_keys(ds_kw, split)
    real_pair_list = ds_kw.pop("real_farend_echo_pair_json", None)
    for _echo_key in ("aec_add_echo_prob", "aec_echo_cancel_training_prob"):
        for _src in (data_cfg, ds_kw):
            if _echo_key in _src and _aec_yaml_value_is_set(_src.get(_echo_key)):
                _raise_removed_aec_yaml_key(
                    _echo_key,
                    _src[_echo_key],
                    split=split,
                    hint=(
                        "请改用 simulation_cfg 的 Layer B 场景概率（prob_echo_only / prob_noise_echo_only / "
                        "prob_target_echo / prob_target_noise_echo 等）控制是否走清单回声与 C-aec。"
                    ),
                )
    if not real_pair_list:
        raise ValueError(
            f"data.mode=dynamic_aec requires real_farend_echo_pair_json in data.{split} "
            "(.json manifest with non-empty stereo paths; ch0=echo, ch1=farend)."
        )
    _pop_meta_keys_for_dataset_kwargs(ds_kw)
    _pop_deprecated_top_level_aec_keys(ds_kw, split)

    # 仅显式启用 state detector 时注入对齐的帧级标签参数。
    if stft_cfg is not None and _aec_yaml_value_is_set(
        ds_kw.get("state_detector_label_type")
    ):
        if stft_cfg.get("center", True) is False:
            raise ValueError(
                "AEC state detector frame labels require stft.center=true; "
                "stft.center=false is unsupported."
            )
        hop = stft_cfg.get("hop_length")
        win = stft_cfg.get("win_length")
        if hop is not None:
            explicit_hop = ds_kw.get("state_detector_hop_length")
            if _aec_yaml_value_is_set(explicit_hop) and int(explicit_hop) != int(hop):
                # 标签帧率由 state_detector_hop_length 决定，而 state head 输出与
                # padding 掩码使用 stft.hop_length；两者不一致会让标签与输出帧错位。
                raise ValueError(
                    f"data.{split}.state_detector_hop_length={explicit_hop} must equal "
                    f"stft.hop_length={hop}: state labels are frame-aligned with the "
                    "model STFT frames."
                )
            ds_kw.setdefault("state_detector_hop_length", hop)
        if win is not None:
            ds_kw.setdefault("state_detector_win_length", win)

    # Task mode: "joint" (default, AEC+SE) or "aec" (echo cancellation only)
    task = ds_kw.pop("task", data_cfg.get("task", "joint"))
    ds_kw["real_farend_echo_pair_json"] = str(Path(real_pair_list).expanduser())
    ds_kw["task"] = task
    # 顶层 AEC input/deployment frontend 配置由 Entry/Trainer 统一解析；Dataset kwargs 只描述 A–F 声学仿真。
    return ds_kw


def validate_aec_dynamic_aec_config(cfg: dict) -> None:
    """“只做配置检查”路径：校验 dynamic_aec 的 train/val dataset 配置。

    与真实构建（``DynamicMixingDataset.__init__``）调用**同一个**
    ``validate_aec_dataset_config``，保证两条路径对同一份 YAML 判定一致。
    """
    stft_cfg = cfg.get("stft")
    for split in ("train", "val"):
        ds_kw = resolve_aec_dataset_kwargs(cfg, split, stft_cfg=stft_cfg)
        _validate_resolved_dataset_kwargs(ds_kw, split)
        validate_aec_dataset_config(**ds_kw)


def build_dynamic_aec_dataset(
    cfg: dict, split: str, stft_cfg: dict | None = None
) -> "Dataset":
    """从整包 ``cfg``（含 ``cfg["data"]``）构建 AEC 动态混合 Dataset。

    ``split`` 为 ``"train"`` 或 ``"val"``（``val`` 对应 ``data.validation``；旧 ``data.val`` 别名已退役且 fail-fast，与 ``train_aec`` 一致）。
    若提供 ``stft_cfg`` 且配置显式设置了 ``state_detector_label_type``，则
    ``state_detector_hop_length`` / ``state_detector_win_length``
    默认从 ``stft_cfg.hop_length`` / ``stft_cfg.win_length`` 继承；
    ``state_detector_hop_length`` 显式值必须等于 ``stft_cfg.hop_length``（帧率对齐
    模型 STFT 帧，不一致直接报错），``state_detector_win_length`` 仍显式值优先。
    同时要求 ``stft_cfg.center`` 不为 ``false``；未设置 label_type 时不注入也不执行
    state detector 专属门禁，避免影响不消费 state label 的模型。
    """
    ds_kw = resolve_aec_dataset_kwargs(cfg, split, stft_cfg=stft_cfg)
    _validate_resolved_dataset_kwargs(ds_kw, split)
    spec = (
        _AEC_RESOLVE_SPEC
        if DynamicMixingDataset is _AEC_RESOLVE_SPEC.dynamic_dataset_cls
        else replace(_AEC_RESOLVE_SPEC, dynamic_dataset_cls=DynamicMixingDataset)
    )
    dataset = instantiate_dynamic_dataset(
        spec,
        ds_kw,
        validator=validate_aec_dataset_config,
    )
    dataset._simulation_split = "train" if split == "train" else "validation"
    attach_dataset_contract(
        dataset,
        build_aec_dataset_contract(infer_dataset_sample_rate(dataset, cfg)),
    )
    return dataset
