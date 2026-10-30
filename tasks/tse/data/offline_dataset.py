"""TSE 离线 WeSep 数据集。

外部 manifest 只经 ``wesep_adapter.load_wesep_records`` 严格导入；载体格式、
路径基准、mixture variant、目标选择和 cue schema 均由配置显式声明。Dataset
保留 manifest 中全部目标与有序音频路径，输出唯一 canonical dict item，并通过
``cue_specs`` 声明实际启用的 typed cue 表示。

已接线的 cue 解码为 speaker ``wav/waveform``、speaker ``npy/spk_embedding``、
spatial ``npy/fields``（static/track）、visual ``npy/muse_frontend`` 与 textual
``json/dae_keyword_phoneme``；raw ``mp4`` 需要可选视频后端，构造期显式拒绝。
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from core.data.constants import DEFAULT_PLACEHOLDER_NUM_SAMPLES
from core.data.simulation_configs import SimulationConfigs
from core.data.offline_multitrack import (
    BaseOfflineMultiTrackDataset,
    OfflineLifecycleOptions,
    ensure_finite,
    load_audio,
    load_audio_window,
    resample,
    resampled_num_frames,
)


def _strict_mono_waveform(waveform: "torch.Tensor", *, context: str) -> "torch.Tensor":
    """enrollment 合同只接受单通道；多通道在 Dataset 边界显式拒绝（与推理侧一致）。"""
    if waveform.ndim != 2:
        raise ValueError(
            f"{context} 期望 channel-first [C, T]，got {tuple(waveform.shape)}"
        )
    if waveform.shape[0] != 1:
        raise ValueError(
            f"{context} 只接受单通道注册音，got C={waveform.shape[0]}"
        )
    return waveform
from core.data.offline_spatial_geometry import resolve_offline_spatial_geometry
from core.data.resource_paths import resolve_config_resource_path, resolve_manifest_resource_path
from core.data.simulation.trajectory import TrajectoryTimebase
from core.data.simulation_utils import read_source_scp
from tools.shared.pickle_io import load_pkl_file
from tasks.tse.cue_contract import TSECue, TSECueSpec, TSECueSpecs, TSECues
from tasks.tse.data.collate import make_tse_sample, make_tse_target
from tasks.tse.data.dynamic_dataset import EnrollmentAugmentationProviderError
from tasks.tse.data.enrollment_augment import EnrollmentAugmentor
from tasks.tse.data.wesep_adapter import (
    MANIFEST_FORMAT_JSONL,
    MANIFEST_FORMAT_PATH_LIST,
    PATH_BASE_MANIFEST_RELATIVE,
    PATH_BASE_RECIPE_ROOT,
    WeSepCueSpec,
    WeSepSampleRecord,
    WeSepTargetRecord,
    load_wesep_records,
)

_log = logging.getLogger(__name__)

_ENROLL_POLICIES = ("random", "fixed")
_CUE_SCHEMAS = ("none", "realtse_raw", "wesep_v0_1")
_IMPLEMENTED_CUES = frozenset({"audio", "spatial", "visual", "textual"})
_SPATIAL_REPRESENTATIONS = frozenset({"static", "track"})
_SPATIAL_ANGLE_UNIT = "radian"
# spatial cue 方向角坐标系标识：array-local，azimuth=atan2(y, x)、elevation 从
# 水平面起算（弧度）。唯一定义见 tasks/tse/data/spatial_cues.py::
# azimuth_elevation_from_positions；与 SSL 的极角口径（projection._doa）
# 不得直接互换，换算 el_水平面 = π/2 - 极角。
_SPATIAL_COORDINATE_FRAME = "array_local_azimuth_elevation_from_xy"
_SPATIAL_TRACK_TIMEBASE = "source_samples"

_VISUAL_TYPE_FORMAT = ("npy", "muse_frontend")
_RAW_VIDEO_TYPE_FORMAT = ("mp4", "raw_video")
_VISUAL_REPRESENTATION = "features"
_TEXTUAL_TYPE_FORMAT = ("json", "dae_keyword_phoneme")
_TEXTUAL_SELECTIONS = ("random", "fixed")
# KCE 前端在 ``_build_keywords`` 里自行拼接 BOS/EOS，因此数据层只产出 keyword
# phoneme id 本身；这里登记前端保留 id，避免数据把保留位当普通 token 传出。
_TEXTUAL_PADDING_ID = 0
_TEXTUAL_BOS_ID = 71
_TEXTUAL_EOS_ID = 72
# visual fps 与 duration_sec 的容许偏差：一个视频帧周期。
_CLOCK_TOLERANCE = 1e-9


def _is_moving_rir_path(path: str) -> bool:
    """读取 RIR 元数据，识别静态卷积增强器不支持的 moving RIR。"""
    if Path(path).suffix.lower() != ".pkl":
        return False
    try:
        payload = load_pkl_file(
            path, purpose="inspect TSE offline enrollment RIR metadata"
        )
    except OSError:
        return False
    return isinstance(payload, dict) and bool(payload.get("moving", False))


def _enabled_cue_names(
    enabled_cues: Mapping[str, bool] | Iterable[str],
) -> tuple[str, ...]:
    if isinstance(enabled_cues, Mapping):
        values = []
        for name, enabled in enabled_cues.items():
            if not isinstance(name, str) or type(enabled) is not bool:
                raise TypeError("enabled_cues must map string names to exact bool values")
            if enabled:
                values.append(name)
    else:
        if isinstance(enabled_cues, (str, bytes)):
            raise TypeError("enabled_cues must be a mapping or iterable of cue names")
        values = list(enabled_cues)
    if any(not isinstance(name, str) or not name for name in values):
        raise TypeError("enabled cue names must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("enabled_cues contains duplicate names")
    order = ("audio", "spatial", "visual", "textual")
    unknown = sorted(set(values) - set(order))
    if unknown:
        raise ValueError(f"unknown enabled cues: {unknown}")
    selected = set(values)
    return tuple(name for name in order if name in selected)


def _load_legacy_raw_cue(cues_yaml: str) -> dict[str, Any]:
    """显式读取历史 REAL-TSE ``type: raw`` speaker waveform 索引。"""
    import yaml

    with open(cues_yaml, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    audio_cue = (cfg.get("cues") or {}).get("audio")
    if not isinstance(audio_cue, dict):
        raise ValueError(f"cues.yaml {cues_yaml!r}: missing 'cues.audio' section")
    if audio_cue.get("type") != "raw" or audio_cue.get("format") is not None:
        raise ValueError(
            "cue_schema='realtse_raw' requires cues.audio.type='raw' and no format; "
            "use cue_schema='wesep_v0_1' for type='wav'/format='waveform'"
        )
    if audio_cue.get("scope", "speaker") != "speaker":
        raise ValueError("realtse_raw cues.audio.scope must be 'speaker'")
    if audio_cue.get("guaranteed", True) is not True:
        raise ValueError("realtse_raw audio cue must be guaranteed")
    policy = audio_cue.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(f"cues.yaml {cues_yaml!r}: missing 'cues.audio.policy'")
    policy_type = policy.get("type")
    key_field = policy.get("key")
    if policy_type not in _ENROLL_POLICIES:
        raise ValueError(f"unsupported realtse_raw policy.type={policy_type!r}")
    if key_field not in ("spk_id", "mix_spk_id"):
        raise ValueError(f"unsupported realtse_raw policy.key={key_field!r}")
    resource_value = policy.get("resource")
    if not isinstance(resource_value, str) or not resource_value.strip():
        raise ValueError("realtse_raw policy.resource must be a non-empty path")
    resource_path = resolve_manifest_resource_path(resource_value, cues_yaml)
    with open(resource_path, encoding="utf-8") as stream:
        resource = json.load(stream)
    if not isinstance(resource, dict) or not resource:
        raise ValueError(f"cues resource {resource_path!r} must be a non-empty object")
    normalized: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for lookup_key, raw_items in resource.items():
        if not isinstance(lookup_key, str) or not lookup_key:
            raise ValueError("cues resource keys must be non-empty strings")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError(f"cues resource {lookup_key!r} must be a non-empty list")
        items = []
        for index, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, dict):
                raise ValueError(f"cues resource {lookup_key!r}[{index}] must be an object")
            path = raw_item.get("path")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"cues resource {lookup_key!r}[{index}] missing path")
            item = dict(raw_item)
            item["path"] = resolve_manifest_resource_path(path, resource_path)
            items.append(item)
        normalized[lookup_key] = tuple(items)
    return {
        "key_field": key_field,
        "policy_type": policy_type,
        "resource_path": resource_path,
        "resource": normalized,
    }


def _resolve_visual_clock(
    item: Mapping[str, Any], num_frames: int, *, source: str
) -> tuple[float, float]:
    """返回 ``(duration_sec, fps)``；两者必须自洽，缺失则显式报错。

    上游只声明 ``duration_sec``；本仓库额外接受显式 ``fps``，但两者同时出现时
    必须落在同一个视频帧周期内，避免按点数猜测时间轴。
    """
    duration = item.get("duration_sec")
    fps = item.get("fps")
    if duration is None and fps is None:
        raise ValueError(
            f"{source} must declare duration_sec or fps to build a visual timebase"
        )
    if fps is not None:
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"{source}.fps must be positive and finite")
    if duration is not None:
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"{source}.duration_sec must be positive and finite")
    if duration is None:
        return num_frames / fps, fps
    if fps is None:
        return duration, num_frames / duration
    if abs(num_frames / fps - duration) > 1.0 / fps + _CLOCK_TOLERANCE:
        raise ValueError(
            f"{source} fps={fps} and duration_sec={duration} disagree for "
            f"{num_frames} frames"
        )
    return duration, fps


def _textual_reserved_ids(spec: WeSepCueSpec) -> tuple[int, int, int]:
    """解析 textual cue 的 padding/BOS/EOS 保留 id。"""
    collate = spec.format_metadata.get("collate") or {}
    if not isinstance(collate, Mapping):
        raise TypeError("textual cue collate metadata must be a mapping")
    fill = collate.get("fill_value", _TEXTUAL_PADDING_ID)
    if type(fill) is not int or fill != _TEXTUAL_PADDING_ID:
        raise ValueError(
            "canonical textual cue zero-fills padding after lengths; "
            f"collate.fill_value must be {_TEXTUAL_PADDING_ID}, got {fill!r}"
        )
    bos = collate.get("bos_id", _TEXTUAL_BOS_ID)
    eos = collate.get("eos_id", _TEXTUAL_EOS_ID)
    for name, value in (("bos_id", bos), ("eos_id", eos)):
        if type(value) is not int or value < 0:
            raise ValueError(f"textual collate.{name} must be a non-negative int")
    if bos == eos:
        raise ValueError("textual collate.bos_id and collate.eos_id must differ")
    return fill, bos, eos


def _validate_textual_tokens(
    labels: Sequence[Sequence[int]], *, source: str, reserved: tuple[int, int, int]
) -> None:
    """校验 keyword phoneme id 不占用 canonical 保留位。"""
    padding, bos, eos = reserved
    for word_index, word in enumerate(labels):
        for token in word:
            if token in (padding, bos, eos):
                raise ValueError(
                    f"{source}.phn_label[{word_index}] contains reserved token "
                    f"{token} (padding/BOS/EOS are owned by the KCE frontend)"
                )


class TSEOfflineDataset(BaseOfflineMultiTrackDataset):
    """严格导入 WeSep 资源并产出 canonical dict item。

    ``manifest_format``、``path_base``、``target_selection``、``mix_variant`` 和
    ``cue_schema`` 均须显式给出。``cue_schema='wesep_v0_1'`` 接受 speaker
    ``wav/waveform``、speaker ``npy/spk_embedding``、spatial ``npy/fields``、
    visual ``npy/muse_frontend`` 与 textual ``json/dae_keyword_phoneme``；历史
    REAL-TSE ``type: raw`` 仅可通过 ``cue_schema='realtse_raw'`` 选择。
    ``cue_schema='none'`` 表示合法的无 cue Dataset。

    visual 只解码预提取 npy：item 的 ``duration_sec``/``fps`` 定义视频帧时间轴，
    crop 按 mixture source 秒域选择帧，typed cue 的 ``timebase`` 为相对 crop 起点
    的秒（float64），不按点数 resize；raw ``mp4`` 需要可选视频后端，构造期拒绝。
    textual 按 ``selection`` 的 ``random``/``fixed`` 策略把 keyword word 的 phoneme
    id 展平为 canonical ``phoneme_ids``，缺失 cue 保持 ``present=False``。

    ``max_duration`` 使用输出采样率域的样点数；``max_duration_seconds`` 优先，
    按输出采样率换算。窗口同时记录 source 与 output 起点/长度，末端窗口仍须
    满足完整输出长度；metadata、实际波形与 placeholder 复用同一输出域上限。
    """

    # cue 解析结果的声明式默认值：``__init__`` 会覆盖，未走构造的 partial 实例
    # （``object.__new__`` 夹具）也保持"未启用 cue"的确定语义。
    _visual_requested: bool = False
    _textual_requested: bool = False
    _visual_feature_dim: int | None = None
    _textual_selection: dict[str, Any] | None = None
    _textual_reserved: tuple[int, int, int] | None = None

    def __init__(
        self,
        samples_list=None,
        samples_jsonl=None,
        cues_yaml=None,
        *,
        manifest_format: str | None = None,
        path_base: str | None = None,
        target_selection: str | Mapping[str, Sequence[str]] | None = None,
        mix_variant: str | None = None,
        recipe_root: str | None = None,
        enabled_cues: Mapping[str, bool] | Iterable[str] = (),
        cue_schema: str | None = None,
        max_duration=-1,
        max_duration_seconds: Optional[float] = None,
        max_enroll_duration_seconds: Optional[float] = None,
        enroll_policy: Optional[str] = None,
        enrollment_augment: Mapping[str, object] | None = None,
        seed: Optional[int] = None,
        filter_by_duration: bool = False,
        filter_min_duration: float = 0.5,
        filter_max_duration: float = 30.0,
        target_sr=None,
        strict_load: bool = True,
        placeholder_num_samples: int = DEFAULT_PLACEHOLDER_NUM_SAMPLES,
        simulation_cfg: dict | None = None,
        simulation_cfg_strict: bool = True,
        spatial_enabled: bool = False,
        array_profile: str | None = None,
        custom_geometry=None,
        reference_channel: int = 0,
        channel_order=None,
        spatial_representation: str | None = None,
        spatial_angle_unit: str | None = None,
        spatial_coordinate_frame: str | None = None,
        spatial_speed_of_sound_mps: float | None = None,
        spatial_track_timebase: str | None = None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"TSEOfflineDataset got unexpected kwargs: {sorted(kwargs)}")
        carriers = [value for value in (samples_list, samples_jsonl) if value is not None]
        if len(carriers) != 1:
            raise ValueError(
                "TSEOfflineDataset requires exactly one of samples_list or samples_jsonl"
            )
        required = {
            "manifest_format": manifest_format,
            "path_base": path_base,
            "target_selection": target_selection,
            "mix_variant": mix_variant,
            "cue_schema": cue_schema,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(f"TSEOfflineDataset requires explicit parameters: {missing}")
        if manifest_format not in {MANIFEST_FORMAT_JSONL, MANIFEST_FORMAT_PATH_LIST}:
            raise ValueError("manifest_format must be 'jsonl' or 'path_list'")
        if path_base not in {PATH_BASE_MANIFEST_RELATIVE, PATH_BASE_RECIPE_ROOT}:
            raise ValueError("path_base must be 'manifest_relative' or 'recipe_root'")
        if cue_schema not in _CUE_SCHEMAS:
            raise ValueError(f"cue_schema must be one of {_CUE_SCHEMAS}, got {cue_schema!r}")
        enabled = _enabled_cue_names(enabled_cues)
        if cue_schema == "none" and (enabled or cues_yaml is not None):
            raise ValueError("cue_schema='none' requires enabled_cues=() and no cues_yaml")
        if cue_schema != "none" and not cues_yaml:
            raise ValueError(f"cue_schema={cue_schema!r} requires cues_yaml")
        if cue_schema == "realtse_raw" and enabled != ("audio",):
            raise ValueError("cue_schema='realtse_raw' requires enabled_cues=('audio',)")
        if enroll_policy is not None and enroll_policy not in _ENROLL_POLICIES:
            raise ValueError(
                f"enroll_policy must be one of {_ENROLL_POLICIES}, got {enroll_policy!r}"
            )
        if max_duration == 0:
            raise ValueError("max_duration=0 is ambiguous; use -1 (no limit) or >0")
        if target_sr is not None and int(target_sr) <= 0:
            raise ValueError(f"target_sr must be positive when provided, got {target_sr}")
        if type(simulation_cfg_strict) is not bool:
            raise TypeError("simulation_cfg_strict must be an exact bool")
        self.sim_cfg = SimulationConfigs.from_dict(
            simulation_cfg or {}, strict=simulation_cfg_strict
        )
        self.manifest_format = manifest_format
        self.path_base = path_base
        self.target_selection = target_selection
        self.mix_variant = mix_variant
        self.recipe_root = recipe_root
        self.enabled_cues = enabled
        self.cue_schema = cue_schema
        self.reference_channel = int(reference_channel)

        # 样本内随机性（注册音抽选 / 裁切起点）的局部 RNG：每次从
        # (seed, epoch, sample_index) 派生独立 Random，打破模块级 random 的
        # 全局状态依赖，保证单卡内部可复现；多 worker/DDP 仍由框架的
        # worker/rank-shuffle 提供样本序随机性。
        self._rng_seed = (int(seed) if seed is not None else 0)
        self.enrollment_augmentor = EnrollmentAugmentor.from_config(enrollment_augment)
        augment_cfg = enrollment_augment if isinstance(enrollment_augment, Mapping) else {}
        self._enrollment_augment_rirs: tuple[str, ...] = ()
        self._enrollment_augment_noises: tuple[str, ...] = ()
        if self.enrollment_augmentor.requires_rir_provider():
            rir_scp = augment_cfg.get("rir_scp")
            if not isinstance(rir_scp, str) or not rir_scp.strip():
                raise ValueError(
                    "enrollment_augment.prob_reverb > 0 requires non-empty rir_scp"
                )
            rir_scp = resolve_config_resource_path(rir_scp)
            _, _, rir_flat, _ = read_source_scp(rir_scp)
            self._enrollment_augment_rirs = tuple(rir_flat.values())
            if not self._enrollment_augment_rirs:
                raise ValueError(
                    "enrollment_augment.prob_reverb > 0 requires a non-empty RIR provider"
                )
            moving_paths = [
                path
                for path in self._enrollment_augment_rirs
                if _is_moving_rir_path(path)
            ]
            if moving_paths and self.enrollment_augmentor.rir_static_only:
                moving = set(moving_paths)
                self._enrollment_augment_rirs = tuple(
                    path for path in self._enrollment_augment_rirs if path not in moving
                )
                if not self._enrollment_augment_rirs:
                    raise ValueError("enrollment rir_static_only selected an empty static RIR pool")
            elif moving_paths:
                raise ValueError(
                    "enrollment augmentation supports static RIR only; moving RIR "
                    f"provider is not supported: {moving_paths[0]!r}"
                )
        if self.enrollment_augmentor.requires_noise_provider():
            noise_scp = augment_cfg.get("noise_scp")
            if not isinstance(noise_scp, str) or not noise_scp.strip():
                raise ValueError(
                    "enrollment_augment.prob_noise > 0 requires non-empty noise_scp"
                )
            noise_scp = resolve_config_resource_path(noise_scp)
            _, _, noise_flat, _ = read_source_scp(noise_scp)
            self._enrollment_augment_noises = tuple(noise_flat.values())
            if not self._enrollment_augment_noises:
                raise ValueError(
                    "enrollment_augment.prob_noise > 0 requires a non-empty noise provider"
                )
        self.max_duration_seconds = (
            float(max_duration_seconds) if max_duration_seconds is not None else None
        )
        self.max_duration = -1 if self.max_duration_seconds is not None else max_duration
        self.max_enroll_duration_seconds = (
            float(max_enroll_duration_seconds)
            if max_enroll_duration_seconds is not None and float(max_enroll_duration_seconds) > 0
            else None
        )
        if type(spatial_enabled) is not bool:
            raise ValueError("spatial_enabled must be a bool")
        if self.sim_cfg.spatial_enabled:
            raise ValueError(
                "offline TSE spatial_enabled is a split-level reader policy, "
                "not a simulation_cfg field"
            )
        self.spatial_enabled = spatial_enabled
        self.resolved_microphone_array, self.channel_order = resolve_offline_spatial_geometry(
            spatial_enabled=spatial_enabled, array_profile=array_profile,
            custom_geometry=custom_geometry, reference_channel=reference_channel,
            channel_order=channel_order,
        )
        spatial_requested = "spatial" in self.enabled_cues
        spatial_options = {
            "spatial_representation": spatial_representation,
            "spatial_angle_unit": spatial_angle_unit,
            "spatial_coordinate_frame": spatial_coordinate_frame,
            "spatial_speed_of_sound_mps": spatial_speed_of_sound_mps,
            "spatial_track_timebase": spatial_track_timebase,
        }
        if spatial_requested:
            if cue_schema != "wesep_v0_1":
                raise ValueError("spatial cue decoding requires cue_schema='wesep_v0_1'")
            if not spatial_enabled or self.resolved_microphone_array is None:
                raise ValueError(
                    "spatial cue decoding requires spatial_enabled=True with geometry"
                )
            if spatial_representation not in _SPATIAL_REPRESENTATIONS:
                raise ValueError(
                    "spatial_representation must be 'static' or 'track'"
                )
            if spatial_angle_unit != _SPATIAL_ANGLE_UNIT:
                raise ValueError("spatial_angle_unit must be explicitly 'radian'")
            if spatial_coordinate_frame != _SPATIAL_COORDINATE_FRAME:
                raise ValueError(
                    "spatial_coordinate_frame must be explicitly "
                    f"{_SPATIAL_COORDINATE_FRAME!r}"
                )
            if (
                isinstance(spatial_speed_of_sound_mps, bool)
                or not isinstance(spatial_speed_of_sound_mps, (int, float))
                or not math.isfinite(float(spatial_speed_of_sound_mps))
                or float(spatial_speed_of_sound_mps) <= 0.0
            ):
                raise ValueError(
                    "spatial_speed_of_sound_mps must be positive and finite"
                )
            if spatial_representation == "track":
                if spatial_track_timebase != _SPATIAL_TRACK_TIMEBASE:
                    raise ValueError(
                        "track spatial cue requires "
                        "spatial_track_timebase='source_samples'"
                    )
            elif spatial_track_timebase is not None:
                raise ValueError(
                    "static spatial cue must not configure spatial_track_timebase"
                )
        elif any(value is not None for value in spatial_options.values()):
            raise ValueError(
                "spatial cue split parameters require enabled_cues to include 'spatial'"
            )
        self.spatial_representation = spatial_representation
        self.spatial_angle_unit = spatial_angle_unit
        self.spatial_coordinate_frame = spatial_coordinate_frame
        self.spatial_speed_of_sound_mps = (
            float(spatial_speed_of_sound_mps)
            if spatial_speed_of_sound_mps is not None
            else None
        )
        self.spatial_track_timebase = spatial_track_timebase
        self._visual_requested = "visual" in self.enabled_cues
        self._textual_requested = "textual" in self.enabled_cues
        if (self._visual_requested or self._textual_requested) and cue_schema != "wesep_v0_1":
            raise ValueError(
                "visual/textual cue decoding requires cue_schema='wesep_v0_1'"
            )
        self.enroll_policy = enroll_policy
        self.target_sr = int(target_sr) if target_sr is not None else None
        self.filter_by_duration = bool(filter_by_duration)
        self.filter_min_duration = float(filter_min_duration)
        self.filter_max_duration = float(filter_max_duration)
        self._audio_info_cache = {}

        manifest_path = carriers[0]
        adapter_cues = self.enabled_cues if cue_schema == "wesep_v0_1" else ()
        self.records = load_wesep_records(
            manifest_path,
            manifest_format=manifest_format,
            path_base=path_base,
            mix_variant=mix_variant,
            target_selection=target_selection,
            recipe_root=recipe_root,
            cues_yaml=cues_yaml if adapter_cues else None,
            enabled_cues=adapter_cues,
        )
        if not self.records:
            raise ValueError(f"no samples parsed from {manifest_path}")
        unsupported = sorted(set(self.enabled_cues) - _IMPLEMENTED_CUES)
        if unsupported:
            raise ValueError(
                "TSEOfflineDataset has indexed but cannot decode enabled cue(s): "
                f"{unsupported}"
            )

        self._legacy_cue: dict[str, Any] | None = None
        if cue_schema == "realtse_raw":
            self._legacy_cue = _load_legacy_raw_cue(str(Path(cues_yaml).resolve()))
            self._audio_representation = "waveform"
            self._cue_policy = self._legacy_cue["policy_type"]
        elif "audio" in self.enabled_cues:
            audio_specs = {
                (target.cue("audio").spec.cue_type, target.cue("audio").spec.cue_format)
                for record in self.records
                for target in record.targets
            }
            if len(audio_specs) != 1:
                raise ValueError(f"audio cue representation changes across records: {audio_specs}")
            audio_type, audio_format = next(iter(audio_specs))
            if (audio_type, audio_format) == ("wav", "waveform"):
                self._audio_representation = "waveform"
            elif (audio_type, audio_format) == ("npy", "spk_embedding"):
                self._audio_representation = "embedding"
            else:
                raise ValueError(
                    f"unsupported decoded audio cue {audio_type!r}/{audio_format!r}"
                )
            policies = {
                target.cue("audio").spec.policy_type
                for record in self.records
                for target in record.targets
            }
            if len(policies) != 1:
                raise ValueError(f"audio cue policy changes across records: {policies}")
            self._cue_policy = next(iter(policies))
        else:
            self._audio_representation = None
            self._cue_policy = None
        self.enroll_policy = enroll_policy or self._cue_policy
        if self._audio_representation is None and enroll_policy is not None:
            raise ValueError("enroll_policy is only valid when audio cue is enabled")
        if self._audio_representation != "waveform" and self.enrollment_augmentor.enabled:
            raise ValueError("enrollment_augment requires speaker waveform cue")

        self._spatial_fields: tuple[str, ...] | None = None
        self._spatial_field_defaults: dict[str, Any] = {}
        if spatial_requested:
            spatial_specs = [
                target.cue("spatial").spec
                for record in self.records
                for target in record.targets
            ]
            first_spatial_spec = spatial_specs[0]
            if (
                first_spatial_spec.cue_type,
                first_spatial_spec.cue_format,
            ) != ("npy", "fields"):
                raise ValueError("offline spatial cue requires npy/fields resources")
            fields = tuple(first_spatial_spec.format_metadata["fields"])
            defaults = dict(first_spatial_spec.format_metadata["field_defaults"])
            if any(
                tuple(spec.format_metadata["fields"]) != fields
                or dict(spec.format_metadata["field_defaults"]) != defaults
                for spec in spatial_specs[1:]
            ):
                raise ValueError(
                    "spatial field order/defaults must be fixed across the split"
                )
            required_fields = {"azimuth", "elevation"}
            if spatial_representation == "track":
                required_fields.add("sample_anchor")
            if set(fields) != required_fields:
                raise ValueError(
                    f"{spatial_representation} spatial fields must contain exactly "
                    f"{sorted(required_fields)}, got {list(fields)}"
                )
            self._spatial_fields = fields
            self._spatial_field_defaults = defaults

        self._textual_selection: dict[str, Any] | None = None
        self._textual_reserved: tuple[int, int, int] | None = None
        if self._visual_requested:
            visual_formats = {
                (target.cue("visual").spec.cue_type, target.cue("visual").spec.cue_format)
                for record in self.records
                for target in record.targets
            }
            if len(visual_formats) != 1:
                raise ValueError(
                    f"visual cue representation changes across records: {visual_formats}"
                )
            visual_format = next(iter(visual_formats))
            if visual_format == _RAW_VIDEO_TYPE_FORMAT:
                raise ValueError(
                    "raw video visual cue requires an optional video decoder; only "
                    "npy/muse_frontend pre-extracted features are supported"
                )
            if visual_format != _VISUAL_TYPE_FORMAT:
                raise ValueError(
                    f"unsupported decoded visual cue {visual_format!r}"
                )
            visual_policies = {
                target.cue("visual").spec.policy_type
                for record in self.records
                for target in record.targets
            }
            if visual_policies != {"fixed"}:
                raise ValueError(
                    f"visual cue policy changes across records: {visual_policies}"
                )
        if self._textual_requested:
            textual_specs = [
                target.cue("textual").spec
                for record in self.records
                for target in record.targets
            ]
            textual_formats = {
                (spec.cue_type, spec.cue_format) for spec in textual_specs
            }
            if len(textual_formats) != 1:
                raise ValueError(
                    f"textual cue representation changes across records: {textual_formats}"
                )
            textual_format = next(iter(textual_formats))
            if textual_format != _TEXTUAL_TYPE_FORMAT:
                raise ValueError(
                    f"unsupported decoded textual cue {textual_format!r}"
                )
            selections = [dict(spec.format_metadata["selection"]) for spec in textual_specs]
            if any(selection != selections[0] for selection in selections[1:]):
                raise ValueError("textual selection must be fixed across the split")
            selection = selections[0]
            if selection.get("type") not in _TEXTUAL_SELECTIONS:
                raise ValueError(
                    f"textual selection.type must be one of {_TEXTUAL_SELECTIONS}"
                )
            self._textual_selection = selection
            self._textual_reserved = _textual_reserved_ids(textual_specs[0])

        identities = sorted(
            {target.speaker_id for record in self.records for target in record.targets}
        )
        self.speaker_identity_to_idx = {
            speaker_id: index for index, speaker_id in enumerate(identities)
        }
        self.uids = [record.key for record in self.records]
        self._records_by_uid = {record.key: record for record in self.records}
        self.mixture = {record.key: record.mixture_paths for record in self.records}
        self.target = {
            record.key: tuple(target.source_paths for target in record.targets)
            for record in self.records
        }
        self._wesep_spk = {
            record.key: tuple(target.speaker_id for target in record.targets)
            for record in self.records
        }
        self.targets_per_mixture = max(len(record.targets) for record in self.records)
        self._resolve_metadata()
        self.cue_specs = self._build_cue_specs(placeholder_num_samples)

        self._full_len = len(self.uids)
        if self.filter_by_duration:
            self._filtered_indices = self._build_filtered_indices()
            if len(self._filtered_indices) == 0:
                raise ValueError(
                    f"filter_by_duration=True: no samples in range "
                    f"[{self.filter_min_duration}s, {self.filter_max_duration}s]. "
                    "Check filter_min_duration / filter_max_duration or disable filter."
                )
            _log.info(
                "Duration filter: kept %d of %d samples (%.2fs-%.2fs)",
                len(self._filtered_indices), self._full_len,
                self.filter_min_duration, self.filter_max_duration,
            )
        else:
            self._filtered_indices = list(range(self._full_len))
        super().__init__(
            full_length=self._full_len,
            filtered_indices=self._filtered_indices,
            lifecycle=OfflineLifecycleOptions(
                strict_load=strict_load,
                placeholder_num_samples=placeholder_num_samples,
                seed=self._rng_seed,
            ),
        )

    def _legacy_lookup_key(self, speaker_id: str, uid: str) -> str:
        assert self._legacy_cue is not None
        if self._legacy_cue["key_field"] == "mix_spk_id":
            return f"{uid}::{speaker_id}"
        return speaker_id

    def _local_rng(self, real_index: int) -> random.Random:
        """样本内随机性（注册音抽选 / 裁切起点）的局部 RNG。

        从 (seed, epoch, real_index) 派生独立实例，避免模块级 random 的全局
        状态依赖；固定 seed 时同一样本可复现。DDP/多 worker 的样本序随机
        仍由框架的 rank/worker/shuffle 负责，本方法只保证"同样本内决策稳定"。
        """
        return self.sample_rng(real_index)

    def set_epoch(self, epoch: int, base_seed: Optional[int] = None) -> None:
        """幂等设置 epoch/base_seed；构造时 seed 永不被累计修改。"""
        super().set_epoch(epoch, base_seed=base_seed)

    def _enrollment_augment_rng(
        self, real_index: int, target_index: int
    ) -> np.random.Generator:
        """为每个目标的 opt-in enrollment augment 派生独立随机流。"""
        entropy = [
            self._configured_seed,
            self._base_seed,
            self._epoch,
            int(real_index),
            int(target_index),
            0x54534545,
        ]
        return np.random.default_rng(np.random.SeedSequence(entropy))

    def _augment_enrollment(
        self,
        enrollment: torch.Tensor,
        sample_rate: int,
        real_index: int,
        target_index: int,
    ) -> torch.Tensor:
        """仅在显式启用时增强最终 enrollment；不触及 Python candidate/crop RNG。"""
        if not self.enrollment_augmentor.enabled:
            return enrollment

        def _load_provider_audio(path: str, output_sr: int) -> np.ndarray:
            waveform, source_sr = load_audio(path)
            waveform = _strict_mono_waveform(
                waveform, context="enrollment augmentation provider audio"
            )
            if source_sr != output_sr:
                waveform = resample(waveform, source_sr, output_sr)
            result = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
                raise ValueError(
                    f"enrollment augmentation provider produced invalid audio: {path!r}"
                )
            return result

        def _rir_provider(output_sr: int, rng: np.random.Generator):
            path = self._enrollment_augment_rirs[
                int(rng.integers(0, len(self._enrollment_augment_rirs)))
            ]
            try:
                rir = _load_provider_audio(path, output_sr).reshape(1, -1)
            except Exception as exc:
                raise EnrollmentAugmentationProviderError(
                    f"failed to read enrollment RIR provider {path!r}"
                ) from exc
            return rir, {"path": path}

        def _noise_provider(output_sr: int, length: int, rng: np.random.Generator):
            path = self._enrollment_augment_noises[
                int(rng.integers(0, len(self._enrollment_augment_noises)))
            ]
            try:
                noise = _load_provider_audio(path, output_sr)
                if noise.size < length:
                    noise = np.tile(noise, (length + noise.size - 1) // noise.size)
                start = (
                    int(rng.integers(0, noise.size - length + 1))
                    if noise.size > length
                    else 0
                )
                noise = noise[start : start + length].reshape(1, -1)
                if noise.shape[1] != length:
                    raise ValueError(
                        f"enrollment noise provider returned {noise.shape[1]} samples, "
                        f"expected {length}"
                    )
            except Exception as exc:
                raise EnrollmentAugmentationProviderError(
                    f"failed to read enrollment noise provider {path!r}"
                ) from exc
            return noise, {"path": path}

        waveform = enrollment.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=True)
        augmented, _ = self.enrollment_augmentor.apply(
            waveform,
            sample_rate,
            self._enrollment_augment_rng(real_index, target_index),
            rir_provider=_rir_provider if self.enrollment_augmentor.requires_rir_provider() else None,
            noise_provider=_noise_provider if self.enrollment_augmentor.requires_noise_provider() else None,
        )
        return torch.from_numpy(augmented).unsqueeze(0)

    def _audio_cue_items(
        self, record: WeSepSampleRecord, target: WeSepTargetRecord
    ) -> tuple[bool, tuple[Mapping[str, Any], ...]]:
        if self._audio_representation is None:
            return False, ()
        if self._legacy_cue is not None:
            key = self._legacy_lookup_key(target.speaker_id, record.key)
            items = self._legacy_cue["resource"].get(key)
            if not items:
                raise KeyError(f"guaranteed legacy audio cue is missing for {key!r}")
            return True, tuple(items)
        cue = target.cue("audio")
        return cue.present, cue.items

    def _select_audio_item(
        self,
        record: WeSepSampleRecord,
        target: WeSepTargetRecord,
        rng: random.Random,
    ) -> Mapping[str, Any] | None:
        present, items = self._audio_cue_items(record, target)
        if not present:
            return None
        if not items:
            raise ValueError(
                f"present audio cue has no resources for {record.key!r}/{target.speaker_id!r}"
            )
        return items[0] if self.enroll_policy == "fixed" else rng.choice(items)

    @staticmethod
    def _load_embedding(path: str) -> torch.Tensor:
        value = np.load(path, allow_pickle=False)
        if not isinstance(value, np.ndarray) or value.ndim != 1 or value.size < 1:
            raise ValueError(f"speaker embedding must be a non-empty 1D array: {path!r}")
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"speaker embedding must be finite floating-point: {path!r}")
        return torch.from_numpy(value.astype(np.float32, copy=False)).contiguous()

    def set_speaker_label_mapping(self, identity_to_idx: dict) -> None:
        """validation 复用 train 的 identity→label 映射；未见 identity 回退 -1。

        与 ``TSEDynamicMixingDataset.set_speaker_label_mapping`` 同一套 dense
        label 空间校验（非负、唯一、从 0 连续），仅 key 空间不同：offline
        manifest 的 speaker key 是字符串（canonical identity tuple 是 dynamic
        仿真的 key）。
        """
        if not isinstance(identity_to_idx, Mapping):
            raise ValueError("speaker label mapping must be a mapping")
        validated: dict[str, int] = {}
        seen_values: set[int] = set()
        for key, value in identity_to_idx.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "speaker label mapping keys must be non-empty speaker id strings"
                )
            if type(value) is not int or value < 0:
                raise ValueError(
                    "speaker label mapping values must be non-negative ints"
                )
            if value in seen_values:
                raise ValueError("speaker label mapping values must be unique")
            seen_values.add(value)
            validated[key] = value
        if sorted(seen_values) != list(range(len(seen_values))):
            raise ValueError(
                "speaker label mapping values must form a contiguous range "
                "starting at 0"
            )
        self.speaker_identity_to_idx = validated

    def _spk_label(self, speaker_id: str) -> int:
        return self.speaker_identity_to_idx.get(speaker_id, -1)

    def _get_audio_info(self, path: str) -> tuple[int, int, int]:
        """返回并缓存 ``(frames, samplerate, channels)``。"""
        if path not in self._audio_info_cache:
            import soundfile as sf

            try:
                info = sf.info(path)
            except Exception as exc:
                raise RuntimeError(f"cannot read audio metadata: {path}") from exc
            if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
                raise RuntimeError(
                    f"invalid audio metadata for {path}: frames={info.frames}, "
                    f"samplerate={info.samplerate}, channels={info.channels}"
                )
            self._audio_info_cache[path] = (
                int(info.frames),
                int(info.samplerate),
                int(info.channels),
            )
        return self._audio_info_cache[path]

    def _audio_group_info(
        self, paths: Sequence[str], *, source: str
    ) -> tuple[int, int, int]:
        infos = [self._get_audio_info(path) for path in paths]
        frames = {info[0] for info in infos}
        sample_rates = {info[1] for info in infos}
        if len(frames) != 1 or len(sample_rates) != 1:
            raise ValueError(
                f"{source} multi-file channels must share frames/sample rate, got {infos}"
            )
        return infos[0][0], infos[0][1], sum(info[2] for info in infos)

    def _load_spatial_payload(
        self, path: str
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self._spatial_fields is not None
        value = np.load(path, allow_pickle=False)
        if not isinstance(value, np.ndarray) or value.size < 1:
            raise ValueError(f"spatial cue must be a non-empty ndarray: {path!r}")
        fields = self._spatial_fields
        columns: dict[str, np.ndarray] = {}
        if value.dtype.names is not None:
            point_shape = value.shape or (1,)
            if len(point_shape) != 1:
                raise ValueError(
                    f"structured spatial cue must have shape [P], got {value.shape}: {path!r}"
                )
            point_count = point_shape[0]
            for field in fields:
                if field in value.dtype.names:
                    column = np.asarray(value[field]).reshape(-1)
                elif field in self._spatial_field_defaults:
                    column = np.full(
                        point_count,
                        self._spatial_field_defaults[field],
                    )
                else:
                    raise ValueError(
                        f"spatial cue {path!r} is missing declared field {field!r}"
                    )
                columns[field] = column
        else:
            if not np.issubdtype(value.dtype, np.number):
                raise TypeError(f"spatial cue must use a numeric dtype: {path!r}")
            matrix = np.asarray(value)
            if self.spatial_representation == "static" and matrix.ndim == 1:
                matrix = matrix[None, :]
            if matrix.ndim != 2 or matrix.shape[1] != len(fields):
                raise ValueError(
                    f"spatial cue must have shape [P,{len(fields)}] in declared "
                    f"field order {list(fields)}, got {value.shape}: {path!r}"
                )
            columns = {
                field: matrix[:, index] for index, field in enumerate(fields)
            }

        point_count = len(next(iter(columns.values())))
        if point_count < 1 or any(len(column) != point_count for column in columns.values()):
            raise ValueError(f"spatial cue fields must share one non-empty P axis: {path!r}")
        if self.spatial_representation == "static" and point_count != 1:
            raise ValueError(f"static spatial cue must contain exactly one point: {path!r}")
        try:
            azimuth = np.asarray(columns["azimuth"], dtype=np.float64)
            elevation = np.asarray(columns["elevation"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"spatial angles must be numeric radians: {path!r}") from exc
        if not np.isfinite(azimuth).all() or not np.isfinite(elevation).all():
            raise ValueError(f"spatial angles must contain only finite radians: {path!r}")
        direction = torch.from_numpy(
            np.stack((azimuth, elevation)).astype(np.float32, copy=False)
        ).contiguous()
        if self.spatial_representation == "static":
            return direction[:, 0], None

        raw_anchors = np.asarray(columns["sample_anchor"])
        if not np.issubdtype(raw_anchors.dtype, np.number):
            raise TypeError(f"spatial sample_anchor must be numeric: {path!r}")
        numeric_anchors = raw_anchors.astype(np.float64)
        if not np.isfinite(numeric_anchors).all() or not np.equal(
            numeric_anchors, np.floor(numeric_anchors)
        ).all():
            raise ValueError(
                f"spatial sample_anchor must contain finite integer samples: {path!r}"
            )
        anchors = torch.from_numpy(numeric_anchors.astype(np.int64)).contiguous()
        if int(anchors[0].item()) != 0:
            raise ValueError(f"spatial track anchors must start at source sample 0: {path!r}")
        if anchors.numel() > 1 and bool((anchors[1:] <= anchors[:-1]).any().item()):
            raise ValueError(f"spatial track anchors must be strictly increasing: {path!r}")
        return direction, anchors

    def _load_visual_payload(
        self, path: str, item: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """读取预提取 MuSE 特征 ``[D,V]``，返回 ``(features, frame_times, duration, fps)``。

        上游 npy 以最后一个轴为视频时间轴（``[D,T]``）；更宽的维度按上游
        ``flatten(1, -2)`` 语义折叠为 ``[D,V]``。帧时间轴由 item 的
        ``duration_sec``/``fps`` 定义，不做按点数 resize。
        """
        try:
            value = np.load(path, allow_pickle=False)
        except OSError as exc:
            raise FileNotFoundError(f"cannot read visual cue npy: {path!r}") from exc
        if not isinstance(value, np.ndarray) or value.size < 1 or value.ndim < 2:
            raise ValueError(
                f"muse_frontend visual cue must be a non-empty ndarray with at "
                f"least 2 dims, got {getattr(value, 'shape', None)}: {path!r}"
            )
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"muse_frontend visual cue must use a numeric dtype: {path!r}")
        matrix = np.asarray(value).reshape(-1, value.shape[-1])
        if matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError(f"muse_frontend visual cue has an empty axis: {path!r}")
        numeric = matrix.astype(np.float32, copy=False)
        if not np.isfinite(numeric).all():
            raise ValueError(
                f"muse_frontend visual cue must contain only finite values: {path!r}"
            )
        features = torch.from_numpy(np.ascontiguousarray(numeric)).contiguous()
        num_frames = int(features.shape[1])
        duration, fps = _resolve_visual_clock(
            item, num_frames, source=f"visual cue {path!r}"
        )
        frame_times = (
            torch.arange(num_frames, dtype=torch.float64) / fps
        ).contiguous()
        return features, frame_times, duration, fps

    def _resolve_metadata(self) -> None:
        """在首个样本解码前完整验证全部波形、embedding、spatial 与 visual/textual 资源。"""
        self.utt2fs: dict[str, int] = {}
        self.lengths: dict[str, int] = {}
        self._mixture_info: dict[str, tuple[int, int, int]] = {}
        self._target_info: dict[tuple[str, str], tuple[int, int, int]] = {}
        self._waveform_cue_info: dict[str, tuple[int, int, int]] = {}
        self._spatial_cue_payloads: dict[
            str, tuple[torch.Tensor, torch.Tensor | None]
        ] = {}
        self._visual_cue_payloads: dict[
            str, tuple[torch.Tensor, torch.Tensor, float, float]
        ] = {}
        self._embedding_dim: int | None = None
        self._visual_feature_dim: int | None = None
        for record in self.records:
            mix_info = self._audio_group_info(
                record.mixture_paths, source=f"TSE key={record.key} mixture"
            )
            self._mixture_info[record.key] = mix_info
            self.lengths[record.key] = mix_info[0]
            self.utt2fs[record.key] = mix_info[1]
            if self.reference_channel < 0 or self.reference_channel >= mix_info[2]:
                raise ValueError(
                    f"reference_channel={self.reference_channel} is invalid for "
                    f"TSE key={record.key} mixture C={mix_info[2]}"
                )
            if mix_info[2] > 1 and self.resolved_microphone_array is None:
                raise ValueError(
                    f"TSE key={record.key} multi-file/multichannel mixture C={mix_info[2]} "
                    "requires spatial_enabled=True with explicit geometry and channel_order"
                )
            if (
                self.resolved_microphone_array is not None
                and mix_info[2] != self.resolved_microphone_array.channels
            ):
                raise ValueError(
                    f"TSE key={record.key} mixture C={mix_info[2]} does not match "
                    f"geometry C={self.resolved_microphone_array.channels}"
                )
            for target in record.targets:
                target_info = self._audio_group_info(
                    target.source_paths,
                    source=f"TSE key={record.key} target={target.speaker_id}",
                )
                if target_info[2] not in {1, mix_info[2]}:
                    raise ValueError(
                        f"TSE key={record.key} target={target.speaker_id} has "
                        f"C={target_info[2]}; expected mono or mixture C={mix_info[2]}"
                    )
                self._target_info[(record.key, target.speaker_id)] = target_info
                _, cue_items = self._audio_cue_items(record, target)
                for item in cue_items:
                    path = item["path"]
                    if self._audio_representation == "waveform":
                        self._waveform_cue_info[path] = self._audio_group_info(
                            (path,),
                            source=(
                                f"TSE key={record.key} speaker={target.speaker_id} "
                                "waveform cue"
                            ),
                        )
                    elif self._audio_representation == "embedding":
                        embedding = self._load_embedding(path)
                        dim = int(embedding.numel())
                        if self._embedding_dim is None:
                            self._embedding_dim = dim
                        elif dim != self._embedding_dim:
                            raise ValueError(
                                "speaker embedding dimensions must be fixed, got "
                                f"{self._embedding_dim} and {dim} at {path!r}"
                            )
                if "spatial" in self.enabled_cues:
                    spatial_record = target.cue("spatial")
                    if not spatial_record.present:
                        if spatial_record.items:
                            raise ValueError("missing spatial cue must not contain resources")
                        continue
                    if len(spatial_record.items) != 1:
                        raise ValueError(
                            "present offline spatial cue requires exactly one npy resource"
                        )
                    spatial_item = spatial_record.items[0]
                    spatial_path = spatial_item["path"]
                    if spatial_path not in self._spatial_cue_payloads:
                        self._spatial_cue_payloads[spatial_path] = (
                            self._load_spatial_payload(spatial_path)
                        )
                    direction, anchors = self._spatial_cue_payloads[spatial_path]
                    if self.spatial_representation == "track":
                        item_sample_rate = spatial_item.get("sample_rate")
                        item_num_samples = spatial_item.get("num_samples")
                        if type(item_sample_rate) is not int or item_sample_rate <= 0:
                            raise ValueError(
                                "track spatial cue item requires positive integer sample_rate"
                            )
                        if type(item_num_samples) is not int or item_num_samples <= 0:
                            raise ValueError(
                                "track spatial cue item requires positive integer num_samples"
                            )
                        if item_sample_rate != mix_info[1] or item_num_samples != mix_info[0]:
                            raise ValueError(
                                f"track spatial cue sample clock {item_sample_rate}/"
                                f"{item_num_samples} does not match mixture "
                                f"{mix_info[1]}/{mix_info[0]} for {record.key!r}"
                            )
                        assert anchors is not None
                        if int(anchors[-1].item()) >= item_num_samples:
                            raise ValueError(
                                "track spatial cue sample_anchor must be within raw mixture"
                            )
                        if "duration_sec" in spatial_item and not math.isclose(
                            float(spatial_item["duration_sec"]),
                            item_num_samples / item_sample_rate,
                            rel_tol=0.0,
                            abs_tol=1.0 / item_sample_rate,
                        ):
                            raise ValueError(
                                "track spatial cue duration_sec does not match sample clock"
                            )
                    elif anchors is not None or tuple(direction.shape) != (2,):
                        raise ValueError("static spatial cue must decode to shape [2]")
                if self._visual_requested:
                    visual_record = target.cue("visual")
                    if not visual_record.present:
                        if visual_record.items:
                            raise ValueError("missing visual cue must not contain resources")
                    else:
                        if len(visual_record.items) != 1:
                            raise ValueError(
                                "present offline visual cue requires exactly one npy resource"
                            )
                        visual_item = visual_record.items[0]
                        visual_path = visual_item["path"]
                        if visual_path not in self._visual_cue_payloads:
                            self._visual_cue_payloads[visual_path] = (
                                self._load_visual_payload(visual_path, visual_item)
                            )
                        features, _, duration, fps = self._visual_cue_payloads[visual_path]
                        dim = int(features.shape[0])
                        if self._visual_feature_dim is None:
                            self._visual_feature_dim = dim
                        elif dim != self._visual_feature_dim:
                            raise ValueError(
                                "visual feature dimensions must be fixed, got "
                                f"{self._visual_feature_dim} and {dim} at {visual_path!r}"
                            )
                        mix_duration = mix_info[0] / mix_info[1]
                        if mix_duration > duration + 1.0 / fps + _CLOCK_TOLERANCE:
                            raise ValueError(
                                f"visual cue {visual_path!r} covers {duration}s but "
                                f"mixture {record.key!r} lasts {mix_duration}s; "
                                "edge padding is not applied by the canonical data layer"
                            )
                if self._textual_requested:
                    textual_record = target.cue("textual")
                    if not textual_record.present:
                        if textual_record.items:
                            raise ValueError("missing textual cue must not contain resources")
                    else:
                        if len(textual_record.items) != 1:
                            raise ValueError(
                                "present offline textual cue requires exactly one json resource"
                            )
                        assert self._textual_reserved is not None
                        _validate_textual_tokens(
                            textual_record.items[0]["phn_label"],
                            source=f"textual cue {textual_record.lookup_key!r}",
                            reserved=self._textual_reserved,
                        )
        if self._audio_representation == "embedding" and self._embedding_dim is None:
            raise ValueError("enabled speaker embedding cue has no present resource to infer D")
        if self._visual_requested and self._visual_feature_dim is None:
            raise ValueError("enabled visual cue has no present resource to infer D")

    def _build_cue_specs(self, temporal_fallback: int) -> TSECueSpecs:
        values: dict[str, TSECueSpec] = {}
        if self._audio_representation == "embedding":
            assert self._embedding_dim is not None
            values["speaker_embedding"] = TSECueSpec(
                "embedding", (self._embedding_dim,)
            )
        elif self._audio_representation == "waveform":
            fallback_capacity = min(
                max(1, int(temporal_fallback)),
                DEFAULT_PLACEHOLDER_NUM_SAMPLES,
            )
            if self.max_enroll_duration_seconds is not None:
                maximum_output_sr = max(
                    self.target_sr or self.utt2fs[record.key]
                    for record in self.records
                )
                fallback_capacity = min(
                    fallback_capacity,
                    max(1, int(self.max_enroll_duration_seconds * maximum_output_sr)),
                )
            values["speaker_waveform"] = TSECueSpec(
                "waveform", (fallback_capacity,)
            )
        if self.spatial_representation == "static":
            values["spatial"] = TSECueSpec("static", (2,))
        elif self.spatial_representation == "track":
            values["spatial"] = TSECueSpec("track", (2, 1))
        if self._visual_feature_dim is not None:
            # 时间容量由 collate 按批内实际帧数扩展，fallback 只声明最小合法值。
            values["visual"] = TSECueSpec(
                _VISUAL_REPRESENTATION, (self._visual_feature_dim, 1)
            )
        if self._textual_requested:
            values["textual"] = TSECueSpec("phoneme_ids", (1,))
        return TSECueSpecs(**values)

    def _build_filtered_indices(self):
        out = []
        for i in range(self._full_len):
            uid = self.uids[i]
            sr = self.utt2fs.get(uid)
            length = self.lengths.get(uid)
            if sr is None or length is None:
                _log.warning("filter_by_duration: missing metadata for uid=%s, skipping", uid)
                continue
            if int(self.filter_min_duration * sr) <= length <= int(self.filter_max_duration * sr):
                out.append(i)
        return out

    def _output_sr(self, real_index: int) -> int:
        uid = self.uids[real_index]
        return self.target_sr or self.utt2fs[uid]

    def _output_cap(self, output_sr: int) -> int:
        return (
            int(self.max_duration_seconds * output_sr)
            if self.max_duration_seconds is not None
            else self.max_duration
        )

    def _aligned_lengths(self, record: WeSepSampleRecord) -> tuple[int, int, int]:
        """返回共同 source 长度、完整 output 长度与限长后的 output 长度。"""
        mix_len, mix_sr, _ = self._mixture_info[record.key]
        aligned_lengths = [mix_len]
        for target in record.targets:
            target_len, target_sr, _ = self._target_info[
                (record.key, target.speaker_id)
            ]
            aligned_lengths.append(resampled_num_frames(target_len, target_sr, mix_sr))
        source_length = min(aligned_lengths)
        output_sr = self.target_sr or mix_sr
        output_length = resampled_num_frames(source_length, mix_sr, output_sr)
        cap = self._output_cap(output_sr)
        length = min(output_length, cap) if cap > 0 else output_length
        return source_length, output_length, length

    def _source_length(self, real_index: int) -> int:
        return self._aligned_lengths(self.records[real_index])[2]

    def _sample_desc(self, real_index: int) -> str:
        return f"uid={self.uids[real_index]}"

    def _sample_description(self, real_index: int) -> str:
        return self._sample_desc(real_index)

    def _missing_cues(self) -> TSECues:
        values: dict[str, TSECue] = {}
        if self._audio_representation == "waveform":
            values["speaker_waveform"] = TSECue(
                "waveform", None, False, 0, None
            )
        elif self._audio_representation == "embedding":
            values["speaker_embedding"] = TSECue(
                "embedding", None, False, None, None
            )
        if self.spatial_representation == "static":
            values["spatial"] = TSECue("static", None, False, None, None)
        elif self.spatial_representation == "track":
            values["spatial"] = TSECue("track", None, False, 0, None)
        if self._visual_feature_dim is not None:
            values["visual"] = TSECue(
                _VISUAL_REPRESENTATION, None, False, 0, None
            )
        if self._textual_requested:
            values["textual"] = TSECue("phoneme_ids", None, False, 0, None)
        return TSECues(**values)

    def _placeholder(self, real_index: int) -> dict[str, Any]:
        record = self.records[real_index]
        sr = self.target_sr or self.utt2fs.get(record.key, 16000)
        cap = self._output_cap(sr)
        if self.max_duration_seconds is not None:
            out_len = max(1, cap)
        else:
            out_len = cap if cap > 0 else self.placeholder_num_samples
        channels = self._mixture_info.get(record.key, (0, 0, 1))[2]
        targets = [
            make_tse_target(
                target=torch.zeros(out_len, dtype=torch.float32),
                cues=self._missing_cues(),
                speaker_id=target.speaker_id,
                spk_label=-1,
                target_present=True,
            )
            for target in record.targets
        ]
        return make_tse_sample(
            key=record.key,
            mixture=torch.zeros(channels, out_len, dtype=torch.float32),
            sr=sr,
            length=out_len,
            targets=targets,
            geometry=self.resolved_microphone_array,
            timebase=TrajectoryTimebase(sr, out_len, out_len, 1),
            provenance={"placeholder": True},
        )

    def _placeholder_sample(self, real_index: int):
        return self._placeholder(real_index)

    def _load_sample(self, real_index: int):
        return self._getitem_impl(real_index)

    def _fallback_or_placeholder(self, index: int, real_index: int, primary_error):
        """仅在失败样本的 output-SR bucket 内做一次有界 fallback。"""
        if isinstance(primary_error, EnrollmentAugmentationProviderError):
            raise primary_error
        failed_sr = int(self._output_sr(real_index))
        public_candidates = list(range(index + 1, len(self))) + list(range(index))
        candidates = [
            self._filtered_indices[candidate]
            for candidate in public_candidates
            if int(self._output_sr(self._filtered_indices[candidate])) == failed_sr
        ]
        if not candidates:
            _log.warning(
                "No TSE fallback sample in output SR bucket %s; returning placeholder",
                failed_sr,
            )
            return self._placeholder(real_index)
        fallback_real = self.sample_rng(real_index, stream=2).choice(candidates)
        try:
            return self._load_sample(fallback_real)
        except EnrollmentAugmentationProviderError:
            raise
        except Exception as fallback_error:
            _log.warning(
                "Both primary and same-SR fallback failed; returning silence placeholder. "
                "primary=%s fallback=%s",
                primary_error,
                fallback_error,
            )
            return self._placeholder(real_index)

    @staticmethod
    def _validate_loaded_group(
        values: Sequence[tuple[torch.Tensor, int]], *, source: str
    ) -> tuple[torch.Tensor, int]:
        sample_rates = {sample_rate for _, sample_rate in values}
        lengths = {int(waveform.shape[-1]) for waveform, _ in values}
        if len(sample_rates) != 1 or len(lengths) != 1:
            raise RuntimeError(
                f"{source} multi-file channels changed after metadata validation"
            )
        waveform = torch.cat([value for value, _ in values], dim=0)
        if waveform.ndim != 2 or waveform.shape[0] < 1 or waveform.shape[1] < 1:
            raise ValueError(f"{source} must decode to channel-first [C,T]")
        return waveform, next(iter(sample_rates))

    def _load_audio_group(
        self, paths: Sequence[str], *, source: str
    ) -> tuple[torch.Tensor, int]:
        return self._validate_loaded_group(
            [load_audio(path) for path in paths], source=source
        )

    def _load_audio_group_window(
        self,
        paths: Sequence[str],
        start: int,
        length: int,
        *,
        source: str,
    ) -> tuple[torch.Tensor, int]:
        return self._validate_loaded_group(
            [load_audio_window(path, start, length) for path in paths],
            source=source,
        )

    def _target_reference(
        self, waveform: torch.Tensor, *, mixture_channels: int, source: str
    ) -> torch.Tensor:
        if waveform.shape[0] == 1:
            return waveform
        if waveform.shape[0] != mixture_channels:
            raise ValueError(
                f"{source} has C={waveform.shape[0]}, expected 1 or {mixture_channels}"
            )
        return waveform[self.reference_channel : self.reference_channel + 1]

    def _window_plan(
        self, record: WeSepSampleRecord, rng: random.Random
    ) -> dict[str, Any]:
        _, mix_sr, _ = self._mixture_info[record.key]
        output_sr = self.target_sr or mix_sr
        source_length, output_length, length = self._aligned_lengths(record)
        source_window = min(
            source_length, resampled_num_frames(length, output_sr, mix_sr)
        )
        max_start = min(
            source_length - source_window,
            (output_length - length) * mix_sr // output_sr,
        )
        start = rng.randint(0, max_start) if max_start > 0 else 0

        cue_plans = []
        for target in record.targets:
            item = self._select_audio_item(record, target, rng)
            if item is None or self._audio_representation != "waveform":
                plan: dict[str, Any] = {"item": item}
            else:
                cue_len, cue_sr, _ = self._waveform_cue_info[item["path"]]
                cue_start = 0
                selected_len = cue_len
                if self.max_enroll_duration_seconds is not None:
                    cue_cap = max(1, int(self.max_enroll_duration_seconds * cue_sr))
                    if cue_len > cue_cap:
                        cue_start = (
                            rng.randint(0, cue_len - cue_cap)
                            if self.enroll_policy == "random"
                            else 0
                        )
                        selected_len = cue_cap
                plan = {
                    "item": item,
                    "start": cue_start,
                    "length": selected_len,
                    "sample_rate": cue_sr,
                }
            plan["textual"] = self._select_textual_tokens(record, target, rng)
            cue_plans.append(plan)
        return {
            "mix_start": start,
            "mix_len": source_window,
            "output_start": resampled_num_frames(start, mix_sr, output_sr),
            "output_len": length,
            "cue_plans": tuple(cue_plans),
        }

    def _getitem_impl(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        win = self._window_plan(record, self._local_rng(index))
        mix_sr = self.utt2fs[record.key]
        target_srs = {
            self._target_info[(record.key, target.speaker_id)][1]
            for target in record.targets
        }
        main_tracks_need_resample = (
            target_srs != {mix_sr}
            or (self.target_sr is not None and self.target_sr != mix_sr)
        )
        if main_tracks_need_resample:
            return self._getitem_impl_full(index, win)
        return self._getitem_fast(index, win, mix_sr)

    def _build_sample(
        self,
        *,
        record: WeSepSampleRecord,
        mixture: torch.Tensor,
        targets: Sequence[torch.Tensor],
        cues: Sequence[TSECues],
        sample_rate: int,
        length: int,
        source_start_sample: int,
        output_start_sample: int,
    ) -> dict[str, Any]:
        target_items = [
            make_tse_target(
                target=target.squeeze(0),
                cues=target_cues,
                speaker_id=target_record.speaker_id,
                spk_label=self._spk_label(target_record.speaker_id),
                target_present=True,
            )
            for target_record, target, target_cues in zip(record.targets, targets, cues)
        ]
        return make_tse_sample(
            key=record.key,
            mixture=mixture,
            sr=sample_rate,
            length=length,
            targets=target_items,
            geometry=self.resolved_microphone_array,
            timebase=TrajectoryTimebase(sample_rate, length, length, 1),
            provenance={
                "placeholder": False,
                "source_start_sample": source_start_sample,
                "source_sample_rate": self.utt2fs[record.key],
                "output_start_sample": output_start_sample,
                "manifest_path": record.manifest_path,
                "mix_variant": record.mix_variant,
                "mixture_paths": record.mixture_paths,
                "target_paths": tuple(target.source_paths for target in record.targets),
                "reference_channel": self.reference_channel,
                "channel_order": self.channel_order,
                "spatial_cue": (
                    {
                        "representation": self.spatial_representation,
                        "angle_unit": self.spatial_angle_unit,
                        "coordinate_frame": self.spatial_coordinate_frame,
                        "speed_of_sound_mps": self.spatial_speed_of_sound_mps,
                        "track_timebase": self.spatial_track_timebase,
                        "field_order": self._spatial_fields,
                        "field_defaults": dict(self._spatial_field_defaults),
                    }
                    if self.spatial_representation is not None
                    else None
                ),
                "visual_cue": (
                    {
                        "representation": _VISUAL_REPRESENTATION,
                        "cue_format": "npy/muse_frontend",
                        "feature_dim": self._visual_feature_dim,
                        "timebase": "seconds_from_output_start",
                        "crop": "mixture_source_seconds",
                    }
                    if self._visual_feature_dim is not None
                    else None
                ),
                "textual_cue": (
                    {
                        "representation": "phoneme_ids",
                        "cue_format": "json/dae_keyword_phoneme",
                        "selection": dict(self._textual_selection),
                        "reserved_ids": {
                            "padding": self._textual_reserved[0],
                            "bos": self._textual_reserved[1],
                            "eos": self._textual_reserved[2],
                        },
                    }
                    if self._textual_selection is not None
                    else None
                ),
            },
        )

    def _load_spatial_cue(
        self,
        *,
        target: WeSepTargetRecord,
        source_start_sample: int,
        source_num_samples: int,
        source_sample_rate: int,
        output_num_samples: int,
        output_sample_rate: int,
    ) -> TSECue | None:
        if self.spatial_representation is None:
            return None
        cue_record = target.cue("spatial")
        if not cue_record.present:
            return (
                TSECue("static", None, False, None, None)
                if self.spatial_representation == "static"
                else TSECue("track", None, False, 0, None)
            )
        item = cue_record.items[0]
        direction, anchors = self._spatial_cue_payloads[item["path"]]
        if self.spatial_representation == "static":
            return TSECue("static", direction.clone(), True, None, None)

        assert anchors is not None
        source_end = source_start_sample + source_num_samples
        active = int(
            torch.searchsorted(
                anchors,
                torch.tensor(source_start_sample, dtype=torch.int64),
                right=True,
            ).item()
        ) - 1
        if active < 0:
            raise ValueError("spatial track has no direction at crop start")
        later = torch.nonzero(
            (anchors > source_start_sample) & (anchors < source_end),
            as_tuple=False,
        ).flatten().tolist()
        selected = [active, *later]
        output_origin = resampled_num_frames(
            source_start_sample, source_sample_rate, output_sample_rate
        )
        output_anchors: list[int] = []
        output_values: list[torch.Tensor] = []
        for position, point_index in enumerate(selected):
            output_anchor = (
                0
                if position == 0
                else resampled_num_frames(
                    int(anchors[point_index].item()),
                    source_sample_rate,
                    output_sample_rate,
                )
                - output_origin
            )
            if output_anchor >= output_num_samples:
                continue
            point_value = direction[:, point_index]
            if output_anchors and output_anchor == output_anchors[-1]:
                output_values[-1] = point_value
            else:
                output_anchors.append(output_anchor)
                output_values.append(point_value)
        if not output_anchors or output_anchors[0] != 0:
            raise RuntimeError("cropped spatial track must start at output sample 0")
        value = torch.stack(output_values, dim=1).contiguous()
        timebase = torch.tensor(output_anchors, dtype=torch.int64)
        return TSECue(
            "track",
            value,
            True,
            len(output_anchors),
            timebase,
        )

    def _select_textual_tokens(
        self,
        record: WeSepSampleRecord,
        target: WeSepTargetRecord,
        rng: random.Random,
    ) -> torch.Tensor | None:
        """按 resource metadata 的 ``selection`` 选词并展平 keyword phoneme id。

        ``random`` 从 ``word_range`` 抽词数与起点，``fixed`` 用 ``kw_candidate``
        词下标；两者都只产出 keyword token，BOS/EOS 由 KCE 前端注入。
        """
        if not self._textual_requested:
            return None
        cue_record = target.cue("textual")
        if not cue_record.present:
            return None
        assert self._textual_selection is not None
        labels = cue_record.items[0]["phn_label"]
        selection = self._textual_selection
        if selection["type"] == "random":
            low, high = selection["word_range"]
            word_count = min(rng.randint(int(low), int(high)), len(labels))
            start = rng.randint(0, len(labels) - word_count)
            selected = labels[start : start + word_count]
        else:
            candidates = cue_record.items[0]["kw_candidate"]
            selected = [labels[index] for index in candidates]
        tokens = [int(token) for word in selected for token in word]
        if not tokens:
            raise ValueError(
                f"textual cue selection produced no phoneme ids for "
                f"{cue_record.lookup_key!r}"
            )
        assert self._textual_reserved is not None
        padding, bos, eos = self._textual_reserved
        reserved = {padding, bos, eos}
        if any(token in reserved for token in tokens):
            raise ValueError(
                f"textual cue {cue_record.lookup_key!r} selected a reserved token "
                "(padding/BOS/EOS are owned by the KCE frontend)"
            )
        return torch.tensor(tokens, dtype=torch.int64).contiguous()

    def _load_textual_cue(self, *, cue_plan: Mapping[str, Any]) -> TSECue | None:
        if not self._textual_requested:
            return None
        tokens = cue_plan.get("textual")
        if tokens is None:
            return TSECue("phoneme_ids", None, False, 0, None)
        assert isinstance(tokens, torch.Tensor)
        return TSECue(
            "phoneme_ids",
            tokens.clone(),
            True,
            int(tokens.numel()),
            None,
        )

    def _load_visual_cue(
        self,
        *,
        target: WeSepTargetRecord,
        source_start_sample: int,
        source_num_samples: int,
        source_sample_rate: int,
        output_num_samples: int,
        output_sample_rate: int,
    ) -> TSECue | None:
        """按 mixture source 秒域 crop 预提取视觉帧，产出 typed ``features`` cue。"""
        if self._visual_feature_dim is None:
            return None
        cue_record = target.cue("visual")
        if not cue_record.present:
            return TSECue(_VISUAL_REPRESENTATION, None, False, 0, None)
        item = cue_record.items[0]
        features, frame_times, duration, fps = self._visual_cue_payloads[item["path"]]
        window_start = source_start_sample / source_sample_rate
        window_end = (source_start_sample + source_num_samples) / source_sample_rate
        if window_start < 0.0 or window_end <= window_start:
            raise ValueError("visual cue crop window must be a non-empty non-negative span")
        if window_end > duration + 1.0 / fps + _CLOCK_TOLERANCE:
            raise ValueError(
                f"visual cue covers {duration}s but the crop window ends at "
                f"{window_end}s"
            )
        active = int(
            torch.searchsorted(
                frame_times,
                torch.tensor(window_start, dtype=torch.float64),
                right=True,
            ).item()
        ) - 1
        if active < 0:
            raise ValueError("visual cue has no frame at crop start")
        later = torch.nonzero(
            (frame_times > window_start) & (frame_times < window_end),
            as_tuple=False,
        ).flatten().tolist()
        selected = [active, *later]
        output_seconds = output_num_samples / output_sample_rate
        indices: list[int] = []
        offsets: list[float] = []
        for position, frame_index in enumerate(selected):
            offset = (
                0.0
                if position == 0
                else float(frame_times[frame_index].item()) - window_start
            )
            if offset >= output_seconds:
                continue
            indices.append(int(frame_index))
            offsets.append(offset)
        if not indices:
            raise RuntimeError("cropped visual cue is empty")
        value = features[:, indices].contiguous()
        timebase = torch.tensor(offsets, dtype=torch.float64).contiguous()
        return TSECue(
            _VISUAL_REPRESENTATION,
            value,
            True,
            len(indices),
            timebase,
        )

    def _load_planned_cue(
        self,
        *,
        record: WeSepSampleRecord,
        target: WeSepTargetRecord,
        cue_plan: Mapping[str, Any],
        real_index: int,
        target_index: int,
        reference_sample_rate: int,
        output_sample_rate: int,
        source_start_sample: int,
        source_num_samples: int,
        output_num_samples: int,
    ) -> TSECues:
        missing = self._missing_cues()
        speaker_waveform = missing.speaker_waveform
        speaker_embedding = missing.speaker_embedding
        item = cue_plan.get("item")
        if item is not None and self._audio_representation == "embedding":
            speaker_embedding = TSECue(
                "embedding", self._load_embedding(item["path"]), True, None, None
            )
        elif item is not None and self._audio_representation == "waveform":
            planned_length = int(cue_plan["length"])
            enrollment, source_sample_rate = self._load_audio_group_window(
                (item["path"],),
                int(cue_plan["start"]),
                planned_length,
                source=f"TSE key={record.key} speaker={target.speaker_id} cue",
            )
            if (
                source_sample_rate != int(cue_plan["sample_rate"])
                or enrollment.shape[-1] < planned_length
            ):
                raise RuntimeError(
                    f"short or mismatched waveform cue read "
                    f"({record.key}/{target.speaker_id})"
                )
            enrollment = _strict_mono_waveform(
                enrollment[:, :planned_length], context="TSE enrollment cue"
            )
            if source_sample_rate != reference_sample_rate and self.strict_load:
                raise ValueError(
                    "strict_load: mixture/enroll sample rate mismatch "
                    f"fs_mixture={reference_sample_rate} fs_enroll={source_sample_rate} "
                    f"({record.key}/{target.speaker_id})"
                )
            if source_sample_rate != output_sample_rate:
                enrollment = resample(
                    enrollment, source_sample_rate, output_sample_rate
                )
            if enrollment.shape[-1] < 1:
                raise RuntimeError(
                    f"empty waveform cue ({record.key}/{target.speaker_id})"
                )
            enrollment = ensure_finite(
                enrollment,
                strict=self.strict_load,
                track="speaker_waveform",
                sample=f"{record.key}/{target.speaker_id}",
            )
            enrollment = self._augment_enrollment(
                enrollment, output_sample_rate, real_index, target_index
            )
            speaker_waveform = TSECue(
                "waveform",
                enrollment.squeeze(0),
                True,
                int(enrollment.shape[-1]),
                None,
            )
        spatial = self._load_spatial_cue(
            target=target,
            source_start_sample=source_start_sample,
            source_num_samples=source_num_samples,
            source_sample_rate=reference_sample_rate,
            output_num_samples=output_num_samples,
            output_sample_rate=output_sample_rate,
        )
        visual = self._load_visual_cue(
            target=target,
            source_start_sample=source_start_sample,
            source_num_samples=source_num_samples,
            source_sample_rate=reference_sample_rate,
            output_num_samples=output_num_samples,
            output_sample_rate=output_sample_rate,
        )
        textual = self._load_textual_cue(cue_plan=cue_plan)
        return TSECues(
            speaker_waveform=speaker_waveform,
            speaker_embedding=speaker_embedding,
            spatial=spatial,
            visual=visual,
            textual=textual,
        )

    def _getitem_fast(
        self, index: int, win: Mapping[str, Any], fs: int, out_sr: int | None = None
    ) -> dict[str, Any]:
        """mixture/target 同采样率路径；cue 按自身采样率独立窗口读取。"""
        del out_sr
        record = self.records[index]
        length = int(win["mix_len"])
        mixture, mix_sr = self._load_audio_group_window(
            record.mixture_paths,
            int(win["mix_start"]),
            length,
            source=f"TSE key={record.key} mixture",
        )
        if mix_sr != fs or int(mixture.shape[-1]) < length:
            raise RuntimeError(f"short or mismatched mixture read ({self._sample_desc(index)})")
        mixture = ensure_finite(
            mixture[:, :length],
            strict=self.strict_load,
            track="mixture",
            sample=record.key,
        )
        targets: list[torch.Tensor] = []
        cues: list[TSECues] = []
        for target_index, (target_record, cue_plan) in enumerate(
            zip(record.targets, win["cue_plans"])
        ):
            target, target_sr = self._load_audio_group_window(
                target_record.source_paths,
                int(win["mix_start"]),
                length,
                source=f"TSE key={record.key} target={target_record.speaker_id}",
            )
            if target_sr != fs or int(target.shape[-1]) < length:
                raise RuntimeError(
                    f"short or mismatched target read ({record.key}/{target_record.speaker_id})"
                )
            target = self._target_reference(
                target,
                mixture_channels=int(mixture.shape[0]),
                source=f"TSE key={record.key} target={target_record.speaker_id}",
            )
            targets.append(
                ensure_finite(
                    target[:, :length],
                    strict=self.strict_load,
                    track="target",
                    sample=f"{record.key}/{target_record.speaker_id}",
                )
            )
            cues.append(
                self._load_planned_cue(
                    record=record,
                    target=target_record,
                    cue_plan=cue_plan,
                    real_index=index,
                    target_index=target_index,
                    reference_sample_rate=fs,
                    output_sample_rate=fs,
                    source_start_sample=int(win["mix_start"]),
                    source_num_samples=int(win["mix_len"]),
                    output_num_samples=length,
                )
            )
        return self._build_sample(
            record=record,
            mixture=mixture,
            targets=targets,
            cues=cues,
            sample_rate=fs,
            length=length,
            source_start_sample=int(win["mix_start"]),
            output_start_sample=int(win["mix_start"]),
        )

    def _getitem_impl_full(
        self, index: int, win: Mapping[str, Any]
    ) -> dict[str, Any]:
        """mixture/target 异构采样率路径；cue 仍独立窗口读取。"""
        record = self.records[index]
        mixture, mix_source_sr = self._load_audio_group(
            record.mixture_paths, source=f"TSE key={record.key} mixture"
        )
        target_values: list[torch.Tensor] = []
        for target_record in record.targets:
            target, target_sr = self._load_audio_group(
                target_record.source_paths,
                source=f"TSE key={record.key} target={target_record.speaker_id}",
            )
            target = self._target_reference(
                target,
                mixture_channels=int(mixture.shape[0]),
                source=f"TSE key={record.key} target={target_record.speaker_id}",
            )
            if target_sr != mix_source_sr:
                if self.strict_load:
                    raise ValueError(
                        "strict_load: mixture/target sample rate mismatch "
                        f"fs_mixture={mix_source_sr} fs_target={target_sr} "
                        f"({record.key}/{target_record.speaker_id})"
                    )
                target = resample(target, target_sr, mix_source_sr)
            target_values.append(target)

        output_sr = self.target_sr or mix_source_sr
        if output_sr != mix_source_sr:
            mixture = resample(mixture, mix_source_sr, output_sr)
            target_values = [
                resample(target, mix_source_sr, output_sr) for target in target_values
            ]
        start = int(win["output_start"])
        planned_length = int(win["output_len"])
        mixture = mixture[:, start : start + planned_length]
        target_values = [
            target[:, start : start + planned_length] for target in target_values
        ]
        length = min(
            planned_length,
            int(mixture.shape[-1]),
            *(int(target.shape[-1]) for target in target_values),
        )
        if length < 1:
            raise RuntimeError(f"empty aligned output window ({record.key})")
        mixture = ensure_finite(
            mixture[:, :length],
            strict=self.strict_load,
            track="mixture",
            sample=record.key,
        )
        target_values = [
            ensure_finite(
                target[:, :length],
                strict=self.strict_load,
                track="target",
                sample=f"{record.key}/{target_record.speaker_id}",
            )
            for target_record, target in zip(record.targets, target_values)
        ]

        cues = [
            self._load_planned_cue(
                record=record,
                target=target_record,
                cue_plan=cue_plan,
                real_index=index,
                target_index=target_index,
                reference_sample_rate=mix_source_sr,
                output_sample_rate=output_sr,
                source_start_sample=int(win["mix_start"]),
                source_num_samples=int(win["mix_len"]),
                output_num_samples=length,
            )
            for target_index, (target_record, cue_plan) in enumerate(
                zip(record.targets, win["cue_plans"])
            )
        ]
        return self._build_sample(
            record=record,
            mixture=mixture,
            targets=target_values,
            cues=cues,
            sample_rate=output_sr,
            length=length,
            source_start_sample=int(win["mix_start"]),
            output_start_sample=start,
        )


def build_offline_tse_dataset(cfg: dict, mode: str = "train") -> TSEOfflineDataset:
    """从整包 config（含 data 段）构建 TSE 离线数据集。

    消费 ``data.train`` / ``data.validation`` 段的官方 wesep 键
    （``samples_list`` / ``samples_jsonl`` / ``cues_yaml``）；
    ``data.sample_rate`` 作为 ``target_sr``；``strict_load`` /
    ``placeholder_num_samples`` 与主链路同一套 ``resolve_*`` 解析。
    """
    from core.data.config_flags import resolve_simulation_cfg_strict
    from core.data.dataset_build import (
        apply_duration_filter_config,
        resolve_dataset_build_options,
    )

    options = resolve_dataset_build_options(cfg, mode, expected_task="tse")
    if options.mode.storage != "offline":
        raise ValueError(
            f"data.mode {options.mode.task_mode!r} is not an offline TSE mode"
        )
    section = options.section
    apply_duration_filter_config(section)
    # 样本内随机性 seed：整包 cfg 的 meta.seed，缺省 0（保证复现时对同样本一致）
    seed = int((cfg.get("meta") or {}).get("seed", 0))
    dataset = TSEOfflineDataset(
        **section,
        seed=seed,
        target_sr=options.target_sr,
        strict_load=options.strict_load,
        simulation_cfg_strict=resolve_simulation_cfg_strict(cfg),
        placeholder_num_samples=options.placeholder_num_samples,
    )
    from core.data.dataset_contract import (
        attach_dataset_contract,
        build_tse_dataset_contract,
        infer_dataset_sample_rate,
    )

    attach_dataset_contract(
        dataset,
        build_tse_dataset_contract(
            infer_dataset_sample_rate(dataset, cfg),
            variable_targets=int(getattr(dataset, "targets_per_mixture", 1)) > 1,
            cue_specs=dataset.cue_specs,
        ),
    )
    return dataset
