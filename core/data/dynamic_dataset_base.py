"""Base dynamic mixing dataset shared by SE, AEC, TSE, and SS.

子类实现 ``__init__``、``_build_sample`` 及任务资源/验证 hooks；``__getitem__``
与 trace 生命周期由基类统一管理。
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import random
import shlex
from bisect import bisect_right
from pathlib import Path
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from numbers import Real
from typing import Any, NamedTuple, Optional

import numpy as np
import soundfile
import torch
from scipy.signal import filtfilt
from core.data.simulation.urgent_impl import codec_compression_result

from core.data.acoustic_scene_plan import (
    AcousticScenePlan,
    C0AssetReplayError,
    NoisePlan,
    NoiseSourcePlan,
    RIRPlan,
    ResolvedAssets,
    SourceGainPlan,
)
from core.data.acoustic_scene_replay import replay_target_acoustic_scene
from core.data.constants import DEFAULT_PLACEHOLDER_NUM_SAMPLES
from core.utils.validation import require_positive_int
from core.data.noise_source_policy import (
    NoiseApplicationRole,
    NoiseCacheProvenance,
    NoiseCategory,
    NoiseSourceFamily,
    NoiseSourcePolicy,
)
from core.data.noise_attributes import NOISE_ATTRIBUTES
from core.data.peak_normalization import (
    PeakNormalizationResult,
    apply_peak_normalization,
)
from core.data.simulation_trace import SimulationTrace, SourceSimulationTrace, TracedSample
from core.data.time_transform import (
    CanvasOp,
    CropOp,
    POLICY_FINAL_CENTER_CROP,
    TimeTransformPlan,
    plan_time_transform,
)
from core.data.waveform_bundle import WaveformBundle
from core.data.simulation.contracts import (
    LayerEFOutput,
    LayerEFResult,
    LayerEOperationResult,
    LayerEResult,
    PeakGuardFacts,
    PeakGuardResult,
    PreparedSpeech,
    apply_deterministic_peak_guard,
)
from core.data.noise_roles import category_balance_name, has_role
from core.data.sampling import (
    DEFAULT_ALPHA,
    DEFAULT_DIFFICULTY,
    difficulty_key,
    SpeechLeafMetadata,
    SpeechPlan,
    make_speech_plan,
)
from core.data.speech_quota_solver import _leaf_key_sort_key
from core.data.simulation_audio_utils import (
    _align_noise_2d_to_length,
    _align_with_window,
    _load_audio_cached_impl,
    _read_cached_2d,
    _read_rir_meta_segment_sec,
    _resample_audio,
)
from core.data.simulation.resource_store import ResourceStore
from core.data.simulation.speech_preparation import SpeechPreparation
from core.data.simulation.acoustic_renderer import AcousticBundleResult, AcousticRenderer
from core.data.simulation.scenario_sampler import ScenarioSampler
from core.data.simulation.sample_state import (
    capture_last_completed_state,
    restore_last_completed_state,
)
from core.data.simulation_utils import (
    _apply_1d_time_plan,
    _build_align_to_max_duration_plan,
    fast_active_rms,
    rng_choice,
    rng_choices_weighted,
    rng_randint,
    rng_random,
    rng_sample,
    rng_uniform,
    read_source_scp,
    read_training_noise_manifest,
)
from core.data.post_augment import (
    apply_post_augmentations,
    colored_noise,
    draw_source_volume_target_db,
    scale_source_volume_stem,
    source_volume_normalize,
    source_volume_shared_max_abs,
)
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_config_contract import validate_simulation_config_contract
from core.data.simulation_views import project_simulation_view, simulation_view_for
from core.data.temporal_config import CanvasSpec
from core.data.simulation.engine import (
    add_moving_reverberation,
    add_paired_reverberation,
    add_reverberation,
    bandwidth_limitation,
    clipping_aug,
    estimate_early_rir,
    mix_noise,
    packet_loss,
    wind_noise,
)
from core.data.simulation.noise_image import NoiseImage, adapt_noise_waveform
from core.data.simulation.role_policy import BackgroundPolicy, BackgroundSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeechSamplingCell:
    """Duration-filtered speech entries sharing balanced-sampling metadata."""

    language: str
    dataset_id: str
    batching_sr: int
    external_indices: tuple[int, ...]
    uids: tuple[str, ...]
    effective_seconds: float


@dataclass(frozen=True)
class SpeechSamplingLeaf:
    """Duration-filtered speech entries sharing one atomic balancing leaf.

    leaf key = ``(language, dataset_id, canonical_speaker_or_none,
    batching_sr)``；``batching_sr`` 以 ``Dataset.get_srs()``（处理 SR）为准，
    不用 manifest ``fs``。``external_indices`` / ``uids`` 按
    ``(uid, external_index)`` 稳定排序一一对应；``effective_seconds`` 为
    manifest ``length / fs`` 的精确 ``Fraction`` 和。
    """

    language: str
    dataset_id: str
    speaker: tuple[str, str] | None
    batching_sr: int
    external_indices: tuple[int, ...]
    uids: tuple[str, ...]
    effective_seconds: Fraction

    @property
    def key(self) -> tuple[str, str, tuple[str, str] | None, int]:
        return (self.language, self.dataset_id, self.speaker, self.batching_sr)

    def planner_leaf(self) -> SpeechLeafMetadata:
        """转换为 planner 消费的 public leaf（只携带容量与 metadata）。"""
        return SpeechLeafMetadata(
            language=self.language,
            dataset_id=self.dataset_id,
            speaker=self.speaker,
            batching_sr=self.batching_sr,
            uid_indices=self.external_indices,
            effective_seconds=self.effective_seconds,
        )


@dataclass(frozen=True)
class SourceSummary:
    """Aggregated duration statistics for one source pool.

    所有时长字段统一使用**小时**，避免训练日志在秒/小时之间来回换算。
    """

    clips: int
    total_hours: float | None
    min_hours: float | None
    max_hours: float | None
    mean_hours: float | None
    sr_counts: dict[int, int]


@dataclass(frozen=True)
class DataSummary:
    """Light-weight data distribution report produced before the first epoch."""

    split: str
    n_samples: int
    n_batches: int
    sources: dict[str, SourceSummary]
    dataloader_config: dict[str, Any]
    noise_categories: list[tuple[str, int, float]] | None = None
    scenario_probs: dict[str, float] | None = None
    extra: dict[str, Any] | None = None


def _compute_source_summary(
    uids: dict[int, list[str]],
    lengths: dict[str, int] | None,
) -> SourceSummary:
    """Compute duration statistics for a source pool.

    ``uids`` maps sample rate to ordered uid lists; ``lengths`` maps uid to
    sample count.  Returns a ``SourceSummary`` with **hour-based** stats when
    lengths are available.
    """
    clips = sum(len(uid_list) for uid_list in uids.values())
    sr_counts = {int(fs): len(uid_list) for fs, uid_list in uids.items()}

    if not lengths or clips == 0:
        return SourceSummary(
            clips=clips,
            total_hours=None,
            min_hours=None,
            max_hours=None,
            mean_hours=None,
            sr_counts=sr_counts,
        )

    seconds_list = [
        lengths[uid] / float(fs)
        for fs, uid_list in uids.items()
        for uid in uid_list
        if uid in lengths
    ]
    if not seconds_list:
        return SourceSummary(
            clips=clips,
            total_hours=None,
            min_hours=None,
            max_hours=None,
            mean_hours=None,
            sr_counts=sr_counts,
        )

    arr = np.asarray(seconds_list, dtype=np.float64) / 3600.0
    return SourceSummary(
        clips=clips,
        total_hours=float(arr.sum()),
        min_hours=float(arr.min()),
        max_hours=float(arr.max()),
        mean_hours=float(arr.mean()),
        sr_counts=sr_counts,
    )


def _speech_leaf_key_sort(key) -> tuple:
    """stable leaf order：委托 solver 模块的唯一排序契约实现。"""
    return _leaf_key_sort_key(key)


def _leaf_seed(seed: int, epoch: int, key) -> int:
    """leaf-local 稳定随机源：以 ``(seed, epoch, leaf_key)`` 为载荷派生。"""
    payload = json.dumps([seed, epoch, key], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _speech_cells_from_leaves(
    leaves: dict[tuple, "SpeechSamplingLeaf"],
) -> dict[tuple[str, str, int], SpeechSamplingCell]:
    """从 leaf 聚合对外三元 cell 视图（sampler 三元校验专用）。"""
    cells: dict[tuple[str, str, int], SpeechSamplingCell] = {}
    for leaf in leaves.values():
        triple = (leaf.language, leaf.dataset_id, leaf.batching_sr)
        existing = cells.get(triple)
        if existing is None:
            cells[triple] = SpeechSamplingCell(
                language=leaf.language,
                dataset_id=leaf.dataset_id,
                batching_sr=leaf.batching_sr,
                external_indices=leaf.external_indices,
                uids=leaf.uids,
                effective_seconds=float(leaf.effective_seconds),
            )
        else:
            cells[triple] = SpeechSamplingCell(
                language=leaf.language,
                dataset_id=leaf.dataset_id,
                batching_sr=leaf.batching_sr,
                external_indices=existing.external_indices + leaf.external_indices,
                uids=existing.uids + leaf.uids,
                effective_seconds=existing.effective_seconds
                + float(leaf.effective_seconds),
            )
    return cells


@dataclass(frozen=True)
class SpeechEpochPlan:
    """Dataset-owned selected speech epoch plan consumed by the sampler."""

    selected_indices: tuple[int, ...]
    selected_uids: tuple[str, ...]
    sample_rates: tuple[int, ...]
    selected_cell_keys: tuple[tuple[str, str, int], ...]
    speech_plan: SpeechPlan


@dataclass(frozen=True)
class SampleBuildResult:
    """任务侧完成的 sample 与其唯一 trace；legacy 状态只在基类发布。"""

    sample: Any
    trace: SimulationTrace


class SampleBuildContractError(RuntimeError):
    """``_build_sample`` 违反基类 trace 生命周期合同时 fail-fast。"""


@dataclass(frozen=True)
class FinalTimePlan:
    """final alignment 的显式 plan 与 legacy ctx Adapter 信息。"""

    transform: TimeTransformPlan | None
    output_length: int
    fixed_silence_canvas_active: bool
    crop_start: int | None = None


# Supported sample rates for bandwidth limitation (same as URGENT2026)
_BANDWIDTH_SAMPLE_RATES = [8000, 16000, 22050, 24000, 32000, 44100, 48000]

# Resampling method names for bandwidth limitation
_BANDWIDTH_RESAMPLE_METHODS = [
    "soxr_vhq", "soxr_hq", "soxr_mq", "soxr_lq", "soxr_qq",
    "kaiser_best", "kaiser_fast", "fft", "scipy",
]

# Selector 派生候选缓存同时受 entry 数和 UID 引用总量约束；单 entry 超预算不缓存。
_NOISE_SELECTOR_CACHE_MAX_ENTRIES = 16
_NOISE_SELECTOR_CACHE_MAX_UID_REFS = 65536

# 公共 source selector 缓存：source 数与 processing-SR view 数始终有界；UID 引用
# 通常不超过预算。单个 source 超预算时允许它作为唯一 stable entry 驻留，避免热路径
# 重复物化 UID tuple；不得截断或去重 UID，切换 source 时可淘汰该 oversized entry。
_SAMPLE_SELECTOR_CACHE_MAX_SOURCES = 8
_SAMPLE_SELECTOR_CACHE_MAX_PROCESSING_SRS = 32
_SAMPLE_SELECTOR_CACHE_MAX_UID_REFS = 1_048_576

class _SampleSelectorTierView:
    """按 source dict 原迭代顺序拼接 per-FS UID tuple 的随机访问 view。"""

    __slots__ = ("_uid_groups", "_ends", "_length")

    def __init__(self, uid_groups: tuple[tuple[str, ...], ...]) -> None:
        self._uid_groups = uid_groups
        total = 0
        ends = []
        for uids in uid_groups:
            total += len(uids)
            ends.append(total)
        self._ends = tuple(ends)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> str:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        group_index = bisect_right(self._ends, index)
        start = 0 if group_index == 0 else self._ends[group_index - 1]
        return self._uid_groups[group_index][index - start]


class _SampleSelectorSourceCache:
    """一个 ``source_by_fs`` 的 lazy per-FS tuple 与 processing-SR view。"""

    __slots__ = (
        "source_by_fs",
        "source_order",
        "sorted_nonempty_sample_rates",
        "uids_by_fs",
        "processing_sr_views",
        "uid_refs",
    )

    def __init__(self, source_by_fs: dict) -> None:
        self.source_by_fs = source_by_fs
        self.source_order = tuple(source_by_fs)
        self.sorted_nonempty_sample_rates = tuple(
            sorted(fs for fs in self.source_order if source_by_fs[fs])
        )
        self.uids_by_fs: dict[int, tuple[str, ...]] = {}
        self.processing_sr_views: OrderedDict[
            int, tuple[str, ...] | _SampleSelectorTierView | None
        ] = OrderedDict()
        self.uid_refs = 0


def _noise_category_meta_valid(metadata: dict) -> bool:
    """噪声类别元数据合法性谓词（与 strict admission 字段无关）。

    category 能力只要求非负整数 ``category_id`` 与非空、无首尾空白的
    ``balance_category``（兼容回退 ``category_name``）。
    """
    category_id = metadata.get("category_id")
    category_name = category_balance_name(metadata)
    return (
        type(category_id) is int
        and category_id >= 0
        and isinstance(category_name, str)
        and bool(category_name)
        and category_name == category_name.strip()
    )


def signal_rms_dbfs(x: np.ndarray) -> float:
    """全段 RMS 电平（dBFS，0 dBFS = ±1.0 满幅）；空/零信号返回 -120。"""
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(a ** 2)))
    if rms <= 0.0:
        return -120.0
    return 20.0 * float(np.log10(rms))


def signal_active_rms_dbfs(x: np.ndarray, threshold_db: float = -40.0) -> float:
    """活跃段 RMS 电平（dBFS）：以帧能量峰值为基准，门限以下帧视为静音。

    帧能量门限逻辑与 tools/prepare_data_tools/_audio_verify_utils.py 的
    ``_compute_active_rms`` 一致（约 100 帧切分，−40 dB 相对门限），
    在 core 侧独立实现，dataset 不依赖 tools。
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        return -120.0
    frame_len = max(1, a.size // 100)
    frames = np.array(
        [np.mean(a[i : i + frame_len] ** 2) for i in range(0, a.size, frame_len)]
    )
    peak = float(np.max(frames))
    if peak <= 0.0:
        return signal_rms_dbfs(a)
    active = frames[frames > peak * (10.0 ** (threshold_db / 10.0))]
    if active.size == 0:
        return signal_rms_dbfs(a)
    rms = float(np.sqrt(np.mean(active)))
    return 20.0 * float(np.log10(rms))


class _MixAcousticOutputTuple(NamedTuple):
    noisy_speech: np.ndarray  # RIR + 加性噪声 + 风噪 + colored noise 后的混合信号 (1, T)
    clean_target: np.ndarray  # 监督 target（early-RIR 或 dry，按 target_type）
    noisy_after_rir: np.ndarray  # 仅过 RIR、未加噪声的信号（residual noise stem 基准）
    additive_noise_2d: np.ndarray  # 全部加性噪声成分之和 (1, T)
    additive_snr_db_list: list[float]  # 各次加性混合的 drawn SNR（末位可能含 C-3 colored noise）
    used_sidechain_wind: bool  # 是否实际施加了 side-chain 风噪
    mix_two: bool  # 是否抽到双段噪声混合（noise_mix_prob）
    want_sidechain: bool  # 是否抽到 side-chain 风噪意图（无风噪样本时回退普通噪声）
    wn_snr: float  # 风噪/主噪声 SNR（与实际生效分支一致）
    snr_primary_drawn_db: float | None  # 主噪声 drawn SNR（need_noise=False 时 None）
    wind_noise_snr_candidate_db: float | None  # 风噪候选 drawn SNR（need_noise=False 时 None）
    use_reverb: bool  # 本样本是否施加 RIR


class MixAcousticOutput(_MixAcousticOutputTuple):
    """``_mix_acoustic`` 的返回结果，保持原 12 元组解包契约。

    ``acoustic_audit``、``source_volume_calibration``、``c0_audit_stems``、
    ``acoustic_scene_plan`` 与 ``resolved_assets`` 都不参与 tuple 位置语义；
    旧的 12 positional 构造、``len`` 和精确 12 项解包均保持兼容。
    """

    acoustic_audit: dict | None
    source_volume_calibration: dict | None
    c0_audit_stems: dict | None
    acoustic_scene_plan: AcousticScenePlan | None
    resolved_assets: ResolvedAssets | None
    noise_source_policies: tuple[NoiseSourcePolicy, ...]

    def __new__(
        cls,
        noisy_speech: np.ndarray,
        clean_target: np.ndarray,
        noisy_after_rir: np.ndarray,
        additive_noise_2d: np.ndarray,
        additive_snr_db_list: list[float],
        used_sidechain_wind: bool,
        mix_two: bool,
        want_sidechain: bool,
        wn_snr: float,
        snr_primary_drawn_db: float | None,
        wind_noise_snr_candidate_db: float | None,
        use_reverb: bool,
        acoustic_audit: dict | None = None,
        source_volume_calibration: dict | None = None,
        c0_audit_stems: dict | None = None,
        acoustic_scene_plan: AcousticScenePlan | None = None,
        resolved_assets: ResolvedAssets | None = None,
    ):
        output = super().__new__(
            cls,
            noisy_speech,
            clean_target,
            noisy_after_rir,
            additive_noise_2d,
            additive_snr_db_list,
            used_sidechain_wind,
            mix_two,
            want_sidechain,
            wn_snr,
            snr_primary_drawn_db,
            wind_noise_snr_candidate_db,
            use_reverb,
        )
        output.acoustic_audit = acoustic_audit
        output.source_volume_calibration = source_volume_calibration
        output.c0_audit_stems = c0_audit_stems
        output.acoustic_scene_plan = acoustic_scene_plan
        output.resolved_assets = resolved_assets
        output.noise_source_policies = ()
        return output


class _SingleItemExcludedView:
    """在已有稳定 sequence 上排除一个元素，不复制其余成员。"""

    __slots__ = ("_items", "_excluded_position")

    def __init__(self, items, excluded_position: int) -> None:
        self._items = items
        self._excluded_position = excluded_position

    def __len__(self) -> int:
        return len(self._items) - 1

    def __getitem__(self, index: int):
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)
        source_index = index + int(index >= self._excluded_position)
        return self._items[source_index]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __eq__(self, other) -> bool:
        return hasattr(other, "__len__") and len(other) == len(self) and all(
            left == right for left, right in zip(self, other)
        )


class _FilteredNoiseCategoryView(Mapping):
    """按 category/UID 排除的只读 mapping view，不复制 category 或 UID tuple。"""

    __slots__ = (
        "base",
        "exclude_category_id",
        "exclude_uid",
        "uid_category_id",
        "uid_position",
        "_length",
    )

    def __init__(
        self,
        base: Mapping[int, tuple[str, ...]],
        *,
        exclude_category_id: int | None,
        exclude_uid: str | None,
    ) -> None:
        self.base = base
        self.exclude_category_id = exclude_category_id
        self.exclude_uid = exclude_uid
        self.uid_category_id = None
        self.uid_position = None
        if exclude_uid is not None:
            for category_id, uids in base.items():
                if category_id == exclude_category_id:
                    continue
                try:
                    position = uids.index(exclude_uid)
                except ValueError:
                    continue
                self.uid_category_id = category_id
                self.uid_position = position
                break
        self._length = len(base) - int(exclude_category_id in base)
        if (
            self.uid_category_id is not None
            and len(base[self.uid_category_id]) == 1
        ):
            self._length -= 1

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for category_id in self.base:
            if category_id == self.exclude_category_id:
                continue
            if (
                category_id == self.uid_category_id
                and len(self.base[category_id]) == 1
            ):
                continue
            yield category_id

    def __getitem__(self, category_id: int):
        if category_id == self.exclude_category_id:
            raise KeyError(category_id)
        uids = self.base[category_id]
        if category_id != self.uid_category_id:
            return uids
        if len(uids) == 1:
            raise KeyError(category_id)
        return _SingleItemExcludedView(uids, self.uid_position)


class _CategoryExcludedView:
    """基于 canonical category order 的 rank-select sequence view。"""

    __slots__ = ("_categories", "_excluded_positions", "_length")

    def __init__(
        self,
        categories: tuple[int, ...],
        candidates: Mapping[int, tuple[str, ...]],
    ) -> None:
        self._categories = categories
        self._excluded_positions = [
            position
            for position, category_id in enumerate(categories)
            if category_id not in candidates
        ]
        self._length = len(categories) - len(self._excluded_positions)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> int:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        low, high = index, len(self._categories) - 1
        while low < high:
            middle = (low + high) // 2
            kept = middle + 1 - bisect_right(self._excluded_positions, middle)
            if kept > index:
                high = middle
            else:
                low = middle + 1
        return self._categories[low]

    def __iter__(self):
        for index in range(self._length):
            yield self[index]

    def __eq__(self, other) -> bool:
        return hasattr(other, "__len__") and len(other) == self._length and all(
            left == right for left, right in zip(self, other)
        )


class _NaturalCategoryExcludedView:
    """基于预建 category positions 的 manifest-order rank-select view。"""

    __slots__ = (
        "_uids",
        "_excluded_positions",
        "_extra_position",
        "_length",
    )

    def __init__(
        self,
        uids: tuple[str, ...],
        positions_by_category: dict[int, tuple[int, ...]],
        exclude_category_id: int,
        exclude_uid_position: int | None,
    ) -> None:
        self._uids = uids
        self._excluded_positions = positions_by_category.get(exclude_category_id, ())
        excluded_rank = (
            bisect_right(self._excluded_positions, exclude_uid_position)
            if exclude_uid_position is not None
            else 0
        )
        uid_is_category_excluded = (
            exclude_uid_position is not None
            and excluded_rank > 0
            and self._excluded_positions[excluded_rank - 1] == exclude_uid_position
        )
        self._extra_position = (
            exclude_uid_position
            if exclude_uid_position is not None and not uid_is_category_excluded
            else None
        )
        self._length = len(uids) - len(self._excluded_positions) - int(
            self._extra_position is not None
        )

    def __len__(self) -> int:
        return self._length

    def _excluded_at_or_before(self, position: int) -> int:
        count = bisect_right(self._excluded_positions, position)
        if self._extra_position is not None and self._extra_position <= position:
            count += 1
        return count

    def __getitem__(self, index: int) -> str:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        low, high = index, len(self._uids) - 1
        while low < high:
            middle = (low + high) // 2
            kept = middle + 1 - self._excluded_at_or_before(middle)
            if kept > index:
                high = middle
            else:
                low = middle + 1
        return self._uids[low]

    def __iter__(self):
        for index in range(self._length):
            yield self[index]

    def __eq__(self, other) -> bool:
        if not hasattr(other, "__len__") or len(other) != self._length:
            return False
        return all(left == right for left, right in zip(self, other))


class _NaturalTierExcludedView:
    """不合并 UID 的多 SR rank-select 计划。"""

    __slots__ = ("_views", "_ends", "_length")

    def __init__(self, views: tuple[_NaturalCategoryExcludedView, ...]) -> None:
        self._views = views
        total = 0
        ends = []
        for view in views:
            total += len(view)
            ends.append(total)
        self._ends = tuple(ends)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> str:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        view_index = bisect_right(self._ends, index)
        start = 0 if view_index == 0 else self._ends[view_index - 1]
        return self._views[view_index][index - start]

    def __iter__(self):
        for view in self._views:
            yield from view

    def __eq__(self, other) -> bool:
        return hasattr(other, "__len__") and len(other) == self._length and all(
            left == right for left, right in zip(self, other)
        )


def _read_speech_length_file(
    path: str,
    expected_uids: set[str],
    *,
    allow_extra_uids: bool = False,
) -> dict[str, int]:
    """严格读取 ``speech_length_file`` 的 ``uid length`` 两列表。

    ``allow_extra_uids=True`` 用于 ``speech_min_sample_rate`` 或
    ``speech_selected_sets`` 过滤后的场景：sidecar 可以覆盖过滤前的全量 speech
    SCP，额外 UID 被忽略；期望 UID 仍必须完整覆盖，缺失仍然 fail-fast。
    """
    lengths: dict[str, int] = {}
    with open(Path(path).expanduser(), encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            location = f"{path}: line {line_number}"
            try:
                parts = shlex.split(raw_line, comments=True)
            except ValueError as exc:
                raise ValueError(f"{location}: invalid shell quoting") from exc
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(
                    f"{location}: expected exactly two columns: uid length"
                )
            uid, length_text = parts
            if uid in lengths:
                raise ValueError(f"{location}: duplicate speech uid {uid!r}")
            if uid not in expected_uids:
                if allow_extra_uids:
                    continue
                raise ValueError(f"{location}: unexpected speech uid {uid!r}")
            try:
                length = int(length_text)
            except ValueError as exc:
                raise ValueError(
                    f"{location}: length must be an integer > 0, got {length_text!r}"
                ) from exc
            if length <= 0:
                raise ValueError(
                    f"{location}: length must be an integer > 0, got {length_text!r}"
                )
            lengths[uid] = length

    missing_uids = expected_uids - lengths.keys()
    if missing_uids:
        examples = ", ".join(repr(uid) for uid in sorted(missing_uids)[:5])
        raise ValueError(
            f"{path}: missing {len(missing_uids)} speech UID(s): {examples}"
        )
    return lengths


def _filter_speech_by_min_sample_rate(
    speech_source: dict,
    speech_uids: dict,
    speech_flt: dict,
    sp_lens: Optional[dict],
    speech_metadata: dict,
    min_sample_rate: int,
) -> tuple[dict, dict, dict, Optional[dict], dict]:
    """按官方 UniPASE/GAP 的 ``fs >= default_fs`` 规则过滤源语音。

    只保留 ``fs >= min_sample_rate`` 的 source，并同步裁剪 uid 列表、flat
    路径表、长度映射与 manifest metadata。返回值全部是新建容器，不修改
    ``read_source_scp`` 的缓存对象；过滤后无任何语音时 fail-fast。
    """
    retained_fs = [fs for fs in speech_source if fs >= min_sample_rate]
    if not retained_fs:
        raise ValueError(
            f"speech_min_sample_rate={min_sample_rate} filtered out every speech "
            f"source; original samplerates={sorted(speech_source)!r}"
        )
    retained_uids = {
        uid for fs in retained_fs for uid in speech_uids.get(fs, ())
    }
    filtered_source = defaultdict(
        dict, {fs: dict(speech_source[fs]) for fs in retained_fs}
    )
    filtered_uids = {fs: list(speech_uids[fs]) for fs in retained_fs}
    filtered_flt = {
        uid: path for uid, path in speech_flt.items() if uid in retained_uids
    }
    filtered_lengths = (
        None
        if sp_lens is None
        else {uid: length for uid, length in sp_lens.items() if uid in retained_uids}
    )
    filtered_metadata = {
        uid: metadata
        for uid, metadata in speech_metadata.items()
        if uid in retained_uids
    }
    return (
        filtered_source,
        filtered_uids,
        filtered_flt,
        filtered_lengths,
        filtered_metadata,
    )


_OFFICIAL_DEFAULT_SPEECH_SETS = (
    "dns5_fullband",
    "vctk",
    "libritts",
    "commonvoice",
    "mls_segments",
    "ears",
)


def _normalize_speech_selected_sets(value: object) -> tuple[str, ...] | None:
    """校验并冻结 split 级 ``speech_selected_sets``（官方 ``selected_set``）。

    ``None`` / ``[]`` 表示不过滤；``"all"`` 按官方 ``DEFAULT_SPEECH_SET`` 展开为
    六个语料标签。非空 list 的每个元素必须是非空、无首尾空白的 ``str``，且不得
    重复；匹配语义与官方 ``s in audio_path`` 一致：大小写敏感的子串匹配，不做
    路径归一化。
    """
    if value is None:
        return None
    if isinstance(value, str) and value == "all":
        return _OFFICIAL_DEFAULT_SPEECH_SETS
    if not isinstance(value, list):
        raise TypeError(
            "speech_selected_sets must be a list[str], 'all', or None, got "
            f"{type(value).__name__}: {value!r}"
        )
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(
                "speech_selected_sets items must be str, got "
                f"{type(item).__name__} at index {index}: {item!r}"
            )
        if not item or item != item.strip():
            raise ValueError(
                "speech_selected_sets items must be non-empty trimmed strings, "
                f"got {item!r} at index {index}"
            )
        if item in normalized:
            raise ValueError(f"speech_selected_sets duplicate item {item!r}")
        normalized.append(item)
    return tuple(normalized) or None


def _filter_speech_by_selected_sets(
    speech_source: dict,
    speech_uids: dict,
    speech_flt: dict,
    sp_lens: Optional[dict],
    speech_metadata: dict,
    selected_sets: tuple[str, ...],
) -> tuple[dict, dict, dict, Optional[dict], dict]:
    """按官方 ``selected_set`` 语义过滤源语音：``audio_path`` 命中任一标签。

    与官方 ``loaders/dataloader.py`` 的 ``s in audio_path`` 一致，匹配大小写敏感，
    不做路径归一化。只保留命中 UID，并丢弃因此变空的采样率分组；同步裁剪 uid
    列表、flat 路径表、长度映射与 manifest metadata。返回值全部是新建容器，不修改
    输入；过滤后无任何语音时 fail-fast 并报告配置。
    """
    retained_source: dict[int, dict[str, str]] = {}
    retained_uids_by_fs: dict[int, list[str]] = {}
    retained_uids: set[str] = set()
    for fs, entries in speech_source.items():
        kept = {
            uid: path
            for uid, path in entries.items()
            if any(tag in path for tag in selected_sets)
        }
        if not kept:
            continue
        retained_source[fs] = kept
        retained_uids_by_fs[fs] = [
            uid for uid in speech_uids.get(fs, ()) if uid in kept
        ]
        retained_uids.update(retained_uids_by_fs[fs])
    if not retained_uids:
        raise ValueError(
            f"speech_selected_sets={list(selected_sets)!r} filtered out every "
            f"speech source; original {len(speech_flt)} UID(s) across "
            f"{len(speech_source)} samplerate(s)"
        )
    filtered_flt = {
        uid: path for uid, path in speech_flt.items() if uid in retained_uids
    }
    filtered_lengths = (
        None
        if sp_lens is None
        else {uid: length for uid, length in sp_lens.items() if uid in retained_uids}
    )
    filtered_metadata = {
        uid: metadata
        for uid, metadata in speech_metadata.items()
        if uid in retained_uids
    }
    return (
        retained_source,
        retained_uids_by_fs,
        filtered_flt,
        filtered_lengths,
        filtered_metadata,
    )


class BaseDynamicMixingDataset(torch.utils.data.Dataset):
    """Shared helpers for on-the-fly dynamic mixing datasets."""

    export_simulation_stems = False
    _simulation_task = "se"
    # ``full`` 保持 SE/AEC/TSE/SS 的完整资源图；``speech_only`` 供 feature-domain
    # SSL 组合使用，跳过通用 AcousticRenderer/ScenarioSampler 和 music 路由；当
    # 配置显式提供 noise/wind 清单时仍加载这些背景资源。该属性是代码级 profile，
    # 不是可由 YAML 注入的运行时键。
    _resource_mode = "full"

    def __init__(
        self,
        speech_scp: str,
        noise_scp: str,
        rir_scp: str,
        windnoise_scp: Optional[str] = None,
        speech_length_file: Optional[str] = None,
        simulation_cfg: Optional[dict] = None,
        strict_load: bool = True,
        max_duration: int = 240000,
        max_duration_seconds: Optional[float] = None,
        repeat_to_max_duration: bool = False,
        repeat_tile_silence_gap: bool = False,
        repeat_tile_silence_gap_range: Optional[list] = None,
        filter_by_duration: bool = True,
        filter_min_duration: float = 0.5,
        filter_max_duration: float = 30.0,
        target_sr: Optional[int] = None,
        speech_min_sample_rate: Optional[int] = None,
        speech_selected_sets: Optional[list] = None,
        placeholder_num_samples: int = DEFAULT_PLACEHOLDER_NUM_SAMPLES,
        export_simulation_stems: bool = False,
        simulation_cfg_strict: bool = True,
        sampling_cfg: Any = None,
        music_scp: Optional[str] = None,
        spatial_rir_provider: Any = None,
    ):
        super().__init__()
        resource_mode = getattr(self, "_resource_mode", "full")
        if resource_mode not in {"full", "speech_only"}:
            raise ValueError(
                "unsupported dynamic dataset resource mode: "
                f"{resource_mode!r}"
            )
        bool_args = {
            "strict_load": strict_load,
            "repeat_to_max_duration": repeat_to_max_duration,
            "repeat_tile_silence_gap": repeat_tile_silence_gap,
            "filter_by_duration": filter_by_duration,
            "export_simulation_stems": export_simulation_stems,
            "simulation_cfg_strict": simulation_cfg_strict,
        }
        for name, value in bool_args.items():
            if type(value) is not bool:
                raise ValueError(f"{name} must be a bool, got {value!r}")
        if type(max_duration) is not int:
            raise ValueError(f"max_duration must be an exact int, got {max_duration!r}")
        if max_duration_seconds is not None and (
            type(max_duration_seconds) is bool
            or not isinstance(max_duration_seconds, Real)
            or not math.isfinite(float(max_duration_seconds))
            or max_duration_seconds <= 0
        ):
            raise ValueError(
                "max_duration_seconds must be a finite number > 0, "
                f"got {max_duration_seconds!r}"
            )
        if target_sr is not None and (type(target_sr) is not int or target_sr <= 0):
            raise ValueError(f"target_sr must be an exact int > 0, got {target_sr!r}")
        if speech_min_sample_rate is not None:
            require_positive_int(speech_min_sample_rate, "speech_min_sample_rate")
        selected_sets = _normalize_speech_selected_sets(speech_selected_sets)
        if type(placeholder_num_samples) is not int or placeholder_num_samples <= 0:
            raise ValueError(
                "placeholder_num_samples must be an exact int > 0, "
                f"got {placeholder_num_samples!r}"
            )
        for name, value in (
            ("filter_min_duration", filter_min_duration),
            ("filter_max_duration", filter_max_duration),
        ):
            if (
                type(value) is bool
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        if filter_min_duration < 0:
            raise ValueError("filter_min_duration must be >= 0")
        if filter_max_duration > 0 and filter_min_duration > filter_max_duration:
            raise ValueError("filter_min_duration must be <= filter_max_duration")
        if repeat_tile_silence_gap_range is not None and (
            not isinstance(repeat_tile_silence_gap_range, (list, tuple))
            or len(repeat_tile_silence_gap_range) != 2
            or any(
                type(value) is bool
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in repeat_tile_silence_gap_range
            )
            or repeat_tile_silence_gap_range[0] < 0
            or repeat_tile_silence_gap_range[0] > repeat_tile_silence_gap_range[1]
        ):
            raise ValueError(
                "repeat_tile_silence_gap_range must be an ordered 2-element "
                f"finite non-negative range, got {repeat_tile_silence_gap_range!r}"
            )

        self.strict_load = strict_load
        self.resource_store = ResourceStore(strict_load=self.strict_load)
        self.speech_preparation = SpeechPreparation()
        self.acoustic_renderer = (
            AcousticRenderer() if resource_mode == "full" else None
        )
        self.scenario_sampler = (
            ScenarioSampler() if resource_mode == "full" else None
        )
        validate_simulation_config_contract(
            task=self._simulation_task,
            split="train",
            train_section={"simulation_cfg": simulation_cfg or {}},
        )
        self.sim_cfg = SimulationConfigs.from_dict(
            simulation_cfg or {}, strict=simulation_cfg_strict
        )
        cfg = self.sim_cfg
        if cfg.rir_backend == "online" and rir_scp is not None:
            raise ValueError(
                "rir_backend='online' does not consume split-level rir_scp; "
                "remove the unused offline RIR resource"
            )
        if cfg.rir_backend == "offline" and cfg.spatial_enabled and not rir_scp:
            raise ValueError("offline spatial simulation requires split-level rir_scp")
        self._simulation_view = project_simulation_view(
            cfg, task=self._simulation_task
        )

        # Subclass-specific scenario probability validation. TSE 在该 callback 前
        # 已挂载资源字段，原初始化顺序保持不变。
        self._validate_simulation_cfg(cfg)

        self.balanced_speech_sampling_enabled = bool(
            sampling_cfg is not None and getattr(sampling_cfg, "speech", None) is not None
        )
        self._speech_sampling_config = (
            getattr(sampling_cfg, "speech", None) if self.balanced_speech_sampling_enabled else None
        )
        self.balanced_noise_sampling_enabled = bool(
            sampling_cfg is not None
            and getattr(sampling_cfg, "noise", None) is not None
            and getattr(sampling_cfg.noise, "balance_categories", False)
        )
        self._noise_sampling_config = (
            sampling_cfg.noise if self.balanced_noise_sampling_enabled else None
        )
        if self.balanced_noise_sampling_enabled and getattr(cfg, "reuse_noise", False):
            raise ValueError("sampling.noise.balance_categories cannot be used with reuse_noise=True")

        # Load source manifests. Speech metadata is loaded for opt-in balancing,
        # or unconditionally when a subclass requires it (TSE strict JSONL
        # canonical speaker identity does not depend on sampling being enabled).
        want_speech_metadata = (
            self.balanced_speech_sampling_enabled
            or self._always_load_speech_manifest_metadata(speech_scp)
        )
        speech_result = read_source_scp(
            speech_scp, return_metadata=want_speech_metadata
        )
        if want_speech_metadata:
            (
                self.speech_source,
                self.speech_uids,
                self.speech_flt,
                sp_lens,
                self._speech_manifest_metadata,
            ) = speech_result
        else:
            self.speech_source, self.speech_uids, self.speech_flt, sp_lens = speech_result
        # 官方 UniPASE/GAP dataloader 只接受 fs >= default_fs 且 audio_path 命中
        # selected_set 的源语音；两类过滤都必须在 samplerates/_uid_to_fs/
        # _fs_sub_lens 等派生状态之前完成，使采样、时长上限与 get_srs 都只看到
        # 保留源。两者都是 per-UID 保留，顺序执行结果等价。
        if speech_min_sample_rate is not None:
            (
                self.speech_source,
                self.speech_uids,
                self.speech_flt,
                sp_lens,
                filtered_speech_metadata,
            ) = _filter_speech_by_min_sample_rate(
                self.speech_source,
                self.speech_uids,
                self.speech_flt,
                sp_lens,
                getattr(self, "_speech_manifest_metadata", {}),
                speech_min_sample_rate,
            )
            if want_speech_metadata:
                self._speech_manifest_metadata = filtered_speech_metadata
        if selected_sets is not None:
            (
                self.speech_source,
                self.speech_uids,
                self.speech_flt,
                sp_lens,
                filtered_speech_metadata,
            ) = _filter_speech_by_selected_sets(
                self.speech_source,
                self.speech_uids,
                self.speech_flt,
                sp_lens,
                getattr(self, "_speech_manifest_metadata", {}),
                selected_sets,
            )
            if want_speech_metadata:
                self._speech_manifest_metadata = filtered_speech_metadata
        if resource_mode == "speech_only" and (
            music_scp is not None or cfg.music_ratio > 0
        ):
            raise ValueError(
                "speech_only dynamic resources do not support music_scp or "
                "music_ratio; use an explicit noise/wind background resource"
            )
        if resource_mode == "speech_only" and noise_scp is None and windnoise_scp is None:
            requested_background = (
                float(getattr(cfg, "noise_mix_prob", 0.0)) > 0.0
                or float(getattr(cfg, "prob_point_source_background", 0.0)) > 0.0
                or float(getattr(cfg, "prob_wind_noise_background", 0.0)) > 0.0
                or float(getattr(cfg, "prob_wind_noise_sidechain", 0.0)) > 0.0
                or float(getattr(cfg, "prob_colored_noise", 0.0)) > 0.0
            )
            if requested_background:
                raise ValueError(
                    "speech_only dynamic resources require noise_scp or "
                    "windnoise_scp when a background probability is positive"
                )
            # 没有任何背景清单时保留空容器，避免共享 sampler/审计 helper 看到
            # 半初始化属性；只要调用方提供 noise/wind 清单，下面的公共读取逻辑
            # 仍会建立背景资源，保持 SSL observation 仿真的原有语义。
            self.noise_source = {}
            self.noise_uids = {}
            self.noise_flt = {}
            self._noise_lengths = {}
            self._noise_manifest_metadata = {}
            self.wind_noises = {}
            self.wn_uids = {}
            self.wn_flt = {}
            self._wind_lengths = {}
            self.rirs = {}
            self.rir_uids = {}
            self.rirs_flt = {}
            self._rir_lengths = {}
            self._music_routing_enabled = False
        else:
            # 训练 runtime 只校验基础 manifest schema/category；Noise Filter policy、
            # admission decision 与 artifact fingerprint 仅由显式 publication 门禁负责。
            if noise_scp is None:
                from core.data.noise_manifest_adapter import NoiseManifestCapabilities, ParsedNoiseManifest

                parsed_noise = ParsedNoiseManifest({}, {}, {}, {}, {}, NoiseManifestCapabilities(False, False, False))
            else:
                parsed_noise = read_training_noise_manifest(noise_scp)
            noise_has_metadata = parsed_noise.capabilities.has_metadata
            wind_required = self._wind_required(cfg)
            self.noise_source = parsed_noise.by_fs
            self.noise_uids = parsed_noise.uids
            self.noise_flt = parsed_noise.flat
            self._noise_lengths = parsed_noise.lengths
            self._noise_manifest_metadata = self._project_noise_metadata(
                parsed_noise.metadata
            )
            # side-chain 风噪池来源：显式四列 windnoise_scp，或分类 JSONL 内嵌风噪类别。
            if windnoise_scp is not None:
                self.wind_noises, self.wn_uids, self.wn_flt, self._wind_lengths = read_source_scp(
                    windnoise_scp
                )
            else:
                self.wind_noises, self.wn_uids, self.wn_flt, self._wind_lengths = {}, {}, {}, {}
            if noise_has_metadata:
                self._extract_jsonl_role(
                    role="wind",
                    prefix="jsonl_wind_",
                    target_source=self.wind_noises,
                    target_uids=self.wn_uids,
                    target_flat=self.wn_flt,
                    target_lengths=self._wind_lengths,
                )
            if wind_required and not self.wn_flt:
                raise ValueError(
                    "wind-noise probability is positive but no wind-noise samples were found; "
                    "ordinary sources require an explicit non-empty windnoise_scp, while "
                    "classified JSONL may embed the wind role"
                )
            if not self._defer_noise_sampling_state():
                if resource_mode == "full" and (music_scp is not None or cfg.music_ratio > 0):
                    from core.data.background_sources import initialize_music_pool

                    initialize_music_pool(self, music_scp, extract_embedded=noise_has_metadata)
                else:
                    self._build_noise_sampling_state()
            self.rirs, self.rir_uids, self.rirs_flt, self._rir_lengths = (
                read_source_scp(rir_scp)
                if rir_scp is not None
                and cfg.rir_backend == "offline"
                and not cfg.spatial_enabled
                else ({}, {}, {}, {})
            )

        # Merge noise sources with prefixes to avoid uid collisions
        self._noise_uid_prefix = "noise:"
        self._wind_uid_prefix = "wind:"
        self.all_noise_flt = {
            f"{self._noise_uid_prefix}{k}": v for k, v in self.noise_flt.items()
        }
        self.all_noise_flt.update(
            {f"{self._wind_uid_prefix}{k}": v for k, v in self.wn_flt.items()}
        )

        self.samplerates = list(self.speech_source.keys())
        self.max_duration_seconds = (
            float(max_duration_seconds) if max_duration_seconds is not None else None
        )
        # uid -> fs for per-sample max_duration caps
        self._uid_to_fs = {
            uid: fs for fs in self.samplerates for uid in self.speech_uids[fs]
        }

        if self.max_duration_seconds is not None:
            max_duration = int(self.max_duration_seconds * max(self.samplerates))

        def _cap_for_uid(uid: str, raw_len: int) -> int:
            if self.max_duration_seconds is not None:
                fs = self._uid_to_fs.get(uid, max(self.samplerates))
                return min(raw_len, int(self.max_duration_seconds * fs))
            if max_duration <= 0:
                return raw_len
            return min(raw_len, max_duration)

        # 构建完整 uid -> length 映射；strict 模式禁止运行时探测掩盖清单缺失。
        expected_speech_uids = set(self.speech_flt)
        if speech_length_file:
            sidecar_lengths = _read_speech_length_file(
                speech_length_file,
                expected_speech_uids,
                allow_extra_uids=(
                    speech_min_sample_rate is not None or selected_sets is not None
                ),
            )
            self.source_length = {
                uid: _cap_for_uid(uid, length)
                for uid, length in sidecar_lengths.items()
            }
        elif sp_lens is not None:
            self.source_length = {
                uid: _cap_for_uid(uid, length)
                for uid, length in sp_lens.items()
            }
        elif self.strict_load:
            raise ValueError(
                "strict_load=True requires complete speech length metadata in "
                "speech_scp or a complete speech_length_file"
            )
        else:
            logger.info(
                "speech_scp has no complete length metadata; probing with sf.info."
            )
            self.source_length = {}
            for uid, audio_path in self.speech_flt.items():
                try:
                    nf = soundfile.info(str(Path(audio_path).expanduser())).frames
                    self.source_length[uid] = _cap_for_uid(uid, nf)
                except Exception as exc:
                    logger.warning("sf.info(%s) failed: %s", audio_path, exc)
                    if self.max_duration_seconds is not None:
                        cap = int(self.max_duration_seconds * self._uid_to_fs[uid])
                    elif max_duration > 0:
                        cap = max_duration
                    else:
                        cap = int(30.0 * self._uid_to_fs[uid])
                    self.source_length[uid] = cap

        actual_speech_uids = set(self.source_length)
        if actual_speech_uids != expected_speech_uids:
            missing = expected_speech_uids - actual_speech_uids
            unexpected = actual_speech_uids - expected_speech_uids
            raise ValueError(
                "speech length metadata must cover speech UIDs exactly; "
                f"missing={sorted(missing)[:5]!r}, unexpected={sorted(unexpected)[:5]!r}"
            )

        self.max_duration = (
            int(self.max_duration_seconds * max(self.samplerates))
            if self.max_duration_seconds is not None
            else max_duration
        )
        self.repeat_to_max_duration = bool(repeat_to_max_duration)
        self.repeat_tile_silence_gap = bool(repeat_tile_silence_gap)
        gap_range = repeat_tile_silence_gap_range
        self.repeat_tile_silence_gap_range = (
            (float(gap_range[0]), float(gap_range[1]))
            if gap_range and len(gap_range) >= 2
            else (0.02, 0.2)
        )
        self.canvas_spec = CanvasSpec(
            max_duration_seconds=self.max_duration_seconds,
            max_duration_samples=int(self.max_duration),
            repeat_to_max_duration=self.repeat_to_max_duration,
            repeat_tile_silence_gap=self.repeat_tile_silence_gap,
            repeat_tile_silence_gap_range=self.repeat_tile_silence_gap_range,
        )
        self.placeholder_num_samples = max(1, int(placeholder_num_samples))
        self.target_sr = int(target_sr) if target_sr is not None else None
        self.speech_min_sample_rate = (
            int(speech_min_sample_rate) if speech_min_sample_rate is not None else None
        )
        self.speech_selected_sets = selected_sets
        self.filter_by_duration = bool(filter_by_duration)
        self.filter_min_duration = float(filter_min_duration)
        self.filter_max_duration = float(filter_max_duration)
        self.export_simulation_stems = bool(export_simulation_stems)
        self._last_simulation_stems: dict | None = None
        self._last_simulation_trace: SimulationTrace | None = None

        # Sanity check: splice target must fit within per-segment cap。模式 B
        # （silence_pad_prob=1）固定使用 single utterance，不执行 splice。
        if (
            getattr(self.sim_cfg, "speech_splice", False)
            and float(getattr(self.sim_cfg, "silence_pad_prob", 0.0)) < 1.0
        ):
            # 与 _get_clean_speech() 运行时取值口径一致：
            # 设置了 speech_splice_seconds_range 时按 range 上限（range[1]）检查，
            # 否则按固定值 speech_splice_seconds 检查。
            seconds_range = getattr(self.sim_cfg, "speech_splice_seconds_range", None)
            if seconds_range and len(seconds_range) >= 2:
                splice_seconds = float(seconds_range[1])
            else:
                splice_seconds = float(self.sim_cfg.speech_splice_seconds)
            ts = self.target_sr
            if ts is not None and self.max_duration_seconds is not None:
                eff_cap = int(self.max_duration_seconds * ts)
                min_required = int(ts * splice_seconds)
            elif self.max_duration_seconds is not None:
                max_sr = max(self.samplerates)
                eff_cap = int(self.max_duration_seconds * max_sr)
                min_required = int(max_sr * splice_seconds)
            else:
                max_sr = int(ts) if ts is not None else max(self.samplerates)
                eff_cap = max_duration
                min_required = int(max_sr * splice_seconds)
            if eff_cap > 0 and eff_cap < min_required:
                raise ValueError(
                    f"speech_splice: segment cap ({eff_cap}) < splice length ({min_required}). "
                    f"Increase max_duration_seconds or max_duration, or decrease "
                    f"speech_splice_seconds / speech_splice_seconds_range[1]."
                )

        self._fs_sub_lens = [len(self.speech_source[k]) for k in self.samplerates]
        self._accum = [
            sum(self._fs_sub_lens[: i + 1]) for i in range(len(self._fs_sub_lens))
        ]
        total_before_filter = sum(self._fs_sub_lens)

        # Build filtered index list for bucketing
        if self.filter_by_duration:
            self._filtered_indices = []
            for fs in self.samplerates:
                for idx in range(len(self.speech_uids[fs])):
                    uid = self.speech_uids[fs][idx]
                    slen = self.source_length[uid]
                    min_len = int(self.filter_min_duration * fs)
                    max_len = int(self.filter_max_duration * fs)
                    if min_len <= slen <= max_len:
                        self._filtered_indices.append((fs, idx))
            if not self._filtered_indices:
                raise ValueError(
                    f"filter_by_duration=True: no samples in range "
                    f"[{self.filter_min_duration}s, {self.filter_max_duration}s]."
                )
            logger.info(
                "Duration filter: kept %d of %d samples",
                len(self._filtered_indices),
                total_before_filter,
            )
        else:
            self._filtered_indices = [
                (fs, idx)
                for fs in self.samplerates
                for idx in range(len(self.speech_uids[fs]))
            ]
        self._total = len(self._filtered_indices)
        self._fixed_speech_plan_key: tuple | None = None
        self._fixed_speech_plan: SpeechPlan | None = None
        self._speaker_splice_identity_by_uid: dict[str, tuple[str, str]] = {}
        self._speaker_splice_uids_by_identity: dict[tuple[str, str], tuple[str, ...]] = {}
        if self.balanced_speech_sampling_enabled:
            self.speech_sampling_leaves = self._build_speech_sampling_leaves()
            self.speech_sampling_cells = _speech_cells_from_leaves(
                self.speech_sampling_leaves
            )
            if (
                self._speech_sampling_config is not None
                and self._speech_sampling_config.speaker is not None
            ):
                identity_uids: dict[tuple[str, str], list[str]] = {}
                invalid_identities: list[tuple[str, tuple[str, ...]]] = []
                for uid in self.speech_flt:
                    metadata = self._speech_manifest_metadata.get(uid, {})
                    invalid_fields = tuple(
                        field
                        for field in ("dataset_id", "speaker_id")
                        if not (
                            isinstance(metadata.get(field), str)
                            and metadata[field]
                            and metadata[field] == metadata[field].strip()
                        )
                    )
                    if invalid_fields:
                        invalid_identities.append((uid, invalid_fields))
                        continue
                    identity = (metadata["dataset_id"], metadata["speaker_id"])
                    self._speaker_splice_identity_by_uid[uid] = identity
                    identity_uids.setdefault(identity, []).append(uid)
                if invalid_identities:
                    examples = ", ".join(
                        f"{uid!r} ({'/'.join(fields)})"
                        for uid, fields in invalid_identities[:5]
                    )
                    raise ValueError(
                        "speaker sampling splice pool metadata invalid for "
                        f"{len(invalid_identities)} manifest UID(s): {examples}"
                    )
                self._speaker_splice_uids_by_identity = {
                    identity: tuple(uids) for identity, uids in identity_uids.items()
                }

        # Augmentation name -> weight mapping for Layer E
        aw = np.array(
            [self.sim_cfg.augmentations[a]["weight"] for a in self.sim_cfg.augmentations_name]
        )
        self._aug_names = list(self.sim_cfg.augmentations_name)
        self._aug_weights = aw / aw.sum()

        self._hp_taps: dict[int, np.ndarray] | None = None
        self._all_speech_uids = list(self.speech_flt)
        self._all_speech_paths = list(self.speech_flt.values())
        self._epoch = 0
        self._base_seed = 0

        # Shared-memory epoch/base-seed tokens: lets persistent DataLoader workers
        # observe every set_epoch() update.  set_epoch() runs in the main
        # process; both fork and spawn semantics prevent it from clearing
        # worker-side caches.  A torch shared-memory tensor is used (not
        # mp.RawValue) because RawValue cannot be pickled by ForkingPickler and
        # therefore crashes DataLoader workers under the "spawn" start method
        # (Windows / macOS default).  share_memory_() reduces to a shared handle
        # that the worker maps to the same memory, so main-process writes remain
        # visible.  Single writer (main) + coarse int reads (workers) needs no lock.
        self._worker_epoch = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._worker_base_seed = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._worker_last_seen_epoch: tuple[int, int] | None = None

        # Reuse caches (cleared on set_epoch for reproducibility)
        self._rir_reuse_cache: (
            tuple[int, np.ndarray]
            | tuple[int, np.ndarray, float | None, str | None]
            | None
        ) = None
        self._noise_only_reuse_cache: tuple[int, np.ndarray] | None = None
        self._noise_single_reuse_cache: tuple[int, np.ndarray] | None = None
        self._noise_dual_reuse_cache: tuple[int, np.ndarray] | None = None
        self._wind_reuse_cache: tuple[int, np.ndarray] | None = None
        self._last_wind_source_identity: str | None = None
        self._last_run_branch: str | None = None
        self._online_rir_provider = None
        self._spatial_rir_provider = None
        if self._needs_online_rir_provider(cfg):
            from core.data.simulation.rir_provider import OnlineRIRProvider

            self._online_rir_provider = OnlineRIRProvider.from_simulation(cfg)
        if cfg.spatial_enabled:
            from core.data.simulation.rir_provider import (
                build_spatial_rir_provider,
                validate_spatial_rir_provider,
            )

            if spatial_rir_provider is not None:
                self._spatial_rir_provider = validate_spatial_rir_provider(
                    spatial_rir_provider
                )
            elif cfg.rir_backend == "online" and self._online_rir_provider is not None:
                self._spatial_rir_provider = validate_spatial_rir_provider(
                    self._online_rir_provider
                )
            else:
                self._spatial_rir_provider = build_spatial_rir_provider(cfg, rir_scp=rir_scp)

    def __getstate__(self):
        """Strip construction-only metadata before pickling to DataLoader workers.

        Large manifest metadata / length dicts are only needed in the main
        process (summary reporting / speech plan building / noise category
        index construction).  Workers only need the uid->path mappings and the
        pre-built category index when balanced noise sampling is enabled.
        """
        state = self.__dict__.copy()
        # Speech metadata is consumed by _build_speech_plan in set_epoch (main process).
        state["_speech_manifest_metadata"] = {}
        # role 拆池、category index 与 summary 均在构造期完成；worker 只保留
        # 预建索引、uid->category 与 uid->path，避免复制原始 manifest 大字段。
        state["_noise_manifest_metadata"] = {}
        # Length dicts are only used by get_data_summary (main process).
        state["_noise_lengths"] = {}
        state["_wind_lengths"] = {}
        state["_rir_lengths"] = {}
        state["_music_lengths"] = {}
        # 候选 view 可由 worker 保留的轻量索引按需重建，避免把主进程热 cache
        # 连同对象图一起复制；冷热路径使用同一稳定顺序，不影响 fixed-seed 序列。
        state["_noise_category_tier_candidates_cache"] = OrderedDict()
        state["_noise_category_tier_candidates_cache_uid_refs"] = 0
        state["_noise_category_order_cache"] = {}
        state["_noise_temperature_base_weights_cache"] = {}
        state["_noise_natural_tier_uids_cache"] = OrderedDict()
        state["_noise_natural_tier_uids_cache_uid_refs"] = 0
        # source_by_fs 会随 dataset 一起序列化；其派生 tuple/view 必须由 worker
        # 基于 worker 内对象重新 lazy 构建，不能保留指向主进程 source 对象的 cache。
        state["_sample_selector_source_cache"] = OrderedDict()
        state["_sample_selector_cache_uid_refs"] = 0
        return state

    def _validate_simulation_cfg(self, cfg: SimulationConfigs) -> None:
        """Hook for SE/AEC/TSE subclasses to validate scenario probabilities.

        Raises ValueError if the configuration is invalid. Base implementation
        does nothing; all three task-specific dynamic datasets override it.
        """
        pass

    def _has_speech_for_rir(self) -> bool:
        """任务可在静态配置已排除所有语音时跳过在线服务启动。"""
        return True

    def _needs_online_rir_provider(self, cfg: SimulationConfigs) -> bool:
        """子类钩子：任务可因自身 RIR 需求（如 TSE enrollment 在线生成）扩展构造条件。"""
        return (
            cfg.rir_backend == "online"
            and self._has_speech_for_rir()
            and (
                cfg.prob_reverberation > 0
                or (
                    cfg.spatial_enabled
                    and cfg.rir_backend == "online"
                )
            )
        )

    def _wind_noise_scenario_probs(self) -> tuple:
        """子类钩子：含 noise 的场景概率；空 tuple 表示不做场景级判定（SE/AEC）。"""
        return ()

    def _wind_required(self, cfg: SimulationConfigs) -> bool:
        """根据 side-chain 或 spatial additive background 配置判断是否需要风噪池。"""
        from core.data.simulation.background import wind_pool_required

        return wind_pool_required(
            getattr(cfg, "prob_wind_noise_sidechain", 0.0),
            self._wind_noise_scenario_probs(),
            getattr(cfg, "prob_wind_noise_background", 0.0),
        )

    def _always_load_speech_manifest_metadata(self, speech_scp) -> bool:
        """子类钩子：无论 sampling 是否启用都加载 speech manifest metadata。

        TSE strict JSONL 的 canonical speaker identity 以整份 manifest 的
        metadata 为主事实来源，不依赖 balanced sampling 是否启用；基类默认
        False（SE/AEC 保持仅 opt-in 加载）。
        """
        return False

    def _defer_noise_sampling_state(self) -> bool:
        """子类钩子：是否在任务专属 role 拆池后再构建 ordinary 分类状态。"""
        return False

    def _build_speech_sampling_leaves(
        self,
    ) -> dict[tuple[str, str, tuple[str, str] | None, int], SpeechSamplingLeaf]:
        """Group duration-eligible JSONL speech entries by atomic balancing leaf.

        leaf key = ``(language, dataset_id, canonical_speaker_or_none,
        batching_sr)``；``batching_sr`` 取 ``get_srs()``（处理 SR），不用
        manifest ``fs``。speaker 仅在 speaker controller 启用时进入 key，
        此时每条 eligible 记录必须有合法 ``speaker_id``；未启用时统一
        ``None``（同三元多 speaker 聚合为一个 leaf）。同 leaf 内按
        ``(uid, external_index)`` 稳定排序；UID 跨 leaf 重复 fail-fast。
        """
        sample_srs = self.get_srs()
        speaker_enabled = (
            self._speech_sampling_config is not None
            and self._speech_sampling_config.speaker is not None
        )
        grouped: dict[tuple, list[tuple[int, str, Fraction]]] = defaultdict(list)
        invalid_uids: list[str] = []
        seen_uids: set[str] = set()

        for external_index, (native_fs, inner_index) in enumerate(self._filtered_indices):
            uid = self.speech_uids[native_fs][inner_index]
            metadata = self._speech_manifest_metadata.get(uid, {})
            language = metadata.get("language")
            dataset_id = metadata.get("dataset_id")
            valid = all(
                isinstance(value, str) and value and value == value.strip()
                for value in (language, dataset_id)
            )
            speaker = None
            if valid and speaker_enabled:
                speaker_id = metadata.get("speaker_id")
                valid = (
                    isinstance(speaker_id, str)
                    and bool(speaker_id)
                    and speaker_id == speaker_id.strip()
                )
                if valid:
                    speaker = (dataset_id, speaker_id)
            if not valid:
                invalid_uids.append(uid)
                continue
            if uid in seen_uids:
                raise ValueError(
                    f"speech sampling duplicate UID across leaves: {uid!r}"
                )
            seen_uids.add(uid)
            grouped[(language, dataset_id, speaker, int(sample_srs[external_index]))].append(
                (
                    external_index,
                    uid,
                    Fraction(int(metadata["length"]), int(metadata["fs"])),
                )
            )

        if invalid_uids:
            examples = ", ".join(repr(uid) for uid in invalid_uids[:5])
            raise ValueError(
                f"speech sampling metadata invalid for {len(invalid_uids)} eligible UID(s): {examples}"
            )

        return {
            key: SpeechSamplingLeaf(
                language=key[0],
                dataset_id=key[1],
                speaker=key[2],
                batching_sr=key[3],
                external_indices=tuple(item[0] for item in entries),
                uids=tuple(item[1] for item in entries),
                effective_seconds=sum((item[2] for item in entries), Fraction(0)),
            )
            for key, entries in (
                (key, sorted(grouped[key], key=lambda item: (item[1], item[0])))
                for key in sorted(grouped, key=_speech_leaf_key_sort)
            )
        }

    @staticmethod
    def _project_noise_metadata(metadata_by_uid: dict[str, dict]) -> dict[str, dict]:
        """只保留训练分类、role 拆池与音频定位所需字段。"""
        fields = (
            "uid",
            "fs",
            "length",
            "path",
            "category_id",
            "balance_category",
            "category_name",
            "roles",
        )
        return {
            uid: {
                key: copy.deepcopy(item[key])
                for key in fields
                if key in item
            }
            for uid, item in metadata_by_uid.items()
            if isinstance(item, dict)
        }

    def _extract_jsonl_role(
        self,
        role: str,
        prefix: str,
        target_source: dict[int, dict[str, str]],
        target_uids: dict[int, list[str]],
        target_flat: dict[str, str],
        target_lengths: dict[str, int] | None = None,
    ) -> None:
        """把分类 JSONL 中具有指定 role 的记录合并进目标池。

        优先读取内嵌 ``roles``；字段缺失时按 ``balance_category`` 或兼容
        ``category_name`` 推导。命中条目从 ordinary noise pool 与轻量 metadata
        中剔除。显式 side SCP 与内嵌 role 因此保持合并扩池语义。
        """
        embedded: dict[str, dict] = {}
        for uid, item in self._noise_manifest_metadata.items():
            if not isinstance(item, dict):
                continue
            roles = item.get("roles")
            if isinstance(roles, list):
                if role in roles:
                    embedded[uid] = item
            elif has_role(category_balance_name(item), role):
                embedded[uid] = item
        if not embedded:
            return
        for uid, item in embedded.items():
            fs = int(item["fs"])
            prefixed = f"{prefix}{uid}"
            if prefixed in target_flat:
                raise ValueError(
                    f"embedded {role} uid collides with explicit source: {prefixed}"
                )
            target_source.setdefault(fs, {})[prefixed] = item["path"]
            target_flat[prefixed] = item["path"]
            if target_lengths is not None and isinstance(item.get("length"), int):
                target_lengths[prefixed] = item["length"]
        target_uids.clear()
        for fs, entries in target_source.items():
            target_uids[fs] = list(entries.keys())
        self._drop_noise_uids(set(embedded))

    def _drop_noise_uids(self, drop: set[str]) -> None:
        """从普通噪声池与均衡采样 metadata 中剔除指定 uid。"""
        if not drop:
            return
        self.noise_flt = {
            uid: path for uid, path in self.noise_flt.items() if uid not in drop
        }
        self.noise_source = {
            fs: {uid: path for uid, path in entries.items() if uid not in drop}
            for fs, entries in self.noise_source.items()
        }
        self.noise_uids = {
            fs: [uid for uid in uids if uid not in drop]
            for fs, uids in self.noise_uids.items()
        }
        self.noise_source = {fs: e for fs, e in self.noise_source.items() if e}
        self.noise_uids = {fs: u for fs, u in self.noise_uids.items() if u}
        self._noise_manifest_metadata = {
            uid: item
            for uid, item in self._noise_manifest_metadata.items()
            if uid not in drop
        }

    def _build_noise_sampling_state(self) -> None:
        """原子重建 ordinary noise 分类状态并清空全部派生 cache。"""
        category_index = self._build_noise_category_index()
        category_id_by_uid = {
            uid: metadata["category_id"]
            for uid, metadata in self._noise_manifest_metadata.items()
            if _noise_category_meta_valid(metadata)
        }
        category_name_by_id: dict[int, str] = {}
        category_id_by_name: dict[str, int] = {}
        inconsistent_mapping = False
        for uid, category_id in category_id_by_uid.items():
            name = category_balance_name(self._noise_manifest_metadata[uid])
            previous_name = category_name_by_id.get(category_id)
            if previous_name is not None and previous_name != name:
                if self.balanced_noise_sampling_enabled:
                    raise ValueError(
                        f"category_id {category_id} maps to multiple category names: "
                        f"{previous_name!r} vs {name!r}"
                    )
                inconsistent_mapping = True
                break
            previous_id = category_id_by_name.get(name)
            if previous_id is not None and previous_id != category_id:
                if self.balanced_noise_sampling_enabled:
                    raise ValueError(
                        f"category name {name!r} maps to multiple category_id values: "
                        f"{previous_id} vs {category_id}"
                    )
                inconsistent_mapping = True
                break
            category_name_by_id[category_id] = name
            category_id_by_name[name] = category_id

        if inconsistent_mapping:
            # natural 模式把任一方向的不一致映射视为能力不完整，整体回退普通读取。
            category_index = {}
            category_id_by_uid = {}
            category_name_by_id = {}
            category_id_by_name = {}

        category_weight_by_id: dict[int, float] = {}
        if (
            self._noise_sampling_config is not None
            and self._noise_sampling_config.balance_mode == "temperature"
        ):
            category_weight_by_id = self._build_noise_temperature_weights(
                category_name_by_id
            )

        natural_uids_by_fs = {
            int(fs): tuple(uids)
            for fs, uids in self.noise_uids.items()
            if uids
        }
        ordinary_uids = tuple(
            uid for uids in natural_uids_by_fs.values() for uid in uids
        )
        categories_complete = bool(ordinary_uids) and all(
            uid in category_id_by_uid for uid in ordinary_uids
        )
        natural_positions_by_fs: dict[int, dict[int, tuple[int, ...]]] = {}
        natural_uid_position_by_fs: dict[int, dict[str, int]] = {}
        if categories_complete:
            for fs, uids in natural_uids_by_fs.items():
                positions: dict[int, list[int]] = defaultdict(list)
                uid_positions: dict[str, int] = {}
                for position, uid in enumerate(uids):
                    positions[category_id_by_uid[uid]].append(position)
                    uid_positions[uid] = position
                natural_positions_by_fs[fs] = {
                    category_id: tuple(values)
                    for category_id, values in positions.items()
                }
                natural_uid_position_by_fs[fs] = uid_positions

        # 所有值先在局部完成并通过校验，再一次性原子替换；重建失败时既有状态保持不变。
        self.__dict__.update(
            {
                "_noise_category_uids_by_fs": category_index,
                "_noise_category_id_by_uid": category_id_by_uid,
                "_noise_category_name_by_id": category_name_by_id,
                "_noise_category_id_by_name": category_id_by_name,
                "_noise_category_weight_by_id": category_weight_by_id,
                "_noise_natural_uids_by_fs": natural_uids_by_fs,
                "_noise_natural_positions_by_fs": natural_positions_by_fs,
                "_noise_natural_uid_position_by_fs": natural_uid_position_by_fs,
                "_noise_categories_complete": categories_complete,
                "_noise_category_tier_candidates_cache": OrderedDict(),
                "_noise_category_tier_candidates_cache_uid_refs": 0,
                "_noise_category_order_cache": {},
                "_noise_temperature_base_weights_cache": {},
                "_noise_natural_tier_uids_cache": OrderedDict(),
                "_noise_natural_tier_uids_cache_uid_refs": 0,
            }
        )

    def _build_noise_category_index(self) -> dict[int, dict[int, tuple[str, ...]]]:
        """按 source FS 和 category 建立稳定 UID 索引。"""
        grouped: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
        invalid_uids: list[str] = []
        for fs, uids in self.noise_uids.items():
            for uid in uids:
                metadata = self._noise_manifest_metadata.get(uid, {})
                if not _noise_category_meta_valid(metadata):
                    invalid_uids.append(uid)
                    continue
                grouped[int(fs)][metadata["category_id"]].append(uid)
        if invalid_uids and self.balanced_noise_sampling_enabled:
            examples = ", ".join(repr(uid) for uid in invalid_uids[:5])
            raise ValueError(
                "sampling.noise.balance_categories requires complete category metadata "
                "(non-negative integer category_id plus non-empty balance_category or "
                f"category_name) for every ordinary noise UID; invalid {len(invalid_uids)} "
                f"UID(s): {examples}"
            )
        return {
            fs: {
                category_id: tuple(sorted(uids))
                for category_id, uids in sorted(categories.items())
            }
            for fs, categories in sorted(grouped.items())
        }

    def _build_noise_temperature_weights(
        self,
        category_name_by_id: dict[int, str] | None = None,
    ) -> dict[int, float]:
        """返回 category→difficulty；tier-local ``n^alpha`` 在选择时计算。"""
        names = (
            self._noise_category_name_by_id
            if category_name_by_id is None
            else category_name_by_id
        )
        missing: set[str] = set()
        difficulties: dict[int, float] = {}
        missing_difficulty: set[str] = set()
        for category_id, name in names.items():
            if name not in NOISE_ATTRIBUTES:
                missing.add(name)
                continue
            statistics, temporal = NOISE_ATTRIBUTES[name][1], NOISE_ATTRIBUTES[name][2]
            key = difficulty_key(statistics, temporal)
            if key not in DEFAULT_DIFFICULTY:
                missing_difficulty.add(key)
                continue
            difficulty = DEFAULT_DIFFICULTY[key]
            if (
                isinstance(difficulty, bool)
                or not isinstance(difficulty, (int, float))
                or not math.isfinite(float(difficulty))
                or difficulty <= 0
            ):
                raise ValueError(f"DEFAULT_DIFFICULTY[{key!r}] must be finite and > 0")
            difficulties[category_id] = float(difficulty)
        if missing:
            examples = ", ".join(sorted(repr(name) for name in missing)[:5])
            raise ValueError(
                f"noise attribute table missing {len(missing)} category name(s): {examples}; "
                "please update core/data/noise_attributes.py via "
                "python -m tools.ops.data.generate_noise_attributes"
            )
        if missing_difficulty:
            examples = ", ".join(sorted(repr(key) for key in missing_difficulty)[:5])
            raise ValueError(
                f"DEFAULT_DIFFICULTY missing {len(missing_difficulty)} (statistics,temporal) key(s): "
                f"{examples}; please update core/data/sampling.py::DEFAULT_DIFFICULTY"
            )
        return difficulties

    def _compute_noise_categories(self, top_k: int = 10) -> list[tuple[str, int, float]] | None:
        """返回分类 JSONL ordinary pool 的 top-k category 分布。"""
        if not self._noise_manifest_metadata:
            return None
        counts: Counter = Counter()
        for uid in self.noise_flt:
            meta = self._noise_manifest_metadata.get(uid, {})
            name = category_balance_name(meta)
            if name is not None:
                counts[name] += 1
        if not counts:
            return None
        total = sum(counts.values())
        return [(name, count, count / total) for name, count in counts.most_common(top_k)]

    def _scenario_probabilities(self) -> dict[str, float] | None:
        """Return scenario branch probabilities when the model defines them."""
        cfg = self.sim_cfg
        known = (
            "prob_target_only",
            "prob_noise_only",
            "prob_echo_only",
            "prob_noise_echo_only",
            "prob_target_noise",
            "prob_target_echo",
            "prob_target_noise_echo",
        )
        probs = {name: float(getattr(cfg, name)) for name in known if hasattr(cfg, name)}
        return probs if probs else None

    def get_data_summary(
        self,
        split: str,
        n_batches: int,
        dataloader_config: dict[str, Any],
    ) -> DataSummary:
        """Build a light-weight data distribution report before training starts.

        Called in the main process (before or after worker spawn is fine because
        ``__getstate__`` does not mutate the original object).
        """
        sources: dict[str, SourceSummary] = {}
        if self.speech_uids:
            sources["speech"] = _compute_source_summary(self.speech_uids, self.source_length)
        if self.noise_uids:
            sources["noise"] = _compute_source_summary(self.noise_uids, self._noise_lengths)
        if self.wn_uids:
            sources["wind"] = _compute_source_summary(self.wn_uids, self._wind_lengths)
        if self.rir_uids:
            sources["rir"] = _compute_source_summary(self.rir_uids, self._rir_lengths)

        return DataSummary(
            split=split,
            n_samples=len(self),
            n_batches=n_batches,
            sources=sources,
            dataloader_config=dataloader_config,
            noise_categories=self._compute_noise_categories(),
            scenario_probs=self._scenario_probabilities(),
            extra={},
        )

    def _noise_category_tier_specs(
        self, sr: int
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """返回 exact→higher→all tier；higher key 仅有有限个 FS 后缀。"""
        target_sr = int(sr)
        sample_rates = tuple(self._noise_category_uids_by_fs)
        return (
            ("exact", (target_sr,) if target_sr in self._noise_category_uids_by_fs else ()),
            ("higher", tuple(fs for fs in sample_rates if fs > target_sr)),
            ("all", sample_rates),
        )

    def _noise_selector_lru_get(self, cache_attr: str, key):
        """读取 selector LRU；命中只更新 recency，不触碰 RNG。"""
        cache = getattr(self, cache_attr)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict(cache)
            setattr(self, cache_attr, cache)
        if key not in cache:
            return None
        cache.move_to_end(key)
        return cache[key]

    def _noise_selector_lru_put(
        self,
        cache_attr: str,
        uid_refs_attr: str,
        key,
        value,
        uid_refs: int,
    ) -> None:
        """在 entry 与 UID-ref 双预算内插入 selector LRU。"""
        if uid_refs > _NOISE_SELECTOR_CACHE_MAX_UID_REFS:
            return
        cache = getattr(self, cache_attr)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict(cache)
            setattr(self, cache_attr, cache)
        total_refs = int(getattr(self, uid_refs_attr, 0))
        previous = cache.pop(key, None)
        if previous is not None:
            previous_refs = (
                sum(len(uids) for uids in previous.values())
                if isinstance(previous, dict)
                else len(previous)
            )
            total_refs -= previous_refs
        while cache and (
            len(cache) >= _NOISE_SELECTOR_CACHE_MAX_ENTRIES
            or total_refs + uid_refs > _NOISE_SELECTOR_CACHE_MAX_UID_REFS
        ):
            _, evicted = cache.popitem(last=False)
            total_refs -= (
                sum(len(uids) for uids in evicted.values())
                if isinstance(evicted, dict)
                else len(evicted)
            )
        if (
            len(cache) >= _NOISE_SELECTOR_CACHE_MAX_ENTRIES
            or total_refs + uid_refs > _NOISE_SELECTOR_CACHE_MAX_UID_REFS
        ):
            setattr(self, uid_refs_attr, total_refs)
            return
        cache[key] = value
        setattr(self, uid_refs_attr, total_refs + uid_refs)

    def _noise_category_tier_view(
        self, tier: str, sample_rates: tuple[int, ...]
    ) -> dict[int, tuple[str, ...]]:
        """返回 category view；exact 复用索引，higher/all 使用双预算 LRU。"""
        if not sample_rates:
            return {}
        if tier == "exact":
            return self._noise_category_uids_by_fs[sample_rates[0]]

        key = (tier, sample_rates)
        cached = self._noise_selector_lru_get(
            "_noise_category_tier_candidates_cache", key
        )
        if cached is not None:
            return cached
        grouped: dict[int, set[str]] = defaultdict(set)
        for fs in sample_rates:
            for category_id, uids in self._noise_category_uids_by_fs[fs].items():
                grouped[category_id].update(uids)
        candidates = {
            category_id: tuple(sorted(uids))
            for category_id, uids in sorted(grouped.items())
            if uids
        }
        uid_refs = sum(len(uids) for uids in candidates.values())
        self._noise_selector_lru_put(
            "_noise_category_tier_candidates_cache",
            "_noise_category_tier_candidates_cache_uid_refs",
            key,
            candidates,
            uid_refs,
        )
        return candidates

    def _noise_category_view_key(
        self, sr: int, tier: str
    ) -> tuple[str, tuple[int, ...]]:
        """把已选 tier 映射到其 canonical cache key。"""
        for candidate_tier, sample_rates in self._noise_category_tier_specs(sr):
            if candidate_tier == tier:
                return candidate_tier, sample_rates
        raise ValueError(f"unknown noise tier: {tier!r}")

    @staticmethod
    def _filter_noise_category_view(
        candidates: Mapping[int, tuple[str, ...]],
        *,
        exclude_category_id: int | None,
        exclude_uid: str | None,
    ) -> Mapping[int, tuple[str, ...]]:
        """按 category/UID 返回零复制只读 view，保持 category 与 UID 稳定顺序。"""
        if exclude_category_id is None and exclude_uid is None:
            return candidates
        return _FilteredNoiseCategoryView(
            candidates,
            exclude_category_id=exclude_category_id,
            exclude_uid=exclude_uid,
        )

    def _noise_category_candidates_for_tier(
        self,
        sr: int,
        *,
        exclude_category_id: int | None = None,
        exclude_uid: str | None = None,
    ) -> tuple[str | None, Mapping[int, tuple[str, ...]]]:
        """返回首个过滤后非空的 SR tier，保持原 exact/higher/all 语义。"""
        effective_exclude_uid = exclude_uid
        if (
            exclude_uid is not None
            and self._noise_category_id_by_uid.get(exclude_uid) == exclude_category_id
        ):
            effective_exclude_uid = None
        for tier, sample_rates in self._noise_category_tier_specs(sr):
            candidates = self._filter_noise_category_view(
                self._noise_category_tier_view(tier, sample_rates),
                exclude_category_id=exclude_category_id,
                exclude_uid=effective_exclude_uid,
            )
            if candidates:
                return tier, candidates
        return None, {}

    def _select_balanced_additive_noise_uid(
        self,
        sr: int,
        rng: np.random.Generator,
        exclude_category_id: int | None = None,
        exclude_uid: str | None = None,
    ) -> str | None:
        """按配置模式在首个可用 SR tier 选择 ordinary additive UID。"""
        if (
            self._noise_sampling_config is not None
            and self._noise_sampling_config.balance_mode == "temperature"
        ):
            return self._select_temperature_additive_noise_uid(
                sr, rng, exclude_category_id=exclude_category_id, exclude_uid=exclude_uid
            )
        return self._select_category_balanced_additive_noise_uid(
            sr, rng, exclude_category_id=exclude_category_id, exclude_uid=exclude_uid
        )

    def _ordered_noise_categories(
        self,
        sr: int,
        tier: str,
        candidates: Mapping[int, tuple[str, ...]],
        *,
        cacheable: bool,
    ) -> tuple[int, ...] | _CategoryExcludedView:
        """复用 canonical category order；过滤路径返回零复制 rank-select view。"""
        key = self._noise_category_view_key(sr, tier)
        ordered = self._noise_category_order_cache.get(key)
        if ordered is None:
            base = (
                candidates.base
                if isinstance(candidates, _FilteredNoiseCategoryView)
                else candidates
            )
            ordered = tuple(base)
            self._noise_category_order_cache[key] = ordered
        if cacheable:
            return ordered
        return _CategoryExcludedView(ordered, candidates)

    def _select_category_balanced_additive_noise_uid(
        self,
        sr: int,
        rng: np.random.Generator,
        exclude_category_id: int | None = None,
        exclude_uid: str | None = None,
    ) -> str | None:
        """先对 category 等权，再在该 category 聚合 UID 内均匀选择。"""
        tier, candidates = self._noise_category_candidates_for_tier(
            sr,
            exclude_category_id=exclude_category_id,
            exclude_uid=exclude_uid,
        )
        if not candidates or tier is None:
            return None
        categories = self._ordered_noise_categories(
            sr,
            tier,
            candidates,
            cacheable=exclude_category_id is None and exclude_uid is None,
        )
        category_id = rng_choice(rng, categories)
        return rng_choice(rng, candidates[category_id])

    def _temperature_candidates_and_weights(
        self,
        sr: int,
        *,
        exclude_category_id: int | None = None,
        exclude_uid: str | None = None,
    ) -> tuple[str | None, dict[int, tuple[str, ...]], dict[int, float]]:
        """返回 runtime 实际使用的 tier-local candidates 与最终 category 权重。"""
        tier, candidates = self._noise_category_candidates_for_tier(
            sr,
            exclude_category_id=exclude_category_id,
            exclude_uid=exclude_uid,
        )
        if tier is None:
            return None, candidates, {}
        cacheable = exclude_category_id is None and exclude_uid is None
        key = self._noise_category_view_key(sr, tier)
        if cacheable:
            cached = self._noise_temperature_base_weights_cache.get(key)
            if cached is not None:
                return tier, candidates, cached
        alpha = self._noise_sampling_config.alpha
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or not 0 < alpha <= 1
        ):
            raise ValueError("noise sampling alpha must be finite and in (0, 1]")
        weights = {
            category_id: (len(uids) ** alpha)
            * self._noise_category_weight_by_id[category_id]
            for category_id, uids in candidates.items()
        }
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
            raise ValueError("noise sampling weights must be finite and > 0")
        if not math.isfinite(math.fsum(weights.values())):
            raise ValueError("noise sampling total weight must be finite")
        if cacheable:
            self._noise_temperature_base_weights_cache[key] = weights
        return tier, candidates, weights

    def _select_temperature_additive_noise_uid(
        self,
        sr: int,
        rng: np.random.Generator,
        exclude_category_id: int | None = None,
        exclude_uid: str | None = None,
    ) -> str | None:
        """按 tier-local ``n_tier^alpha × difficulty`` 选择 category 和 UID。"""
        tier, candidates, weight_by_category = self._temperature_candidates_and_weights(
            sr,
            exclude_category_id=exclude_category_id,
            exclude_uid=exclude_uid,
        )
        if not candidates or tier is None:
            return None
        categories = self._ordered_noise_categories(
            sr,
            tier,
            candidates,
            cacheable=exclude_category_id is None and exclude_uid is None,
        )
        category_id = rng_choices_weighted(
            rng,
            categories,
            [weight_by_category[item] for item in categories],
            k=1,
        )[0]
        return rng_choice(rng, candidates[category_id])

    def _noise_natural_tier_specs(
        self, sr: int
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """按 manifest source 顺序返回 natural exact→higher→all tier。"""
        target_sr = int(sr)
        sample_rates = tuple(self._noise_natural_uids_by_fs)
        return (
            ("exact", (target_sr,) if target_sr in self._noise_natural_uids_by_fs else ()),
            ("higher", tuple(fs for fs in sample_rates if fs > target_sr)),
            ("all", sample_rates),
        )

    def _noise_natural_tier_view(
        self,
        tier: str,
        sample_rates: tuple[int, ...],
        *,
        exclude_category_id: int,
        exclude_uid: str | None = None,
    ) -> tuple[str, ...] | _NaturalTierExcludedView:
        """返回 manifest-order 异类候选；热路径按 category positions rank-select。"""
        # 正常双噪声路径中 exclude_uid 属于已排除 category，不会额外改变候选；
        # key 归一化到 category 级，避免同类不同首段 UID 复制相同 tuple。
        effective_exclude_uid = exclude_uid
        if (
            exclude_uid is not None
            and self._noise_category_id_by_uid.get(exclude_uid) == exclude_category_id
        ):
            effective_exclude_uid = None
        key = (tier, sample_rates, exclude_category_id, effective_exclude_uid)
        cached = self._noise_selector_lru_get(
            "_noise_natural_tier_uids_cache", key
        )
        if cached is not None:
            return cached
        views = tuple(
            _NaturalCategoryExcludedView(
                self._noise_natural_uids_by_fs[fs],
                self._noise_natural_positions_by_fs[fs],
                exclude_category_id,
                self._noise_natural_uid_position_by_fs[fs].get(effective_exclude_uid),
            )
            for fs in sample_rates
        )
        view = _NaturalTierExcludedView(views)
        if len(view) > _NOISE_SELECTOR_CACHE_MAX_UID_REFS:
            return view
        candidates = tuple(view)
        self._noise_selector_lru_put(
            "_noise_natural_tier_uids_cache",
            "_noise_natural_tier_uids_cache_uid_refs",
            key,
            candidates,
            len(candidates),
        )
        return candidates

    def _select_natural_different_category_uid(
        self,
        sr: int,
        rng: np.random.Generator,
        *,
        exclude_category_id: int,
        exclude_uid: str,
    ) -> str | None:
        """classified natural 第二段：按 manifest 原序在异类 UID 中均匀选择。"""
        for tier, sample_rates in self._noise_natural_tier_specs(sr):
            uids = self._noise_natural_tier_view(
                tier,
                sample_rates,
                exclude_category_id=exclude_category_id,
                exclude_uid=exclude_uid,
            )
            if uids:
                return rng_choice(rng, uids)
        return None

    def _ordinary_noise_categories_complete(self) -> bool:
        """O(1) 返回 ordinary pool 的 category map 完整性。"""
        return bool(getattr(self, "_noise_categories_complete", False))

    def get_speech_epoch_plan(
        self,
        *,
        requested: int,
        world_size: int,
        batch_size: int,
        drop_last: bool,
        seed: int,
        epoch: int,
    ) -> SpeechEpochPlan:
        """Build a deterministic selected speech plan for one balanced epoch.

        quota plan 固定（cache key = requested/world_size/batch_size/drop_last/
        effective controller config）；每个 (seed, epoch) 在 leaf 内以
        ``_leaf_seed`` 派生的随机源做稳定无放回 UID 选择，全局按 external
        index 排序（与既有无 speaker fast path 一致）。
        """
        if self.balanced_speech_sampling_enabled is not True:
            raise RuntimeError("balanced speech sampling is not enabled")
        plan_key = (
            requested,
            world_size,
            batch_size,
            drop_last,
            self._speech_sampling_config,
        )
        fixed_plan_key = getattr(self, "_fixed_speech_plan_key", None)
        if fixed_plan_key is None:
            fixed_speech_plan = make_speech_plan(
                [leaf.planner_leaf() for leaf in self.speech_sampling_leaves.values()],
                requested=requested,
                world_size=world_size,
                batch_size=batch_size,
                drop_last=drop_last,
                sampling_config=self._speech_sampling_config,
            )
            self._fixed_speech_plan_key = plan_key
            self._fixed_speech_plan = fixed_speech_plan
        elif self._fixed_speech_plan_key != plan_key:
            raise RuntimeError(
                "balanced speech sampling parameters changed after the fixed quota plan was created"
            )
        speech_plan = self._fixed_speech_plan
        if speech_plan is None:
            raise RuntimeError("balanced speech sampling fixed quota plan is unavailable")
        selected: list[tuple[int, str, int, tuple]] = []
        for leaf_key, quota in speech_plan.leaf_quotas.items():
            if quota <= 0:
                continue
            leaf = self.speech_sampling_leaves[leaf_key]
            order = list(range(len(leaf.external_indices)))
            random.Random(_leaf_seed(seed, epoch, leaf_key)).shuffle(order)
            for position in order[:quota]:
                selected.append(
                    (leaf.external_indices[position], leaf.uids[position], leaf.batching_sr, leaf_key)
                )
        selected.sort(key=lambda item: item[0])
        indices, uids, sample_rates, leaf_keys = (
            zip(*selected) if selected else ((), (), (), ())
        )
        cell_keys = tuple((key[0], key[1], key[3]) for key in leaf_keys)
        if Counter(sample_rates) != Counter(speech_plan.sr_quotas):
            raise RuntimeError("speech epoch plan sample-rate quotas do not match selections")
        if Counter(leaf_keys) != Counter(speech_plan.leaf_quotas):
            raise RuntimeError("speech epoch plan leaf quotas do not match selections")
        return SpeechEpochPlan(
            tuple(indices), tuple(uids), tuple(sample_rates), tuple(cell_keys), speech_plan
        )

    # ------------------------------------------------------------------
    # Sample rate helpers
    # ------------------------------------------------------------------
    def _processing_sr(self, native_fs: int) -> int:
        """Return the sample rate at which mixing runs (target_sr if set, else native)."""
        return int(self.target_sr) if self.target_sr is not None else int(native_fs)

    def _apply_target_sr(
        self, speech: np.ndarray, noisy: np.ndarray, fs_out: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Resample both signals to target_sr if it is set and differs from fs_out."""
        if self.target_sr is not None and fs_out != self.target_sr:
            speech = _resample_audio(speech, fs_out, self.target_sr)
            noisy = _resample_audio(noisy, fs_out, self.target_sr)
            fs_out = int(self.target_sr)
        return speech, noisy, fs_out

    # ------------------------------------------------------------------
    # Epoch-based seed control and cache management
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int, base_seed: int = 0) -> None:
        """Bind epoch to a deterministic seed and clear per-epoch caches.

        Clears per-epoch instance reuse caches. Decoded-audio LRU entries remain
        retained across epochs and are independently constrained by their per-worker
        byte budget, preserving cross-epoch cache hits without weakening memory limits.
        Per-sample 随机性由显式 ``Generator`` 承载，这里不修改进程级 ``random`` /
        ``numpy`` 全局状态。
        """
        self._epoch = int(epoch)
        self._base_seed = int(base_seed)

        # Instance-level reuse caches (one-shot buffers per epoch)
        self._rir_reuse_cache = None
        self._noise_only_reuse_cache = None
        self._noise_single_reuse_cache = None
        self._noise_dual_reuse_cache = None
        self._wind_reuse_cache = None
        self._last_wind_source_identity = None

        # Subclass-specific extra caches (e.g. AEC GCC-PHAT alignment)
        self._clear_extra_epoch_caches()

        # Lazily-built per-sample metadata lists (recomputed on first access)
        for _attr in ("_cached_srs", "_cached_source_length"):
            try:
                delattr(self, _attr)
            except AttributeError:
                pass

        # High-pass filter taps cache (bounded but unnecessary across epochs)
        self._hp_taps = None

        # Signal persistent workers (via shared memory) that epoch/base_seed changed.
        self._worker_base_seed[0] = int(base_seed)
        self._worker_epoch[0] = int(epoch)

    def _clear_extra_epoch_caches(self) -> None:
        """Hook for subclasses to clear extra caches in set_epoch()."""

    def _worker_maybe_clear_caches(self) -> None:
        """共享 seed 或 epoch 变化时清理 worker 的复用缓存；音频 LRU 不受影响。"""
        current = (int(self._worker_base_seed[0]), int(self._worker_epoch[0]))
        if current == self._worker_last_seen_epoch:
            return
        self._worker_last_seen_epoch = current

        # Instance-level reuse buffers (drop stale RIR / noise / wind arrays)
        self._rir_reuse_cache = None
        self._noise_only_reuse_cache = None
        self._noise_single_reuse_cache = None
        self._noise_dual_reuse_cache = None
        self._wind_reuse_cache = None
        self._last_wind_source_identity = None

        # Subclass-specific extra caches
        self._clear_extra_worker_caches()

        # High-pass filter taps (tiny but stale across epochs)
        self._hp_taps = None

        # Decoded-audio LRU 不按 epoch 清空；其 byte/entry 上限始终由缓存自身执行。

    def _clear_extra_worker_caches(self) -> None:
        """Hook for subclasses to clear extra caches in _worker_maybe_clear_caches()."""

    def _get_rng(self, index: int | None = None) -> np.random.Generator:
        """使用共享场景子流；离线/在线均不依赖 worker 或 iterator 的种子。"""
        from core.data.speech_scene import scene_rng

        return scene_rng(self, 0 if index is None else int(index), 0)

    # ------------------------------------------------------------------
    # Noise blending
    # ------------------------------------------------------------------
    @staticmethod
    def _extend_noise_to_length(noise: np.ndarray, target_len: int) -> np.ndarray:
        """Tile a 2-D noise segment to at least target_len samples."""
        if noise.shape[-1] >= target_len:
            return noise
        t2 = noise.shape[-1]
        if t2 <= 0:
            return np.zeros_like(noise)
        reps = (target_len + t2 - 1) // t2
        return np.tile(noise, (1, reps))[:, :target_len]

    # ------------------------------------------------------------------
    # __getitem__ framework
    # ------------------------------------------------------------------
    def __getitem__(self, index: int):
        return self.get_sample_with_trace(index).sample

    def _getitem_simulation_verify(
        self, index: int, force_branch: str | None = None
    ):
        """旧验证入口 Adapter；sample Interface 保持不变。"""
        return self.get_sample_with_trace(index, force_branch=force_branch).sample

    def _simulation_trace_task(self) -> str:
        task = getattr(self, "task", None)
        if task is not None:
            return str(task)
        module_parts = type(self).__module__.split(".")
        return module_parts[1] if len(module_parts) > 1 and module_parts[0] == "tasks" else module_parts[0]

    def _legacy_last_snapshot(self) -> dict[str, tuple[Any, Any]]:
        """捕获上一个完成样本的状态；恢复语义由 sample_state 统一实现。"""
        return capture_last_completed_state(self)

    def _restore_legacy_last_snapshot(
        self, snapshot: Mapping[str, tuple[Any, Any]]
    ) -> None:
        restore_last_completed_state(self, snapshot)

    def _managed_legacy_last_defaults(self) -> dict[str, Any]:
        """返回本任务 last-completed Adapter 的完整中性状态。"""
        return {
            "_last_run_branch": None,
            "_last_layer_e_result": None,
            "_last_layer_f_applied_ops": [],
            "_last_layer_f_clipping_facts": None,
        }

    def _legacy_stems_from_trace(self, trace: SimulationTrace) -> dict | None:
        """投影 task-neutral legacy stems；子类只可适配投影 schema。"""
        return trace.legacy_stems("scene")

    def _publish_completed_trace(self, trace: SimulationTrace) -> None:
        """一次性发布已 seal 的 trace 及其 legacy last-completed 投影。"""
        if not trace.sealed:
            raise SampleBuildContractError("only the base dataset may seal and publish traces")
        legacy = self._managed_legacy_last_defaults()
        legacy.update(trace.legacy_snapshot())
        legacy["_last_simulation_stems"] = (
            self._legacy_stems_from_trace(trace)
            if trace.capture_stems and not trace.placeholder
            else None
        )
        legacy["_last_simulation_trace"] = trace
        for name, value in legacy.items():
            setattr(self, name, value)

    @staticmethod
    def _validate_build_result(
        result: Any, trace: SimulationTrace
    ) -> SampleBuildResult:
        """验证任务 hook 返回值及 supplied trace 所有权合同。"""
        if not isinstance(result, SampleBuildResult):
            raise SampleBuildContractError(
                "_build_sample must return SampleBuildResult; "
                f"got {type(result).__name__}"
            )
        if result.trace is not trace:
            raise SampleBuildContractError(
                "_build_sample must return the supplied trace instance"
            )
        if trace.sealed:
            raise SampleBuildContractError(
                "_build_sample must not seal or publish the supplied trace"
            )
        sealed_roles = tuple(
            role for role, source in trace.sources.items() if source.sealed
        )
        if sealed_roles:
            raise SampleBuildContractError(
                "_build_sample must not seal source traces before owner commit; "
                f"sealed roles: {sealed_roles!r}"
            )
        return result

    def get_sample_with_trace(
        self,
        index: int,
        *,
        force_branch: str | None = None,
        capture_stems: bool | None = None,
    ) -> TracedSample:
        """原子返回 sample 与唯一 trace；strict 失败不发布半成品。"""
        self._worker_maybe_clear_caches()
        fs, idx = self._get_from_index(index)
        uid = self.speech_uids[fs][idx]
        work_sr = self._processing_sr(fs)
        slen = self.source_length[uid]
        capture = self.export_simulation_stems if capture_stems is None else bool(capture_stems)
        trace = SimulationTrace(
            index=index,
            uid=uid,
            task=self._simulation_trace_task(),
            capture_stems=capture,
        )
        previous_last = self._legacy_last_snapshot()
        configured_capture = self.export_simulation_stems
        self._last_simulation_stems = None
        self._last_run_branch = None
        self.export_simulation_stems = capture

        try:
            try:
                completed = self._validate_build_result(
                    self._build_sample(index, uid, slen, work_sr, force_branch, trace=trace),
                    trace,
                )
            except NotImplementedError:
                raise  # Missing subclass override — fail loudly, never silence-fill.
            except SampleBuildContractError:
                raise
            except Exception as exc:
                nonrecoverable = self._is_nonrecoverable_sample_error(exc)
                from core.data.errors import is_recoverable_resource_error

                recoverable = is_recoverable_resource_error(exc)
                if force_branch is not None or nonrecoverable or self.strict_load or not recoverable:
                    if self.strict_load and not nonrecoverable:
                        raise RuntimeError(
                            f"strict_load=True: dynamic mix failed index={index} uid={uid}: {exc}"
                        ) from exc
                    raise
                logger.warning(
                    "Dynamic mix failed (index=%d, uid=%s), returning silence placeholder.",
                    index,
                    uid,
                    exc_info=True,
                )
                self._restore_legacy_last_snapshot(previous_last)
                trace = SimulationTrace(
                    index=index,
                    uid=uid,
                    task=self._simulation_trace_task(),
                    capture_stems=capture,
                )
                trace.mark_placeholder(reason=f"{type(exc).__name__}: {exc}")
                trace.replace_metadata("branch", None)
                scene_trace = trace.source("scene")
                scene_trace.replace_metadata("branch", None)
                completed = SampleBuildResult(
                    sample=self._placeholder_sample(work_sr, index, uid), trace=trace
                )

            trace.seal()
            self._publish_completed_trace(trace)
            return TracedSample(sample=completed.sample, trace=trace)
        except Exception:
            self._restore_legacy_last_snapshot(previous_last)
            raise
        finally:
            self.export_simulation_stems = configured_capture

    def _getitem_impl(
        self, index: int, force_branch: str | None = None
    ):
        """旧内部入口 Adapter；新消费者使用 ``get_sample_with_trace``。"""
        return self.get_sample_with_trace(index, force_branch=force_branch).sample

    def _is_nonrecoverable_sample_error(self, exc: Exception) -> bool:
        """子类可标记不得在 lenient 模式降级为 placeholder 的合同错误。"""
        from core.data.simulation.rir_service import OnlineRIRError
        from core.data.errors import NonRecoverableSimulationError

        return isinstance(exc, (OnlineRIRError, NonRecoverableSimulationError))

    def _build_sample(
        self,
        index: int,
        uid: str,
        slen: int,
        work_sr: int,
        force_branch: str | None,
        *,
        trace: SimulationTrace,
    ) -> SampleBuildResult:
        """使用运行仿真并返回绑定 supplied trace 的 ``SampleBuildResult``。"""
        raise NotImplementedError("subclass must implement _build_sample")

    def _placeholder_sample(
        self, work_sr: int, index: int, uid: str
    ):
        """Subclass hook: return a silence placeholder when mixing fails."""
        raise NotImplementedError("subclass must implement _placeholder_sample")

    # ------------------------------------------------------------------
    # Layer B/C: acoustic mixing (RIR + additive noise + side-chain wind)
    # ------------------------------------------------------------------
    def _acoustic_plan_record(self) -> dict[str, Any] | None:
        """返回当前同步 mix 的内部 recorder；默认路径无额外 Interface。"""
        record = getattr(self, "_active_acoustic_plan_record", None)
        return record if isinstance(record, dict) else None

    def _record_noise_source(
        self,
        *,
        family: str,
        uid: str | None,
        path: str | None,
        crop: Any = None,
        applied: bool = True,
        cache_hit: bool = False,
        application_role: NoiseApplicationRole = NoiseApplicationRole.LAYER_C_ADDITIVE,
        include_in_scene_plan: bool = True,
    ) -> None:
        """记录 shared source policy，并按需投影到既有 Layer C scene plan。"""
        record = self._acoustic_plan_record()
        if record is None:
            return
        metadata_by_uid = getattr(self, "_noise_manifest_metadata", {})
        metadata = metadata_by_uid.get(uid, {}) if uid else {}
        policy_family = NoiseSourceFamily(str(family))
        policy = NoiseSourcePolicy(
            family=policy_family,
            category=NoiseCategory(
                category_id=metadata.get("category_id"),
                category_name=(category_balance_name(metadata) if metadata else None),
            ),
            application_role=application_role,
            cache_provenance=(
                NoiseCacheProvenance.DATASET_REUSE_CACHE
                if cache_hit
                else NoiseCacheProvenance.SHARED_AUDIO_CACHE_UNKNOWN
            ),
            uid=uid,
            path=path,
        )
        record.setdefault("noise_source_policies", []).append(policy)
        # AEC post-E/F 不属于任何 Layer C scene；TSE enrollment 自有 scene plan。
        if not include_in_scene_plan:
            return
        record.setdefault("noise_sources", []).append(
            NoiseSourcePlan(
                family=str(family),
                uid=uid,
                path=path,
                category_id=policy.category.category_id,
                category_name=policy.category.category_name,
                crop=crop,
                applied=bool(applied),
                cache_hit=bool(cache_hit),
            )
        )

    @staticmethod
    def _reuse_noise_source_sidecar(
        record: dict[str, Any] | None, start: int
    ) -> tuple[NoiseSourcePlan, ...]:
        """截取本次 read 新增的 source identity，供 reuse tuple 复用。"""
        if record is None:
            return ()
        sources = record.get("noise_sources")
        if not isinstance(sources, list):
            return ()
        return tuple(source for source in sources[start:] if isinstance(source, NoiseSourcePlan))

    def _record_reused_noise_sources(
        self, sidecar: tuple[NoiseSourcePlan, ...]
    ) -> None:
        """以真实 identity 重放 reuse cache source；不为 legacy cache 伪造来源。"""
        record = self._acoustic_plan_record()
        if record is None:
            return
        for source in sidecar:
            family = NoiseSourceFamily(source.family)
            category = NoiseCategory(source.category_id, source.category_name)
            record.setdefault("noise_source_policies", []).append(
                NoiseSourcePolicy(
                    family=family,
                    category=category,
                    application_role=NoiseApplicationRole.LAYER_C_ADDITIVE,
                    cache_provenance=NoiseCacheProvenance.DATASET_REUSE_CACHE,
                    uid=source.uid,
                    path=source.path,
                )
            )
            record.setdefault("noise_sources", []).append(
                replace(source, cache_hit=True)
            )

    def _mix_acoustic_rir(
        self,
        speech_2d: np.ndarray,
        sr: int,
        rng: np.random.Generator,
        use_reverb: bool,
        *,
        forced_rir_plan: RIRPlan | None = None,
        forced_rir_asset: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """施加 RIR，并返回 noisy、SNR reference 与监督 target。

        ``forced_rir_*`` 仅供 TSE C-0 asset replay 使用：调用者已完成
        计划/asset 的 role、采样率和长度校验。本入口不选择资源、不读盘、不访问
        reuse cache，且不消费 RNG。
        """
        cfg = self.sim_cfg
        if forced_rir_plan is None and getattr(cfg, "rir_backend", "offline") == "online":
            override = getattr(self, "_online_rir_source", None)
            if override is not None:
                if not isinstance(override, Mapping):
                    raise TypeError("online RIR source override must be a mapping")
                use_reverb = bool(override["enabled"])
                if bool(override.get("moving", False)) and use_reverb:
                    if cfg.spatial_enabled:
                        raise ValueError("mono moving RIR override cannot be used for a spatial scene")
                    return self._mix_online_moving_mono(
                        speech_2d,
                        sr,
                        rng,
                        None,
                        rir_seq=override["asset"],
                        facts=override["facts"],
                        segment_sec=override["segment_sec"],
                        normalization_peak=override["normalization_peak"],
                        source_index=int(override["index"]),
                    )
                asset = override["asset"]
                facts = override["facts"]
                index = int(override["index"])
            elif use_reverb:
                provider = getattr(self, "_online_rir_provider", None)
                if provider is None:
                    raise RuntimeError("online RIR provider was not initialized by the Dataset owner")
                if (
                    not cfg.spatial_enabled
                    and cfg.prob_moving_rir > 0.0
                    and rng.random() < cfg.prob_moving_rir
                ):
                    # 在线 mono moving：仅在抽中 reverb 后按 prob_moving_rir 生效。
                    return self._mix_online_moving_mono(speech_2d, sr, rng, provider)
                group, facts = provider.sample_scene(sr, 1, rng, normalize=cfg.rir_normalize)
                asset, index = group[0], 0
            if use_reverb:
                forced_rir_asset = np.asarray(asset, dtype=np.float32).reshape(1, -1)
                forced_rir_plan = RIRPlan(enabled=True, uid=f"{facts['group_uid']}/source/{index}",
                    path=None, moving=False, reuse_enabled=False, cache_hit=facts.get("cache_hit", False),
                    segment_sec=None, normalize=False,
                    early_rir_sec=cfg.early_rir_sec, target_type=cfg.target_type, backend="online")
                record = self._acoustic_plan_record()
                if record is not None: record["online_rir_scene"] = facts
        asset_forced = forced_rir_plan is not None or forced_rir_asset is not None
        if asset_forced and forced_rir_plan is None:
            raise C0AssetReplayError("C-0 asset replay requires an RIR plan")
        if asset_forced and not forced_rir_plan.enabled:
            if forced_rir_asset is not None:
                raise C0AssetReplayError(
                    "C-0 asset replay disabled RIR plan must not carry an asset"
                )
            record = self._acoustic_plan_record()
            if record is not None:
                record["rir_plan"] = forced_rir_plan
            return speech_2d.copy(), speech_2d.copy(), speech_2d.copy()
        if asset_forced:
            if forced_rir_asset is None:
                raise C0AssetReplayError(
                    "C-0 asset replay enabled RIR plan requires an asset"
                )
            rir_sample = np.asarray(forced_rir_asset, dtype=np.float32)
            if (
                rir_sample.ndim not in (1, 2)
                or rir_sample.shape[-1] <= 0
                or not np.isfinite(rir_sample).all()
            ):
                raise C0AssetReplayError(
                    "C-0 asset replay RIR asset must be finite and non-empty"
                )
            rir_sample = np.ascontiguousarray(rir_sample)
            use_reverb = True
        elif not use_reverb:
            record = self._acoustic_plan_record()
            if record is not None:
                record["rir_plan"] = RIRPlan.disabled(
                    early_rir_sec=float(cfg.early_rir_sec),
                    target_type=str(getattr(cfg, "target_type", "early")),
                )
            return speech_2d.copy(), speech_2d.copy(), speech_2d.copy()

        reuse_rir = getattr(cfg, "reuse_rir", False)
        rir_uid: str | None = None
        resolved_segment_sec: float | None = None
        reuse_cache_hit = False
        if asset_forced:
            rir_normalize = bool(forced_rir_plan.normalize)
            target_type = str(forced_rir_plan.target_type)
            early_rir_sec = float(forced_rir_plan.early_rir_sec)
            resolved_segment_sec = forced_rir_plan.segment_sec
        else:
            reuse_cache_hit = (
                reuse_rir
                and self._rir_reuse_cache is not None
                and self._rir_reuse_cache[0] == sr
            )
            if reuse_cache_hit:
                rir_cache = self._rir_reuse_cache
                rir_sample = rir_cache[1].copy()
                if len(rir_cache) >= 3:
                    resolved_segment_sec = rir_cache[2]
                if len(rir_cache) >= 4:
                    rir_uid = rir_cache[3]
            else:
                rir_uid = self._select_sample(sr, self.rirs, rng=rng)
                if rir_uid is None:
                    raise RuntimeError(
                        "prob_reverberation=True but no RIR source available. "
                        "Add RIR entries or set prob_reverberation to 0."
                    )
                rir_sample = _read_cached_2d(
                    self.rirs_flt[rir_uid], sr, strict=self.strict_load, rng=rng
                )
            rir_normalize = bool(getattr(cfg, "rir_normalize", True))
            target_type = str(getattr(cfg, "target_type", "early"))
            early_rir_sec = float(cfg.early_rir_sec)

        record = self._acoustic_plan_record()
        if record is not None:
            record["rir_asset"] = rir_sample
        moving = rir_sample.ndim == 2 and rir_sample.shape[0] > 1
        if asset_forced and moving != bool(forced_rir_plan.moving):
            raise C0AssetReplayError(
                "C-0 asset replay RIR moving shape does not match plan"
            )
        if not asset_forced and reuse_rir and not reuse_cache_hit and not moving:
            self._rir_reuse_cache = (
                sr,
                np.ascontiguousarray(rir_sample, dtype=np.float32).copy(),
            )
        if moving:
            seg_sec = (
                resolved_segment_sec
                if resolved_segment_sec is not None
                else getattr(cfg, "segment_sec", 0.5)
            )
            if not asset_forced and not reuse_cache_hit and rir_uid is not None:
                meta_sec = _read_rir_meta_segment_sec(self.rirs_flt[rir_uid])
                if meta_sec is not None and abs(meta_sec - seg_sec) > 1e-6:
                    logger.warning(
                        "Moving RIR segment_sec mismatch: pkl=%s cfg=%s (use pkl value)",
                        meta_sec,
                        seg_sec,
                    )
                    seg_sec = meta_sec
            resolved_segment_sec = float(seg_sec)
            if not asset_forced and reuse_rir and not reuse_cache_hit:
                self._rir_reuse_cache = (
                    sr,
                    np.ascontiguousarray(rir_sample, dtype=np.float32).copy(),
                    resolved_segment_sec,
                    rir_uid,
                )
            full_sequence_peak = (
                float(np.max(np.abs(rir_sample))) if rir_normalize else None
            )
            noisy_speech = add_moving_reverberation(
                speech_2d, rir_sample, seg_sec, sr,
                normalize_rir=rir_normalize,
                normalization_peak=full_sequence_peak,
            )
        elif rir_sample.ndim == 2 and rir_sample.shape[0] == 1:
            noisy_speech, mix_reference = add_paired_reverberation(
                speech_2d,
                rir_sample,
                early_rir_sec=early_rir_sec,
                fs=sr,
                normalize_rir=rir_normalize,
            )
        else:
            noisy_speech = add_reverberation(
                speech_2d, rir_sample, normalize_rir=rir_normalize
            )
            early_rir = estimate_early_rir(
                rir_sample, early_rir_sec=early_rir_sec, fs=sr
            )
            mix_reference = add_reverberation(
                speech_2d, early_rir, normalize_rir=rir_normalize
            )
        if moving:
            early_rir = estimate_early_rir(
                rir_sample, early_rir_sec=early_rir_sec, fs=sr
            )
            mix_reference = add_moving_reverberation(
                speech_2d, early_rir, seg_sec, sr,
                normalize_rir=rir_normalize,
                normalization_peak=full_sequence_peak,
            )
        clean_target = speech_2d.copy() if target_type == "dry" else mix_reference
        if record is not None:
            record["rir_plan"] = (
                forced_rir_plan
                if asset_forced
                else RIRPlan(
                    enabled=True,
                    uid=rir_uid or "__reuse_cache__",
                    path=(self.rirs_flt.get(rir_uid) if rir_uid is not None else None),
                    moving=bool(moving),
                    reuse_enabled=bool(reuse_rir),
                    cache_hit=bool(reuse_cache_hit),
                    segment_sec=(float(resolved_segment_sec) if moving else None),
                    normalize=bool(rir_normalize),
                    early_rir_sec=early_rir_sec,
                    target_type=target_type,
                )
            )
        return noisy_speech, mix_reference, clean_target

    def _mix_online_moving_mono(
        self,
        speech_2d: np.ndarray,
        sr: int,
        rng: np.random.Generator,
        provider,
        *,
        rir_seq: np.ndarray | None = None,
        facts: Mapping[str, Any] | None = None,
        segment_sec: float | None = None,
        normalization_peak: float | None = None,
        source_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在线 mono moving RIR：逐帧 bank 按 timebase hop 分段卷积。

        early 轨由 full bank 逐段经 ``estimate_early_rir`` 派生，与离线 moving
        PKL 路径保持同一 early 合同；scene renderer 传入共享 room/timebase 下的
        per-source 轨迹时，full/early 共用全 scene 峰值归一化。
        返回 ``(noisy, mix_reference, clean_target)``。
        """
        cfg = self.sim_cfg
        track_set = None
        if rir_seq is None:
            if provider is None:
                raise RuntimeError("online moving RIR requires a provider or a pre-sampled source track")
            num_samples = int(speech_2d.shape[1])
            track_set = provider.sample_moving_spatial_tracks(
                sr, 1, rng, num_samples=num_samples, normalize=cfg.rir_normalize,
                track_kinds=("full",),
            )
            bank = np.asarray(track_set.full, dtype=np.float32)
            if bank.ndim == 3 and bank.shape[1] == 1:  # [P, 1, L] mono canonical bank
                rir_seq = bank[:, 0, :]
            elif bank.ndim == 2:  # [P, L] 单通道 moving bank
                rir_seq = bank
            else:
                raise ValueError(
                    f"online mono moving RIR bank must be [P,K,L] or [P,L], got {bank.shape}"
                )
            segment_sec = (
                float(track_set.plan["trajectory_timebase"]["hop_samples"]) / float(sr)
            )
            facts = track_set.facts
        else:
            rir_seq = np.asarray(rir_seq, dtype=np.float32)
            if rir_seq.ndim != 2 or rir_seq.shape[0] < 2 or rir_seq.shape[1] < 1:
                raise ValueError(
                    f"online mono moving source track must have shape [P,L], got {rir_seq.shape}"
                )
            if not np.isfinite(rir_seq).all():
                raise ValueError("online mono moving source track must contain finite values")
            if segment_sec is None or not np.isfinite(float(segment_sec)) or float(segment_sec) <= 0:
                raise ValueError("pre-sampled online moving track requires a positive segment_sec")
            if not isinstance(facts, Mapping):
                raise TypeError("pre-sampled online moving track requires scene facts")
            facts = dict(facts)
        seg_sec = float(segment_sec)
        rir_normalize = bool(getattr(cfg, "rir_normalize", True))
        full_sequence_peak = (
            float(normalization_peak)
            if rir_normalize and normalization_peak is not None
            else (float(np.max(np.abs(rir_seq))) if rir_normalize else None)
        )
        noisy_speech = add_moving_reverberation(
            speech_2d, rir_seq, seg_sec, sr,
            normalize_rir=rir_normalize,
            normalization_peak=full_sequence_peak,
        )
        early_rir = estimate_early_rir(
            rir_seq, early_rir_sec=float(cfg.early_rir_sec), fs=sr
        )
        mix_reference = add_moving_reverberation(
            speech_2d, early_rir, seg_sec, sr,
            normalize_rir=rir_normalize,
            normalization_peak=full_sequence_peak,
        )
        clean_target = speech_2d.copy() if cfg.target_type == "dry" else mix_reference
        record = self._acoustic_plan_record()
        if record is not None:
            facts = dict(facts or {})
            track_facts = facts.get("track_facts")
            if isinstance(track_facts, Mapping):
                # frame_facts 是逐帧 service 细节（P × 单帧 facts），只保留整轨
                # 身份，避免 trace 体积随轨迹帧数膨胀。
                facts["track_facts"] = {
                    name: {
                        key: value
                        for key, value in dict(entry).items()
                        if key != "frame_facts"
                    }
                    for name, entry in track_facts.items()
                }
            record["online_rir_scene"] = facts
            group_uid = facts.get("group_uid")
            if not isinstance(group_uid, str) or not group_uid:
                raise ValueError("online moving RIR facts must include group_uid")
            rir_uid = (
                f"{group_uid}/moving"
                if track_set is not None
                else f"{group_uid}/source/{source_index}/moving"
            )
            track_facts = facts.get("track_facts")
            track_full_facts = (
                track_facts.get("full", {}) if isinstance(track_facts, Mapping) else {}
            )
            record["rir_plan"] = RIRPlan(
                enabled=True,
                uid=rir_uid,
                path=None,
                moving=True,
                reuse_enabled=False,
                cache_hit=(
                    bool(track_full_facts.get("cache_hit", False))
                    if isinstance(track_full_facts, Mapping)
                    else False
                ),
                segment_sec=seg_sec,
                normalize=False,
                early_rir_sec=float(cfg.early_rir_sec),
                target_type=str(cfg.target_type),
                backend="online",
            )
        return noisy_speech, mix_reference, clean_target

    def _record_post_rir_calibration(
        self,
        full_rir: np.ndarray,
        early_reference: np.ndarray,
        gain: float,
    ) -> None:
        """子类 hook：观察 gain 后 C-0 stems；基类 runtime 不缓存波形。"""

    def _mix_acoustic_get_wind_sample(
        self,
        sr: int,
        max_dur_sr: int,
        rng: np.random.Generator,
        reuse_noise: bool,
    ) -> np.ndarray | None:
        """读取或复用 wind sample，不改变选择与缓存时机。"""
        self._last_wind_source_identity = None
        if (
            reuse_noise
            and self._wind_reuse_cache is not None
            and self._wind_reuse_cache[0] == sr
        ):
            cached = self._wind_reuse_cache
            sample = cached[1].copy()
            sidecar = cached[2] if len(cached) >= 3 else ()
            for source in sidecar:
                uid = getattr(source, "uid", None)
                if uid:
                    self._last_wind_source_identity = f"wind_scp:{uid}"
                    break
            self._record_reused_noise_sources(sidecar)
            record = self._acoustic_plan_record()
            if record is not None:
                record["noise_assets"].append(sample.copy())
            return sample
        uid = self._select_sample(sr, self.wind_noises, rng=rng)
        if uid is None:
            return None
        crop_record: dict[str, Any] = {}
        sample = self.resource_store.read_noise(
            self.wn_flt[uid],
            sr,
            max_dur_sr,
            spatial=False,
            rng=rng,
            crop_record=crop_record,
        )
        record = self._acoustic_plan_record()
        source_start = len(record.get("noise_sources", ())) if record is not None else 0
        self._record_noise_source(
            family="wind",
            uid=uid,
            path=self.wn_flt[uid],
            crop=crop_record.get("crop"),
        )
        self._last_wind_source_identity = f"wind_scp:{uid}"
        if record is not None:
            record["noise_assets"].append(sample.copy())
        if reuse_noise:
            self._wind_reuse_cache = (
                sr,
                np.ascontiguousarray(sample, dtype=np.float32).copy(),
                self._reuse_noise_source_sidecar(record, source_start),
            )
        return sample

    def _render_wind_sidechain_image(
        self,
        sample: np.ndarray,
        *,
        image_geometry,
        background_spec: BackgroundSpec,
        sr: int,
        rng: np.random.Generator,
        source_identity: str | None,
        facts_out: dict[str, Any] | None,
    ) -> np.ndarray:
        """把 sidechain 风噪渲染成与 observation 对齐的 spatial image。

        sidechain 仍由 ``wind_noise`` 执行非加性变换；本 helper 只负责把
        mono wind asset 通过同一 ``BackgroundSpec`` 投影到 ``[C,T]``，禁止
        在空间路径把 mono control 静默广播到阵列。
        """
        if image_geometry is None:
            value = np.asarray(sample, dtype=np.float32)
            if value.ndim == 1:
                value = value[None, :]
            if value.ndim != 2:
                raise ValueError("wind sidechain sample must have shape [T] or [C,T]")
            return value
        if not isinstance(background_spec, BackgroundSpec):
            raise TypeError("spatial wind sidechain requires a BackgroundSpec")
        value = np.asarray(sample, dtype=np.float32)
        if value.ndim == 1:
            mono = value
        elif value.ndim == 2 and value.shape[0] == 1:
            mono = value[0]
        elif value.ndim == 2:
            mono = None
        else:
            raise ValueError("wind sidechain sample must have shape [T] or [C,T]")

        projection_provenance: dict[str, Any] = {}
        if background_spec.policy == BackgroundPolicy.POINT_SOURCE_RIR:
            if mono is None:
                raise ValueError(
                    "point_source_rir wind sidechain requires a mono wind asset; "
                    "multichannel assets must not be rebroadcast"
                )
            provider = background_spec.point_source_provider
            validate_geometry = getattr(provider, "validate_geometry", None)
            if not callable(validate_geometry):
                raise TypeError("point_source_rir wind sidechain provider must validate geometry")
            validate_geometry(image_geometry)
            rendered = provider.render(mono, sample_rate=int(sr))
            if not isinstance(rendered, NoiseImage):
                raise TypeError("point source wind sidechain provider must return NoiseImage")
            image = rendered.waveform
            projection_provenance = dict(rendered.provenance)
        elif background_spec.policy == BackgroundPolicy.DIFFUSE_COHERENCE:
            rendered = adapt_noise_waveform(
                value,
                geometry=image_geometry,
                sample_rate=int(sr),
                rng=rng,
                coherence_policy=self.sim_cfg.spatial_diffuse_coherence_model,
                reference_channel=self.sim_cfg.reference_channel,
                source_identity=source_identity,
                provenance={"stage": "spatial_wind_sidechain_projection"},
            )
            image = rendered.waveform
            projection_provenance = dict(rendered.provenance)
        elif background_spec.policy == BackgroundPolicy.DIRECT_MULTICHANNEL:
            if mono is not None:
                raise ValueError(
                    "direct_multichannel wind sidechain requires explicit [C,T] wind asset; "
                    "mono broadcast is forbidden"
                )
            image = value
            projection_provenance = {
                "projection": "direct_multichannel",
                "input_shape": list(value.shape),
                "source_identity": source_identity,
            }
        else:
            raise ValueError(
                "spatial wind sidechain requires diffuse_coherence, point_source_rir, "
                "or explicit direct_multichannel BackgroundSpec"
            )

        image = np.ascontiguousarray(image, dtype=np.float32)
        geometry_channels = int(np.asarray(image_geometry).shape[0])
        if image.ndim != 2 or image.shape[0] != geometry_channels:
            raise ValueError(
                "spatial wind sidechain must produce explicit [C,T] image without mono broadcast: "
                f"got {image.shape}, expected channels={geometry_channels}"
            )
        if facts_out is not None:
            facts_out.update({
                "shape": list(image.shape),
                "source_identity": source_identity,
                "background_policy": background_spec.policy.value,
                "geometry_bound": bool(background_spec.geometry_bound),
                "coherence_policy": (
                    "point_source_rir"
                    if background_spec.policy == BackgroundPolicy.POINT_SOURCE_RIR
                    else self.sim_cfg.spatial_diffuse_coherence_model
                ),
                "projection": "point_source_rir" if background_spec.policy == BackgroundPolicy.POINT_SOURCE_RIR else (
                    "direct_multichannel" if background_spec.policy == BackgroundPolicy.DIRECT_MULTICHANNEL
                    else "diffuse_coherence"
                ),
                "projection_provenance": projection_provenance,
            })
        return image

    def _mix_acoustic_apply_wind(
        self,
        noisy_speech: np.ndarray,
        mix_reference: np.ndarray,
        additive_noise_2d: np.ndarray,
        sr: int,
        actual_len: int,
        max_dur_sr: int,
        wn_snr: float,
        rng: np.random.Generator,
        reuse_noise: bool,
        acoustic_audit: dict | None,
        image_geometry=None,
        background_spec: BackgroundSpec | None = None,
        wind_facts_out: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """施加 side-chain wind；空间路径先生成严格 ``[C,T]`` control image。"""
        record_reader = getattr(self, "_acoustic_plan_record", None)
        record = record_reader() if callable(record_reader) else None
        source_start = len(record.get("noise_sources", ())) if record is not None else 0
        self._last_wind_source_identity = None
        sample = self._mix_acoustic_get_wind_sample(
            sr, max_dur_sr, rng, reuse_noise
        )
        if sample is None:
            return noisy_speech, additive_noise_2d, False
        sample = _align_noise_2d_to_length(sample, noisy_speech.shape[1], rng)
        source_identity = None
        if record_reader is not None:
            record = record_reader()
            sources = tuple(record.get("noise_sources", ())) if record is not None else ()
            for source in sources[source_start:]:
                uid = getattr(source, "uid", None)
                if uid:
                    source_identity = f"wind_scp:{uid}"
                    break
        source_identity = source_identity or getattr(
            self, "_last_wind_source_identity", None
        )
        sample = self._render_wind_sidechain_image(
            sample,
            image_geometry=image_geometry,
            background_spec=(
                background_spec
                if background_spec is not None
                else BackgroundSpec(BackgroundPolicy.ADDITIVE_MONO)
            ),
            sr=sr,
            rng=rng,
            source_identity=source_identity,
            facts_out=wind_facts_out,
        )
        params = self._gen_wind_noise_params(rng=rng)
        wind_audit = {} if acoustic_audit is not None else None
        noisy_speech, scaled = wind_noise(
            noisy_speech, sample, sr,
            params["threshold"], params["ratio"], params["attack"],
            params["release"], params["sc_gain"], params["clipping"],
            params["clipping_threshold"], wn_snr,
            rng=rng, snr_reference=mix_reference, audit_out=wind_audit,
        )
        # wind_noise 数学内部按 float64 计算；Adapter 边界回到 track 合同 dtype。
        noisy_speech = noisy_speech.astype(np.float32, copy=False)
        if wind_facts_out is not None:
            wind_facts_out["snr_db"] = float(wn_snr)
            wind_facts_out["transform_id"] = "wind_noise_sidechain_v1"
        if acoustic_audit is not None:
            acoustic_audit["wind"] = wind_audit
        additive_noise_2d += scaled.astype(np.float32, copy=False)
        if noisy_speech.shape[1] != actual_len:
            noisy_speech = _align_with_window(
                noisy_speech, actual_len, window_ms=5.0, sr=sr
            )
        return noisy_speech, additive_noise_2d, True

    def _mix_acoustic_read_additive(
        self,
        mode: str,
        sr: int,
        max_dur_sr: int,
        rng: np.random.Generator,
        reuse_noise: bool,
    ) -> np.ndarray:
        """读取 single 或 dual-blend additive sample，并保持各自 reuse cache。"""
        if mode == "single":
            cache_attr = "_noise_single_reuse_cache"
            reader = self._read_one_additive_noise_scp_only
        elif mode == "dual":
            cache_attr = "_noise_dual_reuse_cache"
            reader = self._read_dual_additive_noise_scp_blended
        else:
            raise ValueError(f"unknown additive mode: {mode!r}")
        cached = getattr(self, cache_attr)
        if reuse_noise and cached is not None and cached[0] == sr:
            sample = cached[1].copy()
            sidecar = cached[2] if len(cached) >= 3 else ()
            self._record_reused_noise_sources(sidecar)
            record = self._acoustic_plan_record()
            if record is not None:
                record["noise_assets"].append(sample.copy())
            return sample
        record = self._acoustic_plan_record()
        source_start = len(record.get("noise_sources", ())) if record is not None else 0
        sample = reader(sr, max_dur_sr, rng)
        if reuse_noise:
            setattr(
                self,
                cache_attr,
                (
                    sr,
                    np.ascontiguousarray(sample, dtype=np.float32).copy(),
                    self._reuse_noise_source_sidecar(record, source_start),
                ),
            )
        return sample

    def _mix_acoustic_colored_noise(
        self,
        noisy_speech: np.ndarray,
        mix_reference: np.ndarray,
        additive_noise_2d: np.ndarray,
        additive_snr_db_list: list[float],
        sr: int,
        rng: np.random.Generator,
        acoustic_audit: dict | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """施加 C-3 colored noise；保持原 RNG draw 顺序。"""
        cfg = self.sim_cfg
        if rng_random(rng) < cfg.prob_colored_noise:
            lo, hi = cfg.colored_noise_snr_range
            snr = float(rng_uniform(rng, float(lo), float(hi)))
            colored_audit = {} if acoustic_audit is not None else None
            augmented = colored_noise(
                mix_reference[0], sr, snr, snr, cfg.colored_noise_types,
                rng=rng, audit_out=colored_audit,
            )
            component = (augmented - mix_reference[0]).astype(np.float32)
            noisy_speech = (noisy_speech + component).astype(np.float32)
            additive_noise_2d += component
            additive_snr_db_list.append(snr)
            record = self._acoustic_plan_record()
            if record is not None:
                record["colored_enabled"] = True
                record["colored_snr_db"] = float(snr)
                record["colored_type"] = (
                    str(colored_audit.get("color")) if colored_audit else None
                )
            if acoustic_audit is not None:
                acoustic_audit["colored_noise"] = colored_audit
        return noisy_speech, additive_noise_2d

    def _mix_acoustic(
        self,
        speech_2d: np.ndarray,
        sr: int,
        actual_len: int,
        max_dur_sr: int,
        rng: np.random.Generator,
        need_noise: bool,
        *,
        forced_source_volume_gain: float | None = None,
        forced_scene_plan: AcousticScenePlan | None = None,
        forced_resolved_assets: ResolvedAssets | None = None,
        forced_asset_role: str | None = None,
        resolved_asset_role: str | None = None,
    ) -> MixAcousticOutput:
        """记录并执行共享声学计划；12 positional Interface 保持不变。

        recorder 同步借用只读消费的 RIR，返回前由 ResolvedAssets 发布独立快照。
        ``forced_scene_plan`` + ``forced_resolved_assets`` 是仅限 C-0 RIR 与
        C-0b source gain 的 runtime replay seam。启用时严格禁止 C-1/C-2/C-3，
        不选择资源、不读盘、不访问 cache，也不消费 RNG。
        """
        asset_forced = (
            forced_scene_plan is not None
            or forced_resolved_assets is not None
            or forced_asset_role is not None
        )
        if asset_forced:
            if (
                forced_scene_plan is None
                or forced_resolved_assets is None
                or not isinstance(forced_asset_role, str)
                or not forced_asset_role
            ):
                raise C0AssetReplayError(
                    "C-0 asset replay requires scene plan, resolved assets, and role"
                )
            if need_noise:
                raise C0AssetReplayError("C-0 asset replay only supports need_noise=False")

        previous_record = getattr(self, "_active_acoustic_plan_record", None)
        record: dict[str, Any] = {
            "noise_sources": [],
            "noise_source_policies": [],
            "noise_assets": [],
            "colored_enabled": False,
            "colored_type": None,
            "colored_snr_db": None,
            "mix_ratio": None,
        }
        self._active_acoustic_plan_record = record
        try:
            if asset_forced:
                replayed = replay_target_acoustic_scene(
                    speech_2d,
                    scene_plan=forced_scene_plan,
                    resolved_assets=forced_resolved_assets,
                    asset_role=forced_asset_role,
                    sample_rate=sr,
                    num_samples=actual_len,
                    need_noise=need_noise,
                    source_volume_normalization_enabled=bool(
                        getattr(self.sim_cfg, "source_volume_normalization_enabled", False)
                    ),
                    source_volume_use_rms=bool(
                        getattr(self.sim_cfg, "source_volume_use_rms", True)
                    ),
                )
                record["rir_plan"] = forced_scene_plan.rir
                record["rir_asset"] = forced_resolved_assets.rir
                record["source_gain_plan"] = forced_scene_plan.source_gain
                gain = float(forced_scene_plan.source_gain.gain)
                c0_audit_stems = (
                    {
                        "full": np.asarray(replayed.noisy_speech, dtype=np.float32).ravel().copy(),
                        "early": np.asarray(replayed.mix_reference, dtype=np.float32).ravel().copy(),
                        "source_volume_gain": gain,
                    }
                    if self.export_simulation_stems
                    else None
                )
                self._record_post_rir_calibration(
                    replayed.noisy_speech, replayed.mix_reference, gain
                )
                output = MixAcousticOutput(
                    noisy_speech=replayed.noisy_speech,
                    clean_target=replayed.clean_target,
                    noisy_after_rir=replayed.noisy_speech.copy(),
                    additive_noise_2d=np.zeros_like(replayed.noisy_speech, dtype=np.float32),
                    additive_snr_db_list=[],
                    used_sidechain_wind=False,
                    mix_two=False,
                    want_sidechain=False,
                    wn_snr=0.0,
                    snr_primary_drawn_db=None,
                    wind_noise_snr_candidate_db=None,
                    use_reverb=bool(forced_scene_plan.rir.enabled),
                    acoustic_audit={} if self.export_simulation_stems else None,
                    source_volume_calibration=replayed.source_volume_calibration,
                    c0_audit_stems=c0_audit_stems,
                )
            else:
                output = self._mix_acoustic_impl(
                    speech_2d,
                    sr,
                    actual_len,
                    max_dur_sr,
                    rng,
                    need_noise,
                    forced_source_volume_gain=forced_source_volume_gain,
                )
        except C0AssetReplayError:
            raise
        except (TypeError, ValueError) as exc:
            if asset_forced:
                raise C0AssetReplayError(
                    f"C-0 asset replay contract failed: {exc}"
                ) from exc
            raise
        finally:
            if previous_record is None:
                self.__dict__.pop("_active_acoustic_plan_record", None)
            else:
                self._active_acoustic_plan_record = previous_record
        additive_snr = (
            float(output.additive_snr_db_list[0])
            if output.additive_snr_db_list
            else None
        )
        noise_sources = tuple(
            replace(
                source,
                snr_db=(
                    float(output.wn_snr)
                    if source.family == "wind" and output.used_sidechain_wind
                    else additive_snr
                ),
            )
            for source in record["noise_sources"]
        )
        output.acoustic_scene_plan = AcousticScenePlan(
            identity={"online_rir_scene": record["online_rir_scene"]} if "online_rir_scene" in record else {},
            sample_rate=int(sr),
            num_samples=int(actual_len),
            need_noise=bool(need_noise),
            rir=record.get("rir_plan")
            or RIRPlan.disabled(
                early_rir_sec=float(getattr(self.sim_cfg, "early_rir_sec", 0.0)),
                target_type=str(getattr(self.sim_cfg, "target_type", "early")),
            ),
            source_gain=record.get("source_gain_plan") or SourceGainPlan.disabled(),
            noise=(
                NoisePlan(
                    enabled=True,
                    mode=str(record.get("noise_mode", "single")),
                    mix_ratio=record.get("mix_ratio"),
                    sources=noise_sources,
                    colored_enabled=bool(record["colored_enabled"]),
                    colored_type=record["colored_type"],
                    colored_snr_db=record["colored_snr_db"],
                )
                if need_noise
                else NoisePlan.disabled()
            ),
        )
        output.resolved_assets = ResolvedAssets(
            rir=record.get("rir_asset"),
            noises=tuple(record["noise_assets"]),
            role=resolved_asset_role,
            sample_rate=int(sr),
            num_samples=int(actual_len),
        )
        output.noise_source_policies = tuple(record["noise_source_policies"])
        output.online_rir_scene = record.get("online_rir_scene")
        return output

    def _mix_acoustic_bundle(
        self,
        speech: WaveformBundle,
        sr: int,
        max_dur_sr: int,
        rng: np.random.Generator,
        need_noise: bool,
        *,
        forced_source_volume_gain: float | None = None,
        source_trace: SourceSimulationTrace | None = None,
    ) -> AcousticBundleResult:
        """在共享 seam 上执行声学混合；旧 pipeline 仍可直接调用 `_mix_acoustic`。"""
        renderer = getattr(self, "acoustic_renderer", None) or AcousticRenderer()
        return renderer.render_bundle(
            self,
            speech,
            sr,
            max_dur_sr,
            rng,
            need_noise,
            forced_source_volume_gain=forced_source_volume_gain,
            source_trace=source_trace,
        )

    def _mix_acoustic_impl(
        self,
        speech_2d: np.ndarray,
        sr: int,
        actual_len: int,
        max_dur_sr: int,
        rng: np.random.Generator,
        need_noise: bool,
        *,
        forced_source_volume_gain: float | None = None,
        forced_scene_plan: AcousticScenePlan | None = None,
        forced_resolved_assets: ResolvedAssets | None = None,
    ) -> MixAcousticOutput:
        """Run Layer B/C-0/C-1/C-2/C-3: RIR, target calibration, and noise.

        Args:
            need_noise: Whether this branch needs additive noise + side-chain wind
                + colored noise (C-1/C-2/C-3).
            forced_source_volume_gain: TSE replay 专用；复用主目标 probe 的 gain。
                helper 不抽 RNG，但本 pipeline 会消费等价 target 占位 draw 以对齐状态。

        Returns:
            ``MixAcousticOutput``；结构化审计和音量标定通过非 tuple sidecar 携带。
        """
        cfg = self.sim_cfg
        asset_forced = forced_scene_plan is not None
        if asset_forced:
            if forced_resolved_assets is None:
                raise C0AssetReplayError("C-0 asset replay requires resolved assets")
            if bool(getattr(cfg, "source_volume_normalization_enabled", False)) != bool(
                forced_scene_plan.source_gain.enabled
            ):
                raise C0AssetReplayError(
                    "C-0 asset replay source gain plan does not match config"
                )
            use_reverb = bool(forced_scene_plan.rir.enabled)
            snr = 0.0
        else:
            # 默认路径的调用顺序保持不变。
            use_reverb = rng_random(rng) < cfg.prob_reverberation
            snr = float(rng_uniform(rng, cfg.snr_low, cfg.snr_high))
            override = getattr(self, "_online_rir_source", None)
            if override is not None:
                if not isinstance(override, Mapping):
                    raise TypeError("online RIR source override must be a mapping")
                use_reverb = bool(override["enabled"])

        if asset_forced:
            noisy_speech, mix_reference, clean_target = self._mix_acoustic_rir(
                speech_2d,
                sr,
                rng,
                use_reverb,
                forced_rir_plan=forced_scene_plan.rir,
                forced_rir_asset=forced_resolved_assets.rir,
            )
        else:
            noisy_speech, mix_reference, clean_target = self._mix_acoustic_rir(
                speech_2d, sr, rng, use_reverb
            )

        # C-0 输出合同：full/early/target 均必须可安全表示为 finite float32；
        # 音量标定关闭时也 fail-closed，避免非有限值流入后续 placeholder。
        shared_stem_max_abs = source_volume_shared_max_abs(
            noisy_speech, mix_reference, clean_target
        )

        # C-0b：以 early RIR reference 标定目标音量，并把同一 gain 施加到
        # full/early/clean_target。即使 target_type=dry，gain 仍由 early 推导。
        source_volume_calibration: dict | None = None
        if getattr(cfg, "source_volume_normalization_enabled", False):
            source_volume_calibration = {}
            lo, hi = cfg.source_volume_db_range
            forced_gain = (
                forced_scene_plan.source_gain.gain
                if asset_forced
                else forced_source_volume_gain
            )
            if forced_gain is not None and not asset_forced:
                draw_source_volume_target_db(
                    rng,
                    lo,
                    hi,
                    cfg.source_volume_distribution,
                    cfg.source_volume_beta_a,
                    cfg.source_volume_beta_b,
                )
            scaled_reference, _ = source_volume_normalize(
                mix_reference,
                lo,
                hi,
                cfg.source_volume_use_rms,
                sr,
                rng=rng,
                distribution=cfg.source_volume_distribution,
                beta_a=cfg.source_volume_beta_a,
                beta_b=cfg.source_volume_beta_b,
                forced_gain=forced_gain,
                shared_stem_max_abs=shared_stem_max_abs,
                calibration_out=source_volume_calibration,
            )
            source_volume_calibration["stage"] = "post_rir_pre_noise_echo"
            gain = float(source_volume_calibration["gain"])
            noisy_speech = scale_source_volume_stem(
                noisy_speech, gain, stem_name="full_rir"
            )
            mix_reference = scaled_reference
            clean_target = scale_source_volume_stem(
                clean_target, gain, stem_name="clean_target"
            )
        else:
            gain = 1.0
        record = self._acoustic_plan_record()
        if record is not None:
            record["source_gain_plan"] = (
                forced_scene_plan.source_gain
                if asset_forced
                else SourceGainPlan(
                    enabled=bool(getattr(cfg, "source_volume_normalization_enabled", False)),
                    target_db=(
                        float(source_volume_calibration.get("target_db_requested"))
                        if source_volume_calibration
                        and source_volume_calibration.get("target_db_requested") is not None
                        else None
                    ),
                    gain=float(gain),
                    forced=forced_source_volume_gain is not None,
                    reference=(
                        str(source_volume_calibration.get("reference"))
                        if source_volume_calibration
                        and source_volume_calibration.get("reference") is not None
                        else None
                    ),
                )
            )
        c0_audit_stems = (
            {
                "full": np.asarray(noisy_speech, dtype=np.float32).ravel().copy(),
                "early": np.asarray(mix_reference, dtype=np.float32).ravel().copy(),
                "source_volume_gain": float(gain),
            }
            if self.export_simulation_stems
            else None
        )
        self._record_post_rir_calibration(noisy_speech, mix_reference, gain)

        noisy_after_rir = noisy_speech.copy()
        from core.data.simulation.background import mix_background
        from core.data.simulation.role_policy import build_background_spec

        background = mix_background(
            self, noisy_speech, mix_reference, sr, actual_len,
            max_dur_sr, rng, need_noise, snr,
            background_spec=build_background_spec(image_geometry=None),
        )
        return MixAcousticOutput(
            noisy_speech=background.observation,
            clean_target=clean_target,
            noisy_after_rir=noisy_after_rir,
            additive_noise_2d=background.additive,
            additive_snr_db_list=background.snr_values,
            used_sidechain_wind=background.used_sidechain_wind,
            mix_two=background.mix_two,
            want_sidechain=background.want_sidechain,
            wn_snr=background.wind_snr,
            snr_primary_drawn_db=background.primary_snr,
            wind_noise_snr_candidate_db=background.wind_candidate_snr,
            use_reverb=use_reverb,
            acoustic_audit=background.audit,
            source_volume_calibration=source_volume_calibration,
            c0_audit_stems=c0_audit_stems,
        )

    def _align_mix_acoustic_outputs(
        self,
        noisy_speech: np.ndarray,
        clean_target: np.ndarray,
        noisy_after_rir: np.ndarray,
        additive_noise_2d: np.ndarray,
        actual_len: int,
        sr: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Crop/pad the 2-D outputs of ``_mix_acoustic`` to ``actual_len``."""
        if noisy_speech.shape[1] != actual_len:
            noisy_speech = _align_with_window(
                noisy_speech, actual_len, window_ms=5.0, sr=sr
            )
        if clean_target.shape[1] != actual_len:
            clean_target = _align_with_window(
                clean_target, actual_len, window_ms=5.0, sr=sr
            )
        if noisy_after_rir.shape[1] != actual_len:
            noisy_after_rir = _align_with_window(
                noisy_after_rir, actual_len, window_ms=5.0, sr=sr
            )
        if additive_noise_2d.shape[1] != actual_len:
            additive_noise_2d = _align_with_window(
                additive_noise_2d, actual_len, window_ms=5.0, sr=sr
            )
        return noisy_speech, clean_target, noisy_after_rir, additive_noise_2d

    # ------------------------------------------------------------------
    # Layer A: clean-speech pre-processing
    # ------------------------------------------------------------------
    def _prepare_clean_speech_result(
        self,
        speech_uid: str,
        sr: int,
        max_dur_sr: int,
        rng: np.random.Generator,
        *,
        source_trace: SourceSimulationTrace | None = None,
        record_layer_a_metadata: bool = True,
    ) -> PreparedSpeech:
        """加载、增强、对齐并高通 clean speech，显式返回 Layer A 事实。"""
        return self.speech_preparation.prepare_result(
            self,
            speech_uid,
            sr,
            max_dur_sr,
            rng,
            source_trace=source_trace,
            record_layer_a_metadata=record_layer_a_metadata,
        )

    # ------------------------------------------------------------------
    # Layer D / E / F helpers (shared D-guard, finite checks, post-mix)
    # ------------------------------------------------------------------
    def _guard_finite(
        self,
        noisy_1d: np.ndarray,
        clean_1d: np.ndarray,
        speech_uid: str,
        stage: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Raise or sanitize when the mic/clean stems contain non-finite values."""
        if self.strict_load:
            if not (np.isfinite(noisy_1d).all() and np.isfinite(clean_1d).all()):
                raise ValueError(f"NaN/Inf in {stage} (speech_uid={speech_uid})")
        else:
            noisy_1d = np.nan_to_num(noisy_1d, nan=0, posinf=0, neginf=0)
            clean_1d = np.nan_to_num(clean_1d, nan=0, posinf=0, neginf=0)
        return noisy_1d, clean_1d

    def _apply_peak_normalization_result(
        self,
        noisy: np.ndarray,
        clean: np.ndarray,
        noise: np.ndarray,
        rng: np.random.Generator,
    ) -> PeakNormalizationResult:
        """返回 Layer D-normalize 结构化波形与显式缩放事实。"""
        return apply_peak_normalization(
            noisy,
            clean,
            noise,
            mode=str(getattr(self.sim_cfg, "peak_normalization_mode", "off")),
            scale_range=getattr(
                self.sim_cfg, "peak_normalization_scale_range", [0.5, 0.95]
            ),
            rng=rng,
        )

    def _apply_layer_ef_result(
        self,
        noisy_1d: np.ndarray,
        sr: int,
        actual_len: int,
        rng: np.random.Generator,
        used_sidechain_wind: bool,
    ) -> LayerEFResult:
        """应用 Layer E/F，并按实际 Layer E execution facts gate F clipping。"""
        cfg = self.sim_cfg
        layer_e = self._apply_channel_degradation_result(
            noisy_1d, sr, actual_len, rng, used_sidechain_wind
        )
        applied_codec = layer_e.operation_applied("codec")
        applied_clip = layer_e.operation_applied("clipping")
        ops_collected: list = []
        clipping_facts: dict[str, Any] = {}
        waveform = apply_post_augmentations(
            layer_e.waveform,
            sr,
            cfg,
            skip_clip=applied_clip,
            skip_codec=applied_codec,
            rng=rng,
            applied_ops_out=ops_collected,
            clipping_facts_out=clipping_facts,
            sanitize_nonfinite=not getattr(self, "strict_load", True),
        )
        return LayerEFResult(
            waveform=waveform,
            layer_e=layer_e,
            layer_f_applied_ops=tuple(str(op) for op in ops_collected),
            layer_f_clipping_facts=clipping_facts or None,
        )

    def _apply_layer_ef(
        self,
        noisy_1d: np.ndarray,
        sr: int,
        actual_len: int,
        rng: np.random.Generator,
        used_sidechain_wind: bool,
    ) -> tuple[np.ndarray, list[str], bool, bool]:
        """应用 Layer E/F，并保留结构化执行事实 sidecar。"""
        result = self._apply_layer_ef_result(
            noisy_1d,
            sr,
            actual_len,
            rng,
            used_sidechain_wind,
        )
        self._last_layer_e_result = result.layer_e
        self._last_layer_f_applied_ops = list(result.layer_f_applied_ops)
        self._last_layer_f_clipping_facts = result.layer_f_clipping_facts
        return LayerEFOutput(result)

    def __len__(self) -> int:
        return self._total

    def _apply_channel_degradation_result(
        self,
        noisy_1d: np.ndarray,
        sr: int,
        actual_len: int,
        rng: np.random.Generator,
        used_sidechain_wind: bool,
    ) -> LayerEResult:
        """执行 Layer E，并区分 selector 事实与 observation 事实。"""
        cfg = simulation_view_for(self, task=self._simulation_task).shared.channel
        if not cfg.prob_channel_enabled:
            return LayerEResult(
                waveform=noisy_1d,
                enabled=False,
                num_augmentations_drawn=0,
                selected=(),
                operations=(),
                observation_changed=False,
                skip_reason="disabled",
            )

        n_aug = int(
            rng.choice(
                list(cfg.num_augmentations.keys()),
                p=list(cfg.num_augmentations.values()),
            )
        )
        if n_aug <= 0:
            return LayerEResult(
                waveform=noisy_1d,
                enabled=True,
                num_augmentations_drawn=n_aug,
                selected=(),
                operations=(),
                observation_changed=False,
                skip_reason="draw_not_selected",
            )
        if n_aug > len(self._aug_names):
            raise ValueError(
                "Layer E sampled num_augmentations count "
                f"{n_aug}, but only {len(self._aug_names)} unique augmentations are available"
            )

        selected = tuple(
            str(name)
            for name in rng.choice(
                self._aug_names,
                p=self._aug_weights,
                size=n_aug,
                replace=False,
            )
        )
        noisy_2d = noisy_1d.reshape(1, -1)
        operations: list[LayerEOperationResult] = []
        for aug_name in selected:
            before = noisy_2d
            reason: str | None = None
            if used_sidechain_wind and aug_name == "clipping":
                reason = "conflict"
            elif noisy_2d.size == 0:
                reason = "empty"
            elif aug_name == "bandwidth_limitation":
                fs_new, res_type = self._gen_bandwidth_params(sr, rng=rng)
                if fs_new is None:
                    reason = "no_lower_sr"
                else:
                    try:
                        noisy_2d = bandwidth_limitation(
                            noisy_2d,
                            fs=sr,
                            fs_new=fs_new,
                            res_type=res_type,
                        )
                    except ImportError:
                        reason = "dependency_unavailable"
            elif aug_name == "clipping":
                min_q, max_q = self._gen_clipping_params(rng=rng)
                noisy_2d = clipping_aug(
                    noisy_2d,
                    min_quantile=min_q,
                    max_quantile=max_q,
                )
            elif aug_name == "codec":
                fmt, enc, qs = self._gen_codec_params(rng=rng)
                codec_result = codec_compression_result(
                    noisy_2d,
                    sr,
                    format=fmt,
                    encoder=enc,
                    qscale=qs,
                )
                noisy_2d = codec_result.waveform
                reason = codec_result.reason
            elif aug_name == "packet_loss":
                pli, pkt_ms = self._gen_packet_loss_indices(
                    actual_len, sr, rng=rng
                )
                if not pli:
                    reason = "no_packets"
                else:
                    noisy_2d = packet_loss(noisy_2d, sr, pli, pkt_ms)

            changed = (
                reason is None
                and (
                    noisy_2d.shape != before.shape
                    or not np.array_equal(noisy_2d, before)
                )
            )
            if reason is None and not changed:
                reason = "no_change"
            operations.append(
                LayerEOperationResult(
                    name=aug_name,
                    applied=changed,
                    observation_changed=changed,
                    reason=reason,
                )
            )

        if noisy_2d.shape[1] != actual_len:
            noisy_2d = _align_with_window(
                noisy_2d, actual_len, window_ms=5.0, sr=sr
            )
        waveform = noisy_2d.flatten().astype(np.float32)
        observation_changed = (
            waveform.shape != noisy_1d.shape
            or not np.array_equal(waveform, noisy_1d)
        )
        return LayerEResult(
            waveform=waveform,
            enabled=True,
            num_augmentations_drawn=n_aug,
            selected=selected,
            operations=tuple(operations),
            observation_changed=observation_changed,
        )

    def _enrich_repeat_export_final(self, re: dict, fs_out: int, len_before: int, len_after: int) -> None:
        """Add final alignment info to the repeat-export dict (mutates in place)."""
        max_d = self._max_duration_for_sr(fs_out)
        re['samples_before_final_align'] = int(len_before)
        re['samples_after_final_align'] = int(len_after)
        re['final_max_duration_samples'] = int(max_d)
        if max_d > 0 and len_before > max_d:
            re['final_align_mode'] = 'crop_to_max'
        elif getattr(self, "_fixed_silence_canvas_active", False):
            re['final_align_mode'] = 'already_fixed_canvas'
        elif self.repeat_to_max_duration and max_d > 0:
            if len_before == 0:
                re['final_align_mode'] = 'pad_zero'
            elif len_before < max_d:
                re['final_align_mode'] = 'tile_to_max'
            else:
                re['final_align_mode'] = 'off'
        else:
            re['final_align_mode'] = 'off'

    def _make_silence_placeholder(self, work_sr: int) -> tuple[np.ndarray, int]:
        """Return a silence waveform and its length for placeholder samples."""
        canvas = getattr(self, "canvas_spec", None)
        has_duration_cap = (
            canvas.has_duration_cap
            if canvas is not None
            else bool(getattr(self, "max_duration", 0) > 0 or getattr(self, "max_duration_seconds", None))
        )
        out_len = (
            self._max_duration_for_sr(work_sr)
            if has_duration_cap
            else self.placeholder_num_samples
        )
        return np.zeros(out_len, dtype=np.float32), out_len

    def _make_observation_placeholder(self, work_sr: int) -> tuple[np.ndarray, int]:
        """Return a mono or channel-first spatial observation placeholder."""
        silence, out_len = self._make_silence_placeholder(work_sr)
        if not getattr(self.sim_cfg, "spatial_enabled", False):
            return silence, out_len
        spatial = getattr(self._simulation_view.shared, "spatial", None)
        channels = getattr(spatial, "channels", None)
        if type(channels) is not int:
            positions = getattr(spatial, "microphone_positions", None)
            if positions is None:
                raise RuntimeError("spatial placeholder requires resolved microphone geometry")
            channels = len(positions)
        if channels < 1:
            raise RuntimeError("spatial placeholder requires at least one microphone channel")
        positions = getattr(spatial, "microphone_positions", None)
        if positions is not None and len(positions) != channels:
            raise RuntimeError(
                "spatial placeholder microphone cardinality does not match resolved geometry"
            )
        return np.zeros((channels, out_len), dtype=np.float32), out_len

    def _select_branch(
        self,
        branches: list[tuple[str, float]],
        rng: np.random.Generator,
        force_branch: str | None = None,
        normalize_force: Callable[[str], str] | None = None,
    ) -> tuple[str, float | None]:
        """Draw a simulation branch from an ordered list of (name, probability) pairs.

        The last entry acts as the fallback branch when the cumulative probability
        does not cover the random draw.  When ``force_branch`` is given, the draw
        is skipped and the forced branch is returned (optionally normalised).
        """
        sampler = getattr(self, "scenario_sampler", None) or ScenarioSampler()
        return sampler.select_branch(
            branches,
            rng,
            force_branch=force_branch,
            normalize_force=normalize_force,
        )

    def _resample_simulation_stems_inplace(
        self,
        fs_from: int,
        fs_to: int,
        keys: tuple[str, ...],
    ) -> None:
        """Resample exported Layer-D stems after target_sr changes."""
        if (
            not self.export_simulation_stems
            or not self._last_simulation_stems
            or fs_from == fs_to
        ):
            return
        st = self._last_simulation_stems
        for key in keys:
            if st.get(key) is None:
                continue
            st[key] = _resample_audio(
                np.asarray(st[key], dtype=np.float32).ravel(), fs_from, fs_to
            )

    def _slice_simulation_stems_inplace(
        self,
        start: int,
        length: int,
        ref_len: int,
        keys: tuple[str, ...],
    ) -> None:
        """Crop exported Layer-D stems when the main waveform is cropped."""
        if not self.export_simulation_stems or not self._last_simulation_stems:
            return
        end = int(start) + int(length)
        st = self._last_simulation_stems
        for key in keys:
            if st.get(key) is None:
                continue
            a = np.asarray(st[key], dtype=np.float32).ravel()
            if int(a.size) != int(ref_len):
                continue
            st[key] = a[start:end].astype(np.float32, copy=True)

    def _crop_output_context(
        self,
        ctx: Any | None,
        start: int,
        length: int,
    ) -> Any | None:
        """Hook for subclass-specific output context cropping.

        Default no-op; AEC overrides this to crop farend/echo context.
        """
        return ctx

    def _repeat_ctx_extra_arrays(
        self,
        ctx: Any | None,
        ref_len: int,
    ) -> tuple[list[np.ndarray], Any | None]:
        """Hook: 抽出需要与 clean/noisy 共享同一份 repeat(tile/gap) 时间表的 ctx 波形。

        Returns:
            ``(arrays, rebuild)``：``arrays`` 必须已与 clean/noisy 等长（长度
            ref_len），并与 final ``TimeTransformPlan`` 同步应用；
            ``rebuild(transformed_arrays)`` 把变换后波形写回并返回新的 ctx。
            SE 默认无 ctx 波形，返回 ``([], None)``。
        """
        return [], None

    def _build_final_time_plan(
        self,
        speech: np.ndarray,
        noisy: np.ndarray,
        fs_out: int,
        rng: np.random.Generator,
        *,
        fixed_silence_canvas_active: bool | None = None,
    ) -> FinalTimePlan:
        """在 legacy 决策位置抽样，并冻结 final crop/tile 参数。"""
        max_dur = self._max_duration_for_sr(fs_out)
        len_s, len_n = len(speech), len(noisy)
        fixed_canvas = (
            bool(getattr(self, "_fixed_silence_canvas_active", False))
            if fixed_silence_canvas_active is None
            else bool(fixed_silence_canvas_active)
        )
        if fixed_canvas and max_dur > 0:
            if len_s != len_n or len_s < max_dur:
                raise RuntimeError(
                    "fixed silence canvas output length invariant violated: "
                    f"speech={len_s}, noisy={len_n}, target={max_dur}; refusing to tile"
                )
        if (not self.repeat_to_max_duration and not fixed_canvas) or max_dur <= 0:
            return FinalTimePlan(None, len_s, fixed_canvas)
        if len_s == len_n == max_dur:
            return FinalTimePlan(None, max_dur, fixed_canvas)
        if len_s == len_n > max_dur:
            start = (len_s - max_dur) // 2
            transform = plan_time_transform(
                input_length=len_s,
                operations=(CropOp(start=start, output_length=max_dur),),
                policy=POLICY_FINAL_CENTER_CROP,
            )
            return FinalTimePlan(transform, max_dur, fixed_canvas, crop_start=start)
        if len_s != len_n:
            raise RuntimeError(
                "speech and noisy must have the same length: "
                f"speech={len_s}, noisy={len_n}"
            )
        if len_s == 0:
            transform = plan_time_transform(
                input_length=0,
                operations=(CanvasOp(output_length=max_dur),),
                policy="final_zero_canvas",
            )
        else:
            transform = _build_align_to_max_duration_plan(
                len_s,
                max_dur,
                sr=fs_out,
                insert_silence_gap_between_tiles=self.repeat_tile_silence_gap,
                gap_min_sec=self.repeat_tile_silence_gap_range[0],
                gap_max_sec=self.repeat_tile_silence_gap_range[1],
                rng=rng,
            )
        return FinalTimePlan(transform, max_dur, fixed_canvas)

    def _apply_final_time_plan(
        self,
        plan: FinalTimePlan,
        speech: np.ndarray,
        noisy: np.ndarray,
        ctx: Any | None = None,
        stem_keys: tuple[str, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray, int, Any | None]:
        """无 RNG 地同步应用 final plan；ctx 继续通过任务 Adapter 重建。"""
        if plan.transform is None:
            return speech, noisy, plan.output_length, ctx
        input_length = plan.transform.input_length
        if plan.crop_start is not None:
            speech = _apply_1d_time_plan(plan.transform, speech)
            noisy = _apply_1d_time_plan(plan.transform, noisy)
            self._slice_simulation_stems_inplace(
                plan.crop_start, plan.output_length, input_length, stem_keys
            )
            ctx = self._crop_output_context(
                ctx, plan.crop_start, plan.output_length
            )
            return speech, noisy, plan.output_length, ctx

        extras, rebuild_ctx = self._repeat_ctx_extra_arrays(ctx, input_length)
        speech = _apply_1d_time_plan(plan.transform, speech)
        noisy = _apply_1d_time_plan(plan.transform, noisy)
        if rebuild_ctx is not None:
            ctx = rebuild_ctx(
                [_apply_1d_time_plan(plan.transform, item) for item in extras]
            )
        return speech, noisy, plan.output_length, ctx

    def _export_layer_d_stems(self, **stems: np.ndarray) -> None:
        """Copy post-Layer-D waveforms into the simulation-stem export dict."""
        if self._last_simulation_stems is None:
            return
        for key, value in stems.items():
            self._last_simulation_stems[key] = np.asarray(value, dtype=np.float32).copy()

    def _gen_bandwidth_params(self, sr: int, rng: np.random.Generator | None=None) -> tuple[int | None, str | None]:
        """Pick a lower sample rate and resampling method for bandwidth limitation.

        显式 ``target_sr``（UniPASE/GAP 固定 8000）优先于随机低采样率；只有它低于
        当前采样率时才生效，否则本项增强不执行 ``(None, None)``。未配置时保持随机
        低采样率行为不变。
        """
        aug_cfg = self.sim_cfg.augmentations.get("bandwidth_limitation")
        target_sr = aug_cfg.get("target_sr") if isinstance(aug_cfg, dict) else None
        if target_sr is not None:
            if type(target_sr) is not int or target_sr <= 0:
                raise ValueError(
                    "bandwidth_limitation.target_sr must be an exact positive integer, "
                    f"got {target_sr!r}"
                )
            if target_sr >= sr:
                return (None, None)
            return (target_sr, rng_choice(rng, _BANDWIDTH_RESAMPLE_METHODS))
        fs_opts = [fs for fs in _BANDWIDTH_SAMPLE_RATES if fs < sr]
        if not fs_opts:
            return (None, None)
        return (rng_choice(rng, fs_opts), rng_choice(rng, _BANDWIDTH_RESAMPLE_METHODS))

    def _gen_clipping_params(self, rng: np.random.Generator | None=None) -> tuple[float, float]:
        """Generate quantile range for clipping augmentation."""
        aug_cfg = self.sim_cfg.augmentations['clipping']
        return (rng_uniform(rng, *aug_cfg['clipping_min_quantile']), rng_uniform(rng, *aug_cfg['clipping_max_quantile']))

    def _gen_codec_params(self, rng: np.random.Generator | None=None) -> tuple[str, str, int]:
        """Generate codec format, encoder, and qscale for codec compression."""
        aug_cfg = self.sim_cfg.augmentations['codec']
        codec_cfg = rng_choice(rng, aug_cfg['config'])
        fmt = codec_cfg['format']
        enc = codec_cfg['encoder']
        if isinstance(enc, list):
            enc = rng_choice(rng, enc)
        qs = codec_cfg['qscale']
        if isinstance(qs, (list, tuple)) and len(qs) == 2:
            qs = rng_randint(rng, qs[0], qs[1])
        elif isinstance(qs, int):
            pass
        else:
            qs = rng_randint(rng, 1, 10)
        if fmt == 'ogg' and enc == 'vorbis' and (qs == 0):
            qs = 1
        return (fmt, enc, qs)

    def _gen_packet_loss_indices(self, speech_length: int, sr: int, rng: np.random.Generator | None=None) -> tuple[list[int], int]:
        """Generate packet-loss burst indices for packet-loss augmentation."""
        aug_cfg = self.sim_cfg.augmentations['packet_loss']
        pkt_ms = aug_cfg['packet_duration_ms']
        loss_rate = rng_uniform(rng, *aug_cfg['packet_loss_rate'])
        max_cont = aug_cfg['max_continuous_packet_loss']
        dur_ms = speech_length / sr * 1000
        n_packets = int(dur_ms // pkt_ms)
        n_loss = int(round(loss_rate * dur_ms / pkt_ms))
        if n_loss <= 0 or n_packets <= 0:
            return ([], pkt_ms)
        burst_lens = []
        remaining = n_loss
        while remaining > 0:
            bl = rng_randint(rng, 1, min(max_cont, remaining))
            burst_lens.append(bl)
            remaining -= bl
        starts = rng_sample(rng, range(n_packets), min(len(burst_lens), n_packets))
        indices = set()
        for s, bl in zip(starts, burst_lens):
            for j in range(bl):
                if s + j < n_packets:
                    indices.add(s + j)
        return (sorted(indices), pkt_ms)

    def _gen_wind_noise_params(self, rng: np.random.Generator | None=None) -> dict:
        """Generate side-chain wind-noise parameters."""
        wc = self.sim_cfg.wind_noise_config
        return {'threshold': rng_uniform(rng, *wc['threshold']), 'ratio': rng_uniform(rng, *wc['ratio']), 'attack': rng_uniform(rng, *wc['attack']), 'release': rng_uniform(rng, *wc['release']), 'sc_gain': rng_uniform(rng, *wc['sc_gain']), 'clipping': rng_random(rng) < wc['clipping_chance'], 'clipping_threshold': rng_uniform(rng, *wc['clipping_threshold'])}

    def _speech_splice_candidate_uids(
        self, anchor_uid: str
    ) -> tuple[str, ...] | None:
        """speaker controller 开启时限定 canonical identity；否则保持全路径池。"""
        identity_by_uid = getattr(self, "_speaker_splice_identity_by_uid", None)
        if not identity_by_uid:
            return None
        identity = identity_by_uid.get(anchor_uid)
        if identity is None:
            return (anchor_uid,)
        return self._speaker_splice_uids_by_identity.get(identity, (anchor_uid,))

    def _get_clean_speech(
        self,
        speech_uid: str,
        sr: int,
        rng: np.random.Generator | None = None,
        single_utterance: bool = False,
    ) -> np.ndarray:
        """加载 clean speech；``single_utterance`` 时跳过 splice 与预裁剪。"""
        preparation = self.speech_preparation
        return preparation.get_clean_speech(
            self,
            speech_uid,
            sr,
            rng,
            single_utterance=single_utterance,
        )

    def _get_from_index(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self._total:
            raise IndexError(index)
        return self._filtered_indices[index]

    def _max_duration_for_sr(self, sr: int) -> int:
        """Max duration in samples for a given sample rate."""
        return self.canvas_spec.samples_for_sr(sr)

    def _read_noise_uid(
        self,
        uid: str,
        sr: int,
        max_duration: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """按 UID 读取噪声，并在 decode/crop 后记录实际 source。"""
        path = self.all_noise_flt[f"{self._noise_uid_prefix}{uid}"]
        crop_record: dict[str, Any] = {}
        store = self.resource_store
        sample = store.read_noise(
            path,
            sr,
            max_duration,
            spatial=bool(self.sim_cfg.spatial_enabled),
            rng=rng,
            crop_record=crop_record,
        )
        self._record_noise_source(
            family="ordinary",
            uid=uid,
            path=path,
            crop=crop_record.get("crop"),
        )
        record = self._acoustic_plan_record()
        if record is not None:
            record["noise_assets"].append(sample.copy())
        return sample

    def _read_dual_additive_noise_scp_blended(self, sr: int, max_duration: int | None, rng: np.random.Generator) -> np.ndarray:
        """读取双噪声；category map 完整时第二段必须来自不同 category。"""
        if getattr(self, "_music_routing_enabled", False):
            from core.data.background_sources import read_dual

            return read_dual(self, sr, max_duration, rng)
        max_dur = max_duration if max_duration is not None else self._max_duration_for_sr(sr)
        category_complete = self._ordinary_noise_categories_complete()
        if self.balanced_noise_sampling_enabled or category_complete:
            noise_uid = (
                self._select_balanced_additive_noise_uid(sr, rng)
                if self.balanced_noise_sampling_enabled
                else self._select_sample(sr, self.noise_source, rng=rng)
            )
            if noise_uid is None:
                raise RuntimeError(f'No additive noise source for sr={sr}.')
            first_category_id = self._noise_category_id_by_uid[noise_uid]
            noise = self._read_noise_uid(noise_uid, sr, max_dur, rng)
            if self.balanced_noise_sampling_enabled:
                noise2_uid = self._select_balanced_additive_noise_uid(
                    sr, rng, exclude_category_id=first_category_id, exclude_uid=noise_uid
                )
            else:
                noise2_uid = self._select_natural_different_category_uid(
                    sr,
                    rng,
                    exclude_category_id=first_category_id,
                    exclude_uid=noise_uid,
                )
            if noise2_uid is None:
                return noise
            noise2 = self._read_noise_uid(noise2_uid, sr, max_dur, rng)
        else:
            noise = self._read_one_additive_noise_scp_only(sr, max_dur, rng)
            noise2 = self._read_one_additive_noise_scp_only(sr, max_dur, rng)
        noise2 = self._extend_noise_to_length(noise2, noise.shape[-1])
        ratio = rng_uniform(rng, *self.sim_cfg.noise_mix_ratio_range)
        record = self._acoustic_plan_record()
        if record is not None:
            record["mix_ratio"] = float(ratio)
        return noise + ratio * noise2[:, :noise.shape[-1]]

    def _read_one_additive_noise_scp_only(self, sr: int, max_duration: int | None, rng: np.random.Generator) -> np.ndarray:
        """Read a single additive noise segment from noise_scp only."""
        if getattr(self, "_music_routing_enabled", False):
            from core.data.background_sources import read_one

            return read_one(self, sr, max_duration, rng)
        max_dur = max_duration if max_duration is not None else self._max_duration_for_sr(sr)
        if self.balanced_noise_sampling_enabled:
            noise_uid = self._select_balanced_additive_noise_uid(sr, rng)
        else:
            noise_uid = self._select_sample(sr, self.noise_source, rng=rng)
        if noise_uid is None:
            raise RuntimeError(f'No additive noise source for sr={sr}.')
        return self._read_noise_uid(noise_uid, sr, max_dur, rng)

    def _read_one_additive_noise_scp_only_with_identity(
        self,
        sr: int,
        max_duration: int | None,
        rng: np.random.Generator,
    ) -> tuple[str, np.ndarray]:
        """读取一段 additive noise，并返回实际选中的 source identity。"""
        max_dur = max_duration if max_duration is not None else self._max_duration_for_sr(sr)
        if getattr(self, "_music_routing_enabled", False):
            from core.data.background_sources import read_source, select_source

            source = select_source(self, sr, rng)
            if source is None:
                raise RuntimeError(f"No additive noise source for sr={sr}.")
            family, uid, _ = source
            waveform = read_source(self, source, sr, max_dur, rng)
            return f"{family}_scp:{uid}", waveform

        if self.balanced_noise_sampling_enabled:
            noise_uid = self._select_balanced_additive_noise_uid(sr, rng)
        else:
            noise_uid = self._select_sample(sr, self.noise_source, rng=rng)
        if noise_uid is None:
            raise RuntimeError(f"No additive noise source for sr={sr}.")
        return f"noise_scp:{noise_uid}", self._read_noise_uid(
            noise_uid, sr, max_dur, rng
        )

    def _read_one_speech(self, path: str, sr: int) -> np.ndarray:
        """Load a speech file without occupying the shared audio cache."""
        store = self.resource_store
        return store.read_speech(path, sr)

    def _repeat_export_info(self, sr: int, samples_before: int, samples_after: int, max_dur_sr: int) -> dict:
        """Build a JSON-serializable summary of repeat-to-max-duration state."""
        out: dict = {'repeat_to_max_duration': bool(self.repeat_to_max_duration), 'max_duration_samples': int(max_dur_sr), 'sample_rate': int(sr), 'samples_before_repeat_step': int(samples_before), 'samples_after_repeat_step': int(samples_after), 'tile_silence_gap': bool(self.repeat_tile_silence_gap)}
        if self.repeat_tile_silence_gap:
            out['gap_range_sec'] = [float(self.repeat_tile_silence_gap_range[0]), float(self.repeat_tile_silence_gap_range[1])]
        if self.repeat_to_max_duration and max_dur_sr > 0:
            if samples_before == 0:
                out['align_mode'] = 'pad_zero'
            elif samples_before > max_dur_sr:
                out['align_mode'] = 'crop_to_max'
            elif samples_before < max_dur_sr:
                out['align_mode'] = 'tile_to_max'
            else:
                out['align_mode'] = 'off'
        else:
            out['align_mode'] = 'off'
        return out

    def _sample_selector_source_entry(
        self, source_by_fs: dict
    ) -> tuple[int, _SampleSelectorSourceCache]:
        """返回 identity-keyed source cache；source 数受 LRU 上限约束。"""
        cache = getattr(self, "_sample_selector_source_cache", None)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            self._sample_selector_source_cache = cache
            self._sample_selector_cache_uid_refs = 0
        source_key = id(source_by_fs)
        entry = cache.get(source_key)
        if entry is not None and entry.source_by_fs is source_by_fs:
            cache.move_to_end(source_key)
            return source_key, entry
        if entry is not None:
            self._sample_selector_cache_uid_refs -= entry.uid_refs
            del cache[source_key]
        # oversized source 只允许独占 cache；切换 source 时先释放它。entry 保留
        # source_by_fs 强引用并在命中时校验 identity，避免 id 复用命中陈旧 tuple。
        while cache and (
            len(cache) >= _SAMPLE_SELECTOR_CACHE_MAX_SOURCES
            or self._sample_selector_cache_uid_refs
            > _SAMPLE_SELECTOR_CACHE_MAX_UID_REFS
        ):
            _, evicted = cache.popitem(last=False)
            self._sample_selector_cache_uid_refs -= evicted.uid_refs
        entry = _SampleSelectorSourceCache(source_by_fs)
        cache[source_key] = entry
        return source_key, entry

    def _sample_selector_uids_for_fs(
        self,
        source_key: int,
        entry: _SampleSelectorSourceCache,
        fs: int,
    ) -> tuple[tuple[str, ...], bool]:
        """lazy 物化并驻留一个 FS 的稳定 UID tuple。"""
        cached = entry.uids_by_fs.get(fs)
        if cached is not None:
            return cached, True
        uids = tuple(entry.source_by_fs[fs].keys())
        uid_refs = len(uids)
        total_refs = int(getattr(self, "_sample_selector_cache_uid_refs", 0))
        cache = self._sample_selector_source_cache
        while total_refs + uid_refs > _SAMPLE_SELECTOR_CACHE_MAX_UID_REFS:
            victim_key = next((key for key in cache if key != source_key), None)
            if victim_key is None:
                break
            evicted = cache.pop(victim_key)
            total_refs -= evicted.uid_refs
        # 若当前 source 自身超过预算，它作为唯一 oversized stable entry 驻留；
        # 后续相同 source O(1) 复用，访问其他 source 时由上层先淘汰本 entry。
        entry.uids_by_fs[fs] = uids
        entry.uid_refs += uid_refs
        self._sample_selector_cache_uid_refs = total_refs + uid_refs
        return uids, True

    @staticmethod
    def _sample_selector_remember_processing_view(
        entry: _SampleSelectorSourceCache,
        sr: int,
        view: tuple[str, ...] | _SampleSelectorTierView | None,
    ) -> None:
        """在 per-source processing-SR LRU 中记录不复制 UID 的 tier view。"""
        views = entry.processing_sr_views
        views[sr] = view
        views.move_to_end(sr)
        while len(views) > _SAMPLE_SELECTOR_CACHE_MAX_PROCESSING_SRS:
            views.popitem(last=False)

    def _sample_selector_candidates(
        self, sr: int, source_by_fs: dict
    ) -> tuple[str, ...] | _SampleSelectorTierView | None:
        """返回 exact→首个 higher→all 候选，保持原 dict 迭代顺序。"""
        target_sr = int(sr)
        source_key, entry = self._sample_selector_source_entry(source_by_fs)
        views = entry.processing_sr_views
        if target_sr in views:
            views.move_to_end(target_sr)
            return views[target_sr]

        selected_fs = None
        if target_sr in source_by_fs and source_by_fs[target_sr]:
            selected_fs = target_sr
        else:
            selected_fs = next(
                (fs for fs in entry.sorted_nonempty_sample_rates if fs > target_sr),
                None,
            )
        if selected_fs is not None:
            candidates, cacheable = self._sample_selector_uids_for_fs(
                source_key, entry, selected_fs
            )
            if cacheable:
                self._sample_selector_remember_processing_view(
                    entry, target_sr, candidates
                )
            return candidates

        uid_groups = []
        cacheable = True
        for fs in entry.source_order:
            if not source_by_fs[fs]:
                continue
            uids, retained = self._sample_selector_uids_for_fs(source_key, entry, fs)
            uid_groups.append(uids)
            cacheable = cacheable and retained
        if not uid_groups:
            candidates = None
        elif len(uid_groups) == 1:
            candidates = uid_groups[0]
        else:
            candidates = _SampleSelectorTierView(tuple(uid_groups))
        if cacheable:
            self._sample_selector_remember_processing_view(entry, target_sr, candidates)
        return candidates

    def _select_sample(
        self,
        sr: int,
        source_by_fs: dict,
        rng: np.random.Generator | None = None,
    ) -> str | None:
        """Select one UID via cached exact→higher→all candidates and one RNG draw."""
        candidates = self._sample_selector_candidates(sr, source_by_fs)
        return rng_choice(rng, candidates) if candidates else None

    @staticmethod
    def _speech_splice_export_info(cfg, sr: int, n_samples: int) -> dict:
        """Build a JSON-serializable summary of speech-splice state."""
        enabled = bool(getattr(cfg, 'speech_splice', False))
        out: dict = {'enabled': enabled, 'target_seconds': float(getattr(cfg, 'speech_splice_seconds', 0.0)), 'samples_after_crop_repeat': int(n_samples), 'sample_rate': int(sr)}
        if enabled:
            gr = getattr(cfg, 'speech_splice_gap_range', [0.02, 0.2])
            if isinstance(gr, (list, tuple)) and len(gr) >= 2:
                out['gap_range_sec'] = [float(gr[0]), float(gr[1])]
            out['equal_volume'] = bool(getattr(cfg, 'speech_splice_equal_volume', True))
        return out

    def get_source_length(self) -> list[int]:
        """Return per-sample source-length list (for bucketing)."""
        if not hasattr(self, '_cached_source_length'):
            ts = self.target_sr
            if self.repeat_to_max_duration:
                if ts is not None:
                    cap = self._max_duration_for_sr(ts)
                    self._cached_source_length = [cap] * len(self)
                else:
                    self._cached_source_length = [self._max_duration_for_sr(fs) for fs, _ in self._filtered_indices]
            else:
                lengths = []
                for i in range(len(self)):
                    fs, idx = self._get_from_index(i)
                    uid = self.speech_uids[fs][idx]
                    nat = self.source_length[uid]
                    if ts is not None and fs != ts:
                        cap_ts = self._max_duration_for_sr(ts)
                        lengths.append(min(int(round(nat * ts / fs)), cap_ts))
                    else:
                        lengths.append(nat)
                self._cached_source_length = lengths
        return self._cached_source_length

    def get_srs(self) -> list[int]:
        """Return per-sample sample-rate list (cached after first call)."""
        if not hasattr(self, '_cached_srs'):
            if self.target_sr is not None:
                self._cached_srs = [self.target_sr] * len(self)
            else:
                self._cached_srs = [self._get_from_index(i)[0] for i in range(len(self))]
        return self._cached_srs
