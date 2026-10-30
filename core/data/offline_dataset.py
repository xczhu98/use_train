"""离线预混语音增强数据集。

SE offline 从严格四列 ``clean_scp`` / ``noisy_scp`` 读取配对信号：
``uid fs length path``。clean 必须为单通道；启用 spatial noisy 时，split 必须显式声明
``spatial_enabled``、``array_profile`` 或 ``custom_geometry``、``reference_channel``
及 ``channel_order``，noisy 按 channel-first ``[C,T]`` 读取，并按几何校验通道数。
Dataset 不执行 Dynamic A–F，不重排通道，也不将 mono noisy 广播为阵列信号。

``target_sr=None`` 时保留原采样率；设置 ``target_sr`` 后在读取时重采样。
当 ``max_duration > 0`` 时，长样本按 clean/noisy 同步裁切。短样本默认原样返回并由 collate
补齐；启用 ``repeat_to_max_duration`` 后，配对信号同步重复到目标长度，并可选择在重复段间
插入静音。重复模式会随机选择 tile 起点以增加样本变化。
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from core.data.constants import DEFAULT_PLACEHOLDER_NUM_SAMPLES
from core.data.offline_spatial_geometry import (
    resolve_offline_spatial_geometry,
    validate_offline_audio_channels,
)
from core.data.offline_multitrack import (
    BaseOfflineMultiTrackDataset,
    OfflineLifecycleOptions,
    ensure_finite,
    ensure_mono,
    get_io_pool as _get_io_pool,
    load_audio as _load_audio,
    load_audio_window,
    resample as _resample,
    resampled_num_frames as _resampled_num_frames,
)
from core.data.resource_paths import resolve_manifest_resource_path
from core.data.simulation_configs import SimulationConfigs
from tools.shared.scp_io import parse_scp_fields

_log = logging.getLogger(__name__)


def _repeat_to_length(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    """Tile audio along time axis to target_len frames.

    Args:
        wav:        (C, T) audio tensor
        target_len: target frame count (>= 1)

    Returns:
        (C, target_len) audio tensor, truncate excess, loop-fill shortage.
    """
    cur_len = wav.shape[1]
    if cur_len == 0:
        return wav.new_zeros(wav.shape[0], target_len)
    if cur_len >= target_len:
        return wav[:, :target_len]
    repeats = (target_len + cur_len - 1) // cur_len  # ceil(target_len / cur_len)
    tiled = wav.repeat(1, repeats)                    # (C, cur_len * repeats)
    return tiled[:, :target_len]


def _tile_to_length_with_gap(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    target_len: int,
    sr: int,
    gap_min_sec: float = 0.02,
    gap_max_sec: float = 0.2,
    *,
    rng: random.Random = random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tile short (clean, noisy) to target_len with random silence gaps between repeats.

    Same semantics as dynamic _align_to_max_duration with insert_silence_gap_between_tiles.
    Clean and noisy tracks may have different channel counts but must share ``[T]``.
    """
    clean_channels, clean_length = clean.shape
    noisy_channels, noisy_length = noisy.shape
    if noisy_length != clean_length:
        raise ValueError(
            f"noisy time length {noisy_length} vs clean time length {clean_length}"
        )
    if clean_length >= target_len:
        return clean[:, :target_len], noisy[:, :target_len]
    gap_min_samp = max(1, int(sr * gap_min_sec))
    gap_max_samp = max(gap_min_samp, int(sr * gap_max_sec))
    gap_len = rng.randint(gap_min_samp, gap_max_samp)
    chunks_c = [clean]
    chunks_n = [noisy]
    total = clean_length
    while total < target_len:
        gap_c = clean.new_zeros(clean_channels, gap_len)
        gap_n = noisy.new_zeros(noisy_channels, gap_len)
        chunks_c.append(gap_c)
        chunks_n.append(gap_n)
        chunks_c.append(clean)
        chunks_n.append(noisy)
        total += gap_len + clean_length
    clean_tiled = torch.cat(chunks_c, dim=1)
    noisy_tiled = torch.cat(chunks_n, dim=1)
    start = rng.randint(0, total - target_len) if total > target_len else 0
    return (
        clean_tiled[:, start : start + target_len].clone(),
        noisy_tiled[:, start : start + target_len].clone(),
    )


def _positive_metadata(value, *, field: str, source: str) -> int:
    """把 manifest metadata 规范化为正整数，并拒绝 bool/浮点截断。"""
    if isinstance(value, (bool, np.bool_, float, np.floating)):
        raise ValueError(f"{source}: {field} 必须是正整数，得到 {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {field} 必须是正整数，得到 {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{source}: {field} 必须是正整数，得到 {value!r}")
    return parsed


def validate_loaded_audio_metadata(
    waveform: torch.Tensor,
    actual_sr: int,
    *,
    expected_sr: int,
    expected_length: int,
    source: str,
) -> None:
    """把四列/JSON 声明绑定到实际解码 header，阻止 stale metadata。"""
    actual_length = int(waveform.shape[-1])
    mismatches = []
    if int(actual_sr) != int(expected_sr):
        mismatches.append(f"fs declared={expected_sr} actual={actual_sr}")
    if actual_length != int(expected_length):
        mismatches.append(
            f"length declared={expected_length} actual={actual_length}"
        )
    if mismatches:
        raise ValueError(f"{source}: manifest metadata mismatch: {', '.join(mismatches)}")


def validate_audio_file_metadata(
    path: str,
    *,
    expected_sr: int,
    expected_length: Optional[int],
    source: str,
) -> tuple[int, int, str]:
    """不解码波形地校验 header，返回帧数、采样率与 subtype。"""
    info = sf.info(path)
    actual_length, actual_sr = int(info.frames), int(info.samplerate)
    mismatches = []
    if actual_sr != int(expected_sr):
        mismatches.append(f"fs declared={expected_sr} actual={actual_sr}")
    if expected_length is not None and actual_length != int(expected_length):
        mismatches.append(
            f"length declared={expected_length} actual={actual_length}"
        )
    if mismatches:
        raise ValueError(f"{source}: manifest metadata mismatch: {', '.join(mismatches)}")
    return actual_length, actual_sr, str(info.subtype)


def _read_scp_four_col(path):
    """严格读取 ``uid fs length path`` SCP；相对路径锚定 SCP parent。"""
    uid2path = {}
    uid2fs = {}
    uid2len = {}
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            uid, fs, length, audio_path = parse_scp_fields(
                line, path, lineno, allowed_columns=(4,)
            )
            if uid in uid2path:
                raise ValueError(f"{path}:{lineno}: 重复 uid: {uid!r}")
            source = f"{path}:{lineno}"
            uid2path[uid] = resolve_manifest_resource_path(audio_path, path)
            uid2fs[uid] = _positive_metadata(fs, field="fs", source=source)
            uid2len[uid] = _positive_metadata(length, field="length", source=source)
    if not uid2path:
        raise ValueError(f"{path}: 四列 SCP 中没有有效条目")
    return uid2path, uid2fs, uid2len


class OfflineDataset(BaseOfflineMultiTrackDataset):
    """Offline pre-mixed speech enhancement dataset.

    Args:
        clean_scp / noisy_scp:       成对严格四列 ``uid fs length path`` SCP
        max_duration:                max frames (<=0 = no limit); 与 max_duration_seconds 二选一
        max_duration_seconds:        单条时长上限（秒），按条按采样率换算，支持 target_sr
        repeat_to_max_duration:      short audio handling
            False (default): return short as-is, collate_fn pads
            True: tile short to max_duration, fixed length, no batch padding, better GPU
        repeat_tile_silence_gap:     when True and repeat_to_max_duration=True, insert 20–200ms
            silence between tile repeats (default False)
        repeat_tile_silence_gap_range: [min_sec, max_sec] for gap (default [0.02, 0.2])
        target_sr:                   target sample rate (None = keep original)
        simulation_cfg:              optional SimulationConfigs mapping for strict validation
        simulation_cfg_strict:       reject unknown simulation_cfg keys when True (default)
    """

    def __init__(
        self,
        clean_scp=None,
        noisy_scp=None,
        max_duration=-1,
        max_duration_seconds: Optional[float] = None,
        repeat_to_max_duration: bool = False,
        repeat_tile_silence_gap: bool = False,
        repeat_tile_silence_gap_range: Optional[list] = None,
        target_sr=None,
        filter_by_duration: bool = False,
        filter_min_duration: float = 0.5,
        filter_max_duration: float = 30.0,
        strict_load: bool = True,
        simulation_cfg: dict | None = None,
        simulation_cfg_strict: bool = True,
        placeholder_num_samples: int = DEFAULT_PLACEHOLDER_NUM_SAMPLES,
        seed: int = 0,
        spatial_enabled: bool = False,
        array_profile: str | None = None,
        custom_geometry: dict | None = None,
        reference_channel: int = 0,
        channel_order: list | tuple | None = None,
    ):
        if max_duration == 0:
            raise ValueError("max_duration=0 is ambiguous; use -1 (no limit) or >0")
        if target_sr is not None and (type(target_sr) is not int or target_sr <= 0):
            raise ValueError(
                f"target_sr must be a positive integer when provided, got {target_sr!r}"
            )
        if type(simulation_cfg_strict) is not bool:
            raise TypeError("simulation_cfg_strict must be an exact bool")
        if type(spatial_enabled) is not bool:
            raise TypeError("spatial_enabled must be an exact bool")
        simulation_values = dict(simulation_cfg or {})
        split_geometry = {
            "spatial_enabled": spatial_enabled,
            "array_profile": array_profile,
            "custom_geometry": custom_geometry,
            "reference_channel": reference_channel,
        }
        for key, value in split_geometry.items():
            if key in simulation_values:
                if simulation_values[key] != value:
                    raise ValueError(
                        f"offline {key} must be declared at the data split and must not "
                        "conflict with simulation_cfg"
                    )
                simulation_values.pop(key)
        simulation_values.update(split_geometry)
        self.sim_cfg = SimulationConfigs.from_dict(
            simulation_values, strict=simulation_cfg_strict
        )
        self.spatial_enabled = spatial_enabled
        self.resolved_microphone_array, self.channel_order = resolve_offline_spatial_geometry(
            spatial_enabled=spatial_enabled,
            array_profile=array_profile,
            custom_geometry=custom_geometry,
            reference_channel=reference_channel,
            channel_order=channel_order,
        )
        self.num_channels = (
            self.resolved_microphone_array.channels
            if self.resolved_microphone_array is not None else 1
        )
        self.max_duration_seconds = float(max_duration_seconds) if max_duration_seconds is not None else None
        if self.max_duration_seconds is not None:
            self.max_duration = -1  # 按条按采样率在 __getitem__ / get_source_length 中换算
        else:
            self.max_duration = max_duration
        self.repeat_to_max_duration = bool(repeat_to_max_duration)
        self.repeat_tile_silence_gap = bool(repeat_tile_silence_gap)
        _gap_range = repeat_tile_silence_gap_range
        self.repeat_tile_silence_gap_range = (
            (float(_gap_range[0]), float(_gap_range[1]))
            if _gap_range and len(_gap_range) >= 2
            else (0.02, 0.2)
        )
        self.target_sr = int(target_sr) if target_sr is not None else None
        self._audio_info_cache = {}  # path -> (frames, samplerate)，仅缓存 sf.info 元数据
        self.filter_by_duration = bool(filter_by_duration)
        self.filter_min_duration = float(filter_min_duration)
        self.filter_max_duration = float(filter_max_duration)
        if not clean_scp or not noisy_scp:
            raise ValueError(
                "SE offline requires paired clean_scp and noisy_scp in strict "
                "four-column uid fs length path format"
            )
        self._init_scp(clean_scp, noisy_scp)
        if self.spatial_enabled:
            for uid in self.uids:
                validate_offline_audio_channels(
                    self.clean[uid], 1, source=f"uid={uid} track=clean"
                )
                validate_offline_audio_channels(
                    self.noisy[uid], self.num_channels, source=f"uid={uid} track=noisy"
                )

        # Full length (before duration filter) for internal index bounds
        self._full_len = len(self.uids)

        # Build filtered index list
        if self.filter_by_duration:
            self._filtered_indices = self._build_filtered_indices()
            if len(self._filtered_indices) == 0:
                raise ValueError(
                    f"filter_by_duration=True: no samples in range "
                    f"[{self.filter_min_duration}s, {self.filter_max_duration}s]. "
                    f"Check filter_min_duration / filter_max_duration or disable filter."
                )
            _log.info(
                f"Duration filter: kept {len(self._filtered_indices)} of {self._full_len} samples "
                f"(filter_min_duration={self.filter_min_duration}, filter_max_duration={self.filter_max_duration})"
            )
        else:
            self._filtered_indices = list(range(self._full_len))
        super().__init__(
            full_length=self._full_len,
            filtered_indices=self._filtered_indices,
            lifecycle=OfflineLifecycleOptions(
                strict_load=strict_load,
                placeholder_num_samples=placeholder_num_samples,
                seed=seed,
            ),
        )

    def _init_scp(self, clean_scp, noisy_scp):
        clean_paths, clean_fs, clean_lengths = _read_scp_four_col(clean_scp)
        noisy_paths, noisy_fs, noisy_lengths = _read_scp_four_col(noisy_scp)
        self._init_metadata_pair(
            clean_paths, clean_fs, clean_lengths,
            noisy_paths, noisy_fs, noisy_lengths,
            mode_name="SCP",
        )

    def _init_metadata_pair(
        self,
        clean_paths,
        clean_fs,
        clean_lengths,
        noisy_paths,
        noisy_fs,
        noisy_lengths,
        *,
        mode_name,
    ):
        """保存 clean/noisy 两轨独立 metadata，供采样与长度估算复用。"""
        self._validate_pairing(clean_paths, noisy_paths, mode_name)
        self.clean = clean_paths
        self.noisy = noisy_paths
        self.uids = list(self.clean.keys())
        self.utt2fs = clean_fs
        self.lengths = clean_lengths
        self.noisy_utt2fs = noisy_fs
        self.noisy_lengths = noisy_lengths
        self._mode = "scp"

    def _validate_pairing(self, clean_dict, noisy_dict, mode_name):
        """Validate that clean and noisy have identical uid sets.

        Fail-fast: raise ValueError immediately if pairing is incomplete.
        """
        clean_uids = set(clean_dict.keys())
        noisy_uids = set(noisy_dict.keys())

        missing_in_noisy = clean_uids - noisy_uids
        missing_in_clean = noisy_uids - clean_uids

        errors = []
        if missing_in_noisy:
            errors.append(f"  - {len(missing_in_noisy)} uids in clean but missing in noisy: "
                         f"{list(missing_in_noisy)[:10]}{'...' if len(missing_in_noisy) > 10 else ''}")
        if missing_in_clean:
            errors.append(f"  - {len(missing_in_clean)} uids in noisy but missing in clean: "
                         f"{list(missing_in_clean)[:10]}{'...' if len(missing_in_clean) > 10 else ''}")

        if errors:
            raise ValueError(
                f"OfflineDataset {mode_name} pairing check failed:\n" + "\n".join(errors) +
                f"\nTotal: clean={len(clean_uids)}, noisy={len(noisy_uids)}, matched={len(clean_uids & noisy_uids)}"
            )

        if len(clean_uids) == 0:
            raise ValueError(f"OfflineDataset {mode_name}: no valid uids found in clean/noisy sources")

    def _build_filtered_indices(self):
        """Return list of indices that pass filter_min_duration <= length(sec) <= filter_max_duration."""
        out = []
        for i in range(self._full_len):
            uid = self.uids[i]
            sr = self.utt2fs[uid]
            length = self.lengths[uid]
            min_len = int(self.filter_min_duration * sr)
            max_len = int(self.filter_max_duration * sr)
            if min_len <= length <= max_len:
                out.append(i)
        return out

    def _effective_sr(self, real_index: int) -> int:
        """返回样本进入 collate 时所属的输出采样率 bucket。"""
        if self.target_sr is not None:
            return self.target_sr
        return self.utt2fs[self.uids[real_index]]

    def _placeholder_length(self, sr: int) -> int:
        """Return length for silence placeholder (align with Dynamic: use max_duration when set)."""
        if self.max_duration_seconds is not None:
            return int(self.max_duration_seconds * sr)
        if self.max_duration > 0:
            return self.max_duration
        return self.placeholder_num_samples

    def _placeholder(self, real_index: int):
        """返回与原失败样本 output SR bucket 一致的静音占位。"""
        sr = self._effective_sr(real_index)
        out_len = self._placeholder_length(sr)
        clean = torch.zeros(1, out_len, dtype=torch.float32)
        noisy = torch.zeros(self.num_channels, out_len, dtype=torch.float32)
        return clean, noisy, sr, out_len

    def _output_sr(self, real_index: int) -> int:
        return self._effective_sr(real_index)

    def _source_length(self, real_index: int) -> int:
        """按 metadata 计算单条 paired output-domain 长度。"""
        uid = self.uids[real_index]
        clean_length, clean_sr = self.lengths[uid], self.utt2fs[uid]
        noisy_length = self.noisy_lengths[uid]
        noisy_sr = self.noisy_utt2fs[uid]
        if noisy_sr != clean_sr:
            noisy_length = _resampled_num_frames(noisy_length, noisy_sr, clean_sr)
        output_sr = self.target_sr or clean_sr
        if self.target_sr is not None and clean_sr != self.target_sr:
            clean_length = _resampled_num_frames(
                clean_length, clean_sr, self.target_sr
            )
            noisy_length = _resampled_num_frames(
                noisy_length, clean_sr, self.target_sr
            )
        cap = (
            int(self.max_duration_seconds * output_sr)
            if self.max_duration_seconds is not None
            else self.max_duration
        )
        if cap > 0 and clean_length > cap:
            clean_length = noisy_length = cap
        output_length = min(clean_length, noisy_length)
        if cap > 0 and output_length < cap and self.repeat_to_max_duration:
            output_length = cap
        return output_length

    def _sample_desc(self, real_index: int) -> str:
        """Return a stable UID description for logging."""
        return f"uid={self.uids[real_index]}"

    def _sample_description(self, real_index: int) -> str:
        return self._sample_desc(real_index)

    def _placeholder_sample(self, real_index: int):
        return self._placeholder(real_index)

    def _load_sample(self, real_index: int):
        return self._getitem_impl(real_index)

    def _getitem_impl(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if index < 0 or index >= self._full_len:
            raise IndexError(f"index out of range: {index}, _full_len={self._full_len}")
        # 成对读盘：共享线程池并行 I/O（HDD/网络盘收益更明显）
        uid = self.uids[index]
        p_clean, p_noisy = self.clean[uid], self.noisy[uid]

        rng = self.sample_rng(index)
        crop_start: Optional[int] = None
        windowed = False
        ex = _get_io_pool()

        try:
            expected = (
                ("clean", p_clean, self.utt2fs[uid], self.lengths[uid]),
                ("noisy", p_noisy, self.noisy_utt2fs[uid], self.noisy_lengths[uid]),
            )
            headers = [
                validate_audio_file_metadata(
                    path,
                    expected_sr=expected_sr,
                    expected_length=expected_length,
                    source=f"{self._sample_desc(index)} track={track} path={path}",
                )
                for track, path, expected_sr, expected_length in expected
            ]
            clean_len, clean_sr, clean_subtype = headers[0]
            noisy_len, noisy_sr, noisy_subtype = headers[1]
            cap = (
                int(self.max_duration_seconds * clean_sr)
                if self.max_duration_seconds is not None
                else self.max_duration
            )
            safe_subtypes = {"PCM_S8", "PCM_U8", "PCM_16", "PCM_24", "PCM_32"}
            can_window = (
                cap > 0
                and clean_len > cap
                and noisy_len > 0
                and clean_sr == noisy_sr
                and (self.target_sr is None or self.target_sr == clean_sr)
                and clean_subtype in safe_subtypes
                and noisy_subtype in safe_subtypes
            )
            if can_window:
                crop_start = (
                    rng.randint(0, min(clean_len - cap, noisy_len - cap))
                    if noisy_len >= cap
                    else 0
                )
                f1 = ex.submit(load_audio_window, p_clean, crop_start, cap)
                f2 = ex.submit(load_audio_window, p_noisy, crop_start, cap)
                clean, fs = f1.result()
                noisy, fs_n = f2.result()
                if fs != clean_sr or fs_n != noisy_sr:
                    raise OSError(
                        f"window read sample rate changed: clean={fs} noisy={fs_n}"
                    )
                if clean.shape[1] != cap:
                    raise OSError(
                        f"clean window short read: expected={cap} actual={clean.shape[1]}"
                    )
                if noisy.shape[1] < cap:
                    noisy = torch.nn.functional.pad(
                        noisy, (0, cap - noisy.shape[1]), value=0.0
                    )
                windowed = True
        except Exception:
            _log.warning(
                "offline window read unavailable; falling back to full read",
                exc_info=True,
            )

        if not windowed:
            f1 = ex.submit(_load_audio, p_clean)
            f2 = ex.submit(_load_audio, p_noisy)
            clean, fs = f1.result()
            noisy, fs_n = f2.result()

            validate_loaded_audio_metadata(
                clean,
                fs,
                expected_sr=self.utt2fs[uid],
                expected_length=self.lengths[uid],
                source=f"uid={uid} track=clean path={p_clean}",
            )
            validate_loaded_audio_metadata(
                noisy,
                fs_n,
                expected_sr=self.noisy_utt2fs[uid],
                expected_length=self.noisy_lengths[uid],
                source=f"uid={uid} track=noisy path={p_noisy}",
            )

        if fs_n != fs:
            if self.strict_load:
                raise ValueError(
                    f"strict_load: clean/noisy sample rate mismatch fs_clean={fs} fs_noisy={fs_n} "
                    f"({self._sample_desc(index)})"
                )
            _log.warning(
                "clean/noisy SR differ (%s vs %s), resampling noisy to match clean (%s)",
                fs, fs_n, self._sample_desc(index),
            )
            noisy = _resample(noisy, fs_n, fs)

        clean = ensure_mono(clean)
        if self.spatial_enabled:
            if noisy.ndim != 2 or noisy.shape[0] != self.num_channels:
                raise ValueError(
                    f"spatial noisy must have channel-first shape [{self.num_channels},T], "
                    f"got {tuple(noisy.shape)}"
                )
        else:
            noisy = ensure_mono(noisy)

        if self.target_sr is not None and fs != self.target_sr:
            clean = _resample(clean, fs, self.target_sr)
            noisy = _resample(noisy, fs, self.target_sr)
            fs = self.target_sr
        if fs <= 0:
            raise ValueError(f"Invalid sample rate: {fs}")

        # 以下裁长/齐长：offline 为预混对，必须对 clean 与 noisy 做完全一致的操作（同起点裁、同 tile 同 start）才能对齐
        sample = self._sample_desc(index)
        clean = ensure_finite(
            clean, strict=self.strict_load, track="clean", sample=sample
        )
        noisy = ensure_finite(
            noisy, strict=self.strict_load, track="noisy", sample=sample
        )

        # 与 urgent2026 一致：先按干净语音裁到 cap（offline 无仿真，裁已加载的 clean/noisy 同段）
        # cap 支持动态采样率：max_duration_seconds 时按本条 fs 换算，否则用 max_duration（采样点）
        cap = int(self.max_duration_seconds * fs) if self.max_duration_seconds is not None else self.max_duration
        len_clean = clean.shape[1]
        len_noisy = noisy.shape[1]
        if len_clean == 0 or len_noisy == 0:
            raise ValueError(f"Zero-length audio at index {index}")
        if cap > 0 and len_clean > cap:
            if crop_start is not None:
                s = crop_start
            elif len_noisy >= cap:
                max_s = min(len_clean - cap, len_noisy - cap)
                s = rng.randint(0, max_s)
            else:
                s = 0
            clean = clean[:, s:s + cap]
            end_noisy = min(s + cap, len_noisy)
            noisy = noisy[:, s:end_noisy]
            if noisy.shape[1] < cap:
                noisy = torch.nn.functional.pad(
                    noisy, (0, cap - noisy.shape[1]), value=0.0
                )

        min_len = min(clean.shape[1], noisy.shape[1])
        if min_len == 0:
            raise ValueError(f"Zero-length audio at index {index}")
        clean = clean[:, :min_len]
        noisy = noisy[:, :min_len]

        if cap > 0:
            if min_len < cap and self.repeat_to_max_duration:
                if self.repeat_tile_silence_gap:
                    clean, noisy = _tile_to_length_with_gap(
                        clean, noisy, cap, fs,
                        gap_min_sec=self.repeat_tile_silence_gap_range[0],
                        gap_max_sec=self.repeat_tile_silence_gap_range[1],
                        rng=rng,
                    )
                else:
                    pad_len = cap + min_len
                    clean_tiled = _repeat_to_length(clean, pad_len)
                    noisy_tiled = _repeat_to_length(noisy, pad_len)
                    # s in [0, pad_len - cap] = [0, min_len] to take a contiguous segment of length cap
                    s = rng.randint(0, min_len) if min_len > 0 else 0
                    clean = clean_tiled[:, s:s + cap]
                    noisy = noisy_tiled[:, s:s + cap]

        return clean, noisy, fs, clean.shape[1]


# Public aliases for cross-task offline dataset construction.
get_io_pool = _get_io_pool
load_audio = _load_audio
resample = _resample
resampled_num_frames = _resampled_num_frames
read_scp_four_col = _read_scp_four_col
