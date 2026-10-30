"""AEC 离线（预仿真）数据集。

与 SE 的 ``core.data.offline_dataset.OfflineDataset``（clean/noisy 对）平行，
面向 AEC 预仿真三元组：``mic = nearend + echo(farend)``，监督为 ``nearend``，
可选第四条轨 ``error``（LAEC 线性前端时域残差）。

输入格式（仅 scp 模式，严格四列 ``uid fs length path``）：
  - mic_scp / farend_scp / nearend_scp：必需，uid 集合必须完全一致（fail-fast）。
  - error_scp：可选；给出时 uid 集合同样必须一致，未给出时 error=zeros、
    error_mask=False。

输出（每条样本，供 ``tasks.aec.data.collate.aec_collate_fn`` 的 7 元组输入）：
  ``(nearend, mic, farend, fs, length, error, error_mask)``
  波形均为 channel-first ``(1, T)``；四轨等长 T（farend/error 不足补零、
  超出裁齐），``length = min(len_mic, len_nearend)`` 经裁切后的有效长度。

约定：
  - Dataset 只保证 mic/farend/nearend/error 使用同一起点同步裁切或补齐；这不表示
    已执行部署侧 GCC-PHAT。Trainer 是否再次对齐 farend 由 canonical AEC frontend
    配置决定。
  - ``target_sr`` 给出时全部重采样到该采样率；缺省保持 mic 的采样率，
    farend/nearend/error 采样率不一致时重采样到 mic 的采样率（strict_load
    下直接报错）。
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from core.data.constants import DEFAULT_PLACEHOLDER_NUM_SAMPLES
from core.data.simulation_configs import SimulationConfigs
from core.data.dataset_contract import (
    attach_dataset_contract,
    build_aec_dataset_contract,
    infer_dataset_sample_rate,
)
from core.data.offline_dataset import (
    read_scp_four_col,
    validate_audio_file_metadata,
    validate_loaded_audio_metadata,
)
from core.data.offline_spatial_geometry import (
    resolve_offline_spatial_geometry,
    validate_offline_audio_channels,
)
from core.data.offline_multitrack import (
    BaseOfflineMultiTrackDataset,
    OfflineLifecycleOptions,
    ensure_finite,
    ensure_mono,
    get_io_pool,
    load_audio,
    load_audio_window,
    resample,
    resampled_num_frames,
    slice_or_pad,
)

_log = logging.getLogger(__name__)


class AECOfflineDataset(BaseOfflineMultiTrackDataset):
    """AEC 离线三元组数据集（mic/farend/nearend [+error]）。

    Args:
        mic_scp / farend_scp / nearend_scp:  严格四列 ``uid fs length path`` SCP
        error_scp:                   可选 LAEC 时域残差 SCP
        max_duration:                最大帧数（<=0 = 不限制）；与 max_duration_seconds 二选一
        max_duration_seconds:        单条时长上限（秒），按条按采样率换算
        filter_by_duration:          启用时长过滤（秒，基于 mic 元数据）
        filter_min_duration / filter_max_duration: 过滤区间（秒）
        target_sr:                   目标采样率（None = 保持 mic 原始采样率）
        strict_load:                 True = 读盘/校验失败即抛错；False = fallback 后静音占位
        simulation_cfg:              可选 SimulationConfigs 配置，用于空间字段校验
        simulation_cfg_strict:       True 时未知 simulation_cfg 字段直接报错
        spatial_enabled:              split 级空间读取开关
        array_profile/custom_geometry: 空间 split 的 geometry 来源，C 由 geometry 推导
        reference_channel:            geometry 中的参考通道索引
        channel_order:                磁盘通道排列声明，可使用物理 ID，不执行重排
        placeholder_num_samples:     静音占位长度兜底（max_duration 未设时）
    """

    def __init__(self, mic_scp=None, farend_scp=None, nearend_scp=None,
                 error_scp=None,
                 max_duration=-1,
                 max_duration_seconds: Optional[float] = None,
                 filter_by_duration: bool = False,
                 filter_min_duration: float = 0.5,
                 filter_max_duration: float = 30.0,
                 target_sr=None,
                 strict_load: bool = True,
                 placeholder_num_samples: int = DEFAULT_PLACEHOLDER_NUM_SAMPLES,
                 seed: int = 0,
                 simulation_cfg: dict | None = None,
                 simulation_cfg_strict: bool = True,
                 spatial_enabled: bool = False,
                 array_profile: str | None = None,
                 custom_geometry=None,
                 reference_channel: int = 0,
                 channel_order=None,
                 **kwargs):
        if kwargs:
            raise TypeError(f"AECOfflineDataset got unexpected kwargs: {sorted(kwargs)}")
        if not (mic_scp and farend_scp and nearend_scp):
            raise ValueError(
                "AECOfflineDataset requires mic_scp, farend_scp and nearend_scp"
            )
        if max_duration == 0:
            raise ValueError("max_duration=0 is ambiguous; use -1 (no limit) or >0")
        if target_sr is not None and (type(target_sr) is not int or target_sr <= 0):
            raise ValueError(
                f"target_sr must be a positive integer when provided, got {target_sr!r}"
            )

        if type(simulation_cfg_strict) is not bool:
            raise TypeError("simulation_cfg_strict must be an exact bool")
        self.sim_cfg = SimulationConfigs.from_dict(
            simulation_cfg or {}, strict=simulation_cfg_strict
        )
        self.max_duration_seconds = (
            float(max_duration_seconds) if max_duration_seconds is not None else None
        )
        self.max_duration = -1 if self.max_duration_seconds is not None else max_duration
        if type(spatial_enabled) is not bool:
            raise ValueError("spatial_enabled must be a bool")
        if self.sim_cfg.spatial_enabled:
            raise ValueError(
                "offline AEC spatial_enabled is a split-level reader policy, "
                "not a simulation_cfg field"
            )
        self.spatial_enabled = spatial_enabled
        self.resolved_microphone_array, self.channel_order = resolve_offline_spatial_geometry(
            spatial_enabled=spatial_enabled, array_profile=array_profile,
            custom_geometry=custom_geometry, reference_channel=reference_channel,
            channel_order=channel_order,
        )
        self.target_sr = int(target_sr) if target_sr is not None else None
        self.filter_by_duration = bool(filter_by_duration)
        self.filter_min_duration = float(filter_min_duration)
        self.filter_max_duration = float(filter_max_duration)

        mic_paths, mic_fs, mic_len = read_scp_four_col(mic_scp)
        farend_paths, farend_fs, farend_len = read_scp_four_col(farend_scp)
        nearend_paths, nearend_fs, nearend_len = read_scp_four_col(nearend_scp)
        self._validate_pairing(
            {"mic": mic_paths, "farend": farend_paths, "nearend": nearend_paths}
        )
        error_paths = error_fs = error_len = None
        if error_scp:
            error_paths, error_fs, error_len = read_scp_four_col(error_scp)
            self._validate_pairing({"mic": mic_paths, "error": error_paths})

        self.mic = mic_paths
        self.farend = farend_paths
        self.nearend = nearend_paths
        self.error = error_paths
        self.uids = list(self.mic.keys())
        self._audio_info_cache = {}

        self.utt2fs = mic_fs
        self.lengths = mic_len
        self.farend_utt2fs = farend_fs
        self.farend_lengths = farend_len
        self.nearend_utt2fs = nearend_fs
        self.nearend_lengths = nearend_len
        self.error_utt2fs = error_fs
        self.error_lengths = error_len
        if self.resolved_microphone_array is not None:
            for uid, path in self.mic.items():
                validate_offline_audio_channels(path, self.resolved_microphone_array.channels,
                                                 source=f"AEC uid={uid} mic")

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
                seed=seed,
            ),
        )

    @staticmethod
    def _validate_pairing(named_dicts: dict) -> None:
        """校验多个 scp 的 uid 集合完全一致，不一致即 fail-fast。"""
        names = list(named_dicts.keys())
        ref = set(named_dicts[names[0]].keys())
        errors = []
        for name in names[1:]:
            other = set(named_dicts[name].keys())
            missing = ref - other
            extra = other - ref
            if missing:
                errors.append(
                    f"  - {len(missing)} uids in {names[0]} but missing in {name}: "
                    f"{list(missing)[:10]}{'...' if len(missing) > 10 else ''}"
                )
            if extra:
                errors.append(
                    f"  - {len(extra)} uids in {name} but missing in {names[0]}: "
                    f"{list(extra)[:10]}{'...' if len(extra) > 10 else ''}"
                )
        if errors:
            raise ValueError(
                "AECOfflineDataset SCP pairing check failed:\n" + "\n".join(errors)
            )
        if len(ref) == 0:
            raise ValueError("AECOfflineDataset: no valid uids found in scp sources")

    def _get_audio_info(self, path: str, fallback_sr: int) -> tuple[int, int]:
        """返回并缓存 (frames, samplerate)，不读取波形。"""
        if path not in self._audio_info_cache:
            import soundfile as sf
            try:
                info = sf.info(path)
                metadata = (info.frames, info.samplerate)
            except Exception:
                _log.warning("sf.info failed for %s, using default length", path, exc_info=True)
                metadata = (fallback_sr * 4, fallback_sr)
            self._audio_info_cache[path] = metadata
        return self._audio_info_cache[path]

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

    def _source_length(self, real_index: int) -> int:
        """仅按元数据估计单条输出长度（mic/nearend 取齐 + cap）。"""
        uid = self.uids[real_index]
        mic_len, mic_sr = self.lengths[uid], self.utt2fs[uid]
        near_len = self.nearend_lengths[uid]
        near_sr = self.nearend_utt2fs[uid]
        if near_sr != mic_sr:
            near_len = resampled_num_frames(near_len, near_sr, mic_sr)
        output_sr = self.target_sr or mic_sr
        if self.target_sr is not None and mic_sr != self.target_sr:
            mic_len = resampled_num_frames(mic_len, mic_sr, self.target_sr)
            near_len = resampled_num_frames(near_len, mic_sr, self.target_sr)
        cap = (
            int(self.max_duration_seconds * output_sr)
            if self.max_duration_seconds is not None
            else self.max_duration
        )
        length = min(mic_len, near_len)
        return min(length, cap) if cap > 0 else length

    def _sample_desc(self, real_index: int) -> str:
        return f"uid={self.uids[real_index]}"

    def _sample_description(self, real_index: int) -> str:
        return self._sample_desc(real_index)

    def _placeholder(self, real_index: int):
        """静音占位（strict_load=False 兜底），采样率与失败样本一致。"""
        uid = self.uids[real_index]
        sr = self.target_sr or self.utt2fs.get(uid, 16000)
        if self.max_duration_seconds is not None:
            out_len = int(self.max_duration_seconds * sr)
        elif self.max_duration > 0:
            out_len = self.max_duration
        else:
            out_len = self.placeholder_num_samples
        channels = (
            self.resolved_microphone_array.channels
            if self.resolved_microphone_array is not None else 1
        )
        silence = torch.zeros(channels, out_len, dtype=torch.float32)
        mono_silence = torch.zeros(1, out_len, dtype=torch.float32)
        return (
            mono_silence, silence, mono_silence.clone(), sr, out_len,
            mono_silence.clone(), False,
        )

    def _placeholder_sample(self, real_index: int):
        return self._placeholder(real_index)

    def _load_sample(self, real_index: int):
        return self._getitem_impl(real_index)

    def _load_mono(self, path: str) -> tuple[torch.Tensor, int]:
        wav, sr = load_audio(path)
        return ensure_mono(wav), sr

    def _load_observation(self, path: str) -> tuple[torch.Tensor, int]:
        wav, sr = load_audio(path)
        if wav.ndim != 2 or wav.shape[0] < 1 or wav.shape[1] < 1:
            raise ValueError(f"AEC microphone observation must be [C,T], got {tuple(wav.shape)}")
        if self.resolved_microphone_array is not None and wav.shape[0] != self.resolved_microphone_array.channels:
            raise ValueError(
                f"AEC microphone observation has C={wav.shape[0]}, expected "
                f"geometry C={self.resolved_microphone_array.channels}"
            )
        return wav, sr

    def _getitem_impl(self, index: int):
        uid = self.uids[index]
        paths = [self.mic[uid], self.nearend[uid], self.farend[uid]]
        if self.error is not None:
            paths.append(self.error[uid])

        metadata_specs = [
            ("mic", self.mic[uid], self.utt2fs[uid], self.lengths[uid]),
            (
                "nearend", self.nearend[uid],
                self.nearend_utt2fs[uid], self.nearend_lengths[uid],
            ),
            (
                "farend", self.farend[uid],
                self.farend_utt2fs[uid], self.farend_lengths[uid],
            ),
        ]
        if self.error is not None:
            metadata_specs.append(
                ("error", self.error[uid], self.error_utt2fs[uid], self.error_lengths[uid])
            )

        # 多轨并行读盘（共享进程级 I/O 线程池）。只有 header 可验证为有限值
        # PCM、全轨同 SR、无需重采样且 mic/nearend 必然裁长时才同步 window read。
        ex = get_io_pool()
        crop_start: Optional[int] = None
        windowed = False
        try:
            headers = [
                validate_audio_file_metadata(
                    path,
                    expected_sr=expected_sr,
                    expected_length=expected_length,
                    source=f"uid={uid} track={name} path={path}",
                )
                for name, path, expected_sr, expected_length in metadata_specs
            ]
            sample_rates = {header[1] for header in headers}
            cap = (
                int(self.max_duration_seconds * headers[0][1])
                if self.max_duration_seconds is not None
                else self.max_duration
            )
            # window read 必须对所有轨道成立：mic/nearend/farend/error 任一轨
            # 短于 crop_start+cap 时会在 EOF 处被截断，随后被静默补零。
            paired_length = min(header[0] for header in headers)
            safe_subtypes = {"PCM_S8", "PCM_U8", "PCM_16", "PCM_24", "PCM_32"}
            can_window = (
                cap > 0
                and paired_length > cap
                and len(sample_rates) == 1
                and (self.target_sr is None or self.target_sr == headers[0][1])
                and all(header[2] in safe_subtypes for header in headers)
            )
            if can_window:
                crop_start = self.sample_rng(index).randint(0, paired_length - cap)
                futs = [
                    ex.submit(load_audio_window, path, crop_start, cap)
                    for path in paths
                ]
                tracks = []
                for track_index, future in enumerate(futs):
                    waveform, sample_rate = future.result()
                    tracks.append(
                        (
                            waveform
                            if track_index == 0 and self.spatial_enabled
                            else ensure_mono(waveform),
                            sample_rate,
                        )
                    )
                expected_sr = headers[0][1]
                if any(sample_rate != expected_sr for _, sample_rate in tracks):
                    raise OSError("AEC window read sample rate changed")
                if any(waveform.shape[1] != cap for waveform, _ in tracks):
                    raise OSError("AEC window short read")
                windowed = True
        except Exception as exc:
            _log.warning(
                "AEC offline window read unavailable; falling back to full read: %s", exc
            )

        if not windowed:
            futs = [
                ex.submit(
                    self._load_observation if i == 0 and self.spatial_enabled else self._load_mono,
                    path,
                )
                for i, path in enumerate(paths)
            ]
            tracks = [f.result() for f in futs]
        (mic, fs_mic), (nearend, fs_near), (farend, fs_far) = tracks[0], tracks[1], tracks[2]
        error, fs_err = (tracks[3] if self.error is not None else (None, None))

        if not windowed:
            loaded_metadata = (
                ("mic", mic, fs_mic, self.utt2fs[uid], self.lengths[uid], self.mic[uid]),
                (
                    "nearend", nearend, fs_near,
                    self.nearend_utt2fs[uid], self.nearend_lengths[uid], self.nearend[uid],
                ),
                (
                    "farend", farend, fs_far,
                    self.farend_utt2fs[uid], self.farend_lengths[uid], self.farend[uid],
                ),
            )
            for name, waveform, actual_sr, expected_sr, expected_length, path in loaded_metadata:
                validate_loaded_audio_metadata(
                    waveform,
                    actual_sr,
                    expected_sr=expected_sr,
                    expected_length=expected_length,
                    source=f"uid={uid} track={name} path={path}",
                )
            if error is not None:
                validate_loaded_audio_metadata(
                    error,
                    fs_err,
                    expected_sr=self.error_utt2fs[uid],
                    expected_length=self.error_lengths[uid],
                    source=f"uid={uid} track=error path={self.error[uid]}",
                )

        # 采样率对齐：以 mic 为基准
        fs = fs_mic
        if fs_near != fs:
            if self.strict_load:
                raise ValueError(
                    f"strict_load: mic/nearend sample rate mismatch fs_mic={fs} "
                    f"fs_nearend={fs_near} ({self._sample_desc(index)})"
                )
            nearend = resample(nearend, fs_near, fs)
        if fs_far != fs:
            if self.strict_load:
                raise ValueError(
                    f"strict_load: mic/farend sample rate mismatch fs_mic={fs} "
                    f"fs_farend={fs_far} ({self._sample_desc(index)})"
                )
            farend = resample(farend, fs_far, fs)
        if error is not None and fs_err != fs:
            if self.strict_load:
                raise ValueError(
                    f"strict_load: mic/error sample rate mismatch fs_mic={fs} "
                    f"fs_error={fs_err} ({self._sample_desc(index)})"
                )
            error = resample(error, fs_err, fs)

        if self.target_sr is not None and fs != self.target_sr:
            mic = resample(mic, fs, self.target_sr)
            nearend = resample(nearend, fs, self.target_sr)
            farend = resample(farend, fs, self.target_sr)
            if error is not None:
                error = resample(error, fs, self.target_sr)
            fs = self.target_sr
        if fs <= 0:
            raise ValueError(f"Invalid sample rate: {fs}")

        waveforms = {
            "mic": mic,
            "nearend": nearend,
            "farend": farend,
            **({"error": error} if error is not None else {}),
        }
        for name, wav in waveforms.items():
            waveforms[name] = ensure_finite(
                wav,
                strict=self.strict_load,
                track=name,
                sample=f"index={index} {self._sample_desc(index)}",
            )
        mic = waveforms["mic"]
        nearend = waveforms["nearend"]
        farend = waveforms["farend"]
        if error is not None:
            error = waveforms["error"]

        # 长度对齐：mic/nearend 取公共长度；farend/error 超出裁齐、不足补零
        cap = (
            int(self.max_duration_seconds * fs)
            if self.max_duration_seconds is not None
            else self.max_duration
        )
        len_mic, len_near = mic.shape[1], nearend.shape[1]
        if len_mic == 0 or len_near == 0:
            raise ValueError(f"Zero-length audio at index {index}")

        length = min(len_mic, len_near)
        start = 0
        if cap > 0 and length > cap:
            start = (
                crop_start
                if crop_start is not None
                else self.sample_rng(index).randint(0, length - cap)
            )
            length = cap
        mic = slice_or_pad(mic, start=start, length=length)
        nearend = slice_or_pad(nearend, start=start, length=length)
        farend = slice_or_pad(farend, start=start, length=length)
        if error is not None:
            error = slice_or_pad(error, start=start, length=length)
            error_mask = True
        else:
            error = torch.zeros_like(nearend)
            error_mask = False

        return nearend, mic, farend, fs, length, error, error_mask


def build_offline_aec_dataset(cfg: dict, mode: str = "train") -> AECOfflineDataset:
    """从整包 config（含 data 段）构建 AEC 离线数据集。

    消费 ``data.train`` / ``data.validation`` 段的
    ``mic_scp``/``farend_scp``/``nearend_scp``/``error_scp`` 等键；
    ``data.sample_rate`` 作为 ``target_sr``；``strict_load`` /
    ``placeholder_num_samples`` 与主链路同一套 ``resolve_*`` 解析。
    """
    from core.data.config_flags import resolve_simulation_cfg_strict
    from core.data.dataset_build import (
        apply_duration_filter_config,
        resolve_dataset_build_options,
    )

    options = resolve_dataset_build_options(cfg, mode, expected_task="aec")
    if options.mode.storage != "offline":
        raise ValueError(
            f"data.mode {options.mode.task_mode!r} is not an offline AEC mode"
        )
    section = options.section
    apply_duration_filter_config(section)
    dataset = AECOfflineDataset(
        **section,
        target_sr=options.target_sr,
        strict_load=options.strict_load,
        simulation_cfg_strict=resolve_simulation_cfg_strict(cfg),
        placeholder_num_samples=options.placeholder_num_samples,
    )
    attach_dataset_contract(
        dataset,
        build_aec_dataset_contract(infer_dataset_sample_rate(dataset, cfg)),
    )
    return dataset
