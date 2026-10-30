"""SSL 静态/移动声源共享仿真数据链。

固定 K 个 speech source，static/moving 都先通过共享 scene sampler 固定房间、
阵列和 actor，再由 SpatialSceneRenderer 渲染。SSL 的 observation 不消费加性
noise；speech manifest、Layer A、时间画布和 RIR provider 通过
``DynamicSpeechResourceContext`` 组合复用。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.data.speech_resource_context import DynamicSpeechResourceContext
from core.data.simulation.array_profiles import resolve_microphone_array
from core.data.simulation.rng import SimulationRNG
from tasks.ssl.data.spatial_adapter import (
    ObservationPostMixRenderer,
    SpatialSceneRenderRequest,
    SpatialSceneRenderer,
    _has_spatial_background_resource,
)
from tasks.ssl.data.resource_adapter import SSLSpeechResourceAdapter
from core.data.speech_scene import scene_rng, place_scene_sources
from tasks.ssl.data.projection import prepare_ssl_scene
from tasks.ssl.data.recipe import (
    SSLSimulationRecipe,
    apply_overlap_policy,
    clean_speech_with_webrtcvad,
    resolve_ssl_recipe,
)


def _classify_ssl_noise_source(noise_scp: str | None) -> str:
    """记录 Noise92 与本地其它噪声池的边界，不把路径当成完整数据证明。"""

    if noise_scp is None:
        return "none"
    path = Path(noise_scp).expanduser()
    if path.name.lower() == "noise92.scp" or "noise92" in {
        part.lower() for part in path.parts
    }:
        return "noise92_scp"
    return "other_noise_scp"


class SSLDynamicMixingDataset(torch.utils.data.Dataset):
    """FN-SSL 风格的固定阵列、固定 source 数动态仿真数据集。

    ``simulation_cfg.spatial_enabled`` 必须为 True，且 RIR backend 只能是
    ``offline`` 或 ``online``。离线 RIR 必须是 ``[K,C,L]``；在线
    backend 支持静态和逐段移动。SSL 的监督始终来自独立 direct RIR，
    ``target_type`` 只控制公共 scene 审计中的 mono target，不替代 SSL direct target。
    SSL 只接受 ``strict_load=True``；不能把缺失或损坏资源静默转换为无效监督。
    ``__getitem__`` 在进入场景构造前同步清理 persistent worker 的 epoch cache。
    """

    _simulation_task = "ssl"

    def __len__(self) -> int:
        """返回 speech-only context 的有效样本数。"""

        return len(self._resources)

    @property
    def balanced_speech_sampling_enabled(self) -> bool:
        """显式暴露 GroupedBatchSampler 所需的 shared sampling 能力。"""

        return self._resources.balanced_speech_sampling_enabled

    @property
    def strict_load(self) -> bool:
        """向训练入口公开 strict-loading 门禁。"""

        return self._resources.strict_load

    def get_srs(self) -> list[int]:
        """返回 sampler 使用的每条 speech 样本采样率。"""

        return self._resources.get_srs()

    def get_source_length(self) -> list[int]:
        """返回 sampler 使用的每条 speech 样本有效长度。"""

        return self._resources.get_source_length()

    def get_data_summary(self, split: str, n_batches: int, dataloader_config: dict):
        """保留训练启动日志使用的 speech/background 数据统计摘要。"""

        return self._resources.get_data_summary(split, n_batches, dataloader_config)

    def get_speech_epoch_plan(
        self,
        *,
        requested: int,
        world_size: int,
        batch_size: int,
        drop_last: bool,
        seed: int,
        epoch: int,
    ):
        """转发显式启用的 balanced speech sampler plan。"""

        return self._resources.get_speech_epoch_plan(
            requested=requested,
            world_size=world_size,
            batch_size=batch_size,
            drop_last=drop_last,
            seed=seed,
            epoch=epoch,
        )

    @property
    def _simulation_split(self) -> str:
        return self._resources.simulation_split

    @_simulation_split.setter
    def _simulation_split(self, value: str) -> None:
        self._resources.set_simulation_split(value)

    def set_epoch(self, epoch: int, base_seed: int = 0) -> None:
        """同步 context 的 epoch token，并清理 SSL 侧 Layer A cache。"""

        self._resources.set_epoch(epoch, base_seed=base_seed)
        self._epoch = int(epoch)
        self._base_seed = int(base_seed)
        self._hp_taps = None

    def _worker_maybe_clear_caches(self) -> None:
        """同步 speech/background/RIR 资源缓存并清理 SSL Layer A 高通缓存。"""

        self._resources.worker_maybe_clear_caches()
        self._hp_taps = None

    def __init__(
        self,
        speech_scp: str,
        *,
        tracks: int,
        sample_rate: int,
        segment_seconds: float = 3.0,
        repeat_to_max_duration: bool = False,
        filter_by_duration: bool = False,
        noise_scp: str | None = None,
        rir_scp: str | None = None,
        simulation_cfg: dict[str, Any] | None = None,
        seed: int = 0,
        strict_load: bool = True,
        simulation_cfg_strict: bool = True,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int | None = None,
        window: str = "hamming",
        center: bool = False,
        input_representation: str = "multichannel_stft_real_imag",
        target_representation: str = "dpipd_far_field",
        exclude_dc: bool = False,
        normalization: str = "none",
        forgetting_norm_sample_length: int = 298,
        sound_speed: float | None = None,
        simulation_profile: str | None = None,
        inactive_target_manifold: str = "zero",
        vad_mode: str = "direct_path",
        output_stride: int = 1,
        coordinate_frame: str = "room_xyz_m",
        phase_sign: float = 1.0,
        input_size: int | None = None,
        freqs: int | None = None,
        array_profile: str | None = None,
        custom_geometry: Any = None,
        max_track: int | None = None,
        reference_channel: int | None = None,
        geometry_provenance: Any = None,
        validation_recipe: Any = None,
        trajectory_points: int = 1,
        max_displacement_m: float = 0.25,
        simulation_split: str = 'train',
        vad_cleaning: str | None = None,
        segment_sampling: str | None = None,
        overlap_policy: str | None = None,
        **kwargs: Any,
    ) -> None:
        if type(tracks) is not int or tracks < 1:
            raise ValueError("SSL tracks must be an exact positive int")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("SSL sample_rate must be an exact positive int")
        if not isinstance(segment_seconds, (int, float)) or isinstance(segment_seconds, bool) or segment_seconds <= 0:
            raise ValueError("SSL segment_seconds must be positive")
        if type(seed) is not int:
            raise ValueError("SSL seed must be an exact integer")
        if type(strict_load) is not bool:
            raise TypeError("SSL strict_load must be an exact bool")
        if not strict_load:
            raise ValueError(
                "SSL dynamic supervision requires strict_load=True; "
                "placeholder supervision is unsupported"
            )
        if "channels" in kwargs:
            raise TypeError(
                "SSLDynamicMixingDataset no longer accepts channels; "
                "microphone count is derived from array_profile/custom_geometry"
            )
        # inactive manifold 在构造期显式校验，避免未知值被静默忽略到 projection 才失败。
        if inactive_target_manifold not in ("zero", "bessel_j0"):
            raise ValueError(
                "SSL inactive_target_manifold must be zero or bessel_j0"
            )
        if vad_mode != "direct_path":
            raise ValueError("SSL vad_mode must be 'direct_path'")
        if type(forgetting_norm_sample_length) is not int or forgetting_norm_sample_length < 1:
            raise ValueError(
                "SSL forgetting_norm_sample_length must be an exact integer >= 1"
            )
        if not isinstance(normalization, str) or normalization not in {
            "none",
            "forgetting_laplace",
            "global_mean",
        }:
            raise ValueError(
                "SSL normalization must be none, forgetting_laplace or global_mean"
            )
        if sound_speed is not None and (
            isinstance(sound_speed, bool)
            or not isinstance(sound_speed, (int, float))
            or not np.isfinite(float(sound_speed))
            or float(sound_speed) <= 0.0
        ):
            raise ValueError("SSL sound_speed must be positive and finite when provided")
        recipe = resolve_ssl_recipe(
            simulation_profile,
            vad_cleaning=vad_cleaning,
            segment_sampling=segment_sampling,
            overlap_policy=overlap_policy,
        )
        if recipe is not None and recipe.profile == "fnssl_official" and not np.isclose(
            float(segment_seconds), recipe.source_duration_seconds, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                f"SSL simulation_profile={recipe.profile!r} requires "
                f"segment_seconds={recipe.source_duration_seconds}; "
                f"got {segment_seconds}"
            )
        if (
            isinstance(phase_sign, bool)
            or not isinstance(phase_sign, (int, float))
            or not np.isfinite(phase_sign)
            or float(phase_sign) not in (-1.0, 1.0)
        ):
            raise ValueError("SSL phase_sign must be -1 or +1")
        sim = dict(simulation_cfg or {})
        sim.setdefault("rir_backend", "offline")
        sim.setdefault("prob_reverberation", 1.0)
        sim.setdefault("spatial_enabled", True)
        if reference_channel is not None:
            sim.setdefault('reference_channel', reference_channel)
        if array_profile is not None:
            sim.setdefault('array_profile', array_profile)
        if custom_geometry is not None:
            sim.setdefault('custom_geometry', custom_geometry)
        if geometry_provenance is not None:
            sim.setdefault('geometry_provenance', geometry_provenance)
        if validation_recipe is not None:
            sim.setdefault('validation_recipe', validation_recipe)
        sim.setdefault("target_type", "early")
        if sim["rir_backend"] not in {"offline", "online"}:
            raise ValueError("SSL static dynamic path supports only offline or online RIR")
        if not sim["spatial_enabled"]:
            raise ValueError("SSL static dynamic path requires simulation_cfg.spatial_enabled=True")
        if sim["rir_backend"] == "offline" and not rir_scp:
            raise ValueError("SSL offline spatial path requires split-level rir_scp")
        if sim.get('prob_moving_rir', 0.0) > 0 and sim['rir_backend'] != 'online':
            raise ValueError('moving RIR requires spatial online provider')
        if max_track is not None and max_track != tracks:
            raise ValueError('max_track must match tracks')
        if type(trajectory_points) is not int or not 1 <= trajectory_points <= 128:
            raise ValueError('trajectory_points must be an exact integer in [1,128]')
        if trajectory_points > 1 and sim['rir_backend'] != 'online':
            raise ValueError('moving SSL requires rir_backend=online')
        if sim.get('prob_moving_rir', 0.) > 0 and trajectory_points < 2:
            raise ValueError('moving RIR requires trajectory_points >= 2')
        if isinstance(max_displacement_m, bool) or not np.isfinite(max_displacement_m) or max_displacement_m <= 0:
            raise ValueError('max_displacement_m must be positive and finite')
        if sim['prob_reverberation'] != 1.0:
            raise ValueError('SSL spatial supervision requires prob_reverberation=1')
        if simulation_split not in ('train', 'validation'):
            raise ValueError('simulation_split must be train or validation')
        self.trajectory_points = trajectory_points
        self.max_displacement_m = max_displacement_m
        self.tracks = tracks
        self.sample_rate = sample_rate
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length or n_fft)
        self.window = str(window).lower()
        self.center = bool(center)
        self.input_representation = str(input_representation)
        self.target_representation = str(target_representation)
        self.exclude_dc = bool(exclude_dc)
        self.normalization = str(normalization)
        self.forgetting_norm_sample_length = int(forgetting_norm_sample_length)
        self.sound_speed = None if sound_speed is None else float(sound_speed)
        self.simulation_profile = (
            None if simulation_profile is None else simulation_profile.strip()
        )
        self.ssl_recipe: SSLSimulationRecipe | None = recipe
        self.ssl_noise_source = _classify_ssl_noise_source(noise_scp)
        self.vad_cleaning = None if recipe is None else recipe.vad_cleaning
        self.segment_sampling = None if recipe is None else recipe.segment_sampling
        self.overlap_policy = None if recipe is None else recipe.overlap_policy
        self.inactive_target_manifold = str(inactive_target_manifold)
        self.vad_mode = vad_mode
        self.output_stride = int(output_stride)
        self.coordinate_frame = str(coordinate_frame)
        self.phase_sign = float(phase_sign)
        self.freqs = int(freqs or (self.n_fft // 2 + 1 - int(self.exclude_dc)))
        # 与共享 CanvasSpec 保持一致：固定画布按秒数乘采样率向下取整。
        self.segment_length = int(float(segment_seconds) * sample_rate)
        self.seed = int(seed)
        resource_context = DynamicSpeechResourceContext(
            speech_scp,
            noise_scp,
            rir_scp,
            simulation_task=self._simulation_task,
            simulation_cfg=sim,
            strict_load=strict_load,
            max_duration_seconds=float(segment_seconds),
            repeat_to_max_duration=repeat_to_max_duration,
            filter_by_duration=filter_by_duration,
            target_sr=sample_rate,
            simulation_cfg_strict=simulation_cfg_strict,
            **kwargs,
        )
        self._resources = SSLSpeechResourceAdapter(
            resource_context,
            simulation_split=simulation_split,
        )
        self._simulation_split = simulation_split
        self._hp_taps: dict[int, np.ndarray] | None = None
        self._resources.set_epoch(0, base_seed=self.seed)
        # context 只负责 shared config/resource 合同；SSL 的 moving budget 与
        # trajectory 门禁仍由任务自身执行，顺序保持在 provider 构造之后。
        self._validate_simulation_cfg(self._resources.simulation_config)
        self._resolved_array = resolve_microphone_array(
            array_profile=self._resources.simulation_config.array_profile,
            custom_geometry=self._resources.simulation_config.custom_geometry,
            reference_channel=self._resources.simulation_config.reference_channel,
        )
        if self._resolved_array is None or self._resolved_array.channels < 2:
            raise ValueError("SSL static dynamic path requires an array with at least two microphones")
        channels = self._resolved_array.channels
        default_input_size = channels * 2 if self.input_representation == "multichannel_stft_real_imag" else (channels - 1) * 2
        self.input_size = int(input_size if input_size is not None else default_input_size)
        if self.input_size != default_input_size:
            raise ValueError(
                "SSL input_size must match input_representation feature dimension: "
                f"expected={default_input_size}, got={self.input_size}"
            )
        self._base_seed = self.seed
        self._spatial_scene_renderer = SpatialSceneRenderer()
        self._observation_post_mix_renderer = ObservationPostMixRenderer()
        self._ssl_recipe_source_facts: list[dict[str, Any]] = []

        # ``chapter_random`` is a source-manifest concern, so keep its derived
        # grouping in the SSL task instead of adding metadata to shared SCP
        # readers.  The fallback key preserves strict fixture manifests whose
        # paths do not follow the LibriSpeech speaker/chapter layout.
        self._speech_chapter_uids: dict[tuple[str, str], list[str]] = {}
        self._speech_chapter_info_by_uid: dict[str, tuple[str, str]] = {}
        if self.segment_sampling == "chapter_random":
            for uid, path in self._resources.speech_files.items():
                parts = Path(path).parts
                # LibriSpeech uses numeric ``speaker/chapter`` parents.  A
                # generic fixture path must remain one independent source so
                # enabling an official profile does not invent a shared chapter.
                if (
                    len(parts) >= 3
                    and parts[-3].isdigit()
                    and parts[-2].isdigit()
                ):
                    speaker, chapter = parts[-3], parts[-2]
                else:
                    # Libri2Mix keeps the original ``speaker-chapter-utt``
                    # identity in the filename while adding ``s1/s2`` parents.
                    stem_tokens = Path(path).stem.split("-")
                    if len(stem_tokens) >= 2 and stem_tokens[0].isdigit() and stem_tokens[1].isdigit():
                        speaker, chapter = stem_tokens[0], stem_tokens[1]
                    else:
                        speaker, chapter = str(uid), str(uid)
                self._speech_chapter_info_by_uid[str(uid)] = (speaker, chapter)
                self._speech_chapter_uids.setdefault((speaker, chapter), []).append(uid)
            self._speech_chapter_uids = {
                key: sorted(values) for key, values in self._speech_chapter_uids.items()
            }
            if len(self._speech_chapter_uids) < self.tracks:
                raise ValueError(
                    "SSL chapter_random sampling requires at least tracks distinct "
                    "speaker/chapter groups"
                )
            self._speech_speaker_chapters: dict[str, list[tuple[str, str]]] = {}
            for speaker, chapter in sorted(self._speech_chapter_uids):
                self._speech_speaker_chapters.setdefault(speaker, []).append(
                    (speaker, chapter)
                )
            if len(self._speech_speaker_chapters) < self.tracks:
                raise ValueError(
                    "SSL chapter_random sampling requires at least tracks distinct speakers"
                )

    def _validate_simulation_cfg(self, cfg) -> None:
        # 公共 parser 校验概率和 backend，任务只消费显式可用的移动分支。
        if cfg.prob_moving_rir > 0 and self.trajectory_points < 2:
            raise ValueError('moving RIR requires trajectory_points >= 2')
        if cfg.rir_backend == "online" and self.tracks > cfg.gpurir_max_sources:
            raise ValueError("SSL tracks exceeds gpurir_max_sources budget")

    def _source_uids(self, index: int) -> tuple[str, ...]:
        # shared source parser 按 native sample rate 分桶；SSL 的 K 抽样使用
        # flat uid 语义，避免把 sample-rate bucket 数误当作 utterance 数。
        speech_uids = self._resources.speech_uid_groups
        if isinstance(speech_uids, dict):
            candidates = [uid for bucket in speech_uids.values() for uid in bucket]
        else:
            candidates = list(speech_uids)
        if self._resources.simulation_config.rir_backend == "offline":
            manifest = self._resources.spatial_rir_provider.scene_manifest
            actor_order = tuple(manifest.actor_order)
            if len(actor_order) != self.tracks or not set(actor_order).issubset(candidates):
                raise ValueError("SSL offline scene actor_order must match tracks and speech_scp identities")
            return actor_order
        if len(candidates) < self.tracks:
            raise ValueError("SSL speech_scp must contain at least tracks distinct utterances")
        rng = SimulationRNG.for_task(
            self._resources, index, "ssl", rng_factory=scene_rng
        ).stream("source_selection")
        if self.segment_sampling == "chapter_random":
            speakers = sorted(self._speech_speaker_chapters)
            selected_indices = rng.choice(len(speakers), size=self.tracks, replace=False)
            selected_uids: list[str] = []
            for speaker_index in selected_indices.tolist():
                speaker = speakers[int(speaker_index)]
                chapter_choices = self._speech_speaker_chapters[speaker]
                chapter = chapter_choices[
                    int(rng.integers(0, len(chapter_choices)))
                ]
                uid_choices = self._speech_chapter_uids[chapter]
                selected_uids.append(
                    str(uid_choices[int(rng.integers(0, len(uid_choices)))])
                )
            return tuple(selected_uids)
        selected = rng.choice(np.asarray(candidates, dtype=object), size=self.tracks, replace=False)
        return tuple(str(uid) for uid in selected.tolist())

    def _prepare_clean_speech_result(
        self,
        speech_uid: str,
        sr: int,
        max_dur_sr: int,
        rng: np.random.Generator,
        *,
        source_trace=None,
        record_layer_a_metadata: bool = True,
    ):
        """复用 shared Layer A 后接入已登记官方 VAD/overlap 语义。"""

        prepared = self._resources.prepare_clean_speech_result(
            self,
            speech_uid,
            sr,
            max_dur_sr,
            rng,
            source_trace=source_trace,
            record_layer_a_metadata=record_layer_a_metadata,
        )
        speech = np.asarray(prepared.speech_1d, dtype=np.float32)
        overlap_facts = {"policy": "none", "split": self._simulation_split, "applied": False}
        vad_facts: dict[str, Any] = {"backend": "disabled"}
        if self.overlap_policy is not None:
            speech, overlap_facts = apply_overlap_policy(
                speech,
                policy=self.overlap_policy,
                split=self._simulation_split,
                tracks=self.tracks,
                sample_rate=sr,
                rng=rng,
            )
        if self.vad_cleaning == "webrtcvad_progressive":
            speech, _, vad_facts = clean_speech_with_webrtcvad(speech, sr)
        elif self.vad_cleaning not in (None, "none"):
            raise ValueError(f"unsupported SSL vad_cleaning {self.vad_cleaning!r}")
        speech = np.asarray(speech, dtype=np.float32)
        if speech.ndim != 1 or speech.shape[0] != prepared.speech_1d.shape[0]:
            raise ValueError("SSL recipe must preserve the shared Layer A waveform canvas")
        if source_trace is not None:
            source_trace.replace_metadata(
                "ssl_recipe",
                {
                    "profile": self.simulation_profile,
                    "version": None if self.ssl_recipe is None else self.ssl_recipe.version,
                    "overlap": overlap_facts,
                    "vad": vad_facts,
                },
            )
        facts = {
            "uid": speech_uid,
            "speaker": (
                self._speech_chapter_info_by_uid.get(str(speech_uid), (None, None))[0]
            ),
            "chapter": (
                self._speech_chapter_info_by_uid.get(str(speech_uid), (None, None))[1]
            ),
            "overlap": overlap_facts,
            "vad": vad_facts,
        }
        self._ssl_recipe_source_facts.append(facts)
        return replace(
            prepared,
            speech_1d=speech,
            speech_2d=speech.reshape(1, -1),
            actual_len=int(speech.shape[0]),
        )

    def _align_moving_source_positions(
        self,
        positions: np.ndarray,
        provenance: dict[str, Any],
        *,
        waveform_length: int,
    ) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim == 2:
            return positions
        if positions.ndim != 3 or positions.shape[1:] != (self.tracks, 3):
            raise ValueError(
                "moving SSL source_positions must have shape [P,K,3] before alignment"
            )
        rir_facts = provenance.get("rir")
        if not isinstance(rir_facts, dict):
            raise ValueError("moving SSL provenance must contain RIR trajectory facts")
        trajectory = np.asarray(rir_facts.get("trajectory"), dtype=np.float32)
        timebase = rir_facts.get("trajectory_timebase")
        if trajectory.shape != positions.shape or not isinstance(timebase, dict):
            raise ValueError("moving SSL provenance trajectory does not match source_positions")
        if not np.allclose(trajectory, positions):
            raise ValueError("moving SSL provenance trajectory differs from source_positions")
        if timebase.get("sample_rate") != self.sample_rate or timebase.get("num_samples") != waveform_length:
            raise ValueError("moving SSL trajectory timebase must match waveform sample rate/length")
        hop = timebase.get("hop_samples")
        frame_count = timebase.get("frame_count")
        start_sample = timebase.get("start_sample", 0)
        if type(hop) is not int or hop < 1:
            raise ValueError("moving SSL trajectory requires a positive hop_samples")
        if (
            type(frame_count) is not int
            or frame_count < 1
            or frame_count != positions.shape[0]
        ):
            raise ValueError(
                "moving SSL trajectory frame_count must be a positive value "
                "matching source_positions"
            )
        if type(start_sample) is not int or start_sample < 0:
            raise ValueError("moving SSL trajectory start_sample must be a non-negative integer")
        if start_sample != 0:
            raise ValueError("moving SSL trajectory start_sample must be zero for waveform alignment")
        trajectory_frame_samples = timebase.get("frame_samples")
        expected_frame_samples = start_sample + np.arange(frame_count, dtype=np.int64) * hop
        if trajectory_frame_samples is not None:
            declared_frame_samples = np.asarray(trajectory_frame_samples)
            if (
                declared_frame_samples.ndim != 1
                or declared_frame_samples.shape[0] != frame_count
                or declared_frame_samples.dtype.kind not in "iu"
                or not np.array_equal(declared_frame_samples, expected_frame_samples)
            ):
                raise ValueError("moving SSL trajectory frame_samples disagree with timebase")
        if expected_frame_samples[-1] >= waveform_length:
            raise ValueError("moving SSL trajectory frame_samples exceed waveform length")
        if self.hop_length < 1:
            raise ValueError("moving SSL STFT requires a positive hop_length")
        if self.center:
            frame_count = waveform_length // self.hop_length + 1
        else:
            if waveform_length < self.n_fft:
                raise ValueError("moving SSL waveform is shorter than n_fft")
            frame_count = 1 + (waveform_length - self.n_fft) // self.hop_length
        # 轨迹 timebase 的 frame_samples 是 RIR 分段起点；center=False 的
        # STFT 标签应落在窗的声学中心，故补半个窗，而不是把轨迹点按帧数
        # 强行拉伸到两端。超出最后轨迹点的尾部沿用末点位置。
        acoustic_samples = np.arange(frame_count, dtype=np.float64) * self.hop_length
        if not self.center:
            acoustic_samples += 0.5 * self.n_fft
        aligned = np.empty((frame_count, self.tracks, 3), dtype=np.float32)
        frame_samples = expected_frame_samples.astype(np.float64)
        for track in range(self.tracks):
            for axis in range(3):
                aligned[:, track, axis] = np.interp(
                    acoustic_samples,
                    frame_samples,
                    trajectory[:, track, axis].astype(np.float64),
                    left=float(trajectory[0, track, axis]),
                    right=float(trajectory[-1, track, axis]),
                ).astype(np.float32)
        return aligned

    def __getitem__(self, index: int) -> dict[str, Any]:
        self._worker_maybe_clear_caches()
        self._ssl_recipe_source_facts = []
        named_rng = SimulationRNG.for_task(
            self._resources, index, "ssl", rng_factory=scene_rng
        )
        source_uids = self._source_uids(index)
        prepared = tuple(
            self._prepare_clean_speech_result(
                uid, self.sample_rate, self.segment_length,
                named_rng.stream("speech_preparation", offset=source_index),
            ).speech_1d
            for source_index, uid in enumerate(source_uids)
        )
        prepared, timeline = place_scene_sources(
            prepared,
            self._resources.simulation_view.shared.temporal,
            named_rng.stream("timeline"),
        )
        provider = self._resources.spatial_rir_provider
        if provider is None:
            raise RuntimeError("SSL static path has no initialized spatial RIR provider")
        simulation_config = self._resources.simulation_config
        moving = named_rng.stream("moving_gate").random() < simulation_config.prob_moving_rir
        request = SpatialSceneRenderRequest(
            context=self._resources.scene_context,
            rng=named_rng,
            background_enabled=_has_spatial_background_resource(
                noise_available=self._resources.has_noise_resource,
                wind_available=self._resources.has_wind_resource,
                config=simulation_config,
            ),
            sources=tuple(np.asarray(source, dtype=np.float32) for source in prepared),
            source_uids=source_uids,
            index=index,
            segment_length=self.segment_length,
            sample_rate=self.sample_rate,
            provider=provider,
            moving=moving,
            trajectory_points=self.trajectory_points,
            max_displacement_m=self.max_displacement_m,
            reference_channel=simulation_config.reference_channel,
            early_boundary=int(round(simulation_config.early_rir_sec * self.sample_rate)),
            timeline=timeline,
        )
        rendered = self._spatial_scene_renderer.render(request)
        finished = self._observation_post_mix_renderer.apply(request, rendered)
        provenance = dict(finished.facts)
        if self.ssl_recipe is not None:
            recipe_provenance = {
                **self.ssl_recipe.to_provenance(),
                "configured_segment_seconds": float(self.segment_length / self.sample_rate),
                "configured_tracks": int(self.tracks),
                "noise_source": self.ssl_noise_source,
                "source_facts": list(self._ssl_recipe_source_facts),
            }
            expected_tracks = self.ssl_recipe.official_source_count
            recipe_provenance["official_data_source_match"] = (
                self.ssl_noise_source == "noise92_scp"
                if "Noise92" in self.ssl_recipe.official_data_domain
                else None
            )
            recipe_provenance["simulation_not_official"] = bool(
                recipe_provenance["simulation_not_official"]
                or (expected_tracks is not None and self.tracks != expected_tracks)
                or recipe_provenance["official_data_source_match"] is False
            )
            provenance["ssl_recipe"] = recipe_provenance
        source_positions = self._align_moving_source_positions(
            finished.source_positions,
            provenance,
            waveform_length=np.asarray(finished.observation).shape[-1],
        )
        return prepare_ssl_scene(
            finished.observation,
            finished.direct_tracks,
            sample_rate=self.sample_rate,
            mic_positions=finished.mic_positions,
            source_positions=source_positions,
            reference_channel=simulation_config.reference_channel,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            expected_channels=self._resolved_array.channels,
            expected_tracks=self.tracks,
            expected_freqs=self.freqs,
            expected_input_size=self.input_size,
            backend=provenance.get("backend", simulation_config.rir_backend),
            provenance=provenance,
            input_representation=self.input_representation,
            target_representation=self.target_representation,
            exclude_dc=self.exclude_dc,
            normalization=self.normalization,
            forgetting_norm_sample_length=self.forgetting_norm_sample_length,
            sound_speed=self.sound_speed,
            simulation_profile=self.simulation_profile,
            inactive_target_manifold=self.inactive_target_manifold,
            vad_mode=self.vad_mode,
            output_stride=self.output_stride,
            coordinate_frame=self.coordinate_frame,
            phase_sign=self.phase_sign,
        )


__all__ = ["SSLDynamicMixingDataset"]
