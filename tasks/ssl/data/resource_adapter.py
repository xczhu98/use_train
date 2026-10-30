"""SSL 对 speech-only 共享资源的显式访问边界。"""

from __future__ import annotations

from typing import Any

from core.data.simulation.scene_context import (
    DatasetAcousticRuntimeAdapter,
    SceneRenderContext,
)
from core.data.speech_resource_context import DynamicSpeechResourceContext


class _SSLSpeechPreparationDataset:
    """把 Layer A 需要的少量 Dataset 能力显式绑定到 SSL 与资源 context。"""

    _simulation_task = "ssl"

    def __init__(self, task: Any, resources: DynamicSpeechResourceContext) -> None:
        self._task = task
        self._resources = resources

    @property
    def sim_cfg(self):
        return self._resources.sim_cfg

    @property
    def _simulation_view(self):
        return self._resources._simulation_view

    @property
    def speech_flt(self):
        return self._resources.speech_flt

    @property
    def _all_speech_paths(self):
        return self._resources._all_speech_paths

    @property
    def _all_speech_uids(self):
        return self._resources._all_speech_uids

    @property
    def repeat_to_max_duration(self):
        return self._resources.repeat_to_max_duration

    @property
    def repeat_tile_silence_gap(self):
        return self._resources.repeat_tile_silence_gap

    @property
    def repeat_tile_silence_gap_range(self):
        return self._resources.repeat_tile_silence_gap_range

    @property
    def export_simulation_stems(self):
        return self._resources.export_simulation_stems

    @property
    def _hp_taps(self):
        return self._task.__dict__.get("_hp_taps")

    @_hp_taps.setter
    def _hp_taps(self, value):
        self._task._hp_taps = value

    @property
    def _last_simulation_stems(self):
        return self._task.__dict__.get("_last_simulation_stems")

    @_last_simulation_stems.setter
    def _last_simulation_stems(self, value):
        self._task._last_simulation_stems = value

    @property
    def _fixed_silence_canvas_active(self):
        return self._task.__dict__.get("_fixed_silence_canvas_active", False)

    @_fixed_silence_canvas_active.setter
    def _fixed_silence_canvas_active(self, value):
        self._task._fixed_silence_canvas_active = value

    def _read_one_speech(self, path: str, sample_rate: int):
        return self._resources._read_one_speech(path, sample_rate)

    def _speech_splice_candidate_uids(self, speech_uid: str):
        return self._resources._speech_splice_candidate_uids(speech_uid)

    def _repeat_export_info(
        self, sample_rate: int, samples_before: int, samples_after: int,
        max_duration_samples: int,
    ):
        return self._resources._repeat_export_info(
            sample_rate, samples_before, samples_after, max_duration_samples
        )

    def _speech_splice_export_info(self, cfg, sample_rate: int, samples: int):
        return self._resources._speech_splice_export_info(cfg, sample_rate, samples)


class SSLSpeechResourceAdapter:
    """给 SSL 暴露 speech 清单、RIR provider、缓存生命周期和 scene context。

    少数底层共享能力尚以 ``BaseDynamicMixingDataset`` 字段表示；本类把这些
    字段集中映射成 SSL 的显式接口，不让任务 Dataset 或 renderer 通过
    ``__getattr__`` 猜测资源归属。
    """

    def __init__(
        self,
        resources: DynamicSpeechResourceContext,
        *,
        simulation_split: str,
    ) -> None:
        if not isinstance(resources, DynamicSpeechResourceContext):
            raise TypeError("resources must be DynamicSpeechResourceContext")
        if simulation_split not in ("train", "validation"):
            raise ValueError("simulation_split must be train or validation")
        self._resources = resources
        self._split = simulation_split
        self._scene_context = SceneRenderContext.from_runtime(
            DatasetAcousticRuntimeAdapter(resources)
        )

    def __len__(self) -> int:
        return len(self._resources)

    @property
    def balanced_speech_sampling_enabled(self) -> bool:
        return self._resources.balanced_speech_sampling_enabled is True

    @property
    def strict_load(self) -> bool:
        return self._resources.strict_load is True

    def get_srs(self) -> list[int]:
        """返回共享 speech context 的 sampler 采样率序列。"""

        return self._resources.get_srs()

    def get_source_length(self) -> list[int]:
        """返回与 ``get_srs`` 同索引对齐的 sampler 时长序列。"""

        return self._resources.get_source_length()

    def get_data_summary(self, split: str, n_batches: int, dataloader_config: dict):
        """保留训练入口的可选数据分布摘要能力。"""

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
        """转发共享 sampler 的 balanced speech epoch plan。"""

        return self._resources.get_speech_epoch_plan(
            requested=requested,
            world_size=world_size,
            batch_size=batch_size,
            drop_last=drop_last,
            seed=seed,
            epoch=epoch,
        )

    @property
    def scene_context(self) -> SceneRenderContext:
        return self._scene_context

    @property
    def simulation_config(self):
        return self._resources.sim_cfg

    @property
    def simulation_view(self):
        return self._resources._simulation_view

    @property
    def speech_files(self):
        return self._resources.speech_flt

    @property
    def speech_uid_groups(self):
        return self._resources.speech_uids

    @property
    def spatial_rir_provider(self):
        return self._resources._spatial_rir_provider

    @property
    def has_noise_resource(self) -> bool:
        return bool(self._resources.noise_flt)

    @property
    def has_wind_resource(self) -> bool:
        return bool(self._resources.wn_flt)

    @property
    def _worker_base_seed(self):
        return self._resources._worker_base_seed

    @property
    def _worker_epoch(self):
        return self._resources._worker_epoch

    @property
    def _simulation_split(self) -> str:
        return self._split

    @property
    def simulation_split(self) -> str:
        return self._split

    def set_simulation_split(self, simulation_split: str) -> None:
        if simulation_split not in ("train", "validation"):
            raise ValueError("simulation_split must be train or validation")
        self._split = simulation_split

    def set_epoch(self, epoch: int, base_seed: int) -> None:
        self._resources.set_epoch(epoch, base_seed=base_seed)

    def worker_maybe_clear_caches(self) -> None:
        self._resources._worker_maybe_clear_caches()

    def prepare_clean_speech_result(self, task: Any, *args: Any, **kwargs: Any):
        adapter = _SSLSpeechPreparationDataset(task, self._resources)
        return self._resources.prepare_clean_speech_result(
            adapter, *args, **kwargs
        )


__all__ = ["SSLSpeechResourceAdapter"]
