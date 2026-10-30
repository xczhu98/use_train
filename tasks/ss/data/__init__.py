"""Speech Separation data contracts and dataset builders."""

from core.data.source_separation_batch import SSBatch
from tasks.ss.data.collate import ss_collate_fn
from tasks.ss.data.offline_dataset import SSOfflineDataset
from tasks.ss.data.dynamic_dataset import SSDynamicMixingDataset
from tasks.ss.data.source_pool import parse_source_pool
from tasks.ss.data.config import SSDatasetConfig, resolve_ss_task_options
from tools.verify.simulation_audit_contract import build_ss_trace_metadata_envelope
from core.data.dataset_build import (
    DatasetBuildSpec,
    OfflineDatasetRequest,
    SamplingCapabilities,
    build_dataset as _build_dataset_impl,
    expand_simulation_cfg_layer_a,
)


def _ss_build_spec(cfg: dict, *, training: bool) -> DatasetBuildSpec:
    """把 SS 专属 source cardinality 注入 shared DatasetBuildSpec。"""
    task_options = resolve_ss_task_options(cfg)
    data = cfg.get("data") if isinstance(cfg.get("data"), dict) else cfg
    configured_sr = data.get("sample_rate")

    def build_dynamic(request):
        kwargs = dict(request.dataset_kwargs)
        duration = kwargs.pop("max_duration_seconds", None)
        explicit_segment = "segment_seconds" in kwargs
        segment_seconds = kwargs.pop("segment_seconds", 3.0)
        if duration is not None:
            # 显式 segment_seconds 是 split 级事实；与 max_duration_seconds 冲突时不得静默改写。
            if explicit_segment and float(segment_seconds) != float(duration):
                raise ValueError(
                    "SS segment_seconds conflicts with simulation max_duration_seconds"
                )
            segment_seconds = duration
        if "max_duration" in kwargs:
            raise ValueError(
                "SS uses segment_seconds/max_duration_seconds, not sample-count max_duration"
            )
        kwargs.update(
            num_sources=task_options.num_sources,
            sample_rate=kwargs.get("target_sr") or configured_sr,
            segment_seconds=segment_seconds,
            sir_range=task_options.sir_range,
            seed=request.seed,
        )
        return SSDynamicMixingDataset(**kwargs)

    def build_offline(request: OfflineDatasetRequest):
        section = dict(request.section)
        for key in (
            "speech_scp",
            "sampling",
            "filter_by_duration",
            "filter_min_duration",
            "filter_max_duration",
            "speech_length_file",
            "target_sr",
            "repeat_to_max_duration",
            "repeat_tile_silence_gap",
            "repeat_tile_silence_gap_range",
        ):
            section.pop(key, None)
        simulation_cfg = section.pop("simulation_cfg", None)
        return SSOfflineDataset(
            **section,
            num_sources=task_options.num_sources,
            target_sr=request.target_sr,
            strict_load=request.strict_load,
            simulation_cfg=simulation_cfg,
            simulation_cfg_strict=request.simulation_cfg_strict,
            placeholder_num_samples=request.placeholder_num_samples,
            seed=request.seed,
            training=training,
        )

    return DatasetBuildSpec(
        task="ss",
        dynamic_dataset_cls=SSDynamicMixingDataset,
        dynamic_dataset_factory=build_dynamic,
        offline_dataset_factory=build_offline,
        simulation_cfg_expanders=(expand_simulation_cfg_layer_a,),
        simulation_cfg_strict_from_cfg=True,
        sampling_capabilities=SamplingCapabilities(
            speech_language=True,
            speech_dataset=True,
            speech_speaker=True,
            noise=True,
        ),
        dynamic_validation_default=True,
    )


def build_dataset(cfg: dict, mode: str = "train"):
    """Build SS through the shared lifecycle/mode/sampling builder."""
    if mode not in {"train", "val"}:
        raise ValueError(f"dataset split must be 'train' or 'val', got {mode!r}")
    return _build_dataset_impl(
        cfg,
        mode=mode,
        spec=_ss_build_spec(cfg, training=mode == "train"),
    )

__all__ = [
    "SSBatch",
    "SSDatasetConfig",
    "SSOfflineDataset",
    "build_dataset",
    "ss_collate_fn",
    "build_ss_trace_metadata_envelope",
]
