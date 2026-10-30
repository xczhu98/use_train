"""Dataset 输出能力的共享合同。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.models.io_contract import TensorField


def infer_dataset_sample_rate(dataset, config: dict | None = None) -> int | None:
    """从 Dataset 或配置提取 canonical output sample rate。"""
    for name in ("sample_rate", "target_sr"):
        value = getattr(dataset, name, None)
        if type(value) is int and value > 0:
            return value
    data = config.get("data", config) if isinstance(config, dict) else {}
    value = data.get("sample_rate") if isinstance(data, dict) else None
    return value if type(value) is int and value > 0 else None


def attach_dataset_contract(dataset, contract: DatasetContract):
    """在可变 Dataset 上挂载合同；保留不可变测试替身的原样返回。"""
    try:
        setattr(dataset, "contract", contract)
    except (AttributeError, TypeError):
        return dataset
    return dataset


@dataclass(frozen=True)
class DatasetContract:
    task: str
    fields: tuple[TensorField, ...]
    sample_rate: int | None
    cardinality: Literal["single", "fixed_multi", "variable_multi"]
    num_sources: int | None = None
    spatial_observation_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.task not in {"se", "aec", "tse", "ss"}:
            raise ValueError(f"unsupported DatasetContract task: {self.task!r}")
        if not self.fields or len({field.name for field in self.fields}) != len(self.fields):
            raise ValueError("DatasetContract fields must be non-empty and uniquely named")
        if self.sample_rate is not None and (
            type(self.sample_rate) is not int or self.sample_rate <= 0
        ):
            raise ValueError("DatasetContract.sample_rate must be a positive int or None")
        if self.cardinality not in {"single", "fixed_multi", "variable_multi"}:
            raise ValueError(f"unsupported Dataset cardinality: {self.cardinality!r}")
        if self.cardinality == "fixed_multi" and (
            type(self.num_sources) is not int or self.num_sources < 1
        ):
            raise ValueError("fixed_multi DatasetContract requires num_sources")
        spatial_fields = tuple(self.spatial_observation_fields)
        if len(set(spatial_fields)) != len(spatial_fields):
            raise ValueError("spatial_observation_fields must be unique")
        field_by_name = {field.name: field for field in self.fields}
        unknown = tuple(name for name in spatial_fields if name not in field_by_name)
        if unknown:
            raise ValueError(
                f"spatial_observation_fields contain unknown fields: {unknown}"
            )
        for name in spatial_fields:
            field = field_by_name[name]
            if field.domain != "waveform" or field.axes[0] != "batch" or field.axes[-1] != "time":
                raise ValueError(
                    f"spatial observation field {name!r} must use batch/time waveform axes"
                )
            if "microphone" in field.axes:
                raise ValueError(
                    f"spatial observation field {name!r} already contains microphone axis"
                )
        object.__setattr__(self, "spatial_observation_fields", spatial_fields)

    def field_axes(self, name: str, *, spatial: bool = False) -> tuple[str, ...]:
        """Return canonical axes, optionally inserting an explicit microphone axis."""
        if type(spatial) is not bool:
            raise TypeError("spatial must be an exact bool")
        try:
            field = next(field for field in self.fields if field.name == name)
        except StopIteration as exc:
            raise KeyError(f"unknown DatasetContract field: {name!r}") from exc
        if spatial and name in self.spatial_observation_fields:
            return (field.axes[0], "microphone", *field.axes[1:])
        return field.axes


def build_se_dataset_contract(sample_rate: int | None = None) -> DatasetContract:
    """返回 SE canonical collated batch 的 DatasetContract。"""
    return DatasetContract(
        task="se",
        fields=(
            TensorField("clean", "waveform", ("batch", "time")),
            TensorField("noisy", "waveform", ("batch", "time")),
            TensorField("sr", "label", ("scalar",)),
            TensorField("lengths", "label", ("batch",)),
            TensorField("is_pure_noise", "mask", ("batch",), required=False),
            TensorField("is_pure_echo", "mask", ("batch",), required=False),
            TensorField("is_pure_echo_noise", "mask", ("batch",), required=False),
        ),
        sample_rate=sample_rate,
        cardinality="single",
        spatial_observation_fields=("noisy",),
    )


def build_aec_dataset_contract(sample_rate: int | None = None) -> DatasetContract:
    """返回 AEC canonical collated batch 的 DatasetContract。"""
    return DatasetContract(
        task="aec",
        fields=(
            TensorField("nearend", "waveform", ("batch", "time")),
            TensorField("mic", "waveform", ("batch", "time")),
            TensorField("farend", "waveform", ("batch", "time")),
            TensorField("fs", "label", ("scalar",)),
            TensorField("lengths", "label", ("batch",)),
            TensorField("error", "waveform", ("batch", "time"), required=False),
            TensorField("error_mask", "mask", ("batch",), required=False),
            TensorField("is_pure_noise", "mask", ("batch",), required=False),
            TensorField("is_pure_echo", "mask", ("batch",), required=False),
            TensorField("is_pure_echo_noise", "mask", ("batch",), required=False),
            TensorField("has_echo", "mask", ("batch",), required=False),
            TensorField("state", "label", ("batch", "frame"), required=False),
        ),
        sample_rate=sample_rate,
        cardinality="single",
        spatial_observation_fields=("mic",),
    )


def build_tse_dataset_contract(
    sample_rate: int | None = None,
    *,
    variable_targets: bool = False,
    cue_specs: object | None = None,
) -> DatasetContract:
    """返回 validated ``TSEBatch`` 及实际启用 typed cue 的张量合同。

    mixture 始终保留 microphone 轴，mono 时 C=1；geometry/timebase/provenance
    是按 mixture 保存的 host metadata，不冒充 TensorField。
    """
    fields = [
        TensorField("mixture", "waveform", ("batch", "microphone", "time")),
        TensorField("target", "waveform", ("target", "time"), multiple=True),
        TensorField("sr", "label", ("scalar",)),
        TensorField("lengths", "label", ("batch",)),
        TensorField("num_targets", "label", ("batch",)),
        TensorField("spk_labels", "label", ("target",), multiple=True),
        TensorField("target_present", "mask", ("target",), multiple=True),
    ]
    if cue_specs is not None:
        speaker_waveform = getattr(cue_specs, "speaker_waveform", None)
        speaker_embedding = getattr(cue_specs, "speaker_embedding", None)
        spatial = getattr(cue_specs, "spatial", None)
        visual = getattr(cue_specs, "visual", None)
        textual = getattr(cue_specs, "textual", None)
        if speaker_waveform is not None:
            fields.extend(
                (
                    TensorField(
                        "speaker_waveform",
                        "waveform",
                        ("target", "cue_time"),
                        multiple=True,
                    ),
                    TensorField(
                        "speaker_waveform_lengths", "label", ("target",), multiple=True
                    ),
                    TensorField(
                        "speaker_waveform_present", "mask", ("target",), multiple=True
                    ),
                )
            )
        if speaker_embedding is not None:
            fields.extend(
                (
                    TensorField(
                        "speaker_embedding",
                        "embedding",
                        ("target", "embedding"),
                        multiple=True,
                    ),
                    TensorField(
                        "speaker_embedding_present", "mask", ("target",), multiple=True
                    ),
                )
            )
        if spatial is not None:
            representation = getattr(spatial, "representation", None)
            spatial_axes = (
                ("target", "direction")
                if representation == "static"
                else ("target", "direction", "cue_point")
            )
            fields.extend(
                (
                    TensorField("spatial", "embedding", spatial_axes, multiple=True),
                    TensorField("spatial_present", "mask", ("target",), multiple=True),
                )
            )
            if representation == "track":
                fields.extend(
                    (
                        TensorField("spatial_lengths", "label", ("target",), multiple=True),
                        TensorField(
                            "spatial_timebase",
                            "label",
                            ("target", "cue_point"),
                            multiple=True,
                        ),
                    )
                )
        if visual is not None:
            representation = getattr(visual, "representation", None)
            visual_axes = (
                ("target", "height", "width", "channel", "frame")
                if representation == "raw"
                else ("target", "feature", "frame")
            )
            visual_domain = "label" if representation == "raw" else "embedding"
            fields.extend(
                (
                    TensorField("visual", visual_domain, visual_axes, multiple=True),
                    TensorField("visual_present", "mask", ("target",), multiple=True),
                    TensorField("visual_lengths", "label", ("target",), multiple=True),
                    TensorField(
                        "visual_timebase",
                        "label",
                        ("target", "frame"),
                        multiple=True,
                    ),
                )
            )
        if textual is not None:
            fields.extend(
                (
                    TensorField(
                        "textual", "label", ("target", "token"), multiple=True
                    ),
                    TensorField("textual_present", "mask", ("target",), multiple=True),
                    TensorField("textual_lengths", "label", ("target",), multiple=True),
                )
            )
    return DatasetContract(
        task="tse",
        fields=tuple(fields),
        sample_rate=sample_rate,
        cardinality="variable_multi" if variable_targets else "single",
    )
