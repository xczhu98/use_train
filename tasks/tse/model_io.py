"""TSE 模型的合同驱动 I/O 边界与显式流式状态类型。

trainer 与整句 inference 共同通过本模块装配 waveform、spectrum 和 typed cue；
causal 模型另用 ``TSEStreamingState`` 显式携带 enrollment 条件和时序缓存。
调用方仍负责重采样、短音频 padding、loss 与输出归一化。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from core.models.io_contract import ModelIOContract, TensorField
from tasks.tse.cue_contract import TSECueBatch, TSECueBatches


class _ClonedTensorMapping(Mapping[str, torch.Tensor]):
    """持有私有 tensor 副本，并在读取时返回新副本的只读映射。"""

    def __init__(self, values: Mapping[str, torch.Tensor]) -> None:
        self._values = {key: value.clone() for key, value in values.items()}

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._values[key].clone()

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


@dataclass(frozen=True)
class TSEStreamingState:
    """一次目标说话人流会话的 enrollment 条件与时序缓存。

    ``enrollment`` 由初始化/显式 reset 生成，chunk 前向只读取并携带；调用方不能
    在保留旧 LSTM 历史的同时偷换目标说话人。
    """

    batch_size: int
    enrollment: Mapping[str, torch.Tensor]
    separator_states: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    usef_history: torch.Tensor | None = None

    def __post_init__(self) -> None:
        # MappingProxyType 只能冻结键值映射，不能隔离 tensor 的原地修改；私有副本
        # 与读取副本共同避免初始化输入或公开 state.enrollment 污染会话缓存。
        object.__setattr__(self, "enrollment", _ClonedTensorMapping(self.enrollment))


_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_LEGACY_TSE_MODEL_INPUT_FIELDS = frozenset(
    {"mix_spectra", "enroll_spectra", "enroll_wav"}
)
_LEGACY_FIELD_SPECS = {
    "mix_spectra": ("spectrum", ("batch", "frequency", "frame", "complex")),
    "enroll_spectra": ("spectrum", ("batch", "frequency", "frame", "complex")),
    "enroll_wav": ("waveform", ("batch", "time")),
    "enroll_lengths": ("label", ("batch",)),
    "enroll_spectra_lengths": ("label", ("batch",)),
    "spk_embedding": ("embedding", ("batch", "embedding")),
}
_FIELD_DOMAINS: dict[str, frozenset[str]] = {
    "mix_spectra": frozenset({"spectrum"}),
    "mix_wav": frozenset({"waveform"}),
    "mixture": frozenset({"waveform"}),
    "lengths": frozenset({"label"}),
    "enroll_spectra": frozenset({"spectrum"}),
    "enroll_spectra_lengths": frozenset({"label"}),
    "enroll_wav": frozenset({"waveform"}),
    "enroll_lengths": frozenset({"label"}),
    "enroll_present": frozenset({"mask"}),
    "speaker_waveform": frozenset({"waveform"}),
    "speaker_waveform_lengths": frozenset({"label"}),
    "speaker_waveform_present": frozenset({"mask"}),
    "spk_embedding": frozenset({"embedding"}),
    "spk_embedding_present": frozenset({"mask"}),
    "speaker_embedding": frozenset({"embedding"}),
    "speaker_embedding_present": frozenset({"mask"}),
    "spatial": frozenset({"embedding"}),
    "spatial_present": frozenset({"mask"}),
    "spatial_lengths": frozenset({"label"}),
    "spatial_timebase": frozenset({"label"}),
    "visual": frozenset({"embedding", "label"}),
    "visual_present": frozenset({"mask"}),
    "visual_lengths": frozenset({"label"}),
    "visual_timebase": frozenset({"label"}),
    "textual": frozenset({"label"}),
    "textual_present": frozenset({"mask"}),
    "textual_lengths": frozenset({"label"}),
}
_CUE_FIELD_NAMES = {
    "speaker_waveform": frozenset(
        {
            "enroll_spectra",
            "enroll_spectra_lengths",
            "enroll_wav",
            "enroll_lengths",
            "enroll_present",
            "speaker_waveform",
            "speaker_waveform_lengths",
            "speaker_waveform_present",
        }
    ),
    "speaker_embedding": frozenset(
        {
            "spk_embedding",
            "spk_embedding_present",
            "speaker_embedding",
            "speaker_embedding_present",
        }
    ),
    "spatial": frozenset(
        {"spatial", "spatial_present", "spatial_lengths", "spatial_timebase"}
    ),
    "visual": frozenset(
        {"visual", "visual_present", "visual_lengths", "visual_timebase"}
    ),
    "textual": frozenset(
        {"textual", "textual_present", "textual_lengths"}
    ),
}


def _unwrap_tse_model(model: Any) -> Any:
    """解包 DDP/compile wrapper，读取模型的真实输入合同。"""
    raw_model = model
    seen: set[int] = set()
    while id(raw_model) not in seen:
        seen.add(id(raw_model))
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module
            continue
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
            continue
        break
    return raw_model


def _validate_tse_input_field(field: TensorField) -> None:
    domains = _FIELD_DOMAINS.get(field.name)
    if domains is None:
        raise ValueError(f"unsupported TSE model input field: {field.name!r}")
    if field.domain not in domains:
        raise ValueError(
            f"TSE model input field {field.name!r} must use domain "
            f"{sorted(domains)}, got {field.domain!r}"
        )
    if field.name == "mixture" and "microphone" in field.axes:
        raise ValueError(
            "TSE data['mixture'] is the reference-channel loss waveform; "
            "use mix_wav for a microphone-axis model input"
        )


def _resolve_tse_model_input_contract(model: Any) -> tuple[TensorField, ...]:
    raw_model = _unwrap_tse_model(model)
    contract = getattr(raw_model, "IO_CONTRACT", None)
    if contract is not None:
        if not isinstance(contract, ModelIOContract):
            raise TypeError("model.IO_CONTRACT must be a ModelIOContract")
        if "tse" not in contract.tasks:
            raise ValueError("model.IO_CONTRACT does not support TSE")
        for field in contract.inputs:
            _validate_tse_input_field(field)
        return contract.inputs

    resolver = getattr(raw_model, "required_tse_input_fields", None)
    if resolver is None:
        names = _LEGACY_TSE_MODEL_INPUT_FIELDS
    else:
        if not callable(resolver):
            raise TypeError("model.required_tse_input_fields must be callable")
        names = resolver()
        if not isinstance(names, (set, frozenset)) or any(
            not isinstance(field, str) or not field for field in names
        ):
            raise TypeError(
                "model.required_tse_input_fields() must return a set of non-empty strings"
            )
        names = frozenset(names)
    if "mix_spectra" not in names:
        raise ValueError("legacy TSE model input fields must include mix_spectra")
    unknown = names - _LEGACY_FIELD_SPECS.keys()
    if unknown:
        raise ValueError(f"unsupported legacy TSE model input fields: {sorted(unknown)}")
    return tuple(
        TensorField(name, *_LEGACY_FIELD_SPECS[name]) for name in sorted(names)
    )


def resolve_tse_model_input_fields(model: Any) -> frozenset[str]:
    """解析模型本次配置真正消费的 TSE 输入字段。"""
    return frozenset(field.name for field in _resolve_tse_model_input_contract(model))


def tse_model_accepts_multichannel(model: Any) -> bool:
    """模型只有显式声明 microphone 轴时才可以消费多通道 mixture。"""
    raw_model = _unwrap_tse_model(model)
    return any(
        field.name in {"mix_spectra", "mix_wav"}
        and field.domain in {"spectrum", "waveform"}
        and "microphone" in field.axes
        for field in _resolve_tse_model_input_contract(raw_model)
    )


def validate_tse_multichannel_contract(model: Any) -> None:
    """校验原生多通道 TSE 的实际波形/谱域输入输出合同。"""
    raw_model = _unwrap_tse_model(model)
    if getattr(raw_model, "supports_multichannel_input", False) is not True:
        raise ValueError(
            "spatial TSE requires model.supports_multichannel_input=True; "
            "the configured model is mono-only"
        )
    contract = getattr(raw_model, "IO_CONTRACT", None)
    if not isinstance(contract, ModelIOContract):
        raise ValueError("native multichannel TSE requires an explicit model.IO_CONTRACT")
    inputs = {field.name: field for field in contract.inputs}
    outputs = {field.name: field for field in contract.outputs}
    mixture_fields = [
        field
        for name, field in inputs.items()
        if name in {"mix_spectra", "mix_wav"} and "microphone" in field.axes
    ]
    if not mixture_fields:
        raise ValueError(
            "native multichannel TSE requires mix_spectra or mix_wav with a microphone axis"
        )
    for name in (
        "enroll_spectra",
        "enroll_wav",
        "speaker_waveform",
    ):
        field = inputs.get(name)
        if field is not None and "microphone" in field.axes:
            raise ValueError(f"native multichannel TSE {name} must remain mono")
    estimate = outputs.get("target_estimate_spectra") or outputs.get("target_estimate")
    if estimate is None or "microphone" in estimate.axes:
        raise ValueError("native multichannel TSE output must be mono target estimate")


def _validate_waveform(waveform: torch.Tensor, name: str) -> None:
    if not isinstance(waveform, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(waveform).__name__}")
    if waveform.ndim < 2:
        raise ValueError(f"{name} must have at least [B, T] dimensions, got {tuple(waveform.shape)}")
    if waveform.shape[0] < 1 or waveform.shape[-1] < 1:
        raise ValueError(f"{name} requires non-empty batch and time axes, got {tuple(waveform.shape)}")
    if not waveform.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype, got {waveform.dtype}")
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError(f"{name} contains NaN/Inf")


def _validate_lengths(
    lengths: torch.Tensor,
    *,
    batch_size: int,
    max_length: int,
    name: str,
) -> torch.Tensor:
    if not isinstance(lengths, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(lengths).__name__}")
    if lengths.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must have integer dtype, got {lengths.dtype}")
    if lengths.ndim != 1 or lengths.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [{batch_size}], got {tuple(lengths.shape)}")
    if bool(((lengths < 1) | (lengths > max_length)).any().item()):
        raise ValueError(f"{name} must be in [1, {max_length}], got {lengths.tolist()}")
    return lengths


def _validate_num_targets(num_targets: torch.Tensor, batch_size: int) -> torch.Tensor:
    return _validate_lengths(
        num_targets,
        batch_size=batch_size,
        max_length=torch.iinfo(num_targets.dtype).max
        if isinstance(num_targets, torch.Tensor) and num_targets.dtype in _INTEGER_DTYPES
        else 1,
        name="num_targets",
    )


def mask_waveform_tail(
    waveform: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """按每行有效长度清零尾部，并保留输入 shape。"""
    _validate_waveform(waveform, "waveform")
    safe_lengths = _validate_lengths(
        lengths,
        batch_size=waveform.shape[0],
        max_length=waveform.shape[-1],
        name="lengths",
    ).to(device=waveform.device, dtype=torch.long)
    time = torch.arange(waveform.shape[-1], device=waveform.device)
    invalid = time.unsqueeze(0) >= safe_lengths.unsqueeze(1)
    while invalid.ndim < waveform.ndim:
        invalid = invalid.unsqueeze(1)
    return waveform.masked_fill(invalid, 0)


def _stft_enrollment(
    stft: Any, waveform: torch.Tensor, lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """先裁真实样点再分析，谱尾 padding 只用于组装，不参与模型计算。"""
    if lengths.device != waveform.device:
        raise ValueError("enroll_lengths must use the same device as enroll")
    if lengths.dtype not in _INTEGER_DTYPES or lengths.shape != (waveform.shape[0],):
        raise ValueError("enroll_lengths must be an integer vector aligned with enroll")
    if bool(((lengths < 0) | (lengths > waveform.shape[-1])).any().item()):
        raise ValueError("enroll_lengths exceeds enrollment capacity")
    groups: dict[int, list[int]] = {}
    for row, length in enumerate(lengths.detach().cpu().tolist()):
        groups.setdefault(length, []).append(row)
    n_fft = getattr(stft, "n_fft", None)
    minimum = (n_fft // 2 + 1 if getattr(stft, "center", True) else n_fft) if n_fft else 1
    spectra, indices, frame_lengths = [], [], []
    for length, rows in groups.items():
        index = torch.tensor(rows, device=waveform.device)
        valid = waveform.index_select(0, index)[..., :length]
        valid = F.pad(valid, (0, max(0, minimum - length)))
        spectrum = stft(valid)
        spectra.append(spectrum)
        indices.append(index)
        frame_lengths.extend([spectrum.shape[-2] if length else 0] * len(rows))
    capacity = max(spectrum.shape[-2] for spectrum in spectra)
    order = torch.cat(indices).argsort()
    result = torch.cat([F.pad(spectrum, (0, 0, 0, capacity - spectrum.shape[-2]))
                        for spectrum in spectra]).index_select(0, order)
    sizes = torch.tensor(frame_lengths, dtype=torch.int32, device=waveform.device)
    return result, sizes.index_select(0, order)


def _stft_observation(stft: Any, waveform: torch.Tensor) -> torch.Tensor:
    """对 ``[B,C,T]`` observation 按 channel 调用 mono STFT。"""
    if waveform.ndim == 2:
        return stft(waveform)
    if waveform.ndim != 3 or waveform.shape[1] < 1:
        raise ValueError(f"observation waveform must be [B,T] or [B,C,T], got {tuple(waveform.shape)}")
    batch, channels, length = waveform.shape
    spectra = stft(waveform.reshape(batch * channels, length))
    return spectra.reshape(batch, channels, *spectra.shape[1:])


def per_target_lengths(
    lengths: torch.Tensor,
    num_targets: torch.Tensor | None,
) -> torch.Tensor:
    """把 per-mixture lengths 按 sample-major 目标数展开。"""
    if num_targets is None:
        return lengths
    _validate_num_targets(num_targets, lengths.shape[0])
    return torch.repeat_interleave(lengths, num_targets.long(), dim=0)


def guard_tse_model_sample_rate(model: Any, sample_rate: int) -> None:
    """确保运行时采样率与模型声明一致。"""
    model_sample_rate = getattr(model, "sample_rate", None)
    if model_sample_rate is not None and int(model_sample_rate) != int(sample_rate):
        raise ValueError(
            f"TSE batch sample rate {sample_rate} does not match "
            f"model.sample_rate={model_sample_rate}"
        )


def _resolve_reference_channel(
    model: Any,
    *,
    channels: int,
    reference_channel: int | None,
) -> int:
    raw_model = _unwrap_tse_model(model)
    declared = getattr(
        raw_model,
        "ref_channel",
        getattr(raw_model, "reference_channel", None),
    )
    if declared is not None and (
        isinstance(declared, bool) or not isinstance(declared, int)
    ):
        raise TypeError("model reference channel must be an integer")
    if reference_channel is not None and (
        isinstance(reference_channel, bool) or not isinstance(reference_channel, int)
    ):
        raise TypeError("reference_channel must be an integer")
    if declared is not None and reference_channel is not None and declared != reference_channel:
        raise ValueError(
            f"reference_channel={reference_channel} conflicts with model reference channel {declared}"
        )
    resolved = declared if declared is not None else reference_channel
    if resolved is None:
        if channels != 1:
            raise ValueError(
                "multi-channel TSE requires an explicit model/call reference_channel"
            )
        resolved = 0
    if resolved < 0 or resolved >= channels:
        raise ValueError(
            f"reference_channel={resolved} must be in [0, {channels - 1}]"
        )
    return resolved


def _repeat_target_rows(value: torch.Tensor, repeats: torch.Tensor | None) -> torch.Tensor:
    if repeats is None:
        return value
    return torch.repeat_interleave(value, repeats.long(), dim=0)


def _validate_cue_batch(
    cue: TSECueBatch,
    *,
    slot: str,
    rows: int,
    device: torch.device,
) -> None:
    if not isinstance(cue, TSECueBatch):
        raise TypeError(f"cues.{slot} must be a TSECueBatch")
    if cue.value.ndim < 2 or cue.value.shape[0] != rows:
        raise ValueError(f"cues.{slot}.value must have {rows} target rows")
    if tuple(cue.present.shape) != (rows,) or cue.present.dtype != torch.bool:
        raise ValueError(f"cues.{slot}.present must be bool with shape [{rows}]")
    tensors = [cue.value, cue.present]
    if cue.lengths is not None:
        if tuple(cue.lengths.shape) != (rows,) or cue.lengths.dtype != torch.int32:
            raise ValueError(f"cues.{slot}.lengths must be int32 with shape [{rows}]")
        tensors.append(cue.lengths)
    if cue.timebase is not None:
        if cue.timebase.ndim != 2 or cue.timebase.shape[0] != rows:
            raise ValueError(f"cues.{slot}.timebase must have {rows} target rows")
        tensors.append(cue.timebase)
    if any(tensor.device != device for tensor in tensors):
        raise ValueError(f"cues.{slot} tensors must use the mixture device")


def _cue_value_fields(slot: str) -> frozenset[str]:
    return {
        "speaker_waveform": frozenset(
            {"enroll_spectra", "enroll_wav", "speaker_waveform"}
        ),
        "speaker_embedding": frozenset({"spk_embedding", "speaker_embedding"}),
        "spatial": frozenset({"spatial"}),
        "visual": frozenset({"visual"}),
        "textual": frozenset({"textual"}),
    }[slot]


def build_tse_model_inputs(
    *,
    model: Any,
    stft: Any | None,
    mixture: torch.Tensor,
    sample_rate: int,
    cues: TSECueBatches | None = None,
    enroll: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
    enroll_lengths: torch.Tensor | None = None,
    spk_embedding: torch.Tensor | None = None,
    num_targets: torch.Tensor | None = None,
    reference_channel: int | None = None,
) -> dict[str, Any]:
    """按模型完整 I/O 合同装配 waveform、spectrum 与 typed cue 输入。

    频域模型只对 ``B`` 条 observation 执行一次 STFT，再按 sample-major
    ``num_targets`` 展开。``data['mixture']`` 始终是参考通道 ``[R,T]``，原生
    多通道模型通过带 ``microphone`` 轴的 ``mix_spectra``/``mix_wav`` 消费完整
    observation。旧整句推理仍可显式传 ``enroll``，但新批合同应传 ``cues``。
    """
    guard_tse_model_sample_rate(model, sample_rate)
    input_contract = _resolve_tse_model_input_contract(model)
    fields = {field.name: field for field in input_contract}

    _validate_waveform(mixture, "mixture")
    if mixture.ndim not in (2, 3):
        raise ValueError(
            "mixture must be [B,T] or [B,C,T], "
            f"got {tuple(mixture.shape)}"
        )
    batch_size = int(mixture.shape[0])
    channels = int(mixture.shape[1]) if mixture.ndim == 3 else 1
    if channels > 1 and not tse_model_accepts_multichannel(model):
        raise ValueError(
            "multi-channel TSE mixture requires a native model IO contract with a "
            "microphone axis; reference-channel projection is not supported"
        )
    ref_channel = _resolve_reference_channel(
        model,
        channels=channels,
        reference_channel=reference_channel,
    )

    repeats: torch.Tensor | None = None
    if num_targets is not None:
        repeats = _validate_num_targets(num_targets, batch_size).to(
            device=mixture.device
        )
    expanded_rows = (
        int(repeats.long().sum().item()) if repeats is not None else batch_size
    )
    observation = mask_waveform_tail(mixture, lengths) if lengths is not None else mixture
    target_lengths = per_target_lengths(lengths, num_targets) if lengths is not None else None
    reference = observation[:, ref_channel] if observation.ndim == 3 else observation
    reference_rows = _repeat_target_rows(reference, repeats)

    if target is not None:
        _validate_waveform(target, "target")
        if target.ndim != 2:
            raise ValueError(f"target must be [R,T], got {tuple(target.shape)}")
        if target.shape != (expanded_rows, mixture.shape[-1]):
            raise ValueError(
                "target must have one row per target and the mixture time length: "
                f"got {tuple(target.shape)}, expected [{expanded_rows}, {mixture.shape[-1]}]"
            )
        if target.device != mixture.device or target.dtype != mixture.dtype:
            raise ValueError("target must use the same device and dtype as mixture")
        if target_lengths is not None:
            target = mask_waveform_tail(target, target_lengths)

    if cues is not None and not isinstance(cues, TSECueBatches):
        raise TypeError("cues must be TSECueBatches or None")
    cue_batches = cues if cues is not None else TSECueBatches()
    for slot in _CUE_FIELD_NAMES:
        cue = getattr(cue_batches, slot)
        if cue is None:
            continue
        _validate_cue_batch(
            cue,
            slot=slot,
            rows=expanded_rows,
            device=mixture.device,
        )
        declared = set(fields).intersection(_CUE_FIELD_NAMES[slot])
        if not declared.intersection(_cue_value_fields(slot)):
            raise ValueError(
                f"typed cue {slot!r} is enabled but model.IO_CONTRACT does not consume it"
            )
        presence_fields = {
            name for name in declared if name.endswith("_present") or name == "enroll_present"
        }
        if bool((~cue.present).any().item()) and not presence_fields:
            raise ValueError(
                f"typed cue {slot!r} has missing rows but the model contract has no presence field"
            )

    speaker_waveform = cue_batches.speaker_waveform
    if enroll is not None:
        if speaker_waveform is not None:
            raise ValueError("provide either cues.speaker_waveform or enroll, not both")
        _validate_waveform(enroll, "enroll")
        if enroll.ndim != 2 or enroll.shape[0] != expanded_rows:
            raise ValueError(
                f"enroll must be [target_rows,T] with {expanded_rows} rows, "
                f"got {tuple(enroll.shape)}"
            )
        if enroll.device != mixture.device or enroll.dtype != mixture.dtype:
            raise ValueError("mixture and enroll must use the same device and dtype")
        if enroll_lengths is not None:
            enroll = mask_waveform_tail(enroll, enroll_lengths)
        speaker_waveform_value = enroll
        speaker_waveform_lengths = (
            enroll_lengths
            if enroll_lengths is not None
            else torch.full(
                (expanded_rows,),
                enroll.shape[-1],
                dtype=torch.int32,
                device=enroll.device,
            )
        )
        speaker_waveform_present = torch.ones(
            expanded_rows, dtype=torch.bool, device=enroll.device
        )
    elif speaker_waveform is not None:
        if enroll_lengths is not None:
            raise ValueError("enroll_lengths cannot be combined with typed speaker waveform cue")
        speaker_waveform_value = speaker_waveform.value
        speaker_waveform_lengths = speaker_waveform.lengths
        speaker_waveform_present = speaker_waveform.present
    else:
        if enroll_lengths is not None:
            raise ValueError("enroll_lengths requires enroll or cues.speaker_waveform")
        speaker_waveform_value = None
        speaker_waveform_lengths = None
        speaker_waveform_present = None

    speaker_embedding = cue_batches.speaker_embedding
    if spk_embedding is not None:
        if speaker_embedding is not None:
            raise ValueError("provide either cues.speaker_embedding or spk_embedding, not both")
        if not isinstance(spk_embedding, torch.Tensor) or spk_embedding.ndim != 2:
            raise ValueError("spk_embedding must be [target_rows,D]")
        if spk_embedding.shape[0] != expanded_rows:
            raise ValueError(
                f"spk_embedding rows={spk_embedding.shape[0]} must match target rows={expanded_rows}"
            )
        if spk_embedding.device != mixture.device or spk_embedding.dtype != mixture.dtype:
            raise ValueError("spk_embedding must use the same device and dtype as mixture")
        if not bool(torch.isfinite(spk_embedding).all().item()):
            raise ValueError("spk_embedding contains NaN/Inf")
        speaker_embedding_value = spk_embedding
        speaker_embedding_present = torch.ones(
            expanded_rows, dtype=torch.bool, device=mixture.device
        )
    elif speaker_embedding is not None:
        speaker_embedding_value = speaker_embedding.value
        speaker_embedding_present = speaker_embedding.present
    else:
        speaker_embedding_value = None
        speaker_embedding_present = None

    available: dict[str, torch.Tensor] = {
        "mixture": reference_rows,
    }
    if target_lengths is not None:
        available["lengths"] = target_lengths
    if "mix_wav" in fields:
        source = observation
        if "microphone" not in fields["mix_wav"].axes:
            source = reference
        available["mix_wav"] = _repeat_target_rows(source, repeats)
    if "mix_spectra" in fields:
        if stft is None:
            raise ValueError("model mix_spectra input requires a runtime STFT module")
        observation_spectra = _stft_observation(stft, observation)
        source_spectra = observation_spectra
        if "microphone" not in fields["mix_spectra"].axes and observation.ndim == 3:
            source_spectra = observation_spectra[:, ref_channel]
        available["mix_spectra"] = _repeat_target_rows(source_spectra, repeats)

    if speaker_waveform_value is not None:
        available.update(
            {
                "enroll_wav": speaker_waveform_value,
                "speaker_waveform": speaker_waveform_value,
                "enroll_present": speaker_waveform_present,
                "speaker_waveform_present": speaker_waveform_present,
            }
        )
        if speaker_waveform_lengths is not None:
            available["enroll_lengths"] = speaker_waveform_lengths
            available["speaker_waveform_lengths"] = speaker_waveform_lengths
        if "enroll_spectra" in fields:
            if stft is None:
                raise ValueError("model enroll_spectra input requires a runtime STFT module")
            if speaker_waveform_lengths is None:
                raise ValueError("enroll_spectra requires enrollment sample lengths")
            spectrum, spectrum_lengths = _stft_enrollment(
                stft, speaker_waveform_value, speaker_waveform_lengths,
            )
            if "enroll_spectra_lengths" not in fields and bool(
                (spectrum_lengths != spectrum.shape[-2]).any().item()
            ):
                raise ValueError("variable enrollment spectra require enroll_spectra_lengths in model contract")
            available["enroll_spectra"] = spectrum
            available["enroll_spectra_lengths"] = spectrum_lengths
    if speaker_embedding_value is not None:
        available.update(
            {
                "spk_embedding": speaker_embedding_value,
                "speaker_embedding": speaker_embedding_value,
                "spk_embedding_present": speaker_embedding_present,
                "speaker_embedding_present": speaker_embedding_present,
            }
        )

    for slot in ("spatial", "visual", "textual"):
        cue = getattr(cue_batches, slot)
        if cue is None:
            continue
        available[slot] = cue.value
        available[f"{slot}_present"] = cue.present
        if cue.lengths is not None:
            available[f"{slot}_lengths"] = cue.lengths
        if cue.timebase is not None:
            available[f"{slot}_timebase"] = cue.timebase
    missing = [
        field.name
        for field in input_contract
        if field.required and field.name not in available
    ]
    if missing:
        raise ValueError(f"missing required TSE model inputs: {sorted(missing)}")

    data: dict[str, Any] = {
        "mixture": reference_rows,
        "sample_rate": int(sample_rate),
    }
    for field in input_contract:
        if field.name in available:
            data[field.name] = available[field.name]
    if target is not None:
        data["target"] = target
    if target_lengths is not None:
        data["lengths"] = target_lengths
    return data


def _align_waveform_length(waveform: torch.Tensor, reference_length: int) -> torch.Tensor:
    if waveform.shape[-1] < reference_length:
        return F.pad(waveform, (0, reference_length - waveform.shape[-1]))
    if waveform.shape[-1] > reference_length:
        return waveform[..., :reference_length]
    return waveform


def decode_tse_model_output(
    data: dict[str, Any],
    *,
    stft: Any | None,
    reference_length: int,
    target: torch.Tensor | None = None,
    target_lengths: torch.Tensor | None = None,
) -> dict[str, Any]:
    """把谱域或 waveform-native 输出解码并严格对齐参考长度。"""
    has_estimate_spectra = "target_estimate_spectra" in data
    if has_estimate_spectra:
        if stft is None:
            raise ValueError("spectral TSE model output requires a runtime STFT module")
        estimate = stft.inverse(data["target_estimate_spectra"], reference_length)
    elif "target_estimate" in data:
        estimate = data["target_estimate"]
    else:
        raise RuntimeError(
            "Model must output either 'target_estimate_spectra' or 'target_estimate'. "
            f"Got keys: {list(data.keys())}"
        )

    estimate = _align_waveform_length(estimate, reference_length)
    if target_lengths is not None:
        estimate = mask_waveform_tail(estimate, target_lengths)
    data["target_estimate"] = estimate
    if not has_estimate_spectra and stft is not None:
        data["target_estimate_spectra"] = stft(estimate)

    if target is not None and estimate.shape != target.shape:
        detail = ""
        if stft is not None:
            detail = f" (hop_length={stft.hop_length} may not divide evenly)"
        raise RuntimeError(
            f"TSE shape mismatch: target_estimate {estimate.shape} != target {target.shape}{detail}."
        )
    return data
