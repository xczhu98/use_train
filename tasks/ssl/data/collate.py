"""SSL canonical dict-item collate。"""

from collections.abc import Mapping

import torch

from .batch import (
    SSLBatch,
    SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG,
    SSL_INPUT_REFERENCE_IPD,
    SSL_TARGET_DPIPD_FAR_FIELD,
    SSL_TARGET_DPIPD_NEAR_FIELD,
)


_INPUT_REPRESENTATIONS = frozenset({SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG, SSL_INPUT_REFERENCE_IPD})
_TARGET_REPRESENTATIONS = frozenset({SSL_TARGET_DPIPD_FAR_FIELD, SSL_TARGET_DPIPD_NEAR_FIELD})


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


_REQUIRED_ITEM_KEYS = frozenset(
    {
        "features",
        "target_dpipd",
        "doa",
        "vad",
        "pair_index",
        "pair_mask",
        "geometry",
        "input_lengths",
        "output_lengths",
        "frame_origin",
        "time_stride",
        "sample_rate",
        "provenance",
        "input_representation",
        "target_representation",
        "coordinate_frame",
    }
)
_OPTIONAL_ITEM_KEYS = frozenset(
    {
        "receptive_field",
        "algorithmic_latency",
        "valid_output_mask",
        "source_distances",
        "input_frame_index",
        "output_frame_index",
    }
)
# project_ssl_scene/prepare_ssl_scene 会产出这些场景对齐诊断字段；
# 它们属于合法 item extras，但 SSLBatch 没有对应训练字段。
_PROJECTION_ITEM_EXTRAS = frozenset(
    {"mic_positions", "source_positions", "output_source_positions"}
)
_ALLOWED_ITEM_KEYS = _REQUIRED_ITEM_KEYS | _OPTIONAL_ITEM_KEYS | _PROJECTION_ITEM_EXTRAS


def _exact_integer(value, *, field: str, index: int, minimum: int) -> int:
    """校验样本元数据是精确整数，禁止 dtype 转换时截断小数。"""
    try:
        scalar = torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"SSL item {index} {field} must be an integer scalar") from exc
    if scalar.numel() != 1 or scalar.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            f"SSL item {index} {field} must be an integer scalar, "
            f"got shape={tuple(scalar.shape)}, dtype={scalar.dtype}"
        )
    result = int(scalar.item())
    if result < minimum:
        raise ValueError(
            f"SSL item {index} {field} must be >= {minimum}, got {result}"
        )
    return result


def _pad_frame_axis(values: list[torch.Tensor], size: int) -> torch.Tensor:
    padded = []
    for value in values:
        if value.shape[0] > size:
            raise ValueError("SSL item has more frames than the collate target")
        if value.shape[0] == size:
            padded.append(value)
            continue
        padded.append(
            torch.cat(
                (value, value.new_zeros((size - value.shape[0],) + value.shape[1:])),
                dim=0,
            )
        )
    return torch.stack(padded)


def _require_tensor(item: Mapping, index: int, field: str, *, ndim: int) -> torch.Tensor:
    value = item[field]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"SSL item {index} {field} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(
            f"SSL item {index} {field} must have {ndim} dimensions, got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"SSL item {index} {field} must be finite")
    return value


def _validate_item_frame_index(
    value: torch.Tensor,
    *,
    index: int,
    field: str,
    expected: int,
) -> torch.Tensor:
    """校验未补齐 item 的帧索引合同。"""
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.shape[0] != expected:
        raise ValueError(f"SSL item {index} {field} must have shape [{expected}]")
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"SSL item {index} {field} must use an integer dtype")
    if bool((value < 0).any().item()):
        raise ValueError(f"SSL item {index} {field} must be non-negative")
    if value.shape[0] > 1 and bool((value[1:] <= value[:-1]).any().item()):
        raise ValueError(f"SSL item {index} {field} must be strictly increasing")
    return value


def ssl_collate_fn(items):
    """将 canonical SSL dict item 分别补齐输入/输出帧轴并校验合同。

    时间基准字段：item 可显式携带 ``receptive_field`` /
    ``algorithmic_latency``（标量精确整数，批内必须一致）；缺失时按
    average_pool 语义登记默认 ``time_stride`` / ``time_stride - 1``。
    item 可携带 ``valid_output_mask`` ``[T_out]`` bool（全批提供或全批
    省略），未补齐 item 必须全 True；批级 mask 由 ``output_lengths``
    唯一决定，补齐帧恒为 False。未知 item 字段直接拒绝；projection
    产生的场景位置诊断 extras 会保留在 item 边界并在组 batch 时丢弃。
    """
    if not items:
        raise ValueError("SSL batch must not be empty")
    if any(not isinstance(item, Mapping) for item in items):
        raise TypeError("SSL scene items must be mappings")
    for index, item in enumerate(items):
        unknown = set(item.keys()) - _ALLOWED_ITEM_KEYS
        if unknown:
            raise ValueError(
                f"SSL item {index} contains unknown fields: "
                f"{sorted(unknown, key=repr)!r}"
            )
    if any(not _REQUIRED_ITEM_KEYS <= item.keys() for item in items):
        raise ValueError(
            "SSL scene items must include features, target_dpipd, doa, vad, pair_index, "
            "pair_mask, geometry, input_lengths, output_lengths, frame_origin, "
            "time_stride, sample_rate, provenance, representations and coordinate_frame"
        )

    input_lengths: list[int] = []
    output_lengths: list[int] = []
    sample_rates: list[int] = []
    frame_origins: list[int] = []
    time_strides: list[int] = []
    receptive_fields: list[int] = []
    algorithmic_latencies: list[int] = []
    valid_output_masks: list[torch.Tensor | None] = []
    source_distances: list[torch.Tensor | None] = []
    input_frame_indices: list[torch.Tensor | None] = []
    output_frame_indices: list[torch.Tensor | None] = []
    input_representations: list[str] = []
    target_representations: list[str] = []
    coordinate_frames: list[str] = []
    static_shape: tuple[int, int, int, int, int, int] | None = None

    for index, item in enumerate(items):
        features = _require_tensor(item, index, "features", ndim=3)
        target = _require_tensor(item, index, "target_dpipd", ndim=5)
        doa = _require_tensor(item, index, "doa", ndim=3)
        vad = _require_tensor(item, index, "vad", ndim=2)
        pair_index = item["pair_index"]
        pair_mask = item["pair_mask"]
        geometry = _require_tensor(item, index, "geometry", ndim=2)
        if not isinstance(pair_index, torch.Tensor):
            raise TypeError(f"SSL item {index} pair_index must be a torch.Tensor")
        if pair_index.ndim != 2 or pair_index.shape[1] != 2 or pair_index.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"SSL item {index} pair_index must have shape [P,2] and integer dtype")
        if not isinstance(pair_mask, torch.Tensor):
            raise TypeError(f"SSL item {index} pair_mask must be a torch.Tensor")
        if pair_mask.ndim != 1 or pair_mask.dtype is not torch.bool:
            raise ValueError(f"SSL item {index} pair_mask must be bool [P]")
        if not bool(pair_mask.any().item()):
            raise ValueError(f"SSL item {index} pair_mask must keep at least one pair")
        input_representation = item["input_representation"]
        target_representation = item["target_representation"]
        if input_representation not in _INPUT_REPRESENTATIONS:
            raise ValueError(f"SSL item {index} has unsupported input_representation")
        if target_representation not in _TARGET_REPRESENTATIONS:
            raise ValueError(f"SSL item {index} has unsupported target_representation")
        coordinate_frame = item["coordinate_frame"]
        if not isinstance(coordinate_frame, str) or not coordinate_frame:
            raise ValueError(f"SSL item {index} coordinate_frame must be non-empty")
        if geometry.shape[1] != 3:
            raise ValueError(f"SSL item {index} geometry must have shape [C,3]")
        if target.shape[-1] != 2:
            raise ValueError(f"SSL item {index} target_dpipd must end with real/imag dimension 2")
        if features.shape[2] < 2 or features.shape[2] % 2:
            raise ValueError(f"SSL item {index} features must contain real/imag pairs")
        feature_channels = features.shape[2] // 2
        channels = feature_channels if input_representation == SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG else feature_channels + 1
        if geometry.shape[0] != channels:
            raise ValueError(f"SSL item {index} geometry channel count must equal C={channels}")
        if bool((pair_index < 0).any().item()) or bool((pair_index >= channels).any().item()):
            raise ValueError(f"SSL item {index} pair_index contains an out-of-range microphone")
        if pair_index.shape[0] and torch.unique(pair_index, dim=0).shape[0] != pair_index.shape[0]:
            raise ValueError(f"SSL item {index} pair_index must not contain duplicate pairs")
        if pair_index.shape[0] != target.shape[3] or pair_mask.shape[0] != target.shape[3]:
            raise ValueError(f"SSL item {index} pair_index/pair_mask must match target pair axis")
        if doa.shape != (target.shape[0], target.shape[2], 2):
            raise ValueError(f"SSL item {index} doa must have shape [T_out,K,2]")
        if vad.shape != (target.shape[0], target.shape[2]):
            raise ValueError(f"SSL item {index} vad must have shape [T_out,K]")
        input_length = _exact_integer(item["input_lengths"], field="input_lengths", index=index, minimum=1)
        output_length = _exact_integer(item["output_lengths"], field="output_lengths", index=index, minimum=1)
        if input_length != features.shape[0] or output_length != target.shape[0]:
            raise ValueError(f"SSL item {index} lengths must equal unpadded feature/target frame counts")
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        sample_rates.append(_exact_integer(item["sample_rate"], field="sample_rate", index=index, minimum=1000))
        frame_origins.append(_exact_integer(item["frame_origin"], field="frame_origin", index=index, minimum=0))
        time_strides.append(_exact_integer(item["time_stride"], field="time_stride", index=index, minimum=1))
        # 时间基准缺省时按 average_pool 语义回填；显式提供时必须为精确整数。
        if "receptive_field" in item:
            receptive_fields.append(_exact_integer(
                item["receptive_field"], field="receptive_field", index=index, minimum=1
            ))
        else:
            receptive_fields.append(time_strides[-1])
        if "algorithmic_latency" in item:
            algorithmic_latencies.append(_exact_integer(
                item["algorithmic_latency"], field="algorithmic_latency", index=index, minimum=0
            ))
        else:
            algorithmic_latencies.append(time_strides[-1] - 1)
        valid_mask = item.get("valid_output_mask")
        if valid_mask is not None:
            if not isinstance(valid_mask, torch.Tensor):
                raise TypeError(f"SSL item {index} valid_output_mask must be a torch.Tensor")
            if valid_mask.ndim != 1 or valid_mask.shape[0] != target.shape[0]:
                raise ValueError(f"SSL item {index} valid_output_mask must have shape [T_out]")
            if valid_mask.dtype is not torch.bool:
                raise TypeError(f"SSL item {index} valid_output_mask must use a bool dtype")
            if not bool(valid_mask.all().item()):
                raise ValueError(
                    f"SSL item {index} valid_output_mask must be all True for an unpadded item"
                )
        valid_output_masks.append(valid_mask)

        distance = item.get("source_distances")
        if distance is not None:
            if target_representation != SSL_TARGET_DPIPD_NEAR_FIELD:
                raise ValueError(
                    f"SSL item {index} source_distances require a near-field target or range mapping"
                )
            distance = _require_tensor(item, index, "source_distances", ndim=2)
            if distance.shape != (target.shape[0], target.shape[2]):
                raise ValueError(f"SSL item {index} source_distances must have shape [T_out,K]")
            if bool((distance <= 0).any().item()):
                raise ValueError(f"SSL item {index} source_distances must be strictly positive")
        elif target_representation == SSL_TARGET_DPIPD_NEAR_FIELD:
            raise ValueError(f"SSL item {index} near-field targets require source_distances [T_out,K]")
        source_distances.append(distance)

        for field, expected, collected in (
            ("input_frame_index", features.shape[0], input_frame_indices),
            ("output_frame_index", target.shape[0], output_frame_indices),
        ):
            value = item.get(field)
            if value is not None:
                value = _validate_item_frame_index(
                    value,
                    index=index,
                    field=field,
                    expected=expected,
                )
            collected.append(value)

        shape = (features.shape[1], features.shape[2], target.shape[2], target.shape[3], target.shape[4], channels)
        if static_shape is None:
            static_shape = shape
        elif static_shape != shape:
            raise ValueError("SSL items must agree on frequency, feature, track, pair and channel dimensions")
        input_representations.append(input_representation)
        target_representations.append(target_representation)
        coordinate_frames.append(coordinate_frame)

    if len(set(input_representations)) != 1:
        raise ValueError("SSL batch must not mix input_representation values")
    if len(set(target_representations)) != 1:
        raise ValueError("SSL batch must not mix target_representation values")
    if len(set(coordinate_frames)) != 1:
        raise ValueError("SSL batch must not mix coordinate_frame values")
    if len(set(time_strides)) != 1:
        raise ValueError("SSL batch must not mix time_stride values")
    if len(set(receptive_fields)) != 1:
        raise ValueError("SSL batch must not mix receptive_field values")
    if len(set(algorithmic_latencies)) != 1:
        raise ValueError("SSL batch must not mix algorithmic_latency values")
    if algorithmic_latencies[0] > receptive_fields[0] - 1:
        raise ValueError("SSL algorithmic_latency must not exceed receptive_field - 1")
    if any(value is not None for value in valid_output_masks) and not all(
        value is not None for value in valid_output_masks
    ):
        raise ValueError("SSL batch must provide valid_output_mask for every item or none")
    has_distance = [value is not None for value in source_distances]
    if any(has_distance) and not all(has_distance):
        raise ValueError("SSL batch must either provide source_distances for every item or none")
    if target_representations[0] == SSL_TARGET_DPIPD_NEAR_FIELD and not all(has_distance):
        raise ValueError("near-field SSL batches require source_distances for every item")
    if any(value is not None for value in input_frame_indices) and not all(value is not None for value in input_frame_indices):
        raise ValueError("SSL batch must provide input_frame_index for every item or none")
    if any(value is not None for value in output_frame_indices) and not all(value is not None for value in output_frame_indices):
        raise ValueError("SSL batch must provide output_frame_index for every item or none")

    max_input = max(input_lengths)
    max_output = max(output_lengths)
    features = _pad_frame_axis([item["features"] for item in items], max_input)
    target = _pad_frame_axis([item["target_dpipd"] for item in items], max_output)
    doa = _pad_frame_axis([item["doa"] for item in items], max_output)
    vad = _pad_frame_axis([item["vad"] for item in items], max_output)
    distance = None if not all(has_distance) else _pad_frame_axis(
        [value for value in source_distances if value is not None], max_output
    )
    input_frame_index = None if not all(value is not None for value in input_frame_indices) else _pad_frame_axis(
        [value for value in input_frame_indices if value is not None], max_input
    )
    output_frame_index = None if not all(value is not None for value in output_frame_indices) else _pad_frame_axis(
        [value for value in output_frame_indices if value is not None], max_output
    )
    # 批级有效输出帧由 output_lengths 唯一决定，补齐帧恒为 False。
    valid_output_mask = torch.arange(max_output)[None, :] < torch.tensor(
        output_lengths, dtype=torch.int64
    )[:, None]
    batch = SSLBatch(
        features=features,
        target_dpipd=target,
        input_lengths=torch.tensor(input_lengths, dtype=torch.int64),
        output_lengths=torch.tensor(output_lengths, dtype=torch.int64),
        pair_index=torch.stack([item["pair_index"] for item in items]),
        pair_mask=torch.stack([item["pair_mask"] for item in items]),
        geometry=torch.stack([item["geometry"] for item in items]),
        frame_origin=torch.tensor(frame_origins, dtype=torch.int64),
        time_stride=time_strides[0],
        receptive_field=receptive_fields[0],
        algorithmic_latency=algorithmic_latencies[0],
        valid_output_mask=valid_output_mask,
        doa=doa,
        vad=vad,
        sample_rates=torch.tensor(sample_rates, dtype=torch.int64),
        provenance=tuple(item["provenance"] for item in items),
        source_distances=distance,
        input_representation=input_representations[0],
        target_representation=target_representations[0],
        coordinate_frame=coordinate_frames[0],
        input_frame_index=input_frame_index,
        output_frame_index=output_frame_index,
    )
    batch.validate()
    return batch
