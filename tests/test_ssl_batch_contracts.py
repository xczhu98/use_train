"""SSLBatch 与 canonical collate 的帧轴和近场元数据合同。"""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from tasks.ssl.data import ssl_collate_fn


pytestmark = pytest.mark.unit


def _item(
    *,
    input_frames: int = 4,
    output_frames: int = 2,
    target_representation: str = "dpipd_near_field",
    frame_origin: int = 0,
):
    tracks = 1
    pairs = 1
    frequencies = 3
    item = {
        "features": torch.zeros(input_frames, frequencies, 4),
        "target_dpipd": torch.zeros(output_frames, frequencies, tracks, pairs, 2),
        "doa": torch.zeros(output_frames, tracks, 2),
        "vad": torch.ones(output_frames, tracks),
        "pair_index": torch.tensor([[0, 1]], dtype=torch.int64),
        "pair_mask": torch.ones(pairs, dtype=torch.bool),
        "geometry": torch.tensor(
            [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]], dtype=torch.float32
        ),
        "input_lengths": input_frames,
        "output_lengths": output_frames,
        "frame_origin": frame_origin,
        "time_stride": 2,
        "sample_rate": 16000,
        "input_representation": "multichannel_stft_real_imag",
        "target_representation": target_representation,
        "coordinate_frame": "array_local",
        "provenance": {"backend": "fixture"},
        "input_frame_index": torch.arange(input_frames, dtype=torch.int64),
        "output_frame_index": torch.arange(0, output_frames * 2, 2, dtype=torch.int64),
    }
    if target_representation == "dpipd_near_field":
        item["source_distances"] = torch.ones(output_frames, tracks)
    return item


def test_near_field_distances_allow_zero_padding_after_collate():
    """source_distances 的补齐帧为零，但有效帧仍须严格为正。"""
    batch = ssl_collate_fn([_item(input_frames=4, output_frames=2), _item(input_frames=6, output_frames=3)])

    assert batch.source_distances.shape == (2, 3, 1)
    assert batch.source_distances[0, :, 0].tolist() == [1.0, 1.0, 0.0]
    batch.validate()


def test_batch_validate_checks_distance_only_on_valid_output_frames():
    batch = ssl_collate_fn([_item(input_frames=4, output_frames=2), _item(input_frames=6, output_frames=3)])

    with pytest.raises(TypeError, match="source_distances.*torch.Tensor"):
        replace(batch, source_distances=[[1.0], [1.0]]).validate()

    invalid = replace(batch, source_distances=batch.source_distances.clone())
    invalid.source_distances[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="source_distances.*strictly positive"):
        invalid.validate()

    invalid = replace(batch, source_distances=batch.source_distances.clone())
    invalid.source_distances[0, 2, 0] = 2.0
    with pytest.raises(ValueError, match="source_distances.*zero"):
        invalid.validate()


def _item_with_provenance(**kwargs):
    item = _item(**kwargs)
    item["provenance"] = {
        "backend": "fixture",
        "coordinate_frame": item["coordinate_frame"],
        "sample_rate": item["sample_rate"],
        "geometry": {
            "positions_m": item["geometry"].tolist(),
            "reference_channel": 0,
        },
    }
    if "source_distances" in item:
        item["provenance"]["source_distances"] = item["source_distances"].tolist()
    return item


def test_collate_accepts_projection_extras_but_rejects_unknown_item_fields():
    item = _item()
    item.update(
        mic_positions=item["geometry"].clone(),
        source_positions=torch.zeros(item["input_lengths"], 1, 3),
        output_source_positions=torch.zeros(item["output_lengths"], 1, 3),
    )
    ssl_collate_fn([item]).validate()

    typo = dict(item)
    typo["receptive_filed"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        ssl_collate_fn([typo])


def test_batch_validate_cross_checks_optional_provenance_fields():
    item = _item_with_provenance()
    batch = ssl_collate_fn([item])
    batch.validate()

    mismatches = (
        ("coordinate_frame", "room_xyz_m", "coordinate_frame"),
        ("sample_rate", 8000, "sample_rate"),
    )
    for field, value, match in mismatches:
        provenance = deepcopy(item["provenance"])
        provenance[field] = value
        with pytest.raises(ValueError, match=match):
            replace(batch, provenance=(provenance,)).validate()

    non_integral_rate = deepcopy(item["provenance"])
    non_integral_rate["sample_rate"] = 16000.0
    with pytest.raises(ValueError, match="sample_rate"):
        replace(batch, provenance=(non_integral_rate,)).validate()

    geometry = deepcopy(item["provenance"])
    geometry["geometry"]["positions_m"][0][0] = 0.1
    with pytest.raises(ValueError, match="geometry"):
        replace(batch, provenance=(geometry,)).validate()

    reference = deepcopy(item["provenance"])
    reference["geometry"]["reference_channel"] = 1
    with pytest.raises(ValueError, match="reference_channel"):
        replace(batch, provenance=(reference,)).validate()

    distances = deepcopy(item["provenance"])
    distances["source_distances"][0][0] = 99.0
    with pytest.raises(ValueError, match="source_distances"):
        replace(batch, provenance=(distances,)).validate()


def test_batch_validate_matches_unpadded_near_field_provenance_distances():
    short = _item_with_provenance(output_frames=2)
    long = _item_with_provenance(output_frames=3)
    batch = ssl_collate_fn([short, long])
    batch.validate()
    assert batch.source_distances.shape == (2, 3, 1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_frame_index", torch.tensor([0.0, 1.0, 2.0, 3.0])),
        ("input_frame_index", torch.tensor([0, 2, 1, 3], dtype=torch.int64)),
        ("output_frame_index", torch.tensor([0.0, 2.0])),
        ("output_frame_index", torch.tensor([0, 2, 2], dtype=torch.int64)),
        ("output_frame_index", torch.tensor([0, -2], dtype=torch.int64)),
    ],
)
def test_collate_rejects_invalid_item_frame_indices(field, value):
    item = _item()
    item[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        ssl_collate_fn([item])


def test_batch_validate_checks_frame_index_dtype_monotonicity_and_padding():
    batch = ssl_collate_fn([_item(input_frames=4, output_frames=2), _item(input_frames=6, output_frames=3)])

    with pytest.raises(TypeError, match="input_frame_index.*integer"):
        replace(batch, input_frame_index=batch.input_frame_index.to(torch.float32)).validate()

    non_monotonic = batch.input_frame_index.clone()
    non_monotonic[0, 1] = non_monotonic[0, 0]
    with pytest.raises(ValueError, match="input_frame_index.*increasing"):
        replace(batch, input_frame_index=non_monotonic).validate()

    padded = batch.input_frame_index.clone()
    padded[0, 4] = 1
    with pytest.raises(ValueError, match="input_frame_index.*zero"):
        replace(batch, input_frame_index=padded).validate()

    output_non_monotonic = batch.output_frame_index.clone()
    output_non_monotonic[0, 1] = output_non_monotonic[0, 0]
    with pytest.raises(ValueError, match="output_frame_index.*increasing"):
        replace(batch, output_frame_index=output_non_monotonic).validate()

    output_padded = batch.output_frame_index.clone()
    output_padded[0, 2] = 1
    with pytest.raises(ValueError, match="output_frame_index.*zero"):
        replace(batch, output_frame_index=output_padded).validate()


def test_batch_validate_rejects_negative_frame_origin():
    batch = ssl_collate_fn([_item()])
    with pytest.raises(ValueError, match="frame_origin.*non-negative"):
        replace(batch, frame_origin=torch.tensor([-1], dtype=torch.int64)).validate()
