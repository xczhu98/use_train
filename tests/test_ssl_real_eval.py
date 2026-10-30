"""SSL 真实录音验证 CLI 的输入、半平面指标和 LOCATA adapter 合同。"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.unit


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_real_eval_metrics_keep_two_mic_halfplane_contract():
    from tools.verify.ssl.real_eval import compute_azimuth_metrics, parse_azimuth_domain

    domain = parse_azimuth_domain("halfplane")
    assert domain == (-math.pi / 2, math.pi / 2)
    result = compute_azimuth_metrics(
        np.array([0.1, 0.2, -0.1]),
        np.array([0.1, 0.2, math.pi]),
        target_vad=np.ones(3),
        prediction_valid=np.ones(3, dtype=bool),
        domain=domain,
        vad_threshold=0.5,
        accuracy_threshold_deg=5.0,
    )
    assert result["valid_vad_frames"] == 2
    assert result["excluded_out_of_domain_frames"] == 1
    assert result["evaluated_frames"] == 2
    assert result["mae_deg"] == pytest.approx(0.0)


def test_real_eval_jsonl_record_rejects_missing_geometry(tmp_path: Path):
    from tools.verify.ssl.real_eval import record_from_mapping

    audio = tmp_path / "record.wav"
    audio.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="requires geometry"):
        record_from_mapping(
            {"audio": str(audio), "labels": {"azimuth_rad": 0.0}},
            base_dir=tmp_path,
        )


def test_real_eval_uses_canonical_simulation_sound_speed():
    from core.utils.config import load_merged_config

    from tools.verify.ssl.real_eval import _projection_from_config

    config = load_merged_config("configs/ssl/fixed_array_ipdnet_dynamic_moving.yaml")
    spec = SimpleNamespace(feature_contract="reference_ipd", field_mode="far_field")
    projection = _projection_from_config(config, spec, sample_rate=16000)
    assert projection.sound_speed == pytest.approx(343.0)


def test_locata_adapter_reads_single_source_and_selected_pair(tmp_path: Path):
    sf = pytest.importorskip("soundfile")

    from tools.verify.ssl.real_eval import iter_locata

    directory = tmp_path / "dev" / "task1" / "recording1" / "dicit"
    directory.mkdir(parents=True)
    array_fields = ["x", "y", "z", "ref_vec_x", "ref_vec_y", "ref_vec_z"] + [
        f"rotation_{i}{j}" for i in range(1, 4) for j in range(1, 4)
    ]
    array_rows = []
    for second in (0.0, 1.0):
        row = {field: 0.0 for field in array_fields}
        row.update({"rotation_11": 1.0, "rotation_22": 1.0, "rotation_33": 1.0})
        array_rows.append(row)
    _write_tsv(directory / "position_array_dicit.txt", array_fields, array_rows)
    _write_tsv(
        directory / "required_time.txt",
        ["hour", "minute", "second"],
        [{"hour": 0, "minute": 0, "second": 0}, {"hour": 0, "minute": 0, "second": 1}],
    )
    _write_tsv(
        directory / "position_source_1.txt",
        ["x", "y", "z"],
        [{"x": 1, "y": 0, "z": 0}, {"x": 1, "y": 1, "z": 0}],
    )
    _write_tsv(directory / "VAD_dicit_1.txt", ["VAD"], [{"VAD": 1}, {"VAD": 1}])
    sf.write(directory / "audio_array_dicit.wav", np.zeros((160, 15), dtype=np.float32), 16000)
    sf.write(directory / "audio_source_1.wav", np.zeros(160, dtype=np.float32), 16000)

    records = list(iter_locata(tmp_path, task=1, array="dicit", mic_indices=(0, 1)))
    assert len(records) == 1
    record = records[0]
    assert record.mic_positions.shape == (2, 3)
    assert record.channel_indices == (0, 1)
    assert record.label_times_s.tolist() == [0.0, 1.0]
    assert record.azimuth_rad[0] == pytest.approx(0.0)
    assert record.azimuth_rad[1] == pytest.approx(math.pi / 4)
