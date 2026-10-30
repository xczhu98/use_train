"""AEC stereo echo/ref 离线 GCC 预对齐生成器测试。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from tasks.aec.inference.gcc_phat_core import GccAlignEstimate
from tools.prepare_data_tools.generators.aec import prealign_stereo_echo_ref as prealign
from tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref import (
    RECORDS_NAME,
    SUMMARY_NAME,
    build_parser,
    main,
)

pytestmark = pytest.mark.unit


def _write_stereo(
    path: Path,
    echo: np.ndarray,
    farend: np.ndarray,
    sample_rate: int,
    *,
    format_name: str | None = None,
    subtype: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.stack([echo, farend], axis=1),
        sample_rate,
        format=format_name,
        subtype=subtype,
    )


def _run(input_root: Path, output_root: Path, *extra: str) -> int:
    return main(
        [
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--workers",
            "1",
            *extra,
        ]
    )


def _mark_owned(root: Path) -> None:
    (root / prealign.OWNER_FILE).write_text(
        json.dumps(
            {
                "generator": prealign.GENERATOR_ID,
                "schema_version": prealign.OWNER_SCHEMA,
            }
        ),
        encoding="utf-8",
    )


def _constant_pair(sample_rate: int = 16000, delay_samples: int = 640):
    rng = np.random.default_rng(42)
    farend = (0.2 * rng.standard_normal(sample_rate * 2)).astype(np.float32)
    echo = np.zeros_like(farend)
    echo[delay_samples:] = 0.7 * farend[:-delay_samples]
    return echo, farend


def test_default_limits_match_current_production_aec_configs():
    args = build_parser().parse_args(
        ["--input-root", "/path/to/input", "--output-root", "/path/to/output"]
    )
    assert args.max_tau_seconds == 0.3
    assert args.max_pad_seconds == 0.3
    assert args.bias_seconds == 0.001
    assert args.peak_threshold == 0.05
    assert args.max_residual_error_ms == 5.0
    assert args.on_residual_failure == "copy"
    assert args.output_sample_rate is None
    assert args.output_format is None
    assert args.output_subtype is None


def test_constant_delay_preserves_echo_length_and_passes_staged_residual(
    tmp_path: Path,
):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = input_root / "nested" / "pair.wav"
    echo, farend = _constant_pair()
    _write_stereo(source, echo, farend, 16000, subtype="PCM_16")
    source_audio, _ = sf.read(str(source), dtype="float32", always_2d=True)

    assert _run(input_root, output_root, "--bias-seconds", "0") == 0

    target = output_root / "nested" / "pair.wav"
    output_audio, sample_rate = sf.read(str(target), dtype="float32", always_2d=True)
    record = json.loads((output_root / RECORDS_NAME).read_text(encoding="utf-8"))
    delay = record["applied"]["tau_samples"]
    assert sample_rate == 16000
    assert output_audio.shape == source_audio.shape
    np.testing.assert_array_equal(output_audio[:, 0], source_audio[:, 0])
    assert delay == pytest.approx(640, abs=2)
    np.testing.assert_array_equal(output_audio[:delay, 1], 0.0)
    np.testing.assert_allclose(
        output_audio[delay:, 1], source_audio[:-delay, 1], atol=1.0 / 32768.0
    )
    assert record["generator_version"] == "3.2"
    assert record["policy"] == "apply_constant_front_pad_crop"
    assert record["estimate"]["low_confidence"] is False
    residual = record["residual_check"]
    assert residual["scope"] == "full_pair"
    assert residual["checked"] is True
    assert residual["expected_tau_ms"] == pytest.approx(0.0, abs=0.2)
    assert residual["estimated_tau_ms"] == pytest.approx(
        residual["expected_tau_ms"], abs=0.5
    )
    assert residual["error_ms"] <= 0.5
    assert residual["low_confidence"] is False
    assert residual["time_varying"] is False
    assert residual["passed"] is True

    summary = json.loads((output_root / SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["residual_checked_count"] == 1
    assert summary["residual_passed_count"] == 1
    assert summary["residual_error_ms"] == {
        "max": pytest.approx(residual["error_ms"]),
        "mean": pytest.approx(residual["error_ms"]),
        "p50": pytest.approx(residual["error_ms"]),
        "p95": pytest.approx(residual["error_ms"]),
    }


def test_time_varying_applies_first_reliable_tau_once_and_preserves_drift(
    tmp_path: Path,
):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = input_root / "pair.wav"
    echo, farend = _constant_pair(sample_rate=1000, delay_samples=4)
    _write_stereo(source, echo, farend, 1000, subtype="FLOAT")
    source_audio, _ = sf.read(str(source), dtype="float32", always_2d=True)
    initial = GccAlignEstimate(
        tau_sec=0.012,
        trajectory=((1.0, 0.020), (0.0, 0.004)),
        time_varying=True,
        low_confidence=False,
        mode="windowed_full_scan",
        peak=0.4,
        prominence=9.0,
    )
    residual = GccAlignEstimate(
        tau_sec=0.01,
        trajectory=((0.0, 0.0), (1.0, 0.02)),
        time_varying=True,
        low_confidence=False,
        mode="windowed_full_scan",
        peak=0.5,
        prominence=10.0,
    )
    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        side_effect=[initial, residual],
    ) as estimate_mock:
        assert _run(input_root, output_root, "--bias-seconds", "0") == 0

    record = json.loads((output_root / RECORDS_NAME).read_text(encoding="utf-8"))
    output_audio, _ = sf.read(
        str(output_root / "pair.wav"), dtype="float32", always_2d=True
    )
    np.testing.assert_array_equal(output_audio[:, 0], source_audio[:, 0])
    np.testing.assert_array_equal(output_audio[:4, 1], 0.0)
    np.testing.assert_array_equal(output_audio[4:, 1], source_audio[:-4, 1])
    assert record["policy"] == "apply_time_varying_initial_front_pad_crop"
    assert record["applied"]["tau_samples"] == 4
    assert record["applied"]["tau_seconds"] == pytest.approx(0.004)
    assert record["applied"]["trajectory"] is None
    assert record["residual_check"]["scope"] == "initial_window"
    assert record["residual_check"]["expected_tau_ms"] == pytest.approx(0.0)
    assert record["residual_check"]["time_varying"] is True
    assert record["residual_check"]["passed"] is True
    assert estimate_mock.call_count == 2
    residual_farend, residual_echo, residual_sr = estimate_mock.call_args_list[1].args
    assert residual_sr == 1000
    assert residual_farend.size == residual_echo.size == 1000


@pytest.mark.parametrize(
    ("suffix", "format_name", "subtype"),
    [(".wav", "WAV", "PCM_16"), (".flac", "FLAC", "PCM_16")],
)
def test_low_confidence_is_byte_identical_and_preserves_nested_header(
    tmp_path: Path, suffix: str, format_name: str, subtype: str
):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = input_root / "corpus" / f"pair{suffix}"
    rng = np.random.default_rng(7)
    _write_stereo(
        source,
        rng.normal(0, 0.1, 4000).astype(np.float32),
        rng.normal(0, 0.1, 4000).astype(np.float32),
        16000,
        format_name=format_name,
        subtype=subtype,
    )
    estimate = GccAlignEstimate(
        tau_sec=0.0,
        trajectory=None,
        time_varying=False,
        low_confidence=True,
        mode="low_confidence",
        peak=0.0,
        prominence=0.0,
    )
    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        return_value=estimate,
    ):
        assert _run(input_root, output_root) == 0

    target = output_root / "corpus" / source.name
    assert target.read_bytes() == source.read_bytes()
    source_info = sf.info(str(source))
    target_info = sf.info(str(target))
    assert (
        target_info.samplerate,
        target_info.channels,
        target_info.frames,
        target_info.format,
        target_info.subtype,
        target_info.endian,
    ) == (
        source_info.samplerate,
        source_info.channels,
        source_info.frames,
        source_info.format,
        source_info.subtype,
        source_info.endian,
    )
    record = json.loads((output_root / RECORDS_NAME).read_text(encoding="utf-8"))
    assert record["policy"] == "copy_low_confidence"
    assert record["residual_check"] == {
        "checked": False,
        "passed": False,
        "reason": "original_estimate_low_confidence",
        "scope": None,
    }
    summary = json.loads((output_root / SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["residual_checked_count"] == 0
    assert summary["residual_passed_count"] == 0
    assert summary["residual_error_ms"] == {
        "max": None,
        "mean": None,
        "p50": None,
        "p95": None,
    }


@pytest.mark.parametrize(
    "residual_estimate",
    [
        GccAlignEstimate(0.008, None, False, False, "windowed_fast", 0.5, 10.0),
        GccAlignEstimate(0.0, None, False, True, "low_confidence", 0.0, 0.0),
        GccAlignEstimate(
            0.0,
            ((0.0, 0.0), (1.0, 0.02)),
            True,
            False,
            "windowed_full_scan",
            0.5,
            10.0,
        ),
    ],
    ids=["wrong_tau", "low_confidence", "time_varying"],
)
def test_constant_residual_gate_failure_preserves_old_target_and_has_no_sidecars(
    tmp_path: Path, residual_estimate: GccAlignEstimate
):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = input_root / "pair.wav"
    echo, farend = _constant_pair(sample_rate=1000, delay_samples=10)
    _write_stereo(source, echo, farend, 1000, subtype="FLOAT")
    target = output_root / "pair.wav"
    _write_stereo(
        target,
        np.full(2000, 0.1, dtype=np.float32),
        np.full(2000, -0.1, dtype=np.float32),
        1000,
        subtype="FLOAT",
    )
    old_bytes = target.read_bytes()
    original = GccAlignEstimate(
        0.01, None, False, False, "windowed_fast", 0.5, 10.0
    )

    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        side_effect=[original, residual_estimate],
    ):
        assert _run(
            input_root,
            output_root,
            "--overwrite",
            "--bias-seconds",
            "0",
            "--max-residual-error-ms",
            "2",
            "--on-residual-failure",
            "abort",
        ) == 1

    assert target.read_bytes() == old_bytes
    assert not (output_root / SUMMARY_NAME).exists()
    assert not (output_root / RECORDS_NAME).exists()


def test_time_varying_without_reliable_initial_point_fails_closed(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    echo, farend = _constant_pair(sample_rate=1000, delay_samples=10)
    _write_stereo(input_root / "pair.wav", echo, farend, 1000, subtype="FLOAT")
    invalid = GccAlignEstimate(
        0.01, None, True, False, "windowed_full_scan", 0.5, 10.0
    )
    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        return_value=invalid,
    ):
        assert _run(input_root, output_root, "--bias-seconds", "0") == 1

    assert not (output_root / "pair.wav").exists()
    assert not (output_root / SUMMARY_NAME).exists()
    assert not (output_root / RECORDS_NAME).exists()


@pytest.mark.parametrize(
    "residual_estimate",
    [
        GccAlignEstimate(0.008, None, False, False, "windowed_fast", 0.5, 10.0),
        GccAlignEstimate(0.0, None, False, True, "low_confidence", 0.0, 0.0),
    ],
    ids=["wrong_initial_tau", "low_confidence"],
)
def test_time_varying_initial_residual_gate_failure_publishes_nothing(
    tmp_path: Path, residual_estimate: GccAlignEstimate
):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    echo, farend = _constant_pair(sample_rate=1000, delay_samples=10)
    _write_stereo(input_root / "pair.wav", echo, farend, 1000, subtype="FLOAT")
    original = GccAlignEstimate(
        0.02,
        ((0.0, 0.01), (1.0, 0.03)),
        True,
        False,
        "windowed_full_scan",
        0.5,
        10.0,
    )
    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        side_effect=[original, residual_estimate],
    ):
        assert _run(
            input_root, output_root, "--bias-seconds", "0",
            "--on-residual-failure", "abort",
        ) == 1

    assert not (output_root / "pair.wav").exists()
    assert not (output_root / SUMMARY_NAME).exists()
    assert not (output_root / RECORDS_NAME).exists()


def test_residual_gate_failure_copy_policy_keeps_unaligned_sample(tmp_path: Path):
    """copy（默认）：残差门失败文件改写为不对齐原始音频，任务继续。"""
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = input_root / "pair.wav"
    echo, farend = _constant_pair(sample_rate=1000, delay_samples=10)
    _write_stereo(source, echo, farend, 1000, subtype="FLOAT")
    original = GccAlignEstimate(
        0.01, None, False, False, "windowed_fast", 0.5, 10.0
    )
    wrong_residual = GccAlignEstimate(
        0.008, None, False, False, "windowed_fast", 0.5, 10.0
    )
    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        side_effect=[original, wrong_residual],
    ):
        assert _run(
            input_root, output_root, "--bias-seconds", "0",
            "--max-residual-error-ms", "2",
        ) == 0

    target = output_root / "pair.wav"
    # 回退后目标与源 byte-identical（未施加任何对齐）
    assert target.read_bytes() == source.read_bytes()
    record = json.loads((output_root / RECORDS_NAME).read_text(encoding="utf-8"))
    assert record["policy"] == "copy_residual_gate_failure"
    assert record["applied"]["tau_samples"] == 0
    assert record["status"] == "success"
    residual = record["residual_check"]
    assert residual["checked"] is True
    assert residual["passed"] is False
    assert residual["scope"] == "full_pair"
    assert residual["error_ms"] == pytest.approx(8.0)
    summary = json.loads((output_root / SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["success_count"] == 1
    assert summary["policy_counts"] == {"copy_residual_gate_failure": 1}
    assert summary["residual_checked_count"] == 1
    assert summary["residual_passed_count"] == 0


def test_residual_gate_failure_copy_policy_with_conversion_reencodes(
    tmp_path: Path,
):
    """copy + 转换：回退文件同样重采样重编码，只是不对齐。"""
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = input_root / "pair.wav"
    echo, farend = _constant_pair(sample_rate=16000, delay_samples=640)
    _write_stereo(source, echo, farend, 16000, subtype="PCM_16")
    original = GccAlignEstimate(
        0.04, None, False, False, "windowed_fast", 0.5, 10.0
    )
    wrong_residual = GccAlignEstimate(
        0.02, None, False, False, "windowed_fast", 0.5, 10.0
    )
    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        side_effect=[original, wrong_residual],
    ):
        assert _run(
            input_root,
            output_root,
            "--bias-seconds",
            "0",
            "--max-residual-error-ms",
            "2",
            "--output-sample-rate",
            "32000",
            "--output-format",
            "FLAC",
        ) == 0

    target = output_root / "pair.flac"
    assert target.read_bytes()[:4] == b"fLaC"
    output_audio, sample_rate = sf.read(str(target), dtype="float32", always_2d=True)
    assert sample_rate == 32000
    assert output_audio.shape == (64000, 2)
    # 未对齐：farend 通道为源信号的重采样，而非前补零后的对齐结果
    source_audio, _ = sf.read(str(source), dtype="float32", always_2d=True)
    expected_farend = prealign.resample_audio(source_audio[:, 1], 16000, 32000)
    np.testing.assert_allclose(
        output_audio[:, 1], expected_farend, atol=1.0 / 32768.0
    )
    record = json.loads((output_root / RECORDS_NAME).read_text(encoding="utf-8"))
    assert record["policy"] == "copy_residual_gate_failure"
    assert record["output"]["sample_rate"] == 32000
    assert record["residual_check"]["passed"] is False


def test_residual_early_failure_copy_policy_summary_has_no_error_ms(tmp_path: Path):
    """copy 回退的早期失败结果没有 error_ms 字段，summary 聚合不得崩溃。"""
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = input_root / "pair.wav"
    echo, farend = _constant_pair(sample_rate=1000, delay_samples=10)
    _write_stereo(source, echo, farend, 1000, subtype="FLOAT")
    original = GccAlignEstimate(
        0.01, None, False, False, "windowed_fast", 0.5, 10.0
    )
    low_conf_residual = GccAlignEstimate(
        0.0, None, False, True, "low_confidence", 0.0, 0.0
    )
    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        side_effect=[original, low_conf_residual],
    ):
        assert _run(input_root, output_root, "--bias-seconds", "0") == 0

    assert (output_root / "pair.wav").read_bytes() == source.read_bytes()
    record = json.loads((output_root / RECORDS_NAME).read_text(encoding="utf-8"))
    assert record["policy"] == "copy_residual_gate_failure"
    assert record["residual_check"]["reason"] == "residual_low_confidence"
    assert "error_ms" not in record["residual_check"]
    summary = json.loads((output_root / SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["residual_checked_count"] == 1
    assert summary["residual_passed_count"] == 0
    assert summary["residual_error_ms"] == {
        "max": None,
        "mean": None,
        "p50": None,
        "p95": None,
    }


@pytest.mark.parametrize("invalid_kind", ["mono", "nonfinite"])
def test_invalid_audio_fails_without_completion_marker(
    tmp_path: Path, invalid_kind: str
):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    source = input_root / "bad.wav"
    if invalid_kind == "mono":
        sf.write(str(source), np.ones(100, dtype=np.float32), 16000)
    else:
        audio = np.zeros((100, 2), dtype=np.float32)
        audio[10, 0] = np.nan
        sf.write(str(source), audio, 16000, format="WAV", subtype="FLOAT")

    assert _run(input_root, output_root) == 1
    assert not (output_root / SUMMARY_NAME).exists()
    assert not (output_root / RECORDS_NAME).exists()


def test_rejects_overlapping_roots_and_nonempty_output_without_overwrite(
    tmp_path: Path,
):
    input_root = tmp_path / "input"
    input_root.mkdir()
    _write_stereo(
        input_root / "pair.wav",
        np.zeros(1000, dtype=np.float32),
        np.ones(1000, dtype=np.float32) * 0.1,
        16000,
    )

    assert _run(input_root, input_root / "nested-output", "--dry-run") == 1
    assert _run(input_root, input_root.parent, "--dry-run") == 1

    output_root = tmp_path / "output"
    output_root.mkdir()
    sentinel = output_root / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    assert _run(input_root, output_root) == 1
    assert sentinel.read_text(encoding="utf-8") == "keep"

    estimate = GccAlignEstimate(0.0, None, False, True, "low_confidence", 0.0, 0.0)
    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        return_value=estimate,
    ):
        assert _run(input_root, output_root, "--overwrite") == 1
    assert sentinel.read_text(encoding="utf-8") == "keep"

    _mark_owned(output_root)
    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        return_value=estimate,
    ):
        assert _run(input_root, output_root, "--overwrite") == 0
    assert not sentinel.exists()
    assert {path.name for path in output_root.iterdir()} == {
        "pair.wav",
        RECORDS_NAME,
        SUMMARY_NAME,
        prealign.OWNER_FILE,
    }


def test_hard_failure_after_partial_audio_preserves_complete_old_collection(
    tmp_path: Path,
):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    echo, farend = _constant_pair(delay_samples=320)
    _write_stereo(input_root / "a_good.wav", echo, farend, 16000)
    input_root.mkdir(parents=True, exist_ok=True)
    sf.write(str(input_root / "z_bad.wav"), np.ones(100, dtype=np.float32), 16000)
    output_root.mkdir()
    records_path = output_root / RECORDS_NAME
    summary_path = output_root / SUMMARY_NAME
    records_path.write_text("stale records\n", encoding="utf-8")
    summary_path.write_text("stale summary\n", encoding="utf-8")
    old_records = records_path.read_bytes()
    old_summary = summary_path.read_bytes()

    assert _run(
        input_root, output_root, "--overwrite", "--bias-seconds", "0"
    ) == 1
    assert not (output_root / "a_good.wav").exists()
    assert records_path.read_bytes() == old_records
    assert summary_path.read_bytes() == old_summary
    assert not list(tmp_path.glob(".output.owned-staging.*"))


def test_overwrite_exchange_failure_preserves_complete_old_collection(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    echo, farend = _constant_pair(delay_samples=320)
    source = input_root / "pair.wav"
    _write_stereo(source, echo, farend, 16000)
    output_root.mkdir()
    _mark_owned(output_root)
    sentinel = output_root / "keep.bin"
    sentinel.write_bytes(b"old collection")
    estimate = GccAlignEstimate(0.0, None, False, True, "low_confidence", 0.0, 0.0)

    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        return_value=estimate,
    ), patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref._exchange_directories",
        side_effect=OSError("injected exchange failure"),
    ):
        assert _run(input_root, output_root, "--overwrite") == 1

    assert {path.name for path in output_root.iterdir()} == {
        "keep.bin", prealign.OWNER_FILE
    }
    assert sentinel.read_bytes() == b"old collection"
    assert not list(tmp_path.glob(".output.owned-staging.*"))


def test_overwrite_commit_fsync_failure_rolls_back_old_collection(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    echo, farend = _constant_pair(delay_samples=320)
    _write_stereo(input_root / "pair.wav", echo, farend, 16000)
    output_root.mkdir()
    _mark_owned(output_root)
    sentinel = output_root / "keep.bin"
    sentinel.write_bytes(b"old collection")
    estimate = GccAlignEstimate(0.0, None, False, True, "low_confidence", 0.0, 0.0)
    real_fsync_directory = prealign._fsync_directory
    parent_fsync_calls = 0

    def fail_first_parent_fsync(path: Path) -> None:
        nonlocal parent_fsync_calls
        if path == output_root.parent:
            parent_fsync_calls += 1
            if parent_fsync_calls == 1:
                raise OSError("injected commit fsync failure")
        real_fsync_directory(path)

    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        return_value=estimate,
    ), patch.object(prealign, "_fsync_directory", side_effect=fail_first_parent_fsync):
        assert _run(input_root, output_root, "--overwrite") == 1

    assert parent_fsync_calls == 2
    assert {path.name for path in output_root.iterdir()} == {
        "keep.bin", prealign.OWNER_FILE
    }
    assert sentinel.read_bytes() == b"old collection"
    assert not list(tmp_path.glob(".output.owned-staging.*"))


def test_input_identity_change_before_publish_fails_without_public_output(
    tmp_path: Path,
):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    echo, farend = _constant_pair(delay_samples=320)
    _write_stereo(input_root / "pair.wav", echo, farend, 16000)
    estimate = GccAlignEstimate(0.0, None, False, True, "low_confidence", 0.0, 0.0)

    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        return_value=estimate,
    ), patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref._verify_source_identity",
        side_effect=[None, RuntimeError("input identity changed before publish")],
    ):
        assert _run(input_root, output_root) == 1

    assert not output_root.exists()
    assert not list(tmp_path.glob(".output.owned-staging.*"))


def test_dry_run_only_scans_and_does_not_create_output(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "not-created"
    input_root.mkdir()
    _write_stereo(
        input_root / "pair.wav",
        np.zeros(100, dtype=np.float32),
        np.zeros(100, dtype=np.float32),
        16000,
    )

    assert _run(input_root, output_root, "--dry-run", "--limit", "1") == 0
    assert not output_root.exists()


def test_rejects_output_symlink_ancestor(tmp_path: Path):
    input_root = tmp_path / "input"
    referent = tmp_path / "referent"
    link = tmp_path / "output-link"
    input_root.mkdir()
    referent.mkdir()
    link.symlink_to(referent, target_is_directory=True)
    _write_stereo(
        input_root / "pair.wav",
        np.zeros(100, dtype=np.float32),
        np.zeros(100, dtype=np.float32),
        16000,
    )

    assert _run(input_root, link / "nested", "--dry-run") == 1
    assert not list(referent.iterdir())


def test_transformed_flac_preserves_header(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = input_root / "pair.flac"
    farend = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    echo = farend.copy()
    _write_stereo(source, echo, farend, 16000, format_name="FLAC", subtype="PCM_16")
    estimate = GccAlignEstimate(0.01, None, False, False, "windowed_fast", 0.5, 10.0)
    residual = GccAlignEstimate(0.0, None, False, False, "windowed_fast", 0.5, 10.0)
    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        side_effect=[estimate, residual],
    ):
        assert _run(input_root, output_root, "--bias-seconds", "0") == 0

    source_info = sf.info(str(source))
    target_info = sf.info(str(output_root / "pair.flac"))
    assert (
        target_info.samplerate,
        target_info.channels,
        target_info.frames,
        target_info.format,
        target_info.subtype,
        target_info.endian,
    ) == (
        source_info.samplerate,
        source_info.channels,
        source_info.frames,
        source_info.format,
        source_info.subtype,
        source_info.endian,
    )


def test_process_pool_returns_records_in_relative_path_order(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    for name in ("z.wav", "a.wav"):
        _write_stereo(
            input_root / name,
            np.zeros(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            16000,
        )

    assert main(
        [
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--workers",
            "2",
        ]
    ) == 0
    records = [
        json.loads(line)
        for line in (output_root / RECORDS_NAME).read_text(encoding="utf-8").splitlines()
    ]
    assert [record["relative_path"] for record in records] == ["a.wav", "z.wav"]


def test_convert_to_flac_32k_replaces_extension_and_applies_delay(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = input_root / "nested" / "pair.wav"
    echo, farend = _constant_pair(sample_rate=16000, delay_samples=640)
    _write_stereo(source, echo, farend, 16000, subtype="PCM_16")

    assert _run(
        input_root,
        output_root,
        "--bias-seconds",
        "0",
        "--output-sample-rate",
        "32000",
        "--output-format",
        "FLAC",
    ) == 0

    target = output_root / "nested" / "pair.flac"
    assert target.exists()
    assert not (output_root / "nested" / "pair.wav").exists()
    # fLaC magic：确为真实 FLAC 容器而非改后缀的 WAV
    assert target.read_bytes()[:4] == b"fLaC"
    info = sf.info(str(target))
    assert (info.samplerate, info.channels, info.format) == (32000, 2, "FLAC")
    assert info.subtype == "PCM_16"
    assert info.frames == 64000  # 2s @ 32k

    output_audio, sample_rate = sf.read(str(target), dtype="float32", always_2d=True)
    assert sample_rate == 32000
    # 40ms 延迟在 32k 下对应 1280 个前补零（resample 滤波器边缘允许少量过渡）
    first_nonzero = int(np.argmax(np.abs(output_audio[:, 1]) > 0.05))
    assert abs(first_nonzero - 1280) <= 64

    record = json.loads((output_root / RECORDS_NAME).read_text(encoding="utf-8"))
    assert record["relative_path"] == "nested/pair.flac"
    assert record["sample_rate"] == 16000  # 顶层字段记录源 header
    assert record["applied"]["tau_samples"] == pytest.approx(640, abs=2)
    assert record["output"] == {
        "sample_rate": 32000,
        "frames": 64000,
        "format": "FLAC",
        "subtype": "PCM_16",
    }
    assert record["policy"] == "apply_constant_front_pad_crop"
    assert record["residual_check"]["checked"] is True
    assert record["residual_check"]["passed"] is True


def test_low_confidence_with_conversion_reencodes_instead_of_byte_copy(
    tmp_path: Path,
):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = input_root / "pair.wav"
    rng = np.random.default_rng(7)
    _write_stereo(
        source,
        rng.normal(0, 0.1, 8000).astype(np.float32),
        rng.normal(0, 0.1, 8000).astype(np.float32),
        16000,
        subtype="PCM_16",
    )
    estimate = GccAlignEstimate(
        tau_sec=0.0,
        trajectory=None,
        time_varying=False,
        low_confidence=True,
        mode="low_confidence",
        peak=0.0,
        prominence=0.0,
    )
    with patch(
        "tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref.estimate_gcc_tau_windowed",
        return_value=estimate,
    ):
        assert _run(
            input_root,
            output_root,
            "--output-sample-rate",
            "32000",
            "--output-format",
            "FLAC",
        ) == 0

    target = output_root / "pair.flac"
    assert target.read_bytes()[:4] == b"fLaC"
    assert target.read_bytes() != source.read_bytes()
    info = sf.info(str(target))
    assert (info.samplerate, info.channels, info.frames, info.format) == (
        32000,
        2,
        16000,
        "FLAC",
    )
    record = json.loads((output_root / RECORDS_NAME).read_text(encoding="utf-8"))
    assert record["policy"] == "copy_low_confidence"
    assert record["residual_check"]["checked"] is False
    assert record["output"]["sample_rate"] == 32000


@pytest.mark.parametrize(
    "extra",
    [
        ("--output-subtype", "PCM_16"),  # subtype 无 format
        ("--output-format", "NOT_A_FORMAT"),  # 未知格式
        ("--output-format", "FLAC", "--output-subtype", "FLOAT"),  # 非法组合
        ("--output-sample-rate", "0"),  # 非法采样率
    ],
    ids=[
        "subtype_without_format",
        "unknown_format",
        "flac_float_subtype",
        "zero_sample_rate",
    ],
)
def test_invalid_output_conversion_args_fail_fast(tmp_path: Path, extra: tuple):
    input_root = tmp_path / "input"
    input_root.mkdir()
    _write_stereo(
        input_root / "pair.wav",
        np.zeros(1000, dtype=np.float32),
        np.ones(1000, dtype=np.float32) * 0.1,
        16000,
    )
    output_root = tmp_path / "output"
    assert _run(input_root, output_root, *extra) == 1
    assert not output_root.exists()


def test_extension_replacement_collision_fails_closed(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    echo, farend = _constant_pair()
    _write_stereo(input_root / "pair.wav", echo, farend, 16000)
    _write_stereo(
        input_root / "pair.flac", echo, farend, 16000, format_name="FLAC"
    )

    assert _run(input_root, output_root, "--output-format", "FLAC") == 1
    assert not output_root.exists()
