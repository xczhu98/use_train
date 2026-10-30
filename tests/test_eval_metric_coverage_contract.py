"""SE/AEC 指标覆盖率与 RESULTS 幂等写入合同测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import evaluate.common as eval_common
import evaluate.eval_pipeline_aec as pipeline_aec
import evaluate.eval_pipeline_common as pipeline_common
from evaluate.aggregate_eval_summary import summarize_one_output_dir
from evaluate.common import (
    append_extra_metrics_to_results,
    apply_metric_coverage,
    build_done_summary,
    coverage_status,
    load_metric_coverage,
    metric_coverage_status,
    publish_result_manifest,
)


def _inference_coverage(*, evaluated: int = 3) -> dict[str, int]:
    return {"expected": 3, "produced": 3, "failed": 0, "evaluated": evaluated}


def test_metric_coverage_uses_real_results_and_per_utterance_counts(
    tmp_path: Path,
) -> None:
    (tmp_path / "RESULTS.txt").write_text(
        "PESQ: 2.1000\n"
        "STOI: 0.9000\n"
        "WER: 0.1000\n"
        "_count_PESQ: 2\n"
        "_count_STOI: 1\n",
        encoding="utf-8",
    )
    (tmp_path / "per_utterance.json").write_text(
        json.dumps({
            "records": {
                "u1": {"PESQ": 2.0, "STOI": 0.9, "WER": 0.1},
                "u2": {"PESQ": 2.2, "WER": 0.1},
            }
        }),
        encoding="utf-8",
    )

    metric_coverage = load_metric_coverage(
        tmp_path, ["PESQ", "STOI", "WER"], eligible=3
    )

    assert metric_coverage["PESQ"] == {
        "eligible": 3,
        "attempted": 3,
        "computed": 2,
        "failed": 1,
        "coverage": pytest.approx(2 / 3),
        "status": "partial",
    }
    assert metric_coverage["STOI"]["computed"] == 1
    assert metric_coverage["WER"]["computed"] == 2
    assert metric_coverage_status(metric_coverage) == "partial"
    assert apply_metric_coverage(_inference_coverage(), metric_coverage)[
        "evaluated"
    ] == 1


def test_missing_structured_metric_results_is_unknown_and_not_done(
    tmp_path: Path,
) -> None:
    metric_coverage = load_metric_coverage(tmp_path, ["PESQ"], eligible=3)
    coverage = apply_metric_coverage(_inference_coverage(), metric_coverage)

    assert metric_coverage["PESQ"]["status"] == "unknown"
    assert metric_coverage_status(metric_coverage) == "unknown"
    assert coverage == {
        "expected": 3,
        "produced": 3,
        "failed": 0,
        "evaluated": 0,
    }
    assert coverage_status(
        coverage, allow_partial=False, metric_coverage=metric_coverage
    ) == "partial"

    summary = build_done_summary(
        tmp_path / "enhanced",
        tmp_path,
        coverage=coverage,
        metric_coverage=metric_coverage,
    )
    assert summary["status"] == "partial"
    assert summary["metric_coverage_status"] == "unknown"
    assert set(summary["coverage"]) == {"expected", "produced", "failed", "evaluated"}


def test_complete_metric_counts_allow_done_status(tmp_path: Path) -> None:
    (tmp_path / "RESULTS.txt").write_text(
        "PESQ: 2.1000\n_count_PESQ: 3\n", encoding="utf-8"
    )
    metric_coverage = load_metric_coverage(tmp_path, ["PESQ"], eligible=3)
    coverage = apply_metric_coverage(_inference_coverage(), metric_coverage)

    assert metric_coverage_status(metric_coverage) == "done"
    assert coverage_status(
        coverage, allow_partial=False, metric_coverage=metric_coverage
    ) == "done"


def test_empty_replace_clears_stale_results(tmp_path: Path) -> None:
    results = tmp_path / "RESULTS.txt"
    results.write_text("STALE_METRIC: 9.0000\n", encoding="utf-8")

    append_extra_metrics_to_results(tmp_path, {}, replace=True)

    assert results.exists()
    assert results.read_text(encoding="utf-8") == ""


def test_extra_only_results_repeated_run_replaces_stale_content(tmp_path: Path) -> None:
    results = tmp_path / "RESULTS.txt"
    results.write_text(
        "STALE_METRIC: 9.0000\nERLE_dB: 1.0000\n", encoding="utf-8"
    )

    append_extra_metrics_to_results(
        tmp_path, {"ERLE_dB": 2.0, "AECMOS_DEG": 3.0}, replace=True
    )
    append_extra_metrics_to_results(
        tmp_path, {"ERLE_dB": 4.0, "AECMOS_DEG": 5.0}, replace=True
    )

    assert results.read_text(encoding="utf-8") == (
        "ERLE_dB: 4.0000\nAECMOS_DEG: 5.0000\n"
    )


def test_aec_sim_clears_extra_only_results_before_metric_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    testset_root = tmp_path / "testset"
    testset_root.mkdir()
    eval_root = tmp_path / "eval"
    scenario_dir = eval_root / "aec_sim" / "prob_echo_only"
    enhanced_dir = scenario_dir / "enhanced"
    enhanced_dir.mkdir(parents=True)
    (enhanced_dir / "inf.scp").write_text("u1 /tmp/u1.wav\n", encoding="utf-8")
    stale_results = scenario_dir / "RESULTS.txt"
    stale_results.write_text("OLD_RUN: 9.0000\n", encoding="utf-8")

    args = SimpleNamespace(
        step="evaluate",
        aec_sim_testset_root=str(testset_root),
        allow_partial_inference=True,
        audio_quality=False,
        downstream=False,
        device="cpu",
    )
    coverage = {"expected": 1, "produced": 1, "failed": 0, "evaluated": 0}
    monkeypatch.setattr(
        pipeline_aec, "ensure_inf_scp", lambda _path: enhanced_dir / "inf.scp"
    )
    monkeypatch.setattr(
        pipeline_aec, "load_inference_coverage", lambda *_args, **_kwargs: dict(coverage)
    )

    def fail_metric(*_args, **_kwargs):
        assert stale_results.read_text(encoding="utf-8") == ""
        raise RuntimeError("metric failed")

    monkeypatch.setattr(
        pipeline_aec, "compute_registered_scenario_metrics", fail_metric
    )
    summary = {}
    scp_paths = {
        "aec_sim": {
            "prob_echo_only": {
                "mic": tmp_path / "mic.scp",
                "farend": tmp_path / "far.scp",
                "echo": tmp_path / "echo.scp",
            }
        }
    }

    with pytest.raises(RuntimeError, match="metric failed"):
        pipeline_aec.process_aec_sim(
            args,
            tmp_path / "config.yaml",
            tmp_path / "model.tar",
            eval_root,
            summary,
            scp_paths,
        )

    assert stale_results.exists()
    assert stale_results.read_text(encoding="utf-8") == ""


def test_registered_scenario_metrics_dispatch_existing_aec_implementations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple, dict]] = []

    def fake_erle(mic_scp, enhanced_scp, ref_scp, scenario, *, echo_scp=None, **kwargs):
        calls.append(
            ("erle", (mic_scp, enhanced_scp, ref_scp, scenario), {"echo_scp": echo_scp, **kwargs})
        )
        return {"mean_erle_db": 5.0, "n_samples": 2}

    def fake_noise_reduction(mic_scp, enhanced_scp, scenario, *, eps=1e-10):
        calls.append(("noise_reduction", (mic_scp, enhanced_scp, scenario), {"eps": eps}))
        return {"mean_nr_db": 3.0, "n_samples": 2}

    monkeypatch.setattr(pipeline_common, "compute_erle_for_scenario", fake_erle)
    monkeypatch.setattr(
        pipeline_common, "compute_noise_reduction_for_scenario", fake_noise_reduction
    )

    echo_metrics = pipeline_common.compute_registered_scenario_metrics(
        "prob_target_echo",
        "mic.scp",
        "enhanced.scp",
        ref_scp="nearend.scp",
        echo_scp="echo.scp",
    )
    noise_metrics = pipeline_common.compute_registered_scenario_metrics(
        "prob_noise_only", "noise.scp", "enhanced_noise.scp"
    )

    assert echo_metrics == {"erle": {"mean_erle_db": 5.0, "n_samples": 2}}
    assert noise_metrics == {"noise_reduction": {"mean_nr_db": 3.0, "n_samples": 2}}
    assert calls == [
        (
            "erle",
            ("mic.scp", "enhanced.scp", "nearend.scp", "prob_target_echo"),
            {"echo_scp": "echo.scp", "eps": 1e-10},
        ),
        (
            "noise_reduction",
            ("noise.scp", "enhanced_noise.scp", "prob_noise_only"),
            {"eps": 1e-10},
        ),
    ]


def test_standard_results_preserve_base_and_update_extra_idempotently(
    tmp_path: Path,
) -> None:
    results = tmp_path / "RESULTS.txt"
    results.write_text("PESQ: 2.0000\n_count_PESQ: 1\n", encoding="utf-8")

    append_extra_metrics_to_results(tmp_path, {"AECMOS_DEG": 3.0})
    append_extra_metrics_to_results(tmp_path, {"AECMOS_DEG": 4.0})

    text = results.read_text(encoding="utf-8")
    assert "PESQ: 2.0000\n_count_PESQ: 1\n" in text
    assert text.count("AECMOS_DEG:") == 1
    assert "AECMOS_DEG: 4.0000\n" in text


def test_tse_aggregate_validates_bound_csv_hash(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval"
    metrics_dir = output_dir / "eval_metrics"
    metrics_dir.mkdir(parents=True)
    timing = metrics_dir / "eval_TSE_TIMING.csv"
    timing.write_text(
        "utterance,dataset,language,precision,recall,f1\n"
        "u,AMI,en,0.8,0.7,0.75\n",
        encoding="utf-8",
    )
    summary = {
        "run_id": "run-1",
        "status": "done",
        "metrics": {"active": ["timing"], "skipped": []},
        "coverage": {
            "timing": {
                "eligible": 1, "attempted": 1, "computed": 1,
                "failed": 0, "coverage": 1.0,
            }
        },
    }
    (output_dir / "tse_eval_summary.json").write_text(
        json.dumps({"run_id": "run-1", "summary": summary}),
        encoding="utf-8",
    )
    publish_result_manifest(output_dir, summary, artifact_paths=[timing])

    summarize_one_output_dir(output_dir, None, "eval_metrics")
    timing.write_text(timing.read_text(encoding="utf-8") + "v,AMI,en,1,1,1\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        summarize_one_output_dir(output_dir, None, "eval_metrics")


def test_icassp2023_alias_uses_official_directory_name() -> None:
    plan, unknown = pipeline_aec.build_dataset_plan(
        ["aec_icassp2023"], ["doubletalk"]
    )

    assert plan == [
        (
            "blind_test_set_icassp2023_doubletalk",
            "blind_test_set_icassp2023",
            "doubletalk",
        )
    ]
    assert unknown == []


@pytest.mark.parametrize(
    "invalid_alias",
    ["aec_icassp202", "aec_icassp20230", "aec_interspeech2023"],
)
def test_incomplete_or_unsupported_aec_alias_is_unknown(invalid_alias: str) -> None:
    plan, unknown = pipeline_aec.build_dataset_plan(
        [invalid_alias], ["doubletalk"]
    )

    assert plan == []
    assert unknown == [invalid_alias]


@pytest.mark.parametrize("missing_stream", ["mic", "lpb", "enh"])
def test_aecmos_official_triplet_uid_mismatch_fails_closed(
    tmp_path: Path, missing_stream: str
) -> None:
    paths = {}
    for stream in ("mic", "lpb", "enh"):
        scp = tmp_path / f"{stream}.scp"
        uid = "other" if stream == missing_stream else "sample"
        scp.write_text(f"{uid} {tmp_path / f'{stream}.wav'}\n", encoding="utf-8")
        paths[stream] = scp

    with pytest.raises(ValueError, match="UID mismatch"):
        eval_common.run_aecmos_for_scenario(
            paths["mic"], paths["lpb"], paths["enh"], talk_type="dt"
        )


@pytest.mark.parametrize(
    ("talk_type", "expected_metric_keys"),
    [
        ("st", {"AECMOS_ECHO", "EMOS"}),
        ("nst", {"AECMOS_DEG", "DMOS"}),
        ("dt", {"AECMOS_ECHO", "EMOS", "AECMOS_DEG", "DMOS"}),
    ],
)
def test_aecmos_returns_only_official_metrics_for_talk_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    talk_type: str,
    expected_metric_keys: set[str],
) -> None:
    scps = {}
    for stream in ("mic", "lpb", "enh"):
        scp = tmp_path / f"{stream}.scp"
        scp.write_text(f"sample {tmp_path / f'{stream}.wav'}\n", encoding="utf-8")
        scps[stream] = scp

    class Estimator:
        sampling_rate = 16000

    monkeypatch.setattr(
        eval_common,
        "load_audio",
        lambda _path: (np.ones(16000, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(eval_common, "_get_aecmos_estimator", lambda: Estimator())
    monkeypatch.setattr(
        eval_common,
        "compute_aecmos",
        lambda *_args, **_kwargs: {"AECMOS_ECHO": 4.0, "AECMOS_DEG": 3.0},
    )

    result = eval_common.run_aecmos_for_scenario(
        scps["mic"], scps["lpb"], scps["enh"], talk_type=talk_type
    )

    assert result is not None
    assert set(result) - {"n_samples", "n_failed", "echo_valid"} == expected_metric_keys
    assert result["n_samples"] == 1
    assert result["n_failed"] == 0
    assert result["echo_valid"] is True


def test_aecmos_resamples_all_streams_to_estimator_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scps = {}
    for stream in ("mic", "lpb", "enh"):
        scp = tmp_path / f"{stream}.scp"
        scp.write_text(f"sample {tmp_path / f'{stream}.wav'}\n", encoding="utf-8")
        scps[stream] = scp

    source_rates = {"mic": 48000, "lpb": 32000, "enh": 8000}

    def fake_load_audio(path: str):
        stream = Path(path).stem
        rate = source_rates[stream]
        return np.ones(rate, dtype=np.float32), rate

    class Estimator:
        sampling_rate = 16000

    observed = {}

    def fake_compute(_estimator, lpb, mic, enh, *, talk_type):
        observed["lengths"] = (len(lpb), len(mic), len(enh))
        observed["talk_type"] = talk_type
        return {"AECMOS_ECHO": 4.0, "AECMOS_DEG": 3.0}

    monkeypatch.setattr(eval_common, "load_audio", fake_load_audio)
    monkeypatch.setattr(eval_common, "_get_aecmos_estimator", lambda: Estimator())
    monkeypatch.setattr(eval_common, "compute_aecmos", fake_compute)

    result = eval_common.run_aecmos_for_scenario(
        scps["mic"], scps["lpb"], scps["enh"], talk_type="dt"
    )

    assert observed == {"lengths": (16000, 16000, 16000), "talk_type": "dt"}
    assert result == {
        "AECMOS_ECHO": 4.0,
        "AECMOS_DEG": 3.0,
        "EMOS": 4.0,
        "DMOS": 3.0,
        "n_samples": 1,
        "n_failed": 0,
        "echo_valid": True,
    }


def test_aecmos_all_sample_failures_are_reported_as_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scps = {}
    for stream in ("mic", "lpb", "enh"):
        scp = tmp_path / f"{stream}.scp"
        scp.write_text(
            "".join(
                f"sample-{index} {tmp_path / f'{stream}-{index}.wav'}\n"
                for index in range(2)
            ),
            encoding="utf-8",
        )
        scps[stream] = scp

    class Estimator:
        sampling_rate = 16000

    monkeypatch.setattr(
        eval_common,
        "load_audio",
        lambda _path: (_ for _ in ()).throw(OSError("corrupt audio")),
    )
    monkeypatch.setattr(eval_common, "_get_aecmos_estimator", lambda: Estimator())

    result = eval_common.run_aecmos_for_scenario(
        scps["mic"], scps["lpb"], scps["enh"], talk_type="dt"
    )

    assert result == {
        "skipped": True,
        "skip_reason": "corrupt audio",
        "n_samples": 0,
        "n_failed": 2,
        "echo_valid": True,
    }


pytestmark = [pytest.mark.unit]
