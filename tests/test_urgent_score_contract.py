"""URGENT normalized score sidecar 与报告白名单合同。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import evaluate.eval_pipeline_aec as pipeline_aec
import evaluate.eval_pipeline_se as pipeline_se
import tools.ops.reporting.experiment_report as er
from evaluate.common import (
    append_extra_metrics_to_results,
    build_done_summary,
    load_normalized_score,
    publish_result_manifest,
    validate_result_manifest,
    write_normalized_score,
)
from evaluate.urgent_score import (
    NORMALIZED_METRICS,
    SCORE_FILENAME,
    build_expected_metric_coverage,
    build_normalized_score,
    project_existing_score,
    validate_normalized_score,
)

pytestmark = pytest.mark.unit


def _values() -> dict[str, float]:
    return {
        "DNSMOS": 3.123456789,
        "NISQA": 2.987654321,
        "UTMOS": 3.25,
        "SCOREQ": 2.75,
        "PESQ": 2.125,
        "ESTOI": 0.8125,
    }


def _coverage(eligible: int = 1023) -> dict[str, dict[str, int]]:
    return {
        metric: {"eligible": eligible, "computed": eligible}
        for metric in NORMALIZED_METRICS
    }


def _write_score(results_dir: Path, score: dict) -> Path:
    bound, path = write_normalized_score(results_dir, score)
    score.clear()
    score.update(bound)
    return path


def test_complete_six_dimensions_and_coverage_is_official() -> None:
    score = build_normalized_score(_values(), _coverage())

    assert score["status"] == "official"
    assert score["dimensions"] == {"valid": 6, "required": 6, "missing": []}
    assert score["raw_values"]["DNSMOS"] == 3.123456789
    assert validate_normalized_score(score) == score


def test_six_of_six_with_pesq_1022_of_1023_is_partial() -> None:
    coverage = build_expected_metric_coverage(
        1023,
        {metric: 1022 if metric == "PESQ" else 1023 for metric in NORMALIZED_METRICS},
    )

    score = build_normalized_score(_values(), coverage)

    assert score["dimensions"]["valid"] == 6
    assert score["status"] == "partial"
    assert score["sample_coverage"]["PESQ"] == {
        "eligible": 1023,
        "computed": 1022,
        "coverage": pytest.approx(1022 / 1023),
        "status": "partial",
    }
    rendered = er._format_normalized_score({"_normalized_score": score})
    assert "partial; dimensions=6/6; PESQ=1022/1023" in rendered


def test_expected_population_survives_missing_reference_decode_and_empty_failures() -> None:
    coverage = build_expected_metric_coverage(
        4,
        {
            "DNSMOS": 2,
            "NISQA": 2,
            "UTMOS": 2,
            "SCOREQ": 2,
            "PESQ": 1,
            "ESTOI": 1,
        },
    )

    score = build_normalized_score(_values(), coverage)

    assert score["dimensions"]["valid"] == 6
    assert score["status"] == "partial"
    assert score["sample_coverage"]["DNSMOS"]["eligible"] == 4
    assert score["sample_coverage"]["DNSMOS"]["computed"] == 2
    assert score["sample_coverage"]["PESQ"]["computed"] == 1


def test_four_of_six_is_partial_and_unknown_coverage_is_legacy_unknown() -> None:
    values = _values()
    del values["PESQ"]
    del values["ESTOI"]
    four = build_normalized_score(values, _coverage())
    unknown = build_normalized_score(_values(), None)

    assert four["status"] == "partial"
    assert four["dimensions"]["valid"] == 4
    assert unknown["status"] == "legacy-unknown"
    assert unknown["dimensions"]["valid"] == 6


def test_historical_rounded_projection_is_never_official() -> None:
    projection = project_existing_score(
        {**_values(), "Normalized_Average": 0.5}, _coverage()
    )

    assert projection["dimensions"] == {"valid": 6, "required": 6, "missing": []}
    assert projection["status"] == "partial"
    assert er._format_normalized_score(
        {"_normalized_score_projection": projection}
    ) == (
        "0.5000 (legacy/partial projection; dimensions=6/6; "
        "observed=DNSMOS=1023,NISQA=1023,UTMOS=1023,SCOREQ=1023,"
        "PESQ=1023,ESTOI=1023)"
    )


def test_historical_projection_marks_missing_without_inventing_population() -> None:
    projection = project_existing_score(
        {
            "DNSMOS": 3.0,
            "_count_DNSMOS": 300,
            "Normalized_Average": 0.5,
        },
        {"DNSMOS": {"computed": 300}},
    )

    assert projection["sample_coverage"]["DNSMOS"] == {
        "eligible": None,
        "computed": 300,
        "coverage": None,
        "status": "observed",
    }
    assert projection["sample_coverage"]["PESQ"] == {
        "eligible": None,
        "computed": None,
        "coverage": None,
        "status": "missing",
    }
    rendered = er._format_normalized_score(
        {"_normalized_score_projection": projection}
    )
    assert "observed=DNSMOS=300" in rendered
    assert "missing=NISQA,UTMOS,SCOREQ,PESQ,ESTOI" in rendered
    assert "0/300" not in rendered


def test_validator_rejects_tampered_derived_value() -> None:
    score = build_normalized_score(_values(), _coverage())
    score["value"] += 0.01

    with pytest.raises(ValueError, match="inconsistent"):
        validate_normalized_score(score)


def test_build_done_summary_and_manifest_bind_only_results_sibling(tmp_path: Path) -> None:
    output = tmp_path / "eval"
    results_dir = output / "urgent"
    enhanced = results_dir / "enhanced"
    enhanced.mkdir(parents=True)
    results = results_dir / "RESULTS.txt"
    results.write_text("Normalized_Average: 0.1235\n", encoding="utf-8")
    score = build_normalized_score(_values(), _coverage())
    sidecar = _write_score(results_dir, score)
    stale = output / "stale" / SCORE_FILENAME
    stale.parent.mkdir()
    stale.write_text(json.dumps(score), encoding="utf-8")

    summary = {
        "run_id": "run-1",
        "datasets": {
            "urgent": build_done_summary(enhanced, results_dir),
        },
    }
    manifest_path = publish_result_manifest(output, summary)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    dataset = summary["datasets"]["urgent"]
    assert dataset["normalized_score"] == score
    assert dataset["normalized_score_file"] == str(sidecar.resolve())
    assert set(manifest["files"]) == {
        "urgent/RESULTS.txt",
        f"urgent/{SCORE_FILENAME}",
    }
    assert f"stale/{SCORE_FILENAME}" not in manifest["files"]


def test_manifest_uses_explicit_scenario_results_without_metric_coverage(
    tmp_path: Path,
) -> None:
    output = tmp_path / "eval"
    scenario_dir = output / "aec_sim" / "prob_target_only"
    scenario_dir.mkdir(parents=True)
    results = scenario_dir / "RESULTS.txt"
    results.write_text("Normalized_Average: 0.5000\n", encoding="utf-8")
    score = build_normalized_score(_values(), _coverage())
    sidecar = _write_score(scenario_dir, score)
    summary = {
        "run_id": "run-1",
        "datasets": {
            "aec_sim": {
                "status": "done",
                "eval_root": str(output / "aec_sim"),
                "scenario_results_files": {"prob_target_only": str(results)},
                "scenario_normalized_scores": {"prob_target_only": score},
                "scenario_normalized_score_files": {"prob_target_only": str(sidecar)},
            }
        },
    }

    manifest_path = publish_result_manifest(output, summary)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert set(manifest["files"]) == {
        "aec_sim/prob_target_only/RESULTS.txt",
        f"aec_sim/prob_target_only/{SCORE_FILENAME}",
    }


@pytest.mark.parametrize(
    "dataset",
    [
        {"status": "done", "normalized_score": build_normalized_score(_values(), _coverage())},
        {
            "status": "done",
            "eval_root": "/tmp/aec_sim",
            "scenario_normalized_scores": {
                "prob_target_only": build_normalized_score(_values(), _coverage())
            },
        },
    ],
    ids=["direct", "scenario"],
)
def test_manifest_rejects_one_sided_score_declaration(
    tmp_path: Path, dataset: dict,
) -> None:
    with pytest.raises(ValueError, match="declaration is incomplete"):
        publish_result_manifest(
            tmp_path, {"run_id": "run-1", "datasets": {"dataset": dataset}}
        )


def test_manifest_rejects_scenario_results_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / "aec_sim"
    summary = {
        "run_id": "run-1",
        "datasets": {
            "aec_sim": {
                "status": "done",
                "eval_root": str(root),
                "scenario_results_files": {
                    "../escape": str(root.parent / "escape" / "RESULTS.txt")
                },
            }
        },
    }

    with pytest.raises(ValueError, match="scenario RESULTS name is invalid"):
        publish_result_manifest(tmp_path, summary)


def test_manifest_rejects_missing_declared_scenario_results(tmp_path: Path) -> None:
    results = tmp_path / "aec_sim" / "prob_target_only" / "RESULTS.txt"
    summary = {
        "run_id": "run-1",
        "datasets": {
            "aec_sim": {
                "status": "done",
                "eval_root": str(tmp_path / "aec_sim"),
                "scenario_results_files": {"prob_target_only": str(results)},
            }
        },
    }

    with pytest.raises(FileNotFoundError, match="metric artifact missing"):
        publish_result_manifest(tmp_path, summary)


def test_manifest_rejects_direct_results_with_noncanonical_filename(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "urgent" / "metrics.txt"
    wrong.parent.mkdir()
    wrong.write_text("PESQ: 2.0\n", encoding="utf-8")
    summary = {
        "run_id": "run-1",
        "datasets": {
            "urgent": {"status": "done", "results_file": str(wrong)}
        },
    }

    with pytest.raises(ValueError, match="must name RESULTS.txt"):
        publish_result_manifest(tmp_path, summary)


def test_manifest_rejects_score_dto_that_differs_from_disk(tmp_path: Path) -> None:
    results_dir = tmp_path / "urgent"
    enhanced = results_dir / "enhanced"
    enhanced.mkdir(parents=True)
    (results_dir / "RESULTS.txt").write_text(
        "Normalized_Average: 0.5000\n", encoding="utf-8"
    )
    score = build_normalized_score(_values(), _coverage())
    _write_score(results_dir, score)
    summary = {
        "run_id": "run-1",
        "datasets": {"urgent": build_done_summary(enhanced, results_dir)},
    }
    summary["datasets"]["urgent"]["normalized_score"] = {
        **score,
        "status": "partial",
    }

    with pytest.raises(ValueError, match="does not match current RESULTS sibling"):
        publish_result_manifest(tmp_path, summary)


def test_extra_metrics_refresh_score_after_results_and_replace_does_not_revive_it(
    tmp_path: Path,
) -> None:
    results = tmp_path / "RESULTS.txt"
    results.write_text("Normalized_Average: 0.5000\n", encoding="utf-8")
    score = build_normalized_score(_values(), _coverage())
    sidecar = _write_score(tmp_path, score)

    old_binding = score["results"]
    append_extra_metrics_to_results(tmp_path, {"ERLE_dB": 2.0})

    refreshed, refreshed_path = load_normalized_score(tmp_path)
    assert refreshed_path == sidecar.resolve()
    assert refreshed["raw_values"] == score["raw_values"]
    assert refreshed["sample_coverage"] == score["sample_coverage"]
    assert refreshed["results"] != old_binding

    append_extra_metrics_to_results(tmp_path, {"ERLE_dB": 3.0}, replace=True)

    assert load_normalized_score(tmp_path) is None


def test_explicit_manifest_artifacts_cannot_bypass_score_declaration(
    tmp_path: Path,
) -> None:
    output = tmp_path / "eval"
    results_dir = output / "urgent"
    results_dir.mkdir(parents=True)
    results = results_dir / "RESULTS.txt"
    results.write_text("Normalized_Average: 0.5000\n", encoding="utf-8")
    sidecar = _write_score(results_dir, build_normalized_score(_values(), _coverage()))
    summary = {"run_id": "run-1", "datasets": {}}

    with pytest.raises(ValueError, match="not declared by summary"):
        publish_result_manifest(output, summary, artifact_paths=[results, sidecar])


def test_explicit_manifest_artifacts_must_include_summary_declared_score(
    tmp_path: Path,
) -> None:
    output = tmp_path / "eval"
    results_dir = output / "urgent"
    enhanced = results_dir / "enhanced"
    enhanced.mkdir(parents=True)
    results = results_dir / "RESULTS.txt"
    results.write_text("Normalized_Average: 0.5000\n", encoding="utf-8")
    score = build_normalized_score(_values(), _coverage())
    _write_score(results_dir, score)
    summary = {
        "run_id": "run-1",
        "datasets": {"urgent": build_done_summary(enhanced, results_dir)},
    }

    with pytest.raises(ValueError, match="missing from artifact_paths"):
        publish_result_manifest(output, summary, artifact_paths=[results])


def test_manifest_validation_rejects_omitted_summary_declared_score(
    tmp_path: Path,
) -> None:
    output = tmp_path / "eval"
    results_dir = output / "urgent"
    enhanced = results_dir / "enhanced"
    enhanced.mkdir(parents=True)
    results = results_dir / "RESULTS.txt"
    results.write_text("Normalized_Average: 0.5000\n", encoding="utf-8")
    score = build_normalized_score(_values(), _coverage())
    sidecar = _write_score(results_dir, score)
    summary = {
        "run_id": "run-1",
        "datasets": {"urgent": build_done_summary(enhanced, results_dir)},
    }
    manifest_path = publish_result_manifest(output, summary)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].pop(sidecar.relative_to(output).as_posix())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="omits summary normalized score"):
        validate_result_manifest(output, expected_summary=summary)


def test_manifest_validation_rejects_extra_undeclared_score(
    tmp_path: Path,
) -> None:
    output = tmp_path / "eval"
    results_dir = output / "urgent"
    enhanced = results_dir / "enhanced"
    enhanced.mkdir(parents=True)
    results = results_dir / "RESULTS.txt"
    results.write_text("Normalized_Average: 0.5000\n", encoding="utf-8")
    score = build_normalized_score(_values(), _coverage())
    sidecar = _write_score(results_dir, score)
    stale_dir = output / "stale"
    stale_dir.mkdir()
    stale_results = stale_dir / "RESULTS.txt"
    stale_results.write_text(results.read_text(encoding="utf-8"), encoding="utf-8")
    stale_sidecar = _write_score(stale_dir, score)
    summary = {
        "run_id": "run-1",
        "datasets": {"urgent": build_done_summary(enhanced, results_dir)},
    }
    manifest_path = publish_result_manifest(output, summary)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][stale_results.relative_to(output).as_posix()] = manifest[
        "files"
    ][results.relative_to(output).as_posix()]
    manifest["files"][stale_sidecar.relative_to(output).as_posix()] = manifest[
        "files"
    ][sidecar.relative_to(output).as_posix()]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="undeclared URGENT normalized score"):
        validate_result_manifest(output, expected_summary=summary)


def test_report_ignores_unmanifested_stale_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "eval"
    results_dir = output / "urgent"
    results_dir.mkdir(parents=True)
    results = results_dir / "RESULTS.txt"
    results.write_text(
        "DNSMOS: 3.0000\nNormalized_Average: 0.5000\n", encoding="utf-8"
    )
    _write_score(results_dir, build_normalized_score(_values(), _coverage()))
    summary = {"run_id": "run-1", "datasets": {}}
    publish_result_manifest(output, summary, artifact_paths=[results])
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    parsed = er.parse_all_results(output)["urgent"]

    assert parsed["Normalized_Average"] == 0.5
    assert "_normalized_score" not in parsed


def test_score_load_rejects_same_size_results_replacement_despite_newer_mtime(
    tmp_path: Path,
) -> None:
    results = tmp_path / "RESULTS.txt"
    results.write_text("Normalized_Average: 0.5000\n", encoding="utf-8")
    _write_score(tmp_path, build_normalized_score(_values(), _coverage()))

    results.write_text("Normalized_Average: 0.6000\n", encoding="utf-8")

    assert load_normalized_score(tmp_path) is None


def test_report_reads_and_validates_only_manifested_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "eval"
    results_dir = output / "urgent"
    enhanced = results_dir / "enhanced"
    enhanced.mkdir(parents=True)
    results = results_dir / "RESULTS.txt"
    results.write_text("Normalized_Average: 0.5000\n", encoding="utf-8")
    score = build_normalized_score(_values(), _coverage())
    _write_score(results_dir, score)
    summary = {
        "run_id": "run-1",
        "datasets": {"urgent": build_done_summary(enhanced, results_dir)},
    }
    publish_result_manifest(output, summary)
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    parsed = er.parse_all_results(output)["urgent"]

    assert parsed["Normalized_Average"] == 0.5
    assert parsed["_normalized_score"] == score
    assert er._format_normalized_score(parsed).startswith(f"{score['value']:.4f} (official;")


@pytest.mark.parametrize("module", [pipeline_aec, pipeline_se], ids=["aec", "se"])
def test_aec_sim_summary_declares_evaluator_results_without_requested_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module,
) -> None:
    eval_root = tmp_path / "eval"
    scenario_dir = eval_root / "aec_sim" / "prob_target_only"
    enhanced = scenario_dir / "enhanced"
    enhanced.mkdir(parents=True)
    inf_scp = enhanced / "inf.scp"
    inf_scp.write_text("u1 /tmp/u1.wav\n", encoding="utf-8")
    testset_root = tmp_path / "testset"
    testset_root.mkdir()
    scp_paths = {
        "aec_sim": {
            "prob_target_only": {
                "mic": tmp_path / "mic.scp",
                "farend": tmp_path / "farend.scp",
                "noisy_se": tmp_path / "noisy_se.scp",
                "nearend": None,
            }
        }
    }
    args = SimpleNamespace(
        step="evaluate",
        aec_sim_testset_root=str(testset_root),
        allow_partial_inference=True,
        audio_quality=False,
        downstream=False,
        device="cpu",
        inference_devices=None,
        aec_no_stream_align=False,
        skip_existing_inference=False,
        effective_config_identity=None,
        aecmos=False,
    )
    coverage = {"expected": 1, "produced": 1, "failed": 0, "evaluated": 1}
    monkeypatch.setattr(module, "ensure_inf_scp", lambda _path: inf_scp)
    monkeypatch.setattr(
        module, "load_inference_coverage", lambda *_args, **_kwargs: dict(coverage)
    )
    monkeypatch.setattr(module, "requested_standard_metrics", lambda **_kwargs: [])

    def run_evaluator(**kwargs) -> None:
        results_dir = Path(kwargs["output_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "RESULTS.txt").write_text(
            "Normalized_Average: 0.5000\n", encoding="utf-8"
        )
        _write_score(results_dir, build_normalized_score(_values(), _coverage(1)))

    monkeypatch.setattr(module, "run_evaluate_urgent", run_evaluator)
    summary = {}
    module.process_aec_sim(
        args,
        tmp_path / "config.yaml",
        tmp_path / "model.tar",
        eval_root,
        summary,
        scp_paths,
    )

    dataset = summary["aec_sim"]
    expected_results = scenario_dir / "RESULTS.txt"
    assert dataset["scenario_metric_coverage"] == {}
    assert dataset["scenario_results_files"] == {
        "prob_target_only": str(expected_results.resolve())
    }
    assert dataset["scenario_normalized_scores"]["prob_target_only"]["status"] == "official"
    assert dataset["scenario_normalized_score_files"]["prob_target_only"] == str(
        (scenario_dir / SCORE_FILENAME).resolve()
    )


def test_aec_and_se_share_identical_score_summary_contract(tmp_path: Path) -> None:
    assert pipeline_aec.build_done_summary is build_done_summary
    assert pipeline_se.build_done_summary is build_done_summary
    results_dir = tmp_path / "urgent"
    enhanced = results_dir / "enhanced"
    enhanced.mkdir(parents=True)
    (results_dir / "RESULTS.txt").write_text("PESQ: 2.0\n", encoding="utf-8")
    score = build_normalized_score(_values(), _coverage())
    _write_score(results_dir, score)

    aec_summary = pipeline_aec.build_done_summary(enhanced, results_dir)
    se_summary = pipeline_se.build_done_summary(enhanced, results_dir)

    assert aec_summary["normalized_score"] == se_summary["normalized_score"] == score
    assert aec_summary["normalized_score_file"] == se_summary["normalized_score_file"]
