"""evaluate 审计 P1 回归测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import evaluate.aggregate_eval_summary as aggregate_eval_summary
import evaluate.common as eval_common
import evaluate.eval_pipeline_aec as pipeline_aec
import evaluate.evaluate_urgent as evaluate_urgent
from evaluate.eval_pipeline_common import find_exp_assets

pytestmark = [pytest.mark.unit]


def test_find_exp_assets_uses_complete_same_run_pair(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    (exp / "checkpoints" / "new").mkdir(parents=True)
    (exp / "best_model.tar").write_bytes(b"root-without-config")
    run = exp / "checkpoints" / "new"
    config = run / "config.yaml"
    checkpoint = run / "best_model.tar"
    config.write_text("model: {type: Demo}\n", encoding="utf-8")
    checkpoint.write_bytes(b"same-run")

    assert find_exp_assets(exp, prefer_model_only=False) == (config, checkpoint)


def test_optional_aecmos_does_not_call_evaluator_when_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline_aec,
        "run_aecmos_for_scenario",
        lambda **_kwargs: pytest.fail("AECMOS must not run unless requested"),
    )

    result = pipeline_aec._run_optional_aecmos(
        False,
        "test",
        mic_scp="mic.scp",
        lpb_scp="far.scp",
        enh_scp="enh.scp",
        talk_type="st",
    )

    assert result == {"skipped": True, "skip_reason": "not_requested", "n_samples": 0}


def test_aecmos_estimator_failure_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {}
    for stream in ("mic", "lpb", "enh"):
        path = tmp_path / f"{stream}.scp"
        path.write_text(f"sample {tmp_path / f'{stream}.wav'}\n", encoding="utf-8")
        paths[stream] = path
    monkeypatch.setattr(
        eval_common,
        "_get_aecmos_estimator",
        lambda: (_ for _ in ()).throw(ImportError("onnxruntime unavailable")),
    )

    result = eval_common.run_aecmos_for_scenario(
        paths["mic"], paths["lpb"], paths["enh"], talk_type="dt"
    )

    assert result is not None
    assert result["skipped"] is True
    assert result["n_samples"] == 0
    assert "ImportError" in result["skip_reason"]


def test_tse_internal_results_require_manifest_binding(tmp_path: Path) -> None:
    output_dir = tmp_path / "tse"
    output_dir.mkdir()
    results = output_dir / "tse_eval_results.csv"
    results.write_text(
        "dataset,mixture_utterance,enrolment_speakers_utterance,language\n"
        "AMI,mix,enroll,en\n",
        encoding="utf-8",
    )
    summary = {
        "run_id": "run-1",
        "summary": {
            "run_id": "run-1",
            "status": "done",
            "metrics": {"active": [], "skipped": []},
            "coverage": {},
        },
    }
    (output_dir / "tse_eval_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    manifest = {
        "schema": "evaluation-results-v1",
        "run_id": "run-1",
        "files": {},
    }
    (output_dir / "result_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    output = aggregate_eval_summary.summarize_one_output_dir(
        output_dir, None, "eval_metrics"
    ).read_text(encoding="utf-8")

    assert "official CSV fallback UID union" in output
    assert "internal results composite UID union" not in output


def test_evaluate_urgent_rejects_missing_explicit_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inf_scp = tmp_path / "inf.scp"
    inf_scp.write_text("sample enhanced.wav\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_urgent.py",
            "--inf_scp",
            str(inf_scp),
            "--ref_scp",
            str(tmp_path / "missing.scp"),
            "--output_dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(SystemExit):
        evaluate_urgent.main()


def test_evaluate_urgent_shard_score_coverage_uses_full_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inf_scp = tmp_path / "inf.scp"
    inf_scp.write_text(
        "u1 audio1.wav\nu2 audio2.wav\nu3 audio3.wav\nu4 audio4.wav\n",
        encoding="utf-8",
    )
    captured: dict[str, int] = {}
    real_build_coverage = evaluate_urgent.build_expected_metric_coverage

    def capture_population(expected_population, metric_counts):
        captured["expected_population"] = expected_population
        return real_build_coverage(expected_population, metric_counts)

    monkeypatch.setattr(evaluate_urgent, "build_expected_metric_coverage", capture_population)
    monkeypatch.setattr(evaluate_urgent, "load_audio", lambda _path: ([0.0], 16000))
    monkeypatch.setattr(
        evaluate_urgent,
        "compute_metrics_batch",
        lambda **kwargs: ({}, [{} for _ in kwargs["pairs"]]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_urgent.py",
            "--inf_scp",
            str(inf_scp),
            "--output_dir",
            str(tmp_path / "out"),
            "--nsplits",
            "2",
            "--job",
            "1",
        ],
    )

    evaluate_urgent.main()

    assert captured["expected_population"] == 4


def test_wer_requires_owsm_utils_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """官方 checkout 与 vendored fallback 都缺失时必须在加载阶段 fail-fast。"""
    missing = tmp_path / "evaluation_metrics"
    vendored = tmp_path / "owsm_utils.py"
    monkeypatch.setattr(evaluate_urgent, "_URGENT_METRICS_DIR", missing)
    monkeypatch.setattr(evaluate_urgent, "_VENDORED_OWSM_UTILS", vendored)

    with pytest.raises(ImportError, match="owsm_utils.py"):
        evaluate_urgent._require_owsm_utils()

    missing.mkdir()
    (missing / "owsm_utils.py").write_text("def owsm_predict(*_args, **_kwargs):\n    pass\n", encoding="utf-8")
    assert evaluate_urgent._require_owsm_utils() == missing / "owsm_utils.py"

    (missing / "owsm_utils.py").unlink()
    vendored.write_text("def owsm_predict(*_args, **_kwargs):\n    pass\n", encoding="utf-8")
    assert evaluate_urgent._require_owsm_utils() == vendored
