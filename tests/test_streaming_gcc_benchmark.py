"""Unit tests for the reproducible streaming GCC benchmark."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from tools.verify import streaming_gcc_benchmark as benchmark
from tools.verify.streaming_gcc_benchmark import (
    BASELINE,
    Profile,
    Scenario,
    build_scenarios,
    choose_winner,
    representative_profiles,
    run_scenario,
    summarize,
    synthesize_scenario,
)

pytestmark = pytest.mark.unit


def _gate_records() -> list[dict]:
    cadence = {"mean": 100.0, "p95": 100.0}
    return [
        {
            "name": "stationary_a",
            "group": "stationary_delay",
            "elapsed_s": 0.4,
            "audio_s": 4.0,
            "finite": True,
            "mae_ms": 2.0,
            "p95_error_ms": 10.0,
            "false_lock_rate": 0.01,
            "converged": True,
            "jitter_p95_ms": 1.0,
            "actual_update_cadence_ms": cadence,
        },
        {
            "name": "stationary_b",
            "group": "stationary_delay",
            "elapsed_s": 0.8,
            "audio_s": 4.0,
            "finite": True,
            "mae_ms": 5.0,
            "p95_error_ms": 15.0,
            "false_lock_rate": 0.01,
            "converged": True,
            "jitter_p95_ms": 4.0,
            "actual_update_cadence_ms": cadence,
        },
        {
            "name": "jump",
            "group": "delay_jump",
            "elapsed_s": 0.2,
            "audio_s": 4.0,
            "finite": True,
            "reacquisition_s": 2.0,
            "final_error_ms": 15.0,
            "jitter_p95_ms": 5.0,
            "actual_update_cadence_ms": cadence,
        },
        *[
            {
                "name": f"nearend_only_seed_{index}",
                "group": "nearend_only",
                "elapsed_s": 0.1,
                "audio_s": 4.0,
                "finite": True,
                "final_delay_ms": 10.0,
                "false_activation_threshold_ms": 10.0,
                "false_activation_rate": 0.01,
                "actual_update_cadence_ms": cadence,
            }
            for index in range(benchmark.NEAREND_SEED_COUNT)
        ],
    ]


def _result(
    profile: Profile,
    *,
    score: float,
    hard_gate: bool = True,
    cpu: float = 1.0,
) -> dict:
    return {
        "profile": asdict(profile),
        "name": profile.name,
        "hard_gate": hard_gate,
        "score": score,
        "worst_group_mae_ms": 1.0,
        "worst_group_p95_ms": 2.0,
        "no_convergence_rate": 0.0,
        "false_lock_rate": 0.0,
        "jump_reacquisition_p95_s": 0.1,
        "jitter_p95_ms": 0.2,
        "cpu_seconds_per_audio_second": cpu,
        "scenarios": [],
    }


def test_synthetic_scenario_is_reproducible_and_seed_sensitive():
    scenario = Scenario("jump", 17, 40.0, 0.2, jump_to_ms=350.0, jump_time_s=2.0)
    first = synthesize_scenario(scenario, 16000, 2.5)
    second = synthesize_scenario(scenario, 16000, 2.5)
    changed_seed = synthesize_scenario(replace(scenario, seed=18), 16000, 2.5)

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(first[0], changed_seed[0])
    assert first[2][0] == 40.0
    assert first[2][-1] == 350.0


def test_nearend_only_is_reported_as_negative_control_not_accuracy(monkeypatch):
    class SequenceEstimator:
        def __init__(self):
            self._delays = iter([0.0, 0.5, 10.0, 10.01, 20.0])

        @property
        def delay_ms(self):
            return next(self._delays)

        def push_ref(self, frame):
            pass

        def push_mic(self, frame):
            pass

        def get_aligned_ref_frame(self, frame_size):
            return np.zeros(frame_size, dtype=np.float32)

    monkeypatch.setattr(benchmark, "_make_estimator", lambda *args: SequenceEstimator())
    record = run_scenario(
        BASELINE,
        Scenario("nearend_only", 18, variant="nearend_only", echo_gain=0.0),
        sample_rate=16000,
        duration_s=0.05,
        backend="python",
    )

    assert record["false_activation_threshold_ms"] == 10.0
    assert record["false_activation_rate"] == pytest.approx(0.4)
    assert "mae_ms" not in record
    assert "p95_error_ms" not in record


def test_nearend_negative_control_uses_multiple_pre_registered_seeds():
    for split in ("dev", "holdout"):
        negatives = [item for item in build_scenarios(split) if not item.has_oracle]
        assert len(negatives) == 5
        assert len({item.seed for item in negatives}) == 5
        assert all(item.group == "nearend_only" for item in negatives)


def test_update_cadence_preserves_remainder_for_non_divisible_frame(monkeypatch):
    class ZeroEstimator:
        delay_ms = 0.0

        def push_ref(self, frame):
            pass

        def push_mic(self, frame):
            pass

        def get_aligned_ref_frame(self, frame_size):
            return np.zeros(frame_size, dtype=np.float32)

    monkeypatch.setattr(benchmark, "_make_estimator", lambda *args: ZeroEstimator())
    profile = replace(BASELINE, update_ms=100.0, frame_size=600)
    record = run_scenario(
        profile,
        Scenario("nearend_only", 19, variant="nearend_only", echo_gain=0.0),
        sample_rate=16000,
        duration_s=0.4,
        backend="python",
    )

    assert record["update_trigger_count"] == 4
    assert record["actual_update_cadence_ms"]["mean"] == pytest.approx(100.0)


def test_dev_and_holdout_are_deterministic_and_isolated():
    dev = build_scenarios("dev")
    holdout = build_scenarios("holdout")

    assert dev == build_scenarios("dev")
    assert holdout == build_scenarios("holdout")
    assert {item.seed for item in dev}.isdisjoint({item.seed for item in holdout})
    assert {item.jump_time_s for item in dev if item.jump_to_ms is not None} == {2.0}
    assert {item.jump_time_s for item in holdout if item.jump_to_ms is not None} == {2.35}
    assert "holdout_multipath_175ms" not in {item.name for item in dev}
    assert "holdout_multipath_175ms" in {item.name for item in holdout}
    assert {
        "clean",
        "noise",
        "doubletalk",
        "multipath",
        "clipping",
        "dropout",
        "nearend_only",
    } <= {item.variant for item in dev}

    with pytest.raises(ValueError, match="unknown split"):
        build_scenarios("test")


def test_summarize_uses_worst_case_metrics_and_weighted_score():
    summary = summarize(BASELINE, _gate_records())

    assert summary["hard_gate"] is True
    assert summary["group_metrics"]["stationary_delay"] == {
        "scenario_count": 2,
        "mae_ms": 3.5,
        "p95_error_ms": 15.0,
        "false_lock_rate": 0.01,
        "no_convergence_rate": 0.0,
    }
    assert summary["worst_group_mae_ms"] == 3.5
    assert summary["worst_group_p95_ms"] == 15.0
    assert summary["no_convergence_rate"] == 0.0
    assert summary["false_lock_rate"] == 0.01
    assert summary["oracle_nonjump_all_scenarios_passed"] is True
    assert summary["nearend_false_activation_rate"] == 0.01
    assert summary["nearend_false_activation_p95_rate"] == 0.01
    assert summary["nearend_seed_count"] == benchmark.NEAREND_SEED_COUNT
    assert summary["nearend_seed_pass_rate"] == 1.0
    assert summary["nearend_max_final_delay_ms"] == 10.0
    assert summary["nearend_all_seeds_passed"] is True
    assert summary["jump_reacquisition_p95_s"] == 2.0
    assert summary["jump_max_final_error_ms"] == 15.0
    assert summary["jump_all_scenarios_passed"] is True
    assert summary["jitter_p95_ms"] == 5.0
    assert summary["cpu_seconds_per_audio_second"] == pytest.approx(1.9 / 32.0)
    assert summary["actual_update_cadence_ms"] == 100.0
    assert summary["score"] == pytest.approx(
        3.5 + 1.5 * 15 + 5 * 0.01 + 4 * 2 + 5
    )


@pytest.mark.parametrize(
    ("record_index", "field", "value"),
    [
        (0, "finite", False),
        (0, "p95_error_ms", 15.01),
        (0, "false_lock_rate", 0.011),
        (0, "converged", False),
        (2, "reacquisition_s", 2.01),
        (2, "final_error_ms", 15.01),
        (2, "jitter_p95_ms", 5.01),
        (3, "false_activation_rate", 0.011),
        (3, "final_delay_ms", 10.01),
    ],
)
def test_summarize_hard_gate_fails_closed(record_index, field, value):
    records = _gate_records()
    records[record_index][field] = value
    assert summarize(BASELINE, records)["hard_gate"] is False


def test_nearend_multi_seed_gate_requires_every_seed_to_pass():
    records = _gate_records()
    records[3]["false_activation_rate"] = 0.011
    summary = summarize(BASELINE, records)
    assert summary["nearend_seed_pass_rate"] == pytest.approx(0.8)
    assert summary["nearend_all_seeds_passed"] is False
    assert summary["hard_gate"] is False


def test_choose_winner_requires_dev_holdout_and_cpu_gates():
    challenger = replace(BASELINE, update_ms=50.0)
    baseline_dev = _result(BASELINE, score=100.0)
    baseline_holdout = _result(BASELINE, score=100.0)
    challenger_dev = _result(challenger, score=95.0)

    holdout_failed = _result(challenger, score=80.0, hard_gate=False, cpu=0.5)
    winner = choose_winner(
        [baseline_dev, challenger_dev], [baseline_holdout, holdout_failed]
    )
    assert winner["name"] == BASELINE.name

    cpu_failed = _result(challenger, score=95.0, cpu=1.251)
    winner = choose_winner([baseline_dev, challenger_dev], [baseline_holdout, cpu_failed])
    assert winner["name"] == BASELINE.name

    significant_dev = _result(challenger, score=89.0)
    significant_holdout = _result(challenger, score=95.0, cpu=1.5)
    winner = choose_winner(
        [significant_dev, baseline_dev], [baseline_holdout, significant_holdout]
    )
    assert winner["name"] == BASELINE.name
    assert winner["cpu_ratio_vs_baseline"] == pytest.approx(1.0)


def test_choose_winner_does_not_claim_baseline_when_all_profiles_fail():
    challenger = replace(BASELINE, update_ms=50.0)
    winner = choose_winner(
        [
            _result(BASELINE, score=10.0, hard_gate=False),
            _result(challenger, score=8.0, hard_gate=False),
        ],
        [
            _result(BASELINE, score=10.0, hard_gate=False),
            _result(challenger, score=8.0, hard_gate=False),
        ],
    )

    assert winner["selected"] is False
    assert winner["name"] is None
    assert winner["profile"] is None
    assert winner["dev"] is None
    assert winner["holdout"] is None


def test_choose_winner_tie_breaks_toward_baseline_then_smaller_buffer():
    challenger = replace(BASELINE, update_ms=50.0)
    winner = choose_winner(
        [_result(challenger, score=9.95), _result(BASELINE, score=10.0)],
        [_result(challenger, score=9.95), _result(BASELINE, score=10.0)],
    )
    assert winner["name"] == BASELINE.name

    small = replace(BASELINE, buffer_ms=450.0)
    large = replace(BASELINE, buffer_ms=650.0)
    winner = choose_winner(
        [
            _result(large, score=9.95, cpu=0.8),
            _result(small, score=10.0, cpu=1.2),
            _result(BASELINE, score=20.0, hard_gate=False),
        ],
        [
            _result(large, score=9.95, cpu=0.8),
            _result(small, score=10.0, cpu=1.2),
            _result(BASELINE, score=20.0, hard_gate=False),
        ],
    )
    assert winner["name"] == small.name

    cheap = replace(BASELINE, update_ms=50.0)
    expensive = replace(BASELINE, update_ms=200.0)
    winner = choose_winner(
        [
            _result(expensive, score=9.95, cpu=1.2),
            _result(cheap, score=10.0, cpu=0.8),
            _result(BASELINE, score=20.0, hard_gate=False),
        ],
        [
            _result(expensive, score=9.95, cpu=1.2),
            _result(cheap, score=10.0, cpu=0.8),
            _result(BASELINE, score=20.0, hard_gate=False),
        ],
    )
    assert winner["name"] == cheap.name


def test_profile_defaults_match_promoted_baseline():
    assert Profile() == BASELINE
    assert asdict(BASELINE) == {
        "buffer_ms": 500.0,
        "residual_ms": 50.0,
        "update_ms": 100.0,
        "alpha": 0.15,
        "acps_alpha": 0.8,
        "peak_thresh": 0.05,
        "prominence_db": 6.0,
        "false_accept_prob": benchmark.FALSE_ACCEPT_PROB,
        "min_obs_ms": 200.0,
        "frame_size": 160,
    }


def test_representative_search_is_the_small_deterministic_matrix(monkeypatch):
    profiles = representative_profiles()
    captured = []

    def fake_evaluate(candidate_profiles, scenarios, **kwargs):
        captured.extend(candidate_profiles)
        return [_result(profile, score=float(index)) for index, profile in enumerate(captured)]

    monkeypatch.setattr(benchmark, "evaluate_profiles", fake_evaluate)
    results, stages = benchmark.staged_search(
        "representative",
        build_scenarios("dev")[:1],
        sample_rate=16000,
        duration_s=3.5,
        backend="python",
        search_config=benchmark.load_search_config(None),
    )

    assert profiles == [
        BASELINE,
        replace(BASELINE, residual_ms=500.0, acps_alpha=0.7, min_obs_ms=100.0),
        replace(BASELINE, residual_ms=100.0),
        replace(BASELINE, update_ms=50.0),
        replace(BASELINE, acps_alpha=0.9),
        replace(BASELINE, min_obs_ms=100.0),
    ]
    assert len(set(profiles)) == len(profiles)
    assert profiles == representative_profiles()
    assert captured == profiles
    assert results == stages[0]["results"]
    assert stages[0]["stage"] == "representative"
    assert {profile.frame_size for profile in profiles} == {160}
    assert {profile.false_accept_prob for profile in profiles} == {
        benchmark.FALSE_ACCEPT_PROB
    }


def test_full_grid_starts_from_baseline_and_keeps_old_acps_alpha_control():
    grid = benchmark.DEFAULT_STAGE_GRID["structure_memory"]
    acps_values = grid["acps_alpha"]
    profiles = [
        replace(BASELINE, buffer_ms=buffer, residual_ms=residual, update_ms=update, acps_alpha=acps)
        for buffer, residual in grid["buffer_residual_ms"]
        for update in grid["update_ms"]
        for acps in acps_values
    ]

    assert acps_values[0] == BASELINE.acps_alpha == 0.8
    assert 0.7 in acps_values  # 旧 winner 取值保留为对照
    assert 0.9 in acps_values
    assert len(set(profiles)) == len(profiles)


def test_minimal_candidate_json_schema_is_strict(tmp_path):
    config_path = tmp_path / "search.json"
    config_path.write_text(
        json.dumps({"candidates": [{"update_ms": 50.0, "frame_size": 80}]}),
        encoding="utf-8",
    )

    config = benchmark.load_search_config(config_path)
    assert config["candidates"] == [replace(BASELINE, update_ms=50.0, frame_size=80)]

    config_path.write_text(
        json.dumps({"candidates": [{"unknown_parameter": 1}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown fields"):
        benchmark.load_search_config(config_path)

    config_path.write_text(
        json.dumps({"candidates": [{"false_accept_prob": 1e-3}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fixed at"):
        benchmark.load_search_config(config_path)


def test_scenario_group_can_be_derived_or_explicit():
    assert Scenario("stationary_25ms", 1).group == "stationary_delay"
    assert Scenario("custom", 1, group="deployment_edge").group == "deployment_edge"
    with pytest.raises(ValueError, match="group must be non-empty"):
        Scenario("custom", 1, group="")


def test_real_audio_veto_includes_configurable_jitter_in_passed(monkeypatch, tmp_path):
    mic_path = tmp_path / "sample_mic.wav"
    ref_path = tmp_path / "sample_lpb.wav"
    fake_soundfile = type(
        "FakeSoundFile",
        (),
        {"read": staticmethod(lambda *args, **kwargs: (np.zeros(480, dtype=np.float32), 16000))},
    )

    class SequenceEstimator:
        def __init__(self):
            self._delays = iter([0.0, 10.0, 20.0])

        def push_ref(self, frame):
            pass

        def push_mic(self, frame):
            pass

        @property
        def delay_ms(self):
            return next(self._delays)

    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setattr(
        benchmark, "discover_real_pairs", lambda root, limit: [(mic_path, ref_path)]
    )
    monkeypatch.setattr(benchmark, "_make_estimator", lambda *args: SequenceEstimator())

    failed = benchmark.real_audio_veto(
        [BASELINE],
        tmp_path,
        sample_rate=16000,
        duration_s=0.03,
        limit=1,
        backend="python",
        jitter_threshold_ms=9.9,
    )
    passed = benchmark.real_audio_veto(
        [BASELINE],
        tmp_path,
        sample_rate=16000,
        duration_s=0.03,
        limit=1,
        backend="python",
        jitter_threshold_ms=10.0,
    )

    assert failed["passed"] is False
    assert failed["profiles"][0]["files"][0]["passed"] is False
    assert passed["passed"] is True
    assert passed["profiles"][0]["max_jitter_p95_ms"] == 10.0


@pytest.fixture
def cli_mocks(monkeypatch):
    challenger = replace(BASELINE, update_ms=50.0)
    dev_results = [_result(BASELINE, score=10.0), _result(challenger, score=8.0)]
    holdout_results = [_result(BASELINE, score=10.0), _result(challenger, score=8.0)]
    state = {
        "calls": [],
        "challenger": challenger,
        "dev_results": dev_results,
        "holdout_results": holdout_results,
        "parity": {
            "available": True,
            "passed": True,
            "tolerance_ms": 1.0,
            "profiles": [],
        },
    }

    def fake_staged_search(search, scenarios, **kwargs):
        state["calls"].append(("search", kwargs["backend"], {item.seed for item in scenarios}))
        assert search == "representative"
        assert kwargs["sample_rate"] == 16000
        assert kwargs["duration_s"] == 3.5
        return dev_results, [{"stage": "representative", "results": dev_results}]

    def fake_evaluate(profiles, scenarios, **kwargs):
        state["calls"].append(("holdout", kwargs["backend"], {item.seed for item in scenarios}))
        assert list(profiles) == [BASELINE, challenger]
        return holdout_results

    monkeypatch.setattr(benchmark, "staged_search", fake_staged_search)
    monkeypatch.setattr(benchmark, "evaluate_profiles", fake_evaluate)
    monkeypatch.setattr(
        benchmark,
        "cpp_extension_identity",
        lambda: {
            "path": str(
                benchmark.ROOT
                / "tasks"
                / "aec"
                / "inference"
                / "_native"
                / "delay_estimator_ext.unit-test.so"
            ),
            "sha256": "a" * 64,
            "mtime_ns": 2,
            "size_bytes": 1,
            "source": {
                "path": str(benchmark.CSRC / "delay_estimator_ext.cpp"),
                "sha256": "b" * 64,
                "mtime_ns": 1,
            },
        },
    )
    monkeypatch.setattr(
        benchmark, "backend_parity", lambda *args, **kwargs: state["parity"]
    )
    return state


def test_minimal_cli_keeps_holdout_out_of_search_and_writes_json_schema(
    cli_mocks, tmp_path, capsys
):
    output = tmp_path / "report.json"

    assert benchmark.main(
        [
            "--search",
            "representative",
            "--backend",
            "python",
            "--duration",
            "3.5",
            "--output",
            str(output),
        ]
    ) == 0
    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_report == file_report
    assert {
        "passed",
        "gate_failures",
        "method",
        "stages",
        "dev_final",
        "holdout",
        "winner",
        "parity",
        "real_audio_veto",
    } <= file_report.keys()
    assert file_report["passed"] is True
    assert file_report["gate_failures"] == []
    assert file_report["benchmark_config"]["false_accept_prob"] == 1e-4
    assert file_report["benchmark_config"]["baseline"]["acps_alpha"] == 0.8
    assert file_report["benchmark_config"]["baseline"]["false_accept_prob"] == 1e-4
    assert file_report["method"] == {
        "oracle": "synthetic delay trajectory only",
        "real_audio_policy": "stability veto only; no real delay accuracy is claimed",
        "sample_rate": 16000,
        "duration_s": 3.5,
        "backend": "python",
        "search": "representative",
    }
    assert file_report["parity"] == cli_mocks["parity"]
    assert file_report["real_audio_veto"] == {
        "enabled": False,
        "policy": "stability veto only; no delay oracle and no accuracy claim",
        "root": None,
        "backend": "python",
        "pair_count": 0,
        "jitter_threshold_ms": 5.0,
        "passed": None,
        "profiles": [],
    }
    assert set(file_report["winner"]) == {
        "selected",
        "profile",
        "name",
        "reason",
        "cpu_ratio_vs_baseline",
        "dev",
        "holdout",
    }
    extension = file_report["benchmark_config"]["cpp_extension"]
    assert extension["path"] == str(
        benchmark.ROOT / "tasks" / "aec" / "inference" / "_native" / Path(extension["path"]).name
    )
    assert Path(extension["path"]).is_absolute()
    assert len(extension["sha256"]) == 64
    assert extension["mtime_ns"] >= extension["source"]["mtime_ns"]
    assert cli_mocks["calls"] == [
        ("search", "python", {item.seed for item in build_scenarios("dev")}),
        ("holdout", "python", {item.seed for item in build_scenarios("holdout")}),
    ]


def test_cpp_extension_must_load_from_native_dir(monkeypatch, tmp_path):
    def _reject(name=None):
        raise ImportError(
            f"refusing non-native-dir extension: {tmp_path / 'delay_estimator_ext.so'}"
        )

    monkeypatch.setattr(benchmark, "_CPP_EXTENSION_IDENTITY", None)
    monkeypatch.setattr(benchmark, "_load_delay_estimator_extension", _reject)

    with pytest.raises(RuntimeError, match="non-native-dir"):
        benchmark.cpp_extension_identity()


def test_benchmark_does_not_prepend_csrc_before_project_root():
    root_index = sys.path.index(str(benchmark.ROOT))
    csrc = str(benchmark.CSRC)
    assert csrc not in sys.path or root_index < sys.path.index(csrc)


@pytest.mark.parametrize("requested_backend", ("cpp", "python"))
def test_make_estimator_validates_stable_backend_identity(
    requested_backend, monkeypatch
):
    class FakeEstimator:
        backend_identity = requested_backend

    estimator = FakeEstimator()
    monkeypatch.setattr(benchmark, "cpp_extension_identity", lambda: {})
    monkeypatch.setattr(benchmark, "create_delay_estimator", lambda cfg: estimator)
    monkeypatch.setattr(benchmark, "StreamingDelayEstimator", lambda cfg: estimator)

    assert benchmark._make_estimator(BASELINE, 16000, requested_backend) is estimator


@pytest.mark.parametrize("requested_backend", ("cpp", "python"))
def test_make_estimator_real_factory_smoke(requested_backend):
    if requested_backend == "cpp":
        try:
            benchmark.cpp_extension_identity()
        except RuntimeError as exc:
            pytest.skip(f"C++ extension is not currently build-valid: {exc}")
    estimator = benchmark._make_estimator(BASELINE, 16000, requested_backend)

    assert estimator.backend_identity == requested_backend


@pytest.mark.parametrize(
    ("requested_backend", "actual_backend"),
    (("cpp", "python"), ("python", "cpp")),
)
def test_make_estimator_rejects_backend_identity_mismatch(
    requested_backend, actual_backend, monkeypatch
):
    class FakeEstimator:
        backend_identity = actual_backend

    estimator = FakeEstimator()
    monkeypatch.setattr(benchmark, "cpp_extension_identity", lambda: {})
    monkeypatch.setattr(benchmark, "create_delay_estimator", lambda cfg: estimator)
    monkeypatch.setattr(benchmark, "StreamingDelayEstimator", lambda cfg: estimator)

    with pytest.raises(RuntimeError, match=f"requested {requested_backend}.*got {actual_backend}"):
        benchmark._make_estimator(BASELINE, 16000, requested_backend)


def test_cli_fails_when_no_profile_passes_hard_gates(cli_mocks, capsys):
    for result in [*cli_mocks["dev_results"], *cli_mocks["holdout_results"]]:
        result["hard_gate"] = False

    assert benchmark.main(["--backend", "python", "--duration", "3.5"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert report["gate_failures"] == ["winner_selection"]
    assert report["winner"]["selected"] is False
    assert report["winner"]["name"] is None
    assert report["winner"]["profile"] is None


@pytest.mark.parametrize(
    ("failed_split", "expected_failure"),
    [("dev", "winner_dev_hard_gate"), ("holdout", "winner_holdout_hard_gate")],
)
def test_cli_requires_winner_dev_and_holdout_hard_gates(
    failed_split, expected_failure, cli_mocks, monkeypatch, capsys
):
    winner = choose_winner(cli_mocks["dev_results"], cli_mocks["holdout_results"])
    winner[failed_split]["hard_gate"] = False
    monkeypatch.setattr(benchmark, "choose_winner", lambda *args: winner)

    assert benchmark.main(["--backend", "python", "--duration", "3.5"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert report["gate_failures"] == [expected_failure]


def test_single_backend_fails_closed_when_parity_is_unavailable(cli_mocks, capsys):
    cli_mocks["parity"] = {
        "available": False,
        "passed": False,
        "reason": "C++ backend unavailable",
        "tolerance_ms": 1.0,
        "profiles": [],
    }

    assert benchmark.main(["--backend", "python", "--duration", "3.5"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert report["gate_failures"] == ["backend_parity"]
    assert report["parity"] == cli_mocks["parity"]
    assert "real_audio_veto" in report


@pytest.mark.parametrize("failed_gate", ["backend_parity", "real_audio_veto"])
def test_both_mode_fails_closed_on_parity_or_real_veto(
    failed_gate, cli_mocks, monkeypatch, tmp_path, capsys
):
    argv = ["--backend", "both", "--duration", "3.5"]
    if failed_gate == "backend_parity":
        cli_mocks["parity"] = {
            "available": True,
            "passed": False,
            "tolerance_ms": 1.0,
            "profiles": [{"passed": False}],
        }
    else:
        argv.extend(["--real-root", str(tmp_path)])
        monkeypatch.setattr(
            benchmark,
            "real_audio_veto",
            lambda *args, **kwargs: {
                "enabled": True,
                "policy": "stability veto only; no delay oracle and no accuracy claim",
                "root": str(tmp_path),
                "backend": "cpp",
                "pair_count": 1,
                "jitter_threshold_ms": kwargs["jitter_threshold_ms"],
                "passed": False,
                "profiles": [{"passed": False}],
            },
        )

    assert benchmark.main(argv) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert report["gate_failures"] == [failed_gate]
    assert "parity" in report
    assert "real_audio_veto" in report


def test_both_mode_requires_a_valid_winner_from_each_backend(
    cli_mocks, monkeypatch, capsys
):
    valid = choose_winner(cli_mocks["dev_results"], cli_mocks["holdout_results"])
    invalid = {
        "selected": False,
        "profile": None,
        "name": None,
        "reason": "no eligible profile",
        "cpu_ratio_vs_baseline": None,
        "dev": None,
        "holdout": None,
    }

    def fake_run_backend(args, backend, search_config):
        return {
            "backend": backend,
            "stages": [],
            "dev_final": [],
            "holdout": [],
            "winner": valid if backend == "python" else invalid,
        }

    monkeypatch.setattr(benchmark, "_run_backend", fake_run_backend)

    assert benchmark.main(["--backend", "both", "--duration", "3.5"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert report["gate_failures"] == ["cpp_winner_selection"]


def test_both_mode_requires_matching_winner_profiles(cli_mocks, monkeypatch, capsys):
    challenger_winner = choose_winner(
        cli_mocks["dev_results"], cli_mocks["holdout_results"]
    )
    baseline_dev = _result(BASELINE, score=1.0)
    baseline_holdout = _result(BASELINE, score=1.0)
    baseline_winner = choose_winner([baseline_dev], [baseline_holdout])

    def fake_run_backend(args, backend, search_config):
        winner = challenger_winner if backend == "python" else baseline_winner
        return {
            "backend": backend,
            "stages": [],
            "dev_final": [],
            "holdout": [],
            "winner": winner,
        }

    monkeypatch.setattr(benchmark, "_run_backend", fake_run_backend)

    assert benchmark.main(["--backend", "both", "--duration", "3.5"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert report["gate_failures"] == ["backend_winner_profile_mismatch"]
