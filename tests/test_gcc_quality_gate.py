"""Smoke tests: tools/verify/gcc_quality_gate.py（JAE-M09）。

覆盖：
1. 小规模合成（2 sources × {0,50}ms × clean）：report.json 字段齐全、
   MAE < 5ms、false_lock == 0、exit 0；
2. 不相关白噪声对（far/echo 独立 randn）：low_confidence 比例 = 1.0；
3. --pairs-json 模式：1 对已知 50ms 延迟 + 1 对不相关噪声 → 聚合字段齐全、
   低置信清单含不相关对；
4. --max-pairs 抽样：5 对抽 2 对，同 seed 两次运行选中子集逐位一致，
   manifest_pair_count 记录全量对数；
5. 单路径 stereo 行（训练 manifest 真实 schema）：echo=ch0、farend=ch1 拆分后
   50ms 延迟被正确估计（高置信、tau≈50ms）。
6. gate FAIL 路径：--max-low-conf-rate -0.1 必然超标 → main() 返回 1 且
   report 的 gate.passed=False、violations 非空；
7. 全样本低置信（mae_ms 为 None）时 false-lock 检查不得 vacuous pass。

音频均 ≤ 2s，直接调 main()/函数，不走 subprocess。
"""
import pytest

import json

import numpy as np
import soundfile as sf

from tools.verify.gcc_quality_gate import (
    _fractional_delay,
    _gate_violations,
    _speech_like,
    aggregate_records,
    main,
    run_estimate,
)

SR = 16000


def test_synthetic_smoke(tmp_path):
    out = tmp_path / "rep"
    rc = main(
        [
            "--out",
            str(out),
            "--taus-ms",
            "0,50",
            "--degrades",
            "clean",
            "--num-sources",
            "2",
            "--duration-sec",
            "2.0",
        ]
    )
    assert rc == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    for key in (
        "git_commit",
        "mode",
        "note",
        "params",
        "per_combination",
        "aggregate",
        "gate",
    ):
        assert key in report
    assert report["mode"] == "synthetic"
    agg = report["aggregate"]
    assert agg["total"] == 4
    assert agg["low_confidence_count"] == 0
    assert agg["false_lock_count"] == 0
    assert agg["mae_ms"] < 5.0


def test_unrelated_noise_all_low_confidence():
    rng = np.random.default_rng(0)
    n = SR * 2
    records = [
        run_estimate(
            rng.standard_normal(n).astype(np.float32),
            rng.standard_normal(n).astype(np.float32),
            SR,
        )
        for _ in range(2)
    ]
    agg = aggregate_records(records)
    assert agg["total"] == 2
    assert agg["low_confidence_rate"] == 1.0


def test_pairs_json_mode(tmp_path):
    n = int(1.5 * SR)
    far_ok = _speech_like(n, SR, seed=3)
    echo_ok = _fractional_delay(far_ok, 50.0, SR)  # 已知 50ms 延迟
    rng = np.random.default_rng(1)
    far_noise = (0.1 * rng.standard_normal(n)).astype(np.float32)
    echo_noise = (0.1 * rng.standard_normal(n)).astype(np.float32)

    d = tmp_path / "wavs"
    d.mkdir()
    stereo_ok = d / "stereo_ok.wav"
    stereo_noise = d / "stereo_noise.wav"
    sf.write(str(stereo_ok), np.stack([echo_ok, far_ok], axis=1), SR)
    sf.write(str(stereo_noise), np.stack([echo_noise, far_noise], axis=1), SR)
    manifest = tmp_path / "pairs.json"
    manifest.write_text(
        json.dumps({"pairs": [str(stereo_ok), str(stereo_noise)]}),
        encoding="utf-8",
    )

    out = tmp_path / "rep"
    rc = main(["--pairs-json", str(manifest), "--out", str(out)])
    assert rc == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "pairs_json"
    agg = report["aggregate"]
    assert agg["total"] == 2
    assert agg["low_confidence_count"] == 1
    for key in ("peak", "prominence", "tau_ms"):
        assert set(agg[key]) == {"p5", "p50", "p95"}
    txt = (out / "low_confidence_pairs.txt").read_text(encoding="utf-8")
    assert "stereo_noise" in txt
    assert "stereo_ok" not in txt


def test_pairs_json_max_pairs_deterministic(tmp_path):
    """--max-pairs 抽样：同 seed 两次运行选中子集必须逐位一致。"""
    n = int(0.5 * SR)
    rng = np.random.default_rng(7)
    d = tmp_path / "wavs"
    d.mkdir()
    pairs = []
    for i in range(5):
        far = (0.1 * rng.standard_normal(n)).astype(np.float32)
        echo = _fractional_delay(far, 50.0, SR)
        stereo_p = d / f"stereo_{i}.wav"
        sf.write(str(stereo_p), np.stack([echo, far], axis=1), SR)
        pairs.append(str(stereo_p))
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")

    def run(out_name):
        out = tmp_path / out_name
        rc = main([
            "--pairs-json", str(manifest), "--out", str(out),
            "--max-pairs", "2", "--sample-seed", "0",
        ])
        assert rc == 0
        return json.loads((out / "report.json").read_text(encoding="utf-8"))

    r1, r2 = run("rep1"), run("rep2")
    assert r1["aggregate"]["total"] == 2
    assert r1["aggregate"]["manifest_pair_count"] == 5
    assert r1["params"]["max_pairs"] == 2
    assert r1["params"]["sample_seed"] == 0
    sel1 = sorted(rec["far"] for rec in r1["pairs_detail"])
    sel2 = sorted(rec["far"] for rec in r2["pairs_detail"])
    assert sel1 == sel2


def test_pairs_json_stereo_single_path(tmp_path):
    """单路径 stereo 行按 echo=ch0/farend=ch1 拆分（与 DynamicMixingDataset 一致）。"""
    n = int(1.5 * SR)
    far = _speech_like(n, SR, seed=5)
    echo = _fractional_delay(far, 50.0, SR)  # 已知 50ms 延迟
    stereo_p = tmp_path / "stereo.wav"
    sf.write(str(stereo_p), np.stack([echo, far], axis=1), SR)
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"pairs": [str(stereo_p)]}), encoding="utf-8")

    out = tmp_path / "rep"
    rc = main(["--pairs-json", str(manifest), "--out", str(out)])
    assert rc == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    agg = report["aggregate"]
    assert agg["total"] == 1
    assert agg["manifest_pair_count"] == 1
    assert agg["skipped_unreadable"] == 0
    # 通道拆对：50ms 延迟应被高置信命中；若通道拆反会锁到 -50ms（出窗→低置信）
    assert agg["low_confidence_count"] == 0
    assert abs(report["pairs_detail"][0]["tau_ms"] - 50.0) < 5.0


def test_gate_fail_returns_1_and_report(tmp_path):
    """gate FAIL 路径：--max-low-conf-rate -0.1 必然超标（low_confidence_rate ≥ 0）。"""
    out = tmp_path / "rep"
    rc = main(
        [
            "--out",
            str(out),
            "--taus-ms",
            "0,50",
            "--degrades",
            "clean",
            "--num-sources",
            "2",
            "--duration-sec",
            "1.0",
            "--max-low-conf-rate",
            "-0.1",
        ]
    )
    assert rc == 1
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is False
    assert report["gate"]["violations"]


def test_gate_false_lock_violation_when_no_high_confidence():
    """全样本低置信（errs 为空、mae_ms 为 None）时 false-lock 检查不得 vacuous pass。"""
    rng = np.random.default_rng(0)
    n = SR * 2
    records = [
        run_estimate(
            rng.standard_normal(n).astype(np.float32),
            rng.standard_normal(n).astype(np.float32),
            SR,
        )
        for _ in range(2)
    ]
    agg = aggregate_records(records)
    # 前提确认：不相关噪声对全部低置信，无高置信样本可算误差
    assert agg["low_confidence_rate"] == 1.0
    assert agg["mae_ms"] is None
    violations = _gate_violations(
        agg,
        max_mae_ms=None,
        max_false_lock_rate=0.01,
        max_low_conf_rate=None,
        has_oracle=True,
    )
    assert any("false_lock" in msg for msg in violations)

def test_empty_and_nonfinite_aggregates_fail_closed():
    empty = aggregate_records([])
    violations = _gate_violations(
        empty,
        max_mae_ms=None,
        max_false_lock_rate=None,
        max_low_conf_rate=None,
        has_oracle=False,
    )
    assert any("没有可评估样本" in message for message in violations)

    invalid = dict(empty, total=1, low_confidence_rate=np.nan)
    violations = _gate_violations(
        invalid,
        max_mae_ms=None,
        max_false_lock_rate=None,
        max_low_conf_rate=None,
        has_oracle=False,
    )
    assert any("low_confidence_rate" in message for message in violations)

    with pytest.raises(ValueError, match="non-finite"):
        aggregate_records(
            [
                {
                    "tau_ms": np.inf,
                    "peak": 0.1,
                    "prominence": 2.0,
                    "low_confidence": False,
                    "time_varying": False,
                    "error_ms": None,
                }
            ]
        )


def test_cli_rejects_nonfinite_synthetic_tau():
    with pytest.raises(SystemExit):
        main(["--taus-ms", "nan"])


def test_run_estimate_rejects_nonfinite_input():
    far = np.ones(32, dtype=np.float32)
    echo = far.copy()
    echo[0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        run_estimate(far, echo, SR)


def test_pairs_json_skipped_unreadable_fails_by_default_and_can_be_bounded(tmp_path):
    n = SR
    far = _speech_like(n, SR, seed=11)
    echo = _fractional_delay(far, 50.0, SR)
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    sf.write(str(wav_dir / "far.wav"), far, SR)
    sf.write(str(wav_dir / "echo.wav"), echo, SR)
    sf.write(str(wav_dir / "stereo.wav"), np.stack([echo, far], axis=1), SR)
    manifest = tmp_path / "pairs.json"
    manifest.write_text(
        json.dumps({"pairs": ["wavs/stereo.wav", "wavs/missing.wav"]}),
        encoding="utf-8",
    )

    failed_out = tmp_path / "failed"
    assert main(["--pairs-json", str(manifest), "--out", str(failed_out)]) == 1
    failed = json.loads((failed_out / "report.json").read_text(encoding="utf-8"))
    assert failed["aggregate"]["total"] == 1
    assert failed["aggregate"]["skipped_unreadable"] == 1
    assert any("skipped_unreadable" in item for item in failed["gate"]["violations"])

    allowed_out = tmp_path / "allowed"
    assert main(
        [
            "--pairs-json",
            str(manifest),
            "--out",
            str(allowed_out),
            "--max-skipped-unreadable",
            "1",
        ]
    ) == 0


pytestmark = [pytest.mark.unit]
