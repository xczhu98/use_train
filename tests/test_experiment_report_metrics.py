"""JAE-L02 / JAE-M06 / JAE-M10 报告与指标侧回归测试。

覆盖：
- SCOREQ legend 标为 SCOREQ-NR 且谱系取自 load_scoreq 实际默认值（JAE-L02）
- SCOREQ ref 模式缺 clean reference 时显式失败（JAE-L02）
- compute_normalized_average 固定 6 指标集合、缺项 N/A、返回分母信息（JAE-M06）
- 报告指标表展示样本数 count 与 Normalized_Average 分母（JAE-M06）
- RESULTS.txt 新增字段不破坏旧行格式解析（JAE-M06）
- ONNX 验证产物谱系与 narrow synthetic 口径标注（JAE-M10）
- deepvqe 三路 verify 在报告流程中固定 CPU 跑 PyTorch 侧（float32 档阈值为 CPU 口径）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest
import torch

import tools.ops.reporting.experiment_report as er
from evaluate.artifact_lifecycle import (
    EVALUATION_RESERVATION_TOKEN_ENV,
    RESERVATION_FILENAME,
    begin_artifact_run,
)
from evaluate.common import publish_result_manifest
from evaluate.eval_pipeline_common import find_exp_assets
from evaluate.pipeline_runner import PipelineTaskAdapter, run_pipeline
from evaluate.evaluate_urgent import (
    NORMALIZED_AVERAGE_METRICS,
    NORMALIZATION_RANGES,
    _load_scoreq_for_eval,
    _read_audio_scp,
    compute_normalized_average,
)
from tools.model_tools import extract_model_checkpoint as extract_ckpt


def _report_update_worker(index_file, report_file, exp_name, model_type, ready, start):
    """跨进程并发更新测试 worker。"""
    import tools.ops.reporting.experiment_report as worker_er

    worker_er.INDEX_FILE = Path(index_file)
    worker_er.REPORT_FILE = Path(report_file)
    worker_er.REPORT_DIR = Path(report_file).parent
    exp = {
        "exp_name": exp_name,
        "exp_dir": f"/tmp/{exp_name}",
        "task_type": "aec",
        "checkpoint_path": f"/tmp/{exp_name}.tar",
        "config_path": f"/tmp/{exp_name}.yaml",
        "timestamp": "2026-08-11T00:00:00",
        "resolved_config": {"model": {"type": model_type, "args": {}}},
        "training_metrics": {},
    }
    ready.put(exp_name)
    start.wait(timeout=10)
    worker_er.update_report_snapshot(exp, {})


def _flock_worker(lock_path, label, events, hold_seconds):
    """跨进程持久 flock 测试 worker。"""
    import tools.ops.reporting.experiment_report as worker_er

    with worker_er._persistent_flock(Path(lock_path)):
        events.put((label, "enter"))
        time.sleep(hold_seconds)
        events.put((label, "exit"))


# ---------------------------------------------------------------------------
# JAE-L02: SCOREQ 标签与谱系
# ---------------------------------------------------------------------------
class TestScoreqLegend:
    def test_legend_marks_scoreq_nr_not_reference_based(self):
        legend = "\n".join(er._build_metric_legend())
        assert "SCOREQ-NR" in legend
        assert "Reference-based" not in legend

    def test_legend_provenance_matches_loader_defaults(self):
        """legend 元数据必须与 core.metrics.mos_models.load_scoreq 默认值一致。"""
        import inspect

        from core.metrics.mos_models import _SCOREQ_ONNX_FILES, load_scoreq

        params = inspect.signature(load_scoreq).parameters
        mode = params["mode"].default
        domain = params["data_domain"].default
        fname = _SCOREQ_ONNX_FILES[(domain, mode)]

        prov = er._scoreq_provenance()
        assert prov["mode"] == mode == "nr"
        assert prov["data_domain"] == domain == "natural"
        assert prov["model_file"] == fname

        legend = "\n".join(er._build_metric_legend())
        assert fname in legend
        assert f"mode={mode}" in legend
        assert f"domain={domain}" in legend
        assert prov["version"] in legend

    def test_ref_mode_without_reference_fails(self):
        """ref 模式缺 clean reference 必须显式失败，不得静默退回 NR。"""
        with pytest.raises(SystemExit):
            _load_scoreq_for_eval(has_ref=False, mode="ref")


@pytest.mark.unit
class TestUrgentAudioScpParsing:
    def test_accepts_two_and_four_columns_with_quoted_space_paths(self, tmp_path):
        scp = tmp_path / "audio.scp"
        scp.write_text(
            'two "/audio/enhanced clips/two.wav"\n'
            'four 16000 32000 "/audio/reference clips/four.flac"\n',
            encoding="utf-8",
        )

        assert _read_audio_scp(scp) == {
            "two": "/audio/enhanced clips/two.wav",
            "four": "/audio/reference clips/four.flac",
        }

    def test_rejects_duplicate_uid(self, tmp_path):
        scp = tmp_path / "duplicate.scp"
        scp.write_text("same first.wav\nsame second.wav\n", encoding="utf-8")

        with pytest.raises(ValueError, match="重复 uid"):
            _read_audio_scp(scp)

    @pytest.mark.parametrize(
        "record, columns",
        [
            ("single.wav", 1),
            ("bad 16000 bad.wav", 3),
            ("bad 16000 32000 bad.wav trailing", 5),
        ],
    )
    def test_rejects_columns_outside_two_or_four(self, tmp_path, record, columns):
        scp = tmp_path / "invalid.scp"
        scp.write_text(f"{record}\n", encoding="utf-8")

        with pytest.raises(ValueError, match=rf"非法列数 {columns}.*仅支持 2/4 列"):
            _read_audio_scp(scp)


class TestModelSizeLoader:
    @pytest.mark.unit
    def test_safe_loader_failure_is_fail_closed(self, monkeypatch, tmp_path):
        """安全 loader 失败时不应静默重试 unsafe pickle。"""
        checkpoint = tmp_path / "checkpoint.tar"
        checkpoint.write_bytes(b"not a checkpoint")
        calls = []

        def fake_safe_load(path, **kwargs):
            calls.append((path, kwargs))
            raise RuntimeError("weights-only load rejected checkpoint")

        import core.utils.safe_load as safe_load

        monkeypatch.setattr(safe_load, "safe_torch_load", fake_safe_load)

        assert er._infer_model_size({"checkpoint_path": str(checkpoint)}) == ""
        assert calls == [
            (str(checkpoint), {"purpose": "param count inference"}),
        ]


# ---------------------------------------------------------------------------
# JAE-M06: 固定指标集合的 Normalized_Average
# ---------------------------------------------------------------------------
class TestNormalizedAverage:
    def _full_means(self):
        return {
            "DNSMOS": 3.0, "NISQA": 3.0, "UTMOS": 3.0, "SCOREQ": 3.0,
            "PESQ": 2.0, "ESTOI": 0.5,
        }

    def test_fixed_set_is_preregistered_and_matches_ranges(self):
        """固定集合预注册为模块常量，且与归一化区间表一一对应。"""
        assert isinstance(NORMALIZED_AVERAGE_METRICS, tuple)
        assert len(NORMALIZED_AVERAGE_METRICS) == 6
        assert set(NORMALIZED_AVERAGE_METRICS) == set(NORMALIZATION_RANGES)

    def test_full_set_returns_value_and_denominator(self):
        result = compute_normalized_average(self._full_means())
        assert result["value"] is not None
        assert result["available"] == 6
        assert result["required"] == 6
        assert result["missing"] == []
        # 手工核算：每项归一化后均为 0.5
        assert result["value"] == pytest.approx(0.5)

    def test_missing_any_metric_marks_na_with_denominator(self):
        """缺项时返回可用指标的部分平均，分母如实标注 k/6（k<6 非官方口径）。"""
        result = compute_normalized_average(
            {"DNSMOS": 3.0, "NISQA": 3.0, "UTMOS": 3.0, "SCOREQ": 3.0}
        )
        # 4 项可用 → 部分平均（非 None）；缺 PESQ/ESTOI
        assert result["value"] is not None
        assert result["available"] == 4
        assert result["required"] == 6
        assert set(result["missing"]) == {"PESQ", "ESTOI"}
        assert result["complete"] is False

    def test_none_value_counts_as_missing(self):
        means = self._full_means()
        means["PESQ"] = None
        result = compute_normalized_average(means)
        assert result["missing"] == ["PESQ"]
        assert result["complete"] is False
        # None 计为缺失，其余 5 项仍可求部分平均
        assert result["value"] is not None

    def test_zero_available_returns_none(self):
        result = compute_normalized_average({"DNSMOS": None, "NISQA": None})
        assert result["value"] is None
        assert result["available"] == 0


# ---------------------------------------------------------------------------
# JAE-M06: 报告表格的 count / 分母展示
# ---------------------------------------------------------------------------
class TestMetricsTable:
    def test_cells_show_count_and_denominator(self):
        lines = []
        er._build_metrics_table(lines, {
            "voicebank": {
                "PESQ": 2.5, "_count_PESQ": 500.0,
                "SCOREQ": 3.1, "_count_SCOREQ": 500.0,
                "Normalized_Average": 0.8123,
                "Normalized_Average_Denominator": "6/6",
            },
        })
        table = "\n".join(lines)
        assert "2.5000 (n=500)" in table
        assert (
            "0.8123 (legacy/partial projection; dimensions=2/6; "
            "observed=SCOREQ=500,PESQ=500; "
            "missing=DNSMOS,NISQA,UTMOS,ESTOI)"
        ) in table
        # 历史 RESULTS 的 denominator 不能覆盖按实际指标重建的 dimensions。
        assert "0.8123 (6/6)" not in table
        header = lines[0]
        assert "_count_" not in header
        assert "Normalized_Average_Denominator" not in header

    def test_na_denominator_is_not_recomputed(self):
        lines = []
        er._build_metrics_table(lines, {
            "dns_challenge": {
                "DNSMOS": 3.0, "_count_DNSMOS": 300.0,
                "Normalized_Average": "N/A",
                "Normalized_Average_Denominator": "4/6",
            },
        })
        table = "\n".join(lines)
        assert (
            "N/A (legacy/partial projection; dimensions=1/6; observed=DNSMOS=300; "
            "missing=NISQA,UTMOS,SCOREQ,PESQ,ESTOI)"
        ) in table
        assert "0.5000" not in table

    def test_complete_historical_counts_still_render_as_legacy_partial(self):
        metrics = {
            metric: 3.0 for metric in er.NORMALIZED_METRICS
        }
        metrics.update({
            "PESQ": 2.0,
            "ESTOI": 0.5,
            "Normalized_Average": 0.5,
            **{f"_count_{metric}": 10 for metric in er.NORMALIZED_METRICS},
        })

        rendered = er._format_normalized_score(metrics)

        assert rendered == (
            "0.5000 (legacy/partial projection; dimensions=6/6; "
            "observed=DNSMOS=10,NISQA=10,UTMOS=10,SCOREQ=10,PESQ=10,ESTOI=10)"
        )
        assert "official" not in rendered

    def test_nested_scenario_cells_annotated(self):
        """aec_sim 嵌套场景同样保留历史值并标明未知覆盖。"""
        lines = []
        er._build_metrics_table(lines, {
            "aec_sim": {
                "doubletalk": {
                    "ERLE_dB": 25.0, "_count_ERLE_dB": 100.0,
                    "Normalized_Average": 0.7,
                    "Normalized_Average_Denominator": "6/6",
                },
            },
        })
        table = "\n".join(lines)
        assert "25.0000 (n=100)" in table
        assert (
            "0.7000 (legacy/unknown projection; dimensions=0/6; "
            "missing=DNSMOS,NISQA,UTMOS,SCOREQ,PESQ,ESTOI)"
        ) in table

    def test_ablation_and_cli_use_partial_projection_formatter(self, capsys):
        metrics = {
            "DNSMOS": 3.0,
            "NISQA": 3.0,
            "UTMOS": 3.0,
            "SCOREQ": 3.0,
            "PESQ": 2.0,
            "ESTOI": 0.5,
            "Normalized_Average": 0.5,
            "_count_DNSMOS": 1023,
            "_count_NISQA": 1023,
            "_count_UTMOS": 1023,
            "_count_SCOREQ": 1023,
            "_count_PESQ": 1022,
            "_count_ESTOI": 1023,
        }
        exp = {"eval_results": {"aec_sim": {"prob_target_only": metrics}}}

        extracted = er._extract_main_metrics(exp)
        er._print_summary(exp, exp["eval_results"])
        stdout = capsys.readouterr().out

        expected = (
            "0.5000 (legacy/partial projection; dimensions=6/6; "
            "observed=DNSMOS=1023,NISQA=1023,UTMOS=1023,SCOREQ=1023,"
            "PESQ=1022,ESTOI=1023)"
        )
        assert extracted["aec_sim_clean_NormAvg"] == expected
        assert f"NormAvg={expected}" in stdout

    def test_scoreq_column_displayed_as_scoreq_nr(self):
        lines = []
        er._build_metrics_table(lines, {"voicebank": {"SCOREQ": 3.1}})
        assert "SCOREQ-NR" in lines[0]

    def test_simplify_eval_results_keeps_annotations(self):
        simplified = er._simplify_eval_results({
            "voicebank": {
                "PESQ": 2.5, "_count_PESQ": 500.0,
                "Normalized_Average": 0.8,
                "Normalized_Average_Denominator": "6/6",
                "SOME_OTHER_METRIC": 1.0,
            },
        })
        entry = simplified["voicebank"]
        assert entry["_count_PESQ"] == 500.0
        assert entry["Normalized_Average_Denominator"] == "6/6"
        assert "SOME_OTHER_METRIC" not in entry

    def test_simplify_eval_results_keeps_nearend_sisdr(self):
        """NEAREND_SISDR_dB 与 ERLE 成对保留进索引（aec_sim 嵌套场景）。"""
        simplified = er._simplify_eval_results({
            "aec_sim": {
                "prob_target_echo": {
                    "ERLE_dB": 11.2, "NEAREND_SISDR_dB": 12.3,
                    "PESQ": 2.5, "SOME_OTHER_METRIC": 1.0,
                },
            },
        })
        scen = simplified["aec_sim"]["prob_target_echo"]
        assert scen["ERLE_dB"] == 11.2
        assert scen["NEAREND_SISDR_dB"] == 12.3
        assert "SOME_OTHER_METRIC" not in scen

    def test_is_annotation_key_excludes_annotations(self):
        # _collect_common_metrics 已在 919c59b 中删除，其职责（排除 _count_* 与
        # Normalized_Average_Denominator 注解键）由 _is_annotation_key 内联承接
        assert er._is_annotation_key("_count_PESQ")
        assert er._is_annotation_key("Normalized_Average_Denominator")
        assert not er._is_annotation_key("PESQ")


# ---------------------------------------------------------------------------
# JAE-M06: RESULTS.txt 解析兼容
# ---------------------------------------------------------------------------
class TestResultsTxtCompat:
    def test_new_fields_do_not_break_old_line_format(self, tmp_path):
        p = tmp_path / "RESULTS.txt"
        p.write_text(
            "PESQ: 2.5000\n"
            "DNSMOS: 3.1000\n"
            "_count_PESQ: 500\n"
            "\n"
            "Normalized_Average: 0.8123\n"
            "Normalized_Average_Denominator: 6/6\n",
            encoding="utf-8",
        )
        metrics = er._parse_results_txt(p)
        # 旧行仍是 float；新增字段可解析且不破坏既有键
        assert metrics["PESQ"] == 2.5
        assert metrics["Normalized_Average"] == 0.8123
        assert metrics["_count_PESQ"] == 500.0
        assert metrics["Normalized_Average_Denominator"] == "6/6"

    def test_na_value_parses_as_string(self, tmp_path):
        p = tmp_path / "RESULTS.txt"
        p.write_text(
            "DNSMOS: 3.1000\n"
            "\n"
            "Normalized_Average: N/A\n"
            "Normalized_Average_Denominator: 4/6\n",
            encoding="utf-8",
        )
        metrics = er._parse_results_txt(p)
        assert metrics["Normalized_Average"] == "N/A"
        assert metrics["Normalized_Average_Denominator"] == "4/6"

    def test_na_with_no_available_metric_stays_na(self, tmp_path):
        p = tmp_path / "RESULTS.txt"
        p.write_text(
            "Normalized_Average: N/A\n"
            "Normalized_Average_Denominator: 0/6\n",
            encoding="utf-8",
        )
        metrics = er._parse_results_txt(p)
        metrics = er._parse_results_txt(p)
        assert metrics["Normalized_Average"] == "N/A"


# ---------------------------------------------------------------------------
# JAE-M10: ONNX 验证产物谱系
# ---------------------------------------------------------------------------
class TestOnnxVerifyProvenance:
    def test_artifact_provenance_sha256(self, tmp_path):
        f = tmp_path / "ckpt.tar"
        f.write_bytes(b"fake-checkpoint-bytes")
        prov = er._artifact_provenance(f)
        assert prov["sha256"] == hashlib.sha256(b"fake-checkpoint-bytes").hexdigest()
        assert prov["size_bytes"] == len(b"fake-checkpoint-bytes")
        assert "mtime" in prov

    def test_artifact_provenance_missing_file(self, tmp_path):
        prov = er._artifact_provenance(tmp_path / "missing.tar")
        assert "sha256" not in prov
        assert "error" in prov

    def test_parse_kept_onnx_artifact(self):
        stdout = (
            "Exporting ONNX to /tmp/tmpabc.onnx ...\n"
            "  ONNX artifact kept: /tmp/tmpabc.onnx (12.0 MB, sha256=deadbeef123)\n"
        )
        path, kept, digest = er._parse_onnx_artifact_from_stdout(stdout)
        assert path == "/tmp/tmpabc.onnx"
        assert kept is True
        assert digest == "deadbeef123"

    def test_parse_temp_onnx_deleted_legacy(self):
        path, kept, digest = er._parse_onnx_artifact_from_stdout(
            "Exporting ONNX to /tmp/tmpxyz.onnx ...\n"
        )
        assert path == "/tmp/tmpxyz.onnx"
        assert kept is False
        assert digest is None

    def test_parse_existing_onnx_reuse(self):
        path, kept, _ = er._parse_onnx_artifact_from_stdout(
            "Using existing ONNX: onnx/onnx_model/foo.onnx\n"
        )
        assert path == "onnx/onnx_model/foo.onnx"
        assert kept is True

    def test_parse_dbgnet_state_and_istft_release_gates(self):
        gates = er._parse_dbgnet_release_gates(
            "  use_state_detector=True, state_nclass=4\n"
            "  (A) 离线 PT vs 流式 PT:\n"
            "      State MSE=1.250000e-07  max_abs=2.500000e-04\n"
            "  (B) 流式 PT vs 流式 ORT (逐帧, 1200 frames):\n"
            "      State MSE=2.500000e-07  max_abs=5.000000e-04\n"
            "  State gate: PASS (PT frames=1200/1200, ORT frames=1200/1200)\n"
            "  External iSTFT synthesis capability: PASS "
            "(finite non-empty waveform only; not streaming waveform equivalence)\n"
        )

        assert gates["audit_status"] == "PASS"
        assert gates["state_detector_enabled"] is True
        assert gates["state_metrics"] == [
            {
                "comparison": "(A) 离线 PT vs 流式 PT",
                "mse": pytest.approx(1.25e-7),
                "max_abs": pytest.approx(2.5e-4),
            },
            {
                "comparison": "(B) 流式 PT vs 流式 ORT (逐帧, 1200 frames)",
                "mse": pytest.approx(2.5e-7),
                "max_abs": pytest.approx(5e-4),
            },
        ]
        assert gates["state_gate"]["status"] == "PASS"
        assert gates["external_istft_synthesis_capability"]["status"] == "PASS"
        assert gates["missing"] == []

    def test_parse_dbgnet_disabled_state_still_requires_istft_gate(self):
        audited = er._parse_dbgnet_release_gates(
            "use_state_detector=False, state_nclass=4\n"
            "External iSTFT synthesis capability: PASS (finite non-empty waveform only)\n"
        )
        missing_istft = er._parse_dbgnet_release_gates(
            "use_state_detector=False, state_nclass=4\n总结: ALL PASS\n"
        )

        assert audited["audit_status"] == "PASS"
        assert audited["state_gate"] is None
        assert audited["state_metrics"] == []
        assert missing_istft["audit_status"] == "UNAUDITABLE"
        assert missing_istft["missing"] == ["External iSTFT synthesis capability"]

    def test_parse_dbgnet_enabled_state_missing_metrics_is_unavailable(self):
        gates = er._parse_dbgnet_release_gates(
            "use_state_detector=True, state_nclass=4\n"
            "State gate: PASS (PT frames=1200/1200, ORT frames=1200/1200)\n"
            "External iSTFT synthesis capability: PASS (finite non-empty waveform only)\n"
        )

        assert gates["audit_status"] == "UNAUDITABLE"
        assert gates["missing"] == ["State MSE/max_abs (A)", "State MSE/max_abs (B)"]

    def test_parse_dbgnet_explicit_release_gate_failure(self):
        gates = er._parse_dbgnet_release_gates(
            "use_state_detector=False, state_nclass=4\n"
            "External iSTFT synthesis capability: FAIL (empty waveform)\n"
        )

        assert gates["audit_status"] == "FAIL"
        assert gates["external_istft_synthesis_capability"]["status"] == "FAIL"

    def test_detail_renders_dbgnet_release_gates_independently(self):
        release_gates = er._parse_dbgnet_release_gates(
            "use_state_detector=True, state_nclass=4\n"
            "(A) offline vs streaming:\n"
            "State MSE=1.000000e-07  max_abs=2.000000e-04\n"
            "(B) streaming PT vs ORT:\n"
            "State MSE=3.000000e-07  max_abs=4.000000e-04\n"
            "State gate: PASS (PT frames=1200/1200, ORT frames=1200/1200)\n"
            "External iSTFT synthesis capability: PASS (finite non-empty waveform only)\n"
        )
        detail = "\n".join(er._build_experiment_detail({
            "exp_name": "dbgnet-demo",
            "task_type": "aec",
            "model_type": "DBGNet",
            "onnx_verify": {
                "family": "dbgnet",
                "status": "PASS",
                "checks": [],
                "release_gates": release_gates,
                "scope": {"validation_scope": "widened", "frames": "1200", "reset": "2"},
            },
        }))

        assert "**ONNX 三路精度验证** (离线 PT / 流式 PT / 流式 ORT): **PASS**" in detail
        assert "**DBGNet 独立发布门禁**" in detail
        assert "MSE=1.000000e-07, max_abs=2.000000e-04" in detail
        assert "MSE=3.000000e-07, max_abs=4.000000e-04" in detail
        assert "| State gate | `PT frames=1200/1200, ORT frames=1200/1200` | PASS |" in detail
        assert "| External iSTFT synthesis capability | `finite non-empty waveform only` | PASS |" in detail

    def test_detail_marks_legacy_dbgnet_pass_unavailable(self):
        detail = "\n".join(er._build_experiment_detail({
            "exp_name": "legacy-dbgnet",
            "task_type": "aec",
            "model_type": "DBGNet",
            "onnx_verify": {
                "family": "dbgnet",
                "status": "PASS",
                "checks": [],
                "scope": {"validation_scope": "widened"},
            },
        }))

        assert "**不可审计**" in detail
        assert "widened synthetic PASS" not in detail
        assert "不得依据尾部汇总 PASS" in detail

    def test_detail_marks_narrow_synthetic_pass(self):
        """报告必须标注 narrow synthetic PASS，不得让读者以为是 deployment PASS。"""
        exp = {
            "exp_name": "demo",
            "task_type": "aec",
            "onnx_verify": {
                "status": "PASS",
                "checks": [],
                "log_path": "/tmp/onnx_verify.log",
                "checkpoint_artifact": {"path": "/tmp/ckpt.tar", "sha256": "ab" * 32,
                                        "size_bytes": 1024, "mtime": "2026-07-15T00:00:00"},
                "onnx_temp_path": "/tmp/tmpabc.onnx",
                "onnx_artifact_kept": True,
                "onnx_artifact_sha256": "cd" * 32,
                "scope": {"frames": "164", "reset": "未覆盖", "tail_block": "未覆盖"},
            },
        }
        detail = "\n".join(er._build_experiment_detail(er._to_public_record(exp)))
        assert "narrow synthetic PASS" in detail
        assert "不代表 deployment PASS" in detail
        assert "frames=164" in detail
        assert ("ab" * 32) in detail  # checkpoint SHA256 入报告
        assert "<external>/tmpabc.onnx" in detail  # 仓外 ONNX 仅展示安全语义路径

    def test_detail_marks_widened_synthetic_pass(self):
        """widened scope 必须展示 frames/reset，且不复用 narrow 文案。"""
        exp = {
            "exp_name": "demo",
            "task_type": "aec",
            "onnx_verify": {
                "status": "PASS",
                "checks": [],
                "scope": {
                    "validation_scope": "widened", "frames": "1000", "reset": "2",
                    "tail_block": "整段 STFT 预计算谱图，逐帧送模；未覆盖 waveform 尾块/任意分块",
                    "inputs": "随机白噪声 mic/far waveform，经配置 STFT 预计算谱图",
                },
            },
        }
        detail = "\n".join(er._build_experiment_detail(exp))
        assert "widened synthetic PASS" in detail
        assert "frames=1000" in detail
        assert "resets=2" in detail
        assert "3 段谱图帧序列" in detail
        assert "段间同步 cold reset cache" in detail
        assert "不覆盖流式 STFT/iSTFT" in detail
        assert "narrow synthetic" not in detail


# ---------------------------------------------------------------------------
# deepvqe verify 固定 CPU（用户决策：float32 档阈值为 CPU 口径）
# ---------------------------------------------------------------------------
class TestDeepvqeVerifyPinnedCpu:
    def _exp_data(self):
        return {
            "checkpoint_path": "/tmp/ckpt.tar",
            "config_path": "/tmp/cfg.yaml",
            "resolved_config": {
                "model": {"args": {"variant": "base", "align_delay": 100}},
                "stft": {"n_fft": 512, "hop_length": 128, "center": True},
            },
        }

    def test_deepvqe_ignores_caller_device(self):
        """deepvqe 阈值（1e-9/1e-4）是 CPU 口径，GPU 设备差必 FAIL，固定 cpu。"""
        cmd = er._build_onnx_verify_cmd(
            "deepvqe", Path("onnx/verify/verify_3way_deepvqe.py"),
            self._exp_data(), "cuda:0")
        assert cmd[cmd.index("--device") + 1] == "cpu"

    def test_deepvqe_command_uses_widened_scope_and_save_onnx(self):
        cmd = er._build_onnx_verify_cmd(
            "deepvqe", Path("onnx/verify/verify_3way_deepvqe.py"), self._exp_data(), "cuda:0")
        assert cmd[cmd.index("--frames") + 1] == "1000"
        assert cmd[cmd.index("--resets") + 1] == "2"
        assert cmd[cmd.index("--device") + 1] == "cpu"
        assert "--save-onnx" in cmd

    def test_deepvqe_zero_align_delay_is_preserved(self):
        exp_data = self._exp_data()
        exp_data["resolved_config"]["model"]["args"]["align_delay"] = 0
        cmd = er._build_onnx_verify_cmd(
            "deepvqe", Path("onnx/verify/verify_3way_deepvqe.py"), exp_data, "cpu")
        assert cmd[cmd.index("--align-delay") + 1] == "0"
        assert cmd[cmd.index("--frames") + 1] == "1000"

    def test_deepvqe_missing_align_delay_keeps_verifier_default(self):
        exp_data = self._exp_data()
        del exp_data["resolved_config"]["model"]["args"]["align_delay"]
        cmd = er._build_onnx_verify_cmd(
            "deepvqe", Path("onnx/verify/verify_3way_deepvqe.py"), exp_data, "cpu")
        assert "--align-delay" not in cmd
        assert cmd[cmd.index("--frames") + 1] == "1000"

    @pytest.mark.parametrize("align_delay", [400, 500])
    def test_deepvqe_large_delay_keeps_each_reset_segment_comparable(self, align_delay):
        exp_data = self._exp_data()
        exp_data["resolved_config"]["model"]["args"]["align_delay"] = align_delay
        cmd = er._build_onnx_verify_cmd(
            "deepvqe", Path("onnx/verify/verify_3way_deepvqe.py"), exp_data, "cpu")
        assert int(cmd[cmd.index("--frames") + 1]) >= 3 * (align_delay + 1)
        assert cmd[cmd.index("--resets") + 1] == "2"

    def test_run_onnx_verification_records_widened_scope_from_command(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, "_PROJECT_ROOT", tmp_path)
        exp_data = self._exp_data()
        exp_data.update(exp_name="deepvqe-demo", task_type="aec")
        exp_data["resolved_config"]["model"]["type"] = "DeepVQE"

        def fake_run(cmd, **_kwargs):
            target = Path(cmd[cmd.index("--save-onnx") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"verified onnx")
            digest = hashlib.sha256(b"verified onnx").hexdigest()
            return argparse.Namespace(
                returncode=0,
                stdout=f"总结: ALL PASS\nONNX artifact kept: {target} (sha256={digest})\n",
                stderr="",
            )

        monkeypatch.setattr(er.subprocess, "run", fake_run)
        result = er.run_onnx_verification(exp_data, device="cuda:0")
        assert result["scope"]["validation_scope"] == "widened"
        assert result["scope"]["frames"] == "1000"
        assert result["scope"]["reset"] == "2"
        assert result["scope"]["tail_block"] == (
            "整段 STFT 预计算谱图，逐帧送模；未覆盖 waveform 尾块/任意分块"
        )

    def test_dbgnet_command_uses_widened_scope_and_cpu_reference(self):
        """DBGNet 报告验证固定 1200 帧/2 reset，PyTorch 参考固定 CPU 口径。"""
        cmd = er._build_onnx_verify_cmd(
            "dbgnet", Path("onnx/verify/verify_3way_dbgnet.py"),
            self._exp_data(), "cuda:0")
        assert cmd[cmd.index("--frames") + 1] == "1200"
        assert cmd[cmd.index("--resets") + 1] == "2"
        assert cmd[cmd.index("--device") + 1] == "cpu"

    def test_dbgnet_verification_records_widened_scope(self, monkeypatch, tmp_path):
        """DBGNet scope DTO 必须准确记录 widened 输入、reset 与 tail 边界。"""
        monkeypatch.setattr(er, "_PROJECT_ROOT", tmp_path)
        exp_data = self._exp_data()
        exp_data.update(exp_name="dbgnet-demo", task_type="aec")
        exp_data["resolved_config"]["model"]["type"] = "DBGNet"

        def fake_run(cmd, **_kwargs):
            target = Path(cmd[cmd.index("--save-onnx") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"verified onnx")
            digest = hashlib.sha256(b"verified onnx").hexdigest()
            return argparse.Namespace(
                returncode=0,
                stdout=(
                    "use_state_detector=False, state_nclass=4\n"
                    "External iSTFT synthesis capability: PASS "
                    "(finite non-empty waveform only)\n"
                    f"总结: ALL PASS\nONNX artifact kept: {target} (sha256={digest})\n"
                ),
                stderr="",
            )

        monkeypatch.setattr(er.subprocess, "run", fake_run)
        result = er.run_onnx_verification(exp_data, device="cuda:0")
        assert result["status"] == "PASS"
        assert result["release_gates"]["audit_status"] == "PASS"
        assert result["scope"] == {
            "frames": "1200",
            "reset": "2",
            "tail_block": "整段 STFT 预计算谱图，逐帧送模；未覆盖 waveform 尾块/任意分块",
            "inputs": "随机白噪声 mic/far waveform，经配置 STFT 预计算谱图",
            "provider": "CPU ORT",
            "onnx_artifact": "临时 float ONNX",
            "validation_scope": "widened",
        }


class TestAecLongAudioDevicePolicy:
    @pytest.mark.parametrize(
        ("requested", "num_samples", "threshold", "expected_type"),
        [
            ("cuda:0", 1, 0.0, "cuda"),
            ("cuda:0", 30 * 16000, 30.0, "cuda"),
            ("cuda:0", 30 * 16000 + 1, 30.0, "cpu"),
            ("cpu", 30 * 16000 + 1, 30.0, "cpu"),
        ],
    )
    def test_resolve_aec_file_device_uses_strict_raw_duration_boundary(
        self, requested, num_samples, threshold, expected_type
    ):
        from tasks.aec.inference.device_policy import resolve_aec_file_device

        resolved = resolve_aec_file_device(
            requested, num_samples, 16000, threshold
        )

        assert resolved.type == expected_type

    def test_artifact_fingerprint_changes_when_cpu_threshold_changes(
        self, tmp_path
    ):
        from evaluate.inference_components import build_artifact_fingerprint

        config = tmp_path / "config.yaml"
        checkpoint = tmp_path / "model.tar"
        source = tmp_path / "source.py"
        config.write_text("model: {type: Test}\n", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
        source.write_text("VALUE = 1\n", encoding="utf-8")

        args = argparse.Namespace(
            checkpoint=str(checkpoint),
            config=str(config),
            device="cpu",
            sr=None,
            chunk_seconds=0.0,
            cpu_long_audio_seconds=30.0,
            aec_stream_align=False,
            aec_no_stream_align=False,
        )
        fingerprint_kwargs = {
            "project_root": Path.cwd(),
            "inference_entry": Path("inference.py"),
            "is_aec": True,
            "is_waveform_native": False,
            "source_files": {"source:test.py": source},
        }
        first = build_artifact_fingerprint(
            args, {"model": {"type": "Test"}}, **fingerprint_kwargs
        )

        args.cpu_long_audio_seconds = 31.0
        changed = build_artifact_fingerprint(
            args, {"model": {"type": "Test"}}, **fingerprint_kwargs
        )

        assert first["parameters"]["cpu_long_audio_seconds"] == 30.0
        assert changed["parameters"]["cpu_long_audio_seconds"] == 31.0
        assert first["sha256"] != changed["sha256"]


# ---------------------------------------------------------------------------
# 族内对比表与详情表同口径渲染（JAE-M06）
# ---------------------------------------------------------------------------
class TestComparisonTable:
    def test_comparison_table_uses_format_metric_cell(self):
        """对比表单元格复用 count 与 historical projection 渲染。"""
        exps = [{
            "exp_name": "demo_exp",
            "timestamp": "2026-07-15T10:00:00",
            "eval_results": {
                "voicebank": {
                    "PESQ": 2.5,
                    "_count_PESQ": 500,
                    "Normalized_Average": 0.8123,
                    "Normalized_Average_Denominator": "6/6",
                },
            },
        }]
        table = "\n".join(er._build_comparison_table(exps))
        assert "2.5000 (n=500)" in table
        assert (
            "0.8123 (legacy/partial projection; dimensions=1/6; observed=PESQ=500; "
            "missing=DNSMOS,NISQA,UTMOS,SCOREQ,ESTOI)"
        ) in table


class TestExperimentRootNormalization:
    @pytest.mark.parametrize("run_kind", ["checkpoints", "logs"])
    def test_discover_normalizes_root_but_keeps_run_assets(self, monkeypatch, tmp_path, run_kind):
        root = tmp_path / "deepvqe_small_10ms_delay1000ms_16k"
        run_name = "2026-08-11_10-00-00"
        run_dir = root / run_kind / run_name
        run_dir.mkdir(parents=True)
        captured = {}
        checkpoint_run = root / "checkpoints" / run_name
        config = checkpoint_run / "resolved_config.yaml"
        checkpoint = checkpoint_run / "best_model.tar"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("model: {type: DeepVQE}\n", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")

        def fake_find_exp_assets(path, *, prefer_model_only=True):
            captured["path"] = path
            captured["prefer_model_only"] = prefer_model_only
            return config, checkpoint

        monkeypatch.setattr(er, "find_exp_assets", fake_find_exp_assets)
        monkeypatch.setattr(er, "detect_model_type", lambda _config, **_kwargs: "aec")
        monkeypatch.setattr(er, "load_merged_config", lambda _path: {"model": {"type": "DeepVQE"}})

        exp_data = er.discover_exp_assets(run_dir)

        assert captured["path"] == checkpoint_run.resolve()
        assert captured["prefer_model_only"] is False
        assert exp_data["exp_dir"] == str(root.resolve())
        assert exp_data["exp_name"] == "deepvqe_small_10ms_delay1000ms_16k"

    @pytest.mark.unit
    def test_discover_logs_run_requires_matching_checkpoint_run(self, tmp_path):
        logs_run = tmp_path / "demo" / "logs" / "missing-run"
        logs_run.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="缺少同名 checkpoint run"):
            er.discover_exp_assets(logs_run)

    def test_root_path_is_unchanged(self, tmp_path):
        root = tmp_path / "deepvqe_small_10ms_delay1000ms_16k"
        root.mkdir()

        assert er._normalize_experiment_root(root) == root.resolve()


class TestReportWorkflow:
    def _exp_data(self, **overrides):
        data = {
            "exp_name": "safe-exp",
            "exp_dir": "/tmp/safe-exp",
            "task_type": "aec",
            "checkpoint_path": "/tmp/model.tar",
            "config_path": "/tmp/config.yaml",
            "timestamp": "2026-08-11T00:00:00",
            "resolved_config": {
                "model": {"type": "DBGNet", "args": {}},
                "data": {"sample_rate": 16000},
            },
            "training_metrics": {},
        }
        data.update(overrides)
        return data

    def test_find_eval_root_falls_back_to_checkpoint_run_directory(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, "_PROJECT_ROOT", tmp_path)
        checkpoint = tmp_path / "exp/aec/demo/checkpoints/2026-08-06_22-00-14/best_model.tar"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        semantic_root = tmp_path / "exp/eval/aec/demo"
        semantic_root.mkdir(parents=True)
        legacy_root = tmp_path / "exp/eval/aec/2026-08-06_22-00-14"
        results_dir = legacy_root / "dns_challenge"
        results_dir.mkdir(parents=True)
        (results_dir / "RESULTS.txt").write_text("DNSMOS: 2.0\n")
        (legacy_root / "summary.json").write_text(json.dumps({
            "checkpoint": "exp/aec/demo/checkpoints/2026-08-06_22-00-14/best_model.tar",
        }))

        result = er._find_eval_root({
            "exp_name": "demo",
            "task_type": "aec",
            "checkpoint_path": str(checkpoint),
        })

        assert result == legacy_root

    def test_find_eval_root_prefers_nonempty_semantic_directory(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, "_PROJECT_ROOT", tmp_path)
        checkpoint = tmp_path / "exp/aec/demo/checkpoints/run/best_model.tar"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        semantic_results = tmp_path / "exp/eval/aec/demo/dns_challenge"
        semantic_results.mkdir(parents=True)
        (semantic_results / "RESULTS.txt").write_text("DNSMOS: 2.0\n")
        legacy_results = tmp_path / "exp/eval/aec/run/dns_challenge"
        legacy_results.mkdir(parents=True)
        (legacy_results / "RESULTS.txt").write_text("DNSMOS: 1.0\n")

        result = er._find_eval_root({
            "exp_name": "demo",
            "task_type": "aec",
            "checkpoint_path": str(checkpoint),
        })

        assert result == tmp_path / "exp/eval/aec/demo"

    def test_cmd_evaluate_uses_discovered_experiment_root(self, monkeypatch):
        assert er.AEC_DATASETS == "aec_official,urgent,dns_challenge"
        captured = {}
        canonical_root = "/tmp/deepvqe_small_10ms_delay1000ms_16k"
        monkeypatch.setattr(
            er, "discover_exp_assets",
            lambda _: {"task_type": "aec", "exp_name": "deepvqe_small_10ms_delay1000ms_16k", "exp_dir": canonical_root},
        )
        monkeypatch.setattr(er, "run_eval_pipeline", lambda **kwargs: captured.update(kwargs) or 0)
        args = argparse.Namespace(
            exp_dir=f"{canonical_root}/checkpoints/2026-08-11_10-00-00", datasets="aec_sim",
            no_audio_quality=False, device="cpu", inference_devices="cpu,cuda:1",
            aec_no_stream_align=False, skip_existing_inference=False,
            aec_testset_root="custom/aec",
            urgent_noisy_scp="custom/urgent_noisy.scp",
            urgent_clean_scp="custom/urgent_clean.scp",
            voicebank_clean_scp="custom/clean.scp",
            voicebank_noisy_scp="custom/noisy.scp",
            dns_challenge_noisy_dir="custom/dns",
        )
        assert er.cmd_evaluate(args) == 0
        assert captured["exp_dir"] == canonical_root
        assert captured["datasets"] == "aec_sim"
        assert captured["aec_testset_root"] == "custom/aec"
        assert captured["urgent_noisy_scp"] == "custom/urgent_noisy.scp"
        assert captured["urgent_clean_scp"] == "custom/urgent_clean.scp"
        assert captured["voicebank_clean_scp"] == "custom/clean.scp"
        assert captured["voicebank_noisy_scp"] == "custom/noisy.scp"
        assert captured["dns_challenge_noisy_dir"] == "custom/dns"
        assert captured["inference_devices"] == ["cpu", "cuda:1"]

    def test_cmd_evaluate_accepts_legacy_namespace_without_new_fields(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            er,
            "discover_exp_assets",
            lambda _: {"task_type": "aec", "exp_name": "aec-demo", "exp_dir": "/tmp/aec-demo"},
        )
        monkeypatch.setattr(er, "run_eval_pipeline", lambda **kwargs: captured.update(kwargs) or 0)
        args = argparse.Namespace(
            exp_dir="demo", datasets=None, no_audio_quality=False, device="cpu"
        )

        assert er.cmd_evaluate(args) == 0
        assert captured["aec_no_stream_align"] is False
        assert captured["skip_existing_inference"] is False
        assert captured["aec_testset_root"] is None
        assert captured["urgent_noisy_scp"] is None
        assert captured["urgent_clean_scp"] is None
        assert captured["voicebank_clean_scp"] is None
        assert captured["voicebank_noisy_scp"] is None
        assert captured["dns_challenge_noisy_dir"] is None

    @pytest.mark.parametrize("command", ["evaluate", "run"])
    def test_eval_data_source_parser_defaults_and_overrides(self, monkeypatch, command):
        monkeypatch.setattr(er.sys, "argv", ["experiment_report.py", command, "demo"])
        defaults = er.parse_args()
        assert defaults.aec_testset_root is None
        assert defaults.urgent_noisy_scp is None
        assert defaults.urgent_clean_scp is None
        assert defaults.voicebank_clean_scp is None
        assert defaults.voicebank_noisy_scp is None
        assert defaults.dns_challenge_noisy_dir is None

        monkeypatch.setattr(er.sys, "argv", [
            "experiment_report.py", command, "demo",
            "--aec-testset-root", "override/aec",
            "--urgent-noisy-scp", "override/urgent_noisy.scp",
            "--urgent-clean-scp", "override/urgent_clean.scp",
            "--voicebank-clean-scp", "override/clean.scp",
            "--voicebank-noisy-scp", "override/noisy.scp",
            "--dns-challenge-noisy-dir", "override/dns",
        ])
        overridden = er.parse_args()
        assert overridden.aec_testset_root == "override/aec"
        assert overridden.urgent_noisy_scp == "override/urgent_noisy.scp"
        assert overridden.urgent_clean_scp == "override/urgent_clean.scp"
        assert overridden.voicebank_clean_scp == "override/clean.scp"
        assert overridden.voicebank_noisy_scp == "override/noisy.scp"
        assert overridden.dns_challenge_noisy_dir == "override/dns"

    @pytest.mark.parametrize("task_type", ["aec", "se"])
    def test_eval_command_forwards_supported_data_sources(self, monkeypatch, tmp_path, task_type):
        monkeypatch.chdir(tmp_path)
        cmd = er._build_eval_pipeline_cmd(
            exp_dir="exp/demo",
            task_type=task_type,
            datasets="voicebank,dns_challenge",
            aec_testset_root="override/aec",
            urgent_noisy_scp="override/urgent_noisy.scp",
            urgent_clean_scp="override/urgent_clean.scp",
            voicebank_clean_scp="override/clean.scp",
            voicebank_noisy_scp="override/noisy.scp",
            dns_challenge_noisy_dir="override/dns",
        )
        assert cmd[cmd.index("--voicebank_clean_scp") + 1] == str(
            (tmp_path / "override/clean.scp").resolve()
        )
        assert cmd[cmd.index("--voicebank_noisy_scp") + 1] == str(
            (tmp_path / "override/noisy.scp").resolve()
        )
        assert cmd[cmd.index("--dns_challenge_noisy_dir") + 1] == str(
            (tmp_path / "override/dns").resolve()
        )
        assert ("--aec_testset_root" in cmd) is (task_type == "aec")
        if task_type == "aec":
            assert cmd[cmd.index("--aec_testset_root") + 1] == str(
                (tmp_path / "override/aec").resolve()
            )
            assert cmd[cmd.index("--urgent_noisy_scp") + 1] == str(
                (tmp_path / "override/urgent_noisy.scp").resolve()
            )
            assert cmd[cmd.index("--urgent_clean_scp") + 1] == str(
                (tmp_path / "override/urgent_clean.scp").resolve()
            )
        else:
            assert not any("urgent" in arg for arg in cmd)

    def test_eval_command_uses_repository_defaults_outside_repository(self, monkeypatch, tmp_path):
        monkeypatch.delenv("USE_TRAIN_TEST_DATA_ROOT", raising=False)
        monkeypatch.chdir(tmp_path)

        cmd = er._build_eval_pipeline_cmd(
            exp_dir="exp/demo", task_type="aec", datasets="aec_official"
        )

        expected = {
            "--aec_testset_root": er.DEFAULT_AEC_TESTSET_ROOT,
            "--urgent_noisy_scp": er.DEFAULT_URGENT_NOISY_SCP,
            "--urgent_clean_scp": er.DEFAULT_URGENT_CLEAN_SCP,
            "--voicebank_clean_scp": er.DEFAULT_VOICEBANK_CLEAN_SCP,
            "--voicebank_noisy_scp": er.DEFAULT_VOICEBANK_NOISY_SCP,
            "--dns_challenge_noisy_dir": er.DEFAULT_DNS_CHALLENGE_NOISY_DIR,
        }
        for option, relative_path in expected.items():
            assert cmd[cmd.index(option) + 1] == str(
                (er._PROJECT_ROOT / relative_path).resolve()
            )

    def test_eval_command_defers_unspecified_sources_to_child_environment(self, monkeypatch):
        monkeypatch.setenv("USE_TRAIN_TEST_DATA_ROOT", "/configured/test-data")

        cmd = er._build_eval_pipeline_cmd(
            exp_dir="exp/demo", task_type="aec", datasets="aec_official"
        )

        assert "--aec_testset_root" not in cmd
        assert "--urgent_noisy_scp" not in cmd
        assert "--urgent_clean_scp" not in cmd
        assert "--voicebank_clean_scp" not in cmd
        assert "--voicebank_noisy_scp" not in cmd
        assert "--dns_challenge_noisy_dir" not in cmd

    @pytest.mark.parametrize(
        ("option", "attribute"),
        [
            ("--aec-no-stream-align", "aec_no_stream_align"),
            ("--skip-existing-inference", "skip_existing_inference"),
        ],
    )
    def test_se_cmd_evaluate_rejects_aec_only_options(
        self, monkeypatch, caplog, option, attribute
    ):
        monkeypatch.setattr(
            er,
            "discover_exp_assets",
            lambda _: {"task_type": "se", "exp_name": "se-demo", "exp_dir": "/tmp/se-demo"},
        )
        monkeypatch.setattr(
            er, "run_eval_pipeline", lambda **_kwargs: pytest.fail("不得启动 SE child")
        )
        args = argparse.Namespace(
            exp_dir="demo", datasets=None, no_audio_quality=False, device="cpu"
        )
        setattr(args, attribute, True)

        caplog.set_level("ERROR", logger="experiment_report")
        assert er.cmd_evaluate(args) != 0
        assert option in caplog.text
        assert "仅支持 AEC" in caplog.text

    @pytest.mark.parametrize(
        ("option", "attribute"),
        [
            ("--aec-no-stream-align", "aec_no_stream_align"),
            ("--skip-existing-inference", "skip_existing_inference"),
        ],
    )
    def test_se_cmd_run_rejects_aec_only_options_before_workflow(
        self, monkeypatch, caplog, option, attribute
    ):
        exp = self._exp_data(task_type="se")
        monkeypatch.setattr(er, "discover_exp_assets", lambda _: exp.copy())
        monkeypatch.setattr(
            er, "infer_experiment_purpose", lambda _: pytest.fail("必须在 parent fail-fast")
        )
        args = self._run_args(skip_eval=True)
        setattr(args, attribute, True)

        caplog.set_level("ERROR", logger="experiment_report")
        assert er.cmd_run(args) != 0
        assert option in caplog.text
        assert "仅支持 AEC" in caplog.text

    def test_aec_command_forwards_aec_only_options(self):
        cmd = er._build_eval_pipeline_cmd(
            exp_dir="exp/demo",
            task_type="aec",
            datasets="aec_official",
            aec_no_stream_align=True,
            skip_existing_inference=True,
        )
        assert "--aec_no_stream_align" in cmd
        assert "--skip-existing-inference" in cmd

    def test_cmd_run_uses_discovered_experiment_root(self, monkeypatch):
        captured = {}
        canonical_root = "/tmp/deepvqe_small_10ms_delay1000ms_16k"
        exp = self._exp_data(
            exp_name="deepvqe_small_10ms_delay1000ms_16k", exp_dir=canonical_root,
        )
        monkeypatch.setattr(er, "discover_exp_assets", lambda _: exp.copy())
        monkeypatch.setattr(er, "infer_experiment_purpose", lambda _: "test")
        monkeypatch.setattr(er, "parse_training_metrics", lambda _: {})
        monkeypatch.setattr(er, "run_eval_pipeline", lambda **kwargs: captured.update(kwargs) or 0)
        monkeypatch.setattr(er, "collect_existing_eval_results", lambda _: {})
        monkeypatch.setattr(er, "run_latency_verification", lambda *_args, **_kwargs: {"status": "PASS"})
        monkeypatch.setattr(er, "run_onnx_verification", lambda *_args, **_kwargs: {"status": "PASS"})
        monkeypatch.setattr(er, "_run_badcase_inference", lambda *_args, **_kwargs: {"status": "done"})
        monkeypatch.setattr(er, "update_report_snapshot", lambda *_args: Path("report.md"))
        monkeypatch.setattr(er, "_print_summary", lambda *_args: None)

        assert er.cmd_run(self._run_args(
            exp_dir=f"{canonical_root}/logs/2026-08-11_10-00-00", skip_eval=False,
            aec_testset_root="run/aec",
            urgent_noisy_scp="run/urgent_noisy.scp",
            urgent_clean_scp="run/urgent_clean.scp",
            voicebank_clean_scp="run/clean.scp",
            voicebank_noisy_scp="run/noisy.scp",
            dns_challenge_noisy_dir="run/dns",
        )) == 0
        assert captured["exp_dir"] == canonical_root
        assert captured["aec_testset_root"] == "run/aec"
        assert captured["urgent_noisy_scp"] == "run/urgent_noisy.scp"
        assert captured["urgent_clean_scp"] == "run/urgent_clean.scp"
        assert captured["voicebank_clean_scp"] == "run/clean.scp"
        assert captured["voicebank_noisy_scp"] == "run/noisy.scp"
        assert captured["dns_challenge_noisy_dir"] == "run/dns"

    def test_cmd_run_accepts_legacy_namespace_without_eval_fields(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(er, "discover_exp_assets", lambda _: self._exp_data())
        monkeypatch.setattr(er, "infer_experiment_purpose", lambda _: "test")
        monkeypatch.setattr(er, "parse_training_metrics", lambda _: {})
        monkeypatch.setattr(er, "run_eval_pipeline", lambda **kwargs: captured.update(kwargs) or 0)
        monkeypatch.setattr(er, "collect_existing_eval_results", lambda _: {})
        monkeypatch.setattr(er, "load_experiments_index", lambda: [])
        monkeypatch.setattr(er, "update_report_snapshot", lambda *_args: Path("report.md"))
        monkeypatch.setattr(er, "_print_summary", lambda *_args: None)
        args = self._run_args(
            skip_eval=False,
            skip_latency_verify=True,
            skip_onnx_verify=True,
            skip_badcase=True,
        )
        for attribute in (
            "aec_no_stream_align",
            "skip_existing_inference",
            "aec_testset_root",
            "urgent_noisy_scp",
            "urgent_clean_scp",
            "voicebank_clean_scp",
            "voicebank_noisy_scp",
            "dns_challenge_noisy_dir",
        ):
            delattr(args, attribute)

        assert er.cmd_run(args) == 0
        assert captured["aec_no_stream_align"] is False
        assert captured["skip_existing_inference"] is False
        assert captured["aec_testset_root"] is None
        assert captured["urgent_noisy_scp"] is None
        assert captured["urgent_clean_scp"] is None
        assert captured["voicebank_clean_scp"] is None
        assert captured["voicebank_noisy_scp"] is None
        assert captured["dns_challenge_noisy_dir"] is None

    def test_latency_command_has_stable_machine_readable_contract(self, tmp_path):
        cmd = er._build_latency_verify_cmd(self._exp_data(), "cpu", tmp_path / "latency.json")
        assert cmd == [
            er.sys.executable, "-m", "evaluate.latency_verification", "--task", "aec",
            "--model", "/tmp/model.tar", "--config", "/tmp/config.yaml", "--device", "cpu",
            "--seed", "0", "--json-output", str(tmp_path / "latency.json"),
        ]

    def test_onnx_target_is_sanitized_and_stays_under_project_output(self):
        exp = self._exp_data(exp_name="../../unsafe name")
        cmd = er._build_onnx_verify_cmd("dbgnet", Path("onnx/verify/verify_3way_dbgnet.py"), exp, "cpu")
        target = Path(cmd[cmd.index("--save-onnx") + 1]).resolve()
        assert target.is_relative_to((er._PROJECT_ROOT / "onnx" / "onnx_model").resolve())
        assert target.name == "unsafe_name_DBGNet_streaming.onnx"

    @pytest.mark.parametrize("status_line, reported_target, create_target", [
        ("总结: ALL PASS", "wrong.onnx", False),
        ("总结: FAIL", None, True),
        ("总结: ALL PASS", None, False),
    ])
    def test_onnx_is_not_published_without_verified_expected_artifact(
        self, monkeypatch, tmp_path, status_line, reported_target, create_target,
    ):
        monkeypatch.setattr(er, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(er, "_match_onnx_verify_script", lambda _: ("dbgnet", tmp_path / "verify.py"))

        def fake_run(cmd, **_kwargs):
            target = Path(cmd[cmd.index("--save-onnx") + 1])
            if create_target:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"onnx")
            reported = tmp_path / reported_target if reported_target else target
            return argparse.Namespace(returncode=0, stdout=f"{status_line}\nONNX artifact kept: {reported} (sha256=abc)\n", stderr="")

        monkeypatch.setattr(er.subprocess, "run", fake_run)
        result = er.run_onnx_verification(self._exp_data(), device="cpu")
        assert result["onnx_artifact"]["published"] is False
        if status_line == "总结: ALL PASS":
            assert result["status"] == "FAIL"
            assert result["error"]

    @pytest.mark.parametrize(
        ("artifact_line", "expected_error"),
        [
            ("", "未报告 ONNX artifact 路径"),
            ("ONNX artifact kept: {target} (12 bytes)", "未报告 ONNX artifact SHA256"),
        ],
    )
    def test_all_pass_requires_artifact_path_and_sha_binding(
        self, monkeypatch, tmp_path, artifact_line, expected_error,
    ):
        monkeypatch.setattr(er, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(er, "_match_onnx_verify_script", lambda _: ("dbgnet", tmp_path / "verify.py"))

        def fake_run(cmd, **_kwargs):
            target = Path(cmd[cmd.index("--save-onnx") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"verified onnx")
            stdout = "总结: ALL PASS\n"
            if artifact_line:
                stdout += artifact_line.format(target=target) + "\n"
            return argparse.Namespace(returncode=0, stdout=stdout, stderr="")

        monkeypatch.setattr(er.subprocess, "run", fake_run)
        result = er.run_onnx_verification(self._exp_data(), device="cpu")
        assert result["status"] == "FAIL"
        assert result["onnx_artifact"]["published"] is False
        assert expected_error in result["error"]

    def test_all_pass_requires_reported_artifact_to_exist(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(er, "_match_onnx_verify_script", lambda _: ("dbgnet", tmp_path / "verify.py"))

        def fake_run(cmd, **_kwargs):
            target = Path(cmd[cmd.index("--save-onnx") + 1])
            return argparse.Namespace(
                returncode=0,
                stdout=f"总结: ALL PASS\nONNX artifact kept: {target} (sha256={'0' * 64})\n",
                stderr="",
            )

        monkeypatch.setattr(er.subprocess, "run", fake_run)
        result = er.run_onnx_verification(self._exp_data(), device="cpu")
        assert result["status"] == "FAIL"
        assert result["onnx_artifact"]["published"] is False
        assert "预期 ONNX artifact 不存在或不可读" in result["error"]

    def test_onnx_is_published_with_actual_expected_artifact(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(er, "_match_onnx_verify_script", lambda _: ("dbgnet", tmp_path / "verify.py"))

        def fake_run(cmd, **_kwargs):
            target = Path(cmd[cmd.index("--save-onnx") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"verified onnx")
            digest = hashlib.sha256(b"verified onnx").hexdigest()
            return argparse.Namespace(
                returncode=0,
                stdout=(
                    "use_state_detector=False, state_nclass=4\n"
                    "External iSTFT synthesis capability: PASS "
                    "(finite non-empty waveform only)\n"
                    f"总结: ALL PASS\nONNX artifact kept: {target} (sha256={digest})\n"
                ),
                stderr="",
            )

        monkeypatch.setattr(er.subprocess, "run", fake_run)
        result = er.run_onnx_verification(self._exp_data(), device="cpu")
        artifact = result["onnx_artifact"]
        assert result["status"] == "PASS"
        assert result["release_gates"]["audit_status"] == "PASS"
        assert artifact["published"] is True
        assert artifact["sha256"] == hashlib.sha256(b"verified onnx").hexdigest()
        assert artifact["size_bytes"] == len(b"verified onnx")
        assert Path(artifact["expected_path"]).is_file()

    def test_dbgnet_all_pass_without_external_istft_gate_fails_closed(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(er, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            er, "_match_onnx_verify_script", lambda _: ("dbgnet", tmp_path / "verify.py")
        )

        def fake_run(cmd, **_kwargs):
            target = Path(cmd[cmd.index("--save-onnx") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"verified onnx")
            digest = hashlib.sha256(b"verified onnx").hexdigest()
            return argparse.Namespace(
                returncode=0,
                stdout=(
                    "use_state_detector=False, state_nclass=4\n"
                    f"总结: ALL PASS\nONNX artifact kept: {target} (sha256={digest})\n"
                ),
                stderr="",
            )

        monkeypatch.setattr(er.subprocess, "run", fake_run)
        result = er.run_onnx_verification(self._exp_data(), device="cpu")

        assert result["status"] == "FAIL"
        assert result["onnx_artifact"]["published"] is False
        assert result["release_gates"]["audit_status"] == "UNAUDITABLE"
        assert "External iSTFT synthesis capability" in result["error"]

    def test_onnx_stdout_digest_mismatch_is_not_published(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(er, "_match_onnx_verify_script", lambda _: ("dbgnet", tmp_path / "verify.py"))

        def fake_run(cmd, **_kwargs):
            target = Path(cmd[cmd.index("--save-onnx") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"verified onnx")
            return argparse.Namespace(
                returncode=0,
                stdout=f"总结: ALL PASS\nONNX artifact kept: {target} (sha256={'0' * 64})\n",
                stderr="",
            )

        monkeypatch.setattr(er.subprocess, "run", fake_run)
        result = er.run_onnx_verification(self._exp_data(), device="cpu")
        assert result["status"] == "FAIL"
        assert result["onnx_artifact"]["published"] is False
        assert "sha256" not in result["onnx_artifact"]
        assert "SHA256 与落盘文件不一致" in result["error"]

    def test_latency_result_is_logged_and_validated(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, "_PROJECT_ROOT", tmp_path)

        def fake_run(cmd, **_kwargs):
            json_path = Path(cmd[cmd.index("--json-output") + 1])
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps({
                "metric_kind": "future_dependency", "unit": "ms",
                "min_future_dependency_ms": 0.0, "mean_future_dependency_ms": 1.0,
                "max_future_dependency_ms": 2.0, "values_ms": [0.0, 2.0],
                "task": "aec", "model": "/tmp/model.tar", "config": "/tmp/config.yaml",
            }))
            return argparse.Namespace(returncode=0, stdout="latency stdout", stderr="")

        monkeypatch.setattr(er.subprocess, "run", fake_run)
        result = er.run_latency_verification(self._exp_data(), device="cpu")
        assert result["status"] == "PASS"
        assert result["mean_future_dependency_ms"] == 1.0
        owned_json = Path(result["command"][-1])
        assert result["command"][-2] == "--json-output"
        assert owned_json.name.startswith(".latency_verify_")
        assert not owned_json.exists()
        assert Path(result["log_path"]).read_text() == "latency stdout"
        assert Path(result["json_path"]).name == "latency_verify.json"

    def test_latency_zero_exit_without_current_json_cannot_reuse_stale_result(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, "_PROJECT_ROOT", tmp_path)
        stable = tmp_path / "exp/eval/aec/safe-exp/latency_verify.json"
        stable.parent.mkdir(parents=True, exist_ok=True)
        stable.write_text(json.dumps({"metric_kind": "future_dependency", "unit": "ms"}))
        monkeypatch.setattr(
            er.subprocess, "run",
            lambda *_args, **_kwargs: argparse.Namespace(returncode=0, stdout="", stderr=""),
        )

        result = er.run_latency_verification(self._exp_data(), device="cpu")

        assert result["status"] == "ERROR"
        assert stable.exists()

    def test_latency_rejects_payload_with_wrong_command_identity(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, "_PROJECT_ROOT", tmp_path)

        def fake_run(cmd, **_kwargs):
            output = Path(cmd[cmd.index("--json-output") + 1])
            output.write_text(json.dumps({
                "metric_kind": "future_dependency", "unit": "ms",
                "min_future_dependency_ms": 0.0, "mean_future_dependency_ms": 1.0,
                "max_future_dependency_ms": 2.0,
                "task": "se", "model": "/tmp/model.tar", "config": "/tmp/config.yaml",
            }))
            return argparse.Namespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(er.subprocess, "run", fake_run)
        assert er.run_latency_verification(self._exp_data(), device="cpu")["status"] == "ERROR"

    @pytest.mark.parametrize("payload", [
        None,
        {"metric_kind": "wrong"},
        {"metric_kind": "future_dependency", "unit": "ms", "min_future_dependency_ms": True,
         "mean_future_dependency_ms": 1.0, "max_future_dependency_ms": 2.0},
        {"metric_kind": "future_dependency", "unit": "ms", "min_future_dependency_ms": 0.0,
         "mean_future_dependency_ms": float("nan"), "max_future_dependency_ms": 2.0},
        {"metric_kind": "future_dependency", "unit": "ms", "min_future_dependency_ms": 0.0,
         "mean_future_dependency_ms": float("inf"), "max_future_dependency_ms": 2.0},
        {"metric_kind": "future_dependency", "unit": "s", "min_future_dependency_ms": 0.0,
         "mean_future_dependency_ms": 1.0, "max_future_dependency_ms": 2.0},
        {"metric_kind": "future_dependency", "unit": "ms", "min_future_dependency_ms": 2.0,
         "mean_future_dependency_ms": 1.0, "max_future_dependency_ms": 0.0},
        ["not", "a", "mapping"],
    ])
    def test_latency_missing_or_malformed_json_is_error(self, monkeypatch, tmp_path, payload):
        monkeypatch.setattr(er, "_PROJECT_ROOT", tmp_path)

        def fake_run(cmd, **_kwargs):
            if payload is not None:
                Path(cmd[cmd.index("--json-output") + 1]).write_text(json.dumps(payload))
            return argparse.Namespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(er.subprocess, "run", fake_run)
        assert er.run_latency_verification(self._exp_data(), device="cpu")["status"] == "ERROR"

    def test_index_preserves_latency_when_skip_omits_new_value(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, "INDEX_FILE", tmp_path / "docs/reports/experiments/experiments.json")
        existing = self._exp_data(latency_verify={"status": "PASS"})
        er.update_experiment_index(existing, {})
        er.update_experiment_index(self._exp_data(), {})
        assert er.load_experiments_index()[0]["latency_verify"] == {"status": "PASS"}

    def test_index_records_device_provenance(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, "INDEX_FILE", tmp_path / "docs/reports/experiments/experiments.json")
        exp = self._exp_data(metric_device="cpu", inference_devices=["cuda:0", "cuda:1"])
        er.update_experiment_index(exp, {})
        entry = er.load_experiments_index()[0]
        assert entry["metric_device"] == "cpu"
        assert entry["inference_devices"] == ["cuda:0", "cuda:1"]

    def test_report_storage_is_initialized_when_assets_are_missing(self, monkeypatch, tmp_path):
        root = tmp_path / "docs/reports/experiments"
        monkeypatch.setattr(er, "REPORT_DIR", root)
        monkeypatch.setattr(er, "INDEX_FILE", root / "experiments.json")
        monkeypatch.setattr(er, "REPORT_FILE", root / "experiment_report.md")
        er.update_experiment_index(self._exp_data(), {})
        report_path = er.save_report(er.build_report())
        assert isinstance(json.loads(er.INDEX_FILE.read_text()), list)
        assert report_path.read_text(encoding="utf-8")

    def test_latency_rendering_shows_summary_and_semantic_boundaries(self):
        detail = "\n".join(er._build_experiment_detail(self._exp_data(latency_verify={
            "status": "PASS", "min_future_dependency_ms": -2.0,
            "mean_future_dependency_ms": 1.0, "max_future_dependency_ms": 4.0,
            "threshold": 1e-5, "sample_rate": 16000, "n_perturbations": 5,
        })))
        assert "算法未来依赖" in detail
        assert "最小/均值/最大" in detail
        assert "threshold=1e-05" in detail
        assert "16000 Hz" in detail
        assert "n=5" in detail
        assert "正值表示 look-ahead" in detail
        assert "负值表示输出缓冲或刻意延时" in detail
        assert "不是 wall-clock" in detail
        assert "不是 RTF" in detail
        assert "不是 GCC-PHAT" in detail
        assert "扰动 mic、固定 far-end ref" in detail

    def test_cpu_rtf_rendering_shows_metrics(self):
        detail = "\n".join(er._build_experiment_detail(self._exp_data(cpu_rtf={
            "rtf_median": 0.123456, "rtf_mean": 0.2,
            "median_seconds": 3.7012, "duration_seconds": 30.04,
            "threads": 8, "interop_threads": 1,
        })))
        assert "CPU 推理 RTF" in detail
        assert "| RTF（中位数） | 0.1235 |" in detail
        assert "| RTF（平均） | 0.2000 |" in detail
        assert "| 30 秒输入中位数纯推理耗时 | 3.701 s |" in detail
        assert "| 测试输入时长 | 30.0 s |" in detail
        assert "| CPU 线程 / interop 线程 | 8 / 1 |" in detail
        assert "不是同一个指标" in detail

    def test_cpu_rtf_rendering_falls_back_to_na_for_missing_or_nan(self):
        detail = "\n".join(er._build_experiment_detail(self._exp_data(cpu_rtf={
            "rtf_median": float("nan"),
        })))
        assert "| RTF（中位数） | N/A |" in detail
        assert "| RTF（平均） | N/A |" in detail
        assert "| 30 秒输入中位数纯推理耗时 | N/A |" in detail
        assert "nan" not in detail

    def test_index_preserves_cpu_rtf_when_skip_omits_new_value(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, "INDEX_FILE", tmp_path / "docs/reports/experiments/experiments.json")
        existing = self._exp_data(cpu_rtf={"rtf_median": 0.1})
        er.update_experiment_index(existing, {})
        er.update_experiment_index(self._exp_data(), {})
        assert er.load_experiments_index()[0]["cpu_rtf"] == {"rtf_median": 0.1}

    def _run_args(self, **overrides):
        args = {
            "exp_dir": "demo", "purpose": None, "datasets": None, "skip_eval": True,
            "no_audio_quality": False, "device": "cpu", "aec_no_stream_align": False,
            "skip_existing_inference": False, "skip_latency_verify": False,
            "aec_testset_root": er.DEFAULT_AEC_TESTSET_ROOT,
            "urgent_noisy_scp": er.DEFAULT_URGENT_NOISY_SCP,
            "urgent_clean_scp": er.DEFAULT_URGENT_CLEAN_SCP,
            "voicebank_clean_scp": er.DEFAULT_VOICEBANK_CLEAN_SCP,
            "voicebank_noisy_scp": er.DEFAULT_VOICEBANK_NOISY_SCP,
            "dns_challenge_noisy_dir": er.DEFAULT_DNS_CHALLENGE_NOISY_DIR,
            "skip_onnx_verify": False, "skip_badcase": False,
            "skip_existing_badcase": False, "badcase_with_align": False,
        }
        args.update(overrides)
        return argparse.Namespace(**args)

    def test_cmd_run_skip_eval_with_results_runs_verifiers_and_rewrites_after_badcase(self, monkeypatch):
        exp = self._exp_data()
        calls = []
        monkeypatch.setattr(er, "discover_exp_assets", lambda _: exp.copy())
        monkeypatch.setattr(er, "infer_experiment_purpose", lambda _: "test")
        monkeypatch.setattr(er, "parse_training_metrics", lambda _: {})
        monkeypatch.setattr(er, "collect_existing_eval_results", lambda _: {"aec_official": {"PESQ": 2.0}})
        monkeypatch.setattr(er, "run_latency_verification", lambda *_args, **_kwargs: calls.append("latency") or {"status": "ERROR"})
        monkeypatch.setattr(er, "run_onnx_verification", lambda *_args, **_kwargs: calls.append("onnx") or {"status": "FAIL"})
        monkeypatch.setattr(er, "_run_badcase_inference", lambda *_args, **_kwargs: calls.append("badcase") or {"status": "done", "modes": {"no_align": {}}})
        monkeypatch.setattr(er, "update_report_snapshot", lambda data, _results: calls.append(("snapshot", data.get("badcase"))) or Path("report.md"))
        monkeypatch.setattr(er, "_print_summary", lambda *_: None)
        assert er.cmd_run(self._run_args()) == 0
        assert calls.count("latency") == calls.count("onnx") == calls.count("badcase") == 1
        snapshots = [entry for entry in calls if isinstance(entry, tuple)]
        assert [entry[0] for entry in snapshots] == ["snapshot", "snapshot"]
        assert snapshots[0][1] is None
        assert snapshots[1][1]["status"] == "done"

    def test_cmd_run_logs_public_dynamic_dataset_keys_before_stringify(
        self, monkeypatch, caplog
    ):
        exp = self._exp_data()
        raw_key = "token=logger-private"
        public_key = er._sanitize_public_key(raw_key)
        monkeypatch.setattr(er, "discover_exp_assets", lambda _: exp.copy())
        monkeypatch.setattr(er, "infer_experiment_purpose", lambda _: "test")
        monkeypatch.setattr(er, "parse_training_metrics", lambda _: {})
        monkeypatch.setattr(
            er,
            "collect_existing_eval_results",
            lambda _: {raw_key: {"DNSMOS": 2.0}},
        )
        monkeypatch.setattr(er, "load_experiments_index", lambda: [])
        monkeypatch.setattr(er, "update_report_snapshot", lambda *_args: Path("report.md"))
        monkeypatch.setattr(er, "_print_summary", lambda *_args: None)

        caplog.set_level("INFO", logger="experiment_report")
        assert er.cmd_run(
            self._run_args(
                skip_latency_verify=True,
                skip_onnx_verify=True,
                skip_badcase=True,
            )
        ) == 0

        assert raw_key not in caplog.text
        assert "logger-private" not in caplog.text
        assert public_key in caplog.text

    def test_cmd_run_skip_eval_without_results_skips_verifiers_and_preserves_history(self, monkeypatch, caplog):
        exp = self._exp_data()
        old = {"exp_name": exp["exp_name"], "latency_verify": {"status": "PASS"}, "onnx_verify": {"status": "PASS"}}
        saved = []
        monkeypatch.setattr(er, "discover_exp_assets", lambda _: exp.copy())
        monkeypatch.setattr(er, "infer_experiment_purpose", lambda _: "test")
        monkeypatch.setattr(er, "parse_training_metrics", lambda _: {})
        monkeypatch.setattr(er, "collect_existing_eval_results", lambda _: {})
        monkeypatch.setattr(er, "load_experiments_index", lambda: [old])
        monkeypatch.setattr(er, "run_latency_verification", lambda *_args, **_kwargs: pytest.fail("must skip latency"))
        monkeypatch.setattr(er, "run_onnx_verification", lambda *_args, **_kwargs: pytest.fail("must skip ONNX"))
        monkeypatch.setattr(er, "_run_badcase_inference", lambda *_args, **_kwargs: pytest.fail("must skip badcase"))
        monkeypatch.setattr(er, "update_report_snapshot", lambda data, _results: saved.append(data.copy()) or Path("report.md"))
        monkeypatch.setattr(er, "_print_summary", lambda *_: None)
        assert er.cmd_run(self._run_args()) == 0
        assert saved[0]["latency_verify"] == old["latency_verify"]
        assert saved[0]["onnx_verify"] == old["onnx_verify"]
        assert "算法未来依赖验证、ONNX 验证和 Badcase 推理" in caplog.text

    def test_badcase_output_relpaths_mirror_recursive_categories_and_reject_escape(self):
        samples = [
            ("nested", Path("samples/delay_0_50ms/nested/mic.wav"), Path("far.wav")),
            ("flat", Path("samples/mic.wav"), Path("far.wav")),
        ]

        assert er._build_badcase_output_relpaths(
            samples,
            {"nested": "delay_0_50ms/nested", "flat": None},
        ) == {
            "nested": "delay_0_50ms/nested/nested.wav",
            "flat": "flat.wav",
        }

        with pytest.raises(ValueError, match="输出相对路径不安全"):
            er._build_badcase_output_relpaths(
                [("nested", Path("mic.wav"), Path("far.wav"))],
                {"nested": "../outside"},
            )

    def test_badcase_output_relpath_scp_is_two_column_and_cleaned(self, tmp_path):
        mapping_path = Path(er._create_badcase_output_relpath_scp({
            "nested": "delay_0_50ms/nested/nested.wav",
            "flat": "flat.wav",
        }))
        try:
            assert mapping_path.read_text(encoding="utf-8") == (
                "nested delay_0_50ms/nested/nested.wav\n"
                "flat flat.wav\n"
            )
        finally:
            mapping_path.unlink(missing_ok=True)

    def test_collect_badcase_samples_supports_explicit_reference_naming(self, tmp_path, caplog):
        test_dir = tmp_path / "samples"
        category = test_dir / "delay_0_50ms"
        category.mkdir(parents=True)
        (category / "sample_mic.wav").touch()
        (category / "sample_ref.wav").touch()
        (category / "20260428-002953-753-s30-32k-32bit-1ch-in[ans-mic]_mic.wav").touch()
        (category / "20260428-002953-754-s30-32k-32bit-1ch-in[ans-ref]_ref.wav").touch()
        (category / "orphan_ref.wav").touch()

        caplog.set_level("WARNING", logger="experiment_report")
        samples = er._collect_badcase_samples(test_dir, tmp_path / "refs")

        assert [(uid, mic.name, ref.name) for uid, mic, ref in samples] == [
            (
                "20260428-002953-753-s30-32k-32bit-1ch-in[ans-mic]",
                "20260428-002953-753-s30-32k-32bit-1ch-in[ans-mic]_mic.wav",
                "20260428-002953-754-s30-32k-32bit-1ch-in[ans-ref]_ref.wav",
            ),
            ("sample", "sample_mic.wav", "sample_ref.wav"),
        ]
        assert "未配对的 badcase 参考文件" in caplog.text
        assert "orphan_ref.wav" in caplog.text
        assert "20260428-002953-754-s30-32k-32bit-1ch-in[ans-ref]_ref.wav" not in caplog.text

    def test_badcase_category_preserves_recursive_source_path(self, monkeypatch, tmp_path):
        test_dir = tmp_path / "input"
        monkeypatch.setattr(er, "BADCASE_TEST_DIR", test_dir)
        monkeypatch.setattr(er, "BADCASE_RESULT_DIR", tmp_path / "output")
        test_dir.mkdir()
        nested_mic = test_dir / "delay_0_50ms" / "nested" / "mic.wav"
        nested_mic.parent.mkdir(parents=True)
        samples = [("x", nested_mic, nested_mic.with_name("ref.wav"))]
        monkeypatch.setattr(er, "_collect_badcase_samples", lambda *_: samples)
        monkeypatch.setattr(er, "_create_scp_files", lambda *_: (tmp_path / "mic.scp", tmp_path / "far.scp"))
        calls = []
        monkeypatch.setattr(
            er,
            "_run_badcase_mode",
            lambda *args: calls.append(args[9]) or {"status": "done"},
        )

        er._run_badcase_inference(self._exp_data(), device="cpu")

        assert calls == [{"x": "delay_0_50ms/nested"}]

    def test_badcase_mode_writes_outputs_to_direct_recursive_mirror(
        self, monkeypatch, tmp_path
    ):
        """badcase 通过 output-relpath 直接把 WAV 写入输入镜像目录。"""
        output_dir = tmp_path / "output" / "no_align"
        output_dir.mkdir(parents=True)
        mapped_output = output_dir / "delay_0_50ms" / "nested" / "demo.wav"
        mapped_output.parent.mkdir(parents=True)
        mapped_output.write_bytes(b"audio")
        exp = self._exp_data()
        manifest = {
            "expected_uids": ["demo"],
            "ok_uids": ["demo"],
            "skipped_uids": [],
            "failed_uids": [],
        }
        results = [
            argparse.Namespace(uid="demo", status="ok", output_path=str(mapped_output))
        ]
        captured = {}
        def fake_run(cmd, **_kwargs):
            captured["cmd"] = cmd
            mapping_path = Path(cmd[cmd.index("--output_relpath_scp") + 1])
            captured["mapping"] = mapping_path.read_text(encoding="utf-8")
            return argparse.Namespace(
                returncode=0,
                stdout="demo: AEC done, delay=1.0 ms, device=cpu",
                stderr="",
            )

        monkeypatch.setattr(er.subprocess, "run", fake_run)
        monkeypatch.setattr(
            er,
            "_validate_badcase_review_audio",
            lambda *_args: {
                "channels": 2,
                "channel_layout": ["mic", "enhanced"],
                "sample_rate": 16000,
                "num_frames": 160,
                "ref_all_zero": True,
            },
        )
        monkeypatch.setattr(
            er,
            "load_run_manifest",
            lambda path: captured.update(manifest_path=Path(path))
            or (manifest, results),
        )
        monkeypatch.setattr(
            er, "_validate_badcase_manifest_identity", lambda *args, **kwargs: None
        )

        info = er._run_badcase_mode(
            exp,
            [("demo", Path("mic.wav"), Path("far.wav"))],
            "mic.scp",
            "far.scp",
            "no_align",
            ["--aec_no_stream_align"],
            output_dir,
            "cpu",
            False,
            {"demo": "delay_0_50ms/nested"},
        )

        command = captured["cmd"]
        assert Path(command[command.index("--output_dir") + 1]) == output_dir
        assert captured["mapping"] == "demo delay_0_50ms/nested/demo.wav\n"
        assert captured["manifest_path"] == output_dir / "run_manifest.json"
        assert info["records"][0]["output_path"] == str(mapped_output)
        assert info["records"][0]["output_relpath"] == "delay_0_50ms/nested/demo.wav"

    def test_badcase_default_only_runs_no_align(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, "BADCASE_TEST_DIR", tmp_path / "input")
        monkeypatch.setattr(er, "BADCASE_RESULT_DIR", tmp_path / "output")
        er.BADCASE_TEST_DIR.mkdir()
        monkeypatch.setattr(er, "_collect_badcase_samples", lambda *_: [("x", Path("m"), Path("f"))])
        monkeypatch.setattr(er, "_create_scp_files", lambda *_: (tmp_path / "mic.scp", tmp_path / "far.scp"))
        calls = []
        monkeypatch.setattr(er, "_run_badcase_mode", lambda *args: calls.append((args[4], args[5])) or {"status": "done"})
        er._run_badcase_inference(self._exp_data(), device="cpu")
        assert calls == [("no_align", ["--aec_no_stream_align"])]

    def test_badcase_subprocess_uses_fixed_long_audio_threshold_once(self, monkeypatch, tmp_path):
        captured = {}
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output = output_dir / "demo.wav"
        output.touch()
        exp = self._exp_data()
        manifest = {
            "config": str(Path(exp["config_path"]).resolve()),
            "checkpoint": str(Path(exp["checkpoint_path"]).resolve()),
            "artifact_fingerprint": {"parameters": {
                "aec_stream_align": False,
                "aec_no_stream_align": True,
                "aec_review_multichannel": True,
                "requested_device": "cpu",
                "cpu_long_audio_seconds": 30.0,
            }},
            "expected_uids": ["demo"],
            "ok_uids": ["demo"],
            "skipped_uids": [],
            "failed_uids": [],
        }
        results = [argparse.Namespace(uid="demo", status="ok", output_path=str(output))]

        def fake_run(cmd, **_kwargs):
            captured["cmd"] = cmd
            return argparse.Namespace(returncode=0, stdout="demo: AEC done, delay=1.0 ms, device=cpu", stderr="")

        monkeypatch.setattr(er.subprocess, "run", fake_run)
        monkeypatch.setattr(
            er,
            "_validate_badcase_review_audio",
            lambda *_args: {
                "channels": 2,
                "channel_layout": ["mic", "enhanced"],
                "sample_rate": 16000,
                "num_frames": 160,
                "ref_all_zero": True,
            },
        )
        monkeypatch.setattr(er, "load_run_manifest", lambda _path: (manifest, results))
        mode_info = er._run_badcase_mode(
            exp,
            [("demo", Path("mic.wav"), Path("far.wav"))],
            "mic.scp",
            "far.scp",
            "no_align",
            ["--aec_no_stream_align"],
            output_dir,
            "cpu",
            False,
        )

        cmd = captured["cmd"]
        assert cmd.count(str(er._PROJECT_ROOT / "inference.py")) == 1
        assert cmd.count("--aec-review-multichannel") == 1
        assert cmd[cmd.index("--cpu-long-audio-seconds") + 1] == "30"
        assert mode_info["status"] == "done"
        assert mode_info["n_files"] == 1

    def test_badcase_with_align_adds_explicit_align_mode(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, "BADCASE_TEST_DIR", tmp_path / "input")
        monkeypatch.setattr(er, "BADCASE_RESULT_DIR", tmp_path / "output")
        er.BADCASE_TEST_DIR.mkdir()
        monkeypatch.setattr(er, "_collect_badcase_samples", lambda *_: [("x", Path("m"), Path("f"))])
        monkeypatch.setattr(er, "_create_scp_files", lambda *_: (tmp_path / "mic.scp", tmp_path / "far.scp"))
        calls = []
        monkeypatch.setattr(er, "_run_badcase_mode", lambda *args: calls.append((args[4], args[5])) or {"status": "done"})
        er._run_badcase_inference(self._exp_data(), device="cpu", with_align=True)
        assert calls == [("with_align", ["--aec_stream_align"]), ("no_align", ["--aec_no_stream_align"])]

    def test_badcase_counts_only_manifest_results_and_marks_failed_partial(
        self, monkeypatch, tmp_path
    ):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        current = output_dir / "current.wav"
        current.touch()
        (output_dir / "stale.wav").touch()
        exp = self._exp_data()
        manifest = {
            "config": str(Path(exp["config_path"]).resolve()),
            "checkpoint": str(Path(exp["checkpoint_path"]).resolve()),
            "artifact_fingerprint": {"parameters": {
                "aec_stream_align": False,
                "aec_no_stream_align": True,
                "aec_review_multichannel": True,
                "requested_device": "cpu",
                "cpu_long_audio_seconds": 30.0,
            }},
            "expected_uids": ["current", "failed"],
            "ok_uids": ["current"],
            "skipped_uids": [],
            "failed_uids": ["failed"],
        }
        results = [
            argparse.Namespace(uid="current", status="ok", output_path=str(current)),
            argparse.Namespace(uid="failed", status="failed", output_path=None),
        ]
        monkeypatch.setattr(
            er.subprocess,
            "run",
            lambda *_args, **_kwargs: argparse.Namespace(
                returncode=0,
                stdout="current: AEC done, delay=2.0 ms, device=cpu",
                stderr="",
            ),
        )
        monkeypatch.setattr(
            er,
            "_validate_badcase_review_audio",
            lambda *_args: {
                "channels": 2,
                "channel_layout": ["mic", "enhanced"],
                "sample_rate": 16000,
                "num_frames": 160,
                "ref_all_zero": True,
            },
        )
        monkeypatch.setattr(er, "load_run_manifest", lambda _path: (manifest, results))

        info = er._run_badcase_mode(
            exp,
            [("current", Path("m1"), Path("f1")), ("failed", Path("m2"), Path("f2"))],
            "mic.scp",
            "far.scp",
            "no_align",
            ["--aec_no_stream_align"],
            output_dir,
            "cpu",
            False,
        )

        assert info["status"] == "partial"
        assert info["n_files"] == 1
        assert info["n_failed"] == 1
        assert [record["uid"] for record in info["records"]] == ["current"]
        assert "stale" not in json.dumps(info)

    def test_badcase_failure_path_preserves_existing_wavs(self, monkeypatch, tmp_path):
        """推理失败时必须保留既有审听产物，不得用失败运行抹掉上次成功输出。"""
        output_dir = tmp_path / "output"
        (output_dir / "nested").mkdir(parents=True)
        existing = output_dir / "previous.wav"
        nested = output_dir / "nested" / "previous_nested.wav"
        existing.touch()
        nested.touch()
        monkeypatch.setattr(
            er.subprocess,
            "run",
            lambda *_args, **_kwargs: argparse.Namespace(
                returncode=1, stdout="", stderr="inference failed"
            ),
        )

        info = er._run_badcase_mode(
            self._exp_data(),
            [("demo", Path("m"), Path("f"))],
            "mic.scp",
            "far.scp",
            "no_align",
            ["--aec_no_stream_align"],
            output_dir,
            "cpu",
            False,
        )

        assert info["status"] == "error"
        assert existing.is_file()
        assert nested.is_file()

    def test_badcase_manifest_failure_preserves_existing_wavs(self, monkeypatch, tmp_path):
        """manifest 校验失败时同样不得清理既有产物。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        existing = output_dir / "previous.wav"
        existing.touch()
        monkeypatch.setattr(
            er.subprocess,
            "run",
            lambda *_args, **_kwargs: argparse.Namespace(
                returncode=0, stdout="demo: AEC done", stderr=""
            ),
        )
        monkeypatch.setattr(
            er,
            "load_run_manifest",
            lambda _path: (_ for _ in ()).throw(er.ManifestValidationError("manifest 无效")),
        )

        info = er._run_badcase_mode(
            self._exp_data(),
            [("demo", Path("m"), Path("f"))],
            "mic.scp",
            "far.scp",
            "no_align",
            ["--aec_no_stream_align"],
            output_dir,
            "cpu",
            False,
        )

        assert info["status"] == "error"
        assert existing.is_file()

    def test_badcase_skip_existing_records_skip_without_stale_summary_fields(
        self, monkeypatch, tmp_path
    ):
        """断点续传只保留 canonical 记录字段，旧摘要的 delay/device 等不得泄漏。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output = output_dir / "demo.wav"
        output.touch()
        (output_dir / "delay_summary.json").write_text(json.dumps({
            "records": [{
                "uid": "demo", "delay_ms": 7.5, "device": "cpu",
                "output_path": "/stale/old.wav", "category": "stale",
            }],
            "n_files": 99,
        }))
        exp = self._exp_data()
        manifest = {
            "config": str(Path(exp["config_path"]).resolve()),
            "checkpoint": str(Path(exp["checkpoint_path"]).resolve()),
            "artifact_fingerprint": {"parameters": {
                "aec_stream_align": False,
                "aec_no_stream_align": True,
                "aec_review_multichannel": True,
                "requested_device": "cpu",
                "cpu_long_audio_seconds": 30.0,
            }},
            "expected_uids": ["demo"],
            "ok_uids": [],
            "skipped_uids": ["demo"],
            "failed_uids": [],
        }
        results = [argparse.Namespace(uid="demo", status="skipped", output_path=str(output))]
        monkeypatch.setattr(
            er.subprocess,
            "run",
            lambda *_args, **_kwargs: argparse.Namespace(returncode=0, stdout="", stderr=""),
        )
        monkeypatch.setattr(
            er,
            "_validate_badcase_review_audio",
            lambda *_args: {
                "channels": 2,
                "channel_layout": ["mic", "enhanced"],
                "sample_rate": 16000,
                "num_frames": 160,
                "ref_all_zero": True,
            },
        )
        monkeypatch.setattr(er, "load_run_manifest", lambda _path: (manifest, results))

        info = er._run_badcase_mode(
            exp,
            [("demo", Path("m"), Path("f"))],
            "mic.scp",
            "far.scp",
            "no_align",
            ["--aec_no_stream_align"],
            output_dir,
            "cpu",
            True,
        )

        assert info["records"] == [{
            "uid": "demo", "status": "skipped", "output_relpath": "demo.wav",
            "output_path": str(output), "channels": 2,
            "channel_layout": ["mic", "enhanced"], "sample_rate": 16000,
            "num_frames": 160, "ref_all_zero": True,
        }]
        assert info["n_files"] == 1

    def test_badcase_missing_or_invalid_manifest_is_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            er.subprocess,
            "run",
            lambda *_args, **_kwargs: argparse.Namespace(returncode=0, stdout="", stderr=""),
        )
        monkeypatch.setattr(
            er,
            "load_run_manifest",
            lambda _path: (_ for _ in ()).throw(er.ManifestValidationError("bad manifest")),
        )

        info = er._run_badcase_mode(
            self._exp_data(),
            [("demo", Path("m"), Path("f"))],
            "mic.scp",
            "far.scp",
            "no_align",
            ["--aec_no_stream_align"],
            tmp_path / "output",
            "cpu",
            False,
        )

        assert info["status"] == "error"
        assert info["n_files"] == 0
        assert "bad manifest" in info["error"]

    def test_checkpoint_epoch_prefers_safe_checkpoint_top_level(self, monkeypatch, tmp_path):
        checkpoint = tmp_path / "checkpoint_final.tar"
        checkpoint.touch()
        calls = []

        def fake_safe_load(path, **kwargs):
            calls.append((path, kwargs))
            return {"epoch": 27}

        import core.utils.safe_load as safe_load

        monkeypatch.setattr(safe_load, "safe_torch_load", fake_safe_load)
        epoch, source = er._resolve_checkpoint_epoch({
            "checkpoint_path": str(checkpoint),
            "training_metrics": {"best_val_loss_epoch": 9},
        })

        assert (epoch, source) == (27, "checkpoint.epoch")
        assert calls == [(
            str(checkpoint), {"purpose": "experiment report checkpoint epoch"},
        )]

    def test_checkpoint_epoch_best_model_falls_back_to_training_metric(
        self, monkeypatch, tmp_path
    ):
        checkpoint = tmp_path / "best_model.tar"
        checkpoint.touch()
        import core.utils.safe_load as safe_load

        monkeypatch.setattr(safe_load, "safe_torch_load", lambda *_args, **_kwargs: {})

        assert er._resolve_checkpoint_epoch({
            "checkpoint_path": str(checkpoint),
            "training_metrics": {"best_val_loss_epoch": 13},
        }) == (13, "training_metrics.best_val_loss_epoch")

    @pytest.mark.parametrize(
        ("checkpoint_name", "checkpoint_epoch", "metric_epoch"),
        [
            ("best_model.tar", 0, None),
            ("best_model.tar", 2.5, 0),
            ("checkpoint_final.tar", None, 8),
        ],
    )
    def test_checkpoint_epoch_unknown_when_no_valid_source(
        self, monkeypatch, tmp_path, checkpoint_name, checkpoint_epoch, metric_epoch
    ):
        checkpoint = tmp_path / checkpoint_name
        checkpoint.touch()
        import core.utils.safe_load as safe_load

        monkeypatch.setattr(
            safe_load,
            "safe_torch_load",
            lambda *_args, **_kwargs: {"epoch": checkpoint_epoch},
        )

        assert er._resolve_checkpoint_epoch({
            "checkpoint_path": str(checkpoint),
            "training_metrics": {"best_val_loss_epoch": metric_epoch},
        }) == (None, "epoch_unknown")

    @pytest.mark.parametrize(
        ("epoch", "expected_name"),
        [(7, "safe-exp_epoch0007"), (None, "safe-exp_epoch_unknown")],
    )
    def test_badcase_result_directory_includes_checkpoint_epoch(
        self, monkeypatch, tmp_path, epoch, expected_name
    ):
        monkeypatch.setattr(er, "BADCASE_RESULT_DIR", tmp_path / "outputs")
        monkeypatch.setattr(
            er,
            "_resolve_checkpoint_epoch",
            lambda _exp: (epoch, "checkpoint.epoch" if epoch else "epoch_unknown"),
        )

        result_dir, legacy_dir, actual_epoch, source = er._badcase_result_paths(
            self._exp_data()
        )

        assert result_dir.name == expected_name
        assert legacy_dir.name == "safe-exp"
        assert actual_epoch == epoch
        assert source == ("checkpoint.epoch" if epoch else "epoch_unknown")

    def test_successful_epoch_output_cleans_only_same_experiment_legacy_dir(
        self, monkeypatch, tmp_path
    ):
        input_dir = tmp_path / "samples"
        output_root = tmp_path / "outputs"
        input_dir.mkdir()
        legacy_dir = output_root / "safe-exp"
        other_dir = output_root / "other-exp"
        legacy_dir.mkdir(parents=True)
        other_dir.mkdir()
        (legacy_dir / "legacy.txt").touch()
        (other_dir / "keep.txt").touch()
        sample = input_dir / "nested" / "mic.wav"
        monkeypatch.setattr(er, "BADCASE_TEST_DIR", input_dir)
        monkeypatch.setattr(er, "BADCASE_RESULT_DIR", output_root)
        monkeypatch.setattr(er, "_resolve_checkpoint_epoch", lambda _exp: (4, "checkpoint.epoch"))
        monkeypatch.setattr(
            er,
            "_collect_badcase_samples",
            lambda *_args: [("demo", sample, sample.with_name("ref.wav"))],
        )
        monkeypatch.setattr(
            er, "_create_scp_files", lambda *_args: (tmp_path / "mic.scp", tmp_path / "far.scp")
        )
        monkeypatch.setattr(er, "_run_badcase_mode", lambda *_args: {"status": "done"})

        result = er._run_badcase_inference(self._exp_data(), device="cpu")

        assert result["status"] == "done"
        assert result["checkpoint_epoch"] == 4
        assert result["checkpoint_epoch_source"] == "checkpoint.epoch"
        assert Path(result["result_dir"]).name == "safe-exp_epoch0004"
        assert not legacy_dir.exists()
        assert (other_dir / "keep.txt").exists()

    def test_badcase_review_audio_validates_two_and_three_channel_layouts(self, tmp_path):
        sample_rate = 16000
        mic = er.torch.linspace(-0.5, 0.5, 160)
        enhanced = mic * 0.5
        mic_path = tmp_path / "mic.wav"
        zero_ref_path = tmp_path / "zero_ref.wav"
        nonzero_ref_path = tmp_path / "ref.wav"
        er.torchaudio.save(str(mic_path), mic.unsqueeze(0), sample_rate)
        er.torchaudio.save(str(zero_ref_path), er.torch.zeros(1, 160), sample_rate)
        ref = er.torch.linspace(0.25, -0.25, 160)
        er.torchaudio.save(str(nonzero_ref_path), ref.unsqueeze(0), sample_rate)

        two_channel = tmp_path / "two.wav"
        three_channel = tmp_path / "three.wav"
        er.torchaudio.save(str(two_channel), er.torch.stack([mic, enhanced]), sample_rate)
        er.torchaudio.save(
            str(three_channel), er.torch.stack([mic, ref, enhanced]), sample_rate
        )

        assert er._validate_badcase_review_audio(
            two_channel, mic_path, zero_ref_path, sample_rate
        ) == {
            "channels": 2,
            "channel_layout": ["mic", "enhanced"],
            "sample_rate": sample_rate,
            "num_frames": 160,
            "ref_all_zero": True,
        }
        assert er._validate_badcase_review_audio(
            three_channel, mic_path, nonzero_ref_path, sample_rate
        )["channel_layout"] == ["mic", "ref", "enhanced"]

        invalid = tmp_path / "invalid.wav"
        er.torchaudio.save(str(invalid), enhanced.unsqueeze(0), sample_rate)
        with pytest.raises(er.ManifestValidationError, match="2.*actual=1|expected=2"):
            er._validate_badcase_review_audio(
                invalid, mic_path, zero_ref_path, sample_rate
            )

        wrong_rate = tmp_path / "wrong_rate.wav"
        er.torchaudio.save(
            str(wrong_rate), er.torch.stack([mic, enhanced]), sample_rate // 2
        )
        with pytest.raises(er.ManifestValidationError, match="输出采样率不匹配"):
            er._validate_badcase_review_audio(
                wrong_rate, mic_path, zero_ref_path, sample_rate
            )

        wrong_frames = tmp_path / "wrong_frames.wav"
        er.torchaudio.save(
            str(wrong_frames), er.torch.stack([mic[:-1], enhanced[:-1]]), sample_rate
        )
        with pytest.raises(er.ManifestValidationError, match="输出帧数不匹配"):
            er._validate_badcase_review_audio(
                wrong_frames, mic_path, zero_ref_path, sample_rate
            )

    def test_badcase_manifest_requires_review_multichannel_parameter(self, tmp_path):
        exp = self._exp_data()
        output = tmp_path / "demo.wav"
        output.touch()
        manifest = {
            "config": str(Path(exp["config_path"]).resolve()),
            "checkpoint": str(Path(exp["checkpoint_path"]).resolve()),
            "artifact_fingerprint": {"parameters": {
                "aec_stream_align": False,
                "aec_no_stream_align": True,
                "requested_device": "cpu",
                "cpu_long_audio_seconds": 30.0,
            }},
            "expected_uids": ["demo"],
        }
        results = [argparse.Namespace(uid="demo", status="ok", output_path=str(output))]

        with pytest.raises(er.ManifestValidationError, match="aec_review_multichannel"):
            er._validate_badcase_manifest_identity(
                manifest,
                results,
                exp_data=exp,
                samples=[("demo", Path("mic.wav"), Path("ref.wav"))],
                mode="no_align",
                output_dir=tmp_path,
                device="cpu",
            )

    def test_badcase_report_displays_checkpoint_epoch_and_source(self):
        detail = "\n".join(er._build_experiment_detail(self._exp_data(badcase={
            "status": "done",
            "result_dir": "data/test/badcase/outputs/safe-exp_epoch0012",
            "n_samples": 1,
            "checkpoint_epoch": 12,
            "checkpoint_epoch_source": "checkpoint.epoch",
            "modes": {},
        })))

        assert "Checkpoint Epoch: `0012`" in detail
        assert "来源: `checkpoint.epoch`" in detail


_FORBIDDEN_PREFIXES = ("/tmp/", "/data/", "/data_", "/home/", "/Users/", "/root/")


def _assert_no_machine_absolute_path(rendered: str) -> None:
    for prefix in _FORBIDDEN_PREFIXES:
        assert prefix not in rendered


def _path_rich_exp(project_root: Path) -> dict:
    checkpoint = project_root / "exp/aec/demo/checkpoints/run/best_model.tar"
    return {
        "exp_name": "demo",
        "exp_dir": str(project_root / "exp/aec/demo"),
        "task_type": "aec",
        "model_family": "DBGNet",
        "model_type": "DBGNet",
        "timestamp": "2026-08-11T00:00:00",
        "purpose": "path policy regression",
        "config_path": str(project_root / "exp/aec/demo/checkpoints/run/resolved_config.yaml"),
        "checkpoint_path": str(checkpoint),
        "training_metrics": {},
        "eval_results": {"dns_challenge": {"DNSMOS": 3.0}},
        "latency_verify": {
            "status": "ERROR",
            "log_path": str(project_root / "exp/eval/aec/demo/latency_verify.log"),
            "error": "failed to open /home/alice/private/train.log",
            "command": ["/root/miniconda/bin/python", "--model", str(checkpoint)],
        },
        "onnx_verify": {
            "status": "PASS",
            "checks": [],
            "log_path": str(project_root / "exp/eval/aec/demo/onnx_verify.log"),
            "checkpoint_artifact": {"path": str(checkpoint), "sha256": "ab" * 32},
            "onnx_temp_path": "/tmp/private-build/demo.onnx",
            "onnx_artifact_kept": True,
            "onnx_artifact": {
                "expected_path": str(project_root / "onnx/onnx_model/demo.onnx"),
                "reported_path": "/data/models/private/demo.onnx",
                "published": True,
            },
        },
        "badcase": {
            "status": "done",
            "result_dir": str(project_root / "data/test/badcase/outputs/demo"),
            "n_samples": 1,
            "modes": {
                "no_align": {
                    "output_dir": str(project_root / "data/test/badcase/outputs/demo/no_align"),
                    "n_files": 1,
                    "delay_summary": str(project_root / "data/test/badcase/outputs/demo/no_align/delay_summary.json"),
                    "inference_log": "/Users/alice/private/inference.log",
                }
            },
        },
    }


class TestPrivateRuntimeIndex:
    """ignored runtime index 必须保留可用于 I/O 的原始 locator。"""

    def test_private_runtime_index_preserves_raw_locators(self, monkeypatch, tmp_path):
        project_root = tmp_path / "repo"
        index_file = project_root / "docs/reports/experiments/experiments.json"
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        monkeypatch.setattr(er, "INDEX_FILE", index_file)
        exp = _path_rich_exp(project_root)
        exp["path_object"] = project_root / "private/object.bin"

        er.save_experiments_index([exp])

        saved = json.loads(index_file.read_text(encoding="utf-8"))[0]
        assert saved["config_path"] == exp["config_path"]
        assert saved["checkpoint_path"] == exp["checkpoint_path"]
        assert saved["latency_verify"]["error"] == exp["latency_verify"]["error"]
        assert saved["latency_verify"]["command"] == exp["latency_verify"]["command"]
        assert saved["path_object"] == str(exp["path_object"])
        assert er._report_lock_path().is_file()

    def test_save_report_rebuilds_from_locked_index_not_stale_content(
        self, monkeypatch, tmp_path
    ):
        project_root = tmp_path / "repo"
        root = project_root / "docs/reports/experiments"
        index_file = root / "experiments.json"
        report_file = root / "experiment_report.md"
        fresh = _path_rich_exp(project_root)
        fresh["exp_name"] = "fresh-index-entry"
        root.mkdir(parents=True)
        index_file.write_text(
            json.dumps([fresh], ensure_ascii=False, default=str), encoding="utf-8"
        )
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        monkeypatch.setattr(er, "REPORT_DIR", root)
        monkeypatch.setattr(er, "INDEX_FILE", index_file)
        monkeypatch.setattr(er, "REPORT_FILE", report_file)

        saved = er.save_report("stale content").read_text(encoding="utf-8")

        assert "fresh-index-entry" in saved
        assert "stale content" not in saved

    @pytest.mark.subprocess
    def test_persistent_flock_serializes_processes_without_unlinking_lock(self, tmp_path):
        lock_path = tmp_path / ".persistent.lock"
        context = multiprocessing.get_context("fork")
        events = context.Queue()
        first = context.Process(
            target=_flock_worker, args=(lock_path, "first", events, 0.2)
        )
        second = context.Process(
            target=_flock_worker, args=(lock_path, "second", events, 0.0)
        )
        first.start()
        assert events.get(timeout=5) == ("first", "enter")
        second.start()
        assert events.get(timeout=5) == ("first", "exit")
        assert events.get(timeout=5) == ("second", "enter")
        assert events.get(timeout=5) == ("second", "exit")
        first.join(timeout=5)
        second.join(timeout=5)
        assert first.exitcode == second.exitcode == 0
        assert lock_path.is_file()

    @pytest.mark.subprocess
    def test_concurrent_snapshot_updates_keep_both_index_entries_and_markdown(self, tmp_path):
        root = tmp_path / "docs/reports/experiments"
        index_file = root / "experiments.json"
        report_file = root / "experiment_report.md"
        context = multiprocessing.get_context("fork")
        ready = context.Queue()
        start = context.Event()
        workers = [
            context.Process(
                target=_report_update_worker,
                args=(index_file, report_file, "concurrent-a", "DBGNet", ready, start),
            ),
            context.Process(
                target=_report_update_worker,
                args=(index_file, report_file, "concurrent-b", "DeepVQE", ready, start),
            ),
        ]
        for worker in workers:
            worker.start()
        assert {ready.get(timeout=5), ready.get(timeout=5)} == {
            "concurrent-a", "concurrent-b"
        }
        start.set()
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0

        index = json.loads(index_file.read_text(encoding="utf-8"))
        report = report_file.read_text(encoding="utf-8")
        assert {entry["exp_name"] for entry in index} == {
            "concurrent-a", "concurrent-b"
        }
        assert "concurrent-a" in report
        assert "concurrent-b" in report
        assert (root / er.REPORT_LOCK_FILE.name).is_file()
        assert not list(root.glob(".*.tmp"))

    def test_snapshot_second_replace_failure_rolls_back_index_and_markdown(
        self, monkeypatch, tmp_path
    ):
        import core.utils.file_transaction as transaction

        root = tmp_path / "docs/reports/experiments"
        root.mkdir(parents=True)
        index_file = root / "experiments.json"
        report_file = root / "experiment_report.md"
        index_file.write_text("[]\n", encoding="utf-8")
        report_file.write_text("old report\n", encoding="utf-8")
        monkeypatch.setattr(er, "REPORT_DIR", root)
        monkeypatch.setattr(er, "INDEX_FILE", index_file)
        monkeypatch.setattr(er, "REPORT_FILE", report_file)
        real_replace = transaction.os.replace
        calls = 0

        def fail_second_replace(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("second replace failed")
            return real_replace(*args, **kwargs)

        monkeypatch.setattr(transaction.os, "replace", fail_second_replace)
        exp = {
            "exp_name": "new-exp", "exp_dir": "/tmp/new-exp", "task_type": "aec",
            "checkpoint_path": "/tmp/model.tar", "config_path": "/tmp/config.yaml",
            "timestamp": "2026-09-08T00:00:00",
            "resolved_config": {"model": {"type": "DBGNet", "args": {}}},
            "training_metrics": {},
        }

        with pytest.raises(OSError, match="second replace failed"):
            er.update_report_snapshot(exp, {})

        assert index_file.read_text(encoding="utf-8") == "[]\n"
        assert report_file.read_text(encoding="utf-8") == "old report\n"


class TestPublicPathRendering:
    """只有 public DTO、Markdown 与终端输出执行路径去敏。"""

    def test_rendered_report_uses_repo_relative_and_external_semantic_paths(self, monkeypatch, tmp_path):
        project_root = tmp_path / "repo"
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        monkeypatch.setattr(er, "load_experiments_index", lambda: [_path_rich_exp(project_root)])

        report = er.build_report()

        _assert_no_machine_absolute_path(report)
        assert "exp/aec/demo/checkpoints/run/best_model.tar" in report
        assert "onnx/onnx_model/demo.onnx" in report
        assert "<external>/demo.onnx" in report
        assert "<external>/inference.log" in report

    def test_public_record_conversion_does_not_mutate_runtime_record(self, monkeypatch, tmp_path):
        project_root = tmp_path / "repo"
        runtime = _path_rich_exp(project_root)
        raw_error = runtime["latency_verify"]["error"]
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)

        public = er._to_public_record(runtime)

        assert public is not runtime
        assert public["latency_verify"] is not runtime["latency_verify"]
        assert public["latency_verify"]["error"] == "failed to open <external>/train.log"
        assert runtime["latency_verify"]["error"] == raw_error

    def test_dynamic_dict_keys_are_sanitized_without_collision_or_schema_drift(self):
        runtime = {
            "eval_results": {
                "/home/alice/private/set": {"DNSMOS": 1.0},
                "/root/other/set": {"DNSMOS": 2.0},
                r"C:\Users\alice\scenario": {"DNSMOS": 3.0},
                r"\\server\share\scenario": {"DNSMOS": 4.0},
                "FiLe://server/share/scenario": {"DNSMOS": 5.0},
            },
            "training_metrics": {"total_epochs": 1},
        }

        public = er._to_public_record(runtime)
        keys = list(public["eval_results"])

        assert list(public) == ["eval_results", "training_metrics"]
        assert public["training_metrics"] == {"total_epochs": 1}
        assert len(keys) == 5
        assert len(set(keys)) == 5
        assert all(key.startswith("<external>/") for key in keys)
        serialized = json.dumps(public, ensure_ascii=False)
        for forbidden in ("/home/alice", "/root/other", "Users", "server\\share", "file:"):
            assert forbidden.casefold() not in serialized.casefold()
        assert er._to_public_record(public) == public
        assert runtime["eval_results"]["/home/alice/private/set"]["DNSMOS"] == 1.0

    def test_dynamic_key_sanitization_handles_display_collision_stably(self):
        first = {"/home/alice/a/set": 1, "/root/b/set": 2, "token=private": 3}
        second = dict(reversed(list(first.items())))

        public_first = er._to_public_value(first)
        public_second = er._to_public_value(second)

        assert public_first == public_second
        assert list(public_first.items()) == list(public_second.items())
        assert list(public_first) == sorted(public_first)
        assert len(public_first) == 3
        assert set(public_first.values()) == {1, 2, 3}
        assert all(key.startswith("<external>/") for key in public_first)
        serialized = json.dumps(public_first, ensure_ascii=False)
        assert "alice" not in serialized
        assert "private" not in serialized
        assert er._to_public_value(public_first) == public_first

    def test_dynamic_key_post_sanitize_collision_never_silently_overwrites(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            er,
            "_sanitize_public_key",
            lambda key: "<external>/same" if key in {"first", "second"} else key,
        )

        forward = er._to_public_value({"first": 1, "second": 2})
        reverse = er._to_public_value({"second": 2, "first": 1})

        assert forward == reverse
        assert len(forward) == 2
        assert set(forward.values()) == {1, 2}
        assert all(key.startswith("<external>/same~") for key in forward)

    def test_real_sha256_prefix_collision_never_silently_overwrites(self):
        first_key = "token=private-166653"
        second_key = "token=private-172450"
        first_digest = hashlib.sha256(first_key.encode("utf-8")).hexdigest()
        second_digest = hashlib.sha256(second_key.encode("utf-8")).hexdigest()
        assert first_digest[:8] == second_digest[:8]
        assert first_digest != second_digest

        forward = er._to_public_value({first_key: 1, second_key: 2})
        reverse = er._to_public_value({second_key: 2, first_key: 1})

        assert forward == reverse
        assert len(forward) == 2
        assert set(forward.values()) == {1, 2}
        assert all(key.startswith("<external>/key~") for key in forward)
        assert er._to_public_value(forward) == forward

    def test_final_canonical_ordinal_collision_never_silently_overwrites(self):
        raw = "token=private"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        base = f"<external>/key~{digest}"
        ordinal_one = f"{base}~1"
        ordinal_two = f"{base}~2"
        source = {
            raw: "raw",
            base: "base",
            ordinal_one: "ordinal-one",
            ordinal_two: "ordinal-two",
        }

        forward = er._to_public_value(source)
        reverse = er._to_public_value(dict(reversed(list(source.items()))))

        assert forward == reverse
        assert len(forward) == 4
        assert set(forward.values()) == set(source.values())
        assert len(set(forward)) == len(forward)
        assert er._to_public_value(forward) == forward

    @pytest.mark.parametrize("key", [1, None, 3.5, (1, 2)])
    def test_non_string_dynamic_keys_are_json_safe_and_do_not_crash(self, key):
        public = er._to_public_value({key: "file:secret.txt"})
        serialized = json.dumps(public, ensure_ascii=False)

        assert len(public) == 1
        assert all(isinstance(public_key, str) for public_key in public)
        assert next(iter(public)).startswith("<external>/key~")
        assert "file:" not in serialized.casefold()
        assert "secret.txt" not in serialized
        assert er._to_public_value(public) == public

    @pytest.mark.parametrize(
        "key",
        [float("nan"), float("inf"), float("-inf"), object(), ("nested", object())],
    )
    def test_unstable_non_string_dynamic_keys_are_rejected_explicitly(self, key):
        with pytest.raises(ValueError, match="unsupported dynamic key"):
            er._to_public_value({key: "value"})

    def test_dynamic_key_builtin_subclasses_are_rejected_explicitly(self):
        class StrSubclass(str):
            pass

        class IntSubclass(int):
            pass

        class FloatSubclass(float):
            pass

        class TupleSubclass(tuple):
            pass

        keys = [
            StrSubclass("token=private"),
            IntSubclass(1),
            FloatSubclass(1.5),
            TupleSubclass(("nested", 1)),
        ]
        for key in keys:
            with pytest.raises(ValueError, match="unsupported dynamic key type"):
                er._to_public_value({key: "value"})

    @pytest.mark.parametrize(
        "key",
        [
            "Authorization=Bearer private-token",
            "Authorization: Bearer private-token",
            "auth=private-token",
            "access_key=private-token",
            "private_key=private-token",
            "client_secret=private-token",
            "sig=private-token",
            "bearer=private-token",
        ],
    )
    def test_dynamic_sensitive_assignment_keys_are_fully_redacted(self, key):
        public = er._to_public_record(
            {"eval_results": {key: {"DNSMOS": 1.0}}, "_config_diffs": [(key, {})]}
        )
        serialized = json.dumps(public, ensure_ascii=False)
        assert key not in serialized
        assert "private-token" not in serialized
        assert all(
            display.startswith("<external>/")
            for display in public["eval_results"]
        )
        assert public["_config_diffs"][0][0].startswith("<external>/")
        assert er._to_public_record(public) == public

    @pytest.mark.parametrize(
        "key",
        [
            "/home/alice/private/Authorization=Bearer private-token",
            r"C:\Users\alice\private\client_secret=private-token",
            "safe=ok client_secret=private-token",
            "<external>/Authorization=Bearer private-token",
            "/home/alice/Authorization%3DBearer%20private-token",
            "/home/alice/client_secret%3aprivate-token",
            "/home/alice/Authorization%253DBearer%2520private-token",
            "Authorization%2525253DBearer%25252520private-token",
            "<external>/Authorization%3DBearer%20private-token~12345678",
            "<external>/client_secret%253Dprivate-token~abcdef01",
        ],
    )
    def test_composite_or_forged_dynamic_sensitive_keys_never_preserve_payload(self, key):
        public = er._to_public_record(
            {"eval_results": {key: {"DNSMOS": 1.0}}, "_config_diffs": [(key, {})]}
        )
        serialized = json.dumps(public, ensure_ascii=False)
        display_key = next(iter(public["eval_results"]))

        assert "private-token" not in serialized
        assert "Authorization" not in serialized
        assert "client_secret" not in serialized
        assert display_key.startswith("<external>/key~")
        assert public["_config_diffs"][0][0].startswith("<external>/key~")
        assert er._to_public_record(public) == public

    def test_dynamic_dataset_scenario_and_config_diff_keys_do_not_reach_markdown_or_stdout(
        self, monkeypatch, tmp_path, capsys
    ):
        project_root = tmp_path / "repo"
        exp = _path_rich_exp(project_root)
        exp["eval_results"] = {
            "/home/alice/private/dataset": {
                r"C:\Users\alice\scenario": {"DNSMOS": 3.0}
            }
        }
        exp["_config_diffs"] = [
            ("model.args./home/alice/secret", {"demo": "value"})
        ]
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)

        public = er._to_public_record(exp)
        markdown = "\n".join(er._build_experiment_detail(public))
        lines = er._build_ablation_analysis([public, {**public, "exp_name": "demo_v2"}])
        er._print_summary(public, public["eval_results"])
        stdout = capsys.readouterr().out
        combined = markdown + "\n" + "\n".join(lines) + stdout

        assert "/home/alice" not in combined
        assert "Users" not in combined
        assert "<external>/" in combined

    def test_build_report_reads_runtime_configs_before_public_conversion_from_any_cwd(
        self, monkeypatch, tmp_path
    ):
        project_root = tmp_path / "repo"
        external_root = tmp_path / "external configs"
        other_cwd = tmp_path / "other"
        repo_config = project_root / "configs/demo.yaml"
        external_config = external_root / "demo v2.yaml"
        repo_config.parent.mkdir(parents=True)
        external_config.parent.mkdir(parents=True)
        other_cwd.mkdir()
        repo_config.write_text("model:\n  args:\n    variant: base\n", encoding="utf-8")
        external_config.write_text("model:\n  args:\n    variant: v2\n", encoding="utf-8")
        first = _path_rich_exp(project_root)
        second = _path_rich_exp(project_root)
        first["config_path"] = str(repo_config)
        second.update(exp_name="demo_v2", config_path=str(external_config), timestamp="2026-08-12T00:00:00")
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        monkeypatch.setattr(er, "load_experiments_index", lambda: [first, second])
        monkeypatch.chdir(other_cwd)

        report = er.build_report()

        assert "model.args.variant" in report
        assert "base" in report
        assert '"v2"' in report
        _assert_no_machine_absolute_path(report)
        assert str(external_root) not in report

    def test_save_report_build_report_preserves_html_and_sanitizes_paths(self, monkeypatch, tmp_path):
        project_root = tmp_path / "repo"
        report_dir = project_root / "docs/reports/experiments"
        index_file = report_dir / "experiments.json"
        report_dir.mkdir(parents=True)
        index_file.write_text(
            json.dumps([_path_rich_exp(project_root)], ensure_ascii=False), encoding="utf-8"
        )
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        monkeypatch.setattr(er, "REPORT_DIR", report_dir)
        monkeypatch.setattr(er, "INDEX_FILE", index_file)
        monkeypatch.setattr(er, "REPORT_FILE", report_dir / "experiment_report.md")

        saved = er.save_report("stale content").read_text(encoding="utf-8")

        assert "</details>" in saved
        assert "</summary>" in saved
        assert "</code>" in saved
        assert "<<external>" not in saved
        _assert_no_machine_absolute_path(saved)

    @pytest.mark.parametrize(
        "text",
        [
            "</b>",
            "</code>",
            "</summary>",
            "</details>",
            "https://host.example/a/b",
            "GET /v1/models",
            'GET "/v1/models"',
            "../relative/file",
            "exp/aec/demo/checkpoints/model.tar",
            "PESQ/ESTOI",
            "4/6",
        ],
    )
    def test_free_text_sanitizer_preserves_non_machine_paths(self, text):
        assert er._sanitize_free_text(text) == text

    def test_free_text_sanitizer_preserves_complete_summary_html(self):
        text = "<summary><b>exp1</b> <code>demo</code></summary>"
        assert er._sanitize_free_text(text) == text

    @pytest.mark.parametrize(
        "text",
        [
            "https://host.example/search?next=/v1/models",
            "https://host.example/download?source=/home/alice/model.tar",
            "https://host.example/page#fragment=/root/private/model.tar",
            "see [artifact](https://host.example/download?source=/home/alice/model.tar#part=/v1/a)",
            "see [nested](https://host.example/a_(b)?source=/home/alice/model.tar)",
            "request failed for /v1/models",
            "endpoint: /api/v2/items",
            'endpoint "/v1/models"',
            "route=/v12/items?next=/home/alice/private",
        ],
    )
    def test_reviewer_url_and_api_probe_is_byte_stable_and_idempotent(self, text):
        rendered = er._sanitize_free_text(text)
        assert rendered == text
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        ("text", "expected", "forbidden"),
        [
            (
                "see [local](/home/alice/private/report.md)",
                "see [local](<external>/report.md)",
                "/home/alice",
            ),
            (
                "see [local](</home/alice/My Run/report.md>)",
                "see [local](<external>/path)",
                "My Run",
            ),
            (
                r"see [win](C:\\Users\\alice\\private\\report.md)",
                "see [win](<external>/report.md)",
                "Users",
            ),
            (
                r"see [unc](\\\\server\\share\\private\\report.md)",
                "see [unc](<external>/report.md)",
                "server",
            ),
            (
                "see [file](file:///home/alice/private/report.md)",
                "see [file](<external>/path)",
                "file:///home/alice",
            ),
            (
                "see [escape](../../private/report.md)",
                "see [escape](<external>/report.md)",
                "../..",
            ),
            (
                r"see [escape](..\\..\\private\\report.md)",
                "see [escape](<external>/report.md)",
                r"..\\..",
            ),
        ],
    )
    def test_reviewer_machine_markdown_targets_are_sanitized_and_idempotent(
        self, text, expected, forbidden
    ):
        rendered = er._sanitize_free_text(text)
        assert rendered == expected
        assert forbidden not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "text",
        [
            "see [web](https://host.example/a_(b)?next=/home/alice#frag)",
            "see [api](/api/v2/items?next=/home/alice/private)",
            "see [version](/v12/items#section)",
            "see [fragment](#latency-results)",
            "mail [owner](mailto:owner@example.com)",
            "see [repo](docs/reports/experiments/experiment_report.md)",
            "see [repo](./docs/reports/experiments/experiment_report.md#metrics)",
            "see [spaced](<docs/reports/My Report.md>)",
        ],
    )
    def test_safe_markdown_target_table_is_byte_stable_and_idempotent(self, text):
        rendered = er._sanitize_free_text(text)
        assert rendered == text
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        ("text", "expected", "forbidden"),
        [
            (
                "see [local]( /home/alice/private/report.md)",
                "see [local]( <external>/report.md)",
                "/home/alice",
            ),
            (
                "see [angle]( </home/alice/My Run/report.md> )",
                "see [angle]( <external>/path )",
                "My Run",
            ),
            (
                r"see [win]( C:\Users\alice\private\report.md )",
                "see [win]( <external>/report.md )",
                "Users",
            ),
            (
                r"see [unc]( \\server\share\private\report.md )",
                "see [unc]( <external>/report.md )",
                "server",
            ),
            (
                "see [file]( file:///home/alice/private/report.md )",
                "see [file]( <external>/path )",
                "file:///home/alice",
            ),
        ],
    )
    def test_reviewer_whitespace_markdown_destinations_are_sanitized_and_idempotent(
        self, text, expected, forbidden
    ):
        rendered = er._sanitize_free_text(text)
        assert rendered == expected
        assert forbidden not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        ("text", "expected", "forbidden"),
        [
            (
                '[repo](docs/readme.md "title /home/alice/private/note.txt")',
                '[repo](docs/readme.md "title <external>/note.txt")',
                "/home/alice",
            ),
            (
                r'[mail](mailto:owner@example.com "title C:\Users\alice\note.txt")',
                '[mail](mailto:owner@example.com "title <external>/note.txt")',
                "Users",
            ),
        ],
    )
    def test_safe_destination_does_not_protect_machine_title(
        self, text, expected, forbidden
    ):
        rendered = er._sanitize_free_text(text)
        assert rendered == expected
        assert forbidden not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "text",
        [
            "HTTPS://host.example/a/b?next=/home/alice#frag",
            "see [web](HTTPS://host.example/a/b?next=/home/alice#frag)",
            "mail [owner](MAILTO:owner@example.com)",
        ],
    )
    def test_uppercase_public_uri_schemes_are_byte_stable_and_idempotent(self, text):
        rendered = er._sanitize_free_text(text)
        assert rendered == text
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        ("text", "expected", "forbidden"),
        [
            (
                "FILE:///home/alice/private/report.md",
                "<external>/path",
                "/home/alice",
            ),
            (
                "File:///home/alice/private/report.md",
                "<external>/path",
                "/home/alice",
            ),
            (
                'open "fIlE:///home/alice/My Run/report.md"',
                'open "<external>/path"',
                "My Run",
            ),
        ],
    )
    def test_mixed_case_plain_file_uri_is_sanitized_and_idempotent(
        self, text, expected, forbidden
    ):
        rendered = er._sanitize_free_text(text)
        assert rendered == expected
        assert forbidden not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "uri",
        [
            "file://localhost/home/alice/private/report.md",
            "FiLe://server/share/private/report.md",
            "FILE:/home/alice/private/report.md",
        ],
    )
    def test_all_file_uri_forms_are_sanitized_across_public_boundaries(self, uri):
        plain = er._sanitize_free_text(f"open {uri}")
        quoted = er._sanitize_free_text(f'open "{uri}"')
        markdown = er._sanitize_free_text(f"[file]({uri})")
        argv = er._sanitize_public_argv(
            ["python", "--input", uri, f"--output={uri}", "GET", uri]
        )
        public = er._to_public_record({"purpose": uri, "config_path": uri})

        for rendered in (plain, quoted, markdown, *argv[2:], *public.values()):
            assert "file:" not in rendered.casefold()
            assert "localhost" not in rendered.casefold()
            assert "server/share" not in rendered.casefold()
            assert "alice" not in rendered
        assert markdown == "[file](<external>/path)"
        assert er._sanitize_free_text(plain) == plain
        assert er._sanitize_free_text(quoted) == quoted
        assert er._sanitize_free_text(markdown) == markdown

    @pytest.mark.parametrize(
        "uri",
        [
            "file:relative/private/x",
            "FiLe:secret.txt",
            "file:%2Fhome%2Falice%2Fx",
            "file:C:/Users/alice/private/x",
            "file:relative/x?token=secret#fragment",
            "file%3Arelative/private/x",
            "f%69le%3Asecret.txt",
            "file%253Arelative/private/x",
            "file%252525253Arelative%252525252Fprivate%252525252Fsecret.txt",
        ],
    )
    def test_opaque_file_uri_is_fully_redacted_across_public_boundaries(self, uri):
        outputs = [
            er._sanitize_free_text(uri),
            er._sanitize_free_text(f'open "{uri}"'),
            er._sanitize_free_text(f"[file]({uri})"),
            *er._sanitize_public_argv(["GET", uri, f"--input={uri}"])[1:],
            *er._to_public_record({"purpose": uri, "config_path": uri}).values(),
        ]
        for output in outputs:
            assert "file:" not in output.casefold()
            assert "relative" not in output.casefold()
            assert "secret" not in output.casefold()
            assert "%2fhome" not in output.casefold()
            assert "Users" not in output
        assert outputs[0] == "<external>/path"
        assert outputs[2] == "[file](<external>/path)"
        assert all(er._sanitize_free_text(output) == output for output in outputs)

    @pytest.mark.parametrize(
        "raw",
        [
            "%2Fhome%2Falice%2Fprivate%2Freport.md",
            "%43%3A%5CUsers%5Calice%5Cprivate%5Creport.md",
            "%5C%5Cserver%5Cshare%5Cprivate%5Creport.md",
        ],
    )
    def test_encoded_absolute_paths_are_redacted_across_public_boundaries(self, raw):
        outputs = [
            er._sanitize_free_text(raw),
            er._sanitize_free_text(f'open "{raw}"'),
            er._sanitize_free_text(f"[path]({raw})"),
            *er._sanitize_public_argv([raw, f"--input={raw}"]),
            *er._to_public_record({"purpose": raw, "config_path": raw}).values(),
        ]
        assert all("alice" not in output.casefold() for output in outputs)
        assert all("server" not in output.casefold() for output in outputs)
        assert all("<external>/path" in output for output in outputs)
        assert all(er._sanitize_free_text(output) == output for output in outputs)

    @pytest.mark.parametrize(
        "encoded",
        [
            "%2Fhome%2Falice%2Fprivate%2Freport.md",
            "%43%3A%5CUsers%5Calice%5Cprivate%5Creport.md",
            "%5C%5Cserver%5Cshare%5Cprivate%5Creport.md",
        ],
    )
    @pytest.mark.parametrize(
        "prefix",
        ["prefix=", "path:", "safe;", "x&next=", "?next=", "#next=", "</b>"],
    )
    def test_prefixed_encoded_absolute_paths_are_redacted_across_public_boundaries(
        self, prefix, encoded
    ):
        raw = f"{prefix}{encoded}"
        expected = f"{prefix}<external>/path"
        outputs = [
            er._sanitize_free_text(raw),
            er._sanitize_free_text(f'"{raw}"'),
            er._sanitize_free_text(f"[{raw}](docs/readme.md)"),
            er._sanitize_free_text(f"[path]({raw})"),
            *er._sanitize_public_argv([raw, f"--input={raw}"]),
            er._to_public_record({"purpose": raw})["purpose"],
        ]
        assert outputs == [
            expected,
            f'"{expected}"',
            f"[{expected}](docs/readme.md)",
            "[path](<external>/path)",
            expected,
            f"--input={expected}",
            expected,
        ]
        assert all(er._sanitize_free_text(output) == output for output in outputs)

    @pytest.mark.parametrize(
        "text",
        [
            "<external>/../private/secret.txt",
            "<external>/%2Fhome%2Falice%2Fprivate",
            "<external>/file%253Arelative/private/x",
            "[x](<external>/%5C%5Cserver%5Cshare%5Csecret)",
        ],
    )
    def test_forged_external_placeholder_cannot_bypass_sanitizer(self, text):
        rendered = er._sanitize_free_text(text)
        assert rendered in {"<external>/path", "[x](<external>/path)"}
        assert "private" not in rendered
        assert "server" not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "[/home/alice/private/report.md](docs/readme.md)",
                "[<external>/report.md](docs/readme.md)",
            ),
            (
                r"[C:\Users\alice\private\report.md](https://example.com/docs)",
                "[<external>/report.md](https://example.com/docs)",
            ),
            (
                "[%2Fhome%2Falice%2Fprivate](#local)",
                "[<external>/path](#local)",
            ),
        ],
    )
    def test_markdown_label_is_sanitized_independently_of_destination(
        self, text, expected
    ):
        rendered = er._sanitize_free_text(text)
        assert rendered == expected
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        ("text", "expected", "forbidden"),
        [
            (
                r"[fake-http](HTTP:C:\Users\alice\private\report.md)",
                "[fake-http](<external>/path)",
                "Users",
            ),
            (
                "[fake-https](https:/home/alice/private/report.md)",
                "[fake-https](<external>/path)",
                "/home/alice",
            ),
            (
                "[fake-mail](mailto:/home/alice/private/report.md)",
                "[fake-mail](<external>/path)",
                "/home/alice",
            ),
            (
                r"[fake-mail-win](MAILTO:C:\Users\alice\private\report.md)",
                "[fake-mail-win](<external>/path)",
                "Users",
            ),
        ],
    )
    def test_malformed_public_scheme_machine_payload_is_sanitized_and_idempotent(
        self, text, expected, forbidden
    ):
        rendered = er._sanitize_free_text(text)
        assert rendered == expected
        assert forbidden not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "destination",
        [
            "https:///home/alice/private/report.md",
            "https://C:/Users/alice/private/report.md",
            "https://-bad.example/home/alice/private/report.md",
            "https://host..example/home/alice/private/report.md",
            "https://127.0.0.999/home/alice/private/report.md",
            "https://bad_host.example/home/alice/private/report.md",
        ],
    )
    def test_malformed_http_host_is_not_preprotected_in_markdown(self, destination):
        text = f"[host]({destination})"
        rendered = er._sanitize_free_text(text)
        assert rendered == "[host](<external>/path)"
        assert "alice" not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "text",
        [
            "see [local](</home/alice/private/report.md)",
            "see [local](</home/alice/private/report.md",
            "see [local](/home/alice/private/report.md",
            "see [file](<file%3Arelative/private/x)",
        ],
    )
    def test_malformed_markdown_destination_cannot_bypass_sanitizer(self, text):
        rendered = er._sanitize_free_text(text)
        assert "/home/alice" not in rendered
        assert "relative/private" not in rendered
        assert "file%3a" not in rendered.casefold()
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize("size", [10_000, 20_000])
    @pytest.mark.parametrize(
        "build",
        [
            lambda size: "[" * size + "tail",
            lambda size: "[x](" + "a" * size,
            lambda size: "](" * size + "tail",
            lambda size: "[x](<" + "a" * size,
        ],
    )
    def test_markdown_scanner_has_linear_operation_bound(self, size, build):
        text = build(size)
        _, operations = er._scan_markdown_links(text, lambda target: target)
        assert operations <= len(text) * 3

    @pytest.mark.parametrize(
        "target",
        [
            'docs/readme.md "' + "x" * 20_000,
            "<docs/readme.md " + "x" * 20_000,
            "docs/readme.md " + "'" * 20_000,
        ],
    )
    def test_markdown_target_splitter_has_linear_operation_bound(self, target):
        _, operations = er._split_markdown_target_linear(target)
        assert operations <= len(target) * 3

    @pytest.mark.parametrize(
        "text",
        [
            "%" * 4_000,
            "prefix-" + "a%2F" * 2_000 + "-suffix",
            ("safe%20token " * 2_000) + "%2Fhome%2Falice%2Fprivate",
        ],
    )
    def test_encoded_token_scanner_has_linear_operation_bound(self, text):
        rendered, operations = er._scan_encoded_tokens_linear(
            text,
            lambda raw: "<external>/path" if er._is_absolute_locator(raw) else raw,
        )
        assert operations <= len(text) * 2
        assert len(rendered) <= len(text) + len("<external>/path")

    def test_many_ambiguous_paths_are_sanitized_without_recursion(self):
        segment = (
            "loading /home/alice/My Run/model.tar "
            "[public](https://example.com/docs?ok=1) "
        )
        text = segment * 1_200
        rendered = er._sanitize_free_text(text)
        expected_segment = (
            "loading <external>/path "
            "[public](https://example.com/docs?ok=1) "
        )
        assert rendered == expected_segment * 1_200
        assert "/home/alice" not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    def test_deeply_nested_markdown_titles_are_sanitized_without_recursion(self):
        text = "/home/alice/private.txt"
        for _ in range(200):
            text = f'[label](https://example.com "{text}")'

        rendered = er._sanitize_free_text(text)

        assert rendered == '[label](https://example.com "<external>/path")'
        assert er._sanitize_free_text(rendered) == rendered

    def test_nested_markdown_in_title_remains_well_formed_and_idempotent(self):
        text = '[outer](docs/readme.md "[inner](/home/alice/private.txt)")'

        rendered = er._sanitize_free_text(text)

        assert rendered == '[outer](docs/readme.md "<external>/path")'
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "locator",
        [
            "/home/alice/private.txt",
            r"C:\Users\alice\private.txt",
            "file:///home/alice/private.txt",
        ],
    )
    def test_nested_markdown_destination_never_leaks_locator(self, locator):
        text = f"[outer]([inner]({locator}))"

        rendered = er._sanitize_free_text(text)

        assert rendered == "[outer](<external>/path)"
        assert er._to_public_record({"purpose": text})["purpose"] == rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "destination",
        [
            "https://example.com/x](y)/home/alice/private.txt",
            "mailto:owner@example.com](x)/home/alice/private.txt",
            "/api/v2/items](x)/home/alice/private.txt",
            r"https://example.com/x](y)C:\Users\alice\private.txt",
            "https://example.com/x](y)file:///home/alice/private.txt",
        ],
    )
    def test_public_scheme_cannot_bypass_malformed_destination_guard(
        self, destination
    ):
        text = f"[outer]({destination})"

        rendered = er._sanitize_free_text(text)

        assert rendered == "[outer](<external>/path)"
        assert er._to_public_record({"purpose": text})["purpose"] == rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "destination",
        [
            "https://example.com/?body=%5D%28/home/alice/private.txt",
            "mailto:owner@example.com?body=%255D%2528/home/alice/private.txt",
            "/api/v2/items?body=%5D%28/home/alice/private.txt",
            "https://example.com/?body=%20/home/alice/private.txt",
            "mailto:owner@example.com?body=%0D%0A/home/alice/private.txt",
            "/api/v2/items?body=%5B/home/alice/private.txt",
            "https://example.com/?body=%29/home/alice/private.txt",
            "mailto:owner@example.com?body=%22/home/alice/private.txt",
            "https://example.com/?token=()&body=%29/x",
            "https://u(a)@example.com/?body=%29/x",
            "https://u'x@example.com/?body=%27/x",
        ],
    )
    def test_normalization_cannot_introduce_markdown_destination_boundary(
        self, destination
    ):
        text = f"[outer]({destination})"

        rendered = er._sanitize_free_text(text)

        assert rendered == "[outer](<external>/path)"
        assert er._to_public_record({"purpose": text})["purpose"] == rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "locator",
        [
            "/home/alice/private.txt",
            r"C:\Users\alice\private.txt",
            r"\\server\share\private.txt",
            "file:///home/alice/private.txt",
        ],
    )
    def test_partial_destination_child_fails_closed(self, locator):
        text = f'[outer](docs "[x](https://example.com/a[i](d){locator})")'

        rendered = er._sanitize_free_text(text)

        assert rendered == '[outer](docs "<external>/path")'
        assert locator not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    def test_deep_nested_destinations_use_linear_target_scan_budget(self, monkeypatch):
        nested = "/home/alice/private.txt"
        for _ in range(800):
            nested = f"[inner](a{nested})"
        text = f'[outer](docs/readme.md "{nested}")'
        scanned = 0
        original = er._split_markdown_target_bounds

        def observed(value, start, end):
            nonlocal scanned
            scanned += end - start
            return original(value, start, end)

        monkeypatch.setattr(er, "_split_markdown_target_bounds", observed)
        rendered = er._sanitize_free_text(text)

        assert "/home/alice" not in rendered
        assert scanned <= len(text) * 5

    @pytest.mark.parametrize(
        "body",
        [
            "/home/alice/private",
            "%2Fhome%2Falice%2Fprivate",
            "%252Fhome%252Falice%252Fprivate",
        ],
    )
    def test_api_route_body_placeholder_is_idempotent(self, body):
        text = f"[api](/api/v1/items?body={body}&token=secret)"

        rendered = er._sanitize_free_text(text)

        assert rendered == (
            "[api](/api/v1/items?body=<external>/private&token=REDACTED)"
        )
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "tail",
        [
            r"(y)C:\Users\alice\private",
            r"(y)\\server\share\private",
            "(y)file:///home/alice/private",
            "(y)/home/alice/private",
        ],
    )
    def test_api_route_path_cannot_hide_locator_after_placeholder_prefix(self, tail):
        text = f"[api](/api/<external>/x{tail})"

        rendered = er._sanitize_free_text(text)

        assert rendered != text
        assert tail not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "destination",
        [
            "/api/x?q=<external>/x(y)/home/alice/private",
            "/api/x?<external>/x(y)/home/alice/private=value",
            "/api/x#q=<external>/x(y)/home/alice/private",
            "https://example.com/?q=<external>/x(y)/home/alice/private",
            "mailto:owner@example.com?q=<external>/x(y)/home/alice/private",
        ],
    )
    def test_non_body_uri_fields_cannot_hide_locator_after_placeholder(self, destination):
        text = f"[linked]({destination})"

        rendered = er._sanitize_free_text(text)

        assert rendered == "[linked](<external>/path)"
        assert er._to_public_record({"purpose": text})["purpose"] == rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "path",
        [
            r"/a/C:\Users\alice\private",
            r"/a/\\server\share\private",
            "/a/file:///home/alice/private",
            "/<external>/x(y)/home/alice/private",
            "/a/%43%3A%5CUsers%5Calice%5Cprivate",
        ],
    )
    def test_http_path_cannot_embed_machine_locator(self, path):
        text = f"[web](https://example.com{path})"

        rendered = er._sanitize_free_text(text)

        assert rendered != text
        assert "alice" not in rendered
        assert "server" not in rendered
        assert er._to_public_record({"purpose": text})["purpose"] == rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "route",
        [
            "/api/v2/C:/Users/alice/private",
            r"/api/v2/C:\Users\alice\private",
            "/api/v2/file:///home/alice/private",
            "/api/v2/%43%3A%5CUsers%5Calice%5Cprivate",
            "/v12/x/file%253Arelative/private",
        ],
    )
    def test_plain_api_route_normalizer_failure_is_fail_closed(self, route):
        text = f"request {route}"

        rendered = er._sanitize_free_text(text)

        assert rendered == "request <external>/path"
        assert er._to_public_record({"purpose": text})["purpose"] == rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "request /api/v2/%2Fhome%2Falice/private",
                "request <external>/path",
            ),
            (
                "request /api/v2/%252Fhome%252Falice/private",
                "request <external>/path",
            ),
            (
                "[api](/api/v2/%2Fhome%2Falice/private)",
                "[api](<external>/path)",
            ),
            (
                "[web](https://example.com/api/v2/%2Fhome%2Falice/private)",
                "[web](<external>/path)",
            ),
        ],
    )
    def test_public_uri_encoded_posix_segment_fails_closed(self, text, expected):
        rendered = er._sanitize_free_text(text)

        assert rendered == expected
        assert "%2fhome" not in rendered.casefold()
        assert "alice" not in rendered.casefold()
        assert er._to_public_record({"purpose": text})["purpose"] == rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "[owner](mailto:owner@example.com?token=private&subject=hello)",
                "[owner](mailto:owner@example.com?token=REDACTED&subject=hello)",
            ),
            (
                "[owner](MAILTO:owner@example.com?%2574oken=private&subject=hello)",
                "[owner](MAILTO:owner@example.com?%2574oken=REDACTED&subject=hello)",
            ),
            (
                "[owner](mailto:owner@example.com?body=/home/alice/private)",
                "[owner](mailto:owner@example.com?body=<external>/private)",
            ),
            (
                "[owner](mailto:owner@example.com?body=file%253Arelative/private/x)",
                "[owner](mailto:owner@example.com?body=<external>/path)",
            ),
            (
                "[owner](mailto:owner@example.com?body=%252Fhome%252Falice%252Fprivate)",
                "[owner](mailto:owner@example.com?body=<external>/private)",
            ),
            (
                "[owner](mailto:owner@example.com?body=hello%20world)",
                "[owner](mailto:owner@example.com?body=hello%20world)",
            ),
        ],
    )
    def test_mailto_query_and_body_are_sanitized_and_idempotent(self, text, expected):
        rendered = er._sanitize_free_text(text)
        assert rendered == expected
        assert "private" not in rendered or "<external>/private" in rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "url",
        [
            "https:///home/alice/private/report.md",
            "https://C:/Users/alice/private/report.md",
            "https://-bad.example/home/alice/private/report.md",
            "https://host..example/home/alice/private/report.md",
            "https://bad_host.example/home/alice/private/report.md",
            "https://127.0.0.999/home/alice/private/report.md",
            "https://host.example:99999/home/alice/private/report.md",
            "https://[not-ipv6]/home/alice/private/report.md",
        ],
    )
    def test_plain_malformed_http_uri_is_consumed_and_idempotent(self, url):
        rendered = er._sanitize_free_text(f"failed {url}")
        assert rendered == "failed <external>/path"
        assert "alice" not in rendered
        assert "Users" not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "url",
        [
            "https://user@C:/Users/alice/private/report.md",
            "https://user:pass@C:/Users/alice/private/report.md",
            "HtTpS://user@D:/secret/file.txt",
        ],
    )
    def test_userinfo_wrapped_drive_authority_is_sanitized_everywhere(self, url):
        plain = er._sanitize_free_text(f"failed {url}")
        markdown = er._sanitize_free_text(f"[linked]({url})")
        assert plain == "failed <external>/path"
        assert markdown == "[linked](<external>/path)"
        assert er._sanitize_free_text(plain) == plain
        assert er._sanitize_free_text(markdown) == markdown

    def test_valid_userinfo_http_uri_is_normalized_and_idempotent(self):
        text = "[linked](HtTpS://user:pass@example.com/a_(b)?next=/home/alice#frag)"
        expected = "[linked](HtTpS://example.com/a_(b)?next=/home/alice#frag)"
        assert er._sanitize_free_text(text) == expected
        assert er._sanitize_free_text(expected) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/private",
            "http://api.local/private",
            "http://service.internal/private",
            "http://127.0.0.1/private",
            "http://10.2.3.4/private",
            "http://172.16.1.2/private",
            "http://192.168.1.2/private",
            "http://169.254.1.2/private",
            "http://[::1]/private",
            "http://[fc00::1]/private",
            "http://[fe80::1]/private",
            "http://224.0.0.1/private",
            "http://239.255.255.250/private",
            "http://0.0.0.0/private",
            "http://240.0.0.1/private",
            "http://[ff02::1]/private",
            "http://[::]/private",
        ],
    )
    def test_non_public_http_hosts_are_sanitized_in_plain_and_markdown(self, url):
        plain = er._sanitize_free_text(f"failed {url}")
        markdown = er._sanitize_free_text(f"[internal]({url})")
        assert plain == "failed <external>/path"
        assert markdown == "[internal](<external>/path)"
        assert er._sanitize_free_text(plain) == plain
        assert er._sanitize_free_text(markdown) == markdown

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "[api](/api/v2/items?%2574oken=private&ok=1)",
                "[api](/api/v2/items?%2574oken=REDACTED&ok=1)",
            ),
            (
                "GET /v12/items?client%255fsecret=private#section",
                "GET /v12/items?client%255fsecret=REDACTED#section",
            ),
        ],
    )
    def test_api_route_sensitive_query_is_sanitized_and_idempotent(self, text, expected):
        rendered = er._sanitize_free_text(text)
        assert rendered == expected
        assert "private" not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "[web](https://example.com/api?ok=1%26token%3Dquery-private)",
                "[web](https://example.com/api?REDACTED)",
            ),
            (
                "[api](/api/v2/items?ok=1%3Bapi_key%3Dquery-private)",
                "[api](/api/v2/items?REDACTED)",
            ),
            (
                "[mail](mailto:owner@example.com?subject=hello%2526token%253Dquery-private)",
                "[mail](mailto:owner@example.com?REDACTED)",
            ),
            (
                "[web](https://example.com/api?ok=1&next=x%26token%253Dquery-private)",
                "[web](https://example.com/api?REDACTED)",
            ),
        ],
    )
    def test_encoded_query_delimiters_fail_closed_across_public_uris(
        self, text, expected
    ):
        rendered = er._sanitize_free_text(text)
        assert rendered == expected
        assert "query-private" not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "[web](https://example.com/api#ok=1%26token%3Dfragment-private)",
                "[web](https://example.com/api#REDACTED)",
            ),
            (
                "[api](/api/v2/items#ok=1%3Bsecret%3Dfragment-private)",
                "[api](/api/v2/items#REDACTED)",
            ),
            (
                "[mail](mailto:owner@example.com#token=fragment-private)",
                "[mail](mailto:owner@example.com#token=REDACTED)",
            ),
        ],
    )
    def test_public_uri_fragments_use_bounded_sensitive_classification(
        self, text, expected
    ):
        rendered = er._sanitize_free_text(text)
        assert rendered == expected
        assert "fragment-private" not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        "key",
        ["token", "access_token", "api_key", "apikey", "secret", "password", "passwd", "signature", "credential"],
    )
    def test_sensitive_http_query_values_are_redacted_and_idempotent(self, key):
        text = f"[web](https://example.com/api?ok=1&{key}=private-value#frag)"
        expected = f"[web](https://example.com/api?ok=1&{key}=REDACTED#frag)"
        assert er._sanitize_free_text(text) == expected
        assert "private-value" not in expected
        assert er._sanitize_free_text(expected) == expected

    @pytest.mark.parametrize(
        "key",
        ["%74oken", "%2574oken", "%252574oken", "api%255fkey"],
    )
    def test_encoded_sensitive_http_query_values_are_redacted_with_bounded_decode(self, key):
        text = f"[web](https://example.com/api?ok=1&{key}=private-value#frag)"
        rendered = er._sanitize_free_text(text)
        assert rendered == f"[web](https://example.com/api?ok=1&{key}=REDACTED#frag)"
        assert "private-value" not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    def test_percent_decode_is_bounded_for_long_or_deep_payloads(self):
        deep_key = "%25" * 5000 + "74oken"
        decoded, exhausted = er._bounded_percent_decode(deep_key)
        assert len(decoded) <= er._MAX_PERCENT_DECODE_LENGTH
        assert exhausted is True

    def test_exhausted_percent_encoded_query_key_fails_closed(self):
        key = "%2525252574oken"
        text = f"[web](https://example.com/api?{key}=private-value)"
        rendered = er._sanitize_free_text(text)
        assert rendered == "[web](https://example.com/api?REDACTED)"
        assert "private-value" not in rendered

    def test_dynamic_key_percent_decode_never_exceeds_shared_budget(self, monkeypatch):
        lengths = []
        original = er.unquote_plus

        def observed(raw):
            lengths.append(len(raw))
            return original(raw)

        monkeypatch.setattr(er, "unquote_plus", observed)
        key = "%25" * 5000 + "74oken=private"
        rendered = er._sanitize_public_key(key)
        assert rendered.startswith("<external>/key~")
        assert lengths
        assert max(lengths) <= er._MAX_PERCENT_DECODE_LENGTH

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "file:relative/private/x https://user:pass@example.com/a?token=private, next",
                "<external>/path https://example.com/a?token=REDACTED, next",
            ),
            (
                "file:relative/private/x https://user:pass@example.com/a?token=private.",
                "<external>/path https://example.com/a?token=REDACTED.",
            ),
            (
                "file:relative/private/x https://user:pass@example.com/a?token=private， next",
                "<external>/path https://example.com/a?token=REDACTED， next",
            ),
        ],
    )
    def test_file_uri_before_normalized_url_preserves_punctuation_and_tail(self, text, expected):
        assert er._sanitize_free_text(text) == expected
        assert er._sanitize_free_text(expected) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "see https://user:pass@example.com/a?token=x, next",
                "see https://example.com/a?token=REDACTED, next",
            ),
            (
                "see https://example.com/a?token=x.",
                "see https://example.com/a?token=REDACTED.",
            ),
            (
                "see https://example.com/a?token=x， next",
                "see https://example.com/a?token=REDACTED， next",
            ),
        ],
    )
    def test_normalized_http_url_preserves_trailing_punctuation(self, text, expected):
        assert er._sanitize_free_text(text) == expected
        assert er._sanitize_free_text(expected) == expected

    def test_public_http_url_stays_usable_while_userinfo_and_secrets_are_removed(self):
        text = (
            "see HtTpS://user:pass@docs.example.com:8443/a_(b)"
            "?page=2&API_KEY=abc%2F123&next=/v1/models#section"
        )
        expected = (
            "see HtTpS://docs.example.com:8443/a_(b)"
            "?page=2&API_KEY=REDACTED&next=/v1/models#section"
        )
        assert er._sanitize_free_text(text) == expected
        assert er._sanitize_free_text(expected) == expected

    def test_ambiguous_file_prefix_preserves_complete_public_markdown_fragments(self):
        text = (
            "fIlE:///home/alice/My Run/report.md "
            "[mail](MAILTO:owner@example.com?body=/home/alice/private) "
            "[web](https://host.example/docs?next=/home/alice/private) "
            "[escape]( <../../My Report.md> )"
        )
        expected = (
            "<external>/path "
            "[mail](MAILTO:owner@example.com?body=<external>/private) "
            "[web](https://host.example/docs?next=/home/alice/private) "
            "[escape]( <external>/path )"
        )
        rendered = er._sanitize_free_text(text)
        assert rendered == expected
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "see [escape]( <../../My Report.md> )",
                "see [escape]( <external>/path )",
            ),
            (
                "see [escape](<../../My Report.md>)",
                "see [escape](<external>/path)",
            ),
        ],
    )
    def test_angle_spaced_relative_escape_has_stable_public_rendering(
        self, text, expected
    ):
        rendered = er._sanitize_free_text(text)
        assert rendered == expected
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize(
        ("text", "expected", "forbidden"),
        [
            ('open "/home/alice/My Run/model.tar"', 'open "<external>/model.tar"', "My Run"),
            ("failed /custom/private/model.tar", "failed <external>/model.tar", "/custom/private"),
            (r"failed C:\\Users\\alice\\model.tar", "failed <external>/model.tar", "Users"),
            (r"failed \\\\server\\share\\model.tar", "failed <external>/model.tar", "server"),
            ("open file:///home/alice/private/model.tar", "open <external>/path", "/home/alice"),
            ("loading /home/alice/My Run/model.tar", "loading <external>/path", "Run/model.tar"),
            (r"loading C:\\Users\\alice\\My Run\\model.tar", "loading <external>/path", r"Run\\model.tar"),
            (r"loading \\\\server\\share\\My Run\\model.tar", "loading <external>/path", r"Run\\model.tar"),
            ("loading file:///home/alice/My Run/model.tar", "loading <external>/path", "Run/model.tar"),
        ],
    )
    def test_machine_path_property_table_has_no_parent_or_partial_tail(
        self, text, expected, forbidden
    ):
        rendered = er._sanitize_free_text(text)
        assert rendered == expected
        assert forbidden not in rendered
        assert er._sanitize_free_text(rendered) == rendered

    def test_conservative_summary_restores_protected_url_token_byte_for_byte(self):
        url = "https://host.example/search?next=/v1/models#src=/root/private"
        rendered = er._sanitize_free_text(f"loading /home/alice/My Run/model.tar see {url}")
        assert rendered == f"loading <external>/path {url}"
        assert er._sanitize_free_text(rendered) == rendered

    @pytest.mark.parametrize("forged", ["\ue0000\ue001", "\ue000deadbeef:0\ue001"])
    def test_user_supplied_protected_token_forgery_is_preserved(self, forged):
        url = "https://example.com/docs?token=private"
        markdown = "[repo](docs/readme.md)"
        text = f"{forged} {url} {markdown}"
        expected = f"{forged} https://example.com/docs?token=REDACTED {markdown}"
        assert er._sanitize_free_text(text) == expected
        assert er._sanitize_free_text(expected) == expected

    def test_protected_token_nonce_retries_on_input_collision(self, monkeypatch):
        nonces = iter(["collision", "fresh"])
        monkeypatch.setattr(er.secrets, "token_hex", lambda _size: next(nonces))
        forged = "\ue000collision:0\ue001"
        text = f"{forged} https://example.com/docs?token=private"
        expected = f"{forged} https://example.com/docs?token=REDACTED"
        assert er._sanitize_free_text(text) == expected

    def test_save_report_integration_preserves_urls_routes_html_and_hides_space_path(
        self, monkeypatch, tmp_path
    ):
        project_root = tmp_path / "repo"
        report_dir = project_root / "docs/reports/experiments"
        exp = _path_rich_exp(project_root)
        exp["purpose"] = (
            "FILE:///home/alice/My Run/file.txt, "
            "authority-file FiLe://server/share/private/report.md, "
            "opaque-file [opaque](FiLe:secret.txt), "
            "combined file:relative/private/x "
            "https://user:pass@example.com/a?token=private, next; "
            "[owner](MAILTO:owner@example.com?body=/home/alice/private), "
            "[bad-host](https:///home/alice/private/report.md), "
            "[internal](http://127.0.0.1/private), "
            "[credential](https://user:pass@example.com/api?token=private&ok=1), "
            "plain-invalid https://host..example/home/bob/private/report.md, "
            "plain-drive https://user@C:/Users/alice/private/report.md, "
            "[linked-drive](https://user:pass@D:/secret/file.txt), "
            "[valid-userinfo](HtTpS://user:pass@example.com/docs?next=/home/alice), "
            "see [docs](HTTPS://host.example/search?next=/v1/models#src=/root/private), "
            "[repo](docs/reports/experiments/README.md), "
            "[local](</home/alice/My Run/report.md>), "
            "[whitespace]( /home/alice/private/report.md), "
            "[titled](docs/readme.md \"title /home/alice/private/note.txt\"), "
            "[escape]( <../../My Report.md> ), "
            "[fake](mailto:/home/alice/private/note.txt), "
            "encoded-query https://example.com/api?ok=1%26token%3Dsave-private, "
            "encoded-path %2Fhome%2Falice%2Fprivate%2Fsave.txt, "
            "prefixed= %2Fhome%2Falice%2Fprivate%2Fsave.txt, "
            "field=%2Fhome%2Falice%2Fprivate%2Fsave.txt, "
            "html </b>%5C%5Cserver%5Cshare%5Cprivate%5Csave.txt, "
            "[/home/alice/private/label.md](docs/readme.md), "
            "forged <external>/../private/save.txt and endpoint /api/v2/items"
        )
        exp["latency_verify"]["error"] = "loading /home/alice/My Run/model.tar"
        exp["latency_verify"]["command"].append(
            "--endpoint=https://example.com/api?%26token%3Dcommand-private"
        )
        collision_raw = "token=private"
        collision_digest = hashlib.sha256(collision_raw.encode("utf-8")).hexdigest()
        collision_base = f"<external>/key~{collision_digest}"
        exp["eval_results"] = {
            "/home/alice/private/dataset": {
                r"C:\Users\alice\scenario": {"DNSMOS": 3.0}
            },
            "/home/alice/Authorization%253DBearer%2520private-token": {
                "clean": {"DNSMOS": 2.0}
            },
            "token=private-166653": {"collision-a": {"DNSMOS": 2.1}},
            "token=private-172450": {"collision-b": {"DNSMOS": 2.2}},
            collision_raw: {"terminal-raw": {"DNSMOS": 2.3}},
            collision_base: {"terminal-base": {"DNSMOS": 2.4}},
            f"{collision_base}~1": {"terminal-one": {"DNSMOS": 2.5}},
            f"{collision_base}~2": {"terminal-two": {"DNSMOS": 2.6}},
            None: {"non-string": {"DNSMOS": 2.7}},
        }
        index_file = report_dir / "experiments.json"
        report_dir.mkdir(parents=True)
        index_file.write_text(json.dumps([exp], ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        monkeypatch.setattr(er, "REPORT_DIR", report_dir)
        monkeypatch.setattr(er, "INDEX_FILE", index_file)
        monkeypatch.setattr(er, "REPORT_FILE", report_dir / "experiment_report.md")

        saved = er.save_report("stale content").read_text(encoding="utf-8")
        public = er._to_public_record(exp)

        assert "HTTPS://host.example/search?next=/v1/models#src=/root/private" in saved
        assert "[repo](docs/reports/experiments/README.md)" in saved
        assert "[local](<external>/path)" in saved
        assert "[whitespace]( <external>/report.md)" in saved
        assert '[titled](docs/readme.md "title <external>/note.txt")' in saved
        assert "[escape]( <external>/path )" in saved
        assert "[fake](<external>/path)" in saved
        assert "[bad-host](<external>/path)" in saved
        assert "[linked-drive](<external>/path)" in saved
        assert "[internal](<external>/path)" in saved
        assert "[credential](https://example.com/api?token=REDACTED&ok=1)" in saved
        assert "combined <external>/path https://example.com/a?token=REDACTED, next;" in saved
        assert "[opaque](<external>/path)" in saved
        assert "[owner](MAILTO:owner@example.com?body=<external>/private)" in saved
        assert "encoded-query https://example.com/api?REDACTED" in saved
        assert "encoded-path <external>/path" in saved
        assert "prefixed= <external>/path" in saved
        assert "field=<external>/path" in saved
        assert "html </b><external>/path" in saved
        assert "[<external>/label.md](docs/readme.md)" in saved
        assert "forged <external>/path" in saved
        assert public["latency_verify"]["command"][-1] == (
            "--endpoint=https://example.com/api?REDACTED"
        )
        public_url = "[valid-userinfo](HtTpS://example.com/docs?next=/home/alice)"
        assert public_url in saved
        assert "user:pass@" not in saved
        assert "private-value" not in saved
        assert "private-token" not in saved
        assert "Authorization=Bearer" not in saved
        assert "Authorization%253D" not in saved
        assert "token=private-166653" not in saved
        assert "token=private-172450" not in saved
        assert "collision-a" in saved
        assert "collision-b" in saved
        assert "terminal-raw" in saved
        assert "terminal-base" in saved
        assert "terminal-one" in saved
        assert "terminal-two" in saved
        assert "non-string" in saved
        assert "token=private" not in saved
        assert "FiLe:secret.txt" not in saved
        assert "file:" not in saved.casefold()
        assert "server/share" not in saved.casefold()
        assert "127.0.0.1" not in saved
        assert "host..example" not in saved
        assert "user@C:" not in saved
        assert "user:pass@D:" not in saved
        assert "Users/alice" not in saved
        assert "/home/bob" not in saved
        assert "<external>/path" in saved
        assert "/api/v2/items" in saved
        assert "loading <external>/path" in saved
        assert "/home/alice" not in saved.replace(public_url, "")
        assert "My Run" not in saved
        assert "Run/model.tar" not in saved
        assert "</summary>" in saved
        assert "</details>" in saved
        assert er._sanitize_free_text(saved) == saved

    def test_public_argv_sanitizes_embedded_path_without_touching_api_path(self):
        rendered = er._sanitize_public_argv(
            ["python", "--config=/opt/private/model config.yaml", "GET", "/v1/models"]
        )
        assert rendered == [
            "python",
            "--config=<external>/model config.yaml",
            "GET",
            "/v1/models",
        ]

    def test_public_argv_option_value_uses_same_url_sanitizer_as_plain_arg(self):
        raw = "https://user:pass@example.com/api?token=argv-private&ok=1"
        expected = "https://example.com/api?token=REDACTED&ok=1"
        rendered = er._sanitize_public_argv([raw, f"--endpoint={raw}"])
        assert rendered == [expected, f"--endpoint={expected}"]
        assert "argv-private" not in " ".join(rendered)
        assert er._sanitize_public_argv(rendered) == rendered

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("/home/alice/private/model.tar", "<external>/model.tar"),
            ("%2Fhome%2Falice%2Fprivate%2Fmodel.tar", "<external>/path"),
            ("%43%3A%5CUsers%5Calice%5Cmodel.tar", "<external>/path"),
        ],
    )
    def test_public_argv_option_value_matches_plain_machine_path_policy(
        self, raw, expected
    ):
        rendered = er._sanitize_public_argv([raw, f"--input={raw}"])
        assert rendered == [expected, f"--input={expected}"]
        assert er._sanitize_public_argv(rendered) == rendered

    def test_public_argv_http_method_route_still_normalizes_query(self):
        rendered = er._sanitize_public_argv(
            ["GET", "/api/v1/items?ok=1&token=argv-private#section"]
        )
        assert rendered == [
            "GET",
            "/api/v1/items?ok=1&token=REDACTED#section",
        ]

    def test_discover_and_summary_stdout_sanitize_paths(self, monkeypatch, tmp_path, capsys):
        project_root = tmp_path / "repo"
        exp = _path_rich_exp(project_root)
        exp["timestamp"] = er.datetime.fromisoformat(exp["timestamp"])
        exp["config_path"] = "%2Fhome%2Falice%2Fprivate%2Fresolved_config.yaml"
        exp["purpose"] = (
            "https://example.com/api?ok=1%26token%3Dstdout-private "
            "field=%2Fhome%2Falice%2Fprivate%2Fstdout.txt "
            "</b>%5C%5Cserver%5Cshare%5Cprivate%5Cstdout.txt "
            "[/home/alice/private/label.md](docs/readme.md) "
            "<external>/../private/stdout.txt"
        )
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        monkeypatch.setattr(er, "REPORT_FILE", project_root / "docs/reports/experiments/experiment_report.md")
        monkeypatch.setattr(er, "discover_exp_assets", lambda _: exp)
        malicious_results = {
            "https://example.com/api?ok=1%26token%3Dstdout-private": {
                "DNSMOS": "field=%2Fhome%2Falice%2Fprivate%2Fstdout.txt"
            },
            "[/home/alice/private/label.md](docs/readme.md)": {
                "PESQ": "</b>%5C%5Cserver%5Cshare%5Cprivate%5Cstdout.txt"
            },
            "<external>/../private/stdout.txt": {"DNSMOS": 3.0},
        }
        monkeypatch.setattr(er, "infer_experiment_purpose", lambda _: "test")
        monkeypatch.setattr(er, "parse_training_metrics", lambda _: {})
        monkeypatch.setattr(er, "collect_existing_eval_results", lambda _: malicious_results)

        public_keys = er._public_dynamic_keys(malicious_results)
        assert er.cmd_discover(argparse.Namespace(exp_dir="demo")) == 0
        er._print_summary(exp, malicious_results)

        rendered = capsys.readouterr().out
        _assert_no_machine_absolute_path(rendered)
        assert "<external>/path" in rendered
        assert all(key in rendered for key in public_keys)
        assert "field=<external>/path" in rendered
        assert "</b><external>/path" in rendered
        assert "stdout-private" not in rendered
        assert "%2Fhome" not in rendered
        assert "%5C%5Cserver" not in rendered
        assert "docs/reports/experiments/experiment_report.md" in rendered

    def test_discover_stdout_uses_public_dynamic_dataset_keys_before_stringify(
        self, monkeypatch, capsys
    ):
        exp = _path_rich_exp(Path("/tmp/repo"))
        exp["timestamp"] = er.datetime.fromisoformat(exp["timestamp"])
        raw_keys = ["file%253Arelative/private/x", "%2574oken=stdout-private"]
        public_keys = sorted(er._to_public_value(dict.fromkeys(raw_keys)))
        monkeypatch.setattr(er, "discover_exp_assets", lambda _: exp)
        monkeypatch.setattr(er, "infer_experiment_purpose", lambda _: "test")
        monkeypatch.setattr(er, "parse_training_metrics", lambda _: {})
        monkeypatch.setattr(
            er,
            "collect_existing_eval_results",
            lambda _: {key: {"DNSMOS": 3.0} for key in raw_keys},
        )

        assert er.cmd_discover(argparse.Namespace(exp_dir="demo")) == 0

        rendered = capsys.readouterr().out
        assert "stdout-private" not in rendered
        assert "relative/private" not in rendered
        assert all(key in rendered for key in public_keys)

    def test_logger_filter_sanitizes_encoded_and_markdown_payloads(self, caplog):
        payload = (
            "https://example.com/api?ok=1%26token%3Dlogger-private "
            "%2Fhome%2Falice%2Fprivate%2Flogger.txt "
            "field=%2Fhome%2Falice%2Fprivate%2Flogger.txt "
            "</b>%5C%5Cserver%5Cshare%5Cprivate%5Clogger.txt "
            "[/home/alice/private/label.md](docs/readme.md) "
            "<external>/../private/logger.txt"
        )
        caplog.set_level("INFO", logger="experiment_report")

        er.logger.info(payload)

        rendered = caplog.text
        assert "https://example.com/api?REDACTED" in rendered
        assert "[<external>/label.md](docs/readme.md)" in rendered
        assert "field=<external>/path" in rendered
        assert "</b><external>/path" in rendered
        assert rendered.count("<external>/path") >= 4
        assert "logger-private" not in rendered
        assert "/home/alice" not in rendered
        assert "../private" not in rendered


class TestEvalChildOutput:
    def test_eval_streams_sanitized_terminal_copy_and_keeps_private_raw_log(
        self, monkeypatch, tmp_path, capsys
    ):
        project_root = tmp_path / "repo"
        exp_dir = project_root / "exp/aec/demo"
        raw_line = "loading /home/alice/private/model.tar\n"

        class FakeProcess:
            stdout = iter([raw_line, "GET /v1/models\n"])

            def wait(self, timeout=None):
                return 0

        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        monkeypatch.setattr(er.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

        assert er.run_eval_pipeline(str(exp_dir), "aec", "aec_sim", device="cpu") == 0

        terminal = capsys.readouterr().out
        assert "/home/alice" not in terminal
        assert "<external>/model.tar" in terminal
        assert "GET /v1/models" in terminal
        private_log = project_root / "exp/eval/aec/demo/eval_pipeline.log"
        assert private_log.read_text(encoding="utf-8") == raw_line + "GET /v1/models\n"

    @pytest.mark.unit
    def test_eval_child_receives_bound_config_and_checkpoint(self, monkeypatch, tmp_path):
        project_root = tmp_path / "repo"
        exp_dir = project_root / "exp/aec/demo"
        config = project_root / "run/resolved_config.yaml"
        checkpoint = project_root / "run/best_model.model_only.pt"
        config.parent.mkdir(parents=True)
        config.touch()
        checkpoint.touch()
        captured = {}

        class FakeProcess:
            stdout = iter([])

            def wait(self, timeout=None):
                return 0

        def fake_popen(*args, **kwargs):
            captured.update(kwargs)
            return FakeProcess()

        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        monkeypatch.setattr(er.subprocess, "Popen", fake_popen)

        assert er.run_eval_pipeline(
            str(exp_dir),
            "aec",
            "aec_sim",
            device="cpu",
            config_path=config,
            checkpoint_path=checkpoint,
        ) == 0
        assert captured["env"]["USE_TRAIN_EVAL_CONFIG"] == str(config.resolve())
        assert captured["env"]["USE_TRAIN_EVAL_CHECKPOINT"] == str(checkpoint.resolve())

    @pytest.mark.unit
    def test_parent_reservation_binding_matches_child_adapter(self, monkeypatch, tmp_path):
        project_root = tmp_path / "repo"
        config = project_root / "run/config.yaml"
        checkpoint = project_root / "run/model.model_only.pt"
        config.parent.mkdir(parents=True)
        config.write_text("model:\n  type: DeepVQE\n", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
        identity = er.effective_config_fingerprint(config)
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)

        parent = er._evaluation_reservation_adapter(
            task_type="aec",
            output_name="demo",
            config_path=config,
            checkpoint_path=checkpoint,
            device="cpu",
            inference_devices=None,
            effective_config_identity=identity,
        )
        effective_config = er.load_merged_config(str(config))
        child = PipelineTaskAdapter(
            task="aec",
            pipeline="eval_pipeline_aec",
            step="all",
            output_dir=project_root / "exp/eval/aec/demo",
            provenance={
                "config": config,
                "checkpoint": checkpoint,
                "resolved_config_fingerprint": identity,
                "parameters": {
                    "step": "all",
                    "model_type": er.detect_model_type(
                        effective_config, config_path=config
                    ),
                    "model_subtype": er.detect_aec_subtype(
                        effective_config, config_path=config
                    ),
                    "device": "cpu",
                    "inference_devices": er.normalize_inference_devices(
                        None, fallback="cpu"
                    ),
                    "output_name": "demo",
                },
                "entry": "eval_pipeline_aec",
            },
        )

        assert parent.ownership_binding() == child.ownership_binding()

    @pytest.mark.unit
    def test_parent_reserves_before_launch_and_clears_inherited_claim(
        self, monkeypatch, tmp_path
    ):
        project_root = tmp_path / "repo"
        exp_dir = project_root / "exp/aec/demo"
        config = project_root / "run/config.yaml"
        checkpoint = project_root / "run/model.model_only.pt"
        config.parent.mkdir(parents=True)
        config.write_text("model: {type: DeepVQE}\n", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
        effective_identity = {
            "schema": "effective-merged-config-v1",
            "algorithm": "sha256-json-v1",
            "sha256": "a" * 64,
        }
        captured = {}

        class FakeProcess:
            stdout = iter(())

            def wait(self, timeout=None):
                return 0

        def fake_popen(command, **kwargs):
            output = project_root / "exp/eval/aec/demo"
            reservation_path = output / RESERVATION_FILENAME
            assert reservation_path.is_file()
            token = kwargs["env"][EVALUATION_RESERVATION_TOKEN_ENV]
            reservation_text = reservation_path.read_text(encoding="utf-8")
            assert token not in reservation_text
            assert hashlib.sha256(token.encode("utf-8")).hexdigest() in reservation_text
            assert effective_identity["sha256"] in reservation_text
            captured.update(command=command, **kwargs)
            return FakeProcess()

        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        monkeypatch.setenv(EVALUATION_RESERVATION_TOKEN_ENV, "inherited-claim")
        monkeypatch.setattr(er.subprocess, "Popen", fake_popen)

        assert er.run_eval_pipeline(
            str(exp_dir), "aec", "voicebank", device="cpu",
            config_path=config, checkpoint_path=checkpoint,
            expected_effective_config_identity=effective_identity,
        ) == 0
        expected_option = "--expected-effective-config-sha256"
        assert captured["command"][captured["command"].index(expected_option) + 1] == (
            effective_identity["sha256"]
        )
        assert captured["env"]["USE_TRAIN_EXPECTED_EFFECTIVE_CONFIG_SHA256"] == (
            effective_identity["sha256"]
        )
        assert captured["env"][EVALUATION_RESERVATION_TOKEN_ENV] != "inherited-claim"
        assert not (
            project_root / "exp/eval/aec/demo" / RESERVATION_FILENAME
        ).exists()

    @pytest.mark.unit
    def test_popen_failure_preserves_log_and_releases_unclaimed_reservation(
        self, monkeypatch, tmp_path
    ):
        project_root = tmp_path / "repo"
        output = project_root / "exp/eval/aec/demo"
        output.mkdir(parents=True)
        log_path = output / "eval_pipeline.log"
        log_path.write_bytes(b"previous-log")
        config = project_root / "run/config.yaml"
        checkpoint = project_root / "run/model.model_only.pt"
        config.parent.mkdir(parents=True)
        config.write_text("model: {type: DeepVQE}\n", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        monkeypatch.setattr(
            er.subprocess, "Popen", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("spawn failed"))
        )

        with pytest.raises(OSError, match="spawn failed"):
            er.run_eval_pipeline(
                str(project_root / "exp/aec/demo"), "aec", "voicebank",
                device="cpu", config_path=config, checkpoint_path=checkpoint,
            )

        assert log_path.read_bytes() == b"previous-log"
        assert not (output / RESERVATION_FILENAME).exists()


class TestChildLineagePreflight:
    @pytest.mark.unit
    def test_explicit_preflight_rejects_in_progress_without_truncating_log(
        self, monkeypatch, tmp_path
    ):
        project_root = tmp_path / "repo"
        exp_dir = project_root / "exp/aec/demo"
        assets = project_root / "run"
        assets.mkdir(parents=True)
        config = assets / "config.yaml"
        checkpoint = assets / "model.model_only.pt"
        config.write_text("model: {type: DeepVQE}\n", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
        output_name = "demo@identity"
        output = project_root / "exp/eval/aec" / output_name
        output.mkdir(parents=True)
        log_path = output / "eval_pipeline.log"
        log_path.write_bytes(b"existing private log")
        begin_artifact_run(output, "active")
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        monkeypatch.setattr(
            er.subprocess,
            "Popen",
            lambda *_args, **_kwargs: pytest.fail("preflight failure must not launch child"),
        )

        with pytest.raises(ValueError, match="in-progress"):
            er.run_eval_pipeline(
                str(exp_dir),
                "aec",
                "voicebank",
                device="cpu",
                output_name=output_name,
                config_path=config,
                checkpoint_path=checkpoint,
            )

        assert log_path.read_bytes() == b"existing private log"

    @pytest.mark.unit
    def test_child_environment_clears_inherited_overrides_before_legacy_launch(
        self, monkeypatch, tmp_path
    ):
        project_root = tmp_path / "repo"
        config = project_root / "run/config.yaml"
        checkpoint = project_root / "run/model.model_only.pt"
        config.parent.mkdir(parents=True)
        config.write_text("model: {type: DeepVQE}\n", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
        captured = {}

        class FakeProcess:
            stdout = iter(())

            def wait(self, timeout=None):
                return 0

        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        monkeypatch.setenv("USE_TRAIN_EVAL_CONFIG", "/stale/config.yaml")
        monkeypatch.setenv("USE_TRAIN_EVAL_CHECKPOINT", "/stale/model.tar")
        monkeypatch.setattr(
            er.subprocess,
            "Popen",
            lambda *_args, **kwargs: captured.update(kwargs) or FakeProcess(),
        )

        assert er.run_eval_pipeline(
            str(project_root / "exp/aec/demo"),
            "aec",
            "voicebank",
            device="cpu",
            config_path=config,
            checkpoint_path=checkpoint,
        ) == 0
        assert captured["env"]["USE_TRAIN_EVAL_CONFIG"] == str(config.resolve())
        assert captured["env"]["USE_TRAIN_EVAL_CHECKPOINT"] == str(checkpoint.resolve())


class TestCliErrorBoundary:
    def test_missing_experiment_directory_has_sanitized_error_without_traceback(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            er,
            "parse_args",
            lambda: argparse.Namespace(command="discover", exp_dir="/home/alice/private/missing-exp"),
        )

        assert er.main() == 1

        stderr = capsys.readouterr().err
        assert "FileNotFoundError" in stderr
        assert "<external>/missing-exp" in stderr
        assert "/home/alice" not in stderr
        assert "Traceback" not in stderr



class TestModelOnlyCheckpointExtraction:
    def test_safe_first_and_trusted_only_unsafe_fallback(self, monkeypatch, tmp_path):
        source = tmp_path / "best_model.tar"
        source.touch()
        calls = []

        def fake_load(path, **kwargs):
            calls.append((Path(path), kwargs))
            if not kwargs.get("allow_unsafe"):
                raise RuntimeError("safe rejected")
            return {"model": {"weight": torch.ones(1)}}

        monkeypatch.setattr(extract_ckpt, "safe_torch_load", fake_load)
        with pytest.raises(RuntimeError, match="safe rejected"):
            extract_ckpt._load_training_checkpoint(
                source, trusted_training_checkpoint=False
            )
        assert len(calls) == 1
        assert "allow_unsafe" not in calls[0][1]

        calls.clear()
        loaded = extract_ckpt._load_training_checkpoint(
            source, trusted_training_checkpoint=True
        )
        assert "model" in loaded
        assert len(calls) == 2
        assert "allow_unsafe" not in calls[0][1]
        assert calls[1][1] == {
            "map_location": "cpu",
            "allow_unsafe": True,
            "purpose": (
                "extract model-only weights from trusted self-generated training checkpoint"
            ),
        }

    def test_single_model_strips_training_fields_and_roundtrips_safely(self, tmp_path):
        source = tmp_path / "best_model.tar"
        torch.save(
            {
                "model": {
                    "module.weight": torch.arange(3),
                    "module._extra_state": {"version": 1, "tags": ("safe",)},
                },
                "optimizer": {"state": "discarded"},
                "scheduler": {"step": 4},
                "rng_state": b"discarded",
                "config": {"private": "discarded"},
                "history": [1, 2],
                "epoch": 7,
                "ema": {
                    "decay": 0.99,
                    "num_updates": 5,
                    "shadow_params": [torch.full((3,), 9.0)],
                },
            },
            source,
        )

        output = extract_ckpt.materialize_model_only_checkpoint(
            source, family="single_model"
        )
        loaded = extract_ckpt.validate_model_only_checkpoint(
            output, source_path=source, family="single_model"
        )

        assert set(loaded) == {
            "schema", "family", "source_fingerprint", "model", "ema"
        }
        assert set(loaded["model"]) == {"weight", "_extra_state"}
        assert loaded["model"]["weight"].device.type == "cpu"
        assert loaded["ema"]["shadow_params"][0].device.type == "cpu"
        assert not ({"optimizer", "scheduler", "rng_state", "config", "history", "epoch"} & set(loaded))

    def test_gan_uses_only_canonical_generator_key(self, tmp_path):
        source = tmp_path / "best_model.tar"
        torch.save(
            {
                "model_generator": {"module.weight": torch.ones(2)},
                "model_discriminator": {"weight": torch.zeros(2)},
                "optimizer_generator": {"discard": True},
            },
            source,
        )

        output = extract_ckpt.materialize_model_only_checkpoint(
            source, family="gan_generator"
        )
        loaded = extract_ckpt.validate_model_only_checkpoint(
            output, source_path=source, family="gan_generator"
        )
        assert "model_generator" in loaded
        assert "model" not in loaded
        assert "model_discriminator" not in loaded
        assert set(loaded["model_generator"]) == {"weight"}

    @pytest.mark.parametrize(
        "checkpoint, message",
        [
            ({"model": {"weight": torch.ones(1), "bad": "value"}}, "non_tensor_keys"),
            ({"model": {"weight": torch.ones(1)}, "ema": {"shadow_params": ["bad"]}}, "shadow_params"),
            ({"model": {"weight": torch.ones(1)}, "ema": {"shadow_params": [torch.ones(1)], "bad": object()}}, "不安全值"),
            ({"model": {"weight": torch.ones(1), "_extra_state": object()}}, "不安全值"),
        ],
    )
    def test_rejects_invalid_state_and_ema(self, checkpoint, message):
        with pytest.raises(RuntimeError, match=message):
            extract_ckpt._build_model_only_payload(
                checkpoint,
                family=extract_ckpt.CheckpointFamily.SINGLE_MODEL,
                source_fingerprint={"path": "/tmp/source", "size": 1, "sha256": "ab"},
            )

    def test_validation_rejects_noncanonical_ddp_prefix(self):
        checkpoint = {
            "schema": extract_ckpt.MODEL_ONLY_SCHEMA,
            "family": "single_model",
            "source_fingerprint": {
                "path": "/tmp/source",
                "size": 1,
                "sha256": "ab" * 32,
            },
            "model": {"module.weight": torch.ones(1)},
        }

        with pytest.raises(RuntimeError, match="canonical DDP-free"):
            extract_ckpt.validate_model_only_checkpoint(checkpoint)

    def test_default_output_reuses_then_refreshes_when_source_changes(self, tmp_path):
        source = tmp_path / "best_model.tar"
        torch.save({"model": {"weight": torch.ones(1)}}, source)
        output = extract_ckpt.materialize_model_only_checkpoint(
            source, family="single_model"
        )
        first_fingerprint = extract_ckpt.fingerprint_file(output)
        first_mtime = output.stat().st_mtime_ns

        reused = extract_ckpt.materialize_model_only_checkpoint(
            source, family="single_model"
        )
        assert reused == output
        assert output.stat().st_mtime_ns == first_mtime

        torch.save({"model": {"weight": torch.full((1,), 2.0)}}, source)
        refreshed = extract_ckpt.materialize_model_only_checkpoint(
            source, family="single_model"
        )
        assert refreshed == output
        assert extract_ckpt.fingerprint_file(output) != first_fingerprint
        loaded = extract_ckpt.validate_model_only_checkpoint(
            output, source_path=source, family="single_model"
        )
        torch.testing.assert_close(loaded["model"]["weight"], torch.full((1,), 2.0))

    def test_explicit_output_refuses_overwrite_without_force(self, tmp_path):
        source = tmp_path / "best_model.tar"
        output = tmp_path / "custom.pt"
        torch.save({"model": {"weight": torch.ones(1)}}, source)
        output.write_bytes(b"existing")

        with pytest.raises(FileExistsError, match="拒绝覆盖"):
            extract_ckpt.materialize_model_only_checkpoint(
                source, family="single_model", output=output
            )
        extract_ckpt.materialize_model_only_checkpoint(
            source, family="single_model", output=output, force=True
        )
        extract_ckpt.validate_model_only_checkpoint(
            output, source_path=source, family="single_model"
        )


class TestModelOnlyDiscoveryAndReportFlow:
    def _experiment(self, tmp_path):
        exp = tmp_path / "exp" / "aec" / "demo"
        run = exp / "checkpoints" / "run"
        run.mkdir(parents=True)
        config = run / "resolved_config.yaml"
        config.write_text(
            "model:\n  type: DeepVQE\nstft:\n  n_fft: 320\n",
            encoding="utf-8",
        )
        source = run / "best_model.tar"
        torch.save({"model": {"weight": torch.ones(1)}}, source)
        return exp, config, source

    def test_find_exp_assets_prefers_valid_sibling_and_can_require_source(self, tmp_path):
        exp, config, source = self._experiment(tmp_path)
        model_only = extract_ckpt.materialize_model_only_checkpoint(
            source, family="single_model"
        )

        assert find_exp_assets(exp) == (config, model_only)
        assert find_exp_assets(exp, prefer_model_only=False) == (config, source)

        torch.save({"model": {"weight": torch.full((1,), 3.0)}}, source)
        assert find_exp_assets(exp) == (config, source)

    @pytest.mark.unit
    def test_find_exp_assets_keeps_complete_run_pair_over_root_checkpoint_alias(self, tmp_path):
        exp, config, source = self._experiment(tmp_path)
        model_only = extract_ckpt.materialize_model_only_checkpoint(
            source, family="single_model"
        )
        root_config = exp / "config.yaml"
        root_checkpoint = exp / "checkpoints" / "best_model.tar"
        root_config.write_text("model: {type: DeepVQE}\n", encoding="utf-8")
        torch.save({"model": {"old": torch.zeros(1)}}, root_checkpoint)

        assert find_exp_assets(exp) == (config, model_only)
        # 根目录的 config 不得与 checkpoints/ 根别名跨目录拼配；完整 run pair 优先。
        assert find_exp_assets(exp, prefer_model_only=False) == (config, source)

    @pytest.mark.unit
    def test_find_exp_assets_never_falls_back_to_another_runs_model_only(self, tmp_path):
        exp, _config, source = self._experiment(tmp_path)
        extract_ckpt.materialize_model_only_checkpoint(source, family="single_model")
        selected_run = exp / "checkpoints" / "newer"
        selected_run.mkdir()
        selected_config = selected_run / "resolved_config.yaml"
        selected_config.write_text(
            "model:\n  type: DeepVQE\nstft:\n  n_fft: 320\n",
            encoding="utf-8",
        )
        selected_source = selected_run / "best_model.tar"
        torch.save({"model": {"weight": torch.full((1,), 2.0)}}, selected_source)
        newer_mtime = source.stat().st_mtime_ns + 10_000_000
        os.utime(selected_run, ns=(newer_mtime, newer_mtime))

        assert find_exp_assets(exp) == (selected_config, selected_source)

    @pytest.mark.unit
    def test_report_discovery_ignores_old_model_only_mtime(self, tmp_path, monkeypatch):
        exp, _old_config, old_source = self._experiment(tmp_path)
        newer_run = exp / "checkpoints" / "newer"
        newer_run.mkdir()
        newer_config = newer_run / "resolved_config.yaml"
        newer_config.write_text(
            "model:\n  type: DeepVQE\nstft:\n  n_fft: 320\n",
            encoding="utf-8",
        )
        newer_source = newer_run / "best_model.tar"
        torch.save({"model": {"weight": torch.full((1,), 2.0)}}, newer_source)

        # 旧 run 的派生文件最后写入，不能让其目录 mtime 抢在新的 source pair 之前。
        extract_ckpt.materialize_model_only_checkpoint(
            old_source, family="single_model"
        )
        newer_mtime = old_source.stat().st_mtime_ns + 2_000_000
        os.utime(newer_source, ns=(newer_mtime, newer_mtime))
        monkeypatch.setattr(er, "detect_model_type", lambda _config, **_kwargs: "aec")

        discovered = er.discover_exp_assets(exp)

        assert discovered["config_path"] == str(newer_config.resolve())
        assert discovered["source_checkpoint_path"] == str(newer_source.resolve())
        assert discovered["source_checkpoint_path"] != str(old_source.resolve())

    def test_discover_requires_original_source_and_only_records_existing_derivative(
        self, monkeypatch, tmp_path
    ):
        exp, config, source = self._experiment(tmp_path)
        model_only = extract_ckpt.materialize_model_only_checkpoint(
            source, family="single_model"
        )
        captured = {}

        def wrapped_find(path, *, prefer_model_only=True):
            captured["prefer_model_only"] = prefer_model_only
            return find_exp_assets(path, prefer_model_only=prefer_model_only)

        monkeypatch.setattr(er, "find_exp_assets", wrapped_find)
        monkeypatch.setattr(er, "detect_model_type", lambda _config, **_kwargs: "aec")
        discovered = er.discover_exp_assets(exp)

        assert captured["prefer_model_only"] is False
        assert discovered["source_checkpoint_path"] == str(source)
        assert discovered["checkpoint_path"] == str(model_only)
        assert discovered["model_only_checkpoint_fingerprint"]["sha256"] == hashlib.sha256(model_only.read_bytes()).hexdigest()

    def test_report_does_not_materialize_or_unsafe_load(self, monkeypatch):
        exp = {
            "exp_name": "demo", "exp_dir": "/tmp/demo", "task_type": "aec",
            "config_path": "/tmp/config.yaml", "source_checkpoint_path": "/tmp/best_model.tar",
            "source_checkpoint_fingerprint": {"path": "/tmp/best_model.tar", "size": 1, "sha256": "aa"},
            "expected_model_only_checkpoint_path": "/tmp/best_model.model_only.pt",
            "model_only_checkpoint_path": None, "model_only_checkpoint_fingerprint": None,
            "checkpoint_path": "/tmp/best_model.tar", "checkpoint_family": "single_model",
            "waveform_native": False, "timestamp": "2026-08-31T00:00:00",
            "resolved_config": {"model": {"type": "DeepVQE"}},
        }
        monkeypatch.setattr(er, "discover_exp_assets", lambda _path: exp.copy())
        monkeypatch.setattr(er, "infer_experiment_purpose", lambda _exp: "test")
        monkeypatch.setattr(er, "parse_training_metrics", lambda _exp: {})
        monkeypatch.setattr(er, "collect_existing_eval_results", lambda _exp: {})
        monkeypatch.setattr(er, "update_report_snapshot", lambda *_args: Path("report.md"))
        monkeypatch.setattr(er, "_print_summary", lambda *_args: None)
        monkeypatch.setattr(
            er, "materialize_model_only_checkpoint",
            lambda *_args, **_kwargs: pytest.fail("report 不得生成 checkpoint"),
        )
        args = argparse.Namespace(exp_dir="demo", purpose=None, skip_eval_check=True)
        assert er.cmd_report(args) == 0

    def test_materialized_checkpoint_is_shared_by_latency_onnx_and_badcase(
        self, monkeypatch, tmp_path
    ):
        checkpoint = tmp_path / "best_model.model_only.pt"
        checkpoint.write_bytes(b"model-only")
        exp = {
            "exp_name": "demo", "exp_dir": str(tmp_path), "task_type": "aec",
            "config_path": str(tmp_path / "config.yaml"),
            "checkpoint_path": str(checkpoint),
            "resolved_config": {"model": {"type": "DBGNet", "args": {}}, "data": {"sample_rate": 16000}},
        }
        latency = er._build_latency_verify_cmd(exp, "cpu", tmp_path / "latency.json")
        onnx = er._build_onnx_verify_cmd(
            "dbgnet", Path("onnx/verify/verify_3way_dbgnet.py"), exp, "cpu"
        )
        assert latency[latency.index("--model") + 1] == str(checkpoint)
        assert onnx[onnx.index("--checkpoint") + 1] == str(checkpoint)

        captured = {}
        monkeypatch.setattr(
            er.subprocess, "run",
            lambda cmd, **_kwargs: captured.update(cmd=cmd) or (_ for _ in ()).throw(RuntimeError("stop")),
        )
        info = er._run_badcase_mode(
            exp,
            [("demo", tmp_path / "mic.wav", tmp_path / "far.wav")],
            "mic.scp", "far.scp", "no_align", ["--aec_no_stream_align"],
            tmp_path / "badcase", "cpu", False,
        )
        assert info["status"] == "error"
        assert captured["cmd"][captured["cmd"].index("--checkpoint") + 1] == str(checkpoint)

    def test_evaluate_materializes_before_starting_child_and_forwards_trust(
        self, monkeypatch
    ):
        exp = {
            "exp_name": "demo", "exp_dir": "/tmp/demo", "task_type": "aec",
            "source_checkpoint_path": "/tmp/best_model.tar",
            "checkpoint_family": "single_model", "waveform_native": False,
        }
        events = []

        def fake_materialize(exp_data, *, trusted_training_checkpoint):
            events.append(("materialize", trusted_training_checkpoint))
            exp_data["checkpoint_path"] = "/tmp/best_model.model_only.pt"

        monkeypatch.setattr(er, "discover_exp_assets", lambda _path: exp.copy())
        monkeypatch.setattr(er, "_materialize_consumed_checkpoint", fake_materialize)
        monkeypatch.setattr(
            er, "run_eval_pipeline",
            lambda **_kwargs: events.append(("child", True)) or 0,
        )
        args = argparse.Namespace(
            exp_dir="demo", datasets="aec_official", no_audio_quality=True,
            device="cpu", trusted_training_checkpoint=True,
        )
        assert er.cmd_evaluate(args) == 0
        assert events == [("materialize", True), ("child", True)]

    def test_report_detail_records_source_and_consumed_fingerprints(self):
        detail = "\n".join(er._build_experiment_detail({
            "exp_name": "demo", "task_type": "aec", "model_type": "DBGNet",
            "source_checkpoint_path": "/tmp/best_model.tar",
            "source_checkpoint_fingerprint": {"sha256": "aa" * 32},
            "model_only_checkpoint_path": "/tmp/best_model.model_only.pt",
            "model_only_checkpoint_fingerprint": {"sha256": "bb" * 32},
            "metric_device": "cpu",
            "inference_devices": ["cuda:0", "cuda:1"],
            "eval_results": {},
        }))
        assert "aa" * 32 in detail
        assert "bb" * 32 in detail
        assert "来源训练 Checkpoint" in detail
        assert "消费 Model-only Checkpoint" in detail
        assert "指标/延迟/ONNX/badcase=`cpu`" in detail
        assert "模型推理=`cuda:0,cuda:1`" in detail

    def test_family_and_waveform_native_match_runtime_semantics(self):
        gan = {"model": {"generator": {"type": "FastEnhancer"}}}
        assert extract_ckpt.checkpoint_family_from_config(gan).value == "gan_generator"
        assert er.infer_model_family(gan) == "FastEnhancer"
        assert er._runtime_model_config(gan)["type"] == "FastEnhancer"
        assert er._is_waveform_native_config(
            {"meta": {"allow_no_stft": True}, "model": {"type": "PASE"}}
        )
        assert not er._is_waveform_native_config(
            {"meta": {"allow_no_stft": True}, "stft": {"n_fft": 320}}
        )


class TestExplicitCheckpointReportFlow:
    def _experiment(self, tmp_path):
        project_root = tmp_path / "repo"
        exp = project_root / "exp" / "aec" / "demo"
        run = exp / "checkpoints" / "2026-09-04_19-05-20"
        run.mkdir(parents=True)
        config = run / "resolved_config.yaml"
        config.write_text(
            "model:\n  type: DeepVQE\nstft:\n  n_fft: 320\n",
            encoding="utf-8",
        )
        source = run / "model_0060.pth"
        torch.save({"model": {"weight": torch.ones(1)}, "epoch": 60}, source)
        return project_root, exp, run, config, source

    def _discover_materialized(self, monkeypatch, tmp_path):
        project_root, exp, run, config, source = self._experiment(tmp_path)
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        data = er.discover_exp_assets(exp, checkpoint=source)
        er._materialize_consumed_checkpoint(
            data, trusted_training_checkpoint=False
        )
        return project_root, exp, run, config, source, data

    def _publish_committed_results(
        self,
        exp_data,
        *,
        task="aec",
        output_name=None,
        provenance_config=None,
        provenance_checkpoint=None,
    ):
        output = er._eval_root_for(exp_data)
        results_file = output / "voicebank" / "RESULTS.txt"
        adapter = PipelineTaskAdapter(
            task=task,
            pipeline=f"eval_pipeline_{task}",
            step="all",
            output_dir=output,
            provenance={
                "config": provenance_config or exp_data["config_path"],
                "checkpoint": provenance_checkpoint or exp_data["checkpoint_path"],
                "resolved_config_fingerprint": exp_data["resolved_config_fingerprint"],
                "parameters": {
                    "device": "cpu",
                    "inference_devices": ["cpu"],
                    "output_name": output_name or er._artifact_identity(exp_data),
                },
                "entry": f"eval_pipeline_{task}",
            },
        )

        def on_complete(summary):
            results_file.parent.mkdir(parents=True, exist_ok=True)
            results_file.write_text("PESQ: 2.5\n", encoding="utf-8")
            publish_result_manifest(
                output, summary, artifact_paths=[results_file]
            )
            (output / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

        run_pipeline(
            adapter,
            prepare=lambda: None,
            run=lambda _context: {
                "status": "done",
                "coverage": {"expected": 1, "produced": 1, "failed": 0},
            },
            on_complete=on_complete,
        )
        return output, results_file

    def test_explicit_checkpoint_binds_same_run_and_uses_source_specific_model_only(
        self, monkeypatch, tmp_path
    ):
        project_root, exp, run, config, source = self._experiment(tmp_path)
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)

        discovered = er.discover_exp_assets(exp, checkpoint=source)
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        identity_payload = {
            "schema": "explicit-checkpoint-identity-v3",
            "experiment": "demo",
            "source_run": run.name,
            "source_path": str(source.resolve()),
            "source_name": source.name,
            "source_sha256": source_sha256,
            "effective_config_sha256": er.effective_config_fingerprint(config)["sha256"],
        }
        expected_identity = "demo@model_0060.pth"
        assert discovered["checkpoint_identity_payload"] == identity_payload

        assert discovered["config_path"] == str(config.resolve())
        assert discovered["source_checkpoint_path"] == str(source.resolve())
        assert discovered["explicit_checkpoint"] is True
        assert discovered["checkpoint_identity"] == expected_identity
        assert discovered["source_checkpoint_fingerprint"]["sha256"] == source_sha256
        assert extract_ckpt.default_model_only_path(source) == (
            run / "model_0060.pth.model_only.pt"
        )

        best = run / "best_model.tar"
        torch.save({"model": {"weight": torch.ones(1)}}, best)
        assert extract_ckpt.default_model_only_path(best) == (
            run / "best_model.model_only.pt"
        )

        er._materialize_consumed_checkpoint(
            discovered, trusted_training_checkpoint=False
        )
        consumed = Path(discovered["checkpoint_path"])
        payload = extract_ckpt.validate_model_only_checkpoint(
            consumed, source_path=source, family="single_model"
        )
        assert consumed == run / "model_0060.pth.model_only.pt"
        assert payload["source_fingerprint"] == discovered["source_checkpoint_fingerprint"]

    @pytest.mark.parametrize("changed_asset", ["source", "config"])
    def test_materialization_rejects_assets_changed_after_discovery(
        self, monkeypatch, tmp_path, changed_asset
    ):
        project_root, exp, _run, config, source = self._experiment(tmp_path)
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        discovered = er.discover_exp_assets(exp, checkpoint=source)
        if changed_asset == "source":
            torch.save({"model": {"weight": torch.full((1,), 2.0)}}, source)
            message = "来源训练 checkpoint"
        else:
            config.write_text(
                "model:\n  type: DeepVQE\nstft:\n  n_fft: 512\n",
                encoding="utf-8",
            )
            message = "同 run config"

        with pytest.raises(RuntimeError, match=message):
            er._materialize_consumed_checkpoint(
                discovered, trusted_training_checkpoint=False
            )

    @pytest.mark.unit
    def test_effective_config_change_keeps_identity_and_rejects_reuse(
        self, monkeypatch, tmp_path
    ):
        """base_config 变化不换 identity 命名空间，但复用前必须被消费 binding 拦下。"""
        project_root, exp, run, config, source = self._experiment(tmp_path)
        base = run / "base.yaml"
        base.write_text("stft:\n  n_fft: 320\n", encoding="utf-8")
        config.write_text(
            "base_config: base.yaml\nmodel:\n  type: DeepVQE\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        first = er.discover_exp_assets(exp, checkpoint=source)
        er._materialize_consumed_checkpoint(
            first, trusted_training_checkpoint=False
        )
        self._publish_committed_results(first)

        base.write_text("stft:\n  n_fft: 512\n", encoding="utf-8")
        second = er.discover_exp_assets(exp, checkpoint=source)

        assert first["config_fingerprint"] == second["config_fingerprint"]
        assert first["resolved_config_fingerprint"] != second[
            "resolved_config_fingerprint"
        ]
        assert first["checkpoint_identity"] == second["checkpoint_identity"]
        with pytest.raises(ValueError, match="消费 binding"):
            er.collect_existing_eval_results(first)

    @pytest.mark.unit
    def test_waveform_native_commands_fail_before_launching_unbound_child(
        self, monkeypatch
    ):
        exp = {
            "exp_name": "waveform",
            "exp_dir": "/tmp/waveform",
            "task_type": "se",
            "waveform_native": True,
            "config_path": "/tmp/waveform.yaml",
            "checkpoint_path": None,
        }
        child_calls = []
        monkeypatch.setattr(er, "_discover_from_args", lambda _args: exp.copy())
        monkeypatch.setattr(
            er,
            "_materialize_consumed_checkpoint",
            lambda *_args, **_kwargs: pytest.fail("waveform child guard must run first"),
        )
        monkeypatch.setattr(
            er, "run_eval_pipeline", lambda **_kwargs: child_calls.append(True) or 0
        )
        evaluate_args = argparse.Namespace(
            exp_dir="waveform",
            datasets=None,
            no_audio_quality=True,
            device="cpu",
            inference_devices=None,
            trusted_training_checkpoint=False,
        )
        with pytest.raises(ValueError, match="waveform-native"):
            er.cmd_evaluate(evaluate_args)

        run_args = argparse.Namespace(
            exp_dir="waveform",
            datasets=None,
            no_audio_quality=True,
            device="cpu",
            inference_devices=None,
            trusted_training_checkpoint=False,
            purpose="test",
            skip_eval=False,
            skip_latency_verify=True,
            skip_onnx_verify=True,
            skip_badcase=True,
        )
        monkeypatch.setattr(er, "parse_training_metrics", lambda _data: {})
        monkeypatch.setattr(er, "infer_model_family", lambda _data: "waveform")
        with pytest.raises(ValueError, match="waveform-native"):
            er.cmd_run(run_args)
        assert child_calls == []

    def test_explicit_checkpoint_rejects_unsafe_or_cross_run_binding(
        self, monkeypatch, tmp_path
    ):
        project_root, exp, run, _config, source = self._experiment(tmp_path)
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)

        with pytest.raises(ValueError, match="普通文件"):
            er.discover_exp_assets(exp, checkpoint=run)

        source_link = run / "model_link.pth"
        source_link.symlink_to(source)
        with pytest.raises(ValueError, match="不允许符号链接"):
            er.discover_exp_assets(exp, checkpoint=source_link)

        linked_run = exp / "checkpoints" / "linked-run"
        linked_run.symlink_to(run, target_is_directory=True)
        with pytest.raises(ValueError, match="所在 run 不允许符号链接"):
            er.discover_exp_assets(exp, checkpoint=linked_run / source.name)

        foreign_run = project_root / "exp" / "aec" / "other" / "checkpoints" / "run"
        foreign_run.mkdir(parents=True)
        foreign_source = foreign_run / source.name
        torch.save({"model": {"weight": torch.ones(1)}}, foreign_source)
        with pytest.raises(ValueError, match="该实验 checkpoints"):
            er.discover_exp_assets(exp, checkpoint=foreign_source)

        nested = run / "nested"
        nested.mkdir()
        nested_source = nested / source.name
        torch.save({"model": {"weight": torch.ones(1)}}, nested_source)
        with pytest.raises(ValueError, match="直接位于 checkpoints/<run>"):
            er.discover_exp_assets(exp, checkpoint=nested_source)

        no_config_run = exp / "checkpoints" / "no-config"
        no_config_run.mkdir()
        no_config_source = no_config_run / source.name
        torch.save({"model": {"weight": torch.ones(1)}}, no_config_source)
        with pytest.raises(FileNotFoundError, match="缺少同目录 config"):
            er.discover_exp_assets(exp, checkpoint=no_config_source)

    def test_parent_routes_checkpoint_identity_to_child_and_private_log(
        self, monkeypatch, tmp_path
    ):
        project_root, exp, _run, config, source = self._experiment(tmp_path)
        identity = "demo@model_0060-deadbeefcafe"
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        captured = {}

        class FakeProcess:
            stdout = iter([])

            def wait(self, timeout=None):
                return 0

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return FakeProcess()

        monkeypatch.setattr(er.subprocess, "Popen", fake_popen)
        assert er.run_eval_pipeline(
            str(exp),
            "aec",
            "voicebank",
            device="cpu",
            output_name=identity,
            config_path=config,
            checkpoint_path=source,
        ) == 0
        assert captured["cmd"][captured["cmd"].index("--output-name") + 1] == identity
        assert (
            project_root / "exp" / "eval" / "aec" / identity / "eval_pipeline.log"
        ).is_file()

        for unsafe in ("", "..", "../escape", "a/b", "with space", "中文"):
            with pytest.raises(ValueError, match="安全 basename"):
                er._build_eval_pipeline_cmd(
                    str(exp), "aec", "voicebank", output_name=unsafe
                )

    def test_checkpoint_identity_keeps_index_onnx_and_badcase_outputs_isolated(
        self, monkeypatch, tmp_path
    ):
        _project_root, _exp, _run, _config, _source, selected = (
            self._discover_materialized(monkeypatch, tmp_path)
        )
        base = dict(selected)
        base["checkpoint_identity"] = base["exp_name"]
        base["explicit_checkpoint"] = False
        merged = er._merge_experiment_index([], base, {})
        merged = er._merge_experiment_index(merged, selected, {})

        assert len(merged) == 2
        assert er._index_entry_for_identity(merged, base)["checkpoint_identity"] == "demo"
        assert er._index_entry_for_identity(merged, selected)["checkpoint_identity"] == (
            selected["checkpoint_identity"]
        )
        onnx_target = er._safe_onnx_target(selected, "deepvqe")
        assert onnx_target.name == (
            f"{selected['checkpoint_identity']}_DeepVQE_streaming.onnx"
        )

        monkeypatch.setattr(er, "BADCASE_RESULT_DIR", tmp_path / "badcase-outputs")
        monkeypatch.setattr(
            er, "_resolve_checkpoint_epoch", lambda _data: (60, "checkpoint.epoch")
        )
        result_dir, legacy_dir, epoch, source = er._badcase_result_paths(selected)
        assert result_dir.name == f"{selected['checkpoint_identity']}_epoch0060"
        assert legacy_dir.name == selected["checkpoint_identity"]
        assert (epoch, source) == (60, "checkpoint.epoch")

        samples_root = tmp_path / "badcase-samples"
        samples_root.mkdir()
        sample = samples_root / "mic.wav"
        cleanup_calls = []
        monkeypatch.setattr(er, "BADCASE_TEST_DIR", samples_root)
        monkeypatch.setattr(
            er, "_collect_badcase_samples", lambda *_args: [("demo", sample, sample)]
        )
        monkeypatch.setattr(
            er, "_create_scp_files", lambda *_args: (tmp_path / "mic.scp", tmp_path / "far.scp")
        )
        monkeypatch.setattr(er, "_run_badcase_mode", lambda *_args, **_kwargs: {"status": "done"})
        monkeypatch.setattr(
            er,
            "_cleanup_legacy_badcase_result",
            lambda *_args: cleanup_calls.append(True),
        )
        assert er._run_badcase_inference(selected, device="cpu")["status"] == "done"
        assert cleanup_calls == []

    def test_committed_matching_identity_results_are_reused(self, monkeypatch, tmp_path):
        _project_root, _exp, _run, _config, _source, selected = (
            self._discover_materialized(monkeypatch, tmp_path)
        )
        output, _results_file = self._publish_committed_results(selected)
        envelope = er.load_committed_artifact(output)

        assert envelope["provenance"]["config_fingerprint"]["kind"] == "file"
        assert envelope["provenance"]["checkpoint_fingerprint"]["kind"] == "file"
        assert er.collect_existing_eval_results(selected) == {"voicebank": {"PESQ": 2.5}}
        er._load_eval_device_provenance(selected)
        assert selected["metric_device"] == "cpu"
        assert selected["inference_devices"] == ["cpu"]

    @pytest.mark.parametrize(
        "mutation",
        ["config", "checkpoint", "in_progress", "results"],
    )
    def test_checkpoint_specific_reuse_fails_closed_after_artifact_changes(
        self, monkeypatch, tmp_path, mutation
    ):
        _project_root, _exp, _run, config, _source, selected = (
            self._discover_materialized(monkeypatch, tmp_path)
        )
        output, results_file = self._publish_committed_results(selected)

        if mutation == "config":
            config.write_text(
                "model:\n  type: DeepVQE\nstft:\n  n_fft: 512\n",
                encoding="utf-8",
            )
        elif mutation == "checkpoint":
            Path(selected["checkpoint_path"]).write_bytes(b"changed")
        elif mutation == "in_progress":
            (output / "evaluation_current.json").write_text(
                json.dumps(
                    {
                        "schema": "evaluation-current-v1",
                        "status": "in_progress",
                        "run_id": "different-run",
                    }
                ),
                encoding="utf-8",
            )
        else:
            results_file.write_text("PESQ: 3.0\n", encoding="utf-8")

        with pytest.raises(ValueError):
            er.collect_existing_eval_results(selected)

    def test_checkpoint_specific_reuse_rejects_mismatched_provenance(
        self, monkeypatch, tmp_path
    ):
        _project_root, _exp, _run, _config, _source, selected = (
            self._discover_materialized(monkeypatch, tmp_path)
        )
        wrong_checkpoint = tmp_path / "wrong.model_only.pt"
        wrong_checkpoint.write_bytes(b"wrong")
        self._publish_committed_results(
            selected, provenance_checkpoint=wrong_checkpoint
        )

        with pytest.raises(ValueError, match="checkpoint fingerprint"):
            er.collect_existing_eval_results(selected)

    def test_base_in_progress_output_is_never_reused_for_explicit_checkpoint(
        self, monkeypatch, tmp_path
    ):
        project_root, exp, run, _config, _source = self._experiment(tmp_path)
        best = run / "best_model.tar"
        torch.save({"model": {"weight": torch.ones(1)}, "epoch": 40}, best)
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        selected = er.discover_exp_assets(exp, checkpoint=best)
        assert selected["checkpoint_identity"] != selected["exp_name"]
        assert er._eval_root_for(selected).name == selected["checkpoint_identity"]
        base_root = project_root / "exp" / "eval" / "aec" / "demo"
        base_root.mkdir(parents=True)
        (base_root / "evaluation_current.json").write_text(
            json.dumps(
                {
                    "schema": "evaluation-current-v1",
                    "status": "in_progress",
                    "run_id": "epoch-40",
                }
            ),
            encoding="utf-8",
        )

        assert er._find_eval_root(selected) is None
        assert er.collect_existing_eval_results(selected) == {}
        monkeypatch.setattr(er, "_discover_from_args", lambda _args: selected.copy())
        monkeypatch.setattr(er, "_materialize_consumed_checkpoint", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(er, "infer_experiment_purpose", lambda _data: "test")
        monkeypatch.setattr(er, "parse_training_metrics", lambda _data: {})
        monkeypatch.setattr(
            er,
            "update_report_snapshot",
            lambda *_args: pytest.fail("不得用 base in-progress 输出写报告"),
        )
        args = argparse.Namespace(
            exp_dir=str(exp),
            checkpoint=str(best),
            purpose=None,
            datasets=None,
            skip_eval=True,
            no_audio_quality=False,
            device="cpu",
            inference_devices=None,
            trusted_training_checkpoint=False,
        )
        assert er.cmd_run(args) == 1

    def test_report_skip_eval_check_allows_absent_identity_but_rejects_uncommitted_root(
        self, monkeypatch, tmp_path
    ):
        _project_root, exp, _run, _config, _source, selected = (
            self._discover_materialized(monkeypatch, tmp_path)
        )
        snapshots = []
        monkeypatch.setattr(er, "_discover_from_args", lambda _args: selected.copy())
        monkeypatch.setattr(er, "infer_experiment_purpose", lambda _data: "test")
        monkeypatch.setattr(er, "parse_training_metrics", lambda _data: {})
        monkeypatch.setattr(
            er,
            "update_report_snapshot",
            lambda data, results: snapshots.append((data, results)) or Path("report.md"),
        )
        monkeypatch.setattr(er, "_print_summary", lambda *_args: None)
        args = argparse.Namespace(
            exp_dir=str(exp), checkpoint=None, purpose=None, skip_eval_check=True
        )

        assert er.cmd_report(args) == 0
        assert len(snapshots) == 1
        assert snapshots[0][0]["checkpoint_identity"] == selected["checkpoint_identity"]
        assert snapshots[0][1] == {}

        output = er._eval_root_for(selected)
        output.mkdir(parents=True)
        (output / "evaluation_current.json").write_text(
            json.dumps(
                {
                    "schema": "evaluation-current-v1",
                    "status": "in_progress",
                    "run_id": "unfinished",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="artifact 未提交或无效"):
            er.cmd_report(args)


    def test_explicit_identity_ignores_config_content_and_merges_index(
        self, monkeypatch, tmp_path
    ):
        """identity 只绑定 experiment + checkpoint 名：config 内容变化不换命名空间。"""
        project_root, exp, _run, config, source = self._experiment(tmp_path)
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        first = er.discover_exp_assets(exp, checkpoint=source)
        config.write_text(
            "model:\n  type: DeepVQE\nstft:\n  n_fft: 512\n",
            encoding="utf-8",
        )
        second = er.discover_exp_assets(exp, checkpoint=source)

        assert first["source_checkpoint_fingerprint"] == second["source_checkpoint_fingerprint"]
        assert first["config_fingerprint"] != second["config_fingerprint"]
        assert first["checkpoint_identity"] == second["checkpoint_identity"]
        assert er._eval_root_for(first) == er._eval_root_for(second)
        merged = er._merge_experiment_index([], first, {})
        # config 内容变化属于谱系冲突：合并必须 fail-closed，而不是静默覆盖既有条目。
        with pytest.raises(ValueError, match="谱系冲突"):
            er._merge_experiment_index(merged, second, {})

    def test_explicit_same_named_checkpoints_from_distinct_runs_share_identity(
        self, monkeypatch, tmp_path
    ):
        """同名 checkpoint 跨 run 共用同一 identity；隔离改由 payload 谱系校验承担。"""
        project_root, exp, run, config, source = self._experiment(tmp_path)
        other_run = exp / "checkpoints" / "2026-09-05_19-05-20"
        other_run.mkdir()
        other_config = other_run / config.name
        other_config.write_bytes(config.read_bytes())
        other_source = other_run / source.name
        other_source.write_bytes(source.read_bytes())
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)

        first = er.discover_exp_assets(exp, checkpoint=source)
        second = er.discover_exp_assets(exp, checkpoint=other_source)

        assert first["source_checkpoint_fingerprint"]["sha256"] == (
            second["source_checkpoint_fingerprint"]["sha256"]
        )
        assert first["config_fingerprint"]["sha256"] == second["config_fingerprint"]["sha256"]
        assert first["checkpoint_identity"] == second["checkpoint_identity"]
        assert er._eval_root_for(first) == er._eval_root_for(second)
        # 来源 run 不同 ⇒ payload 谱系不同，复用前由谱系校验拦下。
        assert first["checkpoint_identity_payload"] != second["checkpoint_identity_payload"]

    def test_explicit_checkpoint_uses_only_matching_run_logs_and_git_info(
        self, monkeypatch, tmp_path
    ):
        project_root, exp, run, _config, source = self._experiment(tmp_path)
        matching_log = exp / "logs" / run.name
        matching_log.mkdir(parents=True)
        (matching_log / "val_metrics.csv").write_text(
            "epoch,val_loss\\n60,0.2\\n", encoding="utf-8"
        )
        newer_log = exp / "logs" / "2026-09-06_19-05-20"
        newer_log.mkdir()
        (newer_log / "val_metrics.csv").write_text(
            "epoch,val_loss\\n99,0.1\\n", encoding="utf-8"
        )
        (run / "git_info.json").write_text(
            json.dumps({"commit": "selected"}), encoding="utf-8"
        )
        other_run = exp / "checkpoints" / "2026-09-06_19-05-20"
        other_run.mkdir()
        (other_run / "git_info.json").write_text(
            json.dumps({"commit": "other"}), encoding="utf-8"
        )
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)

        selected = er.discover_exp_assets(exp, checkpoint=source)

        assert selected["log_dir"] == str(matching_log.resolve())
        assert selected["val_metrics_csv"] == str(
            (matching_log / "val_metrics.csv").resolve()
        )
        assert selected["git_info"] == {"commit": "selected"}

    def test_explicit_index_reuse_requires_matching_lineage(self, monkeypatch, tmp_path):
        _project_root, _exp, _run, _config, _source, selected = (
            self._discover_materialized(monkeypatch, tmp_path)
        )
        stale = er._merge_experiment_index([], selected, {})[0]
        stale["config_fingerprint"] = {"sha256": "different"}

        assert er._index_entry_for_identity([stale], selected) is None
        with pytest.raises(ValueError, match="既有索引来源谱系冲突"):
            er._merge_experiment_index([stale], selected, {})

    def test_model_only_paths_are_source_injective_and_canonical_sources_are_rejected(
        self, monkeypatch, tmp_path
    ):
        project_root, exp, run, _config, _source = self._experiment(tmp_path)
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        names = ("best_model.tar", "best_model.pth", "model_0060.pth", "model_0060.tar")
        outputs = {extract_ckpt.default_model_only_path(run / name).name for name in names}
        assert outputs == {
            "best_model.model_only.pt",
            "best_model.pth.model_only.pt",
            "model_0060.pth.model_only.pt",
            "model_0060.tar.model_only.pt",
        }

        canonical = run / "model_0060.pth.model_only.pt"
        canonical.write_bytes(b"not a training checkpoint")
        with pytest.raises(ValueError, match="不能使用 model-only"):
            er.discover_exp_assets(exp, checkpoint=canonical)
        with pytest.raises(ValueError, match="不能作为训练来源"):
            extract_ckpt.materialize_model_only_checkpoint(
                canonical, family="single_model"
            )

    def test_explicit_checkpoint_rejects_symlinked_checkpoints_root(self, monkeypatch, tmp_path):
        project_root = tmp_path / "repo"
        exp = project_root / "exp" / "aec" / "demo"
        exp.mkdir(parents=True)
        storage = tmp_path / "checkpoint-storage"
        run = storage / "2026-09-04_19-05-20"
        run.mkdir(parents=True)
        (run / "resolved_config.yaml").write_text(
            "model:\n  type: DeepVQE\nstft:\n  n_fft: 320\n",
            encoding="utf-8",
        )
        source = run / "best_model.tar"
        torch.save({"model": {"weight": torch.ones(1)}}, source)
        (exp / "checkpoints").symlink_to(storage, target_is_directory=True)
        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)

        with pytest.raises(ValueError, match="checkpoints 目录不允许符号链接"):
            er.discover_exp_assets(exp, checkpoint=source)

    def test_legacy_output_name_is_not_subject_to_explicit_identity_validation(
        self, monkeypatch, tmp_path
    ):
        project_root = tmp_path / "repo"
        legacy_name = "中文 空格+(demo)"
        exp = project_root / "exp" / "aec" / legacy_name
        exp.mkdir(parents=True)
        captured = {}

        class FakeProcess:
            stdout = iter(())

            def wait(self, timeout=None):
                return 0

        monkeypatch.setattr(er, "_PROJECT_ROOT", project_root)
        monkeypatch.setattr(
            er.subprocess,
            "Popen",
            lambda command, **kwargs: captured.update(command=command, kwargs=kwargs) or FakeProcess(),
        )
        assert er.run_eval_pipeline(str(exp), "aec", "voicebank", device="cpu") == 0
        assert "--output-name" not in captured["command"]
        assert (
            project_root / "exp" / "eval" / "aec" / legacy_name / "eval_pipeline.log"
        ).is_file()
        legacy = {
            "exp_name": legacy_name,
            "task_type": "aec",
            "explicit_checkpoint": False,
        }
        assert er._eval_root_for(legacy).name == legacy_name

    def test_explicit_onnx_targets_do_not_collide_after_identity_sanitization(
        self, monkeypatch, tmp_path
    ):
        _project_root, _exp, _run, _config, _source, selected = (
            self._discover_materialized(monkeypatch, tmp_path)
        )
        left = {
            **selected,
            "checkpoint_identity": "demo@model-one",
            "explicit_checkpoint": True,
        }
        right = {
            **selected,
            "checkpoint_identity": "demo_model-one",
            "explicit_checkpoint": True,
        }

        left_target = er._safe_onnx_target(left, "deepvqe")
        right_target = er._safe_onnx_target(right, "deepvqe")
        assert left_target.name != right_target.name
        # '@' 是 identity 合同内的合法字符并被保留，因此不追加第二层哈希也不会撞名。
        assert left_target.name == "demo@model-one_DeepVQE_streaming.onnx"
        assert right_target.name == "demo_model-one_DeepVQE_streaming.onnx"


pytestmark = [pytest.mark.unit]
