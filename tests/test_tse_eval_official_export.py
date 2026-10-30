"""eval_pipeline_tse 官方格式 CSV 导出 + aggregate_eval_summary 汇总链路测试。

用合成 rows 验证：
- 导出的五个 CSV 文件名/列名与官方 aggregate_eval_summary.py 消费约定一致；
- 缺指标 / NaN 行被过滤；
- 移植版 aggregate 能直接消费导出产物并算出正确分组均值。
"""

from __future__ import annotations

import pandas as pd
import pytest

from evaluate.aggregate_eval_summary import summarize_one_output_dir
from tasks.tse.inference import artifacts as tse_inference_artifacts
from evaluate.artifact_lifecycle import attach_run_id, begin_artifact_run, new_run_id
from evaluate.common import publish_result_manifest
from evaluate.eval_pipeline_tse import _export_official_format_csvs, _write_tse_summary


def _rows() -> list[dict]:
    return [
        {
            "mixture_utterance": "utt_zh_1",
            "dataset": "AISHELL-4",
            "language": "zh",
            "ter": 0.5,
            "spksim_tse_enroll": 0.40,
            "spksim_mix_enroll": 0.20,
            "dnsmos_sig": 3.0, "dnsmos_bak": 2.0, "dnsmos_ovrl": 2.5,
            "dnsmos_p808": 3.5,
            "timing_precision": 0.8, "timing_recall": 0.7, "timing_f1": 0.75,
        },
        {
            "mixture_utterance": "utt_en_1",
            "dataset": "AMI",
            "language": "en",
            "ter": 0.25,
            "spksim_tse_enroll": 0.60,
            "spksim_mix_enroll": 0.30,
            "dnsmos_sig": 3.4, "dnsmos_bak": 2.4, "dnsmos_ovrl": 2.9,
            "dnsmos_p808": 3.8,
            "timing_precision": 0.9, "timing_recall": 0.8, "timing_f1": 0.85,
        },
        {  # 指标全缺（模拟推理/指标失败行），应被全部过滤
            "mixture_utterance": "utt_bad",
            "dataset": "AMI",
            "language": "en",
        },
        {  # NaN 指标行，应被过滤
            "mixture_utterance": "utt_nan",
            "dataset": "AMI",
            "language": "en",
            "ter": float("nan"),
            "spksim_tse_enroll": float("nan"),
        },
    ]


def _publish_current_summary(out_dir, payload):
    run_id = new_run_id()
    summary = attach_run_id(payload["summary"], run_id)
    begin_artifact_run(out_dir, run_id)
    artifacts = []
    results = out_dir / "tse_eval_results.csv"
    if results.is_file():
        artifacts.append(results)
    artifacts.extend(sorted((out_dir / "eval_metrics").glob("*.csv")))
    publish_result_manifest(out_dir, summary, artifact_paths=artifacts)
    _write_tse_summary(
        out_dir,
        {**payload, "run_id": run_id, "summary": summary},
        active_run=True,
    )


@pytest.fixture()
def eval_dir(tmp_path):
    out_dir = tmp_path / "mock_eval"
    out_dir.mkdir()
    written = _export_official_format_csvs(_rows(), out_dir)
    return out_dir, written


class TestOfficialFormatExport:
    def test_writes_all_five_csvs_with_official_names(self, eval_dir):
        out_dir, written = eval_dir
        names = sorted(p.name for p in written)
        assert names == sorted([
            "mock_eval_TER.csv",
            "mock_eval_spk_similarity.csv",
            "mock_eval_spk_similarity_mixture_enrol.csv",
            "mock_eval_dnsmos.csv",
            "mock_eval_TSE_TIMING.csv",
        ])
        for p in written:
            assert p.parent == out_dir / "eval_metrics"

    def test_column_contract_and_row_filtering(self, eval_dir):
        out_dir, _ = eval_dir
        sub = out_dir / "eval_metrics"

        ter = pd.read_csv(sub / "mock_eval_TER.csv")
        assert list(ter.columns) == ["mixture_utterance", "source", "language", "wer_or_cer"]
        assert len(ter) == 2  # bad 行与 NaN 行被过滤

        sim = pd.read_csv(sub / "mock_eval_spk_similarity.csv")
        assert list(sim.columns) == [
            "utterance", "dataset", "language", "speaker_cosine_similarity", "status"]
        assert (sim["status"] == "ok").all() and len(sim) == 2

        dns = pd.read_csv(sub / "mock_eval_dnsmos.csv")
        assert list(dns.columns) == ["utterance", "dataset", "SIG", "BAK", "OVRL", "P808", "status"]

        tim = pd.read_csv(sub / "mock_eval_TSE_TIMING.csv")
        assert list(tim.columns) == ["utterance", "dataset", "language", "precision", "recall", "f1"]

    def test_no_metrics_writes_nothing(self, tmp_path):
        out_dir = tmp_path / "empty_eval"
        out_dir.mkdir()
        rows = [{"mixture_utterance": "u", "dataset": "AMI", "language": "en"}]
        assert _export_official_format_csvs(rows, out_dir) == []
        assert not (out_dir / "eval_metrics").exists()


class TestAggregateConsumesExport:
    def test_summary_values(self, eval_dir):
        out_dir, _ = eval_dir
        txt_path = summarize_one_output_dir(out_dir, None, "eval_metrics")
        content = txt_path.read_text(encoding="utf-8")

        assert "REAL-T Aggregated Evaluation Summary" in content
        assert "Mean overall" in content
        # overall TER = mean(0.5, 0.25) = 0.375
        assert "0.375000" in content
        # overall sim_enrol_tse = mean(0.40, 0.60) = 0.5
        assert "0.500000" in content
        # 语种分组行存在（zh→chs 归一化）
        assert "chs" in content
        assert "official CSV fallback UID union" in content

    def test_public_aggregate_writer_uses_output_lock(self, eval_dir):
        out_dir, _ = eval_dir
        with tse_inference_artifacts.exclusive_output_run(out_dir):
            with pytest.raises(RuntimeError, match="已有 writer"):
                summarize_one_output_dir(out_dir, None, "eval_metrics")

    def test_current_summary_ignores_stale_inactive_and_zero_computed_csvs(
        self, tmp_path, monkeypatch
    ):
        out_dir = tmp_path / "current_eval"
        metrics_dir = out_dir / "eval_metrics"
        metrics_dir.mkdir(parents=True)
        stale_ter = metrics_dir / "current_eval_TER.csv"
        stale_dnsmos = metrics_dir / "current_eval_dnsmos.csv"
        stale_ter.write_text(
            "mixture_utterance,source,language,wer_or_cer\nold,AISHELL-4,zh,9.9\n",
            encoding="utf-8",
        )
        stale_dnsmos.write_text(
            "utterance,dataset,SIG,BAK,OVRL,P808,status\n"
            "old,AISHELL-4,9.1,9.2,9.3,9.4,ok\n",
            encoding="utf-8",
        )
        timing = metrics_dir / "current_eval_TSE_TIMING.csv"
        timing.write_text(
            "utterance,dataset,language,precision,recall,f1\n"
            "new,AMI,en,0.8,0.7,0.75\n",
            encoding="utf-8",
        )
        _publish_current_summary(out_dir, {
            "metrics": ["dnsmos", "timing"],
            "summary": {
                "status": "done",
                "metrics": {
                    "requested": ["ter", "dnsmos", "timing"],
                    "active": ["dnsmos", "timing"],
                    "skipped": [{"metric": "ter", "reason": "unavailable"}],
                },
                "coverage": {
                    "dnsmos": {
                        "eligible": 1, "attempted": 0,
                        "computed": 0, "failed": 0, "coverage": 0.0,
                    },
                    "timing": {
                        "eligible": 1, "attempted": 1,
                        "computed": 1, "failed": 0, "coverage": 1.0,
                    },
                },
            },
        })

        real_read_csv = pd.read_csv

        def guarded_read_csv(path, *args, **kwargs):
            assert path not in {stale_ter, stale_dnsmos}
            return real_read_csv(path, *args, **kwargs)

        monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
        content = summarize_one_output_dir(
            out_dir, None, "eval_metrics"
        ).read_text(encoding="utf-8")

        assert "9.900000" not in content
        assert "9.100000" not in content
        assert "0.750000" in content
        assert str(stale_ter) not in content
        assert str(stale_dnsmos) not in content

    def test_samples_use_metric_uid_union_without_ter_or_duplicates(self, tmp_path):
        out_dir = tmp_path / "no_ter_eval"
        metrics_dir = out_dir / "eval_metrics"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "no_ter_eval_spk_similarity.csv").write_text(
            "utterance,dataset,language,speaker_cosine_similarity,status\n"
            "shared,AMI,en,0.6,ok\n"
            "sim_only,AMI,en,0.7,ok\n",
            encoding="utf-8",
        )
        (metrics_dir / "no_ter_eval_spk_similarity_mixture_enrol.csv").write_text(
            "utterance,dataset,language,speaker_cosine_similarity,status\n"
            "shared,AMI,en,0.3,ok\n"
            "sim_only,AMI,en,0.4,ok\n",
            encoding="utf-8",
        )
        (metrics_dir / "no_ter_eval_TSE_TIMING.csv").write_text(
            "utterance,dataset,language,precision,recall,f1\n"
            "shared,AMI,en,0.8,0.7,0.75\n"
            "timing_only,AMI,en,0.9,0.8,0.85\n",
            encoding="utf-8",
        )
        pd.DataFrame([
            {
                "dataset": "AMI", "mixture_utterance": "shared",
                "enrolment_speakers_utterance": "enroll_shared", "language": "en",
                "spksim_tse_enroll": 0.6, "spksim_mix_enroll": 0.3,
                "timing_precision": 0.8, "timing_recall": 0.7,
                "timing_f1": 0.75,
            },
            {
                "dataset": "AMI", "mixture_utterance": "sim_only",
                "enrolment_speakers_utterance": "enroll_sim", "language": "en",
                "spksim_tse_enroll": 0.7, "spksim_mix_enroll": 0.4,
            },
            {
                "dataset": "AMI", "mixture_utterance": "timing_only",
                "enrolment_speakers_utterance": "enroll_timing", "language": "en",
                "timing_precision": 0.9, "timing_recall": 0.8,
                "timing_f1": 0.85,
            },
        ]).to_csv(out_dir / "tse_eval_results.csv", index=False)
        _publish_current_summary(out_dir, {
            "metrics": ["spksim", "timing"],
            "summary": {
                "status": "done",
                "metrics": {
                    "requested": ["spksim", "timing"],
                    "active": ["spksim", "timing"],
                    "skipped": [],
                },
                "coverage": {
                    "ter": {
                        "eligible": 0, "attempted": 0,
                        "computed": 0, "failed": 0, "coverage": None,
                    },
                    "spksim": {
                        "eligible": 2, "attempted": 2,
                        "computed": 2, "failed": 0, "coverage": 1.0,
                    },
                    "timing": {
                        "eligible": 2, "attempted": 2,
                        "computed": 2, "failed": 0, "coverage": 1.0,
                    },
                },
            },
        })

        content = summarize_one_output_dir(
            out_dir, None, "eval_metrics"
        ).read_text(encoding="utf-8")

        ami_row = next(line for line in content.splitlines() if line.startswith("| AMI"))
        en_row = next(line for line in content.splitlines() if line.startswith("| en "))
        overall_row = next(
            line for line in content.splitlines() if line.startswith("| overall")
        )
        assert ami_row.split("|")[2].strip() == "3"
        assert en_row.split("|")[2].strip() == "3"
        assert overall_row.split("|")[2].strip() == "3"
        assert "internal results composite UID union" in content

    def test_samples_distinguish_enrollments_for_same_mixture(self, tmp_path):
        out_dir = tmp_path / "multi_enroll_eval"
        metrics_dir = out_dir / "eval_metrics"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "multi_enroll_eval_TSE_TIMING.csv").write_text(
            "utterance,dataset,language,precision,recall,f1\n"
            "shared,AMI,en,0.8,0.7,0.75\n"
            "shared,AMI,en,0.9,0.8,0.85\n",
            encoding="utf-8",
        )
        pd.DataFrame([
            {
                "dataset": "AMI", "mixture_utterance": "shared",
                "enrolment_speakers_utterance": "enroll_a", "language": "en",
                "timing_precision": 0.8, "timing_recall": 0.7,
                "timing_f1": 0.75,
            },
            {
                "dataset": "AMI", "mixture_utterance": "shared",
                "enrolment_speakers_utterance": "enroll_b", "language": "en",
                "timing_precision": 0.9, "timing_recall": 0.8,
                "timing_f1": 0.85,
            },
        ]).to_csv(out_dir / "tse_eval_results.csv", index=False)
        _publish_current_summary(out_dir, {
            "metrics": ["timing"],
            "summary": {
                "status": "done",
                "metrics": {
                    "requested": ["timing"], "active": ["timing"],
                    "skipped": [],
                },
                "coverage": {
                    "timing": {
                        "eligible": 2, "attempted": 2,
                        "computed": 2, "failed": 0, "coverage": 1.0,
                    },
                },
            },
        })

        content = summarize_one_output_dir(
            out_dir, None, "eval_metrics"
        ).read_text(encoding="utf-8")
        ami_row = next(line for line in content.splitlines() if line.startswith("| AMI"))
        overall_row = next(
            line for line in content.splitlines() if line.startswith("| overall")
        )
        assert ami_row.split("|")[2].strip() == "2"
        assert overall_row.split("|")[2].strip() == "2"

    def test_samples_require_all_composite_metric_fields(self, tmp_path):
        out_dir = tmp_path / "partial_composite_eval"
        metrics_dir = out_dir / "eval_metrics"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "partial_composite_eval_dnsmos.csv").write_text(
            "utterance,dataset,SIG,BAK,OVRL,P808,status\n"
            "complete,AMI,3.0,2.0,2.5,3.5,ok\n",
            encoding="utf-8",
        )
        pd.DataFrame([
            {
                "dataset": "AMI", "mixture_utterance": "complete",
                "enrolment_speakers_utterance": "enroll_a", "language": "en",
                "dnsmos_sig": 3.0, "dnsmos_bak": 2.0,
                "dnsmos_ovrl": 2.5, "dnsmos_p808": 3.5,
            },
            {
                "dataset": "AMI", "mixture_utterance": "partial",
                "enrolment_speakers_utterance": "enroll_b", "language": "en",
                "dnsmos_sig": 3.1, "dnsmos_bak": 2.1,
                "dnsmos_ovrl": 2.6,
            },
        ]).to_csv(out_dir / "tse_eval_results.csv", index=False)
        _publish_current_summary(out_dir, {
            "metrics": ["dnsmos"],
            "summary": {
                "status": "partial",
                "metrics": {
                    "requested": ["dnsmos"], "active": ["dnsmos"],
                    "skipped": [],
                },
                "coverage": {
                    "dnsmos": {
                        "eligible": 2, "attempted": 2,
                        "computed": 1, "failed": 1, "coverage": 0.5,
                    },
                },
            },
        })

        content = summarize_one_output_dir(
            out_dir, None, "eval_metrics"
        ).read_text(encoding="utf-8")
        overall_row = next(
            line for line in content.splitlines() if line.startswith("| overall")
        )
        assert overall_row.split("|")[2].strip() == "1"

pytestmark = [pytest.mark.integration]
