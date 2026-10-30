#!/usr/bin/env python3
"""AEC model evaluation pipeline — one-click inference + metrics across multiple testsets.

Supported datasets
  aec_official      All 3 AEC Challenge blind testsets (default subset: doubletalk)
  aec_icassp2021    AEC Challenge ICASSP 2021
  aec_icassp2022    AEC Challenge ICASSP 2022
  aec_icassp2023    AEC Challenge ICASSP 2023
  aec_sim           Simulated AEC testset (7 scenarios)
  voicebank         VoiceBank-DEMAND (joint AEC models)
  urgent            URGENT2026 nonblind paired set (joint AEC models)
  all               Auto-select based on AEC sub-type

Default auto-selection (``--datasets all``)
  joint / unknown  → aec_official, aec_sim, voicebank, urgent, dns_challenge

Usage
  # Full auto pipeline (detect config/checkpoint from exp dir)
  python evaluate/eval_pipeline_aec.py --exp_dir exp/aec/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1 --datasets all --step all

  # Selected datasets only
  python evaluate/eval_pipeline_aec.py --exp_dir exp/... --datasets aec_official,aec_sim

  # Manual config + checkpoint
  python evaluate/eval_pipeline_aec.py -C configs/aec/.../cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml \
      --checkpoint exp/aec/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1/checkpoints/.../best_model.tar --datasets all --step all --audio_quality

  # Inference only
  python evaluate/eval_pipeline_aec.py --exp_dir exp/... --step inference

  # Evaluate only (reads prior enhanced audio)
  python evaluate/eval_pipeline_aec.py --exp_dir exp/... --step evaluate --audio_quality
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluate.common import (
    aggregate_coverages,
    aggregate_metric_coverages,
    append_extra_metrics_to_results,
    apply_metric_coverage,
    build_done_summary,
    build_error_summary,
    build_metric_coverage_from_counts,
    check_enhanced_dir,
    collect_scenario_normalized_scores,
    coverage_status,
    ensure_inf_scp,
    finalize_pipeline_summary,
    load_inference_coverage,
    load_metric_coverage,
    metric_coverage_status,
    prepare_dataset_paths,
    publish_result_manifest,
    requested_standard_metrics,
    require_scp,
    run_aecmos_for_scenario,
    run_standard_evaluation,
    warn_blind_no_reference,
)
from evaluate.eval_pipeline_common import (
    DEFAULT_PATHS,
    DEFAULT_TEST_DATA_ROOT,
    default_eval_output_root,
    AEC_OFFICIAL_TESTSETS,
    find_exp_assets,
    prefer_model_only_checkpoint,
    load_pipeline_config,
    detect_model_type,
    detect_aec_subtype,
    prepare_testset_scps,
    run_inference_aec_paired,
    run_aec_inference_zero_farend,
    prepare_aec_sim_testset,
    ensure_aec_sim_scps,
    AEC_SIM_REF_SCENARIOS,
    AEC_SIM_SCENARIOS,
    compute_registered_scenario_metrics,
    run_evaluate_urgent,
    print_summary_table,
    resolve_eval_output_name,
    guard_explicit_eval_output,
    validate_evaluation_precommit_snapshot,
)
from evaluate.pipeline_runner import PipelineTaskAdapter, run_pipeline
from evaluate.multi_gpu_inference import normalize_inference_devices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_pipeline_aec")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEC model evaluation pipeline")
    g_model = p.add_argument_group("Model")
    g_model.add_argument("--exp_dir", help="Experiment directory (auto-find config + checkpoint)")
    g_model.add_argument("-C", "--config", help="Model config YAML")
    g_model.add_argument("--checkpoint", help="Model checkpoint path")
    g_model.add_argument("--expected-effective-config-sha256", default=None)
    g_model.add_argument(
        "--output-name",
        help="安全评估输出目录名；默认沿用 --exp_dir 或 config 父目录名称",
    )

    g_data = p.add_argument_group("Datasets")
    g_data.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated tags: aec_official,aec_sim,voicebank,urgent,dns_challenge,all",
    )
    g_data.add_argument(
        "--aec_subsets",
        default="all",
        help="AEC official subsets to test (default: all). Use 'doubletalk' for single subset.",
    )
    g_data.add_argument("--aec_testset_root", default=str(DEFAULT_PATHS["aec_testset_root"]))
    g_data.add_argument("--aec_sim_testset_root", default=str(DEFAULT_PATHS["aec_sim_testset_root"]))
    g_data.add_argument("--voicebank_noisy_scp", default=str(DEFAULT_PATHS["voicebank_noisy_scp"]))
    g_data.add_argument("--voicebank_clean_scp", default=str(DEFAULT_PATHS["voicebank_clean_scp"]))
    g_data.add_argument("--urgent_noisy_scp", default=str(DEFAULT_PATHS["urgent_noisy_scp"]))
    g_data.add_argument("--urgent_clean_scp", default=str(DEFAULT_PATHS["urgent_clean_scp"]))
    g_data.add_argument("--urgent_noisy_dir", default=str(DEFAULT_PATHS["urgent_noisy_dir"]))
    g_data.add_argument("--urgent_clean_dir", default=str(DEFAULT_PATHS["urgent_clean_dir"]))
    g_data.add_argument("--dns_challenge_noisy_dir", default=str(DEFAULT_PATHS["dns_challenge_noisy_dir"]))

    g_run = p.add_argument_group("Run control")
    g_run.add_argument(
        "--step",
        choices=["all", "inference", "evaluate"],
        default="all",
        help="all: inference + evaluate; inference: only; evaluate: only (requires prior outputs)",
    )
    g_run.add_argument("--device", default="cuda:0", help="指标、延迟和其它评估阶段使用的设备")
    g_run.add_argument(
        "--inference-devices",
        default=None,
        help="仅模型推理使用的设备列表，例如 cuda:0,cuda:1；默认使用 --device",
    )
    g_run.add_argument("--audio_quality", action="store_true", help="Enable DNSMOS/NISQA/UTMOS/SCOREQ")
    g_run.add_argument("--aecmos", action="store_true", help="Explicitly enable optional AECMOS metrics")
    g_run.add_argument("--downstream", action="store_true", help="Enable downstream metrics for URGENT")
    g_run.add_argument(
        "--allow-partial-inference",
        action="store_true",
        help="Allow evaluation and partial status when some inference items fail.",
    )
    g_run.add_argument("--num_sim_samples", type=int, default=100, help="Samples per scenario for AEC sim testset")
    g_run.add_argument("--seed", type=int, default=42, help="Random seed for AEC sim testset generation")
    g_run.add_argument(
        "--aec-sim-config",
        default=None,
        help="Override config used for aec_sim testset generation (enables cross-experiment fair comparison).",
    )
    g_run.add_argument(
        "--aec_no_stream_align",
        action="store_true",
        help="AEC: disable stream alignment; by default, alignment behavior follows the YAML configuration.",
    )
    g_run.add_argument(
        "--skip-existing-inference",
        action="store_true",
        help="Skip inference for files whose enhanced output already exists (resume-friendly).",
    )
    return p.parse_args()


def resolve_config_and_checkpoint(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.exp_dir:
        return find_exp_assets(args.exp_dir, prefer_model_only=False)
    if args.config and args.checkpoint:
        logger.warning(
            "显式同时指定 --config 与 --checkpoint：请确认两者来自同一 run，"
            "否则评估结果可能张冠李戴 (A21)。"
        )
        return (
            Path(args.config).expanduser().resolve(),
            Path(args.checkpoint).expanduser().resolve(),
        )
    raise SystemExit("Provide --exp_dir OR both --config and --checkpoint")


def resolve_datasets(tag_str: str) -> list[str]:
    """展开 ``--datasets``；``all`` 固定覆盖 AEC 联合（降噪+回声）评估全集。"""
    tags = [t.strip() for t in tag_str.split(",") if t.strip()]
    if "all" not in tags:
        return tags

    return ["aec_official", "aec_sim", "voicebank", "urgent", "dns_challenge"]


def resolve_aec_subsets(tag_str: str) -> list[str]:
    if tag_str.strip().lower() == "all":
        return ["doubletalk", "farend-singletalk", "nearend-singletalk"]
    return [s.strip() for s in tag_str.split(",") if s.strip()]


_AEC_DATASET_ALIASES = {
    "aec_icassp2023": "blind_test_set_icassp2023",
}


def build_dataset_plan(
    datasets: list[str], aec_subsets: list[str]
) -> tuple[list[tuple[str, str | None, str | None]], list[str]]:
    """构建执行计划，并单独返回未知 dataset 供机器可读错误上报。"""
    plan: list[tuple[str, str | None, str | None]] = []
    unknown: list[str] = []
    for ds in datasets:
        if ds == "aec_official":
            for challenge in AEC_OFFICIAL_TESTSETS:
                for subset in aec_subsets:
                    plan.append((f"{challenge}_{subset}", challenge, subset))
        elif ds in _AEC_DATASET_ALIASES:
            challenge = _AEC_DATASET_ALIASES[ds]
            for subset in aec_subsets:
                plan.append((f"{challenge}_{subset}", challenge, subset))
        elif ds in {"aec_sim", "voicebank", "urgent", "dns_challenge"}:
            plan.append((ds, None, None))
        else:
            unknown.append(ds)
    return plan, unknown


_AECMOS_METRIC_KEYS = ("AECMOS_ECHO", "EMOS", "AECMOS_DEG", "DMOS")


def _aecmos_metrics(result: dict | None) -> dict[str, float]:
    """仅提取评估器实际返回的 AECMOS 指标键。"""
    if result is None:
        return {}
    return {
        key: float(result[key])
        for key in _AECMOS_METRIC_KEYS
        if key in result
    }


def _aecmos_metric_counts(result: dict | None) -> dict[str, int]:
    """为实际存在的 AECMOS 指标生成成功样本数。"""
    if result is None:
        return {}
    count = int(result["n_samples"])
    return {key: count for key in _AECMOS_METRIC_KEYS if key in result}


def _log_aecmos_result(scope: str, result: dict) -> None:
    """记录当前 talk type 实际有效的 AECMOS 指标。"""
    metrics = _aecmos_metrics(result)
    values = " ".join(f"{key}={value:.3f}" for key, value in metrics.items())
    logger.info(
        "[%s] AECMOS %s (n=%s, failed=%s)",
        scope,
        values,
        result["n_samples"],
        result.get("n_failed", 0),
    )


def _run_optional_aecmos(enabled: bool, scope: str, **kwargs) -> dict:
    """仅在显式请求时运行可选 AECMOS，错误降级为可报告的 skipped。"""
    if not enabled:
        return {"skipped": True, "skip_reason": "not_requested", "n_samples": 0}
    try:
        result = run_aecmos_for_scenario(**kwargs)
    except Exception as exc:
        logger.warning("[%s] AECMOS skipped: %s", scope, exc)
        return {
            "skipped": True,
            "skip_reason": f"{type(exc).__name__}: {exc}",
            "n_samples": 0,
        }
    return result or {"skipped": True, "skip_reason": "no_common_samples", "n_samples": 0}


def _aecmos_skipped_coverage(result: dict | None, talk_type: str, eligible: int) -> dict:
    """将可选 AECMOS 的跳过原因写入 coverage，不影响主指标完成状态。"""
    if result is None or not result.get("skipped"):
        return {}
    metric_keys = {
        "st": ("AECMOS_ECHO", "EMOS"),
        "nst": ("AECMOS_DEG", "DMOS"),
        "dt": _AECMOS_METRIC_KEYS,
    }[talk_type]
    return {
        key: {
            "eligible": eligible,
            "attempted": 0,
            "computed": None,
            "failed": None,
            "coverage": None,
            "status": "skipped",
            "reason": result.get("skip_reason", "unknown"),
        }
        for key in metric_keys
    }


# ---------------------------------------------------------------------------
# Per-dataset processors
# ---------------------------------------------------------------------------
def process_aec_official(
    challenge_name: str,
    subset: str,
    args: argparse.Namespace,
    config: Path,
    checkpoint: Path,
    eval_root: Path,
    summary_datasets: dict,
    scp_paths: dict,
):
    ds_tag = f"{challenge_name}_{subset}"
    scp_info = scp_paths.get(ds_tag)
    if scp_info is None:
        logger.warning("AEC official SCPs not found for %s.", ds_tag)
        summary_datasets[ds_tag] = {"status": "error", "reason": "scp_missing"}
        return

    ds_eval = eval_root / ds_tag
    enhanced_dir = ds_eval / "enhanced"
    results_dir = ds_eval

    # ---- Inference ----
    if args.step in ("all", "inference"):
        logger.info(f"[{ds_tag}] Running inference ...")
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_inference_aec_paired(
                config=config,
                checkpoint=checkpoint,
                mic_scp=scp_info["mic"],
                far_scp=scp_info["lpb"],
                output_dir=enhanced_dir,
                device=args.device,
                no_stream_align=args.aec_no_stream_align,
                skip_existing=args.skip_existing_inference,
                allow_partial=args.allow_partial_inference,
                inference_devices=getattr(args, "inference_devices", None),
                expected_effective_config_identity=args.effective_config_identity,
            )
        except Exception as exc:
            summary_datasets[ds_tag] = build_error_summary(
                str(exc), enhanced_dir, expected_scp=scp_info["mic"],
                config=config, checkpoint=checkpoint,
            )
            return

    try:
        inf_scp = ensure_inf_scp(enhanced_dir)
        coverage = load_inference_coverage(
            enhanced_dir,
            expected_scp=scp_info["mic"],
            config=config,
            checkpoint=checkpoint,
            evaluation_performed=False,
        )
    except (OSError, ValueError) as exc:
        summary_datasets[ds_tag] = build_error_summary(
            str(exc), enhanced_dir, expected_scp=scp_info["mic"],
            config=config, checkpoint=checkpoint,
        )
        return

    # ---- Evaluation ----
    if args.step in ("all", "evaluate"):
        inference_status = coverage_status(
            coverage, args.allow_partial_inference, require_evaluated=False
        )
        if inference_status == "error":
            summary_datasets[ds_tag] = build_error_summary(
                "incomplete inference coverage", enhanced_dir, coverage=coverage
            )
            return
        summary_datasets[ds_tag] = build_error_summary(
            "evaluation_incomplete", enhanced_dir, coverage=coverage
        )
        logger.info(f"[{ds_tag}] Running evaluation ...")
        if not check_enhanced_dir(enhanced_dir, ds_tag, summary_datasets):
            summary_datasets[ds_tag] = build_error_summary(
                "no_enhanced", enhanced_dir, coverage=coverage
            )
            return

        if not args.audio_quality:
            logger.warning(
                f"[{ds_tag}] AEC official blind testset has no clean reference. "
                f"Add --audio_quality to enable non-intrusive MOS metrics (DNSMOS/NISQA/UTMOS/SCOREQ)."
            )

        inf_scp = ensure_inf_scp(enhanced_dir)

        # ERLE 仅适用于回声存在且目标近似静音的子集；盲测 doubletalk 为 N/A。
        erle = None
        if subset == "farend-singletalk":
            mic_scp = scp_info.get("mic")
            if mic_scp and inf_scp.exists():
                erle = compute_registered_scenario_metrics(
                    subset, mic_scp, inf_scp
                ).get("erle")
                if erle is not None:
                    logger.info(f"[{ds_tag}] ERLE = {erle['mean_erle_db']:.2f} dB (n={erle['n_samples']})")
        elif subset == "doubletalk":
            logger.info(f"[{ds_tag}] ERLE = N/A (blind doubletalk has no oracle echo; see AECMOS)")

        # AECMOS 是显式可选指标；缺依赖/资源或单条异常不得中断主评估。
        aecmos_result = None
        mic_scp = scp_info.get("mic")
        lpb_scp = scp_info.get("lpb")
        talk_type = {
            "doubletalk": "dt",
            "nearend-singletalk": "nst",
            "farend-singletalk": "st",
        }.get(subset)
        if talk_type is not None:
            if mic_scp and lpb_scp and inf_scp.exists():
                aecmos_result = _run_optional_aecmos(
                    getattr(args, "aecmos", False),
                    ds_tag,
                    mic_scp=mic_scp,
                    lpb_scp=lpb_scp,
                    enh_scp=inf_scp,
                    talk_type=talk_type,
                )
            else:
                aecmos_result = {
                    "skipped": True,
                    "skip_reason": "missing_scp_or_enhanced_output",
                    "n_samples": 0,
                }
            if not aecmos_result.get("skipped"):
                _log_aecmos_result(ds_tag, aecmos_result)

        try:
            run_evaluate_urgent(
                inf_scp=inf_scp,
                output_dir=results_dir,
                ref_scp=None,  # blind testset: no reference
                audio_quality=args.audio_quality,
                device=args.device,
            )
            coverage = load_inference_coverage(
                enhanced_dir,
                expected_scp=scp_info["mic"],
                config=config,
                checkpoint=checkpoint,
                evaluation_performed=True,
            )
        except Exception as exc:
            summary_datasets[ds_tag] = build_error_summary(
                str(exc), enhanced_dir, coverage=coverage
            )
            return

        # Append ERLE / AECMOS to RESULTS.txt
        extra_metrics: dict[str, float] = {}
        if erle is not None:
            extra_metrics["ERLE_dB"] = erle["mean_erle_db"]
        extra_metrics.update(_aecmos_metrics(aecmos_result))
        if extra_metrics:
            append_extra_metrics_to_results(results_dir, extra_metrics)

        metric_coverage = load_metric_coverage(
            results_dir,
            requested_standard_metrics(
                ref_scp=None, audio_quality=args.audio_quality
            ),
            eligible=coverage["evaluated"],
        )
        extra_counts: dict[str, int | None] = {}
        if subset == "farend-singletalk":
            extra_counts["ERLE_dB"] = erle["n_samples"] if erle else None
        extra_counts.update(_aecmos_metric_counts(aecmos_result))
        metric_coverage.update(
            _aecmos_skipped_coverage(aecmos_result, talk_type, coverage["evaluated"])
        )
        metric_coverage.update(
            build_metric_coverage_from_counts(coverage["evaluated"], extra_counts)
        )
        applied_metrics = {
            key: value for key, value in metric_coverage.items()
            if value.get("status") != "skipped"
        }
        coverage = apply_metric_coverage(coverage, applied_metrics)
        summary = build_done_summary(
            enhanced_dir,
            results_dir,
            coverage=coverage,
            allow_partial=args.allow_partial_inference,
            metric_coverage=metric_coverage,
        )
        if erle is not None:
            summary["erle"] = erle
        if aecmos_result is not None:
            summary["aecmos"] = aecmos_result
        summary_datasets[ds_tag] = summary
    else:
        summary_datasets[ds_tag] = {
            "status": coverage_status(
                coverage, args.allow_partial_inference, require_evaluated=False
            ),
            "enhanced_dir": str(enhanced_dir),
            "coverage": coverage,
        }


def process_aec_sim(
    args: argparse.Namespace,
    config: Path,
    checkpoint: Path,
    eval_root: Path,
    summary_datasets: dict,
    scp_paths: dict,
):
    ds_tag = "aec_sim"
    ds_eval = eval_root / ds_tag
    configured_testset_root = Path(args.aec_sim_testset_root)
    testset_root = (
        (_PROJECT_ROOT / configured_testset_root).resolve()
        if configured_testset_root == DEFAULT_PATHS["aec_sim_testset_root"]
        else configured_testset_root.expanduser().resolve()
    )

    # ---- Prepare testset (if missing) ----
    if args.step in ("all", "inference"):
        try:
            prepare_aec_sim_testset(
                config=config,
                output_dir=testset_root,
                num_samples=args.num_sim_samples,
                seed=args.seed,
                sim_config=args.aec_sim_config,
            )
        except Exception as exc:
            logger.error(f"Failed to prepare AEC sim testset: {exc}")
            summary_datasets[ds_tag] = {"status": "error", "reason": str(exc)}
            return

    if not testset_root.exists():
        summary_datasets[ds_tag] = {"status": "error", "reason": "testset_missing"}
        return

    # ---- Ensure SCPs exist ----
    try:
        sim_scps = scp_paths.get("aec_sim", {}) or ensure_aec_sim_scps(testset_root)
    except Exception as exc:
        summary_datasets[ds_tag] = {"status": "error", "reason": str(exc)}
        return

    scenarios = [name for name in AEC_SIM_SCENARIOS if name in sim_scps]
    if args.step in ("all", "evaluate"):
        for scenario in scenarios:
            if scenario in {"prob_echo_only", "prob_noise_echo_only", "prob_noise_only"}:
                append_extra_metrics_to_results(
                    ds_eval / scenario, {}, replace=True
                )

    missing_scenarios = [name for name in AEC_SIM_SCENARIOS if name not in sim_scps]
    if missing_scenarios and not args.allow_partial_inference:
        summary_datasets[ds_tag] = {
            "status": "error",
            "reason": "missing_aec_sim_scenarios",
            "missing_scenarios": missing_scenarios,
        }
        return
    if not scenarios:
        summary_datasets[ds_tag] = {"status": "error", "reason": "no_scenarios_found"}
        return

    # ---- Per-scenario inference + evaluation ----
    scenario_coverages: dict[str, dict] = {}
    scenario_metric_coverages: dict[str, dict] = {}
    evaluator_scenarios: set[str] = set()
    scenario_results_files: dict[str, str] = {}
    for scenario in scenarios:
        scen_scps = sim_scps[scenario]
        scen_eval = ds_eval / scenario
        enhanced_dir = scen_eval / "enhanced"
        results_dir = scen_eval
        mic_scp = scen_scps.get("mic")
        far_scp = scen_scps.get("farend")
        ref_scp = scen_scps.get("nearend") if scenario in AEC_SIM_REF_SCENARIOS else None

        try:
            if not mic_scp or not far_scp:
                raise ValueError(f"[{scenario}] missing mic/farend SCP")
            if args.step in ("all", "inference"):
                enhanced_dir.mkdir(parents=True, exist_ok=True)
                run_inference_aec_paired(
                    config, checkpoint,
                    mic_scp=mic_scp,
                    far_scp=far_scp,
                    output_dir=enhanced_dir,
                    device=args.device,
                    no_stream_align=args.aec_no_stream_align,
                    skip_existing=args.skip_existing_inference,
                    allow_partial=args.allow_partial_inference,
                    inference_devices=getattr(args, "inference_devices", None),
                    expected_effective_config_identity=args.effective_config_identity,
                )
            ensure_inf_scp(enhanced_dir)
            scenario_coverage = load_inference_coverage(
                enhanced_dir,
                expected_scp=mic_scp,
                config=config,
                checkpoint=checkpoint,
                evaluation_performed=False,
            )
            scenario_coverages[scenario] = scenario_coverage
            if args.step in ("all", "evaluate") and coverage_status(
                scenario_coverage,
                args.allow_partial_inference,
                require_evaluated=False,
            ) == "error":
                raise RuntimeError(f"[{scenario}] incomplete inference coverage")
        except Exception as exc:
            error_summary = build_error_summary(
                str(exc), enhanced_dir, expected_scp=mic_scp,
                config=config, checkpoint=checkpoint,
                coverage=scenario_coverages.get(scenario),
            )
            if "coverage" in error_summary:
                scenario_coverages[scenario] = error_summary["coverage"]
            summary_datasets[ds_tag] = {
                "status": "error",
                "reason": str(exc),
                "scenarios": scenarios,
                "scenario_coverage": scenario_coverages,
                "coverage": aggregate_coverages(list(scenario_coverages.values())),
                "eval_root": str(ds_eval),
            }
            return

    if args.step in ("all", "evaluate"):
        for scenario in scenarios:
            scen_scps = sim_scps[scenario]
            results_dir = ds_eval / scenario
            enhanced_dir = results_dir / "enhanced"
            mic_scp = scen_scps["mic"]
            ref_scp = (
                scen_scps.get("nearend")
                if scenario in AEC_SIM_REF_SCENARIOS else None
            )
            try:
                if scenario in {"prob_echo_only", "prob_noise_echo_only", "prob_noise_only"}:
                    logger.info(
                        f"[{scenario}] Skipping MOS metrics (no near-end speech). "
                        "ERLE (echo-bearing) / NOISE_REDUCTION (prob_noise_only) will be reported separately."
                    )
                else:
                    if ref_scp is None and not args.audio_quality:
                        logger.warning(
                            f"[{scenario}] No clean reference available. "
                            "Use --audio_quality for MOS-only metrics."
                        )
                    run_evaluate_urgent(
                        inf_scp=enhanced_dir / "inf.scp",
                        output_dir=results_dir,
                        ref_scp=ref_scp,
                        audio_quality=args.audio_quality,
                        downstream=args.downstream,
                        device=args.device,
                    )
                    evaluator_scenarios.add(scenario)
                    scenario_results_files[scenario] = str(
                        (results_dir / "RESULTS.txt").resolve()
                    )
                    scenario_coverages[scenario] = load_inference_coverage(
                        enhanced_dir,
                        ref_scp,
                        expected_scp=mic_scp,
                        config=config,
                        checkpoint=checkpoint,
                        evaluation_performed=True,
                    )
                    requested_metrics = requested_standard_metrics(
                        ref_scp=ref_scp,
                        audio_quality=args.audio_quality,
                        downstream=args.downstream,
                    )
                    if requested_metrics:
                        scenario_metric_coverages[scenario] = load_metric_coverage(
                            results_dir,
                            requested_metrics,
                            eligible=scenario_coverages[scenario]["evaluated"],
                        )
            except Exception as exc:
                summary_datasets[ds_tag] = {
                    "status": "error",
                    "reason": str(exc),
                    "scenarios": scenarios,
                    "scenario_coverage": scenario_coverages,
                    "coverage": aggregate_coverages(list(scenario_coverages.values())),
                    "eval_root": str(ds_eval),
                }
                return

    # 先保存推理覆盖率，确保后续指标异常时仍保留该信息。
    summary_datasets[ds_tag] = {
        "status": "error",
        "reason": "evaluation_incomplete",
        "scenarios": scenarios,
        "scenario_coverage": scenario_coverages,
        "coverage": aggregate_coverages(list(scenario_coverages.values())),
        "eval_root": str(ds_eval),
    }

    # ---- ERLE (echo-only scenarios) ----
    erle_results: dict[str, dict] = {}
    aecmos_results: dict[str, dict] = {}
    nr_results: dict[str, dict] = {}
    if args.step in ("all", "evaluate"):
        for scenario in scenarios:
            scen_scps = sim_scps[scenario]
            scen_eval = ds_eval / scenario
            enhanced_dir = scen_eval / "enhanced"
            inf_scp = ensure_inf_scp(enhanced_dir)
            mic_scp = scen_scps.get("mic")
            ref_scp = scen_scps.get("nearend") if scenario in AEC_SIM_REF_SCENARIOS else None

            # ERLE / 降噪指标由 registry 按场景分派；doubletalk 的 oracle echo
            # 投影口径及 near-end SI-SDR 仍由原指标实现负责。
            if mic_scp and inf_scp.exists():
                registered_metrics = compute_registered_scenario_metrics(
                    scenario,
                    mic_scp,
                    inf_scp,
                    ref_scp=ref_scp,
                    echo_scp=scen_scps.get("echo"),
                )
                erle = registered_metrics.get("erle")
                if erle is not None:
                    erle_results[scenario] = erle
                    msg = f"[{scenario}] ERLE = {erle['mean_erle_db']:.2f} dB (n={erle['n_samples']})"
                    if "mean_nearend_sisdr_db" in erle:
                        msg += f" | NEAREND_SISDR = {erle['mean_nearend_sisdr_db']:.2f} dB"
                    logger.info(msg)

                nr = registered_metrics.get("noise_reduction")
                if nr is not None:
                    nr_results[scenario] = nr
                    logger.info(f"[{scenario}] NOISE_REDUCTION = {nr['mean_nr_db']:.2f} dB (n={nr['n_samples']})")

            # AECMOS for echo-bearing scenarios（仅 --aecmos 显式请求）。
            if scenario in {"prob_echo_only", "prob_noise_echo_only", "prob_target_echo", "prob_target_noise_echo"}:
                far_scp = scen_scps.get("farend")
                if mic_scp and far_scp and inf_scp.exists():
                    talk_type = "st" if scenario in {"prob_echo_only", "prob_noise_echo_only"} else "dt"
                    aecmos = _run_optional_aecmos(
                        getattr(args, "aecmos", False),
                        scenario,
                        mic_scp=mic_scp,
                        lpb_scp=far_scp,
                        enh_scp=inf_scp,
                        talk_type=talk_type,
                    )
                    aecmos_results[scenario] = aecmos
                    if not aecmos.get("skipped"):
                        _log_aecmos_result(scenario, aecmos)

            # Append per-scenario ERLE / AECMOS / NR to RESULTS.txt
            extra_metrics: dict[str, float] = {}
            if scenario in erle_results:
                extra_metrics["ERLE_dB"] = erle_results[scenario]["mean_erle_db"]
                if "mean_nearend_sisdr_db" in erle_results[scenario]:
                    extra_metrics["NEAREND_SISDR_dB"] = erle_results[scenario][
                        "mean_nearend_sisdr_db"
                    ]
            extra_metrics.update(_aecmos_metrics(aecmos_results.get(scenario)))
            if scenario in nr_results:
                extra_metrics["NOISE_REDUCTION_dB"] = nr_results[scenario]["mean_nr_db"]
            extra_only = scenario in {"prob_echo_only", "prob_noise_echo_only", "prob_noise_only"}
            if extra_metrics or extra_only:
                append_extra_metrics_to_results(
                    scen_eval, extra_metrics, replace=extra_only
                )
                scenario_results_files[scenario] = str(
                    (scen_eval / "RESULTS.txt").resolve()
                )

            eligible = scenario_coverages[scenario]["produced"]
            extra_counts: dict[str, int | None] = {}
            if scenario in {"prob_echo_only", "prob_noise_echo_only", "prob_target_echo", "prob_target_noise_echo"}:
                erle = erle_results.get(scenario)
                aecmos = aecmos_results.get(scenario)
                extra_counts["ERLE_dB"] = erle["n_samples"] if erle else None
                if scenario in {"prob_target_echo", "prob_target_noise_echo"}:
                    extra_counts["NEAREND_SISDR_dB"] = (
                        erle["n_samples"]
                        if erle and "mean_nearend_sisdr_db" in erle else None
                    )
                extra_counts.update(_aecmos_metric_counts(aecmos))
            elif scenario == "prob_noise_only":
                nr = nr_results.get(scenario)
                extra_counts["NOISE_REDUCTION_dB"] = nr["n_samples"] if nr else None
            if extra_counts:
                scenario_metric_coverages.setdefault(scenario, {}).update(
                    build_metric_coverage_from_counts(eligible, extra_counts)
                )
            if scenario in {"prob_echo_only", "prob_noise_echo_only", "prob_target_echo", "prob_target_noise_echo"}:
                talk_type = "st" if scenario in {"prob_echo_only", "prob_noise_echo_only"} else "dt"
                scenario_metric_coverages.setdefault(scenario, {}).update(
                    _aecmos_skipped_coverage(aecmos_results.get(scenario), talk_type, eligible)
                )
            applied_metrics = {
                key: value
                for key, value in scenario_metric_coverages.get(scenario, {}).items()
                if value.get("status") != "skipped"
            }
            scenario_coverages[scenario] = apply_metric_coverage(
                scenario_coverages[scenario], applied_metrics,
            )

    coverage = aggregate_coverages(list(scenario_coverages.values()))
    metric_coverage = aggregate_metric_coverages(
        list(scenario_metric_coverages.values())
    ) if scenario_metric_coverages else None
    status = coverage_status(
        coverage,
        args.allow_partial_inference,
        require_evaluated=args.step in ("all", "evaluate"),
        metric_coverage=metric_coverage,
    )
    if missing_scenarios and status == "done":
        status = "partial"
    scenario_scores, scenario_score_files = collect_scenario_normalized_scores(
        ds_eval, sorted(evaluator_scenarios)
    )
    summary = {
        "status": status,
        "scenarios": scenarios,
        "missing_scenarios": missing_scenarios,
        "scenario_coverage": scenario_coverages,
        "scenario_metric_coverage": scenario_metric_coverages,
        "scenario_results_files": scenario_results_files,
        "coverage": coverage,
        "eval_root": str(ds_eval),
        "erle": erle_results,
    }
    if scenario_scores:
        summary["scenario_normalized_scores"] = scenario_scores
        summary["scenario_normalized_score_files"] = scenario_score_files
    if metric_coverage is not None:
        summary["metric_coverage"] = metric_coverage
        summary["metric_coverage_status"] = metric_coverage_status(metric_coverage)
    if aecmos_results:
        summary["aecmos"] = aecmos_results
    if nr_results:
        summary["noise_reduction"] = nr_results
    summary_datasets[ds_tag] = summary


def _run_aec_se_dataset(
    ds_tag: str,
    args: argparse.Namespace,
    config: Path,
    checkpoint: Path,
    eval_root: Path,
    summary_datasets: dict,
    scp_paths: dict,
    *,
    ref_key: Optional[str] = "clean",
    blind: bool = False,
) -> None:
    """AEC 模型在 SE 测试集（voicebank/urgent/dns_challenge）上的通用评估。"""
    _, enhanced_dir, results_dir = prepare_dataset_paths(eval_root, ds_tag)

    scp_info = require_scp(scp_paths, ds_tag, summary_datasets)
    if scp_info is None:
        return

    noisy_scp = scp_info["noisy"]
    ref_scp = scp_info.get(ref_key) if not blind else None

    if args.step in ("all", "inference"):
        logger.info(f"[{ds_tag}] Running zero-farend AEC inference ...")
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_aec_inference_zero_farend(
                config_path=config,
                checkpoint_path=checkpoint,
                input_scp=noisy_scp,
                output_dir=enhanced_dir,
                device=args.device,
                allow_partial=args.allow_partial_inference,
                inference_devices=getattr(args, "inference_devices", None),
                expected_effective_config_identity=args.effective_config_identity,
            )
        except Exception as exc:
            summary_datasets[ds_tag] = build_error_summary(
                str(exc), enhanced_dir, expected_scp=noisy_scp,
                config=config, checkpoint=checkpoint,
            )
            return

    try:
        ensure_inf_scp(enhanced_dir)
        coverage = load_inference_coverage(
            enhanced_dir,
            expected_scp=noisy_scp,
            config=config,
            checkpoint=checkpoint,
            evaluation_performed=False,
        )
    except (OSError, ValueError) as exc:
        summary_datasets[ds_tag] = build_error_summary(
            str(exc), enhanced_dir, expected_scp=noisy_scp,
            config=config, checkpoint=checkpoint,
        )
        return

    if args.step in ("all", "evaluate"):
        inference_status = coverage_status(
            coverage, args.allow_partial_inference, require_evaluated=False
        )
        if inference_status == "error":
            summary_datasets[ds_tag] = build_error_summary(
                "incomplete inference coverage", enhanced_dir, coverage=coverage
            )
            return
        logger.info(f"[{ds_tag}] Running evaluation ...")
        if not check_enhanced_dir(enhanced_dir, ds_tag, summary_datasets):
            summary_datasets[ds_tag] = build_error_summary(
                "no_enhanced", enhanced_dir, coverage=coverage
            )
            return
        try:
            if blind:
                warn_blind_no_reference(ds_tag)

            # AECMOS for SE testsets（lpb=zeros；仅 --aecmos 显式请求）。
            aecmos_result = None
            inf_scp = ensure_inf_scp(enhanced_dir)
            if inf_scp.exists():
                aecmos_result = _run_optional_aecmos(
                    getattr(args, "aecmos", False),
                    ds_tag,
                    mic_scp=noisy_scp,
                    lpb_scp=None,
                    enh_scp=inf_scp,
                    talk_type="nst",
                )
                if not aecmos_result.get("skipped"):
                    _log_aecmos_result(ds_tag, aecmos_result)

            run_standard_evaluation(
                enhanced_dir=enhanced_dir,
                results_dir=results_dir,
                ref_scp=ref_scp,
                audio_quality=args.audio_quality,
                downstream=args.downstream if ds_tag == "urgent" else False,
                device=args.device,
            )
            coverage = load_inference_coverage(
                enhanced_dir,
                ref_scp,
                expected_scp=noisy_scp,
                config=config,
                checkpoint=checkpoint,
                evaluation_performed=True,
            )
            if aecmos_result is not None:
                append_extra_metrics_to_results(
                    results_dir, _aecmos_metrics(aecmos_result)
                )
        except Exception as exc:
            summary_datasets[ds_tag] = build_error_summary(
                str(exc), enhanced_dir, coverage=coverage
            )
            return

        metric_coverage = load_metric_coverage(
            results_dir,
            requested_standard_metrics(
                ref_scp=ref_scp,
                audio_quality=args.audio_quality,
                downstream=args.downstream if ds_tag == "urgent" else False,
            ),
            eligible=coverage["evaluated"],
        )
        metric_coverage.update(build_metric_coverage_from_counts(
            coverage["evaluated"], _aecmos_metric_counts(aecmos_result)
        ))
        metric_coverage.update(
            _aecmos_skipped_coverage(aecmos_result, "nst", coverage["evaluated"])
        )
        applied_metrics = {
            key: value for key, value in metric_coverage.items()
            if value.get("status") != "skipped"
        }
        coverage = apply_metric_coverage(coverage, applied_metrics)
        summary = build_done_summary(
            enhanced_dir,
            results_dir,
            coverage=coverage,
            allow_partial=args.allow_partial_inference,
            metric_coverage=metric_coverage,
        )
        if aecmos_result is not None:
            summary["aecmos"] = aecmos_result
        summary_datasets[ds_tag] = summary
    else:
        summary_datasets[ds_tag] = {
            "status": coverage_status(
                coverage, args.allow_partial_inference, require_evaluated=False
            ),
            "enhanced_dir": str(enhanced_dir),
            "coverage": coverage,
        }


def process_voicebank(
    args: argparse.Namespace,
    config: Path,
    checkpoint: Path,
    eval_root: Path,
    summary_datasets: dict,
    scp_paths: dict,
):
    _run_aec_se_dataset("voicebank", args, config, checkpoint, eval_root, summary_datasets, scp_paths)


def process_urgent(
    args: argparse.Namespace,
    config: Path,
    checkpoint: Path,
    eval_root: Path,
    summary_datasets: dict,
    scp_paths: dict,
):
    _run_aec_se_dataset("urgent", args, config, checkpoint, eval_root, summary_datasets, scp_paths)


def process_dns_challenge(
    args: argparse.Namespace,
    config: Path,
    checkpoint: Path,
    eval_root: Path,
    summary_datasets: dict,
    scp_paths: dict,
):
    _run_aec_se_dataset(
        "dns_challenge", args, config, checkpoint, eval_root, summary_datasets, scp_paths,
        ref_key=None, blind=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    config, checkpoint = resolve_config_and_checkpoint(args)
    effective_config, effective_identity = load_pipeline_config(
        config,
        expected_sha256=getattr(args, "expected_effective_config_sha256", None),
    )
    if args.exp_dir:
        checkpoint = prefer_model_only_checkpoint(effective_config, checkpoint)
    args.effective_config_identity = effective_identity
    args.device = getattr(args, "device", "cuda:0")
    args.inference_devices = normalize_inference_devices(
        getattr(args, "inference_devices", None), fallback=args.device
    )
    mtype = detect_model_type(effective_config, config_path=config)
    subtype = detect_aec_subtype(effective_config, config_path=config)
    datasets = resolve_datasets(args.datasets)
    aec_subsets = resolve_aec_subsets(args.aec_subsets)

    if mtype != "aec":
        logger.warning(f"Config suggests model type='{mtype}', but you launched eval_pipeline_aec.py")

    exp_name = resolve_eval_output_name(
        getattr(args, "output_name", None), exp_dir=args.exp_dir, config=config
    )
    eval_root = default_eval_output_root(_PROJECT_ROOT) / "aec" / exp_name
    explicit_output_name = getattr(args, "output_name", None) is not None
    if not explicit_output_name:
        eval_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Evaluation root: {eval_root}")
    logger.info(f"Model subtype: {subtype} | Datasets: {datasets} | Step: {args.step}")

    summary = {
        "pipeline": "eval_pipeline_aec",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "model_type": mtype,
        "model_subtype": subtype,
        "step": args.step,
        "device": args.device,
        "inference_devices": args.inference_devices,
        "output_name": exp_name,
        "effective_config_identity": effective_identity,
        "datasets": {},
    }
    adapter = PipelineTaskAdapter(
        task="aec",
        pipeline="eval_pipeline_aec",
        step=args.step,
        output_dir=eval_root,
        provenance={
            "config": config,
            "checkpoint": checkpoint,
            "resolved_config_fingerprint": effective_identity,
            "parameters": {
                "step": args.step,
                "model_type": mtype,
                "model_subtype": subtype,
                "device": args.device,
                "inference_devices": args.inference_devices,
                "output_name": exp_name,
            },
            "entry": "eval_pipeline_aec",
        },
    )
    summary["provenance"] = adapter._provenance()

    def _prepare() -> dict:
        logger.info("Preparing standardized SCPs under %s ...", DEFAULT_TEST_DATA_ROOT)
        scp_paths = prepare_testset_scps(
            aec_testset_root=args.aec_testset_root,
            aec_subsets=aec_subsets,
            urgent_noisy_scp=getattr(args, "urgent_noisy_scp", None),
            urgent_clean_scp=getattr(args, "urgent_clean_scp", None),
            urgent_noisy_dir=args.urgent_noisy_dir,
            urgent_clean_dir=args.urgent_clean_dir,
            voicebank_noisy_scp=args.voicebank_noisy_scp,
            voicebank_clean_scp=args.voicebank_clean_scp,
            dns_challenge_noisy_dir=args.dns_challenge_noisy_dir,
            selected_datasets=datasets,
        )
        logger.info(f"SCP preparation done. Available keys: {list(scp_paths.keys())}")
        return scp_paths

    def _run(scp_paths: dict) -> dict:
        from time import time as _time

        plan, unknown_datasets = build_dataset_plan(datasets, aec_subsets)
        for ds in unknown_datasets:
            logger.error("Unknown dataset tag: %s", ds)
            summary["datasets"][ds] = {
                "status": "error",
                "reason": "unknown_dataset",
                "dataset": ds,
            }

        started = _time()
        for index, (tag, challenge, subset) in enumerate(plan, 1):
            dataset_started = _time()
            logger.info(
                "=" * 60 + f"\n[{index}/{len(plan)}] {tag} 开始评估 ...\n" + "=" * 60
            )
            try:
                if challenge and subset:
                    process_aec_official(
                        challenge,
                        subset,
                        args,
                        config,
                        checkpoint,
                        eval_root,
                        summary["datasets"],
                        scp_paths,
                    )
                elif tag == "aec_sim":
                    process_aec_sim(
                        args, config, checkpoint, eval_root, summary["datasets"], scp_paths
                    )
                elif tag == "voicebank":
                    process_voicebank(
                        args, config, checkpoint, eval_root, summary["datasets"], scp_paths
                    )
                elif tag == "urgent":
                    process_urgent(
                        args, config, checkpoint, eval_root, summary["datasets"], scp_paths
                    )
                elif tag == "dns_challenge":
                    process_dns_challenge(
                        args, config, checkpoint, eval_root, summary["datasets"], scp_paths
                    )
            except Exception as exc:
                logger.exception("[%s] dataset processing failed", tag)
                existing = summary["datasets"].get(tag, {})
                summary["datasets"][tag] = {
                    **existing,
                    "status": "error",
                    "reason": str(exc),
                }

            elapsed_dataset = _time() - dataset_started
            elapsed_total = _time() - started
            eta = (
                (elapsed_total / index) * (len(plan) - index)
                if index < len(plan)
                else 0
            )
            logger.info(
                f"[{index}/{len(plan)}] {tag} 完成 ({elapsed_dataset:.0f}s) | "
                f"累计 {elapsed_total:.0f}s | 预计剩余 {eta:.0f}s"
            )

        finalize_pipeline_summary(summary)
        return summary

    def _on_failure(stage: str, exc: Exception) -> dict:
        logger.exception("AEC pipeline %s failed", stage)
        reason = (
            f"scp_preparation_failed: {exc}"
            if stage == "prepare"
            else f"{stage}_failed: {exc}"
        )
        for ds in datasets:
            summary["datasets"].setdefault(
                ds, {"status": "error", "reason": reason}
            )
        finalize_pipeline_summary(summary)
        summary["status"] = "error"
        summary["reason"] = reason
        return summary

    def _on_complete(result: dict) -> None:
        publish_result_manifest(eval_root, result)
        print_summary_table(result)

    def _precommit(result: dict) -> None:
        validate_evaluation_precommit_snapshot(
            result,
            config=config,
            checkpoint=checkpoint,
            expected_effective_config_identity=effective_identity,
        )

    result = run_pipeline(
        adapter,
        prepare=_prepare,
        run=_run,
        on_complete=_on_complete,
        on_failure=_on_failure,
        precommit=_precommit,
        require_result_manifest=True,
        preflight=(
            (lambda: guard_explicit_eval_output(adapter))
            if explicit_output_name
            else None
        ),
        reraise=False,
    )
    return 0 if result["status"] != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
