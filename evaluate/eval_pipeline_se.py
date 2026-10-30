#!/usr/bin/env python3
"""SE model evaluation pipeline — one-click inference + metrics across multiple testsets.

Supported datasets
  voicebank      VoiceBank-DEMAND
  urgent         URGENT2026 nonblind paired set
  dns_challenge  DNS Challenge V3 blind testset (non-intrusive only)
  aec_sim        Simulated AEC testset (7 scenarios) — optional
  all            voicebank + urgent + dns_challenge (+ aec_sim if --include_aec_sim)

Usage
  # Full auto pipeline
  python evaluate/eval_pipeline_se.py --exp_dir exp/se/gtcrn_causal_offline \
      --datasets all --step all --audio_quality

  # Inference only
  python evaluate/eval_pipeline_se.py --exp_dir exp/... --step inference

  # Evaluate only
  python evaluate/eval_pipeline_se.py --exp_dir exp/... --step evaluate

  # Include AEC simulated testset (SE model on noisy_se input)
  python evaluate/eval_pipeline_se.py --exp_dir exp/... --datasets all --include_aec_sim
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluate.common import (
    aggregate_coverages,
    aggregate_metric_coverages,
    apply_metric_coverage,
    build_done_summary,
    build_error_summary,
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
    run_standard_evaluation,
    warn_blind_no_reference,
)
from evaluate.eval_pipeline_common import (
    DEFAULT_PATHS,
    DEFAULT_TEST_DATA_ROOT,
    default_eval_output_root,
    find_exp_assets,
    prefer_model_only_checkpoint,
    load_pipeline_config,
    detect_model_type,
    prepare_testset_scps,
    run_inference_se,
    prepare_aec_sim_testset,
    ensure_aec_sim_scps,
    AEC_SIM_REF_SCENARIOS,
    AEC_SIM_SCENARIOS,
    run_evaluate_urgent,
    print_summary_table,
    resolve_eval_output_name,
    guard_explicit_eval_output,
    validate_evaluation_precommit_snapshot,
)
from evaluate.pipeline_runner import PipelineTaskAdapter, run_pipeline
from evaluate.multi_gpu_inference import normalize_inference_devices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_pipeline_se")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SE model evaluation pipeline")
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
        help="Comma-separated tags: voicebank,urgent,dns_challenge,aec_sim,mc_sim,all",
    )
    g_data.add_argument("--include_aec_sim", action="store_true", help="Include aec_sim when --datasets=all")
    g_data.add_argument("--include_mc_sim", action="store_true", help="Include mc_sim when --datasets=all")
    g_data.add_argument("--aec_sim_testset_root", default=str(DEFAULT_PATHS["aec_sim_testset_root"]))
    g_data.add_argument("--mc_sim_testset_root", default="data/se/mc_sim_testset")
    g_data.add_argument("--mc_sim_target_scp", default=None)
    g_data.add_argument("--mc_sim_interference_scp", default=None)
    g_data.add_argument("--mc_sim_sir_range_db", type=float, nargs=2, default=(-5.0, 5.0))
    g_data.add_argument("--voicebank_noisy_scp", default=str(DEFAULT_PATHS["voicebank_noisy_scp"]))
    g_data.add_argument("--voicebank_clean_scp", default=str(DEFAULT_PATHS["voicebank_clean_scp"]))
    g_data.add_argument("--urgent_noisy_dir", default=str(DEFAULT_PATHS["urgent_noisy_dir"]))
    g_data.add_argument("--urgent_clean_dir", default=str(DEFAULT_PATHS["urgent_clean_dir"]))
    g_data.add_argument("--dns_challenge_noisy_dir", default=str(DEFAULT_PATHS["dns_challenge_noisy_dir"]))

    g_run = p.add_argument_group("Run control")
    g_run.add_argument(
        "--step",
        choices=["all", "inference", "evaluate"],
        default="all",
        help="all: inference + evaluate; inference: only; evaluate: only",
    )
    g_run.add_argument("--device", default="cuda:0", help="指标、延迟和其它评估阶段使用的设备")
    g_run.add_argument(
        "--inference-devices",
        default=None,
        help="仅模型推理使用的设备列表，例如 cuda:0,cuda:1；默认使用 --device",
    )
    g_run.add_argument("--audio_quality", action="store_true", help="Enable DNSMOS/NISQA/UTMOS/SCOREQ")
    g_run.add_argument("--downstream", action="store_true", help="Enable downstream metrics for URGENT")
    g_run.add_argument(
        "--allow-partial-inference",
        action="store_true",
        help="Allow evaluation and partial status when some inference items fail.",
    )
    g_run.add_argument("--num_sim_samples", type=int, default=100)
    g_run.add_argument("--seed", type=int, default=42)
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


def resolve_datasets(
    tag_str: str, include_aec_sim: bool, include_mc_sim: bool = False
) -> list[str]:
    tags = [t.strip() for t in tag_str.split(",") if t.strip()]
    if "all" not in tags:
        return tags
    base = ["voicebank", "urgent", "dns_challenge"]
    if include_aec_sim:
        base.append("aec_sim")
    if include_mc_sim:
        base.append("mc_sim")
    return base


# ---------------------------------------------------------------------------
# Per-dataset processors
# ---------------------------------------------------------------------------
def _run_se_dataset(
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
    """SE 评估中 voicebank/urgent/dns_challenge 的通用处理。"""
    _, enhanced_dir, results_dir = prepare_dataset_paths(eval_root, ds_tag)

    scp_info = require_scp(scp_paths, ds_tag, summary_datasets)
    if scp_info is None:
        return

    noisy_scp = scp_info["noisy"]
    ref_scp = scp_info.get(ref_key) if not blind else None

    if args.step in ("all", "inference"):
        logger.info(f"[{ds_tag}] Running inference ...")
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_inference_se(
                config=config,
                checkpoint=checkpoint,
                input_scp=noisy_scp,
                output_dir=enhanced_dir,
                device=args.device,
                allow_partial=args.allow_partial_inference,
                inference_devices=getattr(args, "inference_devices", None),
                expected_effective_config_identity=args.effective_config_identity,
                skip_existing=getattr(args, "skip_existing_inference", False),
            )
        except Exception as exc:
            logger.error("[%s] inference failed: %s", ds_tag, exc)
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
        except Exception as exc:
            summary_datasets[ds_tag] = build_error_summary(
                str(exc), enhanced_dir, coverage=coverage
            )
            return
        requested_metrics = requested_standard_metrics(
            ref_scp=ref_scp,
            audio_quality=args.audio_quality,
            downstream=args.downstream if ds_tag == "urgent" else False,
        )
        metric_coverage = (
            load_metric_coverage(
                results_dir, requested_metrics, eligible=coverage["evaluated"]
            )
            if requested_metrics else None
        )
        if metric_coverage is not None:
            coverage = apply_metric_coverage(coverage, metric_coverage)
        summary_datasets[ds_tag] = build_done_summary(
            enhanced_dir,
            results_dir,
            coverage=coverage,
            allow_partial=args.allow_partial_inference,
            metric_coverage=metric_coverage,
        )
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
    _run_se_dataset("voicebank", args, config, checkpoint, eval_root, summary_datasets, scp_paths)


def process_urgent(
    args: argparse.Namespace,
    config: Path,
    checkpoint: Path,
    eval_root: Path,
    summary_datasets: dict,
    scp_paths: dict,
):
    _run_se_dataset("urgent", args, config, checkpoint, eval_root, summary_datasets, scp_paths)


def process_dns_challenge(
    args: argparse.Namespace,
    config: Path,
    checkpoint: Path,
    eval_root: Path,
    summary_datasets: dict,
    scp_paths: dict,
):
    _run_se_dataset(
        "dns_challenge", args, config, checkpoint, eval_root, summary_datasets, scp_paths,
        ref_key=None, blind=True,
    )


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

    missing_scenarios = [name for name in AEC_SIM_SCENARIOS if name not in sim_scps]
    if missing_scenarios and not args.allow_partial_inference:
        summary_datasets[ds_tag] = {
            "status": "error",
            "reason": "missing_aec_sim_scenarios",
            "missing_scenarios": missing_scenarios,
        }
        return
    scenarios = [name for name in AEC_SIM_SCENARIOS if name in sim_scps]
    if not scenarios:
        summary_datasets[ds_tag] = {"status": "error", "reason": "no_scenarios_found"}
        return

    scenario_coverages: dict[str, dict] = {}
    scenario_metric_coverages: dict[str, dict] = {}
    evaluator_scenarios: set[str] = set()
    scenario_results_files: dict[str, str] = {}
    for scenario in scenarios:
        scen_scps = sim_scps[scenario]
        results_dir = ds_eval / scenario
        enhanced_dir = results_dir / "enhanced"
        noisy_se_scp = scen_scps.get("noisy_se")
        try:
            if not noisy_se_scp:
                raise ValueError(f"[{scenario}] missing noisy_se SCP")
            if args.step in ("all", "inference"):
                enhanced_dir.mkdir(parents=True, exist_ok=True)
                run_inference_se(
                    config=config,
                    checkpoint=checkpoint,
                    input_scp=noisy_se_scp,
                    output_dir=enhanced_dir,
                    device=args.device,
                    allow_partial=args.allow_partial_inference,
                    inference_devices=args.inference_devices,
                    expected_effective_config_identity=args.effective_config_identity,
                    skip_existing=getattr(args, "skip_existing_inference", False),
                )
            ensure_inf_scp(enhanced_dir)
            scenario_coverage = load_inference_coverage(
                enhanced_dir,
                expected_scp=noisy_se_scp,
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
                str(exc), enhanced_dir, expected_scp=noisy_se_scp,
                config=config, checkpoint=checkpoint,
                coverage=scenario_coverages.get(scenario),
            )
            if "coverage" in error_summary:
                scenario_coverages[scenario] = error_summary["coverage"]
            aggregate = aggregate_coverages(list(scenario_coverages.values()))
            summary_datasets[ds_tag] = {
                "status": "error",
                "reason": str(exc),
                "scenarios": scenarios,
                "scenario_coverage": scenario_coverages,
                "coverage": aggregate,
                "eval_root": str(ds_eval),
            }
            return

    if args.step in ("all", "evaluate"):
        for scenario in scenarios:
            scen_scps = sim_scps[scenario]
            results_dir = ds_eval / scenario
            enhanced_dir = results_dir / "enhanced"
            noisy_se_scp = scen_scps["noisy_se"]
            ref_scp = (
                scen_scps.get("nearend")
                if scenario in AEC_SIM_REF_SCENARIOS else None
            )
            try:
                run_evaluate_urgent(
                    inf_scp=enhanced_dir / "inf.scp",
                    output_dir=results_dir,
                    ref_scp=ref_scp,
                    audio_quality=args.audio_quality,
                    device=args.device,
                )
                evaluator_scenarios.add(scenario)
                scenario_results_files[scenario] = str(
                    (results_dir / "RESULTS.txt").resolve()
                )
                scenario_coverages[scenario] = load_inference_coverage(
                    enhanced_dir,
                    ref_scp,
                    expected_scp=noisy_se_scp,
                    config=config,
                    checkpoint=checkpoint,
                    evaluation_performed=True,
                )
                requested_metrics = requested_standard_metrics(
                    ref_scp=ref_scp, audio_quality=args.audio_quality
                )
                if requested_metrics:
                    scenario_metric_coverages[scenario] = load_metric_coverage(
                        results_dir,
                        requested_metrics,
                        eligible=scenario_coverages[scenario]["evaluated"],
                    )
                    scenario_coverages[scenario] = apply_metric_coverage(
                        scenario_coverages[scenario], scenario_metric_coverages[scenario]
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
    summary_datasets[ds_tag] = {
        "status": status,
        "scenarios": scenarios,
        "missing_scenarios": missing_scenarios,
        "scenario_coverage": scenario_coverages,
        "scenario_metric_coverage": scenario_metric_coverages,
        "scenario_results_files": scenario_results_files,
        "coverage": coverage,
        "eval_root": str(ds_eval),
    }
    if scenario_scores:
        summary_datasets[ds_tag]["scenario_normalized_scores"] = scenario_scores
        summary_datasets[ds_tag]["scenario_normalized_score_files"] = (
            scenario_score_files
        )
    if metric_coverage is not None:
        summary_datasets[ds_tag]["metric_coverage"] = metric_coverage
        summary_datasets[ds_tag]["metric_coverage_status"] = metric_coverage_status(
            metric_coverage
        )


def process_mc_sim(
    args: argparse.Namespace,
    config: Path,
    checkpoint: Path,
    eval_root: Path,
    summary_datasets: dict,
):
    """Evaluate SpatialNet on a WSJ0-mix-style set and run the oracle MVDR baseline."""
    import hashlib
    import json

    from tools.shared.scp_io import read_scp_dict, write_scp_entries

    configured_root = Path(args.mc_sim_testset_root).expanduser()
    testset_root = (
        (_PROJECT_ROOT / configured_root).resolve()
        if not configured_root.is_absolute()
        else configured_root.resolve()
    )
    manifest_path = testset_root / "mc_sim_manifest.json"
    if not manifest_path.is_file() and args.step in ("all", "inference"):
        if not args.mc_sim_target_scp or not args.mc_sim_interference_scp:
            summary_datasets["mc_sim"] = {
                "status": "error",
                "reason": "mc_sim_requires_target_and_interference_scps",
                "testset_root": str(testset_root),
            }
            return
        try:
            from tools.prepare_data_tools.generators.se.generate_mc_sim_testset import (
                generate_mc_sim_testset,
            )

            generate_mc_sim_testset(
                config_path=config,
                target_scp=args.mc_sim_target_scp,
                interference_scp=args.mc_sim_interference_scp,
                output_dir=testset_root,
                num_samples=args.num_sim_samples,
                seed=args.seed,
                sir_range_db=tuple(args.mc_sim_sir_range_db),
            )
        except Exception as exc:
            summary_datasets["mc_sim"] = {
                "status": "error",
                "reason": f"mc_sim_generation_failed: {exc}",
                "testset_root": str(testset_root),
            }
            return

    if not manifest_path.is_file():
        summary_datasets["mc_sim"] = {
            "status": "error",
            "reason": "mc_sim_testset_missing",
            "testset_root": str(testset_root),
        }
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("generator_id") != "se.mc_sim_testset"
            or manifest.get("generator_version") != "1.0"
            or manifest.get("domain") != "wsj0-mix-style-two-source-spatial-se"
        ):
            raise ValueError("mc_sim manifest identity/version/domain mismatch")
        effective_config = getattr(args, "effective_config", {})
        configured_sr = (effective_config.get("data") or {}).get("sample_rate")
        if type(configured_sr) is int and manifest.get("sample_rate") != configured_sr:
            raise ValueError("mc_sim sample rate does not match model configuration")
        model_args = (effective_config.get("model") or {}).get("args") or {}
        configured_ref = model_args.get("ref_channel")
        if configured_ref is not None and int(configured_ref) != int(manifest["reference_channel"]):
            raise ValueError("mc_sim reference channel does not match model configuration")
        source_hashes = manifest.get("source_scp_sha256") or {}
        for role, argument in (
            ("target", getattr(args, "mc_sim_target_scp", None)),
            ("interference", getattr(args, "mc_sim_interference_scp", None)),
        ):
            if argument is not None:
                source_path = Path(argument).expanduser().resolve()
                if not source_path.is_file():
                    raise FileNotFoundError(f"mc_sim {role} SCP is missing: {source_path}")
                source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
                if source_digest != source_hashes.get(role):
                    raise ValueError(f"mc_sim {role} SCP differs from the frozen manifest")
        if args.step in ("all", "inference"):
            if int(manifest.get("seed", -1)) != args.seed:
                raise ValueError("mc_sim seed differs from the requested frozen testset")
            if int(manifest.get("num_samples", -1)) != args.num_sim_samples:
                raise ValueError("mc_sim sample count differs from the requested frozen testset")
            if tuple(manifest.get("sir_range_db", ())) != tuple(args.mc_sim_sir_range_db):
                raise ValueError("mc_sim SIR range differs from the requested frozen testset")
        mixture_scp = testset_root / "mixture.scp"
        noisy_scp = testset_root / "noisy.scp"
        clean_scp = testset_root / "clean.scp"
        target_image_scp = testset_root / "target_image.scp"
        scp_files = {
            "mixture": mixture_scp,
            "noisy": noisy_scp,
            "clean": clean_scp,
            "target_image": target_image_scp,
        }
        scp_hashes = manifest.get("scps_sha256") or {}
        for name, path in scp_files.items():
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"mc_sim SCP is missing or not a regular file: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != scp_hashes.get(name):
                raise ValueError(f"mc_sim {name} SCP hash mismatch")
        source_maps = {
            name: read_scp_dict(path) for name, path in scp_files.items()
        }
        expected_uids = set(source_maps["mixture"])
        if not expected_uids or any(set(entries) != expected_uids for entries in source_maps.values()):
            raise ValueError("mc_sim SCP files must contain the same non-empty uid inventory")
        if source_maps["noisy"] != source_maps["mixture"]:
            raise ValueError("mc_sim noisy.scp must alias the canonical mixture.scp inventory")
        if len(expected_uids) != int(manifest.get("num_samples", -1)):
            raise ValueError("mc_sim manifest sample count differs from SCP inventory")
        manifest_items = manifest.get("items")
        if not isinstance(manifest_items, list):
            raise ValueError("mc_sim manifest items must be a list")
        items_by_uid = {item.get("uid"): item for item in manifest_items if isinstance(item, dict)}
        if set(items_by_uid) != expected_uids:
            raise ValueError("mc_sim manifest and SCP uid inventories differ")
        for uid, item in items_by_uid.items():
            payload_hashes = item.get("payload_sha256")
            if not isinstance(payload_hashes, dict):
                raise ValueError(f"mc_sim {uid!r} has no payload SHA-256 inventory")
            for name, folder in (("mixture", "mixture"), ("clean", "clean"), ("target_image", "target_image")):
                path = testset_root / folder / f"{uid}.wav"
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"mc_sim payload is missing or not a regular file: {path}")
                if Path(source_maps[name][uid]).resolve() != path.resolve():
                    raise ValueError(f"mc_sim {name} SCP path does not match manifest inventory")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != payload_hashes.get(name):
                    raise ValueError(f"mc_sim {name} payload hash mismatch for {uid!r}")
    except Exception as exc:
        summary_datasets["mc_sim"] = {
            "status": "error",
            "reason": f"mc_sim_testset_invalid: {exc}",
            "testset_root": str(testset_root),
        }
        return

    _run_se_dataset(
        "mc_sim",
        args,
        config,
        checkpoint,
        eval_root,
        summary_datasets,
        {"mc_sim": {"noisy": noisy_scp, "clean": clean_scp}},
    )
    if args.step not in ("all", "evaluate"):
        return

    try:
        import soundfile as sf
        from evaluate.oracle_beamformer import oracle_mvdr

        effective_config = getattr(args, "effective_config", {})
        stft = effective_config.get("stft") or {}
        n_fft = int(stft.get("n_fft", 512))
        hop_length = int(stft.get("hop_length", n_fft // 2))
        win_length = int(stft.get("win_length", n_fft))
        window = str(stft.get("window") or "hann")
        reference_channel = int(manifest["reference_channel"])
        model_channels = model_args.get("n_channel")
        _, _, model_results_dir = prepare_dataset_paths(eval_root, "mc_sim")
        oracle_root = model_results_dir / "oracle_mvdr"
        enhanced_dir = oracle_root / "enhanced"
        metrics_dir = oracle_root / "metrics"
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        output_entries = []
        for uid in sorted(expected_uids):
            mixture, mixture_sr = sf.read(
                source_maps["mixture"][uid], dtype="float32", always_2d=True
            )
            target_image, target_sr = sf.read(
                source_maps["target_image"][uid], dtype="float32", always_2d=True
            )
            if mixture_sr != target_sr or mixture.shape != target_image.shape:
                raise ValueError(f"mc_sim {uid!r} mixture/target_image shape or SR mismatch")
            if model_channels is not None and int(model_channels) != mixture.shape[1]:
                raise ValueError(
                    f"mc_sim C={mixture.shape[1]} does not match model n_channel={model_channels}"
                )
            if mixture_sr != int(manifest["sample_rate"]):
                raise ValueError(f"mc_sim {uid!r} sample rate differs from manifest")
            if not 0 <= reference_channel < mixture.shape[1]:
                raise ValueError(f"mc_sim reference_channel invalid for {uid!r}")
            estimate = oracle_mvdr(
                mixture.T,
                target_image.T,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                reference_channel=reference_channel,
                window=window,
            )
            estimate = estimate.detach().cpu().numpy().astype("float32", copy=False)
            path = enhanced_dir / f"{uid}.wav"
            sf.write(path, estimate, mixture_sr, subtype="FLOAT")
            output_entries.append((uid, int(mixture_sr), int(estimate.size), str(path.resolve())))
        write_scp_entries(output_entries, enhanced_dir / "inf.scp", four_col=True)
        run_standard_evaluation(
            enhanced_dir=enhanced_dir,
            results_dir=metrics_dir,
            ref_scp=clean_scp,
            audio_quality=args.audio_quality,
            device=args.device,
        )
        dataset_summary = summary_datasets.setdefault("mc_sim", {})
        dataset_summary["oracle_mvdr"] = {
            "status": "done",
            "enhanced_dir": str(enhanced_dir),
            "results_dir": str(metrics_dir),
            "results_file": str(metrics_dir / "RESULTS.txt"),
            "num_samples": len(output_entries),
            "reference_channel": reference_channel,
            "stft": {
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": win_length,
                "window": window,
            },
        }
    except Exception as exc:
        dataset_summary = summary_datasets.setdefault("mc_sim", {})
        dataset_summary["oracle_mvdr"] = {"status": "error", "reason": str(exc)}
        dataset_summary["status"] = "error"
        dataset_summary["reason"] = f"mc_sim_oracle_mvdr_failed: {exc}"


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
    args.effective_config = effective_config
    args.inference_devices = normalize_inference_devices(
        args.inference_devices, fallback=args.device
    )
    mtype = detect_model_type(effective_config, config_path=config)
    datasets = resolve_datasets(
        args.datasets, args.include_aec_sim, args.include_mc_sim
    )

    if mtype != "se":
        logger.warning(f"Config suggests model type='{mtype}', but you launched eval_pipeline_se.py")

    exp_name = resolve_eval_output_name(
        getattr(args, "output_name", None), exp_dir=args.exp_dir, config=config
    )
    eval_root = default_eval_output_root(_PROJECT_ROOT) / "se" / exp_name
    explicit_output_name = getattr(args, "output_name", None) is not None
    if not explicit_output_name:
        eval_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Evaluation root: {eval_root}")
    logger.info(f"Datasets: {datasets} | Step: {args.step}")

    summary = {
        "pipeline": "eval_pipeline_se",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "model_type": mtype,
        "step": args.step,
        "device": args.device,
        "inference_devices": args.inference_devices,
        "output_name": exp_name,
        "effective_config_identity": effective_identity,
        "datasets": {},
    }
    adapter = PipelineTaskAdapter(
        task="se",
        pipeline="eval_pipeline_se",
        step=args.step,
        output_dir=eval_root,
        provenance={
            "config": config,
            "checkpoint": checkpoint,
            "resolved_config_fingerprint": effective_identity,
            "parameters": {
                "step": args.step,
                "model_type": mtype,
                "device": args.device,
                "inference_devices": args.inference_devices,
                "output_name": exp_name,
            },
            "entry": "eval_pipeline_se",
        },
    )
    summary["provenance"] = adapter._provenance()

    def _prepare() -> dict:
        logger.info("Preparing standardized SCPs under %s ...", DEFAULT_TEST_DATA_ROOT)
        scp_paths = prepare_testset_scps(
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
        for ds in datasets:
            try:
                if ds == "voicebank":
                    process_voicebank(
                        args, config, checkpoint, eval_root, summary["datasets"], scp_paths
                    )
                elif ds == "urgent":
                    process_urgent(
                        args, config, checkpoint, eval_root, summary["datasets"], scp_paths
                    )
                elif ds == "dns_challenge":
                    process_dns_challenge(
                        args, config, checkpoint, eval_root, summary["datasets"], scp_paths
                    )
                elif ds == "aec_sim":
                    process_aec_sim(
                        args, config, checkpoint, eval_root, summary["datasets"], scp_paths
                    )
                elif ds == "mc_sim":
                    process_mc_sim(
                        args, config, checkpoint, eval_root, summary["datasets"]
                    )
                else:
                    logger.warning(f"Unknown dataset tag: {ds}, skipping.")
                    summary["datasets"][ds] = {
                        "status": "error",
                        "reason": "unknown_dataset",
                    }
            except Exception as exc:
                logger.exception("[%s] dataset processing failed", ds)
                existing = summary["datasets"].get(ds, {})
                summary["datasets"][ds] = {
                    **existing,
                    "status": "error",
                    "reason": str(exc),
                }
        finalize_pipeline_summary(summary)
        return summary

    def _on_failure(stage: str, exc: Exception) -> dict:
        logger.exception("SE pipeline %s failed", stage)
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
