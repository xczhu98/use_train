"""评估流水线公共工具。

抽取 ``eval_pipeline_se.py`` 和 ``eval_pipeline_aec.py`` 中重复的
数据集处理逻辑，并为 ``evaluate_urgent.py`` 提供对接 ``core.metrics`` 的桥接。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from core.metrics import _resample_np, compute_metrics_batch
from core.metrics.aecmos import compute_aecmos, load_aecmos
from core.utils.inference_artifacts import (
    atomic_write_json,
    atomic_write_text,
    fingerprint_file,
    inference_output_lock,
)
from evaluate.artifact_lifecycle import CURRENT_FILENAME, load_committed_artifact
from evaluate.eval_io import (
    generate_scp_from_dir,
    load_audio,
    read_scp,
    validate_inference_artifacts,
)
from evaluate.process_runner import run_evaluate_urgent
from evaluate.urgent_score import (
    RESULTS_FILENAME,
    SCORE_FILENAME,
    ResultsBindingMismatch,
    bind_normalized_score,
    validate_normalized_score,
)

logger = logging.getLogger("evaluate_common")

RESULT_MANIFEST_FILENAME = "result_manifest.json"
RESULT_MANIFEST_SCHEMA = "evaluation-results-v1"


def _results_identity(results_path: Path) -> dict[str, object]:
    fingerprint = fingerprint_file(results_path)
    return {
        "filename": "RESULTS.txt",
        "size": fingerprint["size"],
        "sha256": fingerprint["sha256"],
    }


def write_normalized_score(results_dir: str | Path, score: dict) -> tuple[dict, Path]:
    """将 normalized score 与同目录 RESULTS.txt 内容身份绑定后原子写入。"""
    results = Path(results_dir).expanduser().resolve()
    results_path = results / "RESULTS.txt"
    bound = bind_normalized_score(score, _results_identity(results_path))
    score_path = results / SCORE_FILENAME
    atomic_write_json(score_path, bound)
    return bound, score_path.resolve()


def load_normalized_score(results_dir: str | Path) -> Optional[tuple[dict, Path]]:
    """读取并重算同目录 RESULTS.txt 的 filename/size/SHA256 binding。"""
    results = Path(results_dir).expanduser().resolve()
    results_path = results / "RESULTS.txt"
    score_path = results / SCORE_FILENAME
    if not results_path.is_file() or not score_path.is_file():
        return None
    try:
        payload = json.loads(score_path.read_text(encoding="utf-8"))
        score = validate_normalized_score(
            payload, expected_results=_results_identity(results_path)
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid URGENT normalized score: {score_path}") from exc
    except ResultsBindingMismatch:
        # RESULTS.txt 已被改写：旧 sidecar 与当前内容不再绑定，视为无 score。
        return None
    return score, score_path


def collect_scenario_normalized_scores(
    eval_root: str | Path, scenarios: list[str]
) -> tuple[dict[str, dict], dict[str, str]]:
    """按 summary 已声明场景收集 score DTO 与其 RESULTS sibling 路径。"""
    root = Path(eval_root).expanduser().resolve()
    scores: dict[str, dict] = {}
    files: dict[str, str] = {}
    for scenario in scenarios:
        loaded = load_normalized_score(root / scenario)
        if loaded is None:
            continue
        score, score_path = loaded
        scores[scenario] = score
        files[scenario] = str(score_path)
    return scores, files


def _declared_results_paths(summary: dict) -> set[Path]:
    results_paths: set[Path] = set()
    for item in summary.get("datasets", {}).values():
        if not isinstance(item, dict) or item.get("status") not in {"done", "partial"}:
            continue
        results_file = item.get("results_file")
        if results_file:
            results_path = Path(str(results_file)).expanduser().resolve()
            if results_path.name != "RESULTS.txt":
                raise ValueError("dataset results_file must name RESULTS.txt")
            results_paths.add(results_path)
        scenario_results = item.get("scenario_results_files")
        if scenario_results is None:
            continue
        scenario_root = item.get("eval_root")
        if not isinstance(scenario_results, dict) or not scenario_root:
            raise ValueError("scenario RESULTS declaration is incomplete")
        root = Path(str(scenario_root)).expanduser().resolve()
        for scenario, results_file in scenario_results.items():
            if (
                not isinstance(scenario, str)
                or Path(scenario).name != scenario
                or "\\" in scenario
                or scenario in {"", ".", ".."}
            ):
                raise ValueError("scenario RESULTS name is invalid")
            expected = (root / scenario / "RESULTS.txt").resolve()
            declared = Path(str(results_file)).expanduser().resolve()
            if declared != expected:
                raise ValueError("scenario RESULTS declaration does not match eval_root sibling")
            results_paths.add(declared)
    return results_paths


def _metric_artifact_paths(summary: dict, output_dir: Path) -> list[Path]:
    """从本轮 summary 声明的 RESULTS siblings 收集并校验指标文件。"""
    results_paths = _declared_results_paths(summary)
    artifacts: set[Path] = set(results_paths)
    for results_file in results_paths:
        per_utterance = results_file.with_name("per_utterance.json")
        if per_utterance.is_file():
            artifacts.add(per_utterance.resolve())

    for item in summary.get("datasets", {}).values():
        if not isinstance(item, dict) or item.get("status") not in {"done", "partial"}:
            continue
        results_file = item.get("results_file")
        score_file = item.get("normalized_score_file")
        score = item.get("normalized_score")
        if (score_file is None) != (score is None):
            raise ValueError("normalized score declaration is incomplete")
        if score_file is not None:
            if not results_file:
                raise ValueError("normalized score declaration is missing RESULTS.txt")
            results_path = Path(str(results_file)).expanduser().resolve()
            expected = results_path.with_name(SCORE_FILENAME)
            declared = Path(str(score_file)).expanduser().resolve()
            loaded = load_normalized_score(results_path.parent)
            if (
                results_path not in results_paths
                or declared != expected
                or loaded is None
                or loaded[1] != declared
                or loaded[0] != score
            ):
                raise ValueError(
                    "normalized score declaration does not match current RESULTS sibling"
                )
            artifacts.add(declared)

        scenario_root = item.get("eval_root")
        scenario_files = item.get("scenario_normalized_score_files")
        scenario_scores = item.get("scenario_normalized_scores")
        if (scenario_files is None) != (scenario_scores is None):
            raise ValueError("scenario normalized score declaration is incomplete")
        if scenario_files is not None:
            if (
                not isinstance(scenario_files, dict)
                or not isinstance(scenario_scores, dict)
                or not scenario_root
                or set(scenario_files) != set(scenario_scores)
            ):
                raise ValueError("scenario normalized score declaration is incomplete")
            root = Path(str(scenario_root)).expanduser().resolve()
            for scenario, score_path in scenario_files.items():
                if (
                    not isinstance(scenario, str)
                    or Path(scenario).name != scenario
                    or "\\" in scenario
                    or scenario in {"", ".", ".."}
                ):
                    raise ValueError("scenario normalized score name is invalid")
                results_path = root / scenario / "RESULTS.txt"
                expected = results_path.with_name(SCORE_FILENAME)
                declared = Path(str(score_path)).expanduser().resolve()
                loaded = load_normalized_score(results_path.parent)
                if (
                    results_path not in results_paths
                    or declared != expected
                    or loaded is None
                    or loaded[1] != declared
                    or loaded[0] != scenario_scores[scenario]
                ):
                    raise ValueError(
                        "scenario normalized score declaration does not match current RESULTS sibling"
                    )
                artifacts.add(declared)
    return sorted(artifacts)


def publish_result_manifest(
    output_dir: str | Path,
    summary: dict,
    *,
    artifact_paths: Optional[list[str | Path]] = None,
) -> Path:
    """发布绑定 run_id 与指标文件内容哈希的结果 manifest。"""
    output = Path(output_dir).expanduser().resolve()
    run_id = summary.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("result manifest requires a non-empty summary run_id")
    declared_artifacts = set(_metric_artifact_paths(summary, output))
    if artifact_paths is None:
        paths = sorted(declared_artifacts)
    else:
        paths = [Path(path).expanduser().resolve() for path in artifact_paths]
        explicit = set(paths)
        declared_scores = {
            path for path in declared_artifacts if path.name == SCORE_FILENAME
        }
        explicit_scores = {path for path in explicit if path.name == SCORE_FILENAME}
        undeclared = explicit_scores - declared_scores
        if undeclared:
            raise ValueError(
                f"URGENT normalized score is not declared by summary: {sorted(undeclared)}"
            )
        omitted = declared_artifacts - explicit
        if omitted:
            missing_scores = {path for path in omitted if path.name == SCORE_FILENAME}
            if missing_scores:
                raise ValueError(
                    "summary normalized score is missing from artifact_paths: "
                    f"{sorted(missing_scores)}"
                )
            raise ValueError(
                "summary metric artifact is missing from artifact_paths: "
                f"{sorted(omitted)}"
            )
        for score_path in explicit_scores:
            sibling_results = score_path.with_name(RESULTS_FILENAME)
            if sibling_results not in explicit:
                raise ValueError(
                    "URGENT normalized score requires sibling RESULTS.txt in artifact_paths: "
                    f"{score_path}"
                )
    files: dict[str, dict] = {}
    for path in sorted(set(paths)):
        try:
            relative = path.relative_to(output).as_posix()
        except ValueError as exc:
            raise ValueError(f"metric artifact is outside evaluation root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"metric artifact missing: {path}")
        fingerprint = fingerprint_file(path)
        files[relative] = {
            "size": fingerprint["size"],
            "sha256": fingerprint["sha256"],
        }
    manifest = {
        "schema": RESULT_MANIFEST_SCHEMA,
        "run_id": run_id,
        "files": files,
    }
    path = output / RESULT_MANIFEST_FILENAME
    atomic_write_json(path, manifest)
    manifest_fingerprint = fingerprint_file(path)
    summary["result_manifest"] = {
        "filename": RESULT_MANIFEST_FILENAME,
        "size": manifest_fingerprint["size"],
        "sha256": manifest_fingerprint["sha256"],
    }
    return path


def validate_result_manifest(
    output_dir: str | Path,
    *,
    expected_run_id: Optional[str] = None,
    expected_summary: Optional[dict] = None,
) -> dict:
    """严格校验 manifest 文件身份及 summary 声明资产的完整包含关系。"""
    output = Path(output_dir).expanduser().resolve()
    path = output / RESULT_MANIFEST_FILENAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"result manifest missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"result manifest is invalid: {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != RESULT_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported result manifest schema: {path}")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"result manifest run_id is invalid: {path}")
    if (output / CURRENT_FILENAME).is_file():
        try:
            committed = load_committed_artifact(output)
        except RuntimeError as exc:
            raise ValueError(f"evaluation artifact commit is invalid: {output}") from exc
        committed_run_id = committed.get("run_id")
        if run_id != committed_run_id:
            raise ValueError(
                "result manifest run_id does not match committed evaluation artifact"
            )
        reference = committed.get("summary", {}).get("result_manifest")
        actual_manifest = fingerprint_file(path)
        if (
            not isinstance(reference, dict)
            or reference.get("filename") != RESULT_MANIFEST_FILENAME
            or reference.get("size") != actual_manifest["size"]
            or reference.get("sha256") != actual_manifest["sha256"]
        ):
            raise ValueError("committed summary result manifest fingerprint mismatch")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError(
            f"result manifest run_id mismatch: expected={expected_run_id!r}, actual={run_id!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"result manifest files is invalid: {path}")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, dict):
            raise ValueError(f"result manifest file entry is invalid: {relative!r}")
        candidate = (output / relative).resolve()
        try:
            candidate.relative_to(output)
        except ValueError as exc:
            raise ValueError(f"result manifest path escapes evaluation root: {relative}") from exc
        try:
            actual = fingerprint_file(candidate)
        except (OSError, ValueError) as exc:
            raise ValueError(f"result manifest artifact missing: {relative}") from exc
        if (
            actual["size"] != expected.get("size")
            or actual["sha256"] != expected.get("sha256")
        ):
            raise ValueError(f"result manifest artifact fingerprint mismatch: {relative}")
    manifest_files = set(files)
    manifest_scores = {
        relative for relative in manifest_files
        if Path(relative).name == SCORE_FILENAME
    }
    for relative in manifest_scores:
        sibling = str(Path(relative).with_name(RESULTS_FILENAME).as_posix())
        if sibling not in manifest_files:
            raise ValueError(
                "URGENT normalized score requires sibling RESULTS.txt in manifest: "
                f"{relative}"
            )
    if expected_summary is not None:
        declared_artifacts = {
            candidate.relative_to(output).as_posix()
            for candidate in _metric_artifact_paths(expected_summary, output)
        }
        missing_artifacts = declared_artifacts - manifest_files
        if missing_artifacts:
            missing_scores = {
                relative for relative in missing_artifacts
                if Path(relative).name == SCORE_FILENAME
            }
            if missing_scores:
                raise ValueError(
                    "result manifest omits summary normalized score: "
                    f"{sorted(missing_scores)}"
                )
            raise ValueError(
                "result manifest omits summary metric artifact: "
                f"{sorted(missing_artifacts)}"
            )
        extra_scores = manifest_scores - {
            relative for relative in declared_artifacts
            if Path(relative).name == SCORE_FILENAME
        }
        if extra_scores:
            raise ValueError(
                "result manifest contains undeclared URGENT normalized score: "
                f"{sorted(extra_scores)}"
            )
    return manifest


def prepare_dataset_paths(eval_root: Path, ds_tag: str) -> tuple[Path, Path, Path]:
    """返回 (ds_eval, enhanced_dir, results_dir)。"""
    ds_eval = eval_root / ds_tag
    enhanced_dir = ds_eval / "enhanced"
    results_dir = ds_eval
    return ds_eval, enhanced_dir, results_dir


def require_scp(scp_paths: dict, ds_tag: str, summary_datasets: dict) -> Optional[dict]:
    """检查 SCP 是否已准备，缺失则记录错误并返回 None。"""
    scp_info = scp_paths.get(ds_tag)
    if scp_info is None:
        logger.error(f"{ds_tag} SCPs not prepared")
        summary_datasets[ds_tag] = {"status": "error", "reason": "scp_missing"}
    return scp_info


def check_enhanced_dir(enhanced_dir: Path, ds_tag: str, summary_datasets: dict) -> bool:
    """检查 enhanced 目录是否存在且包含 wav，否则记录错误并返回 False。"""
    if not enhanced_dir.exists() or not any(enhanced_dir.glob("*.wav")):
        logger.warning("No enhanced audio in %s, cannot evaluate.", enhanced_dir)
        summary_datasets[ds_tag] = {"status": "error", "reason": "no_enhanced"}
        return False
    return True


def count_evaluable_outputs(inf_scp: Path, ref_scp: Optional[Path] = None) -> int:
    """统计评估入口可成功解码且（如需要）可与参考配对的输出。"""
    produced_map = read_scp(inf_scp)
    ref_map = read_scp(ref_scp) if ref_scp is not None else {}
    evaluated = 0
    for uid, output_path in produced_map.items():
        if ref_scp is not None and uid not in ref_map:
            continue
        try:
            output_audio, _ = load_audio(str(output_path))
            if ref_scp is None:
                valid = len(output_audio) > 0
            else:
                ref_audio, _ = load_audio(str(ref_map[uid]))
                valid = min(len(output_audio), len(ref_audio)) > 0
        except Exception as exc:
            logger.warning("评估覆盖率解码失败 uid=%s: %s", uid, exc)
            continue
        if valid:
            evaluated += 1
    return evaluated


def load_inference_coverage(
    enhanced_dir: Path,
    ref_scp: Optional[Path] = None,
    *,
    expected_scp: Optional[Path] = None,
    config: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    evaluation_performed: bool = True,
) -> dict:
    """严格校验标准 manifest/inf.scp，并返回覆盖率计数。"""
    manifest_path = enhanced_dir / "run_manifest.json"
    inf_scp = enhanced_dir / "inf.scp"
    produced_map = read_scp(inf_scp)
    produced_uids = set(produced_map)
    missing_files = sorted(uid for uid, path in produced_map.items() if not path.is_file())
    if missing_files:
        raise ValueError(f"inf.scp references missing output files: {missing_files[:5]}")

    source_uids = set(read_scp(expected_scp)) if expected_scp is not None else None
    if (config is not None or checkpoint is not None) and not manifest_path.is_file():
        raise ValueError(
            "run_manifest.json is required when config/checkpoint provenance is provided"
        )
    if manifest_path.exists():
        manifest, _ = validate_inference_artifacts(
            enhanced_dir, config=config, checkpoint=checkpoint
        )
        expected = manifest["expected_uids"]
        failed = manifest["failed_uids"]
        if source_uids is not None and set(expected) != source_uids:
            raise ValueError("manifest expected_uids do not match source SCP uids")
    else:
        expected = sorted(source_uids if source_uids is not None else produced_uids)
        unexpected = produced_uids - set(expected)
        if unexpected:
            raise ValueError(f"inf.scp contains unexpected uids: {sorted(unexpected)[:5]}")
        failed = sorted(set(expected) - produced_uids)

    evaluated = (
        count_evaluable_outputs(inf_scp, ref_scp) if evaluation_performed else 0
    )
    return {
        "expected": len(expected),
        "produced": len(produced_uids),
        "failed": len(failed),
        "evaluated": evaluated,
    }


def metric_coverage_status(metric_coverage: dict[str, dict]) -> str:
    """汇总已请求指标覆盖率；未请求的可选指标不影响状态。"""
    if not metric_coverage:
        return "unknown"
    active_items = [
        item
        for item in metric_coverage.values()
        if not (
            item.get("status") == "skipped"
            and item.get("reason") == "not_requested"
        )
    ]
    if not active_items:
        return "done"
    statuses = {item.get("status", "unknown") for item in active_items}
    if "unknown" in statuses:
        return "unknown"
    if statuses == {"done"}:
        return "done"
    return "partial"


def coverage_status(
    coverage: dict,
    allow_partial: bool,
    *,
    require_evaluated: bool = True,
    metric_coverage: Optional[dict[str, dict]] = None,
) -> str:
    inference_complete = (
        coverage["expected"] > 0
        and coverage["failed"] == 0
        and coverage["produced"] == coverage["expected"]
    )
    if (
        require_evaluated
        and inference_complete
        and metric_coverage is not None
        and metric_coverage_status(metric_coverage) != "done"
    ):
        return "partial"
    evaluation_ok = not require_evaluated or coverage["evaluated"] == coverage["produced"]
    if inference_complete and evaluation_ok:
        return "done"
    partial_evaluation_ok = not require_evaluated or coverage["evaluated"] > 0
    if allow_partial and coverage["produced"] > 0 and partial_evaluation_ok:
        return "partial"
    return "error"


def _metric_coverage_entry(eligible: int, computed: Optional[int]) -> dict:
    if computed is None:
        return {
            "eligible": eligible,
            "attempted": None,
            "computed": None,
            "failed": None,
            "coverage": None,
            "status": "unknown",
        }
    computed = max(0, min(int(computed), eligible))
    status = "done" if eligible > 0 and computed == eligible else "partial"
    return {
        "eligible": eligible,
        "attempted": eligible,
        "computed": computed,
        "failed": eligible - computed,
        "coverage": computed / eligible if eligible else None,
        "status": status,
    }


def build_metric_coverage_from_counts(
    eligible: int, metric_counts: dict[str, Optional[int]]
) -> dict[str, dict]:
    """由评估器真实成功计数构建逐指标覆盖率。"""
    return {
        metric: _metric_coverage_entry(eligible, count)
        for metric, count in metric_counts.items()
    }


def apply_metric_coverage(coverage: dict, metric_coverage: dict[str, dict]) -> dict:
    """保持四键 coverage，并将 evaluated 收敛到最保守的真实指标数。"""
    updated = dict(coverage)
    if not metric_coverage:
        return updated
    computed = [item.get("computed") for item in metric_coverage.values()]
    updated["evaluated"] = (
        min(int(value) for value in computed)
        if computed and all(value is not None for value in computed)
        else 0
    )
    return updated


def requested_standard_metrics(
    *, ref_scp: Optional[Path], audio_quality: bool, downstream: bool = False
) -> list[str]:
    """返回本次 ``evaluate_urgent`` 调用要求产出的指标集合。"""
    metrics = ["PESQ", "STOI", "ESTOI", "SI-SDR"] if ref_scp else []
    if audio_quality:
        metrics.extend(["DNSMOS", "NISQA", "UTMOS", "SCOREQ"])
    if downstream:
        metrics.extend([
            "WER", "CER", "SpeakerSimilarity", "EmotionSimilarity",
            "PhonemeSimilarity", "SpeechBERTScore", "LAcc",
        ])
    return metrics


def load_metric_coverage(
    results_dir: Path,
    requested_metrics: list[str],
    *,
    eligible: int,
) -> dict[str, dict]:
    """从 RESULTS.txt/per_utterance.json 提取逐指标真实成功数。"""
    results_path = results_dir / "RESULTS.txt"
    result_metrics: set[str] = set()
    counts: dict[str, int] = {}
    try:
        for raw_line in results_path.read_text(encoding="utf-8").splitlines():
            if ":" not in raw_line:
                continue
            key, value = (part.strip() for part in raw_line.split(":", 1))
            if key.startswith("_count_"):
                try:
                    counts[key[len("_count_"):]] = int(float(value))
                except ValueError:
                    logger.warning("无效指标计数行: %s", raw_line)
            elif key:
                result_metrics.add(key)
    except (OSError, UnicodeError):
        return build_metric_coverage_from_counts(
            eligible, {metric: None for metric in requested_metrics}
        )

    record_counts: dict[str, int] = {}
    try:
        payload = json.loads(
            (results_dir / "per_utterance.json").read_text(encoding="utf-8")
        )
        records = payload.get("records", {})
        if isinstance(records, dict):
            for record in records.values():
                if isinstance(record, dict):
                    for metric in record:
                        record_counts[metric] = record_counts.get(metric, 0) + 1
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
        pass

    metric_counts: dict[str, Optional[int]] = {}
    for metric in requested_metrics:
        if metric in counts:
            metric_counts[metric] = counts[metric]
        elif metric in result_metrics and metric in record_counts:
            metric_counts[metric] = record_counts[metric]
        else:
            metric_counts[metric] = None
    return build_metric_coverage_from_counts(eligible, metric_counts)


def aggregate_coverages(items: list[dict]) -> dict:
    """按场景累加 expected/produced/failed/evaluated。"""
    return {
        field: sum(item[field] for item in items)
        for field in ("expected", "produced", "failed", "evaluated")
    }


def aggregate_metric_coverages(items: list[dict[str, dict]]) -> dict[str, dict]:
    """按指标跨场景累加覆盖率，任一场景未知则聚合结果保持 unknown。"""
    metrics = sorted({metric for item in items for metric in item})
    aggregated: dict[str, dict] = {}
    for metric in metrics:
        entries = [item[metric] for item in items if metric in item]
        eligible = sum(int(entry.get("eligible", 0)) for entry in entries)
        if any(entry.get("computed") is None for entry in entries):
            aggregated[metric] = _metric_coverage_entry(eligible, None)
        else:
            aggregated[metric] = _metric_coverage_entry(
                eligible, sum(int(entry["computed"]) for entry in entries)
            )
    return aggregated


def finalize_pipeline_summary(summary: dict) -> None:
    """根据 dataset 状态与 coverage 原地补齐根级 status/coverage。"""
    datasets = list(summary.get("datasets", {}).values())
    statuses = [item.get("status", "error") for item in datasets]
    if not statuses or "error" in statuses:
        summary["status"] = "error"
    elif "partial" in statuses:
        summary["status"] = "partial"
    else:
        summary["status"] = "done"
    coverages = [item["coverage"] for item in datasets if "coverage" in item]
    summary["coverage"] = aggregate_coverages(coverages) if coverages else {
        "expected": 0, "produced": 0, "failed": 0, "evaluated": 0
    }


def ensure_inf_scp(enhanced_dir: Path) -> Path:
    """优先校验 run_manifest.json；无 manifest 时才从 wav 生成 inf.scp。"""
    inf_scp = enhanced_dir / "inf.scp"
    manifest = enhanced_dir / "run_manifest.json"
    if manifest.exists():
        if not inf_scp.exists():
            raise FileNotFoundError(f"inference manifest exists but inf.scp is missing: {enhanced_dir}")
        load_inference_coverage(enhanced_dir, evaluation_performed=False)
    elif not inf_scp.exists() or Path(f"{inf_scp}.provenance.json").exists():
        generate_scp_from_dir(enhanced_dir, pattern="*.wav", scp_path=inf_scp, recursive=False)
    else:
        read_scp(inf_scp)
    return inf_scp


def build_error_summary(
    reason: str,
    enhanced_dir: Path,
    *,
    expected_scp: Optional[Path] = None,
    config: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    coverage: Optional[dict] = None,
) -> dict:
    """构建 error 摘要，并尽可能保留可信的推理覆盖率。"""
    summary = {"status": "error", "reason": reason, "enhanced_dir": str(enhanced_dir)}
    if coverage is None:
        try:
            ensure_inf_scp(enhanced_dir)
            coverage = load_inference_coverage(
                enhanced_dir,
                expected_scp=expected_scp,
                config=config,
                checkpoint=checkpoint,
                evaluation_performed=False,
            )
        except Exception as exc:
            logger.warning("无法恢复推理 coverage: %s", exc)
            coverage = None
    if coverage is not None:
        summary["coverage"] = coverage
    return summary


def warn_blind_no_reference(ds_tag: str) -> None:
    """提醒盲测集无参考，需要 --audio_quality 启用 MOS 指标。"""
    logger.warning(
        f"[{ds_tag}] Blind testset has no clean reference. "
        "Add --audio_quality to enable non-intrusive MOS metrics."
    )


def build_done_summary(
    enhanced_dir: Path,
    results_dir: Path,
    extras: Optional[dict] = None,
    *,
    coverage: Optional[dict] = None,
    allow_partial: bool = False,
    metric_coverage: Optional[dict[str, dict]] = None,
) -> dict:
    """构建带 inference/evaluation coverage 的摘要字典。"""
    summary = {
        "status": coverage_status(
            coverage, allow_partial, metric_coverage=metric_coverage
        ) if coverage else "done",
        "enhanced_dir": str(enhanced_dir),
        "results_file": str(results_dir / "RESULTS.txt"),
    }
    if coverage:
        summary["coverage"] = coverage
    if metric_coverage is not None:
        summary["metric_coverage"] = metric_coverage
        summary["metric_coverage_status"] = metric_coverage_status(metric_coverage)
    loaded_score = load_normalized_score(results_dir)
    if loaded_score is not None:
        normalized_score, score_path = loaded_score
        summary["normalized_score"] = normalized_score
        summary["normalized_score_file"] = str(score_path)
    if extras:
        summary.update(extras)
    return summary


def run_standard_evaluation(
    enhanced_dir: Path,
    results_dir: Path,
    ref_scp: Optional[Path],
    audio_quality: bool,
    device: str,
    downstream: bool = False,
) -> None:
    """生成 inf.scp 并调用 evaluate_urgent 计算指标。"""
    inf_scp = ensure_inf_scp(enhanced_dir)
    run_evaluate_urgent(
        inf_scp=inf_scp,
        output_dir=results_dir,
        ref_scp=ref_scp,
        audio_quality=audio_quality,
        downstream=downstream,
        device=device,
    )


def evaluate_audio_quality(
    inf_scp: Path,
    ref_scp: Optional[Path] = None,
    mos_models: Optional[dict[str, object]] = None,
    device: str = "cpu",
    parallel_intrusive: bool = True,
) -> dict[str, float]:
    """从 SCP 加载 (ref, enh, fs) 并调用 ``core.metrics.compute_metrics_batch``。

    Returns:
        {metric: mean_value} 的平均值字典（不含 _count_ 键）。
    """
    inf_map = read_scp(inf_scp)
    ref_map = read_scp(ref_scp) if ref_scp else {}

    pairs: list[tuple[np.ndarray, np.ndarray, int]] = []
    skipped_missing_ref = 0
    for uid, inf_path in inf_map.items():
        if ref_scp and uid not in ref_map:
            # 缺失参考时直接跳过，避免零填充静默拉低指标且避免无效 I/O。
            skipped_missing_ref += 1
            continue
        inf_audio, inf_fs = load_audio(str(inf_path))
        if uid in ref_map:
            ref_audio, ref_fs = load_audio(str(ref_map[uid]))
            if inf_fs != ref_fs:
                inf_audio = _resample_np(inf_audio, inf_fs, ref_fs)
                inf_fs = ref_fs
        else:
            ref_audio = np.zeros_like(inf_audio)

        minlen = min(len(ref_audio), len(inf_audio))
        if minlen == 0:
            continue
        pairs.append((ref_audio[:minlen], inf_audio[:minlen], inf_fs))

    if skipped_missing_ref:
        logger.warning(
            "evaluate_audio_quality: %d uid(s) missing from ref_scp, skipped (no zero-fill)",
            skipped_missing_ref,
        )

    metrics = ["PESQ", "STOI", "ESTOI", "SI-SDR"] if ref_scp else []
    raw = compute_metrics_batch(
        pairs=pairs,
        metrics=metrics,
        mos_models=mos_models or {},
        device=device,
        parallel_intrusive=parallel_intrusive,
    )

    means: dict[str, float] = {}
    for key, total in raw.items():
        if key.startswith("_count_"):
            continue
        count = raw.get(f"_count_{key}", 0.0)
        if count > 0:
            means[key] = total / count
    return means


# ── AECMOS helper ──
_aecmos_estimator = None


def _get_aecmos_estimator():
    global _aecmos_estimator
    if _aecmos_estimator is None:
        _aecmos_estimator = load_aecmos()
    return _aecmos_estimator


def run_aecmos_for_scenario(
    mic_scp: str | Path,
    lpb_scp: str | Path | None,
    enh_scp: str | Path,
    talk_type: str,
) -> dict[str, object] | None:
    """计算一个 scenario 的官方有效 AECMOS 指标。

    Args:
        mic_scp: mic SCP 路径
        lpb_scp: loopback（farend）SCP 路径（SE 测试集传 None，内部用零填充）
        enh_scp: enhanced SCP 路径
        talk_type: "st" | "nst" | "dt"

    Returns:
        ``st`` 仅含 AECMOS_ECHO/EMOS，``nst`` 仅含 AECMOS_DEG/DMOS，
        ``dt`` 同时含两组指标；另含 n_samples、n_failed 和 echo_valid。
        echo_valid 保留兼容语义，表示 loopback 是否来自真实 SCP。
        三路音频在计算前统一重采样到 AECMOS 模型采样率。
    """
    if talk_type not in {"st", "nst", "dt"}:
        raise ValueError(f"unsupported AECMOS talk_type: {talk_type!r}")
    if talk_type in {"st", "dt"} and lpb_scp is None:
        raise ValueError(f"AECMOS talk_type={talk_type!r} requires loopback SCP")

    mic_map = read_scp(mic_scp)
    enh_map = read_scp(enh_scp)
    lpb_map = read_scp(lpb_scp) if lpb_scp else {}
    if lpb_scp is not None:
        uid_sets = {
            "mic": set(mic_map),
            "lpb": set(lpb_map),
            "enh": set(enh_map),
        }
        if len({frozenset(uids) for uids in uid_sets.values()}) != 1:
            details = ", ".join(
                f"{name}={len(uids)}" for name, uids in uid_sets.items()
            )
            raise ValueError(f"AECMOS UID mismatch across mic/lpb/enh SCPs: {details}")
        common = sorted(uid_sets["mic"])
    else:
        common = sorted(set(mic_map) & set(enh_map))
    if not common:
        return None

    try:
        estimator = _get_aecmos_estimator()
        target_sr = int(estimator.sampling_rate)
    except Exception as exc:
        logger.warning("[AECMOS] skipped because estimator is unavailable: %s", exc)
        return {
            "skipped": True,
            "skip_reason": f"{type(exc).__name__}: {exc}",
            "n_samples": 0,
            "n_failed": len(common),
            "echo_valid": lpb_scp is not None,
        }
    if target_sr <= 0:
        raise ValueError(f"AECMOS estimator sampling_rate must be positive: {target_sr}")
    echo_scores: list[float] = []
    deg_scores: list[float] = []
    failed_audio: list[tuple[str, Exception]] = []
    for uid in common:
        try:
            mic_audio, mic_sr = load_audio(str(mic_map[uid]))
            enh_audio, enh_sr = load_audio(str(enh_map[uid]))
            if uid in lpb_map:
                lpb_audio, lpb_sr = load_audio(str(lpb_map[uid]))
            else:
                lpb_audio = np.zeros(len(mic_audio), dtype=np.float32)
                lpb_sr = mic_sr
        except (OSError, RuntimeError) as exc:
            failed_audio.append((uid, exc))
            logger.warning("[AECMOS] audio load failed for %s: %s", uid, exc)
            continue

        try:
            if mic_sr != target_sr:
                mic_audio = _resample_np(mic_audio, mic_sr, target_sr)
            if enh_sr != target_sr:
                enh_audio = _resample_np(enh_audio, enh_sr, target_sr)
            if lpb_sr != target_sr:
                lpb_audio = _resample_np(lpb_audio, lpb_sr, target_sr)
            minlen = min(len(lpb_audio), len(mic_audio), len(enh_audio))
            if minlen <= 0:
                raise ValueError("empty audio after stream alignment")
            scores = compute_aecmos(
                estimator,
                lpb_audio[:minlen],
                mic_audio[:minlen],
                enh_audio[:minlen],
                talk_type=talk_type,
            )
            if talk_type in {"st", "dt"}:
                echo_scores.append(float(scores["AECMOS_ECHO"]))
            if talk_type in {"nst", "dt"}:
                deg_scores.append(float(scores["AECMOS_DEG"]))
        except Exception as exc:
            failed_audio.append((uid, exc))
            logger.warning("[AECMOS] sample failed for %s: %s", uid, exc)

    n_samples = len(echo_scores) if echo_scores else len(deg_scores)
    if n_samples == 0:
        reason = str(failed_audio[-1][1]) if failed_audio else "no valid scores"
        logger.warning("[AECMOS] skipped: all %d samples failed (%s)", len(common), reason)
        return {
            "skipped": True,
            "skip_reason": reason,
            "n_samples": 0,
            "n_failed": len(failed_audio),
            "echo_valid": lpb_scp is not None,
        }
    if failed_audio:
        logger.warning(
            "[AECMOS] completed with %d/%d audio sample failures",
            len(failed_audio),
            len(common),
        )

    result: dict[str, float | int | bool] = {
        "n_samples": n_samples,
        "n_failed": len(failed_audio),
        "echo_valid": lpb_scp is not None,
    }
    if echo_scores:
        echo_mean = float(np.mean(echo_scores))
        result["AECMOS_ECHO"] = echo_mean
        result["EMOS"] = echo_mean
    if deg_scores:
        deg_mean = float(np.mean(deg_scores))
        result["AECMOS_DEG"] = deg_mean
        result["DMOS"] = deg_mean
    return result


def append_extra_metrics_to_results(
    output_dir: str | Path,
    metrics: dict[str, float],
    *,
    replace: bool = False,
) -> None:
    """原子写入额外指标；replace 用于当前 run 只有额外指标的场景。

    RESULTS.txt 与 normalized score sidecar 的 read-modify-write 必须在
    ``output_dir`` 锁内完成，避免并发 writer 互相覆盖。
    """
    with inference_output_lock(output_dir):
        _append_extra_metrics_unlocked(output_dir, metrics, replace=replace)


def _append_extra_metrics_unlocked(
    output_dir: str | Path,
    metrics: dict[str, float],
    *,
    replace: bool = False,
) -> None:
    """在已持有 ``output_dir`` 锁的前提下写入额外指标。"""
    current_score = None if replace else load_normalized_score(output_dir)
    results_path = Path(output_dir) / "RESULTS.txt"
    metric_lines = [f"{key}: {val:.4f}" for key, val in metrics.items()]
    retained: list[str] = []
    if not replace:
        try:
            existing = results_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            existing = []
        metric_keys = set(metrics)
        retained = [
            line for line in existing
            if not any(line.startswith(f"{key}:") for key in metric_keys)
        ]
        while retained and not retained[-1].strip():
            retained.pop()
    lines = retained + ([""] if retained and metric_lines else []) + metric_lines
    atomic_write_text(results_path, "\n".join(lines) + ("\n" if lines else ""))
    if current_score is not None:
        write_normalized_score(output_dir, current_score[0])
    logger.info("Updated extra metrics in %s", results_path)
