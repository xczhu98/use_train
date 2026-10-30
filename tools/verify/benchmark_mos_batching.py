#!/usr/bin/env python3
"""用真实本地 UTMOS/SCOREQ 验证有界混长 batch 的 MOS 一致性。"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "tools" / "audio_filter_pipeline" / "config" / "spk_filter.yaml"
_TARGET_SAMPLE_RATE = 16_000
_SCOREQ_MAX_SAMPLES = 160_000
_SCOREQ_PADDING_MULTIPLE = 320
_MAX_ITEMS = 8
_MAX_BATCH_SIZE = 8
_MAX_SOURCE_SECONDS = 12.0
_DEFAULT_SOURCE_SECONDS = 10.25
_DEFAULT_TOLERANCE = 1e-5
_MODEL_SCORE_KEYS = {"utmos": "utmos_score", "scoreq": "scoreq_mos"}


@dataclass(frozen=True)
class AudioItem:
    """已经按源采样率上限截断的单声道内存音频。"""

    path: str
    waveform: np.ndarray
    sample_rate: int
    original_samples: int
    source_capped: bool

    @property
    def submitted_samples(self) -> int:
        return int(self.waveform.size)

    @property
    def submitted_seconds(self) -> float:
        return self.submitted_samples / self.sample_rate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        choices=("utmos", "scoreq"),
        help="待验证模型；可重复，默认同时验证 utmos 与 scoreq",
    )
    parser.add_argument(
        "--audio",
        action="append",
        type=Path,
        default=[],
        help="显式输入音频路径；可重复，不会扫描目录或数据集",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-items", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-source-seconds", type=float, default=_DEFAULT_SOURCE_SECONDS)
    parser.add_argument("--score-tolerance", type=float, default=_DEFAULT_TOLERANCE)
    parser.add_argument(
        "--require-scoreq-truncation",
        action="store_true",
        help="要求至少一条 SCOREQ 输入在重采样后触发 10 秒截断边界",
    )
    parser.add_argument("--output", type=Path, help="可选 JSON 输出路径；stdout 始终输出 JSON")
    return parser.parse_args(argv)


def selected_models(raw: Sequence[str] | None) -> tuple[str, ...]:
    models = tuple(raw) if raw else ("utmos", "scoreq")
    if not models or any(model not in _MODEL_SCORE_KEYS for model in models):
        raise ValueError(f"unsupported MOS model selection: {models!r}")
    if len(set(models)) != len(models):
        raise ValueError("--model must not contain duplicates")
    return models


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    models = selected_models(args.model)
    if not args.audio:
        raise ValueError("at least one --audio is required")
    if not 1 <= args.max_items <= _MAX_ITEMS:
        raise ValueError(f"max-items must be in [1, {_MAX_ITEMS}]")
    if not 1 <= args.batch_size <= _MAX_BATCH_SIZE:
        raise ValueError(f"batch-size must be in [1, {_MAX_BATCH_SIZE}]")
    if not math.isfinite(args.max_source_seconds) or not 0 < args.max_source_seconds <= _MAX_SOURCE_SECONDS:
        raise ValueError(
            f"max-source-seconds must be finite and in (0, {_MAX_SOURCE_SECONDS}]"
        )
    if not math.isfinite(args.score_tolerance) or args.score_tolerance < 0:
        raise ValueError("score-tolerance must be finite and non-negative")
    if args.require_scoreq_truncation and "scoreq" not in models:
        raise ValueError("--require-scoreq-truncation requires --model scoreq")
    return models


def _first_channel_float32(audio: Any) -> np.ndarray:
    value = np.asarray(audio)
    if value.ndim == 1:
        mono = value
    elif value.ndim == 2:
        mono = value[:, 0]
    else:
        raise ValueError(f"audio must be 1D or time-first 2D, got {value.shape}")
    if mono.size == 0:
        raise ValueError("audio must contain at least one sample")
    if not np.all(np.isfinite(mono)):
        raise ValueError("audio must contain only finite samples")
    return np.ascontiguousarray(mono, dtype=np.float32)


def load_audio(path: Path, *, max_source_seconds: float) -> AudioItem:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("benchmark requires the optional soundfile package") from exc
    if not path.is_file():
        raise FileNotFoundError(f"audio file does not exist: {path}")
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"invalid audio sample rate for {path}: {sample_rate!r}")
    waveform = _first_channel_float32(audio)
    original_samples = int(waveform.size)
    limit = int(math.floor(max_source_seconds * sample_rate))
    if limit < 1:
        raise ValueError("max-source-seconds leaves no source samples")
    capped = original_samples > limit
    if capped:
        waveform = np.ascontiguousarray(waveform[:limit], dtype=np.float32)
    return AudioItem(
        path=str(path),
        waveform=waveform,
        sample_rate=sample_rate,
        original_samples=original_samples,
        source_capped=capped,
    )


def load_audio_items(
    paths: Sequence[Path], *, max_items: int, max_source_seconds: float
) -> list[AudioItem]:
    return [
        load_audio(path, max_source_seconds=max_source_seconds)
        for path in paths[:max_items]
    ]


def _resample_with_torchaudio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == _TARGET_SAMPLE_RATE:
        return audio
    import torch
    import torchaudio

    tensor = torch.from_numpy(audio).float().unsqueeze(0)
    return np.ascontiguousarray(
        torchaudio.functional.resample(tensor, sample_rate, _TARGET_SAMPLE_RATE)
        .numpy()
        .squeeze(0),
        dtype=np.float32,
    )


def utmos_preprocessed_samples(item: AudioItem) -> int:
    """镜像 UTMOSEstimator._load_waveform 的 16 kHz 与最短一秒规则。"""
    audio = _resample_with_torchaudio(item.waveform, item.sample_rate)
    return max(int(audio.size), _TARGET_SAMPLE_RATE)


def _resample_for_scoreq(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """镜像 _SCOREQScorer 单条与 batch 共用的 torchaudio 重采样路径。"""
    return _resample_with_torchaudio(audio, sample_rate)


def scoreq_preprocessed_samples(item: AudioItem) -> tuple[int, int, bool]:
    """返回 SCOREQ 的重采样长度、320 对齐长度及是否触发 10 秒 cap。"""
    audio = _resample_for_scoreq(item.waveform, item.sample_rate)
    post_resample = int(audio.size)
    capped = post_resample > _SCOREQ_MAX_SAMPLES
    usable = min(post_resample, _SCOREQ_MAX_SAMPLES)
    padded = math.ceil(usable / _SCOREQ_PADDING_MULTIPLE) * _SCOREQ_PADDING_MULTIPLE
    return post_resample, padded, capped


def preprocessing_report(model: str, items: Sequence[AudioItem]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        record: dict[str, Any] = {
            "index": index,
            "path": item.path,
            "sample_rate_hz": item.sample_rate,
            "original_samples": item.original_samples,
            "submitted_samples": item.submitted_samples,
            "submitted_seconds": item.submitted_seconds,
            "source_capped": item.source_capped,
        }
        if model == "utmos":
            record["model_preprocessed_samples"] = utmos_preprocessed_samples(item)
        elif model == "scoreq":
            post_resample, padded, ten_second_capped = scoreq_preprocessed_samples(item)
            record.update(
                post_resample_samples=post_resample,
                model_preprocessed_samples=padded,
                ten_second_capped=ten_second_capped,
            )
        else:
            raise ValueError(f"unsupported model: {model}")
        records.append(record)
    return records


def batch_histogram(records: Iterable[Mapping[str, Any]]) -> list[dict[str, int]]:
    counts = Counter(int(record["model_preprocessed_samples"]) for record in records)
    return [
        {"preprocessed_samples": length, "count": count}
        for length, count in sorted(counts.items())
    ]


def derived_expected_chunks(
    model: str, records: Sequence[Mapping[str, Any]], *, batch_size: int, use_gpu: bool
) -> int:
    """仅按公开 batching 规则推导 chunk 上限，不能代表实际 provider 调用数。"""
    counts = Counter(int(record["model_preprocessed_samples"]) for record in records)
    total = 0
    for length, count in counts.items():
        if model == "utmos":
            safe_count = max(1, (_TARGET_SAMPLE_RATE * 300) // max(length, 1))
            implementation_cap = 16 if use_gpu else 4
        elif model == "scoreq":
            safe_count = max(1, 1_000_000 // max(length, 1))
            implementation_cap = safe_count
        else:
            raise ValueError(f"unsupported model: {model}")
        per_chunk = min(batch_size, implementation_cap, safe_count)
        total += math.ceil(count / per_chunk)
    return total


def _score_from_result(model: str, result: Any) -> float:
    if not isinstance(result, Mapping):
        raise TypeError(f"{model} result must be a mapping")
    key = _MODEL_SCORE_KEYS[model]
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{model} result lacks numeric {key}")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{model} returned a non-finite {key}")
    if result.get("fallback") is True:
        raise RuntimeError(f"{model} returned fallback output despite fail-closed loading")
    return score


def _compute_in_bounded_batches(estimator: Any, audio_data: Sequence[tuple[np.ndarray, int]], batch_size: int) -> list[Any]:
    results: list[Any] = []
    for start in range(0, len(audio_data), batch_size):
        chunk = list(audio_data[start : start + batch_size])
        output = estimator.compute_batch(chunk)
        if isinstance(output, (str, bytes)) or not isinstance(output, Sequence):
            raise TypeError("compute_batch must return a sequence")
        if len(output) != len(chunk):
            raise ValueError("compute_batch result count does not match submitted batch")
        results.extend(output)
    return results


def compare_scores(
    model: str, single_results: Sequence[Any], batch_results: Sequence[Any], *, tolerance: float
) -> dict[str, Any]:
    if len(single_results) != len(batch_results):
        raise ValueError("single and batch result counts do not match")
    entries = []
    for index, (single_result, batch_result) in enumerate(zip(single_results, batch_results)):
        single_score = _score_from_result(model, single_result)
        batch_score = _score_from_result(model, batch_result)
        error = abs(single_score - batch_score)
        entries.append(
            {
                "index": index,
                "single_score": single_score,
                "batch_score": batch_score,
                "absolute_error": error,
                "within_tolerance": error <= tolerance,
            }
        )
    errors = [entry["absolute_error"] for entry in entries]
    return {
        "items": entries,
        "max_absolute_error": max(errors) if errors else None,
        "mean_absolute_error": float(np.mean(errors)) if errors else None,
        "all_within_tolerance": bool(entries and all(entry["within_tolerance"] for entry in entries)),
    }


def _session_telemetry(model: str, estimator: Any, before: Mapping[str, Any]) -> dict[str, Any]:
    if model != "scoreq":
        return {
            "actual_onnx_run_count": None,
            "actual_session_reset_count": None,
            "actual_session_build_count": None,
            "unavailable_reason": "UTMOS exposes no ONNX session telemetry",
        }
    scorer = getattr(estimator, "_scoreq", None)
    session = getattr(scorer, "_sess", None)
    before_session = before.get("session")
    return {
        "observed_runs_since_reset_before": before.get("runs_since_reset"),
        "observed_runs_since_reset_after": getattr(scorer, "_runs_since_reset", None),
        "observed_session_identity_changed": (
            None if before_session is None or session is None else session is not before_session
        ),
        "actual_onnx_run_count": None,
        "actual_session_reset_count": None,
        "actual_session_build_count": None,
        "unavailable_reason": (
            "the formal SCOREQ estimator does not expose per-session run, reset, or build counters; "
            "runs_since_reset is reported separately only as an observed scorer field"
        ),
    }


def _telemetry_before(model: str, estimator: Any) -> dict[str, Any]:
    if model != "scoreq":
        return {}
    scorer = getattr(estimator, "_scoreq", None)
    return {
        "runs_since_reset": getattr(scorer, "_runs_since_reset", None),
        "session": getattr(scorer, "_sess", None),
    }


def _resolve_repo_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty path in {CONFIG_PATH}")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def verified_assets(models: Sequence[str], *, use_gpu: bool) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("benchmark requires PyYAML to read canonical MOS asset config") from exc
    from tools.audio_filter_pipeline.processors.stages_v3.mos_asset_preflight import verify_mos_assets
    from tools.audio_filter_pipeline.processors.stages_v3.mos_execution import (
        MosAssetSpec,
        MosExecutionPlan,
    )

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{CONFIG_PATH} must be a mapping")
    assets = []
    for model in models:
        if model == "scoreq":
            assets.append(MosAssetSpec("scoreq", {"model_dir": _resolve_repo_path(raw.get("scoreq_model_dir"), "scoreq_model_dir")}))
        elif model == "utmos":
            assets.append(
                MosAssetSpec(
                    "utmos",
                    {"model_path": _resolve_repo_path(raw.get("utmos_model_path"), "utmos_model_path")},
                    {
                        "model_version": raw.get("utmos_model_version"),
                        "model_sha256": raw.get("utmos_model_sha256"),
                    },
                )
            )
        else:
            raise ValueError(f"unsupported model: {model}")
    if use_gpu:
        gpu_ids = raw.get("step3_gpu_ids")
        if not isinstance(gpu_ids, list):
            raise ValueError("step3_gpu_ids must be a list for --device cuda")
        topology: Mapping[str, Any] = {
            "kind": "gpu_process_pool",
            "device": "cuda",
            "gpu_ids": gpu_ids,
            "worker_idle_timeout_seconds": 900.0,
            "worker_hard_timeout_seconds": None,
            "worker_stderr_tail_bytes": 4096,
            "cpu_fallback": False,
        }
    else:
        topology = {"kind": "single_process", "device": "cpu"}
    return verify_mos_assets(MosExecutionPlan(tuple(models), tuple(assets), topology))


def _option_binding(model: str, options: Mapping[str, Any]):
    from tools.audio_filter_pipeline.processors.mos import bind_scoreq_options, bind_utmos_options

    if model == "scoreq":
        return bind_scoreq_options(options)
    if model == "utmos":
        return bind_utmos_options(options)
    raise ValueError(f"unsupported model: {model}")


def run_model(
    model: str,
    items: Sequence[AudioItem],
    *,
    batch_size: int,
    use_gpu: bool,
    tolerance: float,
    options: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    from tools.audio_filter_pipeline.processors.mos import close_mos_estimators, get_mos_estimator

    preprocessing = preprocessing_report(model, items)
    audio_data = [(item.waveform, item.sample_rate) for item in items]
    started_load = time.perf_counter()
    estimator = None
    try:
        with _option_binding(model, options):
            estimator = get_mos_estimator(
                model, use_gpu=use_gpu, use_cache=False, allow_fallback=False
            )
            load_seconds = time.perf_counter() - started_load
            attestation = estimator.runtime_attestation()
            started_single = time.perf_counter()
            single_results = [estimator.compute(audio_data=value) for value in audio_data]
            single_seconds = time.perf_counter() - started_single
            telemetry_before = _telemetry_before(model, estimator)
            started_batch = time.perf_counter()
            batch_results = _compute_in_bounded_batches(estimator, audio_data, batch_size)
            batch_seconds = time.perf_counter() - started_batch
            comparison = compare_scores(
                model, single_results, batch_results, tolerance=tolerance
            )
            telemetry = _session_telemetry(model, estimator, telemetry_before)
    finally:
        if estimator is not None:
            close_mos_estimators([estimator])
    total_audio_seconds = sum(item.submitted_seconds for item in items)
    return {
        "model": model,
        "asset_provenance": dict(provenance),
        "runtime_attestation": attestation,
        "preprocessing": preprocessing,
        "batch_histogram": batch_histogram(preprocessing),
        "derived_expected_chunks": derived_expected_chunks(
            model, preprocessing, batch_size=batch_size, use_gpu=use_gpu
        ),
        "derived_expected_chunks_note": (
            "derived from public exact-length bucketing and outer batch-size bounds; "
            "not an observed provider run count"
        ),
        "timing_seconds": {
            "load": load_seconds,
            "single": single_seconds,
            "batch": batch_seconds,
        },
        "throughput_audio_seconds_per_wall_second": {
            "single": total_audio_seconds / single_seconds if single_seconds > 0 else None,
            "batch": total_audio_seconds / batch_seconds if batch_seconds > 0 else None,
        },
        "comparison": comparison,
        "telemetry": telemetry,
    }


def model_failure_report(
    model: str,
    items: Sequence[AudioItem],
    *,
    batch_size: int,
    use_gpu: bool,
    error: BaseException,
) -> dict[str, Any]:
    """保留独立模型的 fail-closed 错误，不让其吞没其他模型的真实观测。"""
    preprocessing = preprocessing_report(model, items)
    return {
        "model": model,
        "status": "failed",
        "error": {"type": type(error).__name__, "message": str(error)},
        "preprocessing": preprocessing,
        "batch_histogram": batch_histogram(preprocessing),
        "derived_expected_chunks": derived_expected_chunks(
            model, preprocessing, batch_size=batch_size, use_gpu=use_gpu
        ),
        "derived_expected_chunks_note": (
            "derived from public exact-length bucketing and outer batch-size bounds; "
            "not an observed provider run count"
        ),
        "comparison": None,
        "telemetry": None,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    models = validate_args(args)
    items = load_audio_items(
        args.audio,
        max_items=args.max_items,
        max_source_seconds=args.max_source_seconds,
    )
    if not items:
        raise ValueError("no audio items remain after max-items")
    reports = []
    preflight: dict[str, Any] = {}
    for model in models:
        try:
            verified = verified_assets((model,), use_gpu=args.device == "cuda")
            plan = verified.as_mapping()
            preflight[model] = plan
            report = run_model(
                model,
                items,
                batch_size=args.batch_size,
                use_gpu=args.device == "cuda",
                tolerance=args.score_tolerance,
                options=plan["options_by_model"][model],
                provenance=plan["provenance"]["models"][model],
            )
            report["status"] = "completed"
            reports.append(report)
        except Exception as exc:
            preflight[model] = {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            reports.append(
                model_failure_report(
                    model,
                    items,
                    batch_size=args.batch_size,
                    use_gpu=args.device == "cuda",
                    error=exc,
                )
            )
    scoreq_truncation_observed = any(
        record.get("ten_second_capped") is True
        for report in reports
        if report["model"] == "scoreq"
        for record in report["preprocessing"]
    )
    truncation_requirement_passed = (
        not args.require_scoreq_truncation or scoreq_truncation_observed
    )
    comparison_passed = bool(
        truncation_requirement_passed
        and reports
        and all(
            report.get("status") == "completed"
            and report["comparison"]["all_within_tolerance"]
            for report in reports
        )
    )
    return {
        "benchmark": "real_local_mos_batching",
        "device_requested": args.device,
        "models_requested": list(models),
        "max_items": args.max_items,
        "batch_size": args.batch_size,
        "max_source_seconds": args.max_source_seconds,
        "score_tolerance": args.score_tolerance,
        "submitted_item_count": len(items),
        "submitted_audio_seconds": sum(item.submitted_seconds for item in items),
        "asset_preflight_by_model": preflight,
        "scoreq_ten_second_truncation_observed": scoreq_truncation_observed if "scoreq" in models else None,
        "scoreq_ten_second_truncation_required": bool(args.require_scoreq_truncation),
        "comparison_passed": comparison_passed,
        "models": reports,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(args)
    text = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["comparison_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
