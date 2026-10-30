#!/usr/bin/env python3
"""可重复的流式 GCC-PHAT 配置 benchmark。

主排名只使用带已知 delay oracle 的合成场景。真实 AEC Challenge 音频若通过
``--real-root`` 提供，只计算 finite/抖动/激活稳定性 veto，不报告“精度”。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "tasks" / "aec" / "inference" / "csrc"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.aec.inference.streaming_delay_estimator import (  # noqa: E402
    StreamingDelayEstimator,
    StreamingDelayEstimatorConfig,
    _load_delay_estimator_extension,
    create_delay_estimator,
)


FALSE_ACCEPT_PROB = 1e-4


@dataclass(frozen=True)
class Profile:
    buffer_ms: float = 500.0
    residual_ms: float = 50.0
    update_ms: float = 100.0
    alpha: float = 0.15
    acps_alpha: float = 0.8
    peak_thresh: float = 0.05
    prominence_db: float = 6.0
    false_accept_prob: float = FALSE_ACCEPT_PROB
    min_obs_ms: float = 200.0
    frame_size: int = 160

    def __post_init__(self) -> None:
        if isinstance(self.frame_size, bool) or not isinstance(self.frame_size, int):
            raise ValueError(f"frame_size must be a positive integer, got {self.frame_size!r}")
        if self.frame_size <= 0:
            raise ValueError(f"frame_size must be positive, got {self.frame_size}")
        if self.false_accept_prob != FALSE_ACCEPT_PROB:
            raise ValueError(
                f"false_accept_prob is fixed at {FALSE_ACCEPT_PROB:g} for this benchmark"
            )
        # 复用生产配置的完整范围与 finite 校验。
        self.config(16000)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "Profile":
        if not isinstance(raw, dict):
            raise ValueError(f"candidate profile must be an object, got {type(raw).__name__}")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"candidate profile has unknown fields: {unknown}")
        return cls(**{**asdict(BASELINE), **raw})

    def config(self, sample_rate: int) -> StreamingDelayEstimatorConfig:
        # coarse 与 AECDeploymentFrontend 默认一致（50ms）；Profile 不单独暴露 coarse 字段。
        return StreamingDelayEstimatorConfig(
            sample_rate=sample_rate,
            buffer_duration_ms=self.buffer_ms,
            coarse_delay_ms=50.0,
            max_residual_delay_ms=self.residual_ms,
            update_interval_ms=self.update_ms,
            smooth_alpha=self.alpha,
            acps_alpha=self.acps_alpha,
            peak_threshold=self.peak_thresh,
            peak_prominence_db=self.prominence_db,
            false_accept_prob=self.false_accept_prob,
            min_observation_ms=self.min_obs_ms,
        )

    @property
    def name(self) -> str:
        values = asdict(self)
        return "_".join(f"{key}={value:g}" for key, value in values.items())


BASELINE = Profile()
DEV_SEED_BASE = 1100
HOLDOUT_SEED_BASE = 9100
ERROR_TOLERANCE_MS = 15.0
NEAREND_MIN_ACTIVATION_MS = 10.0
NEAREND_SEEDS = {
    "dev": (1129, 1230, 1331, 1432, 1533),
    "holdout": (9129, 9230, 9331, 9432, 9533),
}
NEAREND_SEED_COUNT = 5
_CPP_EXTENSION_IDENTITY: dict[str, Any] | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpp_extension_identity() -> dict[str, Any]:
    """加载并绑定 tasks/aec/inference/_native/ 中的 C++ 扩展构建身份。"""
    global _CPP_EXTENSION_IDENTITY
    if _CPP_EXTENSION_IDENTITY is not None:
        return _CPP_EXTENSION_IDENTITY
    try:
        module = _load_delay_estimator_extension()
    except ImportError as exc:
        raise RuntimeError(
            f"delay_estimator_ext could not be loaded from the native dir: {exc}"
        ) from exc
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError("delay_estimator_ext has no traceable __file__")
    extension_path = Path(module_file).resolve()
    source_path = (CSRC / "delay_estimator_ext.cpp").resolve()
    extension_stat = extension_path.stat()
    source_stat = source_path.stat()
    _CPP_EXTENSION_IDENTITY = {
        "path": str(extension_path),
        "sha256": _file_sha256(extension_path),
        "mtime_ns": extension_stat.st_mtime_ns,
        "size_bytes": extension_stat.st_size,
        "source": {
            "path": str(source_path),
            "sha256": _file_sha256(source_path),
            "mtime_ns": source_stat.st_mtime_ns,
        },
    }
    return _CPP_EXTENSION_IDENTITY


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    delay_ms: float = 100.0
    echo_gain: float = 0.3
    variant: str = "clean"
    jump_to_ms: float | None = None
    jump_time_s: float = 2.0
    group: str | None = None

    def __post_init__(self) -> None:
        if self.group is not None:
            if not self.group:
                raise ValueError("scenario group must be non-empty")
            return
        if self.jump_to_ms is not None:
            group = "delay_jump"
        elif not self.has_oracle:
            group = "nearend_only"
        elif self.name.startswith("stationary_"):
            group = "stationary_delay"
        elif "weak_echo" in self.name:
            group = "echo_gain"
        else:
            group = self.variant
        object.__setattr__(self, "group", group)

    @property
    def has_oracle(self) -> bool:
        return self.variant != "nearend_only"


def _shaped_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.normal(0.0, 1.0, n).astype(np.float32)
    x = np.convolve(x, np.ones(8, dtype=np.float32) / 8.0, mode="same")
    return (x * (0.35 / (np.std(x) + 1e-12))).astype(np.float32)


def synthesize_scenario(
    scenario: Scenario, sample_rate: int, duration_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 ``observed_ref, mic, oracle_delay_ms``。"""
    rng = np.random.default_rng(scenario.seed)
    n = int(round(duration_s * sample_rate))
    ref_clean = _shaped_noise(rng, n)
    delay_ms = np.full(n, scenario.delay_ms, dtype=np.float32)
    if scenario.jump_to_ms is not None:
        delay_ms[int(round(scenario.jump_time_s * sample_rate)) :] = scenario.jump_to_ms
    delay_samples = np.rint(delay_ms * sample_rate / 1000.0).astype(np.int64)
    source_index = np.arange(n, dtype=np.int64) - delay_samples
    echo = np.zeros(n, dtype=np.float32)
    valid = source_index >= 0
    echo[valid] = ref_clean[source_index[valid]] * scenario.echo_gain

    if scenario.variant == "multipath":
        # seed 同时冻结多径结构；dev/holdout 的不同 seed 因而使用不同反射路径。
        reflection_ms = 23.0 + float(scenario.seed % 31)
        second = delay_samples + int(round(reflection_ms * sample_rate / 1000.0))
        second_index = np.arange(n, dtype=np.int64) - second
        valid_second = second_index >= 0
        echo[valid_second] += min(0.16, scenario.echo_gain * 0.5) * ref_clean[
            second_index[valid_second]
        ]

    observed_ref = ref_clean.copy()
    near_gain = 0.03
    noise_gain = 0.01
    if scenario.variant == "noise":
        noise_gain = 0.08
    elif scenario.variant == "doubletalk":
        near_gain = 0.25
    elif scenario.variant == "nearend_only":
        echo.fill(0.0)
        near_gain = 0.3
    elif scenario.variant == "dropout":
        start = int(round(1.2 * sample_rate))
        stop = int(round(1.8 * sample_rate))
        observed_ref[start:stop] = 0.0

    near = _shaped_noise(np.random.default_rng(scenario.seed + 10000), n) * near_gain
    noise = rng.normal(0.0, noise_gain, n).astype(np.float32)
    mic = echo + near + noise
    if scenario.variant == "clipping":
        mic = np.clip(mic * 4.0, -0.65, 0.65).astype(np.float32)
    return observed_ref, mic.astype(np.float32), delay_ms


def build_scenarios(split: str) -> list[Scenario]:
    if split not in {"dev", "holdout"}:
        raise ValueError(f"unknown split: {split}")
    seed0 = DEV_SEED_BASE if split == "dev" else HOLDOUT_SEED_BASE
    delays = [0, 25, 50, 100, 200, 300, 400]
    scenarios = [
        Scenario(f"stationary_{delay}ms", seed0 + i, delay_ms=delay)
        for i, delay in enumerate(delays)
    ]
    scenarios.extend(
        [
            Scenario("weak_echo_100ms", seed0 + 20, 100, 0.1),
            Scenario("very_weak_echo_400ms", seed0 + 21, 400, 0.05),
            Scenario("noise_300ms", seed0 + 22, 300, 0.1, "noise"),
            Scenario("doubletalk_200ms", seed0 + 23, 200, 0.1, "doubletalk"),
            Scenario("multipath_300ms", seed0 + 24, 300, 0.3, "multipath"),
            Scenario("clipping_100ms", seed0 + 25, 100, 0.3, "clipping"),
            Scenario("farend_dropout_200ms", seed0 + 26, 200, 0.2, "dropout"),
            Scenario("jump_40_to_350ms", seed0 + 27, 40, 0.15, "clean", 350),
            Scenario("jump_350_to_40ms", seed0 + 28, 350, 0.15, "clean", 40),
        ]
    )
    scenarios.extend(
        Scenario(
            f"nearend_only_seed_{index}",
            seed,
            0,
            0.0,
            "nearend_only",
        )
        for index, seed in enumerate(NEAREND_SEEDS[split])
    )
    if split == "holdout":
        # 不同 seed、多径 delay 和 jump 时刻；搜索阶段从不读取这些结果。
        scenarios = [
            replace(s, jump_time_s=2.35 if s.jump_to_ms is not None else s.jump_time_s)
            for s in scenarios
        ]
        scenarios.append(Scenario("holdout_multipath_175ms", seed0 + 30, 175, 0.12, "multipath"))
    return scenarios


def _make_estimator(profile: Profile, sample_rate: int, backend: str):
    cfg = profile.config(sample_rate)
    if backend == "python":
        estimator = StreamingDelayEstimator(cfg)
    else:
        cpp_extension_identity()
        estimator = create_delay_estimator(cfg)
    actual_backend = estimator.backend_identity
    if actual_backend != backend:
        raise RuntimeError(
            f"requested {backend} delay estimator backend, got {actual_backend}"
        )
    return estimator


def run_scenario(
    profile: Profile,
    scenario: Scenario,
    *,
    sample_rate: int,
    duration_s: float,
    backend: str,
) -> dict:
    ref, mic, oracle = synthesize_scenario(scenario, sample_rate, duration_s)
    estimator = _make_estimator(profile, sample_rate, backend)
    delays: list[float] = []
    oracle_frames: list[float] = []
    frame_times: list[float] = []
    update_trigger_times: list[float] = []
    samples_since_update = 0
    update_interval_samples = round(profile.update_ms * sample_rate / 1000.0)
    started = time.perf_counter()
    for start in range(0, ref.size, profile.frame_size):
        stop = min(start + profile.frame_size, ref.size)
        frame_samples = stop - start
        estimator.push_ref(ref[start:stop])
        estimator.push_mic(mic[start:stop])
        estimator.get_aligned_ref_frame(frame_samples)
        delays.append(float(estimator.delay_ms))
        oracle_frames.append(float(oracle[stop - 1]))
        frame_times.append(stop / sample_rate)
        samples_since_update += frame_samples
        if samples_since_update >= update_interval_samples:
            update_trigger_times.append(stop / sample_rate)
            samples_since_update %= update_interval_samples
    elapsed = time.perf_counter() - started
    estimates = np.asarray(delays, dtype=np.float64)
    truth = np.asarray(oracle_frames, dtype=np.float64)
    times = np.asarray(frame_times, dtype=np.float64)
    cadence_ms = np.diff(np.asarray([0.0, *update_trigger_times])) * 1000.0
    record = {
        "name": scenario.name,
        "group": scenario.group,
        "variant": scenario.variant,
        "seed": scenario.seed,
        "elapsed_s": elapsed,
        "audio_s": ref.size / sample_rate,
        "finite": bool(np.all(np.isfinite(estimates))),
        "final_delay_ms": float(estimates[-1]),
        "update_trigger_count": len(update_trigger_times),
        "actual_update_cadence_ms": {
            "mean": float(np.mean(cadence_ms)) if cadence_ms.size else None,
            "p95": float(np.percentile(cadence_ms, 95)) if cadence_ms.size else None,
        },
    }
    if not scenario.has_oracle:
        # 至少跨过一帧且不低于 10 ms；亚采样插值或量化抖动不计作实际误激活。
        activation_threshold_ms = max(
            NEAREND_MIN_ACTIVATION_MS,
            profile.frame_size * 1000.0 / sample_rate,
        )
        record["false_activation_threshold_ms"] = activation_threshold_ms
        record["false_activation_rate"] = float(
            np.mean(np.abs(estimates) > activation_threshold_ms)
        )
        return record

    if scenario.jump_to_ms is None:
        tail = times >= max(duration_s - 1.0, duration_s * 0.6)
        errors = np.abs(estimates[tail] - truth[tail])
        jitter = np.abs(np.diff(estimates[tail]))
        record.update(
            mae_ms=float(np.mean(errors)),
            p95_error_ms=float(np.percentile(errors, 95)),
            false_lock_rate=float(np.mean(errors > ERROR_TOLERANCE_MS)),
            converged=bool(np.percentile(errors, 95) <= ERROR_TOLERANCE_MS),
            jitter_p95_ms=float(np.percentile(jitter, 95)) if jitter.size else 0.0,
        )
    else:
        after = times >= scenario.jump_time_s
        errors = np.abs(estimates - truth)
        indices = np.flatnonzero(after)
        reacq_s = duration_s - scenario.jump_time_s
        for idx in indices:
            if idx + 2 < errors.size and np.all(
                errors[idx : idx + 3] <= ERROR_TOLERANCE_MS
            ):
                reacq_s = max(0.0, float(times[idx] - scenario.jump_time_s))
                break
        steady = times >= max(duration_s - 0.8, scenario.jump_time_s)
        jitter = np.abs(np.diff(estimates[steady]))
        record.update(
            reacquisition_s=reacq_s,
            final_error_ms=float(errors[-1]),
            jitter_p95_ms=float(np.percentile(jitter, 95)) if jitter.size else 0.0,
        )
    return record


def summarize(profile: Profile, records: list[dict]) -> dict:
    stationary = [r for r in records if "p95_error_ms" in r]
    jumps = [r for r in records if "reacquisition_s" in r]
    negatives = [r for r in records if "false_activation_rate" in r]
    grouped: dict[str, list[dict]] = {}
    for record in stationary:
        grouped.setdefault(record["group"], []).append(record)
    group_metrics = {
        name: {
            "scenario_count": len(items),
            "mae_ms": float(np.mean([item["mae_ms"] for item in items])),
            "p95_error_ms": float(max(item["p95_error_ms"] for item in items)),
            "false_lock_rate": float(np.mean([item["false_lock_rate"] for item in items])),
            "no_convergence_rate": float(np.mean([not item["converged"] for item in items])),
        }
        for name, items in sorted(grouped.items())
    }
    cpu = sum(r["elapsed_s"] for r in records) / sum(r["audio_s"] for r in records)
    worst_mae = max((item["mae_ms"] for item in group_metrics.values()), default=float("inf"))
    worst_p95 = max(
        (item["p95_error_ms"] for item in group_metrics.values()), default=float("inf")
    )
    no_convergence = float(np.mean([not r["converged"] for r in stationary])) if stationary else 1.0
    false_lock = float(np.mean([r["false_lock_rate"] for r in stationary])) if stationary else 1.0
    oracle_nonjump_passed = bool(
        stationary
        and all(
            item["converged"]
            and item["p95_error_ms"] <= ERROR_TOLERANCE_MS
            and item["false_lock_rate"] <= 0.01
            for item in stationary
        )
    )
    nearend_rates = [r["false_activation_rate"] for r in negatives]
    nearend_activation = (
        float(np.median(nearend_rates)) if nearend_rates else float("inf")
    )
    nearend_activation_p95 = (
        float(np.percentile(nearend_rates, 95)) if nearend_rates else float("inf")
    )
    nearend_seed_pass_rate = (
        float(np.mean(np.asarray(nearend_rates) <= 0.01)) if nearend_rates else 0.0
    )
    nearend_max_final_delay = max(
        (abs(r["final_delay_ms"]) for r in negatives), default=float("inf")
    )
    nearend_all_seeds_passed = bool(
        len(negatives) == NEAREND_SEED_COUNT
        and all(
            item["false_activation_rate"] <= 0.01
            and abs(item["final_delay_ms"])
            <= item["false_activation_threshold_ms"]
            for item in negatives
        )
    )
    reacquisition_values = [r["reacquisition_s"] for r in jumps]
    reacq = float(np.percentile(reacquisition_values, 95)) if reacquisition_values else float("inf")
    jump_max_final_error = max(
        (r["final_error_ms"] for r in jumps), default=float("inf")
    )
    jump_all_scenarios_passed = bool(
        jumps
        and all(
            item["reacquisition_s"] <= 2.0
            and item["final_error_ms"] <= ERROR_TOLERANCE_MS
            for item in jumps
        )
    )
    jitter = max(
        (r["jitter_p95_ms"] for r in records if "jitter_p95_ms" in r),
        default=0.0,
    )
    cadence_values = [
        r["actual_update_cadence_ms"]["mean"]
        for r in records
        if r["actual_update_cadence_ms"]["mean"] is not None
    ]
    actual_cadence = float(np.mean(cadence_values)) if cadence_values else None
    finite = all(r["finite"] for r in records)
    hard_gate = bool(
        finite
        and oracle_nonjump_passed
        and nearend_all_seeds_passed
        and jump_all_scenarios_passed
        and jitter <= 5.0
    )
    score = (
        worst_mae
        + 1.5 * worst_p95
        + 15.0 * no_convergence
        + 5.0 * false_lock
        + 4.0 * reacq
        + jitter
    )
    return {
        "profile": asdict(profile),
        "name": profile.name,
        "hard_gate": hard_gate,
        "score": float(score),
        "group_metrics": group_metrics,
        "worst_group_mae_ms": float(worst_mae),
        "worst_group_p95_ms": float(worst_p95),
        "no_convergence_rate": float(no_convergence),
        "false_lock_rate": float(false_lock),
        "oracle_nonjump_all_scenarios_passed": oracle_nonjump_passed,
        "nearend_false_activation_rate": float(nearend_activation),
        "nearend_false_activation_p95_rate": float(nearend_activation_p95),
        "nearend_seed_count": len(negatives),
        "nearend_seed_pass_rate": float(nearend_seed_pass_rate),
        "nearend_max_final_delay_ms": float(nearend_max_final_delay),
        "nearend_all_seeds_passed": nearend_all_seeds_passed,
        "jump_reacquisition_p95_s": float(reacq),
        "jump_max_final_error_ms": float(jump_max_final_error),
        "jump_all_scenarios_passed": jump_all_scenarios_passed,
        "jitter_p95_ms": float(jitter),
        "cpu_seconds_per_audio_second": float(cpu),
        "actual_update_cadence_ms": actual_cadence,
        "scenarios": records,
    }


def evaluate_profiles(
    profiles: Iterable[Profile],
    scenarios: list[Scenario],
    *,
    sample_rate: int,
    duration_s: float,
    backend: str,
) -> list[dict]:
    results = []
    for profile in dict.fromkeys(profiles):
        records = [
            run_scenario(
                profile,
                scenario,
                sample_rate=sample_rate,
                duration_s=duration_s,
                backend=backend,
            )
            for scenario in scenarios
        ]
        results.append(summarize(profile, records))
    results.sort(key=lambda item: (not item["hard_gate"], item["score"], item["cpu_seconds_per_audio_second"]))
    return results


def representative_profiles() -> list[Profile]:
    """返回围绕现行基线的轻量候选集。全部候选必须互不相同。"""
    return [
        BASELINE,
        replace(BASELINE, residual_ms=500.0, acps_alpha=0.7, min_obs_ms=100.0),
        replace(BASELINE, residual_ms=100.0),
        replace(BASELINE, update_ms=50.0),
        replace(BASELINE, acps_alpha=0.9),
        replace(BASELINE, min_obs_ms=100.0),
    ]


DEFAULT_STAGE_GRID = {
    "structure_memory": {
        "buffer_residual_ms": [[400, 50], [500, 50], [650, 50], [500, 100], [500, 500]],
        "update_ms": [50, 100, 200],
        "acps_alpha": [0.8, 0.9, 0.0, 0.7],
    },
    "quality_gate": {
        "peak_thresh": [0.03, 0.05, 0.08],
        "prominence_db": [3, 6, 9],
    },
    "smoothing_warmup_frame": {
        "alpha": [0.10, 0.15, 0.30],
        # 200 为现行默认，100 为旧默认值对照。
        "min_obs_ms": [200, 100, 50],
        "frame_size": [80, 160, 320],
    },
}


def load_search_config(path: Path | None) -> dict[str, Any]:
    """读取可选候选/阶段网格；缺省返回预注册网格。"""
    if path is None:
        return {"candidates": None, "stages": DEFAULT_STAGE_GRID}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("search config must be a JSON object")
    unknown = sorted(set(raw) - {"candidates", "stages"})
    if unknown:
        raise ValueError(f"search config has unknown fields: {unknown}")
    candidates_raw = raw.get("candidates")
    candidates = None
    if candidates_raw is not None:
        if not isinstance(candidates_raw, list) or not candidates_raw:
            raise ValueError("search config candidates must be a non-empty list")
        candidates = [Profile.from_mapping(item) for item in candidates_raw]

    stages = json.loads(json.dumps(DEFAULT_STAGE_GRID))
    stage_overrides = raw.get("stages", {})
    if not isinstance(stage_overrides, dict):
        raise ValueError("search config stages must be an object")
    unknown_stages = sorted(set(stage_overrides) - set(stages))
    if unknown_stages:
        raise ValueError(f"search config has unknown stages: {unknown_stages}")
    for stage_name, overrides in stage_overrides.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"stage {stage_name} must be an object")
        unknown_keys = sorted(set(overrides) - set(stages[stage_name]))
        if unknown_keys:
            raise ValueError(f"stage {stage_name} has unknown fields: {unknown_keys}")
        for key, values in overrides.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"stage {stage_name}.{key} must be a non-empty list")
            stages[stage_name][key] = values
    return {"candidates": candidates, "stages": stages}


def _top_profiles(results: list[dict], count: int = 3) -> list[Profile]:
    return [Profile(**item["profile"]) for item in results[:count]]


def staged_search(
    search: str,
    dev_scenarios: list[Scenario],
    *,
    sample_rate: int,
    duration_s: float,
    backend: str,
    search_config: dict[str, Any],
) -> tuple[list[dict], list[dict]]:
    stages: list[dict] = []
    configured_candidates = search_config["candidates"]
    if configured_candidates is not None:
        results = evaluate_profiles(
            [BASELINE, *configured_candidates],
            dev_scenarios,
            sample_rate=sample_rate,
            duration_s=duration_s,
            backend=backend,
        )
        stages.append({"stage": "configured_candidates", "results": results})
        return results, stages
    if search == "representative":
        results = evaluate_profiles(
            representative_profiles(),
            dev_scenarios,
            sample_rate=sample_rate,
            duration_s=duration_s,
            backend=backend,
        )
        stages.append({"stage": "representative", "results": results})
        return results, stages

    grid = search_config["stages"]
    first = grid["structure_memory"]
    stage1_profiles = [
        replace(BASELINE, buffer_ms=b, residual_ms=r, update_ms=u, acps_alpha=a)
        for b, r in first["buffer_residual_ms"]
        for u in first["update_ms"]
        for a in first["acps_alpha"]
    ]
    results = evaluate_profiles(
        stage1_profiles,
        dev_scenarios,
        sample_rate=sample_rate,
        duration_s=duration_s,
        backend=backend,
    )
    stages.append({"stage": "structure_memory", "results": results})
    second = grid["quality_gate"]
    stage2_profiles = [
        replace(profile, peak_thresh=peak, prominence_db=prom)
        for profile in _top_profiles(results)
        for peak in second["peak_thresh"]
        for prom in second["prominence_db"]
    ]
    results = evaluate_profiles(
        stage2_profiles,
        dev_scenarios,
        sample_rate=sample_rate,
        duration_s=duration_s,
        backend=backend,
    )
    stages.append({"stage": "quality_gate", "results": results})
    third = grid["smoothing_warmup_frame"]
    stage3_profiles = [
        replace(profile, alpha=alpha, min_obs_ms=obs, frame_size=frame)
        for profile in _top_profiles(results)
        for alpha in third["alpha"]
        for obs in third["min_obs_ms"]
        for frame in third["frame_size"]
    ]
    results = evaluate_profiles(
        [BASELINE, *stage3_profiles],
        dev_scenarios,
        sample_rate=sample_rate,
        duration_s=duration_s,
        backend=backend,
    )
    stages.append({"stage": "smoothing_warmup_frame", "results": results})
    return results, stages


def choose_winner(dev: list[dict], holdout: list[dict]) -> dict:
    baseline_holdout = next(item for item in holdout if item["profile"] == asdict(BASELINE))
    holdout_by_name = {item["name"]: item for item in holdout}
    eligible = []
    for item in dev:
        held = holdout_by_name.get(item["name"])
        if held is None or not item["hard_gate"] or not held["hard_gate"]:
            continue
        cpu_ratio = held["cpu_seconds_per_audio_second"] / max(
            baseline_holdout["cpu_seconds_per_audio_second"], 1e-12
        )
        if cpu_ratio <= 1.25:
            eligible.append((item, held, cpu_ratio))
    if not eligible:
        return {
            "selected": False,
            "profile": None,
            "name": None,
            "reason": "no profile passed all pre-registered dev/holdout hard and CPU gates",
            "cpu_ratio_vs_baseline": None,
            "dev": None,
            "holdout": None,
        }

    # holdout 只作通过/否决；接近最佳 dev 的候选按 baseline 距离、buffer、CPU 决胜。
    best_score = min(item[0]["score"] for item in eligible)
    statistically_close = [
        item
        for item in eligible
        if item[0]["score"] <= best_score + max(0.01, best_score * 0.01)
    ]

    def tie_break(value):
        dev_item, held_item, _ = value
        profile = dev_item["profile"]
        changed = sum(profile[key] != asdict(BASELINE)[key] for key in profile)
        return (
            changed,
            profile["buffer_ms"],
            held_item["cpu_seconds_per_audio_second"],
            dev_item["score"],
            dev_item["name"],
        )

    dev_winner, held_winner, ratio = min(statistically_close, key=tie_break)
    return {
        "selected": True,
        "profile": held_winner["profile"],
        "name": held_winner["name"],
        "reason": "top dev candidate passed frozen holdout and CPU gates",
        "cpu_ratio_vs_baseline": float(ratio),
        "dev": dev_winner,
        "holdout": held_winner,
    }


def backend_parity(
    profiles: Iterable[Profile], scenarios: list[Scenario], *, sample_rate: int, duration_s: float
) -> dict:
    """冻结配置后比较 Python/C++ 逐场景最终 delay，不参与排名。"""
    profile_reports = []
    for profile in dict.fromkeys(profiles):
        deltas = []
        records = []
        for scenario in scenarios:
            py = run_scenario(
                profile, scenario, sample_rate=sample_rate, duration_s=duration_s, backend="python"
            )
            cpp = run_scenario(
                profile, scenario, sample_rate=sample_rate, duration_s=duration_s, backend="cpp"
            )
            raw_delta = abs(py["final_delay_ms"] - cpp["final_delay_ms"])
            finite = bool(np.isfinite(raw_delta))
            delta = float(raw_delta) if finite else None
            if delta is not None:
                deltas.append(delta)
            records.append(
                {
                    "name": scenario.name,
                    "group": scenario.group,
                    "finite": finite,
                    "final_delay_delta_ms": delta,
                    "passed": bool(finite and delta is not None and delta <= 1.0),
                }
            )
        max_delta = max(deltas, default=None)
        profile_reports.append(
            {
                "profile": asdict(profile),
                "max_final_delay_delta_ms": max_delta,
                "passed": bool(records and all(item["passed"] for item in records)),
                "scenarios": records,
            }
        )
    return {
        "available": True,
        "passed": bool(profile_reports and all(item["passed"] for item in profile_reports)),
        "tolerance_ms": 1.0,
        "profiles": profile_reports,
    }


def discover_real_pairs(root: Path, limit: int) -> list[tuple[Path, Path]]:
    pairs = []
    for mic in sorted(root.rglob("*_mic.wav")):
        lpb = mic.with_name(mic.name.replace("_mic.wav", "_lpb.wav"))
        if lpb.is_file():
            pairs.append((mic, lpb))
        if len(pairs) >= limit:
            break
    return pairs


def real_audio_veto(
    profiles: Iterable[Profile],
    root: Path,
    *,
    sample_rate: int,
    duration_s: float,
    limit: int,
    backend: str,
    jitter_threshold_ms: float,
) -> dict:
    """无 oracle 真实集仅做 finite/抖动稳定性 veto，绝不报告 delay 精度。"""
    if not np.isfinite(jitter_threshold_ms) or jitter_threshold_ms < 0:
        raise ValueError("real-audio jitter threshold must be finite and non-negative")
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("--real-root requires the optional soundfile package") from exc

    pairs = discover_real_pairs(root, limit)
    profile_reports = []
    for profile in dict.fromkeys(profiles):
        files = []
        for mic_path, ref_path in pairs:
            mic, mic_sr = sf.read(mic_path, dtype="float32", always_2d=False)
            ref, ref_sr = sf.read(ref_path, dtype="float32", always_2d=False)
            if mic_sr != sample_rate or ref_sr != sample_rate:
                raise ValueError(
                    f"real veto requires {sample_rate} Hz files, got {mic_sr}/{ref_sr}: {mic_path}"
                )
            if mic.ndim > 1:
                mic = mic[:, 0]
            if ref.ndim > 1:
                ref = ref[:, 0]
            n = min(len(mic), len(ref), int(duration_s * sample_rate))
            estimator = _make_estimator(profile, sample_rate, backend)
            delays = []
            for start in range(0, n, profile.frame_size):
                stop = min(start + profile.frame_size, n)
                estimator.push_ref(ref[start:stop])
                estimator.push_mic(mic[start:stop])
                delays.append(float(estimator.delay_ms))
            values = np.asarray(delays)
            jitter = np.abs(np.diff(values))
            finite = bool(values.size and np.all(np.isfinite(values)))
            jitter_p95_ms = (
                float(np.percentile(jitter, 95))
                if finite and jitter.size
                else (0.0 if finite else None)
            )
            files.append(
                {
                    "mic": str(mic_path.relative_to(root)),
                    "lpb": str(ref_path.relative_to(root)),
                    "finite": finite,
                    "nonzero_activation_rate": (
                        float(np.mean(np.abs(values) > ERROR_TOLERANCE_MS))
                        if finite
                        else None
                    ),
                    "jitter_p95_ms": jitter_p95_ms,
                    "passed": bool(
                        finite
                        and jitter_p95_ms is not None
                        and jitter_p95_ms <= jitter_threshold_ms
                    ),
                }
            )
        finite_jitter = [
            item["jitter_p95_ms"]
            for item in files
            if item["jitter_p95_ms"] is not None
        ]
        max_jitter = max(finite_jitter, default=None)
        profile_reports.append(
            {
                "profile": asdict(profile),
                "passed": bool(files and all(item["passed"] for item in files)),
                "max_jitter_p95_ms": max_jitter,
                "files": files,
            }
        )
    return {
        "enabled": True,
        "policy": "stability veto only; no delay oracle and no accuracy claim",
        "root": str(root),
        "backend": backend,
        "pair_count": len(pairs),
        "jitter_threshold_ms": jitter_threshold_ms,
        "passed": bool(pairs and profile_reports and all(item["passed"] for item in profile_reports)),
        "profiles": profile_reports,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", choices=("representative", "full"), default="representative")
    parser.add_argument("--search-config", type=Path, help="候选或阶段网格 JSON")
    parser.add_argument("--backend", choices=("cpp", "python", "both"), default="cpp")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=4.5)
    parser.add_argument("--real-root", type=Path)
    parser.add_argument("--real-limit", type=int, default=8)
    parser.add_argument(
        "--real-jitter-threshold-ms",
        type=float,
        default=5.0,
        help="真实音频稳定性 veto 的最大 p95 delay 抖动（毫秒）",
    )
    parser.add_argument("--output", type=Path, help="可选 JSON 输出路径；stdout 始终输出 JSON")
    return parser.parse_args(argv)


def _run_backend(
    args: argparse.Namespace,
    backend: str,
    search_config: dict[str, Any],
) -> dict[str, Any]:
    dev_results, stages = staged_search(
        args.search,
        build_scenarios("dev"),
        sample_rate=args.sample_rate,
        duration_s=args.duration,
        backend=backend,
        search_config=search_config,
    )
    finalists = list(dict.fromkeys([BASELINE, *_top_profiles(dev_results, count=5)]))
    holdout_results = evaluate_profiles(
        finalists,
        build_scenarios("holdout"),
        sample_rate=args.sample_rate,
        duration_s=args.duration,
        backend=backend,
    )
    return {
        "backend": backend,
        "stages": stages,
        "dev_final": dev_results,
        "holdout": holdout_results,
        "winner": choose_winner(dev_results, holdout_results),
    }


def _winner_gate_failures(
    backend_results: dict[str, dict[str, Any]], *, require_consistent_profile: bool
) -> list[str]:
    failures: list[str] = []
    valid_profiles: dict[str, dict[str, Any]] = {}
    use_backend_prefix = len(backend_results) > 1
    for backend, result in backend_results.items():
        prefix = f"{backend}_" if use_backend_prefix else ""
        winner = result.get("winner") or {}
        profile = winner.get("profile")
        try:
            parsed_profile = Profile(**profile) if isinstance(profile, dict) else None
        except (TypeError, ValueError):
            parsed_profile = None
        identity_valid = bool(
            winner.get("selected") is True
            and parsed_profile is not None
            and winner.get("name") == parsed_profile.name
            and isinstance(winner.get("dev"), dict)
            and isinstance(winner.get("holdout"), dict)
            and winner["dev"].get("profile") == profile
            and winner["holdout"].get("profile") == profile
        )
        if not identity_valid:
            failures.append(f"{prefix}winner_selection")
            continue
        valid_profiles[backend] = profile
        if not winner["dev"].get("hard_gate", False):
            failures.append(f"{prefix}winner_dev_hard_gate")
        if not winner["holdout"].get("hard_gate", False):
            failures.append(f"{prefix}winner_holdout_hard_gate")
        cpu_ratio = winner.get("cpu_ratio_vs_baseline")
        if (
            isinstance(cpu_ratio, bool)
            or not isinstance(cpu_ratio, (int, float, np.integer, np.floating))
            or not np.isfinite(cpu_ratio)
            or cpu_ratio > 1.25
        ):
            failures.append(f"{prefix}winner_cpu_gate")

    if (
        require_consistent_profile
        and set(valid_profiles) == {"python", "cpp"}
        and valid_profiles["python"] != valid_profiles["cpp"]
    ):
        failures.append("backend_winner_profile_mismatch")
    return failures


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sample_rate != 16000:
        raise ValueError("AEC frontend benchmark is fixed to 16 kHz")
    if not np.isfinite(args.duration) or args.duration < 3.5:
        raise ValueError("duration must be finite and >= 3.5 seconds")
    if args.real_limit <= 0:
        raise ValueError("real-limit must be positive")
    if not np.isfinite(args.real_jitter_threshold_ms) or args.real_jitter_threshold_ms < 0:
        raise ValueError("real-jitter-threshold-ms must be finite and non-negative")

    search_config = load_search_config(args.search_config)
    extension_identity = cpp_extension_identity()
    backends = ["python", "cpp"] if args.backend == "both" else [args.backend]
    backend_results = {
        backend: _run_backend(args, backend, search_config) for backend in backends
    }
    primary_backend = "cpp" if "cpp" in backend_results else "python"
    primary = backend_results[primary_backend]
    winner = primary["winner"]
    comparison_profiles = [BASELINE]
    for result in backend_results.values():
        profile = (result.get("winner") or {}).get("profile")
        if not isinstance(profile, dict):
            continue
        try:
            comparison_profiles.append(Profile(**profile))
        except (TypeError, ValueError):
            continue
    comparison_profiles = list(dict.fromkeys(comparison_profiles))

    parity_scenarios = [
        scenario
        for scenario in build_scenarios("holdout")
        if scenario.name
        in {
            "stationary_400ms",
            "multipath_300ms",
            "jump_40_to_350ms",
            "nearend_only_seed_0",
        }
    ]
    try:
        parity = backend_parity(
            comparison_profiles,
            parity_scenarios,
            sample_rate=args.sample_rate,
            duration_s=args.duration,
        )
    except RuntimeError as exc:
        parity = {
            "available": False,
            "passed": False,
            "reason": str(exc),
            "tolerance_ms": 1.0,
            "profiles": [],
        }

    real_veto = {
        "enabled": False,
        "policy": "stability veto only; no delay oracle and no accuracy claim",
        "root": None,
        "backend": primary_backend,
        "pair_count": 0,
        "jitter_threshold_ms": args.real_jitter_threshold_ms,
        "passed": None,
        "profiles": [],
    }
    if args.real_root is not None:
        real_veto = real_audio_veto(
            comparison_profiles,
            args.real_root,
            sample_rate=args.sample_rate,
            duration_s=args.duration,
            limit=args.real_limit,
            backend=primary_backend,
            jitter_threshold_ms=args.real_jitter_threshold_ms,
        )

    gate_failures = _winner_gate_failures(
        backend_results, require_consistent_profile=args.backend == "both"
    )
    if not parity["passed"]:
        gate_failures.append("backend_parity")
    if real_veto["enabled"] and not real_veto["passed"]:
        gate_failures.append("real_audio_veto")

    configured_candidates = search_config["candidates"]
    report = {
        "passed": not gate_failures,
        "gate_failures": gate_failures,
        "method": {
            "oracle": "synthetic delay trajectory only",
            "real_audio_policy": "stability veto only; no real delay accuracy is claimed",
            "sample_rate": args.sample_rate,
            "duration_s": args.duration,
            "backend": args.backend,
            "search": "configured_candidates" if configured_candidates else args.search,
        },
        "benchmark_config": {
            "primary_backend": primary_backend,
            "dev_seed_base": DEV_SEED_BASE,
            "holdout_seed_base": HOLDOUT_SEED_BASE,
            "nearend_seeds": {key: list(value) for key, value in NEAREND_SEEDS.items()},
            "false_accept_prob": FALSE_ACCEPT_PROB,
            "error_tolerance_ms": ERROR_TOLERANCE_MS,
            "false_activation_threshold_ms": (
                f"max({NEAREND_MIN_ACTIVATION_MS:g}, frame_duration_ms)"
            ),
            "hard_gates": {
                "oracle_nonjump_each_converged": True,
                "oracle_nonjump_each_p95_error_ms": 15.0,
                "oracle_nonjump_each_false_lock_rate": 0.01,
                "nearend_each_false_activation_rate": 0.01,
                "nearend_seed_count": NEAREND_SEED_COUNT,
                "nearend_each_final_delay_threshold_ms": (
                    f"max({NEAREND_MIN_ACTIVATION_MS:g}, frame_duration_ms)"
                ),
                "jump_each_reacquisition_s": 2.0,
                "jump_each_final_error_ms": 15.0,
                "jitter_p95_ms": 5.0,
                "winner_cpu_ratio": 1.25,
                "both_backend_winner_profile_match": True,
            },
            "baseline": asdict(BASELINE),
            "candidate_source": str(args.search_config) if args.search_config else "built-in",
            "candidates": (
                [asdict(item) for item in configured_candidates]
                if configured_candidates is not None
                else None
            ),
            "stage_grid": search_config["stages"],
            "cpp_extension": extension_identity,
        },
        "stages": primary["stages"],
        "dev_final": primary["dev_final"],
        "holdout": primary["holdout"],
        "winner": primary["winner"],
        "parity": parity,
        "real_audio_veto": real_veto,
        "backend_results": backend_results,
    }
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if gate_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
