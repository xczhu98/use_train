#!/usr/bin/env python3
"""Shared utilities for SE / AEC evaluation pipelines.

This module provides:
  - Experiment asset discovery (config + checkpoint from exp dir)
  - Model type detection (SE vs AEC, AEC sub-type)
  - Unified testset SCP preparation (all SCPs under the configured test root)
  - Unified inference wrappers (subprocess or in-process)
  - Evaluation wrappers (subprocess to existing evaluate/*.py scripts)
  - Scenario metric registry for task-specific metric plugins
  - Result aggregation / summary JSON builders
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np
import soundfile as sf

from core.utils.config import load_merged_config
from core.utils.inference_artifacts import (
    InferenceResult,
    atomic_write_audio,
    atomic_write_json,
    build_content_fingerprint,
    build_input_inventory,
    build_run_manifest,
    collect_source_tree_files,
    fingerprint_file,
    fingerprint_json,
    inference_output_lock,
    publish_inference_metadata,
    reusable_uids_from_manifest,
    stat_audio_identity,
)
from evaluate.artifact_lifecycle import (
    ARTIFACT_FILENAME,
    ARTIFACT_SCHEMA,
    CURRENT_FILENAME,
    CURRENT_SCHEMA,
    load_committed_artifact,
)
from evaluate.eval_io import (
    _directory_snapshot as _directory_snapshot_shared,
    generate_scp_from_dir as _generate_scp_from_dir_shared,
    load_audio,
    read_scp,
    validate_inference_artifacts as _validate_inference_artifacts,
)
from evaluate.inference_components import (
    build_artifact_fingerprint,
    configured_checkpoint_files,
    effective_config_identity,
    inference_source_files,
    load_effective_config,
    run_adapter_inference,
    verify_configured_checkpoint_fingerprints,
    validate_inference_publish_snapshot,
)
from evaluate.pipeline_runner import PipelineTaskAdapter, run_pipeline
from evaluate.process_runner import (
    EVAL_SUBPROCESS_TIMEOUT_S,
    PROJECT_ROOT,
    absolute_path,
    death_signal_preexec,
    repository_script,
    run_evaluate_urgent as _run_evaluate_urgent,
)
from tools.shared.scp_io import (
    parse_scp_fields,
    read_scp_dict,
    serialize_scp_entries,
    write_scp_entries,
)
from tools.shared.scp_provenance import (
    build_scp_provenance,
    scp_provenance_fresh,
    validate_scp_provenance,
    write_scp_provenance,
)

logger = logging.getLogger("eval_pipeline")


@dataclass(frozen=True)
class ScenarioMetricSpec:
    """可插拔的场景指标描述。

    评估 pipeline 仍可直接调用既有 ``compute_*_for_scenario`` API；注册表只
    收敛适用场景与统一调用合同，便于后续新增指标时不再向主 pipeline 追加
    一组 if/elif。指标函数必须返回 mapping 或 ``None``，异常由调用方按原有
    语义处理，不在 registry 层吞掉。
    """

    name: str
    scenarios: frozenset[str]
    compute: Callable[..., Mapping[str, Any] | None]

    def supports(self, scenario: str) -> bool:
        """判断该指标是否适用于给定仿真场景。"""
        return scenario in self.scenarios


_SCENARIO_METRIC_REGISTRY: dict[str, ScenarioMetricSpec] = {}

# 适用场景同时供公开单指标 API 和 registry 声明使用，避免 wrapper 回调
# 公开 API 时再次经过同一个 registry 判定而形成递归。
_ERLE_SCENARIOS = frozenset(
    {
        "prob_echo_only",
        "prob_noise_echo_only",
        "prob_target_echo",
        "prob_target_noise_echo",
        "farend-singletalk",
        "doubletalk",
    }
)
_NOISE_REDUCTION_SCENARIOS = frozenset({"prob_noise_only"})


def register_scenario_metric(
    name: str,
    *,
    scenarios: Iterable[str],
    compute: Callable[..., Mapping[str, Any] | None],
) -> ScenarioMetricSpec:
    """注册场景指标插件并返回不可变 spec。

    ``name`` 在当前进程内必须唯一；重复注册直接报错，防止评估指标被
    静默替换。该入口只管理指标适用范围和调用合同，不改变现有结果或
    artifact/provenance 生命周期。
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("scenario metric name must be a non-empty string")
    if name in _SCENARIO_METRIC_REGISTRY:
        raise RuntimeError(f"Duplicate scenario metric registration: {name}")
    scenario_set = frozenset(scenarios)
    if not scenario_set or any(not isinstance(item, str) or not item for item in scenario_set):
        raise ValueError(f"scenario metric {name!r} requires non-empty scenario names")
    if not callable(compute):
        raise TypeError(f"scenario metric {name!r} compute must be callable")
    spec = ScenarioMetricSpec(name=name, scenarios=scenario_set, compute=compute)
    _SCENARIO_METRIC_REGISTRY[name] = spec
    return spec


def iter_scenario_metrics(scenario: str) -> tuple[ScenarioMetricSpec, ...]:
    """按注册顺序返回适用于 ``scenario`` 的指标插件。"""
    return tuple(spec for spec in _SCENARIO_METRIC_REGISTRY.values() if spec.supports(scenario))


def compute_registered_scenario_metrics(
    scenario: str,
    mic_scp: str | Path,
    enhanced_scp: str | Path,
    *,
    ref_scp: str | Path | None = None,
    echo_scp: str | Path | None = None,
    eps: float = 1e-10,
    metric_names: set[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """执行适用的注册指标，供评估 pipeline 统一分派场景指标。

    公开的单指标函数仍保留为兼容 API；默认 AEC 评估路径通过本函数分派。
    传入 ``metric_names`` 时只执行指定名称，并对未知名称 fail-fast。
    """
    selected = None if metric_names is None else set(metric_names)
    if selected is not None:
        unknown = selected - set(_SCENARIO_METRIC_REGISTRY)
        if unknown:
            raise ValueError(f"Unknown scenario metric(s): {sorted(unknown)}")
    results: dict[str, dict[str, Any]] = {}
    for spec in iter_scenario_metrics(scenario):
        if selected is not None and spec.name not in selected:
            continue
        value = spec.compute(
            mic_scp,
            enhanced_scp,
            scenario=scenario,
            ref_scp=ref_scp,
            echo_scp=echo_scp,
            eps=eps,
        )
        if value is not None:
            results[spec.name] = dict(value)
    return results


# ---------------------------------------------------------------------------
# Default dataset paths (centralised so pipelines stay DRY)
#
# 未设置 ``USE_TRAIN_TEST_DATA_ROOT`` 时，``data/test`` 仅作为仓库内占位路径；
# 设置后，输入源和默认 test artifact 根均从该环境变量派生。
# ---------------------------------------------------------------------------
_TEST_DATA_ROOT_ENV = "USE_TRAIN_TEST_DATA_ROOT"
_ENV_TEST_ROOT = os.environ.get(_TEST_DATA_ROOT_ENV, "").strip()
_USING_EXPLICIT_TEST_ROOT = bool(_ENV_TEST_ROOT)
_DEFAULT_TEST_ROOT = (
    Path(_ENV_TEST_ROOT).expanduser().resolve()
    if _USING_EXPLICIT_TEST_ROOT
    else Path("data/test")
)


def _resolve_test_data_root(project_root: Path | None = None) -> Path:
    base = PROJECT_ROOT if project_root is None else Path(project_root)
    return (
        _DEFAULT_TEST_ROOT
        if _USING_EXPLICIT_TEST_ROOT
        else (base / _DEFAULT_TEST_ROOT).resolve()
    )


DEFAULT_TEST_DATA_ROOT = _resolve_test_data_root()
DEFAULT_PATHS = {
    "aec_testset_root": _DEFAULT_TEST_ROOT / "AEC_Challenge_test",
    "aec_sim_testset_root": _DEFAULT_TEST_ROOT / "aec_sim_testset",
    "voicebank_clean_scp": _DEFAULT_TEST_ROOT / "voicebank_demand_16k/clean.scp",
    "voicebank_noisy_scp": _DEFAULT_TEST_ROOT / "voicebank_demand_16k/noisy.scp",
    "urgent_noisy_scp": _DEFAULT_TEST_ROOT
    / "urgent2026/blind_test_set_2026/noisy_16k.scp",
    "urgent_clean_scp": _DEFAULT_TEST_ROOT
    / "urgent2026/blind_test_set_2026/clean.scp",
    "urgent_noisy_dir": _DEFAULT_TEST_ROOT
    / "urgent2026/blind_test_set_2026/noisy_16k.scp",
    "urgent_clean_dir": _DEFAULT_TEST_ROOT
    / "urgent2026/blind_test_set_2026/clean.scp",
    "dns_challenge_noisy_dir": _DEFAULT_TEST_ROOT
    / "DNS_Challenge_test/blind_test_set_v3/noisy.scp",
}


def default_eval_output_root(project_root: Path | None = None) -> Path:
    """根据测试根返回 SE/AEC 评估结果根目录。"""
    base = PROJECT_ROOT if project_root is None else Path(project_root)
    return (
        DEFAULT_TEST_DATA_ROOT / "exp" / "eval"
        if _USING_EXPLICIT_TEST_ROOT
        else (base / "exp" / "eval").resolve()
    )


def _is_repository_placeholder(path: str | Path, default_key: str) -> bool:
    """判断路径是否仍为未配置数据时的仓库内占位路径。"""
    return (
        not _USING_EXPLICIT_TEST_ROOT
        and Path(path) == DEFAULT_PATHS[default_key]
    )


def _require_testset_source(
    path: str | Path,
    *,
    dataset: str,
    argument: str,
    source_kind: str,
    default_key: str,
) -> Path:
    """校验测试集源；仓库占位路径由上层记录为数据集缺失。"""
    configured = Path(path)
    is_placeholder = _is_repository_placeholder(configured, default_key)
    source = (
        (PROJECT_ROOT / configured).resolve()
        if is_placeholder
        else absolute_path(configured)
    )
    exists = source.is_file() if source_kind == "file" else source.is_dir()
    if exists:
        return source
    if is_placeholder:
        logger.warning(
            "%s source is not configured: %s=%s. Pass --%s or set %s.",
            dataset,
            argument,
            source,
            argument,
            _TEST_DATA_ROOT_ENV,
        )
        return source
    raise FileNotFoundError(
        f"{dataset} requires --{argument} ({source_kind}), but the path does not exist: "
        f"{source}. Pass --{argument} with a valid path or set {_TEST_DATA_ROOT_ENV}."
    )


AEC_OFFICIAL_TESTSETS = [
    "blind_test_set_icassp2023",
]

AEC_SUBSETS = ["doubletalk", "farend-singletalk", "nearend-singletalk"]
_SAFE_EVAL_OUTPUT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]*\Z")


def effective_config_fingerprint(
    config_or_mapping: str | Path | Mapping[str, Any],
) -> dict[str, str]:
    """返回实际 merged mapping 的稳定身份；路径调用保留兼容语义。"""
    if isinstance(config_or_mapping, Mapping):
        return effective_config_identity(config_or_mapping)
    _, identity = load_effective_config(config_or_mapping, loader=load_merged_config)
    return identity


def load_pipeline_config(
    config_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[Mapping[str, Any], dict[str, str]]:
    """pipeline 每进程唯一配置加载入口。"""
    expected = expected_sha256 or os.environ.get(
        "USE_TRAIN_EXPECTED_EFFECTIVE_CONFIG_SHA256"
    )
    return load_effective_config(
        config_path,
        expected=expected,
        loader=load_merged_config,
    )


def resolve_eval_output_name(
    output_name: Optional[str], *, exp_dir: Optional[str], config: Path
) -> str:
    """返回评估输出 basename；仅显式名称必须满足安全约束。"""
    if output_name is None:
        return Path(exp_dir).name if exp_dir else config.parent.name
    if (
        not isinstance(output_name, str)
        or _SAFE_EVAL_OUTPUT_NAME_RE.fullmatch(output_name) is None
    ):
        raise ValueError(f"评估输出名称必须是安全 basename: {output_name!r}")
    return output_name


def guard_explicit_eval_output(adapter: PipelineTaskAdapter) -> None:
    """在锁内保护显式评估输出，禁止覆盖不同 provenance 或未完成运行。

    空目录以及仅含 parent 私有日志/持久锁的目录可开始新运行。已提交且 provenance
    完全相同的评估允许重跑；所有 in-progress、损坏或孤立 lifecycle 残留均拒绝。
    """
    configured_output = Path(adapter.output_dir).expanduser()
    if configured_output.is_symlink():
        raise ValueError("显式评估输出目录不允许符号链接")
    output = configured_output.resolve()
    if output.exists() and not output.is_dir():
        raise ValueError(f"显式评估输出不是目录: {output}")
    if not output.exists():
        return

    allowed_private = {".inference.lock", "eval_pipeline.log"}
    children = {child.name: child for child in output.iterdir()}
    lifecycle_names = {CURRENT_FILENAME, ARTIFACT_FILENAME, adapter.summary_filename}
    present_lifecycle = lifecycle_names & set(children)
    if not present_lifecycle:
        unexpected = set(children) - allowed_private
        if unexpected:
            raise ValueError(
                "显式评估输出含未知残留，拒绝覆盖: "
                + ", ".join(sorted(unexpected))
            )
        return
    if CURRENT_FILENAME not in children:
        raise ValueError("显式评估输出 lifecycle artifact 不完整")
    current_path = output / CURRENT_FILENAME
    if current_path.is_symlink():
        raise ValueError("显式评估输出 CURRENT 不允许符号链接")
    try:
        current = json.loads(current_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("显式评估输出 CURRENT 无效") from exc
    if not isinstance(current, dict) or current.get("schema") != CURRENT_SCHEMA:
        raise ValueError("显式评估输出 CURRENT schema 无效")
    if current.get("status") == "in_progress":
        raise ValueError("显式评估输出已有 in-progress 运行，拒绝覆盖")
    if current.get("status") != "committed":
        raise ValueError("显式评估输出 CURRENT 未提交，拒绝覆盖")
    if present_lifecycle != lifecycle_names:
        raise ValueError("显式评估输出 lifecycle artifact 不完整")
    if any(children[name].is_symlink() for name in lifecycle_names):
        raise ValueError("显式评估输出 lifecycle artifact 不允许符号链接")
    try:
        envelope = load_committed_artifact(output)
    except (OSError, RuntimeError) as exc:
        raise ValueError("显式评估输出 committed artifact 无效") from exc
    if (
        envelope.get("schema") != ARTIFACT_SCHEMA
        or envelope.get("task") != adapter.task
        or envelope.get("pipeline") != adapter.pipeline
    ):
        raise ValueError("显式评估输出 task/pipeline provenance 不一致")
    provenance = envelope.get("provenance")
    expected_provenance = adapter._provenance()
    if not isinstance(provenance, dict):
        raise ValueError("显式评估输出缺少 provenance")
    for key in (
        "config",
        "checkpoint",
        "config_fingerprint",
        "checkpoint_fingerprint",
        "resolved_config_fingerprint",
    ):
        if provenance.get(key) != expected_provenance.get(key):
            raise ValueError(f"显式评估输出 {key} provenance 不一致")
    parameters = provenance.get("parameters")
    expected_parameters = expected_provenance.get("parameters")
    if not isinstance(parameters, dict) or not isinstance(expected_parameters, dict):
        raise ValueError("显式评估输出 parameters provenance 无效")
    output_name = expected_parameters.get("output_name")
    if parameters.get("output_name") != output_name:
        raise ValueError("显式评估输出 output_name provenance 不一致")
    summary = envelope.get("summary")
    if not isinstance(summary, dict) or summary.get("output_name") != output_name:
        raise ValueError("显式评估输出 summary output_name 不一致")


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------
def find_exp_assets(
    exp_dir: str | Path, *, prefer_model_only: bool = True
) -> tuple[Path, Path]:
    """Resolve (config_path, checkpoint_path) from an experiment directory.

    ``USE_TRAIN_EVAL_CONFIG`` 与 ``USE_TRAIN_EVAL_CHECKPOINT`` 仅供受控父进程成对
    传递已校验资产；缺一、文件缺失或仓外调用误设时立即失败。

    Config / checkpoint 成对发现（保证同源，fail-closed）:
      1. 仅接受 ``exp_dir`` 根目录中 ``config`` 与 ``best_model.tar`` 的完整 pair。
      2. 根目录不完整时，按 mtime 从 ``checkpoints/<timestamp>/`` 中选择同时含
         config 与 ``best_model.tar`` 的完整 pair。
      3. 禁止从 ``configs/`` 按实验名猜测 config，避免结构、STFT 或数据配置错配。
      4. 默认仅在已选 pair 的 checkpoint 同目录中，对通过安全 schema 与来源
         fingerprint 校验的 ``best_model.model_only.pt`` 使用派生权重。
    """
    override_config = os.environ.get("USE_TRAIN_EVAL_CONFIG")
    override_checkpoint = os.environ.get("USE_TRAIN_EVAL_CHECKPOINT")
    if bool(override_config) != bool(override_checkpoint):
        raise RuntimeError(
            "USE_TRAIN_EVAL_CONFIG 与 USE_TRAIN_EVAL_CHECKPOINT 必须成对设置"
        )
    if override_config and override_checkpoint:
        config_path = Path(override_config).expanduser().resolve(strict=True)
        checkpoint_path = Path(override_checkpoint).expanduser().resolve(strict=True)
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise RuntimeError("评估资产 override 必须指向普通文件")
        return config_path, checkpoint_path

    exp = absolute_path(exp_dir)
    config_names = ("resolved_config.yaml", "resolved_config_resumed.yaml", "config.yaml")

    def resolve_pair(directory: Path, checkpoint: Path) -> tuple[Path, Path] | None:
        if not checkpoint.is_file():
            return None
        config = next(
            (directory / name for name in config_names if (directory / name).is_file()),
            None,
        )
        return None if config is None else (config, checkpoint)

    # 只接受 config 与 checkpoint 位于同一 run 目录的完整 pair；禁止按名称跨目录猜测。
    asset_pairs: list[tuple[Path, Path]] = []
    root_pair = resolve_pair(exp, exp / "best_model.tar")
    if root_pair is not None:
        asset_pairs.append(root_pair)
    checkpoints_dir = exp / "checkpoints"
    if checkpoints_dir.is_dir():
        for run_dir in sorted(
            (path for path in checkpoints_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            pair = resolve_pair(run_dir, run_dir / "best_model.tar")
            if pair is not None:
                asset_pairs.append(pair)
    if not asset_pairs:
        raise FileNotFoundError(
            f"No same-run config/checkpoint pair found in {exp}. "
            "Please specify trusted --config and --checkpoint explicitly."
        )
    config_path, checkpoint_path = asset_pairs[0]

    if prefer_model_only:
        from tools.model_tools.extract_model_checkpoint import (
            checkpoint_family_from_config,
            default_model_only_path,
            validate_model_only_checkpoint,
        )

        model_only_path = default_model_only_path(checkpoint_path)
        if model_only_path.is_file():
            try:
                family = checkpoint_family_from_config(load_merged_config(str(config_path)))
                validate_model_only_checkpoint(
                    model_only_path,
                    source_path=checkpoint_path,
                    family=family,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning(
                    "Ignoring invalid model-only checkpoint %s: %s",
                    model_only_path,
                    exc,
                )
            else:
                return config_path, model_only_path

    return config_path, checkpoint_path


def prefer_model_only_checkpoint(
    config: Mapping[str, Any], checkpoint_path: str | Path
) -> Path:
    """使用已加载 mapping 校验并选择同目录 model-only checkpoint。"""
    from tools.model_tools.extract_model_checkpoint import (
        checkpoint_family_from_config,
        default_model_only_path,
        validate_model_only_checkpoint,
    )

    checkpoint = absolute_path(checkpoint_path)
    model_only_path = default_model_only_path(checkpoint)
    if not model_only_path.is_file():
        return checkpoint
    try:
        family = checkpoint_family_from_config(config)
        validate_model_only_checkpoint(
            model_only_path,
            source_path=checkpoint,
            family=family,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning(
            "Ignoring invalid model-only checkpoint %s: %s",
            model_only_path,
            exc,
        )
        return checkpoint
    return model_only_path


def detect_model_type(
    config_or_mapping: str | Path | Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> str:
    """Return 'se' or 'aec' from the supplied effective mapping."""
    from tasks.aec.models import AEC_MODEL_TYPES

    if isinstance(config_or_mapping, Mapping):
        cfg = config_or_mapping
        path_hint_source = config_path
    else:
        cfg, _ = load_pipeline_config(config_or_mapping)
        path_hint_source = config_or_mapping
    model_cfg = cfg.get("model", {})
    mtype = model_cfg.get("type", "")
    path_hint = model_cfg.get("path", "") or ""
    config_hint = "" if path_hint_source is None else str(path_hint_source).lower()
    if mtype in AEC_MODEL_TYPES:
        return "aec"
    if "aec" in path_hint.lower() or "aec" in config_hint:
        return "aec"
    return "se"


def detect_aec_subtype(
    config_or_mapping: str | Path | Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> str:
    """For AEC models, infer the legacy filename-based subtype hint."""
    if isinstance(config_or_mapping, Mapping):
        path_hint_source = config_path
    else:
        load_pipeline_config(config_or_mapping)
        path_hint_source = config_or_mapping
    name = "" if path_hint_source is None else Path(path_hint_source).stem.lower()
    return "joint" if "joint" in name else "unknown"


# ---------------------------------------------------------------------------
# SCP helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_output_uid(uid: str, *, allow_path_separators: bool = False) -> str:
    """将 UID 编码为单一输出文件名；仅 input_dir 相对路径可含分隔符。"""
    if not isinstance(uid, str) or not uid or uid in {".", ".."}:
        raise ValueError(f"invalid output uid: {uid!r}")
    if (
        (not allow_path_separators and any(character in uid for character in ("/", "\\")))
        or any(ord(character) < 32 or ord(character) == 127 for character in uid)
    ):
        raise ValueError(f"unsafe output uid: {uid!r}")
    encoded = uid.encode("utf-8").hex()
    if not encoded:
        raise ValueError(f"invalid output uid: {uid!r}")
    return f"uid-{encoded}"


def _zero_farend_source_files() -> dict[str, Path]:
    """返回会影响 zero-farend 推理结果的完整源码边界。"""
    return collect_source_tree_files(
        PROJECT_ROOT,
        [
            "core/stft.py",
            "core/models",
            "tasks/__init__.py",
            "tasks/descriptors.py",
            "tasks/registry.py",
            "core/utils/config.py",
            "core/utils/inference_artifacts.py",
            "core/inference",
            "core/utils/safe_load.py",
            "tasks/aec/models",
            "evaluate/checkpoint_loading.py",
            "evaluate/eval_pipeline_common.py",
            "tools/shared/scp_io.py",
        ],
    )


def _read_stable_file_bytes(path: str | Path) -> tuple[Path, bytes, dict]:
    """读取一个稳定文件快照，并返回与 ``fingerprint_file`` 同形的指纹。"""
    resolved = Path(path).expanduser().resolve(strict=True)
    before = resolved.stat()
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"file replaced before read: {resolved}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = resolved.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after_open) or identity(before) != identity(after):
        raise RuntimeError(f"file changed while reading: {resolved}")
    content = b"".join(chunks)
    return resolved, content, {
        "path": str(resolved),
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _directory_snapshot(root: Path, pattern: str, recursive: bool) -> tuple[list[Path], dict]:
    """枚举音频并仅用 ``stat-v1`` 生成 source snapshot（实现与 ``eval_io`` 共享）。"""
    return _directory_snapshot_shared(root, pattern, recursive)


def generate_scp_from_dir(
    input_dir: str | Path,
    pattern: str = "*.wav",
    scp_path: Optional[str | Path] = None,
    recursive: bool = True,
) -> Path:
    """仅按目录路径生成 2 列 SCP，并写入可校验 provenance sidecar。

    评估流水线要求非空源：空目录必须 fail-fast，而不是产出空清单；
    实现与 ``eval_io.generate_scp_from_dir`` 共享。
    """
    return _generate_scp_from_dir_shared(
        input_dir,
        pattern=pattern,
        scp_path=scp_path,
        recursive=recursive,
        require_non_empty=True,
    )


def write_generated_scp(
    entries: list[tuple[str, str]] | list[tuple[str, int, int, str]],
    scp_path: str | Path,
    *,
    source: str | Path,
    source_snapshot: str | dict,
) -> Path:
    """写入预组装 SCP；可信既有 4 列可复用，新路径清单使用 2 列。"""
    scp = Path(scp_path)
    provenance = {
        "kind": "generated",
        "source": str(Path(source).resolve()),
        "source_snapshot": source_snapshot,
    }
    if scp_provenance_fresh(scp, provenance):
        return scp
    uids = [entry[0] for entry in entries]
    if len(uids) != len(set(uids)):
        raise ValueError(f"duplicate uid while generating {scp}")
    four_col = bool(entries and len(entries[0]) == 4)
    write_scp_entries(entries, scp, four_col=four_col)
    write_scp_provenance(
        scp, build_scp_provenance(scp.read_bytes(), scp_path=scp, **provenance)
    )
    return scp


def copy_scp_with_provenance(source: str | Path, destination: str | Path) -> Path:
    """复制严格 2/4 列 SCP，并保持相对路径基于源 SCP 目录的语义。"""
    src, snapshot, source_fingerprint = _read_stable_file_bytes(source)
    dst = Path(destination)
    rows: list[str] = []
    seen: set[str] = set()
    try:
        source_text = snapshot.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"SCP must be UTF-8: {src}") from exc
    for lineno, raw in enumerate(source_text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = parse_scp_fields(line, src, lineno, allowed_columns=(2, 4))
        uid = fields[0]
        if uid in seen:
            raise ValueError(f"{src}:{lineno}: 重复 uid: {uid!r}")
        seen.add(uid)
        source_path = Path(fields[-1]).expanduser()
        resolved_path = (
            source_path.resolve()
            if source_path.is_absolute()
            else (src.parent / source_path).resolve()
        )
        normalized = (*fields[:-1], str(resolved_path))
        rows.append(" ".join(shlex.quote(field) for field in normalized))

    provenance = {
        "kind": "copied",
        "source": str(src),
        "source_fingerprint": source_fingerprint,
    }
    if scp_provenance_fresh(dst, provenance):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(rows) + ("\n" if rows else "")
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, dst)
    write_scp_provenance(
        dst, build_scp_provenance(content, scp_path=dst, **provenance)
    )
    logger.info("Copied SCP: %s → %s", src, dst)
    return dst


def _require_nonempty_scp(
    source: str | Path,
    *,
    dataset: str,
    require_provenance: bool = False,
) -> dict[str, Path]:
    """读取 SCP，并在发布前拒绝空源；标准化输入必须有有效 sidecar。"""
    source_path = Path(source).expanduser().resolve()
    if require_provenance:
        validate_scp_provenance(source_path)
    entries = read_scp(source_path)
    if not entries:
        raise ValueError(f"{dataset} SCP contains zero entries: {source_path}")
    return entries


def _consume_standardized_scp(
    source: str | Path,
    destination: str | Path,
    *,
    dataset: str,
) -> Path:
    """消费已有标准化 SCP；冲突目标不覆盖，目录扫描由调用方显式保留。"""
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()
    _require_nonempty_scp(src, dataset=dataset, require_provenance=True)
    if src == dst:
        return src
    _, _, source_fingerprint = _read_stable_file_bytes(src)
    expected = {
        "kind": "copied",
        "source": str(src),
        "source_fingerprint": source_fingerprint,
    }
    if dst.exists() or Path(f"{dst}.provenance.json").exists():
        if scp_provenance_fresh(dst, expected):
            return dst
        raise ValueError(
            f"{dataset} standardized SCP destination conflict; refusing to overwrite: {dst}"
        )
    return copy_scp_with_provenance(src, dst)


def compute_erle_for_scenario(
    mic_scp: str | Path,
    enh_scp: str | Path,
    ref_scp: str | Path | None,
    scenario: str,
    eps: float = 1e-10,
    echo_scp: str | Path | None = None,
) -> dict[str, float] | None:
    """Compute ERLE (Echo Return Loss Enhancement) for an AEC sim scenario.

    口径：
      - prob_echo_only / prob_noise_echo_only / farend-singletalk（target≈0）:
          ERLE = 10·log10(E[mic²] / E[enh² + eps])。
      - prob_target_echo / prob_target_noise_echo（doubletalk，target≠0）:
          必须提供 oracle echo stem（echo_scp）。投影法使用
          alpha_x=<x,echo>/||echo||²，ERLE =
          10·log10(alpha_mic²·‖echo‖² / (alpha_enh²·‖echo‖² + eps))。
          stem 只提供回声方向，输入/输出电平分别从 mic/enh 投影取得，
          因而对 stem 与 mic 间的 Layer D 增益差免疫，恒等映射为 0 dB。
          必须与近端 SI-SDR(nearend_ref, enh) 成对解读，避免将全静音误判为
          回声消除；无 echo_scp 时返回 None（N/A），不得回退至总误差口径。
      - doubletalk（无 oracle echo）: 返回 None（ERLE N/A）。
      - 其他场景: 无回声 → None。

    Returns:
        {"mean_erle_db": float, "n_samples": int}，doubletalk 场景额外含
        "mean_nearend_sisdr_db"；无法计算时返回 None。
    """
    if scenario not in _ERLE_SCENARIOS:
        return None

    is_dt = scenario in {"prob_target_echo", "prob_target_noise_echo"}
    # 盲测 doubletalk 无 oracle echo，能量比会把近端抑制误算为回声消除。
    if scenario == "doubletalk":
        return None
    # 无 oracle echo stem 时返回 None，而非抛出 FileNotFoundError。
    echo_map = read_scp(echo_scp) if echo_scp and Path(echo_scp).exists() else {}
    if is_dt and not echo_map:
        # 不得回退至 (mic-ref)/(enh-ref) 的总误差口径。
        logger.warning(
            "ERLE N/A for scenario=%s: oracle echo stem (echo.scp) missing; "
            "regenerate the sim testset with generator_version >= 2.2.",
            scenario,
        )
        return None

    mic_map = read_scp(mic_scp)
    enh_map = read_scp(enh_scp)
    ref_map = read_scp(ref_scp) if ref_scp else {}

    common_uids = sorted(set(mic_map.keys()) & set(enh_map.keys()))
    if not common_uids:
        return None

    sisdr_fn = None
    if is_dt:
        from core.metrics.intrusive import compute_sisdr as sisdr_fn

    erles: list[float] = []
    sisdrs: list[float] = []
    for uid in common_uids:
        try:
            mic_audio, _ = load_audio(str(mic_map[uid]))
            enh_audio, _ = load_audio(str(enh_map[uid]))
            minlen = min(len(mic_audio), len(enh_audio))
            if minlen == 0:
                continue
            mic_audio = mic_audio[:minlen]
            enh_audio = enh_audio[:minlen]

            if is_dt:
                # doubletalk（仿真）: 投影法 residual echo + 近端 SI-SDR 成对报告。
                # oracle echo stem 只提供回声"方向"；输入/输出电平各自从 mic/enh
                # 向该方向投影取得（自校准：ERLE = 20·log10|alpha_mic/alpha_enh|，
                # 对 stem 的 Layer D 增益差免疫，恒等映射恒为 0 dB）。
                if uid not in echo_map:
                    continue
                echo_audio, _ = load_audio(str(echo_map[uid]))
                e_len = min(len(echo_audio), minlen)
                if e_len == 0:
                    continue
                echo_audio = echo_audio[:e_len]
                mic_e = mic_audio[:e_len]
                enh_e = enh_audio[:e_len]
                echo_pow = float(np.sum(echo_audio ** 2))
                if echo_pow <= eps:
                    continue
                # alpha = <x, echo> / ||echo||²：x 在 echo 方向上的投影系数。
                # 近端语音与 echo 独立，投影只捕获回声分量（含 1/N 噪声底）。
                alpha_in = float(np.dot(mic_e, echo_audio)) / echo_pow
                alpha_out = float(np.dot(enh_e, echo_audio)) / echo_pow
                resid_in = alpha_in ** 2 * echo_pow
                resid_out = alpha_out ** 2 * echo_pow + eps
                if resid_in <= eps:
                    continue
                erle = 10.0 * np.log10(resid_in / resid_out)
                # 近端保持：SI-SDR(nearend_ref, enh)，与 ERLE 成对解读
                if uid in ref_map:
                    ref_audio, _ = load_audio(str(ref_map[uid]))
                    r_len = min(len(ref_audio), minlen)
                    if r_len > 0:
                        sd = sisdr_fn(ref_audio[:r_len], enh_audio[:r_len])
                        if sd is not None and np.isfinite(sd):
                            sisdrs.append(float(sd))
            else:
                # prob_echo_only / prob_noise_echo_only / farend-singletalk（target≈0）
                echo_in = mic_audio
                echo_out = enh_audio
                power_in = float(np.mean(echo_in ** 2))
                power_out = float(np.mean(echo_out ** 2)) + eps
                if power_out <= 0 or power_in <= 0:
                    continue
                erle = 10.0 * np.log10(power_in / power_out)

            if np.isfinite(erle):
                erles.append(float(erle))
        except (OSError, ValueError, IndexError) as exc:
            # Expected failure families: missing/corrupt audio, length/shape
            # mismatch, or invalid arithmetic (e.g. empty arrays).
            logger.warning("ERLE computation failed for uid=%s scenario=%s: %s", uid, scenario, exc)
            continue
        except Exception:
            # Unexpected errors should not be silently swallowed: log the full
            # traceback so they can be investigated, but keep evaluating other
            # samples.
            logger.exception("Unexpected ERLE computation error for uid=%s scenario=%s", uid, scenario)
            continue

    if not erles:
        return None
    out = {"mean_erle_db": float(np.mean(erles)), "n_samples": len(erles)}
    if sisdrs:
        out["mean_nearend_sisdr_db"] = float(np.mean(sisdrs))
    return out


def compute_noise_reduction_for_scenario(
    mic_scp: str | Path,
    enh_scp: str | Path,
    scenario: str,
    eps: float = 1e-10,
) -> dict[str, float] | None:
    """Compute noise reduction (dB) for the prob_noise_only scenario.

    prob_noise_only 的 mic 仅含近端噪声、目标为静音，理想输出为全零：
      NR = 10·log10(E[mic²] / E[enh² + eps])
    其它场景无意义，返回 None。

    Returns {"mean_nr_db": float, "n_samples": int} or None.
    """
    if scenario not in _NOISE_REDUCTION_SCENARIOS:
        return None

    mic_map = read_scp(mic_scp)
    enh_map = read_scp(enh_scp)
    common_uids = sorted(set(mic_map.keys()) & set(enh_map.keys()))
    if not common_uids:
        return None

    nrs: list[float] = []
    for uid in common_uids:
        try:
            mic_audio, _ = load_audio(str(mic_map[uid]))
            enh_audio, _ = load_audio(str(enh_map[uid]))
            minlen = min(len(mic_audio), len(enh_audio))
            if minlen == 0:
                continue
            power_in = float(np.mean(mic_audio[:minlen] ** 2))
            power_out = float(np.mean(enh_audio[:minlen] ** 2)) + eps
            if power_in <= 0 or power_out <= 0:
                continue
            nr = 10.0 * np.log10(power_in / power_out)
            if np.isfinite(nr):
                nrs.append(float(nr))
        except (OSError, ValueError, IndexError) as exc:
            logger.warning("Noise reduction computation failed for uid=%s: %s", uid, exc)
            continue
        except Exception:
            logger.exception("Unexpected noise reduction computation error for uid=%s", uid)
            continue

    if not nrs:
        return None
    return {"mean_nr_db": float(np.mean(nrs)), "n_samples": len(nrs)}


def _compute_registered_erle(
    mic_scp: str | Path,
    enhanced_scp: str | Path,
    *,
    scenario: str,
    ref_scp: str | Path | None = None,
    echo_scp: str | Path | None = None,
    eps: float = 1e-10,
) -> Mapping[str, Any] | None:
    """将现有 ERLE API 适配到场景指标 registry。"""
    return compute_erle_for_scenario(
        mic_scp,
        enhanced_scp,
        ref_scp,
        scenario,
        eps=eps,
        echo_scp=echo_scp,
    )


def _compute_registered_noise_reduction(
    mic_scp: str | Path,
    enhanced_scp: str | Path,
    *,
    scenario: str,
    ref_scp: str | Path | None = None,
    echo_scp: str | Path | None = None,
    eps: float = 1e-10,
) -> Mapping[str, Any] | None:
    """将现有噪声抑制 API 适配到场景指标 registry。"""
    del ref_scp, echo_scp
    return compute_noise_reduction_for_scenario(
        mic_scp,
        enhanced_scp,
        scenario,
        eps=eps,
    )


register_scenario_metric(
    "erle",
    scenarios=_ERLE_SCENARIOS,
    compute=_compute_registered_erle,
)
register_scenario_metric(
    "noise_reduction",
    scenarios=_NOISE_REDUCTION_SCENARIOS,
    compute=_compute_registered_noise_reduction,
)


# ---------------------------------------------------------------------------
# Unified testset SCP preparation (all SCPs under the configured test root)
# ---------------------------------------------------------------------------
def prepare_testset_scps(
    aec_testset_root: Optional[str | Path] = None,
    aec_subsets: Optional[list[str]] = None,
    urgent_noisy_scp: Optional[str | Path] = None,
    urgent_clean_scp: Optional[str | Path] = None,
    urgent_noisy_dir: Optional[str | Path] = None,
    urgent_clean_dir: Optional[str | Path] = None,
    voicebank_noisy_scp: Optional[str | Path] = None,
    voicebank_clean_scp: Optional[str | Path] = None,
    dns_challenge_noisy_dir: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
    selected_datasets: Optional[list[str]] = None,
) -> dict[str, dict]:
    """Ensure standardized SCPs exist only for the selected testsets.

    ``selected_datasets=None`` 保留独立调用的全量兼容语义；pipeline 必须传入
    本次实际选择的数据集。仓库内占位默认值缺失时仅报告为不可用；显式路径
    以及由 ``USE_TRAIN_TEST_DATA_ROOT`` 派生的路径缺失时直接失败。未提供
    ``output_root`` 时写入 ``DEFAULT_TEST_DATA_ROOT``，显式 output root 优先。

    Returns a nested dict mapping dataset tags to their SCP paths, e.g.:
        {
            "voicebank": {"noisy": Path, "clean": Path},
            "urgent": {"noisy": Path, "clean": Path},
            "aec_icassp2022_doubletalk": {"mic": Path, "lpb": Path},
            ...
        }
    """
    result: dict[str, dict] = {}
    selected = None if selected_datasets is None else set(selected_datasets)

    def wants(dataset: str) -> bool:
        if selected is None:
            return True
        if dataset == "aec_official":
            return any(
                item == "aec_official"
                or item.startswith("aec_icassp")
                or item.startswith("aec_interspeech")
                for item in selected
            )
        return dataset in selected

    if output_root is None:
        out_root = DEFAULT_TEST_DATA_ROOT
    else:
        configured_output = Path(output_root)
        out_root = (
            (PROJECT_ROOT / configured_output).resolve()
            if configured_output == Path("data/test")
            else absolute_path(configured_output)
        )
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- VoiceBank ----
    if wants("voicebank") and voicebank_noisy_scp and voicebank_clean_scp:
        src_noisy = _require_testset_source(
            voicebank_noisy_scp,
            dataset="VoiceBank",
            argument="voicebank_noisy_scp",
            source_kind="file",
            default_key="voicebank_noisy_scp",
        )
        src_clean = _require_testset_source(
            voicebank_clean_scp,
            dataset="VoiceBank",
            argument="voicebank_clean_scp",
            source_kind="file",
            default_key="voicebank_clean_scp",
        )
        if src_noisy.is_file() and src_clean.is_file():
            vb_dir = out_root / "voicebank_demand_16k"
            vb_dir.mkdir(parents=True, exist_ok=True)
            dst_noisy = vb_dir / "noisy.scp"
            dst_clean = vb_dir / "clean.scp"
            copy_scp_with_provenance(src_noisy, dst_noisy)
            copy_scp_with_provenance(src_clean, dst_clean)
            result["voicebank"] = {"noisy": dst_noisy, "clean": dst_clean}

    # ---- URGENT ----
    if wants("urgent"):
        if bool(urgent_noisy_scp) != bool(urgent_clean_scp):
            raise ValueError(
                "URGENT SCP inputs must be provided as a pair: "
                "--urgent_noisy_scp and --urgent_clean_scp"
            )
        if urgent_noisy_scp and urgent_clean_scp:
            src_noisy = _require_testset_source(
                urgent_noisy_scp,
                dataset="URGENT",
                argument="urgent_noisy_scp",
                source_kind="file",
                default_key="urgent_noisy_scp",
            )
            src_clean = _require_testset_source(
                urgent_clean_scp,
                dataset="URGENT",
                argument="urgent_clean_scp",
                source_kind="file",
                default_key="urgent_clean_scp",
            )
            if src_noisy.is_file() and src_clean.is_file():
                noisy_entries = _require_nonempty_scp(
                    src_noisy,
                    dataset="URGENT noisy",
                    require_provenance=Path(f"{src_noisy}.provenance.json").exists(),
                )
                clean_entries = _require_nonempty_scp(
                    src_clean,
                    dataset="URGENT clean",
                    require_provenance=Path(f"{src_clean}.provenance.json").exists(),
                )
                if set(noisy_entries) != set(clean_entries):
                    raise ValueError(
                        "URGENT clean/noisy SCP UID sets do not match: "
                        f"noisy={len(noisy_entries)}, clean={len(clean_entries)}"
                    )
                urg_dir = out_root / "urgent2026" / "blind_test_set_2026"
                urg_dir.mkdir(parents=True, exist_ok=True)
                if Path(f"{src_noisy}.provenance.json").exists():
                    noisy_scp = _consume_standardized_scp(
                        src_noisy, urg_dir / "noisy_16k.scp", dataset="URGENT noisy"
                    )
                    clean_scp = _consume_standardized_scp(
                        src_clean, urg_dir / "clean.scp", dataset="URGENT clean"
                    )
                else:
                    noisy_scp = copy_scp_with_provenance(src_noisy, urg_dir / "noisy_16k.scp")
                    clean_scp = copy_scp_with_provenance(src_clean, urg_dir / "clean.scp")
                result["urgent"] = {"noisy": noisy_scp, "clean": clean_scp}
        if "urgent" not in result and urgent_noisy_dir and urgent_clean_dir:
            noisy_kind = "file" if Path(urgent_noisy_dir).suffix == ".scp" else "directory"
            clean_kind = "file" if Path(urgent_clean_dir).suffix == ".scp" else "directory"
            noisy_source = _require_testset_source(
                urgent_noisy_dir,
                dataset="URGENT",
                argument="urgent_noisy_dir",
                source_kind=noisy_kind,
                default_key="urgent_noisy_dir",
            )
            clean_source = _require_testset_source(
                urgent_clean_dir,
                dataset="URGENT",
                argument="urgent_clean_dir",
                source_kind=clean_kind,
                default_key="urgent_clean_dir",
            )
            if noisy_source.is_file() and clean_source.is_file():
                noisy_entries = _require_nonempty_scp(
                    noisy_source, dataset="URGENT noisy", require_provenance=True
                )
                clean_entries = _require_nonempty_scp(
                    clean_source, dataset="URGENT clean", require_provenance=True
                )
                if set(noisy_entries) != set(clean_entries):
                    raise ValueError("URGENT clean/noisy SCP UID sets do not match")
                urg_dir = out_root / "urgent2026" / "blind_test_set_2026"
                urg_dir.mkdir(parents=True, exist_ok=True)
                result["urgent"] = {
                    "noisy": _consume_standardized_scp(
                        noisy_source, urg_dir / "noisy_16k.scp", dataset="URGENT noisy"
                    ),
                    "clean": _consume_standardized_scp(
                        clean_source, urg_dir / "clean.scp", dataset="URGENT clean"
                    ),
                }
            elif noisy_source.is_dir() and clean_source.is_dir():
                urg_dir = out_root / "urgent2026" / "blind_test_set_2026"
                urg_dir.mkdir(parents=True, exist_ok=True)
                noisy_scp = urg_dir / "noisy_16k.scp"
                clean_scp = urg_dir / "clean.scp"
                generate_scp_from_dir(
                    noisy_source, pattern="*.wav", scp_path=noisy_scp, recursive=False
                )
                generate_scp_from_dir(
                    clean_source, pattern="**/*.flac", scp_path=clean_scp, recursive=True
                )
                result["urgent"] = {"noisy": noisy_scp, "clean": clean_scp}

    # ---- AEC Official ----
    if wants("aec_official") and aec_testset_root:
        aec_source = _require_testset_source(
            aec_testset_root,
            dataset="AEC official",
            argument="aec_testset_root",
            source_kind="directory",
            default_key="aec_testset_root",
        )
        aec_root = out_root / "AEC_Challenge_test"
        aec_root.mkdir(parents=True, exist_ok=True)
        for challenge in AEC_OFFICIAL_TESTSETS:
            challenge_root = aec_source / challenge
            if not challenge_root.is_dir() and aec_source.name == challenge:
                challenge_root = aec_source
            for subset in (aec_subsets or ["doubletalk"]):
                flat_dir = challenge_root / subset
                if not flat_dir.exists():
                    logger.warning("AEC official subset is missing: %s", flat_dir)
                    continue
                out_dir = aec_root / challenge / subset
                out_dir.mkdir(parents=True, exist_ok=True)
                mic_scp = out_dir / "mic.scp"
                lpb_scp = out_dir / "lpb.scp"
                source_mic_scp = flat_dir / "mic.scp"
                source_lpb_scp = flat_dir / "lpb.scp"
                if source_mic_scp.exists() or source_lpb_scp.exists():
                    if not source_mic_scp.is_file() or not source_lpb_scp.is_file():
                        raise ValueError(
                            f"AEC standardized SCP pair is incomplete: {flat_dir}"
                        )
                    mic_entries = _require_nonempty_scp(
                        source_mic_scp, dataset="AEC mic", require_provenance=True
                    )
                    lpb_entries = _require_nonempty_scp(
                        source_lpb_scp, dataset="AEC lpb", require_provenance=True
                    )
                    if set(mic_entries) != set(lpb_entries):
                        raise ValueError(
                            f"AEC mic/lpb SCP UID sets do not match: {flat_dir}"
                        )
                    result[f"{challenge}_{subset}"] = {
                        "mic": _consume_standardized_scp(
                            source_mic_scp, mic_scp, dataset="AEC mic"
                        ),
                        "lpb": _consume_standardized_scp(
                            source_lpb_scp, lpb_scp, dataset="AEC lpb"
                        ),
                    }
                    continue

                mic_files, mic_snapshot = _directory_snapshot(flat_dir.resolve(), "*_mic.wav", False)
                lpb_files, lpb_snapshot = _directory_snapshot(flat_dir.resolve(), "*_lpb.wav", False)
                snapshot = {
                    "algorithm": "stat-v1",
                    "digest": hashlib.sha256(
                        json.dumps(
                            {"mic": mic_snapshot, "farend": lpb_snapshot},
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                }
                mic_entries: list[tuple[str, str]] = []
                lpb_entries: list[tuple[str, str]] = []
                lpb_set = set(lpb_files)
                for mic in mic_files:
                    base = mic.stem[:-4]
                    lpb = flat_dir.resolve() / f"{base}_lpb.wav"
                    if lpb not in lpb_set:
                        continue
                    mic_entries.append((base, str(mic)))
                    lpb_entries.append((base, str(lpb)))
                write_generated_scp(mic_entries, mic_scp, source=flat_dir, source_snapshot=snapshot)
                write_generated_scp(lpb_entries, lpb_scp, source=flat_dir, source_snapshot=snapshot)
                logger.info("Generated AEC SCPs: %s / %s (%d pairs)", mic_scp, lpb_scp, len(mic_entries))

                result[f"{challenge}_{subset}"] = {"mic": mic_scp, "lpb": lpb_scp}

    # ---- DNS Challenge (blind testset, no clean reference) ----
    if wants("dns_challenge") and dns_challenge_noisy_dir:
        dns_kind = (
            "file"
            if Path(dns_challenge_noisy_dir).suffix == ".scp"
            else "directory"
        )
        dns_source = _require_testset_source(
            dns_challenge_noisy_dir,
            dataset="DNS Challenge",
            argument="dns_challenge_noisy_dir",
            source_kind=dns_kind,
            default_key="dns_challenge_noisy_dir",
        )
        dns_dir = out_root / "DNS_Challenge_test" / "blind_test_set_v3"
        dns_dir.mkdir(parents=True, exist_ok=True)
        noisy_scp = dns_dir / "noisy.scp"
        if dns_source.is_file():
            result["dns_challenge"] = {
                "noisy": _consume_standardized_scp(
                    dns_source, noisy_scp, dataset="DNS Challenge"
                )
            }
        elif dns_source.is_dir():
            generate_scp_from_dir(
                dns_source, pattern="*.wav", scp_path=noisy_scp, recursive=False
            )
            result["dns_challenge"] = {"noisy": noisy_scp}

    return result


def _adapter_artifact_fingerprint(
    config_path: Path,
    checkpoint_path: Path,
    *,
    effective_identity: Mapping[str, Any] | None,
    mode: str,
    parameters: Mapping[str, Any] | None = None,
    precomputed: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """为 evaluate 任务 adapter 建立与公共 runner 兼容的源身份。"""
    identity_parameters = {
        "mode": mode,
        "effective_config_identity": dict(effective_identity or {}),
    }
    if parameters:
        identity_parameters.update(dict(parameters))
    source_files = inference_source_files(
        PROJECT_ROOT, is_aec=mode.startswith("aec")
    )
    source_files["source:evaluate/eval_pipeline_common.py"] = Path(__file__)
    return build_content_fingerprint(
        {
            "config": config_path,
            "checkpoint": checkpoint_path,
            "evaluate_entry": Path(__file__),
            **source_files,
        },
        parameters=identity_parameters,
        precomputed=precomputed,
    )


def _scp_items(
    source_scp: str | Path,
    output_dir: Path,
    *,
    roles: tuple[str, ...],
) -> list[Any]:
    """将单输入 SCP 转成公共 runner 的 ``InferenceItem``。"""
    from core.inference import FileInferenceItem

    entries = read_scp(source_scp)
    return [
        FileInferenceItem(
            uid=uid,
            inputs={role: str(entries[uid]) for role in roles},
            output=(output_dir / f"{uid}.wav").resolve(),
        )
        for uid in sorted(entries)
    ]


def _paired_scp_items(
    mic_scp: str | Path,
    far_scp: str | Path,
    output_dir: Path,
) -> list[Any]:
    """构造 AEC mic/farend 双输入的公共 runner item。"""
    from core.inference import FileInferenceItem

    mic = read_scp(mic_scp)
    far = read_scp(far_scp)
    if set(mic) != set(far):
        raise ValueError("AEC mic/farend SCP UID 集合不一致")
    return [
        FileInferenceItem(
            uid=uid,
            inputs={"mic": str(mic[uid]), "farend": str(far[uid])},
            output=(output_dir / f"{uid}.wav").resolve(),
        )
        for uid in sorted(mic)
    ]


def _run_se_waveform_adapter_inference(
    *,
    config_path: Path,
    checkpoint_path: Path,
    input_scp: str | Path,
    output_path: Path,
    device: str,
    effective_identity: Mapping[str, Any] | None,
    allow_partial: bool,
    skip_existing: bool,
    chunk_seconds: float,
) -> Path:
    """将 waveform-native SE 直接接入公共 runner。

    PASE/StuPASE/UniPASE 等模型的权重由配置中的 ``*_ckpt_path`` artifact
    注入；evaluate 传入的训练 checkpoint 仅保留在调用合同中，不参与模型加载。
    """
    import inference as legacy_inference
    from tasks.se.models import build_model

    cfg, resolved_identity = load_pipeline_config(
        config_path,
        expected_sha256=(effective_identity or {}).get("sha256")
        if isinstance(effective_identity, Mapping)
        else None,
    )
    if cfg.get("stft"):
        raise ValueError("waveform-native SE requires no top-level stft config")
    if not (cfg.get("meta") or {}).get("allow_no_stft", False):
        raise ValueError("waveform-native SE requires meta.allow_no_stft=true")

    model_cfg = deepcopy(cfg.get("model", {}))
    checkpoint_files = configured_checkpoint_files(model_cfg)
    config_fingerprint = fingerprint_file(config_path)
    checkpoint_fingerprints = {
        name: fingerprint_file(path) for name, path in checkpoint_files.items()
    }
    model = build_model(model_cfg, stft_cfg=None).to(device).eval()
    source_files = legacy_inference.inference_source_files(
        Path(__file__).resolve().parents[1], is_aec=False
    )
    source_files["evaluate/eval_pipeline_common.py"] = Path(__file__)
    artifact_fingerprint = build_artifact_fingerprint(
        type("WaveformArgs", (), {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "sr": None,
            "chunk_seconds": float(chunk_seconds),
            "aec_stream_align": False,
            "aec_no_stream_align": False,
            "aec_review_multichannel": False,
            "cpu_long_audio_seconds": 0.0,
        })(),
        cfg,
        project_root=Path(__file__).resolve().parents[1],
        inference_entry=Path(legacy_inference.__file__),
        is_aec=False,
        is_waveform_native=True,
        effective_device=str(device),
        source_files=source_files,
        effective_identity=resolved_identity,
        precomputed_files={"config": config_fingerprint, **checkpoint_fingerprints},
    )
    verify_configured_checkpoint_fingerprints(checkpoint_files, artifact_fingerprint)

    multi_sr = legacy_inference._se_model_accepts_fs(model)
    fixed_sr = int((cfg.get("data") or {}).get("sample_rate") or 16000)
    adapter = legacy_inference._SEInferenceAdapter(
        model,
        device,
        multi_sr=multi_sr and not (cfg.get("data") or {}).get("sample_rate"),
        model_fixed_sr=fixed_sr,
        chunk_seconds=chunk_seconds,
    )
    items = _scp_items(input_scp, output_path, roles=("noisy",))
    run_adapter_inference(
        adapter=adapter,
        config=config_path,
        checkpoint=None,
        output_dir=output_path,
        items=items,
        artifact_fingerprint=artifact_fingerprint,
        allow_partial=allow_partial,
        skip_existing=skip_existing,
        metadata={"evaluate_entry": "se_waveform_adapter", "waveform_native": True},
    )
    return output_path


def _run_se_adapter_inference(
    *,
    config_path: Path,
    checkpoint_path: Path,
    input_scp: str | Path,
    output_path: Path,
    device: str,
    effective_identity: Mapping[str, Any] | None,
    allow_partial: bool,
) -> Path:
    """单设备 SE 入口：构造任务 adapter 并直接交给公共 runner。"""
    import inference as legacy_inference
    from tasks.se.inference import SEInferenceAdapter

    cfg, resolved_identity = load_pipeline_config(
        config_path,
        expected_sha256=(effective_identity or {}).get("sha256")
        if isinstance(effective_identity, Mapping)
        else None,
    )
    stft_cfg = cfg.get("stft")
    if not isinstance(stft_cfg, Mapping) or not stft_cfg:
        raise ValueError("SE adapter inference requires a top-level stft config")
    sample_rate = cfg.get("data", {}).get("sample_rate")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("SE adapter inference requires data.sample_rate")
    config_fingerprint = fingerprint_file(config_path)
    checkpoint_fingerprint = fingerprint_file(checkpoint_path)
    model = legacy_inference._load_model(
        cfg,
        checkpoint_path,
        device,
        stft_cfg=dict(stft_cfg),
        expected_checkpoint_fingerprint=checkpoint_fingerprint,
    )
    adapter = SEInferenceAdapter(
        model,
        stft=dict(stft_cfg),
        sample_rate=sample_rate,
    ).to(device)
    items = _scp_items(input_scp, output_path, roles=("noisy",))
    run_adapter_inference(
        adapter=adapter,
        config=config_path,
        checkpoint=checkpoint_path,
        output_dir=output_path,
        items=items,
        artifact_fingerprint=_adapter_artifact_fingerprint(
            config_path, checkpoint_path,
            effective_identity=resolved_identity,
            mode="se",
            precomputed={"config": config_fingerprint, "checkpoint": checkpoint_fingerprint},
        ),
        allow_partial=allow_partial,
        metadata={"evaluate_entry": "se_adapter"},
    )
    return output_path


def _run_aec_adapter_inference(
    *,
    config_path: Path,
    checkpoint_path: Path,
    mic_scp: str | Path,
    far_scp: str | Path,
    output_path: Path,
    device: str,
    effective_identity: Mapping[str, Any] | None,
    allow_partial: bool,
    no_stream_align: bool = False,
    skip_existing: bool = False,
) -> Path:
    """单设备 AEC 入口：构造 dict-spectrum 任务 adapter 并交给公共 runner。"""
    import inference as legacy_inference
    from tasks.aec.inference import (
        AECFrontendConfig,
        AECInferenceAdapter,
        resolve_aec_frontend_config,
    )

    cfg, resolved_identity = load_pipeline_config(
        config_path,
        expected_sha256=(effective_identity or {}).get("sha256")
        if isinstance(effective_identity, Mapping)
        else None,
    )
    stft_cfg = cfg.get("stft")
    sample_rate = cfg.get("data", {}).get("sample_rate")
    if not isinstance(stft_cfg, Mapping) or not stft_cfg:
        raise ValueError("AEC adapter inference requires a top-level stft config")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("AEC adapter inference requires data.sample_rate")
    config_fingerprint = fingerprint_file(config_path)
    checkpoint_fingerprint = fingerprint_file(checkpoint_path)
    model = legacy_inference._load_model(
        cfg,
        checkpoint_path,
        device,
        stft_cfg=dict(stft_cfg),
        expected_checkpoint_fingerprint=checkpoint_fingerprint,
    )
    frontend_config = resolve_aec_frontend_config(cfg)
    if no_stream_align:
        frontend_values = frontend_config.to_dict()
        frontend_values["delay_alignment"]["enabled"] = False
        frontend_config = AECFrontendConfig.from_dict(frontend_values)
    adapter = AECInferenceAdapter(
        model,
        stft=dict(stft_cfg),
        sample_rate=sample_rate,
        frontend=frontend_config,
    )
    items = _paired_scp_items(mic_scp, far_scp, output_path)
    run_adapter_inference(
        adapter=adapter,
        config=config_path,
        checkpoint=checkpoint_path,
        output_dir=output_path,
        items=items,
        artifact_fingerprint=_adapter_artifact_fingerprint(
            config_path,
            checkpoint_path,
            effective_identity=resolved_identity,
            mode="aec_paired",
            parameters={
                "aec_no_stream_align": bool(no_stream_align),
            },
            precomputed={"config": config_fingerprint, "checkpoint": checkpoint_fingerprint},
        ),
        allow_partial=allow_partial,
        skip_existing=skip_existing,
        metadata={
            "evaluate_entry": "aec_adapter",
            "aec_no_stream_align": bool(no_stream_align),
        },
    )
    return output_path


# ---------------------------------------------------------------------------
# Inference wrappers (subprocess → inference.py)
# ---------------------------------------------------------------------------
def validate_inference_artifacts(
    output_dir: str | Path,
    *,
    config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    effective_config: dict | None = None,
    expected_effective_config_identity: dict | str | None = None,
) -> tuple[dict, list[InferenceResult]]:
    """校验标准 manifest、effective mapping 与输出结构身份。"""
    return _validate_inference_artifacts(
        output_dir,
        config=config,
        checkpoint=checkpoint,
        effective_config=effective_config,
        expected_effective_config_identity=expected_effective_config_identity,
    )


def validate_evaluation_precommit_snapshot(
    summary: Mapping[str, Any],
    *,
    config: str | Path,
    checkpoint: str | Path,
    expected_effective_config_identity: Mapping[str, Any],
) -> None:
    """重新加载 effective config，并复核 summary 引用的全部推理 manifest。"""
    expected_sha256 = expected_effective_config_identity.get("sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise RuntimeError("evaluation precommit effective config identity is invalid")
    current_config, current_identity = load_pipeline_config(
        config, expected_sha256=expected_sha256
    )
    validate_summary_inference_identities(
        summary,
        config=config,
        checkpoint=checkpoint,
        effective_config=dict(current_config),
        expected_effective_config_identity=current_identity,
    )


def validate_summary_inference_identities(
    summary: Mapping[str, Any],
    *,
    config: str | Path,
    checkpoint: str | Path,
    effective_config: dict,
    expected_effective_config_identity: Mapping[str, Any] | str,
) -> None:
    """在发布评估结果前复核本轮摘要引用的全部推理 manifest。"""
    seen: set[Path] = set()
    for dataset in summary.get("datasets", {}).values():
        if not isinstance(dataset, Mapping) or dataset.get("status") == "error":
            continue
        enhanced_dirs: list[Path] = []
        enhanced_dir = dataset.get("enhanced_dir")
        if isinstance(enhanced_dir, str):
            enhanced_dirs.append(Path(enhanced_dir))
        eval_root = dataset.get("eval_root")
        scenarios = dataset.get("scenarios")
        if isinstance(eval_root, str) and isinstance(scenarios, list):
            enhanced_dirs.extend(Path(eval_root) / str(item) / "enhanced" for item in scenarios)
        for output_dir in enhanced_dirs:
            resolved = output_dir.expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            _validate_inference_artifacts(
                resolved,
                config=config,
                checkpoint=checkpoint,
                effective_config=effective_config,
                expected_effective_config_identity=expected_effective_config_identity,
            )


def _validate_fresh_inference_manifest(
    output_dir: str | Path,
    config: str | Path,
    checkpoint: str | Path,
    started_ns: int,
    expected_effective_config_identity: Mapping[str, Any] | str | None = None,
) -> None:
    manifest_path = Path(output_dir) / "run_manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_mtime_ns < started_ns:
        raise RuntimeError(f"inference did not produce a fresh manifest: {manifest_path}")
    try:
        validate_inference_artifacts(
            output_dir,
            config=config,
            checkpoint=checkpoint,
            expected_effective_config_identity=expected_effective_config_identity,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"invalid inference artifacts in {output_dir}: {exc}") from exc


def run_inference_se(
    config: str | Path,
    checkpoint: str | Path,
    input_dir: Optional[str | Path] = None,
    input_scp: Optional[str | Path] = None,
    output_dir: str | Path = "",
    device: str = "cuda:0",
    sr: Optional[int] = None,
    allow_partial: bool = False,
    devices: Optional[list[str] | tuple[str, ...] | str] = None,
    inference_devices: Optional[list[str] | tuple[str, ...] | str] = None,
    expected_effective_config_identity: Mapping[str, Any] | str | None = None,
    skip_existing: bool = False,
    chunk_seconds: float = 0.0,
) -> Path:
    """Run SE inference; multiple ``devices`` only shard model inference."""
    if devices is not None and inference_devices is not None:
        raise ValueError("devices 与 inference_devices 只能提供一个")
    selected_devices = inference_devices if inference_devices is not None else devices
    if selected_devices is not None:
        from evaluate.multi_gpu_inference import (
            normalize_inference_devices,
            run_se_inference_multi_gpu,
        )
        normalized = normalize_inference_devices(selected_devices, fallback=device)
        if len(normalized) > 1:
            if input_scp is None:
                raise ValueError("多 GPU SE 编排需要 input_scp")
            return run_se_inference_multi_gpu(
                config, checkpoint, input_scp, output_dir, normalized,
                sr=sr,
                allow_partial=allow_partial,
                expected_effective_config_identity=expected_effective_config_identity,
            )
        device = normalized[0]
    config_path = absolute_path(config)
    checkpoint_path = absolute_path(checkpoint)
    output_path = absolute_path(output_dir)
    cfg_probe, _ = load_pipeline_config(config_path)
    if input_scp is not None and not cfg_probe.get("stft") and (cfg_probe.get("meta") or {}).get("allow_no_stft", False):
        return _run_se_waveform_adapter_inference(
            config_path=config_path, checkpoint_path=checkpoint_path,
            input_scp=input_scp, output_path=output_path, device=device,
            effective_identity=expected_effective_config_identity,
            allow_partial=allow_partial, skip_existing=skip_existing,
            chunk_seconds=chunk_seconds,
        )
    if input_scp is not None and isinstance(cfg_probe.get("stft"), Mapping) and cfg_probe.get("stft"):
        return _run_se_adapter_inference(
            config_path=config_path, checkpoint_path=checkpoint_path,
            input_scp=input_scp or input_dir, output_path=output_path, device=device,
            effective_identity=expected_effective_config_identity,
            allow_partial=allow_partial,
        )
    cmd = [
        sys.executable, str(repository_script("inference.py")),
        "-C", str(config_path),
        "--checkpoint", str(checkpoint_path),
        "--output_dir", str(output_path),
        "--device", device,
    ]
    if expected_effective_config_identity is not None:
        expected_sha256 = (
            expected_effective_config_identity
            if isinstance(expected_effective_config_identity, str)
            else expected_effective_config_identity["sha256"]
        )
        cmd += ["--expected-effective-config-sha256", expected_sha256]
    if skip_existing:
        cmd.append("--skip-existing")
    if input_scp:
        cmd += ["--input_scp", str(absolute_path(input_scp))]
    elif input_dir:
        cmd += ["--input_dir", str(absolute_path(input_dir))]
    else:
        raise ValueError("Provide input_dir or input_scp")
    if sr is not None:
        cmd += ["--sr", str(sr)]
    if allow_partial:
        cmd.append("--allow-partial")

    logger.info("[Inference-SE] %s", " ".join(cmd))
    started_ns = time.time_ns()
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=EVAL_SUBPROCESS_TIMEOUT_S,
            start_new_session=True,
            preexec_fn=death_signal_preexec,
        )
    except subprocess.CalledProcessError:
        _validate_fresh_inference_manifest(
            output_path,
            config_path,
            checkpoint_path,
            started_ns,
            expected_effective_config_identity,
        )
        raise
    _validate_fresh_inference_manifest(
        output_path,
        config_path,
        checkpoint_path,
        started_ns,
        expected_effective_config_identity,
    )
    return output_path


def run_inference_aec_paired(
    config: str | Path,
    checkpoint: str | Path,
    mic_dir: Optional[str | Path] = None,
    mic_scp: Optional[str | Path] = None,
    far_dir: Optional[str | Path] = None,
    far_scp: Optional[str | Path] = None,
    output_dir: str | Path = "",
    device: str = "cuda:0",
    no_stream_align: bool = False,
    skip_existing: bool = False,
    allow_partial: bool = False,
    devices: Optional[list[str] | tuple[str, ...] | str] = None,
    inference_devices: Optional[list[str] | tuple[str, ...] | str] = None,
    expected_effective_config_identity: Mapping[str, Any] | str | None = None,
) -> Path:
    """Run paired AEC inference; multiple devices only shard model inference."""
    if devices is not None and inference_devices is not None:
        raise ValueError("devices 与 inference_devices 只能提供一个")
    selected_devices = inference_devices if inference_devices is not None else devices
    if selected_devices is not None:
        from evaluate.multi_gpu_inference import (
            normalize_inference_devices,
            run_aec_paired_inference_multi_gpu,
        )
        normalized = normalize_inference_devices(selected_devices, fallback=device)
        if len(normalized) > 1:
            if mic_scp is None or far_scp is None:
                raise ValueError("多 GPU AEC 编排需要 mic_scp 与 far_scp")
            return run_aec_paired_inference_multi_gpu(
                config, checkpoint, mic_scp, far_scp, output_dir, normalized,
                no_stream_align=no_stream_align,
                skip_existing=skip_existing,
                allow_partial=allow_partial,
                expected_effective_config_identity=expected_effective_config_identity,
            )
        device = normalized[0]
    config_path = absolute_path(config)
    checkpoint_path = absolute_path(checkpoint)
    output_path = absolute_path(output_dir)
    cfg_probe, _ = load_pipeline_config(config_path)
    if (
        mic_scp is not None
        and far_scp is not None
        and isinstance(cfg_probe.get("stft"), Mapping)
        and cfg_probe.get("stft")
    ):
        return _run_aec_adapter_inference(
            config_path=config_path, checkpoint_path=checkpoint_path,
            mic_scp=mic_scp, far_scp=far_scp, output_path=output_path,
            device=device, effective_identity=expected_effective_config_identity,
            allow_partial=allow_partial,
            no_stream_align=no_stream_align,
            skip_existing=skip_existing,
        )
    cmd = [
        sys.executable, str(repository_script("inference.py")),
        "-C", str(config_path),
        "--checkpoint", str(checkpoint_path),
        "--output_dir", str(output_path),
        "--device", device,
    ]
    if expected_effective_config_identity is not None:
        expected_sha256 = (
            expected_effective_config_identity
            if isinstance(expected_effective_config_identity, str)
            else expected_effective_config_identity["sha256"]
        )
        cmd += ["--expected-effective-config-sha256", expected_sha256]
    if no_stream_align:
        cmd.append("--aec_no_stream_align")
    if skip_existing:
        cmd.append("--skip-existing")
    if allow_partial:
        cmd.append("--allow-partial")
    if mic_scp:
        cmd += ["--input_mic_scp", str(absolute_path(mic_scp))]
    elif mic_dir:
        cmd += ["--input_mic_dir", str(absolute_path(mic_dir))]
    else:
        raise ValueError("Provide mic_dir or mic_scp")

    if far_scp:
        cmd += ["--input_far_scp", str(absolute_path(far_scp))]
    elif far_dir:
        cmd += ["--input_far_dir", str(absolute_path(far_dir))]
    else:
        raise ValueError("Provide far_dir or far_scp")

    logger.info("[Inference-AEC-paired] %s", " ".join(cmd))
    started_ns = time.time_ns()
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=EVAL_SUBPROCESS_TIMEOUT_S,
            start_new_session=True,
            preexec_fn=death_signal_preexec,
        )
    except subprocess.CalledProcessError:
        _validate_fresh_inference_manifest(
            output_path,
            config_path,
            checkpoint_path,
            started_ns,
            expected_effective_config_identity,
        )
        raise
    _validate_fresh_inference_manifest(
        output_path,
        config_path,
        checkpoint_path,
        started_ns,
        expected_effective_config_identity,
    )
    return output_path


# ---------------------------------------------------------------------------
# In-process AEC inference with zero farend (for SE testsets on AEC models)
# ---------------------------------------------------------------------------
def run_aec_inference_zero_farend(
    config_path: str | Path,
    checkpoint_path: str | Path,
    input_dir: Optional[str | Path] = None,
    input_scp: Optional[str | Path] = None,
    output_dir: str | Path = "",
    device: str = "cuda:0",
    allow_partial: bool = False,
    devices: Optional[list[str] | tuple[str, ...] | str] = None,
    inference_devices: Optional[list[str] | tuple[str, ...] | str] = None,
    expected_effective_config_identity: Mapping[str, Any] | str | None = None,
) -> Path:
    """运行 zero-farend AEC；多设备只分片模型推理。"""
    if devices is not None and inference_devices is not None:
        raise ValueError("devices 与 inference_devices 只能提供一个")
    selected_devices = inference_devices if inference_devices is not None else devices
    if selected_devices is not None:
        from evaluate.multi_gpu_inference import (
            normalize_inference_devices,
            run_aec_zero_farend_inference_multi_gpu,
        )
        normalized = normalize_inference_devices(selected_devices, fallback=device)
        if len(normalized) > 1:
            if input_scp is None:
                raise ValueError("多 GPU zero-farend 编排需要 input_scp")
            return run_aec_zero_farend_inference_multi_gpu(
                config_path, checkpoint_path, input_scp, output_dir, normalized,
                allow_partial=allow_partial,
                expected_effective_config_identity=expected_effective_config_identity,
            )
        device = normalized[0]
    with inference_output_lock(output_dir) as locked_output:
        return _run_aec_inference_zero_farend_locked(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            input_dir=input_dir,
            input_scp=input_scp,
            output_dir=locked_output,
            device=device,
            allow_partial=allow_partial,
            expected_effective_config_identity=expected_effective_config_identity,
        )


def _run_aec_inference_zero_farend_locked(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    input_dir: Optional[str | Path],
    input_scp: Optional[str | Path],
    output_dir: Path,
    device: str,
    allow_partial: bool,
    expected_effective_config_identity: Mapping[str, Any] | str | None,
) -> Path:
    """Load an AEC model and run inference with zero-filled farend."""
    import torch
    import torchaudio
    from tasks.se.models import build_model
    from tasks.aec.models import register_aec_models
    from core.stft import build_stft_istft
    from evaluate.checkpoint_loading import CheckpointFamily, load_checkpoint
    import soundfile as sf

    cfg, effective_identity = load_pipeline_config(
        config_path,
        expected_sha256=(
            expected_effective_config_identity
            if isinstance(expected_effective_config_identity, str)
            else (
                expected_effective_config_identity["sha256"]
                if expected_effective_config_identity is not None
                else None
            )
        ),
    )
    _mt = cfg.get("model", {}).get("type")
    register_aec_models(only=str(_mt)) if _mt else register_aec_models()

    stft_cfg = cfg.get("stft", {})
    if not stft_cfg:
        raise ValueError("AEC config missing top-level 'stft'")
    model_cfg = deepcopy(cfg.get("model", {}))
    # GAN-style config: build only generator for inference
    if isinstance(model_cfg, dict) and "generator" in model_cfg:
        model = build_model(model_cfg["generator"], stft_cfg=stft_cfg).to(device).eval()
        is_gan = True
    else:
        model = build_model(model_cfg, stft_cfg=stft_cfg).to(device).eval()
        is_gan = False

    family = (
        CheckpointFamily.GAN_GENERATOR
        if is_gan
        else CheckpointFamily.SINGLE_MODEL
    )
    checkpoint_fingerprint = fingerprint_file(checkpoint_path)
    _, ckpt = load_checkpoint(
        model,
        checkpoint_path,
        family=family,
        purpose="zero-farend AEC evaluation checkpoint",
        expected_fingerprint=checkpoint_fingerprint,
    )
    logger.info("Loaded %s checkpoint: %s", family.value, checkpoint_path)
    if "ema" in ckpt:
        logger.info("Applied EMA weights (zero-farend AEC inference)")

    stft = build_stft_istft(dict(stft_cfg)).to(device)
    sr = int(cfg.get("data", {}).get("sample_rate", 16000))

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if input_scp:
        files_dict = read_scp(input_scp)
        files = [(uid, files_dict[uid]) for uid in sorted(files_dict)]
        allow_path_separators = False
    elif input_dir:
        files = []
        for path in sorted(Path(input_dir).rglob("*.wav")):
            rel = path.relative_to(input_dir).with_suffix("")
            files.append((rel.as_posix(), path))
        allow_path_separators = True
    else:
        raise ValueError("Provide input_dir or input_scp")
    output_names = {
        uid: _safe_output_uid(uid, allow_path_separators=allow_path_separators)
        for uid, _ in files
    }
    if len(set(output_names.values())) != len(output_names):
        raise ValueError("zero-farend output UID encoding collision")

    # checkpoint 内容哈希复用上方经 loader 前后复核的指纹，避免第 4 次全量读取。
    artifact_fingerprint = build_content_fingerprint(
        {
            "inference_entry": Path(__file__),
            "config": config_path,
            "checkpoint": checkpoint_path,
            **_zero_farend_source_files(),
        },
        parameters={
            "mode": "aec_zero_farend",
            "sample_rate": sr,
            "effective_config_identity": effective_identity,
            "resolved_config": cfg,
        },
        precomputed={"checkpoint": checkpoint_fingerprint},
    )
    input_inventory = build_input_inventory(
        (uid, {"noisy": path}) for uid, path in files
    )
    reusable: set[str] = set()
    manifest_path = out_path / "run_manifest.json"
    if manifest_path.is_file():
        try:
            reusable = reusable_uids_from_manifest(
                manifest_path,
                expected_artifact_fingerprint=artifact_fingerprint,
                expected_input_inventory=input_inventory,
                expected_output_paths={
                    uid: out_path / f"{output_names[uid]}.wav" for uid, _ in files
                },
            )
        except (ValueError, KeyError):
            reusable = set()

    results: list[InferenceResult] = []
    for uid, p in files:
        output_file = (out_path / f"{output_names[uid]}.wav").resolve()
        try:
            output_file.relative_to(out_path)
        except ValueError as exc:
            raise RuntimeError(f"zero-farend output escapes output_dir: {uid!r}") from exc
        if uid in reusable:
            results.append(
                InferenceResult(
                    uid=uid,
                    status="skipped",
                    output_path=str(output_file),
                    output_identity=stat_audio_identity(output_file),
                )
            )
            continue
        try:
            wav, fs = torchaudio.load(str(p))
            wav = wav.mean(0, keepdim=True)
            if fs != sr:
                wav = torchaudio.functional.resample(wav.float(), int(fs), sr).to(wav.dtype)
            wav = wav.to(device)

            # center=True 的 STFT 使用 reflect padding，要求 T > n_fft//2；
            # 空输入拒绝，短输入右侧补零到 n_fft//2 + 1，iSTFT 后裁回原长。
            orig_len = int(wav.shape[-1])
            if orig_len == 0:
                raise ValueError(f"empty input waveform (0 samples): {p}")
            min_len = int(stft_cfg["n_fft"]) // 2 + 1
            if orig_len < min_len:
                wav = torch.nn.functional.pad(wav, (0, min_len - orig_len))

            spec = stft(wav)
            far_spec = torch.zeros_like(spec)
            # For zero-farend inference echo is zero, so LAEC residual = mic.
            # Explicitly set error_spectra so UltraDPCAEC stays in three-input mode.
            batch = {"mic_spectra": spec, "farend_spectra": far_spec, "error_spectra": spec}

            with torch.inference_mode():
                out = model(batch)
                ne_spec = out["nearend_estimate_spectra"]
                enh = stft.inverse(ne_spec, wav.shape[-1])

            # 裁回原始长度，保证 enhanced 与输入对齐。
            enh_np = enh[..., :orig_len].squeeze(0).cpu().numpy()
            atomic_write_audio(
                output_file,
                lambda staging: sf.write(str(staging), enh_np, sr),
            )
            results.append(
                InferenceResult(
                    uid=uid,
                    status="ok",
                    output_path=str(output_file),
                    output_identity=stat_audio_identity(output_file),
                )
            )
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            logger.warning("Failed on %s: %s", uid, error)
            results.append(InferenceResult(uid=uid, status="failed", error=error))

    validate_inference_publish_snapshot(
        config=config_path,
        artifact_fingerprint=artifact_fingerprint,
        input_inventory=input_inventory,
        config_loader=load_merged_config,
    )
    manifest = build_run_manifest(
        config=config_path,
        checkpoint=checkpoint_path,
        artifact_fingerprint=artifact_fingerprint,
        input_inventory=input_inventory,
        results=results,
        allow_partial=allow_partial,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    scp_entries = [
        (result.uid, result.output_path)
        for result in results
        if result.status in {"ok", "skipped"} and result.output_path is not None
    ]
    publish_inference_metadata(
        out_path,
        scp_content=serialize_scp_entries(scp_entries),
        manifest=manifest,
    )
    failed_count = len(manifest["failed_uids"])
    logger.info("Zero-farend AEC inference done → %s (%d ok)", out_path, len(scp_entries))
    if failed_count and not allow_partial:
        raise RuntimeError(
            f"Zero-farend AEC inference failed for {failed_count}/{len(files)} file(s); "
            f"see {out_path / 'run_manifest.json'} or pass allow_partial=True."
        )
    return out_path


# ---------------------------------------------------------------------------
# Evaluation wrappers (subprocess → evaluate/*.py)
# ---------------------------------------------------------------------------
def run_evaluate_urgent(
    inf_scp: str | Path,
    output_dir: str | Path,
    ref_scp: Optional[str | Path] = None,
    audio_quality: bool = False,
    downstream: bool = False,
    device: str = "cuda:0",
) -> Path:
    """Run ``evaluate_urgent.py`` without depending on the caller cwd."""
    return _run_evaluate_urgent(
        inf_scp,
        output_dir,
        ref_scp=ref_scp,
        audio_quality=audio_quality,
        downstream=downstream,
        device=device,
    )


# ---------------------------------------------------------------------------
# AEC simulated testset generation
# ---------------------------------------------------------------------------
def prepare_aec_sim_testset(
    config: str | Path,
    output_dir: str | Path,
    num_samples: int = 100,
    seed: int = 42,
    sim_config: str | Path | None = None,
) -> Path:
    """Run the canonical AEC simulated-testset generator module.

    生成器内部做 manifest 校验：指纹（生成器版本/config/数据源/seed/样本数）
    一致则秒级复用并返回 0；manifest 缺失或不一致则 fail closed（非零退出）。
    不再凭"目录存在"复用旧测试集（JAE-C01）。

    Args:
        config: 实验配置（向后兼容默认值）。
        sim_config: 用于生成 aec_sim 测试集的配置；提供时优先使用，
            使多个实验可基于同一测试集做公平对比。
    """
    out = absolute_path(output_dir)
    generation_config = absolute_path(
        sim_config if sim_config is not None else config
    )
    module = "tools.prepare_data_tools.generators.aec.generate_simulated_testset"
    cmd = [
        sys.executable,
        "-m",
        module,
        "-C", str(generation_config),
        "--output-dir", str(out),
        "--num-samples", str(num_samples),
        "--seed", str(seed),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(PROJECT_ROOT), env.get("PYTHONPATH")) if value
    )
    logger.info("[Prepare-AEC-SIM] %s", " ".join(cmd))
    subprocess.run(
        cmd,
        check=True,
        env=env,
        timeout=EVAL_SUBPROCESS_TIMEOUT_S,
        start_new_session=True,
        preexec_fn=death_signal_preexec,
    )
    return out


# ---------------------------------------------------------------------------
# AEC simulated testset SCP helpers
# ---------------------------------------------------------------------------
from core.data.simulation.scene_names import AEC_SCENE_ORDER

AEC_SIM_SCENARIOS = list(AEC_SCENE_ORDER)

AEC_SIM_REF_SCENARIOS = {"prob_target_only", "prob_target_noise", "prob_target_echo", "prob_target_noise_echo"}


def ensure_aec_sim_scps(
    testset_root: str | Path,
    scp_root: str | Path | None = None,
) -> dict:
    """Ensure SCP files exist for all AEC sim scenarios, generating if needed.

    SCP 在缺失或早于测试集 manifest.json（测试集被重新生成过）时重写，
    避免重新生成后消费陈旧清单。

    SCP 默认写入测试集目录自身（测试集根默认即 ``data/test/aec_sim_testset``
    时与旧行为一致）；不再硬编码到 ``DEFAULT_TEST_DATA_ROOT``，自定义
    ``--aec_sim_testset_root`` 时不会再向仓库 ``data/test`` 写入。

    Returns a dict mapping scenario name -> {wav_name: scp_path}.
    """
    src = absolute_path(testset_root)
    out_root = src if scp_root is None else absolute_path(scp_root)
    manifest = src / "manifest.json"
    manifest_sha = _sha256_file(manifest) if manifest.exists() else "manifest-missing"
    wav_names = ["mic", "nearend", "farend", "noisy_se", "echo"]
    result: dict = {}
    for scen in AEC_SIM_SCENARIOS:
        scen_src = src / scen
        if not scen_src.exists():
            continue
        scen_dst = out_root / scen
        scen_dst.mkdir(parents=True, exist_ok=True)
        scen_scps: dict = {}
        samples = sorted(d for d in scen_src.iterdir() if d.is_dir())
        for wav_name in wav_names:
            scp = scen_dst / f"{wav_name}.scp"
            entries: list[tuple[str, str]] = []
            digest = hashlib.sha256(f"{manifest_sha}:{scen}:{wav_name}".encode("utf-8"))
            for sample_dir in samples:
                wav = sample_dir / f"{wav_name}.wav"
                if not wav.exists():
                    continue
                if wav.is_symlink():
                    raise ValueError(f"音频 payload 不允许符号链接: {wav}")
                if not wav.is_file():
                    raise ValueError(f"音频 payload 不是普通文件: {wav}")
                stat = wav.stat()
                digest.update(f"{sample_dir.name}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
                entries.append((sample_dir.name, str(wav.resolve())))
            write_generated_scp(
                entries,
                scp,
                source=scen_src,
                source_snapshot=digest.hexdigest(),
            )
            scen_scps[wav_name] = scp
        result[scen] = scen_scps
    return result


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------
def save_summary(
    summary: dict,
    output_dir: str | Path,
    *,
    publish_artifact: bool = True,
) -> Path:
    """原子写入任务 summary；公共 envelope 复用同一生命周期 seam。"""
    out = Path(output_dir).expanduser().resolve()
    path = out / "summary.json"
    if not publish_artifact:
        atomic_write_json(path, summary)
        logger.info("Summary saved: %s", path)
        return path

    pipeline = str(summary.get("pipeline", "eval_pipeline_unknown"))
    task = str(summary.get("model_type") or pipeline.rsplit("_", 1)[-1])
    adapter = PipelineTaskAdapter(
        task=task,
        pipeline=pipeline,
        step=str(summary.get("step", "unknown")),
        output_dir=out,
        provenance={
            "config": summary.get("config"),
            "checkpoint": summary.get("checkpoint"),
            "parameters": {
                "step": summary.get("step"),
                "model_type": summary.get("model_type"),
                "model_subtype": summary.get("model_subtype"),
            },
            "entry": pipeline,
        },
    )
    run_pipeline(
        adapter,
        prepare=lambda: None,
        run=lambda _context: summary,
        requested_run_id=summary.get("run_id"),
    )
    logger.info("Summary saved: %s", path)
    return path


def print_summary_table(summary: dict):
    """Pretty-print a summary dict to the console."""
    print("\n" + "=" * 80)
    print("EVALUATION PIPELINE SUMMARY")
    print("=" * 80)
    for ds, res in summary.get("datasets", {}).items():
        status = res.get("status", "unknown")
        print(f"\n  [{ds}]  status={status}")
        results_file = res.get("results_file")
        if results_file and Path(results_file).exists():
            print(f"         results → {results_file}")
        enhanced_dir = res.get("enhanced_dir")
        if enhanced_dir:
            print(f"         enhanced → {enhanced_dir}")
    print("=" * 80 + "\n")
