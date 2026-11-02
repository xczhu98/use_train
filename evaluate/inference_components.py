"""根推理入口使用的纯 artifact/source 组件。"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from core.utils.inference_artifacts import (
    InferenceResult,
    ManifestValidationError,
    build_content_fingerprint,
    build_run_manifest,
    collect_source_tree_files,
    fingerprint_file,
    publish_inference_metadata,
    reusable_uids_from_manifest,
    stat_audio_identity,
)
from core.inference import (
    ArtifactBinding,
    FileInferenceItem,
    InferenceReport,
    InferenceSpec,
    ReusePolicy,
    run_inference_report,
)
from tools.shared.scp_io import serialize_scp_entries

logger = logging.getLogger(__name__)

_EFFECTIVE_CONFIG_SCHEMA = "effective-merged-config-v1"
_EFFECTIVE_CONFIG_ALGORITHM = "sha256-json-v1"


def _artifact_binding_from_fingerprint(artifact_fingerprint: Mapping[str, Any]) -> ArtifactBinding:
    """将已计算的 source identity 原样转成 runner binding。"""
    return ArtifactBinding.from_fingerprint(artifact_fingerprint)


def run_adapter_inference(
    *,
    adapter: Any,
    config: str | Path,
    output_dir: str | Path,
    items: Iterable[FileInferenceItem],
    artifact_fingerprint: dict[str, Any],
    checkpoint: str | Path | None = None,
    allow_partial: bool = False,
    skip_existing: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> InferenceReport:
    """使用新 ``InferenceSpec`` 调用公共 runner。"""
    normalized = tuple(items)
    if not normalized:
        raise ValueError("evaluate adapter inference requires at least one item")
    if any(not isinstance(item, FileInferenceItem) for item in normalized):
        raise TypeError("evaluate adapter inference items must be FileInferenceItem")
    spec = InferenceSpec(
        config=config,
        output_dir=output_dir,
        items=normalized,
        checkpoint=checkpoint,
        artifacts=_artifact_binding_from_fingerprint(artifact_fingerprint),
        allow_partial=allow_partial,
        reuse=ReusePolicy(enabled=skip_existing),
        run_metadata=dict(metadata or {}),
    )
    return run_inference_report(spec, adapter)


def effective_config_identity(config: Mapping[str, Any]) -> dict[str, str]:
    """对实际消费的 merged mapping 生成稳定内容身份。"""
    if not isinstance(config, Mapping):
        raise TypeError("effective config 必须是 mapping")
    from core.utils.inference_artifacts import fingerprint_json

    return {
        "schema": _EFFECTIVE_CONFIG_SCHEMA,
        "algorithm": _EFFECTIVE_CONFIG_ALGORITHM,
        "sha256": fingerprint_json(config),
    }


def validate_effective_config_identity(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any] | str | None,
) -> dict[str, str]:
    """校验实际 mapping 身份；expected 为 None 时仅返回实际身份。"""
    identity = effective_config_identity(actual)
    if expected is None:
        return identity
    expected_identity = (
        {
            "schema": _EFFECTIVE_CONFIG_SCHEMA,
            "algorithm": _EFFECTIVE_CONFIG_ALGORITHM,
            "sha256": expected,
        }
        if isinstance(expected, str)
        else dict(expected)
    )
    if expected_identity != identity:
        raise RuntimeError(
            "effective config identity mismatch: "
            f"expected={expected_identity.get('sha256')!r}, "
            f"actual={identity['sha256']!r}"
        )
    return identity


def load_effective_config(
    config_path: str | Path,
    *,
    expected: Mapping[str, Any] | str | None = None,
    loader: Callable[[str], Mapping[str, Any]] | None = None,
) -> tuple[Mapping[str, Any], dict[str, str]]:
    """调用实际 loader 一次，并立即绑定返回 mapping 的身份。"""
    if loader is None:
        from core.utils.config import load_merged_config

        loader = load_merged_config
    config = loader(str(config_path))
    if not isinstance(config, Mapping):
        raise TypeError("load_merged_config 必须返回 mapping")
    return config, validate_effective_config_identity(config, expected)


def validate_inference_publish_snapshot(
    *,
    config: str | Path,
    artifact_fingerprint: Mapping[str, Any],
    input_inventory: Mapping[str, Any],
    config_loader: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """发布 manifest 前复核启动时绑定的配置、文件与输入快照。"""
    from core.utils.inference_artifacts import fingerprint_json

    files = artifact_fingerprint.get("files")
    parameters = artifact_fingerprint.get("parameters")
    if (
        artifact_fingerprint.get("algorithm") != "sha256"
        or not isinstance(files, Mapping)
        or not files
        or not isinstance(parameters, Mapping)
    ):
        raise RuntimeError("推理 artifact fingerprint 发布前结构非法")
    payload = {
        "algorithm": "sha256",
        "files": dict(files),
        "parameters": dict(parameters),
    }
    if artifact_fingerprint.get("sha256") != fingerprint_json(payload):
        raise RuntimeError("推理 artifact fingerprint 发布前聚合身份不一致")
    for name, expected in files.items():
        if expected is None:
            continue
        if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
            raise RuntimeError(f"推理 artifact fingerprint 发布前文件结构非法: {name}")
        try:
            current = fingerprint_file(expected["path"])
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"推理 artifact 文件发布前不可复核: {name}") from exc
        if current != dict(expected):
            raise RuntimeError(f"推理 artifact 文件发布前发生变化: {name}")

    items = input_inventory.get("items")
    algorithm = input_inventory.get("algorithm")
    if algorithm != "stat-v1" or not isinstance(items, list):
        raise RuntimeError("推理 input inventory 发布前结构非法")
    inventory_payload = {"algorithm": algorithm, "items": items}
    if input_inventory.get("digest") != fingerprint_json(inventory_payload):
        raise RuntimeError("推理 input inventory 发布前聚合身份不一致")
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("inputs"), Mapping):
            raise RuntimeError("推理 input inventory 发布前条目结构非法")
        for role, expected in item["inputs"].items():
            if expected is None:
                continue
            if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
                raise RuntimeError(f"推理 input inventory 发布前输入结构非法: {role}")
            try:
                current = stat_audio_identity(expected["path"])
            except (OSError, ValueError, RuntimeError) as exc:
                raise RuntimeError(f"推理 input 发布前不可复核: {role}") from exc
            if current != dict(expected):
                raise RuntimeError(f"推理 input 发布前发生变化: {role}")

    expected_identity = parameters.get("effective_config_identity")
    resolved_config = parameters.get("resolved_config")
    if not isinstance(expected_identity, Mapping) or not isinstance(
        resolved_config, Mapping
    ):
        raise RuntimeError("推理 artifact 发布前缺少 effective config identity")
    validate_effective_config_identity(resolved_config, expected_identity)
    _, current_identity = load_effective_config(
        config,
        expected=expected_identity,
        loader=config_loader,
    )
    return current_identity


def configured_checkpoint_files(value: Any, *, prefix: str = "config") -> dict[str, Path]:
    """提取 waveform-native 配置中实际加载的 ``*_ckpt_path``。"""
    files: dict[str, Path] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}"
            if key.endswith("_ckpt_path") and isinstance(child, str) and child:
                files[name] = Path(child).expanduser()
            else:
                files.update(configured_checkpoint_files(child, prefix=name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            files.update(
                configured_checkpoint_files(child, prefix=f"{prefix}[{index}]")
            )
    return files


def verify_configured_checkpoint_fingerprints(
    checkpoint_files: dict[str, Path],
    artifact_fingerprint: dict,
) -> None:
    """模型构造后复核预先固化的 waveform-native checkpoint 内容身份。"""
    artifact_files = artifact_fingerprint.get("files")
    if not isinstance(artifact_files, dict):
        raise RuntimeError("artifact fingerprint 缺少合法 files mapping")
    for name, path in checkpoint_files.items():
        expected = artifact_files.get(name)
        if not isinstance(expected, dict):
            raise RuntimeError(f"artifact fingerprint 缺少 checkpoint: {name}")
        try:
            current = fingerprint_file(path)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"checkpoint 加载后无法复核内容身份: {path}") from exc
        if current != expected:
            raise RuntimeError(f"checkpoint 加载期间内容发生变化: {path}")


def inference_source_files(
    project_root: str | Path,
    *,
    is_aec: bool,
) -> dict[str, Path]:
    """返回任务会影响推理结果的显式源码边界。"""
    paths = [
        "core/stft.py",
        "core/models",
        "tasks/__init__.py",
        "tasks/descriptors.py",
        "tasks/registry.py",
        "core/utils/config.py",
        "core/utils/inference_artifacts.py",
        "core/inference",
        "core/utils/registry.py",
        "core/utils/safe_load.py",
        "evaluate/checkpoint_loading.py",
        "evaluate/inference_components.py",
        "tools/shared/scp_io.py",
    ]
    if is_aec:
        paths.extend(
            [
                "tasks/aec/models",
                "tasks/aec/inference",
                "tasks/aec/data/linear_aec_frontend.py",
                "evaluate/aec_pairing.py",
            ]
        )
    else:
        paths.append("tasks/se/models")
    return collect_source_tree_files(project_root, paths)


def build_artifact_fingerprint(
    args: Any,
    cfg: dict,
    *,
    project_root: str | Path,
    inference_entry: str | Path,
    is_aec: bool,
    is_waveform_native: bool,
    has_model_artifacts: bool | None = None,
    effective_device: str | None = None,
    source_files: dict[str, Path] | None = None,
    effective_identity: Mapping[str, Any] | None = None,
    precomputed_files: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict:
    """绑定推理入口、源码、配置、模型权重与影响结果的 CLI 参数。

    输出布局由调用方传入 ``output_paths``，在 manifest 复用时严格校验。
    ``has_model_artifacts`` 区分两种 waveform-native 形态：PASE/StuPASE 的权重
    由模型配置内的 artifact 路径提供（默认沿用 ``is_waveform_native``），
    CoFi-Lite 等 checkpoint 训练模型仍必须把 ``--checkpoint`` 纳入身份。
    """
    uses_model_artifacts = (
        bool(is_waveform_native) if has_model_artifacts is None else bool(has_model_artifacts)
    )
    checkpoint = (
        None
        if uses_model_artifacts or not args.checkpoint
        else Path(args.checkpoint).expanduser()
    )
    files = {
        "inference_entry": Path(inference_entry),
        "config": args.config,
        "checkpoint": checkpoint,
        **(
            source_files
            if source_files is not None
            else inference_source_files(project_root, is_aec=is_aec)
        ),
    }
    if uses_model_artifacts:
        files.update(configured_checkpoint_files(cfg.get("model", {})))
    return build_content_fingerprint(
        files,
        parameters={
            "resolved_config": cfg,
            "effective_config_identity": dict(
                effective_identity or effective_config_identity(cfg)
            ),
            "requested_device": args.device,
            "effective_device": effective_device or args.device,
            "sr": args.sr,
            "chunk_seconds": args.chunk_seconds,
            "cpu_long_audio_seconds": getattr(args, "cpu_long_audio_seconds", 0.0),
            "aec_stream_align": bool(args.aec_stream_align),
            "aec_no_stream_align": bool(args.aec_no_stream_align),
            "aec_review_multichannel": bool(
                getattr(args, "aec_review_multichannel", False)
            ),
        },
        precomputed=precomputed_files,
    )


def resolve_reusable_uids(
    *,
    enabled: bool,
    out_dir: Path,
    expected_uids: list[str],
    artifact_fingerprint: dict,
    input_inventory: dict,
    output_paths: dict[str, str | Path] | None = None,
) -> set[str]:
    """只复用 manifest、输入、实现和输出结构身份都可信的 UID。"""
    if not enabled:
        return set()
    manifest_path = out_dir / "run_manifest.json"
    if not manifest_path.is_file():
        logger.warning("--skip-existing 未找到 run_manifest.json；为避免误复用，将重新推理全部输入")
        return set()
    output_paths = output_paths or {
        uid: out_dir / f"{uid}.wav" for uid in expected_uids
    }
    try:
        reusable = reusable_uids_from_manifest(
            manifest_path,
            expected_artifact_fingerprint=artifact_fingerprint,
            expected_input_inventory=input_inventory,
            expected_output_paths=output_paths,
        )
    except (ManifestValidationError, KeyError) as exc:
        logger.warning("旧 run_manifest.json 不可信，禁止复用并重新推理全部输入: %s", exc)
        return set()
    logger.info("Trusted manifest allows reusing %d/%d outputs", len(reusable), len(expected_uids))
    return reusable


def finalize_inference_artifacts(
    *,
    config: str | Path,
    allow_partial: bool,
    out_dir: Path,
    expected_uids: Iterable[str],
    failed_uids: Iterable[str],
    skipped_uids: Iterable[str],
    artifact_fingerprint: dict,
    input_inventory: dict,
    output_paths: dict[str, str | Path] | None = None,
    created_at: str | None = None,
    config_loader: Callable[[str], Mapping[str, Any]] | None = None,
) -> list[InferenceResult]:
    """校验输出结构身份，并以集合事务发布 SCP 与严格 manifest。

    ``inf.scp`` 使用共享序列化器安全引用 UID 与输出路径；manifest 继续记录
    同一个规范化绝对输出路径，旧扁平输出布局保持不变。
    """
    expected = list(expected_uids)
    failed_set = set(failed_uids)
    skipped_set = set(skipped_uids)
    output_paths = output_paths or {
        uid: out_dir / f"{uid}.wav" for uid in expected
    }
    results: list[InferenceResult] = []
    for uid in expected:
        if uid in failed_set:
            results.append(
                InferenceResult(uid=uid, status="failed", error="inference failed; see log")
            )
            continue
        output_path = Path(output_paths.get(uid, out_dir / f"{uid}.wav")).resolve()
        try:
            output_identity = stat_audio_identity(output_path)
        except (OSError, ValueError, RuntimeError) as exc:
            failed_set.add(uid)
            results.append(
                InferenceResult(
                    uid=uid,
                    status="failed",
                    error=f"output validation failed: {exc}",
                )
            )
            logger.warning("输出校验失败 %s: %s", uid, exc)
            continue
        results.append(
            InferenceResult(
                uid=uid,
                status="skipped" if uid in skipped_set else "ok",
                output_path=str(output_path),
                output_identity=output_identity,
            )
        )

    validate_inference_publish_snapshot(
        config=config,
        artifact_fingerprint=artifact_fingerprint,
        input_inventory=input_inventory,
        config_loader=config_loader,
    )
    checkpoint_fingerprint = artifact_fingerprint["files"].get("checkpoint")
    manifest = build_run_manifest(
        config=config,
        checkpoint=(
            checkpoint_fingerprint["path"]
            if checkpoint_fingerprint is not None
            else None
        ),
        artifact_fingerprint=artifact_fingerprint,
        input_inventory=input_inventory,
        results=results,
        allow_partial=allow_partial,
        created_at=created_at or datetime.now().isoformat(timespec="seconds"),
    )
    scp_entries = [
        (result.uid, result.output_path)
        for result in results
        if result.status in {"ok", "skipped"}
    ]
    publish_inference_metadata(
        out_dir,
        scp_content=serialize_scp_entries(scp_entries),
        manifest=manifest,
    )
    return results
