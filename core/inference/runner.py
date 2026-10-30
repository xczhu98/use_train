"""任务无关的最小推理 runner。"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Mapping

from core.utils.artifact_identity import SourceBinding
from core.utils.inference_artifacts import (
    InferenceResult,
    build_run_manifest,
    fingerprint_json,
    publish_inference_metadata,
    reusable_uids_from_manifest,
    stat_audio_identity,
)
from tools.shared.scp_io import serialize_scp_entries

from .contracts import FileInferenceItem, InferenceAdapter, InferenceReport, InferenceSpec, PreparedOutput
from .artifact_binding import verify_file_bindings
from .input_inventory import build_inventory, verify_inventory
from .output_writer import atomic_write_audio, inference_output_lock

logger = logging.getLogger(__name__)


def _bind_run_metadata(
    artifact_fingerprint: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """将会影响复用语义的运行 metadata 纳入 artifact fingerprint。"""
    if not metadata:
        return dict(artifact_fingerprint)
    parameters = artifact_fingerprint.get("parameters")
    files = artifact_fingerprint.get("files")
    if artifact_fingerprint.get("algorithm") != "sha256" or not isinstance(parameters, dict) or not isinstance(files, dict):
        raise ValueError("artifact fingerprint must contain sha256/files/parameters")
    merged_parameters = dict(parameters)
    merged_parameters["inference_metadata"] = dict(metadata)
    payload = {"algorithm": "sha256", "files": dict(files), "parameters": merged_parameters}
    return {**payload, "sha256": fingerprint_json(payload)}


_ABORTED_ITEM_ERROR = "not attempted: run aborted after an earlier item failure"


def _publish_aborted_run(
    *,
    spec: InferenceSpec,
    output_dir: Path,
    artifact_fingerprint: Mapping[str, Any],
    inventory: Mapping[str, Any],
    results: list[InferenceResult],
    not_attempted: list[str],
) -> None:
    """在 allow_partial=False 的中止路径上发布描述真实集合的 manifest/inf.scp。

    音频逐条发布已经落盘，若此处不更新 manifest，输出目录会留下“新音频 +
    上次运行旧 manifest/inf.scp”的不一致集合。
    """
    verify_file_bindings(artifact_fingerprint)
    verify_inventory(inventory)
    aborted = [
        *results,
        *(
            InferenceResult(uid, "failed", error=_ABORTED_ITEM_ERROR)
            for uid in not_attempted
        ),
    ]
    manifest = build_run_manifest(
        config=spec.config,
        checkpoint=spec.checkpoint,
        artifact_fingerprint=artifact_fingerprint,
        input_inventory=inventory,
        results=aborted,
        allow_partial=spec.allow_partial,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    publish_inference_metadata(
        output_dir,
        scp_content=serialize_scp_entries(
            [
                (result.uid, result.output_path)
                for result in aborted
                if result.status in {"ok", "skipped"}
            ]
        ),
        manifest=manifest,
    )


def run_inference(
    spec: InferenceSpec,
    adapter: InferenceAdapter,
    *,
    artifact_fingerprint: dict[str, Any] | None = None,
) -> InferenceReport:
    """执行文件推理、复用已验证输出并发布标准 manifest。"""
    if not isinstance(spec, InferenceSpec):
        raise TypeError("run_inference requires an InferenceSpec")
    if not spec.items:
        raise ValueError("run_inference requires at least one inference item")
    if artifact_fingerprint is None:
        if spec.artifacts is None:
            raise ValueError("InferenceSpec requires artifacts when no fingerprint is supplied")
        source_binding = spec.artifacts.source_binding()
    else:
        source_binding = SourceBinding.from_fingerprint(artifact_fingerprint)
    artifact_fingerprint = _bind_run_metadata(
        source_binding.as_dict(), dict(spec.run_metadata)
    )
    source_binding = SourceBinding.from_fingerprint(artifact_fingerprint)
    artifact_fingerprint = source_binding.as_dict()
    output_dir = Path(spec.output_dir).expanduser()
    output_root = output_dir.resolve()
    missing_output = [item.uid for item in spec.items if item.output is None]
    if missing_output:
        raise ValueError(
            "every InferenceItem must provide an output path; missing="
            + ",".join(missing_output[:5])
        )
    inventory = build_inventory(spec.items)
    results: list[InferenceResult] = []
    with inference_output_lock(output_dir):
        verify_file_bindings(source_binding)
        verify_inventory(inventory)
        expected_output_paths = {
            item.uid: Path(item.output).expanduser().resolve()
            for item in spec.items
        }
        reusable: set[str] = set()
        manifest_path = output_dir / "run_manifest.json"
        if spec.reuse.enabled:
            if len(expected_output_paths) != len(spec.items):
                raise ValueError(
                    "skip_existing requires every InferenceItem to provide an output path"
                )
            if manifest_path.is_file():
                try:
                    reusable = reusable_uids_from_manifest(
                        manifest_path,
                        expected_artifact_fingerprint=artifact_fingerprint,
                        expected_input_inventory=inventory,
                        expected_output_paths=expected_output_paths,
                    )
                except Exception as exc:
                    if spec.reuse.on_rejection == "fail":
                        raise RuntimeError(
                            "skip-existing reuse rejected and ReusePolicy.on_rejection='fail'"
                        ) from exc
                    # 旧 manifest、指纹或输入不匹配时必须全部重新推理，但保留原因。
                    logger.warning("skip-existing reuse rejected; re-infer all items: %s", exc)
                    reusable = set()
        for index, item in enumerate(spec.items):
            if item.uid in reusable:
                output_path = expected_output_paths[item.uid]
                try:
                    identity = stat_audio_identity(output_path)
                except (OSError, ValueError, RuntimeError) as exc:
                    # 复用判定后文件消失/被替换：降级为重推，不中断整个 run。
                    logger.warning(
                        "reusable output %s is no longer valid; re-inferring: %s",
                        item.uid,
                        exc,
                    )
                else:
                    results.append(
                        InferenceResult(item.uid, "skipped", str(output_path), identity)
                    )
                    continue
            try:
                output = adapter.infer(item)
                if not isinstance(output, PreparedOutput):
                    raise TypeError("adapter must return PreparedOutput")
                if output.uid != item.uid:
                    raise ValueError(
                        f"adapter output uid mismatch: {output.uid!r} != {item.uid!r}"
                    )
                declared_path = expected_output_paths.get(item.uid)
                if declared_path is None:
                    raise ValueError(f"missing declared output path for {item.uid!r}")
                output_path = declared_path
                writer = output.writer
                try:
                    output_path.relative_to(output_root)
                except ValueError as exc:
                    raise ValueError(
                        f"adapter output escapes output_dir: {output_path}"
                    ) from exc
                atomic_write_audio(output_path, writer)
                identity = stat_audio_identity(output_path)
                results.append(InferenceResult(item.uid, "ok", str(output_path.resolve()), identity))
            except Exception as exc:
                if not spec.allow_partial:
                    # 中止路径同样要发布与已落盘音频一致的 manifest/inf.scp。
                    try:
                        _publish_aborted_run(
                            spec=spec,
                            output_dir=output_dir,
                            artifact_fingerprint=artifact_fingerprint,
                            inventory=inventory,
                            results=[
                                *results,
                                InferenceResult(
                                    item.uid,
                                    "failed",
                                    error=str(exc) or type(exc).__name__,
                                ),
                            ],
                            not_attempted=[
                                pending.uid for pending in spec.items[index + 1 :]
                            ],
                        )
                    except Exception as publish_exc:
                        logger.error(
                            "failed to publish aborted-run metadata: %s",
                            publish_exc,
                            exc_info=True,
                        )
                    raise
                results.append(InferenceResult(item.uid, "failed", error=str(exc)))
        verify_file_bindings(source_binding)
        verify_inventory(inventory)
        manifest = build_run_manifest(
            config=spec.config,
            checkpoint=spec.checkpoint,
            artifact_fingerprint=artifact_fingerprint,
            input_inventory=inventory,
            results=results,
            allow_partial=spec.allow_partial,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        publish_inference_metadata(
            output_dir,
            scp_content=serialize_scp_entries(
                [
                    (result.uid, result.output_path)
                    for result in results
                    if result.status in {"ok", "skipped"}
                ]
            ),
            manifest=manifest,
        )
    failed = sum(result.status == "failed" for result in results)
    status = "failed" if failed == len(results) else "partial" if failed else "completed"
    return InferenceReport(
        status=status,
        results=tuple(results),
        manifest_path=output_dir / "run_manifest.json",
        scp_path=output_dir / "inf.scp",
    )


def run_inference_report(
    spec: InferenceSpec,
    adapter: InferenceAdapter,
) -> InferenceReport:
    """新 Interface 的报告入口；旧 ``run_inference`` 返回值保持兼容。"""
    return run_inference(spec, adapter)
