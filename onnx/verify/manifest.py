"""ONNX 发布 artifact 与源身份的兼容 manifest 桥接。

该模块复用 ``core.inference.artifact_binding`` 的文件 fingerprint 算法，但不改变
ONNX 的三路验证和原子发布协议。manifest 是发布目标旁边的 sidecar，供
skip-existing 或部署前检查判断源 checkpoint/config/runtime contract 是否仍匹配。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from core.utils.artifact_identity import (
    SourceBinding,
    bind_source_files,
    verify_source_binding,
)
from core.utils.inference_artifacts import fingerprint_file


ONNX_MANIFEST_SCHEMA = 1


@dataclass(frozen=True)
class OnnxReuseDecision:
    """统一的 ONNX skip-existing 判定结果。"""

    reusable: bool
    reason: str
    manifest: dict[str, Any] | None = None


def runtime_contract_from_spec(spec: Any) -> dict[str, Any]:
    """将已验证的 ``StreamingRuntimeSpec`` 固化为可比较的 JSON 合同。"""
    if spec is None:
        raise ValueError("runtime spec is required for ONNX publication")
    def convert(value: Any) -> Any:
        if hasattr(value, "to_metadata"):
            return convert(value.to_metadata())
        if is_dataclass(value):
            return convert(asdict(value))
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (tuple, list, set, frozenset)):
            return [convert(item) for item in value]
        return value
    return {
        "model_id": str(spec.profile.model_id),
        "input_names": convert(spec.input_names),
        "output_names": convert(spec.output_names),
        "cache_shapes": convert(spec.cache_shapes),
        "dynamic_axes": convert(spec.dynamic_axes),
        "stft": convert(spec.stft),
        "model_runtime": convert(
            next((getattr(spec, name) for name in (
                "cagcrn", "deepvqe", "dbgnet", "tse"
            ) if getattr(spec, name, None) is not None), None)
        ),
        "deployment_config": convert(spec.deployment_config or {}),
        "metadata": convert(spec.metadata),
    }


def validate_onnx_manifest_spec(
    *,
    model_id: str,
    source_files: Mapping[str, str | Path | None] | None,
    runtime_contract: Mapping[str, Any],
    source_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """在发布前校验正式 source binding，并返回冻结的源绑定。

    ``source_binding`` 是 verifier 在实际 loader 读取 source 后冻结的快照。
    传入它时绝不重新按 pathname 计算 fingerprint；这阻止 A 权重已加载、B
    权重随后替换路径时 manifest 被错误绑定到 B。
    """
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(runtime_contract, Mapping) or not runtime_contract:
        raise ValueError("runtime_contract must be a non-empty mapping")
    if source_binding is not None:
        binding = SourceBinding.from_fingerprint(source_binding).as_dict()
        parameters = binding.get("parameters", {})
        if parameters.get("model_id") != model_id:
            raise ValueError("source binding model_id does not match manifest")
        return binding
    files = {
        name: path for name, path in (source_files or {}).items() if path is not None
    }
    if not files:
        raise ValueError("source_files must contain at least one existing file")
    return bind_source_files(files, parameters={"model_id": model_id})


def build_onnx_manifest(
    *,
    target: str | Path,
    model_id: str,
    source_files: Mapping[str, str | Path | None] | None,
    runtime_contract: Mapping[str, Any],
    verification: Mapping[str, Any] | None = None,
    source_binding: Mapping[str, Any] | None = None,
    artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构建并校验 ONNX 发布 manifest。

    ``source_files`` 至少应包含实际 checkpoint；config 对 config-based verifier
    也应传入。未提供的可选路径会被忽略，避免把 ``None`` 写入 fingerprint。
    """
    binding = validate_onnx_manifest_spec(
        model_id=model_id,
        source_files=source_files,
        runtime_contract=runtime_contract,
        source_binding=source_binding,
    )
    manifest = {
        "schema_version": ONNX_MANIFEST_SCHEMA,
        "model_id": model_id,
        "target": str(Path(target).resolve()),
        "source_binding": binding,
        "runtime_contract": dict(runtime_contract),
        "verification": dict(verification or {}),
        "artifact": dict(artifact) if artifact is not None else fingerprint_file(target),
    }
    _validate_onnx_manifest(manifest)
    return manifest


def _validate_onnx_manifest(manifest: Mapping[str, Any]) -> None:
    expected = {
        "schema_version", "model_id", "target", "source_binding",
        "runtime_contract", "verification", "artifact",
    }
    if set(manifest) != expected:
        raise ValueError(f"ONNX manifest fields mismatch: expected={sorted(expected)}")
    if manifest["schema_version"] != ONNX_MANIFEST_SCHEMA:
        raise ValueError("unsupported ONNX manifest schema")
    if not isinstance(manifest["model_id"], str) or not manifest["model_id"]:
        raise ValueError("ONNX manifest model_id must be non-empty")
    if not isinstance(manifest["source_binding"], Mapping):
        raise ValueError("ONNX manifest source_binding must be a mapping")
    try:
        SourceBinding.from_fingerprint(manifest["source_binding"])
    except (TypeError, ValueError) as exc:
        raise ValueError("ONNX manifest source_binding is invalid") from exc
    parameters = manifest["source_binding"].get("parameters", {})
    if parameters.get("model_id") != manifest["model_id"]:
        raise ValueError("ONNX manifest source_binding model_id mismatch")
    if not isinstance(manifest["runtime_contract"], Mapping) or not manifest["runtime_contract"]:
        raise ValueError("ONNX manifest runtime_contract must be non-empty")
    if not isinstance(manifest["artifact"], Mapping):
        raise ValueError("ONNX manifest artifact must be a mapping")


def manifest_path(target: str | Path) -> Path:
    """返回与发布 ONNX 同目录同 basename 的 manifest sidecar 路径。"""
    target = Path(target)
    return target.with_suffix(target.suffix + ".manifest.json")


def write_onnx_manifest(manifest: Mapping[str, Any]) -> Path:
    """以同目录临时文件 + 原子替换写出 manifest。"""
    _validate_onnx_manifest(manifest)
    target = Path(str(manifest["target"]))
    path = manifest_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(dict(manifest), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return path


def load_onnx_manifest(target: str | Path) -> dict[str, Any]:
    path = manifest_path(target)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read ONNX manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("ONNX manifest must be a JSON object")
    _validate_onnx_manifest(value)
    return value


def inspect_onnx_manifest_reuse(
    target: str | Path,
    *,
    model_id: str,
    source_files: Mapping[str, str | Path | None],
    runtime_contract: Mapping[str, Any],
) -> OnnxReuseDecision:
    """返回可审计的严格 skip-existing 判定。

    判定失败时返回稳定原因码，供 verifier、发布编排器或外部工具复用；任何
    manifest、源文件、runtime contract 或目标字节异常都按不可复用处理。
    """
    try:
        manifest = load_onnx_manifest(target)
    except (OSError, RuntimeError, ValueError):
        return OnnxReuseDecision(False, "manifest_unavailable", None)
    if manifest["model_id"] != model_id:
        return OnnxReuseDecision(False, "model_id_mismatch", manifest)
    try:
        expected = build_onnx_manifest(
            target=target,
            model_id=model_id,
            source_files=source_files,
            runtime_contract=runtime_contract,
        )
    except (OSError, RuntimeError, ValueError, KeyError):
        return OnnxReuseDecision(False, "source_or_target_unavailable", manifest)
    if manifest["source_binding"] != expected["source_binding"]:
        return OnnxReuseDecision(False, "source_binding_mismatch", manifest)
    if manifest["runtime_contract"] != expected["runtime_contract"]:
        return OnnxReuseDecision(False, "runtime_contract_mismatch", manifest)
    if manifest["target"] != expected["target"]:
        return OnnxReuseDecision(False, "target_mismatch", manifest)
    try:
        artifact_matches = manifest["artifact"] == fingerprint_file(target)
    except (OSError, RuntimeError, ValueError):
        artifact_matches = False
    if not artifact_matches:
        return OnnxReuseDecision(False, "artifact_mismatch", manifest)
    return OnnxReuseDecision(True, "reusable", manifest)


def require_reusable_onnx_manifest(
    target: str | Path,
    *,
    model_id: str,
    source_files: Mapping[str, str | Path | None],
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """严格要求已有 ONNX 可复用，失败时抛出带原因码的异常。"""
    decision = inspect_onnx_manifest_reuse(
        target,
        model_id=model_id,
        source_files=source_files,
        runtime_contract=runtime_contract,
    )
    if not decision.reusable:
        raise RuntimeError(
            f"ONNX artifact is not reusable ({decision.reason}): {Path(target)}"
        )
    return dict(decision.manifest or {})


def verify_onnx_manifest_sources(manifest: Mapping[str, Any]) -> None:
    """在加载已发布 manifest 后复核源文件仍与发布时一致。"""
    _validate_onnx_manifest(manifest)
    verify_source_binding(manifest["source_binding"])
