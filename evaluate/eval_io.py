"""评估侧低层音频、SCP 与推理 artifact 校验。

本模块不依赖 pipeline orchestration，供任务公共逻辑与顶层流水线共同使用。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from core.utils.inference_artifacts import (
    InferenceResult,
    fingerprint_file,
    inference_output_lock,
    load_run_manifest,
    stat_audio_identity,
)
from evaluate.inference_components import validate_effective_config_identity
from tools.shared.scp_io import write_scp_entries
from tools.shared.scp_provenance import (
    build_scp_provenance,
    parse_scp_snapshot,
    scp_provenance_fresh,
    scp_sidecar_path,
    validate_scp_provenance,  # 评估侧对外暴露的 canonical sidecar 校验入口
    write_scp_provenance,
)

logger = logging.getLogger(__name__)


def _read_file_snapshot(path: str | Path) -> tuple[bytes, str]:
    """一次读盘同时取得内容与 sha256，避免解析/指纹之间的 TOCTOU。"""
    content = Path(path).read_bytes()
    return content, hashlib.sha256(content).hexdigest()


def _directory_snapshot(
    root: Path, pattern: str, recursive: bool
) -> tuple[list[Path], dict]:
    if not root.is_dir():
        raise FileNotFoundError(f"SCP source directory does not exist: {root}")
    globber = root.rglob if recursive else root.glob
    files = sorted(path for path in globber(pattern) if path.is_file())
    identities = []
    for path in files:
        if path.is_symlink():
            raise ValueError(f"音频 payload 不允许符号链接: {path}")
        identities.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                **{
                    key: value
                    for key, value in stat_audio_identity(path).items()
                    if key != "path"
                },
            }
        )
    snapshot = {
        "algorithm": "stat-v1",
        "digest": hashlib.sha256(
            json.dumps(
                identities, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }
    return files, snapshot


def generate_scp_from_dir(
    input_dir: str | Path,
    pattern: str = "*.wav",
    scp_path: Optional[str | Path] = None,
    recursive: bool = True,
    *,
    require_non_empty: bool = False,
) -> Path:
    """仅按目录路径生成 2 列 SCP，并写入可校验 provenance sidecar。

    Args:
        require_non_empty: 为 True 时空源必须 fail-fast；评估流水线要求非空清单，
            默认 False 保留"生成空 SCP"的历史行为。
    """
    root = Path(input_dir).resolve()
    target = Path(scp_path) if scp_path is not None else root.parent / f"{root.name}.scp"
    files, snapshot = _directory_snapshot(root, pattern, recursive)
    if require_non_empty and not files:
        raise ValueError(f"cannot generate {target}: source contains zero entries")
    provenance = {
        "kind": "generated",
        "source": str(root),
        "pattern": pattern,
        "recursive": recursive,
        "source_snapshot": snapshot,
    }
    if scp_provenance_fresh(target, provenance):
        return target

    entries = [(path.stem, str(path)) for path in files]
    uids = [entry[0] for entry in entries]
    if len(uids) != len(set(uids)):
        raise ValueError(f"duplicate uid while generating {target}; use unique file stems")
    write_scp_entries(entries, target)
    write_scp_provenance(
        target,
        build_scp_provenance(target.read_bytes(), scp_path=target, **provenance),
    )
    logger.info("Generated SCP: %s (%d files)", target, len(entries))
    return target


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """读取音频并转换为单声道 float32。"""
    data, sample_rate = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, int(sample_rate)


def read_scp(scp_path: str | Path) -> dict[str, Path]:
    """从单次 bytes snapshot 解析 SCP；sidecar 存在时同时验证其哈希。"""
    source = Path(scp_path).expanduser().resolve()
    content, digest = _read_file_snapshot(source)
    parsed = parse_scp_snapshot(source, content)
    sidecar = scp_sidecar_path(source)
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"SCP provenance sidecar is invalid: {sidecar}") from exc
        if payload.get("scp_sha256") != digest:
            raise ValueError(f"SCP provenance fingerprint mismatch: {source}")
    resolved: dict[str, Path] = {}
    for uid, value in parsed.items():
        path = Path(value).expanduser()
        resolved[uid] = (
            path.resolve() if path.is_absolute() else (source.parent / path).resolve()
        )
    return resolved


def validate_inference_artifacts(
    output_dir: str | Path,
    *,
    config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    effective_config: dict | None = None,
    expected_effective_config_identity: dict | str | None = None,
) -> tuple[dict, list[InferenceResult]]:
    """锁内校验 manifest、effective mapping、provenance 与输出结构身份。"""
    with inference_output_lock(output_dir) as output:
        manifest, results = load_run_manifest(output / "run_manifest.json")
        if config is not None or checkpoint is not None:
            expected_config = str(Path(config).resolve()) if config is not None else None
            expected_checkpoint = (
                str(Path(checkpoint).resolve()) if checkpoint is not None else None
            )
            if (
                manifest["config"] != expected_config
                or manifest["checkpoint"] != expected_checkpoint
            ):
                raise ValueError(
                    "inference manifest config/checkpoint provenance does not match this pipeline run"
                )
            artifact_files = manifest["artifact_fingerprint"]["files"]
            for name, path in (("config", config), ("checkpoint", checkpoint)):
                expected_fingerprint = artifact_files.get(name)
                current_fingerprint = None if path is None else fingerprint_file(path)
                if current_fingerprint != expected_fingerprint:
                    raise ValueError(f"inference manifest {name} fingerprint is stale")

        manifest_identity = manifest["artifact_fingerprint"]["parameters"].get(
            "effective_config_identity"
        )
        if not isinstance(manifest_identity, dict):
            raise ValueError("inference manifest missing effective config identity")
        if effective_config is not None:
            actual_identity = validate_effective_config_identity(
                effective_config, expected_effective_config_identity
            )
            if manifest_identity != actual_identity:
                raise ValueError("inference manifest effective config identity mismatch")
        elif expected_effective_config_identity is not None:
            expected_identity = (
                {
                    "schema": "effective-merged-config-v1",
                    "algorithm": "sha256-json-v1",
                    "sha256": expected_effective_config_identity,
                }
                if isinstance(expected_effective_config_identity, str)
                else dict(expected_effective_config_identity)
            )
            if manifest_identity != expected_identity:
                raise ValueError("inference manifest effective config identity mismatch")

        produced_map = read_scp(output / "inf.scp")
        successful = {
            result.uid: result
            for result in results
            if result.status in {"ok", "skipped"}
        }
        if set(produced_map) != set(successful):
            raise ValueError("inf.scp uids do not match manifest ok_uids + skipped_uids")
        for uid, result in successful.items():
            scp_output = Path(produced_map[uid]).resolve()
            manifest_output = Path(result.output_path).resolve()
            if scp_output != manifest_output:
                raise ValueError(
                    f"inf.scp output path does not match manifest for uid={uid}"
                )
            if stat_audio_identity(manifest_output) != result.output_identity:
                raise ValueError(f"inference output identity mismatch for uid={uid}")
        return manifest, results
