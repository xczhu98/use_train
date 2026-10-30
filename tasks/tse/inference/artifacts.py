"""TSE 推理 artifact 的身份、复用、发布与独占运行 Module。"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

from core.utils.config import load_merged_config
from core.utils.file_transaction import publish_staged_files
from core.utils.inference_artifacts import (
    atomic_write_json,
    build_content_fingerprint,
    canonical_json_value,
    collect_source_tree_files,
    fingerprint_file,
    fingerprint_json,
    audio_identity_from_stat,
    ensure_directory_no_symlinks,
    open_parent_directory_no_symlinks,
    open_regular_file_no_symlinks,
    stat_audio_identity_no_symlinks,
)
from core.utils.safe_load import safe_torch_load
from tasks.tse.models.speaker.config import (
    resolve_existing_file,
    resolve_external_speaker_checkpoint,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - 项目部署目标为 Linux。
    fcntl = None


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_log = logging.getLogger(__name__)
ARTIFACT_SCHEMA = "tse-inference-artifact-v6"
MANIFEST_SCHEMA = "tse-inference-manifest-v4"
UID_SCHEME = "sha256-canonical-json-dataset-mixture-cues-v2-128"
CUE_BINDING_SCHEMA = "tse-inference-cue-binding-v1"
SPEAKER_FRONTEND_SCHEMA = "tse-speaker-frontend-v1"
CUE_ROLES = (
    "speaker_waveform",
    "speaker_embedding",
    "spatial",
    "visual",
    "textual",
)
OUTPUT_LOCK_FILENAME = ".inference.lock"
IMPLEMENTATION_PATHS = (
    "inference.py",
    "evaluate/eval_pipeline_tse.py",
    "tasks/__init__.py",
    "tasks/descriptors.py",
    "tasks/registry.py",
    "tasks/tse/inference",
    "tasks/tse/models",
    "core/stft.py",
    "core/models/build.py",
    "core/utils/config.py",
    "core/utils/registry.py",
    "core/utils/safe_load.py",
    "core/utils/inference_artifacts.py",
    "tasks/tse/model_io.py",
)
def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def normalize_cue_declaration(cues: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """把 cue 声明规范化为 compact canonical identity。

    文件 cue 绑定 canonical 绝对路径，inline cue 只绑定其 canonical JSON 的
    SHA-256；声明进入 UID identity，因此同一 mixture 换用不同 cue 资源时 UID
    必然不同。
    """
    if not isinstance(cues, Mapping) or not cues:
        raise ValueError("TSE cue declaration must be a non-empty mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for role, entry in cues.items():
        if role not in CUE_ROLES:
            raise ValueError(f"unknown TSE cue role: {role!r}")
        if not isinstance(entry, Mapping):
            raise TypeError(f"TSE cue declaration for {role!r} must be a mapping")
        kind = entry.get("kind")
        if kind == "file":
            path = entry.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(f"file cue {role!r} requires a non-empty path")
            normalized[role] = {"kind": "file", "path": str(_absolute(path))}
        elif kind == "inline":
            if "value" not in entry:
                raise ValueError(f"inline cue {role!r} requires a value")
            normalized[role] = {
                "kind": "inline",
                "sha256": fingerprint_json(canonical_json_value(entry["value"])),
            }
        else:
            raise ValueError(f"TSE cue {role!r} kind must be 'file' or 'inline'")
    return dict(sorted(normalized.items()))


def _validate_identity_cues(cues: Any) -> None:
    """校验 manifest 中已 compact 化的 cue identity。"""
    if not isinstance(cues, Mapping) or not cues:
        raise ValueError("TSE identity cues must be a non-empty mapping")
    for role, entry in cues.items():
        if role not in CUE_ROLES or not isinstance(entry, Mapping):
            raise ValueError("TSE identity cue entry is invalid")
        kind = entry.get("kind")
        if kind == "file":
            if not isinstance(entry.get("path"), str) or not entry["path"]:
                raise ValueError("TSE identity file cue requires a path")
        elif kind == "inline":
            if re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256"))) is None:
                raise ValueError("TSE identity inline cue requires a sha256")
        else:
            raise ValueError("TSE identity cue kind is invalid")


def row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """返回 UID 所绑定的 dataset/mixture 与 cue identity。

    无 cue 声明的历史行继续绑定 ``enrolment_speakers_utterance``；cue-aware 行
    改为绑定全部实际 cue 文件路径与 canonical inline cue JSON 摘要，此时允许
    不提供 enrollment。
    """
    dataset = row.get("dataset")
    mixture = row.get("mixture_utterance")
    for key, value in (("dataset", dataset), ("mixture_utterance", mixture)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"TSE row {key} 必须是非空字符串")
    identity: dict[str, Any] = {"dataset": dataset, "mixture": mixture}
    cues = row.get("cues")
    if cues is None:
        enrollment = row.get("enrolment_speakers_utterance")
        if not isinstance(enrollment, str) or not enrollment:
            raise ValueError(
                "TSE row 必须提供 cues 声明或非空 enrolment_speakers_utterance"
            )
        identity["enrollment"] = enrollment
        return identity
    identity["cues"] = normalize_cue_declaration(cues)
    return identity


def make_uid(row: Mapping[str, Any]) -> str:
    """返回 canonical row identity JSON 的 SHA-256 前 32 hex。"""
    return hashlib.sha256(_canonical_json_bytes(row_identity(row))).hexdigest()[:32]



def output_basename(row: Mapping[str, Any]) -> str:
    """返回仅由 canonical UID 构成的安全输出 basename。"""
    return make_uid(row)


def validate_batch(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """整批预检 identity、UID hash 与输出 basename，任一重复即整批拒绝。"""
    seen_identities: dict[bytes, int] = {}
    seen_uids: dict[str, int] = {}
    seen_basenames: dict[str, int] = {}
    uids: list[str] = []
    for index, row in enumerate(rows):
        identity_bytes = _canonical_json_bytes(row_identity(row))
        if identity_bytes in seen_identities:
            raise ValueError(
                "TSE batch 存在重复 identity: "
                f"rows {seen_identities[identity_bytes]} and {index}"
            )
        seen_identities[identity_bytes] = index
        uid = make_uid(row)
        if uid in seen_uids:
            raise ValueError(
                f"TSE batch 存在重复 UID hash: rows {seen_uids[uid]} and {index}: {uid}"
            )
        seen_uids[uid] = index
        basename = output_basename(row)
        if basename in seen_basenames:
            raise ValueError(
                "TSE batch 存在重复 output basename: "
                f"rows {seen_basenames[basename]} and {index}: {basename}"
            )
        seen_basenames[basename] = index
        uids.append(uid)
    return uids


def implementation_fingerprint(
    project_root: str | Path = _PROJECT_ROOT,
    source_paths: tuple[str, ...] = IMPLEMENTATION_PATHS,
) -> dict[str, Any]:
    """返回 TSE 推理及任务 descriptor/registry 加载路径的内容身份。"""
    files = collect_source_tree_files(
        project_root, source_paths, name_prefix="implementation"
    )
    return build_content_fingerprint(files, parameters={"source_paths": list(source_paths)})


def _artifact_meta(artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": ARTIFACT_SCHEMA,
        "fingerprint": fingerprint_json(artifacts),
        "artifacts": artifacts,
    }


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _open_parent_without_symlinks(
    path: str | Path, *, create_missing: bool = False
) -> tuple[Path, int]:
    return open_parent_directory_no_symlinks(
        path, create_missing=create_missing
    )


def ensure_output_directory(path: str | Path) -> Path:
    """创建目录并拒绝其任一 ancestor 或 leaf symlink。"""
    return ensure_directory_no_symlinks(path)


def _audio_identity_from_stat(path: Path, file_stat: os.stat_result) -> dict[str, Any]:
    return audio_identity_from_stat(path, file_stat)


def stat_audio_artifact(path: str | Path) -> dict[str, Any]:
    """拒绝 leaf/ancestor symlink，并仅用 lstat 建立音频身份。"""
    return stat_audio_identity_no_symlinks(path)


def _open_regular_fd(path: str | Path, *, label: str) -> tuple[Path, int]:
    return open_regular_file_no_symlinks(path, label=label)


@dataclass(frozen=True)
class ImmutableFileSnapshot:
    """loader 实际消费的不可变字节与其 canonical 内容身份。"""

    path: Path
    content: bytes
    identity: dict[str, Any]


def capture_immutable_file(path: str | Path, *, label: str) -> ImmutableFileSnapshot:
    """从稳定 FD 一次读取字节，拒绝路径替换并绑定 SHA-256 身份。"""
    canonical = _canonical_file(path, label)
    absolute, descriptor = _open_regular_fd(canonical, label=label)
    try:
        before = os.fstat(descriptor)
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            content = stream.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not os.path.samestat(before, after) or (
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"{label} changed while snapshotting: {absolute}")
    identity = {
        "path": str(absolute),
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    return ImmutableFileSnapshot(absolute, content, identity)


def load_immutable_checkpoint(
    path: str | Path,
    loader: Callable[[Any], Any] = safe_torch_load,
    *,
    label: str = "checkpoint",
) -> tuple[Any, dict[str, Any]]:
    """让 checkpoint loader 消费已固化的 bytes，而不是再次按路径打开。"""
    import io

    snapshot = capture_immutable_file(path, label=label)
    stream = io.BytesIO(snapshot.content)
    stream.name = str(snapshot.path)
    loaded = loader(stream)
    verify_file_identity(snapshot.identity, label=label)
    return loaded, dict(snapshot.identity)


def load_bound_payload(
    path: str | Path,
    loader: Callable[[Any], Any],
    expected_identity: Mapping[str, Any] | None = None,
    *,
    label: str = "输入",
) -> tuple[Any, dict[str, Any]]:
    """经稳定 FD 读取任意推理输入，并在 loader 前后及路径复核同一身份。

    mixture/enrollment 与 cue 文件（npy、video、JSON）共用同一条身份绑定路径，
    避免 cue 输入绕过 leaf/ancestor symlink 与读取期间变化的检查。
    """
    absolute, descriptor = _open_regular_fd(path, label=f"{label} path")
    try:
        before = _audio_identity_from_stat(absolute, os.fstat(descriptor))
        if expected_identity is not None and dict(expected_identity) != before:
            raise RuntimeError(f"{label}身份与绑定值不一致: {absolute}")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            loaded = loader(stream)
        after = _audio_identity_from_stat(absolute, os.fstat(descriptor))
    finally:
        os.close(descriptor)
    current = stat_audio_artifact(absolute)
    if before != after or before != current:
        raise RuntimeError(f"{label}在读取期间发生变化: {absolute}")
    return loaded, before


def load_bound_audio(
    path: str | Path,
    loader: Callable[[Any], Any],
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """经稳定 FD 读取音频，并在 loader 前后及路径复核同一身份。"""
    return load_bound_payload(path, loader, expected_identity, label="音频输入")


def verify_input_identities(inputs: Mapping[str, Mapping[str, Any]]) -> None:
    """在输出发布前复核当前输入仍与绑定身份完全一致。"""
    for role, expected in sorted(inputs.items()):
        if not isinstance(expected, Mapping):
            raise TypeError(f"input identity for {role!r} must be a mapping")
        path = expected.get("path")
        if not isinstance(path, str):
            raise ValueError(f"input identity for {role!r} has no path")
        current = stat_audio_artifact(path)
        if current != dict(expected):
            raise RuntimeError(f"音频输入在输出发布前发生变化: role={role}")


@contextmanager
def exclusive_output_run(output_dir: str | Path) -> Iterator[Path]:
    """非阻塞独占整个 output_dir 生命周期；并发 writer 在进入前失败。"""
    if fcntl is None:
        raise RuntimeError("TSE output locking requires fcntl")
    output = ensure_output_directory(output_dir)
    _, parent_fd = _open_parent_without_symlinks(output / OUTPUT_LOCK_FILENAME)
    lock_fd: int | None = None
    try:
        lock_fd = os.open(
            OUTPUT_LOCK_FILENAME,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o644,
            dir_fd=parent_fd,
        )
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise RuntimeError(f"TSE output lock is not a regular file: {output}")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"TSE output_dir 已有 writer: {output}") from exc
        yield output
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(parent_fd)


def _canonical_file(path: str | Path, label: str) -> Path:
    candidate = _absolute(path)
    try:
        absolute, descriptor = _open_regular_fd(candidate, label=label)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} must point to an existing file: {path}") from exc
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
    return absolute


def validate_loaded_file_identity(
    identity: Mapping[str, Any], expected_path: str | Path, label: str
) -> dict[str, Any]:
    """验证 loader 取得的 canonical 内容身份，不重新读取已加载文件。"""
    expected = _canonical_file(expected_path, label)
    expected_text = str(expected)
    if not isinstance(identity, Mapping):
        raise TypeError(f"{label} loaded identity must be a mapping")
    path = identity.get("path")
    size = identity.get("size")
    sha256 = identity.get("sha256")
    if path != expected_text:
        raise ValueError(
            f"{label} loaded identity path mismatch: expected {expected_text}, got {path!r}"
        )
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{label} loaded identity has invalid size: {size!r}")
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError(f"{label} loaded identity has invalid sha256")
    return {"path": expected_text, "size": size, "sha256": sha256}


def verify_file_identity(identity: Mapping[str, Any], *, label: str) -> None:
    """复核当前路径仍指向已绑定的普通文件内容。"""
    path = identity.get("path") if isinstance(identity, Mapping) else None
    if not isinstance(path, str):
        raise ValueError(f"{label} identity has no path")
    try:
        current = fingerprint_file(path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"{label} cannot be verified: {path}") from exc
    if current != dict(identity):
        raise RuntimeError(f"{label} changed after identity capture: {path}")


def _verify_implementation_identity(identity: Mapping[str, Any]) -> None:
    files = identity.get("files") if isinstance(identity, Mapping) else None
    if not isinstance(files, Mapping):
        raise ValueError("inference implementation identity has no files")
    for name, file_identity in sorted(files.items()):
        if not isinstance(file_identity, Mapping):
            raise ValueError(f"inference implementation identity is invalid: {name}")
        verify_file_identity(file_identity, label=f"inference implementation {name}")
    parameters = identity.get("parameters")
    rebuilt = build_content_fingerprint(
        {name: value["path"] for name, value in files.items()},
        parameters=dict(parameters) if isinstance(parameters, Mapping) else {},
    )
    if rebuilt != dict(identity):
        raise RuntimeError("inference implementation identity changed")


def _verify_content_identity(identity: Mapping[str, Any], *, label: str) -> None:
    """复核 ``build_content_fingerprint`` 产生的具名内容身份。"""
    files = identity.get("files") if isinstance(identity, Mapping) else None
    parameters = identity.get("parameters") if isinstance(identity, Mapping) else None
    if not isinstance(files, Mapping) or not isinstance(parameters, Mapping):
        raise ValueError(f"{label} identity is invalid")
    paths: dict[str, str | None] = {}
    for name, file_identity in sorted(files.items()):
        if file_identity is None:
            paths[name] = None
            continue
        if not isinstance(file_identity, Mapping):
            raise ValueError(f"{label} file identity is invalid: {name}")
        verify_file_identity(file_identity, label=f"{label} {name}")
        paths[name] = file_identity["path"]
    rebuilt = build_content_fingerprint(paths, parameters=dict(parameters))
    if rebuilt != dict(identity):
        raise RuntimeError(f"{label} identity changed")


def model_checkpoint_path(model_path: str | Path) -> Path:
    """按 TSEExtractor 规则解析并 canonicalize 实际主 checkpoint。"""
    source = Path(model_path).expanduser()
    candidate = source / "avg_model.pt" if source.is_dir() else source
    return _canonical_file(candidate, "model checkpoint")


def model_config_path(
    model_path: str | Path,
    config_path: str | Path | None = None,
) -> Path | None:
    """按 TSEExtractor 规则解析入口外部配置。"""
    source = Path(model_path).expanduser()
    if config_path is not None:
        candidate = config_path
    elif source.is_dir():
        candidate = source / "config.yaml"
    else:
        return None
    return _canonical_file(candidate, "model config")


def _resolved_config_identity(config: Mapping[str, Any] | None) -> dict[str, str] | None:
    if config is None:
        return None
    return {"algorithm": "sha256-json-v1", "sha256": fingerprint_json(dict(config))}


def _json_identity(value: Any) -> dict[str, str]:
    return {
        "algorithm": "sha256-json-v1",
        "sha256": fingerprint_json(canonical_json_value(value)),
    }


def _speaker_frontend_checkpoint(
    model_args: Mapping[str, Any], *, repo_root: str | Path
) -> dict[str, Any] | None:
    """解析配置声明的 speaker encoder checkpoint，并绑定其内容身份。

    canonical ``speaker_model.speaker_encoder.pretrained`` 与 modular
    ``cues.speaker.encoder.checkpoint`` 两种键都显式支持；两者同时声明不同文件
    时 fail-closed。
    """
    external = resolve_external_speaker_checkpoint(model_args, repo_root=repo_root)
    modular: Path | None = None
    cues = model_args.get("cues")
    if isinstance(cues, Mapping):
        speaker = cues.get("speaker")
        if isinstance(speaker, Mapping):
            encoder = speaker.get("encoder")
            if isinstance(encoder, Mapping):
                raw = encoder.get("checkpoint")
                if isinstance(raw, str) and raw:
                    modular = resolve_existing_file(
                        raw,
                        "model.args.cues.speaker.encoder.checkpoint",
                        repo_root=repo_root,
                    )
    if external is not None and modular is not None and external != modular:
        raise ValueError(
            "model.args 同时声明了不同的 speaker encoder checkpoint: "
            f"{external} != {modular}"
        )
    path = modular if modular is not None else external
    return fingerprint_file(path) if path is not None else None


def _speaker_frontend_fbank(model_args: Mapping[str, Any]) -> Any | None:
    """定位配置声明的 fbank 段；两种键同时声明不同内容时 fail-closed。"""
    candidates: list[Any] = []
    cues = model_args.get("cues")
    if isinstance(cues, Mapping):
        speaker = cues.get("speaker")
        if isinstance(speaker, Mapping):
            encoder = speaker.get("encoder")
            if isinstance(encoder, Mapping) and "fbank" in encoder:
                candidates.append(encoder["fbank"])
    speaker_model = model_args.get("speaker_model")
    if isinstance(speaker_model, Mapping) and "fbank" in speaker_model:
        candidates.append(speaker_model["fbank"])
    if not candidates:
        return None
    first = canonical_json_value(candidates[0])
    if any(canonical_json_value(item) != first for item in candidates[1:]):
        raise ValueError("model.args 同时声明了不同的 speaker fbank 配置")
    return first


def speaker_frontend_identity(
    config: Mapping[str, Any] | None,
    *,
    repo_root: str | Path = _PROJECT_ROOT,
) -> dict[str, Any] | None:
    """从 resolved config 捕获 speaker 前端身份：checkpoint / fbank / preprocess。

    三个身份都由配置派生，因此 ``loaded`` 与 ``inspection`` 两条路径得到同一
    结果。checkpoint 额外按内容指纹绑定实际文件；模型构造后还必须复核
    ``*_ckpt_path`` 属性与该身份一致（见 ``TSEExtractor``）。
    """
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise TypeError("resolved config must be a mapping")
    model = config.get("model")
    if not isinstance(model, Mapping):
        return None
    args = model.get("args")
    if not isinstance(args, Mapping):
        return None
    data = config.get("data")
    sample_rate = args.get("sr")
    if sample_rate is None and isinstance(data, Mapping):
        sample_rate = data.get("sample_rate")
    fbank = _speaker_frontend_fbank(args)
    return {
        "schema": SPEAKER_FRONTEND_SCHEMA,
        "checkpoint": _speaker_frontend_checkpoint(args, repo_root=repo_root),
        "fbank": None if fbank is None else _json_identity(fbank),
        "preprocess": _json_identity(
            {"stft": config.get("stft"), "sample_rate": sample_rate}
        ),
    }


def _verify_speaker_frontend(frontend: Mapping[str, Any]) -> None:
    if not isinstance(frontend, Mapping) or frontend.get("schema") != SPEAKER_FRONTEND_SCHEMA:
        raise ValueError("TSE speaker frontend schema is invalid")
    checkpoint = frontend.get("checkpoint")
    if checkpoint is not None:
        if not isinstance(checkpoint, Mapping):
            raise ValueError("TSE speaker frontend checkpoint identity is invalid")
        verify_file_identity(checkpoint, label="speaker frontend checkpoint")
    for key in ("fbank", "preprocess"):
        value = frontend.get(key)
        if value is None:
            continue
        if (
            not isinstance(value, Mapping)
            or value.get("algorithm") != "sha256-json-v1"
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256"))) is None
        ):
            raise ValueError(f"TSE speaker frontend {key} identity is invalid")


def _cue_binding(cues: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """把 cue 声明转成 artifact 中的绑定段：角色 kind + inline cue canonical JSON。

    空声明（``None`` 或 ``{}``）表示该次推理没有 cue 绑定，写入 ``None``。
    """
    if not cues:
        return None
    normalized = normalize_cue_declaration(cues)
    roles = {role: entry["kind"] for role, entry in normalized.items()}
    inline: dict[str, dict[str, Any]] = {}
    for role, entry in sorted(normalized.items()):
        if entry["kind"] != "inline":
            continue
        value = canonical_json_value(cues[role]["value"])
        if fingerprint_json(value) != entry["sha256"]:
            raise ValueError(f"TSE inline cue {role!r} identity does not bind its value")
        inline[role] = {
            "algorithm": "sha256-json-v1",
            "sha256": entry["sha256"],
            "value": value,
        }
    return {"schema": CUE_BINDING_SCHEMA, "roles": roles, "inline": inline}


def verify_cue_binding(cues: Any, inputs: Mapping[str, Any]) -> None:
    """复核 cue 绑定段：角色 kind、文件 cue 已绑定、inline cue 与其 payload 一致。"""
    if not isinstance(cues, Mapping) or cues.get("schema") != CUE_BINDING_SCHEMA:
        raise ValueError("TSE cue binding schema is invalid")
    roles = cues.get("roles")
    if not isinstance(roles, Mapping) or not roles:
        raise ValueError("TSE cue binding has no roles")
    for role, kind in roles.items():
        if role not in CUE_ROLES:
            raise ValueError(f"TSE cue binding has unknown role: {role!r}")
        if kind == "file":
            if role not in inputs:
                raise ValueError(f"TSE file cue {role!r} is not bound in inputs")
        elif kind == "inline":
            if role in inputs:
                raise ValueError(f"TSE inline cue {role!r} must not be bound as a file")
        else:
            raise ValueError(f"TSE cue binding role {role!r} kind is invalid")
    inline = cues.get("inline")
    if not isinstance(inline, Mapping):
        raise ValueError("TSE cue binding has no inline section")
    for role, entry in inline.items():
        if roles.get(role) != "inline":
            raise ValueError(f"TSE inline cue {role!r} is not declared inline")
        if not isinstance(entry, Mapping) or entry.get("algorithm") != "sha256-json-v1":
            raise ValueError(f"TSE inline cue {role!r} identity is invalid")
        value = canonical_json_value(entry.get("value"))
        if entry.get("sha256") != fingerprint_json(value):
            raise ValueError(f"TSE inline cue {role!r} identity does not bind its value")


def _inspect_model_artifacts(
    model_path: str | Path,
    config_path: str | Path | None,
) -> tuple[
    dict[str, Any] | None,
    Path | None,
    dict[str, Any] | None,
    Path,
    dict[str, Any],
    dict[str, str] | None,
]:
    """恢复 skip 所需依赖；不会构造模型或 SpeakerEncoder。"""
    checkpoint = model_checkpoint_path(model_path)
    loaded_checkpoint, checkpoint_identity = load_immutable_checkpoint(
        checkpoint,
        loader=lambda stream: safe_torch_load(stream, map_location="cpu"),
        label="model checkpoint",
    )
    config_file = model_config_path(model_path, config_path)
    if config_file is not None:
        before = fingerprint_file(config_file)
        config = load_merged_config(before["path"])
        after = fingerprint_file(before["path"])
        if before != after:
            raise RuntimeError(f"模型配置在加载期间发生变化: {config_file}")
        if not isinstance(config, Mapping):
            raise TypeError("load_merged_config must return a mapping")
        loaded = dict(config)
        return (
            loaded,
            Path(after["path"]),
            after,
            checkpoint,
            checkpoint_identity,
            _resolved_config_identity(loaded),
        )

    if not isinstance(loaded_checkpoint, Mapping):
        return None, None, None, checkpoint, checkpoint_identity, None
    for key in ("resolved_config", "config"):
        config = loaded_checkpoint.get(key)
        if isinstance(config, Mapping):
            resolved = dict(config)
            return (
                resolved,
                None,
                None,
                checkpoint,
                checkpoint_identity,
                _resolved_config_identity(resolved),
            )
    return None, None, None, checkpoint, checkpoint_identity, None


def verify_artifact_meta(meta: Mapping[str, Any]) -> None:
    """在 manifest/CURRENT 发布前复核全局依赖与全部已绑定输入。"""
    if meta.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("TSE artifact metadata schema is invalid")
    artifacts = meta.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("TSE artifact metadata has no artifacts")
    if meta.get("fingerprint") != fingerprint_json(dict(artifacts)):
        raise ValueError("TSE artifact metadata fingerprint is invalid")

    model = artifacts.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("TSE artifact metadata has no model identity")
    verify_file_identity(model, label="model checkpoint")

    config = artifacts.get("config")
    if config is not None:
        if not isinstance(config, Mapping) or config.get("kind") != "file":
            raise ValueError("TSE artifact config identity is invalid")
        config_identity = {key: value for key, value in config.items() if key != "kind"}
        verify_file_identity(config_identity, label="model config")
        resolved_config = load_merged_config(config_identity["path"])
        verify_file_identity(config_identity, label="model config")
    else:
        checkpoint, _ = load_immutable_checkpoint(
            model["path"],
            loader=lambda stream: safe_torch_load(stream, map_location="cpu"),
            label="model checkpoint",
        )
        resolved_config = None
        if isinstance(checkpoint, Mapping):
            for key in ("resolved_config", "config"):
                value = checkpoint.get(key)
                if isinstance(value, Mapping):
                    resolved_config = dict(value)
                    break

    expected_resolved = artifacts.get("resolved_config")
    if _resolved_config_identity(resolved_config) != expected_resolved:
        raise RuntimeError("resolved model config changed after identity capture")

    speaker = artifacts.get("speaker_checkpoint")
    if speaker is not None:
        if not isinstance(speaker, Mapping):
            raise ValueError("speaker checkpoint identity is invalid")
        verify_file_identity(speaker, label="speaker checkpoint")
    frontend = artifacts.get("speaker_frontend")
    if frontend is not None:
        _verify_speaker_frontend(frontend)
    dataset = artifacts.get("dataset")
    if dataset is not None:
        if not isinstance(dataset, Mapping):
            raise ValueError("dataset identity is invalid")
        _verify_content_identity(dataset, label="dataset")
    implementation = artifacts.get("inference_implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("inference implementation identity is invalid")
    _verify_implementation_identity(implementation)
    _canonical_json_bytes(artifacts.get("parameters"))
    inputs = artifacts.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("TSE artifact metadata has no inputs")
    verify_input_identities(inputs)
    cues = artifacts.get("cues")
    if cues is not None:
        verify_cue_binding(cues, inputs)


def _speaker_checkpoint_from_config(
    config: Mapping[str, Any] | None, *, strict: bool
) -> tuple[Path | None, str | None]:
    if config is None:
        if strict:
            raise ValueError("无法恢复模型配置，不能在 --skip_inference 时验证 speaker artifact")
        return None, None
    model = config.get("model")
    if not isinstance(model, Mapping):
        return None, None
    args = model.get("args")
    if not isinstance(args, Mapping):
        return None, None
    raw_speaker = args.get("speaker_model")
    configured = None
    if isinstance(raw_speaker, Mapping):
        raw_encoder = raw_speaker.get("speaker_encoder")
        if isinstance(raw_encoder, Mapping):
            pretrained = raw_encoder.get("pretrained")
            if isinstance(pretrained, str):
                configured = pretrained
    try:
        return resolve_external_speaker_checkpoint(args, repo_root=_PROJECT_ROOT), configured
    except (TypeError, ValueError):
        if strict:
            raise
        return None, configured


@dataclass(frozen=True)
class TSEInferenceArtifactSnapshot:
    """一次 TSE 推理或 skip inspection 的全局依赖身份快照。"""

    meta: dict[str, Any]
    mode: str
    configured_speaker_encoder_pretrained: str | None = None
    speaker_checkpoint: Path | None = None

    @classmethod
    def from_loaded_extractor(
        cls,
        extractor: Any,
        inference_params: Mapping[str, Any],
        *,
        dataset_identity: Mapping[str, Any] | None = None,
        implementation_project_root: str | Path = _PROJECT_ROOT,
        implementation_source_paths: tuple[str, ...] = IMPLEMENTATION_PATHS,
    ) -> "TSEInferenceArtifactSnapshot":
        loaded = extractor.loaded_artifact_snapshot()
        if not isinstance(loaded, Mapping):
            raise TypeError("TSEExtractor loaded_artifact_snapshot must return a mapping")
        model_path = loaded.get("model_checkpoint")
        model_identity = loaded.get("model_checkpoint_identity")
        if not isinstance(model_path, str) or not isinstance(model_identity, Mapping):
            raise ValueError("TSEExtractor snapshot lacks loaded model identity")
        return cls._build(
            model_checkpoint=model_path,
            model_checkpoint_identity=model_identity,
            config_path=loaded.get("config_path"),
            config_identity=loaded.get("config_identity"),
            resolved_config_identity=loaded.get("resolved_config_identity"),
            speaker_checkpoint=loaded.get("speaker_checkpoint"),
            speaker_checkpoint_identity=loaded.get("speaker_checkpoint_identity"),
            speaker_frontend=loaded.get("speaker_frontend"),
            inference_params=inference_params,
            dataset_identity=dataset_identity,
            mode="loaded",
            configured_speaker_encoder_pretrained=loaded.get(
                "speaker_encoder_pretrained"
            ),
            implementation_project_root=implementation_project_root,
            implementation_source_paths=implementation_source_paths,
        )

    @classmethod
    def inspect_for_skip(
        cls,
        model_path: str | Path,
        config_path: str | Path | None,
        inference_params: Mapping[str, Any],
        *,
        dataset_identity: Mapping[str, Any] | None = None,
        implementation_project_root: str | Path = _PROJECT_ROOT,
        implementation_source_paths: tuple[str, ...] = IMPLEMENTATION_PATHS,
    ) -> "TSEInferenceArtifactSnapshot":
        config, effective_config, config_identity, checkpoint, checkpoint_identity, resolved = _inspect_model_artifacts(
            model_path, config_path
        )
        speaker, configured = _speaker_checkpoint_from_config(config, strict=True)
        speaker_identity = (
            capture_immutable_file(speaker, label="speaker checkpoint").identity
            if speaker is not None
            else None
        )
        return cls._build(
            model_checkpoint=checkpoint,
            model_checkpoint_identity=checkpoint_identity,
            config_path=effective_config,
            config_identity=config_identity,
            resolved_config_identity=resolved,
            speaker_checkpoint=speaker,
            speaker_checkpoint_identity=speaker_identity,
            speaker_frontend=speaker_frontend_identity(config),
            inference_params=inference_params,
            dataset_identity=dataset_identity,
            mode="inspection",
            configured_speaker_encoder_pretrained=configured,
            implementation_project_root=implementation_project_root,
            implementation_source_paths=implementation_source_paths,
        )

    @classmethod
    def _build(
        cls,
        *,
        model_checkpoint: str | Path,
        model_checkpoint_identity: Mapping[str, Any] | None,
        config_path: str | Path | None,
        config_identity: Mapping[str, Any] | None,
        resolved_config_identity: Mapping[str, Any] | None,
        speaker_checkpoint: str | Path | None,
        speaker_checkpoint_identity: Mapping[str, Any] | None,
        inference_params: Mapping[str, Any],
        dataset_identity: Mapping[str, Any] | None = None,
        mode: str,
        configured_speaker_encoder_pretrained: str | None,
        implementation_project_root: str | Path,
        implementation_source_paths: tuple[str, ...],
        speaker_frontend: Mapping[str, Any] | None = None,
    ) -> "TSEInferenceArtifactSnapshot":
        checkpoint = _canonical_file(model_checkpoint, "model checkpoint")
        model = (
            validate_loaded_file_identity(
                model_checkpoint_identity, checkpoint, "model checkpoint"
            )
            if model_checkpoint_identity is not None
            else fingerprint_file(checkpoint)
        )
        if config_identity is not None:
            if config_path is None:
                raise ValueError("config loaded identity requires config_path")
            config = validate_loaded_file_identity(config_identity, config_path, "model config")
            config["kind"] = "file"
        elif config_path is not None:
            config = fingerprint_file(_canonical_file(config_path, "model config"))
            config["kind"] = "file"
        else:
            config = None
        if resolved_config_identity is not None:
            algorithm = resolved_config_identity.get("algorithm")
            sha256 = resolved_config_identity.get("sha256")
            if (
                algorithm != "sha256-json-v1"
                or not isinstance(sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            ):
                raise ValueError("resolved config identity is invalid")
            resolved = {"algorithm": algorithm, "sha256": sha256}
        else:
            resolved = None
            # 兼容直接调用 _build 的旧入口：没有外部配置和上游 loader 身份时，
            # 尝试从同一 immutable checkpoint 派生内嵌 resolved_config 身份。
            # 该入口允许不可反序列化的占位字节（测试/外部调用），此时无法派生
            # embedded config，resolved 保持 None；但降级必须留痕，且 checkpoint
            # 内容一旦变化必须 fail-closed，不得让 None == None 静默通过复核。
            if config_path is None and model_checkpoint_identity is None:
                try:
                    embedded_checkpoint, embedded_identity = load_immutable_checkpoint(
                        checkpoint,
                        loader=lambda stream: safe_torch_load(
                            stream, map_location="cpu"
                        ),
                        label="model checkpoint",
                    )
                except Exception as exc:
                    if fingerprint_file(checkpoint) != model:
                        raise RuntimeError(
                            "model checkpoint changed while deriving resolved config identity"
                        ) from exc
                    _log.warning(
                        "cannot derive embedded resolved_config identity from %s "
                        "(%s: %s); continuing with resolved_config=None",
                        checkpoint,
                        type(exc).__name__,
                        exc,
                    )
                    embedded_checkpoint = None
                else:
                    if embedded_identity != model:
                        raise RuntimeError(
                            "model checkpoint changed while deriving resolved config identity"
                        )
                if isinstance(embedded_checkpoint, Mapping):
                    for key in ("resolved_config", "config"):
                        embedded_config = embedded_checkpoint.get(key)
                        if isinstance(embedded_config, Mapping):
                            resolved = _resolved_config_identity(
                                dict(embedded_config)
                            )
                            break
        if speaker_checkpoint_identity is not None:
            if speaker_checkpoint is None:
                raise ValueError(
                    "speaker checkpoint loaded identity requires speaker_checkpoint"
                )
            speaker = validate_loaded_file_identity(
                speaker_checkpoint_identity, speaker_checkpoint, "speaker checkpoint"
            )
            speaker_path = Path(speaker["path"])
        elif speaker_checkpoint is not None:
            raise ValueError(
                "speaker checkpoint metadata requires the identity captured by its loader"
            )
        else:
            speaker = None
            speaker_path = None
        dataset = None
        if dataset_identity is not None:
            _verify_content_identity(dataset_identity, label="dataset")
            dataset = dict(dataset_identity)
        if speaker_frontend is None and config_path is not None:
            # 未显式传入时按同一份入口配置派生，保证兼容入口与 inspection 得到
            # 相同的前端身份；内嵌配置由调用方显式传入派生结果。
            speaker_frontend = speaker_frontend_identity(
                load_merged_config(config_path)
            )
        artifacts = {
            "model": model,
            "config": config,
            "resolved_config": resolved,
            "speaker_checkpoint": speaker,
            "speaker_frontend": (
                None if speaker_frontend is None else dict(speaker_frontend)
            ),
            "dataset": dataset,
            "inference_implementation": implementation_fingerprint(
                implementation_project_root, implementation_source_paths
            ),
            "parameters": dict(inference_params),
            "inputs": {},
            "cues": None,
        }
        return cls(
            meta=_artifact_meta(artifacts),
            mode=mode,
            configured_speaker_encoder_pretrained=configured_speaker_encoder_pretrained,
            speaker_checkpoint=speaker_path,
        )

    def source_file_identities(self) -> dict[str, dict[str, Any]]:
        """返回公共推理/ONNX 使用的已加载身份，不按当前路径重新绑定。"""
        artifacts = self.meta["artifacts"]
        identities = {}
        for role, key in (
            ("checkpoint", "model"),
            ("config", "config"),
            ("speaker_checkpoint", "speaker_checkpoint"),
        ):
            identity = artifacts.get(key)
            if identity is not None:
                identities[role] = {
                    field: identity[field] for field in ("path", "size", "sha256")
                }
        return identities

    def bind_inputs(
        self,
        input_paths: Mapping[str, Path],
        *,
        cues: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """绑定本次实际消费的文件输入与 cue 声明。

        ``input_paths`` 覆盖 mixture、enrollment 与全部文件 cue；``cues`` 是
        canonical cue 声明（文件 cue 与 inline cue），会连同 inline cue JSON
        一起进入 artifact fingerprint。
        """
        artifacts = dict(self.meta["artifacts"])
        artifacts["inputs"] = {
            name: stat_audio_artifact(path) for name, path in sorted(input_paths.items())
        }
        artifacts["cues"] = _cue_binding(cues)
        return _artifact_meta(artifacts)

    def verify_inputs(self, expected: Mapping[str, Any]) -> None:
        artifacts = expected.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("expected artifact metadata has no artifacts")
        inputs = artifacts.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("expected artifact metadata has no inputs")
        verify_input_identities(inputs)

    def matches_sidecar(
        self,
        sidecar: Mapping[str, Any] | None,
        expected: Mapping[str, Any],
        output_path: Path | None = None,
    ) -> bool:
        return meta_matches(sidecar, expected, output_path)

    def verify_global_identity(self) -> None:
        verify_artifact_meta(self.meta)

    def publish_audio_output(
        self,
        output_path: Path,
        sidecar_path: Path,
        waveform: Any,
        sample_rate: int,
        expected: Mapping[str, Any],
        *,
        output_dir: Path | None = None,
    ) -> None:
        """委托唯一 WAV+sidecar 集合事务入口。"""
        publish_audio_output(
            output_path,
            sidecar_path,
            waveform,
            sample_rate,
            expected,
            output_dir=output_dir,
        )

    def write_output_sidecar(
        self, sidecar_path: Path, output_path: Path, expected: Mapping[str, Any]
    ) -> None:
        write_output_sidecar(sidecar_path, output_path, expected)


def _create_staging_file(target: Path, *, suffix: str) -> tuple[int, Path]:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=suffix, dir=str(target.parent)
    )
    return descriptor, Path(raw_path)


def _predicted_audio_identity(staging_path: Path, output_path: Path) -> dict[str, Any]:
    identity = stat_audio_artifact(staging_path)
    identity["path"] = str(_absolute(output_path))
    return identity


def _publish_audio_output_locked(
    output_path: Path,
    sidecar_path: Path,
    waveform: Any,
    sample_rate: int,
    expected: Mapping[str, Any],
) -> None:
    """在调用方持有 output-dir 锁时提交不可拆分的 WAV+sidecar 集合。"""
    import soundfile as sf

    output = _absolute(output_path)
    sidecar = _absolute(sidecar_path)
    if output.parent != sidecar.parent:
        raise ValueError("WAV and sidecar must share one parent directory")
    if sidecar.name != f"{output.stem}.meta.json":
        raise ValueError("sidecar name must match the WAV basename")
    ensure_output_directory(output.parent)
    artifacts = expected.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get("inputs"), Mapping
    ):
        raise ValueError("expected artifact metadata has no inputs")
    inputs = artifacts["inputs"]
    staged: list[Path] = []
    try:
        audio_fd, staged_audio = _create_staging_file(
            output, suffix=f".tmp{output.suffix}"
        )
        os.close(audio_fd)
        staged.append(staged_audio)
        sf.write(str(staged_audio), waveform, sample_rate)
        with staged_audio.open("rb") as stream:
            os.fsync(stream.fileno())
        predicted_output = _predicted_audio_identity(staged_audio, output)
        sidecar_payload = {**expected, "output": predicted_output}

        sidecar_fd, staged_sidecar = _create_staging_file(sidecar, suffix=".tmp")
        staged.append(staged_sidecar)
        with os.fdopen(sidecar_fd, "w", encoding="utf-8") as stream:
            json.dump(
                sidecar_payload,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        verification_pending = True

        def fsync_and_verify_published(paths: Sequence[Path]) -> None:
            nonlocal verification_pending
            for parent in sorted({_absolute(path).parent for path in paths}):
                descriptor = os.open(
                    parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            if not verification_pending or any(path.exists() for path in staged):
                return
            verification_pending = False
            try:
                published_matches = (
                    stat_audio_artifact(output) == predicted_output
                    and load_sidecar(sidecar) == sidecar_payload
                )
            except (OSError, ValueError, RuntimeError):
                published_matches = False
            if not published_matches:
                raise RuntimeError("published WAV/sidecar set cannot be verified")

        verify_input_identities(inputs)
        publish_staged_files(
            staged,
            [output, sidecar],
            fsync_parent_directories=fsync_and_verify_published,
        )
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def publish_audio_output(
    output_path: Path,
    sidecar_path: Path,
    waveform: Any,
    sample_rate: int,
    expected: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
) -> None:
    """在同一 output-dir 锁和集合事务内发布 WAV 与 sidecar。"""
    lock_root = output_dir if output_dir is not None else Path(output_path).parent
    with exclusive_output_run(lock_root):
        _publish_audio_output_locked(
            output_path, sidecar_path, waveform, sample_rate, expected
        )


def write_output_sidecar_locked(
    sidecar_path: Path, output_path: Path, expected: Mapping[str, Any]
) -> None:
    """调用方已持有 output-dir 独占锁时的 sidecar writer（公共 seam）。"""
    verify_artifact_meta(expected)
    atomic_write_json(
        sidecar_path,
        {**expected, "output": stat_audio_artifact(output_path)},
    )


def write_output_sidecar(
    sidecar_path: Path, output_path: Path, expected: Mapping[str, Any]
) -> None:
    """兼容已有 WAV 的 sidecar writer，并使用同一 output-dir 锁。"""
    with exclusive_output_run(Path(sidecar_path).parent):
        write_output_sidecar_locked(sidecar_path, output_path, expected)


def bind_input_artifacts(
    base_meta: Mapping[str, Any],
    input_paths: Mapping[str, Path],
    *,
    cues: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """绑定文件输入 stat 身份与 canonical cue 声明。"""
    artifacts = dict(base_meta["artifacts"])
    artifacts["inputs"] = {
        name: stat_audio_artifact(path) for name, path in sorted(input_paths.items())
    }
    artifacts["cues"] = _cue_binding(cues)
    return _artifact_meta(artifacts)


def meta_matches(
    meta: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
    output_path: Path | None = None,
) -> bool:
    matches = bool(
        meta
        and meta.get("schema") == ARTIFACT_SCHEMA
        and expected.get("schema") == ARTIFACT_SCHEMA
        and isinstance(meta.get("fingerprint"), str)
        and meta["fingerprint"] == expected.get("fingerprint")
    )
    if not matches or output_path is None:
        return matches
    try:
        output = stat_audio_artifact(output_path)
    except (OSError, ValueError, RuntimeError):
        return False
    return meta.get("output") == output


def _manifest_item(
    row: Mapping[str, Any],
    uid_for_row: Callable[[Mapping[str, Any]], str],
    *,
    require_artifacts: bool,
) -> dict[str, Any]:
    item = {
        "uid": uid_for_row(row),
        "identity": row_identity(row),
        "status": row.get("inference_status", "failed"),
        "reused": bool(row.get("inference_reused", False)),
        "error": row.get("inference_error"),
        "tse_path": row.get("tse_path"),
        "artifacts": None,
    }
    expected = row.get("_inference_artifacts")
    if item["status"] != "ok":
        return item
    if not isinstance(expected, Mapping):
        if require_artifacts:
            raise ValueError(f"successful TSE row lacks artifact metadata: {item['uid']}")
        return item
    if require_artifacts:
        verify_artifact_meta(expected)
    output_raw = item["tse_path"]
    if not isinstance(output_raw, str) or not output_raw:
        raise ValueError(f"successful TSE row lacks output path: {item['uid']}")
    output = _absolute(output_raw)
    sidecar = output.with_name(f"{output.stem}.meta.json")
    sidecar_payload = load_sidecar(sidecar)
    if not meta_matches(sidecar_payload, expected, output):
        raise RuntimeError(f"TSE sidecar/output identity mismatch: {item['uid']}")
    item["artifacts"] = {
        "inputs": dict(expected["artifacts"]["inputs"]),
        "cues": expected["artifacts"].get("cues"),
        "output": stat_audio_artifact(output),
        "sidecar": fingerprint_file(sidecar),
    }
    return item


def _write_inference_manifest_locked(
    rows: list[dict[str, Any]],
    path: Path,
    uid_for_row: Callable[[Mapping[str, Any]], str] = make_uid,
    *,
    artifact_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """调用方持锁时发布逐项绑定 identity 的 v4 inference manifest。"""
    validate_batch(rows)
    if artifact_meta is not None:
        verify_artifact_meta(artifact_meta)
    items = [
        _manifest_item(
            row, uid_for_row, require_artifacts=artifact_meta is not None
        )
        for row in rows
    ]
    expected_uids = [item["uid"] for item in items]
    if len(set(expected_uids)) != len(expected_uids):
        raise ValueError("TSE inference manifest 存在重复 UID hash")
    ok_uids = [item["uid"] for item in items if item["status"] == "ok"]
    failed_uids = [item["uid"] for item in items if item["status"] != "ok"]
    payload = {
        "schema": MANIFEST_SCHEMA,
        "uid_scheme": UID_SCHEME,
        "artifact_fingerprint": dict(artifact_meta) if artifact_meta is not None else None,
        "expected": len(items),
        "ok": len(ok_uids),
        "failed": len(failed_uids),
        "expected_uids": expected_uids,
        "ok_uids": ok_uids,
        "failed_uids": failed_uids,
        "items": items,
    }
    atomic_write_json(path, payload)
    verify_inference_manifest(path, artifact_meta=artifact_meta)
    return payload


def write_inference_manifest(
    rows: list[dict[str, Any]],
    path: Path,
    uid_for_row: Callable[[Mapping[str, Any]], str] = make_uid,
    *,
    artifact_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """在同一 output-dir 锁内发布 v4 inference manifest。"""
    with exclusive_output_run(Path(path).parent):
        return _write_inference_manifest_locked(
            rows, path, uid_for_row, artifact_meta=artifact_meta
        )


def verify_inference_manifest(
    path: str | Path, *, artifact_meta: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """重读并复核 inference manifest 及其全部全局/逐项 artifact。"""
    manifest_path = _canonical_file(path, "TSE inference manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("TSE inference manifest cannot be decoded") from exc
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("TSE inference manifest schema is invalid")
    if payload.get("uid_scheme") != UID_SCHEME:
        raise ValueError("TSE inference manifest UID scheme is invalid")
    bound_meta = payload.get("artifact_fingerprint")
    if artifact_meta is not None and bound_meta != dict(artifact_meta):
        raise RuntimeError("TSE inference manifest global artifact mismatch")
    if bound_meta is not None:
        if not isinstance(bound_meta, Mapping):
            raise ValueError("TSE inference manifest artifact fingerprint is invalid")
        verify_artifact_meta(bound_meta)
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("TSE inference manifest items are invalid")

    expected_uids: list[str] = []
    ok_uids: list[str] = []
    failed_uids: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("TSE inference manifest item is invalid")
        identity = item.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError("TSE inference manifest item identity is invalid")
        identity_keys = set(identity)
        if identity_keys == {"dataset", "mixture", "enrollment"}:
            if any(
                not isinstance(value, str) or not value
                for value in identity.values()
            ):
                raise ValueError("TSE inference manifest item identity is invalid")
        elif identity_keys == {"dataset", "mixture", "cues"}:
            if any(
                not isinstance(identity[key], str) or not identity[key]
                for key in ("dataset", "mixture")
            ):
                raise ValueError("TSE inference manifest item identity is invalid")
            _validate_identity_cues(identity["cues"])
        else:
            raise ValueError("TSE inference manifest item identity is invalid")
        expected_uid = hashlib.sha256(
            _canonical_json_bytes(dict(identity))
        ).hexdigest()[:32]
        if item.get("uid") != expected_uid:
            raise RuntimeError("TSE inference manifest UID does not bind item identity")
        expected_uids.append(expected_uid)
        status = item.get("status")
        if status == "ok":
            ok_uids.append(expected_uid)
        elif status == "failed":
            failed_uids.append(expected_uid)
        else:
            raise ValueError("TSE inference manifest item status is invalid")
        if status != "ok":
            continue
        item_artifacts = item.get("artifacts")
        if item_artifacts is None and bound_meta is None:
            continue
        if not isinstance(item_artifacts, Mapping):
            raise ValueError("successful TSE manifest item has no artifacts")
        inputs = item_artifacts.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("successful TSE manifest item has no inputs")
        verify_input_identities(inputs)
        output_identity = item_artifacts.get("output")
        sidecar_identity = item_artifacts.get("sidecar")
        if not isinstance(output_identity, Mapping) or not isinstance(
            sidecar_identity, Mapping
        ):
            raise ValueError("successful TSE manifest item identities are invalid")
        tse_path = item.get("tse_path")
        if not isinstance(tse_path, str) or _absolute(tse_path) != _absolute(
            output_identity.get("path", "")
        ):
            raise RuntimeError("TSE manifest item path does not bind output identity")
        expected_sidecar = _absolute(tse_path).with_name(
            f"{_absolute(tse_path).stem}.meta.json"
        )
        if _absolute(sidecar_identity.get("path", "")) != expected_sidecar:
            raise RuntimeError("TSE manifest sidecar identity has unexpected path")
        if stat_audio_artifact(output_identity["path"]) != dict(output_identity):
            raise RuntimeError("TSE manifest output changed")
        verify_file_identity(sidecar_identity, label="TSE output sidecar")
        sidecar_payload = load_sidecar(Path(sidecar_identity["path"]))
        if not isinstance(sidecar_payload, Mapping):
            raise RuntimeError("TSE manifest sidecar cannot be loaded")
        if bound_meta is not None:
            verify_artifact_meta(sidecar_payload)
        elif (
            sidecar_payload.get("schema") != ARTIFACT_SCHEMA
            or not isinstance(sidecar_payload.get("artifacts"), Mapping)
            or sidecar_payload.get("fingerprint")
            != fingerprint_json(dict(sidecar_payload["artifacts"]))
        ):
            raise RuntimeError("TSE manifest sidecar metadata is invalid")
        if sidecar_payload.get("output") != dict(output_identity):
            raise RuntimeError("TSE manifest sidecar does not bind output")
        sidecar_artifacts = sidecar_payload.get("artifacts")
        if not isinstance(sidecar_artifacts, Mapping):
            raise RuntimeError("TSE manifest sidecar artifacts are invalid")
        if sidecar_artifacts.get("inputs") != dict(inputs):
            raise RuntimeError("TSE manifest sidecar does not bind inputs")
        item_cues = item_artifacts.get("cues")
        if item_cues is not None:
            verify_cue_binding(item_cues, inputs)
            if sidecar_artifacts.get("cues") != item_cues:
                raise RuntimeError("TSE manifest sidecar does not bind cues")
        if bound_meta is not None:
            bound_artifacts = bound_meta["artifacts"]
            for key in (
                "model",
                "config",
                "resolved_config",
                "speaker_checkpoint",
                "speaker_frontend",
                "dataset",
                "inference_implementation",
                "parameters",
            ):
                if sidecar_artifacts.get(key) != bound_artifacts.get(key):
                    raise RuntimeError(
                        f"TSE manifest sidecar does not bind global artifact: {key}"
                    )

    if len(set(expected_uids)) != len(expected_uids):
        raise ValueError("TSE inference manifest contains duplicate UID")
    expected_counts = {
        "expected": len(expected_uids),
        "ok": len(ok_uids),
        "failed": len(failed_uids),
        "expected_uids": expected_uids,
        "ok_uids": ok_uids,
        "failed_uids": failed_uids,
    }
    for key, expected_value in expected_counts.items():
        if payload.get(key) != expected_value:
            raise RuntimeError(f"TSE inference manifest {key} is inconsistent")
    return payload


def load_sidecar(path: Path) -> dict[str, Any] | None:
    """解析 sidecar；不存在、symlink 或损坏时以不可复用处理。"""
    try:
        if stat.S_ISLNK(os.lstat(path).st_mode):
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


__all__ = [
    "ARTIFACT_SCHEMA",
    "CUE_BINDING_SCHEMA",
    "CUE_ROLES",
    "IMPLEMENTATION_PATHS",
    "MANIFEST_SCHEMA",
    "OUTPUT_LOCK_FILENAME",
    "SPEAKER_FRONTEND_SCHEMA",
    "UID_SCHEME",
    "ImmutableFileSnapshot",
    "TSEInferenceArtifactSnapshot",
    "bind_input_artifacts",
    "capture_immutable_file",
    "ensure_output_directory",
    "exclusive_output_run",
    "implementation_fingerprint",
    "load_bound_audio",
    "load_bound_payload",
    "load_immutable_checkpoint",
    "load_sidecar",
    "make_uid",
    "meta_matches",
    "model_checkpoint_path",
    "model_config_path",
    "normalize_cue_declaration",
    "output_basename",
    "publish_audio_output",
    "row_identity",
    "speaker_frontend_identity",
    "stat_audio_artifact",
    "validate_batch",
    "validate_loaded_file_identity",
    "verify_artifact_meta",
    "verify_cue_binding",
    "verify_file_identity",
    "verify_inference_manifest",
    "verify_input_identities",
    "write_inference_manifest",
    "write_output_sidecar",
    "write_output_sidecar_locked",
]
