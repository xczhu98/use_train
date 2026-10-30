"""Shared simulation utilities for SE and AEC dynamic datasets.

This module provides common helpers shared by
``tasks.se.data.dynamic_dataset`` and ``tasks.aec.data.dynamic_dataset``:

- RNG wrappers that fall back to global ``random`` / ``numpy.random`` when no
  ``np.random.Generator`` is provided.
- SCP / JSON manifest parsers for speech / noise / RIR sources.
- Fast active-RMS estimator (frame-energy based, no external VAD).
- ``align_to_max_duration`` for cropping / tiling waveforms to a fixed length.

All functions are stateless and operate on plain numpy arrays.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import stat
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)

from core.data.noise_admission import (
    NoiseAdmissionContext,
    build_validated_noise_record,
    validate_noise_manifest_records,
)
from core.data.noise_manifest_adapter import (
    NOISE_SOURCE_PARSER_VERSION,
    NoiseManifestCapabilities,
    ParsedNoiseManifest,
    canonical_json_bytes,
    read_noise_source_manifest,
)
from core.data.simulation_manifest_materialization import (
    MaterializationDependencies,
    materialize_final_noise_manifest as _materialize_final_noise_manifest,
)
from core.data import simulation_manifest_transaction as _manifest_transaction
from core.data.time_transform import (
    CanvasOp,
    CropOp,
    POLICY_LAYER_A_RANDOM_CROP,
    TileOp,
    TimeTransformPlan,
    plan_time_transform,
)

# ---------------------------------------------------------------------------
# RNG helpers — deterministic when *rng* is given, otherwise global random
# ---------------------------------------------------------------------------

def rng_random(rng: np.random.Generator | None) -> float:
    """Uniform [0, 1); falls back to ``random.random()`` when *rng* is None."""
    return float(rng.random()) if rng is not None else random.random()


def rng_uniform(rng: np.random.Generator | None, a: float, b: float) -> float:
    """Uniform [a, b); falls back to ``random.uniform`` when *rng* is None."""
    return float(rng.uniform(a, b)) if rng is not None else random.uniform(a, b)


def rng_uniform_or_beta(
    rng: np.random.Generator | None,
    lo: float,
    hi: float,
    distribution: str = "uniform",
    beta_a: float = 2.0,
    beta_b: float = 2.0,
) -> float:
    """Draw a value in [lo, hi] using *distribution*.

    "uniform" (default): same as :func:`rng_uniform` -- every value in
    [lo, hi] equally likely.

    "beta": draw ``u ~ Beta(beta_a, beta_b)`` and map linearly to
    ``lo + u * (hi - lo)``. Symmetric params (e.g. the default 2.0/2.0)
    center the distribution at the midpoint of [lo, hi] and taper off
    toward both ends, without excluding them -- unlike a truncated normal,
    Beta's support is exactly [0, 1], so no rejection sampling is needed
    and the range bounds stay hard limits.

    Falls back to ``random.betavariate`` when *rng* is None, matching the
    other ``rng_*`` helpers' no-rng behavior.
    """
    if distribution == "uniform":
        return rng_uniform(rng, lo, hi)
    if distribution == "beta":
        u = float(rng.beta(beta_a, beta_b)) if rng is not None else random.betavariate(beta_a, beta_b)
        return lo + u * (hi - lo)
    raise ValueError(f"rng_uniform_or_beta: unknown distribution {distribution!r}; expected 'uniform' or 'beta'")


def weighted_choice(rng: np.random.Generator, probs: dict[str, float]) -> str:
    """Pick one key from *probs*, weighted by its value. Weights need not sum
    to 1 (normalized internally). Requires a real ``rng`` (unlike the other
    ``rng_*`` helpers here, there's no meaningful global-``random`` fallback
    for a dict-keyed weighted choice)."""
    keys = list(probs.keys())
    weights = np.array([float(probs[k]) for k in keys], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(f"weighted_choice: weights must sum to > 0, got {probs}")
    weights = weights / total
    idx = int(rng.choice(len(keys), p=weights))
    return keys[idx]


def rng_randint(rng: np.random.Generator | None, lo: int, hi: int) -> int:
    """Inclusive [lo, hi]; falls back to ``random.randint`` when *rng* is None."""
    if lo > hi:
        return int(lo)
    if rng is None:
        return int(random.randint(lo, hi))
    return int(rng.integers(lo, hi + 1))


def rng_choice(rng: np.random.Generator | None, seq):
    """Choose one element from *seq*; falls back to ``random.choice``."""
    if rng is not None:
        return seq[int(rng.integers(0, len(seq)))]
    return random.choice(seq)


def rng_sample(rng: np.random.Generator | None, population, k: int):
    """Sample *k* unique elements; falls back to ``random.sample``."""
    if rng is not None:
        idx = rng.choice(len(population), size=k, replace=False)
        return [population[i] for i in idx]
    return random.sample(population, k)


def rng_choices_weighted(rng: np.random.Generator | None, seq, weights, k: int = 1):
    """Weighted choice; falls back to ``random.choices``."""
    if rng is not None:
        total = sum(weights)
        p = [w / total for w in weights]
        idx = rng.choice(len(seq), size=k, p=p)
        return [seq[i] for i in idx]
    return random.choices(seq, weights=weights, k=k)


def rng_normal(rng: np.random.Generator | None, loc: float, scale: float, size) -> np.ndarray:
    """Normal distribution; falls back to ``np.random.normal``."""
    if rng is not None:
        return rng.normal(loc, scale, size)
    return np.random.normal(loc, scale, size)


# ---------------------------------------------------------------------------
# Manifest cache helpers
# ---------------------------------------------------------------------------
# Large manifests (e.g. 400 MB noise JSONL / 200 MB speech SCP) are expensive
# to parse on every training run and in every DDP rank.  We transparently cache
# the parsed result next to the source file and reuse it when the source file
# has not changed.
# ---------------------------------------------------------------------------

# parser/runtime 边界变化时必须递增，禁止复用旧语义解析出的 cache。
_MANIFEST_CACHE_VERSION = "6"
_MANIFEST_CACHE_PARSER_VERSION = str(NOISE_SOURCE_PARSER_VERSION)
_MANIFEST_CACHE_SUFFIX = ".manifest_cache.json"
_MANIFEST_CACHE_DISABLE = (
    os.environ.get("USE_TRAIN_MANIFEST_CACHE_DISABLE", "").strip().lower()
    in ("1", "true", "yes")
)


def _manifest_cache_path(source_path: str | Path) -> Path:
    """Return the default cache path next to the source manifest."""
    source = Path(source_path)
    return source.with_suffix(source.suffix + _MANIFEST_CACHE_SUFFIX)


def _manifest_cache_dir_fallback(source_path: str | Path) -> Path:
    """Return a user-writable fallback cache dir when the source dir is RO."""
    source = Path(source_path).resolve()
    cache_root = Path.home() / ".cache" / "use_train" / "manifests"
    # Hash the absolute source path so different manifests do not collide.
    name = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:32]
    return cache_root / f"{name}{_MANIFEST_CACHE_SUFFIX}"


def _manifest_cache_raw_load(cache_path: Path):
    """读取不含可执行对象的版本化 JSON cache。"""
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _manifest_cache_load_from_obj(obj: dict):
    """Reconstruct parser return types from a loaded cache dictionary."""
    if not isinstance(obj, dict):
        raise ValueError("corrupt manifest cache payload")

    def _rebuild_by_fs(raw):
        return defaultdict(dict, {int(fs): dict(entries) for fs, entries in raw.items()})

    by_fs = _rebuild_by_fs(obj["by_fs"])
    uids = {int(fs): list(entries) for fs, entries in obj["uids"].items()}
    flat = dict(obj["flat"])
    lengths_raw = obj.get("lengths")
    lengths = dict(lengths_raw) if lengths_raw is not None else None
    metadata_raw = obj.get("metadata")
    metadata = dict(metadata_raw) if metadata_raw is not None else {}
    return by_fs, uids, flat, lengths, metadata


def _manifest_source_identity(source_path: str | Path) -> tuple[int, int, int, int, int]:
    """采集 manifest 的完整文件身份，用于 cache TOCTOU 校验。"""
    source_stat = Path(source_path).stat()
    return (
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
        source_stat.st_size,
        source_stat.st_ino,
        source_stat.st_dev,
    )


def _manifest_cache_save(
    cache_path: Path,
    source_identity: tuple[int, int, int, int, int],
    return_metadata: bool,
    by_fs,
    uids,
    flat,
    lengths,
    metadata,
    *,
    validation_mode: str = "base",
) -> bool:
    """Atomically write a versioned manifest cache for a verified source identity."""
    source_mtime_ns, source_ctime_ns, source_size, source_inode, source_device = source_identity
    obj = {
        "_cache_version": _MANIFEST_CACHE_VERSION,
        "_parser_version": _MANIFEST_CACHE_PARSER_VERSION,
        "_source_mtime_ns": source_mtime_ns,
        "_source_ctime_ns": source_ctime_ns,
        "_source_size": source_size,
        "_source_inode": source_inode,
        "_source_device": source_device,
        "_return_metadata": return_metadata,
        "_validation_mode": validation_mode,
        "by_fs": {int(fs): dict(entries) for fs, entries in by_fs.items()},
        "uids": {int(fs): list(entries) for fs, entries in uids.items()},
        "flat": dict(flat),
        "lengths": dict(lengths) if lengths is not None else None,
        "metadata": dict(metadata) if metadata is not None else None,
    }
    tmp_path: Path | None = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as cache_file:
            tmp_path = Path(cache_file.name)
            json.dump(
                obj,
                cache_file,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            cache_file.flush()
            os.fsync(cache_file.fileno())
        os.replace(tmp_path, cache_path)
        return True
    except Exception:
        # Best-effort cache; do not let write failures propagate.
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        return False


def _manifest_cache_matches(
    cache_obj: object,
    source_identity: tuple[int, int, int, int, int],
    return_metadata: bool,
    validation_mode: str = "base",
) -> bool:
    return isinstance(cache_obj, dict) and (
        cache_obj.get("_cache_version") == _MANIFEST_CACHE_VERSION
        and cache_obj.get("_parser_version") == _MANIFEST_CACHE_PARSER_VERSION
        and cache_obj.get("_source_mtime_ns") == source_identity[0]
        and cache_obj.get("_source_ctime_ns") == source_identity[1]
        and cache_obj.get("_source_size") == source_identity[2]
        and cache_obj.get("_source_inode") == source_identity[3]
        and cache_obj.get("_source_device") == source_identity[4]
        and cache_obj.get("_return_metadata") == return_metadata
        and cache_obj.get("_validation_mode") == validation_mode
    )


def _manifest_cache_load_or_parse(
    source_path: str | Path,
    return_metadata: bool,
    parse_fn,
    *,
    validation_mode: str = "base",
):
    """Transparent cache wrapper for manifest parsers.

    ``parse_fn`` must return ``(by_fs, uids, flat, lengths, metadata)`` where
    ``metadata`` is an empty dict when ``return_metadata`` is False.
    """
    if _MANIFEST_CACHE_DISABLE:
        return parse_fn()

    cache_path = _manifest_cache_path(source_path)
    fallback_cache_path = _manifest_cache_dir_fallback(source_path)
    source_identity = _manifest_source_identity(source_path)

    for candidate_path in (cache_path, fallback_cache_path):
        try:
            cache_obj = _manifest_cache_raw_load(candidate_path)
            if not _manifest_cache_matches(
                cache_obj,
                source_identity,
                return_metadata,
                validation_mode,
            ):
                continue
            result = _manifest_cache_load_from_obj(cache_obj)
            if _manifest_source_identity(source_path) != source_identity:
                break
            if not return_metadata:
                return result[0], result[1], result[2], result[3]
            return result
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.debug(
                "Manifest cache validation failed for %s at %s: %s; trying next cache location.",
                source_path,
                candidate_path,
                exc,
            )

    parse_identity = _manifest_source_identity(source_path)
    result = parse_fn()
    try:
        cacheable = _manifest_source_identity(source_path) == parse_identity
    except OSError:
        cacheable = False

    if cacheable:
        saved = _manifest_cache_save(
            cache_path,
            parse_identity,
            return_metadata,
            result[0],
            result[1],
            result[2],
            result[3],
            result[4] if return_metadata else {},
            validation_mode=validation_mode,
        )
        if not saved:
            _manifest_cache_save(
                _manifest_cache_dir_fallback(source_path),
                parse_identity,
                return_metadata,
                result[0],
                result[1],
                result[2],
                result[3],
                result[4] if return_metadata else {},
                validation_mode=validation_mode,
            )
    if not return_metadata:
        return result[0], result[1], result[2], result[3]
    return result


# ---------------------------------------------------------------------------
# SCP / manifest parsers
# ---------------------------------------------------------------------------

def is_jsonl_source_manifest(path: str | Path) -> bool:
    """判断 source manifest 是否为逐行 JSON（大小写不敏感）。"""
    return Path(path).suffix.lower() == ".jsonl"


def is_strict_jsonl_manifest(path: str | Path) -> bool:
    """strict JSONL speech manifest 判定的单一事实来源（大小写不敏感）。

    该 helper 保留 TSE speech identity 契约；noise runtime 不能把 JSONL 后缀
    等同于 Noise Filter strict admission。
    """
    return is_jsonl_source_manifest(path)


def read_training_noise_manifest(path: str | Path) -> ParsedNoiseManifest:
    """读取训练噪声基础 schema/category，不重放 production admission。"""

    def parse_training_manifest():
        parsed = read_noise_source_manifest(path)
        return (
            defaultdict(dict, parsed.by_fs),
            parsed.uids,
            parsed.flat,
            parsed.lengths,
            parsed.metadata,
        )

    by_fs, uids, flat, lengths, metadata = _manifest_cache_load_or_parse(
        path,
        True,
        parse_training_manifest,
        validation_mode="training-base-v1",
    )
    has_metadata = is_jsonl_source_manifest(path)
    category_complete = bool(flat) and has_metadata and all(
        type(record.get("category_id")) is int
        and record["category_id"] >= 0
        and isinstance(
            record.get("balance_category", record.get("category_name")), str
        )
        and bool(record.get("balance_category", record.get("category_name")))
        and record.get("balance_category", record.get("category_name"))
        == record.get("balance_category", record.get("category_name")).strip()
        for record in metadata.values()
    )
    return ParsedNoiseManifest(
        by_fs=dict(by_fs),
        uids=uids,
        flat=flat,
        lengths=lengths,
        metadata=metadata,
        capabilities=NoiseManifestCapabilities(
            has_metadata=has_metadata,
            category_complete=category_complete,
            strict_fields_complete=False,
        ),
    )


def _parsed_source_result(path: str | Path, *, return_metadata: bool):
    """把统一 parser 的稳定对象投影为历史 tuple API。"""
    parsed = read_noise_source_manifest(path)
    result = (
        defaultdict(dict, parsed.by_fs),
        parsed.uids,
        parsed.flat,
        parsed.lengths,
        parsed.metadata if return_metadata else {},
    )
    return result


def _read_source_manifest_impl(path: str | Path, return_metadata: bool = True):
    """缓存解析接缝；实际解析统一委托 canonical source parser。"""
    return _parsed_source_result(path, return_metadata=return_metadata)


def read_source_scp(
    path: str | Path,
    return_metadata: bool = False,
    *,
    production_admission: bool = False,
):
    """读取严格四列 SCP 或 JSONL source，并保留 tuple 返回接口。

    基础解析统一复用 :func:`core.data.noise_manifest_adapter.read_noise_source_manifest`。
    普通训练与 metadata-only 读取不会加载 policy；只有
    调用方显式指定 ``production_admission=True`` 时才执行 fail-closed v6 准入。
    基础解析结果按 source 完整文件身份、统一 parser 版本及 metadata 模式缓存。
    """
    if production_admission:
        parsed = read_noise_source_manifest(path, production_admission=True)
        result = (
            defaultdict(dict, parsed.by_fs),
            parsed.uids,
            parsed.flat,
            parsed.lengths,
            parsed.metadata,
        )
        if return_metadata:
            return result
        return result[:4]

    return _manifest_cache_load_or_parse(
        path,
        return_metadata,
        lambda: _read_source_manifest_impl(path, return_metadata=return_metadata),
    )


def _admission_identity(output_record: dict) -> tuple:
    """从已校验的输出记录提取准入身份（含 evidence contract 与 artifact fingerprints）。"""
    return (
        output_record["admission_policy_version"],
        output_record["admission_policy_sha256"],
        output_record["verification"]["evidence_contract_version"],
        output_record["verification"]["score_contract"],
        output_record["verification"]["label_space_sha256"],
        tuple(sorted(output_record["verification"]["artifact_fingerprints"].items())),
    )


def _manifest_file_identity(path: Path) -> tuple[str, int, int, int, int, int]:
    """在同一 FD 上读取 raw digest，并确认 path 仍指向同一 inode。"""
    digest = hashlib.sha256()
    with path.open("rb") as manifest_file:
        before = os.fstat(manifest_file.fileno())
        while chunk := manifest_file.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(manifest_file.fileno())
        path_after = path.stat()
    before_identity = (
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_ino,
        before.st_dev,
    )
    after_identity = (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_ino,
        after.st_dev,
    )
    path_identity = (
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
        path_after.st_ino,
        path_after.st_dev,
    )
    if before_identity != after_identity or path_identity != after_identity:
        raise ValueError(f"{path}: manifest changed while computing input identity")
    return (digest.hexdigest(), *before_identity)


def _capture_manifest_identities(paths: Iterable[Path]) -> dict[Path, tuple[str, int, int, int, int, int]]:
    """采集输入 manifest 的 raw digest、大小、时间与 inode/device。"""
    return {path: _manifest_file_identity(path) for path in paths}


def _assert_manifest_identities(
    paths: Iterable[Path],
    expected: dict[Path, tuple[str, int, int, int, int, int]],
    phase: str,
) -> None:
    """在 materialize 关键边界复验输入 manifest，防止两遍读写混用。"""
    current = _capture_manifest_identities(paths)
    for path in paths:
        if current[path] != expected[path]:
            raise ValueError(f"{path}: line 1: manifest changed {phase}")


def _validate_final_audio_path(
    noise_root: Path,
    audio_path: str,
    location: str,
) -> None:
    """校验 final/audio 路径边界；不得打开或 hash 音频 payload。

    音频 payload 内容不参与跨阶段 identity 复验：发布产物只绑定 manifest 身份。
    """
    audio_root = noise_root / "final" / "audio"
    if audio_root.is_symlink() or not audio_root.is_dir():
        raise ValueError(f"{location}: final audio root must be a real directory: {audio_root}")
    candidate = Path(audio_path)
    if candidate.is_symlink():
        raise ValueError(f"{location}: audio path must not be a symlink: {audio_path}")
    resolved_root = audio_root.resolve(strict=True)
    try:
        resolved_audio = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{location}: invalid final audio path: {audio_path}: {exc}") from exc
    try:
        resolved_audio.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"{location}: audio path escapes final audio root ({resolved_root}): {audio_path}"
        ) from exc
    if not candidate.is_file():
        raise ValueError(f"{location}: final audio path must be a regular file: {audio_path}")


def _build_strict_final_noise_record(
    record: Mapping[str, Any],
    *,
    location: str,
    context: NoiseAdmissionContext,
) -> dict[str, Any]:
    """拒绝旧准入冗余字段，再委托共享 schema v6 builder。"""
    labels = record.get("labels") if isinstance(record, Mapping) else None
    stages = record.get("stages") if isinstance(record, Mapping) else None
    c3 = stages.get("C3") if isinstance(stages, Mapping) else None
    legacy_labels = {"audioset_scores", "audioset_top10"} & set(
        labels if isinstance(labels, Mapping) else {}
    )
    legacy_c3 = {
        "admission_calibration_version",
        "admission_calibration_sha256",
        "calibration_approved",
        "attestation",
    } & set(c3 if isinstance(c3, Mapping) else {})
    if legacy_labels or legacy_c3:
        raise ValueError(
            f"{location}: legacy admission fields are forbidden: "
            f"labels={sorted(legacy_labels)!r}, C3={sorted(legacy_c3)!r}"
        )
    return build_validated_noise_record(record, location=location, context=context)


def _update_record_digest(digest: hashlib._Hash, output_record: dict) -> None:
    digest.update(canonical_json_bytes(output_record))
    digest.update(b"\n")


def _validate_second_pass_record(
    path: Path,
    line_number: int,
    output_record: dict,
    category_vocab: dict[str, int],
    seen_record_ids: set[str],
    digest: hashlib._Hash,
) -> None:
    record_id = output_record["uid"]
    dataset_id = output_record["dataset_id"]
    category_name = output_record["balance_category"]
    if dataset_id != path.stem:
        raise ValueError(f"{path}: line {line_number}: dataset_id changed between validation and write passes")
    if record_id in seen_record_ids:
        raise ValueError(f"{path}: line {line_number}: duplicate record_id changed between validation and write passes")
    if category_name not in category_vocab:
        raise ValueError(f"{path}: line {line_number}: category changed between validation and write passes")
    seen_record_ids.add(record_id)
    _update_record_digest(digest, output_record)


def _validate_final_manifest_path(noise_root: Path, path: Path) -> Path:
    if path.suffix != ".jsonl":
        raise ValueError(f"final manifest must be an existing .jsonl file: {path}")
    try:
        path_stat = path.lstat()
    except OSError as exc:
        raise ValueError(f"final manifest must be an existing .jsonl file: {path}") from exc
    if stat.S_ISLNK(path_stat.st_mode):
        raise ValueError(f"final manifest must not be a symbolic link: {path}")
    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(f"final manifest must be a regular file: {path}")
    resolved = path.resolve()
    expected_parent = noise_root / "final" / "manifests"
    if resolved.parent != expected_parent or not resolved.stem:
        raise ValueError(f"final manifest must match {expected_parent}/<dataset_id>.jsonl: {path}")
    return resolved


def _iter_final_audiorecords(path: Path):
    with path.open(encoding="utf-8") as manifest_file:
        for line_number, raw_line in enumerate(manifest_file, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(f"{path}: line {line_number}: blank JSONL line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {line_number}: AudioRecord must be an object")
            yield line_number, record


_exclusive_output_set_lock = _manifest_transaction.exclusive_output_set_lock
_validate_distinct_output_paths = _manifest_transaction.validate_distinct_output_paths
_open_atomic_temp = _manifest_transaction.open_atomic_temp
_unlink_if_exists = _manifest_transaction.unlink_if_exists


def _publish_output_pair(
    strict_temp_path: Path,
    output: Path,
    vocab_temp_path: Path,
    vocab_output: Path,
) -> None:
    _manifest_transaction.publish_output_pair(
        strict_temp_path,
        output,
        vocab_temp_path,
        vocab_output,
    )


def materialize_final_noise_manifest(
    noise_root: str | Path,
    final_manifest_paths: Iterable[str | Path],
    output_path: str | Path,
    category_vocab_path: str | Path,
) -> dict[str, int]:
    """将 final AudioRecord manifests 双遍校验并事务发布为训练 JSONL。"""
    deps = MaterializationDependencies(
        validate_manifest_path=_validate_final_manifest_path,
        validate_distinct_outputs=_validate_distinct_output_paths,
        output_set_lock=_exclusive_output_set_lock,
        load_admission_context=NoiseAdmissionContext.load,
        capture_manifest_identities=_capture_manifest_identities,
        assert_manifest_identities=_assert_manifest_identities,
        iter_final_records=_iter_final_audiorecords,
        build_strict_record=_build_strict_final_noise_record,
        validate_audio_path=_validate_final_audio_path,
        admission_identity=_admission_identity,
        update_record_digest=_update_record_digest,
        open_atomic_temp=_open_atomic_temp,
        fsync=os.fsync,
        validate_second_pass_record=_validate_second_pass_record,
        validate_noise_records=validate_noise_manifest_records,
        canonical_json_bytes=canonical_json_bytes,
        publish_output_pair=_publish_output_pair,
        unlink_if_exists=_unlink_if_exists,
    )
    return _materialize_final_noise_manifest(
        noise_root,
        final_manifest_paths,
        output_path,
        category_vocab_path,
        deps=deps,
    )


# ---------------------------------------------------------------------------
# Audio analysis helpers
# ---------------------------------------------------------------------------

def fast_active_rms(
    w: np.ndarray,
    sr: int = 16000,
    frame_ms: int = 20,
    threshold_db: float = -50,
) -> float:
    """Fast voiced-segment RMS using frame-energy gating (no external VAD).

    Args:
        w: 1-D waveform.
        sr: Sample rate in Hz.
        frame_ms: Frame length in milliseconds.
        threshold_db: Energy threshold in dB below which frames are considered silent.

    Returns:
        RMS of active frames, or overall RMS if no frames exceed the threshold.
    """
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = max(1, len(w) // frame_len)
    # Guard the single-frame / very-short case so reshape always succeeds.
    if len(w) < frame_len:
        return float(np.sqrt(np.mean(w ** 2) + 1e-8))
    frames = w[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    threshold = 10.0 ** (threshold_db / 20.0)
    active = rms_per_frame[rms_per_frame > threshold]
    if len(active) == 0:
        return float(np.sqrt(np.mean(w ** 2) + 1e-8))
    return float(np.sqrt(np.mean(active ** 2)))


# ---------------------------------------------------------------------------
# Duration alignment
# ---------------------------------------------------------------------------

def _apply_1d_time_plan(plan: TimeTransformPlan, waveform: np.ndarray) -> np.ndarray:
    """无 RNG 地应用单轨 plan；保留 helper 的 float32 ndarray Interface。"""
    current = np.asarray(waveform)
    for operation in plan.operations:
        if isinstance(operation, CropOp):
            current = current[
                operation.start : operation.start + operation.output_length
            ].astype(np.float32)
        elif isinstance(operation, TileOp):
            period = np.zeros(
                len(current) + operation.gap_length, dtype=np.float32
            )
            period[: len(current)] = current
            needed = operation.start + operation.output_length
            repeats = (needed + len(period) - 1) // len(period)
            current = np.tile(period, repeats)[
                operation.start : needed
            ].astype(np.float32, copy=False)
        elif isinstance(operation, CanvasOp):
            result = np.zeros(operation.output_length, dtype=np.float32)
            result[operation.offset : operation.offset + len(current)] = current
            current = result
        else:  # pragma: no cover - shared helper 只构造上述三类 op
            raise TypeError(f"unsupported shared time operation: {type(operation).__name__}")
    return current.astype(np.float32, copy=False)


def _build_align_to_max_duration_plan(
    input_length: int,
    max_duration: int,
    *,
    sr: Optional[int],
    insert_silence_gap_between_tiles: bool,
    gap_min_sec: float,
    gap_max_sec: float,
    rng: np.random.Generator | None,
) -> TimeTransformPlan | None:
    """在旧控制流位置消费完全相同的 draw，并冻结 crop/tile 参数。"""
    if input_length == 0:
        return None
    if input_length >= max_duration:
        start = rng_randint(rng, 0, input_length - max_duration)
        return plan_time_transform(
            input_length=input_length,
            operations=(CropOp(start=start, output_length=max_duration),),
            policy=POLICY_LAYER_A_RANDOM_CROP,
        )

    use_gap = insert_silence_gap_between_tiles and sr is not None and sr > 0
    if use_gap:
        gap_min_samp = max(1, int(sr * gap_min_sec))
        gap_max_samp = max(gap_min_samp, int(sr * gap_max_sec))
        gap_len = rng_randint(rng, gap_min_samp, gap_max_samp)
        period = input_length + gap_len
        repeats = (max_duration + period - 1) // period
        total_len = input_length + max(0, repeats - 1) * period
        while total_len < max_duration:
            total_len += period
        start = (
            rng_randint(rng, 0, total_len - max_duration)
            if total_len > max_duration
            else 0
        )
    else:
        gap_len = 0
        start = rng_randint(rng, 0, input_length) % input_length
    return plan_time_transform(
        input_length=input_length,
        operations=(
            TileOp(
                output_length=max_duration,
                start=start,
                gap_length=gap_len,
            ),
        ),
        policy="repeat_to_max_duration",
    )


def align_to_max_duration(
    speech_1d: np.ndarray,
    noisy_1d: np.ndarray,
    max_duration: int,
    sr: Optional[int] = None,
    insert_silence_gap_between_tiles: bool = False,
    gap_min_sec: float = 0.02,
    gap_max_sec: float = 0.2,
    rng: np.random.Generator | None = None,
    extra_arrays: Optional[list[np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """按一次共享 plan 将所有轨道对齐到 ``max_duration``。"""
    input_length = len(speech_1d)
    if input_length != len(noisy_1d):
        raise ValueError(
            "speech_1d and noisy_1d must have the same length, got "
            f"{input_length} and {len(noisy_1d)}"
        )
    extras = list(extra_arrays) if extra_arrays else []
    for index, extra in enumerate(extras):
        if len(extra) != input_length:
            raise ValueError(
                f"extra_arrays[{index}] must have length {input_length}, got {len(extra)}"
            )

    if input_length == 0:
        speech = np.zeros(max_duration, dtype=np.float32)
        noisy = np.zeros(max_duration, dtype=np.float32)
        # extras 必须落在同一固定时间轴。
        aligned_extras = [np.zeros(max_duration, dtype=np.float32) for _ in extras]
        return (speech, noisy, aligned_extras) if extras else (speech, noisy)

    plan = _build_align_to_max_duration_plan(
        input_length,
        max_duration,
        sr=sr,
        insert_silence_gap_between_tiles=insert_silence_gap_between_tiles,
        gap_min_sec=gap_min_sec,
        gap_max_sec=gap_max_sec,
        rng=rng,
    )
    if plan is None:
        raise ValueError("failed to build alignment plan")
    speech = _apply_1d_time_plan(plan, speech_1d)
    noisy = _apply_1d_time_plan(plan, noisy_1d)
    if not extras:
        return speech, noisy
    return speech, noisy, [_apply_1d_time_plan(plan, item) for item in extras]


def apply_silence_position(
    speech_1d: np.ndarray,
    sr: int,
    max_dur_sr: int,
    silence_pad_prob: float,
    silence_pad_total_range: list,
    silence_pad_position: str,
    silence_pad_mode: str = "uniform",
    silence_pad_short_range: list | None = None,
    silence_pad_long_range: list | None = None,
    silence_pad_short_ratio: float = 0.5,
    silence_pad_beta_a: float = 0.4,
    silence_pad_beta_b: float = 0.4,
    rng: np.random.Generator | None = None,
    fixed_canvas: bool = False,
    preselected_silence_pad: bool = False,
) -> np.ndarray:
    """Place speech at a random position, padding with zeros.

    默认使用 adaptive padding，不切分语音。``fixed_canvas=True`` 时输出固定为
    ``max_dur_sr``：短语音仅写入一次，长语音仅随机裁剪一次。此时配置的静音
    range 仍兼容抽样，但最终静音总量固定为 ``max_dur_sr - speech_len``，position
    policy 只负责分配前后静音。``preselected_silence_pad=True`` 表示调用方已完成
    Bernoulli 判定，本函数不再重复抽样。
    """
    if silence_pad_prob <= 0 or (
        not preselected_silence_pad and rng_random(rng) >= silence_pad_prob
    ):
        return speech_1d
    if max_dur_sr <= 0:
        return speech_1d

    if silence_pad_mode == "bimodal":
        if rng_random(rng) < silence_pad_short_ratio:
            effective_range = silence_pad_short_range or [0.2, 0.8]
        else:
            effective_range = silence_pad_long_range or [2.0, 5.0]
    else:
        effective_range = silence_pad_total_range

    total_pad_samples = int(rng_uniform(rng, effective_range[0], effective_range[1]) * sr)
    speech_len = len(speech_1d)
    if fixed_canvas and speech_len > max_dur_sr:
        start = rng_randint(rng, 0, speech_len - max_dur_sr)
        crop_plan = plan_time_transform(
            input_length=speech_len,
            operations=(CropOp(start=start, output_length=max_dur_sr),),
            policy=POLICY_LAYER_A_RANDOM_CROP,
        )
        speech_1d = _apply_1d_time_plan(crop_plan, speech_1d)
        speech_len = max_dur_sr

    max_pad_samples = max(0, max_dur_sr - speech_len)
    total_pad_samples = max_pad_samples if fixed_canvas else min(total_pad_samples, max_pad_samples)
    if total_pad_samples == 0:
        return speech_1d.astype(np.float32, copy=False)

    target_len = max_dur_sr if fixed_canvas else min(speech_len + total_pad_samples, max_dur_sr)
    available = target_len - total_pad_samples

    # adaptive 模式至少保留 0.5 秒语音；固定画布不改变 speech 本体。
    if not fixed_canvas:
        min_speech = int(0.5 * sr)
        if available < min_speech:
            available = min_speech
            total_pad_samples = target_len - available
            if total_pad_samples < 0:
                return speech_1d

    if silence_pad_position == "front":
        prefix_ratio = rng_uniform(rng, 0.7, 1.0)
    elif silence_pad_position == "back":
        prefix_ratio = rng_uniform(rng, 0.0, 0.3)
    elif silence_pad_position == "center":
        prefix_ratio = rng_uniform(rng, 0.3, 0.7)
    elif silence_pad_position == "turn_boundary":
        # Simulate natural turn-taking: silence is either at the very beginning
        # (listener starts speaking) or at the very end (speaker finishes).
        prefix_ratio = rng_uniform(rng, 0.9, 1.0) if rng_random(rng) < 0.5 else rng_uniform(rng, 0.0, 0.1)
    elif silence_pad_position == "scene_aware":
        # Beta(α,β) U-shaped distribution: favours boundaries (~65% near 0 or 1)
        # while allowing natural mid-speech pauses (~15% near 0.5).
        # α=β=0.4 → peaks at extremes; α=β=1.0 → degrades to Uniform = "random".
        if rng is not None:
            prefix_ratio = float(rng.beta(silence_pad_beta_a, silence_pad_beta_b))
        else:
            prefix_ratio = float(np.random.beta(silence_pad_beta_a, silence_pad_beta_b))
    else:
        prefix_ratio = rng_uniform(rng, 0.0, 1.0)

    # pre_len <= total_pad_samples <= target_len - speech_len is guaranteed。
    pre_len = int(total_pad_samples * prefix_ratio)
    canvas_plan = plan_time_transform(
        input_length=speech_len,
        operations=(CanvasOp(output_length=target_len, offset=pre_len),),
        policy=("fixed_silence_canvas" if fixed_canvas else "adaptive_silence_pad"),
    )
    return _apply_1d_time_plan(canvas_plan, speech_1d)


def align_speech_with_silence_pad(
    speech_1d: np.ndarray,
    sr: int,
    max_dur_sr: int,
    silence_pad_prob: float,
    silence_pad_total_range: list,
    silence_pad_position: str,
    silence_pad_mode: str = "uniform",
    silence_pad_short_range: list | None = None,
    silence_pad_long_range: list | None = None,
    silence_pad_short_ratio: float = 0.5,
    silence_pad_beta_a: float = 0.4,
    silence_pad_beta_b: float = 0.4,
    repeat_to_max_duration: bool = True,
    insert_silence_gap_between_tiles: bool = False,
    gap_min_sec: float = 0.02,
    gap_max_sec: float = 0.2,
    rng: np.random.Generator | None = None,
    fixed_silence_canvas: bool = False,
    return_silence_pad_applied: bool = False,
    preselected_silence_pad: bool | None = None,
) -> np.ndarray | tuple[np.ndarray, bool]:
    """执行 Layer A-3 对齐，并可报告本次是否命中 silence-pad。

    ``fixed_silence_canvas=False`` 默认保留 helper 级历史 adaptive padding 兼容路径。
    生产 SE/AEC/TSE Dynamic Dataset 统一传入 ``True``：命中 silence-pad 时直接构造
    固定画布，短语音只放一次，长语音只连续 crop 一次；未命中时仍走原 crop/tile
    路径。调用方可传入 ``preselected_silence_pad`` 复用提前判定，避免重复
    Bernoulli 抽样。
    """
    silence_pad_applied = (
        bool(preselected_silence_pad)
        if preselected_silence_pad is not None
        else silence_pad_prob > 0.0 and rng_random(rng) < silence_pad_prob
    )
    if silence_pad_applied:
        out = apply_silence_position(
            speech_1d,
            sr,
            max_dur_sr,
            silence_pad_prob=1.0,
            silence_pad_total_range=silence_pad_total_range,
            silence_pad_position=silence_pad_position,
            silence_pad_mode=silence_pad_mode,
            silence_pad_short_range=silence_pad_short_range,
            silence_pad_long_range=silence_pad_long_range,
            silence_pad_short_ratio=silence_pad_short_ratio,
            silence_pad_beta_a=silence_pad_beta_a,
            silence_pad_beta_b=silence_pad_beta_b,
            rng=rng,
            fixed_canvas=fixed_silence_canvas,
            preselected_silence_pad=True,
        )
    elif max_dur_sr <= 0:
        out = speech_1d
    elif len(speech_1d) == max_dur_sr:
        out = speech_1d.astype(np.float32, copy=False)
    elif len(speech_1d) > max_dur_sr:
        start = rng_randint(rng, 0, len(speech_1d) - max_dur_sr)
        out = speech_1d[start : start + max_dur_sr].astype(np.float32)
    elif repeat_to_max_duration:
        out, _ = align_to_max_duration(
            speech_1d,
            speech_1d.copy(),
            max_dur_sr,
            sr=sr,
            insert_silence_gap_between_tiles=insert_silence_gap_between_tiles,
            gap_min_sec=gap_min_sec,
            gap_max_sec=gap_max_sec,
            rng=rng,
        )
    else:
        pad_len = max_dur_sr - len(speech_1d)
        out = np.concatenate((speech_1d, np.zeros(pad_len, dtype=np.float32)))

    if return_silence_pad_applied:
        return out, silence_pad_applied
    return out


__all__ = [
    "rng_random",
    "rng_uniform",
    "rng_uniform_or_beta",
    "rng_randint",
    "rng_choice",
    "rng_sample",
    "rng_choices_weighted",
    "rng_normal",
    "weighted_choice",
    "read_source_scp",
    "materialize_final_noise_manifest",
    "fast_active_rms",
    "align_to_max_duration",
    "apply_silence_position",
    "align_speech_with_silence_pad",
]
