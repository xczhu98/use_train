"""Strict two-pass noise manifest materialization orchestration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class MaterializationDependencies:
    """Facade-provided seams needed by materialization and its compatibility tests."""

    validate_manifest_path: Callable
    validate_distinct_outputs: Callable
    output_set_lock: Callable
    load_admission_context: Callable
    capture_manifest_identities: Callable
    assert_manifest_identities: Callable
    iter_final_records: Callable
    build_strict_record: Callable
    validate_audio_path: Callable
    admission_identity: Callable
    update_record_digest: Callable
    open_atomic_temp: Callable
    fsync: Callable[[int], None]
    validate_second_pass_record: Callable
    validate_noise_records: Callable
    canonical_json_bytes: Callable[[Any], bytes]
    publish_output_pair: Callable
    unlink_if_exists: Callable


def materialize_final_noise_manifest(
    noise_root: str | Path,
    final_manifest_paths: Iterable[str | Path],
    output_path: str | Path,
    category_vocab_path: str | Path,
    *,
    deps: MaterializationDependencies,
) -> dict[str, int]:
    """执行严格双遍校验，并以一组事务发布训练 manifest 与 category vocab。"""
    root = Path(noise_root).resolve()
    paths = [Path(path) for path in final_manifest_paths]
    if not paths:
        raise ValueError("at least one final manifest path is required")
    validated_paths = [deps.validate_manifest_path(root, path) for path in paths]
    output = Path(output_path)
    vocab_output = Path(category_vocab_path)
    deps.validate_distinct_outputs(output, vocab_output)

    with deps.output_set_lock((output, vocab_output)):
        admission_context = deps.load_admission_context()
        expected_input_identities = deps.capture_manifest_identities(validated_paths)
        category_names: set[str] = set()
        seen_record_ids: set[str] = set()
        first_pass_summaries: dict[Path, tuple[int, bytes, tuple[str, ...]]] = {}
        expected_identity: tuple | None = None
        for path in validated_paths:
            digest = hashlib.sha256()
            manifest_ids: set[str] = set()
            count = 0
            for line_number, record in deps.iter_final_records(path):
                location = f"{path}: line {line_number}"
                output_record = deps.build_strict_record(
                    record,
                    location=location,
                    context=admission_context,
                )
                deps.validate_audio_path(
                    root,
                    output_record["path"],
                    location,
                )
                record_id = output_record["uid"]
                dataset_id = output_record["dataset_id"]
                if dataset_id != path.stem:
                    raise ValueError(
                        f"{location}: dataset_id must match the manifest filename"
                    )
                if record_id in seen_record_ids:
                    raise ValueError(
                        f"{location}: duplicate record_id across final manifests: {record_id!r}"
                    )
                seen_record_ids.add(record_id)
                manifest_ids.add(record_id)
                category_names.add(output_record["balance_category"])
                identity = deps.admission_identity(output_record)
                if expected_identity is None:
                    expected_identity = identity
                elif identity != expected_identity:
                    raise ValueError(
                        f"{location}: noise admission identity differs from other final manifests"
                    )
                deps.update_record_digest(digest, output_record)
                count += 1
            if count == 0:
                raise ValueError(f"{path}: final manifest contains no records")
            first_pass_summaries[path] = (
                count,
                digest.digest(),
                tuple(sorted(manifest_ids)),
            )
        deps.assert_manifest_identities(
            validated_paths,
            expected_input_identities,
            "after validation pass",
        )
        category_vocab = {
            name: index for index, name in enumerate(sorted(category_names))
        }

        strict_temp_path: Path | None = None
        vocab_temp_path: Path | None = None
        try:
            deps.assert_manifest_identities(
                validated_paths,
                expected_input_identities,
                "before write pass",
            )
            strict_file, strict_temp_path = deps.open_atomic_temp(output)
            with strict_file:
                second_pass_ids: set[str] = set()
                for path in validated_paths:
                    digest = hashlib.sha256()
                    second_manifest_ids: set[str] = set()
                    count = 0
                    for line_number, record in deps.iter_final_records(path):
                        location = f"{path}: line {line_number}"
                        output_record = deps.build_strict_record(
                            record,
                            location=location,
                            context=admission_context,
                        )
                        deps.validate_audio_path(
                            root,
                            output_record["path"],
                            location,
                        )
                        identity = deps.admission_identity(output_record)
                        if identity != expected_identity:
                            raise ValueError(
                                f"{location}: admission identity changed between validation and write passes"
                            )
                        deps.validate_second_pass_record(
                            path,
                            line_number,
                            output_record,
                            category_vocab,
                            second_pass_ids,
                            digest,
                        )
                        second_manifest_ids.add(output_record["uid"])
                        count += 1
                        strict_record = {
                            **output_record,
                            "category_id": category_vocab[
                                output_record["balance_category"]
                            ],
                        }
                        deps.validate_noise_records(
                            {strict_record["uid"]: strict_record},
                            source_path=location,
                            context=admission_context,
                        )
                        strict_file.write(
                            deps.canonical_json_bytes(strict_record).decode("utf-8")
                            + "\n"
                        )
                    summary = first_pass_summaries[path]
                    current = (
                        count,
                        digest.digest(),
                        tuple(sorted(second_manifest_ids)),
                    )
                    if current != summary:
                        raise ValueError(
                            f"{path}: line 1: manifest changed between validation and write passes"
                        )
                strict_file.flush()
                deps.fsync(strict_file.fileno())

            deps.assert_manifest_identities(
                validated_paths,
                expected_input_identities,
                "after write pass",
            )
            vocab_file, vocab_temp_path = deps.open_atomic_temp(vocab_output)
            with vocab_file:
                vocab_file.write(
                    deps.canonical_json_bytes(category_vocab).decode("utf-8") + "\n"
                )
                vocab_file.flush()
                deps.fsync(vocab_file.fileno())

            deps.assert_manifest_identities(
                validated_paths,
                expected_input_identities,
                "before publish",
            )
            deps.publish_output_pair(
                strict_temp_path,
                output,
                vocab_temp_path,
                vocab_output,
            )
        except Exception:
            if strict_temp_path is not None:
                deps.unlink_if_exists(strict_temp_path)
            if vocab_temp_path is not None:
                deps.unlink_if_exists(vocab_temp_path)
            raise
    return category_vocab


__all__ = ["MaterializationDependencies", "materialize_final_noise_manifest"]
