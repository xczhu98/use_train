"""Step3 canonical scorer 测试替身。"""

from __future__ import annotations

import heapq
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from audio_filter_pipeline.processors.mos import normalize_selectable_mos_models
from audio_filter_pipeline.utils.strict_jsonl import (
    add_cleanup_note,
    decode_json_line,
    iter_bounded_jsonl,
    record_cleanup_error,
)


class CanonicalStep3Scorer:
    """把逐行测试 callback 暴露为 canonical ``score_task_groups`` 契约。"""

    def __init__(self, scorer: Any) -> None:
        self._scorer = scorer

    @staticmethod
    def provenance_payload() -> Mapping[str, Any]:
        return {
            "scorer_execution": "test_canonical",
            "provider_class": "test_canonical",
            "device_class": "cpu",
            "cpu_fallback": False,
        }

    @staticmethod
    def _iter_tasks(path: str) -> Iterator[dict[str, Any]]:
        payloads = iter(iter_bounded_jsonl(
            path,
            "Step3 test tuple task spool",
            max_line_bytes=8 * 1024 * 1024,
            max_file_bytes=512 * 1024 * 1024,
        ))
        try:
            for payload in payloads:
                task = decode_json_line(
                    payload,
                    "Step3 test tuple task row",
                    max_line_bytes=8 * 1024 * 1024,
                )
                if not isinstance(task, dict):
                    raise ValueError("Step3 test tuple task must be an object")
                yield task
        finally:
            primary = sys.exc_info()[1]
            close = getattr(payloads, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup_error:
                    if isinstance(primary, GeneratorExit) or primary is None:
                        raise
                    add_cleanup_note(
                        primary, "Step3 test tuple payload cleanup", cleanup_error,
                    )

    def score_task_groups(
        self, groups: Sequence[Mapping[str, Any]],
    ) -> Iterator[Mapping[str, Any]]:
        descriptors: list[tuple[tuple[str, ...], list[Mapping[str, Any]]]] = []
        seen_models: set[tuple[str, ...]] = set()
        seen_paths: set[str] = set()
        for group in groups:
            models = normalize_selectable_mos_models(group["models"])
            canonical_path = os.path.realpath(group["path"])
            if canonical_path in seen_paths:
                raise ValueError("Step3 test task group path is duplicate")
            seen_paths.add(canonical_path)
            if descriptors and descriptors[-1][0] == models:
                descriptors[-1][1].append(group)
            else:
                if models in seen_models:
                    raise ValueError("Step3 test model tuple parts must be contiguous")
                seen_models.add(models)
                descriptors.append((models, [group]))

        def score_tuple(
            models: tuple[str, ...], parts: Sequence[Mapping[str, Any]],
        ) -> Iterator[Mapping[str, Any]]:
            for group in parts:
                tasks = iter(self._iter_tasks(group["path"]))
                count = 0
                try:
                    for task in tasks:
                        count += 1
                        if count > group["row_count"]:
                            raise ValueError("Step3 test task group row count mismatch")
                        yield {
                            "ordinal": task["ordinal"],
                            "row": task["row"],
                            "models": task["models"],
                            "scores": self._scorer(task["row"], models),
                        }
                    if count != group["row_count"]:
                        raise ValueError("Step3 test task group row count mismatch")
                finally:
                    primary = sys.exc_info()[1]
                    close = getattr(tasks, "close", None)
                    if callable(close):
                        try:
                            close()
                        except BaseException as cleanup_error:
                            if isinstance(primary, GeneratorExit) or primary is None:
                                raise
                            primary.add_note(
                                f"cleanup failed for Step3 test tuple part: {cleanup_error!r}"
                            )

        result_groups = [
            score_tuple(models, parts) for models, parts in descriptors
        ]
        merged = iter(heapq.merge(
            *result_groups, key=lambda item: item["ordinal"],
        ))
        try:
            yield from merged
            return {
                "provider_class": "test_canonical",
                "device_class": "cpu",
            }
        finally:
            primary = sys.exc_info()[1]
            cleanup_error = None
            for iterator in (merged, *result_groups):
                close = getattr(iterator, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as error:
                        cleanup_error = record_cleanup_error(
                            cleanup_error, error, "Step3 test tuple iterator",
                        )
            if isinstance(primary, GeneratorExit) and cleanup_error is not None:
                raise cleanup_error
            if primary is not None:
                if cleanup_error is not None:
                    add_cleanup_note(primary, "Step3 test scorer cleanup", cleanup_error)
            elif cleanup_error is not None:
                raise cleanup_error
