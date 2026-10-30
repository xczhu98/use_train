"""Generation 3 artifact-backed stage execution shell."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from tqdm import tqdm

from ...config import PipelineConfig
from ...utils.artifacts import (
    ArtifactIndex,
    ArtifactReader,
    ArtifactWriter,
    PipelineStep,
    is_production_bound_artifact_index,
    open_artifact_for_read,
    open_artifact_reader,
)
from ...utils.path_utils import v3_artifact_dir


logger = logging.getLogger(__name__)



EMPTY_LINEAGE_DIGEST = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True)
class ArtifactExpectation:
    """Bound identity required to consume one upstream artifact."""

    step: PipelineStep
    config_digest: str
    lineage_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.step, PipelineStep):
            raise TypeError("step must be a PipelineStep")
        for field in ("config_digest", "lineage_digest"):
            digest = getattr(self, field)
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"{field} must be lowercase SHA-256 hex"
                )


_RUNNER_EXPECTATIONS: dict[
    int,
    tuple[
        weakref.ReferenceType[ArtifactExpectation],
        tuple[str, str, str],
    ],
] = {}
_RUNNER_EXPECTATIONS_LOCK = threading.Lock()


def _mint_runner_expectation(
    index: ArtifactIndex,
    expected_step: PipelineStep,
) -> ArtifactExpectation:
    """从 production-bound index 铸造进程内 runner capability。"""
    if not isinstance(index, ArtifactIndex):
        raise TypeError("stage.run must return an ArtifactIndex")
    if index.step != expected_step.value:
        raise ValueError("stage.run returned artifact for the wrong step")
    if not is_production_bound_artifact_index(index):
        raise RuntimeError("runner requires a production-bound artifact index")
    expectation = ArtifactExpectation(
        expected_step, index.config_digest, index.lineage_digest,
    )
    identity = id(expectation)

    def discard(reference) -> None:
        with _RUNNER_EXPECTATIONS_LOCK:
            registered = _RUNNER_EXPECTATIONS.get(identity)
            if registered is not None and registered[0] is reference:
                _RUNNER_EXPECTATIONS.pop(identity, None)

    reference = weakref.ref(expectation, discard)
    binding = (index.output_root, index.dataset_id, index.artifact_digest)
    with _RUNNER_EXPECTATIONS_LOCK:
        _RUNNER_EXPECTATIONS[identity] = (reference, binding)
    return expectation


def _runner_artifact_binding(
    expectation: ArtifactExpectation,
) -> Optional[tuple[str, str, str]]:
    with _RUNNER_EXPECTATIONS_LOCK:
        registered = _RUNNER_EXPECTATIONS.get(id(expectation))
    if registered is None or registered[0]() is not expectation:
        return None
    return registered[1]


@dataclass(frozen=True)
class StageRunContext:
    """Immutable per-run inputs; stage instances keep no run state."""

    output_root: str
    dataset_id: str
    config: PipelineConfig
    upstream: Optional[ArtifactExpectation] = None
    persistent_write_capability: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, PipelineConfig):
            raise TypeError("config must be a PipelineConfig")
        if type(self.output_root) is not str or not self.output_root.strip():
            raise ValueError("output_root must be a non-empty string")
        config_root = self.config.output_root
        if type(config_root) is not str or not config_root.strip():
            raise ValueError("config output_root must be a non-empty string")
        output_root = os.path.abspath(self.output_root)
        if output_root != os.path.abspath(config_root):
            raise ValueError("context output_root must exactly match config output_root")
        object.__setattr__(self, "output_root", output_root)
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.config, PipelineConfig):
            raise TypeError("config must be a PipelineConfig")
        if self.config.pipeline_generation != 3:
            raise ValueError("stage execution requires pipeline generation 3")
        if not os.path.isabs(self.output_root):
            raise ValueError("output_root identity must be absolute")
        if type(self.dataset_id) is not str or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string")
        if self.dataset_id not in self.config.dataset_specs():
            raise ValueError(f"unknown dataset: {self.dataset_id}")


def _canonical_value(
    value: Any,
    path: str = "payload",
    active_containers: Optional[set[int]] = None,
) -> Any:
    if active_containers is None:
        active_containers = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"canonical JSON cycle detected at {path}")
        active_containers.add(identity)
        try:
            canonical = {}
            for key, nested in value.items():
                if type(key) is not str:
                    raise TypeError(f"{path} object keys must be strings")
                if key in canonical:
                    raise ValueError(f"{path} has duplicate object key: {key!r}")
                canonical[key] = _canonical_value(
                    nested, f"{path}.{key}", active_containers
                )
            return canonical
        finally:
            active_containers.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"canonical JSON cycle detected at {path}")
        active_containers.add(identity)
        try:
            return [
                _canonical_value(
                    nested, f"{path}[{index}]", active_containers
                )
                for index, nested in enumerate(value)
            ]
        finally:
            active_containers.remove(identity)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} numbers must be finite")
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(
        f"{path} must contain only canonical JSON values, got "
        f"{type(value).__name__}"
    )


def _add_cleanup_note(primary: BaseException, cleanup: BaseException) -> None:
    note = f"artifact cleanup failed: {cleanup!r}"
    add_note = getattr(primary, "add_note", None)
    if add_note is not None:
        add_note(note)
        return
    notes = list(getattr(primary, "__notes__", ()))
    notes.append(note)
    primary.__notes__ = notes


class ArtifactStage(ABC):
    """Reusable shell that binds stage output to config and upstream lineage."""

    step: PipelineStep
    upstream_step: Optional[PipelineStep]
    # 进度条计数单位；默认按 row 计，段切分等 stage 可覆盖为更直观的单位
    # （如 "source"），避免 1→N 的输出行数与上游源数混淆。
    progress_unit: str = "rows"
    # 进度条显示策略：
    #   "output_rows"（默认）—— 外壳按输出行迭代计数，total 取自 progress_total；
    #   "self_managed" —— 外壳不绘制进度条，由 produce_rows 内部自绘，用于需要
    #   全量读取/分组计算、无法按输出行逐行反映真实进度的 stage（如 Step5）。
    progress_mode: str = "output_rows"

    def progress_total(
        self,
        context: StageRunContext,
        upstream_index: Optional[ArtifactIndex],
    ) -> Optional[int]:
        """返回本 step 可预知的处理总数供进度条使用。

        默认按上游 artifact 的已发布 row_count 计（绝大多数 stage 每个上游行
        处理一次，进度条按「处理源」推进）；Step0 无上游时返回 None → 外壳使用
        不确定模式进度条（只显示计数）。
        """
        if upstream_index is None:
            return None
        return upstream_index.row_count

    @abstractmethod
    def config_payload(self, config: PipelineConfig) -> Mapping[str, Any]:
        """Return only configuration owned by this stage."""

    @abstractmethod
    def produce_rows(
        self,
        context: StageRunContext,
        upstream_index: Optional[ArtifactIndex],
        upstream_rows: Iterable[Mapping[str, Any]],
    ) -> Iterable[Mapping[str, Any]]:
        """Lazily produce output rows from the verified upstream stream."""

    def before_publish(
        self, context: StageRunContext, writer: ArtifactWriter,
    ) -> None:
        """在 rows 完整写入、artifact rename 前执行可选的阶段特定持久化。"""

    def config_digest(self, config: PipelineConfig) -> str:
        payload = self.config_payload(config)
        if not isinstance(payload, Mapping):
            raise TypeError("stage config_payload must return a mapping")
        canonical = _canonical_value(payload)
        encoded = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _validate_contract(self, context: StageRunContext) -> None:
        context.validate()
        if not isinstance(self.step, PipelineStep):
            raise TypeError("stage step must be a PipelineStep")
        if self.upstream_step is not None and not isinstance(
            self.upstream_step, PipelineStep
        ):
            raise TypeError("stage upstream_step must be a PipelineStep or None")
        expectation = context.upstream
        if self.upstream_step is None:
            if expectation is not None:
                raise ValueError("source stage must not receive an upstream expectation")
            return
        if expectation is None:
            raise ValueError("downstream stage requires an upstream expectation")
        if not isinstance(expectation, ArtifactExpectation):
            raise TypeError("upstream must be an ArtifactExpectation")
        if expectation.step is not self.upstream_step:
            raise ValueError(
                f"upstream step mismatch: expected {self.upstream_step.value}, "
                f"got {expectation.step.value}"
            )
        binding = _runner_artifact_binding(expectation)
        if binding is None:
            raise RuntimeError(
                "downstream stage execution requires runner-minted upstream capability"
            )
        bound_root, bound_dataset_id, _artifact_digest = binding
        if (
            bound_root != context.output_root
            or bound_dataset_id != context.dataset_id
        ):
            raise ValueError("runner-minted upstream context identity mismatch")

    def _open_upstream_index(
        self, context: StageRunContext,
    ) -> tuple[Optional[ArtifactIndex], str]:
        """在业务 reader/writer 前完整验证 upstream，但不创建业务 rows spool。"""
        expectation = context.upstream
        if expectation is None:
            return None, EMPTY_LINEAGE_DIGEST
        index = open_artifact_for_read(
            context.output_root,
            expectation.step,
            context.dataset_id,
            expected_config_digest=expectation.config_digest,
            expected_lineage_digest=expectation.lineage_digest,
        )
        binding = _runner_artifact_binding(expectation)
        expected_artifact_digest = None if binding is None else binding[2]
        if index.artifact_digest != expected_artifact_digest:
            raise ValueError("runner-minted upstream artifact identity changed")
        return index, index.artifact_digest

    def _open_upstream_reader(
        self, context: StageRunContext,
    ) -> tuple[ArtifactReader, str]:
        """为 cache miss 打开并完整验证可消费的 upstream rows reader。"""
        expectation = context.upstream
        if expectation is None:
            raise RuntimeError("source stage has no upstream reader")
        reader = open_artifact_reader(
            context.output_root,
            expectation.step,
            context.dataset_id,
            expected_config_digest=expectation.config_digest,
            expected_lineage_digest=expectation.lineage_digest,
        )
        binding = _runner_artifact_binding(expectation)
        expected_artifact_digest = None if binding is None else binding[2]
        if reader.index.artifact_digest != expected_artifact_digest:
            primary = ValueError("runner-minted upstream artifact identity changed")
            try:
                reader.close()
            except BaseException as cleanup:
                _add_cleanup_note(primary, cleanup)
            raise primary
        return reader, reader.index.artifact_digest

    def _open_output(
        self,
        context: StageRunContext,
        config_digest: str,
        lineage_digest: str,
    ) -> ArtifactIndex:
        return open_artifact_for_read(
            context.output_root,
            self.step,
            context.dataset_id,
            expected_config_digest=config_digest,
            expected_lineage_digest=lineage_digest,
        )

    def run(self, context: StageRunContext) -> ArtifactIndex:
        """执行 source 或已由 runner mint capability 的 downstream stage。"""
        self._validate_contract(context)
        config_digest = self.config_digest(context.config)
        output_dir = v3_artifact_dir(
            context.output_root, self.step.value, context.dataset_id
        )
        if os.path.lexists(output_dir):
            _upstream_index, lineage_digest = self._open_upstream_index(context)
            logger.info(
                "Step %s dataset=%s hit output cache, skip (resume)",
                self.step.value, context.dataset_id,
            )
            return self._open_output(context, config_digest, lineage_digest)

        upstream_reader = None
        try:
            if context.upstream is None:
                upstream_index = None
                lineage_digest = EMPTY_LINEAGE_DIGEST
            else:
                upstream_reader, lineage_digest = self._open_upstream_reader(context)
                upstream_index = upstream_reader.index
            writer = ArtifactWriter(
                context.output_root,
                self.step,
                context.dataset_id,
                config_digest,
                lineage_digest,
            )
        except FileExistsError as collision:
            if upstream_reader is not None:
                try:
                    upstream_reader.close()
                except BaseException as cleanup:
                    _add_cleanup_note(collision, cleanup)
                    raise collision
            if getattr(collision, "__notes__", ()):
                raise
            logger.info(
                "Step %s dataset=%s writer collision = cache hit, skip",
                self.step.value, context.dataset_id,
            )
            return self._open_output(context, config_digest, lineage_digest)
        except BaseException as primary:
            if upstream_reader is not None:
                try:
                    upstream_reader.close()
                except BaseException as cleanup:
                    _add_cleanup_note(primary, cleanup)
            raise

        upstream_rows = None
        rows = None
        upstream_row_count = None
        if upstream_index is not None:
            upstream_row_count = upstream_index.row_count
        logger.info(
            "Step %s dataset=%s start input_rows=%s",
            self.step.value, context.dataset_id,
            upstream_row_count if upstream_row_count is not None else "n/a",
        )
        started_at = time.perf_counter()
        try:
            upstream_rows = () if upstream_reader is None else upstream_reader
            rows = self.produce_rows(context, upstream_index, upstream_rows)
            if not isinstance(rows, Iterable):
                raise TypeError("produce_rows must return an iterable")
            # self_managed：stage 内部自绘进度（如 Step5 需全量读取+分组计算，
            # 无法按输出行反映真实过程），外壳只消费输出行、不叠画进度条。
            if self.progress_mode == "self_managed":
                row_count = 0
                for row in rows:
                    writer.append(row)
                    row_count += 1
            else:
                total = self.progress_total(context, upstream_index)
                progress = tqdm(
                    rows,
                    total=total,
                    unit=self.progress_unit,
                    desc=f"{self.step.value}:{context.dataset_id}",
                    file=sys.stdout,
                    disable=None,
                )
                row_count = 0
                for row in progress:
                    writer.append(row)
                    row_count += 1
                progress.close()
            close_rows = getattr(rows, "close", None)
            if close_rows is not None:
                close_rows()
            rows = None
            close_upstream = getattr(upstream_rows, "close", None)
            if close_upstream is not None:
                close_upstream()
            upstream_rows = ()
            self.before_publish(context, writer)
            try:
                published = writer.finalize()
            except FileExistsError:
                if writer.state != "aborted":
                    raise
                elapsed = time.perf_counter() - started_at
                logger.info(
                    "Step %s dataset=%s aborted (collision) rows=%d elapsed=%.1fs",
                    self.step.value, context.dataset_id, row_count, elapsed,
                )
                return self._open_output(context, config_digest, lineage_digest)
        except BaseException as primary:
            for iterable in (rows, upstream_rows):
                close = getattr(iterable, "close", None)
                if close is not None:
                    try:
                        close()
                    except BaseException as cleanup:
                        _add_cleanup_note(primary, cleanup)
            if writer.state in ("active", "cleanup_failed"):
                try:
                    writer.abort()
                except BaseException as cleanup:
                    _add_cleanup_note(primary, cleanup)
            raise

        elapsed = time.perf_counter() - started_at
        logger.info(
            "Step %s dataset=%s done rows=%d elapsed=%.1fs (%.1f %s/s)",
            self.step.value, context.dataset_id, row_count, elapsed,
            row_count / elapsed if elapsed > 0 else 0.0, self.progress_unit,
        )
        reopened = self._open_output(context, config_digest, lineage_digest)
        if published.serialized() != reopened.serialized():
            raise RuntimeError("published artifact metadata mismatch after bound reopen")
        return reopened
