"""Generation-3 artifact-only dependency orchestration."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass
from types import MappingProxyType

from ...config import PipelineConfig
from ...utils.artifacts import ArtifactIndex, PipelineStep
from ...utils.run_lock import RunLock
from .base import (
    ArtifactExpectation,
    ArtifactStage,
    StageRunContext,
    _mint_runner_expectation,
)


StageFactory = Callable[[PipelineConfig, str], ArtifactStage]
_STEP_ORDER = tuple(PipelineStep)
_ACTIVE_DATASET_RUN: ContextVar["DatasetRunContext | None"] = ContextVar(
    "audio_filter_pipeline_active_dataset_run", default=None,
)
_CAPABILITY_MINT = object()


class DatasetRunLockCapability:
    """仅能由活跃 DatasetRunContext 显式交给持久化工作线程的写入能力。"""

    __slots__ = ("_context", "_lock_token", "_mint")

    def __init__(
        self,
        context: "DatasetRunContext",
        lock_token: str,
        mint: object,
    ) -> None:
        if mint is not _CAPABILITY_MINT:
            raise TypeError("dataset run lock capability cannot be constructed directly")
        self._context = context
        self._lock_token = lock_token
        self._mint = mint

    @property
    def output_root(self) -> str:
        return self._context.output_root

    @property
    def dataset_id(self) -> str:
        return self._context.dataset_id

    def validate(self, output_root: str, dataset_id: str) -> None:
        """验证真实持锁状态；不依赖工作线程不可见的 ContextVar。"""
        if (
            self._mint is not _CAPABILITY_MINT
            or self._context._persistent_capability is not self
            or self._context._token is None
            or self._context._lock.token != self._lock_token
        ):
            raise RuntimeError("dataset run lock capability is no longer active")
        if os.path.realpath(os.path.abspath(output_root)) != self.output_root:
            raise ValueError("dataset run lock capability output_root does not match")
        if dataset_id != self.dataset_id:
            raise ValueError("dataset run lock capability dataset_id does not match")


class DatasetRunContext:
    """显式持有一个 output-root/dataset 的完整 orchestration 锁域。"""

    def __init__(self, output_root: str, dataset_id: str) -> None:
        if type(output_root) is not str or not output_root.strip():
            raise ValueError("dataset run output_root must be a nonblank string")
        if type(dataset_id) is not str or not dataset_id.strip():
            raise ValueError("dataset run dataset_id must be a nonblank string")
        self.output_root = os.path.realpath(os.path.abspath(output_root))
        self.dataset_id = dataset_id
        self._lock = RunLock(output_root, dataset_id)
        self._token: Token | None = None
        self._persistent_capability: DatasetRunLockCapability | None = None

    @property
    def active(self) -> bool:
        return self._token is not None and _ACTIVE_DATASET_RUN.get() is self

    def validate(self, output_root: str, dataset_id: str) -> None:
        if not self.active:
            raise RuntimeError("dataset run context must be active")
        if os.path.realpath(os.path.abspath(output_root)) != self.output_root:
            raise ValueError("dataset run output_root does not match runner")
        if dataset_id != self.dataset_id:
            raise ValueError("dataset run dataset_id does not match runner")

    def persistent_write_capability(self) -> DatasetRunLockCapability:
        """返回可显式传入工作线程的持久化写入 capability。"""
        if not self.active or self._persistent_capability is None:
            raise RuntimeError("dataset run context must be active")
        return self._persistent_capability

    def __enter__(self) -> "DatasetRunContext":
        if self._token is not None:
            raise RuntimeError("dataset run context is already active")
        active = _ACTIVE_DATASET_RUN.get()
        if active is not None:
            raise RuntimeError("another dataset run context is already active")
        self._lock.acquire()
        try:
            self._token = _ACTIVE_DATASET_RUN.set(self)
            lock_token = self._lock.token
            if lock_token is None:
                raise RuntimeError("dataset run lock token is unavailable")
            self._persistent_capability = DatasetRunLockCapability(
                self, lock_token, _CAPABILITY_MINT,
            )
        except BaseException:
            if self._token is not None:
                _ACTIVE_DATASET_RUN.reset(self._token)
                self._token = None
            self._persistent_capability = None
            self._lock.release()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        token = self._token
        if token is None:
            raise RuntimeError("dataset run context is not active")
        self._persistent_capability = None
        self._token = None
        _ACTIVE_DATASET_RUN.reset(token)
        return self._lock.__exit__(exc_type, exc, traceback)


def _parse_requested_step(value: PipelineStep | str) -> PipelineStep:
    if isinstance(value, PipelineStep):
        return value
    if type(value) is not str:
        raise ValueError("requested_step must be a PipelineStep or valid step string")
    normalized = value.strip().lower()
    if normalized == "all":
        return PipelineStep.STEP6
    try:
        return PipelineStep(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid requested_step: {value!r}") from exc


@dataclass(frozen=True)
class V3RunResult(Mapping[PipelineStep, ArtifactIndex]):
    """Immutable artifacts produced or resumed through a requested target."""

    artifacts: Mapping[PipelineStep, ArtifactIndex]
    target_step: PipelineStep

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))
        if self.target_step not in self.artifacts:
            raise ValueError("target_step must be present in artifacts")

    @property
    def target(self) -> ArtifactIndex:
        return self.artifacts[self.target_step]

    def __getitem__(self, step: PipelineStep) -> ArtifactIndex:
        return self.artifacts[step]

    def __iter__(self) -> Iterator[PipelineStep]:
        return iter(self.artifacts)

    def __len__(self) -> int:
        return len(self.artifacts)


class V3PipelineRunner:
    """Build a single dataset's artifact dependency chain via injected stages."""

    def __init__(
        self,
        config: PipelineConfig,
        stage_factories: Mapping[PipelineStep, StageFactory],
    ) -> None:
        if not isinstance(config, PipelineConfig):
            raise TypeError("config must be a PipelineConfig")
        if not isinstance(stage_factories, Mapping):
            raise TypeError("stage_factories must be a mapping")
        factories: dict[PipelineStep, StageFactory] = {}
        for step, factory in stage_factories.items():
            if not isinstance(step, PipelineStep):
                raise TypeError("stage factory keys must be PipelineStep values")
            if not callable(factory):
                raise TypeError(f"stage factory for {step.value} must be callable")
            factories[step] = factory
        self._config = config
        self._stage_factories = MappingProxyType(factories)

    @staticmethod
    def _expectation(index: ArtifactIndex, expected_step: PipelineStep) -> ArtifactExpectation:
        return _mint_runner_expectation(index, expected_step)

    def _stage_for(self, step: PipelineStep, dataset_id: str) -> ArtifactStage:
        try:
            factory = self._stage_factories[step]
        except KeyError as exc:
            raise ValueError(f"missing factory for required {step.value}") from exc
        stage = factory(self._config, dataset_id)
        if not isinstance(stage, ArtifactStage):
            raise TypeError(f"factory for {step.value} must return an ArtifactStage")
        if stage.step is not step:
            raise ValueError(f"factory for {step.value} returned stage for {stage.step!r}")
        # 依赖链由各 stage 声明的 upstream_step 决定（允许非相邻依赖，
        # 如 Step5 直接消费 Step3）；顺序与环校验在递归解析阶段完成。
        return stage

    def _dependency_stages(
        self, dataset_id: str, target_step: PipelineStep,
    ) -> tuple[tuple[PipelineStep, ArtifactStage], ...]:
        """解析 target 的 upstream 闭包，并缓存本次解析出的 stage 实例。"""
        visiting: set[PipelineStep] = set()
        resolved: dict[PipelineStep, ArtifactStage] = {}
        ordered: list[PipelineStep] = []

        def visit(step: PipelineStep) -> None:
            if step in visiting:
                raise ValueError(f"cycle detected in stage dependencies at {step.value}")
            if step in resolved:
                return
            visiting.add(step)
            stage = self._stage_for(step, dataset_id)
            if stage.upstream_step is not None:
                if _STEP_ORDER.index(stage.upstream_step) >= _STEP_ORDER.index(step):
                    if stage.upstream_step in visiting:
                        raise ValueError(
                            f"cycle detected in stage dependencies at {step.value}"
                        )
                    raise ValueError(
                        f"{step.value} stage upstream_step must precede its own step"
                    )
                visit(stage.upstream_step)
            visiting.remove(step)
            resolved[step] = stage
            ordered.append(step)

        visit(target_step)
        return tuple((step, resolved[step]) for step in ordered)

    def _dependency_order(
        self, dataset_id: str, target_step: PipelineStep,
    ) -> tuple[PipelineStep, ...]:
        """按 stage 声明的 upstream_step 返回 target 的拓扑依赖闭包。"""
        return tuple(step for step, _stage in self._dependency_stages(dataset_id, target_step))

    def required_steps(
        self, dataset_id: str, requested_step: PipelineStep | str,
    ) -> tuple[PipelineStep, ...]:
        """返回请求目标实际需要的步骤，不包含无关的旁支步骤。"""
        target_step = _parse_requested_step(requested_step)
        StageRunContext(self._config.output_root, dataset_id, self._config)
        if self._config.pipeline_generation != 3:
            raise ValueError("V3PipelineRunner requires pipeline generation 3")
        return self._dependency_order(dataset_id, target_step)

    def _run_locked(
        self, dataset_id: str, target_step: PipelineStep,
    ) -> V3RunResult:
        active = _ACTIVE_DATASET_RUN.get()
        if active is None:
            raise RuntimeError("dataset run orchestration context is not active")
        active.validate(self._config.output_root, dataset_id)

        artifacts: dict[PipelineStep, ArtifactIndex] = {}
        for step, stage in self._dependency_stages(dataset_id, target_step):
            upstream = None
            if stage.upstream_step is not None:
                upstream_index = artifacts.get(stage.upstream_step)
                if upstream_index is None:
                    raise ValueError(
                        f"missing upstream artifact {stage.upstream_step.value} "
                        f"for {step.value}"
                    )
                upstream = self._expectation(upstream_index, stage.upstream_step)
            context = StageRunContext(
                self._config.output_root,
                dataset_id,
                self._config,
                upstream=upstream,
                persistent_write_capability=active.persistent_write_capability(),
            )
            index = stage.run(context)
            self._expectation(index, step)
            artifacts[step] = index
        return V3RunResult(artifacts, target_step)

    def run(self, dataset_id: str, requested_step: PipelineStep | str) -> V3RunResult:
        target_step = _parse_requested_step(requested_step)
        # Validate all global/dataset context constraints before touching a factory.
        StageRunContext(self._config.output_root, dataset_id, self._config)
        if self._config.pipeline_generation != 3:
            raise ValueError("V3PipelineRunner requires pipeline generation 3")
        active = _ACTIVE_DATASET_RUN.get()
        if active is not None:
            return self._run_locked(dataset_id, target_step)
        # CLI orchestration 会在 cleanup/run/export 外层持有完整锁域；这里只为未建立
        # 外层上下文的 direct caller 补 runner artifact 链锁，不覆盖调用方自己的
        # cleanup/export 操作。
        with DatasetRunContext(self._config.output_root, dataset_id):
            return self._run_locked(dataset_id, target_step)
