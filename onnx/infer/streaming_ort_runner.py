"""通用 streaming ONNX Runtime 单帧 runner。

cache 从初始化到最后一帧始终保持 NumPy array，避免逐帧 NumPy/Torch 往返。
STFT/iSTFT 与模型专属音频前后处理仍由各 CLI 管理。
"""

from __future__ import annotations

import time
from collections.abc import Iterator, ValuesView
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from onnx.runtime_spec import (
    StreamingRuntimeSpec,
    runtime_spec_from_session,
    validate_static_batch_graph,
)


class _SnapshotValues(Mapping[str, np.ndarray]):
    """兼容 Mapping.values() 的只读 cache values。"""

    def __init__(self, values: Mapping[str, np.ndarray]) -> None:
        frozen = {}
        for name, value in values.items():
            source = np.ascontiguousarray(value)
            frozen[str(name)] = np.frombuffer(
                source.tobytes(), dtype=source.dtype
            ).reshape(source.shape)
        self._values = MappingProxyType(frozen)

    def __getitem__(self, name: str) -> np.ndarray:
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __call__(self) -> ValuesView[np.ndarray]:
        return ValuesView(self)


@dataclass(frozen=True, eq=False)
class StreamingORTStateSnapshot(Mapping[str, np.ndarray]):
    """仅可由创建它的进程内 runner 恢复的只读状态快照。"""

    model_identity: str
    runner_nonce: str
    cache_abi_version: int | None
    cache_semantics: object | None
    values: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _SnapshotValues(self.values))

    def __getitem__(self, name: str) -> np.ndarray:
        return self.values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class StreamingORTResult:
    """逐帧 ORT 结果；``frames`` 的 time axis 仍为单帧轴。"""

    frames: tuple[np.ndarray, ...]
    final_state: Mapping[str, np.ndarray]
    frame_times: tuple[float, ...]

    def concatenate(self, axis: int = 2) -> np.ndarray:
        if not self.frames:
            raise RuntimeError("Streaming ORT produced no output frames")
        return np.concatenate(self.frames, axis=axis)

    def to_torch(self, axis: int = 2):
        import torch

        return torch.from_numpy(self.concatenate(axis=axis))


class StreamingORTFrameRunner:
    """按 ``StreamingRuntimeSpec`` 校验并执行一个 ORT streaming session。"""

    def __init__(
        self,
        session,
        model_id: str | None = None,
        *,
        spec: StreamingRuntimeSpec | None = None,
        batch_size: int = 1,
    ) -> None:
        if spec is None:
            if model_id is None:
                raise ValueError("model_id is required when spec is not provided")
            spec = runtime_spec_from_session(session, model_id)
        self.session = session
        self.spec = spec
        if isinstance(batch_size, bool):
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = int(batch_size)
        if self.batch_size < 1 or self.batch_size != batch_size:
            raise ValueError("batch_size must be a positive integer")
        static_batch_size = spec.profile.static_batch_size
        if static_batch_size is not None and self.batch_size != static_batch_size:
            raise ValueError(
                f"{spec.profile.model_id} runner requires static batch size "
                f"{static_batch_size}, got {self.batch_size}"
            )

        input_infos = list(session.get_inputs())
        output_infos = list(session.get_outputs())
        validate_static_batch_graph(spec.profile, input_infos, output_infos)
        actual_inputs = tuple(value.name for value in input_infos)
        actual_outputs = tuple(value.name for value in output_infos)
        if actual_inputs != spec.input_names or actual_outputs != spec.output_names:
            raise RuntimeError(
                f"ONNX I/O ABI mismatch: expected inputs={spec.input_names}, "
                f"outputs={spec.output_names}; got inputs={actual_inputs}, "
                f"outputs={actual_outputs}"
            )
        from onnx.utils import materialize_cache_shapes

        cache_infos = list(session.get_inputs())[len(spec.profile.primary_inputs):]
        shapes = materialize_cache_shapes(
            {name: list(shape) for name, shape in spec.cache_shapes.items()},
            cache_infos,
            self.batch_size,
            allowed_dynamic_axes=spec.profile.allowed_cache_dynamic_axes,
        )
        self._primary_infos = {
            value.name: value
            for value in input_infos[:len(spec.profile.primary_inputs)]
        }
        self._cache_infos = {value.name: value for value in cache_infos}
        ort_dtypes = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(bool)": np.bool_,
        }
        cold_state: dict[str, np.ndarray] = {}
        for name in spec.cache_input_names:
            ort_type = getattr(self._cache_infos[name], "type", "tensor(float)")
            try:
                dtype = ort_dtypes[ort_type]
            except KeyError as exc:
                raise RuntimeError(
                    f"Unsupported ORT cache input type for {name!r}: {ort_type!r}"
                ) from exc
            cold_state[name] = np.zeros(tuple(shapes[name]), dtype=dtype)
        self._cold_state = cold_state
        self._state = self._copy_arrays(cold_state)
        self._last_outputs: Mapping[str, np.ndarray] = {}
        self._model_identity = self._resolve_model_identity()
        self._cache_abi_version, self._cache_semantics = (
            self._resolve_cache_abi_contract()
        )
        self._runner_nonce = uuid4().hex

    @staticmethod
    def _copy_arrays(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in values.items()}

    def _resolve_model_identity(self) -> str:
        if self.spec.profile.model_id == "dbgnet":
            if self.spec.dbgnet is None:
                raise RuntimeError("DBGNet runner requires versioned runtime metadata")
            deployment = self.spec.deployment_config
            config_sha256 = (
                deployment.get("sha256") if isinstance(deployment, Mapping) else None
            )
            if config_sha256:
                return f"{self.spec.dbgnet.model_identity}:{config_sha256}"
            return self.spec.dbgnet.model_identity
        return self.spec.profile.model_id

    def _resolve_cache_abi_contract(self) -> tuple[int | None, object | None]:
        # ``StreamingRuntimeSpec`` 自身不含 cache ABI 字段（死分支已移除），
        # ABI 只存在于各模型专属 runtime sub-spec。
        for runtime_spec in (
            self.spec.cagcrn,
            self.spec.deepvqe,
            self.spec.dbgnet,
            self.spec.tse,
        ):
            if runtime_spec is not None and hasattr(runtime_spec, "cache_abi_version"):
                return (
                    runtime_spec.cache_abi_version,
                    getattr(runtime_spec, "cache_semantics", None),
                )
        return None, None

    def _snapshot(self, values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        if self.spec.profile.model_id != "dbgnet":
            return self._copy_arrays(values)
        return StreamingORTStateSnapshot(
            model_identity=self._model_identity,
            runner_nonce=self._runner_nonce,
            cache_abi_version=self._cache_abi_version,
            cache_semantics=self._cache_semantics,
            values=values,
        )

    @property
    def state(self) -> Mapping[str, np.ndarray]:
        return self._snapshot(self._state)

    @property
    def last_outputs(self) -> Mapping[str, np.ndarray]:
        """最近一帧的完整 ORT 输出，供 state probabilities 等辅助 gate 使用。"""
        return self._copy_arrays(self._last_outputs)

    def reset(self, state: Mapping[str, np.ndarray] | None = None) -> None:
        if state is None:
            self._state = self._copy_arrays(self._cold_state)
            self._last_outputs = {}
            return
        if self.spec.profile.model_id == "dbgnet":
            if not isinstance(state, StreamingORTStateSnapshot):
                raise TypeError("DBGNet reset requires a versioned state snapshot")
            if state.model_identity != self._model_identity:
                raise ValueError("DBGNet state model identity mismatch")
            if state.runner_nonce != self._runner_nonce:
                raise ValueError("DBGNet state runner identity mismatch")
            if state.cache_abi_version != self._cache_abi_version:
                raise ValueError("DBGNet state cache ABI version mismatch")
            if state.cache_semantics != self._cache_semantics:
                raise ValueError("DBGNet state cache semantics mismatch")
        if tuple(state) != self.spec.cache_input_names:
            raise ValueError(
                f"state names/order mismatch: expected {self.spec.cache_input_names}, "
                f"got {tuple(state)}"
            )
        validated_state: dict[str, np.ndarray] = {}
        for name in self.spec.cache_input_names:
            value = np.asarray(state[name])
            expected = self._cold_state[name]
            if value.ndim != expected.ndim:
                raise ValueError(
                    f"state rank mismatch for {name!r}: expected {expected.ndim}, "
                    f"got {value.ndim}"
                )
            if value.dtype != expected.dtype:
                raise ValueError(
                    f"state dtype mismatch for {name!r}: expected {expected.dtype}, "
                    f"got {value.dtype}"
                )
            graph_shape = self._cache_infos[name].shape
            for axis, (expected_dim, actual_dim, graph_dim) in enumerate(
                zip(expected.shape, value.shape, graph_shape)
            ):
                dynamic_name = graph_dim.lower() if isinstance(graph_dim, str) else None
                if actual_dim == expected_dim:
                    continue
                if dynamic_name in self.spec.profile.allowed_cache_dynamic_axes:
                    continue
                raise ValueError(
                    f"state shape mismatch for {name!r} axis {axis}: "
                    f"expected {expected_dim}, got {actual_dim}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"state for {name!r} is non-finite")
            validated_state[name] = value.copy()
        self._state = validated_state
        self._last_outputs = {}

    @staticmethod
    def _as_numpy(value) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        try:
            return value.detach().cpu().numpy()
        except AttributeError:
            return np.asarray(value)

    def _validate_primary_input(self, name: str, value: np.ndarray) -> np.ndarray:
        """拒绝与首帧 ORT ABI 不一致或含非有限值的主输入。"""
        info = self._primary_infos[name]
        expected_type = getattr(info, "type", None)
        dtype_by_ort_type = {
            "tensor(float)": np.dtype(np.float32),
            "tensor(float16)": np.dtype(np.float16),
            "tensor(double)": np.dtype(np.float64),
        }
        if expected_type is not None and expected_type not in dtype_by_ort_type:
            raise RuntimeError(
                f"Unsupported ORT primary input type for {name!r}: {expected_type!r}"
            )
        expected_dtype = dtype_by_ort_type.get(expected_type)
        if expected_dtype is not None and value.dtype != expected_dtype:
            raise ValueError(
                f"primary input dtype mismatch for {name!r}: expected {expected_dtype}, "
                f"got {value.dtype}"
            )
        shape = getattr(info, "shape", None)
        if expected_type is not None:
            if shape is None or value.ndim != len(shape):
                raise ValueError(
                    f"primary input rank mismatch for {name!r}: expected "
                    f"{len(shape) if shape is not None else None}, got {value.ndim}"
                )
            for axis, (actual, expected) in enumerate(zip(value.shape, shape)):
                if isinstance(expected, int) and not isinstance(expected, bool) and actual != expected:
                    raise ValueError(
                        f"primary input shape mismatch for {name!r} axis {axis}: "
                        f"expected {expected}, got {actual}"
                    )
        if value.shape[0] != self.batch_size:
            raise ValueError(
                f"primary input batch mismatch for {name!r}: expected {self.batch_size}, "
                f"got {value.shape[0]}"
            )
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError(f"primary input {name!r} is empty or non-finite")
        return value

    def run_frame(self, *primary_inputs) -> tuple[np.ndarray, float]:
        expected = len(self.spec.profile.primary_inputs)
        if len(primary_inputs) != expected:
            raise ValueError(f"Expected {expected} primary inputs, got {len(primary_inputs)}")
        feeds = {
            name: self._validate_primary_input(name, self._as_numpy(value))
            for name, value in zip(self.spec.profile.primary_inputs, primary_inputs)
        }
        feeds.update(self._state)
        started = time.perf_counter()
        values = self.session.run(None, feeds)
        elapsed = time.perf_counter() - started
        if len(values) != len(self.spec.output_names):
            raise RuntimeError(
                f"ORT output count mismatch: expected {len(self.spec.output_names)}, "
                f"got {len(values)}"
            )
        outputs = {
            name: np.asarray(value) for name, value in zip(self.spec.output_names, values)
        }
        primary = outputs[self.spec.profile.primary_output]
        if primary.size == 0 or not np.all(np.isfinite(primary)):
            raise RuntimeError(
                f"ORT primary output {self.spec.profile.primary_output!r} is empty or non-finite"
            )
        for name in self.spec.profile.auxiliary_outputs:
            value = outputs[name]
            if value.size == 0 or not np.all(np.isfinite(value)):
                raise RuntimeError(
                    f"ORT auxiliary output {name!r} is empty or non-finite"
                )
        next_state: dict[str, np.ndarray] = {}
        for cache_input, cache_output in zip(
            self.spec.cache_input_names, self.spec.cache_output_names
        ):
            value = outputs[cache_output]
            current = self._state[cache_input]
            if value.ndim != current.ndim:
                raise RuntimeError(
                    f"ORT cache rank changed for {cache_input!r}: "
                    f"expected {current.ndim}, got {value.ndim}"
                )
            graph_shape = self._cache_infos[cache_input].shape
            for axis, (old_dim, new_dim, graph_dim) in enumerate(
                zip(current.shape, value.shape, graph_shape)
            ):
                dynamic_name = graph_dim.lower() if isinstance(graph_dim, str) else None
                if new_dim == old_dim:
                    continue
                if dynamic_name in self.spec.profile.allowed_cache_dynamic_axes:
                    continue
                raise RuntimeError(
                    f"ORT cache shape changed for {cache_input!r} axis {axis}: "
                    f"expected {old_dim}, got {new_dim}"
                )
            if value.dtype != current.dtype:
                raise RuntimeError(
                    f"ORT cache dtype changed for {cache_input!r}: "
                    f"expected {current.dtype}, got {value.dtype}"
                )
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"ORT cache output {cache_output!r} is non-finite")
            next_state[cache_input] = value.copy()

        # 全部 cache 校验通过后才提交状态，避免半帧污染。
        committed_outputs = self._copy_arrays(outputs)
        self._state = next_state
        self._last_outputs = committed_outputs
        return committed_outputs[self.spec.profile.primary_output].copy(), elapsed

    def run(self, frame_inputs: Iterable[Sequence[object]]) -> StreamingORTResult:
        frames: list[np.ndarray] = []
        frame_times: list[float] = []
        for values in frame_inputs:
            frame, elapsed = self.run_frame(*values)
            frames.append(frame)
            frame_times.append(elapsed)
        if not frames:
            raise RuntimeError("Streaming ORT input contains no frames")
        return StreamingORTResult(
            frames=tuple(frame.copy() for frame in frames),
            final_state=self._snapshot(self._state),
            frame_times=tuple(frame_times),
        )
