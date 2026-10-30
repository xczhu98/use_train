"""任务无关的流式推理 seam。

公共层只约定会话生命周期和输入角色；模型 cache、延迟估计器、enrollment
以及 WOLA 缓存都由任务侧 adapter 私有持有。
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any, Protocol


class StreamState(str, Enum):
    """流式会话的可观察生命周期状态。"""

    READY = "ready"
    ACTIVE = "active"
    FLUSHED = "flushed"
    POISONED = "poisoned"


class StreamSession(Protocol):
    """一次流式推理会话的最小公共合同。

    ``push`` 的输入 role 由具体 adapter 声明；返回值是任务定义的 waveform
    chunk，公共层不解释其 shape 或 dtype。任务内部 state 不得通过该合同暴露。
    """

    @property
    def state(self) -> StreamState:
        ...

    def push(self, inputs: Mapping[str, Any]) -> Any:
        ...

    def flush(self) -> Any:
        ...

    def reset(self) -> None:
        ...


class StreamingAdapter(Protocol):
    """创建独立流式会话的任务 adapter 合同。"""

    @property
    def input_roles(self) -> frozenset[str]:
        ...

    def open_session(self) -> StreamSession:
        ...


class _DelegatingSession:
    """把任务原生 session 接到公共生命周期，隐藏内部 cache。"""

    def __init__(self, delegate: Any, required_roles: frozenset[str]) -> None:
        self._delegate = delegate
        self._required_roles = required_roles
        self._state = StreamState.READY

    @property
    def state(self) -> StreamState:
        return self._state

    def _validate_inputs(self, inputs: Mapping[str, Any]) -> None:
        if not isinstance(inputs, Mapping):
            raise TypeError("stream inputs must be a mapping of declared roles")
        if set(inputs) != set(self._required_roles):
            raise ValueError(
                "stream input roles must exactly match "
                f"{sorted(self._required_roles)}, got {sorted(inputs)}"
            )

    def push(self, inputs: Mapping[str, Any]) -> Any:
        if self._state is StreamState.FLUSHED:
            raise RuntimeError("stream session was flushed; call reset() before reuse")
        if self._state is StreamState.POISONED:
            raise RuntimeError("stream session is poisoned; call reset() before reuse")
        self._validate_inputs(inputs)
        try:
            output = self._push_impl(inputs)
        except Exception:
            self._state = StreamState.POISONED
            raise
        self._state = StreamState.ACTIVE
        return output

    def flush(self) -> Any:
        if self._state is StreamState.POISONED:
            raise RuntimeError("stream session is poisoned; call reset() before reuse")
        if self._state is StreamState.FLUSHED:
            return self._empty_flush()
        try:
            output = self._flush_impl()
        except Exception:
            self._state = StreamState.POISONED
            raise
        self._state = StreamState.FLUSHED
        return output

    def reset(self) -> None:
        try:
            self._reset_impl()
        except Exception:
            # 半重置的 session 不能标为可复用，语义与 push/flush 的 POISONED 一致。
            self._state = StreamState.POISONED
            raise
        self._state = StreamState.READY

    def _push_impl(self, inputs: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def _flush_impl(self) -> Any:
        raise NotImplementedError

    def _reset_impl(self) -> None:
        reset = getattr(self._delegate, "reset", None)
        if not callable(reset):
            raise TypeError("stream delegate must implement reset()")
        reset()

    def _empty_flush(self) -> Any:
        return self._flush_impl()
