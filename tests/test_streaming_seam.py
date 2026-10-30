"""公共 streaming seam 与 AEC/TSE adapter 合同测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from core.inference.streaming import StreamState, _DelegatingSession
from tasks.aec.inference.streaming_aec_frontend import (
    AECStreamingAdapter,
    StreamingAECFrontend,
)
from tasks.aec.inference.streaming_delay_estimator import (
    StreamingDelayEstimator,
    StreamingDelayEstimatorConfig,
)
from tasks.tse.inference.streaming import TSEStreamingSession


pytestmark = pytest.mark.unit


class _Delegate:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.push_calls = []
        self.flush_calls = 0

    def push(self, value):
        self.push_calls.append(value)
        return np.asarray(value, dtype=np.float32)

    def flush(self):
        self.flush_calls += 1
        return np.zeros(0, dtype=np.float32)

    def reset(self):
        self.reset_calls += 1


class _FailingDelegate(_Delegate):
    def push(self, value):
        raise RuntimeError("delegate failure")


class _FailingResetDelegate(_Delegate):
    def reset(self):
        self.reset_calls += 1
        raise RuntimeError("delegate reset failure")


class _TestSession(_DelegatingSession):
    def _push_impl(self, inputs):
        return self._delegate.push(inputs["audio"])

    def _flush_impl(self):
        return self._delegate.flush()


class _AECModel(torch.nn.Module):
    CACHE_KEYS = ("state",)

    def init_cache(self, batch, device, dtype):
        return {"state": torch.zeros(batch, 1, 1, 2, device=device, dtype=dtype)}

    def forward(self, mic_spec, far_spec, cache):
        return mic_spec, cache


def _aec_frontend() -> StreamingAECFrontend:
    return StreamingAECFrontend(
        model=_AECModel(),
        delay_estimator=StreamingDelayEstimator(
            StreamingDelayEstimatorConfig(
                sample_rate=16000,
                buffer_duration_ms=500.0,
                max_residual_delay_ms=100.0,
                update_interval_ms=50.0,
                smooth_alpha=0.3,
            )
        ),
        n_fft=8,
        hop_length=4,
        win_length=8,
    )


def test_delegating_session_enforces_roles_and_lifecycle():
    delegate = _Delegate()
    session = _TestSession(delegate, frozenset({"audio"}))

    assert session.state is StreamState.READY
    with pytest.raises(ValueError, match="input roles"):
        session.push({"other": np.zeros(4, dtype=np.float32)})

    result = session.push({"audio": np.ones(4, dtype=np.float32)})
    assert session.state is StreamState.ACTIVE
    np.testing.assert_array_equal(result, np.ones(4, dtype=np.float32))

    session.flush()
    assert session.state is StreamState.FLUSHED
    with pytest.raises(RuntimeError, match="flushed"):
        session.push({"audio": np.zeros(4, dtype=np.float32)})

    session.reset()
    assert session.state is StreamState.READY
    assert delegate.reset_calls == 1


def test_delegating_session_poison_requires_reset():
    session = _TestSession(_FailingDelegate(), frozenset({"audio"}))
    with pytest.raises(RuntimeError, match="delegate failure"):
        session.push({"audio": np.zeros(4, dtype=np.float32)})
    assert session.state is StreamState.POISONED
    with pytest.raises(RuntimeError, match="poisoned"):
        session.flush()


def test_delegating_session_is_poisoned_when_reset_fails():
    """半重置的 session 不得标为 READY，语义与 push/flush 的 POISONED 一致。"""
    session = _TestSession(_FailingResetDelegate(), frozenset({"audio"}))
    with pytest.raises(RuntimeError, match="delegate reset failure"):
        session.reset()
    assert session.state is StreamState.POISONED
    with pytest.raises(RuntimeError, match="poisoned"):
        session.push({"audio": np.zeros(4, dtype=np.float32)})


def test_aec_adapter_exposes_only_mic_and_farend_roles():
    adapter = AECStreamingAdapter(_aec_frontend())
    assert adapter.input_roles == frozenset({"mic", "farend"})
    session = adapter.open_session()
    assert session.state is StreamState.READY

    frame = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="input roles"):
        session.push({"mic": frame})
    session.push({"mic": frame, "farend": frame})
    assert session.state is StreamState.ACTIVE
    session.reset()
    assert session.state is StreamState.READY


def test_tse_session_hides_delegate_state_and_maps_mixture_role():
    delegate = _Delegate()
    session = TSEStreamingSession(delegate)

    assert session.state is StreamState.READY
    value = np.ones(3, dtype=np.float32)
    np.testing.assert_array_equal(session.push({"mixture": value}), value)
    assert delegate.push_calls == [value]
    assert session.flush().size == 0
    session.reset()
    assert session.state is StreamState.READY
