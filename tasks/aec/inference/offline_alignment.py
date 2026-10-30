"""Offline ref/mic time alignment for PyTorch AEC inference.

Training uses windowed GCC-PHAT (``tasks/aec/inference/gcc_phat_core.py``) to align
the farend reference with the microphone echo before the neural model sees the
pair.  The same alignment must be applied at PyTorch offline inference time;
otherwise the model receives a mis-aligned ref/echo pair and performance
drops.  (The ONNX/streaming path uses :class:`StreamingDelayEstimator`
instead.)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tasks.aec.inference.gcc_phat_core import (
    clamp_gcc_delay_samples,
    estimate_gcc_tau_windowed,
    gcc_phat_tau_with_confidence as _core_gcc_phat_tau_with_confidence,
)


def gcc_phat_tau_seconds(
    sig: np.ndarray,
    refsig: np.ndarray,
    fs: int,
    interp: int = 4,
    *,
    max_tau_seconds: Optional[float] = None,
    bias_seconds: float = 0.0,
    non_negative: bool = True,
) -> float:
    """Estimate delay (seconds) of ``sig`` relative to ``refsig`` via GCC-PHAT.

    Positive delay means ``sig`` lags ``refsig`` (the usual echo case).
    薄封装：FFT 主体在 ``tasks.aec.inference.gcc_phat_core``（与训练数据管道同一份
    实现），正 ``bias_seconds`` 表示从 raw estimate 中扣除的固定估计偏差。

    Args:
        sig: Signal that may be delayed (e.g. microphone).
        refsig: Reference signal (e.g. farend).
        fs: Sample rate (Hz).
        interp: Interpolation factor for sub-sample peak refinement.
        max_tau_seconds: Maximum absolute delay to search.
        bias_seconds: Fixed positive estimator bias subtracted from the raw estimate.
        non_negative: If True, restrict the search to non-negative lags
            (sig lags refsig). This is the physically valid case for AEC,
            where echo cannot arrive before the farend signal.  Default is True.

    Returns:
        Estimated delay in seconds.  Positive means ``sig`` lags ``refsig``.
    """
    tau_sec, _, _ = _core_gcc_phat_tau_with_confidence(
        sig,
        refsig,
        fs,
        interp=interp,
        max_tau_seconds=max_tau_seconds,
        non_negative=non_negative,
    )
    bias = float(bias_seconds)
    if not np.isfinite(bias):
        raise ValueError(f"bias_seconds must be finite, got {bias_seconds!r}")
    return tau_sec - bias


def align_farend_to_mic(
    mic: np.ndarray,
    far: np.ndarray,
    fs: int,
    *,
    enabled: bool = True,
    max_tau_seconds: float = 0.2,
    downsample: Optional[int] = None,
    bias_seconds: float = 0.0,
    pad_policy: str = "apply",
    min_pad_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align farend reference to microphone echo using windowed GCC-PHAT.

    Echo always lags the farend reference.  Therefore we estimate the delay
    ``tau`` of mic relative to far and pad the front of far by ``tau`` samples
    so that the aligned far arrives at the same time as the echo in mic.

    估计主流程与训练数据管道同一内核
    （``tasks.aec.inference.gcc_phat_core.estimate_gcc_tau_windowed``）：head+tail
    双 1s 窗 @4k 降采样快路径，双窗分歧或全灭时升级全量扫描；全窗低置信度
    → tau=0 不补零（zero_pad 策略）。时变对退化为文件级中值 tau 做整段对齐。

    Args:
        mic: Microphone signal, shape ``(N,)``.
        far: Farend/reference signal, shape ``(M,)``.
        fs: Sample rate (Hz).
        enabled: If False, return inputs unchanged and delay 0.
        max_tau_seconds: Maximum absolute delay to search.
        downsample: Explicit GCC downsampling factor. ``None`` selects an integer
            factor targeting approximately 4 kHz; explicit values override it.
        bias_seconds: Fixed positive estimator bias subtracted from the raw delay.
            Default is 0; set only for a measured pipeline offset.
        pad_policy: Alignment policy. Only ``"apply"`` is supported.
        min_pad_samples: Delays smaller than this (in samples) are treated as 0.

    Returns:
        ``(mic, far_aligned, effective_tau_seconds)``. The effective delay is
        bias-adjusted, truncated to integer samples, capped, and filtered by
        ``min_pad_samples``; the aligned far signal is front-padded by that count.
    """
    mic = np.asarray(mic, dtype=np.float32).ravel()
    far = np.asarray(far, dtype=np.float32).ravel()

    _pol = (pad_policy or "apply").strip().lower()
    if _pol != "apply":
        raise ValueError(
            "pad_policy only supports 'apply'; use enabled=False to disable "
            f"alignment, got {_pol!r}"
        )

    if not enabled or mic.size < 2 or far.size < 2:
        return mic, far, 0.0

    # 窗式主流程（与训练同一内核）：估计 mic(echo) 相对 far 的滞后；
    # 低置信度 → tau=0，不补零。搜索固定非负（echo 不可能领先 farend）。
    est = estimate_gcc_tau_windowed(
        far,
        mic,
        fs,
        downsample=downsample,
        max_tau_seconds=max_tau_seconds,
        peak_threshold=0.05,
        prominence_db=6.0,
        tv_iqr_ms=10.0,
        max_windows=8,
    )
    tau = 0.0 if est.low_confidence else float(est.tau_sec)
    tau_samp = clamp_gcc_delay_samples(
        tau,
        fs,
        bias_seconds=bias_seconds,
        max_delay_seconds=max_tau_seconds,
        min_delay_samples=min_pad_samples,
    )

    effective_tau = tau_samp / float(fs)
    if tau_samp == 0:
        return mic, far, effective_tau

    # Echo lags far -> pad far front so ref waits for echo.
    far_aligned = np.concatenate(
        [np.zeros(tau_samp, dtype=far.dtype), far], axis=0
    )
    return mic, far_aligned, effective_tau


if __name__ == "__main__":
    """Self-test: synthetic echo delay."""
    fs = 16000
    true_delay_ms = 87.5
    true_delay_samp = int(round(true_delay_ms * fs / 1000.0))
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1.0, fs).astype(np.float32)
    ref = np.convolve(ref, np.ones(8) / 8, mode="same").astype(np.float32)
    ref *= 0.5 / (np.std(ref) + 1e-12)

    echo = np.zeros_like(ref)
    echo[true_delay_samp:] = ref[:-true_delay_samp] * 0.3
    mic = echo + rng.normal(0, 0.05, ref.size).astype(np.float32)

    _, aligned, tau = align_farend_to_mic(mic, ref, fs)
    print(f"true delay: {true_delay_ms:.2f} ms, estimated: {tau * 1000:.2f} ms")
    print(f"input ref length: {ref.size}, aligned ref length: {aligned.size}")
    assert abs(tau * 1000 - true_delay_ms) < 10.0
    print("Self-test passed.")
