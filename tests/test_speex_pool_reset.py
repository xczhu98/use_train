"""Unit test for Speex MDF optimized ctypes / C batch consistency.

Verifies that:
1. The optimised _SpeexAECBackend (pre-allocated buffers + pointer-offset)
   produces identical output to the legacy per-frame np.zeros path.
2. C batch wrapper (if available) produces identical output to ctypes loop.
"""
import pytest

import ctypes

import numpy as np


def _speex_available():
    try:
        ctypes.CDLL("libspeexdsp.so.1")
        return True
    except OSError:
        return False


_SPEEX_AVAILABLE = _speex_available()


def _legacy_process(mic, ref, frame_size=256, filter_length=2048, sample_rate=16000):
    """Pure ctypes path with per-frame np.zeros (old behaviour)."""
    import ctypes

    SCALE = 32767.0
    lib = ctypes.CDLL("libspeexdsp.so.1")
    lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.speex_echo_state_init.restype = ctypes.c_void_p
    lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
    lib.speex_echo_state_destroy.restype = None
    spx_int16_p = ctypes.POINTER(ctypes.c_int16)
    lib.speex_echo_cancellation.argtypes = [
        ctypes.c_void_p, spx_int16_p, spx_int16_p, spx_int16_p,
    ]
    lib.speex_echo_cancellation.restype = None
    lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.speex_echo_ctl.restype = ctypes.c_int

    mic = np.clip(mic, -1.0, 1.0)
    ref = np.clip(ref, -1.0, 1.0)
    mic_i16 = (mic * SCALE).astype(np.int16)
    ref_i16 = (ref * SCALE).astype(np.int16)

    st = lib.speex_echo_state_init(frame_size, filter_length)
    sr_ctype = ctypes.c_int(sample_rate)
    lib.speex_echo_ctl(st, 24, ctypes.byref(sr_ctype))

    out_i16 = np.empty_like(mic_i16)
    n = len(mic_i16)
    for i in range(0, n, frame_size):
        end = min(i + frame_size, n)
        if end - i < frame_size:
            mic_frame = np.zeros(frame_size, dtype=np.int16)
            ref_frame = np.zeros(frame_size, dtype=np.int16)
            mic_frame[: end - i] = mic_i16[i:end]
            ref_frame[: end - i] = ref_i16[i:end]
        else:
            mic_frame = mic_i16[i:end]
            ref_frame = ref_i16[i:end]
        out_frame = np.empty(frame_size, dtype=np.int16)
        lib.speex_echo_cancellation(
            st,
            mic_frame.ctypes.data_as(spx_int16_p),
            ref_frame.ctypes.data_as(spx_int16_p),
            out_frame.ctypes.data_as(spx_int16_p),
        )
        out_i16[i:end] = out_frame[: end - i]

    lib.speex_echo_state_destroy(st)
    return out_i16.astype(np.float32) * (1.0 / SCALE)


def test_ctypes_optimised_consistency():
    """Optimised ctypes path must match legacy init/destroy bit-for-bit."""
    if not _SPEEX_AVAILABLE:
        pytest.skip("libspeexdsp.so.1 not available")
    from tasks.aec.data.linear_aec_frontend import LinearAECFrontend

    sr = 16000
    duration = 2.0
    t = np.arange(int(sr * duration)) / sr

    ref = np.sin(2 * np.pi * 500 * t + 100 * t * t).astype(np.float32)
    delay_samples = 512
    echo = np.zeros_like(ref)
    echo[delay_samples:] = ref[:-delay_samples] * 0.3
    noise = np.random.default_rng(0).standard_normal(len(t)).astype(np.float32)
    mic = (echo + noise * 0.05).astype(np.float32)

    cfg = {"backend": "speex", "frame_size": 256, "filter_length": 2048}
    frontend = LinearAECFrontend(cfg, sample_rate=sr)

    # run multiple times to ensure determinism
    errs = [frontend.process(mic, ref) for _ in range(3)]
    err_legacy = _legacy_process(mic, ref, frame_size=256, filter_length=2048, sample_rate=sr)

    for idx, err in enumerate(errs):
        np.testing.assert_allclose(
            err, err_legacy,
            rtol=1e-5, atol=1e-6,
            err_msg=f"Optimised run {idx} diverges from legacy init/destroy"
        )
    # inter-run consistency
    for i in range(1, len(errs)):
        np.testing.assert_allclose(
            errs[0], errs[i],
            rtol=1e-7, atol=1e-7,
            err_msg=f"Run {i} diverges from run 0 (non-deterministic?)"
        )


def test_c_batch_consistency():
    """C batch wrapper (if compiled) must match ctypes loop."""
    from tasks.aec.data.linear_aec_frontend import LinearAECFrontend, _c_batch_lib

    if _c_batch_lib is None:
        pytest.skip("C batch wrapper not available (compilation may have failed)")

    sr = 16000
    duration = 2.0
    t = np.arange(int(sr * duration)) / sr
    ref = np.sin(2 * np.pi * 500 * t + 100 * t * t).astype(np.float32)
    delay_samples = 512
    echo = np.zeros_like(ref)
    echo[delay_samples:] = ref[:-delay_samples] * 0.3
    noise = np.random.default_rng(1).standard_normal(len(t)).astype(np.float32)
    mic = (echo + noise * 0.05).astype(np.float32)

    cfg = {"backend": "speex", "frame_size": 256, "filter_length": 2048}
    frontend = LinearAECFrontend(cfg, sample_rate=sr)

    err = frontend.process(mic, ref)
    err_legacy = _legacy_process(mic, ref, frame_size=256, filter_length=2048, sample_rate=sr)

    np.testing.assert_allclose(
        err, err_legacy,
        rtol=1e-5, atol=1e-6,
        err_msg="C batch path diverges from legacy init/destroy"
    )


def test_c_batch_direct_call():
    """直接调用 _process_c_batch，确保 C batch 分支本身可执行。

    process() 中 C batch 失败会 fallback 到 ctypes 循环，可能掩盖该分支自身的
    错误（例如历史上缺失 import ctypes 的 NameError），因此这里绕过 fallback
    直接调用，并与 legacy per-frame 路径对比。
    """
    from tasks.aec.data.linear_aec_frontend import LinearAECFrontend, _c_batch_lib

    if not _SPEEX_AVAILABLE or _c_batch_lib is None:
        pytest.skip("libspeexdsp.so.1 or C batch wrapper not available")

    sr = 16000
    frame_size, filter_length = 256, 2048
    # 非整帧长度，同时覆盖 C wrapper 的 remainder 零填充分支
    n_samples = 16000 + 123
    t = np.arange(n_samples) / sr
    ref = np.sin(2 * np.pi * 500 * t + 100 * t * t).astype(np.float32)
    delay_samples = 512
    echo = np.zeros_like(ref)
    echo[delay_samples:] = ref[:-delay_samples] * 0.3
    noise = np.random.default_rng(2).standard_normal(n_samples).astype(np.float32)
    mic = (echo + noise * 0.05).astype(np.float32)

    cfg = {"backend": "speex", "frame_size": frame_size, "filter_length": filter_length}
    frontend = LinearAECFrontend(cfg, sample_rate=sr)
    backend = frontend._backend
    assert backend._use_c_batch, "C batch wrapper loaded but backend did not enable it"

    scale = 32767.0
    mic_i16 = (mic * scale).astype(np.int16)
    ref_i16 = (ref * scale).astype(np.int16)
    out_i16 = np.empty_like(mic_i16)

    st = backend._lib.speex_echo_state_init(frame_size, filter_length)
    assert st, "speex_echo_state_init returned NULL"
    try:
        sr_ctype = ctypes.c_int(sr)
        backend._lib.speex_echo_ctl(st, 24, ctypes.byref(sr_ctype))
        backend._process_c_batch(st, mic_i16, ref_i16, out_i16, n_samples)
    finally:
        backend._lib.speex_echo_state_destroy(st)

    err_direct = out_i16.astype(np.float32) * (1.0 / scale)
    err_legacy = _legacy_process(mic, ref, frame_size=frame_size,
                                 filter_length=filter_length, sample_rate=sr)
    np.testing.assert_allclose(
        err_direct, err_legacy,
        rtol=1e-5, atol=1e-6,
        err_msg="Direct _process_c_batch diverges from legacy init/destroy"
    )


def test_non_multiple_frame_size():
    """Ensure correct behaviour when sample length is not a multiple of frame_size."""
    if not _SPEEX_AVAILABLE:
        pytest.skip("libspeexdsp.so.1 not available")
    from tasks.aec.data.linear_aec_frontend import LinearAECFrontend

    sr = 16000
    n_samples = 16000 + 123
    t = np.arange(n_samples) / sr
    ref = np.sin(2 * np.pi * 400 * t).astype(np.float32)
    noise = np.random.default_rng(3).standard_normal(n_samples).astype(np.float32)
    mic = (ref * 0.2 + noise * 0.05).astype(np.float32)

    cfg = {"backend": "speex", "frame_size": 256, "filter_length": 2048}
    frontend = LinearAECFrontend(cfg, sample_rate=sr)

    err = frontend.process(mic, ref)
    assert len(err) == n_samples
    assert np.isfinite(err).all()

    err_legacy = _legacy_process(mic, ref, frame_size=256, filter_length=2048, sample_rate=sr)
    np.testing.assert_allclose(err, err_legacy, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    test_ctypes_optimised_consistency()
    test_c_batch_consistency()
    test_c_batch_direct_call()
    test_non_multiple_frame_size()
    print("All tests passed.")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _SPEEX_AVAILABLE, reason="libspeexdsp.so.1 not available"),
]
