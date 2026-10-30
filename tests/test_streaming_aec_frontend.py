"""Unit tests for StreamingAECFrontend."""
import pytest

import unittest

import numpy as np
import torch

from tasks.aec.inference.streaming_aec_frontend import StreamingAECFrontend
from onnx.models.streaming_dbgnet import StreamingDBGNetFrontendAdapter
from tasks.aec.inference.streaming_delay_estimator import (
    StreamingDelayEstimator,
    StreamingDelayEstimatorConfig,
    create_delay_estimator,
)


class _DummyAECModel(torch.nn.Module):
    """AEC model that returns the microphone spectrum unchanged."""

    def forward(self, inputs: dict) -> dict:
        mic = inputs["mic_spectra"]
        return {"nearend_estimate_spectra": mic.clone()}


class _CaptureStreamingModel(torch.nn.Module):
    """Capture per-frame spectra while preserving the streaming model contract."""

    CACHE_KEYS = ["state"]

    def __init__(self, n_bins: int):
        super().__init__()
        self.n_bins = n_bins
        self.frames = []

    def init_cache(self, batch, device, dtype):
        return {"state": torch.zeros(batch, self.n_bins, 1, 2, device=device, dtype=dtype)}

    def forward(self, mic_spec, far_spec, cache):
        self.frames.append(mic_spec.detach().cpu().clone())
        return mic_spec, cache


class _DummyStreamingModel(torch.nn.Module):
    """Stateful streaming model: out = mic + 累积状态（状态被重置时输出恒等于输入）。

    用于验证 frontend 的逐帧 cache 路径：状态必须跨帧保持、由 frontend 持有。
    """

    CACHE_KEYS = ["state"]

    def __init__(self, n_bins: int):
        super().__init__()
        self.n_bins = n_bins

    def init_cache(self, batch, device, dtype):
        return {
            "state": torch.zeros(batch, self.n_bins, 1, 2, device=device, dtype=dtype)
        }

    def forward(self, mic_spec, far_spec, cache):
        new_state = cache["state"] + mic_spec
        return new_state, {"state": new_state}


class _ThreeReturnDBGNetStub(torch.nn.Module):
    """提供 DBGNet 三返回 API 的轻量测试替身。"""

    CACHE_KEYS = ("state",)

    def __init__(self, n_bins: int):
        super().__init__()
        self.n_bins = n_bins
        self.n_fft = (n_bins - 1) * 2
        self.hop_length = self.n_fft // 2
        self.win_length = self.n_fft
        self.target_fs = 16000
        self.window = "hann"
        self.center = False
        self.use_state_detector = True
        self.state_nclass = 3
        self.network_lookahead_frames = 0
        self.algorithmic_delay_frames = 0
        self.algorithmic_delay = 0
        self.deployment_config = None
        self.forward_calls = 0

    def init_cache(self, batch, device, dtype):
        return {"state": torch.zeros(batch, 1, device=device, dtype=dtype)}

    def forward_streaming(self, mic_spec, far_spec, cache):
        self.forward_calls += 1
        updated = {"state": cache["state"] + 1}
        state_logits = torch.zeros(
            mic_spec.shape[0], 1, self.state_nclass,
            device=mic_spec.device, dtype=mic_spec.dtype,
        )
        return mic_spec, state_logits, updated


class _OfflineAlignModel(_DummyAECModel):
    """带 DeepVQE 风格 alignment 块的离线模型（unfold kernel 时间维 = delay）。"""

    def __init__(self, delay: int):
        super().__init__()
        self.align = torch.nn.Module()
        self.align.unfold = torch.nn.Sequential(
            torch.nn.ZeroPad2d([0, 0, delay - 1, 0]), torch.nn.Unfold((delay, 1))
        )


class _FaultyOfflineModel(torch.nn.Module):
    """返回指定非法离线输出，用于验证 frontend 的提交边界。"""

    def __init__(self, fault: str):
        super().__init__()
        self.fault = fault

    def forward(self, inputs: dict):
        output = inputs["mic_spectra"].clone()
        if self.fault == "missing_key":
            return {}
        if self.fault == "shape":
            return {"nearend_estimate_spectra": output[..., 0]}
        output.flatten()[0] = float(self.fault)
        return {"nearend_estimate_spectra": output}


class _FaultyStreamingModel(torch.nn.Module):
    """返回指定非法逐帧输出/cache，用于验证容器/OLA 候选提交与 reset 门禁。"""

    CACHE_KEYS = ("state",)

    def __init__(self, n_bins: int, fault: str, *, complex_cache: bool = False):
        super().__init__()
        self.n_bins = n_bins
        self.fault = fault
        self.complex_cache = complex_cache

    def init_cache(self, batch, device, dtype):
        cache_dtype = torch.complex64 if self.complex_cache else dtype
        return {
            "state": torch.zeros(
                batch, self.n_bins, 1, 2, device=device, dtype=cache_dtype
            )
        }

    def forward(self, mic_spec, far_spec, cache):
        output = mic_spec.clone()
        candidate = {"state": cache["state"] + 1}
        if self.fault == "output_shape":
            output = output[..., 0]
        elif self.fault in ("nan", "inf"):
            output.flatten()[0] = float(self.fault)
        elif self.fault == "cache_shape":
            candidate["state"] = candidate["state"][..., :1]
        elif self.fault == "cache_schema":
            candidate = {"other": candidate["state"]}
        elif self.fault in ("cache_nan", "cache_inf"):
            value = float(self.fault.removeprefix("cache_"))
            candidate["state"].flatten()[0] = value
        return output, candidate


class _InPlaceMutatingThenInvalidStreamingModel(torch.nn.Module):
    """首次调用原地污染共享 cache tensor 后返回非法输出，之后恢复合法输出。"""

    CACHE_KEYS = ("state",)

    def __init__(self, n_bins: int):
        super().__init__()
        self.n_bins = n_bins
        self.forward_calls = 0

    def init_cache(self, batch, device, dtype):
        return {
            "state": torch.zeros(
                batch, self.n_bins, 1, 2, device=device, dtype=dtype
            )
        }

    def forward(self, mic_spec, far_spec, cache):
        self.forward_calls += 1
        cache["state"].add_(1)
        if self.forward_calls == 1:
            return mic_spec[..., 0], cache
        return mic_spec, cache


class TestStreamingAECFrontend(unittest.TestCase):
    """Tests for the streaming AEC frontend."""

    def test_frontend_declares_mono_only_contract(self):
        metadata = StreamingAECFrontend.runtime_metadata()
        self.assertFalse(StreamingAECFrontend.supports_multichannel_input)
        self.assertFalse(metadata["supports_multichannel_input"])
        self.assertEqual(metadata["input_contract"]["mic_spectra"], "(B,F,T,2)")
        self.assertIn("no implicit reference-channel projection", metadata["input_contract"]["multichannel_policy"])

    def test_multichannel_microphone_frame_fails_fast_without_projection(self):
        frontend = self._frontend(hop=256)
        mic = np.zeros((2, 256), dtype=np.float32)
        ref = np.zeros(256, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, r"multichannel streaming \(C>1\) is unsupported"):
            frontend.push(mic, ref)

    def test_multichannel_microphone_chunk_fails_fast_without_projection(self):
        frontend = self._frontend(hop=256)
        mic = np.zeros((2, 512), dtype=np.float32)
        ref = np.zeros(512, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, r"multichannel streaming \(C>1\) is unsupported"):
            frontend.push_chunk(mic, ref)

    def _default_delay_est(self, sr: int = 16000):
        cfg = StreamingDelayEstimatorConfig(
            sample_rate=sr,
            buffer_duration_ms=500.0,
            max_residual_delay_ms=100.0,
            update_interval_ms=50.0,
            smooth_alpha=0.3,
        )
        return StreamingDelayEstimator(cfg)

    def _frontend(
        self,
        model: torch.nn.Module | None = None,
        delay_est: StreamingDelayEstimator | None = None,
        hop: int = 256,
        chunk_frames: int = 4,
        device: str = "cpu",
        center: bool = False,
    ):
        model = model if model is not None else _DummyAECModel()
        delay_est = delay_est if delay_est is not None else self._default_delay_est()
        return StreamingAECFrontend(
            model=model,
            delay_estimator=delay_est,
            n_fft=512,
            hop_length=hop,
            win_length=512,
            chunk_frames=chunk_frames,
            center=center,
            device=device,
        )

    def test_center_true_matches_torch_stft_first_frame_and_frame_count(self):
        """中心化流式帧与 torch.stft(reflect) 的边界和帧数一致。"""
        n_fft = 16
        hop = 8
        model = _CaptureStreamingModel(n_fft // 2 + 1)
        frontend = StreamingAECFrontend(
            model=model,
            delay_estimator=self._default_delay_est(sr=1600),
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            center=True,
        )
        signal = np.arange(4 * hop, dtype=np.float32)
        for i in range(0, signal.size, hop):
            frontend.push(signal[i : i + hop], np.zeros(hop, dtype=np.float32))
        frontend.flush()

        expected = torch.view_as_real(torch.stft(
            torch.from_numpy(signal),
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ))
        actual = torch.cat(model.frames, dim=2)[0]
        self.assertEqual(actual.shape[1], expected.shape[1])
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_center_true_queues_outputs_through_public_push_api(self):
        """centered 路径通过逐 hop 公共 API 保持输出顺序并清空 FIFO。"""
        n_fft = 16
        hop = 8
        model = _CaptureStreamingModel(n_fft // 2 + 1)
        frontend = StreamingAECFrontend(
            model=model,
            delay_estimator=self._default_delay_est(sr=1600),
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            center=True,
        )
        signal = np.linspace(-0.5, 0.5, 12 * hop, dtype=np.float32)
        ref = np.zeros_like(signal)
        outputs = []
        for i in range(0, signal.size, hop):
            out = frontend.push(signal[i : i + hop], ref[i : i + hop])
            if out is not None:
                outputs.append(out)
        outputs.append(frontend.flush())
        actual = np.concatenate(outputs)
        self.assertEqual(actual.size, signal.size)
        self.assertLessEqual(actual.size, len(model.frames) * hop)
        np.testing.assert_allclose(actual, signal, atol=1e-5, rtol=1e-5)
        self.assertEqual(len(frontend._center_output_sample_buf), 0)
        self.assertGreater(len(model.frames), 1)

        with self.assertRaisesRegex(RuntimeError, "call reset\(\) before pushing"):
            frontend.push(np.zeros(hop, dtype=np.float32), np.zeros(hop, dtype=np.float32))

        frontend.reset()
        model.frames.clear()
        for i in range(0, signal.size, hop):
            frontend.push(signal[i : i + hop], ref[i : i + hop])
        frontend.flush()
        self.assertEqual(len(frontend._center_output_sample_buf), 0)
        frontend.reset()
        self.assertEqual(frontend._center_output_sample_buf, [])

    def test_center_true_requires_lookahead_and_reset(self):
        """center=True 显式暴露 n_fft//2 lookahead，reset 后重新冷启动。"""
        hop = 8
        model = _CaptureStreamingModel(9)
        frontend = self._frontend(model=model, hop=hop, center=True)
        self.assertEqual(frontend.lookahead_samples, frontend.n_fft // 2)
        zeros = np.zeros(hop, dtype=np.float32)
        self.assertIsNone(frontend.push(zeros, zeros))
        self.assertEqual(len(model.frames), 0)
        for _ in range(40):
            frontend.push(zeros, zeros)
        self.assertGreaterEqual(len(model.frames), 1)
        frontend.reset()
        self.assertFalse(frontend.warmed_up)
        self.assertFalse(frontend._center_initialized)
        self.assertEqual(frontend._center_next_frame, 0)
        model.frames.clear()
        self.assertIsNone(frontend.push(zeros, zeros))
        self.assertEqual(len(model.frames), 0)

    def test_center_true_rejects_reflect_incompatible_short_input(self):
        """center=True + reflect 在 flush 时拒绝过短输入。"""
        frontend = self._frontend(center=True, hop=256)
        frame = np.zeros(frontend.hop_length, dtype=np.float32)
        frontend.push(frame, frame)
        with self.assertRaisesRegex(ValueError, "requires more than n_fft//2"):
            frontend.flush()

        frontend.reset()
        with self.assertRaisesRegex(ValueError, "requires more than n_fft//2"):
            frontend.flush()

    def test_basic_push_returns_none_during_warmup(self):
        """No output until the STFT/model warm-up period is over."""
        frontend = self._frontend(chunk_frames=4)
        hop = frontend.hop_length
        # First STFT frame needs two input hops (win_length == 2*hop). After that,
        # each push produces one STFT frame, so chunk_frames pushes return None
        # and the (chunk_frames+1)-th push returns the first output frame.
        for _ in range(frontend.chunk_frames):
            out = frontend.push(np.zeros(hop, dtype=np.float32), np.zeros(hop, dtype=np.float32))
            self.assertIsNone(out)
        out = frontend.push(np.zeros(hop, dtype=np.float32), np.zeros(hop, dtype=np.float32))
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (hop,))

    def test_push_chunk_length_matches_input(self):
        """push_chunk returns the same number of samples as the input chunk."""
        frontend = self._frontend(chunk_frames=4)
        sr = 16000
        hop = frontend.hop_length
        n = sr + (hop - sr % hop)  # make it a multiple of hop_length
        ref = np.random.randn(n).astype(np.float32) * 0.1
        mic = np.random.randn(n).astype(np.float32) * 0.1
        out = frontend.push_chunk(mic, ref)
        self.assertEqual(out.size, n)

    def test_push_chunk_accepts_both_empty_without_advancing_state(self):
        """mic/ref 同为空时返回空 float32，且不推进任何流状态。"""
        empty = np.empty(0, dtype=np.float64)
        for center in (False, True):
            with self.subTest(center=center):
                frontend = self._frontend(center=center)
                estimator = frontend.delay_estimator
                state_before = (
                    estimator._ref_buf.size,
                    estimator._mic_buf.size,
                    estimator._samples_since_update,
                    estimator._frame_count,
                    list(estimator._history),
                    list(frontend._mic_sample_buf),
                    list(frontend._ref_sample_buf),
                    frontend._spec_frames_seen,
                    frontend._center_initialized,
                    frontend._center_input_samples,
                    frontend._center_output_samples_emitted,
                )

                out = frontend.push_chunk(empty, empty)

                self.assertEqual(out.shape, (0,))
                self.assertEqual(out.dtype, np.float32)
                self.assertEqual(
                    (
                        estimator._ref_buf.size,
                        estimator._mic_buf.size,
                        estimator._samples_since_update,
                        estimator._frame_count,
                        list(estimator._history),
                        frontend._mic_sample_buf,
                        frontend._ref_sample_buf,
                        frontend._spec_frames_seen,
                        frontend._center_initialized,
                        frontend._center_input_samples,
                        frontend._center_output_samples_emitted,
                    ),
                    state_before,
                )

    def test_noncenter_rolling_chunk_history_is_bounded_for_long_stream(self):
        """non-centered rolling chunk 的频谱历史始终不超过 chunk_frames。"""
        frontend = self._frontend(chunk_frames=4, center=False)
        hop = frontend.hop_length
        frame = np.zeros(hop, dtype=np.float32)

        for _ in range(128):
            frontend.push(frame, frame)
            self.assertLessEqual(
                len(frontend._mic_spec_history), frontend.chunk_frames
            )
            self.assertLessEqual(
                len(frontend._ref_spec_history), frontend.chunk_frames
            )

        self.assertGreater(frontend._spec_frames_seen, frontend.chunk_frames)
        self.assertEqual(len(frontend._mic_spec_history), frontend.chunk_frames)
        self.assertEqual(len(frontend._ref_spec_history), frontend.chunk_frames)

    def test_noncenter_rolling_chunk_bounding_preserves_short_stream_output(self):
        """有界历史与旧版无界历史在短流上产生完全相同的输出。"""

        class _UnboundedHistory(list):
            def __delitem__(self, key):
                if isinstance(key, slice):
                    return
                super().__delitem__(key)

        bounded = self._frontend(chunk_frames=4, center=False)
        legacy = self._frontend(chunk_frames=4, center=False)
        legacy._mic_spec_history = _UnboundedHistory()
        legacy._ref_spec_history = _UnboundedHistory()
        hop = bounded.hop_length
        rng = np.random.default_rng(23)
        mic = rng.normal(0.0, 0.1, 12 * hop).astype(np.float32)
        ref = rng.normal(0.0, 0.1, 12 * hop).astype(np.float32)

        for i in range(0, mic.size, hop):
            bounded_out = bounded.push(mic[i : i + hop], ref[i : i + hop])
            legacy_out = legacy.push(mic[i : i + hop], ref[i : i + hop])
            self.assertEqual(bounded_out is None, legacy_out is None)
            if bounded_out is not None:
                np.testing.assert_array_equal(bounded_out, legacy_out)

        np.testing.assert_array_equal(bounded.flush(), legacy.flush())

    def test_hann_window_is_periodic_torch_definition(self):
        """分析/合成窗严格等同 torch.hann_window(periodic=True)。"""
        frontend = self._frontend()
        expected = torch.hann_window(512, periodic=True, dtype=torch.float32)
        torch.testing.assert_close(frontend._window.cpu(), expected)
        np.testing.assert_allclose(frontend._window_sq, expected.numpy().astype(np.float64) ** 2)

    def test_cola_perfect_reconstruction_with_identity_model(self):
        """With an identity model and silence, output is zero after warm-up."""
        frontend = self._frontend(chunk_frames=4)
        hop = frontend.hop_length
        n_frames = 64
        out_frames = []
        for _ in range(n_frames):
            out = frontend.push(np.zeros(hop, dtype=np.float32), np.zeros(hop, dtype=np.float32))
            out_frames.append(out if out is not None else np.zeros(hop, dtype=np.float32))
        out = np.concatenate(out_frames)
        # After warm-up all frames should be near zero.
        steady = out[4 * hop :]
        self.assertLess(np.max(np.abs(steady)), 1e-4)

    def test_identity_reconstruction_constant_signal(self):
        """Identity model reconstructs a constant signal (steady-state) accurately."""
        frontend = self._frontend(chunk_frames=4)
        hop = frontend.hop_length
        n_frames = 128
        val = 0.5
        out_frames = []
        for _ in range(n_frames):
            out = frontend.push(np.full(hop, val, dtype=np.float32), np.full(hop, val, dtype=np.float32))
            out_frames.append(out if out is not None else np.zeros(hop, dtype=np.float32))
        out = np.concatenate(out_frames)
        # Exclude the causal ramp-up at the beginning and any tail.
        steady = out[8 * hop : -hop]
        self.assertLess(np.max(np.abs(steady - val)), 1e-3)

    def test_reset_clears_estimator_and_prevents_cross_stream_leakage(self):
        """reset 后 estimator/frontend 与全新实例处理第二条流完全一致。"""
        frontend = self._frontend(chunk_frames=4)
        fresh = self._frontend(chunk_frames=4)
        hop = frontend.hop_length
        rng = np.random.default_rng(17)

        first_stream = rng.normal(0.0, 0.2, 10 * hop).astype(np.float32)
        for i in range(0, first_stream.size, hop):
            frame = first_stream[i : i + hop]
            frontend.push(frame, frame)
        self.assertTrue(frontend.warmed_up)
        self.assertGreater(frontend.delay_estimator._ref_buf.size, 0)
        self.assertGreater(frontend.delay_estimator._mic_buf.size, 0)
        self.assertGreater(frontend.delay_estimator._frame_count, 0)

        frontend.reset()
        estimator = frontend.delay_estimator
        self.assertFalse(frontend.warmed_up)
        self.assertEqual(len(frontend._mic_spec_history), 0)
        self.assertEqual(estimator._ref_buf.size, 0)
        self.assertEqual(estimator._mic_buf.size, 0)
        # reset 后估计值回到 coarse 初始值（_default_delay_est 用默认 coarse=50ms）。
        self.assertEqual(estimator.delay_samples, estimator._coarse_samples)
        self.assertEqual(estimator._samples_since_update, 0)
        self.assertEqual(estimator._frame_count, 0)
        self.assertEqual(estimator._history, [])
        self.assertIsNone(estimator._R_accum)

        second_stream = rng.normal(0.0, 0.2, 12 * hop).astype(np.float32)
        reset_outputs = []
        fresh_outputs = []
        for i in range(0, second_stream.size, hop):
            frame = second_stream[i : i + hop]
            reset_outputs.append(frontend.push(frame, frame))
            fresh_outputs.append(fresh.push(frame, frame))

        for reset_out, fresh_out in zip(reset_outputs, fresh_outputs):
            self.assertEqual(reset_out is None, fresh_out is None)
            if reset_out is not None:
                np.testing.assert_allclose(reset_out, fresh_out, atol=1e-6, rtol=1e-6)
        self.assertEqual(estimator.delay_samples, fresh.delay_estimator.delay_samples)
        self.assertEqual(estimator._history, fresh.delay_estimator._history)
        np.testing.assert_array_equal(
            estimator._ref_buf.read_newest(estimator._ref_buf.size),
            fresh.delay_estimator._ref_buf.read_newest(fresh.delay_estimator._ref_buf.size),
        )
        np.testing.assert_array_equal(
            estimator._mic_buf.read_newest(estimator._mic_buf.size),
            fresh.delay_estimator._mic_buf.read_newest(fresh.delay_estimator._mic_buf.size),
        )

    def test_nonfinite_pair_rejected_before_any_state_is_committed(self):
        """任一通道含 NaN/Inf 时，整对输入不得部分写入 estimator/frontend。"""
        for bad_channel in ("mic", "ref"):
            for bad_value in (np.nan, np.inf):
                with self.subTest(bad_channel=bad_channel, bad_value=bad_value):
                    frontend = self._frontend()
                    hop = frontend.hop_length
                    valid = np.linspace(-0.1, 0.1, hop, dtype=np.float32)
                    frontend.push(valid, valid)
                    estimator = frontend.delay_estimator
                    ref_before = estimator._ref_buf.read_newest(estimator._ref_buf.size)
                    mic_before = estimator._mic_buf.read_newest(estimator._mic_buf.size)
                    estimator_state_before = (
                        estimator.delay_samples,
                        estimator._samples_since_update,
                        estimator._frame_count,
                        list(estimator._history),
                    )
                    frontend_state_before = (
                        list(frontend._mic_sample_buf),
                        list(frontend._ref_sample_buf),
                        frontend._spec_frames_seen,
                    )
                    bad = valid.copy()
                    bad[hop // 2] = bad_value
                    mic_frame = bad if bad_channel == "mic" else valid
                    ref_frame = bad if bad_channel == "ref" else valid

                    with self.assertRaisesRegex(ValueError, "finite"):
                        frontend.push(mic_frame, ref_frame)

                    np.testing.assert_array_equal(
                        estimator._ref_buf.read_newest(estimator._ref_buf.size), ref_before
                    )
                    np.testing.assert_array_equal(
                        estimator._mic_buf.read_newest(estimator._mic_buf.size), mic_before
                    )
                    self.assertEqual(
                        (
                            estimator.delay_samples,
                            estimator._samples_since_update,
                            estimator._frame_count,
                            list(estimator._history),
                        ),
                        estimator_state_before,
                    )
                    self.assertEqual(
                        (
                            frontend._mic_sample_buf,
                            frontend._ref_sample_buf,
                            frontend._spec_frames_seen,
                        ),
                        frontend_state_before,
                    )

    def test_push_rejects_non_1d_pair_before_any_state_is_committed(self):
        """push 对 mic/ref 均拒绝可被展平的一维伪装二维输入，且保持状态不变。"""
        for bad_channel in ("mic", "ref"):
            with self.subTest(bad_channel=bad_channel):
                frontend = self._frontend()
                hop = frontend.hop_length
                valid = np.linspace(-0.1, 0.1, hop, dtype=np.float32)
                frontend.push(valid, valid)
                estimator = frontend.delay_estimator
                ref_before = estimator._ref_buf.read_newest(estimator._ref_buf.size)
                mic_before = estimator._mic_buf.read_newest(estimator._mic_buf.size)
                state_before = (
                    estimator.delay_samples,
                    estimator._samples_since_update,
                    estimator._frame_count,
                    list(estimator._history),
                    list(frontend._mic_sample_buf),
                    list(frontend._ref_sample_buf),
                    frontend._spec_frames_seen,
                )
                bad = valid.reshape(1, hop)
                mic_frame = bad if bad_channel == "mic" else valid
                ref_frame = bad if bad_channel == "ref" else valid

                with self.assertRaisesRegex(ValueError, "one-dimensional"):
                    frontend.push(mic_frame, ref_frame)

                np.testing.assert_array_equal(
                    estimator._ref_buf.read_newest(estimator._ref_buf.size), ref_before
                )
                np.testing.assert_array_equal(
                    estimator._mic_buf.read_newest(estimator._mic_buf.size), mic_before
                )
                self.assertEqual(
                    (
                        estimator.delay_samples,
                        estimator._samples_since_update,
                        estimator._frame_count,
                        list(estimator._history),
                        frontend._mic_sample_buf,
                        frontend._ref_sample_buf,
                        frontend._spec_frames_seen,
                    ),
                    state_before,
                )

    def test_push_chunk_prevalidates_entire_pair_before_processing(self):
        """push_chunk 在处理首帧前校验双方维度与全部 finite 样本。"""
        for bad_channel in ("mic", "ref"):
            for bad_kind in ("non_1d", "nonfinite"):
                with self.subTest(bad_channel=bad_channel, bad_kind=bad_kind):
                    frontend = self._frontend()
                    hop = frontend.hop_length
                    valid = np.linspace(-0.1, 0.1, 2 * hop, dtype=np.float32)
                    if bad_kind == "non_1d":
                        bad = valid.reshape(2, hop)
                        error = "one-dimensional"
                    else:
                        bad = valid.copy()
                        bad[hop + hop // 2] = np.nan
                        error = "finite"
                    mic_chunk = bad if bad_channel == "mic" else valid
                    ref_chunk = bad if bad_channel == "ref" else valid
                    estimator = frontend.delay_estimator

                    with self.assertRaisesRegex(ValueError, error):
                        frontend.push_chunk(mic_chunk, ref_chunk)

                    self.assertEqual(estimator._ref_buf.size, 0)
                    self.assertEqual(estimator._mic_buf.size, 0)
                    self.assertEqual(estimator._samples_since_update, 0)
                    self.assertEqual(estimator._frame_count, 0)
                    self.assertEqual(estimator._history, [])
                    self.assertEqual(frontend._mic_sample_buf, [])
                    self.assertEqual(frontend._ref_sample_buf, [])
                    self.assertEqual(frontend._spec_frames_seen, 0)

    def test_matched_filter_overflow_pair_is_rejected_before_either_side_advances(self):
        """mic 越过 matched 限幅时，合法 ref 与 frontend 状态也不得推进。"""
        with self.assertWarnsRegex(RuntimeWarning, "falling back to Python"):
            estimator = create_delay_estimator(
                StreamingDelayEstimatorConfig(
                    algorithm="matched_filter",
                    sample_rate=8000,
                )
            )
        frontend = self._frontend(delay_est=estimator)
        hop = frontend.hop_length
        ref = np.linspace(-0.5, 0.5, hop, dtype=np.float32)
        mic = ref.copy()
        mic[hop // 2] = 16.01
        backend = estimator._estimator

        with self.assertRaisesRegex(ValueError, "normalized audio range"):
            frontend.push(mic, ref)

        self.assertEqual(backend._ref_raw_buf.size, 0)
        self.assertEqual(backend._ref_accum.size, 0)
        self.assertEqual(backend._mic_accum.size, 0)
        self.assertEqual(backend._frame_count, 0)
        self.assertEqual(frontend._mic_sample_buf, [])
        self.assertEqual(frontend._ref_sample_buf, [])
        self.assertEqual(frontend._spec_frames_seen, 0)

    def test_matched_filter_overflow_chunk_is_preflighted_before_first_pair(self):
        """后续 hop 越界时，push_chunk 必须在首个 pair 前拒绝整 chunk。"""
        with self.assertWarnsRegex(RuntimeWarning, "falling back to Python"):
            estimator = create_delay_estimator(
                StreamingDelayEstimatorConfig(
                    algorithm="matched_filter",
                    sample_rate=8000,
                )
            )
        frontend = self._frontend(delay_est=estimator)
        hop = frontend.hop_length
        ref = np.linspace(-0.5, 0.5, 2 * hop, dtype=np.float32)
        mic = ref.copy()
        mic[hop + hop // 2] = 16.01
        backend = estimator._estimator

        with self.assertRaisesRegex(ValueError, "normalized audio range"):
            frontend.push_chunk(mic, ref)

        self.assertEqual(backend._ref_raw_buf.size, 0)
        self.assertEqual(backend._ref_accum.size, 0)
        self.assertEqual(backend._mic_accum.size, 0)
        self.assertEqual(backend._frame_count, 0)
        self.assertEqual(frontend._mic_sample_buf, [])
        self.assertEqual(frontend._ref_sample_buf, [])
        self.assertEqual(frontend._spec_frames_seen, 0)

    def test_center_push_chunk_flush_returns_real_tail_once_and_reuses_after_reset(self):
        """center push_chunk 的占位不吞尾帧，flush 幂等且 reset 后可复用。"""
        n_fft = 16
        hop = 8
        frontend = StreamingAECFrontend(
            model=_CaptureStreamingModel(n_fft // 2 + 1),
            delay_estimator=self._default_delay_est(sr=1600),
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            center=True,
        )
        first = np.linspace(-0.7, 0.7, 12 * hop, dtype=np.float32)
        second = np.linspace(0.3, -0.5, 12 * hop, dtype=np.float32)

        for stream_index, signal in enumerate((first, second)):
            if stream_index:
                frontend.reset()
            pushed = frontend.push_chunk(signal, np.zeros_like(signal))
            tail = frontend.flush()
            combined = np.concatenate([pushed, tail])
            lookahead = frontend.lookahead_samples
            self.assertEqual(pushed.size, signal.size)
            self.assertEqual(tail.size, lookahead)
            np.testing.assert_allclose(
                combined[lookahead : lookahead + signal.size],
                signal,
                atol=1e-5,
                rtol=1e-5,
            )
            self.assertEqual(frontend._center_output_samples_emitted, signal.size)
            self.assertEqual(frontend.flush().size, 0)

    def test_center_true_requires_equal_fft_and_window_lengths(self):
        """center=True 明确拒绝尚未实现的非等长窗配置。"""
        with self.assertRaisesRegex(ValueError, "win_length == n_fft"):
            StreamingAECFrontend(
                model=_DummyAECModel(),
                delay_estimator=self._default_delay_est(),
                n_fft=16,
                hop_length=8,
                win_length=8,
                center=True,
            )

    def test_invalid_hop_length_raises(self):
        """Constructor rejects invalid STFT parameters."""
        model = _DummyAECModel()
        delay_est = self._default_delay_est()
        with self.assertRaises(ValueError):
            StreamingAECFrontend(
                model=model,
                delay_estimator=delay_est,
                n_fft=256,
                hop_length=256,
                win_length=512,
            )
        with self.assertRaises(ValueError):
            StreamingAECFrontend(
                model=model,
                delay_estimator=delay_est,
                n_fft=512,
                hop_length=0,
                win_length=512,
            )

    def test_input_length_mismatch_raises(self):
        """push raises if input frames have the wrong length."""
        frontend = self._frontend()
        with self.assertRaises(ValueError):
            frontend.push(np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32))

    def test_aligned_ref_delays_echo_compensation(self):
        """Frontend delay estimator finds a synthetic delay and output length is correct."""
        sr = 16000
        hop = 256
        true_delay_samples = 1024
        rng = np.random.default_rng(0)
        n = sr + (hop - sr % hop)  # multiple of hop_length
        ref = rng.normal(0, 1.0, n).astype(np.float32)
        ref = np.convolve(ref, np.ones(8) / 8, mode="same").astype(np.float32)
        ref *= 0.5 / (np.std(ref) + 1e-12)
        echo = np.zeros_like(ref)
        echo[true_delay_samples:] = ref[:-true_delay_samples] * 0.3
        mic = echo + rng.normal(0, 0.05, ref.size).astype(np.float32)

        delay_est = self._default_delay_est(sr)
        frontend = self._frontend(delay_est=delay_est, hop=hop, chunk_frames=4)
        out = frontend.push_chunk(mic, ref)
        self.assertEqual(out.size, mic.size)
        # Delay estimate should be within half a hop of the true delay.
        est_delay_ms = frontend.delay_estimator.delay_ms
        true_delay_ms = true_delay_samples / sr * 1000.0
        hop_ms = hop / sr * 1000.0
        self.assertLess(abs(est_delay_ms - true_delay_ms), hop_ms / 2)

    def test_output_frame_index_oldest(self):
        """output_frame_index=0 uses the oldest frame from the model output chunk."""
        frontend = self._frontend(chunk_frames=4, hop=256)
        frontend.output_frame_index = 0
        hop = frontend.hop_length
        for _ in range(8):
            out = frontend.push(np.zeros(hop, dtype=np.float32), np.zeros(hop, dtype=np.float32))
            if out is not None:
                self.assertEqual(out.shape, (hop,))

    def test_streaming_model_warmup_is_one_window(self):
        """真流式模型：攒够第一个 STFT 窗（win=2hop）后即出帧，不等 chunk_frames。"""
        n_bins = 512 // 2 + 1
        frontend = self._frontend(model=_DummyStreamingModel(n_bins), chunk_frames=16)
        hop = frontend.hop_length
        self.assertIsNone(
            frontend.push(np.zeros(hop, dtype=np.float32), np.zeros(hop, dtype=np.float32))
        )
        out = frontend.push(
            np.full(hop, 0.1, dtype=np.float32), np.zeros(hop, dtype=np.float32)
        )
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (hop,))
        self.assertTrue(frontend.warmed_up)
        # 滚动 chunk 路径的帧历史不累积（状态在 cache 里）
        self.assertEqual(len(frontend._mic_spec_history), 0)

    def test_streaming_model_state_persists_across_frames(self):
        """cache 跨帧保持：累积状态模型的输出幅度逐帧增长（不逐帧重置）。"""
        n_bins = 512 // 2 + 1
        frontend = self._frontend(model=_DummyStreamingModel(n_bins), chunk_frames=16)
        hop = frontend.hop_length
        mags = []
        for _ in range(12):
            out = frontend.push(
                np.full(hop, 0.1, dtype=np.float32), np.zeros(hop, dtype=np.float32)
            )
            mags.append(None if out is None else float(np.max(np.abs(out))))
        self.assertIsNone(mags[0])
        steady = [m for m in mags[1:] if m is not None]
        self.assertGreater(len(steady), 4)
        # 状态累积 → 后期输出显著大于初期输出
        self.assertGreater(steady[-1], steady[0] * 2.0)

    def test_streaming_model_reset_reinits_cache(self):
        """reset() 重新初始化流式 cache：输出回到冷启动水平。"""
        n_bins = 512 // 2 + 1
        frontend = self._frontend(model=_DummyStreamingModel(n_bins), chunk_frames=16)
        hop = frontend.hop_length
        for _ in range(10):
            frontend.push(np.full(hop, 0.1, dtype=np.float32), np.zeros(hop, dtype=np.float32))
        frontend.reset()
        self.assertFalse(frontend.warmed_up)
        mags = []
        for _ in range(3):
            out = frontend.push(
                np.full(hop, 0.1, dtype=np.float32), np.zeros(hop, dtype=np.float32)
            )
            mags.append(None if out is None else float(np.max(np.abs(out))))
        self.assertIsNone(mags[0])
        self.assertIsNotNone(mags[1])

    def test_offline_model_rejects_invalid_output_before_ola_commit(self):
        """离线模型的 NaN/Inf/schema/shape 输出均不得污染 OLA。"""
        for fault in ("nan", "inf", "missing_key", "shape"):
            with self.subTest(fault=fault):
                frontend = self._frontend(
                    model=_FaultyOfflineModel(fault), chunk_frames=1
                )
                hop = frontend.hop_length
                zeros = np.zeros(hop, dtype=np.float32)
                frontend.push(zeros, zeros)
                overlap_before = frontend._overlap_buf.copy()
                ola_before = frontend._ola_buf.copy()
                valid_before = frontend._overlap_valid

                with self.assertRaisesRegex(RuntimeError, "model output"):
                    frontend.push(zeros, zeros)

                np.testing.assert_array_equal(frontend._overlap_buf, overlap_before)
                np.testing.assert_array_equal(frontend._ola_buf, ola_before)
                self.assertEqual(frontend._overlap_valid, valid_before)

    def test_streaming_model_rejects_invalid_output_or_cache_before_candidate_commit(self):
        """非原地故障不提交 cache 容器/OLA 候选，并要求 reset。"""
        faults = (
            "nan",
            "inf",
            "output_shape",
            "cache_nan",
            "cache_inf",
            "cache_shape",
            "cache_schema",
        )
        for fault in faults:
            with self.subTest(fault=fault):
                model = _FaultyStreamingModel(257, fault)
                frontend = self._frontend(model=model, chunk_frames=16)
                hop = frontend.hop_length
                zeros = np.zeros(hop, dtype=np.float32)
                frontend.push(zeros, zeros)
                cache_before = frontend._stream_cache
                cache_tensor_before = cache_before["state"].clone()
                overlap_before = frontend._overlap_buf.copy()
                ola_before = frontend._ola_buf.copy()

                with self.assertRaisesRegex(RuntimeError, r"call reset\(\)"):
                    frontend.push(zeros, zeros)

                self.assertIs(frontend._stream_cache, cache_before)
                torch.testing.assert_close(
                    frontend._stream_cache["state"], cache_tensor_before
                )
                np.testing.assert_array_equal(frontend._overlap_buf, overlap_before)
                np.testing.assert_array_equal(frontend._ola_buf, ola_before)
                with self.assertRaisesRegex(RuntimeError, r"call reset\(\)"):
                    frontend.push(zeros, zeros)
                with self.assertRaisesRegex(RuntimeError, r"call reset\(\)"):
                    frontend.flush()

                frontend.reset()
                self.assertFalse(frontend.warmed_up)

    def test_in_place_cache_pollution_poison_requires_reset_even_for_empty_chunk(self):
        """共享 tensor 无法事务回滚；poison 门禁拒绝空 chunk，reset 后恢复。"""
        model = _InPlaceMutatingThenInvalidStreamingModel(257)
        frontend = self._frontend(model=model, chunk_frames=16)
        hop = frontend.hop_length
        zeros = np.zeros(hop, dtype=np.float32)
        empty = np.empty(0, dtype=np.float32)

        frontend.push(zeros, zeros)
        committed_cache = frontend._stream_cache
        committed_storage = committed_cache["state"].untyped_storage().data_ptr()
        overlap_before = frontend._overlap_buf.copy()
        ola_before = frontend._ola_buf.copy()

        with self.assertRaisesRegex(RuntimeError, "invalid model output shape"):
            frontend.push(zeros, zeros)

        self.assertTrue(frontend._reset_required)
        self.assertIs(frontend._stream_cache, committed_cache)
        self.assertEqual(
            frontend._stream_cache["state"].untyped_storage().data_ptr(),
            committed_storage,
        )
        torch.testing.assert_close(
            frontend._stream_cache["state"],
            torch.ones_like(frontend._stream_cache["state"]),
        )
        np.testing.assert_array_equal(frontend._overlap_buf, overlap_before)
        np.testing.assert_array_equal(frontend._ola_buf, ola_before)
        self.assertEqual(model.forward_calls, 1)

        with self.assertRaisesRegex(RuntimeError, r"call reset\(\)"):
            frontend.push_chunk(empty, empty)
        self.assertEqual(model.forward_calls, 1)

        frontend.reset()
        self.assertFalse(frontend._reset_required)
        torch.testing.assert_close(
            frontend._stream_cache["state"],
            torch.zeros_like(frontend._stream_cache["state"]),
        )
        self.assertEqual(frontend.push_chunk(empty, empty).size, 0)
        self.assertIsNone(frontend.push(zeros, zeros))
        self.assertIsNotNone(frontend.push(zeros, zeros))
        self.assertEqual(model.forward_calls, 2)

    def test_streaming_complex_cache_nonfinite_is_rejected(self):
        """complex cache 的实部或虚部非 finite 时同样触发 reset 门禁。"""
        for fault in ("cache_nan", "cache_inf"):
            with self.subTest(fault=fault):
                frontend = self._frontend(
                    model=_FaultyStreamingModel(257, fault, complex_cache=True)
                )
                hop = frontend.hop_length
                zeros = np.zeros(hop, dtype=np.float32)
                frontend.push(zeros, zeros)
                cache_before = frontend._stream_cache["state"].clone()

                with self.assertRaisesRegex(RuntimeError, "stream cache.*finite"):
                    frontend.push(zeros, zeros)

                torch.testing.assert_close(
                    frontend._stream_cache["state"], cache_before
                )

    def test_dbgnet_frontend_adapter_uses_two_return_streaming_cache_path(self):
        """三返回 DBGNet 经 adapter 后应被 frontend 识别并推进 cache。"""
        inner = _ThreeReturnDBGNetStub(257)
        adapter = StreamingDBGNetFrontendAdapter(inner)
        frontend = self._frontend(model=adapter)
        zeros = np.zeros(frontend.hop_length, dtype=np.float32)

        self.assertTrue(frontend._streaming)
        self.assertIsNone(frontend.push(zeros, zeros))
        self.assertIsNotNone(frontend.push(zeros, zeros))
        self.assertEqual(inner.forward_calls, 1)
        torch.testing.assert_close(
            frontend._stream_cache["state"], torch.ones_like(frontend._stream_cache["state"])
        )

    def test_offline_alignment_model_warns_on_small_chunk(self):
        """chunk_frames < alignment 回看深度时发 RuntimeWarning（质量显著退化）。"""
        with self.assertWarns(RuntimeWarning):
            self._frontend(model=_OfflineAlignModel(delay=100), chunk_frames=4)

    def test_offline_alignment_model_no_warn_when_chunk_covers_delay(self):
        """chunk_frames >= alignment 回看深度时不告警。"""
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            self._frontend(model=_OfflineAlignModel(delay=4), chunk_frames=8)


if __name__ == "__main__":
    unittest.main()

pytestmark = [pytest.mark.unit]
