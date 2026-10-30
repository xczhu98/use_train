"""Mono-only streaming AEC frontend combining delay estimation, STFT, neural model and ISTFT.

This frontend is a mono streaming frontend: it expects an AEC model that consumes a dict
with mono ``mic_spectra`` and ``farend_spectra`` of shape ``(B, F, T, 2)`` and
returns a dict containing ``nearend_estimate_spectra`` of the same shape.
DBGNet, DeepVQE and similar spectrum-in/spectrum-out AEC models are
compatible.

Two model execution modes are supported:

- **Stateful streaming models (preferred)**: models exposing ``init_cache()``,
  ``CACHE_KEYS`` and a frame-wise ``forward(mic_spec, far_spec, cache)`` on
  ``(B, F, 1, 2)`` spectra — e.g. ``onnx.models.streaming_deepvqe.StreamingDeepVQE``.
  The frontend carries the returned cache across frames, so GRU state,
  alignment lookback and causal-conv history persist; warm-up reduces to the
  STFT window latency.
- **Offline models on rolling chunks (fallback)**: a dict-in/dict-out model is
  re-run on the last ``chunk_frames`` STFT frames for every output frame.
  Recurrent / alignment / convolution state restarts from zeros on every call,
  so models with long temporal context (e.g. offline DeepVQE with a 100-frame
  alignment block) degrade badly; a ``RuntimeWarning`` is emitted when
  ``chunk_frames`` is smaller than the detected alignment lookback.

Usage:
    from tasks.aec.inference import StreamingDelayEstimator, StreamingAECFrontend

    delay_est = StreamingDelayEstimator(...)
    frontend = StreamingAECFrontend(
        model=dbgnet,
        delay_estimator=delay_est,
        n_fft=512,
        hop_length=256,
        win_length=512,
        chunk_frames=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    for ref_frame, mic_frame in stream:           # each: (hop_length,) float32
        out_frame = frontend.process_pair(ref_frame, mic_frame)
        if out_frame is not None:
            play(out_frame)                        # (hop_length,) float32
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from tasks.aec.inference.streaming_delay_estimator import (
    StreamingDelayEstimator,
    StreamingDelayEstimatorConfig,
)
from core.inference.streaming import StreamState, _DelegatingSession


def _alignment_delay_frames(model: torch.nn.Module) -> Optional[int]:
    """Best-effort 检测模型内 alignment 块的回看深度（帧）；检测不到返回 None。

    针对 DeepVQE 系 ``_AlignBlock``：``align.unfold = Sequential(ZeroPad2d, Unfold)``，
    Unfold kernel 的时间维即 delay 候选数。
    """
    unfold = getattr(getattr(model, "align", None), "unfold", None)
    try:
        return int(unfold[1].kernel_size[0])
    except (TypeError, IndexError, AttributeError):
        return None


def _as_finite_1d_float32(samples: np.ndarray, name: str) -> np.ndarray:
    """校验公开音频输入，并在不改变形状的前提下转为 float32。"""
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1:
        if name in {"mic_frame", "mic_chunk"} and array.ndim >= 2:
            raise ValueError(
                f"{name} must be one-dimensional mono with shape (samples,), got {array.shape}; "
                "multichannel streaming (C>1) is unsupported and no implicit "
                "reference-channel projection is performed"
            )
        raise ValueError(
            f"{name} must be one-dimensional, got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite samples")
    return array


class AECStreamingSession(_DelegatingSession):
    """将 :class:`StreamingAECFrontend` 接入公共 streaming seam。"""

    def __init__(self, frontend: "StreamingAECFrontend") -> None:
        super().__init__(frontend, frozenset({"mic", "farend"}))

    def _push_impl(self, inputs):
        return self._delegate.push(inputs["mic"], inputs["farend"])

    def _flush_impl(self):
        return self._delegate.flush()


class AECStreamingAdapter:
    """创建 AEC 流式会话；任务 cache 不会泄漏到公共合同。"""

    input_roles = frozenset({"mic", "farend"})

    def __init__(self, frontend: "StreamingAECFrontend") -> None:
        if not isinstance(frontend, StreamingAECFrontend):
            raise TypeError("frontend must be a StreamingAECFrontend")
        self._frontend = frontend

    def open_session(self) -> AECStreamingSession:
        return AECStreamingSession(self._frontend)


def _copy_cache_containers(value):
    """复制 cache 容器作为候选提交；tensor storage 仍共享且可能被模型原地污染。"""
    if isinstance(value, dict):
        return {key: _copy_cache_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_cache_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_cache_containers(item) for item in value)
    return value


class StreamingAECFrontend:
    """One-frame-in / one-frame-out mono AEC frontend.

    This frontend is deliberately mono-only. It has no native microphone-axis
    streaming seam: public waveform inputs are ``(hop_length,)`` and the model
    receives ``mic_spectra``/``farend_spectra`` as ``(B, F, T, 2)``. Passing a
    ``(C, hop_length)`` microphone frame is rejected at the public boundary;
    the frontend never selects a reference microphone implicitly.

    The frontend maintains sample buffers for the STFT, a history of STFT frames
    for the neural model, and an overlap-add buffer for the ISTFT. With
    ``center=False`` frames are causal. With ``center=True``, frames follow
    ``torch.stft(..., center=True, pad_mode="reflect")`` and therefore expose
    ``n_fft // 2`` samples of lookahead.

    Args:
        model: AEC model — either a stateful streaming model exposing
            ``init_cache()`` / ``CACHE_KEYS`` and a frame-wise
            ``forward(mic_spec, far_spec, cache)`` on ``(B, F, 1, 2)`` spectra
            (preferred, e.g. ``StreamingDeepVQE``), or an offline model accepting
            ``{"mic_spectra": (B,F,T,2), "farend_spectra": (B,F,T,2)}`` (mono microphone) and
            returning ``{"nearend_estimate_spectra": (B,F,T,2)}`` (rolling-chunk
            fallback).
        delay_estimator: Configured :class:`StreamingDelayEstimator`.
        n_fft: FFT size.
        hop_length: STFT hop length (also the input/output frame size in samples).
        win_length: STFT window length.
        chunk_frames: Number of STFT frames fed to the model at once (rolling-chunk
            mode only; ignored for stateful streaming models).  Larger values give
            the model more temporal context at the cost of compute.
        output_frame_index: Which output frame to use from the model's chunk
            output (rolling-chunk mode only).  ``-1`` means the newest (default);
            ``0`` means the oldest.
        window: Window type; currently only ``"hann"`` is supported.
        center: Whether to use ``torch.stft``-compatible centered frames.  The
            default ``False`` preserves the legacy causal API; ``True`` adds
            ``n_fft // 2`` samples of explicit lookahead.  For centered streams,
            ``flush()`` ends the stream; call ``reset()`` before pushing another
            stream.  Reflect padding requires more than ``n_fft // 2`` input
            samples when flushing.
        device: torch device for model and STFT tensors.
        dtype: torch dtype for STFT tensors.

    Note:
        ``supports_multichannel_input`` is ``False``. A native multichannel
        streaming implementation would require ``mic_spectra`` shaped
        ``(B, C, F, 1, 2)`` and mono ``farend_spectra`` shaped
        ``(B, F, 1, 2)``; this class does not implement that contract.

        For perfect reconstruction the window must satisfy the constant
        overlap-add (COLA) condition for the chosen ``hop_length``.
        ``hann`` with 50% overlap (``hop_length == win_length // 2``) satisfies
        COLA and is the recommended configuration.

        Feeding an offline model with long temporal context (e.g. DeepVQE with a
        100-frame alignment block) through the rolling-chunk path restarts its
        GRU / alignment / conv state on every frame; use its stateful streaming
        wrapper instead.
    """

    supports_multichannel_input = False
    input_contract = {
        "mic": "mono waveform (hop_length,)",
        "farend": "mono waveform (hop_length,)",
        "mic_spectra": "(B,F,T,2)",
        "farend_spectra": "(B,F,T,2)",
        "output": "mono waveform (hop_length,)",
        "multichannel_policy": "reject; no implicit reference-channel projection",
    }

    @classmethod
    def runtime_metadata(cls) -> dict[str, object]:
        """返回部署侧可记录的、不可变语义快照。"""
        return {
            "frontend": cls.__name__,
            "supports_multichannel_input": cls.supports_multichannel_input,
            "input_contract": dict(cls.input_contract),
        }

    def __init__(
        self,
        model: torch.nn.Module,
        delay_estimator: StreamingDelayEstimator,
        n_fft: int,
        hop_length: int,
        win_length: int,
        chunk_frames: int = 16,
        output_frame_index: int = -1,
        window: str = "hann",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        center: bool = False,
    ):
        if hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")
        if win_length < hop_length:
            raise ValueError(
                f"win_length ({win_length}) must be >= hop_length ({hop_length})"
            )
        if n_fft < win_length:
            raise ValueError(f"n_fft ({n_fft}) must be >= win_length ({win_length})")
        if center and n_fft != win_length:
            raise ValueError(
                "center=True currently requires win_length == n_fft to match "
                "the AEC STFT/iSTFT contract; "
                f"got n_fft={n_fft}, win_length={win_length}"
            )
        if chunk_frames <= 0:
            raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")

        self.model = model
        self.delay_estimator = delay_estimator
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.center = bool(center)
        self.lookahead_samples = self.n_fft // 2 if self.center else 0
        self.chunk_frames = int(chunk_frames)
        self.output_frame_index = int(output_frame_index)
        if self.output_frame_index < 0:
            self.output_frame_index = self.chunk_frames + self.output_frame_index
        if not 0 <= self.output_frame_index < self.chunk_frames:
            raise ValueError(
                f"output_frame_index must be in [-chunk_frames, chunk_frames), "
                f"got {output_frame_index}"
            )

        self.device = torch.device(device)
        self.dtype = dtype

        # 真流式模型走 cache 路径，离线模型滚动重算。
        self._streaming = callable(getattr(model, "init_cache", None)) and isinstance(
            getattr(model, "CACHE_KEYS", None), (list, tuple)
        )
        self._stream_cache: Optional[dict] = None
        self._reset_required = False
        self._spec_frames_seen = 0
        if self._streaming:
            self._stream_cache = model.init_cache(1, self.device, self.dtype)
            self._validate_stream_cache(self._stream_cache)
        else:
            delay = _alignment_delay_frames(model)
            if delay is not None and self.chunk_frames < delay:
                warnings.warn(
                    f"chunk_frames={self.chunk_frames} < 模型 alignment 回看深度 "
                    f"{delay} 帧：滚动 chunk 送入离线模型会逐帧重置 GRU/alignment/"
                    "卷积历史，且 chunk 覆盖不满 alignment 候选窗，输出质量显著退化。"
                    "请改用带 init_cache/CACHE_KEYS 的真流式封装（如 StreamingDeepVQE），"
                    f"或把 chunk_frames 调到 >= {delay}。",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if window.lower() != "hann":
            raise ValueError(f"Only 'hann' window is currently supported, got {window}")
        # WOLA 分母必须使用分析/合成窗的逐点平方。
        self._window = torch.hann_window(
            self.win_length,
            periodic=True,
            device=self.device,
            dtype=self.dtype,
        )
        self._window_sq = self._window.detach().cpu().to(torch.float64).numpy() ** 2

        self._mic_sample_buf: list[float] = []
        self._ref_sample_buf: list[float] = []
        self._mic_spec_history: list[Tensor] = []
        self._ref_spec_history: list[Tensor] = []
        self._center_left_pad = self.lookahead_samples if self.center else 0
        self._center_flushed = False
        self._center_initialized = False
        self._center_input_samples = 0
        self._center_output_samples_emitted = 0
        self._center_mic_raw: list[float] = []
        self._center_ref_raw: list[float] = []
        self._center_buffer_start = 0
        self._center_next_frame = 0
        self._center_next_output_frame = 0
        self._center_first_synth_frame: int | None = None
        self._center_last_synth_frame = -1
        self._center_output_sample_buf: list[np.ndarray] = []
        self._center_ola_start = 0
        self._center_ola_samples: list[float] = []
        self._center_ola_norm: list[float] = []
        self._center_window_sq = self._window_sq if self.center else np.array([], dtype=np.float64)
        ola_size = self.n_fft if self.center else self.win_length
        self._overlap_buf = np.zeros(ola_size, dtype=np.float32)
        self._ola_buf = np.zeros(ola_size, dtype=np.float64)
        self._overlap_valid = 0

    @property
    def warmed_up(self) -> bool:
        """True after enough input frames have been processed to produce output."""
        if self._streaming:
            # 逐帧模式：攒够第一个 STFT 窗（win_length 样本）即有输出。
            return self._spec_frames_seen >= 1
        return len(self._mic_spec_history) >= self.chunk_frames

    def _preflight_delay_input(self, samples: np.ndarray, name: str) -> None:
        """执行 estimator-specific 无状态预检；原生 GCC estimator 无额外限制。"""
        preflight = getattr(self.delay_estimator, "preflight_input", None)
        if callable(preflight):
            preflight(samples, name)

    def _validate_stream_cache(self, cache, expected=None) -> None:
        """校验 cache schema，并递归检查 floating/complex tensor 有限性。"""
        if not isinstance(cache, dict):
            raise ValueError(
                f"invalid stream cache schema: expected dict, got {type(cache).__name__}"
            )
        keys = tuple(self.model.CACHE_KEYS)
        if len(keys) != len(set(keys)) or set(cache) != set(keys):
            raise ValueError(
                "invalid stream cache schema: keys must exactly match CACHE_KEYS; "
                f"expected={keys}, got={tuple(cache)}"
            )
        if expected is not None and not isinstance(expected, dict):
            raise ValueError("invalid committed stream cache schema")

        def validate_node(value, reference, path: str) -> None:
            if isinstance(value, Tensor):
                if isinstance(reference, Tensor):
                    if (
                        value.shape != reference.shape
                        or value.dtype != reference.dtype
                        or value.device != reference.device
                    ):
                        raise ValueError(
                            f"invalid stream cache schema at {path}: expected "
                            f"shape={tuple(reference.shape)}, dtype={reference.dtype}, "
                            f"device={reference.device}; got shape={tuple(value.shape)}, "
                            f"dtype={value.dtype}, device={value.device}"
                        )
                elif reference is not None:
                    raise ValueError(
                        f"invalid stream cache schema at {path}: expected "
                        f"{type(reference).__name__}, got Tensor"
                    )
                if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(
                    torch.isfinite(value).all().item()
                ):
                    raise ValueError(
                        f"invalid stream cache at {path}: floating/complex tensor "
                        "must contain only finite values"
                    )
                return
            if isinstance(value, dict):
                if reference is not None and not isinstance(reference, dict):
                    raise ValueError(f"invalid stream cache schema at {path}")
                if isinstance(reference, dict) and set(value) != set(reference):
                    raise ValueError(f"invalid stream cache schema at {path}: keys changed")
                for key, item in value.items():
                    child_reference = reference.get(key) if isinstance(reference, dict) else None
                    validate_node(item, child_reference, f"{path}.{key}")
                return
            if isinstance(value, (list, tuple)):
                if reference is not None and type(value) is not type(reference):
                    raise ValueError(f"invalid stream cache schema at {path}")
                if isinstance(reference, (list, tuple)) and len(value) != len(reference):
                    raise ValueError(f"invalid stream cache schema at {path}: length changed")
                for index, item in enumerate(value):
                    child_reference = reference[index] if isinstance(reference, (list, tuple)) else None
                    validate_node(item, child_reference, f"{path}[{index}]")
                return
            if reference is None:
                if value is not None:
                    raise ValueError(
                        f"invalid stream cache schema at {path}: opaque state must be "
                        "None or Tensor"
                    )
                return
            if type(value) is not type(reference):
                raise ValueError(f"invalid stream cache schema at {path}: type changed")

        for key in keys:
            reference = expected.get(key) if isinstance(expected, dict) else None
            validate_node(cache[key], reference, f"cache.{key}")

    def _validate_model_spectrum(self, spectrum, expected_shape: torch.Size) -> Tensor:
        """校验 pair-real 复谱的 tensor、shape、dtype 与有限性合同。"""
        if not isinstance(spectrum, Tensor):
            raise ValueError(
                f"invalid model output: expected Tensor, got {type(spectrum).__name__}"
            )
        if spectrum.shape != expected_shape:
            raise ValueError(
                "invalid model output shape: expected "
                f"{tuple(expected_shape)}, got {tuple(spectrum.shape)}"
            )
        if not torch.is_floating_point(spectrum):
            raise ValueError(
                f"invalid model output dtype: expected floating pair-real spectrum, got {spectrum.dtype}"
            )
        if not bool(torch.isfinite(spectrum).all().item()):
            raise ValueError("invalid model output: spectrum must contain only finite values")
        return spectrum

    def _run_streaming_model(
        self, mic_input: Tensor, ref_input: Tensor
    ) -> tuple[np.ndarray, dict]:
        """构造并校验 cache/OLA 候选提交；失败后必须 reset。

        这里只复制 cache 容器，不深 clone 潜在的大型 tensor。若模型原地修改
        cache tensor，其 storage 可能已污染已提交 cache，无法事务回滚。
        """
        committed_cache = self._stream_cache
        try:
            trial_cache = _copy_cache_containers(committed_cache)
            self.model.eval()
            with torch.inference_mode():
                result = self.model(mic_input, ref_input, trial_cache)
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise ValueError(
                    "invalid model output schema: streaming model must return "
                    "(spectrum, cache)"
                )
            spectrum, candidate_cache = result
            spectrum = self._validate_model_spectrum(spectrum, mic_input.shape)
            self._validate_stream_cache(candidate_cache, committed_cache)
            wave = self._istft_frame(spectrum[0, :, 0, :])
            if not np.all(np.isfinite(wave)):
                raise ValueError("invalid model output: iSTFT wave must be finite")
        except Exception as exc:
            self._reset_required = True
            raise RuntimeError(f"{exc}; call reset() before continuing") from exc
        return wave, candidate_cache

    def _run_offline_model(
        self, mic_input: Tensor, ref_input: Tensor, frame_index: int
    ) -> np.ndarray:
        """运行离线模型并在 OLA 前校验完整输出及选中帧。"""
        try:
            self.model.eval()
            with torch.inference_mode():
                result = self.model(
                    {"mic_spectra": mic_input, "farend_spectra": ref_input}
                )
            if not isinstance(result, dict) or "nearend_estimate_spectra" not in result:
                raise ValueError(
                    "invalid model output schema: missing nearend_estimate_spectra"
                )
            spectrum = self._validate_model_spectrum(
                result["nearend_estimate_spectra"], mic_input.shape
            )
            wave = self._istft_frame(spectrum[0, :, frame_index, :])
            if not np.all(np.isfinite(wave)):
                raise ValueError("invalid model output: iSTFT wave must be finite")
        except Exception as exc:
            self._reset_required = True
            raise RuntimeError(f"{exc}; call reset() before continuing") from exc
        return wave

    def reset(self) -> None:
        """Reset frontend, model cache, and delay-estimator stream state."""
        self.delay_estimator.reset()
        self._mic_sample_buf.clear()
        self._ref_sample_buf.clear()
        self._mic_spec_history.clear()
        self._ref_spec_history.clear()
        self._center_mic_raw.clear()
        self._center_ref_raw.clear()
        self._center_initialized = False
        self._center_flushed = False
        self._center_input_samples = 0
        self._center_output_samples_emitted = 0
        self._center_output_sample_buf.clear()
        self._center_ola_samples.clear()
        self._center_ola_norm.clear()
        self._center_buffer_start = 0
        self._center_next_frame = 0
        self._center_next_output_frame = 0
        self._center_first_synth_frame = None
        self._center_last_synth_frame = -1
        self._center_ola_start = 0
        self._overlap_buf.fill(0.0)
        self._ola_buf.fill(0.0)
        self._overlap_valid = 0
        self._spec_frames_seen = 0
        if self._streaming:
            self._stream_cache = self.model.init_cache(1, self.device, self.dtype)
            self._validate_stream_cache(self._stream_cache)
        self._reset_required = False

    def _center_reflect_pad(
        self, samples: list[float], pad: int, *, left: bool = True
    ) -> list[float]:
        """Return one side of PyTorch's ``reflect`` padding."""
        if pad == 0:
            return []
        if len(samples) <= pad:
            raise ValueError(
                "center=True requires more than n_fft//2 input samples for "
                'torch.stft(pad_mode="reflect")'
            )
        arr = np.asarray(samples, dtype=np.float32)
        if left:
            return np.pad(arr, (pad, 0), mode="reflect")[:pad].tolist()
        return np.pad(arr, (0, pad), mode="reflect")[-pad:].tolist()

    def _center_synth_frame(self, wave: np.ndarray, frame_index: int) -> None:
        """Add one validated centered iSTFT frame on an absolute padded timeline."""
        start = frame_index * self.hop_length - self._center_left_pad
        updates: list[tuple[int, float, float]] = []
        for offset, value in enumerate(wave):
            index = start + offset - self._center_ola_start
            if index < 0:
                continue
            old_sample = (
                self._center_ola_samples[index]
                if index < len(self._center_ola_samples)
                else 0.0
            )
            old_norm = (
                self._center_ola_norm[index]
                if index < len(self._center_ola_norm)
                else 0.0
            )
            sample = old_sample + float(value)
            norm = old_norm + float(self._center_window_sq[offset])
            if not np.isfinite(sample) or not np.isfinite(norm):
                self._reset_required = True
                raise RuntimeError(
                    "centered overlap-add became non-finite; "
                    "call reset() before continuing"
                )
            updates.append((index, sample, norm))

        if self._center_first_synth_frame is None:
            self._center_first_synth_frame = frame_index
            self._center_next_output_frame = frame_index
        self._center_last_synth_frame = max(self._center_last_synth_frame, frame_index)
        if updates and updates[-1][0] >= len(self._center_ola_samples):
            extend = updates[-1][0] + 1 - len(self._center_ola_samples)
            self._center_ola_samples.extend([0.0] * extend)
            self._center_ola_norm.extend([0.0] * extend)
        for index, sample, norm in updates:
            self._center_ola_samples[index] = sample
            self._center_ola_norm[index] = norm

    def _center_take_output(self, frame_index: int) -> np.ndarray:
        """Take one original-domain hop after its right overlap is available."""
        start = frame_index * self.hop_length - self._center_ola_start
        stop = start + self.hop_length
        if start < 0 or stop > len(self._center_ola_samples):
            return np.zeros(self.hop_length, dtype=np.float32)
        out = np.asarray(self._center_ola_samples[start:stop], dtype=np.float32).copy()
        norm = np.asarray(self._center_ola_norm[start:stop], dtype=np.float64)
        np.divide(out, norm, out=out, where=norm > 1e-6)
        return out

    def _center_discard_ola_before(self, sample_index: int) -> None:
        """Discard OLA samples that can no longer receive future frame overlap."""
        discard = max(0, sample_index - self._center_ola_start)
        if discard <= 0:
            return
        del self._center_ola_samples[:discard]
        del self._center_ola_norm[:discard]
        self._center_ola_start += discard

    def _center_queue_ready_outputs(self) -> None:
        """Queue synthesized hops whose right-hand overlap is available."""
        while self._center_next_output_frame + 1 <= self._center_last_synth_frame:
            frame_index = self._center_next_output_frame
            self._center_output_sample_buf.append(self._center_take_output(frame_index))
            self._center_next_output_frame += 1
            self._center_discard_ola_before(frame_index * self.hop_length + self.hop_length)

    def _center_run_frame(self, mic_spec: Tensor, ref_spec: Tensor, frame_index: int) -> None:
        """Run the existing model/cache contract for one centered STFT frame."""
        if not self._streaming:
            self._mic_spec_history.append(mic_spec)
            self._ref_spec_history.append(ref_spec)
        self._spec_frames_seen += 1
        if self._streaming:
            mic_input = mic_spec.unsqueeze(0).unsqueeze(2).contiguous()
            ref_input = ref_spec.unsqueeze(0).unsqueeze(2).contiguous()
            wave, candidate_cache = self._run_streaming_model(mic_input, ref_input)
            self._center_synth_frame(wave, frame_index)
            self._stream_cache = candidate_cache
        elif len(self._mic_spec_history) >= self.chunk_frames:
            mic_input = (
                torch.stack(self._mic_spec_history[-self.chunk_frames :], dim=1)
                .unsqueeze(0)
                .contiguous()
            )
            ref_input = (
                torch.stack(self._ref_spec_history[-self.chunk_frames :], dim=1)
                .unsqueeze(0)
                .contiguous()
            )
            wave = self._run_offline_model(
                mic_input, ref_input, self.output_frame_index
            )
            selected_frame = frame_index - (
                self.chunk_frames - 1 - self.output_frame_index
            )
            self._center_synth_frame(wave, selected_frame)
            if len(self._mic_spec_history) > self.chunk_frames:
                del self._mic_spec_history[:-self.chunk_frames]
                del self._ref_spec_history[:-self.chunk_frames]

    def _push_center(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> Optional[np.ndarray]:
        """Process centered frames; output is delayed until the right overlap exists."""
        self._center_input_samples += int(mic_frame.size)
        self._center_mic_raw.extend(mic_frame.tolist())
        self._center_ref_raw.extend(ref_frame.tolist())
        pad = self._center_left_pad
        if not self._center_initialized:
            if len(self._center_mic_raw) <= pad:
                return None
            mic_left = self._center_reflect_pad(self._center_mic_raw, pad)
            ref_left = self._center_reflect_pad(self._center_ref_raw, pad)
            self._mic_sample_buf = mic_left + self._center_mic_raw
            self._ref_sample_buf = ref_left + self._center_ref_raw
            self._center_buffer_start = 0
            self._center_initialized = True
        else:
            self._mic_sample_buf.extend(mic_frame.tolist())
            self._ref_sample_buf.extend(ref_frame.tolist())

        # reflect padding 仅保留最后 pad+1 个原始样本。
        if len(self._center_mic_raw) > pad + 1:
            del self._center_mic_raw[: -(pad + 1)]
            del self._center_ref_raw[: -(pad + 1)]

        while (
            self._center_buffer_start + len(self._mic_sample_buf)
            >= self._center_next_frame * self.hop_length + self.n_fft
        ):
            start = self._center_next_frame * self.hop_length - self._center_buffer_start
            mic_win = torch.tensor(
                self._mic_sample_buf[start : start + self.n_fft],
                dtype=self.dtype,
                device=self.device,
            )
            ref_win = torch.tensor(
                self._ref_sample_buf[start : start + self.n_fft],
                dtype=self.dtype,
                device=self.device,
            )
            self._center_run_frame(
                self._stft_frame(mic_win),
                self._stft_frame(ref_win),
                self._center_next_frame,
            )
            self._center_next_frame += 1
            self._center_queue_ready_outputs()
            discard = self._center_next_frame * self.hop_length - self._center_buffer_start
            if discard > 0:
                del self._mic_sample_buf[:discard]
                del self._ref_sample_buf[:discard]
                self._center_buffer_start += discard
        if self._center_output_sample_buf:
            out = self._center_output_sample_buf.pop(0)
            self._center_output_samples_emitted += out.size
            return out
        return None

    def push(
        self,
        mic_frame: np.ndarray,
        ref_frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Push one input frame and return one enhanced frame if available.

        Input frames must have length ``hop_length``.  During the initial
        STFT/model warm-up period ``None`` is returned.  After warm-up, every
        call returns a frame of length ``hop_length``.  Warm-up is one STFT
        window for stateful streaming models, or ``chunk_frames`` STFT frames
        for the rolling-chunk path.

        Args:
            mic_frame: Microphone samples, shape ``(hop_length,)``.
            ref_frame: Reference (farend) samples, shape ``(hop_length,)``.

        Returns:
            Enhanced samples of shape ``(hop_length,)`` or ``None`` during warm-up.

        Raises:
            ValueError: If either frame is not one-dimensional, has the wrong
                length, contains NaN/Inf, or violates the selected estimator's
                amplitude contract.  The complete mic/ref pair is validated before
                either side is submitted to the delay estimator, so rejection does
                not partially advance stream state.
            RuntimeError: If model output/cache schema, shape or finite checks fail.
                Cache containers and OLA updates are only candidate commits, but cache
                tensor storage is intentionally not cloned and may already be polluted
                by an in-place model mutation. Call ``reset()`` before continuing.
        """
        mic_frame = _as_finite_1d_float32(mic_frame, "mic_frame")
        ref_frame = _as_finite_1d_float32(ref_frame, "ref_frame")
        if self._reset_required:
            raise RuntimeError(
                "A previous model/cache/OLA failure poisoned the stream; "
                "call reset() before continuing"
            )
        if self.center and self._center_flushed:
            raise RuntimeError(
                "StreamingAECFrontend.flush() already completed; "
                "call reset() before pushing new audio"
            )
        if mic_frame.size != self.hop_length or ref_frame.size != self.hop_length:
            raise ValueError(
                f"Input frames must have length {self.hop_length}, got "
                f"mic={mic_frame.size}, ref={ref_frame.size}"
            )

        # 完整 pair 预检后再提交，避免部分推进 stream state。
        self._preflight_delay_input(ref_frame, "reference")
        self._preflight_delay_input(mic_frame, "microphone")
        self.delay_estimator.push_ref(ref_frame)
        self.delay_estimator.push_mic(mic_frame)
        aligned_ref = self.delay_estimator.get_aligned_ref_frame(self.hop_length)

        if self.center:
            return self._push_center(mic_frame, aligned_ref)

        # 2. Accumulate time-domain samples.
        self._mic_sample_buf.extend(mic_frame.tolist())
        self._ref_sample_buf.extend(aligned_ref.tolist())

        # 3. Compute STFT frames while enough samples are available.
        produced_output = False
        while len(self._mic_sample_buf) >= self.win_length:
            mic_win = torch.tensor(
                self._mic_sample_buf[: self.win_length],
                dtype=self.dtype,
                device=self.device,
            )
            ref_win = torch.tensor(
                self._ref_sample_buf[: self.win_length],
                dtype=self.dtype,
                device=self.device,
            )

            mic_spec = self._stft_frame(mic_win)  # (F, 2)
            ref_spec = self._stft_frame(ref_win)

            # 帧历史仅用于滚动 chunk 路径。
            if not self._streaming:
                self._mic_spec_history.append(mic_spec)
                self._ref_spec_history.append(ref_spec)
                if len(self._mic_spec_history) > self.chunk_frames:
                    del self._mic_spec_history[:-self.chunk_frames]
                    del self._ref_spec_history[:-self.chunk_frames]
            self._spec_frames_seen += 1

            # Slide sample buffers by one hop.
            self._mic_sample_buf = self._mic_sample_buf[self.hop_length :]
            self._ref_sample_buf = self._ref_sample_buf[self.hop_length :]

            # 校验候选 cache/输出后再提交；cache tensor 若被模型原地污染必须 reset。
            if self._streaming:
                mic_input = mic_spec.unsqueeze(0).unsqueeze(2).contiguous()
                ref_input = ref_spec.unsqueeze(0).unsqueeze(2).contiguous()
                wave, candidate_cache = self._run_streaming_model(
                    mic_input, ref_input
                )
                self._overlap_add(wave)
                self._stream_cache = candidate_cache
                produced_output = True
            elif len(self._mic_spec_history) >= self.chunk_frames:
                mic_input = (
                    torch.stack(self._mic_spec_history[-self.chunk_frames :], dim=1)
                    .unsqueeze(0)
                    .contiguous()
                )
                ref_input = (
                    torch.stack(self._ref_spec_history[-self.chunk_frames :], dim=1)
                    .unsqueeze(0)
                    .contiguous()
                )
                wave = self._run_offline_model(
                    mic_input, ref_input, self.output_frame_index
                )
                self._overlap_add(wave)
                produced_output = True

        if not produced_output:
            return None

        # 5. Return one hop of overlap-added output if available.
        if self._overlap_valid >= self.hop_length:
            ola = self._ola_buf[: self.hop_length]
            out = self._overlap_buf[: self.hop_length].copy()
            # Normalize by the COLA sum to undo window overlap attenuation.
            np.divide(out, ola, out=out, where=ola > 1e-6)
            self._overlap_buf[: -self.hop_length] = self._overlap_buf[self.hop_length :]
            self._overlap_buf[-self.hop_length :] = 0.0
            self._ola_buf[: -self.hop_length] = self._ola_buf[self.hop_length :]
            self._ola_buf[-self.hop_length :] = 0.0
            self._overlap_valid -= self.hop_length
            return out
        return None

    def process_pair(
        self,
        ref_frame: np.ndarray,
        mic_frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Push one ref/mic frame pair and return one enhanced frame if available.

        This is the recommended entry point for streaming inference because it
        guarantees the correct push order (reference before microphone) for the
        internal delay estimator, avoiding the one-hop underestimation that can
        occur when ``push_mic`` is called before ``push_ref``.

        Args:
            ref_frame: Reference (farend) samples, shape ``(hop_length,)``.
            mic_frame: Microphone samples, shape ``(hop_length,)``.

        Returns:
            Enhanced samples of shape ``(hop_length,)`` or ``None`` during warm-up.
        """
        return self.push(mic_frame, ref_frame)

    def push_chunk(
        self,
        mic_chunk: np.ndarray,
        ref_chunk: np.ndarray,
    ) -> np.ndarray:
        """Convenience method: process a chunk of audio frame-by-frame.

        Args:
            mic_chunk: Microphone samples; length must be a multiple of
                ``hop_length``.
            ref_chunk: Reference samples; same length as ``mic_chunk``.

        Returns:
            Enhanced samples with the same length as the input chunk.  If both
            chunks are empty, returns an empty ``float32`` array without advancing
            stream state.  Leading ``None`` frames are represented by zero
            placeholders.  With ``center=True``, those placeholders expose the
            lookahead but do not count as synthesized output: ``flush()`` returns
            the delayed real tail, and
            ``concatenate([push_chunk(...), flush()])[lookahead_samples:]`` contains
            exactly one output sample for each input sample.

        Raises:
            ValueError: If either chunk is not one-dimensional, chunk lengths
                differ or are not a multiple of ``hop_length``, or either complete
                chunk contains NaN/Inf.  Both chunks are fully validated before any
                frame is processed, so rejection does not advance stream state.
            RuntimeError: If a previous model/cache/OLA failure poisoned the stream;
                this also rejects an empty chunk until ``reset()`` is called.
        """
        mic_chunk = _as_finite_1d_float32(mic_chunk, "mic_chunk")
        ref_chunk = _as_finite_1d_float32(ref_chunk, "ref_chunk")
        if self._reset_required:
            raise RuntimeError(
                "A previous model/cache/OLA failure poisoned the stream; "
                "call reset() before continuing"
            )
        if mic_chunk.size != ref_chunk.size:
            raise ValueError(
                f"mic_chunk and ref_chunk must have the same length, got "
                f"{mic_chunk.size} vs {ref_chunk.size}"
            )
        if mic_chunk.size % self.hop_length != 0:
            raise ValueError(
                f"mic_chunk length ({mic_chunk.size}) must be a multiple of "
                f"hop_length ({self.hop_length})"
            )
        if mic_chunk.size == 0:
            return np.empty(0, dtype=np.float32)

        # 预检完整 chunk，避免首个 pair 提前提交。
        self._preflight_delay_input(ref_chunk, "reference")
        self._preflight_delay_input(mic_chunk, "microphone")

        out_frames: list[np.ndarray] = []
        n = mic_chunk.size
        for i in range(0, n, self.hop_length):
            mic_frame = mic_chunk[i : i + self.hop_length]
            ref_frame = ref_chunk[i : i + self.hop_length]
            out = self.push(mic_frame, ref_frame)
            if out is None:
                # Centered warm-up zeros are lookahead placeholders, not emitted
                # synthesized samples. flush() must still return the real tail.
                out = np.zeros(self.hop_length, dtype=np.float32)
            out_frames.append(out)
        return np.concatenate(out_frames)

    def flush(self) -> np.ndarray:
        """End the stream and return buffered output.

        For ``center=True``, this applies the right reflect padding and returns
        the real samples delayed by centered lookahead.  Placeholder zeros from
        ``push_chunk()`` are not included in the emitted-sample count.  A second
        call returns an empty array; call ``reset()`` before starting a new stream.
        Model output/cache validation failures also require ``reset()``. Cache
        containers and OLA updates are candidate commits, but potentially large
        cache tensors are not cloned; an ill-behaved model may have polluted shared
        tensor storage in-place, which cannot be transactionally rolled back.
        """
        if self._reset_required:
            raise RuntimeError(
                "A previous model/cache/OLA failure poisoned the stream; "
                "call reset() before continuing"
            )
        if self.center:
            if self._center_flushed:
                return np.empty(0, dtype=np.float32)
            if self._center_input_samples <= self._center_left_pad:
                self._center_flushed = True
                raise ValueError(
                    "center=True with pad_mode='reflect' requires more than "
                    f"n_fft//2 samples before flush; got {self._center_input_samples}, "
                    f"required > {self._center_left_pad}"
                )
            pad = self._center_left_pad
            mic_right = self._center_reflect_pad(self._center_mic_raw, pad, left=False)
            ref_right = self._center_reflect_pad(self._center_ref_raw, pad, left=False)
            self._mic_sample_buf.extend(mic_right)
            self._ref_sample_buf.extend(ref_right)
            while (
                self._center_buffer_start + len(self._mic_sample_buf)
                >= self._center_next_frame * self.hop_length + self.n_fft
            ):
                start = self._center_next_frame * self.hop_length - self._center_buffer_start
                mic_win = torch.tensor(
                    self._mic_sample_buf[start : start + self.n_fft],
                    dtype=self.dtype,
                    device=self.device,
                )
                ref_win = torch.tensor(
                    self._ref_sample_buf[start : start + self.n_fft],
                    dtype=self.dtype,
                    device=self.device,
                )
                self._center_run_frame(
                    self._stft_frame(mic_win),
                    self._stft_frame(ref_win),
                    self._center_next_frame,
                )
                self._center_next_frame += 1
                self._center_queue_ready_outputs()
                discard = self._center_next_frame * self.hop_length - self._center_buffer_start
                if discard > 0:
                    del self._mic_sample_buf[:discard]
                    del self._ref_sample_buf[:discard]
                    self._center_buffer_start += discard
            outputs = self._center_output_sample_buf
            self._center_output_sample_buf = []
            result = (
                np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
            )
            remaining = max(0, self._center_input_samples - self._center_output_samples_emitted)
            result = result[:remaining]
            self._center_output_samples_emitted += result.size
            self._center_flushed = True
            return result

        ola = self._ola_buf[: self._overlap_valid]
        out = self._overlap_buf[: self._overlap_valid].copy()
        np.divide(out, ola, out=out, where=ola > 1e-6)
        self._overlap_buf.fill(0.0)
        self._ola_buf.fill(0.0)
        self._overlap_valid = 0
        return out

    def _stft_frame(self, x: Tensor) -> Tensor:
        """Compute one causal STFT frame from ``win_length`` time samples."""
        padded = torch.zeros(self.n_fft, dtype=self.dtype, device=x.device)
        padded[: self.win_length] = x * self._window
        spec = torch.fft.rfft(padded)  # (F,)
        return torch.stack([spec.real, spec.imag], dim=-1)  # (F, 2)

    def _istft_frame(self, spec: Tensor) -> np.ndarray:
        """Invert one STFT frame to ``win_length`` time-domain samples."""
        real = spec[:, 0]
        imag = spec[:, 1]
        c = torch.complex(real, imag)
        full = torch.zeros(self.n_fft, dtype=c.dtype, device=c.device)
        full[: c.shape[0]] = c
        wave = torch.fft.irfft(full, n=self.n_fft)  # (n_fft,)
        wave = wave[: self.win_length] * self._window
        return wave.cpu().to(torch.float32).numpy()

    def _overlap_add(self, wave: np.ndarray) -> None:
        """校验并提交一个有限的 overlap-add 候选更新。"""
        if wave.size != self.win_length:
            raise ValueError(
                f"Expected wave length {self.win_length}, got {wave.size}"
            )
        overlap = self._overlap_buf + wave
        ola = self._ola_buf + self._window_sq
        if not np.all(np.isfinite(overlap)) or not np.all(np.isfinite(ola)):
            self._reset_required = True
            raise RuntimeError(
                "overlap-add became non-finite; call reset() before continuing"
            )
        self._overlap_buf[:] = overlap
        self._ola_buf[:] = ola
        self._overlap_valid = self.win_length


if __name__ == "__main__":
    """Self-test with DBGNet on synthetic data."""
    from tasks.aec.models.discriminative.dbgnet import DBGNet

    sr = 16000
    duration_s = 2.0
    true_delay_ms = 87.5
    true_delay_samples = int(round(true_delay_ms * sr / 1000.0))

    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1.0, int(duration_s * sr)).astype(np.float32)
    ref = np.convolve(ref, np.ones(8) / 8, mode="same").astype(np.float32)
    ref *= 0.5 / (np.std(ref) + 1e-12)

    echo = np.zeros_like(ref)
    echo[true_delay_samples:] = ref[:-true_delay_samples] * 0.3
    mic = echo + rng.normal(0, 0.05, ref.size).astype(np.float32)

    n_fft = 512
    hop = 256
    win = 512

    model = DBGNet(use_state_detector=False).eval().cpu()
    delay_est = StreamingDelayEstimator(
        StreamingDelayEstimatorConfig(
            sample_rate=sr,
            buffer_duration_ms=800.0,
            coarse_delay_ms=50.0,
            max_residual_delay_ms=50.0,
            update_interval_ms=100.0,
            smooth_alpha=0.15,
            peak_threshold=0.05,
        )
    )
    frontend = StreamingAECFrontend(
        model=model,
        delay_estimator=delay_est,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        chunk_frames=16,
        device="cpu",
    )

    out = frontend.push_chunk(mic, ref)
    out = np.concatenate([out, frontend.flush()])

    print(f"Input length : {mic.size} samples")
    print(f"Output length: {out.size} samples")
    print(f"Delay estimate: {frontend.delay_estimator.delay_ms:.2f} ms")
    print(f"Model forward calls: {len(frontend._mic_spec_history)}")

    # Sanity: output should be roughly the same length as input (within one frame).
    assert abs(out.size - mic.size) < 2 * hop
    print("Self-test passed.")
