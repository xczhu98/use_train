"""TSE causal 模型的 waveform streaming 前端。

核心模型继续保持 dict 谱进谱出；本模块在 runtime 边界负责与训练一致的
``center=True`` reflect STFT、逐帧模型调用、WOLA 合成和 enrollment 会话绑定。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np
import torch

from core.inference.streaming import _DelegatingSession


def _finite_mono(samples: object, name: str) -> np.ndarray:
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    value = np.asarray(samples, dtype=np.float32)
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite samples")
    return np.ascontiguousarray(value)


class TSEStreamingSession(_DelegatingSession):
    """将 TSE waveform session 接入公共 streaming seam。"""

    def __init__(self, session: "TSEStreamingWaveformSession") -> None:
        super().__init__(session, frozenset({"mixture"}))

    def _push_impl(self, inputs):
        return self._delegate.push(inputs["mixture"])

    def _flush_impl(self):
        return self._delegate.flush()


class TSEStreamingAdapter:
    """创建绑定 enrollment 的 TSE 流式会话。"""

    input_roles = frozenset({"mixture"})

    def __init__(
        self,
        model: torch.nn.Module,
        enrollment_waveform: torch.Tensor,
        *,
        spk_embedding: torch.Tensor | None = None,
    ) -> None:
        self._model = model
        self._enrollment_waveform = enrollment_waveform
        self._spk_embedding = spk_embedding

    def open_session(self) -> TSEStreamingSession:
        return TSEStreamingSession(
            TSEStreamingWaveformSession(
                self._model,
                self._enrollment_waveform,
                spk_embedding=self._spk_embedding,
            )
        )


class CenteredStreamingSTFTOLA:
    """有界缓存的 centered STFT/WOLA 驱动器。

    ``process_frame`` 接收并返回 ``[1, F, 1, 2]`` pair-real 复谱。普通
    ``push()`` 只返回已经获得全部右侧重叠贡献的样本；``flush()`` 添加与
    ``torch.stft(..., center=True, pad_mode='reflect')`` 一致的右 padding，并
    返回剩余尾部。
    """

    def __init__(
        self,
        process_frame: Callable[[torch.Tensor], torch.Tensor],
        *,
        n_fft: int,
        hop_length: int,
        win_length: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        for name, value in (
            ("n_fft", n_fft),
            ("hop_length", hop_length),
            ("win_length", win_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if win_length != n_fft:
            raise ValueError("centered TSE streaming requires win_length == n_fft")
        if hop_length > win_length:
            raise ValueError("hop_length must not exceed win_length")
        if dtype != torch.float32:
            raise ValueError("TSE waveform streaming currently requires torch.float32")
        if not callable(process_frame):
            raise TypeError("process_frame must be callable")

        self.process_frame = process_frame
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.device = torch.device(device)
        self.dtype = dtype
        self.lookahead_samples = n_fft // 2
        self.startup_input_samples = max(
            self.lookahead_samples + 1,
            self.lookahead_samples
            + (self.lookahead_samples // self.hop_length) * self.hop_length,
        )
        self._window = torch.hann_window(
            win_length, periodic=True, device=self.device, dtype=dtype
        )
        self._window_np = self._window.detach().cpu().numpy().astype(np.float64)
        self._window_sq = np.square(self._window_np)
        self.reset()

    def reset(self) -> None:
        self._input_samples = 0
        self._tail: list[float] = []
        self._prebuffer: list[float] = []
        self._analysis: list[float] = []
        self._analysis_start = 0
        self._initialized = False
        self._next_frame = 0
        self._ola: list[float] = []
        self._ola_norm: list[float] = []
        self._ola_start = 0
        self._emitted = 0
        self._flushed = False
        self._poisoned = False

    @property
    def input_samples(self) -> int:
        return self._input_samples

    @property
    def emitted_samples(self) -> int:
        return self._emitted

    def _append_input(self, samples: np.ndarray) -> None:
        values = samples.tolist()
        self._input_samples += len(values)
        self._tail.extend(values)
        keep = self.lookahead_samples + 1
        if len(self._tail) > keep:
            del self._tail[:-keep]
        if not self._initialized:
            self._prebuffer.extend(values)
            if len(self._prebuffer) <= self.lookahead_samples:
                return
            source = np.asarray(self._prebuffer, dtype=np.float32)
            left = np.pad(
                source, (self.lookahead_samples, 0), mode="reflect"
            )[: self.lookahead_samples]
            self._analysis = left.tolist() + self._prebuffer
            self._prebuffer = []
            self._initialized = True
            return
        self._analysis.extend(values)

    def _stft_frame(self, values: list[float]) -> torch.Tensor:
        waveform = torch.tensor(values, device=self.device, dtype=self.dtype)
        spectrum = torch.fft.rfft(waveform * self._window, n=self.n_fft)
        return torch.view_as_real(spectrum).unsqueeze(0).unsqueeze(2)

    def _synthesis_frame(self, spectrum: torch.Tensor) -> np.ndarray:
        if not isinstance(spectrum, torch.Tensor):
            raise TypeError("process_frame must return a torch.Tensor")
        expected = (1, self.n_fft // 2 + 1, 1, 2)
        if tuple(spectrum.shape) != expected:
            raise ValueError(
                f"process_frame output must have shape {expected}, got {tuple(spectrum.shape)}"
            )
        if spectrum.dtype != self.dtype or spectrum.device != self.device:
            raise ValueError("process_frame output device/dtype changed")
        if not bool(torch.isfinite(spectrum).all().item()):
            raise ValueError("process_frame output contains NaN/Inf")
        complex_spectrum = torch.view_as_complex(spectrum[:, :, 0].contiguous())
        waveform = torch.fft.irfft(complex_spectrum, n=self.n_fft)
        waveform = waveform[0] * self._window
        return waveform.detach().cpu().numpy().astype(np.float64)

    def _add_synthesis(self, frame_index: int, frame: np.ndarray) -> None:
        start = frame_index * self.hop_length - self.lookahead_samples
        source_start = max(0, -start)
        target_start = max(0, start)
        target_stop = start + self.n_fft
        if target_stop <= 0:
            return
        relative_start = target_start - self._ola_start
        if relative_start < 0:
            raise RuntimeError("TSE WOLA attempted to rewrite emitted samples")
        required = relative_start + (self.n_fft - source_start)
        if required > len(self._ola):
            grow = required - len(self._ola)
            self._ola.extend([0.0] * grow)
            self._ola_norm.extend([0.0] * grow)
        frame_values = frame[source_start:]
        norm_values = self._window_sq[source_start:]
        for offset, (sample, norm) in enumerate(zip(frame_values, norm_values)):
            index = relative_start + offset
            self._ola[index] += float(sample)
            self._ola_norm[index] += float(norm)

    def _drain(self, stable_stop: int) -> np.ndarray:
        stop = min(max(stable_stop, self._emitted), self._input_samples)
        if stop <= self._emitted:
            return np.empty(0, dtype=np.float32)
        count = stop - self._emitted
        if self._emitted != self._ola_start or count > len(self._ola):
            raise RuntimeError("TSE WOLA stable range is not covered by synthesis frames")
        samples = np.asarray(self._ola[:count], dtype=np.float64)
        norm = np.asarray(self._ola_norm[:count], dtype=np.float64)
        if np.any(norm <= 1e-11):
            raise RuntimeError("TSE WOLA normalization violates the NOLA boundary")
        output = (samples / norm).astype(np.float32)
        if not np.all(np.isfinite(output)):
            raise RuntimeError("TSE WOLA output contains NaN/Inf")
        del self._ola[:count]
        del self._ola_norm[:count]
        self._ola_start = stop
        self._emitted = stop
        return output

    def _run_ready_frames(self) -> list[np.ndarray]:
        outputs: list[np.ndarray] = []
        while (
            self._initialized
            and self._analysis_start + len(self._analysis)
            >= self._next_frame * self.hop_length + self.n_fft
        ):
            start = self._next_frame * self.hop_length - self._analysis_start
            frame_values = self._analysis[start : start + self.n_fft]
            try:
                with torch.inference_mode():
                    estimated = self.process_frame(self._stft_frame(frame_values))
                self._add_synthesis(
                    self._next_frame, self._synthesis_frame(estimated)
                )
            except Exception as exc:
                self._poisoned = True
                raise RuntimeError(
                    f"TSE streaming frame failed; call reset() before reuse: {exc}"
                ) from exc
            stable_stop = (
                (self._next_frame + 1) * self.hop_length - self.lookahead_samples
            )
            self._next_frame += 1
            ready = self._drain(stable_stop)
            if ready.size:
                outputs.append(ready)
            discard_to = self._next_frame * self.hop_length
            discard = discard_to - self._analysis_start
            if discard > 0:
                del self._analysis[:discard]
                self._analysis_start = discard_to
        return outputs

    def push(self, samples: object) -> np.ndarray:
        if self._poisoned:
            raise RuntimeError("TSE stream is poisoned; call reset()")
        if self._flushed:
            raise RuntimeError("TSE stream was flushed; call reset()")
        values = _finite_mono(samples, "samples")
        if values.size == 0:
            return values
        self._append_input(values)
        outputs = self._run_ready_frames()
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        if self._poisoned:
            raise RuntimeError("TSE stream is poisoned; call reset()")
        if self._flushed:
            return np.empty(0, dtype=np.float32)
        self._flushed = True
        if self._input_samples <= self.lookahead_samples:
            raise ValueError(
                "center=True reflect padding requires more than n_fft//2 samples"
            )
        right_source = np.asarray(self._tail, dtype=np.float32)
        right = np.pad(
            right_source, (0, self.lookahead_samples), mode="reflect"
        )[-self.lookahead_samples :]
        self._analysis.extend(right.tolist())
        outputs = self._run_ready_frames()
        expected_frames = self._input_samples // self.hop_length + 1
        if self._next_frame != expected_frames:
            raise RuntimeError(
                f"centered STFT frame count mismatch: expected {expected_frames}, "
                f"got {self._next_frame}"
            )
        tail = self._drain(self._input_samples)
        if tail.size:
            outputs.append(tail)
        result = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
        if self._emitted != self._input_samples:
            raise RuntimeError("TSE flush did not emit the complete input duration")
        return result


def build_streaming_enrollment(
    model: torch.nn.Module,
    enrollment_waveform: torch.Tensor,
    *,
    spk_embedding: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int]:
    """按活动 cue 构造 ``init_streaming_state`` 所需 enrollment。"""
    parameter = next(model.parameters())
    waveform = enrollment_waveform
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[0] < 1 or waveform.shape[1] < 1:
        raise ValueError("enrollment_waveform must have shape [B, T>=1]")
    waveform = waveform.to(device=parameter.device, dtype=parameter.dtype)
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError("enrollment_waveform contains NaN/Inf")
    features = getattr(model, "features_config", {})
    enrollment: dict[str, torch.Tensor | int] = {}
    if any(features.get(name, {}).get("enabled", False) for name in ("listen", "usef", "tfmap")):
        window = torch.hann_window(model.n_fft, device=waveform.device, dtype=waveform.dtype)
        spectrum = torch.stft(
            waveform,
            n_fft=model.n_fft,
            hop_length=model.hop_length,
            win_length=model.n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        enrollment["enroll_spectra"] = torch.view_as_real(spectrum)
    if any(features.get(name, {}).get("enabled", False) for name in ("context", "spkemb")):
        if getattr(model, "joint_training", True):
            enrollment["enroll_wav"] = waveform
            enrollment["enroll_lengths"] = torch.full(
                (waveform.shape[0],),
                waveform.shape[1],
                device=waveform.device,
                dtype=torch.int64,
            )
        else:
            if spk_embedding is None:
                raise ValueError("pre-extracted spkemb streaming requires spk_embedding")
            enrollment["spk_embedding"] = spk_embedding.to(
                device=parameter.device, dtype=parameter.dtype
            )
    if not enrollment:
        enrollment["batch_size"] = waveform.shape[0]
    return enrollment


class TSEStreamingWaveformSession:
    """将 causal ``TSEStreamingState`` 提升为 waveform chunk 会话。"""

    def __init__(
        self,
        model: torch.nn.Module,
        enrollment_waveform: torch.Tensor,
        *,
        spk_embedding: torch.Tensor | None = None,
    ) -> None:
        if not getattr(model, "causal", False):
            raise ValueError("TSE waveform streaming requires a causal model")
        self.model = model.eval()
        self._enrollment_waveform = enrollment_waveform.detach().clone()
        self._spk_embedding = (
            spk_embedding.detach().clone() if spk_embedding is not None else None
        )
        self._bind_enrollment()

    def _bind_enrollment(self) -> None:
        enrollment = build_streaming_enrollment(
            self.model,
            self._enrollment_waveform,
            spk_embedding=self._spk_embedding,
        )
        with torch.inference_mode():
            self._state = self.model.init_streaming_state(enrollment)
        parameter = next(self.model.parameters())
        self.frontend = CenteredStreamingSTFTOLA(
            self._process_frame,
            n_fft=int(self.model.n_fft),
            hop_length=int(self.model.hop_length),
            win_length=int(self.model.n_fft),
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def _process_frame(self, spectrum: torch.Tensor) -> torch.Tensor:
        output, next_state = self.model.forward_stream_frame(
            {"mix_spectra": spectrum, "sample_rate": int(self.model.sample_rate)},
            self._state,
        )
        self._state = next_state
        return output["target_estimate_spectra"]

    @property
    def algorithmic_delay_samples(self) -> int:
        return self.frontend.startup_input_samples

    def push(self, samples: object) -> np.ndarray:
        return self.frontend.push(samples)

    def flush(self) -> np.ndarray:
        return self.frontend.flush()

    def reset(
        self,
        enrollment_waveform: torch.Tensor | None = None,
        *,
        spk_embedding: torch.Tensor | None = None,
    ) -> None:
        if enrollment_waveform is not None:
            self._enrollment_waveform = enrollment_waveform.detach().clone()
            self._spk_embedding = (
                spk_embedding.detach().clone() if spk_embedding is not None else None
            )
        elif spk_embedding is not None:
            raise ValueError("spk_embedding replacement requires enrollment_waveform")
        self._bind_enrollment()
