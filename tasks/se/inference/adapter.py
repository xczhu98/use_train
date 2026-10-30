"""SE waveform inference adapter owned by the SE task."""

from __future__ import annotations

import inspect
import math
from typing import Any, Mapping
from pathlib import Path

import torch
import torch.nn as nn

from core.stft import build_stft_istft
from core.inference import FileInferenceItem, PreparedOutput


class SEInferenceAdapter(nn.Module):
    """Run waveform -> STFT -> SE model -> iSTFT with sample-rate validation.

    The model receives ``[B, F, T, 2]`` for mono input or
    ``[B, C, F, T, 2]`` when it explicitly declares
    ``supports_multichannel_input=True``. It returns the mono estimate spectrum
    directly as a ``[B, F, T, 2]`` tensor; mapping or tuple/list returns are
    rejected instead of being unwrapped.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        *,
        stft: Mapping[str, Any] | None = None,
        sample_rate: int | None = None,
        supports_multichannel_input: bool | None = None,
        stft_module=None,
        multi_sr: bool | None = None,
        model_fixed_sr: int | None = None,
        chunk_seconds: float = 0.0,
    ):
        super().__init__()
        resolved_rate = model_fixed_sr if model_fixed_sr is not None else sample_rate
        if type(resolved_rate) is not int or resolved_rate <= 0:
            raise ValueError("sample_rate must be an exact positive int")
        self.model = model.eval()
        self.device = torch.device(device)
        self.stft = stft_module if stft_module is not None else (
            build_stft_istft(dict(stft)) if stft is not None else None
        )
        self.sample_rate = resolved_rate
        self.multi_sr = bool(multi_sr) if multi_sr is not None else False
        self.chunk_seconds = float(chunk_seconds)
        if not math.isfinite(self.chunk_seconds) or self.chunk_seconds < 0:
            raise ValueError("chunk_seconds must be finite and non-negative")
        declared = getattr(model, "supports_multichannel_input", None)
        contract = getattr(model, "IO_CONTRACT", None)
        contract_inputs = getattr(contract, "inputs", ())
        contract_declares_channels = any(
            "microphone" in getattr(field, "axes", ()) for field in contract_inputs
        )
        if supports_multichannel_input is None:
            supports_multichannel_input = (
                declared is True or contract_declares_channels
            )
        if type(supports_multichannel_input) is not bool:
            raise TypeError("supports_multichannel_input must be bool or None")
        self.supports_multichannel_input = supports_multichannel_input
        declared_rates = {
            name: int(value)
            for name in ("sample_rate", "samplerate", "target_fs")
            if (value := getattr(model, name, None)) is not None
        }
        if len(set(declared_rates.values())) > 1:
            raise ValueError(f"conflicting model sample-rate declarations: {declared_rates}")
        model_rate = next(iter(declared_rates.values()), None)
        if model_rate is not None and model_rate != self.sample_rate:
            raise ValueError(f"model sample rate {model_rate} does not match {self.sample_rate}")
        # 模型调用风格只探测一次：运行期不再用 ``except TypeError`` 回退，否则
        # 模型内部的真实 TypeError 会被吞掉并以缺 lengths/fs 的简化调用重跑
        # （可能掩盖模型 bug，并让忽略 length mask 的路径产出自洽但错误的结果）。
        self._model_accepts_lengths, self._model_accepts_fs = _probe_model_call_style(
            self.model
        )
        if self.multi_sr and not self._model_accepts_fs:
            raise TypeError(
                "multi_sr=True requires a model forward that accepts an 'fs' keyword"
            )

    def infer(self, item: FileInferenceItem) -> PreparedOutput:
        """按公共 runner 合同处理一个 noisy WAV，并返回待发布内容。"""
        input_path = item.inputs.get("noisy")
        if input_path is None:
            raise ValueError("SE inference item requires a noisy path")
        input_path = Path(input_path).expanduser()
        try:
            import torchaudio
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("SE file inference requires torchaudio") from exc
        waveform, input_rate = torchaudio.load(str(input_path))
        if waveform.ndim != 2 or waveform.shape[0] < 1:
            raise ValueError(f"SE input waveform must have shape [C,T]: {input_path}")
        if waveform.shape[0] > 1 and (
            self.stft is None or not self.supports_multichannel_input
        ):
            raise ValueError(
                "multichannel SE input requires a task model IO contract with "
                "a microphone axis; mono inference will not silently downmix"
            )
        waveform = waveform.to(self.device)
        input_rate = int(input_rate)
        if input_rate <= 0:
            raise ValueError("SE input sample rate must be positive")
        original_file_rate = input_rate
        original_file_samples = int(waveform.shape[-1])
        if not self.multi_sr and input_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.float(), input_rate, self.sample_rate
            ).to(waveform.dtype)
            input_rate = self.sample_rate
        if waveform.shape[-1] == 0:
            raise ValueError(f"empty input waveform (0 samples): {input_path}")
        original_samples = int(waveform.shape[-1])
        if self.stft is not None:
            result = self._infer_spectrum_file(waveform, original_samples, input_rate)
            output_rate = input_rate
        else:
            lengths = torch.tensor([original_samples], device=self.device)
            if self.chunk_seconds > 0:
                result = _chunked_waveform_forward(
                    self.model, waveform, input_rate, self.chunk_seconds, self.multi_sr
                )
                # 分块 helper 与模型输出一样保留 batch 轴（[1, T]）；mono 文件推理
                # 需要与 ``_waveform_output`` 相同的出口形状，否则下面的 ndim 校验
                # 会把合法的分块结果判为无效。多通道输入已在上方拒绝。
                if result.ndim == 2 and result.shape[0] == 1:
                    result = result.squeeze(0)
            else:
                result = self._waveform_output(waveform, lengths, input_rate)
            output_rate = int(getattr(self.model, "sr_out", None) or input_rate)
            if output_rate == input_rate:
                result = result[..., :original_samples]
        if result.ndim != 1 or result.numel() == 0 or not torch.isfinite(result).all():
            raise ValueError("SE model returned an empty, non-finite, or invalid waveform")
        # 官方推理按输入文件采样率写出；模型固定 16k 时把输出重采样回原率，
        # 保持时长与文件头一致（多采样率模型由模型自身 sr_out 决定）。
        if (
            not self.multi_sr
            and output_rate == self.sample_rate
            and original_file_rate != output_rate
        ):
            result = torchaudio.functional.resample(
                result.float(), output_rate, original_file_rate
            )
            output_rate = original_file_rate
            result = result[..., :original_file_samples]
        result = result.cpu()

        def writer(path: Path) -> None:
            torchaudio.save(str(path), result.unsqueeze(0), output_rate)

        return PreparedOutput(item.uid, writer)

    def _waveform_output(self, waveform, lengths, sample_rate):
        kwargs = {"fs": sample_rate} if self.multi_sr else {}
        if "normalize_output" in inspect.signature(self.model.forward).parameters:
            # 官方短音频推理统一把输出峰值归一到输入峰值；分块路径另行传 False。
            kwargs["normalize_output"] = True
        output = self.model(waveform, lengths, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "SE waveform model must return a torch.Tensor, "
                f"got {type(output).__name__}"
            )
        return output.squeeze(0)

    def _infer_spectrum_file(self, waveform, original_samples, sample_rate):
        min_len = _min_stft_input_len(self.stft.n_fft)
        if waveform.shape[-1] < min_len:
            waveform = torch.nn.functional.pad(waveform, (0, min_len - waveform.shape[-1]))
        scale = None
        if getattr(self.model, "inference_peak_normalization", False) is True:
            peak = waveform.abs().amax()
            scale = torch.where(peak > 0.0, peak, torch.ones_like(peak))
            waveform = waveform / scale
        lengths = torch.tensor([waveform.shape[-1]], device=waveform.device)
        spectra = self.stft(waveform)
        if waveform.shape[0] > 1:
            spectra = spectra.unsqueeze(0)
        output = self._model_output(spectra, lengths, sample_rate=sample_rate)
        estimate = self.stft.inverse(output, waveform.shape[-1])
        if scale is not None:
            estimate = estimate * scale
        return estimate[..., :original_samples].squeeze(0)

    def _model_output(
        self,
        spectrum: torch.Tensor,
        lengths: torch.Tensor,
        *,
        sample_rate: int | None = None,
    ) -> torch.Tensor:
        effective_rate = self.sample_rate if sample_rate is None else sample_rate
        kwargs: dict[str, Any] = {}
        if self.multi_sr:
            kwargs["fs"] = effective_rate
        if self._model_accepts_lengths:
            kwargs["lengths"] = lengths
        output = self.model(spectrum, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "SE model must return a spectrum torch.Tensor, "
                f"got {type(output).__name__}"
            )
        return output

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor, *, sample_rate: int | None = None) -> dict[str, torch.Tensor]:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim not in (2, 3):
            raise ValueError("waveform must have shape [B, T] or [B, C, T]")
        if waveform.ndim == 3:
            if not self.supports_multichannel_input:
                raise ValueError(
                    "multichannel SE input requires explicit model capability "
                    "supports_multichannel_input=True"
                )
            if waveform.shape[1] < 1:
                raise ValueError("multichannel SE input must contain at least one channel")
            batch, channels, samples = waveform.shape
            stft_input = waveform.reshape(batch * channels, samples)
        else:
            batch, channels, samples = waveform.shape[0], 1, waveform.shape[1]
            stft_input = waveform
        if not waveform.is_floating_point() or not torch.isfinite(waveform).all():
            raise ValueError("waveform must be finite floating-point data")
        effective_rate = self.sample_rate if sample_rate is None else sample_rate
        if type(effective_rate) is not int or effective_rate != self.sample_rate:
            raise ValueError(
                f"runtime sample_rate must match SE model: {effective_rate} != {self.sample_rate}"
            )
        if self.stft is None:
            raise ValueError("SE waveform-native adapter has no spectrum forward path")
        lengths = torch.full(
            (batch,), samples, dtype=torch.int64, device=waveform.device
        )
        spectra = self.stft(stft_input)
        if channels > 1:
            spectra = spectra.reshape(batch, channels, *spectra.shape[1:])
        estimate_spectra = self._model_output(
            spectra, lengths, sample_rate=effective_rate
        )
        if estimate_spectra.ndim != 4 or estimate_spectra.shape[0] != batch:
            raise ValueError(
                "SE model must return mono spectrum with shape [B, F, T, 2]"
            )
        estimate = self.stft.inverse(estimate_spectra, samples)
        if estimate.ndim != 2 or estimate.shape != (batch, samples):
            raise ValueError("SE model must return mono waveform with shape [B, T]")
        return {"estimate": estimate, "estimate_spectra": estimate_spectra}


def _probe_model_call_style(model: nn.Module) -> tuple[bool, bool]:
    """探测 ``model.forward`` 是否接受 ``lengths`` / ``fs`` 关键字。

    判据与 ``core.trainer.metrics_trainer`` 的 ``inspect.signature`` 探测一致；
    声明了 ``**kwargs`` 的前向视为同时接受两者，保持既有 permissive 调用语义。
    """
    parameters = inspect.signature(model.forward).parameters
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return True, True
    return "lengths" in parameters, "fs" in parameters


def _min_stft_input_len(n_fft: int) -> int:
    return int(n_fft) // 2 + 1


def _chunked_waveform_forward(model, waveform, sample_rate, chunk_seconds, multi_sr):
    chunk_seconds = float(chunk_seconds)
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be finite and positive")
    segment = int(chunk_seconds * sample_rate)
    if segment < 1:
        raise ValueError("chunk_seconds is too small to produce at least 1 sample")
    n = waveform.shape[-1]
    output_rate = int(getattr(model, "sr_out", None) or sample_rate)
    hop = max(1, segment // 2)
    spans = []
    start = 0
    while start < n:
        end = min(start + segment, n)
        spans.append((start, end))
        if end == n:
            break
        start += hop
    accepts_normalize = "normalize_output" in inspect.signature(model.forward).parameters
    outputs = []
    for start, end in spans:
        kwargs = {"fs": sample_rate} if multi_sr else {}
        if accepts_normalize:
            kwargs["normalize_output"] = False
        outputs.append(model(waveform[:, start:end], torch.tensor([end - start], device=waveform.device), **kwargs))
    cuts = [0]
    cuts.extend((left[1] + right[0]) // 2 for left, right in zip(spans, spans[1:]))
    cuts.append(n)
    map_sample = lambda value: value * output_rate // sample_rate
    target_cuts = [map_sample(value) for value in cuts]
    parts = []
    for index, ((start, _), output) in enumerate(zip(spans, outputs)):
        if isinstance(output, (tuple, list)):
            output = output[0]
        required = target_cuts[index + 1] - target_cuts[index]
        if output.shape[-1] < required:
            raise ValueError("chunk output too short for target slice")
        local_left = min(max((cuts[index] - start) * output_rate // sample_rate, 0), output.shape[-1] - required)
        parts.append(output[..., local_left : local_left + required])
    result = torch.cat(parts, dim=-1)
    expected = map_sample(n)
    if result.shape[-1] != expected:
        raise ValueError("chunked waveform output length mismatch")
    if accepts_normalize:
        peak = waveform.abs().amax(dim=-1, keepdim=True)
        result = result / (result.abs().amax(dim=-1, keepdim=True) + 1e-8) * peak
    return result


def build_inference_adapter(model, **kwargs) -> SEInferenceAdapter:
    return SEInferenceAdapter(model, **kwargs)
