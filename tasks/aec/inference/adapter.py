"""AEC 文件级推理 adapter。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from core.inference import FileInferenceItem, PreparedOutput
from core.stft import build_stft_istft
from tasks.aec.inference.aec_frontend import AECDeploymentFrontend, resolve_aec_frontend_config
from tasks.aec.inference.device_policy import (
    move_aec_runtime,
    resolve_aec_file_device,
    validate_cpu_long_audio_seconds,
)
from tasks.aec.model_io import aec_model_accepts_multichannel


class AECInferenceAdapter:
    """运行 canonical AEC frontend；多通道 mic 需显式 reference adapter。"""

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        *,
        stft: Mapping[str, Any] | None = None,
        sample_rate: int | None = None,
        frontend: Mapping[str, Any] | None = None,
        model_fixed_sr: int | None = None,
        cpu_long_audio_seconds: float = 0.0,
        review_multichannel: bool = False,
    ) -> None:
        resolved_rate = model_fixed_sr if model_fixed_sr is not None else sample_rate
        if type(resolved_rate) is not int or resolved_rate <= 0:
            raise ValueError("sample_rate must be an exact positive int")
        self.model = model.eval()
        self.device = torch.device(device)
        self.requested_device = torch.device(device)
        if stft is None:
            raise ValueError("AEC inference requires a top-level stft mapping")
        self.stft = build_stft_istft(dict(stft))
        self.frontend = AECDeploymentFrontend(resolve_aec_frontend_config(frontend))
        self.sample_rate = resolved_rate
        self.cpu_long_audio_seconds = validate_cpu_long_audio_seconds(
            cpu_long_audio_seconds
        )
        self.review_multichannel = bool(review_multichannel)
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            pass

    def _activate_device(self, target: torch.device) -> None:
        """按需成组迁移 AEC runtime 到目标设备。"""
        target = torch.device(target)
        if target == self.device:
            return
        move_aec_runtime(self.model, self.stft, target, self.device)
        self.device = target

    @torch.inference_mode()
    def forward(
        self,
        mic: torch.Tensor,
        farend: torch.Tensor,
        *,
        sample_rate: int | None = None,
    ) -> dict[str, Any]:
        if mic.ndim == 1:
            mic = mic.unsqueeze(0)
        if farend.ndim == 1:
            farend = farend.unsqueeze(0)
        mic = mic.to(self.device)
        farend = farend.to(self.device)
        if (
            mic.ndim not in (2, 3)
            or farend.ndim != 2
            or mic.shape[0] != farend.shape[0]
            or mic.shape[-1] != farend.shape[-1]
        ):
            raise ValueError(
                "mic must be [B,T] or [B,C,T], while farend must be mono [B,T]"
            )
        if mic.ndim == 3 and not aec_model_accepts_multichannel(self.model):
            raise ValueError(
                "multichannel AEC microphone requires a native model IO contract "
                "with a microphone axis; reference-channel projection is not supported"
            )
        model_mic_observation = mic
        effective_rate = self.sample_rate if sample_rate is None else sample_rate
        if type(effective_rate) is not int or effective_rate != self.sample_rate:
            raise ValueError("runtime sample_rate must match AEC model")
        lengths = torch.full(
            (mic.shape[0],), mic.shape[-1], dtype=torch.int64, device=mic.device
        )
        aligned, error, error_mask, facts = self.frontend.process_batch(
            model_mic_observation, farend, lengths, self.sample_rate
        )
        routing = self.frontend.config.input_routing
        model_mic = error if routing == "error_farend" else model_mic_observation
        if model_mic is None:
            raise RuntimeError("AEC frontend did not produce the routed error signal")
        model_input: dict[str, torch.Tensor] = {
            "mic_spectra": self.stft(model_mic),
            "farend_spectra": self.stft(aligned),
        }
        if routing == "mic_farend_error":
            if error is None:
                raise RuntimeError("AEC frontend did not produce error_spectra")
            model_input["error_spectra"] = self.stft(error)
            model_input["error_mask"] = error_mask
        output = self.model(model_input)
        if not isinstance(output, Mapping):
            raise TypeError("AEC model must return a mapping")
        estimate_spectra = output.get("nearend_estimate_spectra")
        if not isinstance(estimate_spectra, torch.Tensor):
            raise KeyError("AEC model output requires nearend_estimate_spectra")
        estimate = self.stft.inverse(estimate_spectra, mic.shape[-1])
        if estimate.ndim != 2 or estimate.shape != (mic.shape[0], mic.shape[-1]):
            raise ValueError(
                "AEC model output must be mono [B,T] with the input time length, got "
                f"{tuple(estimate.shape)}"
            )
        return {
            "nearend_estimate_spectra": estimate_spectra,
            "estimate": estimate,
            "aligned_farend": aligned,
            "frontend_facts": facts,
        }

    def infer(self, item: FileInferenceItem) -> PreparedOutput:
        """按公共 runner 合同处理一个 mic/farend WAV 对。"""
        mic_path = item.inputs.get("mic")
        farend_path = item.inputs.get("farend")
        if mic_path is None or farend_path is None:
            raise ValueError("AEC inference item requires mic and farend paths")
        try:
            import torchaudio
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("AEC file inference requires torchaudio") from exc
        mic, mic_rate = torchaudio.load(str(Path(mic_path)))
        farend, far_rate = torchaudio.load(str(Path(farend_path)))
        if farend.shape[0] != 1:
            raise ValueError("AEC file inference requires mono farend")
        # 长音频策略按 mic 原始帧数与原始采样率判定，重采样前完成。
        file_device = resolve_aec_file_device(
            self.requested_device,
            mic.shape[-1],
            int(mic_rate),
            self.cpu_long_audio_seconds,
        )
        self._activate_device(file_device)
        try:
            mic = mic[0].unsqueeze(0) if mic.shape[0] == 1 else mic.unsqueeze(0)
            farend = farend[0].unsqueeze(0)
            mic_rate = int(mic_rate)
            far_rate = int(far_rate)
            if mic_rate <= 0 or far_rate <= 0:
                raise ValueError("mic and farend sample rates must be positive")
            if far_rate != mic_rate:
                farend = torchaudio.functional.resample(
                    farend.float(), far_rate, mic_rate
                ).to(farend.dtype)
            if mic_rate != self.sample_rate:
                mic = torchaudio.functional.resample(
                    mic.float(), mic_rate, self.sample_rate
                ).to(mic.dtype)
                farend = torchaudio.functional.resample(
                    farend.float(), mic_rate, self.sample_rate
                ).to(farend.dtype)
            length = min(mic.shape[-1], farend.shape[-1])
            if length <= 0:
                raise ValueError("AEC inputs must not be empty")
            result = self.forward(
                mic[..., :length], farend[..., :length], sample_rate=self.sample_rate
            )["estimate"].squeeze(0).cpu()
        finally:
            self._activate_device(self.requested_device)

        saved_wav = result.unsqueeze(0)
        save_kwargs: dict[str, Any] = {}
        if self.review_multichannel:
            # 审听口径：推理口径重采样、共同前缀裁剪后的 [mic..., reference?, enhanced]，
            # reference 保留采集信号，不额外应用 frontend 延时对齐。
            review_mic = mic[..., :length].squeeze(0).cpu()
            review_ref = farend[..., :length].squeeze(0).cpu()
            # mono mic 为 [T]，多通道 mic 为 [C, T]；审听通道按 mic 通道顺序展开。
            review_tracks = [*review_mic] if review_mic.ndim == 2 else [review_mic]
            if torch.count_nonzero(review_ref).item() != 0:
                review_tracks.append(review_ref)
            review_tracks.append(result)
            saved_wav = torch.stack(review_tracks)
            save_kwargs = {"encoding": "PCM_F", "bits_per_sample": 32}

        def writer(path: Path) -> None:
            torchaudio.save(str(path), saved_wav, self.sample_rate, **save_kwargs)

        return PreparedOutput(item.uid, writer)


def build_inference_adapter(
    model: nn.Module,
    device: str | torch.device = "cpu",
    *,
    stft_cfg: Mapping[str, Any] | None = None,
    frontend_cfg: Mapping[str, Any] | None = None,
    model_fixed_sr: int | None = None,
    cpu_long_audio_seconds: float = 0.0,
    review_multichannel: bool = False,
) -> AECInferenceAdapter:
    """将根入口的 ``*_cfg`` 参数显式映射到任务 Adapter 合同。"""
    return AECInferenceAdapter(
        model,
        device=device,
        stft=stft_cfg,
        frontend=frontend_cfg,
        model_fixed_sr=model_fixed_sr,
        cpu_long_audio_seconds=cpu_long_audio_seconds,
        review_multichannel=review_multichannel,
    )
