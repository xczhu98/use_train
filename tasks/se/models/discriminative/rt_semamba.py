"""RT-SEMamba：因果 cTF-Mamba 判别式语音增强与 progressive KD。

论文官方实现使用 400/100/400、``center=False`` 的因果谱前端；本文件只消费
框架提供的 ``(B,F,T,2)`` 复谱，不在模型内部执行 STFT/iSTFT。``RT_SEMamba``
符合标准 :class:`BaseSEModel` 契约；``RT_SEMambaDistill`` 是双输入、返回 scalar
loss 的显式例外，由 ``RTSEMambaDistillTrainer`` 驱动。

上游训练还包含 metric discriminator、waveform 与 consistency loss。本仓判别式
适配保留 magnitude/phase/complex task loss 和论文 progressive KD，不声称复现
上游 GAN recipe 或论文指标。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.registry import MODEL_REGISTRY
from core.utils.safe_load import safe_torch_load
from tasks.se.models.base_model import BaseSEModel
from tasks.se.models.modules.rt_semamba_impl import RTSEMambaNetwork, RTSEMambaStreamingState
from tasks.se.models.modules.rt_semamba_impl.network import MixerFactory


def _validate_stft_contract(
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    center: bool,
) -> None:
    expected = (400, 100, 400, False)
    actual = (n_fft, hop_length, win_length, center)
    if actual != expected:
        raise ValueError(
            "RT_SEMamba requires n_fft=400, hop_length=100, win_length=400, "
            f"center=False; got {actual}"
        )
    if window.lower() not in {"hann", "hamming"}:
        raise ValueError(f"RT_SEMamba supports hann or hamming window, got {window!r}")


def _validate_spectrum(spec: torch.Tensor, *, name: str) -> None:
    if spec.ndim != 4 or spec.shape[-1] != 2 or spec.shape[1] != 201:
        raise ValueError(
            f"{name} must have shape (B,201,T,2), got {tuple(spec.shape)}"
        )


def _compressed_components(
    spec: torch.Tensor, compress_factor: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    complex_spec = torch.view_as_complex(spec.float().contiguous())
    magnitude = complex_spec.abs().clamp_min(torch.finfo(torch.float32).eps)
    phase = torch.angle(complex_spec)
    compressed_magnitude = magnitude.pow(compress_factor)
    compressed_complex = torch.stack(
        (
            compressed_magnitude * torch.cos(phase),
            compressed_magnitude * torch.sin(phase),
        ),
        dim=-1,
    )
    return compressed_magnitude, phase, compressed_complex


def _decompress_spectrum(
    compressed_magnitude: torch.Tensor,
    phase: torch.Tensor,
    compress_factor: float,
) -> torch.Tensor:
    magnitude = compressed_magnitude.clamp_min(0).pow(1.0 / compress_factor)
    return torch.stack(
        (magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1
    )


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("RT-SEMamba checkpoint must be a mapping")
    state = checkpoint.get("generator", checkpoint.get("model", checkpoint))
    if not isinstance(state, Mapping) or not state:
        raise ValueError("RT-SEMamba checkpoint contains no non-empty model state_dict")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise TypeError("RT-SEMamba state_dict must map string keys to tensors")
    return state


def _network_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.network.", "network.", "module.")
    keys = list(state)
    for prefix in prefixes:
        if keys and all(key.startswith(prefix) for key in keys):
            return {key[len(prefix) :]: value for key, value in state.items()}
    return dict(state)


def load_rt_semamba_state_strict(
    network: RTSEMambaNetwork,
    checkpoint_path: str,
) -> None:
    """安全、严格加载官方或本仓转换后的 model-only checkpoint。"""
    checkpoint = safe_torch_load(
        str(Path(checkpoint_path).expanduser()),
        map_location="cpu",
        purpose="load RT-SEMamba model-only weights",
    )
    state = _network_state_dict(_extract_state_dict(checkpoint))
    expected = network.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & set(state)
        if expected[key].shape != state[key].shape
        or expected[key].dtype != state[key].dtype
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "RT-SEMamba checkpoint is incompatible: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"shape_or_dtype_mismatch={mismatched[:8]}"
        )
    network.load_state_dict(state, strict=True)


@MODEL_REGISTRY.register("RT_SEMamba")
class RT_SEMamba(BaseSEModel):
    """RT-SEMamba 谱进谱出模型。

    Args:
        num_tfmamba: cTF-Mamba block 数；官方 KD1/KD8 分别为 1/8。
        compress_factor: 幅度压缩指数，官方为 0.3。
        n_fft/hop_length/win_length/window/center: 由顶层 ``stft`` 注入并严格校验。
        mixer_factory: 仅用于 unit test 的显式 Mamba backend 注入点，YAML 不应配置。
    """

    def __init__(
        self,
        num_tfmamba: int = 1,
        hid_feature: int = 64,
        compress_factor: float = 0.3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 4,
        norm_epsilon: float = 1e-5,
        beta: float = 2.0,
        n_fft: int = 400,
        hop_length: int = 100,
        win_length: int = 400,
        window: str = "hann",
        center: bool = False,
        mixer_factory: MixerFactory | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(f"RT_SEMamba got unexpected arguments: {sorted(kwargs)}")
        _validate_stft_contract(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
        )
        if not 0.0 < compress_factor <= 1.0:
            raise ValueError("compress_factor must be in (0, 1]")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window.lower()
        self.center = center
        self.compress_factor = float(compress_factor)
        self.network = RTSEMambaNetwork(
            num_tfmamba=num_tfmamba,
            n_fft=n_fft,
            hid_feature=hid_feature,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            norm_epsilon=norm_epsilon,
            beta=beta,
            mixer_factory=mixer_factory,
        )

    @staticmethod
    def _validate_fs(fs: int | None) -> None:
        if fs is not None and fs != 16000:
            raise ValueError(f"RT_SEMamba supports only fs=16000 Hz, got {fs}")

    def forward_with_features(
        self,
        noisy_spec: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        fs: int | None = None,
    ):
        _validate_spectrum(noisy_spec, name="noisy_spec")
        self._validate_fs(fs)
        if lengths is not None and (
            lengths.ndim != 1 or lengths.shape[0] != noisy_spec.shape[0]
        ):
            raise ValueError("lengths must have shape (B,)")
        mag, phase, _ = _compressed_components(noisy_spec, self.compress_factor)
        return self.network(mag, phase, return_features=True)

    def forward(
        self,
        noisy_spec: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        fs: int | None = None,
    ) -> torch.Tensor:
        if self.training:
            mag, phase, _, _ = self.forward_with_features(noisy_spec, lengths, fs=fs)
        else:
            _validate_spectrum(noisy_spec, name="noisy_spec")
            self._validate_fs(fs)
            if lengths is not None and (
                lengths.ndim != 1 or lengths.shape[0] != noisy_spec.shape[0]
            ):
                raise ValueError("lengths must have shape (B,)")
            noisy_mag, noisy_phase, _ = _compressed_components(
                noisy_spec, self.compress_factor
            )
            state = self.network.init_streaming_state(
                noisy_spec.shape[0], device=noisy_spec.device, dtype=noisy_mag.dtype
            )
            mag, phase, _, _ = self.network.forward_stream_chunk(
                noisy_mag, noisy_phase, state
            )
        return _decompress_spectrum(mag, phase, self.compress_factor)

    def init_streaming_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> RTSEMambaStreamingState:
        return self.network.init_streaming_state(
            batch_size, device=device, dtype=dtype
        )

    def forward_stream_frame(
        self,
        noisy_spec: torch.Tensor,
        state: RTSEMambaStreamingState,
        *,
        fs: int | None = None,
    ) -> tuple[torch.Tensor, RTSEMambaStreamingState]:
        _validate_spectrum(noisy_spec, name="noisy_spec")
        self._validate_fs(fs)
        if noisy_spec.shape[2] != 1:
            raise ValueError("forward_stream_frame expects one spectrum frame")
        mag, phase, _ = _compressed_components(noisy_spec, self.compress_factor)
        enhanced_mag, enhanced_phase, _, next_state = self.network.forward_stream_frame(
            mag, phase, state
        )
        return (
            _decompress_spectrum(enhanced_mag, enhanced_phase, self.compress_factor),
            next_state,
        )

    def forward_stream_chunk(
        self,
        noisy_spec: torch.Tensor,
        state: RTSEMambaStreamingState,
        *,
        fs: int | None = None,
    ) -> tuple[torch.Tensor, RTSEMambaStreamingState]:
        _validate_spectrum(noisy_spec, name="noisy_spec")
        self._validate_fs(fs)
        mag, phase, _ = _compressed_components(noisy_spec, self.compress_factor)
        enhanced_mag, enhanced_phase, _, next_state = self.network.forward_stream_chunk(
            mag, phase, state
        )
        return (
            _decompress_spectrum(enhanced_mag, enhanced_phase, self.compress_factor),
            next_state,
        )


def _anti_wrapped_l1(delta: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * torch.pi
    return (delta - torch.round(delta / two_pi) * two_pi).abs().mean()


def _phase_loss(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    direct = _anti_wrapped_l1(reference - estimate)
    grad_freq = _anti_wrapped_l1(
        torch.diff(reference, dim=1) - torch.diff(estimate, dim=1)
    )
    if reference.shape[2] > 1:
        grad_time = _anti_wrapped_l1(
            torch.diff(reference, dim=2) - torch.diff(estimate, dim=2)
        )
    else:
        grad_time = direct.new_zeros(())
    return direct + grad_freq + grad_time


def _normalize_feature_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


@MODEL_REGISTRY.register("RT_SEMambaDistill")
class RT_SEMambaDistill(nn.Module):
    """8-layer teacher → shallow student progressive KD wrapper。

    这是 ``BaseSEModel`` 的双输入/loss-returning 显式例外。forward 接收 noisy/clean
    spectra 和当前 optimizer step，返回 scalar loss；必须由
    ``tasks.se.trainer.rt_semamba_distill_trainer.RTSEMambaDistillTrainer`` 驱动。
    """

    def __init__(
        self,
        teacher_checkpoint: str,
        student_layers: int = 1,
        teacher_layers: int = 8,
        initialize_student_from_teacher: bool = True,
        ramp_steps: int = 92000,
        lambda_out: float = 0.5,
        lambda_feat: float = 0.1,
        kd_mag_weight: float = 1.0,
        kd_phase_weight: float = 0.3,
        kd_complex_weight: float = 0.5,
        task_mag_weight: float = 0.9,
        task_phase_weight: float = 0.3,
        task_complex_weight: float = 0.1,
        hid_feature: int = 64,
        compress_factor: float = 0.3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 4,
        norm_epsilon: float = 1e-5,
        beta: float = 2.0,
        n_fft: int = 400,
        hop_length: int = 100,
        win_length: int = 400,
        window: str = "hann",
        center: bool = False,
        mixer_factory: MixerFactory | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(
                f"RT_SEMambaDistill got unexpected arguments: {sorted(kwargs)}"
            )
        if not teacher_checkpoint:
            raise ValueError("teacher_checkpoint is required")
        if ramp_steps < 0:
            raise ValueError("ramp_steps must be non-negative")
        common = dict(
            hid_feature=hid_feature,
            compress_factor=compress_factor,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            norm_epsilon=norm_epsilon,
            beta=beta,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            mixer_factory=mixer_factory,
        )
        self.teacher = RT_SEMamba(num_tfmamba=teacher_layers, **common)
        self.student = RT_SEMamba(num_tfmamba=student_layers, **common)
        load_rt_semamba_state_strict(self.teacher.network, teacher_checkpoint)
        if initialize_student_from_teacher:
            self._initialize_student_from_teacher()
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.ramp_steps = int(ramp_steps)
        self.lambda_out = float(lambda_out)
        self.lambda_feat = float(lambda_feat)
        self.kd_weights = (
            float(kd_mag_weight),
            float(kd_phase_weight),
            float(kd_complex_weight),
        )
        self.task_weights = (
            float(task_mag_weight),
            float(task_phase_weight),
            float(task_complex_weight),
        )
        self.compress_factor = float(compress_factor)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _initialize_student_from_teacher(self) -> None:
        student = self.student.network.state_dict()
        teacher = self.teacher.network.state_dict()
        overlap = {
            key: value
            for key, value in teacher.items()
            if key in student and student[key].shape == value.shape
        }
        if not overlap:
            raise RuntimeError("student initialization found no matching teacher tensors")
        student.update(overlap)
        self.student.network.load_state_dict(student, strict=True)

    def forward(
        self,
        noisy_spec: torch.Tensor,
        clean_spec: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        fs: int | None = None,
        step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _validate_spectrum(noisy_spec, name="noisy_spec")
        _validate_spectrum(clean_spec, name="clean_spec")
        if noisy_spec.shape != clean_spec.shape:
            raise ValueError("noisy_spec and clean_spec must share shape")
        if step < 0:
            raise ValueError("step must be non-negative")
        clean_mag, clean_phase, clean_complex = _compressed_components(
            clean_spec, self.compress_factor
        )
        student_mag, student_phase, student_complex, student_features = (
            self.student.forward_with_features(noisy_spec, lengths, fs=fs)
        )
        with torch.no_grad():
            teacher_mag, teacher_phase, teacher_complex, teacher_features = (
                self.teacher.forward_with_features(noisy_spec, lengths, fs=fs)
            )

        task_mag = F.mse_loss(student_mag, clean_mag)
        task_phase = _phase_loss(clean_phase, student_phase)
        task_complex = F.mse_loss(student_complex, clean_complex) * 2.0
        task_loss = (
            self.task_weights[0] * task_mag
            + self.task_weights[1] * task_phase
            + self.task_weights[2] * task_complex
        )

        kd_mag = F.mse_loss(student_mag, teacher_mag)
        kd_phase = F.mse_loss(student_phase, teacher_phase)
        kd_complex = F.mse_loss(student_complex, teacher_complex)
        kd_output = (
            self.kd_weights[0] * kd_mag
            + self.kd_weights[1] * kd_phase
            + self.kd_weights[2] * kd_complex
        )
        student_block = student_features["tfmamba_blocks"][-1]
        teacher_blocks = teacher_features["tfmamba_blocks"]
        teacher_target = torch.stack(teacher_blocks, dim=0).mean(dim=0)
        kd_feature = F.mse_loss(
            _normalize_feature_map(student_block),
            _normalize_feature_map(teacher_target),
        )
        gamma = 1.0 if self.ramp_steps == 0 else min(float(step) / self.ramp_steps, 1.0)
        loss = task_loss + gamma * (
            self.lambda_out * kd_output + self.lambda_feat * kd_feature
        )
        diagnostics = {
            "task": task_loss.detach(),
            "kd_output": kd_output.detach(),
            "kd_feature": kd_feature.detach(),
            "kd_scale": loss.new_tensor(gamma),
        }
        return loss, diagnostics
