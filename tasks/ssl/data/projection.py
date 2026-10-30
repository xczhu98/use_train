"""SSL 空间监督投影。

该模块只负责把已经渲染好的空间 scene 投影成 canonical SSL feature/target。
输入合同：observation ``[C,T]``，direct tracks ``[K,C,T]``；
输出合同：多通道 STFT 实虚特征 ``[T,F,2C]``、成对直达路径 IPD target
``[T,F,K,P,2]``、DOA ``[T,K,2]`` 和 VAD ``[T,K]``。
``vad_mode=direct_path`` 按每轨 direct-path 谱能量相对峰值阈值产生 VAD；
VAD 来源与阈值写入 item provenance。

时间基准字段（随每个 item 显式返回，语义固定如下）：

- ``receptive_field``：输出时间轴上单帧聚合的输入帧跨度。当前
  ``_reduce_output_frames`` 是左对齐平均池化，每个输出帧消费
  ``[t, t+time_stride)`` 输入帧，故 ``receptive_field == time_stride``；
  尾部不足一个 stride 的输入帧被丢弃，和模型的 floor 输出长度保持一致。
- ``algorithmic_latency``：相对输出帧索引（池化组首帧，见
  ``output_frame_index``）计算该输出帧需要前瞻的输入帧数。左对齐平均
  池化下整组 ``[t, t+time_stride)`` 都参与，故登记 ``time_stride - 1``；
  若按池化组声学中心解释则为 ``(time_stride-1)/2``，但本实现索引左对齐，
  统一以前瞻口径登记，causal 时同为 ``time_stride - 1``。
- ``valid_output_mask``：``[T_out]`` bool，标识有效输出帧；projection
  产出的未补齐 item 全为 True，collate 对 ``output_lengths`` 之外的
  补齐帧置 False。

inactive target manifold（``SSLProjectionConfig.inactive_target_manifold``）：

- ``zero``（默认，FN-SSL 官方形式）：直达谱退化的 T-F bin target 为零向量；
- ``bessel_j0``（IPDnet/IPDnet2 官方形式）：退化 bin 的 real 部为
  ``J0(2*pi*f*d_p/c)``、imag 部为零，``d_p`` 为该 pair 的麦克风间距、
  ``c`` 为配置的声速。manifold 只取决于频率、几何与声速，是与样本
  无关的有限常数；退化帧同时被 VAD mask 排除，不参与 masked loss。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any

import numpy as np
import torch
from scipy.special import j0 as _bessel_j0

from .batch import (
    SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG,
    SSL_INPUT_REFERENCE_IPD,
    SSL_TARGET_DPIPD_FAR_FIELD,
    SSL_TARGET_DPIPD_NEAR_FIELD,
)


SSL_INACTIVE_MANIFOLD_ZERO = "zero"
SSL_INACTIVE_MANIFOLD_BESSEL_J0 = "bessel_j0"
SSL_VAD_MODE_DIRECT_PATH = "direct_path"
_INACTIVE_MANIFOLDS = frozenset(
    {SSL_INACTIVE_MANIFOLD_ZERO, SSL_INACTIVE_MANIFOLD_BESSEL_J0}
)
_VAD_MODES = frozenset({SSL_VAD_MODE_DIRECT_PATH})


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _strict_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a strict lowercase 64-hex SHA-256 digest")
    return value


@dataclass(frozen=True)
class SSLSceneInput:
    """已完成声学渲染的 SSL scene 输入。

    ``source_positions`` 可以是静态 ``[K,3]``，也可以是与 STFT frame 对齐的
    ``[T_frame,K,3]``。位置只用于 DOA 投影，不参与 waveform 渲染。
    """

    observation: torch.Tensor | np.ndarray
    direct_tracks: torch.Tensor | np.ndarray
    mic_positions: torch.Tensor | np.ndarray
    source_positions: torch.Tensor | np.ndarray
    sample_rate: int
    provenance: dict[str, Any]
    coordinate_frame: str = "room_xyz_m"

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observation = torch.as_tensor(self.observation, dtype=torch.float32)
        direct_tracks = torch.as_tensor(self.direct_tracks, dtype=torch.float32)
        mic_positions = torch.as_tensor(self.mic_positions, dtype=torch.float32)
        source_positions = torch.as_tensor(self.source_positions, dtype=torch.float32)
        if observation.ndim != 2:
            raise ValueError("SSL observation must have shape [C,T]")
        if direct_tracks.ndim != 3:
            raise ValueError("SSL direct_tracks must have shape [K,C,T]")
        if mic_positions.ndim != 2 or mic_positions.shape[1] != 3:
            raise ValueError("SSL mic_positions must have shape [C,3]")
        if source_positions.ndim not in (2, 3) or source_positions.shape[-1] != 3:
            raise ValueError("SSL source_positions must have shape [K,3] or [T,K,3]")
        channels, length = observation.shape
        tracks, track_channels, track_length = direct_tracks.shape
        if channels < 2 or track_channels != channels:
            raise ValueError("SSL observation/direct_tracks must share C>=2")
        if length != track_length:
            raise ValueError("SSL observation and direct_tracks must share waveform length")
        if mic_positions.shape[0] != channels:
            raise ValueError("SSL mic_positions channel count must equal observation C")
        if source_positions.shape[-2] != tracks:
            raise ValueError("SSL source_positions track count must equal direct_tracks K")
        if type(self.sample_rate) is not int or self.sample_rate < 1000:
            raise ValueError("SSL sample_rate must be an exact integer >= 1000")
        for name, value in (("observation", observation), ("direct_tracks", direct_tracks),
                            ("mic_positions", mic_positions), ("source_positions", source_positions)):
            if value.numel() == 0 or not torch.isfinite(value).all():
                raise ValueError(f"SSL {name} must be non-empty and finite")
        if not isinstance(self.provenance, dict):
            raise TypeError("SSL provenance must be a mapping")
        if not isinstance(self.coordinate_frame, str) or not self.coordinate_frame.strip():
            raise ValueError("SSL coordinate_frame must be a non-empty string")
        return observation, direct_tracks, mic_positions, source_positions


@dataclass(frozen=True)
class SSLProjectionConfig:
    """STFT 与 direct-path target 投影参数。

    ``forgetting_laplace`` 的递推长度通过
    ``forgetting_norm_sample_length`` 显式绑定 profile；默认 298 对应官方
    forgetting norm helper，IPDNet2 配置显式使用其 249 帧 recipe。
    ``global_mean`` 是 IPDnet offline 的全 ``[C,F,T]`` 幅度均值模式。
    """

    n_fft: int = 512
    hop_length: int = 256
    win_length: int | None = None
    window: str = "hamming"
    center: bool = False
    reference_channel: int = 0
    vad_threshold_ratio: float = 1e-2
    input_representation: str = SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG
    target_representation: str = SSL_TARGET_DPIPD_FAR_FIELD
    exclude_dc: bool = False
    normalization: str = "none"
    forgetting_norm_sample_length: int = 298
    output_stride: int = 1
    sound_speed: float = 343.0
    simulation_profile: str | None = None
    pair_layout: str = "reference_to_all"
    inactive_target_manifold: str = SSL_INACTIVE_MANIFOLD_ZERO
    # 仅记录 target 与 DOA mapper 的相位约定，不改变 projection 的互谱公式。
    phase_sign: float = 1.0
    vad_mode: str = SSL_VAD_MODE_DIRECT_PATH

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("n_fft", self.n_fft, 2),
            ("hop_length", self.hop_length, 1),
            ("reference_channel", self.reference_channel, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an exact integer >= {minimum}")
        if self.win_length is not None and (
            type(self.win_length) is not int or not 1 <= self.win_length <= self.n_fft
        ):
            raise ValueError("win_length must be an exact integer in [1, n_fft]")
        if self.window not in {"hann", "hamming"}:
            raise ValueError("SSL STFT window must be hann or hamming")
        if type(self.center) is not bool:
            raise TypeError("center must be a bool")
        if self.input_representation not in {SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG, SSL_INPUT_REFERENCE_IPD}:
            raise ValueError(
                "SSL input_representation must be multichannel_stft_real_imag or reference_ipd"
            )
        if self.target_representation not in {
            SSL_TARGET_DPIPD_FAR_FIELD,
            SSL_TARGET_DPIPD_NEAR_FIELD,
        }:
            raise ValueError("SSL target_representation is unsupported")
        if type(self.exclude_dc) is not bool:
            raise TypeError("exclude_dc must be a bool")
        if not isinstance(self.normalization, str) or self.normalization not in {
            "none",
            "forgetting_laplace",
            "global_mean",
        }:
            raise ValueError(
                "SSL normalization must be none, forgetting_laplace or global_mean"
            )
        if type(self.forgetting_norm_sample_length) is not int or self.forgetting_norm_sample_length < 1:
            raise ValueError(
                "forgetting_norm_sample_length must be an exact integer >= 1"
            )
        if type(self.output_stride) is not int or self.output_stride < 1:
            raise ValueError("output_stride must be an exact positive integer")
        if self.pair_layout != "reference_to_all":
            raise ValueError("SSL projection currently supports reference_to_all pair layout only")
        if self.inactive_target_manifold not in _INACTIVE_MANIFOLDS:
            raise ValueError(
                "SSL inactive_target_manifold must be one of "
                f"{sorted(_INACTIVE_MANIFOLDS)!r}"
            )
        if self.vad_mode not in _VAD_MODES:
            raise ValueError(
                f"SSL vad_mode must be one of {sorted(_VAD_MODES)!r}"
            )
        if (
            isinstance(self.phase_sign, bool)
            or not isinstance(self.phase_sign, (int, float))
            or not math.isfinite(float(self.phase_sign))
            or float(self.phase_sign) not in {-1.0, 1.0}
        ):
            raise ValueError("phase_sign must be -1 or +1")
        if (
            isinstance(self.sound_speed, bool)
            or not isinstance(self.sound_speed, (int, float))
            or not math.isfinite(float(self.sound_speed))
            or float(self.sound_speed) <= 0.0
        ):
            raise ValueError("sound_speed must be a positive finite number")
        if self.simulation_profile is not None and (
            not isinstance(self.simulation_profile, str)
            or not self.simulation_profile.strip()
        ):
            raise ValueError("simulation_profile must be a non-empty string when provided")
        if (
            not isinstance(self.vad_threshold_ratio, (int, float))
            or isinstance(self.vad_threshold_ratio, bool)
            or not math.isfinite(float(self.vad_threshold_ratio))
            or not 0.0 < float(self.vad_threshold_ratio) <= 1.0
        ):
            raise ValueError("vad_threshold_ratio must be finite and in (0,1]")


def _forgetting_laplace_normalize(
    spectra: torch.Tensor, *, sample_length: int
) -> torch.Tensor:
    """按官方 forgetting/Laplace 递推归一化 ``[C,F,T]`` 复数谱。

    官方递推在每个时间帧先对所有 ``C*F`` 幅度取均值，再以
    ``alpha=(L-1)/(L+1)`` 更新；``sample_length`` 必须由 profile 显式登记，
    不能用固定的 0.9/0.1 近似替代。
    """
    if spectra.ndim != 3 or not torch.is_complex(spectra):
        raise ValueError("SSL forgetting normalization expects complex spectra [C,F,T]")
    if type(sample_length) is not int or sample_length < 1:
        raise ValueError("forgetting_norm_sample_length must be an exact integer >= 1")
    magnitude = spectra.abs()
    flat = magnitude.reshape(1, -1, spectra.shape[-1])
    alpha = (sample_length - 1) / (sample_length + 1)
    running = torch.zeros(
        1, 1, dtype=magnitude.dtype, device=magnitude.device
    )
    factors: list[torch.Tensor] = []
    for frame in range(spectra.shape[-1]):
        local = (
            min((frame - 1) / (frame + 1), alpha)
            if frame < sample_length
            else alpha
        )
        running = local * running + (1.0 - local) * flat[:, :, frame].mean(
            dim=1, keepdim=True
        )
        factors.append(running)
    factor = torch.cat(factors, dim=-1).reshape(1, 1, -1)
    # 官方样本的幅度均值非零；零帧仍需保持有限值，避免坏资源产生 inf。
    eps = torch.finfo(magnitude.dtype).eps
    return spectra / factor.clamp_min(eps)


def _global_mean_normalize(spectra: torch.Tensor) -> torch.Tensor:
    """按官方 offline IPDnet 语义用整个 ``[C,F,T]`` 的幅度均值归一化。"""
    if spectra.ndim != 3 or not torch.is_complex(spectra):
        raise ValueError("SSL global_mean normalization expects complex spectra [C,F,T]")
    magnitude = spectra.abs()
    factor = magnitude.mean()
    if not torch.isfinite(factor):
        raise ValueError("SSL global_mean normalization factor must be finite")
    return spectra / factor.clamp_min(torch.finfo(magnitude.dtype).eps)


def _stft(
    waveform: torch.Tensor,
    config: SSLProjectionConfig,
    *,
    apply_normalization: bool = True,
) -> torch.Tensor:
    win_length = config.win_length or config.n_fft
    factory = {"hann": torch.hann_window, "hamming": torch.hamming_window}.get(config.window)
    if factory is None:
        raise ValueError("SSL STFT window must be hann or hamming")
    window = factory(win_length, dtype=waveform.dtype, device=waveform.device)
    spectrum = torch.stft(
        waveform.reshape(-1, waveform.shape[-1]),
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=win_length,
        window=window,
        center=config.center,
        return_complex=True,
    )
    spectrum = spectrum.reshape(*waveform.shape[:-1], spectrum.shape[-2], spectrum.shape[-1])
    if config.exclude_dc:
        spectrum = spectrum[..., 1:, :]
    if apply_normalization and config.normalization == "forgetting_laplace":
        # IPDnet2 的 forgetting normalization 只用于观测特征；保持 direct
        # target 的物理相位，不把幅度归一化泄漏到监督轨迹。
        spectrum = _forgetting_laplace_normalize(
            spectrum, sample_length=config.forgetting_norm_sample_length
        )
    elif apply_normalization and config.normalization == "global_mean":
        spectrum = _global_mean_normalize(spectrum)
    return spectrum


def _raw_stft_real_imag(spectra: torch.Tensor) -> torch.Tensor:
    """``[C,F,T]`` complex STFT 到 ``[T,F,2C]``。"""
    if spectra.ndim != 3 or not torch.is_complex(spectra):
        raise ValueError("SSL multichannel STFT spectra must have complex shape [C,F,T]")
    return torch.view_as_real(spectra.permute(2, 1, 0).contiguous()).reshape(
        spectra.shape[-1], spectra.shape[-2], spectra.shape[0] * 2
    )


def _reference_ipd_features(spectra: torch.Tensor, reference_channel: int) -> torch.Tensor:
    """构造 native baseline 使用的 ``[T,F,2(C-1)]`` reference IPD。"""
    channels = spectra.shape[0]
    if not 0 <= reference_channel < channels:
        raise ValueError("reference_channel is outside observation spectra")
    others = torch.cat((spectra[:reference_channel], spectra[reference_channel + 1 :]), dim=0)
    reference = spectra[reference_channel : reference_channel + 1]
    phase = others * reference.conj()
    phase = phase / phase.abs().clamp_min(torch.finfo(spectra.real.dtype).eps)
    return torch.view_as_real(phase.permute(2, 1, 0).contiguous()).reshape(
        spectra.shape[-1], spectra.shape[-2], (channels - 1) * 2
    )


def _direct_ipd_target(
    direct_spectra: torch.Tensor,
    reference_channel: int,
    manifold: torch.Tensor | None = None,
) -> torch.Tensor:
    """从 direct-path spectra 生成不合并 pair 轴的 ``[T,F,K,P,2]`` target。

    直达谱退化（幅度为 0）的 T-F bin 无法定义单位相位，归一化后得到
    零向量；``manifold`` 为 ``[F,P,2]`` 时把这些 bin 显式替换为登记的
    inactive manifold 常数。manifold 只按频率和 pair 变化，与样本无关。
    """
    tracks, channels, _, _ = direct_spectra.shape
    if not 0 <= reference_channel < channels:
        raise ValueError("reference_channel is outside direct_tracks microphone dimension")
    reference = direct_spectra[:, reference_channel : reference_channel + 1]
    others = torch.cat((direct_spectra[:, :reference_channel], direct_spectra[:, reference_channel + 1:]), dim=1)
    phase = others * reference.conj()
    eps = torch.finfo(phase.real.dtype).eps
    magnitude = phase.abs()
    phase = phase / magnitude.clamp_min(eps)
    target = torch.view_as_real(phase.permute(3, 2, 0, 1).contiguous()).to(torch.float32)
    if manifold is not None:
        if manifold.shape != (target.shape[1], target.shape[3], 2):
            raise ValueError("SSL inactive manifold must have shape [F,P,2]")
        degenerate = (magnitude <= eps).permute(3, 2, 0, 1).unsqueeze(-1)
        target = torch.where(degenerate, manifold[:, None].to(target.dtype), target)
    return target


def _inactive_manifold(
    config: SSLProjectionConfig,
    mic_positions: torch.Tensor,
    pair_index: torch.Tensor,
    freqs_hz: torch.Tensor,
) -> torch.Tensor:
    """构造 ``[F,P,2]`` 的 inactive target manifold 常数。

    ``zero``：全零（FN-SSL 官方形式）；``bessel_j0``：real 部
    ``J0(2*pi*f*d_p/c)``、imag 部零（IPDnet/IPDnet2 官方形式，即整个
    定位空间的 DP-IPD 均值点），``d_p`` 为 pair 麦克风间距。
    """
    freqs = freqs_hz.shape[0]
    pairs = pair_index.shape[0]
    if config.inactive_target_manifold == SSL_INACTIVE_MANIFOLD_ZERO:
        return torch.zeros(freqs, pairs, 2, dtype=torch.float32)
    if config.inactive_target_manifold == SSL_INACTIVE_MANIFOLD_BESSEL_J0:
        reference = mic_positions[pair_index[:, 0].to(torch.long)]
        other = mic_positions[pair_index[:, 1].to(torch.long)]
        distance = torch.linalg.vector_norm(other - reference, dim=-1)  # [P]
        argument = (
            2.0 * math.pi * freqs_hz.to(torch.float64)[:, None]
            * distance.to(torch.float64)[None, :]
            / float(config.sound_speed)
        )
        real = torch.from_numpy(_bessel_j0(argument.numpy())).to(torch.float32)
        return torch.stack((real, torch.zeros_like(real)), dim=-1)
    raise ValueError(
        f"unsupported SSL inactive_target_manifold: {config.inactive_target_manifold!r}"
    )


def _reference_pair_index(channels: int, reference_channel: int) -> torch.Tensor:
    if channels < 2:
        raise ValueError("SSL reference-to-all layout requires at least two microphones")
    others = [channel for channel in range(channels) if channel != reference_channel]
    return torch.tensor(
        [[reference_channel, channel] for channel in others], dtype=torch.int64
    )


def _align_source_positions(source_positions: torch.Tensor, frames: int) -> torch.Tensor:
    """把轨迹控制点线性对齐到最终 observation STFT frame 轴。"""
    if type(frames) is not int or frames < 1:
        raise ValueError("projected STFT frame count must be a positive exact integer")
    if source_positions.ndim == 2:
        return source_positions.unsqueeze(0).expand(frames, -1, -1)
    points = source_positions.shape[0]
    if points == frames:
        return source_positions
    if points == 1:
        return source_positions.expand(frames, -1, -1)
    locations = torch.linspace(
        0.0, float(points - 1), frames,
        dtype=source_positions.dtype, device=source_positions.device,
    )
    left = locations.floor().to(torch.long)
    right = locations.ceil().to(torch.long)
    weight = (locations - left.to(locations.dtype)).view(-1, 1, 1)
    return source_positions[left] * (1.0 - weight) + source_positions[right] * weight


def _doa(mic_positions: torch.Tensor, source_positions: torch.Tensor, reference_channel: int = 0) -> torch.Tensor:
    """返回 ``[..., 2]`` 方向角：``(elevation, azimuth)``。

    注意这里的 "elevation" 是从 +Z 轴起算的极角（zenith/polar angle，
    ``atan2(‖xy‖, z)``），不是声学惯例中从水平面起算的 elevation
    （``atan2(z, ‖xy‖)``）；两者换算关系为 ``el_水平面 = π/2 - 本函数返回值``。
    该口径仅用于 SSL 任务内部自洽的标签/特征生产，**不要**直接当作 TSE
    spatial cue（`{azimuth, elevation}` 字段）或其他跨任务方向约定使用；
    TSE cue 采用 `(azimuth, 水平面 elevation)` 惯例。
    """
    offset = source_positions - mic_positions[reference_channel].view(1, 1, 3)
    azimuth = torch.atan2(offset[..., 1], offset[..., 0])
    elevation = torch.atan2(torch.linalg.vector_norm(offset[..., :2], dim=-1), offset[..., 2])
    return torch.stack((elevation, azimuth), dim=-1).to(torch.float32)


def _vad(
    direct_spectra: torch.Tensor,
    threshold_ratio: float,
    *,
    mode: str = SSL_VAD_MODE_DIRECT_PATH,
) -> torch.Tensor:
    if mode != SSL_VAD_MODE_DIRECT_PATH:
        raise ValueError(f"unsupported SSL vad_mode: {mode!r}")
    if not 0.0 < threshold_ratio <= 1.0:
        raise ValueError("vad_threshold_ratio must be in (0,1]")
    energy = direct_spectra.abs().mean(dim=(1, 2))  # [K,T]
    threshold = energy.amax(dim=1, keepdim=True).clamp_min(1e-8) * threshold_ratio
    return (energy >= threshold).transpose(0, 1).to(torch.float32)


def _average_doa(directions: torch.Tensor) -> torch.Tensor:
    """对 ``[N,K,2]`` 的 polar/azimuth 方向做三维单位向量平均。"""
    polar, azimuth = directions[..., 0], directions[..., 1]
    unit = torch.stack(
        (
            torch.sin(polar) * torch.cos(azimuth),
            torch.sin(polar) * torch.sin(azimuth),
            torch.cos(polar),
        ),
        dim=-1,
    ).mean(dim=0)
    return torch.stack(
        (
            torch.atan2(torch.linalg.vector_norm(unit[..., :2], dim=-1), unit[..., 2]),
            torch.atan2(unit[..., 1], unit[..., 0]),
        ),
        dim=-1,
    )


def _reduce_output_frames(
    targets: torch.Tensor,
    doa: torch.Tensor,
    vad: torch.Tensor,
    source_positions: torch.Tensor,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """按显式 stride 聚合监督并返回输出帧索引。"""
    frames = targets.shape[0]
    if stride == 1:
        return targets, doa, vad, source_positions, torch.arange(
            frames, dtype=torch.int64, device=targets.device
        )
    if frames < stride:
        raise ValueError(
            "SSL projection needs at least one complete output-stride window: "
            f"frames={frames}, stride={stride}"
        )
    # AvgPool1d/official causal convolutions drop a trailing partial window;
    # target aggregation must use the same floor(frame_count / stride) rule.
    groups = [
        torch.arange(start, start + stride, device=targets.device)
        for start in range(0, frames - stride + 1, stride)
    ]
    target_values = []
    doa_values = []
    vad_values = []
    position_values = []
    for group in groups:
        complex_target = torch.view_as_complex(targets[group].contiguous()).mean(dim=0)
        complex_target = complex_target / complex_target.abs().clamp_min(1e-8)
        target_values.append(torch.view_as_real(complex_target))
        doa_values.append(_average_doa(doa[group]))
        vad_values.append(vad[group].mean(dim=0))
        position_values.append(source_positions[group].mean(dim=0))
    return (
        torch.stack(target_values),
        torch.stack(doa_values),
        torch.stack(vad_values),
        torch.stack(position_values),
        torch.tensor([int(group[0]) for group in groups], dtype=torch.int64, device=targets.device),
    )


def project_ssl_scene(scene: SSLSceneInput, config: SSLProjectionConfig | None = None) -> dict[str, Any]:
    """将 direct-path scene 投影为 canonical SSL item。

    除特征/监督张量外，还显式返回时间基准 ``receptive_field``、
    ``algorithmic_latency``、``valid_output_mask``（语义见模块 docstring），
    并把 inactive manifold 选择与时间基准写入 provenance。
    """
    config = config or SSLProjectionConfig()
    if config.n_fft < 2 or config.hop_length < 1 or not 1 <= (config.win_length or config.n_fft) <= config.n_fft:
        raise ValueError("invalid SSL STFT configuration")
    observation, direct_tracks, mic_positions, source_positions = scene.tensors()
    if not 0 <= config.reference_channel < observation.shape[0]:
        raise ValueError("reference_channel is outside observation microphone dimension")
    observation_spectra = _stft(observation, config)
    direct_spectra = _stft(direct_tracks, config, apply_normalization=False)
    if config.input_representation == SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG:
        features = _raw_stft_real_imag(observation_spectra)
    elif config.input_representation == SSL_INPUT_REFERENCE_IPD:
        features = _reference_ipd_features(observation_spectra, config.reference_channel)
    else:
        raise ValueError(f"unsupported SSL input representation: {config.input_representation!r}")
    pair_index = _reference_pair_index(observation.shape[0], config.reference_channel)
    freq_bins = features.shape[1]
    first_bin = 1 if config.exclude_dc else 0
    freqs_hz = (
        torch.arange(first_bin, first_bin + freq_bins, dtype=torch.float64)
        * (float(scene.sample_rate) / float(config.n_fft))
    )
    manifold = _inactive_manifold(config, mic_positions, pair_index, freqs_hz)
    targets = _direct_ipd_target(direct_spectra, config.reference_channel, manifold)
    input_frames = features.shape[0]
    if targets.shape[:2] != (input_frames, freq_bins):
        raise ValueError("observation and direct_tracks STFT layouts do not match")
    frame_positions = _align_source_positions(source_positions, input_frames)
    doa = _doa(mic_positions, frame_positions, config.reference_channel)
    vad = _vad(
        direct_spectra,
        config.vad_threshold_ratio,
        mode=config.vad_mode,
    )
    targets, doa, vad, output_positions, output_frame_index = _reduce_output_frames(
        targets, doa, vad, frame_positions, config.output_stride
    )
    # 时间基准：左对齐平均池化下单个输出帧消费 time_stride 个输入帧，
    # 输出帧索引对齐池化组首帧，前瞻为 time_stride - 1（见模块 docstring）。
    receptive_field = config.output_stride
    algorithmic_latency = config.output_stride - 1
    provenance = dict(scene.provenance)
    declared_coordinate_frame = provenance.get("coordinate_frame")
    if declared_coordinate_frame is not None and declared_coordinate_frame != scene.coordinate_frame:
        raise ValueError(
            "SSL provenance.coordinate_frame does not match scene coordinate_frame"
        )
    provenance["coordinate_frame"] = scene.coordinate_frame
    declared_sample_rate = provenance.get("sample_rate")
    if declared_sample_rate is not None and declared_sample_rate != scene.sample_rate:
        raise ValueError("SSL provenance.sample_rate does not match scene sample_rate")
    provenance["sample_rate"] = int(scene.sample_rate)
    geometry = provenance.get("geometry")
    if geometry is None:
        provenance["geometry"] = {
            "positions_m": mic_positions.detach().cpu().tolist(),
            "reference_channel": int(config.reference_channel),
        }
    elif isinstance(geometry, dict):
        if "positions_m" not in geometry or geometry["positions_m"] is None:
            raise ValueError(
                "SSL provenance.geometry requires positions_m when geometry is provided"
            )
        declared_positions = torch.as_tensor(geometry["positions_m"], dtype=torch.float32)
        if declared_positions.shape != mic_positions.shape or not torch.allclose(
            declared_positions, mic_positions, rtol=0.0, atol=1e-6
        ):
            raise ValueError("SSL provenance.geometry does not match mic_positions")
        declared_reference = geometry.get("reference_channel", config.reference_channel)
        if type(declared_reference) is not int or declared_reference != config.reference_channel:
            raise ValueError(
                "SSL provenance.geometry reference_channel does not match projection"
            )
        canonical_geometry = dict(geometry)
        canonical_geometry["reference_channel"] = int(config.reference_channel)
        provenance["geometry"] = canonical_geometry
    elif not isinstance(geometry, dict):
        raise ValueError("SSL provenance.geometry must be a mapping when provided")
    stft_contract = {
        "n_fft": int(config.n_fft),
        "hop_length": int(config.hop_length),
        "win_length": int(config.win_length or config.n_fft),
        "window": config.window,
        "center": bool(config.center),
        "exclude_dc": bool(config.exclude_dc),
    }
    declared_stft = provenance.get("stft")
    if declared_stft is not None:
        if not isinstance(declared_stft, dict):
            raise ValueError("SSL provenance.stft must be a mapping when provided")
        for key, expected in stft_contract.items():
            if key not in declared_stft:
                continue
            actual = declared_stft[key]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"SSL provenance.stft.{key} does not match projection configuration"
                )
        stft_contract = {**declared_stft, **stft_contract}
    provenance["stft"] = stft_contract
    for key in ("sound_speed_mps", "sound_speed"):
        if key in provenance:
            value = provenance[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) != float(config.sound_speed)
            ):
                raise ValueError(
                    "SSL provenance sound_speed does not match projection configuration"
                )
    room = provenance.get("room")
    if isinstance(room, dict):
        for key in ("sound_speed_mps", "sound_speed"):
            if key not in room:
                continue
            value = room[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) != float(config.sound_speed)
            ):
                raise ValueError(
                    "SSL provenance.room sound_speed does not match projection configuration"
                )
    provenance["sound_speed_mps"] = float(config.sound_speed)
    declared_profile = provenance.get("simulation_profile")
    if declared_profile is not None and (
        not isinstance(declared_profile, str) or not declared_profile.strip()
    ):
        raise ValueError("SSL provenance.simulation_profile must be a non-empty string")
    if config.simulation_profile is not None:
        profile = config.simulation_profile.strip()
        if declared_profile is not None and declared_profile.strip() != profile:
            raise ValueError(
                "SSL provenance.simulation_profile does not match projection configuration"
            )
        provenance["simulation_profile"] = profile
    elif declared_profile is not None:
        provenance["simulation_profile"] = declared_profile.strip()
    if "rir_sha256" in provenance:
        provenance["rir_sha256"] = _strict_sha256(
            provenance["rir_sha256"], name="SSL provenance.rir_sha256"
        )
    direct_hash = hashlib.sha256(
        np.ascontiguousarray(direct_tracks.detach().cpu().numpy(), dtype="<f4").tobytes()
    ).hexdigest()
    if "direct_path_content_sha256" in provenance and _strict_sha256(
        provenance["direct_path_content_sha256"],
        name="SSL provenance.direct_path_content_sha256",
    ) != direct_hash:
        raise ValueError(
            "SSL provenance.direct_path_content_sha256 does not match direct_path content"
        )
    provenance["direct_path_content_sha256"] = direct_hash
    provenance["inactive_target_manifold"] = config.inactive_target_manifold
    provenance["vad_supervision"] = {
        "mode": config.vad_mode,
        "activity_source": "direct_path_spectra",
        "threshold_ratio": float(config.vad_threshold_ratio),
        "aggregation": (
            "mean_over_output_stride" if config.output_stride > 1 else "framewise"
        ),
    }
    provenance["phase_sign"] = float(config.phase_sign)
    provenance["normalization"] = {
        "mode": config.normalization,
        "forgetting_norm_sample_length": (
            int(config.forgetting_norm_sample_length)
            if config.normalization == "forgetting_laplace"
            else None
        ),
        "aggregation": {
            "none": "disabled",
            "forgetting_laplace": "framewise_c_f_mean",
            "global_mean": "global_c_f_t_mean",
        }[config.normalization],
    }
    provenance["timebase"] = {
        "time_stride": config.output_stride,
        "receptive_field": receptive_field,
        "algorithmic_latency": algorithmic_latency,
        "alignment": "pool_group_start",
    }
    item = {
        "features": features.to(torch.float32),
        "target_dpipd": targets.to(torch.float32),
        "doa": doa.to(torch.float32),
        "vad": vad.to(torch.float32),
        "pair_index": pair_index,
        "pair_mask": torch.ones(pair_index.shape[0], dtype=torch.bool),
        "geometry": mic_positions.to(torch.float32),
        "input_lengths": input_frames,
        "output_lengths": targets.shape[0],
        "frame_origin": 0,
        "time_stride": config.output_stride,
        "receptive_field": receptive_field,
        "algorithmic_latency": algorithmic_latency,
        "valid_output_mask": torch.ones(targets.shape[0], dtype=torch.bool),
        "sample_rate": scene.sample_rate,
        "source_positions": frame_positions if source_positions.ndim == 3 else source_positions,
        "output_source_positions": output_positions.to(torch.float32),
        "input_frame_index": torch.arange(input_frames, dtype=torch.int64),
        "output_frame_index": output_frame_index,
        "input_representation": config.input_representation,
        "target_representation": config.target_representation,
        "coordinate_frame": scene.coordinate_frame,
        "provenance": provenance,
    }
    if config.target_representation == SSL_TARGET_DPIPD_NEAR_FIELD:
        source_distances = torch.linalg.vector_norm(
            output_positions - mic_positions[config.reference_channel].view(1, 1, 3), dim=-1
        ).to(torch.float32)
        if not torch.isfinite(source_distances).all() or (source_distances <= 0).any():
            raise ValueError("SSL near-field source_distances must be finite and strictly positive")
        item["source_distances"] = source_distances
        provenance["source_distances"] = source_distances.detach().cpu().tolist()
    return item


_SCENE_BACKENDS = frozenset({"online", "offline"})


def _finite_scene_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite array")
    return array


def _declared_rir_hash(provenance: dict[str, Any]) -> str | None:
    """只从 full-RIR 身份字段取 hash，并拒绝同一身份的冲突声明。"""
    declarations: list[tuple[str, str]] = []
    pending: list[tuple[str, dict[str, Any]]] = [("provenance", provenance)]
    visited: set[int] = set()
    while pending:
        path, candidate = pending.pop(0)
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        # direct/early 是不同 track kind，不能与 full RIR hash 比较；full
        # 及 scene-level aliases 则必须保持同一身份。
        track_kind = candidate.get("track_kind")
        is_non_full = track_kind in {"direct", "early"} or any(
            part in {"direct", "early"} for part in path.split(".")
        )
        if not is_non_full:
            for key in ("rir_sha256", "content_hash", "full_rir_sha256"):
                if key in candidate:
                    declarations.append(
                        (
                            f"{path}.{key}",
                            _strict_sha256(
                                candidate[key], name=f"SSL provenance.{path}.{key}"
                            ),
                        )
                    )
            track_set = candidate.get("track_set")
            if isinstance(track_set, dict) and "full" in track_set:
                full_value = track_set["full"]
                # ``SpatialRIRTrackSet`` identity carries either a digest or a
                # nested ``{track_kind, content_hash, provenance}`` fact.  The
                # latter is visited below; only scalar declarations are
                # validated at this level.
                if not isinstance(full_value, dict):
                    declarations.append(
                        (
                            f"{path}.track_set.full",
                            _strict_sha256(
                                full_value,
                                name=f"SSL provenance.{path}.track_set.full",
                            ),
                        )
                    )
        for key in (
            "rir",
            "track_identity",
            "track_facts",
            "track_set",
            "full",
            "direct",
            "early",
            "assets",
            "asset",
            "provenance",
        ):
            nested = candidate.get(key)
            if isinstance(nested, dict):
                pending.append((f"{path}.{key}", nested))
    if not declarations:
        return None
    hashes = {value for _, value in declarations}
    if len(hashes) != 1:
        raise ValueError(
            "SSL RIR identity hash declarations disagree: "
            f"{declarations!r}"
        )
    return declarations[0][1]


def _declared_generator_version(provenance: dict[str, Any]) -> str | None:
    """从 RIR 身份 facts 读取真实 generator version。

    ``prepare_ssl_scene`` 的输入通常是 scene facts：generator version 位于
    ``rir``/``track_facts`` 内层，而不是 post-mix 顶层。不能用 projection
    代码版本伪造声学生成器身份；若顶层和 RIR facts 同时声明且不一致，直接
    拒绝这条样本。
    """
    declared: list[tuple[str, str]] = []
    pending: list[tuple[str, dict[str, Any]]] = [("provenance", provenance)]
    visited: set[int] = set()
    while pending:
        path, candidate = pending.pop(0)
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if "generator_version" in candidate:
            value = candidate["generator_version"]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"SSL provenance.{path}.generator_version must be non-empty")
            declared.append((path, value.strip()))
        for key in (
            "rir",
            "track_facts",
            "track_identity",
            "track_set",
            "full",
            "direct",
            "early",
            "assets",
            "asset",
            "pre_rendered_provenance",
            "provenance",
        ):
            nested = candidate.get(key)
            if isinstance(nested, dict):
                pending.append((f"{path}.{key}", nested))
    if not declared:
        return None
    # 顶层声明是 canonical 发布值；内层 RIR facts 必须与之相同。
    values = {value for _, value in declared}
    if len(values) != 1:
        raise ValueError(
            "SSL provenance generator_version declarations disagree: "
            f"{declared!r}"
        )
    return declared[0][1]


def _declared_source_order(
    provenance: dict[str, Any], *, expected_tracks: int
) -> list[str] | None:
    """读取并校验 RIR facts 的 source/actor order。

    Dataset 投影必须保留 ``[K,C,T]`` 的 source 轴语义；缺少顺序或把
    duplicate/空 UID 透传到监督会使 track identity 不可审计。core spatial
    executor 通常同时写入 ``source_order`` 和 ``actor_order``，两者若都
    存在也必须完全一致。
    """
    declarations: list[tuple[str, tuple[str, ...]]] = []
    pending: list[tuple[str, dict[str, Any]]] = [("provenance", provenance)]
    visited: set[int] = set()
    while pending:
        path, candidate = pending.pop(0)
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        for key in ("source_order", "actor_order"):
            if key not in candidate:
                continue
            value = candidate[key]
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"SSL provenance.{path}.{key} must be a string list")
            normalized = tuple(
                entry.strip() if isinstance(entry, str) else entry for entry in value
            )
            if (
                len(normalized) != expected_tracks
                or any(not isinstance(entry, str) or not entry for entry in normalized)
                or len(set(normalized)) != len(normalized)
            ):
                raise ValueError(
                    f"SSL provenance.{path}.{key} must contain exactly "
                    f"{expected_tracks} unique non-empty source UIDs"
                )
            declarations.append((f"{path}.{key}", normalized))
        for key in (
            "rir",
            "track_identity",
            "track_facts",
            "track_set",
            "full",
            "direct",
            "early",
            "provenance",
        ):
            nested = candidate.get(key)
            if isinstance(nested, dict):
                pending.append((f"{path}.{key}", nested))
    if not declarations:
        return None
    orders = {order for _, order in declarations}
    if len(orders) != 1:
        raise ValueError(
            "SSL provenance source/actor order declarations disagree: "
            f"{declarations!r}"
        )
    return list(next(iter(orders)))


def prepare_ssl_scene(
    observation: np.ndarray,
    direct_path: np.ndarray,
    *,
    sample_rate: int,
    mic_positions: np.ndarray,
    source_positions: np.ndarray,
    reference_channel: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    center: bool,
    expected_channels: int,
    expected_tracks: int,
    expected_freqs: int,
    expected_input_size: int,
    backend: str,
    provenance: dict[str, Any],
    input_representation: str = SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG,
    target_representation: str = SSL_TARGET_DPIPD_FAR_FIELD,
    exclude_dc: bool = False,
    normalization: str = "none",
    forgetting_norm_sample_length: int = 298,
    sound_speed: float | None = None,
    simulation_profile: str | None = None,
    output_stride: int = 1,
    coordinate_frame: str = "room_xyz_m",
    inactive_target_manifold: str = SSL_INACTIVE_MANIFOLD_ZERO,
    vad_mode: str = SSL_VAD_MODE_DIRECT_PATH,
    phase_sign: float = 1.0,
) -> dict[str, Any]:
    """统一把已渲染 scene 交给 ``project_ssl_scene`` 的 Dataset 侧入口。

    只做形状/合同校验与身份绑定，不实现第二套 STFT、IPD、DOA 或 VAD 投影。
    ``direct_path`` 的物理语义由调用方声明；本模块不会把 early path 静默升级
    为 exact direct path。
    """
    observation = np.array(observation, dtype=np.float32, copy=True)
    if observation.ndim == 1:
        observation = observation[None, :]
    elif observation.ndim == 2 and observation.shape[0] != expected_channels and observation.shape[1] == expected_channels:
        observation = observation.T
    direct_path = np.array(direct_path, dtype=np.float32, copy=True)
    if direct_path.ndim == 2:
        if direct_path.shape[0] == expected_channels:
            direct_path = direct_path[None, :]
        elif direct_path.shape[1] == expected_channels:
            direct_path = direct_path.T[None, :]
    elif direct_path.ndim == 3 and direct_path.shape[1] != expected_channels and direct_path.shape[2] == expected_channels:
        direct_path = direct_path.transpose(0, 2, 1)
    if observation.ndim != 2 or observation.shape[0] != expected_channels:
        raise ValueError(f"observation must have shape [C,T] with C={expected_channels}, got {observation.shape}")
    if direct_path.ndim != 3 or direct_path.shape[1] != expected_channels:
        raise ValueError(f"direct_path must have shape [K,C,T], got {direct_path.shape}")
    if direct_path.shape[0] != expected_tracks:
        raise ValueError(f"direct_path tracks K={direct_path.shape[0]} != configured {expected_tracks}")
    if observation.shape[-1] != direct_path.shape[-1]:
        raise ValueError(
            "SSL observation and direct_path must share waveform length"
        )
    length = observation.shape[-1]
    if length < 1:
        raise ValueError("SSL scene waveforms must be non-empty")
    observation = observation[:, :length]
    direct_path = direct_path[:, :, :length]
    mic_positions = _finite_scene_array(mic_positions, name="mic_positions", ndim=2)
    source_positions = _finite_scene_array(source_positions, name="source_positions")
    if mic_positions.shape != (expected_channels, 3):
        raise ValueError(f"mic_positions must have shape [C,3], got {mic_positions.shape}")
    if source_positions.ndim not in (2, 3) or source_positions.shape[-2:] != (expected_tracks, 3):
        raise ValueError(f"source_positions must have shape [K,3] or [T,K,3], got {source_positions.shape}")
    expected_feature_size = (
        expected_channels * 2
        if input_representation == SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG
        else (expected_channels - 1) * 2
    )
    if expected_input_size != expected_feature_size:
        raise ValueError(
            "SSL expected_input_size must equal the declared feature dimension: "
            f"expected={expected_feature_size}, got={expected_input_size}"
        )
    if backend not in _SCENE_BACKENDS:
        raise ValueError(f"SSL scene backend is unsupported: {backend!r}")
    provenance = dict(provenance or {})
    generator_version = _declared_generator_version(provenance)
    if generator_version is None:
        raise ValueError(
            "SSL provenance must include the real RIR generator_version; "
            "missing generator identity is not accepted"
        )
    existing_generator_version = provenance.get("generator_version")
    if (
        existing_generator_version is not None
        and existing_generator_version != generator_version
    ):
        raise ValueError(
            "SSL provenance.generator_version does not match RIR generator identity"
        )
    provenance["generator_version"] = generator_version
    content_hash = hashlib.sha256(
        np.ascontiguousarray(direct_path, dtype="<f4").tobytes()
    ).hexdigest()
    rir_hash = _declared_rir_hash(provenance)
    if rir_hash is None:
        raise ValueError(
            "SSL provenance must include an RIR identity hash (rir_sha256/content_hash)"
        )
    provenance["rir_sha256"] = rir_hash
    declared_backend = provenance.get("backend")
    if declared_backend is not None and declared_backend != backend:
        raise ValueError("SSL provenance.backend does not match scene backend")
    source_order = _declared_source_order(provenance, expected_tracks=expected_tracks)
    if source_order is None:
        raise ValueError(
            "SSL provenance must include source_order/actor_order for every direct track"
        )
    provenance["source_order"] = source_order
    declared_direct_hash = provenance.get("direct_path_content_sha256")
    if declared_direct_hash is not None and _strict_sha256(
        declared_direct_hash, name="SSL provenance.direct_path_content_sha256"
    ) != content_hash:
        raise ValueError(
            "SSL provenance.direct_path_content_sha256 does not match direct_path content"
        )
    provenance["direct_path_content_sha256"] = content_hash
    provenance["backend"] = backend
    projection_kwargs = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "window": window,
        "center": center,
        "reference_channel": reference_channel,
        "input_representation": input_representation,
        "target_representation": target_representation,
        "exclude_dc": exclude_dc,
        "normalization": normalization,
        "forgetting_norm_sample_length": forgetting_norm_sample_length,
        "output_stride": output_stride,
        "simulation_profile": simulation_profile,
        "inactive_target_manifold": inactive_target_manifold,
        "vad_mode": vad_mode,
        "phase_sign": phase_sign,
    }
    if sound_speed is not None:
        projection_kwargs["sound_speed"] = sound_speed
    item = project_ssl_scene(
        SSLSceneInput(
            observation=observation,
            direct_tracks=direct_path,
            mic_positions=mic_positions,
            source_positions=source_positions,
            sample_rate=sample_rate,
            provenance=provenance,
            coordinate_frame=coordinate_frame,
        ),
        SSLProjectionConfig(**projection_kwargs),
    )
    if item["features"].shape[1] != expected_freqs:
        raise ValueError("configured SSL frequency bins do not match n_fft")
    item["mic_positions"] = torch.from_numpy(mic_positions)
    item["source_positions"] = item["source_positions"].to(torch.float32)
    return item
