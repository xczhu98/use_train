# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
"""WeSep MuSE visual frontend 的 canonical TSE 帧级条件适配。

来源：WeSep commit ``dc11cb18`` 的 ``wesep/modules/visual/visual_frontend.py`` 与
``wesep/modules/visual/muse.py``。上游把 lip 预处理、视觉 encoder、时间 adapter
和融合层放在同一个 ``MuseVisualFeature`` 里；本模块只保留 frontend 职责：消费数据
层给出的视觉 cue tensor，输出帧级条件特征交给 ``CueComposer``，不持有 fusion 层、
不持有 separator STFT，也不解码视频文件。

对齐是时间驱动的：用数据侧有效 ``frame_times``（秒）或 ``fps`` 与音频帧网格
（``hop_length / sample_rate`` 间隔，帧中心约定）做 step-hold 取值，不做按点数
resize。缺 cue 行用固定图 ``torch.where`` 门控为零；adapter/encoder 始终 eval，
因此零占位行不会污染 BatchNorm running stats。``freeze_frontend`` 只控制是否保留
梯度，不改变 eval 语义。
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from tasks.tse.models.modules.visual.muse import (
    MUSE_FRONTEND_FEATURE_DIM,
    Muse_LipROIProcessor,
    Muse_VisualFrontend,
    build_visual_adapter,
    compute_muse_frontend,
    load_muse_frontend,
)


VisualRepresentation = Literal["features", "raw"]

_MUSE_FEATURE = "muse_visual"
_MUSE_KEYS = frozenset({"checkpoint", "freeze_frontend", "upsample", "adapter"})
_COMPOSER_KEYS = frozenset({"fusion", "multi_fuse"})
_REPRESENTATIONS = frozenset({"features", "raw"})
_RAW_DEFAULT_ADAPTER = {
    "type": "direct",
    "input_dim": MUSE_FRONTEND_FEATURE_DIM,
    "output_dim": MUSE_FRONTEND_FEATURE_DIM,
}


@dataclass(frozen=True)
class VisualFrontendOutput:
    """帧级视觉条件特征，不负责与 separator latent 融合。

    ``embedding`` 为 ``[R,D_out,T]``，``T`` 是音频帧数；缺 cue 行与超出视觉
    时间范围的部分都取最后一个有效视频帧，缺 cue 行整体为零。
    ``frame_times`` 为 ``[R,T]`` float64 秒（相对 mixture 起点），缺 cue 行全零。
    """

    embedding: torch.Tensor
    present: torch.Tensor
    frame_times: torch.Tensor
    representation: str


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact int, got {value!r}")
    return value


class VisualFrontend(nn.Module):
    """把视觉 cue 对齐为音频帧级条件特征 ``[R,D_out,T]``。

    Args:
        representation: canonical ``features``（预提取 ``[R,D,V]``）或 ``raw``
            （raw video ``[R,H,W,3,V]``）。
        features: 来自 ``CueSpec.options["features"]`` 的 feature map；只接受
            ``muse_visual``，其 ``fusion`` / ``multi_fuse`` 由 ``CueComposer`` 持有。
        sample_rate: 音频采样率，仅用于把音频帧号换算成秒。
        hop_length: 音频帧 hop，仅用于时间对齐；本模块不创建 window，也不做 STFT。
    """

    def __init__(
        self,
        *,
        representation: VisualRepresentation,
        features: Mapping[str, Mapping[str, object]],
        sample_rate: int,
        hop_length: int,
    ) -> None:
        super().__init__()
        if representation not in _REPRESENTATIONS:
            raise ValueError(
                f"visual representation must be one of {sorted(_REPRESENTATIONS)}, "
                f"got {representation!r}"
            )
        if not isinstance(features, Mapping) or not features:
            raise ValueError("features must be a non-empty mapping")
        unknown_features = sorted(set(features) - {_MUSE_FEATURE})
        if unknown_features:
            raise ValueError(
                f"visual frontend contains unknown features: {unknown_features}"
            )
        config = features[_MUSE_FEATURE]
        if not isinstance(config, Mapping):
            raise TypeError("features.muse_visual must be a mapping")
        composer_owned = sorted(set(config) & _COMPOSER_KEYS)
        if composer_owned:
            raise ValueError(
                f"visual frontend does not consume {composer_owned}; "
                "fusion/multi_fuse are held by CueComposer"
            )
        unknown = sorted(set(config) - _MUSE_KEYS)
        if unknown:
            raise ValueError(f"features.muse_visual contains unknown keys: {unknown}")
        if "upsample" in config and config["upsample"] is not True:
            raise ValueError(
                "the visual frontend always aligns cues to audio frames through "
                "frame_times/fps; upsample=False is unsupported"
            )
        freeze_frontend = config.get("freeze_frontend", True)
        if type(freeze_frontend) is not bool:
            raise TypeError("features.muse_visual.freeze_frontend must be a bool")
        checkpoint = config.get("checkpoint")
        if checkpoint is not None and (type(checkpoint) is not str or not checkpoint):
            raise ValueError(
                "features.muse_visual.checkpoint must be a non-empty string"
            )
        adapter_config = config.get("adapter")
        if adapter_config is None:
            if representation != "raw":
                raise ValueError(
                    "pre-extracted visual features require an explicit adapter with "
                    "input_dim/output_dim"
                )
            adapter_config = dict(_RAW_DEFAULT_ADAPTER)

        self.representation = representation
        self.sample_rate = _positive_int(sample_rate, "sample_rate")
        self.hop_length = _positive_int(hop_length, "hop_length")
        self.frame_interval = self.hop_length / self.sample_rate
        self.freeze_frontend = freeze_frontend
        self.adapter = build_visual_adapter(adapter_config)
        self.input_dim = self.adapter.input_dim
        self.output_dim = self.adapter.output_dim

        self.roi = None
        self.encoder = None
        if representation == "raw":
            if checkpoint is None:
                raise ValueError(
                    "raw visual input requires the MuSE encoder checkpoint path; "
                    "missing resources must fail before training starts"
                )
            self.roi = Muse_LipROIProcessor()
            self.encoder = Muse_VisualFrontend()
            load_muse_frontend(self.encoder, checkpoint)
        elif checkpoint is not None:
            raise ValueError(
                "features.muse_visual.checkpoint is only consumed by raw visual input"
            )

        if self.freeze_frontend:
            self.requires_grad_(False)
        # adapter/encoder 始终 eval：零占位行与 padding 帧不得更新 BN running stats。
        self.eval()

    def train(self, mode: bool = True) -> "VisualFrontend":
        """保持 adapter/encoder 恒为 eval；外层 train 不改变 BN/Dropout 状态。"""
        super().train(mode)
        self.adapter.eval()
        if self.encoder is not None:
            self.encoder.eval()
        return self

    def compute(
        self,
        value: torch.Tensor,
        *,
        present: torch.Tensor,
        num_frames: int,
        frame_times: torch.Tensor | None = None,
        fps: float | None = None,
        lengths: torch.Tensor | None = None,
    ) -> VisualFrontendOutput:
        """计算 ``[R,D_out,T]`` 帧级条件特征。

        ``present`` 为 ``[R]`` bool；``num_frames`` 是目标音频帧数 ``T``。
        ``frame_times``（``[R,V]`` 秒）与 ``fps`` 必须二选一；``lengths`` 为
        ``[R]`` int32 有效视频帧数，省略时视为全部 ``V`` 帧有效。
        """
        if not isinstance(present, torch.Tensor) or present.dtype != torch.bool:
            raise TypeError("present must be torch.bool with shape [R]")
        if present.ndim != 1 or present.numel() < 1:
            raise ValueError(f"present must have shape [R], got {tuple(present.shape)}")
        rows = int(present.shape[0])
        device = present.device

        if isinstance(value, (str, bytes, os.PathLike)):
            raise TypeError(
                "the visual frontend consumes decoded cue tensors; raw video decoding "
                "is owned by the data layer"
            )
        if not isinstance(value, torch.Tensor):
            raise TypeError("visual cue value must be a tensor")
        if value.device != device:
            raise ValueError("visual cue value and present must use the same device")
        if value.ndim < 1 or int(value.shape[0]) != rows:
            raise ValueError(
                f"visual cue value must have row axis [{rows}], got {tuple(value.shape)}"
            )
        video_frames = int(value.shape[-1])
        if video_frames < 1:
            raise ValueError("visual cue value requires at least one video frame")

        if self.representation == "features":
            if not value.dtype.is_floating_point:
                raise TypeError("pre-extracted visual features must be floating-point")
            if value.ndim != 3:
                raise ValueError(
                    "pre-extracted visual features must have shape [R,D,V], got "
                    f"{tuple(value.shape)}"
                )
            if int(value.shape[1]) != self.input_dim:
                raise ValueError(
                    f"pre-extracted visual features must provide {self.input_dim} "
                    f"channels, got {tuple(value.shape)}"
                )
        else:
            if value.dtype not in (torch.uint8, torch.float32):
                raise TypeError(
                    f"raw video must be uint8 or float32, got {value.dtype}"
                )
            if value.ndim != 5 or int(value.shape[3]) != 3:
                raise ValueError(
                    "raw video must have shape [R,H,W,3,V], got "
                    f"{tuple(value.shape)}"
                )
        if value.dtype.is_floating_point and not bool(torch.isfinite(value).all().item()):
            raise ValueError("visual cue value must contain only finite values")

        num_frames = _positive_int(num_frames, "num_frames")
        resolved_lengths = self._resolve_lengths(lengths, present, video_frames)
        times = self._resolve_video_times(
            frame_times=frame_times,
            fps=fps,
            present=present,
            lengths=resolved_lengths,
            video_frames=video_frames,
            device=device,
        )

        native = self._native_features(value)
        adapted = self.adapter(native)
        indices = self._frame_indices(times, num_frames, device)
        embedding = torch.gather(
            adapted, 2, indices.unsqueeze(1).expand(-1, self.output_dim, -1)
        )
        embedding = torch.where(present[:, None, None], embedding, 0)

        audio_times = (
            torch.arange(num_frames, dtype=torch.float64, device=device)
            * self.frame_interval
        )
        output_times = torch.where(
            present[:, None], audio_times.unsqueeze(0).expand(rows, -1), 0.0
        )
        return VisualFrontendOutput(
            embedding=embedding,
            present=present,
            frame_times=output_times,
            representation=self.representation,
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        present: torch.Tensor,
        num_frames: int,
        frame_times: torch.Tensor | None = None,
        fps: float | None = None,
        lengths: torch.Tensor | None = None,
    ) -> VisualFrontendOutput:
        return self.compute(
            value,
            present=present,
            num_frames=num_frames,
            frame_times=frame_times,
            fps=fps,
            lengths=lengths,
        )

    def _native_features(self, value: torch.Tensor) -> torch.Tensor:
        """视频帧率下的视觉表示 ``[R,D_in,V]``；adapter 在原生帧率上运行。"""
        if self.encoder is not None:
            assert self.roi is not None
            return compute_muse_frontend(value, self.roi, self.encoder)
        return value.flatten(1, -2)

    @staticmethod
    def _resolve_lengths(
        lengths: torch.Tensor | None,
        present: torch.Tensor,
        video_frames: int,
    ) -> torch.Tensor:
        if lengths is None:
            return torch.full(
                (int(present.shape[0]),),
                video_frames,
                dtype=torch.int32,
                device=present.device,
            )
        if not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.int32:
            raise TypeError("visual lengths must be torch.int32 with shape [R]")
        if tuple(lengths.shape) != (int(present.shape[0]),):
            raise ValueError(f"visual lengths must have shape [R], got {tuple(lengths.shape)}")
        if lengths.device != present.device:
            raise ValueError("visual lengths and present must use the same device")
        invalid = (present & ((lengths < 1) | (lengths > video_frames))) | (
            ~present & (lengths != 0)
        )
        if bool(invalid.any().item()):
            raise ValueError(
                "visual lengths must be within [1,V] for present rows and zero otherwise"
            )
        return lengths

    @staticmethod
    def _resolve_video_times(
        *,
        frame_times: torch.Tensor | None,
        fps: float | None,
        present: torch.Tensor,
        lengths: torch.Tensor,
        video_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        """返回 ``[R,V]`` float64 视频帧时间戳，无效区已置为 ``inf``。"""
        if (frame_times is None) == (fps is None):
            raise ValueError(
                "visual alignment requires exactly one of frame_times or fps"
            )
        valid = (
            torch.arange(video_frames, device=device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        if fps is not None:
            if type(fps) not in (int, float):
                raise TypeError(f"fps must be a finite positive number, got {fps!r}")
            resolved = float(fps)
            if not math.isfinite(resolved) or resolved <= 0.0:
                raise ValueError(f"fps must be a finite positive number, got {fps!r}")
            grid = torch.arange(video_frames, dtype=torch.float64, device=device) / resolved
            times = grid.unsqueeze(0).expand(int(present.shape[0]), -1).clone()
            return torch.where(valid, times, float("inf"))

        if not isinstance(frame_times, torch.Tensor):
            raise TypeError("frame_times must be a floating-point tensor with shape [R,V]")
        if not frame_times.dtype.is_floating_point:
            raise TypeError("frame_times must be a floating-point tensor with shape [R,V]")
        if tuple(frame_times.shape) != (int(present.shape[0]), video_frames):
            raise ValueError(
                f"frame_times must have shape [R,{video_frames}], got "
                f"{tuple(frame_times.shape)}"
            )
        if frame_times.device != device:
            raise ValueError("frame_times and present must use the same device")
        times = frame_times.to(dtype=torch.float64)
        if not bool(torch.isfinite(times[valid]).all().item()):
            raise ValueError("valid frame_times must be finite")
        if bool((times[valid] < 0).any().item()):
            raise ValueError(
                "frame_times must be relative to mixture start and non-negative"
            )
        present_times = times[present]
        if present_times.numel() and bool((present_times[:, 0] != 0).any().item()):
            raise ValueError("valid frame_times must start at 0")
        if bool((~present).any().item()) and bool((times[~present] != 0).any().item()):
            raise ValueError(
                "missing visual rows must use zero frame_times placeholders"
            )
        if video_frames > 1 and bool(
            ((times[:, 1:] < times[:, :-1]) & valid[:, 1:] & valid[:, :-1]).any().item()
        ):
            raise ValueError("frame_times must be monotonic non-decreasing")
        if bool(((times != 0) & ~valid).any().item()):
            raise ValueError("frame_times after the valid length must be zero")
        return torch.where(valid, times, float("inf"))

    def _frame_indices(
        self,
        times: torch.Tensor,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        """音频帧 → 有效视频帧的 step-hold 索引 ``[R,T]`` int64。"""
        audio_times = (
            torch.arange(num_frames, dtype=torch.float64, device=device)
            * self.frame_interval
        )
        queries = audio_times.unsqueeze(0).expand(int(times.shape[0]), -1).contiguous()
        indices = torch.searchsorted(times.contiguous(), queries, right=True) - 1
        return indices.clamp_(min=0)


__all__ = ["VisualFrontend", "VisualFrontendOutput", "VisualRepresentation"]
