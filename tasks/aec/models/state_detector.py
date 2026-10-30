"""Reusable frame-level AEC state detector head.

The state detector predicts a per-frame acoustic scenario class such as
far-end single talk (FST), double talk (DT), or near-end single talk (NST).
It is model-agnostic: any AEC encoder that can provide paired mic/far
features at one or more resolutions can attach this head and output
``state_probs`` of shape ``(B, T, nclass)``.

This module was originally embedded in :class:`tasks.aec.models.discriminative.dbgnet.DBGNet`
and has been extracted so that other AEC models (CAGCRN, etc.)
can reuse the same state-detection supervision.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _reject_unexpected_kwargs(class_name: str, kwargs: dict[str, object]) -> None:
    if kwargs:
        raise TypeError(f"{class_name} got unexpected keyword arguments: {sorted(kwargs)}")


def _require_positive_int(name: str, value: object) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _require_spatial_layout(spatial_layout: object) -> None:
    if spatial_layout not in ("F_T", "T_F"):
        raise ValueError(
            f"spatial_layout must be 'F_T' or 'T_F', got {spatial_layout!r}"
        )


class AECStateFeatureExtractor(nn.Module):
    """Project one resolution of paired mic/far features to a state embedding.

    The input is a pair of tensors with the same shape ``(B, C, *, *)``
    representing microphone and far-end features from one encoder layer.
    Each tensor is mean-pooled over channels, optionally passed through a
    light normalization + ReLU, concatenated along frequency, and projected
    to ``output_size`` dimensions.

    Args:
        input_size: Linear input dimension (``2 * number_of_frequency_bins``).
        output_size: Projection output dimension.
        spatial_layout: Encoder feature layout.  ``"F_T"`` means
            ``(B, C, F, T)`` (e.g. DBGNet); ``"T_F"`` means ``(B, C, T, F)``
            (e.g. DeepVQE).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 64,
        spatial_layout: str = "F_T",
        **kwargs,
    ):
        super().__init__()
        _reject_unexpected_kwargs(type(self).__name__, kwargs)
        _require_positive_int("input_size", input_size)
        _require_positive_int("output_size", output_size)
        _require_spatial_layout(spatial_layout)
        self.spatial_layout = spatial_layout
        self.pp = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, mic_feat: Tensor, far_feat: Tensor) -> Tensor:
        """Args:
            mic_feat: Microphone feature, shape ``(B, C, F, T)`` or ``(B, C, T, F)``.
            far_feat: Far-end feature, same shape as ``mic_feat``.

        Returns:
            State embedding of shape ``(B, T, output_size)``.
        """
        if self.spatial_layout == "F_T":
            # (B, C, F, T) → mean over C → (B, 1, F, T) → (B, F, T)
            mic = torch.mean(mic_feat, dim=1, keepdim=True)
            mic = self.pp(mic).squeeze(1)
            far = torch.mean(far_feat, dim=1)
            # concat along F → (B, 2F, T) → permute → (B, T, 2F)
            z = torch.cat([mic, far], dim=1).permute(0, 2, 1)
        else:
            # (B, C, T, F) → mean over C → (B, 1, T, F) → (B, T, F)
            mic = torch.mean(mic_feat, dim=1, keepdim=True)
            mic = self.pp(mic).squeeze(1)
            far = torch.mean(far_feat, dim=1)
            # concat along F → (B, T, 2F)
            z = torch.cat([mic, far], dim=2)
        return self.linear(z)


class AECStateDetector(nn.Module):
    """Frame-level state classifier.

    Consumes a concatenation of per-resolution state embeddings and outputs
    class logits for each time frame.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        nclass: int = 3,
        **kwargs,
    ):
        super().__init__()
        _reject_unexpected_kwargs(type(self).__name__, kwargs)
        _require_positive_int("input_size", input_size)
        _require_positive_int("hidden_size", hidden_size)
        _require_positive_int("nclass", nclass)
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_size, nclass)

    def forward(self, z: Tensor) -> Tensor:
        """Args:
            z: Concatenated state embeddings, shape ``(B, T, input_size)``.

        Returns:
            Class logits of shape ``(B, T, nclass)``.
        """
        z, _ = self.rnn(z)
        return self.linear(z)


class AECStateDetectorHead(nn.Module):
    """Convenience head combining feature extractors + classifier.

    Args:
        feature_freq_sizes: List of frequency-bin counts ``F`` for each
            mic/far feature pair.  The linear input size is ``2 * F`` because
            mic and far features are concatenated along the frequency axis
            after channel mean-pooling.
        projection_size: Output size of each ``AECStateFeatureExtractor``.
        hidden_size: GRU hidden size in ``AECStateDetector``.
        nclass: Number of state classes (e.g. 3 for FST/DT/NST, 8 for scenario).
        spatial_layout: Feature layout, ``"F_T"`` for ``(B, C, F, T)`` or
            ``"T_F"`` for ``(B, C, T, F)``.
    """

    def __init__(
        self,
        feature_freq_sizes: Iterable[int],
        *,
        projection_size: int = 64,
        hidden_size: int = 128,
        nclass: int = 3,
        spatial_layout: str = "F_T",
        **kwargs,
    ):
        super().__init__()
        _reject_unexpected_kwargs(type(self).__name__, kwargs)
        try:
            freq_sizes = list(feature_freq_sizes)
        except TypeError as exc:
            raise TypeError("feature_freq_sizes must be an iterable of integers") from exc
        if not freq_sizes:
            raise ValueError("feature_freq_sizes must be non-empty")
        for index, freq_size in enumerate(freq_sizes):
            _require_positive_int(f"feature_freq_sizes[{index}]", freq_size)
        _require_positive_int("projection_size", projection_size)
        _require_positive_int("hidden_size", hidden_size)
        _require_positive_int("nclass", nclass)
        _require_spatial_layout(spatial_layout)
        self.feature_extractors = nn.ModuleList(
            AECStateFeatureExtractor(2 * f, projection_size, spatial_layout)
            for f in freq_sizes
        )
        self.detector = AECStateDetector(
            input_size=len(freq_sizes) * projection_size,
            hidden_size=hidden_size,
            nclass=nclass,
        )

    def forward(self, feature_pairs: list[tuple[Tensor, Tensor]]) -> Tensor:
        """Args:
            feature_pairs: List of ``(mic_feat, far_feat)`` tensors.  All
                tensors must match the configured ``spatial_layout``.

        Returns:
            State logits of shape ``(B, T_max, nclass)`` where ``T_max`` is
            the finest (largest) time dimension among all feature pairs.
            Lower-resolution embeddings are upsampled via linear interpolation.
        """
        if len(feature_pairs) != len(self.feature_extractors):
            raise ValueError(
                f"Expected {len(self.feature_extractors)} feature pairs, got {len(feature_pairs)}"
            )
        embeddings: list[torch.Tensor] = [
            extractor(mic, far)
            for extractor, (mic, far) in zip(self.feature_extractors, feature_pairs)
        ]
        # Multi-resolution time alignment: encoder levels with stride>1 produce
        # coarser time grids.  Upsample all to the finest resolution so that
        # torch.cat along dim=-1 succeeds and the output aligns with STFT frames.
        t_max = max(e.shape[1] for e in embeddings)
        for i, e in enumerate(embeddings):
            if e.shape[1] != t_max:
                embeddings[i] = F.interpolate(
                    e.transpose(1, 2),  # (B, C, T_i)
                    size=t_max,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)  # (B, T_max, C)
        z = torch.cat(embeddings, dim=-1)  # (B, T_max, len(freq_sizes) * projection_size)
        return self.detector(z)
