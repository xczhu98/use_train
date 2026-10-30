"""模块化 TSE 使用的谱域 separator Adapter。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, TypeAlias

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from tasks.tse.models.modules.separator.bsrnn import (
    BSRNN_Separator,
    BandMasker,
    BandSplit,
    SubbandNorm,
)


class LatentFusion(Protocol):
    def __call__(self, latent: Tensor) -> Tensor: ...


class BlockConditioner(Protocol):
    def __call__(self, latent: Tensor, condition: Tensor) -> Tensor: ...


InitialState: TypeAlias = dict[str, tuple[Tensor, Tensor]]
InitialStates: TypeAlias = InitialState | Sequence[InitialState | None] | None
BlockConditions: TypeAlias = Tensor | Sequence[Tensor | None] | None


def _default_band_width(sample_rate: int, n_fft: int) -> tuple[int, ...]:
    frequency_bins = n_fft // 2 + 1
    widths = [int(np.floor(100 / (sample_rate / 2.0) * frequency_bins))] * 15
    widths += [int(np.floor(200 / (sample_rate / 2.0) * frequency_bins))] * 10
    widths += [int(np.floor(500 / (sample_rate / 2.0) * frequency_bins))] * 5
    widths += [int(np.floor(2000 / (sample_rate / 2.0) * frequency_bins))]
    widths.append(frequency_bins - sum(widths))
    return tuple(widths)


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be an exact positive int, got {value!r}")
    return value


class BSRNNSeparatorAdapter(nn.Module):
    """将 runtime STFT 表示映射为框架 RI 频谱。

    ``spectral_repr`` 为 ``[R, D, F, T]`` 实值特征，``reference_spec`` 为
    ``[R, F, T]`` 复数参考通道谱。Adapter 只拥有 BSRNN separator 参数；
    cue frontend、fusion/conditioner 由外层模型持有并在调用时传入。
    ``causal=True`` 当前仍是 full-sequence forward，不声明 streaming 能力。
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        input_dim: int,
        feature_dim: int = 128,
        num_repeat: int = 6,
        causal: bool = False,
        nspk: int = 1,
        norm_type: str | None = None,
        band_width: Sequence[int] | None = None,
        eps: float = torch.finfo(torch.float32).eps,
        rnn_norm_eps: float | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = _positive_int(sample_rate, "sample_rate")
        self.n_fft = _positive_int(n_fft, "n_fft")
        if self.n_fft % 2:
            raise ValueError(f"n_fft must be even, got {self.n_fft}")
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.feature_dim = _positive_int(feature_dim, "feature_dim")
        self.num_repeat = _positive_int(num_repeat, "num_repeat")
        if type(causal) is not bool:
            raise TypeError(f"causal must be a bool, got {causal!r}")
        if type(nspk) is not int or nspk != 1:
            raise ValueError("BSRNNSeparatorAdapter requires nspk=1 for target extraction")
        if norm_type is None:
            norm_type = "cLN" if causal else "GN"
        if norm_type not in {"GN", "cLN", "gLN", "BN"}:
            raise ValueError(f"unsupported norm_type: {norm_type!r}")
        if type(eps) not in {int, float} or not math.isfinite(float(eps)) or eps <= 0:
            raise ValueError(f"eps must be a finite positive number, got {eps!r}")

        frequency_bins = self.n_fft // 2 + 1
        raw_widths = (
            _default_band_width(self.sample_rate, self.n_fft)
            if band_width is None
            else tuple(band_width)
        )
        if not raw_widths:
            raise ValueError("band_width must not be empty")
        resolved_widths = tuple(
            _positive_int(width, f"band_width[{index}]")
            for index, width in enumerate(raw_widths)
        )
        if sum(resolved_widths) != frequency_bins:
            raise ValueError(
                f"band_width must sum to F={frequency_bins}, got {sum(resolved_widths)}"
            )

        self.causal = causal
        self.nspk = nspk
        self.norm_type = norm_type
        self.eps = float(eps)
        if rnn_norm_eps is not None and (
            type(rnn_norm_eps) not in {int, float}
            or not math.isfinite(rnn_norm_eps) or rnn_norm_eps <= 0
        ):
            raise ValueError("rnn_norm_eps must be finite and positive")
        self.rnn_norm_eps = self.eps if rnn_norm_eps is None else float(rnn_norm_eps)
        self.band_width = resolved_widths
        self.nband = len(resolved_widths)

        self.band_split = BandSplit(resolved_widths)
        self.subband_norm = SubbandNorm(
            band_width=resolved_widths,
            spec_dim=self.input_dim,
            nband=self.nband,
            feature_dim=self.feature_dim,
            norm_type=norm_type,
            eps=self.eps,
        )
        self.separator = BSRNN_Separator(
            nband=self.nband,
            num_repeat=self.num_repeat,
            feature_dim=self.feature_dim,
            causal=causal,
            norm_type=norm_type,
            eps=self.rnn_norm_eps,
        )
        self.band_masker = BandMasker(
            band_width=resolved_widths,
            nband=self.nband,
            feature_dim=self.feature_dim,
            norm_type=norm_type,
            nspk=1,
            eps=self.eps,
        )

    @property
    def num_blocks(self) -> int:
        return self.separator.num_blocks

    def _validate_spectral_inputs(
        self, spectral_repr: Tensor, reference_spec: Tensor
    ) -> None:
        if spectral_repr.ndim != 4:
            raise ValueError(
                "spectral_repr must have shape [R, D, F, T], "
                f"got {tuple(spectral_repr.shape)}"
            )
        if not spectral_repr.is_floating_point():
            raise TypeError("spectral_repr must be a floating-point real tensor")
        if spectral_repr.shape[1] != self.input_dim:
            raise ValueError(
                f"spectral_repr D must equal input_dim={self.input_dim}, "
                f"got {spectral_repr.shape[1]}"
            )
        expected_frequency_bins = self.n_fft // 2 + 1
        if spectral_repr.shape[2] != expected_frequency_bins:
            raise ValueError(
                f"spectral_repr F must equal n_fft//2+1={expected_frequency_bins}, "
                f"got {spectral_repr.shape[2]}"
            )
        if reference_spec.ndim != 3:
            raise ValueError(
                "reference_spec must have shape [R, F, T], "
                f"got {tuple(reference_spec.shape)}"
            )
        if not reference_spec.is_complex():
            raise TypeError("reference_spec must be a complex tensor")
        if tuple(reference_spec.shape) != (
            spectral_repr.shape[0],
            spectral_repr.shape[2],
            spectral_repr.shape[3],
        ):
            raise ValueError(
                "reference_spec [R, F, T] must align with spectral_repr, got "
                f"{tuple(reference_spec.shape)} and {tuple(spectral_repr.shape)}"
            )
        if reference_spec.device != spectral_repr.device:
            raise ValueError("spectral_repr and reference_spec must use the same device")
        if reference_spec.real.dtype != spectral_repr.dtype:
            raise TypeError(
                "reference_spec real dtype must match spectral_repr dtype, got "
                f"{reference_spec.real.dtype} and {spectral_repr.dtype}"
            )

    def _validate_latent(self, latent: Tensor, *, source: str) -> None:
        if latent.ndim != 4:
            raise ValueError(f"{source} must return [R, nband, E, T]")
        if latent.shape[1] != self.nband or latent.shape[2] != self.feature_dim:
            raise ValueError(
                f"{source} must preserve nband={self.nband} and "
                f"feature_dim={self.feature_dim}, got {tuple(latent.shape)}"
            )

    def encode(self, spectral_repr: Tensor) -> Tensor:
        """执行 band split 与 subband normalization。"""
        if spectral_repr.ndim != 4 or spectral_repr.shape[1] != self.input_dim:
            raise ValueError(
                f"spectral_repr must be [R, {self.input_dim}, F, T], "
                f"got {tuple(spectral_repr.shape)}"
            )
        expected_frequency_bins = self.n_fft // 2 + 1
        if spectral_repr.shape[2] != expected_frequency_bins:
            raise ValueError(
                f"spectral_repr F must equal {expected_frequency_bins}, "
                f"got {spectral_repr.shape[2]}"
            )
        latent = self.subband_norm(self.band_split(spectral_repr))
        self._validate_latent(latent, source="subband_norm")
        return latent

    def _resolve_block_conditions(
        self, conditions: BlockConditions
    ) -> tuple[Tensor | None, ...]:
        if conditions is None:
            return (None,) * self.num_blocks
        if isinstance(conditions, Tensor):
            return (conditions,) * self.num_blocks
        resolved = tuple(conditions)
        if len(resolved) != self.num_blocks:
            raise ValueError(
                f"block_conditions must contain {self.num_blocks} items, "
                f"got {len(resolved)}"
            )
        return resolved

    def separate(
        self,
        latent: Tensor,
        *,
        latent_fusion: LatentFusion | None = None,
        block_conditioners: Sequence[BlockConditioner | None] | None = None,
        block_conditions: BlockConditions = None,
        initial_states: InitialStates = None,
    ) -> Tensor:
        """执行一次 latent fusion，并在每个 BSNet 前提供条件注入点。"""
        self._validate_latent(latent, source="latent")
        if latent_fusion is not None:
            latent = latent_fusion(latent)
            self._validate_latent(latent, source="latent_fusion")

        if block_conditioners is None:
            conditioners: tuple[BlockConditioner | None, ...] = (None,) * self.num_blocks
        else:
            conditioners = tuple(block_conditioners)
            if len(conditioners) != self.num_blocks:
                raise ValueError(
                    f"block_conditioners must contain {self.num_blocks} items, "
                    f"got {len(conditioners)}"
                )
        conditions = self._resolve_block_conditions(block_conditions)
        if isinstance(initial_states, Sequence) and not isinstance(initial_states, dict):
            states = tuple(initial_states)
            if len(states) != self.num_blocks:
                raise ValueError(
                    f"initial_states must contain {self.num_blocks} items, got {len(states)}"
                )
        else:
            states = (initial_states,) * self.num_blocks

        for index, (conditioner, condition, initial_state) in enumerate(
            zip(conditioners, conditions, states)
        ):
            if conditioner is None and condition is not None:
                raise ValueError(
                    f"block_conditions[{index}] was supplied without a conditioner"
                )
            if conditioner is not None:
                if condition is None:
                    raise ValueError(
                        f"block_conditioners[{index}] requires a condition"
                    )
                latent = conditioner(latent, condition)
                self._validate_latent(latent, source=f"block_conditioners[{index}]")
            latent = self.separator.forward_block(
                latent,
                index,
                initial_state=initial_state,
            )
        return latent

    def render(self, latent: Tensor, reference_spec: Tensor) -> Tensor:
        """将 mask 应用于参考谱并返回 ``[R, F, T, 2]``。"""
        self._validate_latent(latent, source="latent")
        expected_shape = (
            latent.shape[0],
            self.n_fft // 2 + 1,
            latent.shape[3],
        )
        if (
            reference_spec.ndim != 3
            or not reference_spec.is_complex()
            or tuple(reference_spec.shape) != expected_shape
        ):
            raise ValueError(
                f"reference_spec must be complex {expected_shape}, "
                f"got {tuple(reference_spec.shape)}"
            )
        if reference_spec.device != latent.device:
            raise ValueError("latent and reference_spec must use the same device")
        if reference_spec.real.dtype != latent.dtype:
            raise TypeError("reference_spec real dtype must match latent dtype")
        estimated_ri = self.band_masker(latent, self.band_split(reference_spec))
        return estimated_ri[:, :, 0].permute(0, 2, 3, 1).contiguous()

    def forward(
        self,
        spectral_repr: Tensor,
        reference_spec: Tensor,
        *,
        latent_fusion: LatentFusion | None = None,
        block_conditioners: Sequence[BlockConditioner | None] | None = None,
        block_conditions: BlockConditions = None,
        initial_states: InitialStates = None,
    ) -> Tensor:
        self._validate_spectral_inputs(spectral_repr, reference_spec)
        latent = self.encode(spectral_repr)
        latent = self.separate(
            latent,
            latent_fusion=latent_fusion,
            block_conditioners=block_conditioners,
            block_conditions=block_conditions,
            initial_states=initial_states,
        )
        return self.render(latent, reference_spec)


__all__ = [
    "BSRNNSeparatorAdapter",
    "BlockConditioner",
    "InitialState",
    "InitialStates",
    "LatentFusion",
]
