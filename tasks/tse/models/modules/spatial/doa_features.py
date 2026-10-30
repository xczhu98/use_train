# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
"""WeSep DOA 条件特征的 canonical cue 移植。

来源：WeSep commit ``dc11cb18`` 的
``wesep/modules/spatial/doa_features.py``。输入方向已由数据侧按真实 sample
anchor 对齐为 ``[B,2,T]``；本模块不做按点数 resize。
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
import torch.nn as nn

from tasks.tse.models.modules.spatial.pos_encoding import CycPosEncoding


_FUSIONS = frozenset({"additive", "concat", "multiply", "FiLM"})


def _config(
    value: Mapping[str, object],
    *,
    allowed: frozenset[str],
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} config must be a mapping")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{name} config contains unknown keys: {unknown}")
    return dict(value)


def _positive_int(value: object, name: str, default: int) -> int:
    resolved = default if value is None else value
    if type(resolved) is not int or resolved <= 0:
        raise ValueError(f"{name} must be a positive exact int")
    return resolved


def _positive_float(value: object, name: str, default: float) -> float:
    resolved = default if value is None else value
    if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(resolved)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _bool(value: object, name: str, default: bool) -> bool:
    resolved = default if value is None else value
    if type(resolved) is not bool:
        raise TypeError(f"{name} must be a bool")
    return resolved


def _validate_inputs(
    direction_rad: torch.Tensor,
    present: torch.Tensor,
) -> tuple[int, int]:
    if (
        not isinstance(direction_rad, torch.Tensor)
        or direction_rad.ndim != 3
        or direction_rad.shape[1] != 2
        or not direction_rad.dtype.is_floating_point
    ):
        raise ValueError("direction_rad must be floating-point [B,2,T]")
    batch, _, frames = direction_rad.shape
    if not isinstance(present, torch.Tensor) or tuple(present.shape) != (batch,):
        raise ValueError(f"present must have shape [{batch}]")
    if present.dtype != torch.bool or present.device != direction_rad.device:
        raise TypeError("present must be torch.bool on the direction device")
    if not bool(torch.isfinite(direction_rad).all().item()):
        raise ValueError("direction_rad must contain finite radians")
    return batch, frames


class CyclicDOAFeature(nn.Module):
    """输出 ``[B,1,mix_dim,T]``，供 Composer 在 BSRNN band 特征上融合。"""

    def __init__(
        self,
        config: Mapping[str, object],
        *,
        mix_dim: int,
    ):
        super().__init__()
        if type(mix_dim) is not int or mix_dim <= 0:
            raise ValueError("mix_dim must be a positive exact int")
        resolved = _config(
            config,
            allowed=frozenset({"alpha", "embedding_dim", "use_elevation", "fusion"}),
            name="cyc_doaemb",
        )
        self.use_elevation = _bool(
            resolved.get("use_elevation"), "cyc_doaemb.use_elevation", True
        )
        embedding_dim = _positive_int(
            resolved.get("embedding_dim"), "cyc_doaemb.embedding_dim", 40
        )
        alpha = _positive_float(resolved.get("alpha"), "cyc_doaemb.alpha", 20.0)
        fusion = resolved.get("fusion", "multiply")
        if not isinstance(fusion, str) or fusion not in _FUSIONS:
            raise ValueError(f"cyc_doaemb.fusion must be one of {sorted(_FUSIONS)}")
        self.fusion = fusion
        self.position_encoding = CycPosEncoding(embedding_dim, alpha)
        input_dim = embedding_dim * (2 if self.use_elevation else 1)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, mix_dim),
            nn.LayerNorm(mix_dim),
            nn.PReLU(),
        )
        self.output_dim = mix_dim

    def compute(self, direction_rad: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        batch, frames = _validate_inputs(direction_rad, present)
        encoded = [self.position_encoding(direction_rad[:, 0])]
        if self.use_elevation:
            encoded.append(self.position_encoding(direction_rad[:, 1]))
        feature = self.projection(torch.cat(encoded, dim=-1))
        feature = feature.transpose(1, 2).unsqueeze(1)
        feature = feature * present.view(batch, 1, 1, 1).to(feature.dtype)
        if tuple(feature.shape) != (batch, 1, self.output_dim, frames):
            raise RuntimeError("cyclic DOA feature produced an invalid shape")
        return feature


class InitStateDOAFeature(nn.Module):
    """生成与 ``BSRNN_Separator.forward(initial_states=...)`` 一致的状态。

    每个 block 返回 ``{"band": (h,c), "communication": (h,c)}``；四个
    tensor 均为 ``[B,2*mix_dim]``。track 使用已对齐 STFT frame 的首帧方向。
    """

    def __init__(
        self,
        config: Mapping[str, object],
        *,
        mix_dim: int,
        num_layers: int,
    ):
        super().__init__()
        if type(mix_dim) is not int or mix_dim <= 0:
            raise ValueError("mix_dim must be a positive exact int")
        if type(num_layers) is not int or num_layers <= 0:
            raise ValueError("num_layers must be a positive exact int")
        resolved = _config(
            config,
            allowed=frozenset({"alpha", "embedding_dim", "use_elevation"}),
            name="initstate_emb",
        )
        self.use_elevation = _bool(
            resolved.get("use_elevation"), "initstate_emb.use_elevation", True
        )
        embedding_dim = _positive_int(
            resolved.get("embedding_dim"), "initstate_emb.embedding_dim", 40
        )
        alpha = _positive_float(resolved.get("alpha"), "initstate_emb.alpha", 20.0)
        self.position_encoding = CycPosEncoding(embedding_dim, alpha)
        input_dim = embedding_dim * (2 if self.use_elevation else 1)
        self.state_dim = mix_dim * 2
        self.projections = nn.ModuleList(
            nn.Linear(input_dim, self.state_dim * 4) for _ in range(num_layers)
        )

    def compute(
        self,
        direction_rad: torch.Tensor,
        present: torch.Tensor,
    ) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], ...]:
        batch, _ = _validate_inputs(direction_rad, present)
        encoded = [self.position_encoding(direction_rad[:, 0, 0])]
        if self.use_elevation:
            encoded.append(self.position_encoding(direction_rad[:, 1, 0]))
        direction = torch.cat(encoded, dim=-1)
        mask = present.view(batch, 1).to(direction.dtype)
        states = []
        for projection in self.projections:
            band_h, band_c, communication_h, communication_c = projection(direction).chunk(
                4, dim=-1
            )
            band_h = band_h * mask
            band_c = band_c * mask
            communication_h = communication_h * mask
            communication_c = communication_c * mask
            states.append(
                {
                    "band": (band_h, band_c),
                    "communication": (communication_h, communication_c),
                }
            )
        return tuple(states)


__all__ = ["CyclicDOAFeature", "InitStateDOAFeature"]
