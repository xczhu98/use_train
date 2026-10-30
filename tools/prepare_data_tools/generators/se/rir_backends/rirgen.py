"""rir_generator backend。"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from tools.prepare_data_tools.generators.se.rir_backends.base import (
    BaseRIRBackend,
    trim_or_pad,
)
from tools.prepare_data_tools.generators.se.rir_config import RIRConfig
from tools.prepare_data_tools.generators.se.rir_scene import sample_room_and_positions


_RIR_GENERATOR: Any | None = None


class RIRGeneratorBackend(BaseRIRBackend):
    """基于 rir_generator 的 image-source backend。"""

    name = "rirgen"

    def is_available(self) -> bool:
        return importlib.util.find_spec("rir_generator") is not None

    def availability_error(self) -> str:
        return "backend 'rirgen' requires `pip install rir_generator`"

    def _ensure_import(self):
        global _RIR_GENERATOR
        if _RIR_GENERATOR is None:
            import rir_generator as rir

            _RIR_GENERATOR = rir
        return _RIR_GENERATOR

    def generate_one(self, index: int, config: RIRConfig, gpu_id: int = 0) -> dict[str, Any]:
        if config.moving:
            raise ValueError("rirgen backend does not support moving-source RIR")
        rir = self._ensure_import()
        room_dim, source_loc, mic_loc, rt60, distance = sample_room_and_positions(
            config, index, gpu_id=gpu_id
        )
        options = config.backend_options.rirgen
        response = rir.generate(
            c=config.c,
            fs=config.fs,
            r=mic_loc,
            s=source_loc,
            L=room_dim,
            reverberation_time=rt60,
            nsample=config.n_samples,
            mtype=rir.mtype.omnidirectional,
            order=options.reflection_order,
            dim=options.dimension,
            orientation=None,
            hp_filter=options.highpass_filter,
        )
        response = np.asarray(response, dtype=np.float32)
        if response.ndim > 1:
            response = response[0]
        response = trim_or_pad(response, config.n_samples)
        return {
            "rir": response,
            "fs": config.fs,
            "rt60": rt60,
            "dist": distance,
            "room_dim": room_dim,
            "source_loc": source_loc,
            "mic_loc": mic_loc,
        }


__all__ = ["RIRGeneratorBackend"]
