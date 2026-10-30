"""gpuRIR backend。"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from tools.prepare_data_tools.generators.se.rir_backends.base import (
    BaseRIRBackend,
    trim_or_pad,
)
from tools.prepare_data_tools.generators.se.rir_config import RIRConfig
from tools.prepare_data_tools.generators.se.rir_scene import (
    sample_moving_scene,
    sample_room_and_positions,
)


_GPURIR: Any | None = None


class GPURIRBackend(BaseRIRBackend):
    """基于 gpuRIR 的 GPU image-source backend。"""

    name = "gpurir"

    def is_available(self) -> bool:
        return importlib.util.find_spec("gpuRIR") is not None

    def availability_error(self) -> str:
        return "backend 'gpurir' requires `pip install gpuRIR`"

    def _ensure_import(self):
        global _GPURIR
        if _GPURIR is None:
            import gpuRIR

            _GPURIR = gpuRIR
        return _GPURIR

    def generate_one(self, index: int, config: RIRConfig, gpu_id: int = 0) -> dict[str, Any]:
        gpu_rir = self._ensure_import()
        if config.moving:
            room_dim, trajectory, mic_loc, rt60, distance, speed = sample_moving_scene(
                config, index, gpu_id=gpu_id
            )
            mic_positions = np.array([mic_loc], dtype=np.float32)
        else:
            room_dim, source_loc, mic_loc, rt60, distance = sample_room_and_positions(
                config, index, gpu_id=gpu_id
            )
            source_positions = np.array([source_loc], dtype=np.float32)
            mic_positions = np.array([mic_loc], dtype=np.float32)

        from core.data.simulation.gpurir_backend import generate_rir_group

        options = config.backend_options.gpurir
        def simulate(position):
            # 离线生成器复用 core 的 GPU 调用；保留现有几何采样和输出格式。
            plan = {"room_dim": room_dim, "mic_loc": mic_loc, "source_loc": position.tolist(),
                    "rt60": rt60, "num_samples": config.n_samples, "sample_rate": config.fs,
                    "diffuse_db": options.diffuse_transition_db, "sound_speed": config.c,
                    "speaker_pattern": options.speaker_pattern,
                    "microphone_pattern": options.microphone_pattern,
                    "speaker_orientation": None if options.speaker_orientation is None else [options.speaker_orientation],
                    "microphone_orientation": None if options.microphone_orientation is None else [options.microphone_orientation]}
            responses, _ = generate_rir_group(plan, gpu_rir)
            return responses[0]

        if config.moving:
            response = np.stack(
                [simulate(trajectory[t : t + 1]) for t in range(trajectory.shape[0])]
            )
        else:
            response = simulate(source_positions)
        response = trim_or_pad(response, config.n_samples)
        result = {
            "rir": response,
            "fs": config.fs,
            "rt60": rt60,
            "dist": distance,
            "room_dim": room_dim,
            "mic_loc": mic_loc,
        }
        if config.moving:
            result.update(
                {
                    "source_loc": trajectory[0].tolist(),
                    "moving": True,
                    "n_segments": int(trajectory.shape[0]),
                    "segment_sec": float(config.segment_sec),
                    "trajectory": np.asarray(trajectory, dtype=np.float32),
                    "speed": float(speed),
                }
            )
            distances = np.linalg.norm(
                np.asarray(trajectory, dtype=np.float32)
                - np.asarray(mic_loc, dtype=np.float32),
                axis=1,
            )
            result["dist_min"] = float(distances.min())
            result["dist_max"] = float(distances.max())
        else:
            result["source_loc"] = source_loc
        return result


__all__ = ["GPURIRBackend"]
