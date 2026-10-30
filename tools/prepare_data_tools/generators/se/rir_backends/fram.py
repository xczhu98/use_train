"""FRAM-RIR backend。"""

from __future__ import annotations

import importlib.util
import math
from contextlib import contextmanager
from typing import Any

import numpy as np

from tools.prepare_data_tools.generators.se.rir_config import RIRConfig
from tools.prepare_data_tools.generators.se.rir_backends.base import (
    BaseRIRBackend,
    trim_or_pad,
)
from tools.prepare_data_tools.generators.se.rir_scene import (
    resolve_scene_seed,
    sample_room_and_positions,
)

# 模块级可选依赖缓存，避免每个 backend 实例重复 import
_TORCH: Any | None = None
_TORCHAUDIO: Any | None = None

@contextmanager
def _isolated_random_state(seed: int):
    """为 FRAM 临时设置 NumPy/Torch 随机状态，并在退出时恢复。"""
    import torch

    numpy_state = np.random.get_state()
    try:
        np.random.seed(int(seed % (1 << 32)))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed % (1 << 63)))
            yield
    finally:
        np.random.set_state(numpy_state)


def _sample_uniform_unit_vectors_torch(n_src: int, n_img: int):
    """在三维单位球面上均匀采样 ``[n_src, n_img, 3]`` 方向。"""
    import torch

    azimuth = torch.empty(n_src, n_img).uniform_(-math.pi, math.pi)
    z = torch.empty(n_src, n_img).uniform_(-1.0, 1.0)
    radius_xy = torch.sqrt(torch.clamp(1.0 - z.square(), min=0.0))
    return torch.stack(
        [radius_xy * torch.cos(azimuth), radius_xy * torch.sin(azimuth), z],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Backend: FRAM-RIR
# ---------------------------------------------------------------------------

class FRAMRIRBackend(BaseRIRBackend):
    """基于 FRAM-RIR（Fast Random Approximation of Multi-channel RIR）。

    实现参考原 rir2.py， cleaned up 以提升数值稳定性与可读性。
    """

    name = "fram"

    def is_available(self) -> bool:
        # 使用 find_spec 避免在主进程导入 CUDA/重依赖
        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("torchaudio") is not None
        )

    def availability_error(self) -> str:
        return "backend 'fram' requires PyTorch and torchaudio"

    def _ensure_import(self) -> tuple[Any, Any]:
        global _TORCH, _TORCHAUDIO
        if _TORCH is None or _TORCHAUDIO is None:
            import torch
            import torchaudio

            _TORCH = torch
            _TORCHAUDIO = torchaudio
        return _TORCH, _TORCHAUDIO

    @staticmethod
    def _cos_from_orientation(orientation_rad):
        """[..., 2] (azimuth, elevation) -> [..., 3] 方向向量。"""
        import torch

        azm = orientation_rad[..., 0]
        ele = orientation_rad[..., 1]
        return torch.stack(
            [
                torch.cos(azm) * torch.sin(ele),
                torch.sin(azm) * torch.sin(ele),
                torch.cos(ele),
            ],
            dim=-1,
        )

    @staticmethod
    def _freq_invariant_pattern(cos_theta, pattern: str = "omni"):
        import torch

        if pattern == "omni":
            return torch.ones_like(cos_theta)
        if pattern == "cardioid":
            return 0.5 + 0.5 * cos_theta
        if pattern == "bidirectional":
            return cos_theta
        if pattern == "hyper_cardioid":
            return 0.25 + 0.75 * cos_theta
        if pattern == "sub_cardioid":
            return 0.75 + 0.25 * cos_theta
        if pattern == "half_omni":
            return (cos_theta > 0).float()
        raise NotImplementedError(f"pattern {pattern}")

    def _fram_rir(
        self,
        mic_pos: np.ndarray,
        src_pos: np.ndarray,
        sr: int,
        T60: float,
        room_dim: list[float],
        c: float = 340.0,
        n_image: tuple[int, int] = (512, 2049),
        a: float = -2.0,
        b: float = 2.0,
        tau: float = 0.25,
        direct_range: tuple[float, float] = (-6, 50),
        src_pattern: str = "omni",
        mic_pattern: str = "omni",
        highpass_hz: float = 80.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """生成单声源单麦克风的 full / direct RIR。"""
        import torch
        from torchaudio.transforms import Resample
        from torchaudio.functional import highpass_biquad

        n_img = int(np.random.choice(range(n_image[0], n_image[1])))
        R = 1.0 / (2.0 * (1.0 / room_dim[0] + 1.0 / room_dim[1] + 1.0 / room_dim[2]))
        eps = float(np.finfo(np.float16).eps)

        mic_t = torch.from_numpy(mic_pos)  # [n_mic, 3]
        src_t = torch.from_numpy(src_pos)  # [n_src, 3]
        n_mic = mic_t.shape[0]
        n_src = src_t.shape[0]

        # direct distance [n_mic, n_src]
        direct_dist = torch.sqrt(
            (mic_t.unsqueeze(1) - src_t.unsqueeze(0)).pow(2).sum(-1) + eps
        )
        nearest_dist, nearest_mic_idx = direct_dist.min(0)  # [n_src]
        nearest_mic_position = mic_t[nearest_mic_idx]  # [n_src, 3]

        ns = n_mic * n_src
        ratio = 64
        sample_sr = sr * ratio
        velocity = float(c)
        if not math.isfinite(velocity) or velocity <= 0.0:
            raise ValueError(f"sound speed must be finite and positive, got {c}")
        T60_t = torch.tensor(T60, dtype=torch.float32)

        direct_idx = torch.ceil(direct_dist * sample_sr / velocity).long().view(ns)
        rir_length = int(math.ceil(sample_sr * T60))

        resample1 = Resample(sample_sr, sample_sr // int(math.sqrt(ratio)))
        resample2 = Resample(sample_sr // int(math.sqrt(ratio)), sr)

        sabine_constant = 24.0 * math.log(10.0) / velocity
        reflect_coef = torch.sqrt(
            1.0 - (1.0 - torch.exp(-sabine_constant * R / T60_t)).pow(2)
        )

        # distance ratios for image sources
        dist_range = [
            torch.linspace(
                1.0,
                velocity * T60 / max(nearest_dist[i].item(), eps) - 1.0,
                rir_length,
            )
            for i in range(n_src)
        ]
        dist_prob = torch.linspace(0.0, 1.0, rir_length)
        dist_prob /= dist_prob.sum()
        dist_select_idx = dist_prob.multinomial(
            num_samples=n_img * n_src, replacement=True
        ).view(n_src, n_img)
        dist_nearest_ratio = torch.stack(
            [dist_range[i][dist_select_idx[i]] for i in range(n_src)], dim=0
        )

        unit_3d = _sample_uniform_unit_vectors_torch(n_src, n_img)

        image2nearest_dist = nearest_dist.view(-1, 1, 1) * dist_nearest_ratio.unsqueeze(-1)
        image_position = nearest_mic_position.unsqueeze(1) + image2nearest_dist * unit_3d

        # [n_mic, n_src, n_img]
        dist = torch.sqrt(
            (mic_t.view(-1, 1, 1, 3) - image_position.unsqueeze(0)).pow(2).sum(-1) + eps
        )

        # reflection perturbation
        reflect_max = (math.log10(velocity * T60) - 3.0) / math.log10(reflect_coef.item())
        reflect_ratio = (dist / (velocity * T60)) * (reflect_max - 1.0) + 1.0
        reflect_perturb = (
            torch.FloatTensor(n_src, n_img).uniform_(a, b)
            * dist_nearest_ratio.pow(tau)
        )
        reflect_ratio = torch.maximum(
            reflect_ratio + reflect_perturb.unsqueeze(0),
            torch.ones(1),
        )

        # concat direct path
        dist = torch.cat([direct_dist.unsqueeze(2), dist], dim=2)
        reflect_ratio = torch.cat(
            [torch.zeros(n_mic, n_src, 1), reflect_ratio], dim=2
        )

        delta_idx = torch.minimum(
            torch.ceil(dist * sample_sr / velocity),
            torch.tensor(rir_length - 1, dtype=torch.float32),
        ).long()
        delta_idx = delta_idx.view(ns, -1)
        delta_decay = (reflect_coef.pow(reflect_ratio) / dist).view(ns, -1)

        # source orientation simulation
        if src_pattern != "omni":
            img_orientation_rad = torch.FloatTensor(n_src, n_img, 2).uniform_(
                -math.pi, math.pi
            )
            img_cos_theta = torch.cos(img_orientation_rad[..., 0]) * torch.cos(
                img_orientation_rad[..., 1]
            )
            img_orientation_decay = self._freq_invariant_pattern(
                img_cos_theta, pattern=src_pattern
            )

            src_orientation_azi = torch.FloatTensor(n_src).uniform_(-math.pi, math.pi)
            src_orientation_ele = torch.FloatTensor(n_src).uniform_(-math.pi, math.pi)
            src_orientation_rad = torch.stack(
                [src_orientation_azi, src_orientation_ele], dim=-1
            )

            orV_src = self._cos_from_orientation(src_orientation_rad).unsqueeze(0)
            rcv_to_src_vec = mic_t.unsqueeze(1) - src_t.unsqueeze(0)
            cos_theta = (rcv_to_src_vec * orV_src).sum(-1)
            cos_theta /= torch.sqrt(rcv_to_src_vec.pow(2).sum(-1))
            cos_theta /= torch.sqrt(orV_src.pow(2).sum(-1))
            src_orientation_decay = self._freq_invariant_pattern(
                cos_theta, pattern=src_pattern
            )

            delta_decay_view = delta_decay.view(n_mic, n_src, -1)
            delta_decay_view[:, :, 0] *= src_orientation_decay
            delta_decay_view[:, :, 1:] *= img_orientation_decay.unsqueeze(0)
            delta_decay = delta_decay_view.view(ns, -1)

        if mic_pattern != "omni":
            mic_orientation_rad = torch.stack(
                [torch.zeros(n_mic), torch.zeros(n_mic)], dim=-1
            )
            all_src_img_pos = torch.cat(
                [src_t.unsqueeze(1), image_position], dim=1
            )  # [n_src, 1+n_img, 3]
            orV_mic = self._cos_from_orientation(mic_orientation_rad).view(
                -1, 1, 1, 3
            )
            img_to_rcv_vec = all_src_img_pos.unsqueeze(0) - mic_t.unsqueeze(1).unsqueeze(1)
            cos_theta = (img_to_rcv_vec * orV_mic).sum(-1)
            cos_theta /= torch.sqrt(img_to_rcv_vec.pow(2).sum(-1))
            cos_theta /= torch.sqrt(orV_mic.pow(2).sum(-1))
            mic_orientation_decay = self._freq_invariant_pattern(
                cos_theta, pattern=mic_pattern
            )
            delta_decay = (delta_decay.view(n_mic, n_src, -1) * mic_orientation_decay).view(
                ns, -1
            )

        # accumulate impulses
        rir = torch.zeros(ns, rir_length)
        for i in range(ns):
            rir[i] = torch.bincount(
                delta_idx[i], weights=delta_decay[i], minlength=rir_length
            ).float()

        # direct path mask
        direct_mask = torch.zeros(ns, rir_length).float()
        half_win_ms = (direct_range[1] - direct_range[0]) / 2.0 * sample_sr / 1000.0
        for i in range(ns):
            center = direct_idx[i].item()
            start = max(center + int(sample_sr * direct_range[0] / 1000.0), 0)
            end = min(center + int(sample_sr * direct_range[1] / 1000.0), rir_length)
            direct_mask[i, start:end] = 1.0

        rir_direct = rir * direct_mask
        all_rir = torch.stack([rir, rir_direct], dim=1).view(ns * 2, -1)
        rir_down = resample1(all_rir)
        rir_hp = highpass_biquad(
            rir_down, sample_sr // int(math.sqrt(ratio)), float(highpass_hz)
        )
        rir_out = resample2(rir_hp).float().view(n_mic, n_src, 2, -1)

        full = rir_out[:, :, 0, :].data.numpy()
        direct = rir_out[:, :, 1, :].data.numpy()
        return full, direct

    def generate_one(self, idx: int, cfg: RIRConfig, gpu_id: int = 0) -> dict[str, Any]:
        if cfg.moving:
            raise ValueError("fram backend does not support moving-source RIR (use gpurir)")
        self._ensure_import()

        room_dim, source_loc, mic_loc, rt60_tgt, near_mic_dist = sample_room_and_positions(
            cfg, idx, gpu_id=gpu_id
        )

        mic_pos = np.array([mic_loc], dtype=np.float32)  # [1, 3]
        src_pos = np.array([source_loc], dtype=np.float32)  # [1, 3]

        with _isolated_random_state(resolve_scene_seed(cfg, idx)):
            options = cfg.backend_options.fram
            perturbation_range = options.reflection_perturbation_range
            full, _ = self._fram_rir(
                mic_pos=mic_pos,
                src_pos=src_pos,
                sr=cfg.fs,
                T60=rt60_tgt,
                room_dim=room_dim,
                c=cfg.c,
                src_pattern="omni",
                mic_pattern="omni",
                n_image=options.n_image_range,
                a=float(perturbation_range[0]),
                b=float(perturbation_range[1]),
                tau=options.reflection_tau,
                direct_range=options.direct_window_range_ms,
                highpass_hz=options.highpass_cutoff_hz,
            )

        h = trim_or_pad(full[0, 0, :], cfg.n_samples)
        return {
            "rir": h,
            "fs": cfg.fs,
            "rt60": rt60_tgt,
            "dist": near_mic_dist,
            "room_dim": room_dim,
            "source_loc": source_loc,
            "mic_loc": mic_loc,
        }
