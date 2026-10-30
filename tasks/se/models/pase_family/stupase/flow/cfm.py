# Copyright (c) 2025 ASLP-lab. MIT license.
# Source: https://github.com/ASLP-lab/SenSE/blob/main/src/sense/model/cfm.py
# Ported into use_train framework for StuPASE reproduction.
#
# ein notation:
# b - batch, n - sequence, nt - text sequence, nw - raw wave length, d - dimension

from __future__ import annotations

import logging
from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from ...common.utils import load_artifact_module
from .dit import DiT
from .modules import MelSpec
from .utils import default, exists, lens_to_mask, mask_from_frac_lengths

logger = logging.getLogger(__name__)


class CFM(nn.Module):
    """Conditional Flow Matching acoustic-enhancement module for StuPASE.

    ``forward()`` computes the training-time conditional-flow-matching MSE loss
    directly (used by ``StuPASEFlowTrainModel`` + ``DualInputLossTrainer``).
    ``sample()`` performs classifier-free-guidance ODE sampling (inference only,
    used by the top-level ``StuPASE`` model).
    """

    def __init__(
        self,
        dit_kwargs: dict = dict(),
        mel_spec_kwargs: dict = dict(),
        odeint_kwargs: dict = dict(method="euler"),
        sigma=0.0,
        audio_drop_prob=0.5,
        noisy_audio_drop_prob=0.3,
        text_drop_prob=0.3,
        cond_drop_prob=0.2,
        frac_lengths_mask=(0.7, 1.0),
        frac_lengths_mask_noisy=(0.5, 1.0),
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask
        self.frac_lengths_mask_noisy = frac_lengths_mask_noisy

        # mel spec
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        num_channels = self.mel_spec.n_mel_channels
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.noisy_audio_drop_prob = noisy_audio_drop_prob
        self.text_drop_prob = text_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = DiT(**dit_kwargs)
        self.dim = self.transformer.dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

    @classmethod
    def from_pretrained(cls, ckpt_path: str, frozen: bool = True) -> "CFM":
        """加载 ``{cfg, generator|model}`` 形式的 CFM 发布 artifact。

        加载语义统一走 ``common.utils.load_artifact_module``
        （weights_only=True + key 匹配报告，任何 missing/unexpected 都 fail closed）。
        """
        return load_artifact_module(cls, ckpt_path, frozen=frozen)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond,
        cond_noisy,
        text,
        duration=None,
        *,
        lens=None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=6000,
        vocoder: Callable = None,
        no_ref_audio=False,
        no_noisy_audio=False,
        no_token=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        if cond_noisy.ndim == 2:
            cond_noisy = self.mel_spec(cond_noisy)
            cond_noisy = cond_noisy.permute(0, 2, 1)
            assert cond_noisy.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)
        cond_noisy = cond_noisy.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        _, noisy_cond_seq_len = cond_noisy.shape[:2]
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if no_ref_audio:
            cond_seq_len = 0

        if duration is None:
            duration = cond_seq_len + noisy_cond_seq_len

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        if no_ref_audio:
            cond = F.pad(cond, (0, 0, 0, max_duration - cond.shape[1]), value=0.0)
        else:
            cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)

        cond_noisy = F.pad(cond_noisy, (0, 0, cond_seq_len, max_duration - cond_seq_len - noisy_cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)
        if no_noisy_audio:
            cond_noisy = torch.zeros_like(cond_noisy)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up; single-utterance inference needs no mask
            mask = None

        def fn(t, x):
            pred = self.transformer(
                x=x, cond=step_cond, cond_noisy=cond_noisy, text=text, time=t, mask=mask,
                drop_audio_cond=False, drop_noisy_audio_cond=False,
                drop_text=True if no_token else False, cache=True,
            )

            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, cond_noisy=cond_noisy, text=text, time=t, mask=mask,
                drop_audio_cond=True, drop_noisy_audio_cond=True, drop_text=True, cache=True,
            )

            return pred + (pred - null_pred) * cfg_strength

        # 用局部 Generator 固定种子：同批每项独立采样，且不改写全局 RNG 状态
        # （原实现在循环内 manual_seed，导致同批各项初始噪声逐项相同）。
        generator = (
            torch.Generator(device=self.device).manual_seed(int(seed))
            if exists(seed)
            else None
        )
        y0 = [
            torch.randn(
                dur,
                self.num_channels,
                device=self.device,
                dtype=step_cond.dtype,
                generator=generator,
            )
            for dur in duration
        ]
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # odeint 中途异常（OOM/中断）也必须清缓存，否则下一次 sample() 会静默
        # 复用上一次的 text 嵌入，且缓存张量常驻内存。
        try:
            trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        finally:
            self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp,
        noisy_inp,
        text,
        *,
        lens=None,
        noise_scheduler: str | None = None,
    ):
        # handle raw wave input (auto mel-extraction)
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        if noisy_inp.ndim == 2:
            noisy_inp = self.mel_spec(noisy_inp)
            noisy_inp = noisy_inp.permute(0, 2, 1)
            assert noisy_inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _sigma1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        frac_lengths_noisy = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask_noisy)
        rand_span_mask_noisy = mask_from_frac_lengths(lens, frac_lengths_noisy)

        if exists(mask):
            rand_span_mask &= mask
            rand_span_mask_noisy &= mask

        # x1 is mel, x0 is gaussian noise
        x1 = inp
        x0 = torch.randn_like(x1)

        time = torch.rand((batch,), dtype=dtype, device=self.device)

        # sample xt (phi_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        phi = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
        cond_noisy = torch.where(rand_span_mask_noisy[..., None], torch.zeros_like(noisy_inp), noisy_inp)

        # noisy/text 独立 drop；cond_drop_prob 是无条件 CFG 的 drop-all 概率。
        drop_audio_cond = random() < self.audio_drop_prob
        drop_noisy_audio_cond = random() < self.noisy_audio_drop_prob
        drop_text = random() < self.text_drop_prob
        if random() < self.cond_drop_prob:  # p_uncond: classifier-free guidance 无条件分支，drop all
            drop_audio_cond = True
            drop_noisy_audio_cond = True
            drop_text = True

        pred = self.transformer(
            x=phi, cond=cond, cond_noisy=cond_noisy, text=text, time=time,
            drop_audio_cond=drop_audio_cond, drop_noisy_audio_cond=drop_noisy_audio_cond, drop_text=drop_text,
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred
