"""频域 / 谱域损失。无重复注册，每类仅一个 type 名。

  MultiResSTFTLoss           — 多分辨率 SC + log 幅度 L1（SEtrain-plus）
  MultiResL1SpecLoss         — ESPNet 式时域 L1 + 多窗幅度 L1（URGENT/BSRNN）
  MultiResFreqLoss           — 多分辨率复谱 MSE + 相对幅度（se_train）
  MagnitudePlusComplexL1Loss — 幅度 L1 + 复谱 L1，单分辨率（Fast-ULCNet 论文）

接口: forward(enhanced, clean, lengths=None) -> 标量
"""

import torch
import torch.nn as nn
from core.losses.utils import (
    create_frame_mask,
    prepare_centered_stft_inputs,
    resolve_waveform_lengths,
)
from core.utils.registry import LOSS_REGISTRY


class _STFTLoss(nn.Module):
    """Single-resolution STFT loss: spectral convergence + log-magnitude L1.

    SC  = ||Y_mag - X_mag||_F / ||Y_mag||_F    (normalized Frobenius norm)
    Mag = L1(log(X_mag), log(Y_mag))           (log-domain magnitude error)

    SC 与 Mag 均先逐样本计算再按 batch 平均；逐样本 target 谱范数近零时，
    SC 返回 graph-connected zero，该样本仅保留 log-magnitude residual suppression。
    lengths 给定时: 先在有效边界 reflect 延展(输入波形须已由调用方按 lengths
    置零, 使 STFT 与 trim 后逐点一致, JAE-H02), 两项均按有效帧归约。
    """
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024):
        super().__init__()
        self.n_fft = n_fft; self.hop_length = hop_length; self.win_length = win_length
        self.register_buffer("_window", torch.hann_window(win_length), persistent=False)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x, y, lengths=None):
        (x, y), lengths, _ = prepare_centered_stft_inputs(
            x, y, lengths=lengths, n_fft=self.n_fft
        )
        kw = dict(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                  window=self._window, return_complex=True, center=True)
        xm = torch.abs(torch.stft(x.float(), **kw))
        ym = torch.abs(torch.stft(y.float(), **kw))
        mask = create_frame_mask(
            lengths,
            xm.shape[-1],
            self.n_fft,
            self.win_length,
            self.hop_length,
            xm.device,
        ).unsqueeze(1).to(xm.dtype)

        raw_target_norm = torch.linalg.vector_norm(ym * mask, dim=(1, 2))
        xm = xm.clamp(min=1e-8)
        ym = ym.clamp(min=1e-8)
        log_error = (torch.log(xm) - torch.log(ym)).abs() * mask
        denom = (mask.sum(dim=(1, 2)) * xm.shape[1]).clamp(min=1.0)
        mag = log_error.sum(dim=(1, 2)) / denom
        # vector_norm 在零向量处给出有限零次梯度；非零处与原 sqrt(sum(z²))
        # 公式完全一致，避免完美重建时 SC 的 sqrt(0) 反向产生 NaN。
        sc_numerator = torch.linalg.vector_norm((ym - xm) * mask, dim=(1, 2))
        target_norm = torch.linalg.vector_norm(ym * mask, dim=(1, 2))
        valid_sc = raw_target_norm > 1e-8
        safe_target_norm = torch.where(
            valid_sc, target_norm, torch.ones_like(target_norm)
        )
        sc = sc_numerator / safe_target_norm
        # target 谱严格为零时标准 SC 无定义，近零时分母病态；阈值分支仅关闭
        # SC，不重复计算 log-mag，并以乘零保留计算图连接。
        sc = torch.where(valid_sc, sc, sc_numerator * 0.0)
        return (sc + mag).mean()


@LOSS_REGISTRY.register("MultiResSTFTLoss")
class MultiResSTFTLoss(nn.Module):
    """Multi-resolution STFT loss (Multi-Resolution STFT Loss).

    Zip ``fft_sizes[i]``, ``hop_sizes[i]`` and ``win_lengths[i]`` into one
    resolution, compute one SC + log-magnitude L1 term, then average all
    resolution terms. Each term evaluates enhanced and clean with the same STFT
    geometry. Different resolutions capture long-term low-frequency structure
    and high-frequency transients.
    Source: SEtrain-plus loss_factory.py

    Note: Ensure win_length <= n_fft for each resolution to avoid STFT errors.
    The default fft/win schedules (2048, 1024, 512)/(1200, 600, 240)
    satisfy this constraint.
    """
    def __init__(self, fft_sizes=(2048, 1024, 512), hop_sizes=(240, 120, 50), win_lengths=(1200, 600, 240)):
        super().__init__()
        schedules = {}
        for name, values in (
            ("fft_sizes", fft_sizes),
            ("hop_sizes", hop_sizes),
            ("win_lengths", win_lengths),
        ):
            try:
                schedule = tuple(values)
            except TypeError as exc:
                raise TypeError(f"{name} schedule must be iterable") from exc
            for index, value in enumerate(schedule):
                if type(value) is not int:
                    raise TypeError(
                        f"{name}[{index}] must be an exact positive integer, got {value!r}"
                    )
                if value <= 0:
                    raise ValueError(
                        f"{name}[{index}] must be a positive integer, got {value!r}"
                    )
            schedules[name] = schedule

        schedule_lengths = {len(values) for values in schedules.values()}
        if schedule_lengths == {0}:
            raise ValueError("MultiResSTFTLoss schedules must not be empty")
        if len(schedule_lengths) != 1:
            raise ValueError(
                "MultiResSTFTLoss schedules must have identical lengths; "
                f"got fft_sizes={len(schedules['fft_sizes'])}, "
                f"hop_sizes={len(schedules['hop_sizes'])}, "
                f"win_lengths={len(schedules['win_lengths'])}"
            )

        resolutions = tuple(
            zip(
                schedules["fft_sizes"],
                schedules["hop_sizes"],
                schedules["win_lengths"],
            )
        )
        for index, (fft, hop, win) in enumerate(resolutions):
            if win > fft:
                raise ValueError(
                    "MultiResSTFTLoss resolution "
                    f"{index} must satisfy win_length <= n_fft; "
                    f"got hop_length={hop}, win_length={win}, n_fft={fft}"
                )
        self.losses = nn.ModuleList(
            [_STFTLoss(fft, hop, win) for fft, hop, win in resolutions]
        )

    def forward(self, enhanced, clean, lengths=None):
        return sum(l(enhanced, clean, lengths) for l in self.losses) / len(self.losses)


@LOSS_REGISTRY.register("MultiResL1SpecLoss")
class MultiResL1SpecLoss(nn.Module):
    """Multi-resolution L1 magnitude spec + time-domain L1 loss (Multi-Resolution L1 Spec Loss).

    Reproduces espnet2 ``MultiResL1SpecLoss``, used by urgent2026 BSRNN baseline.

    Formula (strict espnet2/URGENT2026 alignment):
      scaling_factor = sum(est*ref) / (sum(est^2) + eps)   [per-sample optimal LS scaling]
      L1_time  = sum(|est*scaling_factor - ref|)            (time-domain L1, sum reduction)
      L1_mag_i = sum(||STFT_i(est*scaling_factor)| - |STFT_i(ref)||)
      loss = time_domain_weight * L1_time + (1 - time_domain_weight) * mean(L1_mag_i)

    Options:
      normalize_variance=True: normalize est and ref by std first (then apply scaling)
      gama != 1.0: compress magnitude as (mag + eps)^gama

    Reference:
      Lu et al., "Towards Low-Distortion Multi-Channel Speech Enhancement:
      The ESPNET-Se Submission to the L3DAS22 Challenge", ICASSP 2022.

    ``lengths`` 给定时，方差、缩放及时域 L1 仅使用有效 samples；每个分辨率
    使用统一的 centered-STFT 边界准备和有效帧 mask。未提供 ``lengths`` 时仍与
    espnet2 原始全长 sum 公式一致。
    """

    def __init__(self, window_sz=(256, 512, 768, 1024), hop_sz=None,
                 eps=1e-6, time_domain_weight=0.5,
                 normalize_variance=True, gama=1.0):
        super().__init__()
        try:
            self.window_sz = tuple(window_sz)
        except TypeError as exc:
            raise TypeError("window_sz must be an iterable of positive integers") from exc
        self.hop_sz = (
            tuple(w // 2 for w in self.window_sz)
            if hop_sz is None
            else tuple(hop_sz)
        )
        if not self.window_sz or len(self.window_sz) != len(self.hop_sz):
            raise ValueError("window_sz and hop_sz must be non-empty and have equal lengths")
        for name, values in (("window_sz", self.window_sz), ("hop_sz", self.hop_sz)):
            for index, value in enumerate(values):
                if type(value) is not int or value <= 0:
                    raise ValueError(
                        f"{name}[{index}] must be an exact positive integer, got {value!r}"
                    )
        if type(eps) not in (int, float) or eps <= 0:
            raise ValueError(f"eps must be positive, got {eps!r}")
        if type(gama) not in (int, float) or gama <= 0:
            raise ValueError(f"gama must be positive, got {gama!r}")
        self.eps = eps
        self.time_domain_weight = time_domain_weight
        self.normalize_variance = normalize_variance
        self.gama = gama
        for w in self.window_sz:
            self.register_buffer(f"_win_{w}", torch.hann_window(w), persistent=False)

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, enhanced, clean, lengths=None):
        enhanced = enhanced.float()
        clean = clean.float()
        has_explicit_lengths = lengths is not None
        lengths, sample_mask = resolve_waveform_lengths(clean, lengths)
        sample_mask = sample_mask.to(clean.dtype)
        valid_count = lengths.unsqueeze(1).to(clean.dtype).clamp(min=1.0)
        enhanced = enhanced * sample_mask
        clean = clean * sample_mask
        if self.normalize_variance:
            if has_explicit_lengths:
                mean_c = clean.sum(dim=1, keepdim=True) / valid_count
                mean_e = enhanced.sum(dim=1, keepdim=True) / valid_count
                var_c = (
                    (clean - mean_c).square() * sample_mask
                ).sum(dim=1, keepdim=True) / valid_count + self.eps
                var_e = (
                    (enhanced - mean_e).square() * sample_mask
                ).sum(dim=1, keepdim=True) / valid_count + self.eps
            else:
                # 无 lengths 时保持 espnet2 原始全长方差公式。
                var_c = torch.var(clean, dim=1, keepdim=True, unbiased=False) + self.eps
                var_e = torch.var(enhanced, dim=1, keepdim=True, unbiased=False) + self.eps
            denom_c = torch.sqrt(var_c)
            denom_e = torch.sqrt(var_e)
            denom_c = torch.where(var_c > self.eps, denom_c, denom_c.new_tensor(1.0))
            denom_e = torch.where(var_e > self.eps, denom_e, denom_e.new_tensor(1.0))
            clean = clean / denom_c
            enhanced = enhanced / denom_e

        # Scale-invariant: optimal LS scaling of estimate toward ref (espnet2/URGENT2026)
        scaling = (torch.sum(enhanced * clean, dim=-1, keepdim=True)
                   / (torch.sum(enhanced ** 2, dim=-1, keepdim=True) + self.eps))
        enhanced_scaled = enhanced * scaling

        time_loss = torch.sum(
            torch.abs(enhanced_scaled - clean) * sample_mask, dim=-1
        )

        spec_loss = torch.zeros_like(time_loss)
        for w, h in zip(self.window_sz, self.hop_sz):
            (enh_stft, clean_stft), _, _ = prepare_centered_stft_inputs(
                enhanced_scaled, clean, lengths=lengths, n_fft=w
            )
            win = getattr(self, f"_win_{w}")
            kw = dict(n_fft=w, hop_length=h, win_length=w, window=win,
                      center=True, return_complex=True)
            cln_mag = torch.abs(torch.stft(clean_stft, **kw))
            enh_mag = torch.abs(torch.stft(enh_stft, **kw))
            if self.gama != 1.0:
                cln_mag = (cln_mag + self.eps) ** self.gama
                enh_mag = (enh_mag + self.eps) ** self.gama
            frame_mask = create_frame_mask(
                lengths, cln_mag.shape[-1], w, w, h, cln_mag.device
            ).unsqueeze(1).to(cln_mag.dtype)
            spec_loss = spec_loss + torch.sum(
                torch.abs(enh_mag - cln_mag) * frame_mask, dim=(1, 2)
            )

        spec_loss = spec_loss / len(self.window_sz)
        return (self.time_domain_weight * time_loss
                + (1 - self.time_domain_weight) * spec_loss).mean()


@LOSS_REGISTRY.register("MultiResFreqLoss")
class MultiResFreqLoss(nn.Module):
    """Multi-resolution complex spectrum L2 + relative magnitude error (Multi-Resolution Frequency Loss).

    Per resolution computes:
      cspec = mean(|STFT(enh) - STFT(cln)|^2)             (complex spectrum MSE)
      rel   = mean(((|enh| - |cln|) / (|enh| + |cln|))^2) (relative magnitude error)
    Source: se_train monaural_loss.py
    """
    def __init__(self, fft_sizes=(64, 128, 256, 512, 1024, 2048)):
        super().__init__()
        try:
            self.fft_sizes = tuple(fft_sizes)
        except TypeError as exc:
            raise TypeError("fft_sizes must be an iterable of positive integers") from exc
        if not self.fft_sizes:
            raise ValueError("fft_sizes must not be empty")
        for index, n_fft in enumerate(self.fft_sizes):
            if type(n_fft) is not int or n_fft <= 0:
                raise ValueError(
                    f"fft_sizes[{index}] must be an exact positive integer, got {n_fft!r}"
                )
        for n in self.fft_sizes:
            self.register_buffer(f"_win_{n}", torch.hann_window(n), persistent=False)

    @torch.amp.autocast("cuda", enabled=False)
    def _one(self, enhanced, clean, lengths, n_fft):
        (enhanced, clean), lengths, _ = prepare_centered_stft_inputs(
            enhanced, clean, lengths=lengths, n_fft=n_fft
        )
        w = getattr(self, f"_win_{n_fft}")
        kw = dict(n_fft=n_fft, hop_length=n_fft//2, win_length=n_fft, window=w,
                  center=True, return_complex=True)
        es = torch.stft(enhanced.float(), **kw)
        cs = torch.stft(clean.float(), **kw)
        em, cm = torch.abs(es).clamp(1e-8), torch.abs(cs).clamp(1e-8)
        frame_mask = create_frame_mask(
            lengths, es.shape[-1], n_fft, n_fft, n_fft // 2, es.device
        ).unsqueeze(1).to(es.real.dtype)
        n_bins = es.shape[1]
        n_valid_bins = (frame_mask.sum() * n_bins).clamp(min=1.0)
        complex_error = (
            torch.view_as_real(es) - torch.view_as_real(cs)
        ).square() * frame_mask.unsqueeze(-1)
        cspec = complex_error.sum() / (2 * n_valid_bins)
        rel = ((((em - cm) / (em + cm + 1e-7)) ** 2) * frame_mask).sum()
        rel = rel / n_valid_bins
        return cspec + rel

    def forward(self, enhanced, clean, lengths=None):
        return sum(
            self._one(enhanced, clean, lengths, n) for n in self.fft_sizes
        ) / len(self.fft_sizes)


@LOSS_REGISTRY.register("MagnitudePlusComplexL1Loss")
class MagnitudePlusComplexL1Loss(nn.Module):
    """单分辨率 STFT：幅度 L1 + 复谱 L1（Fast-ULCNet 论文式）。

    公式: (1/(T*F)) * sum( |S|-|Ŝ| + |S-Ŝ| )，S=clean、Ŝ=enhanced。
    默认 n_fft=512, hop=256, win=512；支持 lengths 有效帧掩码。
    YAML 使用 type: MagnitudePlusComplexL1Loss 即可。
    """
    def __init__(self, n_fft: int = 512, hop_length: int = 256, win_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("_window", torch.hann_window(win_length), persistent=False)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        enhanced = enhanced.float()
        clean = clean.float()
        (enhanced, clean), lengths, _ = prepare_centered_stft_inputs(
            enhanced, clean, lengths=lengths, n_fft=self.n_fft
        )
        kw = dict(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window,
            return_complex=True,
            center=True,
        )
        S = torch.stft(clean, **kw)
        S_hat = torch.stft(enhanced, **kw)
        per_bin = (torch.abs(S) - torch.abs(S_hat)).abs() + torch.abs(S - S_hat)
        frame_mask = create_frame_mask(
            lengths,
            per_bin.shape[-1],
            self.n_fft,
            self.win_length,
            self.hop_length,
            per_bin.device,
        ).unsqueeze(1).to(per_bin.dtype)
        n_valid = (frame_mask.sum() * per_bin.shape[1]).clamp(min=1.0)
        return (per_bin * frame_mask).sum() / n_valid
