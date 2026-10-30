"""MetricGAN Composite Trainer with PESQ injection and multi-term generator loss.

Supports the training paradigm used by MP-SENet, MambAttention, and RWSAMamba-UNet:
- MetricGAN discriminator trained against real PESQ scores
- Generator composite loss：压缩 mag/complex/consistency + phase + waveform time

Usage in YAML:
    trainer:
      type: MetricGANCompositeTrainer
      use_pesq_injection: true
      pesq_sr: 16000
      loss_mag_weight: 0.9
      loss_pha_weight: 0.3
      loss_com_weight: 0.1
      loss_time_weight: 0.2
      loss_stft_weight: 0.1
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from core.losses.model_specific.metricgan import MPSEnetCompositeLoss
from tasks.se.trainer.gan_trainer import GANTrainer

logger = logging.getLogger(__name__)


def _load_pesq_backend():
    """加载 PESQ 实现；启用注入时禁止静默退化为普通 LSGAN。"""
    try:
        from pesq import pesq
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "trainer.use_pesq_injection=true requires the optional 'pesq' "
            "package, but it could not be imported"
        ) from exc
    if not callable(pesq):
        raise RuntimeError("the imported 'pesq.pesq' backend is not callable")
    return pesq


class MetricGANCompositeTrainer(GANTrainer):
    """Extended GANTrainer for MetricGAN + composite loss models.

    Supports MP-SENet, MambAttention, RWSAMamba-UNet and similar architectures.
    The specific Generator is selected via ``model.generator.type`` in config.

    Only overrides hooks:
      - ``_get_discriminator_real_scores`` → real-time PESQ
      - ``_forward_generator`` → extract mag/pha for composite loss
      - ``_extract_gan_mag_pha`` → discriminator sees the compressed mags
      - ``_compute_reconstruction_loss`` → composite 5-term loss
      - ``_compute_validation_loss`` → composite loss on val set
    """

    def __init__(self, **kw: Any):
        initial_tc = (kw.get("config") or {}).get("trainer", {})
        initial_use_pesq = bool(initial_tc.get("use_pesq_injection", True))
        self._pesq_backend = _load_pesq_backend() if initial_use_pesq else None
        super().__init__(**kw)
        tc = self.config.get("trainer", {})

        # PESQ injection（逐样本串行 CPU 计算，暂不支持并行 worker）
        self.use_pesq_injection = bool(tc.get("use_pesq_injection", True))
        self.pesq_sr = int(tc.get("pesq_sr", 16000))
        if self.pesq_sr not in (8000, 16000):
            raise ValueError(
                f"trainer.pesq_sr must be 8000 or 16000, got {self.pesq_sr}"
            )

        # Composite loss weights
        self.loss_weights = {
            "mag": float(tc.get("loss_mag_weight", 0.9)),
            "pha": float(tc.get("loss_pha_weight", 0.3)),
            "com": float(tc.get("loss_com_weight", 0.1)),
            "time": float(tc.get("loss_time_weight", 0.2)),
            "stft": float(tc.get("loss_stft_weight", 0.1)),
        }

        # Build composite loss from config
        loss_cfg = self.config.get("loss", {})
        stft_cfg = self.config.get("stft", {})
        # 本 trainer 不走 self.loss_fn（重构 loss 由 composite_loss 承担），
        # 该重建 loss 仅供 entry 的 build_loss 构造；YAML 中 loss.type 必须
        # 为 MPSEnetCompositeLoss（与实现一致时不告警），这里仅读取
        # loss.args 中的 compress_factor/window。
        if loss_cfg.get("type") not in (None, "", "MPSEnetCompositeLoss"):
            logger.warning(
                "MetricGANCompositeTrainer ignores configured loss.type=%r "
                "(reconstruction loss is MPSEnetCompositeLoss); "
                "only loss.args.compress_factor/window are consumed.",
                loss_cfg.get("type"),
            )
        loss_args = loss_cfg.get("args", {}) or {}
        self.composite_loss = MPSEnetCompositeLoss(
            weight_mag=self.loss_weights["mag"],
            weight_pha=self.loss_weights["pha"],
            weight_com=self.loss_weights["com"],
            weight_time=self.loss_weights["time"],
            weight_stft=self.loss_weights["stft"],
            stft_n_fft=int(stft_cfg.get("n_fft", 400)),
            stft_hop_length=int(stft_cfg.get("hop_length", 100)),
            stft_win_length=int(stft_cfg.get("win_length", 400)),
            # compress_factor 优先取 loss.args（4 个 MetricGAN YAML 的实际写法），
            # 再回退到 loss 顶层键，最后默认 0.3
            compress_factor=float(
                loss_args.get("compress_factor", loss_cfg.get("compress_factor", 0.3))
            ),
            window=str(
                loss_args.get("window", loss_cfg.get("window", stft_cfg.get("window", "hann")))
            ),
        )

        logger.info(
            "MetricGANCompositeTrainer: pesq_injection=%s, weights=%s, stft=%s",
            self.use_pesq_injection,
            self.loss_weights,
            stft_cfg,
        )

    # ------------------------------------------------------------------
    # PESQ scoring (CPU, batched)
    # ------------------------------------------------------------------
    def _compute_batch_pesq(
        self,
        clean_wav: torch.Tensor,
        enhanced_wav: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute normalized PESQ scores [0,1] for each sample in batch.

        Runs on CPU to avoid GPU memory pressure. 任何评分失败都直接报错，避免
        PESQ injection 在实验中静默退化为零伪标签。
        ``lengths`` 提供时按每条样本的有效长度裁剪，避免 padded 尾部参与评分。
        """
        pesq = self._pesq_backend
        if pesq is None:
            raise RuntimeError(
                "PESQ injection is enabled but the PESQ backend is unavailable"
            )

        # .numpy() 本身强制 host-device 同步，无需（也无法）给 .cpu() 传 non_blocking；
        # torch 2.4 的 Tensor.cpu() 不接受 non_blocking kwarg（会 TypeError）。
        clean_np = clean_wav.detach().cpu().numpy()
        enh_np = enhanced_wav.detach().cpu().numpy()

        scores = []
        mode = "wb" if self.pesq_sr == 16000 else "nb"
        lens_np = (
            lengths.detach().cpu().numpy()
            if lengths is not None
            else [None] * len(clean_np)
        )
        for sample_index, (c, e, n) in enumerate(zip(clean_np, enh_np, lens_np)):
            c1 = c.squeeze() if c.ndim > 1 else c
            e1 = e.squeeze() if e.ndim > 1 else e
            if n is not None:
                # 裁剪到有效长度，padding 段不参与 PESQ 评分
                n = max(int(n), 1)
                c1, e1 = c1[:n], e1[:n]
            try:
                score = pesq(self.pesq_sr, c1, e1, mode)
                scores.append(max((score - 1.0) / 3.5, 0.0))  # normalize to [0, 1], clamp
            except Exception as exc:
                raise RuntimeError(
                    "PESQ injection failed for sample "
                    f"{sample_index} with {len(c1)} valid samples at "
                    f"{self.pesq_sr} Hz"
                ) from exc

        return torch.tensor(scores, dtype=torch.float32, device=clean_wav.device)

    # ------------------------------------------------------------------
    # Hook: real scores for discriminator
    # ------------------------------------------------------------------
    def _get_discriminator_real_scores(
        self, forward_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        if not self.use_pesq_injection:
            return None
        # 不重采样时错误采样率会生成无效伪标签，必须在调用 PESQ 前显式失败。
        fs_val = self._batch_fs_hz(forward_dict.get("fs"), forward_dict["clean"].shape[0])
        if fs_val != self.pesq_sr:
            raise ValueError(
                f"PESQ injection requires batch fs={self.pesq_sr}, got {fs_val}; "
                "configure matching pesq_sr or resample the dataset."
            )
        return self._compute_batch_pesq(
            forward_dict["clean"], forward_dict["enhanced"], forward_dict.get("lengths")
        )

    # ------------------------------------------------------------------
    # Hook: forward generator with mag/pha extraction
    # ------------------------------------------------------------------
    def _forward_generator(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Generator forward that also extracts mag/pha/complex for D and loss."""
        stft = self.stft_module
        if stft is None:
            raise RuntimeError("MetricGANCompositeTrainer requires stft_module")

        noisy = batch_dict["noisy"]
        clean = batch_dict["clean"]
        lengths = batch_dict["lengths"]

        noisy_spec = stft(noisy)
        clean_spec = stft(clean)
        enhanced_spec = self.generator(noisy_spec, lengths)
        enhanced = stft.inverse(enhanced_spec, noisy.shape[-1])

        # 与官方 mag_phase_stft(addeps=False) 一致：直接使用 abs/angle。
        def _mag_pha(spec):
            complex_spec = torch.view_as_complex(spec.contiguous())
            return torch.abs(complex_spec), torch.angle(complex_spec)

        clean_mag, clean_pha = _mag_pha(clean_spec)
        enh_mag, enh_pha = _mag_pha(enhanced_spec)

        # Compress mag for composite loss (matching original MP-SENet / MambAttention / RWSAMamba-UNet)
        compress_factor = self.composite_loss.compress_factor
        clean_mag_c = torch.pow(clean_mag, compress_factor)
        enh_mag_c = torch.pow(enh_mag, compress_factor)

        return {
            **batch_dict,
            "noisy_spec": noisy_spec,
            "clean_spec": clean_spec,
            "enhanced_spec": enhanced_spec,
            "enhanced": enhanced,
            "clean_mag": clean_mag_c,
            "clean_pha": clean_pha,
            "enhanced_mag": enh_mag_c,
            "enhanced_pha": enh_pha,
        }

    def _sanitize_forward_output(
        self, forward_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Sanitize enhanced waveform，并重算所有 enhanced 侧派生谱。"""
        forward_dict = super()._sanitize_forward_output(forward_dict)
        enhanced_spec = forward_dict["enhanced_spec"]
        enhanced_complex = torch.view_as_complex(enhanced_spec.contiguous())
        enhanced_mag = torch.abs(enhanced_complex)
        enhanced_pha = torch.angle(enhanced_complex)
        forward_dict["enhanced_spec"] = enhanced_spec
        forward_dict["enhanced_mag"] = torch.pow(
            enhanced_mag, self.composite_loss.compress_factor
        )
        forward_dict["enhanced_pha"] = enhanced_pha
        return forward_dict

    # ------------------------------------------------------------------
    # Hook: 判别器输入直接使用 sanitize 后重算的压缩 mag/pha
    # ------------------------------------------------------------------
    def _extract_gan_mag_pha(
        self, fwd: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 forward dict 中压缩 mag 与相位（父类 4 元组契约）。

        父类默认从 ``*_spec`` 重新提取**未压缩** mag，与本类 composite loss
        及 MetricGAN 判别器输入分布不一致，因此在此覆盖。enhanced 侧字段已由
        ``_sanitize_forward_output`` 基于 sanitize 后波形统一重算。
        """
        return (
            fwd["clean_mag"],
            fwd["clean_pha"],
            fwd["enhanced_mag"],
            fwd["enhanced_pha"],
        )

    # ------------------------------------------------------------------
    # Hook: composite reconstruction loss
    # ------------------------------------------------------------------
    def _compute_reconstruction_loss(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        forward_dict: dict[str, torch.Tensor] | None = None,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        if forward_dict is None:
            raise RuntimeError("MetricGANCompositeTrainer requires forward_dict")
        return self.composite_loss(forward_dict)

    # ------------------------------------------------------------------
    # Hook: validation loss
    # ------------------------------------------------------------------
    def _compute_validation_loss(self, forward_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.composite_loss(forward_dict)
