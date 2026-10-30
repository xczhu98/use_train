"""SE-specific SETrainerBase.

Extends the core ``MetricsCapableTrainer`` with SE batch handling,
forward/loss hooks (train ``_forward_and_compute_loss`` and validation
``_val_forward_loss``), and batch validity checks.  The train/validation
epoch skeletons live in the core base class.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

from core.losses.composite import CompositeLoss
from core.models.execution_plan import resolve_trainer_execution_plan
from core.trainer.metrics_trainer import (
    MetricsCapableTrainer as _MetricsCapableTrainer,
    _composite_loss_comp_log_label,
    _direct_spectral_loss_kwargs,
    _loss_has_active_capability,
    _waveform_loss_kwargs,
    validate_waveform_lengths,
)
from tasks.se.data.batch import SEBatch

logger = logging.getLogger(__name__)


def _unpack_se_batch(batch: SEBatch) -> SEBatch:
    """消费 collate 的结构化批，拒绝位置 tuple/list。"""
    if not isinstance(batch, SEBatch):
        raise TypeError(
            f"SETrainer requires an SEBatch from collate_fn, got {type(batch).__name__}"
        )
    batch.validate()
    return batch


def _loss_requires_noisy_reference(loss_fn: object) -> bool:
    """是否需要向 loss 注入 noisy reference。"""
    return _loss_has_active_capability(loss_fn, "REQUIRES_NOISY_REFERENCE")


def _loss_requires_sample_rate(loss_fn: object) -> bool:
    """是否需要向 loss 注入 batch 实际采样率。"""
    return _loss_has_active_capability(loss_fn, "REQUIRES_SAMPLE_RATE")


def _composite_pure_flag_kwargs(
    loss_fn: object, batch: SEBatch, device: torch.device
) -> dict[str, torch.Tensor]:
    """仅向 CompositeLoss 注入 ``is_pure_*`` 分支标志。

    CompositeLoss 负责把标志应用到 SISNR 等 speech-reference 分量；单一 loss
    不接收这些 metadata，避免其自行裁剪 batch。普通 loss 同样不接收额外 kwargs。
    """
    inner = getattr(loss_fn, "loss", loss_fn)
    if not isinstance(inner, CompositeLoss):
        return {}
    out: dict[str, torch.Tensor] = {}
    for k in ("is_pure_noise", "is_pure_echo", "is_pure_echo_noise"):
        v = getattr(batch, k, None)
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
    return out


def _se_loss_kwargs(
    loss_fn: object,
    batch: SEBatch,
    enhanced: torch.Tensor,
    sample_rate: int,
    *,
    noisy: torch.Tensor | None = None,
    clean: torch.Tensor | None = None,
    reference_channel: int = 0,
) -> dict[str, Any]:
    """集中组装 SE loss 上下文；多通道 observation 投影到参考麦克风。"""
    noisy = batch.noisy if noisy is None else noisy
    clean = batch.clean if clean is None else clean
    if torch.is_tensor(noisy):
        if noisy.ndim == 3:
            if type(reference_channel) is not int or not 0 <= reference_channel < noisy.shape[1]:
                raise ValueError("SE loss reference_channel must index noisy microphone channels")
            noisy = noisy[:, reference_channel, :]
        elif noisy.ndim == 2 and reference_channel != 0:
            raise ValueError("mono SE loss input requires reference_channel=0")
    out = _waveform_loss_kwargs(
        loss_fn,
        noisy,
        clean,
        enhanced,
        sample_rate,
    )
    out.update(_composite_pure_flag_kwargs(loss_fn, batch, clean.device))
    return out


class SETrainerBase(_MetricsCapableTrainer):
    SUPPORTS_WAVEFORM_VAL_METRICS = True
    VALIDATION_OUTPUT_DOMAIN = "waveform"

    def __init__(self, *args: Any, **kwargs: Any):
        config = kwargs.get("config") or {}
        val_metrics = (config.get("trainer") or {}).get("val_metrics") or {}
        if (
            not self.SUPPORTS_WAVEFORM_VAL_METRICS
            and bool(val_metrics.get("enabled", False))
        ):
            raise ValueError(
                f"{type(self).__name__} validation output domain is "
                f"{self.VALIDATION_OUTPUT_DOMAIN!r}, so objective audio "
                "trainer.val_metrics are unsupported; set "
                "trainer.val_metrics.enabled=false."
            )
        super().__init__(*args, **kwargs)
        self.execution_plan = resolve_trainer_execution_plan(
            self.config,
            self._raw_model(),
            task="se",
            sample_rate=self.sample_rate,
            dataset_contract=getattr(self.train_dataset, "contract", None),
            stft_module=self.stft_module,
        )

    def _forward_model(
        self,
        noisy,
        lengths,
        fs,
        return_lsnr: bool = False,
        return_alpha: bool = False,
    ) -> tuple[torch.Tensor, Any, Any]:
        """Returns ``(enhanced_wav, lsnr_or_None, df_alpha_or_None)``."""
        self._last_model_enhanced_spectra = None
        if return_alpha and not return_lsnr:
            raise ValueError("return_alpha=True requires return_lsnr=True")
        want_lsnr = bool(return_lsnr and self._model_accepts_return_lsnr)
        want_alpha = bool(return_alpha and self._model_accepts_return_alpha)

        input_domain = getattr(
            getattr(self, "execution_plan", None), "input_domain", None
        )
        use_runtime_stft = (
            input_domain == "spectrum"
            if input_domain is not None
            else self.stft_module is not None
        )
        if use_runtime_stft:
            if self.stft_module is None:
                raise RuntimeError(
                    "SE spectrum execution plan requires stft_module"
                )
            if noisy.ndim == 3:
                batch_size, channels, samples = noisy.shape
                noisy_spec = self.stft_module(noisy.reshape(batch_size * channels, samples))
                noisy_spec = noisy_spec.reshape(batch_size, channels, *noisy_spec.shape[1:])
            else:
                noisy_spec = self.stft_module(noisy)
            if want_lsnr:
                call_kw: dict[str, Any] = {"return_lsnr": True}
                if want_alpha:
                    call_kw["return_alpha"] = True
                if self._model_accepts_fs:
                    out = self.model(noisy_spec, lengths, fs=fs, **call_kw)
                else:
                    out = self.model(noisy_spec, lengths, **call_kw)
                if isinstance(out, tuple) and len(out) >= 3:
                    enhanced_spec, lsnr, alpha = out[0], out[1], out[2]
                elif isinstance(out, tuple) and len(out) >= 2:
                    enhanced_spec, lsnr = out[0], out[1]
                    alpha = None
                else:
                    raise RuntimeError(
                        f"Expected (spec, lsnr[, alpha]) from model with return_lsnr, got {type(out)}"
                    )
                self._last_model_enhanced_spectra = enhanced_spec
                wav = self.stft_module.inverse(enhanced_spec, noisy.shape[-1])
                return wav, lsnr, alpha
            if self._model_accepts_fs:
                enhanced_spec = self.model(noisy_spec, lengths, fs=fs)
            else:
                enhanced_spec = self.model(noisy_spec, lengths)
            self._last_model_enhanced_spectra = enhanced_spec
            wav = self.stft_module.inverse(enhanced_spec, noisy.shape[-1])
            return wav, None, None
        if want_lsnr:
            call_kw = {"return_lsnr": True}
            if want_alpha:
                call_kw["return_alpha"] = True
            if self._model_accepts_fs:
                out = self.model(noisy, lengths, fs=fs, **call_kw)
            else:
                out = self.model(noisy, lengths, **call_kw)
            if isinstance(out, tuple) and len(out) >= 3:
                return out[0], out[1], out[2]
            if isinstance(out, tuple) and len(out) >= 2:
                return out[0], out[1], None
            raise RuntimeError(f"Expected (wav, lsnr[, alpha]) from model, got {type(out)}")
        if self._model_accepts_fs:
            return self.model(noisy, lengths, fs=fs), None, None
        return self.model(noisy, lengths), None, None

    def _val_prepare_batch(
        self, raw_batch: Any, epoch: int, n_loops: int
    ) -> tuple[SEBatch, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """在 DDP forward 前完成 SE batch 解包、校验与 device 搬运。"""
        se_batch = _unpack_se_batch(raw_batch)
        validate_waveform_lengths(
            se_batch.lengths,
            batch_size=se_batch.clean.shape[0],
            max_length=se_batch.clean.shape[-1],
            context="SE validation",
        )
        se_batch = se_batch.to(self.device, non_blocking=True)
        se_batch = self._sanitize_batch_inputs(se_batch, epoch, n_loops)
        clean = se_batch.clean
        noisy = se_batch.noisy
        lengths = se_batch.lengths
        fs_val = self._batch_fs_hz(se_batch.sr, clean.shape[0])
        return se_batch, clean, noisy, lengths, fs_val

    def _val_forward_prepared(
        self,
        prepared_batch: tuple[SEBatch, torch.Tensor, torch.Tensor, torch.Tensor, int],
        epoch: int,
        n_loops: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Any]:
        """消费已同步通过 prepare 门禁的 SE batch，执行模型与 Loss。

        Train/Val 仅向 CompositeLoss 注入 ``is_pure_*`` 分支标志，由其裁剪 SISNR
        等 speech-reference 分量；单一 loss 与普通 loss 均不接收。若 Composite 开启
        SS，仍注入 ``undesired_*``。
        """
        del epoch, n_loops
        se_batch, clean, noisy, lengths, fs_val = prepared_batch
        use_lsnr = (
            self._use_lsnr_loss
            and self._model_accepts_return_lsnr
            and self._loss_accepts_lsnr
        )
        use_alpha = (
            use_lsnr
            and self._loss_needs_predicted_alpha
            and self._model_accepts_return_alpha
        )

        def _forward_and_build_loss_kwargs():
            with torch.amp.autocast(
                self.device.type, enabled=self.use_amp, dtype=self.amp_dtype
            ):
                enhanced, lsnr, df_alpha = self._forward_model(
                    noisy,
                    lengths,
                    fs_val,
                    return_lsnr=use_lsnr,
                    return_alpha=use_alpha,
                )
                enhanced = self._sanitize_model_output(enhanced)
                if self._strict_checks and enhanced.shape != clean.shape:
                    raise ValueError(
                        f"Model output shape mismatch: enhanced={tuple(enhanced.shape)} "
                        f"clean={tuple(clean.shape)}"
                    )
                loss_kwargs = _se_loss_kwargs(
                    self.loss_fn,
                    se_batch,
                    enhanced,
                    fs_val,
                    noisy=noisy,
                    clean=clean,
                    reference_channel=getattr(self._raw_model(), "ref_channel", 0),
                )
                loss_kwargs.update(
                    _direct_spectral_loss_kwargs(
                        self.loss_fn,
                        getattr(self, "stft_module", None),
                        clean,
                        getattr(self, "_last_model_enhanced_spectra", None),
                        lengths=lengths,
                    )
                )
                if use_lsnr and lsnr is not None:
                    loss_kwargs["predicted_lsnr"] = lsnr
                    loss_kwargs["noisy"] = noisy
                    if df_alpha is not None:
                        loss_kwargs["predicted_alpha"] = df_alpha
                return enhanced, loss_kwargs

        enhanced, loss_kwargs = self._run_synced_validation_subphase(
            _forward_and_build_loss_kwargs,
            phase="validation_model_output_error",
            context="SE validation model/output preflight failed",
        )
        with torch.amp.autocast(
            self.device.type, enabled=self.use_amp, dtype=self.amp_dtype
        ):
            loss = self.loss_fn(enhanced, clean, lengths, **loss_kwargs)
        return loss, enhanced, clean, lengths, fs_val, None

    def _val_forward_loss(
        self, raw_batch: Any, epoch: int, n_loops: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Any]:
        """兼容旧调用方：顺序组合 prepare 与 prepared-forward 两阶段。"""
        prepared = self._val_prepare_batch(raw_batch, epoch, n_loops)
        return self._val_forward_prepared(prepared, epoch, n_loops)

    def _unpack_batch(self, raw_batch: Any) -> Any:
        return _unpack_se_batch(raw_batch)

    def _validate_batch_metadata(
        self, batch: Any, epoch: int, batch_idx: int
    ) -> None:
        if isinstance(batch, SEBatch):
            validate_waveform_lengths(
                batch.lengths,
                batch_size=batch.clean.shape[0],
                max_length=batch.clean.shape[-1],
                context=f"SE train E{epoch} B{batch_idx}",
            )

    def _maybe_count_silence_placeholders(self, batch: Any) -> int:
        if not isinstance(batch, SEBatch):
            return 0
        # 占位样本是逐字段全零波形（见 BaseDynamicMixingDataset._make_silence_placeholder），
        # 其长度随 max_duration / max_duration_seconds 配置变化（如 10s→160000），
        # 与 _placeholder_num_samples（默认 1024）并不相等，旧的
        # lengths == _placeholder_num_samples 判据因此永远漏检（AEC 侧已改为能量判据）。
        # 这里对齐 AEC：clean + noisy 同时为零能量才是占位样本；要求 noisy 也为零，
        # 排除 clean 静音但 noisy 含噪声/回声的合法样本。
        # einsum 逐行点积不物化 (B, T) 平方张量（[B, C, T] 直接 flatten 成行），
        # 每 micro-batch 的额外显存/内存从 O(B*T) 降到 O(B)。
        clean_flat = batch.clean.flatten(start_dim=1)
        noisy_flat = batch.noisy.flatten(start_dim=1)
        energy_clean = torch.einsum("ij,ij->i", clean_flat, clean_flat)
        energy_noisy = torch.einsum("ij,ij->i", noisy_flat, noisy_flat)
        return int(((energy_clean < 1e-10) & (energy_noisy < 1e-10)).sum().item())

    def _sanitize_batch_inputs(self, batch: Any, epoch: int, batch_idx: int) -> Any:
        if not isinstance(batch, SEBatch):
            return batch
        if getattr(self, "_strict_checks", False):
            self._assert_batch_valid(batch.clean, batch.noisy, batch.lengths)
        nonfinite_fields: list[str] = []
        if not torch.isfinite(batch.clean).all():
            nonfinite_fields.append("clean")
            batch.clean = torch.nan_to_num(
                batch.clean, nan=0.0, posinf=0.0, neginf=0.0
            )
        if not torch.isfinite(batch.noisy).all():
            nonfinite_fields.append("noisy")
            batch.noisy = torch.nan_to_num(
                batch.noisy, nan=0.0, posinf=0.0, neginf=0.0
            )
        if nonfinite_fields:
            if self.rank == 0:
                logger.warning(
                    "[E%s B%s] NaN/Inf input fields %s, non-finite values zeroed out",
                    epoch,
                    batch_idx,
                    ", ".join(nonfinite_fields),
                )
        return batch

    def _forward_and_compute_loss(
        self,
        batch: Any,
        epoch: int,
        batch_idx: int,
        fs_val: int,
    ) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        use_lsnr = (
            self._use_lsnr_loss
            and self._model_accepts_return_lsnr
            and self._loss_accepts_lsnr
        )
        use_alpha = (
            use_lsnr
            and self._loss_needs_predicted_alpha
            and self._model_accepts_return_alpha
        )

        enhanced, lsnr, df_alpha = self._forward_model(
            batch.noisy,
            batch.lengths,
            fs_val,
            return_lsnr=use_lsnr,
            return_alpha=use_alpha,
        )

        # A23：与 val 侧同一套 sanitize（nan_to_num + clamp(-1,1)），
        # 训练/验证目标一致；触发计数在 epoch 末汇总上报。
        enhanced = self._sanitize_model_output(enhanced)

        if self._strict_checks and enhanced.shape != batch.clean.shape:
            # 与 val 侧同一 fail-fast 形式：assert 会被 python -O 整体移除，
            # 训练/验证的严格度必须一致。
            raise ValueError(
                f"Model output shape mismatch: enhanced={tuple(enhanced.shape)} "
                f"clean={tuple(batch.clean.shape)}"
            )

        _loss_kw = _se_loss_kwargs(
            self.loss_fn,
            batch,
            enhanced,
            fs_val,
            reference_channel=getattr(self._raw_model(), "ref_channel", 0),
        )
        _loss_kw.update(
            _direct_spectral_loss_kwargs(
                self.loss_fn,
                getattr(self, "stft_module", None),
                batch.clean,
                getattr(self, "_last_model_enhanced_spectra", None),
                lengths=batch.lengths,
            )
        )
        if use_lsnr and lsnr is not None:
            _lk = {
                "predicted_lsnr": lsnr,
                "noisy": batch.noisy,
                **_loss_kw,
            }
            if df_alpha is not None:
                _lk["predicted_alpha"] = df_alpha
            loss = self.loss_fn(enhanced, batch.clean, batch.lengths, **_lk)
        else:
            loss = self.loss_fn(enhanced, batch.clean, batch.lengths, **_loss_kw)

        loss_was_finite = bool(torch.isfinite(loss).item())
        info: dict[str, Any] = {"loss_was_finite": loss_was_finite}
        if not loss_was_finite:
            e_safe = torch.nan_to_num(
                enhanced.detach(), nan=0.0, posinf=0.0, neginf=0.0
            )
            info["enhanced_safe_min"] = float(e_safe.min().item())
            info["enhanced_safe_max"] = float(e_safe.max().item())
        return loss, loss_was_finite, info

    def _assert_batch_valid(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        # 全部用 raise 而非 assert：python -O 会移除 assert，使 strict_checks
        # 的训练侧校验整体失效（val 侧本就是 raise）。
        if clean.dim() != 2 or noisy.dim() not in (2, 3):
            raise ValueError(
                f"clean must be [B, T] and noisy [B, T] or [B, C, T], got clean={tuple(clean.shape)}, noisy={tuple(noisy.shape)}"
            )
        expected_noisy = (
            noisy.shape[:1] + noisy.shape[2:] if noisy.dim() == 3 else noisy.shape
        )
        if clean.shape != expected_noisy:
            raise ValueError(
                f"clean/noisy shape mismatch: {tuple(clean.shape)} vs {tuple(noisy.shape)}"
            )
        if noisy.dim() == 3:
            input_domain = getattr(getattr(self, "execution_plan", None), "input_domain", None)
            if input_domain != "spectrum":
                raise ValueError(
                    "multi-channel SE input requires a spectrum-domain model"
                )
        if lengths.dim() != 1 or lengths.shape[0] != clean.shape[0]:
            raise ValueError(
                f"lengths must be [B], got {tuple(lengths.shape)} for B={clean.shape[0]}"
            )
        if not (torch.isfinite(clean).all() and torch.isfinite(noisy).all()):
            raise ValueError("Input contains NaN/Inf")
