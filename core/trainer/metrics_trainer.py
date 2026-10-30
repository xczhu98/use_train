"""MetricsCapableTrainer — core base class for task-specific trainers.

Extracts common validation metrics, loss introspection, and checkpoint/curve
logic shared between speech enhancement and acoustic echo cancellation trainers.

This module intentionally does **not** depend on SE- or AEC-specific batch
types: the training-loop template (``_train_epoch``) and validation-loop
template (``_validation_epoch``) live here, while subclasses inject
task-specific behaviour through the protected hooks (batch unpacking,
forward/loss, extra accumulators).
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import math
import time
from dataclasses import is_dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.losses.composite import CompositeLoss, StrictFiniteLossError
from core.losses.loss_context import LossContext
from core.losses.constants import LOSS_DISPLAY_NAMES
from core.losses.utils import prepare_stft_waveform_at_lengths
from core.data.batch_contract import (
    batch_size_from_contract,
    is_batch_contract,
    move_batch_to_device,
    move_mapping_tensors,
)
from core.data.waveform_batch import validate_waveform_lengths
from core.trainer.base_trainer import BaseTrainer
from core.trainer.validation_metrics_config import (
    validate_no_trainer_mos_config,
    validate_validation_metric_runtime_result,
)


class LossContextValidationError(RuntimeError):
    """DDP typed LossContext 预检已同步的结构校验失败。"""


def _loss_has_active_capability(loss_fn: object, capability: str) -> bool:
    """判断单一 loss 或 Composite 非零权重分量是否声明指定能力。"""
    inner = getattr(loss_fn, "loss", loss_fn)
    if isinstance(inner, CompositeLoss):
        return any(
            weight != 0.0 and bool(getattr(component, capability, False))
            for component, weight in zip(inner.losses, inner._weights_float)
        )
    return bool(getattr(inner, capability, False))


def _composite_ss_undesired_kwargs(
    loss_fn: object,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    enhanced: torch.Tensor,
    context: LossContext | None = None,
) -> dict[str, torch.Tensor]:
    """按活动 loss 组装 desired/undesired complement waveform。

    这里的 ``SS`` 指 CompositeLoss 的 undesired-energy 机制，不是 ``ss``
    source-separation 任务。对 TSE，返回的是 ``mixture-target``，它可能包含
    interference、background、target full/early 差异和 observation degradation，
    不能解释为纯非目标说话人轨道。
    """
    context = context or LossContext.from_waveforms(noisy, clean, enhanced)
    inner = getattr(loss_fn, "loss", loss_fn)
    if not isinstance(inner, CompositeLoss):
        if bool(getattr(inner, "REQUIRES_UNDESIRED_REFERENCE", False)):
            return {
                "undesired_ref": context.undesired_ref,
                "undesired_enhanced": context.undesired_estimate,
            }
        return {}
    effective_ss = getattr(inner, "_effective_ss", [])
    needs_undesired = any(
        weight != 0.0
        and (
            (idx < len(effective_ss) and bool(effective_ss[idx]))
            or bool(getattr(inner.losses[idx], "REQUIRES_UNDESIRED_REFERENCE", False))
        )
        for idx, weight in enumerate(inner._weights_float)
    )
    if not needs_undesired:
        return {}
    return {
        "undesired_ref": context.undesired_ref,
        "undesired_enhanced": context.undesired_estimate,
    }


def _waveform_loss_kwargs(
    loss_fn: object,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    enhanced: torch.Tensor,
    sample_rate: int,
) -> dict[str, Any]:
    """按活动 capability 组装任务无关的三路 waveform loss 上下文。"""
    inner = getattr(loss_fn, "loss", loss_fn)
    needs_noisy = _loss_has_active_capability(loss_fn, "REQUIRES_NOISY_REFERENCE")
    needs_ss = bool(getattr(inner, "any_ss_enabled", False))
    if noisy is None:
        if needs_noisy or needs_ss:
            raise ValueError(
                "waveform loss requires a noisy reference for its active components"
            )
        out: dict[str, Any] = {}
    else:
        context = LossContext.from_waveforms(
            noisy, clean, enhanced, lengths=None, sample_rate=sample_rate
        )
        out = _composite_ss_undesired_kwargs(
            loss_fn, noisy, clean, enhanced, context=context
        )
    if _loss_has_active_capability(loss_fn, "REQUIRES_NOISY_REFERENCE"):
        out["noisy"] = noisy
    if _loss_has_active_capability(loss_fn, "REQUIRES_SAMPLE_RATE"):
        out["sample_rate"] = sample_rate
    return out


def _direct_spectral_loss_kwargs(
    loss_fn: object,
    stft_module: object | None,
    clean: torch.Tensor,
    enhanced_spectra: torch.Tensor | None,
    *,
    lengths: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """按真实有效边界生成目标谱，并对齐模型原生预测谱 frame layout。"""
    if enhanced_spectra is None or not _loss_has_active_capability(
        loss_fn, "SUPPORTS_DIRECT_SPECTRA"
    ):
        return {}
    if stft_module is None:
        raise RuntimeError(
            "direct-spectrum loss context requires the framework stft_module"
        )

    inner = getattr(loss_fn, "loss", loss_fn)
    if isinstance(inner, CompositeLoss):
        active_direct = [
            (component, bool(inner._effective_ss[index]))
            for index, (component, weight) in enumerate(
                zip(inner.losses, inner._weights_float)
            )
            if weight != 0.0
            and bool(getattr(component, "SUPPORTS_DIRECT_SPECTRA", False))
        ]
        if any(effective_ss for _, effective_ss in active_direct):
            raise ValueError(
                "direct-spectrum loss does not support enable_ss_undesired_energy: "
                "undesired prediction/reference spectra are not available"
            )
        components = [component for component, _ in active_direct]
    else:
        components = [inner]
    framework_geometry = (
        int(getattr(stft_module, "n_fft")),
        int(getattr(stft_module, "hop_length")),
        int(getattr(stft_module, "win_length")),
        bool(getattr(stft_module, "center")),
        str(getattr(stft_module, "window_name")),
        bool(getattr(stft_module, "normalized", False)),
    )
    for component in components:
        component_geometry = (
            int(getattr(component, "n_fft")),
            int(getattr(component, "hop_length")),
            int(getattr(component, "win_length")),
            bool(getattr(component, "center")),
            str(getattr(component, "window_name")),
            bool(getattr(component, "normalized", False)),
        )
        if component_geometry != framework_geometry:
            raise ValueError(
                "direct-spectrum loss STFT geometry must match the framework STFT: "
                f"loss={component_geometry}, framework={framework_geometry}"
            )

    target_input, _ = prepare_stft_waveform_at_lengths(
        clean,
        lengths,
        n_fft=framework_geometry[0],
        center=framework_geometry[3],
    )
    target_spectra = stft_module(target_input)
    prediction_frames = (
        enhanced_spectra.shape[-1]
        if enhanced_spectra.is_complex()
        else enhanced_spectra.shape[-2]
    )
    target_frames = target_spectra.shape[-2]
    if target_frames < prediction_frames:
        raise ValueError(
            "length-aware target spectrum has fewer frames than model prediction: "
            f"target={target_frames}, prediction={prediction_frames}"
        )
    target_spectra = target_spectra[..., :prediction_frames, :]
    return {
        "enhanced_spectra": enhanced_spectra,
        "target_spectra": target_spectra,
    }


def _composite_loss_comp_log_label(inner: object, idx: int, log_key: str) -> str:
    """Console-friendly label for one component (uses YAML ``name`` / registry type)."""
    types = getattr(inner, "_component_types", None)
    names = getattr(inner, "_component_names", None)
    if types and names and idx < len(types) and idx < len(names):
        return str(LOSS_DISPLAY_NAMES.get(types[idx], names[idx]))
    return str(LOSS_DISPLAY_NAMES.get(log_key, log_key))

logger = logging.getLogger(__name__)


from core.utils.ddp_error import (
    is_unrecoverable_cuda_exception as _is_unrecoverable_cuda_exception,
)


def _composite_loss_comp_keys(inner: object) -> list[str] | None:
    """Ordered keys for loss component sums (TB / DDP); prefers ``yaml_*::type``."""
    keys = getattr(inner, "_component_log_keys", None)
    if keys:
        return list(keys)
    names = getattr(inner, "_component_names", None)
    return list(names) if names else None


def sanitize_model_output(
    wav: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train/Val 统一的模型输出约束：非有限值置 0，幅度钳到 [-1, 1]（审计 A23）。

    以往 train 侧仅 ``nan_to_num``（或不处理）、val 侧 ``nan_to_num + clamp``，
    模型选择目标与训练目标不一致；统一后两侧均调用本函数，数值语义沿用既有
    val 侧（非有限 → 0，|x|>1 → 钳位）。

    设计说明（梯度语义是有意的，勿当 bug 修）：
    - ``clamp`` 区间外梯度为 0、``nan_to_num`` 非有限点梯度为 0，二者共同构成
      **防火墙**：真实训练中 |x|>1 大多伴随 NaN/Inf/AMP 溢出等病态输出，其梯度
      是垃圾甚至毒物，置零是保护而非缺陷。
    - 越界样本只是"弃权"一步：模型参数被全 batch 所有采样点共享，界内样本的
      梯度仍提供把幅度拉回 [-1, 1] 的纠正力（主力 loss 均为幅度敏感型），
      越界样本缺席不影响该纠正。
    - 因此**不要**对 clamp 做 STE 直通、软钳位或越界惩罚项替代：那会把病态
      样本的垃圾梯度重新引入计算图，收益≈0 且破坏防火墙语义。
    - 监控：触发计数按 epoch 累加并在 epoch 末汇总 warning
      （``_log_output_sanitize_summary``）。若某 run 频繁出现大计数，应排查
      该 run 的 AMP/学习率/架构根因，而非修改本函数。

    Returns:
        ``(sanitized, n_nonfinite, n_clamped)``：修正后的波形、以及与输出同设备的
        计数标量 Tensor。调用方按 epoch 累加，只有日志或异常路径才物化计数。
    """
    finite = torch.isfinite(wav)
    n_nonfinite = (~finite).sum(dtype=torch.int64)
    out = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    out_of_range = (out > 1.0) | (out < -1.0)
    n_clamped = out_of_range.sum(dtype=torch.int64)
    # 区间外梯度为 0 是有意的防火墙语义（见 docstring），勿改 STE/软钳位。
    out = out.clamp(-1.0, 1.0)
    return out, n_nonfinite, n_clamped


class MetricsCapableTrainer(BaseTrainer):
    """Base class for trainers that support validation metrics and composite loss tracking.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        sample_rate: Default sample rate.
        stft_module: Optional STFT/iSTFT module.
    """

    def _ddp_rank_quota(self, global_quota: int) -> int:
        """把全局 quota 精确分配到各 rank，余数优先给低 rank。"""
        base, remainder = divmod(int(global_quota), self.world_size)
        return base + int(self.rank < remainder)

    def __init__(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        sample_rate: int = 16000,
        stft_module: Any = None,
        **kw: Any,
    ):
        # 直构造 Trainer 也必须与 Entry 共用同一 fail-fast 契约，并在昂贵的
        # BaseTrainer 初始化、模型搬运或 resume 前拒绝已移除的训练期 MOS 配置。
        validate_no_trainer_mos_config(kw.get("config") or {})
        # BaseTrainer 构造末尾可能立即 resume；预先绑定 loader 以便在任何
        # model/optimizer/live state 修改前校验 checkpoint cursor 上界。
        self.train_loader = train_loader
        super().__init__(**kw)
        # Extract approximate batch size for _handle_epoch_tail_gradients warning
        try:
            bs = getattr(train_loader, "batch_sampler", None)
            self._approx_batch_size = int(getattr(bs, "batch_size", 0))
        except Exception:
            self._approx_batch_size = 0
        self.val_loader = val_loader
        self.sample_rate = sample_rate
        self.stft_module = stft_module
        tc = self.config.get("trainer", {})
        self.output_clamp = bool(tc.get("output_clamp", True))
        if self.stft_module is not None:
            self.stft_module = self.stft_module.to(self.device)
        # Prefer explicit config flags over inspect.signature reflection
        _margs = self.config.get("model", {}).get("args", {})
        _fwd = self._raw_model().forward
        self._model_accepts_fs = _margs.get(
            "accepts_fs", "fs" in inspect.signature(_fwd).parameters
        )
        self._model_accepts_return_lsnr = _margs.get(
            "accepts_return_lsnr", "return_lsnr" in inspect.signature(_fwd).parameters
        )
        self._model_accepts_return_alpha = _margs.get(
            "accepts_return_alpha", "return_alpha" in inspect.signature(_fwd).parameters
        )
        tc = self.config.get("trainer", {})
        self._use_lsnr_loss = bool(tc.get("use_lsnr_loss", False))
        self._loss_accepts_lsnr = self._loss_accepts_predicted_lsnr()
        self._loss_needs_predicted_alpha = self._compute_loss_needs_predicted_alpha()

        # ── Validation objective metrics config ──
        # max_utterances / max_utterances_metrics: 表示「每 epoch 全局总数」，各 rank 自动按 world_size 均分
        vm = tc.get("val_metrics", {})
        self._vm_enabled     = vm.get("enabled", False)
        self._vm_metrics     = vm.get("metrics", ["SI-SDR", "PESQ", "ESTOI"])
        self._vm_interval    = max(1, int(vm.get("interval", 5)))
        max_utts_global      = max(1, int(vm.get("max_utterances", 200)))
        self._vm_max_utts_global = max_utts_global
        self._vm_max_utts    = self._ddp_rank_quota(max_utts_global)
        _max_metrics_global  = vm.get("max_utterances_metrics")
        if _max_metrics_global is not None:
            _max_metrics_global = max(1, int(_max_metrics_global))
        else:
            _max_metrics_global = max_utts_global
        self._vm_max_metrics_global = _max_metrics_global
        self._vm_max_metrics = self._ddp_rank_quota(_max_metrics_global)
        _max_erle_global = max(
            1, int(vm.get("max_utterances_erle", max_utts_global))
        )
        self._vm_max_erle_global = _max_erle_global
        self._vm_max_erle = self._ddp_rank_quota(_max_erle_global)
        self._vm_chunk_size  = max(1, int(vm.get("metrics_chunk_size", 512)))
        self._vm_strict      = bool(vm.get("strict", False))
        self._vm_min_coverage = {
            str(metric): float(threshold)
            for metric, threshold in (vm.get("min_coverage") or {}).items()
        }
        _parallel_intrusive = bool(vm.get("parallel_intrusive", True))
        if self._vm_enabled and not _parallel_intrusive and self.rank == 0:
            logger.warning(
                "val_metrics.parallel_intrusive=false is unsafe during training; "
                "intrusive metrics will always use an isolated spawn pool."
            )
        self._vm_parallel_intrusive = True if self._vm_enabled else _parallel_intrusive
        _mw = vm.get("metrics_workers", 0)
        self._vm_metrics_workers = None if (_mw is None or int(_mw) == 0) else max(1, int(_mw))  # None/0 = auto
        self._strict_checks = bool(tc.get("strict_checks", True))
        # BaseTrainer.__init__ 会在 subclass 初始化继续执行前完成 resume；不得用默认值
        # 覆盖 _after_resume_checkpoint 已恢复的 validation interval 相位。
        self._vm_epoch_count = int(getattr(self, "_vm_epoch_count", 0))

        self._check_loss_alpha_compatibility()

        # Epoch index or optimizer step (when validation is step-only); aligns with val_metrics_history rows.
        # resume 已在 BaseTrainer.__init__ 内执行，初始化只能补缺省，不能覆盖恢复值。
        self.val_metrics_epochs = list(getattr(self, "val_metrics_epochs", []))
        self.val_metrics_history = dict(getattr(self, "val_metrics_history", {}))
        self.val_metrics_coverage_history = dict(
            getattr(self, "val_metrics_coverage_history", {})
        )
        # Val composite sub-losses (weighted w×raw) at each val_metrics row
        self.val_metrics_loss_comp_hist = dict(
            getattr(self, "val_metrics_loss_comp_hist", {})
        )

        if self._vm_enabled and self.rank == 0:
            logger.info(
                f"Val metrics enabled: {self._vm_metrics}"
                + f" | interval={self._vm_interval}"
                + f" | max_utterances={self._vm_max_utts_global} (global, {self._vm_max_utts}/rank {self.rank})"
                + (f" | max_metrics={self._vm_max_metrics_global} (global)" if self._vm_max_metrics_global != self._vm_max_utts_global else "")
                + f" | chunk_size={self._vm_chunk_size}"
                + f" | strict={self._vm_strict}"
            )
        if self.rank == 0 and self.max_val_batches > 0:
            logger.info(f"Validation capped at max_val_batches={self.max_val_batches}")

        # CompositeLoss: epoch-aligned sub-loss series for train_val_loss.png (with rank-0 hook)
        self._last_train_comp: dict[str, float] = {}
        self._last_val_comp: dict[str, float] = {}
        self._train_loss_comp_hist = dict(
            getattr(self, "_train_loss_comp_hist", {})
        )
        self._val_loss_comp_hist = dict(
            getattr(self, "_val_loss_comp_hist", {})
        )
        self._curve_comp_names = getattr(self, "_curve_comp_names", None)
        # Cache CompositeLoss component keys (computed once; used in train/val/logging)
        self._comp_keys: list[str] | None = _composite_loss_comp_keys(self._inner_loss_fn())


    def _check_loss_alpha_compatibility(self) -> None:
        """启动期检查：loss 需要 predicted_alpha 但模型不支持 return_alpha。

        strict_checks 开启时直接 raise 拦截；否则 rank 0 打 warning（alpha 项将恒为 0）。
        子类可覆写以定制消息或跳过检查（如 GAN 等不使用该 loss 形态的 trainer）。
        """
        if self._strict_checks and self._loss_needs_predicted_alpha and not self._model_accepts_return_alpha:
            raise RuntimeError(
                "strict_checks: loss needs predicted_alpha (e.g. DfAlphaLossComponent weight != 0), "
                "but model.forward has no return_alpha parameter; use a model that supports "
                "return_alpha or remove DfAlpha from CompositeLoss."
            )
        if (
            self._loss_needs_predicted_alpha
            and not self._model_accepts_return_alpha
            and self.rank == 0
        ):
            logger.warning(
                "Loss contains DfAlphaLossComponent (or DeepFilterNetFullLossWithLSNR use_alpha_loss), "
                "but current model has no return_alpha; alpha term will always be 0. "
                "Enable strict_checks to intercept at startup."
            )

    # ── Shared epoch-tail / allreduce helpers (used by both SE and AEC trainers) ──

    def _handle_epoch_tail_gradients(
        self,
        accum_count: int,
        accum: int,
        epoch: int,
        *,
        step_allowed: Optional[bool] = None,
        step_callback: Optional[Callable] = None,
        batch_cursor: Optional[int] = None,
        flush_fn: Optional[Callable] = None,
        label: str = "",
        models: Optional[list] = None,
        zero_grad_fn: Optional[Callable] = None,
        step_fn: Optional[Callable] = None,
    ) -> bool:
        """处理 epoch 尾部残余梯度，并只在真实 step 后触发副作用。

        DDP 中尾窗梯度尚未 all-reduce，固定 flush/zero/drop，绝不执行未同步
        的 optimizer step。单卡会按实际尾窗长度修正梯度；``step_allowed=False``
        用于全 NaN、无梯度或梯度无效的窗口。``step_fn`` 返回值只有显式
        ``False`` 才表示跳过；``optimizer.step`` 返回 ``None`` 不表示跳过。
        """
        if accum_count <= 0:
            # 本方法是唯一 epoch-tail 入口：尾窗为空时无可修正梯度，必须显式拒绝，
            # 否则 scale_correction = accum / accum_count 会直接 ZeroDivisionError。
            raise ValueError(
                f"epoch-tail gradient handler requires accum_count > 0, got {accum_count}"
            )
        models = list(models) if models is not None else [self.model]
        zero_grad_fn = zero_grad_fn or (lambda: self.optimizer.zero_grad(set_to_none=True))
        step_fn = step_fn or self._optimizer_step

        if self.world_size > 1:
            _wasted_samples = accum_count * getattr(self, "_approx_batch_size", 0)
            _detail = (
                f", ~{_wasted_samples} samples wasted"
                if _wasted_samples > 0 else ""
            )
            if self.rank == 0:
                logger.warning(
                    "epoch-tail residual step%s: %d batches unaligned with accum=%d, "
                    "dropping unsynchronized gradients (world_size=%d)%s. "
                    "To avoid this, set accumulation_steps to divide total batches evenly.",
                    f" ({label})" if label else "",
                    accum_count, accum, self.world_size,
                    _detail,
                )
            if flush_fn is not None:
                flush_fn()
            zero_grad_fn()
            return False

        has_grad = False
        has_bad = False
        for model in models:
            trainable = [p for p in model.parameters() if p.requires_grad]
            model_grads = [p.grad for p in trainable if p.grad is not None]
            has_grad = has_grad or bool(model_grads)
            if not model_grads or any(not bool(torch.isfinite(g).all().item()) for g in model_grads):
                has_bad = True
        amp_fp16 = self.use_amp and self.amp_dtype == torch.float16
        eligible = step_allowed is not False and (
            amp_fp16 or (has_grad and not has_bad)
        )
        if not eligible:
            if self.rank == 0:
                reason = "disallowed window" if step_allowed is False else "NaN/Inf or missing grad"
                logger.warning(
                    "epoch-tail residual step%s: %s, skip optimizer step",
                    f" ({label})" if label else "",
                    reason,
                )
            if flush_fn is not None:
                flush_fn()
            zero_grad_fn()
            return False

        scale_correction = accum / accum_count
        for model in models:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale_correction)
        if flush_fn is not None:
            flush_fn()
        result = step_fn()
        stepped = result is not False
        if stepped:
            if batch_cursor is not None:
                self._last_completed_batch_cursor = batch_cursor
            if step_callback is not None:
                step_callback(epoch, self.n_steps, batch_cursor)
            self._log_train_loss_after_step()
        return stepped

    # ------------------------------------------------------------------
    # Shared training-loop template
    # ------------------------------------------------------------------

    def _unpack_batch(self, raw_batch: Any) -> Any:
        """Unpack a DataLoader batch into a task-specific batch object.

        Subclasses must override this hook.
        """
        raise NotImplementedError("_unpack_batch must be implemented by subclass")

    def _validate_batch_metadata(
        self, batch: Any, epoch: int, batch_idx: int
    ) -> None:
        """在 device 搬运前校验轻量 batch 元数据；默认无操作。"""
        return None

    def _batch_to_device(self, batch: Any) -> Any:
        """Move all tensors in ``batch`` to the training device.

        结构化 batch 通过 ``BatchContract.to`` 自己声明字段语义；plain dict
        仅保留给 GAN 的显式路径。未实现 ``BatchContract`` 的 dataclass 在
        训练边界直接失败，避免新增字段后被反射搬运悄悄遗漏。
        """
        if is_batch_contract(batch):
            return move_batch_to_device(batch, self.device, non_blocking=True)
        if isinstance(batch, dict):
            return move_mapping_tensors(batch, self.device, non_blocking=True)
        if is_dataclass(batch):
            raise TypeError(
                f"{type(batch).__name__} must implement BatchContract before trainer device transfer"
            )
        raise TypeError(
            "trainer batch must implement BatchContract or use the explicit GAN dict path; "
            f"got {type(batch).__name__}"
        )

    def _maybe_count_silence_placeholders(self, batch: Any) -> int:
        """Return the number of silence-placeholder samples in this batch.

        Default returns 0; SE/AEC override to detect load-fail fallbacks.
        """
        return 0

    def _sanitize_batch_inputs(self, batch: Any, epoch: int, batch_idx: int) -> Any:
        """Sanitize NaN/Inf inputs before forward.

        Subclasses may override for task-specific input validation/replacement.
        """
        return batch

    def _get_batch_fs(self, batch: Any) -> Any:
        """读取合同 batch 的 ``sr``/``fs`` 或 GAN dict 的显式字段。"""
        if isinstance(batch, dict):
            fs = batch.get("fs")
            if fs is None:
                fs = batch.get("sr")
            return fs
        if is_batch_contract(batch):
            sample_rate = getattr(batch, "sr", None)
            return sample_rate if sample_rate is not None else getattr(batch, "fs", None)
        return None

    def _batch_size_from_batch(self, batch: Any) -> int:
        """读取结构化 batch 主轴或 GAN dict 的 canonical ``clean`` 主轴。"""
        structured_size = batch_size_from_contract(batch)
        if structured_size is not None:
            return structured_size
        if isinstance(batch, dict):
            clean = batch.get("clean")
            if isinstance(clean, torch.Tensor) and clean.ndim >= 1:
                size = int(clean.shape[0])
                if size > 0:
                    return size
            raise ValueError(
                "GAN dict batch requires a non-empty tensor field 'clean' "
                "to define its sample axis"
            )
        raise ValueError(
            "trainer batch must implement BatchContract to define its sample axis; "
            f"got {type(batch).__name__}"
        )

    def _throughput_row_counts(self, batch: Any) -> tuple[int, int]:
        """返回主样本数与任务展开行数；默认二者相同。"""
        primary = self._batch_size_from_batch(batch)
        return primary, primary

    def _log_task_throughput(
        self, primary_rate: float, task_row_rate: float, epoch: int
    ) -> None:
        """任务侧吞吐日志 hook；默认公共 samples/s 已足够。"""
        return None

    def _prepare_batch(
        self,
        raw_batch: Any,
        epoch: int,
        batch_idx: int,
    ) -> tuple[Any, int, int]:
        """Common batch preparation: unpack → device → sanitize → count → fs.

        Returns:
            (batch, fs_val, silence_placeholder_count)
        """
        batch = self._unpack_batch(raw_batch)
        self._validate_batch_metadata(batch, epoch, batch_idx)
        batch = self._batch_to_device(batch)
        batch = self._sanitize_batch_inputs(batch, epoch, batch_idx)
        n_placeholders = self._maybe_count_silence_placeholders(batch)
        batch_size = self._batch_size_from_batch(batch)
        fs_val = self._batch_fs_hz(self._get_batch_fs(batch), batch_size)
        return batch, fs_val, n_placeholders

    def _make_train_progress_bar(self, epoch: int):
        """Create a tqdm progress bar for the training epoch."""
        total = getattr(self.train_loader, "__len__", None)
        if total is not None:
            try:
                total = int(total())
            except TypeError:
                total = None
        return tqdm(
            enumerate(self.train_loader),
            total=total,
            disable=self.rank != 0,
            desc=f"Epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
        )

    def _forward_and_compute_loss(
        self,
        batch: Any,
        epoch: int,
        batch_idx: int,
        fs_val: int,
    ) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        """Forward pass and loss computation.

        Subclasses must override this hook.  It is called inside the AMP autocast
        and gradient-accumulation ``no_sync`` context managed by the base class.

        Args:
            batch: Task-specific batch object (already moved to device and augmented).
            epoch: Current epoch index.
            batch_idx: Batch index within the epoch.
            fs_val: Sampling rate (Hz) for this batch.

        Returns:
            (loss, loss_was_finite, info_dict).  ``info_dict`` is opaque and is
            passed to ``_maybe_log_step_diagnostics``.
        """
        raise NotImplementedError("_forward_and_compute_loss must be implemented by subclass")

    def _maybe_log_step_diagnostics(
        self,
        batch: Any,
        loss_info: dict[str, Any],
        epoch: int,
        batch_idx: int,
    ) -> None:
        """Optional per-step diagnostics (e.g. AEC output energy collapse)."""
        return

    # Subclasses can put either a ``*_safe`` tensor or ``*_safe_min`` / ``*_safe_max``
    # scalars into ``loss_info`` to have its range logged on non-finite loss warnings.
    _NONFINITE_DEBUG_KEYS = ("enhanced_safe", "nearend_estimate_safe")

    def _format_nonfinite_loss_debug(self, loss_info: dict[str, Any]) -> str:
        """Format a short debug string for non-finite loss warnings/errors."""
        parts: list[str] = []
        for key in self._NONFINITE_DEBUG_KEYS:
            val = loss_info.get(key)
            if isinstance(val, torch.Tensor):
                if val.numel() == 0:
                    parts.append(f" {key}: empty")
                else:
                    parts.append(f" {key}: [{val.min():.2e}, {val.max():.2e}]")
                continue
            min_v = loss_info.get(f"{key}_min")
            max_v = loss_info.get(f"{key}_max")
            if isinstance(min_v, (int, float)) and isinstance(max_v, (int, float)):
                parts.append(f" {key}: [{min_v:.2e}, {max_v:.2e}]")
        return "".join(parts)

    @staticmethod
    def _forward_graph_tensors(output: Any) -> list[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return [output] if output.requires_grad and output.grad_fn is not None else []
        if isinstance(output, dict):
            output = output.values()
        elif not isinstance(output, (tuple, list)):
            return []
        return [tensor for value in output
                for tensor in MetricsCapableTrainer._forward_graph_tensors(value)]

    @contextlib.contextmanager
    def _capture_forward_graph(self, model):
        """仅在当前 forward/backward 作用域保存真实输出，退出即释放 hook 与图引用。"""
        outputs: list[torch.Tensor] = []
        handle = model.register_forward_hook(
            lambda _model, _inputs, output: outputs.extend(self._forward_graph_tensors(output))
        )
        try:
            yield outputs
        finally:
            handle.remove()
            outputs.clear()

    def _graph_preserving_zero_loss(
        self, like: torch.Tensor, *, outputs: Any
    ) -> torch.Tensor:
        """沿本次 forward 图执行零 backward；unused 参数交由 DDP reducer 处理。

        禁止图外连接参数或回退到 detached leaf。GAN 必须传当前子步的输出，
        不能重用已 backward 的其他模型图。病态算子的梯度仍由 AMP gate 兜底。
        """
        tensors = self._forward_graph_tensors(outputs)
        if not tensors:
            raise RuntimeError("zero loss requires a differentiable current forward graph")
        zero = torch.zeros((), device=like.device, dtype=like.dtype)
        for tensor in tensors:
            value = torch.view_as_real(tensor) if tensor.is_complex() else tensor
            zero = zero + (torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) * 0).sum()
        return zero

    def _has_persistent_state_repair_pending(self) -> bool:
        """上一次 forward 是否污染了需在 backward 后修复的模型持久状态。

        默认无持久状态修复需求；子类（如 AEC 的 BatchNorm running buffers）
        可重写。返回 True 时本 batch 按 non-finite loss 走零图分支，且
        ``_flush_persistent_state_repair`` 会在 backward 完成后被调用。
        """
        return False

    def _flush_persistent_state_repair(self) -> None:
        """在 ``backward()`` 完成后修复被污染的模型持久状态。默认 no-op。"""
        return None

    def _repair_persistent_state_after_abandoned_graph(self) -> None:
        """当前 graph 因异常不再 backward 时清理持久状态。"""
        self._flush_persistent_state_repair()

    @contextlib.contextmanager
    def _persistent_state_repair_guard(self):
        """异常放弃 graph 时清理持久状态，正常 backward 前绝不改写 buffer。"""
        try:
            yield
        except BaseException:
            self._repair_persistent_state_after_abandoned_graph()
            raise

    def _get_epoch_tail_label(self) -> str:
        """Label passed to ``_handle_epoch_tail_gradients`` for logging.

        Subclasses can set the class attribute ``_train_epoch_tail_label`` to
        override the default empty label.
        """
        return getattr(self, "_train_epoch_tail_label", "")

    def _strict_finite_checks_enabled(self) -> bool:
        """当前 loss 是否会以专用异常执行 strict finite 检查。"""
        loss_fn = self.loss_fn
        if bool(getattr(loss_fn, "STRICT_FINITE_CHECKS", False)):
            return True
        return isinstance(loss_fn, CompositeLoss) and bool(loss_fn.strict)

    def _sync_strict_finite_error(
        self,
        local_error: StrictFiniteLossError | None,
        *,
        phase: str,
    ) -> str | None:
        """同步 strict finite 失败，并保留首个失败 rank 的原始上下文。"""
        if self.world_size <= 1:
            return str(local_error) if local_error is not None else None
        # validation 在 torch.inference_mode() 内调用；collective 的 in-place 更新
        # 必须操作普通 tensor，不能让 helper 在 inference mode 中创建 inference tensor。
        with torch.inference_mode(False):
            all_ok = self._allreduce_bool_and(local_error is None, phase=phase)
            if all_ok:
                return None
            local_text = (
                f"rank {self.rank}: {local_error}" if local_error is not None else None
            )
            messages: list[str | None] = [None] * self.world_size
            dist.all_gather_object(messages, local_text)
        return "; ".join(message for message in messages if message is not None)

    def _sync_train_batch_exception(
        self,
        local_error: Exception | None,
        *,
        phase: str,
    ) -> str | None:
        """在 DDP batch 边界同步普通异常，避免单 rank 提前退出。

        所有 rank 必须在同一静态边界调用。成功仅 tensor all-reduce，失败才
        all_gather_object 汇集异常；全部 rank 在 reducer collective 前一致抛错。
        单 rank 保持原始异常与零 collective 开销。
        """
        if getattr(self, "world_size", 1) <= 1:
            return None
        with torch.inference_mode(False):
            if self._allreduce_bool_and(local_error is None, phase=phase):
                return None
            local_text = (
                f"rank {self.rank}: {type(local_error).__name__}: {local_error}"
                if local_error is not None
                else None
            )
            messages: list[str | None] = [None] * self.world_size
            dist.all_gather_object(messages, local_text)
        return "; ".join(message for message in messages if message is not None) or None

    def _iter_loader_with_synced_fetch(self, loader, *, phase: str):
        """在 DataLoader ``next()`` 边界同步 worker/collate 异常。

        ``for item in loader`` 会在循环体外执行 ``next()``；因此 worker 进程或
        ``collate_fn`` 抛错时，原有 batch body 的异常同步根本不会被调用。这里
        把 iterator 初始化和每次取 batch 都放进同一个 DDP 边界。成功路径同步
        异常与 exhaustion 状态，失败路径才收集异常文本；单卡仍保持零 collective。
        """
        iterator = None
        local_error: Exception | None = None
        try:
            iterator = iter(loader)
        except Exception as exc:
            local_error = exc
        error_context = None
        if getattr(self, "world_size", 1) > 1:
            error_context = self._sync_train_batch_exception(
                local_error,
                phase=f"{phase}_iterator_init_error",
            )
        if error_context is not None:
            raise RuntimeError(
                f"{phase} DataLoader iterator initialization failed: {error_context}"
            ) from local_error
        if local_error is not None:
            raise local_error
        assert iterator is not None

        while True:
            item = None
            local_error = None
            exhausted = False
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
            except Exception as exc:
                local_error = exc
            error_context = None
            if getattr(self, "world_size", 1) > 1:
                error_context = self._sync_train_batch_exception(
                    local_error,
                    phase=f"{phase}_fetch_error",
                )
            if error_context is not None:
                raise RuntimeError(
                    f"{phase} DataLoader fetch failed: {error_context}"
                ) from local_error
            if local_error is not None:
                raise local_error
            if getattr(self, "world_size", 1) > 1:
                # validation 运行在 inference_mode 中；collective 的归约 tensor
                # 必须显式恢复为普通 tensor，避免 in-place all_reduce 被禁止。
                with torch.inference_mode(False):
                    all_exhausted, all_active = self._allreduce_bool_ands(
                        (exhausted, not exhausted),
                        phase=f"{phase}_exhaustion",
                    )
                if not all_exhausted and not all_active:
                    raise RuntimeError(
                        f"{phase} DataLoader exhausted on only a subset of ranks"
                    )
                if all_exhausted:
                    return
            elif exhausted:
                return
            yield item

    def _run_synced_validation_subphase(
        self,
        operation: Callable[[], Any],
        *,
        phase: str,
        context: str,
    ) -> Any:
        """在 validation loss collective 前同步模型与参数构造异常。"""
        local_error: Exception | None = None
        result: Any = None
        try:
            result = operation()
        except Exception as exc:
            if _is_unrecoverable_cuda_exception(exc):
                raise
            local_error = exc
        error_context = self._sync_train_batch_exception(local_error, phase=phase)
        if error_context is not None:
            raise RuntimeError(f"{context}: {error_context}") from local_error
        if local_error is not None:
            raise local_error
        return result

    def _sync_train_window_finite_eligibility(
        self, local_has_finite_loss: bool, *, phase: str
    ) -> bool:
        """同步 accumulation window 的 finite eligibility。

        这里需要的是跨 rank 的 OR：只要一个 rank 在窗口内有 finite loss，
        该窗口仍可用 DDP 聚合后的梯度更新；只有所有 rank 都没有 finite
        loss 时才跳步。通过对 ``not local`` 做 MIN all-reduce 实现 OR，
        保证每个 rank 在同一窗口执行固定的 collective。
        """
        return not self._allreduce_bool_and(
            not bool(local_has_finite_loss), phase=phase
        )

    # ------------------------------------------------------------------
    # 输出 sanitize 统计（A23：train/val 一致约束 + 异常按 epoch 汇总上报）
    # ------------------------------------------------------------------

    def _reset_output_sanitize_stats(self) -> None:
        """每个 train/val epoch 开始时清零输出修正计数。"""
        self._sanitize_n_nonfinite = torch.zeros((), dtype=torch.int64, device=self.device)
        self._sanitize_n_clamped = torch.zeros((), dtype=torch.int64, device=self.device)
        self._sanitize_stats_global = False

    def _accumulate_output_sanitize(
        self, n_nonfinite: torch.Tensor | int, n_clamped: torch.Tensor | int
    ) -> None:
        """在输出所在设备累加本 epoch 的元素级修正计数，不触发 host 同步。"""
        if not torch.is_tensor(n_nonfinite):
            n_nonfinite = torch.tensor(n_nonfinite, dtype=torch.int64, device=self.device)
        if not torch.is_tensor(n_clamped):
            n_clamped = torch.tensor(n_clamped, dtype=torch.int64, device=self.device)
        if getattr(self, "_sanitize_n_nonfinite", None) is None:
            self._sanitize_n_nonfinite = torch.zeros((), dtype=torch.int64, device=n_nonfinite.device)
            self._sanitize_n_clamped = torch.zeros((), dtype=torch.int64, device=n_clamped.device)
        self._sanitize_n_nonfinite += n_nonfinite
        self._sanitize_n_clamped += n_clamped

    def _sanitize_model_output(self, wav: torch.Tensor) -> torch.Tensor:
        """统一 train/val 输出约束并累加 epoch 计数（语义见 ``sanitize_model_output``）。"""
        # 保留 getattr 缺省：测试替身可能绕过 __init__ 直接调用本方法。
        if getattr(self, "output_clamp", True):
            out, n_nonfinite, n_clamped = sanitize_model_output(wav)
        else:
            # 官方 PASE 家族训练不做 [-1,1] 截断；仅保留非有限值清零。
            finite = torch.isfinite(wav)
            n_nonfinite = (~finite).sum(dtype=torch.int64)
            out = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
            n_clamped = torch.zeros((), dtype=torch.int64, device=wav.device)
        self._accumulate_output_sanitize(n_nonfinite, n_clamped)
        return out

    def _allreduce_output_sanitize_stats(self) -> None:
        """在 epoch boundary 按固定顺序 SUM DDP sanitize 计数。"""
        n_nonfinite = getattr(self, "_sanitize_n_nonfinite", None)
        n_clamped = getattr(self, "_sanitize_n_clamped", None)
        if n_nonfinite is None or n_clamped is None:
            n_nonfinite = torch.zeros((), dtype=torch.int64, device=self.device)
            n_clamped = torch.zeros((), dtype=torch.int64, device=self.device)
        if self.world_size > 1:
            # 固定 [non-finite, clamped] 顺序，所有 rank 在相同 epoch 位置执行一次 SUM。
            counts = torch.stack((n_nonfinite, n_clamped))
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            self._sanitize_n_nonfinite = counts[0]
            self._sanitize_n_clamped = counts[1]
        else:
            self._sanitize_n_nonfinite = n_nonfinite
            self._sanitize_n_clamped = n_clamped
        self._sanitize_stats_global = True

    def _log_output_sanitize_summary(self, epoch: int, tag: str) -> None:
        """每 epoch 最多一条汇总 warning：输出被 sanitize/clamp 的元素数（A23）。"""
        # 汇总点才允许一次性物化设备计数；正常 step 不触发 host sync。
        if not getattr(self, "_sanitize_stats_global", False):
            self._allreduce_output_sanitize_stats()
        n_nonfinite_i = int(self._sanitize_n_nonfinite.item())
        n_clamped_i = int(self._sanitize_n_clamped.item())
        if self.rank == 0 and (n_nonfinite_i > 0 or n_clamped_i > 0):
            logger.warning(
                f"{tag} E{epoch}: model output sanitized — "
                f"non-finite elements zeroed: {n_nonfinite_i}, "
                f"out-of-range clamped to [-1, 1]: {n_clamped_i}"
            )

    def _val_iter_is_ddp_padding(self, n_loops: int) -> bool:
        """当前 val 迭代是否为 DDP 尾部对齐复制的 batch。

        ``GroupedBatchSampler`` 为 DDP 对齐在末尾复制 batch（broadcast_buffers=True
        时每次 forward 都有 buffer 广播 collective，各 rank 迭代数必须一致），这些
        batch 的 forward 必须照常执行以维持同步，但其 loss/指标贡献不得计入，否则
        验证集样本被重复计数。非 ``GroupedBatchSampler``、单卡或未补齐时恒返回
        ``False``（行为与改动前一致）。
        """
        bs = getattr(self.val_loader, "batch_sampler", None)
        real = getattr(bs, "_last_iter_real_len", None)
        return real is not None and n_loops > real

    # ------------------------------------------------------------------
    # 验证循环共享片段（SE / AEC / GAN / DualInput 四个 _validation_epoch 复用）
    #
    # DDP 约束：下列含 collective 的 helper 必须保持各 rank 调用序列完全一致
    # （调用与否不得依赖本地数据，只能依赖各 rank 一致的静态/配置状态）。
    # ------------------------------------------------------------------

    def _accumulate_val_loss(
        self,
        loss: torch.Tensor,
        total_loss_gpu: torch.Tensor,
        n: int,
        is_ddp_pad: bool,
        epoch: int,
        n_loops: int,
        weight: int = 1,
    ) -> tuple[torch.Tensor, int, bool]:
        """按真实输入行数累计 val loss；DDP padding 与非有限 loss 不计入均值。

        非有限 loss 的 batch 跳过、不占分母；每个 epoch 前 3 个此类 batch 在
        rank 0 打 warning。返回 ``(total_loss_gpu, n, loss_was_finite)``。
        """
        loss_det = loss.detach()
        loss_was_finite = bool(torch.isfinite(loss_det).item())
        if loss_was_finite and not is_ddp_pad:
            total_loss_gpu += loss_det * weight
            n += weight
        elif not is_ddp_pad and self.rank == 0 and n_loops <= 3:
            logger.warning(
                f"[Val] E{epoch} batch {n_loops}: non-finite loss, "
                "skipped (excluded from mean)"
            )
        return total_loss_gpu, n, loss_was_finite

    def _accumulate_val_loss_comp(
        self,
        val_loss_comp_sums: dict[str, float],
        loss_was_finite: bool,
        is_ddp_pad: bool,
        weight: int = 1,
    ) -> None:
        """按真实输入行数累计 CompositeLoss 加权分量（in-place）。"""
        if loss_was_finite and not is_ddp_pad:
            bd = getattr(self.loss_fn, "_last_breakdown", None)
            if bd and val_loss_comp_sums:
                for name, _w, _raw, weighted in bd:
                    wv = float(weighted)
                    if math.isfinite(wv):
                        val_loss_comp_sums[name] = (
                            val_loss_comp_sums.get(name, 0.0) + wv * weight
                        )

    def _append_val_metric_pairs(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        fs_val: int,
        pairs: list[tuple[np.ndarray, np.ndarray, int]],
        total_pairs_for_metrics: int,
    ) -> int:
        """逐 utterance 收集 (ref, enh, fs) 指标对，返回更新后的 ``total_pairs_for_metrics``。

        跳过近静音参考（noise-only / echo-only 分支的安全过滤）；总数达到
        ``_vm_max_metrics``（per-rank 上限）后停止收集。
        """
        enh_cpu = enhanced.detach().float().cpu()
        cln_cpu = clean.float().cpu()
        lens_cpu = lengths.cpu()
        batch_size = cln_cpu.shape[0]
        for i in range(batch_size):
            if total_pairs_for_metrics >= self._vm_max_metrics:
                break
            utt_len = int(lens_cpu[i].item())
            ref_np = cln_cpu[i, :utt_len].numpy()
            # Skip near-silent references (safety filter for noise-only / echo-only branches)
            if np.max(np.abs(ref_np)) < 1e-6:
                continue
            pairs.append((ref_np, enh_cpu[i, :utt_len].numpy(), fs_val))
            total_pairs_for_metrics += 1
        return total_pairs_for_metrics

    def _allreduce_val_loss_stats(
        self,
        total_loss_gpu: torch.Tensor,
        finite_loss_samples: int,
        total_samples: int,
    ) -> tuple[float, float, float]:
        """原子归约 loss sum、finite loss 样本数与总样本数。"""
        stats = torch.tensor(
            [
                float(total_loss_gpu.item()),
                float(finite_loss_samples),
                float(total_samples),
            ],
            device=self.device,
            dtype=torch.float64,
        )
        if self.world_size > 1:
            self._nccl_flow_log("val_loss_stats_all_reduce")
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return tuple(float(value) for value in stats.tolist())

    def _allreduce_val_loss_sum_count(
        self, total_loss_gpu: torch.Tensor, n: int
    ) -> tuple[float, float]:
        """兼容旧验证循环：归约 loss sum 与 finite loss 样本数。"""
        loss_sum, n_global, _ = self._allreduce_val_loss_stats(
            total_loss_gpu, n, n
        )
        return loss_sum, n_global

    def _log_val_loss_coverage(
        self,
        finite_loss_samples: float,
        total_samples: float,
        val_x: int,
    ) -> None:
        """记录 validation loss 的有限样本覆盖率。"""
        if self.rank != 0:
            return
        coverage = (
            finite_loss_samples / total_samples if total_samples > 0 else 0.0
        )
        self._log_scalar("val/finite_loss_samples", finite_loss_samples, val_x)
        self._log_scalar("val/total_samples", total_samples, val_x)
        self._log_scalar("val/finite_loss_coverage_ratio", coverage, val_x)

    def _allreduce_val_loss_comp_sums(
        self,
        val_loss_comp_sums: dict[str, float],
        comp_names: list[str] | None,
    ) -> None:
        """Composite 分量：allreduce 各 rank 的分量加和（in-place 更新 dict）。

        只要 ``comp_names`` 非 None 就必须 allreduce，不得用 dict 是否为空做门控
        ——空 dict 会跳过 allreduce，各 rank 不一致时死锁。
        """
        if self.world_size > 1 and comp_names is not None:
            t = torch.tensor(
                [val_loss_comp_sums.get(nm, 0.0) for nm in comp_names],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            for i, nm in enumerate(comp_names):
                val_loss_comp_sums[nm] = float(t[i].item())

    def _val_avg_and_curve_x(
        self, loss_sum: float, n_global: float, epoch: int
    ) -> tuple[float, int]:
        """全局均值与曲线 x 轴（step-only 验证用 optimizer step，否则 epoch）。"""
        avg = (loss_sum / n_global) if n_global > 0 else float("nan")
        val_x = self.n_steps if getattr(self, "_curve_x_is_step", False) else epoch
        return avg, val_x

    def _log_val_loss_summary(
        self,
        epoch: int,
        avg: float,
        n_global: float,
        val_loss_comp_sums: dict[str, float],
        t_forward_sec: float,
        val_x: int,
    ) -> None:
        """rank-0 汇总：写 val/loss 与 composite 分量的 TB/控制台日志，并更新 ``_last_val_comp``。"""
        if self.rank == 0:
            if n_global > 0:
                self._log_scalar("val/loss", avg, val_x)
            else:
                logger.warning(
                    f"[Val] E{epoch}: no finite val loss samples (n=0), "
                    "val/loss not logged"
                )
            if val_loss_comp_sums and n_global > 0:
                inv_n = 1.0 / n_global
                self._last_val_comp = {
                    name: float(s * inv_n) for name, s in val_loss_comp_sums.items()
                }
                for name, s in val_loss_comp_sums.items():
                    v = s * inv_n
                    self._log_scalar(f"val/loss_comp/{name}", v, val_x)
                comp_str = " | ".join(
                    f"{LOSS_DISPLAY_NAMES.get(name, name)}={s * inv_n:.4f}"
                    for name, s in val_loss_comp_sums.items()
                )
                if getattr(self, "_curve_x_is_step", False):
                    logger.info(
                        f"  [Val] step {val_x} (epoch {epoch}) | vm_count={self._vm_epoch_count} | loss {avg:.4f} | forward {t_forward_sec:.1f}s | {comp_str}"
                    )
                else:
                    logger.info(
                        f"  [Val] epoch {epoch} | vm_count={self._vm_epoch_count} | loss {avg:.4f} | forward {t_forward_sec:.1f}s | {comp_str}"
                    )
            else:
                self._last_val_comp = {}
                if getattr(self, "_curve_x_is_step", False):
                    logger.info(
                        f"  [Val] step {val_x} (epoch {epoch}) | vm_count={self._vm_epoch_count} | loss {avg:.4f} | forward {t_forward_sec:.1f}s"
                    )
                else:
                    logger.info(f"  [Val] epoch {epoch} | vm_count={self._vm_epoch_count} | loss {avg:.4f} | forward {t_forward_sec:.1f}s")

    # ------------------------------------------------------------------
    # Shared validation-loop template（SE / TSE / AEC 共用骨架）
    #
    # 任务差异全部经下列 hooks 注入；模板本身不依赖任何任务特定 batch 类型。
    # DDP 约束：hooks 内含 collective 的实现（如 AEC state allreduce）必须保持
    # 各 rank 调用序列完全一致（门控只能依赖各 rank 一致的静态/配置状态）。
    # ------------------------------------------------------------------

    def _val_forward_loss(
        self, raw_batch: Any, epoch: int, n_loops: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Any]:
        """单 batch 验证前向：unpack → device → strict 检查 → AMP autocast 内
        forward → 输出 sanitize → loss。

        子类必须实现。返回 ``(loss, enhanced, clean, lengths, fs_val, ctx)``；
        ``ctx`` 为不透明任务上下文，原样透传给 ``_val_accumulate_extra`` 与
        ``_collect_val_metric_pairs``（AEC 用它携带 data/state/mic/farend/has_echo）。
        """
        raise NotImplementedError("_val_forward_loss must be implemented by subclass")

    def _val_loss_weight(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        ctx: Any,
    ) -> int:
        """返回 loss 均值代表的真实输入行数；TSE 多目标验证覆写为 target rows。"""
        return int(clean.shape[0])

    def _init_val_extra_state(self) -> Any:
        """val epoch 级额外累加器（AEC：erle_pairs / state 计数）；默认无。"""
        return None

    def _val_accumulate_extra(self, extra: Any, ctx: Any, is_ddp_pad: bool) -> None:
        """loss 累计之后的 per-batch 额外累计（AEC：state detector 计数）；默认无操作。"""
        return None

    def _collect_val_metric_pairs(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        fs_val: int,
        pairs: list[tuple[np.ndarray, np.ndarray, int]],
        total_pairs_for_metrics: int,
        extra: Any,
        ctx: Any,
    ) -> int:
        """收集 objective metrics 的 (ref, enh, fs) 对；默认走 ``_append_val_metric_pairs``。

        AEC 覆写：同一循环内附带收集 ERLE pairs（需 mic/farend/has_echo，经 ``ctx`` 传入）。
        """
        return self._append_val_metric_pairs(
            enhanced, clean, lengths, fs_val, pairs, total_pairs_for_metrics
        )

    def _val_discard_aborted_extra(self, extra: Any) -> None:
        """清空中止验证的任务侧本地累计；默认无额外状态。"""
        return None

    def _val_post_allreduce_extra(self, extra: Any, val_x: int) -> None:
        """val loss allreduce 之后、日志之前的额外汇总（AEC：state 计数 allreduce）；默认无操作。"""
        return None

    def _val_log_extra_metrics(self, extra: Any, val_x: int) -> None:
        """``_log_val_loss_summary`` 之后的额外指标日志（AEC：state accuracy/F1）；默认无操作。"""
        return None

    def _val_run_extra_metrics(
        self, epoch: int, extra: Any, do_metrics: bool, val_x: int
    ) -> bool:
        """运行任务侧额外指标并返回全局一致的 validation aborted 状态。"""
        return False

    def _val_task_metric_results(
        self, extra: Any
    ) -> dict[str, tuple[float, float, float]]:
        """返回任务侧已全局归约的 ``mean, valid, eligible`` 指标。"""
        return {}

    def _validation_epoch_resources(self):
        """返回 validation epoch 级资源 context；任务侧可覆写。"""
        return contextlib.nullcontext()

    def _val_prepare_batch(self, raw_batch: Any, epoch: int, n_loops: int) -> Any:
        """在任何 DDP forward 前执行并同步的 validation prepare hook。"""
        return raw_batch

    def _val_forward_prepared(
        self, prepared_batch: Any, epoch: int, n_loops: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Any]:
        """消费已通过跨 rank prepare 门禁的 batch；默认兼容旧任务 hook。"""
        return self._val_forward_loss(prepared_batch, epoch, n_loops)

    def _validation_epoch(self, epoch: int) -> Optional[float]:
        """在 epoch 资源 context 中运行统一 validation 模板。"""
        with self._validation_epoch_resources():
            return self._validation_epoch_impl(epoch)

    def _validation_epoch_impl(self, epoch: int) -> Optional[float]:
        """Unified validation epoch template（SE / TSE / AEC 共用骨架）。

        骨架顺序：sanitize 计数清零 → composite 分量容器 → metrics 触发判定 →
        公共无 collective batch-loop（DDP padding → task forward → loss/extra/pairs）→
        loss 与分量 allreduce → 全局均值与日志 → sanitize 汇总 → objective/extra
        metrics → 清 CUDA cache。

        非有限 loss 的 batch 不计入 loss 均值，但不默认过滤 objective metric pairs；
        整轮全部非有限时返回 NaN 且不提交该轮 metric history。BaseTrainer 对该 NaN
        保持 best/scheduler/early-stop fail-closed。
        """
        from core.trainer.validation_orchestration import (
            ValidationBatchResult,
            run_validation_batch_loop,
        )

        trainer = self
        self._reset_output_sanitize_stats()
        comp_names = self._comp_keys
        val_loss_comp_sums: dict[str, float] = (
            {name: 0.0 for name in comp_names} if comp_names else {}
        )
        do_metrics = self._advance_and_should_run_metrics()
        extra = self._init_val_extra_state()

        class _TaskValidationAdapter:
            def prepare_batch(self, raw_batch, current_epoch, loop_count):
                return trainer._val_prepare_batch(
                    raw_batch, current_epoch, loop_count
                )

            def forward_batch(self, prepared_batch, current_epoch, loop_count):
                # 普通异常与 StrictFiniteLossError 必须走同一个外层阶段
                # collective；嵌套另一套 strict collective 会在 rank-local 普通
                # 异常时造成 all_reduce/all_gather 次序错位。
                result = trainer._val_forward_prepared(
                    prepared_batch, current_epoch, loop_count
                )
                loss, enhanced, clean, lengths, fs_val, ctx = result
                return ValidationBatchResult(
                    loss, enhanced, clean, lengths, fs_val, ctx
                )

            def loss_weight(self, batch):
                return trainer._val_loss_weight(
                    batch.enhanced, batch.clean, batch.lengths, batch.context
                )

            def after_loss(
                self,
                batch,
                *,
                is_ddp_padding,
                loss_was_finite,
                weight,
            ):
                trainer._accumulate_val_loss_comp(
                    val_loss_comp_sums,
                    loss_was_finite,
                    is_ddp_padding,
                    weight=weight,
                )
                trainer._val_accumulate_extra(
                    extra, batch.context, is_ddp_padding
                )

            def collect_metric_pairs(self, batch, pairs, total_pairs):
                return trainer._collect_val_metric_pairs(
                    batch.enhanced,
                    batch.clean,
                    batch.lengths,
                    batch.fs,
                    pairs,
                    total_pairs,
                    extra,
                    batch.context,
                )

            def on_max_batches(self, current_epoch, max_batches):
                if trainer.rank == 0:
                    logger.info(
                        f"  [Val] capped at max_val_batches={max_batches}"
                    )

        loop_result = run_validation_batch_loop(
            self,
            epoch,
            _TaskValidationAdapter(),
            collect_metrics=do_metrics,
        )
        aborted = bool(loop_result.aborted)
        if aborted:
            # 中止验证仍必须执行原有固定 reductions；所有本地贡献先清零，确保
            # partial loss/components/state/ERLE 不进入日志、best 或 history。
            loop_result.loss_sum.zero_()
            local_finite_loss_samples = 0
            local_total_samples = 0
            for name in val_loss_comp_sums:
                val_loss_comp_sums[name] = 0.0
            self._val_discard_aborted_extra(extra)
        else:
            local_finite_loss_samples = loop_result.finite_loss_samples
            local_total_samples = loop_result.total_samples
        loss_sum, n_global, total_samples_global = self._allreduce_val_loss_stats(
            loop_result.loss_sum,
            local_finite_loss_samples,
            local_total_samples,
        )
        self._allreduce_val_loss_comp_sums(val_loss_comp_sums, comp_names)

        avg, val_x = self._val_avg_and_curve_x(loss_sum, n_global, epoch)
        val_loss_components = (
            {
                name: float(total / n_global)
                for name, total in val_loss_comp_sums.items()
                if math.isfinite(float(total))
            }
            if n_global > 0
            else {}
        )
        self._val_post_allreduce_extra(extra, val_x)
        extra_aborted = bool(
            self._val_run_extra_metrics(epoch, extra, do_metrics, val_x)
        )
        aborted = aborted or extra_aborted
        metrics_aborted = self._run_validation_metrics_from_pairs(
            epoch,
            loop_result.metric_pairs,
            do_metrics,
            val_x=val_x,
            validation_aborted=aborted,
            commit_history=n_global > 0,
            task_metric_results=self._val_task_metric_results(extra),
            val_loss_components=val_loss_components,
        )
        aborted = aborted or metrics_aborted

        if aborted:
            self._val_discard_aborted_extra(extra)
            self._vm_epoch_count -= 1
        else:
            self._log_val_loss_summary(
                epoch,
                avg,
                n_global,
                val_loss_comp_sums,
                loop_result.forward_seconds,
                val_x,
            )
            self._log_val_loss_coverage(n_global, total_samples_global, val_x)
            self._val_log_extra_metrics(extra, val_x)
            self._log_output_sanitize_summary(epoch, "[Val]")

        return None if aborted else avg

    def _train_epoch(
        self,
        epoch: int,
        step_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> Optional[float]:
        """Unified training epoch template.

        Subclasses provide task-specific behaviour through the protected hooks
        above.  This method owns the common skeleton: tqdm, device transfer,
        gradient accumulation, AMP, DDP ``no_sync``, NaN safety, optimizer stepping,
        EMA updates, epoch-tail handling, and DDP allreduce/logging.
        """
        if self._sync_max_steps_reached():
            self._train_epoch_completed = False
            self._last_completed_batch_cursor = int(
                getattr(self, "_resume_skip_batches", 0)
            )
            return None

        epoch_wall_start = time.perf_counter()
        running_loss_gpu = torch.zeros(1, device=self.device)
        n_batches = 0
        accum = self.accumulation_steps
        accum_count = 0
        window_has_finite_loss = False
        loss: Optional[torch.Tensor] = None
        nan_loss_count = 0
        strict_failure_seen = False
        strict_failure_message: str | None = None
        self._reset_output_sanitize_stats()
        silence_placeholder_samples = 0
        total_samples_this_epoch = 0
        # loss/breakdown 先暂存在 accumulation window；只有真实 optimizer step
        # 才合入 epoch 统计，避免停机、AMP overflow、tail drop 等未提交梯度污染曲线。
        window_loss_sum = torch.zeros(1, device=self.device)
        window_n = 0
        window_silence_placeholder_samples = 0
        window_total_samples = 0
        window_comp_weighted: dict[str, torch.Tensor] = {}
        window_comp_count: dict[str, int] = {}
        epoch_comp_weighted: dict[str, torch.Tensor] = {}
        epoch_comp_count: dict[str, int] = {}
        # data_wait_sec 明确定义为上一 batch 结束到本 batch prepare 完成的 host gap，
        # 包含 DataLoader yield、unpack、H2D 与 task prepare；CUDA compute 使用采样
        # event，只在 epoch 末 synchronize，避免每 step 强制 host 等待 GPU。
        timing_interval = max(1, int(getattr(self, "log_interval", 100)))
        data_wait_sec = 0.0
        compute_wall_sec = 0.0
        timing_samples = 0
        timing_task_rows = 0
        timing_sampled_batches = 0
        cuda_timing = self.device.type == "cuda" and torch.cuda.is_available()
        cuda_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        last_batch_end = time.perf_counter()
        timing_batches = 0

        def _finish_batch_timing(
            start_event: torch.cuda.Event | None,
            compute_start: float,
            batch_size: int,
            sampled: bool,
        ) -> None:
            nonlocal compute_wall_sec, timing_sampled_batches
            nonlocal timing_batches, last_batch_end
            timing_batches += 1
            if sampled:
                timing_sampled_batches += 1
            if start_event is None:
                compute_wall_sec += max(0.0, time.perf_counter() - compute_start)
            else:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                cuda_event_pairs.append((start_event, end_event))
            last_batch_end = time.perf_counter()

        def _commit_train_step_window() -> None:
            """将已真实更新参数的 accumulation window 写入 epoch 统计。"""
            nonlocal n_batches, window_n, silence_placeholder_samples
            nonlocal total_samples_this_epoch, window_silence_placeholder_samples
            nonlocal window_total_samples
            if window_n <= 0:
                return
            running_loss_gpu.add_(window_loss_sum)
            n_batches += window_n
            silence_placeholder_samples += window_silence_placeholder_samples
            total_samples_this_epoch += window_total_samples
            self._train_log_loss_win.append(window_loss_sum.item() / window_n)
            for name, value in window_comp_weighted.items():
                prev = epoch_comp_weighted.get(name)
                epoch_comp_weighted[name] = value if prev is None else prev + value
                epoch_comp_count[name] = (
                    epoch_comp_count.get(name, 0) + window_comp_count[name]
                )
            window_loss_sum.zero_()
            window_n = 0
            window_silence_placeholder_samples = 0
            window_total_samples = 0
            window_comp_weighted.clear()
            window_comp_count.clear()

        def _discard_train_step_window() -> None:
            """丢弃没有真实 optimizer step 的 accumulation window 统计。"""
            nonlocal window_n, window_silence_placeholder_samples, window_total_samples
            window_loss_sum.zero_()
            window_n = 0
            window_silence_placeholder_samples = 0
            window_total_samples = 0
            window_comp_weighted.clear()
            window_comp_count.clear()

        stopped_by_max_steps = False
        stopped_by_shutdown = False
        completed_batch_cursor = 0
        self._train_epoch_completed = False
        pbar = self._make_train_progress_bar(epoch)

        # A22：从 middle checkpoint resume 时，本 epoch 跳过前 N 个已完整完成的 batch
        # （cursor 已对齐 accumulation 边界，accum_count 从 0 开始是精确的）。
        # sampler 顺序由 seed+epoch 确定性派生，跳过即复现 crash 时的位置。
        skip_batches = self._resume_skip_batches if epoch == self.start_epoch else 0
        # 立即停机时必须保留已有 resume cursor；不能回退为 0 或跳过当前 epoch。
        completed_batch_cursor = int(skip_batches)
        if skip_batches > 0 and self.rank == 0:
            logger.info(
                f"  Resuming epoch {epoch} from middle checkpoint: "
                f"skipping first {skip_batches} batches"
            )

        shutdown_synced_at_last_batch = False
        for bi, raw_batch in self._iter_loader_with_synced_fetch(
            pbar, phase="train_loader"
        ):
            if bi < skip_batches:
                continue
            shutdown_synced_at_last_batch = False
            local_prepare_error: Exception | None = None
            prepared = None
            try:
                prepared = self._prepare_batch(raw_batch, epoch, bi)
            except Exception as exc:
                if _is_unrecoverable_cuda_exception(exc):
                    raise
                local_prepare_error = exc
            prepare_error_context = self._sync_train_batch_exception(
                local_prepare_error,
                phase="train_prepare_batch_error",
            )
            if prepare_error_context is not None:
                raise RuntimeError(
                    f"[E{epoch} B{bi}] batch preparation failed: {prepare_error_context}"
                ) from local_prepare_error
            if local_prepare_error is not None:
                raise local_prepare_error
            assert prepared is not None
            batch, fs_val, n_placeholders = prepared
            compute_start = time.perf_counter()
            data_wait_sec += max(0.0, compute_start - last_batch_end)
            batch_size_for_timing, task_rows_for_timing = self._throughput_row_counts(
                batch
            )
            timing_samples += batch_size_for_timing
            timing_task_rows += task_rows_for_timing
            timing_sampled = (bi % timing_interval == 0)
            if timing_sampled and cuda_timing:
                timing_start_event = torch.cuda.Event(enable_timing=True)
                timing_start_event.record()
            else:
                timing_start_event = None

            is_last_accum = (accum_count + 1 >= accum)
            sync_ctx = (
                contextlib.nullcontext()
                if (is_last_accum or self.world_size <= 1)
                else self._no_sync_all()
            )

            with self._persistent_state_repair_guard(), sync_ctx, self._capture_forward_graph(self.model) as forward_graph:
                local_strict_error: StrictFiniteLossError | None = None
                local_forward_error: Exception | None = None
                with torch.amp.autocast(
                    self.device.type, enabled=self.use_amp, dtype=self.amp_dtype
                ):
                    try:
                        loss, loss_was_finite, loss_info = (
                            self._forward_and_compute_loss(batch, epoch, bi, fs_val)
                        )
                        loss = loss / accum
                    except StrictFiniteLossError as exc:
                        local_strict_error = exc
                        loss = torch.zeros((), device=self.device)
                        loss_was_finite = False
                        loss_info = {}
                    except Exception as exc:
                        if _is_unrecoverable_cuda_exception(exc):
                            raise
                        local_forward_error = exc

                if local_forward_error is None and not forward_graph:
                    local_forward_error = RuntimeError(
                        "training requires a differentiable current forward graph"
                    )
                forward_error_context = self._sync_train_batch_exception(
                    local_forward_error,
                    phase="train_forward_loss_error",
                )
                if forward_error_context is not None:
                    raise RuntimeError(
                        f"[E{epoch} B{bi}] forward/loss failed: {forward_error_context}"
                    ) from local_forward_error
                if local_forward_error is not None:
                    raise local_forward_error

                if self._strict_finite_checks_enabled():
                    strict_error_context = self._sync_strict_finite_error(
                        local_strict_error,
                        phase="train_strict_finite_error",
                    )
                    if strict_error_context is not None:
                        strict_failure_seen = True
                        loss_was_finite = False
                        if strict_failure_message is None:
                            strict_failure_message = (
                                f"[E{epoch} B{bi}] strict finite loss failed: "
                                f"{strict_error_context}"
                            )

                # forward 污染了模型持久状态（如 BatchNorm running buffers）时，
                # 该 batch 梯度不可信，按 non-finite loss 走零图分支；buffer 修复
                # 推迟到 backward 之后（见下方 _flush_persistent_state_repair），
                # 否则 BN backward 保存的 running statistics 版本会失效。
                if self._has_persistent_state_repair_pending():
                    loss_was_finite = False

                # strict_checks 下也必须先同步本 batch 的 finite 状态，再决定是否
                # 使用零图 backward；否则某个 rank 直接 raise 会让其他 rank 卡在 DDP
                # gradient all-reduce。一次失败会使本 epoch 后续窗口全部跳过更新，待
                # 固定的 epoch 末 collective 完成后所有 rank 再一致抛错。
                if self._strict_checks and self.world_size > 1:
                    all_ranks_loss_finite = self._allreduce_bool_and(
                        bool(loss_was_finite), phase="train_strict_loss_finite"
                    )
                else:
                    all_ranks_loss_finite = bool(loss_was_finite)
                if self._strict_checks:
                    if not all_ranks_loss_finite:
                        strict_failure_seen = True
                        if strict_failure_message is None:
                            debug = self._format_nonfinite_loss_debug(loss_info)
                            strict_failure_message = (
                                f"[E{epoch} B{bi}] non-finite loss under strict_checks.{debug}"
                            )
                    if strict_failure_seen:
                        loss_was_finite = False

                if not loss_was_finite:
                    debug = self._format_nonfinite_loss_debug(loss_info)
                    if self._strict_checks and self.world_size <= 1:
                        # 单 rank strict 路径不会再执行 backward；若本次 forward
                        # 污染了 BN running buffers，此时立即修复已无版本风险。
                        self._flush_persistent_state_repair()
                        raise RuntimeError(
                            strict_failure_message
                            or f"[E{epoch} B{bi}] non-finite loss under strict_checks.{debug}"
                        )
                    nan_loss_count += 1
                    if not self._strict_checks and nan_loss_count <= 3 and self.rank == 0:
                        logger.warning(
                            f"[E{epoch} B{bi}] NaN/Inf loss (#{nan_loss_count} this epoch), "
                            f"using zero loss.{debug}"
                        )
                    # 用保留模型计算图的全零 loss 做 backward（而非 detached
                    # leaf），保证 DDP gradient hooks 正常注册，避免其他 rank
                    # 在 all-reduce 处永久等待；零梯度不污染累积与参数更新。
                    loss = self._graph_preserving_zero_loss(loss, outputs=forward_graph)

                self.scaler.scale(loss).backward()

                # backward 已消费完 BN backward 保存的统计版本，此时修复被
                # 污染的持久 buffer（如 BatchNorm running_mean/running_var）
                # 不会再触发 inplace version 错误。
                self._flush_persistent_state_repair()

            self._maybe_log_step_diagnostics(batch, loss_info, epoch, bi)

            accum_count += 1
            loss_detached = loss.detach() * accum
            # 仅 finite batch 暂存到当前窗口；窗口只有真实 step 后才提交到 epoch。
            # NaN batch 的 loss 已被替换为保留计算图的全零 loss，不会污染平均。
            window_has_finite_loss = window_has_finite_loss or loss_was_finite
            if loss_was_finite:
                window_loss_sum.add_(loss_detached)
                window_n += 1
                window_silence_placeholder_samples += n_placeholders
                window_total_samples += self._batch_size_from_batch(batch)

            bd = getattr(self.loss_fn, "_breakdown_tensors", None)
            if bd and loss_was_finite:
                # 直接消费 detached tensor 并 GPU 累计，避免每 micro-batch 经
                # _last_breakdown property 物化 .item()。
                for name, _w, _raw, weighted in bd:
                    prev = window_comp_weighted.get(name)
                    window_comp_weighted[name] = (
                        weighted if prev is None else prev + weighted
                    )
                    window_comp_count[name] = window_comp_count.get(name, 0) + 1

            max_steps_reached = False
            if is_last_accum:
                # 停机只在 optimizer boundary 同步，避免每个 microbatch 的阻塞
                # all_reduce；收到请求时当前未提交窗口整体丢弃，cursor 不前移。
                shutdown_synced_at_last_batch = True
                if self._sync_shutdown_signum():
                    stopped_by_shutdown = True
                    for opt in self.optimizers.values():
                        opt.zero_grad(set_to_none=True)
                    accum_count = 0
                    window_has_finite_loss = False
                    _discard_train_step_window()
                    if self.rank == 0:
                        logger.info(
                            f"  Shutdown requested, dropping train window ending "
                            f"at batch {bi}."
                        )
                else:
                    # 所有 rank 都在 accumulation 边界执行相同 collective 序列；
                    # 窗口提交后 cursor 才能前移，max_steps 也只在此同步判定。
                    window_step_allowed = self._sync_train_window_finite_eligibility(
                        window_has_finite_loss,
                        phase="train_window_finite_eligibility",
                    ) and not strict_failure_seen
                    stepped = self._optimizer_step(skip_step=not window_step_allowed)
                    accum_count = 0
                    window_has_finite_loss = False
                    completed_batch_cursor = bi + 1
                    if stepped:
                        _commit_train_step_window()
                        if step_callback is not None:
                            step_callback(epoch, self.n_steps, completed_batch_cursor)
                        self._log_train_loss_after_step()
                    else:
                        _discard_train_step_window()
                        if self.rank == 0:
                            # 仅在 accumulation 边界为 pbar 显示做一次 .item() 同步；
                            # 非边界 micro-batch 不再触发 host 端 CUDA 同步
                            loss_scalar = loss_detached.item()
                            pbar.set_postfix(loss=f"{loss_scalar:.4f}", step=self.n_steps)
                    max_steps_reached = self._sync_max_steps_reached()

            _finish_batch_timing(
                timing_start_event,
                compute_start,
                batch_size_for_timing,
                timing_sampled,
            )
            if stopped_by_shutdown:
                break
            if (
                self.save_batch_interval > 0
                and (bi + 1) % self.save_batch_interval == 0
            ):
                self._save_middle_checkpoint(epoch, bi)
            if max_steps_reached:
                stopped_by_max_steps = True
                if self.rank == 0:
                    logger.info(f"  Reached max_steps={self.max_steps}, stopping epoch.")
                break

        # 部分尾窗或零 batch epoch 尚未执行停机同步；所有 rank 在该静态一致的
        # epoch-tail boundary 补一次，再决定丢弃尾窗或走统一 tail helper。
        if not stopped_by_max_steps and not stopped_by_shutdown and not shutdown_synced_at_last_batch:
            stopped_by_shutdown = bool(self._sync_shutdown_signum())
        tail_has_gradients = (
            not (stopped_by_max_steps or stopped_by_shutdown)
            and accum_count > 0
            and loss is not None
        )
        tail_step_allowed = self._sync_train_window_finite_eligibility(
            tail_has_gradients and window_has_finite_loss,
            phase="train_tail_finite_eligibility",
        ) and not strict_failure_seen
        if tail_has_gradients:
            # helper 会在 step_fn 返回后写 callback/log；故在 step_fn 内先提交
            # window，保证尾窗日志与普通 accumulation boundary 的可见数据一致。
            tail_window_committed = False

            def _step_tail_window() -> bool:
                nonlocal tail_window_committed
                stepped = self._optimizer_step()
                if stepped:
                    _commit_train_step_window()
                    tail_window_committed = True
                return stepped

            tail_stepped = self._handle_epoch_tail_gradients(
                accum_count, accum, epoch,
                step_allowed=tail_step_allowed,
                step_callback=step_callback,
                batch_cursor=bi + 1,
                step_fn=_step_tail_window,
                label=self._get_epoch_tail_label(),
            )
            if not tail_stepped or not tail_window_committed:
                _discard_train_step_window()
        elif (stopped_by_max_steps or stopped_by_shutdown) and accum_count > 0:
            # max_steps/停机信号触发后清零所有优化器；未提交的累积窗口由
            # checkpoint cursor 指向上一安全边界，恢复时完整重做。
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)
            _discard_train_step_window()

        loader_batches = getattr(pbar, "total", None)
        self._last_completed_batch_cursor = int(completed_batch_cursor)
        self._train_epoch_completed = not (stopped_by_max_steps or stopped_by_shutdown) or (
            loader_batches is not None
            and completed_batch_cursor >= int(loader_batches)
        )

        total_loss, total_batches, nan_g, comp_avgs = self._allreduce_epoch_stats(
            running_loss_gpu, n_batches, nan_loss_count,
            epoch_comp_weighted, epoch_comp_count,
        )
        if cuda_timing:
            # epoch boundary 统一同步：既读取 GPU 诊断事件，也让 wall-clock 覆盖已排队工作。
            torch.cuda.synchronize()
            sampled_compute_sec = sum(
                start.elapsed_time(end) for start, end in cuda_event_pairs
            ) / 1000.0
            compute_sec = (
                sampled_compute_sec * timing_batches / timing_sampled_batches
                if timing_sampled_batches > 0
                else 0.0
            )
        else:
            compute_sec = compute_wall_sec
        # sanitize 计数与 timing 均在 epoch boundary 固定顺序汇总；summary 只读结果。
        self._allreduce_output_sanitize_stats()
        epoch_wall_sec = max(0.0, time.perf_counter() - epoch_wall_start)
        (
            data_wait_log,
            compute_log,
            timing_samples_global,
            elapsed_log,
        ) = self._allreduce_timing_stats(
            data_wait_sec, compute_sec, timing_samples, epoch_wall_sec
        )
        timing_task_rows_global = self._allreduce_sum(
            timing_task_rows, phase="train_timing_task_rows"
        )
        samples_per_sec = (
            timing_samples_global / elapsed_log if elapsed_log > 0.0 else 0.0
        )
        task_rows_per_sec = (
            timing_task_rows_global / elapsed_log if elapsed_log > 0.0 else 0.0
        )
        if self.rank == 0:
            self._log_scalar("system/data_wait_sec", data_wait_log, epoch)
            self._log_scalar("system/compute_sec", compute_log, epoch)
            self._log_scalar("system/epoch_wall_sec", elapsed_log, epoch)
            self._log_scalar("system/samples_per_sec", samples_per_sec, epoch)
            self._log_task_throughput(samples_per_sec, task_rows_per_sec, epoch)
            logger.info(
                f"  [E{epoch}] throughput: host_loader_prepare_gap={data_wait_log:.2f}s "
                f"compute={compute_log:.2f}s epoch_wall={elapsed_log:.2f}s "
                f"samples/s={samples_per_sec:.1f}"
            )
        if strict_failure_seen:
            # 先完成 epoch 统计 collective，再由所有 rank 一致抛错；不能让
            # 某个 rank 在 batch 内提前退出而把其他 rank 留在 collective 中。
            raise RuntimeError(
                strict_failure_message
                or f"[E{epoch}] non-finite loss under strict_checks."
            )
        if total_batches > 0:
            epoch_avg = total_loss / total_batches
        elif nan_g > 0:
            # 全 NaN epoch 时无 finite batch，按约定返回 0.0（见 tests/test_nan_loss.py）
            epoch_avg = 0.0
        else:
            epoch_avg = None

        if self.rank == 0 and epoch_avg is not None:
            self._log_scalar("train/epoch_loss", epoch_avg, epoch)
            if comp_avgs:
                inner_c = self._inner_loss_fn()
                for name, avg_val in comp_avgs:
                    self._log_scalar(f"train/loss_comp/{name}", avg_val, epoch)
                comp_str = " | ".join(
                    f"{_composite_loss_comp_log_label(inner_c, i, name)}={v:.4f}"
                    for i, (name, v) in enumerate(comp_avgs)
                )
                logger.info(f"  [E{epoch}] train loss {epoch_avg:.4f} | {comp_str}")
            if nan_g > 0:
                scope = "all ranks summed" if self.world_size > 1 else "this run"
                logger.warning(
                    f"[E{epoch}] NaN/Inf loss micro-batches ({scope}): {nan_g} / {int(total_batches)}"
                )
            if silence_placeholder_samples > 0:
                ratio = silence_placeholder_samples / max(1, total_samples_this_epoch)
                self._log_scalar("train/silence_placeholder_ratio", ratio, epoch)
                logger.warning(
                    f"[E{epoch}] silence placeholder samples: {silence_placeholder_samples}/"
                    f"{total_samples_this_epoch} ({ratio:.2%}). Check data paths."
                )
        # A23：输出 sanitize/clamp 触发时按 epoch 汇总上报（每 epoch 最多一条）
        self._log_output_sanitize_summary(epoch, "[Train]")
        self._last_train_comp = {n: float(v) for n, v in comp_avgs} if comp_avgs else {}
        return epoch_avg

    def _allreduce_timing_stats(
        self,
        data_wait_sec: float,
        compute_sec: float,
        timing_samples: int,
        epoch_wall_sec: float,
    ) -> tuple[float, float, float, float]:
        """聚合 timing：诊断时长与端到端 wall-clock 均取 MAX，样本数取 SUM。

        ``data_wait_sec`` 是上一 batch 结束至本 batch prepare 完成的 host gap，
        覆盖 DataLoader yield、unpack、H2D 与 task prepare。``compute_sec`` 是
        sampled GPU event 或 CPU forward/step 的诊断估计；二者不构成吞吐分母。
        ``epoch_wall_sec`` 才是端到端 epoch elapsed，覆盖完整 epoch 与同步。
        """
        local_elapsed_sec = float(epoch_wall_sec)
        if self.world_size > 1:
            elapsed_t = torch.tensor(
                [float(data_wait_sec), float(compute_sec), local_elapsed_sec],
                dtype=torch.float64,
                device=self.device,
            )
            samples_t = torch.tensor(
                [float(timing_samples)], dtype=torch.float64, device=self.device
            )
            dist.all_reduce(elapsed_t, op=dist.ReduceOp.MAX)
            dist.all_reduce(samples_t, op=dist.ReduceOp.SUM)
            return (
                float(elapsed_t[0].item()),
                float(elapsed_t[1].item()),
                float(samples_t.item()),
                float(elapsed_t[2].item()),
            )
        return (
            float(data_wait_sec),
            float(compute_sec),
            float(timing_samples),
            local_elapsed_sec,
        )

    def _allreduce_epoch_stats(
        self,
        running_loss_gpu: torch.Tensor,
        n_batches: int,
        nan_loss_count: int,
        epoch_comp_weighted: dict[str, torch.Tensor],
        epoch_comp_count: dict[str, int],
    ) -> tuple[float, float, int, list[tuple[str, float]] | None]:
        """Allreduce per-rank epoch stats across DDP ranks; return global aggregates.

        epoch_comp_weighted 为 GPU tensor 累计值（训练热路径不做 host 同步），
        本函数在 epoch 边界统一物化。

        Returns: (total_loss, total_batches, nan_global, comp_avgs_or_None)
        """
        if self.world_size > 1:
            n_batches_t = torch.tensor(float(n_batches), device=self.device)
            dist.all_reduce(running_loss_gpu, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_batches_t, op=dist.ReduceOp.SUM)
            total_loss = running_loss_gpu.item()
            total_batches = n_batches_t.item()
            nan_t = torch.tensor(
                [float(nan_loss_count)], device=self.device, dtype=torch.float64
            )
            dist.all_reduce(nan_t, op=dist.ReduceOp.SUM)
            nan_g = int(nan_t.item())
        else:
            total_loss = running_loss_gpu.item()
            total_batches = float(n_batches)
            nan_g = nan_loss_count

        comp_avgs: list[tuple[str, float]] | None = None
        _comp_names = self._comp_keys
        if self.world_size > 1 and _comp_names is not None:
            # epoch 结束时才物化为 float（训练热路径全程 GPU tensor 累计）
            t = torch.tensor(
                [float(epoch_comp_weighted[n].item()) if n in epoch_comp_weighted else 0.0
                 for n in _comp_names],
                device=self.device, dtype=torch.float64,
            )
            c = torch.tensor(
                [float(epoch_comp_count.get(n, 0)) for n in _comp_names],
                device=self.device, dtype=torch.float64,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            dist.all_reduce(c, op=dist.ReduceOp.SUM)
            comp_avgs = [
                (n, (t[i].item() / c[i].item()) if c[i].item() > 0 else 0.0)
                for i, n in enumerate(_comp_names)
            ]
        elif self.rank == 0 and epoch_comp_weighted and epoch_comp_count:
            _cn = _comp_names or sorted(epoch_comp_weighted.keys())
            comp_avgs = [
                (n, float(epoch_comp_weighted[n].item()) / max(epoch_comp_count[n], 1))
                for n in _cn if n in epoch_comp_weighted
            ]

        return total_loss, total_batches, nan_g, comp_avgs

    def _loss_accepts_predicted_lsnr(self) -> bool:
        if not hasattr(self.loss_fn, "forward"):
            return False
        sig = inspect.signature(self.loss_fn.forward)
        params = sig.parameters
        return "predicted_lsnr" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    def _compute_loss_needs_predicted_alpha(self) -> bool:
        inner = self._inner_loss_fn()
        if isinstance(inner, CompositeLoss):
            for i, fn in enumerate(inner.losses):
                if type(fn).__name__ == "DfAlphaLossComponent" and float(inner.weights[i].item()) != 0.0:
                    return True
        if type(inner).__name__ == "DeepFilterNetFullLossWithLSNR" and getattr(
            inner, "use_alpha_loss", False
        ):
            return True
        return False

    def _inner_loss_fn(self) -> Any:
        """内层 loss 模块（解包 strict 包装器，得到 CompositeLoss 等）。"""
        return getattr(self.loss_fn, "loss", self.loss_fn)

    def _batch_fs_hz(self, fs, batch_size: int) -> int:
        """标量 fs 或 shape [B] 且各条相同采样率；混采样率 batch 会报错。"""
        if fs is None:
            return int(self.sample_rate)
        if isinstance(fs, (int, float)):
            return int(fs)
        if isinstance(fs, torch.Tensor):
            fs_flat = fs.reshape(-1)
            if fs_flat.numel() == 1:
                return int(fs_flat[0].item())
            if fs_flat.numel() == batch_size:
                if not bool(torch.all(fs_flat == fs_flat[0]).item()):
                    raise ValueError(
                        "Mixed sample rates in one batch are not supported: fs must be "
                        "scalar or 1D [B] with identical values per batch. "
                        "Use one sample rate per batch or a collate that splits by sr."
                    )
                return int(fs_flat[0].item())
            raise ValueError(
                f"fs must be scalar or [B]={batch_size}, got shape {tuple(fs.shape)}"
            )
        return int(fs)

    def _advance_and_should_run_metrics(self) -> bool:
        """递增 validation 指标轮次计数，并返回本轮是否计算 objective 指标。"""
        # _vm_epoch_count 随 checkpoint 恢复以保持 interval 相位；若本轮 validation
        # 在 forward 或任务额外指标阶段中止，调用方会回滚本次递增。
        self._vm_epoch_count += 1
        return (
            self._vm_enabled
            and bool(self._vm_metrics)
            and (self._vm_epoch_count % self._vm_interval == 0)
        )

    def _run_validation_metrics_from_pairs(
        self,
        epoch: int,
        pairs: list[tuple[np.ndarray, np.ndarray, int]],
        do_metrics: bool,
        val_x: Optional[int] = None,
        *,
        validation_aborted: bool = False,
        commit_history: Optional[bool] = None,
        task_metric_results: Optional[
            dict[str, tuple[float, float, float]]
        ] = None,
        val_loss_components: Optional[dict[str, float]] = None,
    ) -> bool:
        """分块归约并记录带参考的 validation objective 指标。

        DDP 一致性：do_metrics=True 时，无论本地 pairs 是否为空，所有 rank 必须执行
        相同的 chunk、shutdown、总样本数和固定指标 key collective 序列。
        """
        if val_x is None:
            val_x = epoch
        local_finite_loss_samples = self.__dict__.pop(
            "_validation_loop_finite_loss_samples", None
        )
        if not do_metrics:
            return bool(validation_aborted)
        if commit_history is None:
            if local_finite_loss_samples is None:
                commit_history = True
            else:
                finite_loss_samples_global = self._allreduce_sum(
                    local_finite_loss_samples,
                    phase="val_metrics_finite_loss_samples",
                )
                commit_history = finite_loss_samples_global > 0
        aggregated_intrusive: dict = {}
        effective_pairs = [] if validation_aborted else pairs
        total_pairs_for_metrics = len(effective_pairs)
        cs = self._vm_chunk_size
        chunk_list = [
            effective_pairs[i : i + cs]
            for i in range(0, len(effective_pairs), cs)
        ]
        n_local = len(chunk_list)
        if self.world_size > 1:
            t_nc = torch.tensor([n_local], device=self.device, dtype=torch.long)
            dist.all_reduce(t_nc, op=dist.ReduceOp.MAX)
            n_chunk_max = int(t_nc.item())
        else:
            n_chunk_max = n_local
        metrics_aborted = bool(validation_aborted)
        for i in range(n_chunk_max):
            chunk_batch = chunk_list[i] if i < n_local else []
            self._merge_metrics_chunk(chunk_batch, aggregated_intrusive)
            if self._sync_shutdown_signum():
                metrics_aborted = True
                break

        # 即使没有 chunk，也固定同步 shutdown，保证 strict 分支前各 rank 一致。
        metrics_aborted = bool(self._sync_shutdown_signum()) or metrics_aborted
        if metrics_aborted:
            aggregated_intrusive.clear()
        t_metrics_start = time.monotonic()
        if self.world_size > 1:
            self._nccl_flow_log("val_metrics_total_utts_all_reduce")
            _n = torch.tensor(float(total_pairs_for_metrics), device=self.device)
            dist.all_reduce(_n, op=dist.ReduceOp.SUM)
            total_utts_global = int(_n.item())
        else:
            total_utts_global = total_pairs_for_metrics
        self._log_val_metrics(
            None,
            epoch,
            aggregated_intrusive=aggregated_intrusive,
            val_x=val_x,
            metrics_aborted=metrics_aborted,
            eligible_count=total_utts_global,
            commit_history=commit_history,
            task_metric_results=task_metric_results,
            val_loss_components=val_loss_components,
        )
        if self.rank == 0:
            t_metrics_sec = time.monotonic() - t_metrics_start
            if metrics_aborted:
                logger.info("  [Val] metrics aborted by shutdown request")
            else:
                self._log_scalar("system/val_metrics_time_sec", t_metrics_sec, val_x)
                logger.info(
                    f"  [Val] metrics computed in {t_metrics_sec:.1f}s "
                    f"({total_utts_global} utts)"
                )
        return metrics_aborted

    def _shared_validation_metrics(self) -> list[str]:
        """返回共享 ``compute_metrics_batch`` 可计算的指标列表。

        ERLE 需要 AEC 的 mic/nearend/farend 语义，由 ``AECTrainer`` 专用路径计算；
        即使配置列表包含 ERLE，也不得把它送入仅接收 (ref, enh, fs) 的共享路径。
        """
        return [
            metric
            for metric in getattr(self, "_vm_metrics", ("SI-SDR", "PESQ", "ESTOI"))
            if metric != "ERLE"
        ]

    def _merge_metrics_chunk(
        self,
        chunk: list[tuple[np.ndarray, np.ndarray, int]],
        aggregated_intrusive: dict,
    ) -> None:
        """计算一个侵入式指标 chunk，并原地合并 sum/count。"""
        from core.metrics import compute_metrics_batch

        raw_int = compute_metrics_batch(
            chunk,
            metrics=self._shared_validation_metrics(),
            strict=False,
            parallel_intrusive=self._vm_parallel_intrusive,
            max_workers_intrusive=self._vm_metrics_workers,
            device=str(self.device),
            should_abort=self.shutdown_requested,
        )
        for k, v in raw_int.items():
            aggregated_intrusive[k] = aggregated_intrusive.get(k, 0.0) + v
        # n_chunk_max is already synchronized via all_reduce(MAX), no barrier needed here

    def _log_val_metrics(
        self,
        pairs: Optional[list[tuple[np.ndarray, np.ndarray, int]]],
        epoch: int,
        *,
        aggregated_intrusive: Optional[dict] = None,
        val_x: Optional[int] = None,
        metrics_aborted: bool = False,
        eligible_count: Optional[int] = None,
        commit_history: bool = True,
        task_metric_results: Optional[
            dict[str, tuple[float, float, float]]
        ] = None,
        val_loss_components: Optional[dict[str, float]] = None,
    ) -> None:
        """计算或归约 objective 指标，并记录 TensorBoard 与历史曲线。"""
        from core.metrics import compute_metrics_batch

        log_x = val_x if val_x is not None else epoch

        if aggregated_intrusive is not None:
            raw_intrusive = aggregated_intrusive
        else:
            raw_intrusive = compute_metrics_batch(
                pairs or [],
                metrics=self._shared_validation_metrics(),
                strict=False,
                parallel_intrusive=self._vm_parallel_intrusive,
                max_workers_intrusive=self._vm_metrics_workers,
                device=str(self.device),
                should_abort=self.shutdown_requested,
            )
        # 固定使用共享指标配置顺序，保证单卡/多卡逐项执行同一 collective；
        # AEC 专用 ERLE 已由 task hook 在此前以固定 shape 单独归约。
        all_intrusive = self._shared_validation_metrics()

        if eligible_count is None:
            eligible_count = int(
                self._allreduce_sum(
                    len(pairs or []), phase="val_metrics_eligible_count"
                )
            )
            # 兼容直接传入预聚合结果的 helper 调用；真实 validation 总是显式传入
            # metric pair 总数。这里以 observed valid count 的最大值恢复最小 eligible。
            if aggregated_intrusive is not None and eligible_count == 0:
                eligible_count = int(
                    max(
                        (
                            raw_intrusive.get(f"_count_{metric}", 0.0)
                            for metric in all_intrusive
                        ),
                        default=0.0,
                    )
                )
        eligible_count = max(0, int(eligible_count))

        metric_results: dict[str, tuple[float, float, float, float]] = {}
        self._nccl_flow_log("val_intrusive_allreduce_loop")
        for m in all_intrusive:
            sum_v = self._allreduce_sum(
                raw_intrusive.get(m, 0.0), phase=f"val_intrusive_{m}"
            )
            count_v = self._allreduce_sum(
                raw_intrusive.get(f"_count_{m}", 0.0),
                phase=f"val_intrusive_count_{m}",
            )
            mean_v = (sum_v / count_v) if count_v > 0 else float("nan")
            coverage = count_v / eligible_count if eligible_count > 0 else 0.0
            metric_results[m] = (mean_v, count_v, float(eligible_count), coverage)
        for metric, (mean_v, count_v, task_eligible) in (
            task_metric_results or {}
        ).items():
            if metric not in self._vm_metrics:
                continue
            coverage = count_v / task_eligible if task_eligible > 0 else 0.0
            metric_results[metric] = (mean_v, count_v, task_eligible, coverage)
        metric_order = [metric for metric in self._vm_metrics if metric in metric_results]

        # 每个 rank 已执行相同的逐指标 allreduce，因此 runtime gate 结论全局一致；
        # strict/min_coverage 必须在 TensorBoard/history/curve 等状态写入前 fail-closed。
        if not metrics_aborted:
            for metric in metric_order:
                mean_v, count_v, metric_eligible, _coverage = metric_results[metric]
                if count_v > 0 and not math.isfinite(mean_v):
                    raise RuntimeError(
                        f"{metric} validation returned non-finite mean: {mean_v!r}"
                    )
                validate_validation_metric_runtime_result(
                    metric,
                    count=count_v,
                    eligible=metric_eligible,
                    strict=self._vm_strict,
                    min_coverage=getattr(self, "_vm_min_coverage", {}).get(metric),
                )

        log_parts: list[str] = []
        this_epoch_values: dict[str, float] = {}
        if self.rank == 0:
            for m in metric_order:
                mean_v, count_v, metric_eligible, coverage = metric_results[m]
                this_epoch_values[m] = mean_v
                if not metrics_aborted:
                    self._log_scalar(
                        f"val/{m}/eligible_count", metric_eligible, log_x
                    )
                    self._log_scalar(f"val/{m}/valid_count", count_v, log_x)
                    self._log_scalar(f"val/{m}/coverage_ratio", coverage, log_x)
                if (
                    commit_history
                    and not metrics_aborted
                    and count_v > 0
                    and math.isfinite(mean_v)
                ):
                    self._log_scalar(f"val/{m}", mean_v, log_x)
                    log_parts.append(f"{m}={mean_v:.4f}")

            if commit_history and not metrics_aborted and log_parts:
                self.val_metrics_epochs.append(log_x)
                coverage_history = getattr(
                    self, "val_metrics_coverage_history", None
                )
                if coverage_history is None:
                    coverage_history = {}
                    self.val_metrics_coverage_history = coverage_history
                for m in metric_order:
                    self.val_metrics_history.setdefault(m, []).append(
                        this_epoch_values.get(m, float("nan"))
                    )
                    _mean, valid_count, metric_eligible, coverage = metric_results[m]
                    coverage_history.setdefault(m, []).append(
                        {
                            "eligible_count": float(metric_eligible),
                            "valid_count": float(valid_count),
                            "coverage_ratio": float(coverage),
                        }
                    )
                _cnames = self._comp_keys
                if _cnames:
                    new_len = len(self.val_metrics_epochs)
                    for nm in sorted(_cnames):
                        components = (
                            self._last_val_comp
                            if val_loss_components is None
                            else val_loss_components
                        )
                        v = components.get(nm, float("nan"))
                        lst = self.val_metrics_loss_comp_hist.setdefault(nm, [])
                        if len(lst) >= new_len:
                            continue
                        while len(lst) < new_len - 1:
                            lst.append(float("nan"))
                        lst.append(v)
                logger.info(f"  [Val] metrics | " + " | ".join(log_parts))
                if self._should_save_curve(len(self.val_metrics_epochs)):
                    self._save_val_metrics_curves()
            elif not commit_history and not metrics_aborted:
                logger.warning(
                    "  [Val] metrics computed but history was not committed because "
                    "the validation epoch had no finite loss samples"
                )
            elif self._vm_enabled and not metrics_aborted:
                logger.warning(
                    "  [Val] metrics: no valid results (check if pesq/pystoi are installed)"
                )

    # Metric names grouped by scale for same-plot drawing
    _VM_GROUP_0_1 = {"STOI", "ESTOI"}  # range 0–1
    _VM_GROUP_PESQ = {"PESQ"}  # ~1–4.5
    _VM_GROUP_DB = {"SI-SDR", "SI-SNR", "SDR", "SAR", "SIR", "ERLE", "ERLU"}  # dB（ERLU=AEC 口误别名，与 ERLE 同尺度）
    # 仅用于读取旧 checkpoint history 并按原尺度绘图；Trainer 不再新增 MOS 指标。
    _VM_GROUP_MOS_PREFIXES = ("DNSMOS_", "NISQA_", "UTMOS_", "SCOREQ_")

    def _val_metric_group(self, name: str) -> str:
        if name in self._VM_GROUP_0_1:
            return "0-1"
        if name in self._VM_GROUP_PESQ:
            return "pesq"
        if name in self._VM_GROUP_DB or "SDR" in name or "SNR" in name:
            return "dB"
        if any(name.startswith(p) for p in self._VM_GROUP_MOS_PREFIXES):
            return "mos"
        return "other"

    def _on_loss_curve_point_added(
        self, x_val: int, train_loss: Optional[float], val_loss: float
    ) -> None:
        if self.rank != 0:
            return
        names = self._comp_keys
        if not names:
            return
        if self._curve_comp_names is None:
            self._curve_comp_names = list(names)
        tr_nan = (
            train_loss is not None
            and isinstance(train_loss, float)
            and math.isnan(train_loss)
        )
        last_t = self._last_train_comp
        last_v = self._last_val_comp
        # 与 train_losses 行对齐：resume 后首次追加前用 nan 补齐前面 epoch
        target_len = len(self.train_losses)
        for name in self._curve_comp_names:
            tv = float("nan") if tr_nan else float(last_t.get(name, float("nan")))
            vv = float(last_v.get(name, float("nan")))
            lt = self._train_loss_comp_hist.setdefault(name, [])
            lv = self._val_loss_comp_hist.setdefault(name, [])
            while len(lt) < len(lv):
                lt.append(float("nan"))
            while len(lv) < len(lt):
                lv.append(float("nan"))
            if len(lt) >= target_len:
                continue
            while len(lt) < target_len - 1:
                lt.append(float("nan"))
                lv.append(float("nan"))
            lt.append(tv)
            lv.append(vv)

    def _after_resume_checkpoint(self, ckpt: dict) -> None:
        self._vm_epoch_count = int(ckpt.get("val_metrics_epoch_count", 0))
        self.val_metrics_epochs = list(ckpt.get("val_metrics_epochs") or [])
        self.val_metrics_history = dict(ckpt.get("val_metrics_history") or {})
        self.val_metrics_coverage_history = dict(
            ckpt.get("val_metrics_coverage_history") or {}
        )
        self._train_loss_comp_hist = dict(ckpt.get("train_loss_comp_hist") or {})
        self._val_loss_comp_hist = dict(ckpt.get("val_loss_comp_hist") or {})
        self._curve_comp_names = ckpt.get("curve_comp_names")
        self.val_metrics_loss_comp_hist = dict(
            ckpt.get("val_metrics_loss_comp_hist") or {}
        )

    def _build_state(self, epoch: int, **kw: Any) -> dict:
        state = super()._build_state(epoch, **kw)
        state["val_metrics_epoch_count"] = int(
            getattr(self, "_vm_epoch_count", 0)
        )
        state["val_metrics_epochs"] = getattr(self, "val_metrics_epochs", [])
        state["val_metrics_history"] = getattr(self, "val_metrics_history", {})
        state["val_metrics_coverage_history"] = getattr(
            self, "val_metrics_coverage_history", {}
        )
        state["train_loss_comp_hist"] = getattr(self, "_train_loss_comp_hist", {})
        state["val_loss_comp_hist"] = getattr(self, "_val_loss_comp_hist", {})
        state["curve_comp_names"] = getattr(self, "_curve_comp_names", None)
        state["val_metrics_loss_comp_hist"] = getattr(self, "val_metrics_loss_comp_hist", {})
        return state

    def _save_loss_curve(self) -> None:
        if self.rank != 0 or not self.train_losses or not self.val_losses:
            return
        if self._comp_keys and self._train_loss_comp_hist:
            self._save_loss_curve_with_components()
        else:
            super()._save_loss_curve()

    def _save_loss_curve_with_components(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.cm as mpl_cm
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping loss curve save")
            return
        n = min(len(self.train_losses), len(self.val_losses), len(self.curve_epochs))
        if n == 0:
            return
        x = self.curve_epochs[:n]
        x_is_step = getattr(self, "_curve_x_is_step", False)
        x_name = "Step" if x_is_step else "Epoch"
        model_type = self.config.get("model", {}).get("type", "model")
        loss_type = self.config.get("loss", {}).get("type", "loss")
        title = f"Train & Validation Loss — {model_type} | {loss_type}"
        names = self._curve_comp_names or sorted(self._train_loss_comp_hist.keys())
        if not names:
            super()._save_loss_curve()
            return

        def _pad(arr: list[float], length: int) -> np.ndarray:
            if len(arr) < length:
                arr = list(arr) + [float("nan")] * (length - len(arr))
            return np.asarray(arr[:length], dtype=np.float64)

        w, h = self.curve_figsize
        fig, (ax0, ax1) = plt.subplots(
            2,
            1,
            figsize=(w, h * 1.75),
            height_ratios=[1, 1.15],
            sharex=True,
        )
        tl = np.asarray(self.train_losses[:n], dtype=np.float64)
        train_mask = np.isfinite(tl)
        if train_mask.any():
            ax0.plot(
                np.asarray(x)[train_mask],
                tl[train_mask],
                "b-",
                label="Train Loss",
                linewidth=0.8,
            )
        elif x_is_step:
            ax0.plot(
                [],
                [],
                "b-",
                label="Train Loss (N/A until epoch end in step-val mode)",
                linewidth=0.8,
            )
        ax0.plot(x, self.val_losses[:n], "c-", label="Val loss", linewidth=0.8)
        best_x = None
        best_lbl = None
        if x_is_step and getattr(self, "best_at_step", None) is not None:
            bx = self.best_at_step
            if bx is not None and bx in x:
                best_x, best_lbl = bx, f"Best (step {bx})"
        elif not x_is_step and self.best_epoch >= 0 and self.best_epoch in x:
            best_x, best_lbl = self.best_epoch, f"Best (E{self.best_epoch})"
        if best_x is not None:
            ax0.axvline(
                x=best_x,
                color="green",
                linestyle="--",
                alpha=0.7,
                linewidth=0.8,
                label=best_lbl,
            )
        ax0.set_title(title, fontsize=11)
        ax0.set_ylabel("Loss")
        ax0.legend(loc="best", fontsize=8)
        ax0.grid(True, alpha=0.3)
        self._apply_curve_xticks(ax0, x)

        cmap = mpl_cm.tab10(np.linspace(0, 1, max(len(names), 1)))
        for i, name in enumerate(names):
            c = cmap[i % 10]
            lbl = LOSS_DISPLAY_NAMES.get(name, name)
            tr = _pad(self._train_loss_comp_hist.get(name, []), n)
            va = _pad(self._val_loss_comp_hist.get(name, []), n)
            ax1.plot(
                x,
                tr,
                "--",
                color=c,
                linewidth=0.75,
                label=f"{lbl} (train)",
            )
            ax1.plot(x, va, "-", color=c, linewidth=0.75, label=f"{lbl} (val)")
        ax1.set_xlabel(x_name)
        ax1.set_ylabel("Component (weighted)")
        ax1.set_title("Composite loss components (weight × raw)", fontsize=10)
        ax1.legend(loc="best", fontsize=6, ncol=2)
        ax1.grid(True, alpha=0.3)
        self._apply_curve_xticks(ax1, x)

        exp_path = self.log_dir.parent.parent
        fig.text(
            0.5,
            0.01,
            f"{exp_path}  |  Top: total train/val  |  Bottom: weighted sub-losses (dashed=train, solid=val)",
            ha="center",
            fontsize=7,
            color="gray",
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        path = self.log_dir / "train_val_loss.png"
        if self._save_curve_fig(fig, path):
            logger.info(f"Loss curve saved: {path}")

        self._write_train_val_curve_csv(n, x, x_is_step)

    def _write_train_val_curve_csv(self, n: int, x: list, x_is_step: bool) -> None:
        """Same as BaseTrainer, plus train_loss_comp_* / val_loss_comp_* (weighted w×raw, same scale as total loss)."""
        names = self._curve_comp_names or sorted(self._train_loss_comp_hist.keys())
        if not names:
            super()._write_train_val_curve_csv(n, x, x_is_step)
            return
        csv_path = self.log_dir / "train_val_curve.csv"
        xcol = "step" if x_is_step else "epoch"
        try:
            comp_t = [f"train_loss_comp_{nm}" for nm in names]
            comp_v = [f"val_loss_comp_{nm}" for nm in names]
            headers = [xcol, "train_loss", "val_loss"] + comp_t + comp_v
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                for i in range(n):
                    tr = self.train_losses[i]
                    if isinstance(tr, (int, float)) and tr == tr:
                        tr_s = f"{float(tr):.6f}"
                    else:
                        tr_s = "nan"
                    row: list[str] = [str(x[i]), tr_s, f"{self.val_losses[i]:.6f}"]
                    for nm in names:
                        trc = self._train_loss_comp_hist.get(nm, [])
                        v = trc[i] if i < len(trc) else float("nan")
                        row.append(
                            f"{float(v):.6f}"
                            if isinstance(v, float) and v == v
                            else "nan"
                        )
                    for nm in names:
                        vac = self._val_loss_comp_hist.get(nm, [])
                        v = vac[i] if i < len(vac) else float("nan")
                        row.append(
                            f"{float(v):.6f}"
                            if isinstance(v, float) and v == v
                            else "nan"
                        )
                    f.write(",".join(row) + "\n")
        except OSError as e:
            logger.warning("Could not write loss curve CSV: %s", e)

    def _save_val_metrics_curves(self) -> None:
        """Plot validation metrics by group (same scale on one figure), save to log_dir."""
        if not self.val_metrics_epochs or not self.val_metrics_history:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        groups: dict[str, list[str]] = {}
        for m in self.val_metrics_history:
            ys = self.val_metrics_history[m]
            if not ys:
                continue
            if all(isinstance(y, float) and y != y for y in ys):
                continue  # all NaN, skip
            g = self._val_metric_group(m)
            groups.setdefault(g, []).append(m)
        if not groups:
            return
        model_type = self.config.get("model", {}).get("type", "model")
        loss_type = self.config.get("loss", {}).get("type", "loss")
        base_title = f"Val metrics — {model_type} | {loss_type}"
        x_all = self.val_metrics_epochs
        x_name = "Step" if getattr(self, "_curve_x_is_step", False) else "Epoch"
        for group_name, metrics in groups.items():
            fig, ax = plt.subplots(figsize=self.curve_figsize)
            for m in sorted(metrics):
                ys = self.val_metrics_history[m]
                n_plot = min(len(x_all), len(ys))
                if n_plot == 0:
                    continue
                xs = x_all[:n_plot]
                ys = ys[:n_plot]
                ax.plot(xs, ys, "-o", markersize=2, linewidth=0.8, label=m)
            ax.set_title(f"{base_title} ({group_name})", fontsize=11)
            ax.set_xlabel(x_name)
            ax.set_ylabel("Score")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
            self._apply_curve_xticks(ax, x_all)
            fig.tight_layout()
            path = self.log_dir / f"val_metrics_{group_name}.png"
            if self._save_curve_fig(fig, path):
                logger.info(f"Val metrics curve saved: {path}")
        self._write_val_metrics_csv()

    def _write_val_metrics_csv(self) -> None:
        """写 metric 值、覆盖统计与可选 loss component 的 validation CSV。"""
        if not self.val_metrics_epochs or not self.val_metrics_history:
            return

        def _csv_quote(s: str) -> str:
            if "," in s or '"' in s or "\n" in s:
                return '"' + s.replace('"', '""') + '"'
            return s

        loss_comp_names = sorted(self.val_metrics_loss_comp_hist.keys())
        coverage_history = getattr(self, "val_metrics_coverage_history", {})
        csv_path = self.log_dir / "val_metrics.csv"
        try:
            metrics = sorted(self.val_metrics_history.keys())
            n_rows = len(self.val_metrics_epochs)
            xcol = "step" if getattr(self, "_curve_x_is_step", False) else "epoch"
            coverage_columns = [
                f"{metric}_{field}"
                for metric in metrics
                for field in ("eligible_count", "valid_count", "coverage_ratio")
            ]
            extra = [f"val_loss_comp_{nm}" for nm in loss_comp_names]
            columns = metrics + coverage_columns + extra
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(xcol + "," + ",".join(_csv_quote(c) for c in columns) + "\n")
                for i in range(n_rows):
                    row = [str(self.val_metrics_epochs[i])]
                    for m in metrics:
                        vals = self.val_metrics_history[m]
                        v = vals[i] if i < len(vals) else None
                        # NaN (e.g. metric failed this epoch) -> empty cell
                        if v is None or (isinstance(v, float) and v != v):
                            row.append("")
                        else:
                            row.append(f"{v:.6f}")
                    for m in metrics:
                        history = coverage_history.get(m, [])
                        coverage = history[i] if i < len(history) else {}
                        for field in (
                            "eligible_count",
                            "valid_count",
                            "coverage_ratio",
                        ):
                            value = coverage.get(field)
                            row.append("" if value is None else f"{float(value):.6f}")
                    for nm in loss_comp_names:
                        lst = self.val_metrics_loss_comp_hist.get(nm, [])
                        v = lst[i] if i < len(lst) else float("nan")
                        row.append(
                            f"{float(v):.6f}"
                            if isinstance(v, float) and v == v
                            else "nan"
                        )
                    f.write(",".join(row) + "\n")
        except OSError as e:
            logger.warning("Could not write val_metrics CSV: %s", e)
