"""Acoustic echo cancellation trainer AECTrainer.

Implements AEC training loop on top of MetricsCapableTrainer:
  - AMP autocast (FP16/BF16) + gradient scaling
  - non_blocking GPU transfer (async with pin_memory)
  - Gradient accumulation (accumulation_steps for larger effective batch)
    Uses model.no_sync() during intermediate accumulation steps to suppress
    redundant DDP AllReduce — only the final step triggers synchronization.
  - NaN safety (merged isfinite checks — 1 GPU sync per check, not 2)
  - Auto-detect if model needs fs parameter
  - Validation objective metrics: SI-SDR / PESQ / STOI / ESTOI + ERLE

Validation metrics config (trainer.val_metrics in YAML):
  val_metrics:
    enabled: true
    metrics: [SI-SDR, PESQ, STOI, ERLE]  # ERLE 仅消费显式 echo stem 或无噪声 echo_only
    interval: 5                      # compute every N validations
    max_utterances: 200              # 每 epoch 全局总数；各 rank 自动取 max_utterances // world_size
    max_utterances_metrics: null     # 用于指标计算的全局总数（null = 用 max_utterances）；同上按 rank 均分
    max_utterances_erle: 200         # ERLE 的独立全局 quota；同上按 rank 精确均分
    metrics_chunk_size: 512         # chunk size for incremental metric computation; reduces CPU memory
    strict: false                   # true: missing objective result raises error
    min_coverage: {ERLE: 0.9}       # valid / 实际提交计算的 eligible utterance
    metrics_workers: 0               # ProcessPool workers for intrusive/ERLE (0 = auto, cap by cpu_count)

ERLE 在 validation epoch 内使用 spawn ProcessPool，epoch 结束或异常时统一回收。
``val/ERLE`` 保留历史逐 utterance dB 宏平均键；显式 echo stem 存在时直接以
echo 与 ``estimate-nearend`` 计算，否则只允许可证明无噪声的 ``echo_only``
使用 residual 近似。兼容键 ``val/ERLE_residual_macro`` 与
``val/ERLE_energy_micro`` 保持不变；显式 echo stem 同时记录
``val/ERLE_oracle``，并始终用对应 count 暴露支持度。
DNSMOS/NISQA/UTMOS/SCOREQ 等 MOS 通过 ``evaluate/`` 离线计算。
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from tasks.aec.data.batch import AECBatch
from tasks.aec.inference.aec_frontend import (
    AECDeploymentFrontend,
    AECFrontendConfig,
    resolve_aec_frontend_config,
)
from tasks.aec.model_io import aec_model_accepts_multichannel
from core.metrics.aec import compute_erle
from core.metrics.batch import _shutdown_intrusive_pool
from core.models.execution_plan import resolve_trainer_execution_plan
from core.trainer.metrics_trainer import (
    MetricsCapableTrainer,
    _composite_loss_comp_log_label,
    _composite_ss_undesired_kwargs as _core_composite_ss_undesired_kwargs,
    _direct_spectral_loss_kwargs,
    _waveform_loss_kwargs,
    sanitize_model_output,
    validate_waveform_lengths,
)
from core.trainer.validation_metrics_config import (
    validate_validation_metric_runtime_result,
)
from tasks.aec.metrics.state_metrics import compute_state_counts, state_metrics_from_counts
from tasks.aec.data.state_label import FST_DT_NST_CLASS_NAMES, SCENARIO_CLASS_NAMES

try:
    from core.losses.composite import CompositeLoss
except ImportError:
    CompositeLoss = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

_ERLE_CHUNK_TIMEOUT_SEC = 120.0
_ERLE_WAIT_POLL_SEC = 0.5


def _stft_observation(stft: Any, waveform: torch.Tensor) -> torch.Tensor:
    """对 ``[B,C,T]`` microphone observation 逐 channel 计算 STFT。"""
    if waveform.ndim == 2:
        return stft(waveform)
    if waveform.ndim != 3 or waveform.shape[1] < 1:
        raise ValueError(
            f"AEC microphone must be [B,T] or [B,C,T], got {tuple(waveform.shape)}"
        )
    batch, channels, length = waveform.shape
    spectra = stft(waveform.reshape(batch * channels, length))
    return spectra.reshape(batch, channels, *spectra.shape[1:])


def _compute_erle_one(task):
    """spawn worker 入口：按可信 stem 优先级计算 ERLE，并返回异常值。"""
    mic_np, near_np, est_np, far_np, echo_np, fs = task
    try:
        if echo_np is None:
            residual = compute_erle(
                mic_np,
                near_np,
                est_np,
                fs=fs,
                far=far_np,
                return_energy_sums=True,
            )
        else:
            # 显式 echo stem 是唯一不受 nearend/noise 分解误差影响的输入；
            # 输出 residual 对应 estimate - nearend。
            residual = compute_erle(
                echo_np,
                np.zeros_like(echo_np),
                est_np - near_np,
                fs=fs,
                far=far_np,
                return_energy_sums=True,
            )
        if residual is not None and not isinstance(residual, tuple):
            # 兼容自定义/测试中的旧 compute_erle 替身；真实实现返回四元组。
            residual = (float(residual), 0.0, 0.0, 0)
        oracle = residual if echo_np is not None else None
        return residual, oracle
    except Exception as exc:
        return exc


def _aec_composite_metadata_kwargs(
    loss_fn: object,
    data: dict | None,
) -> dict[str, torch.Tensor]:
    """组装 AEC Composite 专属的 pure/state metadata。"""
    if data is None:
        return {}
    inner = getattr(loss_fn, "loss", loss_fn)
    if CompositeLoss is None or not isinstance(inner, CompositeLoss):
        return {}
    out: dict[str, torch.Tensor] = {}
    for key in ("is_pure_noise", "is_pure_echo", "is_pure_echo_noise"):
        if key in data:
            out[key] = data[key]
    if _has_active_state_ce(loss_fn):
        for key in ("state", "state_probs"):
            if key in data:
                out[key] = data[key]
    return out


def _composite_ss_undesired_kwargs(
    loss_fn: object,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    enhanced: torch.Tensor,
    data: dict | None = None,
) -> dict[str, torch.Tensor]:
    """兼容旧测试 seam：共享 SS/FW-SDR waveform 加 AEC pure/state metadata。"""
    out = _core_composite_ss_undesired_kwargs(loss_fn, noisy, clean, enhanced)
    out.update(_aec_composite_metadata_kwargs(loss_fn, data))
    return out


def _aec_loss_kwargs(
    loss_fn: object,
    mic: torch.Tensor,
    nearend: torch.Tensor,
    nearend_estimate: torch.Tensor,
    sample_rate: int,
    data: dict | None = None,
    stft_module: object | None = None,
) -> dict[str, Any]:
    """按活动 capability 组装 waveform、direct-spectrum 与 AEC metadata。"""
    out = _waveform_loss_kwargs(
        loss_fn,
        mic,
        nearend,
        nearend_estimate,
        sample_rate,
    )
    out.update(_aec_composite_metadata_kwargs(loss_fn, data))
    if data is not None:
        out.update(
            _direct_spectral_loss_kwargs(
                loss_fn,
                stft_module,
                nearend,
                data.get("loss_enhanced_spectra"),
                lengths=data.get("lengths"),
            )
        )
    return out


def _has_active_state_ce(loss_fn: object) -> bool:
    """判断 loss 是否启用了非零权重的 StateCrossEntropyLoss 分量。

    先解包 ``_StrictLossWrapper``（``inner = getattr(loss_fn, "loss", loss_fn)``），
    仅当内层是 ``CompositeLoss`` 且 ``_component_types`` 含 ``StateCrossEntropyLoss``
    且对应 ``_weights_float[i] != 0`` 时返回 True（weight=0 的分量视为未启用，
    不构成对 state 监督的依赖；照 tse_trainer._composite_spk_ce_kwargs 的
    weight 过滤语义）。
    """
    inner = getattr(loss_fn, "loss", loss_fn)
    if CompositeLoss is None or not isinstance(inner, CompositeLoss):
        return False
    return any(
        typ == "StateCrossEntropyLoss" and weight != 0.0
        for typ, weight in zip(
            getattr(inner, "_component_types", []),
            getattr(inner, "_weights_float", []),
        )
    )


def _unpack_aec_batch(batch: AECBatch) -> AECBatch:
    """消费 collate 的结构化批，拒绝位置 tuple/list。"""
    if not isinstance(batch, AECBatch):
        raise TypeError(
            f"AECTrainer requires an AECBatch from aec_collate_fn, got {type(batch).__name__}"
        )
    batch.validate()
    return batch


class AECTrainer(MetricsCapableTrainer):
    """Acoustic echo cancellation trainer.

    AEC models expect a dict input with ``mic_spectra`` / ``farend_spectra``
    and return a dict with ``nearend_estimate_spectra``.  The trainer therefore
    uses ``_forward_aec_batch`` (STFT → model(dict) → iSTFT) instead of the
    simpler ``_forward_model`` path used by SETrainer.

    Args:
        train_loader:  Training DataLoader（collate 后为 ``AECBatch``）。
        val_loader:    Validation DataLoader（合同同 ``train_loader``）。
        sample_rate:   Default sample rate.
        stft_module:   Framework STFT/iSTFT module (required for spectrum AEC models).
    """


    def __init__(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        sample_rate: int = 16000,
        stft_module: Any = None,
        **kw: Any,
    ):
        super().__init__(
            train_loader=train_loader,
            val_loader=val_loader,
            sample_rate=sample_rate,
            stft_module=stft_module,
            **kw,
        )
        raw_model = self._raw_model()
        self.execution_plan = resolve_trainer_execution_plan(
            self.config,
            raw_model,
            task="aec",
            sample_rate=self.sample_rate,
            dataset_contract=getattr(self.train_dataset, "contract", None),
            stft_module=self.stft_module,
        )
        self._aec_frontend_config = resolve_aec_frontend_config(self.config)
        self._aec_frontend = AECDeploymentFrontend(self._aec_frontend_config)
        self._aec_frontend_enabled = bool(
            self._aec_frontend_config.delay_alignment.enabled
            or self._aec_frontend_config.linear_aec is not None
        )
        self._validate_aec_frontend_capabilities()
        self._aec_persistent_state_repair_pending = False

    def _scan_aec_persistent_state(self) -> None:
        """仅检测 AEC 模型 BatchNorm 运行状态是否被污染，不做任何原地修改。

        ``strict_checks=False`` 的正常路径依赖标量 loss finite fail-safe，不能为
        此恢复多个全波形扫描。AEC 的离线模型没有跨调用 hidden state，但多个模型
        使用 BatchNorm 的 ``running_mean`` / ``running_var``；异常输入可能先污染
        这些小型持久 buffer，再被输出 sanitize 掩盖。这里仅扫描这些 buffer，计数
        留在设备上，epoch 汇总时才物化。

        训练路径 forward 之后、backward 之前绝不能原地改写 running buffers：
        BatchNorm backward 保存了这些统计的版本，提前改写会报
        ``RuntimeError: ... modified by an inplace operation``。因此训练路径只缓存
        设备侧 bad_count；loss finite 检查时一次物化并置 pending，随后由训练循环
        在 backward 完成后调用 ``_flush_persistent_state_repair`` 统一修复。
        """
        self._aec_last_state_bad_count = None
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "modules"):
            return
        buffers = getattr(self, "_aec_persistent_state_buffers", None)
        if buffers is None:
            buffers = tuple(
                (getattr(module, name), replacement)
                for module in model.modules()
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                for name, replacement in (("running_mean", 0.0), ("running_var", 1.0))
                if getattr(module, name, None) is not None
                and getattr(module, name).is_floating_point()
            )
            self._aec_persistent_state_buffers = buffers
        bad_count: torch.Tensor | None = None
        for state, _replacement in buffers:
            nonfinite = (~torch.isfinite(state)).sum(dtype=torch.int64)
            bad_count = nonfinite if bad_count is None else bad_count + nonfinite
        if bad_count is None:
            return
        previous = getattr(self, "_aec_state_n_nonfinite", None)
        self._aec_state_n_nonfinite = (
            bad_count if previous is None else previous + bad_count
        )
        # 训练路径不在这里 .item()：由 _forward_and_compute_loss 将污染位与
        # loss finite 位编码到同一个标量后一次物化，避免每 batch 额外 CUDA stall。
        self._aec_last_state_bad_count = bad_count

    def _materialize_aec_state_repair_pending(self) -> bool:
        """消费最近一次状态扫描结果并更新 pending 标记。"""
        bad_count = getattr(self, "_aec_last_state_bad_count", None)
        self._aec_last_state_bad_count = None
        if bad_count is not None and int(bad_count.item()) > 0:
            self._aec_persistent_state_repair_pending = True
        return self._has_persistent_state_repair_pending()

    def _repair_aec_persistent_state_now(self) -> None:
        """立即把非有限的 BatchNorm running buffers 重置为有限默认值。

        只允许在确定不会再对当前 graph 做 backward 的边界调用：forward 抛错、
        验证/推理路径（inference_mode），或训练循环在 backward 完成后的 flush。
        """
        buffers = getattr(self, "_aec_persistent_state_buffers", None)
        if not buffers:
            return
        for state, replacement in buffers:
            # 调用方已确认当前 graph 不会再 backward；直接原地修复所有小型
            # running buffers，避免逐 buffer ``any().item()`` 的额外 GPU 同步。
            torch.nan_to_num_(
                state,
                nan=replacement,
                posinf=replacement,
                neginf=replacement,
            )

    def _has_persistent_state_repair_pending(self) -> bool:
        """模板 hook：上一次 forward 是否污染了待修复的持久状态。"""
        return bool(getattr(self, "_aec_persistent_state_repair_pending", False))

    def _flush_persistent_state_repair(self) -> None:
        """模板 hook：训练循环在 backward 完成后调用，此时改写 buffers 已安全。"""
        if not getattr(self, "_aec_persistent_state_repair_pending", False):
            return
        self._repair_aec_persistent_state_now()
        self._aec_persistent_state_repair_pending = False
        self._aec_last_state_bad_count = None

    def _repair_persistent_state_after_abandoned_graph(self) -> None:
        """异常已放弃当前 graph，物化最近扫描并立即修复 BatchNorm 状态。"""
        if self._materialize_aec_state_repair_pending():
            self._flush_persistent_state_repair()

    def _reset_output_sanitize_stats(self) -> None:
        """清零输出与 AEC 持久状态修正计数。"""
        super()._reset_output_sanitize_stats()
        self._aec_state_n_nonfinite = None

    def _allreduce_aec_state_sanitize_count(self) -> torch.Tensor:
        """在 epoch boundary 固定归约 BatchNorm 状态修正计数。

        即使本地没有发现坏 buffer，也必须参与同一个 SUM collective；否则
        DDP 中不同 rank 的本地异常情况会改变 collective 序列。计数只在这里
        归约并由 rank 0 物化，正常 forward 热路径不触发 host 同步。
        """
        count = getattr(self, "_aec_state_n_nonfinite", None)
        if count is None:
            count = torch.zeros((), dtype=torch.int64, device=self.device)
        else:
            count = count.to(dtype=torch.int64, device=self.device)
        if getattr(self, "world_size", 1) > 1:
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        return count

    def _log_output_sanitize_summary(self, epoch: int, tag: str) -> None:
        """在 epoch boundary 归约并由 rank 0 报告输出/BatchNorm 修正统计。"""
        super()._log_output_sanitize_summary(epoch, tag)
        count = self._allreduce_aec_state_sanitize_count()
        if getattr(self, "rank", 0) != 0:
            return
        count_i = int(count.item())
        if count_i > 0:
            logger.warning(
                "%s E%s: AEC BatchNorm running state contained %d non-finite elements "
                "globally; replaced with finite defaults",
                tag,
                epoch,
                count_i,
            )

    def _validate_aec_frontend_capabilities(self) -> None:
        """在首个 batch 前校验 canonical routing 与模型输入能力。"""
        config = self._aec_frontend_config
        if config.input_routing != "mic_farend_error":
            return
        capabilities = self._aec_input_capabilities()
        if capabilities is not None and "error_spectra" not in capabilities:
            raise ValueError(
                "aec_frontend.input_routing='mic_farend_error' requires model "
                "capability 'error_spectra'; configured model only declares "
                f"{sorted(capabilities)}"
            )

    @staticmethod
    def _waveform_length_mask(waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """保形清零 declared length 外的尾部，避免 padding 进入前端或 STFT。"""
        if waveform.dim() not in (2, 3):
            raise ValueError(f"AEC waveform must be [B,T] or [B,C,T], got {tuple(waveform.shape)}")
        if waveform.dim() == 3 and waveform.shape[1] < 1:
            raise ValueError(f"AEC waveform channel dimension must be positive, got {tuple(waveform.shape)}")
        if lengths.dim() != 1 or lengths.shape[0] != waveform.shape[0]:
            raise ValueError(
                f"AEC lengths must be [B]={waveform.shape[0]}, got {tuple(lengths.shape)}"
            )
        sample_indices = torch.arange(waveform.shape[-1], device=waveform.device)
        valid = sample_indices.unsqueeze(0) < lengths.to(waveform.device).unsqueeze(1)
        while valid.ndim < waveform.ndim:
            valid = valid.unsqueeze(1)
        return waveform.masked_fill(~valid, 0.0)

    @staticmethod
    def _batch_has_valid_error(batch: AECBatch) -> bool:
        """判断 legacy/offline batch 是否在有效 waveform 区间携带预生成残差。"""
        if batch.error is None:
            return False
        if batch.error_mask is not None:
            AECTrainer._validate_error_mask(
                batch.error_mask,
                batch_size=batch.error.shape[0],
                device=batch.error.device,
            )
            return bool(batch.error_mask.any().item())
        valid_error = AECTrainer._waveform_length_mask(batch.error, batch.lengths)
        return bool((valid_error.abs().amax(dim=-1) > 1e-6).any().item())

    @staticmethod
    def _validate_error_mask(
        error_mask: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        """校验 residual 可用性掩码的逐样本 bool 合同。"""
        if not isinstance(error_mask, torch.Tensor):
            raise TypeError(
                f"AEC error_mask must be torch.Tensor, got {type(error_mask)}"
            )
        if error_mask.dtype != torch.bool:
            raise TypeError(
                f"AEC error_mask must use torch.bool, got {error_mask.dtype}"
            )
        if error_mask.shape != (batch_size,):
            raise ValueError(
                f"AEC error_mask must have shape ({batch_size},), got "
                f"{tuple(error_mask.shape)}"
            )
        if error_mask.device != device:
            raise ValueError(
                f"AEC error_mask device {error_mask.device} does not match "
                f"waveform device {device}"
            )

    def _resolve_aec_frontend(self) -> tuple[AECFrontendConfig, AECDeploymentFrontend, bool]:
        """兼容跳过 ``__init__`` 的 helper 测试，同时保持单一前端实例。"""
        config = getattr(self, "_aec_frontend_config", None)
        if config is None:
            config = resolve_aec_frontend_config(getattr(self, "config", None))
            self._aec_frontend_config = config
        frontend = getattr(self, "_aec_frontend", None)
        if frontend is None:
            frontend = AECDeploymentFrontend(config)
            self._aec_frontend = frontend
        enabled = getattr(self, "_aec_frontend_enabled", None)
        if enabled is None:
            enabled = bool(
                config.delay_alignment.enabled or config.linear_aec is not None
            )
            self._aec_frontend_enabled = enabled
        return config, frontend, bool(enabled)

    def _run_aec_frontend(
        self,
        batch: AECBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """在 STFT 前运行统一部署前端并落实显式 input routing。"""
        config, frontend, enabled = self._resolve_aec_frontend()
        has_batch_error = self._batch_has_valid_error(batch)
        if config.linear_aec is not None and has_batch_error:
            raise RuntimeError(
                "runtime LinearAEC conflicts with pre-generated batch.error/error_scp; "
                "disable runtime linear_aec or remove the pre-generated error track"
            )

        if enabled:
            sample_rate = self._batch_fs_hz(batch.fs, batch.mic.shape[0])
            frontend_mic = self._adapt_aec_mic(batch.mic)
            aligned_farend, runtime_error, runtime_mask, _facts = frontend.process_batch(
                frontend_mic,
                batch.farend,
                batch.lengths,
                sample_rate,
            )
            # shared frontend 在 inference_mode 下运行；clone 将返回值转成可供
            # 训练 autograd 保存的普通 tensor，同时保持前端本身不可微。
            aligned_farend = aligned_farend.clone()
            if runtime_error is not None:
                runtime_error = runtime_error.clone()
        else:
            # 严格 no-op：未启用 AEC input/deployment frontend 时不调用
            # frontend/backend，也不复制波形，保留原 tensor identity。
            aligned_farend = batch.farend
            runtime_error = None
            runtime_mask = None

        if config.input_routing == "error_farend":
            if runtime_error is None:
                raise RuntimeError("error_farend routing requires runtime LinearAEC output")
            return runtime_error, aligned_farend, None, None
        if config.input_routing == "mic_farend_error":
            if runtime_error is None or runtime_mask is None:
                raise RuntimeError(
                    "mic_farend_error routing requires runtime LinearAEC output"
                )
            self._validate_error_mask(
                runtime_mask,
                batch_size=runtime_error.shape[0],
                device=runtime_error.device,
            )
            return batch.mic, aligned_farend, runtime_error, runtime_mask

        # 预生成 error 只作为旧 7-tuple 的兼容输入；canonical runtime
        # mic_farend 路由本身绝不生成或附加 error。
        if (
            config.linear_aec is None
            and has_batch_error
            and self._aec_needs_input("error_spectra")
        ):
            error = batch.error
            if not getattr(self, "_strict_checks", True):
                error = torch.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
            elif not torch.isfinite(error).all():
                logger.warning(
                    "pre-generated LinearAEC error contains NaN/Inf; zeroing out"
                )
                error = torch.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
            error_mask = (
                batch.error_mask
                if batch.error_mask is not None
                else (error.abs().amax(dim=-1) > 1e-6).bool()
            )
            self._validate_error_mask(
                error_mask,
                batch_size=error.shape[0],
                device=error.device,
            )
            return batch.mic, aligned_farend, error, error_mask
        return batch.mic, aligned_farend, None, None

    def _require_stft_module(self) -> Any:
        """Return the configured STFT module or fail with the AEC-specific hint."""
        stft = self.stft_module
        if stft is None:
            raise RuntimeError(
                "AECTrainer requires stft_module. Set top-level 'stft' in YAML "
                "or pass a built module from train.py."
            )
        return stft

    def _adapt_aec_mic(self, mic: torch.Tensor) -> torch.Tensor:
        """校验 AEC 模型是否声明了原生多通道 microphone 输入。"""
        if mic.ndim == 2:
            return mic
        if mic.ndim != 3 or mic.shape[1] < 1:
            raise ValueError(
                f"AEC microphone must have shape [B,T] or [B,C,T], got {tuple(mic.shape)}"
            )
        if not aec_model_accepts_multichannel(self._raw_model()):
            raise ValueError(
                "multichannel AEC microphone requires a native model IO contract "
                "with a microphone axis; reference-channel projection is not supported"
            )
        return mic

    def _aec_input_capabilities(self) -> frozenset[str] | None:
        """从已解析 execution plan 返回模型谱输入能力。"""
        plan = getattr(self, "execution_plan", None)
        if plan is not None:
            return frozenset(field.name for field in plan.input_fields)
        model = getattr(self, "model", None)
        if model is not None and hasattr(self, "_raw_model"):
            model = self._raw_model()
        capabilities = getattr(model, "aec_input_capabilities", None)
        if capabilities is None:
            return None
        return frozenset(str(key) for key in capabilities)

    def _aec_needs_input(self, key: str) -> bool:
        """仅对显式声明的能力做输入裁剪，避免破坏旧 AEC 模型。"""
        capabilities = self._aec_input_capabilities()
        return capabilities is None or key in capabilities

    def _build_aec_model_input(
        self,
        batch: AECBatch,
        *,
        model_mic: torch.Tensor | None = None,
        aligned_farend: torch.Tensor | None = None,
        error: torch.Tensor | None = None,
        error_mask: torch.Tensor | None = None,
        nontarget: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """按 canonical routing 构造 AEC 模型消费的波形/谱域字典。"""
        stft = self._require_stft_module()
        model_mic = self._adapt_aec_mic(
            batch.mic if model_mic is None else model_mic
        )
        aligned_farend = batch.farend if aligned_farend is None else aligned_farend
        if aligned_farend.ndim != 2 or aligned_farend.shape[0] != model_mic.shape[0] or aligned_farend.shape[-1] != model_mic.shape[-1]:
            raise ValueError(
                "AEC farend must remain mono [B,T] and share batch/time axes with "
                "the microphone observation, got "
                f"mic={tuple(model_mic.shape)}, farend={tuple(aligned_farend.shape)}"
            )
        data: dict[str, torch.Tensor] = {
            "mic": model_mic,
            "farend": aligned_farend,
            "nearend": batch.nearend,
            "lengths": batch.lengths,
        }
        if nontarget is not None:
            data["nontarget"] = nontarget
        # 标签（is_pure_* / state）只经 _restore_batch_metadata 进入 loss/metrics
        # 消费的 dict，不得作为模型输入，避免新模型直接读取标签。

        # microphone observation keeps its channel axis; far-end remains mono.
        data["mic_spectra"] = _stft_observation(stft, model_mic)
        data["farend_spectra"] = stft(aligned_farend)
        if error is not None:
            if error_mask is None:
                raise RuntimeError("error_spectra requires an explicit error_mask")
            if error.ndim != 2 or error.shape[0] != model_mic.shape[0] or error.shape[-1] != model_mic.shape[-1]:
                raise ValueError(
                    "AEC error must be mono [B,T] and share batch/time axes with "
                    "the microphone observation, got "
                    f"error={tuple(error.shape)}, mic={tuple(model_mic.shape)}"
                )
            self._validate_error_mask(
                error_mask,
                batch_size=error.shape[0],
                device=error.device,
            )
            # mixed batch 中无 residual 的行必须使用模型声明的 mic fallback；
            # 零 residual 会让第三输入语义依赖 batch 组成。
            effective_error = torch.where(
                error_mask.unsqueeze(-1), error, model_mic
            )
            data["error_spectra"] = stft(effective_error)
            data["error_mask"] = error_mask
        if self._aec_needs_input("nearend_spectra"):
            data["nearend_spectra"] = stft(batch.nearend)
        if nontarget is not None and self._aec_needs_input("nontarget_spectra"):
            data["nontarget_spectra"] = stft(nontarget)
        data["stft_n_fft"] = torch.tensor(int(stft.n_fft), device=batch.mic.device)
        data["stft_hop_length"] = torch.tensor(
            int(stft.hop_length), device=batch.mic.device
        )
        return data

    def _decode_aec_model_output(
        self,
        data: dict[str, torch.Tensor],
        batch: AECBatch,
    ) -> dict[str, torch.Tensor]:
        """Decode spectrum/waveform model output and enforce the target shape."""
        stft = self._require_stft_module()
        ref_len = batch.mic.shape[-1]

        # Compatibility: NKF and similar models may output a waveform because they
        # own an internal STFT.  Pad/trim it and attach the spectrum expected by loss.
        if "nearend_estimate_spectra" not in data and "nearend_estimate" in data:
            nearend_estimate = data["nearend_estimate"]
            if nearend_estimate.shape[-1] < ref_len:
                nearend_estimate = torch.nn.functional.pad(
                    nearend_estimate, (0, ref_len - nearend_estimate.shape[-1])
                )
            elif nearend_estimate.shape[-1] > ref_len:
                nearend_estimate = nearend_estimate[..., :ref_len]
            data["nearend_estimate"] = nearend_estimate
            data["nearend_estimate_spectra"] = stft(nearend_estimate)
        elif "nearend_estimate_spectra" in data:
            nearend_estimate = stft.inverse(data["nearend_estimate_spectra"], ref_len)
            min_len = min(nearend_estimate.shape[-1], ref_len)
            data["nearend_estimate"] = nearend_estimate[..., :min_len]
        else:
            raise RuntimeError(
                "Model must output either 'nearend_estimate_spectra' or 'nearend_estimate'. "
                f"Got keys: {list(data.keys())}"
            )

        if data["nearend_estimate"].shape != batch.nearend.shape:
            raise RuntimeError(
                f"AEC shape mismatch: nearend_estimate {data['nearend_estimate'].shape} "
                f"!= nearend {batch.nearend.shape}. This usually means iSTFT output length "
                f"differs from reference (hop_length={stft.hop_length} may not divide evenly)."
            )
        return data

    def _mask_state_padding(self, batch: AECBatch) -> None:
        """将 declared waveform length 外的 state 帧标记为 ignore。"""
        state = batch.state
        if state is None:
            return
        if state.dtype != torch.int64:
            raise TypeError(
                f"AEC state labels must use torch.int64, got {state.dtype}"
            )
        if state.dim() != 2 or state.shape[0] != batch.lengths.shape[0]:
            raise ValueError(
                "AEC state must be mutually-exclusive class indices with shape "
                "[B, T_state], got "
                f"state={tuple(state.shape)}, lengths={tuple(batch.lengths.shape)}"
            )
        hop_length = int(self._require_stft_module().hop_length)
        if hop_length <= 0:
            raise ValueError(f"AEC STFT hop_length must be positive, got {hop_length}")
        frame_indices = torch.arange(state.shape[1], device=state.device)
        valid_frames = batch.lengths.to(state.device, dtype=torch.long).div(
            hop_length, rounding_mode="floor"
        ) + 1
        invalid = frame_indices.unsqueeze(0) >= valid_frames.unsqueeze(1)
        batch.state = state.masked_fill(invalid, -1)

    @staticmethod
    def _restore_batch_metadata(data: dict[str, torch.Tensor], batch: AECBatch) -> None:
        """以 AECBatch 为权威恢复 loss/metric 所需 metadata。"""
        data["lengths"] = batch.lengths
        for key in ("is_pure_noise", "is_pure_echo", "is_pure_echo_noise", "state"):
            value = getattr(batch, key)
            if value is None:
                data.pop(key, None)
            else:
                data[key] = value

    def _forward_aec_batch(
        self,
        batch: AECBatch,
        *,
        nontarget: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run AEC input/deployment frontend → STFT → model → decode for train and val."""
        # 即使调用方未走 train/val sanitize hook，也不能让物理 padding 进入前端或 STFT。
        batch.mic = self._waveform_length_mask(batch.mic, batch.lengths)
        batch.farend = self._waveform_length_mask(batch.farend, batch.lengths)
        batch.nearend = self._waveform_length_mask(batch.nearend, batch.lengths)
        if batch.error is not None:
            batch.error = self._waveform_length_mask(batch.error, batch.lengths)
        self._mask_state_padding(batch)
        model_mic, aligned_farend, error, error_mask = self._run_aec_frontend(batch)
        data = self._build_aec_model_input(
            batch,
            model_mic=model_mic,
            aligned_farend=aligned_farend,
            error=error,
            error_mask=error_mask,
            nontarget=nontarget,
        )
        try:
            data = self.model(data)
            if not isinstance(data, dict):
                raise TypeError(f"AEC model must return dict, got {type(data)}")
            self._restore_batch_metadata(data, batch)
            model_enhanced_spectra = data.get("nearend_estimate_spectra")
            data = self._decode_aec_model_output(data, batch)
            if model_enhanced_spectra is not None:
                data["loss_enhanced_spectra"] = model_enhanced_spectra
        except BaseException:
            # forward/decode 抛错后不会再有 backward，可立即检测并修复持久状态。
            self._scan_aec_persistent_state()
            if self._materialize_aec_state_repair_pending():
                self._flush_persistent_state_repair()
            raise
        # 仅检查小型 BatchNorm running buffers；不恢复输入 waveform 全量扫描。
        self._scan_aec_persistent_state()
        if not torch.is_grad_enabled() and self._materialize_aec_state_repair_pending():
            # 验证/推理路径无 backward，可在 batch 边界直接修复；训练路径
            # 必须等 backward 完成后由训练循环 flush，否则会破坏 BN backward
            # 保存的 running statistics 版本。
            self._flush_persistent_state_repair()
        return data

    # NOTE: AEC training always uses _forward_aec_batch (STFT → model dict → iSTFT).
    # The MetricsCapableTrainer._forward_model path is not applicable; forward is routed
    # through _forward_aec_batch in both _train_epoch and _validation_epoch.

    def _assert_batch_valid(
        self,
        mic: torch.Tensor,
        farend: torch.Tensor,
        nearend: torch.Tensor,
        lengths: torch.Tensor,
        error: torch.Tensor | None = None,
    ) -> None:
        """严格校验 mic observation 与全部 mono AEC 轨道的边界合同。"""
        if mic.dim() not in (2, 3) or farend.dim() != 2 or nearend.dim() != 2:
            raise ValueError(
                "mic must be [B, T] or [B, C, T]; farend and nearend must be [B, T], got "
                f"mic={tuple(mic.shape)}, farend={tuple(farend.shape)}, "
                f"nearend={tuple(nearend.shape)}"
            )
        if mic.shape[0] != farend.shape[0] or mic.shape[-1] != farend.shape[-1] or farend.shape != nearend.shape:
            raise ValueError(
                f"Shape mismatch: mic={tuple(mic.shape)}, "
                f"farend={tuple(farend.shape)}, nearend={tuple(nearend.shape)}"
            )
        if mic.dim() == 3 and mic.shape[1] < 1:
            raise ValueError(f"AEC mic channel dimension must be positive, got {tuple(mic.shape)}")
        if error is not None and (error.dim() != 2 or error.shape != nearend.shape):
            raise ValueError(
                "AEC error must be mono [B,T] aligned with nearend, got "
                f"error={tuple(error.shape)}, nearend={tuple(nearend.shape)}"
            )
        if lengths.dim() != 1 or lengths.shape[0] != mic.shape[0]:
            raise ValueError(
                f"lengths must be [B], got {tuple(lengths.shape)} for B={mic.shape[0]}"
            )
        values = (mic, farend, nearend) if error is None else (mic, farend, nearend, error)
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("AEC batch contains NaN/Inf")

    # ------------------------------------------------------------------
    # Template-method hooks (unified in core.trainer.metrics_trainer)
    # ------------------------------------------------------------------

    _train_epoch_tail_label = "AEC"

    def _unpack_batch(self, raw_batch: Any) -> AECBatch:
        return _unpack_aec_batch(raw_batch)

    def _validate_batch_metadata(
        self, batch: AECBatch, epoch: int, batch_idx: int
    ) -> None:
        validate_waveform_lengths(
            batch.lengths,
            batch_size=batch.mic.shape[0],
            max_length=batch.mic.shape[-1],
            context=f"AEC train E{epoch} B{batch_idx}",
        )

    def _maybe_count_silence_placeholders(self, batch: AECBatch) -> int:
        # 非 strict 模式只保留 loss finite fail-safe，跳过额外的全 batch 能量扫描。
        if not self._strict_checks:
            return 0
        nearend = batch.nearend
        mic = batch.mic
        # 占位样本是逐字段全零波形（见 Dataset._placeholder_sample →
        # BaseDynamicMixingDataset._make_silence_placeholder），其长度随
        # max_duration / max_duration_seconds 配置变化（如 10s→160000），
        # 与 _placeholder_num_samples（默认 1024）并不相等，旧的
        # lengths == _placeholder_num_samples 判据因此永远漏检。
        # 改为直接以能量判据识别：mic + nearend 同时为零能量才是占位样本；
        # 要求 mic 也为零可排除 echo_only 等近端静音但 mic 含回声的真实样本。
        energy_ne = (nearend * nearend).sum(dim=1)
        energy_mic = (mic * mic).flatten(start_dim=1).sum(dim=1)
        return int(((energy_ne < 1e-10) & (energy_mic < 1e-10)).sum().item())

    def _sanitize_batch_inputs(
        self, batch: AECBatch, epoch: int, batch_idx: int
    ) -> AECBatch:
        if self._strict_checks:
            self._assert_batch_valid(
                batch.mic, batch.farend, batch.nearend, batch.lengths, batch.error
            )
        else:
            # 非 strict 模式不扫描 finite 状态，直接将异常输入清零，避免污染模型计算图。
            batch.mic = torch.nan_to_num(batch.mic, nan=0.0, posinf=0.0, neginf=0.0)
            batch.farend = torch.nan_to_num(
                batch.farend, nan=0.0, posinf=0.0, neginf=0.0
            )
            batch.nearend = torch.nan_to_num(
                batch.nearend, nan=0.0, posinf=0.0, neginf=0.0
            )
            if batch.error is not None:
                batch.error = torch.nan_to_num(
                    batch.error, nan=0.0, posinf=0.0, neginf=0.0
                )
        # declared length 是唯一有效样本边界；保留 batch shape 以兼容卷积/RNN。
        batch.mic = self._waveform_length_mask(batch.mic, batch.lengths)
        batch.farend = self._waveform_length_mask(batch.farend, batch.lengths)
        batch.nearend = self._waveform_length_mask(batch.nearend, batch.lengths)
        if batch.error is not None:
            batch.error = self._waveform_length_mask(batch.error, batch.lengths)
        return batch

    def _forward_and_compute_loss(
        self,
        batch: AECBatch,
        epoch: int,
        batch_idx: int,
        fs_val: int,
    ) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        data = self._forward_aec_batch(batch)
        try:
            nearend_estimate = data["nearend_estimate"]
            # A23：train/val 统一 sanitize（nan_to_num + clamp(-1,1)）；
            # 非有限/钳位元素数按 epoch 累加，边界处汇总上报。
            nearend_estimate, _n_nf, _n_cl = sanitize_model_output(nearend_estimate)
            self._accumulate_output_sanitize(_n_nf, _n_cl)

            _vloss_kw = _aec_loss_kwargs(
                self.loss_fn,
                self._adapt_aec_mic(batch.mic),
                batch.nearend,
                nearend_estimate,
                fs_val,
                data,
                getattr(self, "stft_module", None),
            )
            # fail-fast：非零 StateCE 已启用时，模型输出缺 state_probs（如无 state head
            # 的模型用 **kwargs 吸收 use_state_detector）或数据缺 state（如旧 tuple 无
            # state label）均会让 StateCrossEntropyLoss 静默返回零，辅助监督静默失效；
            # 模型/数据双查直接报错阻止继续训练。
            if _has_active_state_ce(self.loss_fn) and (
                "state_probs" not in data or "state" not in data
            ):
                raise ValueError(
                    "非零 StateCrossEntropyLoss 已启用但模型输出缺 state_probs 或数据缺"
                    " state，state 监督将静默失效"
                )
            loss = self.loss_fn(
                nearend_estimate,
                batch.nearend,
                batch.lengths,
                **_vloss_kw,
            )
            loss_finite = torch.isfinite(loss).to(dtype=torch.int64)
            bad_count = getattr(self, "_aec_last_state_bad_count", None)
            if bad_count is None:
                status = int(loss_finite.item())
            else:
                # bit0=loss finite，bit1=BatchNorm state corrupted；一次 .item()
                # 同时物化两个状态，避免为 running buffers 额外同步 GPU。
                packed = loss_finite | ((bad_count > 0).to(dtype=torch.int64) << 1)
                status = int(packed.item())
            loss_was_finite = bool(status & 1)
            if status & 2:
                self._aec_persistent_state_repair_pending = True
            # 常态只在日志间隔采集 collapse 诊断；非有限 loss 仍立即保留异常诊断。
            log_interval = max(1, int(getattr(self, "log_interval", 100)))
            collect_diagnostics = (batch_idx % log_interval == 0) or not loss_was_finite
            info = (
                self._collect_step_diagnostics(data, nearend_estimate, batch)
                if collect_diagnostics
                else {}
            )
            info["loss_was_finite"] = loss_was_finite
            if not loss_was_finite:
                ne_safe = torch.nan_to_num(
                    nearend_estimate.detach(), nan=0.0, posinf=0.0, neginf=0.0
                )
                info["nearend_estimate_safe_min"] = float(ne_safe.min().item())
                info["nearend_estimate_safe_max"] = float(ne_safe.max().item())
            return loss, loss_was_finite, info
        except BaseException:
            # forward 已成功，但 sanitize/loss/diagnostics 任一环节抛错后当前 graph
            # 都会被放弃；此时才物化并修复 running buffers，不触发版本错误。
            self._repair_persistent_state_after_abandoned_graph()
            raise

    def _collect_step_diagnostics(
        self,
        data: dict[str, Any],
        nearend_estimate: torch.Tensor,
        batch: AECBatch,
    ) -> dict[str, Any]:
        """Compute lightweight scalar diagnostics; avoids storing full tensors in loss_info."""
        info: dict[str, Any] = {"out_energy": float(nearend_estimate.abs().mean().item())}
        if info["out_energy"] >= 1e-5:
            return info
        with torch.no_grad():
            info["mic_energy"] = float(batch.mic.abs().mean().item())
            info["nearend_energy"] = float(batch.nearend.abs().mean().item())
            info["farend_energy"] = float(batch.farend.abs().mean().item())
            _pure_n = data.get("is_pure_noise")
            _pure_e = data.get("is_pure_echo")
            _pure_en = data.get("is_pure_echo_noise")
            _num_pure = 0
            if _pure_n is not None:
                _num_pure += int(
                    (_pure_n | (_pure_e if _pure_e is not None else False) | (_pure_en if _pure_en is not None else False)).sum().item()
                )
            info["num_pure"] = _num_pure
            _err = data.get("error_spectra")
            info["error_spectra_energy"] = float(_err.abs().mean().item()) if _err is not None else None
            _err_m = data.get("error_mask")
            info["error_mask_ratio"] = (
                float(_err_m.float().mean().item()) if _err_m is not None else None
            )
        return info

    def _maybe_log_step_diagnostics(
        self,
        batch: AECBatch,
        loss_info: dict[str, Any],
        epoch: int,
        batch_idx: int,
    ) -> None:
        if self.rank != 0:
            return
        out_energy = loss_info.get("out_energy")
        if out_energy is None or out_energy >= 1e-5:
            return
        num_pure = loss_info.get("num_pure", 0)
        mic_e = loss_info.get("mic_energy")
        ne_e = loss_info.get("nearend_energy")
        ref_e = loss_info.get("farend_energy")
        err_e = loss_info.get("error_spectra_energy")
        err_m = loss_info.get("error_mask_ratio")
        logger.warning(
            f"[E{epoch} B{batch_idx}] Output energy collapsed: {out_energy:.2e}. "
            f"Pure={num_pure}/{batch.mic.size(0)}. "
            f"InputE: mic={mic_e:.2e} nearend={ne_e:.2e} ref={ref_e:.2e} "
            f"err={err_e if err_e is not None else 'N/A'} "
            f"err_mask={err_m if err_m is not None else 'N/A'}. "
            f"Check Layer D silent-ref handling."
        )

    # ------------------------------------------------------------------
    # Validation hooks（core ``_validation_epoch`` 模板的 AEC 任务差异：
    # dict batch 前向、frame-level state detector 统计、ERLE）
    # ------------------------------------------------------------------

    def _advance_and_should_run_metrics(self) -> bool:
        do_metrics = super()._advance_and_should_run_metrics()
        self._aec_do_metrics = do_metrics
        return do_metrics

    def _shutdown_erle_executor(self, *, terminate: bool) -> None:
        """有界关闭 lazy ERLE spawn 池；异常时强制 terminate/kill。"""
        executor = self.__dict__.pop("_erle_executor", None)
        if executor is None:
            return
        try:
            _shutdown_intrusive_pool(executor, terminate=terminate)
        except Exception as exc:
            self.__dict__.setdefault("_erle_executor_errors", []).append(
                f"{type(exc).__name__}: {exc}"
            )

    @contextlib.contextmanager
    def _validation_epoch_resources(self):
        """整轮 validation 复用一个 lazy ERLE spawn 池，并有界关闭。"""
        self._erle_executor = None
        self._erle_executor_errors = []
        try:
            yield
        finally:
            self._shutdown_erle_executor(terminate=False)

    def _close_erle_executor(self, extra: dict[str, Any]) -> None:
        """有界关闭 ERLE 池并把生命周期异常并入本轮错误状态。"""
        self._shutdown_erle_executor(terminate=False)
        lifecycle_errors = self.__dict__.pop("_erle_executor_errors", [])
        if lifecycle_errors:
            extra.setdefault("erle_errors", []).extend(lifecycle_errors)

    def _init_val_extra_state(self) -> dict[str, Any]:
        # 根据模型架构静态判断是否存在 state detector，保证 DDP 各 rank 对 allreduce
        # 的参与与否达成一致，避免部分 rank 未进入分支导致死锁。
        raw_model = self._raw_model()
        num_state_classes = int(getattr(raw_model, "state_nclass", 0))
        state_metrics_available = bool(
            getattr(raw_model, "use_state_detector", False)
            and num_state_classes > 0
            and _has_active_state_ce(self.loss_fn)
        )
        # Accumulators for frame-level state-detector metrics.
        if state_metrics_available:
            state_tp = torch.zeros(num_state_classes, dtype=torch.int64, device=self.device)
            state_fp = torch.zeros(num_state_classes, dtype=torch.int64, device=self.device)
            state_fn = torch.zeros(num_state_classes, dtype=torch.int64, device=self.device)
        else:
            state_tp = None
            state_fp = None
            state_fn = None
        return {
            # 保持逐 utterance 先算 dB、再累计 sum/count 的历史合同；同时累计
            # 线性能量供 energy-micro 使用。oracle 只消费显式 echo stem。
            "erle_db_sum": 0.0,
            "erle_count": 0,
            "erle_eligible_count": 0,
            "erle_energy_sum_in": 0.0,
            "erle_energy_sum_out": 0.0,
            "erle_energy_count": 0,
            "oracle_erle_db_sum": 0.0,
            "oracle_erle_count": 0,
            "erle_errors": [],
            "erle_remaining": int(self._vm_max_erle),
            "num_state_classes": num_state_classes,
            "state_metrics_available": state_metrics_available,
            "state_tp": state_tp,
            "state_fp": state_fp,
            "state_fn": state_fn,
            "state_correct": torch.zeros(1, dtype=torch.int64, device=self.device),
            "state_valid": torch.zeros(1, dtype=torch.int64, device=self.device),
            "state_metrics": None,
        }

    def _val_prepare_batch(
        self, raw_batch: Any, epoch: int, n_loops: int
    ) -> tuple[AECBatch, int]:
        """在 DDP forward 前完成 AEC batch 解析、搬运与输入门禁。"""
        aec_batch = self._unpack_batch(raw_batch)
        validate_waveform_lengths(
            aec_batch.lengths,
            batch_size=aec_batch.mic.shape[0],
            max_length=aec_batch.mic.shape[-1],
            context=f"AEC validation E{epoch} B{n_loops}",
        )
        aec_batch = self._batch_to_device(aec_batch)
        fs_val = self._batch_fs_hz(
            aec_batch.fs, aec_batch.mic.shape[0]
        )  # handles int/float/Tensor natively
        aec_batch = self._sanitize_batch_inputs(aec_batch, epoch, n_loops)
        return aec_batch, fs_val

    def _val_forward_prepared(
        self, prepared_batch: Any, epoch: int, n_loops: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Any]:
        """仅消费已通过共享 prepare 异常同步门禁的 AEC batch。"""
        del epoch, n_loops
        if (
            not isinstance(prepared_batch, tuple)
            or len(prepared_batch) != 2
            or not isinstance(prepared_batch[0], AECBatch)
            or type(prepared_batch[1]) is not int
        ):
            raise TypeError(
                "AEC validation prepared batch must be (AECBatch, sample_rate_int)"
            )
        aec_batch, fs_val = prepared_batch
        def _forward_and_build_loss_kwargs():
            with torch.amp.autocast(
                self.device.type, enabled=self.use_amp, dtype=self.amp_dtype
            ):
                data = self._forward_aec_batch(aec_batch)
                nearend_estimate = self._sanitize_model_output(
                    data["nearend_estimate"]
                )
                if (
                    self._strict_checks
                    and nearend_estimate.shape != aec_batch.nearend.shape
                ):
                    raise ValueError(
                        "Model output shape mismatch: "
                        f"nearend_estimate={tuple(nearend_estimate.shape)} "
                        f"nearend={tuple(aec_batch.nearend.shape)}"
                    )
                loss_kwargs = _aec_loss_kwargs(
                    self.loss_fn,
                    self._adapt_aec_mic(aec_batch.mic),
                    aec_batch.nearend,
                    nearend_estimate,
                    fs_val,
                    data,
                    getattr(self, "stft_module", None),
                )
                if _has_active_state_ce(self.loss_fn) and (
                    "state_probs" not in data or "state" not in data
                ):
                    raise ValueError(
                        "非零 StateCrossEntropyLoss 已启用但模型输出缺 state_probs "
                        "或数据缺 state，state 监督将静默失效"
                    )
                return data, nearend_estimate, loss_kwargs

        data, nearend_estimate, loss_kwargs = self._run_synced_validation_subphase(
            _forward_and_build_loss_kwargs,
            phase="validation_model_output_error",
            context="AEC validation model/output preflight failed",
        )
        with torch.amp.autocast(
            self.device.type, enabled=self.use_amp, dtype=self.amp_dtype
        ):
            loss = self.loss_fn(
                nearend_estimate,
                aec_batch.nearend,
                aec_batch.lengths,
                **loss_kwargs,
            )
        ctx = {
            "data": data,
            "state": aec_batch.state,
            "mic": self._adapt_aec_mic(aec_batch.mic),
            "farend": aec_batch.farend,
            "nearend": aec_batch.nearend,
            "nearend_estimate": nearend_estimate,
            "lengths": aec_batch.lengths,
            "fs_val": fs_val,
            "has_echo": aec_batch.has_echo,
            "is_pure_echo": aec_batch.is_pure_echo,
            "is_pure_echo_noise": aec_batch.is_pure_echo_noise,
            # 标准 AEC tuple 未承诺可验证的 echo stem，绝不以 mic-nearend 伪造 oracle。
        }
        return (
            loss,
            nearend_estimate,
            aec_batch.nearend,
            aec_batch.lengths,
            fs_val,
            ctx,
        )

    def _val_forward_loss(
        self, raw_batch: Any, epoch: int, n_loops: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Any]:
        """兼容直接调用旧 hook 的测试与外部扩展。"""
        prepared = self._val_prepare_batch(raw_batch, epoch, n_loops)
        return self._val_forward_prepared(prepared, epoch, n_loops)

    def _validation_metric_reference_channel(self, mic: torch.Tensor) -> int | None:
        """Return the canonical microphone channel used by waveform metrics."""
        if mic.ndim == 2:
            return None
        if mic.ndim != 3:
            raise ValueError(f"AEC validation mic must be [B,T] or [B,C,T], got {tuple(mic.shape)}")
        from core.models.io_contract import resolve_dataset_microphone_array

        dataset = getattr(self.val_loader, "dataset", None)
        geometry = resolve_dataset_microphone_array(dataset)
        if geometry is None:
            raise ValueError(
                "AEC multichannel validation metrics require resolved microphone geometry"
            )
        reference_channel = int(geometry.reference_channel)
        if reference_channel >= mic.shape[1]:
            raise ValueError(
                "AEC validation reference channel exceeds microphone tensor: "
                f"reference_channel={reference_channel}, channels={mic.shape[1]}"
            )
        return reference_channel

    def _materialize_val_waveforms(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """一次性把 ERLE/objective 所需波形搬到 CPU，并在 ctx 内复用视图。"""
        cached = ctx.get("waveforms_cpu")
        if cached is not None:
            return cached
        # 组合后只执行一次 waveform GPU→CPU copy；flags 是小型元数据。
        mic = ctx["mic"]
        reference_channel = self._validation_metric_reference_channel(mic)
        if reference_channel is not None:
            mic = mic[:, reference_channel]
        waveform_keys = ["mic", "nearend", "nearend_estimate", "farend"]
        metric_waveforms = [mic, ctx["nearend"], ctx["nearend_estimate"], ctx["farend"]]
        echo = ctx.get("echo")
        if echo is not None:
            waveform_keys.append("echo")
            metric_waveforms.append(echo)
        waveforms_cpu = torch.stack(metric_waveforms, dim=1).detach().float().cpu()
        lengths_cpu = ctx["lengths"].detach().cpu().numpy()
        has_echo = ctx.get("has_echo")
        has_echo_cpu = has_echo.detach().cpu().numpy() if has_echo is not None else None
        is_pure_echo = ctx.get("is_pure_echo")
        is_pure_echo_cpu = (
            is_pure_echo.detach().cpu().numpy()
            if is_pure_echo is not None
            else None
        )
        is_pure_echo_noise = ctx.get("is_pure_echo_noise")
        is_pure_echo_noise_cpu = (
            is_pure_echo_noise.detach().cpu().numpy()
            if is_pure_echo_noise is not None
            else None
        )
        cached = {
            key if key != "nearend_estimate" else "estimate": waveforms_cpu[:, i].numpy()
            for i, key in enumerate(waveform_keys)
        }
        cached.update(
            {
                "lengths": lengths_cpu,
                "has_echo": has_echo_cpu,
                "is_pure_echo": is_pure_echo_cpu,
                "is_pure_echo_noise": is_pure_echo_noise_cpu,
            }
        )
        ctx["waveforms_cpu"] = cached
        return cached

    def _erle_workers(self) -> int:
        """按配置、CPU 与本 rank 整轮 quota 返回固定 ERLE 池容量。"""
        cpu_count = max(1, os.cpu_count() or 1)
        configured = getattr(self, "_vm_metrics_workers", None)
        if configured is None:
            configured = min(4, cpu_count)
        rank_quota = max(1, int(getattr(self, "_vm_max_erle", 1)))
        return max(1, min(int(configured), cpu_count, rank_quota))

    def _val_accumulate_extra(self, extra: Any, ctx: Any, is_ddp_pad: bool) -> None:
        if is_ddp_pad:
            return

        if (
            getattr(self, "_aec_do_metrics", False)
            and "ERLE" in getattr(self, "_vm_metrics", ("ERLE",))
            and not self.shutdown_requested()
        ):
            try:
                remaining = int(extra.get("erle_remaining", getattr(self, "_vm_max_erle", 1)))
                cached = self._materialize_val_waveforms(ctx) if remaining > 0 else None
                tasks = []
                if cached is not None:
                    for i, length in enumerate(cached["lengths"]):
                        if cached["has_echo"] is not None and not bool(cached["has_echo"][i]):
                            continue
                        has_explicit_echo = "echo" in cached
                        is_clean_echo_only = (
                            cached["is_pure_echo"] is not None
                            and bool(cached["is_pure_echo"][i])
                            and (
                                cached["is_pure_echo_noise"] is None
                                or not bool(cached["is_pure_echo_noise"][i])
                            )
                        )
                        # nearend_echo 与 nearend_noise_echo 在 canonical batch
                        # flags 上不可区分；没有显式 echo stem 时仅 echo_only
                        # 能证明 mic-nearend 不含背景噪声。
                        if not has_explicit_echo and not is_clean_echo_only:
                            continue
                        utt_len = int(length)
                        tasks.append(
                            (
                                cached["mic"][i, :utt_len],
                                cached["nearend"][i, :utt_len],
                                cached["estimate"][i, :utt_len],
                                cached["farend"][i, :utt_len],
                                cached["echo"][i, :utt_len]
                                if "echo" in cached
                                else None,
                                ctx["fs_val"],
                            )
                        )
                        if len(tasks) >= remaining:
                            break
                extra["erle_remaining"] = max(0, remaining - len(tasks))
                # eligible 是实际进入 ERLE worker 的 echo utterance；valid 是返回
                # 有限 residual ERLE 的 utterance。二者共同定义 runtime coverage。
                extra["erle_eligible_count"] = (
                    extra.get("erle_eligible_count", 0) + len(tasks)
                )

                executor = getattr(self, "_erle_executor", None)
                erle_values = []
                if tasks:
                    if executor is None:
                        executor = ProcessPoolExecutor(
                            max_workers=self._erle_workers(),
                            mp_context=multiprocessing.get_context("spawn"),
                        )
                        self._erle_executor = executor
                    futures = {executor.submit(_compute_erle_one, task) for task in tasks}
                    pending = set(futures)
                    deadline = time.monotonic() + _ERLE_CHUNK_TIMEOUT_SEC
                    while pending:
                        if self.shutdown_requested():
                            raise InterruptedError("ERLE aborted by shutdown request")
                        remaining_sec = deadline - time.monotonic()
                        if remaining_sec <= 0:
                            raise TimeoutError(
                                f"ERLE exceeded chunk deadline of {_ERLE_CHUNK_TIMEOUT_SEC:.0f}s"
                            )
                        done, pending = wait(
                            pending,
                            timeout=min(_ERLE_WAIT_POLL_SEC, remaining_sec),
                            return_when="FIRST_COMPLETED",
                        )
                        erle_values.extend(future.result() for future in done)
                for erle_value in erle_values:
                    if isinstance(erle_value, Exception):
                        extra.setdefault("erle_errors", []).append(
                            f"{type(erle_value).__name__}: {erle_value}"
                        )
                        continue
                    residual, oracle = erle_value
                    if residual is not None:
                        erle_db, sum_in, sum_out, count = residual
                        extra["erle_db_sum"] = extra.get("erle_db_sum", 0.0) + erle_db
                        extra["erle_count"] = extra.get("erle_count", 0) + 1
                        extra["erle_energy_sum_in"] = (
                            extra.get("erle_energy_sum_in", 0.0) + sum_in
                        )
                        extra["erle_energy_sum_out"] = (
                            extra.get("erle_energy_sum_out", 0.0) + sum_out
                        )
                        extra["erle_energy_count"] = (
                            extra.get("erle_energy_count", 0) + count
                        )
                    if oracle is not None:
                        extra["oracle_erle_db_sum"] = (
                            extra.get("oracle_erle_db_sum", 0.0) + oracle[0]
                        )
                        extra["oracle_erle_count"] = (
                            extra.get("oracle_erle_count", 0) + 1
                        )
            except InterruptedError:
                # 主动停机不是 ERLE 错误：仅强制回收进程池，部分结果在
                # _val_run_extra_metrics 中按全局同步的 shutdown 状态统一丢弃。
                self._shutdown_erle_executor(terminate=True)
            except Exception as exc:
                # submit/wait/result/timeout 均不得绕过最终 collective；异常池必须
                # 立即 terminate/kill，禁止 validation 资源退出时无界等待。
                self._shutdown_erle_executor(terminate=True)
                extra.setdefault("erle_errors", []).append(
                    f"{type(exc).__name__}: {exc}"
                )

        # Accumulate frame-level state-detector counts if available.
        # state_metrics_available 由模型架构静态决定，确保 DDP 各 rank 一致参与 allreduce。
        if not extra["state_metrics_available"]:
            return
        data = ctx["data"]
        state = ctx["state"]
        if "state_probs" not in data:
            return
        state_logits = data["state_probs"]
        if not torch.isfinite(state_logits).all():
            raise ValueError("AEC state_probs contains NaN or Inf")
        if state is not None:
            num_state_classes = extra["num_state_classes"]
            state_labels = state
            counts = compute_state_counts(
                state_logits,
                state_labels,
                ignore_index=-1,
                num_classes=num_state_classes,
            )
            extra["state_tp"] += counts["tp"].to(self.device)
            extra["state_fp"] += counts["fp"].to(self.device)
            extra["state_fn"] += counts["fn"].to(self.device)
            preds = state_logits.argmax(dim=-1)
            valid_mask = state_labels != -1
            extra["state_correct"] += (preds[valid_mask] == state_labels[valid_mask]).long().sum()
            extra["state_valid"] += valid_mask.long().sum()

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
        del extra
        if not self._shared_validation_metrics():
            return total_pairs_for_metrics
        cached = self._materialize_val_waveforms(ctx)
        batch_size = cached["nearend"].shape[0]
        for i in range(batch_size):
            if total_pairs_for_metrics >= self._vm_max_metrics:
                break
            utt_len = int(cached["lengths"][i])
            nearend_np = cached["nearend"][i, :utt_len]
            # Skip near-silent references for SI-SDR/PESQ/ESTOI
            # (noise_only / echo_only / noise_echo_only branches)
            if np.max(np.abs(nearend_np)) < 1e-6:
                continue
            enh_np = cached["estimate"][i, :utt_len]
            pairs.append((nearend_np, enh_np, fs_val))
            total_pairs_for_metrics += 1
        return total_pairs_for_metrics

    def _val_discard_aborted_extra(self, extra: Any) -> None:
        """清空 partial ERLE/state 累计，但保留固定 reduction tensor 形状。"""
        for key in (
            "erle_db_sum",
            "erle_count",
            "erle_eligible_count",
            "erle_energy_sum_in",
            "erle_energy_sum_out",
            "erle_energy_count",
            "oracle_erle_db_sum",
            "oracle_erle_count",
        ):
            extra[key] = 0.0
        extra["erle_errors"] = []
        extra["erle_result"] = None
        extra["erle_metrics"] = {}
        for key in ("state_tp", "state_fp", "state_fn", "state_correct", "state_valid"):
            value = extra.get(key)
            if isinstance(value, torch.Tensor):
                value.zero_()
        extra["state_metrics"] = None

    def _val_post_allreduce_extra(self, extra: Any, val_x: int) -> None:
        # All-reduce state-detector counts across DDP ranks.
        state_tp = extra["state_tp"]
        state_metrics_available = state_tp is not None
        if not state_metrics_available:
            extra["state_metrics"] = None
            return
        state_fp = extra["state_fp"]
        state_fn = extra["state_fn"]
        state_correct = extra["state_correct"]
        state_valid = extra["state_valid"]
        if self.world_size > 1:
            dist.all_reduce(state_tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(state_fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(state_fn, op=dist.ReduceOp.SUM)
            dist.all_reduce(state_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(state_valid, op=dist.ReduceOp.SUM)
        state_metrics = state_metrics_from_counts(state_tp, state_fp, state_fn)
        valid_count = int(state_valid.item())
        # 全局无有效标签时只上报 support，避免把“无数据”伪装成 0 性能。
        if valid_count > 0:
            state_metrics["state_accuracy"] = float(state_correct.item()) / float(
                valid_count
            )
        extra["state_metrics"] = state_metrics

    def _val_log_extra_metrics(self, extra: Any, val_x: int) -> None:
        state_metrics = extra["state_metrics"]
        if self.rank == 0 and state_metrics is not None:
            for name, value in state_metrics.items():
                self._log_scalar(f"val/{name}", value, val_x)
            state_tp = extra["state_tp"]
            num_state_classes = int(state_tp.shape[0]) if state_tp is not None else 0
            support_count = int(state_metrics["state_support_count"])
            if support_count == 0:
                logger.info("  [Val] state metrics unavailable (support=0)")
            else:
                if num_state_classes == len(FST_DT_NST_CLASS_NAMES):
                    class_names = FST_DT_NST_CLASS_NAMES
                elif num_state_classes == len(SCENARIO_CLASS_NAMES):
                    class_names = SCENARIO_CLASS_NAMES
                else:
                    class_names = tuple(f"class_{c}" for c in range(num_state_classes))
                f1_str = " | ".join(
                    f"{class_names[c]}={state_metrics[f'state_f1_class_{c}']:.3f}"
                    for c in range(num_state_classes)
                )
                logger.info(
                    f"  [Val] state accuracy={state_metrics['state_accuracy']:.3f} "
                    f"macro_f1={state_metrics['state_macro_f1']:.3f} "
                    f"supported_macro_f1={state_metrics['state_macro_f1_supported']:.3f} "
                    f"support={support_count} | {f1_str}"
                )
        erle_result = extra.get("erle_result")
        erle_metrics = extra.get("erle_metrics", {})
        if self.rank == 0 and erle_result is not None:
            for name, value in erle_metrics.items():
                if name != "ERLE":
                    self._log_scalar(f"val/{name}", value, val_x)
            erle, count = erle_result
            micro = erle_metrics.get("ERLE_energy_micro")
            oracle = erle_metrics.get("ERLE_oracle")
            logger.info(
                f"  [Val] ERLE={erle:.2f} dB (utterances={int(count)}, "
                f"energy_micro={micro if micro is not None else 'N/A'}, "
                f"oracle={oracle if oracle is not None else 'N/A'})"
            )

    def _val_run_extra_metrics(
        self, epoch: int, extra: Any, do_metrics: bool, val_x: int
    ) -> bool:
        # 与 objective metrics 使用同一个各 rank 一致的 do_metrics 判定；空 rank
        # 也以固定 shape all-reduce 参与，保持 collective 顺序。
        if not do_metrics:
            return False

        # 关闭进程池前后都固定同步 shutdown：等待 ERLE 或关闭 worker 期间只有
        # 单 rank 收到信号时，也必须先传播到所有 rank，再统一清空 partial 结果。
        aborted = bool(self._sync_shutdown_signum())
        self._close_erle_executor(extra)
        aborted = bool(self._sync_shutdown_signum()) or aborted
        if aborted:
            for key in (
                "erle_db_sum",
                "erle_count",
                "erle_eligible_count",
                "erle_energy_sum_in",
                "erle_energy_sum_out",
                "erle_energy_count",
                "oracle_erle_db_sum",
                "oracle_erle_count",
            ):
                extra[key] = 0.0
            extra["erle_errors"] = []
        if "ERLE" in getattr(self, "_vm_metrics", ("ERLE",)):
            extra["erle_result"] = self._compute_and_log_erle(
                epoch,
                extra,
                val_x=val_x,
                log_result=False,
                metrics_aborted=aborted,
            )
        else:
            extra["erle_result"] = None
            extra["erle_metric_result"] = None
        return aborted

    def _val_task_metric_results(
        self, extra: Any
    ) -> dict[str, tuple[float, float, float]]:
        result = extra.get("erle_metric_result")
        return {"ERLE": result} if result is not None else {}

    def _compute_and_log_erle(
        self,
        epoch: int,
        erle_stats: dict[str, Any],
        val_x: Optional[int] = None,
        *,
        log_result: bool = True,
        metrics_aborted: bool = False,
    ) -> tuple[float, float] | None:
        """固定 shape 归约 residual macro、energy-micro 与可选 oracle ERLE。"""
        log_x = val_x if val_x is not None else epoch
        local_errors = erle_stats.get("erle_errors", [])
        local_count = float(erle_stats.get("erle_count", 0))
        # 旧 helper 直调未携带 eligible 时，按已产出结果数兼容；真实 validation
        # 由 _val_accumulate_extra 显式累计实际提交 worker 的 echo utterance。
        local_eligible = float(erle_stats.get("erle_eligible_count", local_count))
        totals = torch.tensor(
            [
                float(erle_stats.get("erle_db_sum", 0.0)),
                local_count,
                local_eligible,
                float(erle_stats.get("erle_energy_sum_in", 0.0)),
                float(erle_stats.get("erle_energy_sum_out", 0.0)),
                float(erle_stats.get("erle_energy_count", 0)),
                float(erle_stats.get("oracle_erle_db_sum", 0.0)),
                float(erle_stats.get("oracle_erle_count", 0)),
                1.0 if local_errors else 0.0,
            ],
            dtype=torch.float64,
            device=self.device,
        )
        if self.world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        (
            db_sum,
            count,
            eligible_count,
            energy_sum_in,
            energy_sum_out,
            energy_count,
            oracle_db_sum,
            oracle_count,
            error_count,
        ) = totals.tolist()
        detail = local_errors[0] if local_errors else "another rank failed"
        if metrics_aborted:
            erle_stats["erle_metric_result"] = None
            erle_stats["erle_metrics"] = {}
            return None
        erle = db_sum / count if count > 0.0 else float("nan")
        if count > 0.0 and not np.isfinite(erle):
            raise RuntimeError(f"ERLE validation returned non-finite mean: {erle!r}")
        valid = validate_validation_metric_runtime_result(
            "ERLE",
            count=count,
            eligible=eligible_count,
            strict=bool(getattr(self, "_vm_strict", False)),
            min_coverage=getattr(self, "_vm_min_coverage", {}).get("ERLE"),
            error=detail if error_count > 0.0 else None,
        )
        erle_stats["erle_metric_result"] = (
            float(erle),
            float(count),
            float(eligible_count),
        )
        if not valid:
            erle_stats["erle_metrics"] = {
                "ERLE_oracle_count": float(oracle_count)
            }
            return None

        metrics = {
            "ERLE": float(erle),
            "ERLE_residual_macro": float(erle),
            "ERLE_residual_count": float(count),
            "ERLE_oracle_count": float(oracle_count),
        }
        denominator = energy_sum_out + 1e-10 * energy_count
        if energy_count > 0.0 and energy_sum_in > 0.0 and denominator > 0.0:
            micro = 10.0 * np.log10(energy_sum_in / denominator)
            if np.isfinite(micro):
                metrics["ERLE_energy_micro"] = float(micro)
        if oracle_count > 0.0:
            oracle_erle = oracle_db_sum / oracle_count
            if np.isfinite(oracle_erle):
                metrics["ERLE_oracle"] = float(oracle_erle)
        erle_stats["erle_metrics"] = metrics

        result = (float(erle), float(count))
        if self.rank == 0 and log_result:
            for name, value in metrics.items():
                self._log_scalar(f"val/{name}", value, log_x)
            logger.info(
                f"  [Val] ERLE={erle:.2f} dB (utterances={int(count)}, "
                f"energy_micro={metrics.get('ERLE_energy_micro', 'N/A')}, "
                f"oracle={metrics.get('ERLE_oracle', 'N/A')})"
            )
        return result
