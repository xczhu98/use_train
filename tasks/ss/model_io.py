"""统一 waveform 与 external-STFT separator 的深 Model Runtime Module。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch

from core.data.dataset_contract import DatasetContract
from core.losses.loss_contract import LossContract, resolve_loss_contract
from core.models.execution_plan import ResolvedExecutionPlan, resolve_execution_plan
from core.models.legacy_contract import resolve_model_io_contract


@dataclass(frozen=True)
class SSModelResult:
    """Trainer、inference 和 verifier 共用的 canonical source 结果。"""

    sources: torch.Tensor
    source_spectra: torch.Tensor | None
    diagnostics: Mapping[str, Any]


def resolve_ss_execution_plan(
    config: dict,
    model,
    *,
    dataset_contract: DatasetContract | None = None,
    loss=None,
) -> ResolvedExecutionPlan:
    raw_model = model
    seen: set[int] = set()
    while id(raw_model) not in seen:
        seen.add(id(raw_model))
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module
            continue
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
            continue
        break
    model_sample_rate = getattr(raw_model, "sample_rate", None)
    loss_contract = resolve_loss_contract(loss) if loss is not None else None
    return resolve_execution_plan(
        config,
        model_contract=resolve_model_io_contract(model, config),
        dataset_contract=dataset_contract,
        loss_contract=loss_contract,
        model_sample_rate=model_sample_rate,
        require_loss_contract=loss is not None,
    )


class SSModelRuntime:
    """隐藏模型输入域、source spectra 解码和严格输出校验。"""

    def __init__(self, model, *, plan: ResolvedExecutionPlan, stft_module=None) -> None:
        if not isinstance(plan, ResolvedExecutionPlan) or plan.task != "ss":
            raise ValueError("SSModelRuntime requires an SS ResolvedExecutionPlan")
        if plan.input_domain == "spectrum" and stft_module is None:
            raise ValueError("spectrum-domain SS execution plan requires stft_module")
        if plan.stft is not None and stft_module is not None:
            for field, attr in (
                ("n_fft", "n_fft"),
                ("hop_length", "hop_length"),
                ("win_length", "win_length"),
                ("center", "center"),
            ):
                if getattr(stft_module, attr) != plan.stft[field]:
                    raise ValueError(f"stft_module {field} does not match execution plan")
        self.model = model
        self.plan = plan
        self.stft_module = stft_module

    def separate(
        self,
        mixture: torch.Tensor,
        lengths: torch.Tensor,
        sample_rate: int | None,
    ) -> SSModelResult:
        if not isinstance(mixture, torch.Tensor) or mixture.ndim not in (2, 3):
            raise ValueError("mixture must have shape [B, T] or [B, C, T]")
        if mixture.ndim == 3:
            if self.plan.input_domain != "spectrum":
                raise ValueError("multi-channel SS runtime requires a spectrum-domain model")
            accepts_microphones = any(
                field.domain == "spectrum" and "microphone" in field.axes
                for field in self.plan.input_fields
            )
            if not accepts_microphones:
                raise ValueError(
                    "multi-channel SS runtime requires the model contract to declare "
                    "a microphone axis"
                )
        if not isinstance(lengths, torch.Tensor) or lengths.shape != (mixture.shape[0],):
            raise ValueError("lengths must have shape [B]")
        if lengths.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("lengths must use an integer dtype")
        if bool(((lengths < 1) | (lengths > mixture.shape[-1])).any()):
            raise ValueError("lengths must be within the mixture time axis")
        effective_sample_rate = self.plan.sample_rate if sample_rate is None else sample_rate
        if type(effective_sample_rate) is not int or effective_sample_rate <= 0:
            raise ValueError(
                "sample_rate must be provided or declared by the SS model"
            )
        if effective_sample_rate != self.plan.sample_rate:
            raise ValueError(
                "runtime sample_rate does not match SS execution plan: "
                f"{effective_sample_rate} != {self.plan.sample_rate}"
            )

        if self.plan.input_domain == "waveform":
            model_input = mixture
        elif mixture.ndim == 2:
            model_input = self.stft_module(mixture)
        else:
            batch, channels, samples = mixture.shape
            channel_specs = self.stft_module(mixture.reshape(batch * channels, samples))
            model_input = channel_specs.reshape(batch, channels, *channel_specs.shape[1:])
        output = self.model(model_input, lengths, fs=effective_sample_rate)
        if not isinstance(output, dict):
            raise TypeError("SS model output must be a mapping")
        source_spectra = output.get("est_sources_spectra")
        sources = output.get("est_sources")
        if source_spectra is not None:
            if not isinstance(source_spectra, torch.Tensor):
                raise TypeError("est_sources_spectra must be a tensor when provided")
            if (
                source_spectra.ndim != 5
                or source_spectra.shape[-1] != 2
                or source_spectra.shape[:2]
                != (mixture.shape[0], self.plan.num_sources)
            ):
                raise ValueError(
                    "est_sources_spectra must have shape [B, K, F, T, 2] "
                    "matching the execution plan"
                )
            if not bool(torch.isfinite(source_spectra).all()):
                raise ValueError("est_sources_spectra must contain only finite values")
        if sources is None and source_spectra is not None:
            if self.stft_module is None:
                raise RuntimeError("source spectra decoding requires stft_module")
            batch, source_count = source_spectra.shape[:2]
            flat = source_spectra.reshape(
                batch * source_count, *source_spectra.shape[2:]
            )
            sources = self.stft_module.inverse(flat, mixture.shape[-1]).reshape(
                batch, source_count, -1
            )
        if not isinstance(sources, torch.Tensor) or sources.ndim != 3:
            raise ValueError("SS model must provide sources with shape [B, K, T]")
        if sources.shape[:2] != (mixture.shape[0], self.plan.num_sources):
            raise ValueError(
                f"SS model output must have B={mixture.shape[0]}, K={self.plan.num_sources}; "
                f"got {tuple(sources.shape)}"
            )
        if sources.shape[-1] != mixture.shape[-1]:
            raise ValueError(
                "SS model output time axis must exactly match the mixture; "
                f"got {sources.shape[-1]} and {mixture.shape[-1]}"
            )
        if not bool(torch.isfinite(sources).all()):
            raise ValueError("est_sources must contain only finite values")
        diagnostics = {
            key: value
            for key, value in output.items()
            if key not in {"est_sources", "est_sources_spectra"}
        }
        return SSModelResult(
            sources=sources,
            source_spectra=source_spectra,
            diagnostics=MappingProxyType(diagnostics),
        )
