"""启动期把 Dataset/Model/Loss 合同解析为不可变执行计划。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from core.data.constants import normalize_data_mode
from core.data.dataset_contract import DatasetContract
from core.losses.loss_contract import LossContract
from core.models.io_contract import ModelIOContract, TensorField
from core.stft import resolve_stft_config


_DIRECT_TRAINER_DEFAULT_MODES = {
    "se": "offline",
    "aec": "offline_aec",
    "tse": "offline_tse",
}


@dataclass(frozen=True)
class ResolvedExecutionPlan:
    task: str
    input_fields: tuple[TensorField, ...]
    output_fields: tuple[TensorField, ...]
    input_domain: str
    output_domain: str
    forward_kind: str
    frontend_owner: str
    num_sources: int | None
    sample_rate: int
    stft: Mapping[str, object] | None
    dataset_contract: DatasetContract | None = None


def resolve_trainer_execution_plan(
    config: dict,
    model,
    *,
    task: str,
    sample_rate: int,
    dataset_contract: DatasetContract | None = None,
    stft_module=None,
) -> ResolvedExecutionPlan:
    """为直接构造 Trainer 的兼容路径生成只读 contract 配置视图。

    正式 YAML 仍由 Entry 强制显式声明 ``data.mode``；旧测试和嵌入式调用不经
    Entry 时，Trainer 已知的 task/sample_rate 只用于补齐执行计划解析上下文。
    """
    if task not in _DIRECT_TRAINER_DEFAULT_MODES:
        raise ValueError(f"unsupported trainer execution task: {task!r}")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("trainer sample_rate must be an exact positive int")
    data = config.get("data") or {}
    if not isinstance(data, dict):
        raise TypeError("config.data must be a mapping")
    plan_config = config
    if data.get("mode") is None or data.get("sample_rate") is None:
        plan_config = dict(config)
        plan_data = dict(data)
        plan_data.setdefault("mode", _DIRECT_TRAINER_DEFAULT_MODES[task])
        plan_data.setdefault("sample_rate", sample_rate)
        plan_config["data"] = plan_data
    from core.models.legacy_contract import resolve_model_io_contract

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
    model_contract = resolve_model_io_contract(raw_model, plan_config)
    needs_runtime_stft = (
        model_contract.frontend_owner == "runtime"
        and any(
            field.domain == "spectrum"
            for field in (*model_contract.inputs, *model_contract.outputs)
        )
    )
    if needs_runtime_stft and not plan_config.get("stft") and stft_module is not None:
        plan_config = dict(plan_config)
        plan_config["stft"] = {
            "n_fft": int(stft_module.n_fft),
            "hop_length": int(stft_module.hop_length),
            "win_length": int(stft_module.win_length),
            "window": getattr(stft_module, "window_name", "hann"),
            "center": bool(getattr(stft_module, "center", True)),
        }
    model_sample_rate = getattr(
        raw_model,
        "sample_rate",
        getattr(raw_model, "target_fs", None),
    )
    return resolve_execution_plan(
        plan_config,
        model_contract=model_contract,
        dataset_contract=dataset_contract,
        model_sample_rate=model_sample_rate,
    )


def resolve_execution_plan(
    config: dict,
    *,
    model_contract: ModelIOContract,
    dataset_contract: DatasetContract | None = None,
    loss_contract: LossContract | None = None,
    model_sample_rate: int | None = None,
    require_loss_contract: bool = False,
) -> ResolvedExecutionPlan:
    if type(require_loss_contract) is not bool:
        raise TypeError("require_loss_contract must be bool")
    data = config.get("data") or {}
    normalized = normalize_data_mode(data.get("mode"))
    task = normalized.task
    if task not in model_contract.tasks:
        raise ValueError(
            f"model contract tasks {sorted(model_contract.tasks)} do not support task {task!r}"
        )
    sample_rate = data.get("sample_rate")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("data.sample_rate must be an exact positive int")
    if model_sample_rate is not None:
        if type(model_sample_rate) is not int or model_sample_rate <= 0:
            raise ValueError("model sample_rate must be an exact positive int")
        if model_sample_rate != sample_rate:
            raise ValueError(
                "model sample_rate does not match data.sample_rate: "
                f"{model_sample_rate} != {sample_rate}"
            )
    num_sources = None
    if task == "ss":
        ss_config = config.get("ss")
        if not isinstance(ss_config, dict):
            raise ValueError("SS execution plan requires a top-level ss mapping")
        num_sources = ss_config.get("num_sources")
        if type(num_sources) is not int or num_sources < 1:
            raise ValueError("ss.num_sources must be an exact positive int")
    if dataset_contract is not None:
        if dataset_contract.task != task:
            raise ValueError("DatasetContract task does not match data.mode")
        if dataset_contract.sample_rate not in {None, sample_rate}:
            raise ValueError("DatasetContract sample_rate does not match config")
        if task == "ss" and dataset_contract.num_sources is None:
            raise ValueError("SS DatasetContract must declare num_sources")
        if (
            num_sources is not None
            and dataset_contract.num_sources is not None
            and dataset_contract.num_sources != num_sources
        ):
            raise ValueError("DatasetContract num_sources does not match ss.num_sources")

    domains = {
        field.domain for field in (*model_contract.inputs, *model_contract.outputs)
    }
    stft = None
    if "spectrum" in domains and model_contract.frontend_owner == "runtime":
        if not config.get("stft"):
            raise ValueError("runtime-owned spectrum model requires top-level STFT")
        stft = MappingProxyType(resolve_stft_config(config["stft"]))
    if task == "ss" and require_loss_contract and loss_contract is None:
        raise ValueError("SS training execution plan requires an explicit LossContract")
    if loss_contract is not None and task == "ss":
        missing_capabilities = [
            name
            for name, enabled in (
                ("permutation_invariant", loss_contract.permutation_invariant),
                ("supports_lengths", loss_contract.supports_lengths),
                ("supports_source_mask", loss_contract.supports_source_mask),
            )
            if not enabled
        ]
        if missing_capabilities:
            raise ValueError(
                "SS loss contract is missing required PIT capabilities: "
                f"{missing_capabilities}"
            )
        dataset_names = (
            {field.name for field in dataset_contract.fields}
            if dataset_contract is not None
            else set()
        )
        derived_names = {field.name for field in model_contract.outputs}
        if "est_sources_spectra" in derived_names and stft is not None:
            derived_names.add("est_sources")
        available = dataset_names | derived_names
        missing = set(loss_contract.inputs) - available
        if missing:
            raise ValueError(f"LossContract inputs are unavailable: {sorted(missing)}")
    return ResolvedExecutionPlan(
        task=task,
        input_fields=model_contract.inputs,
        output_fields=model_contract.outputs,
        input_domain=model_contract.inputs[0].domain,
        output_domain=model_contract.outputs[0].domain,
        forward_kind=model_contract.forward_kind,
        frontend_owner=model_contract.frontend_owner,
        num_sources=num_sources,
        sample_rate=sample_rate,
        stft=stft,
        dataset_contract=dataset_contract,
    )
