"""AEC 模型输入能力解析。"""

from __future__ import annotations

from typing import Any


def _unwrap_model(model: Any) -> Any:
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "_orig_mod"):
            current = current._orig_mod
            continue
        break
    return current


def aec_model_accepts_multichannel(model: Any) -> bool:
    """仅允许显式声明 microphone 轴的 AEC 模型接收 ``[B,C,T]``。"""
    raw_model = _unwrap_model(model)
    if getattr(raw_model, "supports_multichannel_input", False) is True:
        return True
    contract = getattr(raw_model, "IO_CONTRACT", None)
    return any(
        field.domain == "spectrum" and "microphone" in field.axes
        for field in getattr(contract, "inputs", ())
    )


def validate_aec_multichannel_contract(model: Any) -> None:
    """校验原生多通道 AEC 的完整谱域合同。

    ``supports_multichannel_input`` 只是能力声明；还必须有可审计的
    ``IO_CONTRACT``，其中 microphone 输入保留 microphone 轴，far-end 和
    输出保持 mono。这样错误声明会在 build 阶段失败，而不是延迟到 forward。
    """
    raw_model = _unwrap_model(model)
    if getattr(raw_model, "supports_multichannel_input", False) is not True:
        raise ValueError(
            "spatial AEC requires model.supports_multichannel_input=True; "
            "the configured model is mono-only"
        )
    contract = getattr(raw_model, "IO_CONTRACT", None)
    if contract is None:
        raise ValueError("native multichannel AEC requires an explicit model.IO_CONTRACT")
    inputs = {field.name: field for field in contract.inputs}
    outputs = {field.name: field for field in contract.outputs}
    mic = inputs.get("mic_spectra")
    far = inputs.get("farend_spectra")
    if mic is None or "microphone" not in mic.axes:
        raise ValueError("native multichannel AEC mic_spectra must include a microphone axis")
    if far is None or "microphone" in far.axes:
        raise ValueError("native multichannel AEC farend_spectra must remain mono")
    estimate = outputs.get("nearend_estimate_spectra")
    if estimate is None or "microphone" in estimate.axes:
        raise ValueError("native multichannel AEC output must be mono nearend estimate")


__all__ = ["aec_model_accepts_multichannel", "validate_aec_multichannel_contract"]
