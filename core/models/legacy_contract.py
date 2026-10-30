"""未声明 IO_CONTRACT 的模型集中式 Legacy Adapter。"""

from __future__ import annotations

from core.data.constants import normalize_data_mode
from core.models.io_contract import ModelIOContract, TensorField


def _unwrap_model(model):
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


def resolve_model_io_contract(model, config: dict) -> ModelIOContract:
    model = _unwrap_model(model)
    explicit = getattr(model, "IO_CONTRACT", None)
    if explicit is not None:
        if not isinstance(explicit, ModelIOContract):
            raise TypeError("model.IO_CONTRACT must be a ModelIOContract")
        return explicit

    task = normalize_data_mode((config.get("data") or {}).get("mode")).task
    input_domain = getattr(model, "input_domain", None)
    output_domain = getattr(model, "output_domain", None)
    if task == "ss" and input_domain in {"waveform", "spectrum"}:
        output_domain = output_domain or input_domain
        if output_domain not in {"waveform", "spectrum"}:
            raise ValueError(f"unsupported SS output_domain: {output_domain!r}")
        input_name = "mixture" if input_domain == "waveform" else "mixture_spectra"
        output_name = "est_sources" if output_domain == "waveform" else "est_sources_spectra"
        input_axes = (
            ("batch", "time")
            if input_domain == "waveform"
            else ("batch", "frequency", "frame", "complex")
        )
        output_axes = (
            ("batch", "source", "time")
            if output_domain == "waveform"
            else ("batch", "source", "frequency", "frame", "complex")
        )
        return ModelIOContract(
            tasks=frozenset({"ss"}),
            inputs=(TensorField(input_name, input_domain, input_axes),),
            outputs=(TensorField(output_name, output_domain, output_axes, multiple=True),),
            forward_kind=getattr(model, "forward_kind", "estimate"),
            frontend_owner=getattr(
                model,
                "frontend_owner",
                "model" if input_domain == "waveform" else "runtime",
            ),
        )
    if task == "se":
        domain = input_domain or ("spectrum" if config.get("stft") else "waveform")
        if domain not in {"waveform", "spectrum"}:
            raise ValueError(f"unsupported SE input_domain: {domain!r}")
        resolved_output_domain = output_domain or domain
        if resolved_output_domain not in {"waveform", "spectrum", "loss"}:
            raise ValueError(
                f"unsupported SE output_domain: {resolved_output_domain!r}"
            )
        suffix = "_spectra" if domain == "spectrum" else ""
        input_axes = (
            ("batch", "frequency", "frame", "complex")
            if domain == "spectrum"
            else ("batch", "time")
        )
        output_axes = (
            ("scalar",)
            if resolved_output_domain == "loss"
            else ("batch", "frequency", "frame", "complex")
            if resolved_output_domain == "spectrum"
            else ("batch", "time")
        )
        output_name = (
            "loss"
            if resolved_output_domain == "loss"
            else "enhanced_spectra"
            if resolved_output_domain == "spectrum"
            else "enhanced"
        )
        return ModelIOContract(
            tasks=frozenset({"se"}),
            inputs=(TensorField(f"noisy{suffix}", domain, input_axes),),
            outputs=(TensorField(output_name, resolved_output_domain, output_axes),),
            forward_kind=getattr(model, "forward_kind", "estimate"),
            frontend_owner=getattr(
                model,
                "frontend_owner",
                "runtime" if domain == "spectrum" else "model",
            ),
        )
    if task == "aec":
        spectrum_axes = ("batch", "frequency", "frame", "complex")
        capabilities = getattr(
            model,
            "aec_input_capabilities",
            frozenset({"mic_spectra", "farend_spectra"}),
        )
        if not isinstance(capabilities, (set, frozenset)):
            raise TypeError("model.aec_input_capabilities must be a set-like value")
        fields = []
        for name in sorted(capabilities):
            if name not in {"mic_spectra", "farend_spectra", "error_spectra"}:
                raise ValueError(f"unsupported AEC input capability: {name!r}")
            fields.append(
                TensorField(
                    name,
                    "spectrum",
                    spectrum_axes,
                    required=name != "error_spectra",
                )
            )
        return ModelIOContract(
            tasks=frozenset({"aec"}),
            inputs=tuple(fields),
            outputs=(TensorField("nearend_estimate_spectra", "spectrum", spectrum_axes),),
            forward_kind="estimate",
            frontend_owner="runtime",
        )
    if task != "tse":
        raise ValueError(
            f"no IO contract for task {task!r}: the model must declare IO_CONTRACT "
            "or be a registered legacy se/aec/ss/tse model"
        )
    spectrum_axes = ("batch", "frequency", "frame", "complex")
    return ModelIOContract(
        tasks=frozenset({"tse"}),
        inputs=(
            TensorField("mix_spectra", "spectrum", spectrum_axes),
            TensorField("enroll_spectra", "spectrum", spectrum_axes, required=False),
            TensorField("enroll_wav", "waveform", ("batch", "time"), required=False),
            TensorField("spk_embedding", "embedding", ("batch", "embedding"), required=False),
        ),
        outputs=(TensorField("target_estimate_spectra", "spectrum", spectrum_axes),),
        forward_kind="estimate",
        frontend_owner="runtime",
    )
