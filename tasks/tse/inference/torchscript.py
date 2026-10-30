"""TSE streaming flat ABI 的 TorchScript 发布工具。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from core.utils.file_transaction import publish_staged_files
from core.utils.inference_artifacts import fingerprint_file
from onnx.runtime_spec import runtime_spec_from_wrapper


def _compare_outputs(
    expected: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...],
) -> float:
    if len(expected) != len(actual):
        raise RuntimeError(
            f"TorchScript output count mismatch: {len(actual)} != {len(expected)}"
        )
    max_abs = 0.0
    for index, (reference, candidate) in enumerate(zip(expected, actual)):
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            raise RuntimeError(
                f"TorchScript output[{index}] shape/dtype mismatch: "
                f"{tuple(candidate.shape)}/{candidate.dtype} != "
                f"{tuple(reference.shape)}/{reference.dtype}"
            )
        if not bool(torch.isfinite(candidate).all().item()):
            raise RuntimeError(f"TorchScript output[{index}] contains NaN/Inf")
        torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=2e-6)
        if candidate.numel():
            max_abs = max(
                max_abs,
                float(torch.max(torch.abs(candidate - reference)).item()),
            )
    return max_abs


def export_streaming_tse_torchscript(
    flat_model: torch.nn.Module,
    output_path: str | Path,
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    batch_size: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    """trace、重载验证并原子发布 TSE TorchScript 与 ABI sidecar。"""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    for name in (
        "init_cache_flat",
        "get_input_names",
        "get_output_names",
        "get_dynamic_axes",
    ):
        if not callable(getattr(flat_model, name, None)):
            raise TypeError(f"flat_model must provide {name}()")

    output = Path(output_path).expanduser()
    sidecar = output.with_name(f"{output.name}.meta.json")
    if not overwrite:
        existing = [str(path) for path in (output, sidecar) if path.exists() or path.is_symlink()]
        if existing:
            raise FileExistsError(
                "TorchScript output already exists; pass overwrite=True: "
                + ", ".join(existing)
            )
    output.parent.mkdir(parents=True, exist_ok=True)

    config_before = fingerprint_file(config_path)
    checkpoint_before = fingerprint_file(checkpoint_path)
    flat_model = flat_model.cpu().eval()
    frame = torch.zeros(
        batch_size,
        int(flat_model.n_fft) // 2 + 1,
        1,
        2,
        dtype=torch.float32,
    )
    cache = flat_model.init_cache_flat(batch_size, "cpu", torch.float32)
    example_inputs = (frame, *cache)
    with torch.inference_mode():
        expected = tuple(flat_model(*example_inputs))
        traced = torch.jit.trace(
            flat_model,
            example_inputs,
            strict=True,
            check_trace=True,
        )
        traced_outputs = tuple(traced(*example_inputs))
    trace_max_abs = _compare_outputs(expected, traced_outputs)

    runtime_spec = runtime_spec_from_wrapper(flat_model, "tse")
    if runtime_spec.tse is None:
        raise RuntimeError("TSE TorchScript export requires TSERuntimeSpec")
    metadata: dict[str, Any] = {
        "schema": "tse-streaming-torchscript-v1",
        "runtime_kind": "positional-flat-cache",
        "trace_batch_size": batch_size,
        "input_names": list(runtime_spec.input_names),
        "output_names": list(runtime_spec.output_names),
        "dynamic_axes": {
            name: {str(axis): symbol for axis, symbol in axes.items()}
            for name, axes in runtime_spec.dynamic_axes.items()
        },
        "stft": runtime_spec.stft.to_metadata(),
        "tse": runtime_spec.tse.to_metadata(),
        "config": config_before,
        "checkpoint": checkpoint_before,
        "trace_max_abs_diff": trace_max_abs,
    }

    staged_model: Path | None = None
    staged_sidecar: Path | None = None
    try:
        descriptor, staged_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        staged_model = Path(staged_name)
        traced.save(str(staged_model))
        with staged_model.open("rb") as handle:
            os.fsync(handle.fileno())

        reloaded = torch.jit.load(str(staged_model), map_location="cpu").eval()
        with torch.inference_mode():
            loaded_outputs = tuple(reloaded(*example_inputs))
        metadata["reload_max_abs_diff"] = _compare_outputs(
            expected, loaded_outputs
        )

        descriptor, staged_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
        )
        staged_sidecar = Path(staged_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if fingerprint_file(config_path) != config_before:
            raise RuntimeError("TSE TorchScript config changed during export")
        if fingerprint_file(checkpoint_path) != checkpoint_before:
            raise RuntimeError("TSE TorchScript checkpoint changed during export")
        publish_staged_files(
            (staged_model, staged_sidecar),
            (output, sidecar),
        )
        staged_model = None
        staged_sidecar = None
    finally:
        for staged in (staged_model, staged_sidecar):
            if staged is not None:
                try:
                    staged.unlink()
                except FileNotFoundError:
                    pass
    return metadata


__all__ = ["export_streaming_tse_torchscript"]
