#!/usr/bin/env python3
"""TSE 离线 PT / 流式 PT / ORT 三路验证与正式发布门禁。"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.utils.artifact_identity import bind_source_files
from onnx.utils import export_onnx
from onnx.verify.manifest import runtime_contract_from_spec
from onnx.verify.verify_common import VerifiedOnnxArtifact, owned_onnx_cleanup
from tasks.tse.inference.artifacts import TSEInferenceArtifactSnapshot


def _comparison(name: str, left: torch.Tensor, right: torch.Tensor, threshold: float):
    from onnx.verify.verify_common import ThreeWayComparison

    difference = left - right
    mse = float(torch.mean(difference.square()).item())
    max_abs = float(torch.max(torch.abs(difference)).item())
    count = int(left.numel()) if left.shape == right.shape else 0
    status = "PASS" if count > 0 and np.isfinite(mse) and mse < threshold else "FAIL"
    return ThreeWayComparison(name, mse, max_abs, count, threshold, status)


def _waveform_from_frontend(frontend, waveform: np.ndarray, chunk_samples: int) -> np.ndarray:
    values = []
    for start in range(0, waveform.size, chunk_samples):
        part = frontend.push(waveform[start : start + chunk_samples])
        if part.size:
            values.append(part)
    tail = frontend.flush()
    if tail.size:
        values.append(tail)
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def _build_configured_wrapper_with_identity(
    config_path: Path, checkpoint: Path, enrollment_samples: int
):
    """用 TSEExtractor 的实际 loader 构造 wrapper 并冻结 source identity。"""
    from onnx.models.streaming_tse import StreamingTSEFlat
    from tasks.tse.inference.extractor import TSEExtractor

    extractor = TSEExtractor(
        str(checkpoint),
        config_path=str(config_path),
        device="cpu",
        output_peak=None,
    )
    snapshot = TSEInferenceArtifactSnapshot.from_loaded_extractor(
        extractor,
        {
            "device": "cpu",
            "output_peak": None,
            "enrollment_samples": int(enrollment_samples),
        },
    )
    if type(extractor.model).__name__ != "TSE_BSRNN_SPK":
        raise ValueError(
            "TSE streaming runtime requires TSE_BSRNN_SPK, got "
            f"{type(extractor.model).__name__}"
        )
    wrapper = StreamingTSEFlat(
        extractor.model, enrollment_samples=enrollment_samples
    ).eval()
    return wrapper, snapshot


def _tse_onnx_source_binding(
    snapshot: TSEInferenceArtifactSnapshot,
) -> tuple[dict[str, Any], dict[str, str]]:
    """返回以 loader 身份冻结的 ONNX source binding 与路径视图。"""
    snapshot.verify_global_identity()
    identities = snapshot.source_file_identities()
    source_files = {name: identity["path"] for name, identity in identities.items()}
    binding = bind_source_files(
        source_files,
        parameters={"model_id": "tse"},
        precomputed=identities,
    )
    return binding, source_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TSE offline PT / streaming PT / ORT three-way verification"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--enrollment-samples", type=int, default=48000)
    parser.add_argument("--chunk-samples", type=int, default=777)
    parser.add_argument("--mse-threshold", type=float, default=1e-4)
    parser.add_argument("--waveform-mse-threshold", type=float, default=1e-4)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--save-onnx", type=Path, default=None)
    args = parser.parse_args()
    if args.frames < 4:
        parser.error("--frames must be at least 4 for centered waveform verification")
    if args.enrollment_samples < 1 or args.chunk_samples < 1:
        parser.error("enrollment/chunk sample counts must be positive")

    from core.stft import StftIstftModule
    from core.utils.config import load_merged_config
    from onnx.export import write_streaming_runtime_metadata
    from onnx.infer.infer_streaming_tse_ort import (
        TSEStreamingORTProcessor,
        _build_condition_state,
    )
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner
    from onnx.runtime_spec import runtime_spec_from_wrapper
    from onnx.verify.verify_common import (
        ThreeWayResult,
        require_release_preflight,
        require_strict_runtime_spec,
    )
    from tasks.tse.inference.streaming import (
        CenteredStreamingSTFTOLA,
        build_streaming_enrollment,
    )

    require_release_preflight(
        save_onnx=args.save_onnx,
        checkpoint=args.checkpoint,
        config=args.config,
        require_config=True,
        thresholds={
            "mse_threshold": (args.mse_threshold, 1e-4),
            "waveform_mse_threshold": (args.waveform_mse_threshold, 1e-4),
        },
    )
    if args.checkpoint is None:
        parser.error("--checkpoint is required for TSE three-way verification")

    torch.manual_seed(42)
    flat, tse_snapshot = _build_configured_wrapper_with_identity(
        args.config, args.checkpoint, args.enrollment_samples
    )
    base = flat.model
    config = load_merged_config(args.config)
    stft_cfg = config.get("stft") or {}
    stft = StftIstftModule(
        n_fft=int(stft_cfg["n_fft"]),
        hop_length=int(stft_cfg["hop_length"]),
        win_length=int(stft_cfg.get("win_length", stft_cfg["n_fft"])),
        window=str(stft_cfg.get("window", "hann")),
        center=bool(stft_cfg.get("center", True)),
        normalized=bool(stft_cfg.get("normalized", False)),
    )
    waveform_length = (args.frames - 1) * flat.hop_length
    mixture = torch.randn(1, waveform_length) * 0.03
    enrollment_wav = torch.randn(1, args.enrollment_samples) * 0.03
    embedding = None
    if not base.joint_training and base.features_config["spkemb"]["enabled"]:
        embedding = torch.randn(1, base.features_config["spkemb"]["embed_dim"])
    enrollment = build_streaming_enrollment(
        base, enrollment_wav, spk_embedding=embedding
    )
    mixture_spec = stft(mixture)
    if mixture_spec.shape[2] != args.frames:
        raise RuntimeError("verifier waveform/STFT frame geometry drifted")

    with torch.inference_mode():
        offline_inputs = {"mix_spectra": mixture_spec, **enrollment}
        offline = base(offline_inputs)["target_estimate_spectra"]
        cache_pt = flat.init_cache_flat_from_enrollment(enrollment)
        pt_frames = []
        for index in range(args.frames):
            values = flat(mixture_spec[:, :, index : index + 1], *cache_pt)
            pt_frames.append(values[0])
            cache_pt = list(values[1:])
        streaming_pt = torch.cat(pt_frames, dim=2)

    with owned_onnx_cleanup() as lifecycle:
        fd, raw_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        onnx_path = lifecycle.own(raw_path)
        export_cache = flat.init_cache_flat(1, "cpu", torch.float32)
        export_onnx(
            flat,
            (mixture_spec[:, :, :1], *export_cache),
            str(onnx_path),
            input_names=flat.get_input_names(),
            output_names=flat.get_output_names(),
            dynamic_axes=flat.get_dynamic_axes(),
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
        write_streaming_runtime_metadata(str(onnx_path), flat)
        artifact = VerifiedOnnxArtifact(onnx_path)
        session = artifact.create_session()
        artifact_spec = require_strict_runtime_spec(session, "tse")
        wrapper_spec = runtime_spec_from_wrapper(flat, "tse")
        contract_match = (
            artifact_spec.input_names == wrapper_spec.input_names
            and artifact_spec.output_names == wrapper_spec.output_names
            and artifact_spec.stft == wrapper_spec.stft
            and artifact_spec.tse == wrapper_spec.tse
        )
        if not contract_match:
            raise RuntimeError("TSE exported graph/runtime metadata drifted from wrapper")

        condition = _build_condition_state(flat, enrollment_wav, embedding)
        processor = TSEStreamingORTProcessor(session, condition)
        ort_frames = []
        for index in range(args.frames):
            ort_frames.append(processor(mixture_spec[:, :, index : index + 1]))
        streaming_ort = torch.cat(ort_frames, dim=2)

        comparisons = ThreeWayResult(
            _comparison(
                "offline PT vs streaming PT", offline, streaming_pt, args.mse_threshold
            ),
            _comparison(
                "streaming PT vs ORT", streaming_pt, streaming_ort, args.mse_threshold
            ),
            _comparison(
                "offline PT vs ORT", offline, streaming_ort, args.mse_threshold
            ),
        )

        pt_cache = flat.init_cache_flat_from_enrollment(enrollment)

        def pt_process(frame):
            nonlocal pt_cache
            values = flat(frame, *pt_cache)
            pt_cache = list(values[1:])
            return values[0]

        pt_frontend = CenteredStreamingSTFTOLA(
            pt_process,
            n_fft=flat.n_fft,
            hop_length=flat.hop_length,
            win_length=flat.win_length,
        )
        processor.reset(condition)
        ort_frontend = CenteredStreamingSTFTOLA(
            processor,
            n_fft=flat.n_fft,
            hop_length=flat.hop_length,
            win_length=flat.win_length,
        )
        offline_wave = stft.inverse(offline, waveform_length)[0]
        mixture_np = mixture[0].numpy()
        pt_wave = torch.from_numpy(
            _waveform_from_frontend(pt_frontend, mixture_np, args.chunk_samples)
        )
        ort_wave = torch.from_numpy(
            _waveform_from_frontend(ort_frontend, mixture_np, args.chunk_samples)
        )
        waveform_pt = _comparison(
            "offline waveform vs streaming PT waveform",
            offline_wave,
            pt_wave,
            args.waveform_mse_threshold,
        )
        waveform_ort = _comparison(
            "offline waveform vs ORT waveform",
            offline_wave,
            ort_wave,
            args.waveform_mse_threshold,
        )
        all_pass = (
            comparisons.all_pass
            and waveform_pt.passed
            and waveform_ort.passed
            and contract_match
        )
        for item in (*comparisons.comparisons, waveform_pt, waveform_ort):
            print(
                f"{item.name}: MSE={item.mse:.6e}, max_abs={item.max_abs:.6e} "
                f"[{item.status}]"
            )
        print(f"Runtime contract: {'PASS' if contract_match else 'FAIL'}")
        print(f"Summary: {'ALL PASS' if all_pass else 'SOME FAILED'}")
        if args.save_onnx:
            source_binding, source_files = _tse_onnx_source_binding(tse_snapshot)
            published = artifact.publish(
                args.save_onnx, all_pass=all_pass,
                manifest={
                    "model_id": "tse",
                    "source_files": source_files,
                    "source_binding": source_binding,
                    "runtime_contract": runtime_contract_from_spec(artifact_spec),
                    "verification": {
                        "statuses": [item.status for item in comparisons.comparisons],
                        "tse_artifact_snapshot": tse_snapshot.meta,
                    },
                },
                require_manifest=True,
            )
            if published:
                print(
                    f"ONNX artifact kept: {published['path']} "
                    f"sha256={published['sha256']}"
                )
        artifact.close()
        return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
