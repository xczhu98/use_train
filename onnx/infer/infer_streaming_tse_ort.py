#!/usr/bin/env python3
"""TSE streaming ONNX Runtime 推理，condition 仅在 session 初始化时计算。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class TSEStreamingORTProcessor:
    """绑定固定 enrollment condition 的 ORT 单帧处理器。"""

    def __init__(self, session, condition_state: Mapping[str, object]) -> None:
        from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

        self.runner = StreamingORTFrameRunner(session, "tse", batch_size=1)
        if self.runner.spec.tse is None:
            raise RuntimeError("TSE artifact is missing versioned runtime metadata")
        self._condition_names = self.runner.spec.tse.condition_names
        self.reset(condition_state)

    @property
    def spec(self):
        return self.runner.spec

    def reset(self, condition_state: Mapping[str, object]) -> None:
        if tuple(condition_state) != self._condition_names:
            raise ValueError(
                f"TSE condition names/order mismatch: expected {self._condition_names}, "
                f"got {tuple(condition_state)}"
            )
        self.runner.reset()
        state = dict(self.runner.state)
        for name in self._condition_names:
            value = condition_state[name]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            state[name] = np.ascontiguousarray(value, dtype=np.float32)
        self.runner.reset(state)

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        value, _ = self.runner.run_frame(frame)
        return torch.from_numpy(value)


def _build_condition_state(flat, enrollment_waveform: torch.Tensor, spk_embedding=None):
    from tasks.tse.inference.streaming import build_streaming_enrollment

    enrollment = build_streaming_enrollment(
        flat.model, enrollment_waveform, spk_embedding=spk_embedding
    )
    cache = flat.init_cache_flat_from_enrollment(enrollment)
    values = dict(zip(flat.CACHE_KEYS, cache))
    return {name: values[name] for name in flat.condition_names}


def _build_configured_wrapper(config_path: Path, checkpoint: Path, enrollment_samples: int):
    from onnx.models.streaming_tse import StreamingTSEFlat
    from tasks.tse.inference.extractor import TSEExtractor

    extractor = TSEExtractor(
        str(checkpoint),
        config_path=str(config_path),
        device="cpu",
        output_peak=None,
    )
    model = extractor.model.eval()
    if type(model).__name__ != "TSE_BSRNN_SPK":
        raise ValueError(
            "TSE streaming runtime requires TSE_BSRNN_SPK, got "
            f"{type(model).__name__}"
        )
    return StreamingTSEFlat(model, enrollment_samples=enrollment_samples).eval()


def _read_mono(path: Path, expected_sample_rate: int) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required for TSE ORT audio I/O") from exc
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"audio sample rate must be {expected_sample_rate}, got {sample_rate}: {path}"
        )
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if mono.size == 0 or not np.all(np.isfinite(mono)):
        raise ValueError(f"audio is empty or non-finite: {path}")
    return np.ascontiguousarray(mono)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TSE causal streaming ONNX Runtime waveform inference"
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--enrollment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speaker-embedding-npy", type=Path, default=None)
    parser.add_argument("--chunk-samples", type=int, default=4096)
    parser.add_argument("--tse-enrollment-samples", type=int, default=48000)
    args = parser.parse_args()
    if args.chunk_samples < 1:
        parser.error("--chunk-samples must be positive")

    from onnx.runtime_spec import runtime_spec_from_wrapper
    from onnx.utils import create_ort_session
    from tasks.tse.inference.streaming import CenteredStreamingSTFTOLA

    try:
        from core.utils.inference_artifacts import fingerprint_file
        from onnx.verify.manifest import (
            load_onnx_manifest,
            runtime_contract_from_spec,
            verify_onnx_manifest_sources,
        )

        session = create_ort_session(args.onnx)
        manifest = load_onnx_manifest(args.onnx)
        verify_onnx_manifest_sources(manifest)
        if manifest["artifact"] != fingerprint_file(args.onnx):
            raise RuntimeError("TSE ONNX manifest artifact fingerprint does not match file")
        source_files = manifest["source_binding"]["files"]
        for role, path in (("checkpoint", args.checkpoint), ("config", args.config)):
            declared = source_files.get(role)
            if declared is not None and Path(declared["path"]).resolve() != Path(path).resolve():
                raise RuntimeError(
                    f"TSE ONNX manifest {role} source does not match CLI input"
                )
        flat = _build_configured_wrapper(
            args.config, args.checkpoint, args.tse_enrollment_samples
        )
        artifact_spec = __import__(
            "onnx.runtime_spec", fromlist=["runtime_spec_from_session"]
        ).runtime_spec_from_session(session, "tse")
        if manifest["runtime_contract"] != runtime_contract_from_spec(artifact_spec):
            raise RuntimeError("TSE ONNX manifest runtime contract does not match artifact")
        wrapper_spec = runtime_spec_from_wrapper(flat, "tse")
        if (
            artifact_spec.input_names != wrapper_spec.input_names
            or artifact_spec.output_names != wrapper_spec.output_names
            or artifact_spec.stft != wrapper_spec.stft
            or artifact_spec.tse != wrapper_spec.tse
        ):
            raise RuntimeError("TSE ONNX artifact ABI does not match config/checkpoint")
        sample_rate = int(artifact_spec.stft.sample_rate)
        mixture = _read_mono(args.input, sample_rate)
        enrollment = _read_mono(args.enrollment, sample_rate)
        embedding = None
        if args.speaker_embedding_npy is not None:
            embedding = torch.from_numpy(
                np.load(args.speaker_embedding_npy, allow_pickle=False)
            ).float()
            if embedding.ndim == 1:
                embedding = embedding.unsqueeze(0)
        condition = _build_condition_state(
            flat, torch.from_numpy(enrollment).unsqueeze(0), embedding
        )
        processor = TSEStreamingORTProcessor(session, condition)
        frontend = CenteredStreamingSTFTOLA(
            processor,
            n_fft=int(artifact_spec.stft.n_fft),
            hop_length=int(artifact_spec.stft.hop_length),
            win_length=int(artifact_spec.stft.win_length),
        )
    except Exception as exc:
        parser.error(f"Failed to initialize TSE streaming runtime: {exc}")

    pieces = []
    for start in range(0, mixture.size, args.chunk_samples):
        value = frontend.push(mixture[start : start + args.chunk_samples])
        if value.size:
            pieces.append(value)
    tail = frontend.flush()
    if tail.size:
        pieces.append(tail)
    output = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)
    if output.shape != mixture.shape:
        raise RuntimeError(
            f"TSE streaming output length mismatch: {output.size} != {mixture.size}"
        )
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required for TSE ORT audio I/O") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), output, sample_rate, subtype="FLOAT")
    print(
        f"Saved {args.output} samples={output.size} "
        f"delay_samples={artifact_spec.tse.waveform_output_delay_samples}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
