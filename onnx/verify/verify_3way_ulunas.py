#!/usr/bin/env python3
"""UL-UNAS 三路验证：离线 PT、逐帧流式 PT、逐帧 ORT。

不提供 checkpoint 时仅用于随机权重的手工结构验证。``--save-onnx`` 正式发布
还要求 checkpoint、float32、未放宽的预注册阈值，以及三路全部 PASS，之后才注入
官方 cache metadata 并发布到 ``onnx/onnx_model``。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from onnx.utils import export_onnx
from onnx.verify.manifest import runtime_contract_from_spec
from onnx.verify.verify_common import VerifiedOnnxArtifact, owned_onnx_cleanup


def _evaluate_three_way_statuses(
    frame_mses: list[float],
    offline_pt_mse: float,
    offline_ort_mse: float,
    expected_frames: int,
    threshold: float,
) -> tuple[list[bool], float]:
    """Return finite, thresholded UL-UNAS gate statuses."""
    frame_count = len(frame_mses)
    frame_mses_finite = (
        frame_count == expected_frames
        and all(np.isfinite(mse) for mse in frame_mses)
    )
    pt_ort_mse = max(frame_mses) if frame_mses else float("nan")
    statuses = [
        frame_mses_finite and np.isfinite(offline_pt_mse) and offline_pt_mse < threshold,
        frame_mses_finite and np.isfinite(pt_ort_mse) and pt_ort_mse < threshold,
        frame_mses_finite and np.isfinite(offline_ort_mse) and offline_ort_mse < threshold,
    ]
    return statuses, pt_ort_mse


def main() -> int:
    parser = argparse.ArgumentParser(description="UL-UNAS offline PT / streaming PT / ORT three-way verification")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="发布时绑定的 resolved config")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--mse-threshold", type=float, default=1e-4)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--save-onnx", type=Path, default=None)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")

    from onnx.export import write_streaming_runtime_metadata
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner
    from onnx.models.streaming_ulunas import StreamingULUNAS, StreamingULUNASFlat
    from onnx.verify.verify_common import (
        load_checkpoint_strict,
        release_results_pass,
        require_release_preflight,
        require_strict_runtime_spec,
    )
    from tasks.se.models.discriminative.ulunas import ULUNAS

    require_release_preflight(
        save_onnx=args.save_onnx,
        checkpoint=args.checkpoint,
        config=args.config,
        require_config=True,
        thresholds={"mse_threshold": (args.mse_threshold, 1e-4)},
    )
    torch.manual_seed(42)
    base = ULUNAS(n_fft=512).eval()
    source_binding = None
    if args.checkpoint:
        source_binding = load_checkpoint_strict(
            base,
            args.checkpoint,
            return_source_binding=True,
            source_files={"config": args.config},
            model_id="ulunas",
        )
    stream = StreamingULUNAS(base).eval()
    flat = StreamingULUNASFlat(stream).eval()
    spec = torch.randn(1, 257, args.frames, 2)

    with torch.inference_mode():
        offline = base(spec)
        cache_pt = stream.init_cache(1, "cpu", spec.dtype)
        pt_frames = []
        for index in range(args.frames):
            enhanced, cache_pt = stream(spec[:, :, index : index + 1, :], cache_pt)
            pt_frames.append(enhanced)
        streaming_pt = torch.cat(pt_frames, dim=2)

    with owned_onnx_cleanup() as lifecycle:
        fd, raw_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        onnx_path = lifecycle.own(raw_path)
        cache = flat.init_cache_flat(1, "cpu", spec.dtype)
        export_onnx(
            flat,
            (spec[:, :, :1, :], *cache),
            str(onnx_path),
            input_names=flat.get_input_names(),
            output_names=flat.get_output_names(),
            dynamic_axes=flat.get_dynamic_axes(),
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
        write_streaming_runtime_metadata(str(onnx_path), flat)
        onnx_artifact = VerifiedOnnxArtifact(onnx_path)
        session = onnx_artifact.create_session()
        runtime_spec = require_strict_runtime_spec(session, "ulunas")
        ort_runner = StreamingORTFrameRunner(session, spec=runtime_spec)
        ort_frames = []
        frame_mses = []
        for index in range(args.frames):
            ort_value, _ = ort_runner.run_frame(spec[:, :, index : index + 1, :])
            ort_frames.append(torch.from_numpy(ort_value))
            frame_mses.append(float(torch.mean((pt_frames[index] - ort_frames[-1]) ** 2).item()))
        streaming_ort = torch.cat(ort_frames, dim=2)

        offline_pt_mse = float(torch.mean((offline - streaming_pt) ** 2).item())
        offline_ort_mse = float(torch.mean((offline - streaming_ort) ** 2).item())
        statuses, pt_ort_mse = _evaluate_three_way_statuses(
            frame_mses,
            offline_pt_mse,
            offline_ort_mse,
            args.frames,
            args.mse_threshold,
        )
        print(f"Offline PT vs streaming PT: MSE={offline_pt_mse:.6e} [{'PASS' if statuses[0] else 'FAIL'}]")
        print(f"Streaming PT vs ORT: max frame MSE={pt_ort_mse:.6e} [{'PASS' if statuses[1] else 'FAIL'}]")
        print(f"Offline PT vs streaming ORT: MSE={offline_ort_mse:.6e} [{'PASS' if statuses[2] else 'FAIL'}]")
        all_pass = release_results_pass(
            ["PASS" if status else "FAIL" for status in statuses],
            compared_counts=[args.frames, len(frame_mses), args.frames],
            metrics=[offline_pt_mse, pt_ort_mse, offline_ort_mse],
            thresholds=[args.mse_threshold] * 3,
        )
        print(f"Summary: {'ALL PASS' if all_pass else 'SOME FAILED'}")

        if args.save_onnx:
            published = onnx_artifact.publish(
                args.save_onnx, all_pass=all_pass,
                manifest={
                    "model_id": "ulunas",
                    "source_files": {"checkpoint": args.checkpoint, "config": args.config},
                    "source_binding": source_binding,
                    "runtime_contract": runtime_contract_from_spec(runtime_spec),
                    "verification": {"statuses": ["PASS" if status else "FAIL" for status in statuses]},
                },
                require_manifest=True,
            )
            if published:
                print(f"ONNX artifact kept: {published['path']} sha256={published['sha256']}")
        onnx_artifact.close()
        return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
