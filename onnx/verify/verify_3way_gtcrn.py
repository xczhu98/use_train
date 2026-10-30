#!/usr/bin/env python3
"""GTCRN 三路验证：离线 PT、逐帧流式 PT、逐帧 ORT。"""

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


def _comparison(name, left, right, count: int, threshold: float):
    from onnx.verify.verify_common import ThreeWayComparison

    mse = float(torch.mean((left - right) ** 2).item()) if count > 0 else float("nan")
    max_abs = float(torch.max(torch.abs(left - right)).item()) if count > 0 else float("nan")
    status = "PASS" if count > 0 and np.isfinite(mse) and np.isfinite(max_abs) and mse < threshold else "FAIL"
    return ThreeWayComparison(name, mse, max_abs, count, threshold, status)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GTCRN offline PT / streaming PT / ORT three-way verification",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="发布时绑定的 resolved config")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--mse-threshold", type=float, default=1e-5)
    parser.add_argument("--save-onnx", type=Path, default=None)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")

    from onnx.export import write_streaming_runtime_metadata
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner
    from onnx.models.streaming_gtcrn import StreamingGTCRN, convert_gtcrn_to_streaming
    from onnx.verify.verify_common import (
        ThreeWayResult,
        load_checkpoint_strict,
        require_release_preflight,
        require_strict_runtime_spec,
    )
    from tasks.se.models.discriminative.gtcrn import GTCRN

    require_release_preflight(
        save_onnx=args.save_onnx,
        checkpoint=args.checkpoint,
        config=args.config,
        require_config=True,
        thresholds={"mse_threshold": (args.mse_threshold, 1e-5)},
    )
    torch.manual_seed(42)
    base = GTCRN(n_fft=512).eval()
    source_binding = None
    if args.checkpoint:
        source_binding = load_checkpoint_strict(
            base,
            args.checkpoint,
            return_source_binding=True,
            source_files={"config": args.config},
            model_id="gtcrn",
        )
    stream = StreamingGTCRN(n_fft=512, hop_length=256, win_length=512).eval()
    convert_gtcrn_to_streaming(stream, base)

    spec = torch.randn(1, 257, args.frames, 2)
    with torch.inference_mode():
        offline = base(spec)
        cache = stream.init_cache(1, torch.device("cpu"), spec.dtype)
        pt_frames = []
        for index in range(args.frames):
            enhanced, *cache_values = stream(
                spec[:, :, index:index + 1, :],
                *(cache[name] for name in stream.CACHE_KEYS),
            )
            cache = dict(zip(stream.CACHE_KEYS, cache_values))
            pt_frames.append(enhanced)
        streaming_pt = torch.cat(pt_frames, dim=2)

    with owned_onnx_cleanup() as lifecycle:
        fd, raw_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        onnx_path = lifecycle.own(raw_path)
        export_cache = stream.init_cache(1, torch.device("cpu"), spec.dtype)
        export_onnx(
            stream,
            (spec[:, :, :1, :], *(export_cache[name] for name in stream.CACHE_KEYS)),
            str(onnx_path),
            input_names=stream.get_input_names(),
            output_names=stream.get_output_names(),
            dynamic_axes=stream.get_dynamic_axes(),
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
        write_streaming_runtime_metadata(str(onnx_path), stream)
        onnx_artifact = VerifiedOnnxArtifact(onnx_path)
        session = onnx_artifact.create_session()
        runtime_spec = require_strict_runtime_spec(session, "gtcrn")
        runner = StreamingORTFrameRunner(session, spec=runtime_spec)
        ort_result = runner.run(
            ((spec[:, :, index:index + 1, :],) for index in range(args.frames))
        )
        streaming_ort = ort_result.to_torch()

        result = ThreeWayResult(
            _comparison("offline PT vs streaming PT", offline, streaming_pt, args.frames, args.mse_threshold),
            _comparison("streaming PT vs ORT", streaming_pt, streaming_ort, args.frames, args.mse_threshold),
            _comparison("offline PT vs ORT", offline, streaming_ort, args.frames, args.mse_threshold),
        )
        for comparison in result.comparisons:
            print(
                f"{comparison.name}: MSE={comparison.mse:.6e}, "
                f"max_abs={comparison.max_abs:.6e} [{comparison.status}]"
            )
        print(f"Summary: {'ALL PASS' if result.all_pass else 'SOME FAILED'}")
        if args.save_onnx:
            published = onnx_artifact.publish(
                args.save_onnx, all_pass=result.all_pass,
                manifest={
                    "model_id": "gtcrn",
                    "source_files": {"checkpoint": args.checkpoint, "config": args.config},
                    "source_binding": source_binding,
                    "runtime_contract": runtime_contract_from_spec(runtime_spec),
                    "verification": {"statuses": [item.status for item in result.comparisons]},
                },
                require_manifest=True,
            )
            if published:
                print(f"ONNX artifact kept: {published['path']} sha256={published['sha256']}")
        onnx_artifact.close()
        return 0 if result.all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
