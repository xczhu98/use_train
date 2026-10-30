#!/usr/bin/env python3
"""导出并验证与 TSE ONNX 相同 flat cache ABI 的 TorchScript artifact。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export verified TSE streaming TorchScript flat-cache runtime"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enrollment-samples", type=int, default=48000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from onnx.infer.infer_streaming_tse_ort import _build_configured_wrapper
    from tasks.tse.inference.torchscript import export_streaming_tse_torchscript

    flat = _build_configured_wrapper(
        args.config,
        args.checkpoint,
        args.enrollment_samples,
    )
    metadata = export_streaming_tse_torchscript(
        flat,
        args.output,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(
        f"Saved {args.output} with {len(metadata['input_names'])} inputs, "
        f"cache_abi={metadata['tse']['cache_abi_version']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
