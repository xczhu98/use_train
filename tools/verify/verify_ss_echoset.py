"""使用 SS checkpoint 在 strict EchoSet manifest 上计算 SI-SDR/SI-SDRi。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.metrics.separation import si_sdr, si_sdr_improvement
from core.stft import build_stft_istft
from core.utils.inference_artifacts import atomic_write_json, fingerprint_file
from core.utils.safe_load import safe_torch_load
from core.utils.config import load_merged_config
from tasks.ss.data.offline_dataset import SSOfflineDataset
from tasks.ss.inference.extractor import SSExtractor
from tasks.ss.model_io import resolve_ss_execution_plan
from tasks.ss.models import build_model


def _load_model_state(path: Path) -> tuple[dict, dict | None]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is required for this checkpoint") from exc
        return load_file(str(path), device="cpu"), None
    checkpoint = safe_torch_load(
        path,
        purpose="verify public SS model-only checkpoint",
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("SS checkpoint must be a mapping")
    state = checkpoint.get("state_dict", checkpoint)
    model_args = checkpoint.get("model_args")
    if not isinstance(state, dict):
        raise TypeError("SS checkpoint state_dict must be a mapping")
    if model_args is not None and not isinstance(model_args, dict):
        raise TypeError("SS checkpoint model_args must be a mapping when present")
    return state, model_args


def evaluate(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    max_samples: int,
) -> dict:
    if type(max_samples) is not int or max_samples <= 0:
        raise ValueError("max_samples must be an exact positive int")
    config = load_merged_config(config_path)
    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    state, checkpoint_args = _load_model_state(checkpoint)
    model_cfg = dict(config["model"])
    model_cfg["args"] = dict(checkpoint_args or model_cfg.get("args", {}))
    stft_cfg = config.get("stft")
    model = build_model(
        model_cfg,
        stft_cfg=stft_cfg,
        num_sources=int((config.get("ss") or {})["num_sources"]),
    ).eval()
    model.load_state_dict(state, strict=True)
    stft = build_stft_istft(stft_cfg) if stft_cfg else None
    dataset = SSOfflineDataset(
        manifest_path,
        num_sources=int((config.get("ss") or {})["num_sources"]),
        strict_load=True,
    )
    plan = resolve_ss_execution_plan(
        config,
        model,
        dataset_contract=dataset.contract,
    )
    extractor = SSExtractor(model, plan=plan, stft_module=stft)
    sample_count = min(max_samples, len(dataset))
    # 聚合指标以 sample_count 为分母：空 manifest 会在下方除零，必须先 fail-fast
    if sample_count <= 0:
        raise ValueError(
            f"manifest 内没有可评估样本（dataset size={len(dataset)}, max_samples={max_samples}）"
        )
    sisdr_values: list[float] = []
    sisdri_values: list[float] = []
    elapsed_values: list[float] = []
    items = []
    for index in range(sample_count):
        sample = dataset[index]
        mixture = sample["mixture"].unsqueeze(0)
        reference = torch.stack(sample["sources"]).unsqueeze(0)
        lengths = torch.tensor([sample["length"]], dtype=torch.int64)
        start = time.perf_counter()
        estimated = extractor(mixture, sample_rate=sample["sr"])
        elapsed = time.perf_counter() - start
        sisdr, assignment = si_sdr(estimated, reference, lengths=lengths)
        sisdri, _ = si_sdr_improvement(
            estimated, reference, mixture, lengths=lengths
        )
        sisdr_mean = float(sisdr.mean())
        sisdri_mean = float(sisdri.mean())
        sisdr_values.append(sisdr_mean)
        sisdri_values.append(sisdri_mean)
        elapsed_values.append(elapsed)
        items.append(
            {
                "uid": sample["uid"],
                "assignment": assignment.tolist()[0],
                "si_sdr_db": sisdr.tolist()[0],
                "si_sdri_db": sisdri.tolist()[0],
                "si_sdr_mean_db": sisdr_mean,
                "si_sdri_mean_db": sisdri_mean,
                "elapsed_seconds": elapsed,
            }
        )
    return {
        "schema": "ss-echoset-evaluation-v1",
        "config": str(Path(config_path).resolve()),
        "checkpoint": fingerprint_file(checkpoint),
        "manifest": fingerprint_file(manifest_path),
        "evaluated_samples": sample_count,
        "aggregate": {
            "si_sdr_mean_db": sum(sisdr_values) / sample_count,
            "si_sdri_mean_db": sum(sisdri_values) / sample_count,
            "elapsed_mean_seconds": sum(elapsed_values) / sample_count,
            "elapsed_total_seconds": sum(elapsed_values),
        },
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        max_samples=args.max_samples,
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
