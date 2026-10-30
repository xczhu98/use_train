#!/usr/bin/env python3
"""
AudioTag subprocess 包装脚本。

DEPRECATED：C2 processor 已移除 subprocess 模式（收敛为 dry_run /
subprocess_daemon），本脚本无 processor 调用方，仅保留给手工调试使用，
不删除文件。功能已被 audiotag_daemon.py 完全覆盖。

对应 icefall egs/audioset/AT/zipformer/pretrained.py，批量输出固定 top-5
证据（``audioset_top5``，排序自 canonical 527 sigmoid 分数的内部排名，
不持久化完整 527 数组）。

用法：
  python 04_audioset_label_audiotag.py \
    --checkpoint third_party_scripts/icefall_audiotaq/exp/pretrained.pt \
    --label-dict third_party_scripts/icefall_audiotaq/data/class_labels_indices.csv \
    --input_path input.json \
    --output_path output.json \
    --batch-size 16 \
    --sample-rate 16000

input.json 格式：
  [{"record_id": "id-1", "audio_path": "/path/to/a.wav"}, ...]

每个 output.json 项包含 ``record_id``、``audioset_top5``（恰好 5 项，prob
降序、index 升序，index/mid/label 与 canonical CSV 一致）、``score_contract``、
``class_count`` 与 ``label_space_sha256``，不包含任何 policy 或业务分组字段。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 独立 icefall_audiotaq 环境 subprocess 运行时 cwd 不保证是仓库根。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# recipe 的 train.py 必须优先于仓库根同名模块。
_RECIPE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "icefall",
    "egs",
    "audioset",
    "AT",
    "zipformer",
)
if _RECIPE_DIR not in sys.path:
    sys.path.insert(0, _RECIPE_DIR)

from audiotag_result import format_audiotag_results, load_label_metadata  # noqa: E402
from core.utils.file_transaction import write_text_files_transactionally  # noqa: E402
from core.utils.safe_load import safe_torch_load  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from torch.nn.utils.rnn import pad_sequence  # noqa: E402

try:
    import kaldifeat  # noqa: E402
    from train import add_model_arguments, get_model, get_params  # noqa: E402
except ImportError as exc:  # 可选 icefall 环境缺失时仍允许基础 import/--help。
    kaldifeat = None
    add_model_arguments = None
    get_model = None
    get_params = None
    _OPTIONAL_IMPORT_ERROR: ImportError | None = exc
else:
    _OPTIONAL_IMPORT_ERROR = None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是严格正整数")
    return parsed


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="pretrained.pt 路径")
    parser.add_argument("--label-dict", required=True, help="class_labels_indices.csv 路径")
    parser.add_argument("--input_path", required=True, help="输入 json 文件路径")
    parser.add_argument("--output_path", required=True, help="输出 json 文件路径")
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument("--sample-rate", type=_positive_int, default=16000)
    if add_model_arguments is not None:
        add_model_arguments(parser)
    return parser


def read_audio(path: str, expected_sr: int, device: torch.device) -> torch.Tensor:
    wave, sr = torchaudio.load(path)
    if type(sr) is not int or sr <= 0:
        raise ValueError(f"音频采样率必须是严格正整数: {path}: {sr!r}")
    if sr != expected_sr:
        wave = torchaudio.functional.resample(wave, sr, expected_sr)
    return wave[0].contiguous().to(device)


def main():
    parser = get_parser()
    args = parser.parse_args()
    if _OPTIONAL_IMPORT_ERROR is not None:
        parser.error(
            "AudioTag inference requires the icefall environment dependencies: "
            f"{_OPTIONAL_IMPORT_ERROR}"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device: %s", device)

    params = get_params()
    params.update(vars(args))

    logging.info("Loading model")
    model = get_model(params)
    checkpoint = safe_torch_load(
        args.checkpoint,
        map_location="cpu",
        purpose="load trusted AudioTag inference checkpoint",
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            f"AudioTag checkpoint {args.checkpoint} 缺少顶层 'model' state dict"
        )
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"AudioTag checkpoint strict 恢复失败: missing={sorted(missing)[:10]}, "
            f"unexpected={sorted(unexpected)[:10]}"
        )
    model.to(device)
    model.eval()

    label_metadata = load_label_metadata(args.label_dict)

    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = args.sample_rate
    opts.mel_opts.num_bins = params.feature_dim
    opts.mel_opts.high_freq = -400
    fbank = kaldifeat.Fbank(opts)

    with open(args.input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    if not isinstance(input_data, list) or not input_data:
        raise ValueError("input JSON 必须是非空 object list")
    for index, item in enumerate(input_data):
        if not isinstance(item, dict) or set(item) != {"record_id", "audio_path"}:
            raise ValueError(
                f"input JSON[{index}] 必须精确包含 record_id/audio_path"
            )
        if not isinstance(item["audio_path"], str) or not item["audio_path"]:
            raise ValueError(f"input JSON[{index}].audio_path 必须是非空字符串")

    audio_paths = [item["audio_path"] for item in input_data]
    record_ids = [item["record_id"] for item in input_data]
    if (
        any(not isinstance(record_id, str) or not record_id for record_id in record_ids)
        or len(set(record_ids)) != len(record_ids)
    ):
        raise ValueError("input JSON 的 record_id 必须是唯一非空字符串")
    results: List[Dict[str, Any]] = []

    for start in range(0, len(audio_paths), args.batch_size):
        batch_paths = audio_paths[start : start + args.batch_size]
        waves = [read_audio(p, args.sample_rate, device) for p in batch_paths]

        features = fbank(waves)
        feature_lengths = [f.size(0) for f in features]
        features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
        feature_lengths = torch.tensor(feature_lengths, device=device)

        encoder_out, encoder_out_lens = model.forward_encoder(features, feature_lengths)
        logits = model.forward_audio_tagging(encoder_out, encoder_out_lens)
        probs = logits.sigmoid()
        if probs.size(1) != label_metadata.class_count:
            raise RuntimeError(
                f"AudioTag 模型类别维 {probs.size(1)} 与 label dict "
                f"{label_metadata.class_count} 不一致"
            )
        batch_results = format_audiotag_results(
            probs.cpu().tolist(),
            label_metadata,
            record_ids=record_ids[start : start + args.batch_size],
        )
        results.extend(batch_results)

    payload = json.dumps(
        results, ensure_ascii=False, indent=2, allow_nan=False
    )
    write_text_files_transactionally(
        [(Path(args.output_path), lambda handle: handle.write(payload))],
        overwrite=True,
    )

    logging.info("Done: %d files -> %s", len(results), args.output_path)


if __name__ == "__main__":
    main()
