#!/usr/bin/env python
"""Apollo 超分独立推理脚本（供 noise_filter_pipeline D1 subprocess_daemon 模式调用）。

接口约定（与 processors/apollo_superres.py `_ApolloDaemonPool` 对应）：
  --input_path       输入 JSON：list[{"audio_path": ...}]
  --output_path      输出 JSON：list[{"audio_path": ..., "enhanced_audio_path": ...}]
  --output_audio_dir 增强音频输出目录（统一写 target_sr 的 wav）
  --batch_size       模型 batch 大小
  --config_path      Apollo 配置（默认取脚本同目录 config_apollo_uni.yaml）
  --checkpoint       Apollo 权重（默认取脚本同目录 apollo_model_uni.ckpt）

多卡由调用方通过 CUDA_VISIBLE_DEVICES 分片控制，本脚本只用单卡。
输出统一为 target_sr(44.1kHz) wav；采样率/格式转换由调用方
copy_or_convert_audio 完成。

--daemon 长驻模式（与 icefall_audiotaq/audiotag_daemon.py 协议同构）：
模型只在启动时加载一次，之后通过 stdin/stdout 以 JSON Lines 交互，
日志一律写 stderr，避免污染 stdout 上的协议输出：
  {"type": "ping"}
    -> {"type": "pong"}
  {"type": "enhance", "batch_id": N, "items": [{"audio_path": ...}, ...]}
    -> {"type": "result", "batch_id": N, "results": [
         {"audio_path": ..., "enhanced_audio_path": ...}
         或 {"audio_path": ..., "error": ...}, ...]}
  {"type": "stop"}
    -> 回复 {"type": "stopped"} 后退出
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("inference_apollo")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """用多相滤波重采样（scipy），避免额外依赖。"""
    if orig_sr == target_sr:
        return audio
    from scipy.signal import resample_poly

    gcd = math.gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // gcd, orig_sr // gcd).astype(np.float32)


def _prepare_chunks(
    audio: np.ndarray, sr: int, chunk_sec: float = 10.0, overlap_sec: float = 2.0
):
    """把音频切分为带重叠的 chunks（与 pipeline 中 _prepare_chunks 逻辑一致）。"""
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    step_samples = max(1, chunk_samples - overlap_samples)
    fade = overlap_samples
    border = overlap_samples  # 首尾的 reflect pad 等于重叠长度

    original_len = audio.shape[0]
    if original_len > 2 * border and border > 0:
        audio = np.pad(audio, (border, border), mode="reflect")

    chunks = []
    positions = []
    i = 0
    while i < audio.shape[0]:
        part = audio[i : i + chunk_samples]
        length = part.shape[0]
        if length < chunk_samples:
            if length > chunk_samples // 2 + 1:
                part = np.pad(part, (0, chunk_samples - length), mode="reflect")
            else:
                part = np.pad(part, (0, chunk_samples - length), mode="constant")
        chunks.append(part)
        positions.append((i, length))
        i += step_samples

    return chunks, positions, audio.shape[0], original_len, fade, border


def _get_window(chunk_size: int, fade_size: int) -> np.ndarray:
    window = np.ones(chunk_size)
    fade_in = np.linspace(0, 1, fade_size)
    fade_out = np.linspace(1, 0, fade_size)
    window[:fade_size] *= fade_in
    window[-fade_size:] *= fade_out
    return window


def _ensure_look2hear_importable() -> None:
    """确保 look2hear 可导入。

    look2hear 没有 PyPI 包，其代码 vendored 在 Apollo 仓库中。查找顺序：
      1. 已安装（pip / PYTHONPATH）；
      2. 环境变量 APOLLO_REPO 指向的 Apollo 仓库根目录；
      3. 脚本同级的 ../apollo_repo（third_party_scripts/apollo_repo）。
    """
    try:
        import look2hear  # noqa: F401

        return
    except ImportError:
        pass
    candidates = []
    if os.environ.get("APOLLO_REPO"):
        candidates.append(os.environ["APOLLO_REPO"])
    candidates.append(os.path.join(_SCRIPT_DIR, "..", "apollo_repo"))
    for cand in candidates:
        cand = os.path.abspath(cand)
        if os.path.isdir(os.path.join(cand, "look2hear")):
            sys.path.insert(0, cand)
            return
    raise ImportError(
        "找不到 look2hear 包；请设置 APOLLO_REPO 指向 Apollo 仓库根目录，"
        "或将 Apollo 仓库克隆到 third_party_scripts/apollo_repo"
    )


def _load_model(config_path: str, checkpoint: str, device: str = None):
    _ensure_look2hear_importable()
    import torch
    import look2hear.models
    from ml_collections import ConfigDict
    import yaml

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    # hydra 风格配置带有 _target_ 键，from_pretrain 不会过滤，需调用方剥离
    model_cfg = {k: v for k, v in dict(cfg["model"]).items() if k != "_target_"}
    model = look2hear.models.BaseModel.from_pretrain(checkpoint, **model_cfg).to(device)
    model.eval()
    logger.info("Apollo 模型加载完成: device=%s, checkpoint=%s", device, checkpoint)
    return model, device


def _enhance_one(
    model,
    device: str,
    audio: np.ndarray,
    target_sr: int,
    chunk_sec: float,
    overlap_sec: float,
    batch_size: int,
) -> np.ndarray:
    """对单条音频（已重采样到 target_sr）做分块 batch 推理 + overlap-add。"""
    import torch

    chunks, positions, padded_len, original_len, fade, border = _prepare_chunks(
        audio, target_sr, chunk_sec, overlap_sec
    )
    if not chunks:
        return audio

    out_chunk_len = chunks[0].shape[-1]
    window = _get_window(out_chunk_len, fade)
    result = np.zeros(padded_len, dtype=np.float32)
    counter = np.zeros(padded_len, dtype=np.float32)

    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch = torch.stack(
            [torch.from_numpy(c).unsqueeze(0).float() for c in batch_chunks]
        ).to(device)
        with torch.no_grad():
            outputs = model(batch).squeeze(1).cpu().numpy()
        for chunk_idx, out in zip(range(start, start + len(batch_chunks)), outputs):
            pos, length = positions[chunk_idx]
            w = window.copy()
            # 单 chunk 音频同时满足首尾，需要分别处理，不能 elif
            if chunk_idx == 0:
                w[:fade] = 1.0
            if pos + out_chunk_len >= padded_len:
                w[-fade:] = 1.0
            result[pos : pos + length] += out[:length] * w[:length]
            counter[pos : pos + length] += w[:length]

    result = np.divide(result, counter, out=np.zeros_like(result), where=counter > 0)
    if padded_len > 2 * border and border > 0:
        result = result[border:-border]
    return result[:original_len]


def _enhance_item_to_file(
    model,
    device: str,
    src: str,
    output_audio_dir: str,
    target_sr: int,
    chunk_sec: float,
    overlap_sec: float,
    batch_size: int,
) -> str:
    """增强单条音频并写出到 output_audio_dir，返回输出路径。

    输出命名为 {stem}.wav，同名（不同目录来源）时追加 _{n} 序号去重。
    """
    audio, sr = sf.read(src)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    audio = _resample(audio, sr, target_sr)

    enhanced = _enhance_one(
        model,
        device,
        audio,
        target_sr,
        chunk_sec,
        overlap_sec,
        max(1, batch_size),
    )

    stem = os.path.splitext(os.path.basename(src))[0]
    out_path = os.path.join(output_audio_dir, f"{stem}.wav")
    # 同名输入（不同目录）时追加序号，避免覆盖
    n = 1
    while os.path.exists(out_path):
        out_path = os.path.join(output_audio_dir, f"{stem}_{n}.wav")
        n += 1
    sf.write(out_path, enhanced, target_sr)
    return out_path


def _daemon_reply(resp: Dict[str, Any], stream) -> None:
    """向协议通道写一条 JSON 响应并立即 flush。"""
    stream.write(json.dumps(resp, ensure_ascii=False) + "\n")
    stream.flush()


def _run_daemon(args: argparse.Namespace) -> int:
    """长驻 daemon 模式：模型只加载一次，循环处理 stdin 上的 JSON 命令。"""
    # daemon 协议走 stdout，日志必须切到 stderr，避免污染协议输出
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    os.makedirs(args.output_audio_dir, exist_ok=True)

    # look2hear 等库在模型加载/推理时会直接 print 到 stdout（Python 层和
    # C 扩展的 fd 1 都有），协议输出必须独占 stdout：
    # 1) 先复制一份原始 fd 1 作为协议专用通道 proto_out（之后所有协议回复
    #    都走它，不走 sys.stdout）；
    # 2) Python 层 sys.stdout 永久切到 stderr；
    # 3) fd 1 在 daemon 整个生命周期内保持重定向到 fd 2（stderr），直到
    #    进程退出都不恢复——否则推理期间 C 层对 fd 1 的写入会污染协议通道。
    proto_out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    sys.stdout = sys.stderr
    os.dup2(2, 1)
    model, device = _load_model(args.config_path, args.checkpoint, args.device)
    logger.info("Apollo daemon 就绪，等待 stdin 命令")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            _daemon_reply({"type": "error", "message": f"Invalid JSON: {e}"}, proto_out)
            continue

        cmd_type = cmd.get("type")
        if cmd_type == "ping":
            _daemon_reply({"type": "pong"}, proto_out)
        elif cmd_type == "stop":
            _daemon_reply({"type": "stopped"}, proto_out)
            logger.info("Apollo daemon 收到 stop，退出")
            return 0
        elif cmd_type == "enhance":
            batch_id = cmd.get("batch_id")
            results: List[Dict[str, Any]] = []
            for item in cmd.get("items", []):
                src = item.get("audio_path")
                try:
                    out_path = _enhance_item_to_file(
                        model,
                        device,
                        src,
                        args.output_audio_dir,
                        args.target_sr,
                        args.chunk_sec,
                        args.overlap_sec,
                        max(1, args.batch_size),
                    )
                    results.append({"audio_path": src, "enhanced_audio_path": out_path})
                except Exception as e:  # noqa: BLE001
                    # 单条失败不阻断整批；调用方按 error 字段把记录标记为 rejected
                    logger.error("增强失败 %s: %s", src, e)
                    results.append({"audio_path": src, "error": str(e)})
            _daemon_reply({"type": "result", "batch_id": batch_id, "results": results}, proto_out)
        else:
            _daemon_reply({"type": "error", "message": f"Unknown type: {cmd_type}"}, proto_out)

    # stdin 关闭（父进程退出）时随之退出
    logger.info("Apollo daemon stdin 关闭，退出")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Apollo 超分独立推理脚本")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="长驻 daemon 模式：模型只加载一次，stdin/stdout JSON Lines 交互",
    )
    parser.add_argument("--input_path", help="输入 JSON: list[{audio_path}]（非 daemon 模式必填）")
    parser.add_argument("--output_path", help="输出 JSON 路径（非 daemon 模式必填）")
    parser.add_argument("--output_audio_dir", required=True, help="增强音频输出目录")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--config_path",
        default=os.path.join(_SCRIPT_DIR, "config_apollo_uni.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(_SCRIPT_DIR, "apollo_model_uni.ckpt"),
    )
    parser.add_argument("--target_sr", type=int, default=44100)
    parser.add_argument("--chunk_sec", type=float, default=10.0)
    parser.add_argument("--overlap_sec", type=float, default=2.0)
    parser.add_argument(
        "--device",
        default=None,
        help="推理设备（如 cuda:0 / cpu）；默认自动检测。daemon 多卡时由调用方"
        "通过 CUDA_VISIBLE_DEVICES 限定后传 cuda:0",
    )
    args = parser.parse_args()

    if args.daemon:
        return _run_daemon(args)
    if not args.input_path or not args.output_path:
        parser.error("非 daemon 模式必须提供 --input_path 和 --output_path")

    with open(args.input_path, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)
    os.makedirs(args.output_audio_dir, exist_ok=True)

    model, device = _load_model(args.config_path, args.checkpoint, args.device)

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        src = item["audio_path"]
        try:
            out_path = _enhance_item_to_file(
                model,
                device,
                src,
                args.output_audio_dir,
                args.target_sr,
                args.chunk_sec,
                args.overlap_sec,
                max(1, args.batch_size),
            )
            results.append({"audio_path": src, "enhanced_audio_path": out_path})
        except Exception as e:  # noqa: BLE001
            # 单条失败不阻断整批；缺失条目的记录会被调用方标记为 rejected
            logger.error("增强失败 %s: %s", src, e)
        if (idx + 1) % 100 == 0:
            logger.info("已完成 %d/%d", idx + 1, len(items))

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("完成：%d/%d 条成功，输出 %s", len(results), len(items), args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
