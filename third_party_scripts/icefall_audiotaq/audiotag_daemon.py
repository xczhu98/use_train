#!/usr/bin/env python3
"""
AudioTag 长驻 daemon，基于 icefall audioset/AT/zipformer 预训练模型。

通过 stdin/stdout 以 JSON Lines 协议接收命令，模型只加载一次：

  {"cmd": "infer", "input_path": "...", "output_path": "...", "record_ids": ["..."]}
  -> {"status": "ok"} / {"status": "error", "error_kind": "...", "error_code": "...", "message": "...", "record_ids": [...]}

  {"cmd": "exit"}
  -> daemon 退出

C2 仅输出固定 top-5 证据（``audioset_top5``，排序自 canonical 527 sigmoid
分数的内部排名，不持久化完整 527 数组），不加载或解释任何准入 policy。
checkpoint 使用安全加载且以 strict 方式恢复，任何 missing/unexpected key 都
fail-closed。输出携带 record_id，供 processor 做数量、顺序与身份守恒校验。

用法（在 icefall_audiotaq conda 环境内）：
  python audiotag_daemon.py \
    --checkpoint exp/pretrained.pt \
    --label-dict data/class_labels_indices.csv \
    --sample-rate 16000 \
    --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

try:
    import numpy as np
    import soundfile as _soundfile
except ImportError:  # 环境缺 soundfile 时回退 torchaudio 解码路径
    np = None
    _soundfile = None


# daemon 由 pipeline 以 subprocess 拉起、cwd 不保证是仓库根；repo-root bootstrap
# 必须位于任何 core.* import 之前，独立 icefall_audiotaq 环境才能解析 core 包。
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = (_SCRIPT_DIR / ".." / "..").resolve()
for _path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from core.utils.file_transaction import write_text_files_transactionally  # noqa: E402
from core.utils.safe_load import safe_torch_load  # noqa: E402

# 把 icefall audioset/AT/zipformer 路径加入 sys.path，以便导入 train/model
_ICEFALL_ZIPFORMER = (_SCRIPT_DIR / ".." / "icefall" / "egs" / "audioset" / "AT" / "zipformer").resolve()
if str(_ICEFALL_ZIPFORMER) not in sys.path:
    sys.path.insert(0, str(_ICEFALL_ZIPFORMER))

# 同时需要 icefall 根目录（import icefall 包）
_ICEFALL_ROOT = (_SCRIPT_DIR / ".." / "icefall").resolve()
if str(_ICEFALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ICEFALL_ROOT))

from audiotag_result import (  # noqa: E402
    AUDIOTAG_TOP_K,
    CLASSIFIER_LABEL_COUNT,
    EVIDENCE_CONTRACT_VERSION,
    SCORE_CONTRACT,
    load_label_metadata,
)


class _AudioDecodeError(RuntimeError):
    """单条音频内容无法由 torchaudio.load 解码。"""


class _AudioStorageUnavailableError(RuntimeError):
    """读取音频时发生权限、设备或文件系统故障。"""


def _error_response(kind: str, code: str, message: str, record_ids: List[str]) -> Dict[str, Any]:
    return {"status": "error", "error_kind": kind, "error_code": code,
            "message": message, "record_ids": record_ids}


def _validate_record_ids(value: Any, source: str) -> List[str]:
    """严格校验协议中的有序 record_id 列表。"""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source} record_ids 必须是非空 list[str]")
    if any(not isinstance(record_id, str) or not record_id for record_id in value):
        raise ValueError(f"{source} record_ids 必须全部是非空字符串")
    if len(set(value)) != len(value):
        raise ValueError(f"{source} record_ids 不允许重复")
    return list(value)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是严格正整数")
    return parsed


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AudioTag 长驻 daemon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained.pt",
    )
    parser.add_argument(
        "--label-dict",
        type=str,
        required=True,
        help="Path to class_labels_indices.csv",
    )
    parser.add_argument(
        "--sample-rate",
        type=_positive_int,
        default=16000,
        help="输入音频采样率",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=16,
        help="推理 batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备",
    )
    parser.add_argument("--cpu-threads", type=_positive_int, default=1)
    parser.add_argument("--interop-threads", type=_positive_int, default=1)
    parser.add_argument("--audio-workers", type=_positive_int, default=4)
    parser.add_argument("--timing-log-interval", type=_positive_int, default=20)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印详细日志到 stderr",
    )
    return parser


class AudioTagDaemon:
    """AudioTag 长驻推理 daemon。"""

    def __init__(
        self,
        checkpoint: str,
        label_dict_path: str,
        sample_rate: int = 16000,
        batch_size: int = 16,
        device: Optional[str] = None,
        audio_workers: int = 4,
        timing_log_interval: int = 20,
    ):
        self.checkpoint = checkpoint
        self.label_dict_path = label_dict_path
        for name, value in (
            ("sample_rate", sample_rate),
            ("batch_size", batch_size),
            ("audio_workers", audio_workers),
            ("timing_log_interval", timing_log_interval),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是严格正整数")
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.audio_workers = audio_workers
        self.timing_log_interval = timing_log_interval
        # 每个 source rate 缓存一个推理设备上的重采样实例；仅主线程串行调用，
        # 无需锁。重采样放在 GPU 上是为了把 CPU 从 sinc 插值中解放出来——
        # 解码是唯一剩下的 CPU 重活，可占满 audio worker 池。
        self._resamplers: Dict[int, torch.nn.Module] = {}
        self._timing = {
            key: 0.0
            for key in (
                "decode",
                "resample",
                "h2d_fbank",
                "model",
                "serialize",
                "total",
            )
        }
        self._timing_batches = 0
        self._timing_files = 0

        # 延迟导入，避免脚本启动时因环境缺失直接失败
        from train import add_model_arguments, get_model, get_params
        import kaldifeat

        self._add_model_arguments = add_model_arguments
        self._get_model = get_model
        self._get_params = get_params
        self._kaldifeat = kaldifeat

        self.label_metadata = load_label_metadata(label_dict_path)
        if self.label_metadata.class_count != CLASSIFIER_LABEL_COUNT:
            raise ValueError(
                f"AudioTag label dict 必须包含 {CLASSIFIER_LABEL_COUNT} 类"
            )
        self._expected_class_count = CLASSIFIER_LABEL_COUNT
        self.model = self._load_model()
        self.fbank = self._build_fbank()
        self._audio_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=audio_workers,
            thread_name_prefix="audiotag-audio",
        )
        # 单线程预取池：infer 内做双缓冲——当前 batch 在 GPU 上跑时，下一 batch
        # 的解码在 audio worker 池并行进行，隐藏 CPU 解码墙钟。
        self._prefetch_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="audiotag-prefetch",
        )

    def _load_model(self) -> torch.nn.Module:
        logging.info("Loading model from %s", self.checkpoint)
        params = self._get_params()
        # 构造一个最小 parser，让 add_model_arguments 写入默认值
        parser = argparse.ArgumentParser()
        self._add_model_arguments(parser)
        model_args = parser.parse_args([])
        params.update(vars(model_args))
        params.sample_rate = self.sample_rate

        model = self._get_model(params)
        checkpoint = safe_torch_load(
            self.checkpoint,
            map_location="cpu",
            purpose="load trusted AudioTag inference checkpoint",
        )
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(
                f"AudioTag checkpoint {self.checkpoint} 缺少顶层 'model' state dict"
            )
        missing, unexpected = model.load_state_dict(
            checkpoint["model"], strict=True
        )
        if missing or unexpected:
            raise RuntimeError(
                f"AudioTag checkpoint strict 恢复失败: missing={sorted(missing)[:10]}, "
                f"unexpected={sorted(unexpected)[:10]}"
            )
        model.to(self.device)
        model.eval()
        num_param = sum(p.numel() for p in model.parameters())
        logging.info("Model loaded, params=%d, device=%s", num_param, self.device)
        # 保留 params 供 fbank 读取 feature_dim，与 one-shot 脚本保持一致
        self._params = params
        return model

    def _build_fbank(self):
        opts = self._kaldifeat.FbankOptions()
        opts.device = self.device
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = self.sample_rate
        # 使用模型参数中的 feature_dim，避免硬编码 80 与 checkpoint 不匹配
        opts.mel_opts.num_bins = int(getattr(self._params, "feature_dim", 80))
        opts.mel_opts.high_freq = -400
        return self._kaldifeat.Fbank(opts)

    @staticmethod
    def _decode_wave(path: str) -> tuple[torch.Tensor, int]:
        # backend 对缺失路径、权限和设备故障也可能只抛 RuntimeError；先直接
        # 探测存储可达性，避免依赖 backend 异常文本猜测错误类别。
        try:
            with open(path, "rb") as audio_file:
                first_byte = audio_file.read(1)
        except OSError as exc:
            raise _AudioStorageUnavailableError(
                f"音频存储不可用: {path}: {exc}"
            ) from exc
        if not first_byte:
            raise _AudioDecodeError(f"音频解码失败: {path}: 文件为空")

        if _soundfile is not None:
            # 优先 soundfile：libsndfile 在 C 层解码并释放 GIL，多线程真正并行；
            # torchaudio 2.x dispatcher 的 ffmpeg 路径持 GIL，加再多 worker 也串行
            # （实测 4→16 线程吞吐不变，soundfile 近似线性扩展）。
            try:
                data, sample_rate = _soundfile.read(
                    path, dtype="float32", always_2d=True
                )
            except OSError as exc:
                raise _AudioStorageUnavailableError(
                    f"音频存储不可用: {path}: {exc}"
                ) from exc
            except RuntimeError:
                # soundfile 不支持的编码（如部分 mp3/ogg 变体）落回 torchaudio
                pass
            else:
                return torch.from_numpy(np.ascontiguousarray(data[:, 0])), sample_rate
        try:
            wave, sample_rate = torchaudio.load(path)
        except OSError as exc:
            raise _AudioStorageUnavailableError(
                f"音频存储不可用: {path}: {exc}"
            ) from exc
        except RuntimeError as exc:
            raise _AudioDecodeError(f"音频解码失败: {path}: {exc}") from exc
        return wave[0].contiguous(), sample_rate

    def _resample_on_device(self, wave: torch.Tensor, source_rate: int) -> torch.Tensor:
        """在推理设备上重采样；按 source rate 缓存实例（仅主线程调用）。"""
        if type(source_rate) is not int or source_rate <= 0:
            raise _AudioDecodeError(f"音频采样率必须是严格正整数: {source_rate!r}")
        if source_rate == self.sample_rate:
            return wave
        resampler = self._resamplers.get(source_rate)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(
                source_rate, self.sample_rate
            ).to(self.device)
            self._resamplers[source_rate] = resampler
        return resampler(wave)

    def _decode_waves(
        self, audio_paths: List[str]
    ) -> Tuple[List[Tuple[torch.Tensor, int]], float]:
        """并行解码并保持输入顺序；重采样在推理设备主线程统一执行。"""
        executor = self._audio_executor
        if executor is None:
            raise RuntimeError("AudioTag daemon 已关闭")

        started = time.perf_counter()
        decoded = list(executor.map(self._decode_wave, audio_paths))
        decode_seconds = time.perf_counter() - started
        return decoded, decode_seconds

    def _record_timing(self, files: int, **timing: float) -> None:
        unknown = timing.keys() - self._timing.keys()
        if unknown:
            raise ValueError(f"未知 AudioTag timing 字段: {sorted(unknown)}")
        for key, value in timing.items():
            self._timing[key] += value
        self._timing_batches += 1
        self._timing_files += files
        self._flush_timing(force=False)

    def _flush_timing(self, *, force: bool) -> None:
        if self._timing_batches == 0:
            return
        if not force and self._timing_batches < self.timing_log_interval:
            return
        total = self._timing["total"]
        logging.info(
            "AudioTag timing wall(s): decode=%.3f resample=%.3f h2d_fbank=%.3f "
            "model=%.3f serialize=%.3f total=%.3f files/s=%.2f "
            "files=%d batches=%d",
            *(self._timing[key] for key in self._timing),
            self._timing_files / total if total > 0.0 else 0.0,
            self._timing_files,
            self._timing_batches,
        )
        self._timing = {key: 0.0 for key in self._timing}
        self._timing_batches = 0
        self._timing_files = 0

    def _format_top5(
        self, probs: torch.Tensor, record_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """从完整 527 维 sigmoid 概率直接产出固定 top-5 证据。

        在 CPU 上做稳定降序排序：等值保持 index 升序，与
        ``format_audiotag_results`` 的 ``(-prob, index)`` 排序契约严格等价；
        只把 top-5 标量搬回 Python，省掉整行 527 维 ``tolist`` + 逐值校验 +
        Python 排序的开销。
        """
        if not bool(torch.isfinite(probs).all()):
            raise RuntimeError("AudioTag 模型输出含 NaN/Inf，拒绝生成证据")
        sorted_probs, sorted_indices = torch.sort(
            probs.cpu(), dim=1, descending=True, stable=True
        )
        top_probs = sorted_probs[:, :AUDIOTAG_TOP_K].tolist()
        top_indices = sorted_indices[:, :AUDIOTAG_TOP_K].tolist()
        metadata = self.label_metadata
        results: List[Dict[str, Any]] = []
        for row_index, record_id in enumerate(record_ids):
            top5 = []
            for prob, index in zip(top_probs[row_index], top_indices[row_index]):
                mid = metadata.index_to_mid[index]
                top5.append({
                    "index": index,
                    "mid": mid,
                    "label": metadata.label_of(mid),
                    "prob": float(prob),
                })
            results.append({
                "record_id": record_id,
                "audioset_top5": top5,
                "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
                "score_contract": SCORE_CONTRACT,
                "class_count": CLASSIFIER_LABEL_COUNT,
                "label_space_sha256": metadata.label_space_sha256,
            })
        return results

    @torch.no_grad()
    def infer(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对一组 {record_id, audio_path} 做推理，返回与输入等长、同序的结果。

        输出固定 top-5 证据（排序自完整 527 独立 sigmoid 分数的内部排名，
        不持久化完整数组），并携带 record_id（缺失即报错，防止 processor
        静默错配）。

        批间双缓冲：当前 batch 的 H2D/重采样/fbank/模型在 GPU 上执行时，
        下一 batch 的解码已由预取线程提交到 audio worker 池并行进行。
        """
        if not items:
            return []
        record_ids = []
        audio_paths = []
        for item in items:
            record_id = item.get("record_id")
            audio_path = item.get("audio_path")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("AudioTag daemon 输入项缺少合法 record_id")
            if not isinstance(audio_path, str) or not audio_path:
                raise ValueError("AudioTag daemon 输入项缺少 audio_path")
            record_ids.append(record_id)
            audio_paths.append(audio_path)

        prefetch_executor = self._prefetch_executor
        if prefetch_executor is None:
            raise RuntimeError("AudioTag daemon 已关闭")

        all_results: List[Dict[str, Any]] = []
        pending_decode = prefetch_executor.submit(
            self._decode_waves, audio_paths[: self.batch_size]
        )
        for start in range(0, len(audio_paths), self.batch_size):
            batch_started = time.perf_counter()
            decoded, decode_seconds = pending_decode.result()
            next_start = start + self.batch_size
            if next_start < len(audio_paths):
                pending_decode = prefetch_executor.submit(
                    self._decode_waves,
                    audio_paths[next_start : next_start + self.batch_size],
                )

            stage_started = time.perf_counter()
            waves = [wave.to(self.device, non_blocking=True) for wave, _ in decoded]
            resample_started = time.perf_counter()
            waves = [
                self._resample_on_device(wave, source_rate)
                for wave, (_, source_rate) in zip(waves, decoded)
            ]
            resample_seconds = time.perf_counter() - resample_started
            features = self.fbank(waves)
            feature_lengths = [f.size(0) for f in features]
            features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
            feature_lengths = torch.tensor(feature_lengths, device=self.device)
            h2d_fbank_seconds = time.perf_counter() - stage_started - resample_seconds

            stage_started = time.perf_counter()
            encoder_out, encoder_out_lens = self.model.forward_encoder(features, feature_lengths)
            logits = self.model.forward_audio_tagging(encoder_out, encoder_out_lens)
            probs = logits.sigmoid()
            if probs.size(1) != self._expected_class_count:
                raise RuntimeError(
                    f"AudioTag 模型类别维 {probs.size(1)} 与 label dict "
                    f"{self._expected_class_count} 不一致"
                )
            model_seconds = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            batch_results = self._format_top5(
                probs, record_ids[start : start + self.batch_size]
            )
            all_results.extend(batch_results)
            serialize_seconds = time.perf_counter() - stage_started
            self._record_timing(
                len(decoded),
                decode=decode_seconds,
                resample=resample_seconds,
                h2d_fbank=h2d_fbank_seconds,
                model=model_seconds,
                serialize=serialize_seconds,
                total=time.perf_counter() - batch_started,
            )

        if len(all_results) != len(items):
            raise RuntimeError(
                f"AudioTag daemon 输出数量 {len(all_results)} != 输入数量 {len(items)}"
            )
        for result, item in zip(all_results, items):
            if result.get("record_id") != item.get("record_id"):
                raise RuntimeError("AudioTag daemon 输出 record_id 与输入不一致")
        return all_results

    def _handle_infer(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        try:
            command_record_ids = _validate_record_ids(
                cmd.get("record_ids"), "infer 命令"
            )
        except ValueError as exc:
            return _error_response(
                "internal", "protocol_invalid", str(exc), []
            )

        input_path = cmd.get("input_path")
        output_path = cmd.get("output_path")
        if not isinstance(input_path, str) or not input_path:
            return _error_response(
                "infra", "request_io_failed",
                "infer 命令需要非空 input_path", command_record_ids,
            )
        if not isinstance(output_path, str) or not output_path:
            return _error_response(
                "infra", "request_io_failed",
                "infer 命令需要非空 output_path", command_record_ids,
            )

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception as exc:
            logging.exception("Infer request read failed")
            return _error_response(
                "infra", "request_io_failed", str(exc), command_record_ids
            )

        try:
            if not isinstance(items, list) or not items:
                raise ValueError("input JSON 必须是非空 object list")
            for index, item in enumerate(items):
                if not isinstance(item, dict) or set(item) != {
                    "record_id", "audio_path"
                }:
                    raise ValueError(
                        f"input JSON[{index}] 必须精确包含 record_id/audio_path"
                    )
                if not isinstance(item["audio_path"], str) or not item["audio_path"]:
                    raise ValueError(
                        f"input JSON[{index}].audio_path 必须是非空字符串"
                    )
            manifest_record_ids = _validate_record_ids(
                [item["record_id"] for item in items], "input JSON"
            )
            if manifest_record_ids != command_record_ids:
                raise ValueError(
                    "infer 命令 record_ids 与 input JSON record_ids 不同序一致"
                )
        except ValueError as exc:
            return _error_response(
                "internal", "protocol_invalid", str(exc), command_record_ids
            )

        # request 文件成功读取后，推理身份始终以 manifest 中的 ID 为准。
        record_ids = manifest_record_ids
        try:
            results = self.infer(items)
        except _AudioStorageUnavailableError as exc:
            logging.exception("Infer audio storage unavailable")
            return _error_response(
                "infra", "audio_storage_unavailable", str(exc), record_ids
            )
        except _AudioDecodeError as exc:
            logging.exception("Infer audio decode failed")
            return _error_response("content", "audio_decode_failed", str(exc), record_ids)
        except Exception as exc:
            logging.exception("Infer failed")
            return _error_response("internal", "unexpected_exception", str(exc), record_ids)
        try:
            # 紧凑 JSON：每 batch 一次文件往返，indent 只增加体积无任何消费方。
            payload = json.dumps(results, ensure_ascii=False, allow_nan=False)
            write_text_files_transactionally(
                [(Path(output_path), lambda handle: handle.write(payload))],
                overwrite=True,
            )
        except Exception as exc:
            logging.exception("Infer result write failed")
            return _error_response("infra", "request_io_failed", str(exc), record_ids)
        return {"status": "ok"}

    def run(self) -> None:
        logging.info("AudioTag daemon ready, waiting for commands on stdin")
        try:
            self._run_loop()
        finally:
            self.close()
        logging.info("AudioTag daemon exiting")

    def _run_loop(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError as e:
                self._reply(
                    _error_response("internal", "protocol_invalid", str(e), [])
                )
                continue
            if not isinstance(cmd, dict):
                self._reply(
                    _error_response(
                        "internal",
                        "protocol_invalid",
                        "daemon 命令 JSON 必须是 object",
                        [],
                    )
                )
                continue

            cmd_type = cmd.get("cmd")
            if cmd_type == "infer":
                response = self._handle_infer(cmd)
            elif cmd_type == "ping":
                response = {"status": "ok", "message": "pong"}
            elif cmd_type == "exit":
                self._reply({"status": "ok", "message": "bye"})
                break
            else:
                response = {"status": "error", "message": f"Unknown cmd: {cmd_type}"}
            self._reply(response)

    def close(self) -> None:
        for attr in ("_audio_executor", "_prefetch_executor"):
            executor = getattr(self, attr, None)
            if executor is not None:
                setattr(self, attr, None)
                executor.shutdown(wait=True)
        self._flush_timing(force=True)

    def _reply(self, response: Dict[str, Any]) -> None:
        sys.stdout.write(
            json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n"
        )
        sys.stdout.flush()


def main() -> int:
    parser = get_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=level,
        stream=sys.stderr,
    )

    # 必须早于模型和并行计算初始化；生产环境设置失败直接终止。
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(args.interop_threads)
    logging.info(
        "PyTorch threads: intra_op=%d inter_op=%d audio_workers=%d",
        torch.get_num_threads(), torch.get_num_interop_threads(), args.audio_workers,
    )

    daemon = AudioTagDaemon(
        checkpoint=args.checkpoint,
        label_dict_path=args.label_dict,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        device=args.device,
        audio_workers=args.audio_workers,
        timing_log_interval=args.timing_log_interval,
    )
    daemon.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
