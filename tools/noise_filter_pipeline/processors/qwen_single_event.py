"""
C1: Qwen 人声候选三分类。

使用 Qwen2.5-Omni 直接"听"每段音频，分为
Speech（可识别的个体人声或人体发声，拒绝）/ Babble（不可懂背景群体人声候选）/
Noise（普通噪声候选）三类；最终准入仅由下游 C3 admission policy 决定。当前 loader 不支持
Qwen3-Omni。对应 Hive pipeline/code/03_filter_single_event_qwen.py。

特性：
  - 延迟加载模型，缺失 transformers/qwen_omni_utils 时给出清晰报错
  - CUDA OOM 自动 batch 折半
  - 磁盘缓存，支持断点续跑
  - 支持 dry_run 模式用于测试与回归验证
  - 区分内容级失败与基础设施失败：
      * 内容级（speech/babble/noise/不可解析输出）为终态，写入 checkpoint
      * 基础设施失败（无原始输出）标记 status=failed、不写 checkpoint，
        resume 时自动重试，stage 结束时有汇总告警
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import threading
from typing import Any, Dict, List, Optional

from ..core.checkpoint import CheckpointTracker
from ..core.execution_contract import write_stage_c_processor_mode
from ..core.manifest_io import (
    AudioRecord,
    read_manifest_with_progress,
)
from ..core.manifest_per_dataset import PerDatasetManifestWriter, init_checkpointed_output
from .llm_base import LLMBatchProcessor, extract_assistant_reply

logger = logging.getLogger(__name__)

# C1/C2 共用同一模型实例，避免同进程中重复加载。
_MODEL_CACHE: Dict[tuple, "QwenSingleEventModel"] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_DEFAULT_MAX_NEW_TOKENS = 8


def _ensure_cudnn_usable() -> None:
    """探测 cuDNN 可用性，不可用时禁用并回退到原生 CUDA 卷积。

    现象：torch 2.10+cu128 自带的 cuDNN 9.x 与过旧的 NVIDIA 驱动（如 555.x）
    不匹配时，任何 conv 前向都会抛 CUDNN_STATUS_NOT_INITIALIZED——Qwen
    音频编码器的 conv1d 首当其冲。禁用 cuDNN 后 conv 回退到原生 CUDA 实现，
    LLM 主体的矩阵运算走 cuBLAS 不受影响。
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available() or not torch.backends.cudnn.enabled:
        return
    try:
        probe = torch.nn.Conv1d(1, 1, 3).cuda()
        probe(torch.randn(1, 1, 64, device="cuda"))
    except RuntimeError as e:
        if "CUDNN" not in str(e) and "cuDNN" not in str(e):
            raise
        torch.backends.cudnn.enabled = False
        logger.warning(
            "cuDNN 不可用（%s），已禁用并回退到原生 CUDA 卷积；"
            "建议升级 NVIDIA 驱动以恢复 cuDNN 加速", e,
        )


class _QwenSystemPromptWarningFilter(logging.Filter):
    """丢弃 Qwen processor 在纯文本任务中发出的 TTS system prompt warning。

    C1 始终 ``return_audio=False``，且加载后会禁用 talker；该 warning 对
    当前推理无影响。Transformers 用 ``logging.warning`` 直写 root logger，
    无法按 logger 名屏蔽，只能按消息内容过滤。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "audio output may not work as expected" not in record.getMessage()


def _suppress_qwen_system_prompt_warning() -> None:
    root = logging.getLogger()
    for f in root.filters:
        if isinstance(f, _QwenSystemPromptWarningFilter):
            return
    root.addFilter(_QwenSystemPromptWarningFilter())


def get_qwen_model(
    model_path: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    cache_dir: Optional[str] = None,
    qwen_model_sha256: str = "unset:qwen_model_sha256",
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
) -> "QwenSingleEventModel":
    """按已验证模型 SHA 与关键参数返回进程内唯一的 Qwen 实例。"""
    key = (
        model_path,
        device_map,
        str(torch_dtype),
        cache_dir,
        qwen_model_sha256,
        max_new_tokens,
    )
    with _MODEL_CACHE_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = QwenSingleEventModel(
                model_path=model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                qwen_model_sha256=qwen_model_sha256,
                max_new_tokens=max_new_tokens,
            )
        return _MODEL_CACHE[key]


# parse_responses 返回值约定：
#   "speech"/"babble"/"noise" — 内容级终态分类
#   UNPARSEABLE — 模型有原始输出但无法解析（内容级终态，拒绝并写 checkpoint）
#   None — 无原始输出（基础设施失败，不写 checkpoint，resume 时重试）
UNPARSEABLE = "__unparseable__"

DEFAULT_PROMPT = (
    "You are performing C1 triage for a speech-enhancement noise dataset. "
    "Classify the audio into exactly one category, with priority: Speech > Babble > Noise. "
    "C1 only produces candidates; final admission is decided only by downstream C3 admission policy.\n\n"
    "- Speech: Any identifiable individual human voice or human-produced vocalization that "
    "must be rejected, even if faint, short, unintelligible, or partially masked. This "
    "includes talking, singing, humming, laughing, crying, shouting, breathing, wheezing, "
    "snoring, gasping, panting, snorting, sniffing, coughing, throat clearing, sneezing, "
    "hiccuping, burping, chewing, gargling, whistling, yawning, sighing, groaning, grunting, "
    "moaning, cheering, and any identifiable single or few speakers.\n"
    "- Babble candidate: Only unstructured, unintelligible background group voices or crowd "
    "murmur with many overlapping speech-like voices and no identifiable individual word, "
    "phrase, voice, or vocalization. Pass this candidate downstream for strict validation; "
    "Babble is not final admission.\n"
    "- Noise candidate: Ordinary mechanical, environmental, or non-human noise with no "
    "human voice or human-produced vocalization detected. Pass this candidate downstream; "
    "Noise is not final admission.\n\n"
    "If any individual human voice or vocalization is identifiable, choose Speech. If only "
    "unintelligible background group voices are present, choose Babble. When uncertain "
    "whether human vocal activity is present, choose Speech or Babble rather than Noise.\n\n"
    "Return only one word: Speech, Babble, or Noise."
)


class QwenSingleEventModel:
    """封装 Qwen-Omni 模型加载与 batch 推理。"""

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        cache_dir: Optional[str] = None,
        qwen_model_sha256: str = "unset:qwen_model_sha256",
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    ):
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("C1 max_new_tokens 必须是严格正整数")
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self.qwen_model_sha256 = qwen_model_sha256
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._process_mm_info = None
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            # 加载逻辑必须持锁执行，否则并发线程会重复加载模型（双份显存）
            try:
                from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
                from qwen_omni_utils import process_mm_info
            except ImportError as e:
                raise RuntimeError(
                    "C1 Qwen 推理依赖 transformers / accelerate / qwen_omni_utils，"
                    f"请先安装 Qwen-Omni 环境（见 README「环境安装」章节）: {e}"
                ) from e

            logger.info("Loading Qwen model from %s", self.model_path)
            # cuDNN 与驱动不匹配时先回退，否则音频编码器 conv1d 会直接崩溃
            _ensure_cudnn_usable()
            # 屏蔽 Qwen processor 在纯文本任务中提示的 TTS system prompt warning。
            _suppress_qwen_system_prompt_warning()
            # C1 只做音频理解，不需要 TTS speaker embedding；跳过 spk_dict.pt 加载
            # 以避免低版本 torch 因 CVE-2025-32434 限制无法 torch.load .pt 文件。
            # 注意：这会全局修改 Qwen2_5OmniForConditionalGeneration 的类行为；
            # 本进程仅用于 C1 音频理解，不启用 TTS，因此可接受。
            Qwen2_5OmniForConditionalGeneration.load_speakers = lambda self, path: None
            self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                cache_dir=self.cache_dir,
            )
            # generate 即使 return_audio=False 也会校验默认 speaker；跳过 speaker
            # 文件加载后注入占位条目，满足该兼容检查。纯文本路径不会读取其值。
            if not getattr(self._model, "speaker_map", None):
                try:
                    self._model.speaker_map = {"Chelsie": None}
                except AttributeError:
                    # 测试替身或极简模型可能不允许动态属性；真实 Qwen 模型允许。
                    pass
            # C1 永远只返回文本；按官方建议卸载 talker/token2wav，降低显存
            # 占用。保留 return_audio=False 约束，未来需要 TTS 时必须移除此调用。
            disable_talker = getattr(self._model, "disable_talker", None)
            if callable(disable_talker):
                disable_talker()
                logger.info("Qwen talker disabled for text-only inference")
            # 固定慢版 image processor，避免 Transformers 默认切换 fast processor
            # 导致预处理行为和输出出现无意漂移。
            self._processor = Qwen2_5OmniProcessor.from_pretrained(
                self.model_path,
                cache_dir=self.cache_dir,
                use_fast=False,
            )
            self._process_mm_info = process_mm_info
            logger.info("Qwen model loaded")

    def batch_query_raw(
        self,
        items: List[Dict[str, Any]],
        timeout_sec: Optional[float] = None,
    ) -> Optional[List[str]]:
        """
        通用 batch 推理：items 每个元素含 audio_path 与 prompt。
        返回原始文本列表（未经解析）。

        Args:
            timeout_sec: 生成超时（秒），None 表示使用环境变量
                ``USE_TRAIN_QWEN_GENERATE_TIMEOUT_SEC`` 或不限时。
        """
        self._load()
        if timeout_sec is None:
            try:
                timeout_sec = float(os.environ.get("USE_TRAIN_QWEN_GENERATE_TIMEOUT_SEC", "0") or "0")
            except ValueError:
                timeout_sec = 0.0
        timeout_sec = timeout_sec if timeout_sec > 0 else None
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": item["audio_path"]},
                        {"type": "text", "text": item["prompt"]},
                    ],
                }
            ]
            for item in items
        ]

        use_audio_in_video = True
        text = self._processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = self._process_mm_info(
            conversations, use_audio_in_video=use_audio_in_video
        )
        inputs = self._processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(self._model.device).to(self._model.dtype)
        generate_kwargs = {
            **inputs,
            "use_audio_in_video": use_audio_in_video,
            "return_audio": False,
            "thinker_max_new_tokens": self.max_new_tokens,
        }
        if timeout_sec is not None:
            generate_kwargs["max_time"] = timeout_sec
        text_ids = self._model.generate(**generate_kwargs)
        return self._processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


class QwenSingleEventProcessor(LLMBatchProcessor):
    """C1: Qwen 单事件过滤 Processor。"""

    def __init__(self, config: Dict[str, Any]):
        for name, default in (
            ("batch_size", 8),
            ("max_retries", 3),
            ("num_workers", 1),
            ("max_consecutive_worker_infra_failures", 3),
            ("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS),
            ("checkpoint_save_interval", 100),
            ("manifest_write_batch_size", 100),
        ):
            value = config.get(name, default)
            if type(value) is not int or value <= 0:
                raise ValueError(f"C1 {name} 必须是严格正整数")
        for name, default in (("dry_run", False), ("enable_checkpoint", True)):
            if type(config.get(name, default)) is not bool:
                raise ValueError(f"C1 {name} 必须是严格 bool")
        self.prompt = config.get("prompt", DEFAULT_PROMPT)
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("C1 prompt 必须是非空字符串")
        expected_prompt_sha256 = hashlib.sha256(
            self.prompt.encode("utf-8")
        ).hexdigest()
        configured_identity = config.get("artifact_fingerprints")
        if configured_identity is None:
            configured_identity = {
                "qwen_model_sha256": "unset:qwen_model_sha256",
                "c1_prompt_sha256": expected_prompt_sha256,
            }
        expected_keys = {"qwen_model_sha256", "c1_prompt_sha256"}
        if not isinstance(configured_identity, dict) or set(configured_identity) != expected_keys:
            raise ValueError("C1 identity 必须精确包含 Qwen model/prompt SHA")
        qwen_sha256 = configured_identity["qwen_model_sha256"]
        if qwen_sha256 != "unset:qwen_model_sha256" and (
            not isinstance(qwen_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", qwen_sha256) is None
        ):
            raise ValueError("C1 qwen_model_sha256 必须是 SHA-256")
        if configured_identity["c1_prompt_sha256"] != expected_prompt_sha256:
            raise ValueError("C1 c1_prompt_sha256 与实际 prompt 不一致")
        self.identity = dict(configured_identity)
        config["artifact_fingerprints"] = dict(self.identity)
        self.max_new_tokens = config.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS)
        config["max_new_tokens"] = self.max_new_tokens
        cache_dir = config.get("cache_dir") or config.get("qwen_cache_dir")
        # 只传相对子目录名，由基类与 cache_dir 拼接；生成 token 上限同时进入
        # processor fingerprint 与 cache key，避免切换上限后复用旧结论。
        cache_subdir = "c1_qwen" if cache_dir else None
        super().__init__(
            config,
            stage_name="C1",
            cache_subdir=cache_subdir,
            qwen_model_sha256=qwen_sha256,
            cache_key_salt=f"max_new_tokens={self.max_new_tokens}",
        )
        for name, value in (
            ("batch_size", self.batch_size),
            ("max_retries", self.max_retries),
            ("num_workers", self.num_workers),
            (
                "max_consecutive_worker_infra_failures",
                self._max_consecutive_worker_infra_failures,
            ),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"C1 {name} 必须是严格正整数")
        for name, value in (
            ("dry_run", self.dry_run),
            ("enable_checkpoint", self.enable_checkpoint),
        ):
            if type(value) is not bool:
                raise ValueError(f"C1 {name} 必须是严格 bool")
        # dry_run 时三类分布：非语音 / 嘈杂人声 / 清晰语音
        # 默认 babble 和 noise 占多数，模拟真实噪声库构成
        self.dry_run_distribution = config.get(
            "dry_run_distribution", {"noise": 0.6, "babble": 0.3, "speech": 0.1}
        )
        self._cache_value_key = "intelligibility"

        # checkpoint / manifest 批量写入，降低百万级文件 I/O
        self.checkpoint_path = config.get("checkpoint_path")
        self.checkpoint_save_interval = config.get("checkpoint_save_interval", 100)
        self.manifest_write_batch_size = config.get("manifest_write_batch_size", 100)
        for name, value in (
            ("checkpoint_save_interval", self.checkpoint_save_interval),
            ("manifest_write_batch_size", self.manifest_write_batch_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"C1 {name} 必须是严格正整数")
        self._retryable_failure_count = 0

    def has_retryable_failures(self) -> bool:
        """是否有未写 checkpoint、可在下次运行重试的推理失败。"""
        return self._retryable_failure_count > 0

    def _load_model(self) -> None:
        if not self.model_path:
            raise ValueError("C1 需要 model_path，请配置 qwen_model 或 model_path")
        self._model = get_qwen_model(
            model_path=self.model_path,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            cache_dir=self.cache_dir,
            qwen_model_sha256=self.qwen_model_sha256,
            max_new_tokens=self.max_new_tokens,
        )

    def build_prompts(self, records: List[AudioRecord]) -> List[str]:
        return [self.prompt] * len(records)

    def parse_responses(self, texts: List[str]) -> List[Optional[str]]:
        """解析 Qwen 输出为 intelligibility 类别。

        仅接受完整单标签，允许大小写、首尾空白、外层单/双引号和一个句末
        标点；解释文本或复合标签均 fail-closed 为 UNPARSEABLE。无原始输出
        （None）仍表示可重试的基础设施失败。
        """
        results: List[Optional[str]] = []
        for text in texts:
            if text is None:
                # 基础设施失败（worker/推理未产出原始文本）：保持 None，可重试
                results.append(None)
                continue
            reply = extract_assistant_reply(text)
            if reply is None:
                results.append(UNPARSEABLE)
                continue

            normalized = reply.strip()
            punctuation_seen = False
            if normalized[-1:] in ".!?":
                normalized = normalized[:-1].rstrip()
                punctuation_seen = True
            if (
                len(normalized) >= 2
                and normalized[0] in ("'", '"')
                and normalized[-1] == normalized[0]
            ):
                normalized = normalized[1:-1].strip()
            if normalized[-1:] in ".!?":
                if punctuation_seen:
                    results.append(UNPARSEABLE)
                    continue
                normalized = normalized[:-1].rstrip()

            match = re.fullmatch(r"speech|babble|noise", normalized, flags=re.IGNORECASE)
            results.append(match.group(0).lower() if match else UNPARSEABLE)
        return results

    def _dry_run_results(self, records: List[AudioRecord]) -> List[str]:
        # per-record seed（sha1(record_id)）：同一记录无论 batch 组成、
        # batch_size、resume 断点如何变化，dry_run 结果都可复现；
        # 用 sha1 而非内置 hash()（PYTHONHASHSEED 随机化），局部 Random 不污染全局状态
        dist = self.dry_run_distribution
        classes = ["noise", "babble", "speech"]
        weights = [float(dist.get(k, 0.0)) for k in classes]
        total_weight = sum(weights)
        if total_weight <= 0:
            weights = [1.0, 1.0, 1.0]
            total_weight = 3.0
        weights = [w / total_weight for w in weights]
        results: List[str] = []
        for rec in records:
            seed = int.from_bytes(hashlib.sha1(rec.record_id.encode("utf-8")).digest()[:8], "big")
            results.append(random.Random(seed).choices(classes, weights=weights)[0])
        return results

    def _flush_pending(
        self,
        pending: List[AudioRecord],
        manifest_writer: PerDatasetManifestWriter,
        ckpt: CheckpointTracker,
        counts: Dict[str, int],
    ) -> None:
        """对 micro-batch 做缓存查询/推理，并批量写出结果。"""
        if not pending:
            return

        results = self.process_records(pending)
        execution_mode = self.config.get(
            "execution_mode", "dry_run" if self.dry_run else "production"
        )
        for rec, cls in zip(pending, results):
            rec.provenance = write_stage_c_processor_mode(
                rec.provenance, "c1", execution_mode
            )
            infra_failed = cls is None
            identity = dict(self.identity)
            if cls == "speech":
                # 可识别的个体人声或人体发声：直接拒绝
                rec.mark_stage(
                    "C1", "rejected",
                    extra={"intelligibility": "speech", **identity},
                )
                rec.rejected_reason = "c1_speech"
                counts["rejected"] += 1
            elif cls in ("babble", "noise"):
                # 仅作为候选下传；最终准入由下游 C3 admission policy 决定
                rec.mark_stage(
                    "C1", "passed", extra={"intelligibility": cls, **identity}
                )
                counts["kept"] += 1
            elif cls == UNPARSEABLE:
                # 模型有输出但无法解析：内容级终态，重试无意义，
                # 标记 rejected 并写 checkpoint，避免每次 resume 重复烧算力
                rec.mark_stage(
                    "C1", "rejected", extra={"reason": "parse_failed", **identity}
                )
                rec.rejected_reason = "c1_parse_failed"
                counts["rejected"] += 1
                counts["parse_failed"] += 1
            else:
                # 基础设施失败（无原始输出）：标记 failed（非终态 rejected），
                # 不写 checkpoint，resume 时自动重试
                rec.mark_stage(
                    "C1", "failed",
                    extra={"reason": "inference_failed", "retryable": True, **identity},
                )
                counts["infra_failed"] += 1
                self._retryable_failure_count += 1
            manifest_writer.append(rec)
            if not infra_failed:
                # 基础设施失败（推理失败）不写入 checkpoint，resume 时可重试；
                # 内容级结论（speech/babble/noise/不可解析）才是终态
                ckpt.mark_done(rec.record_id)

        pending.clear()

    def process(
        self,
        input_manifest: str,
        output_dir: str,
        fingerprint_input: Optional[str] = None,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_manifest = os.path.join(output_dir, "c1_manifest.jsonl")


        manifest_writer, ckpt, prev_records = init_checkpointed_output(
            output_manifest, output_dir, self.config, input_manifest,
            enable_checkpoint=self.enable_checkpoint,
            checkpoint_path=self.checkpoint_path,
            save_interval=self.checkpoint_save_interval,
            manifest_batch_size=self.manifest_write_batch_size,
            stage_label="Stage C1",
            dry_run=self.dry_run,
            fingerprint_input=fingerprint_input,
        )

        total = 0
        counts: Dict[str, int] = {"kept": 0, "rejected": 0, "parse_failed": 0, "infra_failed": 0}
        pending: List[AudioRecord] = []
        # flush 阈值随 worker 数缩放：多卡时每 worker 凑满一个 batch，
        # 避免 batch_size < num_workers 时部分 GPU 空转
        flush_threshold = self.batch_size * max(1, self.num_workers)

        try:
            with manifest_writer:
                for rec in read_manifest_with_progress(input_manifest, self.stage_name):
                    total += 1
                    if self.should_skip(rec):
                        manifest_writer.append(rec)
                        continue
                    if ckpt.is_done(rec.record_id):
                        prev = prev_records.get(rec.record_id)
                        if prev is not None:
                            # 上次输出仍在：恢复已处理记录
                            manifest_writer.append(prev)
                            continue
                        # 上次输出缺失（如被清理）：回退为重新处理，避免丢失结果
                    pending.append(rec)
                    if len(pending) >= flush_threshold:
                        self._flush_pending(pending, manifest_writer, ckpt, counts)

                self._flush_pending(pending, manifest_writer, ckpt, counts)
                ckpt.save(force=True)
        finally:
            self.close()

        logger.info(
            "C1 处理完成：%d 条，保留（noise/babble）%d，拒绝（speech/parse失败）%d，"
            "基础设施失败（可重试）%d，输出 %s",
            total,
            counts["kept"],
            counts["rejected"],
            counts["infra_failed"],
            output_manifest,
        )
        if counts["infra_failed"] > 0:
            logger.warning(
                "C1 有 %d 条因基础设施失败未处理（输出中标记 status=failed、未写 checkpoint）。"
                "Stage C 将中止且不写完成 marker；重跑会自动重试这些记录。",
                counts["infra_failed"],
            )
        return output_dir
