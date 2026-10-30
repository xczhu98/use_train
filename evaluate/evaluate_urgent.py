#!/usr/bin/env python3
"""URGENT2026 comprehensive evaluation script.

Builds on standard audio quality metrics (PESQ/STOI/ESTOI/SI-SDR), adds URGENT2026
downstream impact metrics, provides unified evaluation entry and normalized total score.

Metric categories:
  +---------------------------------------------------------------+
  | A. Audio Quality                                              |
  |    Intrusive:  PESQ, STOI, ESTOI, SI-SDR                     |
  |    Non-intrusive: DNSMOS, NISQA, UTMOS, SCOREQ                |
  +---------------------------------------------------------------+
  | B. Downstream Impact                                          |
  |    WER / CER          (ASR, OWSM v3.1)                       |
  |    SpeakerSimilarity  (speaker embedding cosine similarity)  |
  |    EmotionSimilarity  (emotion embedding cosine similarity)  |
  |    PhonemeSimilarity  (phoneme sequence Levenshtein sim)     |
  |    SpeechBERTScore    (speech BERT precision score)          |
  |    LAcc               (language identification accuracy)     |
  +---------------------------------------------------------------+
  | C. Normalized Average                                         |
  |    Normalize DNSMOS/NISQA/UTMOS/SCOREQ/PESQ/ESTOI to [0,1]   |
  |    Report available dimensions and per-metric sample coverage;|
  |    official requires 6/6 dimensions and complete coverage     |
  +---------------------------------------------------------------+

All metric groups optional, controlled by CLI switches; install hints when deps missing.

Usage:
    # Minimal evaluation (audio quality)
    python evaluate/evaluate_urgent.py --ref_scp data/ref.scp --inf_scp data/enh.scp

    # Full URGENT evaluation
    python evaluate/evaluate_urgent.py --ref_scp data/ref.scp --inf_scp data/enh.scp \\
        --dnsmos --nisqa --utmos --scoreq \\
        --wer --meta_tsv data/meta.tsv --utt2lang data/utt2lang \\
        --speaker_similarity --emotion_similarity \\
        --phoneme_similarity --speechbert --lid \\
        --output_dir results/urgent_eval

    # Downstream metrics only
    python evaluate/evaluate_urgent.py --inf_scp data/enh.scp \\
        --wer --meta_tsv data/meta.tsv --utt2lang data/utt2lang \\
        --utmos --dnsmos --output_dir results/

Dependency installation:
    # Basic (audio quality)
    pip install pesq pystoi soundfile soxr

    # Audio Quality models (all from local third_party_scripts/, no extra install)
    # DNSMOS/UTMOS/NISQA/SCOREQ: models already in third_party_scripts/

    # WER
    pip install espnet espnet_model_zoo Levenshtein

    # Speaker Similarity
    pip install espnet espnet_model_zoo

    # Emotion Similarity
    git clone https://github.com/ftshijt/emotion2vec && pip install -e emotion2vec/

    # Phoneme Similarity
    pip install transformers Levenshtein

    # SpeechBERTScore
    pip install discrete-speech-metrics

    # LID
    pip install espnet espnet_model_zoo iso639
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("evaluate_urgent")

# Ensure project root is on sys.path so "from evaluate import ..." works
# when running as: python evaluate/evaluate_urgent.py
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ──────────────────────────────────────────────────────────────
# A. Audio Quality Metrics (reuse from evaluate.py)
# ──────────────────────────────────────────────────────────────
from core.metrics import (
    _resample_np,
    compute_metrics_batch,
    load_nisqa,
    load_scoreq,
)
from evaluate import load_audio, load_dnsmos, load_utmos
from evaluate.common import write_normalized_score
from evaluate.urgent_score import (
    NORMALIZATION_RANGES,
    NORMALIZED_METRICS,
    build_expected_metric_coverage,
    build_normalized_score,
    normalize_metric,
)
from tools.shared.scp_io import parse_scp_fields


# ──────────────────────────────────────────────────────────────
# A (additional): NISQA / SCOREQ
# ──────────────────────────────────────────────────────────────
# These use local models from third_party_scripts/ (no urgent2026_challenge_track1-main dependency)


# 单路 enhanced audio 评分固定使用 NR（no-reference）模式。
SCOREQ_MODE = "nr"
SCOREQ_DATA_DOMAIN = "natural"


def _load_scoreq_for_eval(has_ref: bool, mode: str = SCOREQ_MODE,
                          data_domain: str = SCOREQ_DATA_DOMAIN):
    """加载 SCOREQ 并固定其评估谱系（JAE-L02）。

    ref 模式必须提供 clean reference（--ref_scp）；缺 reference 时显式失败，
    不允许静默退回 NR 还把结果标成 REF 指标。
    """
    if mode == "ref" and not has_ref:
        logger.error("SCOREQ ref 模式需要 clean reference，请提供 --ref_scp")
        sys.exit(1)
    return load_scoreq(mode=mode, data_domain=data_domain)


# ──────────────────────────────────────────────────────────────
# B. Downstream Impact Metrics
# ──────────────────────────────────────────────────────────────

# B1: WER / CER
_URGENT_METRICS_DIR = (
    Path(__file__).parent.parent / "urgent2026_challenge_track1-main" / "evaluation_metrics"
)
# 仓库内 vendored 副本：官方 checkout 缺失时的本地 fallback（根盘已满，不再放外部 checkout）。
_VENDORED_OWSM_UTILS = Path(__file__).parent / "owsm_utils.py"


def _require_owsm_utils() -> Path:
    """定位 ``owsm_utils.py``：优先官方 checkout，缺失时回退到仓库内 vendored 副本。"""
    module_path = _URGENT_METRICS_DIR / "owsm_utils.py"
    if module_path.is_file():
        return module_path
    if _VENDORED_OWSM_UTILS.is_file():
        return _VENDORED_OWSM_UTILS
    raise ImportError(
        "WER requires owsm_utils.py:\n"
        f"  missing official: {module_path}\n"
        f"  missing vendored: {_VENDORED_OWSM_UTILS}\n"
        "  请将 urgent2026_challenge_track1-main 放到仓库根目录，"
        "或恢复 evaluate/owsm_utils.py（vendored 副本）"
    )


def load_wer_model(device: str = "cpu"):
    _require_owsm_utils()
    try:
        from espnet2.bin.s2t_inference import Speech2Text
        from espnet2.text.cleaner import TextCleaner
    except ImportError:
        raise ImportError(
            "WER requires espnet2:\n"
            "  pip install espnet espnet_model_zoo Levenshtein"
        )
    model = Speech2Text.from_pretrained(
        model_tag="espnet/owsm_v3.1_ebf",
        device=device,
        task_sym="<asr>",
        beam_size=5,
        predict_time=False,
    )
    textcleaner = TextCleaner("whisper_basic")
    return model, textcleaner


def compute_wer(
    model, textcleaner, ref_txt: str, inf_audio: np.ndarray, lang_id: str, fs: int
) -> dict:
    """Compute WER and CER."""
    from Levenshtein import opcodes

    owsm_dir = _require_owsm_utils().parent
    sys.path.insert(0, str(owsm_dir))
    try:
        from owsm_utils import owsm_predict
    finally:
        sys.path.pop(0)

    if ref_txt == "<not-available>":
        return {"WER": {}, "CER": {}}
    if fs != 16000:
        inf_audio = _resample_np(inf_audio, fs, 16000)
        fs = 16000

    with torch.no_grad():
        inf_txt = owsm_predict(
            model, inf_audio, fs,
            src_lang=lang_id, beam_size=5,
            long_form=len(inf_audio) > 30 * fs,
        )

    ref_txt = textcleaner(ref_txt)
    inf_txt = textcleaner(inf_txt)

    ref_words = ref_txt.split()
    inf_words = inf_txt.split()
    ret_wer = {"delete": 0, "insert": 0, "replace": 0, "equal": 0}
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, inf_words):
        if op == "insert":
            ret_wer[op] += inf_et - inf_st
        else:
            ret_wer[op] += ref_et - ref_st

    ref_chars = list(ref_txt)
    inf_chars = list(inf_txt)
    ret_cer = {"delete": 0, "insert": 0, "replace": 0, "equal": 0}
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_chars, inf_chars):
        if op == "insert":
            ret_cer[op] += inf_et - inf_st
        else:
            ret_cer[op] += ref_et - ref_st

    return {"WER": ret_wer, "CER": ret_cer}


# B2: Speaker Similarity
def load_speaker_model(device: str = "cpu"):
    try:
        from espnet2.bin.spk_inference import Speech2Embedding
    except ImportError:
        raise ImportError(
            "SpeakerSimilarity requires espnet2:\n"
            "  pip install espnet espnet_model_zoo"
        )
    model = Speech2Embedding.from_pretrained(
        model_tag="espnet/voxcelebs12_rawnet3", device=device
    )
    model.spk_model.eval()
    return model


def compute_speaker_similarity(model, ref: np.ndarray, inf: np.ndarray, fs: int) -> float:
    if fs != 16000:
        ref = _resample_np(ref, fs, 16000)
        inf = _resample_np(inf, fs, 16000)
    with torch.no_grad():
        ref_emb = model(ref)
        inf_emb = model(inf)
        return torch.cosine_similarity(ref_emb, inf_emb, dim=-1).item()


# B3: Emotion Similarity
def load_emotion_model(device: str = "cpu"):
    try:
        from emo2vec_versa.emo2vec_class import EMO2VEC
    except ImportError:
        raise ImportError(
            "EmotionSimilarity requires emotion2vec:\n"
            "  git clone https://github.com/ftshijt/emotion2vec\n"
            "  cd emotion2vec && pip install -e ./"
        )
    model_path = "emotion2vec_base.pt"
    if not Path(model_path).exists():
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="emotion2vec/emotion2vec_base",
            filename="emotion2vec_base.pt",
            local_dir=str(Path.cwd()),
        )
    use_gpu = "cuda" in device
    return EMO2VEC(model_path, use_gpu=use_gpu)


def compute_emotion_similarity(model, ref: np.ndarray, inf: np.ndarray, fs: int) -> float:
    if fs != 16000:
        ref = _resample_np(ref, fs, 16000)
        inf = _resample_np(inf, fs, 16000)
    emb_ref = model.extract_feature(ref, fs=16000)
    emb_inf = model.extract_feature(inf, fs=16000)
    return float(np.dot(emb_ref, emb_inf) / (np.linalg.norm(emb_ref) * np.linalg.norm(emb_inf) + 1e-8))


# B4: Phoneme Similarity
def load_phoneme_model(device: str = "cpu"):
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    except ImportError:
        raise ImportError(
            "PhonemeSimilarity requires transformers:\n"
            "  pip install transformers Levenshtein"
        )
    checkpoint = os.environ.get(
        "PHONEME_MODEL_DIR", "facebook/wav2vec2-lv-60-espeak-cv-ft"
    )
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint).to(device)
    model.eval()
    return processor, model, device


def compute_phoneme_similarity(models, ref: np.ndarray, inf: np.ndarray, fs: int) -> float:
    processor, model, device = models
    if fs != 16000:
        ref = _resample_np(ref, fs, 16000)
        inf = _resample_np(inf, fs, 16000)

    def _predict(audio):
        input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
        if len(input_values.shape) == 3:
            input_values = input_values.squeeze(0)
        logits = model(input_values.to(device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0].split(" ")

    with torch.no_grad():
        ref_phonemes = _predict(ref)
        inf_phonemes = _predict(inf)

    try:
        # Levenshtein.distance() expects strings → use character-level distance
        from Levenshtein import distance as lev_distance
        ref_str = " ".join(ref_phonemes)
        inf_str = " ".join(inf_phonemes)
        dist = lev_distance(ref_str, inf_str)
        return 1.0 - dist / max(len(ref_str), 1)
    except ImportError:
        # editdistance.eval() works on arbitrary sequences → use token-level distance
        import editdistance
        dist = editdistance.eval(ref_phonemes, inf_phonemes)
        max_len = max(len(ref_phonemes), len(inf_phonemes), 1)
        return 1.0 - dist / max_len


# B5: SpeechBERTScore
def load_speechbert_model(device: str = "cpu"):
    try:
        from discrete_speech_metrics import SpeechBERTScore as SBS
    except ImportError:
        raise ImportError(
            "SpeechBERTScore requires:\n"
            "  pip install discrete-speech-metrics"
        )
    return SBS(sr=16000, model_type="mhubert-147", layer=8, use_gpu="cuda" in device)


def compute_speechbert_score(model, ref: np.ndarray, inf: np.ndarray, fs: int) -> float:
    if fs != 16000:
        ref = _resample_np(ref, fs, 16000)
        inf = _resample_np(inf, fs, 16000)
    with torch.no_grad():
        precision, recall, f1 = model.score(ref, inf)
    return float(precision)


# B6: Language Identification
def load_lid_model(device: str = "cpu"):
    try:
        from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
    except ImportError:
        raise ImportError(
            "LID requires espnet2:\n"
            "  pip install espnet espnet_model_zoo iso639"
        )
    return Speech2TextGreedySearch.from_pretrained(
        "espnet/owsm_ctc_v4_1B",
        device=device,
        generate_interctc_outputs=False,
        lang_sym="<nolang>",
        task_sym="<asr>",
    )


def compute_lid(model, inf: np.ndarray, fs: int, ref_lang: str) -> int:
    if fs != 16000:
        inf = _resample_np(inf, fs, 16000)
    text, tokens, tokens_int, text_nospecial, hyp = model(inf)[0]
    pred_lang = tokens[0].replace("<", "").replace(">", "")
    ref_lang_clean = ref_lang.replace("<", "").replace(">", "")
    return int(pred_lang == ref_lang_clean)


# ──────────────────────────────────────────────────────────────
# C. Score Normalization (URGENT2026 ranking formula)
# ──────────────────────────────────────────────────────────────
NORMALIZED_AVERAGE_METRICS = NORMALIZED_METRICS


def compute_normalized_average(means: dict) -> dict:
    """兼容旧调用；这里只表达六个 score dimensions，不代表样本覆盖完整。"""
    score = build_normalized_score(means, None)
    dimensions = score["dimensions"]
    return {
        "value": score["value"],
        "available": dimensions["valid"],
        "required": dimensions["required"],
        "missing": dimensions["missing"],
        "complete": dimensions["valid"] == dimensions["required"],
    }


# ──────────────────────────────────────────────────────────────
# Helper: WER aggregation
# ──────────────────────────────────────────────────────────────
def aggregate_wer(wer_results: list[dict]) -> Optional[float]:
    """Aggregate WER."""
    totals = {"delete": 0, "insert": 0, "replace": 0, "equal": 0}
    valid = 0
    for r in wer_results:
        if not r:
            continue
        for k in totals:
            totals[k] += r.get(k, 0)
        valid += 1
    if valid == 0:
        return None
    denom = totals["delete"] + totals["replace"] + totals["equal"]
    if denom == 0:
        return None
    return (totals["delete"] + totals["insert"] + totals["replace"]) / denom


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def _read_audio_scp(scp_path: str | Path) -> dict[str, str]:
    """读取严格的 2/4 列音频 SCP，并拒绝重复 uid。"""
    result: dict[str, str] = {}
    with open(scp_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = parse_scp_fields(
                line, scp_path, lineno, allowed_columns=(2, 4)
            )
            uid, audio_path = fields[0], fields[-1]
            if uid in result:
                raise ValueError(f"{scp_path}:{lineno}: 重复 uid: {uid!r}")
            result[uid] = audio_path
    return result


def parse_args():
    p = argparse.ArgumentParser(
        description="URGENT2026 evaluation (audio quality + downstream impact + normalized score)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g_io = p.add_argument_group("Input/Output")
    g_io.add_argument("--ref_scp", help="Clean reference SCP (required for intrusive metrics)")
    g_io.add_argument("--inf_scp", required=True, help="Enhanced SCP")
    g_io.add_argument("--output_dir", default="results/urgent_eval", help="Output directory")
    g_io.add_argument("--device", default="cuda:0", help="Inference device")

    g_preset = p.add_argument_group("Presets (convenience)")
    g_preset.add_argument(
        "--all",
        action="store_true",
        help="Enable all metrics (audio quality + downstream). WER needs --meta_tsv; LID needs --utt2lang.",
    )
    g_preset.add_argument(
        "--audio_quality",
        action="store_true",
        help="Enable all 语音信号质量 metrics: DNSMOS, NISQA, UTMOS, SCOREQ (intrusive PESQ/STOI/ESTOI/SI-SDR need --ref_scp).",
    )
    g_preset.add_argument(
        "--downstream",
        action="store_true",
        help="Enable all 语音后端任务 metrics: WER, SpeakerSimilarity, EmotionSimilarity, PhonemeSimilarity, SpeechBERTScore, LAcc. WER and LID need --utt2lang (LID uses it as language labels). WER also needs --meta_tsv.",
    )

    g_aq = p.add_argument_group("A. Audio quality (intrusive default, non-intrusive optional)")
    g_aq.add_argument("--dnsmos", action="store_true", help="Enable DNSMOS (local ONNX)")
    g_aq.add_argument("--nisqa", action="store_true", help="Enable NISQA (local PyTorch)")
    g_aq.add_argument("--utmos", action="store_true", help="Enable UTMOS (local torch)")
    g_aq.add_argument("--scoreq", action="store_true", help="Enable SCOREQ (local ONNX)")

    g_ds = p.add_argument_group("B. Downstream impact (all optional)")
    g_ds.add_argument("--wer", action="store_true", help="Enable WER/CER")
    g_ds.add_argument("--meta_tsv", help="WER metadata (transcript scp/tsv)")
    g_ds.add_argument("--utt2lang", help="Language scp (required for WER and LID)")
    g_ds.add_argument("--speaker_similarity", action="store_true", help="Enable speaker similarity")
    g_ds.add_argument("--emotion_similarity", action="store_true", help="Enable emotion similarity")
    g_ds.add_argument("--phoneme_similarity", action="store_true", help="Enable phoneme similarity")
    g_ds.add_argument("--speechbert", action="store_true", help="Enable SpeechBERTScore")
    g_ds.add_argument("--lid", action="store_true", help="Enable language ID (LAcc), uses --utt2lang")

    g_misc = p.add_argument_group("Misc")
    g_misc.add_argument("--nsplits", type=int, default=1, help="Number of parallel shards")
    g_misc.add_argument("--job", type=int, default=1, help="Current shard index (1-based)")

    return p.parse_args()


def _apply_presets(args):
    """Apply --all / --audio_quality / --downstream to individual flags."""
    if args.all:
        args.audio_quality = True
        args.downstream = True
    if args.audio_quality:
        args.dnsmos = True
        args.nisqa = True
        args.utmos = True
        args.scoreq = True
    if args.downstream:
        args.wer = True
        args.speaker_similarity = True
        args.emotion_similarity = True
        args.phoneme_similarity = True
        args.speechbert = True
        args.lid = True


def main():
    args = parse_args()
    _apply_presets(args)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw):
            return x

    device = args.device if torch.cuda.is_available() else "cpu"
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Input validation ──
    if not Path(args.inf_scp).exists():
        logger.error(f"inf_scp not found: {args.inf_scp}")
        sys.exit(1)

    # ── Load enhanced SCP ──
    inf_data = _read_audio_scp(args.inf_scp)

    # ── Load reference SCP (optional, but explicit input must be valid) ──
    ref_data = {}
    ref_pairing_coverage = None
    if args.ref_scp:
        ref_path = Path(args.ref_scp)
        if not ref_path.is_file() or ref_path.is_symlink():
            logger.error("ref_scp must be an existing non-symlink regular file: %s", ref_path)
            sys.exit(1)
        ref_data = _read_audio_scp(ref_path)
        paired_uids = set(inf_data) & set(ref_data)
        ref_pairing_coverage = {
            "enhanced_uids": len(inf_data),
            "reference_uids": len(ref_data),
            "paired_uids": len(paired_uids),
            "unpaired_enhanced_uids": len(set(inf_data) - set(ref_data)),
            "unpaired_reference_uids": len(set(ref_data) - set(inf_data)),
        }
        logger.info("Reference pairing coverage: %s", ref_pairing_coverage)

    # ── Shard splitting ──
    uids = list(inf_data.keys())
    expected_population = len(uids)
    size = expected_population
    if size == 0:
        logger.error("inf_scp is empty")
        sys.exit(1)

    if not (1 <= args.job <= args.nsplits <= size):
        logger.error(
            f"Invalid shard config: job={args.job}, nsplits={args.nsplits}, "
            f"total_samples={size}. Must satisfy 1 <= job <= nsplits <= total_samples."
        )
        sys.exit(1)
    interval = size // args.nsplits
    start = (args.job - 1) * interval
    end = size if args.job == args.nsplits else start + interval
    uids = uids[start:end]
    suffix = "" if args.nsplits == args.job == 1 else f".{args.job}"
    logger.info(f"[Job {args.job}/{args.nsplits}] Evaluating {len(uids)}/{size} samples")

    has_ref = bool(ref_data)

    # ── Load WER metadata ──
    transcripts, lang_ids = {}, {}
    if args.wer:
        if not args.meta_tsv:
            logger.error("--wer requires --meta_tsv")
            sys.exit(1)
        with open(args.meta_tsv, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]
        if args.utt2lang:
            with open(args.utt2lang, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        lang_ids[parts[0]] = parts[1]

    # ── Load LID metadata (reuse utt2lang) ──
    lid_labels = {}
    if args.lid:
        if not args.utt2lang:
            logger.error("--lid requires --utt2lang (language labels)")
            sys.exit(1)
        with open(args.utt2lang, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    lid_labels[parts[0]] = parts[1]

    # ── Load models ──
    mos_models = {}
    if args.dnsmos:
        mos_models["DNSMOS"] = load_dnsmos()
    if args.utmos:
        mos_models["UTMOS"] = load_utmos(device=device)
    if args.nisqa:
        mos_models["NISQA"] = load_nisqa()
    if args.scoreq:
        # 固定使用 NR；ref 模式缺少 reference 时在加载处显式失败。
        mos_models["SCOREQ"] = _load_scoreq_for_eval(has_ref)
    wer_model, textcleaner = load_wer_model(device) if args.wer else (None, None)
    spk_model = load_speaker_model(device) if args.speaker_similarity else None
    emo_model = load_emotion_model(device) if args.emotion_similarity else None
    phoneme_models = load_phoneme_model(device) if args.phoneme_similarity else None
    speechbert_model = load_speechbert_model(device) if args.speechbert else None
    lid_model = load_lid_model(device) if args.lid else None

    # ── Build evaluation entries and audio-quality pairs ──
    # `pairs` covers every file (used for non-intrusive MOS, enh-only).
    # `ref_pairs` covers only files with a real per-file reference (used for
    # intrusive PESQ/STOI/ESTOI/SI-SDR — never run against a zero reference).
    pairs: list[tuple[np.ndarray, np.ndarray, int]] = []
    ref_pairs: list[tuple[np.ndarray, np.ndarray, int]] = []
    entries: list[tuple[str, np.ndarray, int, Optional[np.ndarray], Optional[int]]] = []
    # 与 ref_pairs 索引对齐的 uid 列表，用于逐样本 AQ 映射。
    ref_uids: list[str] = []
    for uid in tqdm(uids, desc="Loading audio"):
        try:
            inf_audio, inf_fs = load_audio(inf_data[uid])
            ref_audio: Optional[np.ndarray] = None
            ref_fs: Optional[int] = None
            if has_ref and uid in ref_data:
                ref_audio, ref_fs = load_audio(ref_data[uid])
                if inf_fs != ref_fs:
                    inf_audio = _resample_np(inf_audio, inf_fs, ref_fs)
                    inf_fs = ref_fs
                minlen = min(len(ref_audio), len(inf_audio))
                if minlen == 0:
                    continue
                pair = (ref_audio[:minlen], inf_audio[:minlen], inf_fs)
                pairs.append(pair)
                ref_pairs.append(pair)
                ref_uids.append(uid)
            else:
                # 跳过零长 enhanced，避免批评测裁剪后与逐样本结果错位。
                if len(inf_audio) == 0:
                    continue
                pairs.append((np.zeros_like(inf_audio), inf_audio, inf_fs))
            entries.append((uid, inf_audio, inf_fs, ref_audio, ref_fs))
        except Exception as e:
            logger.warning(f"[{uid}] Failed loading audio: {e}")

    # ── Audio quality (batch via core.metrics) ──
    aq_means: dict[str, float] = {}
    aq_counts: dict[str, int] = {}
    # 逐样本 AQ：uid -> {metric: value}，最终并入 per_utterance.json。
    aq_per_utt: dict[str, dict[str, float]] = {}
    # Map MOS sub-score keys to legacy names used by URGENT normalization
    # （聚合均值与逐样本记录共用同一映射）
    _mos_key_map = {
        "DNSMOS_OVRL": "DNSMOS",
        "NISQA_MOS": "NISQA",
        "UTMOS_MOS": "UTMOS",
        "SCOREQ_MOS": "SCOREQ",
    }
    if pairs:
        raw: dict[str, float] = {}
        # Intrusive metrics: only over pairs that have a real reference.
        if has_ref and ref_pairs:
            raw_intrusive, ps_intrusive = compute_metrics_batch(
                pairs=ref_pairs,
                metrics=["PESQ", "STOI", "ESTOI", "SI-SDR"],
                mos_models=None,
                device=device,
                parallel_intrusive=True,
                return_per_sample=True,
            )
            raw.update(raw_intrusive)
            # ref_pairs 与 ref_uids 索引对齐（加载循环同步 append）
            for si, uid in enumerate(ref_uids):
                if ps_intrusive[si]:
                    aq_per_utt.setdefault(uid, {}).update(ps_intrusive[si])
        # Non-intrusive MOS: over every file (enh-only, no reference needed).
        raw_mos, ps_mos = compute_metrics_batch(
            pairs=pairs,
            metrics=[],
            mos_models=mos_models,
            device=device,
            parallel_intrusive=True,
            return_per_sample=True,
        )
        raw.update(raw_mos)
        # pairs 与 entries 索引对齐（加载循环同步 append，零长已显式跳过）
        if len(ps_mos) == len(entries):
            for si, (uid, *_unused) in enumerate(entries):
                rec = {_mos_key_map.get(k, k): v for k, v in ps_mos[si].items()}
                if rec:
                    aq_per_utt.setdefault(uid, {}).update(rec)
        else:
            logger.warning(
                f"MOS per-sample count ({len(ps_mos)}) != entries ({len(entries)}); "
                "skip per-utterance AQ mapping"
            )
        for key, total in raw.items():
            if key.startswith("_count_"):
                continue
            count = raw.get(f"_count_{key}", 0.0)
            if count > 0:
                aq_means[key] = total / count
                aq_counts[key] = int(count)

        for old_key, new_key in _mos_key_map.items():
            if old_key in aq_means:
                aq_means[new_key] = aq_means.pop(old_key)
                if old_key in aq_counts:
                    aq_counts[new_key] = aq_counts.pop(old_key)

    # ── Output writers (downstream metrics only) ──
    downstream_metrics = []
    if args.wer:
        downstream_metrics.extend(["WER", "CER"])
    if args.speaker_similarity:
        downstream_metrics.append("SpeakerSimilarity")
    if args.emotion_similarity:
        downstream_metrics.append("EmotionSimilarity")
    if args.phoneme_similarity:
        downstream_metrics.append("PhonemeSimilarity")
    if args.speechbert:
        downstream_metrics.append("SpeechBERTScore")
    if args.lid:
        downstream_metrics.append("LAcc")

    import contextlib
    with contextlib.ExitStack() as stack:
        writers = {
            m: stack.enter_context(
                (outdir / f"{m}{suffix}.scp").open("w", encoding="utf-8")
            )
            for m in downstream_metrics
        }

        # ── Downstream evaluation ──
        all_results = defaultdict(list)
        wer_results, cer_results = [], []
        n_ok, n_fail = 0, 0
        # 逐样本记录（JAE-M06）：下游指标在循环中逐样本可得，累积后写 per_utterance.json
        per_utt_records: dict[str, dict] = {}

        for uid, inf_audio, inf_fs, ref_audio, ref_fs in tqdm(entries, desc="URGENT evaluation"):
            scores = {}
            metric_errors: list[str] = []

            def _try_metric(name, fn):
                """单个指标独立失败：记录后继续其余指标，避免整组结果被丢弃。"""
                try:
                    return fn()
                except Exception as exc:
                    metric_errors.append(f"{name}: {exc}")
                    logger.warning(f"[{uid}] {name} failed: {exc}")
                    return None

            # B1: WER/CER
            if wer_model is not None and uid in transcripts:
                ref_txt = transcripts[uid]
                lang = lang_ids.get(uid, "<eng>")
                if ref_txt != "<not-available>":
                    wer_result = _try_metric(
                        "WER/CER",
                        lambda: compute_wer(
                            wer_model, textcleaner, ref_txt, inf_audio, lang, inf_fs
                        ),
                    )
                    if wer_result is not None:
                        scores["WER"] = wer_result["WER"]
                        scores["CER"] = wer_result["CER"]
                        wer_results.append(wer_result["WER"])
                        cer_results.append(wer_result["CER"])

            # B2: Speaker Similarity
            if spk_model is not None and ref_audio is not None and ref_fs is not None:
                value = _try_metric(
                    "SpeakerSimilarity",
                    lambda: compute_speaker_similarity(spk_model, ref_audio, inf_audio, ref_fs),
                )
                if value is not None:
                    scores["SpeakerSimilarity"] = value

            # B3: Emotion Similarity
            if emo_model is not None and ref_audio is not None and ref_fs is not None:
                value = _try_metric(
                    "EmotionSimilarity",
                    lambda: compute_emotion_similarity(emo_model, ref_audio, inf_audio, ref_fs),
                )
                if value is not None:
                    scores["EmotionSimilarity"] = value

            # B4: Phoneme Similarity
            if phoneme_models is not None and ref_audio is not None and ref_fs is not None:
                value = _try_metric(
                    "PhonemeSimilarity",
                    lambda: compute_phoneme_similarity(phoneme_models, ref_audio, inf_audio, ref_fs),
                )
                if value is not None:
                    scores["PhonemeSimilarity"] = value

            # B5: SpeechBERTScore
            if speechbert_model is not None and ref_audio is not None and ref_fs is not None:
                value = _try_metric(
                    "SpeechBERTScore",
                    lambda: compute_speechbert_score(speechbert_model, ref_audio, inf_audio, ref_fs),
                )
                if value is not None:
                    scores["SpeechBERTScore"] = value

            # B6: LID
            if lid_model is not None and uid in lid_labels:
                value = _try_metric(
                    "LAcc", lambda: compute_lid(lid_model, inf_audio, inf_fs, lid_labels[uid])
                )
                if value is not None:
                    scores["LAcc"] = value

            # Write results（写盘失败单独记录，不影响已算出的指标）
            try:
                for m, v in scores.items():
                    if v is not None and m in writers:
                        if m in ("WER", "CER"):
                            writers[m].write(f"{uid} {json.dumps(v)}\n")
                        else:
                            writers[m].write(f"{uid} {v}\n")
                        if m not in ("WER", "CER"):
                            all_results[m].append(v)
                        per_utt_records.setdefault(uid, {})[m] = v
            except Exception as exc:
                metric_errors.append(f"write: {exc}")
                logger.warning(f"[{uid}] Failed writing results: {exc}")

            if metric_errors:
                n_fail += 1
            else:
                n_ok += 1

    # writers closed automatically by ExitStack when the `with` block exits

    # ── Summary ──
    if args.job == 1:
        print("\n" + "=" * 76)
        print("  URGENT2026 Evaluation Results")
        print("=" * 76)

        means = dict(aq_means)

        print(f"\n  {'-' * 30} A. Audio Quality {'-' * 30}")
        print(f"  {'Metric':22s} | {'Mean':>12s} | {'Count':>8s}")
        print(f"  {'-' * 60}")
        for m in ["PESQ", "STOI", "ESTOI", "SI-SDR", "DNSMOS", "NISQA", "UTMOS", "SCOREQ"]:
            if m in means:
                print(f"  {m:22s} | {means[m]:12.4f} | {aq_counts.get(m, len(pairs)):8d}")

        print(f"\n  {'-' * 30} B. Downstream Impact {'-' * 30}")
        print(f"  {'Metric':22s} | {'Value':>12s} | {'Count':>8s}")
        print(f"  {'-' * 60}")

        if wer_results:
            wer_score = aggregate_wer(wer_results)
            if wer_score is not None:
                means["WER"] = wer_score
                print(f"  {'WER':22s} | {wer_score:12.4f} | {len(wer_results):8d}")
        if cer_results:
            cer_score = aggregate_wer(cer_results)
            if cer_score is not None:
                means["CER"] = cer_score
                print(f"  {'CER':22s} | {cer_score:12.4f} | {len(cer_results):8d}")

        for m in ["SpeakerSimilarity", "EmotionSimilarity", "PhonemeSimilarity", "SpeechBERTScore", "LAcc"]:
            vals = all_results.get(m, [])
            if vals:
                mean = np.nanmean(vals)
                means[m] = mean
                print(f"  {m:22s} | {mean:12.4f} | {len(vals):8d}")

        score_coverage = build_expected_metric_coverage(
            expected_population,
            {metric: aq_counts.get(metric, 0) for metric in NORMALIZED_METRICS},
        )
        normalized_score = build_normalized_score(means, score_coverage)
        dimensions = normalized_score["dimensions"]
        denom_str = f"{dimensions['valid']}/{dimensions['required']}"
        print(f"\n  {'-' * 30} C. Normalized Score {'-' * 28}")
        for m, raw in normalized_score["raw_values"].items():
            print(f"  {m:22s} | {raw:8.4f} → {normalized_score['normalized_values'][m]:.4f}")
        print(f"  {'-' * 60}")
        if normalized_score["value"] is not None:
            print(
                f"  {'Normalized avg':18s} | {normalized_score['value']:12.4f}  "
                f"({normalized_score['status']}; dimensions={denom_str})"
            )
        else:
            print(
                f"  {'Normalized avg':18s} | {'N/A':>12s}  "
                f"({normalized_score['status']}; dimensions={denom_str})"
            )

        print("\n" + "=" * 76)
        print(f"  {n_ok} succeeded, {n_fail} failed")
        print("=" * 76)

        with open(outdir / "RESULTS.txt", "w", encoding="utf-8") as f:
            for m, v in means.items():
                f.write(f"{m}: {v:.4f}\n")
            # ``_count_`` 记录逐指标样本数，保持 ``key: value`` 行格式。
            for m, c in sorted(aq_counts.items()):
                f.write(f"_count_{m}: {c}\n")
            if normalized_score["value"] is not None:
                f.write(f"\nNormalized_Average: {normalized_score['value']:.4f}\n")
            else:
                f.write("\nNormalized_Average: N/A\n")
            f.write(f"Normalized_Average_Denominator: {denom_str}\n")
            f.write(f"Normalized_Average_Status: {normalized_score['status']}\n")
            for metric, item in normalized_score["sample_coverage"].items():
                f.write(
                    f"Normalized_Average_Coverage_{metric}: "
                    f"{item['computed']}/{item['eligible']} ({item['status']})\n"
                )
            if ref_pairing_coverage is not None:
                for key, value in ref_pairing_coverage.items():
                    f.write(f"_ref_pairing_{key}: {value}\n")

        normalized_score, score_path = write_normalized_score(outdir, normalized_score)
        logger.info(f"Results saved: {outdir / 'RESULTS.txt'}")
        logger.info(f"Normalized score saved: {score_path}")

        # 合并下游与音频质量的逐样本指标后写入 per_utterance.json。
        for uid, aq_rec in aq_per_utt.items():
            per_utt_records.setdefault(uid, {}).update(aq_rec)
        if per_utt_records:
            per_utt_path = outdir / "per_utterance.json"
            with open(per_utt_path, "w", encoding="utf-8") as f:
                json.dump({
                    "note": "下游与 audio-quality 指标均为逐样本值；"
                            "逐指标样本数见 RESULTS.txt 的 _count_ 行",
                    "audio_quality_counts": {m: aq_counts[m] for m in sorted(aq_counts)},
                    "records": per_utt_records,
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Per-utterance records saved: {per_utt_path}")

    logger.info(f"Detailed metric files: {outdir}")

    # 失败样本（含单指标失败）必须反映到退出码，避免"部分失败但 exit 0"。
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
