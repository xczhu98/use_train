"""Layer A speech preparation seam for dynamic datasets."""

from __future__ import annotations

import logging

import numpy as np

from core.data.augment import apply_speech_augmentations
from core.data.simulation.contracts import PreparedSpeech
from core.data.simulation.engine import get_highpass_taps
from core.data.simulation.urgent_impl import fast_filtfilt_fir
from core.data.simulation_views import simulation_view_for
from core.data.simulation_utils import (
    align_speech_with_silence_pad,
    fast_active_rms,
    rng_randint,
    rng_random,
    rng_uniform,
)

logger = logging.getLogger(__name__)

SPEECH_SPLICE_UNLIMITED_MAX_ATTEMPTS = 10_000
SPEECH_SPLICE_MAX_CONSECUTIVE_FAILURES = 64


def _layer_a_config(dataset):
    """返回 Layer A 的只读配置切片。

    正常 Dataset 初始化时已附着严格解析后的 simulation view；保留未初始化
    Dataset 实例的 ``sim_cfg`` 读取，以支持直接调用本模块的低层适配器。
    """
    if getattr(dataset, "_simulation_view", None) is None:
        return dataset.sim_cfg
    return simulation_view_for(
        dataset, task=getattr(dataset, "_simulation_task", None)
    ).shared.layer_a


class SpeechPreparation:
    """执行 Layer A clean speech preparation；Dataset 提供资源和运行上下文。"""

    def get_clean_speech(
        self,
        dataset,
        speech_uid: str,
        sample_rate: int,
        rng: np.random.Generator | None = None,
        *,
        single_utterance: bool = False,
        accounting: dict | None = None,
    ) -> np.ndarray:
        """加载 clean speech，并按配置执行 splice。

        ``accounting`` 是本次 Layer A 的显式写入目标；不再通过 Dataset
        临时属性传递 splice 状态，避免嵌套调用之间共享隐藏上下文。
        """
        cfg = _layer_a_config(dataset)
        path = dataset.speech_flt[speech_uid]
        clean = dataset._read_one_speech(path, sample_rate)
        if single_utterance or not cfg.speech_splice:
            return clean
        seconds_range = getattr(cfg, "speech_splice_seconds_range", None)
        if seconds_range and len(seconds_range) >= 2:
            splice_seconds = float(
                rng_uniform(rng, float(seconds_range[0]), float(seconds_range[1]))
            )
        else:
            splice_seconds = float(cfg.speech_splice_seconds)
        target_len = int(splice_seconds * sample_rate)
        if target_len <= 0:
            return clean
        if cfg.speech_splice_equal_volume:
            ref_rms = fast_active_rms(clean, sample_rate)
            if not np.isfinite(ref_rms) or ref_rms <= 0:
                ref_rms = 1e-8
        gap_range = getattr(cfg, "speech_splice_gap_range", [0.02, 0.2])
        gap_min_sec = float(gap_range[0]) if gap_range and len(gap_range) >= 1 else 0.02
        gap_max_sec = float(gap_range[1]) if gap_range and len(gap_range) >= 2 else 0.2
        log_gap_min = np.log(max(gap_min_sec, 1e-6))
        log_gap_max = np.log(max(gap_max_sec, gap_min_sec + 1e-6))
        max_repeats = int(getattr(cfg, "speech_splice_max_repeat_times", 0))
        candidate_uids = dataset._speech_splice_candidate_uids(speech_uid)
        candidate_pool = (
            dataset._all_speech_paths
            if candidate_uids is None
            else candidate_uids or (speech_uid,)
        )
        if accounting is not None:
            accounting["candidate_scope"] = (
                "manifest" if candidate_uids is None else "canonical_identity"
            )
        repeat_count = 0
        consecutive_failures = 0
        unlimited_max_attempts = int(SPEECH_SPLICE_UNLIMITED_MAX_ATTEMPTS)
        max_consecutive_failures = int(SPEECH_SPLICE_MAX_CONSECUTIVE_FAILURES)
        while len(clean) < target_len:
            exhausted = (
                (max_repeats > 0 and repeat_count >= max_repeats)
                or (
                    max_repeats <= 0
                    and repeat_count >= unlimited_max_attempts
                )
                or consecutive_failures >= max_consecutive_failures
            )
            if exhausted:
                pad_len = target_len - len(clean)
                clean = np.concatenate((clean, np.zeros(pad_len, dtype=np.float32)))
                if accounting is not None:
                    accounting["zero_pad_samples"] += int(pad_len)
                break
            gap_sec = np.exp(rng_uniform(rng, log_gap_min, log_gap_max))
            gap_len = int(gap_sec * sample_rate)
            gap = np.zeros(gap_len, dtype=np.float32)
            choice_index = rng_randint(rng, 0, len(candidate_pool) - 1)
            selected = candidate_pool[choice_index]
            if candidate_uids is None:
                extra_path = selected
                all_speech_uids = getattr(dataset, "_all_speech_uids", None)
                extra_uid = (
                    all_speech_uids[choice_index]
                    if all_speech_uids is not None
                    else selected
                )
            else:
                extra_uid = selected
                extra_path = dataset.speech_flt[extra_uid]
            repeat_count += 1
            if accounting is not None:
                accounting["attempt_count"] += 1
                accounting["attempted_extra_uids"].append(extra_uid)
            try:
                extra = dataset._read_one_speech(extra_path, sample_rate)
                if len(extra) == 0:
                    raise ValueError("zero-length splice segment")
            except Exception as exc:
                logger.warning(
                    "Splice extra load failed (path=%s), skipping segment: %s",
                    extra_path,
                    exc,
                )
                consecutive_failures += 1
                if accounting is not None:
                    accounting["failure_count"] += 1
                continue
            consecutive_failures = 0
            if accounting is not None:
                accounting["extra_uids"].append(extra_uid)
            if cfg.speech_splice_equal_volume:
                extra_rms = fast_active_rms(extra, sample_rate)
                lo, hi = cfg.speech_splice_equal_volume_range
                db_offset = rng_uniform(rng, lo, hi)
                target_rms = ref_rms * 10.0 ** (db_offset / 20.0)
                extra = extra * (target_rms / (extra_rms + 1e-8))
            clean = np.concatenate((clean, gap, extra))
        if len(clean) > target_len:
            start = (
                rng_randint(rng, 0, len(clean) - target_len)
                if cfg.speech_random_start
                else 0
            )
            clean = clean[start : start + target_len]
        return clean

    def prepare_result(
        self,
        dataset,
        speech_uid: str,
        sample_rate: int,
        max_duration_samples: int,
        rng: np.random.Generator,
        *,
        source_trace=None,
        record_layer_a_metadata: bool = True,
    ):
        """完成 Layer A，并返回结构化 ``PreparedSpeech`` 结果。"""
        cfg = _layer_a_config(dataset)
        fixed_canvas_enabled = max_duration_samples > 0
        silence_pad_preselected = None
        if fixed_canvas_enabled:
            silence_pad_prob = float(getattr(cfg, "silence_pad_prob", 0.0))
            silence_pad_preselected = bool(
                silence_pad_prob > 0.0 and rng_random(rng) < silence_pad_prob
            )
        splice_accounting = {
            "primary_uid": speech_uid,
            "candidate_scope": "manifest",
            "quota_scope": "primary_only",
            "attempted_extra_uids": [],
            "extra_uids": [],
            "attempt_count": 0,
            "failure_count": 0,
            "zero_pad_samples": 0,
            "single_utterance": bool(silence_pad_preselected),
        }
        speech_1d = self.get_clean_speech(
            dataset,
            speech_uid,
            sample_rate,
            rng=rng,
            single_utterance=bool(silence_pad_preselected),
            accounting=splice_accounting,
        )
        speech_1d = apply_speech_augmentations(
            speech_1d,
            sample_rate,
            cfg,
            skip_extra_space=cfg.speech_splice,
            rng=rng,
        )
        len_before_align = len(speech_1d)
        speech_1d, silence_pad_applied = align_speech_with_silence_pad(
            speech_1d,
            sample_rate,
            max_duration_samples,
            silence_pad_prob=float(getattr(cfg, "silence_pad_prob", 0.0)),
            silence_pad_total_range=getattr(cfg, "silence_pad_total_range", [1.0, 4.0]),
            silence_pad_position=getattr(cfg, "silence_pad_position", "random"),
            silence_pad_mode=getattr(cfg, "silence_pad_mode", "uniform"),
            silence_pad_short_range=getattr(cfg, "silence_pad_short_range", None),
            silence_pad_long_range=getattr(cfg, "silence_pad_long_range", None),
            silence_pad_short_ratio=float(getattr(cfg, "silence_pad_short_ratio", 0.5)),
            silence_pad_beta_a=float(getattr(cfg, "silence_pad_beta_a", 0.4)),
            silence_pad_beta_b=float(getattr(cfg, "silence_pad_beta_b", 0.4)),
            repeat_to_max_duration=dataset.repeat_to_max_duration,
            insert_silence_gap_between_tiles=dataset.repeat_tile_silence_gap,
            gap_min_sec=dataset.repeat_tile_silence_gap_range[0],
            gap_max_sec=dataset.repeat_tile_silence_gap_range[1],
            rng=rng,
            fixed_silence_canvas=fixed_canvas_enabled,
            return_silence_pad_applied=True,
            preselected_silence_pad=silence_pad_preselected,
        )
        fixed_silence_canvas_active = bool(fixed_canvas_enabled and silence_pad_applied)
        len_after_align = len(speech_1d)
        repeat_export = dataset._repeat_export_info(
            sample_rate, len_before_align, len_after_align, max_duration_samples
        )
        if fixed_silence_canvas_active:
            repeat_export["align_mode"] = "silence_fixed_canvas"
        speech_splice_export = dataset._speech_splice_export_info(
            cfg, sample_rate, len(speech_1d)
        )
        speech_splice_export.update(splice_accounting)
        if source_trace is not None:
            source_trace.replace_metadata("speech_splice_export", speech_splice_export)
            source_trace.replace_metadata("repeat_export", repeat_export)
            if record_layer_a_metadata:
                source_trace.replace_metadata(
                    "layer_a",
                    {
                        "speech_uid": speech_uid,
                        "sample_rate": sample_rate,
                        "samples_before_align": len_before_align,
                        "samples_after_align": len_after_align,
                        "fixed_silence_canvas_active": fixed_silence_canvas_active,
                    },
                )
        if source_trace is None and dataset.export_simulation_stems:
            dataset._last_simulation_stems = {
                "speech_splice_export": speech_splice_export,
                "repeat_export": repeat_export,
            }
        elif source_trace is None:
            dataset._last_simulation_stems = None
        if getattr(cfg, "use_high_pass", True):
            if dataset._hp_taps is None:
                dataset._hp_taps = {}
            if sample_rate not in dataset._hp_taps:
                dataset._hp_taps[sample_rate] = get_highpass_taps(sample_rate)
            taps = dataset._hp_taps[sample_rate]
            if len(speech_1d) > len(taps) and np.all(np.isfinite(taps)):
                speech_1d = fast_filtfilt_fir(taps, speech_1d).astype(np.float32)
        actual_len = len(speech_1d)
        if actual_len == 0:
            actual_len = 1
            speech_1d = np.zeros(1, dtype=np.float32)
        result = PreparedSpeech(
            speech_uid=speech_uid,
            sample_rate=int(sample_rate),
            speech_1d=speech_1d,
            speech_2d=speech_1d.reshape(1, -1),
            actual_len=actual_len,
            fixed_silence_canvas_active=fixed_silence_canvas_active,
            speech_splice_export=speech_splice_export,
            repeat_export=repeat_export,
        )
        dataset._fixed_silence_canvas_active = result.fixed_silence_canvas_active
        return result


__all__ = [
    "SPEECH_SPLICE_MAX_CONSECUTIVE_FAILURES",
    "SPEECH_SPLICE_UNLIMITED_MAX_ATTEMPTS",
    "SpeechPreparation",
]
