"""eval_pipeline_tse 纯函数单测（无需模型/音频重资源）。

覆盖：``_edit_error_rate`` / ``_normalize_transcript`` / ``_ter_error_rate`` /
``_resolve_audio_path`` / ``_make_uid`` / ``_timing_prf`` /
``_escape_csv_formula`` / ``_meta_matches``，以及 timing 官方口径的
区间/帧级指标/GT label 纯函数与 wespeaker 语种选择逻辑。

涉及可选依赖的用例（whisper-normalizer / opencc / meeteval /
wespeakerruntime）做 importorskip/文件存在守卫。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pytest

from evaluate.eval_pipeline_tse import (
    _apply_collar_to_segments,
    _clip_intervals,
    _edit_error_rate,
    _escape_csv_formula,
    _finite_nonnegative,
    _finite_positive,
    _make_uid,
    _merge_intervals,
    _meta_matches,
    _normalize_transcript,
    _parse_dataset_lang_overrides,
    _parse_mixture_utterance,
    _peak_normalize_to_int16,
    _resolve_audio_path,
    _segments_to_frame_mask,
    _ter_error_rate,
    _timing_frame_metrics,
    _timing_micro_prf,
    _timing_prf,
    _TimingGTLabels,
    _wespeaker_lang_for_row,
)


class TestEditErrorRate:
    def test_identical_is_zero(self):
        assert _edit_error_rate(["a", "b"], ["a", "b"]) == 0.0

    def test_both_empty_is_zero(self):
        assert _edit_error_rate([], []) == 0.0

    def test_empty_ref_nonempty_hyp_is_one(self):
        assert _edit_error_rate([], ["a"]) == 1.0

    def test_substitution(self):
        assert _edit_error_rate(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(1 / 3)

    def test_insertion_and_deletion(self):
        # ref "a b c" → hyp "a c"：删 1 字，CER=1/3
        assert _edit_error_rate(["a", "b", "c"], ["a", "c"]) == pytest.approx(1 / 3)
        # hyp 多插 1 字
        assert _edit_error_rate(["a", "c"], ["a", "b", "c"]) == pytest.approx(1 / 2)

    def test_empty_hyp(self):
        assert _edit_error_rate(["a", "b"], []) == 1.0


class TestNormalizeTranscript:
    """对齐官方 utils/asr_metrics.py 的 normalizer_for_transcript。"""

    @pytest.fixture(autouse=True)
    def _require_deps(self):
        pytest.importorskip("whisper_normalizer")
        pytest.importorskip("opencc")

    def test_en_predicted_whisper_normalized(self):
        # hyp 走 whisper EnglishTextNormalizer：小写、去标点、展开缩写
        assert _normalize_transcript("Hello, World!", "Predicted", "en") == "hello world"
        assert _normalize_transcript("i've got it", "Predicted", "en") == "i have got it"

    def test_en_ground_truth_not_whisper_normalized(self):
        # 官方口径：en GT 不做 whisper 归一化（保留大小写与逗号），仅去 "." 与 "…"
        assert _normalize_transcript("Hello, World!", "Ground Truth", "en") == "Hello, World!"
        assert _normalize_transcript("a… b. c", "Ground Truth", "en") == "a b c"

    def test_zh_char_spaced(self):
        # zh 统一 t2s 后逐字空格分隔（字级 token）
        assert _normalize_transcript("你好世界", "Predicted", "zh") == "你 好 世 界"

    def test_zh_traditional_to_simplified(self):
        assert _normalize_transcript("天氣不錯", "Ground Truth", "zh") == "天 气 不 错"

    def test_zh_ground_truth_keeps_punctuation_chars(self):
        # 官方口径：zh GT 不去标点（标点成为独立 token），与 hyp 分支不同
        assert _normalize_transcript("你好，世界", "Ground Truth", "zh") == "你 好 ， 世 界"

    def test_invalid_option_or_language(self):
        with pytest.raises(ValueError, match="Invalid option"):
            _normalize_transcript("a", "bad", "fr")
        with pytest.raises(ValueError, match="Invalid language"):
            _normalize_transcript("a", "Predicted", "fr")


class TestTerErrorRate:
    def test_identical_is_zero(self):
        assert _ter_error_rate("hello world", "hello world") == 0.0

    def test_deletion(self):
        assert _ter_error_rate("a b c", "a c") == pytest.approx(1 / 3)

    def test_empty_ref_is_nan(self):
        # 官方 meeteval 对空 GT 返回 None → 记 NaN（汇总剔除）
        assert math.isnan(_ter_error_rate("", "a b"))

    def test_matches_meeteval_when_available(self):
        meeteval = pytest.importorskip("meeteval")
        from meeteval.wer.wer.siso import siso_word_error_rate

        for gt, pred in [("a b c", "a x c"), ("你 好 世 界", "你 好 界"),
                         ("hello", "hello there")]:
            assert _ter_error_rate(gt, pred) == pytest.approx(
                siso_word_error_rate(gt, pred).error_rate)


@pytest.mark.unit
class TestResolveAudioPath:
    def test_absolute_existing(self, tmp_path: Path):
        f = tmp_path / "a.wav"
        f.write_bytes(b"x")
        assert _resolve_audio_path(str(f), None, {}) == f

    def test_audio_root_join(self, tmp_path: Path):
        f = tmp_path / "sub.wav"
        f.write_bytes(b"x")
        assert _resolve_audio_path("sub.wav", str(tmp_path), {}) == f

    def test_audio_root_suffix_completion(self, tmp_path: Path):
        f = tmp_path / "utt1.wav"
        f.write_bytes(b"x")
        assert _resolve_audio_path("utt1", str(tmp_path), {}) == f

    @pytest.mark.parametrize("subdir", ["mixtures", "enrolment_speakers"])
    def test_realt_official_audio_subdirectories(self, tmp_path: Path, subdir: str):
        audio_dir = tmp_path / subdir
        audio_dir.mkdir()
        f = audio_dir / "utt1.wav"
        f.write_bytes(b"x")
        assert _resolve_audio_path("utt1", str(tmp_path), {}) == f

    def test_mapping_priority(self, tmp_path: Path):
        f = tmp_path / "mapped.wav"
        f.write_bytes(b"x")
        assert _resolve_audio_path("uttX", str(tmp_path), {"uttX": str(f)}) == f

    def test_mapping_relative_path_is_resolved_under_audio_root(self, tmp_path: Path):
        f = tmp_path / "mapped.wav"
        f.write_bytes(b"x")
        assert _resolve_audio_path("uttX", str(tmp_path), {"uttX": "mapped.wav"}) == f

    def test_mapping_rejects_directory(self, tmp_path: Path):
        directory = tmp_path / "audio-dir"
        directory.mkdir()
        with pytest.raises(ValueError, match="普通文件"):
            _resolve_audio_path("uttX", str(tmp_path), {"uttX": str(directory)})

    def test_mapping_rejects_symlink(self, tmp_path: Path):
        target = tmp_path / "target.wav"
        target.write_bytes(b"x")
        linked = tmp_path / "linked.wav"
        linked.symlink_to(target)
        with pytest.raises(ValueError, match="符号链接"):
            _resolve_audio_path("uttX", str(tmp_path), {"uttX": str(linked)})

    def test_mapping_rejects_path_outside_audio_root(self, tmp_path: Path):
        outside = tmp_path / "outside.wav"
        outside.write_bytes(b"x")
        root = tmp_path / "root"
        root.mkdir()
        with pytest.raises(ValueError, match="超出 audio_root"):
            _resolve_audio_path("uttX", str(root), {"uttX": str(outside)})

    def test_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            _resolve_audio_path("nope", str(tmp_path), {})


def test_mapping_rejects_duplicate_utterance(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "utterance,path\nutt,a.wav\nutt,b.wav\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate utterance"):
        __import__("evaluate.eval_pipeline_tse", fromlist=["_load_mapping"])._load_mapping(
            str(mapping)
        )


class TestTimingParameterBounds:
    @pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "bad"])
    def test_frame_shift_must_be_finite_and_positive(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            _finite_positive(value)

    @pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf", "bad"])
    def test_nonnegative_timing_values_reject_invalid_inputs(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            _finite_nonnegative(value)

    def test_zero_is_allowed_for_nonnegative_timing_values(self):
        assert _finite_nonnegative("0") == 0.0


class TestMakeUid:
    def test_format_is_sha256_prefix_32_hex(self):
        row = {"dataset": "ds", "mixture_utterance": "m1",
               "enrolment_speakers_utterance": "e1"}
        uid = _make_uid(row)
        assert uid == "b14766a7171b726e8257f2076ae0c7f1"
        assert len(uid) == 32
        assert int(uid, 16) >= 0

    def test_dataset_name_is_bound_into_digest(self):
        base = {"mixture_utterance": "m1", "enrolment_speakers_utterance": "e1"}
        uid_a = _make_uid({**base, "dataset": "dev"})
        uid_b = _make_uid({**base, "dataset": "test"})
        assert uid_a != uid_b

    def test_mixture_enroll_disambiguation(self):
        # 不同 (mixture, enroll) 对必须产生不同 uid
        uid_a = _make_uid({"dataset": "ds", "mixture_utterance": "m1",
                           "enrolment_speakers_utterance": "e1"})
        uid_b = _make_uid({"dataset": "ds", "mixture_utterance": "m1",
                           "enrolment_speakers_utterance": "e2"})
        assert uid_a != uid_b

    def test_deterministic(self):
        row = {"dataset": "ds", "mixture_utterance": "m1",
               "enrolment_speakers_utterance": "e1"}
        assert _make_uid(row) == _make_uid(dict(row))


class TestTimingPrf:
    """mixture-VAD 近似口径（无 GT label 时的 fallback）。"""

    def test_perfect_match(self):
        ref = np.array([True, True, False])
        out = _timing_prf(ref, ref.copy())
        assert out["timing_precision"] == 1.0
        assert out["timing_recall"] == 1.0
        assert out["timing_f1"] == 1.0

    def test_no_positive_hyp_precision_nan(self):
        # tp+fp==0（hyp 全静音）：precision 无定义 → NaN（汇总时剔除）
        ref = np.array([True, True, False])
        hyp = np.array([False, False, False])
        out = _timing_prf(ref, hyp)
        assert math.isnan(out["timing_precision"])
        assert math.isnan(out["timing_f1"])
        assert out["timing_recall"] == 0.0

    def test_length_mismatch_truncated(self):
        ref = np.array([True, True, True, True])
        hyp = np.array([True, False])
        out = _timing_prf(ref, hyp)
        # 对齐到前 2 帧：tp=1, fp=0, fn=1
        assert out["timing_precision"] == 1.0
        assert out["timing_recall"] == 0.5

    def test_empty_masks(self):
        out = _timing_prf(np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))
        assert math.isnan(out["timing_precision"])


class TestTimingOfficialHelpers:
    """timing 官方口径纯函数（对齐 tse_timing_evaluation.py）。"""

    def test_parse_mixture_utterance(self):
        assert _parse_mixture_utterance("S1_mixture_10.5_25.0") == ("S1", 10.5, 25.0)

    def test_parse_mixture_utterance_invalid(self):
        with pytest.raises(ValueError):
            _parse_mixture_utterance("no_timing_info")

    def test_merge_intervals(self):
        assert _merge_intervals([(3.0, 5.0), (1.0, 2.0), (4.5, 6.0)]) == [
            (1.0, 2.0), (3.0, 6.0)]

    def test_clip_intervals(self):
        assert _clip_intervals([(-1.0, 2.0), (3.0, 9.0)], 0.0, 5.0) == [
            (0.0, 2.0), (3.0, 5.0)]

    def test_apply_collar(self):
        # 双侧扩 0.05 后裁剪到 [0, duration]
        assert _apply_collar_to_segments([(1.0, 2.0)], 0.05, 3.0) == [(0.95, 2.05)]
        assert _apply_collar_to_segments([(0.01, 2.99)], 0.05, 3.0) == [(0.0, 3.0)]

    def test_segments_to_frame_mask_floor_ceil(self):
        # 起点 floor、终点 ceil（官方 segments_to_mask）
        mask = _segments_to_frame_mask([(0.015, 0.035)], 0.1, 0.01)
        assert mask.tolist() == [False, True, True, True] + [False] * 6

    def test_frame_metrics_perfect(self):
        m = _timing_frame_metrics([(1.0, 2.0)], [(1.0, 2.0)], 4.0, 0.01)
        assert m["timing_precision"] == 1.0
        assert m["timing_recall"] == 1.0
        assert m["timing_f1"] == 1.0
        assert m["timing_tp_dur"] == pytest.approx(1.0)

    def test_frame_metrics_empty_pred_zero_precision(self):
        # 官方 safe_divide：分母为 0 时 P/R/F1 记 0.0（区别于近似口径的 NaN）
        m = _timing_frame_metrics([], [(1.0, 2.0)], 4.0, 0.01)
        assert m["timing_precision"] == 0.0
        assert m["timing_recall"] == 0.0
        assert m["timing_f1"] == 0.0
        assert m["timing_fn_dur"] == pytest.approx(1.0)

    def test_micro_prf_aggregates_durations_first(self):
        # micro：先累加 TP/FP/FN 时长再算比值（不是逐行 F1 求均值）
        rows = [
            {"timing_tp_dur": 1.0, "timing_fp_dur": 1.0, "timing_fn_dur": 0.0},
            {"timing_tp_dur": 0.0, "timing_fp_dur": 0.0, "timing_fn_dur": 1.0},
        ]
        micro = _timing_micro_prf(rows)
        assert micro["timing_precision"] == pytest.approx(0.5)
        assert micro["timing_recall"] == pytest.approx(0.5)
        assert micro["timing_f1"] == pytest.approx(0.5)
        assert micro["timing_tp_dur_total"] == pytest.approx(1.0)

    def test_micro_prf_none_without_raw_keys(self):
        assert _timing_micro_prf([{"timing_precision": 0.5}]) is None


class TestTimingGTLabels:
    """_TimingGTLabels：官方 phase-1 GT label 构造（overlap json → 相对段 → collar）。"""

    @pytest.fixture()
    def gt(self, tmp_path: Path):
        overlap = {
            "sessionA": [
                {"overlap_info": {"overlap_start_time": 10.0, "overlap_end_time": 20.0},
                 "overlap_segments": [
                     {"speaker": "spk1", "start_time": 11.0, "end_time": 13.0},
                     {"speaker": "spk2", "start_time": 12.0, "end_time": 15.0},
                     {"speaker": "spk1", "start_time": 16.0, "end_time": 19.5},
                 ]},
            ]
        }
        (tmp_path / "ds1").mkdir()
        (tmp_path / "ds1" / "overlap_records.json").write_text(
            json.dumps(overlap), encoding="utf-8")
        return _TimingGTLabels(str(tmp_path), collar=0.05, match_tolerance=0.02)

    def test_label_segments_relative_with_collar(self, gt):
        out = gt.label_for_row({
            "dataset": "ds1", "speaker": "spk1",
            "mixture_utterance": "sessionA_mixture_10.0_20.0"})
        assert out is not None
        mix_duration, segs = out
        assert mix_duration == pytest.approx(10.0)
        # 相对段 (1,3)/(6,9.5) 加 collar 0.05 → (0.95,3.05)/(5.95,9.55)
        assert segs == [(0.95, 3.05), (5.95, 9.55)]

    def test_missing_speaker_returns_none(self, gt):
        assert gt.label_for_row({
            "dataset": "ds1", "speaker": None,
            "mixture_utterance": "sessionA_mixture_10.0_20.0"}) is None

    def test_bad_utterance_format_returns_none(self, gt):
        assert gt.label_for_row({
            "dataset": "ds1", "speaker": "spk1",
            "mixture_utterance": "bad_format"}) is None

    def test_no_matched_record_returns_none(self, gt):
        assert gt.label_for_row({
            "dataset": "ds1", "speaker": "spk1",
            "mixture_utterance": "sessionA_mixture_0.0_5.0"}) is None

    def test_missing_dataset_json_returns_none(self, gt):
        assert gt.label_for_row({
            "dataset": "nope", "speaker": "spk1",
            "mixture_utterance": "sessionA_mixture_10.0_20.0"}) is None


class TestPeakNormalizeToInt16:
    def test_peak_scaled(self):
        wav = np.array([0.0, 0.5, -0.25], dtype=np.float32)
        out = _peak_normalize_to_int16(wav, peak=0.9)
        assert out.dtype == np.int16
        # 峰值 0.5 → 0.9：0.9*32767 ≈ 29490
        assert out[1] == np.int16(np.clip(0.9 * 32767.0, -32768, 32767))

    def test_silent_returns_zeros(self):
        out = _peak_normalize_to_int16(np.zeros(100, dtype=np.float32))
        assert out.dtype == np.int16
        assert not out.any()

    def test_stereo_mixed_down(self):
        wav = np.stack([np.full(10, 0.5, dtype=np.float32),
                        np.full(10, 0.5, dtype=np.float32)], axis=1)
        out = _peak_normalize_to_int16(wav)
        assert out.shape == (10,)


class TestWeSpeakerLangSelection:
    def test_parse_overrides(self):
        ov = _parse_dataset_lang_overrides("AISHELL-4:chs,AliMeeting:chs")
        assert ov == {"AISHELL-4": "chs", "AliMeeting": "chs"}

    def test_parse_overrides_ignores_invalid(self):
        assert _parse_dataset_lang_overrides("") == {}
        assert _parse_dataset_lang_overrides("A:fr,B:chs,bad") == {"B": "chs"}

    def test_language_column_priority(self):
        ov = {"AISHELL-4": "chs"}
        assert _wespeaker_lang_for_row(
            {"language": "zh", "dataset": "AMI"}, ov, "en") == "chs"
        assert _wespeaker_lang_for_row(
            {"language": "en", "dataset": "AISHELL-4"}, ov, "en") == "en"

    def test_dataset_override_then_default(self):
        ov = {"AISHELL-4": "chs"}
        assert _wespeaker_lang_for_row(
            {"language": None, "dataset": "AISHELL-4"}, ov, "en") == "chs"
        assert _wespeaker_lang_for_row(
            {"language": None, "dataset": "AMI"}, ov, "en") == "en"


class TestEscapeCsvFormula:
    @pytest.mark.parametrize("v", ["=1+1", "+cmd", "-2+3", "@SUM(A1)"])
    def test_dangerous_prefix_quoted(self, v):
        assert _escape_csv_formula(v) == "'" + v

    @pytest.mark.parametrize("v", ["hello", "", "a=b", 1.5, None])
    def test_safe_values_untouched(self, v):
        assert _escape_csv_formula(v) == v


class TestMetaMatches:
    def test_v6_fingerprint_match(self):
        m = {
            "schema": "tse-inference-artifact-v6",
            "fingerprint": "a" * 64,
            "artifacts": {},
        }
        assert _meta_matches(dict(m), m)

    def test_v5_fingerprint_is_rejected(self):
        m = {
            "schema": "tse-inference-artifact-v5",
            "fingerprint": "a" * 64,
            "artifacts": {},
        }
        assert not _meta_matches(dict(m), m)

    def test_v4_fingerprint_is_rejected(self):
        m = {
            "schema": "tse-inference-artifact-v4",
            "fingerprint": "a" * 64,
            "artifacts": {},
        }
        assert not _meta_matches(dict(m), m)

    def test_v3_fingerprint_is_rejected(self):
        m = {
            "schema": "tse-inference-artifact-v3",
            "fingerprint": "a" * 64,
            "artifacts": {},
        }
        assert not _meta_matches(dict(m), m)

    def test_v2_fingerprint_is_rejected(self):
        m = {
            "schema": "tse-inference-artifact-v2",
            "fingerprint": "a" * 64,
            "artifacts": {},
        }
        assert not _meta_matches(dict(m), m)

    def test_fingerprint_mismatch(self):
        assert not _meta_matches(
            {"schema": "tse-inference-artifact-v6", "fingerprint": "a" * 64},
            {"schema": "tse-inference-artifact-v6", "fingerprint": "b" * 64},
        )

    def test_missing_sidecar_no_match(self):
        expected = {
            "schema": "tse-inference-artifact-v6",
            "fingerprint": "a" * 64,
        }
        assert not _meta_matches(None, expected)

    def test_legacy_path_mtime_sidecar_is_not_verifiable(self):
        legacy = {"model": "a.tar", "model_mtime": 1.0}
        assert not _meta_matches(legacy, legacy)


pytestmark = [pytest.mark.unit, pytest.mark.contract]
