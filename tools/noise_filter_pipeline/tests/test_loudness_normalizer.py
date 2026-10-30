"""
Tests for processors.loudness_normalizer (D2 checkpoint / resume).
"""
import json
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from noise_filter_pipeline.core.manifest_io import AudioRecord, read_manifest, write_manifest
from noise_filter_pipeline.processors.loudness_normalizer import LoudnessNormalizerProcessor

pytestmark = pytest.mark.pipeline


def _make_wav(path: str, sr: int = 16000, duration: float = 3.0, seed: int = 0) -> str:
    rng = np.random.RandomState(seed)
    audio = rng.randn(int(sr * duration)) * 0.1
    sf.write(path, audio, sr)
    return path


def _make_record(record_id: str, audio_path: str) -> AudioRecord:
    return AudioRecord(
        record_id=record_id,
        dataset_id="ds",
        source_id=record_id,
        audio={"path": audio_path, "sr": 16000, "duration": 3.0},
        labels={"audioset": "noise"},
    )


def test_d2_checkpoint_resume_skips_done_and_reprocesses_missing():
    """D2 断点续跑：已处理且产物可读的记录跳过恢复；产物缺失的记录回退重处理。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src1 = _make_wav(os.path.join(tmpdir, "s1.wav"), seed=1)
        src2 = _make_wav(os.path.join(tmpdir, "s2.wav"), seed=2)

        input_manifest = os.path.join(tmpdir, "input.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        cfg = {
            "output_layout": "flat",
            "output_format": "wav",
            "num_workers": 1,
            "min_duration": 0.5,
            "max_duration": 60.0,
        }

        # 第一次：处理 2 条
        write_manifest(iter([_make_record("ds__s1", src1), _make_record("ds__s2", src2)]), input_manifest)
        out_manifest = LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)

        recs = list(read_manifest(out_manifest))
        assert len(recs) == 2
        assert all(r.is_stage_passed("D2") for r in recs)
        out1, out2 = recs[0].audio["path"], recs[1].audio["path"]
        assert os.path.isfile(out1) and os.path.isfile(out2)
        ckpt_path = os.path.join(out_dir, "checkpoint.json")
        assert os.path.isfile(ckpt_path)

        # 模拟中断重跑（输入 manifest 不变）：s1 的产物被清理（崩溃残留）
        os.remove(out1)
        mtime2 = os.path.getmtime(out2)

        # 第二次：s1 产物缺失 -> 回退重处理；s2 恢复跳过（输入未变、产物未动）
        LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)

        recs2 = {r.record_id: r for r in read_manifest(out_manifest)}
        assert set(recs2) == {"ds__s1", "ds__s2"}
        assert all(r.is_stage_passed("D2") for r in recs2.values())
        # s1 重新生成产物；s2 产物未被重写（mtime 不变）
        assert os.path.isfile(recs2["ds__s1"].audio["path"])
        assert os.path.isfile(recs2["ds__s2"].audio["path"])
        assert os.path.getmtime(recs2["ds__s2"].audio["path"]) == mtime2
        # checkpoint 累计 2 条
        with open(ckpt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert set(data["done"]) == {"ds__s1", "ds__s2"}


def test_d2_rejected_record_does_not_reuse_stale_input_fingerprint():
    """D2 rejected 缓存仅能在输入指纹未变时恢复，输入替换后必须重算。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "source.wav")
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        cfg = {"output_layout": "flat", "output_format": "wav", "num_workers": 1,
               "min_duration": 0.5, "max_duration": 60.0}

        missing_record = _make_record("ds__source", src)
        write_manifest(iter([missing_record]), input_manifest)
        out_manifest = LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        assert list(read_manifest(out_manifest))[0].rejected_reason == "d2_missing_audio"

        _make_wav(src, seed=11)
        LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        rec = list(read_manifest(out_manifest))[0]
        assert rec.is_stage_passed("D2")
        assert os.path.isfile(rec.audio["path"])


@pytest.mark.parametrize("output_layout", ("flat", "dataset_label"))
def test_d2_symlink_input_fingerprint_is_not_checkpoint_restored(output_layout):
    """symlink 输入无 regular-file 指纹时必须拒绝且不得恢复旧 checkpoint。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = _make_wav(os.path.join(tmpdir, "target.wav"), seed=12)
        alias = os.path.join(tmpdir, "alias.wav")
        try:
            os.symlink(target, alias)
        except OSError as exc:
            pytest.skip(f"symlink unsupported: {exc}")

        rec = _make_record("ds__symlink", alias)
        if output_layout == "dataset_label":
            rec.stages["C3"] = {"status": "passed", "balance_category": "Vehicle"}
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        cfg = {
            "output_layout": output_layout,
            "output_format": "wav",
            "num_workers": 1,
            "min_duration": 0.5,
            "max_duration": 60.0,
        }
        write_manifest(iter([rec]), input_manifest)
        out_manifest = LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        first = list(read_manifest(out_manifest))[0]
        assert first.rejected_reason == "d2_input_fingerprint_unavailable"
        assert first.stages["D2"]["input_fingerprint"] is None

        calls = []
        proc = LoudnessNormalizerProcessor(cfg)
        original_process_one = proc._process_one

        def _record_process_one(record):
            calls.append(record.record_id)
            return original_process_one(record)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(proc, "_process_one", _record_process_one)
            proc.process(input_manifest, out_dir)
        assert calls == ["ds__symlink"]


def test_d2_rejects_duplicate_record_id_before_processing():
    """主循环必须拒绝重复 record_id，避免 checkpoint 与 ordinal 歧义。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sources = [
            _make_wav(os.path.join(tmpdir, name), seed=13 + index)
            for index, name in enumerate(("a.wav", "b.wav"))
        ]
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        write_manifest(
            iter([_make_record("duplicate", source) for source in sources]),
            input_manifest,
        )

        with pytest.raises(ValueError, match="duplicate record_id"):
            LoudnessNormalizerProcessor({
                "output_layout": "flat",
                "output_format": "wav",
                "num_workers": 1,
                "min_duration": 0.5,
                "max_duration": 60.0,
            }).process(input_manifest, os.path.join(tmpdir, "out"))


def test_d2_passed_record_rechecks_input_fingerprint_on_resume():
    """D2 passed 记录恢复时也必须比对输入指纹。

    回归测试：此前 passed 恢复只查输出可读。上游 stage 重算后 D1 输出
    路径不变（确定性 hash）但内容变化，D2 会静默恢复旧归一化结果。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_wav(os.path.join(tmpdir, "s.wav"), seed=1)
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        cfg = {"output_layout": "flat", "output_format": "wav", "num_workers": 1,
               "min_duration": 0.5, "max_duration": 60.0}

        write_manifest(iter([_make_record("ds__s", src)]), input_manifest)
        out_manifest = LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        rec = list(read_manifest(out_manifest))[0]
        assert rec.is_stage_passed("D2")
        first_out = rec.audio["path"]
        first_mtime = os.path.getmtime(first_out)

        # 输入文件内容被替换（模拟上游重算产物）：重跑必须重处理
        _make_wav(src, seed=99)
        LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        rec2 = list(read_manifest(out_manifest))[0]
        assert rec2.is_stage_passed("D2")
        # 产物被重写（新内容），而非静默复用旧产物
        assert os.path.getmtime(rec2.audio["path"]) != first_mtime


def test_d2_config_change_rejects_old_checkpoint():
    """换配置后 D2 的旧 checkpoint 被拒绝，全部记录重新处理。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_wav(os.path.join(tmpdir, "s.wav"), seed=5)
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        cfg = {
            "output_layout": "flat",
            "output_format": "wav",
            "num_workers": 1,
            "min_duration": 0.5,
            "max_duration": 60.0,
        }
        write_manifest(iter([_make_record("ds__s", src)]), input_manifest)

        LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        ckpt_path = os.path.join(out_dir, "checkpoint.json")
        with open(ckpt_path, "r", encoding="utf-8") as f:
            before = json.load(f)
        assert set(before["done"]) == {"ds__s"}

        # 配置变更（target_rms 不同 -> config_hash 不同）：旧断点被忽略，重新处理
        cfg2 = dict(cfg, target_rms=0.2)
        LoudnessNormalizerProcessor(cfg2).process(input_manifest, out_dir)
        with open(ckpt_path, "r", encoding="utf-8") as f:
            after = json.load(f)
        # 处理结果仍产出（全量重算），且 checkpoint 以新配置落盘
        assert set(after["done"]) == {"ds__s"}


def test_d2_multi_worker_checkpoint():
    """num_workers>1 时多线程处理 + checkpoint 恢复正常工作。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        srcs = [_make_wav(os.path.join(tmpdir, f"s{i}.wav"), seed=i) for i in range(5)]
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        cfg = {
            "output_layout": "flat",
            "output_format": "wav",
            "num_workers": 4,
            "min_duration": 0.5,
            "max_duration": 60.0,
        }
        write_manifest(iter([_make_record(f"ds__s{i}", srcs[i]) for i in range(5)]), input_manifest)

        out_manifest = LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        recs = list(read_manifest(out_manifest))
        assert len(recs) == 5
        assert all(r.is_stage_passed("D2") for r in recs)

        # 重跑（resume）：全部 checkpoint 命中恢复，输出一致
        LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        recs2 = list(read_manifest(out_manifest))
        assert len(recs2) == 5
        assert all(r.is_stage_passed("D2") for r in recs2)


def test_d2_rejected_record_resume():
    """rejected 记录也写 checkpoint；resume 时因产物（输入缺失）不可读回退重处理，
    结果幂等（仍 rejected）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        missing = os.path.join(tmpdir, "missing.wav")  # 文件不存在
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        cfg = {
            "output_layout": "flat",
            "output_format": "wav",
            "num_workers": 1,
            "min_duration": 0.5,
            "max_duration": 60.0,
        }
        write_manifest(iter([_make_record("ds__missing", missing)]), input_manifest)

        out_manifest = LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        recs = list(read_manifest(out_manifest))
        assert len(recs) == 1
        assert recs[0].rejected_reason == "d2_missing_audio"

        # resume：prev 的 audio.path（缺失输入）不可读 -> 回退重处理 -> 仍 rejected
        LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        recs2 = list(read_manifest(out_manifest))
        assert len(recs2) == 1
        assert recs2[0].rejected_reason == "d2_missing_audio"


def test_d2_upstream_config_change_rejects_checkpoint():
    """上游配置变更（输入 manifest 的 provenance.config_hash 变化）时，
    即使本阶段配置不变，D2 checkpoint 也必须失效重算——避免恢复基于旧输入的结果。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_wav(os.path.join(tmpdir, "s.wav"), seed=7)
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        cfg = {
            "output_layout": "flat",
            "output_format": "wav",
            "num_workers": 1,
            "min_duration": 0.5,
            "max_duration": 60.0,
        }

        def _rec_with_upstream(hash_val):
            rec = _make_record("ds__s", src)
            rec.provenance = {"config_hash": hash_val}
            return rec

        write_manifest(iter([_rec_with_upstream("full_v1")]), input_manifest)
        out_manifest = LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        ckpt_path = os.path.join(out_dir, "checkpoint.json")
        with open(ckpt_path, "r", encoding="utf-8") as f:
            first_hash = json.load(f)["config_hash"]
        assert list(read_manifest(out_manifest))[0].is_stage_passed("D2")

        # 上游全量配置 hash 变化（如改 B2 静音参数），本阶段配置不变
        write_manifest(iter([_rec_with_upstream("full_v2")]), input_manifest)
        LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        with open(ckpt_path, "r", encoding="utf-8") as f:
            second_hash = json.load(f)["config_hash"]
        assert second_hash != first_hash
        # 全部记录被重算（checkpoint 更新为新指纹）
        assert list(read_manifest(out_manifest))[0].is_stage_passed("D2")

        # 补强：证明指纹确实包含上游 config_hash 的贡献，而非仅因 manifest
        # 重写（mtime/size）变化而变化。构造一个除 config_hash 外逐字节相同、
        # size 相同且 mtime 对齐的 manifest，指纹仍必须不同——差异只能来自
        # 上游 hash 本身。
        from noise_filter_pipeline.core.manifest_io import build_checkpoint_fingerprint

        v3_path = os.path.join(tmpdir, "input_v3.jsonl")
        with open(input_manifest, "r", encoding="utf-8") as f:
            v3_content = f.read().replace('"full_v2"', '"full_v3"')
        with open(v3_path, "w", encoding="utf-8") as f:
            f.write(v3_content)
        st = os.stat(input_manifest)
        os.utime(v3_path, (st.st_atime, st.st_mtime))
        assert os.path.getsize(v3_path) == os.path.getsize(input_manifest)
        fp_v3 = build_checkpoint_fingerprint(cfg, v3_path)
        assert fp_v3 != second_hash


def test_d2_oserror_marks_failed_and_retried_on_rerun(monkeypatch):
    """D2 异常分类：OSError 属基础设施失败——标 failed（非终态 rejected）、
    不设 rejected_reason、不写 checkpoint，重跑时应重新处理；
    内容级异常（ValueError 等）保持 rejected 终态并写 checkpoint。"""
    import noise_filter_pipeline.processors.loudness_normalizer as norm_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_wav(os.path.join(tmpdir, "s1.wav"), seed=1)
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        write_manifest(iter([_make_record("ds__s1", src)]), input_manifest)
        out_dir = os.path.join(tmpdir, "out")
        cfg = {
            "output_layout": "flat",
            "output_format": "wav",
            "num_workers": 1,
            "min_duration": 0.5,
            "max_duration": 60.0,
        }

        def _boom_oserror(*args, **kwargs):
            raise OSError("disk full")

        # 打桩只覆盖第一次跑：monkeypatch 默认作用域是整个测试函数，
        # 若不加 context 限制，下面的"重跑（不再故障）"仍会命中 OSError。
        with monkeypatch.context() as m:
            m.setattr(norm_mod, "save_audio_atomically", _boom_oserror)
            proc = LoudnessNormalizerProcessor(cfg)
            out_manifest = proc.process(input_manifest, out_dir)

        recs = list(read_manifest(out_manifest))
        assert len(recs) == 1
        d2 = recs[0].stages.get("D2", {})
        assert d2.get("status") == "failed", d2
        assert d2.get("retryable") is True, d2
        assert not recs[0].rejected_reason
        # 基础设施失败不写 checkpoint
        with open(os.path.join(out_dir, "checkpoint.json"), "r", encoding="utf-8") as f:
            done = set(json.load(f).get("done", []))
        assert "ds__s1" not in done

        # 重跑（不再故障）：未写 checkpoint 的记录被重新处理并通过
        proc2 = LoudnessNormalizerProcessor(cfg)
        out_manifest2 = proc2.process(input_manifest, out_dir)
        recs2 = list(read_manifest(out_manifest2))
        assert len(recs2) == 1
        assert recs2[0].is_stage_passed("D2")
        assert os.path.isfile(recs2[0].audio["path"])

        # 内容级异常（ValueError）保持 rejected 终态且写 checkpoint（独立输出目录，
        # 避免复用上一段已 passed 的 checkpoint）
        out_dir3 = os.path.join(tmpdir, "out3")

        def _boom_value(*args, **kwargs):
            raise ValueError("bad content")

        with monkeypatch.context() as m:
            m.setattr(norm_mod, "save_audio_atomically", _boom_value)
            proc3 = LoudnessNormalizerProcessor(cfg)
            out_manifest3 = proc3.process(input_manifest, out_dir3)
        recs3 = list(read_manifest(out_manifest3))
        assert len(recs3) == 1
        assert recs3[0].stages.get("D2", {}).get("status") == "rejected"
        assert recs3[0].rejected_reason == "d2_exception"
        with open(os.path.join(out_dir3, "checkpoint.json"), "r", encoding="utf-8") as f:
            done3 = set(json.load(f).get("done", []))
        assert "ds__s1" in done3


def test_d2_invalid_output_layout_raises():
    """非法 output_layout 取值应在 __init__ 即抛 ValueError，不得静默回退。"""
    with pytest.raises(ValueError, match="output_layout"):
        LoudnessNormalizerProcessor({"output_layout": "bogus"})
    # 合法取值不受影响
    for layout in ("flat", "sr_dataset", "sr_dataset_label", "sr_label_dataset", "dataset_label"):
        LoudnessNormalizerProcessor({"output_layout": layout})


@pytest.mark.parametrize(
    "output_format",
    (None, "", ".flac", "flac/wav", "../flac", "flac\\\\wav", "flac..wav"),
)
def test_d2_rejects_unsafe_output_format_for_all_layouts(output_format):
    """output_format 必须是安全扩展名 token，不能注入路径或父目录。"""
    for layout in ("flat", "sr_dataset", "sr_dataset_label", "sr_label_dataset", "dataset_label"):
        with pytest.raises(ValueError, match="output_format"):
            LoudnessNormalizerProcessor({"output_layout": layout, "output_format": output_format})


def test_d2_accepts_alphanumeric_output_format_for_all_layouts():
    """合法字母数字扩展名在全部输出布局下均可构造。"""
    for layout in ("flat", "sr_dataset", "sr_dataset_label", "sr_label_dataset", "dataset_label"):
        proc = LoudnessNormalizerProcessor({"output_layout": layout, "output_format": "FLAC"})
        assert proc.output_format == "FLAC"


def test_d2_sr_label_dataset_layout_with_shard():
    """sr_label_dataset 目标布局：audio/sr_<sr>/<label>/<dataset>/<shard2>/<file>。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_wav(os.path.join(tmpdir, "s.wav"), seed=7)
        rec = _make_record("ds__s", src)
        rec.stages["C3"] = {
            "status": "passed",
            "balance_category": "Inside, small room",
        }
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        write_manifest(iter([rec]), input_manifest)
        out_dir = os.path.join(tmpdir, "out")

        proc = LoudnessNormalizerProcessor({
            "output_layout": "sr_label_dataset", "output_format": "flac",
            "num_workers": 1, "min_duration": 0.5, "max_duration": 60.0,
        })
        out_manifest = proc.process(input_manifest, out_dir)

        out_recs = list(read_manifest(out_manifest))
        assert len(out_recs) == 1
        out_path = out_recs[0].audio["path"]
        rel = os.path.relpath(out_path, os.path.join(out_dir, "audio"))
        parts = rel.split(os.sep)
        # sr / label（sanitize 后）/ dataset / shard2 / 文件名 共 5 段
        assert len(parts) == 5
        assert parts[0] == "sr_16k"
        assert parts[1] == "Inside_small_room"
        assert parts[2] == "ds"
        assert len(parts[3]) == 2 and all(c in "0123456789abcdef" for c in parts[3])
        assert parts[4].endswith(".flac")
        # shard 确定性：同输入重算得到同一路径（base = 去掉 shard2/文件名 两级）
        from tools.shared.audio_io import make_unique_output_path
        base_dir = os.path.dirname(os.path.dirname(out_path))
        assert make_unique_output_path(base_dir, src, ext=".flac", shard=True) == out_path


def test_d2_sr_dataset_label_layout_unchanged():
    """存量布局 sr_dataset_label 不受影响：audio/sr_<sr>/<dataset>/<label>/<file>（无 shard）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_wav(os.path.join(tmpdir, "s.wav"), seed=8)
        rec = _make_record("ds__s", src)
        rec.stages["C3"] = {"status": "passed", "balance_category": "Vehicle"}
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        write_manifest(iter([rec]), input_manifest)
        out_dir = os.path.join(tmpdir, "out")

        proc = LoudnessNormalizerProcessor({
            "output_layout": "sr_dataset_label", "output_format": "flac",
            "num_workers": 1, "min_duration": 0.5, "max_duration": 60.0,
        })
        out_manifest = proc.process(input_manifest, out_dir)

        out_recs = list(read_manifest(out_manifest))
        rel = os.path.relpath(out_recs[0].audio["path"], os.path.join(out_dir, "audio"))
        assert rel.split(os.sep)[:3] == ["sr_16k", "ds", "Vehicle"]


def test_d2_dataset_label_layout_uses_dataset_ordinals_and_provenance():
    """dataset_label 按 dataset 内输入顺序写入 shard 与版本化 provenance。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sources = [
            _make_wav(os.path.join(tmpdir, f"sample{i}.wav"), seed=9 + i)
            for i in range(2)
        ]
        records = []
        for i, src in enumerate(sources):
            rec = _make_record(f"ds__sample{i}", src)
            rec.stages["C3"] = {
                "status": "passed",
                "balance_category": "Inside, small room" if i == 0 else "Vehicle",
            }
            records.append(rec)
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        write_manifest(iter(records), input_manifest)

        out_manifest = LoudnessNormalizerProcessor({
            "output_layout": "dataset_label", "output_format": "flac",
            "num_workers": 1, "min_duration": 0.5, "max_duration": 60.0,
        }).process(input_manifest, out_dir)

        out_recs = list(read_manifest(out_manifest))
        assert [r.provenance["dataset_label_ordinal"] for r in out_recs] == [0, 1]
        assert all(r.provenance["dataset_label_naming_version"] == 1 for r in out_recs)
        assert all(r.provenance["dataset_label_shard_size"] == 10000 for r in out_recs)
        paths = [os.path.relpath(r.audio["path"], os.path.join(out_dir, "audio")) for r in out_recs]
        assert paths[0].split(os.sep) == ["ds", "Inside_small_room", "shard_00000", "0000.flac"]
        assert paths[1].split(os.sep) == ["ds", "Vehicle", "shard_00000", "0001.flac"]
        assert all(os.path.isfile(r.audio["path"]) for r in out_recs)


def test_d2_dataset_label_input_order_and_rerun_are_stable_with_multiple_workers():
    """多 worker 不改变 manifest 顺序；同一输入顺序重跑恢复同一路径与 ordinal。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sources = {
            f"ds{i}__sample": _make_wav(
                os.path.join(tmpdir, f"sample{i}.wav"), seed=20 + i
            )
            for i in range(4)
        }
        order = ["ds0__sample", "ds1__sample", "ds0__sample2", "ds1__sample2"]
        sources["ds0__sample2"] = _make_wav(os.path.join(tmpdir, "sample4.wav"), seed=24)
        sources["ds1__sample2"] = _make_wav(os.path.join(tmpdir, "sample5.wav"), seed=25)
        records = []
        for record_id in order:
            dataset_id = record_id.split("__", 1)[0]
            rec = _make_record(record_id, sources[record_id])
            rec.dataset_id = dataset_id
            rec.stages["C3"] = {"status": "passed", "balance_category": "Vehicle"}
            records.append(rec)
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        write_manifest(iter(records), input_manifest)
        cfg = {
            "output_layout": "dataset_label", "output_format": "flac",
            "num_workers": 4, "min_duration": 0.5, "max_duration": 60.0,
        }

        out_manifest = LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        first = list(read_manifest(out_manifest))
        first_signature = [
            (r.record_id, r.provenance["dataset_label_ordinal"], r.audio["path"])
            for r in first
        ]
        first_by_id = {r.record_id: r for r in first}
        assert set(first_by_id) == set(order)
        assert [first_by_id[key].provenance["dataset_label_ordinal"] for key in order] == [0, 0, 1, 1]

        LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        second = list(read_manifest(out_manifest))
        second_signature = sorted(
            (r.record_id, r.provenance["dataset_label_ordinal"], r.audio["path"])
            for r in second
        )
        assert sorted(first_signature) == second_signature


def test_d2_dataset_label_path_helper_shards_at_10000_boundary():
    """ordinal 9999/10000 必须分别落入 shard_00000/9999 与 shard_00001/0000。"""
    proc = LoudnessNormalizerProcessor({
        "output_layout": "dataset_label", "output_format": "flac",
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        proc._output_audio_dir = os.path.join(tmpdir, "audio")
        for ordinal, expected in (
            (9999, os.path.join("ds", "Vehicle", "shard_00000", "9999.flac")),
            (10000, os.path.join("ds", "Vehicle", "shard_00001", "0000.flac")),
        ):
            rec = _make_record("ds__boundary", os.path.join(tmpdir, "source.wav"))
            rec.stages["C3"] = {"status": "passed", "balance_category": "Vehicle"}
            proc._set_dataset_label_provenance(rec, ordinal)
            path = proc._output_path_for(16000, rec, rec.audio["path"], ordinal)
            assert os.path.relpath(path, proc._output_audio_dir) == expected


def test_d2_dataset_label_rejected_and_failed_records_consume_ordinals():
    """rejected/failed 记录也消耗 ordinal，成功记录之间保留可见空洞。"""
    import noise_filter_pipeline.processors.loudness_normalizer as norm_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        good0 = _make_wav(os.path.join(tmpdir, "good0.wav"), seed=30)
        failed = _make_wav(os.path.join(tmpdir, "failed.wav"), seed=31)
        good3 = _make_wav(os.path.join(tmpdir, "good3.wav"), seed=32)
        records = []
        for record_id, path in (
            ("ds__good0", good0),
            ("ds__missing", os.path.join(tmpdir, "missing.wav")),
            ("ds__failed", failed),
            ("ds__good3", good3),
        ):
            rec = _make_record(record_id, path)
            rec.stages["C3"] = {"status": "passed", "balance_category": "Vehicle"}
            records.append(rec)
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        write_manifest(iter(records), input_manifest)
        cfg = {
            "output_layout": "dataset_label", "output_format": "flac",
            "num_workers": 1, "min_duration": 0.5, "max_duration": 60.0,
        }
        original_save = norm_mod.save_audio_atomically

        def _fail_one(audio, sr, dst_path):
            if dst_path.endswith(os.path.join("shard_00000", "0002.flac")):
                raise OSError("disk full")
            return original_save(audio, sr, dst_path)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(norm_mod, "save_audio_atomically", _fail_one)
            out_manifest = LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)

        first = {r.record_id: r for r in read_manifest(out_manifest)}
        assert [first[key].provenance["dataset_label_ordinal"] for key in (
            "ds__good0", "ds__missing", "ds__failed", "ds__good3"
        )] == [0, 1, 2, 3]
        assert first["ds__missing"].rejected_reason == "d2_missing_audio"
        assert first["ds__failed"].stages["D2"]["status"] == "failed"
        assert first["ds__good0"].audio["path"].endswith(os.path.join("shard_00000", "0000.flac"))
        assert first["ds__good3"].audio["path"].endswith(os.path.join("shard_00000", "0003.flac"))
        assert not os.path.exists(os.path.join(out_dir, "audio", "ds", "Vehicle", "shard_00000", "0002.flac"))


def test_d2_dataset_label_old_checkpoint_without_provenance_is_not_restored():
    """旧 checkpoint 缺少 dataset_label provenance 时必须重新处理。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_wav(os.path.join(tmpdir, "source.wav"), seed=40)
        rec = _make_record("ds__source", src)
        rec.stages["C3"] = {"status": "passed", "balance_category": "Vehicle"}
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        write_manifest(iter([rec]), input_manifest)
        cfg = {
            "output_layout": "dataset_label", "output_format": "flac",
            "num_workers": 1, "min_duration": 0.5, "max_duration": 60.0,
        }
        out_manifest = LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        output_file = os.path.join(out_dir, "d2_manifest_ds.jsonl")
        previous = list(read_manifest(output_file))
        previous[0].provenance.pop("dataset_label_naming_version")
        write_manifest(iter(previous), output_file)

        calls = []
        import noise_filter_pipeline.processors.loudness_normalizer as norm_mod
        original_save = norm_mod.save_audio_atomically

        def _record_save(audio, sr, dst_path):
            calls.append(dst_path)
            return original_save(audio, sr, dst_path)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(norm_mod, "save_audio_atomically", _record_save)
            LoudnessNormalizerProcessor(cfg).process(input_manifest, out_dir)
        assert len(calls) == 1


def test_d2_dataset_label_shard_size_is_fail_fast():
    """dataset_label_shard_size 只能使用正式值 10000，非法值不得静默接受。"""
    for shard_size in (0, 9999, 10001, 10000.0, True, "10000", None):
        config = {"output_layout": "dataset_label", "dataset_label_shard_size": shard_size}
        with pytest.raises(ValueError, match="dataset_label_shard_size"):
            LoudnessNormalizerProcessor(config)

    # 未配置时使用正式默认值，不应被误判为非法。
    LoudnessNormalizerProcessor({"output_layout": "dataset_label"})


@pytest.mark.parametrize(
    "c3_stage, message",
    [
        (None, "passed C3"),
        ({"status": "rejected", "balance_category": "Vehicle"}, "passed C3"),
        ({"status": "passed"}, "C3 balance_category"),
    ],
)
def test_d2_category_layout_requires_passed_c3_balance_category(c3_stage, message):
    """category layout 必须从 passed C3 读取 balance_category，否则 fail-closed。"""
    rec = _make_record("ds__s", "/tmp/not-read.wav")
    if c3_stage is not None:
        rec.stages["C3"] = c3_stage
    proc = LoudnessNormalizerProcessor({"output_layout": "sr_dataset_label"})
    proc._output_audio_dir = "/tmp/not-written"

    with pytest.raises(ValueError, match=message):
        proc._output_dir_for(16000, rec)


def test_d2_num_workers_excluded_from_fingerprint():
    """num_workers 只是并发度，不得进入 checkpoint 配置指纹：

    stage_d_enhancement 默认 min(8, cpu_count) 随机器变化，若进入指纹，
    换机/改并发会静默失效 checkpoint 导致全量重算。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_wav(os.path.join(tmpdir, "s.wav"), seed=3)
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        write_manifest(iter([_make_record("ds__s", src)]), input_manifest)
        base_cfg = {"output_layout": "flat", "output_format": "wav", "min_duration": 0.5, "max_duration": 60.0}

        hashes = []
        for workers in (1, 4):
            proc = LoudnessNormalizerProcessor({**base_cfg, "num_workers": workers})
            assert "num_workers" not in proc.config
            assert proc.num_workers == workers
            out_dir = os.path.join(tmpdir, f"out_w{workers}")
            proc.process(input_manifest, out_dir)
            with open(os.path.join(out_dir, "checkpoint.json"), "r", encoding="utf-8") as f:
                hashes.append(json.load(f)["config_hash"])
        assert hashes[0] == hashes[1]


def test_d2_dataset_label_reorders_same_dataset_without_restoring_old_ordinals():
    """同 dataset 重排输入后，旧 checkpoint 记录不得按 record_id 静默恢复。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_a = _make_wav(os.path.join(tmpdir, "a.wav"), seed=101)
        src_b = _make_wav(os.path.join(tmpdir, "b.wav"), seed=202)
        source_a, _ = sf.read(src_a)
        source_b, _ = sf.read(src_b)
        assert not np.array_equal(source_a, source_b)

        def _record(record_id, source_path):
            rec = _make_record(record_id, source_path)
            rec.stages["C3"] = {"status": "passed", "balance_category": "Vehicle"}
            return rec

        input_manifest = os.path.join(tmpdir, "input.jsonl")
        # 使用独立且不变的 fingerprint 输入，确保第二次运行仍会检查旧记录的
        # provenance/path，而不是因为输入 manifest 的 mtime 变化而全量失效 checkpoint。
        fingerprint_anchor = os.path.join(tmpdir, "fingerprint_anchor.jsonl")
        write_manifest(iter([]), fingerprint_anchor)
        out_dir = os.path.join(tmpdir, "out")
        cfg = {
            "output_layout": "dataset_label",
            "output_format": "flac",
            "num_workers": 1,
            "min_duration": 0.5,
            "max_duration": 60.0,
        }

        # 首次顺序为 A,B，建立旧 checkpoint 和旧 ordinal/path。
        write_manifest(iter([_record("ds__A", src_a), _record("ds__B", src_b)]), input_manifest)
        out_manifest = LoudnessNormalizerProcessor(cfg).process(
            input_manifest, out_dir, fingerprint_input=fingerprint_anchor
        )
        first = {rec.record_id: rec for rec in read_manifest(out_manifest)}
        assert [first[record_id].provenance["dataset_label_ordinal"] for record_id in (
            "ds__A", "ds__B"
        )] == [0, 1]
        old_paths = {record_id: rec.audio["path"] for record_id, rec in first.items()}
        assert all(os.path.isfile(path) for path in old_paths.values())

        # 第二次改为 B,A；checkpoint 指纹保持可验证，只有 ordinal/provenance
        # 校验能决定旧记录是否恢复。
        write_manifest(iter([_record("ds__B", src_b), _record("ds__A", src_a)]), input_manifest)
        calls = []
        import noise_filter_pipeline.processors.loudness_normalizer as norm_mod

        original_save = norm_mod.save_audio_atomically

        def _record_save(audio, sr, dst_path):
            calls.append(dst_path)
            return original_save(audio, sr, dst_path)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(norm_mod, "save_audio_atomically", _record_save)
            LoudnessNormalizerProcessor(cfg).process(
                input_manifest, out_dir, fingerprint_input=fingerprint_anchor
            )

        second = {rec.record_id: rec for rec in read_manifest(out_manifest)}
        assert [second[record_id].provenance["dataset_label_ordinal"] for record_id in (
            "ds__B", "ds__A"
        )] == [0, 1]
        assert second["ds__B"].audio["path"] != old_paths["ds__B"]
        assert second["ds__A"].audio["path"] != old_paths["ds__A"]
        assert second["ds__B"].audio["path"].endswith(
            os.path.join("ds", "Vehicle", "shard_00000", "0000.flac")
        )
        assert second["ds__A"].audio["path"].endswith(
            os.path.join("ds", "Vehicle", "shard_00000", "0001.flac")
        )
        assert set(calls) == {rec.audio["path"] for rec in second.values()}
        assert len(calls) == 2

        # 当前 manifest 的 path/provenance 必须对应当前顺序，且两个新产物都可读、
        # 内容仍可区分，避免仅凭文件存在或 mtime 判断重处理成功。
        assert second["ds__B"].stages["D2"]["input_path"] == src_b
        assert second["ds__A"].stages["D2"]["input_path"] == src_a
        output_b, sr_b = sf.read(second["ds__B"].audio["path"])
        output_a, sr_a = sf.read(second["ds__A"].audio["path"])
        assert sr_a == sr_b == 16000
        assert output_a.size > 0 and output_b.size > 0
        assert not np.allclose(output_a, output_b)


@pytest.mark.pipeline
def test_d2_dataset_label_rejects_sanitized_dataset_id_collision():
    """不同原始 dataset_id 清洗到同一目录时必须先报错，禁止覆盖。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sources = [
            _make_wav(os.path.join(tmpdir, name), seed=301 + index)
            for index, name in enumerate(("a.wav", "b.wav"))
        ]
        records = []
        for (record_id, dataset_id), source in zip(
            (("record_a", "dataset A"), ("record_b", "dataset_A")),
            sources,
        ):
            rec = _make_record(record_id, source)
            rec.dataset_id = dataset_id
            rec.stages["C3"] = {"status": "passed", "balance_category": "Vehicle"}
            records.append(rec)

        input_manifest = os.path.join(tmpdir, "input.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        write_manifest(iter(records), input_manifest)

        with pytest.raises(ValueError, match="dataset_id|目录|collision"):
            LoudnessNormalizerProcessor({
                "output_layout": "dataset_label", "output_format": "flac",
                "num_workers": 1, "min_duration": 0.5, "max_duration": 60.0,
            }).process(input_manifest, out_dir)

        assert not os.path.exists(
            os.path.join(out_dir, "audio", "dataset_A", "Vehicle", "shard_00000", "0000.flac")
        )


@pytest.mark.pipeline
def test_d2_dataset_label_rejects_sanitized_category_collision():
    """同一 dataset 内原始 category 清洗到同一目录时必须先报错。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sources = [
            _make_wav(os.path.join(tmpdir, name), seed=311 + index)
            for index, name in enumerate(("a.wav", "b.wav"))
        ]
        records = []
        for record_id, category, source_path in (
            ("record_a", "A B", sources[0]),
            ("record_b", "A_B", sources[1]),
        ):
            rec = _make_record(record_id, source_path)
            rec.stages["C3"] = {"status": "passed", "balance_category": category}
            records.append(rec)

        input_manifest = os.path.join(tmpdir, "input.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        write_manifest(iter(records), input_manifest)

        with pytest.raises(ValueError, match="category|目录|collision"):
            LoudnessNormalizerProcessor({
                "output_layout": "dataset_label", "output_format": "flac",
                "num_workers": 1, "min_duration": 0.5, "max_duration": 60.0,
            }).process(input_manifest, out_dir)

        assert not os.path.exists(
            os.path.join(out_dir, "audio", "ds", "A_B", "shard_00000", "0000.flac")
        )


@pytest.mark.pipeline
def test_d2_dataset_label_checkpoint_category_change_reprocesses():
    """ordinal 不变但 C3 category 变化时，checkpoint 不得恢复旧目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_wav(os.path.join(tmpdir, "source.wav"), seed=321)
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        fingerprint_anchor = os.path.join(tmpdir, "fingerprint_anchor.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        write_manifest(iter([]), fingerprint_anchor)
        cfg = {
            "output_layout": "dataset_label", "output_format": "flac",
            "num_workers": 1, "min_duration": 0.5, "max_duration": 60.0,
        }

        def _record(category):
            rec = _make_record("shared", src)
            rec.stages["C3"] = {"status": "passed", "balance_category": category}
            return rec

        write_manifest(iter([_record("Vehicle")]), input_manifest)
        out_manifest = LoudnessNormalizerProcessor(cfg).process(
            input_manifest, out_dir, fingerprint_input=fingerprint_anchor
        )
        first = list(read_manifest(out_manifest))[0]
        old_path = first.audio["path"]

        write_manifest(iter([_record("Music")]), input_manifest)
        calls = []
        import noise_filter_pipeline.processors.loudness_normalizer as norm_mod
        original_save = norm_mod.save_audio_atomically

        def _record_save(audio, sr, dst_path):
            calls.append(dst_path)
            return original_save(audio, sr, dst_path)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(norm_mod, "save_audio_atomically", _record_save)
            LoudnessNormalizerProcessor(cfg).process(
                input_manifest, out_dir, fingerprint_input=fingerprint_anchor
            )

        second = list(read_manifest(out_manifest))[0]
        assert len(calls) == 1
        assert second.audio["path"] != old_path
        assert second.audio["path"].endswith(
            os.path.join("ds", "Music", "shard_00000", "0000.flac")
        )
        assert os.path.isfile(second.audio["path"])


@pytest.mark.pipeline
def test_d2_dataset_label_checkpoint_dataset_change_reprocesses():
    """record_id 错误复用但 dataset_id 变化时，checkpoint 不得恢复旧记录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_wav(os.path.join(tmpdir, "source.wav"), seed=331)
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        fingerprint_anchor = os.path.join(tmpdir, "fingerprint_anchor.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        write_manifest(iter([]), fingerprint_anchor)
        cfg = {
            "output_layout": "dataset_label", "output_format": "flac",
            "num_workers": 1, "min_duration": 0.5, "max_duration": 60.0,
        }

        def _record(dataset_id):
            rec = _make_record("reused-record-id", src)
            rec.dataset_id = dataset_id
            rec.stages["C3"] = {"status": "passed", "balance_category": "Vehicle"}
            return rec

        write_manifest(iter([_record("old_ds")]), input_manifest)
        out_manifest = LoudnessNormalizerProcessor(cfg).process(
            input_manifest, out_dir, fingerprint_input=fingerprint_anchor
        )
        old = list(read_manifest(out_manifest))[0]
        old_path = old.audio["path"]

        write_manifest(iter([_record("new_ds")]), input_manifest)
        calls = []
        import noise_filter_pipeline.processors.loudness_normalizer as norm_mod
        original_save = norm_mod.save_audio_atomically

        def _record_save(audio, sr, dst_path):
            calls.append(dst_path)
            return original_save(audio, sr, dst_path)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(norm_mod, "save_audio_atomically", _record_save)
            LoudnessNormalizerProcessor(cfg).process(
                input_manifest, out_dir, fingerprint_input=fingerprint_anchor
            )

        new_manifest = os.path.join(out_dir, "d2_manifest_new_ds.jsonl")
        assert len(calls) == 1
        assert os.path.isfile(new_manifest)
        new = list(read_manifest(new_manifest))[0]
        assert new.dataset_id == "new_ds"
        assert new.audio["path"] != old_path
        assert new.audio["path"].endswith(
            os.path.join("new_ds", "Vehicle", "shard_00000", "0000.flac")
        )
        assert os.path.isfile(new.audio["path"])


@pytest.mark.parametrize(
    "mutation",
    ("source_id", "chunk_id", "split", "audio_path", "fingerprint"),
)
def test_d2_dataset_label_checkpoint_source_identity_changes_reprocess(
    mutation,
):
    """dataset_label 恢复必须核对源身份及原始音频路径/指纹。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_wav(os.path.join(tmpdir, "source.wav"), seed=341)
        alternate_src = _make_wav(os.path.join(tmpdir, "alternate.wav"), seed=342)
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        fingerprint_anchor = os.path.join(tmpdir, "fingerprint_anchor.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        write_manifest(iter([]), fingerprint_anchor)
        cfg = {
            "output_layout": "dataset_label", "output_format": "flac",
            "num_workers": 1, "min_duration": 0.5, "max_duration": 60.0,
        }

        def _record(
            *,
            source_id="source-v1",
            chunk_id="chunk-v1",
            split="train",
            audio_path=src,
        ):
            rec = AudioRecord(
                record_id="stable-record-id",
                dataset_id="stable-dataset",
                source_id=source_id,
                chunk_id=chunk_id,
                split=split,
                audio={"path": audio_path, "sr": 16000, "duration": 3.0},
                labels={"audioset": "noise"},
            )
            rec.stages["C3"] = {
                "status": "passed", "balance_category": "Vehicle",
            }
            return rec

        write_manifest(iter([_record()]), input_manifest)
        out_manifest = LoudnessNormalizerProcessor(cfg).process(
            input_manifest, out_dir, fingerprint_input=fingerprint_anchor
        )
        first = list(read_manifest(out_manifest))[0]
        old_path = first.audio["path"]
        assert first.provenance["dataset_label_ordinal"] == 0

        mutation_kwargs = {
            "source_id": "source-v2",
            "chunk_id": "chunk-v2",
            "split": "test",
            "audio_path": alternate_src,
        }
        if mutation == "fingerprint":
            before = os.stat(src)
            _make_wav(src, seed=343)
            os.utime(src, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
        second_kwargs = {mutation: mutation_kwargs[mutation]} if mutation != "fingerprint" else {}
        write_manifest(iter([_record(**second_kwargs)]), input_manifest)

        calls = []
        import noise_filter_pipeline.processors.loudness_normalizer as norm_mod
        original_save = norm_mod.save_audio_atomically

        def _record_save(audio, sr, dst_path):
            calls.append(dst_path)
            return original_save(audio, sr, dst_path)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(norm_mod, "save_audio_atomically", _record_save)
            LoudnessNormalizerProcessor(cfg).process(
                input_manifest, out_dir, fingerprint_input=fingerprint_anchor
            )

        second = list(read_manifest(out_manifest))[0]
        assert second.provenance["dataset_label_ordinal"] == 0
        assert len(calls) == 1
        assert calls == [second.audio["path"]]
        assert second.audio["path"] == old_path
        assert second.stages["D2"]["input_path"] == second_kwargs.get(
            "audio_path", src
        )


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    (("duration", "d2_duration_out_of_range"), ("zero", "d2_zero_rms")),
)
def test_d2_dataset_label_rejected_resume_skips_normalize_and_save(
    case, expected_reason,
):
    """可恢复的 duration/zero rejected 重跑不得再次 normalize 或 save。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, f"{case}.wav")
        if case == "duration":
            _make_wav(src, duration=0.25, seed=351)
        else:
            sf.write(src, np.zeros(16000 * 3, dtype=np.float32), 16000)
        input_manifest = os.path.join(tmpdir, "input.jsonl")
        fingerprint_anchor = os.path.join(tmpdir, "fingerprint_anchor.jsonl")
        out_dir = os.path.join(tmpdir, "out")
        write_manifest(iter([]), fingerprint_anchor)
        rec = _make_record("rejected-record", src)
        rec.dataset_id = "rejected-dataset"
        rec.stages["C3"] = {"status": "passed", "balance_category": "Vehicle"}
        write_manifest(iter([rec]), input_manifest)
        cfg = {
            "output_layout": "dataset_label", "output_format": "flac",
            "num_workers": 1, "min_duration": 0.5, "max_duration": 60.0,
        }

        normalize_calls = []
        save_calls = []
        import noise_filter_pipeline.processors.loudness_normalizer as norm_mod
        original_normalize = norm_mod.LoudnessNormalizerProcessor._normalize
        original_save = norm_mod.save_audio_atomically

        def _load_without_silence_rejection(path, **kwargs):
            audio, sr = sf.read(path, dtype="float32")
            return np.asarray(audio), sr

        def _record_normalize(processor, audio, sr):
            normalize_calls.append(True)
            return original_normalize(processor, audio, sr)

        def _record_save(audio, sr, dst_path):
            save_calls.append(dst_path)
            return original_save(audio, sr, dst_path)

        with pytest.MonkeyPatch.context() as monkeypatch:
            # 让全零 WAV 到达 D2._normalize，以覆盖 zero_rms rejected。
            monkeypatch.setattr(norm_mod, "load_audio_robust", _load_without_silence_rejection)
            out_manifest = LoudnessNormalizerProcessor(cfg).process(
                input_manifest, out_dir, fingerprint_input=fingerprint_anchor
            )
            first = list(read_manifest(out_manifest))[0]
            assert first.rejected_reason == expected_reason
            assert first.stages["D2"]["status"] == "rejected"

            monkeypatch.setattr(
                norm_mod.LoudnessNormalizerProcessor, "_normalize", _record_normalize
            )
            monkeypatch.setattr(norm_mod, "save_audio_atomically", _record_save)
            LoudnessNormalizerProcessor(cfg).process(
                input_manifest, out_dir, fingerprint_input=fingerprint_anchor
            )

        second = list(read_manifest(out_manifest))[0]
        assert second.rejected_reason == expected_reason
        assert second.stages["D2"]["status"] == "rejected"
        assert normalize_calls == []
        assert save_calls == []
