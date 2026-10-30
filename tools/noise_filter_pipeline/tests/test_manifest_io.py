"""
Tests for core.manifest_io
"""
import os
import json
import tempfile
import pytest

from noise_filter_pipeline.core.manifest_io import (
    AudioRecord,
    BatchedManifestWriter,
    append_manifest,
    write_manifest,
    read_manifest,
    read_manifest_strict,
    sanitize_stage_c_input_record,
    sanitize_stage_c2_input_record,
    validate_record,
    count_manifest,
    filter_by_stage,
    write_json_file_transactionally,
    write_jsonl_transactionally,
)
from noise_filter_pipeline.core.manifest_per_dataset import (
    PerDatasetManifestWriter,
    find_legacy_single_manifests,
    reject_legacy_single_manifest,
    split_manifest_by_dataset,
)

from .module_patch_test_helpers import patched_os

pytestmark = pytest.mark.unit


def test_audio_record_roundtrip():
    rec = AudioRecord(
        record_id="ds__src__chunk_00",
        dataset_id="ds",
        source_id="src",
        chunk_id="chunk_00",
        split="train",
        audio={"path": "/tmp/a.wav", "sr": 16000, "duration": 10.0},
        labels={"original": "Ocean"},
    )
    rec.mark_stage("A", "passed")
    data = rec.to_dict()
    restored = AudioRecord.from_dict(data)
    assert restored.record_id == rec.record_id
    assert restored.is_stage_passed("A")


@pytest.mark.parametrize(
    "record_patch, expected_field",
    [
        ({"stages": {"C1": {"status": "passed"}}}, "stages.C1"),
        ({"stages": {"C4": {"status": "passed"}}}, "stages.C4"),
        ({"labels": {"audioset_top5": []}}, "labels.audioset_top5"),
        ({"labels": {"leaf": "Rain"}}, "labels.leaf"),
        ({"rejected_reason": "c3_reject"}, "rejected_reason"),
        ({"provenance": {"stage_c_execution": {}}}, "provenance.stage_c_execution"),
    ],
)
def test_sanitize_stage_c_input_rejects_precomputed_evidence(
    record_patch,
    expected_field,
):
    data = {
        "record_id": "ds__src__chunk_00",
        "stages": {"A": {"status": "passed"}},
        "labels": {"original": "source-noise"},
        "provenance": {"source": "original"},
        **record_patch,
    }

    with pytest.raises(ValueError, match=expected_field):
        sanitize_stage_c_input_record(data)


def test_sanitize_stage_c_input_accepts_clean_upstream_record_unchanged():
    data = {
        "record_id": "ds__src__chunk_00",
        "stages": {"A": {"status": "passed"}, "B4": {"status": "passed"}},
        "labels": {"original": "source-noise"},
        "provenance": {"source": "original"},
    }

    assert sanitize_stage_c_input_record(data) is data


def test_sanitize_stage_c2_input_rejects_derived_fields_but_allows_c1():
    data = {
        "record_id": "ds__src__chunk_00",
        "stages": {"C1": {"status": "passed", "intelligibility": "noise"}},
        "labels": {
            "original": "source-noise",
            "admission_decision": "accept_noise",
        },
    }

    with pytest.raises(ValueError, match="labels.admission_decision"):
        sanitize_stage_c2_input_record(data)

    clean = {
        "record_id": "ds__src__chunk_01",
        "stages": {"C1": {"status": "passed", "intelligibility": "noise"}},
        "labels": {"original": "source-noise"},
    }
    assert sanitize_stage_c2_input_record(clean) is clean


def test_write_read_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.jsonl")
        recs = [
            AudioRecord(
                record_id=f"ds__src__{i}",
                dataset_id="ds",
                source_id="src",
                chunk_id=f"chunk_{i}",
                audio={"path": f"/tmp/{i}.wav", "sr": 16000, "duration": 1.0},
            )
            for i in range(5)
        ]
        write_manifest(iter(recs), path)
        read_recs = list(read_manifest(path))
        assert len(read_recs) == 5
        assert read_recs[0].record_id == "ds__src__0"


def test_write_manifest_rejects_nan_without_touching_old_file(tmp_path):
    """NaN/Inf 是非法 JSON：写入侧必须报错，且不得破坏已有 manifest。"""
    path = tmp_path / "m.jsonl"
    path.write_text('{"record_id": "old"}\n', encoding="utf-8")
    rec = AudioRecord(
        record_id="x", dataset_id="ds", source_id="src",
        audio={"path": "/a.wav", "sr": 16000, "duration": float("nan")},
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        write_manifest(iter([rec]), str(path))

    assert path.read_text(encoding="utf-8") == '{"record_id": "old"}\n'


def test_append_manifest_rejects_nan(tmp_path):
    """append 路径与 write 路径同口径，拒绝写出非标准 JSON。"""
    path = tmp_path / "m.jsonl"
    rec = AudioRecord(
        record_id="x", dataset_id="ds", source_id="src",
        audio={"path": "/a.wav", "sr": 16000, "duration": float("inf")},
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        append_manifest(iter([rec]), str(path))


def test_validate_record_missing_audio_path():
    rec = AudioRecord(
        record_id="x",
        dataset_id="ds",
        source_id="src",
        audio={"sr": 16000, "duration": 1.0},  # missing path
    )
    errors = validate_record(rec)
    assert any("path" in e for e in errors)


def test_transactional_json_rejects_nan_and_preserves_old_output(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Out of range float values"):
        write_json_file_transactionally({"score": float("nan")}, path)

    assert path.read_text(encoding="utf-8") == "old\n"


def test_transactional_jsonl_rejects_nan_and_preserves_old_output(tmp_path):
    path = tmp_path / "review.jsonl"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Out of range float values"):
        write_jsonl_transactionally(
            [{"record_id": "ok"}, {"record_id": "bad", "score": float("nan")}],
            path,
        )

    assert path.read_text(encoding="utf-8") == "old\n"


def test_transactional_writer_rejects_protected_output_aliases(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("source\n", encoding="utf-8")
    alias = tmp_path / "alias.jsonl"
    os.link(source, alias)

    with pytest.raises(ValueError):
        write_jsonl_transactionally(
            [{"record_id": "new"}], alias, protected_paths=(source,)
        )

    assert source.read_text(encoding="utf-8") == "source\n"
    assert alias.read_text(encoding="utf-8") == "source\n"


def test_transactional_writer_rejects_symlink_output_and_protected_root(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("source\n", encoding="utf-8")
    symlink = tmp_path / "alias.jsonl"
    symlink.symlink_to(source)

    with pytest.raises(ValueError, match="符号链接"):
        write_jsonl_transactionally([{"record_id": "new"}], symlink)

    input_root = tmp_path / "stage_c"
    input_root.mkdir()
    manifest = input_root / "c_manifest_test.jsonl"
    manifest.write_text("source\n", encoding="utf-8")
    output = input_root / "review.jsonl"
    with pytest.raises(ValueError, match="受保护输入目录"):
        write_jsonl_transactionally(
            [{"record_id": "new"}], output, protected_roots=(input_root,)
        )

    assert manifest.read_text(encoding="utf-8") == "source\n"
    assert not output.exists()


def test_read_manifest_skips_non_dict_json_lines():
    """合法 JSON 但非 object 的行（纯数字/数组）应告警跳过，不得中断整个读取。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.jsonl")
        good = AudioRecord(
            record_id="r1", dataset_id="ds", source_id="s1",
            audio={"path": "/a.wav", "sr": 16000, "duration": 1.0},
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write("123\n")
            f.write('["a", "b"]\n')
            f.write(json.dumps(good.to_dict(), ensure_ascii=False) + "\n")
            f.write('"just a string"\n')
        recs = list(read_manifest(path))
        assert [r.record_id for r in recs] == ["r1"]


@pytest.mark.parametrize(
    "content, expected",
    [
        ("{broken json}\n", "invalid JSON"),
        ("[]\n", "JSON object"),
        (json.dumps({"dataset_id": "ds", "source_id": "src", "audio": {"path": "/a.wav", "sr": 16000, "duration": 1.0}}) + "\n", "record_id"),
        (json.dumps({"record_id": "r1", "dataset_id": "ds", "source_id": "src", "audio": {"sr": 16000, "duration": 1.0}}) + "\n", "path"),
        (json.dumps({"record_id": "r1", "dataset_id": "ds", "source_id": "src", "audio": {"path": "/a.wav", "sr": True, "duration": 1.0}}) + "\n", "positive integer"),
        (json.dumps({"record_id": "r1", "dataset_id": "ds", "source_id": "src", "audio": {"path": "/a.wav", "sr": 0, "duration": 1.0}}) + "\n", "positive integer"),
        (json.dumps({"record_id": "r1", "dataset_id": "ds", "source_id": "src", "audio": {"path": "/a.wav", "sr": 16000, "duration": float("nan")}}) + "\n", "positive finite"),
        (json.dumps({"record_id": "r1", "dataset_id": "ds", "source_id": "src", "audio": {"path": "/a.wav", "sr": 16000, "duration": float("inf")}}) + "\n", "positive finite"),
        (json.dumps({"record_id": "r1", "dataset_id": "ds", "source_id": "src", "audio": {"path": "/a.wav", "sr": 16000, "duration": False}}) + "\n", "positive finite"),
    ],
    ids=[
        "invalid-json",
        "non-object",
        "missing-record-id",
        "missing-audio-path",
        "bool-sr",
        "zero-sr",
        "nan-duration",
        "infinite-duration",
        "bool-duration",
    ],
)
def test_read_manifest_strict_rejects_invalid_lines(tmp_path, content, expected):
    path = tmp_path / "strict.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        list(read_manifest_strict(str(path)))

    assert f"{path}:1" in str(exc_info.value)
    assert expected in str(exc_info.value)


def test_read_manifest_strict_rejects_blank_line_with_location(tmp_path):
    path = tmp_path / "strict.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"strict\.jsonl:1: blank JSONL line"):
        list(read_manifest_strict(str(path)))


def test_read_manifest_strict_rejects_duplicate_ids_when_requested(tmp_path):
    path = tmp_path / "strict.jsonl"
    record = {
        "record_id": "r1",
        "dataset_id": "ds",
        "source_id": "src",
        "audio": {"path": "/a.wav", "sr": 16000, "duration": 1.0},
    }
    path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        list(read_manifest_strict(str(path), reject_duplicate_ids=True))

    assert f"{path}:2" in str(exc_info.value)
    assert "duplicate record_id" in str(exc_info.value)


def test_read_manifest_strict_rejects_snapshot_change(tmp_path):
    """消费迭代器期间原地改写 manifest，结束复验必须 fail closed。"""
    path = tmp_path / "strict.jsonl"
    records = [
        {
            "record_id": f"r{index}",
            "dataset_id": "ds",
            "source_id": f"src{index}",
            "audio": {"path": f"/{index}.wav", "sr": 16000, "duration": 1.0},
        }
        for index in range(2)
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")

    reader = read_manifest_strict(str(path))
    assert next(reader).record_id == "r0"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        list(reader)


def test_read_manifest_strict_rejects_same_size_path_replacement(tmp_path):
    """读取中原子替换为同 size 文件也必须被 FD/path identity 复验拦截。"""
    path = tmp_path / "strict.jsonl"
    first = {
        "record_id": "r0", "dataset_id": "ds", "source_id": "src0",
        "audio": {"path": "/0.wav", "sr": 16000, "duration": 1.0},
    }
    second = {
        "record_id": "r1", "dataset_id": "ds", "source_id": "src1",
        "audio": {"path": "/1.wav", "sr": 16000, "duration": 1.0},
    }
    original = json.dumps(first) + "\n" + json.dumps(second) + "\n"
    path.write_text(original, encoding="utf-8")
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text(original.replace('"r1"', '"x1"'), encoding="utf-8")
    assert replacement.stat().st_size == path.stat().st_size

    reader = read_manifest_strict(str(path))
    assert next(reader).record_id == "r0"
    os.replace(replacement, path)
    with pytest.raises(ValueError, match="snapshot changed"):
        list(reader)


def test_validate_record_non_dict_audio():
    """audio 字段为非 dict 时按缺失处理报错，不得 AttributeError。"""
    rec = AudioRecord(
        record_id="x", dataset_id="ds", source_id="src",
        audio=["not", "a", "dict"],
    )
    errors = validate_record(rec)
    assert any("path" in e for e in errors)


def test_filter_by_stage():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.jsonl")
        recs = [
            AudioRecord(record_id="r1", dataset_id="ds", source_id="s1", audio={"path": "/a.wav", "sr": 16000, "duration": 1.0}),
            AudioRecord(record_id="r2", dataset_id="ds", source_id="s2", audio={"path": "/b.wav", "sr": 16000, "duration": 1.0}),
        ]
        recs[0].mark_stage("A", "passed")
        recs[1].mark_stage("A", "rejected")
        write_manifest(iter(recs), path)
        passed = list(filter_by_stage(path, "A", "passed"))
        assert len(passed) == 1
        assert passed[0].record_id == "r1"


def test_count_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.jsonl")
        recs = [AudioRecord(record_id=f"r{i}", dataset_id="ds", source_id="s", audio={"path": "/a.wav", "sr": 16000, "duration": 1.0}) for i in range(3)]
        write_manifest(iter(recs), path)
        assert count_manifest(path) == 3


def test_batched_manifest_writer():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.jsonl")
        recs = [
            AudioRecord(record_id=f"r{i}", dataset_id="ds", source_id="s", audio={"path": "/a.wav", "sr": 16000, "duration": 1.0})
            for i in range(7)
        ]

        with BatchedManifestWriter(path, batch_size=3) as writer:
            for rec in recs:
                writer.append(rec)
            # 7 条中前 6 条已满两次 batch 刷盘，第 7 条仍在缓冲
            assert count_manifest(path) == 6

        # __exit__ 刷出尾部
        assert count_manifest(path) == 7
        read_recs = list(read_manifest(path))
        assert [r.record_id for r in read_recs] == [f"r{i}" for i in range(7)]


def test_batched_manifest_writer_concurrent_flush():
    """多线程并发 append+flush 后，读回的行数与 JSON 完整性必须正确。

    回归测试：flush 曾在锁外写盘，两个线程并发 flush 会导致大行 JSONL
    交错损坏（行数少于写入数或出现 JSON 解析失败的行）。

    并发测试固有局限：交错缺陷只在特定调度窗口触发，测试通过不代表
    无 bug（假阴）；本机文件系统/调度器差异也可能导致偶发失败（假阳）。
    此测试作为回归锚保留，不作确定性证明。
    """
    import threading

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "concurrent.jsonl")
        num_threads = 8
        recs_per_thread = 200
        # 大行记录放大交错窗口
        big_label = "x" * 4096

        with BatchedManifestWriter(path, batch_size=10) as writer:
            def worker(tid: int) -> None:
                for i in range(recs_per_thread):
                    rec = AudioRecord(
                        record_id=f"t{tid}__r{i}",
                        dataset_id="ds",
                        source_id=f"s{tid}",
                        audio={"path": f"/tmp/t{tid}_{i}.wav", "sr": 16000, "duration": 1.0},
                        labels={"original": big_label},
                    )
                    writer.append(rec)
                    # 与 append 触发的 flush 竞争
                    if i % 7 == 0:
                        writer.flush()

            threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        expected = num_threads * recs_per_thread
        read_recs = list(read_manifest(path))
        assert len(read_recs) == expected, "并发 flush 后行数必须等于写入总数"
        # read_manifest 对 JSON 解析失败的行只 warning 跳过，能读到完整集合即说明无交错损坏
        assert {r.record_id for r in read_recs} == {
            f"t{t}__r{i}" for t in range(num_threads) for i in range(recs_per_thread)
        }


def test_link_or_copy_file(tmp_path):
    """硬链接成功时两路径共享 inode；目标已存在时先删再链。"""
    from noise_filter_pipeline.core.manifest_io import link_or_copy_file

    src = tmp_path / "src.jsonl"
    src.write_text('{"a": 1}\n', encoding="utf-8")
    dst = tmp_path / "dst.jsonl"
    link_or_copy_file(str(src), str(dst))
    assert dst.read_text(encoding="utf-8") == '{"a": 1}\n'
    if hasattr(os, "link"):
        assert os.stat(src).st_ino == os.stat(dst).st_ino

    # 目标已存在：先删再链（旧内容被替换）。
    # 注意先 unlink 再写：直接 write_text 会透过硬链接改写共享 inode
    os.unlink(dst)
    dst.write_text("stale", encoding="utf-8")
    link_or_copy_file(str(src), str(dst))
    assert dst.read_text(encoding="utf-8") == '{"a": 1}\n'


def test_link_or_copy_file_rejects_symlink_ancestor_and_preserves_target(tmp_path):
    """发布目标的祖先或叶子为符号链接时必须拒绝且不改旧目标。"""
    from noise_filter_pipeline.core.manifest_io import link_or_copy_file

    source = tmp_path / "source.jsonl"
    source.write_text("new\n", encoding="utf-8")
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    alias_dir = tmp_path / "alias"
    alias_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        link_or_copy_file(str(source), str(alias_dir / "out.jsonl"))

    target = tmp_path / "target.jsonl"
    target.write_text("old\n", encoding="utf-8")
    target.unlink()
    target.symlink_to(real_dir / "target.jsonl")
    with pytest.raises(ValueError, match="符号链接"):
        link_or_copy_file(str(source), str(target))
    assert target.is_symlink()


def test_build_checkpoint_fingerprint_uses_upstream(tmp_path):
    """checkpoint 指纹 = 本阶段配置 hash + 输入 manifest 的上游配置 hash。

    上游配置变更（provenance.config_hash 变化）必须反映到指纹，否则下游
    checkpoint 会恢复基于旧输入的结果。
    """
    from noise_filter_pipeline.core.manifest_io import build_checkpoint_fingerprint, config_hash

    local_cfg = {"target_rms": 0.1}
    manifest = tmp_path / "input.jsonl"

    # 输入为空：上游指纹为空串，指纹仍可区分（前缀 local::）
    manifest.write_text("", encoding="utf-8")
    fp_empty = build_checkpoint_fingerprint(local_cfg, str(manifest))
    assert fp_empty.startswith(f"{config_hash(local_cfg)}::")

    # 带上游 hash v1
    manifest.write_text(
        '{"record_id": "r1", "dataset_id": "ds", "source_id": "s", '
        '"provenance": {"config_hash": "full_v1"}}\n',
        encoding="utf-8",
    )
    fp_v1 = build_checkpoint_fingerprint(local_cfg, str(manifest))

    # 上游 hash 变化 -> 指纹变化（下游 checkpoint 应失效）
    manifest.write_text(
        '{"record_id": "r1", "dataset_id": "ds", "source_id": "s", '
        '"provenance": {"config_hash": "full_v2"}}\n',
        encoding="utf-8",
    )
    fp_v2 = build_checkpoint_fingerprint(local_cfg, str(manifest))
    assert fp_v2 != fp_v1

    # 本阶段配置变化 -> 指纹同样变化
    fp_cfg2 = build_checkpoint_fingerprint({"target_rms": 0.2}, str(manifest))
    assert fp_cfg2 != fp_v2


def test_build_checkpoint_fingerprint_senses_upstream_recompute(tmp_path):
    """上游 stage 重算（配置未变、产物路径不变）必须使下游指纹失效。

    回归测试：provenance.config_hash 只在 Stage A 写入、下游不刷新，上游
    重跑后下游 checkpoint 原样恢复旧结果（如 D1 参数变更重算后 D2 静默
    复用旧归一化结果）。指纹计入输入 manifest 文件 {name, size, mtime_ns}
    后，上游重跑必然重写输出 manifest（mtime 变化），下游指纹随之失效。
    """
    from noise_filter_pipeline.core.manifest_io import build_checkpoint_fingerprint

    local_cfg = {"target_rms": 0.1}
    manifest = tmp_path / "input.jsonl"
    content = (
        '{"record_id": "r1", "dataset_id": "ds", "source_id": "s", '
        '"provenance": {"config_hash": "full_v1"}}\n'
    )
    manifest.write_text(content, encoding="utf-8")
    fp_before = build_checkpoint_fingerprint(local_cfg, str(manifest))

    # 内容完全不变，仅重写文件并改变 mtime：内容指纹应保持稳定。
    import os
    import time

    manifest.write_text(content, encoding="utf-8")
    future = time.time() + 10
    os.utime(manifest, (future, future))
    fp_after = build_checkpoint_fingerprint(local_cfg, str(manifest))
    assert fp_after == fp_before

    # 内容替换为同长度数据：即使 size/mtime 可伪造，也必须失效。
    replacement = content.replace("full_v1", "full_v2")
    assert len(replacement) == len(content)
    manifest.write_text(replacement, encoding="utf-8")
    os.utime(manifest, (future, future))
    assert build_checkpoint_fingerprint(local_cfg, str(manifest)) != fp_before


def test_fingerprint_stable_when_eligible_rewritten_same_content(tmp_path):
    """子阶段 eligible 中间文件同内容重写（崩溃重跑）不得使 checkpoint 指纹失效。

    回归：子阶段 eligible 文件每次崩溃重跑都可能原子重写；同内容重写
    不应让断点失效。stat_input 指定稳定上游产物后，只有上游内容变化才失效。
    """
    from noise_filter_pipeline.core.manifest_io import build_checkpoint_fingerprint

    import os
    import time

    local_cfg = {"a": 1}
    upstream = tmp_path / "upstream.jsonl"
    upstream.write_text(
        '{"record_id": "r1", "provenance": {"config_hash": "h1"}}\n', encoding="utf-8"
    )
    eligible = tmp_path / "c1_input_eligible.jsonl"
    eligible.write_text(
        '{"record_id": "r1", "provenance": {"config_hash": "h1"}}\n', encoding="utf-8"
    )
    fp1 = build_checkpoint_fingerprint(local_cfg, str(eligible), stat_input=str(upstream))

    # 同内容重写 eligible（模拟崩溃后重跑重新生成）→ 指纹不变
    eligible.write_text(
        '{"record_id": "r1", "provenance": {"config_hash": "h1"}}\n', encoding="utf-8"
    )
    fp2 = build_checkpoint_fingerprint(local_cfg, str(eligible), stat_input=str(upstream))
    assert fp1 == fp2

    # 上游仅 mtime 变化但内容不变：内容指纹不应失效。
    future = time.time() + 10
    os.utime(upstream, (future, future))
    fp3 = build_checkpoint_fingerprint(local_cfg, str(eligible), stat_input=str(upstream))
    assert fp3 == fp1

    # 上游内容同长度替换：必须失效。
    upstream.write_text(
        '{"record_id": "r1", "provenance": {"config_hash": "h2"}}\n',
        encoding="utf-8",
    )
    assert build_checkpoint_fingerprint(local_cfg, str(eligible), stat_input=str(upstream)) != fp1


@pytest.mark.unit
def test_checkpoint_fingerprint_rejects_parent_replacement(tmp_path, monkeypatch):
    """扫描持有 parent fd 时，路径可见 parent 被替换必须 fail closed。"""
    from noise_filter_pipeline.core import manifest_io

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "b_manifest_ds.jsonl").write_text("{}\n", encoding="utf-8")
    displaced = tmp_path / "displaced"
    real_listdir = manifest_io.os.listdir
    replaced = False

    def replace_after_enumeration(fd):
        nonlocal replaced
        names = real_listdir(fd)
        if not replaced:
            replaced = True
            manifest_dir.rename(displaced)
            manifest_dir.mkdir()
            (manifest_dir / "b_manifest_ds.jsonl").write_text("{}\n", encoding="utf-8")
        return names

    monkeypatch.setattr(
        manifest_io, "os", patched_os(listdir=replace_after_enumeration)
    )
    with pytest.raises(OSError, match="parent 身份变化"):
        manifest_io.build_checkpoint_fingerprint({"a": 1}, str(manifest_dir))


@pytest.mark.unit
def test_checkpoint_fingerprint_hashes_hardlink_alias_once(tmp_path, monkeypatch):
    """input/stat_input 指向同一物理文件时不得重复读取并哈希。"""
    from noise_filter_pipeline.core import manifest_io

    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        '{"provenance":{"config_hash":"h1"}}\n', encoding="utf-8"
    )
    alias = tmp_path / "alias.jsonl"
    os.link(manifest, alias)
    real_sha256 = manifest_io.hashlib.sha256
    streaming_hashes = 0

    def observed_sha256(*args, **kwargs):
        nonlocal streaming_hashes
        if not args and not kwargs:
            streaming_hashes += 1
        return real_sha256(*args, **kwargs)

    monkeypatch.setattr(manifest_io.hashlib, "sha256", observed_sha256)
    manifest_io.build_checkpoint_fingerprint(
        {"a": 1}, str(manifest), stat_input=str(alias)
    )
    assert streaming_hashes == 1


def _legacy_checkpoint_fingerprint(local_config, input_manifest):
    """独立参考实现：配置 hash、逐行 provenance hash 与文件 SHA-256。"""
    import hashlib
    from noise_filter_pipeline.core.manifest_io import _iter_manifest_files, config_hash, read_manifest

    upstream_hashes = set()
    for rec in read_manifest(input_manifest):
        upstream_hashes.add((rec.provenance or {}).get("config_hash") or "")
    fingerprints = []
    for fp in _iter_manifest_files(input_manifest):
        digest = hashlib.sha256(fp.read_bytes()).hexdigest()
        fingerprints.append({"name": fp.name, "sha256": digest})
    upstream_hashes.add("content:" + hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest())
    return f"{config_hash(local_config)}::{config_hash({'upstream_hashes': sorted(upstream_hashes)})}"


def test_build_checkpoint_fingerprint_matches_legacy_semantics(tmp_path):
    """P2：行级正则抽取实现必须与旧 AudioRecord 全量构造实现结果等价
    （单 hash / 多 hash / 缺 provenance / null 等场景）。"""
    from noise_filter_pipeline.core.manifest_io import build_checkpoint_fingerprint

    local_cfg = {"target_rms": 0.1}
    manifest = tmp_path / "input.jsonl"
    cases = [
        # 空 manifest
        "",
        # 单 hash
        '{"record_id":"r1","dataset_id":"ds","source_id":"s1","provenance":{"config_hash":"h1"}}\n',
        # 多 hash（含重复值，集合去重后仍为多 hash）
        '{"record_id":"r1","dataset_id":"ds","source_id":"s1","provenance":{"config_hash":"h1"}}\n'
        '{"record_id":"r2","dataset_id":"ds","source_id":"s2","provenance":{"config_hash":"h2"}}\n'
        '{"record_id":"r3","dataset_id":"ds","source_id":"s3","provenance":{"config_hash":"h1"}}\n',
        # 无 provenance -> 空串
        '{"record_id":"r1","dataset_id":"ds","source_id":"s1"}\n',
        # 混合：有 hash + 无 provenance -> 两元素集合哈希
        '{"record_id":"r1","dataset_id":"ds","source_id":"s1","provenance":{"config_hash":"h1"}}\n'
        '{"record_id":"r2","dataset_id":"ds","source_id":"s2"}\n',
        # provenance 存在但无 config_hash 键 -> 空串
        '{"record_id":"r1","dataset_id":"ds","source_id":"s1","provenance":{"pipeline_version":"2.1.0"}}\n',
        # config_hash 为 null -> 空串
        '{"record_id":"r1","dataset_id":"ds","source_id":"s1","provenance":{"config_hash":null}}\n',
    ]
    for i, content in enumerate(cases):
        manifest.write_text(content, encoding="utf-8")
        new_fp = build_checkpoint_fingerprint(local_cfg, str(manifest))
        legacy_fp = _legacy_checkpoint_fingerprint(local_cfg, str(manifest))
        assert new_fp == legacy_fp, f"case {i} 指纹不等价: {new_fp} != {legacy_fp}"

    # per-dataset 目录输入（多文件、多 hash）同样等价
    d = tmp_path / "dir"
    d.mkdir()
    (d / "b_manifest_ds1.jsonl").write_text(
        '{"record_id":"r1","dataset_id":"ds1","source_id":"s1","provenance":{"config_hash":"hA"}}\n',
        encoding="utf-8",
    )
    (d / "b_manifest_ds2.jsonl").write_text(
        '{"record_id":"r2","dataset_id":"ds2","source_id":"s2","provenance":{"config_hash":"hB"}}\n',
        encoding="utf-8",
    )
    assert build_checkpoint_fingerprint(local_cfg, str(d)) == _legacy_checkpoint_fingerprint(
        local_cfg, str(d)
    )


def test_build_checkpoint_fingerprint_avoids_record_construction(tmp_path, monkeypatch):
    """P2 性能回归：指纹构建必须走行级正则抽取，不得全量构造 AudioRecord
    （百万级记录 × 大 JSON 行时全量构造是分钟级开销）。"""
    from noise_filter_pipeline.core import manifest_io

    def _boom(*_args, **_kwargs):
        raise AssertionError("指纹构建不得构造 AudioRecord")

    monkeypatch.setattr(manifest_io.AudioRecord, "from_dict", staticmethod(_boom))

    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        '{"record_id":"r1","dataset_id":"ds","source_id":"s1","provenance":{"config_hash":"h1"}}\n'
        '{"record_id":"r2","dataset_id":"ds","source_id":"s2","provenance":{"config_hash":"h1"}}\n',
        encoding="utf-8",
    )
    fp = manifest_io.build_checkpoint_fingerprint({"a": 1}, str(manifest))
    # upstream 集合（config_hash + 文件统计）经 config_hash 聚合，不再保留
    # 原始 "::h1" 字面量；此处只校验前缀（本阶段配置部分）与整体非空
    assert fp.startswith(f"{manifest_io.config_hash({'a': 1})}::")
    assert len(fp) > len(f"{manifest_io.config_hash({'a': 1})}::")


def _rec(record_id: str, dataset_id: str) -> AudioRecord:
    return AudioRecord(
        record_id=record_id,
        dataset_id=dataset_id,
        source_id=record_id,
        audio={"path": f"/tmp/{record_id}.wav", "sr": 16000, "duration": 1.0},
    )


def test_per_dataset_writer_routes_and_truncates(tmp_path):
    """PerDatasetManifestWriter 按 dataset 路由、首次写截断覆盖；默认不删遗留单文件。"""
    base = str(tmp_path / "b_manifest.jsonl")
    # 预置一个遗留单文件（模拟旧代码产物），验证增量写不会误删它
    write_manifest(iter([_rec("old__a", "old")]), base)

    with PerDatasetManifestWriter(base, batch_size=1) as w:
        w.append(_rec("ds1__a", "ds1"))
        w.append(_rec("ds2__a", "ds2"))
        w.append(_rec("ds1__b", "ds1"))

    # 两个数据集各一个文件
    f1 = tmp_path / "b_manifest_ds1.jsonl"
    f2 = tmp_path / "b_manifest_ds2.jsonl"
    assert f1.exists() and f2.exists()
    # 默认不删除遗留单文件：旧数据记录必须保留（增量跑未迁移时不能误删）
    assert (tmp_path / "b_manifest.jsonl").exists(), "增量写不得删除遗留单文件"

    # 同数据集重写（覆盖而非追加）：ds1 从 2 条变 1 条
    with PerDatasetManifestWriter(base, batch_size=1) as w:
        w.append(_rec("ds1__c", "ds1"))
    recs1 = list(read_manifest(str(f1)))
    assert [r.record_id for r in recs1] == ["ds1__c"]
    # ds2 文件不受影响
    assert len(list(read_manifest(str(f2)))) == 1


def test_per_dataset_writer_concurrent_flush_no_torn_lines(tmp_path, monkeypatch):
    """两个线程对同一 dataset 高频 append 大记录（>16KB 字段）+ batch_size=1 强制
    每次 append 都触发 flush，读回逐行 json.loads 校验无坏行且条数守恒。

    回归测试：flush 曾在 _lock 内 swap、锁外 append_manifest，主线程与后台 flush
    线程并发写同一 dataset 文件时，>8KB 记录被拆成多个 write syscall，线程间交错
    撕裂 JSONL 行。本机文件系统上单次 write() 通常不交错，故用「两截写入 +
    rendezvous」确定性模拟 syscall 交错：无串行锁时两线程在 barrier 对齐、
    必然撕裂；修复后写盘段被 _write_lock 串行化，barrier 超时退化为顺序写。

    并发测试固有局限（不得删除测试）：barrier 对齐依赖线程调度，极端环境
    下可能假阳（修复前也偶发通过/失败）；而 rendezvous 只模拟了"两截写入"
    一种交错形态，其他撕裂窗口无法全部覆盖（假阴）。作为回归锚有效，
    非确定性证明。
    """
    import threading
    from noise_filter_pipeline.core import manifest_per_dataset as mpd

    base = str(tmp_path / "b_manifest.jsonl")
    big_label = "x" * 16384  # 单条记录 >16KB
    num_threads = 2
    recs_per_thread = 20

    # 模拟大记录被拆成多个 write syscall：每条记录分两截落盘，中间 rendezvous；
    # 两线程并发进入写盘段时在 barrier 对齐，两批半截必然交错
    rendezvous = threading.Barrier(num_threads)
    start_gate = threading.Barrier(num_threads + 1)

    def chunked_append(records, path):
        lines = [json.dumps(rec.to_dict(), ensure_ascii=False) + "\n" for rec in records]
        halves = [(line[: len(line) // 2], line[len(line) // 2:]) for line in lines]
        with open(path, "a", encoding="utf-8") as f:
            for first, _ in halves:
                f.write(first)
            try:
                rendezvous.wait(timeout=2)
            except threading.BrokenBarrierError:
                # 写盘段已被串行化（或 barrier 已破）：无并发者，直接继续
                pass
            for _, second in halves:
                f.write(second)
        return len(lines)

    monkeypatch.setattr(mpd, "append_manifest", chunked_append)

    with PerDatasetManifestWriter(base, batch_size=1) as writer:
        def worker(tid: int) -> None:
            start_gate.wait(timeout=5)
            for i in range(recs_per_thread):
                rec = AudioRecord(
                    record_id=f"t{tid}__r{i}",
                    dataset_id="ds",
                    source_id=f"s{tid}",
                    audio={"path": f"/tmp/t{tid}_{i}.wav", "sr": 16000, "duration": 1.0},
                    labels={"original": big_label},
                )
                writer.append(rec)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        start_gate.wait(timeout=5)  # 同时放行，最大化并发进入写盘段的机会
        for t in threads:
            t.join()

    path = tmp_path / "b_manifest_ds.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    expected = num_threads * recs_per_thread
    assert len(lines) == expected, "并发 flush 后行数必须守恒"
    ids = set()
    for line in lines:
        data = json.loads(line)  # 撕裂行会直接在这里抛 JSONDecodeError
        ids.add(data["record_id"])
    assert ids == {f"t{t}__r{i}" for t in range(num_threads) for i in range(recs_per_thread)}


def test_per_dataset_writer_remove_legacy_base(tmp_path):
    """remove_legacy_base=True（本 stage merged→per-dataset 拆分）时 close 删除 merged 单文件。"""
    base = str(tmp_path / "b_manifest.jsonl")
    write_manifest(iter([_rec("old__a", "old")]), base)

    with PerDatasetManifestWriter(base, batch_size=1, remove_legacy_base=True) as w:
        w.append(_rec("ds1__a", "ds1"))
    assert not (tmp_path / "b_manifest.jsonl").exists(), "显式拆分后应删除 merged 单文件"
    assert (tmp_path / "b_manifest_ds1.jsonl").exists()


def test_reject_legacy_single_manifest_fails_fast_with_migration_guidance(tmp_path):
    """遗留单文件 manifest 必须显式拒绝，报错给出手工迁移指引且不改写源文件。"""
    legacy = tmp_path / "b_manifest.jsonl"
    write_manifest(iter([_rec("ds1__a", "ds1")]), str(legacy))

    assert find_legacy_single_manifests(str(tmp_path)) == [str(legacy)]
    with pytest.raises(RuntimeError, match="遗留单文件") as raised:
        reject_legacy_single_manifest(str(tmp_path), stage_label="Stage B 输出目录")
    assert "split_manifest_by_dataset" in str(raised.value)
    assert legacy.exists(), "拒绝路径不得改写或删除遗留单文件"

    # 人工处理后放行；无遗留单文件时不产生副作用
    legacy.unlink()
    reject_legacy_single_manifest(str(tmp_path), stage_label="Stage B 输出目录")
    assert find_legacy_single_manifests(str(tmp_path)) == []


def test_reject_legacy_single_manifest_allows_per_dataset_layout(tmp_path):
    """per-dataset 文件与不存在目录都不触发拒绝。"""
    write_manifest(
        iter([_rec("ds1__a", "ds1")]), str(tmp_path / "b_manifest_ds1.jsonl"),
    )

    reject_legacy_single_manifest(str(tmp_path), stage_label="Stage B 输出目录")
    reject_legacy_single_manifest(
        str(tmp_path / "missing"), stage_label="Stage B 输出目录",
    )
    assert find_legacy_single_manifests(str(tmp_path)) == []
    assert find_legacy_single_manifests(str(tmp_path / "missing")) == []


def test_split_manifest_strictly_rejects_bad_json(tmp_path):
    """普通 split 同样不得沿用 read_manifest 的坏行静默跳过行为。"""
    src = tmp_path / "c3_merged.jsonl"
    src.write_text("{broken json}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        split_manifest_by_dataset(str(src), str(tmp_path / "c_manifest.jsonl"))


def test_split_manifest_by_dataset_removes_merged_source(tmp_path):
    """split_manifest_by_dataset(remove_legacy_base=True)：拆分本 stage merged 输出后删除它。"""
    base = str(tmp_path / "b4_manifest.jsonl")
    write_manifest(iter([
        _rec("ds1__a", "ds1"),
        _rec("ds2__a", "ds2"),
        _rec("ds1__b", "ds1"),
    ]), base)

    n = split_manifest_by_dataset(base, base, remove_legacy_base=True)
    assert n == 3
    assert not os.path.exists(base), "拆分后 merged 单文件应删除"
    assert (tmp_path / "b4_manifest_ds1.jsonl").exists()
    assert (tmp_path / "b4_manifest_ds2.jsonl").exists()

    # 幂等：无 merged 单文件时返回 0，不产生副作用
    assert split_manifest_by_dataset(base, base, remove_legacy_base=True) == 0


def test_split_manifest_by_dataset(tmp_path):
    """split_manifest_by_dataset：单文件按 dataset 拆到指定 base 的 per-dataset 文件。"""
    src = str(tmp_path / "c3_merged.jsonl")
    write_manifest(iter([
        _rec("ds1__a", "ds1"),
        _rec("ds2__a", "ds2"),
    ]), src)

    dst_base = str(tmp_path / "c_manifest.jsonl")
    n = split_manifest_by_dataset(src, dst_base)
    assert n == 2
    assert (tmp_path / "c_manifest_ds1.jsonl").exists()
    assert (tmp_path / "c_manifest_ds2.jsonl").exists()
    # 源文件保留（merged 是中间产物，不删除）
    assert os.path.exists(src)


def test_read_manifest_directory_merge(tmp_path):
    """read_manifest 目录模式：合并所有 *_manifest_*.jsonl，忽略非 manifest 文件。"""
    (tmp_path / "x_manifest_ds1.jsonl").write_text(
        json.dumps(_rec("ds1__a", "ds1").to_dict()) + "\n", encoding="utf-8"
    )
    (tmp_path / "x_manifest_ds2.jsonl").write_text(
        json.dumps(_rec("ds2__a", "ds2").to_dict()) + "\n", encoding="utf-8"
    )
    (tmp_path / "unrelated.jsonl").write_text(
        json.dumps(_rec("x__a", "x").to_dict()) + "\n", encoding="utf-8"
    )
    (tmp_path / "checkpoint.json").write_text('{"done": []}\n', encoding="utf-8")

    recs = list(read_manifest(str(tmp_path)))
    assert {r.record_id for r in recs} == {"ds1__a", "ds2__a"}
    # 空目录：返回空
    empty = tmp_path / "empty"
    empty.mkdir()
    assert list(read_manifest(str(empty))) == []
