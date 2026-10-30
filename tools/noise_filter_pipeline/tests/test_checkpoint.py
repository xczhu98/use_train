"""
Tests for CheckpointTracker.
"""
import json
import os
import tempfile
import threading

import pytest

from noise_filter_pipeline.core.checkpoint import CheckpointTracker

pytestmark = pytest.mark.pipeline


def test_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.json")
        ckpt = CheckpointTracker(checkpoint_path=path, enabled=True)

        assert not ckpt.is_done("r1")
        ckpt.mark_done("r1")
        ckpt.mark_done("r2")
        ckpt.save()

        ckpt2 = CheckpointTracker(checkpoint_path=path, enabled=True)
        assert ckpt2.is_done("r1")
        assert ckpt2.is_done("r2")
        assert not ckpt2.is_done("r3")


def test_checkpoint_concurrent_saves_preserve_latest_snapshot(monkeypatch, tmp_path):
    """并发 save 必须串行，最终 checkpoint 不得被较旧快照覆盖。

    并发测试固有局限（不得删除测试，但需知晓）：竞争类缺陷只在特定交错
    时机暴露，调度器未制造出该窗口时可能假阴（有 bug 也通过）；反之在
    高负载/单核环境下时序偏移也可能假阳。本测试通过 callback 栅栏人为
    放大交错窗口来降低假阴率，但仍非确定性证明。
    """
    path = str(tmp_path / "checkpoint.json")
    ckpt = CheckpointTracker(checkpoint_path=path, enabled=True)
    ckpt.mark_done("first")
    first_callback_started = threading.Event()
    release_first_callback = threading.Event()
    callback_calls = 0
    callback_lock = threading.Lock()

    def callback():
        nonlocal callback_calls
        with callback_lock:
            callback_calls += 1
            call = callback_calls
        if call == 1:
            first_callback_started.set()
            release_first_callback.wait(timeout=2)

    ckpt._pre_save_callbacks = [callback]
    first = threading.Thread(target=ckpt.save, kwargs={"force": True})
    first.start()
    assert first_callback_started.wait(timeout=2)
    ckpt.mark_done("second")
    second = threading.Thread(target=ckpt.save, kwargs={"force": True})
    second.start()
    release_first_callback.set()
    first.join(timeout=2)
    second.join(timeout=2)

    with open(path, "r", encoding="utf-8") as f:
        assert set(json.load(f)["done"]) == {"first", "second"}


def test_checkpoint_fingerprint_uses_all_manifest_records(tmp_path):
    """第二条记录的上游指纹变化也必须失效 checkpoint。"""
    from noise_filter_pipeline.core.manifest_io import build_checkpoint_fingerprint

    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        '{"record_id":"r1","dataset_id":"ds","source_id":"s1",'
        '"provenance":{"config_hash":"upstream-a"}}\n'
        '{"record_id":"r2","dataset_id":"ds","source_id":"s2",'
        '"provenance":{"config_hash":"upstream-b"}}\n',
        encoding="utf-8",
    )
    fingerprint = build_checkpoint_fingerprint({"setting": 1}, str(manifest))
    manifest.write_text(
        '{"record_id":"r1","dataset_id":"ds","source_id":"s1",'
        '"provenance":{"config_hash":"upstream-a"}}\n'
        '{"record_id":"r2","dataset_id":"ds","source_id":"s2",'
        '"provenance":{"config_hash":"upstream-c"}}\n',
        encoding="utf-8",
    )
    assert build_checkpoint_fingerprint({"setting": 1}, str(manifest)) != fingerprint


def test_checkpoint_disabled():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.json")
        ckpt = CheckpointTracker(checkpoint_path=path, enabled=False)
        ckpt.mark_done("r1")
        ckpt.save(force=True)
        assert not os.path.isfile(path)


def test_checkpoint_persists_valid_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.json")
        ckpt = CheckpointTracker(checkpoint_path=path, enabled=True)
        ckpt.mark_done_batch(["a", "b", "c"])
        ckpt.save(force=True)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert set(data["done"]) == {"a", "b", "c"}
        assert data["count"] == 3


def test_checkpoint_batch_save_interval():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.json")
        ckpt = CheckpointTracker(checkpoint_path=path, enabled=True, save_interval=5)

        # 未达阈值时不应落盘
        for i in range(4):
            ckpt.mark_done(f"r{i}")
        assert not os.path.isfile(path)

        # 第 5 条触发自动保存
        ckpt.mark_done("r4")
        assert os.path.isfile(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["count"] == 5

        # 再 mark 3 条，未达阈值；force save 后全部落盘
        for i in range(5, 8):
            ckpt.mark_done(f"r{i}")
        ckpt.save(force=True)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["count"] == 8


def test_checkpoint_batch_save_interval_on_batch_mark():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.json")
        ckpt = CheckpointTracker(checkpoint_path=path, enabled=True, save_interval=3)
        ckpt.mark_done_batch(["a", "b", "c", "d"])
        assert os.path.isfile(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert set(data["done"]) == {"a", "b", "c", "d"}


def test_checkpoint_config_hash_roundtrip(caplog):
    """save 写入组合 config_hash；相同新格式指纹正常 resume 且不告警。"""
    import logging

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.json")
        config_hash = "abc123::upstream_hash"
        ckpt = CheckpointTracker(
            checkpoint_path=path,
            enabled=True,
            config_hash=config_hash,
        )
        ckpt.mark_done("r1")
        ckpt.save(force=True)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["config_hash"] == config_hash

        # 相同新格式指纹 resume：不出现配置变更 warning
        with caplog.at_level(logging.WARNING, logger="noise_filter_pipeline.core.checkpoint"):
            ckpt2 = CheckpointTracker(
                checkpoint_path=path,
                enabled=True,
                config_hash=config_hash,
            )
        assert not caplog.records
        assert ckpt2.is_done("r1")


def test_checkpoint_legacy_hash_rejects_composite_current_fingerprint(caplog):
    """当前指纹为组合格式时，旧 local hash 无法验证上游，必须拒绝恢复。"""
    import logging

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.json")
        # 旧版本写下的纯 local hash（无 :: 分隔）
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"done": ["r1"], "count": 1, "config_hash": "abc123"}, f)

        # 即使 local 部分相同，也无法验证 upstream，必须全量重算
        with caplog.at_level(logging.WARNING, logger="noise_filter_pipeline.core.checkpoint"):
            ckpt = CheckpointTracker(
                checkpoint_path=path,
                enabled=True,
                config_hash="abc123::upstream_hash",
            )
        assert not ckpt.is_done("r1")
        assert any("忽略旧断点" in rec.message for rec in caplog.records)


def test_checkpoint_legacy_hash_mismatch_rejects(caplog):
    """旧格式 checkpoint 但本阶段配置已变更（local 不一致）时拒绝恢复。"""
    import logging

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"done": ["r1"], "count": 1, "config_hash": "old_local"}, f)

        with caplog.at_level(logging.WARNING, logger="noise_filter_pipeline.core.checkpoint"):
            ckpt = CheckpointTracker(checkpoint_path=path, enabled=True, config_hash="new_local::upstream")
        assert not ckpt.is_done("r1")


def test_checkpoint_config_hash_mismatch_rejects(caplog):
    """换配置 resume 时旧 checkpoint 必须被忽略（拒绝恢复），触发 warning 并全量重算。

    否则改配置重跑会静默恢复旧配置的结果，新配置不生效（正确性风险）。
    """
    import logging

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.json")
        ckpt = CheckpointTracker(checkpoint_path=path, enabled=True, config_hash="old_hash")
        ckpt.mark_done("r1")
        ckpt.save(force=True)

        with caplog.at_level(logging.WARNING, logger="noise_filter_pipeline.core.checkpoint"):
            ckpt2 = CheckpointTracker(checkpoint_path=path, enabled=True, config_hash="new_hash")
        assert any("忽略旧断点" in rec.message for rec in caplog.records)
        # 拒绝恢复：已处理记录不再生效，后续将全量重算
        assert not ckpt2.is_done("r1")


def test_checkpoint_config_hash_missing_rejects(caplog):
    """旧 checkpoint 无 config_hash 字段时无法验证配置一致性，同样拒绝恢复。"""
    import logging

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"done": ["r1"], "count": 1}, f)

        with caplog.at_level(logging.WARNING, logger="noise_filter_pipeline.core.checkpoint"):
            ckpt = CheckpointTracker(checkpoint_path=path, enabled=True, config_hash="new_hash")
        assert any("缺少 config_hash" in rec.message for rec in caplog.records)
        assert not ckpt.is_done("r1")


def test_checkpoint_pre_save_callback_failure_aborts_save():
    """pre-save callback（如 flush manifest）失败时必须中止落盘并保持 dirty，
    否则 checkpoint 会领先于 manifest，resume 时静默丢数据。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.json")
        should_fail = {"flag": True}

        def flaky_callback():
            if should_fail["flag"]:
                raise RuntimeError("manifest flush failed")

        ckpt = CheckpointTracker(
            checkpoint_path=path, enabled=True, pre_save_callbacks=[flaky_callback]
        )
        ckpt.mark_done("r1")
        ckpt.save(force=True)
        # callback 失败：checkpoint 不得落盘
        assert not os.path.isfile(path)

        # callback 恢复后再次 save：dirty 标记仍在，数据能落盘
        should_fail["flag"] = False
        ckpt.save(force=True)
        assert os.path.isfile(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert set(data["done"]) == {"r1"}


def test_checkpoint_pre_save_failure_restores_marks():
    """callback 失败中止时 marks 计数一并恢复（视同 save 未发生），
    后续 mark 累计到 save_interval 仍能自动触发落盘。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.json")
        should_fail = {"flag": True}

        def flaky_callback():
            if should_fail["flag"]:
                raise RuntimeError("boom")

        ckpt = CheckpointTracker(
            checkpoint_path=path, enabled=True, save_interval=2,
            pre_save_callbacks=[flaky_callback],
        )
        ckpt.mark_done("r1")
        ckpt.mark_done("r2")  # 达到 interval 触发 save，callback 失败中止
        assert not os.path.isfile(path)

        # marks 已恢复为 2；callback 恢复后，再 mark 一条即超过 interval 自动落盘
        should_fail["flag"] = False
        ckpt.mark_done("r3")
        assert os.path.isfile(path)
        with open(path, "r", encoding="utf-8") as f:
            assert json.load(f)["count"] == 3
