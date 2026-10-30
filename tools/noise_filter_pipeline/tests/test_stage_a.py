"""
Tests for Stage A ingestion loaders (directory / JSON / CSV / HuggingFace).
"""
import csv
import json
import logging
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from noise_filter_pipeline.core.manifest_io import read_manifest
from noise_filter_pipeline.core.pipeline_config import UnifiedPipelineConfig
from noise_filter_pipeline.core.registry import DatasetRegistry
from noise_filter_pipeline.stages.stage_a_ingestion import (
    StageAOrchestrator,
    _audio_info_cache_path,
    _file_fingerprint,
    _get_audio_info,
    _load_audio_info_cache,
    _prune_audio_info_cache,
    _save_audio_info_cache,
)

pytestmark = pytest.mark.pipeline


def _make_wav(path, duration=1.0, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav = np.random.randn(int(duration * sr)) * 0.001
    sf.write(path, wav, sr)


def _make_hf_audio_dataset(wav_paths, labels=None):
    """用 Arrow 原始 path/bytes 构造 Audio feature，避免依赖 torchcodec 编码。"""
    import pyarrow as pa
    from datasets import Audio, Dataset, DatasetInfo, Features, Value

    if isinstance(wav_paths, (str, os.PathLike)):
        wav_paths = [wav_paths]
    wav_paths = [os.fspath(path) for path in wav_paths]
    payloads = []
    for path in wav_paths:
        with open(path, "rb") as f:
            payloads.append(f.read())
    sr = sf.info(wav_paths[0]).samplerate
    audio_struct = pa.array(
        [{"bytes": payload, "path": None} for payload in payloads],
        type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
    )
    if labels is None:
        labels = ["noise"] * len(wav_paths)
    label_type = pa.string() if all(isinstance(x, str) for x in labels) else pa.int64()
    table = pa.table(
        {"audio": audio_struct, "labels": pa.array(labels, type=label_type)}
    )
    info = DatasetInfo(
        features=Features(
            {
                "audio": Audio(sampling_rate=sr),
                "labels": Value("string" if label_type == pa.string() else "int64"),
            }
        )
    )
    return Dataset(table, info=info)


class _FakePbar:
    """tqdm 最小替身：disable=True 的 tqdm 不会累计 n，测试需要精确计数。"""

    def __init__(self):
        self.total = 0
        self.n = 0

    def update(self, n=1):
        self.n += n

    def refresh(self):
        pass

    def close(self):
        pass


def test_stage_a_directory_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "ds")
        _make_wav(os.path.join(dataset_dir, "a.wav"))
        _make_wav(os.path.join(dataset_dir, "b.wav"))

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_dir
    name: Test Directory
    enabled: true
    type: directory
    path: {dataset_dir}
    input_pattern: "**/*.wav"
    label_source: filename
    splits: [train]
    split_strategy: none
""")

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            registry_path=registry_path,
        )
        orch = StageAOrchestrator(config)
        manifest = orch.run(
            input_manifest=None,
            output_dir=os.path.join(tmpdir, "out", "stage_a"),
            output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
        )
        recs = list(read_manifest(manifest))
        assert len(recs) == 2
        assert all(r.dataset_id == "test_dir" for r in recs)
        assert all(r.is_stage_passed("A") for r in recs)


def test_stage_a_json_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_a = os.path.join(tmpdir, "a.wav")
        wav_b = os.path.join(tmpdir, "b.wav")
        _make_wav(wav_a)
        _make_wav(wav_b)

        json_path = os.path.join(tmpdir, "data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([
                {"audio_path": wav_a, "text_label": "noise"},
                {"audio_path": wav_b, "text_label": ["speech", "music"]},
            ], f, ensure_ascii=False)

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_json
    name: Test JSON
    enabled: true
    type: json
    path: {json_path}
    audio_field: audio_path
    label_field: text_label
    splits: [train]
""")

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            registry_path=registry_path,
        )
        orch = StageAOrchestrator(config)
        manifest = orch.run(
            input_manifest=None,
            output_dir=os.path.join(tmpdir, "out", "stage_a"),
            output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
        )
        recs = list(read_manifest(manifest))
        assert len(recs) == 2
        assert recs[0].labels["original"] == "noise"
        assert recs[1].labels["original"] == ["speech", "music"]


def test_stage_a_json_relative_paths_have_stable_unique_source_ids():
    """metadata 相对路径应相对 JSON 所在目录解析，并保留子目录消除同名冲突。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_wav(os.path.join(tmpdir, "left", "same.wav"))
        _make_wav(os.path.join(tmpdir, "right", "same.wav"))
        json_path = os.path.join(tmpdir, "data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([
                {"audio_path": "left/same.wav", "text_label": "one"},
                {"audio_path": "right/same.wav", "text_label": "two"},
            ], f)
        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_json
    type: json
    path: {json_path}
    audio_field: audio_path
    label_field: text_label
""")

        config = UnifiedPipelineConfig(output_root=os.path.join(tmpdir, "out"), registry_path=registry_path)
        manifest = StageAOrchestrator(config).run(
            input_manifest=None,
            output_dir=os.path.join(tmpdir, "out", "stage_a"),
            output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
        )
        recs = list(read_manifest(manifest))
        assert {r.source_id for r in recs} == {"left/same", "right/same"}
        assert len({r.record_id for r in recs}) == 2
        assert all(os.path.isabs(r.audio["path"]) for r in recs)


def test_stage_a_skips_duplicate_metadata_record_ids(caplog):
    """重复 record_id 不再中止整个 Stage A：告警并跳过重复条目（保留首条）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_wav(os.path.join(tmpdir, "same.wav"))
        json_path = os.path.join(tmpdir, "data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([
                {"audio_path": "same.wav", "text_label": "one"},
                {"audio_path": "same.wav", "text_label": "two"},
            ], f)
        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_json
    type: json
    path: {json_path}
    audio_field: audio_path
""")
        config = UnifiedPipelineConfig(output_root=os.path.join(tmpdir, "out"), registry_path=registry_path)
        out_dir = os.path.join(tmpdir, "out", "stage_a")
        out_manifest = os.path.join(out_dir, "a_manifest.jsonl")
        StageAOrchestrator(config).run(
            None, out_dir, out_manifest,
        )
        # 重复条目被跳过：只产出 1 条记录，且留有告警审计日志
        # （Stage A 输出为 per-dataset 文件，读目录聚合）
        recs = list(read_manifest(out_dir))
        assert len(recs) == 1
        assert recs[0].labels.get("original") == "one"
        assert "重复" in caplog.text


def test_stage_a_rejects_unknown_dataset_filter():
    with tempfile.TemporaryDirectory() as tmpdir:
        # registry 必须已注册 enabled 数据集（空目录即可）：否则 ValueError 会
        # 提前命中 run() 的"没有 enabled 数据集"分支，dataset_ids 拼写校验
        # （"不存在或未启用的 dataset"）从未执行，测试验证错对象。
        dataset_dir = os.path.join(tmpdir, "ds")
        os.makedirs(dataset_dir)
        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: real_ds
    name: Real Dataset
    enabled: true
    type: directory
    path: {dataset_dir}
    input_pattern: "**/*.wav"
    label_source: filename
    splits: [train]
""")
        config = UnifiedPipelineConfig(output_root=os.path.join(tmpdir, "out"), registry_path=registry_path)
        orch = StageAOrchestrator(config, dataset_ids={"typo"})
        with pytest.raises(ValueError, match="不存在或未启用的 dataset") as excinfo:
            orch.run(None, os.path.join(tmpdir, "out", "stage_a"), os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"))
        # 错误信息必须列出实际范围：拼错的 id 在列、registry 中的合法 id 不在列
        assert "typo" in str(excinfo.value)
        assert "real_ds" not in str(excinfo.value)


def test_stage_a_csv_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_a = os.path.join(tmpdir, "a.wav")
        wav_b = os.path.join(tmpdir, "b.wav")
        _make_wav(wav_a)
        _make_wav(wav_b)

        csv_path = os.path.join(tmpdir, "data.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "tag"])
            writer.writerow([wav_a, "noise"])
            writer.writerow([wav_b, "speech,music"])

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_csv
    name: Test CSV
    enabled: true
    type: csv
    path: {csv_path}
    audio_column: path
    label_column: tag
    label_separator: ","
    splits: [train]
""")

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            registry_path=registry_path,
        )
        orch = StageAOrchestrator(config)
        manifest = orch.run(
            input_manifest=None,
            output_dir=os.path.join(tmpdir, "out", "stage_a"),
            output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
        )
        recs = list(read_manifest(manifest))
        assert len(recs) == 2
        assert recs[0].labels["original"] == "noise"
        assert recs[1].labels["original"] == ["speech", "music"]


def test_stage_a_huggingface_loader():
    try:
        import datasets  # noqa: F401
    except ImportError:
        pytest.skip("datasets 库未安装，跳过 HF loader 测试")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_a = os.path.join(tmpdir, "a.wav")
        _make_wav(wav_a, duration=1.0)

        # 构造本地 HF dataset（Audio 列存原始 bytes，不依赖 torchcodec 编码）
        ds = _make_hf_audio_dataset(wav_a)
        hf_path = os.path.join(tmpdir, "hf_ds")
        ds.save_to_disk(hf_path)

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_hf
    name: Test HF
    enabled: true
    type: huggingface
    path: {hf_path}
    audio_field: audio
    label_field: labels
    splits: [train]
""")

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            registry_path=registry_path,
        )
        orch = StageAOrchestrator(config)
        manifest = orch.run(
            input_manifest=None,
            output_dir=os.path.join(tmpdir, "out", "stage_a"),
            output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
        )
        recs = list(read_manifest(manifest))
        assert len(recs) == 1
        assert recs[0].labels["original"] == "noise"
        assert os.path.isfile(recs[0].audio["path"])


def test_iter_hf_samples_applies_shuffle_indices():
    """Dataset.shuffle 的 indices 映射必须作用于 raw Arrow 迭代。"""
    try:
        import datasets  # noqa: F401
    except ImportError:
        pytest.skip("datasets 库未安装，跳过 HF loader 测试")

    with tempfile.TemporaryDirectory() as tmpdir:
        wavs = []
        for i in range(4):
            path = os.path.join(tmpdir, f"{i}.wav")
            _make_wav(path)
            wavs.append(path)
        ds = _make_hf_audio_dataset(wavs, labels=list(range(len(wavs))))
        shuffled = ds.shuffle(seed=0)
        indices = getattr(shuffled, "_indices", None)
        assert indices is not None, "datasets 版本未使用 indices 映射，测试前提失效"
        expected = indices.table.column("indices").to_pylist()
        got = [
            row["labels"]
            for _, row in StageAOrchestrator._iter_hf_samples(shuffled, "audio")
        ]
        assert got == expected


def test_stage_a_json_loader_without_audio_field_uses_audio_default():
    """registry 未配置 audio_field 时，json 类型默认 audio 键（旧行为兼容）正常 ingest。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_a = os.path.join(tmpdir, "a.wav")
        wav_b = os.path.join(tmpdir, "b.wav")
        _make_wav(wav_a)
        _make_wav(wav_b)

        json_path = os.path.join(tmpdir, "data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([
                {"audio": wav_a, "text_label": "noise"},
                {"audio": wav_b, "text_label": "speech"},
            ], f, ensure_ascii=False)

        # 故意不写 audio_field：json 类型默认回退 "audio"（旧行为兼容，
        # 未显式配置 audio_field 的存量 registry 不会被静默跳过）
        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_json_default
    name: Test JSON Default Field
    enabled: true
    type: json
    path: {json_path}
    label_field: text_label
    splits: [train]
""")

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            registry_path=registry_path,
        )
        orch = StageAOrchestrator(config)
        manifest = orch.run(
            input_manifest=None,
            output_dir=os.path.join(tmpdir, "out", "stage_a"),
            output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
        )
        recs = list(read_manifest(manifest))
        assert len(recs) == 2
        assert {r.labels["original"] for r in recs} == {"noise", "speech"}


def test_stage_a_warns_when_enabled_dataset_produces_zero_records(caplog):
    """enabled 数据集最终 0 产出时，run() 结束处应打 warning 并列出数据集 id。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 目录存在但没有任何匹配音频 → 0 产出
        empty_dir = os.path.join(tmpdir, "empty_ds")
        os.makedirs(empty_dir)

        wav_ok = os.path.join(tmpdir, "ok_ds", "a.wav")
        _make_wav(wav_ok)

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: empty_ds
    name: Empty Dataset
    enabled: true
    type: directory
    path: {empty_dir}
    input_pattern: "**/*.wav"
    label_source: filename
    splits: [train]
  - id: ok_ds
    name: OK Dataset
    enabled: true
    type: directory
    path: {os.path.join(tmpdir, "ok_ds")}
    input_pattern: "**/*.wav"
    label_source: filename
    splits: [train]
""")

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            registry_path=registry_path,
        )
        orch = StageAOrchestrator(config)
        with caplog.at_level(logging.WARNING):
            orch.run(
                input_manifest=None,
                output_dir=os.path.join(tmpdir, "out", "stage_a"),
                output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
            )
        assert "empty_ds" in caplog.text
        # 正常产出的数据集不应出现在 0 产出 warning 中
        zero_warnings = [r for r in caplog.records if "0 产出" in r.getMessage()]
        assert zero_warnings, "应存在 0 产出 warning"
        assert all("ok_ds" not in r.getMessage() for r in zero_warnings)


def test_stage_a_ratio_split_uses_success_ordinal_and_pbar_covers_skips():
    """ratio split 应基于成功序号分配；跳过条目也推进全局进度条。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        items = []
        for i in range(4):
            p = os.path.join(tmpdir, f"ok{i}.wav")
            _make_wav(p)
            items.append({"audio_path": p, "text_label": "x"})
        # 2 个缺失音频条目会被跳过；旧行为下它们消耗 idx 导致 split 比例偏离
        items.append({"audio_path": os.path.join(tmpdir, "missing1.wav"), "text_label": "x"})
        items.append({"audio_path": os.path.join(tmpdir, "missing2.wav"), "text_label": "x"})

        json_path = os.path.join(tmpdir, "data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_ratio
    name: Test Ratio
    enabled: true
    type: json
    path: {json_path}
    audio_field: audio_path
    label_field: text_label
    splits: [train, test]
    split_strategy: ratio
    split_ratio: [0.5, 0.5]
""")

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            registry_path=registry_path,
        )
        orch = StageAOrchestrator(config)
        pbar = _FakePbar()
        manifest = orch.run(
            input_manifest=None,
            output_dir=os.path.join(tmpdir, "out", "stage_a"),
            output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
            global_pbar=pbar,
        )
        recs = list(read_manifest(manifest))
        assert len(recs) == 4
        # 成功序号 0..3，total=6：pos 0、1/6、2/6 < 0.5 → train；3/6=0.5 → test
        assert sorted(r.split for r in recs) == ["test", "train", "train", "train"]
        # 进度条：成功 4 + 跳过 2 = 6 次 update，与预统计 total 一致
        assert pbar.n == 6


def test_stage_a_directory_failed_files_still_advance_global_pbar():
    """directory 读取失败条目同样推进全局进度条，保持 total/update 一致。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "ds")
        _make_wav(os.path.join(dataset_dir, "ok.wav"))
        # 损坏文件：内容是文本不是音频，sf.info 读取失败
        bad_path = os.path.join(dataset_dir, "bad.wav")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("not an audio file")

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_dir_fail
    name: Test Directory Failure
    enabled: true
    type: directory
    path: {dataset_dir}
    input_pattern: "**/*.wav"
    label_source: filename
    splits: [train]
""")

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            registry_path=registry_path,
        )
        orch = StageAOrchestrator(config)
        pbar = _FakePbar()
        manifest = orch.run(
            input_manifest=None,
            output_dir=os.path.join(tmpdir, "out", "stage_a"),
            output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
            global_pbar=pbar,
        )
        recs = list(read_manifest(manifest))
        assert len(recs) == 1
        assert pbar.n == 2


def test_stage_a_bounded_prefetch_and_chunked_info(monkeypatch):
    """预取超限时回退重扫；元信息分块不改变记录数与进度条语义。"""
    import noise_filter_pipeline.stages.stage_a_ingestion as stage_a_module

    monkeypatch.setattr(stage_a_module, "_PREFETCH_MAX_TOTAL_FILES", 1)
    monkeypatch.setattr(stage_a_module, "_INFO_CHUNK_SIZE", 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "ds")
        _make_wav(os.path.join(dataset_dir, "a.wav"))
        _make_wav(os.path.join(dataset_dir, "b.wav"))
        with open(os.path.join(dataset_dir, "bad.wav"), "w", encoding="utf-8") as f:
            f.write("not an audio file")

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_bounded
    type: directory
    path: {dataset_dir}
    input_pattern: "**/*.wav"
    label_source: filename
    splits: [train]
""")

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            registry_path=registry_path,
        )
        orch = StageAOrchestrator(config)
        pbar = _FakePbar()
        manifest = orch.run(
            input_manifest=None,
            output_dir=os.path.join(tmpdir, "out", "stage_a"),
            output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
            global_pbar=pbar,
        )
        recs = list(read_manifest(manifest))
        assert len(recs) == 2
        assert pbar.n == 3
        # 3 个路径超过上限 1，预取缓存必须为空（ingest 阶段重新扫描）
        assert orch._prefetched_files == {}


def test_save_audio_info_cache_atomic_on_serialize_failure():
    """缓存保存中途失败时旧缓存必须保持完整（tmp+replace 原子写）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "cache.json")
        _save_audio_info_cache(cache_path, {"k": {"sr": 16000}})
        # 不可序列化对象会让 json.dump 在写了一半时抛 TypeError；
        # 直接 "w" 写入此时已截断旧文件，原子写则保证旧内容完好
        _save_audio_info_cache(cache_path, {"bad": object()})
        assert _load_audio_info_cache(cache_path) == {"k": {"sr": 16000}}
        assert not os.path.exists(cache_path + ".tmp")


def test_get_audio_info_negative_cache_typed_trust():
    """负缓存分型信任策略：read_failed 保持信任；implausible 上限变化重探；旧格式重算。"""
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmpdir:
        wav = os.path.join(tmpdir, "a.wav")
        _make_wav(wav)
        key = _file_fingerprint(_Path(wav))
        limit_sec = 6.0 * 3600.0

        # read_failed 负条目：即使文件实际可读也保持信任，不重探
        sr, _, _, _, reason = _get_audio_info(
            _Path(wav),
            {key: {"error": True, "reason": "read_failed", "limit_hours": 6.0}},
            limit_sec,
        )
        assert sr is None and reason == "read_failed"

        # implausible 且判定上限与当前配置一致：信任
        sr, _, _, _, reason = _get_audio_info(
            _Path(wav),
            {key: {"error": True, "reason": "implausible", "limit_hours": 6.0}},
            limit_sec,
        )
        assert sr is None and reason == "implausible"

        # implausible 但配置上限已变化：不信任该负条目，重新读取真实元信息
        sr, duration, channels, _, reason = _get_audio_info(
            _Path(wav),
            {key: {"error": True, "reason": "implausible", "limit_hours": 24.0}},
            limit_sec,
        )
        assert sr == 16000 and reason is None

        # 旧格式负条目（缺 reason/limit_hours）：按不可复用重算，不得崩溃
        sr, _, _, _, reason = _get_audio_info(
            _Path(wav),
            {key: {"error": True, "path": wav}},
            limit_sec,
        )
        assert sr == 16000 and reason is None


def test_stage_a_directory_writes_typed_negative_cache_entry():
    """读取失败的文件应写入带 reason/limit_hours 的分型负缓存条目。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "ds")
        os.makedirs(dataset_dir)
        with open(os.path.join(dataset_dir, "bad.wav"), "w", encoding="utf-8") as f:
            f.write("not an audio file")

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_negcache
    name: Test Negative Cache
    enabled: true
    type: directory
    path: {dataset_dir}
    input_pattern: "**/*.wav"
    label_source: filename
    splits: [train]
""")

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            registry_path=registry_path,
        )
        output_dir = os.path.join(tmpdir, "out", "stage_a")
        StageAOrchestrator(config).run(
            input_manifest=None,
            output_dir=output_dir,
            output_manifest=os.path.join(output_dir, "a_manifest.jsonl"),
        )
        cache = _load_audio_info_cache(_audio_info_cache_path(output_dir, "test_negcache"))
        assert len(cache) == 1
        cache_entry = next(iter(cache.values()))
        assert cache_entry["error"] is True
        assert cache_entry["reason"] == "read_failed"
        assert cache_entry["limit_hours"] == 6.0


def test_prune_audio_info_cache_drops_entries_for_missing_files():
    """指向已删除文件的旧缓存条目必须被清理，避免缓存无界增长。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        alive = os.path.join(tmpdir, "alive.wav")
        _make_wav(alive)
        cache = {
            "keep": {"sr": 16000, "duration": 1.0, "channels": 1, "path": alive},
            "gone": {
                "sr": 16000, "duration": 1.0, "channels": 1,
                "path": os.path.join(tmpdir, "gone.wav"),
            },
            "no_path": {"error": True, "reason": "read_failed"},
        }

        assert _prune_audio_info_cache(cache, keep_keys=set()) == 2
        assert set(cache) == {"keep"}

        # 本轮刚刷新过的条目不做 stat（keep_keys 保护），即使路径取不到也保留
        fresh = {"fresh": {"path": os.path.join(tmpdir, "not-created.wav")}}
        assert _prune_audio_info_cache(fresh, keep_keys={"fresh"}) == 0
        assert set(fresh) == {"fresh"}


def test_stage_a_cache_flushed_before_generator_exhausted(monkeypatch):
    """消费方提前中断时，已读取的元信息必须已落盘。

    回归：缓存曾在生成器尾部一次性写回，后续数据集失败/取消会丢掉本轮
    全部 sf.info 结果，下次运行全量重扫。
    """
    import noise_filter_pipeline.stages.stage_a_ingestion as stage_a_module

    monkeypatch.setattr(stage_a_module, "_CACHE_FLUSH_ENTRIES", 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "ds")
        os.makedirs(dataset_dir)
        for i in range(3):
            _make_wav(os.path.join(dataset_dir, f"a{i}.wav"))

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_cache_flush
    name: Test Cache Flush
    enabled: true
    type: directory
    path: {dataset_dir}
    input_pattern: "**/*.wav"
    label_source: filename
    splits: [train]
""")

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            registry_path=registry_path,
        )
        orchestrator = StageAOrchestrator(config)
        # run() 才会注入 global_pbar；直接驱动 _ingest_directory 时显式置空
        orchestrator.global_pbar = None
        registry = DatasetRegistry(registry_path)
        entry = next(e for e in registry.entries.values() if e.id == "test_cache_flush")
        output_dir = os.path.join(tmpdir, "out", "stage_a")
        os.makedirs(output_dir, exist_ok=True)

        gen = orchestrator._ingest_directory(
            entry, {"output_dir": output_dir}, registry, num_workers=1
        )
        try:
            next(gen)
            next(gen)
            cache = _load_audio_info_cache(
                _audio_info_cache_path(output_dir, "test_cache_flush")
            )
            # 生成器尚未耗尽（还剩 1 个文件），前 2 条已落盘
            assert len(cache) == 2
        finally:
            gen.close()


def test_stage_a_orchestrator_annotations_resolve():
    """__init__ 注解 Optional[Set[str]] 必须可解析（typing Set 导入回归）。"""
    import typing

    hints = typing.get_type_hints(StageAOrchestrator.__init__)
    assert "dataset_ids" in hints


def test_stage_a_json_skips_non_dict_items(caplog):
    """JSON 条目混入非 dict 元素时应跳过并告警，不影响其余条目 ingest。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_wav(os.path.join(tmpdir, "a.wav"))
        json_path = os.path.join(tmpdir, "data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([
                {"audio_path": "a.wav", "text_label": "ok"},
                "broken-entry",
                42,
            ], f, ensure_ascii=False)

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_json_bad
    type: json
    path: {json_path}
    audio_field: audio_path
    label_field: text_label
""")

        config = UnifiedPipelineConfig(output_root=os.path.join(tmpdir, "out"), registry_path=registry_path)
        with caplog.at_level(logging.WARNING):
            manifest = StageAOrchestrator(config).run(
                input_manifest=None,
                output_dir=os.path.join(tmpdir, "out", "stage_a"),
                output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
            )
        recs = list(read_manifest(manifest))
        assert len(recs) == 1
        assert recs[0].labels["original"] == "ok"
        assert "不是 dict" in caplog.text


def test_stage_a_csv_null_label_separator_returns_whole_string():
    """label_separator 显式配置 null 时不拆分，含逗号标签原样返回整串。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_a = os.path.join(tmpdir, "a.wav")
        _make_wav(wav_a)

        csv_path = os.path.join(tmpdir, "data.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "tag"])
            writer.writerow([wav_a, "speech,music"])

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_csv_nullsep
    type: csv
    path: {csv_path}
    audio_column: path
    label_column: tag
    label_separator: null
""")

        config = UnifiedPipelineConfig(output_root=os.path.join(tmpdir, "out"), registry_path=registry_path)
        manifest = StageAOrchestrator(config).run(
            input_manifest=None,
            output_dir=os.path.join(tmpdir, "out", "stage_a"),
            output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
        )
        recs = list(read_manifest(manifest))
        assert len(recs) == 1
        assert recs[0].labels["original"] == "speech,music"


def test_stage_a_hf_missing_split_error_lists_available(caplog):
    """load_from_disk 得到 DatasetDict 且配置 split 不存在时，明确报错并列出可用 split。"""
    try:
        import datasets
    except ImportError:
        pytest.skip("datasets 库未安装，跳过 HF DatasetDict 测试")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_a = os.path.join(tmpdir, "a.wav")
        _make_wav(wav_a, duration=1.0)

        def _make_ds():
            return _make_hf_audio_dataset(wav_a)

        hf_path = os.path.join(tmpdir, "hf_ds")
        datasets.DatasetDict({"train": _make_ds(), "validation": _make_ds()}).save_to_disk(hf_path)

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: test_hf_split
    type: huggingface
    path: {hf_path}
    audio_field: audio
    label_field: labels
    splits: [test]
""")

        config = UnifiedPipelineConfig(output_root=os.path.join(tmpdir, "out"), registry_path=registry_path)
        with caplog.at_level(logging.ERROR):
            manifest = StageAOrchestrator(config).run(
                input_manifest=None,
                output_dir=os.path.join(tmpdir, "out", "stage_a"),
                output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
            )
        # 不崩溃、0 产出，且 error 日志列出可用 split
        assert len(list(read_manifest(manifest))) == 0
        assert "可用 split" in caplog.text
        assert "validation" in caplog.text


def test_stage_a_esc50_label_mapping():
    """ESC-50 文件名（1-100032-A-0）应按末段类别序号 +1 命中映射键 "1".."50"。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "esc50")
        # 类别 0 → 键 "1" → Dog；类别 4 → 键 "5" → Frog
        _make_wav(os.path.join(dataset_dir, "1-100032-A-0.wav"))
        _make_wav(os.path.join(dataset_dir, "2-200036-B-4.wav"))

        # label_mapping 仅文件名时按 <registry 目录>/labels/ 解析
        # （Stage A 与 validator 统一口径，registry 在哪个目录就以谁为基准）
        labels_dir = os.path.join(tmpdir, "labels")
        os.makedirs(labels_dir)
        with open(os.path.join(labels_dir, "esc50_to_audioset.yaml"), "w", encoding="utf-8") as f:
            f.write('mappings:\n  "1": Dog\n  "5": Frog\n')

        registry_path = os.path.join(tmpdir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write(f"""
datasets:
  - id: esc50
    name: ESC-50
    enabled: true
    type: directory
    path: {dataset_dir}
    input_pattern: "**/*.wav"
    label_source: mapping
    label_mapping: esc50_to_audioset.yaml
    splits: [train]
    split_strategy: none
""")

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            registry_path=registry_path,
        )
        orch = StageAOrchestrator(config)
        manifest = orch.run(
            input_manifest=None,
            output_dir=os.path.join(tmpdir, "out", "stage_a"),
            output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
        )
        recs = {os.path.basename(r.audio["path"]): r for r in read_manifest(manifest)}
        assert len(recs) == 2
        assert recs["1-100032-A-0.wav"].labels["original"] == "Dog"
        assert recs["2-200036-B-4.wav"].labels["original"] == "Frog"


def test_stage_a_relative_entry_path_resolved_against_registry_dir():
    """registry 中的相对 entry.path 应按 registry 文件所在目录解析为绝对路径。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # registry 放在子目录 reg/，数据集路径写相对值 "data"
        reg_dir = os.path.join(tmpdir, "reg")
        os.makedirs(reg_dir)
        _make_wav(os.path.join(reg_dir, "data", "a.wav"))
        _make_wav(os.path.join(reg_dir, "data", "b.wav"))

        registry_path = os.path.join(reg_dir, "registry.yaml")
        with open(registry_path, "w", encoding="utf-8") as f:
            f.write("""
datasets:
  - id: rel_ds
    name: Relative Path Dataset
    enabled: true
    type: directory
    path: data
    input_pattern: "**/*.wav"
    label_source: filename
    splits: [train]
""")

        config = UnifiedPipelineConfig(
            output_root=os.path.join(tmpdir, "out"),
            registry_path=registry_path,
        )
        orch = StageAOrchestrator(config)
        manifest = orch.run(
            input_manifest=None,
            output_dir=os.path.join(tmpdir, "out", "stage_a"),
            output_manifest=os.path.join(tmpdir, "out", "stage_a", "a_manifest.jsonl"),
        )
        recs = list(read_manifest(manifest))
        assert len(recs) == 2
        # 相对路径解析后产出绝对音频路径（测试进程 cwd 与 registry 目录不同，
        # 若按 cwd 解析会找不到文件而 0 产出）
        assert all(os.path.isabs(r.audio["path"]) for r in recs)
        assert all(r.audio["path"].startswith(os.path.join(reg_dir, "data")) for r in recs)
