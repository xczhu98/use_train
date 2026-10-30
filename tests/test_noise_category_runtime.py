"""Runtime contracts for opt-in additive-noise category balancing."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import soundfile

from core.data.noise_admission import (
    NOISE_MANIFEST_SCHEMA_VERSION,
    load_canonical_voice_policy,
)
from core.data.dynamic_dataset_base import BaseDynamicMixingDataset
from core.data.noise_admission_policy import (
    ADMISSION_POLICY_VERSION,
    CLASSIFIER_LABEL_COUNT,
    EVIDENCE_CONTRACT_VERSION,
    GENERIC_NOISE_MID,
    SCORE_CONTRACT,
)
from core.data.noise_manifest_adapter import read_noise_source_manifest
from core.data.sampling import SamplingConfig
from core.data.simulation.resource_store import ResourceStore
from tasks.se.data.dynamic_dataset import DynamicMixingDataset as SEDynamicMixingDataset


class ScriptedRng:
    """Minimal deterministic RNG exposing the interface used by rng_choice."""

    def __init__(self, choices: list[int | str]):
        self._choices = iter(choices)

    def integers(self, low: int, high: int) -> int:
        value = next(self._choices)
        if value == "last":
            return high - 1
        assert isinstance(value, int)
        assert low <= value < high
        return value

    def random(self) -> float:
        return 1.0

    def uniform(self, low: float, high: float) -> float:
        return low


def _write_wav(path: Path, value: float, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, np.full(sr, value, dtype=np.float32), sr)


def _write_scp(path: Path, uid: str, wav: Path, sr: int = 16000) -> None:
    path.write_text(f"{uid} {sr} {sr} {wav}\n", encoding="utf-8")


@lru_cache(maxsize=1)
def _canonical_policy():
    return load_canonical_voice_policy()[0]


def _artifact_fingerprints() -> dict[str, str]:
    policy = _canonical_policy()
    return {
        "voice_policy_sha256": policy.sha256,
        "audiotag_label_dict_sha256": policy.label_space_sha256,
        "audiotag_checkpoint_sha256": "2" * 64,
        "qwen_model_sha256": "3" * 64,
        "c1_prompt_sha256": "4" * 64,
    }


def _top5_hits(category_name: str, *, c1_branch: str) -> list[dict]:
    """构造 C2 audioset_top5 对应的 v6 发布审计，不持久化旧分数/阈值字段。"""
    policy = _canonical_policy()
    selected_mid = next(
        (
            mid
            for mid, label in policy.mid_to_label.items()
            if label == category_name and policy.class_of(mid) == "noise"
        ),
        GENERIC_NOISE_MID,
    )
    mids = [selected_mid]
    if c1_branch == "babble":
        mids.append("/m/07qfr4h")
    for mid in policy.index_to_mid.values():
        if len(mids) == 5:
            break
        if mid not in mids and policy.class_of(mid) == "noise":
            mids.append(mid)
    return [
        {
            "index": policy.mid_to_index[mid],
            "mid": mid,
            "label": policy.label_of(mid),
            "kind": policy.class_of(mid),
            "prob": 1.0 - rank * 0.1,
        }
        for rank, mid in enumerate(mids)
    ]


def _v6_verification(category_name: str, c1_intelligibility: str) -> dict:
    policy = _canonical_policy()
    return {
        "c1_status": "passed",
        "c1_intelligibility": c1_intelligibility,
        "c2_status": "passed",
        "c3_status": "passed",
        "d2_status": "passed",
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "score_contract": SCORE_CONTRACT,
        "class_count": CLASSIFIER_LABEL_COUNT,
        "label_space_sha256": policy.label_space_sha256,
        "artifact_fingerprints": _artifact_fingerprints(),
        "top5_hits": _top5_hits(category_name, c1_branch=c1_intelligibility),
    }


def _category_noise_record(
    uid: str,
    fs: int,
    path: Path,
    category_id,
    category_name,
    dataset_id,
    *,
    c1_branch: str = "noise",
    roles: list[str] | None = None,
) -> dict:
    """构造仅用于 category runtime 行为测试的轻量 metadata。"""
    del c1_branch
    record = {
        "uid": uid,
        "fs": fs,
        "length": fs,
        "path": str(path),
        "balance_category": category_name,
        "category_name": category_name,
        "category_id": category_id,
        "dataset_id": dataset_id,
    }
    if roles is not None:
        record["roles"] = roles
    return record


def _v6_noise_record(
    uid: str,
    fs: int,
    path: Path,
    category_id: int,
    category_name: str,
    dataset_id: str,
    *,
    c1_branch: str = "noise",
) -> dict:
    """构造可通过 production admission gate 的 schema v6 记录。"""
    policy = _canonical_policy()
    top5_hits = _top5_hits(category_name, c1_branch=c1_branch)
    noise_hits = [hit for hit in top5_hits if hit["kind"] == "noise"]
    mid = noise_hits[0]["mid"] if noise_hits else GENERIC_NOISE_MID
    decision = "accept_babble" if c1_branch == "babble" else "accept_noise"
    return {
        "uid": uid,
        "fs": fs,
        "length": fs,
        "path": str(path),
        "manifest_schema_version": NOISE_MANIFEST_SCHEMA_VERSION,
        "balance_category_mid": mid,
        "balance_category": policy.label_of(mid),
        "balance_category_source": "noise_top5_hit",
        "category_name": policy.label_of(mid),
        "category_id": category_id,
        "dataset_id": dataset_id,
        "admission_allowed": True,
        "admission_decision": decision,
        "admission_reason_codes": [
            "admitted_babble" if c1_branch == "babble" else "admitted_noise"
        ],
        "admission_policy_version": ADMISSION_POLICY_VERSION,
        "admission_policy_sha256": policy.sha256,
        "verification": _v6_verification(category_name, c1_branch),
    }


def _legacy_noise_record(uid: str, fs: int, path: Path, category_id, category_name, dataset_id) -> dict:
    """旧分类 JSONL 记录（无 v4 准入字段）。

    训练 runtime 只要求基础 schema/category，允许这类记录与 publication 记录
    混合消费；production_admission 门禁会在准入处拒绝它（见负向用例）。
    """
    return {
        "uid": uid,
        "fs": fs,
        "length": fs,
        "path": str(path),
        "category_id": category_id,
        "category_name": category_name,
        "dataset_id": dataset_id,
    }


def _make_dataset(tmp_path: Path, noise_records: list[dict], *, sampling: SamplingConfig | None, reuse_noise: bool = False):
    speech = tmp_path / "speech.wav"
    rir = tmp_path / "rir.wav"
    wind = tmp_path / "wind.wav"
    _write_wav(speech, 0.1)
    _write_wav(rir, 0.2)
    _write_wav(wind, 0.3)
    speech_scp = tmp_path / "speech.scp"
    rir_scp = tmp_path / "rir.scp"
    wind_scp = tmp_path / "wind.scp"
    _write_scp(speech_scp, "speech", speech)
    _write_scp(rir_scp, "rir", rir)
    _write_scp(wind_scp, "wind", wind)
    noise_scp = tmp_path / "noise.jsonl"
    noise_scp.write_text("\n".join(json.dumps(record) for record in noise_records) + "\n", encoding="utf-8")
    return SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=str(wind_scp),
        sampling_cfg=sampling,
        simulation_cfg={"prob_reverberation": 0.0, "reuse_noise": reuse_noise},
        filter_by_duration=False,
    )


def _balanced_sampling() -> SamplingConfig:
    return SamplingConfig.from_dict({"noise": {"balance_categories": True}})


def _set_noise_reader(dataset, reader) -> None:
    """通过正式 ResourceStore 注入测试读取器，不修改 Dataset 模块级符号。"""
    dataset.resource_store = ResourceStore(
        strict_load=bool(getattr(dataset, "strict_load", True)),
        noise_reader=reader,
    )


def test_enabled_noise_sampling_requires_complete_category_metadata(tmp_path):
    """均衡仅依赖轻量 category metadata，不依赖 strict v6 admission 字段。"""
    paths = [tmp_path / f"noise-{index}.wav" for index in range(3)]
    for path in paths:
        _write_wav(path, 0.1)
    good = _legacy_noise_record("good", 16000, paths[0], 1, "traffic", "")
    bad_id = _legacy_noise_record("bad-id", 16000, paths[1], True, "traffic", "")
    bad_name = _legacy_noise_record("bad-name", 16000, paths[2], 2, " traffic ", "")

    with pytest.raises(ValueError, match=r"complete category metadata.*bad-id.*bad-name"):
        _make_dataset(tmp_path, [good, bad_id, bad_name], sampling=_balanced_sampling())


def test_enabled_noise_sampling_rejects_metadata_free_scp(tmp_path):
    wav = tmp_path / "noise.wav"
    _write_wav(wav, 0.1)
    _make_dataset(
        tmp_path,
        [_legacy_noise_record("noise", 16000, wav, 0, "traffic", "")],
        sampling=None,
    )
    legacy = tmp_path / "legacy.scp"
    _write_scp(legacy, "noise", wav)
    with pytest.raises(ValueError, match="complete category metadata"):
        SEDynamicMixingDataset(
            speech_scp=str(tmp_path / "speech.scp"), noise_scp=str(legacy),
            rir_scp=str(tmp_path / "rir.scp"), windnoise_scp=str(tmp_path / "wind.scp"),
            sampling_cfg=_balanced_sampling(), simulation_cfg={"prob_reverberation": 0.0},
            filter_by_duration=False,
        )


def test_enabled_noise_sampling_rejects_reuse_noise(tmp_path):
    wav = tmp_path / "noise.wav"
    _write_wav(wav, 0.1)
    with pytest.raises(ValueError, match="reuse_noise"):
        _make_dataset(tmp_path, [_category_noise_record("noise", 16000, wav, 0, "traffic", "set-a")], sampling=_balanced_sampling(), reuse_noise=True)


def test_balanced_additive_selection_chooses_category_before_uid(tmp_path):
    paths = [tmp_path / f"noise-{index}.wav" for index in range(3)]
    for path in paths:
        _write_wav(path, 0.1)
    dataset = _make_dataset(tmp_path, [
        _category_noise_record("a-1", 16000, paths[0], 1, "a", "set-a"),
        _category_noise_record("a-2", 16000, paths[1], 1, "a", "set-a"),
        _category_noise_record("b-1", 16000, paths[2], 2, "b", "set-b"),
    ], sampling=_balanced_sampling())

    assert dataset._select_balanced_additive_noise_uid(16000, ScriptedRng([0, 1])) == "a-2"
    assert dataset._select_balanced_additive_noise_uid(16000, ScriptedRng([1, 0])) == "b-1"


def test_noise_reader_records_identity_category_crop_and_runtime_asset(tmp_path):
    path = tmp_path / "traffic.wav"
    _write_wav(path, 0.1)
    dataset = _make_dataset(
        tmp_path,
        [_category_noise_record("traffic", 16000, path, 7, "Traffic", "set")],
        sampling=_balanced_sampling(),
    )
    record = {"noise_sources": [], "noise_assets": []}
    dataset._active_acoustic_plan_record = record

    sample = dataset._read_noise_uid(
        "traffic", 16000, 1000, np.random.default_rng(17)
    )

    source = record["noise_sources"][0]
    assert source.uid == "traffic"
    assert source.path == str(path)
    assert source.category_id == 7
    assert source.category_name == "Traffic"
    assert source.crop is not None
    assert source.crop.num_samples == 1000
    np.testing.assert_array_equal(record["noise_assets"][0], sample)


def test_balanced_additive_selection_falls_back_exact_then_higher_then_any(tmp_path):
    paths = [tmp_path / f"noise-{index}.wav" for index in range(3)]
    for path in paths:
        _write_wav(path, 0.1)
    dataset = _make_dataset(tmp_path, [
        _category_noise_record("low", 8000, paths[0], 0, "low", "set"),
        _category_noise_record("exact", 16000, paths[1], 1, "exact", "set"),
        _category_noise_record("high", 48000, paths[2], 2, "high", "set"),
    ], sampling=_balanced_sampling())

    assert dataset._select_balanced_additive_noise_uid(16000, ScriptedRng([0, 0])) == "exact"
    assert dataset._select_balanced_additive_noise_uid(24000, ScriptedRng([0, 0])) == "high"
    assert dataset._select_balanced_additive_noise_uid(96000, ScriptedRng([0, 0])) == "low"


def test_balanced_runtime_switches_only_ordinary_additive_reading(tmp_path, monkeypatch):
    paths = [tmp_path / f"noise-{index}.wav" for index in range(2)]
    for path in paths:
        _write_wav(path, 0.1)
    dataset = _make_dataset(tmp_path, [
        _category_noise_record("noise", 16000, paths[0], 0, "traffic", "set"),
    ], sampling=_balanced_sampling())
    captured_paths = []
    _set_noise_reader(
        dataset,
        lambda path, *args, **kwargs: captured_paths.append(path)
        or np.zeros((1, 4), dtype=np.float32),
    )

    dataset._read_one_additive_noise_scp_only(16000, 4, ScriptedRng([0, 0]))

    assert captured_paths == [str(paths[0])]


def test_dual_balanced_additive_searches_other_tiers_for_different_category(tmp_path, monkeypatch):
    paths = [tmp_path / f"noise-{index}.wav" for index in range(4)]
    for path in paths:
        _write_wav(path, 0.1)
    dataset = _make_dataset(tmp_path, [
        _category_noise_record("exact-first", 16000, paths[0], 0, "traffic", "set"),
        _category_noise_record("exact-same", 16000, paths[1], 0, "traffic", "set"),
        _category_noise_record("low-same", 8000, paths[2], 0, "traffic", "set"),
        _category_noise_record("high-other", 48000, paths[3], 1, "Thunder", "set"),
    ], sampling=_balanced_sampling())
    captured_paths = []
    _set_noise_reader(
        dataset,
        lambda path, *args, **kwargs: captured_paths.append(path)
        or np.ones((1, 4), dtype=np.float32),
    )

    dataset._read_dual_additive_noise_scp_blended(16000, 4, ScriptedRng([0, 0, 0, 0]))

    assert captured_paths == [str(paths[0]), str(paths[3])]


def test_dual_balanced_additive_requires_different_category_or_single(tmp_path, monkeypatch):
    paths = [tmp_path / f"noise-{index}.wav" for index in range(2)]
    for path in paths:
        _write_wav(path, 0.1)
    dataset = _make_dataset(tmp_path, [
        _category_noise_record("first", 16000, paths[0], 0, "traffic", "set"),
        _category_noise_record("second", 16000, paths[1], 1, "Thunder", "set"),
    ], sampling=_balanced_sampling())
    captured_paths = []
    _set_noise_reader(
        dataset,
        lambda path, *args, **kwargs: captured_paths.append(path)
        or np.ones((1, 4), dtype=np.float32),
    )

    dataset._read_dual_additive_noise_scp_blended(16000, 4, ScriptedRng([0, 0, 0, 0]))
    assert captured_paths == [str(paths[0]), str(paths[1])]

    same_category = _make_dataset(tmp_path / "same-category", [
        _category_noise_record("one", 16000, tmp_path / "same-category" / "one.wav", 0, "traffic", "set"),
        _category_noise_record("two", 16000, tmp_path / "same-category" / "two.wav", 0, "traffic", "set"),
    ], sampling=_balanced_sampling())
    _set_noise_reader(
        same_category,
        lambda path, *args, **kwargs: captured_paths.append(path)
        or np.ones((1, 4), dtype=np.float32),
    )
    assert same_category._select_balanced_additive_noise_uid(
        16000, ScriptedRng([0, 0]), exclude_category_id=0, exclude_uid="one"
    ) is None

    captured_paths.clear()
    same_category._read_dual_additive_noise_scp_blended(16000, 4, ScriptedRng([0, 0]))
    assert len(captured_paths) == 1


def test_dual_classified_natural_uses_different_category_uid_uniformly(
    tmp_path, monkeypatch
):
    """非均衡 classified JSONL 的第二段在异类 UID 合集中直接均匀选择。"""
    paths = [tmp_path / f"natural-{index}.wav" for index in range(4)]
    for path in paths:
        _write_wav(path, 0.1)
    dataset = _make_dataset(
        tmp_path,
        [
            _category_noise_record("a-1", 16000, paths[0], 0, "Vehicle", "set"),
            _category_noise_record("a-2", 16000, paths[1], 0, "Vehicle", "set"),
            _category_noise_record("b-1", 16000, paths[2], 1, "Thunder", "set"),
            _category_noise_record("b-2", 16000, paths[3], 1, "Thunder", "set"),
        ],
        sampling=None,
    )
    assert dataset._select_natural_different_category_uid(
        16000, ScriptedRng([0]), exclude_category_id=0, exclude_uid="a-1"
    ) == "b-1"
    assert dataset._select_natural_different_category_uid(
        16000, ScriptedRng([1]), exclude_category_id=0, exclude_uid="a-1"
    ) == "b-2"
    captured_paths = []
    _set_noise_reader(
        dataset,
        lambda path, *args, **kwargs: captured_paths.append(path)
        or np.ones((1, 4), dtype=np.float32),
    )

    dataset._read_dual_additive_noise_scp_blended(
        16000, 4, ScriptedRng([0, 1])
    )

    assert captured_paths == [str(paths[0]), str(paths[3])]


def test_dual_ordinary_scp_preserves_legacy_two_reads(tmp_path, monkeypatch):
    """普通 SCP 无类别语义且无需逃生开关，双噪声继续独立读取两次。"""
    paths = [tmp_path / "legacy-a.wav", tmp_path / "legacy-b.wav"]
    for path in paths:
        _write_wav(path, 0.1)
    speech_scp, rir_scp = _write_speech_rir(tmp_path)
    noise_scp = tmp_path / "legacy.scp"
    noise_scp.write_text(
        "".join(f"legacy-{i} 16000 16000 {path}\n" for i, path in enumerate(paths)),
        encoding="utf-8",
    )
    dataset = SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=None,
        simulation_cfg={"prob_reverberation": 0.0, "prob_wind_noise_sidechain": 0.0},
        filter_by_duration=False,
    )
    captured_paths = []
    _set_noise_reader(
        dataset,
        lambda path, *args, **kwargs: captured_paths.append(path)
        or np.ones((1, 4), dtype=np.float32),
    )

    dataset._read_dual_additive_noise_scp_blended(
        16000, 4, ScriptedRng([0, 1])
    )

    assert captured_paths == [str(paths[0]), str(paths[1])]


def test_dual_unverified_partial_category_map_falls_back_to_legacy_reads(
    tmp_path, monkeypatch
):
    """旧未验证 JSONL 的 category map 不完整时不得索引崩溃或伪称异类。"""
    paths = [tmp_path / "mapped.wav", tmp_path / "unmapped.wav"]
    for path in paths:
        _write_wav(path, 0.1)
    mapped = _legacy_noise_record("mapped", 16000, paths[0], 0, "traffic", "set")
    unmapped = {"uid": "unmapped", "fs": 16000, "length": 16000, "path": str(paths[1])}
    dataset = _make_dataset(tmp_path, [mapped, unmapped], sampling=None)
    assert dataset._noise_category_id_by_uid == {"mapped": 0}
    captured_paths = []
    _set_noise_reader(
        dataset,
        lambda path, *args, **kwargs: captured_paths.append(path)
        or np.ones((1, 4), dtype=np.float32),
    )

    dataset._read_dual_additive_noise_scp_blended(
        16000, 4, ScriptedRng([1, 0, 0])
    )

    assert captured_paths == [str(paths[1]), str(paths[0])]


def test_natural_inconsistent_category_mapping_falls_back_to_plain_reads(tmp_path):
    """同一 category_id 对应多个名称时，非均衡模式不崩溃且禁用异类约束。"""
    paths = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for path in paths:
        _write_wav(path, 0.1)
    records = [
        _legacy_noise_record("a", 16000, paths[0], 0, "Vehicle", ""),
        _legacy_noise_record("b", 16000, paths[1], 0, "Thunder", ""),
    ]
    dataset = _make_dataset(tmp_path, records, sampling=None)
    assert dataset._noise_category_id_by_uid == {}
    assert dataset._ordinary_noise_categories_complete() is False


# ---------------------------------------------------------------------------
# 风噪/music 内嵌分类 JSONL（windnoise_scp 省略时从 noise jsonl 提取）
# ---------------------------------------------------------------------------

def _noise_records_with_embedded(*, wind: bool = True, music: bool = False, tmp_path: Path):
    """构造含 traffic / wind / music 类别的分类 JSONL 记录列表。"""
    records = []
    for name, cat_id, cat_name in [("traffic", 0, "traffic")]:
        path = tmp_path / f"{name}.wav"
        _write_wav(path, 0.1)
        records.append(_category_noise_record(name, 16000, path, cat_id, cat_name, "set"))
    if wind:
        for _idx, (name, cat_name) in enumerate(
            [("wind-a", "Wind"), ("wind-b", "Wind noise (microphone)")]
        ):
            path = tmp_path / f"{name}.wav"
            _write_wav(path, 0.1)
            # 每个 wind 类别独立 category_id（v4 契约：同 id 必须唯一映射类别名）
            records.append(_category_noise_record(name, 16000, path, 412 + _idx, cat_name, "set"))
    if music:
        path = tmp_path / "music.wav"
        _write_wav(path, 0.1)
        records.append(_category_noise_record("music", 16000, path, 100, "Music", "set"))
    return records


def _write_speech_rir(tmp_path: Path) -> tuple[Path, Path]:
    """写 speech.wav / rir.wav 及对应 4 列 SCP，返回 (speech_scp, rir_scp)。"""
    speech = tmp_path / "speech.wav"
    rir = tmp_path / "rir.wav"
    _write_wav(speech, 0.1)
    _write_wav(rir, 0.2)
    speech_scp = tmp_path / "speech.scp"
    rir_scp = tmp_path / "rir.scp"
    _write_scp(speech_scp, "speech", speech)
    _write_scp(rir_scp, "rir", rir)
    return speech_scp, rir_scp


def test_embedded_wind_requires_jsonl_noise_scp(tmp_path):
    """普通 SCP 不受 strict JSONL 门禁；启用风噪但无显式 wind SCP 时才报错。"""
    wav = tmp_path / "noise.wav"
    _write_wav(wav, 0.1)
    legacy = tmp_path / "noise.scp"
    _write_scp(legacy, "noise", wav)
    speech_scp, rir_scp = _write_speech_rir(tmp_path)
    with pytest.raises(ValueError, match="ordinary sources require an explicit non-empty windnoise_scp"):
        SEDynamicMixingDataset(
            speech_scp=str(speech_scp),
            noise_scp=str(legacy),
            rir_scp=str(rir_scp),
            windnoise_scp=None,
            simulation_cfg={"prob_reverberation": 0.0, "prob_wind_noise_sidechain": 0.05},
            filter_by_duration=False,
        )


def test_embedded_wind_skipped_when_wind_disabled_and_noise_is_legacy_scp(tmp_path):
    """风噪关闭时，普通 SCP 无需 allow_unverified 即可构造。"""
    wav = tmp_path / "noise.wav"
    _write_wav(wav, 0.1)
    legacy = tmp_path / "noise.scp"
    _write_scp(legacy, "noise", wav)
    speech_scp, rir_scp = _write_speech_rir(tmp_path)
    dataset = SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(legacy),
        rir_scp=str(rir_scp),
        windnoise_scp=None,
        simulation_cfg={
            "prob_reverberation": 0.0,
            "prob_wind_noise_sidechain": 0.0,
        },
        filter_by_duration=False,
    )
    assert dataset.wn_flt == {}
    assert set(dataset.noise_flt) == {"noise"}


def test_embedded_wind_extracts_wind_categories_only(tmp_path):
    """windnoise_scp=None + 分类 jsonl：wind 池只含风噪类别，普通噪声池不含 wind。"""
    records = _noise_records_with_embedded(tmp_path=tmp_path)
    speech_scp, rir_scp = _write_speech_rir(tmp_path)
    noise_scp = tmp_path / "noise.jsonl"
    noise_scp.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    dataset = SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=None,
        simulation_cfg={
            "prob_reverberation": 0.0,
            "prob_wind_noise_sidechain": 1.0,
        },
        filter_by_duration=False,
    )
    # wind 池：uid 带 jsonl_wind_ 前缀；sidechain-only 仍可从分类 JSONL 建立。
    assert dataset._wind_required(dataset.sim_cfg) is True
    assert set(dataset.wn_flt) == {"jsonl_wind_wind-a", "jsonl_wind_wind-b"}
    # 普通噪声池剔除 wind 类别
    assert set(dataset.noise_flt) == {"traffic"}
    # 均衡 metadata 不含 wind
    assert set(dataset._noise_manifest_metadata) == {"traffic"}


def test_embedded_wind_keeps_roles_override_and_name_fallback(tmp_path):
    """内嵌 roles 优先；字段缺失时按 category name 推导。"""
    traffic = _category_noise_record("traffic", 16000, tmp_path / "traffic.wav", 0, "traffic", "set")
    _write_wav(tmp_path / "traffic.wav", 0.1)
    fake_wind = _category_noise_record("fake-wind", 16000, tmp_path / "fake-wind.wav", 1, "Breeze", "set")
    fake_wind["roles"] = ["wind"]
    _write_wav(tmp_path / "fake-wind.wav", 0.1)
    old_wind = _category_noise_record("old-wind", 16000, tmp_path / "old-wind.wav", 2, "Wind_noise_microphone", "set")
    _write_wav(tmp_path / "old-wind.wav", 0.1)

    speech_scp, rir_scp = _write_speech_rir(tmp_path)
    noise_scp = tmp_path / "noise.jsonl"
    noise_scp.write_text(
        "\n".join(json.dumps(record) for record in [traffic, fake_wind, old_wind]) + "\n",
        encoding="utf-8",
    )
    dataset = SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=None,
        simulation_cfg={"prob_reverberation": 0.0},
        filter_by_duration=False,
    )
    assert set(dataset.wn_flt) == {"jsonl_wind_fake-wind", "jsonl_wind_old-wind"}
    assert set(dataset.noise_flt) == {"traffic"}
    assert dataset.balanced_noise_sampling_enabled is False


def test_embedded_wind_role_is_consumed_without_strict_admission(tmp_path):
    """runtime 消费显式 role，不复核发布端 canonical role/admission。"""
    rain = _legacy_noise_record("rain", 16000, tmp_path / "rain.wav", 0, "Rain", "")
    rain["roles"] = ["wind"]
    _write_wav(tmp_path / "rain.wav", 0.1)

    dataset = _make_dataset(tmp_path, [rain], sampling=None)
    assert set(dataset.wn_flt) == {"wind", "jsonl_wind_rain"}
    assert dataset.noise_flt == {}


def test_embedded_wind_merges_with_explicit_windnoise_scp(tmp_path):
    """windnoise_scp 显式提供 + 分类 jsonl：wind 池 = 显式 ∪ jsonl 内嵌。"""
    records = _noise_records_with_embedded(tmp_path=tmp_path)
    speech_scp, rir_scp = _write_speech_rir(tmp_path)
    noise_scp = tmp_path / "noise.jsonl"
    noise_scp.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    wind = tmp_path / "explicit_wind.wav"
    _write_wav(wind, 0.1)
    wind_scp = tmp_path / "wind.scp"
    _write_scp(wind_scp, "explicit-wind", wind)
    dataset = SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=str(wind_scp),
        simulation_cfg={"prob_reverberation": 0.0},
        filter_by_duration=False,
    )
    # 显式 scp 条目原样 + jsonl 内嵌带前缀
    assert set(dataset.wn_flt) == {"explicit-wind", "jsonl_wind_wind-a", "jsonl_wind_wind-b"}
    assert set(dataset.noise_flt) == {"traffic"}


def test_embedded_wind_no_wind_records_yields_empty_wind_pool(tmp_path):
    """jsonl 无 wind 类别且 windnoise_scp=None：wind 池为空、普通噪声池完整、不报错。"""
    records = [
        _category_noise_record("traffic", 16000, tmp_path / "traffic.wav", 0, "traffic", "set"),
    ]
    _write_wav(tmp_path / "traffic.wav", 0.1)
    speech_scp, rir_scp = _write_speech_rir(tmp_path)
    noise_scp = tmp_path / "noise.jsonl"
    noise_scp.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    dataset = SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=None,
        simulation_cfg={
            "prob_reverberation": 0.0,
            "prob_wind_noise_sidechain": 0.0,
        },
        filter_by_duration=False,
    )
    assert dataset.wn_flt == {}
    assert set(dataset.noise_flt) == {"traffic"}


def test_embedded_wind_required_rejects_empty_wind_pool(tmp_path):
    """side-chain 风噪启用但分类 JSONL 无风噪时，构造期直接报错。"""
    records = _noise_records_with_embedded(wind=False, tmp_path=tmp_path)
    speech_scp, rir_scp = _write_speech_rir(tmp_path)
    noise_scp = tmp_path / "noise.jsonl"
    noise_scp.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="no wind-noise samples"):
        SEDynamicMixingDataset(
            speech_scp=str(speech_scp),
            noise_scp=str(noise_scp),
            rir_scp=str(rir_scp),
            windnoise_scp=None,
            simulation_cfg={
                "prob_reverberation": 0.0,
                "prob_wind_noise_sidechain": 1.0,
            },
            filter_by_duration=False,
        )


def _write_aec_sources(tmp_path: Path, records: list[dict]):
    """写 AEC 全量源（speech/rir/noise.jsonl/farend_echo_pair），返回路径。"""
    speech_scp, rir_scp = _write_speech_rir(tmp_path)
    noise_scp = tmp_path / "noise.jsonl"
    noise_scp.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    farend_wav = tmp_path / "farend.wav"
    echo_wav = tmp_path / "echo.wav"
    _write_wav(farend_wav, 0.1)
    _write_wav(echo_wav, 0.1)
    stereo_wav = tmp_path / "stereo.wav"
    echo, sr = soundfile.read(echo_wav, dtype="float32")
    farend, _ = soundfile.read(farend_wav, dtype="float32")
    soundfile.write(stereo_wav, np.stack([echo, farend], axis=1), sr)
    pair_json = tmp_path / "farend_echo.json"
    pair_json.write_text(
        json.dumps({"pairs": [str(stereo_wav)]}),
        encoding="utf-8",
    )
    return speech_scp, rir_scp, noise_scp, pair_json


def _write_tse_sources(tmp_path: Path, records: list[dict]):
    """写 TSE strict JSONL speech、rir/noise 与嵌套 enrollment 源，返回路径。"""
    _, rir_scp = _write_speech_rir(tmp_path)
    speech = tmp_path / "speech.wav"
    speech_jsonl = tmp_path / "speech.jsonl"
    speech_jsonl.write_text(
        json.dumps(
            {
                "uid": "speech",
                "fs": 16000,
                "length": 16000,
                "path": str(speech),
                "language": "zh",
                "dataset_id": "fixture",
                "speaker_id": "spk1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    noise_scp = tmp_path / "noise.jsonl"
    noise_scp.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    spk2enroll = tmp_path / "spk2enroll.json"
    spk2enroll.write_text(
        json.dumps({"fixture": {"spk1": [str(speech)]}}), encoding="utf-8"
    )
    return speech_jsonl, rir_scp, noise_scp, spk2enroll


def test_embedded_music_merges_with_explicit_music_scp(tmp_path):
    """TSE：music_scp 显式提供 + jsonl 含 Music/乐器：music 池 = 显式 ∪ jsonl 内嵌。"""
    from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset

    records = _noise_records_with_embedded(music=True, tmp_path=tmp_path)
    # 追加乐器类别（Guitar）验证内嵌音乐类别集合不只 Music 一种
    guitar_path = tmp_path / "guitar.wav"
    _write_wav(guitar_path, 0.1)
    records.append(_category_noise_record("guitar", 16000, guitar_path, 101, "Guitar", "set"))
    speech_scp, rir_scp, noise_scp, spk2enroll = _write_tse_sources(tmp_path, records)
    explicit = tmp_path / "explicit_music.wav"
    _write_wav(explicit, 0.1)
    music_scp = tmp_path / "music.scp"
    _write_scp(music_scp, "explicit-music", explicit)

    dataset = TSEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=None,
        spk2enroll_json=str(spk2enroll),
        music_scp=str(music_scp),
        simulation_cfg={"prob_target_only": 1.0, "prob_reverberation": 0.0, "music_ratio": 0.1},
        filter_by_duration=False,
    )
    assert set(dataset.music_flt) == {"explicit-music", "jsonl_music_music", "jsonl_music_guitar"}
    # Music / 乐器类别不归为普通噪声（wind 仍由 windnoise_scp=None 从 jsonl 提取）
    assert set(dataset.noise_flt) == {"traffic"}


def test_embedded_music_ratio_requires_entries(tmp_path):
    """TSE：music_ratio>0 且 music_scp 与 jsonl Music 均无 → 构造期报错。"""
    from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset

    records = [
        _category_noise_record("traffic", 16000, tmp_path / "traffic.wav", 0, "traffic", "set"),
    ]
    _write_wav(tmp_path / "traffic.wav", 0.1)
    speech_scp, rir_scp, noise_scp, spk2enroll = _write_tse_sources(tmp_path, records)

    with pytest.raises(ValueError, match="music_ratio > 0 but no music entries"):
        TSEDynamicMixingDataset(
            speech_scp=str(speech_scp),
            noise_scp=str(noise_scp),
            rir_scp=str(rir_scp),
            windnoise_scp=None,
            spk2enroll_json=str(spk2enroll),
            music_scp=None,
            simulation_cfg={"prob_target_only": 1.0,
                "prob_reverberation": 0.0,
                "prob_wind_noise_sidechain": 0.0,
                "music_ratio": 0.1,
            },
            filter_by_duration=False,
        )


def test_embedded_music_categories_stay_ordinary_noise_in_se(tmp_path):
    """SE：音乐类别（Music/乐器）不剔除，仍是普通噪声（music 内嵌仅 TSE 生效）。"""
    records = _noise_records_with_embedded(music=True, tmp_path=tmp_path)
    guitar_path = tmp_path / "guitar.wav"
    _write_wav(guitar_path, 0.1)
    records.append(_category_noise_record("guitar", 16000, guitar_path, 101, "Guitar", "set"))
    speech_scp, rir_scp = _write_speech_rir(tmp_path)
    noise_scp = tmp_path / "noise.jsonl"
    noise_scp.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    dataset = SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=None,
        simulation_cfg={"prob_reverberation": 0.0},
        filter_by_duration=False,
    )
    # Music / Guitar 保留在普通噪声池（不是 music 池——SE 无 music 概念）
    assert set(dataset.noise_flt) == {"traffic", "music", "guitar"}
    # 风噪仍按 jsonl 内嵌提取
    assert set(dataset.wn_flt) == {"jsonl_wind_wind-a", "jsonl_wind_wind-b"}
    # 无 music 池属性（SE dataset 不建立 music 池）
    assert not hasattr(dataset, "music_flt")


def test_embedded_wind_extracted_in_aec(tmp_path):
    """AEC：windnoise_scp=None + 分类 jsonl 时，wind 类别从 jsonl 内嵌提取。"""
    from tasks.aec.data.dynamic_dataset import DynamicMixingDataset as AECDynamicMixingDataset

    records = _noise_records_with_embedded(tmp_path=tmp_path)
    speech_scp, rir_scp, noise_scp, pair_json = _write_aec_sources(tmp_path, records)
    dataset = AECDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=None,
        real_farend_echo_pair_json=str(pair_json),
        simulation_cfg={
            "prob_reverberation": 0.0,
            "prob_wind_noise_sidechain": 0.0,
            "prob_target_only": 0.15,
            "prob_noise_only": 0.15,
            "prob_echo_only": 0.15,
            "prob_noise_echo_only": 0.15,
            "prob_target_noise": 0.15,
            "prob_target_echo": 0.15,
            "prob_target_noise_echo": 0.10,
        },
        filter_by_duration=False,
    )
    assert set(dataset.wn_flt) == {"jsonl_wind_wind-a", "jsonl_wind_wind-b"}
    assert set(dataset.noise_flt) == {"traffic"}


def test_tse_balancing_validates_ordinary_pool_after_music_role_split(tmp_path):
    """music role 可无 category；均衡完整性只检查拆池后的 ordinary pool。"""
    from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset

    traffic = tmp_path / "traffic.wav"
    music = tmp_path / "music.wav"
    _write_wav(traffic, 0.1)
    _write_wav(music, 0.1)
    records = [
        _legacy_noise_record("traffic", 16000, traffic, 0, "traffic", ""),
        {
            "uid": "music", "fs": 16000, "length": 16000,
            "path": str(music), "roles": ["music"],
        },
    ]
    speech_scp, rir_scp, noise_scp, spk2enroll = _write_tse_sources(tmp_path, records)
    dataset = TSEDynamicMixingDataset(
        speech_scp=str(speech_scp), noise_scp=str(noise_scp),
        rir_scp=str(rir_scp), windnoise_scp=None,
        spk2enroll_json=str(spk2enroll),
        sampling_cfg=_balanced_sampling(),
        simulation_cfg={"prob_target_only": 1.0,
            "prob_reverberation": 0.0,
            "prob_wind_noise_sidechain": 0.0,
            "music_ratio": 0.1,
        },
        filter_by_duration=False,
    )
    assert set(dataset.noise_flt) == {"traffic"}
    assert set(dataset.music_flt) == {"jsonl_music_music"}
    assert dataset._noise_category_id_by_uid == {"traffic": 0}


def test_tse_balanced_noise_and_music_ratio_are_independent(tmp_path, monkeypatch):
    """TSE 启用 music_ratio 时，additive noise 仍走 category 均衡；music 不被均衡计数。"""
    from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset

    traffic = tmp_path / "traffic.wav"
    car = tmp_path / "car.wav"
    music = tmp_path / "music.wav"
    _write_wav(traffic, 0.1)
    _write_wav(car, 0.1)
    _write_wav(music, 0.1)
    records = [
        _category_noise_record("traffic", 16000, traffic, 0, "traffic", "set"),
        _category_noise_record("car", 16000, car, 1, "Car", "set"),
        _category_noise_record("music", 16000, music, 2, "Music", "set"),
    ]
    speech_scp, rir_scp, noise_scp, spk2enroll = _write_tse_sources(tmp_path, records)

    dataset = TSEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=None,
        spk2enroll_json=str(spk2enroll),
        music_scp=None,
        sampling_cfg=_balanced_sampling(),
        simulation_cfg={"prob_target_only": 1.0,
            "prob_reverberation": 0.0,
            "prob_wind_noise_sidechain": 0.0,
            "music_ratio": 0.5,
        },
        filter_by_duration=False,
    )
    assert dataset.balanced_noise_sampling_enabled is True

    captured_paths: list[str] = []
    _set_noise_reader(
        dataset,
        lambda path, *args, **kwargs: captured_paths.append(path)
        or np.ones((1, 1000), dtype=np.float32),
    )

    rng = np.random.default_rng(42)
    for _ in range(200):
        dataset._read_one_additive_noise_scp_only(16000, 1000, rng)

    noise_paths = [p for p in captured_paths if "music" not in Path(p).stem]
    music_paths = [p for p in captured_paths if "music" in Path(p).stem]
    assert music_paths, "music_ratio=0.5 应至少抽到 music"
    assert noise_paths, "additive noise 路径不应为空"
    assert any("traffic" in Path(p).stem for p in noise_paths), "应抽到 traffic 类别"
    assert any("car" in Path(p).stem for p in noise_paths), "应抽到 car 类别"

    pairs = [dataset._select_tse_dual_sources(16000, rng) for _ in range(300)]
    assert any(
        len(pair) == 2 and {item[0] for item in pair} == {"music", "ordinary"}
        for pair in pairs
    )
    assert any(len(pair) == 2 and all(item[0] == "ordinary" for item in pair) for pair in pairs)
    assert all(
        not (len(pair) == 2 and pair[0][0] == pair[1][0] == "music")
        for pair in pairs
    )
    for pair in pairs:
        ordinary = [item for item in pair if item[0] == "ordinary"]
        if len(ordinary) == 2:
            categories = [dataset._noise_category_id_by_uid[item[1]] for item in ordinary]
            assert categories[0] != categories[1]


# ---------------------------------------------------------------------------
# 训练读取与 Noise Filter strict v6 发布审计的边界
# ---------------------------------------------------------------------------

def test_canonical_mid_classes_cover_frozen_babble_and_hard_negative_labels():
    policy = _canonical_policy()
    assert len(policy.mid_to_class) == CLASSIFIER_LABEL_COUNT
    assert policy.class_of("/m/0261r1") == "speech"  # AudioSet Babbling
    assert policy.class_of("/m/07rkbfh") == "babble"  # Chatter
    assert policy.class_of("/m/07qfr4h") == "babble"  # Hubbub
    assert policy.class_of("/m/04rlf") == "noise"  # Music hard-negative
    assert policy.class_of("/m/025td0t") == "noise"  # Background music
    assert policy.class_of("/t/dd00127") == "noise"  # Inside public space


def test_noise_manifest_gate_allows_ordinary_scp_and_jsonl(tmp_path):
    """非均衡普通四列 SCP / JSONL 保持合法，不属于 strict v6 边界。"""
    wav = tmp_path / "noise.wav"
    _write_wav(wav, 0.1)
    speech_scp, rir_scp = _write_speech_rir(tmp_path)
    canonical_scp = tmp_path / "canonical.scp"
    _write_scp(canonical_scp, "noise", wav)
    canonical_jsonl = tmp_path / "canonical.jsonl"
    canonical_jsonl.write_text(
        json.dumps({"uid": "noise", "fs": 16000, "length": 16000, "path": str(wav)})
        + "\n",
        encoding="utf-8",
    )
    for noise_scp in (canonical_scp, canonical_jsonl):
        dataset = SEDynamicMixingDataset(
            speech_scp=str(speech_scp),
            noise_scp=str(noise_scp),
            rir_scp=str(rir_scp),
            windnoise_scp=None,
            simulation_cfg={
                "prob_reverberation": 0.0,
                "prob_wind_noise_sidechain": 0.0,
            },
            filter_by_duration=False,
        )
        assert set(dataset.noise_flt) == {"noise"}

    legacy_json = tmp_path / "legacy.json"
    legacy_json.write_text(
        json.dumps([{"uid": "noise", "fs": 16000, "length": 16000, "path": str(wav)}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"must use \.scp .* or \.jsonl"):
        SEDynamicMixingDataset(
            speech_scp=str(speech_scp),
            noise_scp=str(legacy_json),
            rir_scp=str(rir_scp),
            windnoise_scp=None,
            simulation_cfg={
                "prob_reverberation": 0.0,
                "prob_wind_noise_sidechain": 0.0,
            },
            filter_by_duration=False,
        )

def test_lightweight_jsonl_does_not_require_strict_v6(tmp_path):
    """JSONL 仅表示逐行 JSON；轻量分类字段即可驱动 natural category state。"""
    wav = tmp_path / "noise.wav"
    _write_wav(wav, 0.1)
    dataset = _make_dataset(
        tmp_path,
        [_legacy_noise_record("noise", 16000, wav, 0, "traffic", "")],
        sampling=None,
    )
    assert dataset._noise_category_id_by_uid == {"noise": 0}
    assert set(dataset._noise_manifest_metadata["noise"]) <= {
        "uid", "fs", "length", "path", "category_id",
        "balance_category", "category_name", "roles",
    }


def test_production_admission_consumes_schema_v6_top5_without_thresholds(tmp_path):
    wav = tmp_path / "rain.wav"
    _write_wav(wav, 0.1)
    record = _v6_noise_record("rain", 16000, wav, 0, "Rain", "set")
    manifest = tmp_path / "noise.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    parsed = read_noise_source_manifest(manifest, production_admission=True)

    assert parsed.parser_version == 10
    assert record["manifest_schema_version"] == NOISE_MANIFEST_SCHEMA_VERSION == 6
    assert len(record["verification"]["top5_hits"]) == 5
    assert all(
        set(hit) == {"index", "mid", "label", "kind", "prob"}
        for hit in record["verification"]["top5_hits"]
    )
    assert not ({"threshold_hits", "calibration_status"} & set(record["verification"]))
    assert not ({"admission_calibration_version", "attestation"} & set(record))


def test_training_noise_runtime_does_not_load_production_admission(tmp_path, monkeypatch):
    """动态训练只消费基础 schema/category，不加载 canonical admission policy。"""
    from core.data.noise_admission import NoiseAdmissionContext

    wav = tmp_path / "rain.wav"
    _write_wav(wav, 0.1)

    def unexpected_load(cls):
        raise AssertionError("training runtime must not load production admission")

    monkeypatch.setattr(NoiseAdmissionContext, "load", classmethod(unexpected_load))
    dataset = _make_dataset(
        tmp_path,
        [_v6_noise_record("rain", 16000, wav, 0, "Rain", "set")],
        sampling=None,
    )

    assert set(dataset.noise_flt) == {"rain"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.__setitem__("manifest_schema_version", 5),
        lambda record: record.__setitem__(
            "labels", {"audioset_scores": [0.0] * CLASSIFIER_LABEL_COUNT}
        ),
        lambda record: record.__setitem__("labels", {"audioset_top10": []}),
        lambda record: record.__setitem__("admission_calibration_version", 1),
        lambda record: record.__setitem__("strict_fields_complete", False),
    ],
    ids=["schema-v5", "full-527", "top-10", "calibration", "incomplete-marker"],
)
def test_training_noise_runtime_ignores_admission_contract_fields(
    tmp_path, mutate
):
    """训练 runtime 不重放旧 admission 或 strict-like metadata。"""
    wav = tmp_path / "rain.wav"
    _write_wav(wav, 0.1)
    record = _v6_noise_record("rain", 16000, wav, 0, "Rain", "set")
    mutate(record)

    dataset = _make_dataset(tmp_path, [record], sampling=None)

    assert set(dataset.noise_flt) == {"rain"}


def test_training_noise_runtime_accepts_mixed_metadata_contracts(tmp_path):
    """训练只要求基础 schema/category，允许 publication 与传统记录混合消费。"""
    strict_wav = tmp_path / "strict.wav"
    legacy_wav = tmp_path / "legacy.wav"
    _write_wav(strict_wav, 0.1)
    _write_wav(legacy_wav, 0.1)
    records = [
        _v6_noise_record("strict", 16000, strict_wav, 0, "Rain", "set"),
        _legacy_noise_record("legacy", 16000, legacy_wav, 1, "traffic", "set"),
    ]

    dataset = _make_dataset(tmp_path, records, sampling=None)

    assert set(dataset.noise_flt) == {"strict", "legacy"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.__setitem__("manifest_schema_version", 5),
        lambda record: record.__setitem__("admission_calibration_version", 1),
        lambda record: record["verification"].__setitem__("threshold_hits", []),
    ],
    ids=["schema-v5", "calibration", "thresholds"],
)
def test_production_admission_rejects_legacy_contract_fields(tmp_path, mutate):
    wav = tmp_path / "rain.wav"
    _write_wav(wav, 0.1)
    record = _v6_noise_record("rain", 16000, wav, 0, "Rain", "set")
    mutate(record)
    manifest = tmp_path / "noise.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_noise_source_manifest(manifest, production_admission=True)


def test_jsonl_metadata_supports_balancing_and_json_array_is_rejected(tmp_path):
    """JSONL 保留分类字段；JSON array source manifest 必须 fail-closed。"""
    wav = tmp_path / "noise.wav"
    _write_wav(wav, 0.1)
    speech_scp, rir_scp = _write_speech_rir(tmp_path)
    record = _legacy_noise_record("noise", 16000, wav, 0, "traffic", "")
    manifest = tmp_path / "noise.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = SEDynamicMixingDataset(
        speech_scp=str(speech_scp), noise_scp=str(manifest),
        rir_scp=str(rir_scp), windnoise_scp=None,
        sampling_cfg=_balanced_sampling(),
        simulation_cfg={"prob_reverberation": 0.0, "prob_wind_noise_sidechain": 0.0},
        filter_by_duration=False,
    )
    assert dataset._noise_category_id_by_uid == {"noise": 0}

    legacy_json = tmp_path / "noise.json"
    legacy_json.write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(ValueError, match=r"must use \.scp .* or \.jsonl"):
        SEDynamicMixingDataset(
            speech_scp=str(speech_scp), noise_scp=str(legacy_json),
            rir_scp=str(rir_scp), windnoise_scp=None,
            sampling_cfg=_balanced_sampling(),
            simulation_cfg={"prob_reverberation": 0.0, "prob_wind_noise_sidechain": 0.0},
            filter_by_duration=False,
        )


def test_removed_escape_hatch_constructor_argument_is_rejected(tmp_path):
    wav = tmp_path / "noise.wav"
    _write_wav(wav, 0.1)
    speech_scp, rir_scp = _write_speech_rir(tmp_path)
    with pytest.raises(TypeError, match="allow_unverified_noise_manifest"):
        SEDynamicMixingDataset(
            speech_scp=str(speech_scp), noise_scp=str(tmp_path / "missing.scp"),
            rir_scp=str(rir_scp), windnoise_scp=None,
            allow_unverified_noise_manifest=True,
        )


def test_balance_category_drives_category_index_and_summary(tmp_path):
    """Babble 记录按 Noise category 参与索引和 summary，不建立 Babbling bucket。"""
    wav = tmp_path / "noise.wav"
    babble_wav = tmp_path / "babble.wav"
    _write_wav(wav, 0.1)
    _write_wav(babble_wav, 0.1)
    records = [
        _category_noise_record("traffic", 16000, wav, 0, "traffic", "set"),
        _category_noise_record(
            "babble", 16000, babble_wav, 900, "Noise", "set", c1_branch="babble"
        ),
    ]
    dataset = _make_dataset(tmp_path, records, sampling=_balanced_sampling())
    cats = {name: count for name, count, _ in (dataset._compute_noise_categories() or [])}
    assert cats == {"traffic": 1, "Noise": 1}
    index = dataset._noise_category_uids_by_fs[16000]
    assert index[900] == ("babble",)
    assert set(dataset._noise_category_id_by_uid) == {"traffic", "babble"}
    assert dataset._noise_category_id_by_uid["babble"] == 900
    assert "Babbling" not in cats


def test_babble_joins_temperature_balancing_as_noise_category(tmp_path):
    """Babble 使用 Noise bucket 参与 temperature 均衡。"""
    babble_wav = tmp_path / "babble.wav"
    _write_wav(babble_wav, 0.1)
    records = [
        _category_noise_record(
            "babble-1", 16000, babble_wav, 900, "Noise", "set", c1_branch="babble"
        ),
    ]
    records += _write_category_records(tmp_path, "vehicle", 200, "Vehicle", 3)
    dataset = _make_dataset(tmp_path, records, sampling=_temperature_sampling())
    assert 900 in dataset._noise_category_weight_by_id
    picks = [
        dataset._select_balanced_additive_noise_uid(16000, np.random.default_rng(seed))
        for seed in range(300)
    ]
    assert "babble-1" in picks


def test_tse_music_uid_absent_from_balanced_sampling_state(tmp_path):
    """TSE 抽取内嵌 music 后重建统一均衡状态：music UID 同时从 category
    index、uid->category 轻量表与 temperature 权重中消失。"""
    from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset

    records = _noise_records_with_embedded(music=True, tmp_path=tmp_path)
    speech_scp, rir_scp, noise_scp, spk2enroll = _write_tse_sources(tmp_path, records)
    dataset = TSEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(noise_scp),
        rir_scp=str(rir_scp),
        windnoise_scp=None,
        spk2enroll_json=str(spk2enroll),
        music_scp=None,
        sampling_cfg=_balanced_sampling(),
        simulation_cfg={"prob_target_only": 1.0,
            "prob_reverberation": 0.0,
            "prob_wind_noise_sidechain": 0.0,
            "music_ratio": 0.1,
        },
        filter_by_duration=False,
    )
    index = dataset._noise_category_uids_by_fs[16000]
    indexed_uids = [uid for uids in index.values() for uid in uids]
    assert indexed_uids == ["traffic"], f"index 应只剩普通噪声，got {indexed_uids}"
    assert "music" not in dataset._noise_category_id_by_uid
    assert "wind-a" not in dataset._noise_category_id_by_uid
    # 普通噪声池与 metadata 同样不含 music（抽取已剔除）
    assert set(dataset.noise_flt) == {"traffic"}
    assert set(dataset.music_flt) == {"jsonl_music_music"}


# ---------------------------------------------------------------------------
# temperature 温度采样（balance_mode="temperature"）：
# p(category) ∝ n(category)^alpha × difficulty(category)
# - n：该 category 的 UID 数（数据量）
# - difficulty：由 (统计×时间) 组合查 DEFAULT_DIFFICULTY（领域先验 1..5）
# 无桶预算层；category 级连续权重 + 加权抽样。
# 测试用 label 来自 core/data/noise_attributes.py 真实属性表：
#   Vehicle -> (statistics=quasi_stationary, temporal=continuous)  difficulty 2.0
#   Thunder -> (impulsive, transient)                              difficulty 5.0
#   Bird    -> (non_stationary, intermittent)                      difficulty 3.5
#   Pulse   -> (stationary, periodic)                              difficulty 2.0
# ---------------------------------------------------------------------------

def _temperature_sampling(**kw) -> SamplingConfig:
    cfg = {"balance_categories": True, "balance_mode": "temperature"}
    cfg.update(kw)
    return SamplingConfig.from_dict({"noise": cfg})


def _write_category_records(tmp_path: Path, category: str, cat_id: int, label: str, n: int):
    """为属性表内 label 写 n 条 audio 记录，返回 records 列表。"""
    records = []
    for i in range(n):
        path = tmp_path / category / f"{category}-{i}.wav"
        _write_wav(path, 0.1)
        records.append(_category_noise_record(f"{category}-{i}", 16000, path, cat_id, label, "set"))
    return records


def test_temperature_weight_formula_is_tier_local(tmp_path):
    """构建期仅存 difficulty；runtime 按首个 tier 的剩余 UID 数计算最终权重。"""
    records = _write_category_records(tmp_path, "vehicle", 200, "Vehicle", 10)
    dataset = _make_dataset(tmp_path, records, sampling=_temperature_sampling())
    assert dataset._noise_category_weight_by_id[200] == 2.0
    tier, candidates, weights = dataset._temperature_candidates_and_weights(16000)
    assert tier == "exact"
    assert len(candidates[200]) == 10
    assert abs(weights[200] - (10 ** 0.5) * 2.0) < 1e-9
    assert abs(dataset._noise_sampling_config.alpha - 0.5) < 1e-9


def test_temperature_exact_tier_ignores_same_category_other_sr_counts(tmp_path):
    """exact 权重只计算 exact UID，不能混入同类别其他 SR 的数量。"""
    records = []
    specs = [("a-exact", 16000, 200, "Vehicle")]
    specs += [(f"a-high-{i}", 48000, 200, "Vehicle") for i in range(9)]
    specs += [(f"b-exact-{i}", 16000, 300, "Thunder") for i in range(4)]
    for uid, fs, category_id, label in specs:
        path = tmp_path / f"{uid}.wav"
        _write_wav(path, 0.1)
        records.append(_category_noise_record(uid, fs, path, category_id, label, "set"))
    dataset = _make_dataset(tmp_path, records, sampling=_temperature_sampling())

    tier, candidates, weights = dataset._temperature_candidates_and_weights(16000)

    assert tier == "exact"
    assert candidates == {200: ("a-exact",), 300: tuple(f"b-exact-{i}" for i in range(4))}
    assert weights == {200: 2.0, 300: 10.0}


def test_temperature_higher_tier_aggregates_category_across_sample_rates(tmp_path):
    """higher 聚合全部更高 SR；同一 category 只出现一次并合计 eligible UID。"""
    records = []
    specs = [
        ("vehicle-32k", 32000, 200, "Vehicle"),
        ("vehicle-48k-a", 48000, 200, "Vehicle"),
        ("vehicle-48k-b", 48000, 200, "Vehicle"),
        ("thunder-48k", 48000, 300, "Thunder"),
    ]
    for uid, fs, category_id, label in specs:
        path = tmp_path / f"{uid}.wav"
        _write_wav(path, 0.1)
        records.append(_category_noise_record(uid, fs, path, category_id, label, "set"))
    dataset = _make_dataset(tmp_path, records, sampling=_temperature_sampling())

    tier, candidates, weights = dataset._temperature_candidates_and_weights(24000)

    assert tier == "higher"
    assert candidates[200] == ("vehicle-32k", "vehicle-48k-a", "vehicle-48k-b")
    assert candidates[300] == ("thunder-48k",)
    assert weights[200] == pytest.approx((3 ** 0.5) * 2.0)
    assert weights[300] == 5.0


def test_temperature_all_fallback_uses_total_eligible_uid_counts(tmp_path):
    """无 exact/higher 时 all 聚合全部 SR，并按总 eligible UID 数计算权重。"""
    records = []
    specs = [
        ("vehicle-8k", 8000, 200, "Vehicle"),
        ("vehicle-16k", 16000, 200, "Vehicle"),
        ("thunder-8k", 8000, 300, "Thunder"),
    ]
    for uid, fs, category_id, label in specs:
        path = tmp_path / f"{uid}.wav"
        _write_wav(path, 0.1)
        records.append(_category_noise_record(uid, fs, path, category_id, label, "set"))
    dataset = _make_dataset(tmp_path, records, sampling=_temperature_sampling())

    tier, candidates, weights = dataset._temperature_candidates_and_weights(96000)

    assert tier == "all"
    assert candidates == {
        200: ("vehicle-16k", "vehicle-8k"),
        300: ("thunder-8k",),
    }
    assert weights[200] == pytest.approx((2 ** 0.5) * 2.0)
    assert weights[300] == 5.0


def test_temperature_exclude_uid_recomputes_tier_local_weight(tmp_path):
    """exclude_uid 后候选数与最终 weight 必须按剩余 UID 重算。"""
    records = _write_category_records(tmp_path, "vehicle", 200, "Vehicle", 3)
    records += _write_category_records(tmp_path, "thunder", 300, "Thunder", 1)
    dataset = _make_dataset(tmp_path, records, sampling=_temperature_sampling())

    tier, candidates, weights = dataset._temperature_candidates_and_weights(
        16000, exclude_uid="vehicle-0"
    )

    assert tier == "exact"
    assert candidates[200] == ("vehicle-1", "vehicle-2")
    assert weights[200] == pytest.approx((2 ** 0.5) * 2.0)
    assert weights[300] == 5.0


def test_temperature_selection_respects_difficulty(tmp_path):
    """同数据量下难度提权：Vehicle(d=2) vs Thunder(d=5) → Thunder 占 5/7 ≈ 71%。"""
    records = _write_category_records(tmp_path, "vehicle", 200, "Vehicle", 10)
    records += _write_category_records(tmp_path, "thunder", 300, "Thunder", 10)
    dataset = _make_dataset(tmp_path, records, sampling=_temperature_sampling())
    picks = [
        dataset._select_balanced_additive_noise_uid(16000, np.random.default_rng(seed))
        for seed in range(1200)
    ]
    stems = [Path(p).name for p in picks if p]
    thunder_share = sum(1 for s in stems if s.startswith("thunder")) / len(stems)
    assert 0.60 <= thunder_share <= 0.80, f"thunder share={thunder_share:.3f}"


def test_temperature_alpha_controls_tier_local_weights(tmp_path):
    """α 只作用于首个候选 tier 的聚合 UID 数。"""
    for alpha, ratio in ((0.5, 10.0), (1.0, 100.0)):
        records = _write_category_records(tmp_path / f"a{alpha}", "big", 200, "Vehicle", 100)
        records += _write_category_records(tmp_path / f"a{alpha}_s", "rare", 201, "Bird", 1)
        ds = _make_dataset(
            tmp_path / f"d{alpha}", records, sampling=_temperature_sampling(alpha=alpha)
        )
        _, _, weights = ds._temperature_candidates_and_weights(16000)
        got = (weights[200] / 2.0) / (weights[201] / 3.5)
        assert abs(got - ratio) < 1e-6


def test_temperature_rare_category_still_sampled(tmp_path):
    """稀有标签（n 很小）仍可被抽到（p>0），但被 n^alpha 平滑不主导。"""
    records = _write_category_records(tmp_path, "veh", 200, "Vehicle", 200)
    records += _write_category_records(tmp_path, "pulse", 500, "Pulse", 1)
    dataset = _make_dataset(tmp_path, records, sampling=_temperature_sampling())
    picks = [
        dataset._select_balanced_additive_noise_uid(16000, np.random.default_rng(seed))
        for seed in range(2000)
    ]
    stems = [Path(p).name for p in picks if p]
    pulse = sum(1 for s in stems if s.startswith("pulse"))
    vehicle = sum(1 for s in stems if s.startswith("veh"))
    assert pulse > 0, "稀有类别不应被完全饿死"
    assert vehicle > pulse * 5, "大类仍应占主导（被 n^0.5 平滑但未消失）"


def test_temperature_exclude_category_id(tmp_path):
    """temperature 模式 exclude_category_id 排除对应细类（双噪声异类语义）。"""
    records = _write_category_records(tmp_path, "vehicle", 200, "Vehicle", 3)
    records += _write_category_records(tmp_path, "thunder", 300, "Thunder", 3)
    dataset = _make_dataset(tmp_path, records, sampling=_temperature_sampling())
    for seed in range(50):
        uid = dataset._select_balanced_additive_noise_uid(
            16000, np.random.default_rng(seed), exclude_category_id=200
        )
        assert uid is not None
        assert not Path(uid).name.startswith("vehicle")


def test_temperature_missing_label_fails(tmp_path):
    """temperature 模式要求 category 收录于属性表，未收录 fail-fast。"""
    wav = tmp_path / "noise.wav"
    _write_wav(wav, 0.1)
    record = _category_noise_record("custom", 16000, wav, 0, "Custom noise", "set")
    with pytest.raises(ValueError, match="noise attribute table missing"):
        _make_dataset(tmp_path, [record], sampling=_temperature_sampling())


def test_temperature_sampling_config_validation(tmp_path):
    """temperature 配置非法时报错；旧的桶级配置项被拒绝。"""
    with pytest.raises(ValueError, match="balance_mode"):
        _temperature_sampling(balance_mode="attribute")
    with pytest.raises(ValueError, match="alpha"):
        _temperature_sampling(alpha=0)
    with pytest.raises(ValueError, match="alpha"):
        _temperature_sampling(alpha=1.5)
    with pytest.raises(ValueError, match="unknown"):
        _temperature_sampling(bucket_dims=["statistics", "temporal"])
    with pytest.raises(ValueError, match="unknown"):
        _temperature_sampling(within_dims=["physical"])


def test_temperature_missing_difficulty_fails(tmp_path, monkeypatch):
    """(统计×时间) 组合未收录于 DEFAULT_DIFFICULTY 时 fail-fast，不静默默认 1.0。"""
    import core.data.dynamic_dataset_base as ddb
    from core.data.sampling import DEFAULT_DIFFICULTY

    trimmed = dict(DEFAULT_DIFFICULTY)
    trimmed.pop("quasi_stationary:continuous")  # Vehicle 所属组合
    monkeypatch.setattr(ddb, "DEFAULT_DIFFICULTY", trimmed)
    records = _write_category_records(tmp_path, "vehicle", 200, "Vehicle", 3)
    with pytest.raises(ValueError, match="DEFAULT_DIFFICULTY missing"):
        _make_dataset(tmp_path, records, sampling=_temperature_sampling())


def test_temperature_category_id_name_mismatch_fails(tmp_path):
    """同一 category_id 映射多个 category_name 时 fail-fast（数据异常）。"""
    records = _write_category_records(tmp_path, "vehicle", 200, "Vehicle", 2)
    records += _write_category_records(tmp_path, "engine", 200, "Engine", 2)  # 同 ID 异名
    with pytest.raises(ValueError, match="maps to multiple category names"):
        _make_dataset(tmp_path, records, sampling=_temperature_sampling())


def test_temperature_dual_blend_survives_worker_metadata_strip(tmp_path):
    """回归：真实 __getstate__ 后 worker 剥离 _noise_manifest_metadata，dual-blend
    经轻量 _noise_category_id_by_uid 表取 first noise 的 category（避免再次 KeyError）。"""
    records = _write_category_records(tmp_path, "vehicle", 200, "Vehicle", 3)
    records += _write_category_records(tmp_path, "thunder", 300, "Thunder", 3)
    ds = _make_dataset(tmp_path, records, sampling=_temperature_sampling())
    # 走真实序列化路径（对应 DataLoader worker 的 __getstate__）
    state = ds.__getstate__()
    assert state["_noise_manifest_metadata"] == {}  # temperature worker 已剥离 metadata
    assert "_noise_category_id_by_uid" in state  # 轻量表必须被保留
    worker = SEDynamicMixingDataset.__new__(SEDynamicMixingDataset)
    worker.__dict__.update(state)
    out = worker._read_dual_additive_noise_scp_blended(16000, 4000, np.random.default_rng(0))
    assert out is not None and out.size > 0
    assert {200, 300} <= set(worker._noise_category_id_by_uid.values())


def test_category_mode_dual_blend_uses_uid_map(tmp_path):
    """balance_mode=category 同样经轻量 uid->category 表取 first category（等价路径）。"""
    records = _write_category_records(tmp_path, "vehicle", 200, "Vehicle", 3)
    records += _write_category_records(tmp_path, "thunder", 300, "Thunder", 3)
    ds = _make_dataset(
        tmp_path, records,
        sampling=SamplingConfig.from_dict({"noise": {"balance_categories": True}}),
    )
    assert ds._noise_sampling_config.balance_mode == "category"
    assert {200, 300} <= set(ds._noise_category_id_by_uid.values())
    out = ds._read_dual_additive_noise_scp_blended(16000, 4000, np.random.default_rng(0))
    assert out is not None and out.size > 0


def _multi_sr_selector_records(tmp_path: Path) -> list[dict]:
    """构造可覆盖 exact/higher/all 的稳定多 SR category fixture。"""
    specs = [
        ("vehicle-8k", 8000, 200, "Vehicle"),
        ("thunder-16k", 16000, 300, "Thunder"),
        ("bird-48k", 48000, 400, "Bird"),
        ("vehicle-48k", 48000, 200, "Vehicle"),
    ]
    records = []
    for uid, fs, category_id, label in specs:
        path = tmp_path / f"{uid}.wav"
        _write_wav(path, 0.1, sr=fs)
        records.append(_category_noise_record(uid, fs, path, category_id, label, "set"))
    return records


@pytest.mark.parametrize(
    "sampling",
    [_balanced_sampling(), _temperature_sampling()],
    ids=["category", "temperature"],
)
def test_selector_warm_and_cold_cache_keep_fixed_seed_sequence(tmp_path, sampling):
    """候选/权重 cache 冷热不改变 RNG 调用次数及 fixed-seed UID 序列。"""
    dataset = _make_dataset(
        tmp_path,
        _multi_sr_selector_records(tmp_path),
        sampling=sampling,
    )
    cold_rng = np.random.default_rng(20260822)
    cold = [
        dataset._select_balanced_additive_noise_uid(24000, cold_rng)
        for _ in range(64)
    ]

    # 第一轮已经将 higher view 与 selector 派生项预热；同 seed 热路径必须逐项一致。
    assert dataset._noise_category_tier_candidates_cache
    assert dataset._noise_category_order_cache
    if sampling.noise.balance_mode == "temperature":
        assert dataset._noise_temperature_base_weights_cache
    warm_rng = np.random.default_rng(20260822)
    warm = [
        dataset._select_balanced_additive_noise_uid(24000, warm_rng)
        for _ in range(64)
    ]
    assert warm == cold


def test_category_tier_cache_reuses_exact_and_enforces_dual_budget(tmp_path, monkeypatch):
    """exact 复用索引；higher/all LRU 同时约束 entry 与 UID-ref 总量。"""
    import core.data.dynamic_dataset_base as ddb

    monkeypatch.setattr(ddb, "_NOISE_SELECTOR_CACHE_MAX_ENTRIES", 2)
    monkeypatch.setattr(ddb, "_NOISE_SELECTOR_CACHE_MAX_UID_REFS", 4)
    dataset = _make_dataset(
        tmp_path,
        _multi_sr_selector_records(tmp_path),
        sampling=_temperature_sampling(),
    )
    tier, exact = dataset._noise_category_candidates_for_tier(16000)
    assert tier == "exact"
    assert exact is dataset._noise_category_uids_by_fs[16000]
    assert dataset._noise_category_tier_candidates_cache == {}

    # higher(24000) 含 2 UID，可缓存并在命中时复用同一对象。
    higher_tier, higher = dataset._noise_category_candidates_for_tier(24000)
    again_tier, again = dataset._noise_category_candidates_for_tier(24000)
    assert (higher_tier, again_tier) == ("higher", "higher")
    assert again is higher
    assert dataset._noise_category_tier_candidates_cache_uid_refs == 2

    # all(96000) 含 4 UID；为满足 UID-ref=4 预算，淘汰较旧 higher entry。
    all_tier, all_candidates = dataset._noise_category_candidates_for_tier(96000)
    assert all_tier == "all"
    assert all_candidates
    assert len(dataset._noise_category_tier_candidates_cache) == 1
    assert dataset._noise_category_tier_candidates_cache_uid_refs == 4
    assert sum(
        len(uids)
        for entry in dataset._noise_category_tier_candidates_cache.values()
        for uids in entry.values()
    ) == 4


def test_category_oversized_single_entry_is_not_cached(tmp_path, monkeypatch):
    """higher/all 单 entry 超 UID-ref 预算时返回候选但不进入 LRU。"""
    import core.data.dynamic_dataset_base as ddb

    monkeypatch.setattr(ddb, "_NOISE_SELECTOR_CACHE_MAX_ENTRIES", 4)
    monkeypatch.setattr(ddb, "_NOISE_SELECTOR_CACHE_MAX_UID_REFS", 1)
    dataset = _make_dataset(
        tmp_path,
        _multi_sr_selector_records(tmp_path),
        sampling=_temperature_sampling(),
    )
    tier, candidates = dataset._noise_category_candidates_for_tier(24000)
    assert tier == "higher"
    assert sum(len(uids) for uids in candidates.values()) == 2
    assert dataset._noise_category_tier_candidates_cache == {}
    assert dataset._noise_category_tier_candidates_cache_uid_refs == 0


def test_natural_lru_keeps_order_and_warm_cold_fixed_seed(tmp_path):
    """natural 常见池缓存 manifest-order tuple，冷热路径 fixed-seed 序列一致。"""
    specs = [
        ("first-noise", 0, "Vehicle"),
        ("manifest-first", 20, "Thunder"),
        ("manifest-second", 10, "Bird"),
    ]
    records = []
    for uid, category_id, label in specs:
        path = tmp_path / f"{uid}.wav"
        _write_wav(path, 0.1)
        records.append(_category_noise_record(uid, 16000, path, category_id, label, "set"))
    dataset = _make_dataset(tmp_path, records, sampling=None)

    cold_rng = np.random.default_rng(20260822)
    cold = [
        dataset._select_natural_different_category_uid(
            16000, cold_rng, exclude_category_id=0, exclude_uid="first-noise"
        )
        for _ in range(64)
    ]
    cache_key = ("exact", (16000,), 0, None)
    assert dataset._noise_natural_tier_uids_cache[cache_key] == (
        "manifest-first",
        "manifest-second",
    )
    assert dataset._noise_natural_tier_uids_cache_uid_refs == 2

    warm_rng = np.random.default_rng(20260822)
    warm = [
        dataset._select_natural_different_category_uid(
            16000, warm_rng, exclude_category_id=0, exclude_uid="first-noise"
        )
        for _ in range(64)
    ]
    assert cold == warm


def test_natural_lru_enforces_entry_and_uid_ref_budgets(tmp_path, monkeypatch):
    """natural LRU 淘汰旧 entry，且保留 UID 引用总数不超过预算。"""
    import core.data.dynamic_dataset_base as ddb

    monkeypatch.setattr(ddb, "_NOISE_SELECTOR_CACHE_MAX_ENTRIES", 3)
    monkeypatch.setattr(ddb, "_NOISE_SELECTOR_CACHE_MAX_UID_REFS", 20)
    records = []
    for category_id in range(6):
        for uid_index in range(2):
            uid = f"noise-{category_id}-{uid_index}"
            path = tmp_path / f"{uid}.wav"
            _write_wav(path, 0.1)
            records.append(
                _category_noise_record(uid, 16000, path, category_id, f"category-{category_id}", "set")
            )
    dataset = _make_dataset(tmp_path, records, sampling=None)

    first_key = ("exact", (16000,), 0, None)
    second_key = ("exact", (16000,), 1, None)
    for category_id in range(6):
        if category_id == 2:
            # 命中 category 0，把它提升为 MRU；下一次淘汰应先移除 category 1。
            dataset._select_natural_different_category_uid(
                16000,
                np.random.default_rng(99),
                exclude_category_id=0,
                exclude_uid="noise-0-1",
            )
        if category_id == 3:
            assert first_key in dataset._noise_natural_tier_uids_cache
            assert second_key not in dataset._noise_natural_tier_uids_cache
        uid = dataset._select_natural_different_category_uid(
            16000,
            np.random.default_rng(category_id),
            exclude_category_id=category_id,
            exclude_uid=f"noise-{category_id}-0",
        )
        assert uid is not None
        assert dataset._noise_category_id_by_uid[uid] != category_id
    assert len(dataset._noise_natural_tier_uids_cache) <= 3
    assert first_key not in dataset._noise_natural_tier_uids_cache
    assert dataset._noise_natural_tier_uids_cache_uid_refs <= 20
    assert dataset._noise_natural_tier_uids_cache_uid_refs == sum(
        len(entry) for entry in dataset._noise_natural_tier_uids_cache.values()
    )


def test_natural_oversized_single_entry_is_not_cached(tmp_path, monkeypatch):
    """单候选 tuple 超 UID-ref 预算时不缓存，安全回退零复制 view。"""
    import core.data.dynamic_dataset_base as ddb

    monkeypatch.setattr(ddb, "_NOISE_SELECTOR_CACHE_MAX_ENTRIES", 4)
    monkeypatch.setattr(ddb, "_NOISE_SELECTOR_CACHE_MAX_UID_REFS", 3)
    records = []
    for uid_index in range(6):
        category_id = 0 if uid_index == 0 else 1
        uid = f"noise-{uid_index}"
        path = tmp_path / f"{uid}.wav"
        _write_wav(path, 0.1)
        records.append(
            _category_noise_record(uid, 16000, path, category_id, f"category-{category_id}", "set")
        )
    dataset = _make_dataset(tmp_path, records, sampling=None)

    view = dataset._noise_natural_tier_view(
        "exact", (16000,), exclude_category_id=0, exclude_uid="noise-0"
    )
    assert isinstance(view, ddb._NaturalTierExcludedView)
    assert len(view) == 5
    assert dataset._noise_natural_tier_uids_cache == {}
    assert dataset._noise_natural_tier_uids_cache_uid_refs == 0
    cold = dataset._select_natural_different_category_uid(
        16000, np.random.default_rng(77), exclude_category_id=0, exclude_uid="noise-0"
    )
    warm = dataset._select_natural_different_category_uid(
        16000, np.random.default_rng(77), exclude_category_id=0, exclude_uid="noise-0"
    )
    assert warm == cold
    assert dataset._noise_natural_tier_uids_cache == {}


def test_reverse_category_mapping_conflict_fails_balanced_and_disables_natural(tmp_path):
    """同 category name 对应多个 ID：balanced fail-fast，natural 整体回退。"""
    records = []
    for uid, category_id in (("vehicle-a", 200), ("vehicle-b", 201)):
        path = tmp_path / f"{uid}.wav"
        _write_wav(path, 0.1)
        records.append(_category_noise_record(uid, 16000, path, category_id, "Vehicle", "set"))

    with pytest.raises(ValueError, match=r"category name 'Vehicle'.*200 vs 201"):
        _make_dataset(tmp_path / "balanced", records, sampling=_balanced_sampling())

    natural = _make_dataset(tmp_path / "natural", records, sampling=None)
    assert natural._noise_category_uids_by_fs == {}
    assert natural._noise_category_id_by_uid == {}
    assert natural._noise_category_name_by_id == {}
    assert natural._noise_category_id_by_name == {}
    assert natural._ordinary_noise_categories_complete() is False


def test_balanced_category_exclusion_uses_zero_copy_rank_select_view(monkeypatch):
    """排除 category 后不重建剩余 category tuple，并保持 sequence 契约。"""
    import builtins
    import core.data.dynamic_dataset_base as ddb

    dataset = SEDynamicMixingDataset.__new__(SEDynamicMixingDataset)
    dataset._noise_category_order_cache = {}
    dataset._noise_category_uids_by_fs = {
        16000: {
            10: ("a",),
            20: ("b",),
            30: ("c",),
            40: ("d",),
        }
    }
    candidates = dataset._noise_category_uids_by_fs[16000]

    # 先建立未过滤 canonical category order；审查项只统计 exclusion 热路径。
    assert dataset._ordered_noise_categories(
        16000, "exact", candidates, cacheable=True
    ) == (10, 20, 30, 40)

    class CountingTupleMeta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, builtins.tuple)

    class CountingTuple(builtins.tuple, metaclass=CountingTupleMeta):
        copied_items = 0

        def __new__(cls, iterable=()):
            value = builtins.tuple(iterable)
            cls.copied_items += len(value)
            return builtins.tuple.__new__(cls, value)

    monkeypatch.setattr(
        dataset,
        "_noise_category_view_key",
        lambda _sr, _tier: ("exact", (16000,)),
    )
    monkeypatch.setitem(
        BaseDynamicMixingDataset._ordered_noise_categories.__globals__,
        "tuple",
        CountingTuple,
    )
    filtered = dataset._filter_noise_category_view(
        candidates,
        exclude_category_id=20,
        exclude_uid="b",
    )
    categories = dataset._ordered_noise_categories(
        16000, "exact", filtered, cacheable=False
    )

    assert CountingTuple.copied_items == 0
    assert len(categories) == 3
    assert list(categories) == [10, 30, 40]
    assert [categories[index] for index in range(len(categories))] == [10, 30, 40]
    assert categories[-1] == 40
    with pytest.raises(IndexError):
        _ = categories[3]
    with pytest.raises(IndexError):
        _ = categories[-4]


@pytest.mark.parametrize(
    "sampling",
    [_balanced_sampling(), _temperature_sampling()],
    ids=["category", "temperature"],
)
def test_balanced_category_exclusion_rank_select_keeps_fixed_seed_sequence(
    tmp_path, sampling
):
    """rank-select 与 legacy dict 过滤在 order 与 fixed-seed 选择上逐项一致。"""
    dataset = _make_dataset(
        tmp_path,
        _multi_sr_selector_records(tmp_path),
        sampling=sampling,
    )
    tier, base = dataset._noise_category_candidates_for_tier(96000)
    assert tier == "all"
    expected_candidates = {
        category_id: uids
        for category_id, uids in base.items()
        if category_id != 200
    }
    expected_categories = tuple(expected_candidates)

    for seed in range(64):
        expected_rng = np.random.default_rng(seed)
        if sampling.noise.balance_mode == "temperature":
            alpha = sampling.noise.alpha
            weights = {
                category_id: (len(uids) ** alpha)
                * dataset._noise_category_weight_by_id[category_id]
                for category_id, uids in expected_candidates.items()
            }
            expected_category = expected_categories[
                int(
                    expected_rng.choice(
                        len(expected_categories),
                        p=np.asarray(
                            [weights[item] for item in expected_categories],
                            dtype=np.float64,
                        )
                        / sum(weights.values()),
                    )
                )
            ]
        else:
            expected_category = expected_categories[
                int(expected_rng.integers(0, len(expected_categories)))
            ]
        expected_uid = expected_candidates[expected_category][
            int(expected_rng.integers(0, len(expected_candidates[expected_category])))
        ]

        actual = dataset._select_balanced_additive_noise_uid(
            96000,
            np.random.default_rng(seed),
            exclude_category_id=200,
            exclude_uid="vehicle-8k",
        )
        assert actual == expected_uid


def test_dual_selector_warm_cache_keeps_heterogeneous_fixed_seed_pair(tmp_path):
    """双噪声第二段按 category 排除，冷热 cache 生成同一异类 fixed-seed 对。"""
    records = _write_category_records(tmp_path, "vehicle", 200, "Vehicle", 3)
    records += _write_category_records(tmp_path, "thunder", 300, "Thunder", 3)
    dataset = _make_dataset(tmp_path, records, sampling=_temperature_sampling())

    def select_pair(seed: int) -> tuple[str, str]:
        rng = np.random.default_rng(seed)
        first = dataset._select_balanced_additive_noise_uid(16000, rng)
        assert first is not None
        second = dataset._select_balanced_additive_noise_uid(
            16000,
            rng,
            exclude_category_id=dataset._noise_category_id_by_uid[first],
            exclude_uid=first,
        )
        assert second is not None
        return first, second

    cold = select_pair(314159)
    warm = select_pair(314159)
    assert warm == cold
    assert dataset._noise_category_id_by_uid[cold[0]] != dataset._noise_category_id_by_uid[cold[1]]


def test_noise_state_rebuild_and_getstate_clear_derived_caches_atomically(tmp_path):
    """成功重建及 worker 序列化清 cache；失败重建不覆盖旧分类状态。"""
    dataset = _make_dataset(
        tmp_path,
        _multi_sr_selector_records(tmp_path),
        sampling=_temperature_sampling(),
    )
    dataset._select_balanced_additive_noise_uid(24000, np.random.default_rng(7))
    dataset._select_natural_different_category_uid(
        16000,
        np.random.default_rng(8),
        exclude_category_id=300,
        exclude_uid="thunder-16k",
    )
    assert dataset._ordinary_noise_categories_complete() is True
    assert dataset._noise_category_tier_candidates_cache
    assert dataset._noise_natural_tier_uids_cache
    assert dataset._noise_category_tier_candidates_cache_uid_refs > 0
    assert dataset._noise_natural_tier_uids_cache_uid_refs > 0

    dataset._build_noise_sampling_state()
    derived_cache_names = (
        "_noise_category_tier_candidates_cache",
        "_noise_category_order_cache",
        "_noise_temperature_base_weights_cache",
        "_noise_natural_tier_uids_cache",
    )
    derived_cache_ref_names = (
        "_noise_category_tier_candidates_cache_uid_refs",
        "_noise_natural_tier_uids_cache_uid_refs",
    )
    assert all(getattr(dataset, name) == {} for name in derived_cache_names)
    assert all(getattr(dataset, name) == 0 for name in derived_cache_ref_names)

    dataset._select_balanced_additive_noise_uid(24000, np.random.default_rng(9))
    old_index = dataset._noise_category_uids_by_fs
    old_cache = dataset._noise_category_tier_candidates_cache
    dataset._noise_manifest_metadata["bird-48k"]["balance_category"] = "Missing label"
    dataset._noise_manifest_metadata["bird-48k"]["category_name"] = "Missing label"
    with pytest.raises(ValueError, match="noise attribute table missing"):
        dataset._build_noise_sampling_state()
    assert dataset._noise_category_uids_by_fs is old_index
    assert dataset._noise_category_tier_candidates_cache is old_cache

    state = dataset.__getstate__()
    assert all(state[name] == {} for name in derived_cache_names)
    assert all(state[name] == 0 for name in derived_cache_ref_names)
    assert dataset._noise_category_tier_candidates_cache is old_cache


def test_natural_large_pool_reuses_prebuilt_rank_select_indexes():
    """十万级候选重复抽样只做 rank-select，不重扫 UID/category 映射。"""
    import core.data.dynamic_dataset_base as ddb

    dataset = SEDynamicMixingDataset.__new__(SEDynamicMixingDataset)
    excluded_uid = "vehicle"
    other_uids = tuple(f"thunder-{index}" for index in range(100_000))
    dataset.noise_uids = {16000: [excluded_uid, *other_uids]}
    dataset._noise_manifest_metadata = {
        excluded_uid: {"category_id": 200, "category_name": "Vehicle"},
        **{
            uid: {"category_id": 300, "category_name": "Thunder"}
            for uid in other_uids
        },
    }
    dataset.balanced_noise_sampling_enabled = False
    dataset._noise_sampling_config = None
    dataset._build_noise_sampling_state()

    positions = dataset._noise_natural_positions_by_fs[16000]
    uids = dataset._noise_natural_uids_by_fs[16000]
    assert len(uids) == 100_001
    assert len(positions[300]) == 100_000
    for seed in range(64):
        selected = dataset._select_natural_different_category_uid(
            16000,
            np.random.default_rng(seed),
            exclude_category_id=200,
            exclude_uid=excluded_uid,
        )
        assert selected is not None and selected.startswith("thunder-")
        assert dataset._noise_natural_positions_by_fs[16000] is positions
        assert dataset._noise_natural_uids_by_fs[16000] is uids
    assert dataset._noise_natural_tier_uids_cache == {}
    view = dataset._noise_natural_tier_view(
        "exact", (16000,), exclude_category_id=200, exclude_uid=excluded_uid
    )
    assert isinstance(view, ddb._NaturalTierExcludedView)
    assert len(view) == 100_000


def test_natural_selector_has_no_repeat_retry_cap():
    """唯一异类候选可无限重复，不引入隐式重试上限或状态。"""
    dataset = SEDynamicMixingDataset.__new__(SEDynamicMixingDataset)
    dataset.noise_uids = {16000: ["vehicle", "thunder"]}
    dataset._noise_manifest_metadata = {
        "vehicle": {"category_id": 200, "category_name": "Vehicle"},
        "thunder": {"category_id": 300, "category_name": "Thunder"},
    }
    dataset.balanced_noise_sampling_enabled = False
    dataset._noise_sampling_config = None
    dataset._build_noise_sampling_state()

    selected = [
        dataset._select_natural_different_category_uid(
            16000,
            np.random.default_rng(seed),
            exclude_category_id=200,
            exclude_uid="vehicle",
        )
        for seed in range(100)
    ]
    assert selected == ["thunder"] * 100


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True])
def test_temperature_rejects_nonfinite_or_boolean_difficulty(tmp_path, monkeypatch, invalid):
    import core.data.dynamic_dataset_base as ddb

    difficulties = dict(ddb.DEFAULT_DIFFICULTY)
    difficulties["quasi_stationary:continuous"] = invalid
    monkeypatch.setattr(ddb, "DEFAULT_DIFFICULTY", difficulties)
    records = _write_category_records(tmp_path, "vehicle", 200, "Vehicle", 1)

    with pytest.raises(ValueError, match="must be finite and > 0"):
        _make_dataset(tmp_path, records, sampling=_temperature_sampling())


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True])
def test_temperature_runtime_rejects_invalid_alpha(tmp_path, invalid):
    from dataclasses import replace

    records = _write_category_records(tmp_path, "vehicle", 200, "Vehicle", 1)
    dataset = _make_dataset(tmp_path, records, sampling=_temperature_sampling())
    dataset._noise_sampling_config = replace(dataset._noise_sampling_config, alpha=invalid)

    with pytest.raises(ValueError, match="alpha must be finite"):
        dataset._temperature_candidates_and_weights(16000)


def test_temperature_runtime_rejects_nonfinite_final_weight(tmp_path):
    records = _write_category_records(tmp_path, "vehicle", 200, "Vehicle", 1)
    dataset = _make_dataset(tmp_path, records, sampling=_temperature_sampling())
    dataset._noise_category_weight_by_id[200] = float("inf")

    with pytest.raises(ValueError, match="weights must be finite"):
        dataset._temperature_candidates_and_weights(16000)


def test_plain_selector_100k_pool_reuses_stable_uid_tuple_and_strips_worker_cache(
    monkeypatch,
):
    """超预算普通池在主进程与 worker 中各只冷建一次稳定 UID tuple。"""
    import core.data.dynamic_dataset_base as ddb

    monkeypatch.setattr(ddb, "_SAMPLE_SELECTOR_CACHE_MAX_UID_REFS", 3)

    class CountingKeysDict(dict):
        def __init__(self, items):
            super().__init__(items)
            self.keys_calls = 0

        def keys(self):
            self.keys_calls += 1
            return super().keys()

    exact = CountingKeysDict((f"noise-{index}", None) for index in range(100_001))
    source = {16000: exact}
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)

    cold_rng = np.random.default_rng(20260822)
    cold = [dataset._select_sample(16000, source, cold_rng) for _ in range(64)]
    warm_rng = np.random.default_rng(20260822)
    warm = [dataset._select_sample(16000, source, warm_rng) for _ in range(64)]

    assert warm == cold
    assert exact.keys_calls == 1
    assert dataset._sample_selector_cache_uid_refs == len(exact) > 3
    assert len(dataset._sample_selector_source_cache) == 1
    oversized_entry = next(iter(dataset._sample_selector_source_cache.values()))
    assert oversized_entry.source_by_fs is source
    assert oversized_entry.uids_by_fs[16000][0] == "noise-0"
    assert len(oversized_entry.uids_by_fs[16000]) == len(exact)

    state = dataset.__getstate__()
    assert state["_sample_selector_source_cache"] == {}
    assert state["_sample_selector_cache_uid_refs"] == 0
    worker = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    worker.__dict__.update(state)
    worker_rng = np.random.default_rng(20260822)
    worker_draws = [worker._select_sample(16000, source, worker_rng) for _ in range(64)]
    assert worker_draws == cold
    assert exact.keys_calls == 2  # 主进程一次、worker 冷建一次，worker 热路径不再扫描。


def test_plain_selector_preserves_exact_higher_all_order_and_duplicate_weight():
    """缓存不得改变 dict 顺序、higher 首选或跨 FS 重复 UID 的抽样权重。"""
    source = {
        48000: {"dup": None, "high": None},
        8000: {"low": None, "dup": None},
        16000: {"exact": None},
    }
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)

    assert dataset._select_sample(16000, source, ScriptedRng([0])) == "exact"
    assert dataset._select_sample(24000, source, ScriptedRng([1])) == "high"
    # all tier 按 source dict 原迭代顺序拼接；dup 保留两次，不做隐藏去重。
    all_draws = [dataset._select_sample(96000, source, ScriptedRng([index])) for index in range(5)]
    assert all_draws == ["dup", "high", "low", "dup", "exact"]


def test_plain_selector_cache_enforces_source_processing_sr_and_uid_budgets(monkeypatch):
    """缓存状态只能是常规预算内，或独占的单个 oversized stable entry。"""
    import core.data.dynamic_dataset_base as ddb

    monkeypatch.setattr(ddb, "_SAMPLE_SELECTOR_CACHE_MAX_SOURCES", 2)
    monkeypatch.setattr(ddb, "_SAMPLE_SELECTOR_CACHE_MAX_PROCESSING_SRS", 2)
    monkeypatch.setattr(ddb, "_SAMPLE_SELECTOR_CACHE_MAX_UID_REFS", 3)
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    sources = [{16000: {f"uid-{index}": None}} for index in range(3)]

    for index, source in enumerate(sources):
        assert dataset._select_sample(16000, source, ScriptedRng([0])) == f"uid-{index}"
    assert len(dataset._sample_selector_source_cache) == 2
    assert dataset._sample_selector_cache_uid_refs == 2 <= 3

    active = sources[-1]
    dataset._select_sample(8000, active, ScriptedRng([0]))
    dataset._select_sample(12000, active, ScriptedRng([0]))
    assert len(dataset._sample_selector_source_cache[id(active)].processing_sr_views) == 2

    class CountingKeysDict(dict):
        def __init__(self, items):
            super().__init__(items)
            self.keys_calls = 0

        def keys(self):
            self.keys_calls += 1
            return super().keys()

    oversized_uids = CountingKeysDict((f"large-{index}", None) for index in range(4))
    oversized = {16000: oversized_uids}
    for processing_sr in (16000, 8000, 12000, 16000):
        dataset._select_sample(processing_sr, oversized, ScriptedRng([0]))
    assert oversized_uids.keys_calls == 1
    assert len(dataset._sample_selector_source_cache) == 1
    assert list(dataset._sample_selector_source_cache) == [id(oversized)]
    oversized_entry = dataset._sample_selector_source_cache[id(oversized)]
    assert oversized_entry.source_by_fs is oversized
    assert list(oversized_entry.uids_by_fs) == [16000]
    assert len(oversized_entry.processing_sr_views) == 2
    assert dataset._sample_selector_cache_uid_refs == 4 > 3

    replacement_uids = CountingKeysDict((f"new-{index}", None) for index in range(5))
    replacement = {16000: replacement_uids}
    dataset._select_sample(16000, replacement, ScriptedRng([0]))
    dataset._select_sample(16000, replacement, ScriptedRng([0]))
    assert replacement_uids.keys_calls == 1
    assert len(dataset._sample_selector_source_cache) == 1
    assert list(dataset._sample_selector_source_cache) == [id(replacement)]
    assert dataset._sample_selector_source_cache[id(replacement)].source_by_fs is replacement
    assert dataset._sample_selector_cache_uid_refs == 5 > 3


pytestmark = [pytest.mark.unit]
