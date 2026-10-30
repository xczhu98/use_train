"""offline AEC/TSE geometry、builder 透传和 observation 合同测试。"""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tasks.aec.data.offline_dataset import AECOfflineDataset
from tasks.aec.data import build_dataset as build_aec_dataset
from tasks.tse.data.offline_dataset import TSEOfflineDataset
from tasks.tse.data import build_dataset as build_tse_dataset

pytestmark = pytest.mark.unit


def _wav(path: Path, channels: int, frames: int = 32):
    sf.write(path, np.zeros((frames, channels), dtype=np.float32), 16000)


def _aec_cfg(root: Path, channels: int = 2):
    paths = {}
    for track in ("mic", "farend", "nearend"):
        path = root / f"{track}.wav"
        _wav(path, channels if track == "mic" else 1)
        scp = root / f"{track}.scp"
        scp.write_text(f"u 16000 32 {path}\n")
        paths[track] = scp
    return {"data": {"mode": "offline_aec", "train": {
        "mic_scp": str(paths["mic"]), "farend_scp": str(paths["farend"]),
        "nearend_scp": str(paths["nearend"]), "spatial_enabled": True,
        "array_profile": "linear_2ch_80mm", "reference_channel": 1,
        "channel_order": ["left", "right"],
    }}}


def test_aec_real_dataset_and_builder_preserve_spatial_kwargs(tmp_path):
    cfg = _aec_cfg(tmp_path)
    dataset = build_aec_dataset(cfg, "train")
    assert dataset.resolved_microphone_array.channels == 2
    assert dataset.channel_order == ("left", "right")
    assert dataset[0][1].shape == (2, 32)


def test_aec_offline_rejects_missing_geometry_and_wrong_order(tmp_path):
    cfg = _aec_cfg(tmp_path)
    section = cfg["data"]["train"]
    with pytest.raises(ValueError, match="requires array_profile or custom_geometry"):
        AECOfflineDataset(**{k: v for k, v in section.items() if k not in {"array_profile", "channel_order"}})
    with pytest.raises(ValueError, match="exactly C=2"):
        AECOfflineDataset(**{**section, "channel_order": ["left"]})


def test_aec_offline_rejects_actual_channel_count_mismatch(tmp_path):
    cfg = _aec_cfg(tmp_path, channels=1)
    with pytest.raises(ValueError, match="expected geometry C=2"):
        build_aec_dataset(cfg, "train")


def test_tse_real_dataset_and_builder_preserve_spatial_kwargs(tmp_path):
    mix = tmp_path / "mix.wav"
    target = tmp_path / "target.wav"
    enroll = tmp_path / "enroll.wav"
    _wav(mix, 2); _wav(target, 1); _wav(enroll, 1)
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps({"key": "u", "spk": ["s"], "mix": {"default": [str(mix)]}, "src": {"s": [str(target)]}}) + "\n")
    resource = tmp_path / "audio.json"
    resource.write_text(json.dumps({"s": [{"utt_id": "e", "path": str(enroll)}]}))
    cues = tmp_path / "cues.yaml"
    cues.write_text(
        f"cues:\n  audio:\n    type: raw\n    guaranteed: true\n"
        f"    scope: speaker\n    policy:\n      type: fixed\n"
        f"      key: spk_id\n      resource: {resource}\n"
    )
    cfg = {"data": {"mode": "offline_tse", "train": {
        "samples_jsonl": str(samples), "manifest_format": "jsonl",
        "path_base": "manifest_relative", "target_selection": "all",
        "mix_variant": "default", "cues_yaml": str(cues),
        "cue_schema": "realtse_raw", "enabled_cues": ["audio"],
        "spatial_enabled": True, "array_profile": "linear_2ch_80mm",
        "reference_channel": 1, "channel_order": [10, 20],
    }}}
    dataset = build_tse_dataset(cfg, "train")
    assert dataset.resolved_microphone_array.channels == 2
    assert dataset.channel_order == (10, 20)
    assert dataset[0]["mixture"].shape == (2, 32)


def test_tse_offline_rejects_wrong_order_and_actual_channels(tmp_path):
    mix = tmp_path / "mix.wav"; target = tmp_path / "target.wav"; enroll = tmp_path / "enroll.wav"
    _wav(mix, 1); _wav(target, 1); _wav(enroll, 1)
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps({"key": "u", "spk": ["s"], "mix": {"default": [str(mix)]}, "src": {"s": [str(target)]}}) + "\n")
    resource = tmp_path / "audio.json"; resource.write_text(json.dumps({"s": [{"path": str(enroll)}]}))
    cues = tmp_path / "cues.yaml"
    cues.write_text(
        f"cues:\n  audio:\n    type: raw\n    guaranteed: true\n"
        f"    scope: speaker\n    policy:\n      type: fixed\n"
        f"      key: spk_id\n      resource: {resource}\n"
    )
    common = {
        "samples_jsonl": str(samples), "manifest_format": "jsonl",
        "path_base": "manifest_relative", "target_selection": "all",
        "mix_variant": "default", "cues_yaml": str(cues),
        "cue_schema": "realtse_raw", "enabled_cues": ["audio"],
        "spatial_enabled": True, "array_profile": "linear_2ch_80mm",
    }
    with pytest.raises(ValueError, match="does not match geometry C=2"):
        TSEOfflineDataset(**common, channel_order=[0, 1])
    with pytest.raises(ValueError, match="exactly C=2"):
        TSEOfflineDataset(**common, channel_order=[0])
