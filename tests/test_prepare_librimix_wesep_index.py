"""Libri2Mix WeSep v0.1 speaker-cue 清单生成器测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from tasks.tse.data.wesep_adapter import load_wesep_records
from tools.prepare_data_tools.generators.tse import prepare_librimix_wesep_index as producer

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _wav(path: Path, frames: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.linspace(-0.2, 0.2, frames, dtype=np.float32), 16000)


def _build_layout(tmp_path: Path, split: str = "train-100") -> tuple[Path, Path]:
    """构造迷你 Libri2Mix 布局：2 个样本（3 个 speaker）+ nested enrollment。"""
    root = tmp_path / "wav16k" / "min"
    keys = ["103-1240-0000_1235-135887-0000", "103-1240-0001_1272-128104-0000"]
    for key in keys:
        for sub in ("mix_clean", "s1", "s2"):
            _wav(root / split / sub / f"{key}.wav")
    enroll_root = tmp_path / "dynamic_tse"
    nested = {
        f"librimix_{split}": {
            spk: [str((tmp_path / "enroll" / f"{spk}-{i}.wav").resolve()) for i in range(2)]
            for spk in ("103", "1235", "1272")
        }
    }
    for paths in nested[f"librimix_{split}"].values():
        for raw in paths:
            _wav(Path(raw))
    (enroll_root / split).mkdir(parents=True)
    (enroll_root / split / "spk2enroll.json").write_text(
        json.dumps(nested), encoding="utf-8"
    )
    return root, enroll_root


def test_prepare_split_publishes_wesep_contract(tmp_path: Path) -> None:
    root, enroll_root = _build_layout(tmp_path)
    stats = producer.prepare_split(root, enroll_root, "train-100", mix_dir_name="mix_clean", overwrite=False)

    split_dir = root / "train-100"
    assert stats["samples"] == 2
    assert stats["speakers"] == 3
    assert all(action == "written" for action in stats["actions"].values())

    cues = yaml.safe_load((split_dir / "cues.yaml").read_text(encoding="utf-8"))
    assert cues["cues"]["audio"]["type"] == "wav"
    assert cues["cues"]["audio"]["format"] == "waveform"
    assert cues["cues"]["audio"]["policy"] == {
        "type": "random",
        "key": "spk_id",
        "resource": "enroll.json",
    }
    assert (split_dir / "raw.list").read_text(encoding="utf-8") == "samples.jsonl\n"

    # 生成产物必须能被生产 adapter 严格消费
    records = load_wesep_records(
        split_dir / "samples.jsonl",
        manifest_format="jsonl",
        path_base="manifest_relative",
        mix_variant="default",
        target_selection="first",
        cues_yaml=split_dir / "cues.yaml",
        enabled_cues=("audio",),
    )
    assert len(records) == 2
    first = records[0]
    assert first.key == "103-1240-0000_1235-135887-0000"
    assert Path(first.mixture_paths[0]).is_file()
    target = first.targets[0]
    assert target.speaker_id == "103"
    cue = target.cue("audio")
    assert cue.present is True and cue.items
    assert all(Path(item["path"]).is_file() for item in cue.items)
    assert all(item["duration_sec"] > 0 for item in cue.items)


def test_dev_split_writes_fixed_policy(tmp_path: Path) -> None:
    root, enroll_root = _build_layout(tmp_path, split="dev")
    producer.prepare_split(root, enroll_root, "dev", mix_dir_name="mix_clean", overwrite=False)
    cues = yaml.safe_load((root / "dev" / "cues.yaml").read_text(encoding="utf-8"))
    assert cues["cues"]["audio"]["policy"]["type"] == "fixed"


def test_missing_s1_file_rejected(tmp_path: Path) -> None:
    root, enroll_root = _build_layout(tmp_path)
    orphan = root / "train-100" / "s1" / "103-1240-0001_1272-128104-0000.wav"
    orphan.unlink()
    with pytest.raises(ValueError, match="WAV ID"):
        producer.prepare_split(root, enroll_root, "train-100", mix_dir_name="mix_clean", overwrite=False)


def test_enroll_must_cover_all_speakers(tmp_path: Path) -> None:
    root, enroll_root = _build_layout(tmp_path)
    nested = json.loads((enroll_root / "train-100" / "spk2enroll.json").read_text(encoding="utf-8"))
    del nested["librimix_train-100"]["1272"]
    (enroll_root / "train-100" / "spk2enroll.json").write_text(json.dumps(nested), encoding="utf-8")
    with pytest.raises(ValueError, match="enrollment"):
        producer.prepare_split(root, enroll_root, "train-100", mix_dir_name="mix_clean", overwrite=False)


def test_rerun_reuses_identical_products_and_rejects_drift(tmp_path: Path) -> None:
    root, enroll_root = _build_layout(tmp_path)
    first = producer.prepare_split(root, enroll_root, "train-100", mix_dir_name="mix_clean", overwrite=False)
    second = producer.prepare_split(root, enroll_root, "train-100", mix_dir_name="mix_clean", overwrite=False)
    assert all(action == "written" for action in first["actions"].values())
    assert all(action == "reused" for action in second["actions"].values())

    (root / "train-100" / "cues.yaml").write_text("cues: {}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        producer.prepare_split(root, enroll_root, "train-100", mix_dir_name="mix_clean", overwrite=False)
