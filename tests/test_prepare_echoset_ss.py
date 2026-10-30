import json

import pytest
import soundfile as sf
import torch

from core.data.separation_manifest import parse_separation_manifest
from tools.prepare_data_tools.generators.ss.prepare_echoset import prepare_echoset


@pytest.mark.unit
def test_prepare_echoset_publishes_strict_split_manifests(tmp_path):
    root = tmp_path / "EchoSet"
    for split in ("train", "val", "test"):
        sample = root / split / "scene" / "room" / "pair"
        sample.mkdir(parents=True)
        for name in ("mix.wav", "spk1_reverb.wav", "spk2_reverb.wav"):
            sf.write(sample / name, torch.zeros(320).numpy(), 16000)
    output = tmp_path / "manifests"
    summary = prepare_echoset(root, output)
    assert summary["generator_version"] == "1.0"
    assert summary["splits"]["train"]["records"] == 1
    records = parse_separation_manifest(output / "train.jsonl")
    assert records[0].sample_rate == 16000
    assert records[0].length == 320
    metadata = json.loads((output / "echoset_metadata.json").read_text())
    assert metadata["splits"]["test"]["records"] == 1
