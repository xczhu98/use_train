"""显式启用的真实资源验收：双卡 NCCL、online RIR、持久 worker 与跨 epoch 恢复。

USE_TRAIN_REAL_DATA_STABILITY=1 pytest -m 'integration and gpu'
tests/test_real_data_training_stability.py
该测试是有限规模的工程稳定性检查，不用于评估模型效果。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from core.utils.safe_load import safe_torch_load

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/se/spatial/online_spatialnet_dynamic_mhsa_lf10s_36e.yaml"
pytestmark = [pytest.mark.integration, pytest.mark.gpu,
              pytest.mark.skipif(os.environ.get("USE_TRAIN_REAL_DATA_STABILITY") != "1",
                                 reason="真实数据稳定性验收需显式启用")]


def _speech_subset(source: Path, target: Path, count: int) -> None:
    """保留真实音频路径，固定一个 native SR，避免小集合的 DDP 空 shard。"""
    rows = []
    sample_rate = None
    with source.open() as stream:
        for line in stream:
            fields = line.strip().split(maxsplit=3)
            if len(fields) != 4:
                continue
            uid, rate, length, audio = fields
            rate, length = int(rate), int(length)
            if not 2 <= length / rate <= 60:
                continue
            sample_rate = rate if sample_rate is None else sample_rate
            if rate != sample_rate:
                continue
            path = Path(audio)
            assert path.is_absolute() and path.is_file(), audio
            rows.append(f"{uid} {rate} {length} {path}\n")
            if len(rows) == count:
                break
    assert len(rows) == count
    target.write_text("".join(rows))


def _config(tmp_path: Path) -> dict:
    config = yaml.safe_load(CONFIG.read_text())
    config["meta"].update(compile_model=False, use_amp=True, use_ema=False)
    config["stft"].update(n_fft=128, hop_length=64, win_length=128)
    config["model"]["args"].update(n_fft=128, n_channel=2, num_layers=2,
                                  dim_hidden=16, dim_ffn=32, dim_squeeze=4,
                                  num_heads=2, conv_groups=[4, 4], attention="mhsa(32)")
    data = config["data"]
    data["sample_rate"] = 16000
    for split, count in (("train", 32), ("validation", 8)):
        source = ROOT / data[split]["speech_scp"]
        if not source.is_file():
            pytest.skip(f"真实资源缺失：{source}")
        subset = tmp_path / f"{split}.scp"
        _speech_subset(source, subset, count)
        data[split]["speech_scp"] = str(subset)
    data["train"]["max_duration_seconds"] = 1.0
    data["train"]["filter_min_duration"] = 0.5
    sim = data["train"]["simulation_cfg"]
    sim["speech"]["silence"]["prob"] = 0.0
    sim["spatial"]["geometry"]["profile"] = "linear_2ch_80mm"
    # 每条 observation 必经 codec；MP3/Vorbis/Opus 候选继承真实生产配置。
    sim["channel"]["augmentations_name"] = ["codec"]
    sim["channel"]["num_augmentations"] = {1: 1.0}
    sim["channel"]["augmentations"] = {"codec": sim["channel"]["augmentations"]["codec"]}
    data["dataloader"].update(batch_size=2, num_workers=1, val_num_workers=1,
                              persistent_workers=True, val_persistent_workers=True,
                              prefetch_factor=2, timeout=120)
    config["loss"] = {"type": "L1WavLoss"}
    config["trainer"].update(epochs=2, exp_path=str(tmp_path / "experiment"),
                             log_interval=1, max_val_batches=0, save_curve_interval=0,
                             save_checkpoint_interval=1, save_top_k=1,
                             val_metrics={"enabled": False})
    return config


def _train(config: dict, tmp_path: Path, name: str, resume: Path | None = None) -> tuple[dict, str]:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env["PYTHONPATH"] = str(ROOT)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node=2", "train.py", "-C", str(path)]
    if resume is not None:
        command.extend(["--resume-from", str(resume)])
    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    (tmp_path / f"{name}.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-16000:]
    assert "Training complete." in result.stdout
    checkpoints = list((Path(config["trainer"]["exp_path"]) / "checkpoints").glob("*/latest_model.tar"))
    checkpoint = max(checkpoints, key=lambda item: item.stat().st_mtime_ns)
    # 仅加载本测试刚生成的可信训练状态，以检查 optimizer/scheduler/RNG 恢复。
    state = safe_torch_load(checkpoint, map_location="cpu", allow_unsafe=True,
                            purpose="inspect locally generated real-data stability checkpoint")
    state["_checkpoint_path"] = checkpoint
    return state, result.stdout


def test_real_two_rank_training_cross_epoch_and_resume(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("需要至少两张 CUDA GPU")
    config = _config(tmp_path)
    first, _ = _train(config, tmp_path, "first")
    assert first["epoch"] == 2 and first["n_steps"] == 16
    config["trainer"]["epochs"] = 4
    resumed, log = _train(config, tmp_path, "resumed", first["_checkpoint_path"])
    assert resumed["epoch"] == 4 and resumed["n_steps"] == 32
    assert "[Val] epoch 3" in log and "[Val] epoch 4" in log
    assert len(resumed["rng_states_by_rank"]) == 2
    assert resumed["curve_epochs"] == [1, 2, 3, 4]
    for name in ("train_losses", "val_losses"):
        assert len(resumed[name]) == 4 and np.isfinite(resumed[name]).all()
        assert resumed[name][:2] == first[name]
    assert resumed["scheduler"]["last_epoch"] - first["scheduler"]["last_epoch"] == 2
    assert resumed["optimizer"]["state"]
    for key, value in resumed["optimizer"]["state"].items():
        if "step" in value:
            assert int(value["step"]) == int(first["optimizer"]["state"][key]["step"]) + 16
    assert any(not torch.equal(value, resumed["model"][key]) for key, value in first["model"].items())
    evidence = {"epochs": 4, "world_size": 2, "backend": "nccl", "steps": 32,
                "train_records": 32, "validation_records": 8,
                "train_losses": resumed["train_losses"], "val_losses": resumed["val_losses"],
                "persistent_workers": True, "rir_backend": "online", "codec_probability": 1.0}
    (tmp_path / "stability_evidence.json").write_text(json.dumps(evidence, indent=2))
