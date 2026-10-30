"""FNSSL one-epoch CPU smoke：dataset→projection→collate→FNSSL→loss→optimizer→validation→checkpoint。

参照 tests/test_ssl_trainer_loop.py 的 FixedArrayIPDNet smoke 模式，
模型换成官方 full/narrow LSTM 结构移植的 ``FNSSL``，输入合同为
``multichannel_stft_real_imag``，loss 为 ``SSLDPIPDPITLoss``。
只使用合成 fixture，不依赖真实语料。
"""

import json

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

import tasks.ssl.losses  # noqa: F401 -- register SSL losses
from core.data.simulation.scene_manifest import build_track_identity
from core.losses.build import build_loss
from tasks.ssl.data import ssl_collate_fn
from tasks.ssl.data.dynamic_dataset import SSLDynamicMixingDataset
from tasks.ssl.models import build_model
from tasks.ssl.trainer import SSLTrainer


pytestmark = pytest.mark.unit

# FNSSL 官方平均池化 stride=12；projection 与模型都丢弃尾部 partial window。
# 这里使用 24 个完整输入帧，便于 one-epoch smoke 的输出长度固定为 2。
_SR = 8000
_LENGTH = 200
_STFT = {"n_fft": 16, "hop_length": 8, "win_length": 16, "window": "hamming", "center": False}


def _fnssl_dataset(tmp_path):
    time = np.arange(_LENGTH, dtype=np.float32) / _SR
    speech_paths = []
    for index, frequency in enumerate((240.0, 420.0)):
        path = tmp_path / f"speech_{index}.wav"
        sf.write(path, np.sin(2 * np.pi * frequency * time).astype(np.float32), _SR, subtype="FLOAT")
        speech_paths.append(path)
    speech_scp = tmp_path / "speech.scp"
    speech_scp.write_text(
        "\n".join(f"source_{i} {_SR} {_LENGTH} {path}" for i, path in enumerate(speech_paths)) + "\n",
        encoding="utf-8",
    )

    rir_paths = []
    direct_paths = []
    for index, delay in enumerate((0, 1)):
        rir = np.zeros((2, 8), dtype=np.float32)
        rir[0, delay] = 1.0
        rir[1, delay + 1] = 0.8
        path = tmp_path / f"rir_{index}.npy"
        np.save(path, rir)
        rir_paths.append(path)
        direct = rir.copy()
        direct[0, delay] *= 0.9
        direct_path = tmp_path / f"direct_{index}.npy"
        np.save(direct_path, direct)
        direct_paths.append(direct_path)
    rir_manifest = tmp_path / "spatial_rir.json"
    rir_manifest.write_text(
        json.dumps(
            {
                "entries": {
                    "source_0": {"full": str(rir_paths[0]), "direct": str(direct_paths[0]), "early": str(direct_paths[0])},
                    "source_1": {"full": str(rir_paths[1]), "direct": str(direct_paths[1]), "early": str(direct_paths[1])},
                },
                "metadata": {
                    "schema_version": 1,
                    "sample_rate": _SR,
                    "scene_uid": "ssl-fnssl-smoke-scene",
                    "actor_order": ["source_0", "source_1"],
                    "room": {"room_uid": "ssl-room", "room_dim_m": [6, 6, 3]},
                    "actor_positions_m": {"source_0": [1, 1, 1.5], "source_1": [2, 2, 1.5]},
                    "generator_version": "ssl-fixture-v1",
                    "track_identities": {
                        f"source_{i}": {
                            "full": build_track_identity(rir_paths[i]),
                            "direct": build_track_identity(direct_paths[i]),
                            "early": build_track_identity(direct_paths[i]),
                        }
                        for i in range(2)
                    },
                    "geometry": {
                        "positions_m": [[2.0, 2.0, 1.0], [2.08, 2.0, 1.0]],
                        "reference_channel": 0,
                        "array_profile": "linear_2ch_80mm",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return SSLDynamicMixingDataset(
        str(speech_scp),
        tracks=2,
        sample_rate=_SR,
        segment_seconds=_LENGTH / _SR,
        rir_scp=str(rir_manifest),
        simulation_cfg={
            "spatial_enabled": True,
            "reference_channel": 0,
            "prob_reverberation": 1.0,
            "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": False,
            "peak_normalization_mode": "off",
        },
        custom_geometry={"positions_m": [[2.0, 2.0, 1.0], [2.08, 2.0, 1.0]]},
        n_fft=_STFT["n_fft"],
        hop_length=_STFT["hop_length"],
        win_length=_STFT["win_length"],
        window=_STFT["window"],
        center=_STFT["center"],
        input_representation="multichannel_stft_real_imag",
        output_stride=12,
    )


def _fnssl_trainer(tmp_path, loader):
    model = build_model(
        {
            "type": "FNSSL",
            "args": {
                "input_size": 1,
                "hidden_size": 8,
                "max_track": 2,
                "n_fft": _STFT["n_fft"],
                "stride": 12,
                "causal": False,
            },
        },
        stft_cfg=_STFT,
    )
    loss = build_loss(
        {"type": "SSLDPIPDPITLoss", "strict": True, "args": {"max_tracks": 2}},
        stft_cfg=_STFT,
        sample_rate=_SR,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = {
        "meta": {"allow_no_stft": True, "seed": 1},
        "model": {"type": "FNSSL", "args": {}},
        "data": {"sample_rate": _SR},
        "loss": {"type": "SSLDPIPDPITLoss", "args": {"max_tracks": 2}},
        "trainer": {
            "epochs": 1,
            "exp_path": str(tmp_path / "experiment"),
            "save_checkpoint_interval": 1,
            "validation_epoch_interval": 1,
            "max_val_batches": 1,
            "strict_checks": True,
            "val_metrics": {"enabled": False},
        },
    }
    return SSLTrainer(
        train_loader=loader,
        val_loader=loader,
        sample_rate=_SR,
        stft_module=None,
        rank=0,
        world_size=1,
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        loss_fn=loss,
        config=config,
    )


def test_fnssl_dynamic_cpu_epoch_saves_and_loads_checkpoint(tmp_path):
    dataset = _fnssl_dataset(tmp_path)
    item = dataset[0]
    assert item["features"].shape[-1] == 4  # [T_in,F,2C]，C=2
    assert item["features"].shape[0] == 24
    assert item["target_dpipd"].shape[0] == 2  # 24 / stride 12
    loader = DataLoader(dataset, batch_size=1, collate_fn=ssl_collate_fn)
    trainer = _fnssl_trainer(tmp_path, loader)
    trainer.train()
    checkpoint = trainer.ckpt_dir / "latest_model.tar"
    assert checkpoint.is_file()

    restored = _fnssl_trainer(tmp_path / "restored", loader)
    restored.load_pretrained(str(checkpoint))
    assert all(torch.isfinite(parameter).all() for parameter in restored.model.parameters())
