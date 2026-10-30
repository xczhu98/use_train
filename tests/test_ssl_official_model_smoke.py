"""官方 SSL 模型的端到端 one-epoch smoke（IPDNetFixed / IPDNetVariable / IPDNet2）。

每个可跑模型覆盖以下完整训练链路：
合成 fixture → ``SSLDynamicMixingDataset`` → projection → ``ssl_collate_fn`` →
模型 → ``SSLDPIPDPITLoss`` → optimizer step → validation → checkpoint 保存/恢复。

CPU 用例标记 ``unit``（IPDNet2 因 ``mamba_ssm`` 为 CUDA-only 而无法在 CPU 前向，
只覆盖构造/合同边界）；IPDNet2 的端到端 loop 标记 ``gpu``。本文件不使用全局
``pytestmark``，避免把 ``gpu`` 用例一起选进 ``-m unit``。
"""

import json
from dataclasses import replace

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


# 官方实现的公共对齐口径：
# - IPDnet（fixed / variable）CausCnnBlock 的 time stride 固定 12，forward 断言
#   ``T_out = T // 12``，因此输入帧数必须是 12 的倍数；
# - IPDNet2 的 time_pooling 是 ``AvgPool2d(time_compression_ratio, 1)``（频率轴取 1、
#   时间轴取 time_compression_ratio），forward 要求 ``T % time_compression_ratio == 0``；
#   实际输入频率永远满足整除链，因为 ``num_freqs=128`` 推出了 ``freq_after_fold2=64``。
# 画布长度由目标帧数反推（center=False 时 projection 的帧公式为 T = floor((N - n_fft) / hop) + 1），
# 这里统一取 T=480、hop=8、n_fft=16 / win_length=16 / center=False -> N=3864。
_SR = 8000
_STFT = {"n_fft": 16, "hop_length": 8, "win_length": 16, "window": "hamming", "center": False}
_FRAMES = 480
_FREQS = _STFT["n_fft"] // 2 + 1  # 9，含 DC

# IPDNet2 需要更大的频率轴：``num_freqs`` 必须恰好满足
# ``num_freqs == (num_freqs // 2 // 8) * fre_compression_ratio``（fre_compression_ratio=16）,
# n_fft 也相应提高到 256（运行 STFT 上下文见 ``_ipdnet2_stft_cfg``）。
_IPDNET2_N_FFT = 256
_IPDNET2_FREQS = 128


def _length_for_frames(frames, n_fft=_STFT["n_fft"]):
    """由目标帧数反推画布样本数（与 projection 的帧公式互逆）。"""
    return (frames - 1) * _STFT["hop_length"] + n_fft


def _speech_scp(tmp_path, length):
    """只生成 SCP 引用；实际 waveform 由 dataset 从 scp 读取后自行渲染。"""
    time = np.arange(length, dtype=np.float32) / _SR
    paths = []
    for index, frequency in enumerate((240.0, 420.0)):
        path = tmp_path / f"speech_{index}.wav"
        sf.write(path, np.sin(2 * np.pi * frequency * time).astype(np.float32), _SR, subtype="FLOAT")
        paths.append(path)
    speech_scp = tmp_path / "speech.scp"
    speech_scp.write_text(
        "\n".join(f"source_{i} {_SR} {length} {path}" for i, path in enumerate(paths)) + "\n",
        encoding="utf-8",
    )
    return paths, speech_scp


def _rir_manifest(tmp_path):
    """两个张成 ``[C,T]``（channel-first）的观测 RIR actor manifest。"""
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
    manifest = tmp_path / "spatial_rir.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": {
                    f"source_{i}": {
                        "full": str(rir_paths[i]),
                        "direct": str(direct_paths[i]),
                        "early": str(direct_paths[i]),
                    }
                    for i in range(2)
                },
                "metadata": {
                    "schema_version": 1,
                    "sample_rate": _SR,
                    "scene_uid": "ssl-official-smoke-scene",
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
    return manifest


def _dataset(tmp_path, *, frames=_FRAMES, output_stride, exclude_dc, normalization, n_fft=None):
    """两麦固定阵列的 canonical SSL 场景；口径与 configs/ssl/*.yaml 一致。"""
    n_fft = _STFT["n_fft"] if n_fft is None else int(n_fft)
    length = _length_for_frames(frames, n_fft)
    _, speech_scp = _speech_scp(tmp_path, length)
    manifest = _rir_manifest(tmp_path)
    return SSLDynamicMixingDataset(
        str(speech_scp),
        tracks=2,
        sample_rate=_SR,
        segment_seconds=length / _SR,
        rir_scp=str(manifest),
        simulation_cfg={
            "spatial_enabled": True,
            "reference_channel": 0,
            "prob_reverberation": 1.0,
            "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": False,
            "peak_normalization_mode": "off",
        },
        custom_geometry={"positions_m": [[2.0, 2.0, 1.0], [2.08, 2.0, 1.0]]},
        n_fft=n_fft,
        hop_length=_STFT["hop_length"],
        win_length=n_fft,
        window=_STFT["window"],
        center=_STFT["center"],
        input_representation="multichannel_stft_real_imag",
        output_stride=output_stride,
        exclude_dc=exclude_dc,
        normalization=normalization,
    )


def _trainer(tmp_path, loader, *, model_cfg, max_tracks, stft_cfg=None):
    stft_cfg = _STFT if stft_cfg is None else stft_cfg
    model = build_model(model_cfg, stft_cfg=stft_cfg)
    loss = build_loss(
        {"type": "SSLDPIPDPITLoss", "strict": True, "args": {"max_tracks": max_tracks}},
        stft_cfg=stft_cfg,
        sample_rate=_SR,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = {
        "meta": {"allow_no_stft": True, "seed": 1},
        "model": model_cfg,
        "data": {"sample_rate": _SR},
        "loss": {"type": "SSLDPIPDPITLoss", "args": {"max_tracks": max_tracks}},
        "distributed": {"find_unused_parameters": False},
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


def _run_epoch_and_checkpoint(tmp_path, dataset, trainer_factory):
    """执行 one epoch（train step + validation + checkpoint），并验证恢复前向。

    返回 (model, checkpoint)。恢复后在 eval 模式前向同一 batch，确认权重可用；
    CUDA 上 batch 与参数会自动落到 ``BaseTrainer`` 选定的设备。
    """
    loader = DataLoader(dataset, batch_size=1, collate_fn=ssl_collate_fn)
    trainer = trainer_factory(loader)
    trainer.train()
    checkpoint = trainer.ckpt_dir / "latest_model.tar"
    assert checkpoint.is_file()

    restored = trainer_factory(loader)
    restored.load_pretrained(str(checkpoint))
    # trainer.model 是可能被 DDP/compile 包装的句柄；前向用解包后的原始模型。
    model = restored._raw_model()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    model.eval()
    with torch.no_grad():
        output = model(**_restored_forward_kwargs(model, restored.device, loader))
    assert torch.isfinite(output).all()
    return model, checkpoint


def _restored_forward_kwargs(model, device, loader):
    """按模型声明的 feature contract 复用同一 batch 做恢复后前向。

    ``runtime.model_kwargs`` 会为声明了 ``pair_index`` 形参的模型（官方
    variable-array IPDnet）一并传入 canonical ``pair_index``。
    """
    from tasks.ssl.runtime import SSLRuntimeInputs

    batch = next(iter(loader))
    runtime = SSLRuntimeInputs._from_validated_batch(batch)
    return {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in runtime.model_kwargs(model).items()
    }


@pytest.mark.unit
def test_runtime_model_kwargs_unwraps_compile_and_ddp_contracts():
    """训练 wrapper 不能隐藏官方模型的输入合同或 ``pair_index``。"""
    from tasks.ssl.runtime import SSLRuntimeInputs

    model = build_model(_ipdnet_variable_cfg(), stft_cfg=_STFT)
    runtime = SSLRuntimeInputs(
        features=torch.zeros(1, 12, _FREQS, 4),
        target_dpipd=torch.zeros(1, 1, _FREQS, 2, 1, 2),
        pair_index=torch.tensor([[[0, 1]]], dtype=torch.int64),
        pair_mask=torch.ones(1, 1, dtype=torch.bool),
        geometry=torch.zeros(1, 2, 3),
        source_distances=None,
        input_lengths=torch.tensor([12]),
        output_lengths=torch.tensor([1]),
        frame_origin=torch.tensor([0], dtype=torch.int64),
        time_stride=12,
        input_representation="multichannel_stft_real_imag",
        target_representation="dpipd_far_field",
        coordinate_frame="array_local",
        doa=None,
        vad=None,
        sample_rates=None,
    )

    class _CompileWrapper:
        def __init__(self, inner):
            self._orig_mod = inner

    class _DDPWrapper:
        def __init__(self, inner):
            self.module = inner

    kwargs = runtime.model_kwargs(_DDPWrapper(_CompileWrapper(model)))
    assert set(kwargs) == {"multichannel_stft_real_imag", "pair_index"}
    assert kwargs["pair_index"].shape == (1, 1, 2)


# ---------------------------------------------------------------------------
# IPDNetFixed / IPDNetVariable：CPU one-epoch smoke
# ---------------------------------------------------------------------------


def _ipdnet_fixed_cfg(*, causal=True):
    return {
        "type": "IPDNetFixed",
        "args": {
            "input_size": 4,
            "hidden_size": 4,
            "max_track": 2,
            "n_fft": _STFT["n_fft"],
            "stride": 12,
            "causal": causal,
            "dropout": 0.0,
            "cnn_hidden_dim": 4,
            "reference_channel": 0,
        },
    }


def _ipdnet_variable_cfg(*, causal=True):
    return {
        "type": "IPDNetVariable",
        "args": {
            "input_size": 4,
            "hidden_size": 4,
            "max_track": 2,
            "n_fft": _STFT["n_fft"],
            "stride": 12,
            "causal": causal,
            "dropout": 0.0,
            "cnn_hidden_dim": 4,
            "reference_channel": 0,
            "pair_channel_order": "per_mic_interleaved",
        },
    }


@pytest.mark.unit
def test_ipdnet_fixed_dataset_and_projection_contract(tmp_path):
    """投影侧口径：T=480/12=40 输出帧、F=9、P=1、C=2，pair_index 为 reference-to-all。"""
    dataset = _dataset(
        tmp_path, frames=_FRAMES, output_stride=12, exclude_dc=False, normalization="none"
    )
    item = dataset[0]
    assert item["features"].shape == (_FRAMES, _FREQS, 4)
    assert item["target_dpipd"].shape == (_FRAMES // 12, _FREQS, 2, 1, 2)
    assert item["pair_index"].tolist() == [[0, 1]]
    assert item["time_stride"] == 12
    assert item["sample_rate"] == _SR


@pytest.mark.unit
def test_ipdnet_fixed_shape_contract_is_enforced():
    """前向合同：错 feature 维度 / 错频点 / 错帧数必须显式报错。"""
    model = build_model(_ipdnet_fixed_cfg(), stft_cfg=_STFT)
    with pytest.raises(ValueError, match="frequency bins"):
        model(torch.randn(1, _FRAMES, _FREQS + 1, 4))
    with pytest.raises(ValueError, match="at least 12 input frames"):
        model(torch.randn(1, 6, _FREQS, 4))
    # 官方排布要求输入为 ``2*C`` 个通道；少一对会直接拒绝。
    with pytest.raises(ValueError, match="frequency bins|feature|2\\*C"):
        model(torch.randn(1, _FRAMES, _FREQS, 2))


@pytest.mark.unit
def test_ipdnet_fixed_dynamic_cpu_epoch_saves_and_loads_checkpoint(tmp_path):
    """IPDNetFixed 端到端 one epoch：train step + validation + checkpoint 恢复前向。"""
    dataset = _dataset(
        tmp_path, frames=_FRAMES, output_stride=12, exclude_dc=False, normalization="none"
    )
    model, checkpoint = _run_epoch_and_checkpoint(
        tmp_path,
        dataset,
        lambda loader: _trainer(
            tmp_path, loader, model_cfg=_ipdnet_fixed_cfg(), max_tracks=2
        ),
    )
    assert type(model).__name__ == "IPDNetFixed"
    assert checkpoint.is_file()


@pytest.mark.unit
def test_ipdnet_fixed_offline_cpu_epoch_saves_and_loads_checkpoint(tmp_path):
    """IPDNetFixed offline（双向 narrow-band）走完整 train/restore 闭环。"""
    dataset = _dataset(
        tmp_path, frames=_FRAMES, output_stride=12, exclude_dc=False, normalization="global_mean"
    )
    model, checkpoint = _run_epoch_and_checkpoint(
        tmp_path,
        dataset,
        lambda loader: _trainer(
            tmp_path,
            loader,
            model_cfg=_ipdnet_fixed_cfg(causal=False),
            max_tracks=2,
        ),
    )
    assert type(model).__name__ == "IPDNetFixed"
    assert model.is_online is False
    assert model.block_1.narr_lstm.bidirectional is True
    assert checkpoint.is_file()


@pytest.mark.unit
def test_ipdnet_variable_dynamic_cpu_epoch_saves_and_loads_checkpoint(tmp_path):
    """IPDNetVariable 端到端 one epoch。

    ``runtime.model_kwargs`` 会为声明了 ``pair_index`` 形参的模型补上 canonical
    ``pair_index``，因此这里直接走 trainer 的真实前向路径。
    """
    dataset = _dataset(
        tmp_path, frames=_FRAMES, output_stride=12, exclude_dc=False, normalization="none"
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=ssl_collate_fn)
    trainer = _trainer(
        tmp_path, loader, model_cfg=_ipdnet_variable_cfg(), max_tracks=2
    )
    trainer.train()
    checkpoint = trainer.ckpt_dir / "latest_model.tar"
    assert checkpoint.is_file()

    restored = _trainer(
        tmp_path / "restored", loader, model_cfg=_ipdnet_variable_cfg(), max_tracks=2
    )
    restored.load_pretrained(str(checkpoint))
    assert all(torch.isfinite(parameter).all() for parameter in restored.model.parameters())
    device = restored.device
    with torch.no_grad():
        output = restored.model(
            **_restored_forward_kwargs(restored.model, device, loader)
        )
    assert output.shape == (1, _FRAMES // 12, _FREQS, 2, 1, 2)
    assert torch.isfinite(output).all()


@pytest.mark.unit
def test_ipdnet_variable_offline_cpu_epoch_saves_and_loads_checkpoint(tmp_path):
    """IPDNetVariable offline（双向 narrow-band）走完整 train/restore 闭环。"""
    dataset = _dataset(
        tmp_path, frames=_FRAMES, output_stride=12, exclude_dc=False, normalization="global_mean"
    )
    model, checkpoint = _run_epoch_and_checkpoint(
        tmp_path,
        dataset,
        lambda loader: _trainer(
            tmp_path,
            loader,
            model_cfg=_ipdnet_variable_cfg(causal=False),
            max_tracks=2,
        ),
    )
    assert type(model).__name__ == "IPDNetVariable"
    assert model.is_online is False
    assert model.block_1.narr_lstm.bidirectional is True
    assert checkpoint.is_file()


@pytest.mark.unit
def test_ipdnet_variable_requires_pair_index_and_validates_pairs():
    """构造/前向合同：``pair_index`` 必填，且必须落在通道范围内。"""
    model = build_model(_ipdnet_variable_cfg(), stft_cfg=_STFT)
    features = torch.randn(1, _FRAMES, _FREQS, 4)
    with pytest.raises(TypeError, match="pair_index"):
        model(features)
    with pytest.raises(ValueError, match="out-of-range microphone"):
        model(features, pair_index=torch.tensor([[[0, 5]]], dtype=torch.int64))
    with pytest.raises(ValueError, match="self pairs"):
        model(features, pair_index=torch.tensor([[[1, 1]]], dtype=torch.int64))
    with pytest.raises(ValueError, match="at least 12 input frames"):
        model(
            torch.randn(1, 6, _FREQS, 4),
            pair_index=torch.tensor([[[0, 1]]], dtype=torch.int64),
        )


@pytest.mark.unit
def test_ssl_dpipd_pit_loss_rejects_wrong_prediction_shape(tmp_path):
    """loss 的 batch 形状校验：prediction 必须与 canonical target 逐轴一致。"""
    dataset = _dataset(
        tmp_path, frames=_FRAMES, output_stride=12, exclude_dc=False, normalization="none"
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=ssl_collate_fn)
    from tasks.ssl.runtime import SSLRuntimeInputs

    batch = next(iter(loader))
    runtime = SSLRuntimeInputs._from_validated_batch(batch)
    loss = build_loss(
        {"type": "SSLDPIPDPITLoss", "strict": True, "args": {"max_tracks": 2}},
        stft_cfg=_STFT,
        sample_rate=_SR,
    )
    good = batch.target_dpipd.clone()
    assert torch.isfinite(loss(good, **runtime.loss_kwargs()))
    with pytest.raises(ValueError, match="share B,T,F"):
        loss(good[:, :-1], **runtime.loss_kwargs())
    with pytest.raises(ValueError, match="real/imag layout"):
        loss(good[..., :1], **runtime.loss_kwargs())
    with pytest.raises(ValueError, match="track dimensions are invalid"):
        loss(good[:, :, :, :1], **runtime.loss_kwargs())


@pytest.mark.unit
def test_ssl_batch_validate_rejects_inconsistent_metadata(tmp_path):
    """``SSLBatch.validate`` 必须拒绝被篡改的 time_stride / pair_index。"""
    dataset = _dataset(
        tmp_path, frames=_FRAMES, output_stride=12, exclude_dc=False, normalization="none"
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=ssl_collate_fn)
    batch = next(iter(loader))
    with pytest.raises(ValueError, match="time_stride"):
        replace(batch, time_stride=0).validate()
    with pytest.raises(ValueError, match="self pairs"):
        replace(batch, pair_index=torch.tensor([[[0, 0]]], dtype=torch.int64)).validate()


# ---------------------------------------------------------------------------
# IPDNet2：CPU 合同边界 + GPU 端到端 one-epoch smoke
# ---------------------------------------------------------------------------


def _cuda_available():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _report_gpu_usage(label, peak_bytes):
    """把峰值显存档位输出到 stdout，便于在无 -s 的运行里回读实测值。"""
    print(f"[ssl-official-smoke] {label} peak_cuda_bytes={int(peak_bytes)}")


def _ipdnet2_cfg():
    """对齐 configs/ssl/ipdnet2_dynamic.yaml 的字段组合。

    与配置模板的差异只有尺寸与光栅参数：n_fft=256（模板 512）及配套的
    ``num_freqs=128``。官方整除链
    ``num_freqs == (num_freqs//2//8) * fre_compression_ratio`` 在模板的
    ``num_freqs=512//2=256`` 时自动成立（256//2//8*16 = 256），在本文件的
    ``num_freqs=128`` 下同样成立（128//2//8*16 = 128）。
    """
    return {
        "type": "IPDNet2",
        "args": {
            "input_size": 1,
            "hidden_size": 8,
            "max_track": 2,
            "n_fft": _IPDNET2_N_FFT,
            "num_freqs": _IPDNET2_FREQS,
            "stride": 5,
            "causal": True,
            "num_layers": 2,
            "num_heads": 2,
            "dim_squeeze": 8,
            "attention": "mamba(16,4)",
            "fre_compression_ratio": 16,
            "time_compression_ratio": 5,
            "time_compression_layer": 0,
            "reference_channel": 0,
            "forgetting_norm_sample_length": None,
        },
    }


@pytest.mark.unit
def test_ipdnet2_cpu_contract_rejects_malformed_input():
    """CPU 上可校验的输入合同（无需 ``mamba_ssm``）。

    模型声明 n_fft=256，与两麦 fixture 的顶层 stft 上下文不同，因此显式传
    ``stft_cfg=None`` 构造；这正对应配置模板里模型 n_fft 与运行 STFT 的显式口径分离。
    """
    model = build_model(_ipdnet2_cfg(), stft_cfg=None)
    with pytest.raises(ValueError, match="feature dimension"):
        model(torch.randn(1, _FRAMES, _IPDNET2_FREQS, 8))
    with pytest.raises(ValueError, match="frequency bins"):
        model(torch.randn(1, _FRAMES, 20, 4))
    with pytest.raises(ValueError, match="divisible by"):
        model(torch.randn(1, 6, _IPDNET2_FREQS, 4))
    # canonical 全带宽 129（n_fft//2+1）会被内部删 DC 变成 128，属于官方口径；
    # 真正越界的是既不等于 num_freqs 也不等于 n_fft//2+1 的频点数。
    with pytest.raises(ValueError, match="frequency bins"):
        model(torch.randn(1, _FRAMES, _IPDNET2_FREQS + 2, 4))


@pytest.mark.unit
def test_ipdnet2_dataset_uses_exclude_dc_freqs(tmp_path):
    """``exclude_dc=true`` 时数据侧频点数为 n_fft//2（与模型 num_freqs 配套）。"""
    frames, freqs = _FRAMES, _IPDNET2_FREQS
    dataset = _dataset(
        tmp_path,
        frames=frames,
        output_stride=5,
        exclude_dc=True,
        normalization="forgetting_laplace",
        n_fft=_IPDNET2_N_FFT,
    )
    item = dataset[0]
    assert item["features"].shape == (frames, freqs, 4)
    assert item["target_dpipd"].shape == (frames // 5, freqs, 2, 1, 2)


def _ipdnet2_stft_cfg():
    """IPDNet2 的运行 STFT 上下文：F=n_fft//2+1=129，删 DC 后 128 = ``num_freqs``。"""
    return {
        "n_fft": _IPDNET2_N_FFT,
        "hop_length": 8,
        "win_length": _IPDNET2_N_FFT,
        "window": "hamming",
        "center": False,
    }


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="mamba_ssm is CUDA-only")
def test_ipdnet2_dynamic_gpu_epoch_saves_and_loads_checkpoint(tmp_path):
    """IPDNet2 端到端 one epoch（CUDA），含 validation 与 checkpoint 恢复前向。"""
    torch.cuda.reset_peak_memory_stats()
    dataset = _dataset(
        tmp_path,
        frames=_FRAMES,
        output_stride=5,
        exclude_dc=True,
        normalization="forgetting_laplace",
        n_fft=_IPDNET2_N_FFT,
    )
    model, checkpoint = _run_epoch_and_checkpoint(
        tmp_path,
        dataset,
        lambda loader: _trainer(
            tmp_path,
            loader,
            model_cfg=_ipdnet2_cfg(),
            max_tracks=2,
            stft_cfg=_ipdnet2_stft_cfg(),
        ),
    )
    assert type(model).__name__ == "IPDNet2"
    assert checkpoint.is_file()
    peak_bytes = torch.cuda.max_memory_allocated()
    assert peak_bytes > 0
    _report_gpu_usage("ipdnet2_e2e", peak_bytes)


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="mamba_ssm is CUDA-only")
def test_ipdnet2_gpu_loss_rejects_wrong_prediction_shape(tmp_path):
    """CUDA 上同样验证 loss 的 canonical 形状门禁。"""
    dataset = _dataset(
        tmp_path,
        frames=_FRAMES,
        output_stride=5,
        exclude_dc=True,
        normalization="forgetting_laplace",
        n_fft=_IPDNET2_N_FFT,
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=ssl_collate_fn)
    from tasks.ssl.runtime import SSLRuntimeInputs

    batch = next(iter(loader))
    runtime = SSLRuntimeInputs._from_validated_batch(batch)
    loss = build_loss(
        {"type": "SSLDPIPDPITLoss", "strict": True, "args": {"max_tracks": 2}},
        stft_cfg=_STFT,
        sample_rate=_SR,
    )
    good = batch.target_dpipd.clone()
    good = good.to("cuda")
    kwargs = {key: (value.to("cuda") if torch.is_tensor(value) else value) for key, value in runtime.loss_kwargs().items()}
    assert torch.isfinite(loss(good, **kwargs))
    with pytest.raises(ValueError, match="share B,T,F"):
        loss(good[:, :-1], **kwargs)


# ---------------------------------------------------------------------------
# 接口缺口登记
# ---------------------------------------------------------------------------
#
# 1) IPDNetVariable 的 ``pair_index`` 接线已收口：``SSLRuntimeInputs.model_kwargs``
#    现在会为 forward 声明了 ``pair_index`` 形参的模型补上 canonical
#    ``pair_index``（解开 DDP/torch.compile 包装后再看签名），因此
#    ``test_ipdnet_variable_dynamic_cpu_epoch_saves_and_loads_checkpoint`` 直接走
#    trainer 真实前向路径，不再需要替换 ``_forward_prediction``。
#
# 2) IPDNet2 的 ``num_freqs`` 与运行 STFT 的 ``n_fft`` 必须同源，但二者在配置里
#    是两处独立声明。``configs/ssl/ipdnet2_dynamic.yaml`` 的 n_fft=512 / num_freqs=256
#    恰好满足官方整除链；一旦按结构惯例把 n_fft 调小（如 16），官方频带链
#    ``num_freqs == (num_freqs//2//8) * fre_compression_ratio`` 立即失配，而
#    ``core/models/build.py`` 只在 ``n_fft`` 上做上下文一致性校验，不会校验
#    ``num_freqs`` 与 ``n_fft//2`` 的关系。本文件因此改用 STFT 上下文
#    ``_ipdnet2_stft_cfg()``（n_fft=256 / hop=8）与配套的 ``num_freqs=128``。
