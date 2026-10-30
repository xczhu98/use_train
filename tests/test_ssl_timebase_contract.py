"""SSL 时间基准字段与 inactive target manifold 合同测试。"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.special import j0

from tasks.ssl.data import (
    SSL_INACTIVE_MANIFOLD_BESSEL_J0,
    SSL_INACTIVE_MANIFOLD_ZERO,
    SSLProjectionConfig,
    SSLSceneInput,
    project_ssl_scene,
    ssl_collate_fn,
)
from tasks.ssl.data.projection import _reduce_output_frames
from tasks.ssl.losses import SSLDPIPDPITLoss


pytestmark = pytest.mark.unit


def _scene(*, length=64, silent_prefix=0, seed=0):
    """双麦单源 scene；``silent_prefix`` 将 direct path 前段置零制造 inactive 帧。

    声源用宽带噪声，保证 active 帧所有频点非退化；observation 叠加的
    额外噪声只用于验证 manifold 与样本随机性无关。
    """
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(length, generator=generator)
    direct = torch.stack((source, torch.roll(source, 1))).unsqueeze(0)
    if silent_prefix:
        direct[:, :, :silent_prefix] = 0.0
    observation = direct[0] + 0.01 * torch.randn(
        direct.shape[-2:], generator=generator
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]])
    source_positions = torch.tensor([[1.0, 0.0, 1.0]])
    return SSLSceneInput(observation, direct, positions, source_positions, 16000, {"backend": "fixture"})


def _config(**overrides):
    values = {"n_fft": 16, "hop_length": 8, "win_length": 16}
    values.update(overrides)
    return SSLProjectionConfig(**values)


def test_projection_emits_explicit_timebase_fields():
    item = project_ssl_scene(_scene(), _config(output_stride=3))
    assert item["time_stride"] == 3
    assert item["receptive_field"] == 3
    assert item["algorithmic_latency"] == 2
    mask = item["valid_output_mask"]
    assert mask.dtype is torch.bool
    assert mask.shape == (item["output_lengths"],)
    assert bool(mask.all().item())
    timebase = item["provenance"]["timebase"]
    assert timebase["receptive_field"] == 3
    assert timebase["algorithmic_latency"] == 2
    assert timebase["time_stride"] == 3
    assert item["provenance"]["inactive_target_manifold"] == SSL_INACTIVE_MANIFOLD_ZERO


def test_projection_timebase_defaults_for_stride_one():
    item = project_ssl_scene(_scene(), _config())
    assert item["receptive_field"] == 1
    assert item["algorithmic_latency"] == 0
    assert item["valid_output_mask"].shape == (item["input_lengths"],)


def test_projection_stride_drops_partial_group_like_average_pool_model():
    """模型的 stride 池化只产出完整窗口，target 时间轴必须保持同一合同。"""
    frames, stride = 280, 12
    targets = torch.zeros(frames, 4, 1, 1, 2)
    doa = torch.zeros(frames, 1, 2)
    vad = torch.ones(frames, 1)
    source_positions = torch.zeros(frames, 1, 3)
    reduced = _reduce_output_frames(targets, doa, vad, source_positions, stride)
    assert reduced[0].shape[0] == frames // stride == 23
    assert reduced[-1].tolist() == [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264]


def test_projection_rejects_input_shorter_than_one_stride_window():
    targets = torch.zeros(2, 4, 1, 1, 2)
    doa = torch.zeros(2, 1, 2)
    vad = torch.ones(2, 1)
    source_positions = torch.zeros(2, 1, 3)
    with pytest.raises(ValueError, match="at least one complete"):
        _reduce_output_frames(targets, doa, vad, source_positions, stride=3)


def test_collate_marks_padded_frames_invalid():
    short = project_ssl_scene(_scene(length=64), _config())
    long = project_ssl_scene(_scene(length=96), _config())
    batch = ssl_collate_fn([short, long])
    batch.validate()
    assert batch.valid_output_mask.dtype is torch.bool
    assert batch.valid_output_mask.shape == (2, long["output_lengths"])
    assert bool(batch.valid_output_mask[0, : short["output_lengths"]].all().item())
    assert not bool(batch.valid_output_mask[0, short["output_lengths"] :].any().item())
    assert bool(batch.valid_output_mask[1].all().item())
    assert batch.receptive_field == 1
    assert batch.algorithmic_latency == 0


def test_collate_keeps_explicit_timebase_scalars():
    first = project_ssl_scene(_scene(length=64), _config(output_stride=4))
    second = project_ssl_scene(_scene(length=96), _config(output_stride=4))
    batch = ssl_collate_fn([first, second])
    batch.validate()
    assert batch.time_stride == 4
    assert batch.receptive_field == 4
    assert batch.algorithmic_latency == 3


def _valid_batch():
    item = project_ssl_scene(_scene(), _config())
    return ssl_collate_fn([item])


def test_batch_validate_rejects_malformed_valid_output_mask():
    batch = _valid_batch()
    frames = batch.valid_output_mask.shape[1]
    with pytest.raises(ValueError, match="valid_output_mask"):
        replace(batch, valid_output_mask=None).validate()
    with pytest.raises(ValueError, match="valid_output_mask"):
        replace(batch, valid_output_mask=torch.ones(1, frames + 1, dtype=torch.bool)).validate()
    with pytest.raises(TypeError, match="valid_output_mask"):
        replace(batch, valid_output_mask=torch.ones(1, frames)).validate()
    # 补齐帧不允许标记为有效。
    padded = replace(batch, output_lengths=torch.tensor([frames - 1]))
    with pytest.raises(ValueError, match="valid_output_mask"):
        padded.validate()


def test_batch_validate_rejects_inconsistent_timebase_scalars():
    batch = _valid_batch()
    with pytest.raises(ValueError, match="receptive_field"):
        replace(batch, receptive_field=0).validate()
    with pytest.raises(ValueError, match="receptive_field"):
        replace(batch, receptive_field=2.0).validate()
    with pytest.raises(ValueError, match="algorithmic_latency"):
        replace(batch, algorithmic_latency=-1).validate()
    with pytest.raises(ValueError, match="algorithmic_latency"):
        replace(batch, algorithmic_latency=1, receptive_field=1).validate()


def test_collate_rejects_mixed_or_invalid_item_timebase():
    item = project_ssl_scene(_scene(), _config())
    bad_mask = dict(item)
    bad_mask["valid_output_mask"] = torch.zeros(item["output_lengths"], dtype=torch.bool)
    with pytest.raises(ValueError, match="valid_output_mask"):
        ssl_collate_fn([bad_mask])
    no_mask = {key: value for key, value in item.items() if key != "valid_output_mask"}
    with pytest.raises(ValueError, match="valid_output_mask"):
        ssl_collate_fn([item, no_mask])
    bad_rf = dict(item, receptive_field=2)
    with pytest.raises(ValueError, match="receptive_field"):
        ssl_collate_fn([item, bad_rf])
    bad_latency = dict(item, algorithmic_latency=1)
    with pytest.raises(ValueError, match="algorithmic_latency"):
        ssl_collate_fn([item, bad_latency])


def test_inactive_manifold_defaults_to_zero_and_is_sample_independent():
    # n_fft=16/hop=8、length=64 共 7 帧；前 32 个采样置零使前两帧 inactive。
    item = project_ssl_scene(_scene(silent_prefix=32), _config())
    inactive = item["vad"][:, 0] <= 0.5
    assert int(inactive.sum().item()) >= 2
    target = item["target_dpipd"]
    assert torch.isfinite(target).all()
    assert not target[inactive].any()
    # active 帧 target 保持单位模复互谱。
    modulus = target[~inactive].square().sum(dim=-1)
    torch.testing.assert_close(modulus, torch.ones_like(modulus))
    # manifold 不依赖 observation 的样本随机性。
    other = project_ssl_scene(_scene(silent_prefix=32, seed=7), _config())
    torch.testing.assert_close(target[inactive], other["target_dpipd"][inactive])


def test_inactive_manifold_bessel_j0_matches_official_form():
    config = _config(inactive_target_manifold=SSL_INACTIVE_MANIFOLD_BESSEL_J0)
    item = project_ssl_scene(_scene(silent_prefix=32), config)
    inactive = item["vad"][:, 0] <= 0.5
    assert int(inactive.sum().item()) >= 2
    target = item["target_dpipd"]
    assert torch.isfinite(target).all()
    # IPDnet/IPDnet2 官方形式：real 部 J0(2*pi*f*d/c)、imag 部零。
    pair_distance = 0.08
    freqs = np.arange(target.shape[1], dtype=np.float64) * (16000.0 / 16)
    expected_real = torch.from_numpy(
        j0(2 * np.pi * freqs * pair_distance / 343.0)
    ).to(torch.float32)
    inactive_target = target[inactive]  # [N,F,1,1,2]
    torch.testing.assert_close(
        inactive_target[..., 0],
        expected_real.view(1, -1, 1, 1).expand_as(inactive_target[..., 0]),
    )
    assert not inactive_target[..., 1].any()
    # 每个 inactive 帧的 manifold 是同一有限常数，与样本无关。
    for frame in inactive_target:
        torch.testing.assert_close(frame, inactive_target[0])
    modulus = target[~inactive].square().sum(dim=-1)
    torch.testing.assert_close(modulus, torch.ones_like(modulus))
    assert item["provenance"]["inactive_target_manifold"] == SSL_INACTIVE_MANIFOLD_BESSEL_J0
    batch = ssl_collate_fn([item])
    batch.validate()


def test_inactive_manifold_rejects_unknown_value():
    with pytest.raises(ValueError, match="inactive_target_manifold"):
        _config(inactive_target_manifold="random_fill")


def test_frames_without_valid_target_have_well_defined_pit_cost():
    # masked-loss 不变量：无有效 target（全 inactive）的帧整帧
    # 被剔除，不参与均值分母；代价有限、不产生 NaN/Inf。
    loss_fn = SSLDPIPDPITLoss(max_tracks=1)
    generator = torch.Generator().manual_seed(0)
    prediction = torch.randn(1, 3, 4, 1, 1, 2, generator=generator) * 0.1
    target = torch.randn(1, 3, 4, 1, 1, 2, generator=generator) * 0.1
    vad = torch.tensor([[[1.0], [1.0], [0.0]]])
    full = loss_fn(
        prediction,
        target,
        output_lengths=torch.tensor([3]),
        vad=vad,
    )
    assert torch.isfinite(full)
    truncated = loss_fn(
        prediction[:, :2],
        target[:, :2],
        output_lengths=torch.tensor([2]),
        vad=vad[:, :2],
    )
    # 第 3 帧 inactive，被整帧剔除，故与只保留前两帧的损失逐值相等。
    torch.testing.assert_close(full, truncated)


def test_all_inactive_frames_are_excluded_from_pit_cost():
    # 全部帧 inactive 时没有任何有效 target：代价必须是有限的零，而不是 NaN。
    loss_fn = SSLDPIPDPITLoss(max_tracks=1)
    generator = torch.Generator().manual_seed(1)
    prediction = torch.randn(1, 2, 4, 1, 1, 2, generator=generator)
    target = torch.randn(1, 2, 4, 1, 1, 2, generator=generator)
    loss = loss_fn(
        prediction,
        target,
        output_lengths=torch.tensor([2]),
        vad=torch.zeros(1, 2, 1),
    )
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.zeros_like(loss))


def _entry_config(**ssl_overrides):
    """加载官方 FN-SSL 合同模板；只做配置合并，不加载任何训练资源。"""
    from pathlib import Path

    from core.utils.config import load_merged_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_merged_config(root / "configs/ssl/fnssl_dynamic.yaml")
    cfg["ssl"].update(ssl_overrides)
    return cfg


def test_entry_propagates_inactive_manifold_to_projection_config():
    """ssl.inactive_target_manifold 必须穿过 entry 白名单并到达 projection 配置。"""
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    cfg = _entry_config(inactive_target_manifold=SSL_INACTIVE_MANIFOLD_BESSEL_J0)
    entry = SSLTrainingEntry("ssl_contract")
    entry._validate_config(cfg)
    section = SSLTrainingEntry._dataset_config(cfg, "train")
    assert section["inactive_target_manifold"] == SSL_INACTIVE_MANIFOLD_BESSEL_J0
    # section 中的值即 Dataset 转发的值：直接进入 projection 后 provenance 必须反映它。
    item = project_ssl_scene(
        _scene(silent_prefix=32),
        _config(inactive_target_manifold=section["inactive_target_manifold"]),
    )
    assert item["provenance"]["inactive_target_manifold"] == SSL_INACTIVE_MANIFOLD_BESSEL_J0


def test_entry_propagates_phase_sign_to_projection_provenance():
    """ssl.phase_sign 必须经 entry/dataset section 到达 projection provenance。"""
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    cfg = _entry_config(phase_sign=-1.0)
    section = SSLTrainingEntry._dataset_config(cfg, "train")
    assert section["phase_sign"] == -1.0
    item = project_ssl_scene(_scene(), _config(phase_sign=section["phase_sign"]))
    assert item["provenance"]["phase_sign"] == -1.0


def test_entry_rejects_unknown_inactive_manifold_value():
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    cfg = _entry_config(inactive_target_manifold="random_fill")
    with pytest.raises(ValueError, match="inactive_target_manifold.*zero or bessel_j0"):
        SSLTrainingEntry("ssl_contract")._validate_config(cfg)
    with pytest.raises(ValueError, match="inactive_target_manifold.*zero or bessel_j0"):
        SSLTrainingEntry._dataset_config(cfg, "train")


def test_dataset_rejects_unknown_inactive_manifold_before_loading_resources():
    """Dataset 构造期显式校验，未知值不得被静默忽略。"""
    from tasks.ssl.data.dynamic_dataset import SSLDynamicMixingDataset

    with pytest.raises(ValueError, match="inactive_target_manifold"):
        SSLDynamicMixingDataset(
            "missing.speech.scp",
            tracks=1,
            sample_rate=8000,
            inactive_target_manifold="random_fill",
        )


def test_entry_defaults_inactive_manifold_to_zero_with_identical_targets():
    """省略该键时默认 zero，且与显式 zero 生成逐位一致的 target。"""
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    cfg = _entry_config()
    section = SSLTrainingEntry._dataset_config(cfg, "train")
    assert section["inactive_target_manifold"] == SSL_INACTIVE_MANIFOLD_ZERO
    implicit = project_ssl_scene(
        _scene(silent_prefix=32),
        _config(inactive_target_manifold=section["inactive_target_manifold"]),
    )
    explicit = project_ssl_scene(
        _scene(silent_prefix=32),
        _config(inactive_target_manifold=SSL_INACTIVE_MANIFOLD_ZERO),
    )
    assert torch.equal(implicit["target_dpipd"], explicit["target_dpipd"])
