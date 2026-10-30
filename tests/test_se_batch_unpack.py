"""Tests for SE batch unpacking and SEBatch dataclass."""
import numpy as np
import pytest

import torch

from core.data.collate import collate_fn
from tasks.se.data.batch import SEBatch
from tasks.se.trainer.metrics_trainer import _unpack_se_batch


def _make_se_tuple(n_fields: int):
    t = 16000
    clean = torch.randn(t)
    noisy = torch.randn(t)
    sr = 16000
    length = t
    if n_fields == 4:
        return (clean, noisy, sr, length)
    if n_fields == 7:
        return (clean, noisy, sr, length, False, False, False)
    raise ValueError(n_fields)


def test_unpack_collated_sample_formats():
    """4/7 元组只用于样本；Trainer 直接消费已校验的 SEBatch。"""
    for n_fields in (4, 7):
        sample = _make_se_tuple(n_fields)
        batch = collate_fn([sample])
        out = _unpack_se_batch(batch)
        assert isinstance(out, SEBatch), n_fields
        assert out is batch
        torch.testing.assert_close(out.clean[0], sample[0])
        torch.testing.assert_close(out.noisy[0], sample[1])
        assert out.sr == sample[2]
        assert out.lengths == sample[3]


@pytest.mark.parametrize("n_fields", [5, 6], ids=("legacy-5-tuple", "6-tuple"))
def test_unpack_rejects_unsupported_tuple_lengths(n_fields):
    """仅 is_pure_noise 的 5 元组旧格式已移除，其他非法长度同样显式拒绝。"""
    sample = _make_se_tuple(4) + (False,) * (n_fields - 4)
    with pytest.raises(TypeError, match="requires an SEBatch"):
        _unpack_se_batch(sample)


@pytest.mark.parametrize(
    ("sample_flags", "expected_flags"),
    [
        ((), None),
        ((False, True, True), (False, True, True)),
    ],
    ids=("4-tuple", "7-tuple"),
)
def test_collate_to_consumer_preserves_sample_fields(
    sample_flags, expected_flags
):
    """样本字段在 collate 后以 SEBatch 按名消费。"""
    clean = torch.tensor([0.1, -0.2, 0.3, 0.4])
    noisy = torch.tensor([0.5, -0.6, 0.7, 0.8])
    sample = (clean, noisy, 16000, 3, *sample_flags)

    collated = collate_fn([sample])
    batch = _unpack_se_batch(collated)

    assert isinstance(collated, SEBatch)
    assert batch is collated
    torch.testing.assert_close(batch.clean, torch.tensor([[0.1, -0.2, 0.3]]))
    torch.testing.assert_close(batch.noisy, torch.tensor([[0.5, -0.6, 0.7]]))
    assert batch.sr.dtype == torch.int32
    assert batch.sr.item() == 16000
    assert batch.lengths.dtype == torch.int32
    assert batch.lengths.tolist() == [3]

    actual_flags = (
        batch.is_pure_noise,
        batch.is_pure_echo,
        batch.is_pure_echo_noise,
    )
    if expected_flags is None:
        assert actual_flags == (None, None, None)
    else:
        for actual, expected in zip(actual_flags, expected_flags):
            assert actual.dtype == torch.bool
            assert actual.tolist() == [expected]


@pytest.mark.parametrize(
    "shape",
    [(4,), (1, 4), (4, 1)],
    ids=("T", "1xT", "Tx1"),
)
def test_collate_accepts_only_supported_mono_layouts(shape):
    """公共入口只规范化三种明确的 mono 布局。"""
    clean = torch.arange(4, dtype=torch.float32).reshape(shape)
    noisy = (clean + 1).clone()

    batch = collate_fn([(clean, noisy, 16000, 4)])

    assert batch.clean.shape == batch.noisy.shape == (1, 4)
    torch.testing.assert_close(batch.clean[0], torch.arange(4, dtype=torch.float32))


@pytest.mark.parametrize(
    "shape",
    [(0,), (1, 0), (0, 1), (2, 4), (4, 2), (1, 1, 4)],
    ids=("empty-T", "empty-1xT", "empty-Tx1", "CxT", "TxC", "high-dim"),
)
def test_collate_rejects_unsupported_waveform_layouts(shape):
    """空波形、多通道与高维输入必须失败，禁止静默混音或降维。"""
    wav = torch.empty(shape)
    with pytest.raises(RuntimeError, match="mono waveform"):
        collate_fn([(wav, wav, 16000, 1)])


def test_collate_zeros_each_sample_tail_after_declared_length():
    """预填充 waveform 在 declared length 之后不得泄漏到训练 batch。"""
    first = torch.tensor([1.0, 2.0, 91.0, 92.0])
    second = torch.tensor([3.0, 4.0, 5.0])

    batch = collate_fn(
        [(first, first, 16000, 2), (second, second, 16000, 3)]
    )

    torch.testing.assert_close(batch.lengths, torch.tensor([2, 3], dtype=torch.int32))
    torch.testing.assert_close(
        batch.clean,
        torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]]),
    )
    torch.testing.assert_close(batch.noisy, batch.clean)


@pytest.mark.parametrize("integer_type", [np.int32, np.int64])
def test_collate_accepts_numpy_integer_sr_and_length_scalars(integer_type):
    """SE 旧 tuple 的 NumPy 整数 SR/length 必须保持兼容。"""
    batch = collate_fn(
        [(torch.ones(4), torch.ones(4), integer_type(16000), integer_type(3))]
    )

    assert batch.clean.shape == batch.noisy.shape == (1, 3)
    assert batch.sr.item() == 16000
    assert batch.lengths.tolist() == [3]


@pytest.mark.parametrize("value", [True, 16000.0, "16000", torch.tensor([16000, 16000])])
def test_collate_rejects_non_integer_sr_and_length_scalars(value):
    """SR/length 仅接受整数标量或单元素整型 tensor。"""
    sample = (torch.ones(4), torch.ones(4), value, 4)
    with pytest.raises(TypeError, match="integer scalar"):
        collate_fn([sample])
    sample = (torch.ones(4), torch.ones(4), 16000, value)
    with pytest.raises(TypeError, match="integer scalar"):
        collate_fn([sample])


@pytest.mark.parametrize("batch", [[None], [None, _make_se_tuple(4)]])
def test_collate_fails_fast_on_none_samples(batch):
    """None 表示上游加载失败，不得静默缩小或伪造 batch。"""
    with pytest.raises(RuntimeError, match="None"):
        collate_fn(batch)


@pytest.mark.parametrize(
    "flag",
    [torch.tensor([True, False]), [True], (False,)],
    ids=("tensor", "list", "tuple"),
)
def test_collate_rejects_non_scalar_flags(flag):
    """门控 flag 必须是标量，禁止隐式 any 或容器 truthiness。"""
    sample = (torch.ones(4), torch.ones(4), 16000, 4, flag, False, False)
    with pytest.raises(TypeError, match="scalar"):
        collate_fn([sample])


def test_collate_accepts_binary_numeric_scalar_flags():
    """7 元组接受有限数值标量 0/1，并统一输出 bool tensor。"""
    sample = (
        torch.ones(4), torch.ones(4), 16000, 4,
        torch.tensor(1.0), 0.0, torch.tensor(1),
    )

    collated = collate_fn([sample])

    assert isinstance(collated, SEBatch)
    flags = (collated.is_pure_noise, collated.is_pure_echo, collated.is_pure_echo_noise)
    for actual, expected in zip(flags, (True, False, True)):
        assert actual.dtype == torch.bool
        assert actual.tolist() == [expected]


@pytest.mark.parametrize(
    "flag",
    ["1", float("nan"), float("inf"), float("-inf"), -1, 2],
    ids=("string", "nan", "pos-inf", "neg-inf", "minus-one", "two"),
)
def test_collate_rejects_non_binary_scalar_flags(flag):
    """7 元组拒绝字符串、非有限值及 0/1 以外的数值。"""
    sample = (torch.ones(4), torch.ones(4), 16000, 4, False, False, flag)

    with pytest.raises((TypeError, ValueError), match="finite numeric 0/1"):
        collate_fn([sample])


@pytest.mark.parametrize("n_fields", [5, 6, 8], ids=("legacy-5-tuple", "6-tuple", "8-tuple"))
def test_collate_rejects_unsupported_tuple_lengths(n_fields):
    """仅 is_pure_noise 的 5 元组旧格式已移除，其他非法长度同样显式拒绝。"""
    sample = _make_se_tuple(4) + (False,) * (n_fields - 4)
    with pytest.raises(TypeError, match="must be 4 or 7"):
        collate_fn([sample])


def test_unpack_accepts_se_batch():
    """Passing an SEBatch through _unpack_se_batch returns the same object."""
    batch = collate_fn([_make_se_tuple(7)])
    assert _unpack_se_batch(batch) is batch


@pytest.mark.parametrize(
    ("n_fields", "expected_keys"),
    [
        (4, {"clean", "noisy", "fs", "lengths"}),
        (
            7,
            {
                "clean", "noisy", "fs", "lengths",
                "is_pure_noise", "is_pure_echo", "is_pure_echo_noise",
            },
        ),
    ],
)
def test_gan_unpack_batch_reuses_the_se_batch_contract(n_fields, expected_keys):
    """GAN 循环不得再维护第二份 SE 批合同实现。"""
    from tasks.se.trainer.gan_trainer import GANTrainer

    trainer = object.__new__(GANTrainer)
    unpacked = trainer._unpack_batch(collate_fn([_make_se_tuple(n_fields)]))

    assert set(unpacked) == expected_keys
    assert unpacked["fs"] == 16000


def test_gan_unpack_batch_accepts_se_batch_and_rejects_other_lengths():
    from tasks.se.trainer.gan_trainer import GANTrainer

    trainer = object.__new__(GANTrainer)
    se_batch = collate_fn([_make_se_tuple(4)])
    assert trainer._unpack_batch(se_batch)["clean"] is se_batch.clean

    with pytest.raises(TypeError, match="requires an SEBatch"):
        trainer._unpack_batch(_make_se_tuple(4) + (False,) * 4)


def test_collate_explicitly_preserves_spatial_noisy_and_mono_target():
    clean_a = torch.arange(5, dtype=torch.float32)
    clean_b = torch.arange(6, dtype=torch.float32)
    noisy_a = torch.stack((clean_a, clean_a + 10.0))
    noisy_b = torch.stack((clean_b, clean_b + 10.0))

    with pytest.raises(RuntimeError, match="mono waveform"):
        collate_fn([(clean_a, noisy_a, 16000, 4)])

    batch = collate_fn(
        [
            (clean_a, noisy_a, 16000, 4),
            (clean_b, noisy_b, 16000, 5),
        ],
        allow_spatial_noisy=True,
    )

    assert batch.clean.shape == (2, 5)
    assert batch.noisy.shape == (2, 2, 5)
    torch.testing.assert_close(batch.clean[0], torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0]))
    torch.testing.assert_close(batch.noisy[0, 1], torch.tensor([10.0, 11.0, 12.0, 13.0, 0.0]))
    torch.testing.assert_close(batch.lengths, torch.tensor([4, 5], dtype=torch.int32))


def test_collate_rejects_mixed_spatial_channel_counts():
    clean = torch.ones(4)
    with pytest.raises(RuntimeError, match="channel count"):
        collate_fn(
            [
                (clean, torch.ones(2, 4), 16000, 4),
                (clean, torch.ones(3, 4), 16000, 4),
            ],
            allow_spatial_noisy=True,
        )


pytestmark = [pytest.mark.unit]
