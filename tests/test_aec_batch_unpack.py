"""Tests for AEC batch unpacking and collate/unpack contract."""
import pytest

import torch

from tasks.aec.data.batch import AECBatch
from tasks.aec.data.collate import aec_collate_fn
from tasks.aec.trainer.aec_trainer import _unpack_aec_batch


def _make_tensors(
    n_fields: int,
    with_state: bool = False,
    state_is_2d: bool = False,
    state_len: int = 63,
):
    """Build a single sample tuple with the requested number of fields."""
    t = 16000
    nearend = torch.randn(t)
    mic = torch.randn(t)
    farend = torch.randn(t)
    fs = 16000
    length = t
    err = torch.randn(t)
    err_mask = True
    pn = pe = pen = he = False

    if n_fields == 7:
        return (nearend, mic, farend, fs, length, err, err_mask)
    if n_fields == 11:
        return (nearend, mic, farend, fs, length, err, err_mask, pn, pe, pen, he)
    if n_fields == 12:
        state = (
            torch.randn(state_len, 3)
            if state_is_2d
            else torch.full((state_len,), 0, dtype=torch.long)
        )
        return (nearend, mic, farend, fs, length, err, err_mask, pn, pe, pen, he, state)
    raise ValueError(n_fields)


def _make_removed_tensors(n_fields: int):
    """构造已移除格式，用于验证边界显式拒绝。"""
    base = _make_tensors(12)
    if n_fields == 13:
        return base + (torch.ones(16000, dtype=torch.bool),)
    if 5 <= n_fields <= 10:
        return base[:n_fields]
    raise ValueError(n_fields)


def test_unpack_collated_sample_formats():
    """11/12 元组样本先 collate，Trainer 只接受 AECBatch。"""
    for n_fields in (11, 12):
        sample = _make_tensors(n_fields)
        batch = aec_collate_fn([sample])
        out = _unpack_aec_batch(batch)
        assert isinstance(out, AECBatch), n_fields
        assert out is batch
        torch.testing.assert_close(out.nearend[0], sample[0])
        torch.testing.assert_close(out.mic[0], sample[1])
        torch.testing.assert_close(out.farend[0], sample[2])
        if n_fields == 12:
            torch.testing.assert_close(out.state[0], sample[11])
        else:
            assert out.state is None


@pytest.mark.parametrize("n_fields", [5, 6, 7, 10, 13])
def test_unpack_rejects_removed_lengths(n_fields):
    """Trainer 边界显式拒绝已移除的历史元组长度。"""
    sample = _make_removed_tensors(n_fields)

    with pytest.raises(TypeError, match="requires an AECBatch"):
        _unpack_aec_batch(sample)


def test_unpack_rejects_unknown_14_tuple():
    """trainer 不得静默截断未来尾部字段。"""
    sample = _make_tensors(12) + (torch.ones(1), torch.ones(1))

    with pytest.raises(TypeError, match="requires an AECBatch"):
        _unpack_aec_batch(sample)


def test_collate_and_unpack_round_trip():
    """collate_fn output can be consumed directly by _unpack_aec_batch."""
    s1 = _make_tensors(12)
    s2 = _make_tensors(12)
    batch = aec_collate_fn([s1, s2])
    assert isinstance(batch, AECBatch)
    unpacked = _unpack_aec_batch(batch)
    assert isinstance(unpacked, AECBatch)
    assert unpacked is batch
    assert isinstance(unpacked.state, torch.Tensor)


def test_aec_batch_state_survives_transfer_without_optional_fields():
    sample = _make_tensors(12)
    state = sample[11].unsqueeze(0)
    structured = AECBatch(
        nearend=sample[0].unsqueeze(0),
        mic=sample[1].unsqueeze(0),
        farend=sample[2].unsqueeze(0),
        fs=torch.tensor(sample[3], dtype=torch.int32),
        lengths=torch.tensor([sample[4]], dtype=torch.int32),
        state=state,
    )

    structured.validate()
    restored = structured.to("cpu")
    restored.validate()
    assert restored.error is None
    assert restored.is_pure_noise is None
    assert restored.state is state


def test_collate_rejects_multilabel_state_attributes():
    """当前 softmax head/loss/metrics 不接受 ``(T, 3)`` multi-label。"""
    sample = _make_tensors(12, state_is_2d=True, state_len=20)

    with pytest.raises(ValueError, match="multi-label attributes are unsupported"):
        aec_collate_fn([sample])


@pytest.mark.parametrize(
    "invalid_state",
    [None, torch.zeros(10), torch.zeros(10, 2, dtype=torch.int64)],
)
def test_collate_rejects_invalid_declared_state_schema(invalid_state):
    sample = list(_make_tensors(12))
    sample[11] = invalid_state

    with pytest.raises(ValueError, match=r"batch\[0\].*state"):
        aec_collate_fn([tuple(sample)])


def test_collate_rejects_mixed_state_schemas_before_padding():
    class_state = _make_tensors(12, state_len=20)
    attribute_state = _make_tensors(12, state_is_2d=True, state_len=20)

    with pytest.raises(ValueError, match="multi-label attributes are unsupported"):
        aec_collate_fn([class_state, attribute_state])


def test_collate_all_failed_returns_placeholder():
    """When every sample is None, collate returns a silent placeholder batch."""
    batch = aec_collate_fn([None, None], placeholder_samples=8000)
    assert isinstance(batch, AECBatch)
    assert batch.state is None
    assert batch.nearend.shape == (2, 8000)
    assert batch.lengths.tolist() == [8000, 8000]


def test_collate_rejects_partial_none_without_shrinking_batch():
    """部分失败样本不得被过滤，避免 DDP rank-local batch size 分叉。"""
    with pytest.raises(RuntimeError, match="refusing to shrink"):
        aec_collate_fn([None, _make_tensors(11)])


def test_collate_supported_lengths_without_state():
    """7/11 元组样本产出无 state 的 AECBatch。"""
    for n_fields in (7, 11):
        batch = aec_collate_fn([_make_tensors(n_fields), _make_tensors(n_fields)])
        assert isinstance(batch, AECBatch), n_fields
        he = batch.has_echo
        assert he.dtype == torch.bool
        assert not he.any(), n_fields
        # collate 输出必须能被 trainer 侧 unpack 直接消费
        unpacked = _unpack_aec_batch(batch)
        assert isinstance(unpacked, AECBatch)
        assert unpacked.state is None


@pytest.mark.parametrize("n_fields", [5, 6, 8, 9, 10, 13])
def test_collate_rejects_removed_lengths(n_fields):
    """collate 侧显式拒绝已移除的历史元组长度。"""
    sample = _make_removed_tensors(n_fields)

    with pytest.raises(ValueError, match=r"tuple length must be one of 7/11/12"):
        aec_collate_fn([sample])


def test_collate_accepts_declared_length_with_prefilled_waveforms():
    sample = list(_make_tensors(7))
    sample[0] = torch.randn(18)
    sample[1] = torch.randn(18)
    sample[2] = torch.randn(18)
    sample[4] = 10
    sample[5] = torch.randn(18)

    batch = aec_collate_fn([tuple(sample)])

    assert batch.nearend.shape == batch.mic.shape == batch.farend.shape == batch.error.shape == (1, 18)
    assert batch.lengths.tolist() == [10]


@pytest.mark.parametrize("declared", [0, -1, 11])
def test_collate_rejects_declared_length_outside_waveforms(declared):
    sample = list(_make_tensors(7))
    sample[0] = torch.randn(10)
    sample[1] = torch.randn(10)
    sample[2] = torch.randn(10)
    sample[4] = declared
    sample[5] = torch.randn(10)

    with pytest.raises(
        RuntimeError,
        match=r"invalid declared length=.*1 <= declared length <= waveform length",
    ):
        aec_collate_fn([tuple(sample)])


def test_collate_finite_scan_is_flag_gated():
    """check_finite=False 时跳过逐样本 isfinite 扫描，默认仍拦截 NaN。"""
    sample = list(_make_tensors(7))
    sample[0] = torch.full((16000,), float("nan"))

    with pytest.raises(RuntimeError, match="contains NaN/Inf"):
        aec_collate_fn([tuple(sample)])

    batch = aec_collate_fn([tuple(sample)], check_finite=False)

    assert not torch.isfinite(batch.nearend).all()


def test_aec_batch_optional_fields_remain_absent_during_transfer():
    device = torch.device("meta")
    batch = AECBatch(
        nearend=torch.empty(2, 8, device=device),
        mic=torch.empty(2, 8, device=device),
        farend=torch.empty(2, 8, device=device),
        fs=torch.empty((), dtype=torch.int32, device=device),
        lengths=torch.empty(2, dtype=torch.int32, device=device),
        is_pure_noise=torch.empty(2, dtype=torch.bool, device=device),
    )

    out = batch.to(device)
    assert out.is_pure_noise.device == device
    assert out.is_pure_noise.dtype == torch.bool
    assert out.error is out.error_mask is out.is_pure_echo is out.has_echo is None


pytestmark = [pytest.mark.unit]
