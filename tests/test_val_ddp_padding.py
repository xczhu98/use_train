"""_val_iter_is_ddp_padding 单测：DDP 尾部对齐复制批的识别语义。

GroupedBatchSampler 为 DDP 对齐在末尾复制 batch（forward 内的 buffer 广播要求
各 rank 迭代数一致）；val 循环对复制批只跑 forward、不计贡献。helper 依据
batch_sampler._last_iter_real_len 判断当前迭代是否为复制批。
"""
import pytest

from types import SimpleNamespace

from core.trainer.metrics_trainer import MetricsCapableTrainer


pytestmark = pytest.mark.unit


def _fake_trainer(real_len):
    batch_sampler = (
        SimpleNamespace(_last_iter_real_len=real_len) if real_len is not None else None
    )
    return SimpleNamespace(val_loader=SimpleNamespace(batch_sampler=batch_sampler))


def test_padding_iterations_detected():
    f = MetricsCapableTrainer._val_iter_is_ddp_padding
    self_ = _fake_trainer(3)
    assert [f(self_, i) for i in range(1, 7)] == [False, False, False, True, True, True]


def test_no_padding_info_means_no_padding():
    f = MetricsCapableTrainer._val_iter_is_ddp_padding
    # batch_sampler 缺失/无属性（非 GroupedBatchSampler）→ 恒 False，行为与改动前一致
    assert f(_fake_trainer(None), 5) is False
    assert f(SimpleNamespace(val_loader=SimpleNamespace(batch_sampler=SimpleNamespace())), 5) is False


def test_no_padding_needed_means_no_padding():
    f = MetricsCapableTrainer._val_iter_is_ddp_padding
    # real_len 等于迭代总数（各 rank 批数本就一致，未补齐）→ 恒 False
    self_ = _fake_trainer(10)
    assert all(not f(self_, i) for i in range(1, 11))
