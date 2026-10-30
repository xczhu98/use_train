"""A17 回归测试：CUDA cache 清理策略（core/trainer/base_trainer.py）。

旧行为：每 epoch 结束与每次验证前无条件 ``synchronize + empty_cache + gc.collect``，
强制 host 等待所有在途 kernel，形成 GPU 空泡。修复后：
- 默认（``trainer.clear_cache_each_epoch=False``）：训练/验证路径不调用 ``empty_cache``；
- 捕获 ``torch.cuda.OutOfMemoryError`` 时清 cache 后向上抛；
- 显式开启 ``clear_cache_each_epoch=True`` 时恢复每 epoch 末 + 验证前清理。

测试通过 monkeypatch 强制走 CUDA 分支，CPU/GPU 环境均可运行。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from core.trainer.base_trainer import BaseTrainer


pytestmark = pytest.mark.unit


def _close_log_handlers(exp_dir: Path) -> None:
    """关闭指向 exp_dir 下 train.log 的 FileHandler，避免临时目录清理失败。"""
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                if str(exp_dir.resolve()) in str(Path(h.baseFilename).resolve()):
                    h.close()
                    root.removeHandler(h)
            except Exception:
                # 清理失败不应掩盖测试失败，但要留下可诊断的记录。
                logging.debug("failed to close log handler %s", h, exc_info=True)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _CachePolicyTrainer(BaseTrainer):
    """最小可运行 Trainer：train/val epoch 行为由外部注入。"""

    def __init__(self, *args, train_impl=None, val_impl=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_impl = train_impl
        self._val_impl = val_impl

    def _train_epoch(self, epoch, step_callback=None):
        if self._train_impl is not None:
            return self._train_impl(epoch)
        return 1.0

    def _validation_epoch(self, epoch):
        if self._val_impl is not None:
            return self._val_impl(epoch)
        return 0.5


def _build_trainer(tmp_path: Path, clear_cache: bool, **kwargs) -> _CachePolicyTrainer:
    config = {
        "model": {"type": "TinyModel", "args": {}},
        "trainer": {
            "epochs": 2,
            "exp_path": str(tmp_path),
            "log_interval": 1,
            "save_curve_interval": 0,
            "clear_cache_each_epoch": clear_cache,
        },
        "meta": {"seed": 42},
        "data": {},
    }
    torch.manual_seed(42)
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    return _CachePolicyTrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=nn.MSELoss(),
        config=config,
        **kwargs,
    )


@pytest.fixture
def empty_cache_counter(monkeypatch):
    """计数 torch.cuda.empty_cache 调用次数（仍执行真实调用，不改变数值行为）。"""
    calls = {"n": 0}
    real = torch.cuda.empty_cache

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "empty_cache", _counting)
    return calls


def _force_cuda_branch(monkeypatch) -> None:
    """强制走训练循环里的 CUDA 分支（需在 trainer 构造之后调用，避免影响 device 选择）。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)


def test_no_unconditional_empty_cache(tmp_path, monkeypatch, empty_cache_counter):
    """默认关闭：完整跑 2 个 epoch（含验证），empty_cache 一次都不调用。"""
    trainer = _build_trainer(tmp_path, clear_cache=False)
    _force_cuda_branch(monkeypatch)
    try:
        trainer.train()
    finally:
        _close_log_handlers(tmp_path)
    assert empty_cache_counter["n"] == 0


def test_clear_cache_each_epoch_enabled(tmp_path, monkeypatch, empty_cache_counter):
    """显式开启：每 epoch 末 + 每次验证前各清一次，2 个 epoch 共 4 次。"""
    trainer = _build_trainer(tmp_path, clear_cache=True)
    _force_cuda_branch(monkeypatch)
    try:
        trainer.train()
    finally:
        _close_log_handlers(tmp_path)
    assert empty_cache_counter["n"] == 4


@pytest.mark.parametrize("stage", ["train", "validation"])
def test_oom_triggers_cache_cleanup(tmp_path, monkeypatch, empty_cache_counter, stage):
    """OOM 路径：train/val 任一阶段抛 OutOfMemoryError 时清 cache 后向上抛。"""
    def _boom(epoch):
        raise torch.cuda.OutOfMemoryError("simulated OOM")

    impls = {"train_impl": _boom} if stage == "train" else {"val_impl": _boom}
    trainer = _build_trainer(tmp_path, clear_cache=False, **impls)
    _force_cuda_branch(monkeypatch)
    try:
        with pytest.raises(torch.cuda.OutOfMemoryError):
            trainer.train()
    finally:
        _close_log_handlers(tmp_path)
    assert empty_cache_counter["n"] == 1
