"""A22 端到端测试：middle checkpoint 真正可 resume。

覆盖：
1. resume 到 (epoch E, batch B) 时在 epoch 内从 batch B 继续（跳过前 B 个 batch），
   global step（n_steps）连续，且与不间断训练的对照实例最终参数一致；
2. gradient accumulation 中途 crash 时 cursor 对齐到上一个 accumulation 边界，
   恢复后与对照实例参数仍一致；
3. 无 latest_model.tar 时自动发现最新 middle checkpoint（E/B 最大者）；
   latest_model.tar 存在时优先于 middle checkpoint（行为与旧版一致）；
4. 旧格式（只有 batch、无 batch_cursor）的 middle checkpoint 兼容；
5. CLI --resume-from 参数解析与向后兼容。

用合成数据 + 极简 SE 模型走真实 SETrainer / MetricsCapableTrainer._train_epoch
生产路径（含 batch 跳过逻辑），CPU 可跑。
"""

from __future__ import annotations
import pytest

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from core.utils.safe_load import safe_torch_load
from tasks.se.data.batch import SEBatch, SEBatchItem
from tasks.se.trainer.se_trainer import SETrainer


def _close_log_handlers(exp_dir: Path) -> None:
    """关闭指向 exp_dir 下 train.log 的 FileHandler，避免临时目录清理失败。"""
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                if str(exp_dir.resolve()) in str(Path(h.baseFilename).resolve()):
                    h.close()
                    root.removeHandler(h)
            except Exception as exc:  # 清理路径不能中断，但必须留下信号
                logging.getLogger(__name__).warning(
                    "关闭日志处理器失败 %s: %s", h.baseFilename, exc
                )


class _TinySEModel(nn.Module):
    """极简 wav-in/wav-out SE 模型（确定性，无随机算子）。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, noisy: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.conv(noisy.unsqueeze(1)).squeeze(1)


class _SimpleWavLoss(nn.Module):
    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.mse_loss(enhanced, clean)


class _SyntheticSEDataset(Dataset):
    """每个样本由 manual_seed(42+idx) 确定性生成；任意子集同 idx 内容一致。"""

    def __init__(self, n_samples: int, sr: int = 16000, length: int = 1024) -> None:
        self.n_samples = n_samples
        self.sr = sr
        self.length = length

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        torch.manual_seed(42 + idx)
        clean = torch.randn(self.length) * 0.3
        noisy = clean + torch.randn(self.length) * 0.1
        return SEBatchItem(clean=clean, noisy=noisy, sr=self.sr, length=self.length)


def _collate_se_batch(items):
    return SEBatch(
        clean=torch.stack([it.clean for it in items]),
        noisy=torch.stack([it.noisy for it in items]),
        sr=torch.tensor([it.sr for it in items], dtype=torch.long),
        lengths=torch.tensor([it.length for it in items], dtype=torch.long),
    )


class _RecordingSETrainer(SETrainer):
    """记录每个 epoch 实际参与训练的 batch 下标（用于断言跳过位置）。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.processed: list[tuple[int, int]] = []

    def _prepare_batch(self, raw_batch, epoch, batch_idx):
        self.processed.append((epoch, batch_idx))
        return super()._prepare_batch(raw_batch, epoch, batch_idx)


def _build_trainer(
    exp_dir: Path,
    n_train: int,
    *,
    seed: int = 42,
    accum: int = 1,
    resume: bool = False,
    resume_from: Path | None = None,
) -> _RecordingSETrainer:
    config = {
        "model": {"type": "TinySEModel", "args": {}},
        "trainer": {
            "epochs": 1,
            "gradient_clip_value": 5.0,
            "accumulation_steps": accum,
            "validation_epoch_interval": 0,
            "log_interval": 1,
            "exp_path": str(exp_dir),
            "max_val_batches": 1,
        },
        "meta": {"seed": seed},
        "data": {"placeholder_num_samples": 1024},
    }
    torch.manual_seed(seed)
    model = _TinySEModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    train_loader = DataLoader(
        _SyntheticSEDataset(n_train), batch_size=2, collate_fn=_collate_se_batch
    )
    val_loader = DataLoader(
        _SyntheticSEDataset(2), batch_size=2, collate_fn=_collate_se_batch
    )
    return _RecordingSETrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=_SimpleWavLoss(),
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        sample_rate=16000,
        resume=resume,
        resume_from=str(resume_from) if resume_from is not None else None,
    )


def test_middle_checkpoint_resume_matches_uninterrupted(tmp_path: Path) -> None:
    """跑 1 个 batch → 保存 middle ckpt → resume：从 batch 1 继续，参数与对照一致。"""
    # 阶段 1：trainer_a 只跑 batch 0（loader 限 2 样本 = 1 个 batch），保存 middle ckpt
    trainer_a = _build_trainer(tmp_path / "run_a", n_train=2)
    trainer_a._train_epoch(epoch=1)
    assert trainer_a.n_steps == 1
    trainer_a._save_middle_checkpoint(epoch=1, batch=0)
    mid = trainer_a.ckpt_dir / "model_E0001_B000000.pth"
    assert mid.exists()

    # 阶段 2：trainer_b 用完整 6 样本 loader（3 个 batch）从 middle ckpt resume
    trainer_b = _build_trainer(tmp_path / "run_b", n_train=6, resume=True, resume_from=mid)
    assert trainer_b.start_epoch == 1, "cursor>0 时应从 epoch E 继续，而非 E+1"
    assert trainer_b._resume_skip_batches == 1
    assert trainer_b.n_steps == 1, "global step 应连续"
    trainer_b._train_epoch(epoch=1)
    assert trainer_b.processed == [(1, 1), (1, 2)], "应跳过 batch 0，从 batch 1 继续"
    assert trainer_b.n_steps == 3

    # 对照：trainer_c 同 seed 不间断跑完整 epoch
    trainer_c = _build_trainer(tmp_path / "run_c", n_train=6)
    trainer_c._train_epoch(epoch=1)
    assert trainer_c.processed == [(1, 0), (1, 1), (1, 2)]
    assert trainer_c.n_steps == 3

    for pb, pc in zip(trainer_b.model.parameters(), trainer_c.model.parameters()):
        torch.testing.assert_close(pb, pc)
    _close_log_handlers(tmp_path)


def test_middle_checkpoint_cursor_aligns_to_accumulation_boundary(tmp_path: Path) -> None:
    """accum=2 时 crash 在 accumulation 中途：cursor 对齐到上一个边界，恢复后参数与对照一致。"""
    # trainer_a 跑完第一个 accumulation 窗口（batch 0,1 → 1 个 step）后停下，
    # 模拟"batch 2 已处理但窗口未 step"时 crash，保存 middle ckpt
    trainer_a = _build_trainer(tmp_path / "run_a", n_train=12, accum=2)
    trainer_a.max_steps = 1
    trainer_a._train_epoch(epoch=1)
    assert trainer_a.n_steps == 1
    trainer_a._save_middle_checkpoint(epoch=1, batch=2)
    mid = trainer_a.ckpt_dir / "model_E0001_B000002.pth"
    # cursor = ((2+1)//2)*2 = 2（对齐到上一个 accumulation 边界）
    state = safe_torch_load(
        mid,
        allow_unsafe=True,
        purpose="inspect middle checkpoint cursor in resume regression test",
    )
    assert state["batch_cursor"] == 2

    trainer_b = _build_trainer(tmp_path / "run_b", n_train=12, accum=2, resume=True, resume_from=mid)
    assert trainer_b._resume_skip_batches == 2
    assert trainer_b.start_epoch == 1
    assert trainer_b.n_steps == 1
    trainer_b._train_epoch(epoch=1)
    # 重做 batch 2（其窗口未 step），之后正常：batch 2..5
    assert trainer_b.processed == [(1, 2), (1, 3), (1, 4), (1, 5)]
    assert trainer_b.n_steps == 3

    # 对照：同 seed 不间断跑完整 epoch（6 batch / accum 2 = 3 step）
    trainer_c = _build_trainer(tmp_path / "run_c", n_train=12, accum=2)
    trainer_c._train_epoch(epoch=1)
    assert trainer_c.n_steps == 3
    for pb, pc in zip(trainer_b.model.parameters(), trainer_c.model.parameters()):
        torch.testing.assert_close(pb, pc)
    _close_log_handlers(tmp_path)


def test_max_steps_checkpoint_resumes_same_epoch_from_completed_boundary(
    tmp_path: Path,
) -> None:
    trainer = _build_trainer(tmp_path / "max_steps", n_train=6)
    trainer.max_steps = 1

    trainer.train()

    checkpoint = trainer.ckpt_dir / "latest_model.tar"
    state = safe_torch_load(
        checkpoint,
        allow_unsafe=True,
        purpose="inspect latest checkpoint cursor in resume regression test",
    )
    assert trainer.n_steps == 1
    assert trainer.processed == [(1, 0)]
    assert state["epoch"] == 1
    assert state["batch_cursor"] == 1

    resumed = _build_trainer(
        tmp_path / "resumed",
        n_train=6,
        resume=True,
        resume_from=checkpoint,
    )
    assert resumed.start_epoch == 1
    assert resumed._resume_skip_batches == 1
    _close_log_handlers(tmp_path)


def test_middle_checkpoint_interval_uses_completed_batch_count(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path / "middle_interval", n_train=6)
    trainer.save_batch_interval = 2
    saved_batches: list[int] = []
    trainer._save_middle_checkpoint = (
        lambda _epoch, batch: saved_batches.append(batch)
    )

    trainer._train_epoch(epoch=1)

    assert saved_batches == [1]
    _close_log_handlers(tmp_path)


def test_legacy_middle_checkpoint_without_cursor_key(tmp_path: Path) -> None:
    """旧格式 middle ckpt（只有 batch 无 batch_cursor）按 batch+1 对齐后恢复。"""
    trainer_a = _build_trainer(tmp_path / "run_a", n_train=2)
    trainer_a._train_epoch(epoch=1)
    trainer_a._save_middle_checkpoint(epoch=1, batch=0)
    mid = trainer_a.ckpt_dir / "model_E0001_B000000.pth"
    state = safe_torch_load(
        mid,
        allow_unsafe=True,
        purpose="edit legacy middle checkpoint in resume regression test",
    )
    del state["batch_cursor"]
    torch.save(state, mid)

    trainer_b = _build_trainer(tmp_path / "run_b", n_train=2, resume=True, resume_from=mid)
    assert trainer_b._resume_skip_batches == 1
    assert trainer_b.start_epoch == 1
    _close_log_handlers(tmp_path)


def test_resume_discovers_middle_checkpoint_and_latest_wins(tmp_path: Path) -> None:
    """无 latest_model.tar 时自动发现 E/B 最大的 middle ckpt；latest_model.tar 优先。"""
    exp_dir = tmp_path / "run"
    trainer_a = _build_trainer(exp_dir, n_train=8)
    trainer_a._train_epoch(epoch=1)
    trainer_a._save_middle_checkpoint(epoch=1, batch=0)
    trainer_a._save_middle_checkpoint(epoch=1, batch=2)

    # 无 latest_model.tar：resume=True 自动发现 B000002（E/B 最大），cursor=3
    trainer_b = _build_trainer(exp_dir, n_train=8, resume=True)
    assert trainer_b.ckpt_dir == trainer_a.ckpt_dir
    assert trainer_b.start_epoch == 1
    assert trainer_b._resume_skip_batches == 3

    # latest_model.tar 存在时优先（无 cursor → 从 epoch+1 开始，与旧行为一致）
    trainer_a._save_checkpoint(epoch=1)
    trainer_c = _build_trainer(exp_dir, n_train=8, resume=True)
    assert trainer_c.start_epoch == 2
    assert trainer_c._resume_skip_batches == 0
    _close_log_handlers(tmp_path)


def test_entry_parses_resume_from(monkeypatch) -> None:
    """CLI --resume-from 解析及向后兼容（缺省为 None）。"""
    from tasks.se.trainer.entry import SETrainingEntry

    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "-C", "cfg.yaml", "--resume-from", "/tmp/x/model_E0001_B000002.pth"],
    )
    args = SETrainingEntry("test").parse_args()
    assert args.resume_from == "/tmp/x/model_E0001_B000002.pth"

    monkeypatch.setattr("sys.argv", ["train.py", "-C", "cfg.yaml"])
    args = SETrainingEntry("test").parse_args()
    assert args.resume_from is None

pytestmark = [pytest.mark.integration]
