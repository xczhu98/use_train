"""AEC 端到端训练 smoke-test。

用合成数据构造最小训练循环，验证重构后 AECTrainer 能完成
forward / backward / optimizer step / checkpoint save / resume。
"""

from __future__ import annotations
import pytest

import logging
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tasks.aec.data.batch import AECBatch, AECBatchItem
from tasks.aec.trainer.aec_trainer import AECTrainer
from core.stft import build_stft_istft


def _close_log_handlers(exp_dir: Path) -> None:
    """关闭并移除指向 exp_dir 下 train.log 的 FileHandler，避免 Windows 临时目录清理失败。"""
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                path = Path(h.baseFilename).resolve()
                if str(exp_dir.resolve()) in str(path):
                    h.close()
                    root.removeHandler(h)
            except Exception:
                pass


def _rmtree_ignore(path: Path) -> None:
    """尝试删除目录，失败则忽略（Windows 上日志文件句柄可能仍被占用）。"""
    shutil.rmtree(str(path), ignore_errors=True)


class _TinyAECModel(nn.Module):
    """极简 spec-in/spec-out AEC 模型，仅用于 smoke-test。"""

    def __init__(self, n_fft: int = 512) -> None:
        super().__init__()
        self.n_fft = n_fft
        # 输入：mic_spectra 与 farend_spectra 在最后一维拼接 -> (B, F, T, 4)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 2, kernel_size=3, padding=1),
        )

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mic = data["mic_spectra"]          # (B, F, T, 2)
        ref = data["farend_spectra"]       # (B, F, T, 2)
        x = torch.cat([mic, ref], dim=-1).permute(0, 3, 1, 2)  # (B, 4, F, T)
        out = self.conv(x).permute(0, 2, 3, 1)                # (B, F, T, 2)
        return {"nearend_estimate_spectra": out}


class _SimpleAECLoss(nn.Module):
    """吸收 CompositeLoss 额外 kwargs 的简单 MSE loss。"""

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.mse_loss(enhanced, clean)


class _RecordingAECLoss(_SimpleAECLoss):
    """记录 train/validation 传入的逐样本有效长度。"""

    def __init__(self) -> None:
        super().__init__()
        self.lengths_seen: list[torch.Tensor] = []

    def forward(self, enhanced, clean, lengths, **kwargs) -> torch.Tensor:
        self.lengths_seen.append(lengths.detach().clone())
        return super().forward(enhanced, clean, lengths, **kwargs)


class _NaNAECLoss(nn.Module):
    """返回 NaN 的 AEC loss，验证坏 batch 使用零图而不更新参数。"""

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.full((), float("nan"), device=enhanced.device, requires_grad=True)


class _SyntheticAECDataset(Dataset):
    """返回合成 (nearend, mic, farend) 波形对的 Dataset。"""

    def __init__(self, n_samples: int = 4, sr: int = 16000, length: int = 1024) -> None:
        self.n_samples = n_samples
        self.sr = sr
        self.length = length

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        torch.manual_seed(42 + idx)
        farend = torch.randn(self.length) * 0.3
        echo = torch.zeros_like(farend)
        delay = 256
        echo[delay:] = farend[:-delay] * 0.3
        nearend = torch.randn(self.length) * 0.2
        mic = echo + nearend + torch.randn(self.length) * 0.05
        return AECBatchItem(
            nearend=nearend,
            mic=mic,
            farend=farend,
            fs=self.sr,
            length=self.length,
        )


def _collate_aec_batch(items):
    """将 AECBatchItem 列表拼成 AECBatch。"""
    nearend = torch.stack([it.nearend for it in items])
    mic = torch.stack([it.mic for it in items])
    farend = torch.stack([it.farend for it in items])
    fs = torch.tensor([it.fs for it in items], dtype=torch.long)
    lengths = torch.tensor([it.length for it in items], dtype=torch.long)
    return AECBatch(nearend=nearend, mic=mic, farend=farend, fs=fs, lengths=lengths)


def _build_aec_trainer(
    exp_dir: Path, device: str = "cpu", loss_fn: nn.Module | None = None
) -> AECTrainer:
    """构造一个只跑少量 step 的 AECTrainer。"""
    sr = 16000
    n_fft = 512
    hop_length = 256
    stft_cfg = {"n_fft": n_fft, "hop_length": hop_length, "win_length": n_fft, "window": "hann"}
    stft_module = build_stft_istft(stft_cfg)

    config = {
        "model": {"type": "TinyAECModel", "args": {"n_fft": n_fft}},
        "stft": stft_cfg,
        "trainer": {
            "epochs": 1,
            "max_steps": 4,
            "gradient_clip_value": 5.0,
            "accumulation_steps": 1,
            "validation_epoch_interval": 0,
            "log_interval": 1,
            "strict_checks": False,
            "exp_path": str(exp_dir),
            "max_val_batches": 1,
        },
        "meta": {"seed": 42},
        "data": {"placeholder_num_samples": 1024},
    }

    model = _TinyAECModel(n_fft=n_fft)
    loss_fn = loss_fn or _SimpleAECLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    train_ds = _SyntheticAECDataset(n_samples=4)
    val_ds = _SyntheticAECDataset(n_samples=2)
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=_collate_aec_batch)
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=_collate_aec_batch)

    return AECTrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        sample_rate=sr,
        stft_module=stft_module,
    )


def test_aec_train_and_validation_pass_batch_lengths():
    """AEC train/validation 必须向 loss 透传同一组逐样本长度。"""
    tmpdir = tempfile.mkdtemp()
    try:
        loss_fn = _RecordingAECLoss()
        trainer = _build_aec_trainer(Path(tmpdir), loss_fn=loss_fn)
        batch = next(iter(trainer.train_loader))
        batch.lengths = torch.tensor([901, 733], dtype=torch.long)
        expected = batch.lengths.clone()
        batch = trainer._batch_to_device(batch)

        trainer._forward_and_compute_loss(batch, epoch=0, batch_idx=0, fs_val=16000)
        trainer._val_forward_loss(batch, epoch=0, n_loops=0)

        assert len(loss_fn.lengths_seen) == 2
        assert all(torch.equal(value.cpu(), expected) for value in loss_fn.lengths_seen)
    finally:
        _close_log_handlers(Path(tmpdir))
        _rmtree_ignore(Path(tmpdir))


def test_aec_train_two_steps():
    """跑两个训练 step，断言 loss 有限且模型参数被更新。"""
    tmpdir = tempfile.mkdtemp()
    try:
        exp_dir = Path(tmpdir)
        trainer = _build_aec_trainer(exp_dir)

        # 记录训练前参数
        param_before = next(trainer.model.parameters()).detach().clone()

        loss = trainer._train_epoch(epoch=0)

        assert loss is not None
        assert torch.isfinite(torch.tensor(loss)), f"AEC smoke loss is non-finite: {loss}"

        param_after = next(trainer.model.parameters()).detach().clone()
        assert not torch.allclose(param_before, param_after), "AEC model parameters did not update"
    finally:
        _close_log_handlers(Path(tmpdir))
        _rmtree_ignore(Path(tmpdir))


def test_aec_all_nan_loss_does_not_update_parameters():
    """坏 batch 的 NaN loss 必须走零图 fail-safe，不能更新参数或 step。"""
    tmpdir = tempfile.mkdtemp()
    try:
        trainer = _build_aec_trainer(Path(tmpdir), loss_fn=_NaNAECLoss())
        params_before = [p.detach().clone() for p in trainer.model.parameters()]
        n_steps_before = trainer.n_steps

        assert trainer._train_epoch(epoch=0) == 0.0
        assert all(
            torch.equal(before, after.detach())
            for before, after in zip(params_before, trainer.model.parameters())
        )
        assert trainer.n_steps == n_steps_before
    finally:
        _close_log_handlers(Path(tmpdir))
        _rmtree_ignore(Path(tmpdir))


def test_aec_checkpoint_save_resume():
    """保存 checkpoint 后能正确 resume。"""
    tmpdir = tempfile.mkdtemp()
    try:
        exp_dir = Path(tmpdir)
        trainer = _build_aec_trainer(exp_dir)
        trainer._train_epoch(epoch=0)
        trainer._save_checkpoint(epoch=0, is_best=False)

        ckpt_path = list((exp_dir / "checkpoints").rglob("*.tar"))
        assert len(ckpt_path) > 0, "AEC checkpoint not saved"

        # 重新构造 trainer 并 resume
        trainer2 = _build_aec_trainer(exp_dir)
        trainer2._resume_checkpoint()
        # 简单验证 resume 后模型与前一个 trainer 一致
        for p1, p2 in zip(trainer.model.parameters(), trainer2.model.parameters()):
            torch.testing.assert_close(p1, p2)
    finally:
        _close_log_handlers(Path(tmpdir))
        _rmtree_ignore(Path(tmpdir))


if __name__ == "__main__":
    test_aec_train_two_steps()
    test_aec_checkpoint_save_resume()
    print("AEC end-to-end smoke tests passed.")

pytestmark = [pytest.mark.integration]
