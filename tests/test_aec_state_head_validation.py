"""AEC state-head 模型完整 validation epoch 测试。

回归目标：``AECTrainer._validation_epoch`` 的 state-detector 统计路径
（``compute_state_counts`` 调用处）曾因变量名 ``num_classes`` 未定义
（应为 ``num_state_classes``）在带 state head 的模型（如 DBGNet）上必抛
NameError。本测试用带 state head 的极简模型跑完整 validation epoch，
断言不抛异常且返回有限指标。
"""

from __future__ import annotations
import pytest

import logging
import math
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
    """关闭并移除指向 exp_dir 下 train.log 的 FileHandler，避免临时目录清理失败。"""
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


class _TinyStateAECModel(nn.Module):
    """带 state head 的极简 spec-in/spec-out AEC 模型。"""

    state_nclass = 3

    def __init__(self, n_fft: int = 512) -> None:
        super().__init__()
        self.n_fft = n_fft
        # 输入：mic_spectra 与 farend_spectra 在最后一维拼接 -> (B, F, T, 4)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 2, kernel_size=3, padding=1),
        )
        # state head：频率维平均后做帧级分类 -> (B, T, state_nclass)
        self.state_head = nn.Linear(8, self.state_nclass)

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mic = data["mic_spectra"]          # (B, F, T, 2)
        ref = data["farend_spectra"]       # (B, F, T, 2)
        x = torch.cat([mic, ref], dim=-1).permute(0, 3, 1, 2)  # (B, 4, F, T)
        feat = self.conv[0](x)                              # (B, 8, F, T)
        out = self.conv[1:](feat).permute(0, 2, 3, 1)       # (B, F, T, 2)
        state_probs = self.state_head(feat.mean(dim=2).transpose(1, 2))  # (B, T, C)
        return {"nearend_estimate_spectra": out, "state_probs": state_probs}


class _SimpleAECLoss(nn.Module):
    """吸收 CompositeLoss 额外 kwargs 的简单 MSE loss。"""

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.mse_loss(enhanced, clean)


class _SyntheticStateAECDataset(Dataset):
    """返回合成 (nearend, mic, farend, state) 的 Dataset。"""

    def __init__(self, n_samples: int = 2, sr: int = 16000, length: int = 1024, n_state: int = 5) -> None:
        self.n_samples = n_samples
        self.sr = sr
        self.length = length
        self.n_state = n_state

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        torch.manual_seed(42 + idx)
        farend = torch.randn(self.length) * 0.3
        nearend = torch.randn(self.length) * 0.2
        mic = farend * 0.3 + nearend
        state = torch.randint(0, _TinyStateAECModel.state_nclass, (self.n_state,))
        return AECBatchItem(
            nearend=nearend,
            mic=mic,
            farend=farend,
            fs=self.sr,
            length=self.length,
            state=state,
        )


def _collate_aec_batch(items):
    """将 AECBatchItem 列表拼成 AECBatch（含 state 标签）。"""
    nearend = torch.stack([it.nearend for it in items])
    mic = torch.stack([it.mic for it in items])
    farend = torch.stack([it.farend for it in items])
    fs = torch.tensor([it.fs for it in items], dtype=torch.long)
    lengths = torch.tensor([it.length for it in items], dtype=torch.long)
    state = torch.stack([it.state for it in items])
    return AECBatch(nearend=nearend, mic=mic, farend=farend, fs=fs, lengths=lengths, state=state)


def test_state_head_validation_epoch_runs():
    """带 state head 的模型跑完整 validation epoch：不抛异常且返回有限 loss。"""
    tmpdir = tempfile.mkdtemp()
    try:
        exp_dir = Path(tmpdir)
        sr = 16000
        n_fft = 512
        hop_length = 256
        stft_cfg = {"n_fft": n_fft, "hop_length": hop_length, "win_length": n_fft, "window": "hann"}
        stft_module = build_stft_istft(stft_cfg)

        config = {
            "model": {"type": "TinyStateAECModel", "args": {"n_fft": n_fft}},
            "stft": stft_cfg,
            "trainer": {
                "epochs": 1,
                "max_steps": 4,
                "gradient_clip_value": 5.0,
                "accumulation_steps": 1,
                "validation_epoch_interval": 0,
                "log_interval": 1,
                "exp_path": str(exp_dir),
                "max_val_batches": 1,
            },
            "meta": {"seed": 42},
            "data": {"placeholder_num_samples": 1024},
        }

        model = _TinyStateAECModel(n_fft=n_fft)
        loss_fn = _SimpleAECLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        val_ds = _SyntheticStateAECDataset(n_samples=2)
        val_loader = DataLoader(val_ds, batch_size=2, collate_fn=_collate_aec_batch)

        trainer = AECTrainer(
            rank=0,
            world_size=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            config=config,
            train_loader=val_loader,
            val_loader=val_loader,
            sample_rate=sr,
            stft_module=stft_module,
        )

        avg = trainer._validation_epoch(epoch=0)

        assert avg is not None, "validation returned None"
        assert math.isfinite(avg), f"validation loss is non-finite: {avg}"
    finally:
        _close_log_handlers(Path(tmpdir))
        shutil.rmtree(str(tmpdir), ignore_errors=True)


if __name__ == "__main__":
    test_state_head_validation_epoch_runs()
    print("AEC state-head validation test passed.")

pytestmark = [pytest.mark.unit]
