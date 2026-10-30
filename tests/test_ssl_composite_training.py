"""注册 SSL 模型的组合损失训练、验证和完整状态恢复验收。

使用动态仿真合成 fixture；组合损失采用 DP-IPD 默认权重，不为官方模型
添加辅助 head。本文件覆盖 CPU 和 CUDA 生命周期，不评价模型收敛或论文指标。
"""

from copy import deepcopy
import logging
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from core.losses.build import build_loss
from tasks.ssl.data import ssl_collate_fn
from tasks.ssl.losses import SSLDOACompositeLoss
from tasks.ssl.model_io import validate_ssl_loss_output_contract
from tasks.ssl.models import build_model
from tasks.ssl.trainer import SSLTrainer
from test_ssl_fnssl_smoke import _fnssl_dataset
from test_ssl_official_model_smoke import (
    _dataset,
    _ipdnet_fixed_cfg,
    _ipdnet_variable_cfg,
    _ipdnet2_cfg,
    _ipdnet2_stft_cfg,
)
from test_ssl_trainer_loop import _dynamic_dataset


_MODEL_VARIANTS = (
    "native",
    "fnssl",
    "fixed_online",
    "fixed_offline",
    "variable_online",
    "variable_offline",
)
_STFT = {"n_fft": 16, "hop_length": 8, "win_length": 16, "window": "hamming", "center": False}
_LOSS = {"type": "SSLDOACompositeLoss", "strict": True, "args": {"max_tracks": 2}}


def _training_case(tmp_path, variant):
    if variant == "native":
        return _dynamic_dataset(tmp_path), {
            "type": "FixedArrayIPDNet",
            "args": {"input_size": 1, "hidden_size": 4, "max_track": 2},
        }, _STFT
    if variant == "fnssl":
        return _fnssl_dataset(tmp_path), {
            "type": "FNSSL",
            "args": {
                "input_size": 1, "hidden_size": 8, "max_track": 2,
                "n_fft": 16, "stride": 12, "causal": False,
            },
        }, _STFT
    if variant == "ipdnet2":
        return _dataset(
            tmp_path, frames=60, output_stride=5, exclude_dc=True,
            normalization="forgetting_laplace", n_fft=256,
        ), _ipdnet2_cfg(), _ipdnet2_stft_cfg()
    causal = variant.endswith("_online")
    model_cfg = (
        _ipdnet_fixed_cfg(causal=causal)
        if variant.startswith("fixed_")
        else _ipdnet_variable_cfg(causal=causal)
    )
    dataset = _dataset(
        tmp_path, frames=24, output_stride=12, exclude_dc=False,
        normalization="none" if causal else "global_mean",
    )
    return dataset, model_cfg, _STFT


def _trainer(tmp_path, loader, model_cfg, stft_cfg, *, epochs, resume_from=None):
    model = build_model(deepcopy(model_cfg), stft_cfg=stft_cfg)
    validate_ssl_loss_output_contract(
        model, _LOSS, target_representation="dpipd_far_field"
    )
    loss = build_loss(deepcopy(_LOSS), stft_cfg=stft_cfg, sample_rate=8000)
    assert isinstance(getattr(loss, "loss", loss), SSLDOACompositeLoss)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    config = {
        "meta": {"allow_no_stft": True, "seed": 7},
        "model": deepcopy(model_cfg),
        "data": {"sample_rate": 8000},
        "ssl": {"num_tracks": 2, "vad_mode": "direct_path"},
        "loss": deepcopy(_LOSS),
        "trainer": {
            "epochs": epochs,
            "exp_path": str(tmp_path / "experiment"),
            "save_checkpoint_interval": 1,
            "validation_epoch_interval": 1,
            "max_val_batches": 1,
            "strict_checks": True,
            "val_metrics": {"enabled": False},
        },
    }
    return SSLTrainer(
        train_loader=loader, val_loader=loader, train_dataset=loader.dataset,
        sample_rate=8000, stft_module=None, rank=0, world_size=1,
        model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss,
        config=config, resume=resume_from is not None,
        resume_from=str(resume_from) if resume_from is not None else None,
    )


def _check_training_and_resume(tmp_path, variant, device):
    torch.manual_seed(7)
    dataset, model_cfg, stft_cfg = _training_case(tmp_path, variant)
    loader = DataLoader(dataset, batch_size=1, collate_fn=ssl_collate_fn)
    trainer = _trainer(tmp_path, loader, model_cfg, stft_cfg, epochs=1)
    assert trainer.device.type == device
    before = [value.detach().clone() for value in trainer.model.parameters()]
    trainer.train()
    assert trainer.n_steps == len(loader)
    assert trainer.curve_epochs == [1]
    assert torch.isfinite(torch.tensor(trainer.train_losses + trainer.val_losses)).all()
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, trainer.model.parameters())
    )
    checkpoint = trainer.ckpt_dir / "latest_model.tar"
    assert checkpoint.is_file()
    saved_weights = deepcopy(trainer._raw_model().state_dict())
    saved_optimizer = deepcopy(trainer.optimizer.state_dict())
    saved_scheduler = deepcopy(trainer.scheduler.state_dict())
    first_losses = (list(trainer.train_losses), list(trainer.val_losses))

    restored = _trainer(
        tmp_path, loader, model_cfg, stft_cfg, epochs=2, resume_from=checkpoint
    )
    assert restored.start_epoch == 2
    assert restored.n_steps == len(loader)
    torch.testing.assert_close(restored._raw_model().state_dict(), saved_weights, rtol=0, atol=0)
    torch.testing.assert_close(restored.optimizer.state_dict(), saved_optimizer, rtol=0, atol=0)
    assert restored.scheduler.state_dict() == saved_scheduler
    assert (restored.train_losses, restored.val_losses) == first_losses
    restored.train()
    assert restored.n_steps == 2 * len(loader)
    assert restored.curve_epochs == [1, 2]
    assert restored.train_losses[:1] == first_losses[0]
    assert restored.val_losses[:1] == first_losses[1]
    assert torch.isfinite(torch.tensor(restored.train_losses + restored.val_losses)).all()
    assert restored.scheduler.last_epoch == 2
    assert all(torch.isfinite(value).all() for value in restored.model.parameters())
    assert max(int(state["step"].item()) for state in restored.optimizer.state.values()) == restored.n_steps
    assert (restored.ckpt_dir / "latest_model.tar").is_file()


@pytest.fixture(autouse=True)
def _close_owned_logs(tmp_path):
    yield
    root_logger = logging.getLogger()
    for handler in tuple(root_logger.handlers):
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).is_relative_to(tmp_path):
            handler.close()
            root_logger.removeHandler(handler)


@pytest.mark.unit
@pytest.mark.parametrize("variant", _MODEL_VARIANTS)
def test_ssl_composite_cpu_training_and_full_resume(tmp_path, monkeypatch, variant):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _check_training_and_resume(tmp_path, variant, "cpu")


@pytest.mark.gpu
@pytest.mark.parametrize("variant", (*_MODEL_VARIANTS, "ipdnet2"))
def test_ssl_composite_cuda_training_and_full_resume(tmp_path, variant):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU acceptance")
    if variant == "ipdnet2":
        pytest.importorskip("mamba_ssm")
    torch.cuda.reset_peak_memory_stats()
    _check_training_and_resume(tmp_path, variant, "cuda")
    assert torch.cuda.max_memory_allocated() > 0
    print(f"[ssl-composite-resume] {variant} peak_cuda_bytes={torch.cuda.max_memory_allocated()}")
