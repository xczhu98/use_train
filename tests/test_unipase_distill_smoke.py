"""UniPASE Stage 1 (DeWavLM-Omni distill) smoke-test.

用 tiny WavLM（2 层 / 32 维，随机权重落盘为 fairseq 风格 ckpt）驱动真实的
``DeWavLMDistillModel(use_plc_frontend=True)``，验证：
  1. forward/backward 有限且只更新 student（teacher 冻结）；
  2. PLC 路径真正被触发——带 20ms 零化丢包区域的输入经
     ``detect_packet_loss_indices`` 检出非空 mask。

无需官方仓库 / 真实 WavLM-Large，CI 可直接运行。
"""

from __future__ import annotations
import pytest

import shutil
import tempfile
from pathlib import Path

import torch

from tasks.se.models.pase_family.common.dewavlm_distill import DeWavLMDistillModel
from tasks.se.models.pase_family.unipase.wavlm_frontend_plc import (
    detect_packet_loss_indices,
)

_TINY_WAVLM_CFG = dict(
    encoder_layers=2,
    encoder_embed_dim=32,
    encoder_ffn_embed_dim=64,
    encoder_attention_heads=2,
    layer_norm_first=True,
    gru_rel_pos=True,
    relative_position_embedding=True,
)


def _make_tiny_wavlm_ckpt(path: Path) -> None:
    """落盘一个 tiny WavLM ckpt（{"cfg", "model"}，与 WavLM_feat 的加载约定一致）。"""
    from tasks.se.models.pase_family.common.wavlm.WavLM import WavLM, WavLMConfig

    torch.manual_seed(0)
    model = WavLM(WavLMConfig(dict(_TINY_WAVLM_CFG)))
    torch.save({"cfg": dict(_TINY_WAVLM_CFG), "model": model.state_dict()}, path)


def _noisy_with_packet_loss(clean: torch.Tensor, fs: int = 16000) -> torch.Tensor:
    """在 (B, L) 波形上零化两个 20ms 包（模拟 Layer E packet_loss 的产物）。"""
    packet = int(fs * 20 / 1000)  # 320 samples @16k
    noisy = clean.clone()
    noisy[:, 2 * packet : 3 * packet] = 0.0
    noisy[:, 5 * packet : 6 * packet] = 0.0
    return noisy


def test_unipase_distill_plc_forward_backward():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        ckpt = tmpdir / "tiny_wavlm.pt"
        _make_tiny_wavlm_ckpt(ckpt)

        model = DeWavLMDistillModel(
            teacher_ckpt_path=str(ckpt),
            student_ckpt_path=str(ckpt),
            output_layer=2,
            use_plc_frontend=True,
        )
        model.train()

        torch.manual_seed(1)
        clean = torch.randn(2, 6400) * 0.3
        noisy = clean + torch.randn(2, 6400) * 0.05
        noisy = _noisy_with_packet_loss(noisy)

        # PLC 检测本身：零化的两个包必须被检出（pad 到 320 倍数+80 后尾部
        # 零包也会被检出，这里只断言我们零化的包在内）
        mask = detect_packet_loss_indices(noisy, fs=16000)
        assert mask is not None and mask[:, 2].all() and mask[:, 5].all()

        loss = model(noisy, clean, None, fs=16000)
        assert torch.isfinite(loss).item(), f"loss is non-finite: {loss}"
        loss.backward()

        student_grads = [p.grad for p in model.student.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in student_grads), (
            "student received no gradients"
        )
        assert all(not p.requires_grad for p in model.teacher.parameters()), (
            "teacher must stay frozen"
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_unipase_distill_plc_frontend_classes():
    """use_plc_frontend=True 时学生必须是 enable_plc 的 WavLM_feat_PLC。"""
    from tasks.se.models.pase_family.unipase.wavlm_frontend_plc import WavLM_feat_PLC

    tmpdir = Path(tempfile.mkdtemp())
    try:
        ckpt = tmpdir / "tiny_wavlm.pt"
        _make_tiny_wavlm_ckpt(ckpt)
        model = DeWavLMDistillModel(
            teacher_ckpt_path=str(ckpt),
            student_ckpt_path=str(ckpt),
            output_layer=2,
            use_plc_frontend=True,
        )
        assert isinstance(model.student, WavLM_feat_PLC)
        assert model.student.enable_plc is True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


pytestmark = [pytest.mark.external]
