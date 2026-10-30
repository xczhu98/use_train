"""CompositeLoss speech-reference 分量的 DDP 全局均值回归测试。"""

from __future__ import annotations

import os
import types
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from conftest import spawn_ddp_workers
from core.losses.composite import CompositeLoss, StrictFiniteLossError
from core.losses.hybrid import GTCRNLogRatioLoss  # noqa: F401 — 触发 loss 注册
from core.losses.time_domain import L1WavLoss, SISNRLoss  # noqa: F401 — 触发 loss 注册
from core.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("_ValidElementMeanSpeechLoss")
class _ValidElementMeanSpeechLoss(nn.Module):
    """按所有有效 waveform 元素求均值的 speech-reference 测试 loss。"""

    REQUIRES_SPEECH_REFERENCE = True

    def speech_reference_reduction_mass(
        self,
        clean: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        if lengths is None:
            return torch.full(
                (clean.shape[0],),
                clean.shape[-1],
                device=clean.device,
                dtype=torch.float32,
            )
        return lengths.to(device=clean.device, dtype=torch.float32)

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        error = (enhanced - clean).square()
        if lengths is None:
            return error.mean()
        mask = (
            torch.arange(clean.shape[-1], device=clean.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        return (error * mask).sum() / mask.sum().clamp(min=1)


@LOSS_REGISTRY.register("_FractionalMassDDPSpeechLoss")
class _FractionalMassDDPSpeechLoss(nn.Module):
    """按 rank 注入小数质量，验证全局质量小于 1 时的 DDP 缩放。"""

    REQUIRES_SPEECH_REFERENCE = True

    def __init__(self, mass: float) -> None:
        super().__init__()
        self.mass = float(mass)

    def speech_reference_reduction_mass(self, clean, lengths):
        return torch.full(
            (clean.shape[0],),
            self.mass,
            device=clean.device,
            dtype=torch.float32,
        )

    def forward(self, enhanced, clean, lengths=None):
        return (enhanced - clean).square().mean()


@LOSS_REGISTRY.register("_ZeroWeightDDPSpeechLoss")
class _ZeroWeightDDPSpeechLoss(nn.Module):
    """零权重 DDP 组件，任一 callback/forward 调用都立即失败。"""

    REQUIRES_SPEECH_REFERENCE = True

    def speech_reference_valid_mask(self, clean, lengths):
        raise AssertionError("zero-weight valid callback must not run")

    def speech_reference_reduction_mass(self, clean, lengths):
        raise AssertionError("zero-weight mass callback must not run")

    def forward(self, enhanced, clean, lengths=None):
        raise AssertionError("zero-weight forward must not run")


_WORLD_SIZE = 2
_LOCAL_BATCH_SIZE = 4
_TIMEOUT_SECONDS = 30


class _AffineWaveform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.35))

    def forward(self, base: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        return base + self.scale * direction


def _global_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260831)
    clean = torch.randn(_WORLD_SIZE * _LOCAL_BATCH_SIZE, 96, generator=generator)
    base = 0.7 * clean + 0.2 * torch.randn(clean.shape, generator=generator)
    direction = torch.randn(clean.shape, generator=generator)
    return base, direction, clean


def _speech_mask(distribution: tuple[int, int]) -> torch.Tensor:
    mask = torch.zeros(_WORLD_SIZE * _LOCAL_BATCH_SIZE, dtype=torch.bool)
    for rank, count in enumerate(distribution):
        start = rank * _LOCAL_BATCH_SIZE
        mask[start : start + count] = True
    return mask


def _expected_global_gradient(
    loss_type: str,
    base: torch.Tensor,
    direction: torch.Tensor,
    clean: torch.Tensor,
    speech_mask: torch.Tensor,
) -> torch.Tensor:
    scale = torch.tensor(0.35, requires_grad=True)
    enhanced = base + scale * direction
    speech_idx = speech_mask.nonzero(as_tuple=True)[0]
    loss_cls = GTCRNLogRatioLoss if loss_type == "GTCRNLogRatioLoss" else SISNRLoss
    loss = loss_cls()(enhanced.index_select(0, speech_idx), clean.index_select(0, speech_idx))
    loss.backward()
    assert scale.grad is not None
    return scale.grad.detach()


def _valid_element_mean_worker(rank: int, init_file: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=_TIMEOUT_SECONDS),
    )
    try:
        base, direction, clean = _global_inputs()
        lengths = torch.tensor([12, 18, 24, 30, 48, 60, 72, 90])
        speech_mask = torch.tensor([True, False, True, True, True, True, False, True])
        start = rank * _LOCAL_BATCH_SIZE
        stop = start + _LOCAL_BATCH_SIZE

        model = DistributedDataParallel(_AffineWaveform())
        enhanced = model(base[start:stop], direction[start:stop])
        loss_fn = CompositeLoss(
            components=[{"type": "_ValidElementMeanSpeechLoss", "weight": 1.0}]
        )
        loss = loss_fn(
            enhanced,
            clean[start:stop],
            lengths[start:stop],
            is_pure_noise=~speech_mask[start:stop],
        )
        loss.backward()

        scale = torch.tensor(0.35, requires_grad=True)
        global_enhanced = base + scale * direction
        speech_idx = speech_mask.nonzero(as_tuple=True)[0]
        expected = _ValidElementMeanSpeechLoss()(
            global_enhanced.index_select(0, speech_idx),
            clean.index_select(0, speech_idx),
            lengths.index_select(0, speech_idx),
        )
        expected.backward()
        assert scale.grad is not None
        actual = model.module.scale.grad
        assert actual is not None
        torch.testing.assert_close(actual, scale.grad, rtol=2e-5, atol=2e-6)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _tap_frame_mean_worker(rank: int, init_file: str, checkpoint: str) -> None:
    from core.losses.tap import TAPLoss

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=_TIMEOUT_SECONDS),
    )
    try:
        generator = torch.Generator().manual_seed(20260901)
        clean = torch.randn(4, 640, generator=generator)
        base = 0.7 * clean + 0.2 * torch.randn(clean.shape, generator=generator)
        direction = torch.randn(clean.shape, generator=generator)
        lengths = torch.tensor([320, 480, 560, 640])
        speech_mask = torch.tensor([True, True, False, True])
        start = rank * 2
        stop = start + 2

        def fake_forward_unpadded(self, enhanced, clean):
            n_frames = enhanced.shape[-1] // self.hop_length + 1
            enhanced_acoustics = enhanced[:, :n_frames, None].expand(-1, -1, 25)
            clean_acoustics = clean[:, :n_frames, None].expand(-1, -1, 25)
            energy = torch.zeros(
                enhanced.shape[0], n_frames, device=enhanced.device
            )
            return enhanced_acoustics, clean_acoustics, energy

        model = DistributedDataParallel(_AffineWaveform())
        enhanced = model(base[start:stop], direction[start:stop])
        loss_fn = CompositeLoss(
            components=[
                {
                    "type": "TAPLoss",
                    "weight": 1.0,
                    "args": {"acoustic_model_path": checkpoint, "loss_type": "l1"},
                }
            ]
        )
        loss_fn.losses[0]._forward_unpadded = types.MethodType(
            fake_forward_unpadded, loss_fn.losses[0]
        )
        loss = loss_fn(
            enhanced,
            clean[start:stop],
            lengths[start:stop],
            is_pure_noise=~speech_mask[start:stop],
        )
        loss.backward()

        scale = torch.tensor(0.35, requires_grad=True)
        global_enhanced = base + scale * direction
        speech_idx = speech_mask.nonzero(as_tuple=True)[0]
        expected_loss = TAPLoss(acoustic_model_path=checkpoint, loss_type="l1")
        expected_loss._forward_unpadded = types.MethodType(
            fake_forward_unpadded, expected_loss
        )
        expected = expected_loss(
            global_enhanced.index_select(0, speech_idx),
            clean.index_select(0, speech_idx),
            lengths.index_select(0, speech_idx),
        )
        expected.backward()
        assert scale.grad is not None
        actual = model.module.scale.grad
        assert actual is not None
        torch.testing.assert_close(actual, scale.grad, rtol=2e-5, atol=2e-6)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _ddp_worker(rank: int, init_file: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=_TIMEOUT_SECONDS),
    )
    try:
        base, direction, clean = _global_inputs()
        start = rank * _LOCAL_BATCH_SIZE
        stop = start + _LOCAL_BATCH_SIZE

        for distribution in ((0, _LOCAL_BATCH_SIZE), (1, 3)):
            speech_mask = _speech_mask(distribution)
            # metadata 有效但 clean 静音的样本由组件自身 energy guard 排除。
            guarded_clean = clean.clone()
            guarded_index = _LOCAL_BATCH_SIZE if distribution[0] == 0 else _LOCAL_BATCH_SIZE + 1
            guarded_clean[guarded_index].zero_()

            for loss_type in ("GTCRNLogRatioLoss", "SISNRLoss"):
                model = DistributedDataParallel(_AffineWaveform())
                enhanced = model(base[start:stop], direction[start:stop])
                loss_fn = CompositeLoss(components=[{"type": loss_type, "weight": 1.0}])
                loss = loss_fn(
                    enhanced,
                    guarded_clean[start:stop],
                    is_pure_noise=~speech_mask[start:stop],
                )
                loss.backward()

                expected = _expected_global_gradient(
                    loss_type, base, direction, guarded_clean, speech_mask
                )
                actual = model.module.scale.grad
                assert actual is not None
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                dist.barrier()
    finally:
        dist.destroy_process_group()


def _fractional_mass_worker(rank: int, init_file: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=_TIMEOUT_SECONDS),
    )
    try:
        base = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
        direction = torch.tensor([[0.4, -0.2], [-0.3, 0.6]])
        clean = torch.zeros_like(base)
        masses = (0.1, 0.2)

        model = DistributedDataParallel(_AffineWaveform())
        enhanced = model(base[rank : rank + 1], direction[rank : rank + 1])
        loss_fn = CompositeLoss(
            components=[
                {
                    "type": "_FractionalMassDDPSpeechLoss",
                    "weight": 1.0,
                    "args": {"mass": masses[rank]},
                }
            ]
        )
        loss_fn(enhanced, clean[rank : rank + 1]).backward()

        scale = torch.tensor(0.35, requires_grad=True)
        global_enhanced = base + scale * direction
        local_losses = global_enhanced.square().mean(dim=1)
        expected = (
            masses[0] * local_losses[0] + masses[1] * local_losses[1]
        ) / sum(masses)
        expected.backward()
        assert scale.grad is not None
        actual = model.module.scale.grad
        assert actual is not None
        torch.testing.assert_close(actual, scale.grad, rtol=1e-6, atol=1e-7)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _zero_weight_collective_worker(rank: int, init_file: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=_TIMEOUT_SECONDS),
    )
    try:
        enhanced = torch.full((2, 8), float(rank + 1), requires_grad=True)
        clean = torch.zeros_like(enhanced)
        loss_fn = CompositeLoss(
            components=[
                {"type": "_ZeroWeightDDPSpeechLoss", "weight": 0.0},
                {"type": "L1WavLoss", "weight": 1.0},
            ]
        )
        reduced = []
        original_all_reduce = dist.all_reduce

        def recording_all_reduce(value, op=None):
            reduced.append(value.detach().clone())
            return original_all_reduce(value, op=op)

        dist.all_reduce = recording_all_reduce
        try:
            actual = loss_fn(enhanced, clean)
        finally:
            dist.all_reduce = original_all_reduce

        assert len(reduced) == 2
        assert reduced[0].dtype == torch.float32
        assert reduced[0].item() == 0.0
        torch.testing.assert_close(actual, (enhanced - clean).abs().mean())
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _strict_nonfinite_input_worker(rank: int, init_file: str, name: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=_TIMEOUT_SECONDS),
    )
    try:
        values = {
            "clean": torch.ones(2, 16),
            "enhanced": torch.zeros(2, 16),
            "noisy": torch.zeros(2, 16),
        }
        if rank == 0:
            values[name][0, 0] = float("nan")
        loss_fn = CompositeLoss(
            components=[{"type": "SISNRLoss", "weight": 1.0}],
            strict=True,
        )
        try:
            loss_fn(
                values["enhanced"],
                values["clean"],
                noisy=values["noisy"],
                is_pure_noise=torch.tensor([True, False]),
            )
        except StrictFiniteLossError as exc:
            message = str(exc)
        else:
            raise AssertionError("strict non-finite input was not reported")

        messages = [None] * _WORLD_SIZE
        dist.all_gather_object(messages, message)
        assert messages[0] == messages[1]
        assert f"'{name}' has NaN/Inf" in message
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo backend unavailable",
)
def test_speech_reference_ddp_gradient_uses_global_effective_sample_mean(tmp_path):
    """DDP 梯度应等价于 metadata 全局有效 speech-reference 样本均值。"""
    init_file = str(Path(tmp_path) / "init_store")
    spawn_ddp_workers(_ddp_worker, args=(init_file,), nprocs=_WORLD_SIZE)


def test_fractional_global_mass_ddp_gradient_is_not_clamped(tmp_path):
    """global mass 位于 (0, 1) 时仍须按真实质量归一化。"""
    init_file = str(Path(tmp_path) / "fractional_mass_init_store")
    spawn_ddp_workers(_fractional_mass_worker, args=(init_file,), nprocs=_WORLD_SIZE)


def test_zero_weight_speech_component_uses_fixed_zero_collective_on_all_ranks(tmp_path):
    """零权重 speech 组件跳过所有 callback/forward，但保留零质量 collective。"""
    init_file = str(Path(tmp_path) / "zero_weight_init_store")
    spawn_ddp_workers(
        _zero_weight_collective_worker,
        args=(init_file,),
        nprocs=_WORLD_SIZE,
    )


@pytest.mark.parametrize("name", ["clean", "enhanced", "noisy"])
def test_strict_nonfinite_input_is_synchronized_across_ranks(tmp_path, name):
    """单 rank 非有限输入即使落在 pure 样本中也必须让所有 rank 一致报错。"""
    init_file = str(Path(tmp_path) / f"strict_{name}_init_store")
    spawn_ddp_workers(
        _strict_nonfinite_input_worker,
        args=(init_file, name),
        nprocs=_WORLD_SIZE,
    )


def test_speech_reference_ddp_gradient_uses_component_reduction_mass(tmp_path):
    """不等长 rank 的元素均值组件必须按有效元素质量做全局归一化。"""
    init_file = str(Path(tmp_path) / "mass_init_store")
    spawn_ddp_workers(
        _valid_element_mean_worker,
        args=(init_file,),
        nprocs=_WORLD_SIZE,
    )


def test_tap_ddp_gradient_uses_global_valid_frame_mean(tmp_path):
    """TAPLoss 在不等长 rank 上必须按全局有效帧数归一化。"""
    from core.losses.tap import _TAPAcousticEstimator

    checkpoint = Path(tmp_path) / "tap.pt"
    torch.save(
        {"model_state_dict": _TAPAcousticEstimator().state_dict()},
        checkpoint,
    )
    init_file = str(Path(tmp_path) / "tap_init_store")
    spawn_ddp_workers(
        _tap_frame_mean_worker,
        args=(init_file, str(checkpoint)),
        nprocs=_WORLD_SIZE,
    )


pytestmark = [pytest.mark.unit, pytest.mark.ddp, pytest.mark.subprocess]
