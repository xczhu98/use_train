"""CompositeLoss/validation 固定 preflight 的两 rank CPU/Gloo 回归测试。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from core.losses.composite import CompositeLoss
from core.trainer.validation_orchestration import _run_synced_phase
from core.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("_PreflightSpeechLoss")
class _PreflightSpeechLoss(nn.Module):
    REQUIRES_SPEECH_REFERENCE = True

    def __init__(self, *, cache: bool = False):
        super().__init__()
        self.mass_calls = 0
        if cache:
            self.fft_sizes = (64,)
            self.hop_sizes = (32,)

    def speech_reference_reduction_mass(self, clean, lengths):
        self.mass_calls += 1
        return torch.ones(clean.shape[0], device=clean.device)

    def forward(self, enhanced, clean, lengths=None, stft_cache=None):
        return (enhanced - clean).square().mean()


_WORLD_SIZE = 2
_TIMEOUT_SECONDS = 12

# 每个畸形输入应在两 rank 汇聚后的 RuntimeError 中带上 rank 0 的真实拒绝原因；
# 只断言"抛了 RuntimeError"会把 gloo 超时/OOM 误判为校验生效。
_EXPECTED_REJECTIONS = {
    "metadata": r"rank 0: ValueError: is_pure_noise must be a \[B\] Tensor",
    "ss": r"rank 0: ValueError: undesired_enhanced / undesired_ref must match enhanced/clean shape",
    "shape": r"rank 0: ValueError: CompositeLoss enhanced and clean must have identical shapes",
    "cache": r"rank 0: (RuntimeError|ValueError): \S",
}


def _loss_for(scenario: str) -> CompositeLoss:
    if scenario == "cache":
        return CompositeLoss(
            components=[
                {"type": "_PreflightSpeechLoss", "weight": 1.0, "args": {"cache": True}},
                {"type": "_PreflightSpeechLoss", "weight": 1.0, "args": {"cache": True}},
            ]
        )
    return CompositeLoss(
        components=[{"type": "_PreflightSpeechLoss", "weight": 1.0}],
        enable_ss_undesired_energy=scenario == "ss",
    )


def _worker(scenario: str, rank: int, init_file: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=_TIMEOUT_SECONDS),
    )
    try:
        loss_fn = _loss_for(scenario)
        enhanced = torch.ones(2, 128)
        clean = torch.zeros_like(enhanced)
        kwargs = {}
        if scenario == "metadata":
            kwargs["is_pure_noise"] = torch.zeros(1 if rank == 0 else 2, dtype=torch.bool)
        elif scenario == "ss":
            kwargs["undesired_enhanced"] = (
                torch.ones(2, 127) if rank == 0 else torch.ones_like(enhanced)
            )
            kwargs["undesired_ref"] = torch.ones_like(enhanced)
        elif scenario == "shape":
            clean = torch.zeros(2, 127) if rank == 0 else clean
        elif scenario == "cache":
            enhanced = torch.ones(2, 1) if rank == 0 else enhanced
            clean = torch.zeros_like(enhanced)

        if scenario == "finite":
            kwargs["is_pure_noise"] = torch.tensor(
                [True, True] if rank == 0 else [False, False], dtype=torch.bool
            )
            all_reduces = 0
            original_all_reduce = dist.all_reduce

            def recording_all_reduce(value, op=None):
                nonlocal all_reduces
                all_reduces += 1
                return original_all_reduce(value, op=op)

            dist.all_reduce = recording_all_reduce
            try:
                value = loss_fn(enhanced, clean, **kwargs)
            finally:
                dist.all_reduce = original_all_reduce
            assert torch.isfinite(value)
            assert all_reduces == 2
            calls = [None] * _WORLD_SIZE
            dist.all_gather_object(calls, loss_fn.losses[0].mass_calls)
            assert calls == [1, 1]
            dist.barrier()
            return

        with pytest.raises(RuntimeError, match=_EXPECTED_REJECTIONS[scenario]):
            loss_fn(enhanced, clean, **kwargs)

        calls = [None] * _WORLD_SIZE
        dist.all_gather_object(calls, loss_fn.losses[0].mass_calls)
        assert calls == [0, 0], calls
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_workers(tmp_dir: Path, scenario: str) -> list[subprocess.CompletedProcess[str]]:
    init_file = tmp_dir / f"{scenario}.init"
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), environment.get("PYTHONPATH", "")]
    )
    processes = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", scenario, str(rank), str(init_file)],
            cwd=repo_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for rank in range(_WORLD_SIZE)
    ]
    results: list[subprocess.CompletedProcess[str]] = []
    for process in processes:
        try:
            stdout, _ = process.communicate(timeout=_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, _ = process.communicate(timeout=_TIMEOUT_SECONDS)
        results.append(
            subprocess.CompletedProcess(process.args, process.returncode, stdout, None)
        )
    return results


@pytest.mark.unit
@pytest.mark.ddp
@pytest.mark.subprocess
@pytest.mark.parametrize("scenario", ["metadata", "ss", "shape", "cache"])
def test_two_rank_malformed_loss_inputs_fail_consistently(tmp_path, scenario):
    results = _run_workers(tmp_path, scenario)
    assert all(result.returncode == 0 for result in results), [
        (result.returncode, result.stdout) for result in results
    ]


@pytest.mark.unit
@pytest.mark.ddp
@pytest.mark.subprocess
def test_two_rank_finite_loss_keeps_zero_contribution_rank_in_collectives(tmp_path):
    results = _run_workers(tmp_path, "finite")
    assert all(result.returncode == 0 for result in results), [
        (result.returncode, result.stdout) for result in results
    ]


@pytest.mark.unit
def test_validation_fatal_cuda_error_skips_exception_collective():
    calls = []
    trainer = type(
        "Trainer",
        (),
        {"_sync_train_batch_exception": lambda self, error, *, phase: calls.append((error, phase))},
    )()

    def callback():
        raise RuntimeError("CUDA error: device-side assert triggered")

    with pytest.raises(RuntimeError, match="device-side assert"):
        _run_synced_phase(trainer, "forward", callback)
    assert calls == []


@pytest.mark.unit
def test_validation_object_collective_leaves_inference_mode():
    observed = []
    trainer = type(
        "Trainer",
        (),
        {
            "_sync_train_batch_exception": lambda self, error, *, phase: observed.append(
                torch.is_inference_mode_enabled()
            )
        },
    )()

    with torch.inference_mode():
        assert _run_synced_phase(trainer, "prepare", lambda: "ok") == "ok"
    assert observed == [False]


def _worker_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("scenario")
    parser.add_argument("rank", type=int)
    parser.add_argument("init_file")
    args = parser.parse_args(argv)
    _worker(args.scenario, args.rank, args.init_file)


if __name__ == "__main__":
    _worker_main(sys.argv[1:])
