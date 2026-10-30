"""Test PFDKFTorch against NumPy reference for numerical equivalence.

Run: pytest tests/test_pfdkf_torch.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks.aec.data.pfdkf import PFDKF, PFDKFRES
from tasks.aec.data.pfdkf_torch import PFDKFTorch, PFDKFRESTorch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PARAMS = {"N": 4, "M": 1024, "hop": 512, "A": 0.999, "P_initial": 1.0, "keep_m_gate": 0.5}
SR = 16000
DUR = 10
T = SR * DUR
NUM_BLOCK = T // PARAMS["hop"]


def _numpy_pfdkf(mic: np.ndarray, ref: np.ndarray, use_res: bool = False) -> np.ndarray:
    """Reference NumPy implementation (matches _PFDKFBackend.process)."""
    cls = PFDKFRES if use_res else PFDKF
    ft = cls(**PARAMS, res=use_res)
    hop = PARAMS["hop"]
    num_block = len(mic) // hop
    e = []
    for n in range(num_block):
        e_n, _ = ft.filt(ref[n * hop : (n + 1) * hop],
                         mic[n * hop : (n + 1) * hop])
        ft.update()
        e.append(e_n)
    e = np.concatenate(e)
    # Trailing partial block
    trailing = len(mic) - num_block * hop
    if trailing > 0:
        ref_n = np.pad(ref[num_block * hop :], (0, hop - trailing), mode="constant")
        mic_n = np.pad(mic[num_block * hop :], (0, hop - trailing), mode="constant")
        e_n, _ = ft.filt(ref_n, mic_n)
        ft.update()
        e = np.concatenate([e, e_n[:trailing]])
    return e


def _torch_pfdkf(mic: np.ndarray, ref: np.ndarray, device: str, use_res: bool = False) -> np.ndarray:
    """PyTorch implementation wrapper."""
    if use_res:
        model = PFDKFRESTorch(**PARAMS, res=True)
    else:
        model = PFDKFTorch(**PARAMS)
    model.to(device)
    model.eval()
    with torch.no_grad():
        err = model.process(
            torch.from_numpy(mic).to(device),
            torch.from_numpy(ref).to(device),
        )
    return err.cpu().numpy()


def _erle(mic: np.ndarray, err: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(mic ** 2) / np.mean(err ** 2) + 1e-10))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def random_pair():
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(T).astype(np.float32) * 0.1
    mic = rng.standard_normal(T).astype(np.float32) * 0.1
    return mic, ref


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNumericalEquivalence:
    """Verify PyTorch outputs match NumPy to float32 epsilon."""

    def test_single_sample_cpu(self, random_pair):
        mic, ref = random_pair
        err_np = _numpy_pfdkf(mic, ref)
        err_cpu = _torch_pfdkf(mic, ref, "cpu")

        max_diff = float(np.max(np.abs(err_np - err_cpu)))
        rmse = float(np.sqrt(np.mean((err_np - err_cpu) ** 2)))
        assert max_diff < 5e-7, f"CPU max_diff {max_diff} >= 5e-7"
        assert rmse < 5e-8, f"CPU RMSE {rmse} >= 5e-8"

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_single_sample_gpu(self, random_pair):
        mic, ref = random_pair
        err_np = _numpy_pfdkf(mic, ref)
        err_gpu = _torch_pfdkf(mic, ref, "cuda")

        max_diff = float(np.max(np.abs(err_np - err_gpu)))
        rmse = float(np.sqrt(np.mean((err_np - err_gpu) ** 2)))
        assert max_diff < 5e-7, f"GPU max_diff {max_diff} >= 5e-7"
        assert rmse < 5e-8, f"GPU RMSE {rmse} >= 5e-8"

    def test_res_mode_cpu(self, random_pair):
        mic, ref = random_pair
        err_np = _numpy_pfdkf(mic, ref, use_res=True)
        err_cpu = _torch_pfdkf(mic, ref, "cpu", use_res=True)

        max_diff = float(np.max(np.abs(err_np - err_cpu)))
        assert max_diff < 5e-7, f"RES CPU max_diff {max_diff} >= 5e-7"

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_res_mode_gpu(self, random_pair):
        mic, ref = random_pair
        err_np = _numpy_pfdkf(mic, ref, use_res=True)
        err_gpu = _torch_pfdkf(mic, ref, "cuda", use_res=True)

        max_diff = float(np.max(np.abs(err_np - err_gpu)))
        assert max_diff < 5e-7, f"RES GPU max_diff {max_diff} >= 5e-7"


class TestERLEConsistency:
    """Verify ERLE metric is identical across implementations."""

    def test_erle_cpu(self, random_pair):
        mic, ref = random_pair
        err_np = _numpy_pfdkf(mic, ref)
        err_cpu = _torch_pfdkf(mic, ref, "cpu")

        erle_np = _erle(mic[: NUM_BLOCK * PARAMS["hop"]], err_np)
        erle_cpu = _erle(mic[: NUM_BLOCK * PARAMS["hop"]], err_cpu)
        assert abs(erle_cpu - erle_np) < 0.001, f"ERLE diff CPU = {abs(erle_cpu - erle_np):.4f} dB"

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_erle_gpu(self, random_pair):
        mic, ref = random_pair
        err_np = _numpy_pfdkf(mic, ref)
        err_gpu = _torch_pfdkf(mic, ref, "cuda")

        erle_np = _erle(mic[: NUM_BLOCK * PARAMS["hop"]], err_np)
        erle_gpu = _erle(mic[: NUM_BLOCK * PARAMS["hop"]], err_gpu)
        assert abs(erle_gpu - erle_np) < 0.001, f"ERLE diff GPU = {abs(erle_gpu - erle_np):.4f} dB"

    def test_multi_seed(self):
        """Run 20 random seeds; all must satisfy tight tolerance."""
        rng = np.random.default_rng(123)
        max_diffs = []
        erle_diffs = []
        for seed in range(20):
            sub = np.random.default_rng(seed)
            ref = sub.standard_normal(T).astype(np.float32) * 0.1
            mic = sub.standard_normal(T).astype(np.float32) * 0.1
            err_np = _numpy_pfdkf(mic, ref)
            err_cpu = _torch_pfdkf(mic, ref, "cpu")
            max_diffs.append(np.max(np.abs(err_np - err_cpu)))
            erle_diffs.append(
                _erle(mic[: NUM_BLOCK * PARAMS["hop"]], err_cpu)
                - _erle(mic[: NUM_BLOCK * PARAMS["hop"]], err_np)
            )

        assert np.max(max_diffs) < 5e-7, f"Multi-seed max_diff worst = {np.max(max_diffs):.3e}"
        assert np.max(np.abs(erle_diffs)) < 0.001, f"Multi-seed ERLE worst = {np.max(np.abs(erle_diffs)):.4f} dB"


class TestBatchCorrectness:
    """Verify batched processing equals per-sample processing."""

    def test_batch_4_cpu(self):
        rng = np.random.default_rng(77)
        mics = [rng.standard_normal(T).astype(np.float32) * 0.1 for _ in range(4)]
        refs = [rng.standard_normal(T).astype(np.float32) * 0.1 for _ in range(4)]

        # Per-sample
        errs_single = [_torch_pfdkf(m, r, "cpu") for m, r in zip(mics, refs)]

        # Batch
        model = PFDKFTorch(**PARAMS)
        model.eval()
        mic_batch = torch.from_numpy(np.stack(mics))
        ref_batch = torch.from_numpy(np.stack(refs))
        with torch.no_grad():
            err_batch = model.process(mic_batch, ref_batch).numpy()

        for i in range(4):
            max_diff = np.max(np.abs(errs_single[i] - err_batch[i]))
            assert max_diff < 1e-6, f"Batch item {i} max_diff = {max_diff:.3e}"

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_batch_8_gpu(self):
        rng = np.random.default_rng(88)
        mics = [rng.standard_normal(T).astype(np.float32) * 0.1 for _ in range(8)]
        refs = [rng.standard_normal(T).astype(np.float32) * 0.1 for _ in range(8)]

        errs_single = [_torch_pfdkf(m, r, "cuda") for m, r in zip(mics, refs)]

        model = PFDKFTorch(**PARAMS).cuda()
        model.eval()
        mic_batch = torch.from_numpy(np.stack(mics)).cuda()
        ref_batch = torch.from_numpy(np.stack(refs)).cuda()
        with torch.no_grad():
            err_batch = model.process(mic_batch, ref_batch).cpu().numpy()

        for i in range(8):
            max_diff = np.max(np.abs(errs_single[i] - err_batch[i]))
            assert max_diff < 1e-6, f"GPU batch item {i} max_diff = {max_diff:.3e}"


class TestSpeed:
    """Quick sanity-check that GPU batch is meaningfully faster."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_gpu_batch_speedup(self):
        B = 32
        rng = np.random.default_rng(99)
        mic_batch = torch.from_numpy(rng.standard_normal((B, T)).astype(np.float32) * 0.1).cuda()
        ref_batch = torch.from_numpy(rng.standard_normal((B, T)).astype(np.float32) * 0.1).cuda()

        model = PFDKFTorch(**PARAMS).cuda()
        model.eval()

        # Warmup
        for _ in range(3):
            model.process(mic_batch, ref_batch)
        torch.cuda.synchronize()

        # Time
        steps = 20
        t0 = time.perf_counter()
        for _ in range(steps):
            model.process(mic_batch, ref_batch)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        gpu_ms = (t1 - t0) / steps * 1000
        # NumPy 单样本基准在本机实测，避免硬编码常量随机器与 numpy 版本漂移
        mic_sample = mic_batch[0].cpu().numpy()
        ref_sample = ref_batch[0].cpu().numpy()
        np_times_ms = []
        for _ in range(3):
            np_started = time.perf_counter()
            _numpy_pfdkf(mic_sample, ref_sample)
            np_times_ms.append((time.perf_counter() - np_started) * 1000)
        np_time_ms = min(np_times_ms)
        naive_total_ms = np_time_ms * B
        speedup = naive_total_ms / gpu_ms

        print(f"\n  GPU batch B={B}: {gpu_ms:.1f} ms  |  Naive NumPy: {naive_total_ms:.0f} ms  |  Speedup: {speedup:.1f}x")
        assert speedup > 5.0, f"Expected >5x speedup, got {speedup:.1f}x"


# CPU 数值对齐用例进入默认 unit 套件；GPU/性能用例显式选择 gpu marker。
pytestmark = [pytest.mark.unit]
