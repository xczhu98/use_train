"""Streaming GCC-PHAT 与 PFDKF 独立串联的合同测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tasks.aec.data.linear_aec_frontend import LinearAECFrontend
from tasks.aec.data.pfdkf_torch import PFDKFTorch
from tasks.aec.inference.aec_frontend import AECDeploymentFrontend
from tasks.aec.inference.streaming_delay_estimator import (
    AECStreamAlignConfig,
    create_delay_estimator,
)


pytestmark = pytest.mark.unit

_SAMPLE_RATE = 16000
_FRAME_SIZE = 160
_DELAY_SAMPLES = 320
_PFDKF_CFG = {
    "backend": "pfdkf",
    "N": 4,
    "M": 256,
    "hop": 128,
    "A": 0.999,
    "P_initial": 1.0,
    "keep_m_gate": 0.5,
    "use_res": False,
}
_ALIGN_CFG = {
    "enabled": True,
    "algorithm": "gcc_phat",
    "frame_size": _FRAME_SIZE,
    "buffer_ms": 100.0,
    "coarse_ms": _DELAY_SAMPLES * 1000.0 / _SAMPLE_RATE,
    "residual_ms": 0.0,
    "update_ms": 20.0,
    "min_obs_ms": 20.0,
}


def _load_pair(idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """生成固定 bulk delay 的自包含 echo pair。"""
    sample_count = 3200
    time = np.arange(sample_count, dtype=np.float32) / _SAMPLE_RATE
    frequency = 220.0 + idx * 70.0
    ref = (0.4 * np.sin(2 * np.pi * frequency * time + idx)).astype(np.float32)
    echo = np.zeros_like(ref)
    echo[_DELAY_SAMPLES:] = 0.35 * ref[:-_DELAY_SAMPLES]
    noise = np.random.default_rng(42 + idx).normal(0.0, 0.01, sample_count)
    mic = (echo + noise).astype(np.float32)
    return mic, ref


def _stream_align(mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """仅通过独立 Streaming GCC-PHAT API 生成对齐 farend。"""
    config = AECStreamAlignConfig.from_dict(_ALIGN_CFG)
    estimator = create_delay_estimator(config.to_estimator_config(_SAMPLE_RATE))
    frames: list[np.ndarray] = []
    for start in range(0, len(mic), config.frame_size):
        end = min(start + config.frame_size, len(mic))
        estimator.push_ref(ref[start:end])
        estimator.push_mic(mic[start:end])
        frames.append(estimator.get_aligned_ref_frame(end - start))
    assert estimator.delay_samples == _DELAY_SAMPLES
    return np.concatenate(frames).astype(np.float32, copy=False)


def _torch_pfdkf(aligned: np.ndarray, mic: np.ndarray, device: torch.device) -> np.ndarray:
    kwargs = {key: value for key, value in _PFDKF_CFG.items() if key not in {"backend", "use_res"}}
    model = PFDKFTorch(**kwargs).to(device).eval()
    with torch.no_grad():
        error = model.process(
            torch.from_numpy(mic).unsqueeze(0).to(device),
            torch.from_numpy(aligned).unsqueeze(0).to(device),
        )
    return error[0].cpu().numpy()


def test_streaming_gcc_phat_then_pfdkf_matches_canonical_cpu_path():
    mic, ref = _load_pair()
    aligned = _stream_align(mic, ref)
    expected = _torch_pfdkf(aligned, mic, torch.device("cpu"))
    frontend = AECDeploymentFrontend(
        {
            "delay_alignment": _ALIGN_CFG,
            "linear_aec": {**_PFDKF_CFG, "backend": "pfdkf_torch"},
            "input_routing": "mic_farend_error",
        }
    )

    actual_aligned, actual_error, mask, facts = frontend.process_batch(
        torch.from_numpy(mic).unsqueeze(0),
        torch.from_numpy(ref).unsqueeze(0),
        torch.tensor([len(mic)]),
        _SAMPLE_RATE,
    )

    assert actual_error is not None
    np.testing.assert_allclose(actual_aligned[0].numpy(), aligned, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(actual_error[0].numpy(), expected, atol=1e-6, rtol=1e-6)
    assert mask.tolist() == [True]
    assert facts["delay_alignment"]["algorithm"] == "gcc_phat"
    assert facts["delay_alignment"]["delay_samples"] == [_DELAY_SAMPLES]
    assert facts["linear_aec"]["backend_identity"] == "pfdkf_torch"


def test_numpy_pfdkf_consumes_pre_aligned_reference_without_hidden_alignment():
    mic, ref = _load_pair(1)
    aligned = _stream_align(mic, ref)
    frontend = LinearAECFrontend(_PFDKF_CFG, sample_rate=_SAMPLE_RATE)

    actual = frontend.process(mic, aligned)
    expected = _torch_pfdkf(aligned, mic, torch.device("cpu"))

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert not hasattr(frontend, "apply_tdc")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_streaming_gcc_phat_then_gpu_batch_pfdkf_matches_cpu():
    pairs = [_load_pair(index) for index in range(2)]
    mics = np.stack([pair[0] for pair in pairs])
    aligned = np.stack([_stream_align(*pair) for pair in pairs])
    kwargs = {key: value for key, value in _PFDKF_CFG.items() if key not in {"backend", "use_res"}}

    cpu_model = PFDKFTorch(**kwargs).eval()
    gpu_model = PFDKFTorch(**kwargs).cuda().eval()
    with torch.no_grad():
        expected = cpu_model.process(torch.from_numpy(mics), torch.from_numpy(aligned))
        actual = gpu_model.process(
            torch.from_numpy(mics).cuda(), torch.from_numpy(aligned).cuda()
        ).cpu()

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
