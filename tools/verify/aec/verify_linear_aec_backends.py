#!/usr/bin/env python3
"""验证 use_train 中两种线性 AEC backend 的原理与精度。

任一被请求的 backend 执行失败（``BackendVerificationError``）即非零退出，
不会因为其它 backend 成功而掩盖；未请求/环境不可用导致的 skip 不算失败。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tasks.aec.data.linear_aec_frontend import LinearAECFrontend  # noqa: E402


class BackendVerificationError(RuntimeError):
    """backend 已发起但执行失败；与「未请求/环境不可用而跳过」（返回 ``None``）区分。"""


def erle_db(mic: np.ndarray, error: np.ndarray) -> float:
    """Echo Return Loss Enhancement in dB."""
    mic_power = float(np.mean(mic**2)) + 1e-10
    err_power = float(np.mean(error**2)) + 1e-10
    return 10.0 * np.log10(mic_power / err_power)


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-Invariant SDR."""
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-10)
    target = alpha * ref
    noise = est - target
    return 10.0 * np.log10(np.dot(target, target) / (np.dot(noise, noise) + 1e-10))


def _validate_output(name: str, output: np.ndarray, expected_length: int) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim != 1:
        raise ValueError(f"{name} output must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} output must be non-empty")
    if len(array) != expected_length:
        raise ValueError(
            f"{name} output length mismatch: expected {expected_length}, got {len(array)}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} output must contain only finite values")
    return array


def _validate_metrics(name: str, metrics: dict[str, float]) -> None:
    invalid = [key for key, value in metrics.items() if not np.isfinite(value)]
    if invalid:
        raise ValueError(f"{name} metrics must be finite: {', '.join(invalid)}")


def load_pair(json_path: str, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
    with open(json_path, encoding="utf-8") as f:
        pairs = json.load(f)["pairs"]
    path = str(pairs[idx])
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1:
        raise ValueError(f"Expected stereo file (ch0=echo, ch1=farend), got mono: {path}")
    return data[:, 0].copy(), data[:, 1].copy(), sr


def run_speex(mic: np.ndarray, farend: np.ndarray, sr: int) -> np.ndarray | None:
    try:
        frontend = LinearAECFrontend(
            {"backend": "speex", "frame_size": 256, "filter_length": 2048},
            sample_rate=sr,
        )
        return frontend.process(mic, farend)
    except Exception as exc:
        raise BackendVerificationError(f"[Speex] Failed: {exc}") from exc


def run_pfdkf(mic: np.ndarray, farend: np.ndarray, sr: int) -> np.ndarray | None:
    try:
        frontend = LinearAECFrontend(
            {
                "backend": "pfdkf",
                "N": 5,
                "M": 1024,
                "hop": 512,
                "A": 0.999,
                "keep_m_gate": 0.3,
                "use_res": False,
            },
            sample_rate=sr,
        )
        return frontend.process(mic, farend)
    except Exception as exc:
        raise BackendVerificationError(f"[PFDKF] Failed: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify linear AEC backends")
    ap.add_argument("--pair", default="data/lite_realtime_denoie_dynamic_16k/farend_echo_pair.json")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out_dir", default="/tmp/aec_verify")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    echo, farend, sr = load_pair(args.pair, args.idx)
    try:
        echo = _validate_output("Echo input", echo, len(echo))
        farend = _validate_output("Farend input", farend, len(echo))
        input_metrics = {
            "farend_rms": float(np.sqrt(np.mean(farend.astype(np.float64) ** 2))),
            "echo_rms": float(np.sqrt(np.mean(echo.astype(np.float64) ** 2))),
        }
        _validate_metrics("Input", input_metrics)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    mic = echo.copy()

    print(f"Sample rate: {sr} Hz")
    print(f"Duration:    {len(mic) / sr:.2f} s")
    print(f"Farend RMS:  {input_metrics['farend_rms']:.4f}")
    print(f"Echo RMS:    {input_metrics['echo_rms']:.4f}\n")

    results: dict[str, dict[str, float]] = {}
    artifacts: list[tuple[Path, np.ndarray]] = []
    failed_backends: list[str] = []

    def run_backend(name: str, runner, *runner_args, **runner_kwargs):
        """执行单个 backend：跳过返回 ``None``，失败记录后继续验证其余 backend。"""
        try:
            return runner(*runner_args, **runner_kwargs)
        except BackendVerificationError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            failed_backends.append(name)
            return None

    def record_result(
        name: str,
        output: np.ndarray,
        metric_name: str,
        filename: str,
    ) -> bool:
        try:
            checked = _validate_output(name, output, len(mic))
            metrics = {
                "erle": erle_db(mic, checked),
                metric_name: float(
                    np.sqrt(np.mean(checked.astype(np.float64) ** 2))
                ),
            }
            _validate_metrics(name, metrics)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return False
        results[name] = metrics
        artifacts.append((out_dir / filename, checked))
        print(f"  ERLE:   {metrics['erle']:.2f} dB")
        print(f"  {'Out' if metric_name == 'out_rms' else 'Err'} RMS: {metrics[metric_name]:.4f}")
        return True

    print("=" * 60)
    print("Running Speex MDF ...")
    speex_err = run_backend("Speex", run_speex, mic, farend, sr)
    if speex_err is not None and not record_result(
        "Speex", speex_err, "err_rms", "speex_error.wav"
    ):
        return 1
    print()

    print("=" * 60)
    print("Running PFDKF ...")
    pfdkf_err = run_backend("PFDKF", run_pfdkf, mic, farend, sr)
    if pfdkf_err is not None and not record_result(
        "PFDKF", pfdkf_err, "err_rms", "pfdkf_error.wav"
    ):
        return 1
    print()

    if not results:
        print("  No backend completed successfully", file=sys.stderr)
        return 1
    if failed_backends:
        print(
            f"ERROR: backend 验证失败: {', '.join(failed_backends)}",
            file=sys.stderr,
        )
        return 1
    for path, waveform in artifacts:
        sf.write(path, waveform, sr)

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, values in results.items():
        print(f"  {name:10s}: ERLE = {values['erle']:7.2f} dB")
    print(f"\nOutput WAVs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
