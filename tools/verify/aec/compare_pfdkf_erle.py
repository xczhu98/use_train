#!/usr/bin/env python3
"""ERLE 对比：NumPy PFDKF vs PyTorch PFDKF（CPU & GPU）。

在真实 farend-echo pair 数据上运行，统计 ERLE 分布并验证两者一致性。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tasks.aec.data.linear_aec_frontend import LinearAECFrontend  # noqa: E402


def erle_db(mic: np.ndarray, error: np.ndarray) -> float:
    """Echo Return Loss Enhancement in dB."""
    mic64 = np.asarray(mic, dtype=np.float64)
    error64 = np.asarray(error, dtype=np.float64)
    mic_power = float(np.mean(mic64**2)) + 1e-10
    err_power = float(np.mean(error64**2)) + 1e-10
    return 10.0 * np.log10(mic_power / err_power)


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


def _finite_metric(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def run_pfdkf_numpy(mic: np.ndarray, ref: np.ndarray, sr: int) -> np.ndarray:
    frontend = LinearAECFrontend(
        {
            "backend": "pfdkf",
            "N": 4,
            "M": 1024,
            "hop": 512,
            "A": 0.999,
            "keep_m_gate": 0.5,
            "use_res": False,
        },
        sample_rate=sr,
    )
    return frontend.process(mic, ref)


def run_pfdkf_torch_cpu(mic: np.ndarray, ref: np.ndarray, sr: int) -> np.ndarray:
    frontend = LinearAECFrontend(
        {
            "backend": "pfdkf_torch",
            "N": 4,
            "M": 1024,
            "hop": 512,
            "A": 0.999,
            "keep_m_gate": 0.5,
            "use_res": False,
            "device": "cpu",
        },
        sample_rate=sr,
    )
    return frontend.process(mic, ref)


def run_pfdkf_torch_gpu(mic: np.ndarray, ref: np.ndarray, sr: int) -> np.ndarray | None:
    if not torch.cuda.is_available():
        return None
    frontend = LinearAECFrontend(
        {
            "backend": "pfdkf_torch",
            "N": 4,
            "M": 1024,
            "hop": 512,
            "A": 0.999,
            "keep_m_gate": 0.5,
            "use_res": False,
            "device": "cuda",
        },
        sample_rate=sr,
    )
    return frontend.process(mic, ref)


def load_pair(json_path: str, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
    with open(json_path, encoding="utf-8") as f:
        pairs = json.load(f)["pairs"]
    path = str(pairs[idx])
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1:
        raise ValueError(f"Expected stereo (ch0=echo, ch1=farend), got mono: {path}")
    return data[:, 0].copy(), data[:, 1].copy(), sr


def _stat(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("statistics require a non-empty value list")
    if not np.isfinite(array).all():
        raise ValueError("statistics require only finite values")
    stats = {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }
    for name, value in stats.items():
        _finite_metric(f"statistics {name}", value)
    return stats


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _save_waveform_plot(
    path: Path,
    numpy_error: np.ndarray,
    torch_error: np.ndarray,
    difference: np.ndarray,
) -> None:
    """按显式 --plot 请求惰性加载 matplotlib 并保存波形差异图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plot requires the optional matplotlib dependency") from exc

    figure, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(numpy_error)
    axes[0].set_title("NumPy PFDKF error")
    axes[1].plot(torch_error)
    axes[1].set_title("Torch CPU PFDKF error")
    axes[2].plot(difference)
    axes[2].set_title("NumPy - Torch CPU")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare PFDKF ERLE: NumPy vs PyTorch")
    ap.add_argument("--pair", default="data/lite_realtime_denoie_dynamic_16k/farend_echo_pair.json")
    ap.add_argument(
        "--n_samples",
        type=_positive_int,
        default=100,
        help="Number of pairs to test",
    )
    ap.add_argument("--out_dir", default="/tmp/pfdkf_erle_compare")
    ap.add_argument("--plot", action="store_true", help="Save waveform diff plots")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.pair, encoding="utf-8") as f:
        all_pairs = json.load(f)["pairs"]
    if not isinstance(all_pairs, list) or not all_pairs:
        ap.error("pair manifest must contain a non-empty 'pairs' list")
    n = min(args.n_samples, len(all_pairs))
    indices = np.random.default_rng(42).choice(len(all_pairs), size=n, replace=False)

    erles_np: list[float] = []
    erles_cpu: list[float] = []
    erles_gpu: list[float] = []
    diffs_cpu: list[float] = []
    diffs_gpu: list[float] = []
    max_diffs_cpu: list[float] = []
    max_diffs_gpu: list[float] = []
    waveforms: list[tuple[Path, np.ndarray, int]] = []
    plots: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = []

    print(f"Running ERLE comparison on {n} real echo pairs ...\n")
    for i, idx in enumerate(indices, 1):
        echo, farend, sr = load_pair(args.pair, int(idx))
        try:
            echo = _validate_output("Echo input", echo, len(echo))
            farend = _validate_output("Farend input", farend, len(echo))
            mic = echo.copy()
            err_np = _validate_output(
                "NumPy PFDKF", run_pfdkf_numpy(mic, farend, sr), len(mic)
            )
            err_cpu = _validate_output(
                "Torch CPU PFDKF", run_pfdkf_torch_cpu(mic, farend, sr), len(mic)
            )
            raw_gpu = run_pfdkf_torch_gpu(mic, farend, sr)
            err_gpu = (
                None
                if raw_gpu is None
                else _validate_output("Torch GPU PFDKF", raw_gpu, len(mic))
            )

            erle_np = _finite_metric("NumPy PFDKF ERLE", erle_db(mic, err_np))
            erle_cpu = _finite_metric("Torch CPU PFDKF ERLE", erle_db(mic, err_cpu))
            diff_cpu = err_np.astype(np.float64) - err_cpu.astype(np.float64)
            diff_cpu = _validate_output("Torch CPU difference", diff_cpu, len(mic))
            rmse_cpu = _finite_metric(
                "Torch CPU RMSE", np.sqrt(np.mean(diff_cpu**2))
            )
            max_diff_cpu = _finite_metric(
                "Torch CPU maximum absolute error", np.max(np.abs(diff_cpu))
            )

            erle_gpu = None
            diff_gpu = None
            if err_gpu is not None:
                erle_gpu = _finite_metric(
                    "Torch GPU PFDKF ERLE", erle_db(mic, err_gpu)
                )
                diff_gpu = err_np.astype(np.float64) - err_gpu.astype(np.float64)
                diff_gpu = _validate_output("Torch GPU difference", diff_gpu, len(mic))
                diffs_gpu.append(
                    _finite_metric("Torch GPU RMSE", np.sqrt(np.mean(diff_gpu**2)))
                )
                max_diffs_gpu.append(
                    _finite_metric(
                        "Torch GPU maximum absolute error", np.max(np.abs(diff_gpu))
                    )
                )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        erles_np.append(erle_np)
        erles_cpu.append(erle_cpu)
        diffs_cpu.append(rmse_cpu)
        max_diffs_cpu.append(max_diff_cpu)
        if erle_gpu is not None:
            erles_gpu.append(erle_gpu)

        if i <= 3 or i == n:
            print(
                f"  [{i:3d}/{n}]  NP ERLE={erles_np[-1]:7.2f} dB  "
                f"CPU ERLE={erles_cpu[-1]:7.2f} dB  diff={erles_cpu[-1]-erles_np[-1]:+.4f} dB  "
                f"max_err={max_diffs_cpu[-1]:.3e}"
            )
        if i <= 3:
            waveforms.extend(
                [
                    (out_dir / f"{i:03d}_numpy_error.wav", err_np, sr),
                    (out_dir / f"{i:03d}_torch_cpu_error.wav", err_cpu, sr),
                    (out_dir / f"{i:03d}_diff_cpu.wav", diff_cpu, sr),
                ]
            )
            if err_gpu is not None:
                waveforms.append((out_dir / f"{i:03d}_torch_gpu_error.wav", err_gpu, sr))
            if args.plot:
                plots.append(
                    (
                        out_dir / f"{i:03d}_waveform_diff.png",
                        err_np,
                        err_cpu,
                        diff_cpu,
                    )
                )

    try:
        s_np = _stat(erles_np)
        s_cpu = _stat(erles_cpu)
        s_gpu = _stat(erles_gpu) if erles_gpu else {}
        s_md_cpu = _stat(max_diffs_cpu)
        s_rmse_cpu = _stat(diffs_cpu)
        s_md_gpu = _stat(max_diffs_gpu) if max_diffs_gpu else {}
        s_rmse_gpu = _stat(diffs_gpu) if diffs_gpu else {}
        erle_diff_cpu = np.asarray(erles_cpu) - np.asarray(erles_np)
        s_erle_diff_cpu = _stat(erle_diff_cpu.tolist())
        max_erle_diff_cpu = _finite_metric(
            "Torch CPU maximum absolute ERLE difference",
            np.max(np.abs(erle_diff_cpu)),
        )
        erle_diff_gpu = np.asarray(erles_gpu) - np.asarray(erles_np)
        s_erle_diff_gpu = _stat(erle_diff_gpu.tolist()) if erles_gpu else {}
        max_erle_diff_gpu = (
            _finite_metric(
                "Torch GPU maximum absolute ERLE difference",
                np.max(np.abs(erle_diff_gpu)),
            )
            if erles_gpu
            else None
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for path, waveform, sample_rate in waveforms:
        sf.write(path, waveform, sample_rate)
    for path, numpy_error, torch_error, difference in plots:
        try:
            _save_waveform_plot(path, numpy_error, torch_error, difference)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    print("\n" + "=" * 70)
    print("ERLE Summary (dB)")
    print("=" * 70)
    print(f"  {'Backend':15s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'NumPy PFDKF':15s} {s_np['mean']:10.2f} {s_np['std']:10.2f} {s_np['min']:10.2f} {s_np['max']:10.2f}")
    print(f"  {'Torch CPU':15s} {s_cpu['mean']:10.2f} {s_cpu['std']:10.2f} {s_cpu['min']:10.2f} {s_cpu['max']:10.2f}")
    if s_gpu:
        print(f"  {'Torch GPU':15s} {s_gpu['mean']:10.2f} {s_gpu['std']:10.2f} {s_gpu['min']:10.2f} {s_gpu['max']:10.2f}")

    print("\n" + "=" * 70)
    print("Signal Error (NumPy reference)")
    print("=" * 70)
    print(
        f"  vs Torch CPU:  RMSE={s_rmse_cpu['mean']:.3e} (std={s_rmse_cpu['std']:.3e})  "
        f"max_abs={s_md_cpu['mean']:.3e} (std={s_md_cpu['std']:.3e})"
    )
    if s_md_gpu:
        print(
            f"  vs Torch GPU:  RMSE={s_rmse_gpu['mean']:.3e} (std={s_rmse_gpu['std']:.3e})  "
            f"max_abs={s_md_gpu['mean']:.3e} (std={s_md_gpu['std']:.3e})"
        )

    print(
        f"\n  ERLE diff CPU: mean={s_erle_diff_cpu['mean']:+.4f} dB  "
        f"std={s_erle_diff_cpu['std']:.4f} dB  max_abs={max_erle_diff_cpu:.4f} dB"
    )
    if s_erle_diff_gpu:
        print(
            f"  ERLE diff GPU: mean={s_erle_diff_gpu['mean']:+.4f} dB  "
            f"std={s_erle_diff_gpu['std']:.4f} dB  max_abs={max_erle_diff_gpu:.4f} dB"
        )

    print(f"\nOutput files: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
