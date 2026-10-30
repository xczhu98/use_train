#!/usr/bin/env python3
"""流式 DeepVQE 推理脚本：PyTorch / ONNX Runtime 单帧 STFT 逐帧处理。

用法示例::

    # PyTorch 推理（checkpoint 路径必须同时提供 config 参数真源）
    python onnx/infer/infer_streaming_deepvqe_ort.py \
        --checkpoint exp/aec/deepvqe_dynamic_joint/checkpoints/model_best.pth \
        --config configs/aec/.../deepvqe.yaml \
        --mic data/test/mic.wav --far data/test/far.wav \
        --output enhanced.wav

    # ONNX Runtime 纯推理（参数来自已严格校验的 artifact metadata）
    python onnx/infer/infer_streaming_deepvqe_ort.py \
        --onnx deepvqe_stream.onnx \
        --mic data/test/mic.wav --far data/test/far.wav \
        --output enhanced.wav

    # ONNX Runtime + PT 对照（config 还会与 artifact metadata 做一致性校验）
    python onnx/infer/infer_streaming_deepvqe_ort.py \
        --checkpoint ... --onnx deepvqe_stream.onnx --config ... \
        --mic ... --far ... --output ...
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf  # type: ignore[import]
import torch
import torchaudio


def _load_wav(path: Path, target_sr: int, device: torch.device) -> torch.Tensor:
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = torch.from_numpy(data.T.copy())
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.to(device)


def _preflight_onnx_session(session):
    """在读取音频前严格校验 ONNX runtime metadata 与 I/O ABI。"""
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner

    return StreamingORTFrameRunner(session, "deepvqe", batch_size=1)


def _runtime_values_from_config(config: Mapping[str, object]) -> tuple[dict, dict]:
    from onnx.runtime_spec import deployment_config_binding

    binding = deployment_config_binding(config, "deepvqe")
    payload = binding["config"]
    stft = payload["stft"]
    args = payload["model"]["args"]
    values = {
        "variant": args.get("variant", "base"),
        "compression": float(args.get("compression", 0.3)),
        "align_delay": int(args.get("align_delay", 100)),
        "n_fft": int(stft["n_fft"]),
        "hop_length": int(stft["hop_length"]),
        "win_length": int(stft["win_length"]),
        "sample_rate": int(stft["sample_rate"]),
        "window": stft["window"],
        "center": stft["center"],
    }
    return values, binding


def _runtime_values_from_artifact(spec) -> dict:
    if spec.deepvqe is None or spec.deployment_config is None:
        raise ValueError("DeepVQE artifact runtime contract is incomplete")
    return {
        "variant": spec.deepvqe.variant,
        "compression": spec.deepvqe.compression,
        "align_delay": spec.deepvqe.alignment_window_frames,
        "n_fft": spec.stft.n_fft,
        "hop_length": spec.stft.hop_length,
        "win_length": spec.stft.win_length,
        "sample_rate": spec.stft.sample_rate,
        "window": spec.stft.window,
        "center": spec.stft.center,
    }


def _resolve_runtime_parameters(
    *,
    config: Mapping[str, object] | None,
    artifact_spec=None,
    cli_values: Mapping[str, object] | None = None,
) -> dict:
    """以 config/artifact 为真源；CLI 仅可重复声明同一值。"""
    config_values = None
    config_binding = None
    if config is not None:
        config_values, config_binding = _runtime_values_from_config(config)
    artifact_values = (
        _runtime_values_from_artifact(artifact_spec)
        if artifact_spec is not None else None
    )
    if config_values is None and artifact_values is None:
        raise ValueError("DeepVQE inference requires --config or a validated ONNX artifact")
    if config_binding is not None and artifact_spec is not None:
        if config_binding != dict(artifact_spec.deployment_config):
            raise ValueError("DeepVQE config does not match ONNX deployment metadata")
    values = config_values or artifact_values
    for name, requested in (cli_values or {}).items():
        if requested is None:
            continue
        actual = values[name]
        equal = (
            math.isclose(float(requested), float(actual), rel_tol=0.0, abs_tol=1e-12)
            if isinstance(actual, float) else requested == actual
        )
        if not equal:
            raise ValueError(
                f"DeepVQE --{name.replace('_', '-')}={requested!r} conflicts with "
                f"config/artifact value {actual!r}"
            )
    return dict(values)


def main() -> int:
    ap = argparse.ArgumentParser(description="Streaming DeepVQE inference (PT / ORT)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="PyTorch checkpoint (.pth)；仅在需要 PT 推理时提供，纯 ORT 不需要")
    ap.add_argument("--onnx", type=Path, default=None, help="ONNX model path；若提供则跑 ORT 推理（可单独使用，无需 --checkpoint）")
    ap.add_argument("--config", type=Path, default=None, help="训练/部署 YAML；PT 路径必需，ORT 路径可用于 artifact 一致性断言")
    ap.add_argument("--variant", type=str, default=None, choices=["base", "small"], help="只可重复声明 config/artifact 中的 variant")
    ap.add_argument("--compression", type=float, default=None, help="只可重复声明 config/artifact 中的 compression")
    ap.add_argument("--n-fft", type=int, default=None, help="只可重复声明 config/artifact 中的 n_fft")
    ap.add_argument("--hop-length", type=int, default=None, help="只可重复声明 config/artifact 中的 hop_length")
    ap.add_argument("--win-length", type=int, default=None, help="只可重复声明 config/artifact 中的 win_length")
    ap.add_argument("--align-delay", type=int, default=None, help="只可重复声明 config/artifact 中的 alignment window")
    ap.add_argument("--sample-rate", type=int, default=None, help="只可重复声明 config/artifact 中的 sample rate")
    ap.add_argument("--window", type=str, default=None, help="只可重复声明 config/artifact 中的 window")
    ap.add_argument("--mic", type=Path, required=True, help="Mic input wav")
    ap.add_argument("--far", type=Path, required=True, help="Far-end reference wav")
    ap.add_argument("--output", type=Path, required=True, help="Output wav")
    ap.add_argument("--device", type=str, default="cpu", help="PyTorch device")
    ap.add_argument("--ort-cuda", action="store_true", help="ORT CUDAExecutionProvider")
    center_group = ap.add_mutually_exclusive_group()
    center_group.add_argument("--center", dest="center", action="store_true", help="只可重复声明 config/artifact 中的 center=True")
    center_group.add_argument("--no-center", dest="center", action="store_false", help="只可重复声明 config/artifact 中的 center=False")
    ap.set_defaults(center=None)
    ap.add_argument("--progress-every", type=int, default=500, help="Print progress every N frames")
    ns = ap.parse_args()

    if not ns.checkpoint and not ns.onnx:
        ap.error("Must provide at least one of --checkpoint or --onnx")

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from onnx.utils import create_ort_session

    ort_providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ns.ort_cuda else ["CPUExecutionProvider"]
    )
    runner = None
    if ns.onnx:
        try:
            session = create_ort_session(ns.onnx.resolve(), providers=ort_providers)
            runner = _preflight_onnx_session(session)
        except Exception as exc:
            ap.error(f"Failed to validate ONNX runtime metadata: {exc}")

    cfg = None
    if ns.config:
        from core.utils.config import load_merged_config

        cfg = load_merged_config(ns.config)
    if ns.checkpoint and cfg is None:
        ap.error("--checkpoint requires --config as the DeepVQE parameter source")
    cli_values = {
        "variant": ns.variant,
        "compression": ns.compression,
        "n_fft": ns.n_fft,
        "hop_length": ns.hop_length,
        "win_length": ns.win_length,
        "align_delay": ns.align_delay,
        "sample_rate": ns.sample_rate,
        "window": ns.window,
        "center": ns.center,
    }
    try:
        config_values = _resolve_runtime_parameters(
            config=cfg,
            artifact_spec=runner.spec if runner is not None else None,
            cli_values=cli_values,
        )
    except (TypeError, ValueError) as exc:
        ap.error(str(exc))

    variant = str(config_values["variant"])
    compression = float(config_values["compression"])
    n_fft = int(config_values["n_fft"])
    hop = int(config_values["hop_length"])
    win = int(config_values["win_length"])
    align_delay = int(config_values["align_delay"])
    sr = int(config_values["sample_rate"])
    window_name = str(config_values["window"])
    center = bool(config_values["center"])
    if win > n_fft or hop <= 0:
        ap.error(f"Invalid STFT config: n_fft={n_fft}, hop_length={hop}, win_length={win}")
    if window_name != "hann":
        ap.error(f"Unsupported DeepVQE STFT window: {window_name!r}; expected 'hann'")
    print(f"Runtime config: variant={variant}, compression={compression}, sr={sr}, "
          f"n_fft={n_fft}, hop={hop}, win={win}, window={window_name}, "
          f"center={center}, align_delay={align_delay}"
          + (f" (lookahead={n_fft // 2} samples)" if center else ""))

    dev = torch.device(ns.device)

    # Load audio
    mic_wav = _load_wav(ns.mic, sr, dev)
    far_wav = _load_wav(ns.far, sr, dev)
    if mic_wav.shape != far_wav.shape:
        min_len = min(mic_wav.shape[-1], far_wav.shape[-1])
        mic_wav = mic_wav[:, :min_len]
        far_wav = far_wav[:, :min_len]
    B, N = mic_wav.shape
    audio_sec = N / sr

    # STFT
    window = torch.hann_window(win, device=dev)
    mic_spec = torch.view_as_real(torch.stft(
        mic_wav, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=center,
    ))  # (B, F, T, 2)
    far_spec = torch.view_as_real(torch.stft(
        far_wav, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=center,
    ))
    _, F_bins, T, _ = mic_spec.shape

    # ------------------------------------------------------------------
    # PyTorch streaming inference (only when checkpoint is provided)
    # ------------------------------------------------------------------
    if ns.checkpoint:
        from tasks.aec.models.discriminative.deepvqe import DeepVQE
        from onnx.models.streaming_deepvqe import StreamingDeepVQE
        from onnx.verify.verify_common import load_checkpoint_strict

        base_model = DeepVQE(
            variant=variant,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            align_delay=align_delay,
            compression=compression,
        ).eval().to(dev)
        load_checkpoint_strict(base_model, ns.checkpoint)

        stream = StreamingDeepVQE(
            base_model,
            win_length=win,
            target_fs=sr,
            window=window_name,
            center=center,
            deployment_config=cfg,
        ).eval().to(dev)
        cache = stream.init_cache(B, dev, mic_spec.dtype)
        outs: list[torch.Tensor] = []
        pt_times: list[float] = []

        t0 = time.perf_counter()
        with torch.inference_mode():
            for t in range(T):
                if ns.progress_every and t > 0 and t % ns.progress_every == 0:
                    print(f"  PT frame {t}/{T}", flush=True)
                sp = mic_spec[:, :, t : t + 1, :]
                fp = far_spec[:, :, t : t + 1, :]
                tic = time.perf_counter()
                enh, cache = stream(sp, fp, cache)
                pt_times.append(time.perf_counter() - tic)
                outs.append(enh)
        pt_total = time.perf_counter() - t0

        enh_spec = torch.cat(outs, dim=2)
        enh_wav = torch.istft(
            torch.view_as_complex(enh_spec),
            n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, center=center,
        )
        if enh_wav.shape[-1] > N:
            enh_wav = enh_wav[..., :N]
        elif enh_wav.shape[-1] < N:
            enh_wav = torch.nn.functional.pad(enh_wav, (0, N - enh_wav.shape[-1]))

        out_dir = ns.output.parent
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(ns.output),
            enh_wav.squeeze(0).cpu().numpy().astype(np.float32),
            sr,
            subtype="PCM_16",
        )

        pt_rtf = pt_total / audio_sec if audio_sec > 0 else float("nan")
        pt_measure_times = pt_times[1:] or pt_times  # 单帧时保留唯一计时样本
        pt_mean_ms = np.mean(pt_measure_times) * 1000 if pt_measure_times else float("nan")
        pt_max_ms = np.max(pt_measure_times) * 1000 if pt_measure_times else float("nan")
        print(f"Saved: {ns.output}")
        print(f"  Audio duration: {audio_sec:.2f}s  STFT frames: {T}")
        print(f"  PT total time:  {pt_total:.3f}s  RTF: {pt_rtf:.4f}")
        print(f"  PT per-frame:   mean={pt_mean_ms:.2f}ms  max={pt_max_ms:.2f}ms")

    # ------------------------------------------------------------------
    # ONNX Runtime inference (optional)
    # ------------------------------------------------------------------
    if ns.onnx is not None and ns.onnx.exists():
        if runner is None:
            raise RuntimeError("ONNX session did not complete ABI preflight")
        if B != runner.batch_size:
            raise RuntimeError(
                f"Preflight batch size {runner.batch_size} does not match audio batch {B}"
            )

        ort_frames: list[np.ndarray] = []
        ort_times: list[float] = []
        t0 = time.perf_counter()
        for t in range(T):
            if ns.progress_every and t > 0 and t % ns.progress_every == 0:
                print(f"  ORT frame {t}/{T}", flush=True)
            frame, elapsed = runner.run_frame(
                mic_spec[:, :, t : t + 1, :],
                far_spec[:, :, t : t + 1, :],
            )
            ort_frames.append(frame)
            ort_times.append(elapsed)
        ort_total = time.perf_counter() - t0
        ort_spec = torch.from_numpy(np.concatenate(ort_frames, axis=2)).to(dev)
        ort_wav = torch.istft(
            torch.view_as_complex(ort_spec),
            n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, center=center,
        )
        if ort_wav.shape[-1] > N:
            ort_wav = ort_wav[..., :N]
        elif ort_wav.shape[-1] < N:
            ort_wav = torch.nn.functional.pad(ort_wav, (0, N - ort_wav.shape[-1]))

        if ns.checkpoint:
            ort_out_path = ns.output.parent / (ns.output.stem + "_ort.wav")
        else:
            ort_out_path = ns.output
            out_dir = ns.output.parent
            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(ort_out_path),
            ort_wav.squeeze(0).cpu().numpy().astype(np.float32),
            sr,
            subtype="PCM_16",
        )

        ort_rtf = ort_total / audio_sec if audio_sec > 0 else float("nan")
        ort_measure_times = ort_times[1:] or ort_times  # 单帧时保留唯一计时样本
        ort_mean_ms = np.mean(ort_measure_times) * 1000 if ort_measure_times else float("nan")
        ort_max_ms = np.max(ort_measure_times) * 1000 if ort_measure_times else float("nan")
        print(f"Saved ORT: {ort_out_path}")
        print(f"  ORT total time: {ort_total:.3f}s  RTF: {ort_rtf:.4f}")
        print(f"  ORT per-frame:  mean={ort_mean_ms:.2f}ms  max={ort_max_ms:.2f}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
