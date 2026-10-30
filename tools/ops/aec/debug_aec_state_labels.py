"""Visualize frame-level AEC state labels for a single sample.

Supports both 8-class scenario labels and legacy 3-class FST/DT/NST labels.
The binary presence strips ``[near_present, echo_present, noise_present]`` are
derived from the class index for visualization.  This script is useful for
tuning the state-detector thresholds in the YAML config. The example Hybrid-only
config keeps StateCE and the model state head disabled; this tool inspects the
dataset labels independently.

Example:
    python -m tools.ops.aec.debug_aec_state_labels \\
        -C configs/aec/microsoft_challenge/dbgnet/dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml \\
        --index 0 --output /tmp/state_attr_debug.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

# 支持直接运行本模块时导入仓库内的 tasks/core
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tasks.aec.data.dynamic_aec_build import build_dynamic_aec_dataset  # noqa: E402

logger = logging.getLogger(__name__)

ATTR_NAMES = ["near", "echo", "noise"]
ATTR_COLORS = ["#377eb8", "#e41a1c", "#4daf4a"]  # blue, red, green


def _load_dataset(cfg: dict, split: str = "train"):
    """Build the dynamic AEC dataset for the requested split."""
    ds = build_dynamic_aec_dataset(cfg, split=split)
    logger.info("Dataset length: %d", len(ds))
    return ds


def _plot_state_attributes(
    sample: tuple,
    sample_rate: int,
    output_path: str,
    title: str = "AEC State Attributes",
    hop_length: int | None = None,
    show_rms: bool = True,
) -> None:
    """Plot nearend/mic/farend waveforms and binary state attribute strips.

    The input ``sample`` follows DynamicMixingDataset's tuple layout when
    ``state_detector_hop_length`` is enabled:
        (nearend, mic, farend, fs, length, err, err_mask, is_pure_noise,
         is_pure_echo, is_pure_echo_noise, has_echo, state_label)
    ``state_label`` may be a 1-D integer tensor of frame-level class indices
    (8-class scenario or legacy 3-class FST/DT/NST) or a 2-D tensor of shape
    ``(n_frames, 3)`` with binary values for ``[near, echo, noise]``.
    """
    nearend = sample[0].numpy()
    mic = sample[1].numpy()
    farend = sample[2].numpy()

    state_attrs = sample[-1].numpy()
    if state_attrs.ndim == 1:
        # 1-D class labels: convert to presence bits for visualization.
        n_frames = len(state_attrs)
        attrs = np.zeros((n_frames, 3), dtype=np.float32)
        if state_attrs.max() > 2:
            # 8-class scenario label: class = near*4 + echo*2 + noise*1.
            attrs[:, 0] = ((state_attrs >> 2) & 1).astype(np.float32)  # near
            attrs[:, 1] = ((state_attrs >> 1) & 1).astype(np.float32)  # echo
            attrs[:, 2] = (state_attrs & 1).astype(np.float32)         # noise
        else:
            # Legacy 3-class FST/DT/NST heuristic mapping.
            attrs[state_attrs == 0, 1] = 1.0  # FST -> echo
            attrs[state_attrs == 1, 0] = 1.0  # DT -> near
            attrs[state_attrs == 1, 1] = 1.0  # DT -> echo
            attrs[state_attrs == 2, 0] = 1.0  # NST -> near
        state_attrs = attrs

    n_frames = state_attrs.shape[0]
    if hop_length is None:
        hop_length = max(1, len(nearend) // max(n_frames - 1, 1))
    frame_times = (np.arange(n_frames) + 0.5) * hop_length / sample_rate

    n_axes = 4 if show_rms else 3
    fig, axes = plt.subplots(
        n_axes,
        1,
        figsize=(14, 10 if show_rms else 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 1, 1]} if show_rms else [2, 2, 1],
    )
    if show_rms:
        ax_wave_near, ax_wave_mic, ax_attrs, ax_rms = axes
    else:
        ax_wave_near, ax_wave_mic, ax_attrs = axes

    time = np.arange(len(nearend)) / sample_rate
    ax_wave_near.plot(time, nearend, color="steelblue", linewidth=0.5)
    ax_wave_near.set_ylabel("nearend / clean")
    ax_wave_near.set_title(title)
    ax_wave_near.set_xlim(0, time[-1])
    ax_wave_near.set_ylim(-1.05, 1.05)

    ax_wave_mic.plot(time, mic, color="darkorange", linewidth=0.5)
    ax_wave_mic.set_ylabel("mic")
    ax_wave_mic.set_ylim(-1.05, 1.05)

    # Binary attribute strips.
    for attr_idx, (name, color) in enumerate(zip(ATTR_NAMES, ATTR_COLORS)):
        y_base = 2 - attr_idx
        active = state_attrs[:, attr_idx] > 0.5
        for i in np.flatnonzero(active):
            start = i * hop_length / sample_rate
            end = (i + 1) * hop_length / sample_rate
            ax_attrs.axvspan(start, end, ymin=(y_base - 0.4) / 3.0, ymax=(y_base + 0.4) / 3.0, color=color, alpha=0.8)
    ax_attrs.set_ylabel("state attr")
    ax_attrs.set_yticks([0, 1, 2])
    ax_attrs.set_yticklabels(reversed(ATTR_NAMES))
    ax_attrs.set_ylim(-0.5, 2.5)
    ax_attrs.set_xlabel("time (s)")

    # Legend.
    handles = [
        matplotlib.patches.Patch(color=color, label=name)
        for name, color in zip(ATTR_NAMES, ATTR_COLORS)
    ]
    ax_attrs.legend(handles=handles, loc="upper right")

    if show_rms:
        # Per-frame RMS of nearend / farend / mic for threshold diagnosis.
        win_length = 2 * hop_length
        half_win = win_length // 2
        padded_near = np.pad(nearend, (half_win, half_win), mode="constant")
        padded_far = np.pad(farend, (half_win, half_win), mode="constant")
        padded_mic = np.pad(mic, (half_win, half_win), mode="constant")
        near_rms = np.empty(n_frames)
        far_rms = np.empty(n_frames)
        mic_rms = np.empty(n_frames)
        for i in range(n_frames):
            start = i * hop_length
            near_rms[i] = float(np.sqrt(np.mean(padded_near[start : start + win_length] ** 2)))
            far_rms[i] = float(np.sqrt(np.mean(padded_far[start : start + win_length] ** 2)))
            mic_rms[i] = float(np.sqrt(np.mean(padded_mic[start : start + win_length] ** 2)))
        ax_rms.plot(frame_times, 20 * np.log10(near_rms + 1e-10), label="nearend RMS", color="steelblue")
        ax_rms.plot(frame_times, 20 * np.log10(far_rms + 1e-10), label="farend RMS", color="firebrick")
        ax_rms.plot(frame_times, 20 * np.log10(mic_rms + 1e-10), label="mic RMS", color="darkorange")
        ax_rms.set_ylabel("RMS (dB)")
        ax_rms.set_xlabel("time (s)")
        ax_rms.legend(loc="upper right")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    logger.info("Saved state-attribute debug plot to %s", output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize AEC frame-level state labels.")
    parser.add_argument("-C", "--config", required=True, help="Path to YAML config.")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split.")
    parser.add_argument("--index", type=int, default=0, help="Sample index to visualize.")
    parser.add_argument("--output", default="/tmp/state_attr_debug.png", help="Output PNG path.")
    parser.add_argument("--no-rms", action="store_true", help="Do not plot RMS sub-panel.")
    parser.add_argument("--skip-state-label", action="store_true", help="Only plot waveforms, skip attributes.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    ds = _load_dataset(cfg, split=args.split)
    if args.index < 0 or args.index >= len(ds):
        raise IndexError(f"index {args.index} out of range [0, {len(ds)})")

    ds.set_epoch(0)
    sample = ds[args.index]
    hop_length = cfg.get("data", {}).get(args.split, {}).get("state_detector_hop_length")
    sample_rate = int(sample[3])

    if args.skip_state_label:
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        names = ["nearend / clean", "mic", "farend ref"]
        for ax, wave, name in zip(axes, [sample[0], sample[1], sample[2]], names):
            ax.plot(np.arange(len(wave)) / sample_rate, wave.numpy(), linewidth=0.5)
            ax.set_ylabel(name)
            ax.set_ylim(-1.05, 1.05)
        axes[-1].set_xlabel("time (s)")
        plt.tight_layout()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        logger.info("Saved waveform-only plot to %s", args.output)
        plt.close(fig)
        return

    _plot_state_attributes(
        sample,
        sample_rate=sample_rate,
        output_path=args.output,
        title=f"AEC State Attributes — {Path(args.config).stem} — index {args.index}",
        hop_length=hop_length,
        show_rms=not args.no_rms,
    )


if __name__ == "__main__":
    main()
