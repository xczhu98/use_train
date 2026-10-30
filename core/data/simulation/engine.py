"""Simulation engine - unified interface for audio augmentation.

This module provides a clean API for speech enhancement data simulation.
All implementations are now self-contained in core.data.simulation.urgent_impl
for full framework independence.

Author: xincheng.zhu
"""

from __future__ import annotations

# Import all functions from internal implementation
from core.data.simulation.urgent_impl import (
    add_moving_reverberation,
    add_paired_reverberation,
    add_reverberation,
    bandwidth_limitation,
    clipping_aug,
    estimate_early_rir,
    get_highpass_taps,
    mix_noise,
    packet_loss,
    read_audio,
    wind_noise,
)

# Export all public APIs
__all__ = [
    "read_audio",
    "mix_noise",
    "add_reverberation",
    "add_paired_reverberation",
    "add_moving_reverberation",
    "estimate_early_rir",
    "wind_noise",
    "bandwidth_limitation",
    "clipping_aug",
    "packet_loss",
    "get_highpass_taps",
]
