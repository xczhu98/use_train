"""Pre-mix speech augmentations (Layer A).

Applied to clean speech waveform before RIR convolution and noise mixing.
Source: se_train perturb.py (SpeedPerturb / PitchPerturb / EQPerturb)

Contains:
  - speed_perturb:  Kaldi-style speed perturbation (changes both speed and pitch)
  - pitch_shift:    Pitch shift (preserves tempo, librosa-based)
  - random_eq:      Random multi-band EQ
  - add_extra_space: Random silence padding at head/tail (simulates different recording scenarios)

All functions operate on 1-D numpy float32 arrays.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# RNG helpers — use *rng* (np.random.Generator) when provided, else global random
# ---------------------------------------------------------------------------
def _rng_random(rng: Optional[np.random.Generator]) -> float:
    return float(rng.random()) if rng is not None else random.random()

def _rng_uniform(rng: Optional[np.random.Generator], a: float, b: float) -> float:
    return float(rng.uniform(a, b)) if rng is not None else random.uniform(a, b)

def _rng_randint(rng: Optional[np.random.Generator], a: int, b: int) -> int:
    """Inclusive [a, b]."""
    if rng is not None:
        return int(rng.integers(a, b + 1))
    return random.randint(a, b)


# ---------------------------------------------------------------------------
# Core augmentation functions
# ---------------------------------------------------------------------------
def speed_perturb(data: np.ndarray, sr: int,
                  min_rate: float = 0.8, max_rate: float = 1.2,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Kaldi-style speed perturbation: resample to change speed AND pitch.

    rate > 1 → faster & higher pitch;  rate < 1 → slower & lower pitch.
    Pretend audio was recorded at sr*rate, then resample back to sr.
    """
    import librosa
    rate = _rng_uniform(rng, min_rate, max_rate)
    if abs(rate - 1.0) < 0.01:
        return data
    resampled = librosa.resample(
        data.astype(np.float32), orig_sr=int(sr * rate), target_sr=sr,
        res_type="soxr_hq",
    )
    return resampled


def pitch_shift(data: np.ndarray, sr: int,
                semitone_min: float = -1.5, semitone_max: float = 1.5,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    import librosa
    n_steps = _rng_uniform(rng, semitone_min, semitone_max)
    if abs(n_steps) < 0.05:
        return data
    return librosa.effects.pitch_shift(data.astype(np.float32), sr=sr, n_steps=n_steps)


def random_eq(data: np.ndarray, sr: int,
              db_min: float = -5.0, db_max: float = 5.0,
              num_bands_min: int = 1, num_bands_max: int = 5,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    import librosa
    n_fft = 2048
    spectra = librosa.stft(data.astype(np.float32), n_fft=n_fft, hop_length=512)
    num_bands = _rng_randint(rng, num_bands_min, num_bands_max)
    min_freq, max_freq = 100, sr / 2 - 100

    for _ in range(num_bands):
        freq = _rng_uniform(rng, min_freq, max_freq)
        q = _rng_uniform(rng, 0.5, 3.0)
        db = _rng_uniform(rng, db_min, db_max)
        bw = freq / q
        low = max(0, int(round((freq - bw / 2) / (sr / n_fft))))
        high = min(spectra.shape[0], int(round((freq + bw / 2) / (sr / n_fft))))
        gain = 10 ** (db / 20)
        spectra[low:high] *= gain

    return librosa.istft(spectra, n_fft=n_fft, hop_length=512, length=len(data))


# ---------------------------------------------------------------------------
# Extra space (random silence padding at head/tail)
# ---------------------------------------------------------------------------
def add_extra_space(data: np.ndarray, sr: int, prob: float = 0.8,
                    max_head_sec: float = 0.3, max_tail_sec: float = 0.3,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if _rng_random(rng) >= prob:
        return data
    head = _rng_randint(rng, 0, int(max_head_sec * sr)) * int(_rng_random(rng) < 0.8)
    tail = _rng_randint(rng, 0, int(max_tail_sec * sr)) * int(_rng_random(rng) < 0.8)
    if head + tail == 0:
        return data
    return np.pad(data, (head, tail), mode="constant")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def apply_speech_augmentations(speech: np.ndarray, sr: int, cfg,
                                skip_extra_space: bool = False,
                                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply configured pre-mix speech augmentations sequentially.

    Args:
        skip_extra_space: When True, skip add_extra_space (used when speech_splice
                          is on, since splice already fills the target duration).
        rng: Per-sample random generator for reproducibility. Falls back to
             global ``random`` when *None*.
    """
    if _rng_random(rng) < cfg.prob_speed_perturb:
        lo, hi = cfg.speed_rate_range
        speech = speed_perturb(speech, sr, lo, hi, rng=rng)

    if _rng_random(rng) < cfg.prob_pitch_shift:
        lo, hi = cfg.semitone_range
        speech = pitch_shift(speech, sr, lo, hi, rng=rng)

    if _rng_random(rng) < cfg.prob_eq_perturb:
        lo, hi = cfg.eq_db_range
        speech = random_eq(speech, sr, lo, hi, rng=rng)

    if not skip_extra_space and cfg.prob_add_extra_space > 0:
        speech = add_extra_space(speech, sr, prob=cfg.prob_add_extra_space, rng=rng)

    return speech
