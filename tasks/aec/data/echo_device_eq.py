"""Optional C-aec echo device EQ: linear playback-path frequency coloring.

This block gives echo an independent device timbre that Layer E/F's later
whole-mic degradations cannot represent: after ``noisy = nearend + echo +
noise`` is mixed, whole-mic EQ colors nearend and echo identically. Generic
``random_eq`` can approximate playback-device response diversity; room and
placement effects belong primarily to the RIR simulation.

``tasks/aec/data/dynamic_dataset.py`` wires this into C-aec immediately after echo
load and before SER / ``aec_echo_clipping``. It is applied to echo only under
the explicit assumption that the reference tap is located before playback
coloring (DAC/speaker or optional user/vendor EQ), so farend never carries this
coloring. Linear shaping precedes loudness control and nonlinear distortion in
the modeled signal chain. Any RMS shift is absorbed by SER's power-based
rescale, as Layer C-0b post-RIR/pre-noise-echo ``source_volume_normalize``
absorbs Layer A-2 ``random_eq`` RMS shifts for near-end speech.

Both variants reuse ``core.data.post_augment`` implementations: ``post_eq`` is
generic multi-band random EQ; the historically named ``bass_boost`` preserves
low frequencies while attenuating above-cutoff content to create relative bass
emphasis. The latter approximates an optional user/vendor bass, loudness, or
speaker-compensation EQ preset, not a small speaker's inherent bass roll-off.
No Layer F skip/coordination is needed here (unlike ``aec_echo_clipping``'s
Layer F ``prob_clip`` skip): linear echo-only coloring and later whole-mic EQ
are compositional rather than duplicate nonlinear clipping.
"""

from __future__ import annotations

import numpy as np

from core.data.post_augment import bass_boost, post_eq
from core.data.simulation_utils import weighted_choice

ECHO_DEVICE_EQ_VARIANTS = frozenset({"random_eq", "bass_boost"})


def select_echo_device_eq_variant(
    rng: np.random.Generator, prob: float, variant_probs: dict[str, float]
) -> str | None:
    """Decide whether to trigger device EQ and, if so, which variant.

    Returns ``None`` when not triggered (probability ``1 - prob``), otherwise
    one of ``variant_probs``' keys, chosen by weight.
    """
    if prob <= 0.0 or rng.random() >= prob:
        return None
    return weighted_choice(rng, variant_probs)


def apply_echo_device_eq(
    echo: np.ndarray,
    sr: int,
    variant: str,
    random_eq_db_range: tuple[float, float],
    bass_boost_highpass_cutoff_range: tuple[float, float],
    bass_boost_attenuation_min_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the already-selected *variant* to *echo*.

    ``variant`` must be one produced by :func:`select_echo_device_eq_variant`
    (i.e. a member of :data:`ECHO_DEVICE_EQ_VARIANTS`).
    """
    if variant == "random_eq":
        out = post_eq(echo, sr, random_eq_db_range[0], random_eq_db_range[1], rng=rng)
    elif variant == "bass_boost":
        out = bass_boost(
            echo,
            sr,
            bass_boost_highpass_cutoff_range[0],
            bass_boost_highpass_cutoff_range[1],
            bass_boost_attenuation_min_db,
            rng=rng,
        )
    else:
        raise ValueError(
            f"apply_echo_device_eq: unknown variant {variant!r}; expected one of {sorted(ECHO_DEVICE_EQ_VARIANTS)}"
        )
    return np.asarray(out, dtype=np.float32)
