# Copyright 2022 Shimin Zhang <shmzhang@npu-aslp.org>
# Version: 1.0
#
# reference paper: "STATE-SPACE ARCHITECTURE OF THE PARTITIONED-BLOCK-BASED
# ACOUSTIC ECHO CONTROLLER"
# =============================================================================

"""Partitioned-Block-Based Frequency Domain Kalman Filter (PFDKF).

Ported from se_train to use_train as an optional linear AEC backend.

Two-Stage AEC design note
-------------------------
In the use_train framework, PFDKF serves as the Stage-1 linear AEC frontend.
The output ``error`` (residual signal) is fed into a neural network
(UltraDPCAEC) as the third input ``error_spectra``.

**Why PFDKFRES (residual echo suppression) defaults to ``False``?**

- Stage-1 (PFDKF) should only cancel *linear* echo and **preserve** residual
  echo (including nonlinear distortion, unmodeled reverberation).
- Stage-2 (NN) learns to handle these residuals. If PFDKFRES aggressively
  suppresses residuals with its frequency-domain post-filter ``W``, the NN
  loses the signal it needs to learn from.
- PFDKFRES can also distort near-end speech because the suppression mask
  ``W = 1 - sum(mu * |X|^2)`` operates on a linear assumption and may
  attenuate speech frequencies that overlap with echo.
- Therefore, in Two-Stage training, use ``use_res: false`` (default).
  Only enable ``use_res: true`` for single-stage deployments without NN.

Reference paper: "State-Space Architecture of the Partitioned-Block-Based
Acoustic Echo Controller" (Shimin Zhang, 2022).
"""

import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft


class PFDKF:
    """Partitioned-block frequency-domain Kalman filter.

    Args:
        N: Number of blocks.
        M: Filter length in one block (FFT size, must be 2*V).
        hop: Hop size (frame shift).
        A: Diagonal coefficient (state transition), default 0.999.
        P_initial: Initial posterior variance, default 1.
        keep_m_gate: Smoothing factor for error power estimate, default 0.5.
        res: Whether to apply residual echo suppression (not used in base class).
    """

    def __init__(
        self,
        N: int,
        M: int,
        hop: int,
        A: float = 0.999,
        P_initial: float = 1.0,
        keep_m_gate: float = 0.5,
        res: bool = False,
    ):
        if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N <= 0:
            raise ValueError(f"N must be a positive integer, got {N!r}")
        if isinstance(M, bool) or not isinstance(M, (int, np.integer)) or M <= 0:
            raise ValueError(f"M must be a positive integer, got {M!r}")
        if isinstance(hop, bool) or not isinstance(hop, (int, np.integer)) or hop <= 0:
            raise ValueError(f"hop must be a positive integer, got {hop!r}")
        if M != 2 * hop:
            raise ValueError(f"PFDKF requires M == 2 * hop, got M={M}, hop={hop}")
        if not all(np.isfinite(value) for value in (A, P_initial, keep_m_gate)):
            raise ValueError("A, P_initial and keep_m_gate must be finite")
        if P_initial < 0.0 or not 0.0 <= keep_m_gate <= 1.0:
            raise ValueError("P_initial must be >= 0 and keep_m_gate must be in [0, 1]")
        self.N = N
        self.M = M
        self.hop = hop
        self.freq_len = self.M // 2 + 1
        self.A = A
        self.A2 = A**2
        self.m_smooth_factor = keep_m_gate
        self.res = res

        self.x = np.zeros(shape=(self.M), dtype=np.float32)
        self.m = np.zeros(shape=(self.freq_len), dtype=np.float32)
        self.P = np.full((self.N, self.freq_len), P_initial)
        self.X = np.zeros((self.N, self.freq_len), dtype=complex)
        self.H = np.zeros((self.N, self.freq_len), dtype=complex)
        self.mu = np.zeros((self.N, self.freq_len), dtype=complex)
        # Gradient constraint window.  NOTE: this keeps the *first* half and
        # zeros the *second* half to match the original se_train reference
        # implementation (Shimin Zhang, 2022).  The theoretical PBFDAF
        # convention is the opposite, but we follow the reference for
        # bit-exact reproducibility.
        self.half_window = np.concatenate(
            ([np.float32(1.0)] * (self.M // 2), [np.float32(0.0)] * (self.M // 2))
        )

    def filt(self, x: np.ndarray, d: np.ndarray):
        """Process one frame.

        Args:
            x: Reference (farend) frame, shape (hop,).
            d: Microphone frame, shape (hop,).

        Returns:
            (e, y): Error and estimated echo, both shape (hop,).
        """
        assert len(x) == self.hop

        self.x = np.concatenate([self.x[self.hop :], x])
        X = fft(self.x)
        self.X[1:] = self.X[:-1]
        self.X[0] = X
        Y = np.sum(self.H * self.X, axis=0)
        y = ifft(Y)
        y = y.real[-self.hop :]
        e = d - y

        e_fft = np.concatenate((np.zeros(shape=(self.M - self.hop,), dtype=np.float32), e))
        self.E = fft(e_fft)
        self.m = self.m_smooth_factor * self.m + (1 - self.m_smooth_factor) * np.abs(self.E) ** 2
        R = np.sum(self.X * self.P * self.X.conj(), 0) + 2 * self.m / self.N
        self.mu = self.P / (R + 1e-10)
        return e, y

    def update(self):
        """Update filter coefficients after filt()."""
        G = self.mu * self.X.conj()
        self.P = self.A2 * (1 - 0.5 * G * self.X) * self.P + (1 - self.A2) * np.abs(self.H) ** 2
        self.H = self.A * (self.H + fft(self.half_window * (ifft(self.E * G).real)))


class PFDKFRES(PFDKF):
    """PFDKF with residual echo suppression (RES).

    Applies a frequency-domain suppression mask W after Kalman filtering
    to further attenuate residual echo.
    """

    def filt(self, x: np.ndarray, d: np.ndarray):
        assert len(x) == self.hop

        self.x = np.concatenate([self.x[self.hop :], x])
        X = fft(self.x)
        self.X[1:] = self.X[:-1]
        self.X[0] = X
        Y = np.sum(self.H * self.X, axis=0)
        y = ifft(Y)
        y = y.real[-self.hop :]
        e = d - y

        e_fft = np.concatenate((np.zeros(shape=(self.M - self.hop,), dtype=np.float32), e))
        self.E = fft(e_fft)
        self.m = self.m_smooth_factor * self.m + (1 - self.m_smooth_factor) * np.abs(self.E) ** 2
        R = np.sum(self.X * self.P * self.X.conj(), 0) + 2 * self.m / self.N
        self.mu = self.P / (R + 1e-10)

        if self.res:
            W = 1 - np.sum(self.mu * np.abs(self.X) ** 2, 0)
            W = np.clip(W, 0.0, 1.0)
            E_res = W * self.E
            e = ifft(E_res).real[-self.hop :]
            y = d - e
        return e, y


def pfdkf(
    x: np.ndarray,
    d: np.ndarray,
    N: int = 5,
    M: int = 1024,
    hop: int = 512,
    A: float = 0.999,
    P_initial: float = 1.0,
    keep_m_gate: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """High-level wrapper: run PFDKF on entire signals.

    Args:
        x: Reference (farend) signal, shape (T,).
        d: Microphone signal, shape (T,).
        N, M, hop, A, P_initial, keep_m_gate: PFDKF hyperparameters.

    Returns:
        (e, y): Error signal and estimated echo, both shape (num_block*hop,).
        Trailing samples beyond integer number of blocks are discarded.
    """
    ft = PFDKF(N, M, hop, A, P_initial, keep_m_gate)

    num_block = min(len(x), len(d)) // hop
    e = np.zeros(num_block * hop)
    y = np.zeros(num_block * hop)

    for n in range(num_block):
        x_n = x[n * hop : (n + 1) * hop]
        d_n = d[n * hop : (n + 1) * hop]
        e_n, y_n = ft.filt(x_n, d_n)
        ft.update()
        e[n * hop : (n + 1) * hop] = e_n
        y[n * hop : (n + 1) * hop] = y_n
    return e, y


def pfdkfres(
    x: np.ndarray,
    d: np.ndarray,
    N: int = 5,
    M: int = 1024,
    hop: int = 512,
    A: float = 0.999,
    P_initial: float = 1.0,
    keep_m_gate: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """High-level wrapper: run PFDKFRES on entire signals."""
    ft = PFDKFRES(N, M, hop, A, P_initial, keep_m_gate, res=True)

    num_block = min(len(x), len(d)) // hop
    e = np.zeros(num_block * hop)
    y = np.zeros(num_block * hop)

    for n in range(num_block):
        x_n = x[n * hop : (n + 1) * hop]
        d_n = d[n * hop : (n + 1) * hop]
        e_n, y_n = ft.filt(x_n, d_n)
        ft.update()
        e[n * hop : (n + 1) * hop] = e_n
        y[n * hop : (n + 1) * hop] = y_n
    return e, y
