"""
Single-Channel Wind Noise Generator

Authors   : Daniele Mirabilii and Emanuël Habets

Reference : D. Mirabilii, A. Lodermeyer, F. Czwielong, S. Becker and E.A P. Habets,
            Simulating wind noise with airflow speed-dependent characteristics,
            Proc. of International Workshop on Acoustic Signal Enhancement (IWAENC), 2022.

Copyright (C) 2023 Friedrich-Alexander-Universität Erlangen-Nürnberg, Germany

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

----------------------------------------------------------------------------
本文件 vendored 自 URGENT2026 challenge track1 simulation/sc_wind_noise_generator.py
（上游仓库：https://github.com/fgnt/sc-wind-noise-generator，MIT License）。

本地改动清单（相对上游）：
1. 移除 matplotlib / sounddevice / spectrum / time 顶层依赖；
   plot_signals() / play_signal() / save_signal() 演示函数一并移除
   （本仓库只保留离线批量生成所需的核心能力，写 wav 由调用方负责）。
2. spectrum.lsf2poly 替换为本文件内的 lsf2poly()（同一标准算法，
   参考 A.M. Kondoz "Digital Speech" 1994 Chapter 4），数值已对齐验证。
3. __init__ 中 samples 显式取整（int(round(fs * duration))），
   允许 float 类型的 duration。
----------------------------------------------------------------------------
"""

import numpy as np
import scipy as sp


def lsf2poly(lsf):
    """Convert line spectral frequencies to prediction filter coefficients.

    标准 LSF -> LPC 转换（Reference: A.M. Kondoz, "Digital Speech: Coding for
    Low Bit Rate Communications Systems", John Wiley & Sons 1994, Chapter 4），
    与 spectrum.lsf2poly 算法一致（数值等价性见
    tests/test_generate_wind_noise.py）。
    """
    lsf = np.asarray(lsf, dtype=float)

    if lsf.size == 0:
        raise ValueError("lsf must not be empty")
    if max(lsf) > np.pi or min(lsf) < 0:
        raise ValueError("Line spectral frequencies must be between 0 and pi.")

    p = len(lsf)  # model order

    # Form zeros using the LSFs and unit amplitudes
    z = np.exp(1.0j * lsf)

    # Separate the zeros to those belonging to P and Q
    rQ = z[0::2]
    rP = z[1::2]

    # Include the conjugates as well
    rQ = np.concatenate((rQ, rQ.conjugate()))
    rP = np.concatenate((rP, rP.conjugate()))

    # Form the polynomials P and Q, note that these should be real
    Q = np.poly(rQ)
    P = np.poly(rP)

    # Form the sum and difference filters by including known roots at z = 1
    # and z = -1
    if p % 2:
        # Odd order: z = +1 and z = -1 are roots of the difference filter, P1(z)
        P1 = np.convolve(P, [1, 0, -1])
        Q1 = Q
    else:
        # Even order: z = -1 is a root of the sum filter, Q1(z) and z = 1 is a
        # root of the difference filter, P1(z)
        P1 = np.convolve(P, [1, -1])
        Q1 = np.convolve(Q, [1, 1])

    # Prediction polynomial is formed by averaging P1 and Q1
    a = 0.5 * (P1 + Q1)
    return a[0:-1:1]  # do not return last element


class WindNoiseGenerator:
    """Wind Noise Generator Class"""

    def __init__(self, fs=48000, duration=5, generate=True, wind_profile=None, gustiness=3, short_term_var=True, start_seed=None):
        """Initizalize object"""

        self.fs = fs
        self.duration = duration
        self.samples = int(round(fs * duration))
        self.generate = generate
        self.gustiness = gustiness
        self.wind_profile = wind_profile
        self.short_term_var = short_term_var
        if start_seed is not None:
            np.random.seed(start_seed)

    def generate_wind_noise(self):
        """Generate single-channel wind noise by filtering excitation signal"""

        if self.generate:
            wind_profile = self._generate_wind_speed_profile()
        else:
            wind_profile = self._import_wind_speed_profile()

        exc = self.generate_excitation_signal(wind_profile)
        exc_filtered = self._filter(exc, wind_profile, 2048)
        exc_filtered = 0.95*exc_filtered / \
            max(np.abs(exc_filtered))

        return exc_filtered, wind_profile

    def generate_excitation_signal(self, wind_profile):
        """Generate excitation signal"""

        window_size = 128
        hops = window_size // 2  # overlap
        hann_window = np.hanning(window_size)  # hanning window

        wgn = np.concatenate(
            (np.zeros(window_size), np.random.randn(self.samples), np.zeros(window_size)))
        wgn_length = len(wgn)

        lt_var = self._generate_long_term_variance(wind_profile)
        lt_var = np.concatenate((np.zeros(window_size), lt_var, np.zeros(window_size)))

        st_var = self._generate_short_term_variance_garch(wind_profile)
        cond_var = np.abs(st_var)

        num_windows = (wgn_length - window_size) // hops + 1
        exc = np.zeros(wgn_length)

        for time_frame in range(num_windows-1):
            start_idx = time_frame * hops
            end_idx = start_idx + window_size
            idx = np.arange(start_idx, end_idx)

            gain_ltst = lt_var[idx]
            if self.short_term_var:
                gain_ltst *= np.sqrt(cond_var[time_frame])
            noise_seg_ltst = gain_ltst * wgn[idx] * hann_window
            exc[idx] += noise_seg_ltst

        exc = exc[window_size:-window_size]

        return exc

    def _generate_short_term_variance_garch(self, wind_profile):
        """Generate short-term variance of GARCH process"""

        window_size = 128
        hops = window_size // 2 # overlap

        profile = np.concatenate(
            (2 * np.ones(window_size), wind_profile, 2 * np.ones(window_size)))
        profile_length = len(profile)

        num_windows = (profile_length - window_size) // hops + 1
        st_var = np.zeros(num_windows)
        cond_var = np.zeros(num_windows)

        for time_frame in range(num_windows):
            start_idx = time_frame * hops
            end_idx = start_idx + window_size
            idx = np.arange(start_idx, end_idx)

            speed = np.clip(np.mean(profile[idx]), 2, 18)
            alpha, beta, omega = self._speed2par(speed)

            if alpha + beta > 1:
                beta = 0

            cond_var[time_frame] = omega + alpha * \
                st_var[time_frame-1]**2 + beta*(cond_var[time_frame-1])
            st_var[time_frame] = np.sqrt(np.abs(cond_var[time_frame])) * \
                np.random.randn()

        return st_var/max(np.abs(st_var))

    def _generate_long_term_variance(self, wind_profile):
        """Generate long-term variance"""

        # Regression parameter noise variance/wind speed
        regression_coeff = np.array([8.00071114414022, -220.332082908370])

        # Long-term noise variance based on wind speed profile in dB scale
        variance_profile_db = np.polyval(regression_coeff, wind_profile)

        # Long-term noise variance in linear scale
        variance_profile = 10 ** (variance_profile_db / 10)
        var_lt = np.sqrt(np.abs(variance_profile))  # long-term gain

        return var_lt

    def _generate_wind_speed_profile(self, b_par=2, a_par=2):
        """Generate the wind speed profile by sampling a Weibull distribution"""

        speed_points = int(
            self.gustiness)  # gustiness, 1 = constant speed, 10 = highly-variable speed

        # Sample from the Weibull distribution (change b and a for different distributions)
        wind_speed_profile_lt = b_par * np.random.weibull(a_par, speed_points)

        # Interpolate speed values as required audio samples
        wind_speed_profile = sp.signal.resample(
            wind_speed_profile_lt, self.samples)

        # Additive speed fluctuations
        fluctuations = 10 * np.random.randn(self.samples)

        # Smoothing of the fluctuations
        hann_window = np.hanning(self.fs * 100e-3)
        hann_window /= sum(hann_window)  # hanning window for the smoothing
        fluctuations = sp.signal.lfilter(hann_window, 1, fluctuations)

        # Add the fluctuations to the generated wind speed profile
        wind_speed_profile += fluctuations

        return wind_speed_profile

    def _import_wind_speed_profile(self):
        """Read the wind speed profile from input"""

        wind_speed_profile_lt = self.wind_profile  # load speed values

        # Interpolate speed values as required audio samples
        wind_speed_profile = sp.signal.resample(
            wind_speed_profile_lt, self.samples)
        fluctuations = 10 * np.random.randn(self.samples)  # additive speed fluctuations

        # Smoothing of the fluctuations
        hann_window = np.hanning(self.fs * 100e-3)
        hann_window /= sum(hann_window)  # hanning window for the smoothing
        fluctuations = sp.signal.lfilter(hann_window, 1, fluctuations)

        # Add the fluctuations to the generated wind speed profile
        wind_speed_profile += fluctuations

        return wind_speed_profile

    def _filter(self, exc, wind_profile, window_size):
        """Filter the excitation signals with the AR filter coefficients"""

        hops = window_size // 2  # overlap
        hann_window = np.hanning(window_size)  # hanning window

        profile = np.concatenate(
            (2 * np.ones(window_size), wind_profile, 2 * np.ones(window_size)))

        exc = np.concatenate((np.zeros(window_size), exc, np.zeros(window_size)))
        exc_length = len(exc)

        # Overlap-add approach for the time-varying filtering of the excitation signal
        num_windows = (exc_length - window_size) // hops + 1
        exc_filtered = np.zeros(exc_length)

        for time_frame in range(num_windows):
            start_idx = time_frame * hops
            end_idx = start_idx + window_size
            idx = np.arange(start_idx, end_idx)

            speed = np.clip(np.mean(profile[idx]), 2, 18)
            lpc = self._lsf2lpc(speed)

            exc_seg = exc[idx] * hann_window
            exc_seg_filtered = sp.signal.lfilter(
                np.array([1.0]), lpc, exc_seg)

            exc_filtered[idx] += exc_seg_filtered

        exc_filtered = exc_filtered[window_size:-window_size]

        return exc_filtered

    def _speed2par(self, speed):
        """Convert speed to GARCH parameters"""

        gp_alpha = np.array([-2.73244444508231e-05, 0.00141129711949206, -
                            0.0274652794467908, 0.257613241095714, -0.139824587447063])
        gp_beta = np.array(
            [-9.75160902595897e-05, 0.00464300106846736, -0.0871968755558256, 0.651013973757802])
        gp_omega = np.array(
            [9.69585296574741e-05, -0.00231853830578967, 0.0124681159197788])

        alpha = np.polyval(gp_alpha, speed)
        beta = np.polyval(gp_beta, speed)
        omega = np.polyval(gp_omega, speed)

        return alpha, beta, omega

    def _lsf2lpc(self, speed):
        """Generate LPC coefficients from the LSF-speed models given a speed value"""

        # Regression coefficients of the LFS-speed model
        # The n-th LFS coefficient corresponds to the n-th column
        regression_coeff = np.array([[-2.63412497797108e-06, 5.93162248595821e-05,
                                      0.000215613938043173, -0.000149723789407121,
                                      -0.000213703084399375],
                                     [9.50240139044154e-05,	-0.00271741166649528,
                                      -0.0103783584000284, 0.00483963669507075,
                                      0.00931864887930701],
                                     [-0.000699199223507821, 0.0428714179385289,
                                      0.177250839818556, -0.0329542145779793,
                                      -0.129910107562929],
                                     [0.0106849674771013, -0.234688122194936,
                                      -1.21337646113093, -0.168053225019258,
                                      0.568371362156217],
                                     [-0.000966851130291645, 0.541693139684727,
                                      3.24796925730457, 2.54984352038733,
                                      1.86097523205089]])
        order = 5

        # Estimate LFS based on the speed value
        lfs_estimated = np.zeros(order)

        for order_idx in range(order):
            lfs_estimated[order_idx] = np.polyval(regression_coeff[:, order_idx], speed)

        # Convert LFS into LPC coefficients
        lpc_a = lsf2poly(lfs_estimated)

        return lpc_a
