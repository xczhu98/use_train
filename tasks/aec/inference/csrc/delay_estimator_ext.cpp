/**
 * delay_estimator_ext.cpp - C++ kernels for AEC delay estimation.
 *
 * Provides two algorithms callable from Python via pybind11:
 *   1. GCC-PHAT + ACPS (frequency-domain cross-correlation)
 *   2. Matched Filter (NLMS adaptive filter + histogram voting)
 *
 * Cross-platform: Linux / Windows / macOS.
 * 构建：pip install -e tasks/aec/inference/csrc/
 */

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <cmath>
#include <algorithm>
#include <complex>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "pocketfft_hdronly.h"

namespace py = pybind11;
using cfloat = std::complex<float>;

// =========================================================================
// FFT helpers (backed by pocketfft — same library numpy uses)
// =========================================================================
static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void fft_forward(const std::vector<float>& in, std::vector<cfloat>& out, int nfft) {
    pocketfft::shape_t shape{(size_t)nfft};
    pocketfft::stride_t stride_in{sizeof(float)};
    pocketfft::stride_t stride_out{sizeof(cfloat)};
    pocketfft::shape_t axes{0};
    std::vector<float> padded(nfft, 0.f);
    std::memcpy(padded.data(), in.data(), std::min((int)in.size(), nfft) * sizeof(float));
    out.resize(nfft / 2 + 1);
    pocketfft::r2c(shape, stride_in, stride_out, axes, pocketfft::FORWARD,
                    padded.data(), out.data(), 1.0f);
}

static void fft_backward(const std::vector<cfloat>& in, std::vector<float>& out, int nfft) {
    pocketfft::shape_t shape{(size_t)nfft};
    pocketfft::stride_t stride_in{sizeof(cfloat)};
    pocketfft::stride_t stride_out{sizeof(float)};
    pocketfft::shape_t axes{0};
    out.resize(nfft);
    pocketfft::c2r(shape, stride_in, stride_out, axes, pocketfft::BACKWARD,
                    in.data(), out.data(), 1.0f / nfft);
}

// =========================================================================
// Biquad IIR filter section (Direct Form II Transposed)
// =========================================================================
struct BiquadState { float s1 = 0, s2 = 0; };

struct BiquadCoeffs { float b0, b1, b2, a1, a2; };

static inline float biquad_process_sample(const BiquadCoeffs& c, BiquadState& s, float x) {
    float y = c.b0 * x + s.s1;
    s.s1 = c.b1 * x - c.a1 * y + s.s2;
    s.s2 = c.b2 * x - c.a2 * y;
    return y;
}

// =========================================================================
// Ring Buffer
// =========================================================================
class RingBuffer {
    std::vector<float> buf_;
    int capacity_, head_, size_;
public:
    RingBuffer(int capacity) : buf_(capacity, 0.f), capacity_(capacity), head_(0), size_(0) {}

    void push(const float* data, int n) {
        if (n >= capacity_) {
            std::memcpy(buf_.data(), data + n - capacity_, capacity_ * sizeof(float));
            head_ = 0; size_ = capacity_; return;
        }
        int first = std::min(n, capacity_ - head_);
        std::memcpy(buf_.data() + head_, data, first * sizeof(float));
        if (first < n) std::memcpy(buf_.data(), data + first, (n - first) * sizeof(float));
        head_ = (head_ + n) % capacity_;
        size_ = std::min(size_ + n, capacity_);
    }

    void read_newest(float* out, int n) const {
        int avail = std::min(n, size_);
        std::memset(out, 0, n * sizeof(float));
        if (avail == 0) return;
        int tail = (head_ - 1 + capacity_) % capacity_;
        for (int i = avail - 1; i >= 0; --i) {
            out[n - avail + i] = buf_[(tail - (avail - 1 - i) + capacity_) % capacity_];
        }
    }

    int size() const { return size_; }
    int capacity() const { return capacity_; }
    void reset() { std::fill(buf_.begin(), buf_.end(), 0.f); head_ = 0; size_ = 0; }
};

// =========================================================================
// Downsampled Render Buffer (newest at offset 0)
// =========================================================================
class DSRenderBuffer {
    std::vector<float> buf_;
    int capacity_, head_;
public:
    DSRenderBuffer(int cap) : buf_(cap, 0.f), capacity_(cap), head_(0) {}

    void insert(const float* data, int n) {
        head_ = ((head_ - n) % capacity_ + capacity_) % capacity_;
        for (int i = 0; i < n; ++i) buf_[(head_ + i) % capacity_] = data[i];
    }

    void read(int offset, int length, float* out) const {
        int start = (head_ + offset) % capacity_;
        for (int i = 0; i < length; ++i) out[i] = buf_[(start + i) % capacity_];
    }

    void reset() { std::fill(buf_.begin(), buf_.end(), 0.f); head_ = 0; }
};

// =========================================================================
// GCC-PHAT + ACPS Estimator (C++ kernel)
// =========================================================================
static float require_finite(float v, const char* name) {
    if (!std::isfinite(v))
        throw std::invalid_argument(std::string(name) + " must be finite");
    return v;
}

static float require_positive(float v, const char* name) {
    require_finite(v, name);
    if (!(v > 0.f))
        throw std::invalid_argument(std::string(name) + " must be > 0");
    return v;
}

static float require_nonnegative(float v, const char* name) {
    require_finite(v, name);
    if (v < 0.f)
        throw std::invalid_argument(std::string(name) + " must be >= 0");
    return v;
}

static int require_positive_int(int v, const char* name) {
    if (v <= 0)
        throw std::invalid_argument(std::string(name) + " must be > 0");
    return v;
}

static bool finite_complex(const cfloat& value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

enum class ConfidenceResult { invalid = -1, rejected = 0, accepted = 1 };

static ConfidenceResult selected_lag_confidence(
        const float* mic, const float* ref, int n, int lag,
        int candidate_count, double false_accept_prob) {
    if (candidate_count <= 0 || !std::isfinite(false_accept_prob)
            || !(false_accept_prob > 0.0) || !(false_accept_prob < 1.0))
        return ConfidenceResult::invalid;
    long long lag_abs = std::abs((long long)lag);
    long long overlap_wide = (long long)n - lag_abs;
    if (overlap_wide < 3) return ConfidenceResult::invalid;
    int overlap = (int)overlap_wide;
    int diff_count = overlap - 1;
    int mic_start = lag > 0 ? lag : 0;
    int ref_start = lag < 0 ? (int)lag_abs : 0;

    double mic_mean = 0.0, ref_mean = 0.0;
    double mic_raw_energy = 0.0, ref_raw_energy = 0.0;
    for (int i = 0; i < diff_count; ++i) {
        double mic_value = (double)mic[mic_start + i + 1]
                           - (double)mic[mic_start + i];
        double ref_value = (double)ref[ref_start + i + 1]
                           - (double)ref[ref_start + i];
        if (!std::isfinite(mic_value) || !std::isfinite(ref_value))
            return ConfidenceResult::invalid;
        mic_mean += mic_value;
        ref_mean += ref_value;
        mic_raw_energy += mic_value * mic_value;
        ref_raw_energy += ref_value * ref_value;
    }
    mic_mean /= diff_count;
    ref_mean /= diff_count;

    double numerator = 0.0, mic_energy = 0.0, ref_energy = 0.0;
    for (int i = 0; i < diff_count; ++i) {
        double mic_value = (double)mic[mic_start + i + 1]
                           - (double)mic[mic_start + i] - mic_mean;
        double ref_value = (double)ref[ref_start + i + 1]
                           - (double)ref[ref_start + i] - ref_mean;
        numerator += mic_value * ref_value;
        mic_energy += mic_value * mic_value;
        ref_energy += ref_value * ref_value;
    }
    if (!std::isfinite(numerator) || !std::isfinite(mic_raw_energy)
            || !std::isfinite(ref_raw_energy) || !std::isfinite(mic_energy)
            || !std::isfinite(ref_energy))
        return ConfidenceResult::invalid;

    constexpr double relative_eps = std::numeric_limits<double>::epsilon();
    if (mic_energy <= relative_eps * mic_raw_energy
            || ref_energy <= relative_eps * ref_raw_energy)
        return ConfidenceResult::invalid;
    double denominator = std::sqrt(mic_energy * ref_energy);
    if (!std::isfinite(denominator) || !(denominator > 0.0))
        return ConfidenceResult::invalid;

    double abs_ncc = std::abs(numerator / denominator);
    double threshold = std::sqrt(
        2.0 * std::log(2.0 * (double)candidate_count / false_accept_prob)
        / (double)diff_count);
    if (!std::isfinite(abs_ncc) || !std::isfinite(threshold) || threshold >= 1.0)
        return ConfidenceResult::invalid;
    return abs_ncc >= threshold
               ? ConfidenceResult::accepted
               : ConfidenceResult::rejected;
}

// The C++ decimation path uses Butterworth coefficients (AA_COEFFS/HP_COEFFS)
// baked for butter(3, 0.85/4) and butter(2, 200/8000), i.e. ds==4 & sr==16000.
// Other values are only supported by the Python _Decimator. The matched-filter
// raw config validator calls this before the private constructor derives sizes.
static int require_supported_config(int sr, int ds) {
    if (ds != 4 || sr != 16000)
        throw std::invalid_argument(
            "MatchedFilterEstimatorCpp: hardcoded Butterworth coefficients only "
            "support ds==4 and sr==16000. Use the Python MatchedFilterDelayEstimator "
            "for other downsampling factors / sample rates.");
    return sr;
}

class GccPhatEstimatorCpp {
    int sr_, buffer_samples_, coarse_samples_, max_residual_samples_;
    int update_interval_, min_obs_;
    float ref_thresh_, peak_thresh_, prominence_linear_, false_accept_prob_, alpha_;
    float acps_alpha_, eps_;
    int history_len_;
    std::vector<float> history_;
    RingBuffer mic_buf_, ref_buf_;
    float current_delay_;
    int samples_since_update_;
    mutable std::mutex mutex_;

    // ACPS state
    std::vector<cfloat> R_accum_;
    int acps_nfft_;

public:
    GccPhatEstimatorCpp(int sr, float buffer_ms, float coarse_ms, float residual_ms,
                        float update_ms, float smooth_alpha, float ref_thresh_db,
                        float peak_thresh, float prominence_db, float false_accept_prob,
                        float acps_alpha, float min_obs_ms, float eps)
        : sr_(require_positive_int(sr, "sr")),
          buffer_samples_((int)std::round(require_positive(buffer_ms, "buffer_ms") * sr_ / 1000.f)),
          coarse_samples_((int)std::round(require_nonnegative(coarse_ms, "coarse_ms") * sr_ / 1000.f)),
          max_residual_samples_((int)std::round(require_nonnegative(residual_ms, "residual_ms") * sr_ / 1000.f)),
          update_interval_((int)std::round(require_positive(update_ms, "update_ms") * sr_ / 1000.f)),
          min_obs_(std::max(256, (int)std::round(require_positive(min_obs_ms, "min_obs_ms") * sr_ / 1000.f))),
          ref_thresh_(std::pow(10.f, require_finite(ref_thresh_db, "ref_thresh_db") / 20.f)),
          peak_thresh_(require_positive(peak_thresh, "peak_thresh")),
          prominence_linear_(std::pow(10.f, -require_nonnegative(prominence_db, "prominence_db") / 20.f)),
          false_accept_prob_(require_positive(false_accept_prob, "false_accept_prob")),
          alpha_(require_positive(smooth_alpha, "smooth_alpha")),
          acps_alpha_(require_nonnegative(acps_alpha, "acps_alpha")),
          eps_(require_positive(eps, "eps")),
          history_len_(std::max(3, (int)std::floor(1.f / alpha_ + 0.5f))),
          mic_buf_(std::max(1, buffer_samples_)),
          ref_buf_(std::max(1, buffer_samples_ + coarse_samples_ + max_residual_samples_ + 1024)),
          current_delay_((float)coarse_samples_),
          samples_since_update_(0), acps_nfft_(0) {
        if (buffer_samples_ <= 0 || update_interval_ <= 0)
            throw std::invalid_argument("duration values must resolve to at least one sample");
        if (min_obs_ > buffer_samples_)
            throw std::invalid_argument("min_obs_ms resolves beyond the analysis buffer");
        if (alpha_ > 1.f)
            throw std::invalid_argument("smooth_alpha must be in (0, 1]");
        if (acps_alpha_ >= 1.f)
            throw std::invalid_argument("acps_alpha must be in [0, 1)");
        if (peak_thresh_ > 1.f)
            throw std::invalid_argument("peak_thresh must be in (0, 1]");
        if (false_accept_prob_ >= 1.f)
            throw std::invalid_argument("false_accept_prob must be in (0, 1)");
    }

    void push_ref(py::array_t<float, py::array::c_style | py::array::forcecast> frame) {
        auto r = frame.unchecked<1>();
        int n = (int)r.shape(0);
        std::vector<float> samples(n);
        for (int i = 0; i < n; ++i) {
            float value = r(i);
            if (!std::isfinite(value))
                throw py::value_error("reference frame must contain only finite samples");
            samples[i] = value;
        }
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        ref_buf_.push(samples.data(), n);
    }

    void push_mic(py::array_t<float, py::array::c_style | py::array::forcecast> frame) {
        auto r = frame.unchecked<1>();
        int n = (int)r.shape(0);
        std::vector<float> samples(n);
        for (int i = 0; i < n; ++i) {
            float value = r(i);
            if (!std::isfinite(value))
                throw py::value_error("microphone frame must contain only finite samples");
            samples[i] = value;
        }
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        mic_buf_.push(samples.data(), n);
        long long elapsed = (long long)samples_since_update_ + n;
        if (elapsed >= update_interval_) {
            samples_since_update_ = (int)(elapsed % update_interval_);
            update_delay();
        } else {
            samples_since_update_ = (int)elapsed;
        }
    }

    float delay_ms() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_delay_ * 1000.f / sr_;
    }

    int delay_samples() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return (int)std::round(current_delay_);
    }

    py::array_t<float> get_aligned_ref_frame(int frame_size) {
        if (frame_size < 0)
            throw py::value_error("frame_size must be non-negative");
        auto out = py::array_t<float>(frame_size);
        if (frame_size == 0) return out;
        auto ptr = out.mutable_unchecked<1>();
        std::lock_guard<std::mutex> lock(mutex_);
        int delay = (int)std::round(current_delay_);
        int total = frame_size + delay;
        if (total > ref_buf_.capacity())
            throw py::value_error("frame_size + delay exceeds reference buffer capacity");
        if (total <= 0) {
            ref_buf_.read_newest(ptr.mutable_data(0), frame_size);
        } else {
            std::vector<float> block(total);
            ref_buf_.read_newest(block.data(), total);
            std::memcpy(ptr.mutable_data(0), block.data(), frame_size * sizeof(float));
        }
        return out;
    }

    void reset() {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        mic_buf_.reset(); ref_buf_.reset();
        current_delay_ = (float)coarse_samples_;
        samples_since_update_ = 0;
        history_.clear();
        R_accum_.clear(); acps_nfft_ = 0;
    }

private:
    // Caller holds mutex_ and has released the GIL.
    void update_delay() {
        if (mic_buf_.size() < min_obs_ || ref_buf_.size() < min_obs_) return;

        int n = std::min({mic_buf_.size(), ref_buf_.size(), buffer_samples_});
        std::vector<float> mic(n), ref(n);
        mic_buf_.read_newest(mic.data(), n);
        ref_buf_.read_newest(ref.data(), n);

        // Energy gate. Double accumulation avoids finite float inputs overflowing.
        double ref_energy = 0.0;
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(mic[i]) || !std::isfinite(ref[i])) return;
            ref_energy += (double)ref[i] * (double)ref[i];
        }
        double ref_rms = std::sqrt(ref_energy / n);
        if (!std::isfinite(ref_rms) || ref_rms < ref_thresh_) return;

        int search_center = (int)std::round(current_delay_);
        int nfft = next_pow2(2 * n - 1);
        int nfreq = nfft / 2 + 1;

        // FFT of both signals (real-to-complex via pocketfft)
        std::vector<cfloat> SIG, REF_F;
        fft_forward(mic, SIG, nfft);
        fft_forward(ref, REF_F, nfft);

        // Cross-power spectrum (half-spectrum).
        std::vector<cfloat> R(nfreq);
        for (int i = 0; i < nfreq; ++i) {
            if (!finite_complex(SIG[i]) || !finite_complex(REF_F[i])) return;
            R[i] = SIG[i] * std::conj(REF_F[i]);
            if (!finite_complex(R[i])) return;
        }

        // Build ACPS locally and commit only after the normalized correlation is finite.
        std::vector<cfloat> candidate;
        if (acps_alpha_ > 0) {
            bool old_valid = acps_nfft_ == nfft && (int)R_accum_.size() == nfreq;
            if (old_valid) {
                for (const auto& value : R_accum_) {
                    if (!finite_complex(value)) { old_valid = false; break; }
                }
            }
            candidate = old_valid ? R_accum_ : R;
            if (old_valid) {
                for (int i = 0; i < nfreq; ++i)
                    candidate[i] = acps_alpha_ * candidate[i] + (1.f - acps_alpha_) * R[i];
            }
        } else {
            candidate = R;
        }

        std::vector<cfloat> normalized(nfreq);
        for (int i = 0; i < nfreq; ++i) {
            if (!finite_complex(candidate[i])) return;
            float mag = std::abs(candidate[i]) + eps_;
            if (!std::isfinite(mag) || !(mag > 0.f)) return;
            normalized[i] = candidate[i] / mag;
            if (!finite_complex(normalized[i])) return;
        }

        // IFFT (complex-to-real via pocketfft)
        std::vector<float> cc_raw;
        fft_backward(normalized, cc_raw, nfft);
        for (float value : cc_raw) if (!std::isfinite(value)) return;

        // Reconstruct linear correlation layout: lags [-(n-1), ..., 0, ..., +(n-1)]
        int cc_len = 2 * n - 1;
        std::vector<float> cc(cc_len);
        for (int i = 0; i < n - 1; ++i) cc[i] = cc_raw[nfft - (n - 1) + i];
        for (int i = 0; i < n; ++i) cc[n - 1 + i] = cc_raw[i];

        // Search range
        int min_lag = std::max(0, search_center - max_residual_samples_);
        int max_lag = std::min(search_center + max_residual_samples_, n - 1);
        if (min_lag > max_lag) return;

        int offset = n - 1;
        int start_idx = offset + min_lag;
        int end_idx = offset + max_lag + 1;
        int search_len = end_idx - start_idx;

        // Find peak: mirror Python _find_peak_in_cc() — local peaks filtered by
        // a strong-magnitude threshold, then pick the most prominent peak closest
        // to the current search center.
        std::vector<float> abs_cc(search_len);
        float peak_mag = 0.f, mean_mag = 0.f;
        for (int i = 0; i < search_len; ++i) {
            float v = std::abs(cc[start_idx + i]);
            abs_cc[i] = v;
            mean_mag += v;
            if (v > peak_mag) peak_mag = v;
        }
        mean_mag /= search_len;
        if (peak_mag <= eps_) return;

        int center_rel = search_center - min_lag;
        float strong_threshold = std::max(peak_thresh_, peak_mag * 0.7f);

        std::vector<int> peak_rels;
        if (search_len == 1) {
            peak_rels.push_back(0);
        } else {
            if (abs_cc[0] >= abs_cc[1]) peak_rels.push_back(0);
            for (int i = 1; i < search_len - 1; ++i) {
                if (abs_cc[i] >= abs_cc[i - 1] && abs_cc[i] >= abs_cc[i + 1])
                    peak_rels.push_back(i);
            }
            if (abs_cc[search_len - 1] >= abs_cc[search_len - 2])
                peak_rels.push_back(search_len - 1);
        }

        std::vector<int> strong_rels;
        for (int pr : peak_rels) {
            if (abs_cc[pr] >= strong_threshold) strong_rels.push_back(pr);
        }
        if (strong_rels.empty()) {
            int best = 0;
            for (int i = 1; i < search_len; ++i)
                if (abs_cc[i] > abs_cc[best]) best = i;
            strong_rels.push_back(best);
        }

        int prominence_half = std::max(1, std::min(8, search_len / 16));
        std::vector<float> prominences;
        prominences.reserve(strong_rels.size());
        float max_prominence = -1e30f;
        for (int pr : strong_rels) {
            int lo = std::max(0, pr - prominence_half);
            int hi = std::min(search_len, pr + prominence_half + 1);
            float neighbourhood_sum = 0.f;
            int neighbourhood_count = 0;
            for (int j = lo; j < pr; ++j) {
                neighbourhood_sum += abs_cc[j];
                ++neighbourhood_count;
            }
            for (int j = pr + 1; j < hi; ++j) {
                neighbourhood_sum += abs_cc[j];
                ++neighbourhood_count;
            }
            float neighbourhood_mean = neighbourhood_count > 0
                                           ? neighbourhood_sum / neighbourhood_count
                                           : 0.f;
            float prom = abs_cc[pr] - neighbourhood_mean;
            prominences.push_back(prom);
            if (prom > max_prominence) max_prominence = prom;
        }

        float prom_threshold = max_prominence * 0.9f;
        int peak_idx = -1;
        int best_dist = 0;
        for (size_t i = 0; i < strong_rels.size(); ++i) {
            if (prominences[i] < prom_threshold) continue;
            int dist = std::abs(strong_rels[i] - center_rel);
            if (peak_idx < 0 || dist < best_dist) {
                best_dist = dist;
                peak_idx = strong_rels[i];
            }
        }
        if (peak_idx < 0) return;

        // Sub-sample peak refinement via parabolic interpolation around the
        // three nearest samples.  Mirrors the Python _find_peak_in_cc() logic.
        float peak_idx_float = (float)peak_idx;
        if (peak_idx > 0 && peak_idx < search_len - 1) {
            float y_m1 = abs_cc[peak_idx - 1];
            float y_0 = abs_cc[peak_idx];
            float y_p1 = abs_cc[peak_idx + 1];
            float denom = y_m1 - 2.0f * y_0 + y_p1;
            if (std::abs(denom) > eps_) {
                float delta = 0.5f * (y_m1 - y_p1) / denom;
                // Clamp to the nearest sample interval for robustness.
                delta = std::max(-0.5f, std::min(0.5f, delta));
                peak_idx_float += delta;
            }
        }

        float tau = (float)min_lag + peak_idx_float;
        if (!std::isfinite(tau)) return;

        // 生产搜索保持非负物理 AEC delay；helper 本身支持 signed lag。
        int lag = (int)std::floor(tau + 0.5f);
        ConfidenceResult confidence = selected_lag_confidence(
            mic.data(), ref.data(), n, lag, search_len, false_accept_prob_);
        if (confidence == ConfidenceResult::invalid) return;

        // 有限置信观测即使本次未接受，也可推进 ACPS，保留真实稳态与跳变延时适应能力。
        if (acps_alpha_ > 0) {
            R_accum_ = std::move(candidate);
            acps_nfft_ = nfft;
        }
        if (confidence == ConfidenceResult::rejected) return;

        // Quality gates must be evaluated on the selected peak, not the global
        // maximum, so that confidence is tied to the delay we actually emit.
        float selected_peak = abs_cc[peak_idx];
        if (selected_peak < peak_thresh_) return;
        if (selected_peak < mean_mag / prominence_linear_) return;

        tau = std::max(0.f, tau);
        if (!history_.empty()) {
            float dev = std::abs(tau - current_delay_);
            if (dev > max_residual_samples_ && selected_peak < 0.5f) return;
        }

        history_.push_back(tau);
        if ((int)history_.size() > history_len_) history_.erase(history_.begin());

        // Median; even histories average the two middle values like numpy.median.
        std::vector<float> sorted_h = history_;
        std::sort(sorted_h.begin(), sorted_h.end());
        size_t mid = sorted_h.size() / 2;
        current_delay_ = sorted_h.size() % 2
                             ? sorted_h[mid]
                             : 0.5f * (sorted_h[mid - 1] + sorted_h[mid]);
    }
};

// =========================================================================
// Matched Filter Estimator (C++ kernel)
// =========================================================================

// Pre-computed Butterworth filter coefficients (from scipy)
static const BiquadCoeffs AA_COEFFS[] = {
    {2.1105448865395348e-02f, 4.2210897730790696e-02f, 2.1105448865395348e-02f,
     -4.8503481296777096e-01f, 0.0f},
    {1.0f, 1.0f, 0.0f, -1.1993719142265766e+00f, 5.2724571069856163e-01f},
};
static const BiquadCoeffs HP_COEFFS[] = {
    {9.4597685600279002e-01f, -1.8919537120055800e+00f, 9.4597685600279002e-01f,
     -1.8890330793945245e+00f, 8.9487434461663540e-01f},
};

// 24 dB headroom above normalized full scale accepts ordinary clipping and
// moderate upstream gain while rejecting values that can destabilize IIR/NLMS.
static constexpr float MATCHED_FILTER_AUDIO_ABS_LIMIT = 16.f;
// One second at 16 kHz, and larger than the default 9728-sample search span.
static constexpr size_t MATCHED_FILTER_MAX_PENDING_SAMPLES = 16384;

struct MatchedFilterConfig {
    int sr, ds, num_filters, flen_subblocks, lag_shift_subblocks;
    float smoothing, excitation_limit;
    int hist_window, init_thresh, conv_thresh, block_size;
};

static MatchedFilterConfig validate_matched_filter_config(
        int sr, int ds, int num_filters, int flen_subblocks,
        int lag_shift_subblocks, float smoothing, float excitation_limit,
        int hist_window, int init_thresh, int conv_thresh, int block_size) {
    require_positive_int(sr, "sr");
    require_positive_int(ds, "ds");
    require_positive_int(num_filters, "num_filters");
    require_positive_int(flen_subblocks, "flen_subblocks");
    require_positive_int(lag_shift_subblocks, "lag_shift_subblocks");
    require_positive_int(hist_window, "hist_window");
    require_positive_int(init_thresh, "init_thresh");
    require_positive_int(conv_thresh, "conv_thresh");
    require_positive_int(block_size, "block_size");
    require_finite(smoothing, "smoothing");
    require_finite(excitation_limit, "excitation_limit");

    if (smoothing < 0.f || smoothing > 1.f)
        throw std::invalid_argument("smoothing must be in [0, 1]");
    if (excitation_limit < 0.f)
        throw std::invalid_argument("excitation_limit must be >= 0");
    if (block_size % ds != 0)
        throw std::invalid_argument("block_size must be divisible by ds");
    if (init_thresh > conv_thresh)
        throw std::invalid_argument("init_thresh must be <= conv_thresh");

    require_supported_config(sr, ds);
    return {sr, ds, num_filters, flen_subblocks, lag_shift_subblocks,
            smoothing, excitation_limit, hist_window, init_thresh,
            conv_thresh, block_size};
}

class MatchedFilterEstimatorCpp {
    int sr_, ds_factor_, block_size_, sub_block_size_;
    int num_filters_, filter_length_, lag_shift_;
    float smoothing_, excitation_limit_;
    int hist_window_, initial_thresh_, converged_thresh_;

    // Decimators
    BiquadState aa_ref_[2], aa_mic_[2], hp_ref_[1], hp_mic_[1];

    // Render buffer
    DSRenderBuffer render_buf_;

    // Matched filters
    std::vector<std::vector<float>> filters_;

    // Histogram
    std::vector<int> histogram_;
    std::vector<int> hist_ring_;
    int hist_idx_, hist_count_;
    bool significant_found_;
    int max_lag_;

    // State
    float current_delay_;
    std::vector<float> ref_accum_, mic_accum_;
    RingBuffer ref_raw_buf_;
    mutable std::mutex mutex_;

    void decimate(const float* in, int n, float* out, int& out_n,
                  BiquadState* aa_st, BiquadState* hp_st) {
        BiquadState aa_candidate[2] = {aa_st[0], aa_st[1]};
        BiquadState hp_candidate[1] = {hp_st[0]};
        out_n = n / ds_factor_;
        for (int i = 0; i < n; ++i) {
            float x = in[i];
            for (int s = 0; s < 2; ++s)
                x = biquad_process_sample(AA_COEFFS[s], aa_candidate[s], x);
            x = biquad_process_sample(HP_COEFFS[0], hp_candidate[0], x);
            if (!std::isfinite(x))
                throw std::runtime_error("matched-filter internal non-finite decimator state");
            if (i % ds_factor_ == 0) out[i / ds_factor_] = x;
        }
        for (const auto& state : aa_candidate) {
            if (!std::isfinite(state.s1) || !std::isfinite(state.s2))
                throw std::runtime_error("matched-filter internal non-finite decimator state");
        }
        if (!std::isfinite(hp_candidate[0].s1) || !std::isfinite(hp_candidate[0].s2))
            throw std::runtime_error("matched-filter internal non-finite decimator state");
        aa_st[0] = aa_candidate[0]; aa_st[1] = aa_candidate[1];
        hp_st[0] = hp_candidate[0];
    }

    void nlms_update(int filter_idx, const float* capture, int cap_n,
                     int& out_lag, float& out_accuracy, bool& out_reliable) {
        auto& h = filters_[filter_idx];
        std::vector<float> candidate_h = h;
        int flen = filter_length_;
        int shift = filter_idx * lag_shift_;
        float x2_thresh = flen * excitation_limit_ * excitation_limit_;
        float error_sum = 0, y_sum = 0;
        std::vector<float> x(flen);

        // Python is canonical: error_sum is computed ONCE from the PRE-UPDATE
        // filter coefficients, not progressively-updated coefficients.
        for (int i = 0; i < cap_n; ++i) {
            render_buf_.read(shift + i, flen, x.data());
            float s = 0;
            for (int k = 0; k < flen; ++k) s += candidate_h[k] * x[k];
            float y = capture[i];
            float e = y - s;
            error_sum += e * e;
            y_sum += y * y;
        }

        for (int i = 0; i < cap_n; ++i) {
            render_buf_.read(shift + i, flen, x.data());
            float x2 = 0, s = 0;
            for (int k = 0; k < flen; ++k) {
                s += candidate_h[k] * x[k];
                x2 += x[k] * x[k];
            }
            float y = capture[i];
            float e = y - s;
            if (x2 > x2_thresh && std::abs(y) < 0.95f) {
                float alpha = smoothing_ * e / (x2 + 1e-12f);
                for (int k = 0; k < flen; ++k) candidate_h[k] += alpha * x[k];
            }
        }

        if (!std::isfinite(error_sum) || !std::isfinite(y_sum))
            throw std::runtime_error("matched-filter internal non-finite NLMS candidate");
        int lag = 0; float peak = 0;
        for (int k = 0; k < flen; ++k) {
            if (!std::isfinite(candidate_h[k]))
                throw std::runtime_error("matched-filter internal non-finite NLMS state");
            float v = std::abs(candidate_h[k]);
            if (v > peak) { peak = v; lag = k; }
        }
        out_lag = shift + lag;
        out_accuracy = y_sum - error_sum;
        if (!std::isfinite(out_accuracy))
            throw std::runtime_error("matched-filter internal non-finite NLMS candidate");
        out_reliable = (lag > 2 && lag < flen - 10 && y_sum > 1e-12f && error_sum < 0.2f * y_sum);
        h = std::move(candidate_h);
    }

    explicit MatchedFilterEstimatorCpp(const MatchedFilterConfig& cfg)
        : sr_(cfg.sr), ds_factor_(cfg.ds), block_size_(cfg.block_size),
          sub_block_size_(cfg.block_size / cfg.ds),
          num_filters_(cfg.num_filters),
          filter_length_(cfg.flen_subblocks * sub_block_size_),
          lag_shift_(cfg.lag_shift_subblocks * sub_block_size_),
          smoothing_(cfg.smoothing), excitation_limit_(cfg.excitation_limit),
          hist_window_(cfg.hist_window), initial_thresh_(cfg.init_thresh),
          converged_thresh_(cfg.conv_thresh),
          render_buf_(cfg.num_filters * cfg.lag_shift_subblocks * sub_block_size_
                      + cfg.flen_subblocks * sub_block_size_ + 1024),
          max_lag_((cfg.num_filters - 1) * cfg.lag_shift_subblocks * sub_block_size_
                   + cfg.flen_subblocks * sub_block_size_ - 1),
          histogram_(cfg.num_filters * cfg.lag_shift_subblocks * sub_block_size_
                     + cfg.flen_subblocks * sub_block_size_ + 1, 0),
          hist_ring_(cfg.hist_window, 0), hist_idx_(0), hist_count_(0),
          significant_found_(false), current_delay_(0),
          ref_raw_buf_(max_lag_ * cfg.ds + cfg.block_size * 32)
    {
        filters_.resize(cfg.num_filters, std::vector<float>(filter_length_, 0.f));
        std::memset(aa_ref_, 0, sizeof(aa_ref_));
        std::memset(aa_mic_, 0, sizeof(aa_mic_));
        std::memset(hp_ref_, 0, sizeof(hp_ref_));
        std::memset(hp_mic_, 0, sizeof(hp_mic_));
    }

public:
    MatchedFilterEstimatorCpp(int sr = 16000, int ds = 4, int num_filters = 5,
                              int flen_subblocks = 32, int lag_shift_subblocks = 24,
                              float smoothing = 0.7f, float excitation_limit = 0.005f,
                              int hist_window = 250, int init_thresh = 5, int conv_thresh = 20,
                              int block_size = 64)
        : MatchedFilterEstimatorCpp(validate_matched_filter_config(
              sr, ds, num_filters, flen_subblocks, lag_shift_subblocks,
              smoothing, excitation_limit, hist_window, init_thresh,
              conv_thresh, block_size)) {}

    void push_ref(py::array_t<float, py::array::c_style | py::array::forcecast> frame) {
        auto r = frame.unchecked<1>();
        int n = (int)r.shape(0);
        std::vector<float> samples(n);
        for (int i = 0; i < n; ++i) {
            float value = r(i);
            if (!std::isfinite(value))
                throw py::value_error("reference frame must contain only finite samples");
            if (std::abs(value) > MATCHED_FILTER_AUDIO_ABS_LIMIT)
                throw py::value_error("reference frame exceeds normalized audio range [-16, 16]");
            samples[i] = value;
        }
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        if (ref_accum_.size() + samples.size() > MATCHED_FILTER_MAX_PENDING_SAMPLES)
            throw std::invalid_argument(
                "reference pending skew exceeds fixed limit 16384 samples");
        ref_raw_buf_.push(samples.data(), n);
        ref_accum_.insert(ref_accum_.end(), samples.begin(), samples.end());
        try {
            process_blocks();
        } catch (const std::exception& exc) {
            reset_locked();
            throw std::runtime_error(
                std::string("matched-filter internal non-finite state; estimator reset: ")
                + exc.what());
        }
    }

    void push_mic(py::array_t<float, py::array::c_style | py::array::forcecast> frame) {
        auto r = frame.unchecked<1>();
        int n = (int)r.shape(0);
        std::vector<float> samples(n);
        for (int i = 0; i < n; ++i) {
            float value = r(i);
            if (!std::isfinite(value))
                throw py::value_error("microphone frame must contain only finite samples");
            if (std::abs(value) > MATCHED_FILTER_AUDIO_ABS_LIMIT)
                throw py::value_error("microphone frame exceeds normalized audio range [-16, 16]");
            samples[i] = value;
        }
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        if (mic_accum_.size() + samples.size() > MATCHED_FILTER_MAX_PENDING_SAMPLES)
            throw std::invalid_argument(
                "microphone pending skew exceeds fixed limit 16384 samples");
        mic_accum_.insert(mic_accum_.end(), samples.begin(), samples.end());
        try {
            process_blocks();
        } catch (const std::exception& exc) {
            reset_locked();
            throw std::runtime_error(
                std::string("matched-filter internal non-finite state; estimator reset: ")
                + exc.what());
        }
    }

    float delay_ms() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_delay_ * 1000.f / sr_;
    }

    int delay_samples() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return (int)current_delay_;
    }

    py::array_t<float> get_aligned_ref_frame(int frame_size) {
        if (frame_size < 0)
            throw py::value_error("frame_size must be non-negative");
        auto out = py::array_t<float>(frame_size);
        if (frame_size == 0) return out;
        auto ptr = out.mutable_unchecked<1>();
        std::lock_guard<std::mutex> lock(mutex_);
        int delay = (int)current_delay_;
        int total = frame_size + delay;
        if (total > ref_raw_buf_.capacity())
            throw py::value_error("frame_size + delay exceeds reference buffer capacity");
        if (total <= 0) {
            ref_raw_buf_.read_newest(ptr.mutable_data(0), frame_size);
        } else {
            std::vector<float> block(total);
            ref_raw_buf_.read_newest(block.data(), total);
            std::memcpy(ptr.mutable_data(0), block.data(), frame_size * sizeof(float));
        }
        return out;
    }

    void reset() {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        reset_locked();
    }

private:
    // Caller holds mutex_ and has released the GIL.
    void reset_locked() {
        for (auto& f : filters_) std::fill(f.begin(), f.end(), 0.f);
        std::fill(histogram_.begin(), histogram_.end(), 0);
        std::fill(hist_ring_.begin(), hist_ring_.end(), 0);
        hist_idx_ = 0; hist_count_ = 0; significant_found_ = false;
        render_buf_.reset(); ref_raw_buf_.reset();
        std::memset(aa_ref_, 0, sizeof(aa_ref_)); std::memset(aa_mic_, 0, sizeof(aa_mic_));
        std::memset(hp_ref_, 0, sizeof(hp_ref_)); std::memset(hp_mic_, 0, sizeof(hp_mic_));
        current_delay_ = 0;
        ref_accum_.clear(); mic_accum_.clear();
    }

    void process_blocks() {
        int bs = block_size_;
        while ((int)ref_accum_.size() >= bs && (int)mic_accum_.size() >= bs) {
            // Decimate
            std::vector<float> ds_ref(sub_block_size_), ds_mic(sub_block_size_);
            int dn_ref, dn_mic;
            decimate(ref_accum_.data(), bs, ds_ref.data(), dn_ref, aa_ref_, hp_ref_);
            decimate(mic_accum_.data(), bs, ds_mic.data(), dn_mic, aa_mic_, hp_mic_);
            ref_accum_.erase(ref_accum_.begin(), ref_accum_.begin() + bs);
            mic_accum_.erase(mic_accum_.begin(), mic_accum_.begin() + bs);

            render_buf_.insert(ds_ref.data(), dn_ref);

            // Update all matched filters
            int best_lag = -1; float best_acc = -1e30f; bool any_reliable = false;
            for (int f = 0; f < num_filters_; ++f) {
                int lag; float acc; bool rel;
                nlms_update(f, ds_mic.data(), dn_mic, lag, acc, rel);
                if (rel && acc > best_acc) {
                    best_lag = lag; best_acc = acc; any_reliable = true;
                }
            }

            // Histogram voting
            if (any_reliable && best_lag >= 0 && best_lag <= max_lag_) {
                if (hist_count_ >= hist_window_) {
                    histogram_[hist_ring_[hist_idx_]]--;
                }
                hist_ring_[hist_idx_] = best_lag;
                histogram_[best_lag]++;
                hist_idx_ = (hist_idx_ + 1) % hist_window_;
                hist_count_ = std::min(hist_count_ + 1, hist_window_);
            }

            if (hist_count_ > 0) {
                int candidate = (int)(std::max_element(histogram_.begin(), histogram_.end()) - histogram_.begin());
                int count = histogram_[candidate];
                int thresh = significant_found_ ? converged_thresh_ : initial_thresh_;
                if (count >= thresh) {
                    if (count >= converged_thresh_) significant_found_ = true;
                    current_delay_ = (float)(candidate * ds_factor_);
                }
            }
        }
    }
};

// =========================================================================
// Python bindings
// =========================================================================
PYBIND11_MODULE(delay_estimator_ext, m) {
    m.doc() = "C++ accelerated delay estimators for AEC (GCC-PHAT+ACPS & MatchedFilter)";

    m.def("selected_lag_confidence", [](
              py::array_t<float, py::array::c_style | py::array::forcecast> mic,
              py::array_t<float, py::array::c_style | py::array::forcecast> ref,
              int lag, int candidate_count, double false_accept_prob) -> py::object {
        if (mic.ndim() != 1 || ref.ndim() != 1 || mic.size() != ref.size())
            return py::none();
        auto result = selected_lag_confidence(
            mic.data(), ref.data(), (int)mic.size(), lag,
            candidate_count, false_accept_prob);
        if (result == ConfidenceResult::invalid) return py::none();
        return py::bool_(result == ConfidenceResult::accepted);
    }, py::arg("mic"), py::arg("ref"), py::arg("lag"),
       py::arg("candidate_count"), py::arg("false_accept_prob"));

    py::class_<GccPhatEstimatorCpp>(m, "GccPhatEstimatorCpp")
        .def(py::init<int, float, float, float, float, float, float, float, float, float, float, float, float>(),
             py::arg("sr") = 16000,
             py::arg("buffer_ms") = 500.f,
             py::arg("coarse_ms") = 50.f,
             py::arg("residual_ms") = 50.f,
             py::arg("update_ms") = 100.f,
             py::arg("smooth_alpha") = 0.15f,
             py::arg("ref_thresh_db") = -50.f,
             py::arg("peak_thresh") = 0.05f,
             py::arg("prominence_db") = 6.f,
             py::arg("false_accept_prob") = 1e-4f,
             py::arg("acps_alpha") = 0.8f,
             py::arg("min_obs_ms") = 200.f,
             py::arg("eps") = 1e-12f)
        .def("push_ref", &GccPhatEstimatorCpp::push_ref)
        .def("push_mic", &GccPhatEstimatorCpp::push_mic)
        .def("get_aligned_ref_frame", &GccPhatEstimatorCpp::get_aligned_ref_frame)
        .def("reset", &GccPhatEstimatorCpp::reset)
        .def_property_readonly("delay_ms", &GccPhatEstimatorCpp::delay_ms)
        .def_property_readonly("delay_samples", &GccPhatEstimatorCpp::delay_samples)
        .def_property_readonly("current_delay", &GccPhatEstimatorCpp::delay_samples);

    py::class_<MatchedFilterEstimatorCpp>(m, "MatchedFilterEstimatorCpp")
        .def(py::init<int, int, int, int, int, float, float, int, int, int, int>(),
             py::arg("sr") = 16000,
             py::arg("ds") = 4,
             py::arg("num_filters") = 5,
             py::arg("flen_subblocks") = 32,
             py::arg("lag_shift_subblocks") = 24,
             py::arg("smoothing") = 0.7f,
             py::arg("excitation_limit") = 0.005f,
             py::arg("hist_window") = 250,
             py::arg("init_thresh") = 5,
             py::arg("conv_thresh") = 20,
             py::arg("block_size") = 64)
        .def("push_ref", &MatchedFilterEstimatorCpp::push_ref)
        .def("push_mic", &MatchedFilterEstimatorCpp::push_mic)
        .def("get_aligned_ref_frame", &MatchedFilterEstimatorCpp::get_aligned_ref_frame)
        .def("reset", &MatchedFilterEstimatorCpp::reset)
        .def_property_readonly("delay_ms", &MatchedFilterEstimatorCpp::delay_ms)
        .def_property_readonly("delay_samples", &MatchedFilterEstimatorCpp::delay_samples)
        .def_property_readonly("current_delay", &MatchedFilterEstimatorCpp::delay_samples);
}
