#include "mann_kendall.h"

#ifndef _OPENMP
#ifndef OMP_DUMMIES_DEFINED
#define OMP_DUMMIES_DEFINED
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_set_num_threads(x) (void)(x)
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>

namespace zeit {
namespace mannkendall {

namespace {

// ---- per-thread scratch buffers ---------------------------------------
//
// The hot path (fit_mann_kendall_batch) used to allocate ~10 fresh
// std::vectors per pixel (drop_nan, variance_s's sorted copy, sens_slope's
// pairwise-differences array, rankdata's index/rank arrays, acf's centered
// copy and result, ...). Benchmarking showed that for realistic annual
// series lengths (n=8-30) this fixed per-pixel allocation overhead is
// 30-80% of the total time, and dwarfs the O(n^2) arithmetic itself for
// short series - and it makes multi-threaded scaling sub-linear (allocator
// contention). MKScratch pools all of those buffers once per OpenMP thread
// (not per pixel): every helper below now writes into caller-owned buffers
// instead of returning freshly-allocated vectors, and callers just
// `.clear()`/`.resize()` a persistent buffer between pixels instead of
// reallocating. Capacities are sized once from n_time (constant for an
// entire fit_mann_kendall_batch call), so per-pixel `.resize()`/`.assign()`
// calls never grow the underlying allocation.
struct MKScratch {
    std::vector<double> y;             // raw per-pixel series (with NaNs)
    std::vector<double> x;             // NaN-dropped series
    std::vector<double> sorted_x;      // variance_s: sorted copy of x
    std::vector<double> tie_sizes;     // variance_s: tie-group sizes
    std::vector<double> pairwise;      // sens_slope_full: pairwise slopes
    std::vector<double> valid_vals;    // sens_slope_full: non-NaN values
    std::vector<double> valid_idx;     // sens_slope_full: non-NaN indices
    std::vector<double> x_detrend;     // mk_test_impl: detrended series
    std::vector<int>    rank_idx;      // rankdata: sort permutation
    std::vector<double> rank_out;      // rankdata: output ranks
    std::vector<double> acf_centered;  // acf: mean-centered copy
    std::vector<double> acf_out;       // acf: output autocorrelations

    std::vector<std::vector<double>> seasonal_cols;   // seasonal reshape
    std::vector<double> seasonal_pooled;              // seasonal Sen's slope pairs
    std::vector<double> seasonal_valid_vals;
    std::vector<double> seasonal_valid_idx;

    void reserve_for(int n_time, int period) {
        size_t pair_cap = (n_time > 1) ? (size_t)n_time * (n_time - 1) / 2 : 0;

        y.reserve(n_time);
        x.reserve(n_time);
        sorted_x.reserve(n_time);
        tie_sizes.reserve(n_time);
        pairwise.reserve(pair_cap);
        valid_vals.reserve(n_time);
        valid_idx.reserve(n_time);
        x_detrend.reserve(n_time);
        rank_idx.reserve(n_time);
        rank_out.reserve(n_time);
        acf_centered.reserve(n_time);
        acf_out.reserve(n_time + 1);

        int p = std::max(period, 1);
        int rows = (n_time + p - 1) / p;
        seasonal_cols.resize(p);
        for (auto& col : seasonal_cols) col.reserve(rows);
        seasonal_pooled.reserve(pair_cap);
        seasonal_valid_vals.reserve(n_time);
        seasonal_valid_idx.reserve(n_time);
    }
};

// ---- small numeric helpers -------------------------------------------

// In-place nth_element median. Takes the buffer by mutable reference (no
// copy) since every call site below owns a scratch buffer it no longer
// needs in a particular order afterward.
double median(std::vector<double>& v) {
    if (v.empty()) return std::nan("");
    size_t n = v.size();
    size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double hi = v[mid];
    if (n % 2 == 1) return hi;
    std::nth_element(v.begin(), v.begin() + mid - 1, v.begin() + mid);
    double lo = v[mid - 1];
    return 0.5 * (lo + hi);
}

void drop_nan(const std::vector<double>& x, std::vector<double>& out) {
    out.clear();
    for (double v : x) {
        if (!std::isnan(v)) out.push_back(v);
    }
}

double norm_pdf(double x) {
    static const double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

// Exact standard normal CDF via std::erfc.
double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// Inverse standard normal CDF: Acklam's rational approximation, refined
// with one Newton step against the exact erfc-based CDF above (~1e-15).
double norm_ppf(double p) {
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return std::numeric_limits<double>::infinity();

    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01,
        2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00,
        2.938163982698783e+00};
    static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};

    const double p_low = 0.02425;
    double x;
    if (p < p_low) {
        double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= 1.0 - p_low) {
        double q = p - 0.5;
        double r = q*q;
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
              ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }

    double e = norm_cdf(x) - p;
    double u = e / norm_pdf(x);
    x = x - u / (1.0 + x * u / 2.0);
    return x;
}

// ---- Mann-Kendall building blocks (mirror pymannkendall's private fns) --

// __mk_score: sum of signs over all i<j pairs. O(n^2), inherent to the test.
double mk_score(const std::vector<double>& x) {
    int n = (int)x.size();
    double s = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (x[j] > x[i]) s += 1.0;
            else if (x[j] < x[i]) s -= 1.0;
        }
    }
    return s;
}

// __variance_s, with the tie correction. Writes its working buffers into
// the caller-owned sorted_x_buf/tie_sizes_buf instead of allocating them.
double variance_s(const std::vector<double>& x, std::vector<double>& sorted_x_buf, std::vector<double>& tie_sizes_buf) {
    int n = (int)x.size();
    sorted_x_buf.assign(x.begin(), x.end());
    std::sort(sorted_x_buf.begin(), sorted_x_buf.end());

    tie_sizes_buf.clear();
    int i = 0;
    while (i < n) {
        int j = i;
        while (j < n && sorted_x_buf[j] == sorted_x_buf[i]) ++j;
        tie_sizes_buf.push_back((double)(j - i));
        i = j;
    }

    int g = (int)tie_sizes_buf.size();
    double nd = (double)n;
    if (g == n) {
        return nd*(nd-1.0)*(2.0*nd+5.0)/18.0;
    }
    double tie_sum = 0.0;
    for (double t : tie_sizes_buf) tie_sum += t*(t-1.0)*(2.0*t+5.0);
    return (nd*(nd-1.0)*(2.0*nd+5.0) - tie_sum) / 18.0;
}

double z_score(double s, double var_s) {
    // No var_s guard here, matching pymannkendall exactly: sqrt of a
    // negative var_s (possible after the Hamed-Rao correction, which is not
    // clamped) yields NaN via IEEE754, same as numpy's np.sqrt.
    if (s > 0.0) return (s - 1.0) / std::sqrt(var_s);
    if (s < 0.0) return (s + 1.0) / std::sqrt(var_s);
    return 0.0;
}

struct PValueResult { double p; bool h; int trend; };

PValueResult p_value(double z, double alpha) {
    double p = 2.0 * (1.0 - norm_cdf(std::fabs(z)));
    bool h = std::fabs(z) > norm_ppf(1.0 - alpha / 2.0);
    int trend = 0;
    if (z < 0.0 && h) trend = -1;
    else if (z > 0.0 && h) trend = 1;
    return {p, h, trend};
}

// __acf: autocorrelation function up to nlags (inclusive), matching
// np.correlate(y, y, 'full')[n-1:] / n, normalized by lag-0 autocovariance.
// Writes into caller-owned centered_buf/result instead of allocating them.
void acf(const std::vector<double>& x, int nlags, std::vector<double>& centered_buf, std::vector<double>& result) {
    int n = (int)x.size();
    nlags = std::max(0, std::min(nlags, n - 1));

    double mean = std::accumulate(x.begin(), x.end(), 0.0) / n;
    centered_buf.resize(n);
    for (int i = 0; i < n; ++i) centered_buf[i] = x[i] - mean;

    double acov0 = 0.0;
    for (int i = 0; i < n; ++i) acov0 += centered_buf[i] * centered_buf[i];
    acov0 /= n;

    result.assign(nlags + 1, 0.0);
    for (int lag = 0; lag <= nlags; ++lag) {
        double s = 0.0;
        for (int i = 0; i < n - lag; ++i) s += centered_buf[i] * centered_buf[i + lag];
        s /= n;
        result[lag] = (acov0 != 0.0) ? s / acov0 : s;
    }
}

// scipy.stats.rankdata(method='average') equivalent. Writes into
// caller-owned idx_buf/ranks instead of allocating them.
void rankdata(const std::vector<double>& x, std::vector<int>& idx_buf, std::vector<double>& ranks) {
    int n = (int)x.size();
    idx_buf.resize(n);
    std::iota(idx_buf.begin(), idx_buf.end(), 0);
    std::sort(idx_buf.begin(), idx_buf.end(), [&](int a, int b) { return x[a] < x[b]; });

    ranks.resize(n);
    int i = 0;
    while (i < n) {
        int j = i;
        while (j < n && x[idx_buf[j]] == x[idx_buf[i]]) ++j;
        double avg_rank = (i + j + 1) / 2.0; // 1-indexed average rank
        for (int k = i; k < j; ++k) ranks[idx_buf[k]] = avg_rank;
        i = j;
    }
}

// sens_slope(x_old): computed on the ORIGINAL (possibly NaN-gapped) series,
// using true positional spacing (j-i) as the time-step denominator, and
// nanmedian-style skipping of any pair touching a NaN - mirrors
// pymannkendall's sens_slope() operating on x_old rather than the
// NaN-compacted array, so missing observations don't collapse the time axis.
// Writes into caller-owned pairwise_buf/valid_vals_buf/valid_idx_buf.
void sens_slope_full(const std::vector<double>& x_old, double& slope, double& intercept,
                      std::vector<double>& pairwise_buf, std::vector<double>& valid_vals_buf,
                      std::vector<double>& valid_idx_buf) {
    int n = (int)x_old.size();
    pairwise_buf.clear();
    for (int i = 0; i < n - 1; ++i) {
        if (std::isnan(x_old[i])) continue;
        for (int j = i + 1; j < n; ++j) {
            if (std::isnan(x_old[j])) continue;
            pairwise_buf.push_back((x_old[j] - x_old[i]) / (double)(j - i));
        }
    }
    slope = median(pairwise_buf);

    valid_vals_buf.clear();
    valid_idx_buf.clear();
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(x_old[i])) {
            valid_vals_buf.push_back(x_old[i]);
            valid_idx_buf.push_back((double)i);
        }
    }
    intercept = median(valid_vals_buf) - median(valid_idx_buf) * slope;
}

// Core original/Hamed-Rao/Yue-Wang test (SEASONAL is handled separately).
MKResult mk_test_impl(const std::vector<double>& y_old, MKMethod method, double alpha, int lag, MKScratch& sc) {
    MKResult res;
    drop_nan(y_old, sc.x);
    std::vector<double>& x = sc.x;
    int n = (int)x.size();
    if (n < 4) return res;

    double s = mk_score(x);
    double var_s = variance_s(x, sc.sorted_x, sc.tie_sizes);
    double tau = s / (0.5 * n * (n - 1));

    double slope, intercept;
    sens_slope_full(y_old, slope, intercept, sc.pairwise, sc.valid_vals, sc.valid_idx);

    if (method == MKMethod::HAMED_RAO || method == MKMethod::YUE_WANG) {
        int L = (lag < 0) ? n : (lag + 1);
        L = std::max(1, std::min(L, n));

        sc.x_detrend.resize(n);
        for (int i = 0; i < n; ++i) sc.x_detrend[i] = x[i] - (double)(i + 1) * slope;

        double interval = norm_ppf(1.0 - alpha / 2.0) / std::sqrt((double)n);
        double sni = 0.0;

        if (method == MKMethod::HAMED_RAO) {
            rankdata(sc.x_detrend, sc.rank_idx, sc.rank_out);
            acf(sc.rank_out, L - 1, sc.acf_centered, sc.acf_out);
            for (int i = 1; i < L; ++i) {
                if (!(sc.acf_out[i] <= interval && sc.acf_out[i] >= -interval)) {
                    sni += (double)(n - i) * (n - i - 1) * (n - i - 2) * sc.acf_out[i];
                }
            }
            // No abs() on sni here: matches the current pymannkendall (>=1.4.3)
            // formula, which fixed an earlier abs(sni) bug present in some
            // older releases (negative net autocorrelation should be able to
            // reduce var_s, not just inflate it).
            double n_ns = 1.0 + (2.0 / ((double)n * (n - 1) * (n - 2))) * sni;
            var_s *= n_ns;
        } else {
            acf(sc.x_detrend, L - 1, sc.acf_centered, sc.acf_out);
            for (int i = 1; i < L; ++i) {
                sni += (1.0 - (double)i / n) * sc.acf_out[i];
            }
            double n_ns = 1.0 + 2.0 * sni;
            var_s *= n_ns;
        }
    }

    double z = z_score(s, var_s);
    PValueResult pv = p_value(z, alpha);

    res.trend = pv.trend;
    res.h = pv.h;
    res.p = pv.p;
    res.z = z;
    res.tau = tau;
    res.s = s;
    res.var_s = var_s;
    res.slope = slope;
    res.intercept = intercept;
    return res;
}

// Reshape a flat series into `period` season columns, padding the tail with
// NaN (mirrors pymannkendall's seasonal_test reshape). Writes into the
// caller-owned `cols` (already sized to `period` entries by MKScratch::reserve_for).
void reshape_seasonal(const std::vector<double>& x_old, int period, std::vector<std::vector<double>>& cols) {
    int n = (int)x_old.size();
    int rows = (n + period - 1) / period;
    for (auto& col : cols) col.assign(rows, std::nan(""));
    for (int i = 0; i < n; ++i) {
        cols[i % period][i / period] = x_old[i];
    }
}

// seasonal_test == multivariate_test over the reshaped columns, with
// seasonal_sens_slope for the slope/intercept (pymannkendall).
MKResult seasonal_mk_test(const std::vector<double>& y_old, int period, double alpha, MKScratch& sc) {
    MKResult res;
    if (period < 2) period = 1;

    reshape_seasonal(y_old, period, sc.seasonal_cols);
    std::vector<std::vector<double>>& cols = sc.seasonal_cols;

    double s = 0.0, var_s = 0.0, denom = 0.0;
    for (int c = 0; c < period; ++c) {
        drop_nan(cols[c], sc.x); // sc.x reused as generic scratch (seasonal/non-seasonal paths never overlap)
        int nc = (int)sc.x.size();
        if (nc < 2) continue;
        s += mk_score(sc.x);
        var_s += variance_s(sc.x, sc.sorted_x, sc.tie_sizes);
        denom += 0.5 * nc * (nc - 1);
    }
    if (denom <= 0.0) return res;

    double tau = s / denom;
    double z = z_score(s, var_s);
    PValueResult pv = p_value(z, alpha);

    // Seasonal Sen's slope: pool per-column pairwise slopes, where the
    // denominator is the ROW gap (i.e. in units of one full `period` cycle).
    sc.seasonal_pooled.clear();
    for (int c = 0; c < period; ++c) {
        const std::vector<double>& col = cols[c];
        int rows = (int)col.size();
        for (int i = 0; i < rows - 1; ++i) {
            if (std::isnan(col[i])) continue;
            for (int j = i + 1; j < rows; ++j) {
                if (std::isnan(col[j])) continue;
                sc.seasonal_pooled.push_back((col[j] - col[i]) / (double)(j - i));
            }
        }
    }
    double slope = median(sc.seasonal_pooled);

    sc.seasonal_valid_vals.clear();
    sc.seasonal_valid_idx.clear();
    for (int i = 0; i < (int)y_old.size(); ++i) {
        if (!std::isnan(y_old[i])) {
            sc.seasonal_valid_vals.push_back(y_old[i]);
            sc.seasonal_valid_idx.push_back((double)i);
        }
    }
    double intercept = median(sc.seasonal_valid_vals) - (median(sc.seasonal_valid_idx) / period) * slope;

    res.trend = pv.trend;
    res.h = pv.h;
    res.p = pv.p;
    res.z = z;
    res.tau = tau;
    res.s = s;
    res.var_s = var_s;
    res.slope = slope;
    res.intercept = intercept;
    return res;
}

} // namespace

MKResult mk_test(const std::vector<double>& y, MKMethod method, double alpha, int lag, int period) {
    // Low-frequency single-series entry point (unit tests / interactive use):
    // allocating one scratch buffer here is negligible; the batch path below
    // allocates a scratch once per OpenMP thread instead of once per call.
    MKScratch sc;
    sc.reserve_for((int)y.size(), period);
    if (method == MKMethod::SEASONAL) return seasonal_mk_test(y, period, alpha, sc);
    return mk_test_impl(y, method, alpha, lag, sc);
}

pybind11::tuple mk_test_single(std::vector<double> y, int method, double alpha, int lag, int period) {
    MKResult r = mk_test(y, static_cast<MKMethod>(method), alpha, lag, period);
    return pybind11::make_tuple(r.trend, r.h, r.p, r.z, r.tau, r.s, r.var_s, r.slope, r.intercept);
}

pybind11::array_t<double> fit_mann_kendall_batch(
    pybind11::array_t<double> values_array,
    int method_int,
    double alpha,
    int lag,
    int period,
    int min_valid,
    int n_jobs)
{
    auto buf = values_array.request();
    if (buf.ndim != 2) throw std::runtime_error("values_array must be 2D [pixels, time]");

    int n_pixels = (int)buf.shape[0];
    int n_time = (int)buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);

    const int n_metrics = 9; // trend, h, p, z, tau, s, var_s, slope, intercept
    pybind11::array_t<double> out_arr({n_metrics, n_pixels});
    double* out_ptr = static_cast<double*>(out_arr.request().ptr);
    for (int i = 0; i < n_metrics * n_pixels; ++i) out_ptr[i] = std::nan("");

    if (n_jobs <= 0) n_jobs = std::max(1, omp_get_max_threads() - 1);
    MKMethod method = static_cast<MKMethod>(method_int);

    {
        pybind11::gil_scoped_release release;
        #pragma omp parallel num_threads(n_jobs)
        {
            // One scratch buffer set per OpenMP thread (not per pixel) - see
            // MKScratch's comment above for why this matters.
            MKScratch scratch;
            scratch.reserve_for(n_time, period);

            #pragma omp for
            for (int p = 0; p < n_pixels; ++p) {
                scratch.y.resize(n_time);
                int valid_count = 0;
                for (int t = 0; t < n_time; ++t) {
                    double v = ptr[(size_t)p * n_time + t];
                    scratch.y[t] = v;
                    if (!std::isnan(v)) ++valid_count;
                }
                if (valid_count < min_valid) continue;

                MKResult r = (method == MKMethod::SEASONAL)
                    ? seasonal_mk_test(scratch.y, period, alpha, scratch)
                    : mk_test_impl(scratch.y, method, alpha, lag, scratch);

                out_ptr[0 * n_pixels + p] = (double)r.trend;
                out_ptr[1 * n_pixels + p] = r.h ? 1.0 : 0.0;
                out_ptr[2 * n_pixels + p] = r.p;
                out_ptr[3 * n_pixels + p] = r.z;
                out_ptr[4 * n_pixels + p] = r.tau;
                out_ptr[5 * n_pixels + p] = r.s;
                out_ptr[6 * n_pixels + p] = r.var_s;
                out_ptr[7 * n_pixels + p] = r.slope;
                out_ptr[8 * n_pixels + p] = r.intercept;
            }
        }
    }

    return out_arr;
}

} // namespace mannkendall
} // namespace zeit
