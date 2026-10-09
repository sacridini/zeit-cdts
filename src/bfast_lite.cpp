#include "bfast_lite.h"

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
#include <limits>
#include <stdexcept>
#include <Eigen/Dense>

namespace zeit {
namespace bfastlite {

namespace {

const double PI = 3.14159265358979323846;

// Same trend+harmonic design matrix as bfast_monitor.cpp's
// build_design_matrix (kept as an independent copy - each algorithm file in
// this codebase is self-contained). See R/bfastpp.R.
Eigen::MatrixXd build_design_matrix(int n_time, double start_time, int frequency, int order) {
    order = std::min(order, frequency);
    bool drop_nyquist = (2 * order == frequency);
    int n_cols = 2 + 2 * order - (drop_nyquist ? 1 : 0);

    Eigen::MatrixXd X(n_time, n_cols);
    for (int i = 0; i < n_time; ++i) {
        double t = start_time + (double)i / (double)frequency;
        X(i, 0) = 1.0;
        X(i, 1) = (double)(i + 1);
        int col = 2;
        for (int k = 1; k <= order; ++k) X(i, col++) = std::cos(2.0 * PI * t * (double)k);
        for (int k = 1; k <= order; ++k) {
            if (drop_nyquist && k == order) continue;
            X(i, col++) = std::sin(2.0 * PI * t * (double)k);
        }
    }
    return X;
}

// Recursive (Brown-Durbin-Evans) residuals for the segment [seg_start, n-1]
// of X/y, matching strucchangeRcpp's recresid.default (R/recresid.R) in
// spirit: residual r uses only the fit on rows before r, so the cumulative
// sum of squared residuals equals the OLS RSS of the growing segment - the
// standard trick that makes the segment-RSS table an O(n^2) computation
// instead of O(n^3) (a fresh OLS refit per candidate segment).
//
// Unlike R's version (and an earlier version of this port), this
// refactorizes the accumulated Gram matrix from scratch at every step
// (LDLT of a k x k matrix, k = number of regressors - cheap) instead of
// propagating (X'X)^-1 through a Sherman-Morrison rank-1 update. The
// bootstrap fit uses exactly k rows for k regressors (the minimum possible
// sample), which is frequently ill-conditioned (e.g. a handful of
// consecutive observations can't distinguish several harmonic terms with
// limited phase diversity yet) - propagating an inverse through that via
// Sherman-Morrison compounds the ill-conditioning over subsequent steps
// (verified empirically: it eventually drives the "1 + leverage" term
// negative, i.e. sqrt() of a negative number, NaN-poisoning the whole
// cumulative sum). Refitting from the actual accumulated (X'X) each step
// self-corrects instead of compounding, at the cost of O(k^3) instead of
// O(k^2) per step - negligible for the small k (a handful of harmonic +
// trend regressors) this is used for.
//
// Deliberately NOT ridge-regularized: the sum of squared recursive
// residuals is only exactly equal to the segment's OLS RSS (the identity
// this whole table depends on) for the *exact* least-squares fit at every
// step. Adding even a tiny ridge term at every one of the ~n steps
// (as an earlier version of this function did, to paper over the
// ill-conditioned first step) measurably biases the final cumulative sum -
// confirmed empirically against a direct OLS fit and against R. LDLT on
// the raw (unregularized) Gram matrix tolerates the ill-conditioned
// bootstrap step fine in practice (it only loses some precision there,
// not catastrophically), and does not compound like the Sherman-Morrison
// update did.
//
// Writes into `cum_rss` (cumulative sum of squared recursive residuals),
// length (n - seg_start - k); cum_rss[t] is the RSS of the OLS fit on
// [seg_start, seg_start + k + t] (t+k+1 points).
void recursive_cum_rss(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int seg_start, int n, int k,
                        std::vector<double>& cum_rss) {
    int len = n - seg_start;
    cum_rss.assign(std::max(0, len - k), 0.0);
    if (len <= k) return;

    Eigen::MatrixXd XtX = X.middleRows(seg_start, k).transpose() * X.middleRows(seg_start, k);
    Eigen::VectorXd Xty = X.middleRows(seg_start, k).transpose() * y.segment(seg_start, k);

    for (int t = 0; t < len - k; ++t) {
        Eigen::LDLT<Eigen::MatrixXd> ldlt(XtX);

        Eigen::VectorXd xr = X.row(seg_start + k + t);
        double yr = y(seg_start + k + t);
        Eigen::VectorXd X1xr = ldlt.solve(xr);
        double fr = 1.0 + xr.dot(X1xr);
        double betar_dot_xr = xr.dot(ldlt.solve(Xty));
        double w = (yr - betar_dot_xr) / std::sqrt(fr);
        cum_rss[t] = (t == 0 ? 0.0 : cum_rss[t - 1]) + w * w;

        XtX += xr * xr.transpose();
        Xty += xr * yr;
    }
}

// LWZ (Liu, Wu & Zidek, 1997) model-selection criterion for m breaks
// (m+1 segments), matching strucchangeRcpp's LWZ.breakpointsfull /
// AIC.breakpointsfull(k = 0.299*log(n)^2.1) applied to the segmented
// regression's Gaussian log-likelihood.
double lwz_score(double rss, int n, int n_regressors, int m_breaks) {
    double n_d = (double)n;
    double logl_penalty_free = n_d * (std::log(rss / n_d) + 1.0 + std::log(2.0 * PI));
    double df = (double)(n_regressors + 1) * (double)(m_breaks + 1);
    double lwz_k = 0.299 * std::pow(std::log(n_d), 2.1);
    return logl_penalty_free + lwz_k * df;
}

// Bai & Perron (2003) optimal multiple-breakpoint dynamic program, matching
// strucchangeRcpp's breakpoints.matrix() (R/breakpoints.R). Builds the
// segment-RSS triangle via recursive residuals, then the DP recursion
//   f_0(i) = RSS(0, i)
//   f_m(i) = min_{j} [ f_{m-1}(j) + RSS(j+1, i) ],  m*H-1 <= j <= i-H
// for candidate segment-end positions i, tracking the argmin j for
// backtracking. Evaluates the LWZ criterion at every feasible break count
//0..max_m and returns the LWZ-optimal one (matching bfastlite's own
// default `breaks="LWZ"`).
BFLResult bfast_lite_impl(const double* y_raw, int n_raw, const Eigen::MatrixXd& X_full,
                           double h, int max_breaks_output, int min_valid) {
    BFLResult res;
    int k = (int)X_full.cols();

    // Drop NaN rows (matches bfastpp's default na.action=na.omit), keeping
    // the surviving rows' original trend/harmonic regressor values.
    std::vector<int> valid_rows;
    valid_rows.reserve(n_raw);
    for (int i = 0; i < n_raw; ++i) if (!std::isnan(y_raw[i])) valid_rows.push_back(i);
    int n = (int)valid_rows.size();
    res.n_valid = (double)n;

    int H = (int)std::floor(h * (double)n);
    bool ok = (H > k) && (H <= n / 2) && (n >= min_valid);
    if (!ok) return res;
    res.valid = 1.0;

    Eigen::MatrixXd X(n, k);
    Eigen::VectorXd y(n);
    for (int r = 0; r < n; ++r) {
        X.row(r) = X_full.row(valid_rows[r]);
        y(r) = y_raw[valid_rows[r]];
    }

    // RSS triangle: rss_triang[s][t] = RSS of the OLS fit on [s, s+k+t].
    std::vector<std::vector<double>> rss_triang(n - H + 1);
    for (int s = 0; s <= n - H; ++s) {
        recursive_cum_rss(X, y, s, n, k, rss_triang[s]);
    }
    auto RSS = [&](int i, int j) -> double {
        // RSS of the OLS fit on rows [i, j] inclusive (0-indexed), requires j-i+1 >= k+1.
        return rss_triang[i][j - i - k];
    };

    int max_m = std::min(max_breaks_output, (int)std::ceil((double)n / (double)H) - 2);
    max_m = std::max(max_m, 0);

    const double NaN = std::numeric_limits<double>::quiet_NaN();
    // f[m][i] = optimal total RSS of segmenting [0, i] into exactly m+1
    // pieces (m breaks), each of size >= H - a *self-contained* quantity
    // for [0, i], not "m breaks with room left for more". So the m-break
    // answer for the *whole* series is simply f[m][n-1] directly - no extra
    // trailing segment glued on at extraction time (an earlier version of
    // this function did that, double-counting one segment; caught by the
    // R cross-validation, which kept landing on an extra, wrong break).
    std::vector<std::vector<double>> f(max_m + 1, std::vector<double>(n, NaN));
    std::vector<std::vector<int>> ptr(max_m + 1, std::vector<int>(n, -1));

    for (int i = H - 1; i <= n - 1; ++i) f[0][i] = RSS(0, i);

    int max_m_reached = 0;
    for (int m = 1; m <= max_m; ++m) {
        int i_lo = (m + 1) * H - 1, i_hi = n - 1;
        if (i_lo > i_hi) break;
        bool any_feasible = false;
        for (int i = i_lo; i <= i_hi; ++i) {
            int j_lo = m * H - 1, j_hi = i - H;
            double best = std::numeric_limits<double>::infinity();
            int best_j = -1;
            for (int j = j_lo; j <= j_hi; ++j) {
                if (std::isnan(f[m - 1][j])) continue;
                double val = f[m - 1][j] + RSS(j + 1, i);
                if (val < best) { best = val; best_j = j; }
            }
            if (best_j >= 0) { f[m][i] = best; ptr[m][i] = best_j; any_feasible = true; }
        }
        if (!any_feasible) break;
        max_m_reached = m;
    }

    // Evaluate LWZ at every feasible break count and pick the minimizer.
    double best_lwz = std::numeric_limits<double>::infinity();
    int best_m = 0;
    double best_rss = RSS(0, n - 1);
    std::vector<int> best_breaks;

    {
        double rss0 = RSS(0, n - 1);
        double lwz0 = lwz_score(rss0, n, k, 0);
        best_lwz = lwz0; best_m = 0; best_rss = rss0;
    }

    for (int m = 1; m <= max_m_reached; ++m) {
        if (std::isnan(f[m][n - 1])) continue;
        double best_total = f[m][n - 1];
        int best_last = n - 1;

        double lwz_m = lwz_score(best_total, n, k, m);
        if (lwz_m < best_lwz) {
            best_lwz = lwz_m;
            best_m = m;
            best_rss = best_total;

            // ptr[level][cur] gives the level-th breakpoint position itself
            // (not just a pointer to another table cell), so each backtrack
            // step both consumes and produces a breakpoint - see the
            // comment above the f[][] declaration.
            std::vector<int> breaks(m);
            int cur = best_last;
            for (int level = m; level >= 1; --level) {
                cur = ptr[level][cur];
                breaks[level - 1] = cur;
            }
            best_breaks = breaks;
        }
    }

    res.n_breaks = (double)best_m;
    res.rss = best_rss;
    res.lwz = best_lwz;
    res.breakpoint_idx.resize(best_m);
    for (int i = 0; i < best_m; ++i) {
        // Map back to the original (pre-NaN-drop) row index, matching
        // bfastpp's behavior of preserving each surviving row's own time.
        res.breakpoint_idx[i] = (double)valid_rows[best_breaks[i]];
    }

    // Magnitude of each break: the OLS model of each segment, and the right
    // segment's minus the left one's at the first observation after the
    // break (both models on the same date, so the seasonal terms cancel out
    // and only the change of level, trend and season shape remains).
    if (best_m > 0) {
        std::vector<Eigen::VectorXd> coefs(best_m + 1);
        int start = 0;
        for (int s = 0; s <= best_m; ++s) {
            int end = s < best_m ? best_breaks[s] : n - 1;
            int len = end - start + 1;
            coefs[s] = X.block(start, 0, len, k).colPivHouseholderQr().solve(y.segment(start, len));
            start = end + 1;
        }
        res.magnitude.resize(best_m);
        res.first_after_idx.resize(best_m);
        for (int i = 0; i < best_m; ++i) {
            res.magnitude[i] = X.row(best_breaks[i] + 1).dot(coefs[i + 1] - coefs[i]);
            res.first_after_idx[i] = (double)valid_rows[best_breaks[i] + 1];
        }
    }

    return res;
}

} // namespace

BFLResult bfast_lite(const std::vector<double>& y, double start_time, int frequency, int order, double h, int max_breaks_output) {
    Eigen::MatrixXd X = build_design_matrix((int)y.size(), start_time, frequency, order);
    return bfast_lite_impl(y.data(), (int)y.size(), X, h, max_breaks_output, 20);
}

pybind11::array_t<double> fit_bfast_lite_batch(
    pybind11::array_t<double> values_array,
    double start_time,
    int frequency,
    int order,
    double h,
    int max_breaks_output,
    int min_valid,
    int n_jobs)
{
    auto buf = values_array.request();
    if (buf.ndim != 2) throw std::runtime_error("values_array must be 2D [pixels, time]");

    int n_pixels = (int)buf.shape[0];
    int n_time = (int)buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);

    Eigen::MatrixXd X = build_design_matrix(n_time, start_time, frequency, order);

    const int n_metrics = 5 + 3 * max_breaks_output;
    pybind11::array_t<double> out_arr({n_metrics, n_pixels});
    double* out_ptr = static_cast<double*>(out_arr.request().ptr);
    for (int i = 0; i < n_metrics * n_pixels; ++i) out_ptr[i] = std::nan("");

    if (n_jobs <= 0) n_jobs = std::max(1, omp_get_max_threads() - 1);

    #pragma omp parallel num_threads(n_jobs)
    {
        #pragma omp for
        for (int p = 0; p < n_pixels; ++p) {
            const double* y = ptr + (size_t)p * n_time;
            BFLResult r = bfast_lite_impl(y, n_time, X, h, max_breaks_output, min_valid);

            out_ptr[0 * n_pixels + p] = r.n_breaks;
            out_ptr[1 * n_pixels + p] = r.rss;
            out_ptr[2 * n_pixels + p] = r.lwz;
            out_ptr[3 * n_pixels + p] = r.n_valid;
            out_ptr[4 * n_pixels + p] = r.valid;
            for (int b = 0; b < (int)r.breakpoint_idx.size() && b < max_breaks_output; ++b) {
                out_ptr[(size_t)(5 + b) * n_pixels + p] = r.breakpoint_idx[b];
                out_ptr[(size_t)(5 + max_breaks_output + b) * n_pixels + p] = r.magnitude[b];
                out_ptr[(size_t)(5 + 2 * max_breaks_output + b) * n_pixels + p] = r.first_after_idx[b];
            }
        }
    }

    return out_arr;
}

} // namespace bfastlite
} // namespace zeit
