#include "bfast.h"
#include "stl_decompose.h"

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
namespace bfast {

namespace {

const double PI = 3.14159265358979323846;

// `~ trend` design matrix (bfastpp's `trend = 1:NROW(y)` convention, same as
// bfast_lite.cpp/bfast_monitor.cpp - an affine reparametrization of R's
// fractional-year `ti`, which doesn't change the OLS fit or breakpoints).
Eigen::MatrixXd build_trend_design_matrix(int n_time) {
    Eigen::MatrixXd X(n_time, 2);
    for (int i = 0; i < n_time; ++i) {
        X(i, 0) = 1.0;
        X(i, 1) = (double)(i + 1);
    }
    return X;
}

// `~ harmonic` design matrix, no intercept/trend column - matches R bfast()'s
// `season == "harmonic"` model (`Wt ~ co + si + co2 + si2 + co3 + si3`, R/bfast.R),
// order fixed at 3 there but kept configurable here as in bfast_lite/bfast_monitor.
Eigen::MatrixXd build_harmonic_design_matrix(int n_time, double start_time, int frequency, int order) {
    order = std::min(order, frequency);
    bool drop_nyquist = (2 * order == frequency);
    int n_cols = 2 * order - (drop_nyquist ? 1 : 0);

    Eigen::MatrixXd X(n_time, n_cols);
    for (int i = 0; i < n_time; ++i) {
        double t = start_time + (double)i / (double)frequency;
        int col = 0;
        for (int k = 1; k <= order; ++k) X(i, col++) = std::cos(2.0 * PI * t * (double)k);
        for (int k = 1; k <= order; ++k) {
            if (drop_nyquist && k == order) continue;
            X(i, col++) = std::sin(2.0 * PI * t * (double)k);
        }
    }
    return X;
}

// Recursive (Brown-Durbin-Evans) cumulative-RSS table - identical in spirit
// to bfast_lite.cpp's own copy (each algorithm file in this codebase is
// self-contained; see that file's extensive comments for why the Gram
// matrix is refactorized from scratch each step instead of a Sherman-
// Morrison update).
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

// BIC model-selection criterion for m breaks (m+1 segments), matching
// strucchangeRcpp's `BIC.breakpointsfull` (= `AIC.breakpointsfull(k = log(n))`)
// applied to the segmented regression's Gaussian log-likelihood - this is
// `breakpoints()`'s own default selection rule when `breaks = NULL`, which
// is what classic bfast() uses (unlike bfastlite's `breaks = "LWZ"` default;
// see bfast_lite.cpp's `lwz_score`, the only difference being the penalty
// constant multiplying `df`).
double bic_score(double rss, int n, int n_regressors, int m_breaks) {
    double n_d = (double)n;
    double logl_penalty_free = n_d * (std::log(rss / n_d) + 1.0 + std::log(2.0 * PI));
    double df = (double)(n_regressors + 1) * (double)(m_breaks + 1);
    return logl_penalty_free + std::log(n_d) * df;
}

// ---- Retrospective OLS-MOSUM structural-stability pre-check -------------
// Ported from strucchangeRcpp's `efp(..., type="OLS-MOSUM")` process +
// `sctest.efp`/`pvalue.efp` for the "Brownian bridge increments" limiting
// process, `functional="max"`, `alt.boundary=FALSE` (all defaults) - the
// exact call bfast() itself makes (`sctest(efp(formula, h=h,
// type="OLS-MOSUM"))`) before every `breakpoints()` search. Critical-value
// table `sc.me` (60 rows = 6 k-values x 10 h-values, 4 columns = significance
// levels 0.1/0.05/0.025/0.01) dumped verbatim from the installed
// strucchangeRcpp R package (`get("sc.me", envir=asNamespace("strucchangeRcpp"))`)
// - including its row 52 (k=2 block... k=6,h=0.10) outlier in column 4
// (1.6220, breaking the otherwise-smooth pattern), which is preserved
// exactly since matching R's actual behavior (quirks included) is the goal,
// not a "corrected" table R itself doesn't use.
const double kScMe[6][10][4] = {
    { // k = 1
        {0.7552, 0.8017, 0.8444, 0.8977}, {0.9809, 1.0483, 1.1119, 1.1888},
        {1.1211, 1.2059, 1.2845, 1.3767}, {1.2170, 1.3158, 1.4053, 1.5131},
        {1.2811, 1.3920, 1.4917, 1.6118}, {1.3258, 1.4448, 1.5548, 1.6863},
        {1.3514, 1.4789, 1.5946, 1.7339}, {1.3628, 1.4956, 1.6152, 1.7572},
        {1.3610, 1.4976, 1.6210, 1.7676}, {1.3751, 1.5115, 1.6341, 1.7808},
    },
    { // k = 2
        {0.7997, 0.8431, 0.8838, 0.9351}, {1.0448, 1.1067, 1.1654, 1.2388},
        {1.2030, 1.2805, 1.3509, 1.4362}, {1.3112, 1.4042, 1.4881, 1.5876},
        {1.3870, 1.4865, 1.5779, 1.6930}, {1.4422, 1.5538, 1.6530, 1.7724},
        {1.4707, 1.5900, 1.6953, 1.8223}, {1.4892, 1.6105, 1.7206, 1.8559},
        {1.4902, 1.6156, 1.7297, 1.8668}, {1.5067, 1.6319, 1.7455, 1.8827},
    },
    { // k = 3
        {0.8250, 0.8668, 0.9040, 0.9519}, {1.0802, 1.1419, 1.1986, 1.2700},
        {1.2491, 1.3259, 1.3951, 1.4820}, {1.3647, 1.4516, 1.5326, 1.6302},
        {1.4449, 1.5421, 1.6322, 1.7470}, {1.5045, 1.6089, 1.7008, 1.8143},
        {1.5353, 1.6560, 1.7510, 1.8756}, {1.5588, 1.6751, 1.7809, 1.9105},
        {1.5630, 1.6828, 1.7901, 1.9190}, {1.5785, 1.6981, 1.8071, 1.9395},
    },
    { // k = 4
        {0.8414, 0.8828, 0.9205, 0.9681}, {1.1066, 1.1663, 1.2217, 1.2918},
        {1.2792, 1.3533, 1.4212, 1.5013}, {1.3973, 1.4506, 1.5593, 1.6536},
        {1.4852, 1.5791, 1.6690, 1.7741}, {1.5429, 1.6465, 1.7420, 1.8573},
        {1.5852, 1.6927, 1.7941, 1.9140}, {1.6057, 1.7195, 1.8212, 1.9450},
        {1.6089, 1.7245, 1.8269, 1.9592}, {1.6275, 1.7435, 1.8495, 1.9787},
    },
    { // k = 5
        {0.8541, 0.8948, 0.9321, 0.9799}, {1.1247, 1.1846, 1.2395, 1.3088},
        {1.3040, 1.3765, 1.4440, 1.5252}, {1.4250, 1.5069, 1.5855, 1.6791},
        {1.5154, 1.6077, 1.6921, 1.7967}, {1.5738, 1.6770, 1.7687, 1.8837},
        {1.6182, 1.7217, 1.8176, 1.9377}, {1.6460, 1.7540, 1.8553, 1.9788},
        {1.6462, 1.7574, 1.8615, 1.9897}, {1.6644, 1.7777, 1.8816, 2.0085},
    },
    { // k = 6
        {0.8653, 0.9048, 0.9414, 0.9880}, {1.1415, 1.1997, 1.2530, 1.6220},
        {1.3223, 1.3938, 1.4596, 1.5392}, {1.4483, 1.5305, 1.6100, 1.7014},
        {1.5392, 1.6317, 1.7139, 1.8154}, {1.6025, 1.7018, 1.7930, 1.9061},
        {1.6462, 1.7499, 1.8439, 1.9605}, {1.6697, 1.7769, 1.8763, 1.9986},
        {1.6802, 1.7889, 1.8932, 2.0163}, {1.6939, 1.8052, 1.9074, 2.0326},
    },
};

// R's `approx(x, y, xout, rule=2)`: piecewise-linear interpolation, clamped
// to the endpoint y-values outside [x[0], x[back]]. Requires x sorted
// ascending.
double approx_rule2(const std::vector<double>& xs, const std::vector<double>& ys, double xout) {
    int n = (int)xs.size();
    if (xout <= xs[0]) return ys[0];
    if (xout >= xs[n - 1]) return ys[n - 1];
    for (int i = 0; i < n - 1; ++i) {
        if (xout >= xs[i] && xout <= xs[i + 1]) {
            double frac = (xout - xs[i]) / (xs[i + 1] - xs[i]);
            return ys[i] + frac * (ys[i + 1] - ys[i]);
        }
    }
    return ys[n - 1];
}

// `pvalue.efp(stat, "Brownian bridge increments", alt.boundary=FALSE,
// functional="max", h, k)`.
double pvalue_bbi_max(double stat, int k, double h) {
    k = std::max(1, std::min(k, 6));
    static const std::vector<double> table_h = {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50};
    static const std::vector<double> table_p = {0.1, 0.05, 0.025, 0.01};

    std::vector<double> crit_at_h(4);
    for (int col = 0; col < 4; ++col) {
        std::vector<double> col_vals(10);
        for (int row = 0; row < 10; ++row) col_vals[row] = kScMe[k - 1][row][col];
        crit_at_h[col] = approx_rule2(table_h, col_vals, h);
    }

    std::vector<double> xs = {0.0, crit_at_h[0], crit_at_h[1], crit_at_h[2], crit_at_h[3]};
    std::vector<double> ys = {1.0, table_p[0], table_p[1], table_p[2], table_p[3]};
    return approx_rule2(xs, ys, stat);
}

// `sctest(efp(y ~ X, h, type="OLS-MOSUM"))$p.value` for the whole (single-
// segment) fit - matches R's `sdev()` helper exactly (centers on the
// residuals' own mean before computing the variance, which matters for a
// no-intercept design like the harmonic-only season model, where OLS
// residuals don't necessarily sum to zero).
double ols_mosum_sctest_pvalue(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double h) {
    int n = (int)X.rows(), k = (int)X.cols();
    int df = n - k;
    int nh = (int)std::floor(h * (double)n);
    if (df <= 0 || nh < 1 || nh > n) return 1.0;

    Eigen::VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);
    Eigen::VectorXd e = y - X * beta;
    double mean_e = e.mean();
    double sigma = std::sqrt((e.array() - mean_e).square().sum() / (double)df);
    if (!(sigma > 0.0) || !std::isfinite(sigma)) return 1.0;

    std::vector<double> P(n + 1, 0.0);
    for (int i = 0; i < n; ++i) P[i + 1] = P[i] + e(i);

    double denom = sigma * std::sqrt((double)n);
    double max_abs = 0.0;
    for (int j = 0; j <= n - nh; ++j) {
        double w = (P[nh + j] - P[j]) / denom;
        max_abs = std::max(max_abs, std::fabs(w));
    }

    return pvalue_bbi_max(max_abs, k, h);
}

// Result of one segmented-regression fit (either the trend or the season
// sub-model, for one bfast() iteration): the optimal (BIC-minimizing)
// breakpoint set, per-row fitted values (needed to form the *other*
// component's residual for the next half-step), and per-segment
// coefficients (needed for the trend model's magnitude/time computation).
struct SegFit {
    int n_breaks = 0;
    std::vector<int> breakpoints;             // 0-based end-of-segment row indices (within the fitted rows), size n_breaks
    std::vector<double> fitted;                // length n
    std::vector<Eigen::VectorXd> seg_coeffs;   // size n_breaks+1, one OLS coefficient vector per segment
    double rss = std::numeric_limits<double>::quiet_NaN();
};

// Reusable buffers for fit_segmented's RSS triangle + DP tables. Since n, h
// and X (hence H, max_m) are the same across all of a pixel's iterations for
// a given component (only y changes iteration to iteration), and typically
// the same across an entire batch call too (all pixels share n_time), these
// buffers are sized once and then just overwritten on every subsequent
// call - `.resize()`/loop-fill below are no-ops once at steady-state size,
// avoiding the ~n small heap allocations `fit_segmented` used to make on
// every single call (bfast() calls this up to ~2*max_iter times per pixel,
// vs. bfastlite's one call per pixel - see bfast.md's Performance section).
struct FitScratch {
    std::vector<std::vector<double>> rss_triang;
    std::vector<std::vector<double>> f;
    std::vector<std::vector<int>> ptr;
};

SegFit fit_segmented(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double h, int max_breaks_output,
                      FitScratch& sc, bool skip_search = false) {
    int n = (int)X.rows();
    int k = (int)X.cols();
    SegFit out;
    out.fitted.resize(n);

    auto fit_range = [&](int a, int b, Eigen::VectorXd& beta_out) {
        int len = b - a + 1;
        Eigen::MatrixXd Xs = X.middleRows(a, len);
        Eigen::VectorXd ys = y.segment(a, len);
        beta_out = (Xs.transpose() * Xs).ldlt().solve(Xs.transpose() * ys);
        Eigen::VectorXd fit = Xs * beta_out;
        for (int r = 0; r < len; ++r) out.fitted[a + r] = fit(r);
        return (ys - fit).squaredNorm();
    };

    if (skip_search) {
        // Matches R's `nobp` path taken when the preliminary sctest.efp
        // pre-check doesn't reject stability: fit once, don't search breaks.
        Eigen::VectorXd beta;
        out.rss = fit_range(0, n - 1, beta);
        out.seg_coeffs.push_back(beta);
        return out;
    }

    int H = (int)std::floor(h * (double)n);
    if (!(H > k && H <= n / 2)) {
        // Too short to consider even one break at this h - fall back to a
        // single-segment OLS fit (matches bfast()'s own "nobp" no-break path).
        Eigen::VectorXd beta;
        out.rss = fit_range(0, n - 1, beta);
        out.seg_coeffs.push_back(beta);
        return out;
    }

    sc.rss_triang.resize(n - H + 1); // no-op once at steady-state size
    for (int s = 0; s <= n - H; ++s) recursive_cum_rss(X, y, s, n, k, sc.rss_triang[s]);
    auto RSS = [&](int i, int j) -> double { return sc.rss_triang[i][j - i - k]; };

    int max_m = std::min(max_breaks_output, (int)std::ceil((double)n / (double)H) - 2);
    max_m = std::max(max_m, 0);

    const double NaN = std::numeric_limits<double>::quiet_NaN();
    if ((int)sc.f.size() != max_m + 1) { sc.f.assign(max_m + 1, std::vector<double>(n, NaN)); sc.ptr.assign(max_m + 1, std::vector<int>(n, -1)); }
    for (auto& row : sc.f) { row.resize(n); std::fill(row.begin(), row.end(), NaN); }
    for (auto& row : sc.ptr) { row.resize(n); std::fill(row.begin(), row.end(), -1); }
    auto& f = sc.f;
    auto& ptr = sc.ptr;

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

    double best_bic, best_rss;
    int best_m = 0;
    {
        double rss0 = RSS(0, n - 1);
        best_bic = bic_score(rss0, n, k, 0);
        best_rss = rss0;
    }
    std::vector<int> best_breaks;

    for (int m = 1; m <= max_m_reached; ++m) {
        if (std::isnan(f[m][n - 1])) continue;
        double bic_m = bic_score(f[m][n - 1], n, k, m);
        if (bic_m < best_bic) {
            best_bic = bic_m;
            best_m = m;
            best_rss = f[m][n - 1];

            std::vector<int> breaks(m);
            int cur = n - 1;
            for (int level = m; level >= 1; --level) {
                cur = ptr[level][cur];
                breaks[level - 1] = cur;
            }
            best_breaks = breaks;
        }
    }

    if (best_m == 0) {
        Eigen::VectorXd beta;
        out.rss = fit_range(0, n - 1, beta);
        out.seg_coeffs.push_back(beta);
        return out;
    }

    out.n_breaks = best_m;
    out.breakpoints = best_breaks;
    out.rss = best_rss;

    int seg_start = 0;
    double total_rss = 0.0;
    for (int s = 0; s <= best_m; ++s) {
        int seg_end = (s < best_m) ? best_breaks[s] : n - 1;
        Eigen::VectorXd beta;
        total_rss += fit_range(seg_start, seg_end, beta);
        out.seg_coeffs.push_back(beta);
        seg_start = seg_end + 1;
    }
    out.rss = total_rss; // refit RSS (should match `best_rss` up to floating-point noise)

    return out;
}

// Fills leading/trailing NaN runs by nearest-value extrapolation and
// internal NaN runs by linear interpolation - used only to bootstrap the
// one-time STL seasonal estimate on a complete, regularly-spaced grid (see
// bfast.h's Scope note). Precondition: at least one non-NaN value.
void linear_interpolate_nan(std::vector<double>& v) {
    int n = (int)v.size();
    int first = -1;
    for (int i = 0; i < n; ++i) if (!std::isnan(v[i])) { first = i; break; }
    if (first < 0) return;
    for (int i = 0; i < first; ++i) v[i] = v[first];

    int last = -1;
    for (int i = n - 1; i >= 0; --i) if (!std::isnan(v[i])) { last = i; break; }
    for (int i = last + 1; i < n; ++i) v[i] = v[last];

    int prev = first;
    int i = first + 1;
    while (i <= last) {
        if (!std::isnan(v[i])) { prev = i; ++i; continue; }
        int next = i;
        while (next <= last && std::isnan(v[next])) ++next;
        double v0 = v[prev], v1 = v[next];
        for (int j = prev + 1; j < next; ++j) {
            v[j] = v0 + (v1 - v0) * (double)(j - prev) / (double)(next - prev);
        }
        prev = next;
        i = next + 1;
    }
}

BFResult bfast_impl(
    const double* y_raw, int n_raw,
    double start_time, int frequency, int order,
    double h, int max_breaks_trend, int max_breaks_season, int max_iter, double level, int min_valid,
    const Eigen::MatrixXd& Xtrend_full, const Eigen::MatrixXd& Xharm_full,
    FitScratch& trend_scratch, FitScratch& season_scratch)
{
    BFResult res;
    if (frequency < 2 || n_raw <= 2 * frequency) return res;

    std::vector<int> valid_rows;
    valid_rows.reserve(n_raw);
    for (int i = 0; i < n_raw; ++i) if (!std::isnan(y_raw[i])) valid_rows.push_back(i);
    int n = (int)valid_rows.size();
    res.n_valid = (double)n;
    if (n < min_valid || n <= 2 * frequency) return res;
    res.valid = 1.0;

    // One-time STL "periodic" seasonal seed, on the interpolated full grid
    // (see bfast.h's Scope note on why NaN gaps are interpolated here only).
    std::vector<double> y_interp(y_raw, y_raw + n_raw);
    linear_interpolate_nan(y_interp);
    std::vector<double> St_full = zeit::stl::periodic_seasonal(y_interp, frequency);

    Eigen::MatrixXd Xt(n, 2), Xh(n, Xharm_full.cols());
    Eigen::VectorXd Y(n);
    std::vector<double> St(n);
    for (int r = 0; r < n; ++r) {
        int i = valid_rows[r];
        Xt.row(r) = Xtrend_full.row(i);
        Xh.row(r) = Xharm_full.row(i);
        Y(r) = y_raw[i];
        St[r] = St_full[i];
    }

    std::vector<int> prev_trend_bp, prev_season_bp; // empty == "no breaks", matching R's Vt.bp/Wt.bp <- 0 sentinel
    SegFit trend_fit, season_fit;
    int n_iter_run = 0;
    Eigen::VectorXd Vt(n), Wt(n); // reused across iterations - only the contents change

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int r = 0; r < n; ++r) Vt(r) = Y(r) - St[r];
        bool trend_stable = ols_mosum_sctest_pvalue(Xt, Vt, h) > level;
        trend_fit = fit_segmented(Xt, Vt, h, max_breaks_trend, trend_scratch, /*skip_search=*/trend_stable);

        for (int r = 0; r < n; ++r) Wt(r) = Y(r) - trend_fit.fitted[r];
        bool season_stable = ols_mosum_sctest_pvalue(Xh, Wt, h) > level;
        season_fit = fit_segmented(Xh, Wt, h, max_breaks_season, season_scratch, /*skip_search=*/season_stable);
        for (int r = 0; r < n; ++r) St[r] = season_fit.fitted[r];

        n_iter_run = iter + 1;
        bool converged = (trend_fit.breakpoints == prev_trend_bp) && (season_fit.breakpoints == prev_season_bp);
        prev_trend_bp = trend_fit.breakpoints;
        prev_season_bp = season_fit.breakpoints;
        if (converged) break;
    }

    res.n_trend_breaks = (double)trend_fit.n_breaks;
    res.n_season_breaks = (double)season_fit.n_breaks;
    res.n_iter = (double)n_iter_run;

    res.trend_breakpoint_idx.resize(trend_fit.n_breaks);
    for (int i = 0; i < trend_fit.n_breaks; ++i) res.trend_breakpoint_idx[i] = (double)valid_rows[trend_fit.breakpoints[i]];
    res.season_breakpoint_idx.resize(season_fit.n_breaks);
    for (int i = 0; i < season_fit.n_breaks; ++i) res.season_breakpoint_idx[i] = (double)valid_rows[season_fit.breakpoints[i]];

    // Magnitude/time of the largest trend-component jump, matching R
    // bfast()'s `Mag`/`Magnitude`/`Time` (R/bfast.R): for each trend break,
    // compare the left segment's fitted value at its last point to the
    // right segment's fitted value at its first point.
    if (trend_fit.n_breaks > 0) {
        double best_jump = 0.0;
        int best_row = -1;
        res.trend_magnitude.resize(trend_fit.n_breaks);
        res.trend_first_after_idx.resize(trend_fit.n_breaks);
        for (int i = 0; i < trend_fit.n_breaks; ++i) {
            int end_left = trend_fit.breakpoints[i];
            int start_right = end_left + 1;
            const Eigen::VectorXd& bl = trend_fit.seg_coeffs[i];
            const Eigen::VectorXd& br = trend_fit.seg_coeffs[i + 1];
            double y1 = Xt.row(end_left).dot(bl);
            double y2 = Xt.row(start_right).dot(br);
            double jump = y2 - y1;
            res.trend_magnitude[i] = jump;
            res.trend_first_after_idx[i] = (double)valid_rows[start_right];
            if (i == 0 || std::fabs(jump) > std::fabs(best_jump)) {
                best_jump = jump;
                best_row = valid_rows[end_left];
            }
        }
        res.magnitude = best_jump;
        res.time = start_time + (double)best_row / (double)frequency;
    } else {
        res.magnitude = 0.0;
        res.time = std::nan("");
    }

    return res;
}

} // namespace

BFResult bfast(
    const std::vector<double>& y,
    double start_time,
    int frequency,
    int order,
    double h,
    int max_breaks_trend,
    int max_breaks_season,
    int max_iter,
    double level)
{
    int n_raw = (int)y.size();
    Eigen::MatrixXd Xtrend_full = build_trend_design_matrix(n_raw);
    Eigen::MatrixXd Xharm_full = build_harmonic_design_matrix(n_raw, start_time, frequency, order);
    FitScratch trend_scratch, season_scratch;
    return bfast_impl(y.data(), n_raw, start_time, frequency, order, h,
                       max_breaks_trend, max_breaks_season, max_iter, level, 20,
                       Xtrend_full, Xharm_full, trend_scratch, season_scratch);
}

pybind11::array_t<double> fit_bfast_batch(
    pybind11::array_t<double> values_array,
    double start_time,
    int frequency,
    int order,
    double h,
    int max_breaks_trend,
    int max_breaks_season,
    int max_iter,
    double level,
    int min_valid,
    int n_jobs)
{
    auto buf = values_array.request();
    if (buf.ndim != 2) throw std::runtime_error("values_array must be 2D [pixels, time]");

    int n_pixels = (int)buf.shape[0];
    int n_time = (int)buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);

    const int n_metrics = 7 + 3 * max_breaks_trend + max_breaks_season;
    pybind11::array_t<double> out_arr({n_metrics, n_pixels});
    double* out_ptr = static_cast<double*>(out_arr.request().ptr);
    for (int i = 0; i < n_metrics * n_pixels; ++i) out_ptr[i] = std::nan("");

    // Computed once for the whole batch (n_time/start_time/frequency/order
    // are shared by every pixel) instead of once per pixel.
    Eigen::MatrixXd Xtrend_full = build_trend_design_matrix(n_time);
    Eigen::MatrixXd Xharm_full = build_harmonic_design_matrix(n_time, start_time, frequency, order);

    if (n_jobs <= 0) n_jobs = std::max(1, omp_get_max_threads() - 1);

    #pragma omp parallel num_threads(n_jobs)
    {
        // One pair of scratch buffers per thread, reused across every pixel
        // it processes (see FitScratch's comment) - not per pixel, since
        // n_time/h/max_breaks are shared batch-wide, so buffer sizes are
        // typically stable across a thread's whole share of pixels.
        FitScratch trend_scratch, season_scratch;

        #pragma omp for
        for (int p = 0; p < n_pixels; ++p) {
            const double* y = ptr + (size_t)p * n_time;
            BFResult r = bfast_impl(y, n_time, start_time, frequency, order, h,
                                     max_breaks_trend, max_breaks_season, max_iter, level, min_valid,
                                     Xtrend_full, Xharm_full, trend_scratch, season_scratch);

            out_ptr[0 * n_pixels + p] = r.n_trend_breaks;
            out_ptr[1 * n_pixels + p] = r.n_season_breaks;
            out_ptr[2 * n_pixels + p] = r.magnitude;
            out_ptr[3 * n_pixels + p] = r.time;
            out_ptr[4 * n_pixels + p] = r.n_iter;
            out_ptr[5 * n_pixels + p] = r.n_valid;
            out_ptr[6 * n_pixels + p] = r.valid;
            for (int b = 0; b < (int)r.trend_breakpoint_idx.size() && b < max_breaks_trend; ++b) {
                out_ptr[(size_t)(7 + b) * n_pixels + p] = r.trend_breakpoint_idx[b];
                out_ptr[(size_t)(7 + max_breaks_trend + max_breaks_season + b) * n_pixels + p] = r.trend_magnitude[b];
                out_ptr[(size_t)(7 + 2 * max_breaks_trend + max_breaks_season + b) * n_pixels + p] =
                    r.trend_first_after_idx[b];
            }
            for (int b = 0; b < (int)r.season_breakpoint_idx.size() && b < max_breaks_season; ++b) {
                out_ptr[(size_t)(7 + max_breaks_trend + b) * n_pixels + p] = r.season_breakpoint_idx[b];
            }
        }
    }

    return out_arr;
}

} // namespace bfast
} // namespace zeit
