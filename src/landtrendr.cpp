#include "landtrendr.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <numeric>

#ifdef _OPENMP
#ifdef _OPENMP
#ifdef _OPENMP
#include <omp.h>
#endif
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_set_num_threads(x) (void)(x)
#endif
#endif

// -------------------------------------------------------------
// Math Functions for Statistical Significance (P-value / F-Stat)
// -------------------------------------------------------------
double gammln(double xx) {
    return std::lgamma(xx);
}

double betacf(double a, double b, double x) {
    int m, m2;
    double aa, c, d, del, h, qab, qam, qap;
    qab = a + b;
    qap = a + 1.0;
    qam = a - 1.0;
    c = 1.0;
    d = 1.0 - qab * x / qap;
    if (std::abs(d) < 1.0e-30) d = 1.0e-30;
    d = 1.0 / d;
    h = d;
    for (m = 1; m <= 1000; m++) {
        m2 = 2 * m;
        aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < 1.0e-30) d = 1.0e-30;
        c = 1.0 + aa / c;
        if (std::abs(c) < 1.0e-30) c = 1.0e-30;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < 1.0e-30) d = 1.0e-30;
        c = 1.0 + aa / c;
        if (std::abs(c) < 1.0e-30) c = 1.0e-30;
        d = 1.0 / d;
        del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < 1.0e-15) break;
    }
    return h;
}

double betai(double a, double b, double x) {
    double bt;
    if (x == 0.0 || x == 1.0) bt = 0.0;
    else bt = std::exp(gammln(a + b) - gammln(a) - gammln(b) + a * std::log(x) + b * std::log(1.0 - x));
    
    if (x < (a + 1.0) / (a + b + 2.0)) return bt * betacf(a, b, x) / a;
    else return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

double f_pval(double f_stat, double df1, double df2) {
    if (f_stat < 0.0) return 1.0;
    if (df1 <= 0 || df2 <= 0) return 1.0;
    double x = df2 / (df2 + df1 * f_stat);
    return betai(df2 / 2.0, df1 / 2.0, x);
}

// -------------------------------------------------------------
// Core Algorithm Logic
// -------------------------------------------------------------

namespace zeit {
namespace landtrendr {

// Single-index version of find_correction.pro's per-point formula -- shared
// by desawtooth()'s initial full pass and its incremental per-step update
// (only a fixed-width window around whichever index was just corrected can
// change, so recomputing just those beats rebuilding the whole array).
inline void correction_at(const std::vector<double>& v, int i,
                           double& correction, double& prop_correction) {
    double diff_2 = std::abs(v[i - 1] - v[i + 1]);
    double diff_minus1 = std::abs(v[i] - v[i + 1]);
    double diff_plus1 = std::abs(v[i] - v[i - 1]);

    double md = std::max(diff_minus1, diff_plus1);
    if (md == 0.0) {
        md = diff_2; // avoid division by zero -- diff_2 is then 0 too, so prop_correction is 0.
    }

    prop_correction = (md > 0.0) ? (1.0 - diff_2 / md) : 0.0;
    correction = prop_correction * (((v[i - 1] + v[i + 1]) / 2.0) - v[i]);
}

std::vector<double> desawtooth(const std::vector<double>& vals, double stopat) {
    std::vector<double> v = vals;
    int n = static_cast<int>(v.size());
    if (n < 3) return v;

    std::vector<double> correction(n, 0.0), prop_correction(n, 0.0);
    for (int i = 1; i < n - 1; ++i) {
        correction_at(v, i, correction[i], prop_correction[i]);
    }

    // desawtooth.pro's while loop checks `prop` *before* each pass using the
    // value left over from the previous pass (seeded at 1.0), then always
    // applies that pass's strongest correction regardless of its own
    // magnitude -- only the *next* pass's entry is gated by it. So the very
    // first correction always happens, however small, and only further
    // corrections are conditional on the threshold. `wh_max` also tracks the
    // true (possibly negative) argmax of prop_correction, not max(0, ...).
    double prop = 1.0;
    while (prop > stopat) {
        int wh_max = 0;
        double max_prop = prop_correction[0];
        for (int i = 1; i < n; ++i) {
            if (prop_correction[i] > max_prop) {
                max_prop = prop_correction[i];
                wh_max = i;
            }
        }

        v[wh_max] = v[wh_max] + correction[wh_max];
        prop = max_prop;

        // Only entries whose formula reads v[wh_max] can have changed: i and
        // i's immediate neighbors read v[i-1..i+1], so indices wh_max-2 ..
        // wh_max+2 are the full blast radius.
        int lo = std::max(1, wh_max - 2);
        int hi = std::min(n - 2, wh_max + 2);
        for (int i = lo; i <= hi; ++i) {
            correction_at(v, i, correction[i], prop_correction[i]);
        }
    }

    return v;
}

double angle_diff(double x0, double x1, double x2,
                  double y0, double y1, double y2,
                  double yrange, double distweightfactor) {
    double ydiff2 = y2 - y1;
    double ydiff1 = y1 - y0;

    double angle1 = std::atan(ydiff1 / (x1 - x0));
    double angle2 = std::atan(ydiff2 / (x2 - x1));

    double scaler = std::max(0.0, (ydiff2 * distweightfactor) / yrange) + 1.0;
    double diff = std::max(std::abs(angle1), std::abs(angle2)) * scaler;
    return diff;
}

std::vector<int> vet_verts(const std::vector<int>& x, const std::vector<double>& y, 
                           const std::vector<int>& vertices, int desired_count, 
                           double distweightfactor = 2.0) {
    int n_verts = vertices.size();
    int n_to_remove = n_verts - desired_count;

    if (n_to_remove <= 0 || n_verts <= 3) {
        return vertices;
    }

    // Min and max for y
    double min_y = *std::min_element(y.begin(), y.end());
    double max_y = *std::max_element(y.begin(), y.end());
    double yr = max_y - min_y;
    if (yr == 0.0) yr = 1.0; // avoid division by zero

    double range_x = x.back() - x.front();
    if (range_x == 0.0) range_x = 1.0;

    // Scale y
    std::vector<double> yscale(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        yscale[i] = ((y[i] - min_y) / yr) * range_x;
    }

    double sc_yr = *std::max_element(yscale.begin(), yscale.end()) - *std::min_element(yscale.begin(), yscale.end());
    if (sc_yr == 0.0) sc_yr = 1.0;

    std::vector<int> v = vertices;
    std::vector<double> slope_ratios(n_verts - 2);

    for (int i = 1; i < n_verts - 1; ++i) {
        slope_ratios[i - 1] = angle_diff(x[v[i - 1]], x[v[i]], x[v[i + 1]],
                                         yscale[v[i - 1]], yscale[v[i]], yscale[v[i + 1]],
                                         sc_yr, distweightfactor);
    }

    int count = n_verts;

    for (int step = 0; step < n_to_remove; ++step) {
        // Find minimum slope ratio
        double min_val = std::numeric_limits<double>::max();
        int worst = -1;
        for (int i = 0; i < count - 2; ++i) {
            if (slope_ratios[i] < min_val) {
                min_val = slope_ratios[i];
                worst = i;
            }
        }

        if (worst == -1) break; // Should not happen

        int worst_idx = worst + 1; // Index in vertex array

        // Remove the vertex
        v.erase(v.begin() + worst_idx);
        
        // Remove the corresponding slope ratio
        slope_ratios.erase(slope_ratios.begin() + worst);

        count--;

        // Recalculate neighbors
        if (worst_idx != 1) { // has left neighbor to recalculate
            int left_idx = worst_idx - 1;
            slope_ratios[left_idx - 1] = angle_diff(x[v[left_idx - 1]], x[v[left_idx]], x[v[left_idx + 1]],
                                                    yscale[v[left_idx - 1]], yscale[v[left_idx]], yscale[v[left_idx + 1]],
                                                    sc_yr, distweightfactor);
        }

        if (worst_idx != count - 1) { // has right neighbor to recalculate (which shifted to worst_idx)
            int right_idx = worst_idx;
            slope_ratios[right_idx - 1] = angle_diff(x[v[right_idx - 1]], x[v[right_idx]], x[v[right_idx + 1]],
                                                     yscale[v[right_idx - 1]], yscale[v[right_idx]], yscale[v[right_idx + 1]],
                                                     sc_yr, distweightfactor);
        }
    }

    return v;
}

// OLS fit of y over x for the contiguous index range [lo, hi]; returns the
// per-point fitted values and the fit's SSE. Shared by find_vertices() below
// for both scoring a segment (SSE/span) and finding its best split point --
// tbcd_v2.pro's score_segments and the regression inside split_series.
struct SegRegression { std::vector<double> fitted; double sse; };

SegRegression regress_range(const std::vector<int>& x, const std::vector<double>& y, int lo, int hi) {
    int m = hi - lo + 1;
    double x0 = static_cast<double>(x[lo]);
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = lo; i <= hi; ++i) {
        double dx = x[i] - x0;
        sum_x += dx; sum_y += y[i]; sum_xx += dx * dx; sum_xy += dx * y[i];
    }
    double denom = m * sum_xx - sum_x * sum_x;
    double slope = 0.0, intercept = sum_y / m;
    if (std::abs(denom) > 1e-9) {
        slope = (m * sum_xy - sum_x * sum_y) / denom;
        intercept = (sum_y - slope * sum_x) / m;
    }
    SegRegression out;
    out.fitted.resize(m);
    out.sse = 0.0;
    for (int i = lo; i <= hi; ++i) {
        double f = intercept + slope * (x[i] - x0);
        out.fitted[i - lo] = f;
        double err = y[i] - f;
        out.sse += err * err;
    }
    return out;
}

// Finds the best new vertex within segment [lo, hi]: regress the segment and take
// the point with the largest absolute residual, excluding the segment's own
// endpoints (tbcd_v2.pro's split_series). If this is the rightmost segment in the
// current vertex set, the second-to-last point's candidacy is suppressed unless
// the series is still rising there, right at the trailing edge -- this blocks a
// spurious one-year "recovery" vertex from being manufactured at the very end of
// the record. Returns -1 if nothing valid is left to split on.
int split_series(const std::vector<int>& x, const std::vector<double>& y, int lo, int hi,
                  bool is_end_segment, bool disttest) {
    SegRegression reg = regress_range(x, y, lo, hi);
    int m = hi - lo + 1;
    std::vector<double> diff(m);
    for (int i = 0; i < m; ++i) diff[i] = std::abs(y[lo + i] - reg.fitted[i]);
    diff[0] = 0.0;
    diff[m - 1] = 0.0;
    if (disttest && is_end_segment && m >= 3 && !(y[hi] > y[hi - 1])) {
        diff[m - 2] = 0.0;
    }

    int best_idx = 0;
    double best_val = diff[0];
    for (int i = 1; i < m; ++i) {
        if (diff[i] > best_val) { best_val = diff[i]; best_idx = i; }
    }
    if (best_idx == 0) return -1;
    return lo + best_idx;
}

// Regression-based recursive vertex identification (Kennedy et al. 2010, Section
// 2.5.2's first strategy, complementary to vet_verts()'s angle-based culling
// below) -- a faithful port of tbcd_v2.pro's find_vertices. Starts with just the
// first/last observation as vertices, then repeatedly splits whichever current
// segment has the largest SSE/span (not raw SSE -- a long segment with moderate
// error can outrank a short one with more), retrying the next-worst segment if
// split_series rejects the split. Stops once `target_count` vertices are reached
// (LT-GEE's max_segments + 1 + vertexCountOvershoot), no segment has a valid
// split left, or (matching the original's own runaway guard) 20 vertices have
// been added.
std::vector<int> find_vertices(const std::vector<int>& x, const std::vector<double>& y,
                                int target_count, double distweightfactor) {
    int n = static_cast<int>(x.size());
    int m = std::min(target_count, n - 2);
    std::vector<int> verts = {0, n - 1};
    bool disttest = (distweightfactor != 0.0);
    int count = 0;

    while (static_cast<int>(verts.size()) < m) {
        int nseg = static_cast<int>(verts.size()) - 1;
        std::vector<double> mses(nseg, 0.0);
        for (int s = 0; s < nseg; ++s) {
            int lo = verts[s], hi = verts[s + 1];
            double span = static_cast<double>(hi - lo + 1);
            if (span > 2.0) {
                mses[s] = regress_range(x, y, lo, hi).sse / span;
            }
        }

        int split_at = -1;
        while (true) {
            int s = static_cast<int>(std::max_element(mses.begin(), mses.end()) - mses.begin());
            if (mses[s] <= 0.0) { split_at = -1; break; }

            bool is_end_segment = (s == nseg - 1);
            int candidate = split_series(x, y, verts[s], verts[s + 1], is_end_segment, disttest);
            if (candidate != -1) { split_at = candidate; break; }
            mses[s] = 0.0; // rejected -- try the next-worst segment
        }
        if (split_at == -1) break;

        verts.push_back(split_at);
        std::sort(verts.begin(), verts.end());

        if (++count > 20) break;
    }

    return verts;
}

// Simple matrix inversion for small matrices using Gauss-Jordan. Flat row-major
// storage (A[i*n+j]) instead of vector<vector<double>>: one allocation instead
// of n+1, and contiguous memory instead of n separately-heap-allocated rows.
bool invert_matrix(std::vector<double>& A, int n) {
    std::vector<double> I(n * n, 0.0);
    for (int i = 0; i < n; ++i) I[i * n + i] = 1.0;

    for (int i = 0; i < n; ++i) {
        // Find pivot
        double max_el = std::abs(A[i * n + i]);
        int pivot = i;
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(A[k * n + i]) > max_el) {
                max_el = std::abs(A[k * n + i]);
                pivot = k;
            }
        }
        if (max_el == 0.0) return false; // Singular

        // Swap rows
        if (pivot != i) {
            for (int j = 0; j < n; ++j) {
                std::swap(A[i * n + j], A[pivot * n + j]);
                std::swap(I[i * n + j], I[pivot * n + j]);
            }
        }

        // Scale row
        double diag = A[i * n + i];
        for (int j = 0; j < n; ++j) {
            A[i * n + j] /= diag;
            I[i * n + j] /= diag;
        }

        // Eliminate column
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = A[k * n + i];
                for (int j = 0; j < n; ++j) {
                    A[k * n + j] -= factor * A[i * n + j];
                    I[k * n + j] -= factor * I[i * n + j];
                }
            }
        }
    }
    A = I;
    return true;
}

// Piecewise linear OLS fit with fixed breakpoints
std::vector<double> fit_piecewise_ols(const std::vector<int>& x, const std::vector<double>& y, const std::vector<int>& verts) {
    int n = x.size();
    int k = verts.size();

    // Build design matrix X_mat (n x k), flat row-major.
    std::vector<double> X_mat(n * k, 0.0);
    for (int i = 0; i < n; ++i) {
        int xi = x[i];
        for (int j = 0; j < k; ++j) {
            int vj = x[verts[j]];
            if (j > 0 && xi >= x[verts[j-1]] && xi <= vj) {
                int v_prev = x[verts[j-1]];
                if (vj > v_prev) {
                    X_mat[i * k + j] = static_cast<double>(xi - v_prev) / (vj - v_prev);
                }
            } else if (j < k - 1 && xi >= vj && xi <= x[verts[j+1]]) {
                int v_next = x[verts[j+1]];
                if (v_next > vj) {
                    X_mat[i * k + j] = static_cast<double>(v_next - xi) / (v_next - vj);
                }
            } else if (xi == vj) {
                X_mat[i * k + j] = 1.0;
            }
        }
    }

    // X^T * X, flat k x k
    std::vector<double> XtX(k * k, 0.0);
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            double s = 0.0;
            for (int r = 0; r < n; ++r) {
                s += X_mat[r * k + i] * X_mat[r * k + j];
            }
            XtX[i * k + j] = s;
        }
    }

    // Invert (X^T * X)
    if (!invert_matrix(XtX, k)) {
        // Fallback: just return the original Y values at vertices
        std::vector<double> fallback(k);
        for(int i=0; i<k; ++i) fallback[i] = y[verts[i]];
        return fallback;
    }

    // X^T * Y
    std::vector<double> XtY(k, 0.0);
    for (int i = 0; i < k; ++i) {
        double s = 0.0;
        for (int r = 0; r < n; ++r) {
            s += X_mat[r * k + i] * y[r];
        }
        XtY[i] = s;
    }

    // Beta = (X^T * X)^-1 * X^T * Y
    std::vector<double> beta(k, 0.0);
    for (int i = 0; i < k; ++i) {
        double s = 0.0;
        for (int j = 0; j < k; ++j) {
            s += XtX[i * k + j] * XtY[j];
        }
        beta[i] = s;
    }

    return beta;
}

// Fits vertex y-values early-to-late, choosing per segment between a
// point-to-point line (endpoints pinned to the actual data values) and a
// simple regression line fit over that segment's own observations -- LT-GEE's
// flexible per-segment fitting (Kennedy et al. 2010, Section 2.5.3). For
// segments after the first, the regression is anchored at the already-fixed
// start value so consecutive segments stay connected. This is tbcd_v2.pro's
// find_best_trace; if `slopes` is given it receives each segment's slope the
// way find_best_trace records it (the chosen line's own slope coefficient,
// not a re-derived vertex difference -- check_slopes compares it against a
// threshold, so the rounding path matters on exact ties).
std::vector<double> fit_piecewise_sequential(const std::vector<int>& x, const std::vector<double>& y,
                                              const std::vector<int>& verts,
                                              std::vector<double>* slopes = nullptr) {
    int k = static_cast<int>(verts.size());
    std::vector<double> fitted(k, 0.0);
    if (slopes) slopes->assign(std::max(0, k - 1), 0.0);
    if (k < 2) {
        if (k == 1) fitted[0] = y[verts[0]];
        return fitted;
    }

    for (int j = 0; j < k - 1; ++j) {
        int i0 = verts[j];
        int i1 = verts[j + 1];
        int x0 = x[i0];
        int x1 = x[i1];

        // Points in this segment are the contiguous index range [i0, i1] -- x is
        // sorted and verts are indices into it, so the range is iterated directly
        // (three times below) instead of first materializing an index vector.
        double span = static_cast<double>(x1 - x0);

        // Point-to-point candidate: the segment is just the line between the
        // (already-fixed, for j>0) start value and the next vertex's actual value.
        double p2p_y0 = (j == 0) ? y[verts[j]] : fitted[j];
        double p2p_y1 = y[verts[j + 1]];
        double p2p_slope = (span > 0.0) ? (p2p_y1 - p2p_y0) / span : 0.0;
        double p2p_sse = 0.0;
        for (int i = i0; i <= i1; ++i) {
            double err = y[i] - (p2p_y0 + p2p_slope * (x[i] - x0));
            p2p_sse += err * err;
        }

        // Regression candidate: free (2-parameter) OLS for the first segment,
        // anchored (1-parameter, pinned at the fixed start) for later ones.
        double reg_y0, reg_slope;
        if (j == 0) {
            double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
            int m = i1 - i0 + 1;
            for (int i = i0; i <= i1; ++i) {
                double dx = x[i] - x0;
                sum_x += dx; sum_y += y[i]; sum_xx += dx * dx; sum_xy += dx * y[i];
            }
            double denom = m * sum_xx - sum_x * sum_x;
            if (std::abs(denom) > 1e-9) {
                reg_slope = (m * sum_xy - sum_x * sum_y) / denom;
                reg_y0 = (sum_y - reg_slope * sum_x) / m;
            } else {
                reg_slope = p2p_slope;
                reg_y0 = p2p_y0;
            }
        } else {
            reg_y0 = fitted[j];
            double num = 0.0, den = 0.0;
            for (int i = i0; i <= i1; ++i) {
                double dx = x[i] - x0;
                num += dx * (y[i] - reg_y0);
                den += dx * dx;
            }
            reg_slope = (den > 1e-9) ? num / den : p2p_slope;
        }
        double reg_y1 = reg_y0 + reg_slope * span;
        double reg_sse = 0.0;
        for (int i = i0; i <= i1; ++i) {
            double err = y[i] - (reg_y0 + reg_slope * (x[i] - x0));
            reg_sse += err * err;
        }

        if (reg_sse < p2p_sse) {
            fitted[j] = reg_y0;
            fitted[j + 1] = reg_y1;
            if (slopes) (*slopes)[j] = reg_slope;
        } else {
            fitted[j] = p2p_y0;
            fitted[j + 1] = p2p_y0 + p2p_slope * span;
            if (slopes) (*slopes)[j] = p2p_slope;
        }
    }
    return fitted;
}

// Removes whichever single interior vertex tbcd_v2.pro's take_out_weakest (and
// the "run_mse" branch of take_out_weakest2) would remove. For each candidate,
// draws a straight line directly between its two flanking (already-fitted)
// vertex values -- skipping the candidate -- and scores it by that line's SSE
// against the actual observations in that local window, divided by the window's
// x-span. The first minimum wins, matching IDL's where(mse eq min(mse))[0].
std::vector<int> take_out_weakest(const std::vector<int>& x, const std::vector<double>& y,
                                   const std::vector<int>& verts, const std::vector<double>& vertvals) {
    int k = static_cast<int>(verts.size());
    if (k <= 2) return verts;

    double best_mse = std::numeric_limits<double>::max();
    int best_remove = -1;
    for (int i = 1; i < k - 1; ++i) {
        int i0 = verts[i - 1];
        int i1 = verts[i + 1];
        double span = static_cast<double>(x[i1] - x[i0]);
        if (span <= 0.0) continue;
        double slope = (vertvals[i + 1] - vertvals[i - 1]) / span;
        double mse = 0.0;
        for (int j = i0; j <= i1; ++j) {
            double fitted_y = vertvals[i - 1] + slope * static_cast<double>(x[j] - x[i0]);
            double err = y[j] - fitted_y;
            mse += err * err;
        }
        mse /= span;
        if (mse < best_mse) {
            best_mse = mse;
            best_remove = i;
        }
    }

    std::vector<int> result = verts;
    if (best_remove != -1) result.erase(result.begin() + best_remove);
    return result;
}

// One rung of tbcd_v2.pro's model ladder (one element of its `info` array).
// Everything is in modifier-space, indexed into the valid-observation arrays.
struct CandidateModel {
    std::vector<int> verts;
    std::vector<double> vertvals;
    std::vector<double> slopes;  // per segment, exactly as tbcd_v2.pro stores info.slope
    double f_stat;
    double pval;
};

// Piecewise-linear fitted value at every observation (fill_from_vertices.pro).
std::vector<double> interpolate_fit(const std::vector<int>& x, const std::vector<int>& verts,
                                    const std::vector<double>& vertvals) {
    std::vector<double> yfit(x.size(), 0.0);
    for (size_t j = 0; j + 1 < verts.size(); ++j) {
        int i0 = verts[j], i1 = verts[j + 1];
        double span = static_cast<double>(x[i1] - x[i0]);
        double slope = (span > 0.0) ? (vertvals[j + 1] - vertvals[j]) / span : 0.0;
        for (int i = i0; i <= i1; ++i) yfit[i] = vertvals[j] + slope * (x[i] - x[i0]);
    }
    return yfit;
}

// calc_fitting_stats3.pro. Every call site in tbcd_v2.pro passes
// n_predictors = 2*(vertex count) - 2 (each segment counted as slope+intercept).
//
// The p-value deliberately reproduces the original's single-precision
// `p_of_f = 1 - f_test1(...)`: the F CDF is rounded to float32 and subtracted
// from 1 in float32, so every p below ~6e-8 collapses to exactly 0. That is not
// cosmetic -- pick_best_model6's threshold is (2 - bestmodelproportion) * min(p),
// so when several candidates tie at p == 0 the most-vertex one wins, whereas a
// full-precision p would keep them distinct and pick a simpler model. Verified
// against the original IDL run under GDL: without this, strong single
// disturbances are systematically under-segmented relative to the reference.
struct FitStats { double f_stat; double pval; };

FitStats calc_fitting_stats(const std::vector<double>& y, const std::vector<double>& yfit, int n_predictors) {
    int n = static_cast<int>(y.size());
    double mean_y = 0.0;
    for (double v : y) mean_y += v;
    mean_y /= n;
    double ss = 0.0, ss_resid = 0.0;
    for (int i = 0; i < n; ++i) {
        ss += (y[i] - mean_y) * (y[i] - mean_y);
        ss_resid += (y[i] - yfit[i]) * (y[i] - yfit[i]);
    }
    if (ss_resid > ss) ss_resid = ss;  // rounding guard when there is no trend

    int df_regr = n_predictors;
    int df_resid = n - n_predictors - 1;
    if (df_regr <= 0 || df_resid <= 0) return {0.0, 1.0};

    double ms_regr = (ss - ss_resid) / df_regr;
    double ms_resid = ss_resid / df_resid;
    // "because of glitch in f_test1, a zero mistakenly gets f score of 1"
    double f_stat = (ms_regr < 0.00001) ? 0.00001 : ms_regr / ms_resid;

    double p_upper = f_pval(f_stat, df_regr, df_resid);
    float cdf = static_cast<float>(1.0 - p_upper);
    double pval = static_cast<double>(1.0f - cdf);
    return {f_stat, pval};
}

double value_range(const std::vector<double>& v) {
    return *std::max_element(v.begin(), v.end()) - *std::min_element(v.begin(), v.end());
}

// check_slopes.pro: a model is rejected if any recovery-direction (negative,
// in modifier-space) segment is steeper than recovery_threshold, measured as a
// proportion of the model's own fitted range (the range of a piecewise-linear
// fit is the range of its vertex values).
bool check_slopes(const CandidateModel& m, double threshold) {
    double range_of_vals = value_range(m.vertvals);
    for (double s : m.slopes) {
        if (s < 0.0 && std::abs(s) / range_of_vals > threshold) return false;
    }
    return true;
}

// take_out_weakest2.pro, the vertex-removal step of the primary (F6) ladder.
// If the previous rung has a recovery segment faster than `threshold`, the
// vertex that ends that segment is treated as the culprit: an interior one is
// dropped outright, the final one is flattened to the prior observation. In
// both cases the OBSERVATION under that vertex is overwritten in `y` -- which
// the original passes by reference, so the edit persists for every later
// rung, for the F7 fallback and for the final flat-line mean. Otherwise it
// falls through to take_out_weakest's local-MSE choice.
std::vector<int> take_out_weakest2(const CandidateModel& info, double threshold,
                                    const std::vector<int>& x, std::vector<double>& y,
                                    const std::vector<int>& verts, std::vector<double> vertvals) {
    int k = static_cast<int>(verts.size());
    double range_of_vals = value_range(info.vertvals);

    int violator = -1;
    double worst = -1.0;
    for (size_t s = 0; s < info.slopes.size(); ++s) {
        double slope = info.slopes[s];
        if (slope < 0.0 && slope != -1.0) {  // `ne -1` is in the original too
            double scaled = std::abs(slope) / range_of_vals;
            if (scaled > worst) { worst = scaled; violator = static_cast<int>(s); }
        }
    }

    if (violator != -1 && worst > threshold) {
        int vi = violator + 1;
        int idx = verts[vi];
        if (vi == k - 1) {
            y[idx] = y[idx - 1];
            vertvals[k - 1] = y[idx];
        } else {
            double slope = (y[idx + 1] - y[idx - 1]) / static_cast<double>(x[idx + 1] - x[idx - 1]);
            y[idx] = (x[idx] - x[idx - 1]) * slope + y[idx - 1];
            std::vector<int> result = verts;
            result.erase(result.begin() + vi);
            return result;
        }
    }
    return take_out_weakest(x, y, verts, vertvals);
}

// tbcd_v2.pro's selection loop around pick_best_model6 + check_slopes.
// pick_best_model6 (use_fstat=0): threshold = (2 - bestmodelproportion) *
// min(p) over the current working p-values; the first (= most-vertex) rung
// within it wins, or none when bestmodelproportion > 1. A pick that fails
// check_slopes has its p poisoned to 1 and selection retries, at most
// n_vertices_orig times. With no pick, the primary (F6) ladder falls back to
// the rung with the lowest ORIGINAL f_stat; the F7 ladder falls back to its
// single-segment rung, marked non-significant (p = 1) so it ends up flat.
int select_model(std::vector<CandidateModel>& ladder, double recovery_threshold,
                 double best_model_proportion, int n_vertices_orig, bool is_f7) {
    std::vector<double> fstats(ladder.size());
    for (size_t i = 0; i < ladder.size(); ++i) fstats[i] = ladder[i].f_stat;

    int best = 0;
    int increment = 0;
    bool notdone = true;
    while (notdone) {
        ++increment;
        double mn = ladder[0].pval;
        for (const auto& m : ladder) mn = std::min(mn, m.pval);
        double thr = (2.0 - best_model_proportion) * mn;
        int picked = -1;
        for (size_t i = 0; i < ladder.size(); ++i) {
            if (ladder[i].pval <= thr) { picked = static_cast<int>(i); break; }
        }

        if (picked != -1) {
            best = picked;
            bool ok = check_slopes(ladder[best], recovery_threshold);
            if (!ok) ladder[best].pval = 1.0;
            notdone = !ok && !(increment > n_vertices_orig);
        } else {
            if (is_f7) {
                best = static_cast<int>(ladder.size()) - 1;
                ladder[best].pval = 1.0;
            } else {
                best = static_cast<int>(std::min_element(fstats.begin(), fstats.end()) - fstats.begin());
            }
            notdone = false;
        }
    }
    return best;
}

// Adds a flat vertex at the start (or end) of the full year range when the
// first (last) year had no valid observation -- tbcd_v2.pro's "front end" /
// "other end" blocks. If that pushes the model past max_count vertices, the
// interior vertex with the least bend (angle_diff over all-year indices and
// vertex values, scaled by their range) is dropped again.
//
// The original runs this on info.vertvals, an INTEGER array, so the angles
// see vertex values truncated toward zero. In the "other end" block the
// working array is a concatenation of those integers, which additionally
// turns angle_diff's disturbance weight (ydiff2 * 2) / range into integer
// division. Both quirks decide which vertex gets dropped, so both are kept.
void extend_to_edge(std::vector<int>& idx, std::vector<double>& vals, bool front,
                    int n_all, int max_count) {
    if (front) {
        idx.insert(idx.begin(), 0);
        vals.insert(vals.begin(), vals.front());
    } else {
        idx.push_back(n_all - 1);
        vals.push_back(vals.back());
    }
    int nv = static_cast<int>(idx.size());
    if (nv - 1 <= max_count - 1) return;

    std::vector<long long> tv(nv);
    for (int i = 0; i < nv; ++i) tv[i] = static_cast<long long>(std::trunc(vals[i]));
    long long sc_yr = *std::max_element(tv.begin(), tv.end()) - *std::min_element(tv.begin(), tv.end());
    if (sc_yr == 0) sc_yr = 1;

    int minv = -1;
    double min_ratio = std::numeric_limits<double>::max();
    for (int i = 1; i < nv - 1; ++i) {
        long long ydiff1 = tv[i] - tv[i - 1];
        long long ydiff2 = tv[i + 1] - tv[i];
        double angle1 = std::atan(static_cast<double>(ydiff1) / (idx[i] - idx[i - 1]));
        double angle2 = std::atan(static_cast<double>(ydiff2) / (idx[i + 1] - idx[i]));
        double weight = front ? std::max(0.0, (ydiff2 * 2.0) / static_cast<double>(sc_yr))
                              : static_cast<double>(std::max(0LL, (ydiff2 * 2) / sc_yr));
        double r = std::max(std::abs(angle1), std::abs(angle2)) * (weight + 1.0);
        if (r < min_ratio) { min_ratio = r; minv = i; }
    }
    idx.erase(idx.begin() + minv);
    vals.erase(vals.begin() + minv);
}

// A faithful port of fit_trajectory_v2.pro + tbcd_v2.pro (LandTrendr-2012,
// Kennedy et al. 2010), validated vertex-for-vertex against the original IDL
// source run under GDL. Deliberate API-level differences: NaN observations are
// the "not in goods" years; returned vertex values are multiplied back out of
// modifier-space and not truncated to integers (the original stores them in
// an intarr); and degenerate inputs (too few observations, or no possible
// split) return a sensible trajectory instead of the original's zeroed
// placeholder structure.
TrajectoryResult fit_trajectory_impl(const std::vector<int>& years,
                                      const std::vector<double>& values,
                                      const LandTrendrParams& params) {
    TrajectoryResult out;
    std::vector<Vertex>& vertices = out.vertices;
    int n_all = static_cast<int>(years.size());
    if (n_all == 0 || static_cast<int>(values.size()) != n_all) {
        return out;
    }

    // `goods`: the observations actually used for fitting.
    std::vector<int> goods;
    for (int i = 0; i < n_all; ++i) {
        if (!std::isnan(values[i])) goods.push_back(i);
    }
    int n = static_cast<int>(goods.size());
    if (n == 0) return out;

    // Too few observations to justify fitting/simplifying at all (LT-GEE's
    // minObservationsNeeded) -- pass the raw trajectory through unsegmented.
    // No fit was performed, so there's no meaningful RMSE (left at 0).
    if (n < std::max(2, params.min_observations_needed)) {
        for (int g : goods) vertices.push_back({years[g], values[g]});
        return out;
    }

    std::vector<int> x(n);
    std::vector<double> raw(n);
    for (int i = 0; i < n; ++i) { x[i] = years[goods[i]]; raw[i] = values[goods[i]]; }

    // 1. desawtooth, then flip into modifier-space ("this sets everything so
    // disturbance is always positive"). This desawtoothed series is what the
    // original feeds to EVERY later step -- vertex search, fitting, and the
    // F-test -- not just vertex search.
    std::vector<double> y = (params.spike_threshold < 1.0) ? desawtooth(raw, params.spike_threshold) : raw;
    for (auto& v : y) v *= params.modifier;

    // 2-3. Candidate vertices: regression-based splitting up to
    // max_count + vertexcountovershoot, then angle-based culling back to max_count.
    int max_count = params.max_segments + 1;
    int overshoot_count = max_count + std::max(0, params.vertex_count_overshoot);
    std::vector<int> orig_v = vet_verts(x, y, find_vertices(x, y, overshoot_count, 2.0), max_count, 2.0);
    int n_vertices = static_cast<int>(orig_v.size());

    // Primary (F6) model: find_best_trace -- per segment, early to late, the
    // better of point-to-point and (anchored) regression.
    auto make_f6 = [&](const std::vector<int>& v) {
        CandidateModel m;
        m.verts = v;
        m.vertvals = fit_piecewise_sequential(x, y, v, &m.slopes);
        FitStats st = calc_fitting_stats(y, interpolate_fit(x, v, m.vertvals), 2 * static_cast<int>(v.size()) - 2);
        m.f_stat = st.f_stat;
        m.pval = st.pval;
        return m;
    };

    // Fallback (F7) model: find_best_trace3 -- all vertex values fit jointly
    // (mpfitfun/Levenberg-Marquardt in the original; the model is linear in the
    // vertex values, so the exact OLS solve is the same optimum). Its slopes
    // divide by the vertex INDEX span, as find_best_trace3 does.
    auto make_f7 = [&](const std::vector<int>& v) {
        CandidateModel m;
        m.verts = v;
        m.vertvals = fit_piecewise_ols(x, y, v);
        m.slopes.resize(v.size() - 1);
        for (size_t s = 0; s + 1 < v.size(); ++s) {
            m.slopes[s] = (m.vertvals[s + 1] - m.vertvals[s]) / static_cast<double>(v[s + 1] - v[s]);
        }
        FitStats st = calc_fitting_stats(y, interpolate_fit(x, v, m.vertvals), 2 * static_cast<int>(v.size()) - 2);
        m.f_stat = st.f_stat;
        m.pval = st.pval;
        return m;
    };

    // 4. F6 ladder: every vertex count from n_vertices down to 2, each rung
    // simplified from the previous one by take_out_weakest2 (which may edit y).
    std::vector<CandidateModel> ladder;
    ladder.push_back(make_f6(orig_v));
    for (int i = 1; i <= n_vertices - 2; ++i) {
        const CandidateModel& prev = ladder.back();
        std::vector<int> v = take_out_weakest2(prev, params.recovery_threshold, x, y, prev.verts, prev.vertvals);
        ladder.push_back(make_f6(v));
    }
    int best = select_model(ladder, params.recovery_threshold, params.best_model_proportion, n_vertices, false);

    // 5. If the chosen F6 model isn't significant, rebuild the whole ladder
    // from the original vertices with joint fitting and plain take_out_weakest.
    if (ladder[best].pval > params.pval_threshold) {
        ladder.clear();
        ladder.push_back(make_f7(orig_v));
        for (int i = 1; i <= n_vertices - 2; ++i) {
            const CandidateModel& prev = ladder.back();
            std::vector<int> v = take_out_weakest(x, y, prev.verts, prev.vertvals);
            ladder.push_back(make_f7(v));
        }
        best = select_model(ladder, params.recovery_threshold, params.best_model_proportion, n_vertices, true);
    }
    const CandidateModel& chosen = ladder[best];

    // 6. Output in all-year terms. A model that is still not significant
    // becomes a flat line at the mean of the (possibly edited) working series
    // across the whole year range; otherwise map vertices back to all-year
    // indices and pad missing first/last years with flat vertices.
    std::vector<int> out_idx;
    std::vector<double> out_vals;
    if (chosen.pval > params.pval_threshold) {
        double mean_y = 0.0;
        for (double v : y) mean_y += v;
        mean_y /= n;
        out_idx = {0, n_all - 1};
        out_vals = {mean_y, mean_y};
    } else {
        for (int v : chosen.verts) out_idx.push_back(goods[v]);
        out_vals = chosen.vertvals;
        if (goods.front() != 0) extend_to_edge(out_idx, out_vals, true, n_all, max_count);
        if (goods.back() != n_all - 1) extend_to_edge(out_idx, out_vals, false, n_all, max_count);
    }

    // Vertex values leave modifier-space (modifier is always +-1, so
    // multiplying again undoes it) so callers see original-scale values.
    for (size_t i = 0; i < out_idx.size(); ++i) {
        vertices.push_back({years[out_idx[i]], out_vals[i] * params.modifier});
    }

    // RMSE of the output trajectory against every valid raw observation --
    // LT-GEE's per-pixel noise estimate for DSNR (see TrajectoryResult).
    std::vector<double> all_fit = interpolate_fit(years, out_idx, out_vals);
    double sse = 0.0;
    for (int i = 0; i < n; ++i) {
        double err = raw[i] * params.modifier - all_fit[goods[i]];
        sse += err * err;
    }
    int n_params = static_cast<int>(out_idx.size());
    out.rmse = std::sqrt(sse / std::max(1, n - n_params));

    return out;
}

// Convenience wrapper for callers that only need the vertices.
std::vector<Vertex> fit_trajectory(const std::vector<int>& years,
                                   const std::vector<double>& values,
                                   const LandTrendrParams& params) {
    return fit_trajectory_impl(years, values, params).vertices;
}

// ftv_v1.pro's flat vertex at the start (or end) of the year range when the
// band has no observation there; past max_count vertices, the interior vertex
// with the least bend (angle_diff over all-year indices and vertex values,
// distweightfactor 2) is dropped again. Unlike tbcd_v2.pro's extend_to_edge
// the vertex values are floats here (ftv_v1 builds them with fltarr).
void extend_ftv_to_edge(std::vector<int>& idx, std::vector<double>& vals, bool front, int n_all, int max_count) {
    if (front) {
        idx.insert(idx.begin(), 0);
        vals.insert(vals.begin(), vals.front());
    } else {
        idx.push_back(n_all - 1);
        vals.push_back(vals.back());
    }
    int nv = static_cast<int>(idx.size());
    if (nv - 1 <= max_count - 1) return;
    double sc_yr = value_range(vals);
    if (sc_yr == 0.0) sc_yr = 1.0;
    int minv = -1;
    double min_ratio = std::numeric_limits<double>::max();
    for (int i = 1; i < nv - 1; ++i) {
        double r = angle_diff(idx[i - 1], idx[i], idx[i + 1], vals[i - 1], vals[i], vals[i + 1], sc_yr, 2.0);
        if (r < min_ratio) { min_ratio = r; minv = i; }
    }
    if (minv < 0) return;
    idx.erase(idx.begin() + minv);
    vals.erase(vals.begin() + minv);
}

// Fitted-to-vertices (LT-GEE's ftv bands): apply_fitted_trajectory_v1.pro +
// ftv_v1.pro. The series of another band is desawtoothed (not flipped by the
// modifier, which the original leaves commented out) and fitted with
// find_best_trace on the vertex years found by the segmentation band. A
// vertex year the band has no observation for moves to the band's last
// observation before it (or the first after, when that one is taken already
// or there is none before). Missing first/last years get flat vertices, as in
// the segmentation. Returns the fitted value of every year; all NaN without
// observations or vertices.
std::vector<double> fit_to_vertices(const std::vector<int>& years, const std::vector<double>& values,
                                    const std::vector<int>& vertex_years, double spike_threshold) {
    const int n_all = static_cast<int>(years.size());
    std::vector<double> yfit(n_all, std::numeric_limits<double>::quiet_NaN());
    std::vector<int> goods;
    for (int i = 0; i < n_all; ++i) {
        if (!std::isnan(values[i])) goods.push_back(i);
    }
    const int n = static_cast<int>(goods.size());
    const int k = static_cast<int>(vertex_years.size());
    if (n == 0 || k == 0) return yfit;

    std::vector<int> x(n);
    std::vector<double> raw(n);
    for (int i = 0; i < n; ++i) { x[i] = years[goods[i]]; raw[i] = values[goods[i]]; }
    std::vector<double> y = (spike_threshold < 1.0) ? desawtooth(raw, spike_threshold) : raw;

    // The vertices on the band's own observations (indices into x).
    std::vector<int> v;
    for (int i = 0; i < k; ++i) {
        const int vy = vertex_years[i];
        int at = -1, before = -1, after = -1;
        for (int j = 0; j < n; ++j) {
            if (x[j] == vy) at = j;
            if (x[j] < vy) before = j;
            if (x[j] > vy && after == -1) after = j;
        }
        if (at == -1) {
            at = before;
            if (before == -1) at = after;
            else if (!v.empty() && at == v.back() && at != n - 1) at = after;
        }
        if (at != -1) v.push_back(at);  // (the original stops on a vertex it cannot place)
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());

    std::vector<int> out_idx;
    std::vector<double> out_vals;
    if (v.size() < 2) {  // no segment: the mean, flat over every year
        double mean_y = 0.0;
        for (double val : y) mean_y += val;
        mean_y /= n;
        out_idx = {0, n_all - 1};
        out_vals = {mean_y, mean_y};
    } else {
        out_vals = fit_piecewise_sequential(x, y, v);
        for (int vi : v) out_idx.push_back(goods[vi]);
        if (goods.front() != 0) extend_ftv_to_edge(out_idx, out_vals, true, n_all, k);
        if (goods.back() != n_all - 1) extend_ftv_to_edge(out_idx, out_vals, false, n_all, k);
    }
    if (out_idx.front() == out_idx.back()) {  // a single year
        std::fill(yfit.begin(), yfit.end(), out_vals.front());
        return yfit;
    }
    std::vector<double> fit = interpolate_fit(years, out_idx, out_vals);
    for (int i = 0; i < n_all; ++i) {
        if (i < out_idx.front() || i > out_idx.back()) fit[i] = std::numeric_limits<double>::quiet_NaN();
    }
    return fit;
}

pybind11::array_t<double> fit_to_vertices_batch(pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> values_array,
                                                pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> years_array,
                                                pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> vertex_years,
                                                pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> counts,
                                                double spike_threshold, int n_jobs) {
    if (values_array.ndim() != 2 || vertex_years.ndim() != 2 || years_array.ndim() != 1 || counts.ndim() != 1) {
        throw std::invalid_argument("values (pixels, time), years (time,), vertex_years (pixels, vertices), counts (pixels,)");
    }
    const auto num_pixels = static_cast<int>(values_array.shape(0));
    const auto times = static_cast<int>(values_array.shape(1));
    const auto max_vertices = static_cast<int>(vertex_years.shape(1));
    if (years_array.shape(0) != times || vertex_years.shape(0) != num_pixels || counts.shape(0) != num_pixels) {
        throw std::invalid_argument("values, years, vertex_years and counts do not match");
    }
    const double* val_ptr = values_array.data();
    const int* vy_ptr = vertex_years.data();
    const int* count_ptr = counts.data();
    std::vector<int> years(years_array.data(), years_array.data() + times);
    pybind11::array_t<double> out({num_pixels, times});
    double* out_ptr = out.mutable_data();
    std::fill(out_ptr, out_ptr + static_cast<std::size_t>(num_pixels) * times, std::numeric_limits<double>::quiet_NaN());
    {
        pybind11::gil_scoped_release release;
        #ifdef _OPENMP
        int num_threads = n_jobs > 0 ? n_jobs : std::max(1, omp_get_num_procs() - 1);
        #pragma omp parallel num_threads(num_threads)
        #endif
        {
            std::vector<double> pixel(times);
            std::vector<int> verts;
            #ifdef _OPENMP
            #pragma omp for schedule(dynamic, 64)
            #endif
            for (int p = 0; p < num_pixels; ++p) {
                const int nv = std::min(count_ptr[p], max_vertices);
                if (nv <= 0) continue;
                verts.assign(vy_ptr + static_cast<std::size_t>(p) * max_vertices,
                             vy_ptr + static_cast<std::size_t>(p) * max_vertices + nv);
                std::copy(val_ptr + static_cast<std::size_t>(p) * times, val_ptr + static_cast<std::size_t>(p + 1) * times,
                          pixel.begin());
                std::vector<double> fit = fit_to_vertices(years, pixel, verts, spike_threshold);
                std::copy(fit.begin(), fit.end(), out_ptr + static_cast<std::size_t>(p) * times);
            }
        }
    }
    return out;
}

pybind11::tuple fit_trajectory_batch(
    pybind11::array_t<double> values_array, // Shape: [Y, X, Time]
    pybind11::array_t<int> years_array,     // Shape: [Time]
    LandTrendrParams params,
    double no_data_value,
    int n_jobs
) {
    auto val_buf = values_array.request();
    auto year_buf = years_array.request();
    
    int height = val_buf.shape[0];
    int width = val_buf.shape[1];
    int times = val_buf.shape[2];
    int num_pixels = height * width;
    
    double* val_ptr = static_cast<double*>(val_buf.ptr);
    int* year_ptr = static_cast<int*>(year_buf.ptr);
    
    std::vector<int> years(year_ptr, year_ptr + times);
    int max_vertices = params.max_segments + 1;
    
    // Output arrays
    pybind11::array_t<double> vertices_out({num_pixels, max_vertices, 2});
    auto vert_ptr = static_cast<double*>(vertices_out.request().ptr);
    
    pybind11::array_t<int> counts_out(num_pixels);
    auto counts_ptr = static_cast<int*>(counts_out.request().ptr);

    // Per-pixel RMSE of the selected fit -- LT-GEE's DSNR is magnitude / this.
    pybind11::array_t<double> rmse_out(num_pixels);
    auto rmse_ptr = static_cast<double*>(rmse_out.request().ptr);

    std::fill(vert_ptr, vert_ptr + (num_pixels * max_vertices * 2), no_data_value);
    std::fill(counts_ptr, counts_ptr + num_pixels, 0);
    std::fill(rmse_ptr, rmse_ptr + num_pixels, 0.0);

    #ifdef _OPENMP
    int num_threads = n_jobs > 0 ? n_jobs : std::max(1, omp_get_num_procs() - 1);
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    #endif
    {
        // One reusable buffer per thread instead of one heap allocation per
        // pixel (millions of pixels otherwise reallocate this every iteration).
        std::vector<double> pixel_values(times);

        #ifdef _OPENMP
        #pragma omp for schedule(dynamic)
        #endif
        for (int p = 0; p < num_pixels; ++p) {
            bool has_valid_data = false;

            for (int t = 0; t < times; ++t) {
                double v = val_ptr[p * times + t];
                // no-data years are simply absent from the fit (NaN = not in
                // the original's `goods`), never fitted as real values.
                if (v == no_data_value || std::isnan(v)) {
                    pixel_values[t] = std::numeric_limits<double>::quiet_NaN();
                } else {
                    pixel_values[t] = v;
                    has_valid_data = true;
                }
            }

            if (!has_valid_data) continue;

            TrajectoryResult result = fit_trajectory_impl(years, pixel_values, params);

            counts_ptr[p] = result.vertices.size();
            for (size_t i = 0; i < result.vertices.size() && (int)i < max_vertices; ++i) {
                vert_ptr[p * max_vertices * 2 + i * 2 + 0] = result.vertices[i].year;
                vert_ptr[p * max_vertices * 2 + i * 2 + 1] = result.vertices[i].value;
            }
            rmse_ptr[p] = result.rmse;
        }
    }

    return pybind11::make_tuple(vertices_out, counts_out, rmse_out);
}

} // namespace landtrendr
} // namespace zeit

