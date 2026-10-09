// CCDC -- a port of the original MATLAB implementation (Zhu & Woodcock 2014;
// GERSL/CCDC TrendSeasonalFit_v12_30Line.m with autoTSFit, autoTSPred,
// autoTmask, autoRobustFit, robustfit_cor, update_cft and the bundled Fortran
// GLMnet), validated against that code run under GNU Octave.
//
// Two numerical details of the original are reproduced on purpose because
// they change results, not just rounding noise:
//  * every model fit is a lasso (glmnet, lambda = 20) run by the bundled
//    Fortran GLMnet in SINGLE precision -- glmnetMex.F copies the double
//    inputs into `real` arrays, including the date column (~7e5, where float32
//    resolution is 1/16 day). glmnet_lasso() below is a float32 port of
//    elnetu/standard/elnet1 with the same operation order. Being derived from
//    GLMnet (GPL-2.0-only), this file is distributed under GPL version 2; see
//    THIRD_PARTY_NOTICES.md.
//  * the design matrix uses MATLAB datenum time (= Python ordinal + 366);
//    a lasso is not invariant to shifting the harmonic phase.
#include "ccdc.h"
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace zeit {
namespace ccdc {

namespace {

const double PI = 3.14159265358979323846;
const double W = 2.0 * PI / 365.25;       // annual cycle, as in the original
const int DATENUM_OFFSET = 366;          // MATLAB datenum = Python ordinal + 366

// Fixed constants of TrendSeasonalFit_v12_30Line.m
const int MIN_NUM_C = 4;
const int MID_NUM_C = 6;
const int MAX_NUM_C = 8;
const int N_TIMES = 3;
const double NUM_YRS = 365.25;
const double T_CONST = 3.89;
const double MINI_YRS = 1.0;
const double T_SN = 0.75;
const double T_CLR = 0.25;
const double CLOUD_SCREEN_OFFSET = 400.0;  // "median(green) + 400" in the insufficient-clear branch
const float GLMNET_LAMBDA = 20.0f;
const float GLMNET_THRESH = 1e-4f;

// ---------------------------------------------------------------------------
// Chi-square quantile (MATLAB chi2inv), accurate to ~1e-14.

double gamma_p(double a, double x) {
    if (x <= 0.0) return 0.0;
    double gln = std::lgamma(a);
    if (x < a + 1.0) {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 0; n < 10000; ++n) {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (std::abs(del) < std::abs(sum) * 1e-17) break;
        }
        return sum * std::exp(-x + a * std::log(x) - gln);
    }
    const double tiny = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int i = 1; i < 10000; ++i) {
        double an = -i * (i - a);
        b += 2.0;
        d = an * d + b; if (std::abs(d) < tiny) d = tiny;
        c = b + an / c; if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < 1e-17) break;
    }
    return 1.0 - std::exp(-x + a * std::log(x) - gln) * h;
}

double chi2inv(double p, int df) {
    double lo = 0.0, hi = 1.0;
    while (gamma_p(df / 2.0, hi / 2.0) < p) hi *= 2.0;
    for (int i = 0; i < 200 && hi - lo > 1e-14 * hi; ++i) {
        double mid = 0.5 * (lo + hi);
        if (gamma_p(df / 2.0, mid / 2.0) < p) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// ---------------------------------------------------------------------------
// MATLAB helpers

double median_of(std::vector<double> v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double norm2_sq(const std::vector<double>& v) {  // norm(v)^2, rounded like MATLAB's
    double s = 0.0;
    for (double x : v) s += x * x;
    double n = std::sqrt(s);
    return n * n;
}

// ---------------------------------------------------------------------------
// GLMnet (5/17/08) elnetu -> standard -> elnet1, gaussian lasso (alpha = 1),
// one user lambda, standardize = true, unit weights and penalty factors.
// Single precision throughout, exactly like the original's `real` Fortran.
// x: column-major no x ni. Returns a0 and beta (length ni).

void glmnet_lasso(std::vector<float> x, std::vector<float> y, int no, int ni,
                  float ulam, float thr, float& a0_out, std::vector<float>& beta_out) {
    beta_out.assign(ni, 0.0f);
    auto X = [&](int i, int j) -> float& { return x[static_cast<size_t>(j) * no + i]; };

    // chkvars
    std::vector<int> ju(ni, 0);
    for (int j = 0; j < ni; ++j) {
        float t = X(0, j);
        for (int i = 1; i < no; ++i) if (X(i, j) != t) { ju[j] = 1; break; }
    }
    bool any_ju = false;
    for (int j = 0; j < ni; ++j) any_ju = any_ju || ju[j];
    if (!any_ju) {  // jerr = 7777: the MATLAB wrapper then gets no fit at all
        float ym = 0.0f;
        for (int i = 0; i < no; ++i) ym = ym + (1.0f / no) * y[i];
        a0_out = ym;
        return;
    }

    // standard
    std::vector<float> w(no, 1.0f), v(no), xm(ni, 0.0f), xs(ni, 1.0f), xv(ni, 0.0f), g(ni, 0.0f);
    float sw = 0.0f;
    for (int i = 0; i < no; ++i) sw = sw + w[i];
    for (int i = 0; i < no; ++i) { w[i] = w[i] / sw; v[i] = std::sqrt(w[i]); }
    for (int j = 0; j < ni; ++j) {
        if (!ju[j]) continue;
        float s = 0.0f;
        for (int i = 0; i < no; ++i) s = s + w[i] * X(i, j);
        xm[j] = s;
        for (int i = 0; i < no; ++i) X(i, j) = v[i] * (X(i, j) - xm[j]);
        s = 0.0f;
        for (int i = 0; i < no; ++i) s = s + X(i, j) * X(i, j);
        xv[j] = s;
    }
    for (int j = 0; j < ni; ++j) {
        if (!ju[j]) continue;
        xs[j] = std::sqrt(xv[j]);
        for (int i = 0; i < no; ++i) X(i, j) = X(i, j) / xs[j];
    }
    for (int j = 0; j < ni; ++j) xv[j] = 1.0f;
    float ym = 0.0f;
    for (int i = 0; i < no; ++i) ym = ym + w[i] * y[i];
    for (int i = 0; i < no; ++i) y[i] = v[i] * (y[i] - ym);
    float ys = 0.0f;
    for (int i = 0; i < no; ++i) ys = ys + y[i] * y[i];
    ys = std::sqrt(ys);
    for (int i = 0; i < no; ++i) y[i] = y[i] / ys;
    for (int j = 0; j < ni; ++j) {
        if (!ju[j]) continue;
        float s = 0.0f;
        for (int i = 0; i < no; ++i) s = s + y[i] * X(i, j);
        g[j] = s;
    }
    float alm = ulam / ys;

    // elnet1 (flmin >= 1: the single given lambda; beta = alpha = 1)
    const int nx = ni;
    const float ab = alm * 1.0f, dem = alm * 0.0f;
    std::vector<float> a(ni, 0.0f), da(ni, 0.0f), c(static_cast<size_t>(ni) * nx, 0.0f);
    auto C = [&](int j, int l) -> float& { return c[static_cast<size_t>(l) * ni + j]; };
    std::vector<int> mm(ni, 0), ia(nx, 0);
    int nin = 0, iz = 0, jz = 1;
    float rsq = 0.0f;
    while (true) {
        if (!(iz * jz != 0)) {
            float dlx = 0.0f;
            bool overflow = false;
            for (int k = 0; k < ni; ++k) {
                if (!ju[k]) continue;
                float ak = a[k];
                float u = g[k] + ak * xv[k];
                float vv = std::abs(u) - 1.0f * ab;
                a[k] = 0.0f;
                if (vv > 0.0f) a[k] = std::copysign(vv, u) / (xv[k] + 1.0f * dem);
                if (a[k] == ak) continue;
                if (mm[k] == 0) {
                    nin = nin + 1;
                    if (nin > nx) { overflow = true; break; }
                    for (int j = 0; j < ni; ++j) {
                        if (!ju[j]) continue;
                        if (mm[j] != 0) { C(j, nin - 1) = C(k, mm[j] - 1); continue; }
                        if (j == k) { C(j, nin - 1) = xv[j]; continue; }
                        float s = 0.0f;
                        for (int i = 0; i < no; ++i) s = s + X(i, j) * X(i, k);
                        C(j, nin - 1) = s;
                    }
                    mm[k] = nin;
                    ia[nin - 1] = k;
                }
                float del = a[k] - ak;
                rsq = rsq + del * (2.0f * g[k] - del * xv[k]);
                dlx = std::max(std::abs(del) / std::sqrt(xv[k]), dlx);
                for (int j = 0; j < ni; ++j) if (ju[j]) g[j] = g[j] - C(j, mm[k] - 1) * del;
            }
            if (overflow || dlx < thr) break;
            if (nin > nx) break;
        }
        iz = 1;
        for (int l = 0; l < nin; ++l) da[l] = a[ia[l]];
        while (true) {
            float dlx = 0.0f;
            for (int l = 0; l < nin; ++l) {
                int k = ia[l];
                float ak = a[k];
                float u = g[k] + ak * xv[k];
                float vv = std::abs(u) - 1.0f * ab;
                a[k] = 0.0f;
                if (vv > 0.0f) a[k] = std::copysign(vv, u) / (xv[k] + 1.0f * dem);
                if (a[k] == ak) continue;
                float del = a[k] - ak;
                rsq = rsq + del * (2.0f * g[k] - del * xv[k]);
                dlx = std::max(std::abs(del) / std::sqrt(xv[k]), dlx);
                for (int j = 0; j < nin; ++j) g[ia[j]] = g[ia[j]] - C(ia[j], mm[k] - 1) * del;
            }
            if (dlx < thr) break;
        }
        for (int l = 0; l < nin; ++l) da[l] = a[ia[l]] - da[l];
        for (int j = 0; j < ni; ++j) {
            if (mm[j] != 0) continue;
            if (!ju[j]) continue;
            float s = 0.0f;
            for (int l = 0; l < nin; ++l) s = s + da[l] * C(j, l);
            g[j] = g[j] - s;
        }
        jz = 0;
    }
    int nk = std::min(nin, nx);

    // back in elnetu: unstandardize, intercept (dot_product in active order)
    std::vector<float> ca(nk);
    for (int l = 0; l < nk; ++l) ca[l] = ys * a[ia[l]] / xs[ia[l]];
    float dp = 0.0f;
    for (int l = 0; l < nk; ++l) dp = dp + ca[l] * xm[ia[l]];
    a0_out = ym - dp;
    for (int l = 0; l < nk; ++l) beta_out[ia[l]] = ca[l];
}

// ---------------------------------------------------------------------------
// autoTSPred / autoTSFit

double ts_pred(double t, const double* c) {
    double terms[8] = {1.0, t, std::cos(W * t), std::sin(W * t),
                       std::cos(2 * W * t), std::sin(2 * W * t),
                       std::cos(3 * W * t), std::sin(3 * W * t)};
    double s = 0.0;
    for (int k = 0; k < 8; ++k) s += terms[k] * c[k];
    return s;
}

struct TSFit {
    std::vector<double> cft;    // 8
    double rmse = 0.0;
    std::vector<double> v_dif;  // residuals
};

// Fits one band over rows `idx` of (x, y) with df coefficients.
TSFit auto_ts_fit(const std::vector<double>& x, const std::vector<double>& y, int df) {
    int n = static_cast<int>(x.size());
    int ni = df - 1;
    std::vector<float> X(static_cast<size_t>(n) * ni);
    std::vector<float> Y(n);
    for (int i = 0; i < n; ++i) {
        double t = x[i];
        double cols[7] = {t, std::cos(W * t), std::sin(W * t), std::cos(2 * W * t),
                          std::sin(2 * W * t), std::cos(3 * W * t), std::sin(3 * W * t)};
        for (int j = 0; j < ni; ++j) X[static_cast<size_t>(j) * n + i] = static_cast<float>(cols[j]);
        Y[i] = static_cast<float>(y[i]);
    }
    float a0;
    std::vector<float> beta;
    glmnet_lasso(X, Y, n, ni, GLMNET_LAMBDA, GLMNET_THRESH, a0, beta);

    TSFit out;
    out.cft.assign(8, 0.0);
    out.cft[0] = a0;
    for (int j = 0; j < ni; ++j) out.cft[j + 1] = beta[j];
    out.v_dif.resize(n);
    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
        out.v_dif[i] = y[i] - ts_pred(x[i], out.cft.data());
        ss += out.v_dif[i] * out.v_dif[i];
    }
    out.rmse = std::sqrt(ss) / std::sqrt(static_cast<double>(n - df));
    return out;
}

// ---------------------------------------------------------------------------
// robustfit_cor / statrobustfit_cor (bisquare, tune 4.685) and autoTmask

Eigen::VectorXd robust_fit(const Eigen::MatrixXd& Xin, const Eigen::VectorXd& y) {
    int n = static_cast<int>(Xin.rows());
    int p = static_cast<int>(Xin.cols()) + 1;
    Eigen::MatrixXd X(n, p);
    X.col(0).setOnes();
    X.rightCols(p - 1) = Xin;
    const double eps = std::numeric_limits<double>::epsilon();

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);
    qr.setThreshold(std::max(n, p) * eps);
    int xrank = static_cast<int>(qr.rank());
    Eigen::VectorXd b = qr.solve(y);

    // leverage from the full-rank part of the QR: E = X(:,perm) / R
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, xrank);
    Eigen::VectorXd h = Q.rowwise().squaredNorm();
    Eigen::VectorXd adjfactor(n);
    for (int i = 0; i < n; ++i) adjfactor(i) = 1.0 / std::sqrt(1.0 - std::min(0.9999, h(i)));

    double mean_y = y.mean();
    double sd_y = (n > 1) ? std::sqrt((y.array() - mean_y).square().sum() / (n - 1)) : 0.0;
    double tiny_s = 1e-6 * sd_y;
    if (tiny_s == 0.0) tiny_s = 1.0;
    const double D = std::sqrt(eps);
    const double tune = 4.685;

    Eigen::VectorXd b0 = Eigen::VectorXd::Zero(p);
    int wxrank = xrank;
    int iter = 1;
    auto not_converged = [&]() {
        for (int k = 0; k < p; ++k)
            if (std::abs(b(k) - b0(k)) > D * std::max(std::abs(b(k)), std::abs(b0(k)))) return true;
        return false;
    };
    while (not_converged()) {
        ++iter;
        if (iter > 5) break;
        Eigen::VectorXd radj = (y - X * b).cwiseProduct(adjfactor);
        std::vector<double> rs(n);
        for (int i = 0; i < n; ++i) rs[i] = std::abs(radj(i));
        std::sort(rs.begin(), rs.end());
        std::vector<double> tail(rs.begin() + std::max(1, wxrank) - 1, rs.end());
        double mad_s = median_of(tail) / 0.6745;
        double scale = std::max(mad_s, tiny_s) * tune;
        Eigen::VectorXd sw(n);
        for (int i = 0; i < n; ++i) {
            double r = radj(i) / scale;
            double wi = (std::abs(r) < 1.0) ? (1.0 - r * r) * (1.0 - r * r) : 0.0;
            sw(i) = std::sqrt(wi);
        }
        b0 = b;
        Eigen::MatrixXd xw = X.array().colwise() * sw.array();
        Eigen::VectorXd yw = y.cwiseProduct(sw);
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> wqr(xw);
        wxrank = static_cast<int>(wqr.rank());
        b = wqr.solve(yw);
    }
    return b;
}

// autoTmask: flags observations whose green or SWIR1 residual from a robust
// 2-frequency harmonic fit exceeds T_const x adj_rmse.
std::vector<int> auto_tmask(const std::vector<double>& jd, const std::vector<double>& b1,
                            const std::vector<double>& b2, double yr, double var1, double var2) {
    yr = std::ceil(yr);
    int n = static_cast<int>(jd.size());
    Eigen::MatrixXd X(n, 4);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = std::cos(W * jd[i]);
        X(i, 1) = std::sin(W * jd[i]);
        X(i, 2) = std::cos((W / yr) * jd[i]);
        X(i, 3) = std::sin((W / yr) * jd[i]);
    }
    Eigen::VectorXd y1 = Eigen::Map<const Eigen::VectorXd>(b1.data(), n);
    Eigen::VectorXd y2 = Eigen::Map<const Eigen::VectorXd>(b2.data(), n);
    Eigen::VectorXd c1 = robust_fit(X, y1), c2 = robust_fit(X, y2);
    std::vector<int> mask(n, 0);
    for (int i = 0; i < n; ++i) {
        double t = jd[i];
        double p1 = c1(0) + c1(1) * std::cos(t * W) + c1(2) * std::sin(t * W)
                  + c1(3) * std::cos(t * W / yr) + c1(4) * std::sin(t * W / yr);
        double p2 = c2(0) + c2(1) * std::cos(t * W) + c2(2) * std::sin(t * W)
                  + c2(3) * std::cos(t * W / yr) + c2(4) * std::sin(t * W / yr);
        if (std::abs(b1[i] - p1) > T_CONST * var1 || std::abs(b2[i] - p2) > T_CONST * var2) mask[i] = 1;
    }
    return mask;
}

int update_cft(int i_span, int num_c) {
    if (i_span < MID_NUM_C * N_TIMES) return std::min(MIN_NUM_C, num_c);
    if (i_span < MAX_NUM_C * N_TIMES) return std::min(MID_NUM_C, num_c);
    return std::min(MAX_NUM_C, num_c);
}

// ---------------------------------------------------------------------------
// One pixel of TrendSeasonalFit_v12_30Line.m. Indices written 1-based like the
// original (i, i_start, i_break, ...) and mapped to 0-based storage in the
// accessors, so the code can be read against the MATLAB line by line.

struct Rec {
    bool set = false;
    CCDCSegment s;
};

struct PixelData {
    std::vector<double> x;               // datenum dates
    std::vector<std::vector<double>> y;  // [obs][band]
    double X(int i) const { return x[i - 1]; }
    double Y(int i, int b) const { return y[i - 1][b]; }
    int len() const { return static_cast<int>(x.size()); }
    void erase(int i) { x.erase(x.begin() + (i - 1)); y.erase(y.begin() + (i - 1)); }
    std::vector<double> xs(int a, int b) const { return std::vector<double>(x.begin() + (a - 1), x.begin() + b); }
    std::vector<double> ys(int a, int b, int band) const {
        std::vector<double> o; o.reserve(b - a + 1);
        for (int i = a; i <= b; ++i) o.push_back(y[i - 1][band]);
        return o;
    }
};

struct Ctx {
    int nb;
    std::vector<int> B_detect;
    int B1, B2;
    int conse, num_c;
    double T_cg, Tmax_cg;
    const CCDCParams* p;
};

bool is_detect(const Ctx& c, int b) {
    return std::find(c.B_detect.begin(), c.B_detect.end(), b) != c.B_detect.end();
}

std::vector<CCDCSegment> fit_pixel(const std::vector<double>& sdate,
                                   const std::vector<std::vector<double>>& line,  // [time][band]
                                   const std::vector<int>& qa, const Ctx& c) {
    const int num_t = static_cast<int>(sdate.size());
    const int nb = c.nb;
    std::vector<Rec> rec;
    auto R = [&](int k) -> CCDCSegment& {
        if (static_cast<int>(rec.size()) < k) rec.resize(k);
        rec[k - 1].set = true;
        return rec[k - 1].s;
    };
    auto new_seg = [&](CCDCSegment& s) {
        s.rmse.assign(nb, 0.0); s.magnitude.assign(nb, 0.0);
        s.coefs.assign(nb, std::vector<double>(8, 0.0));
    };

    int n_exist = 0;
    for (int t = 0; t < num_t; ++t) n_exist += (qa[t] < 255);
    if (static_cast<double>(n_exist) / num_t < 0.5) return {};

    std::vector<char> idrange(num_t), idclr(num_t), idsn(num_t);
    int n_clr = 0, n_all = 0, n_snow = 0;
    for (int t = 0; t < num_t; ++t) {
        bool ok = true;
        for (int b = 0; b < nb; ++b) {
            double v = line[t][b];
            if (b == c.p->thermal_band) ok = ok && v > c.p->thermal_min && v < c.p->thermal_max;
            else ok = ok && v > c.p->valid_min && v < c.p->valid_max;
        }
        idrange[t] = ok;
        idclr[t] = qa[t] < 2;
        idsn[t] = qa[t] == 3;
        n_clr += idclr[t]; n_all += (qa[t] < 255); n_snow += idsn[t];
    }
    double clr_pct = static_cast<double>(n_clr) / n_all;
    double sn_pct = n_snow / (n_clr + n_snow + 0.01);
    const int n_min = N_TIMES * MIN_NUM_C;

    if (clr_pct < T_CLR) {
        if (sn_pct > T_SN) {
            // mostly snow: one 4-coefficient model per band over clear + snow
            PixelData d;
            for (int t = 0; t < num_t; ++t)
                if (idsn[t] || idclr[t]) { d.x.push_back(sdate[t]); d.y.push_back(line[t]); }
            int n_sn = d.len();
            if (n_sn < n_min) return {};
            CCDCSegment s; new_seg(s);
            for (int b = 0; b < nb; ++b) {
                std::vector<double> xx, yy;
                for (int i = 1; i <= n_sn; ++i) {
                    double v = d.Y(i, b);
                    bool good = (b == c.p->thermal_band) ? (v > c.p->thermal_min && v < c.p->thermal_max)
                                                         : (v < c.p->valid_max);
                    if (good) { xx.push_back(d.X(i)); yy.push_back(v); }
                }
                if (b != c.p->thermal_band && static_cast<int>(xx.size()) < n_min) {
                    s.coefs[b][0] = 10000.0;
                } else {
                    TSFit f = auto_ts_fit(xx, yy, MIN_NUM_C);
                    s.coefs[b] = f.cft; s.rmse[b] = f.rmse;
                }
            }
            s.t_start = static_cast<int>(d.X(1)); s.t_end = static_cast<int>(d.X(n_sn)); s.t_break = 0;
            s.change_prob = 0; s.num_obs = n_sn; s.category = 50 + MIN_NUM_C;
            return {s};
        }
        // not enough clear observations: screen clouds with green < median + 400
        PixelData d;
        for (int t = 0; t < num_t; ++t) if (idrange[t]) { d.x.push_back(sdate[t]); d.y.push_back(line[t]); }
        std::vector<double> g1 = d.ys(1, d.len(), c.B1);
        double thr = median_of(g1) + CLOUD_SCREEN_OFFSET;
        PixelData e;
        for (int i = 1; i <= d.len(); ++i) if (d.Y(i, c.B1) < thr) { e.x.push_back(d.X(i)); e.y.push_back(d.y[i - 1]); }
        if (e.len() < n_min) return {};
        CCDCSegment s; new_seg(s);
        for (int b = 0; b < nb; ++b) {
            TSFit f = auto_ts_fit(e.x, e.ys(1, e.len(), b), MIN_NUM_C);
            s.coefs[b] = f.cft; s.rmse[b] = f.rmse;
        }
        s.t_start = static_cast<int>(e.X(1)); s.t_end = static_cast<int>(e.X(e.len())); s.t_break = 0;
        s.change_prob = 0; s.num_obs = e.len(); s.category = 40 + MIN_NUM_C;
        return {s};
    }

    // ---- normal procedure
    PixelData d;
    for (int t = 0; t < num_t; ++t)
        if (idclr[t] && idrange[t]) { d.x.push_back(sdate[t]); d.y.push_back(line[t]); }
    std::vector<double> adj_rmse(nb, std::numeric_limits<double>::quiet_NaN());
    for (int b = 0; b < nb; ++b) {
        std::vector<double> diffs;
        for (int i = 2; i <= d.len(); ++i) diffs.push_back(std::abs(d.Y(i, b) - d.Y(i - 1, b)));
        adj_rmse[b] = median_of(diffs);
    }

    const int conse = c.conse;
    int i = n_min, i_start = 1, BL_train = 0;
    int num_fc = 1;
    const int rec_fc = num_fc;
    double i_count = 0.0;
    std::vector<std::vector<double>> fit_cft(nb, std::vector<double>(8, 0.0));
    std::vector<double> rmse(nb, 0.0), tmpcg_rmse(nb, 0.0);
    std::vector<std::vector<double>> rec_v_dif;  // [obs][band]
    std::vector<std::vector<double>> v_dif, v_dif_mag;  // [row][band]
    std::vector<double> vec_mag;
    std::vector<int> IDsOld;

    auto fit_all = [&](int a, int b, int df, bool keep_resid) {
        if (keep_resid) rec_v_dif.assign(b - a + 1, std::vector<double>(nb, 0.0));
        for (int bb = 0; bb < nb; ++bb) {
            TSFit f = auto_ts_fit(d.xs(a, b), d.ys(a, b, bb), df);
            fit_cft[bb] = f.cft; rmse[bb] = f.rmse;
            if (keep_resid) for (int k = 0; k < b - a + 1; ++k) rec_v_dif[k][bb] = f.v_dif[k];
        }
    };
    auto vec_row = [&](const std::vector<double>& row) {
        std::vector<double> sel;
        for (int b : c.B_detect) sel.push_back(row[b]);
        return norm2_sq(sel);
    };
    auto median_rows = [&](const std::vector<std::vector<double>>& m, int r0, int r1) {  // 0-based [r0, r1)
        std::vector<double> out(nb, 0.0);
        for (int b = 0; b < nb; ++b) {
            std::vector<double> col;
            for (int r = r0; r < r1; ++r) col.push_back(m[r][b]);
            out[b] = median_of(col);
        }
        return out;
    };
    auto first_at_or_after = [&](double t) {
        for (int k = 1; k <= d.len(); ++k) if (d.X(k) >= t) return k;
        return d.len() + 1;
    };

    while (i <= d.len() - conse) {
        int i_span = i - i_start + 1;
        double time_span = (d.X(i) - d.X(i_start)) / NUM_YRS;
        if (i_span >= n_min && time_span >= MINI_YRS) {
            if (BL_train == 0) {
                // Tmask over i_start .. i + conse
                std::vector<double> jx = d.xs(i_start, i + conse);
                std::vector<int> bl = auto_tmask(jx, d.ys(i_start, i + conse, c.B1), d.ys(i_start, i + conse, c.B2),
                                                 (d.X(i + conse) - d.X(i_start)) / NUM_YRS,
                                                 adj_rmse[c.B1], adj_rmse[c.B2]);
                int n_ids = static_cast<int>(bl.size());
                std::vector<int> rmIDs;
                i_span = 0;
                for (int k = 0; k < n_ids - conse; ++k) {
                    if (bl[k] == 1) rmIDs.push_back(i_start + k); else ++i_span;
                }
                if (i_span < n_min) { i = i + 1; continue; }
                PixelData cp = d;
                for (int k = static_cast<int>(rmIDs.size()) - 1; k >= 0; --k) cp.erase(rmIDs[k]);
                int i_rec = i;
                i = i_start + i_span - 1;
                time_span = (cp.X(i) - cp.X(i_start)) / NUM_YRS;
                if (time_span < MINI_YRS) { i = i_rec; i = i + 1; continue; }
                d = cp;
                fit_all(i_start, i, MIN_NUM_C, true);
                std::vector<double> vd(nb, 0.0);
                for (int b : c.B_detect) {
                    double mini_rmse = std::max(adj_rmse[b], rmse[b]);
                    double v_start = rec_v_dif.front()[b] / mini_rmse;
                    double v_end = rec_v_dif.back()[b] / mini_rmse;
                    double v_slope = fit_cft[b][1] * (d.X(i) - d.X(i_start)) / mini_rmse;
                    vd[b] = std::abs(v_slope) + std::abs(v_start) + std::abs(v_end);
                }
                if (vec_row(vd) > c.T_cg) { i_start = i_start + 1; i = i + 1; continue; }

                BL_train = 1;
                i_count = 0.0;
                int i_break = (num_fc == rec_fc) ? 1 : first_at_or_after(R(num_fc - 1).t_break);
                if (i_start > i_break) {
                    int ini_from = i_start - 1;
                    for (int i_ini = ini_from; i_ini >= i_break; --i_ini) {
                        int ini_conse = (i_start - i_break < conse) ? i_start - i_break : conse;
                        v_dif.assign(ini_conse, std::vector<double>(nb, 0.0));
                        v_dif_mag = v_dif;
                        vec_mag.assign(ini_conse, 0.0);
                        for (int ic = 1; ic <= ini_conse; ++ic) {
                            int o = i_ini - ic + 1;
                            for (int b = 0; b < nb; ++b) {
                                v_dif_mag[ic - 1][b] = d.Y(o, b) - ts_pred(d.X(o), fit_cft[b].data());
                                if (is_detect(c, b))
                                    v_dif[ic - 1][b] = v_dif_mag[ic - 1][b] / std::max(adj_rmse[b], rmse[b]);
                            }
                            vec_mag[ic - 1] = vec_row(v_dif[ic - 1]);
                        }
                        if (*std::min_element(vec_mag.begin(), vec_mag.end()) > c.T_cg) break;
                        else if (vec_mag[0] > c.Tmax_cg) { d.erase(i_ini); i = i - 1; }
                        i_start = i_ini;
                    }
                }
                if (num_fc == rec_fc && i_start - i_break >= conse) {
                    fit_all(i_break, i_start - 1, MIN_NUM_C, false);
                    CCDCSegment& s = R(num_fc); new_seg(s);
                    s.t_end = static_cast<int>(d.X(i_start - 1));
                    s.coefs = fit_cft; s.rmse = rmse;
                    s.t_break = static_cast<int>(d.X(i_start));
                    s.change_prob = 1;
                    s.t_start = static_cast<int>(d.X(1));
                    s.category = 10 + MIN_NUM_C;
                    s.num_obs = i_start - i_break;
                    std::vector<double> med = median_rows(v_dif_mag, 0, static_cast<int>(v_dif_mag.size()));
                    for (int b = 0; b < nb; ++b) s.magnitude[b] = -med[b];
                    num_fc = num_fc + 1;
                }
            }

            if (BL_train == 1) {
                int ids_a = i_start, ids_b = i;
                i_span = i - i_start + 1;
                int update_num_c = update_cft(i_span, c.num_c);
                if (i_count == 0.0 || i_span <= MAX_NUM_C * N_TIMES) {
                    i_count = d.X(i) - d.X(i_start);
                    fit_all(ids_a, ids_b, update_num_c, true);
                    CCDCSegment& s = R(num_fc); new_seg(s);
                    s.t_start = static_cast<int>(d.X(i_start));
                    s.t_end = static_cast<int>(d.X(i));
                    s.t_break = 0;
                    s.coefs = fit_cft; s.rmse = rmse;
                    s.change_prob = 0;
                    s.num_obs = i - i_start + 1;
                    s.category = 0 + update_num_c;
                    v_dif.assign(conse, std::vector<double>(nb, 0.0));
                    v_dif_mag = v_dif;
                    vec_mag.assign(conse, 0.0);
                    for (int ic = 1; ic <= conse; ++ic) {
                        for (int b = 0; b < nb; ++b) {
                            v_dif_mag[ic - 1][b] = d.Y(i + ic, b) - ts_pred(d.X(i + ic), fit_cft[b].data());
                            if (is_detect(c, b))
                                v_dif[ic - 1][b] = v_dif_mag[ic - 1][b] / std::max(adj_rmse[b], rmse[b]);
                        }
                        vec_mag[ic - 1] = vec_row(v_dif[ic - 1]);
                    }
                    IDsOld.clear();
                    for (int k = ids_a; k <= ids_b; ++k) IDsOld.push_back(k);
                } else {
                    if (d.X(i) - d.X(i_start) >= 1.33 * i_count) {
                        i_count = d.X(i) - d.X(i_start);
                        fit_all(ids_a, ids_b, update_num_c, true);
                        CCDCSegment& s = R(num_fc);
                        s.coefs = fit_cft; s.rmse = rmse;
                        s.num_obs = i - i_start + 1;
                        s.category = 0 + update_num_c;
                        IDsOld.clear();
                        for (int k = ids_a; k <= ids_b; ++k) IDsOld.push_back(k);
                    }
                    CCDCSegment& s = R(num_fc);
                    s.t_end = static_cast<int>(d.X(i));
                    int n_rmse = N_TIMES * s.category;
                    // RMSE from the n_rmse observations closest in season to i + conse
                    std::vector<double> d_yr(IDsOld.size());
                    for (size_t k = 0; k < IDsOld.size(); ++k) {
                        double d_rt = d.X(IDsOld[k]) - d.X(i + conse);
                        d_yr[k] = std::abs(std::round(d_rt / NUM_YRS) * NUM_YRS - d_rt);
                    }
                    std::vector<int> order(IDsOld.size());
                    std::iota(order.begin(), order.end(), 0);
                    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return d_yr[a] < d_yr[b]; });
                    order.resize(std::min<size_t>(order.size(), n_rmse));
                    for (int b : c.B_detect) {
                        std::vector<double> sel;
                        for (int k : order) sel.push_back(rec_v_dif[IDsOld[k] - IDsOld[0]][b]);
                        double ss = 0.0;
                        for (double v : sel) ss += v * v;
                        tmpcg_rmse[b] = std::sqrt(ss) / std::sqrt(static_cast<double>(n_rmse - s.category));
                    }
                    for (int r = 0; r < conse - 1; ++r) {
                        v_dif[r] = v_dif[r + 1]; v_dif_mag[r] = v_dif_mag[r + 1]; vec_mag[r] = vec_mag[r + 1];
                    }
                    v_dif[conse - 1].assign(nb, 0.0); v_dif_mag[conse - 1].assign(nb, 0.0); vec_mag[conse - 1] = 0.0;
                    for (int b = 0; b < nb; ++b) {
                        v_dif_mag[conse - 1][b] = d.Y(i + conse, b) - ts_pred(d.X(i + conse), fit_cft[b].data());
                        if (is_detect(c, b))
                            v_dif[conse - 1][b] = v_dif_mag[conse - 1][b] / std::max(adj_rmse[b], tmpcg_rmse[b]);
                    }
                    vec_mag[conse - 1] = vec_row(v_dif[conse - 1]);
                }
                if (*std::min_element(vec_mag.begin(), vec_mag.end()) > c.T_cg) {
                    CCDCSegment& s = R(num_fc);
                    s.t_break = static_cast<int>(d.X(i + 1));
                    s.change_prob = 1;
                    s.magnitude = median_rows(v_dif_mag, 0, conse);
                    num_fc = num_fc + 1;
                    i_start = i + 1;
                    BL_train = 0;
                } else if (vec_mag[0] > c.Tmax_cg) {
                    d.erase(i + 1);
                    i = i - 1;
                }
            }
        }
        i = i + 1;
    }

    if (BL_train == 1) {
        int id_last = conse;
        for (int ic = conse; ic >= 1; --ic) {
            if (vec_mag[ic - 1] <= c.T_cg) { id_last = ic; break; }
        }
        CCDCSegment& s = R(num_fc);
        s.change_prob = static_cast<double>(conse - id_last) / conse;
        s.t_end = static_cast<int>(d.X(d.len() - conse + id_last));
        if (conse > id_last) {
            s.t_break = static_cast<int>(d.X(d.len() - conse + id_last + 1));
            s.magnitude = median_rows(v_dif_mag, id_last, conse);
        }
    } else {
        int i_st = (num_fc == rec_fc) ? 1 : first_at_or_after(R(num_fc - 1).t_break);
        if (d.len() - i_st + 1 > conse) {
            std::vector<int> bl = auto_tmask(d.xs(i_st, d.len()), d.ys(i_st, d.len(), c.B1), d.ys(i_st, d.len(), c.B2),
                                             (d.X(d.len()) - d.X(i_st)) / NUM_YRS, adj_rmse[c.B1], adj_rmse[c.B2]);
            int n_ids = static_cast<int>(bl.size());
            for (int k = n_ids - conse - 1; k >= 0; --k) if (bl[k] == 1) d.erase(i_st + k);
        }
        if (d.len() - i_st + 1 >= conse) {
            fit_all(i_st, d.len(), MIN_NUM_C, false);
            CCDCSegment& s = R(num_fc); new_seg(s);
            s.t_start = static_cast<int>(d.X(i_st));
            s.t_end = static_cast<int>(d.X(d.len()));
            s.t_break = 0;
            s.coefs = fit_cft; s.rmse = rmse;
            s.change_prob = 0;
            s.num_obs = d.len() - i_st + 1;
            s.category = 20 + MIN_NUM_C;
        }
    }

    std::vector<CCDCSegment> out;
    for (auto& r : rec) if (r.set) out.push_back(r.s);
    return out;
}

Ctx make_ctx(const CCDCParams& params, int nb) {
    Ctx c;
    c.nb = nb;
    c.p = &params;
    c.B_detect = params.detection_bands;
    if (c.B_detect.empty()) {
        if (nb >= 6) c.B_detect = {1, 2, 3, 4, 5};
        else for (int b = 0; b < nb; ++b) c.B_detect.push_back(b);
    }
    std::vector<int> tb = params.tmask_bands;
    if (tb.empty()) tb = (nb >= 5) ? std::vector<int>{1, 4} : std::vector<int>{0, nb > 1 ? 1 : 0};
    if (tb.size() != 2) throw std::invalid_argument("tmask_bands must hold exactly 2 band indices");
    for (int b : c.B_detect) if (b < 0 || b >= nb) throw std::invalid_argument("detection band index out of range");
    for (int b : tb) if (b < 0 || b >= nb) throw std::invalid_argument("tmask band index out of range");
    c.B1 = tb[0]; c.B2 = tb[1];
    c.conse = params.conseq_anom;
    c.num_c = params.num_c;
    int df = static_cast<int>(c.B_detect.size());
    c.T_cg = chi2inv(params.chi2_prob_threshold, df);
    c.Tmax_cg = chi2inv(params.tmax_cg_prob_threshold, df);
    return c;
}

std::vector<CCDCSegment> fit_one(const std::vector<int>& dates, const std::vector<std::vector<double>>& band_t,
                                 const std::vector<int>& qa, const Ctx& c) {
    int T = static_cast<int>(dates.size());
    // the original reads the images in date order
    std::vector<int> order(T);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return dates[a] < dates[b]; });
    std::vector<double> sdate(T);
    std::vector<std::vector<double>> line(T, std::vector<double>(c.nb));
    std::vector<int> q(T);
    for (int k = 0; k < T; ++k) {
        int t = order[k];
        sdate[k] = static_cast<double>(dates[t]) + DATENUM_OFFSET;
        q[k] = qa[t];
        for (int b = 0; b < c.nb; ++b) line[k][b] = band_t[b][t];
    }
    std::vector<CCDCSegment> segs = fit_pixel(sdate, line, q, c);
    for (auto& s : segs) {
        s.t_start -= DATENUM_OFFSET;
        s.t_end -= DATENUM_OFFSET;
        if (s.t_break > 0) s.t_break -= DATENUM_OFFSET;
    }
    return segs;
}

}  // namespace

std::vector<double> robustfit_bisquare(const std::vector<double>& x, int n_columns, const std::vector<double>& y) {
    const int n = static_cast<int>(y.size());
    Eigen::MatrixXd X(n, n_columns);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n_columns; ++j) X(i, j) = x[static_cast<std::size_t>(i) * n_columns + j];
    Eigen::VectorXd b = robust_fit(X, Eigen::Map<const Eigen::VectorXd>(y.data(), n));
    return std::vector<double>(b.data(), b.data() + b.size());
}

std::vector<CCDCSegment> fit_ccdc(const std::vector<int>& dates,
                                  const std::vector<std::vector<double>>& band_values,
                                  const std::vector<int>& qa,
                                  CCDCParams params) {
    if (band_values.empty() || dates.empty()) return {};
    if (qa.size() != dates.size()) throw std::invalid_argument("qa and dates must have the same length");
    for (const auto& b : band_values)
        if (b.size() != dates.size()) throw std::invalid_argument("every band must have one value per date");
    Ctx c = make_ctx(params, static_cast<int>(band_values.size()));
    return fit_one(dates, band_values, qa, c);
}

pybind11::tuple fit_ccdc_batch(
    pybind11::array_t<double> values_array, // Shape: [Y, X, Bands, Time]
    pybind11::array_t<int> qa_array,        // Shape: [Y, X, Time]
    pybind11::array_t<int> dates_array,     // Shape: [Time]
    CCDCParams params,
    int max_segments,
    bool return_coefs,
    int n_jobs)
{
    auto val_buf = values_array.request();
    auto qa_buf = qa_array.request();
    auto dates_buf = dates_array.request();

    int height = static_cast<int>(val_buf.shape[0]);
    int width = static_cast<int>(val_buf.shape[1]);
    int num_bands = static_cast<int>(val_buf.shape[2]);
    int times = static_cast<int>(val_buf.shape[3]);
    int num_pixels = height * width;

    double* val_ptr = static_cast<double*>(val_buf.ptr);
    int* qa_ptr = static_cast<int*>(qa_buf.ptr);
    int* dates_ptr = static_cast<int*>(dates_buf.ptr);
    std::vector<int> dates(dates_ptr, dates_ptr + times);

    Ctx c = make_ctx(params, num_bands);

    // per segment: t_start, t_end, t_break, then per band rmse + 8 coefficients
    int params_per_segment = return_coefs ? (3 + num_bands * 9) : 1;
    pybind11::array_t<double> segments_out({num_pixels, max_segments, params_per_segment});
    auto seg_ptr = static_cast<double*>(segments_out.request().ptr);
    pybind11::array_t<int> counts_out(num_pixels);
    auto counts_ptr = static_cast<int*>(counts_out.request().ptr);
    std::fill(seg_ptr, seg_ptr + (static_cast<size_t>(num_pixels) * max_segments * params_per_segment), 0.0);
    std::fill(counts_ptr, counts_ptr + num_pixels, 0);

    #ifdef _OPENMP
    int num_threads = n_jobs > 0 ? n_jobs : std::max(1, omp_get_num_procs() - 1);
    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    #endif
    for (int p = 0; p < num_pixels; ++p) {
        std::vector<std::vector<double>> pixel_bands(num_bands, std::vector<double>(times));
        std::vector<int> pixel_qa(times);
        bool any_data = false;
        for (int t = 0; t < times; ++t) {
            pixel_qa[t] = qa_ptr[static_cast<size_t>(p) * times + t];
            bool nan_here = false;
            for (int b = 0; b < num_bands; ++b) {
                double v = val_ptr[static_cast<size_t>(p) * num_bands * times + static_cast<size_t>(b) * times + t];
                pixel_bands[b][t] = v;
                if (std::isnan(v)) nan_here = true;
                else if (v != 0.0) any_data = true;
            }
            if (nan_here) {  // no observation on this date
                pixel_qa[t] = 255;
                for (int b = 0; b < num_bands; ++b) pixel_bands[b][t] = 0.0;
            }
        }
        if (!any_data) continue;

        std::vector<CCDCSegment> segs;
        try {
            segs = fit_one(dates, pixel_bands, pixel_qa, c);
        } catch (...) {
        }

        int n_segs = std::min(static_cast<int>(segs.size()), max_segments);
        counts_ptr[p] = n_segs;
        for (int i = 0; i < n_segs; ++i) {
            const auto& seg = segs[i];
            size_t base = static_cast<size_t>(p) * max_segments * params_per_segment + static_cast<size_t>(i) * params_per_segment;
            if (return_coefs) {
                seg_ptr[base + 0] = seg.t_start;
                seg_ptr[base + 1] = seg.t_end;
                seg_ptr[base + 2] = seg.t_break;
                size_t idx = 3;
                for (int b = 0; b < num_bands; ++b) {
                    seg_ptr[base + idx++] = seg.rmse[b];
                    for (int k = 0; k < 8; ++k) seg_ptr[base + idx++] = seg.coefs[b][k];
                }
            } else {
                seg_ptr[base + 0] = seg.t_break;
            }
        }
    }

    return pybind11::make_tuple(segments_out, counts_out);
}

} // namespace ccdc
} // namespace zeit
