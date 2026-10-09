#include "whittaker.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_num_procs() 1
#endif

namespace zeit {
namespace smooth {

std::vector<double> whittaker(const std::vector<double>& x, const std::vector<double>& y,
                              const std::vector<double>& w, double lambda) {
    const int n = static_cast<int>(y.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> z(n, nan);
    int weighted = 0;
    std::vector<double> wy(n), ww(n);
    for (int i = 0; i < n; ++i) {
        const bool ok = std::isfinite(y[i]) && std::isfinite(w[i]) && w[i] > 0.0;
        ww[i] = ok ? w[i] : 0.0;
        wy[i] = ok ? w[i] * y[i] : 0.0;
        if (ok) ++weighted;
    }
    if (weighted < 2) return z;
    if (n < 3 || lambda <= 0.0) {  // nothing to penalise: the observations themselves
        for (int i = 0; i < n; ++i) z[i] = ww[i] > 0.0 ? y[i] : nan;
        return z;
    }

    // The median step, so that an even series has the classic penalty.
    std::vector<double> steps(n - 1);
    for (int i = 0; i + 1 < n; ++i) steps[i] = x[i + 1] - x[i];
    std::vector<double> sorted = steps;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    const double h0 = sorted[sorted.size() / 2] > 0.0 ? sorted[sorted.size() / 2] : 1.0;

    // A = W + lambda D'D: symmetric, two diagonals above the main one.
    std::vector<double> a0(ww), a1(n, 0.0), a2(n, 0.0);
    for (int k = 0; k + 2 < n; ++k) {
        const double h1 = steps[k] / h0, h2 = steps[k + 1] / h0;
        if (!(h1 > 0.0) || !(h2 > 0.0)) throw std::invalid_argument("whittaker: x must increase");
        const double c[3] = {2.0 / (h1 * (h1 + h2)), -2.0 / (h1 * h2), 2.0 / (h2 * (h1 + h2))};
        for (int p = 0; p < 3; ++p) {
            a0[k + p] += lambda * c[p] * c[p];
            if (p < 2) a1[k + p] += lambda * c[p] * c[p + 1];
        }
        a2[k] += lambda * c[0] * c[2];
    }

    // Banded Cholesky A = L L' (L: main diagonal d, below it l1, two below l2).
    std::vector<double> d(n), l1(n, 0.0), l2(n, 0.0);
    for (int i = 0; i < n; ++i) {
        if (i >= 2) l2[i] = a2[i - 2] / d[i - 2];
        if (i >= 1) l1[i] = (a1[i - 1] - (i >= 2 ? l2[i] * l1[i - 1] : 0.0)) / d[i - 1];
        const double s = a0[i] - l1[i] * l1[i] - l2[i] * l2[i];
        if (!(s > 0.0)) return z;  // not positive definite (too few weighted observations)
        d[i] = std::sqrt(s);
    }
    std::vector<double> u(n);  // L u = W y
    for (int i = 0; i < n; ++i) {
        double s = wy[i];
        if (i >= 1) s -= l1[i] * u[i - 1];
        if (i >= 2) s -= l2[i] * u[i - 2];
        u[i] = s / d[i];
    }
    for (int i = n - 1; i >= 0; --i) {  // L' z = u
        double s = u[i];
        if (i + 1 < n) s -= l1[i + 1] * z[i + 1];
        if (i + 2 < n) s -= l2[i + 2] * z[i + 2];
        z[i] = s / d[i];
    }
    return z;
}

pybind11::array_t<double> whittaker_batch(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> values,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> x,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> weights,
    double lambda, int n_jobs) {
    if (values.ndim() != 2 || x.ndim() != 1) throw std::invalid_argument("values (pixels, time), x (time,)");
    const auto pixels = static_cast<long long>(values.shape(0));
    const auto t = static_cast<int>(values.shape(1));
    if (x.shape(0) != t) throw std::invalid_argument("one x per time step");
    const bool has_w = weights.size() > 0;
    if (has_w && (weights.ndim() != 2 || weights.shape(0) != values.shape(0) || weights.shape(1) != t)) {
        throw std::invalid_argument("weights must be (pixels, time), as values");
    }
    const double* v = values.data();
    const double* wp = has_w ? weights.data() : nullptr;
    std::vector<double> xs(x.data(), x.data() + t);
    pybind11::array_t<double> out({static_cast<pybind11::ssize_t>(pixels), static_cast<pybind11::ssize_t>(t)});
    double* o = out.mutable_data();
    {
        pybind11::gil_scoped_release release;
        const int threads = n_jobs > 0 ? n_jobs : std::max(1, omp_get_num_procs() - 1);
        #pragma omp parallel num_threads(threads)
        {
            std::vector<double> y(t), w(t, 1.0);
            #pragma omp for schedule(dynamic, 256)
            for (long long p = 0; p < pixels; ++p) {
                std::copy(v + p * t, v + (p + 1) * t, y.begin());
                if (wp != nullptr) std::copy(wp + p * t, wp + (p + 1) * t, w.begin());
                std::vector<double> z = whittaker(xs, y, w, lambda);
                std::copy(z.begin(), z.end(), o + p * t);
            }
        }
    }
    return out;
}

} // namespace smooth
} // namespace zeit
