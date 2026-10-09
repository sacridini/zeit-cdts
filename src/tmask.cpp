#include "tmask.h"
#include "ccdc.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_num_procs() 1
#endif

namespace zeit {
namespace tmask {

namespace {
const double W = 2.0 * 3.14159265358979323846 / 365.25;  // the annual cycle
}

std::vector<std::uint8_t> tmask_pixel(const std::vector<double>& days, const std::vector<double>& green,
                                      const std::vector<double>& swir, double scale, double cloud_threshold,
                                      double shadow_threshold) {
    const int n = static_cast<int>(days.size());
    if (static_cast<int>(green.size()) != n || static_cast<int>(swir.size()) != n) {
        throw std::invalid_argument("tmask_pixel: days, green and swir must have the same length");
    }
    std::vector<std::uint8_t> clear(n, 1);
    if (n < 5) return clear;
    // Days from the first date: the fit is the same with any origin, and better conditioned.
    const double t0 = *std::min_element(days.begin(), days.end());
    std::vector<double> x(static_cast<std::size_t>(n) * 3);
    for (int i = 0; i < n; ++i) {
        const double t = days[i] - t0;
        x[3 * i] = t;
        x[3 * i + 1] = std::cos(W * days[i]);
        x[3 * i + 2] = std::sin(W * days[i]);
    }
    std::vector<double> g(n), s(n);
    for (int i = 0; i < n; ++i) {
        g[i] = green[i] / scale;
        s[i] = swir[i] / scale;
    }
    const std::vector<double> bg = zeit::ccdc::robustfit_bisquare(x, 3, g);
    const std::vector<double> bs = zeit::ccdc::robustfit_bisquare(x, 3, s);
    for (int i = 0; i < n; ++i) {
        const double* xi = &x[3 * i];
        const double pg = bg[0] + bg[1] * xi[0] + bg[2] * xi[1] + bg[3] * xi[2];
        const double ps = bs[0] + bs[1] * xi[0] + bs[2] * xi[1] + bs[3] * xi[2];
        if (g[i] - pg > cloud_threshold || s[i] - ps < -shadow_threshold) clear[i] = 0;
    }
    return clear;
}

pybind11::array_t<std::uint8_t> tmask_batch(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> green,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> swir,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> days,
    double scale, int min_observations, double cloud_threshold, double shadow_threshold, int n_jobs) {
    if (green.ndim() != 2 || swir.ndim() != 2 || days.ndim() != 1) {
        throw std::invalid_argument("green and swir (pixels, time), days (time,)");
    }
    const auto pixels = static_cast<long long>(green.shape(0));
    const auto t = static_cast<int>(green.shape(1));
    if (swir.shape(0) != green.shape(0) || swir.shape(1) != t || days.shape(0) != t) {
        throw std::invalid_argument("green, swir and days do not match");
    }
    const double* gp = green.data();
    const double* sp = swir.data();
    std::vector<double> all_days(days.data(), days.data() + t);
    pybind11::array_t<std::uint8_t> out({static_cast<pybind11::ssize_t>(pixels), static_cast<pybind11::ssize_t>(t)});
    std::uint8_t* o = out.mutable_data();
    std::fill(o, o + pixels * t, static_cast<std::uint8_t>(0));
    {
        pybind11::gil_scoped_release release;
        const int threads = n_jobs > 0 ? n_jobs : std::max(1, omp_get_num_procs() - 1);
        #pragma omp parallel num_threads(threads)
        {
            std::vector<int> idx;
            std::vector<double> d, g, s;
            #pragma omp for schedule(dynamic, 64)
            for (long long p = 0; p < pixels; ++p) {
                idx.clear();
                for (int i = 0; i < t; ++i) {
                    const double gv = gp[p * t + i], sv = sp[p * t + i];
                    if (std::isfinite(gv) && std::isfinite(sv) && gv > 0.0 && sv > 0.0) idx.push_back(i);
                }
                std::uint8_t* row = o + p * t;
                if (static_cast<int>(idx.size()) <= min_observations) {
                    for (int i : idx) row[i] = 1;
                    continue;
                }
                d.resize(idx.size());
                g.resize(idx.size());
                s.resize(idx.size());
                for (std::size_t k = 0; k < idx.size(); ++k) {
                    d[k] = all_days[idx[k]];
                    g[k] = gp[p * t + idx[k]];
                    s[k] = sp[p * t + idx[k]];
                }
                const std::vector<std::uint8_t> clear = tmask_pixel(d, g, s, scale, cloud_threshold, shadow_threshold);
                for (std::size_t k = 0; k < idx.size(); ++k) row[idx[k]] = clear[k];
            }
        }
    }
    return out;
}

} // namespace tmask
} // namespace zeit
