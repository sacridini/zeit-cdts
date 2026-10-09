#include "sma.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <Eigen/Dense>

namespace zeit {
namespace sma {

namespace {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Least squares on the columns in `passive` (the others stay 0).
Eigen::VectorXd solve_passive(const Matrix& A, const Eigen::VectorXd& b, const std::vector<bool>& passive) {
    const int n = static_cast<int>(A.cols());
    std::vector<int> cols;
    for (int j = 0; j < n; ++j) if (passive[j]) cols.push_back(j);
    Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
    if (cols.empty()) return z;
    Matrix sub(A.rows(), static_cast<Eigen::Index>(cols.size()));
    for (size_t c = 0; c < cols.size(); ++c) sub.col(static_cast<Eigen::Index>(c)) = A.col(cols[c]);
    Eigen::VectorXd s = sub.colPivHouseholderQr().solve(b);
    for (size_t c = 0; c < cols.size(); ++c) z(cols[c]) = s(static_cast<Eigen::Index>(c));
    return z;
}

Eigen::VectorXd nnls_impl(const Matrix& A, const Eigen::VectorXd& b, int max_iter) {
    // Lawson & Hanson (1974), chapter 23, as in scipy.optimize.nnls.
    const int n = static_cast<int>(A.cols());
    if (max_iter < 0) max_iter = 3 * n;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    std::vector<bool> passive(n, false);
    const double tol = 10.0 * std::numeric_limits<double>::epsilon() * A.cwiseAbs().maxCoeff()
                       * static_cast<double>(std::max(A.rows(), A.cols()));
    Eigen::VectorXd w = A.transpose() * (b - A * x);
    int iter = 0;
    for (;;) {
        int best = -1;
        double best_w = tol;
        for (int j = 0; j < n; ++j) if (!passive[j] && w(j) > best_w) { best_w = w(j); best = j; }
        if (best < 0) break;
        passive[best] = true;
        for (;;) {
            if (++iter > max_iter) return x;
            Eigen::VectorXd z = solve_passive(A, b, passive);
            bool feasible = true;
            for (int j = 0; j < n; ++j) if (passive[j] && z(j) <= 0) { feasible = false; break; }
            if (feasible) { x = z; break; }
            double alpha = std::numeric_limits<double>::infinity();
            for (int j = 0; j < n; ++j)
                if (passive[j] && z(j) <= 0) alpha = std::min(alpha, x(j) / (x(j) - z(j)));
            x += alpha * (z - x);
            for (int j = 0; j < n; ++j)
                if (passive[j] && std::abs(x(j)) <= tol) { passive[j] = false; x(j) = 0.0; }
        }
        w = A.transpose() * (b - A * x);
    }
    return x;
}

} // namespace

std::vector<double> nnls(const std::vector<double>& A, const std::vector<double>& b, int m, int n, int max_iter) {
    if (static_cast<int>(A.size()) != m * n || static_cast<int>(b.size()) != m)
        throw std::runtime_error("nnls: A must be m x n and b of length m");
    Matrix a = Eigen::Map<const Matrix>(A.data(), m, n);
    Eigen::VectorXd bb = Eigen::Map<const Eigen::VectorXd>(b.data(), m);
    Eigen::VectorXd x = nnls_impl(a, bb, max_iter);
    return std::vector<double>(x.data(), x.data() + n);
}

namespace {

// The system of one spectrum: E^T (bands x k), plus the sum-to-one row when asked.
void solve_one(const double* y, const Matrix& system, int bands, int k, bool sum_to_one, bool nonneg,
               double delta, double* out) {
    const int rows = bands + (sum_to_one ? 1 : 0);
    Eigen::VectorXd b(rows);
    for (int i = 0; i < bands; ++i) {
        if (!std::isfinite(y[i])) {
            for (int j = 0; j <= k; ++j) out[j] = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        b(i) = y[i];
    }
    if (sum_to_one) b(bands) = delta;
    Eigen::VectorXd f = nonneg ? nnls_impl(system, b, -1) : Eigen::VectorXd(system.colPivHouseholderQr().solve(b));
    Eigen::VectorXd rebuilt = system.topRows(bands) * f;
    double sse = 0.0;
    for (int i = 0; i < bands; ++i) sse += (rebuilt(i) - y[i]) * (rebuilt(i) - y[i]);
    for (int j = 0; j < k; ++j) out[j] = f(j);
    out[k] = std::sqrt(sse / bands);
}

Matrix make_system(const double* endmembers, int k, int bands, bool sum_to_one, double delta) {
    Matrix system(bands + (sum_to_one ? 1 : 0), k);
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < bands; ++i) system(i, j) = endmembers[j * bands + i];
        if (sum_to_one) system(bands, j) = delta;
    }
    return system;
}

} // namespace

std::vector<double> unmix_pixel(const std::vector<double>& y, const std::vector<double>& endmembers, int k,
                                bool sum_to_one, bool nonneg, double delta) {
    const int bands = static_cast<int>(y.size());
    if (k <= 0 || static_cast<int>(endmembers.size()) != k * bands)
        throw std::runtime_error("unmix: endmembers must be k x bands");
    Matrix system = make_system(endmembers.data(), k, bands, sum_to_one, delta);
    std::vector<double> out(k + 1);
    solve_one(y.data(), system, bands, k, sum_to_one, nonneg, delta, out.data());
    return out;
}

pybind11::array_t<double> unmix_batch(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> values,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> endmembers,
    bool sum_to_one, bool nonneg, double delta, int n_jobs) {
    auto v = values.unchecked<2>();
    auto e = endmembers.unchecked<2>();
    const int n = static_cast<int>(v.shape(0)), bands = static_cast<int>(v.shape(1));
    const int k = static_cast<int>(e.shape(0));
    if (static_cast<int>(e.shape(1)) != bands)
        throw std::runtime_error("unmix: the spectra and the endmembers must have the same bands");
    Matrix system = make_system(endmembers.data(), k, bands, sum_to_one, delta);
    pybind11::array_t<double> out_arr({n, k + 1});
    double* out = out_arr.mutable_data();
    const double* in = values.data();
    if (n_jobs <= 0) n_jobs = std::max(1, omp_get_max_threads() - 1);
    {
        pybind11::gil_scoped_release release;
        #pragma omp parallel for num_threads(n_jobs) schedule(static)
        for (int p = 0; p < n; ++p)
            solve_one(in + static_cast<size_t>(p) * bands, system, bands, k, sum_to_one, nonneg, delta,
                      out + static_cast<size_t>(p) * (k + 1));
    }
    return out_arr;
}

} // namespace sma
} // namespace zeit
