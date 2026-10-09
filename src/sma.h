#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace zeit {
namespace sma {

// Linear spectral mixture analysis of one spectrum: the fractions f of the endmembers E
// (k endmembers x b bands) that best rebuild y (b bands), y ~ E^T f.
//
// - nonneg and sum_to_one: fully constrained least squares (Heinz & Chang 2001): the
//   non-negative least squares (Lawson & Hanson 1974) of the system with one more row,
//   delta * 1^T f = delta, so the fractions add up to one up to ~1/delta^2;
// - nonneg only: NNLS; sum_to_one only: least squares with the same extra row;
// - neither: ordinary least squares.
//
// Returns the k fractions followed by the RMSE of the rebuilt spectrum (over the bands).
std::vector<double> unmix_pixel(const std::vector<double>& y, const std::vector<double>& endmembers, int k,
                                bool sum_to_one = true, bool nonneg = true, double delta = 1000.0);

// unmix_pixel on (pixels, bands) spectra with OpenMP. endmembers: (k, bands). A spectrum with
// a non-finite band gives NaN. Returns (pixels, k + 1): the fractions, then the RMSE.
pybind11::array_t<double> unmix_batch(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> values,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> endmembers,
    bool sum_to_one = true, bool nonneg = true, double delta = 1000.0, int n_jobs = -1);

// Lawson & Hanson's NNLS: min ||A x - b|| subject to x >= 0, A (m x n) row-major. The
// reference algorithm (as scipy.optimize.nnls), for the small systems of unmixing.
std::vector<double> nnls(const std::vector<double>& A, const std::vector<double>& b, int m, int n,
                         int max_iter = -1);

} // namespace sma
} // namespace zeit
