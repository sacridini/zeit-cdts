#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace zeit {
namespace smooth {

// Whittaker smoother (Eilers 2003) of one series at positions x: minimises
// sum w (y - z)^2 + lambda sum (second differences of z)^2, with divided
// differences for uneven x, scaled by the median step so that an even series
// gets the classic [1, -2, 1] penalty (and lambda keeps its usual meaning).
// NaN observations weigh 0 (the curve goes through the gaps). All NaN with
// fewer than two weighted observations.
std::vector<double> whittaker(const std::vector<double>& x, const std::vector<double>& y,
                              const std::vector<double>& w, double lambda);

// whittaker on (pixels, time) values with OpenMP; weights: (pixels, time) or empty.
pybind11::array_t<double> whittaker_batch(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> values,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> x,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> weights,
    double lambda, int n_jobs = -1);

} // namespace smooth
} // namespace zeit
