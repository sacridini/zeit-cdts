#pragma once
#include <cstdint>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace zeit {
namespace tmask {

// Tmask (Zhu & Woodcock 2014) of one pixel's clear-candidate observations: a robust
// (bisquare, MATLAB robustfit as CCDC's autoTmask runs it) fit of
// a0 + c1 t + a1 cos(wt) + b1 sin(wt), w = 2 pi / 365.25, to the green and the SWIR
// series; an observation whose green residual is above cloud_threshold (cloud) or whose
// SWIR residual is below -shadow_threshold (shadow), in reflectance (values / scale), is
// flagged. days: dates in days (any origin). Returns 1 = clear, 0 = flagged. With fewer
// than 5 observations nothing is flagged.
std::vector<std::uint8_t> tmask_pixel(const std::vector<double>& days, const std::vector<double>& green,
                                      const std::vector<double>& swir, double scale = 10000.0,
                                      double cloud_threshold = 0.04, double shadow_threshold = 0.04);

// tmask_pixel on (pixels, time) green and SWIR with OpenMP. An observation is valid when
// both values are finite and above 0; pixels with more than min_observations valid ones are
// screened, the others have their valid observations clear. Returns (pixels, time) uint8:
// 1 = clear, 0 = flagged or not valid.
pybind11::array_t<std::uint8_t> tmask_batch(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> green,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> swir,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> days,
    double scale = 10000.0, int min_observations = 5, double cloud_threshold = 0.04,
    double shadow_threshold = 0.04, int n_jobs = -1);

} // namespace tmask
} // namespace zeit
