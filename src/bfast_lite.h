#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// bfastlite: single-pass multiple-breakpoint detection, ported from the R
// package `bfast`'s bfastlite() and its `strucchangeRcpp` dependency's
// breakpoints() (the Bai & Perron, 2003 optimal multiple-breakpoint dynamic
// program, via Brown-Durbin-Evans recursive residuals for numerically
// stable segment RSS computation - Bai & Perron 2003, "Computation and
// Analysis of Multiple Structural Change Models", Journal of Applied
// Econometrics).
//
// Scope: same design matrix (`response ~ trend + harmon`) as bfast_monitor,
// fit on the WHOLE series (no history/monitoring split). The optimal number
// of breaks is chosen by minimizing the LWZ criterion (Liu, Wu & Zidek,
// 1997 - bfastlite's own default), matching R's `breaks="LWZ"`. Unlike
// bfastmonitor, this needs no critical-value table - the DP directly
// minimizes a penalized RSS.

namespace zeit {
namespace bfastlite {

struct BFLResult {
    double n_breaks = std::nan("");   // number of breaks selected by LWZ, or NaN if invalid
    double rss = std::nan("");        // total RSS at the selected number of breaks
    double lwz = std::nan("");        // LWZ score at the selected number of breaks
    double n_valid = std::nan("");    // number of valid (non-NaN) observations used
    double valid = 0.0;               // 1.0 if the series had enough observations to fit at all
    std::vector<double> breakpoint_idx; // 0-based indices into the full series (NaN rows included) of the last
                                        // observation before each break, size = n_breaks
    std::vector<double> magnitude;      // per break: the right segment's model minus the left one's, both at the
                                        // first observation after the break, size = n_breaks
    std::vector<double> first_after_idx; // 0-based index into the full series of the first observation after
                                         // each break, size = n_breaks
};

// Runs bfastlite on a single pixel's time series.
BFLResult bfast_lite(
    const std::vector<double>& y,
    double start_time,
    int frequency,
    int order = 3,
    double h = 0.15,
    int max_breaks_output = 5);

// Batch entry point: values_array [n_pixels, n_time] -> out
// [5 + 3 * max_breaks_output, n_pixels]. Row order: n_breaks, rss, lwz, n_valid,
// valid, breakpoint_idx_1 .. breakpoint_idx_{max_breaks_output}, magnitude_1 ..
// magnitude_{max_breaks_output}, first_after_idx_1 .. first_after_idx_{max_breaks_output}
// (NaN-padded past n_breaks).
pybind11::array_t<double> fit_bfast_lite_batch(
    pybind11::array_t<double> values_array,
    double start_time,
    int frequency,
    int order = 3,
    double h = 0.15,
    int max_breaks_output = 5,
    int min_valid = 20,
    int n_jobs = -1);

} // namespace bfastlite
} // namespace zeit
