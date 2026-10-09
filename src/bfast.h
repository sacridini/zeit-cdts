#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// bfast: the classic iterative break detection in the seasonal and trend
// components of a time series, ported from the R package `bfast`'s
// `bfast()` (Verbesselt, Hyndman & Newnham, 2010, doi:10.1016/j.rse.2009.08.014;
// Verbesselt, Zeileis & Herold, 2012, doi:10.1016/j.rse.2011.09.024) and its
// `strucchangeRcpp` dependency's `breakpoints()` (the same Bai & Perron,
// 2003 optimal multiple-breakpoint dynamic program bfast_lite.cpp already
// implements).
//
// Unlike bfastlite (a single-pass fit of trend+harmonic together) and
// bfastmonitor (single break/no-break, near-real-time), the classic bfast()
// alternates between two segmented regressions until they agree:
//   1. Trend breakpoints on Vt = Yt - St (deseasonalized), model `~ trend`.
//   2. Season breakpoints on Wt = Yt - Tt (detrended), model `~ harmonic`.
// St is seeded once via an STL "periodic" decomposition (see
// stl_decompose.h) before the first iteration, then re-estimated from the
// season model's own fit every iteration after. Iterates until both
// breakpoint sets stop changing (or `max_iter` is reached).
//
// Scope (see docs/tutorials/bfast.md for the full rationale and R
// cross-validation results):
//  - season = "harmonic" only (matches bfast_lite/bfast_monitor's own design
//    matrix; R's `season = "dummy"` seasonal-factor model is not ported).
//  - `breaks = "BIC"` only (R's own default when `breaks = NULL`; bfastlite's
//    `breaks = "LWZ"` alternative is not exposed here).
//  - The preliminary `sctest(efp(Vt ~ ti, h, type="OLS-MOSUM"))` /
//    `sctest(efp(smod, h, type="OLS-MOSUM"))` structural-stability pre-check
//    IS ported (see `ols_mosum_sctest` in bfast.cpp): a breakpoint search is
//    only attempted when this retrospective OLS-MOSUM test's p-value is
//    <= `level` (matching R exactly, including strucchange's `sc.me`
//    critical-value table for the "Brownian bridge increments" limiting
//    process) - this matters empirically, not just theoretically: an
//    earlier version of this port always attempted the BIC search, which
//    measurably over-detects weak/borderline breaks relative to R's default
//    (confirmed by comparing against real R output - see bfast.md's
//    Validation section).
//  - `decomp = "stl"` only (R's `decomp = "stlplus"` NA-tolerant alternative
//    is not ported). Series with internal NaN gaps are linearly interpolated
//    for this one-time STL seed only - the iterative trend/season fits
//    themselves still skip NaN rows natively, as in bfast_lite/bfast_monitor.

namespace zeit {
namespace bfast {

struct BFResult {
    double n_trend_breaks = std::nan("");
    double n_season_breaks = std::nan("");
    double magnitude = std::nan("");   // size of the largest trend-component jump, or 0 if no trend break
    double time = std::nan("");        // fractional-year time of the largest trend-component jump
    double n_iter = std::nan("");      // number of iterations run before convergence (or hitting max_iter)
    double n_valid = std::nan("");
    double valid = 0.0;
    // 0-based indices into the full series (NaN rows included) of the last observation before each break
    std::vector<double> trend_breakpoint_idx;
    std::vector<double> season_breakpoint_idx;
    std::vector<double> trend_magnitude;       // per trend break, the jump `magnitude` is the largest of
    std::vector<double> trend_first_after_idx; // per trend break, the index of the first observation after it
};

// Runs bfast on a single pixel's time series.
BFResult bfast(
    const std::vector<double>& y,
    double start_time,
    int frequency,
    int order = 3,
    double h = 0.15,
    int max_breaks_trend = 5,
    int max_breaks_season = 5,
    int max_iter = 10,
    double level = 0.05);

// Batch entry point: values_array [n_pixels, n_time] -> out
// [7 + 3 * max_breaks_trend + max_breaks_season, n_pixels]. Row order:
// n_trend_breaks, n_season_breaks, magnitude, time, n_iter, n_valid, valid,
// trend_breakpoint_idx_1..max_breaks_trend,
// season_breakpoint_idx_1..max_breaks_season,
// trend_magnitude_1..max_breaks_trend,
// trend_first_after_idx_1..max_breaks_trend (NaN-padded past each count).
pybind11::array_t<double> fit_bfast_batch(
    pybind11::array_t<double> values_array,
    double start_time,
    int frequency,
    int order = 3,
    double h = 0.15,
    int max_breaks_trend = 5,
    int max_breaks_season = 5,
    int max_iter = 10,
    double level = 0.05,
    int min_valid = 20,
    int n_jobs = -1);

} // namespace bfast
} // namespace zeit
