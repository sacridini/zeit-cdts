#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace zeit {
namespace coded {

// Continuous Degradation Detection (CODED; Bullock, Woodcock & Olofsson 2020), the change
// monitoring of one pixel's NDFI series, as in CODED's paper version (its cdd_simple.py):
//
// 1. Training: an ordinary least squares fit of NDFI = c + a sin(2 pi t) + b cos(2 pi t)
//    (t in decimal years) to the observations in [train_start, train_start + train_years),
//    with its RMSE; at least min_obs observations, or nothing is monitored.
// 2. Monitoring, from the end of the training period: each observation's residual from the
//    model, divided by the RMSE. An observation beyond thresh (below -thresh with loss_only)
//    adds one to a run of consecutive ones, another resets it (missing ones do neither). A run
//    of consec observations is a change, dated at its first observation, with the mean
//    residual of the run as its (signed) magnitude.
// 3. After a change, a new model is fitted to the observations of the next min_years years
//    (the land cover after the change, which tells degradation from deforestation) and the
//    monitoring goes on from there with it; without min_obs observations it stops.
//
// The same models are fitted to the other series (`features`, e.g. the fractions GV, NPV,
// soil and shade), as the features of the land cover classification.
//
// Output, per pixel (layout in coded_layout): the training model of NDFI (c, a, b), its RMSE,
// its number of observations, the training models of the features, the number of changes,
// then for each change: its date, the date of the observation before it, its magnitude, the
// NDFI model after it (c, a, b), that model's RMSE and the feature models after it.
// NaN where there is nothing.
struct Params {
    double train_start = 0.0;
    double train_years = 3.0;
    int consec = 3;
    double thresh = 3.0;
    double min_years = 3.0;
    int min_obs = 6;
    bool loss_only = true;
    int max_events = 3;
};

std::vector<double> coded_pixel(const std::vector<double>& t, const std::vector<double>& ndfi,
                                const std::vector<double>& features, int n_features, const Params& p);

// coded_pixel on (pixels, time) NDFI and (pixels, n_features, time) features with OpenMP.
// Returns (pixels, coded_size(n_features, max_events)).
pybind11::array_t<double> coded_batch(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> t,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> ndfi,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> features,
    const Params& params, int n_jobs = -1);

int coded_size(int n_features, int max_events);

} // namespace coded
} // namespace zeit
