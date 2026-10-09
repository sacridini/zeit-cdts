#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace zeit {
namespace landtrendr {

// Struct to hold parameters for LandTrendr
struct LandTrendrParams {
    int max_segments = 6;
    double pval_threshold = 0.05;
    bool prevent_fast_recovery = true;     // Reject biologically impossible rapid recoveries
    double recovery_threshold = 0.25;      // Max recovery rate per year
    double spike_threshold = 0.9;          // Desawtooth dampening factor (1.0 = no dampening)
    double best_model_proportion = 0.75;   // Prefer the most-vertex model whose p-value is
                                            // at most (2 - this) times the lowest p-value
                                            // found among candidate models (pick_best_model6)
    int vertex_count_overshoot = 3;        // Extra vertices allowed in the initial candidate
                                            // pool beyond max_segments + 1, pruned back down
                                            // before model selection (LT-GEE's vertexCountOvershoot)
    int min_observations_needed = 6;       // Below this many observations, skip fitting entirely
                                            // and pass the raw trajectory through unsegmented
                                            // (LT-GEE's minObservationsNeeded)
    double modifier = 1.0;                 // +1.0 or -1.0. fit_trajectory_v2.pro multiplies the
                                            // (desawtoothed) series by this before segmentation so
                                            // that whichever direction of change the caller cares
                                            // about always reads as an INCREASE internally -- the
                                            // asymmetric heuristics (split_series' trailing-edge
                                            // recovery suppression, check_slopes' recovery-rate
                                            // eligibility check) are only meaningful relative to
                                            // that convention. Output vertex values are multiplied
                                            // back by modifier before being returned, so callers
                                            // always see real, original-scale values regardless.
};

// Struct to hold output vertices
struct Vertex {
    int year;
    double value;
};

// Result of fitting a single pixel's trajectory: the selected model's
// vertices, plus the RMSE of that fit against every observation. The RMSE is
// LT-GEE's per-pixel noise estimate, used downstream to compute DSNR
// (disturbance magnitude / fit RMSE) for change-map filtering/sorting.
struct TrajectoryResult {
    std::vector<Vertex> vertices;
    double rmse = 0.0;
};

// Core function to run LandTrendr on a single pixel time series
TrajectoryResult fit_trajectory_impl(const std::vector<int>& years,
                                      const std::vector<double>& values,
                                      const LandTrendrParams& params);

// Convenience wrapper over fit_trajectory_impl for callers that only need the
// vertices (e.g. the single-pixel Python binding).
std::vector<Vertex> fit_trajectory(const std::vector<int>& years,
                                   const std::vector<double>& values,
                                   const LandTrendrParams& params);

// Desawtooth function
std::vector<double> desawtooth(const std::vector<double>& vals, double stopat = 0.9);

// Fitted-to-vertices of another band (ftv_v1.pro): the fitted value of every
// year, from the vertex years the segmentation found.
std::vector<double> fit_to_vertices(const std::vector<int>& years, const std::vector<double>& values,
                                    const std::vector<int>& vertex_years, double spike_threshold = 0.9);

// fit_to_vertices on a batch of pixels with OpenMP: values (pixels, time),
// vertex_years (pixels, vertices) with counts (pixels) of them -> (pixels, time).
pybind11::array_t<double> fit_to_vertices_batch(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> values_array,
    pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> years_array,
    pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> vertex_years,
    pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> counts,
    double spike_threshold = 0.9, int n_jobs = -1);

// New batch fit function
pybind11::tuple fit_trajectory_batch(
    pybind11::array_t<double> values_array, // Shape: [Y, X, Time]
    pybind11::array_t<int> years_array,     // Shape: [Time]
    LandTrendrParams params,
    double no_data_value = -9999.0,
    int n_jobs = -1);

} // namespace landtrendr
} // namespace zeit
