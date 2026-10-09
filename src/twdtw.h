#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace zeit {
namespace twdtw {

struct TWDTWParams {
    double alpha = 0.1;
    double beta = 0.05;
    double gamma = 50.0;
    int max_time_warp = 365; // Sakoe-Chiba band
    bool subsequence_matching = false; // Open-ended DTW
    // Length of the time cycle in date units (366 for days of the year): the time
    // elapsed between two dates goes around it, as R's twdtw (cycle_length = "year",
    // time_scale = "day"). 0: the plain difference of the dates.
    double cycle_length = 0.0;
};

// Time elapsed between two dates, around the cycle when there is one (twdtw's ellapsed).
inline int elapsed_time(int a, int b, double cycle_length) {
    int r = a > b ? a - b : b - a;
    if (cycle_length > 0.0 && r > cycle_length / 2.0) {
        double wrapped = cycle_length - r;
        r = static_cast<int>(wrapped < 0 ? -wrapped : wrapped);
    }
    return r;
}

struct TWDTWResult {
    double distance;
    std::vector<std::pair<int, int>> path; // [(ts_idx, pat_idx), ...] Empty if not requested
};

// Calculates the TWDTW distance and optionally the path
TWDTWResult fit_twdtw(
    const std::vector<double>& ts_values,
    const std::vector<int>& ts_dates,
    const std::vector<double>& pattern_values,
    const std::vector<int>& pattern_dates,
    int num_bands = 1,
    const TWDTWParams& params = TWDTWParams(),
    double abort_threshold = std::numeric_limits<double>::infinity(),
    bool return_path = false
);

// Batch version for applying TWDTW to a 3D or 4D array
pybind11::array_t<double> fit_twdtw_batch(
    pybind11::array_t<double> values_array, // Shape: [Y, X, Time] or [Y, X, Time, Bands]
    pybind11::array_t<int> dates_array,     // Shape: [Time]
    pybind11::array_t<double> pattern_values_array, // Shape: [PatternTime] or [PatternTime, Bands]
    pybind11::array_t<int> pattern_dates_array,     // Shape: [PatternTime]
    const TWDTWParams& params,
    double abort_threshold = std::numeric_limits<double>::infinity(),
    int n_jobs = -1
);

} // namespace twdtw
} // namespace zeit
