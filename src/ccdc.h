#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace zeit {
namespace ccdc {

// One time-series model of a pixel -- one element of the original's rec_cg.
struct CCDCSegment {
    int t_start;         // first observation of the model (input date convention)
    int t_end;           // last observation of the model
    int t_break;         // first observation of the change, 0 if none
    std::vector<double> rmse;               // per band
    std::vector<std::vector<double>> coefs; // [band][8]: a0 c1 a1 b1 a2 b2 a3 b3 (time axis = date + 366)
    std::vector<double> magnitude;          // per band, 0 if no change
    double change_prob = 0.0;               // 0..1
    int category = 0;                       // the original's QA code + number of coefficients
    int num_obs = 0;                        // clear observations used by the model
};

// MATLAB's robustfit as CCDC's autoTmask runs it (bisquare, tune 4.685, at most 5
// iterations; robust_fit in ccdc.cpp): the coefficients [intercept, x columns...] of y on
// the n x n_columns predictors x (row-major), an intercept added.
std::vector<double> robustfit_bisquare(const std::vector<double>& x, int n_columns, const std::vector<double>& y);

struct CCDCParams {
    int min_obs = 12;                       // kept for API compatibility; the original's 3 x 4 is fixed
    int conseq_anom = 6;                    // the original's `conse`
    double chi2_prob_threshold = 0.99;      // change probability -> T_cg
    double tmax_cg_prob_threshold = 0.999999; // Tmax_cg = chi2inv(1 - 1e-6)
    int num_c = 8;                          // max number of coefficients: 4, 6 or 8
    std::vector<int> detection_bands;       // B_detect (0-based); empty -> {1..5} if >= 6 bands, else all
    std::vector<int> tmask_bands;           // Tmask/cloud-screen bands (0-based); empty -> {1, 4} (green, SWIR1)
    int thermal_band = -1;                  // index of a brightness-temperature band (deg C x 100), -1 if none
    double valid_min = 0.0;                 // optical range test (exclusive), surface reflectance x 10000
    double valid_max = 10000.0;
    double thermal_min = -9320.0;           // thermal range test (exclusive)
    double thermal_max = 7070.0;
};

// CCDC for one pixel (TrendSeasonalFit_v12_30Line.m). dates: integer days
// (Python ordinal days), bands: [band][time], qa: Fmask codes (0 clear land,
// 1 water, 2 shadow, 3 snow, 4 cloud, 255 no data).
std::vector<CCDCSegment> fit_ccdc(const std::vector<int>& dates,
                                  const std::vector<std::vector<double>>& bands,
                                  const std::vector<int>& qa,
                                  CCDCParams params = CCDCParams());

// Batch fit function for arrays
pybind11::tuple fit_ccdc_batch(
    pybind11::array_t<double> values_array, // Shape: [Y, X, Bands, Time]
    pybind11::array_t<int> qa_array,        // Shape: [Y, X, Time]
    pybind11::array_t<int> dates_array,     // Shape: [Time]
    CCDCParams params,
    int max_segments = 6,
    bool return_coefs = true,
    int n_jobs = -1);

} // namespace ccdc
} // namespace zeit
