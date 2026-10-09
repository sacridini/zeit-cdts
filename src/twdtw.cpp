#include "twdtw.h"
#include <cmath>
#include <algorithm>
#include <limits>
#ifdef _OPENMP
#ifdef _OPENMP
#include <omp.h>
#endif
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_set_num_threads(x) (void)(x)
#endif
#include <Eigen/Dense>

namespace zeit {
namespace twdtw {

class TWDTW_LUT {
public:
    std::vector<double> lut;
    double cycle_length;
    TWDTW_LUT(const TWDTWParams& p, int max_diff = 2000) : cycle_length(p.cycle_length) {
        lut.resize(max_diff + 1);
        for(int i = 0; i <= max_diff; ++i) {
            lut[i] = p.alpha / (1.0 + std::exp(-p.beta * (i - p.gamma)));
        }
    }
    inline double get(int t1, int t2) const {
        int dt = elapsed_time(t1, t2, cycle_length);
        if (dt >= static_cast<int>(lut.size())) return lut.back();
        return lut[dt];
    }
};

TWDTWResult fit_twdtw(const std::vector<double>& ts_values,
                 const std::vector<int>& ts_dates,
                 const std::vector<double>& pattern_values,
                 const std::vector<int>& pattern_dates,
                 int num_bands,
                 const TWDTWParams& params,
                 double abort_threshold,
                 bool return_path) {
    
    int n = ts_dates.size();
    int m = pattern_dates.size();
    TWDTWResult result;
    result.distance = std::numeric_limits<double>::infinity();

    if (n == 0 || m == 0) return result;

    TWDTW_LUT lut(params);

    if (return_path) {
        std::vector<std::vector<double>> d(n + 1, std::vector<double>(m + 1, std::numeric_limits<double>::infinity()));
        
        if (params.subsequence_matching) {
            for (int i = 0; i <= n; ++i) d[i][0] = 0.0;
        } else {
            d[0][0] = 0.0;
        }

        for (int i = 1; i <= n; ++i) {
            double min_in_row = std::numeric_limits<double>::infinity();
            for (int j = 1; j <= m; ++j) {
                if (elapsed_time(ts_dates[i - 1], pattern_dates[j - 1], params.cycle_length) > params.max_time_warp) continue;

                double spatial_dist = 0.0;
                if (num_bands == 1) {
                    spatial_dist = std::abs(ts_values[i - 1] - pattern_values[j - 1]);
                } else {
                    Eigen::Map<const Eigen::VectorXd> v1(const_cast<double*>(ts_values.data()) + (i-1)*num_bands, num_bands);
                    Eigen::Map<const Eigen::VectorXd> v2(const_cast<double*>(pattern_values.data()) + (j-1)*num_bands, num_bands);
                    spatial_dist = (v1 - v2).norm();
                }
                
                double temp_penalty = lut.get(ts_dates[i - 1], pattern_dates[j - 1]);
                double cvalue = spatial_dist + temp_penalty;

                d[i][j] = cvalue + std::min({ d[i - 1][j - 1], d[i - 1][j], d[i][j - 1] });
                min_in_row = std::min(min_in_row, d[i][j]);
            }
            if (min_in_row > abort_threshold) return result;
        }

        int best_i = n;
        if (params.subsequence_matching) {
            double min_val = std::numeric_limits<double>::infinity();
            for (int i = 1; i <= n; ++i) {
                if (d[i][m] < min_val) {
                    min_val = d[i][m];
                    best_i = i;
                }
            }
            result.distance = min_val;
        } else {
            result.distance = d[n][m];
        }

        if (result.distance != std::numeric_limits<double>::infinity()) {
            int i = best_i;
            int j = m;
            while (i > 0 && j > 0) {
                result.path.push_back({i - 1, j - 1});
                double diag = d[i - 1][j - 1];
                double up = d[i - 1][j];
                double left = d[i][j - 1];
                if (diag <= up && diag <= left) { i--; j--; }
                else if (up <= diag && up <= left) { i--; }
                else { j--; }
            }
            std::reverse(result.path.begin(), result.path.end());
        }

    } else {
        std::vector<double> prev_row(m + 1, std::numeric_limits<double>::infinity());
        std::vector<double> curr_row(m + 1, std::numeric_limits<double>::infinity());
        
        if (params.subsequence_matching) {
            prev_row[0] = 0.0;
        } else {
            prev_row[0] = 0.0;
        }

        double final_min = std::numeric_limits<double>::infinity();

        for (int i = 1; i <= n; ++i) {
            if (params.subsequence_matching) curr_row[0] = 0.0;
            else curr_row[0] = std::numeric_limits<double>::infinity();
            
            double min_in_row = std::numeric_limits<double>::infinity();

            for (int j = 1; j <= m; ++j) {
                if (elapsed_time(ts_dates[i - 1], pattern_dates[j - 1], params.cycle_length) > params.max_time_warp) {
                    curr_row[j] = std::numeric_limits<double>::infinity();
                    continue;
                }

                double spatial_dist = 0.0;
                if (num_bands == 1) {
                    spatial_dist = std::abs(ts_values[i - 1] - pattern_values[j - 1]);
                } else {
                    Eigen::Map<const Eigen::VectorXd> v1(const_cast<double*>(ts_values.data()) + (i-1)*num_bands, num_bands);
                    Eigen::Map<const Eigen::VectorXd> v2(const_cast<double*>(pattern_values.data()) + (j-1)*num_bands, num_bands);
                    spatial_dist = (v1 - v2).norm();
                }

                double temp_penalty = lut.get(ts_dates[i - 1], pattern_dates[j - 1]);
                double cvalue = spatial_dist + temp_penalty;

                curr_row[j] = cvalue + std::min({ prev_row[j - 1], prev_row[j], curr_row[j - 1] });
                min_in_row = std::min(min_in_row, curr_row[j]);
            }
            
            if (min_in_row > abort_threshold) return result;
            
            if (params.subsequence_matching) {
                final_min = std::min(final_min, curr_row[m]);
            }

            std::swap(prev_row, curr_row);
        }
        
        if (params.subsequence_matching) {
            result.distance = final_min;
        } else {
            result.distance = prev_row[m];
        }
    }
    
    return result;
}

pybind11::array_t<double> fit_twdtw_batch(
    pybind11::array_t<double> values_array,
    pybind11::array_t<int> dates_array,
    pybind11::array_t<double> pattern_values_array,
    pybind11::array_t<int> pattern_dates_array,
    const TWDTWParams& params,
    double abort_threshold,
    int n_jobs) {
    
    auto buf_values = values_array.request();
    auto buf_dates = dates_array.request();
    auto buf_pat_values = pattern_values_array.request();
    auto buf_pat_dates = pattern_dates_array.request();

    int Y = buf_values.shape[0];
    int X = buf_values.shape[1];
    int T = buf_values.shape[2];
    int num_bands = 1;
    if (buf_values.ndim == 4) {
        num_bands = buf_values.shape[3];
    }

    int P = buf_pat_dates.shape[0];

    const double* values_ptr = static_cast<double*>(buf_values.ptr);
    const int* dates_ptr = static_cast<int*>(buf_dates.ptr);
    const double* pat_values_ptr = static_cast<double*>(buf_pat_values.ptr);
    const int* pat_dates_ptr = static_cast<int*>(buf_pat_dates.ptr);

    auto result_array = pybind11::array_t<double>({Y, X});
    auto buf_result = result_array.request();
    double* result_ptr = static_cast<double*>(buf_result.ptr);

    std::vector<double> pat_vals(pat_values_ptr, pat_values_ptr + P * num_bands);
    std::vector<int> pat_dates(pat_dates_ptr, pat_dates_ptr + P);
    std::vector<int> ts_dates(dates_ptr, dates_ptr + T);

    TWDTW_LUT lut(params);

    // LB_Keogh Precomputation
    bool use_lb = (abort_threshold < std::numeric_limits<double>::infinity()) && !params.subsequence_matching;
    std::vector<std::vector<double>> U(T, std::vector<double>(num_bands, -std::numeric_limits<double>::infinity()));
    std::vector<std::vector<double>> L(T, std::vector<double>(num_bands, std::numeric_limits<double>::infinity()));
    std::vector<bool> valid_window(T, false);

    if (use_lb) {
        for (int t = 0; t < T; ++t) {
            int t_date = ts_dates[t];
            for (int p = 0; p < P; ++p) {
                if (elapsed_time(t_date, pat_dates[p], params.cycle_length) <= params.max_time_warp) {
                    valid_window[t] = true;
                    for (int b = 0; b < num_bands; ++b) {
                        double val = pat_vals[p * num_bands + b];
                        U[t][b] = std::max(U[t][b], val);
                        L[t][b] = std::min(L[t][b], val);
                    }
                }
            }
        }
    }

    #pragma omp parallel for collapse(2) num_threads(n_jobs > 0 ? n_jobs : std::max(1, omp_get_max_threads() - 1))
    for (int y = 0; y < Y; ++y) {
        for (int x = 0; x < X; ++x) {
            std::vector<double> ts_vals(T * num_bands);
            for (int t = 0; t < T * num_bands; ++t) {
                ts_vals[t] = values_ptr[y * X * T * num_bands + x * T * num_bands + t];
            }
            // Dates with a missing value (NaN in any band) leave this pixel's series,
            // as R's twdtw drops incomplete cases.
            std::vector<int> pixel_dates;
            bool complete = true;
            for (int t = 0; t < T && complete; ++t) {
                for (int b = 0; b < num_bands; ++b) {
                    if (std::isnan(ts_vals[t * num_bands + b])) { complete = false; break; }
                }
            }
            if (!complete) {
                std::vector<double> kept;
                for (int t = 0; t < T; ++t) {
                    bool ok = true;
                    for (int b = 0; b < num_bands; ++b) ok = ok && !std::isnan(ts_vals[t * num_bands + b]);
                    if (!ok) continue;
                    for (int b = 0; b < num_bands; ++b) kept.push_back(ts_vals[t * num_bands + b]);
                    pixel_dates.push_back(ts_dates[t]);
                }
                ts_vals.swap(kept);
                if (pixel_dates.empty()) {
                    result_ptr[y * X + x] = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
            }

            if (use_lb && complete) {
                double lb_dist = 0.0;
                bool valid_lb = true;
                for (int t = 0; t < T; ++t) {
                    if (!valid_window[t]) {
                        lb_dist = std::numeric_limits<double>::infinity();
                        break;
                    }
                    double spatial_dist = 0.0;
                    if (num_bands == 1) {
                        double v = ts_vals[t];
                        if (v > U[t][0]) spatial_dist = v - U[t][0];
                        else if (v < L[t][0]) spatial_dist = L[t][0] - v;
                    } else {
                        double sq_dist = 0.0;
                        for (int b = 0; b < num_bands; ++b) {
                            double v = ts_vals[t * num_bands + b];
                            if (v > U[t][b]) sq_dist += (v - U[t][b]) * (v - U[t][b]);
                            else if (v < L[t][b]) sq_dist += (L[t][b] - v) * (L[t][b] - v);
                        }
                        spatial_dist = std::sqrt(sq_dist);
                    }
                    lb_dist += spatial_dist;
                    if (lb_dist > abort_threshold) {
                        break;
                    }
                }
                
                if (lb_dist > abort_threshold) {
                    result_ptr[y * X + x] = std::numeric_limits<double>::infinity();
                    continue; // LB Keogh successfully pruned this calculation!
                }
            }

            TWDTWResult res = fit_twdtw(ts_vals, complete ? ts_dates : pixel_dates, pat_vals, pat_dates, num_bands,
                                        params, abort_threshold, false);
            result_ptr[y * X + x] = res.distance;
        }
    }

    return result_array;
}

} // namespace twdtw
} // namespace zeit
