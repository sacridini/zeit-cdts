#include "utils.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace zeit {
namespace utils {

py::array_t<double> compute_medoid(
    py::array_t<double> input_array, // Expected shape: [Y, X, Time, Band]
    double no_data_value = -9999.0
) {
    auto buf = input_array.request();
    auto ptr = static_cast<double*>(buf.ptr);

    if (buf.ndim != 4) {
        throw std::runtime_error("Input array must have exactly 4 dimensions: [Y, X, Time, Band]");
    }

    int height = buf.shape[0];
    int width  = buf.shape[1];
    int times  = buf.shape[2];
    int bands  = buf.shape[3];

    // Output array [Y, X, Band]
    py::array_t<double> output_array({height, width, bands});
    auto out_ptr = static_cast<double*>(output_array.request().ptr);

    {
        pybind11::gil_scoped_release release;
        // OMP: Divide the Y-axis loop among processor threads
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
            
                size_t pixel_base_idx = (static_cast<size_t>(y) * width + x) * times * bands;
                size_t out_base_idx = (static_cast<size_t>(y) * width + x) * bands;

                Eigen::VectorXd median_vec(bands);
                bool has_valid_data = false;

                // 1. Calculate the median (band by band)
                for (int b = 0; b < bands; ++b) {
                    std::vector<double> valid_vals;
                    valid_vals.reserve(times);
                
                    for (int t = 0; t < times; ++t) {
                        double val = ptr[pixel_base_idx + (t * bands) + b];
                        if (val != no_data_value && !std::isnan(val)) {
                            valid_vals.push_back(val);
                        }
                    }

                    if (valid_vals.empty()) {
                        median_vec[b] = no_data_value;
                    } else {
                        has_valid_data = true;
                        size_t n = valid_vals.size() / 2;
                        std::nth_element(valid_vals.begin(), valid_vals.begin() + n, valid_vals.end());
                        if (valid_vals.size() % 2 == 0) {
                            auto max_it = std::max_element(valid_vals.begin(), valid_vals.begin() + n);
                            median_vec[b] = (*max_it + valid_vals[n]) / 2.0;
                        } else {
                            median_vec[b] = valid_vals[n]; 
                        }
                    }
                }

                // If the entire year is cloud/nodata, skip.
                if (!has_valid_data) {
                    for (int b = 0; b < bands; ++b) out_ptr[out_base_idx + b] = no_data_value;
                    continue;
                }

                // 2. Calculate Distance using EIGEN SIMD
                double min_dist = std::numeric_limits<double>::max();
                int best_t = -1;

                for (int t = 0; t < times; ++t) {
                    // Map memory directly to Eigen (Zero Copy)
                    Eigen::Map<Eigen::VectorXd> current_t_vec(&ptr[pixel_base_idx + (t * bands)], bands);
                
                    // If the slice has nodata, we ignore this time
                    bool has_nodata = false;
                    for(int b=0; b<bands; ++b) {
                        if (current_t_vec[b] == no_data_value || std::isnan(current_t_vec[b])) {
                            has_nodata = true;
                            break;
                        }
                    }
                
                    if (has_nodata) {
                        continue;
                    }

                    // SIMD Vectorization: Calculate the euclidean distance of all bands at once
                    double dist = (current_t_vec - median_vec).squaredNorm();

                    if (dist < min_dist) {
                        min_dist = dist;
                        best_t = t;
                    }
                }

                // 3. Save the best real observation
                for (int b = 0; b < bands; ++b) {
                    if (best_t != -1) {
                        out_ptr[out_base_idx + b] = ptr[pixel_base_idx + (best_t * bands) + b];
                    } else {
                        out_ptr[out_base_idx + b] = no_data_value;
                    }
                }
            }
        }
    }

    return output_array;
}

} // namespace utils
} // namespace zeit
