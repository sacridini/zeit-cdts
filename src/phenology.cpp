#include "phenology.h"
#include "phenology_curves.h"

#ifndef _OPENMP
#ifndef OMP_DUMMIES_DEFINED
#define OMP_DUMMIES_DEFINED
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_set_num_threads(x) (void)(x)
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <algorithm>
#include <optional>


namespace phenology {

PhenologyMetrics extract_metrics(const Eigen::VectorXd& params, CurveType type, const Eigen::VectorXd& t_segment) {
    PhenologyMetrics metrics = {
        std::nan(""), std::nan(""), // trs2
        std::nan(""), std::nan(""), // trs5
        std::nan(""), std::nan(""), // trs6
        std::nan(""), std::nan(""), std::nan(""), // der
        std::nan(""), std::nan(""), std::nan(""), std::nan(""), // gu
        std::nan(""), std::nan(""), std::nan(""), std::nan(""), // zhang
        std::nan(""), std::nan("") // los, pop
    };
    
    if (t_segment.size() == 0) return metrics;

    int n_pts = 365;
    double t_min = t_segment.minCoeff();
    double t_max = t_segment.maxCoeff();
    if (t_max - t_min < 1.0) t_max = t_min + 365.0;
    
    Eigen::VectorXd t_eval = Eigen::VectorXd::LinSpaced(n_pts, t_min, t_max);
    
    Eigen::VectorXd fvec = evaluate_curve(type, params, t_eval);
    
    double min_val = fvec.minCoeff();
    double max_val = fvec.maxCoeff();
    double amplitude = max_val - min_val;
    double threshold = min_val + 0.2 * amplitude;
    
    int pop_idx;
    fvec.maxCoeff(&pop_idx);
    metrics.pop = t_eval(pop_idx);
    
    Eigen::VectorXd d1(n_pts), d2(n_pts), d3(n_pts), k(n_pts), dk(n_pts);
    for(int i = 0; i < n_pts; ++i) {
        if(i == 0) d1[i] = (fvec[i+1] - fvec[i]) / (t_eval[i+1] - t_eval[i]);
        else if(i == n_pts - 1) d1[i] = (fvec[i] - fvec[i-1]) / (t_eval[i] - t_eval[i-1]);
        else d1[i] = (fvec[i+1] - fvec[i-1]) / (t_eval[i+1] - t_eval[i-1]);
    }
    
    for(int i = 0; i < n_pts; ++i) {
        if(i == 0) d2[i] = (d1[i+1] - d1[i]) / (t_eval[i+1] - t_eval[i]);
        else if(i == n_pts - 1) d2[i] = (d1[i] - d1[i-1]) / (t_eval[i] - t_eval[i-1]);
        else d2[i] = (d1[i+1] - d1[i-1]) / (t_eval[i+1] - t_eval[i-1]);
        k[i] = d2[i] / std::pow(1.0 + d1[i]*d1[i], 1.5);
    }
    
    for(int i = 0; i < n_pts; ++i) {
        if(i == 0) d3[i] = (d2[i+1] - d2[i]) / (t_eval[i+1] - t_eval[i]);
        else if(i == n_pts - 1) d3[i] = (d2[i] - d2[i-1]) / (t_eval[i] - t_eval[i-1]);
        else d3[i] = (d2[i+1] - d2[i-1]) / (t_eval[i+1] - t_eval[i-1]);
        
        if(i == 0) dk[i] = (k[i+1] - k[i]) / (t_eval[i+1] - t_eval[i]);
        else if(i == n_pts - 1) dk[i] = (k[i] - k[i-1]) / (t_eval[i] - t_eval[i-1]);
        else dk[i] = (k[i+1] - k[i-1]) / (t_eval[i+1] - t_eval[i-1]);
    }

    // 1. DERIVATIVE (DER)
    double max_d1 = -1e9; int der_sos_idx = 0;
    for(int i = 0; i <= pop_idx; ++i) { if(d1[i] > max_d1) { max_d1 = d1[i]; der_sos_idx = i; } }
    metrics.der_sos = t_eval(der_sos_idx);
    
    double min_d1 = 1e9; int der_eos_idx = n_pts - 1;
    for(int i = pop_idx; i < n_pts; ++i) { if(d1[i] < min_d1) { min_d1 = d1[i]; der_eos_idx = i; } }
    metrics.der_eos = t_eval(der_eos_idx);
    metrics.der_pos = metrics.pop;
    
    // LOS based on DER (standard Zeit default)
    metrics.los = metrics.der_eos - metrics.der_sos;

    // 2. THRESHOLDS (TRS2: 20%, TRS5: 50%, TRS6: 60%)
    double thr2 = min_val + 0.2 * amplitude;
    double thr5 = min_val + 0.5 * amplitude;
    double thr6 = min_val + 0.6 * amplitude;
    
    auto find_trs = [&](double thr, double& sos_out, double& eos_out) {
        for (int i = 0; i < pop_idx; ++i) {
            if (fvec[i] < thr && fvec[i+1] >= thr) {
                double w = (thr - fvec[i]) / (fvec[i+1] - fvec[i]);
                sos_out = t_eval[i] + w * (t_eval[i+1] - t_eval[i]);
                break;
            }
        }
        if (std::isnan(sos_out)) sos_out = t_eval[0];

        for (int i = pop_idx; i < n_pts - 1; ++i) {
            if (fvec[i] >= thr && fvec[i+1] < thr) {
                double w = (thr - fvec[i]) / (fvec[i+1] - fvec[i]);
                eos_out = t_eval[i] + w * (t_eval[i+1] - t_eval[i]);
                break;
            }
        }
        if (std::isnan(eos_out)) eos_out = t_eval[n_pts - 1];
    };
    
    find_trs(thr2, metrics.trs2_sos, metrics.trs2_eos);
    find_trs(thr5, metrics.trs5_sos, metrics.trs5_eos);
    find_trs(thr6, metrics.trs6_sos, metrics.trs6_eos);

    // 3. GU METHOD
    double max_d2_left = -1e9; int ud_idx = 0;
    double min_d2_left = 1e9; int sd_idx = 0;
    for(int i = 0; i <= der_sos_idx; ++i) { if(d2[i] > max_d2_left) { max_d2_left = d2[i]; ud_idx = i; } }
    for(int i = der_sos_idx; i <= pop_idx; ++i) { if(d2[i] < min_d2_left) { min_d2_left = d2[i]; sd_idx = i; } }
    metrics.gu_ud = t_eval(ud_idx);
    metrics.gu_sd = t_eval(sd_idx);
    
    double min_d2_right = 1e9; int dd_idx = n_pts - 1;
    double max_d2_right = -1e9; int rd_idx = n_pts - 1;
    for(int i = pop_idx; i <= der_eos_idx; ++i) { if(d2[i] < min_d2_right) { min_d2_right = d2[i]; dd_idx = i; } }
    for(int i = der_eos_idx; i < n_pts; ++i) { if(d2[i] > max_d2_right) { max_d2_right = d2[i]; rd_idx = i; } }
    metrics.gu_dd = t_eval(dd_idx);
    metrics.gu_rd = t_eval(rd_idx);

    // 4. ZHANG METHOD
    double max_k_left = -1e9; int greenup_idx = 0;
    double min_k_left = 1e9; int maturity_idx = 0;
    for(int i = 0; i <= der_sos_idx; ++i) { if(k[i] > max_k_left) { max_k_left = k[i]; greenup_idx = i; } }
    for(int i = der_sos_idx; i <= pop_idx; ++i) { if(k[i] < min_k_left) { min_k_left = k[i]; maturity_idx = i; } }
    metrics.zhang_greenup = t_eval(greenup_idx);
    metrics.zhang_maturity = t_eval(maturity_idx);
    
    double min_k_right = 1e9; int sen_idx = n_pts - 1;
    double max_k_right = -1e9; int dorm_idx = n_pts - 1;
    for(int i = pop_idx; i <= der_eos_idx; ++i) { if(k[i] < min_k_right) { min_k_right = k[i]; sen_idx = i; } }
    for(int i = der_eos_idx; i < n_pts; ++i) { if(k[i] > max_k_right) { max_k_right = k[i]; dorm_idx = i; } }
    metrics.zhang_senescence = t_eval(sen_idx);
    metrics.zhang_dormancy = t_eval(dorm_idx);

    if (std::isnan(metrics.los) || metrics.los <= 0.0) {
        metrics.los = std::nan("");
    }
    
    return metrics;
}

GoodnessOfFit compute_gof(const Eigen::VectorXd& y_seg, const Eigen::VectorXd& yfit_seg, const Eigen::VectorXd& w_seg) {
    double wsum = w_seg.sum();
    if (wsum < 1e-9 || y_seg.size() < 2) return {std::nan(""), std::nan("")};

    double ybar_w = (w_seg.array() * y_seg.array()).sum() / wsum;

    double ss_res = (w_seg.array() * (y_seg - yfit_seg).array().square()).sum();
    double ss_tot = (w_seg.array() * (y_seg.array() - ybar_w).square()).sum();

    double rmse = std::sqrt(ss_res / wsum);
    double r2 = (ss_tot > 1e-12) ? (1.0 - ss_res / ss_tot) : std::nan("");

    return {r2, rmse};
}

pybind11::array_t<double> fit_phenology_batch(
    pybind11::array_t<double> values_array,
    pybind11::array_t<double> dates_array,
    int curve_type_int,
    int extraction_method,
    int max_seasons,
    double whittaker_lambda,
    bool apply_whittaker,
    bool apply_hants,
    int hants_frequencies,
    double hants_threshold,
    int min_season_length,
    double min_amplitude,
    double min_pixel_amplitude,
    double rtrough_max,
    double r_min_filter,
    int n_jobs,
    pybind11::object weights_array,
    bool season_retry)
{
    auto vals_buf = values_array.request();
    auto dates_buf = dates_array.request();

    if (vals_buf.ndim != 2) throw std::runtime_error("values_array must be 2D [pixels, time]");
    if (dates_buf.ndim != 1) throw std::runtime_error("dates_array must be 1D [time]");

    int n_pixels = vals_buf.shape[0];
    int n_time = vals_buf.shape[1];

    if (dates_buf.shape[0] != n_time) throw std::runtime_error("dates_array length must match values_array time dimension");

    double* vals_ptr = static_cast<double*>(vals_buf.ptr);
    double* dates_ptr = static_cast<double*>(dates_buf.ptr);

    bool has_weights = !weights_array.is_none();
    pybind11::array_t<double> weights_np;
    double* weights_ptr = nullptr;
    if (has_weights) {
        weights_np = weights_array.cast<pybind11::array_t<double>>();
        auto w_buf = weights_np.request();
        if (w_buf.ndim != 2 || w_buf.shape[0] != n_pixels || w_buf.shape[1] != n_time) {
            throw std::runtime_error("weights_array must be 2D [pixels, time] with the same shape as values_array");
        }
        weights_ptr = static_cast<double*>(w_buf.ptr);
    }

    CurveType curve_type = static_cast<CurveType>(curve_type_int);

    const int n_metrics = 21; // 19 phenology metrics + R2 + RMSE
    pybind11::array_t<double> out_arr({n_metrics, n_pixels, max_seasons});
    double* out_ptr = static_cast<double*>(out_arr.request().ptr);

    for (int i = 0; i < n_metrics * n_pixels * max_seasons; ++i) {
        out_ptr[i] = std::nan("");
    }

    if (n_jobs <= 0) n_jobs = std::max(1, omp_get_max_threads() - 1);

    Eigen::Map<Eigen::VectorXd> t_all(dates_ptr, n_time);
    // Shared, read-only across pixels: lets split_growing_seasons() measure
    // season duration in real elapsed time instead of observation count.
    std::vector<double> dates_vec(dates_ptr, dates_ptr + n_time);

    {
        pybind11::gil_scoped_release release;
        #pragma omp parallel for num_threads(n_jobs)
        for (int p = 0; p < n_pixels; ++p) {
            std::vector<double> y_raw(n_time);
            std::optional<std::vector<double>> w_raw;
            if (has_weights) {
                std::vector<double> w_pixel(n_time);
                for (int t = 0; t < n_time; ++t) {
                    double w = weights_ptr[p * n_time + t];
                    w_pixel[t] = std::isnan(w) ? 1.0 : std::clamp(w, 0.0, 1.0);
                }
                w_raw = std::move(w_pixel);
            }

            bool valid_pixel = false;
            double pixel_min = 99999.0;
            double pixel_max = -99999.0;

            for (int t = 0; t < n_time; ++t) {
                double v = vals_ptr[p * n_time + t];
                y_raw[t] = v;
                if (!std::isnan(v) && v != -9999.0) {
                    valid_pixel = true;
                    if (v < pixel_min) pixel_min = v;
                    if (v > pixel_max) pixel_max = v;
                }
            }

            if (!valid_pixel) continue;

            // 1. Skip Logic: Early exit for low amplitude (water, urban, bare soil)
            if (pixel_max - pixel_min < min_pixel_amplitude) continue;

            std::vector<double> y_smooth = y_raw;
            if (apply_whittaker) {
                try {
                    // QC weights (when provided) feed straight into the Whittaker
                    // fit, same role as phenofit's `w` argument to check_input/
                    // smooth_wWHIT: low-quality observations pull the smoothed
                    // curve less instead of being treated as equally trustworthy.
                    y_smooth = eigen_whittaker(y_raw, w_raw, whittaker_lambda);
                } catch (...) {
                    continue;
                }
            } else if (apply_hants) {
                try {
                    std::vector<double> t_vec(t_all.data(), t_all.data() + t_all.size());
                    y_smooth = eigen_hants(y_raw, t_vec, hants_frequencies, hants_threshold, w_raw);
                } catch (...) {
                    continue;
                }
            }

            auto seasons = split_growing_seasons(y_smooth, dates_vec, min_season_length, min_amplitude, rtrough_max, r_min_filter, season_retry);
        
            int s_idx = 0;
            for (const auto& season : seasons) {
                if (s_idx >= max_seasons) break;
            
                int start = season.start_idx;
                int end = season.end_idx;
                int len = end - start + 1;
            
                if (len < 4) continue;
            
                Eigen::VectorXd t_seg(len);
                Eigen::VectorXd y_seg(len);
                for (int i = 0; i < len; ++i) {
                    t_seg(i) = t_all(start + i);
                    y_seg(i) = y_raw[start + i];
                }
            
                Eigen::VectorXd params;
                switch(curve_type) {
                    case CurveType::BECK: params = Eigen::VectorXd::Ones(6); break;
                    case CurveType::ELMORE: params = Eigen::VectorXd::Ones(7); break;
                    case CurveType::GU: params = Eigen::VectorXd::Ones(9); break;
                    case CurveType::KLOS: params = Eigen::VectorXd::Ones(13); break;
                    case CurveType::ZHANG: params = Eigen::VectorXd::Ones(7); break;
                    case CurveType::AG: params = Eigen::VectorXd::Ones(7); break;
                    case CurveType::DL: params = Eigen::VectorXd::Ones(6); break;
                }
            
                // Iterative fitting with wTSM weights, seeded from the external
                // QC weights (if any) instead of assuming every observation in
                // the season is equally reliable.
                Eigen::VectorXd w_seg = Eigen::VectorXd::Ones(len);
                if (w_raw.has_value()) {
                    for (int i = 0; i < len; ++i) {
                        w_seg(i) = (*w_raw)[start + i];
                    }
                }
                bool converged = false;
            
                // Phenofit default: 2 iterations
                int iters = 2;
                for (int iter = 1; iter <= iters; ++iter) {
                    converged = fit_curve(t_seg, y_seg, w_seg, params, curve_type);
                    if (converged && iter < iters) {
                        Eigen::VectorXd yfit = evaluate_curve(curve_type, params, t_seg);
                        // nptperyear for MODIS 16-day is 23. wfact default is 0.5
                        w_seg = eigen_wTSM(y_seg, yfit, w_seg, iter, 23, 0.5);
                    }
                }
            
                if (converged) {
                    PhenologyMetrics metrics = extract_metrics(params, curve_type, t_seg);
                    int base_idx = p * max_seasons + s_idx;
                    int stride = n_pixels * max_seasons;
                
                    out_ptr[0 * stride + base_idx] = metrics.trs2_sos;
                    out_ptr[1 * stride + base_idx] = metrics.trs2_eos;
                    out_ptr[2 * stride + base_idx] = metrics.trs5_sos;
                    out_ptr[3 * stride + base_idx] = metrics.trs5_eos;
                    out_ptr[4 * stride + base_idx] = metrics.trs6_sos;
                    out_ptr[5 * stride + base_idx] = metrics.trs6_eos;
                    out_ptr[6 * stride + base_idx] = metrics.der_sos;
                    out_ptr[7 * stride + base_idx] = metrics.der_pos;
                    out_ptr[8 * stride + base_idx] = metrics.der_eos;
                    out_ptr[9 * stride + base_idx] = metrics.gu_ud;
                    out_ptr[10 * stride + base_idx] = metrics.gu_sd;
                    out_ptr[11 * stride + base_idx] = metrics.gu_dd;
                    out_ptr[12 * stride + base_idx] = metrics.gu_rd;
                    out_ptr[13 * stride + base_idx] = metrics.zhang_greenup;
                    out_ptr[14 * stride + base_idx] = metrics.zhang_maturity;
                    out_ptr[15 * stride + base_idx] = metrics.zhang_senescence;
                    out_ptr[16 * stride + base_idx] = metrics.zhang_dormancy;
                    out_ptr[17 * stride + base_idx] = metrics.los;
                    out_ptr[18 * stride + base_idx] = metrics.pop;

                    Eigen::VectorXd yfit_seg = evaluate_curve(curve_type, params, t_seg);
                    GoodnessOfFit gof = compute_gof(y_seg, yfit_seg, w_seg);
                    out_ptr[19 * stride + base_idx] = gof.r2;
                    out_ptr[20 * stride + base_idx] = gof.rmse;

                    s_idx++;
                }
            }
        }
    
    }

    return out_arr;
}

std::vector<std::tuple<int, int, int>> debug_split_seasons(
    std::vector<double> y,
    std::optional<std::vector<double>> dates,
    int min_season_length,
    double min_amplitude,
    double rtrough_max,
    double r_min_filter,
    bool retry_on_empty)
{
    std::vector<double> dates_vec;
    if (dates.has_value()) {
        if (dates->size() != y.size()) {
            throw std::runtime_error("dates must have the same length as y");
        }
        dates_vec = *dates;
    } else {
        dates_vec.resize(y.size());
        for (size_t i = 0; i < y.size(); ++i) dates_vec[i] = static_cast<double>(i);
    }

    auto seasons = split_growing_seasons(y, dates_vec, min_season_length, min_amplitude, rtrough_max, r_min_filter, retry_on_empty);
    std::vector<std::tuple<int, int, int>> out;
    out.reserve(seasons.size());
    for (const auto& s : seasons) {
        out.emplace_back(s.start_idx, s.peak_idx, s.end_idx);
    }
    return out;
}

} // namespace phenology
