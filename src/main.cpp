#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "landtrendr.h"
#include "ccdc.h"
#include "utils.h"
#include "twdtw.h"
#include "som.h"
#include "phenology.h"
#include "mann_kendall.h"
#include "bfast_monitor.h"
#include "bfast_lite.h"
#include "bfast.h"
#include "snic.h"
#include "warp_python.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ backend for zeit";

    // LandTrendr sub-module
    py::module_ lt = m.def_submodule("landtrendr", "LandTrendr algorithms");

    py::class_<zeit::landtrendr::LandTrendrParams>(lt, "LandTrendrParams")
        .def(py::init<>())
        .def_readwrite("max_segments", &zeit::landtrendr::LandTrendrParams::max_segments)
        .def_readwrite("pval_threshold", &zeit::landtrendr::LandTrendrParams::pval_threshold)
        .def_readwrite("prevent_fast_recovery", &zeit::landtrendr::LandTrendrParams::prevent_fast_recovery)
        .def_readwrite("recovery_threshold", &zeit::landtrendr::LandTrendrParams::recovery_threshold)
        .def_readwrite("spike_threshold", &zeit::landtrendr::LandTrendrParams::spike_threshold)
        .def_readwrite("best_model_proportion", &zeit::landtrendr::LandTrendrParams::best_model_proportion)
        .def_readwrite("vertex_count_overshoot", &zeit::landtrendr::LandTrendrParams::vertex_count_overshoot)
        .def_readwrite("min_observations_needed", &zeit::landtrendr::LandTrendrParams::min_observations_needed)
        .def_readwrite("modifier", &zeit::landtrendr::LandTrendrParams::modifier);

    py::class_<zeit::landtrendr::Vertex>(lt, "Vertex")
        .def(py::init<int, double>())
        .def_readwrite("year", &zeit::landtrendr::Vertex::year)
        .def_readwrite("value", &zeit::landtrendr::Vertex::value);

    // Expose the fit_trajectory function to Python
    lt.def("fit_trajectory", &zeit::landtrendr::fit_trajectory, 
           "Run LandTrendr on a single pixel time series",
           py::arg("years"), py::arg("values"), py::arg("params"));

    // Expose the fit_trajectory_batch function to Python
    lt.def("fit_trajectory_batch", &zeit::landtrendr::fit_trajectory_batch, 
           "Run LandTrendr on a batch of pixels (3D array: [Y, X, Time]) with OpenMP",
           py::arg("values_array"), py::arg("years_array"), py::arg("params"), py::arg("no_data_value") = -9999.0, py::arg("n_jobs") = -1);

    // Expose desawtooth function for testing
    lt.def("desawtooth", &zeit::landtrendr::desawtooth,
           "Remove spikes from a time series",
           py::arg("vals"), py::arg("stopat") = 0.9);

    // CCDC sub-module
    py::module_ mc = m.def_submodule("ccdc", "CCDC algorithms");

    py::class_<zeit::ccdc::CCDCParams>(mc, "CCDCParams")
        .def(py::init<>())
        .def_readwrite("min_obs", &zeit::ccdc::CCDCParams::min_obs)
        .def_readwrite("conseq_anom", &zeit::ccdc::CCDCParams::conseq_anom)
        .def_readwrite("chi2_prob_threshold", &zeit::ccdc::CCDCParams::chi2_prob_threshold)
        .def_readwrite("tmax_cg_prob_threshold", &zeit::ccdc::CCDCParams::tmax_cg_prob_threshold)
        .def_readwrite("detection_bands", &zeit::ccdc::CCDCParams::detection_bands)
        .def_readwrite("num_c", &zeit::ccdc::CCDCParams::num_c)
        .def_readwrite("tmask_bands", &zeit::ccdc::CCDCParams::tmask_bands)
        .def_readwrite("thermal_band", &zeit::ccdc::CCDCParams::thermal_band)
        .def_readwrite("valid_min", &zeit::ccdc::CCDCParams::valid_min)
        .def_readwrite("valid_max", &zeit::ccdc::CCDCParams::valid_max)
        .def_readwrite("thermal_min", &zeit::ccdc::CCDCParams::thermal_min)
        .def_readwrite("thermal_max", &zeit::ccdc::CCDCParams::thermal_max);

    py::class_<zeit::ccdc::CCDCSegment>(mc, "CCDCSegment")
        .def(py::init<>())
        .def_readwrite("t_start", &zeit::ccdc::CCDCSegment::t_start)
        .def_readwrite("t_end", &zeit::ccdc::CCDCSegment::t_end)
        .def_readwrite("t_break", &zeit::ccdc::CCDCSegment::t_break)
        .def_readwrite("coefs", &zeit::ccdc::CCDCSegment::coefs)
        .def_readwrite("rmse", &zeit::ccdc::CCDCSegment::rmse)
        .def_readwrite("magnitude", &zeit::ccdc::CCDCSegment::magnitude)
        .def_readwrite("change_prob", &zeit::ccdc::CCDCSegment::change_prob)
        .def_readwrite("category", &zeit::ccdc::CCDCSegment::category)
        .def_readwrite("num_obs", &zeit::ccdc::CCDCSegment::num_obs);

    mc.def("fit_ccdc", &zeit::ccdc::fit_ccdc,
           "Run CCDC on a single pixel time series",
           py::arg("dates"), py::arg("values"), py::arg("qa"),
           py::arg("params") = zeit::ccdc::CCDCParams());

    // Expose the fit_ccdc_batch function to Python
    mc.def("fit_ccdc_batch", &zeit::ccdc::fit_ccdc_batch, "Run CCDC on a batch of pixels with OpenMP",
             py::arg("values_array"), py::arg("qa_array"), py::arg("dates_array"), 
             py::arg("params"), py::arg("max_segments") = 6, py::arg("return_coefs") = true, py::arg("n_jobs") = -1);

    // Utilities sub-module
    py::module_ utils = m.def_submodule("utils", "Geospatial utilities and processing");
    
    utils.def("compute_medoid", &zeit::utils::compute_medoid,
           "Computes the multidimensional medoid composite over the time axis",
           py::arg("input_array"), py::arg("no_data_value") = -9999.0);

    // TWDTW sub-module
    py::module_ tw = m.def_submodule("twdtw", "TWDTW algorithms");

    py::class_<zeit::twdtw::TWDTWParams>(tw, "TWDTWParams")
        .def(py::init<>())
        .def_readwrite("alpha", &zeit::twdtw::TWDTWParams::alpha)
        .def_readwrite("beta", &zeit::twdtw::TWDTWParams::beta)
        .def_readwrite("gamma", &zeit::twdtw::TWDTWParams::gamma)
        .def_readwrite("max_time_warp", &zeit::twdtw::TWDTWParams::max_time_warp)
        .def_readwrite("subsequence_matching", &zeit::twdtw::TWDTWParams::subsequence_matching);

    py::class_<zeit::twdtw::TWDTWResult>(tw, "TWDTWResult")
        .def(py::init<>())
        .def_readwrite("distance", &zeit::twdtw::TWDTWResult::distance)
        .def_readwrite("path", &zeit::twdtw::TWDTWResult::path);

    tw.def("fit_twdtw", &zeit::twdtw::fit_twdtw,
           "Run TWDTW on a single time series against a pattern",
           py::arg("ts_values"), py::arg("ts_dates"), 
           py::arg("pattern_values"), py::arg("pattern_dates"),
           py::arg("num_bands") = 1,
           py::arg("params") = zeit::twdtw::TWDTWParams(),
           py::arg("abort_threshold") = std::numeric_limits<double>::infinity(),
           py::arg("return_path") = false);

    tw.def("fit_twdtw_batch", &zeit::twdtw::fit_twdtw_batch,
           "Run TWDTW on a batch of pixels with OpenMP",
           py::arg("values_array"), py::arg("dates_array"),
           py::arg("pattern_values_array"), py::arg("pattern_dates_array"),
           py::arg("params"), 
           py::arg("abort_threshold") = std::numeric_limits<double>::infinity(),
           py::arg("n_jobs") = -1);

    // SOM submodule
    py::module_ som = m.def_submodule("som", "SOM C++ implementations");
    som.def("train_online", &zeit::som::train_online,
            "Online SOM training (port of MiniSom.train); updates weights in place",
            py::arg("weights").noconvert(), py::arg("data"), py::arg("order"),
            py::arg("num_iteration"), py::arg("use_epochs"),
            py::arg("learning_rate"), py::arg("sigma"),
            py::arg("lr_decay"), py::arg("sigma_decay"), py::arg("neighborhood"),
            py::arg("xx"), py::arg("yy"));
    som.def("train_batch", &zeit::som::train_batch,
            "Batch SOM training (port of MiniSom.train_batch_offline); updates weights in place",
            py::arg("weights").noconvert(), py::arg("data"), py::arg("num_iteration"),
            py::arg("learning_rate"), py::arg("sigma"),
            py::arg("lr_decay"), py::arg("sigma_decay"), py::arg("neighborhood"),
            py::arg("xx"), py::arg("yy"), py::arg("n_jobs") = -1);
    som.def("predict_bmus", &zeit::som::predict_bmus, "Find BMU for samples",
            py::arg("data"), py::arg("weights"), py::arg("n_jobs") = -1);

    // Phenology sub-module
    py::module_ ph = m.def_submodule("phenology", "Phenology extraction");
    
    py::enum_<phenology::CurveType>(ph, "CurveType")
        .value("BECK", phenology::CurveType::BECK)
        .value("ELMORE", phenology::CurveType::ELMORE)
        .value("GU", phenology::CurveType::GU)
        .value("KLOS", phenology::CurveType::KLOS)
        .value("ZHANG", phenology::CurveType::ZHANG)
        .value("AG", phenology::CurveType::AG)
        .value("DL", phenology::CurveType::DL)
        .export_values();

    py::enum_<phenology::ExtractionMethod>(ph, "ExtractionMethod")
        .value("THRESHOLD", phenology::ExtractionMethod::THRESHOLD)
        .value("DERIVATIVE", phenology::ExtractionMethod::DERIVATIVE)
        .value("GU", phenology::ExtractionMethod::GU)
        .value("KLOSTERMAN", phenology::ExtractionMethod::KLOSTERMAN)
        .export_values();

    ph.def("fit_phenology_batch", &phenology::fit_phenology_batch,
           "Run phenology extraction on a batch of pixels with OpenMP",
           py::arg("values_array"), py::arg("dates_array"),
           py::arg("curve_type"), 
           py::arg("extraction_method") = 0,
           py::arg("max_seasons") = 2,
           py::arg("whittaker_lambda") = 10.0,
           py::arg("apply_whittaker") = true,
           py::arg("apply_hants") = false,
           py::arg("hants_frequencies") = 3,
           py::arg("hants_threshold") = 0.1,
           py::arg("min_season_length") = 0,
           py::arg("min_amplitude") = 0.0,
           py::arg("min_pixel_amplitude") = 0.1,
           py::arg("rtrough_max") = 0.6,
           py::arg("r_min_filter") = 0.02,
           py::arg("n_jobs") = -1,
           py::arg("weights_array") = py::none(),
           py::arg("season_retry") = true);

    ph.def("debug_split_seasons", &phenology::debug_split_seasons,
           "Directly run the season-boundary detector (no smoothing/curve fitting) for unit testing",
           py::arg("y"), py::arg("dates") = py::none(), py::arg("min_season_length") = 0, py::arg("min_amplitude") = 0.0,
           py::arg("rtrough_max") = 0.6, py::arg("r_min_filter") = 0.02, py::arg("retry_on_empty") = true);

    // Mann-Kendall sub-module
    py::module_ mkmod = m.def_submodule("mannkendall", "Mann-Kendall trend test family + Sen's slope");

    py::enum_<zeit::mannkendall::MKMethod>(mkmod, "MKMethod")
        .value("ORIGINAL", zeit::mannkendall::MKMethod::ORIGINAL)
        .value("HAMED_RAO", zeit::mannkendall::MKMethod::HAMED_RAO)
        .value("YUE_WANG", zeit::mannkendall::MKMethod::YUE_WANG)
        .value("SEASONAL", zeit::mannkendall::MKMethod::SEASONAL)
        .export_values();

    mkmod.def("fit_mann_kendall_batch", &zeit::mannkendall::fit_mann_kendall_batch,
           "Pixel-wise Mann-Kendall trend test + Sen's slope on a batch of time series with OpenMP",
           py::arg("values_array"), py::arg("method") = 1, py::arg("alpha") = 0.05,
           py::arg("lag") = -1, py::arg("period") = 1, py::arg("min_valid") = 4, py::arg("n_jobs") = -1);

    mkmod.def("mk_test_single", &zeit::mannkendall::mk_test_single,
           "Run the Mann-Kendall test on a single time series (unit-testing helper)",
           py::arg("y"), py::arg("method") = 1, py::arg("alpha") = 0.05,
           py::arg("lag") = -1, py::arg("period") = 1);

    // bfastmonitor sub-module
    py::module_ bfm = m.def_submodule("bfastmonitor", "bfastmonitor: near-real-time structural change monitoring");

    py::class_<zeit::bfastmonitor::BFMResult>(bfm, "BFMResult")
        .def(py::init<>())
        .def_readwrite("breakpoint", &zeit::bfastmonitor::BFMResult::breakpoint)
        .def_readwrite("breakpoint_idx", &zeit::bfastmonitor::BFMResult::breakpoint_idx)
        .def_readwrite("magnitude", &zeit::bfastmonitor::BFMResult::magnitude)
        .def_readwrite("sigma", &zeit::bfastmonitor::BFMResult::sigma)
        .def_readwrite("n_history", &zeit::bfastmonitor::BFMResult::n_history)
        .def_readwrite("has_break", &zeit::bfastmonitor::BFMResult::has_break)
        .def_readwrite("valid", &zeit::bfastmonitor::BFMResult::valid);

    bfm.def("bfast_monitor", &zeit::bfastmonitor::bfast_monitor,
           "Run bfastmonitor on a single pixel time series (unit-testing helper)",
           py::arg("y"), py::arg("start_time"), py::arg("monitor_start_time"),
           py::arg("frequency"), py::arg("order") = 3, py::arg("h") = 0.25,
           py::arg("period") = 10, py::arg("alpha") = 0.05);

    bfm.def("fit_bfast_monitor_batch", &zeit::bfastmonitor::fit_bfast_monitor_batch,
           "Run bfastmonitor on a batch of pixels with OpenMP",
           py::arg("values_array"), py::arg("start_time"), py::arg("monitor_start_time"),
           py::arg("frequency"), py::arg("order") = 3, py::arg("h") = 0.25,
           py::arg("period") = 10, py::arg("alpha") = 0.05, py::arg("min_valid") = 10,
           py::arg("n_jobs") = -1);

    // bfastlite sub-module
    py::module_ bfl = m.def_submodule("bfastlite", "bfastlite: single-pass multiple-breakpoint detection");

    py::class_<zeit::bfastlite::BFLResult>(bfl, "BFLResult")
        .def(py::init<>())
        .def_readwrite("n_breaks", &zeit::bfastlite::BFLResult::n_breaks)
        .def_readwrite("rss", &zeit::bfastlite::BFLResult::rss)
        .def_readwrite("lwz", &zeit::bfastlite::BFLResult::lwz)
        .def_readwrite("n_valid", &zeit::bfastlite::BFLResult::n_valid)
        .def_readwrite("valid", &zeit::bfastlite::BFLResult::valid)
        .def_readwrite("breakpoint_idx", &zeit::bfastlite::BFLResult::breakpoint_idx);

    bfl.def("bfast_lite", &zeit::bfastlite::bfast_lite,
           "Run bfastlite on a single pixel time series (unit-testing helper)",
           py::arg("y"), py::arg("start_time"), py::arg("frequency"),
           py::arg("order") = 3, py::arg("h") = 0.15, py::arg("max_breaks_output") = 5);

    bfl.def("fit_bfast_lite_batch", &zeit::bfastlite::fit_bfast_lite_batch,
           "Run bfastlite on a batch of pixels with OpenMP",
           py::arg("values_array"), py::arg("start_time"), py::arg("frequency"),
           py::arg("order") = 3, py::arg("h") = 0.15, py::arg("max_breaks_output") = 5,
           py::arg("min_valid") = 20, py::arg("n_jobs") = -1);

    // bfast sub-module (the classic iterative trend+season break detection)
    py::module_ bf = m.def_submodule("bfast", "bfast: classic iterative trend+season break detection");

    py::class_<zeit::bfast::BFResult>(bf, "BFResult")
        .def(py::init<>())
        .def_readwrite("n_trend_breaks", &zeit::bfast::BFResult::n_trend_breaks)
        .def_readwrite("n_season_breaks", &zeit::bfast::BFResult::n_season_breaks)
        .def_readwrite("magnitude", &zeit::bfast::BFResult::magnitude)
        .def_readwrite("time", &zeit::bfast::BFResult::time)
        .def_readwrite("n_iter", &zeit::bfast::BFResult::n_iter)
        .def_readwrite("n_valid", &zeit::bfast::BFResult::n_valid)
        .def_readwrite("valid", &zeit::bfast::BFResult::valid)
        .def_readwrite("trend_breakpoint_idx", &zeit::bfast::BFResult::trend_breakpoint_idx)
        .def_readwrite("season_breakpoint_idx", &zeit::bfast::BFResult::season_breakpoint_idx);

    bf.def("bfast", &zeit::bfast::bfast,
           "Run bfast on a single pixel time series (unit-testing helper)",
           py::arg("y"), py::arg("start_time"), py::arg("frequency"),
           py::arg("order") = 3, py::arg("h") = 0.15,
           py::arg("max_breaks_trend") = 5, py::arg("max_breaks_season") = 5,
           py::arg("max_iter") = 10, py::arg("level") = 0.05);

    bf.def("fit_bfast_batch", &zeit::bfast::fit_bfast_batch,
           "Run bfast on a batch of pixels with OpenMP",
           py::arg("values_array"), py::arg("start_time"), py::arg("frequency"),
           py::arg("order") = 3, py::arg("h") = 0.15,
           py::arg("max_breaks_trend") = 5, py::arg("max_breaks_season") = 5,
           py::arg("max_iter") = 10, py::arg("level") = 0.05,
           py::arg("min_valid") = 20, py::arg("n_jobs") = -1);

    // SNIC sub-module (superpixel segmentation of images and image time series)
    py::module_ sn = m.def_submodule("snic", "SNIC: Simple Non-Iterative Clustering superpixels");

    sn.def("snic_segment", &zeit::snic::snic_segment,
           "Run SNIC on a planar [features, rows, cols] image from [n, 2] (row, col) seeds, tiles in parallel with OpenMP",
           py::arg("data"), py::arg("seeds"), py::arg("compactness") = 10.0,
           py::arg("tile_height") = 0, py::arg("tile_width") = 0, py::arg("n_jobs") = -1);

    // Warp sub-module (GDAL's resampling kernels, for load_raster(..., like=))
    py::module_ wp = m.def_submodule("warp", "GDAL's warp kernel on fixed source coordinates");
    zeit::warp::register_python(wp);
}
