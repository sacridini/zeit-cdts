//! zeit._rust.phenology: the bindings of zeit._core.phenology.

use numpy::{AllowTypeChange, PyArray1, PyArray3, PyArrayLikeDyn, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::phenology as ph;

use crate::arrays::c_order;

/// The curves, as zeit._core's CurveType.
#[pyclass(name = "CurveType", eq, eq_int, module = "zeit._rust.phenology")]
#[derive(Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
enum CurveType {
    BECK = 0,
    ELMORE = 1,
    GU = 2,
    KLOS = 3,
    ZHANG = 4,
    AG = 5,
    DL = 6,
}

/// The extraction methods, as zeit._core's ExtractionMethod (every metric
/// family is computed either way).
#[pyclass(name = "ExtractionMethod", eq, eq_int, module = "zeit._rust.phenology")]
#[derive(Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
enum ExtractionMethod {
    THRESHOLD = 0,
    DERIVATIVE = 1,
    GU = 2,
    KLOSTERMAN = 3,
}

const CURVES: [(&str, CurveType); 7] = [
    ("BECK", CurveType::BECK),
    ("ELMORE", CurveType::ELMORE),
    ("GU", CurveType::GU),
    ("KLOS", CurveType::KLOS),
    ("ZHANG", CurveType::ZHANG),
    ("AG", CurveType::AG),
    ("DL", CurveType::DL),
];

const METHODS: [(&str, ExtractionMethod); 4] = [
    ("THRESHOLD", ExtractionMethod::THRESHOLD),
    ("DERIVATIVE", ExtractionMethod::DERIVATIVE),
    ("GU", ExtractionMethod::GU),
    ("KLOSTERMAN", ExtractionMethod::KLOSTERMAN),
];

// `__members__`, as pybind11's enums have: zeit.phenology looks the names up there.
#[pymethods]
impl CurveType {
    #[classattr]
    fn __members__(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let d = PyDict::new(py);
        for (name, value) in CURVES {
            d.set_item(name, value)?;
        }
        Ok(d)
    }
}

#[pymethods]
impl ExtractionMethod {
    #[classattr]
    fn __members__(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let d = PyDict::new(py);
        for (name, value) in METHODS {
            d.set_item(name, value)?;
        }
        Ok(d)
    }
}

/// An int argument, which pybind11 took as an int or as one of its enums.
fn as_int(v: &Bound<'_, PyAny>) -> PyResult<i32> {
    if let Ok(c) = v.cast::<CurveType>() {
        return Ok(*c.borrow() as i32);
    }
    if let Ok(m) = v.cast::<ExtractionMethod>() {
        return Ok(*m.borrow() as i32);
    }
    v.extract::<i32>()
}

/// Phenology of a batch of pixels: values_array (pixels, time), dates_array
/// (time,) -> (21, pixels, max_seasons): 19 metrics, R2, RMSE.
#[pyfunction]
#[pyo3(signature = (values_array, dates_array, curve_type, extraction_method = None, max_seasons = 2,
                    whittaker_lambda = 10.0, apply_whittaker = true, apply_hants = false, hants_frequencies = 3,
                    hants_threshold = 0.1, min_season_length = 0, min_amplitude = 0.0, min_pixel_amplitude = 0.1,
                    rtrough_max = 0.6, r_min_filter = 0.02, n_jobs = -1, weights_array = None,
                    season_retry = true))]
#[allow(clippy::too_many_arguments)]
fn fit_phenology_batch<'py>(
    py: Python<'py>,
    values_array: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    dates_array: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    curve_type: &Bound<'py, PyAny>,
    extraction_method: Option<&Bound<'py, PyAny>>,
    max_seasons: i32,
    whittaker_lambda: f64,
    apply_whittaker: bool,
    apply_hants: bool,
    hants_frequencies: i32,
    hants_threshold: f64,
    min_season_length: i32,
    min_amplitude: f64,
    min_pixel_amplitude: f64,
    rtrough_max: f64,
    r_min_filter: f64,
    n_jobs: i32,
    weights_array: Option<PyArrayLikeDyn<'py, f64, AllowTypeChange>>,
    season_retry: bool,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let curve_type = as_int(curve_type)?;
    if let Some(m) = extraction_method {
        as_int(m)?; // every metric family is computed either way
    }
    let (v, d) = (values_array.as_array(), dates_array.as_array());
    if v.ndim() != 2 {
        return Err(PyRuntimeError::new_err("values_array must be 2D [pixels, time]"));
    }
    if d.ndim() != 1 {
        return Err(PyRuntimeError::new_err("dates_array must be 1D [time]"));
    }
    let (n_pixels, n_time) = (v.shape()[0], v.shape()[1]);
    if d.len() != n_time {
        return Err(PyRuntimeError::new_err(
            "dates_array length must match values_array time dimension",
        ));
    }
    let w_arr = weights_array.as_ref().map(|w| w.as_array());
    if let Some(w) = &w_arr {
        if w.ndim() != 2 || w.shape()[0] != n_pixels || w.shape()[1] != n_time {
            return Err(PyRuntimeError::new_err(
                "weights_array must be 2D [pixels, time] with the same shape as values_array",
            ));
        }
    }
    let max_seasons =
        usize::try_from(max_seasons).map_err(|_| PyRuntimeError::new_err("negative dimensions are not allowed"))?;
    let settings = ph::Settings {
        curve_type: curve_type as i64,
        max_seasons,
        whittaker_lambda,
        apply_whittaker,
        apply_hants,
        hants_frequencies,
        hants_threshold,
        min_season_length,
        min_amplitude,
        min_pixel_amplitude,
        rtrough_max,
        r_min_filter,
        season_retry,
    };
    let (v, d) = (c_order(&v), c_order(&d));
    let w = w_arr.as_ref().map(c_order);
    let (v, d): (&[f64], &[f64]) = (&v, &d);
    let w: Option<&[f64]> = w.as_deref();
    let out = py.detach(|| ph::fit_phenology_batch(v, d, n_pixels, w, &settings, n_jobs));
    PyArray1::from_vec(py, out).reshape([ph::N_METRICS, n_pixels, max_seasons])
}

/// The season detector alone: (start, peak, end) indices of each season of `y`.
#[pyfunction]
#[pyo3(signature = (y, dates = None, min_season_length = 0, min_amplitude = 0.0, rtrough_max = 0.6,
                    r_min_filter = 0.02, retry_on_empty = true))]
fn debug_split_seasons(
    y: Vec<f64>,
    dates: Option<Vec<f64>>,
    min_season_length: i32,
    min_amplitude: f64,
    rtrough_max: f64,
    r_min_filter: f64,
    retry_on_empty: bool,
) -> PyResult<Vec<(i64, i64, i64)>> {
    let dates = match dates {
        Some(d) => {
            if d.len() != y.len() {
                return Err(PyRuntimeError::new_err("dates must have the same length as y"));
            }
            d
        }
        None => (0..y.len()).map(|i| i as f64).collect(),
    };
    let seasons = ph::split_growing_seasons(
        &y,
        &dates,
        min_season_length,
        min_amplitude,
        rtrough_max,
        r_min_filter,
        retry_on_empty,
    );
    Ok(seasons
        .iter()
        .map(|s| (s.start as i64, s.peak as i64, s.end as i64))
        .collect())
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "phenology")?;
    m.add("__doc__", "Phenology extraction")?;
    m.add_class::<CurveType>()?;
    m.add_class::<ExtractionMethod>()?;
    // the values at module level too, as pybind11's export_values(); GU is
    // ExtractionMethod.GU there, the last one exported
    for (name, value) in CURVES {
        m.add(name, value)?;
    }
    for (name, value) in METHODS {
        m.add(name, value)?;
    }
    m.add_function(wrap_pyfunction!(fit_phenology_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(debug_split_seasons, &m)?)?;
    Ok(m)
}
