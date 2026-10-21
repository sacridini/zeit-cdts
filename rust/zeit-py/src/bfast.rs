//! zeit._core.bfast: the Python module.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike2, PyArrayMethods};
use pyo3::prelude::*;
use zeit_core::bfast as bf;

use crate::arrays::c_order;

#[pyclass(name = "BFResult", module = "zeit._core.bfast", get_all, set_all)]
#[derive(Clone)]
pub struct BFResult {
    n_trend_breaks: f64,
    n_season_breaks: f64,
    magnitude: f64,
    time: f64,
    n_iter: f64,
    n_valid: f64,
    valid: f64,
    trend_breakpoint_idx: Vec<f64>,
    season_breakpoint_idx: Vec<f64>,
    trend_magnitude: Vec<f64>,
    trend_first_after_idx: Vec<f64>,
}

#[pymethods]
impl BFResult {
    #[new]
    fn new() -> Self {
        BFResult::from(bf::BfResult::default())
    }
}

impl From<bf::BfResult> for BFResult {
    fn from(r: bf::BfResult) -> Self {
        BFResult {
            n_trend_breaks: r.n_trend_breaks,
            n_season_breaks: r.n_season_breaks,
            magnitude: r.magnitude,
            time: r.time,
            n_iter: r.n_iter,
            n_valid: r.n_valid,
            valid: r.valid,
            trend_breakpoint_idx: r.trend_breakpoint_idx,
            season_breakpoint_idx: r.season_breakpoint_idx,
            trend_magnitude: r.trend_magnitude,
            trend_first_after_idx: r.trend_first_after_idx,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn params(
    start_time: f64,
    frequency: i32,
    h: f64,
    max_breaks_trend: i32,
    max_breaks_season: i32,
    max_iter: i32,
    level: f64,
    min_valid: i32,
) -> bf::Params {
    bf::Params {
        start_time,
        frequency,
        h,
        max_breaks_trend,
        max_breaks_season,
        max_iter,
        level,
        min_valid,
    }
}

/// bfast on one series (unit-testing helper).
#[pyfunction]
#[pyo3(signature = (y, start_time, frequency, order = 3, h = 0.15, max_breaks_trend = 5, max_breaks_season = 5,
                    max_iter = 10, level = 0.05))]
#[allow(clippy::too_many_arguments)]
fn bfast(
    y: Vec<f64>,
    start_time: f64,
    frequency: i32,
    order: i32,
    h: f64,
    max_breaks_trend: i32,
    max_breaks_season: i32,
    max_iter: i32,
    level: f64,
) -> BFResult {
    let p = params(
        start_time,
        frequency,
        h,
        max_breaks_trend,
        max_breaks_season,
        max_iter,
        level,
        20,
    );
    BFResult::from(bf::bfast(&y, order, &p))
}

/// bfast on (pixels, time) values: (7 + 3 max_breaks_trend + max_breaks_season, pixels).
#[pyfunction]
#[pyo3(signature = (values_array, start_time, frequency, order = 3, h = 0.15, max_breaks_trend = 5,
                    max_breaks_season = 5, max_iter = 10, level = 0.05, min_valid = 20, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn fit_bfast_batch<'py>(
    py: Python<'py>,
    values_array: PyArrayLike2<'py, f64, AllowTypeChange>,
    start_time: f64,
    frequency: i32,
    order: i32,
    h: f64,
    max_breaks_trend: i32,
    max_breaks_season: i32,
    max_iter: i32,
    level: f64,
    min_valid: i32,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = values_array.as_array();
    let (n_pixels, n_time) = view.dim();
    let values = c_order(&view);
    let values: &[f64] = &values;
    let p = params(
        start_time,
        frequency,
        h,
        max_breaks_trend,
        max_breaks_season,
        max_iter,
        level,
        min_valid,
    );
    let out = py.detach(|| bf::batch(values, n_pixels, n_time, order, &p, n_jobs));
    let rows = bf::n_metrics(max_breaks_trend.max(0) as usize, max_breaks_season.max(0) as usize);
    PyArray1::from_vec(py, out).reshape([rows, n_pixels])
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "bfast")?;
    m.add("__doc__", "bfast: classic iterative trend+season break detection")?;
    m.add_class::<BFResult>()?;
    m.add_function(wrap_pyfunction!(bfast, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_bfast_batch, &m)?)?;
    Ok(m)
}
