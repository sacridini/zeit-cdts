//! zeit._core.bfastmonitor: the Python module.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use zeit_core::bfast_monitor as bfm;

use crate::arrays::c_order;

#[pyclass(name = "BFMResult", module = "zeit._core.bfastmonitor", get_all, set_all)]
#[derive(Clone)]
pub struct BFMResult {
    breakpoint: f64,
    breakpoint_idx: f64,
    magnitude: f64,
    sigma: f64,
    n_history: f64,
    has_break: f64,
    valid: f64,
}

#[pymethods]
impl BFMResult {
    #[new]
    fn new() -> Self {
        BFMResult::from(bfm::BfmResult::default())
    }
}

impl From<bfm::BfmResult> for BFMResult {
    fn from(r: bfm::BfmResult) -> Self {
        BFMResult {
            breakpoint: r.breakpoint,
            breakpoint_idx: r.breakpoint_idx,
            magnitude: r.magnitude,
            sigma: r.sigma,
            n_history: r.n_history,
            has_break: r.has_break,
            valid: r.valid,
        }
    }
}

/// bfastmonitor on one series (unit-testing helper).
#[pyfunction]
#[pyo3(signature = (y, start_time, monitor_start_time, frequency, order = 3, h = 0.25, period = 10, alpha = 0.05))]
#[allow(clippy::too_many_arguments)]
fn bfast_monitor(
    y: Vec<f64>,
    start_time: f64,
    monitor_start_time: f64,
    frequency: i32,
    order: i32,
    h: f64,
    period: i32,
    alpha: f64,
) -> PyResult<BFMResult> {
    bfm::bfast_monitor(&y, start_time, monitor_start_time, frequency, order, h, period, alpha)
        .map(BFMResult::from)
        .map_err(PyRuntimeError::new_err)
}

/// bfastmonitor on (pixels, time) values: (7, pixels).
#[pyfunction]
#[pyo3(signature = (values_array, start_time, monitor_start_time, frequency, order = 3, h = 0.25, period = 10,
                    alpha = 0.05, min_valid = 10, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn fit_bfast_monitor_batch<'py>(
    py: Python<'py>,
    values_array: PyArrayLike2<'py, f64, AllowTypeChange>,
    start_time: f64,
    monitor_start_time: f64,
    frequency: i32,
    order: i32,
    h: f64,
    period: i32,
    alpha: f64,
    min_valid: i32,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = values_array.as_array();
    let (n_pixels, n_time) = view.dim();
    let values = c_order(&view);
    let values: &[f64] = &values;
    let out = py
        .detach(|| {
            bfm::batch(
                values,
                n_pixels,
                n_time,
                start_time,
                monitor_start_time,
                frequency,
                order,
                h,
                period,
                alpha,
                min_valid,
                n_jobs,
            )
        })
        .map_err(PyRuntimeError::new_err)?;
    PyArray1::from_vec(py, out).reshape([bfm::N_METRICS, n_pixels])
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "bfastmonitor")?;
    m.add("__doc__", "bfastmonitor: near-real-time structural change monitoring")?;
    m.add_class::<BFMResult>()?;
    m.add_function(wrap_pyfunction!(bfast_monitor, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_bfast_monitor_batch, &m)?)?;
    Ok(m)
}
