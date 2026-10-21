//! zeit._rust.bfastlite: the bindings of zeit._core.bfastlite.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike2, PyArrayMethods};
use pyo3::prelude::*;
use zeit_core::bfast_lite as bfl;

use crate::arrays::c_order;

#[pyclass(name = "BFLResult", module = "zeit._rust.bfastlite", get_all, set_all)]
#[derive(Clone)]
pub struct BFLResult {
    n_breaks: f64,
    rss: f64,
    lwz: f64,
    n_valid: f64,
    valid: f64,
    breakpoint_idx: Vec<f64>,
    magnitude: Vec<f64>,
    first_after_idx: Vec<f64>,
}

#[pymethods]
impl BFLResult {
    #[new]
    fn new() -> Self {
        BFLResult::from(bfl::BflResult::default())
    }
}

impl From<bfl::BflResult> for BFLResult {
    fn from(r: bfl::BflResult) -> Self {
        BFLResult {
            n_breaks: r.n_breaks,
            rss: r.rss,
            lwz: r.lwz,
            n_valid: r.n_valid,
            valid: r.valid,
            breakpoint_idx: r.breakpoint_idx,
            magnitude: r.magnitude,
            first_after_idx: r.first_after_idx,
        }
    }
}

/// bfastlite on one series (unit-testing helper).
#[pyfunction]
#[pyo3(signature = (y, start_time, frequency, order = 3, h = 0.15, max_breaks_output = 5))]
fn bfast_lite(y: Vec<f64>, start_time: f64, frequency: i32, order: i32, h: f64, max_breaks_output: i32) -> BFLResult {
    BFLResult::from(bfl::bfast_lite(&y, start_time, frequency, order, h, max_breaks_output))
}

/// bfastlite on (pixels, time) values: (5 + 3 max_breaks_output, pixels).
#[pyfunction]
#[pyo3(signature = (values_array, start_time, frequency, order = 3, h = 0.15, max_breaks_output = 5, min_valid = 20,
                    n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn fit_bfast_lite_batch<'py>(
    py: Python<'py>,
    values_array: PyArrayLike2<'py, f64, AllowTypeChange>,
    start_time: f64,
    frequency: i32,
    order: i32,
    h: f64,
    max_breaks_output: i32,
    min_valid: i32,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = values_array.as_array();
    let (n_pixels, n_time) = view.dim();
    let values = c_order(&view);
    let values: &[f64] = &values;
    let out = py.detach(|| {
        bfl::batch(
            values,
            n_pixels,
            n_time,
            start_time,
            frequency,
            order,
            h,
            max_breaks_output,
            min_valid,
            n_jobs,
        )
    });
    let rows = bfl::n_metrics(max_breaks_output.max(0) as usize);
    PyArray1::from_vec(py, out).reshape([rows, n_pixels])
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "bfastlite")?;
    m.add("__doc__", "bfastlite: single-pass multiple-breakpoint detection")?;
    m.add_class::<BFLResult>()?;
    m.add_function(wrap_pyfunction!(bfast_lite, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_bfast_lite_batch, &m)?)?;
    Ok(m)
}
