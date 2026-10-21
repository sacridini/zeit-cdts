//! zeit._rust.mannkendall: the bindings of zeit._core.mannkendall.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike2, PyArrayMethods};
use pyo3::prelude::*;
use zeit_core::mann_kendall::{self as mk, Method};

use crate::arrays::c_order;

/// The test of `method`, as zeit._core's MKMethod.
#[pyclass(name = "MKMethod", eq, eq_int, module = "zeit._rust.mannkendall")]
#[derive(Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms, non_camel_case_types)]
enum MKMethod {
    ORIGINAL = 0,
    HAMED_RAO = 1,
    YUE_WANG = 2,
    SEASONAL = 3,
}

/// Pixel-wise Mann-Kendall trend test + Sen's slope on a batch of time series:
/// values_array (pixels, time) -> (9, pixels): trend, h, p, z, tau, s, var_s,
/// slope, intercept; NaN where fewer than min_valid observations.
#[pyfunction]
#[pyo3(signature = (values_array, method = 1, alpha = 0.05, lag = -1, period = 1, min_valid = 4, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn fit_mann_kendall_batch<'py>(
    py: Python<'py>,
    values_array: PyArrayLike2<'py, f64, AllowTypeChange>,
    method: i32,
    alpha: f64,
    lag: i32,
    period: i32,
    min_valid: i32,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = values_array.as_array();
    let (n_pixels, n_time) = view.dim();
    let values = c_order(&view);
    let values: &[f64] = &values;
    let out = py.detach(|| {
        mk::batch(
            values,
            n_pixels,
            n_time,
            Method::from_code(method),
            alpha,
            lag,
            period,
            min_valid,
            n_jobs,
        )
    });
    PyArray1::from_vec(py, out).reshape([mk::N_METRICS, n_pixels])
}

/// The Mann-Kendall test on one series (unit-testing helper):
/// (trend, h, p, z, tau, s, var_s, slope, intercept).
#[pyfunction]
#[pyo3(signature = (y, method = 1, alpha = 0.05, lag = -1, period = 1))]
fn mk_test_single(
    y: Vec<f64>,
    method: i32,
    alpha: f64,
    lag: i32,
    period: i32,
) -> (i32, bool, f64, f64, f64, f64, f64, f64, f64) {
    let r = mk::mk_test(&y, Method::from_code(method), alpha, lag, period);
    (r.trend, r.h, r.p, r.z, r.tau, r.s, r.var_s, r.slope, r.intercept)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "mannkendall")?;
    m.add("__doc__", "Mann-Kendall trend test family + Sen's slope")?;
    m.add_class::<MKMethod>()?;
    // the values at module level too, as pybind11's export_values()
    for (name, value) in [
        ("ORIGINAL", MKMethod::ORIGINAL),
        ("HAMED_RAO", MKMethod::HAMED_RAO),
        ("YUE_WANG", MKMethod::YUE_WANG),
        ("SEASONAL", MKMethod::SEASONAL),
    ] {
        m.add(name, value)?;
    }
    m.add_function(wrap_pyfunction!(fit_mann_kendall_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(mk_test_single, &m)?)?;
    Ok(m)
}
